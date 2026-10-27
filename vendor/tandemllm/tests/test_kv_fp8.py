"""CPU tests for the e4m3 KV cache and the decode-attention launch rule.

The kernel itself runs on the board (`tools/attn_kernels.py::check`). What can be checked here is
everything around it that would be wrong quietly: the quantiser's error bound, a cache that puts
codes and scales at the positions the kernel will read, a snapshot that carries the scales with the
codes, the float32 reference the kernel is checked against, and the chunk rule that keeps a decode
step and a verify block on the same arithmetic.
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache as C  # noqa: E402
from engine.model import KVCache  # noqa: E402
from tools.attn_kernels import pick_launch, quantize_kv, reference  # noqa: E402

CFG = SimpleNamespace(attention_layers=[3, 7], num_key_value_heads=4, head_dim=256)


def test_quantize_kv_error_is_inside_e4m3_half_step():
    torch.manual_seed(0)
    x = (torch.randn(1, 4, 50, 256) * 3).to(torch.bfloat16)
    codes, s = quantize_kv(x)
    assert codes.dtype == torch.float8_e4m3fn and s.shape == (1, 4, 50)
    back = codes.float() * s[..., None]
    # e4m3 carries 3 mantissa bits: a round-to-nearest error of at most 2^-4 of the value, and
    # the subnormal floor below 2^-6 of the scale's range
    err = (back - x.float()).abs()
    bound = x.float().abs() * 2 ** -4 + s[..., None] * 2 ** -9
    assert bool((err <= bound + 1e-6).all()), float((err - bound).max())


def test_fp8_cache_writes_codes_and_scales_where_the_kernel_reads():
    kv = KVCache(CFG, 64, "cpu", fp8=True)
    k = torch.randn(1, 4, 5, 256).to(torch.bfloat16)
    v = torch.randn(1, 4, 5, 256).to(torch.bfloat16)
    kk, vv = kv.append(7, k, v, 10)
    assert kk.dtype == torch.float8_e4m3fn and kk.shape == (1, 4, 15, 256)
    ks, vs = kv.scales(7, 15)
    assert ks.shape == (1, 4, 15) and bool((ks[..., :10] == 0).all()) and bool((ks[..., 10:] > 0).all())
    dk, dv = kv.dequant(7, 15)
    assert (dk[..., 10:, :].float() - k.float()).abs().max() < 0.07 * k.float().abs().max()
    assert (dv[..., 10:, :].float() - v.float()).abs().max() < 0.07 * v.float().abs().max()
    # the other layer is untouched
    assert bool((kv.ks[0] == 0).all())


def test_bf16_cache_is_what_it_was():
    kv = KVCache(CFG, 64, "cpu")
    assert kv.k.dtype == torch.bfloat16 and kv.ks is None and kv.scales(3, 4) == (None, None)
    k = torch.randn(1, 4, 3, 256).to(torch.bfloat16)
    kk, _ = kv.append(3, k, k, 0)
    assert torch.equal(kk, k)


def test_a_snapshot_carries_the_scales():
    kv = KVCache(CFG, 64, "cpu", fp8=True)
    x = torch.randn(1, 4, 6, 256).to(torch.bfloat16)
    kv.append(3, x, x, 0)
    kv.append(7, x, x, 0)
    kv.length = 6
    state = SimpleNamespace(S=torch.zeros(2), conv=torch.zeros(2), primed=True, nbytes=16)
    eng = SimpleNamespace(kv=kv, state=state, cfg=SimpleNamespace(attention_layers=[3, 7],
                          num_key_value_heads=4, head_dim=256), _trace=None, tree=None,
                          trace=None, settle=lambda: None)
    from engine.models.qwen3_5 import adapter as A      # the 27B's adapter verbs on the stand-in
    for f in (A.cache_state, A.cost, A.resume_at, A.discard_pending):
        setattr(eng, f.__name__, types.MethodType(f, eng))
    snap = C.capture(eng)
    ks = snap.views[2]                                  # k, v, then the e4m3 scales
    assert ks.shape == (2, 1, 4, 6)
    saved = kv.ks[..., :6].clone()
    kv.ks.zero_()
    kv.k.zero_()
    C.restore(eng, snap)
    assert torch.equal(kv.ks[..., :6], saved)
    assert snap.nbytes > C.StateSnapshot(6, snap.views[:2], snap.state).nbytes


def test_reference_is_sdpa_on_the_engine_mask():
    torch.manual_seed(1)
    T, start = 5, 20
    q = torch.randn(1, 24, T, 256)
    k = torch.randn(1, 4, start + T, 256)
    v = torch.randn(1, 4, start + T, 256)
    bm = torch.ones(T, T, dtype=torch.bool).tril()
    bm[3:, 1:3] = False
    vis = torch.zeros(T, start + T, dtype=torch.bool)
    vis[:, :start] = True
    vis[:, start:] = bm
    want = F.scaled_dot_product_attention(q, k.repeat_interleave(6, 1), v.repeat_interleave(6, 1),
                                          attn_mask=vis)
    assert (reference(q, k, v, start, bm) - want).abs().max() < 1e-4


def test_the_chunk_is_a_function_of_the_context_and_changes_only_at_powers_of_two():
    # a decode step at context p and a verify block reaching p + 16 cut the context identically
    for p in (100, 4000, 20000, 30000, 70000, 200000):
        _, _, _, c1 = pick_launch(1, 6, p)
        _, _, _, c17 = pick_launch(17, 6, p + 16)
        assert c1 == c17, (p, c1, c17)
    assert pick_launch(1, 6, 300) == (32, 1, 1, 512)          # one tile height for every T
    bm, groups, ns, chunk = pick_launch(17, 6, 32768)
    assert (bm, groups) == (32, 4) and chunk == 512 and ns == 64
    assert pick_launch(1, 6, 131072)[3] == 2048


def test_the_quality_gate_tail_is_verify_shaped_and_covers_every_token():
    from tools.prefill_quality import segments
    seg = segments(8192, 2048, 256, 16)
    assert seg[0] == (0, 2048) and seg[3] == (6144, 7936)
    assert all(b - a == 16 for a, b in seg[4:]) and seg[-1] == (8176, 8192)
    assert [a for a, _ in seg[1:]] == [b for _, b in seg[:-1]]          # contiguous, no gaps
    assert segments(100, 2048, 0, 16) == [(0, 100)]
    assert segments(10, 2048, 256, 16) == [(0, 10)]


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
