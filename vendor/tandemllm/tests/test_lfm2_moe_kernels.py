"""ENG-287: LFM2 MoE's row kernels against their torch twins, and row invariance (a row of an M-row call
is the one-row call's row bit for bit). They run on the GPU, or on the CPU under TRITON_INTERPRET=1;
without Triton (a Mac) the tests are skipped and say so."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.lfm2_moe import kernels as K  # noqa: E402

SKIP = None if (K.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))) else "skipped: Triton on a GPU, or TRITON_INTERPRET=1 (the CPU interpreter)"
DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
G = torch.Generator(device="cpu").manual_seed(0)


def same(a, b) -> bool:
    """Bit for bit on the GPU; under the interpreter, whose float -> bf16 cast truncates, within two bf16
    steps (the kernels' own row invariance is checked bit for bit either way)."""
    if DEV == "cuda":
        return torch.equal(a, b)
    return bool((a.float() - b.float()).abs().max() <= 2 ** -5 * b.float().abs().max())


def rnd(*shape, s=1.0, dtype=torch.bfloat16):
    return (torch.randn(*shape, generator=G) * s).to(dtype).to(DEV)


def test_add_rms():
    if SKIP:
        return SKIP
    h, d, w = rnd(5, 256), rnd(5, 256), rnd(256, s=0.3) + 1
    hn, nn = K.add_rms(h, d, w, 1e-5)
    rh, rn = K.add_rms_reference(h, d, w, 1e-5)
    assert same(hn, rh)
    assert (nn.float() - rn.float()).abs().max() <= 2 ** -6 * rn.float().abs().max()
    for r in range(5):
        assert torch.equal(K.add_rms(h[r:r + 1], d[r:r + 1], w, 1e-5)[1], nn[r:r + 1])


def test_conv_rows_tree_and_decode():
    if SKIP:
        return SKIP
    H, KT = 256, 3
    parents = [-1, 0, 1, 0, 3]
    bcx, state, wc = rnd(5, 3 * H), rnd(KT - 1, H), rnd(H, KT, s=0.5)
    src = torch.tensor(K.window_src(parents, KT), dtype=torch.int32, device=DEV)
    assert K.window_src([-1], 3) == [[-1, -2]] and K.window_src([-1, 0, 1], 3)[2] == [0, 1]
    rec = torch.empty(5, H, dtype=torch.bfloat16, device=DEV)
    y = K.conv_rows(bcx, state, wc, src, rec)
    ry, rbx = K.conv_rows_reference(bcx, state, wc, src)
    assert same(rec, rbx)
    assert (y.float() - ry.float()).abs().max() <= 2 ** -6 * ry.float().abs().max()
    # row 4 alone, its path decoded one token at a time from the state
    st = state.clone()
    one = torch.empty(1, H, dtype=torch.bfloat16, device=DEV)
    s1 = torch.tensor(K.window_src([-1], KT), dtype=torch.int32, device=DEV)
    for r in (0, 3, 4):
        y1 = K.conv_rows(bcx[r:r + 1], st, wc, s1, one, write=True)
    assert torch.equal(y1[0], y[4])
    assert torch.equal(st[-1], rec[4]) and torch.equal(st[-2], rec[3])


def test_qk_prep_and_attention_rows():
    if SKIP:
        return SKIP
    HQ, HK, D, cap = 8, 2, 64, 700
    pos = torch.tensor([600, 601, 602, 601], device=DEV)            # a tree: rows 1 and 3 children of 0
    slot = torch.tensor([600, 601, 602, 603], device=DEV)
    qkv = rnd(4, (HQ + 2 * HK) * D)
    kc, vc = rnd(HK, cap, D), rnd(HK, cap, D)
    cos, sin = rnd(cap, D), rnd(cap, D)
    qw, kw = rnd(D, s=0.2) + 1, rnd(D, s=0.2) + 1
    q = K.qk_prep(qkv, pos, slot, qw, kw, cos, sin, kc, vc, HQ, HK, 1e-5)
    rq, rk, rv = K.qk_prep_reference(qkv, pos, qw, kw, cos, sin, HQ, HK, D, 1e-5)
    # the per-head norm's rsqrt and sum order differ from torch's: within one bf16 step, a quarter of
    # the values one step apart on the GPU (2026-10-14); the row invariance below is bit for bit
    for a, b in ((q, rq), (kc[:, 600:604].transpose(0, 1), rk)):
        assert float((a.float() - b.float()).abs().max()) <= (2 ** -6 if DEV == "cuda" else 2 ** -4) * float(b.float().abs().max())
    assert torch.equal(vc[:, 600:604].transpose(0, 1), rv)
    anc = torch.zeros(4, K.ANC, dtype=torch.int32, device=DEV)
    for i, pa in enumerate([[0], [0, 1], [0, 1, 2], [0, 3]]):
        anc[i, :len(pa)] = 600 + torch.tensor(pa, dtype=torch.int32)
    ctl = torch.tensor([[600 + len(pa), 600] for pa in ([0], [0, 1], [0, 1, 2], [0, 3])], dtype=torch.int32, device=DEV)
    ns = K.class_of(604, 4096)
    o = K.attn_rows(q, kc, vc, ctl, anc, ns)
    ro = K.attn_rows_reference(q, kc, vc, ctl, anc)
    assert (o.float() - ro).abs().max() < 2e-2
    # a row alone, and in a larger class (more programs): the same bits
    q1 = K.qk_prep(qkv[3:4], pos[3:4], slot[3:4], qw, kw, cos, sin, kc, vc, HQ, HK, 1e-5)
    assert torch.equal(q1[0], q[3])
    assert torch.equal(K.attn_rows(q[3:4], kc, vc, ctl[3:4], anc[3:4], ns)[0], o[3])
    assert torch.equal(K.attn_rows(q, kc, vc, ctl, anc, ns * 2), o)
    # ENG-363: partials sized for 8 rows serve 4 rows and then 1 (a call reads only what it wrote);
    # more rows than they hold is refused, never a buffer swapped under a captured graph
    bufs = {"rows": 8}
    assert torch.equal(K.attn_rows(q, kc, vc, ctl, anc, ns, bufs=bufs), o)
    assert bufs["pacc"].shape[0] == 8 * HK * ns * 16
    assert torch.equal(K.attn_rows(q[3:4], kc, vc, ctl[3:4], anc[3:4], ns, bufs=bufs)[0], o[3])
    small = {"rows": 2}
    K.attn_rows(q[:2], kc, vc, ctl[:2], anc[:2], ns, bufs=small)
    try:
        K.attn_rows(q, kc, vc, ctl, anc, ns, bufs=small)
        raise AssertionError("4 rows in partials sized for 2")
    except ValueError:
        pass


def test_router():
    if SKIP:
        return SKIP
    lg = (torch.randn(6, 16, generator=G) * 2).to(DEV)
    lg[2, 3] = lg[2, 7]                                    # a tie: the lower id first
    bias = (torch.randn(16, generator=G) * 0.1).to(DEV)
    bias[3] = bias[7] = 0.0
    ids, wts = K.router(lg, bias, 4, 1.0)
    rids, rwts = K.router_reference(lg, bias, 4, 1.0)
    assert torch.equal(ids.sort(1).values, rids.sort(1).values)  # the choice is exact either way
    got = {(r, int(i)): float(w) for r in range(6) for i, w in zip(ids[r], wts[r])}
    want = {(r, int(i)): float(w) for r in range(6) for i, w in zip(rids[r], rwts[r])}
    assert all(abs(got[k] - want[k]) <= 2 ** -7 for k in want)


def test_linear_rows_invariant():
    if SKIP:
        return SKIP
    x, w = rnd(20, 384), rnd(272, 384, s=0.1)
    y = K.linear_rows(x, w)
    assert (y.float() - (x.float() @ w.float().T)).abs().max() < 0.05
    for r in (0, 7, 19):
        assert torch.equal(K.linear_rows(x[r:r + 1], w)[0], y[r])


def test_moe_bf16():
    if SKIP:
        return SKIP
    E, H, I = 4, 256, 128
    x, w13, w2 = rnd(3, H), rnd(E, 2 * I, H, s=0.1), rnd(E, H, I, s=0.1)
    ids = torch.tensor([[0, 2], [3, 0], [1, 1]], dtype=torch.int32, device=DEV)
    yd = K.moe_bf16(x, ids, w13, w2)
    ref = K.moe_bf16_reference(x, ids, w13, w2)
    assert (yd - ref).abs().max() <= (1e-2 if DEV == "cuda" else 3e-2) * ref.abs().max()
    assert torch.equal(K.moe_bf16(x[1:2], ids[1:2], w13, w2), yd[2:4])


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
