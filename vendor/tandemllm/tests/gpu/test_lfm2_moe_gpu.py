"""On the board (ENG-287): the shared EXL3 kernels at LFM2.5-24B-A2B's shapes, and its engine on a tiny model.

  * tools/fn_kernels.dense, the projections up to 32 rows, at every shape the model multiplies by
    (in_proj 2048 -> 6144, q|k|v 2048 -> 3072, out_proj and the conv's out 2048 -> 2048, the dense
    MLP's w1|w3 2048 -> 23552 and w2 11776 -> 2048, a 16,384-column slice of the head in fp32): the
    torch reference within the bound below, and every row of an M-row call (M = 2..32) the M = 1
    call's row bit for bit;
  * the MoE bank kernels with LFM's experts (hidden 2048, expert 1536, 4 slots a row, no shared
    expert): the one-row output against the fixed-order torch reference, and row invariance with rows
    sharing experts;
  * ENG-285: the half-integer rates (1.5, 2.5, 3.5 bits) decoded as exllamav3 decodes them, through
    the dense, MoE and prefill kernels, and 7-bit experts;
  * ENG-288: the decode step is one CUDA graph replay a token, its logits the eager step's;
  * the tiny-model engine tests of tests/test_lfm2_moe_engine.py and the row kernels of
    tests/test_lfm2_moe_kernels.py, here on the GPU.

Run on the box: python tests/gpu/test_lfm2_moe_gpu.py   (under flock ~/.qwen38-box.flock)
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as FK  # noqa: E402

H, I = 2048, 1536
# the reference makes the kernels' roundings (rotated input to fp16, output to bf16) with fp32 sums in
# another order: the output within a few bf16 steps of the largest value
BOUND = 2 ** -6


def _rel(a, b) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


def test_dense_at_lfm_shapes():
    out = []
    for N, Kin, kb, fp32 in ((6144, 2048, 3, False), (3072, 2048, 4, False), (2048, 2048, 2, False),
                             (23552, 2048, 4, False), (2048, 11776, 3, False), (16384, 2048, 6, True)):
        w = X.random_block(N, Kin, kb, seed=N + Kin)
        w.out_fp32 = fp32
        x = (torch.randn(32, Kin, device="cuda") * 0.5).to(torch.bfloat16)
        y = FK.dense(x, w)
        rel = _rel(y[:1], X.reference_matmul(x[:1], w))
        assert rel < BOUND, (N, Kin, kb, rel)
        for M in (2, 5, 16, 32):
            assert torch.equal(FK.dense(x[:M], w)[M - 1], FK.dense(x[M - 1:M], w)[0]), (N, Kin, M)
        out.append(f"{N}x{Kin}@K{kb} {rel:.1e}")
    return ", ".join(out)


def test_moe_bank_at_lfm_shapes():
    E = 8
    experts = [tuple(X.random_block(n, k, kb, seed=100 * e + j) for j, (n, k) in enumerate(((I, H), (I, H), (H, I))))
               for e, kb in zip(range(E), (2, 3, 4, 3, 2, 4, 3, 4))]
    bank = FK.MoEBank([tuple((b.t, b.suh[0], b.svh) for b in ex) for ex in experts], "cuda")
    x = (torch.randn(16, H, device="cuda") * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(1)
    ids = torch.stack([torch.randperm(E, generator=g)[:4] for _ in range(16)]).to(torch.int32).cuda()
    wts = torch.rand(16, 4, generator=g).cuda()
    wts = (wts / wts.sum(1, keepdim=True)).to(torch.bfloat16).float()
    yd = FK.moe_experts(x[:1], ids[:1], bank, order=torch.empty(0, dtype=torch.int32, device="cuda"))
    one = FK.moe_combine(yd, ids[:1], wts[:1])
    ref = FK.moe_reference(x[:1], ids[:1], wts[:1], experts)
    rel = _rel(one, ref)
    assert rel < BOUND, rel
    full = FK.moe_experts(x, ids, bank)
    for m in range(16):
        alone = FK.moe_experts(x[m:m + 1], ids[m:m + 1], bank, order=torch.empty(0, dtype=torch.int32, device="cuda"))
        assert torch.equal(full[4 * m:4 * m + 4], alone), m
    return f"one row vs reference {rel:.1e}; 16 rows each its one-row bits"


def test_half_rates_decode_as_exllamav3():
    """ENG-285: 1.5, 2.5 and 3.5 bits a weight (tiles 16 K + 8 int16 wide): the device decode
    (exl3_recon) equals the torch reference decode bit for bit, and exllamav3's own LinearEXL3 where
    its package imports (tools/lfm_check.py xcheck runs it on a real pack)."""
    try:
        from exllamav3.modules.quant.exl3 import LinearEXL3
    except Exception:                                              # noqa: BLE001
        LinearEXL3 = None
    out = []
    for kb in (15, 25, 35, 2, 3, 4):
        w = X.random_block(512, 256, kb, seed=kb)
        dev = w.inner()
        ref = X.decode_trellis(w.t.cpu()).to(dev.device)
        assert torch.equal(dev, ref), kb
        if LinearEXL3 is not None:
            mul1 = torch.tensor([X.MUL1 - (1 << 32)], dtype=torch.int32, device="cuda")
            lin = LinearEXL3(None, w.K, w.N, suh=w.suh[0].contiguous(), svh=w.svh.contiguous(),
                             trellis=w.t.contiguous(), mul1=mul1)
            assert torch.equal(dev, lin.get_inner_weight_tensor().to(torch.float16)), kb
        out.append(f"{X.rate(kb):g}")
    return f"rates {', '.join(out)}: recon = reference" + (" = LinearEXL3" if LinearEXL3 else " (no exllamav3 here)")


def test_dense_and_experts_at_half_rates():
    out = []
    for N, Kin, kb in ((6144, 2048, 25), (3072, 2048, 35), (2048, 2048, 15), (2048, 11776, 25)):
        w = X.random_block(N, Kin, kb, seed=N + Kin + kb)
        x = (torch.randn(32, Kin, device="cuda") * 0.5).to(torch.bfloat16)
        rel = _rel(FK.dense(x[:1], w), X.reference_matmul(x[:1], w))
        assert rel < BOUND, (N, Kin, kb, rel)
        for M in (2, 5, 16, 32):
            assert torch.equal(FK.dense(x[:M], w)[M - 1], FK.dense(x[M - 1:M], w)[0]), (N, Kin, kb, M)
        # rows past the row kernels' 32 take the slab (the skinny GEMM decodes whole rates)
        x64 = (torch.randn(64, Kin, device="cuda") * 0.5).to(torch.bfloat16)
        assert _rel(X.exl3_matmul(x64, w), X.reference_matmul(x64, w)) < BOUND, (N, Kin, kb)
        out.append(f"{N}x{Kin}@{X.rate(kb):g} {rel:.1e}")
    E = 8
    experts = [tuple(X.random_block(n, k, kb, seed=100 * e + j) for j, (n, k) in enumerate(((I, H), (I, H), (H, I))))
               for e, kb in zip(range(E), (25, 3, 35, 15, 2, 4, 25, 7))]
    bank = FK.MoEBank([tuple((b.t, b.suh[0], b.svh) for b in ex) for ex in experts], "cuda")
    assert bank.ks[-1] == 7, bank.ks                          # the widest tile sizes the stages
    x = (torch.randn(16, H, device="cuda") * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(2)
    ids = torch.stack([torch.randperm(E, generator=g)[:4] for _ in range(16)]).to(torch.int32).cuda()
    wts = torch.rand(16, 4, generator=g).cuda()
    wts = (wts / wts.sum(1, keepdim=True)).to(torch.bfloat16).float()
    none = torch.empty(0, dtype=torch.int32, device="cuda")
    rel = _rel(FK.moe_combine(FK.moe_experts(x[:1], ids[:1], bank, order=none), ids[:1], wts[:1]),
               FK.moe_reference(x[:1], ids[:1], wts[:1], experts))
    assert rel < BOUND, rel
    full = FK.moe_experts(x, ids, bank)
    for m in range(16):
        assert torch.equal(full[4 * m:4 * m + 4], FK.moe_experts(x[m:m + 1], ids[m:m + 1], bank, order=none)), m
    from tools import fn_prefill_kernels as PK
    xp = (torch.randn(64, H, device="cuda") * 0.5).to(torch.bfloat16)
    idp = torch.stack([torch.randperm(E, generator=g)[:4] for _ in range(64)]).to(torch.int32).cuda()
    wp = torch.full((64, 4), 0.25, device="cuda")
    prel = _rel(FK.moe_combine(PK.moe_grouped(xp, idp, bank), idp, wp), FK.moe_reference(xp, idp, wp, experts))
    assert prel < BOUND, prel
    out.append(f"experts at 1.5..3.5 and 7: one row {rel:.1e}, 16 rows each its one-row bits, prefill {prel:.1e}")
    return "; ".join(out)


def test_tiny_engine_and_kernels_on_the_gpu():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_lfm2_moe_engine as E
    import test_lfm2_moe_kernels as Kt
    assert E.DEV == "cuda" and Kt.DEV == "cuda"
    n = 0
    for mod in (Kt, E):
        for name, fn in sorted(vars(mod).items()):
            if name.startswith("test_") and callable(fn):
                r = fn()
                assert not (r and "skipped" in r), name
                n += 1
    return f"{n} tests"


def test_the_decode_step_is_one_graph():
    """ENG-288 "the decode step is one graph": on a warm DecodeGraph each decoded token is one replay of
    one captured CUDA graph (no capture after the first inside an attention class, no other launch path),
    and its logits are the eager step's bit for bit."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_lfm2_moe_engine as E

    from engine.models.lfm2_moe.engine import DecodeGraph
    cfg, eng, _ = E.tiny()
    a, b = eng.new_state(128), eng.new_state(128)
    eng.prefill(E.PROMPT, a)
    eng.prefill(E.PROMPT, b)
    toks = [11, 50, 3, 7, 19]
    want = E._decode(eng, b, toks)
    dg = DecodeGraph(eng, a)
    dg(toks[0])                                   # warm: the class's capture
    assert dg.captures == 1 and len(dg.graphs) == 1
    replays = []
    orig = torch.cuda.CUDAGraph.replay
    torch.cuda.CUDAGraph.replay = lambda g: (replays.append(g), orig(g))[1]
    try:
        got = [dg(t)[0].clone() for t in toks[1:]]
    finally:
        torch.cuda.CUDAGraph.replay = orig
    torch.cuda.synchronize()
    assert len(replays) == len(toks) - 1 and len({id(g) for g in replays}) == 1, len(replays)
    assert dg.captures == 1, dg.captures
    assert torch.equal(torch.stack(got), want[1:]), "graph logits differ from the eager step's"
    return f"{len(replays)} tokens, {len(replays)} replays of 1 graph, logits = eager"


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
