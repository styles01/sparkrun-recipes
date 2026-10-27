"""On the board (ENG-225): the prefill kernels of tools/fn_prefill_kernels.py.

  * the grouped MoE (pairs sorted by expert, one GEMM over the experts' row groups) equals the torch
    reference with the same roundings within the bound below, at items of 32 and 64 rows, with experts
    of no row, one row and several items; it equals the decode path's expert kernel (one CTA a pair)
    within the same bound, and its combined output through `combine_inject_norm` too;
  * experts no pair selects are never read (their trellis overwritten, svh NaN: output unchanged);
  * every K the packs use;
  * mm_rows (fp16 dequantised mixer weights on the library GEMM) and mix_tail equal the decode
    path's gemv and mix_up within fp16 rounding of the weights;
  * the QSA block scores of a chunk on the library GEMM equal the decode path's score kernel within
    fp32 summation order, -inf past each row's complete blocks, and select the same blocks.

Run on the box: python tests/gpu/test_fn_prefill_kernels_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.models.flashnext.pack import Int8Rows  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as K  # noqa: E402
from tools import fn_prefill_kernels as PK  # noqa: E402

H, I = 2560, 640
DEV = "cuda"
# the decode kernels and the grouped GEMM make the same bf16 roundings; fp32 summation order differs
# (8 contiguous warp slices against 4 quarters of 16-row mma tiles), which can flip a bf16 rounding of
# an intermediate: max |diff| / max |ref| below two bf16 steps, as tests/gpu/test_fn_kernels_gpu.py
MOE_REL = 1e-2


def _x(M, K_, seed, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(M, K_, generator=g, device=DEV) * scale).to(torch.bfloat16)


def _bank(n_exp: int, kbs, shared_k: int, seed: int = 0):
    blocks = []
    for e in range(n_exp + 1):
        kb = shared_k if e == n_exp else kbs[e % len(kbs)]
        gt = X.random_block(I, H, kb, seed=seed + 3 * e)
        up = X.random_block(I, H, kb, seed=seed + 3 * e + 1)
        dn = X.random_block(H, I, kb, seed=seed + 3 * e + 2)
        blocks.append((gt, up, dn))
    trip = [tuple((b.t, b.suh[0].contiguous(), b.svh) for b in eb) for eb in blocks]
    return K.MoEBank(trip, DEV), blocks


def _ids(M, n_exp, k, seed, skew=False):
    """Router-like ids [M, k + 1] (the shared expert last); `skew`: expert 0 in every row (several
    items), expert 1 in one row only, the last two experts in none."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for m in range(M):
        if skew:
            r = [0] + (torch.randperm(n_exp - 4, generator=g)[:k - 1] + 2).tolist()
            if m == 3:
                r[-1] = 1
        else:
            r = torch.randperm(n_exp, generator=g)[:k].tolist()
        rows.append(r)
    return torch.tensor([r + [n_exp] for r in rows], dtype=torch.int32, device=DEV)


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def test_group_plan():
    ids = _ids(37, 24, 10, 5, skew=True)
    g = PK.group_plan(ids.reshape(-1), 25, 32)
    flat = ids.reshape(-1).cpu()
    o, cnt, iend = g["order"].cpu(), g["cnt"], g["iend"]
    assert sorted(o.tolist()) == list(range(flat.numel()))
    assert torch.equal(g["eid"].cpu(), flat[o].int()) and bool((g["eid"][1:] >= g["eid"][:-1]).all())
    for e in range(25):                                    # stable inside an expert
        idx = o[flat[o] == e]
        assert bool((idx[1:] > idx[:-1]).all()), e
    assert int(cnt[0]) == 37 and int(cnt[1]) == 1 and int(cnt[23]) == 0 and int(cnt[24]) == 37
    assert int(iend[-1]) == sum((int(c) + 31) // 32 for c in cnt) <= g["items"]
    return f"{flat.numel()} pairs, experts of 37 / 1 / 0 rows, {int(iend[-1])} items (bound {g['items']})"


def test_moe_grouped_reference_and_decode_kernel():
    bank, _ = _bank(24, (3,), 5)
    x = _x(37, H, 3, 0.5)
    ids = _ids(37, 24, 10, 9, skew=True)
    out = []
    ref = PK.moe_grouped_reference(x, ids, bank)
    dec = K.moe_experts(x, ids, bank)
    for rows in (32, 64):
        got = PK.moe_grouped(x, ids, bank, rows=rows)
        r1, r2 = _rel(got, ref), _rel(got, dec)
        assert r1 < MOE_REL and r2 < MOE_REL, (rows, r1, r2)
        out.append(f"{rows}-row items: vs reference {r1:.1e}, vs decode kernel {r2:.1e}")
    g = torch.Generator(device=DEV).manual_seed(63)
    h0 = torch.randn(37, 4 * H, generator=g, device=DEV).to(torch.bfloat16)
    d = torch.randn(37, 324, generator=g, device=DEV) * 3
    wn = (torch.randn(4 * H, generator=g, device=DEV) * 0.1).half()
    w = torch.rand(37, 11, device=DEV).to(torch.bfloat16).float()
    hg, ng = K.combine_inject_norm(h0, PK.moe_grouped(x, ids, bank), ids, w, d, 320, wn, 4, 1e-6)
    hd, nd = K.combine_inject_norm(h0, dec, ids, w, d, 320, wn, 4, 1e-6)
    r3 = _rel(hg - h0, hd - h0)
    assert r3 < MOE_REL, r3
    return "; ".join(out) + f"; combined + injected vs decode path {r3:.1e}"


def test_moe_grouped_reads_only_selected_experts():
    bank, blocks = _bank(32, (3,), 5, seed=7)
    x = _x(9, H, 4, 0.5)
    ids = _ids(9, 32, 10, 11)
    want = PK.moe_grouped(x, ids, bank)
    used = set(ids.flatten().tolist())
    n = 0
    for e in range(32):
        if e not in used:
            n += 1
            for b in blocks[e]:
                b.t.copy_(torch.randint(-32768, 32767, b.t.shape, dtype=torch.int32, device=DEV).to(torch.int16))
                b.svh.fill_(float("nan"))
    got = PK.moe_grouped(x, ids, bank)
    assert n > 0 and torch.equal(got, want)
    return f"{n} of 32 experts unselected and overwritten (trellis random, svh NaN): output unchanged"


def test_moe_grouped_every_k():
    out = []
    for kb in (2, 3, 4, 5, 6, 8):
        bank, _ = _bank(12, (kb,), 6 if kb != 6 else 5, seed=100 * kb)
        x = _x(20, H, kb, 0.5)
        ids = _ids(20, 12, 8, kb)
        r = _rel(PK.moe_grouped(x, ids, bank), K.moe_experts(x, ids, bank))
        assert r < MOE_REL, (kb, r)
        out.append(f"K{kb} {r:.0e}")
    return ", ".join(out)


def test_mm_rows_and_mix_tail():
    g = torch.Generator(device=DEV).manual_seed(5)
    out = []
    for N, Kk, kind in ((324, 10240, "int8"), (513, 2560, "fp16"), (96, 2560, "fp16")):
        w16 = (torch.randn(N, Kk, generator=g, device=DEV) * 0.02).half()
        w = Int8Rows.of(w16) if kind == "int8" else w16
        x = _x(300, Kk, N)
        r = _rel(PK.mm_rows(x, w), K.gemv(x, w))
        assert r < 2e-3, (N, Kk, kind, r)
        out.append(f"{N}x{Kk} {kind} {r:.0e}")
    up = Int8Rows.of((torch.randn(4 * H, 320, generator=g, device=DEV) * 0.05).half())
    d = torch.randn(300, 324, generator=g, device=DEV) * 2
    n = _x(300, 4 * H, 8)
    a = torch.nn.functional.silu(d[:, :320].to(torch.bfloat16).float() / 4).to(torch.bfloat16)
    got = PK.mix_tail(PK.mm_rows(a, up), n, 4)
    want = K.mix_up(d, n, up, 4)
    r = _rel(got, want)
    assert r < 2e-2, r
    return "; ".join(out) + f"; mixer up + stream mean {r:.0e} (bf16 steps of the gate)"


def test_qsa_scores_rows():
    from tools import fn_seq_kernels as SK
    g = torch.Generator(device=DEV).manual_seed(9)
    M, NB = 300, 2048
    iq = (torch.randn(M, 4, 128, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    pool = (torch.randn(NB + 10, 128, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    nblk = torch.randint(513, NB + 1, (M,), generator=g, device=DEV)
    got = PK.qsa_scores_rows(iq, pool, nblk, NB, block=128)
    qctl = torch.zeros(M, 4, dtype=torch.int32, device=DEV)
    qctl[:, 0] = nblk
    qctl[:, 1] = nblk
    want = SK.qsa_scores(iq, pool, torch.zeros(M, 1, 128, dtype=torch.bfloat16, device=DEV), qctl, NB)
    fin = torch.isfinite(want)
    assert torch.equal(fin, torch.isfinite(got))
    r = _rel(got[fin], want[fin])
    assert r < 1e-5, r
    tail = (4 * nblk[:, None] + torch.arange(3, device=DEV)[None, :]).to(torch.int32)
    sa, sb = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV), torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)
    SK.qsa_select(got, tail, sa)
    SK.qsa_select(want, tail, sb)
    same = int((sa[:, :2048] == sb[:, :2048]).all(1).sum())
    assert same >= M - 3, same
    short = PK.qsa_scores_rows(iq, pool[:NB - 300], nblk.clamp(max=NB - 310), NB, block=128)
    assert short.shape == (M, NB) and bool(torch.isinf(short[:, NB - 310:]).all())
    return (f"vs the score kernel {r:.0e}; -inf past the complete blocks; the same 512 blocks in {same} of {M} rows;"
            " a pool shorter than the score width")


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:<46} ok   {fn() or ''}", flush=True)
                passed += 1
            except AssertionError as e:
                print(f"  {name:<46} FAIL {e}", flush=True)
                failed += 1
    print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
