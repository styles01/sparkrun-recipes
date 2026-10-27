"""The chunked gated delta rule at PREFILL, as two kernels instead of two thousand.

`engine/gdn.py::chunk_gated_delta_rule` is the reference implementation's shape, in PyTorch, and it
is written the way the reference is written: every intermediate is a whole tensor. At an 8k prefill,
one GDN layer of this model is H = 48 value heads, Dk = Dv = 128 and 128 chunks of 64, so those
intermediates are

    query, key, value, k_beta, v_beta, k_cumdecay   [1, 48, 128, 64, 128] fp32   201 MB each
    decay_mask, attn, (I - attn)^-1                 [1, 48, 128, 64,  64] fp32   101 MB each

and the pass writes and re-reads about six gigabytes of them per layer, then runs a serial loop of
128 iterations issuing a dozen kernels each. The arithmetic underneath is 71 GFLOP a layer, which
is 2.3 ms on this board. The mixer measures 110-128 ms. Two earlier attempts aimed at the
arithmetic -- a blocking sweep and a precision sweep -- were both negative, and that is why: the
arithmetic was never the cost. **The temporaries and the launches are.**

So this file keeps the same arithmetic in the same order and never writes most of those tensors
down. Two kernels:

`_intra` -- grid (head, chunk), 6,144 independent programs at 8k. Everything inside one chunk that
does not depend on the recurrent state: the l2 normalisation of the keys, the chunk-local
cumulative gate, the decay matrix, the UT transform's strictly-lower `attn`, its inverse, and the
two products the scan needs. `decay_mask` and `attn` are born, used and die in registers; only
`u_beta = (I - attn)^-1 v_beta` and `k_cumdecay` are written out.

**The inverse is a forward substitution, and the cheaper idea was wrong.** `attn` is strictly lower
triangular and therefore nilpotent with `attn^64 = 0`, so `(I - attn)^-1 = SUM_{p<64} attn^p` and six
doublings of `S_2n = S_n + attn^n . S_n` reach it in eleven 64x64 products instead of a batched
triangular solve over 6,144 systems. It is four times cheaper than what ships here and it produces
**NaN on this model**: the series is only stable while the powers of `attn` stay small, and on real
keys they do not. It survives a synthetic check because independent random keys in 128 dimensions
are very nearly orthogonal, so the matrix being inverted is very nearly the identity and every
scheme passes. `check(corr=...)` mixes a common direction into the keys and kills it in one run;
`QWEN38_GDN_PREFILL_SERIES=1` still selects it, so the failure can be reproduced.

What ships instead is `engine/gdn.py`'s own forward substitution -- the loop behind
`QWEN38_UT_INVERSE=0` -- one row at a time in registers, where nothing ever exceeds the size of the
answer.

`_scan` -- grid (head, value block). The only serial part, and the state tile never leaves
registers: for each chunk in order it reads q, k, `u_beta`, `k_cumdecay` and the gate, produces the
chunk's output, and advances `S`. The 128 iterations become one kernel launch per program.

NUMERICS. The tensors and the recurrent state are fp32, in the reference's order. The PRODUCTS are
`bf16x3`: each fp32 operand is split into a bf16 head and a bf16 tail and three of the four cross
terms are kept, which is about sixteen mantissa bits on the tensor cores. That is not a preference,
it is what the board left: Triton's `ieee` runs these shapes on the CUDA cores at fourteen times the
tf32 time and its `tf32x3` at thirty, while plain `tf32` moves the recurrent state by 2.6e-3
relative. The two products that never touch the state -- `q . k^T` and its product with the chunk's
pseudo-values -- stay at `tf32`, because they only ever reach a bf16 output.

The l2 normalisation of q and k is left in PyTorch and in the INPUT dtype, which is where
`engine/gdn.py` does it: redoing it in fp32 inside the kernel would be more accurate and therefore
a different answer, carried to the end of the sequence by the state.

**And on "lossless": at a prefill there is nothing to be lossless against.** `tools/prefill_gate.py
--control` runs the SHIPPED path against ITSELF with its other exact inverse -- `solve_triangular`
against the forward substitution, both already in `engine/gdn.py`, both exact -- and on this model
that moves an 8k prefill's logits by 8.2 bf16 ulp and makes a greedy continuation diverge after ten
tokens. The fused path moves them by 8.7 and diverges after the same ten. Forty-eight recurrent
layers amplify any reordering of the same fp32 to that, and the control is the only honest
yardstick for a kernel that is, by construction, a reordering.
"""

from __future__ import annotations

import math
import os
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in __import__("sys").path:
    # run as a script from tools/: the repo root, appended (lowest priority), for engine.settings
    __import__("sys").path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


# The chunk this file is built for. 64 is the reference's and the one the blocking sweep chose.
CHUNK = 64
# Doublings: SUM_{p<2^STEPS} attn^p. 2^STEPS must be at least the chunk, or the series is truncated.
STEPS = int(math.log2(CHUNK))


if HAVE_TRITON:

    @triton.jit
    def _mm(a, b, PREC: tl.constexpr):
        """One product at the requested precision.

        `tf32`, `tf32x3` and `ieee` are Triton's own. On this board the second and third are not
        usable -- `ieee` runs the 64x128x64 products on the CUDA cores at 14x the tf32 time and
        `tf32x3` is worse still -- and plain `tf32` leaves ten mantissa
        bits, which moves the recurrent state by 2.6e-3 relative and the prefill's logits by nine
        bf16 ulps.

        `bf16x3` is the way out and it is an old one: split each fp32 operand into a bf16 head and
        a bf16 tail, `a = ah + al` exactly, and keep the three cross terms whose magnitudes matter.

            a . b  =  ah.bh + ah.bl + al.bh  +  al.bl
                      ~~~~~~~~~~~~~~~~~~~~~     ~~~~~ dropped, O(2^-16) relative

        Three bf16 products on the tensor cores, each accumulating in fp32, for about sixteen
        mantissa bits -- thirty-two times tf32's accuracy at three times tf32's cost, where the
        alternatives were forty times the cost. `bf16x4` keeps the fourth term as well.
        """
        if PREC == "bf16x3" or PREC == "bf16x4":
            ah = a.to(tl.bfloat16)
            al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            bh = b.to(tl.bfloat16)
            bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(ah, bh) + tl.dot(ah, bl) + tl.dot(al, bh)
            if PREC == "bf16x4":
                acc = acc + tl.dot(al, bl)
            return acc
        return tl.dot(a, b, input_precision=PREC)

    @triton.jit
    def _intra(K, V, G, BETA, UB, KCD,
               T, s_kt, s_kh, s_vt, s_vh, s_gt, s_gh,
               C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
               STEPS: tl.constexpr, PREC: tl.constexpr, SERIES: tl.constexpr,
               REP: tl.constexpr = 1):
        """One (head, chunk): everything that does not read the recurrent state."""
        h = tl.program_id(0)
        c = tl.program_id(1)
        i = tl.arange(0, C)
        rows = c * C + i
        m = rows < T
        ok = tl.arange(0, DK)
        ov = tl.arange(0, DV)

        k = tl.load(K + rows[:, None] * s_kt + (h // REP) * s_kh + ok[None, :],
                    mask=m[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + rows[:, None] * s_vt + h * s_vh + ov[None, :],
                    mask=m[:, None], other=0.0).to(tl.float32)
        g = tl.load(G + rows * s_gt + h * s_gh, mask=m, other=0.0).to(tl.float32)
        beta = tl.load(BETA + rows * s_gt + h * s_gh, mask=m, other=0.0).to(tl.float32)

        # k arrives already l2-normalised, in the input dtype: the reference normalises in bf16
        # and rounds there, and a kernel that redid it in fp32 would be MORE accurate and therefore
        # a different answer, propagated through a state that sums over the whole sequence.
        # The gate compares against the reference, so the reference's rounding is kept.
        # The gate is cumulative WITHIN the chunk; a padded row carries g = 0, which is what
        # `F.pad` puts there and what keeps the chunk's last gate equal to the last real row's.
        gc = tl.cumsum(g, axis=0)
        kb = k * beta[:, None]
        vb = v * beta[:, None]

        # decay[i, j] = exp(gc[i] - gc[j]) on and below the diagonal, 0 above it. Masked BEFORE the
        # exponential, because gc[i] - gc[j] above the diagonal is unbounded above.
        lower = i[:, None] >= i[None, :]
        decay = tl.where(lower, tl.exp(tl.where(lower, gc[:, None] - gc[None, :], 0.0)), 0.0)
        A = -_mm(kb, tl.trans(k), PREC) * decay
        A = tl.where(i[:, None] > i[None, :], A, 0.0)          # strictly lower, as the UT transform

        if SERIES:
            # The doubling. Exact in exact arithmetic, four times cheaper than the substitution --
            # and NOT SAFE on real keys, which is why it is not the default. See the file header.
            inv = tl.where(i[:, None] == i[None, :], 1.0, 0.0)     # S_1 = I
            P = A                                                  # attn^1
            for _ in tl.static_range(STEPS):
                inv = inv + _mm(P, inv, PREC)
                P = _mm(P, P, PREC)
        else:
            # Forward substitution, one row at a time: `engine/gdn.py`'s own loop, the one behind
            # `QWEN38_UT_INVERSE=0`, ported into registers. Row r of the inverse is row r of A plus
            # that row times the rows above it, and nothing in it ever exceeds the size of the
            # answer.
            for r in tl.static_range(1, C):
                here = i == r
                row = tl.sum(tl.where(here[:, None], A, 0.0), axis=0)
                row = row + tl.sum(row[:, None] * A, axis=0) * (i < r)
                A = tl.where(here[:, None], row[None, :], A)
            inv = A + tl.where(i[:, None] == i[None, :], 1.0, 0.0)

        ub = _mm(inv, vb, PREC)
        kcd = _mm(inv, kb * tl.exp(gc)[:, None], PREC)
        tl.store(UB + h * T * DV + rows[:, None] * DV + ov[None, :], ub, mask=m[:, None])
        tl.store(KCD + h * T * DK + rows[:, None] * DK + ok[None, :], kcd, mask=m[:, None])

    @triton.jit
    def _scan(Q, K, G, UB, KCD, OUT, S,
              T, NC, s_kt, s_kh, s_gt, s_gh, s_ot, s_oh,
              C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
              SCALE: tl.constexpr, PREC: tl.constexpr, PREC_A: tl.constexpr,
              REP: tl.constexpr = 1):
        """One (head, value block): the chunks in order, with the state tile in registers.
        q and k are read at key head h // REP (REP > 1: the caller did not widen them)."""
        h = tl.program_id(0)
        hq = h // REP
        vbk = tl.program_id(1)
        i = tl.arange(0, C)
        ok = tl.arange(0, DK)
        ov = vbk * BV + tl.arange(0, BV)
        sp = S + h * DK * DV + ok[:, None] * DV + ov[None, :]
        s = tl.load(sp)

        for c in range(NC):
            rows = c * C + i
            m = rows < T
            q = tl.load(Q + rows[:, None] * s_kt + hq * s_kh + ok[None, :],
                        mask=m[:, None], other=0.0).to(tl.float32)
            k = tl.load(K + rows[:, None] * s_kt + hq * s_kh + ok[None, :],
                        mask=m[:, None], other=0.0).to(tl.float32)
            g = tl.load(G + rows * s_gt + h * s_gh, mask=m, other=0.0).to(tl.float32)
            q = q * SCALE          # already l2-normalised by the caller, as in _intra
            gc = tl.cumsum(g, axis=0)
            u = tl.load(UB + h * T * DV + rows[:, None] * DV + ov[None, :],
                        mask=m[:, None], other=0.0)
            kcd = tl.load(KCD + h * T * DK + rows[:, None] * DK + ok[None, :],
                          mask=m[:, None], other=0.0)

            lower = i[:, None] >= i[None, :]
            decay = tl.where(lower, tl.exp(tl.where(lower, gc[:, None] - gc[None, :], 0.0)), 0.0)
            a = _mm(q, tl.trans(k), PREC_A) * decay
            v_prime = _mm(kcd, s, PREC)
            v_new = u - v_prime
            inter = _mm(q * tl.exp(gc)[:, None], s, PREC)
            o = inter + _mm(a, v_new, PREC_A)
            tl.store(OUT + rows[:, None] * s_ot + h * s_oh + ov[None, :],
                     o.to(OUT.dtype.element_ty), mask=m[:, None])

            g_last = tl.sum(tl.where(i == C - 1, gc, 0.0))
            s = s * tl.exp(g_last) + _mm(
                tl.trans(k * tl.exp(g_last - gc)[:, None]), v_new, PREC)
        tl.store(sp, s)


# The value-column block. 32 measured best at 8k: 128 spills, and 64 spills once the
# products are split three ways. Four programs a head, 192 on 48 SMs.
BV = int(_S.get("GDN_PREFILL_BV"))
# Warps per program. Both kernels hold several 64x64 and 64x128 fp32 tiles at once -- the scan
# holds the [128, BV] state tile for the whole sequence on top of them -- so four warps spill and
# eight do not. Tunable because the answer is a measurement, not an argument.
WARPS_INTRA = int(_S.get("GDN_PREFILL_WARPS_INTRA"))
WARPS_SCAN = int(_S.get("GDN_PREFILL_WARPS_SCAN"))
# The scan's loop carries several 64x128 and 64x64 fp32 tiles plus the state; Triton pipelines a
# loop across `num_stages` and multiplies the shared memory it needs by it, which at the default
# asks for 224 kiB against this board's 99 kiB. There is nothing to prefetch here anyway -- the
# next chunk cannot start before the state is written.
STAGES_SCAN = int(_S.get("GDN_PREFILL_STAGES"))
# How the products are done. `ieee` is true fp32 on the CUDA cores, which is what `torch.matmul`
# does here and what the recurrent state was written for; `tf32` is ten mantissa bits on the tensor
# cores; `tf32x3` is three tf32 passes, which reproduces fp32 to about a bit and still runs on the
# tensor cores. Which one ships is a measurement against the gate, not a preference.
PREC = _S.get("GDN_PREFILL_PREC")
# The scan may need a different one from the intra pass: they carry different error.
PREC_SCAN = _S.get("GDN_PREFILL_PREC_SCAN")
# The two products inside the scan that never touch the recurrent state -- `q . k^T` and its product
# with the chunk's pseudo-values -- only reach a bf16 output, so they do not need the split. That is
# two of the scan's five products back at one pass instead of three.
PREC_A = _S.get("GDN_PREFILL_PREC_A")
# How the UT transform's matrix is inverted: `0` the forward substitution the reference uses, `1`
# the doubling series. The series is four times cheaper and it is WRONG on this model -- see the
# file header. It stays selectable because the measurement that says so belongs in the ledger.
SERIES = _S.get("GDN_PREFILL_SERIES") == "1"


def fused_prefill_refusal(chunk: int, have_triton: bool = HAVE_TRITON) -> str:
    """Why `fused_chunk_prefill` cannot run with this blocking, or "" if it can.

    The two preconditions below are raises, not fallbacks, and the caller pays for them in the
    middle of a prefill -- which at 16k is minutes of work already done. `QWEN38_GDN_CHUNK` is a
    documented knob and a machine without Triton is a supported configuration, so the engine asks
    this once, at startup, and keeps the reference path when the answer is no.
    """
    if not have_triton:
        return "triton is not available"
    if chunk != CHUNK:
        return f"this kernel is built for chunk {CHUNK}, not {chunk}"
    return ""


def fused_chunk_prefill(query, key, value, g, beta, state=None, *, chunk_size: int = CHUNK,
                        bv: int | None = None, prec: str | None = None,
                        series: bool | None = None, prec_scan: str | None = None,
                        prec_a: str | None = None):
    """The chunked gated delta rule over a whole sequence, two kernels, no chunk-shaped temporaries.

    Signature and shapes are `engine.gdn.chunk_gated_delta_rule`'s: `[B, T, H, D]` in, `[B, T, H,
    Dv]` and the final state `[B, H, Dk, Dv]` out. It is the PREFILL path only -- no tree, no
    factors, no partial accept -- because those callers pass a single chunk, where the reference's
    whole-tensor form costs nothing and its factor outputs are the point.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    B, T, Hk, Dk = query.shape
    H, Dv = value.shape[2], value.shape[-1]
    if H % Hk:
        raise ValueError(f"{H} value heads over {Hk} key heads")
    rep = H // Hk
    if B != 1:
        raise ValueError(f"fused_chunk_prefill is a single sequence, got B={B}")
    if chunk_size != CHUNK:
        raise ValueError(f"this kernel is built for chunk {CHUNK}, got {chunk_size}")
    bv = bv or BV
    prec = prec or PREC
    prec_scan = prec_scan or PREC_SCAN or prec
    prec_a = prec_a or PREC_A
    series = SERIES if series is None else series
    if Dv % bv:
        raise ValueError(f"value head dim {Dv} is not a multiple of bv={bv}")

    # The l2 normalisation stays in PyTorch and in the INPUT dtype, which is where the reference
    # does it: two elementwise passes over a bf16 tensor, against a state that sums over the whole
    # sequence and would carry any difference in them to the end of it.
    from engine import gdn as _gdn
    q = _gdn.l2norm(query, dim=-1, eps=1e-6).reshape(T, Hk, Dk).contiguous()
    k = _gdn.l2norm(key, dim=-1, eps=1e-6).reshape(T, Hk, Dk).contiguous()
    v = value.reshape(T, H, Dv).contiguous()
    gg = g.reshape(T, H).contiguous()
    bb = beta.reshape(T, H).contiguous()
    nc = (T + chunk_size - 1) // chunk_size

    ub = torch.empty(H, T, Dv, dtype=torch.float32, device=q.device)
    kcd = torch.empty(H, T, Dk, dtype=torch.float32, device=q.device)
    out = torch.empty(T, H, Dv, dtype=query.dtype, device=q.device)
    S = (torch.zeros(1, H, Dk, Dv, dtype=torch.float32, device=q.device)
         if state is None else state.clone().float().reshape(1, H, Dk, Dv))

    _intra[(H, nc)](k, v, gg, bb, ub, kcd,
                    T, k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                    gg.stride(0), gg.stride(1),
                    C=chunk_size, DK=Dk, DV=Dv, STEPS=STEPS, PREC=prec, SERIES=series,
                    REP=rep, num_warps=WARPS_INTRA)
    _scan[(H, Dv // bv)](q, k, gg, ub, kcd, out, S,
                         T, nc, q.stride(0), q.stride(1), gg.stride(0), gg.stride(1),
                         out.stride(0), out.stride(1),
                         C=chunk_size, DK=Dk, DV=Dv, BV=bv,
                         SCALE=Dk ** -0.5, PREC=prec_scan, PREC_A=prec_a, REP=rep,
                         num_warps=WARPS_SCAN, num_stages=STAGES_SCAN)
    return out.reshape(B, T, H, Dv), S.reshape(B, H, Dk, Dv)


def check(T: int = 2048, H: int = 48, Dk: int = 128, Dv: int = 128, device: str = "cuda",
          seed: int = 0, bv: int | None = None, prec: str | None = None,
          series: bool | None = None, corr: float = 0.0,
          prec_scan: str | None = None) -> dict:
    """The fused path against `engine/gdn.py`'s, on the real shapes, with a state that is not zero.

    Reports the largest absolute difference in the output and in the final state, both beside the
    bf16 ulp at the same magnitude -- which is the unit the engine-level gate is stated in.
    """
    from engine import gdn
    torch.manual_seed(seed)
    # `corr` is the whole point of this argument list. Independent random keys in 128 dimensions are
    # very nearly orthogonal, so the matrix the UT transform inverts is very nearly the identity and
    # ANY inversion scheme passes. Real keys, after a convolution and a shared residual stream, are
    # not. `corr` mixes a per-head common direction into every key and is what makes this check able
    # to fail.
    q = (torch.randn(1, T, H, Dk, device=device) * 0.5)
    k = (torch.randn(1, T, H, Dk, device=device) * 0.5)
    if corr:
        common = torch.randn(1, 1, H, Dk, device=device)
        k = k * (1 - corr) + common * corr
        q = q * (1 - corr) + common * corr
    q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)
    v = (torch.randn(1, T, H, Dv, device=device, dtype=torch.bfloat16) * 0.5)
    beta = torch.rand(1, T, H, device=device, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, device=device, dtype=torch.float32) * 0.05
    S0 = torch.randn(1, H, Dk, Dv, device=device, dtype=torch.float32) * 0.01
    with torch.no_grad():
        ref_o, ref_S = gdn.chunk_gated_delta_rule(q, k, v, g, beta, S0, chunk_size=CHUNK)
        got_o, got_S = fused_chunk_prefill(q, k, v, g, beta, S0, bv=bv, prec=prec, series=series,
                                            prec_scan=prec_scan)
    do = (got_o.float() - ref_o.float()).abs().max().item()
    ds = (got_S - ref_S).abs().max().item()
    scale_o = ref_o.float().abs().max().item()
    scale_s = ref_S.abs().max().item()
    return {
        "T": T, "max_abs_out": do, "max_abs_state": ds,
        "out_ulp_bf16": do / (scale_o * 2 ** -8 + 1e-30),
        "state_rel": ds / (scale_s + 1e-30),
    }


def timed(fn, reps: int = 3, warm: int = 1) -> float:
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        for _ in range(reps):
            fn()
        ev[1].record()
        torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) / reps


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="check the fused prefill kernels against the reference")
    ap.add_argument("--lens", default="256,2048,8192")
    ap.add_argument("--bv", type=int, default=None)
    ap.add_argument("--time", action="store_true", help="also time one mixer, both paths")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--bvs", default="32,64,128")
    ap.add_argument("--precs", default="tf32,tf32x3,ieee")
    ap.add_argument("--corrs", default="0.0,0.7")
    ap.add_argument("--precs-scan", default="")
    ap.add_argument("--both-inverses", action="store_true")
    a = ap.parse_args()
    for T in [int(x) for x in a.lens.split(",")]:
        for prec in a.precs.split(","):
            for series in ([False, True] if a.both_inverses else [SERIES]):
                for ps in (a.precs_scan or prec).split(","):
                    for corr in [float(x) for x in a.corrs.split(",")]:
                        r = check(T=T, bv=a.bv, prec=prec, series=series, corr=corr, prec_scan=ps)
                        print(f"T={r['T']:6d} intra {prec:>7} scan {ps:>7} "
                              f"{'series' if series else 'subst':>6} corr {corr:4.2f}  "
                              f"max|out| {r['max_abs_out']:10.3e} = {r['out_ulp_bf16']:8.2f} ulp   "
                              f"max|S| {r['max_abs_state']:10.3e} rel {r['state_rel']:.2e}")
        r = check(T=T, bv=a.bv)
        if not a.time:
            continue
        from engine import gdn as _g
        torch.manual_seed(0)
        H, Dk, Dv = 48, 128, 128
        q = torch.randn(1, T, H, Dk, device="cuda", dtype=torch.bfloat16) * 0.5
        k = torch.randn(1, T, H, Dk, device="cuda", dtype=torch.bfloat16) * 0.5
        v = torch.randn(1, T, H, Dv, device="cuda", dtype=torch.bfloat16) * 0.5
        bt = torch.rand(1, T, H, device="cuda", dtype=torch.bfloat16)
        gg = -torch.rand(1, T, H, device="cuda", dtype=torch.bfloat16) * 0.05
        S0 = torch.zeros(1, H, Dk, Dv, device="cuda", dtype=torch.float32)
        ref_ms = timed(lambda: _g.chunk_gated_delta_rule(q, k, v, gg, bt, S0, chunk_size=CHUNK),
                       reps=a.reps)
        print(f"        reference {ref_ms:8.3f} ms   ({ref_ms * 48 / 1e3:.2f} s over 48 layers)")
        print(f"        {'bv':>4}{'intra':>8}{'scan':>8}{'intra ms':>10}{'scan ms':>10}"
              f"{'total':>9}{'x ref':>8}{'48 layers':>11}")
        for bv in ([a.bv] if a.bv else [int(x) for x in a.bvs.split(",")]):
            for prec in a.precs.split(","):
                qn = _g.l2norm(q, dim=-1, eps=1e-6).reshape(T, 48, 128).contiguous()
                kn = _g.l2norm(k, dim=-1, eps=1e-6).reshape(T, 48, 128).contiguous()
                vv = v.reshape(T, 48, 128).contiguous()
                g2 = gg.reshape(T, 48).contiguous()
                b2 = bt.reshape(T, 48).contiguous()
                nc = (T + CHUNK - 1) // CHUNK
                ub = torch.empty(48, T, 128, dtype=torch.float32, device="cuda")
                kcd = torch.empty(48, T, 128, dtype=torch.float32, device="cuda")
                o = torch.empty(T, 48, 128, dtype=torch.bfloat16, device="cuda")
                SS = S0.clone()
                try:
                    i_ms = timed(lambda: _intra[(48, nc)](
                        kn, vv, g2, b2, ub, kcd, T, kn.stride(0), kn.stride(1), vv.stride(0),
                        vv.stride(1), g2.stride(0), g2.stride(1), C=CHUNK, DK=128, DV=128,
                            STEPS=STEPS, PREC=prec, SERIES=SERIES,
                        num_warps=WARPS_INTRA), reps=a.reps)
                    i_done = True
                except Exception as exc:
                    print(f"        {bv:>4}{prec:>8} intra {type(exc).__name__}")
                    continue
                for ps in (a.precs_scan or prec).split(","):
                    try:
                        s_ms = timed(lambda ps=ps: _scan[(48, 128 // bv)](
                            qn, kn, g2, ub, kcd, o, SS, T, nc, qn.stride(0), qn.stride(1),
                            g2.stride(0), g2.stride(1), o.stride(0), o.stride(1), C=CHUNK, DK=128,
                            DV=128, BV=bv, SCALE=128 ** -0.5, PREC=ps, PREC_A=PREC_A,
                            num_warps=WARPS_SCAN,
                            num_stages=STAGES_SCAN), reps=a.reps)
                    except Exception as exc:
                        print(f"        {bv:>4}{prec:>8}{ps:>8}   scan {type(exc).__name__}")
                        continue
                    tot = i_ms + s_ms
                    print(f"        {bv:>4}{prec:>8}{ps:>8}{i_ms:>10.3f}{s_ms:>10.3f}{tot:>9.3f}"
                          f"{ref_ms / tot:>8.2f}{tot * 48 / 1e3:>10.2f}s")
