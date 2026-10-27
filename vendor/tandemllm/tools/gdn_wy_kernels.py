"""The GDN verify recurrence in the WY / UT form: every row of a block at once.

`tools/gdn_verify_kernels.py::_block_step` / `_tree_step` walk a verify block's T rows one after
another: a program holds a [128, BV] state tile and, per row, decays it, reduces it against the key,
updates it and reduces it against the query -- two 128-long reductions and a rank-1 update a row, T
rows in a dependency chain. At T = 16 that is ~81 us a layer against a 13 us byte floor for the
state; at the Phase 2 budgets (24-node trees, T up to 32) it is longer still.

The same arithmetic has a closed form over the block. With G_t the cumulative gate along the path
to row t, and anc(t, j) "j is t or an ancestor of t" (a chain: j <= t):

    S_t   = exp(G_t) S0 + sum_{j anc t} exp(G_t - G_j) k_j d_j^T
    d_t   = beta_t (v_t - exp(G_t) k_t^T S0 - sum_{j strict anc t} exp(G_t - G_j) (k_t . k_j) d_j)
    o_t   = exp(G_t) q_t^T S0 + sum_{j anc t} exp(G_t - G_j) (q_t . k_j) d_j

so with A[t, j] = beta_t exp(G_t - G_j) (k_t . k_j) on the strict ancestors, (I + A) d = beta (v -
exp(G) K S0), and inverting the unit lower-triangular (I + A) once per head:

    d  = U0 - W S0          U0 = (I + A)^-1 (beta v)        W = (I + A)^-1 (beta exp(G) k)
    o  = Qe S0 + QKD d      Qe = exp(G) q                   QKD[t, j] = exp(G_t - G_j) (q_t . k_j)
    S_T = exp(G_T) S0 + (w k)^T d                           w_j = exp(G_T - G_j)    (a chain's end)

A tree is the chain's kernel with the ancestor mask in place of `j <= t` (DFS pre-order keeps it
lower-triangular): `d_t` are the commit's per-row updates `u`, exactly the factors the sequential
kernels write, so the commit and the fold read them unchanged.

Two kernels a layer:

  _wy_prep   grid (H,): per value head, everything that does not read the state -- the key/query
             l2 norms (fp32, as the sequential kernels), the path gate (a tree's, from the ancestor
             mask; a chain's is the gate kernel's cumulative sum), the decays, A and its inverse by
             forward substitution in registers (never the doubling series: NaN on this model, see
             `tools/gdn_prefill_kernels.py`), U0, W, Qe, QKD, and the factors kk / gc.
  _wy_apply  grid (H, Dv / BV): the pending commit applied to the state tile (the `_pending`,
             op for op), then d, o, and for a chain that stores its walk, S_T. No loop over rows.

Same mathematics as the sequential walk in a different order: not bit-identical. `check()` bounds
the difference against the sequential kernels; the losslessness gate decides.
"""

from __future__ import annotations

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

if HAVE_TRITON:
    from tools.gdn_prefill_kernels import _mm
    from tools.gdn_verify_kernels import _pending

    @triton.jit
    def _unit_lower_inverse(L, tt, TP: tl.constexpr, B: tl.constexpr, PREC: tl.constexpr):
        """(I - L)^-1 for L strictly lower triangular [TP, TP], without a TP-long chain of row
        reductions. The diagonal blocks of B rows by forward substitution, all blocks at once and one
        product a step (rows r, r + B, ... take their final value together); then with D^-1 the
        block-diagonal inverse and N the strictly block-lower part of L, M = D^-1 N is nilpotent of
        order TP / B and (I - L)^-1 = (I + M + ... + M^(TP/B - 1)) D^-1 -- a polynomial of degree at
        most three, never the doubling series over the whole matrix (unstable on this model's keys,
        tools/gdn_prefill_kernels.py). B = TP is the plain substitution."""
        blk = tt // B
        same = blk[:, None] == blk[None, :]
        Ld = tl.where(same, L, 0.0)
        for r in tl.static_range(1, B):
            rows = tl.where(((tt % B) == r)[:, None], Ld, 0.0)
            Ld = Ld + _mm(rows, Ld, PREC)
        eye = tl.where(tt[:, None] == tt[None, :], 1.0, 0.0)
        dinv = Ld + eye
        if TP // B > 1:
            M = _mm(dinv, tl.where(same, 0.0, L), PREC)
            X = eye + M
            if TP // B > 2:
                M2 = _mm(M, M, PREC)
                X = X + M2 + _mm(M, M2, PREC)
            dinv = _mm(X, dinv, PREC)
        return dinv

    @triton.jit
    def _conv_rows(X, s_xt, CST, s_cs, CW, WIN, ch, tt, mt, TREE: tl.constexpr,
                   WIDTH: tl.constexpr):
        """`_verify_conv`'s output for channels `ch` and all rows at once -- the window of
        row t is joined columns t .. t+WIDTH-1 of [conv state | x rows] (a tree's from WIN), summed
        in fp32, SiLU, rounded to bf16 where `_verify_conv` stores it -- instead of a walk down the
        rows in a 40-program launch of its own."""
        acc = tl.zeros([tt.shape[0], ch.shape[0]], dtype=tl.float32)
        for w in tl.static_range(WIDTH):
            if TREE:
                idx = tl.load(WIN + tt * WIDTH + w, mask=mt, other=0).to(tl.int32)
            else:
                idx = tt + w
            fs = idx < WIDTH - 1
            sv = tl.load(CST + ch[None, :] * s_cs + idx[:, None],
                         mask=mt[:, None] & fs[:, None], other=0.0)
            xv = tl.load(X + (idx - (WIDTH - 1))[:, None] * s_xt + ch[None, :],
                         mask=mt[:, None] & (~fs)[:, None], other=0.0)
            wv = tl.load(CW + ch * WIDTH + w).to(tl.float32)
            acc += tl.where(fs[:, None], sv, xv).to(tl.float32) * wv[None, :]
        acc = acc * tl.sigmoid(acc)
        return acc.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _conv_shift(X, s_xt, CST, s_cs, ch, T, WIDTH: tl.constexpr):
        """A chain's new convolution state: the last WIDTH-1 columns of its last row's window."""
        for j in tl.static_range(WIDTH - 1):
            col = T + j
            if col < WIDTH - 1:
                val = tl.load(CST + ch * s_cs + col)
            else:
                val = tl.load(X + (col - (WIDTH - 1)) * s_xt + ch)
            tl.store(CST + ch * s_cs + j, val)

    @triton.jit
    def _conv_probe(X, s_xt, CST, s_cs, CW, WIN, OUT, T, C, TP: tl.constexpr, NC: tl.constexpr,
                    TREE: tl.constexpr, WIDTH: tl.constexpr):
        """`_conv_rows` on its own, for the checks: the fused path's convolution rows, whose bits the
        per-element sum order fixes whatever the layout."""
        tt = tl.arange(0, TP)
        mt = tt < T
        ch = tl.program_id(0) * NC + tl.arange(0, NC)
        y = _conv_rows(X, s_xt, CST, s_cs, CW, WIN, ch, tt, mt, TREE, WIDTH)
        tl.store(OUT + tt[:, None] * C + ch[None, :], y.to(OUT.dtype.element_ty),
                 mask=mt[:, None] & (ch < C)[None, :])

    @triton.jit
    def _qk_rows(Q, K, s_t, X, s_xt, CST, s_cs, CW, WIN, KEY_DIM, hk, oc, tt, mt,
                 DK: tl.constexpr, FUSED: tl.constexpr, TREE: tl.constexpr, WIDTH: tl.constexpr):
        """The unnormalised query and key rows [TP, len(oc)] of key head `hk`, channels `oc` of its
        DK: convolved from the raw projection (FUSED) or loaded from the convolution's output."""
        if FUSED:
            q = _conv_rows(X, s_xt, CST, s_cs, CW, WIN, hk * DK + oc, tt, mt, TREE, WIDTH)
            k = _conv_rows(X, s_xt, CST, s_cs, CW, WIN, KEY_DIM + hk * DK + oc, tt, mt, TREE,
                           WIDTH)
        else:
            q = tl.load(Q + tt[:, None] * s_t + hk * DK + oc[None, :], mask=mt[:, None],
                        other=0.0).to(tl.float32)
            k = tl.load(K + tt[:, None] * s_t + hk * DK + oc[None, :], mask=mt[:, None],
                        other=0.0).to(tl.float32)
        return q, k

    @triton.jit
    def _wy_prep(Q, K, s_t, G, BETA, GC, ANC, s_anc, T, H,
                 KK, kk_h, gc_h, QE, INV, QKD, KW, EGT,
                 X, s_xt, CST, s_cs, CW, WIN, A, B_, s_a, s_b, ALOG, DTB, BETA_S, KEY_DIM,
                 TP: tl.constexpr, DK: tl.constexpr, REP: tl.constexpr, EPS: tl.constexpr,
                 SCALE: tl.constexpr, TREE: tl.constexpr, B: tl.constexpr, PREC: tl.constexpr,
                 STORE: tl.constexpr, FUSED: tl.constexpr, WIDTH: tl.constexpr,
                 KC: tl.constexpr, DBG: tl.constexpr = 0):
        """Per value head, what does not read the state: the normalised keys (the commit's factor)
        and queries, the path gate, (I + A)^-1, and QKD. FUSED: the convolution of the q
        and k channels and the gates from the raw projections here, no kernels of their own.
        KC < DK (a 32-row tile): q and k in slices of KC key channels, see below."""
        h = tl.program_id(0)
        hk = h // REP
        tt = tl.arange(0, TP)
        mt = tt < T
        ok = tl.arange(0, DK)
        if KC < DK:
            # at a 32-row tile the whole [TP, DK] q and k tiles and their fp32 products
            # do not fit (ptxas: 16-22 KB of spills on the fused tree, 5 KB on a 24-row chain). So
            # the Gram products run over KC-channel slices of the raw rows, summed with the norms'
            # squares, and are scaled by the norms afterwards; a second pass over the slices writes
            # the normalised keys and Qe. Same mathematics, the norm and the products summed in
            # another order.
            oc = tl.arange(0, KC)
            sq = tl.zeros([TP], dtype=tl.float32)
            sk = tl.zeros([TP], dtype=tl.float32)
            gkk = tl.zeros([TP, TP], dtype=tl.float32)
            gqk = tl.zeros([TP, TP], dtype=tl.float32)
            for c in range(DK // KC):
                qc, kc = _qk_rows(Q, K, s_t, X, s_xt, CST, s_cs, CW, WIN, KEY_DIM, hk, c * KC + oc,
                                  tt, mt, DK, FUSED, TREE, WIDTH)
                sq += tl.sum(qc * qc, axis=1)
                sk += tl.sum(kc * kc, axis=1)
                gkk += _mm(kc, tl.trans(kc), PREC)
                gqk += _mm(qc, tl.trans(kc), PREC)
            rk = tl.rsqrt(sk + EPS)
            rq = tl.rsqrt(sq + EPS) * SCALE
            kkt = gkk * rk[:, None] * rk[None, :]
            qkt = gqk * rq[:, None] * rk[None, :]
        else:
            # The order is the register budget: q and k -- two [TP, DK] fp32 tiles -- are loaded
            # (or convolved) together, give their two products and their stores, and die before the
            # inverse's [TP, TP] tiles are born (at TP = 32 the other order spilled: hold 6; loading
            # k, its product, then q spilled worse on the fused path: hold 7).
            q, k = _qk_rows(Q, K, s_t, X, s_xt, CST, s_cs, CW, WIN, KEY_DIM, hk, ok, tt, mt, DK,
                            FUSED, TREE, WIDTH)
            q = q * tl.rsqrt(tl.sum(q * q, axis=1) + EPS)[:, None] * SCALE
            k = k * tl.rsqrt(tl.sum(k * k, axis=1) + EPS)[:, None]
            tl.store(KK + h * kk_h + tt[:, None] * DK + ok[None, :], k, mask=mt[:, None])
            kkt = _mm(k, tl.trans(k), PREC)
            # DBG (timing only, `bench`): 1 skips the inverse, 2 the QK product, 3 both
            if DBG == 2 or DBG == 3:
                qkt = kkt
            else:
                qkt = _mm(q, tl.trans(k), PREC)
        if FUSED:
            a = tl.load(A + tt * s_a + h, mask=mt, other=0.0).to(tl.float32)
            b = tl.load(B_ + tt * s_b + h, mask=mt, other=0.0).to(tl.float32)
            x = a + tl.load(DTB + h).to(tl.float32)
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            g = tl.where(mt, -tl.exp(tl.load(ALOG + h).to(tl.float32)) * sp, 0.0)
            beta = tl.where(mt, 1.0 / (1.0 + tl.exp(-b)), 0.0)
            tl.store(BETA_S + h * TP + tt, beta)
        else:
            beta = tl.load(BETA + tt * H + h, mask=mt, other=0.0)
            g = tl.load(G + tt * H + h, mask=mt, other=0.0)
        if TREE:
            anc = tl.load(ANC + tt[:, None] * s_anc + tt[None, :],
                          mask=mt[:, None] & mt[None, :], other=0) != 0
            gc = tl.sum(tl.where(anc, g[None, :], 0.0), axis=1)
            tl.store(GC + h * gc_h + tt, gc, mask=mt)
        else:
            anc = (tt[:, None] >= tt[None, :]) & mt[:, None] & mt[None, :]
            if FUSED:
                gc = tl.cumsum(g, axis=0)
                tl.store(GC + h * gc_h + tt, gc, mask=mt)
            else:
                gc = tl.load(GC + h * gc_h + tt, mask=mt, other=0.0)
        eg = tl.exp(gc)
        base = h * TP
        if STORE:
            # a chain that stores its walk: S_T = exp(G_T) S0 + sum_j exp(G_T - G_j) k_j d_j^T
            gT = tl.sum(tl.where(tt == T - 1, gc, 0.0))
            tl.store(EGT + h, tl.exp(gT))
        if KC < DK:
            # the second pass: the slices again, normalised, into the factor and Qe
            for c in range(DK // KC):
                qc, kc = _qk_rows(Q, K, s_t, X, s_xt, CST, s_cs, CW, WIN, KEY_DIM, hk, c * KC + oc,
                                  tt, mt, DK, FUSED, TREE, WIDTH)
                kc = kc * rk[:, None]
                tl.store(KK + h * kk_h + tt[:, None] * DK + (c * KC + oc)[None, :], kc,
                         mask=mt[:, None])
                tl.store(QE + (base + tt[:, None]) * DK + (c * KC + oc)[None, :],
                         qc * (rq * eg)[:, None])
                if STORE:
                    kw = kc * tl.where(mt, tl.exp(gT - gc), 0.0)[:, None]
                    tl.store(KW + (base + tt[:, None]) * DK + (c * KC + oc)[None, :], kw)
        else:
            tl.store(QE + (base + tt[:, None]) * DK + ok[None, :], q * eg[:, None])
            if STORE:
                kw = k * tl.where(mt, tl.exp(gT - gc), 0.0)[:, None]
                tl.store(KW + (base + tt[:, None]) * DK + ok[None, :], kw)
        strict = anc & (tt[:, None] != tt[None, :])
        dec = tl.where(anc, tl.exp(tl.where(anc, gc[:, None] - gc[None, :], 0.0)), 0.0)
        tl.store(QKD + (base + tt[:, None]) * TP + tt[None, :], qkt * dec)
        L = tl.where(strict, -kkt * dec * beta[:, None], 0.0)
        if DBG == 1 or DBG == 3:
            inv = L
        else:
            inv = _unit_lower_inverse(L, tt, TP, B, PREC)
        tl.store(INV + (base + tt[:, None]) * TP + tt[None, :], inv)

    @triton.jit
    def _wy_apply(V, s_t, BETA, GC, gc_h, S, S_OUT, OUT, DELTA, KK, kk_h, QE, INV, QKD, KW, EGT,
                  T, s_h, s_k, s_v, u_h,
                  PK, PU, PG, PROWS, PN, pk_h, pk_t, pu_h, pu_t, pg_h,
                  X, s_xt, CST, s_cs, CW, WIN, BETA_S, KEY_DIM,
                  TP: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
                  PEND: tl.constexpr, STORE: tl.constexpr, PREC: tl.constexpr,
                  FUSED: tl.constexpr, TREE: tl.constexpr, REP: tl.constexpr,
                  WIDTH: tl.constexpr, NQ: tl.constexpr, KC: tl.constexpr):
        """Per (value head, value block): d = (I + A)^-1 beta (v - exp(G) K S0), o = Qe S0 + QKD d,
        and a stored chain's S_T -- the state tile read once, no loop over rows. FUSED: v's
        convolution here, and a chain's convolution state advanced (the v channels by every program,
        q and k by the programs of each key head's first value head, a slice a value block -- after
        `_wy_prep` has read them). KC < DK (a 32-row tile): the state tile in KC-row slices."""
        h = tl.program_id(0)
        vb = tl.program_id(1)
        H = tl.num_programs(0)
        tt = tl.arange(0, TP)
        mt = tt < T
        ok = tl.arange(0, DK)
        ov = vb * BV + tl.arange(0, BV)
        tile = h * s_h + ok[:, None] * s_k + ov[None, :] * s_v
        if KC < DK:
            # K S0 and Qe S0 summed over KC-row slices of the state tile, each slice's
            # pending commit applied and written back as `_pending` does it element for element (the
            # same bits), so no [DK, BV] tile and no [TP, DK] operand is live at once
            base = h * TP
            oc = tl.arange(0, KC)
            ks = tl.zeros([TP, BV], dtype=tl.float32)
            qs = tl.zeros([TP, BV], dtype=tl.float32)
            for c in range(DK // KC):
                okc = c * KC + oc
                tc = h * s_h + okc[:, None] * s_k + ov[None, :] * s_v
                if PEND:
                    s0c = _pending(S, tc, PK, PU, PG, PROWS, PN, pk_h, pk_t, pu_h, pu_t, pg_h, h,
                                   okc, ov)
                else:
                    s0c = tl.load(S + tc)
                kc = tl.load(KK + h * kk_h + tt[:, None] * DK + okc[None, :], mask=mt[:, None],
                             other=0.0)
                ks += _mm(kc, s0c, PREC)
                qec = tl.load(QE + (base + tt[:, None]) * DK + okc[None, :])
                qs += _mm(qec, s0c, PREC)
        else:
            if PEND:
                s0 = _pending(S, tile, PK, PU, PG, PROWS, PN, pk_h, pk_t, pu_h, pu_t, pg_h, h, ok,
                              ov)
            else:
                s0 = tl.load(S + tile)
            base = h * TP
            k = tl.load(KK + h * kk_h + tt[:, None] * DK + ok[None, :], mask=mt[:, None],
                        other=0.0)
        if FUSED:
            vch = 2 * KEY_DIM + h * DV + ov
            v = _conv_rows(X, s_xt, CST, s_cs, CW, WIN, vch, tt, mt, TREE, WIDTH)
            beta = tl.load(BETA_S + h * TP + tt)
            if not TREE:
                # every thread's window loads are in before any thread overwrites the state
                tl.debug_barrier()
                _conv_shift(X, s_xt, CST, s_cs, vch, T, WIDTH)
                if h % REP == 0:
                    sub = vb * NQ + tl.arange(0, NQ)
                    _conv_shift(X, s_xt, CST, s_cs, (h // REP) * DK + sub, T, WIDTH)
                    _conv_shift(X, s_xt, CST, s_cs, KEY_DIM + (h // REP) * DK + sub, T, WIDTH)
        else:
            v = tl.load(V + tt[:, None] * s_t + h * DV + ov[None, :], mask=mt[:, None],
                        other=0.0).to(tl.float32)
            beta = tl.load(BETA + tt * H + h, mask=mt, other=0.0)
        eg = tl.exp(tl.load(GC + h * gc_h + tt, mask=mt, other=0.0))
        if KC < DK:
            r = (v - eg[:, None] * ks) * beta[:, None]
        else:
            r = (v - eg[:, None] * _mm(k, s0, PREC)) * beta[:, None]
        inv = tl.load(INV + (base + tt[:, None]) * TP + tt[None, :])
        d = _mm(inv, r, PREC)
        if KC < DK:
            qkd = tl.load(QKD + (base + tt[:, None]) * TP + tt[None, :])
            o = qs + _mm(qkd, d, PREC)
        else:
            qe = tl.load(QE + (base + tt[:, None]) * DK + ok[None, :])
            qkd = tl.load(QKD + (base + tt[:, None]) * TP + tt[None, :])
            o = _mm(qe, s0, PREC) + _mm(qkd, d, PREC)
        tl.store(OUT + (tt[:, None] * H + h) * DV + ov[None, :], o.to(OUT.dtype.element_ty),
                 mask=mt[:, None])
        tl.store(DELTA + h * u_h + tt[:, None] * DV + ov[None, :], d, mask=mt[:, None])
        if STORE:
            if KC < DK:
                # the entry slices again (the pending commit is in memory now), a slice at a time
                tl.debug_barrier()
                egt = tl.load(EGT + h)
                for c in range(DK // KC):
                    okc = c * KC + oc
                    tc = h * s_h + okc[:, None] * s_k + ov[None, :] * s_v
                    kwt = tl.load(KW + (base + tt[None, :]) * DK + okc[:, None])
                    tl.store(S_OUT + tc, egt * tl.load(S + tc) + _mm(kwt, d, PREC))
            else:
                # loaded transposed ([DK, TP]) rather than transposed in registers: at TP = 32 the
                # register transpose of a [TP, DK] tile spilled (hold 4: 31 ms a 48-layer block)
                kwt = tl.load(KW + (base + tt[None, :]) * DK + ok[:, None])
                s = tl.load(EGT + h) * s0 + _mm(kwt, d, PREC)
                tl.store(S_OUT + tile, s)


# `_wy_apply`'s value block and warps, `_wy_prep`'s warps and diagonal block, and the products'
# precision (`ieee` = fp32 FMA on the CUDA cores; `bf16x3` = three bf16 tensor-core passes, about
# sixteen mantissa bits; see tools/gdn_prefill_kernels.py::_mm).
BV = int(_S.get("WY_BV"))
WARPS = int(_S.get("WY_WARPS"))
WARPS_PREP = int(_S.get("WY_WARPS_PREP"))
BLOCK = int(_S.get("WY_BLOCK"))
PREC = _S.get("WY_PREC")
DBG = int(_S.get("WY_DBG"))            # timing experiments only; 0 = the kernel


def wy_recurrence(q, k, v, s_t: int, g, beta, gc, state, out, delta, kk, T: int, *,
                  key_heads: int, value_heads: int, head_k: int, head_v: int,
                  anc: torch.Tensor | None = None, out_state=None, store_state: bool = True,
                  pend=None, bv: int | None = None, warps: int | None = None,
                  prec: str | None = None, block: int | None = None, fused: dict | None = None,
                  kc: int | None = None) -> None:
    """`verify_mixer`'s recurrence in the WY form. Arguments as its `_block_step` / `_tree_step`
    launches read them: q/k/v views of the convolution's rows (row stride `s_t`), g/beta [T, H],
    gc [H, >= T] (a chain's cumulative gate in, a tree's path gate out), state [H, Dk, Dv], out
    [T, H, Dv], delta [H, >= T, Dv], kk [H, >= T, Dk]; `anc` the tree's ancestor mask [T, T]
    (inclusive), None for a chain.

    `fused`: the raw inputs instead -- dict(mixed [T, C], conv_state [C, W-1], conv_w [C, W],
    window [T, W] or None, a_raw, b_raw [T, H], a_log, dt_bias [H], key_dim) -- and q/k/v/g/beta are
    not read: the convolution and the gates run inside the two kernels.

    `kc`: the key channels a slice at a tile past 16 rows (default QWEN38_GDNV_WY_KC; 0 =
    the whole DK, the kernels as shipped them). A 16-row tile always takes the whole DK."""
    from tools import gdn_verify_kernels as V
    H, dk, dv = value_heads, head_k, head_v
    tp = max(16, triton.next_power_of_2(T))
    kc = V.WY_KC if kc is None else kc
    kc = kc if tp > 16 and 0 < kc < dk else dk
    blk = min(block or BLOCK, tp)
    if block is None and tp // blk > 4:
        # ENG-213: a 64-row tile in 16-row blocks (the polynomial carries 4); a tile of 32 rows or
        # less keeps QWEN38_WY_BLOCK, so its bits do not move
        blk = tp // 4
    if tp // blk > 4:
        raise ValueError(f"the block inverse's polynomial carries 4 blocks, not {tp // blk}")
    dev = state.device
    f32 = torch.float32
    qe = torch.empty(H, tp, dk, dtype=f32, device=dev)
    inv = torch.empty(H, tp, tp, dtype=f32, device=dev)
    qkd = torch.empty(H, tp, tp, dtype=f32, device=dev)
    tree = anc is not None
    chain_store = not tree and store_state
    kw = torch.empty(H, tp, dk, dtype=f32, device=dev) if chain_store else qe
    egt = torch.empty(H, dtype=f32, device=dev) if chain_store else gc
    prec = prec or PREC
    a8 = anc.view(torch.uint8) if tree else gc
    bv = bv or BV
    if fused is not None:
        x, cst, cw = fused["mixed"], fused["conv_state"], fused["conv_w"]
        win = fused["window"] if tree else x
        beta_s = torch.empty(H, tp, dtype=f32, device=dev)
        fz = (x, x.stride(0), cst, cst.stride(0), cw, win)
        fprep = fz + (fused["a_raw"], fused["b_raw"], fused["a_raw"].stride(0),
                      fused["b_raw"].stride(0), fused["a_log"], fused["dt_bias"], beta_s,
                      fused["key_dim"])
        fapply = fz + (beta_s, fused["key_dim"])
        width = cw.shape[-1]
        q = k = v = g = beta = gc                               # never read
    else:
        fprep = (gc, 0, gc, 0, gc, gc, gc, gc, 0, 0, gc, gc, gc, 0)   # never read
        fapply = (gc, 0, gc, 0, gc, gc, gc, 0)
        width = 4
    _wy_prep[(H,)](q, k, s_t, g, beta, gc, a8, anc.stride(0) if tree else 0, T, H,
                   kk, kk.stride(0), gc.stride(0), qe, inv, qkd, kw, egt, *fprep,
                   TP=tp, DK=dk, REP=H // key_heads, EPS=1e-6, SCALE=dk ** -0.5, TREE=tree,
                   B=blk, PREC=prec, STORE=chain_store, FUSED=fused is not None, WIDTH=width,
                   KC=kc, DBG=DBG, num_warps=WARPS_PREP)
    if pend is not None:
        pk, pu, pg, prows, pn = pend
        pargs = (pk, pu, pg, prows, pn, pk.stride(0), pk.stride(1), pu.stride(0), pu.stride(1),
                 pg.stride(0))
    else:
        pargs = (gc, gc, gc, gc, gc, 0, 0, 0, 0, 0)            # never read
    S = state
    So = S if out_state is None else out_state
    _wy_apply[(H, dv // bv)](v, s_t, beta, gc, gc.stride(0), S, So, out, delta, kk, kk.stride(0),
                             qe, inv, qkd, kw, egt, T, S.stride(0), S.stride(1), S.stride(2),
                             delta.stride(0), *pargs, *fapply, TP=tp, DK=dk, DV=dv, BV=bv,
                             PEND=pend is not None, STORE=chain_store, PREC=prec,
                             FUSED=fused is not None, TREE=tree, REP=H // key_heads, WIDTH=width,
                             NQ=dk * bv // dv, KC=kc, num_warps=warps or WARPS)


def conv_rows(mixed, conv_state, conv_w, window=None) -> torch.Tensor:
    """The fused path's convolution output [T, C] (bf16), for `compare`."""
    T, C = mixed.shape
    W = conv_w.shape[-1]
    st = conv_state.reshape(C, W - 1)
    out = torch.empty(T, C, dtype=mixed.dtype, device=mixed.device)
    tp = max(16, triton.next_power_of_2(T))
    _conv_probe[(triton.cdiv(C, 128),)](mixed, mixed.stride(0), st, st.stride(0), conv_w,
                                        window if window is not None else mixed, out, T, C,
                                        TP=tp, NC=128, TREE=window is not None, WIDTH=W)
    return out


def unit_lower_inverse(L: torch.Tensor, block: int) -> torch.Tensor:
    """`_unit_lower_inverse` in torch, for the CPU tests: (I - L)^-1 for strictly lower L [..., n, n]
    by block substitution plus the block-lower polynomial, the same steps in the same order."""
    n = L.shape[-1]
    tt = torch.arange(n)
    blk = tt // block
    same = blk[:, None] == blk[None, :]
    Ld = torch.where(same, L, torch.zeros_like(L))
    for r in range(1, block):
        rows = torch.where(((tt % block) == r)[:, None], Ld, torch.zeros_like(Ld))
        Ld = Ld + rows @ Ld
    eye = torch.eye(n, dtype=L.dtype).expand_as(L)
    dinv = Ld + eye
    nb = n // block
    if nb > 1:
        M = dinv @ torch.where(same, torch.zeros_like(L), L)
        X = eye + M
        if nb > 2:
            M2 = M @ M
            X = X + M2 + M @ M2
        dinv = X @ dinv
    return dinv


def sliced_products(q, k, kc: int, scale: float, eps: float = 1e-6):
    """`_wy_prep`'s sliced path in torch, for the CPU tests: raw q, k [T, Dk] in slices of
    kc channels -> (K K^T, Q K^T of the normalised rows, the normalised k, q * scale), summing the
    squares and the products slice by slice and scaling by the norms afterwards, as the kernel does."""
    T, dk = k.shape
    sq = torch.zeros(T, dtype=q.dtype)
    sk = torch.zeros(T, dtype=q.dtype)
    gkk = torch.zeros(T, T, dtype=q.dtype)
    gqk = torch.zeros(T, T, dtype=q.dtype)
    for c in range(dk // kc):
        qc, kc_ = q[:, c * kc:(c + 1) * kc], k[:, c * kc:(c + 1) * kc]
        sq += (qc * qc).sum(1)
        sk += (kc_ * kc_).sum(1)
        gkk += kc_ @ kc_.t()
        gqk += qc @ kc_.t()
    rk = torch.rsqrt(sk + eps)
    rq = torch.rsqrt(sq + eps) * scale
    return gkk * rk[:, None] * rk[None, :], gqk * rq[:, None] * rk[None, :], k * rk[:, None], \
        q * rq[:, None]


def wy_math(q, k, v, g, beta, S0, anc, block: int = 8):
    """The WY form in torch over [H] heads: q, k [T, H, Dk] (normalised), v [T, H, Dv], g, beta
    [T, H], S0 [H, Dk, Dv], anc [T, T] inclusive ancestors. Returns (o [T, H, Dv], d [H, T, Dv],
    gc [H, T]) -- what `_wy_prep` + `_wy_apply` compute, for the CPU tests."""
    T = q.shape[0]
    n = max(16, 1 << (T - 1).bit_length())
    qh, kh = q.permute(1, 0, 2), k.permute(1, 0, 2)                     # [H, T, Dk]
    vh = v.permute(1, 0, 2)
    gh, bh = g.t(), beta.t()                                            # [H, T]
    a = anc.to(qh.dtype)
    gc = gh @ a.t()                                                     # path gate [H, T]
    dec = torch.where(anc, torch.exp(gc[:, :, None] - gc[:, None, :]), torch.zeros(()))
    strict = anc & ~torch.eye(T, dtype=torch.bool)
    L = torch.where(strict, -(kh @ kh.transpose(1, 2)) * dec * bh[:, :, None], torch.zeros(()))
    Lp = torch.zeros(L.shape[0], n, n, dtype=L.dtype)
    Lp[:, :T, :T] = L
    inv = unit_lower_inverse(Lp, min(block, n))[:, :T, :T]
    eg = torch.exp(gc)
    r = (vh - eg[:, :, None] * (kh @ S0)) * bh[:, :, None]
    d = inv @ r
    o = (qh * eg[:, :, None]) @ S0 + ((qh @ kh.transpose(1, 2)) * dec) @ d
    return o.permute(1, 0, 2), d, gc


# --------------------------------------------------------------------------- checks (board)


def _inputs(n: int, gen, corr: float = 0.0):
    """One layer's verify inputs; `corr` mixes one common direction into every row's keys (real
    keys are far from orthogonal, which is what breaks an unstable inverse)."""
    kd, vh, dv, W = 2048, 48, 128, 4
    C = 2 * kd + vh * dv
    r = lambda *s, sc=1.0: (torch.randn(*s, device="cuda", generator=gen) * sc)  # noqa: E731
    mixed = r(n, C, sc=0.5)
    if corr:
        mixed[:, kd:2 * kd] += corr * r(1, kd)
    return dict(mixed=mixed.to(torch.bfloat16),
                conv_state=r(1, C, W - 1, sc=0.5).to(torch.bfloat16),
                conv_w=r(C, W, sc=0.3).to(torch.bfloat16),
                a_raw=r(n, vh).to(torch.bfloat16), b_raw=r(n, vh).to(torch.bfloat16),
                a_log=r(vh, sc=0.5).to(torch.bfloat16), dt_bias=r(vh, sc=0.5).to(torch.bfloat16),
                state=r(1, vh, 128, dv, sc=0.05))


KW = dict(key_dim=2048, key_heads=16, value_heads=48, head_k=128, head_v=128)


def _tree_args(tree):
    win = torch.tensor(tree.conv_windows(4), dtype=torch.long, device="cuda")
    dep = torch.tensor(tree.depths(), dtype=torch.long, device="cuda")
    anc = torch.tensor(tree.ancestor_mask(), dtype=torch.bool, device="cuda")
    return dict(window=win, depths=dep, anc=anc, max_depth=max(tree.depths()))


def reference64(qkv, a_raw, b_raw, a_log, dt_bias, S0, parents=None, *, key_dim=2048,
                key_heads=16, value_heads=48, head_k=128, head_v=128):
    """The sequential walk in float64 from the convolution's output rows: (o [T, H, Dv], u [H, T,
    Dv], the last row's state [H, Dk, Dv]). A tree walks each node from its parent's state."""
    T = qkv.shape[0]
    H, rep = value_heads, value_heads // key_heads
    x = qkv.double()
    q = x[:, :key_dim].view(T, key_heads, head_k).repeat_interleave(rep, dim=1)
    k = x[:, key_dim:2 * key_dim].view(T, key_heads, head_k).repeat_interleave(rep, dim=1)
    v = x[:, 2 * key_dim:].view(T, H, head_v)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * head_k ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    g = -a_log.double().exp() * torch.nn.functional.softplus(a_raw.double() + dt_bias.double())
    beta = torch.sigmoid(b_raw.double())
    parents = parents if parents is not None else [-1] + list(range(T - 1))
    S0 = S0.reshape(H, head_k, head_v).double()
    states, o, u = [], torch.empty(T, H, head_v, dtype=torch.float64, device=qkv.device), \
        torch.empty(H, T, head_v, dtype=torch.float64, device=qkv.device)
    for t in range(T):
        s = (S0 if parents[t] < 0 else states[parents[t]]) * g[t].exp()[:, None, None]
        kv = torch.einsum("hk,hkv->hv", k[t], s)
        d = (v[t] - kv) * beta[t][:, None]
        s = s + k[t][:, :, None] * d[:, None, :]
        o[t] = torch.einsum("hk,hkv->hv", q[t], s)
        u[:, t] = d
        states.append(s)
    return o, u, states[-1]


class every_size:
    """The WY form at every block size while a check runs, whatever QWEN38_GDNV_WY_*MAXT the
    environment serves with (the batteries run under the candidate's flags)."""

    def __enter__(self):
        from tools import gdn_verify_kernels as V
        self.keep = (V.WY_MAXT, V.WY_CHAIN_MAXT, V.WY_FUSED_MAXT)
        V.WY_MAXT = V.WY_CHAIN_MAXT = V.WY_FUSED_MAXT = 1 << 30

    def __exit__(self, *exc):
        from tools import gdn_verify_kernels as V
        V.WY_MAXT, V.WY_CHAIN_MAXT, V.WY_FUSED_MAXT = self.keep


def _committed(S0, fac, n: int):
    """The state after a chain block whose n rows are all accepted: the commit kernel on the
    factors, as the engine's pending commit applies it."""
    from tools.gdn_commit_kernels import fused_commit
    kk, u, gc = fac
    S = S0.clone()
    fused_commit(S[None], S[None], kk, u, gc, list(range(n)))
    return S


def compare(n: int, tree=None, seed: int = 0, corr: float = 0.0, fused: bool = False) -> dict:
    """One layer's verify mixer, sequential and WY from the same inputs, each against a float64
    walk: the largest relative error of the output (before its bf16 rounding is not visible here:
    after it), the per-row updates u and (a chain) the walked state; and WY against sequential."""
    from tools.gdn_verify_kernels import verify_mixer
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = _inputs(n, gen, corr)
    ta = _tree_args(tree) if tree is not None else {}
    res = {}
    outs = []
    for wy in (False, True):
        y = {k: v.clone() for k, v in x.items()}
        # a chain as the engine serves it: its walked state not stored (the fold keeps the commit
        # pending) and rebuilt here by the commit kernel from the factors, every row accepted
        with every_size():
            o, fac, qkv = verify_mixer(**y, **KW, **ta, wy=wy, fused=fused and wy,
                                       store_state=False)
        so = _committed(x["state"], fac, n) if tree is None else None
        outs.append((o, fac, so, y["conv_state"], qkv))
    rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()  # noqa
    (o0, f0, s0, c0, qkv), (o1, f1, s1, c1, _) = outs
    if fused:
        # the reference walks the fused path's own convolution rows; how many of them differ from
        # `_verify_conv`'s (the four-wide sum in another order) is reported beside it
        fq = conv_rows(x["mixed"], x["conv_state"], x["conv_w"], ta.get("window"))
        res["conv rows differ"] = float((fq != qkv).sum()) / qkv.numel()
        qkv = fq
    ro, ru, rS = reference64(qkv, x["a_raw"], x["b_raw"], x["a_log"], x["dt_bias"], x["state"],
                             tree.parents if tree is not None else None)
    res["o seq"], res["o wy"] = rel(o0[0], ro), rel(o1[0], ro)
    res["u seq"], res["u wy"] = rel(f0[1][0], ru), rel(f1[1][0], ru)
    if s0 is not None:
        res["S seq"], res["S wy"] = rel(s0[0], rS), rel(s1[0], rS)
    res["kk"], res["gc"] = rel(f1[0], f0[0]), rel(f1[2], f0[2])
    res["conv"] = rel(c1, c0)
    return res


def drift(blocks: int = 64, T: int = 16, seed: int = 1) -> float:
    """The recurrent state carried over `blocks` chain blocks (1,024 tokens at 16), sequential
    against WY, each from its own previous state: the relative difference at the end."""
    from tools.gdn_verify_kernels import verify_mixer
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = _inputs(T, gen)
    Sa, Sb = x["state"].clone(), x["state"].clone()
    for _ in range(blocks):
        x = _inputs(T, gen)
        for S, wy in ((Sa, False), (Sb, True)):
            y = {k: v.clone() for k, v in x.items() if k != "state"}
            with every_size():
                _, fac, _ = verify_mixer(**y, state=S, **KW, wy=wy, store_state=False)
            S.copy_(_committed(S, fac, T))
    return ((Sb - Sa).abs().max() / Sa.abs().max()).item()


def check() -> list[str]:
    import random
    from engine.tree import DraftTree
    out, worst = [], {}

    def rand_tree(n, rng, depth_cap=None):
        parents, path = [-1], [0]
        for i in range(1, n):
            cut = rng.randint(1, len(path))
            if depth_cap is not None:
                cut = min(cut, depth_cap)
            path = path[:cut]
            parents.append(path[-1])
            path.append(i)
        return DraftTree(tokens=[0] * n, parents=parents)

    for n in (2, 3, 5, 8, 9, 15, 16, 17, 24, 32):
        r = compare(n)
        out.append(f"chain T={n:2d}: " + "  ".join(f"{k} {v:.1e}" for k, v in r.items()))
        for k, v in r.items():
            worst[k] = max(worst.get(k, 0.0), v)
    rng = random.Random(7)
    for n in (4, 9, 16, 24, 32):
        for _ in range(2):
            r = compare(n, rand_tree(n, rng, depth_cap=15))
            out.append(f"tree  T={n:2d}: " + "  ".join(f"{k} {v:.1e}" for k, v in r.items()))
            for k, v in r.items():
                worst[k] = max(worst.get(k, 0.0), v)
    for n in (2, 3, 16, 24, 32):
        for tree in (None, rand_tree(n, rng, depth_cap=15)):
            r = compare(n, tree, seed=13, fused=True)
            out.append(f"{'tree ' if tree else 'chain'} T={n:2d} FUSED conv+gates: " +
                       "  ".join(f"{k} {v:.1e}" for k, v in r.items()))
            for k, v in r.items():
                worst[k] = max(worst.get(k, 0.0), v)
    for n in (16, 32):
        for tree in (None, rand_tree(n, rng, depth_cap=15)):
            r = compare(n, tree, seed=11, corr=3.0)
            out.append(f"{'tree ' if tree else 'chain'} T={n:2d} correlated keys: " +
                       "  ".join(f"{k} {v:.1e}" for k, v in r.items()))
            for k, v in r.items():
                worst[k] = max(worst.get(k, 0.0), v)
    d = drift()
    out.append(f"state after 64 chain blocks of 16 (1,024 tokens), WY against sequential: {d:.2e}")
    out.append("worst: " + "  ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    return out


GRID = ("ieee:8:32:4:4,ieee:16:32:4:4,ieee:32:32:4:4,bf16x3:8:32:4:4,ieee:8:16:2:4,ieee:8:32:4:8,"
        "ieee:8:32:4:4:f,bf16x3:8:32:4:4:f,ieee:8:16:2:4:f")
# the sliced kernels (`kcNN`) against the whole-DK ones, apart and fused
GRID_KC = ("ieee:8:32:4:4,ieee:8:32:4:4:kc32,ieee:8:32:4:8:kc32,ieee:8:64:4:4:kc32,"
           "ieee:8:32:4:4:f,ieee:8:32:4:4:f:kc32,ieee:8:32:4:8:f:kc32,ieee:8:64:4:4:f:kc32,"
           "ieee:8:64:8:8:f:kc32,ieee:8:32:4:4:f:kc16")


def bench(layers: int = 48, reps: int = 10, sizes=(16, 24, 32), grid: str = GRID) -> list[str]:
    """48 layers' worth of the verify mixer, chain and a branching tree, the sequential kernels
    against WY per `prec:block:bv:warps:warps_prep[:f][:kcNN]` (`f` fused, `kcNN` the slices of
    the key channels), and each WY kernel's own share (profiler)."""
    global PREC, BLOCK, BV, WARPS, WARPS_PREP
    from torch.profiler import ProfilerActivity, profile
    from engine.tree import DraftTree
    from tools import gdn_verify_kernels as V
    from tools.gdn_verify_kernels import verify_mixer
    keep = (PREC, BLOCK, BV, WARPS, WARPS_PREP, V.WY_KC)
    gen = torch.Generator(device="cuda").manual_seed(2)
    out = []

    def timed(fn):
        fn()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(reps):
            fn()
        t1.record()
        t1.synchronize()
        return t0.elapsed_time(t1) / reps

    for T in sizes:
        ins = [_inputs(T, gen) for _ in range(layers)]
        # spines off the root, 12 deep at most (the sequential kernel carries 16 levels)
        parents = [-1]
        while len(parents) < T:
            n = min(12, T - len(parents))
            parents += [0] + list(range(len(parents), len(parents) + n - 1))
        ta = _tree_args(DraftTree(tokens=[0] * T, parents=parents))
        scratch = [torch.empty_like(x["state"]) for x in ins]

        def run(kind, wy, fused=False):
            def one():
                with every_size():
                    for x, sc in zip(ins, scratch):
                        verify_mixer(**x, **KW, **(ta if kind == "tree" else {}),
                                     store_state=False, wy=wy, fused=fused)
            return one
        def kernels(fn):
            """The mixer's own kernels' device time (a graph replays these without the host)."""
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            ker = {e.key: getattr(e, "device_time_total", 0.0) / 1e3 for e in prof.key_averages()
                   if e.key.startswith(("_wy_", "_verify_", "_block_step", "_tree_step"))}
            return sum(ker.values()), " ".join(f"{k.strip('_')} {v:.2f}" for k, v in sorted(ker.items()))

        seq = {kind: timed(run(kind, False)) for kind in ("chain", "tree")}
        ks = {kind: kernels(run(kind, False)) for kind in ("chain", "tree")}
        out.append(f"T={T:2d} sequential: chain {seq['chain']:.3f} (kernels {ks['chain'][0]:.2f}: "
                   f"{ks['chain'][1]})  tree {seq['tree']:.3f} (kernels {ks['tree'][0]:.2f}: "
                   f"{ks['tree'][1]}) ms ({layers} layers, whole mixer)")
        for cfg in grid.split(","):
            p, b, bv, w, wp, *opt = cfg.split(":")
            PREC, BLOCK, BV, WARPS, WARPS_PREP = p, int(b), int(bv), int(w), int(wp)
            fused = "f" in opt
            V.WY_KC = next((int(o[2:]) for o in opt if o.startswith("kc")), 0)
            row = []
            for kind in ("chain", "tree"):
                try:
                    ms = timed(run(kind, True, fused))
                except Exception as e:                     # noqa: BLE001 -- a config that fails
                    row.append(f"{kind} FAIL {type(e).__name__}")
                    continue
                tot, parts = kernels(run(kind, True, fused))
                row.append(f"{kind} {ms:.3f} (kernels {tot:.2f}: {parts})")
            out.append(f"   WY {cfg}: " + "   ".join(row))
    PREC, BLOCK, BV, WARPS, WARPS_PREP, V.WY_KC = keep
    return out


if __name__ == "__main__":
    for line in (bench(grid=os.environ.get("QWEN38_WY_GRID", GRID),
                       sizes=tuple(int(x) for x in os.environ.get("QWEN38_WY_SIZES",
                                                                  "16,24,32").split(",")))
                 if "--bench" in sys.argv else check()):
        print(line, flush=True)
