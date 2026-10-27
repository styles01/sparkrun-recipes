"""LFM2 MoE's row kernels (ENG-287): Triton, one program a row (or a row's head, a pair's columns).

The decode step and every row of a verify block run through these with the same instruction
sequence: no reduction crosses rows and no tile shape depends on the row count, so a row of an M-row
call is the one-row call's row bit for bit, the property speculation needs (docs/kernels.md, "Verify
twins"). The projections are tools/fn_kernels.dense (EXL3) or `linear_rows` (BF16), the EXL3 experts
the bank kernels of tools/fn_kernels; what is here is the rest of a row, with the reference's bf16
roundings (engine/models/lfm2_moe/ref.py):

  add_rms        h = bf16(h + d) (optional), n = bf16(w * bf16(h * rsqrt(mean(h^2) + eps))): the
                 residual add and the next norm in one launch
  conv_rows      the double-gated short conv: bx = bf16(B * x), the depthwise conv over the row's
                 window (the state's last inputs, or its ancestors' bx recomputed from their B and x),
                 y = bf16(C * bf16(conv)); every row's bx kept for the commit; one row (decode) can
                 shift the state in place
  qk_prep        q and k per head: the RMS norm (plain w), NEOX rotary at the row's position; k and v
                 into the caches at the row's slot
  attn_rows      softmax(q k^T / sqrt(D)) v of a row's query heads over its visible keys: the cache's
                 positions before the block's start L, then the row's ancestors (`anc`); fixed chunks
                 of CHUNK keys, one program each, combined in chunk order (an empty chunk adds exact
                 zeros, so the number of programs -- the context class -- does not change a row's bits)
  router         sigmoid of the bf16 logits, the expert bias added for the choice, the top k (ties to
                 the lower id), the chosen sigmoids renormalised (+ 1e-6) and scaled, bf16 as the
                 reference rounds them
  linear_rows    x [M, K] bf16 @ W^T (W bf16 [N, K]): 16-row tiles on the tensor cores, K in order
  moe_bf16       the BF16 experts of every (row, slot) pair: gate and up, a = bf16(bf16(silu(g)) u),
                 down to fp32 (`fn_kernels.moe_combine` adds them as for the EXL3 bank)

Each has a torch twin (`*_reference`) for the CPU tests (TRITON_INTERPRET=1 runs the kernels
themselves on the CPU).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:          # noqa: BLE001  (a Mac: the references only)
    HAVE_TRITON = False

#: keys a program of `attn_rows` reads (fixed: a row's bits do not depend on the context class)
CHUNK = 512
#: ancestors a verify row may have (32 rows of a block and the anchor)
ANC = 33
#: Triton's CPU interpreter multiplies bf16 tiles wrongly in `tl.dot` (Triton 3.7: garbage, fp32 tiles
#: exact): under TRITON_INTERPRET=1 the dots take fp32 tiles, so the CPU tests check the indexing; the
#: GPU always multiplies bf16 on the tensor cores
DOT32 = os.environ.get("TRITON_INTERPRET") == "1"


def classes(max_len: int) -> list[int]:
    """The attention classes: the number of CHUNK programs a call launches, doubling from 4 (2,048
    keys) to what covers `max_len`. A call takes the smallest class that covers its last row."""
    out, n = [], 4
    while True:
        out.append(n)
        if n * CHUNK >= max_len:
            return out
        n *= 2


def class_of(rows: int, max_len: int) -> int:
    for n in classes(max_len):
        if n * CHUNK >= rows:
            return n
    raise ValueError(f"{rows} keys exceed the window {max_len}")


if HAVE_TRITON:

    @triton.jit
    def _add_rms(H_IN, D_IN, H_OUT, N_OUT, W, eps, N: tl.constexpr, BLOCK: tl.constexpr, ADD: tl.constexpr):
        r = tl.program_id(0)
        o = tl.arange(0, BLOCK)
        m = o < N
        h = tl.load(H_IN + r * N + o, mask=m, other=0.0)
        if ADD:
            d = tl.load(D_IN + r * N + o, mask=m, other=0.0)
            h = (h.to(tl.float32) + d.to(tl.float32)).to(tl.bfloat16)
            tl.store(H_OUT + r * N + o, h, mask=m)
        hf = h.to(tl.float32)
        var = tl.sum(hf * hf, 0) / N
        nrm = (hf * tl.rsqrt(var + eps)).to(tl.bfloat16)
        w = tl.load(W + o, mask=m, other=0.0)
        tl.store(N_OUT + r * N + o, (nrm.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16), mask=m)

    @triton.jit
    def _conv_rows(BCX, s_bcx, STATE, WC, SRC, REC, Y, H: tl.constexpr, KT: tl.constexpr, BLOCK: tl.constexpr,
                   WRITE: tl.constexpr):
        # KT taps; SRC[r, j] for j < KT - 1: where the input at distance KT - 1 - j lives (a row of the
        # block >= 0, or -1 - a state column); the last tap is the row itself
        r = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        base = BCX + r * s_bcx
        b = tl.load(base + c).to(tl.float32)
        x = tl.load(base + 2 * H + c).to(tl.float32)
        bx = (b * x).to(tl.bfloat16)
        tl.store(REC + r * H + c, bx)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(KT - 1):
            src = tl.load(SRC + r * (KT - 1) + j)
            row = tl.maximum(src, 0)
            col = tl.maximum(-1 - src, 0)
            vb = (tl.load(BCX + row * s_bcx + c).to(tl.float32)
                  * tl.load(BCX + row * s_bcx + 2 * H + c).to(tl.float32)).to(tl.bfloat16)
            vs = tl.load(STATE + col * H + c)
            v = tl.where(src >= 0, vb, vs).to(tl.float32)
            acc += v * tl.load(WC + c * KT + j).to(tl.float32)
        acc += bx.to(tl.float32) * tl.load(WC + c * KT + (KT - 1)).to(tl.float32)
        cv = acc.to(tl.bfloat16).to(tl.float32)
        ch = tl.load(base + H + c).to(tl.float32)
        tl.store(Y + r * H + c, (ch * cv).to(tl.bfloat16))
        if WRITE:
            # one row: the state shifts by one input (every program owns its channels' columns)
            for j in tl.static_range(KT - 2):
                tl.store(STATE + j * H + c, tl.load(STATE + (j + 1) * H + c))
            tl.store(STATE + (KT - 2) * H + c, bx)

    @triton.jit
    def _qk_prep(QKV, s_qkv, POS, SLOT, QW, KW, COS, SIN, Q, KC, VC, s_kh, s_kn, s_vh, s_vn, eps,
                 HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr):
        r = tl.program_id(0)
        hh = tl.program_id(1)
        HALF: tl.constexpr = D // 2
        o = tl.arange(0, HALF)
        pos = tl.load(POS + r)
        is_q = hh < HQ
        off = tl.where(is_q, hh * D, HQ * D + (hh - HQ) * D)
        x1 = tl.load(QKV + r * s_qkv + off + o).to(tl.float32)
        x2 = tl.load(QKV + r * s_qkv + off + HALF + o).to(tl.float32)
        var = (tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / D
        rs = tl.rsqrt(var + eps)
        w1 = tl.where(is_q, tl.load(QW + o), tl.load(KW + o)).to(tl.float32)
        w2 = tl.where(is_q, tl.load(QW + HALF + o), tl.load(KW + HALF + o)).to(tl.float32)
        n1 = ((x1 * rs).to(tl.bfloat16).to(tl.float32) * w1).to(tl.bfloat16).to(tl.float32)
        n2 = ((x2 * rs).to(tl.bfloat16).to(tl.float32) * w2).to(tl.bfloat16).to(tl.float32)
        c1 = tl.load(COS + pos * D + o).to(tl.float32)
        c2 = tl.load(COS + pos * D + HALF + o).to(tl.float32)
        s1 = tl.load(SIN + pos * D + o).to(tl.float32)
        s2 = tl.load(SIN + pos * D + HALF + o).to(tl.float32)
        o1 = ((n1 * c1).to(tl.bfloat16).to(tl.float32) + (-n2 * s1).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        o2 = ((n2 * c2).to(tl.bfloat16).to(tl.float32) + (n1 * s2).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        if hh < HQ:
            tl.store(Q + (r * HQ + hh) * D + o, o1)
            tl.store(Q + (r * HQ + hh) * D + HALF + o, o2)
        else:
            kvh = hh - HQ
            slot = tl.load(SLOT + r)
            tl.store(KC + kvh * s_kh + slot * s_kn + o, o1)
            tl.store(KC + kvh * s_kh + slot * s_kn + HALF + o, o2)
            voff = HQ * D + HK * D + kvh * D
            tl.store(VC + kvh * s_vh + slot * s_vn + o, tl.load(QKV + r * s_qkv + voff + o))
            tl.store(VC + kvh * s_vh + slot * s_vn + HALF + o, tl.load(QKV + r * s_qkv + voff + HALF + o))

    @triton.jit
    def _attn_split(Q, K, V, CTL, ANCT, PM, PL, PACC, s_kh, s_kn, s_vh, s_vn,
                    SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, CHUNK_: tl.constexpr, NA: tl.constexpr, NS: tl.constexpr,
                    DOT32_: tl.constexpr):
        # CTL[t] = (visible keys n, block start L): key c < L is cache slot c, c >= L the row's
        # ancestor at depth c - L (ANCT[t, c - L], a cache slot)
        t = tl.program_id(0)
        kvh = tl.program_id(1)
        sp = tl.program_id(2)
        n = tl.load(CTL + t * 2)
        L = tl.load(CTL + t * 2 + 1)
        rows = tl.arange(0, BM)
        rmask = rows < REP
        qh = kvh * REP + rows
        od = tl.arange(0, D)
        q = tl.load(Q + (t * HQ + qh[:, None]) * D + od[None, :], mask=rmask[:, None], other=0.0)
        c0 = sp * CHUNK_
        c1 = tl.minimum(c0 + CHUNK_, n)
        m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for n0 in range(c0, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            tail = cols >= L
            a = tl.load(ANCT + t * NA + (cols - L), mask=cmask & tail, other=0)
            slot = tl.where(tail, a, cols)
            k = tl.load(K + kvh * s_kh + slot[:, None] * s_kn + od[None, :], mask=cmask[:, None], other=0.0)
            if DOT32_:
                s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * SCALE
            else:
                s = tl.dot(q, tl.trans(k)) * SCALE
            s = tl.where(cmask[None, :] & rmask[:, None], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + kvh * s_vh + slot[:, None] * s_vn + od[None, :], mask=cmask[:, None], other=0.0)
            if DOT32_:
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32))
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        base = ((t * HK + kvh) * NS + sp) * BM + rows
        tl.store(PM + base, m_i)
        tl.store(PL + base, l_i)
        tl.store(PACC + base[:, None] * D + od[None, :], acc)

    @triton.jit
    def _attn_combine(PM, PL, PACC, OUT, REP: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                      BM: tl.constexpr, NS: tl.constexpr):
        t = tl.program_id(0)
        qh = tl.program_id(1)
        kvh = qh // REP
        r = qh % REP
        od = tl.arange(0, D)
        b0 = (t * HK + kvh) * NS * BM + r
        mx = tl.load(PM + b0)
        for s in range(1, NS):
            mx = tl.maximum(mx, tl.load(PM + b0 + s * BM))
        mx = tl.where(mx == -float("inf"), 0.0, mx)
        num = tl.zeros([D], dtype=tl.float32)
        den = 0.0
        for s in range(0, NS):
            w = tl.exp(tl.load(PM + b0 + s * BM) - mx)
            num = num + tl.load(PACC + (b0 + s * BM) * D + od) * w
            den = den + tl.load(PL + b0 + s * BM) * w
        o = num / tl.where(den == 0.0, 1.0, den)
        tl.store(OUT + (t * HQ + qh) * D + od, o.to(tl.bfloat16))

    @triton.jit
    def _router(LOG, s_log, BIAS, IDS, WTS, scale, E: tl.constexpr, KK: tl.constexpr, BIASED: tl.constexpr):
        r = tl.program_id(0)
        o = tl.arange(0, E)
        lb = tl.load(LOG + r * s_log + o).to(tl.bfloat16).to(tl.float32)
        s = (1.0 / (1.0 + tl.exp(-lb))).to(tl.bfloat16).to(tl.float32)
        if BIASED:
            sc = s + tl.load(BIAS + o)
        else:
            sc = s
        tot = 0.0
        for j in tl.static_range(KK):
            mx = tl.max(sc, 0)
            i = tl.min(tl.where(sc == mx, o, E), 0)          # the lower id on a tie
            tl.store(IDS + r * KK + j, i)
            wj = tl.sum(tl.where(o == i, s, 0.0), 0)
            tl.store(WTS + r * KK + j, wj)
            tot = tot + wj
            sc = tl.where(o == i, -float("inf"), sc)
        den = (tot.to(tl.bfloat16).to(tl.float32) + 1e-6).to(tl.bfloat16).to(tl.float32)
        for j in tl.static_range(KK):
            wj = tl.load(WTS + r * KK + j)
            w = (wj / den).to(tl.bfloat16).to(tl.float32)
            tl.store(WTS + r * KK + j, (w * scale).to(tl.bfloat16).to(tl.float32))

    @triton.jit
    def _linear_rows(X, W, Y, M, N, K, s_x, s_w, s_y, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     F32: tl.constexpr, DOT32_: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * BM + tl.arange(0, BM)
        cols = pn * BN + tl.arange(0, BN)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            x = tl.load(X + rows[:, None] * s_x + ks[None, :], mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
            w = tl.load(W + cols[:, None] * s_w + ks[None, :], mask=(cols[:, None] < N) & (ks[None, :] < K), other=0.0)
            if DOT32_:
                acc += tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)))
            else:
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & (cols[None, :] < N))

    @triton.jit
    def _moe_gu(X, s_x, IDS, W13, A, S: tl.constexpr, H: tl.constexpr, I: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr):
        p = tl.program_id(0)
        nb = tl.program_id(1)
        row = p // S
        e = tl.load(IDS + p).to(tl.int64)
        n = nb * BN + tl.arange(0, BN)
        g = tl.zeros([BN], dtype=tl.float32)
        u = tl.zeros([BN], dtype=tl.float32)
        wb = W13 + e * (2 * I * H)
        for k0 in range(0, H, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(X + row * s_x + k).to(tl.float32)
            g += tl.sum(tl.load(wb + n[:, None] * H + k[None, :]).to(tl.float32) * x[None, :], 1)
            u += tl.sum(tl.load(wb + (I + n[:, None]) * H + k[None, :]).to(tl.float32) * x[None, :], 1)
        gb = g.to(tl.bfloat16).to(tl.float32)
        sl = (gb / (1.0 + tl.exp(-gb))).to(tl.bfloat16).to(tl.float32)
        tl.store(A + p * I + n, (sl * u.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16))

    @triton.jit
    def _moe_down(A, IDS, W2, YD, S: tl.constexpr, H: tl.constexpr, I: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr):
        p = tl.program_id(0)
        nb = tl.program_id(1)
        e = tl.load(IDS + p).to(tl.int64)
        n = nb * BN + tl.arange(0, BN)
        acc = tl.zeros([BN], dtype=tl.float32)
        wb = W2 + e * (H * I)
        for k0 in range(0, I, BK):
            k = k0 + tl.arange(0, BK)
            a = tl.load(A + p * I + k).to(tl.float32)
            acc += tl.sum(tl.load(wb + n[:, None] * I + k[None, :]).to(tl.float32) * a[None, :], 1)
        tl.store(YD + p * H + n, acc)


# ------------------------------------------------------------------------------ the wrappers
def add_rms(h: torch.Tensor, d: torch.Tensor | None, w: torch.Tensor, eps: float, h_out: torch.Tensor | None = None,
            n_out: torch.Tensor | None = None):
    """(h + d, norm(h + d)) as bf16 [M, N] (d None: (h, norm(h)))."""
    M, N = h.shape
    hn = h if d is None else (torch.empty_like(h) if h_out is None else h_out)
    nn = torch.empty_like(h) if n_out is None else n_out
    _add_rms[(M,)](h, d if d is not None else h, hn, nn, w, eps, N=N, BLOCK=triton.next_power_of_2(N),
                   ADD=d is not None, num_warps=4)
    return hn, nn


def add_rms_reference(h, d, w, eps):
    if d is not None:
        h = (h.float() + d.float()).to(torch.bfloat16)
    hf = h.float()
    n = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
    return h, (n.float() * w.float()).to(torch.bfloat16)


def conv_rows(bcx: torch.Tensor, state: torch.Tensor, wc: torch.Tensor, src: torch.Tensor, rec: torch.Tensor,
              y: torch.Tensor | None = None, write: bool = False) -> torch.Tensor:
    """bcx [M, 3 H] bf16 (B | C | x), state [KT - 1, H] bf16 (oldest first), wc [H, KT] bf16, src int32
    [M, KT - 1] (`window_src`), rec [M, H] bf16 (written: each row's bx) -> y [M, H] bf16."""
    M = bcx.shape[0]
    H, KT = wc.shape
    if y is None:
        y = torch.empty(M, H, dtype=torch.bfloat16, device=bcx.device)
    BLOCK = 256
    assert H % BLOCK == 0 and bcx.stride(1) == 1 and (not write or M == 1)
    _conv_rows[(M, H // BLOCK)](bcx, bcx.stride(0), state, wc, src, rec, y, H=H, KT=KT, BLOCK=BLOCK, WRITE=write,
                                num_warps=2)
    return y


def window_src(parents: list[int], taps: int) -> list[list[int]]:
    """For every row of a block, where its inputs at distance taps - 1 .. 1 live: the row of its
    ancestor at that distance (>= 0), or -1 - the state column ([.., taps - 1] columns, the last the
    newest) when the walk leaves the block."""
    S = taps - 1
    out = []
    for i in range(len(parents)):
        anc = [i]
        while parents[anc[-1]] >= 0 and len(anc) <= S:
            anc.append(parents[anc[-1]])
        row = []
        for d in range(S, 0, -1):
            row.append(anc[d] if d < len(anc) else -1 - (S - (d - len(anc) + 1)))
        out.append(row)
    return out


def conv_rows_reference(bcx, state, wc, src):
    H = wc.shape[0]
    B, C, X = bcx[:, :H].float(), bcx[:, H:2 * H].float(), bcx[:, 2 * H:].float()
    bx = (B * X).to(torch.bfloat16)
    out = []
    for r in range(bcx.shape[0]):
        acc = torch.zeros(H, dtype=torch.float32, device=bcx.device)
        for j, s in enumerate(src[r].tolist()):
            v = bx[s] if s >= 0 else state[-1 - s]
            acc = acc + v.float() * wc[:, j].float()
        acc = acc + bx[r].float() * wc[:, -1].float()
        out.append((C[r] * acc.to(torch.bfloat16).float()).to(torch.bfloat16))
    return torch.stack(out), bx


def qk_prep(qkv: torch.Tensor, pos: torch.Tensor, slot: torch.Tensor, qw, kw, cos, sin, kc, vc, hq: int, hk: int,
            eps: float, q: torch.Tensor | None = None) -> torch.Tensor:
    """qkv [M, (Hq + 2 Hk) D] bf16, pos / slot int64 [M], cos / sin bf16 [max_len, D], kc / vc
    [Hk, cap, D] -> q [M, Hq, D] bf16 (normed, rotated); k (normed, rotated) and v into the caches."""
    M = qkv.shape[0]
    D = kc.shape[2]
    if q is None:
        q = torch.empty(M, hq, D, dtype=torch.bfloat16, device=qkv.device)
    _qk_prep[(M, hq + hk)](qkv, qkv.stride(0), pos, slot, qw, kw, cos, sin, q, kc, vc, kc.stride(0), kc.stride(1),
                           vc.stride(0), vc.stride(1), eps, HQ=hq, HK=hk, D=D, num_warps=1)
    return q


def qk_prep_reference(qkv, pos, qw, kw, cos, sin, hq: int, hk: int, D: int, eps: float):
    """(q [M, Hq, D], k [M, Hk, D], v [M, Hk, D]) with the reference's roundings."""
    M = qkv.shape[0]

    def norm(x, w):
        xf = x.float()
        n = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
        return (n.float() * w.float()).to(torch.bfloat16)

    def rot(x):
        c, s = cos[pos][:, None].float(), sin[pos][:, None].float()
        h = D // 2
        rh = torch.cat([-x[..., h:], x[..., :h]], -1).float()
        return ((x.float() * c).to(torch.bfloat16).float() + (rh * s).to(torch.bfloat16).float()).to(torch.bfloat16)
    q = qkv[:, :hq * D].view(M, hq, D)
    k = qkv[:, hq * D:(hq + hk) * D].view(M, hk, D)
    v = qkv[:, (hq + hk) * D:].view(M, hk, D)
    return rot(norm(q, qw)), rot(norm(k, kw)), v


def attn_rows(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, ctl: torch.Tensor, anc: torch.Tensor, ns: int,
              bufs: dict | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """q [M, Hq, D] bf16, kc / vc [Hk, cap, D] bf16, ctl int32 [M, 2] (visible keys, block start),
    anc int32 [M, ANC] (cache slots of the row's ancestors by depth), `ns` CHUNK programs (the class,
    `class_of`) -> o [M, Hq, D] bf16."""
    M, HQ, D = q.shape
    HK = kc.shape[0]
    REP = HQ // HK
    BM = 16
    assert REP <= BM and D == kc.shape[2]
    b = bufs if bufs is not None else {}
    n = M * HK * ns * BM
    pm = b.get("pm")
    if pm is None or pm.shape[0] < n:
        # `bufs["rows"]`: the row count the partials are sized for, shared by calls of fewer rows (written
        # whole before they are read); a buffer is never replaced under a graph that captured it
        if pm is not None and b.get("rows"):
            raise ValueError(f"{M} rows in partials sized for {b['rows']}")
        cap = max(M, int(b.get("rows") or M)) * HK * ns * BM
        pm = torch.empty(cap, dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty(cap, D, dtype=torch.float32, device=q.device)
        if bufs is not None:
            bufs.update(pm=pm, pl=pl, pacc=pacc)
    else:
        pl, pacc = b["pl"], b["pacc"]
    pm, pl, pacc = pm[:n], pl[:n], pacc[:n]
    if out is None:
        out = torch.empty(M, HQ, D, dtype=torch.bfloat16, device=q.device)
    _attn_split[(M, HK, ns)](q, kc, vc, ctl, anc, pm, pl, pacc, kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1),
                             SCALE=D ** -0.5, REP=REP, D=D, HQ=HQ, HK=HK, BM=BM, BN=32, CHUNK_=CHUNK,
                             NA=anc.shape[1], NS=ns, DOT32_=DOT32, num_warps=4, num_stages=2)
    _attn_combine[(M, HQ)](pm, pl, pacc, out, REP=REP, D=D, HQ=HQ, HK=HK, BM=BM, NS=ns, num_warps=1)
    return out


def attn_rows_reference(q, kc, vc, ctl, anc):
    M, HQ, D = q.shape
    rep = HQ // kc.shape[0]
    out = []
    for t in range(M):
        n, L = (int(x) for x in ctl[t, :2])
        pos = torch.arange(n, device=q.device)
        slot = torch.where(pos >= L, anc[t].long()[(pos - L).clamp(min=0, max=anc.shape[1] - 1)], pos)
        k = kc[:, slot].float().repeat_interleave(rep, 0)
        v = vc[:, slot].float().repeat_interleave(rep, 0)
        s = torch.einsum("hd,hnd->hn", q[t].float(), k) * D ** -0.5
        out.append(torch.einsum("hn,hnd->hd", torch.softmax(s, -1), v))
    return torch.stack(out)


def router(logits: torch.Tensor, bias: torch.Tensor | None, k: int, scale: float, ids=None, wts=None):
    """logits fp32 [M, E] -> ids int32 [M, k], wts fp32 [M, k] (bf16 values)."""
    M, E = logits.shape
    if ids is None:
        ids = torch.empty(M, k, dtype=torch.int32, device=logits.device)
        wts = torch.empty(M, k, dtype=torch.float32, device=logits.device)
    _router[(M,)](logits, logits.stride(0), bias if bias is not None else logits, ids, wts, float(scale), E=E, KK=k,
                  BIASED=bias is not None, num_warps=1)
    return ids, wts


def router_reference(logits, bias, k: int, scale: float):
    """The reference's routing (ref.py `route`) from fp32 logits rounded to bf16, in bf16."""
    lb = logits.to(torch.bfloat16)
    s = lb.sigmoid()
    sc = s.float() + bias if bias is not None else s.float()
    order = torch.sort(sc, dim=-1, descending=True, stable=True).indices[:, :k]
    w = torch.gather(s, 1, order)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
    return order.to(torch.int32), (w * scale).float()


def linear_rows(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
    """x [M, K] bf16 @ w^T (bf16 [N, K]) -> [M, N] bf16 (fp32 with `f32`)."""
    M, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    BN = 16 if N <= 8192 else 64
    _linear_rows[(triton.cdiv(M, 16), triton.cdiv(N, BN))](x, w, out, M, N, K, x.stride(0), w.stride(0), out.stride(0),
                                                          BM=16, BN=BN, BK=128, F32=f32, DOT32_=DOT32, num_warps=4)
    return out


def moe_bf16(x: torch.Tensor, ids: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Every (row, slot) pair through its BF16 expert: x [M, H], ids int32 [M, S], w13 [E, 2 I, H], w2
    [E, H, I] -> yd [M S, H] fp32 (unweighted, the pairs in their own order)."""
    M, S = ids.shape
    E, I2, H = w13.shape
    I = I2 // 2
    P = M * S
    a = torch.empty(P, I, dtype=torch.bfloat16, device=x.device)
    yd = torch.empty(P, H, dtype=torch.float32, device=x.device)
    flat = ids.reshape(-1)
    _moe_gu[(P, I // 32)](x, x.stride(0), flat, w13, a, S=S, H=H, I=I, BN=32, BK=128, num_warps=4)
    _moe_down[(P, H // 32)](a, flat, w2, yd, S=S, H=H, I=I, BN=32, BK=128, num_warps=4)
    return yd


def moe_bf16_reference(x, ids, w13, w2):
    M, S = ids.shape
    I = w13.shape[1] // 2
    out = []
    for p, e in enumerate(ids.reshape(-1).tolist()):
        xr = x[p // S].float()
        g = (w13[e, :I].float() @ xr).to(torch.bfloat16).float()
        u = (w13[e, I:].float() @ xr).to(torch.bfloat16).float()
        a = (F.silu(g).to(torch.bfloat16).float() * u).to(torch.bfloat16).float()
        out.append(w2[e].float() @ a)
    return torch.stack(out)


def combine_reference(yd: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """`fn_kernels.moe_combine`'s sum: a row's slots in ascending expert id, w * y in fp32, one rounding."""
    M, S = ids.shape
    out = torch.zeros(M, yd.shape[1], dtype=torch.float32, device=yd.device)
    for m in range(M):
        es = ids[m].tolist()
        for s in sorted(range(S), key=lambda s: es[s]):
            out[m] = out[m] + float(wts[m, s]) * yd[m * S + s]
    return out.to(torch.bfloat16)
