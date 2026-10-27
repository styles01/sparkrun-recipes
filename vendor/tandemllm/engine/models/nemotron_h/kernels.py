"""NemotronH's row kernels: Triton, one program a row (or a row's head, group or column block).

The decode step and every row of a verify block run through these with the same instruction
sequence: no reduction crosses rows and no tile shape depends on the row count, so a row of an M-row
call is the one-row call's row bit for bit, the property speculation needs (docs/kernels.md, "Verify
twins"). With the reference's roundings (engine/models/nemotron_h/ref.py):

  add_rms        h = bf16(h + d) (optional), n = bf16(w * h * rsqrt(mean(h^2) + eps)) in fp32
  conv_rows      Mamba2's causal depthwise conv over xBC: the row's window (the state's last inputs,
                 or its ancestors' xBC from the block), fp32 taps oldest first after the bias, silu,
                 bf16; one row (decode) shifts the state in place
  ssm_rows       the selective state update walked from the committed state S0 along the row's
                 ancestors (`anc`, up to ANC): dt = softplus(dt_raw + dt_bias), S = S exp(dt A) +
                 dt x B, all fp32, then y = bf16(S C + D x). `write`: the walk's last S replaces S0
                 (a decode step; a commit runs it over the kept path). A block of rows walks once a
                 head: a row whose parent is the row before it goes on from that row's S, any other
                 row walks its path from S0 again -- every row's S is the same products in the same
                 order as its own walk, and S0 is read once, not once a row
  gated_norm     y silu(z) in fp32, RMS over each group, times w in fp32, one rounding
  router         fp32 logits: sigmoid, the correction bias for the choice, top k (ties to the lower
                 id), the chosen sigmoids / (sum + 1e-20) x scale, fp32
  moe_relu2      every (row, slot) pair through its expert: a = bf16(relu(bf16(up x))^2), down to a
                 bf16-rounded fp32 row; `combine` adds a row's pairs in ascending expert id, w y in
                 fp32, one rounding

Attention reuses the LFM2 row kernels (`attn_rows`, fixed chunks of keys; NoPE needs no prep kernel)
and the projections their `linear_rows` (bf16 16-row tiles on the tensor cores, K in order). Each
kernel has a torch twin (`*_reference`) for the CPU tests (TRITON_INTERPRET=1 runs the kernels
themselves on the CPU).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.models.lfm2_moe import kernels as LK
from engine.models.lfm2_moe.kernels import (ANC, CHUNK, DOT32, HAVE_TRITON, attn_rows, attn_rows_reference,  # noqa: F401
                                            class_of, classes, linear_rows, window_src)

if HAVE_TRITON:
    import triton
    import triton.language as tl

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
        w = tl.load(W + o, mask=m, other=0.0).to(tl.float32)
        tl.store(N_OUT + r * N + o, (w * (hf * tl.rsqrt(var + eps))).to(tl.bfloat16), mask=m)

    @triton.jit
    def _conv_rows(P, s_p, OFF, STATE, WC, BC, SRC, Y, CD: tl.constexpr, KT: tl.constexpr, BLOCK: tl.constexpr,
                   WRITE: tl.constexpr):
        # SRC[r, j] for j < KT - 1: where the input at distance KT - 1 - j lives (a row of the block
        # >= 0, or -1 - a state column); the last tap is the row itself
        r = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        cm = c < CD
        acc = tl.load(BC + c, mask=cm, other=0.0).to(tl.float32)
        for j in tl.static_range(KT - 1):
            src = tl.load(SRC + r * (KT - 1) + j)
            row = tl.maximum(src, 0)
            col = tl.maximum(-1 - src, 0)
            vb = tl.load(P + row * s_p + OFF + c, mask=cm, other=0.0)
            vs = tl.load(STATE + col * CD + c, mask=cm, other=0.0)
            v = tl.where(src >= 0, vb, vs).to(tl.float32)
            acc += v * tl.load(WC + c * KT + j, mask=cm, other=0.0).to(tl.float32)
        xr = tl.load(P + r * s_p + OFF + c, mask=cm, other=0.0)
        acc += xr.to(tl.float32) * tl.load(WC + c * KT + (KT - 1), mask=cm, other=0.0).to(tl.float32)
        tl.store(Y + r * CD + c, (acc / (1.0 + tl.exp(-acc))).to(tl.bfloat16), mask=cm)
        if WRITE:
            # one row: the state shifts by one input (every program owns its channels' columns)
            for j in tl.static_range(KT - 2):
                tl.store(STATE + j * CD + c, tl.load(STATE + (j + 1) * CD + c, mask=cm, other=0.0), mask=cm)
            tl.store(STATE + (KT - 2) * CD + c, xr, mask=cm)

    @triton.jit
    def _ssm_rows(P, s_p, DT_OFF, XBC, s_x, ANCT, NANC, S0, A, DTB, DP, Y, s_y,
                  I: tl.constexpr, GN: tl.constexpr, HD: tl.constexpr, N: tl.constexpr, REP: tl.constexpr,
                  NA: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr, WRITE: tl.constexpr):
        r = tl.program_id(0)
        h = tl.program_id(1)
        db = tl.program_id(2)
        d = db * BD + tl.arange(0, BD)
        n = tl.arange(0, BN)
        nm = n < N
        g = h // REP
        sp = S0 + h * HD * N + d[:, None] * N + n[None, :]
        S = tl.load(sp, mask=nm[None, :], other=0.0)
        a_h = tl.load(A + h)
        b_h = tl.load(DTB + h)
        cnt = tl.load(NANC + r)
        for t in range(0, cnt):
            a = tl.load(ANCT + r * NA + t)
            dt = tl.load(P + a * s_p + DT_OFF + h).to(tl.float32) + b_h
            dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
            dA = tl.exp(dt * a_h)
            x = tl.load(XBC + a * s_x + h * HD + d).to(tl.float32)
            B = tl.load(XBC + a * s_x + I + g * N + n, mask=nm, other=0.0).to(tl.float32)
            S = S * dA + (dt * x)[:, None] * B[None, :]
        C = tl.load(XBC + r * s_x + I + GN + g * N + n, mask=nm, other=0.0).to(tl.float32)
        xr = tl.load(XBC + r * s_x + h * HD + d).to(tl.float32)
        y = tl.sum(S * C[None, :], 1) + tl.load(DP + h) * xr
        tl.store(Y + r * s_y + h * HD + d, y.to(tl.bfloat16))
        if WRITE:
            tl.store(sp, S, mask=nm[None, :])

    @triton.jit
    def _ssm_walk(P, s_p, DT_OFF, XBC, s_x, ANCT, NANC, S0, A, DTB, DP, Y, s_y, M,
                  I: tl.constexpr, GN: tl.constexpr, HD: tl.constexpr, N: tl.constexpr, REP: tl.constexpr,
                  NA: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
        h = tl.program_id(0)
        db = tl.program_id(1)
        d = db * BD + tl.arange(0, BD)
        n = tl.arange(0, BN)
        nm = n < N
        g = h // REP
        S0v = tl.load(S0 + h * HD * N + d[:, None] * N + n[None, :], mask=nm[None, :], other=0.0)
        a_h = tl.load(A + h)
        b_h = tl.load(DTB + h)
        dp = tl.load(DP + h)
        S = S0v
        for r in range(0, M):
            cnt = tl.load(NANC + r)
            par = tl.load(ANCT + r * NA + tl.maximum(cnt - 2, 0))
            on = (cnt >= 2) & (par == r - 1)              # the walk so far is this row's path less itself
            S = tl.where(on, S, S0v)
            for t in range(tl.where(on, cnt - 1, 0), cnt):
                a = tl.load(ANCT + r * NA + t)
                dt = tl.load(P + a * s_p + DT_OFF + h).to(tl.float32) + b_h
                dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
                dA = tl.exp(dt * a_h)
                x = tl.load(XBC + a * s_x + h * HD + d).to(tl.float32)
                B = tl.load(XBC + a * s_x + I + g * N + n, mask=nm, other=0.0).to(tl.float32)
                S = S * dA + (dt * x)[:, None] * B[None, :]
            C = tl.load(XBC + r * s_x + I + GN + g * N + n, mask=nm, other=0.0).to(tl.float32)
            xr = tl.load(XBC + r * s_x + h * HD + d).to(tl.float32)
            y = tl.sum(S * C[None, :], 1) + dp * xr
            tl.store(Y + r * s_y + h * HD + d, y.to(tl.bfloat16))

    @triton.jit
    def _gated_norm(Yv, s_y, Z, s_z, W, OUT, eps, I: tl.constexpr, GS: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        g = tl.program_id(1)
        o = tl.arange(0, BLOCK)
        m = o < GS
        c = g * GS + o
        y = tl.load(Yv + r * s_y + c, mask=m, other=0.0).to(tl.float32)
        z = tl.load(Z + r * s_z + c, mask=m, other=0.0).to(tl.float32)
        x = y * (z / (1.0 + tl.exp(-z)))
        var = tl.sum(x * x, 0) / GS
        w = tl.load(W + c, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + r * I + c, (x * tl.rsqrt(var + eps) * w).to(tl.bfloat16), mask=m)

    @triton.jit
    def _router(LOG, s_log, BIAS, IDS, WTS, scale, E: tl.constexpr, BE: tl.constexpr, KK: tl.constexpr):
        r = tl.program_id(0)
        o = tl.arange(0, BE)
        m = o < E
        lg = tl.load(LOG + r * s_log + o, mask=m, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-lg))
        sc = tl.where(m, s + tl.load(BIAS + o, mask=m, other=0.0), -float("inf"))
        tot = 0.0
        for j in tl.static_range(KK):
            mx = tl.max(sc, 0)
            i = tl.min(tl.where(sc == mx, o, BE), 0)          # the lower id on a tie
            tl.store(IDS + r * KK + j, i)
            wj = tl.sum(tl.where(o == i, s, 0.0), 0)
            tl.store(WTS + r * KK + j, wj)
            tot = tot + wj
            sc = tl.where(o == i, -float("inf"), sc)
        den = tot + 1e-20
        for j in tl.static_range(KK):
            wj = tl.load(WTS + r * KK + j)
            tl.store(WTS + r * KK + j, (wj / den) * scale)

    @triton.jit
    def _moe_up(X, s_x, IDS, WU, A, S: tl.constexpr, KIN: tl.constexpr, I: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr):
        p = tl.program_id(0)
        nb = tl.program_id(1)
        row = p // S
        e = tl.load(IDS + p).to(tl.int64)
        n = nb * BN + tl.arange(0, BN)
        u = tl.zeros([BN], dtype=tl.float32)
        wb = WU + e * (I * KIN)
        for k0 in range(0, KIN, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(X + row * s_x + k).to(tl.float32)
            u += tl.sum(tl.load(wb + n[:, None] * KIN + k[None, :]).to(tl.float32) * x[None, :], 1)
        ub = tl.maximum(u.to(tl.bfloat16).to(tl.float32), 0.0)
        tl.store(A + p * I + n, (ub * ub).to(tl.bfloat16))

    @triton.jit
    def _moe_down(A, IDS, WD, YD, KIN: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        p = tl.program_id(0)
        nb = tl.program_id(1)
        e = tl.load(IDS + p).to(tl.int64)
        n = nb * BN + tl.arange(0, BN)
        acc = tl.zeros([BN], dtype=tl.float32)
        wb = WD + e * (KIN * I)
        for k0 in range(0, I, BK):
            k = k0 + tl.arange(0, BK)
            a = tl.load(A + p * I + k).to(tl.float32)
            acc += tl.sum(tl.load(wb + n[:, None] * I + k[None, :]).to(tl.float32) * a[None, :], 1)
        tl.store(YD + p * KIN + n, acc.to(tl.bfloat16).to(tl.float32))

    @triton.jit
    def _combine(YD, IDS, WTS, OUT, H: tl.constexpr, S: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr):
        r = tl.program_id(0)
        nb = tl.program_id(1)
        n = nb * BN + tl.arange(0, BN)
        o = tl.arange(0, BS)
        ids = tl.load(IDS + r * S + o, mask=o < S, other=2 ** 30)
        acc = tl.zeros([BN], dtype=tl.float32)
        for _ in tl.static_range(S):
            mn = tl.min(ids, 0)
            j = tl.min(tl.where(ids == mn, o, BS), 0)         # the slot of the lowest remaining id
            w = tl.load(WTS + r * S + j)
            acc += w * tl.load(YD + (r * S + j) * H + n)
            ids = tl.where(o == j, 2 ** 30, ids)
        tl.store(OUT + r * H + n, acc.to(tl.bfloat16))


# ------------------------------------------------------------------------------ the wrappers
def add_rms(h: torch.Tensor, d: torch.Tensor | None, w: torch.Tensor, eps: float, h_out=None, n_out=None):
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
    return h, (w.float() * (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps))).to(torch.bfloat16)


def conv_rows(proj: torch.Tensor, off: int, state: torch.Tensor, wc: torch.Tensor, bc: torch.Tensor, src: torch.Tensor,
              y: torch.Tensor | None = None, write: bool = False) -> torch.Tensor:
    """proj [M, *] bf16 (xBC at columns off .. off + conv_dim), state [KT - 1, conv_dim] bf16 (oldest
    first), wc [conv_dim, KT] bf16, bc [conv_dim], src int32 [M, KT - 1] (`window_src`) -> xbc [M,
    conv_dim] bf16 (after silu)."""
    M = proj.shape[0]
    CD, KT = wc.shape
    if y is None:
        y = torch.empty(M, CD, dtype=torch.bfloat16, device=proj.device)
    BLOCK = 256
    assert proj.stride(1) == 1 and (not write or M == 1)
    _conv_rows[(M, triton.cdiv(CD, BLOCK))](proj, proj.stride(0), off, state, wc, bc, src, y, CD=CD, KT=KT, BLOCK=BLOCK,
                                            WRITE=write, num_warps=2)
    return y


def conv_rows_reference(proj, off: int, state, wc, bc, src):
    CD = wc.shape[0]
    xr = proj[:, off:off + CD]
    out = []
    for r in range(proj.shape[0]):
        acc = bc.float().clone()
        for j, s in enumerate(src[r].tolist()):
            v = xr[s] if s >= 0 else state[-1 - s]
            acc = acc + v.float() * wc[:, j].float()
        acc = acc + xr[r].float() * wc[:, -1].float()
        out.append(F.silu(acc).to(torch.bfloat16))
    return torch.stack(out)


def ssm_rows(proj: torch.Tensor, dt_off: int, xbc: torch.Tensor, anc: torch.Tensor, nanc: torch.Tensor,
             S0: torch.Tensor, A: torch.Tensor, dtb: torch.Tensor, Dp: torch.Tensor, n_groups: int,
             y: torch.Tensor | None = None, write: bool = False) -> torch.Tensor:
    """proj [M, *] bf16 (dt_raw at columns dt_off ..), xbc [M, conv_dim] bf16 (x | B | C after the
    conv), anc int32 [M, NA] (the rows of each row's path, root first), nanc int32 [M] (its length),
    S0 [heads, hd, N] fp32, A = -exp(A_log), dt_bias, D fp32 [heads] -> y [M, heads hd] bf16, M the
    rows of `anc` (a commit: one row whose ancestors are the kept path, its y discarded)."""
    M = anc.shape[0]
    Hh, HD, N = S0.shape
    I = Hh * HD
    GN = n_groups * N
    if y is None:
        y = torch.empty(M, I, dtype=torch.bfloat16, device=xbc.device)
    BD = 16 if HD % 16 == 0 else HD
    if M > 1 and not write:
        _ssm_walk[(Hh, HD // BD)](proj, proj.stride(0), dt_off, xbc, xbc.stride(0), anc, nanc, S0, A, dtb, Dp, y,
                                  y.stride(0), M, I=I, GN=GN, HD=HD, N=N, REP=Hh // n_groups, NA=anc.shape[1], BD=BD,
                                  BN=triton.next_power_of_2(N), num_warps=4)
        return y
    _ssm_rows[(M, Hh, HD // BD)](proj, proj.stride(0), dt_off, xbc, xbc.stride(0), anc, nanc, S0, A, dtb, Dp, y, y.stride(0),
                                 I=I, GN=GN, HD=HD, N=N, REP=Hh // n_groups, NA=anc.shape[1], BD=BD,
                                 BN=triton.next_power_of_2(N), WRITE=write, num_warps=4)
    return y


def ssm_rows_reference(proj, dt_off: int, xbc, anc, nanc, S0, A, dtb, Dp, n_groups: int):
    """(y [M, I] bf16, the last row's walked S)."""
    Hh, HD, N = S0.shape
    I = Hh * HD
    rep = Hh // n_groups
    x = xbc[:, :I].float().view(-1, Hh, HD)
    B = xbc[:, I:I + n_groups * N].float().view(-1, n_groups, N).repeat_interleave(rep, 1)
    C = xbc[:, I + n_groups * N:].float().view(-1, n_groups, N).repeat_interleave(rep, 1)
    dt = proj[:, dt_off:dt_off + Hh].float() + dtb
    dt = torch.where(dt <= 20.0, torch.log(1.0 + torch.exp(dt)), dt)
    ys = []
    for r in range(xbc.shape[0]):
        S = S0.clone()
        for t in range(int(nanc[r])):
            a = int(anc[r, t])
            S = S * torch.exp(dt[a] * A)[:, None, None] + (dt[a][:, None] * x[a])[:, :, None] * B[a][:, None, :]
        ys.append(((S * C[r][:, None, :]).sum(-1) + Dp[:, None] * x[r]).to(torch.bfloat16).reshape(-1))
    return torch.stack(ys), S


def gated_norm(y: torch.Tensor, z: torch.Tensor, w: torch.Tensor, n_groups: int, eps: float,
               out: torch.Tensor | None = None) -> torch.Tensor:
    """y [M, I] bf16, z [M, I] bf16 (a view of in_proj's rows), w [I] -> [M, I] bf16."""
    M, I = y.shape
    GS = I // n_groups
    if out is None:
        out = torch.empty(M, I, dtype=torch.bfloat16, device=y.device)
    _gated_norm[(M, n_groups)](y, y.stride(0), z, z.stride(0), w, out, eps, I=I, GS=GS,
                               BLOCK=triton.next_power_of_2(GS), num_warps=4)
    return out


def gated_norm_reference(y, z, w, n_groups: int, eps: float):
    x = y.float() * F.silu(z.float())
    g = x.view(x.shape[0], n_groups, -1)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)
    return (g.view_as(x) * w.float()).to(torch.bfloat16)


def router(logits: torch.Tensor, bias: torch.Tensor, k: int, scale: float, ids=None, wts=None):
    """logits fp32 [M, E] -> ids int32 [M, k], wts fp32 [M, k]."""
    M, E = logits.shape
    if ids is None:
        ids = torch.empty(M, k, dtype=torch.int32, device=logits.device)
        wts = torch.empty(M, k, dtype=torch.float32, device=logits.device)
    _router[(M,)](logits, logits.stride(0), bias, ids, wts, float(scale), E=E, BE=triton.next_power_of_2(E), KK=k,
                  num_warps=4)
    return ids, wts


def router_reference(logits, bias, k: int, scale: float):
    s = logits.float().sigmoid()
    order = torch.sort(s + bias.float(), dim=-1, descending=True, stable=True).indices[:, :k]
    w = torch.gather(s, 1, order)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return order.to(torch.int32), w * scale


def moe_relu2(x: torch.Tensor, ids: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor) -> torch.Tensor:
    """Every (row, slot) pair through its expert: x [M, Kin] bf16, ids int32 [M, S], wu [E, I, Kin], wd
    [E, Kin, I] -> yd [M S, Kin] fp32 (bf16 values, unweighted, the pairs in their own order)."""
    M, S = ids.shape
    E, I, KIN = wu.shape
    P = M * S
    a = torch.empty(P, I, dtype=torch.bfloat16, device=x.device)
    yd = torch.empty(P, KIN, dtype=torch.float32, device=x.device)
    flat = ids.reshape(-1)
    assert I % 32 == 0 and KIN % 32 == 0, (I, KIN)
    BK = next(b for b in (128, 64, 32) if KIN % b == 0)
    BKD = next(b for b in (128, 64, 32) if I % b == 0)
    _moe_up[(P, I // 32)](x, x.stride(0), flat, wu, a, S=S, KIN=KIN, I=I, BN=32, BK=BK, num_warps=4)
    _moe_down[(P, KIN // 32)](a, flat, wd, yd, KIN=KIN, I=I, BN=32, BK=BKD, num_warps=4)
    return yd


def moe_relu2_reference(x, ids, wu, wd):
    M, S = ids.shape
    out = []
    for p, e in enumerate(ids.reshape(-1).tolist()):
        u = (wu[e].float() @ x[p // S].float()).to(torch.bfloat16).float()
        a = F.relu(u).square().to(torch.bfloat16).float()
        out.append((wd[e].float() @ a).to(torch.bfloat16).float())
    return torch.stack(out)


def combine(yd: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """yd [M S, H] fp32, ids int32 / wts fp32 [M, S] -> [M, H] bf16: a row's pairs in ascending expert
    id, w y in fp32, one rounding."""
    M, S = ids.shape
    H = yd.shape[1]
    if out is None:
        out = torch.empty(M, H, dtype=torch.bfloat16, device=yd.device)
    _combine[(M, H // 32)](yd, ids, wts, out, H=H, S=S, BS=triton.next_power_of_2(S), BN=32, num_warps=1)
    return out


def combine_reference(yd, ids, wts):
    return LK.combine_reference(yd, ids, wts)
