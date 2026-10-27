"""Dense attention over a row's visible keys at any context length (ENG-383): Triton, a verify twin.

    o = softmax(q k^T / sqrt(D)) v     per query head of one row

The row's visible keys are virtual columns 0..n-1: column c < L (the block's start) is cache slot c,
column c >= L the row's ancestor at depth c - L (`anc`, a cache slot): the decode step is one row at
L = its position, a verify block's rows see the committed cache and their own path. Fixed chunks of
CHUNK keys, one program each (a query head group of one KV head), combined in chunk order; an empty
chunk adds exact zeros, so the number of programs -- the context class, `class_of` -- never changes a
row's bits, and a row's arithmetic is a function of its own query and visible keys only: a row of an
M-row call is the one-row call's row bit for bit. This is LFM2's row attention (engine/models/lfm2_moe/
kernels.py) with the control rows read at a stride (the shared row buffers keep four ints a row) and
its partials kept per class (a captured graph of one class keeps its buffers when another is captured).

`attn_rows_reference` is the torch twin for the CPU tests.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:          # noqa: BLE001  (a Mac: the reference only)
    HAVE_TRITON = False

#: keys a program reads (fixed: a row's bits do not depend on the class)
CHUNK = 512
#: the query heads of one KV head a program holds (Hq / Hk <= BM)
BM = 16
DOT32 = os.environ.get("TRITON_INTERPRET") == "1"


def classes(max_len: int) -> list[int]:
    """The attention classes: CHUNK programs a call launches, doubling from 4 (2,048 keys) to what
    covers `max_len`."""
    out, n = [], 4
    while True:
        out.append(n)
        if n * CHUNK >= max_len:
            return out
        n *= 2


def class_of(keys: int, max_len: int) -> int:
    """The smallest class whose programs cover `keys` visible keys."""
    for n in classes(max(max_len, keys)):
        if n * CHUNK >= keys:
            return n
    raise ValueError(f"{keys} keys")


if HAVE_TRITON:

    @triton.jit
    def _attn_split(Q, K, V, CTL, s_ctl, ANCT, s_anc, PM, PL, PACC, s_kh, s_kn, s_vh, s_vn,
                    SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                    BM_: tl.constexpr, BN: tl.constexpr, CHUNK_: tl.constexpr, NS: tl.constexpr,
                    DOT32_: tl.constexpr):
        t = tl.program_id(0)
        kvh = tl.program_id(1)
        sp = tl.program_id(2)
        n = tl.load(CTL + t * s_ctl)
        L = tl.load(CTL + t * s_ctl + 1)
        rows = tl.arange(0, BM_)
        rmask = rows < REP
        qh = kvh * REP + rows
        od = tl.arange(0, D)
        q = tl.load(Q + (t * HQ + qh[:, None]) * D + od[None, :], mask=rmask[:, None], other=0.0)
        c0 = sp * CHUNK_
        c1 = tl.minimum(c0 + CHUNK_, n)
        m_i = tl.full([BM_], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM_], dtype=tl.float32)
        acc = tl.zeros([BM_, D], dtype=tl.float32)
        for n0 in range(c0, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            tail = cols >= L
            a = tl.load(ANCT + t * s_anc + (cols - L), mask=cmask & tail, other=0)
            slot = tl.where(tail, a, cols).to(tl.int64)
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
        base = ((t * HK + kvh) * NS + sp) * BM_ + rows
        tl.store(PM + base, m_i)
        tl.store(PL + base, l_i)
        tl.store(PACC + base[:, None] * D + od[None, :], acc)

    @triton.jit
    def _attn_combine(PM, PL, PACC, OUT, REP: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                      BM_: tl.constexpr, NS: tl.constexpr):
        t = tl.program_id(0)
        qh = tl.program_id(1)
        kvh = qh // REP
        r = qh % REP
        od = tl.arange(0, D)
        b0 = (t * HK + kvh) * NS * BM_ + r
        mx = tl.load(PM + b0)
        for s in range(1, NS):
            mx = tl.maximum(mx, tl.load(PM + b0 + s * BM_))
        mx = tl.where(mx == -float("inf"), 0.0, mx)
        num = tl.zeros([D], dtype=tl.float32)
        den = 0.0
        for s in range(0, NS):
            w = tl.exp(tl.load(PM + b0 + s * BM_) - mx)
            num = num + tl.load(PACC + (b0 + s * BM_) * D + od) * w
            den = den + tl.load(PL + b0 + s * BM_) * w
        o = num / tl.where(den == 0.0, 1.0, den)
        tl.store(OUT + (t * HQ + qh) * D + od, o.to(tl.bfloat16))


def attn_rows(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, ctl: torch.Tensor, anc: torch.Tensor, ns: int,
              bufs: dict | None = None) -> torch.Tensor:
    """q [M, Hq, D] bf16, kc / vc [Hk, cap, D] bf16, ctl int32 [M, >= 2] (visible keys, block start),
    anc int32 [M, A] (the cache slots of the row's ancestors by depth), `ns` CHUNK programs (`class_of`)
    -> o [M, Hq, D] bf16. `bufs`: the call site's partials and output, per class (allocated once)."""
    M, HQ, D = q.shape
    HK = kc.shape[0]
    REP = HQ // HK
    assert REP <= BM and D == kc.shape[2] and ctl.stride(1) == 1 and anc.stride(1) == 1
    b = (bufs if bufs is not None else {}).setdefault(("attn", ns, M), {})
    if not b:
        n = M * HK * ns * BM
        b["pm"] = torch.empty(n, dtype=torch.float32, device=q.device)
        b["pl"] = torch.empty_like(b["pm"])
        b["pacc"] = torch.empty(n, D, dtype=torch.float32, device=q.device)
        b["out"] = torch.empty(M, HQ, D, dtype=torch.bfloat16, device=q.device)
    out = b["out"]
    _attn_split[(M, HK, ns)](q, kc, vc, ctl, ctl.stride(0), anc, anc.stride(0), b["pm"], b["pl"], b["pacc"],
                             kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1),
                             SCALE=D ** -0.5, REP=REP, D=D, HQ=HQ, HK=HK, BM_=BM, BN=32, CHUNK_=CHUNK, NS=ns,
                             DOT32_=DOT32, num_warps=4, num_stages=2)
    _attn_combine[(M, HQ)](b["pm"], b["pl"], b["pacc"], out, REP=REP, D=D, HQ=HQ, HK=HK, BM_=BM, NS=ns, num_warps=1)
    return out


def attn_rows_reference(q, kc, vc, ctl, anc):
    """The kernel's result in fp32 torch (`ctl[:, 0]` visible keys, `ctl[:, 1]` block start)."""
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
