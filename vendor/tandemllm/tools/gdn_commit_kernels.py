"""The commit of a speculative block, for all 48 linear-attention layers in one launch.

After a verify pass the engine knows which rows it keeps: a prefix of a chain, or a path through a
tree. The recurrent state after those rows is the rank-k identity the rollback already uses,

    S_path = exp(gc[last]) . S_entry + SUM_{r in path} exp(gc[last] - gc[r]) . k_r (x) u_r

with the normalised keys, the pseudo-values and the cumulative gate the verify pass wrote. In torch
that is, per layer, an exp, a gather, a scale, a transpose, a matmul, an add and a copy -- eight
kernels and four passes over a 3.1 MB state -- forty-eight times a block, after a full 151 MB copy
of the entry state that every layer then overwrites anyway. On new text almost every block is a
partial accept (0.3 % of wide blocks commit all sixteen rows), so
that is the price of nearly every block.

`_gdn_commit` reads each [128, BV] tile of the entry state once, adds the path's rank-k update in
registers, and writes the tile once: 302 MB for the whole state, one launch. The convolution tails
are four torch ops over stacked tensors instead of two per layer.

It reads the ENTRY state and writes the live one, which may be the same tensor (a tree verify does
not advance the state) or a different one (a chain verify, whose pass wrote its final state into
the spare buffer -- see `GDNState.swap`). A program reads its whole tile before it writes it, so
the in-place case is safe.

Arithmetic: the same identity in fp32, summed row by row in path order, where torch's matmul sums
the same products in cuBLAS's order. `check()` compares the two; the losslessness gate decides.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _gdn_commit(S_IN, S_OUT, KK, U, GC, ROWS, P,
                    s_l, s_h, s_k, s_v, k_l, k_h, k_t, u_l, u_h, u_t, g_l, g_h,
                    DK: tl.constexpr, BV: tl.constexpr):
        l = tl.program_id(0)
        h = tl.program_id(1)
        vb = tl.program_id(2)
        ok = tl.arange(0, DK)
        ov = vb * BV + tl.arange(0, BV)
        tile = l * s_l + h * s_h + ok[:, None] * s_k + ov[None, :] * s_v
        gbase = GC + l * g_l + h * g_h
        last = tl.load(ROWS + P - 1)
        gl = tl.load(gbase + last)
        s = tl.load(S_IN + tile) * tl.exp(gl)
        for j in range(P):
            r = tl.load(ROWS + j)
            w = tl.exp(gl - tl.load(gbase + r))
            k = tl.load(KK + l * k_l + h * k_h + r * k_t + ok)
            u = tl.load(U + l * u_l + h * u_h + r * u_t + ov)
            s += (w * k)[:, None] * u[None, :]
        tl.store(S_OUT + tile, s)


def commit_reference(S_in, kk, u, gc, rows):
    """The torch rank-k of `Qwen38Engine.rollback_to` / `commit_tree`, over stacked layers.

    S_in [L, H, Dk, Dv]; kk [L, H, T, Dk]; u [L, H, T, Dv]; gc [L, H, T]; rows a list.
    """
    idx = torch.as_tensor(rows, dtype=torch.long, device=S_in.device)
    gl = gc[:, :, rows[-1]]                                    # [L, H]
    w = torch.exp(gl[..., None] - gc[:, :, idx])               # [L, H, P]
    kw = (kk[:, :, idx] * w[..., None]).transpose(-1, -2)      # [L, H, Dk, P]
    return gl[..., None, None].exp() * S_in + kw @ u[:, :, idx]


_ROWS: dict = {}


def _rows_tensor(rows: list[int], device) -> torch.Tensor:
    """The path as a device tensor, cached by value: a chain's `range(keep)` recurs every block,
    and a fresh host-to-device copy of a pageable list is a synchronisation."""
    key = (tuple(rows), str(device))
    t = _ROWS.get(key)
    if t is None:
        if len(_ROWS) > 4096:
            _ROWS.clear()
        t = _ROWS[key] = torch.tensor(rows, dtype=torch.int32, device=device)
    return t


def fused_commit(S_in: torch.Tensor, S_out: torch.Tensor, kk: torch.Tensor, u: torch.Tensor,
                 gc: torch.Tensor, rows: list[int], *, bv: int = 32) -> None:
    """S_out[l] = rank-k(S_in[l]) along `rows`, every layer at once.

    S_in / S_out [L, 1, H, Dk, Dv] fp32 (the GDNState layout); kk [L, H, T, Dk], u [L, H, T, Dv],
    gc [L, H, T] fp32 contiguous.
    """
    L, _, H, Dk, Dv = S_in.shape
    if not (HAVE_TRITON and S_in.is_cuda):
        S_out.copy_(commit_reference(S_in[:, 0], kk, u, gc, rows)[:, None])
        return
    Si, So = S_in[:, 0], S_out[:, 0]
    assert Si.stride() == So.stride(), (Si.stride(), So.stride())
    _gdn_commit[(L, H, Dv // bv)](
        Si, So, kk, u, gc, _rows_tensor(rows, S_in.device), len(rows),
        Si.stride(0), Si.stride(1), Si.stride(2), Si.stride(3),
        kk.stride(0), kk.stride(1), kk.stride(2), u.stride(0), u.stride(1), u.stride(2),
        gc.stride(0), gc.stride(1),
        DK=Dk, BV=bv, num_warps=4)


def conv_commit(conv_entry: torch.Tensor, conv_out: torch.Tensor, raw: torch.Tensor,
                rows: list[int]) -> None:
    """The convolution tails of every layer: the last W-1 raw inputs along `rows`.

    conv_entry / conv_out [L, 1, C, W-1] bf16; raw [L, C, T] the pre-convolution projections.
    """
    w1 = conv_entry.shape[-1]
    idx = _rows_tensor(rows, raw.device).long()
    joined = torch.cat([conv_entry[:, 0], raw.index_select(2, idx)], dim=-1)   # [L, C, W-1+P]
    conv_out[:, 0].copy_(joined[:, :, -w1:])


def check(L: int = 48, H: int = 48, Dk: int = 128, Dv: int = 128, T: int = 16,
          seed: int = 0) -> list[str]:
    """The kernel against the torch rank-k, on chains and on tree paths, in place and not."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    S = torch.randn(L, 1, H, Dk, Dv, device="cuda", generator=g) * 0.1
    kk = torch.nn.functional.normalize(torch.randn(L, H, T, Dk, device="cuda", generator=g), dim=-1)
    u = torch.randn(L, H, T, Dv, device="cuda", generator=g) * 0.05
    gc = (-torch.rand(L, H, T, device="cuda", generator=g) * 0.3).cumsum(-1)
    out = []
    worst = 0.0
    for rows in ([0], [0, 1], list(range(8)), list(range(16)), [0, 2, 5, 9], [0, 1, 3, 7, 12, 15]):
        ref = commit_reference(S[:, 0], kk, u, gc, rows)
        o = torch.empty_like(S)
        fused_commit(S, o, kk, u, gc, rows)
        d = (o[:, 0] - ref).abs().max().item()
        rel = d / ref.abs().max().item()
        worst = max(worst, rel)
        inplace = S.clone()
        fused_commit(inplace, inplace, kk, u, gc, rows)
        assert torch.equal(inplace, o), f"in place differs from out of place on {rows}"
        out.append(f"rows {rows}: max|fused - torch| {d:.3e} (rel {rel:.2e}), in place == out of place")
    assert worst < 1e-5, worst
    out.append(f"worst relative {worst:.2e}")
    return out


def bench(L: int = 48, H: int = 48, Dk: int = 128, Dv: int = 128, T: int = 16, reps: int = 20):
    S = torch.randn(L, 1, H, Dk, Dv, device="cuda") * 0.1
    kk = torch.randn(L, H, T, Dk, device="cuda")
    u = torch.randn(L, H, T, Dv, device="cuda")
    gc = torch.randn(L, H, T, device="cuda").cumsum(-1)
    rows = list(range(4))
    o = torch.empty_like(S)

    def torch_path():
        o.copy_(S)                                  # the full copy rollback_to starts with
        for i in range(L):
            o[i].copy_(commit_reference(S[i:i + 1, 0], kk[i:i + 1], u[i:i + 1], gc[i:i + 1],
                                        rows)[:, None][0])

    res = []
    for name, fn in (("torch, per layer", torch_path),
                     ("fused", lambda: fused_commit(S, o, kk, u, gc, rows))):
        fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(reps):
            fn()
        t1.record()
        t1.synchronize()
        res.append(f"{name:>18}: {t0.elapsed_time(t1) / reps:.3f} ms")
    return res


if __name__ == "__main__":
    for line in check() + bench():
        print(line, flush=True)
