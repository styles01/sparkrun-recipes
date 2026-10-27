"""A small bf16 projection with a fixed reduction order: the linear-attention gate inputs `a | b`.

Every GDN layer projects its input onto two 48-wide weights, `in_proj_a` and `in_proj_b`, in bf16.
They are the only library GEMMs left in a verify pass, and a library GEMM picks its algorithm from
heuristics that depend on the row count and on the workspace it is given -- which is not the same
inside a CUDA-graph capture as outside one. On 2026-09-23 (K4) a graphed verify moved chat's text
where the eager verify of the same block did not, and these two GEMMs are the suspects.

This kernel computes both in one launch over the concatenated [96, K] weight, one program per
(row block, 16 outputs), the K loop in a fixed order with an fp32 accumulator, rounded once to bf16
-- `F.linear`'s contract. The same kernel serves the verify, its graph and the single-token decode
step, so a row's gate inputs are the same bits whichever of the three computed them.
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
    def _small_linear(X, W, Y, M, N, K, s_xm, s_wn, s_ym,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mm, mn = rm < M, rn < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < K
            x = tl.load(X + rm[:, None] * s_xm + rk[None, :], mask=mm[:, None] & mk[None, :],
                        other=0.0)
            w = tl.load(W + rn[:, None] * s_wn + rk[None, :], mask=mn[:, None] & mk[None, :],
                        other=0.0)
            acc += tl.sum(x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :], axis=2)
        tl.store(Y + rm[:, None] * s_ym + rn[None, :], acc.to(tl.bfloat16),
                 mask=mm[:, None] & mn[None, :])


def small_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """X [M, K] bf16 @ w [N, K]^T bf16 -> [M, N] bf16, M small, the same bits for every M."""
    M, K = x.shape
    N = w.shape[0]
    x = x.contiguous()
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    # BM = 1: each row is its own program row, so a row's sum never depends on its neighbours
    _small_linear[(M, triton.cdiv(N, 16))](x, w, y, M, N, K, x.stride(0), w.stride(0),
                                           y.stride(0), BM=1, BN=16, BK=256, num_warps=4)
    return y


def check(K: int = 5120, N: int = 96) -> list[str]:
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    x = torch.randn(16, K, device="cuda", generator=g).to(torch.bfloat16)
    ref = (x.float() @ w.float().T)
    y = small_linear(x, w)
    d = (y.float() - ref).abs().max().item()
    lib = torch.nn.functional.linear(x, w)
    dl = (lib.float() - ref).abs().max().item()
    for m in range(1, 17):
        assert torch.equal(small_linear(x[:m], w), y[:m]), m
    return [f"small_linear: max|y - fp32| {d:.3e} (F.linear {dl:.3e}); rows 1..16 give the same "
            f"bits as the 16-row call"]


if __name__ == "__main__":
    print("\n".join(check()))
