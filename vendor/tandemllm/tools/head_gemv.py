"""The vocabulary projection as an own kernel.

`lm_head` is [248,320 x 5,120] in bf16 -- 2.54 GB, 13 % of an NVFP4 step's read set and the single
largest tensor the engine touches. It is read once per verify pass and, when a drafter has no head
of its own, once per drafted token as well. The step breakdown charges it 14.5 ms, which is
176 GB/s: a third below what a plain streaming read reaches on this board.

At M = 1 this is a matrix-vector product and the library's general path is not built for it. The
kernel here is: one program per tile of `BN` vocabulary rows, each looping over the 5,120 reduction
elements in `BK` chunks with the activation held in registers. 248,320 / 64 = 3,880 programs
against 48 SMs, every one of them reading `BN x 5,120 x 2` contiguous bytes, which is what a
bandwidth-bound kernel wants.

SPLIT-K. Splitting the reduction is the standard answer when a GEMV cannot fill the machine, and
here it cannot be the answer -- 3,880 tiles already fill it many times over, and splitting would add
an atomic pass over a 993 KB output for nothing. What `--split-k` measures instead is the opposite
question: whether *fewer, longer* tiles beat more short ones. Both are swept by `bench()` and the
winner is a measurement, not an argument.

The logits are computed in fp32 and returned in fp32, which is what `argmax` and the drafter's
`topk` consume. Nothing downstream reads them in bf16.
"""

from __future__ import annotations

import os
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in __import__("sys").path:
    # run as a script from tools/: the repo root, appended (lowest priority), for engine.settings
    __import__("sys").path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _head_gemv(X, W, Y, M, N, K,
                   s_wn, s_wk,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rn = pid * BN + tl.arange(0, BN)
        mn = rn < N
        rm = tl.arange(0, BM)
        mm = rm < M
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < K
            x = tl.load(X + rm[:, None] * K + rk[None, :],
                        mask=mm[:, None] & mk[None, :], other=0.0).to(tl.float32)
            w = tl.load(W + rn[:, None] * s_wn + rk[None, :] * s_wk,
                        mask=mn[:, None] & mk[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
        tl.store(Y + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & mn[None, :])


if HAVE_TRITON:

    @triton.jit
    def _head_gemv_fp8(X, W, S, Y, M, N, K, s_wn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """One row against an e4m3 head with one fp32 scale per vocabulary row.

        The scale is a per-output-channel constant, so it multiplies the finished fp32 accumulator
        once instead of every decoded weight -- which is both cheaper and closer to the bf16 head,
        since the only rounding left is e4m3's own three mantissa bits.
        """
        pid = tl.program_id(0)
        rn = pid * BN + tl.arange(0, BN)
        rm = tl.arange(0, BM)
        mm = rm < M
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            x = tl.load(X + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
            w = tl.load(W + rn[:, None] * s_wn + rk[None, :]).to(tl.float32)
            acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
        s = tl.load(S + rn).to(tl.float32)
        tl.store(Y + rm[:, None] * N + rn[None, :], acc * s[None, :], mask=mm[:, None])

    @triton.jit
    def _head_gemm_fp8(X, W, S, Y, M, N, K, s_xm, s_wn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """The same head at a verified block's row count, where the tensor cores are the right tool.

        The lesson of 11:26 in the ledger: a GEMV routed at eight rows does eight times the scalar
        arithmetic and made the verify pass 2.5x slower than the library GEMM it replaced. An fp8
        head has no library GEMM to fall back on, so the block path is written as a GEMM.
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm = rm < M
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            x = tl.load(X + rm[:, None] * s_xm + kk[None, :], mask=mm[:, None], other=0.0)
            w = tl.load(W + rn[:, None] * s_wn + kk[None, :])
            acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
        s = tl.load(S + rn).to(tl.float32)
        tl.store(Y + rm[:, None] * N + rn[None, :], acc * s[None, :], mask=mm[:, None])


class FP8Head:
    """`lm_head` as e4m3 codes plus one fp32 scale per vocabulary row.

    2.54 GB of bf16 becomes 1.27 GB of codes and 0.99 MB of scales. The head is read once by the
    verify pass and once more by any drafter that takes `topk` over the full vocabulary, so at a
    block of eight this is about 11 ms off a 161 ms block -- the largest single item left in the
    step after the MLPs.

    Per ROW, not per 128x128 block. The rows of a vocabulary projection are the tokens, and their
    norms differ by more than a block scale shared across 128 neighbouring token ids can follow;
    a row scale costs 4 bytes per 5,120 weights and removes the question.
    """

    __slots__ = ("w", "s", "N", "K", "_bf16")

    def __init__(self, codes: torch.Tensor, scale: torch.Tensor):
        assert codes.dtype == torch.float8_e4m3fn and codes.dim() == 2, (codes.dtype, codes.shape)
        assert scale.dtype == torch.float32 and scale.shape == (codes.shape[0],), scale.shape
        self.w = codes.contiguous()
        self.s = scale.contiguous()
        self.N, self.K = codes.shape
        self._bf16 = None

    @property
    def shape(self):
        return (self.N, self.K)

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() * 4

    # so the byte accounting in `Weights.decode_step_bytes` and `DFlash2Drafter.draft_bytes`
    # reads this head the same way it reads a plain tensor
    def numel(self) -> int:
        return self.w.numel()

    def element_size(self) -> int:
        return 1

    def dequant(self, rows: int = 8192) -> torch.Tensor:
        y = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.w.device)
        for r0 in range(0, self.N, rows):
            r1 = min(r0 + rows, self.N)
            y[r0:r1] = (self.w[r0:r1].float() * self.s[r0:r1, None]).to(torch.bfloat16)
        return y

    def bf16_cached(self) -> torch.Tensor:
        if self._bf16 is None:
            self._bf16 = self.dequant()
        return self._bf16

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """The Linear interface: fp32 logits [..., N], as `engine.model.head_logits` gives."""
        return head_matmul_fp8(x, self).view(*x.shape[:-1], self.N)


# the block GEMM's launch knobs that keep every logit's K order -- the N tile, the warps, the
# pipeline stages -- as module attributes (QWEN38_HEAD_GEMM="bn:warps:stages"; the default is what
# shipped), so an in-process block A/B can flip them; the verify and draft graphs key on them.
_hg = _S.get("HEAD_GEMM")
HEAD_BN, HEAD_WARPS, HEAD_STAGES = (tuple(int(v) for v in _hg.split(":")) if _hg else (64, 4, 3))


def head_matmul_fp8(x: torch.Tensor, head: FP8Head, *, bn: int | None = None,
                    bk: int | None = None) -> torch.Tensor:
    """`x @ head^T` for an e4m3 head with per-row scales. Returns fp32 [M, N].

    One row takes the GEMV; anything above it takes the GEMM. Both return fp32, which is what
    `argmax` and a drafter's `topk` consume, and which is a strictly better floor than the bf16
    logits the library path produces -- the ledger's 10:07 entry is about two logits landing one
    bf16 ulp apart.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = flat.shape
    N = head.N
    assert K == head.K, (K, head.K)
    y = torch.empty(M, N, dtype=torch.float32, device=flat.device)
    if M == 1:
        bn_, bk_ = (bn or 32), (bk or 256)
        _head_gemv_fp8[(triton.cdiv(N, bn_),)](flat, head.w, head.s, y, M, N, K,
                                               head.w.stride(0), BM=1, BN=bn_, BK=bk_,
                                               num_warps=4)
        return y
    bn_, bk_ = (bn or HEAD_BN), (bk or 128)
    bm = 16 if M <= 16 else (32 if M <= 32 else 64)
    _head_gemm_fp8[(triton.cdiv(N, bn_), triton.cdiv(M, bm))](
        flat, head.w, head.s, y, M, N, K, flat.stride(0), head.w.stride(0),
        BM=bm, BN=bn_, BK=bk_, num_warps=HEAD_WARPS, num_stages=HEAD_STAGES)
    return y


def quantize_head_fp8(w: torch.Tensor, *, ratios=(1.0,), rows: int = 8192) -> FP8Head:
    """Bf16 [N, K] -> e4m3 codes with one fp32 scale per row.

    `scale = amax(row) * ratio / 448`. With more than one ratio the scale is searched per row on
    plain squared error: a smaller scale represents the bulk of a row more finely and clips its
    outlier, and which side wins is a property of the row, not of an argument.
    """
    N, K = w.shape
    codes = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty(N, dtype=torch.float32, device=w.device)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        ref = w[r0:r1].float()
        amax = ref.abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        best_err = None
        best_s = None
        for ratio in ratios:
            s = (amax * ratio / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
            q = (ref / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
            err = (ref - q).pow(2).sum(dim=1, keepdim=True)
            if best_err is None:
                best_err, best_s = err, s
            else:
                take = err < best_err
                best_err = torch.where(take, err, best_err)
                best_s = torch.where(take, s, best_s)
            del s, q, err
        codes[r0:r1] = (ref / best_s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        scale[r0:r1] = best_s[:, 0]
        del ref, amax, best_err, best_s
    return FP8Head(codes, scale)


def head_to_nvfp4(head, *, rows: int = 8192):
    """The vocabulary head as an NVFP4 block, 0.72 GB instead of the e4m3 head's 1.27.

    For the DRAFTER only. The block drafter reads the whole head once a block to turn its
    rows into candidates, and nothing it proposes reaches the output unverified, so a coarser head
    there can cost acceptance and never correctness; the target's own head is untouched. Quantised
    a row block at a time from whatever the engine holds (the e4m3 head or a bf16 one), with the
    per-tensor scale taken over the whole head first so every block shares it -- which makes the
    result identical to quantising the dequantised head in one piece.
    """
    from tools.nvfp4_linear import E2M1_MAX, E4M3_MAX, NVFP4Block, quantize_to_nvfp4

    def chunk(r0: int, r1: int) -> torch.Tensor:
        if isinstance(head, FP8Head):
            return head.w[r0:r1].float() * head.s[r0:r1, None]
        return head[r0:r1].float()

    N, K = head.shape
    amax = max(chunk(r0, min(r0 + rows, N)).abs().max() for r0 in range(0, N, rows)).float()
    scale_2 = float((amax / (E2M1_MAX * E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny))
    codes, scales = [], []
    for r0 in range(0, N, rows):
        b = quantize_to_nvfp4(chunk(r0, min(r0 + rows, N)), scale_2=scale_2)
        codes.append(b.w)
        scales.append(b.s)
    return NVFP4Block(torch.cat(codes), torch.cat(scales), scale_2)


def head_matmul(x: torch.Tensor, w: torch.Tensor, *, bn: int = 32, bk: int = 256,
                bm: int = 1) -> torch.Tensor:
    """`x @ w.T` for a bf16 head [N, K] and a short activation [M, K]. Returns fp32 [M, N]."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    flat = x.reshape(-1, x.shape[-1])
    M, K = flat.shape
    N = w.shape[0]
    if M > bm:
        raise ValueError(f"head_matmul was built for M <= {bm}, got {M}")
    flat = flat.contiguous()
    y = torch.empty(M, N, dtype=torch.float32, device=x.device)
    _head_gemv[(triton.cdiv(N, bn),)](flat, w, y, M, N, K, w.stride(0), w.stride(1),
                                      BM=bm, BN=bn, BK=bk, num_warps=4)
    return y


# --------------------------------------------------------------------------- the gate

def check(N: int = 248320, K: int = 5120, device: str = "cuda", **kw) -> dict:
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, K, device=device, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w).float()
    got = head_matmul(x, w, **kw)
    d = (ref - got).abs()
    return {"absmax": float(d.max()), "scale": float(ref.abs().max()),
            "argmax_same": int(ref.argmax()) == int(got.argmax()),
            "top8_same": torch.equal(ref.topk(8).indices, got.topk(8).indices)}


def bench(N: int = 248320, K: int = 5120, device: str = "cuda", iters: int = 50,
          sweep=((64, 128), (64, 256), (128, 128), (128, 256), (256, 256), (32, 256))) -> list[str]:
    import time
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, K, device=device, dtype=torch.bfloat16)
    nbytes = N * K * 2
    out = []

    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    ms = timed(lambda: torch.nn.functional.linear(x, w))
    out.append(f"torch F.linear        {ms:7.2f} ms  {nbytes / (ms * 1e-3) / 1e9:6.1f} GB/s")
    for bn, bk in sweep:
        try:
            ms = timed(lambda: head_matmul(x, w, bn=bn, bk=bk))
        except Exception as exc:                               # pragma: no cover
            out.append(f"triton bn={bn:3d} bk={bk:3d}  failed: {exc}")
            continue
        out.append(f"triton bn={bn:3d} bk={bk:3d}   {ms:7.2f} ms  "
                   f"{nbytes / (ms * 1e-3) / 1e9:6.1f} GB/s")
    return out


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(check())
    for line in bench():
        print(line)
