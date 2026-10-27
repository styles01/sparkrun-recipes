"""Y = x @ W^T with W stored in the NVFP4 layout, read as packed bytes and decoded in registers.

NVFP4 is three things stacked, and all three have to be present or the numbers are wrong:

  * `weight`          uint8 [N, K/2]  -- two e2m1 codes per byte, low nibble = the even K element
  * `weight_scale`    fp8 e4m3 [N, K/16] -- one scale per 16 consecutive K elements of a row
  * `weight_scale_2`  fp32 scalar -- one per tensor, the scale the e4m3 scale table itself carries

and the weight is `e2m1(code) * float(weight_scale[n, k // 16]) * weight_scale_2`. The block of 16
is what makes four bits usable: a 128x128 fp8 block covers 16,384 weights with one scale, an NVFP4
group covers 16, so the grid adapts to a row's local dynamic range instead of a whole tile's.

Bytes per weight: 0.5 for the code plus 1/16 for the e4m3 scale = **0.5625**, against the stored
fp8 format's 1 + 1/16384 = 1.0001. The MLP of one layer goes from 267.4 MB to 150.4 MB.

The kernel is the decode-side one: M is one row at greedy decode and up to a verified block at
speculation, so the shape is a GEMV in disguise and the whole cost is reading the packed bytes.
Three things make that fast on this board and none of them are optional:

  * 64-byte row tiles. A 16-byte-wide row tile caps around 130 GB/s here; 64 bytes reaches the
    board's practical copy bandwidth. One tile is 128 logical K = four 16-byte chunks, split in
    registers with no shared-memory round trip.
  * `cvt.rn.f16x2.e2m1x2`, the hardware FP4 decoder, one byte to two fp16 in one instruction.
  * the group scale multiplied onto the *decoded weight* rather than onto the accumulator. With a
    group of 16 and an even/odd split the accumulator would need a 8-wide `tl.dot`, which the
    tensor cores do not do. Scaling the weight is exact: an e2m1 value has two significand bits and
    an e4m3 scale has three, so the product needs six and fp16 has ten.

`BLOCK_N = 32` at decode M rather than 128: at M = 1 the grid is one M-block wide, so a wide N tile
leaves most of the board idle and the DRAM latency is never hidden.
"""

from __future__ import annotations

import os
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in __import__("sys").path:
    # run as a script from tools/: the repo root, appended (lowest priority), for engine.settings
    __import__("sys").path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUP = 16          # NVFP4 scale group, along K
E2M1_MAX = 6.0      # the largest representable e2m1 magnitude
E4M3_MAX = 448.0    # the largest representable e4m3 magnitude

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

# One 32-bit register holds four packed bytes = eight FP4 values. `cvt.rn.f16x2.e2m1x2` turns one
# byte into {f16(high nibble), f16(low nibble)}; the prmt shuffles regroup the four results into
# the even elements (low nibbles) and the odd elements (high nibbles), four fp16 each. The .b8
# source operand is mandatory -- ptxas rejects a .b16/.b32 source.
_FP4_DECODE_ASM = tl.constexpr("""
{
.reg .b8 b0, b1, b2, b3;
.reg .b32 t0, t1, t2, t3;
mov.b32 {b0, b1, b2, b3}, $4;
cvt.rn.f16x2.e2m1x2 t0, b0;
cvt.rn.f16x2.e2m1x2 t1, b1;
cvt.rn.f16x2.e2m1x2 t2, b2;
cvt.rn.f16x2.e2m1x2 t3, b3;
prmt.b32 $0, t0, t1, 0x5410;
prmt.b32 $1, t2, t3, 0x5410;
prmt.b32 $2, t0, t1, 0x7632;
prmt.b32 $3, t2, t3, 0x7632;
}
""")


@triton.jit
def _fp4_decode(packed):
    """Uint8 tile [..., 16] -> (even fp16 tile, odd fp16 tile), same shape, values on the e2m1 grid."""
    return tl.inline_asm_elementwise(
        _FP4_DECODE_ASM, "=r,=r,=r,=r,r", [packed], dtype=(tl.float16, tl.float16),
        is_pure=True, pack=4)


@triton.jit
def _split4(t, BN: tl.constexpr, W: tl.constexpr):
    """[BN, 4*W] register tile -> four [BN, W] column chunks, in order, without shared memory."""
    t = tl.reshape(t, [BN, 2, 2, W])
    t = tl.permute(t, [0, 3, 2, 1])
    a, b = tl.split(t)
    c0, c1 = tl.split(a)
    c2, c3 = tl.split(b)
    return c0, c1, c2, c3


@triton.jit
def _split8(s, BN: tl.constexpr):
    """[BN, 8] register tile -> eight [BN] columns, in order."""
    t = tl.permute(tl.reshape(s, [BN, 2, 2, 2]), [0, 3, 2, 1])
    a, b = tl.split(t)          # columns 0..3 / 4..7
    aa, ab = tl.split(a)
    ba, bb = tl.split(b)
    s0, s1 = tl.split(aa)
    s2, s3 = tl.split(ab)
    s4, s5 = tl.split(ba)
    s6, s7 = tl.split(bb)
    return s0, s1, s2, s3, s4, s5, s6, s7


@triton.jit
def _chunk_dot(x_base, xk, mask_m, packed, s_lo, s_hi, half, BN: tl.constexpr):
    """One 32-wide K chunk = two NVFP4 groups.

    `packed` is [BN, 16] bytes holding logical K 0..31 of the chunk. The even elements (logical 2j)
    and the odd ones (2j+1) both take their scale from group `j // 8`, so one 16-wide column mask
    selects between the chunk's two group scales for both halves.
    """
    we, wo = _fp4_decode(packed)
    sc = tl.where(half, s_lo[:, None], s_hi[:, None]).to(tl.float16)
    xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
    xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
    p = tl.dot(xe, tl.trans(we * sc))
    p = tl.dot(xo, tl.trans(wo * sc), acc=p)
    return p


@triton.jit
def _nvfp4_dequant_kernel(W, S, Y, N, K, s2, stride_wn, stride_sn, stride_yn,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Unpack a whole projection back to bf16, once.

    At decode M the weight read is the whole cost and unpacking into memory would double it. At
    prefill M it is the opposite: the weight is read once and multiplied by hundreds of rows, so
    the cost is arithmetic, and the arithmetic of a hand-written W4A16 GEMM is a long way below
    what the library's bf16 GEMM reaches. This kernel buys that trade: it writes 2K bytes a row to
    save the difference.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    nm = rn < N
    km = rk < K
    byte = tl.load(W + rn[:, None] * stride_wn + (rk[None, :] // 2),
                   mask=nm[:, None] & km[None, :], other=0)
    nib = tl.where(rk[None, :] % 2 == 0, byte & 0x0F, byte >> 4)
    idx = (nib & 7).to(tl.int32)
    # the e2m1 grid, 0 0.5 1 1.5 2 3 4 6, without a table lookup
    hi = tl.where(idx == 4, 2.0, tl.where(idx == 5, 3.0, tl.where(idx == 6, 4.0, 6.0)))
    val = tl.where(idx < 4, idx.to(tl.float32) * 0.5, hi)
    val = tl.where(nib >= 8, -val, val)
    sc = tl.load(S + rn[:, None] * stride_sn + (rk[None, :] // 16),
                 mask=nm[:, None] & km[None, :], other=0.0).to(tl.float32)
    tl.store(Y + rn[:, None] * stride_yn + rk[None, :], (val * sc * s2).to(tl.bfloat16),
             mask=nm[:, None] & km[None, :])


@triton.jit
def _nvfp4_linear_kernel(X, W, S, Y, M, N, KQ, s2,
                         stride_xm, stride_wn, stride_sn, stride_yk, stride_ym,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         SPLIT_K: tl.constexpr):
    """Y[M, N] = (X[M, K] @ W[N, K]^T) * s2; W packed [N, K/2] uint8, S [N, K/16] fp8. KQ = K/128.

    With SPLIT_K > 1 each program walks every SPLIT_K-th 128-wide K step and writes its own fp32
    partial plane, which is reduced outside. A tall-and-thin shape like `down_proj` -- N = 5120,
    K = 17408 -- has too few output tiles to fill the board on the N axis alone and its K loop is
    long enough that the DRAM latency is never hidden; splitting K is what fills it.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N
    rn_ = tl.where(n_mask, rn, 0)          # the weight tile is loaded unmasked, 64 bytes per row
    m_mask = (rm < M)[:, None]
    rm_ = tl.where(rm < M, rm, 0)
    xk = 2 * tl.arange(0, 16)[None, :]
    half = (tl.arange(0, 16) < 8)[None, :]
    x_base = X + rm_[:, None] * stride_xm
    w_tile = W + rn_[:, None] * stride_wn + tl.arange(0, 64)[None, :]
    s_tile = S + rn_[:, None] * stride_sn + tl.arange(0, 8)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for q in range(pid_k, KQ, SPLIT_K):
        packed = tl.load(w_tile + q * 64)
        c0, c1, c2, c3 = _split4(packed, BLOCK_N, 16)
        s = tl.load(s_tile + q * 8).to(tl.float16)
        s0, s1, s2_, s3, s4, s5, s6, s7 = _split8(s, BLOCK_N)
        xb = x_base + q * 128
        acc += _chunk_dot(xb, xk, m_mask, c0, s0, s1, half, BLOCK_N)
        acc += _chunk_dot(xb + 32, xk, m_mask, c1, s2_, s3, half, BLOCK_N)
        acc += _chunk_dot(xb + 64, xk, m_mask, c2, s4, s5, half, BLOCK_N)
        acc += _chunk_dot(xb + 96, xk, m_mask, c3, s6, s7, half, BLOCK_N)
    if SPLIT_K == 1:
        tl.store(Y + rm[:, None] * stride_ym + rn[None, :], (acc * s2).to(tl.bfloat16),
                 mask=m_mask & n_mask[None, :])
    else:
        # a private fp32 plane per K slice, reduced outside: an atomic would make the reduction
        # order depend on the scheduler, and this engine's correctness gate is bit-level equality
        # between a speculative and a non-speculative greedy run.
        tl.store(Y + pid_k * stride_yk + rm[:, None] * stride_ym + rn[None, :], acc * s2,
                 mask=m_mask & n_mask[None, :])


class NVFP4Block:
    """A projection weight in the NVFP4 layout: e2m1 codes, an e4m3 group-of-16 scale table, one
    fp32 per-tensor scale. Nothing is dequantised into memory; the decode step is these bytes."""

    __slots__ = ("w", "s", "s2", "N", "K", "_bf16", "_srun")      # _srun: the scale runs

    def __init__(self, codes: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor | float):
        assert codes.dtype == torch.uint8 and codes.dim() == 2, (codes.dtype, codes.shape)
        assert scale.dtype == torch.float8_e4m3fn, scale.dtype
        self.w = codes.contiguous()
        self.s = scale.contiguous()
        self.s2 = float(scale_2)
        self.N = codes.shape[0]
        self.K = codes.shape[1] * 2
        assert self.K % 128 == 0, self.K
        assert tuple(self.s.shape) == (self.N, self.K // GROUP), (self.s.shape, self.N, self.K)
        self._bf16 = None

    @property
    def shape(self):
        return (self.N, self.K)

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel()

    def dequant_fast(self) -> torch.Tensor:
        """The same weight as `dequant`, in one kernel. Used only above the row count where
        unpacking pays for itself; `tools/nvfp4_linear.py` says where that is."""
        y = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.w.device)
        _nvfp4_dequant_kernel[(triton.cdiv(self.N, 64), triton.cdiv(self.K, 256))](
            self.w, self.s, y, self.N, self.K, self.s2,
            self.w.stride(0), self.s.stride(0), y.stride(0),
            BLOCK_N=64, BLOCK_K=256, num_warps=4, num_stages=2)
        return y

    def dequant(self, rows: int = 2048) -> torch.Tensor:
        """The bf16 weight, for the reference path and for tests. Plain torch, row block at a time:
        it is never on the decode path, and a 17408x5120 fp32 intermediate is 356 MB."""
        grid = FP4_GRID.to(self.w.device)
        y = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.w.device)
        for r0 in range(0, self.N, rows):
            r1 = min(r0 + rows, self.N)
            b = self.w[r0:r1]
            lo = (b & 0x0F).long()
            hi = (b >> 4).long()
            vals = torch.empty(r1 - r0, self.K, dtype=torch.float32, device=self.w.device)
            vals[:, 0::2] = grid[lo & 7] * torch.where(lo >= 8, -1.0, 1.0)
            vals[:, 1::2] = grid[hi & 7] * torch.where(hi >= 8, -1.0, 1.0)
            s = self.s[r0:r1].float().repeat_interleave(GROUP, 1) * self.s2
            y[r0:r1] = (vals * s).to(torch.bfloat16)
            del b, lo, hi, vals, s
        return y

    def bf16_cached(self) -> torch.Tensor:
        if self._bf16 is None:
            self._bf16 = self.dequant()
        return self._bf16

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """The Linear interface: `x[..., K] @ W^T` -> [..., N], every served dispatch."""
        return nvfp4_matmul(x.reshape(-1, x.shape[-1]), self).view(*x.shape[:-1], self.N)


def use_skinny(M: int) -> bool:
    from tools.nvfp4_skinny import use_skinny as _u
    return _u(M)


def use_v2(M: int) -> bool:
    """V2 owns a row count only when it is switched on for it; see tools/nvfp4_linear_v2.py."""
    from tools.nvfp4_linear_v2 import use_v2 as _u
    return _u(M)


def nvfp4_matmul(x: torch.Tensor, w: NVFP4Block, *, block_m: int | None = None,
                 block_n: int | None = None, split_k: int | None = None,
                 num_warps: int | None = None, num_stages: int | None = None,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """Y[M, N] = x[M, K] @ W[N, K]^T, W in the NVFP4 layout. x is bf16."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == w.K, (x.shape, w.shape)
    M = x.shape[0]
    x = x.contiguous()
    if x.device.type != "cuda":
        return torch.nn.functional.linear(x, w.dequant())
    if block_n is None and split_k is None and prefill_v2(M):
        from tools.nvfp4_linear_v2 import nvfp4_matmul_v2
        return nvfp4_matmul_v2(x, w, out=out, **PREFILL_V2_TILE)
    if DEQUANT_FROM and M >= DEQUANT_FROM:
        # Unpack, multiply, drop. The unpacked copy lives for one call: at 17,408 x 5,120 it is
        # 178 MB, and the peak footprint is one projection rather than a bf16 model.
        y = torch.nn.functional.linear(x, w.dequant_fast())
        return y if out is None else out.copy_(y)
    if block_n is None and split_k is None and use_skinny(M):
        # QWEN38_NVFP4_SKINNY: the CUDA kernel that feeds the tensor cores from registers, for
        # every decode-side row count at once (one kernel, one order; tools/nvfp4_skinny.py)
        from tools.nvfp4_skinny import nvfp4_matmul_skinny
        return nvfp4_matmul_skinny(x, w, out=out)
    if block_n is None and split_k is None and use_v2(M):
        from tools.nvfp4_linear_v2 import nvfp4_matmul_v2
        return nvfp4_matmul_v2(x, w, block_m=block_m, num_warps=num_warps,
                               num_stages=num_stages, out=out)
    cfg = pick_config(w.N, w.K, M)
    block_m = cfg["block_m"] if block_m is None else block_m
    block_n = cfg["block_n"] if block_n is None else block_n
    split_k = cfg["split_k"] if split_k is None else split_k
    num_warps = cfg["num_warps"] if num_warps is None else num_warps
    num_stages = cfg["num_stages"] if num_stages is None else num_stages
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.bfloat16, device=x.device)
    kq = w.K // 128
    if split_k > 1:
        parts = torch.empty(split_k, M, w.N, dtype=torch.float32, device=x.device)
        _nvfp4_linear_kernel[(triton.cdiv(w.N, block_n), triton.cdiv(M, block_m), split_k)](
            x, w.w, w.s, parts, M, w.N, kq, w.s2,
            x.stride(0), w.w.stride(0), w.s.stride(0), parts.stride(0), parts.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, SPLIT_K=split_k,
            num_warps=num_warps, num_stages=num_stages)
        out.copy_(parts.sum(0).to(torch.bfloat16))
        return out
    _nvfp4_linear_kernel[(triton.cdiv(w.N, block_n), triton.cdiv(M, block_m), 1)](
        x, w.w, w.s, out, M, w.N, kq, w.s2,
        x.stride(0), w.w.stride(0), w.s.stride(0), 0, out.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, SPLIT_K=1,
        num_warps=num_warps, num_stages=num_stages)
    return out


# ------------------------------------------------------------------ quantisation
def _round_e2m1(a: torch.Tensor) -> torch.Tensor:
    """|a| in [0, 6] -> an index into FP4_GRID, round-to-nearest with ties to even, which is what
    the hardware converter does. Ties are not measure-zero here: the input is an fp8 value over a
    scale, so it lands exactly on 0.25 / 0.75 / 1.25 / 1.75 / 2.5 / 3.5 / 5.0 often."""
    grid = FP4_GRID.to(a.device)
    mid = (grid[1:] + grid[:-1]) / 2
    dn = torch.bucketize(a, mid, right=False, out_int32=True)
    up = torch.bucketize(a, mid, right=True, out_int32=True)
    return torch.where((up != dn) & (dn % 2 == 1), up, dn).to(torch.uint8)


def quantize_to_nvfp4(ref: torch.Tensor, *, scale_2: float | None = None,
                      rows: int = 2048) -> NVFP4Block:
    """Bf16/fp32 [N, K] -> NVFP4Block, round to nearest on both levels.

    The two-level scale is the whole design. `scale_2` is chosen so that the per-group scales
    `amax_group / 6` land inside e4m3's range with the largest group at e4m3's own maximum:
    `scale_2 = amax_tensor / (6 * 448)`. Then every group scale is `(amax_group / 6) / scale_2`,
    rounded to e4m3, and rounded **up** where rounding down would clip -- a clipped outlier in a
    group of 16 is a much larger error than the rounding it saves.
    """
    N, K = ref.shape
    assert K % GROUP == 0, (K, GROUP)
    dev = ref.device
    if scale_2 is None:
        amax = ref.abs().max().float()
        scale_2 = float((amax / (E2M1_MAX * E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny))
    codes = torch.empty(N, K // 2, dtype=torch.uint8, device=dev)
    scales = torch.empty(N, K // GROUP, dtype=torch.float8_e4m3fn, device=dev)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        x = ref[r0:r1].float().reshape(-1, GROUP)
        gmax = x.abs().amax(dim=1, keepdim=True)
        s = (gmax / E2M1_MAX) / scale_2
        s8 = s.clamp(0.0, E4M3_MAX).to(torch.float8_e4m3fn)
        sf = s8.float()
        # round up to the next e4m3 step wherever rounding down would clip the group's outlier
        need = sf * scale_2 * E2M1_MAX < gmax
        if need.any():
            bump = (sf * 1.0625).clamp(0.0, E4M3_MAX).to(torch.float8_e4m3fn).float()
            sf = torch.where(need, bump, sf)
            s8 = torch.where(need, sf, s8.float()).to(torch.float8_e4m3fn)
            sf = s8.float()
        eff = (sf * scale_2).clamp_min(torch.finfo(torch.float32).tiny)
        v = x / eff
        idx = _round_e2m1(v.abs().clamp(0.0, E2M1_MAX))
        nib = (idx | ((v < 0).to(torch.uint8) * 8)).reshape(r1 - r0, K)
        codes[r0:r1] = nib[:, 0::2] | (nib[:, 1::2] << 4)
        scales[r0:r1] = s8.reshape(r1 - r0, K // GROUP)
        del x, gmax, s, s8, sf, eff, v, idx, nib
    return NVFP4Block(codes, scales, scale_2)


# ------------------------------------------------------------------ tile choice
# Measured on this board (docs/kernels.md). The key is that
# the two MLP shapes want opposite things: `gate_proj` / `up_proj` are wide (N = 17408) and short
# (K = 5120), so the N axis alone fills the board and a narrow tile with one warp keeps the most
# loads in flight; `down_proj` is narrow (N = 5120) and long (K = 17408) and needs its K loop split.
#
# 2026-09-17, phase 7: the `block` bucket. A speculative verify is not a decode step and it is not
# a prefill -- it is two to sixteen rows -- and until now it was served the `decode` tile, which was
# chosen at one row. `tools/nvfp4_wide_probe.py` over all eight projections says what that costs:
# the same tile that reaches 216 GB/s on `gate_proj` at M = 1 reaches 138 at M = 14, and a tile
# chosen AT M = 14 reaches 149. Every entry below is the winner at M = 14-16 in that sweep, which
# is the width the shipped router actually submits (budget 16, 14.2 nodes a block measured).
_CONFIG: dict[tuple[int, int], dict[str, dict]] = {
    # gate_proj / up_proj: wide (N = 17408) and short (K = 5120)
    (17408, 5120): {
        "decode":  {"block_n": 16, "split_k": 8, "num_warps": 1, "num_stages": 3},
        "block":   {"block_n": 32, "split_k": 1, "num_warps": 8, "num_stages": 3},
        "mid":     {"block_m": 64, "block_n": 32, "split_k": 1, "num_warps": 4, "num_stages": 3},
        "prefill": {"block_m": 128, "block_n": 64, "split_k": 1, "num_warps": 4, "num_stages": 3},
    },
    # down_proj: narrow (N = 5120) and long (K = 17408). The only shape the sweep did NOT move:
    # its decode tile wins at every M up to 16, so `block` repeats it rather than omitting it.
    (5120, 17408): {
        "decode":  {"block_n": 16, "split_k": 8, "num_warps": 1, "num_stages": 3},
        "block":   {"block_n": 16, "split_k": 8, "num_warps": 1, "num_stages": 3},
        "mid":     {"block_m": 64, "block_n": 128, "split_k": 2, "num_warps": 4, "num_stages": 3},
        "prefill": {"block_m": 128, "block_n": 64, "split_k": 1, "num_warps": 4, "num_stages": 3},
    },
    # GDN in_proj_qkv, 48 per step
    (10240, 5120): {
        "block":   {"block_n": 32, "split_k": 1, "num_warps": 2, "num_stages": 3},
    },
    # GDN in_proj_z, 48 per step -- the shape that gains most, 162 -> 240 GB/s at M = 14
    (6144, 5120): {
        "block":   {"block_n": 128, "split_k": 2, "num_warps": 4, "num_stages": 3},
    },
    # GDN out_proj and attention o_proj share this shape: 48 + 16 per step
    (5120, 6144): {
        "block":   {"block_n": 64, "split_k": 8, "num_warps": 4, "num_stages": 3},
    },
    # attention q_proj (24 heads x 256, doubled for the output gate), 16 per step
    (12288, 5120): {
        "block":   {"block_n": 32, "split_k": 1, "num_warps": 2, "num_stages": 3},
    },
    # attention k_proj / v_proj, 32 per step. 2.9 MB is too small to reach the board's rate at any
    # M and no tile in the sweep changed that; the entry exists so the table is complete.
    (1024, 5120): {
        "block":   {"block_n": 16, "split_k": 8, "num_warps": 1, "num_stages": 3},
    },
}

_FALLBACK = {
    "decode":  {"block_n": 16, "split_k": 8, "num_warps": 1, "num_stages": 3},
    # An unmeasured shape keeps the tile it had: the block bucket must never be a guess.
    "block":   {"block_n": 16, "split_k": 8, "num_warps": 1, "num_stages": 3},
    "mid":     {"block_m": 64, "block_n": 64, "split_k": 1, "num_warps": 4, "num_stages": 3},
    "prefill": {"block_m": 128, "block_n": 64, "split_k": 1, "num_warps": 4, "num_stages": 3},
}

_SPLIT_K_OVERRIDE = int(_S.get("NVFP4_SPLITK"))

# The `block` bucket, OFF by default, and the reason is a measurement rather than caution.
#
#   "0"   every M keeps the tile chosen at M = 1 (what the engine shipped through phase 6)
#   "all" the whole measured table
#   "splitk" only the entries that keep `split_k` above one
#
# In isolation, on a 3 GB cold chain of distinct weights, the table is worth 11-14 % at M = 14-16
# on `gate_proj`. In the engine it is worth -1 %. The two disagree because a chain of INDEPENDENT
# matmuls lets the scheduler overlap one kernel's tail with the next one's head, and a verify pass
# is a dependency chain -- one kernel in flight at a time, where the programs of a single launch
# are all the occupancy there is. `split_k = 8` gives eight times as many programs as `split_k = 1`
# and that is worth more, in the engine, than the tile shape the isolated sweep prefers.
_BLOCK_TILES = _S.get("NVFP4_BLOCK_TILES")

# Above this row count a projection is unpacked to bf16 once and handed to the library GEMM instead
# of being decoded in registers. 0 turns it off, which restores the W4A16 path at every M.
#
# 512 is where the trade turns, measured: at 384 rows the W4A16 kernel is 1.8 % ahead and at 512 the
# unpack is 12.6 %. Below it the unpack's 2K bytes a row are not yet amortised; above it the GEMM's
# arithmetic rate is what matters and the library's is several times the kernel's. At 8,192 rows a
# prefill goes 466 to 849 tok/s.
DEQUANT_FROM = int(_S.get("NVFP4_DEQUANT_FROM"))

# The v2 kernel with prefill-sized tiles against both of the paths above, cold:
# 1.4-1.65x ahead of unpack + library GEMM at 512 rows, level at 2048, 25-30 % behind at 8192.
# With this on, every row count above the decode band and below
# PREFILL_V2_UNTIL takes v2 with the tile below; from PREFILL_V2_UNTIL up the unpack path is kept.
# Off by default: it changes a prefill's arithmetic, so it is quality-gated, not bit-gated.
PREFILL_V2 = _S.get("NVFP4_PREFILL_V2") == "1"
PREFILL_V2_UNTIL = int(_S.get("NVFP4_PREFILL_V2_UNTIL"))
PREFILL_V2_TILE = {"block_m": 128, "block_n": 128, "num_warps": 8, "num_stages": 2}


def prefill_v2(M: int) -> bool:
    """Whether a row count takes the v2 prefill tile rather than v1's or the unpack path."""
    return PREFILL_V2 and not use_v2(M) and M > 32 and M < PREFILL_V2_UNTIL


def set_config(N: int, K: int, bucket: str, cfg: dict) -> None:
    _CONFIG.setdefault((N, K), {})[bucket] = dict(cfg)


def pick_config(N: int, K: int, M: int) -> dict:
    """The tile shape for one projection at one row count.

    FOUR regimes, and the fourth is the one this engine spends its time in. At M = 1 the whole cost
    is the weight read, the N axis alone does not fill the board on a narrow output, and the K loop
    has to be split to keep enough loads in flight. At prefill M the row axis fills the machine by
    itself and the wider row loads win. Between them is a verified speculative block -- two to
    sixteen rows -- and it used to be served the M = 1 tile. It is not an M = 1 problem: the same
    weight tile now feeds a real `tl.dot` rather than a masked one, and what the sweep finds is
    that the shapes want a wider N tile and more warps as soon as M leaves 1.
    """
    if (N, K) not in _CONFIG:
        from tools.tile_warn import missing
        missing("nvfp4_linear", N, K, "_FALLBACK")
    bucket = ("decode" if M <= 1 else "block" if M <= 32 else
              "mid" if M <= 128 else "prefill")
    if bucket == "block":
        cand = _CONFIG.get((N, K), {}).get("block")
        if (_BLOCK_TILES == "0" or cand is None
                or (_BLOCK_TILES == "splitk" and cand.get("split_k", 1) < 2)):
            bucket = "decode"
    cfg = dict(_CONFIG.get((N, K), _FALLBACK).get(bucket, _FALLBACK[bucket]))
    cfg.setdefault("block_m", 16 if M <= 16 else (32 if M <= 32 else 64))
    if bucket in ("decode", "block"):
        cfg["block_m"] = 16 if M <= 16 else 32
    if _SPLIT_K_OVERRIDE:
        cfg["split_k"] = _SPLIT_K_OVERRIDE
    return cfg
