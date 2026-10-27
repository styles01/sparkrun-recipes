"""Y = x @ W^T with W left in the checkpoint's stored format.

The checkpoint stores every quantised projection as fp8 e4m3 codes [N, K] plus a bf16 scale table
[N/128, K/128] -- one scale per 128x128 block of the weight -- and the dequantised weight is
`code.to(f32) * scale[n // 128, k // 128]`. Dequantising into memory would double both the resident
footprint and the bytes read per decode step, and the decode step of this model is nothing but
weight reads, so the weights stay as codes and the scale is applied inside the GEMM.

Tiling follows the scale table: BLOCK_N = BLOCK_K = 128 makes every tile step cover exactly one
scale block, so the scale is a single scalar per step, applied to the fp32 accumulator rather than
to the weights. That is both cheaper and closer to the reference than scaling bf16 weights: e4m3 ->
bf16 is exact (four significand bits into eight), the products accumulate in fp32, and one fp32
multiply at the end carries the block scale.

Where the block scale is applied is a real choice, not a detail:

  * `SCALE_W = True` folds it into the weight in bf16 first, which is the checkpoint's own
    definition of the dequantised weight and therefore reproduces a stock bf16 module.
  * `SCALE_W = False` leaves the weight as exact bf16 codes and scales the fp32 accumulator, which
    is arithmetically better -- it drops a bf16 rounding of every weight -- and is what a serving
    stack does with a block-scaled fp8 GEMM.

They differ by about 0.2 % per projection, which is invisible in one layer and is not invisible
after sixty-four of them. The default is the reference-matching one; the other is a measured option.

Small N with a small M leaves the board under-occupied -- `out_proj` is N = 5120, which is 40 tiles
-- so the kernel can split the K loop across programs and reduce in a second pass. SPLIT_K = 1 is
the plain path.
"""

from __future__ import annotations

import os
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in __import__("sys").path:
    # run as a script from tools/: the repo root, appended (lowest priority), for engine.settings
    __import__("sys").path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

import torch
import triton
import triton.language as tl

BLOCK = 128  # the checkpoint's quantisation block, in both N and K


@triton.jit
def _fp8_gemm_kernel(X, W, S, Y,
                     M, N, K,
                     stride_xm, stride_wn, stride_sn, stride_ym,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SCALE_W: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # BLOCK_N divides the 128-row scale block (128 or 64), so the whole n-tile shares one
    # scale row: pid_n itself at 128, every second pid_n at 64.
    s_row = (pid_n * BLOCK_N) // 128
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(X + rm[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :])
        s = tl.load(S + s_row * stride_sn + k0 // BLOCK_K)
        if SCALE_W:
            wb = w.to(tl.bfloat16) * s.to(tl.bfloat16)
            acc += tl.dot(x, tl.trans(wb), out_dtype=tl.float32)
        else:
            acc += s.to(tl.float32) * tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], acc.to(tl.bfloat16), mask=m_mask[:, None])


@triton.jit
def _fp8_gemm_splitk_kernel(X, W, S, Y,
                            M, N, K,
                            stride_xm, stride_wn, stride_sn, stride_yk, stride_ym,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            SPLIT_K: tl.constexpr, SCALE_W: tl.constexpr):
    """Same product, K split across SPLIT_K programs, each writing its own fp32 partial plane."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    s_row = pid_n
    n_steps = K // BLOCK_K
    for step in range(pid_k, n_steps, SPLIT_K):
        kk = step * BLOCK_K + rk
        x = tl.load(X + rm[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :])
        s = tl.load(S + s_row * stride_sn + step)
        if SCALE_W:
            wb = w.to(tl.bfloat16) * s.to(tl.bfloat16)
            acc += tl.dot(x, tl.trans(wb), out_dtype=tl.float32)
        else:
            acc += s.to(tl.float32) * tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    tl.store(Y + pid_k * stride_yk + rm[:, None] * stride_ym + rn[None, :], acc, mask=m_mask[:, None])


@triton.jit
def _fp8_dequant_kernel(W, S, Y, N, K, stride_wn, stride_sn, stride_yn,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    w = tl.load(W + rn[:, None] * stride_wn + rk[None, :])
    s = tl.load(S + pid_n * stride_sn + pid_k).to(tl.float32)
    tl.store(Y + rn[:, None] * stride_yn + rk[None, :], (w.to(tl.float32) * s).to(tl.bfloat16))


class FP8Block:
    """A projection weight as the checkpoint stores it: fp8 codes plus a bf16 128x128 scale table."""

    __slots__ = ("w", "s", "N", "K", "_bf16")  # `w` is cleared by tools that expand a weight once

    def __init__(self, codes: torch.Tensor, scale_inv: torch.Tensor):
        assert codes.dtype == torch.float8_e4m3fn and codes.dim() == 2, (codes.dtype, codes.shape)
        self.w = codes
        self.s = scale_inv.to(torch.bfloat16).contiguous()
        self.N, self.K = codes.shape
        assert self.N % BLOCK == 0 and self.K % BLOCK == 0, (self.N, self.K)
        assert tuple(self.s.shape) == (self.N // BLOCK, self.K // BLOCK), (self.s.shape, codes.shape)
        self._bf16 = None

    @property
    def shape(self):
        return (self.N, self.K)

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() * 2

    def dequant(self) -> torch.Tensor:
        """The bf16 weight, as the reference implementation would hold it."""
        if self.w.device.type != "cuda":
            s = self.s.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
            return (self.w.to(torch.float32) * s).to(torch.bfloat16)
        y = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.w.device)
        _fp8_dequant_kernel[(self.N // BLOCK, self.K // BLOCK)](
            self.w, self.s, y, self.N, self.K,
            self.w.stride(0), self.s.stride(0), y.stride(0),
            BLOCK_N=BLOCK, BLOCK_K=BLOCK, num_warps=4, num_stages=2)
        return y

    def bf16_cached(self) -> torch.Tensor:
        if self._bf16 is None:
            self._bf16 = self.dequant()
        return self._bf16

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """The Linear interface: `x[..., K] @ W^T` -> [..., N]."""
        return fp8_matmul(x.reshape(-1, x.shape[-1]), self).view(*x.shape[:-1], self.N)


class FP8Group(FP8Block):
    """Several FP8 projections of the same K as ONE weight, so they are one launch.

    The FP8 counterpart of `tools/nvfp4_linear_v2.NVFP4Group`: `gate_proj` + `up_proj`, `q/k/v`,
    and the linear-attention `in_proj_qkv` + `in_proj_z` read the same activation and not each
    other's output. The codes and the 128x128 scale tables are concatenated along N -- every N is a
    multiple of 128, so each member's scale rows stay aligned with its codes -- and each member is
    re-pointed at its slice, so nothing is duplicated and the unfused path still works.

    It IS an `FP8Block`, so `fp8_matmul` takes it unchanged: a program owns one 128-wide N tile and
    walks the same K loop with the same `block_m` whatever the tile's neighbours are, so each
    member's columns of the fused product are byte-identical to the member's own call.
    """

    __slots__ = ("sizes", "names")

    def __init__(self, blocks: list, names: list[str]):
        assert blocks, "an empty group"
        K = blocks[0].K
        for b in blocks:
            assert isinstance(b, FP8Block) and b.K == K, (type(b).__name__, b.K, K)
        codes = torch.cat([b.w for b in blocks], dim=0).contiguous()
        scale = torch.cat([b.s for b in blocks], dim=0).contiguous()
        super().__init__(codes, scale)
        self.sizes = [int(b.N) for b in blocks]
        self.names = list(names)
        off = 0
        for b in blocks:
            n = int(b.N)
            b.w = self.w[off:off + n]
            b.s = self.s[off // BLOCK:(off + n) // BLOCK]
            b._bf16 = None
            off += n


SCALE_ON_WEIGHT = _S.get("SCALE_ON") == "weight"


def default_block_m(M: int) -> int:
    """The row tile for M rows. 32 for a 17..32-row verify: the old 64 left
    half of every tile masked there, and on the board the plain-FP8 projections of a 17..28-row
    verify took 219.6-220.1 ms a step against 114.8-115.7 at 32 (tools/fp8_probe.py, the served
    shapes). Every row count's rows are bit-identical to the same row computed alone at M = 1 with
    any of the tiles 16/32/64 -- a program's row never sees another row, and the K loop is the same
    -- so this is speed only: the walk and the verify stay the same numbers."""
    return 16 if M <= 16 else (32 if M <= 32 else (64 if M <= 128 else 128))


# a launch configuration per projection shape for the verify side (M <= 32): the N tile (128
# or 64), warps and pipeline stages. One configuration per SHAPE, never per row count, so the one-row
# walk and every verify of that projection run the same program shape; a table entry is written only
# for a configuration whose output is bit-identical to the default's at 1..32 rows
# (tools/fp8_probe.py --sweep), so it changes the speed and not one bit of the text.
# QWEN38_FP8_TILES names the JSON table ({"17408x5120": {"bn": 128, "warps": 4, "stages": 4}}).
_TILES: dict[tuple[int, int], dict] = {}
TILE_DEFAULT = {"bn": 128, "warps": 4, "stages": 3}
TILES_MAX_M = 32


def load_tiles(path: str | None) -> int:
    import json
    _TILES.clear()
    if not path:
        return 0
    if not os.path.isfile(path):
        print(f"[fp8] WARNING: QWEN38_FP8_TILES={path} does not exist; every shape takes "
              f"{TILE_DEFAULT}", flush=True)
        return 0
    for key, v in json.load(open(path)).items():
        if key.startswith("_"):
            continue
        n, k = (int(t) for t in key.split("x"))
        cfg = {"bn": int(v.get("bn", 128)), "warps": int(v.get("warps", 4)), "stages": int(v.get("stages", 3))}
        assert cfg["bn"] in (64, 128) and n % cfg["bn"] == 0, (key, cfg)
        _TILES[(n, k)] = cfg
    return len(_TILES)


def tile_for(N: int, K: int, M: int) -> dict:
    if M <= TILES_MAX_M:
        return _TILES.get((N, K), TILE_DEFAULT)
    return TILE_DEFAULT


load_tiles(_S.get("FP8_TILES"))


def fp8_matmul(x: torch.Tensor, w: FP8Block, *, block_m: int | None = None,
               split_k: int = 1, num_warps: int | None = None, num_stages: int | None = None,
               out: torch.Tensor | None = None, scale_on_weight: bool | None = None,
               block_n: int | None = None) -> torch.Tensor:
    """Y[M, N] = x[M, K] @ W[N, K]^T, W in the stored format. x is bf16."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == w.K, (x.shape, w.shape)
    M = x.shape[0]
    x = x.contiguous()
    if x.device.type != "cuda":
        return torch.nn.functional.linear(x, w.dequant())
    if block_m is None:
        block_m = default_block_m(M)
    t = tile_for(w.N, w.K, M)
    block_n = t["bn"] if block_n is None else block_n
    num_warps = t["warps"] if num_warps is None else num_warps
    num_stages = t["stages"] if num_stages is None else num_stages
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.bfloat16, device=x.device)
    scale_w = SCALE_ON_WEIGHT if scale_on_weight is None else scale_on_weight
    if block_m >= 128:
        num_stages = min(num_stages, 2)  # a 128x128 tile pair per stage is the shared-memory limit
    if split_k > 1:
        parts = torch.empty(split_k, M, w.N, dtype=torch.float32, device=x.device)
        _fp8_gemm_splitk_kernel[(w.N // BLOCK, triton.cdiv(M, block_m), split_k)](
            x, w.w, w.s, parts, M, w.N, w.K,
            x.stride(0), w.w.stride(0), w.s.stride(0), parts.stride(0), parts.stride(1),
            BLOCK_M=block_m, BLOCK_N=BLOCK, BLOCK_K=BLOCK, SPLIT_K=split_k, SCALE_W=scale_w,
            num_warps=num_warps, num_stages=num_stages)
        out.copy_(parts.sum(0).to(torch.bfloat16))
        return out
    _fp8_gemm_kernel[(w.N // block_n, triton.cdiv(M, block_m))](
        x, w.w, w.s, out, M, w.N, w.K,
        x.stride(0), w.w.stride(0), w.s.stride(0), out.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=BLOCK, SCALE_W=scale_w,
        num_warps=num_warps, num_stages=num_stages)
    return out
