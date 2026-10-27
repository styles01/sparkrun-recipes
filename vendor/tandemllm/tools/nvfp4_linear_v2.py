"""The same W4A16 product as `tools/nvfp4_linear.py`, with the activation read rewritten.

The v1 kernel loses more than a third of its rate between one row and sixteen: 186 GB/s at M = 1
and 127 at M = 16 on a cold 2.96 GB chain, and 139 GB/s in the engine's own verify pass where the
fp8 `lm_head` holds 217 on the same block. The tile table phase 7 built does not close it, and the
reason it cannot is that the loss is not in the tiling.

It is in how `x` is read. v1's `_chunk_dot` loads the activation twice per 32-wide K chunk:

    xe = tl.load(x_base + 2 * tl.arange(0, 16))          # logical K 0, 2, 4, ... 30
    xo = tl.load(x_base + 2 * tl.arange(0, 16) + 1)      # logical K 1, 3, 5, ... 31

Those are **strided gathers of 2-byte elements at a 4-byte pitch**, eight of them per 128-wide K
step, each covering BLOCK_M rows. At M = 1 fifteen of sixteen rows are masked away and the gather
costs almost nothing; at M = 16 every row is live and the kernel issues sixteen times the
uncoalesced accesses. That is the whole M curve, and it explains why it is a curve rather than a
step: it is linear in the number of live rows.

v2 reads the same 128 logical K values as ONE contiguous 256-byte-per-row tile and does the
even/odd split in registers, the way v1 already splits the weight bytes:

    xt      = tl.load(x_base + q * 128 + tl.arange(0, 128))   # [BLOCK_M, 128], coalesced
    xe, xo  = tl.split(tl.reshape(xt, [BLOCK_M, 64, 2]))      # register shuffle, no memory

and then, because a dot product does not care in what order K is summed, it never puts the halves
back in logical order. The weight tile decodes to the same two halves, the group scales are
broadcast onto the decoded weight (exact: e2m1 has two significand bits, e4m3 has three, fp16
holds ten), and the product is taken as either

  * `DOTS = 1` -- one `tl.dot` at K = 128 over the concatenation [even | odd], or
  * `DOTS = 2` -- two `tl.dot` at K = 64, one per half, which skips the two concatenations,

against v1's **eight** `tl.dot` at K = 16 per 128-wide K step, one per NVFP4 scale group half.

At M = 1 the gather v2 removes was nearly free -- fifteen of sixteen rows masked away -- and v2 is
ahead there anyway, 203 against 185 GB/s, because `split_k` goes with it. `pick_config_v2` covers M = 1..32.
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

from tools.nvfp4_linear import NVFP4Block, _fp4_decode  # noqa: E402


@triton.jit
def _scale_128(s, BN: tl.constexpr):
    """[BN, 8] e4m3 group scales -> [BN, 64] fp16, each scale repeated over its eight columns.

    Group `j` of an NVFP4 row covers logical K 16j..16j+15. The even half of the decoded tile holds
    logical 2c at column c, so column c takes group c // 8; the odd half holds 2c+1, same group.
    One [BN, 64] expansion therefore serves both halves.
    """
    return tl.reshape(tl.broadcast_to(tl.expand_dims(s, 2), [BN, 8, 8]), [BN, 64])


@triton.jit
def _cat2(a, b, B: tl.constexpr, W: tl.constexpr):
    """[B, W] + [B, W] -> [B, 2W], a's columns first."""
    return tl.reshape(tl.permute(tl.join(a, b), [0, 2, 1]), [B, 2 * W])


@triton.jit
def _nvfp4_linear_v2_kernel(X, W, S, Y, M, N, KQ, s2, S2,
                            stride_xm, stride_wn, stride_sn, stride_yk, stride_ym,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            SPLIT_K: tl.constexpr, DOTS: tl.constexpr,
                            PREFETCH: tl.constexpr, PER_ROW_S2: tl.constexpr):
    """Y[M, N] = (X[M, K] @ W[N, K]^T) * s2; W packed [N, K/2] uint8, S [N, K/16] fp8. KQ = K/128.

    Same grid, same split-K partial planes, same output contract as `_nvfp4_linear_kernel`. What
    differs is inside the K loop: one contiguous activation load and one or two wide dots instead
    of eight narrow ones over eight strided gathers.
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
    xk = tl.arange(0, 128)[None, :]
    x_base = X + rm_[:, None] * stride_xm
    w_tile = W + rn_[:, None] * stride_wn + tl.arange(0, 64)[None, :]
    s_tile = S + rn_[:, None] * stride_sn + tl.arange(0, 8)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for q in range(pid_k, KQ, SPLIT_K):
        packed = tl.load(w_tile + q * 64)
        if PREFETCH:
            # Spark-specific: llama.cpp PR #26705 measured +4.8 % median decode from touching the
            # next tile's line early. `evict_last` on a read of the next step's first byte is the
            # portable way to say the same thing in Triton; the value is discarded.
            tl.load(w_tile + (q + SPLIT_K) * 64, mask=(q + SPLIT_K) < KQ, other=0,
                    eviction_policy="evict_last")
        we, wo = _fp4_decode(packed)                        # [BN, 64] each: logical 2c and 2c+1
        sc = _scale_128(tl.load(s_tile + q * 8).to(tl.float16), BLOCK_N)
        we = we * sc
        wo = wo * sc
        xt = tl.load(x_base + q * 128 + xk, mask=m_mask, other=0.0).to(tl.float16)
        xe, xo = tl.split(tl.reshape(xt, [BLOCK_M, 64, 2]))
        if DOTS == 1:
            acc = tl.dot(_cat2(xe, xo, BLOCK_M, 64),
                         tl.trans(_cat2(we, wo, BLOCK_N, 64)), acc=acc)
        else:
            acc = tl.dot(xe, tl.trans(we), acc=acc)
            acc = tl.dot(xo, tl.trans(wo), acc=acc)
    # PER_ROW_S2 is what makes a GROUPED launch possible: several projections concatenated along N
    # do not share a per-tensor scale, and a scale read per output column costs one 256-byte load
    # per tile and is the same fp32 multiply per element. With the flag off this compiles to
    # exactly the scalar form, which is the shipped path.
    if PER_ROW_S2:
        scale = tl.load(S2 + rn_)[None, :]
    else:
        scale = s2
    if SPLIT_K == 1:
        tl.store(Y + rm[:, None] * stride_ym + rn[None, :], (acc * scale).to(tl.bfloat16),
                 mask=m_mask & n_mask[None, :])
    else:
        tl.store(Y + pid_k * stride_yk + rm[:, None] * stride_ym + rn[None, :], acc * scale,
                 mask=m_mask & n_mask[None, :])


# ------------------------------------------------------------------ tile choice
# From the cold sweep (docs/kernels.md): every projection on a 2-3 GB chain of
# distinct weights, M = 1..32, against v1's shipped decode tile. Two things came out of it and the
# second is the larger.
#
#   * v2 beats v1 at the SAME tile -- 144 against 133 GB/s on `gate_proj` at M = 14 -- which is the
#     contiguous activation read on its own.
#   * and `split_k` has to go. A split-K launch writes SPLIT_K x M x N fp32 partials and reads them
#     back: at M = 16 on `gate_proj` that is 8.9 MB written plus 8.9 MB read against 50.1 MB of
#     weight, a 36 % traffic surcharge that is LINEAR IN M and invisible at M = 1 where it is 1.1 MB.
#     That, not the tiling, is the M curve both kernels were showing. `split_k = 1` with a wider N
#     tile and more warps takes v2 from 144 to 199 GB/s at M = 14.
#
# Every entry is the cold winner at M = 14-16, which is the width the shipped router submits.
_V2_CONFIG: dict[tuple[int, int], dict] = {
    (17408, 5120): {"block_n": 64, "split_k": 1, "num_warps": 8, "num_stages": 3},   # gate/up 199
    (5120, 17408): {"block_n": 64, "split_k": 1, "num_warps": 4, "num_stages": 3},   # down    187
    (10240, 5120): {"block_n": 64, "split_k": 1, "num_warps": 8, "num_stages": 3},   # gdn qkv 194
    (6144, 5120):  {"block_n": 64, "split_k": 1, "num_warps": 8, "num_stages": 3},   # gdn z   193
    (5120, 6144):  {"block_n": 64, "split_k": 1, "num_warps": 4, "num_stages": 3},   # out/o   190
    (12288, 5120): {"block_n": 64, "split_k": 1, "num_warps": 8, "num_stages": 3},   # attn q  196
    # k/v: 2.9 MB is too small to reach the board's rate at any M -- 117 GB/s is the ceiling here,
    # as it was for v1 at 89. The entry exists so the table is complete.
    (1024, 5120):  {"block_n": 64, "split_k": 1, "num_warps": 4, "num_stages": 3},   # attn kv 116
    # The fused groups. Each inherits the tile its dominant member was tuned to -- the members are
    # the same rows in the same layout and the tile is a property of the shape of the read, not of
    # which projection the rows belong to.
    (34816, 5120): {"block_n": 64, "split_k": 1, "num_warps": 8, "num_stages": 3},   # gate+up
    (14336, 5120): {"block_n": 64, "split_k": 1, "num_warps": 8, "num_stages": 3},   # q+k+v
    (16384, 5120): {"block_n": 64, "split_k": 1, "num_warps": 8, "num_stages": 3},   # gdn qkv+z
}

_V2_FALLBACK = {"block_n": 64, "split_k": 1, "num_warps": 4, "num_stages": 3}

# ON by default since phase 8, and the range starts at ONE rather than two on purpose.
#
# v2 is faster at M = 1 as well (203 against 185 GB/s cold on gate/up), but that is not why the
# range includes it. This engine's correctness gate is bit-level equality between a speculative
# greedy run and a non-speculative one, and the non-speculative one decodes at M = 1. Leaving M = 1
# on v1 would put the two runs on two different accumulation orders and the gate would fail on
# arithmetic rather than on anything real. One kernel for every decode-side row count, one order.
#
#   "0"    off; the engine is exactly what phase 7 shipped
#   "1"    on for V2_MIN..V2_MAX
#   "all"  on at every M the W4A16 path sees
V2 = _S.get("NVFP4_V2")
V2_MIN = int(_S.get("NVFP4_V2_MIN"))
V2_MAX = int(_S.get("NVFP4_V2_MAX"))
DOTS = int(_S.get("NVFP4_V2_DOTS"))
PREFETCH = int(_S.get("NVFP4_V2_PREFETCH"))


def set_config_v2(N: int, K: int, cfg: dict) -> None:
    _V2_CONFIG[(N, K)] = dict(cfg)


# A whole-table override, for the in-engine A/B the phase-7 trap list demands: a tile that wins on
# a free-running chain can lose in a dependency chain, where the programs of the single launch in
# flight are all the occupancy there is. `BN` forces one N tile on every shape, `W` one warp count.
_BN = int(_S.get("NVFP4_V2_BN"))
_W = int(_S.get("NVFP4_V2_W"))


def pick_config_v2(N: int, K: int, M: int) -> dict:
    if (N, K) not in _V2_CONFIG:
        from tools.tile_warn import missing
        missing("nvfp4_linear_v2", N, K, _V2_FALLBACK)
    cfg = dict(_V2_CONFIG.get((N, K), _V2_FALLBACK))
    if _BN:
        cfg["block_n"] = _BN
    if _W:
        cfg["num_warps"] = _W
    cfg.setdefault("block_m", 16 if M <= 16 else 32)
    cfg["block_m"] = 16 if M <= 16 else (32 if M <= 32 else 64)
    return cfg


def use_v2(M: int) -> bool:
    if V2 == "0":
        return False
    if V2 == "all":
        return True
    return V2_MIN <= M <= V2_MAX


def nvfp4_matmul_v2(x: torch.Tensor, w: NVFP4Block, *, block_m: int | None = None,
                    block_n: int | None = None, split_k: int | None = None,
                    num_warps: int | None = None, num_stages: int | None = None,
                    dots: int | None = None, prefetch: int | None = None,
                    out: torch.Tensor | None = None) -> torch.Tensor:
    """Y[M, N] = x[M, K] @ W[N, K]^T, W in the NVFP4 layout, v2 kernel. x is bf16."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == w.K, (x.shape, w.shape)
    M = x.shape[0]
    x = x.contiguous()
    cfg = pick_config_v2(w.N, w.K, M)
    block_m = cfg["block_m"] if block_m is None else block_m
    block_n = cfg["block_n"] if block_n is None else block_n
    split_k = cfg["split_k"] if split_k is None else split_k
    num_warps = cfg["num_warps"] if num_warps is None else num_warps
    num_stages = cfg["num_stages"] if num_stages is None else num_stages
    dots = DOTS if dots is None else dots
    prefetch = PREFETCH if prefetch is None else prefetch
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.bfloat16, device=x.device)
    kq = w.K // 128
    per_row = getattr(w, "s2v", None)
    if split_k > 1:
        parts = torch.empty(split_k, M, w.N, dtype=torch.float32, device=x.device)
        _nvfp4_linear_v2_kernel[(triton.cdiv(w.N, block_n), triton.cdiv(M, block_m), split_k)](
            x, w.w, w.s, parts, M, w.N, kq, 0.0 if per_row is not None else w.s2,
            per_row if per_row is not None else x,
            x.stride(0), w.w.stride(0), w.s.stride(0), parts.stride(0), parts.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, SPLIT_K=split_k, DOTS=dots, PREFETCH=prefetch,
            PER_ROW_S2=per_row is not None,
            num_warps=num_warps, num_stages=num_stages)
        out.copy_(parts.sum(0).to(torch.bfloat16))
        return out
    _nvfp4_linear_v2_kernel[(triton.cdiv(w.N, block_n), triton.cdiv(M, block_m), 1)](
        x, w.w, w.s, out, M, w.N, kq, 0.0 if per_row is not None else w.s2,
        per_row if per_row is not None else x,
        x.stride(0), w.w.stride(0), w.s.stride(0), 0, out.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, SPLIT_K=1, DOTS=dots, PREFETCH=prefetch,
        PER_ROW_S2=per_row is not None,
        num_warps=num_warps, num_stages=num_stages)
    return out


# ------------------------------------------------------------------ grouped projections
class NVFP4Group:
    """Several NVFP4 projections of the same K, laid out as one weight so they are ONE launch.

    The phase-8 handoff priced the gap this closes: the W4A16 set moves 13.685 GB a verify step at
    169 GB/s in the engine and 194 GB/s cold on the same table, and the 25 GB/s between them is the
    dependency chain. `split_k = 1` at `block_n = 64` puts 272 programs on the widest shape against
    48 SMs, and a verify runs **one kernel at a time**: the tail of every launch is SMs going idle
    while the last programs finish, and there are eleven launches a layer to pay it on.

    More programs per launch is the cheapest way to get more kernels in flight, and three sets of
    projections in this model can be launched together because they read the same activation and
    do not read each other's output:

      * `gate_proj` and `up_proj` -- 272 programs each, 544 together;
      * `q_proj`, `k_proj` and `v_proj` -- 192 + 16 + 16, and the two 16-program launches were
        never going to fill the board on their own;
      * the linear-attention `in_proj_qkv` and `in_proj_z`.

    The concatenation is along N, so each member's rows stay contiguous and the individual
    `NVFP4Block`s remain valid VIEWS of the same storage -- nothing is duplicated and the
    unfused path still works. What cannot be concatenated is the per-tensor scale `s2`, which
    differs per projection; `s2v` carries it per output column and `PER_ROW_S2` reads it there.
    The arithmetic per element is unchanged: the same fp32 accumulator times the same fp32 scale.
    """

    __slots__ = ("w", "s", "s2v", "N", "K", "sizes", "names", "s2", "_srun")

    def __init__(self, blocks: list, names: list[str]):
        assert blocks, "an empty group"
        K = blocks[0].K
        for b in blocks:
            assert b.K == K, (b.K, K)
        self.K = K
        self.sizes = [int(b.N) for b in blocks]
        self.N = sum(self.sizes)
        self.names = list(names)
        dev = blocks[0].w.device
        self.w = torch.cat([b.w for b in blocks], dim=0).contiguous()
        self.s = torch.cat([b.s for b in blocks], dim=0).contiguous()
        self.s2v = torch.cat([torch.full((int(b.N),), float(b.s2), dtype=torch.float32,
                                         device=dev) for b in blocks])
        self.s2 = float(blocks[0].s2)
        # Re-point every member at its slice of the fused storage. `torch.cat` along dim 0 leaves
        # each member's rows contiguous, so these are ordinary views and the originals can go.
        off = 0
        for b in blocks:
            n = int(b.N)
            b.w = self.w[off:off + n]
            b.s = self.s[off:off + n]
            off += n

    @property
    def shape(self):
        return (self.N, self.K)

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() + self.s2v.numel() * 4


def nvfp4_matmul_group(x: torch.Tensor, g: NVFP4Group) -> torch.Tensor:
    """`x[M, K] @ [g.N, K]^T` in one launch. The caller splits the result on `g.sizes`."""
    from tools.nvfp4_skinny import nvfp4_matmul_skinny, use_skinny
    if use_skinny(x.shape[0]):
        return nvfp4_matmul_skinny(x, g)
    return nvfp4_matmul_v2(x, g)


#: Off until it is measured in the engine, which is the only place the ranking counts
#: (the phase-7 trap list, first entry). `QWEN38_FUSE_PROJ=1` turns it on.
FUSE_PROJ = _S.get("FUSE_PROJ") == "1"
