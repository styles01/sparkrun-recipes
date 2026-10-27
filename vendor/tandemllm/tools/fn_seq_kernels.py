"""Flash-Next's sequence kernels (ENG-227, ENG-147): the parts of a row that read OTHER positions.

A row of the trunk reads other positions in four places: the GDN recurrence (the state after the
row's parent), the GDN convolution (the three positions before it), the PLE convolution (the
positions 3, 6 and 9 before it) and attention (every visible key). In a verify block the positions
before a row are its ancestors, which are not the rows before it in a tree, and some of them sit in
the committed state rather than in the block. Every kernel here therefore takes, per row, WHERE its
predecessors are, and computes the row with an instruction sequence that does not depend on the
other rows or on where they are. The decode step is the same kernels at one row: a verified row is
the decode step's row bit for bit, which is the property lossless speculation needs (docs/kernels.md,
"Verify twins").

  gdn_pre_rows   the GDN depthwise causal conv (4 taps) and SiLU over the row's window, and the
                 gates g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b). The window's taps
                 come from rows of the block or from the conv state (`src`). One row (decode) can
                 also shift the state in place.
  gdn_rows       the gated delta rule, row after row. A CTA owns one value head and 16 value
                 columns of its [128, 128] state; lane l holds key rows l, l + 32, l + 64, l + 96 of
                 four columns, so both reductions over the key dimension are one lane's four values
                 in order and then a fixed xor tree inside a warp. Per row the tile comes from the
                 previous row (registers), the entry state, or a scratch slot where an earlier row
                 with a second child left it (`ctl`); the tile is written back to the state only
                 when asked (decode, commit). Parked rows (`pre`, the last commit's path) are walked
                 first and written back: the commit folded into the next call.
  ple_gate       PLE's gate: g = bf16(sum bf16(key query) / sqrt(H)) per (row, stream), its signed
                 square root, gv = bf16(bf16(sigmoid(g)) value) -- the reference's bf16 roundings.
  ple_conv       PLE's dilated depthwise conv (4 taps, dilation 3) over the normed gv of the row and
                 of the positions 3, 6, 9 before it, SiLU, + gv.
  attn_rows      softmax(q k^T / sqrt(D)) v of every query head of one row over the row's visible
                 keys, in VIRTUAL order: visible key c is position c (dense) or the c-th selected
                 position (QSA, ENG-147); a position before the block's start L is its cache slot, a
                 later one is the row's ancestor at that depth (`anc`). The virtual columns are cut
                 into fixed chunks of 256 (nine cover the 2,051 keys a query ever sees), each chunk a
                 program, combined in chunk order. A row's arithmetic is a function of its own query
                 and visible keys only.
  qsa_pool       a pooled indexer key: the mean of a block's 4 raw keys, the norm, the rope at the
                 block's first position (the shared pool, and a verify row's own blocks)
  qsa_scores     a row's block scores: relu(q_h . pooled) summed over the heads in order
  qsa_topk       the 512 best blocks of a row (ties to the lower block), ascending, then the tail:
                 one bitonic sort in shared memory a row

Built on first use for sm_121a (CUDA via torch.utils.cpp_extension) and Triton (attention).
"""

from __future__ import annotations

import hashlib
import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine.settings import SETTINGS as _S  # noqa: E402

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

#: virtual columns a split program of `attn_rows` owns, and the splits: a query never sees more than
#: budget + compress_ratio - 1 = 2,051 keys (dense up to that context, 512 selected blocks of 4 and a
#: tail of at most 3 past it), so nine chunks of 256 hold every row at every context length
ATTN_CHUNK = 256
ATTN_SPLITS = 9
#: depths an ancestor table holds (a verify block is at most 32 rows)
ANC = 33
#: visible positions a row's QSA selection list holds (2,048 selected + a tail of at most 3, padded)
NSEL = 2064

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float b2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" :::); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------------------ GDN conv + gates
// grid (nC + 1, M), 256 threads. CTA x < nC: 256 channels of row m. The window of row m is
// [tap3, tap2, tap1, x_m]; SRC[m][j] (j = 0, 1, 2 for distance 3, 2, 1) is a row of the block
// (>= 0) or state column -1 - SRC (< 0; column 2 is the newest). acc = fma over the taps in order,
// y = bf16(silu(acc)). SHIFT (one row, decode): the state becomes [col1, col2, x] (all reads first).
// CTA x == nC: the gates of row m's H heads.
__global__ void __launch_bounds__(256) gdn_pre_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        __nv_bfloat16* __restrict__ CS, const __nv_bfloat16* __restrict__ W, const int* __restrict__ SRC,
        __nv_bfloat16* __restrict__ Y, int ldy, int C, int nC,
        const __nv_bfloat16* __restrict__ AB, int ldab, const float* __restrict__ ALOG,
        const float* __restrict__ DTB, float* __restrict__ G, float* __restrict__ BETA, int H, int shift) {
    pdl_wait();
    pdl_trigger();
    const int m = blockIdx.y;
    if ((int)blockIdx.x == nC) {
        const int h = threadIdx.x;
        if (h >= H) return;
        const float a = b2f(AB[(size_t)m * ldab + h]), b = b2f(AB[(size_t)m * ldab + H + h]);
        const float x = a + DTB[h];
        const float sp = x > 20.0f ? x : ::logf(1.0f + ::expf(x));
        G[(size_t)m * H + h] = -::expf(ALOG[h]) * sp;
        BETA[(size_t)m * H + h] = 1.0f / (1.0f + ::expf(-b));
        return;
    }
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    const float x = b2f(X[(size_t)m * ldx + c]);
    float tap[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int s = SRC[m * 3 + j];
        tap[j] = s >= 0 ? b2f(X[(size_t)s * ldx + c]) : b2f(CS[(size_t)c * 3 + (-1 - s)]);
    }
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) acc = fmaf(tap[j], b2f(W[(size_t)c * 4 + j]), acc);
    acc = fmaf(x, b2f(W[(size_t)c * 4 + 3]), acc);
    Y[(size_t)m * ldy + c] = __float2bfloat16_rn(acc / (1.0f + ::expf(-acc)));
    if (shift) {
        const __nv_bfloat16 s1 = CS[(size_t)c * 3 + 1], s2 = CS[(size_t)c * 3 + 2];
        CS[(size_t)c * 3 + 0] = s1;
        CS[(size_t)c * 3 + 1] = s2;
        CS[(size_t)c * 3 + 2] = X[(size_t)m * ldx + c];
    }
}

// ------------------------------------------------------------------------------ GDN recurrence
// One row of the rule on the thread's 16 values of the tile (lane l: key rows l + 32 j; warp w:
// columns c0 .. c0 + 3):
//   q, k <- l2-normalised (q also times 1/sqrt(128)), key head hk
//   s <- s exp(g); kv = s^T k; delta = (v - kv) beta; s += k (x) delta; out = s^T q
// OUT null: no output (a parked commit row).
__device__ __forceinline__ void gdn_row(float (&s)[4][4], const __nv_bfloat16* __restrict__ row, int koff, int voff,
        float g, float beta, int h, int hk, int lane, int c0, __nv_bfloat16* __restrict__ out) {
    float q[4], k[4], qq = 0.f, kk = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        q[j] = b2f(row[hk * 128 + lane + 32 * j]);
        k[j] = b2f(row[koff + hk * 128 + lane + 32 * j]);
        qq = fmaf(q[j], q[j], qq);
        kk = fmaf(k[j], k[j], kk);
    }
    const float rq = rsqrtf(warp_sum(qq) + 1e-6f) * 0.08838834764831845f;
    const float rk = rsqrtf(warp_sum(kk) + 1e-6f);
#pragma unroll
    for (int j = 0; j < 4; ++j) { q[j] *= rq; k[j] *= rk; }
    const float eg = ::expf(g);
    float o4[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float p = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) { s[j][c] *= eg; p = fmaf(s[j][c], k[j], p); }
        const float kv = warp_sum(p);
        const float delta = (b2f(row[voff + h * 128 + c0 + c]) - kv) * beta;
        float o = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) { s[j][c] = fmaf(k[j], delta, s[j][c]); o = fmaf(s[j][c], q[j], o); }
        o4[c] = warp_sum(o);
    }
    if (out != nullptr && lane == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) out[c0 + c] = __float2bfloat16_rn(o4[c]);
    }
}

__device__ __forceinline__ void tile_load(float (&s)[4][4], const float* __restrict__ src, int lane, int c0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(src + (size_t)(lane + 32 * j) * 128 + c0);
        s[j][0] = t.x; s[j][1] = t.y; s[j][2] = t.z; s[j][3] = t.w;
    }
}

__device__ __forceinline__ void tile_store(const float (&s)[4][4], float* __restrict__ dst, int lane, int c0) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
        *reinterpret_cast<float4*>(dst + (size_t)(lane + 32 * j) * 128 + c0) = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
}

// grid (H, 128 / 16), 128 threads: CTA (h, vb) owns columns vb 16 .. vb 16 + 15 of head h's state.
// First the parked rows of the last commit (PRE, *PN of them, the commit folded into this call): the
// entry tile walked over them and written back (the committed state). Then the block's rows: CTL[m] =
// (load, save): load -1 continue from the previous row's tile, -2 the entry state S, >= 0 scratch
// slot; save >= 0 leaves the tile in that slot after the row. WRITE: the last row's tile goes back
// to S (a decode step).
__global__ void __launch_bounds__(128) gdn_rows_kernel(const __nv_bfloat16* __restrict__ QKV, int ldq,
        int koff, int voff, const float* __restrict__ G, const float* __restrict__ BETA,
        float* __restrict__ S, float* __restrict__ STK, const int* __restrict__ CTL, int M,
        __nv_bfloat16* __restrict__ OUT, int H, int REP, int write,
        const __nv_bfloat16* __restrict__ PRE, int ldp, const float* __restrict__ PG, const float* __restrict__ PB,
        const int* __restrict__ PN) {
    pdl_wait();
    pdl_trigger();
    const int h = blockIdx.x, vb = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int c0 = vb * 16 + warp * 4;
    const int hk = h / REP;
    const size_t hs = (size_t)h * 128 * 128;
    float s[4][4];
    const int P = PN != nullptr ? *PN : 0;
    if (P > 0) {
        tile_load(s, S + hs, lane, c0);
        for (int p = 0; p < P; ++p)
            gdn_row(s, PRE + (size_t)p * ldp, koff, voff, PG[(size_t)p * H + h], PB[(size_t)p * H + h], h, hk, lane, c0, nullptr);
        tile_store(s, S + hs, lane, c0);
    }
    for (int m = 0; m < M; ++m) {
        const int ld = CTL[2 * m], sv = CTL[2 * m + 1];
        if (ld != -1 || m == 0)
            tile_load(s, (ld >= 0) ? STK + (size_t)ld * H * 128 * 128 + hs : S + hs, lane, c0);
        gdn_row(s, QKV + (size_t)m * ldq, koff, voff, G[(size_t)m * H + h], BETA[(size_t)m * H + h], h, hk, lane, c0,
                OUT + ((size_t)m * H + h) * 128);
        if (sv >= 0) tile_store(s, STK + (size_t)sv * H * 128 * 128 + hs, lane, c0);
    }
    if (write && M > 0) tile_store(s, S + hs, lane, c0);
}

// --------------------------------------------------------------------------------------- PLE
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = sh[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) t += sh[w];
    return t;
}

// grid (M, hc), 1024 threads: KEY, QRY [M, hc H] (normed), VAL [M, H] -> GV [M, hc H]
__global__ void __launch_bounds__(1024) ple_gate_kernel(const __nv_bfloat16* __restrict__ KEY,
        const __nv_bfloat16* __restrict__ QRY, const __nv_bfloat16* __restrict__ VAL,
        __nv_bfloat16* __restrict__ GV, int Hd, float sqrt_h) {
    __shared__ float sh[32];
    pdl_wait();
    pdl_trigger();
    const int m = blockIdx.x, s = blockIdx.y, hc = gridDim.y;
    const size_t o = ((size_t)m * hc + s) * Hd;
    float acc = 0.f;
    for (int i = threadIdx.x; i < Hd; i += blockDim.x) acc += bfr(b2f(KEY[o + i]) * b2f(QRY[o + i]));
    const float g = bfr(bfr(block_sum(acc, sh)) / sqrt_h);
    float r = bfr(::sqrtf(fmaxf(fabsf(g), 1e-6f)));
    r = g > 0.f ? r : (g < 0.f ? -r : 0.f);
    const float sg = bfr(1.0f / (1.0f + ::expf(-r)));
    for (int i = threadIdx.x; i < Hd; i += blockDim.x)
        GV[o + i] = __float2bfloat16_rn(sg * b2f(VAL[(size_t)m * Hd + i]));
}

// grid (C / 256, M): y = bf16(gv + bf16(silu(bf16(acc)))), acc = fma over taps at distance 9, 6, 3
// (SRC[m][0..2]: a row of the block, or -1 - state column; column 8 is the newest) and the row's
// own normed value, the weights rounded to bf16 by the caller (the reference casts them)
__global__ void __launch_bounds__(256) ple_conv_kernel(const __nv_bfloat16* __restrict__ GVN,
        const __nv_bfloat16* __restrict__ GV, const __nv_bfloat16* __restrict__ PS, int slen,
        const __nv_bfloat16* __restrict__ W, const int* __restrict__ SRC, __nv_bfloat16* __restrict__ Y, int C) {
    pdl_wait();
    pdl_trigger();
    const int m = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int s = SRC[m * 3 + j];
        const float v = s >= 0 ? b2f(GVN[(size_t)s * C + c]) : b2f(PS[(size_t)c * slen + (-1 - s)]);
        acc = fmaf(v, b2f(W[(size_t)c * 4 + j]), acc);
    }
    acc = fmaf(b2f(GVN[(size_t)m * C + c]), b2f(W[(size_t)c * 4 + 3]), acc);
    const float a = bfr(acc);
    const float sl = bfr(a / (1.0f + ::expf(-a)));
    Y[(size_t)m * C + c] = __float2bfloat16_rn(b2f(GV[(size_t)m * C + c]) + sl);
}

// ------------------------------------------------------------------------------------- QSA
// One pooled indexer key per CTA (128 threads = the indexer head dim): the mean of the block's 4 raw
// keys (fp32, in key order, / 4, rounded to bf16), the RMS norm (1 + w), the rope at the block's first
// position (bf16(bf16(n c) + bf16(rot(n) s)) over the first ROT dims). SLOTS[b][0..3]: the cache slots
// of the block's keys; POSB[b]: its first position.
__global__ void __launch_bounds__(128) qsa_pool_kernel(const __nv_bfloat16* __restrict__ IKC,
        const int* __restrict__ SLOTS, const int* __restrict__ POSB, const __nv_bfloat16* __restrict__ KLN,
        const __nv_bfloat16* __restrict__ COS, const __nv_bfloat16* __restrict__ SIN, __nv_bfloat16* __restrict__ OUT,
        int ROT, float eps) {
    __shared__ float sh[4];
    __shared__ float nv[128];
    pdl_wait();
    pdl_trigger();
    const int b = blockIdx.x, t = threadIdx.x;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += b2f(IKC[(size_t)SLOTS[b * 4 + i] * 128 + t]);
    const float v = bfr(acc / 4.0f);
    const float r = rsqrtf(block_sum(v * v, sh) / 128.0f + eps);
    const float n = bfr(v * r * (1.0f + b2f(KLN[t])));
    nv[t] = n;
    __syncthreads();
    float o = n;
    if (t < ROT) {
        const int pos = POSB[b];
        const float c = b2f(COS[(size_t)pos * ROT + t]), sn = b2f(SIN[(size_t)pos * ROT + t]);
        const float rh = t < ROT / 2 ? -nv[t + ROT / 2] : nv[t - ROT / 2];
        o = bfr(bfr(n * c) + bfr(rh * sn));
    }
    OUT[(size_t)b * 128 + t] = __float2bfloat16_rn(o);
}

// The indexer's score of block b for row t: sum over the NH query heads in order of
// relu(q_h . pooled_b) (fp32, lane-strided then a fixed xor tree), / sqrt(128). grid (M, NB / 8), 8
// warps a CTA, a warp a block. QCTL[t] = (complete blocks, first own block): blocks below the
// first own block read the shared pool, the rest the row's own pooled blocks; blocks past the
// complete ones score -inf.
__global__ void __launch_bounds__(256) qsa_scores_kernel(const __nv_bfloat16* __restrict__ IQ,
        const __nv_bfloat16* __restrict__ POOL, const __nv_bfloat16* __restrict__ OWN, int NO,
        const int* __restrict__ QCTL, float* __restrict__ SC, int NB, int NH) {
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x, lane = threadIdx.x & 31;
    const int b = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (b >= NB) return;
    const int nblk = QCTL[t * 4], own0 = QCTL[t * 4 + 1];
    if (b >= nblk) {
        if (lane == 0) SC[(size_t)t * NB + b] = -INFINITY;
        return;
    }
    const __nv_bfloat16* pk = b < own0 ? POOL + (size_t)b * 128 : OWN + ((size_t)t * NO + (b - own0)) * 128;
    float k[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) k[e] = b2f(pk[lane * 4 + e]);
    float tot = 0.f;
    for (int h = 0; h < NH; ++h) {
        const __nv_bfloat16* q = IQ + ((size_t)t * NH + h) * 128 + lane * 4;
        float d = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) d = fmaf(b2f(q[e]), k[e], d);
        tot += fmaxf(warp_sum(d), 0.f);
    }
    if (lane == 0) SC[(size_t)t * NB + b] = tot / sqrtf(128.0f);
}

// The selection of one row (a CTA, 1,024 threads): keys (~ord(score) << 32 | block) over the row's NB
// scores bitonic-sorted ascending in shared memory -- score descending, the lower block first on ties
// (a stable descending sort's order) -- then the first TOPK blocks sorted ascending, 4 positions each,
// then the row's tail. NB a power of two, <= 8,192.
__device__ __forceinline__ uint32_t ord_f(float f) {
    const uint32_t u = __float_as_uint(f + 0.0f);          // -0.0 ties with +0.0, as in a float compare
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(1024) qsa_topk_kernel(const float* __restrict__ SC, int NB,
        const int* __restrict__ TAIL, int* __restrict__ SEL, int nsel, int TOPK) {
    extern __shared__ unsigned long long keys[];
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < NB; i += blockDim.x)
        keys[i] = ((unsigned long long)(~ord_f(SC[(size_t)t * NB + i])) << 32) | (unsigned)i;
    __syncthreads();
    for (int k = 2; k <= NB; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < NB; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const unsigned long long a = keys[i], b = keys[ixj];
                    if (((i & k) == 0) == (a > b)) { keys[i] = b; keys[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
    // the kept blocks, ascending: keys[0 .. TOPK) hold them; sort their block numbers
    unsigned* blk = reinterpret_cast<unsigned*>(keys + TOPK);
    for (int i = threadIdx.x; i < TOPK; i += blockDim.x) blk[i] = (unsigned)(keys[i] & 0xffffffffu);
    __syncthreads();
    for (int k = 2; k <= TOPK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < TOPK; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const unsigned a = blk[i], b = blk[ixj];
                    if (((i & k) == 0) == (a > b)) { blk[i] = b; blk[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
    int* out = SEL + (size_t)t * nsel;
    for (int i = threadIdx.x; i < 4 * TOPK; i += blockDim.x) out[i] = (int)blk[i >> 2] * 4 + (i & 3);
    if (threadIdx.x < 3) out[4 * TOPK + threadIdx.x] = TAIL[t * 3 + threadIdx.x];
}

}  // namespace

static int g_pdl = 1;

template <typename... KArgs, typename... Args>
static void launch(void (*k)(KArgs...), dim3 grid, dim3 block, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = 0; cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...);
    TORCH_CHECK(e == cudaSuccess, "fn_seq_kernels launch: ", cudaGetErrorString(e));
}

void seq_set_pdl(int64_t on) { g_pdl = on ? 1 : 0; }

void seq_gdn_pre(torch::Tensor x, torch::Tensor cs, torch::Tensor w, torch::Tensor src, torch::Tensor y,
                 torch::Tensor ab, torch::Tensor alog, torch::Tensor dtb, torch::Tensor g, torch::Tensor beta, bool shift) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = x.size(0), C = y.size(1), H = alog.numel();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.size(1) >= C, "x bf16 [M, >= C]");
    TORCH_CHECK(cs.dtype() == torch::kBFloat16 && cs.is_contiguous() && cs.numel() == (int64_t)C * 3, "conv state bf16 [C, 3]");
    TORCH_CHECK(w.dtype() == torch::kBFloat16 && w.is_contiguous() && w.numel() == (int64_t)C * 4, "conv weight bf16 [C, 4]");
    TORCH_CHECK(src.dtype() == torch::kInt && src.is_contiguous() && src.numel() == M * 3, "src int32 [M, 3]");
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.is_contiguous() && y.size(0) == M, "y bf16 [M, C]");
    TORCH_CHECK(ab.dtype() == torch::kBFloat16 && ab.stride(1) == 1 && ab.size(1) == 2 * H, "ab bf16 [M, 2 H]");
    TORCH_CHECK(alog.dtype() == torch::kFloat && dtb.dtype() == torch::kFloat && H <= 256, "A_log, dt_bias fp32 [H]");
    TORCH_CHECK(!shift || M == 1, "the state shifts in a one-row call only");
    const int nC = (C + 255) / 256;
    launch(gdn_pre_kernel, dim3(nC + 1, M), dim3(256), st, (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
        (__nv_bfloat16*)cs.data_ptr(), (const __nv_bfloat16*)w.data_ptr(), (const int*)src.data_ptr(),
        (__nv_bfloat16*)y.data_ptr(), (int)y.stride(0), C, nC, (const __nv_bfloat16*)ab.data_ptr(), (int)ab.stride(0),
        (const float*)alog.data_ptr(), (const float*)dtb.data_ptr(), (float*)g.data_ptr(), (float*)beta.data_ptr(),
        H, shift ? 1 : 0);
}

void seq_gdn_rows(torch::Tensor qkv, int64_t koff, int64_t voff, torch::Tensor g, torch::Tensor beta, torch::Tensor S,
                  torch::Tensor stk, torch::Tensor ctl, torch::Tensor out, int64_t rep, bool write,
                  c10::optional<torch::Tensor> pre, c10::optional<torch::Tensor> pg, c10::optional<torch::Tensor> pb,
                  c10::optional<torch::Tensor> pn) {
    const at::cuda::OptionalCUDAGuard guard(qkv.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = qkv.size(0), H = S.size(-3);
    TORCH_CHECK(qkv.dtype() == torch::kBFloat16 && qkv.stride(1) == 1, "qkv bf16 [M, C]");
    TORCH_CHECK(S.dtype() == torch::kFloat && S.is_contiguous() && S.size(-1) == 128 && S.size(-2) == 128, "S fp32 [H, 128, 128]");
    TORCH_CHECK(stk.dtype() == torch::kFloat && stk.is_contiguous(), "scratch fp32");
    TORCH_CHECK(ctl.dtype() == torch::kInt && ctl.is_contiguous() && ctl.numel() == 2 * M, "ctl int32 [M, 2]");
    TORCH_CHECK(g.dtype() == torch::kFloat && beta.dtype() == torch::kFloat && g.numel() == (int64_t)M * H, "g, beta fp32 [M, H]");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.numel() == (int64_t)M * H * 128, "out bf16 [M, H, 128]");
    const bool has_pre = pn.has_value();
    TORCH_CHECK(!has_pre || (pre.has_value() && pg.has_value() && pb.has_value() && pre->dtype() == torch::kBFloat16
                && pre->stride(1) == 1 && pg->dtype() == torch::kFloat && pb->dtype() == torch::kFloat
                && pn->dtype() == torch::kInt), "parked rows: pre bf16 [P, C], pg / pb fp32 [P, H], pn int32 [1]");
    launch(gdn_rows_kernel, dim3(H, 8), dim3(128), st, (const __nv_bfloat16*)qkv.data_ptr(), (int)qkv.stride(0),
        (int)koff, (int)voff, (const float*)g.data_ptr(), (const float*)beta.data_ptr(), (float*)S.data_ptr(),
        (float*)stk.data_ptr(), (const int*)ctl.data_ptr(), M, (__nv_bfloat16*)out.data_ptr(), H, (int)rep, write ? 1 : 0,
        has_pre ? (const __nv_bfloat16*)pre->data_ptr() : (const __nv_bfloat16*)nullptr, has_pre ? (int)pre->stride(0) : 0,
        has_pre ? (const float*)pg->data_ptr() : (const float*)nullptr, has_pre ? (const float*)pb->data_ptr() : (const float*)nullptr,
        has_pre ? (const int*)pn->data_ptr() : (const int*)nullptr);
}

void seq_ple_gate(torch::Tensor key, torch::Tensor qry, torch::Tensor val, torch::Tensor gv, int64_t hc) {
    const at::cuda::OptionalCUDAGuard guard(key.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = val.size(0), Hd = val.size(1);
    TORCH_CHECK(key.is_contiguous() && qry.is_contiguous() && val.is_contiguous() && gv.is_contiguous(), "contiguous");
    TORCH_CHECK(key.numel() == (int64_t)M * hc * Hd && qry.numel() == key.numel() && gv.numel() == key.numel(), "shapes");
    launch(ple_gate_kernel, dim3(M, hc), dim3(1024), st, (const __nv_bfloat16*)key.data_ptr(),
        (const __nv_bfloat16*)qry.data_ptr(), (const __nv_bfloat16*)val.data_ptr(), (__nv_bfloat16*)gv.data_ptr(),
        Hd, (float)std::sqrt((double)Hd));
}

void seq_ple_conv(torch::Tensor gvn, torch::Tensor gv, torch::Tensor ps, torch::Tensor w, torch::Tensor src, torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard guard(gvn.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = gvn.size(0), C = gvn.size(1), slen = ps.size(-1);
    TORCH_CHECK(gvn.is_contiguous() && gv.is_contiguous() && y.is_contiguous() && ps.is_contiguous() && w.is_contiguous(), "contiguous");
    TORCH_CHECK(ps.numel() == (int64_t)C * slen && w.numel() == (int64_t)C * 4 && w.dtype() == torch::kBFloat16, "state [C, slen], w bf16 [C, 4]");
    TORCH_CHECK(src.dtype() == torch::kInt && src.numel() == M * 3, "src int32 [M, 3]");
    launch(ple_conv_kernel, dim3((C + 255) / 256, M), dim3(256), st, (const __nv_bfloat16*)gvn.data_ptr(),
        (const __nv_bfloat16*)gv.data_ptr(), (const __nv_bfloat16*)ps.data_ptr(), slen, (const __nv_bfloat16*)w.data_ptr(),
        (const int*)src.data_ptr(), (__nv_bfloat16*)y.data_ptr(), C);
}

void seq_qsa_pool(torch::Tensor ikc, torch::Tensor slots, torch::Tensor posb, torch::Tensor kln, torch::Tensor cos,
                  torch::Tensor sin, torch::Tensor out, double eps) {
    const at::cuda::OptionalCUDAGuard guard(ikc.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int NB = posb.numel();
    TORCH_CHECK(ikc.dtype() == torch::kBFloat16 && ikc.is_contiguous() && ikc.size(1) == 128, "indexer keys bf16 [cap, 128]");
    TORCH_CHECK(slots.dtype() == torch::kInt && slots.is_contiguous() && slots.numel() == NB * 4, "slots int32 [NB, 4]");
    TORCH_CHECK(posb.dtype() == torch::kInt && posb.is_contiguous(), "posb int32 [NB]");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.numel() == (int64_t)NB * 128, "out bf16 [NB, 128]");
    TORCH_CHECK(cos.dtype() == torch::kBFloat16 && cos.size(1) <= 128, "cos/sin bf16 [L, ROT]");
    if (NB == 0) return;
    launch(qsa_pool_kernel, dim3(NB), dim3(128), st, (const __nv_bfloat16*)ikc.data_ptr(), (const int*)slots.data_ptr(),
        (const int*)posb.data_ptr(), (const __nv_bfloat16*)kln.data_ptr(), (const __nv_bfloat16*)cos.data_ptr(),
        (const __nv_bfloat16*)sin.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (int)cos.size(1), (float)eps);
}

void seq_qsa_topk(torch::Tensor sc, torch::Tensor tail, torch::Tensor sel, int64_t topk) {
    const at::cuda::OptionalCUDAGuard guard(sc.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = sc.size(0), NB = sc.size(1);
    TORCH_CHECK(sc.dtype() == torch::kFloat && sc.is_contiguous() && NB <= 8192 && (NB & (NB - 1)) == 0 && NB >= topk,
                "scores fp32 [M, NB], NB a power of two <= 8192");
    TORCH_CHECK(tail.dtype() == torch::kInt && tail.is_contiguous() && sel.dtype() == torch::kInt && sel.is_contiguous()
                && sel.size(1) >= 4 * topk + 3, "tail int32 [M, 3], sel int32 [M, >= 4 topk + 3]");
    const size_t smem = (size_t)NB * 8 + (size_t)topk * 4;
    cudaFuncSetAttribute(qsa_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(M); cfg.blockDim = dim3(1024); cfg.dynamicSmemBytes = smem; cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, qsa_topk_kernel, (const float*)sc.data_ptr(), NB, (const int*)tail.data_ptr(),
                                       (int*)sel.data_ptr(), (int)sel.size(1), (int)topk);
    TORCH_CHECK(e == cudaSuccess, "qsa_topk launch: ", cudaGetErrorString(e));
}

void seq_qsa_scores(torch::Tensor iq, torch::Tensor pool, torch::Tensor own, torch::Tensor qctl, torch::Tensor sc) {
    const at::cuda::OptionalCUDAGuard guard(iq.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = iq.size(0), NH = iq.size(1), NB = sc.size(1), NO = own.size(1);
    TORCH_CHECK(iq.dtype() == torch::kBFloat16 && iq.is_contiguous() && iq.size(2) == 128, "iq bf16 [M, NH, 128]");
    TORCH_CHECK(pool.is_contiguous() && own.is_contiguous() && own.size(0) == M, "pool [*, 128], own [M, NO, 128]");
    TORCH_CHECK(qctl.dtype() == torch::kInt && qctl.is_contiguous() && qctl.numel() == M * 4, "qctl int32 [M, 4]");
    TORCH_CHECK(sc.dtype() == torch::kFloat && sc.is_contiguous() && sc.size(0) == M, "sc fp32 [M, NB]");
    launch(qsa_scores_kernel, dim3(M, (NB + 7) / 8), dim3(256), st, (const __nv_bfloat16*)iq.data_ptr(),
        (const __nv_bfloat16*)pool.data_ptr(), (const __nv_bfloat16*)own.data_ptr(), NO, (const int*)qctl.data_ptr(),
        (float*)sc.data_ptr(), NB, NH);
}
"""

_CPP = (
    "void seq_set_pdl(int64_t on);\n"
    "void seq_gdn_pre(torch::Tensor x, torch::Tensor cs, torch::Tensor w, torch::Tensor src, torch::Tensor y, "
    "torch::Tensor ab, torch::Tensor alog, torch::Tensor dtb, torch::Tensor g, torch::Tensor beta, bool shift);\n"
    "void seq_gdn_rows(torch::Tensor qkv, int64_t koff, int64_t voff, torch::Tensor g, torch::Tensor beta, "
    "torch::Tensor S, torch::Tensor stk, torch::Tensor ctl, torch::Tensor out, int64_t rep, bool write, "
    "c10::optional<torch::Tensor> pre, c10::optional<torch::Tensor> pg, c10::optional<torch::Tensor> pb, "
    "c10::optional<torch::Tensor> pn);\n"
    "void seq_ple_gate(torch::Tensor key, torch::Tensor qry, torch::Tensor val, torch::Tensor gv, int64_t hc);\n"
    "void seq_ple_conv(torch::Tensor gvn, torch::Tensor gv, torch::Tensor ps, torch::Tensor w, torch::Tensor src, "
    "torch::Tensor y);\n"
    "void seq_qsa_pool(torch::Tensor ikc, torch::Tensor slots, torch::Tensor posb, torch::Tensor kln, torch::Tensor cos, "
    "torch::Tensor sin, torch::Tensor out, double eps);\n"
    "void seq_qsa_scores(torch::Tensor iq, torch::Tensor pool, torch::Tensor own, torch::Tensor qctl, torch::Tensor sc);\n"
    "void seq_qsa_topk(torch::Tensor sc, torch::Tensor tail, torch::Tensor sel, int64_t topk);\n")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        tag = hashlib.sha1((_CPP + _CUDA).encode()).hexdigest()[:10]
        _MOD = load_inline(name=f"fn_seq_kernels_{tag}", cpp_sources=[_CPP], cuda_sources=[_CUDA],
                           functions=["seq_set_pdl", "seq_gdn_pre", "seq_gdn_rows", "seq_ple_gate", "seq_ple_conv",
                                      "seq_qsa_pool", "seq_qsa_scores", "seq_qsa_topk"],
                           extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                           verbose=False)
        _MOD.seq_set_pdl(int(_S.get("FN_PDL") == "1"))
    return _MOD


# ------------------------------------------------------------------------------------- GDN
def gdn_pre(x, conv_state, conv_w, src, ab, a_log, dt_bias, shift: bool = False, out=None):
    """x [M, >= C] bf16 (the qkv projection rows), conv_state [1, C, 3] bf16, conv_w [C, 4] bf16,
    src int32 [M, 3] (the taps at distance 3, 2, 1: a row, or -1 - state column), ab [M, 2 H] bf16,
    a_log / dt_bias fp32 [H] -> (y [M, C] bf16 = silu(conv), g [M, H] fp32, beta [M, H] fp32).
    `shift` (one row): the state advances in place, as the decode step's."""
    M = x.shape[0]
    C = conv_w.shape[0]
    H = a_log.numel()
    y = out if out is not None else torch.empty(M, C, dtype=torch.bfloat16, device=x.device)
    g = torch.empty(M, H, dtype=torch.float32, device=x.device)
    beta = torch.empty(M, H, dtype=torch.float32, device=x.device)
    _module().seq_gdn_pre(x, conv_state, conv_w, src, y, ab, a_log, dt_bias, g, beta, shift)
    return y, g, beta


def gdn_rows(qkv, key_dim: int, g, beta, S, scratch, ctl, rep: int, write: bool, out=None, pre=None):
    """The delta rule over M rows of qkv [M, 2 key_dim + H 128] bf16 (q | k | v after the conv),
    g / beta [M, H] fp32, S [1, H, 128, 128] fp32 (the entry state; written with `write`), scratch
    fp32 [slots, H, 128, 128], ctl int32 [M, 2] (load, save) -> out [M, H, 128] bf16. `pre` = (rows
    [P, C] bf16, g [P, H], beta [P, H], n int32 [1] on the device): n parked rows applied to S first
    and S written back (a commit folded into this call; n = 0 does nothing)."""
    M = qkv.shape[0]
    H = S.shape[-3]
    if out is None:
        out = torch.empty(M, H, 128, dtype=torch.bfloat16, device=qkv.device)
    pr, pg, pb, pn = pre if pre is not None else (None, None, None, None)
    _module().seq_gdn_rows(qkv, key_dim, 2 * key_dim, g, beta, S, scratch, ctl, out, rep, write, pr, pg, pb, pn)
    return out


def gdn_rows_reference(qkv, key_dim: int, g, beta, S, parents: list[int], rep: int):
    """The recurrence in torch over a tree (`parents`, -1 for the entry state), fp32: (out [M, H, 128]
    fp32, the state after each row [M, H, 128, 128])."""
    M = qkv.shape[0]
    H = S.shape[-3]
    q = qkv[:, :key_dim].float().view(M, -1, 128).repeat_interleave(rep, 1)
    k = qkv[:, key_dim:2 * key_dim].float().view(M, -1, 128).repeat_interleave(rep, 1)
    v = qkv[:, 2 * key_dim:2 * key_dim + H * 128].float().view(M, H, 128)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    states, outs = [], []
    for m in range(M):
        s = (S.view(H, 128, 128) if parents[m] < 0 else states[parents[m]]).clone()
        s = s * torch.exp(g[m]).view(H, 1, 1)
        kv = torch.einsum("hkv,hk->hv", s, k[m])
        delta = (v[m] - kv) * beta[m].view(H, 1)
        s = s + k[m].unsqueeze(-1) * delta.unsqueeze(1)
        outs.append(torch.einsum("hkv,hk->hv", s, q[m]))
        states.append(s)
    return torch.stack(outs), torch.stack(states)


def tree_ctl(parents: list[int]) -> tuple[list[tuple[int, int]], int]:
    """The (load, save) of every row of a DFS pre-order block for `gdn_rows`, and the scratch slots it
    uses: a row continues from the previous row's tile when that row is its parent, starts from the
    entry state when it is a root, and otherwise loads its parent's tile from the slot the parent
    saved it in (a parent saves when it has a child other than the next row)."""
    M = len(parents)
    need = [False] * M
    for i in range(1, M):
        p = parents[i]
        if p >= 0 and p != i - 1:
            need[p] = True
    slot, n = {}, 0
    for i in range(M):
        if need[i]:
            slot[i] = n
            n += 1
    out = []
    for i, p in enumerate(parents):
        if i == 0 or p < 0:
            ld = -2
        elif p == i - 1:
            ld = -1
        else:
            ld = slot[p]
        out.append((ld, slot.get(i, -1)))
    return out, n


def window_src(parents: list[int], dists: tuple[int, ...], state_len: int) -> list[list[int]]:
    """For every row of a block, where the position `d` before it lives, for each d in `dists`: the
    row of its ancestor at that distance (>= 0), or -1 - the column of the state ([.., state_len]
    columns, the last the newest) when the walk leaves the block."""
    out = []
    for i in range(len(parents)):
        anc = [i]
        while parents[anc[-1]] >= 0:
            anc.append(parents[anc[-1]])
        row = []
        for d in dists:
            if d < len(anc):
                row.append(anc[d])
            else:
                row.append(-1 - (state_len - (d - len(anc) + 1)))
        out.append(row)
    return out


# ------------------------------------------------------------------------------------- PLE
def ple_gate(key, qry, val, hc: int):
    """key, qry [M, hc H] bf16 (normed), val [M, H] bf16 -> gv [M, hc H] bf16."""
    gv = torch.empty_like(key)
    _module().seq_ple_gate(key, qry, val, gv, hc)
    return gv


def ple_conv(gvn, gv, state, w_bf16, src):
    """gvn, gv [M, C] bf16, state [1, C, slen] bf16, w [C, 4] bf16, src int32 [M, 3] (distance 9, 6, 3)
    -> gv + silu(conv) [M, C] bf16."""
    y = torch.empty_like(gv)
    _module().seq_ple_conv(gvn, gv, state, w_bf16, src, y)
    return y


# ------------------------------------------------------------------------------------- QSA
#: blocks a row may own past the shared pool: a block that holds one of the round's positions is
#: pooled per row from the row's own path (a verify block is at most 32 rows past its start)
OWN = 9
#: the selection: budget / compress ratio blocks of 4 keys
TOPK = 512


def qsa_pool(ikc, slots, posb, kln, cos, sin, eps: float, out=None):
    """Pooled indexer keys: ikc [cap, 128] bf16 (raw keys), slots int32 [NB, 4] (the block's key
    slots), posb int32 [NB] (its first position), kln [128] (k_layernorm, 1 + w), cos / sin [L, ROT]
    -> [NB, 128] bf16 (mean, norm, rope at the block's first position)."""
    NB = posb.numel()
    if out is None:
        out = torch.empty(NB, 128, dtype=torch.bfloat16, device=ikc.device)
    _module().seq_qsa_pool(ikc, slots, posb, kln, cos, sin, out, eps)
    return out


def qsa_scores(iq, pool, own, qctl, NB: int, out=None):
    """Block scores of M rows: iq [M, NH, 128] bf16 (normed, rotated), pool [*, 128] (committed
    blocks), own [M, OWN, 128] (the row's blocks from qctl[:, 1] on), qctl int32 [M, 4] (complete
    blocks, first own block) -> fp32 [M, NB], -inf past a row's complete blocks."""
    M = iq.shape[0]
    if out is None:
        out = torch.empty(M, NB, dtype=torch.float32, device=iq.device)
    _module().seq_qsa_scores(iq, pool, own, qctl, out)
    return out


def qsa_select(sc, tail, sel):
    """The visible positions of each row from its block scores: the TOPK best blocks (ties to the
    lower block: a stable sort), in ascending order, 4 positions each, then the row's tail positions
    (tail int32 [M, 3], the positions past the last complete block; `actl` holds how many are real)
    -> written into sel int32 [M, >= 4 TOPK + 3]. Fixed shapes, no host sync. One kernel a call up to
    8,192 blocks (32k of context), torch sorts past it (and on the CPU); the same selection."""
    if sc.is_cuda and sc.shape[1] <= 8192:
        _module().seq_qsa_topk(sc, tail, sel, TOPK)
        return sel
    return qsa_select_torch(sc, tail, sel)


def qsa_select_torch(sc, tail, sel):
    """`qsa_select` in torch ops (the reference for the kernel; contexts past 32k)."""
    M = sc.shape[0]
    top = torch.sort(sc, dim=1, descending=True, stable=True).indices[:, :TOPK]
    blk = torch.sort(top, dim=1).values
    pos = (blk.unsqueeze(-1) * 4 + torch.arange(4, device=sc.device)).view(M, 4 * TOPK)
    sel[:, :4 * TOPK].copy_(pos)
    sel[:, 4 * TOPK:4 * TOPK + 3].copy_(tail)
    return sel


def qsa_reference(iq, keys, kln, cos, sin, p: int, eps: float):
    """The reference's selection for one query at position p over raw keys [>p, 128] (engine/models/
    flashnext/ref.py qsa_select's arithmetic in torch): the visible positions in ascending order."""
    ratio = 4
    nblk = (p + 1) // ratio
    if nblk <= TOPK:
        return torch.arange(p + 1, device=keys.device)
    g = keys[: nblk * ratio].view(nblk, ratio, -1).float().mean(1).to(torch.bfloat16).float()
    g = (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps) * (1 + kln.float())).to(torch.bfloat16)
    starts = torch.arange(nblk, device=keys.device) * ratio
    c, s = cos[starts], sin[starts]
    d = c.shape[-1]
    xr = g[:, :d]
    rot = torch.cat([-xr[:, d // 2:], xr[:, :d // 2]], -1)
    g = torch.cat([((xr * c).to(torch.bfloat16) + (rot * s).to(torch.bfloat16)).to(torch.bfloat16), g[:, d:]], -1)
    sc = torch.relu(iq.float() @ g.float().T).sum(0) / 128 ** 0.5
    sel = torch.sort(torch.sort(sc, descending=True, stable=True).indices[:TOPK]).values
    tok = (sel.unsqueeze(-1) * ratio + torch.arange(ratio, device=keys.device)).flatten()
    return torch.cat([tok, torch.arange(nblk * ratio, p + 1, device=keys.device)])


# ------------------------------------------------------------------------------- attention
if HAVE_TRITON:

    @triton.jit
    def _attn_rows_split(Q, K, V, CTL, ANCT, SEL, PM, PL, PACC,
                         s_kh, s_kn, s_vh, s_vn,
                         SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr,
                         HK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, CHUNK: tl.constexpr,
                         NA: tl.constexpr, NSEL: tl.constexpr, NS: tl.constexpr):
        # CTL[t] = (visible keys n, block start L, dense) int32
        t = tl.program_id(0)
        kvh = tl.program_id(1)
        sp = tl.program_id(2)
        n = tl.load(CTL + t * 4)
        L = tl.load(CTL + t * 4 + 1)
        dense = tl.load(CTL + t * 4 + 2)
        rows = tl.arange(0, BM)
        rmask = rows < REP
        qh = kvh * REP + rows
        od = tl.arange(0, D)
        q = tl.load(Q + (t * HQ + qh[:, None]) * D + od[None, :], mask=rmask[:, None], other=0.0)
        c0 = sp * CHUNK
        c1 = tl.minimum(c0 + CHUNK, n)
        m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for n0 in range(c0, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            sel = tl.load(SEL + t * NSEL + cols, mask=cmask & (dense == 0), other=0)
            pos = tl.where(dense != 0, cols, sel)
            tail = pos >= L
            a = tl.load(ANCT + t * NA + (pos - L), mask=cmask & tail, other=0)
            slot = tl.where(tail, a, pos)
            k = tl.load(K + kvh * s_kh + slot[:, None] * s_kn + od[None, :], mask=cmask[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k))
            s = s * SCALE
            vis = cmask[None, :] & rmask[:, None]
            s = tl.where(vis, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + kvh * s_vh + slot[:, None] * s_vn + od[None, :], mask=cmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        base = ((t * HK + kvh) * NS + sp) * BM + rows
        tl.store(PM + base, m_i)
        tl.store(PL + base, l_i)
        tl.store(PACC + base[:, None] * D + od[None, :], acc)

    @triton.jit
    def _attn_rows_combine(PM, PL, PACC, OUT, REP: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr,
                           HK: tl.constexpr, BM: tl.constexpr, NS: tl.constexpr):
        t = tl.program_id(0)
        qh = tl.program_id(1)
        kvh = qh // REP
        r = qh % REP
        od = tl.arange(0, D)
        b0 = (t * HK + kvh) * NS * BM + r
        mx = tl.load(PM + b0)
        for s in tl.static_range(1, NS):
            mx = tl.maximum(mx, tl.load(PM + b0 + s * BM))
        mx = tl.where(mx == -float("inf"), 0.0, mx)
        num = tl.zeros([D], dtype=tl.float32)
        den = 0.0
        # in chunk order: the chunks past a row's last key add exact zeros
        for s in tl.static_range(0, NS):
            w = tl.exp(tl.load(PM + b0 + s * BM) - mx)
            num = num + tl.load(PACC + (b0 + s * BM) * D + od) * w
            den = den + tl.load(PL + b0 + s * BM) * w
        o = num / tl.where(den == 0.0, 1.0, den)
        tl.store(OUT + (t * HQ + qh) * D + od, o.to(tl.bfloat16))


def attn_rows(q, kc, vc, ctl, anc, sel, bufs: dict | None = None):
    """q [M, Hq, D] bf16 (normed, rotated); kc / vc [Hk, cap, D] bf16 (the caches); ctl int32 [M, 4]
    (visible keys, block start L, dense flag, -); anc int32 [M, ANC] (the cache slot of the row's
    ancestor at each depth, 0 the block's first row); sel int32 [M, NSEL] (QSA: the visible
    positions in ascending order; read only for rows not dense) -> o [M, Hq, D] bf16."""
    M, HQ, D = q.shape
    HK = kc.shape[0]
    REP = HQ // HK
    BM = 16
    assert REP <= BM and D == kc.shape[2]
    b = bufs if bufs is not None else {}
    pm = b.get("pm")
    if pm is None or pm.shape[0] != M * HK * ATTN_SPLITS * BM:
        pm = torch.empty(M * HK * ATTN_SPLITS * BM, dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty(pm.shape[0], D, dtype=torch.float32, device=q.device)
        if bufs is not None:
            bufs.update(pm=pm, pl=pl, pacc=pacc)
    else:
        pl, pacc = b["pl"], b["pacc"]
    out = torch.empty(M, HQ, D, dtype=torch.bfloat16, device=q.device)
    _attn_rows_split[(M, HK, ATTN_SPLITS)](
        q, kc, vc, ctl, anc, sel, pm, pl, pacc,
        kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1),
        SCALE=D ** -0.5, REP=REP, D=D, HQ=HQ, HK=HK, BM=BM, BN=32, CHUNK=ATTN_CHUNK,
        NA=anc.shape[1], NSEL=sel.shape[1], NS=ATTN_SPLITS, num_warps=4, num_stages=2)
    _attn_rows_combine[(M, HQ)](pm, pl, pacc, out, REP=REP, D=D, HQ=HQ, HK=HK, BM=BM, NS=ATTN_SPLITS,
                                num_warps=4)
    return out


def attn_rows_reference(q, kc, vc, ctl, anc, sel):
    """`attn_rows` in torch, fp32, from the same visible-key lists."""
    M, HQ, D = q.shape
    HK = kc.shape[0]
    rep = HQ // HK
    out = []
    for t in range(M):
        n, L, dense = (int(x) for x in ctl[t, :3])
        pos = torch.arange(n, device=q.device) if dense else sel[t, :n].long()
        slot = torch.where(pos >= L, anc[t].long()[(pos - L).clamp(min=0, max=anc.shape[1] - 1)], pos)
        k = kc[:, slot].float().repeat_interleave(rep, 0)
        v = vc[:, slot].float().repeat_interleave(rep, 0)
        s = torch.einsum("hd,hnd->hn", q[t].float(), k) * D ** -0.5
        out.append(torch.einsum("hn,hnd->hd", torch.softmax(s, -1), v))
    return torch.stack(out)
