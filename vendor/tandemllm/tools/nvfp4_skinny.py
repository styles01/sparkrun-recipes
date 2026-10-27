"""The W4A16 product for 1..32 rows as a CUDA kernel that feeds the tensor cores from registers.

`tools/nvfp4_linear_v2.py` reads the verify's 13.7 GB of NVFP4 projections at about 169 GB/s in
the engine and 194 GB/s cold, against a practical ceiling of about 235 GB/s on this board. Triton
decodes each weight tile into a blocked register layout and then has to move it into the tensor
core's operand layout, which it does through shared memory, for every 128-wide K step of every
tile. This kernel never does that, because of one observation about the stored format:

    `cvt.rn.f16x2.e2m1x2` turns ONE packed byte into ONE f16x2 register holding (k, k+1), the low
    half being the even element -- which is exactly how an `mma.m16n8k16` operand register holds
    two consecutive K elements.

So a thread can load 16 contiguous bytes of a weight row with one vector load and decode each byte
straight into an operand register. The only thing that has to agree is which logical K each of the
instruction's sixteen K slots stands for, and a dot product does not care about the order of its
terms, so the slots are assigned to match the bytes a thread already holds:

    lane (g, t) of a warp holds bytes 16t .. 16t+15 of a 64-byte row chunk (128 logical K).
    In the chunk's j-th k16 instruction (j = 0..7) its B registers are bytes 16t+2j and 16t+2j+1,
    i.e. logical K 32t+4j .. 32t+4j+3; its A registers are the ACTIVATION at the same four K,
    which are four consecutive bf16 of the activation row -- also a contiguous load.

The weights are used as the B operand (8 weight rows per instruction) and the activations as A
(16 rows per instruction), so a verify block of up to 16 rows is one instruction per weight tile
and 17..32 rows are two. The stored layout does not change: the prefill paths, the v1/v2 kernels
and the dequantiser read the same bytes they always have, and nothing is duplicated in memory.

Scales. Group j of a row covers logical K 16j..16j+15, eight bytes. Lane t's sixteen bytes are
groups 2t and 2t+1 of the chunk, instructions 0..3 use the first and 4..7 the second, and both
scales arrive in one 16-bit load, decoded by `cvt.rn.f16x2.e4m3x2`. The scale multiplies the
decoded weight in fp16 before the product, exactly as v2 does (exact: e2m1 x e4m3 needs six
significand bits and fp16 has eleven), and the per-tensor scale multiplies the fp32 accumulator.

Determinism. The reduction order over K is a function of the weight's shape only -- the tile table
below is keyed by (N, K) -- never of the row count, so a row's result is the same bits whether it
is verified alone or in a block of sixteen, which is what the losslessness gate compares. K is split
across the warps of a CTA and summed in warp order through shared memory; there are no atomics.

Built on first use with `torch.utils.cpp_extension.load_inline` for sm_121a (the e2m1 converter is
an arch-specific instruction), cached under ~/.cache/torch_extensions by source hash.
"""

from __future__ import annotations

import os
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in __import__("sys").path:
    # run as a script from tools/: the repo root, appended (lowest priority), for engine.settings
    __import__("sys").path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# QWEN38_NVFP4_SKINNY=1 routes every NVFP4 projection of 1..32 rows here instead of to v2.
SKINNY = _S.get("NVFP4_SKINNY") == "1"
SKINNY_MAX = 32
# launch with programmatic dependent launch -- the grid starts while the kernel before it
# (an RMS norm that releases its dependents at once, tools/norm_kernels.py) still runs, loads its
# first weights, and waits for the activation. The same loads in the same order: the same bits.
PDL = _S.get("SKINNY_PDL") == "1"
# the weight loads' L2 hint, compiled in (one build a value; see `ld_w`). 0 = as shipped.
LDW = int(_S.get("SKINNY_LDW"))
# /, TIMING ONLY -- the output is wrong with either on. XSTUB: the activation loads
# return a constant, so the compiler drops the activation's registers and loads (the ceiling a
# kernel that kept the activation out of the lane's registers could reach). SSTUB: the scale loads
# return a constant (the ceiling of a perfect scale stream). Never set in a served environment.
XSTUB = int(_S.get("SKINNY_XSTUB"))
SSTUB = int(_S.get("SKINNY_SSTUB"))
if XSTUB or SSTUB:
    print(f"[skinny] TIMING-ONLY BUILD: XSTUB={XSTUB} SSTUB={SSTUB} -- the output is wrong",
          flush=True)
# read the scales from a copy laid out in runs of 16 rows x one 128-wide K step (8 bytes a
# row, 128 contiguous bytes; `scale_runs`) instead of 8 bytes in each of 8 rows a scale-row apart.
# The same bytes in the same registers: the same bits. Compiled in (one build a value, like LDW);
# the copy is made on a weight's first skinny call and costs 1/9 of its bytes.
SRUN = int(_S.get("SKINNY_SRUN"))

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#ifndef XSTUB
#define XSTUB 0
#endif
#ifndef SSTUB
#define SSTUB 0
#endif
#ifndef SRUN
#define SRUN 0
#endif
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint4 ld_w(const uint8_t* p) {
    // weights are read once per step: keep them out of L1 so the activation stays there.
    // LDW_HINT (QWEN38_SKINNY_LDW): 1 marks them first to go from L2 as well (a 15 GB
    // stream through a 24 MiB L2 has nothing to reuse; what should stay is the activations and the
    // recurrent state), 2 asks the L2 for 256-byte lines on the miss, 3 both. Same bytes, same bits.
    uint4 r;
#if LDW_HINT == 1 || LDW_HINT == 3
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
#endif
#if LDW_HINT == 1
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
#elif LDW_HINT == 2
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
#elif LDW_HINT == 3
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
#else
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
#endif
    return r;
}

__device__ __forceinline__ uint32_t ld_s(const uint8_t* p) {
#if SSTUB
    return 0x3838u;                       // timing only: two e4m3 1.0 scales, no load
#else
    unsigned short v;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
#endif
}

__device__ __forceinline__ uint4 ld_x(const void* p) {
#if XSTUB
    return make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u);  // timing only
#else
    return __ldg(reinterpret_cast<const uint4*>(p));
#endif
}

// four packed bytes -> four f16x2 registers, byte i -> register i, low half = low nibble
__device__ __forceinline__ void dec4(uint32_t w, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                     uint32_t& r3) {
    asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
        " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        " cvt.rn.f16x2.e2m1x2 %2, b2;\n cvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(w));
}

// two e4m3 scales (low byte first) -> f16x2
__device__ __forceinline__ uint32_t dec_s(uint32_t s) {
    uint32_t r;
    unsigned short h = (unsigned short)s;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(h));
    return r;
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// bf16x2 -> f16x2, round to nearest (the same conversion as `x.to(float16)` through fp32)
__device__ __forceinline__ uint32_t bf2h(uint32_t u) {
    float lo = __uint_as_float(u << 16), hi = __uint_as_float(u & 0xffff0000u);
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Y[M, N] = (X[M, K] @ W[N, K]^T) * s2, M <= 16 * MT.
// grid.x = ceil(N / (8 NT)); block = 32 WK: WK warps share the CTA's 8 NT weight rows and split
// its 128-wide K steps into contiguous slices, summed in warp order at the end.
// PF: 0 no prefetch; 1 the next step's weights, scales and activation rows in registers; 2 the
// weights and scales only (the activation is an L1 hit, and its double buffer is 32 registers a
// row). MINB asks the compiler for registers that fit that many CTAs on an SM: at 166 registers
// and 256 threads one CTA fits, and a grid of 160 CTAs runs as 3.3 waves.
// IL: 0, warp w sums the contiguous steps [w per, (w + 1) per); 1, the steps w, w + WK, w + 2 WK, ..
// so the CTA's warps read neighbouring 64-byte chunks of each row at the same time. Either
// way the order is a function of (K, WK, IL) only, never of M or NT: an 8-row N tile (NT = 1) sums
// a row exactly as a 16-row one does, and a row is the same bits alone or in a block.
// KR (17..32 rows): the products in the order r (weight register) -> i (N tile) -> a (row
// tile) instead of i -> r -> a, with each register's activation (one 16-byte vector a row) loaded right
// before its products. Weight register r of a K step pairs with exactly one activation vector per row:
// a lane holds 2 MT vectors at a time instead of 4 x 2 MT (the wide tile spills at MT 2 holding them
// all). Every acc[a][i] still receives (r0 j0, r0 j1, r1 j0, ...) with the same values: the same bits.
// SPW (47, 2026-09-26): one warp sums SPW consecutive slices of the K split, each from zero and
// kept apart (in shared memory), and warp 0 adds them in slice order, an empty slice adding 0.f -- the
// served reduction, op for op. A CTA then needs ceil(nonempty slices / SPW) warps (7 for K = 5120,
// whose 16 slices are 13 of 3 steps, one of 1 and two empty) and two CTAs fit an SM, so one streams
// while the other reduces. SPW = 1 is the kernel as it was.
template <int NT, int MT, int WK, int PF, int MINB, int IL = 0, int KR = 0, int SPW = 1>
__global__ void __launch_bounds__(32 * WK / SPW, MINB)
skinny_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W,
              const uint8_t* __restrict__ S, const float* __restrict__ S2V, float s2,
              __nv_bfloat16* __restrict__ Y, int M, int N, int KQ,
              int ldx, int ldw, int lds, int ldy, int pdl) {
    extern __shared__ float red[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * (8 * NT);
    const int per = (KQ + WK - 1) / WK;
    const int qs = IL ? WK : 1;
    const int q0 = IL ? warp : warp * SPW * per, q1 = IL ? KQ : min(KQ, q0 + SPW * per);

    // this lane's weight rows (clamped: a row past N is read and never stored)
    const uint8_t* wrow[NT];
    const uint8_t* srow[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int r = min(n0 + 8 * i + g, N - 1);
        wrow[i] = W + (size_t)r * ldw + 16 * t;
#if SRUN
        srow[i] = S + ((size_t)(r >> 4) * KQ * 16 + (r & 15)) * 8 + 2 * t;
#else
        srow[i] = S + (size_t)r * lds + 2 * t;
#endif
    }
    // this lane's activation rows: g and g + 8 of each 16-row tile
    const __nv_bfloat16* xrow[2 * MT];
    bool xon[2 * MT];
#pragma unroll
    for (int m = 0; m < 2 * MT; ++m) {
        int r = (m >> 1) * 16 + (m & 1) * 8 + g;
        xon[m] = r < M;
        xrow[m] = X + (size_t)(xon[m] ? r : 0) * ldx + 32 * t;
    }

    float acc[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][i][c] = 0.f;

    constexpr int NB = PF ? 2 : 1;             // weight buffers
    constexpr int XB = PF == 1 ? 2 : 1;        // activation buffers
    uint4 wb[NB][NT];
    uint32_t sb[NB][NT];
    uint4 xb[XB][2 * MT][4];

    auto load_x = [&](int buf, int q) {
#pragma unroll
        for (int m = 0; m < 2 * MT; ++m)
#pragma unroll
            for (int v = 0; v < 4; ++v)
                xb[buf][m][v] = xon[m] ? ld_x(xrow[m] + (size_t)q * 128 + 8 * v)
                                       : make_uint4(0, 0, 0, 0);
    };
    auto load = [&](int buf, int q) {
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            wb[buf][i] = ld_w(wrow[i] + (size_t)q * 64);
            sb[buf][i] = ld_s(srow[i] + (size_t)q * (SRUN ? 128 : 8));
        }
        if (PF != 2 && !KR) load_x(XB == 2 ? buf : 0, q);
    };

    auto compute = [&](int buf, int q) {
        if (KR) {
            uint32_t s_lo[NT], s_hi[NT];
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                uint32_t sc = dec_s(sb[buf][i]);
                s_lo[i] = __byte_perm(sc, sc, 0x1010);
                s_hi[i] = __byte_perm(sc, sc, 0x3232);
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                uint32_t xr[2 * MT][4];
#pragma unroll
                for (int m = 0; m < 2 * MT; ++m) {
                    const uint4 u = xon[m] ? ld_x(xrow[m] + (size_t)q * 128 + 8 * r)
                                           : make_uint4(0, 0, 0, 0);
                    xr[m][0] = bf2h(u.x); xr[m][1] = bf2h(u.y);
                    xr[m][2] = bf2h(u.z); xr[m][3] = bf2h(u.w);
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    const uint32_t w = r == 0 ? wb[buf][i].x : r == 1 ? wb[buf][i].y
                                     : r == 2 ? wb[buf][i].z : wb[buf][i].w;
                    uint32_t d0, d1, d2, d3;
                    dec4(w, d0, d1, d2, d3);
                    const uint32_t s = r < 2 ? s_lo[i] : s_hi[i];
                    d0 = hmul2(d0, s); d1 = hmul2(d1, s); d2 = hmul2(d2, s); d3 = hmul2(d3, s);
#pragma unroll
                    for (int a = 0; a < MT; ++a) {
                        mma(acc[a][i], xr[2 * a][0], xr[2 * a + 1][0], xr[2 * a][1],
                            xr[2 * a + 1][1], d0, d1);
                        mma(acc[a][i], xr[2 * a][2], xr[2 * a + 1][2], xr[2 * a][3],
                            xr[2 * a + 1][3], d2, d3);
                    }
                }
            }
            return;
        }
        if (PF == 2) load_x(0, q);
        const int xbuf = XB == 2 ? buf : 0;
        // activation operands: 16 f16x2 per row, pair p = logical K 32t + 2p, 2p + 1
        uint32_t xa[2 * MT][16];
#pragma unroll
        for (int m = 0; m < 2 * MT; ++m)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                xa[m][4 * v + 0] = bf2h(xb[xbuf][m][v].x);
                xa[m][4 * v + 1] = bf2h(xb[xbuf][m][v].y);
                xa[m][4 * v + 2] = bf2h(xb[xbuf][m][v].z);
                xa[m][4 * v + 3] = bf2h(xb[xbuf][m][v].w);
            }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t sc = dec_s(sb[buf][i]);
            uint32_t s_lo = __byte_perm(sc, sc, 0x1010), s_hi = __byte_perm(sc, sc, 0x3232);
            uint32_t wr[4] = {wb[buf][i].x, wb[buf][i].y, wb[buf][i].z, wb[buf][i].w};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                uint32_t d0, d1, d2, d3;
                dec4(wr[r], d0, d1, d2, d3);
                const uint32_t s = r < 2 ? s_lo : s_hi;       // bytes 0..7 group 2t, 8..15 2t+1
                d0 = hmul2(d0, s); d1 = hmul2(d1, s); d2 = hmul2(d2, s); d3 = hmul2(d3, s);
                // register r holds bytes 4r..4r+3 = instructions j = 2r (bytes 4r, 4r+1) and 2r+1
#pragma unroll
                for (int a = 0; a < MT; ++a) {
                    const int j0 = 2 * r, j1 = 2 * r + 1;
                    mma(acc[a][i], xa[2 * a][2 * j0], xa[2 * a + 1][2 * j0],
                        xa[2 * a][2 * j0 + 1], xa[2 * a + 1][2 * j0 + 1], d0, d1);
                    mma(acc[a][i], xa[2 * a][2 * j1], xa[2 * a + 1][2 * j1],
                        xa[2 * a][2 * j1 + 1], xa[2 * a + 1][2 * j1 + 1], d2, d3);
                }
            }
        }
    };

    // SPW > 1: a slice ends after step q -> its partial goes to shared memory and the next starts at 0
    constexpr int R = MT * NT * 4;
    auto slice_done = [&](int q) {
        if constexpr (SPW > 1) {
            if ((q + 1) % per == 0 || q + 1 == q1) {
                const int sl = q / per;
#pragma unroll
                for (int a = 0; a < MT; ++a)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            red[(sl * R + (a * NT + i) * 4 + c) * 32 + lane] = acc[a][i][c];
                            acc[a][i][c] = 0.f;
                        }
            }
        }
    };

    //, programmatic dependent launch: this grid may start while the kernel before it runs.
    // Nothing of the activation may be read until that kernel is done; the weights and scales are
    // not its output, so the first step's weights (PF = 2 loads no activation with them) are in
    // flight before the wait. Without the launch attribute the wait returns at once.
    if (PF != 2 && pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
    if (PF) {
        // two steps an iteration so every buffer index is a constant: a register array indexed
        // at run time is a local-memory array. Steps are still consumed in order, so PF = 0 and
        // PF = 1 sum K in the same order and give the same bits.
        int q = q0;
        if (q < q1) load(0, q);
        if (PF == 2 && pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
        for (; q + qs < q1; q += 2 * qs) {
            load(1, q + qs);
            compute(0, q);
            slice_done(q);
            if (q + 2 * qs < q1) load(0, q + 2 * qs);
            compute(1, q + qs);
            slice_done(q + qs);
        }
        if (q < q1) { compute(0, q); slice_done(q); }
    } else {
        for (int q = q0; q < q1; q += qs) {
            load(0, q);
            compute(0, q);
            slice_done(q);
        }
    }

    // the next projection may begin its own weight prologue while this one reduces and stores
    if (pdl) asm volatile("griddepcontrol.launch_dependents;");
    if constexpr (SPW > 1) {
        __syncthreads();
        if (warp > 0) return;
#pragma unroll
        for (int a = 0; a < MT; ++a)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    acc[a][i][c] = red[((a * NT + i) * 4 + c) * 32 + lane];
        for (int w = 1; w < WK; ++w) {
            const bool on = w * per < KQ;
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        acc[a][i][c] += on ? red[(w * R + (a * NT + i) * 4 + c) * 32 + lane] : 0.f;
        }
    } else if (WK > 1) {
        if (warp > 0) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        red[((warp - 1) * R + (a * NT + i) * 4 + c) * 32 + lane] = acc[a][i][c];
        }
        __syncthreads();
        if (warp > 0) return;
        for (int w = 1; w < WK; ++w)
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        acc[a][i][c] += red[((w - 1) * R + (a * NT + i) * 4 + c) * 32 + lane];
    }

    // D[m][n]: c0, c1 = row g, columns 2t, 2t+1; c2, c3 = row g + 8
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            const int n = n0 + 8 * i + 2 * t;
            if (n >= N) continue;
            const float sa = S2V ? S2V[n] : s2, sb2 = S2V ? S2V[min(n + 1, N - 1)] : s2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = a * 16 + h * 8 + g;
                if (m >= M) continue;
                __nv_bfloat16* y = Y + (size_t)m * ldy + n;
                const __nv_bfloat16 v0 = __float2bfloat16_rn(acc[a][i][2 * h] * sa);
                if (n + 1 < N) {
                    const __nv_bfloat16 v1 = __float2bfloat16_rn(acc[a][i][2 * h + 1] * sb2);
                    __nv_bfloat162 p; p.x = v0; p.y = v1;
                    *reinterpret_cast<__nv_bfloat162*>(y) = p;
                } else {
                    *y = v0;
                }
            }
        }
}

template <int NT, int MT, int WK, int PF, int MINB, int IL = 0, int KR = 0, int SPW = 1>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s,
            const float* s2v, float s2, torch::Tensor& y, int pdl) {
    const int M = x.size(0), N = w.size(0), K = w.size(1) * 2;
    const int smem = SPW > 1 ? WK * MT * NT * 4 * 32 * 4
                             : (WK > 1 ? (WK - 1) * MT * NT * 4 * 32 * 4 : 0);
    // SPW > 1: the warps the non-empty slices need, SPW a warp
    const int kq_ = K / 128, per_ = (kq_ + WK - 1) / WK;
    const int warps = SPW > 1 ? ((kq_ + per_ - 1) / per_ + SPW - 1) / SPW : WK;
    // the K-split partials live in shared memory: 99 KB an SM on this board
    TORCH_CHECK(smem <= 99 * 1024, "skinny tile nt=", NT, " wk=", WK, " mt=", MT, " needs ",
                smem, " bytes of shared memory for its K-split partials; the SM has 101376");
    auto kern = skinny_kernel<NT, MT, WK, PF, MINB, IL, KR, SPW>;
    static bool attr = false;
    if (!attr && smem > 48 * 1024) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    attr = true;
    dim3 grid((N + 8 * NT - 1) / (8 * NT));
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const uint8_t* wp = w.data_ptr<uint8_t>();
    const uint8_t* sp = reinterpret_cast<const uint8_t*>(s.data_ptr());
    __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
    const int kq = K / 128, lx = x.stride(0), lw = w.stride(0), ls = s.stride(0), ly = y.stride(0);
    if (pdl) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = dim3(32 * warps);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = at::cuda::getCurrentCUDAStream();
        cudaLaunchAttribute la[1];
        la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        la[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = la;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, kern, xp, wp, sp, s2v, s2, yp, M, N, kq, lx, lw, ls, ly, pdl);
    } else {
        kern<<<grid, 32 * warps, smem, at::cuda::getCurrentCUDAStream()>>>(
            xp, wp, sp, s2v, s2, yp, M, N, kq, lx, lw, ls, ly, pdl);
    }
}

#define WK4(NT, MT, PF, MINB)                                                             \
    switch (wk) {                                                                         \
        case 1: launch<NT, MT, 1, PF, MINB>(x, w, s, p, s2, y, pdl); return;                  \
        case 2: launch<NT, MT, 2, PF, MINB>(x, w, s, p, s2, y, pdl); return;                  \
        case 4: launch<NT, MT, 4, PF, MINB>(x, w, s, p, s2, y, pdl); return;                  \
        case 8: launch<NT, MT, 8, PF, MINB>(x, w, s, p, s2, y, pdl); return;                  \
    }
// sixteen K-split warps (512 threads) only with one CTA an SM in mind
#define WK5(NT, MT, PF)                                                                   \
    if (wk == 16) { launch<NT, MT, 16, PF, 1>(x, w, s, p, s2, y, pdl); return; }             \
    WK4(NT, MT, PF, 1)
// the interleaved K split, for the few tiles the down / out / o sweep asks for
#define IL1(NT, MT, PF)                                                                   \
    if (wk == 8) { launch<NT, MT, 8, PF, 1, 1>(x, w, s, p, s2, y, pdl); return; }            \
    if (wk == 16) { launch<NT, MT, 16, PF, 1, 1>(x, w, s, p, s2, y, pdl); return; }


//, the slice-serial tile (`ser`). The K split above (WK slices of contiguous 128-wide steps,
// each summed from zero, then added in slice order) is a row's summation order; nothing in it says
// that slice w must be summed by warp w. Here every warp walks ALL the slices of its own 8 NT weight
// rows in order, keeping the slice's partial (from zero, the same mma sequence per accumulator as the
// served kernel) and a running total (the first slice's partial, then + each next one, empty slices
// adding 0.f as the served reduction does). Every row is therefore the same bits as the served tile
// of the same WK. What that buys: the W warps of a CTA now read the SAME activation step at the same
// time, so the CTA stages it once in shared memory (cp.async, NB buffers, zero-filled past M) and the
// lanes read their A operands from there -- no activation registers in flight, a 1/(8 NT W)-th of the
// activation traffic per weight row instead of all of it, and the weights double-buffered in registers.
// Row layout in shared memory: a row's 16 pieces of 16 bytes (8 bf16) as [r][t] (piece kk = 4t + r is
// the lane t's register-r vector), rows XSTRIDE bytes apart so a quarter-warp's 16-byte loads hit 32
// distinct banks.
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <int NT, int MT, int W, int NB, int MINB, int G = 1>
__global__ void __launch_bounds__(32 * W, MINB)
skinny_ser_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ Wt,
                  const uint8_t* __restrict__ S, const float* __restrict__ S2V, float s2,
                  __nv_bfloat16* __restrict__ Y, int M, int N, int KQ, int WK,
                  int ldx, int ldw, int lds, int ldy) {
    constexpr int ROWS = 16 * MT;
    constexpr int XSTRIDE = 320;
    constexpr int STEPB = ROWS * XSTRIDE;      // one K step's staged activation
    constexpr int CHUNK = G * STEPB;           // G steps a barrier
    static_assert(G == 1 || G % 2 == 0, "G steps a chunk: 1 or even (the weights' buffer is g's parity)");
    extern __shared__ __align__(16) uint8_t xs[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * (8 * NT * W) + warp * (8 * NT);
    const int per = (KQ + WK - 1) / WK;

    const uint8_t* wrow[NT];
    const uint8_t* srow[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int r = min(n0 + 8 * i + g, N - 1);
        wrow[i] = Wt + (size_t)r * ldw + 16 * t;
#if SRUN
        srow[i] = S + ((size_t)(r >> 4) * KQ * 16 + (r & 15)) * 8 + 2 * t;
#else
        srow[i] = S + (size_t)r * lds + 2 * t;
#endif
    }
    // this lane's rows inside a staged chunk: g and g + 8 of each 16-row tile
    int xoff[2 * MT];
#pragma unroll
    for (int m = 0; m < 2 * MT; ++m) xoff[m] = ((m >> 1) * 16 + (m & 1) * 8 + g) * XSTRIDE + 16 * t;

    // chunk c = K steps c G .. c G + G - 1 (those below KQ), one commit group a chunk
    auto issue = [&](int c) {
        uint8_t* dst = xs + (c % NB) * CHUNK;
        for (int p = threadIdx.x; p < G * ROWS * 16; p += 32 * W) {
            const int gs = p / (ROWS * 16), rp = p - gs * (ROWS * 16);
            const int q = c * G + gs;
            if (q >= KQ) break;
            const int row = rp >> 4, kk = rp & 15;
            const bool on = row < M;
            const __nv_bfloat16* src = X + (size_t)(on ? row : 0) * ldx + (size_t)q * 128 + 8 * kk;
            const uint32_t d = smem_u32(dst + gs * STEPB + row * XSTRIDE + ((kk & 3) * 4 + (kk >> 2)) * 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                         :: "r"(d), "l"(src), "r"(on ? 16 : 0) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };

    float tot[MT][NT][4], part[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int c = 0; c < 4; ++c) { tot[a][i][c] = 0.f; part[a][i][c] = 0.f; }

    uint4 wb[2][NT];
    uint32_t sb[2][NT];
    auto load = [&](int buf, int q) {
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            wb[buf][i] = ld_w(wrow[i] + (size_t)q * 64);
            sb[buf][i] = ld_s(srow[i] + (size_t)q * (SRUN ? 128 : 8));
        }
    };
    // the served kernel's per-accumulator order: register r (j0 then j1), for r = 0..3
    auto compute = [&](int buf, const uint8_t* xb) {
        uint32_t s_lo[NT], s_hi[NT];
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t sc = dec_s(sb[buf][i]);
            s_lo[i] = __byte_perm(sc, sc, 0x1010);
            s_hi[i] = __byte_perm(sc, sc, 0x3232);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            uint32_t xr[2 * MT][4];
#pragma unroll
            for (int m = 0; m < 2 * MT; ++m) {
                const uint4 u = *reinterpret_cast<const uint4*>(xb + xoff[m] + 64 * r);
                xr[m][0] = bf2h(u.x); xr[m][1] = bf2h(u.y);
                xr[m][2] = bf2h(u.z); xr[m][3] = bf2h(u.w);
            }
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                const uint32_t w = r == 0 ? wb[buf][i].x : r == 1 ? wb[buf][i].y
                                 : r == 2 ? wb[buf][i].z : wb[buf][i].w;
                uint32_t d0, d1, d2, d3;
                dec4(w, d0, d1, d2, d3);
                const uint32_t s = r < 2 ? s_lo[i] : s_hi[i];
                d0 = hmul2(d0, s); d1 = hmul2(d1, s); d2 = hmul2(d2, s); d3 = hmul2(d3, s);
#pragma unroll
                for (int a = 0; a < MT; ++a) {
                    mma(part[a][i], xr[2 * a][0], xr[2 * a + 1][0], xr[2 * a][1],
                        xr[2 * a + 1][1], d0, d1);
                    mma(part[a][i], xr[2 * a][2], xr[2 * a + 1][2], xr[2 * a][3],
                        xr[2 * a + 1][3], d2, d3);
                }
            }
        }
    };
    int w = 0, wend = min(KQ, per);
    auto fold = [&](int q) {
        // after step q: a slice ends here -> its partial into the total, in slice order
        while (w < WK && q + 1 == wend) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        tot[a][i][c] = w == 0 ? part[a][i][c] : tot[a][i][c] + part[a][i][c];
                        part[a][i][c] = 0.f;
                    }
            ++w;
            wend = min(KQ, (w + 1) * per);
            if (w * per >= KQ) break;     // the rest are empty slices, added below
        }
    };
#pragma unroll
    for (int c = 0; c < NB - 1; ++c) issue(c);
    load(0, 0);
    if constexpr (G == 1) {
        auto step = [&](int buf, int q) {
            asm volatile("cp.async.wait_group %0;" :: "n"(NB - 2) : "memory");
            __syncthreads();              // every lane's pieces of step q landed; step q-1 is done
            issue(q + NB - 1);
            compute(buf, xs + (q % NB) * CHUNK);
            fold(q);
        };
        int q = 0;
        for (; q + 1 < KQ; q += 2) {
            load(1, q + 1);
            step(0, q);
            if (q + 2 < KQ) load(0, q + 2);
            step(1, q + 1);
        }
        if (q < KQ) step(0, q);
    } else {
        // one barrier every G steps; the steps inside a chunk run in order as above
        const int nch = (KQ + G - 1) / G;
        for (int c = 0; c < nch; ++c) {
            asm volatile("cp.async.wait_group %0;" :: "n"(NB - 2) : "memory");
            __syncthreads();
            issue(c + NB - 1);
            const uint8_t* xc = xs + (c % NB) * CHUNK;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int q = c * G + g;
                if (q < KQ) {
                    if (q + 1 < KQ) load((g + 1) & 1, q + 1);
                    compute(g & 1, xc + g * STEPB);
                    fold(q);
                }
            }
        }
    }
    // empty slices (KQ not a multiple of WK): the served reduction adds their zero partials too
    for (; w < WK; ++w)
#pragma unroll
        for (int a = 0; a < MT; ++a)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c) tot[a][i][c] = w == 0 ? 0.f : tot[a][i][c] + 0.f;
    asm volatile("cp.async.wait_group 0;" ::: "memory");

#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            const int n = n0 + 8 * i + 2 * t;
            if (n >= N) continue;
            const float sa = S2V ? S2V[n] : s2, sb2 = S2V ? S2V[min(n + 1, N - 1)] : s2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = a * 16 + h * 8 + g;
                if (m >= M) continue;
                __nv_bfloat16* y = Y + (size_t)m * ldy + n;
                const __nv_bfloat16 v0 = __float2bfloat16_rn(tot[a][i][2 * h] * sa);
                if (n + 1 < N) {
                    const __nv_bfloat16 v1 = __float2bfloat16_rn(tot[a][i][2 * h + 1] * sb2);
                    __nv_bfloat162 p; p.x = v0; p.y = v1;
                    *reinterpret_cast<__nv_bfloat162*>(y) = p;
                } else {
                    *y = v0;
                }
            }
        }
}

template <int NT, int MT, int W, int NB, int MINB, int G = 1>
void launch_ser(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s,
                const float* s2v, float s2, torch::Tensor& y, int wk) {
    const int M = x.size(0), N = w.size(0), K = w.size(1) * 2;
    const int smem = NB * G * 16 * MT * 320;
    auto kern = skinny_ser_kernel<NT, MT, W, NB, MINB, G>;
    static bool attr = false;
    if (!attr && smem > 48 * 1024) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    attr = true;
    dim3 grid((N + 8 * NT * W - 1) / (8 * NT * W));
    const int kq = K / 128, lx = x.stride(0), lw = w.stride(0), ls = s.stride(0), ly = y.stride(0);
    kern<<<grid, 32 * W, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(s.data_ptr()), s2v, s2,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N, kq, wk, lx, lw, ls, ly);
}

}  // namespace

void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor s2v, double s2,
            torch::Tensor y, int64_t nt, int64_t wk, int64_t pf, int64_t minb, int64_t il,
            int64_t pdl_, int64_t kr, int64_t spw) {
    const int pdl = (int)pdl_;
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "x");
    TORCH_CHECK(w.scalar_type() == at::kByte && w.stride(1) == 1 && s.stride(1) == 1, "w/s");
    TORCH_CHECK(w.size(1) % 64 == 0 && x.size(1) == w.size(1) * 2, "K");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 32, "rows");
    TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 16 == 0, "alignment");
    const float* p = s2v.numel() ? s2v.data_ptr<float>() : nullptr;
    const int mt = x.size(0) > 16 ? 2 : 1;
    // SPW (slices a warp): the instances the 2026-09-26 sweep asks for
    if (spw > 1) {
        TORCH_CHECK(il == 0 && wk == 16, "spw needs the contiguous 16-slice split");
#define SPWI(NT, MT, PF, MINB, KR, SPW)                                                   \
        if (nt == NT && mt == MT && pf == PF && minb == MINB && kr == KR && spw == SPW) {  \
            launch<NT, MT, 16, PF, MINB, 0, KR, SPW>(x, w, s, p, s2, y, pdl); return; }
        SPWI(2, 1, 2, 2, 0, 2) SPWI(2, 1, 2, 4, 0, 4) SPWI(4, 1, 2, 2, 0, 2) SPWI(1, 1, 2, 2, 0, 2)
        SPWI(2, 1, 2, 1, 0, 2)
        SPWI(4, 2, 0, 2, 1, 2) SPWI(2, 2, 2, 2, 0, 2) SPWI(2, 2, 0, 2, 1, 2) SPWI(4, 2, 0, 1, 1, 2)
#undef SPWI
        TORCH_CHECK(false, "no spw instance for nt=", nt, " mt=", mt, " pf=", pf, " minb=", minb,
                    " kr=", kr, " spw=", spw);
    }
    // Seventeen rows and up take eight weight rows a warp: sixteen would not fit in registers
    // twice over. The K split (`wk`) is what fixes a row's summation order, and it is unchanged.
    if (mt == 2 && nt == 16) nt = 8;
    // the register-sequential order for 17..32 rows (up to 16 rows it is not instantiated)
    if (kr && mt == 2) {
        TORCH_CHECK((pf == 0 || pf == 2) && minb == 1 && il == 0 && (wk == 8 || wk == 16) &&
                    (nt == 2 || nt == 4), "no kr instance for nt=", nt, " wk=", wk, " pf=", pf,
                    " minb=", minb, " il=", il);
#define KR1(NT, PF)                                                                       \
        if (wk == 16) { launch<NT, 2, 16, PF, 1, 0, 1>(x, w, s, p, s2, y, pdl); return; }    \
        launch<NT, 2, 8, PF, 1, 0, 1>(x, w, s, p, s2, y, pdl); return;
        if (nt == 2) { if (pf == 0) { KR1(2, 0) } KR1(2, 2) }
        if (pf == 0) { KR1(4, 0) } KR1(4, 2)
    }
    // every (nt, pf, minb, wk) the sweep asks for; minb = 2 only for up to 16 rows
#define PFS1(NT)                                                                          \
    if (minb == 2) {                                                                      \
        if (pf == 0) { WK4(NT, 1, 0, 2) } else if (pf == 1) { WK4(NT, 1, 1, 2) }          \
        else { WK4(NT, 1, 2, 2) }                                                         \
    } else {                                                                              \
        if (pf == 0) { WK5(NT, 1, 0) } else if (pf == 1) { WK5(NT, 1, 1) }                \
        else { WK5(NT, 1, 2) }                                                            \
    }
#define PFS2(NT)                                                                          \
    if (pf == 0) { WK5(NT, 2, 0) } else if (pf == 1) { WK5(NT, 2, 1) } else { WK5(NT, 2, 2) }
    if (il) {
        if (minb == 1 && (nt == 1 || nt == 2) && (pf == 1 || pf == 2)) {
            if (mt == 1) {
                if (nt == 1) { if (pf == 1) { IL1(1, 1, 1) } else { IL1(1, 1, 2) } }
                else { if (pf == 1) { IL1(2, 1, 1) } else { IL1(2, 1, 2) } }
            } else {
                if (nt == 1) { if (pf == 1) { IL1(1, 2, 1) } else { IL1(1, 2, 2) } }
                else { if (pf == 1) { IL1(2, 2, 1) } else { IL1(2, 2, 2) } }
            }
        }
        TORCH_CHECK(false, "no interleaved instance for nt=", nt, " wk=", wk, " pf=", pf,
                    " minb=", minb, " mt=", mt);
    }
    // an 8-row N tile at 512 threads: two CTAs an SM if the registers allow it
    if (mt == 1 && nt == 1 && minb == 2 && wk == 16) {
        if (pf == 1) { launch<1, 1, 16, 1, 2>(x, w, s, p, s2, y, pdl); return; }
        if (pf == 2) { launch<1, 1, 16, 2, 2>(x, w, s, p, s2, y, pdl); return; }
    }
    if (mt == 1) {
        if (nt == 1) { PFS1(1) } else if (nt == 2) { PFS1(2) } else if (nt == 4) { PFS1(4) }
        else if (nt == 8) { PFS1(8) } else { PFS1(16) }
    } else {
        if (nt == 1) { PFS2(1) } else if (nt == 2) { PFS2(2) } else if (nt == 4) { PFS2(4) }
        else { PFS2(8) }
    }
    TORCH_CHECK(false, "no instance for nt=", nt, " wk=", wk, " pf=", pf, " minb=", minb,
                " mt=", mt);
}

// the slice-serial tile. `wk` is the served K split (the summation order), `w` the warps a CTA
// (each 8 NT weight rows over the whole K), `nb` the staged activation buffers.
void skinny_ser(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor s2v, double s2,
                torch::Tensor y, int64_t nt, int64_t wk, int64_t wn, int64_t nb, int64_t g) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "x");
    TORCH_CHECK(w.scalar_type() == at::kByte && w.stride(1) == 1 && s.stride(1) == 1, "w/s");
    TORCH_CHECK(w.size(1) % 64 == 0 && x.size(1) == w.size(1) * 2, "K");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 32, "rows");
    TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 16 == 0 && x.data_ptr() != nullptr &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "alignment");
    const float* p = s2v.numel() ? s2v.data_ptr<float>() : nullptr;
    const float f = (float)s2;
    const int mt = x.size(0) > 16 ? 2 : 1, k = (int)wk;
#define SER(NT, MT, W, NB, MINB)                                                          \
    if (nt == NT && mt == MT && wn == W && nb == NB && g == 1) {                          \
        launch_ser<NT, MT, W, NB, MINB>(x, w, s, p, f, y, k); return; }
#define SERG(NT, MT, W, NB, MINB, G)                                                      \
    if (nt == NT && mt == MT && wn == W && nb == NB && g == G) {                          \
        launch_ser<NT, MT, W, NB, MINB, G>(x, w, s, p, f, y, k); return; }
    SER(1, 1, 4, 2, 4) SER(1, 1, 8, 2, 2) SER(2, 1, 4, 2, 4) SER(2, 1, 8, 2, 2)
    SER(2, 1, 4, 3, 4) SER(2, 1, 8, 3, 2) SER(1, 1, 16, 2, 1) SER(2, 1, 16, 3, 1)
    SER(1, 2, 4, 2, 4) SER(1, 2, 8, 2, 2) SER(1, 2, 16, 2, 1)
    SER(2, 2, 4, 2, 4) SER(2, 2, 8, 2, 2) SER(2, 2, 16, 2, 1)
    SER(2, 2, 4, 3, 4) SER(2, 2, 8, 3, 2) SER(2, 2, 16, 3, 1)
    SER(4, 2, 4, 2, 2) SER(4, 2, 8, 2, 1) SER(4, 2, 4, 3, 2) SER(4, 2, 8, 3, 1)
    SER(1, 2, 8, 3, 2) SER(1, 2, 16, 3, 1) SER(1, 2, 8, 4, 2)
    // G steps a barrier (MT 1: 5 KB a step; MT 2: 10 KB a step)
    SERG(2, 1, 8, 2, 2, 2) SERG(2, 1, 8, 2, 1, 4) SERG(1, 1, 16, 2, 1, 4) SERG(2, 1, 16, 2, 1, 2)
    SERG(2, 2, 8, 2, 2, 2) SERG(1, 2, 16, 2, 1, 2) SERG(1, 2, 16, 2, 1, 4) SERG(2, 2, 16, 2, 1, 2)
    SERG(4, 2, 4, 2, 2, 2)
#undef SER
#undef SERG
    TORCH_CHECK(false, "no slice-serial instance for nt=", nt, " mt=", mt, " w=", wn, " nb=", nb,
                " g=", g);
}
"""

_CPP = ("void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor s2v, "
        "double s2, torch::Tensor y, int64_t nt, int64_t wk, int64_t pf, int64_t minb, "
        "int64_t il, int64_t pdl_, int64_t kr, int64_t spw);\n"
        "void skinny_ser(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor s2v, "
        "double s2, torch::Tensor y, int64_t nt, int64_t wk, int64_t wn, int64_t nb, int64_t g);")

_MOD = None
# one built module a weight-load hint, so an in-process block A/B can flip `LDW` (the
# verify graphs key on it, engine/verify_graph.py signature) without a second process
_MODS: dict = {}
_MOD_LDW = None


def _module(ldw: int | None = None):
    """The built kernel module for a weight-load hint: `LDW` by default, or a tile entry's own `ldw`
    (a hint for the 17..32-row tile alone, 2026-09-26)."""
    global _MOD, _MOD_LDW
    if ldw is not None and ldw != LDW:
        if ldw not in _MODS:
            _MODS[ldw] = _build(ldw)
        return _MODS[ldw]
    if _MOD is not None and _MOD_LDW == LDW:
        return _MOD
    if _MOD is not None and LDW in _MODS:
        _MOD, _MOD_LDW = _MODS[LDW], LDW
        return _MOD
    _MOD = _build(LDW)
    _MODS[LDW], _MOD_LDW = _MOD, LDW
    return _MOD


def _build(ldw: int):
    from torch.utils.cpp_extension import load_inline
    venv_bin = os.path.dirname(sys.executable)
    if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    # the build directory is named by the source's hash, so two checkouts with different
    # kernels on one box (the service and a branch under test) never rebuild over each other
    import hashlib
    src = f"#define LDW_HINT {ldw}\n" + _CUDA
    if XSTUB or SSTUB:
        src = f"#define XSTUB {XSTUB}\n#define SSTUB {SSTUB}\n" + src
    if SRUN:
        src = f"#define SRUN {SRUN}\n" + src
    tag = hashlib.sha1((_CPP + src).encode()).hexdigest()[:10]
    return load_inline(name=f"qwen38_nvfp4_skinny_{tag}", cpp_sources=[_CPP],
                       cuda_sources=[src],
                       functions=["skinny", "skinny_ser"],
                       # the e2m1 converter is an arch-specific instruction: sm_121a and
                       # nothing else (an explicit arch flag also stops torch adding its own)
                       extra_cuda_cflags=["-O3", "-lineinfo",
                                          "-gencode=arch=compute_121a,code=sm_121a"],
                       verbose=False)


# (N, K) -> tile. Keyed by shape only, never by row count: the K split is the reduction order.
# Filled from the cold sweep and the in-engine A/B (docs/kernels.md).
_CONFIG: dict[tuple[int, int], dict] = {}
# A shape the table does not name -- the drafter's 4096- and 25600-wide projections -- takes the
# first sweep's winner, which is what the kb-* rows ran (ops/skinny-tiles.json names the target's).
_FALLBACK = {"nt": 4, "wk": 8, "pf": 1, "minb": 1}
# A measured table from a file, for the in-engine A/B before a table is written into this one:
# {"17408x5120": {"nt": 8, "wk": 4, "pf": 1}, ...}
if _S.get("SKINNY_TILES") and not os.path.isfile(_S.get("SKINNY_TILES")):
    # a missing table would silently serve every shape on the fallback tile
    print(f"[skinny] WARNING: QWEN38_SKINNY_TILES={_S.get('SKINNY_TILES')} does not "
          f"exist; every projection takes the fallback tile {_FALLBACK}", flush=True)
if _S.get("SKINNY_TILES") and os.path.isfile(_S.get("SKINNY_TILES")):
    import json as _json
    for _k, _v in _json.load(open(_S.get("SKINNY_TILES"))).items():
        _n, _kk = (int(v) for v in _k.split("x"))
        _CONFIG[(_n, _kk)] = {"nt": int(_v["nt"]), "wk": int(_v["wk"]), "pf": int(_v["pf"]),
                              "minb": int(_v.get("minb", 1)), "il": int(_v.get("il", 0)),
                              **({"kr": 1} if int(_v.get("kr", 0)) else {}),
                              **({"spw": int(_v["spw"])} if int(_v.get("spw", 1)) > 1 else {})}


# Two more tables for an in-process A/B: QWEN38_SKINNY_TILES_B / _C name them, and `ALT` /
# `ALT2` -- switches `tools/block_budget.py --ab tools.nvfp4_skinny:ALT` flips -- route the shapes
# each names through it (ALT2 first when both are on). Off (the default) is the first table, exactly.
def _table(env: str) -> dict:
    out: dict[tuple[int, int], dict] = {}
    path = _S.get(env[len("QWEN38_"):]) if env.startswith("QWEN38_") else os.environ.get(env)
    if path and os.path.isfile(path):
        import json as _json
        for k, v in _json.load(open(path)).items():
            n, kk = (int(x) for x in k.split("x"))
            out[(n, kk)] = {"nt": int(v["nt"]), "wk": int(v["wk"]), "pf": int(v["pf"]),
                            "minb": int(v.get("minb", 1)), "il": int(v.get("il", 0)),
                            **({"kr": 1} if int(v.get("kr", 0)) else {}),
                            **({"spw": int(v["spw"])} if int(v.get("spw", 1)) > 1 else {}),
                            **({"ldw": int(v["ldw"])} if "ldw" in v else {}),
                            **({"ser": 1, "wn": int(v["wn"]), "nb": int(v["nb"]),
                                "g": int(v.get("g", 1))} if int(v.get("ser", 0)) else {})}
    return out


_ALT, _ALT2 = _table("QWEN38_SKINNY_TILES_B"), _table("QWEN38_SKINNY_TILES_C")
# the tile for 17..32 rows. At 32 rows the 16-row winner (nt2:wk16:pf2) falls to
# 164-184 GB/s on the wide shapes, a verify of 32 rows paying ~17 ms more in this kernel than one of
# 16. A second table, read only past sixteen rows, may change the N tile and the prefetch but NEVER
# the K split: the K split is a row's summation order, so a row keeps its bits whatever the block's
# width (the losslessness gate's row independence). An entry with another `wk` is refused at load.
if _S.get("SKINNY_TILES_WIDE") and not os.path.isfile(_S.get("SKINNY_TILES_WIDE")):
    print(f"[skinny] WARNING: QWEN38_SKINNY_TILES_WIDE={_S.get('SKINNY_TILES_WIDE')} does "
          f"not exist; 17..32-row verifies take the base tile", flush=True)
_WIDE = _table("QWEN38_SKINNY_TILES_WIDE")
for (_n, _kk), _v in list(_WIDE.items()):
    _base = _CONFIG.get((_n, _kk), _FALLBACK)
    if _v["wk"] != _base["wk"] or _v.get("il", 0) != _base.get("il", 0):
        print(f"[skinny] WARNING: QWEN38_SKINNY_TILES_WIDE {_n}x{_kk} splits K as wk{_v['wk']}, "
              f"the base tile as wk{_base['wk']}: refused (a row's bits would depend on the block's "
              f"width)", flush=True)
        del _WIDE[(_n, _kk)]
# a second 17..32-row table for the in-process block A/B. QWEN38_SKINNY_TILES_WIDE_B
# names it and `WIDE_B` (off by default) routes the shapes it names through it past sixteen rows. The
# same rule as the wide table: an entry that splits K otherwise than the base tile is refused.
_WIDE_B = _table("QWEN38_SKINNY_TILES_WIDE_B")
for (_n, _kk), _v in list(_WIDE_B.items()):
    _base = _CONFIG.get((_n, _kk), _FALLBACK)
    if _v["wk"] != _base["wk"] or _v.get("il", 0) != _base.get("il", 0):
        print(f"[skinny] WARNING: QWEN38_SKINNY_TILES_WIDE_B {_n}x{_kk} splits K as wk{_v['wk']}, "
              f"the base tile as wk{_base['wk']}: refused", flush=True)
        del _WIDE_B[(_n, _kk)]
ALT = False
ALT2 = False
WIDE_B = False
# The weight-load hint for 17..32-row calls only (None: the entry's own `ldw`, else LDW). In-process A/B.
WIDE_LDW = None


def _alt(N: int, K: int) -> dict | None:
    if ALT2 and (N, K) in _ALT2:
        return _ALT2[(N, K)]
    if ALT and (N, K) in _ALT:
        return _ALT[(N, K)]
    return None


from tools.tile_warn import missing as _tile_missing  # noqa: E402


def pick(N: int, K: int, M: int = 1) -> dict:
    if M > 16 and WIDE_B and (N, K) in _WIDE_B:
        return _WIDE_B[(N, K)]
    if M > 16 and (N, K) in _WIDE:
        return _WIDE[(N, K)]
    return _alt(N, K) or _CONFIG.get((N, K), _FALLBACK)


def set_config(N: int, K: int, cfg: dict) -> None:
    _CONFIG[(N, K)] = dict(cfg)


_EMPTY: dict = {}


def nvfp4_matmul_skinny(x: torch.Tensor, w, *, out: torch.Tensor | None = None,
                        nt: int | None = None, wk: int | None = None,
                        pf: int | None = None, minb: int | None = None,
                        il: int | None = None, kr: int | None = None,
                        ser: int | None = None, wn: int | None = None,
                        nb: int | None = None, g: int | None = None,
                        spw: int | None = None) -> torch.Tensor:
    """Y[M, N] = x[M, K] @ W[N, K]^T for M <= 32; W an NVFP4Block or an NVFP4Group."""
    M = x.shape[0]
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and 1 <= M <= SKINNY_MAX, x.shape
    x = x.contiguous()
    if x.data_ptr() % 16:
        # the activation is read with 16-byte loads; a view at an odd offset is copied once
        x = x.clone()
    cfg = pick(w.N, w.K, M)
    if (w.N, w.K) not in _CONFIG and _alt(w.N, w.K) is None:
        if getattr(w, "sizes", None):
            # a fused group inherits its first member's tile: same K split, so a grouped launch
            # sums each row in the order the member's own launch would
            cfg = pick(w.sizes[0], w.K, M)
            if (w.sizes[0], w.K) not in _CONFIG and _alt(w.sizes[0], w.K) is None:
                _tile_missing("nvfp4_skinny", w.sizes[0], w.K, _FALLBACK)
        else:
            _tile_missing("nvfp4_skinny", w.N, w.K, _FALLBACK)
    if nt is not None or wk is not None or pf is not None:
        # an explicit tile (a test, a bench) is the whole tile: the table entry's order and layout
        # extras -- kr, spw, ser, its own hint -- belong to the entry and are not carried over to it
        # (phase6: with kr1 as the served wide table, an explicit nt4:pf0 would have run kr1)
        cfg = {k: v for k, v in cfg.items() if k in ("nt", "wk", "pf", "minb", "il")}
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.bfloat16, device=x.device)
    s2v = getattr(w, "s2v", None)
    if s2v is None:
        s2v = _EMPTY.get(x.device)
        if s2v is None:
            s2v = _EMPTY[x.device] = torch.empty(0, dtype=torch.float32, device=x.device)
    sv = w.s.view(torch.uint8)
    if SRUN:
        sv = getattr(w, "_srun", None)
        if sv is None:
            sv = w._srun = scale_runs(w.s).view(-1, 8)
    mod = _module(WIDE_LDW if M > 16 and WIDE_LDW is not None else cfg.get("ldw"))
    if (cfg.get("ser", 0) if ser is None else ser):
        # the slice-serial tile, the same K split (`wk`) summed by one warp in slice order
        mod.skinny_ser(x, w.w, sv, s2v, float(w.s2), out,
                             cfg["nt"] if nt is None else nt, cfg["wk"] if wk is None else wk,
                             cfg["wn"] if wn is None else wn, cfg["nb"] if nb is None else nb,
                             cfg.get("g", 1) if g is None else g)
        return out
    mod.skinny(x, w.w, sv, s2v, float(w.s2), out,
                     cfg["nt"] if nt is None else nt, cfg["wk"] if wk is None else wk,
                     cfg["pf"] if pf is None else pf,
                     cfg.get("minb", 1) if minb is None else minb,
                     cfg.get("il", 0) if il is None else il, int(PDL),
                     cfg.get("kr", 0) if kr is None else kr,
                     cfg.get("spw", 1) if spw is None else spw)
    return out


def scale_runs(s: torch.Tensor) -> torch.Tensor:
    """The e4m3 scales [N, K/16] as runs of 16 rows x one 128-wide K step.

    Run (G, q) holds bytes 8q .. 8q + 7 of rows 16G .. 16G + 15, in row order: 128 contiguous bytes,
    which a warp's lanes read for its 8 NT rows in one or two lines where they read 8 bytes in each
    of 8 rows before. A permutation of the same bytes; rows past N up to a multiple of 16 are zero
    and never read (the kernel clamps a row index to N - 1)."""
    N, C = s.shape
    u = s.contiguous().view(torch.uint8)
    n16 = (N + 15) // 16 * 16
    if n16 != N:
        u = torch.cat([u, u.new_zeros(n16 - N, C)])
    return u.view(n16 // 16, 16, C // 8, 8).permute(0, 2, 1, 3).contiguous().view(-1)


def use_skinny(M: int) -> bool:
    return SKINNY and 1 <= M <= SKINNY_MAX


# ------------------------------------------------------------------ gate and bench
def _exact(w) -> torch.Tensor:
    """The weight in fp32 with no rounding at all: e2m1 x e4m3 x s2, the value every kernel means."""
    from tools.nvfp4_linear import FP4_GRID, GROUP
    grid = FP4_GRID.to(w.w.device)
    lo, hi = (w.w & 0x0F).long(), (w.w >> 4).long()
    vals = torch.empty(w.N, w.K, dtype=torch.float32, device=w.w.device)
    vals[:, 0::2] = grid[lo & 7] * torch.where(lo >= 8, -1.0, 1.0)
    vals[:, 1::2] = grid[hi & 7] * torch.where(hi >= 8, -1.0, 1.0)
    return vals * w.s.float().repeat_interleave(GROUP, 1) * w.s2


def check(shapes=((17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144),
                  (12288, 5120), (1024, 5120)),
          rows=(1, 2, 7, 8, 9, 14, 16, 17, 24, 32), seed: int = 0) -> list[str]:
    """Against the exact product and against v2, on random NVFP4 weights.

    The bar is v2's own: `max|skinny - v2|` at or below `max|v2 - exact|` (two fp32 orders of the
    same fp16 products), and row independence -- a row computed alone equals the same row inside a
    block, bit for bit -- at every row count, for every tile in the sweep.
    """
    from tools.nvfp4_linear import quantize_to_nvfp4
    from tools.nvfp4_linear_v2 import nvfp4_matmul_v2
    g = torch.Generator(device="cuda").manual_seed(seed)
    out, worst = [], 0.0
    for (N, K) in shapes:
        ref = torch.randn(N, K, device="cuda", generator=g, dtype=torch.float32) * 0.02
        w = quantize_to_nvfp4(ref)
        wd = _exact(w)
        x = torch.randn(max(rows), K, device="cuda", generator=g).to(torch.bfloat16)
        for cfg in ({"nt": 4, "wk": 1, "pf": 0}, {"nt": 8, "wk": 4, "pf": 1},
                    {"nt": 16, "wk": 2, "pf": 0}, {"nt": 4, "wk": 8, "pf": 1},
                    {"nt": 2, "wk": 16, "pf": 2}, {"nt": 4, "wk": 8, "pf": 2, "minb": 2}):
            single = None
            for M in rows:
                xm = x[:M]
                y = nvfp4_matmul_skinny(xm, w, **cfg).float()
                y2 = nvfp4_matmul_v2(xm, w).float()
                exact = xm.float().to(torch.float16).float() @ wd.T
                d_s = (y - exact).abs().max().item()
                d_v2 = (y2 - exact).abs().max().item()
                d_sv = (y - y2).abs().max().item()
                ratio = d_sv / max(d_v2, 1e-12)
                worst = max(worst, ratio)
                if single is None:
                    single = y[:1].clone()
                indep = (y[:1] - single).abs().max().item()
                assert indep == 0.0, f"row independence {N}x{K} {cfg} M={M}: {indep}"
                assert d_s <= 2 * d_v2 + 1e-6, f"{N}x{K} {cfg} M={M}: {d_s} vs v2 {d_v2}"
            out.append(f"{N}x{K} {cfg}: last |skinny-exact| {d_s:.3e}  |v2-exact| {d_v2:.3e}  "
                       f"|skinny-v2| {d_sv:.3e}")
        del w, wd, ref
    out.append(f"worst |skinny-v2| / |v2-exact| = {worst:.3f}; row independence exact everywhere")
    return out


def bench_cold(N: int, K: int, gb: float = 2.5, rows=(1, 8, 16, 32), tiles=None,
               reps: int = 5) -> list[str]:
    """A chain of distinct weights bigger than the board's caches, so every byte is a DRAM read.

    Same method as tools/cold_bw.py: the chain is `n` independent copies, each projection timed
    over the whole chain. A ranking, never a budget -- the engine's own verify decides.
    """
    from tools.nvfp4_linear import quantize_to_nvfp4
    from tools.nvfp4_linear_v2 import nvfp4_matmul_v2
    one = quantize_to_nvfp4(torch.randn(N, K, device="cuda") * 0.02)
    per = one.nbytes
    n = max(4, int(gb * 1e9 / per))
    ws = []
    for _ in range(n):
        b = type(one).__new__(type(one))
        b.w, b.s, b.s2, b.N, b.K, b._bf16 = one.w.clone(), one.s.clone(), one.s2, one.N, one.K, None
        if SRUN:
            b._srun = scale_runs(b.s).view(-1, 8)       # built before the clock, as the engine does
        ws.append(b)
    del one
    tiles = tiles or [{"nt": 8, "wk": 4, "pf": 1}]
    lines = [f"### {N}x{K}  {per / 1e6:.1f} MB x {n} = {n * per / 1e9:.2f} GB"]
    for M in rows:
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        cands = [("v2", lambda b: nvfp4_matmul_v2(x, b))]
        for c in tiles:
            cands.append((tile_name(c), lambda b, c=c: nvfp4_matmul_skinny(x, b, **c)))
        for name, fn in cands:
            try:
                for b in ws[:2]:
                    fn(b)
                torch.cuda.synchronize()
            except RuntimeError as e:          # a tile the kernel refuses (shared memory, no instance)
                lines.append(f"  M={M:3d} {name:>18} refused: {str(e).splitlines()[0][:80]}")
                continue
            best = float("inf")
            for _ in range(reps):
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                for b in ws:
                    fn(b)
                t1.record()
                t1.synchronize()
                best = min(best, t0.elapsed_time(t1))
            lines.append(f"  M={M:3d} {name:>18} {best:8.2f} ms  {n * per / best / 1e6:7.1f} GB/s")
            RESULTS.setdefault(f"{N}x{K}", {}).setdefault(name, {})[M] = n * per / best / 1e6
    return lines


RESULTS: dict = {}


def tile_name(c: dict) -> str:
    """`nt2:wk16:pf2:mb1`, with `:il1` for the interleaved K split."""
    if c.get("ser", 0):
        return (f"nt{c['nt']}:wk{c['wk']}:sr1:wn{c['wn']}:nb{c['nb']}"
                + (f":gs{c['g']}" if c.get("g", 1) != 1 else ""))
    return (f"nt{c['nt']}:wk{c['wk']}:pf{c['pf']}:mb{c.get('minb', 1)}"
            + (":il1" if c.get("il", 0) else "") + (":kr1" if c.get("kr", 0) else "")
            + (f":sw{c['spw']}" if c.get("spw", 1) > 1 else ""))


def parse_tile(name: str) -> dict:
    keys = {"nt": "nt", "wk": "wk", "pf": "pf", "mb": "minb", "il": "il", "kr": "kr", "sr": "ser",
            "wn": "wn", "nb": "nb", "gs": "g", "sw": "spw"}
    out = {"minb": 1, "il": 0}
    if ":sr1" in name:
        out["pf"] = 2
    for part in name.split(":"):
        out[keys[part[:2]]] = int(part[2:])
    return out


def best_tiles(rows=(8, 16)) -> dict:
    """Per shape, the tile with the best mean GB/s over the verify's row counts."""
    out = {}
    for shape, by_tile in RESULTS.items():
        scored = [(sum(v.get(m, 0.0) for m in rows) / len(rows), name)
                  for name, v in by_tile.items() if name != "v2"]
        if not scored:
            continue
        gbps, name = max(scored)
        out[shape] = dict(parse_tile(name), gbps=round(gbps, 1),
                          v2_gbps=round(sum(by_tile["v2"].get(m, 0.0) for m in rows) / len(rows), 1))
    return out


def read_ceiling(gb: float = 2.5, reps: int = 5) -> str:
    """What a plain streaming read reaches here: a float sum over a buffer bigger than the caches."""
    buf = torch.empty(int(gb * 1e9 / 4), dtype=torch.float32, device="cuda").uniform_()
    buf.sum()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        buf.sum()
        t1.record()
        t1.synchronize()
        best = min(best, t0.elapsed_time(t1))
    n = buf.numel() * 4
    del buf
    return f"read ceiling: torch sum over {n / 1e9:.2f} GB in {best:.2f} ms = {n / best / 1e6:.1f} GB/s"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", default="", help="N:K[,N:K...]")
    ap.add_argument("--rows", default="1,8,16")
    ap.add_argument("--tiles", default="nt4:wk8:pf1:mb1",
                    help="nt:wk:pf:mb[:il1], comma separated (mb = CTAs an SM the registers must "
                         "allow; il1 = the interleaved K split)")
    ap.add_argument("--gb", type=float, default=2.5)
    ap.add_argument("--write-tiles", default="", help="write the best tile per shape here")
    ap.add_argument("--best-rows", default="8,16",
                    help="the row counts --write-tiles averages over (24,32 for the wide table)")
    a = ap.parse_args()
    if a.bench:
        print(read_ceiling(a.gb), flush=True)
    if a.check:
        for line in check():
            print(line, flush=True)
    tiles = [parse_tile(t) for t in a.tiles.split(",")]
    for shp in filter(None, a.bench.split(",")):
        N, K = (int(v) for v in shp.split(":"))
        for line in bench_cold(N, K, a.gb, [int(r) for r in a.rows.split(",")], tiles):
            print(line, flush=True)
    if a.write_tiles:
        import json
        best = best_tiles(tuple(int(r) for r in a.best_rows.split(",")))
        json.dump(best, open(a.write_tiles, "w"), indent=1)
        for shape, c in best.items():
            print(f"best {shape}: nt{c['nt']}:wk{c['wk']}:pf{c['pf']} {c['gbps']} GB/s "
                  f"(v2 {c['v2_gbps']})", flush=True)
