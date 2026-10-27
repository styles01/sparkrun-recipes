"""EXL3 weights (exllamav3's trellis format) behind the engine's Linear interface.

A quantised projection in an EXL3 checkpoint (exllamav3 v1.6.0, the converter the
TandemLLM-Qwen3.8-27B-EXL3-* sets were made with) is, for in_features Kin and out_features N:

    <name>.trellis  int16 [Kin/16, N/16, 16 K]   K bits a weight, K an integer 2..8 per tensor
    <name>.suh      fp16  [Kin]                  input channel factor (sign x scale)
    <name>.svh      fp16  [N]                    output channel factor (sign x scale)
    <name>.mul1     the codebook's multiplier    0x83DCD12D: the "mul1" procedural codebook

and the weight it stands for (exllamav3 `LinearEXL3.get_weight_tensor`, in [Kin, N] orientation) is

    W = diag(suh) . H . Wi . H . diag(svh)        H = the 128-point Sylvester Hadamard / sqrt(128),
                                                     block-diagonal over 128-wide blocks
    y = x @ W = ( H( H(x * suh) @ Wi ) ) * svh

where Wi is the trellis decoded. A 16x16 tile of Wi is one cyclic bit string of 256 K bits. The
weight at position p (0..255) is the 16-bit window ending at bit (p + 1) K (wrapping round the
tile; the bit string is MSB-first inside each little-endian uint32 word), put through the codebook:

    v = window * 0x83DCD12D (mod 2^32);  h = fp16 bits (0x6400 + the sum of v's four bytes)
    w = fma(h, fp16(0x1eee), fp16(0xc931))        one fp16 rounding

and position p = 8 l + j belongs to (row k, column n) of the tile with
    k = 2 (l % 4) + (j & 1) + 8 ((j >> 1) & 1),   n = l / 4 + 8 (j >> 2)
which is exactly the tensor core's B-operand fragment of `mma.m16n8k16` for lane l: positions
8l..8l+3 are the operand registers of the tile's first eight columns, 8l+4..8l+7 of the second.
So a lane decodes its own operand registers in place; nothing is shuffled.

The kernels here (ENG-199) are this engine's own, written for the GB10 (sm_121) and the verify
shapes M = 1..32; the bit extraction and the codebook decode are ported from exllamav3's
`exl3_dq.cuh` / `codebook.cuh` (MIT, Copyright (c) 2025 Turboderp; the notice is kept with the
CUDA source below). Arithmetic, per call of `exl3_matmul`:

  1. `rot_in`: xh = fp16( H(fp32(x) * fp32(suh)) )           one warp per row and 128-block
  2. `gemm`:   y' = xh @ Wi in fp32 on the tensor cores (fp16 x fp16 -> fp32), the weight decoded
               from the trellis into the B registers
  3. epilogue: y = fp32( H(y') ) * svh -> bf16 (or fp32 for the vocabulary head)

Determinism (the losslessness gate's row independence). Every row is computed by the same
instruction sequence whatever M is: rows 1..15 of an M = 1 call are zeros in the same mma, and
17..32 a second mma. A CTA owns 128 output columns (one Hadamard block) and a contiguous K range;
its warps split that range into contiguous slices, each summed in K order by the mma's own
accumulator, and the slices are added in warp order through shared memory; a cluster of S CTAs
splits K further and rank 0 adds the ranks' partials in rank order through distributed shared
memory, then applies the output rotation. (S, WK) are a function of the weight's shape only
(`pick`), never of M, so a row's bits do not depend on the block it is verified in. No atomics.

Prefill (M > 32): the trellis is decoded to an fp16 Wi slab by a reconstruct kernel, the product
runs on the library GEMM, the rotations as above (ENG-203).

Built on first use with `torch.utils.cpp_extension.load_inline` for sm_121a, cached under
~/.cache/torch_extensions by source hash (the engine's pattern, tools/nvfp4_skinny.py).
"""

from __future__ import annotations

import hashlib
import math
import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

import torch  # noqa: E402

MUL1 = 0x83DCD12D
KS = (2, 3, 4, 5, 6, 7, 8)               # every integer rate the recipes use (head 4/5/6/8)
# ENG-285: the half-integer rates of exllamav3 1.6.0's mul1 packs below 4 bpw (1.5, 2.5, 3.5), as the
# kernels name them: ten times the rate (a tile 16 K + 8 int16 wide)
HALF_KS = (15, 25, 35)


def kb_of(width: int) -> int:
    """The kernels' bits of a trellis whose tiles are `width` int16 wide: K for 16 K, ten times the rate
    for a half-integer one (16 K + 8 -> 10 K + 5)."""
    return width // 16 if width % 16 == 0 else 10 * (width // 16) + 5


def rate(kb: int) -> float:
    """Bits a weight of the kernels' bits `kb`."""
    return kb / 10 if kb >= 10 else float(kb)


def tile_width(kb: int) -> int:
    """int16 a 256-weight tile at the kernels' bits `kb`."""
    return 16 * (kb // 10) + 8 if kb >= 10 else 16 * kb
SKINNY_MAX = 32                           # rows the decode-in-registers kernel takes in one call
# rows up to which a call runs the skinny kernel in 32-row chunks instead of decoding a slab
# (one weight read per chunk against a 2-byte write and read of the slab per weight)
CHUNK_UNTIL = int(_S.get("EXL3_CHUNK_UNTIL"))
# 33..CHUNK_UNTIL rows in one launch whose 32-row chunks are adjacent CTAs (ENG-213); 0 = one launch a chunk
ROWGRID = _S.get("EXL3_ROWGRID") == "1"
# cp.async stages of the skinny GEMM's trellis pipeline (a warp's stages in flight); the
# summation order does not depend on it
STAGES = int(_S.get("EXL3_STAGES"))
SLAB_N = 16384                            # columns a prefill slab decodes at once (fp16 Kin x SLAB_N)

_CUDA = r"""
// ---------------------------------------------------------------------------------------------
// EXL3 trellis kernels for the GB10 (sm_121). The bit-window extraction (dq8 / dq4 / dq2x2 and the
// aligned 2- and 4-bit forms) and the mul1 codebook decode are ported from exllamav3
// (exllamav3_ext/quant/exl3_dq.cuh, codebook.cuh), under its licence:
//
// MIT License -- Copyright (c) 2025 Turboderp
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions: The above copyright notice and this
// permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Everything else (the skinny GEMM, its K split and reduction, the cluster reduction, the fused
// output rotation, the rotation kernels) is this engine's.
// ---------------------------------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
namespace cg = cooperative_groups;

namespace {

struct FragB { uint32_t r[2]; };

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift) {
    uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}
#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

// two mul1 codebook entries -> one f16x2 (low half = w0), exllamav3's decode_mul1_product_2
__device__ __forceinline__ uint32_t dec2(uint32_t w0, uint32_t w1) {
    uint32_t x0 = (w0 & 0xffffu) * 0x83DCD12Du;
    uint32_t x1 = (w1 & 0xffffu) * 0x83DCD12Du;
    uint32_t s0 = __dp4a(x0, 0x01010101u, 0x6400u);
    uint32_t s1 = __dp4a(x1, 0x01010101u, 0x6400u);
    __half2 h = __halves2half2(__ushort_as_half((unsigned short)s0), __ushort_as_half((unsigned short)s1));
    __half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
    return *reinterpret_cast<uint32_t*>(&r);
}

template <int bits>
__device__ __forceinline__ void dq4(const uint32_t* ptr, int t_offset, FragB& frag) {
    int b0 = (t_offset + 257) * bits - 16;
    int b1 = b0 + 3 * bits;
    int b2 = b1 + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w3 = fshift(b, a, s2);
    uint32_t w2 = fshift(b, a, s2 + bits);
    uint32_t w1 = fshift(b, a, s2 + bits * 2);
    uint32_t w0 = fshift(b, a, s2 + bits * 3);
    frag.r[0] = dec2(w0, w1);
    frag.r[1] = dec2(w2, w3);
}

template <int bits>
__device__ __forceinline__ void dq2x2(const uint32_t* ptr, int t_offset, FragB& frag) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int b0 = (t_offset + 2 * i + 257) * bits - 16;
        int b1 = b0 + 1 * bits;
        int b2 = b1 + 16;
        int i0 = b0 / 32;
        int i2 = (b2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - b2;
        uint32_t a = ptr[i0 % (bits * 256 / 32)];
        uint32_t b = ptr[i2 % (bits * 256 / 32)];
        uint32_t w1 = fshift(b, a, s2);
        uint32_t w0 = fshift(b, a, s2 + bits);
        frag.r[i] = dec2(w0, w1);
    }
}

template <int bits, int align>
__device__ __forceinline__ void dq8(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1) {
    int b1 = (t_offset + 257) * bits;
    int b0 = b1 - 16;
    int b2 = b1 + bits * 7;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    if constexpr (align == 1) {
        w7 = fshift(b, a, s2);            w6 = fshift(b, a, s2 + bits);
        w5 = fshift(b, a, s2 + bits * 2); w4 = fshift(b, a, s2 + bits * 3);
        w3 = fshift(b, a, s2 + bits * 4); w2 = fshift(b, a, s2 + bits * 5);
        w1 = fshift(b, a, s2 + bits * 6); w0 = fshift(b, a, s2 + bits * 7);
    }
    if constexpr (align == 4) {
        w7 = fshift(b, a, s2); w6 = w7 >> bits; w5 = w6 >> bits; w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4); w2 = w3 >> bits; w1 = w2 >> bits; w0 = w1 >> bits;
    }
    frag0.r[0] = dec2(w0, w1); frag0.r[1] = dec2(w2, w3);
    frag1.r[0] = dec2(w4, w5); frag1.r[1] = dec2(w6, w7);
}

__device__ __forceinline__ void dq8_aligned_4bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1) {
    uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 3;
    i0 = (i1 + 31) & 31;
    a = ptr[i0];
    b = ptr[i1];
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    frag0.r[0] = dec2(w0, w1); frag0.r[1] = dec2(w2, w3);
    frag1.r[0] = dec2(w4, w5); frag1.r[1] = dec2(w6, w7);
}

__device__ __forceinline__ void dq8_aligned_2bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1) {
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 4;
    i0 = (i1 + 15) & 15;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    frag0.r[0] = dec2(w0, w1); frag0.r[1] = dec2(w2, w3);
    frag1.r[0] = dec2(w4, w5); frag1.r[1] = dec2(w6, w7);
}

// ENG-285: a half-integer rate (1.5, 2.5 or 3.5 bits, which exllamav3 1.6.0 gives mul1 tensors below
// 4 bpw) is ten times the rate in these kernels' bits (15, 25, 35). Its 256-weight tile is a ring of
// 256 KA + 128 bits (8 KA + 4 words, a tile 16 KA + 8 int16 wide): each weight adds KA bits and an odd
// one a bit more (exllamav3's MASK 0xAAAA), so weight i's 16-bit window ends at ring bit
// (i + 1) KA + (i + 1) / 2 (exllamav3_ext/quant/frac.cu). An integer rate's tile is 8 K words.
__host__ __device__ constexpr int tile_u32(int bits) { return bits >= 10 ? 8 * (bits / 10) + 4 : 8 * bits; }

template <int KA>
__device__ __forceinline__ void dq8_frac(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1) {
    constexpr int NW = 8 * KA + 4, R = 32 * NW;
    // t_offset is a multiple of 8: weight t_offset + j's window starts j KA + (j + 1) / 2 bits after
    // weight t_offset's, so the lane's 8 windows lie in the 3 words from the first one's
    const int s0 = ((t_offset + 1) * KA + (t_offset >> 1) - 16 + R) % R;
    const int i0 = s0 >> 5, o0 = s0 & 31;
    const uint32_t a = ptr[i0], b = ptr[(i0 + 1) % NW], c = ptr[(i0 + 2) % NW];
    uint32_t w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = o0 + j * KA + ((j + 1) >> 1);
        w[j] = r < 32 ? fshift(b, a, 48 - r) : fshift(c, b, 80 - r);
    }
    frag0.r[0] = dec2(w[0], w[1]); frag0.r[1] = dec2(w[2], w[3]);
    frag1.r[0] = dec2(w[4], w[5]); frag1.r[1] = dec2(w[6], w[7]);
}

// the 8 weights of lane (t_offset / 8) of one 16x16 tile, as two B fragments (columns 0..7, 8..15)
template <int bits>
__device__ __forceinline__ void dq_tile(const uint32_t* ptr, int idx, FragB& f0, FragB& f1) {
    if constexpr (bits >= 10) dq8_frac<bits / 10>(ptr, idx, f0, f1);
    else if constexpr (bits == 2) dq8_aligned_2bits(ptr, idx, f0, f1);
    else if constexpr (bits == 3) dq8<3, 4>(ptr, idx, f0, f1);
    else if constexpr (bits == 4) dq8_aligned_4bits(ptr, idx, f0, f1);
    else if constexpr (bits == 7) { dq2x2<7>(ptr, idx, f0); dq2x2<7>(ptr, idx + 4, f1); }
    else { dq4<bits>(ptr, idx, f0); dq4<bits>(ptr, idx + 4, f1); }   // 5, 6, 8
}

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* glob) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(glob));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int n> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(n) : "memory"); }

// natural-order Walsh-Hadamard over 128 values held 4 a lane (index 4 lane + e), unnormalised
__device__ __forceinline__ void wht128(float& v0, float& v1, float& v2, float& v3, int lane) {
    float a0 = v0 + v1, a1 = v0 - v1, a2 = v2 + v3, a3 = v2 - v3;
    v0 = a0 + a2; v1 = a1 + a3; v2 = a0 - a2; v3 = a1 - a3;
#pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        float p0 = __shfl_xor_sync(0xffffffffu, v0, i);
        float p1 = __shfl_xor_sync(0xffffffffu, v1, i);
        float p2 = __shfl_xor_sync(0xffffffffu, v2, i);
        float p3 = __shfl_xor_sync(0xffffffffu, v3, i);
        if (lane & i) { v0 = p0 - v0; v1 = p1 - v1; v2 = p2 - v2; v3 = p3 - v3; }
        else          { v0 = v0 + p0; v1 = v1 + p1; v2 = v2 + p2; v3 = v3 + p3; }
    }
}

constexpr float RSQRT128 = 0.08838834764831845f;

// ------------------------------------------------------------------------------- input rotation
// XH[g, m, :] = fp16( H(fp32(X[m, :]) * fp32(SUH[g, :])) / sqrt(128) ). grid (Kin/128/4, M, G), 4 warps.
__global__ void __launch_bounds__(128) rot_in_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const __half* __restrict__ SUH, __half* __restrict__ XH, int M, int Kin) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= Kin) return;
    const int m = blockIdx.y, g = blockIdx.z;
    const int k = blk * 128 + 4 * lane;
    const uint2 xr = *reinterpret_cast<const uint2*>(X + (size_t)m * ldx + k);
    const uint2 sr = *reinterpret_cast<const uint2*>(SUH + (size_t)g * Kin + k);
    float v0 = __uint_as_float(xr.x << 16), v1 = __uint_as_float(xr.x & 0xffff0000u);
    float v2 = __uint_as_float(xr.y << 16), v3 = __uint_as_float(xr.y & 0xffff0000u);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
    v0 *= __low2float(s01); v1 *= __high2float(s01); v2 *= __low2float(s23); v3 *= __high2float(s23);
    wht128(v0, v1, v2, v3, lane);
    __half2 o01 = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
    __half2 o23 = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
    uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(XH + ((size_t)g * M + m) * Kin + k) = o;
}

// The MLP's activation and the down projection's input rotation in one pass, bit for bit the two
// torch kernels and rot_in: A = bf16(bf16(silu(G)) * U) with silu(x) = x / (1 + exp(-x)) in fp32 (ATen's
// silu), then XH = fp16(H(fp32(A) * SUH) / sqrt(128)). Y holds gate | up: gate at column k, up at Kin + k.
__global__ void __launch_bounds__(128) rot_in_silu_kernel(const __nv_bfloat16* __restrict__ Y, int ldy,
        const __half* __restrict__ SUH, __half* __restrict__ XH, int M, int Kin) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= Kin) return;
    const int m = blockIdx.y;
    const int k = blk * 128 + 4 * lane;
    const uint2 gr = *reinterpret_cast<const uint2*>(Y + (size_t)m * ldy + k);
    const uint2 ur = *reinterpret_cast<const uint2*>(Y + (size_t)m * ldy + Kin + k);
    const uint2 sr = *reinterpret_cast<const uint2*>(SUH + k);
    float gv[4] = {__uint_as_float(gr.x << 16), __uint_as_float(gr.x & 0xffff0000u),
                   __uint_as_float(gr.y << 16), __uint_as_float(gr.y & 0xffff0000u)};
    float uv[4] = {__uint_as_float(ur.x << 16), __uint_as_float(ur.x & 0xffff0000u),
                   __uint_as_float(ur.y << 16), __uint_as_float(ur.y & 0xffff0000u)};
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float sl = __bfloat162float(__float2bfloat16_rn(gv[e] / (1.0f + ::expf(-gv[e]))));
        v[e] = __bfloat162float(__float2bfloat16_rn(sl * uv[e]));
    }
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
    float v0 = v[0] * __low2float(s01), v1 = v[1] * __high2float(s01);
    float v2 = v[2] * __low2float(s23), v3 = v[3] * __high2float(s23);
    wht128(v0, v1, v2, v3, lane);
    __half2 o01 = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
    __half2 o23 = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
    uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(XH + (size_t)m * Kin + k) = o;
}

// The MLP of a prefill slab in one pass between the two GEMMs, bit for bit rot_out (gate and up,
// fp32 products -> bf16), ATen's silu and mul, and rot_in of the down projection: per row and
// 128-column block, YG/YU the library GEMM's fp32 products of gate and up, XH the down's rotated
// input. grid (n/128/4, M), 4 warps, one 128-block a warp.
__global__ void __launch_bounds__(128) mlp_bridge_kernel(const float* __restrict__ YG, const float* __restrict__ YU,
        int ldp, const __half* __restrict__ SVG, const __half* __restrict__ SVU,
        const __half* __restrict__ SUH, __half* __restrict__ XH, int ldx, int N) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= N) return;
    const int m = blockIdx.y;
    const int n = blk * 128 + 4 * lane;
    float g[4], u[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) { g[e] = YG[(size_t)m * ldp + n + e]; u[e] = YU[(size_t)m * ldp + n + e]; }
    wht128(g[0], g[1], g[2], g[3], lane);
    wht128(u[0], u[1], u[2], u[3], lane);
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float gb = __bfloat162float(__float2bfloat16_rn(g[e] * RSQRT128 * __half2float(SVG[n + e])));
        const float ub = __bfloat162float(__float2bfloat16_rn(u[e] * RSQRT128 * __half2float(SVU[n + e])));
        const float sl = __bfloat162float(__float2bfloat16_rn(gb / (1.0f + ::expf(-gb))));
        v[e] = __bfloat162float(__float2bfloat16_rn(sl * ub)) * __half2float(SUH[n + e]);
    }
    wht128(v[0], v[1], v[2], v[3], lane);
    __half2 o01 = __floats2half2_rn(v[0] * RSQRT128, v[1] * RSQRT128);
    __half2 o23 = __floats2half2_rn(v[2] * RSQRT128, v[3] * RSQRT128);
    uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(XH + (size_t)m * ldx + n) = o;
}

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }

// ------------------------------------------------------------------------------ output rotation
// Y[m, n] = H(YP[m, block]) / sqrt(128) * SVH[n], YP fp32 or fp16, Y bf16 or fp32. grid (N/128/4, M).
template <typename TI, typename TO>
__global__ void __launch_bounds__(128) rot_out_kernel(const TI* __restrict__ YP, int ldp,
        const __half* __restrict__ SVH, TO* __restrict__ Y, int ldy, int N) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= N) return;
    const int m = blockIdx.y;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = to_f(YP[(size_t)m * ldp + n + e]);
    wht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        float r = v[e] * RSQRT128 * __half2float(SVH[n + e]);
        if constexpr (sizeof(TO) == 4) Y[(size_t)m * ldy + n + e] = r;
        else Y[(size_t)m * ldy + n + e] = __float2bfloat16_rn(r);
    }
}

// ----------------------------------------------------------------------------------- reconstruct
// WI[Kin, N] fp16 (rotated basis, row k, column n) from the trellis, columns [n_off, n_off + N).
// grid (N/128, Kin/16), 8 warps: one 16x16 tile a warp.
template <int K>
__global__ void __launch_bounds__(256) recon_kernel(const uint16_t* __restrict__ T, int ldt,
        int n_off_tiles, __half* __restrict__ WI, int ldw) {
    constexpr int TW = tile_u32(K);                // uint32 a tile
    __shared__ uint32_t s[8][TW];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kt = blockIdx.y, nt = blockIdx.x * 8 + warp;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(T) + ((size_t)kt * ldt + n_off_tiles + nt) * TW;
    for (int i = lane; i < TW; i += 32) s[warp][i] = src[i];
    __syncwarp();
    FragB f0, f1;
    dq_tile<K>(s[warp], lane * 8, f0, f1);
    const int a = lane & 3, col = lane >> 2;
    __half* out = WI + (size_t)(kt * 16) * ldw + nt * 16;
    const __half2* h00 = reinterpret_cast<const __half2*>(&f0.r[0]);
    const __half2* h01 = reinterpret_cast<const __half2*>(&f0.r[1]);
    const __half2* h10 = reinterpret_cast<const __half2*>(&f1.r[0]);
    const __half2* h11 = reinterpret_cast<const __half2*>(&f1.r[1]);
    out[(2 * a) * ldw + col]         = __low2half(*h00);
    out[(2 * a + 1) * ldw + col]     = __high2half(*h00);
    out[(2 * a + 8) * ldw + col]     = __low2half(*h01);
    out[(2 * a + 9) * ldw + col]     = __high2half(*h01);
    out[(2 * a) * ldw + col + 8]     = __low2half(*h10);
    out[(2 * a + 1) * ldw + col + 8] = __high2half(*h10);
    out[(2 * a + 8) * ldw + col + 8] = __low2half(*h11);
    out[(2 * a + 9) * ldw + col + 8] = __high2half(*h11);
}

// ---------------------------------------------------------------------------------- skinny GEMM
// Y[M, N] = rot_out( XH[member][M, Kin] @ WI ), M <= 16 MT.
// grid (S, N/128), cluster (S, 1, 1); block 32 WN WK. A CTA owns the 128 columns [128 by, +128):
// warp (wn, wk) the 16 NTW columns wn*16*NTW.. of them over its slice of the CTA's K-tile range.
// KB: bits a weight. ST: cp.async stages of a warp's (NTW tiles x one K tile) of trellis.
template <int KB, int MT, int NTW, int WN, int WK, int ST, bool OF32, bool XSM, int MINB>
__global__ void __launch_bounds__(32 * WN * WK, MINB) gemm_kernel(
        const __half* __restrict__ XH, const uint32_t* __restrict__ T, const __half* __restrict__ SVH,
        void* __restrict__ Yv, int ldy, int Mtot, int Kin, int N, int ldt, int b1, int b2, int S) {
    constexpr int TW = 8 * KB;                      // uint32 a 16x16 tile
    // XSM: the activations ride the cp.async stages with the trellis; otherwise they are loaded
    // into registers one K tile ahead (no shared memory, so more CTAs fit an SM)
    constexpr int XR = 12;                          // uint32 a staged activation row (16 fp16 + pad)
    constexpr int XW = XSM ? 16 * MT * XR : 0;      // uint32 of a stage's activation rows
    constexpr int SW = NTW * TW + XW;               // uint32 a warp's stage: trellis, then rows
    constexpr int RP = 132;                         // padded row of the reduction buffer
    extern __shared__ uint32_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wn = warp % WN, wk = warp / WN;
    const int g = lane >> 2, t = lane & 3;
    // row grid (ENG-213): blockIdx.x = chunk * S + rank. A chunk is 16 MT rows; the chunks of one
    // column block are adjacent CTAs, so the trellis the first streams is still in L2 for the next,
    // and every chunk runs the instruction sequence a call with only its rows would (row bits do not
    // depend on how many chunks there are).
    const int rank = blockIdx.x % S, chunk = blockIdx.x / S;
    const int cb = blockIdx.y;
    const int KT = Kin / 16;               // ldt: tiles a trellis row (a group member is a view)
    const int member = (cb >= b1) + (cb >= b2);
    const int M = min(16 * MT, Mtot - chunk * 16 * MT);
    const __half* xh = XH + (size_t)member * Mtot * Kin + (size_t)chunk * 16 * MT * Kin;
    Yv = OF32 ? (void*)(reinterpret_cast<float*>(Yv) + (size_t)chunk * 16 * MT * ldy)
              : (void*)(reinterpret_cast<__nv_bfloat16*>(Yv) + (size_t)chunk * 16 * MT * ldy);

    uint32_t* stage = smem + (size_t)warp * ST * SW;
    float* red = reinterpret_cast<float*>(smem + (size_t)WN * WK * ST * SW);

    // K range: the cluster's ranks, then the CTA's warps, both contiguous
    const int per_r = (KT + S - 1) / S;
    const int r0 = min(KT, rank * per_r), r1 = min(KT, r0 + per_r);
    const int per_w = (r1 - r0 + WK - 1) / WK;
    const int k0 = min(r1, r0 + wk * per_w), k1 = min(r1, k0 + per_w);
    const int nt0 = cb * 8 + wn * NTW;
    const int mrows = min(M, 16 * MT);

    // rows past M are never copied: zero them once in every stage
    if (XSM) for (int e = lane; e < ST * XW; e += 32) {
        const int st_ = e / XW, w_ = e % XW;
        if (w_ / XR >= mrows) stage[st_ * SW + NTW * TW + w_] = 0u;
    }
    __syncwarp();

    float acc[MT][NTW][2][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[a][j][h][c] = 0.f;

    // one stage: the NTW tiles of K tile kt (contiguous), and the K tile's 16 activations of
    // every row (two 16-byte chunks a row), all through cp.async
    auto issue = [&](int kt, int buf) {
        if (kt < k1) {
            const uint4* src = reinterpret_cast<const uint4*>(T + ((size_t)kt * ldt + nt0) * TW);
            uint4* dst = reinterpret_cast<uint4*>(stage + buf * SW);
            for (int c = lane; c < NTW * TW / 4; c += 32) cp_async16(dst + c, src + c);
            if (XSM) {
                uint32_t* xs = stage + buf * SW + NTW * TW;
                for (int c = lane; c < 2 * mrows; c += 32) {
                    const int r = c >> 1, h = c & 1;
                    cp_async16(xs + r * XR + 4 * h, xh + (size_t)r * Kin + kt * 16 + 8 * h);
                }
            }
        }
        cp_commit();
    };
    // the register path's operands, one K tile ahead
    uint32_t xn[MT][4];
    auto load_x = [&](int kt) {
        const int kk = kt * 16 + 2 * t;
#pragma unroll
        for (int a = 0; a < MT; ++a) {
            const int ra = 16 * a + g, rb = ra + 8;
            const uint32_t* pa = reinterpret_cast<const uint32_t*>(xh + (size_t)ra * Kin + kk);
            const uint32_t* pb = reinterpret_cast<const uint32_t*>(xh + (size_t)rb * Kin + kk);
            xn[a][0] = ra < M ? __ldg(pa) : 0u;
            xn[a][1] = rb < M ? __ldg(pb) : 0u;
            xn[a][2] = ra < M ? __ldg(pa + 4) : 0u;
            xn[a][3] = rb < M ? __ldg(pb + 4) : 0u;
        }
    };

#pragma unroll
    for (int s = 0; s < ST - 1; ++s) issue(k0 + s, s);
    if (!XSM && k0 < k1) load_x(k0);
    for (int kt = k0; kt < k1; ++kt) {
        const int i = kt - k0;
        uint32_t xa[MT][4];
        if (!XSM) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int c = 0; c < 4; ++c) xa[a][c] = xn[a][c];
            if (kt + 1 < k1) load_x(kt + 1);
        }
        issue(kt + ST - 1, (i + ST - 1) % ST);
        cp_wait<ST - 1>();
        __syncwarp();
        const uint32_t* buf = stage + (i % ST) * SW;
        if (XSM) {
            // activation operands: rows g and 8 + g, K 2t.. and 2t + 8..
            const uint32_t* xs = buf + NTW * TW;
#pragma unroll
            for (int a = 0; a < MT; ++a) {
                xa[a][0] = xs[(16 * a + g) * XR + t];
                xa[a][1] = xs[(16 * a + 8 + g) * XR + t];
                xa[a][2] = xs[(16 * a + g) * XR + t + 4];
                xa[a][3] = xs[(16 * a + 8 + g) * XR + t + 4];
            }
        }
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
            FragB f0, f1;
            dq_tile<KB>(buf + j * TW, lane * 8, f0, f1);
#pragma unroll
            for (int a = 0; a < MT; ++a) {
                mma(acc[a][j][0], xa[a][0], xa[a][1], xa[a][2], xa[a][3], f0.r[0], f0.r[1]);
                mma(acc[a][j][1], xa[a][0], xa[a][1], xa[a][2], xa[a][3], f1.r[0], f1.r[1]);
            }
        }
        __syncwarp();
    }
    cp_wait<0>();

    // the CTA's warps, in K order (wk), into the reduction buffer [16 MT][128]
    for (int step = 0; step < WK; ++step) {
        if (wk == step) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int j = 0; j < NTW; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = (wn * NTW + j) * 16 + h * 8 + 2 * t;
                        float* r_lo = red + (16 * a + g) * RP + col;
                        float* r_hi = red + (16 * a + 8 + g) * RP + col;
                        if (step == 0) {
                            r_lo[0] = acc[a][j][h][0]; r_lo[1] = acc[a][j][h][1];
                            r_hi[0] = acc[a][j][h][2]; r_hi[1] = acc[a][j][h][3];
                        } else {
                            r_lo[0] += acc[a][j][h][0]; r_lo[1] += acc[a][j][h][1];
                            r_hi[0] += acc[a][j][h][2]; r_hi[1] += acc[a][j][h][3];
                        }
                    }
        }
        __syncthreads();
    }

    const int nthreads = 32 * WN * WK;
    const int rows = min(M, 16 * MT);
    if (S > 1) {
        cg::cluster_group cluster = cg::this_cluster();
        cluster.sync();
        if (rank == 0) {
            for (int r = 1; r < S; ++r) {
                const float* rem = cluster.map_shared_rank(red, r);
                for (int e = threadIdx.x; e < rows * 128; e += nthreads) {
                    const int m = e >> 7, c = e & 127;
                    red[m * RP + c] += rem[m * RP + c];
                }
                __syncthreads();
            }
        }
        cluster.sync();
        if (rank != 0) return;
    }

    // output rotation: one warp a row
    for (int m = warp; m < rows; m += WN * WK) {
        const float* rr = red + m * RP + 4 * lane;
        float v0 = rr[0], v1 = rr[1], v2 = rr[2], v3 = rr[3];
        wht128(v0, v1, v2, v3, lane);
        const int n = cb * 128 + 4 * lane;
        const uint2 sv = *reinterpret_cast<const uint2*>(SVH + n);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
        v0 = v0 * RSQRT128 * __low2float(s01);  v1 = v1 * RSQRT128 * __high2float(s01);
        v2 = v2 * RSQRT128 * __low2float(s23);  v3 = v3 * RSQRT128 * __high2float(s23);
        if constexpr (OF32) {
            float4* y = reinterpret_cast<float4*>(reinterpret_cast<float*>(Yv) + (size_t)m * ldy + n);
            *y = make_float4(v0, v1, v2, v3);
        } else {
            __nv_bfloat162 o01 = __floats2bfloat162_rn(v0, v1), o23 = __floats2bfloat162_rn(v2, v3);
            uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(Yv) + (size_t)m * ldy + n) = o;
        }
    }
}

template <int KB, int MT, int NTW, int WN, int WK, int ST, bool OF32, bool XSM, int MINB>
void launch_gemm(const at::Tensor& xh, const at::Tensor& t, const at::Tensor& svh, at::Tensor& y,
                 int M, int Kin, int N, int ldt, int S, int b1, int b2, cudaStream_t stream) {
    constexpr int TW = 8 * KB;
    const size_t smem = (size_t)WN * WK * ST * (NTW * TW + (XSM ? 16 * MT * 12 : 0)) * 4 + (size_t)16 * MT * 132 * 4;
    auto kern = gemm_kernel<KB, MT, NTW, WN, WK, ST, OF32, XSM, MINB>;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        attr = true;
    }
    cudaLaunchConfig_t cfg = {};
    const int R = (M + 16 * MT - 1) / (16 * MT);    // row chunks (ENG-213 row grid)
    cfg.gridDim = dim3(S * R, N / 128, 1);
    cfg.blockDim = dim3(32 * WN * WK, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = S; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = S > 1 ? 1 : 0;           // one CTA a column block: a plain launch
    cudaError_t e = cudaLaunchKernelEx(&cfg, kern,
        (const __half*)xh.data_ptr(), (const uint32_t*)t.data_ptr(), (const __half*)svh.data_ptr(),
        (void*)y.data_ptr(), (int)y.stride(0), M, Kin, N, ldt, b1, b2, S);
    TORCH_CHECK(e == cudaSuccess, "exl3 gemm launch: ", cudaGetErrorString(e));
}

}  // namespace

// cfg: 0 -> MT1 NTW8 WN1 (activations staged), 1 -> MT1 NTW4 WN2 (registers, 2 CTAs an SM),
//      2 -> MT2 NTW4 WN2, 3 -> MT2 NTW2 WN4, 4 -> MT1 NTW8 WN1 (registers)
#define EXL3_WK_CASES(KB, MT, NTW, WN, OF32, XSM, MINB)                                          \
    switch (wk) {                                                                                \
        case 4: launch_gemm<KB, MT, NTW, WN, 4, EXL3_ST, OF32, XSM, MINB>(xh, t, svh, y, M, Kin, N, ldt, S, b1, b2, st); return; \
        case 8: launch_gemm<KB, MT, NTW, WN, 8, EXL3_ST, OF32, XSM, MINB>(xh, t, svh, y, M, Kin, N, ldt, S, b1, b2, st); return; \
        default: TORCH_CHECK(false, "exl3 gemm: wk must be 4 or 8, got ", wk);                   \
    }
#define EXL3_CFG_CASES(KB, OF32)                                                                 \
    switch (cfg) {                                                                               \
        case 0: EXL3_WK_CASES(KB, 1, 8, 1, OF32, true, 1)                                        \
        case 1: EXL3_WK_CASES(KB, 1, 4, 2, OF32, false, 1)                                       \
        case 2: EXL3_WK_CASES(KB, 2, 4, 2, OF32, false, 1)                                       \
        case 3: EXL3_WK_CASES(KB, 2, 2, 4, OF32, false, 1)                                       \
        case 4: EXL3_WK_CASES(KB, 1, 8, 1, OF32, false, 1)                                       \
        default: TORCH_CHECK(false, "exl3 gemm: bad cfg ", cfg);                                 \
    }
#define EXL3_KB_CASES(OF32)                                                                      \
    switch (kb) {                                                                                \
        case 2: EXL3_CFG_CASES(2, OF32)                                                          \
        case 3: EXL3_CFG_CASES(3, OF32)                                                          \
        case 4: EXL3_CFG_CASES(4, OF32)                                                          \
        case 5: EXL3_CFG_CASES(5, OF32)                                                          \
        case 6: EXL3_CFG_CASES(6, OF32)                                                          \
        case 7: EXL3_CFG_CASES(7, OF32)                                                          \
        case 8: EXL3_CFG_CASES(8, OF32)                                                          \
        default: TORCH_CHECK(false, "exl3 gemm: K bits must be 2..8, got ", kb);                 \
    }

void exl3_gemm(torch::Tensor xh, torch::Tensor t, torch::Tensor svh, torch::Tensor y,
               int64_t kb, int64_t cfg, int64_t wk, int64_t S, int64_t b1, int64_t b2, int64_t rmax) {
    const at::cuda::OptionalCUDAGuard guard(xh.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.dim() == 3, "xh: fp16 [G, M, Kin]");
    TORCH_CHECK(t.dtype() == torch::kInt16 && t.dim() == 3, "trellis: int16 [Kin/16, N/16, 16K]");
    TORCH_CHECK(t.size(2) == 16 * kb && t.stride(2) == 1 && t.stride(1) == 16 * kb, "trellis tiles must be dense, K ", kb);
    const int M = xh.size(1), Kin = xh.size(2), N = t.size(1) * 16;
    const int ldt = t.stride(0) / t.stride(1);
    TORCH_CHECK(t.size(0) * 16 == Kin, "trellis K tiles vs Kin");
    TORCH_CHECK(N % 128 == 0 && Kin % 128 == 0, "N and Kin must be multiples of 128");
    TORCH_CHECK(y.size(0) == M && y.size(1) == N && y.stride(1) == 1, "y [M, N]");
    TORCH_CHECK(M >= 1 && M <= ((cfg == 2 || cfg == 3) ? 32 : 16) * rmax, "rows ", M, " for cfg ", cfg, " and ", rmax, " row chunks");
    TORCH_CHECK(S >= 1 && S <= 16, "cluster size");
    TORCH_CHECK(svh.dtype() == torch::kHalf && svh.numel() == N, "svh fp16 [N]");
    if (y.dtype() == torch::kFloat) { EXL3_KB_CASES(true) }
    else { TORCH_CHECK(y.dtype() == torch::kBFloat16, "y bf16 or fp32"); EXL3_KB_CASES(false) }
}

void exl3_rot_in(torch::Tensor x, torch::Tensor suh, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1, "x bf16 [M, Kin]");
    TORCH_CHECK(suh.dtype() == torch::kHalf && suh.dim() == 2 && suh.is_contiguous(), "suh fp16 [G, Kin]");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous(), "xh fp16 [G, M, Kin]");
    const int M = x.size(0), Kin = x.size(1), G = suh.size(0);
    TORCH_CHECK(Kin % 128 == 0 && suh.size(1) == Kin, "Kin");
    dim3 grid((Kin / 128 + 3) / 4, M, G);
    rot_in_kernel<<<grid, 128, 0, st>>>((const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
        (const __half*)suh.data_ptr(), (__half*)xh.data_ptr(), M, Kin);
}

void exl3_rot_in_silu(torch::Tensor y, torch::Tensor suh, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(y.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.dim() == 2 && y.stride(1) == 1, "y bf16 [M, 2 Kin]");
    TORCH_CHECK(suh.dtype() == torch::kHalf && suh.dim() == 2 && suh.size(0) == 1 && suh.is_contiguous(), "suh fp16 [1, Kin]");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous(), "xh fp16 [1, M, Kin]");
    const int M = y.size(0), Kin = suh.size(1);
    TORCH_CHECK(Kin % 128 == 0 && y.size(1) == 2 * Kin && xh.numel() == (int64_t)M * Kin, "shapes");
    TORCH_CHECK(y.stride(0) % 4 == 0 && ((uintptr_t)y.data_ptr()) % 8 == 0, "y alignment");
    dim3 grid((Kin / 128 + 3) / 4, M, 1);
    rot_in_silu_kernel<<<grid, 128, 0, st>>>((const __nv_bfloat16*)y.data_ptr(), (int)y.stride(0),
        (const __half*)suh.data_ptr(), (__half*)xh.data_ptr(), M, Kin);
}

void exl3_mlp_bridge(torch::Tensor yg, torch::Tensor yu, torch::Tensor svg, torch::Tensor svu,
                     torch::Tensor suh, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(yg.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(yg.dtype() == torch::kFloat && yu.dtype() == torch::kFloat && yg.sizes() == yu.sizes(), "yg/yu fp32, same shape");
    TORCH_CHECK(yg.stride(1) == 1 && yu.stride(1) == 1 && yg.stride(0) == yu.stride(0), "yg/yu rows");
    const int M = yg.size(0), N = yg.size(1);
    TORCH_CHECK(N % 128 == 0 && svg.numel() == N && svu.numel() == N && suh.numel() == N, "bridge columns");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.stride(1) == 1 && xh.size(0) == M && xh.size(1) == N, "xh fp16 [M, N] view");
    TORCH_CHECK(svg.is_contiguous() && svu.is_contiguous() && suh.is_contiguous(), "factors contiguous");
    dim3 grid((N / 128 + 3) / 4, M);
    mlp_bridge_kernel<<<grid, 128, 0, st>>>((const float*)yg.data_ptr(), (const float*)yu.data_ptr(), (int)yg.stride(0),
        (const __half*)svg.data_ptr(), (const __half*)svu.data_ptr(), (const __half*)suh.data_ptr(),
        (__half*)xh.data_ptr(), (int)xh.stride(0), N);
}

void exl3_rot_out(torch::Tensor yp, torch::Tensor svh, torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard guard(yp.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = yp.size(0), N = yp.size(1);
    TORCH_CHECK(N % 128 == 0 && svh.numel() == N && yp.stride(1) == 1 && y.stride(1) == 1, "rot_out shapes");
    dim3 grid((N / 128 + 3) / 4, M);
    if (yp.dtype() == torch::kFloat && y.dtype() == torch::kBFloat16)
        rot_out_kernel<float, __nv_bfloat16><<<grid, 128, 0, st>>>((const float*)yp.data_ptr(), (int)yp.stride(0),
            (const __half*)svh.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (int)y.stride(0), N);
    else if (yp.dtype() == torch::kFloat && y.dtype() == torch::kFloat)
        rot_out_kernel<float, float><<<grid, 128, 0, st>>>((const float*)yp.data_ptr(), (int)yp.stride(0),
            (const __half*)svh.data_ptr(), (float*)y.data_ptr(), (int)y.stride(0), N);
    else if (yp.dtype() == torch::kHalf && y.dtype() == torch::kBFloat16)
        rot_out_kernel<__half, __nv_bfloat16><<<grid, 128, 0, st>>>((const __half*)yp.data_ptr(), (int)yp.stride(0),
            (const __half*)svh.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (int)y.stride(0), N);
    else if (yp.dtype() == torch::kHalf && y.dtype() == torch::kFloat)
        rot_out_kernel<__half, float><<<grid, 128, 0, st>>>((const __half*)yp.data_ptr(), (int)yp.stride(0),
            (const __half*)svh.data_ptr(), (float*)y.data_ptr(), (int)y.stride(0), N);
    else TORCH_CHECK(false, "rot_out dtypes");
}

void exl3_recon(torch::Tensor t, torch::Tensor wi, int64_t kb, int64_t n_off) {
    const at::cuda::OptionalCUDAGuard guard(t.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(t.dtype() == torch::kInt16 && t.size(2) == 2 * tile_u32(kb) && t.stride(2) == 1 && t.stride(1) == t.size(2), "trellis");
    TORCH_CHECK(wi.dtype() == torch::kHalf && wi.stride(1) == 1, "wi fp16");
    const int Kin = t.size(0) * 16, ntn = t.size(1), N = wi.size(1);
    const int ldt = t.stride(0) / t.stride(1);
    TORCH_CHECK(wi.size(0) == Kin && N % 128 == 0 && n_off % 128 == 0 && n_off + N <= ntn * 16, "recon shapes");
    dim3 grid(N / 128, Kin / 16);
    const uint16_t* tp = (const uint16_t*)t.data_ptr();
    __half* wp = (__half*)wi.data_ptr();
    const int ld = wi.stride(0), off = n_off / 16;
    switch (kb) {
        case 2: recon_kernel<2><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 3: recon_kernel<3><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 4: recon_kernel<4><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 5: recon_kernel<5><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 6: recon_kernel<6><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 7: recon_kernel<7><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 8: recon_kernel<8><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 15: recon_kernel<15><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 25: recon_kernel<25><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 35: recon_kernel<35><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        default: TORCH_CHECK(false, "recon: K bits 2..8, or 15 / 25 / 35 for 1.5 / 2.5 / 3.5");
    }
}
"""

_CPP = (
    "void exl3_gemm(torch::Tensor xh, torch::Tensor t, torch::Tensor svh, torch::Tensor y, "
    "int64_t kb, int64_t cfg, int64_t wk, int64_t S, int64_t b1, int64_t b2, int64_t rmax);\n"
    "void exl3_rot_in(torch::Tensor x, torch::Tensor suh, torch::Tensor xh);\n"
    "void exl3_rot_in_silu(torch::Tensor y, torch::Tensor suh, torch::Tensor xh);\n"
    "void exl3_mlp_bridge(torch::Tensor yg, torch::Tensor yu, torch::Tensor svg, torch::Tensor svu, "
    "torch::Tensor suh, torch::Tensor xh);\n"
    "void exl3_rot_out(torch::Tensor yp, torch::Tensor svh, torch::Tensor y);\n"
    "void exl3_recon(torch::Tensor t, torch::Tensor wi, int64_t kb, int64_t n_off);\n")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        src = f"#define EXL3_ST {STAGES}\n" + _CUDA
        tag = hashlib.sha1((_CPP + src).encode()).hexdigest()[:10]
        _MOD = load_inline(name=f"qwen38_exl3_{tag}", cpp_sources=[_CPP], cuda_sources=[src],
                           functions=["exl3_gemm", "exl3_rot_in", "exl3_rot_out", "exl3_recon",
                                      "exl3_rot_in_silu", "exl3_mlp_bridge"],
                           extra_cuda_cflags=["-O3", "-lineinfo",
                                              "-gencode=arch=compute_121a,code=sm_121a"],
                           verbose=False)
    return _MOD


# ------------------------------------------------------------------------------------- tiles
# (N, Kin, K bits) -> {"S": cluster split of K, "wk": warps splitting a CTA's K}. Both set the
# summation order, so they are keyed by the weight's shape and rate only; the column tiling
# ("cfg" for 1..16 rows, "cfg2" for 17..32) does not touch the order and may differ by row count.
_CONFIG: dict[tuple[int, int, int], dict] = {}
# QWEN38_EXL3_TILES unset: ops/exl3-tiles-fused.json, the fused group shapes' column tiling at the
# default order (bit-identical to default_tile, faster; 2026-10-09 A/B). "none" = default_tile only.
_TILES = _S.get("EXL3_TILES") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                              "ops", "exl3-tiles-fused.json")
if _TILES != "none" and os.path.isfile(_TILES):
    import json as _json
    for _k, _v in _json.load(open(_TILES)).items():
        if _k.startswith("_"):                  # provenance
            continue
        _n, _kk, _kb = (int(v) for v in _k.split("x"))
        _CONFIG[(_n, _kk, _kb)] = {"S": int(_v["S"]), "wk": int(_v["wk"]),
                                   "cfg": int(_v.get("cfg", 0)), "cfg2": int(_v.get("cfg2", 2))}


def default_tile(N: int, Kin: int, kb: int) -> dict:
    """The order-defining split for a shape no table names. One CTA a 128-column block is the
    best measured split for every wide shape (2026-10-08 sweep: the cluster reduction costs more
    than the 40 CTAs of a 5120-wide output lose); only narrow outputs (k/v, 8 blocks) split K
    over a cluster, keeping at least eight K tiles a warp."""
    cbs = N // 128
    kt = Kin // 16
    S = 1
    while cbs * S < 32 and S < 8 and kt // (S * 2 * 8) >= 8:
        S *= 2
    wk = 8 if kt // (S * 8) >= 8 else 4
    return {"S": S, "wk": wk, "cfg": 0, "cfg2": 2}


def pick(N: int, Kin: int, kb: int) -> dict:
    return _CONFIG.get((N, Kin, kb)) or default_tile(N, Kin, kb)


def set_config(N: int, Kin: int, kb: int, cfg: dict) -> None:
    _CONFIG[(N, Kin, kb)] = dict(cfg)


# --------------------------------------------------------------------------------- the weight
class EXL3Block:
    """One EXL3 projection (or a fused group of them sharing the input and the rate).

    Engine orientation: `shape` is (N, Kin) like every Linear here; the trellis is exllamav3's
    [Kin/16, N/16, 16 K]. For a group, the members' trellises are concatenated along N tiles, their
    suh stacked [G, Kin] (each member rotates the input with its own factors), svh concatenated;
    `sizes` are the members' N, `bounds` the 128-column block where members 2 and 3 start.
    `out_fp32`: the vocabulary head returns fp32 logits, as the e4m3 head does.
    """

    __slots__ = ("t", "suh", "svh", "kb", "N", "K", "sizes", "names", "bounds", "out_fp32", "_bf16",
                 "tile_n")

    def __init__(self, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, *,
                 sizes: list[int] | None = None, names: list[str] | None = None,
                 out_fp32: bool = False):
        assert trellis.dtype == torch.int16 and trellis.dim() == 3, (trellis.dtype, trellis.shape)
        kb = kb_of(trellis.shape[2])
        assert trellis.shape[2] == tile_width(kb) and kb in KS + HALF_KS, \
            f"trellis {tuple(trellis.shape)}: K must be 2..8, 1.5, 2.5 or 3.5"
        # dense tiles; a group member is a view (its trellis rows are the group's, ldt apart)
        self.t = trellis if (trellis.stride(2) == 1 and trellis.stride(1) == trellis.shape[2]) \
            else trellis.contiguous()
        self.kb = kb
        self.K = trellis.shape[0] * 16
        self.N = trellis.shape[1] * 16
        assert self.K % 128 == 0 and self.N % 128 == 0, (self.K, self.N)
        suh = suh.to(torch.float16)
        self.suh = (suh if suh.dim() == 2 else suh.view(1, -1)).contiguous()
        self.svh = svh.to(torch.float16).contiguous().view(-1)
        assert self.suh.shape[1] == self.K and self.svh.numel() == self.N
        self.sizes = list(sizes) if sizes else None
        self.names = list(names) if names else None
        G = self.suh.shape[0]
        assert G <= 3 and (G == 1 or (self.sizes and len(self.sizes) == G)), "a group is 2..3 members"
        if self.sizes:
            assert sum(self.sizes) == self.N and all(s % 128 == 0 for s in self.sizes), self.sizes
            starts = [sum(self.sizes[:i]) // 128 for i in range(1, len(self.sizes))]
            starts += [1 << 30] * (2 - len(starts))
            self.bounds = tuple(starts[:2])
        else:
            self.bounds = (1 << 30, 1 << 30)
        self.out_fp32 = out_fp32
        self._bf16 = None
        # the N the tile table is read at: a group member reads its group's entry, so a member
        # used on its own sums K in the order the group's launch does (one order a weight)
        self.tile_n = self.N

    @classmethod
    def group(cls, blocks: list["EXL3Block"], names: list[str]) -> "EXL3Block":
        """One launch for projections that read the same activation at the same rate. Nothing is
        duplicated in the end: the members are dropped by the caller (`Weights.fuse_exl3_groups`)."""
        assert len({b.kb for b in blocks}) == 1 and len({b.K for b in blocks}) == 1
        t = torch.cat([b.t for b in blocks], dim=1)
        suh = torch.cat([b.suh for b in blocks], dim=0)
        svh = torch.cat([b.svh for b in blocks], dim=0)
        return cls(t, suh, svh, sizes=[b.N for b in blocks], names=names)

    def member(self, i: int) -> "EXL3Block":
        """Member i of a group as its own EXL3Block, a view into the group's tensors."""
        off = sum(self.sizes[:i])
        n = self.sizes[i]
        m = EXL3Block(self.t[:, off // 16:(off + n) // 16], self.suh[i], self.svh[off:off + n],
                      out_fp32=self.out_fp32)
        m.tile_n = self.N
        return m

    @property
    def shape(self):
        return (self.N, self.K)

    @property
    def nbytes(self) -> int:
        return self.t.numel() * 2 + self.suh.numel() * 2 + self.svh.numel() * 2

    # byte accounting reads a head as numel * element_size (FP8Head does the same)
    def numel(self) -> int:
        return self.nbytes

    def element_size(self) -> int:
        return 1

    @property
    def device(self):
        return self.t.device

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """The Linear interface: `x[..., K] @ W^T` -> [..., N] (bf16; fp32 for the head)."""
        return exl3_matmul(x.reshape(-1, x.shape[-1]), self).view(*x.shape[:-1], self.N)

    def inner(self, n0: int = 0, n1: int | None = None) -> torch.Tensor:
        """Wi[:, n0:n1] in fp16 (the rotated basis), decoded on the device."""
        n1 = self.N if n1 is None else n1
        wi = torch.empty(self.K, n1 - n0, dtype=torch.float16, device=self.t.device)
        if self.t.is_cuda:
            _module().exl3_recon(self.t, wi, self.kb, n0)
        else:
            wi.copy_(decode_trellis(self.t, n0 // 16, n1 // 16))
        return wi

    def dequant(self) -> torch.Tensor:
        """The weight in the engine's [N, K] orientation, bf16, from fp32 rotations (tests, the
        drafter's NVFP4 head copy). Never on the decode path."""
        out = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.t.device)
        H = hadamard128(self.t.device)
        for g, n0, n1 in segments(self):
            wi = self.inner(n0, n1).float()                                 # [K, n]
            w = (H @ wi.view(-1, 128, n1 - n0)).view(self.K, n1 - n0)      # H . Wi
            w = w * self.suh[g].float().view(-1, 1)
            w = (w.view(self.K, -1, 128) @ H).view(self.K, n1 - n0)         # . H
            w = w * self.svh[n0:n1].float().view(1, -1)
            out[n0:n1] = w.T.to(torch.bfloat16)
        return out

    def bf16_cached(self) -> torch.Tensor:
        if self._bf16 is None:
            self._bf16 = self.dequant()
        return self._bf16


def segments(w: EXL3Block, step: int = 4096):
    """(member, n0, n1) column ranges of at most `step`, never across a member boundary."""
    starts = [0]
    for sz in (w.sizes or [w.N]):
        starts.append(starts[-1] + sz)
    for g in range(len(starts) - 1):
        for n0 in range(starts[g], starts[g + 1], step):
            yield g, n0, min(starts[g + 1], n0 + step)


def member_of(w: EXL3Block, col: int) -> int:
    if not w.sizes:
        return 0
    acc = 0
    for i, s in enumerate(w.sizes):
        acc += s
        if col < acc:
            return i
    raise IndexError(col)


_HAD: dict = {}


def hadamard128(device) -> torch.Tensor:
    """H / sqrt(128), H[i, j] = (-1)^popcount(i & j): Sylvester's construction, exllamav3's had_128."""
    key = str(device)
    if key not in _HAD:
        i = torch.arange(128)
        bits = (i.view(-1, 1) & i.view(1, -1))
        pc = torch.zeros_like(bits)
        for b in range(7):
            pc += (bits >> b) & 1
        _HAD[key] = ((1 - 2 * (pc & 1)).float() / math.sqrt(128)).to(device)
    return _HAD[key]


# ------------------------------------------------------------------- the torch reference decode
def _tile_perm() -> torch.Tensor:
    """position p (0..255) -> k * 16 + n inside a 16x16 tile (exllamav3 `tensor_core_perm`)."""
    p = torch.empty(256, dtype=torch.long)
    for lane in range(32):
        r0 = (lane % 4) * 2
        c0 = lane // 4
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        for j in range(8):
            r = rows[j & 3]
            c = c0 + 8 * (j >> 2)
            p[lane * 8 + j] = r * 16 + c
    return p


def decode_trellis(t: torch.Tensor, nt0: int = 0, nt1: int | None = None) -> torch.Tensor:
    """Pure-torch decode of trellis tiles [KT, nt0:nt1] -> Wi fp16 [Kin, 16 (nt1 - nt0)].
    Bit for bit what the kernels decode (the gate checks it); slow, for tests and the CPU path."""
    KT, NT, W = t.shape
    kb = kb_of(W)
    nt1 = NT if nt1 is None else nt1
    tt = t[:, nt0:nt1].contiguous().view(torch.int32).long() & 0xFFFFFFFF   # [KT, n, W / 2] words
    n = nt1 - nt0
    words = tt.view(KT * n, W // 2)
    nbits = 16 * W
    # the bit string, MSB-first per word: bit b of the string is bit (31 - b % 32) of word b // 32
    shifts = 31 - torch.arange(32, device=t.device)
    bitstr = ((words.unsqueeze(-1) >> shifts) & 1).view(KT * n, nbits)            # [tiles, 256 kb]
    p = torch.arange(256, device=t.device)
    # exclusive end; a half-integer rate's odd weights take one bit more (exl3_linear dq8_frac)
    end = (p + 1) * kb if kb < 10 else (p + 1) * (kb // 10) + (p + 1) // 2
    idx = (end.view(-1, 1) - 16 + torch.arange(16, device=t.device).view(1, -1)) % nbits   # [256, 16]
    win = bitstr[:, idx]                                                             # [tiles, 256, 16]
    weights = (1 << (15 - torch.arange(16, device=t.device)))
    v = (win * weights).sum(-1)                                                      # 16-bit windows
    x = (v * MUL1) & 0xFFFFFFFF
    s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
    h = (1024 + s).float()                                                           # fp16 0x6400 + s
    k_inv = torch.tensor(0x1eee, dtype=torch.int16).view(torch.float16).float()
    k_bias = torch.tensor(0xc931 - 0x10000, dtype=torch.int16).view(torch.float16).float()
    val = (h * k_inv + k_bias).to(torch.float16)                                     # exact in fp32
    perm = _tile_perm().to(t.device)
    tile = torch.empty(KT * n, 256, dtype=torch.float16, device=t.device)
    tile[:, perm] = val
    tile = tile.view(KT, n, 16, 16).permute(0, 2, 1, 3).reshape(KT * 16, n * 16)
    return tile


def reference_matmul(x: torch.Tensor, w: EXL3Block) -> torch.Tensor:
    """The product the kernels mean, in fp32 except for the two roundings the kernels make on
    purpose (the rotated input to fp16, the output to bf16/fp32): the correctness gate's yardstick."""
    H = hadamard128(x.device)
    M = x.shape[0]
    outs = []
    for g, n0, n1 in segments(w):
        xs = x.float() * w.suh[g].float()
        xh = (xs.view(M, -1, 128) @ H).view(M, w.K).to(torch.float16).float()
        yp = xh @ w.inner(n0, n1).float()
        y = (yp.view(M, -1, 128) @ H).view(M, n1 - n0) * w.svh[n0:n1].float()
        outs.append(y)
    y = torch.cat(outs, 1)
    return y if w.out_fp32 else y.to(torch.bfloat16)


# ---------------------------------------------------------------------------------- the product
def rotate_in(x: torch.Tensor, w: EXL3Block) -> torch.Tensor:
    G = w.suh.shape[0]
    xh = torch.empty(G, x.shape[0], w.K, dtype=torch.float16, device=x.device)
    _module().exl3_rot_in(x, w.suh, xh)
    return xh


# The MLP's silu(gate) * up fused into the down projection's input rotation (`exl3_down_silu`):
# the same bits as the two torch kernels and rot_in (tests/gpu/test_exl3_silu_gpu.py), two launches
# fewer a layer. 0 = the torch path.
SILU_FUSED = _S.get("EXL3_SILU_FUSED") == "1"
# A prefill MLP's gate|up products straight into the down projection's rotated input
# (`exl3_mlp_slab`): the same bits, the bf16 gate|up and the activation never written. 0 = off.
MLP_BRIDGE = _S.get("EXL3_MLP_BRIDGE") == "1"


def exl3_down_silu(y: torch.Tensor, w: EXL3Block) -> torch.Tensor:
    """down(silu(gate) * up) for y = [gate | up] (bf16 [M, 2 Kin]), w the down projection."""
    assert y.dtype == torch.bfloat16 and y.dim() == 2 and y.shape[1] == 2 * w.K and w.suh.shape[0] == 1
    if y.stride(1) != 1 or y.stride(0) % 4 or y.data_ptr() % 8:
        y = y.contiguous()
    xh = torch.empty(1, y.shape[0], w.K, dtype=torch.float16, device=y.device)
    _module().exl3_rot_in_silu(y, w.suh, xh)
    return exl3_matmul(None, w, xh=xh, M=y.shape[0], dev=y.device)


def exl3_mlp_slab(x: torch.Tensor, gu: EXL3Block, wd: EXL3Block) -> torch.Tensor | None:
    """A prefill MLP (rows past CHUNK_UNTIL) whose gate+up products go straight from the library
    GEMM's fp32 into the down projection's rotated input (`mlp_bridge_kernel`): the bits of
    rot_out -> silu * up -> rot_in without writing the bf16 gate|up and the activation. None when
    the shapes do not pair (the caller takes the usual path)."""
    F = wd.K
    if not (gu.sizes and len(gu.sizes) == 2 and gu.sizes[0] == gu.sizes[1] == F
            and wd.suh.shape[0] == 1 and x.shape[0] > CHUNK_UNTIL):
        return None
    if _mm_out_fp32() is False:
        return None
    mod = _module()
    M = x.shape[0]
    xh = rotate_in(x, gu)                                    # [2, M, K]
    segs = _slabs(gu) or [(g, n0, n1, None) for g, n0, n1 in segments(gu, SLAB_N)]
    gs = [t for t in segs if t[0] == 0]
    us = [t for t in segs if t[0] == 1]
    if len(gs) != len(us):
        return None
    xd = torch.empty(1, M, F, dtype=torch.float16, device=x.device)
    for (_, a0, a1, wg), (_, b0, b1, wu) in zip(gs, us):
        if b0 - F != a0 or b1 - a1 != b0 - a0:
            return None
        yg = _mm_fp32(xh[0], wg if wg is not None else gu.inner(a0, a1))
        yu = _mm_fp32(xh[1], wu if wu is not None else gu.inner(b0, b1))
        mod.exl3_mlp_bridge(yg, yu, gu.svh[a0:a1], gu.svh[b0:b1], wd.suh[0, a0:a1].contiguous(),
                            xd[0, :, a0:a1])
        del yg, yu
    return exl3_matmul(None, wd, xh=xd, M=M, dev=x.device)


def exl3_matmul(x: torch.Tensor | None, w: EXL3Block, *, out: torch.Tensor | None = None,
                cfg: int | None = None, S: int | None = None, wk: int | None = None,
                xh: torch.Tensor | None = None, M: int | None = None, dev=None) -> torch.Tensor:
    """Y[M, N] = x[M, K] @ W^T for an EXL3Block (or group). x bf16. `xh`: the input already
    rotated ([G, M, K] fp16, what `rotate_in(x, w)` returns) in place of x."""
    if xh is None:
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == w.K, (x.dtype, x.shape, w.shape)
        M = x.shape[0]
        if not x.is_cuda:
            return reference_matmul(x, w)
        if x.stride(1) != 1 or x.stride(0) % 4 or x.data_ptr() % 8:
            x = x.contiguous()
            if x.data_ptr() % 8:
                x = x.clone()
        dev = x.device
    odt = torch.float32 if w.out_fp32 else torch.bfloat16
    if out is None:
        out = torch.empty(M, w.N, dtype=odt, device=dev)
    mod = _module()
    # the skinny GEMM decodes integer rates; a half-integer one (an LFM2.5 pack's, ENG-285) takes the
    # slab at any row count (its rows <= 32 run on tools/fn_kernels.dense)
    if M > CHUNK_UNTIL or w.kb >= 10:
        return _slab_matmul(x, w, out, xh=xh)
    tile = pick(w.tile_n, w.K, w.kb)
    S_ = tile["S"] if S is None else S
    wk_ = tile["wk"] if wk is None else wk
    if M > SKINNY_MAX and ROWGRID:
        # ENG-213 row grid: one rotation and ONE launch for every 32-row chunk, the chunks of a
        # column block adjacent CTAs (the second reads the trellis from L2); bit for bit the
        # chunked launches below
        c = cfg if cfg is not None else tile["cfg2"]
        mod.exl3_gemm(rotate_in(x, w) if xh is None else xh, w.t, w.svh, out, w.kb, c, wk_, S_,
                      w.bounds[0], w.bounds[1], (M + SKINNY_MAX - 1) // SKINNY_MAX)
        return out
    for r0 in range(0, M, SKINNY_MAX):
        r1 = min(M, r0 + SKINNY_MAX)
        if xh is None:
            xs = rotate_in(x[r0:r1], w)
        else:
            xs = xh[:, r0:r1].contiguous()
        c = cfg if cfg is not None else (tile["cfg"] if r1 - r0 <= 16 else tile["cfg2"])
        mod.exl3_gemm(xs, w.t, w.svh, out[r0:r1], w.kb, c, wk_, S_, w.bounds[0], w.bounds[1], 1)
    return out


# The layer-major prefill's slab cache (engine/model.py `prefill_window`): while it is on, a
# weight's decoded fp16 slabs are kept from its first prefill call to `slab_cache_clear`, so the
# chunks of a window that pass one layer's weights decode each slab once instead of once a chunk.
# The slab is the same fp16 tensor either way and every GEMM keeps its shape: bit for bit the
# uncached calls. Holds one layer's projections (~0.8 GB at 4.5 bpw) between clears.
_SLAB_CACHE: dict | None = None


def slab_cache_on() -> None:
    global _SLAB_CACHE
    _SLAB_CACHE = {}


def slab_cache_clear() -> None:
    if _SLAB_CACHE is not None:
        _SLAB_CACHE.clear()


def slab_cache_off() -> None:
    global _SLAB_CACHE
    _SLAB_CACHE = None


def _slabs(w: EXL3Block) -> list:
    """(member, n0, n1, Wi[:, n0:n1]) for every slab of `w`; cached while the cache is on."""
    c = _SLAB_CACHE
    if c is None:
        return None
    got = c.get(id(w))
    if got is None or got[0] is not w:
        got = c[id(w)] = (w, [(g, n0, n1, w.inner(n0, n1)) for g, n0, n1 in segments(w, SLAB_N)])
    return got[1]


def _slab_matmul(x: torch.Tensor, w: EXL3Block, out: torch.Tensor,
                 xh: torch.Tensor | None = None) -> torch.Tensor:
    """Prefill: decode a column slab of Wi to fp16, the library GEMM, the output rotation."""
    mod = _module()
    if xh is None:
        xh = rotate_in(x, w)                                # [G, M, K]
    cached = _slabs(w)
    if cached is not None:
        for g, n0, n1, wi in cached:
            yp = _mm_fp32(xh[g], wi)
            mod.exl3_rot_out(yp, w.svh[n0:n1], out[:, n0:n1])
            del yp
        return out
    for g, n0, n1 in segments(w, SLAB_N):
        wi = w.inner(n0, n1)
        yp = _mm_fp32(xh[g], wi)
        mod.exl3_rot_out(yp, w.svh[n0:n1], out[:, n0:n1])
        del wi, yp
    return out


_MM_OUT_DTYPE: bool | None = None


def _mm_out_fp32() -> bool | None:
    """Whether the library GEMM keeps the fp32 product (None before the first prefill asked)."""
    if _MM_OUT_DTYPE is None and torch.cuda.is_available():
        a = torch.zeros(16, 16, dtype=torch.float16, device="cuda")
        _mm_fp32(a, a)
    return _MM_OUT_DTYPE


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp16 x fp16 on the library GEMM, the product kept in fp32 when torch can (`out_dtype`),
    else rounded to fp16 as exllamav3's own prefill GEMM rounds it."""
    global _MM_OUT_DTYPE
    if _MM_OUT_DTYPE is None:
        try:
            torch.mm(a[:1, :16], b[:16, :16], out_dtype=torch.float32)
            _MM_OUT_DTYPE = True
        except Exception:  # noqa: BLE001
            _MM_OUT_DTYPE = False
    return torch.mm(a, b, out_dtype=torch.float32) if _MM_OUT_DTYPE else torch.mm(a, b)


# ------------------------------------------------------------------------------- reading files
def read_block(f, base: str, *, out_fp32: bool = False) -> EXL3Block:
    """One EXL3 linear from an open safetensors handle: `base.trellis`, `base.suh`/`base.svh` (or the
    packed sign fields `su`/`sv`), the mul1 codebook only."""
    keys = set(f.keys())
    t = f.get_tensor(f"{base}.trellis")
    if f"{base}.mcg" in keys:
        raise ValueError(f"{base}: the mcg codebook is not supported (mul1 only)")
    if f"{base}.mul1" in keys:
        m = f.get_tensor(f"{base}.mul1").view(torch.int32).item() & 0xFFFFFFFF
        if m != MUL1:
            raise ValueError(f"{base}: mul1 multiplier {m:#x} != {MUL1:#x}")
    else:
        raise ValueError(f"{base}: no mul1 tensor (3inst codebook is not supported)")
    suh = f.get_tensor(f"{base}.suh") if f"{base}.suh" in keys else unpack_signs(f.get_tensor(f"{base}.su"))
    svh = f.get_tensor(f"{base}.svh") if f"{base}.svh" in keys else unpack_signs(f.get_tensor(f"{base}.sv"))
    if f"{base}.bias" in keys:
        raise ValueError(f"{base}: a bias is not supported")
    return EXL3Block(t, suh, svh, out_fp32=out_fp32)


def unpack_signs(bits: torch.Tensor) -> torch.Tensor:
    """exllamav3 `unpack_bf`: int16 bit fields -> +-1 fp16 (bit set = -1), LSB first."""
    b = bits.view(torch.int16).to(torch.int32) & 0xFFFF
    masks = 1 << torch.arange(16, device=b.device)
    e = ((b.unsqueeze(-1) & masks) > 0).flatten()
    return (1.0 - e.to(torch.float16) * 2.0).contiguous()


# ----------------------------------------------------------------------------- gate and bench
def check_block(w: EXL3Block, rows=(1, 2, 4, 8, 16, 24, 32), seed: int = 0, scale: float = 1.0,
                x: torch.Tensor | None = None) -> dict:
    """Kernel against the torch reference on `w`, and row independence, every row count."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if x is None:
        x = (torch.randn(max(rows), w.K, device="cuda", generator=g) * scale).to(torch.bfloat16)
    ref = reference_matmul(x, w).float()
    worst, single = 0.0, None
    res = {}
    for M in rows:
        y = exl3_matmul(x[:M], w).float()
        d = (y - ref[:M]).abs().max().item()
        rel = d / max(ref[:M].abs().max().item(), 1e-30)
        worst = max(worst, rel)
        if single is None:
            single = y[:1].clone()
        indep = (y[:1] - single).abs().max().item()
        res[M] = {"max_abs": d, "rel": rel, "row0_vs_M1": indep}
    res["worst_rel"] = worst
    return res


def bench(w: EXL3Block, rows=(1, 8, 16, 32), reps: int = 20, cfg=None, S=None, wk=None) -> dict:
    """Warm (L2-resident or not, depending on the size) timing of one weight, CUDA events."""
    out = {}
    for M in rows:
        x = torch.randn(M, w.K, device="cuda").to(torch.bfloat16)
        for _ in range(3):
            exl3_matmul(x, w, cfg=cfg, S=S, wk=wk)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            exl3_matmul(x, w, cfg=cfg, S=S, wk=wk)
        e1.record()
        torch.cuda.synchronize()
        us = e0.elapsed_time(e1) * 1000 / reps
        out[M] = {"us": us, "GBps": w.nbytes / us / 1e3}
    return out


def random_block(N: int, Kin: int, kb: int, device="cuda", seed: int = 0) -> EXL3Block:
    """Random trellis bits (every bit string is a valid code) and plausible channel factors."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randint(-32768, 32767, (Kin // 16, N // 16, tile_width(kb)), generator=g, dtype=torch.int32)
    t = t.to(torch.int16)
    suh = (torch.rand(Kin, generator=g) * 0.02 + 0.005) * (torch.randint(0, 2, (Kin,), generator=g) * 2 - 1)
    svh = (torch.rand(N, generator=g) * 0.5 + 0.5) * (torch.randint(0, 2, (N,), generator=g) * 2 - 1)
    return EXL3Block(t.to(device), suh.half().to(device), svh.half().to(device))
