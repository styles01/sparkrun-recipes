"""Flash-Next prefill kernels (ENG-225): a prompt chunk of many rows on the tensor cores.

The decode and verify kernels (tools/fn_kernels.py, tools/fn_seq_kernels.py) compute every row on its
own, so a verified row is the decode step's bit for bit. A prompt chunk has no such contract: it has
to agree with the reference forward on the next token (teacher-forced argmax), not bit for bit. So
the kernels here share work across rows:

  moe_grouped   the routed and shared experts of every (row, slot) pair of a chunk, grouped by expert:
                the pairs sorted by expert id (stable), each pair's input rotated with its expert's own
                suh (`pair_rot`), then one GEMM over the experts' row groups (`group_gemm`): a CTA owns
                one item -- up to 16 MT consecutive pairs of one expert -- one matrix and one 128-column
                block, decodes the expert's trellis tiles into the mma B registers once for all of its
                rows, sums its K range over 4 warps (added in warp order), and applies the output
                rotation and svh. gate|up land in sorted order (bf16), the down projection's input is
                bf16(silu(g)) * u rotated with the down suh (`pair_silu_rot`, moe_down_body's
                arithmetic), and the down products (fp32) land at each pair's original place, so the
                decode path's `combine_inject_norm` finishes the layer. Experts with no pair are not read.
                The item table is a function of the routing on the device (counts, starts, an inclusive
                scan of items; a CTA finds its item by binary search): no host sync.
  mm_rows       x @ W^T for int8-row or fp16 weights (the hyper-connection mixers, the router, PLE and
                GDN a/b): the weight dequantised to fp16 for the call, the library GEMM, fp32 out.
  mix_tail      the mixer's stream mean from the up projection's fp32 products (`mix_up_kernel`'s
                roundings: bf16(sigmoid(bf16(u))) * n per stream, summed in stream order, / hc).

The trellis decode and the mma helpers are tools/exl3_linear.py's (`_HELPERS`), which ports the bit
extraction and the codebook from exllamav3 (MIT, Copyright (c) 2025 Turboderp; the notice travels
with that source). Built on first use for sm_121a.
"""

from __future__ import annotations

import hashlib
import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine.settings import SETTINGS as _S  # noqa: E402
from tools import exl3_linear as X  # noqa: E402

_HELPERS = X._CUDA[: X._CUDA.index("constexpr float RSQRT128")]

_CUDA = r"""
constexpr float RSQRT128 = 0.08838834764831845f;

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

struct Bank {
    const uint32_t* const* tr;    // [G, 3] trellis pointers (gate, up, down)
    const int* kb;                 // [G, 3] bits
    const __half* const* su;       // [G, 3] suh pointers
    const __half* const* sv;       // [G, 3] svh pointers
};

// A fragment order: a matrix [R, Kin] fp16 (R a multiple of 16) as [R / 16][Kin / 16][32 lanes][4 regs]
// of half2, lane l = 4 g + t and register c holding row g + 8 (c & 1), columns 2 t + 8 (c >> 1) and the
// next: the operand registers of mma.m16n8k16 (row-major A). The half offset of element (r, k):
__device__ __forceinline__ size_t frag_at(int r, int k, int KT) {
    const int rr = r & 15, kk = k & 15;
    const int lane = (rr & 7) * 4 + ((kk & 7) >> 1), reg = (rr >> 3) + 2 * (kk >> 3);
    return (((((size_t)(r >> 4) * KT + (k >> 4)) * 32 + lane) * 4 + reg) << 1) + (kk & 1);
}

__device__ __forceinline__ void store_frag4(__half* XH, int r, int k, int KT, float v0, float v1, float v2, float v3) {
    *reinterpret_cast<__half2*>(XH + frag_at(r, k, KT)) = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
    *reinterpret_cast<__half2*>(XH + frag_at(r, k + 2, KT)) = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
}

// XH[j] row PROW[i] = fp16(H(x[ROW[i]] * suh(EID[i], MAT0 + j)) / sqrt(128)), fragment order: pair i's input
// rotated with its expert's factors (rot_in_kernel's arithmetic). grid (P, Kin / 512, nmat), 4 warps, a
// 128-block a warp; XH [nmat] of R padded rows.
__global__ void __launch_bounds__(128) pair_rot_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const int* __restrict__ ROW, const int* __restrict__ EID, const int* __restrict__ PROW, Bank a, int mat0,
        __half* __restrict__ XH, int R, int Kin) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.y * 4 + warp;
    if (blk * 128 >= Kin) return;
    const int i = blockIdx.x, j = blockIdx.z;
    const int e = EID[i], m = ROW[i];
    const int k = blk * 128 + 4 * lane;
    const uint2 xr = *reinterpret_cast<const uint2*>(X + (size_t)m * ldx + k);
    const uint2 sr = *reinterpret_cast<const uint2*>(a.su[e * 3 + mat0 + j] + k);
    float v0 = bf_lo(xr.x), v1 = bf_hi(xr.x), v2 = bf_lo(xr.y), v3 = bf_hi(xr.y);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
    v0 *= __low2float(s01); v1 *= __high2float(s01); v2 *= __low2float(s23); v3 *= __high2float(s23);
    wht128(v0, v1, v2, v3, lane);
    store_frag4(XH + (size_t)j * R * Kin, PROW[i], k, Kin / 16, v0, v1, v2, v3);
}

// XH row r = fp16(H(bf16(bf16(silu(g)) * u) * suh(EID[i], down)) / sqrt(128)) from Y[r] = [g (I) | u (I)]
// bf16, r = PROW[i] (moe_down_body's arithmetic), fragment order. grid (P, I / 512), 4 warps.
__global__ void __launch_bounds__(128) pair_silu_rot_kernel(const __nv_bfloat16* __restrict__ Y,
        const int* __restrict__ EID, const int* __restrict__ PROW, Bank a, __half* __restrict__ XH, int I) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.y * 4 + warp;
    if (blk * 128 >= I) return;
    const int i = blockIdx.x, r = PROW[i];
    const int k = blk * 128 + 4 * lane;
    const __nv_bfloat16* yg = Y + (size_t)r * 2 * I;
    const uint2 gr = *reinterpret_cast<const uint2*>(yg + k);
    const uint2 ur = *reinterpret_cast<const uint2*>(yg + I + k);
    const uint2 sr = *reinterpret_cast<const uint2*>(a.su[EID[i] * 3 + 2] + k);
    const float gv[4] = {bf_lo(gr.x), bf_hi(gr.x), bf_lo(gr.y), bf_hi(gr.y)};
    const float uv[4] = {bf_lo(ur.x), bf_hi(ur.x), bf_lo(ur.y), bf_hi(ur.y)};
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = bfr(bfr(gv[q] / (1.0f + ::expf(-gv[q]))) * uv[q]);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
    float v0 = v[0] * __low2float(s01), v1 = v[1] * __high2float(s01);
    float v2 = v[2] * __low2float(s23), v3 = v[3] * __high2float(s23);
    wht128(v0, v1, v2, v3, lane);
    store_frag4(XH, r, k, I / 16, v0, v1, v2, v3);
}

// ------------------------------------------------------------------------------- the group GEMM
// One item: rows [r0, r0 + nr) of XH (fp16, fragment order, r0 a multiple of 16, nr <= 16 MT) times one
// expert matrix's 128-column block cb. 8 warps: wn = warp % 2 owns 64 columns (NTW = 4 tiles of 16), wk =
// warp / 2 a quarter of the K tiles (contiguous). The trellis tiles stream through ST cp.async stages a
// warp; the activations ride registers two K tiles ahead, one 16-byte load a lane and row tile (rows
// past nr zeroed). The four K quarters are added in order through shared memory, then one warp a row
// applies the output rotation and svh.
constexpr int GG_NTW = 4, GG_WN = 2, GG_WK = 4, GG_ST = 3, GG_RP = 132;

template <int KB, int MT, bool OF32>
__device__ __forceinline__ void group_gemm_body(const __half* __restrict__ xh, int Kin,
        const uint32_t* __restrict__ T, int ldt, const __half* __restrict__ SVH, int cb, int nr,
        void* __restrict__ Yv, int ldy, int r0, int col0, const int* __restrict__ DST, uint32_t* smem) {
    constexpr int TW = tile_u32(KB);
    constexpr int SW = GG_NTW * TW;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wn = warp % GG_WN, wk = warp / GG_WN;
    const int g = lane >> 2, t = lane & 3;
    uint32_t* stage = smem + (size_t)warp * GG_ST * SW;
    float* red = reinterpret_cast<float*>(smem + (size_t)GG_WN * GG_WK * GG_ST * SW);
    const int KT = Kin / 16;
    const int per_w = (KT + GG_WK - 1) / GG_WK;
    const int k0 = min(KT, wk * per_w), k1 = min(KT, k0 + per_w);
    const int nt0 = cb * 8 + wn * GG_NTW;

    float acc[MT][GG_NTW][2][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int j = 0; j < GG_NTW; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[a][j][h][c] = 0.f;

    auto issue = [&](int kt, int buf) {
        if (kt < k1) {
            const uint4* src = reinterpret_cast<const uint4*>(T + ((size_t)kt * ldt + nt0) * TW);
            uint4* dst = reinterpret_cast<uint4*>(stage + buf * SW);
            for (int c = lane; c < SW / 4; c += 32) cp_async16(dst + c, src + c);
        }
        cp_commit();
    };
    // xh: the item's first row tile; a row tile of one K tile is 32 lanes x 16 bytes
    const uint4* xf = reinterpret_cast<const uint4*>(xh);
    uint4 n1[MT], n2[MT];
    auto load_x = [&](int kt, uint4* dst) {
#pragma unroll
        for (int a = 0; a < MT; ++a)
            dst[a] = 16 * a < nr ? __ldg(xf + ((size_t)a * KT + kt) * 32 + lane) : make_uint4(0u, 0u, 0u, 0u);
    };
#pragma unroll
    for (int s = 0; s < GG_ST - 1; ++s) issue(k0 + s, s);
    if (k0 < k1) load_x(k0, n1);
    if (k0 + 1 < k1) load_x(k0 + 1, n2);
    for (int kt = k0; kt < k1; ++kt) {
        const int i = kt - k0;
        uint32_t xa[MT][4];
#pragma unroll
        for (int a = 0; a < MT; ++a) {
            const uint4 v = n1[a];
            const bool lo = 16 * a + g < nr, hi = 16 * a + 8 + g < nr;      // rows g and g + 8 of the tile
            xa[a][0] = lo ? v.x : 0u; xa[a][1] = hi ? v.y : 0u; xa[a][2] = lo ? v.z : 0u; xa[a][3] = hi ? v.w : 0u;
            n1[a] = n2[a];
        }
        if (kt + 2 < k1) load_x(kt + 2, n2);
        issue(kt + GG_ST - 1, (i + GG_ST - 1) % GG_ST);
        cp_wait<GG_ST - 1>();
        __syncwarp();
        const uint32_t* buf = stage + (i % GG_ST) * SW;
#pragma unroll
        for (int j = 0; j < GG_NTW; ++j) {
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
    for (int step = 0; step < GG_WK; ++step) {
        if (wk == step) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int j = 0; j < GG_NTW; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = (wn * GG_NTW + j) * 16 + h * 8 + 2 * t;
                        float* lo = red + (16 * a + g) * GG_RP + col;
                        float* hi = red + (16 * a + 8 + g) * GG_RP + col;
                        if (step == 0) { lo[0] = acc[a][j][h][0]; lo[1] = acc[a][j][h][1]; hi[0] = acc[a][j][h][2]; hi[1] = acc[a][j][h][3]; }
                        else { lo[0] += acc[a][j][h][0]; lo[1] += acc[a][j][h][1]; hi[0] += acc[a][j][h][2]; hi[1] += acc[a][j][h][3]; }
                    }
        }
        __syncthreads();
    }
    const int n = cb * 128 + 4 * lane;
    const uint2 sv = *reinterpret_cast<const uint2*>(SVH + n);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
    for (int m = warp; m < nr; m += GG_WN * GG_WK) {
        const float* rr = red + m * GG_RP + 4 * lane;
        float v0 = rr[0], v1 = rr[1], v2 = rr[2], v3 = rr[3];
        wht128(v0, v1, v2, v3, lane);
        v0 = v0 * RSQRT128 * __low2float(s01);  v1 = v1 * RSQRT128 * __high2float(s01);
        v2 = v2 * RSQRT128 * __low2float(s23);  v3 = v3 * RSQRT128 * __high2float(s23);
        const size_t row = DST ? (size_t)DST[r0 + m] : (size_t)(r0 + m);
        if constexpr (OF32) {
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(Yv) + row * ldy + col0 + n) = make_float4(v0, v1, v2, v3);
        } else {
            __nv_bfloat162 o01 = __floats2bfloat162_rn(v0, v1), o23 = __floats2bfloat162_rn(v2, v3);
            uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(Yv) + row * ldy + col0 + n) = o;
        }
    }
}

// grid (items upper bound x nmat N / 128), 256 threads. Groups g = 0..G-1 own CNT[g] consecutive rows
// of XH (P rows a matrix, fragment order) from PSTART[g] (a multiple of 16); IEND[g] = items of groups
// 0..g, an item = up to 16 MT rows. blockIdx.x = item (nmat N / 128) + (matrix j, column block cb): the
// CTAs of one item are adjacent, so its rows stream from memory once and from L2 for the others; matrix
// MAT0 + j of group g reads XH[j]. Output row: DST[row] (null: the row), columns j N + n.
template <int MT, bool OF32>
__global__ void __launch_bounds__(256) group_gemm_kernel(const __half* __restrict__ XH, int P, int Kin, int N,
        const int* __restrict__ CNT, const int* __restrict__ PSTART, const int* __restrict__ IEND, int G,
        Bank a, int mat0, int nmat, void* __restrict__ Y, int ldy, const int* __restrict__ DST) {
    extern __shared__ uint32_t smem[];
    const int ncb = N / 128, nyc = nmat * ncb;
    const int item = blockIdx.x / nyc, yc = blockIdx.x % nyc;
    if (item >= IEND[G - 1]) return;
    int lo = 0, hi = G - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (IEND[mid] > item) hi = mid; else lo = mid + 1;
    }
    const int gi = lo;
    const int j = item - (gi ? IEND[gi - 1] : 0);
    const int r0 = PSTART[gi] + j * 16 * MT;
    const int nr = min(16 * MT, CNT[gi] - j * 16 * MT);
    const int jm = yc / ncb, cb = yc % ncb;
    const int mat = mat0 + jm;
    const __half* xh = XH + ((size_t)jm * P + r0) * Kin;          // r0 / 16 row tiles in
    const uint32_t* T = a.tr[gi * 3 + mat];
    const __half* svh = a.sv[gi * 3 + mat];
    const int ldt = N / 16;
    switch (a.kb[gi * 3 + mat]) {
        case 2: group_gemm_body<2, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 3: group_gemm_body<3, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 4: group_gemm_body<4, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 5: group_gemm_body<5, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 6: group_gemm_body<6, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 8: group_gemm_body<8, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 7: group_gemm_body<7, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 15: group_gemm_body<15, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 25: group_gemm_body<25, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 35: group_gemm_body<35, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        default: break;
    }
}

// ----------------------------------------------------------------------------------- mixer tail
// X[m, j] = bf16(sum over streams s in order of bf16(bf16(sigmoid(bf16(U[m, s H + j]))) * Nn[m, s H + j]) / hc)
// (mix_up_kernel's roundings after its up projection). grid (H / 256, M), 256 threads, hc = 4.
__global__ void __launch_bounds__(256) mix_tail_kernel(const float* __restrict__ U, const __nv_bfloat16* __restrict__ Nn,
        int ldn, __nv_bfloat16* __restrict__ Xo, int ldx, int H, float inv_hc) {
    const int j = blockIdx.x * 256 + threadIdx.x, m = blockIdx.y;
    if (j >= H) return;
    const float* u = U + (size_t)m * 4 * H + j;
    const __nv_bfloat16* nn = Nn + (size_t)m * ldn + j;
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float sg = bfr(1.0f / (1.0f + ::expf(-bfr(u[(size_t)s * H]))));
        acc += bfr(sg * __bfloat162float(nn[(size_t)s * H]));
    }
    Xo[(size_t)m * ldx + j] = __float2bfloat16_rn(acc * inv_hc);
}

}  // namespace

static Bank bank_of(const torch::Tensor& tr, const torch::Tensor& kb, const torch::Tensor& su, const torch::Tensor& sv) {
    Bank a;
    a.tr = (const uint32_t* const*)tr.data_ptr();
    a.kb = (const int*)kb.data_ptr();
    a.su = (const __half* const*)su.data_ptr();
    a.sv = (const __half* const*)sv.data_ptr();
    return a;
}

void pre_pair_rot(torch::Tensor x, torch::Tensor row, torch::Tensor eid, torch::Tensor prow, torch::Tensor tr,
                  torch::Tensor kb, torch::Tensor su, torch::Tensor sv, int64_t mat0, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x bf16 [M, Kin]");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.dim() == 3 && xh.size(1) % 16 == 0,
                "xh fp16 [nmat, R, Kin], R padded rows (a multiple of 16)");
    const int nmat = xh.size(0), R = xh.size(1), Kin = xh.size(2), P = row.numel();
    TORCH_CHECK(row.dtype() == torch::kInt && eid.dtype() == torch::kInt && prow.dtype() == torch::kInt
                && eid.numel() == P && prow.numel() == P, "row, eid, prow int32 [P]");
    TORCH_CHECK(Kin % 128 == 0 && x.size(1) == Kin, "Kin");
    if (P == 0) return;
    pair_rot_kernel<<<dim3(P, (Kin / 128 + 3) / 4, nmat), 128, 0, st>>>((const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
        (const int*)row.data_ptr(), (const int*)eid.data_ptr(), (const int*)prow.data_ptr(), bank_of(tr, kb, su, sv),
        (int)mat0, (__half*)xh.data_ptr(), R, Kin);
}

void pre_pair_silu_rot(torch::Tensor y, torch::Tensor eid, torch::Tensor prow, torch::Tensor tr, torch::Tensor kb,
                       torch::Tensor su, torch::Tensor sv, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(y.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int R = y.size(0), I = y.size(1) / 2, P = eid.numel();
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.is_contiguous() && I % 128 == 0, "y bf16 [R, 2 I]");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.numel() == (int64_t)R * I, "xh fp16 [R, I]");
    TORCH_CHECK(prow.dtype() == torch::kInt && prow.numel() == P, "prow int32 [P]");
    if (P == 0) return;
    pair_silu_rot_kernel<<<dim3(P, (I / 128 + 3) / 4), 128, 0, st>>>((const __nv_bfloat16*)y.data_ptr(),
        (const int*)eid.data_ptr(), (const int*)prow.data_ptr(), bank_of(tr, kb, su, sv), (__half*)xh.data_ptr(), I);
}

template <int MT, bool OF32>
static void gg_launch(const torch::Tensor& xh, int P, int Kin, int N, const torch::Tensor& cnt, const torch::Tensor& pstart,
                      const torch::Tensor& iend, int G, Bank a, int mat0, int nmat, torch::Tensor& y,
                      const int* dst, int items, int kbmax, cudaStream_t st) {
    const size_t smem = (size_t)GG_WN * GG_WK * GG_ST * GG_NTW * tile_u32(kbmax) * 4 + (size_t)16 * MT * GG_RP * 4;
    auto kern = group_gemm_kernel<MT, OF32>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    kern<<<dim3(items * nmat * (N / 128)), 256, smem, st>>>((const __half*)xh.data_ptr(), P, Kin, N,
        (const int*)cnt.data_ptr(), (const int*)pstart.data_ptr(), (const int*)iend.data_ptr(), G, a, mat0, nmat,
        y.data_ptr(), (int)y.stride(0), dst);
}

void pre_group_gemm(torch::Tensor xh, torch::Tensor cnt, torch::Tensor pstart, torch::Tensor iend, torch::Tensor tr,
                    torch::Tensor kb, torch::Tensor su, torch::Tensor sv, int64_t mat0, int64_t N, torch::Tensor y,
                    c10::optional<torch::Tensor> dst, int64_t items, int64_t mt, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(xh.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.dim() == 3, "xh fp16 [nmat, P, Kin]");
    const int nmat = xh.size(0), P = xh.size(1), Kin = xh.size(2), G = cnt.numel();
    TORCH_CHECK(cnt.dtype() == torch::kInt && pstart.dtype() == torch::kInt && iend.dtype() == torch::kInt
                && pstart.numel() == G && iend.numel() == G, "cnt, pstart, iend int32 [G]");
    TORCH_CHECK(Kin % 128 == 0 && N % 128 == 0 && ((kbmax >= 2 && kbmax <= 8) || kbmax == 15 || kbmax == 25 || kbmax == 35),
                "shapes (K 2..8, or 15 / 25 / 35 for 1.5 / 2.5 / 3.5)");
    TORCH_CHECK(y.stride(1) == 1 && y.size(1) >= nmat * N, "y [rows, >= nmat N]");
    TORCH_CHECK(P % 16 == 0, "rows a matrix: a multiple of 16 (fragment order)");
    TORCH_CHECK(!dst.has_value() || (dst->dtype() == torch::kInt && dst->numel() == P), "dst int32 [P]");
    if (items == 0 || P == 0) return;
    const int* dp = dst.has_value() ? (const int*)dst->data_ptr() : nullptr;
    const Bank a = bank_of(tr, kb, su, sv);
    if (y.dtype() == torch::kFloat) {
        if (mt == 4) gg_launch<4, true>(xh, P, Kin, N, cnt, pstart, iend, G, a, mat0, nmat, y, dp, items, kbmax, st);
        else gg_launch<2, true>(xh, P, Kin, N, cnt, pstart, iend, G, a, mat0, nmat, y, dp, items, kbmax, st);
    } else {
        TORCH_CHECK(y.dtype() == torch::kBFloat16, "y bf16 or fp32");
        if (mt == 4) gg_launch<4, false>(xh, P, Kin, N, cnt, pstart, iend, G, a, mat0, nmat, y, dp, items, kbmax, st);
        else gg_launch<2, false>(xh, P, Kin, N, cnt, pstart, iend, G, a, mat0, nmat, y, dp, items, kbmax, st);
    }
}

void pre_mix_tail(torch::Tensor u, torch::Tensor n, torch::Tensor x, int64_t hc) {
    const at::cuda::OptionalCUDAGuard guard(u.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = x.size(0), H = x.size(1);
    TORCH_CHECK(hc == 4 && u.dtype() == torch::kFloat && u.is_contiguous() && u.size(0) == M && u.size(1) == 4 * H, "u fp32 [M, 4 H]");
    TORCH_CHECK(n.dtype() == torch::kBFloat16 && n.stride(1) == 1 && n.size(1) == 4 * H, "n bf16 [M, 4 H]");
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1, "x bf16 [M, H]");
    if (M == 0) return;
    mix_tail_kernel<<<dim3((H + 255) / 256, M), 256, 0, st>>>((const float*)u.data_ptr(), (const __nv_bfloat16*)n.data_ptr(),
        (int)n.stride(0), (__nv_bfloat16*)x.data_ptr(), (int)x.stride(0), H, 1.0f / (float)hc);
}
"""

_CPP = (
    "void pre_pair_rot(torch::Tensor x, torch::Tensor row, torch::Tensor eid, torch::Tensor prow, torch::Tensor tr, "
    "torch::Tensor kb, torch::Tensor su, torch::Tensor sv, int64_t mat0, torch::Tensor xh);\n"
    "void pre_pair_silu_rot(torch::Tensor y, torch::Tensor eid, torch::Tensor prow, torch::Tensor tr, torch::Tensor kb, "
    "torch::Tensor su, torch::Tensor sv, torch::Tensor xh);\n"
    "void pre_group_gemm(torch::Tensor xh, torch::Tensor cnt, torch::Tensor pstart, torch::Tensor iend, torch::Tensor tr, "
    "torch::Tensor kb, torch::Tensor su, torch::Tensor sv, int64_t mat0, int64_t N, torch::Tensor y, "
    "c10::optional<torch::Tensor> dst, int64_t items, int64_t mt, int64_t kbmax);\n"
    "void pre_mix_tail(torch::Tensor u, torch::Tensor n, torch::Tensor x, int64_t hc);\n")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        src = "#define EXL3_ST 3\n" + _HELPERS + _CUDA
        tag = hashlib.sha1((_CPP + src).encode()).hexdigest()[:10]
        _MOD = load_inline(name=f"fn_prefill_kernels_{tag}", cpp_sources=[_CPP], cuda_sources=[src],
                           functions=["pre_pair_rot", "pre_pair_silu_rot", "pre_group_gemm", "pre_mix_tail"],
                           extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                           verbose=False)
    return _MOD


# rows an item of the group GEMM takes (16 MT; 32 or 64): 32 measured faster at 2k and 4k rows a chunk
# (2026-10-11, tools/fn_moe_prefill_bench.py: 9.7 vs 10.7 ms, 15.8 vs 17.2 ms a layer)
GROUP_ROWS = 32


def group_plan(ids_flat: torch.Tensor, groups: int, rows: int = GROUP_ROWS) -> dict:
    """The pairs of a chunk grouped by expert, on the device, no host sync:

      order   int64 [P]      the pairs in ascending expert id (stable)
      eid     int32 [P]      the sorted pairs' experts
      cnt     int32 [G]      pairs an expert
      pstart  int32 [G]      an expert's first row in the padded layout (its rows padded to 16)
      prow    int32 [P]      each sorted pair's row in the padded layout
      iend    int32 [G]      the items of experts 0..g (an item = up to `rows` pairs)
      R       int            padded rows (an upper bound: P + 15 G, a multiple of 16)
      items   int            the item count's upper bound
    """
    P = ids_flat.numel()
    order = torch.sort(ids_flat, stable=True).indices
    eid = ids_flat.index_select(0, order).to(torch.int32)
    cnt = torch.bincount(ids_flat.long(), minlength=groups)
    first = torch.cumsum(cnt, 0) - cnt
    pad = (cnt + 15) // 16 * 16
    pstart = torch.cumsum(pad, 0) - pad
    e = eid.long()
    prow = (pstart.index_select(0, e) + torch.arange(P, device=ids_flat.device) - first.index_select(0, e)).to(torch.int32)
    iend = torch.cumsum((cnt + rows - 1) // rows, 0).to(torch.int32)
    return {"order": order, "eid": eid, "cnt": cnt.to(torch.int32), "pstart": pstart.to(torch.int32), "prow": prow,
            "iend": iend, "R": (P + 15 * groups + 15) // 16 * 16, "items": P // rows + groups}


def moe_grouped(x: torch.Tensor, ids: torch.Tensor, bank, rows: int = GROUP_ROWS) -> torch.Tensor:
    """Every (row, slot) pair's expert output, unweighted: x [M, H] bf16, ids int32 [M, S] (the router's,
    the shared expert as id E) -> yd [M S, H] fp32 in the pairs' own order (`combine_inject_norm`'s
    input, as `fn_kernels.moe_experts` returns it)."""
    M, S = ids.shape
    P = M * S
    mod = _module()
    g = group_plan(ids.reshape(-1), bank.n, rows)
    R, prow = g["R"], g["prow"]
    row = torch.div(g["order"], S, rounding_mode="floor").to(torch.int32)
    kbmax = bank.ks[-1]
    xh = torch.empty(2, R, bank.H, dtype=torch.float16, device=x.device)
    mod.pre_pair_rot(x, row, g["eid"], prow, bank.tr, bank.kb, bank.su, bank.sv, 0, xh)
    ygu = torch.empty(R, 2 * bank.I, dtype=torch.bfloat16, device=x.device)
    mod.pre_group_gemm(xh, g["cnt"], g["pstart"], g["iend"], bank.tr, bank.kb, bank.su, bank.sv, 0, bank.I, ygu, None,
                       g["items"], rows // 16, kbmax)
    del xh
    xd = torch.empty(1, R, bank.I, dtype=torch.float16, device=x.device)
    mod.pre_pair_silu_rot(ygu, g["eid"], prow, bank.tr, bank.kb, bank.su, bank.sv, xd)
    del ygu
    dst = torch.empty(R, dtype=torch.int32, device=x.device)
    dst.index_copy_(0, prow.long(), g["order"].to(torch.int32))
    yd = torch.empty(P, bank.H, dtype=torch.float32, device=x.device)
    mod.pre_group_gemm(xd, g["cnt"], g["pstart"], g["iend"], bank.tr, bank.kb, bank.su, bank.sv, 2, bank.H, yd, dst,
                       g["items"], rows // 16, kbmax)
    return yd


# the library GEMM of `mm_rows`: "fp32" = the exact weights (int8 times its scale in fp32) and fp32 sums
# (the decode path's GEMV arithmetic up to summation order), "fp16" = weights rounded to fp16 on the
# tensor cores (QWEN38_FN_PREFILL_MM). 2026-10-11: fp16 failed the agreement gate at 3k code (kernels vs
# decode path 95.6 % against the decode path's 97.2 % with the reference), fp32 passed every context
# and class at 1k/3k/8k, at 1,243 vs 1,441 tok/s (8k); two fp16 terms failed too and were slower.
MM = _S.get("FN_PREFILL_MM")


def mm_rows(x: torch.Tensor, w) -> torch.Tensor:
    """x [M, K] bf16 @ W^T -> fp32 [M, N] on the library GEMM: W int8 with fp32 scales (a row's or a
    group's, `pack.Int8Rows`) or fp16 [N, K]. `MM` fp32: the weights exact in fp32, fp32 sums; fp16: int8 rows
    dequantised to fp16 (q s in fp32, rounded once), x rounded to fp16, the tensor cores."""
    if MM == "fp32":
        wf = w.dequant() if hasattr(w, "q") else w.float()
        return x.float() @ wf.T
    if hasattr(w, "q"):
        N = w.q.shape[0]
        sc = w.s.view(N, -1, 1)                          # a scale a row, or a group of the row's weights
        wf = torch.empty(w.q.shape, dtype=torch.float16, device=w.q.device)
        torch.mul(w.q.view(N, sc.shape[1], -1), sc, out=wf.view(N, sc.shape[1], -1))
    else:
        wf = w if w.dtype == torch.float16 else w.half()
    return X._mm_fp32(x.half(), wf.T)


def qsa_scores_rows(iq: torch.Tensor, pool: torch.Tensor, nblk: torch.Tensor, NB: int, block: int = 1024) -> torch.Tensor:
    """The indexer's block scores of a chunk's rows on the library GEMM, the reference's arithmetic
    (`ref.qsa_mask`: fp32 products, relu, summed over the heads in order, / sqrt(head dim)): iq [M, NH, D]
    bf16, pool [blocks, D] bf16 (every complete block pooled; may hold fewer than NB rows), nblk int [M]
    (a row's complete blocks)
    -> fp32 [M, NB], -inf past a row's complete blocks. In blocks of `block` rows (the products of one
    block are [block NH, NB] fp32)."""
    M, NH, D = iq.shape
    nb = min(NB, pool.shape[0])              # the score width may pass the pool (a cache under NB blocks)
    pk = pool[:nb].float().T
    out = torch.full((M, NB), float("-inf"), dtype=torch.float32, device=iq.device)
    cols = torch.arange(nb, device=iq.device)
    for r0 in range(0, M, block):
        r1 = min(M, r0 + block)
        sc = torch.relu(iq[r0:r1].reshape(-1, D).float() @ pk).view(r1 - r0, NH, nb)
        acc = sc[:, 0]
        for h in range(1, NH):
            acc = acc + sc[:, h]
        out[r0:r1, :nb] = (acc / D ** 0.5).masked_fill(cols[None, :] >= nblk[r0:r1, None], float("-inf"))
    return out


def mix_tail(u: torch.Tensor, n: torch.Tensor, hc: int) -> torch.Tensor:
    """The mixer's stream mean from the up projection's products u [M, hc H] fp32 and the normed
    streams n [M, hc H] bf16 -> x [M, H] bf16."""
    x = torch.empty(n.shape[0], n.shape[1] // hc, dtype=torch.bfloat16, device=n.device)
    _module().pre_mix_tail(u, n, x, hc)
    return x


def moe_grouped_reference(x: torch.Tensor, ids: torch.Tensor, bank) -> torch.Tensor:
    """`moe_grouped` in torch, fp32 products (tests): each pair through its expert's dequantised
    matrices with the kernels' roundings (gate/up to bf16, the activation bf16(bf16(silu(g)) u))."""
    M, S = ids.shape
    out = torch.zeros(M * S, bank.H, dtype=torch.float32, device=x.device)
    flat = ids.reshape(-1).tolist()
    for p, e in enumerate(flat):
        xr = x[p // S:p // S + 1]
        g = X.reference_matmul(xr, bank.block(e, 0)).float()
        u = X.reference_matmul(xr, bank.block(e, 1)).float()
        a = (g / (1.0 + torch.exp(-g))).to(torch.bfloat16).float() * u
        out[p] = X.reference_matmul(a.to(torch.bfloat16), _fp32(bank.block(e, 2)))[0]
    return out


def _fp32(b):
    b.out_fp32 = True
    return b
