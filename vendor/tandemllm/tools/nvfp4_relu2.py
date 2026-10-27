"""The routed relu^2 experts of a NemotronH MoE layer in NVIDIA's ModelOpt NVFP4 checkpoint (Nemotron 3.5 Super
VL 120B-A12B), and the checkpoint's per-tensor FP8 projections, for engine/models/nemotron_h.

`NVFP4Bank`: every expert's up [I, X] and down [X, I] as stored (`weight` uint8 [N, K / 2], two e2m1 codes a
byte with the even K in the low nibble; `weight_scale` e4m3 [N, K / 16]; `weight_scale_2` fp32 a tensor),
stacked [E, ...] on the device. `moe_relu2` runs each (row, slot) pair through its expert with the BF16
experts' roundings (engine/models/nemotron_h/kernels.py `moe_relu2`): u = up x rounded to bf16, a = relu(u)^2
rounded to bf16, y = down a rounded to bf16. One program a pair and 32 outputs, K walked in order: a pair's
bytes do not depend on the other pairs or the row count, so a verify block's row is the decode step's. The
products are fp32: an e2m1 code times its e4m3 scale is exact, times the bf16 activation rounds once, the
sum is fp32, and the tensor scale multiplies the sum. No fp16 anywhere, so relu^2 activations past 65,504
(the down projection's input) are taken as they are.

`FP8Linear`: a projection stored as e4m3 codes [N, K] and one fp32 scale (ModelOpt FP8, the Mamba2 and
attention projections and the head of the same checkpoint). The codes are exact in bf16; one tensor-core
product a K step in fp32, the scale on the sum, one rounding (fp32 logits for the head). A program owns
16 rows and BN outputs with a K step fixed by the shape, so a row's result is the same at 1..32 rows.
The checkpoint's `input_scale` (static FP8 activations) is not used: activations stay bf16 (W8A16).
"""

from __future__ import annotations

import os
import sys

import torch

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                  # pragma: no cover (the CPU reference still works)
    HAVE_TRITON = False

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
#: rows a program of FP8Linear covers; the row-invariant limit of both classes
ROWS_MAX = 32


def e4m3_table(device) -> torch.Tensor:
    """fp32 value of every e4m3 byte (the scales' decode: a gather, exact)."""
    return torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().to(device)


def fp4_table(device) -> torch.Tensor:
    return torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)


def dequant(w: torch.Tensor, s: torch.Tensor, s2: float) -> torch.Tensor:
    """fp32 [N, K] of one NVFP4 weight (CPU or device): code x scale x tensor scale."""
    N = w.shape[0]
    lut = fp4_table(w.device)
    v = torch.stack([lut[(w & 0x0F).long()], lut[(w >> 4).long()]], -1).reshape(N, -1)
    return v * s.float().repeat_interleave(16, 1) * float(s2)


if HAVE_TRITON:

    @triton.jit
    def _nvfp4_bank(X, s_x, IDS, W, S, S2, F4, F8, Y, NOUT: tl.constexpr, KIN: tl.constexpr, S_: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr, RELU2: tl.constexpr):
        # pair p = (row p // S_, its expert IDS[p]); BN outputs; K in steps of BK (BK / 2 bytes, BK / 16 scales)
        p = tl.program_id(0)
        nb = tl.program_id(1)
        row = p // S_
        e = tl.load(IDS + p).to(tl.int64)
        n = nb * BN + tl.arange(0, BN)
        j = tl.arange(0, BK // 2)
        wb = W + e * (NOUT * (KIN // 2)) + n[:, None] * (KIN // 2)
        sb = S + e * (NOUT * (KIN // 16)) + n[:, None] * (KIN // 16)
        acc = tl.zeros([BN], dtype=tl.float32)
        for k0 in range(0, KIN, BK):
            byte = tl.load(wb + k0 // 2 + j[None, :]).to(tl.int32)
            sc = tl.load(F8 + tl.load(sb + k0 // 16 + j[None, :] // 8).to(tl.int32))
            lo = tl.load(F4 + (byte & 15)) * sc
            hi = tl.load(F4 + (byte >> 4)) * sc
            xe = tl.load(X + row * s_x + k0 + 2 * j).to(tl.float32)
            xo = tl.load(X + row * s_x + k0 + 2 * j + 1).to(tl.float32)
            acc += tl.sum(lo * xe[None, :] + hi * xo[None, :], 1)
        acc = acc * tl.load(S2 + e)
        if RELU2:
            u = tl.maximum(acc.to(tl.bfloat16).to(tl.float32), 0.0)
            tl.store(Y + p * NOUT + n, (u * u).to(tl.bfloat16))
        else:
            tl.store(Y + p * NOUT + n, acc.to(tl.bfloat16).to(tl.float32))

    @triton.jit
    def _nvfp4_unpack(W, S, F4, F8, s2, Y, K: tl.constexpr, BK: tl.constexpr):
        # one row of one expert's weight to bf16 (a prompt chunk's cuBLAS operand)
        r = tl.program_id(0)
        kb = tl.program_id(1)
        k = kb * BK + tl.arange(0, BK)
        byte = tl.load(W + r * (K // 2) + k // 2).to(tl.int32)
        code = tl.where(k % 2 == 0, byte & 15, byte >> 4)
        sc = tl.load(F8 + tl.load(S + r * (K // 16) + k // 16).to(tl.int32))
        tl.store(Y + r * K + k, (tl.load(F4 + code) * sc * s2).to(tl.bfloat16))

    @triton.jit
    def _fp8_rows(X, s_x, W, scale, Y, s_y, M, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  F32: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        rm = rows < M
        cm = cols < N
        kk = tl.arange(0, BK)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for k0 in range(0, K, BK):
            x = tl.load(X + rows[:, None] * s_x + k0 + kk[None, :], mask=rm[:, None], other=0.0)
            w = tl.load(W + cols[:, None] * K + k0 + kk[None, :], mask=cm[:, None], other=0.0)
            acc = tl.dot(x, tl.trans(w.to(tl.bfloat16)), acc=acc, out_dtype=tl.float32)
        y = acc * tl.load(scale)
        msk = rm[:, None] & cm[None, :]
        if F32:
            tl.store(Y + rows[:, None] * s_y + cols[None, :], y, mask=msk)
        else:
            tl.store(Y + rows[:, None] * s_y + cols[None, :], y.to(tl.bfloat16), mask=msk)


class NVFP4Bank:
    """The routed experts of one MoE block from a ModelOpt NVFP4 checkpoint, stacked on the device."""

    def __init__(self, uw, us, us2, dw, ds, ds2):
        """uw [E, I, X / 2] uint8, us [E, I, X / 16] e4m3, us2 [E] fp32; dw, ds, ds2 the down side [E, X, I / 2] ..."""
        self.uw, self.us, self.dw, self.ds = uw.contiguous(), us.contiguous().view(torch.uint8), \
            dw.contiguous(), ds.contiguous().view(torch.uint8)
        self.us2, self.ds2 = us2.float().contiguous(), ds2.float().contiguous()
        self.E, self.I, half = uw.shape
        self.X = half * 2
        if tuple(dw.shape) != (self.E, self.X, self.I // 2):
            raise ValueError(f"NVFP4 experts: up {tuple(uw.shape)}, down {tuple(dw.shape)}")
        for k in (self.X, self.I):
            if k % 64:
                raise ValueError(f"NVFP4 experts need K a multiple of 64, not {k}")
        dev = uw.device
        self._f4, self._f8 = fp4_table(dev), e4m3_table(dev)

    @classmethod
    def stack(cls, experts: list, device) -> "NVFP4Bank":
        """From [(up (w, s, s2), down (w, s, s2))] a expert, host tensors as stored."""
        E = len(experts)
        (uw0, us0, _), (dw0, ds0, _) = experts[0]
        uw = torch.empty(E, *uw0.shape, dtype=torch.uint8, device=device)
        us = torch.empty(E, *us0.shape, dtype=torch.uint8, device=device)
        dw = torch.empty(E, *dw0.shape, dtype=torch.uint8, device=device)
        ds = torch.empty(E, *ds0.shape, dtype=torch.uint8, device=device)
        us2 = torch.empty(E, dtype=torch.float32)
        ds2 = torch.empty(E, dtype=torch.float32)
        for e, ((a, b, c), (d, f, g)) in enumerate(experts):
            uw[e].copy_(a)
            us[e].copy_(b.view(torch.uint8))
            us2[e] = float(c)
            dw[e].copy_(d)
            ds[e].copy_(f.view(torch.uint8))
            ds2[e] = float(g)
        return cls(uw, us, us2.to(device), dw, ds, ds2.to(device))

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.uw, self.us, self.dw, self.ds, self.us2, self.ds2))

    def expert(self, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert e's (up [I, X], down [X, I]) in bf16, for a prompt chunk's cuBLAS."""
        return self._unpack(self.uw[e], self.us[e], self.us2[e], self.I, self.X), \
            self._unpack(self.dw[e], self.ds[e], self.ds2[e], self.X, self.I)

    def _unpack(self, w, s, s2, N, K) -> torch.Tensor:
        if not w.is_cuda:
            return dequant(w, s.view(torch.float8_e4m3fn), float(s2)).to(torch.bfloat16)
        y = torch.empty(N, K, dtype=torch.bfloat16, device=w.device)
        BK = 256 if K % 256 == 0 else 64
        _nvfp4_unpack[(N, K // BK)](w, s, self._f4, self._f8, float(s2), y, K=K, BK=BK, num_warps=2)
        return y

    def moe_relu2(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """x [M, X] bf16, ids int32 [M, S] -> yd [M S, X] fp32 (bf16 values, unweighted, pair order)."""
        M, S = ids.shape
        P = M * S
        flat = ids.reshape(-1).contiguous()
        if not x.is_cuda:
            return self._reference(x, flat, S)
        a = torch.empty(P, self.I, dtype=torch.bfloat16, device=x.device)
        yd = torch.empty(P, self.X, dtype=torch.float32, device=x.device)
        _nvfp4_bank[(P, self.I // 32)](x, x.stride(0), flat, self.uw, self.us, self.us2, self._f4, self._f8, a,
                                       NOUT=self.I, KIN=self.X, S_=S, BN=32, BK=128 if self.X % 128 == 0 else 64,
                                       RELU2=True, num_warps=4)
        _nvfp4_bank[(P, self.X // 32)](a, a.stride(0), flat, self.dw, self.ds, self.ds2, self._f4, self._f8, yd,
                                       NOUT=self.X, KIN=self.I, S_=1, BN=32, BK=128 if self.I % 128 == 0 else 64,
                                       RELU2=False, num_warps=4)
        return yd

    def _reference(self, x, flat, S) -> torch.Tensor:
        out = []
        for p, e in enumerate(flat.tolist()):
            u = (dequant(self.uw[e], self.us[e].view(torch.float8_e4m3fn), float(self.us2[e])) @ x[p // S].float())
            a = torch.relu(u.to(torch.bfloat16).float()).square().to(torch.bfloat16).float()
            y = dequant(self.dw[e], self.ds[e].view(torch.float8_e4m3fn), float(self.ds2[e])) @ a
            out.append(y.to(torch.bfloat16).float())
        return torch.stack(out)


class FP8Linear:
    """A ModelOpt FP8 projection: e4m3 codes [N, K] and its fp32 tensor scale, on the device."""

    def __init__(self, codes: torch.Tensor, scale, bn: int | None = None):
        assert codes.dtype == torch.float8_e4m3fn and codes.dim() == 2, (codes.dtype, codes.shape)
        self.w = codes.contiguous()
        self.N, self.K = self.w.shape
        self.shape = (self.N, self.K)
        self.scale = torch.tensor([float(scale)], dtype=torch.float32, device=self.w.device)
        self.bk = 128 if self.K % 128 == 0 else 64
        # the N tile, fixed by the shape (never by the row count): narrow enough to fill the board at one row
        self.bn = bn or (32 if self.N <= 8192 else 64)

    @property
    def nbytes(self) -> int:
        return self.w.numel() + 4

    def dequant(self) -> torch.Tensor:
        return (self.w.float() * self.scale.cpu().item() if not self.w.is_cuda else self.w.float() * self.scale)

    def matmul(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
        x2 = x.reshape(-1, self.K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        if not x2.is_cuda or M > ROWS_MAX:
            # a prompt chunk: the codes widened to bf16 (exact) for cuBLAS, the scale on the fp32 product
            y = torch.nn.functional.linear(x2.float() if not x2.is_cuda else x2, self.w.to(x2.dtype if x2.is_cuda else torch.float32))
            y = y.float() * self.scale.to(y.device)
            y = y if f32 else y.to(torch.bfloat16)
            return (y if out is None else out.copy_(y)).view(*x.shape[:-1], self.N)
        if out is None:
            out = torch.empty(M, self.N, dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
        _fp8_rows[(triton.cdiv(M, 16), triton.cdiv(self.N, self.bn))](x2, x2.stride(0), self.w, self.scale, out,
                                                                      out.stride(0), M, self.N, K=self.K, BN=self.bn,
                                                                      BK=self.bk, F32=f32, num_warps=4)
        return out.view(*x.shape[:-1], self.N)


class NVFP4Wide:
    """A ModelOpt NVFP4 projection on tools/nvfp4_skinny (decode, verify) and tools/nvfp4_linear (prompt chunks),
    whose kernels multiply in fp16: a row whose largest value passes 2^15 is divided by a power of two first and
    the product multiplied back (both exact), so a relu^2 input (the shared expert's down projection) cannot
    overflow. Row by row, so a row's result does not depend on the others."""

    LIMIT = 2.0 ** 15

    def __init__(self, blk, wide: bool = False):
        self.blk, self.wide = blk, wide
        self.N, self.K = blk.N, blk.K
        self.shape = (blk.N, blk.K)
        self.nbytes = blk.nbytes

    def dequant(self) -> torch.Tensor:
        return self.blk.dequant()

    def matmul(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
        from tools.nvfp4_linear import nvfp4_matmul
        from tools.nvfp4_skinny import SKINNY_MAX, nvfp4_matmul_skinny
        x2 = x.reshape(-1, self.K).contiguous()
        M = x2.shape[0]
        s = None
        if self.wide:
            s = torch.exp2(torch.ceil(torch.log2(x2.abs().amax(-1, keepdim=True).float() / self.LIMIT)).clamp_min(0.0))
            x2 = (x2.float() / s).to(torch.bfloat16)
        if not x2.is_cuda:
            y = torch.nn.functional.linear(x2.float(), self.blk.dequant().float()).to(torch.bfloat16)
        elif M <= SKINNY_MAX:
            y = nvfp4_matmul_skinny(x2, self.blk, out=None if (s is not None or f32) else out)
        else:
            y = nvfp4_matmul(x2, self.blk, out=None if (s is not None or f32) else out)
        if s is not None:
            y = (y.float() * s).to(torch.bfloat16)
        if f32:
            y = y.float()
        if out is not None and y.data_ptr() != out.data_ptr():
            out.copy_(y)
            y = out
        return y.view(*x.shape[:-1], self.N)
