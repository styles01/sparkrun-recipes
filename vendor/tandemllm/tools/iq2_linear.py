"""GGUF IQ2_XXS, NVFP4 and Q8_0 weights: the dequant twins from the format, and row-invariant GEMMs
that decode the blocks on the fly.

IQ2_XXS (ggml type 16, 2.0625 bits a weight): a block of 256 weights in 66 bytes, an fp16 scale `d` and
eight groups of 32 weights, each group two u32 words:

  q0   four bytes, each an index into the 256-entry E8 lattice grid (`iq2xxs_grid`: 8 magnitudes of
       8, 25 or 43 an entry)
  q1   bits 0..27: four 7-bit sign indices (`ksigns_iq2xs`: the 7 bits and an eighth that makes the
       parity even); bits 28..31: the group's 4-bit scale s

  w = d * (0.5 + s) * 0.25 * grid[q0 byte l][j] * (sign bit j of ksigns[q1 >> 7 l] ? -1 : 1)

in fp32 in that order (llama.cpp's `dequantize_row_iq2_xxs` and gguf-py's `IQ2_XXS.dequantize_blocks`;
the grid and sign tables are ggml's, MIT). `dequant_iq2_xxs` is the twin; tests/test_iq2_kernels.py
holds it to gguf-py bit for bit. Q8_0 (type 8): 32 int8 with an fp16 scale, w = d * q.

NVFP4 (ggml type 40, 4.5 bits a weight): a block of 64 weights in 36 bytes, four UE4M3 scales (one a
sub-block of 16) and 32 bytes of E2M1 codes; byte j of sub-block s holds weight 16 s + j in its low
nibble and 16 s + j + 8 in its high one:

  w = kvalues[code] * ue4m3(d[s])      kvalues = E2M1 doubled (0, 1, 2, 3, 4, 6, 8, 12, and negatives),
                                       ue4m3 = (1 + m / 8) 2^(e - 7) (m 2^-9 at e = 0, 0 at 0 and 0x7F) / 2

(llama.cpp's `dequantize_row_nvfp4`, gguf-py's `NVFP4.dequantize_blocks`); every product is exact in fp32.
A tensor may carry a per-tensor fp32 scale (`<name>.scale`, shape [1]: Bolt's Mamba2 in_proj and
out_proj), which llama.cpp multiplies into the matmul's fp32 output (`build_lora_mm`); `NVFP4Linear`
does the same before its one rounding.

These are the values of the checkpoints' NVFP4 (tools/nvfp4_linear.py's `NVFP4Block`: E2M1 codes, an
E4M3 scale a group of 16, the per-tensor scale), since kvalues = 2 E2M1 and ue4m3 = E4M3 / 2, and the
code nibbles mean the same values. `repack_nvfp4` moves a GGUF tensor into that layout without changing
a value (the nibbles reordered to two consecutive weights a byte, the scale bytes kept with bit 7, which
ggml does not read, cleared, and 0x7F, which ggml reads as zero, written as zero), so on the GPU the
rows run on the 27B's row-invariant tensor-core kernel (tools/nvfp4_skinny.py) and a prompt chunk on
`nvfp4_matmul`; the Triton GEMM below serves the CPU interpreter.

`IQ2Linear` is the Linear interface (`shape`, `nbytes`, `matmul`) over a packed [N, K] tensor (rows of
K / 256 blocks, as stored): `iq2_gemm` decodes 4 groups of 32 weights of 8 outputs a step once for up
to 16 rows of x and sums each row's products in fp32, K in order, by the same reduction whatever the
rows a program, so a row of an M-row call is the one-row call's row bit for bit (docs/kernels.md,
"Verify twins").
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:          # noqa: BLE001
    HAVE_TRITON = False

QK = 256
BLOCK_BYTES = 66
#: rows up to which a packed weight multiplies on its row-invariant kernel (the engine's ROWS_MAX)
ROWS_MAX = 32
# ggml's iq2xxs_grid, each entry's 8 magnitudes as 2-bit codes of (8, 25, 43), 16 bits an entry (hex,
# little-endian u16s): the packing gguf-py uses
_GRID_HEX = (
    b"00000200050008000a00110014002000220028002a0041004400500058006100"
    b"6400800082008a00a20001010401100115014001840198010002020222028202"
    b"010404041004210424044004420448046004810484049004a404000502050805"
    b"200546056905800591050906100640068406a406000805080808140828084108"
    b"440850085208880804094009020a140a01100410101021104010601084109010"
    b"951000110811201150115a118011241245120014081420142514491480141815"
    b"6215001616160118041810184018811800190519a019511a002002200a204420"
    b"6120802082202921482100220222012404241024402456240025412564259026"
    b"082820289428442a014004401040184021402440404048405640604081408440"
    b"9040004120416141804185410142104248425642684200440844204480449944"
    b"124524450046014804481048404845480049584961498249454a904a00500850"
    b"1150195020508050885004514251a4519152905492540a550156545600581158"
    b"195864584059085a046010604060686000615561186260620064056410651265"
    b"84654268008002800a8041808280048118814081118201840484108415844084"
    b"608400854685948509864086608602880489118a0490109024904090a1901691"
    b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
)


def grid_table() -> np.ndarray:
    """The IQ2_XXS grid: float32 [256, 8] of 8, 25 or 43."""
    raw = np.frombuffer(_GRID_HEX, dtype=np.uint8).reshape(-1, 2)
    nib = np.where(raw > 0x40, raw + 9, raw) & 0x0F
    b = (nib[:, 0] << 4) | nib[:, 1]                        # 512 bytes, 4 codes a byte, low bits first
    codes = (b[:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)[None, :]) & 3
    return np.array([8, 25, 43], dtype=np.float32)[codes.reshape(256, 8)]


def ksigns_table() -> np.ndarray:
    """ksigns_iq2xs: the 7 sign bits and the parity bit (bit 7) that makes the count even, uint8 [128]."""
    i = np.arange(128, dtype=np.uint8)
    par = np.array([bin(int(v)).count("1") & 1 for v in i], dtype=np.uint8)
    return i | (par << 7)


GRID = grid_table()
KSIGNS = ksigns_table()
#: the grid with an entry's 8 magnitudes as the bytes of one int64 (byte j = magnitude j)
GRID8 = (GRID.astype(np.int64) << (8 * np.arange(8, dtype=np.int64))[None, :]).sum(1)


def split_blocks(raw: np.ndarray, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """A packed [N, K] IQ2_XXS tensor (uint8, rows of K / 256 blocks) -> (d fp16 [N, nb], qs int32 [N, nb, 16])."""
    nb = K // QK
    b = np.ascontiguousarray(raw).reshape(N, nb, BLOCK_BYTES)
    d = np.ascontiguousarray(b[..., :2]).view(np.float16).reshape(N, nb)
    qs = np.ascontiguousarray(b[..., 2:]).view(np.int32).reshape(N, nb, 16)
    return d, qs


def dequant_iq2_xxs(raw: np.ndarray, N: int, K: int) -> torch.Tensor:
    """The twin: a packed [N, K] IQ2_XXS tensor -> fp32 [N, K] (torch, CPU)."""
    d, qs = split_blocks(raw, N, K)
    return dequant_blocks(torch.from_numpy(d), torch.from_numpy(qs))


def dequant_blocks(d: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
    """The twin's arithmetic on split blocks on any device: d fp16 [N, nb], qs int32 [N, nb, 16] -> fp32
    [N, nb 256]."""
    dev = d.device
    N = d.shape[0]
    d = d.float()                                                       # [N, nb]
    q = (qs.long() & 0xFFFFFFFF).view(N, -1, 8, 2)                      # [N, nb, 8 groups, (q0, q1)]
    q0, q1 = q[..., 0], q[..., 1]
    scale = (q1 >> 28) & 0xF
    db = d[..., None] * (0.5 + scale.to(torch.float32))                 # fp32: d * (0.5 + s)
    db = db * 0.25
    idx = (q0[..., None] >> torch.tensor([0, 8, 16, 24], device=dev)) & 0xFF   # [N, nb, 8, 4]
    g = torch.from_numpy(GRID).to(dev)[idx]                             # [N, nb, 8, 4, 8]
    si = (q1[..., None] >> torch.tensor([0, 7, 14, 21], device=dev)) & 0x7F
    ks = torch.from_numpy(KSIGNS.astype(np.int64)).to(dev)[si]          # [N, nb, 8, 4]
    bit = (ks[..., None] >> torch.arange(8, device=dev)) & 1
    sign = torch.where(bit == 1, torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
    w = (db[..., None, None] * g) * sign
    return w.reshape(N, -1)


NVFP4_QK = 64
NVFP4_BLOCK_BYTES = 36
#: E2M1 doubled (ggml's kvalues_mxfp4)
NVFP4_KVALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


def ue4m3_table() -> np.ndarray:
    """ggml_ue4m3_to_fp32 of every byte (float32 [256]), the 0.5 of the kvalues convention included."""
    x = np.arange(256)
    e, m = (x >> 3) & 0xF, (x & 7).astype(np.float32)
    raw = np.where(e == 0, m * np.float32(2.0 ** -9), (1.0 + m / 8.0) * np.exp2(e - 7.0)).astype(np.float32)
    return np.where((x == 0) | (x == 0x7F), np.float32(0.0), raw * np.float32(0.5)).astype(np.float32)


UE4M3 = ue4m3_table()


def dequant_nvfp4(raw: np.ndarray, N: int, K: int) -> torch.Tensor:
    """The twin: a packed [N, K] NVFP4 tensor -> fp32 [N, K] (torch, CPU), without the tensor's scale."""
    b = np.ascontiguousarray(raw).reshape(N, K // NVFP4_QK, NVFP4_BLOCK_BYTES)
    d = UE4M3[b[..., :4]]                                               # [N, nb, 4]
    qs = b[..., 4:].reshape(N, -1, 4, 8)
    codes = np.concatenate([qs & 0x0F, qs >> 4], -1)                    # [N, nb, 4, 16]
    return torch.from_numpy((NVFP4_KVALUES[codes] * d[..., None]).reshape(N, K))


def dequant_q8_0(raw: np.ndarray, N: int, K: int) -> torch.Tensor:
    """A packed [N, K] Q8_0 tensor -> fp32 [N, K]: w = d * q per block of 32."""
    b = np.ascontiguousarray(raw).reshape(N, K // 32, 34)
    d = torch.from_numpy(np.ascontiguousarray(b[..., :2]).view(np.float16).reshape(N, K // 32).astype(np.float32))
    q = torch.from_numpy(np.ascontiguousarray(b[..., 2:]).view(np.int8).astype(np.float32))
    return (d[..., None] * q).reshape(N, K)


if HAVE_TRITON:

    @triton.jit
    def _iq2_gemm(X, s_x, D, QS, G8, KS, Y, s_y, M, N, NB: tl.constexpr, BN: tl.constexpr, GB: tl.constexpr,
                  R: tl.constexpr, F32: tl.constexpr):
        # R rows of x a program, BN outputs, GB groups of 32 weights a step: each group's two words, its four
        # grid entries gathered as packed bytes (G8: one int64 an entry), the signs and the scale as in the
        # twin, decoded once for the R rows; each row's products summed in fp32 by the same reduction as in
        # the one-row build, K in order, so a row's bits do not depend on R or on the other rows
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        gg = tl.arange(0, GB)
        l4 = tl.arange(0, 4)
        j8 = tl.arange(0, 8)
        kk = tl.arange(0, GB * 32)
        rr = tl.arange(0, R)
        acc = tl.zeros([R, BN], dtype=tl.float32)
        for st in range(0, NB * 8 // GB):
            gidx = st * GB + gg
            b = gidx // 8
            wq = (cols[:, None] * NB + b[None, :]) * 16 + 2 * (gidx % 8)[None, :]
            d = tl.load(D + cols[:, None] * NB + b[None, :], mask=cm[:, None], other=0.0).to(tl.float32)   # [BN, GB]
            q0 = tl.load(QS + wq, mask=cm[:, None], other=0)
            q1 = tl.load(QS + wq + 1, mask=cm[:, None], other=0)
            db = (d * (0.5 + ((q1 >> 28) & 0xF).to(tl.float32))) * 0.25
            ge = tl.load(G8 + ((q0[:, :, None] >> (8 * l4[None, None, :])) & 0xFF))                       # [BN, GB, 4]
            g = ((ge[:, :, :, None] >> (8 * j8[None, None, None, :])) & 0xFF).to(tl.float32)              # [BN, GB, 4, 8]
            ks = tl.load(KS + ((q1[:, :, None] >> (7 * l4[None, None, :])) & 0x7F))
            neg = ((ks[:, :, :, None] >> j8[None, None, None, :]) & 1) == 1
            w = db[:, :, None, None] * g
            w = tl.reshape(tl.where(neg, -w, w), (BN, GB * 32))
            for r in tl.static_range(R):
                row = pm * R + r
                x = tl.load(X + row * s_x + st * GB * 32 + kk, mask=(kk >= 0) & (row < M), other=0.0).to(tl.float32)
                acc = tl.where(rr[:, None] == r, acc + tl.sum(w * x[None, :], 1)[None, :], acc)
        rows = pm * R + rr
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


if HAVE_TRITON:

    @triton.jit
    def _iq2_dequant(D, QS, G8, KS, Y, NB: tl.constexpr):
        # one block of 256 weights a program, as the twin computes them, rounded once to bf16
        r = tl.program_id(0)
        b = tl.program_id(1)
        g8 = tl.arange(0, 8)
        l4 = tl.arange(0, 4)
        j8 = tl.arange(0, 8)
        d = tl.load(D + r * NB + b).to(tl.float32)
        q0 = tl.load(QS + (r * NB + b) * 16 + 2 * g8)
        q1 = tl.load(QS + (r * NB + b) * 16 + 2 * g8 + 1)
        db = (d * (0.5 + ((q1 >> 28) & 0xF).to(tl.float32))) * 0.25                                  # [8]
        ge = tl.load(G8 + ((q0[:, None] >> (8 * l4[None, :])) & 0xFF))                               # [8, 4]
        g = ((ge[:, :, None] >> (8 * j8[None, None, :])) & 0xFF).to(tl.float32)                      # [8, 4, 8]
        ks = tl.load(KS + ((q1[:, None] >> (7 * l4[None, :])) & 0x7F))
        neg = ((ks[:, :, None] >> j8[None, None, :]) & 1) == 1
        w = db[:, None, None] * g
        w = tl.reshape(tl.where(neg, -w, w), (256,))
        tl.store(Y + (r * NB + b) * 256 + tl.arange(0, 256), w.to(tl.bfloat16))


def dequant_bf16(d: torch.Tensor, qs: torch.Tensor, g8: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    """Split IQ2_XXS blocks (d [R, nb], qs [R, nb, 16]) on the GPU -> bf16 [R, nb 256]: the twin's values
    rounded once, for a prompt chunk's cuBLAS."""
    R, nb = d.shape
    y = torch.empty(R, nb * QK, dtype=torch.bfloat16, device=d.device)
    _iq2_dequant[(R, nb)](d, qs, g8, ks, y, NB=nb, num_warps=1)
    return y


if HAVE_TRITON:

    @triton.jit
    def _iq2_bank(X, s_x, IDS, D, QS, G, KS, Y, N, S: tl.constexpr, NB: tl.constexpr, BN: tl.constexpr,
                  RELU2: tl.constexpr):
        # one (row, slot) pair p: y[p] = W[e] x[p // S] over expert e = IDS[p], K in order, fp32; RELU2:
        # bf16(relu(bf16(y))^2) as the expert's hidden (the up projection), else bf16-rounded fp32 (down)
        p = tl.program_id(0)
        pn = tl.program_id(1)
        row = p // S
        e = tl.load(IDS + p).to(tl.int64)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        k32 = tl.arange(0, 32)
        grp = k32 // 8
        el = k32 % 8
        base = e * N * NB
        acc = tl.zeros([BN], dtype=tl.float32)
        for b in range(0, NB):
            d = tl.load(D + base + cols * NB + b, mask=cm, other=0.0).to(tl.float32)
            for j in tl.static_range(8):
                q0 = tl.load(QS + (base + cols * NB + b) * 16 + 2 * j, mask=cm, other=0)
                q1 = tl.load(QS + (base + cols * NB + b) * 16 + 2 * j + 1, mask=cm, other=0)
                db = (d * (0.5 + ((q1 >> 28) & 0xF).to(tl.float32))) * 0.25
                idx = (q0[:, None] >> (8 * grp[None, :])) & 0xFF
                w = db[:, None] * tl.load(G + idx * 8 + el[None, :])
                ks = tl.load(KS + ((q1[:, None] >> (7 * grp[None, :])) & 0x7F))
                w = tl.where(((ks >> el[None, :]) & 1) == 1, -w, w)
                x = tl.load(X + row * s_x + b * 256 + j * 32 + k32).to(tl.float32)
                acc += tl.sum(w * x[None, :], 1)
        if RELU2:
            u = tl.maximum(acc.to(tl.bfloat16).to(tl.float32), 0.0)
            tl.store(Y + p * N + cols, (u * u).to(tl.bfloat16), mask=cm)
        else:
            tl.store(Y + p * N + cols, acc.to(tl.bfloat16).to(tl.float32), mask=cm)


class IQ2Bank:
    """The routed experts of one MoE layer as packed IQ2_XXS banks: up [E, I, X] and down [E, X, I] (GGUF's
    ffn_up_exps / ffn_down_exps), for `moe_relu2` with the same pair order and roundings as the BF16
    experts (engine/models/nemotron_h/kernels.py)."""

    def __init__(self, up_raw: np.ndarray, down_raw: np.ndarray, E: int, I: int, X: int, device="cuda"):
        for name, k in (("up", X), ("down", I)):
            if k % QK:
                raise ValueError(f"IQ2_XXS {name} experts need K a multiple of {QK}, not {k}")
        self.E, self.I, self.X = E, I, X
        du, qu = split_blocks(up_raw, E * I, X)
        dd, qd = split_blocks(down_raw, E * X, I)
        self.ud, self.uq = torch.from_numpy(du).to(device), torch.from_numpy(qu).to(device)
        self.dd, self.dq = torch.from_numpy(dd).to(device), torch.from_numpy(qd).to(device)
        self._g = torch.from_numpy(GRID.reshape(-1)).to(device)
        self._g8 = torch.from_numpy(GRID8).to(device)
        self._ks = torch.from_numpy(KSIGNS.astype(np.int32)).to(device)
        self.nbytes = (E * I * (X // QK) + E * X * (I // QK)) * BLOCK_BYTES

    def expert(self, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert e's (up [I, X], down [X, I]) decoded to bf16 on the device, for a prompt chunk's cuBLAS."""
        I, X = self.I, self.X
        if self.ud.device.type != "cuda":
            return (dequant_blocks(self.ud[e * I:(e + 1) * I], self.uq[e * I:(e + 1) * I]).to(torch.bfloat16),
                    dequant_blocks(self.dd[e * X:(e + 1) * X], self.dq[e * X:(e + 1) * X]).to(torch.bfloat16))
        return (dequant_bf16(self.ud[e * I:(e + 1) * I], self.uq[e * I:(e + 1) * I], self._g8, self._ks),
                dequant_bf16(self.dd[e * X:(e + 1) * X], self.dq[e * X:(e + 1) * X], self._g8, self._ks))

    def moe_relu2(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """x [M, X] bf16, ids int32 [M, S] -> yd [M S, X] fp32 (bf16 values, unweighted, pair order)."""
        M, S = ids.shape
        P = M * S
        flat = ids.reshape(-1).contiguous()
        a = torch.empty(P, self.I, dtype=torch.bfloat16, device=x.device)
        yd = torch.empty(P, self.X, dtype=torch.float32, device=x.device)
        _iq2_bank[(P, triton.cdiv(self.I, 32))](x, x.stride(0), flat, self.ud, self.uq, self._g, self._ks, a, self.I, S=S,
                                                NB=self.X // QK, BN=32, RELU2=True, num_warps=2)
        _iq2_bank[(P, triton.cdiv(self.X, 32))](a, a.stride(0), flat,
                                                self.dd, self.dq, self._g, self._ks, yd, self.X, S=1, NB=self.I // QK, BN=32,
                                                RELU2=False, num_warps=2)
        return yd


class IQ2Linear:
    """A packed IQ2_XXS weight [N, K] on the device: d fp16 [N, K / 256], qs int32 [N, K / 256, 16]."""

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        if K % QK:
            raise ValueError(f"IQ2_XXS needs K a multiple of {QK}, not {K}")
        d, qs = split_blocks(raw, N, K)
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(d).to(device)
        self.qs = torch.from_numpy(qs).to(device)
        self.nbytes = N * (K // QK) * BLOCK_BYTES
        self._g8 = torch.from_numpy(GRID8).to(device)
        self._ks = torch.from_numpy(KSIGNS.astype(np.int32)).to(device)

    def dequant(self) -> torch.Tensor:
        raw = np.concatenate([self.d.cpu().numpy().view(np.uint8).reshape(self.N, -1, 2),
                              self.qs.cpu().numpy().view(np.uint8).reshape(self.N, -1, 64)], -1)
        return dequant_iq2_xxs(raw.reshape(-1), self.N, self.K)

    def matmul(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
        x2 = x.reshape(-1, self.K).contiguous()
        M = x2.shape[0]
        if M > ROWS_MAX:
            # a prompt chunk: the weight decoded to bf16 for one cuBLAS call (the rows' bits are not kept)
            w = dequant_bf16(self.d, self.qs, self._g8, self._ks) if x2.is_cuda else \
                dequant_blocks(self.d, self.qs).to(torch.bfloat16)
            y = torch.nn.functional.linear(x2, w)
            y = y.float() if f32 else y
            return (y if out is None else out.copy_(y)).view(*x.shape[:-1], self.N)
        if out is None:
            out = torch.empty(M, self.N, dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
        # 8 outputs and 4 groups a step, one warp; the rows a program by the call's row count (the builds give
        # a row the same bits): a 5,376 x 4,096 weight takes 0.10 ms for one row, 0.36 for 8, 0.47 for 16 on GB10
        R = 1 if M == 1 else 4 if M <= 4 else 8 if M <= 8 else 16
        _iq2_gemm[(triton.cdiv(M, R), triton.cdiv(self.N, 8))](x2, x2.stride(0), self.d, self.qs, self._g8, self._ks, out,
                                                               out.stride(0), M, self.N, NB=self.K // QK, BN=8, GB=4, R=R,
                                                               F32=f32, num_warps=1)
        return out.view(*x.shape[:-1], self.N)


if HAVE_TRITON:

    @triton.jit
    def _nvfp4_gemm(X, s_x, QB, LUT, KV, Y, s_y, M, N, scale, NB: tl.constexpr, BN: tl.constexpr, F32: tl.constexpr):
        # QB: the blocks as int32 words [N, NB, 9] (4 scale bytes, then 32 code bytes); a [BN, 64] slice of
        # weights decoded a step, multiplied in fp32 (IEEE) into 16-row tiles of x, K in order; the
        # tensor's scale on the fp32 sum, then one rounding
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        k64 = tl.arange(0, 64)
        sub = k64 // 16
        byte = 4 + sub * 8 + (k64 % 8)                         # the code byte of weight k within the block
        hi = (k64 % 16) >= 8
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for b in range(0, NB):
            base = (cols * NB + b) * 9                                                     # [BN] words
            dw = tl.load(QB + base, mask=cm, other=0)                                      # the 4 scale bytes
            dsc = (dw[:, None] >> (8 * sub[None, :])) & 0xFF                               # [BN, 64]
            qw = tl.load(QB + base[:, None] + byte[None, :] // 4, mask=cm[:, None], other=0)
            q = (qw >> (8 * (byte[None, :] % 4))) & 0xFF
            code = tl.where(hi[None, :], q >> 4, q & 0xF)
            w = tl.load(KV + code) * tl.load(LUT + dsc)                                    # [BN, 64] fp32, exact
            k = b * 64 + k64
            x = tl.load(X + rows[:, None] * s_x + k[None, :], mask=rows[:, None] < M, other=0.0).to(tl.float32)
            acc += tl.dot(x, tl.trans(w), input_precision="ieee")
        acc = acc * scale
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


def repack_nvfp4(raw: torch.Tensor, N: int, K: int, scale: float):
    """A GGUF NVFP4 tensor (uint8, any device) as an `NVFP4Block` with the same values: codes [N, K / 2]
    with weight 2 i in byte i's low nibble, E4M3 scales [N, K / 16], `scale` as the per-tensor scale."""
    from tools.nvfp4_linear import NVFP4Block
    b = raw.reshape(N, K // NVFP4_QK, NVFP4_BLOCK_BYTES)
    d = b[..., :4]
    if bool((d == 0xFF).any()):
        raise ValueError("an NVFP4 scale byte 0xFF (480 in ggml) has no E4M3 value")
    d = torch.where((d & 0x7F) == 0x7F, torch.zeros_like(d), d & 0x7F)
    qs = b[..., 4:].reshape(N, -1, 4, 8)
    codes = torch.cat([qs & 0x0F, qs >> 4], -1).reshape(N, K // 2, 2)
    packed = (codes[..., 0] | (codes[..., 1] << 4)).contiguous()
    return NVFP4Block(packed, d.reshape(N, K // 16).contiguous().view(torch.float8_e4m3fn), scale)


class NVFP4Linear:
    """A packed NVFP4 weight [N, K] on the device and its per-tensor scale (1.0 when the file has none):
    on a GPU with K a multiple of 128 the checkpoints' layout (`repack_nvfp4`, `blk`), else the blocks as
    stored (int32 words [N, K / 64, 9], the Triton GEMM)."""

    def __init__(self, raw: np.ndarray, N: int, K: int, scale: float = 1.0, device="cuda"):
        if K % NVFP4_QK:
            raise ValueError(f"NVFP4 needs K a multiple of {NVFP4_QK}, not {K}")
        self.N, self.K, self.shape, self.scale = N, K, (N, K), float(scale)
        self.nbytes = N * (K // NVFP4_QK) * NVFP4_BLOCK_BYTES
        self.blk = self.qb = None
        if torch.device(device).type == "cuda" and K % 128 == 0:          # the checkpoints' layout's K step
            self.blk = repack_nvfp4(torch.from_numpy(np.ascontiguousarray(raw)).to(device), N, K, self.scale)
            return
        self.qb = torch.from_numpy(np.ascontiguousarray(raw).view(np.int32).reshape(N, K // NVFP4_QK, 9).copy()).to(device)
        self._lut = torch.from_numpy(UE4M3).to(device)
        self._kv = torch.from_numpy(NVFP4_KVALUES).to(device)

    def dequant(self) -> torch.Tensor:
        """The weight without the scale, fp32 [N, K] on the CPU."""
        if self.blk is not None:
            from tools.nvfp4_linear import FP4_GRID
            c = torch.stack([self.blk.w & 0x0F, self.blk.w >> 4], -1).reshape(self.N, self.K).long().cpu()
            v = FP4_GRID[c & 7] * torch.where(c >= 8, -1.0, 1.0)
            return v * self.blk.s.float().cpu().repeat_interleave(16, 1)
        return dequant_nvfp4(self.qb.cpu().numpy().view(np.uint8).reshape(-1), self.N, self.K)

    def matmul(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
        x2 = x.reshape(-1, self.K).contiguous()
        M = x2.shape[0]
        if self.blk is not None:
            if f32:
                raise ValueError("NVFP4Linear writes bf16 rows on the GPU")
            from tools.nvfp4_linear import nvfp4_matmul
            from tools.nvfp4_skinny import SKINNY_MAX, nvfp4_matmul_skinny
            y = nvfp4_matmul_skinny(x2, self.blk, out=out) if M <= SKINNY_MAX else nvfp4_matmul(x2, self.blk, out=out)
            return y.view(*x.shape[:-1], self.N)
        if out is None:
            out = torch.empty(M, self.N, dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
        BN = 32
        _nvfp4_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, BN))](x2, x2.stride(0), self.qb, self._lut, self._kv, out,
                                                                   out.stride(0), M, self.N, self.scale,
                                                                   NB=self.K // NVFP4_QK, BN=BN, F32=f32, num_warps=4)
        return out.view(*x.shape[:-1], self.N)


if HAVE_TRITON:

    @triton.jit
    def _q8_gemm(X, s_x, D, Q, Y, s_y, M, N, K, BN: tl.constexpr, SB: tl.constexpr, F32: tl.constexpr):
        # 16-row tiles of x, BN outputs, SB blocks of 32 a step: each block one bf16 tensor-core product
        # (the int8 codes are exact in bf16, x is bf16: every product exact, summed in fp32) times the
        # block's d, K in order; a row's sum does not depend on the other rows of its tile
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        k32 = tl.arange(0, 32)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, K // 32, SB):
            for j in tl.static_range(SB):
                k = (kb + j) * 32 + k32
                q = tl.load(Q + cols[:, None] * K + k[None, :], mask=cm[:, None], other=0).to(tl.bfloat16)
                d = tl.load(D + cols * (K // 32) + kb + j, mask=cm, other=0.0).to(tl.float32)
                x = tl.load(X + rows[:, None] * s_x + k[None, :], mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(q)) * d[None, :]
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


class Q8Linear:
    """A packed Q8_0 weight [N, K] on the device: d fp16 [N, K / 32], q int8 [N, K]. Rows up to ROWS_MAX on
    the tile kernel (row-invariant; Bolt's head: 3.8 ms for 1 to 32 rows on GB10, the bf16 head 4.7), a
    prompt chunk on cuBLAS over the twin in bf16."""

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        b = np.ascontiguousarray(raw).reshape(N, K // 32, 34)
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(b[..., :2]).view(np.float16).reshape(N, K // 32)).to(device)
        self.q = torch.from_numpy(np.ascontiguousarray(b[..., 2:]).view(np.int8).reshape(N, K)).to(device)
        self.nbytes = N * (K // 32) * 34

    def dequant(self) -> torch.Tensor:
        """The twin (fp32 [N, K]) on the weight's device."""
        return (self.d.float().repeat_interleave(32, 1) * self.q.float())

    def matmul(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
        x2 = x.reshape(-1, self.K).contiguous()
        M = x2.shape[0]
        if M > ROWS_MAX:
            y = torch.nn.functional.linear(x2, self.dequant().to(torch.bfloat16))
            y = y.float() if f32 else y
            return (y if out is None else out.copy_(y)).view(*x.shape[:-1], self.N)
        if out is None:
            out = torch.empty(M, self.N, dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        _q8_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.d, self.q, out, out.stride(0), M,
                                                                self.N, self.K, BN=32, SB=8, F32=f32, num_warps=2,
                                                                num_stages=3)
        return out.view(*x.shape[:-1], self.N)


def matmul_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The twin of `IQ2Linear.matmul`: x (as fp32) @ w^T (the dequantised fp32 weight), fp32."""
    return x.float() @ w.float().T
