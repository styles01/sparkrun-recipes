"""IQ2_XXS and NVFP4: the dequant twins (tools/iq2_linear.py) against gguf-py's `IQ2_XXS.dequantize_blocks`
and `NVFP4.dequantize_blocks` bit for bit on random blocks (every bit pattern of a block is a valid
block), Q8_0 likewise; the on-the-fly GEMMs against the twins' fp32 matmul, and their rows invariant to
the row count up to 33 (bit for bit). The GEMMs run on the GPU or under TRITON_INTERPRET=1; the twins
anywhere. NVFP4 needs a gguf-py with the type (llama.cpp after GGML_TYPE_NVFP4 = 40)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import iq2_linear as Q  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
KERNELS = Q.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))


def random_iq2(N: int, K: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nb = N * (K // 256)
    b = rng.integers(0, 256, size=(nb, 66), dtype=np.uint8)
    d = (rng.standard_normal(nb) * 0.02).astype(np.float16)
    b[:, :2] = d.view(np.uint8).reshape(nb, 2)
    return b.reshape(-1)


def random_nvfp4(N: int, K: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=N * (K // 64) * 36, dtype=np.uint8)


def test_tables():
    assert Q.GRID.shape == (256, 8) and set(np.unique(Q.GRID)) == {8.0, 25.0, 43.0}
    assert Q.KSIGNS[0] == 0 and Q.KSIGNS[1] == 0x81 and Q.KSIGNS[3] == 0x03 and Q.KSIGNS[127] == 0xFF
    unpacked = (Q.GRID8[:, None] >> (8 * np.arange(8))[None, :]) & 0xFF
    assert np.array_equal(unpacked.astype(np.float32), Q.GRID)


def test_twin_equals_gguf_py():
    try:
        from gguf.constants import GGMLQuantizationType
        from gguf.quants import dequantize
    except Exception:  # noqa: BLE001
        return "skipped: gguf-py missing"
    for N, K in ((3, 256), (5, 1024)):
        raw = random_iq2(N, K, seed=N)
        want = dequantize(raw.reshape(N, -1), GGMLQuantizationType.IQ2_XXS).astype(np.float32)
        got = Q.dequant_iq2_xxs(raw, N, K).numpy()
        assert want.shape == got.shape and np.array_equal(want.view(np.uint32), got.view(np.uint32)), \
            (N, K, np.abs(want - got).max())
        rng = np.random.default_rng(N)
        nb = N * K // 32
        b8 = rng.integers(0, 256, size=(nb, 34), dtype=np.uint8)
        b8[:, :2] = (rng.standard_normal(nb) * 0.02).astype(np.float16).view(np.uint8).reshape(nb, 2)
        w8 = dequantize(b8.reshape(N, -1), GGMLQuantizationType.Q8_0).astype(np.float32)
        assert np.array_equal(w8.view(np.uint32), Q.dequant_q8_0(b8.reshape(-1), N, K).numpy().view(np.uint32))
    return "IQ2_XXS and Q8_0 bit for bit"


def test_ue4m3_table():
    assert Q.UE4M3[0] == 0.0 and Q.UE4M3[0x7F] == 0.0
    assert Q.UE4M3[0x38] == 0.5                              # e 7, m 0: 1.0, halved
    assert Q.UE4M3[0x01] == 2.0 ** -10                       # subnormal: 1 x 2^-9, halved
    assert Q.UE4M3[0x7E] == (1 + 6 / 8) * 2.0 ** 8 * 0.5     # the largest finite
    assert Q.UE4M3[0x80] == 0.0 and Q.UE4M3[0xB8] == 0.5     # bit 7 is not read


def test_nvfp4_twin_equals_gguf_py():
    try:
        from gguf.constants import GGMLQuantizationType
        from gguf.quants import dequantize
        qt = GGMLQuantizationType.NVFP4
    except Exception:  # noqa: BLE001
        return "skipped: gguf-py without NVFP4"
    for N, K in ((3, 64), (5, 1024)):
        raw = random_nvfp4(N, K, seed=N)
        want = dequantize(raw.reshape(N, -1), qt).astype(np.float32)
        got = Q.dequant_nvfp4(raw, N, K).numpy()
        assert want.shape == got.shape and np.array_equal(want.view(np.uint32), got.view(np.uint32)), \
            (N, K, np.abs(want - got).max())
    return "NVFP4 bit for bit"


def test_nvfp4_repack_keeps_every_value():
    from tools.nvfp4_linear import FP4_GRID
    N, K = 6, 256
    raw = random_nvfp4(N, K, seed=9)
    raw.reshape(-1, 36)[:, :4][raw.reshape(-1, 36)[:, :4] == 0xFF] = 0x80
    raw.reshape(-1, 36)[0, 0] = 0x7F                                   # ggml: zero
    blk = Q.repack_nvfp4(torch.from_numpy(raw), N, K, 0.5)
    c = torch.stack([blk.w & 0x0F, blk.w >> 4], -1).reshape(N, K).long()
    v = FP4_GRID[c & 7] * torch.where(c >= 8, -1.0, 1.0) * blk.s.float().repeat_interleave(16, 1)
    assert torch.equal(v, Q.dequant_nvfp4(raw, N, K)) and blk.s2 == 0.5
    bad = raw.copy()
    bad[0] = 0xFF
    try:
        Q.repack_nvfp4(torch.from_numpy(bad), N, K, 1.0)
    except ValueError as e:
        assert "0xFF" in str(e)
    else:
        raise AssertionError("a 0xFF scale byte was repacked")


def test_nvfp4_layout():
    # one block: sub-block 1's scale byte 0x38 (0.5), byte 3 of its codes = 0x7A: weight 16 + 3 is
    # code 0xA (-2) and weight 16 + 3 + 8 code 0x7 (12)
    b = np.zeros(36, dtype=np.uint8)
    b[1] = 0x38
    b[4 + 8 + 3] = 0x7A
    w = Q.dequant_nvfp4(b, 1, 64).numpy()[0]
    assert w[19] == -1.0 and w[27] == 6.0 and np.count_nonzero(w) == 2


def test_nvfp4_gemm_near_the_twin_and_row_invariant():
    if not KERNELS:
        return "skipped: Triton on a GPU, or TRITON_INTERPRET=1"
    N, K, scale = 80, 256, 1.5e-4
    raw = random_nvfp4(N, K, seed=5)
    raw.reshape(-1, 36)[:, :4][raw.reshape(-1, 36)[:, :4] == 0xFF] = 0x7E      # 0xFF has no E4M3 value
    lin = Q.NVFP4Linear(raw, N, K, scale=scale, device=DEV)
    w = Q.dequant_nvfp4(raw, N, K)
    assert torch.equal(lin.dequant(), w)                       # the repack (GPU) keeps every value
    g = torch.Generator().manual_seed(2)
    x = torch.randn(33, K, generator=g).to(torch.bfloat16).to(DEV)
    want = Q.matmul_reference(x.cpu(), w) * scale
    if DEV == "cuda":
        # the skinny kernel (bf16 rows): near the twin, every row of 1..32 its lone call's bits
        y = lin.matmul(x[:32]).float()
        want = want[:32]
        err = ((y.cpu() - want).abs().max() / want.abs().max()).item()
        assert err < 1e-2, err
        for M in (1, 2, 7, 16, 17, 32):
            assert torch.equal(lin.matmul(x[:M]).float(), y[:M]), M
        return f"rel err {err:.1e} (bf16 rows); rows 1..32 bit for bit"
    y = lin.matmul(x, f32=True)
    err = ((y.cpu() - want).abs().max() / want.abs().max()).item()
    assert err < 1e-5, err
    for M in (1, 2, 16, 17, 33):
        assert torch.equal(lin.matmul(x[:M], f32=True), y[:M]), M
    return f"rel err {err:.1e}; rows 1..33 bit for bit"


def test_gemm_near_the_twin_and_row_invariant():
    if not KERNELS:
        return "skipped: Triton on a GPU, or TRITON_INTERPRET=1"
    N, K = 96, 512
    raw = random_iq2(N, K, seed=7)
    lin = Q.IQ2Linear(raw, N, K, device=DEV)
    w = Q.dequant_iq2_xxs(raw, N, K)
    assert torch.equal(lin.dequant(), w)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, K, generator=g).to(torch.bfloat16).to(DEV)
    y = lin.matmul(x[:32], f32=True)
    want = Q.matmul_reference(x.cpu(), w)
    err = ((y.cpu() - want[:32]).abs().max() / want.abs().max()).item()
    assert err < 1e-5, err
    for M in (1, 2, 16, 17, 32):
        y1 = lin.matmul(x[:M], f32=True)
        assert torch.equal(y1, y[:M]), M
    # a prompt chunk (past 32 rows): the weight decoded to bf16, one cuBLAS call
    yc = lin.matmul(x, f32=True)
    errc = ((yc.cpu() - want).abs().max() / want.abs().max()).item()
    assert errc < 1e-2, errc
    return f"rel err {err:.1e}; rows 1..32 bit for bit; a 40-row chunk rel {errc:.1e}"


def test_q8_gemm_near_the_twin_and_row_invariant():
    if not KERNELS:
        return "skipped: Triton on a GPU, or TRITON_INTERPRET=1"
    N, K = 72, 512
    rng = np.random.default_rng(3)
    nb = N * K // 32
    b = rng.integers(-127, 128, size=(nb, 34)).astype(np.int8).view(np.uint8)
    b[:, :2] = (rng.random(nb) * 0.004 + 0.001).astype(np.float16).view(np.uint8).reshape(nb, 2)
    raw = b.reshape(-1)
    lin = Q.Q8Linear(raw, N, K, device=DEV)
    w = Q.dequant_q8_0(raw, N, K)
    assert torch.equal(lin.dequant().cpu(), w)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(40, K, generator=g).to(torch.bfloat16).to(DEV)
    want = Q.matmul_reference(x.cpu(), w)
    y = lin.matmul(x[:32], f32=True)
    err = ((y.cpu() - want[:32]).abs().max() / want.abs().max()).item()
    assert err < 1e-5, err
    for M in (1, 2, 15, 16, 17, 32):
        assert torch.equal(lin.matmul(x[:M], f32=True), y[:M]), M
    yc = lin.matmul(x, f32=True)
    errc = ((yc.cpu() - want).abs().max() / want.abs().max()).item()
    assert errc < 1e-2, errc
    return f"rel err {err:.1e}; rows 1..32 bit for bit; a 40-row chunk rel {errc:.1e}"


def test_expert_bank_near_the_twin_and_pair_invariant():
    if not KERNELS:
        return "skipped: Triton on a GPU, or TRITON_INTERPRET=1"
    E, I, X, M, S = 4, 256, 512, 5, 2
    up, down = random_iq2(E * I, X, seed=11), random_iq2(E * X, I, seed=12)
    bank = Q.IQ2Bank(up, down, E, I, X, device=DEV)
    wu = Q.dequant_iq2_xxs(up, E * I, X).view(E, I, X)
    wd = Q.dequant_iq2_xxs(down, E * X, I).view(E, X, I)
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(M, X, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:S] for _ in range(M)]).to(torch.int32)
    yd = bank.moe_relu2(x.to(DEV), ids.to(DEV)).cpu()
    for p, e in enumerate(ids.reshape(-1).tolist()):
        u = (wu[e] @ x[p // S].float()).to(torch.bfloat16).float()
        a = torch.relu(u).square().to(torch.bfloat16).float()
        want = (wd[e] @ a).to(torch.bfloat16).float()
        err = ((yd[p] - want).abs().max() / (want.abs().max() + 1e-6)).item()
        assert err < 2e-2, (p, err)                    # one bf16 step where the sum orders differ
    for r in range(M):
        y1 = bank.moe_relu2(x[r:r + 1].to(DEV), ids[r:r + 1].to(DEV)).cpu()
        assert torch.equal(y1, yd[r * S:(r + 1) * S]), r
    for e in range(E):                                    # a prompt chunk's expert: the twin, in bf16
        up_e, down_e = bank.expert(e)
        assert torch.equal(up_e.cpu(), wu[e].to(torch.bfloat16)) and torch.equal(down_e.cpu(), wd[e].to(torch.bfloat16))
    return f"{M * S} pairs; each row's pairs = its lone call bit for bit"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
