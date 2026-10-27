"""On the board: `exl3_down_silu` (silu(gate) * up fused into the down projection's input rotation)
is bit-identical to the torch path `F.silu(gate) * up` -> `exl3_matmul`, at every row count the
verify, the row grid and the prefill slab take, and on the edge values of silu (large |x|, zeros,
subnormal-range products). Run on the box under the lock: python tests/gpu/test_exl3_silu_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import exl3_linear as X  # noqa: E402

ROWS = (1, 2, 7, 16, 17, 24, 32, 33, 64, 100, 128, 129, 300, 1024)
KIN = 17408


def _y(M: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    y = torch.randn(M, 2 * KIN, generator=g) * 3.0
    if M > 4:
        y[0, :64] = torch.tensor([0.0, -0.0, 88.0, -88.0, 1e-30, -1e-30, 20.0, -20.0] * 8)
        y[1, ::7] *= 30.0                                      # far tails of silu
        y[2, KIN::5] = 1e-20                                   # tiny products
    return y.to(torch.bfloat16).cuda()


def test_rotation_bits():
    w = X.random_block(5120, KIN, 4, seed=4)
    for M in (1, 16, 300):
        y = _y(M, M)
        gate, up = y.split([KIN, KIN], dim=-1)
        ref = X.rotate_in((F.silu(gate) * up).contiguous(), w)
        got = torch.empty_like(ref)
        X._module().exl3_rot_in_silu(y, w.suh, got)
        assert torch.equal(ref, got), f"rows {M}: {(ref != got).sum().item()} fp16 values differ"
    return "rotated input identical at 1, 16, 300 rows"


def test_down_silu_bit_identical():
    for kb in (4, 5, 6):
        w = X.random_block(5120, KIN, kb, seed=kb)
        for M in ROWS:
            y = _y(M, 100 + M)
            gate, up = y.split([KIN, KIN], dim=-1)
            ref = X.exl3_matmul(F.silu(gate) * up, w).clone()
            got = X.exl3_down_silu(y, w)
            assert torch.equal(ref, got), f"K{kb} rows {M}: {(ref != got).sum().item()} differ"
    return f"K4/K5/K6 x rows {ROWS}: identical"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:40s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
