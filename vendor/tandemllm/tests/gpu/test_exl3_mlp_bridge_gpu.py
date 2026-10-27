"""On the board: the prefill MLP bridge (`exl3_mlp_slab`, QWEN38_EXL3_MLP_BRIDGE) is bit-identical to
the path it replaces -- gate|up through the slab GEMM and rot_out to bf16, ATen's silu and mul, the
down projection's rot_in and slab GEMM -- at prefill row counts, with and without the layer-major
slab cache, for the K mixes the 4.5 recipe uses. Run on the box: python tests/gpu/test_exl3_mlp_bridge_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import exl3_linear as X  # noqa: E402

D, FF = 5120, 17408


def _mlp_ref(x, gu, wd):
    y = X.exl3_matmul(x, gu)
    gate, up = y.split(gu.sizes, dim=-1)
    return X.exl3_matmul(F.silu(gate) * up, wd)


def test_bridge_bit_identical():
    done = []
    for kg, kd in ((4, 5), (5, 4), (6, 6)):
        g = X.random_block(FF, D, kg, seed=10 + kg)
        u = X.random_block(FF, D, kg, seed=20 + kg)
        gu = X.EXL3Block.group([g, u], ["gate", "up"])
        wd = X.random_block(D, FF, kd, seed=30 + kd)
        for M in (129, 300, 1024, 2048):
            gen = torch.Generator(device="cpu").manual_seed(M)
            x = (torch.randn(M, D, generator=gen) * 0.7).to(torch.bfloat16).cuda()
            ref = _mlp_ref(x, gu, wd)
            got = X.exl3_mlp_slab(x, gu, wd)
            assert got is not None
            assert torch.equal(ref, got), f"K{kg}/{kd} rows {M}: {(ref != got).sum().item()} differ"
            X.slab_cache_on()
            try:
                got2 = X.exl3_mlp_slab(x, gu, wd)
                got3 = X.exl3_mlp_slab(x, gu, wd)          # from the cached slabs
            finally:
                X.slab_cache_off()
            assert torch.equal(ref, got2) and torch.equal(ref, got3), f"cached K{kg}/{kd} {M}"
            done.append((kg, kd, M))
    return f"{len(done)} cases (K mixes x rows 129..2048, cache off and on): identical"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:32s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
