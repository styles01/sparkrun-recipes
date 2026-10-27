"""On the board (ENG-213/214, phase 2): the EXL3 row grid and the fused-shape tile table are bit-identical
to the chunked launches and the default tiles, at every row count 1..64, every K the builds use.
Run on the box under the lock: python tests/gpu/test_exl3_rowgrid_gpu.py
"""

from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import exl3_linear as X  # noqa: E402

ROWS = (1, 2, 7, 16, 17, 24, 32, 33, 40, 48, 63, 64, 65, 100, 128)
SHAPES = ((34816, 5120), (5120, 17408), (16384, 5120), (14336, 5120), (5120, 6144))


def _run(x, w, rowgrid: bool, tile: dict | None):
    X.ROWGRID = rowgrid
    key = (w.tile_n, w.K, w.kb)
    X._CONFIG.pop(key, None)
    if tile is not None:
        X.set_config(*key, tile)
    return X.exl3_matmul(x, w).clone()


def test_rowgrid_and_tiles_bit_identical():
    tab = {k: v for k, v in json.load(open(os.path.join(ROOT, "ops", "exl3-tiles-fused.json"))).items()
           if not k.startswith("_")}
    keep = dict(X._CONFIG), X.ROWGRID
    try:
        for N, K in SHAPES:
            for kb in (2, 3, 4, 5, 6):
                w = X.random_block(N, K, kb, seed=kb)
                x = (torch.randn(max(ROWS), K, device="cuda") * 0.5).to(torch.bfloat16)
                tile = tab.get(f"{N}x{K}x{kb}")
                one = _run(x[:1].contiguous(), w, False, None)
                for M in ROWS:
                    xm = x[:M].contiguous()
                    ref = _run(xm, w, False, None)
                    got = _run(xm, w, True, tile)
                    assert torch.equal(ref, got), (N, K, kb, M)
                    assert torch.equal(got[:1], one), (N, K, kb, M, "row 0")
    finally:
        X._CONFIG.clear()
        X._CONFIG.update(keep[0])
        X.ROWGRID = keep[1]


if __name__ == "__main__":
    test_rowgrid_and_tiles_bit_identical()
    print("PASS")
