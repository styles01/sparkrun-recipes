"""The greedy lattice walk with one device read instead of one per slot.

`DFlash2Module.walk_host` must pick exactly the tokens `walk` picks -- same argmaxes, same chain --
and hand back the same candidate table `propose_tree` used to read with its own `.tolist()`.
Random lattices, ties included (integer-valued scores make ties common, and `argmax` must break
them the same way on both paths because both call it on the same tensor).
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.dflash2 import DFlash2Module  # noqa: E402


def test_host_walk_equals_walk():
    g = torch.Generator().manual_seed(0)
    n = 0
    for L, k in ((7, 16), (15, 16), (2, 4), (1, 16)):
        for trial in range(40):
            cand = torch.randint(0, 248320, (L, k), generator=g)
            if trial % 2:
                scores = torch.randint(-3, 3, (L, k, k), generator=g).float()   # many ties
            else:
                scores = torch.randn(L, k, k, generator=g)
            ref = [int(x) for x in DFlash2Module.walk(cand, scores)]
            got, table = DFlash2Module.walk_host(cand, scores)
            assert got == ref, (L, k, trial, got, ref)
            assert table == cand.tolist()
            n += 1
    return f"{n} random lattices, ties included: same tokens, same candidate table"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
