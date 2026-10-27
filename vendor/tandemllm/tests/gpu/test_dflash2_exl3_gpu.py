"""On the board (ENG-215/216): the EXL3 drafter's fused launches are the member launches bit for bit.

`engine/drafters/dflash2.py::exl3_projections` groups q|k|v and gate|up (`EXL3Block.group`) and
replaces each member with a view of its group; `forward_block` runs the group once and splits it,
`context_kv` and the packed draft use the member views alone. Both must give the same bits, at the
drafter's row counts (a 16-row block, 1..4 sync rows). Run on the box: python tests/gpu/test_dflash2_exl3_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import exl3_linear as X  # noqa: E402

H = 5120


def test_group_equals_members():
    cases = 0
    for kb, shapes in ((3, (4096, 1024, 1024)), (2, (17408, 17408)), (3, (17408, 17408))):
        blocks = [X.random_block(n, H, kb, seed=7 * j + kb) for j, n in enumerate(shapes)]
        grp = X.EXL3Block.group(blocks, [f"m{j}" for j in range(len(shapes))])
        for M in (1, 3, 16, 32):
            x = (torch.randn(M, H, device="cuda") * 0.5).to(torch.bfloat16)
            parts = X.exl3_matmul(x, grp).split(grp.sizes, dim=-1)
            for j in range(len(shapes)):
                alone = X.exl3_matmul(x, grp.member(j))
                assert torch.equal(parts[j], alone), (kb, shapes, M, j)
                cases += 1
    return f"{cases} member launches equal their slice of the group launch"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:32s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
