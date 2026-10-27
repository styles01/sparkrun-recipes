"""A tree verify must not read the device to learn the tree's depth.

`fused_tree_step` checked `int(depths.max())` -- a device-to-host synchronisation -- once per
linear-attention layer, 48 times a tree verify. With `QWEN38_TREE_HOST_DEPTH` the engine hands it the
depth from the tree's host-side copy. The kernel is Triton, so the test replaces it with a stand-in
that records what it was given and returns correctly shaped zeros, on the random four-layer model.
Failing first: before the change the engine passed no depth at all.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402

import torch  # noqa: E402

import engine.model as M  # noqa: E402
from tools import gdn_tree_kernels as GT  # noqa: E402


def test_the_tree_depth_comes_from_the_host():
    seen = []

    def stand_in(q, k, v, g, beta, depths, state, *, bv=16, max_depth=None):
        seen.append(max_depth)
        B, Tn, H, Dv = v.shape
        z = torch.zeros(B, Tn, H, Dv, dtype=torch.float32)
        return z.to(q.dtype), z, torch.zeros(B, Tn, H)

    orig, fused = GT.fused_tree_step, M.FUSED["gdntree"]
    GT.fused_tree_step, M.FUSED["gdntree"], M.TREE_HOST_DEPTH = stand_in, True, True
    try:
        eng, pos = T.build(seed=5)
        tree = T.BRANCHY
        with torch.no_grad():
            eng.forward_tree(torch.tensor(tree.tokens), tree.parents, start=pos)
    finally:
        GT.fused_tree_step, M.FUSED["gdntree"], M.TREE_HOST_DEPTH = orig, fused, False
    assert len(seen) == 3, seen                          # the model's three linear layers
    assert all(isinstance(d, int) for d in seen), seen
    assert set(seen) == {max(tree.depths())}, seen
    return f"every linear layer got max_depth={seen[0]} from the host"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
