"""/64: FP8Group -- a fused projection group of plain FP8 weights is the members, byte for byte.

`Weights.fuse_nvfp4_groups` (QWEN38_FUSE_PROJ=1) now also fuses a group whose members are all plain
`FP8Block`s. The fused weight is the members' codes and 128x128 scale tables concatenated along N;
every member becomes a view of its slice. On the CPU this checks the layout (the views, the scale
rows staying aligned), that the fused product's slices equal each member's own product, that the
members still work alone afterwards, and that a mixed NVFP4/FP8 group is left unfused. The board
check of the same identity on the Triton tile is `tools/fp8_probe.py --group`.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.loader import PROJ_GROUPS, Weights  # noqa: E402
from engine.model import matmul_group  # noqa: E402
from tools.fp8_linear import BLOCK, FP8Block, FP8Group, fp8_matmul  # noqa: E402


def rand_block(N, K, g):
    codes = (torch.randn(N, K, generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale = (torch.rand(N // BLOCK, K // BLOCK, generator=g) * 1e-3 + 1e-4).to(torch.bfloat16)
    return FP8Block(codes, scale)


def test_group_equals_members():
    g = torch.Generator().manual_seed(0)
    K = 256
    blocks = [rand_block(n, K, g) for n in (384, 128, 128)]
    x = (torch.randn(5, K, generator=g) * 0.5).to(torch.bfloat16)
    sep = [fp8_matmul(x, b) for b in blocks]
    before = [(b.w.clone(), b.s.clone()) for b in blocks]
    grp = FP8Group(blocks, ["q", "k", "v"])
    assert grp.sizes == [384, 128, 128] and grp.N == 640 and grp.K == K
    assert grp.nbytes == sum(b.nbytes for b in blocks)
    for b, (w0, s0) in zip(blocks, before):
        assert torch.equal(b.w.view(torch.uint8), w0.view(torch.uint8)), "member codes moved"
        assert torch.equal(b.s, s0), "member scales moved (misaligned scale rows)"
        assert b.w.untyped_storage().data_ptr() == grp.w.untyped_storage().data_ptr(), "not a view"
    fused = matmul_group(x, grp).split(grp.sizes, dim=-1)
    for i, (a, b) in enumerate(zip(sep, fused)):
        assert torch.equal(a.view(torch.int16), b.contiguous().view(torch.int16)), f"member {i} differs"
    again = [fp8_matmul(x, b) for b in blocks]
    for a, b in zip(sep, again):
        assert torch.equal(a, b), "a member alone changed after fusing"
    return "3 members: views, aligned scales, fused slices byte-identical, members still work"


class _W(Weights):
    def __init__(self, q):          # only what fuse_nvfp4_groups reads
        self.q, self.g = q, {}


def test_loader_fuses_fp8_and_skips_mixed():
    g = torch.Generator().manual_seed(1)
    K = 128
    q = {}
    for layer in range(2):
        for key, members in PROJ_GROUPS:
            for m in members:
                q[f"layers.{layer}.{m}"] = rand_block(256, K, g)

    class FakeNV:                    # stands in for an NVFP4Block: not an FP8Block
        N, K = 256, 128
    q["layers.1.mlp.up_proj"] = FakeNV()
    w = _W(q)
    made = w.fuse_nvfp4_groups(2)
    keys = sorted(w.g)
    assert "layers.1.mlp.gate_up" not in w.g, "a mixed group was fused"
    assert made == 5 and all(isinstance(w.g[k], FP8Group) for k in keys), (made, keys)
    return f"{made} FP8 groups, the mixed one skipped"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<40} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<40} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
