"""/128: one Linear interface for FP8, NVFP4, BF16 (and the e4m3 head); tile tables warn.

`engine.model.linear` used to test `isinstance` for each weight format. Now every stored format has
`matmul` and the model calls it. Each `matmul` must return exactly what the old branch returned --
the same kernel with the same arguments -- which these CPU tests check byte for byte on random
weights (the board check is flag-off identity on the served model, where the kernels are the CUDA
ones). A shape missing from a tile table prints one line naming it, once.
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stderr

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.linear import BF16Block, Linear  # noqa: E402
from engine.model import linear  # noqa: E402
from tools.fp8_linear import BLOCK, FP8Block, FP8Group, fp8_matmul  # noqa: E402
from tools.head_gemv import FP8Head  # noqa: E402
from tools.nvfp4_linear import NVFP4Block  # noqa: E402

g = torch.Generator().manual_seed(0)


def fp8(N, K):
    codes = (torch.randn(N, K, generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    return FP8Block(codes, (torch.rand(N // BLOCK, K // BLOCK, generator=g) * 1e-3 + 1e-4).to(torch.bfloat16))


def test_every_format_is_a_linear():
    n = torch.randint(0, 255, (256, 64), dtype=torch.uint8)
    s = (torch.rand(256, 8) * 0.01 + 0.001).to(torch.float8_e4m3fn)
    blocks = [fp8(256, 128), BF16Block(torch.randn(8, 4)), NVFP4Block(n, s, 1.0),
              FP8Head(torch.zeros(4, 8, dtype=torch.float8_e4m3fn), torch.ones(4))]
    for b in blocks:
        assert isinstance(b, Linear), type(b).__name__
    # a torch.Tensor also has shape, nbytes and matmul, so it satisfies the protocol structurally;
    # that is why `linear()` tests for a tensor FIRST and sends it through F.linear
    assert isinstance(torch.randn(2, 2), Linear)
    return ", ".join(type(b).__name__ for b in blocks)


def test_fp8_matmul_unchanged():
    w = fp8(384, 256)
    for shape in ((1, 256), (1, 7, 256), (3, 5, 256)):
        x = (torch.randn(*shape, generator=g) * 0.5).to(torch.bfloat16)
        old = fp8_matmul(x.reshape(-1, 256), w).view(*shape[:-1], w.N)
        new = linear(x, w)
        assert new.shape == old.shape and torch.equal(new.view(torch.int16), old.view(torch.int16)), shape
    grp = FP8Group([fp8(256, 256), fp8(128, 256)], ["a", "b"])
    x = (torch.randn(4, 256, generator=g) * 0.5).to(torch.bfloat16)
    assert torch.equal(linear(x, grp), fp8_matmul(x, grp))
    return "FP8Block and FP8Group through linear() == fp8_matmul, 3 input ranks"


def test_bf16_and_tensor():
    w = torch.randn(6, 4, generator=g)
    x = torch.randn(2, 3, 4, generator=g)
    assert torch.equal(linear(x, w), F.linear(x, w))
    assert torch.equal(linear(x, BF16Block(w)), F.linear(x, w))
    return "plain tensor and BF16Block == F.linear"


def test_nvfp4_cpu_path_unchanged():
    from tools.nvfp4_linear import nvfp4_matmul
    n = torch.randint(0, 255, (256, 64), dtype=torch.uint8)
    s = (torch.rand(256, 8, generator=g) * 0.5 + 0.1).to(torch.float8_e4m3fn)
    w = NVFP4Block(n, s, 0.5)
    x = (torch.randn(2, 3, 128, generator=g) * 0.5).to(torch.bfloat16)
    old = nvfp4_matmul(x.reshape(-1, 128), w).view(2, 3, 256)
    assert torch.equal(linear(x, w), old)
    return "NVFP4Block through linear() == nvfp4_matmul"


def test_tile_warning_once():
    from tools import nvfp4_linear, tile_warn
    buf = io.StringIO()
    with redirect_stderr(buf):
        nvfp4_linear.pick_config(4224, 1152, 1)
        nvfp4_linear.pick_config(4224, 1152, 8)
    lines = [l for l in buf.getvalue().splitlines() if l.startswith("[tiles]")]
    assert len(lines) == 1 and "N=4224 K=1152" in lines[0], lines
    assert ("nvfp4_linear", 4224, 1152) in tile_warn.seen()
    return lines[0]


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<32} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<32} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
