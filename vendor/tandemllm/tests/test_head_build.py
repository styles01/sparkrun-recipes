"""/119: the e4m3 head built at load (`--fp8-head build`) is the head the file path loads.

`tools/quant_head.py build` writes the served head file from the checkpoint's bf16 `lm_head` with
`quantize_head_fp8`; `Weights.build_fp8_head` runs the same function at load. These tests pin the
contract on a random head on the CPU: the spec parser, byte identity between the build path and a
file written and loaded the served way, the byte accounting, and that no spec leaves the bf16 head.
The board check (built at load == ~/nvfp4/head-fp8.safetensors, every byte) is in the ledger.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.loader import HEAD_BUILD_RATIOS, Weights, parse_head_build  # noqa: E402
from tools.head_gemv import FP8Head, quantize_head_fp8  # noqa: E402


def _weights(seed: int = 0, n: int = 1000, k: int = 256) -> Weights:
    g = torch.Generator().manual_seed(seed)
    w = Weights.__new__(Weights)          # no checkpoint: only the fields the head paths touch
    w.device = "cpu"
    head = torch.randn(n, k, generator=g) * 0.05
    head[3, 7] = 1.5                      # an outlier row, where the ratio search matters
    w.t = {"lm_head.weight": head.to(torch.bfloat16)}
    w.bytes_other = n * k * 2
    w.fp8_head_source = None
    return w


def test_parse():
    assert parse_head_build(None) is None
    assert parse_head_build("") is None
    assert parse_head_build("~/nvfp4/head-fp8.safetensors") is None
    assert parse_head_build("build") == HEAD_BUILD_RATIOS == (1.0, 0.95, 0.90)
    assert parse_head_build("build:1.0") == (1.0,)
    assert parse_head_build(" build:1.0,0.9 ") == (1.0, 0.9)
    for bad in ("build:", "build:0", "build:1.5", "build:-1"):
        try:
            parse_head_build(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} accepted")
    return "build / build:ratios / path / empty"


def test_build_equals_file():
    """Built at load == quantised the way quant_head.py writes the file, then loaded from it."""
    a = _weights()
    ref = a.t["lm_head.weight"].clone()
    a.build_fp8_head()
    head = a.t["lm_head.weight"]
    assert isinstance(head, FP8Head)
    assert a.fp8_head_source == "build:1,0.95,0.9", a.fp8_head_source
    direct = quantize_head_fp8(ref, ratios=HEAD_BUILD_RATIOS)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "head-fp8.safetensors")
        save_file({"lm_head.weight": direct.w.cpu(), "lm_head.weight_scale": direct.s.cpu()}, path,
                  metadata={"format": "fp8-e4m3", "scale": "per-row", "ratios": "1.0,0.95,0.90"})
        b = _weights()
        b.load_fp8_head(path)
    fh = b.t["lm_head.weight"]
    assert torch.equal(head.w.view(torch.uint8), fh.w.view(torch.uint8)), "codes differ"
    assert torch.equal(head.s, fh.s), "scales differ"
    assert a.bytes_other == b.bytes_other
    return f"{head.N}x{head.K}: codes and scales byte-identical, bytes_other {a.bytes_other}"


def test_ratios_matter():
    """The ratio set is part of the contract: a single ratio gives other bytes on the outlier row."""
    a, b = _weights(), _weights()
    a.build_fp8_head()
    b.build_fp8_head((1.0,))
    same = torch.equal(a.t["lm_head.weight"].s, b.t["lm_head.weight"].s)
    assert not same, "the ratio search changed nothing; the outlier fixture is too weak"
    return "served ratios differ from (1.0,) as expected"


def test_no_spec_keeps_bf16():
    w = _weights()
    before = w.t["lm_head.weight"].clone()
    spec = None
    if parse_head_build(spec) is not None:
        w.build_fp8_head()
    assert torch.equal(w.t["lm_head.weight"], before)
    return "bf16 head untouched"


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
