"""CPU tests for the NVFP4 drafter projections.

A drafter only proposes, so the output of the engine cannot change; what has to hold is narrower:
exactly the seven projections of every layer are quantised and nothing else, the bf16 path is the
F.linear it always was, and an NVFP4 projection computes the dequantised weight's product.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.dflash2 import _PROJ, _lin, quantise_projections  # noqa: E402


def _w(layers=2):
    torch.manual_seed(0)
    w = {"fc.weight": torch.randn(256, 512, dtype=torch.bfloat16),
         "norm.weight": torch.ones(256, dtype=torch.bfloat16)}
    for i in range(layers):
        for proj in _PROJ:
            w[f"layers.{i}.{proj}.weight"] = (torch.randn(256, 256) * 0.02).to(torch.bfloat16)
        w[f"layers.{i}.input_layernorm.weight"] = torch.ones(256, dtype=torch.bfloat16)
    return w


def test_only_the_seven_projections_are_quantised():
    w = _w()
    q = quantise_projections(w, 2)
    for k, v in q.items():
        is_proj = any(k.endswith(f"{p}.weight") for p in _PROJ)
        assert isinstance(v, torch.Tensor) != is_proj, k
    assert q["fc.weight"] is w["fc.weight"]
    assert isinstance(w["layers.0.mlp.up_proj.weight"], torch.Tensor)     # the input is untouched


def test_the_bf16_path_is_f_linear():
    x = torch.randn(3, 256, dtype=torch.bfloat16)
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    assert torch.equal(_lin(x, w), F.linear(x, w))


def test_an_nvfp4_projection_is_its_dequantised_product():
    w = _w(1)
    q = quantise_projections(w, 1)
    blk = q["layers.0.mlp.gate_proj.weight"]
    x = torch.randn(2, 5, 256, dtype=torch.bfloat16)
    got = _lin(x, blk)
    assert got.shape == (2, 5, 256)
    want = F.linear(x.reshape(-1, 256), blk.dequant()).view(2, 5, 256)
    assert torch.equal(got, want)
    # and it is close to the bf16 product it replaces (4-bit weights: loose)
    ref = F.linear(x, w["layers.0.mlp.gate_proj.weight"]).float()
    rel = (got.float() - ref).norm() / ref.norm()
    assert rel < 0.2, float(rel)


def test_the_quantised_set_is_built_once_per_snapshot():
    w = _w(1)
    a = quantise_projections(w, 1, key="snap-x")
    b = quantise_projections(_w(1), 1, key="snap-x")
    assert a is b


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
