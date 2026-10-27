"""On the board: the drafter's context projection in NVFP4.

The real shape ([5120, 25600], five tapped hidden states in, one out) through the engine's W4A16
paths at every sync row count a block can commit (1..16), at a prompt-length sync (260 rows, the
prefill tile) and a long one (1,200 rows, the unpack path): against the product with the
dequantised weight, within the bf16 rounding the skinny/v2 kernels are held to. Flag off: the
projection is the bf16 `F.linear`, bit for bit.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from engine.drafters import dflash2  # noqa: E402
from tools import nvfp4_skinny as SK  # noqa: E402

G = torch.Generator(device="cuda").manual_seed(2)


def _module():
    w = {"fc.weight": (torch.randn(5120, 25600, device="cuda", generator=G) * 0.01).to(torch.bfloat16),
         "hidden_norm.weight": (1.0 + 0.1 * torch.randn(5120, device="cuda", generator=G)).to(torch.bfloat16)}
    cfg = SimpleNamespace(target_layer_ids=[5, 19, 33, 47, 61], hidden_size=5120, rms_norm_eps=1e-6)
    return SimpleNamespace(w=w, cfg=cfg)


def test_every_sync_row_count_and_the_prefill_paths():
    m = _module()
    old, old_sk = dflash2.DRAFT_FC_NVFP4, SK.SKINNY
    worst = 0.0
    try:
        dflash2.DRAFT_FC_NVFP4 = True
        for skinny in (False, True):
            SK.SKINNY = skinny
            for n in list(range(1, 17)) + [260, 1200]:
                x = torch.randn(n, 25600, device="cuda", generator=G).to(torch.bfloat16)
                got = dflash2.DFlash2Module.project_context(m, x).float()
                fc = m.w["fc.nvfp4"]
                ref = dflash2._rms(F.linear(x, SK._exact(fc).to(torch.bfloat16)),
                                   m.w["hidden_norm.weight"], 1e-6).float()
                rel = ((got - ref).abs().max() / ref.abs().max()).item()
                worst = max(worst, rel)
                assert rel < 2e-2, (skinny, n, rel)
                assert torch.isfinite(got).all()
    finally:
        dflash2.DRAFT_FC_NVFP4, SK.SKINNY = old, old_sk
    return f"rows 1..16, 260, 1200 on v2 and on the skinny kernel: worst rel {worst:.1e} " \
           f"against the dequantised product"


def test_flag_off_is_the_bf16_projection_bit_for_bit():
    m = _module()
    old = dflash2.DRAFT_FC_NVFP4
    try:
        dflash2.DRAFT_FC_NVFP4 = False
        for n in (1, 3, 16, 260):
            x = torch.randn(n, 25600, device="cuda", generator=G).to(torch.bfloat16)
            got = dflash2.DFlash2Module.project_context(m, x)
            want = dflash2._rms(F.linear(x, m.w["fc.weight"]), m.w["hidden_norm.weight"], 1e-6)
            assert torch.equal(got, want), n
    finally:
        dflash2.DRAFT_FC_NVFP4 = old
    return "flag off: F.linear on the bf16 fc, bit for bit, at 1, 3, 16 and 260 rows"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
