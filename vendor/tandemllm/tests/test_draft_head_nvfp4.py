"""The drafter's NVFP4 vocabulary head.

Quantised a row block at a time with one per-tensor scale taken over the whole head, so the result
must be exactly what quantising the dequantised head in one piece gives -- the block size is a
memory decision and may not be a numerical one. And the flag is off by default.
"""

from __future__ import annotations

import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_block_quantisation_equals_one_piece():
    from tools.head_gemv import FP8Head, head_to_nvfp4
    from tools.nvfp4_linear import E2M1_MAX, E4M3_MAX, quantize_to_nvfp4
    g = torch.Generator().manual_seed(0)
    w = torch.randn(300, 128, generator=g) * 0.05
    w[17, 5] = 0.9                                   # an outlier in one block only
    amax = w.abs().amax(1, keepdim=True)
    scale = (amax / 448.0).squeeze(1)
    head = FP8Head((w / amax * 448.0).clamp(-448, 448).to(torch.float8_e4m3fn), scale)
    blocked = head_to_nvfp4(head, rows=64)
    full = head.w.float() * head.s[:, None]
    s2 = float((full.abs().max() / (E2M1_MAX * E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny))
    one = quantize_to_nvfp4(full, scale_2=s2)
    assert blocked.s2 == one.s2
    assert torch.equal(blocked.w, one.w)
    assert torch.equal(blocked.s.view(torch.uint8), one.s.view(torch.uint8))
    assert blocked.shape == (300, 128)
    return "row blocks of 64 over 300 rows == one piece, codes, scales and s2"


def test_off_by_default():
    from engine.drafters import dflash2
    old = os.environ.pop("QWEN38_DRAFT_HEAD_NVFP4", None)
    try:
        assert importlib.reload(dflash2).DRAFT_HEAD_NVFP4 is False
    finally:
        if old is not None:
            os.environ["QWEN38_DRAFT_HEAD_NVFP4"] = old
        importlib.reload(dflash2)
    return "QWEN38_DRAFT_HEAD_NVFP4 unset -> the e4m3 head"


def test_the_context_projection_in_nvfp4():
    """With the flag the drafter's `fc` is quantised once, cached beside the bf16 weight
    (which the module still reports its dtype and device from), and used for the projection."""
    from types import SimpleNamespace

    from engine.drafters import dflash2
    from tools.nvfp4_linear import NVFP4Block
    g = torch.Generator().manual_seed(1)
    H, taps = 16, 2
    w = {"fc.weight": (torch.randn(H, taps * 128, generator=g) * 0.05).to(torch.bfloat16),
         # the drafter's norm is `normalize(x) * w`, not the target's `(1 + w)`
         "hidden_norm.weight": torch.ones(H, dtype=torch.bfloat16)}
    fake = SimpleNamespace(w=w, cfg=SimpleNamespace(target_layer_ids=[1, 2], hidden_size=128,
                                                     rms_norm_eps=1e-6))
    fake.cfg.hidden_size = taps * 128 // len(fake.cfg.target_layer_ids)
    x = (torch.randn(3, taps * 128, generator=g)).to(torch.bfloat16)
    ref = dflash2.DFlash2Module.project_context(fake, x)
    old = dflash2.DRAFT_FC_NVFP4
    dflash2.DRAFT_FC_NVFP4 = True
    try:
        got = dflash2.DFlash2Module.project_context(fake, x)
    finally:
        dflash2.DRAFT_FC_NVFP4 = old
    assert isinstance(w.get("fc.nvfp4"), NVFP4Block) and w["fc.weight"].dtype == torch.bfloat16
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert 0.0 < rel < 0.2, rel
    return f"quantised once and cached, relative error {rel:.3f} against bf16"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
