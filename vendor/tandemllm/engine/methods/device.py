"""The device a per-device file is keyed by (price files now; tile tables with ENG-251)."""

from __future__ import annotations

import re


def device_id(name: str | None = None) -> str:
    """The DGX Spark's GB10 is "dgx-spark"; any other GPU is its name in lower case with dashes
    ("nvidia-rtx-pro-6000" ...); no GPU is "cpu". `name`: the GPU's name (default: device 0's)."""
    if name is None:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        name = torch.cuda.get_device_name(0)
    if "GB10" in name:
        return "dgx-spark"
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
