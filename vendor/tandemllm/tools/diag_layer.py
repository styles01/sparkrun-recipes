"""Localise a numerical difference: run one layer on the reference's own input.

`refcheck --layers` shows how far the engine's hidden states have drifted by layer L, which mixes
the error the layer itself makes with the error it inherited. This tool removes the inheritance: it
feeds `hidden[L]` from the reference dump into the engine's layer L, with state and cache empty, and
compares the result against `hidden[L + 1]`. A layer that is implemented correctly comes out at the
bf16 noise floor no matter how bad the accumulated drift is.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine, rms_norm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/qwen38-ref.pt")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", default="0,1,2,3,4,7,8,11,30,31,60,63")
    a = ap.parse_args()

    ref = torch.load(a.ref, map_location="cpu")
    hidden = ref["hidden"]
    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=hidden[0].shape[0] + 16, device=a.device)

    print(f"{'layer':>5} {'kind':>7} {'mixer rel':>11} {'out rel':>11} {'out max':>9}")
    for L in [int(x) for x in a.layers.split(",")]:
        h_in = hidden[L].to(a.device, torch.bfloat16)[None]
        h_ref = hidden[L + 1].to(a.device, torch.float32)
        eng.reset()
        p = f"layers.{L}"
        positions = torch.arange(0, h_in.shape[1], device=a.device)
        with torch.no_grad():
            x = rms_norm(h_in, w.norm(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps)
            if cfg.is_linear(L):
                mix = eng.linear_attention(x, p, L, False)
                kind = "linear"
            else:
                mix = eng.attention(x, p, L, 0, positions)
                kind = "ATTN"
            h = h_in + mix
            y = rms_norm(h, w.norm(f"{p}.post_attention_layernorm.weight"), cfg.rms_norm_eps)
            h = h + eng.mlp(y, p)
        d = (h[0].float() - h_ref).abs()
        rms = h_ref.pow(2).mean().sqrt()
        # the mixer's own share, isolated by subtracting the reference's residual input
        dm = ((h_in[0].float() + mix[0].float()) - h_ref).abs()
        print(f"{L:5d} {kind:>7} {(dm.mean() / rms).item():11.3e} "
              f"{(d.mean() / rms).item():11.3e} {d.max().item():9.4f}")


if __name__ == "__main__":
    main()
