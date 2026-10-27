"""Is the chunked delta rule the same answer when its own matmuls change precision?

The chunked form is written in fp32 throughout, like the reference, and on a prefill it is 55 % of
the pass: 110 ms a layer at 8k, 5.3 s of a 9.6 s pass, against 3.4 TFLOP of small matrix products
that in true fp32 run on the CUDA cores while the tensor cores idle.

The tensors stay fp32 either way and so does the recurrent state, which is the part the research
note is emphatic about. What changes is the arithmetic of the products: `tf32` keeps the operands
fp32 and lets the tensor cores do them at ten mantissa bits; `bf16` casts the operands, which halves
their read as well.

This is the yardstick, not a tolerance: the same three numbers the unpacked-prefill gate used, on
the same held-out prose, in one process, against the engine compared with itself on the identical
path. A change is inside the yardstick if it disagrees with the engine less than the published
reference disagrees with itself when only its attention kernel changes -- argmax 0.9875, KL 0.00060.

    python tools/mm_gate.py --tokens 1024 --nvfp4 ...,...,... --fp8-head ...
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
from engine.model import Qwen38Engine  # noqa: E402
import engine.gdn as gdn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--modes", default="fp32,tf32,bf16")
    ap.add_argument("--text", default=None)
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(a.model)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=max(2048, a.tokens + 8))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    text = open(a.text or os.path.join(root, "bench", "heldout_prose.txt")).read()
    ids = tok(text, return_tensors="pt").input_ids[0][:a.tokens].cuda()

    def run(mode: str) -> torch.Tensor:
        gdn.PREFILL_MM = mode
        eng.reset()
        with torch.no_grad():
            return eng.forward(ids, start=0)[0].float()

    modes = a.modes.split(",")
    base = run(modes[0])
    again = run(modes[0])
    rows = [(f"{modes[0]} vs itself", base, again)]
    for mode in modes[1:]:
        rows.append((f"{modes[0]} vs {mode}", base, run(mode)))
    gdn.PREFILL_MM = "fp32"

    def nll(z: torch.Tensor) -> float:
        return float(-F.log_softmax(z[:-1], -1).gather(1, ids[1:, None]).mean())

    for name, x, y in rows:
        am = (x.argmax(-1) == y.argmax(-1)).float().mean()
        kl = F.kl_div(F.log_softmax(y, -1), F.log_softmax(x, -1),
                      log_target=True, reduction="batchmean")
        print(f"{name:22s} argmax {float(am):.6f}  max|dlogit| {float((x - y).abs().max()):.4f}  "
              f"KL {float(kl):.8f} nats   NLL {nll(y):.4f}")
    print(f"{'yardstick':22s} the published reference vs itself: argmax 0.987500, KL 0.00060000")


if __name__ == "__main__":
    main()
