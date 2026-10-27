"""The gate for the two-stream projections: the same bytes, or it does not ship.

Every other fused kernel in this engine is a different arithmetic ORDER for the same quantity, and
each of them had to be argued for against a tolerance. This one is not: `par2` issues the same two
kernels, on the same inputs, with the same tiles, and moves one of them to a second stream. Nothing
about the arithmetic changes, so the only acceptable result is BIT-IDENTICAL -- `max |dlogit|`
exactly zero, the same recurrent state to the last bit, and the same tokens.

A tolerance here would be a bug, not a margin. If this ever reads anything but zero, the two streams
are racing rather than overlapping and the events are wrong.

Three shapes, because the three call sites are reached at different row counts: a prefill (the
general mixer), a verify block of eight and sixteen rows (the block path), and a single token (the
`_linear_attention_decode` glue, which has its own copy of the projection pair).

    python tools/twostream_gate.py --nvfp4 $NV --fp8-head $HEAD
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--greedy", type=int, default=64)
    ap.add_argument("--blocks", default="1,8,16")
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    a = ap.parse_args()

    if a.nvfp4:
        os.environ["QWEN38_NVFP4"] = a.nvfp4
    if a.fp8_head:
        os.environ["QWEN38_FP8_HEAD"] = a.fp8_head
    os.environ.setdefault("QWEN38_NVFP4_DEQUANT_FROM", "512")

    import engine.model as M
    from engine.loader import Weights

    cfg = load_config(a.model)
    t0 = time.time()
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    print(f"[load] {time.time() - t0:.1f}s  {w.report()}")
    blocks = [int(x) for x in a.blocks.split(",")]
    eng = M.Qwen38Engine(cfg, w, max_len=a.prompt_len + a.greedy + sum(blocks) + 64,
                         device=a.device)
    ids = torch.randint(1000, 100000, (a.prompt_len,), device=a.device)

    def arm(on: bool) -> dict:
        M.TWO_STREAM = on
        eng.reset()
        out = {}
        with torch.no_grad():
            logits = eng.forward(ids, start=0, last_only=True)
            out["prefill"] = logits.float().reshape(-1).clone()
            out["state"] = eng.state.S.clone()
            pos = a.prompt_len
            tok = int(out["prefill"].argmax())
            toks = []
            for _ in range(a.greedy):
                toks.append(tok)
                lg = eng.forward(torch.tensor([tok], device=a.device), start=pos, last_only=True)
                tok = int(lg.float().reshape(-1).argmax())
                pos += 1
            out["tokens"] = toks
            out["decode"] = lg.float().reshape(-1).clone()
            for B in blocks:
                blk = torch.randint(1000, 100000, (B,), device=a.device,
                                    generator=torch.Generator(device=a.device).manual_seed(7 + B))
                out[f"block{B}"] = eng.forward_block(blk, start=pos).float().clone()
                eng.rollback_to(0)
        M.TWO_STREAM = False
        return out

    # Both orders, because a first reading in a fresh process is a different allocation history --
    # and if the two streams were racing, one order would be likelier to show it than the other.
    off1, on1 = arm(False), arm(True)
    on2, off2 = arm(True), arm(False)

    ok = True
    print(f"\n{'tensor':>12}{'max |d| off/on':>18}{'max |d| on/on':>16}{'max |d| off/off':>18}")
    for key in ("prefill", "state", "decode", *(f"block{B}" for B in blocks)):
        d1 = float((on1[key] - off1[key]).abs().max())
        d2 = float((on2[key] - on1[key]).abs().max())
        d3 = float((off2[key] - off1[key]).abs().max())
        ok = ok and d1 == 0.0 and d2 == 0.0 and d3 == 0.0
        print(f"{key:>12}{d1:>18.3e}{d2:>16.3e}{d3:>18.3e}")
    same_tokens = off1["tokens"] == on1["tokens"] == on2["tokens"] == off2["tokens"]
    ok = ok and same_tokens
    print(f"\n{a.greedy} greedy tokens identical across all four runs: {same_tokens}")
    print(f"GATE  two streams over the independent projections are bit-identical: "
          f"{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
