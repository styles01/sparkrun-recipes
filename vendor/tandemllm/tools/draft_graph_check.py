"""Does the graphed draft call propose what the eager one proposes?

Loads the target and one block drafter, prefills a prompt, then at several positions -- the first
block, one inside the first 2,048 tokens and one past them, where the sliding window starts to slide
-- asks the drafter for a block eagerly and from its graph, and compares the proposed tokens, the
candidate table and the lattice scores. The drafter only proposes, so a difference could not change
the engine's output; this says whether it changes the proposals. Timing of both on the same calls.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--nvfp4", default=os.environ.get("QWEN38_NVFP4"))
    ap.add_argument("--fp8-head", default=os.environ.get("QWEN38_FP8_HEAD"))
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    from engine.config import load_config
    from engine.drafters import dflash2
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    dflash2.HOST_WALK = True
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=8192)
    d = DFlash2Drafter(eng, a.ckpt, blocks=1, path="greedy", max_len=8192, block=a.block)
    d._build()
    g = torch.Generator().manual_seed(11)
    fails, lines, t_e, t_g = 0, [], [], []
    # both sides of the 1,024 class boundary, and one past the 2,048 window
    for n in (300, 900, 1100, 3000):
        ids = torch.randint(1000, 100000, (n,), generator=g).tolist()
        eng.reset()
        d.reset()
        with torch.no_grad():
            eng.forward(torch.tensor(ids).cuda(), start=0, last_only=True)
            d.sync(ids, eng.hidden_post_norm[0], 0)
            ctx = ids + [int(ids[-1])]
            out = {}
            for flag in (False, True):
                dflash2.DRAFT_GRAPH = flag
                d.propose(ctx, a.block - 1)                     # warm / capture
                torch.cuda.synchronize()
                ts = []
                for _ in range(a.reps):
                    t = time.perf_counter()
                    toks = d.propose(ctx, a.block - 1)
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t) * 1e3)
                out[flag] = (toks, d._lattice[0].clone(), d._lattice[1].clone())
                (t_g if flag else t_e).append(statistics.median(ts))
            dflash2.DRAFT_GRAPH = False
        same_t = out[False][0] == out[True][0]
        same_c = torch.equal(out[False][1], out[True][1])
        ds = (out[False][2].float() - out[True][2].float()).abs().max().item()
        fails += not same_t
        lines.append(f"context {n}: tokens {'same' if same_t else 'DIFFER'}, candidates "
                     f"{'same' if same_c else 'differ'}, max|d score| {ds:.2e}; draft call "
                     f"eager {t_e[-1]:.2f} ms, graph {t_g[-1]:.2f} ms")
    print("\n".join(lines))
    g = getattr(d, "_graph")
    print(f"DRAFT GRAPH {'PASS' if fails == 0 else 'FAIL'}: {len(lines) - fails}/{len(lines)} "
          f"positions propose the same block; graph stats {g.stats}, classes {sorted(g.graphs)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
