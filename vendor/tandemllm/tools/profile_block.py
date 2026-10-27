"""How much a verify step costs as a function of how many tokens it verifies.

This is the number the whole speculative design rests on. The step reads 26.93 GB of weights
whatever B is, so if verifying eight tokens costs what verifying one costs, then every accepted
token past the first is free and throughput is acceptance length divided by step time. Where the
curve starts to bend is where a longer draft stops paying.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402

PEAK_GBPS = 273.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--blocks", default="1,2,4,6,8,12,16,24,32")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--sweep-fused", action="store_true",
                    help="repeat the curve for each combination of the fused kernels")
    ap.add_argument("--two-stream", default="",
                    help="repeat the curve at these settings of QWEN38_TWO_STREAM, in this order, "
                         "in ONE process: e.g. off,on,off. Measured first is measured slowest, so "
                         "the setting that is read twice is the one that says how much of a "
                         "difference belongs to the order")
    a = ap.parse_args()

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    step_bytes = w.decode_step_bytes(cfg.num_hidden_layers)["total_GB"]
    eng = Qwen38Engine(cfg, w, max_len=a.prompt_len + 2048, device=a.device)
    ids = torch.randint(1000, 100000, (a.prompt_len,), device=a.device)
    with torch.no_grad():
        eng.forward(ids, start=0, last_only=True)
    pos = a.prompt_len

    from engine.model import FUSED
    if a.two_stream:
        import engine.model as M
        for setting in a.two_stream.split(","):
            M.TWO_STREAM = setting.strip() == "on"
            print(f"\n--- two streams over the independent projections: {setting.strip()} ---")
            run_curve(a, eng, pos, step_bytes)
        return

    names = ["norm", "gdn", "head", "attn", "gdnblock"]
    ship = ["norm", "gdn", "head"]
    sets = ([[], ["norm"], ["gdnblock"], ship, ship + ["gdnblock"]]
            if a.sweep_fused else [None])
    for combo in sets:
        if combo is not None:
            for n in names:
                FUSED[n] = n in combo
            print(f"\n--- fused: {'+'.join(combo) or 'none'} ---")
        run_curve(a, eng, pos, step_bytes)


def run_curve(a, eng, pos, step_bytes):
    print(f"{'B':>4} {'verify ms':>10} {'ms/token':>9} {'GB/s':>7} {'rollback ms':>12} "
          f"{'tok/s if all accepted':>22}")
    for B in [int(x) for x in a.blocks.split(",")]:
        blk = torch.randint(1000, 100000, (B,), device=a.device)
        with torch.no_grad():
            for _ in range(2):
                eng.forward_block(blk, start=pos)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.reps):
                eng.forward_block(blk, start=pos)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / a.reps
            keep = max(1, B // 2)
            eng.forward_block(blk, start=pos)
            eng.rollback_to(keep)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.reps):
                eng.rollback_to(keep)
            torch.cuda.synchronize()
            rb = (time.perf_counter() - t0) / a.reps
        print(f"{B:4d} {dt * 1e3:10.2f} {dt / B * 1e3:9.2f} {step_bytes / dt:7.1f} "
              f"{rb * 1e3:12.2f} {B / dt:22.2f}")

    tr = eng._trace
    print(f"block trace at B={B}: {tr.nbytes / 2**20:.1f} MiB "
          f"({(tr.S_entry.numel() * 4) / 2**20:.1f} MiB of it the entry state)")


if __name__ == "__main__":
    main()
