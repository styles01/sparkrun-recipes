"""What a CUDA graph of the verify step would save, measured instead of argued.

`tools/block_budget.py` says how much of a block the GPU spends idle while the host catches up.
Two different fixes are on offer for that idle time and they are bounded differently:

  * a native host loop (the loop in Rust or C++ instead of Python) removes the host's own cost --
    the Python between launches and the Python that acts on each device-to-host read -- but it still
    launches every kernel one at a time and still waits on every read;
  * a CUDA graph of the verify pass removes the launches themselves inside the verify, from Python
    as it is, and leaves the rest of the block alone.

This measures the second one directly: the same verify block, eager and replayed from a captured
graph, at the same position with the same inputs, each timed with a synchronise on both sides so
the number is the wall clock the loop would see. The difference is the most a verify graph can be
worth. It is an upper bound: the engine's real verify changes position every block, and a graph
that follows it needs the position, the KV length and the tree shape to live on the device.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402


def timed(fn, reps: int) -> list[float]:
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1e3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--widths", default="8,16")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    cfg = load_config(a.model)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.prompt_len + 256)
    ids = torch.randint(1000, 100000, (a.prompt_len,), device="cuda")
    with torch.no_grad():
        eng.forward(ids, start=0, last_only=True)
    pos = a.prompt_len
    snap = eng.state.clone()

    for T in [int(x) for x in a.widths.split(",")]:
        tokens = torch.randint(1000, 100000, (T,), device="cuda")

        def eager():
            eng.forward_block(tokens, start=pos)

        with torch.no_grad():
            for _ in range(3):
                eng.state.restore(snap)
                eager()
            eng.state.restore(snap)
            e_ms = timed(eager, a.reps)
            eng.state.restore(snap)
            graph = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            try:
                with torch.cuda.stream(side):
                    eager()                                   # one more warm call on the side stream
                torch.cuda.current_stream().wait_stream(side)
                eng.state.restore(snap)
                with torch.cuda.graph(graph, stream=side):
                    eager()
            except Exception as exc:                          # noqa: BLE001 -- the answer IS the error
                print(f"T={T}: eager {statistics.median(e_ms):.2f} ms; the verify does not capture: "
                      f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}", flush=True)
                torch.cuda.synchronize()
                continue
            for _ in range(3):
                graph.replay()
            g_ms = timed(graph.replay, a.reps)
        me, mg = statistics.median(e_ms), statistics.median(g_ms)
        print(f"T={T}: eager {me:.2f} ms (p10 {sorted(e_ms)[len(e_ms) // 10]:.2f}), graph replay "
              f"{mg:.2f} ms (p10 {sorted(g_ms)[len(g_ms) // 10]:.2f}): a verify graph is worth "
              f"{me - mg:.2f} ms a block at most", flush=True)


if __name__ == "__main__":
    main()
