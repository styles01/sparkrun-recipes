"""Kernel names inside the parts of a block, from a trace `tools/block_budget.py --keep-traces` kept.

block_budget charges each kernel to the part that launched it; this prints, for the parts asked for,
which kernels make up their time -- the step between "the GDN mixer is 4.4 ms" and "the recurrence
kernel is 3 ms of it".

    python tools/trace_kernels.py results/kernels/k3-graph/trace-chat-16.json --parts gdn,commit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--parts", default="", help="substrings of the part names to break down")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    ev = json.load(open(a.trace))["traceEvents"]
    launch_ts, kernels, ann = {}, [], []
    for e in ev:
        cat, args = e.get("cat", ""), e.get("args") or {}
        if cat == "kernel":
            kernels.append((e["name"], e.get("dur", 0.0), args.get("correlation")))
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in args:
            launch_ts[args["correlation"]] = e["ts"]
        elif cat == "user_annotation" and str(e.get("name", "")).startswith(("C::", "PH::")):
            ann.append((e["ts"], e["ts"] + e.get("dur", 0.0), e["name"]))
    ann.sort(key=lambda x: (x[0], -x[1]))
    blocks = sum(1 for s, t, n in ann if n == "PH::draft") or 1
    order = sorted((launch_ts[c], i) for i, (_, _, c) in enumerate(kernels) if c in launch_ts)
    stack, ai, per = [], 0, defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for ts, i in order:
        while ai < len(ann) and ann[ai][0] <= ts:
            while stack and stack[-1][1] < ann[ai][0]:
                stack.pop()
            stack.append(ann[ai])
            ai += 1
        while stack and stack[-1][1] < ts:
            stack.pop()
        part = next((n[3:] for s, t, n in reversed(stack) if n.startswith("C::")), "(glue)")
        phase = next((n[4:] for s, t, n in reversed(stack) if n.startswith("PH::")), "?")
        name, dur, _ = kernels[i]
        row = per[f"{phase}|{part}"][name[:90]]
        row[0] += dur
        row[1] += 1
    want = [p for p in a.parts.split(",") if p]
    for key in sorted(per, key=lambda k: -sum(v[0] for v in per[k].values())):
        if want and not any(w in key for w in want):
            continue
        tot = sum(v[0] for v in per[key].values())
        print(f"{key}: {tot / 1e3 / blocks:.3f} ms a block")
        for name, (dur, n) in sorted(per[key].items(), key=lambda x: -x[1][0])[:a.top]:
            print(f"    {dur / 1e3 / blocks:8.3f} ms  {n / blocks:6.1f} calls  {name}")


if __name__ == "__main__":
    main()
