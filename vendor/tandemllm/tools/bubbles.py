"""The GPU's idle time inside a block, with the graphs ON (gate, number).

`tools/block_budget.py` traces with the torch profiler, which charges a replayed CUDA graph to the
replay: it cannot see the gaps between the kernels inside a graph, and since the verify is a
graph. Nsight Systems can (`--cuda-graph-trace=node`). This runs the served loop under it and sums,
over the decode window only, the time between one kernel's end and the next one's start:

    python tools/bubbles.py run --workload chat --max-new 192 ...    (under nsys, see below)
    python tools/bubbles.py report trace.sqlite --blocks N

    nsys profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi \\
        --capture-range-end=stop -o /tmp/bub python tools/bubbles.py run ...
    nsys export --type sqlite -o /tmp/bub.sqlite /tmp/bub.nsys-rep

`run` warms the loop, then brackets ONE generation's decode with cudaProfilerStart/Stop and prints
the number of blocks. `report` reads the kernel table of the export and prints, per block, kernel
time, idle time split by gap size (<= 5 us, 5-50 us, > 50 us), and the kernel count. What it
cannot see is a kernel's own fill/drain tail (idle SMs inside a running kernel); that is in the
kernel time.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BUCKETS = ((5_000, "<=5us"), (50_000, "5-50us"), (float("inf"), ">50us"))


def gaps(intervals: list[tuple[int, int]]) -> dict:
    """Kernel [start, end) intervals in ns (any order, overlaps allowed) -> busy and idle time.
    Overlapping kernels (two streams) count once; idle is the uncovered time between the first
    start and the last end."""
    iv = sorted(intervals)
    busy = 0
    idle = {name: 0 for _, name in BUCKETS}
    count = {name: 0 for _, name in BUCKETS}
    if not iv:
        return {"busy_ns": 0, "span_ns": 0, "idle_ns": idle, "idle_count": count, "kernels": 0}
    cur_s, cur_e = iv[0]
    for s, e in iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
            continue
        busy += cur_e - cur_s
        g = s - cur_e
        for lim, name in BUCKETS:
            if g <= lim:
                idle[name] += g
                count[name] += 1
                break
        cur_s, cur_e = s, e
    busy += cur_e - cur_s
    return {"busy_ns": busy, "span_ns": iv[-1][1] - iv[0][0], "idle_ns": idle,
            "idle_count": count, "kernels": len(iv)}


def read_kernels(path: str) -> list[tuple[int, int]]:
    con = sqlite3.connect(path)
    try:
        rows = con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    finally:
        con.close()
    return [(int(s), int(e)) for s, e in rows]


def report(g: dict, blocks: int) -> str:
    b = max(blocks, 1)
    ms = lambda ns: ns / 1e6 / b                                        # noqa: E731
    idle = sum(g["idle_ns"].values())
    lines = [f"{blocks} blocks, {g['kernels'] / b:.0f} kernels a block",
             f"span {ms(g['span_ns']):.2f} ms a block = kernels {ms(g['busy_ns']):.2f} + idle "
             f"{ms(idle):.2f}",
             "idle a block: " + ", ".join(f"{name} {g['idle_count'][name] / b:.0f} = "
                                          f"{ms(g['idle_ns'][name]):.2f} ms"
                                          for _, name in BUCKETS)]
    return "\n".join(lines)


def run(a) -> None:
    import torch
    from transformers import AutoTokenizer

    from tools import profile_cycle as pc
    from tools.block_budget import PROMPTS
    cfg, eng, drafter, arms, ng, k = pc.build(a)
    tk = AutoTokenizer.from_pretrained(cfg.path)
    s = tk.apply_chat_template([{"role": "user", "content": PROMPTS[a.workload]}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
    ids = tk(s, return_tensors="pt").input_ids[0].cuda()
    for _ in range(2):
        pc.cycle(eng, drafter, ids, a.max_new, k, pc.Phases(strict=False))
    if eng._graphs_for(2, 0) is not None:
        with torch.no_grad():
            print(f"[bubbles] verify graphs captured: {eng._graphs.precapture()}", flush=True)
    pc.cycle(eng, drafter, ids, a.max_new, k, pc.Phases(strict=False))
    ph = pc.Phases(strict=False)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    out, st = pc.cycle(eng, drafter, ids, a.max_new, k, ph)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"[bubbles] {a.workload}: {len(ph.block_ms)} blocks, {len(out)} tokens, "
          f"block {sum(ph.block_ms) / max(len(ph.block_ms), 1):.2f} ms (untraced by torch)",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default=None)
    r.add_argument("--nvfp4", default=None)
    r.add_argument("--fp8-head", default=None)
    r.add_argument("--ckpt8", required=True)
    r.add_argument("--ckpt16", required=True)
    r.add_argument("--corpus", default="")
    r.add_argument("--max-len", type=int, default=4096)
    r.add_argument("--max-new", type=int, default=192)
    r.add_argument("--workload", default="chat")
    r.add_argument("--fixed", type=int, default=0)
    r.add_argument("--latch", action="store_true", default=True)
    r.add_argument("--drop-idle", action="store_true", default=True)
    p = sub.add_parser("report")
    p.add_argument("sqlite")
    p.add_argument("--blocks", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        print(report(gaps(read_kernels(a.sqlite)), a.blocks))


if __name__ == "__main__":
    main()
