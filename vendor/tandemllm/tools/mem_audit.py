"""Where a token of window actually goes — the P0 audit of the context plan.

The 64k attempt (2026-09-18, 07:47, reverted two minutes later) read the board at 111 GB used /
9 GB free and concluded the window has to wait for a paged cache. That reading cannot be a
steady-state one: the log of the same start says `loaded in 76.5 s`, and the engine was reverted
95 seconds after it was started — the reading was taken during a load, with the previous engine
still draining. It measured two engines, not one.

This tool measures what that reading was supposed to measure: the same serving stack the server
builds (`server/app.py`: weights -> engine -> two drafter arms -> lookup), one `max_len` at a
time, one process, no server, no forwards. The number to watch is `allocated`; the arithmetic it
is checked against is 64 KiB/token of engine KV and 20 KiB/token per drafter arm, i.e.
~106.5 KiB/token of window for the served configuration's two arms.

Run it with the box held — it is an engine load:

    cd ~/qwen38-spark-engine
    PYTHONPATH=~/pylibs ~/recipes/ling3-flash-dgx-spark/.venv/bin/python -u tools/mem_audit.py \
        --lengths 4096,32768,65536,131072,262144 \
        --nvfp4 ~/nvfp4/mlp-clip.safetensors,~/nvfp4/gdn-clip.safetensors,~/nvfp4/attn-clip.safetensors \
        --fp8-head ~/nvfp4/head-fp8.safetensors

It skips a length rather than risk the box when `/proc/meminfo` says less than `--min-avail-gb`
(24) is available.
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import torch


def meminfo() -> dict:
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            d[k] = int(v.split()[0]) / 1e6  # kB -> GB
    return d


def report(tag: str) -> None:
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    p = torch.cuda.max_memory_allocated() / 1e9
    mi = meminfo()
    print(f"[mem] {tag:>22}  alloc={a:7.2f}  resv={r:7.2f}  peak={p:7.2f}  "
          f"avail={mi['MemAvailable']:6.1f}  cached={mi['Cached']:6.1f}  "
          f"free={mi['MemFree']:6.1f}", flush=True)


def census(root, label: str, min_mb: float = 64.0) -> None:
    """Every tensor reachable through __dict__/containers, by name, biggest first."""
    out: list[tuple[str, int]] = []
    seen: set[int] = set()

    def walk(obj, name: str, depth: int) -> None:
        if depth > 3 or id(obj) in seen:
            return
        seen.add(id(obj))
        if torch.is_tensor(obj):
            out.append((name, obj.numel() * obj.element_size()))
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items())[:64]:
                walk(v, f"{name}.{k}", depth + 1)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj[:64]):
                walk(v, f"{name}[{i}]", depth + 1)
        elif hasattr(obj, "__dict__"):
            for k, v in list(vars(obj).items())[:64]:
                walk(v, f"{name}.{k}", depth + 1)

    walk(root, label, 0)
    total = sum(b for _, b in out) / 1e9
    print(f"[census] {label}: {total:.2f} GB in {len(out)} tensors", flush=True)
    for n, b in sorted(out, key=lambda kv: -kv[1])[:14]:
        if b >= min_mb * 1e6:
            print(f"   {b/1e9:7.3f} GB  {n}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="4096,32768,65536,131072,262144")
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--ckpt8", default="train/ft-b8-v2")
    ap.add_argument("--ckpt16", default="train/ft-b16")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--min-avail-gb", type=float, default=24.0)
    ap.add_argument("--census-at", type=int, default=0,
                    help="take the tensor census at this length (0 = the largest)")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    if not a.census_at:
        a.census_at = max(lengths)

    from engine.config import load_config
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.drafters.ngram import NgramDrafter
    from engine.loader import Weights
    from engine.model import Qwen38Engine

    report("start (no model)")
    cfg = load_config(None)
    t0 = time.time()
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4,
                fp8_head=os.path.expanduser(a.fp8_head) if a.fp8_head else None)
    print(f"[load] {w.report()}  in {time.time()-t0:.1f} s", flush=True)
    report("weights")

    for L in lengths:
        mi = meminfo()
        if mi["MemAvailable"] < a.min_avail_gb:
            print(f"[mem] SKIP max_len={L}: available {mi['MemAvailable']:.1f} GB "
                  f"< {a.min_avail_gb}", flush=True)
            continue
        t0 = time.time()
        eng = Qwen38Engine(cfg, w, max_len=L)
        report(f"engine {L}")
        small = DFlash2Drafter(eng, a.ckpt8, blocks=1, path="greedy", max_len=L, block=8)
        small._build()
        large = DFlash2Drafter(eng, a.ckpt16, blocks=1, path="greedy", max_len=L, block=16)
        large._build()
        report(f"+ drafters {L}")
        ng = NgramDrafter(corpus_path=a.corpus, min_order=3, max_depth=16, node_budget=15,
                          branch_top_k=3, min_expected=0.2, alpha=0.6, corpus_weight=0.5,
                          min_corpus_order=8, verify_base_ms=121.7,
                          verify_per_node_ms=(129.2 - 121.7) / 8)
        report(f"+ lookup {L}")
        kv = (eng.kv.k.numel() + eng.kv.v.numel()) * 2 / 1e9
        dk = (small._ck.numel() + small._cv.numel()
              + large._ck.numel() + large._cv.numel()) * 2 / 1e9
        print(f"[arith] {L:>7}: engine KV {kv:6.3f} GB  drafter KV {dk:6.3f} GB  "
              f"expected {(kv+dk):6.3f} GB  ({1000*kv/L:.1f}+{1000*dk/L:.1f} bytes/token)",
              flush=True)
        if L == a.census_at:
            census(eng, "engine")
            census(small, "drafter-small")
            census(large, "drafter-large")
        del eng, small, large, ng
        gc.collect()
        torch.cuda.empty_cache()
        report(f"freed {L}")
        print(f"[mem] max_len={L} done in {time.time()-t0:.1f} s", flush=True)

    report("end")


if __name__ == "__main__":
    main()
