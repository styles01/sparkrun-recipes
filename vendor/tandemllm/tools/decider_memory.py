#!/usr/bin/env python3
"""What the decision models take from the board, loaded the way the engine loads them (ENG-179).

    python tools/decider_memory.py --dir ~/.qwen38-spark-engine/deciders/laya [--device cuda] \\
        [--models laya-english,laya-multilingual,laya-typed-decisions] [--abort-below-gib 23] \\
        [--asks 20] [--service-start] [--json OUT]

Stage A's memory gate: with the three checkpoints loaded beside the engine, MemAvailable stays at
or above the floor plus the margin (15 + 6 GiB). This loads them one after the other through the
real registry and the residency planner (`server/decider_api.py`: the decider's claim against
MemAvailable, never during a hold), each into its own memory pool and stream, and reads
MemAvailable after each load; then asks each model one question `--asks` times through the
batcher (the idle-window gate is this process's own lock: there is no engine here) and reports the
median; then unloads them and reads MemAvailable again, which says whether the pools went back.

It runs in its own process, so it pays a CUDA context the engine would not (about 0.5 to 1 GiB):
an upper bound on what the engine would take. If MemAvailable falls under `--abort-below-gib`
at any point, everything is unloaded at once and the run says so. `--service-start` is the
planner's start-of-service rule (the box lock is not read; the hold's pause file still is), for a
measurement beside a lock holder that is not a hold, such as an eval against the live engine.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.decision import catalogue, routers  # noqa: E402
from engine.decision.device import Device  # noqa: E402
from engine.decision.registry import Registry  # noqa: E402
from server import decider_api, memsplit  # noqa: E402

GIB = float(1 << 30)
QUESTION = {"cls": {"type": "choice", "instructions": "What kind of work does this request ask for?",
                    "criteria": {"debug": "find and fix a bug", "code_edit": "write or change code",
                                 "plan": "plan the work", "chat": "a short question"}}}
STATE = "The build fails since this morning: TypeError in src/parser.ts line 41 after the refactor."


def mem() -> float | None:
    b = memsplit.mem_available()
    return round(b / GIB, 2) if b is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/.qwen38-spark-engine/deciders/laya"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default="laya-english,laya-multilingual,laya-typed-decisions")
    ap.add_argument("--abort-below-gib", type=float, default=23.0)
    ap.add_argument("--asks", type=int, default=20)
    ap.add_argument("--service-start", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import torch
    names = [n.strip() for n in a.models.split(",") if n.strip()]
    models = catalogue(names, a.dir)
    dev = Device(a.device)
    reg = Registry(models, gate=threading.Lock(), planner=decider_api.planner({}), device=dev,
                   routers=routers(models), log=lambda s: print(s, flush=True))
    out = {"device": str(dev), "mem_before_gib": mem(), "loads": [], "asks": {}, "aborted": None}
    print(f"MemAvailable before: {out['mem_before_gib']} GiB; device {dev}", flush=True)

    def guard(where: str) -> bool:
        m = mem()
        if m is not None and m < a.abort_below_gib:
            out["aborted"] = f"MemAvailable {m} GiB under {a.abort_below_gib} at {where}"
            print("ABORT: " + out["aborted"], flush=True)
            for n in reg.loaded():
                reg.unload(n)
            return False
        return True

    for n in names:
        t0 = time.time()
        d = reg.load(n, service_start=a.service_start, wait=True)
        row = {"model": n, "allowed": d.get("allowed"), "reason": d.get("reason"),
               "planner_mem_gib": d.get("mem_available_gib"), "state": reg.state[n],
               "load_s": round(time.time() - t0, 1), "mem_after_gib": mem(),
               "device_bytes_gib": round(reg.models[n].device_bytes() / GIB, 3),
               "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 3)
               if dev.cuda else None, "error": reg.error.get(n)}
        out["loads"].append(row)
        print(json.dumps(row), flush=True)
        if not d.get("allowed") or not guard(f"after loading {n}"):
            break
    out["mem_loaded_gib"] = mem()
    if out["aborted"] is None:
        for n in reg.loaded():
            lat = []
            for _ in range(a.asks):
                t0 = time.perf_counter()
                r = reg.ask(STATE, QUESTION, model=n, deadline_ms=10_000).result(60)
                lat.append((time.perf_counter() - t0) * 1e3)
            out["asks"][n] = {"n": len(lat), "p50_ms": round(statistics.median(lat), 2),
                              "min_ms": round(min(lat), 2), "max_ms": round(max(lat), 2),
                              "first_ms": round(lat[0], 2), "choice": r["answers"]["cls"]["choice"],
                              "path": r["path"]}
            print(f"{n}: one question {out['asks'][n]}", flush=True)
            if not guard(f"asking {n}"):
                break
    for n in reg.loaded():
        reg.unload(n)
    import gc
    gc.collect()
    time.sleep(2)
    out["mem_unloaded_gib"] = mem()
    out["allocated_unloaded_gib"] = round(torch.cuda.memory_allocated() / GIB, 3) if dev.cuda else None
    out["reserved_unloaded_gib"] = round(torch.cuda.memory_reserved() / GIB, 3) if dev.cuda else None
    print(f"MemAvailable: before {out['mem_before_gib']}, all loaded {out['mem_loaded_gib']}, "
          f"after unload {out['mem_unloaded_gib']} GiB; aborted: {out['aborted']}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if out["aborted"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
