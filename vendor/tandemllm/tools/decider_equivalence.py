#!/usr/bin/env python3
"""Does the engine's decision runtime answer exactly as the Laya SDK does? (ENG-179's gate)

    python tools/decider_equivalence.py --dir ~/.qwen38-spark-engine/deciders/laya \\
        --sdk ~/deciders-equiv/sdk [--device cpu|cuda] [--models laya-english,...] [--json OUT]

Arbiter's tools/equivalence.py (the code source), on the engine's code. Two paths over the same
22 questions (tools/decider_cases.py):

  reference  `laya.Agent.system_one`, one forward per request, exactly as the SDK ships it: on
             CUDA fp32 parameters under torch.autocast(bf16), on the CPU plain fp32. The SDK is a
             dev dependency of this gate only: `--sdk` puts the directory holding its `laya`
             package on the path; the engine never imports it.
  engine     `engine.decision.laya.model.LayaModel`, loaded as the registry loads it (its own
             stream and memory pool on CUDA), the same arithmetic: once per call, which is what
             the runtime does by default (`QWEN38_DECIDER_COLLATE=call`: each caller's rows in a
             forward of their own), and once with all of a checkpoint's rows in one padded
             forward, which is `collate=rows` (several callers' rows collated, arbiter's way).

Reported per checkpoint and path: the largest absolute difference in any probability the client
sees (rounded to four decimals, so 1e-4 is the floor of every figure), and whether the argmax
(the option chosen, the rounded score level, true or false) ever moves. Stage A's gate is the
default path: 0.00e+00 and every argmax unchanged. The collated path is held to arbiter's gate,
under 5e-3 with every argmax unchanged, and reported. (Arbiter's tool overwrote its batched
answers with the per-request ones before comparing, so its "0.00 batched" was the per-request
path; on the GB10 under autocast the collated path is 3.4e-3, 2026-10-01.) Also checked: the
router sends each case where arbiter's did.

Checkpoints are loaded one at a time and freed in between: one model in memory at a time.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decider_cases import CASES  # noqa: E402
from engine.decision.laya import router as laya_router  # noqa: E402
from engine.decision.laya.model import NAMES, SUBFOLDER, LayaModel  # noqa: E402


def probs_of(a: dict) -> np.ndarray:
    if a["type"] == "noul":
        return np.array([1.0 - a["noul"], a["noul"]])
    return np.array(list(a["probabilities"].values()), dtype=float)


def pick_of(a: dict) -> str:
    if a["type"] == "choice":
        return a["choice"]
    if a["type"] == "noul":
        return "true" if a["noul"] >= 0.5 else "false"
    return str(int(np.argmax(list(a["probabilities"].values()))))


def compare(ref: dict, got: dict) -> tuple[float, int, int, list[str]]:
    """(max |dp|, argmax agreements, compared, the keys whose argmax moved)."""
    worst, agree, moved = 0.0, 0, []
    keys = [k for k in ref if k in got]
    for k in keys:
        pa, pb = probs_of(ref[k]), probs_of(got[k])
        n = min(len(pa), len(pb))
        worst = max(worst, float(np.abs(pa[:n] - pb[:n]).max()))
        if pick_of(ref[k]) == pick_of(got[k]):
            agree += 1
        else:
            moved.append(k)
    return worst, agree, len(keys), moved


def free() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_reference(name: str, root: str, device: str, cases: list) -> dict:
    from laya.agent import Agent
    agent = Agent(root, device=device, subfolder=SUBFOLDER[name])
    out = {}
    for case in cases:
        for qid, ans in agent.system_one(case["state"], case["questions"])["answers"].items():
            out[f"{case['name']}/{qid}"] = ans
    del agent
    free()
    return out


def run_engine(name: str, root: str, device: str, cases: list) -> tuple[dict, dict, dict]:
    """(batched answers, per-request answers, what the load reported)."""
    where = None
    if device.startswith("cuda"):
        from engine.decision.device import Device, Placement
        where = Placement(Device(device))
    m = LayaModel(name, root)
    t0 = time.time()
    m.load(where)
    info = {"precision": m.precision(), "load_s": round(time.time() - t0, 1),
            "device_bytes": m.device_bytes(), "weight_hash": m.weight_hash()}
    rows, keys = [], []
    for case in cases:
        for row in m.build_rows(case["state"], case["questions"]):
            rows.append(row)
            keys.append(f"{case['name']}/{row['qid']}")
    answers, _, _ = m.predict_rows(rows)
    batched = dict(zip(keys, answers))
    single = {}
    for case in cases:
        crows = m.build_rows(case["state"], case["questions"])
        cans, _, _ = m.predict_rows(crows)
        for r, a in zip(crows, cans):
            single[f"{case['name']}/{r['qid']}"] = a
    m.unload()
    if where is not None:
        where.release()
    free()
    return batched, single, info


def routing_check() -> list[str]:
    """The router sends every case where arbiter's did (explicit model names are explicit)."""
    bad = []
    for case in CASES:
        got = case.get("model") or laya_router.route(case["state"], case["questions"])["model"]
        if got != case["expect_route"]:
            bad.append(f"{case['name']}: routed to {got}, arbiter routed to {case['expect_route']}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/.qwen38-spark-engine/deciders/laya"))
    ap.add_argument("--sdk", default=None, help="a directory holding the Laya SDK's `laya` package")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reference-device", default=None, help="where laya.Agent runs (--device)")
    ap.add_argument("--models", default=",".join(NAMES))
    ap.add_argument("--json", default=None, help="write the results here")
    a = ap.parse_args()
    if a.sdk:
        sys.path.insert(0, os.path.expanduser(a.sdk))
    import torch
    import transformers
    import laya
    names = [n.strip() for n in a.models.split(",") if n.strip()]
    ref_dev = a.reference_device or a.device
    print(f"device {a.device}, reference {ref_dev}; torch {torch.__version__}, transformers "
          f"{transformers.__version__}, laya SDK {laya.__version__} ({laya.__file__})\n", flush=True)
    lines = ["| checkpoint | path | questions | max abs delta p vs SDK | argmax agreement |",
             "|-|-|-:|-:|-:|"]
    result = {"device": a.device, "reference_device": ref_dev, "torch": torch.__version__,
              "transformers": transformers.__version__, "sdk": laya.__version__,
              "checkpoints": {}, "routing": routing_check()}
    ok = not result["routing"]
    rows_ok = True
    for name in names:
        cases = [c for c in CASES if c["expect_route"] == name]
        if not cases:
            continue
        nq = sum(len(c["questions"]) for c in cases)
        print(f"== {name} ({len(cases)} cases, {nq} questions)", flush=True)
        ref = run_reference(name, a.dir, ref_dev, cases)
        batched, single, info = run_engine(name, a.dir, a.device, cases)
        result["checkpoints"][name] = {"engine": info, "paths": {}}
        for label, got in (("per call", single), ("collated", batched)):
            w, agree, total, moved = compare(ref, got)
            result["checkpoints"][name]["paths"][label] = {
                "max_abs_dp": w, "argmax": f"{agree}/{total}", "moved": moved}
            if label == "per call":
                ok = ok and w == 0.0 and agree == total
            else:
                rows_ok = rows_ok and w < 5e-3 and agree == total
            lines.append(f"| {name} | {label} | {total} | {w:.2e} | {agree}/{total} |")
            print(f"   {label:12s} max|dp|={w:.2e} argmax {agree}/{total} {moved or ''}  "
                  f"({info['precision']}, load {info['load_s']} s)", flush=True)
    result["gate"] = "PASS" if ok else "FAIL"
    result["collated"] = "within 5e-3, argmax unchanged" if rows_ok else "OUTSIDE 5e-3 or argmax moved"
    print("\n" + "\n".join(lines))
    if result["routing"]:
        print("\nrouting: " + "; ".join(result["routing"]))
    print(f"\nstage A gate, per call (0.00e+00 and every argmax unchanged, routing as arbiter's): "
          f"{result['gate']}\ncollate=rows (arbiter's gate, under 5e-3 and every argmax unchanged): "
          f"{result['collated']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
