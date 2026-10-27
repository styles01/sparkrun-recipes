"""Where a Flash-Next decode step spends its time: GPU kernel time by kernel, from graph replays.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_profile.py --pack ~/fn2/packs/3.05 --ctx 1024 --steps 16

Loads the pack, prefills `--ctx` tokens of held-out prose, captures the decode step and replays it
`--steps` times under torch.profiler (CUDA activity). Prints the kernels summed over the steps,
divided by the steps (ms a token), grouped by family, next to the wall time of a replay; and the
per-part byte budget of the pack (`PackIndex.decode_bytes`) for the GB/s each family reaches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402

FAMILIES = (("moe_gu", "moe gate|up"), ("moe_down", "moe down"), ("moe_combine", "moe combine"),
            ("router_kernel", "router"), ("mix_up", "mixer up"), ("gemv_kernel", "gemv (mixer down, router, a/b, ple)"),
            ("gemm_kernel", "exl3 gemm"), ("rot_in", "exl3 rot_in"), ("rot_out", "exl3 rot_out"),
            ("_attn", "attention"), ("fused_decode", "gdn recurrence"), ("_gdn", "gdn pre"),
            ("index", "index ops"), ("elementwise", "elementwise"), ("reduce", "reductions"),
            ("sort", "sort"), ("copy", "copies"))


def family(name: str) -> str:
    n = name.lower()
    for key, fam in FAMILIES:
        if key.lower() in n:
            return fam
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--json")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from tools.fn_generate import class_prompt
    idx, W, banks, tables, _ = load(a.pack, "cuda", first=a.first, log=None)
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=4096)
    tok = AutoTokenizer.from_pretrained(a.pack)
    ids = class_prompt(tok, "prose", a.ctx)
    st = eng.new_state(4096)
    logits = eng.prefill(torch.tensor(ids), st)
    g = DecodeGraph(eng, st)
    g.capture()
    t = int(logits[0].argmax())
    for _ in range(4):
        t = g.greedy(t)
    torch.cuda.synchronize()
    # wall time of a replay alone (no staging) and of a whole step (staging + replay + argmax)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    for _ in range(a.steps):
        g.graph.replay()
    ev[1].record()
    torch.cuda.synchronize()
    replay_ms = ev[0].elapsed_time(ev[1]) / a.steps
    t0 = time.time()
    for _ in range(a.steps):
        t = g.greedy(t)
    step_ms = (time.time() - t0) * 1e3 / a.steps
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(a.steps):
            g.graph.replay()
        torch.cuda.synchronize()
    # critical-path time: with programmatic dependent launch a kernel starts before its predecessor
    # ends and waits; charge each kernel only from max(its start, the predecessor's end) to its end
    kev = sorted(((e.time_range.start, e.time_range.end, e.name) for e in prof.events()
                  if e.device_type.name == "CUDA"), key=lambda t: t[0])
    eff = defaultdict(float)
    eff_k = defaultdict(float)
    prev_end = None
    for t0, t1, name in kev:
        start = t0 if prev_end is None else max(t0, prev_end)
        if t1 > start:
            eff[family(name)] += (t1 - start) / 1e3 / a.steps
            eff_k[name] += (t1 - start) / 1e3 / a.steps
        prev_end = t1 if prev_end is None else max(prev_end, t1)
    per = defaultdict(float)
    cnt = defaultdict(int)
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0):
            us = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
            per[e.key] += us / 1e3 / a.steps
            cnt[e.key] += e.count // a.steps
    fam = defaultdict(float)
    for k, v in per.items():
        fam[family(k)] += v
    kernel_ms = sum(per.values())
    budget = idx.decode_bytes(1, context=a.ctx)
    out = {"ctx": len(ids), "replay_ms": round(replay_ms, 3), "step_ms": round(step_ms, 3),
           "kernel_ms": round(kernel_ms, 3), "launches": sum(cnt.values()),
           "families_ms": {k: round(v, 3) for k, v in sorted(fam.items(), key=lambda x: -x[1])},
           "critical_path_ms": {k: round(v, 3) for k, v in sorted(eff.items(), key=lambda x: -x[1])},
           "critical_path_total_ms": round(sum(eff.values()), 3),
           "critical_path_top": [(k[:90], round(v, 4)) for k, v in sorted(eff_k.items(), key=lambda x: -x[1])[:a.top]],
           "top": [(k[:90], round(v, 4), cnt[k]) for k, v in sorted(per.items(), key=lambda x: -x[1])[:a.top]],
           "bytes": budget["parts"], "bytes_total": budget["total"],
           "ceiling_ms_at_273": round(budget["total"] / 273e9 * 1e3, 3)}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
