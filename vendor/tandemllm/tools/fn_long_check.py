"""ENG-147 on the board: decode past 2,051 tokens on the kernels, against the reference forward.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_long_check.py --pack ~/fn2/packs/3.05 \
        --ctx 3072,8192,32768 --classes prose,code --steps 32 --new 256 --json out.json

Per context and class (tools/fn_generate.py's prompts, filled to `--ctx` tokens): the prompt is
prefilled once and the state cloned. The engine decodes `--steps` tokens through its decode graph
(QSA on the kernels: pooled keys kept by every step, per-row block scores, a stable top-512, attention
over the selected keys); the reference forward (engine/models/flashnext/ref.py's attention with its
QSA mask, the reference GDN and PLE, on the same weight kernels) is fed the same tokens on the clone.
Recorded per step: the argmax agrees, max |logit diff|, and on a disagreement the reference's own
top-2 margin (a near tie or not). Then plain greedy decode of `--new` tokens from a fresh prefill:
tok/s and p50 (single stream).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402


@torch.no_grad()
def check(eng: FlashNextEngine, ids: list[int], steps: int) -> dict:
    st = eng.new_state(eng.max_len)
    t0 = time.time()
    tok = int(eng.prefill(torch.tensor(ids), st)[0].argmax())
    prefill_s = time.time() - t0
    ref = st.clone()
    g = DecodeGraph(eng, st)
    agree, worst, near = 0, 0.0, []
    for i in range(steps):
        x = g(tok).clone()
        y = eng.forward(torch.tensor([tok], device=eng.device), ref)[-1:].float()
        a, b = int(x.argmax()), int(y.argmax())
        agree += int(a == b)
        worst = max(worst, (x - y).abs().max().item())
        if a != b:
            top = torch.topk(y[0], 2).values
            near.append({"step": i, "ref_margin": round((top[0] - top[1]).item(), 4)})
        tok = a
    return {"prompt_tokens": len(ids), "prefill_s": round(prefill_s, 1), "steps": steps, "argmax_agree": agree,
            "max_abs_diff": round(worst, 4), "disagreements": near}


@torch.no_grad()
def plain(eng: FlashNextEngine, ids: list[int], new: int, eos: set) -> dict:
    st = eng.new_state(eng.max_len)
    out = [int(eng.prefill(torch.tensor(ids), st)[0].argmax())]
    g = DecodeGraph(eng, st)
    g._ready()
    eng.host_ms = {}
    per = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while len(out) < new and out[-1] not in eos:
        ts = time.perf_counter()
        out.append(g.greedy(out[-1]))
        per.append(time.perf_counter() - ts)
    dt = time.perf_counter() - t0
    return {"tokens": len(out), "tok_s": round((len(out) - 1) / dt, 2), "ms_p50": round(1e3 * statistics.median(per), 2),
            "stage_ms_a_token": round(eng.host_ms.get("stage", 0.0) / max(len(out) - 1, 1), 3),
            "graph_classes": sorted(g.graphs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", default="3072,8192")
    ap.add_argument("--classes", default="prose,code")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    ctxs = [int(x) for x in a.ctx.split(",")]
    idx, W, banks, tables, timing = load(a.pack, "cuda")
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max(ctxs) + a.new + 64)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    res = {"pack": a.pack, "load": timing, "mem_after_load_gib": mem_available_gib(),
           "knobs": {"ctx": ctxs, "steps": a.steps, "new": a.new, "max_len": eng.max_len, "greedy": True,
                     "thinking": False, "kv_dtype": "bf16", "qsa": "budget 2048, ratio 4, top 512 (kernels)"},
           "rows": []}
    for n in ctxs:
        for c in a.classes.split(","):
            ids = class_prompt(tok, c, n)
            r = {"ctx": n, "class": c, "check": check(eng, ids, a.steps)}
            r["plain"] = plain(eng, ids, a.new, eos)
            r["mem_gib"] = mem_available_gib()
            res["rows"].append(r)
            print(f"[long] {json.dumps(r)}", flush=True)
            torch.cuda.empty_cache()
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
