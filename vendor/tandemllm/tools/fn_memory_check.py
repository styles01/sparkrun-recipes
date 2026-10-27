"""ENG-239 / SRV-335 on the box: what Flash-Next holds of the machine's memory, under each table policy.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_memory_check.py --pack ~/fn2/packs/3.05 --json out.json

One process, memory measurements only (no sweeps): after the load (policy warm, the table warmed),
MemAvailable, the allocator and the table's page-cache bytes (cachestat); a prefill of 8k and of 32k
tokens with the allocator's peak above what it held before (the prefill workspace, per context
token); greedy text at 1k and 8k (`--new` tokens through the decode graph) and one single-stream
decode rate at 1k. Then the policy switches to cold (`QWEN38_FN_TABLE=cold` at load does the same):
the table's page cache is dropped, the same greedy texts are decoded again and must be equal token
for token, the table's resident bytes after a 1k prefill and the decode are reported (0 expected),
the hot-row cache's counts, and the cold decode rate. The MemAvailable guard of the session runner
applies; nothing here runs two models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402

GIB = float(1 << 30)


def table_of(eng):
    return next(iter(eng.tables.values()))


@torch.no_grad()
def prefill_peak(eng, ids: list[int]) -> dict:
    st = eng.new_state(eng.max_len)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    eng.prefill(torch.tensor(ids), st)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    out = {"tokens": len(ids), "s": round(time.perf_counter() - t0, 2), "workspace_bytes": int(peak - base),
           "workspace_gib": round((peak - base) / GIB, 3), "mem_available_gib": mem_available_gib()}
    del st
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def greedy(eng, ids: list[int], new: int, eos: set) -> tuple[list[int], float]:
    st = eng.new_state(eng.max_len)
    first = int(eng.prefill(torch.tensor(ids), st)[0].argmax())
    g = DecodeGraph(eng, st)
    g.capture()
    out = [first]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while len(out) < new and out[-1] not in eos:
        out.append(g.greedy(out[-1]))
    rate = (len(out) - 1) / (time.perf_counter() - t0)
    del g, st
    torch.cuda.empty_cache()
    return out, round(rate, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=33_000)
    ap.add_argument("--settle-s", type=float, default=60.0, help="wait for the table's warm-up to land")
    ap.add_argument("--cold-only", action="store_true", help="a process loaded with QWEN38_FN_TABLE=cold: the "
                    "table's page cache after a 1k prefill and the decode (0 expected), the cold decode rate")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    res = {"pack": a.pack, "mem_before_gib": mem_available_gib()}
    idx, W, banks, tables, timing = load(a.pack, "cuda", mtp=False)
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=a.max_len)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    tab = table_of(eng)
    if a.cold_only:
        assert tab.policy == "cold", "start it with QWEN38_FN_TABLE=cold"
        out, rate = greedy(eng, class_prompt(tok, "prose", 1024), a.new, eos)
        res.update(table=tab.report(), tokens=out, decode_tok_s=rate, mem_available_gib=mem_available_gib())
        print(f"[mem] cold-only: {len(out)} tokens, {rate} tok/s, table {json.dumps(res['table'])}", flush=True)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
        return
    time.sleep(a.settle_s)
    res["load"] = {"timing": timing, "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 2),
                   "mem_available_gib": mem_available_gib(), "table": tab.report()}
    print(f"[mem] after load: {json.dumps(res['load']['table'])}, MemAvailable {res['load']['mem_available_gib']} GiB",
          flush=True)
    res["prefill"] = {}
    for n in (8192, 32768):
        res["prefill"][n] = r = prefill_peak(eng, class_prompt(tok, "prose", n))
        print(f"[mem] prefill {n}: {r}", flush=True)
    prompts = {n: class_prompt(tok, "prose", n) for n in (1024, 8192)}
    res["warm"] = {}
    for n, ids in prompts.items():
        out, rate = greedy(eng, ids, a.new, eos)
        res["warm"][n] = {"tokens": out, "decode_tok_s": rate}
        print(f"[mem] warm {n}: {len(out)} tokens, {rate} tok/s", flush=True)
    res["warm_table"] = tab.report()
    tab.set_policy("cold")
    res["cold_after_switch"] = tab.report()
    res["cold"] = {}
    for n, ids in prompts.items():
        out, rate = greedy(eng, ids, a.new, eos)
        res["cold"][n] = {"tokens": out, "decode_tok_s": rate, "equal_warm": out == res["warm"][n]["tokens"]}
        print(f"[mem] cold {n}: {len(out)} tokens, {rate} tok/s, equal to warm {out == res['warm'][n]['tokens']}",
              flush=True)
    res["cold_table"] = tab.report()
    res["mem_end_gib"] = mem_available_gib()
    w = res["prefill"]
    res["workspace_per_token"] = int((w[32768]["workspace_bytes"] - w[8192]["workspace_bytes"]) / (32768 - 8192))
    res["workspace_floor"] = int(w[8192]["workspace_bytes"] - res["workspace_per_token"] * 8192)
    res["all_equal"] = all(r["equal_warm"] for r in res["cold"].values())
    print(f"[mem] workspace {res['workspace_floor']} + {res['workspace_per_token']} a token; cold table "
          f"{json.dumps(res['cold_table'])}; all equal {res['all_equal']}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
