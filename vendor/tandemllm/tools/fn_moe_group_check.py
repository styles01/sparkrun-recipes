"""The grouped expert kernels against the per-pair ones, in one process (ENG-241, the joint pass's
largest part): the same rows, ids and bank through both, every pair's output compared bit for bit and
both timed; then the load-time self-check with the grouped kernels (verify rows of every width against
the decode step's single row, on real routing).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_moe_group_check.py --pack ~/flashnext-packs/3.05 \
        --rows 2,5,8,15,24,32 --layers 0,24,47 --json out.json

Rows: each layer's router on random inputs (bf16, the residual stream's scale), and a second set where
the rows repeat (every pair of an expert in one group of up to 8). Timing: CUDA events around each
call, the median of `--reps`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.serving import build  # noqa: E402
from tools import fn_kernels as K  # noqa: E402


def timed(fn, reps: int) -> float:
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--rows", default="2,5,8,15,24,32")
    ap.add_argument("--layers", default="0,24,47")
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    eng, _, _ = build(a.pack, 4096, tok, spec=False, log=lambda s: None)
    fn = eng.fn
    c = fn.cfg
    g = torch.Generator(device="cuda").manual_seed(7)
    res = {"rows": []}
    for li in [int(x) for x in a.layers.split(",")]:
        bank = fn.banks[li]
        for M in [int(x) for x in a.rows.split(",")]:
            for kind in ("random", "repeated"):
                x = (torch.randn(M, c.hidden_size, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
                if kind == "repeated":
                    x = x[:1].repeat(M, 1).contiguous()
                logits = K.gemv(x, fn.W[fn.pre(li) + "mlp.router"])
                ids, _ = K.router(logits, c.num_experts, c.num_experts_per_tok)
                K.MOE_GROUP = False
                y0 = K.moe_experts(x, ids, bank).clone()
                t0 = timed(lambda: K.moe_experts(x, ids, bank), a.reps)
                K.MOE_GROUP = True
                y1 = K.moe_experts(x, ids, bank).clone()
                t1 = timed(lambda: K.moe_experts(x, ids, bank), a.reps)
                K.MOE_GROUP = False
                experts = int(torch.unique(ids).numel())
                r = {"layer": li, "rows": M, "kind": kind, "pairs": int(ids.numel()), "experts": experts,
                     "equal": bool(torch.equal(y0, y1)), "max_diff": float((y0 - y1).abs().max()),
                     "per_pair_ms": round(t0, 3), "grouped_ms": round(t1, 3)}
                print(f"[group] {r}", flush=True)
                res["rows"].append(r)
    K.MOE_GROUP = True
    text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    check = fn.self_check(tok(text, add_special_tokens=False)["input_ids"], log=lambda s: print(s, flush=True))
    res["self_check_grouped"] = check
    print(f"[group] self-check with the grouped kernels: {check}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
