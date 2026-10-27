"""A/B of Flash-Next decode switches in one process (one load, same prompts, alternated).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_ab.py --pack ~/fn2/packs/3.05 \
        --variants base,nodense,nopdl --rounds 2 --new 128 --json ab.json

Variants (switches the engine reads per call, so no reload): `base` = the build's defaults;
`dense` / `nodense` = EXL3 projections of up to 32 rows on tools/fn_kernels.dense or on
tools/exl3_linear's kernel (engine.DENSE_MAX); `nosplit` = few-row GEMVs on the one-warp-a-row kernel
(fn_set_gemv_split(0)); `nopdl` =
Flash-Next kernels launched without programmatic dependent launch (fn_set_pdl(0)). Every round
runs every variant in order, then the next round in reverse order (A B C, C B A, ...), each on the
prose, code and chat prompts at `--ctx` (tools/fn_generate.py's), decode tok/s through a freshly
captured DecodeGraph, greedy. The texts of a class must be identical across variants that do not
change arithmetic (`nopdl`); `nodense` changes the projections' summation order and may differ.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext import engine as E  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools import fn_kernels as K  # noqa: E402
from tools.fn_generate import class_prompt, greedy  # noqa: E402


BASE_DENSE = E.DENSE_MAX


def apply(variant: str) -> None:
    E.DENSE_MAX = {"dense": 32, "nodense": 0}.get(variant, BASE_DENSE)
    K._module().fn_set_pdl(0 if variant == "nopdl" else 1)
    K._module().fn_set_gemv_split(0 if variant == "nosplit" else int(E._S.get("FN_GEMV_SPLIT") == "1"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--variants", default="base,nodense,nopdl")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--classes", default="prose,code,chat")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    idx, W, banks, tables, timing = load(a.pack, "cuda")
    eng = E.FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=4096)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    prompts = {c: class_prompt(tok, c, a.ctx) for c in a.classes.split(",")}
    variants = a.variants.split(",")
    rows = []
    texts: dict = {}
    for r in range(a.rounds):
        order = variants if r % 2 == 0 else variants[::-1]
        for v in order:
            apply(v)
            for c, ids in prompts.items():
                out, m = greedy(eng, ids, a.new, eos)
                texts.setdefault((c, v), out)
                rows.append({"round": r, "variant": v, "class": c, "decode_tok_s": m["decode_tok_s"],
                             "ms_per_token": m["ms_per_token"], "same_text_as_round0": out == texts[(c, v)]})
                print(json.dumps(rows[-1]), flush=True)
    apply("base")
    summ = {}
    for v in variants:
        for c in prompts:
            vals = [x["decode_tok_s"] for x in rows if x["variant"] == v and x["class"] == c]
            summ[f"{v}/{c}"] = round(statistics.mean(vals), 2)
    same = {f"{c}": all(texts[(c, v)] == texts[(c, "base")] for v in variants if v == "nopdl")
            for c in prompts if (c, "base") in texts}
    res = {"knobs": {"ctx": a.ctx, "new": a.new, "rounds": a.rounds, "greedy": True, "thinking": False},
           "rows": rows, "mean_tok_s": summ, "same_text": same, "load": timing}
    print(json.dumps({"mean_tok_s": summ, "same_text": same}), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
