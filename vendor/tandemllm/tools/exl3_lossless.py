"""The lossless check for a weight profile without a release text (ENG-E6, ruling A restated for EXL3):
the reference is the profile's own greedy decode (one token a forward); a speculative run with the
served router may differ only where the reference's top two logits are near-tied.

For every trace in --traces (prompt_ids; the output is re-decoded here so the reference and the
speculative runs share one process and one build) and every variant in --variants
(`name:ENV=VAL,ENV=VAL` applied to the loaded modules between runs, e.g. `wy32:GDNV_WY_MAXT=32`), it
runs the served tree router (`tools/verify_spec.py::tree_router`, the wide arm as LEN_MODE says) for
--new tokens and compares token by token with the reference. A run is `identical`, or diverges first
at token i with the reference's top1-top2 gap g there: `near-tie` when g <= --tie-ulp bf16 ulps of the
top logit (default 11: the measured verify-vs-decode drift of 2026-09-29), else `other`. Text after
the first divergence is a different continuation and is not compared.

Usage: exl3_lossless.py --traces DIR --only a,b --new 256 --variants 'base:;wy32:GDNV_WY_MAXT=32' 
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_greedy, generate_spec_tree  # noqa: E402
from tools.verify_spec import bf16_ulp, tree_router  # noqa: E402

# where each knob lives once imported (the env is read at import)
KNOBS = {
    "GDNV_WY_MAXT": ("tools.gdn_verify_kernels", "WY_MAXT", int),
    "GDNV_WY_FUSED_MAXT": ("tools.gdn_verify_kernels", "WY_FUSED_MAXT", int),
    "EXL3_ROWGRID": ("tools.exl3_linear", "ROWGRID", lambda v: v == "1"),
    "VERIFY_ROWS": ("engine.model", "VERIFY_ROWS", int),
}


def apply(spec: str) -> dict:
    old = {}
    for kv in filter(None, spec.split(",")):
        k, v = kv.split("=", 1)
        mod, attr, conv = KNOBS[k]
        m = sys.modules.get(mod) or __import__(mod, fromlist=["x"])
        old[k] = (m, attr, getattr(m, attr))
        setattr(m, attr, conv(v))
    return old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--variants", default="base:")
    ap.add_argument("--tie-ulp", type=float, default=11.0)
    ap.add_argument("--ckpt8", default="train/ft-b8-v2")
    ap.add_argument("--ckpt16", default="train/ft-b16")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.traces, "*.json")))
    if a.only:
        want = set(a.only.split(","))
        files = [f for f in files if os.path.basename(f)[:-5] in want]
    cfg = load_config(None)
    w = Weights(cfg.path, device="cuda", skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device="cuda")
    ns = types.SimpleNamespace(dflash2_ckpt=a.ckpt8, dflash2_path="greedy", max_len=a.max_len,
                               lenrouter=a.ckpt16, corpus=a.corpus, drop_idle=True, deep=int(
                                   os.environ.get("QWEN38_DEEP", "0")),
                               deep_after=int(os.environ.get("QWEN38_DEEP_AFTER", "2")),
                               len_fixed=0)
    variants = [v.split(":", 1) for v in a.variants.split(";") if v]
    refs = {}
    out: dict = {"tie_ulp": a.tie_ulp, "new": a.new, "variants": {}}
    for f in files:
        tr = json.load(open(f))
        ids = torch.tensor(tr["prompt_ids"], device="cuda")
        base, st = generate_greedy(eng, ids, a.new, record_gaps=True)
        refs[tr["name"]] = (ids, base, st.gaps, st.tops)
    for name, spec in variants:
        old = apply(spec)
        rows = {}
        counts = {"identical": 0, "near-tie": 0, "other": 0}
        lr = tree_router(eng, ns, narrow=False)
        t0 = time.time()
        tok_total = rounds = 0
        for tname, (ids, base, gaps, tops) in refs.items():
            got, s = generate_spec_tree(eng, ids, a.new, lr, 15)
            tok_total += s.tokens
            rounds += s.blocks
            i = next((j for j, (x, y) in enumerate(zip(base, got)) if x != y), None)
            if i is None and len(got) == len(base):
                kind, row = "identical", {"first": None}
            elif i is None:
                kind, row = "identical", {"first": None, "len": [len(base), len(got)]}
            else:
                g = gaps[i] if i < len(gaps) else float("nan")
                u = g / max(bf16_ulp(tops[i]), 1e-30) if i < len(tops) else float("nan")
                kind = "near-tie" if u <= a.tie_ulp else "other"
                row = {"first": i, "gap": round(g, 5), "ulp": round(u, 2), "ref": base[i],
                       "got": got[i]}
            counts[kind] += 1
            row["kind"] = kind
            rows[tname] = row
            print(f"[{name}] {tname:16s} {kind:9s} {json.dumps(row)}", flush=True)
        lr.detach()
        out["variants"][name] = {"spec": spec, "counts": counts, "rows": rows,
                                 "tokens_per_round": tok_total / max(rounds, 1),
                                 "seconds": round(time.time() - t0, 1)}
        print(f"[{name}] {counts}  tokens/round {tok_total / max(rounds, 1):.2f}", flush=True)
        for k, (m, attr, v) in old.items():
            setattr(m, attr, v)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
