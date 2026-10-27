"""The fast top-p filter (`engine/sample.py` TOPP_K) against the sort, on real target rows.

Teacher-forced forwards over held-out text (prose, code, calibration) give every position's logits;
for each (temperature, top_p) in the grid both filters run on the same rows and the tool counts rows
whose kept set differs and rows whose final probabilities are not bit-identical, plus the timing of
both filters on a 24-row block (the verify's) and a 16-row one (the drafter's).

Usage: sample_filter_check.py --tokens 4096 --k 1024 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import sample as S  # noqa: E402
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from tools.prefill_window_check import prompt_ids  # noqa: E402


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--grid", default="0.6:0.95,0.7:0.95,0.7:0.8,1.0:0.95,1.0:0.9")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    cfg = load_config(None)
    w = Weights(cfg.path, device="cuda", skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=a.tokens + 16, device="cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    ids = prompt_ids(tok, repo, a.tokens)
    rows = []
    eng.reset()
    with torch.no_grad():
        for i in range(0, len(ids), 512):
            blk = torch.tensor(ids[i:i + 512], device="cuda")
            lg = eng.forward(blk, start=i, last_only=False)[0]
            rows.append(lg.float()[::4].clone())        # every 4th position, ~25 % of the text
            del lg
    L = torch.cat(rows)
    del rows
    out = {"rows": int(L.shape[0]), "k": a.k, "grid": {}}
    for spec in a.grid.split(","):
        t, p = (float(x) for x in spec.split(":"))
        smp = S.Sampler(t, p)
        diff_set = diff_bits = wide = 0
        for i in range(0, L.shape[0], 64):
            x = L[i:i + 64]
            S.TOPP_K = 0
            ref = smp.probs_rows(x)
            S.TOPP_K = a.k
            got = smp.probs_rows(x)
            ks, kg = ref > 0, got > 0
            diff_set += int((ks != kg).any(-1).sum())
            diff_bits += int((ref != got).any(-1).sum())
            lt = x / t
            lse = torch.logsumexp(lt, -1, keepdim=True)
            mass = torch.exp(torch.topk(lt, min(a.k, lt.shape[-1]), -1).values - lse).sum(-1)
            wide += int((mass <= p).sum())
        S.TOPP_K = 0
        blk24, blk16 = L[:24], L[24:40]
        r = {"rows_set_differs": diff_set, "rows_bits_differ": diff_bits, "rows_wider_than_k": wide,
             "ms_sort_24": timed(lambda: smp.probs_rows(blk24)),
             "ms_sort_16": timed(lambda: smp.probs_rows(blk16))}
        S.TOPP_K = a.k
        r["ms_fast_24"] = timed(lambda: smp.probs_rows(blk24))
        r["ms_fast_16"] = timed(lambda: smp.probs_rows(blk16))
        S.TOPP_K = 0
        out["grid"][spec] = r
        print(f"[filter] T {t} top_p {p}: {json.dumps(r)}", flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
