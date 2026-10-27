"""One verify, two settings of a kernel flag: are the logits (and the trace's factors) bit-identical,
and what does each cost. The engine is prefilled to --ctx tokens of calibration text, its state
captured, and every (rows, shape) verify runs eager from that state with the flag off and on.

Usage: verify_flag_check.py --flag tools.gdn_verify_kernels.TREE_PF --rows 16,24,32 --ctx 1024
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache  # noqa: E402
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from tools.verify_curve import make  # noqa: E402


def run(eng, toks, parents, start, snap):
    cache.restore(eng, snap)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lg = eng.forward_tree(toks, parents, start=start)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1e3
    fac = []
    tr = getattr(eng, "trace", None)
    if tr is not None and getattr(tr, "factors", None):
        for f in tr.factors.values() if isinstance(tr.factors, dict) else tr.factors:
            if f is not None:
                fac.extend(x.clone() for x in f if torch.is_tensor(x))
    return lg.float().clone(), fac, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", required=True, help="module.ATTR (bool)")
    ap.add_argument("--rows", default="16,24,32")
    ap.add_argument("--shapes", default="spine,bushy")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    modname, attr = a.flag.rsplit(".", 1)
    mod = importlib.import_module(modname)
    cfg = load_config(None)
    eng = Qwen38Engine(cfg, Weights(cfg.path, device="cuda", skip_mtp=True), max_len=a.ctx + 256,
                       device="cuda")
    eng.graphs_off = True
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    text = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "bench", "calib.txt")).read()
    ids = tok(text).input_ids
    ids = (ids * (a.ctx // max(1, len(ids)) + 1))[:a.ctx]      # repeated to reach the context
    rng = random.Random(3)
    out = []
    with torch.no_grad():
        eng.reset()
        cache.prefill(eng, None, ids, "cuda", chunk=8192)
        snap = cache.capture(eng)
        for T in (int(r) for r in a.rows.split(",")):
            for shape in a.shapes.split(","):
                t = make(shape, T, rng)
                toks = torch.tensor(t.tokens, device="cuda")
                res = {}
                for val in (False, True):
                    setattr(mod, attr, val)
                    r0 = run(eng, toks, t.parents, len(ids), snap)
                    ts = [run(eng, toks, t.parents, len(ids), snap)[2] for _ in range(a.reps)]
                    res[val] = (r0, statistics.median(ts))
                (l0, f0, _), m0 = res[False]
                (l1, f1, _), m1 = res[True]
                same = torch.equal(l0, l1) and len(f0) == len(f1) and all(
                    torch.equal(x, y) for x, y in zip(f0, f1))
                row = {"rows": T, "shape": shape, "identical": same, "factors": len(f0),
                       "off_ms": round(m0, 2), "on_ms": round(m1, 2)}
                print(f"[flag] {json.dumps(row)}", flush=True)
                out.append(row)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
