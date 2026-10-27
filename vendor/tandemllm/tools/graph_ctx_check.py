"""Verify graphs past 32k of context (SPD-95): is the graphed verify at a context class above
`QWEN38_GRAPH_MAX_CTX`'s old 32,768 bit-identical to the eager verify there, and how much faster.

Run with QWEN38_GRAPH_MAX_CTX raised (e.g. 262144). For every context in --contexts the prompt is
prefilled (`cache.prefill`, 8,192-row chunks), the engine state is captured, and for every row count
and shape a verify runs eager (`graphs_off`), then graphed (capture, then replays), each from the
same captured state: logits and the trace's factors are compared bit for bit, and the median wall of
--reps eager and graphed verifies is printed.

Usage: QWEN38_GRAPH_MAX_CTX=262144 graph_ctx_check.py --contexts 30000,50000,70000,100000
"""

from __future__ import annotations

import argparse
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


def verify(eng, toks, parents, start, snap):
    cache.restore(eng, snap)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lg = eng.forward_tree(toks, parents, start=start)
    torch.cuda.synchronize()
    return lg.float().clone(), (time.perf_counter() - t0) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="30000,50000,70000")
    ap.add_argument("--rows", default="16,24,32")
    ap.add_argument("--shapes", default="spine,chain")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    from engine import verify_graph as VG
    ctxs = [int(c) for c in a.contexts.split(",") if c]
    cfg = load_config(None)
    eng = Qwen38Engine(cfg, Weights(cfg.path, device="cuda", skip_mtp=True),
                       max_len=max(ctxs) + 512, device="cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    text = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "bench", "calib.txt")).read()
    ids_all = tok(text).input_ids
    out = {"graph_max_ctx": VG.MAX_CTX, "rows": []}
    rng = random.Random(7)
    for L in ctxs:
        ids = (ids_all * (L // len(ids_all) + 1))[:L]
        with torch.no_grad():
            eng.reset()
            cache.prefill(eng, None, ids, "cuda", chunk=8192)
            snap = cache.capture(eng)
            for T in (int(r) for r in a.rows.split(",")):
                for shape in a.shapes.split(","):
                    t = make(shape, T, rng)
                    toks = torch.tensor(t.tokens, device="cuda")
                    eng.graphs_off = True
                    ref, _ = verify(eng, toks, t.parents, L, snap)
                    te = [verify(eng, toks, t.parents, L, snap)[1] for _ in range(a.reps)]
                    eng.graphs_off = False
                    g0 = dict(eng._graphs.stats) if eng._graphs is not None else {}
                    got = [verify(eng, toks, t.parents, L, snap) for _ in range(a.reps + 2)]
                    g1 = dict(eng._graphs.stats) if eng._graphs is not None else {}
                    replayed = g1.get("replayed", 0) - g0.get("replayed", 0)
                    same = all(torch.equal(ref, x) for x, _ in got)
                    row = {"ctx": L, "rows": T, "shape": shape, "identical": same,
                           "replays": replayed,
                           "eager_ms": round(statistics.median(te), 2),
                           "graph_ms": round(statistics.median(x for _, x in got[2:]), 2)}
                    out["rows"].append(row)
                    print(f"[graphctx] {json.dumps(row)}", flush=True)
            del snap
            torch.cuda.empty_cache()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
