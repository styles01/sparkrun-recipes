"""The verify's cost past sixteen rows, in the served configuration, and which kernel pays for it.

on 2026-09-23 a verify of 17 rows cost ~15 ms more than 16, and (a 24- or 32-node
tree) is blocked on it. `tools/verify_tree.py --curve` measured the curve before the graphs and the
fold; this measures it as the loop runs today -- `forward_tree` then `commit_tree`, the pending
commit applied by the next verify, graphs where the engine takes them -- for three shapes at each
row count:

    chain    a line (the deep chain's shape,; a chain-shaped tree delegates to forward_block)
    spine    a 15-deep line with the rest of the nodes hung off it as alternatives, near the top --
             what `lattice_tree` builds from a 16-slot lattice at a larger budget
    bushy    random parents, at most three children a node (verify_tree's `random_tree`)

and then, with the graphs off so every kernel is visible, the device time by kernel name at 16 rows
against each wider count: the kernels whose time jumps are the cliff.

    python tools/verify_curve.py --rows 8,16,17,20,24,28,32 --out results/spd41/curve.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.tree import DraftTree, TreeBuilder  # noqa: E402


def spine_tree(anchor: int, n: int, rng: random.Random, depth: int = 15) -> DraftTree:
    """`n` drafted nodes: a line of min(n, depth), the rest as alternatives at depths 1, 2, 3, ..."""
    b = TreeBuilder(anchor)
    line = min(n, depth)
    parent = 0
    for i in range(line):
        parent = b.add(parent, rng.randrange(1000, 200000), 0.9 ** i, "spine")
    spine = [0] + list(range(1, line + 1))
    for j in range(n - line):
        at = spine[j % max(1, line)]
        b.add(at, rng.randrange(1000, 200000), 0.5 ** (j + 1), "alt")
    return b.build()


def bushy_tree(anchor: int, n: int, rng: random.Random) -> DraftTree:
    from tools.verify_tree import random_tree
    return random_tree(anchor, rng, n, 200000, branch=3)


def make(shape: str, rows: int, rng: random.Random) -> DraftTree:
    n = rows - 1
    if shape == "chain":
        return DraftTree.chain(1000, [rng.randrange(1000, 200000) for _ in range(n)])
    return spine_tree(1000, n, rng) if shape == "spine" else bushy_tree(1000, n, rng)


def accepted_path(t: DraftTree, rng: random.Random) -> list[int]:
    """A partial accept: the anchor and a few nodes down one branch, as a block usually ends."""
    p = t.path(t.leaves()[0])
    return p[:min(len(p), 1 + rng.randrange(1, 4))]


def time_pair(eng, t: DraftTree, start: int, reps: int, rng: random.Random) -> float:
    toks = torch.tensor(t.tokens, device=eng.device)
    path = accepted_path(t, rng)
    for _ in range(3):
        eng.forward_tree(toks, t.parents, start=start)
        eng.commit_tree(path)
        eng.kv.length = start
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        eng.forward_tree(toks, t.parents, start=start)
        eng.commit_tree(path)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
        eng.kv.length = start
    return statistics.median(ts)


def kernels(eng, t: DraftTree, start: int, reps: int) -> dict[str, float]:
    """Device ms a verify+commit, by kernel name, graphs off."""
    from torch.profiler import ProfilerActivity, profile
    toks = torch.tensor(t.tokens, device=eng.device)
    path = t.path(t.leaves()[0])[:2]
    for _ in range(2):
        eng.forward_tree(toks, t.parents, start=start)
        eng.commit_tree(path)
        eng.kv.length = start
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            eng.forward_tree(toks, t.parents, start=start)
            eng.commit_tree(path)
            eng.kv.length = start
        torch.cuda.synchronize()
    out: dict[str, float] = defaultdict(float)
    for ev in prof.key_averages():
        us = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
        if us:
            out[ev.key[:70]] += us / 1e3 / reps
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--rows", default="8,16,17,20,24,28,32")
    ap.add_argument("--shapes", default="chain,spine,bushy")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--prompt", type=int, default=300, help="context tokens before the verify")
    ap.add_argument("--kernels", default="16,17,24,32", help="row counts to itemise ('' = none)")
    ap.add_argument("--out", default="")
    ap.add_argument("--contexts", default="",
                    help="several context lengths in one load (e.g. 1024,4096,8192,16384,32768); "
                         "with --stair-out the per-class staircase (chain, tree = the spine shape) "
                         "the lenrouter reads as QWEN38_STAIR_TABLES")
    ap.add_argument("--stair-out", default="")
    a = ap.parse_args()

    import engine.model as M
    cfg = load_config(a.model)
    eng = Qwen38Engine(cfg, Weights(cfg.path, device="cuda", skip_mtp=True), max_len=a.max_len,
                       device="cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    text = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "bench", "calib.txt")).read()
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    ctxs = [int(c) for c in a.contexts.split(",") if c] or [a.prompt]
    stair: dict = {}
    for ctx in ctxs:
        a.prompt = ctx
        res = _one(a, eng, ids_all, M)
        stair[str(ctx)] = {"chain": {str(k): v for k, v in res["ms"].get("chain", {}).items()},
                           "tree": {str(k): v for k, v in res["ms"].get("spine", {}).items()}}
        if a.out and len(ctxs) > 1:
            root, ext = os.path.splitext(a.out)
            json.dump(res, open(f"{root}-{ctx}{ext or '.json'}", "w"), indent=1)
        elif a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
            print(f"[curve] {a.out}")
        a.kernels = ""                            # the itemisation once, at the first context
    if a.stair_out:
        json.dump(stair, open(a.stair_out, "w"), indent=1)
        print(f"[curve] staircase -> {a.stair_out}")


def _one(a, eng, ids_all, M) -> dict:
    ids = ids_all
    while ids.numel() < a.prompt:                 # a long context repeats the text
        ids = torch.cat([ids, ids])
    ids = ids[:a.prompt].to("cuda")
    with torch.no_grad():
        eng.reset()
        eng.forward(ids, start=0, last_only=True)
        start = ids.numel()
        rows = [int(r) for r in a.rows.split(",")]
        shapes = a.shapes.split(",")
        rng = random.Random(7)
        res: dict = {"rows": rows, "shapes": shapes, "graph": bool(M.VERIFY_GRAPH),
                     "verify_rows": M.VERIFY_ROWS,
                     "fold": bool(M.COMMIT_IN_VERIFY), "ms": {}, "kernels": {}}
        print(f"verify + commit, ms (median of {a.reps}), graphs {'on' if M.VERIFY_GRAPH else 'off'}"
              f", fold {'on' if M.COMMIT_IN_VERIFY else 'off'}, VERIFY_ROWS {M.VERIFY_ROWS}, "
              f"context {start}")
        print(f"{'rows':>5} " + " ".join(f"{s:>16}" for s in shapes))
        for T in rows:
            line = []
            for s in shapes:
                t = make(s, T, rng)
                g0 = dict(eng._graphs.stats) if eng._graphs is not None else {}
                ms = time_pair(eng, t, start, a.reps, rng)
                g1 = dict(eng._graphs.stats) if eng._graphs is not None else {}
                graphed = g1.get("replayed", 0) > g0.get("replayed", 0)
                res["ms"].setdefault(s, {})[T] = ms
                res.setdefault("graphed", {}).setdefault(s, {})[T] = graphed
                base = res["ms"][s].get(16)
                line.append(f"{ms:7.2f}{'g' if graphed else ' '}"
                            + (f" ({ms / base:4.2f}x)" if base else "        "))
            print(f"{T:5d} " + " ".join(f"{x:>16}" for x in line), flush=True)
        print("(g = served from a verify graph)")
        if a.kernels:
            M.VERIFY_GRAPH = False
            krows = [int(r) for r in a.kernels.split(",")]
            for s in ("spine", "chain"):
                if s not in shapes:
                    continue
                per = {T: kernels(eng, make(s, T, rng), start, 4) for T in krows}
                res["kernels"][s] = per
                names = sorted(set().union(*per.values()),
                               key=lambda k: -max(per[T].get(k, 0.0) - per[krows[0]].get(k, 0.0)
                                                  for T in krows))
                print(f"\n{s}: device ms by kernel, graphs off (sorted by the largest rise over "
                      f"{krows[0]} rows)")
                print(f"{'kernel':<70} " + " ".join(f"{T:>8d}" for T in krows))
                tot = {T: sum(per[T].values()) for T in krows}
                print(f"{'TOTAL':<70} " + " ".join(f"{tot[T]:8.2f}" for T in krows))
                for k in names[:24]:
                    print(f"{k:<70} " + " ".join(f"{per[T].get(k, 0.0):8.2f}" for T in krows))
            M.VERIFY_GRAPH = True if res["graph"] else M.VERIFY_GRAPH
    return res


if __name__ == "__main__":
    main()
