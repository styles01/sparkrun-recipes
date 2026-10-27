"""The served tree cut replayed on greedy-text and sampled-text lattices (CPU only).

A sampled request on `--sampled-tree=det` verifies the tree a greedy request would build and walks it
by drawing the target's token at each node. On a trace whose text is the target's own SAMPLED decode
(`tools/regen_traces.py --temperature`), "the longest tree path the recorded continuation follows" is
that walk in distribution, so the replay below prices the det path exactly as it prices greedy on a
greedy trace. What it can then answer:

  * the gap: the served cut's committed tokens a block on greedy text against sampled text, by class;
  * what a tree built for the sampled walk is worth: the same lattices, the cut's path probabilities
    at another softmax temperature (`--temps`), and optionally a flatter slot-0 row (`--slot0-temps`).
    The tree stays a deterministic function of the lattice, so the det walk is exact on it whatever
    temperature built it (no draw looks at the tree).

The cut: `engine/tree.py::lattice_tree` at `--nodes` (the served 31) then `MergedRouter._stair_cut`
with the served snaps (7, 15, 23, 31 nodes), node value = the raw path probability (calibration 1),
priced on `--stair` (what the server prices with) and costed on `--cost-stair` (what the round pays).

    python tools/sampled_tree_replay.py --stair ops/stair-tables-nvfp4.json \
        --cost-stair runs/stair-tables-exl3-4.5.json greedy=lat-g/b16 sampled=lat-s/b16 --temps 1.4,2,3
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.router import MergedRouter, verify_ms  # noqa: E402
from engine.tree import DraftTree, lattice_tree  # noqa: E402
from tools.tree_sweep import accepted, greedy_walk, load  # noqa: E402


def log_softmax_rows(x: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """log_softmax over the last dim with a temperature per slot (x [L, k, k], temps [L])."""
    y = x / temps[:, None, None]
    y = y - y.max(axis=-1, keepdims=True)
    return y - np.log(np.exp(y).sum(axis=-1, keepdims=True))


def cut(tree: DraftTree, table: dict[int, float], draft_ms: float, host_ms: float, budget: int,
        snap: tuple | None) -> DraftTree:
    stub = types.SimpleNamespace(
        stair_table=table, tree_table=table, verify_table=table, node_budget=budget,
        stair_price_fn=None, rollback_ms=0.0, commit_ms=host_ms, stair_factor=None, stair_snap=snap,
        _node_q=lambda n, t, d, r=0: min(1.0, t.scores[n]))
    t, _ = MergedRouter._stair_cut(stub, tree, draft_ms)
    return t if t is not None else DraftTree.chain(tree.tokens[0], [])


def walk(tr: dict, temps: np.ndarray, a, tab, ctab) -> tuple[int, int, float, collections.Counter,
                                                              list]:
    """One trace under one policy: blocks, committed, modelled ms, rows histogram, per-depth
    [reached, accepted] counts (a_i = accepted / reached at depth i of the committed run)."""
    target = tr["target"]
    blocks = committed = 0
    ms = 0.0
    rows = collections.Counter()
    depth = [[0, 0] for _ in range(40)]
    at = 0
    while at < len(target) - 1:
        i = tr["index"].get(at)
        if i is None:
            at += 1
            continue
        cand, scores = tr["cand"][i], tr["scores"][i]
        greedy = greedy_walk(scores)
        logp = log_softmax_rows(scores, temps[:scores.shape[0]]).tolist()
        t = lattice_tree(int(target[at]), cand.tolist(), logp, greedy, a.nodes)
        t = cut(t, tab, a.draft_ms, a.host_ms, a.nodes, a.snap)
        acc = accepted(t, target, at)
        n = len(t.tokens)
        rows[n] += 1
        blocks += 1
        c = min(acc + 1, len(target) - 1 - at)
        committed += c
        ms += verify_ms(n, ctab) + a.draft_ms + a.host_ms
        maxd = max(t.depths()) if n > 1 else 0
        for d in range(min(maxd, 39)):
            if d <= acc:
                depth[d][0] += 1
                if d < acc:
                    depth[d][1] += 1
        at += acc + 1
    return blocks, committed, ms, rows, depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="label=lattice_dir")
    ap.add_argument("--stair", required=True, help="stair tables the cut prices with")
    ap.add_argument("--cost-stair", default="", help="stair tables the round pays (default --stair)")
    ap.add_argument("--ctx", default="1024")
    ap.add_argument("--nodes", type=int, default=31)
    ap.add_argument("--snap", default="7,15,23,31")
    ap.add_argument("--draft-ms", type=float, default=15.0)
    ap.add_argument("--host-ms", type=float, default=3.5)
    ap.add_argument("--temps", default="1.4", help="cut temperatures (all slots)")
    ap.add_argument("--slot0-temps", default="", help="extra arms: 'T0' at slot 0, the --temps[0] rest")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    a.snap = tuple(int(x) for x in a.snap.split(",")) if a.snap else None
    tab = {int(k): float(v) for k, v in json.load(open(a.stair))[a.ctx]["tree"].items()}
    ctab = ({int(k): float(v) for k, v in json.load(open(a.cost_stair))[a.ctx]["tree"].items()}
            if a.cost_stair else tab)
    temps = [float(x) for x in a.temps.split(",")]
    arms = [(f"T{t:g}", t, None) for t in temps]
    base = temps[0]
    arms += [(f"T{base:g}/s0:{float(x):g}", base, float(x)) for x in a.slot0_temps.split(",") if x]
    out = {}
    for spec in a.sets:
        label, d = spec.split("=", 1)
        traces = load(os.path.expanduser(d))
        out[label] = {}
        print(f"\n== {label}: {d} ({len(traces)} traces)")
        print(f"{'arm':14s} {'class':6s} {'blocks':>6s} {'tok/blk':>7s} {'rows':>5s} {'tok/s*':>7s}  "
              f"a_1..a_8 (P(depth i accepted | reached))")
        for name, t0, s0 in arms:
            agg = collections.defaultdict(lambda: [0, 0, 0.0, collections.Counter(),
                                                   [[0, 0] for _ in range(40)]])
            for tr in traces:
                L = tr["scores"].shape[1]
                tv = np.full(L, t0, dtype=np.float32)
                if s0 is not None:
                    tv[0] = s0
                b, c, ms, rows, dep = walk(tr, tv, a, tab, ctab)
                for k in (tr["klass"], "all"):
                    g = agg[k]
                    g[0] += b
                    g[1] += c
                    g[2] += ms
                    g[3].update(rows)
                    for j in range(40):
                        g[4][j][0] += dep[j][0]
                        g[4][j][1] += dep[j][1]
            res = {}
            for k in sorted(agg, key=lambda x: (x == "all", x)):
                b, c, ms, rows, dep = agg[k]
                mr = sum(r * n for r, n in rows.items()) / max(b, 1)
                curve = [dep[j][1] / dep[j][0] if dep[j][0] else 0.0 for j in range(8)]
                print(f"{name:14s} {k:6s} {b:6d} {c / max(b, 1):7.3f} {mr:5.1f} "
                      f"{1000 * c / max(ms, 1e-9):7.2f}  " + " ".join(f"{x:.3f}" for x in curve))
                res[k] = {"blocks": b, "committed": c, "tok_blk": c / max(b, 1), "rows_mean": mr,
                          "tok_s_model": 1000 * c / max(ms, 1e-9), "a": curve,
                          "rows_hist": {str(r): n for r, n in sorted(rows.items())}}
            out[label][name] = res
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
