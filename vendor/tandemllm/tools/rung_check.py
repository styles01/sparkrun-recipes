"""ENG-209 gate 1 over recorded lattices (CPU only): the Rung fill (`engine/rung.py`) against the host
builder (`engine/tree.py::lattice_tree`) node for node at 16, 32 and 64 rows, at every recorded
position of `tools/record_lattice.py`'s traces; and, on the same positions, what each cut policy
commits against the target's own text:

  host16       lattice_tree at the served narrow budget (16 nodes), cut by the host StairCut rule
               (`MergedRouter._stair_cut`, score-order admission, any node count) on the staircase
  rungT        the T-row Rung, uncut (T = 16, 32, 64)
  rung-pick    the Rung step picked by (1 + E[acc](T)) / (V(T) + D + H), from {16, 32} and {16, 32, 64}

Committed tokens a block include the target's own token; the block walk is tools/tree_sweep.py's.
Prices: --stair (a stair-tables JSON, context class 1024, the tree column), --v64 for the 64-row price
(no measured 64-row step yet: V(16) + --v64-over), draft --draft-ms, host --host-ms.

Usage: rung_check.py --stair stair-tables-exl3-4.5.json DIR [DIR ...]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import rung  # noqa: E402
from engine.router import MergedRouter, verify_ms  # noqa: E402
from engine.tree import DraftTree, lattice_tree  # noqa: E402
from tools.tree_sweep import accepted, greedy_walk, load, log_softmax  # noqa: E402


def host_cut(tree: DraftTree, table: dict[int, float], draft_ms: float, host_ms: float,
             budget: int) -> DraftTree:
    """The host StairCut on a lattice tree, calibration = the raw path probabilities."""
    stub = types.SimpleNamespace(
        stair_table=table, tree_table=table, verify_table=table, node_budget=budget,
        stair_price_fn=None, rollback_ms=0.0, commit_ms=host_ms, stair_factor=None, stair_snap=None,
        _node_q=lambda n, t, d, r=0: min(1.0, t.scores[n]))
    cut, _ = MergedRouter._stair_cut(stub, tree, draft_ms)
    return cut if cut is not None else DraftTree.chain(tree.tokens[0], [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--stair", required=True)
    ap.add_argument("--ctx", default="1024")
    ap.add_argument("--cost-stair", default="", help="the staircase the tok/s column is costed on "
                    "(default --stair); --stair is then only what the cut prices with")
    ap.add_argument("--draft-ms", type=float, default=14.7)
    ap.add_argument("--host-ms", type=float, default=3.5)
    ap.add_argument("--v64-over", type=float, default=8.0, help="V(64) = V(16) + this")
    ap.add_argument("--budget", type=int, default=16, help="the served narrow node budget")
    ap.add_argument("--extra", default="", help="staircase rows past 32 'rows:ms,...' (measured "
                    "or projected); with it the host cut may use up to --budget-wide nodes")
    ap.add_argument("--budget-wide", type=int, default=63)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    tab = {int(k): float(v) for k, v in json.load(open(a.stair))[a.ctx]["tree"].items()}
    ctab = ({int(k): float(v) for k, v in json.load(open(a.cost_stair))[a.ctx]["tree"].items()}
            if a.cost_stair else tab)
    v16 = verify_ms(16, ctab)
    extra = {int(k): float(v) for k, v in (x.split(":") for x in a.extra.split(",") if x)}
    wide = {**tab, **extra}
    cwide = {**ctab, **extra}
    if extra:
        price = lambda T: verify_ms(T, cwide)  # noqa: E731
    else:
        price = lambda T: verify_ms(T, ctab) if T <= 32 else v16 + a.v64_over  # noqa: E731
    out = {}
    for d in a.dirs:
        traces = load(os.path.expanduser(d))
        ident = collections.Counter()
        npos = 0
        pol = ["host16", "rung16", "rung32", "rung64", "pick16-32", "pick16-64"]
        rows_used = collections.Counter()
        if extra:
            pol.append("hostwide")
        by = collections.defaultdict(lambda: {p: [0, 0, 0.0] for p in pol})   # blocks, committed, ms
        for tr in traces:
            target = tr["target"]
            res = {p: [0, 0, 0.0] for p in pol}
            pos = {p: 0 for p in pol}
            # identity at every recorded position
            for at, i in tr["index"].items():
                cand, scores = tr["cand"][i], tr["scores"][i]
                greedy = greedy_walk(scores)
                logp = log_softmax(scores, a.temp).tolist()
                cl = cand.tolist()
                f = rung.fill(int(target[at]), cl, logp, greedy)
                for T in rung.SIZES:
                    h = lattice_tree(int(target[at]), cl, logp, greedy, T - 1)
                    r = rung.cut(f, T)
                    ok = r.tokens.tolist() == h.tokens and r.parents.tolist() == h.parents \
                        and r.scores.tolist() == h.scores
                    ident[(T, ok)] += 1
                npos += 1
            # each policy's own walk over the trace
            for p in pol:
                at = 0
                while at < len(target) - 1:
                    i = tr["index"].get(at)
                    if i is None:
                        at += 1
                        continue
                    cand, scores = tr["cand"][i], tr["scores"][i]
                    greedy = greedy_walk(scores)
                    logp = log_softmax(scores, a.temp).tolist()
                    cl = cand.tolist()
                    anchor = int(target[at])
                    if p == "host16":
                        t = host_cut(lattice_tree(anchor, cl, logp, greedy, a.budget), tab,
                                     a.draft_ms, a.host_ms, a.budget)
                        rows = len(t.tokens)
                    elif p == "hostwide":
                        t = host_cut(lattice_tree(anchor, cl, logp, greedy, a.budget_wide), wide,
                                     a.draft_ms, a.host_ms, a.budget_wide)
                        rows = len(t.tokens)
                    else:
                        f = rung.fill(anchor, cl, logp, greedy)
                        if p.startswith("rung"):
                            T = int(p[4:])
                        else:
                            sizes = (16, 32) if p == "pick16-32" else (16, 32, 64)
                            T, _ = rung.pick(f, price, a.draft_ms, a.host_ms, sizes)
                        r = rung.cut(f, T)
                        t = DraftTree(r.tokens.tolist(), r.parents.tolist())
                        rows = T
                    acc = accepted(t, target, at)
                    res[p][0] += 1
                    if p == "host16":
                        rows_used[rows] += 1
                    res[p][1] += min(acc + 1, len(target) - 1 - at)
                    res[p][2] += price(rows) + a.draft_ms + a.host_ms
                    at += acc + 1
            for p in pol:
                for j in range(3):
                    by[tr["klass"]][p][j] += res[p][j]
        print(f"\n== {d}: {npos} positions; identical to lattice_tree: " +
              ", ".join(f"T{T} {ident[(T, True)]}/{ident[(T, True)] + ident[(T, False)]}"
                        for T in rung.SIZES))
        nb = sum(rows_used.values()) or 1
        print("host cut rows: mean %.1f, <=16: %.2f, 17-24: %.2f, 25-32: %.2f" % (
            sum(r * c for r, c in rows_used.items()) / nb, sum(c for r, c in rows_used.items() if r <= 16) / nb,
            sum(c for r, c in rows_used.items() if 17 <= r <= 24) / nb,
            sum(c for r, c in rows_used.items() if 25 <= r <= 32) / nb))
        print("class     " + " ".join(f"{p:>18s}" for p in pol) + "   (committed/block, tok/s at the prices)")
        rows_out = {}
        for k, v in sorted(by.items()):
            cells = []
            for p in pol:
                b, c, ms = v[p]
                cells.append(f"{c / max(b, 1):6.2f} {1000 * c / max(ms, 1e-9):6.1f}/s")
            print(f"{k:9s} " + " ".join(f"{x:>18s}" for x in cells))
            rows_out[k] = {p: {"committed": v[p][1] / max(v[p][0], 1),
                               "tok_s_model": 1000 * v[p][1] / max(v[p][2], 1e-9)} for p in pol}
        out[d] = {"positions": npos, "identical": {str(T): [ident[(T, True)], ident[(T, False)]]
                                                   for T in rung.SIZES}, "classes": rows_out}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
