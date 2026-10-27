"""The tree builder's knobs, swept on recorded lattices: the tree the engine would build, replayed.

`tools/tree_ceiling.py` prices the best branching vector there is -- an upper bound no builder
reaches. This replays the builder the engine runs (`engine/tree.py::lattice_paths` or
`lattice_tree`, the selector's scores through a softmax at `--temps`, optionally `DraftTree.prune`
under the verify curve) at every block of the loop, exactly: in a lossless engine the committed
prefix is the greedy prefix whatever the tree was, so the lattice at each anchor is the recorded
one (`tools/record_lattice.py`), and a block accepts the longest path of its tree that matches the
continuation. What it reports per class and setting is committed tokens a block -- the accepted run
plus the target's own token -- and nodes a block.

The head's tree only: the lookup drafter's merge (which fires on copies, not on new text) and the
length router's latch are not replayed. (the knobs) and (the budget) read it.

    python tools/tree_sweep.py results/lat-b16 --budgets 16,24,32 --temps 0.5,0.7,1,1.4,2 \\
        --modes paths,nodes --prune off,on
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.tree import DraftTree, lattice_paths, lattice_tree  # noqa: E402

# The tree verify's cost for the prune rule, nodes including the anchor: the engine's block after
# Phase 1 (verify ~79 ms of a 93 ms block at 16 rows). Only the per-node slope and the base matter
# to the rule; the curve replaces them where it has measured more.
PRUNE_BASE_MS = 79.0
PRUNE_PER_NODE_MS = 0.5


def load(tdir: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(tdir, "*.json"))):
        npz = path[:-5] + ".lattice.npz"
        if not os.path.exists(npz):
            continue
        tr = json.load(open(path))
        z = np.load(npz)
        out.append({"name": tr["name"], "klass": tr.get("klass", tr["name"].split("-")[0]),
                    "target": [int(t) for t in tr["output_ids"]], "cand": z["cand"],
                    "scores": z["scores"].astype(np.float32),
                    "index": {int(p): i for i, p in enumerate(z["at"])}})
    return out


def log_softmax(x: np.ndarray, temp: float) -> np.ndarray:
    y = x / temp
    y = y - y.max(axis=-1, keepdims=True)
    return y - np.log(np.exp(y).sum(axis=-1, keepdims=True))


def greedy_walk(scores: np.ndarray) -> list[int]:
    """The released greedy walk: slot 0's argmax, then each slot's argmax in the row the previous
    slot took (as `tools/sim_draft.py::_ReplayDFlash2._walk` and the drafter's host walk)."""
    idx = int(scores[0, 0].argmax())
    path = [idx]
    for e in range(1, scores.shape[0]):
        idx = int(scores[e, idx].argmax())
        path.append(idx)
    return path


def build(cand: np.ndarray, scores: np.ndarray, *, anchor: int, budget: int, temp: float,
          mode: str, prune: bool, base_ms: float = PRUNE_BASE_MS,
          per_node_ms: float = PRUNE_PER_NODE_MS) -> DraftTree:
    """The head's tree at one anchor, as `DFlash2Drafter.propose_tree` builds it (then pruned as
    `MergedRouter` prunes, when asked). `budget` counts nodes INCLUDING the anchor."""
    greedy = greedy_walk(scores)
    logp = log_softmax(scores, temp).tolist()
    fn = lattice_paths if mode == "paths" else lattice_tree
    t = fn(anchor, cand.tolist(), logp, greedy, budget - 1)
    if prune and t.n_draft > 0:
        t = t.prune(budget - 1, per_node_ms=per_node_ms, base_ms=base_ms)
    return t


def accepted(t: DraftTree, target: list[int], a: int) -> int:
    """The longest path of `t` that the continuation after anchor `a` follows."""
    kids: dict[int, dict[int, int]] = {}
    for i, p in enumerate(t.parents[1:], start=1):
        kids.setdefault(p, {})[t.tokens[i]] = i
    node, n = 0, 0
    while a + 1 + n < len(target):
        nxt = kids.get(node, {}).get(target[a + 1 + n])
        if nxt is None:
            break
        node, n = nxt, n + 1
    return n


def replay(tr: dict, **kw) -> tuple[int, int, int]:
    """The loop over one trace: blocks, committed tokens, verified nodes."""
    target = tr["target"]
    a, blocks, committed, nodes = 0, 0, 0, 0
    while a < len(target) - 1:
        i = tr["index"].get(a)
        if i is None:                     # no lattice here (the recorder skipped it): one token
            a += 1
            continue
        t = build(tr["cand"][i], tr["scores"][i], anchor=target[a], **kw)
        acc = accepted(t, target, a)
        blocks += 1
        committed += min(acc + 1, len(target) - 1 - a)     # no token past the continuation
        nodes += len(t.tokens)
        a += acc + 1
    return blocks, committed, nodes


# The verify + commit a block pays, by rows (tools/verify_curve.py, hold 3: graphs, fold, VERIFY_ROWS 32,
# the wide tile; `chain` for a line, the dearer of spine and bushy for a tree), and the rest of a block
# (the wide drafter's call, ~12.7 ms, and the host).
CHAIN_MS = {8: 76.66, 16: 76.66, 17: 79.59, 20: 80.96, 24: 83.28, 28: 88.17, 32: 93.36}
TREE_MS = {8: 77.90, 16: 78.88, 17: 83.16, 20: 83.68, 24: 87.04, 28: 92.12, 32: 96.98}
OTHER_MS = 13.7


def _interp(table: dict, rows: int) -> float:
    keys = sorted(table)
    if rows <= keys[0]:
        return table[keys[0]]
    if rows >= keys[-1]:
        return table[keys[-1]]
    for lo, hi in zip(keys, keys[1:]):
        if lo <= rows <= hi:
            return table[lo] + (rows - lo) / (hi - lo) * (table[hi] - table[lo])
    return table[keys[-1]]


def block_ms(t: DraftTree) -> float:
    rows = len(t.tokens)
    chain = all(p == i - 1 for i, p in enumerate(t.parents[1:], start=1))
    return _interp(CHAIN_MS if chain else TREE_MS, rows) + OTHER_MS


def replay_after(tr: dict, early: int, late: int, after: int, **kw) -> tuple[int, float, int]:
    """The loop over one trace with the budget `early` until `after` tokens are committed and `late`
    from then on (delayed wide tree). Returns tokens, milliseconds, blocks."""
    target = tr["target"]
    a, ms, blocks, committed = 0, 0.0, 0, 0
    while a < len(target) - 1:
        i = tr["index"].get(a)
        if i is None:
            a += 1
            continue
        t = build(tr["cand"][i], tr["scores"][i], anchor=target[a],
                  budget=late if committed >= after else early, **kw)
        acc = accepted(t, target, a)
        n = min(acc + 1, len(target) - 1 - a)
        committed += n
        ms += block_ms(t)
        blocks += 1
        a += acc + 1
    return committed, ms, blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", help="a directory of traces with their .lattice.npz")
    ap.add_argument("--budgets", default="16,24,32")
    ap.add_argument("--temps", default="0.5,0.7,1,1.4,2")
    ap.add_argument("--modes", default="paths,nodes")
    ap.add_argument("--prune", default="off,on")
    ap.add_argument("--base-ms", type=float, default=PRUNE_BASE_MS)
    ap.add_argument("--per-node-ms", type=float, default=PRUNE_PER_NODE_MS)
    ap.add_argument("--band", type=float, default=0.05, help="the tie band, committed tokens")
    ap.add_argument("--json", default="")
    ap.add_argument("--after", default="",
                    help="the delayed wide tree: committed-token thresholds to try, e.g. "
                         "0,16,32,48,64,96,inf, with --early / --late budgets; prints each "
                         "request's tok/s under the block cost model and the row class's mean / "
                         "median / max")
    ap.add_argument("--early", type=int, default=16)
    ap.add_argument("--late", type=int, default=24)
    ap.add_argument("--klass", default="row", help="with --after: the class to report")
    a = ap.parse_args()
    traces = load(a.traces)
    if a.after:
        mode, temp = a.modes.split(",")[0], float(a.temps.split(",")[0])
        rows = [t for t in traces if t["klass"] == a.klass]
        print(f"{len(rows)} {a.klass} traces; early {a.early} nodes, late {a.late}, {mode}, temp {temp}")
        print(f"{'after':>6} {'mean':>8} {'median':>8} {'max':>8} {'shortest':>10} {'tok/blk':>8}")
        for x in a.after.split(","):
            after = 10 ** 9 if x == "inf" else int(x)
            rates, toks, blks = [], 0, 0
            for tr in rows:
                c, ms, b = replay_after(tr, a.early, a.late, after, temp=temp, mode=mode,
                                        prune=False)
                rates.append((c / (ms / 1000.0), len(tr["target"]), tr["name"]))
                toks += c
                blks += b
            r = sorted(x[0] for x in rates)
            short = min(rates, key=lambda x: x[1])
            print(f"{x:>6} {sum(r) / len(r):8.2f} {r[len(r) // 2]:8.2f} {r[-1]:8.2f} "
                  f"{short[0]:7.2f} ({short[1]}) {toks / blks:8.3f}")
        return
    if not traces:
        raise SystemExit(f"no traces with lattices in {a.traces}")
    classes = sorted({t["klass"] for t in traces})
    print(f"{len(traces)} traces, classes {classes}, "
          f"{sum(len(t['index']) for t in traces)} recorded positions")
    rows = []
    t0 = time.perf_counter()
    for budget, mode, temp, pr in itertools.product(
            [int(x) for x in a.budgets.split(",")], a.modes.split(","),
            [float(x) for x in a.temps.split(",")], a.prune.split(",")):
        per: dict[str, list[int]] = {}
        for tr in traces:
            b, c, n = replay(tr, budget=budget, temp=temp, mode=mode, prune=pr == "on",
                             base_ms=a.base_ms, per_node_ms=a.per_node_ms)
            for k in (tr["klass"], "ALL"):
                acc = per.setdefault(k, [0, 0, 0])
                acc[0] += b; acc[1] += c; acc[2] += n
        rows.append({"budget": budget, "mode": mode, "temp": temp, "prune": pr,
                     "committed": {k: v[1] / v[0] for k, v in per.items() if v[0]},
                     "nodes": {k: v[2] / v[0] for k, v in per.items() if v[0]}})
    cols = classes + ["ALL"]
    print(f"({time.perf_counter() - t0:.1f} s)\n\ncommitted tokens a block (nodes a block for ALL)")
    print(f"{'budget':>6} {'mode':>6} {'temp':>5} {'prune':>5} "
          + " ".join(f"{c:>8}" for c in cols) + f" {'nodes':>6}")
    for r in rows:
        print(f"{r['budget']:6d} {r['mode']:>6} {r['temp']:5.2f} {r['prune']:>5} "
              + " ".join(f"{r['committed'].get(c, float('nan')):8.3f}" for c in cols)
              + f" {r['nodes']['ALL']:6.1f}")
    # the served default at each budget, and whether any setting beats it by more than the band
    print()
    for budget in sorted({r["budget"] for r in rows}):
        here = [r for r in rows if r["budget"] == budget]
        base = next((r for r in here if r["mode"] == "paths" and r["temp"] == 1.0
                     and r["prune"] == "off"), None)
        best = max(here, key=lambda r: r["committed"]["ALL"])
        if base is None:
            continue
        gain = best["committed"]["ALL"] - base["committed"]["ALL"]
        verdict = (f"{best['mode']} temp {best['temp']} prune {best['prune']} "
                   f"+{gain:.3f} ({100 * gain / base['committed']['ALL']:+.1f} %)"
                   if gain > a.band else f"no change (best +{gain:.3f} is inside the {a.band} band)")
        print(f"budget {budget}: served default {base['committed']['ALL']:.3f} committed a block; "
              f"best {verdict}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
