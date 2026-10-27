"""The length router's policies replayed exactly on recorded lattices of both checkpoints.

In a lossless engine the committed text is the greedy text whatever the drafter proposed, so a
policy can be replayed without the board: at every anchor the recorded lattice of the narrow
(`lat-b8`) and of the wide checkpoint (`lat-b16`) is the one the drafter would have produced there
(`tools/record_lattice.py`), the tree is built from it as the drafter builds it, and the block
accepts the longest path of the tree that the recorded continuation follows. What differs from the
engine is only what no lattice records: the draft cost (a constant here) and tie flips.

The objects are the served ones: `LengthRouter` over two `MergedRouter` arms sharing one
`NgramDrafter` (the local index and, with `--corpus`, the corpus), so the lookup trees, the merge,
the prune, the latch, the deep chain and the switch policies all run their own code. Only the heads
are replays.

Cost a block: the verify on the served staircase (tree or chain by shape, `tools/tree_sweep.py`'s
tables), plus `--other-ms` (the draft call and the host), plus `--switch-ms` for every change of
the drafting checkpoint, swept.

    python tools/router_replay.py results/lat-b8 results/lat-b16 --corpus corpus \\
        --policies fixed8,fixed16,latch,lp,C,B,narrow --switch-ms 0,9,20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.ngram import NgramDrafter  # noqa: E402
from engine.lenrouter import LengthRouter  # noqa: E402
from engine.router import MergedRouter  # noqa: E402
from engine.tree import DraftTree, lattice_paths, lattice_tree  # noqa: E402
from tools.tree_sweep import CHAIN_MS, TREE_MS, _interp, accepted, greedy_walk  # noqa: E402

SERVED = {8: 77.9, 16: 78.9, 24: 87.0, 32: 97.0}
RHO_PRIOR: dict = {}


def load_rho_prior(path: str, weight: float = 50.0) -> dict:
    """{"src|m|runbin": [successes, failures]} from an uncensored offline count, as a Beta prior
    of `weight` pseudo-trials per bucket."""
    out = {}
    for k, v in json.load(open(path)).items():
        if k.startswith("_"):
            continue
        s_, f_ = v
        src, m, rb = k.split("|")
        if s_ + f_ >= 20:
            w = min(weight, s_ + f_)
            rho = s_ / (s_ + f_)
            out[(src, int(m), int(rb))] = [w * rho, w * (1 - rho)]
    return out


def load(d8: str, d16: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(d8, "*.json"))):
        name = os.path.basename(path)[:-5]
        p16 = os.path.join(d16, name + ".json")
        if not (os.path.exists(path[:-5] + ".lattice.npz") and os.path.exists(p16)):
            continue
        tr = json.load(open(path))
        tr16 = json.load(open(p16))
        if tr["output_ids"] != tr16["output_ids"]:
            continue
        lat = {}
        for key, p in (("s", path), ("l", p16)):
            z = np.load(p[:-5] + ".lattice.npz")
            lat[key] = (z["cand"], z["scores"].astype(np.float32),
                        {int(a): i for i, a in enumerate(z["at"])})
        out.append({"name": name, "klass": tr.get("klass", name.split("-")[0]),
                    "prompt": [int(t) for t in tr["prompt_ids"]],
                    "target": [int(t) for t in tr["output_ids"]], "lat": lat})
    return out


class _Cfg:
    def __init__(self, block):
        self.block_size = block


class _Eng:
    tap = None


class ReplayHead:
    """A DFlash2 head that proposes the tree its recorded lattice gives at the current anchor."""

    wants_rows = True

    def __init__(self, block: int, key: str, eng, temp: float = 1.0, mode: str = "nodes"):
        self.cfg = _Cfg(block)
        self.key = key
        self.eng = eng
        self.tree_temp = temp
        self.tree_mode = mode
        self._lattice = None
        self.last_det_tree = None
        self.last_q = None
        self.trace = None
        self.calls = 0

    def _on_tap(self, h):
        pass

    def reset(self):
        pass

    def sync(self, *a, **k):
        pass

    def set_sampling(self, s):
        pass

    def release(self, free_cache=False):
        pass

    def propose_tree(self, context, budget=16, **_):
        tr = self.trace
        a = len(context) - len(tr["prompt"]) - 1
        cand, scores, index = tr["lat"][self.key]
        i = index.get(a)
        if i is None or budget <= 0:
            return None
        self.calls += 1
        sc = scores[i] / self.tree_temp
        sc = sc - sc.max(axis=-1, keepdims=True)
        logp = (sc - np.log(np.exp(sc).sum(axis=-1, keepdims=True))).tolist()
        fn = lattice_paths if self.tree_mode == "paths" else lattice_tree
        return fn(tr["target"][a], cand[i].tolist(), logp, greedy_walk(scores[i]), budget)


def build_router(ng, heads, policy: dict, table: dict) -> LengthRouter:
    arms = [MergedRouter(ng, h, mtp_depth=h.cfg.block_size - 1,
                         node_budget=(16 if h.cfg.block_size <= 8 else 24) - 1,
                         mtp_ms_per_token=0.0, head_fixed_ms=27.0, adaptive_depth=False,
                         rollback_ms=6.4, verify_ms_table=dict(table), tree_ms_table=dict(table))
            for h in heads]
    kw = dict(tree=True, ngram=ng, latch=True, drop_idle=True, deep=32, deep_after=2,
              tree_wide_after=32, learn_cost=False, latch_table=dict(table),
              latch_price=False, switch=False)
    kw.update(policy.get("kw", {}))
    r = LengthRouter(arms[0], arms[1], **kw)
    for arm in arms:
        if "head_fixed_ms" in policy:
            arm.head_fixed_ms = policy["head_fixed_ms"]
        if policy.get("calib_off"):
            arm._source_calib = lambda i, tree: 1.0
        if "stair_table" in policy:
            arm.stair_table = dict(policy["stair_table"])
        if "snap" in policy:
            arm.stair_snap = tuple(policy["snap"])
        if policy.get("rho"):
            arm.stair_rho = True
            if RHO_PRIOR:
                arm.rho_prior = dict(RHO_PRIOR)
            arm.rho_min_bin = policy.get("rho_min_bin", 0)
            arm.rho_min_m = policy.get("rho_min_m", 3)
            arm.rho_online = policy.get("rho_online", True)
            arm.rho_nonlocal_min_bin = policy.get("nl_bin", 0)
            arm.rho_min_bin = policy.get("min_bin", 0)
            arm.skip_min_bin = policy.get("skip_bin", 0)
        if policy.get("rankcal"):
            arm.stair_rankcal = True
        if policy.get("lookrate"):
            arm.stair_lookrate = True
        if policy.get("buckets"):
            arm.stair_buckets = True
        if "opts" in policy:
            arm.stair_opts = tuple(policy["opts"])
    for h in heads:
        if "temp" in policy:
            h.tree_temp = policy["temp"]
        elif not r.calc:
            h.tree_temp = 1.0
    if policy.get("served_prices"):
        # the served loop's learned prices: the verify timed at the graph launch
        r.vms[r.w_small].value = r.vms[r.w_large].value = 0.2
        r.dms["s"].value, r.dms["l"].value = 12.4, 12.7
    return r


def block_ms(tree: DraftTree) -> float:
    rows = len(tree.tokens)
    chain = all(p == i - 1 for i, p in enumerate(tree.parents[1:], start=1))
    return _interp(CHAIN_MS if chain else TREE_MS, rows)


def replay(tr: dict, r: LengthRouter, heads, other_ms: float) -> dict:
    for h in heads:
        h.trace = tr
    target, prompt = tr["target"], tr["prompt"]
    r.reset()
    r.prime(prompt)
    ctx = prompt + [target[0]]
    r.observe([target[0]])
    a, ms, blocks, switches, prev = 0, 0.0, 0, 0, None
    calib = []
    while a < len(target) - 1:
        k = min(31, len(target) - 1 - a)
        tree = r.propose_tree(ctx, k)
        key = r.last_key if not r.last_deep else "d"
        if tree is None or tree.n_draft == 0:
            n, cost = 1, _interp(TREE_MS, 1)
        else:
            acc = accepted(tree, target, a)
            n = min(acc + 1, len(target) - 1 - a)
            cost = block_ms(tree)
        drafting = r.last_key if (tree is not None and not r.last_deep) else None
        if drafting and tree is not None and tree.n_draft:
            arm = r.small if drafting == "s" else r.large
            calib.append((min(arm._calibrated_expected(tree), float(tree.n_draft)),
                          min(accepted(tree, target, a), tree.n_draft)))
        if drafting and prev and drafting != prev:
            switches += 1
        if drafting:
            prev = drafting
        new = target[a + 1:a + 1 + n]
        r.sync(new, None, len(prompt) + a, rows=None)
        r.observe(new)
        ctx += new
        a += n
        ms += cost + other_ms
        blocks += 1
    return {"tokens": len(target) - 1, "ms": ms, "blocks": blocks, "switches": switches,
            "calib": calib,
            "latched": r.stats.get("latched"), "report": r.report()}


POLICIES = {
    "fixed8": {"kw": {"fixed": 8}},
    "fixed16": {"kw": {"fixed": 16}},
    "latch": {"served_prices": True},
    "lp": {"kw": {"latch_price": True}},
    "C": {"kw": {"switch": True, "switch_mode": "b", "exits": False}},
    "B": {"kw": {"switch": True, "switch_mode": "b"}},
    "B1": {"kw": {"switch": True, "switch_mode": "b", "switch_after": 1}},
    "B4": {"kw": {"switch": True, "switch_mode": "b", "switch_after": 4}},
    "narrow": {"kw": {"switch": True, "switch_mode": "narrow"}},
    "f8calc": {"kw": {"fixed": 8, "switch": True, "switch_mode": "calc"}},
    "f16calc": {"kw": {"fixed": 16, "switch": True, "switch_mode": "calc"}},
    "calc": {"kw": {"switch": True, "switch_mode": "calc"}},
    "f16calc7": {"kw": {"fixed": 16, "switch": True, "switch_mode": "calc"}, "head_fixed_ms": 7.1},
    "f16calc15": {"kw": {"fixed": 16, "switch": True, "switch_mode": "calc"}, "head_fixed_ms": 15.0},
    "wide": {"kw": {"switch": True, "switch_mode": "wide"}},
    "w-calib1": {"kw": {"switch": True, "switch_mode": "wide"}, "calib_off": True},
    "w-linear": {"kw": {"switch": True, "switch_mode": "wide"},
                 "stair_table": {8: 77.9, 32: 77.9 + 24 * 0.125}},
    "w-nochain": {"kw": {"switch": True, "switch_mode": "wide"}, "opts": ("mtp", "ngram", "merged")},
    "w-nomerge": {"kw": {"switch": True, "switch_mode": "wide"}, "opts": ("mtp", "chain", "ngram")},
    "w-nolookup": {"kw": {"switch": True, "switch_mode": "wide"}, "opts": ("mtp", "chain")},
    "w-cap23": {"kw": {"switch": True, "switch_mode": "wide", "max_nodes": 23}},
    "w-t07": {"kw": {"switch": True, "switch_mode": "wide"}, "temp": 0.7},
    "w-t14": {"kw": {"switch": True, "switch_mode": "wide"}, "temp": 1.4},
    "w-nosnap": {"kw": {"switch": True, "switch_mode": "wide", "stair_snap": None}},
    "w-lk": {"kw": {"switch": True, "switch_mode": "wide"}, "lookrate": True},
    "w-lk-nosnap": {"kw": {"switch": True, "switch_mode": "wide", "stair_snap": None}, "lookrate": True},
    "w-rank": {"kw": {"switch": True, "switch_mode": "wide"}, "rankcal": True},
    "w-rank-rho": {"kw": {"switch": True, "switch_mode": "wide"}, "rankcal": True, "rho": True},
    "w-rho": {"kw": {"switch": True, "switch_mode": "wide"}, "rho": True},
    "w-rho1": {"kw": {"switch": True, "switch_mode": "wide"}, "rho": True, "rho_min_bin": 1},
    "w-rho2": {"kw": {"switch": True, "switch_mode": "wide"}, "rho": True, "rho_min_bin": 2},
    "w-rho-m6": {"kw": {"switch": True, "switch_mode": "wide"}, "rho": True, "rho_min_m": 6},
    "w-rho-m8-skip-old": {"kw": {"switch": True, "switch_mode": "wide", "stair_skip": True},
                          "rho": True, "rho_min_m": 8, "rho_online": False},
    "w-rho-m8-skip-nl1": {"kw": {"switch": True, "switch_mode": "wide", "stair_skip": True},
                          "rho": True, "rho_min_m": 8, "nl_bin": 1},
    "w-rho-m8-skip-nl2": {"kw": {"switch": True, "switch_mode": "wide", "stair_skip": True},
                          "rho": True, "rho_min_m": 8, "nl_bin": 2},
    "w-rho-m8-skip-b1": {"kw": {"switch": True, "switch_mode": "wide", "stair_skip": True},
                         "rho": True, "rho_min_m": 8, "min_bin": 1},
    "w-rho-m8-skip-s1": {"kw": {"switch": True, "switch_mode": "wide", "stair_skip": True},
                         "rho": True, "rho_min_m": 8, "skip_bin": 1},
    "w-rho-m8-skip": {"kw": {"switch": True, "switch_mode": "wide", "stair_skip": True}, "rho": True, "rho_min_m": 8},
    "w-rho-m8": {"kw": {"switch": True, "switch_mode": "wide"}, "rho": True, "rho_min_m": 8},
    "w-rho-nosnap": {"kw": {"switch": True, "switch_mode": "wide", "stair_snap": None}, "rho": True},
    "w-snap": {"kw": {"switch": True, "switch_mode": "wide"}, "snap": (7, 15, 23, 31)},
    "w-snap2": {"kw": {"switch": True, "switch_mode": "wide"}, "snap": (3, 7, 11, 15, 19, 23, 27, 31)},
    "w-bucket": {"kw": {"switch": True, "switch_mode": "wide"}, "buckets": True},
    "calcw": {"kw": {"switch": True, "switch_mode": "calc", "calc_start": "l"}},
    "calcw0": {"kw": {"switch": True, "switch_mode": "calc", "calc_start": "l", "switch_margin": 0.0}},
    "calcw10": {"kw": {"switch": True, "switch_mode": "calc", "calc_start": "l", "switch_margin": 0.10}},
    "f16c24": {"kw": {"fixed": 16, "switch": True, "switch_mode": "calc", "max_nodes": 23}},
    "f16c16": {"kw": {"fixed": 16, "switch": True, "switch_mode": "calc", "max_nodes": 15}},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lat8")
    ap.add_argument("lat16")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--policies", default="fixed8,fixed16,latch,lp,C,B,B1,B4,narrow")
    ap.add_argument("--switch-ms", default="0,9,20")
    ap.add_argument("--other-ms", type=float, default=13.7)
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--reliability", default="", help="a policy: per-class reliability bins")
    ap.add_argument("--rho-prior", default="", help="offline lookup continuation counts (json)")
    a = ap.parse_args()
    traces = load(a.lat8, a.lat16)
    if a.rho_prior:
        RHO_PRIOR.update(load_rho_prior(a.rho_prior))
    if a.only:
        traces = [t for t in traces if t["klass"] in a.only.split(",")]
    eng = _Eng()
    heads = [ReplayHead(8, "s", eng), ReplayHead(16, "l", eng)]
    ng = NgramDrafter(corpus_path=a.corpus, min_order=3, max_depth=16, node_budget=15,
                      branch_top_k=3, min_expected=0.2, alpha=0.6, corpus_weight=0.5,
                      min_corpus_order=8, verify_base_ms=SERVED[8],
                      verify_per_node_ms=(SERVED[16] - SERVED[8]) / 8)
    pens = [float(x) for x in a.switch_ms.split(",")]
    names = a.policies.split(",")
    res = defaultdict(lambda: defaultdict(list))
    t0 = time.perf_counter()
    for name in names:
        pol = POLICIES.get(name)
        if pol is None and name.startswith("w-t"):
            # w-tNN: the wide mode with the lattice read at temperature NN/10
            pol = {"kw": {"switch": True, "switch_mode": "wide"}, "temp": int(name[3:]) / 10}
        r = build_router(ng, heads, pol, SERVED)
        for tr in traces:
            out = replay(tr, r, heads, a.other_ms)
            res[tr["klass"]][name].append(out)
            res["ALL"][name].append(out)
    print(f"{len(traces)} traces, {time.perf_counter() - t0:.1f} s; corpus "
          f"{'on' if a.corpus else 'off'}; other {a.other_ms} ms a block")
    for pen in pens:
        print(f"\nswitch price {pen:g} ms: mean tok/s per request (committed a block)")
        print(f"{'class':8s} " + " ".join(f"{n:>14s}" for n in names))
        for klass in sorted(res):
            cells = []
            for n in names:
                xs = res[klass][n]
                tps = statistics.mean(o["tokens"] / ((o["ms"] + pen * o["switches"]) / 1e3)
                                      for o in xs)
                tpb = sum(o["tokens"] for o in xs) / sum(o["blocks"] for o in xs)
                cells.append(f"{tps:8.2f} ({tpb:4.2f})")
            print(f"{klass:8s} " + " ".join(f"{c:>14s}" for c in cells))
    if a.reliability:
        print(f"\nper-class reliability of {a.reliability} (expected against realised accepted, bins "
              "of the calibrated expectation; slope = realised / expected over all rounds)")
        for klass in sorted(res):
            pairs = [c for o in res[klass][a.reliability] for c in o["calib"]]
            if not pairs:
                continue
            cells = []
            for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 99)):
                sel = [(x, y) for x, y in pairs if lo <= x < hi]
                if len(sel) >= 50:
                    me, mg = statistics.mean(x for x, _ in sel), statistics.mean(y for _, y in sel)
                    cells.append(f"[{lo},{hi}) n={len(sel)} {me:.2f}->{mg:.2f} ({100 * (mg / me - 1):+.0f}%)")
            se = sum(x for x, _ in pairs)
            sg = sum(y for _, y in pairs)
            print(f"  {klass:10s} rounds {len(pairs):5d} slope {sg / se:.3f} | " + " | ".join(cells))
    print("\nexpected (calibrated, the drafting arm's own) against realised accepted tokens a block")
    for n in names:
        pairs = [c for o in res["ALL"][n] for c in o["calib"]]
        if not pairs:
            continue
        e = [x for x, _ in pairs]
        g = [y for _, y in pairs]
        bins = []
        for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 99)):
            sel = [(x, y) for x, y in pairs if lo <= x < hi]
            if sel:
                bins.append(f"E[{lo},{hi}) n={len(sel)} E={statistics.mean(x for x, _ in sel):.2f} "
                            f"got={statistics.mean(y for _, y in sel):.2f}")
        corr = float(np.corrcoef(e, g)[0, 1]) if len(pairs) > 2 else float("nan")
        print(f"  {n:8s} blocks {len(pairs)} mean E {statistics.mean(e):.2f} got "
              f"{statistics.mean(g):.2f} corr {corr:.2f} | " + " | ".join(bins))
    sw = {n: statistics.mean(o["switches"] for o in res["ALL"][n]) for n in names}
    print("\nswitches a request: " + "  ".join(f"{n} {v:.2f}" for n, v in sw.items()))
    if a.json:
        json.dump({k: {n: v for n, v in d.items()} for k, d in res.items()}, open(a.json, "w"),
                  indent=1, default=str)


if __name__ == "__main__":
    main()
