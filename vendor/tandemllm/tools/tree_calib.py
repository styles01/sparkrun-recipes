"""Tree priorities that are acceptance probabilities, decided offline on recorded lattices.

The tree builder (`engine/tree.py::lattice_tree`, `nodes` mode, served) puts the greedy path in first
and then buys alternatives best-first by path probability, softmax(selector scores / T) multiplied
down the path. Those are the selector's own numbers, not acceptance probabilities. This fits, on
half the recorded traces, what a node's value should be -- P(candidate accepted | its parent
accepted) at its slot -- and replays the builder with it on the OTHER half, exactly as
`tools/tree_sweep.py` replays it (a lossless engine's continuation is the recorded one, so a block
accepts the longest path of its tree that the continuation follows):

    raw        log_softmax(scores / 1.0)                   the served priority
    temp       a temperature per slot, fitted by coordinate descent on the fit half's replay
    iso        per slot, an isotonic map from the softmax probability to the acceptance rate
    rank       per slot, the acceptance rate of the candidate's rank in its row (Sequoia-style)
    iso+rank   the geometric mean of the two

Fit data: every recorded position, walked along the TRUE continuation while it stays inside the
lattice; at slot l, in the row of the true slot l-1 candidate, each of the k candidates is one
example (softmax probability, rank, whether it is the true token). The greedy path itself is
unchanged by every method (the builder adds it first whatever its score); what moves is which
alternatives fill the rest of the budget.

    python tools/tree_calib.py results/lat/lat-b8 --budgets 12,16 --json results/p5/eng110-b8.json
    python tools/tree_calib.py results/lat/lat-b16 --budgets 24,32
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.tree import lattice_tree  # noqa: E402
from tools.tree_sweep import accepted, greedy_walk, load, log_softmax  # noqa: E402

FLOOR = 1e-6          # an acceptance estimate of zero still orders by the raw probability


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    return np.exp(log_softmax(x, temp))


def examples(traces: list[dict]):
    """(slot, rank, prob, label) for every candidate in the true path's rows."""
    rows = []
    for tr in traces:
        target = tr["target"]
        for a, i in tr["index"].items():
            cand, scores = tr["cand"][i], tr["scores"][i]
            L, k = cand.shape
            p = 0
            for slot in range(L):
                if a + 1 + slot >= len(target):
                    break
                true = target[a + 1 + slot]
                q = softmax(scores[slot, p])
                rank = np.argsort(np.argsort(-scores[slot, p]))
                hit = np.nonzero(cand[slot] == true)[0]
                for c in range(k):
                    rows.append((slot, int(rank[c]), float(q[c]), int(len(hit) and hit[0] == c)))
                if not len(hit):
                    break
                p = int(hit[0])
    return np.array(rows, dtype=np.float64)


def pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic (non-decreasing) regression of y on x by pool-adjacent-violators.
    Returns the block upper bounds in x and the fitted value of each block."""
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o]
    val, wt, hi = [], [], []
    for xv, yv in zip(xs, ys):
        val.append(yv); wt.append(1.0); hi.append(xv)
        while len(val) > 1 and val[-2] > val[-1]:
            w = wt[-2] + wt[-1]
            v = (val[-2] * wt[-2] + val[-1] * wt[-1]) / w
            val[-2:] = [v]; wt[-2:] = [w]; hi[-2:] = [hi[-1]]
    return np.array(hi), np.array(val)


class Calib:
    """A priority function over one lattice: scores [L, k, k] -> logp [L, k, k]."""

    def __init__(self, kind: str, L: int, temps=None, iso=None, rank=None):
        self.kind, self.L = kind, L
        self.temps = temps if temps is not None else np.ones(L)
        self.iso, self.rank = iso, rank

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        L = scores.shape[0]
        if self.kind in ("raw", "temp"):
            return np.stack([log_softmax(scores[l], float(self.temps[min(l, self.L - 1)]))
                             for l in range(L)])
        q = softmax(scores)
        out = np.empty_like(q)
        rk = np.argsort(np.argsort(-scores, axis=-1), axis=-1)
        for l in range(L):
            s = min(l, self.L - 1)
            parts = []
            if self.kind in ("iso", "iso+rank"):
                hi, val = self.iso[s]
                j = np.minimum(np.searchsorted(hi, q[l], side="left"), len(val) - 1)
                parts.append(np.log(val[j] + FLOOR * q[l]))
            if self.kind in ("rank", "iso+rank"):
                parts.append(np.log(self.rank[s][rk[l]] + FLOOR * q[l]))
            out[l] = sum(parts) / len(parts)
        return out


def replay(traces: list[dict], calib: Calib, budget: int) -> dict[str, list[int]]:
    """Blocks and committed tokens per class (and ALL) under `calib` at `budget` nodes."""
    per: dict[str, list[int]] = {}
    for tr in traces:
        target = tr["target"]
        a, b, c = 0, 0, 0
        while a < len(target) - 1:
            i = tr["index"].get(a)
            if i is None:
                a += 1
                continue
            cand, scores = tr["cand"][i], tr["scores"][i]
            t = lattice_tree(target[a], cand.tolist(), calib(scores).tolist(),
                             greedy_walk(scores), budget - 1)
            acc = accepted(t, target, a)
            b += 1
            c += min(acc + 1, len(target) - 1 - a)
            a += acc + 1
        for key in (tr["klass"], "ALL"):
            s = per.setdefault(key, [0, 0])
            s[0] += b
            s[1] += c
    return per


def rate(per: dict, key: str = "ALL") -> float:
    b, c = per.get(key, (0, 0))
    return c / b if b else float("nan")


def fit(traces: list[dict], L: int, budget: int, grid=(0.5, 0.7, 1.0, 1.4, 2.0, 3.0)) -> dict:
    ex = examples(traces)
    iso, rank = [], []
    k = traces[0]["cand"].shape[-1]
    for l in range(L):
        e = ex[ex[:, 0] == l]
        if len(e) == 0:
            iso.append(iso[-1]); rank.append(rank[-1]); continue
        iso.append(pav(e[:, 2], e[:, 3]))
        rank.append(np.array([e[e[:, 1] == r, 3].mean() if (e[:, 1] == r).any() else 0.0
                              for r in range(k)]))
    # per-slot temperature: coordinate descent on the fit half's own replay
    temps = np.ones(L)
    best = rate(replay(traces, Calib("temp", L, temps=temps), budget))
    for _ in range(2):
        for l in range(L):
            for T in grid:
                trial = temps.copy(); trial[l] = T
                r = rate(replay(traces, Calib("temp", L, temps=trial), budget))
                if r > best + 1e-9:
                    best, temps = r, trial
    return {"raw": Calib("raw", L), "temp": Calib("temp", L, temps=temps),
            "iso": Calib("iso", L, iso=iso), "rank": Calib("rank", L, rank=rank),
            "iso+rank": Calib("iso+rank", L, iso=iso, rank=rank),
            "_examples": int(len(ex)), "_temps": temps.tolist(),
            "_rank": [r.round(4).tolist() for r in rank]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces")
    ap.add_argument("--budgets", default="16")
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    traces = load(a.traces)
    traces.sort(key=lambda t: t["name"])
    halves = (traces[0::2], traces[1::2])
    L = traces[0]["cand"].shape[1]
    classes = sorted({t["klass"] for t in traces}) + ["ALL"]
    print(f"{len(traces)} traces ({len(halves[0])} / {len(halves[1])}), {L} slots, "
          f"{sum(len(t['index']) for t in traces)} positions")
    report = {"traces": a.traces, "slots": L, "band": a.band, "budgets": {}}
    for budget in (int(x) for x in a.budgets.split(",")):
        out = {}
        # fit on one half, score on the other, both ways; the two held-out halves are the whole set
        for fit_i, ev_i in ((0, 1), (1, 0)):
            cal = fit(halves[fit_i], L, budget)
            print(f"\nbudget {budget}: fit on half {fit_i} ({cal['_examples']} examples), "
                  f"temps {np.round(cal['_temps'], 2).tolist()}")
            for name in ("raw", "temp", "iso", "rank", "iso+rank"):
                per = replay(halves[ev_i], cal[name], budget)
                acc = out.setdefault(name, {})
                for key, (b, c) in per.items():
                    s = acc.setdefault(key, [0, 0])
                    s[0] += b
                    s[1] += c
            out.setdefault("_fits", []).append({"temps": cal["_temps"], "rank": cal["_rank"]})
        print(f"\nbudget {budget}, held out (each half scored by the other half's fit):")
        print(f"{'method':>9} " + " ".join(f"{c:>8}" for c in classes) + "   vs raw")
        base = rate(out["raw"])
        for name in ("raw", "temp", "iso", "rank", "iso+rank"):
            d = rate(out[name]) - base
            print(f"{name:>9} " + " ".join(f"{rate(out[name], c):8.3f}" for c in classes)
                  + f"   {d:+.3f} ({100 * d / base:+.1f} %)")
        best = max(("temp", "iso", "rank", "iso+rank"), key=lambda n: rate(out[n]))
        gain = rate(out[best]) - base
        verdict = (f"{best} +{gain:.3f} committed a block" if gain >= a.band
                   else f"STOP: best ({best}) {gain:+.3f} is inside the {a.band} band")
        print(f"budget {budget}: {verdict}")
        report["budgets"][budget] = {
            "committed": {n: {c: rate(v, c) for c in classes} for n, v in out.items()
                          if not n.startswith("_")},
            "fits": out["_fits"], "best": best, "gain": gain, "verdict": verdict}
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(report, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
