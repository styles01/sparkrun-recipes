"""Relaxed acceptance, priced on the target's own recorded distribution before it is switched on.

Greedy speculative verification accepts a drafted token only when it is the target's argmax. That
rule is what makes the engine's output identical to unspeculated greedy, and it is also what caps
the engine at the 14:20 entry's numbers. Relaxing it is the only route above them, and it is a
quality change, so it is off by default and it never gets reported without its cost beside it.

Two rules, one knob each:

    tau     accept a drafted token `t` when `p(t) >= tau * p(argmax)`   (typical acceptance)
    rank    accept a drafted token `t` when it is among the target's `r` most likely  (lenient)

WHAT THIS TOOL COMPUTES, AND WHAT IT CANNOT
-------------------------------------------
Everything here follows from the target's top-64 at each recorded position, under one explicit model
of the drafter: that it proposes a token drawn from the target's own distribution. That model is the
same one the 14:20 entry used to put a ceiling on the lossless engine, where it matched the measured
block drafter to three per cent, so it is not a wild assumption -- but it is an assumption, and the
numbers here are a *first-order* estimate to be confirmed on the board at the two or three settings
worth confirming.

    acceptance    mass of the accept set, which is the per-token agreement probability `a`
    tok/s         `a` through the measured block cost, exactly as the lossless table does it
    NLL per token the expected increase in the negative log-likelihood of the emitted text under
                  the exact target, against greedy: `sum_{t in S} p(t) * (log p1 - log p(t))`
    argmax kept   the share of emitted tokens that are still the greedy token

What it cannot see is drift. A relaxed engine emits a token greedy would not have, and every token
after it is conditioned on a text greedy would never have written. The per-token NLL above is the
cost of the step, not the cost of the trajectory, and only a generation on the board shows the
second. That is the measurement this tool exists to aim, not to replace.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch

VERIFY_MS = {8: 114.1, 16: 129.2}
DRAFT_MS = 25.0


def load(paths: list[str], dedup: bool) -> list[dict]:
    out, seen = [], set()
    for path in paths:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        digest = hashlib.sha1(raw["ids"].numpy().tobytes()).digest()
        if dedup and digest in seen:
            continue
        seen.add(digest)
        lo = int(raw.get("gen_start", 0))
        out.append({
            "lp": raw["top_lp"][lo:].float().numpy().astype(np.float64),
            "topic": raw.get("topic", "?"),
            "name": raw.get("name", os.path.basename(path)),
        })
    return out


def score(lp: np.ndarray, keep: np.ndarray) -> tuple[float, float, float]:
    """Given log-probs [n, 64] and a boolean accept mask, return (acceptance, dNLL, argmax kept)."""
    p = np.exp(lp)
    mass = (p * keep).sum(axis=1)
    # sum_{t in S} p(t) * (log p1 - log p(t)) -- the expected extra nats per position
    delta = (p * keep * (lp[:, :1] - lp)).sum(axis=1)
    kept = p[:, 0] + (1.0 - mass)
    return float(mass.mean()), float(delta.mean()), float(kept.mean())


def run_length(a: float, m: int) -> float:
    return sum(a ** j for j in range(1, m + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="glob of recorded .pt sequences")
    parser.add_argument("--blocks", default="8,16")
    parser.add_argument("--taus", default="1.0,0.5,0.3,0.2,0.1,0.05,0.02,0.01")
    parser.add_argument("--ranks", default="1,2,3,4,6,8")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--by-topic", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    sequences = load(sorted(glob.glob(args.data)), not args.keep_duplicates)
    blocks = [int(b) for b in args.blocks.split(",")]
    lp = np.concatenate([s["lp"] for s in sequences])
    print(f"{len(sequences)} distinct sequences, {len(lp):,} generated positions, "
          f"top-64 mass {np.exp(lp).sum(axis=1).mean():.4f}")
    print()

    rows = []
    header = (f"{'rule':<12}{'accept a':>10}{'argmax kept':>13}{'dNLL/token':>12}"
              + "".join(f"{'tok/s@' + str(m):>11}" for m in blocks))
    print(header)
    print("-" * len(header))

    ranked = np.argsort(-lp, axis=1)  # already sorted, but make it explicit
    del ranked
    for tau in [float(x) for x in args.taus.split(",")]:
        keep = lp >= (lp[:, :1] + np.log(tau))
        a, dnll, kept = score(lp, keep)
        row = {"rule": f"tau {tau:g}", "tau": tau, "a": a, "dnll": dnll, "kept": kept}
        line = f"{'tau ' + format(tau, 'g'):<12}{a:>10.4f}{kept * 100:>12.2f}%{dnll:>12.4f}"
        for m in blocks:
            rate = (run_length(a, m) + 1) / (VERIFY_MS[m] + DRAFT_MS) * 1000
            row[f"tok_s_{m}"] = rate
            line += f"{rate:>11.1f}"
        print(line)
        rows.append(row)
    print()
    for r in [int(x) for x in args.ranks.split(",")]:
        keep = np.zeros_like(lp, dtype=bool)
        keep[:, :r] = True
        a, dnll, kept = score(lp, keep)
        row = {"rule": f"rank {r}", "rank": r, "a": a, "dnll": dnll, "kept": kept}
        line = f"{'rank ' + str(r):<12}{a:>10.4f}{kept * 100:>12.2f}%{dnll:>12.4f}"
        for m in blocks:
            rate = (run_length(a, m) + 1) / (VERIFY_MS[m] + DRAFT_MS) * 1000
            row[f"tok_s_{m}"] = rate
            line += f"{rate:>11.1f}"
        print(line)
        rows.append(row)

    print("\nA drafter is not a sampler from `p`; the measured one sits at a = 0.797 where this")
    print("model puts a faithful sampler at 0.818. The table above is therefore an upper reading of")
    print("what each setting buys, and a lower reading of what it costs.")

    # Where the substitutions land. A token swapped at a position the target was unsure about is a
    # different adjective; one swapped where the target was certain is a different fact. The accept
    # set is self-limiting here -- at p1 = 0.99 a tolerance of 0.3 admits only tokens above 0.297 --
    # and this is the measurement of how self-limiting it actually is.
    print()
    print("where a substitution lands, by the target's confidence at that position")
    bands = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)]
    p1 = np.exp(lp[:, 0])
    print(f"{'rule':<12}{'swaps/1000 tok':>16}" + "".join(f"{f'p1 {lo}-{hi}':>14}" for lo, hi in bands))
    for tau in (0.5, 0.3, 0.2, 0.1, 0.05):
        keep = lp >= (lp[:, :1] + np.log(tau))
        probs = np.exp(lp)
        # probability the emitted token is not the argmax, per position
        swap = (probs * keep).sum(axis=1) - probs[:, 0]
        line = f"{'tau ' + format(tau, 'g'):<12}{swap.mean() * 1000:>16.1f}"
        total = swap.sum()
        for lo, hi in bands:
            band = (p1 >= lo) & (p1 < hi)
            line += f"{swap[band].sum() / total * 100:>13.1f}%"
        print(line)

    if args.by_topic:
        print()
        topics = sorted({s["topic"] for s in sequences})
        print(f"{'topic':<14}{'pos':>7}" + "".join(f"{'t=' + format(t, 'g'):>9}"
                                                   for t in (1.0, 0.3, 0.1, 0.02)))
        for topic in topics:
            sub = np.concatenate([s["lp"] for s in sequences if s["topic"] == topic])
            line = f"{topic:<14}{len(sub):>7}"
            for tau in (1.0, 0.3, 0.1, 0.02):
                keep = sub >= (sub[:, :1] + np.log(tau))
                a, _, _ = score(sub, keep)
                line += f"{a:>9.3f}"
            print(line)

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(rows, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
