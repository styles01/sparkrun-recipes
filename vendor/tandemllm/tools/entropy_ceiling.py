"""What speculation can and cannot buy on this target, computed from the target's own logits.

Two ceilings sit above every drafter this engine will ever have, and neither of them is about the
drafter.

THE HARDWARE CEILING is arithmetic on the measured block cost. A block proposes `m` tokens, the
verify pass costs `V(m)`, the drafter costs `d`, and the block yields at most `m + 1` tokens
because the verify pass also produces the target's own next token. So no lossless scheme can
exceed `(m + 1) / (V(m) + d)`, whatever it drafts with.

THE STATISTICAL CEILING is about the target. Greedy verification accepts a drafted token only when
it equals the target's argmax. A drafter that is a faithful model of the target's distribution --
one that has learned `p` and samples from it -- lands on the argmax with probability `p1`, the
target's own top-1 probability at that position. Its expected accepted run over a block is then the
product-sum of `p1` down the block, and that is a property of the text, not of the drafter. A
drafter can beat it only by being *more* deterministic than the target it imitates, which is what a
distilled greedy drafter is; a drafter can also fall short of it, and every one measured here does.

The margin `p1 - p2` is the second reading of the same thing. A drafter whose distribution is
within total variation `delta` of the target's is *guaranteed* to pick the target's argmax wherever
`p1 - p2 > 2 * delta`, and has no guarantee at all elsewhere. So the fraction of positions above
that margin is a floor on agreement for a drafter of a given quality, and the margin histogram says
how fast that floor moves as the drafter improves.

Input is the recorded training data: sequences with the target's top-64 at every position, both
kinds -- `gen`, where the conditioning text is the target's own continuation (the serving
distribution), and `corp`, where it is public text teacher-forced in one pass.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

# Measured block costs, phase 4, NVFP4 projections + fp8 head (ledger 12:29-12:47).
VERIFY_MS = {1: 92.2, 2: 104.0, 4: 107.6, 8: 114.1, 12: 121.0, 16: 129.2}
TREE_PREMIUM_MS = 12.5     # a tree of N nodes against a chain of N (ledger, tree section 13:48)
DRAFT_MS = 25.0            # the block drafter, one call, eight proposals
ROLLBACK_MS = 6.5


def load(path: str) -> dict:
    """Read one recorded sequence, keeping only the small tensors."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "ids": raw["ids"].numpy(),
        "label": raw["label"].numpy(),
        "top_lp": raw["top_lp"].float().numpy(),
        "top_ids": raw["top_ids"].numpy(),
        "gen_start": int(raw.get("gen_start", 0)),
        "kind": raw.get("kind", "?"),
        "topic": raw.get("topic", "?"),
        "name": raw.get("name", os.path.basename(path)),
    }


def run_length(p: np.ndarray, block: int) -> float:
    """Expected accepted tokens per block for per-position agreement probabilities `p`.

    A block starting at position i accepts its j-th token only if the j - 1 before it were accepted
    too, so the expectation is a product-sum and not a mean. Averaged over every starting position
    that has a full block after it.
    """
    n = len(p)
    if n <= block:
        return float("nan")
    total = 0.0
    count = 0
    for start in range(n - block):
        running = 1.0
        acc = 0.0
        for j in range(block):
            running *= p[start + j]
            acc += running
        total += acc
        count += 1
    return total / count


def invert_run_length(target: float, block: int) -> float:
    """The constant per-token agreement that would give `target` accepted tokens per block."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        value = sum(mid ** j for j in range(1, block + 1))
        if value < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="directory of recorded .pt sequences")
    parser.add_argument("--blocks", default="8,16", help="block lengths to report")
    parser.add_argument("--limit", type=int, default=0, help="read at most this many files per kind")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="do not drop sequences whose token ids are byte-identical to an earlier one")
    args = parser.parse_args()

    blocks = [int(b) for b in args.blocks.split(",")]
    groups: dict[str, list[dict]] = {}
    seen: set[bytes] = set()
    dropped = 0
    for path in sorted(glob.glob(os.path.join(args.data, "*.pt"))):
        kind = os.path.basename(path).split("-")[0]
        if args.limit and len(groups.get(kind, [])) >= args.limit:
            continue
        sequence = load(path)
        digest = hashlib.sha1(sequence["ids"].tobytes()).digest()
        if not args.keep_duplicates and digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        groups.setdefault(kind, []).append(sequence)
    if dropped:
        print(f"dropped {dropped} sequences whose token ids repeat an earlier one\n")
    if not groups:
        print(f"no sequences in {args.data}")
        return 1

    print("THE HARDWARE CEILING -- lossless, any drafter, measured block costs")
    print(f"{'shape':<22}{'m':>4}{'block ms':>10}{'max tok':>9}{'ceiling tok/s':>15}")
    for m in blocks:
        verify = VERIFY_MS[m]
        cost = verify + DRAFT_MS
        print(f"{'chain':<22}{m:>4}{cost:>10.1f}{m + 1:>9}{(m + 1) / cost * 1000:>15.1f}")
    for m in blocks:
        cost = VERIFY_MS[m] + TREE_PREMIUM_MS + DRAFT_MS
        print(f"{'tree (longest path)':<22}{m:>4}{cost:>10.1f}{m + 1:>9}{(m + 1) / cost * 1000:>15.1f}")
    print()

    report: dict = {"hardware": {}, "classes": {}}
    for m in blocks:
        report["hardware"][m] = {
            "chain_ms": VERIFY_MS[m] + DRAFT_MS,
            "chain_ceiling": (m + 1) / (VERIFY_MS[m] + DRAFT_MS) * 1000,
        }

    print("THE TARGET'S OWN CONFIDENCE, per position, on its own continuations")
    header = (f"{'kind':<8}{'seqs':>6}{'pos':>8}{'mean p1':>9}{'med p1':>8}{'p1<0.5':>8}"
              f"{'mean H':>8}{'top64':>7}{'mean gap':>9}")
    print(header)
    for kind, sequences in sorted(groups.items()):
        p1s, gaps, entropies, masses = [], [], [], []
        for seq in sequences:
            lo = seq["gen_start"] if kind == "gen" else 0
            lp = seq["top_lp"][lo:]
            probs = np.exp(lp.astype(np.float64))
            p1s.append(probs[:, 0])
            gaps.append(probs[:, 0] - probs[:, 1])
            mass = probs.sum(axis=1).clip(1e-9, 1.0)
            masses.append(mass)
            entropies.append(-(probs * np.log(probs.clip(1e-12, None))).sum(axis=1))
        p1 = np.concatenate(p1s)
        gap = np.concatenate(gaps)
        ent = np.concatenate(entropies)
        mass = np.concatenate(masses)
        print(f"{kind:<8}{len(sequences):>6}{len(p1):>8}{p1.mean():>9.3f}{np.median(p1):>8.3f}"
              f"{(p1 < 0.5).mean() * 100:>7.1f}%{ent.mean():>8.3f}{mass.mean():>7.3f}"
              f"{gap.mean():>9.3f}")
        report["classes"][kind] = {
            "sequences": len(sequences), "positions": int(len(p1)),
            "mean_p1": float(p1.mean()), "median_p1": float(np.median(p1)),
            "frac_p1_below_half": float((p1 < 0.5).mean()),
            "mean_entropy_top64": float(ent.mean()), "mean_top64_mass": float(mass.mean()),
            "mean_gap": float(gap.mean()),
            "p1_deciles": [float(x) for x in np.percentile(p1, np.arange(0, 101, 10))],
        }
    print()

    print("p1 deciles (a drafter that models p exactly and samples from it hits the argmax at p1)")
    print(f"{'kind':<8}" + "".join(f"{d:>7}%" for d in range(0, 101, 10)))
    for kind in sorted(groups):
        deciles = report["classes"][kind]["p1_deciles"]
        print(f"{kind:<8}" + "".join(f"{x:>8.3f}" for x in deciles))
    print()

    print("THE STATISTICAL CEILING -- expected accepted tokens per block, and the tok/s it gives")
    print(f"{'kind':<8}{'block':>7}{'E[acc] at p1':>14}{'tok/block':>11}{'tok/s':>9}"
          f"{'a needed 70':>13}{'a needed 100':>14}")
    for kind, sequences in sorted(groups.items()):
        for m in blocks:
            per_seq = []
            for seq in sequences:
                lo = seq["gen_start"] if kind == "gen" else 0
                probs = np.exp(seq["top_lp"][lo:, 0].astype(np.float64))
                value = run_length(probs, m)
                if not math.isnan(value):
                    per_seq.append(value)
            if not per_seq:
                continue
            expected = float(np.mean(per_seq))
            cost = VERIFY_MS[m] + DRAFT_MS
            # A block yields the accepted prefix plus the target's own token.
            toks = expected + 1.0
            rate = toks / cost * 1000
            need70 = invert_run_length(70 * cost / 1000 - 1.0, m)
            need100 = invert_run_length(100 * cost / 1000 - 1.0, m)
            print(f"{kind:<8}{m:>7}{expected:>14.2f}{toks:>11.2f}{rate:>9.1f}"
                  f"{need70:>13.3f}{need100:>14.3f}")
            report.setdefault("ceiling", {}).setdefault(kind, {})[m] = {
                "expected_accepted": expected, "tok_s": rate,
                "a_for_70": need70, "a_for_100": need100,
            }
    print()

    print("THE MARGIN FLOOR -- share of positions a drafter at total variation `delta` must get right")
    deltas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    print(f"{'kind':<8}" + "".join(f"{'d=' + str(d):>9}" for d in deltas))
    for kind, sequences in sorted(groups.items()):
        gaps = []
        for seq in sequences:
            lo = seq["gen_start"] if kind == "gen" else 0
            probs = np.exp(seq["top_lp"][lo:].astype(np.float64))
            gaps.append(probs[:, 0] - probs[:, 1])
        gap = np.concatenate(gaps)
        print(f"{kind:<8}" + "".join(f"{(gap > 2 * d).mean():>9.3f}" for d in deltas))
    print()

    print("WHERE THE MEASURED DRAFTER SITS")
    for label, measured, m in (("block drafter, prose", 3.28, 8), ("block drafter, mean", 4.30, 8),
                               ("merged router, edit", 8.40, 16)):
        implied = invert_run_length(measured, m)
        print(f"  {label:<24} {measured:>5.2f} accepted/block of {m:<3} implied per-token a = {implied:.3f}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
