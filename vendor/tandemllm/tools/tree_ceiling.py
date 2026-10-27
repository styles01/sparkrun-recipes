"""How fast could a draft tree possibly go? The ceiling, from the drafter's own candidate sets.

A chain accepts position l only where the drafter's top-1 is the target's token. A tree accepts it
wherever the target's token is among the candidates the tree carries at that slot. So a chain is
bounded by P(argmax) and a tree by **top-k coverage**, and the gap between those two numbers is the
entire size of the prize -- before any question of what a verify step costs.

`tools/record_lattice.py` writes the block drafter's candidate sets at EVERY position of a trace,
and they are exact: the lattice at position p depends only on the committed prefix, and in a
lossless engine the committed prefix is the greedy prefix whatever the drafting policy was. So this
file computes, not estimates:

  * `coverage[l][k]` -- how often the target's token at slot l is in that slot's top k;
  * for a branching vector `(b_0 .. b_{L-1})`, the EXACT distribution of accepted run length, by
    replaying every recorded position rather than multiplying marginals. Coverage at slot l is not
    independent of coverage at slot l-1 and a product of marginals gets it wrong in both directions;
  * the best branching under a node budget, and what that is worth in tokens per second under the
    measured verify curve and under a flattened one.

The node count of a branching vector is `SUM_l PROD_{j<=l} b_j`: the tree carries the top `b_l`
candidates at slot l under every surviving parent. The candidate SET at a slot does not depend on
the parent -- it is a top-k over the head's logits for that row -- so a path is in the tree exactly
when each of its tokens is in its slot's top `b_l`. That is what makes the replay exact.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The measured tree verify curve, nodes including the anchor (with today's
# kernels). A staircase: the W4A16 kernel does sixteen rows of tensor-core work whatever it is
# asked for, and a second tile costs a second sixteen.
TREE_MS = {2: 131.58, 4: 135.57, 8: 141.74, 12: 146.73, 16: 154.73, 24: 213.61, 32: 205.57}
COMMIT_MS = 6.8
DRAFT_MS = 35.0


def verify_ms(nodes: int, flat_from: int = 0, flat_factor: float = 1.3) -> float:
    """Today's staircase, or a flattened curve where verify(64) = `flat_factor` x verify(16).

    `flat_from` is the node count past which a wide tensor-core GEMM is assumed: the weights are
    read once whatever M is, and at M = 64 the step's arithmetic is about 3.5 TFLOP, which the
    tensor cores do in single-digit milliseconds. The flattened curve is what step (2) of the
    programme is trying to buy, and quoting the ceiling under both is the only honest way to say
    what the tree is worth.
    """
    b = nodes
    if flat_from and b > flat_from:
        base = TREE_MS[16]
        # linear in the number of 16-row tiles, scaled so that 64 nodes costs flat_factor x 16
        return base * (1.0 + (flat_factor - 1.0) * (b - 16) / (64 - 16))
    keys = sorted(TREE_MS)
    if b <= keys[0]:
        return TREE_MS[keys[0]]
    if b >= keys[-1]:
        slope = (TREE_MS[32] - TREE_MS[24]) / 8.0
        return TREE_MS[keys[-1]] + slope * (b - keys[-1])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= b <= hi:
            f = (b - lo) / (hi - lo)
            return TREE_MS[lo] + f * (TREE_MS[hi] - TREE_MS[lo])
    return TREE_MS[keys[-1]]


def load(tdir: str):
    """Per trace: the class, and a [positions, slots] integer matrix of the target's RANK.

    `rank[i][l]` is where the target's own token sits in slot l's candidate list at position i, or
    a large number if it is not in the list at all. Everything below is a comparison against it, so
    the whole analysis is one small integer matrix per trace.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(tdir, "*.json"))):
        npz = path[:-5] + ".lattice.npz"
        if not os.path.exists(npz):
            continue
        tr = json.load(open(path))
        z = np.load(npz)
        cand, at = z["cand"], z["at"]              # [N, slots, k], [N]
        target = np.array(tr["output_ids"], dtype=np.int64)
        N, L, k = cand.shape
        rank = np.full((N, L), 1 << 20, dtype=np.int32)
        for n in range(N):
            i = int(at[n])
            for l in range(L):
                j = i + 1 + l
                if j >= len(target):
                    continue
                hit = np.nonzero(cand[n, l] == target[j])[0]
                if hit.size:
                    rank[n, l] = int(hit[0])
        out.append((tr.get("klass", tr["name"].split("-")[0]), tr["name"], rank, k))
    return out


def coverage(rank: np.ndarray, k: int) -> np.ndarray:
    """Per slot, the fraction of positions whose target token is in that slot's top k."""
    return (rank < k).mean(axis=0)


def accepted_run(rank: np.ndarray, b: tuple[int, ...]) -> np.ndarray:
    """Exact accepted draft length at every position, for a tree with branching `b`.

    A path survives to depth l+1 only if every slot up to l carried the target's token, so this is
    the length of the leading run of `rank[:, l] < b[l]`. No independence assumption anywhere.
    """
    ok = rank[:, :len(b)] < np.array(b, dtype=np.int32)[None, :]
    # leading run of True per row
    return (np.cumprod(ok, axis=1)).sum(axis=1)


def nodes_of(b: tuple[int, ...]) -> int:
    n, prod = 0, 1
    for x in b:
        prod *= x
        n += prod
        if n > 1 << 20:
            break
    return n


def best_branching(rank: np.ndarray, budget: int, slots: int, kmax: int):
    """The branching vector under `budget` nodes that maximises the mean accepted run.

    Depth-first over the branching factors with the node count pruning it: a budget of 128 admits
    b_0 = 16 only if everything below it is nearly a chain, so the space collapses immediately.
    """
    best = (0.0, (1,) * slots)

    def rec(b: tuple[int, ...], used: int, prod: int):
        nonlocal best
        if b:
            score = accepted_run(rank, b).mean()
            if score > best[0] + 1e-12:
                best = (score, b)
        if len(b) == slots:
            return
        for nxt in range(1, kmax + 1):
            p2 = prod * nxt
            u2 = used + p2
            if u2 > budget - 1:                    # -1 for the anchor
                break
            rec(b + (nxt,), u2, p2)

    rec((), 0, 1)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=None)
    ap.add_argument("--budgets", default="8,16,32,64,128")
    ap.add_argument("--draft-ms", type=float, default=DRAFT_MS)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tdir = a.traces or os.path.join(root, "results", "traces")
    data = load(tdir)
    if not data:
        raise SystemExit(f"no *.lattice.npz in {tdir}; run tools/record_lattice.py on the board")
    slots, kmax = data[0][2].shape[1], data[0][3]
    print(f"{len(data)} traces with lattices, {sum(d[2].shape[0] for d in data):,} positions, "
          f"{slots} slots, {kmax} candidates a slot")

    classes = sorted({d[0] for d in data})
    by_class = {c: np.concatenate([d[2] for d in data if d[0] == c]) for c in classes}
    by_class["ALL"] = np.concatenate([d[2] for d in data])

    print("\n### top-k coverage of the target, by slot")
    print(f"{'class':6s} {'k':>3} " + " ".join(f"slot{l}" for l in range(slots)) + "   mean")
    for c in list(classes) + ["ALL"]:
        for k in (1, 2, 4, 8, 16):
            if k > kmax:
                continue
            cov = coverage(by_class[c], k)
            print(f"{c:6s} {k:3d} " + " ".join(f"{x:5.3f}" for x in cov) + f"   {cov.mean():5.3f}")
        print()

    print("### the ceiling: best branching under a node budget, and what it is worth")
    print(f"{'class':6s} {'N':>4} {'branching':22s} {'nodes':>6} {'E[run]':>7} {'tok/blk':>8} "
          f"{'tok/s now':>10} {'tok/s flat':>11}")
    for c in list(classes) + ["ALL"]:
        rank = by_class[c]
        chain = accepted_run(rank, (1,) * slots).mean()
        for N in [int(x) for x in a.budgets.split(",")]:
            score, b = best_branching(rank, N, slots, kmax)
            n = nodes_of(b) + 1
            gained = score + 1.0
            now = gained / ((verify_ms(n) + COMMIT_MS + a.draft_ms) / 1000.0)
            flat = gained / ((verify_ms(n, flat_from=16) + COMMIT_MS + a.draft_ms) / 1000.0)
            # past 32 nodes the verify curve is extrapolated, not measured: say so rather than
            # printing a number that looks like one
            mark = "*" if n > 32 else " "
            print(f"{c:6s} {N:4d} {str(b):22s} {n:6d} {score:7.3f} {gained:8.3f} "
                  f"{now:9.2f}{mark} {flat:10.2f}{mark}")
        chain_n = slots + 1
        chain_tps = (chain + 1.0) / ((verify_ms(chain_n) + a.draft_ms) / 1000.0)
        print(f"{c:6s} {'':4s} {'chain (all 1s)':22s} {chain_n:6d} {chain:7.3f} {chain + 1:8.3f} "
              f"{chain_tps:10.2f} {'':11s}")
        # what the target would require, which is arithmetic and not a projection
        best_gain = max(accepted_run(rank, b).mean()
                        for b in [(1,) * slots]) + 1.0
        wide = accepted_run(rank, (3, 4, 3, 2) + (1,) * max(0, slots - 4)).mean() + 1.0
        for target in (70.0, 100.0):
            print(f"{c:6s} {'':4s} to reach {target:5.0f} tok/s a block must take "
                  f"{best_gain / target * 1000:6.1f} ms as a chain "
                  f"({best_gain:.2f} tok) or {wide / target * 1000:6.1f} ms as a wide tree "
                  f"({wide:.2f} tok);  the byte floor is 71.2 ms and 16 nodes measure 143.9")
        print()
    print("* the verify curve is measured to 32 nodes and extrapolated past it")
    print("A block yields at most `slots + 1` tokens. That cap, not the tree and not the kernel,")
    print("is what stands between this drafter and a three-digit prose number.")


if __name__ == "__main__":
    main()
