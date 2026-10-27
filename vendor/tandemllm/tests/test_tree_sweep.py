"""CPU tests for tools/tree_sweep.py: the tree the engine builds, replayed.

The replay is only worth a decision if it is the loop: a lattice whose top-1 is always the target
must fill every block, an alternative the tree carries must be accepted when the target takes it,
the budget must count the anchor, and the two knobs must do what they are for -- a hotter selector
spreads the budget over more alternatives, a colder one spends it down the best path, and the prune
drops nodes that do not pay for their rows.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.tree import DraftTree  # noqa: E402
from tools.tree_sweep import accepted, build, greedy_walk, replay  # noqa: E402

SLOTS, K = 7, 4


def _lattice(target, truth_rank=None, margin=4.0, seed=0):
    """A lattice at every position: slot l's candidates put the target's token at rank
    `truth_rank(i, l)` (0 = top-1), the selector's scores prefer rank 0 by `margin`."""
    rng = np.random.default_rng(seed)
    n = len(target)
    cand = np.zeros((n, SLOTS, K), dtype=np.int32)
    scores = np.zeros((n, SLOTS, K, K), dtype=np.float32)
    for i in range(n):
        for l in range(SLOTS):
            j = i + 1 + l
            true = target[j] if j < n else 999_999
            others = [t for t in rng.integers(10_000, 20_000, size=K + 2) if t != true][:K - 1]
            r = truth_rank(i, l) if truth_rank else 0
            row = others[:r] + [true] + others[r:]
            cand[i, l] = row[:K]
            scores[i, l] = -margin * np.arange(K)[None, :]      # rank 0 best, from every row
    return {"name": "t", "klass": "t", "target": list(target), "cand": cand, "scores": scores,
            "index": {i: i for i in range(n)}}


def test_a_lattice_that_is_always_right_fills_every_block():
    target = list(range(100, 100 + 64))
    tr = _lattice(target)
    blocks, committed, nodes = replay(tr, budget=SLOTS + 1, temp=1.0, mode="paths", prune=False)
    assert committed == len(target) - 1, (blocks, committed)
    assert blocks == -(-(len(target) - 1) // (SLOTS + 1)), blocks
    assert nodes == blocks * (SLOTS + 1), "the budget counts the anchor"


def test_an_alternative_the_tree_carries_is_accepted():
    target = list(range(100, 100 + 40))
    # at every position slot 0's top-1 is wrong and the target is the second candidate
    tr = _lattice(target, truth_rank=lambda i, l: 1 if l == 0 else 0)
    _, c_chain, _ = replay(tr, budget=SLOTS + 1, temp=1.0, mode="paths", prune=False)
    _, c_wide, _ = replay(tr, budget=2 * (SLOTS + 1), temp=1.0, mode="paths", prune=False)
    assert c_chain == len(target) - 1 and c_wide == len(target) - 1
    b_chain, _, _ = replay(tr, budget=SLOTS + 1, temp=1.0, mode="paths", prune=False)
    b_wide, _, _ = replay(tr, budget=2 * (SLOTS + 1), temp=1.0, mode="paths", prune=False)
    assert b_chain == len(target) - 1, "the greedy path alone commits one token a block"
    assert b_wide < b_chain / 4, "a second full path from slot 0's alternative commits the block"


def test_accepted_follows_the_branch_the_target_takes():
    t = DraftTree(tokens=[0, 5, 6, 7, 8, 9], parents=[-1, 0, 1, 0, 3, 4])
    assert accepted(t, [0, 7, 8, 9, 1], 0) == 3
    assert accepted(t, [0, 5, 6, 1], 0) == 2
    assert accepted(t, [0, 4], 0) == 0
    assert accepted(t, [0, 7], 0) == 1, "the continuation ends"


def test_a_hotter_selector_spreads_the_budget_and_a_colder_one_goes_deep():
    target = list(range(100, 140))
    tr = _lattice(target, margin=1.0)
    cand, scores = tr["cand"][0], tr["scores"][0]
    width = {}
    for temp in (0.3, 3.0):
        t = build(cand, scores, anchor=target[0], budget=16, temp=temp, mode="nodes", prune=False)
        width[temp] = sum(1 for d in t.depths() if d == 1)
        assert len(t.tokens) == 16
    assert width[3.0] > width[0.3], width


def test_the_prune_drops_nodes_that_do_not_pay():
    target = list(range(100, 140))
    tr = _lattice(target, margin=6.0)                 # alternatives are nearly worthless
    cand, scores = tr["cand"][0], tr["scores"][0]
    full = build(cand, scores, anchor=target[0], budget=24, temp=1.0, mode="nodes", prune=False)
    cut = build(cand, scores, anchor=target[0], budget=24, temp=1.0, mode="nodes", prune=True,
                base_ms=80.0, per_node_ms=2.0)
    assert len(full.tokens) == 24 and len(cut.tokens) < len(full.tokens)
    assert greedy_walk(scores) == [0] * SLOTS


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
