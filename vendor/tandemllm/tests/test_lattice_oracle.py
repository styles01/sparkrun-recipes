"""RND-32: the lattice oracle's replay on a synthetic lattice with a known answer (CPU)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from tools import lattice_oracle as O


def _trace(L=4, k=3, n=12):
    target = list(range(100, 100 + n))
    cand = np.zeros((n, L, k), np.int64)
    scores = np.zeros((n, L, k, k), np.float32)
    for a in range(n):
        for s in range(L):
            truth = target[a + 1 + s] if a + 1 + s < n else 0
            # the true token is candidate 1 (not the drafter's top-1) at slot 2, else candidate 0
            row = [truth, 900 + s, 950 + s] if s != 2 else [800, truth, 850]
            cand[a, s] = row
            scores[a, s, :, 0] = 2.0                 # every row prefers candidate 0
    return {"name": "t", "klass": "prose", "target": target, "cand": cand, "scores": scores,
            "index": {a: a for a in range(n)}}


def test_chain_stops_where_top1_is_wrong_and_cover_does_not():
    tr = _trace()
    cand, scores = tr["cand"][0], tr["scores"][0]
    assert O.chain_run(cand, scores, tr["target"], 0) == 2       # slots 0, 1 right; slot 2 wrong
    assert O.coverage_run(cand, tr["target"], 0) == 4            # the truth is in every slot
    b, c, fm, rk = O.replay(tr, "chain")
    assert fm[2] >= 1 and rk[1] >= 1                             # first miss at slot 2, rank 2
    bc, cc, _, _ = O.replay(tr, "cover")
    assert cc / bc > c / b                                       # the full lattice commits more


def test_tree_budget_buys_the_sibling():
    tr = _trace()
    b16, c16, _, _ = O.replay(tr, "tree", 16)
    b, c, _, _ = O.replay(tr, "chain")
    assert c16 / b16 >= c / b


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
