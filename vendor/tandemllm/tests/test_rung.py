"""ENG-209 step 1 / ENG-210: the Rung's CPU fill equals the host builder node for node."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import random

import numpy as np

from engine import rung
from engine.tree import lattice_tree


def _lattice(seed: int, L: int = 7, k: int = 16, ties: bool = False):
    rng = random.Random(seed)
    cand = [rng.sample(range(1000), k) for _ in range(L)]
    scores = np.array([[[rng.gauss(0, 2) for _ in range(k)] for _ in range(k)] for _ in range(L)])
    if ties:
        scores = np.round(scores)              # many exact ties: the tie counter must decide
    y = scores - scores.max(-1, keepdims=True)
    logp = (y - np.log(np.exp(y).sum(-1, keepdims=True))).astype(np.float32).tolist()
    idx = int(scores[0, 0].argmax())
    greedy = [idx]
    for e in range(1, L):
        idx = int(scores[e, idx].argmax())
        greedy.append(idx)
    return cand, logp, greedy


def _same(r: rung.Rung, t) -> bool:
    return (r.tokens.tolist() == t.tokens and r.parents.tolist() == t.parents
            and all(math.isclose(a, b, rel_tol=0, abs_tol=0) for a, b in zip(r.scores.tolist(), t.scores)))


def test_cuts_equal_host_builder():
    for seed in range(60):
        for L in (7, 15):
            cand, logp, greedy = _lattice(seed, L=L, ties=seed % 3 == 0)
            f = rung.fill(42, cand, logp, greedy)
            for T in rung.SIZES:
                host = lattice_tree(42, cand, logp, greedy, T - 1)
                r = rung.cut(f, T)
                assert _same(r, host), (seed, L, T)
                host.check()


def test_nesting_and_shapes():
    cand, logp, greedy = _lattice(7, L=15)
    f = rung.fill(5, cand, logp, greedy)
    small, big = rung.cut(f, 16), rung.cut(f, 32)
    assert set(small.order.tolist()) <= set(big.order.tolist())
    assert small.T == 16 and big.T == 32
    m = big.ancestor_mask()
    assert m.diagonal().all() and not np.triu(m, 1).any()          # pre-order: ancestors below
    from engine.tree import DraftTree
    t = DraftTree(big.tokens.tolist(), big.parents.tolist())
    assert big.conv_windows(4).tolist() == t.conv_windows(4)
    assert big.depths.tolist() == t.depths()


def test_pick_prefers_bigger_only_when_it_pays():
    cand, logp, greedy = _lattice(3, L=15)
    f = rung.fill(5, cand, logp, greedy)
    flat = lambda T: 70.0                                           # noqa: E731
    steep = lambda T: 70.0 if T <= 16 else 700.0                    # noqa: E731
    assert rung.pick(f, flat, 10.0, 3.0)[0] == 32
    assert rung.pick(f, steep, 10.0, 3.0)[0] == 16


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
