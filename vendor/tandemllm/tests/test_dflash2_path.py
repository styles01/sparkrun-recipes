"""The two ways of walking the block drafter's lattice, checked against brute force.

The selector scores a whole draft block as a first-order chain: a unary term per slot plus a
pairwise term between neighbouring slots. `DFlash2Module.walk` is what the released implementation
does -- take the best candidate in slot 0, then the best successor of whatever slot 0 chose, and so
on -- and `DFlash2Module.viterbi` is the maximiser of the same objective.

The lattice is small enough (7 slots x 16 candidates) that the exhaustive answer can be computed
for a scaled-down version of it, so the test is not "does Viterbi look right" but "is it equal to
the argmax over every path". That is the only claim worth making about it: a *correct* maximiser
does not guarantee a better draft, it guarantees that the selector's own preference is what gets
proposed.

CPU only. No checkpoint, no board.
"""

from __future__ import annotations

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.dflash2 import DFlash2Module  # noqa: E402


def _lattice(L: int, k: int, seed: int):
    """A random lattice in the shape the drafter builds: ids [L, k], scores [L, k_pred, k_cand].

    Slot 0's predecessor rows are all the same anchor, so `scores[0]` is made constant down its
    predecessor axis -- exactly what `lattice()` produces for the anchor row.
    """
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(1000, generator=g)[: L * k].view(L, k)
    scores = torch.randn(L, k, k, generator=g)
    scores[0] = scores[0, 0].expand(k, k).clone()
    return ids, scores


def _path_score(scores: torch.Tensor, path: list[int]) -> float:
    total = float(scores[0, 0, path[0]])
    for l in range(1, len(path)):
        total += float(scores[l, path[l - 1], path[l]])
    return total


def _brute_force(scores: torch.Tensor) -> tuple[float, list[int]]:
    L, k = scores.shape[0], scores.shape[1]
    best, arg = None, None
    for path in itertools.product(range(k), repeat=L):
        s = _path_score(scores, list(path))
        if best is None or s > best:
            best, arg = s, list(path)
    return best, arg


def _indices_of(ids: torch.Tensor, tokens: torch.Tensor) -> list[int]:
    return [int((ids[l] == tokens[l]).nonzero()[0, 0]) for l in range(ids.shape[0])]


def test_viterbi_equals_brute_force():
    for seed in range(12):
        ids, scores = _lattice(L=4, k=5, seed=seed)
        best, _ = _brute_force(scores)
        got = DFlash2Module.viterbi(ids, scores)
        assert _path_score(scores, _indices_of(ids, got)) == best or \
            abs(_path_score(scores, _indices_of(ids, got)) - best) < 1e-5, seed


def test_viterbi_never_scores_below_the_greedy_walk():
    for seed in range(40):
        ids, scores = _lattice(L=7, k=16, seed=seed)
        g = _path_score(scores, _indices_of(ids, DFlash2Module.walk(ids, scores)))
        v = _path_score(scores, _indices_of(ids, DFlash2Module.viterbi(ids, scores)))
        assert v >= g - 1e-5, (seed, g, v)


def test_the_greedy_walk_is_sometimes_strictly_worse():
    """If this ever stops being true the change is not worth its code."""
    beaten = 0
    for seed in range(40):
        ids, scores = _lattice(L=7, k=16, seed=seed)
        g = _path_score(scores, _indices_of(ids, DFlash2Module.walk(ids, scores)))
        v = _path_score(scores, _indices_of(ids, DFlash2Module.viterbi(ids, scores)))
        beaten += v > g + 1e-5
    assert beaten > 0, "the greedy walk was optimal on all 40 random lattices"
    print(f"    (greedy beaten on {beaten} of 40 random lattices)")


def test_both_walks_return_one_token_per_slot_and_take_them_from_the_candidates():
    ids, scores = _lattice(L=7, k=16, seed=7)
    for fn in (DFlash2Module.walk, DFlash2Module.viterbi):
        out = fn(ids, scores)
        assert out.shape == (7,), out.shape
        for l in range(7):
            assert int(out[l]) in set(ids[l].tolist())


def test_a_single_slot_is_the_unary_argmax_for_both():
    ids, scores = _lattice(L=1, k=16, seed=3)
    want = int(ids[0, int(scores[0, 0].argmax())])
    assert int(DFlash2Module.walk(ids, scores)[0]) == want
    assert int(DFlash2Module.viterbi(ids, scores)[0]) == want


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
