"""The batched sampled walk and the fast top-p filter (engine/sample.py BATCH, TOPP_K) on the CPU.

The batched walk must emit exactly the target's sampled distribution, as the one-draw-a-node walk
does: test_sample.py's statistical gates rerun with BATCH on, plus a q-aware tree (a sampled spine
child carrying its q beside a deterministic sibling), whose first emitted token must follow p
whatever q is. The fast filter must keep the sort's set on rows whose nucleus fits its k and fall
back to the sort on rows whose nucleus is wider.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import test_sample as TS  # noqa: E402
from engine import sample as S  # noqa: E402


def _batched(fn):
    old = S.BATCH
    S.BATCH = True
    try:
        return fn()
    finally:
        S.BATCH = old


def test_batched_tree_walk_matches_direct_sampling():
    _batched(TS.test_tree_walk_matches_direct_sampling)
    return "chain-shaped tree: first and second token follow their rows"


def test_batched_walk_is_exact_on_a_branched_tree():
    _batched(TS.test_tree_walk_is_exact_on_a_branched_tree)
    return "branched tree: every prefix within 0.012 of the chain's probability"


def test_batched_q_aware_tree_follows_p():
    V = 6
    g = torch.Generator().manual_seed(5)
    p0 = (torch.randn(V, generator=g) * 1.2).softmax(-1)
    q_row = (torch.randn(V, generator=g) * 2.0).softmax(-1)       # a miscalibrated proposal
    p1 = (torch.randn(V, generator=g)).softmax(-1)
    s = S.Sampler(temperature=1.0)
    dists = torch.stack([p0, p1, p1])
    counts = [0] * V
    n = 40000
    top = int(p0.argmax())

    def run():
        for _ in range(n):
            # the drafter samples its spine token from q every round; the sibling is
            # deterministic (p's argmax, or the next token when the spine took it)
            spine = int(torch.multinomial(q_row, 1, generator=g))
            sib = top if top != spine else (top + 1) % V
            path, new = s.tree_walk(dists, [0, spine, sib], [-1, 0, 0], q=[None, q_row, None])
            counts[new[0]] += 1
    _batched(run)
    for t in range(V):
        assert abs(counts[t] / n - float(p0[t])) < 0.012, (t, counts[t] / n, float(p0[t]))
    return "q-aware spine + sibling: first token follows p within 0.012"


def test_fast_top_p_keeps_the_sorts_set():
    g = torch.Generator().manual_seed(11)
    rows = torch.randn(200, 50, generator=g) * 3.0
    rows[0] = 0.0                                                  # flat: nucleus wider than k
    s = S.Sampler(temperature=0.7, top_p=0.9)
    old = S.TOPP_K
    try:
        S.TOPP_K = 0
        ref = s.probs_rows(rows)
        S.TOPP_K = 8
        got = s.probs_rows(rows)
    finally:
        S.TOPP_K = old
    assert torch.equal(ref > 0, got > 0), "kept sets differ"
    assert torch.allclose(ref, got, atol=0, rtol=0) or (ref - got).abs().max() < 1e-6
    return "200 rows (one wider than k = 8): same kept set, same probabilities"


def _nucleus(fn):
    old = (S.NUCLEUS, S.BATCH, S.TOPP_K)
    S.NUCLEUS, S.BATCH, S.TOPP_K = True, True, 8
    try:
        return fn()
    finally:
        S.NUCLEUS, S.BATCH, S.TOPP_K = old


def test_row_draws_follow_the_filtered_rows():
    """SAMPLE_NUCLEUS: every row's draws follow probs_rows(row), including a row whose nucleus is
    wider than k (the sort), and never leave the kept set."""
    g = torch.Generator().manual_seed(21)
    rows = torch.randn(4, 40, generator=g) * 3.0
    rows[1] = torch.randn(40, generator=g) * 0.1                   # flat: wider than k = 8
    s = S.Sampler(temperature=0.7, top_p=0.9)
    s.generator = torch.Generator().manual_seed(3)

    def run():
        ref = s.probs_rows(rows)
        n = 30000
        counts = torch.zeros_like(ref)
        for _ in range(n // 100):
            for _ in range(100):
                x = s.row_draws(rows)
                counts[torch.arange(4), x] += 1
        assert bool(((counts > 0) & (ref == 0)).sum() == 0), "a draw outside the kept set"
        err = (counts / n - ref).abs().max().item()
        assert err < 0.012, err
        return err
    err = _nucleus(run)
    return f"4 rows (one wider than k = 8), 30k draws: max |freq - p| {err:.4f}, kept set respected"


def test_tree_walk_logits_is_exact_on_a_branched_tree():
    """The nucleus walk on raw logits: the first two emitted tokens follow the filtered rows, as
    test_sample's branched-tree gate checks the probability walk."""
    g = torch.Generator().manual_seed(8)
    V = 12
    rows = torch.randn(4, V, generator=g) * 2.0
    s = S.Sampler(temperature=0.8, top_p=0.95)
    s.generator = torch.Generator().manual_seed(4)
    p = s.probs_rows(rows)
    a, b = int(p[0].argsort(descending=True)[0]), int(p[0].argsort(descending=True)[1])
    tokens, parents = [0, a, b, int(p[1].argmax())], [-1, 0, 0, 1]

    def run():
        n = 30000
        first = torch.zeros(V)
        second = torch.zeros(V)
        n_a = 0
        for _ in range(n):
            path, new = s.tree_walk_logits(rows, tokens, parents)
            first[new[0]] += 1
            if new[0] == a and len(new) > 1:
                second[new[1]] += 1
                n_a += 1
        e1 = (first / n - p[0]).abs().max().item()
        e2 = (second / max(n_a, 1) - p[1]).abs().max().item()
        assert e1 < 0.012 and e2 < 0.015, (e1, e2)
        return e1, e2
    e1, e2 = _nucleus(run)
    return f"branched tree, 30k walks: first token within {e1:.4f}, second (after a) within {e2:.4f}"


def test_tree_walk_logits_falls_back():
    """Seeded (keyed), q-carrying and min_p requests keep probs_rows + tree_walk."""
    rows = torch.randn(3, 10)
    called = []
    s = S.Sampler(temperature=0.7, top_p=0.9, seed=5)
    orig = s.row_draws if hasattr(s, "row_draws") else None

    def spy(r):
        called.append(1)
        return orig(r)

    def run():
        S.Sampler.row_draws, keep = (lambda self, r: spy(r)), S.Sampler.row_draws
        try:
            s.tree_walk_logits(rows, [0, 1, 2], [-1, 0, 1], start=10)                 # keyed
            u = S.Sampler(temperature=0.7, top_p=0.9, min_p=0.05)
            u.tree_walk_logits(rows, [0, 1, 2], [-1, 0, 1])                            # min_p
            v = S.Sampler(temperature=0.7, top_p=0.9)
            v.tree_walk_logits(rows, [0, 1, 2], [-1, 0, 1], q=[None, rows[0].softmax(-1), None])
            assert not called, "a fallback request took the nucleus draws"
            v.tree_walk_logits(rows, [0, 1, 2], [-1, 0, 1])
            assert called, "an unseeded det walk did not take the nucleus draws"
        finally:
            S.Sampler.row_draws = keep
    _nucleus(run)
    return "keyed / min_p / q requests fall back; an unseeded det walk takes the nucleus"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:48s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
