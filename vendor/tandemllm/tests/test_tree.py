"""The draft-tree invariants, which the GDN verify path will depend on and cannot check itself.

Run: `python -m pytest tests/ -q`, or `python tests/test_tree.py` with no pytest installed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tree import DraftTree, TreeBuilder  # noqa: E402


def test_chain_is_a_tree():
    t = DraftTree.chain(5, [1, 2, 3])
    t.check()
    assert t.n_draft == 3
    assert t.depths() == [0, 1, 2, 3]
    assert t.leaves() == [3]


def test_builder_emits_dfs_preorder():
    b = TreeBuilder(9)
    a = b.add(0, 1, 0.9)
    b.add(a, 11, 0.8)
    c = b.add(0, 2, 0.5)
    b.add(c, 22, 0.4)
    t = b.build()
    t.check()
    # depth-first: the whole subtree of the first child comes before the second child
    assert t.tokens == [9, 1, 11, 2, 22]
    assert t.parents == [-1, 0, 1, 0, 3]


def test_shared_prefixes_collapse():
    t = DraftTree.from_sequences(7, [([1, 2, 3], 0.6), ([1, 2, 9], 0.3), ([4], 0.1)])
    t.check()
    # 1 and 2 are shared, so five nodes rather than seven
    assert t.n_draft == 5
    assert t.tokens.count(1) == 1


def test_ancestor_mask_is_lower_triangular():
    t = DraftTree.from_sequences(0, [([1, 2, 3], 1.0), ([1, 5], 0.5), ([8, 9], 0.4)])
    t.check()
    m = t.ancestor_mask()
    n = len(t)
    for i in range(n):
        assert m[i][i]
        for j in range(i + 1, n):
            assert not m[i][j], "a node may never attend to a later one"
    # every node attends to exactly its own path
    for i in range(n):
        assert [j for j in range(n) if m[i][j]] == sorted(t.path(i))


def test_accepted_against_picks_the_matching_branch():
    t = DraftTree.from_sequences(0, [([1, 2, 3], 0.6), ([1, 5, 6], 0.3)])
    assert t.accepted_against([1, 5, 6, 7]) == 3
    assert t.accepted_against([1, 2, 9]) == 2
    assert t.accepted_against([4]) == 0
    assert t.accepted_against([]) == 0


def test_expected_accepted_is_the_sum_of_path_probabilities():
    t = DraftTree(tokens=[0, 1, 2, 3], parents=[-1, 0, 1, 0], scores=[1.0, 0.7, 0.5, 0.3])
    assert abs(t.expected_accepted() - 1.5) < 1e-9


def test_prune_keeps_ancestors_and_respects_the_budget():
    seqs = [([i, i + 100, i + 200], 1.0 / (i + 1)) for i in range(10)]
    t = DraftTree.from_sequences(0, seqs)
    p = t.prune(budget=6)
    p.check()
    assert p.n_draft <= 6
    # anything kept has its whole path kept, which `check` proves structurally
    assert p.expected_accepted() <= t.expected_accepted() + 1e-9


def test_prune_refuses_nodes_that_do_not_pay():
    # one strong branch and a long tail of near-zero candidates
    seqs = [([1, 2, 3, 4], 100.0)] + [([9, 9, i], 0.001) for i in range(20)]
    t = DraftTree.from_sequences(0, seqs)
    p = t.prune(budget=48)
    p.check()
    assert p.n_draft < t.n_draft, "the tail should not survive the marginal-value test"
    assert p.accepted_against([1, 2, 3, 4]) == 4


def test_merge_shares_common_nodes():
    a = DraftTree.from_sequences(0, [([1, 2, 3], 0.9)], source="ngram")
    b = DraftTree.chain(0, [1, 2, 7], source="mtp")
    m = a.merge(b)
    m.check()
    # 1 and 2 are shared: five nodes, not six
    assert m.n_draft == 4
    assert m.accepted_against([1, 2, 7]) == 3
    assert m.accepted_against([1, 2, 3]) == 3


def test_merge_requires_the_same_anchor():
    try:
        DraftTree.chain(1, [2]).merge(DraftTree.chain(3, [4]))
    except AssertionError:
        return
    raise AssertionError("merging trees with different anchors must fail")


def test_value_follows_the_measured_cost_curve():
    t = DraftTree.chain(0, [1, 2, 3, 4, 5, 6, 7, 8])
    # eight nodes all certain: 9 tokens for 149.1 + 8 * 1.896 ms
    expected = 9.0 / ((149.1 + 8 * 1.896) / 1000.0)
    assert abs(t.value() - expected) < 1e-6


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


def test_lattice_tree_keeps_the_greedy_path_and_spends_the_rest_best_first():
    """The block drafter's lattice as a tree: greedy first, then the highest-probability nodes.

    The greedy path is in the tree whatever it scores. That is not an aesthetic choice: the 10:28
    entry measured the exact maximiser of the selector's own objective accepting 3.22 tokens a block
    against the released greedy walk's 4.30, because a verify pays for the expected accepted PREFIX
    and slot 0 multiplies every later term.
    """
    import math
    import random

    from engine.tree import lattice_tree

    rng = random.Random(3)
    L, k = 7, 16
    cand = [[100 + l * 100 + c for c in range(k)] for l in range(L)]
    logp = [[[math.log(p) for p in _norm([rng.random() for _ in range(k)])]
             for _ in range(k)] for _ in range(L)]
    greedy = [rng.randrange(k) for _ in range(L)]

    for budget in (2, 7, 12, 16, 31):
        t = lattice_tree(1, cand, logp, greedy, budget)
        t.check()
        assert t.n_draft <= budget, (t.n_draft, budget)
        # the greedy path survives, as far as the budget reaches
        want = [cand[l][greedy[l]] for l in range(min(L, budget))]
        assert t.accepted_against(want) == len(want), (budget, t.accepted_against(want))
        # scores are path probabilities, so they never increase down a path
        for i in range(1, len(t.tokens)):
            assert t.scores[i] <= t.scores[t.parents[i]] + 1e-12
        # no node sits deeper than the lattice has slots
        assert max(t.depths()) <= L
    # a bigger budget is never a worse tree
    small = lattice_tree(1, cand, logp, greedy, 7)
    big = lattice_tree(1, cand, logp, greedy, 16)
    assert big.expected_accepted() >= small.expected_accepted()


def test_truncate_refuses_to_drop_the_anchor():
    """`truncate(n <= 0)` used to return a tree with no nodes at all -- the anchor gone."""
    t = DraftTree.chain(5, [1, 2, 3])
    for n in (0, -1):
        try:
            t.truncate(n)
            raise AssertionError(f"truncate({n}) must refuse")
        except ValueError as e:
            assert "anchor" in str(e)
    one = t.truncate(1)
    assert one.tokens == [5] and one.n_draft == 0, "n = 1 is the anchor alone"

def test_a_wide_budget_from_a_sixteen_slot_lattice():
    """Budgets 24, 32 and 64 from the wide drafter's fifteen slots, both builders. More
    budget buys branches, never depth; the tree stays DFS pre-order with ancestor-closed masks;
    merged with a lookup tree and pruned it keeps to the budget; and a bigger budget never loses
    the greedy path or expected acceptance."""
    import math
    import random

    from engine.tree import lattice_paths, lattice_tree

    rng = random.Random(9)
    L, k = 15, 16
    cand = [[1000 + l * 100 + c for c in range(k)] for l in range(L)]
    logp = [[[math.log(p) for p in _norm([rng.random() ** 3 for _ in range(k)])]
             for _ in range(k)] for _ in range(L)]
    # the released greedy walk: slot 0's argmax, then each slot's argmax in the row it came from
    greedy, row = [], 0
    for l in range(L):
        row = max(range(k), key=lambda c: logp[l][row][c])
        greedy.append(row)
    lookup = DraftTree.chain(1, [cand[0][1], 7, 8, 9, 10])
    for build in (lattice_tree, lattice_paths):
        prev = None
        for budget in (15, 23, 31, 63):
            t = build(1, cand, logp, greedy, budget)
            t.check()
            assert t.n_draft <= budget and max(t.depths()) <= L, (build.__name__, budget)
            m = t.ancestor_mask()
            for i in range(len(t.tokens)):
                assert all(m[i][j] == (j in t.path(i)) for j in range(len(t.tokens)))
            want = [cand[l][greedy[l]] for l in range(L)]
            assert t.accepted_against(want) == L, "the greedy path is always in"
            if prev is not None and build is lattice_tree:
                assert t.expected_accepted() >= prev.expected_accepted() - 1e-12
            prev = t
            merged = t.merge(lookup).prune(budget, per_node_ms=0.0, base_ms=100.0)
            merged.check()
            assert merged.n_draft <= budget


def _norm(xs):
    s = sum(xs)
    return [x / s for x in xs]


if __name__ == "__main__":
    _main()
