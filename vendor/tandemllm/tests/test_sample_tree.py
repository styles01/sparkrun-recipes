"""Sampled requests on the tree, exact -- on a CPU.

A sampled request's tree (`engine/tree.py::spine_tree`) is the drafter's SAMPLED chain as a spine,
each spine node carrying the distribution q its token was drawn from, with deterministic siblings
from the lattice. `Sampler.tree_walk` accepts a spine node with `min(1, p/q)`, otherwise draws from
the residual `(p - q)+` and follows any child carrying the draw: recursive rejection sampling with
the siblings as deterministic candidates. What has to hold:

  * the emitted tokens follow the target exactly: histograms of the first three emitted tokens over
    N = 30,000 walks on a Markov target match the chain's own probabilities, whatever q looks like
    and however many siblings a level has -- and a construction that picks its spine by the draw
    (the bias the causal builder exists to avoid) FAILS the same test, so the test can see it;
  * the builder is causal: siblings at depth i depend on the spine only down to depth i, the size is
    fixed before the draw, every spine node carries its q and no sibling does;
  * anything that reshapes a tree drops q (the walk is then the plain one, exact for any tree);
  * a seeded request ignores q and takes the keyed walk, so a seed still reproduces a request;
  * the router makes every choice on the deterministic tree, never on the sampled spine.

Run: python tests/test_sample_tree.py
"""

from __future__ import annotations

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.sample import Sampler  # noqa: E402
from engine.tree import DraftTree, lattice_tree, level_quota, spine_tree  # noqa: E402

V = 6          # vocabulary of the toy target
ANCHOR = 0


def _markov(seed=0, sharp=1.5):
    """P[a] = the target's next-token distribution after token a (a Markov chain over V tokens)."""
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(V, V, generator=g, dtype=torch.float64) * sharp, dim=-1)


def _lattice(L, seed=1):
    """A block drafter's lattice: every token a candidate at every slot, a pairwise log-score."""
    g = torch.Generator().manual_seed(seed)
    cand = [list(range(V)) for _ in range(L)]
    logp = torch.log_softmax(torch.randn(L, V, V, generator=g, dtype=torch.float64) * 2.0,
                             dim=-1).tolist()
    return cand, logp


def _qrows(L, seed=2, sharp=3.0):
    """The drafter's proposal rows, one a slot, deliberately unlike the target."""
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(L, V, generator=g, dtype=torch.float64) * sharp, dim=-1)


def _walk_once(s, P, tree):
    dists = torch.stack([P[t] for t in tree.tokens]).float()
    q = None if tree.q is None else [None if r is None else r.float() for r in tree.q]
    _, new = s.tree_walk(dists, tree.tokens, tree.parents, q=q)
    return new


def _emitted_prefixes(build, P, n, depth=3, seed=1234):
    """Walk `n` trees built by `build(gen)`; each walk's emitted tokens, extended to `depth` by
    sampling the target directly (what the next block would do), as a tuple."""
    s = Sampler(temperature=1.0, seed=seed)           # no `start`: the unkeyed walk, q honoured
    gen = torch.Generator().manual_seed(seed + 1)     # the drafter's own randomness
    out = []
    for _ in range(n):
        new = list(_walk_once(s, P, build(gen)))
        while len(new) < depth:
            new.append(int(torch.multinomial(P[new[-1]], 1, generator=gen)))
        out.append(tuple(new[:depth]))
    return out


def _worst(prefixes, P, depth=3):
    """The largest gap between the empirical probability of an emitted prefix (lengths 1..depth)
    and the target's."""
    n = len(prefixes)
    worst = 0.0
    for d in range(1, depth + 1):
        counts: dict = {}
        for p in prefixes:
            counts[p[:d]] = counts.get(p[:d], 0) + 1
        for seq in itertools.product(range(V), repeat=d):
            want, prev = 1.0, ANCHOR
            for t in seq:
                want *= float(P[prev, t])
                prev = t
            worst = max(worst, abs(counts.get(seq, 0) / n - want))
    return worst


def _spine_builder(L, quota, q, cand, logp):
    def build(gen):
        spine = [int(torch.multinomial(q[l], 1, generator=gen)) for l in range(L)]
        return spine_tree(ANCHOR, spine, list(q), cand, logp, quota)
    return build


N = 30000
TOL = 0.012          # ~4 sigma at p = 0.5, N = 30,000


def test_the_spine_tree_walk_emits_the_target_exactly():
    P = _markov()
    L = 4
    cand, logp = _lattice(L)
    worst = 0.0
    for quota, q in (([2, 1, 1, 0], _qrows(L)), ([0, 0, 0, 0], _qrows(L, seed=5, sharp=0.5)),
                     ([4, 2, 0, 1], _qrows(L, seed=6, sharp=6.0))):
        w = _worst(_emitted_prefixes(_spine_builder(L, quota, q, cand, logp), P, N), P)
        assert w < TOL, (quota, w)
        worst = max(worst, w)
    return f"3 sibling quotas x 3 kinds of q, {N} walks each: worst prefix gap {worst:.4f} < {TOL}"


def test_a_spine_chosen_by_its_own_draw_is_caught():
    """The negative control: keep the spine's first node only when the drafter drew its top token
    (a choice made BY the draw), and the same histogram test fails -- so it can see the bias."""
    P = _markov()
    L = 3
    cand, logp = _lattice(L)
    q = _qrows(L, seed=7, sharp=1.0)
    top = int(P[ANCHOR].argmax())
    q[0] = torch.full((V,), 0.3 / (V - 1), dtype=torch.float64)
    q[0, top] = 0.7                       # the drafter likes the target's own favourite

    def build(gen):
        tree = _spine_builder(L, [1, 1, 0], q, cand, logp)(gen)
        if tree.spine_chain().tokens[1] != top:
            tree.q = None                      # the draw decided: walk this one without q
        return tree

    w = _worst(_emitted_prefixes(build, P, N, seed=99), P)
    assert w > 3 * TOL, w
    return f"the biased construction misses by {w:.4f} (> {3 * TOL:.3f})"


def test_the_builder_is_causal_and_sized_before_the_draw():
    L = 5
    cand, logp = _lattice(L, seed=3)
    q = list(_qrows(L))
    quota = [3, 1, 2, 0, 1]
    a = spine_tree(ANCHOR, [1, 2, 3, 4, 5], q, cand, logp, quota)
    b = spine_tree(ANCHOR, [1, 2, 3, 0, 0], q, cand, logp, quota)
    for t in (a, b):
        t.check()
        assert t.n_draft == L + sum(quota)
        d = t.depths()
        spine = [i for i in range(1, len(t.tokens)) if t.q[i] is not None]
        assert [d[i] for i in spine] == list(range(1, L + 1))
        for i in range(1, len(t.tokens)):
            if t.q[i] is None:                                   # a sibling: a leaf beside the spine
                assert t.parents[i] == 0 or t.q[t.parents[i]] is not None
                assert i in t.leaves()
                assert t.tokens[i] != t.tokens[next(c for c in spine if d[c] == d[i])]

    def upto(t, depth):
        d = t.depths()
        keep = sorted(i for i in range(len(t.tokens)) if d[i] <= depth)
        return [(d[i], t.tokens[t.parents[i]] if i else -1, t.tokens[i]) for i in keep]
    # the two spines agree to depth 3: everything down to depth 3 is the same, siblings included
    assert sorted(upto(a, 3)) == sorted(upto(b, 3))
    assert sorted(upto(a, 4)) != sorted(upto(b, 4))
    # the shape comes from the deterministic tree of the same lattice
    det = lattice_tree(ANCHOR, cand, logp, [0] * L, 12)
    assert sum(level_quota(det)) == det.n_draft - max(det.depths())
    return "sizes fixed by the quota, spine carries q, siblings are leaves, depth <= i sees only spine <= i"


def test_reshaping_a_tree_drops_q():
    L = 4
    cand, logp = _lattice(L)
    t = spine_tree(ANCHOR, [1, 2, 3, 4], list(_qrows(L)), cand, logp, [2, 1, 0, 0])
    assert t.truncate(len(t.tokens)) is t and t.q is not None
    assert t.truncate(4).q is None
    assert t.subset(set(range(len(t.tokens)))).q is None
    assert t.merge(DraftTree.chain(ANCHOR, [5, 5])).q is None
    assert t.prune(3).q is None
    c = t.spine_chain()
    assert c.tokens == [ANCHOR, 1, 2, 3, 4] and all(r is not None for r in c.q[1:])
    return "truncate/subset/merge/prune return trees without q; the spine chain keeps it"


def test_a_seeded_request_ignores_q_and_takes_the_keyed_walk():
    P = _markov(seed=4)
    L = 4
    cand, logp = _lattice(L)
    q = _qrows(L)
    t = spine_tree(ANCHOR, [1, 2, 3, 4], list(q), cand, logp, [2, 1, 1, 0])
    dists = torch.stack([P[x] for x in t.tokens]).float()
    for seed in range(20):
        a = Sampler(temperature=1.0, seed=seed).tree_walk(dists, t.tokens, t.parents, start=50,
                                                          q=[None if r is None else r.float()
                                                             for r in t.q])
        b = Sampler(temperature=1.0, seed=seed).tree_walk(dists, t.tokens, t.parents, start=50)
        assert a == b, seed
    return "20 seeds: the walk with q == the keyed walk without it"


class _SpineHead:
    """A block drafter that returns a sampled spine tree and the deterministic tree it was sized on,
    as DFlash2Drafter.propose_tree does under a sampled request."""

    def __init__(self, spine):
        self.spine = spine
        L = len(spine)
        self.cand, self.logp = _lattice(L, seed=8)
        self.q = list(_qrows(L))
        self.last_det_tree = None

    def reset(self):
        pass

    def propose_tree(self, context, budget):
        det = lattice_tree(context[-1], self.cand, self.logp, [0] * len(self.spine), budget)
        self.last_det_tree = det
        return spine_tree(context[-1], self.spine, self.q, self.cand, self.logp, level_quota(det))


class _NoLookup:
    max_depth = 8

    def propose_tree(self, context, k):
        return None

    def reset(self):
        pass

    def observe(self, tokens):
        pass


def test_the_router_chooses_on_the_deterministic_tree_not_the_draw():
    from engine.router import MergedRouter
    labels, trees = [], []
    for spine in ([1, 2, 3, 4, 5, 0, 1], [5, 5, 5, 5, 5, 5, 5], [0, 1, 0, 1, 0, 1, 0]):
        r = MergedRouter(_NoLookup(), _SpineHead(spine), node_budget=12)
        t = r.propose_tree([7, ANCHOR], 12)
        labels.append(r.last)
        trees.append(t)
        # what the router learns from is the deterministic tree too
        assert r.last_head_tree is r.mtp.last_det_tree
    assert len(set(labels)) == 1, labels
    if labels[0] in ("mtp", "chain"):
        assert all(t.q is not None for t in trees)
    return f"three different draws, one choice ({labels[0]}), the spine tree returned where it won"


def test_the_block_drafter_builds_the_spine_tree_from_its_sample_and_lattice():
    """DFlash2Drafter.propose_tree: under a sampled request it asks `_tokens_from` for the lattice
    beside the sample, returns the spine tree (q on the sampled chain) sized on the deterministic
    tree of the same lattice, and leaves that tree for the router; greedy is what it was."""
    from engine.drafters.dflash2 import DFlash2Drafter
    L, k = 7, 16
    g = torch.Generator().manual_seed(3)
    cand = torch.randperm(1000, generator=g)[: L * k].view(L, k)
    scores = torch.randn(L, k, k, generator=g)
    scores[0] = scores[0, 0].expand(k, k).clone()
    qrows = [torch.softmax(torch.randn(1000, generator=g), -1) for _ in range(L)]

    d = object.__new__(DFlash2Drafter)
    d.cfg = type("C", (), {"block_size": L + 1})()
    d.tree_temp, d.tree_mode, d._want_lattice, d.last_det_tree = 1.0, "nodes", False, None
    seen = {}

    def propose_steps(context, n, logp=False):
        seen["want"] = d._want_lattice
        d._lattice, d._cand_host = (cand, scores), None
        sampled = d.sampler is not None
        d.last_q = list(qrows) if sampled else None
        # the sample: slot l's second candidate, which the greedy walk would not take
        return [int(cand[l, 1]) for l in range(n)] if sampled else [int(cand[l, 0]) for l in range(n)]
        yield                  # a generator, as the drafter's own since (propose_tree_steps)
    d._propose_steps = propose_steps

    d.sampler = Sampler(temperature=0.7)
    t = d.propose_tree([5, 9], 15)
    assert seen["want"] and not d._want_lattice
    t.check()
    assert d.last_det_tree is not None and d.last_det_tree.q is None
    assert t.n_draft == d.last_det_tree.n_draft == 15
    spine = t.spine_chain()
    assert spine.tokens[1:] == [int(cand[l, 1]) for l in range(L)]
    assert all(a is b for a, b in zip(spine.q[1:], qrows))
    assert d.last_q is None
    d.sampler = None
    greedy = d.propose_tree([5, 9], 15)
    assert greedy.q is None and not seen["want"]
    return "sampled: spine = the sample with its q rows, 15 nodes like the lattice's own tree; greedy unchanged"


def test_the_served_loop_passes_q_and_a_seed_still_reproduces():
    """Server/app.py with `sampled_tree` = mixed / det: an unseeded request runs through the q walk,
    and a seeded one is the drafter-less keyed sample, tree or not."""
    import test_app_loop as T
    from server import app

    class QTree(T.QDrafter):
        def propose_tree(self, ctx, k):
            n = min(k, self.width)
            spine = self.propose(ctx, n)
            if self.last_q is None:
                return DraftTree.chain(ctx[-1], spine)
            L = len(spine)
            cand = [list(range(97)) for _ in range(L)]
            logp = [[[0.0] * 97 for _ in range(97)] for _ in range(L)]
            return spine_tree(ctx[-1], spine, list(self.last_q), cand, logp, [2] + [0] * (L - 1))

    ref = T._sampled(None)
    for mode in ("mixed", "det"):
        T.serve(QTree(5), tree=True)
        app.STATE["sampled_tree"] = mode
        s = app.Sampler(temperature=0.9, top_p=0.95, seed=42)
        out = list(app.generate_stream(torch.tensor([5, 6, 7, 8, 9]), 48, set(), sampler=s))
        assert out == ref, (mode, out, ref)
        T.serve(QTree(5), tree=True)
        app.STATE["sampled_tree"] = mode
        s = app.Sampler(temperature=0.9, top_p=0.95)
        out = list(app.generate_stream(torch.tensor([5, 6, 7, 8, 9]), 24, set(), sampler=s))
        assert len(out) == 24, (mode, out)
    return "mixed and det: seeded == the drafter-less keyed run; unseeded runs the q walk"



def test_the_cut_spine_tree_walk_emits_the_target_exactly():
    """SAMPLED_CUT (`DraftTree.cut_like`): the spine tree cut to a deterministic tree's depth
    profile (fewer levels, fewer siblings) still emits the target exactly, and keeps q."""
    P = _markov()
    L = 4
    cand, logp = _lattice(L)
    worst = 0.0
    for budget in (2, 4, 7):
        det = lattice_tree(ANCHOR, cand, logp, [0] * L, budget)
        prof = sorted(det.depths()[1:])
        for quota, q in (([2, 1, 1, 0], _qrows(L)), ([4, 2, 0, 1], _qrows(L, seed=6, sharp=6.0))):
            base = _spine_builder(L, quota, q, cand, logp)

            def build(gen, base=base, det=det):
                t = base(gen).cut_like(det)
                t.check()
                assert t.q is not None and t.n_draft <= det.n_draft
                assert all(max(t.depths()) <= max(det.depths()) for _ in (0,))
                return t
            w = _worst(_emitted_prefixes(build, P, N), P)
            assert w < TOL, (budget, quota, w)
            worst = max(worst, w)
    return f"3 cut profiles x 2 quotas, {N} walks each: worst prefix gap {worst:.4f} < {TOL}"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:62s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
