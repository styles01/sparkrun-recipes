"""A draft tree, and the one layout property that lets a linear-attention model verify it.

A chain drafter proposes `t1 t2 t3`; a tree drafter proposes `t1 -> {t2a, t2b}`, and on this board
that is nearly free. The measured verify curve is `V(N) = 149.1 ms + 1.896 ms * N`, so a node is
worth adding when it has about a 5 % chance of extending the accepted prefix.
That is a much lower bar than on a server, where a verify pass is compute-bound and width costs
real time.

**The layout rule.** Nodes are stored in DFS pre-order, so every node's index is greater than its
parent's and each subtree occupies a contiguous range. That is not a convenience. For the 48 Gated
DeltaNet layers, tree verification is a *mask change* to the UT factorisation the chunked kernel
already runs, and the forward substitution in `solve_tril` is only valid if the ancestor-masked
matrix is strictly lower triangular -- which holds exactly when every ancestor has a lower index
than its descendant. DFS pre-order guarantees it; breadth-first order does not.

This module is the contract two drafters have to agree on before their candidates can share one
verify call: the lookup drafter's tree (`engine/drafters/ngram.py`) and the block drafter's lattice.
`merge` is the graft operation -- one tree, two sources, rather than a router choosing between them.

Nothing here touches the GPU or torch, so it is all testable on a laptop; the mask this produces is
consumed by the verify path, which is not written yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Verify cost on this board (FP8 weights). `N` counts drafted nodes, not
# the anchor. Overridden by the caller for other weight sets.
VERIFY_BASE_MS = 149.1
VERIFY_PER_NODE_MS = 1.896


@dataclass
class DraftTree:
    """Candidate continuations of one anchor token, in DFS pre-order.

    `tokens[0]` is the anchor -- the last token the model has already committed to -- and carries
    parent -1. Every other node is a drafted token. `len(tree) - 1` is what the cost model calls N.
    """

    tokens: list[int]
    parents: list[int]
    scores: list[float] = field(default_factory=list)
    source: list[str] = field(default_factory=list)
    # per node, the distribution its token was SAMPLED from (a sampled request's spine), or
    # None for a node placed deterministically. Only `spine_tree` sets it, and every operation that
    # reshapes a tree (subset, truncate, merge, prune) returns a tree without it: the walk then
    # treats every node as deterministic, which is exact for any tree (engine/sample.py).
    q: list | None = None

    def __post_init__(self) -> None:
        if not self.scores:
            self.scores = [1.0] + [0.0] * (len(self.tokens) - 1)
        if not self.source:
            self.source = ["root"] + ["?"] * (len(self.tokens) - 1)

    # --- basics -------------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def n_draft(self) -> int:
        """Drafted nodes, excluding the anchor. This is `N` in the verify cost model."""
        return len(self.tokens) - 1

    def check(self) -> None:
        """Assert the invariants the verify path depends on. Cheap enough to run in tests only."""
        assert self.parents[0] == -1, "node 0 must be the anchor"
        assert len(self.parents) == len(self.tokens) == len(self.scores) == len(self.source)
        for i in range(1, len(self.tokens)):
            assert 0 <= self.parents[i] < i, f"node {i} parent {self.parents[i]} breaks pre-order"
        # contiguity of subtrees, which is what "DFS pre-order" adds over "parent < child"
        for i in range(len(self.tokens)):
            end = i + 1
            while end < len(self.tokens) and self._is_descendant(end, i):
                end += 1
            for j in range(end, len(self.tokens)):
                assert not self._is_descendant(j, i), f"subtree of {i} is not contiguous at {j}"

    def _is_descendant(self, node: int, of: int) -> bool:
        while node > of:
            node = self.parents[node]
        return node == of

    def depths(self) -> list[int]:
        """Depth of each node; the anchor is 0. These are the position offsets for RoPE."""
        d = [0] * len(self.tokens)
        for i in range(1, len(self.tokens)):
            d[i] = d[self.parents[i]] + 1
        return d

    def path(self, node: int) -> list[int]:
        """Node indices from the anchor down to `node`, inclusive."""
        out = []
        while node >= 0:
            out.append(node)
            node = self.parents[node]
        return list(reversed(out))

    def ancestor_mask(self) -> list[list[bool]]:
        """`m[i][j]` is True when j is i or an ancestor of i: what each node may attend to.

        Strictly lower triangular plus the diagonal, which is the property `solve_tril` needs.
        """
        n = len(self.tokens)
        m = [[False] * n for _ in range(n)]
        for i in range(n):
            j = i
            while j >= 0:
                m[i][j] = True
                j = self.parents[j]
        return m

    def conv_windows(self, width: int) -> list[list[int]]:
        """For each node, the `width` columns its depthwise causal convolution reads.

        Indices address `cat([conv_state, block], -1)`: columns `0 .. width-2` are the convolution
        state (oldest first, so column `width-2` is the token immediately before the block's anchor)
        and column `width-1+j` is node j. Each row is oldest first with the node itself last, which
        is the order `F.conv1d` applies the kernel's taps in.

        On a chain this reproduces the ordinary sliding window exactly. On a tree it follows the
        node's own ancestors, which in DFS pre-order are not adjacent columns -- that is the whole
        difference, and it is why the convolution needs a gather rather than a shift.
        """
        out: list[list[int]] = []
        for i in range(len(self.tokens)):
            chain: list[int] = []
            cur = i
            while cur >= 0 and len(chain) < width:
                chain.append(width - 1 + cur)
                cur = self.parents[cur]
            for j in range(width - len(chain)):
                chain.append(width - 2 - j)
            out.append(list(reversed(chain)))
        return out

    def leaves(self) -> list[int]:
        has_child = [False] * len(self.tokens)
        for p in self.parents[1:]:
            has_child[p] = True
        return [i for i, h in enumerate(has_child) if not h]

    # --- construction -------------------------------------------------------------------------

    @staticmethod
    def chain(anchor: int, tokens: list[int], scores: list[float] | None = None,
              source: str = "chain") -> "DraftTree":
        n = len(tokens)
        scores = list(scores) if scores else [1.0] * n
        return DraftTree(tokens=[anchor] + list(tokens),
                         parents=[-1] + list(range(n)),
                         scores=[1.0] + scores,
                         source=["root"] + [source] * n)

    @staticmethod
    def from_sequences(anchor: int, seqs: list[tuple[list[int], float]],
                       source: str = "tree") -> "DraftTree":
        """Build the prefix tree of a set of scored continuations.

        A node's score is the largest score of any sequence passing through it, which for the
        frequency scores this engine uses is the path probability of its most likely completion.
        """
        b = TreeBuilder(anchor)
        for toks, score in seqs:
            node = 0
            for t in toks:
                node = b.add(node, t, score, source)
        return b.build()

    # --- operations ---------------------------------------------------------------------------

    def expected_accepted(self) -> float:
        """Expected accepted draft tokens, under the scores read as path probabilities.

        A drafted token is accepted only if every token above it on its path is, so the expected
        accepted length of a tree is the sum of its nodes' path probabilities. This is the quantity
        the firing policy and the node budget are both decided by.
        """
        return sum(self.scores[1:])

    def value(self, draft_ms: float = 0.0, base_ms: float = VERIFY_BASE_MS,
              per_node_ms: float = VERIFY_PER_NODE_MS) -> float:
        """Expected tokens per second if every step looked like this one."""
        gained = self.expected_accepted() + 1.0        # the model's own token is always accepted
        cost = (base_ms + per_node_ms * self.n_draft + draft_ms) / 1000.0
        return gained / cost if cost > 0 else 0.0

    def prune(self, budget: int, per_node_ms: float = VERIFY_PER_NODE_MS,
              base_ms: float = VERIFY_BASE_MS, draft_ms: float = 0.0) -> "DraftTree":
        """Keep at most `budget` nodes, best-first, and only while each one pays for itself.

        The marginal rule is the one in RESEARCH section 0.2: with throughput `tau / (V + d)`, a node
        earns its 1.896 ms when it adds more than `tau / (V + d) * per_node_ms` expected tokens.
        Nodes are admitted in descending score with their ancestors, which keeps DFS pre-order
        reachable and never admits a node whose parent was rejected.
        """
        if self.n_draft <= 0:
            return self
        order = sorted(range(1, len(self.tokens)), key=lambda i: -self.scores[i])
        keep = {0}
        gained = 1.0
        cost_ms = base_ms + draft_ms
        for i in order:
            if len(keep) - 1 >= budget:
                break
            chain = [n for n in self.path(i) if n not in keep]
            add = sum(self.scores[n] for n in chain)
            if len(keep) - 1 + len(chain) > budget:
                continue
            threshold = (gained / max(cost_ms, 1e-6)) * per_node_ms * len(chain)
            if add < threshold:
                continue
            keep.update(chain)
            gained += add
            cost_ms += per_node_ms * len(chain)
        return self.subset(keep)

    def truncate(self, n: int) -> "DraftTree":
        """The first `n` nodes, or self if it already fits. The row clamp.

        DFS pre-order makes a prefix ancestor-closed (every parent has a lower index), so cutting
        a tree short leaves a valid tree: the dropped nodes are simply not verified, which costs
        at most acceptance and never correctness. The decode loops call this when a block would
        cross `max_len`, because the budget they hand the drafter counts output tokens while the
        KV write counts rows -- the anchor's own row for a chain, every node for a tree.

        `n < 1` raises: there is no tree without its anchor, and a caller with no row left for
        the anchor has already overrun the window. The loops never get here -- a request
        is clamped to leave the anchor its row -- so this names the bug if one ever does.
        """
        if n < 1:
            raise ValueError(f"truncate({n}): a tree keeps at least its anchor")
        if n >= len(self.tokens):
            return self
        return self.subset(set(range(n)))

    def subset(self, keep: set[int]) -> "DraftTree":
        """The tree restricted to `keep`, re-indexed in DFS pre-order. `keep` must be ancestor-closed."""
        children: dict[int, list[int]] = {}
        for i in range(1, len(self.tokens)):
            if i in keep:
                children.setdefault(self.parents[i], []).append(i)
        tokens: list[int] = []
        parents: list[int] = []
        scores: list[float] = []
        source: list[str] = []
        remap: dict[int, int] = {}

        def walk(old: int, parent_new: int) -> None:
            new = len(tokens)
            remap[old] = new
            tokens.append(self.tokens[old])
            parents.append(parent_new)
            scores.append(self.scores[old])
            source.append(self.source[old])
            for c in sorted(children.get(old, []), key=lambda c: -self.scores[c]):
                walk(c, new)

        walk(0, -1)
        return DraftTree(tokens, parents, scores, source)

    def cut_like(self, det: "DraftTree") -> "DraftTree":
        """A sampled spine tree (`spine_tree`) cut to the depth profile of `det`, the deterministic
        tree the router's staircase cut chose for the same lattice: spine nodes to det's depth, and
        at each depth at most (det's nodes there - 1) siblings, the best-scored first. Every kept
        count comes from det, which never looks at the draw, and a sibling's score depends only
        on the spine above it, so the spine stays distributed as its q and the walk stays exact
        (as `spine_tree`'s own quota). q is carried over. A tree without q is returned as is."""
        if self.q is None or det is None or det.n_draft == 0:
            return self
        from collections import Counter
        want = Counter(det.depths()[1:])
        maxd = max(want)
        dep = self.depths()
        keep = {0}
        used: Counter = Counter()
        for i in range(1, len(self.tokens)):
            if self.q[i] is not None and dep[i] <= maxd and self.parents[i] in keep:
                keep.add(i)
                used[dep[i]] += 1
        sibs = sorted((i for i in range(1, len(self.tokens)) if self.q[i] is None),
                      key=lambda i: -self.scores[i])
        for i in sibs:
            d = dep[i]
            if self.parents[i] in keep and used[d] < want.get(d, 0):
                keep.add(i)
                used[d] += 1
        if len(keep) == len(self.tokens):
            return self
        order = [i for i in range(len(self.tokens)) if i in keep]   # a pre-order subsequence
        new = {old: n for n, old in enumerate(order)}
        return DraftTree([self.tokens[i] for i in order],
                         [-1] + [new[self.parents[i]] for i in order[1:]],
                         [self.scores[i] for i in order], [self.source[i] for i in order],
                         q=[self.q[i] for i in order])

    def merge(self, other: "DraftTree", prefer: str = "max") -> "DraftTree":
        """Graft `other` onto this tree. Both must share the same anchor.

        Where the two sources propose the same token at the same place the node is shared, so a
        merged tree costs strictly less than the sum of its parts. This is the operation that
        replaces the router: instead of choosing a drafter per step, both put candidates in the
        same verify call and the node budget decides how much each gets.
        """
        assert self.tokens[0] == other.tokens[0], "trees must share an anchor"
        b = TreeBuilder(self.tokens[0])

        def graft(tree: "DraftTree", src_node: int, dst_node: int) -> None:
            for c in tree.children_of(src_node):
                n = b.add(dst_node, tree.tokens[c], tree.scores[c], tree.source[c], prefer=prefer)
                graft(tree, c, n)

        graft(self, 0, 0)
        graft(other, 0, 0)
        return b.build()

    def children_of(self, node: int) -> list[int]:
        return [i for i in range(1, len(self.tokens)) if self.parents[i] == node]

    def spine_chain(self) -> "DraftTree":
        """The nodes that carry a q (a sampled spine), as a chain that keeps them -- the
        q-aware chain of in tree form."""
        path, node = [], 0
        while True:
            nxt = next((c for c in self.children_of(node) if self.q[c] is not None), None)
            if nxt is None:
                break
            path.append(nxt)
            node = nxt
        out = DraftTree.chain(self.tokens[0], [self.tokens[i] for i in path],
                              scores=[self.scores[i] for i in path], source="df2-spine")
        out.q = [None] + [self.q[i] for i in path]
        return out

    # --- offline evaluation --------------------------------------------------------------------

    def accepted_against(self, continuation: list[int]) -> int:
        """How many drafted tokens a greedy verify would accept, given what the target really wrote.

        Greedy verification accepts the unique root-to-node path whose tokens match the target's own
        continuation, and stops at the first place the tree has no matching child. This makes offline
        evaluation exact rather than an estimate -- see tools/sim_draft.py.
        """
        node, n = 0, 0
        for want in continuation:
            nxt = next((c for c in self.children_of(node) if self.tokens[c] == want), None)
            if nxt is None:
                break
            node, n = nxt, n + 1
        return n


class TreeBuilder:
    """Accumulates nodes in any order and emits them in DFS pre-order."""

    def __init__(self, anchor: int):
        self.tokens = [anchor]
        self.parents = [-1]
        self.scores = [1.0]
        self.source = ["root"]
        self.kids: list[dict[int, int]] = [{}]

    def add(self, parent: int, token: int, score: float, source: str = "?",
            prefer: str = "max") -> int:
        existing = self.kids[parent].get(token)
        if existing is not None:
            if prefer == "max" and score > self.scores[existing]:
                self.scores[existing] = score
            elif prefer == "sum":
                self.scores[existing] = min(1.0, self.scores[existing] + score)
            return existing
        idx = len(self.tokens)
        self.tokens.append(token)
        self.parents.append(parent)
        self.scores.append(score)
        self.source.append(source)
        self.kids.append({})
        self.kids[parent][token] = idx
        return idx

    def build(self) -> DraftTree:
        tokens: list[int] = []
        parents: list[int] = []
        scores: list[float] = []
        source: list[str] = []

        def walk(old: int, parent_new: int) -> None:
            new = len(tokens)
            tokens.append(self.tokens[old])
            parents.append(parent_new)
            scores.append(self.scores[old])
            source.append(self.source[old])
            for _, c in sorted(self.kids[old].items(), key=lambda kv: -self.scores[kv[1]]):
                walk(c, new)

        walk(0, -1)
        return DraftTree(tokens, parents, scores, source)


def lattice_paths(anchor: int, cand: list[list[int]], logp: list[list[list[float]]],
                  greedy: list[int], budget: int, source: str = "df2") -> DraftTree:
    """The same lattice, spent on whole BRANCHES instead of on individual nodes.

    `lattice_tree` below buys the highest-probability nodes there are, and on this lattice they are
    nearly all siblings near the root. That is the wrong shape, and the reason is worth stating:
    **an alternative node only pays if it has descendants.** A leaf hung beside slot 0 extends the
    accepted path by exactly one token in the rare case the drafter's top-1 was wrong, and by
    nothing the rest of the time. What is wanted where slot 0 is wrong is a whole second
    continuation, so that the block still accepts four or five tokens instead of one.

    So each unit of budget here buys a branch: pop the best frontier alternative, then follow the
    released greedy walk from it to the end of the lattice. On a seven-slot lattice with a budget of
    sixteen that is the greedy path, a second full path rooted at the best alternative, and a
    fragment of a third.

    This is k-best over the lattice in the sense the drafter's own selector means it, rather than
    best-first over marginals. Which is worth more is a measurement, and both are here so that it
    can be one.
    """
    import heapq
    import math

    b = TreeBuilder(anchor)
    L = len(cand)
    k = len(cand[0]) if L else 0
    if not L:
        return b.build()
    state = {"n": 0, "tie": 0}
    frontier: list[tuple] = []

    def walk_from(parent: int, slot: int, first: int, lp0: float, tag: str) -> None:
        """Take candidate `first` at `slot` under `parent`, then greedy-walk to the end."""
        row, node, lp = first, parent, lp0
        for l in range(slot, L):
            if state["n"] >= budget:
                return
            c = row if l == slot else int(max(range(k), key=lambda x: logp[l][row][x]))
            parent_lp = lp
            lp = lp + logp[l][row][c]
            here, node = node, b.add(node, cand[l][c], math.exp(lp), tag)
            state["n"] += 1
            for c2 in range(k):                      # every sibling becomes a branch root
                if c2 != c:
                    state["tie"] += 1
                    heapq.heappush(frontier, (-(parent_lp + logp[l][row][c2]), state["tie"],
                                              here, l, c2, parent_lp))
            row = c

    walk_from(0, 0, greedy[0] if greedy else 0, 0.0, f"{source}-greedy")
    while frontier and state["n"] < budget:
        negp, _, parent, slot, c, parent_lp = heapq.heappop(frontier)
        walk_from(parent, slot, c, parent_lp, f"{source}-alt")
    return b.build()


def level_quota(tree: "DraftTree") -> list[int]:
    """Nodes a tree holds at each depth past the first, per level: [depth 1 count - 1, ...]. The
    shape a sampled spine tree copies from the deterministic tree the same lattice builds."""
    d = tree.depths()
    out = [0] * max(d)
    for x in d[1:]:
        out[x - 1] += 1
    return [max(0, c - 1) for c in out]


def spine_tree(anchor: int, spine: list[int], qrows: list, cand: list[list[int]],
               logp: list[list[list[float]]], quota: list[int], source: str = "df2") -> DraftTree:
    """A sampled request's tree -- the drafter's SAMPLED chain as the spine, each spine node
    carrying the distribution it was drawn from, and deterministic siblings from the lattice.

    The walk accepts a spine node by rejection sampling against its q (`min(1, p/q)`), then draws
    from the residual and follows any child carrying the draw (engine/sample.py `tree_walk`). That is
    recursive rejection sampling with the siblings as deterministic candidates, and it is exact only
    if, wherever the walk stands on the spine, the next spine token is still distributed as its q.
    So the construction is causal: the siblings at depth i depend on the lattice and on the spine
    tokens at depths i-1 and i (the candidates at slot i-1 given the predecessor, minus the spine's
    own token), never on a deeper spine token, and every spine node is always in the tree. The
    per-level counts `quota` are decided before the spine is drawn (`level_quota` of the
    deterministic tree the same lattice builds), so the size never depends on the draw either:
    `len(spine) + sum(quota)` drafted nodes at most. Siblings are leaves. A spine token that is not
    among its slot's candidates gives the level below it no siblings (there is no lattice row for
    it), which is again decided by tokens at depth <= i.
    """
    import math

    b = TreeBuilder(anchor)
    L = min(len(spine), len(cand))
    qs: dict[int, object] = {}
    node, lp, row = 0, 0.0, 0
    for slot in range(L):
        tok = spine[slot]
        here = node
        if row is not None:
            want = quota[slot] if slot < len(quota) else 0
            order = sorted(range(len(cand[slot])), key=lambda c: -logp[slot][row][c])
            added = 0
            for c in order:
                if added >= want:
                    break
                if cand[slot][c] == tok:
                    continue
                b.add(here, cand[slot][c], math.exp(lp + logp[slot][row][c]), f"{source}-alt")
                added += 1
        ci = cand[slot].index(tok) if tok in cand[slot] else None
        if row is not None and ci is not None:
            lp += logp[slot][row][ci]
        node = b.add(here, tok, math.exp(lp), f"{source}-spine")
        qs[node] = qrows[slot]
        row = ci
    # the builder emits DFS pre-order; carry each spine node's q to its new index
    built = b.build()
    order = _builder_order(b)
    built.q = [qs.get(old) for old in order]
    return built


def _builder_order(b: "TreeBuilder") -> list[int]:
    """The builder's node ids in the order `build()` emits them."""
    out: list[int] = []

    def walk(old: int) -> None:
        out.append(old)
        for _, c in sorted(b.kids[old].items(), key=lambda kv: -b.scores[kv[1]]):
            walk(c)

    walk(0)
    return out


def lattice_tree(anchor: int, cand: list[list[int]], logp: list[list[list[float]]],
                 greedy: list[int], budget: int, source: str = "df2") -> DraftTree:
    """A tree out of a block drafter's lattice: the greedy path, then the best nodes around it.

    `cand[l][c]` is the c-th candidate token at slot l, `logp[l][p][c]` is the log probability of
    candidate c given that slot l-1 took candidate p (for slot 0 every predecessor row is the
    anchor, so row 0 is the one that means anything), and `greedy[l]` is the candidate index the
    released greedy walk took at slot l.

    Two decisions, both from measurements already in the ledger.

    **The greedy path goes in first, whatever its score.** The 10:28 entry measured the exact
    maximiser of the selector's own objective -- Viterbi -- accepting 3.22 tokens a block against
    greedy's 4.30. What a verify pays for is the expected accepted PREFIX, in which slot 0
    multiplies every later term, and a path maximiser will happily trade slot 0 away for a better
    total. Greedy's slot 0 is the head's own top-1, the most reliable single signal in the lattice.

    **The rest is best-first on path probability.** A node's path probability is its parent's times
    a conditional, so priorities fall monotonically down any path; popping in descending order
    therefore yields the highest-probability ancestor-closed set of nodes of that size, which is
    exactly the set that maximises the expected accepted length for a given budget.

    Pure Python and no torch, so the simulator scores the tree the engine will actually build --
    `tools/sim_draft.py` and `engine/drafters/dflash2.py` call this same function.
    """
    import heapq
    import math

    b = TreeBuilder(anchor)
    heap: list[tuple] = []
    tie = 0
    n = 0
    L = len(cand)
    k = len(cand[0]) if L else 0
    row, parent, lp = 0, 0, 0.0
    for slot, c in enumerate(greedy[:L]):
        if n >= budget:
            break
        base = lp
        lp += logp[slot][row][c]
        here, parent = parent, b.add(parent, cand[slot][c], math.exp(lp), f"{source}-greedy")
        n += 1
        for c2 in range(k):
            if c2 != c:
                tie += 1
                heapq.heappush(heap, (-(base + logp[slot][row][c2]), tie, here, slot, c2))
        row = c
    while heap and n < budget:
        negp, _, parent_id, slot, c = heapq.heappop(heap)
        lp = -negp
        nid = b.add(parent_id, cand[slot][c], math.exp(lp), f"{source}-tree")
        n += 1
        if slot + 1 < L:
            for c2 in range(k):
                tie += 1
                heapq.heappush(heap, (-(lp + logp[slot + 1][c][c2]), tie, nid, slot + 1, c2))
    return b.build()
