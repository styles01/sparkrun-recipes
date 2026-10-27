"""A drafter that proposes deliberately wrong tokens.

Its only purpose is to force the verify path down every branch it has: a block where nothing is
accepted, a block where some prefix is accepted, a block where everything is. A speculative decoder
that is correct must produce exactly the same greedy output with this drafter attached as with no
drafter at all, and it must do so at every rejection point, which a good drafter would never reach.
"""

from __future__ import annotations

import random

from . import Drafter


class AdversarialDrafter(Drafter):
    name = "adversarial"

    def __init__(self, vocab: int, seed: int = 0, truth: list[int] | None = None,
                 accept_prefix: int = 0):
        self.rng = random.Random(seed)
        self.vocab = vocab
        self.truth = truth or []
        self.accept_prefix = accept_prefix
        self.n = 0

    def propose(self, context: list[int], k: int) -> list[int]:
        """The first `accept_prefix` tokens are the true continuation, the rest are noise."""
        out = []
        for i in range(k):
            j = self.n + i
            if i < self.accept_prefix and j < len(self.truth):
                out.append(self.truth[j])
            else:
                out.append(self.rng.randrange(1000, min(self.vocab, 200000)))
        return out

    def observe(self, tokens: list[int]) -> None:
        self.n += len(tokens)


class AdversarialTreeDrafter(AdversarialDrafter):
    """The same idea, shaped as a tree: branches that are wrong on purpose, around one that is not.

    A chain verify has one rejection point and a tree has as many as it has nodes, so the tree
    commit has strictly more ways to be wrong than the chain rollback did. What this drives, block
    after block, is the accepted path being an arbitrary walk through the DFS order rather than a
    prefix of it: a gather that quietly took the first L rows instead of the path's L rows would be
    invisible under any drafter that proposes a chain, and fails here on the first block.
    """

    name = "adversarial-tree"

    def __init__(self, vocab: int, seed: int = 0, truth: list[int] | None = None,
                 accept_prefix: int = 0, budget: int = 12, branch: int = 3):
        super().__init__(vocab, seed, truth, accept_prefix)
        self.budget = budget
        self.branch = branch

    def propose_tree(self, context: list[int], k: int):
        from engine.tree import TreeBuilder

        b = TreeBuilder(int(context[-1]))
        count = 0
        # one branch that is right for a random number of tokens, so the accepted path ends at a
        # different depth on every block and the commit is exercised at every length
        want = self.rng.randrange(0, self.accept_prefix + 1) if self.accept_prefix else 0
        node = 0
        for i in range(min(want, k)):
            j = self.n + i
            if j >= len(self.truth):
                break
            node = b.add(node, int(self.truth[j]), 0.9 ** (i + 1), "true")
            count += 1
        # and noise hung off arbitrary nodes, which is what puts the accepted path out of DFS order
        guard = 0
        while count < min(self.budget, k) and guard < 200:
            guard += 1
            parent = self.rng.randrange(0, count + 1)
            if len(b.kids[parent]) >= self.branch:
                continue
            b.add(parent, self.rng.randrange(1000, min(self.vocab, 200000)), 0.1, "noise")
            count += 1
        return b.build() if count else None
