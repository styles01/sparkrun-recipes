"""The suffix memory, second version: two stores, counted candidates, and a tree instead of a chain.

Version one (`engram.py`) kept one dictionary per n-gram order over the current sequence and
returned the single continuation that followed the most recent occurrence of the longest match. It
worked -- 46-100 % of its drafts were accepted -- and it fired on 8 % of steps.
Coverage, not precision, was the whole problem: an exact n-gram of order >= 3 rarely recurs inside
128 fresh tokens of a 300-token context.

Three changes follow from that, and they are the difference between this file and the old one.

**A second store, and it is much larger than the first.** Besides the per-sequence index over
prompt plus output, there is a static store over a corpus -- code, prose, transcripts, the model's
own past outputs. It is a suffix array over a token array, both memory-mapped, built once by
`tools/build_corpus.py`. Asking it costs three binary searches; it holds tens of millions of tokens
and it is the answer to "has this n-gram ever occurred", rather than "has it occurred since this
request started". This board has ~90 GB of unused unified memory, so the store is free in the only
currency that matters here.

**Counts, not recency.** Every occurrence of a match contributes a vote, so the drafter knows the
difference between a continuation seen once and one seen forty times, and it can say how much
agreement there was. That number is what the firing policy and the node budget are decided by; the
old drafter had no such signal and the router that consumed it was estimating acceptance from
single-digit sample counts.

**A tree, not a chain.** When the context is ambiguous the store usually knows the two or three
things that could come next, and on this board a second candidate costs 1.9 ms against a 151 ms
step. The output is a `DraftTree`, which is also the shape the block drafter's lattice will be
merged into -- one verify call, two sources.

Costs nothing in weights, which is the point: on a step where this drafter fires, the block costs
the verify pass and nothing else. A lookup against a 38-million-token store measures 0.20 ms, against
a verify step of 126 ms.

WHAT THE MEASUREMENT SAID ABOUT THAT FIRST PARAGRAPH
-----------------------------------------------------
Coverage was the wrong diagnosis, or rather it was the right diagnosis of the wrong problem. The
corpus does raise the fire rate, 6.2 % to 9.1 %, and it lowers the accepted tokens per fired block
from 1.80 to 1.40. Tuned for throughput rather than for coverage the policy goes the other way: it
fires on 2.6 % of steps and gets 3.2 tokens when it does.

The reason is the alternative. A step this drafter declines is not a step lost, it is a step the
prediction head takes, and that head costs 4 ms with a trimmed vocabulary and delivers about one
token. A wrong proposal from here costs a whole verify plus a 23 ms rollback. So the bar is not
"has this n-gram occurred", it is "has it occurred often enough that I will beat a cheap neural
drafter", and that bar is high. `min_order` tunes to 5 and `min_corpus_order` to 7.

Where it does pay it pays enormously: on an editing workload the router with this drafter runs at
45.1 tok/s against 25.2 for the head alone. It is a specialist, and the firing policy exists to keep
it from pretending otherwise.

No text is stored here, only token ids, and the corpus store is built on the box and never leaves
it.
"""

from __future__ import annotations

import json
import os
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

from engine.tree import DraftTree, TreeBuilder
from . import Drafter

DEFAULT_ORDERS = (8, 7, 6, 5, 4, 3, 2)


class LocalSuffixIndex:
    """Incremental index over one sequence: prompt, then every token as it is accepted.

    One dictionary per order, mapping an n-gram to the positions that followed its occurrences.
    Keeping positions rather than successor tokens is what makes a continuation of any length
    available from the same entry. A lookup reads the latest `max_positions` of them.

    `prime` does not rebuild when the new prompt starts with the sequence the index already
    holds, which is every turn of an agent conversation: it drops the positions past the shared
    part and indexes the rest. The lists keep every position, not only the latest
    `max_positions`, because a dropped position must not leave a list shorter than a rebuild
    would make it; so the index after `prime` is the index a rebuild makes, list for list. A
    rebuild over 190k tokens took 328 ms before every prefill on the box CPU.

    `reset` only marks the index empty: the served length router's arms reset the index they
    share on every request, just before it is primed. The old sequence is dropped by the first
    call that is not a `prime`, so an index read after `reset` is empty as it always was.
    """

    def __init__(self, orders: tuple[int, ...] = DEFAULT_ORDERS, max_positions: int = 24):
        self.orders = tuple(sorted(orders, reverse=True))
        self.max_positions = max_positions
        self.tokens: list[int] = []
        self.index: dict[int, dict[tuple, list[int]]] = {n: {} for n in self.orders}
        self.stats = {"primes": 0, "rebuilt": 0, "kept": 0, "indexed": 0}
        self._stale = False

    def reset(self) -> None:
        self._stale = True

    def _clear(self) -> None:
        self.tokens = []
        self.index = {n: {} for n in self.orders}
        self._stale = False

    def _fresh(self) -> None:
        if self._stale:
            self._clear()

    def _record(self, i: int) -> None:
        """Record that position `i` follows the n-grams ending at `i`."""
        t = self.tokens
        for n in self.orders:
            if i >= n:
                key = tuple(t[i - n:i])
                bucket = self.index[n].get(key)
                if bucket is None:
                    self.index[n][key] = [i]
                else:
                    bucket.append(i)

    def extend(self, tokens: list[int]) -> None:
        self._fresh()
        for tok in tokens:
            self.tokens.append(tok)
            self._record(len(self.tokens) - 1)

    def truncate(self, c: int) -> None:
        """Forget positions `c` and after, as if the sequence had only ever been `tokens[:c]`."""
        self._fresh()
        t = self.tokens
        for i in range(len(t) - 1, c - 1, -1):
            for n in self.orders:
                if i >= n:
                    d = self.index[n]
                    key = tuple(t[i - n:i])
                    bucket = d[key]
                    bucket.pop()                     # positions go in ascending, so `i` is last
                    if not bucket:
                        del d[key]
        del t[c:]

    def prime(self, tokens: list[int]) -> None:
        tokens = list(tokens)
        self._stale = False                      # what the index holds is reused, or rebuilt
        c = _shared(self.tokens, tokens)
        self.stats["primes"] += 1
        if c == 0 or len(self.tokens) - c > c:
            # sharing nothing, or less than would have to be forgotten: a rebuild is cheaper
            self._clear()
            self.extend(tokens)
            self.stats["rebuilt"] += 1
            self.stats["indexed"] += len(tokens)
            return
        self.truncate(c)
        self.extend(tokens[c:])
        self.stats["kept"] += c
        self.stats["indexed"] += len(tokens) - c

    def lookup(self, context: list[int], min_order: int) -> tuple[int, list[int]]:
        """Longest order with a hit: returns (match length, positions that followed it)."""
        self._fresh()
        for n in self.orders:
            if n < min_order or len(context) < n:
                continue
            hits = self.index[n].get(tuple(context[-n:]))
            if hits:
                # the current position is in the index too; it is not a precedent for itself
                usable = [p for p in hits[-self.max_positions:] if p < len(self.tokens)]
                if usable:
                    return n, usable
        return 0, []

    def continuation(self, pos: int, k: int) -> list[int]:
        self._fresh()
        return self.tokens[pos:pos + k]


def _shared(a: list, b: list, step: int = 4096) -> int:
    """How many leading tokens two sequences share; one pass over the shared part."""
    m = min(len(a), len(b))
    i = 0
    while i < m:
        j = min(m, i + step)
        if a[i:j] != b[i:j]:
            while a[i] == b[i]:
                i += 1
            return i
        i = j
    return m


class CorpusSuffixStore:
    """A static store over a token corpus: a suffix array, sorted by the first `max_order` tokens.

    Sorting by a bounded prefix rather than the full suffix is not an approximation here -- the
    drafter never matches more than `max_order` tokens -- and it makes the build three doubling
    rounds instead of log n of them. Both arrays are memory-mapped, so a 40-million-token store
    costs 320 MB of address space and no resident memory until it is touched.
    """

    def __init__(self, tokens, sa, max_order: int = 8, meta: dict | None = None):
        self.tokens = tokens
        self.sa = sa
        self.max_order = max_order
        self.meta = meta or {}
        self.n = len(tokens)
        # documents are concatenated with a separator above every real id; a continuation stops
        # there rather than running from the end of one file into the beginning of another
        self.doc_sep = int(self.meta.get("doc_sep", 1 << 30))

    @classmethod
    def load(cls, path: str, tokenizer_sha: str | None = None) -> "CorpusSuffixStore | None":
        """Open a store built by tools/build_corpus.py, or return None if there is none. With
        `tokenizer_sha`, a store recorded for another tokenizer is refused."""
        if not path or not os.path.isdir(path):
            return None
        meta_path = os.path.join(path, "meta.json")
        if not os.path.exists(meta_path):
            return None
        import numpy as np
        with open(meta_path) as f:
            meta = json.load(f)
        from engine.tokfp import check_store
        check_store(meta, tokenizer_sha, path)
        tokens = np.load(os.path.join(path, "tokens.npy"), mmap_mode="r")
        sa = np.load(os.path.join(path, "sa.npy"), mmap_mode="r")
        return cls(tokens, sa, max_order=meta.get("max_order", 8), meta=meta)

    def _cmp_at(self, sa_i: int, pattern) -> int:
        """Compare the suffix at `sa[sa_i]` with `pattern`: -1, 0 or +1 on the bounded prefix.

        `tolist()` before the loop is not decoration. Indexing a memory-mapped array element by
        element costs about ten times what indexing a list does, and this runs 50 times per binary
        search.
        """
        start = int(self.sa[sa_i])
        m = len(pattern)
        seg = self.tokens[start:start + m].tolist()
        if len(seg) < m:
            seg += [-1] * (m - len(seg))
        for a, b in zip(seg, pattern):
            if a != b:
                return -1 if a < b else 1
        return 0

    def _range(self, pattern) -> tuple[int, int]:
        """The half-open SA range of suffixes starting with `pattern`; empty if there is none."""
        lo, hi = 0, len(self.sa)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cmp_at(mid, pattern) < 0:
                lo = mid + 1
            else:
                hi = mid
        start = lo
        hi = len(self.sa)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cmp_at(mid, pattern) <= 0:
                lo = mid + 1
            else:
                hi = mid
        return start, lo

    def lookup(self, context: list[int], min_order: int,
               max_samples: int = 64) -> tuple[int, list[int]]:
        """Longest suffix of `context`, at most `max_order` long, that occurs in the corpus.

        A short n-gram can occur tens of thousands of times in thirty million tokens, and the SA
        range is sorted by what *follows* the match, so the first 64 entries of a large range are
        all the same continuation. Sampling across the range instead keeps the vote counts an
        estimate of the real distribution rather than of the alphabet.
        """
        top = min(self.max_order, len(context))
        for n in range(top, min_order - 1, -1):
            pattern = context[-n:]
            lo, hi = self._range(pattern)
            width = hi - lo
            if width <= 0:
                continue
            if width <= max_samples:
                idx = range(lo, hi)
            else:
                step = width / max_samples
                idx = (lo + int(j * step) for j in range(max_samples))
            return n, [int(self.sa[i]) + n for i in idx]
        return 0, []

    def continuation(self, pos: int, k: int) -> list[int]:
        seg = self.tokens[pos:pos + k].tolist()
        for i, x in enumerate(seg):
            if x >= self.doc_sep:
                return seg[:i]
        return seg


class NgramDrafter(Drafter):
    """Engram v2: propose a scored tree of continuations, or decline.

    Declining is a first-class outcome. The whole economic argument for a lookup drafter is that it
    is free where it is right and silent where it is not; a firing policy that proposes on every
    step turns it back into a bad neural drafter. `propose_tree` returns None when the expected
    value of the block it could build does not beat what the caller says it can get otherwise.
    """

    name = "ngram"

    def __init__(self, corpus_path: str | None = None, orders: tuple[int, ...] = DEFAULT_ORDERS,
                 min_order: int = 3, max_depth: int = 16, node_budget: int = 16,
                 branch_top_k: int = 3, min_expected: float = 0.6, alpha: float = 0.6,
                 corpus_weight: float = 0.5, min_corpus_order: int = 5,
                 verify_base_ms: float = 149.1, verify_per_node_ms: float = 1.896,
                 tokenizer_sha: str | None = None):
        self.local = LocalSuffixIndex(orders)
        self.corpus = CorpusSuffixStore.load(
            corpus_path if corpus_path is not None
            else _S.get("CORPUS"), tokenizer_sha=tokenizer_sha)
        self.min_order = min_order
        self.min_corpus_order = min_corpus_order
        self.max_depth = max_depth
        self.node_budget = node_budget
        self.branch_top_k = branch_top_k
        self.min_expected = min_expected
        self.alpha = alpha
        self.corpus_weight = corpus_weight
        self.verify_base_ms = verify_base_ms
        self.verify_per_node_ms = verify_per_node_ms
        self.stats = {"calls": 0, "fired": 0, "nodes": 0, "expected": 0.0,
                      "order_hist": {}, "source_hist": {"local": 0, "corpus": 0, "both": 0}}
        self.last_match_len = 0
        self.last_support = 0
        self.last_expected = 0.0
        # A per-level continuation rate for this round's tree, when a caller sets `rho_fn`
        # (src, match_len) -> rho: a node's score is then the product over its path of rho times
        # the vote share at that level, instead of the vote share smoothed by `alpha` at every
        # level. None is the drafter it was.
        self.rho_fn = None
        self.last_top: list[int] = []
        self.last_source = "?"

    # --- Drafter interface ---------------------------------------------------------------------

    def reset(self) -> None:
        self.local.reset()

    def prime(self, tokens: list[int]) -> None:
        self.local.prime(tokens)

    def observe(self, tokens: list[int]) -> None:
        self.local.extend(tokens)

    def add_store(self, store) -> None:
        """Ask one more suffix store alongside the corpus, at the same price.

        The second store the server hands over is `engine.cache.PersistentSuffixStore`: what this
        engine itself has read and written, kept across restarts. Two binary searches instead of
        one, on a step where the alternative is a 4 ms prediction head.
        """
        from engine.cache import SuffixStoreSet
        have = (list(self.corpus.stores) if isinstance(self.corpus, SuffixStoreSet)
                else ([self.corpus] if self.corpus is not None else []))
        have.append(store)
        self.corpus = SuffixStoreSet(have)

    def propose(self, context: list[int], k: int) -> list[int]:
        """The chain interface the current verify path speaks: the tree's best single branch."""
        tree = self.propose_tree(context, min(k, self.max_depth))
        if tree is None or tree.n_draft == 0:
            return []
        node, chain = 0, []
        while True:
            kids = tree.children_of(node)
            if not kids or len(chain) >= k:
                break
            node = max(kids, key=lambda c: tree.scores[c])
            chain.append(tree.tokens[node])
        return chain

    # --- the drafter proper ----------------------------------------------------------------------

    def candidates(self, context: list[int], depth: int) -> tuple[int, list[tuple[list[int], float]]]:
        """Scored continuations for this context, from both stores.

        A continuation's weight is the number of times it was seen; corpus votes are discounted,
        because a match inside the current request is evidence about *this* text and a match in the
        corpus is evidence about text in general. Scores are the smoothed share of the vote, read as
        the probability that the first token of that continuation is the one the model will pick.
        """
        self.last_top = []
        n_local, pos_local = self.local.lookup(context, self.min_order)
        n_corpus, pos_corpus = (0, [])
        if self.corpus is not None:
            n_corpus, pos_corpus = self.corpus.lookup(context, max(self.min_order,
                                                                  self.min_corpus_order))
        if not pos_local and not pos_corpus:
            return 0, []

        best = max(n_local, n_corpus)
        votes: dict[tuple, float] = {}
        if pos_local and n_local >= best:
            for p in pos_local:
                cont = tuple(self.local.continuation(p, depth))
                if cont:
                    votes[cont] = votes.get(cont, 0.0) + 1.0
        if pos_corpus and n_corpus >= best:
            for p in pos_corpus:
                cont = tuple(self.corpus.continuation(p, depth))
                if cont:
                    votes[cont] = votes.get(cont, 0.0) + self.corpus_weight
        if not votes:
            return 0, []
        total = sum(votes.values())
        out = [(list(c), v) for c, v in votes.items()]
        out.sort(key=lambda cs: -cs[1])
        self.last_support = int(round(total))
        src = ("both" if (pos_local and pos_corpus and n_local == n_corpus)
               else "local" if n_local >= n_corpus else "corpus")
        self.stats["source_hist"][src] += 1
        self.last_source = src
        # the top continuation, for a caller that scores the lookup against what was committed
        self.last_top = list(out[0][0])
        return best, out

    def build_tree(self, anchor: int, cands: list[tuple[list[int], float]],
                   depth: int, rho: float | None = None) -> DraftTree:
        """Prefix tree of the candidates, scored by path probability.

        A node's score is the summed weight of the candidates passing through it, divided by the
        weight of the candidates passing through its parent, multiplied by the parent's score --
        the probability that the whole path is accepted, which is exactly the term
        `DraftTree.expected_accepted` adds up.

        The division carries `alpha` in its denominator at *every* level, not only at the root. A
        continuation seen once agrees with itself all the way down, so without that the tree would
        claim a sixteen-token draft from a single observation is worth ten accepted tokens; the
        first dry run measured 4.4. With the smoothing the claim decays geometrically, at
        `count / (count + alpha)` per level, and `alpha` is the one number the simulator tunes.
        """
        b = TreeBuilder(anchor)
        # weight of every prefix, so a node's conditional probability is a division
        mass: dict[tuple, float] = {(): 0.0}
        kids: dict[tuple, set] = {}
        for cont, w in cands:
            mass[()] += w
            for d in range(1, min(len(cont), depth) + 1):
                key = tuple(cont[:d])
                mass[key] = mass.get(key, 0.0) + w
                kids.setdefault(key[:-1], set()).add(key)
        nodes: dict[tuple, int] = {(): 0}
        scores: dict[tuple, float] = {(): 1.0}
        frontier = [()]
        while frontier:
            parent_key = frontier.pop(0)
            children = sorted(kids.get(parent_key, ()), key=lambda k: -mass[k])
            parent_mass = mass[parent_key]
            for key in children[:self.branch_top_k]:
                if rho is not None:
                    share = mass[key] / parent_mass if parent_mass else 0.0
                    p = scores[parent_key] * rho * share
                else:
                    denom = parent_mass + self.alpha
                    p = scores[parent_key] * (mass[key] / denom if denom else 0.0)
                if p <= 0.0:
                    continue
                nodes[key] = b.add(nodes[parent_key], key[-1], p, "ngram")
                scores[key] = p
                frontier.append(key)
        return b.build()

    def propose_tree(self, context: list[int], depth: int | None = None,
                     alternative_value: float = 0.0) -> DraftTree | None:
        """A pruned tree, or None when firing is not worth it.

        `alternative_value` is what the caller could get instead, in tokens per second -- the block
        drafter's expected throughput, or the plain non-speculative step. The policy is the cost
        model and nothing else: fire when this tree's expected tokens per second beats it.
        """
        self.stats["calls"] += 1
        depth = depth or self.max_depth
        if not context:
            return None
        match_len, cands = self.candidates(context, depth)
        self.last_match_len = match_len
        if not cands:
            self.last_expected = 0.0
            return None
        rho = (self.rho_fn(self.last_source, match_len) if self.rho_fn is not None else None)
        tree = self.build_tree(context[-1], cands, depth, rho=rho)
        tree = tree.prune(self.node_budget, per_node_ms=self.verify_per_node_ms,
                          base_ms=self.verify_base_ms)
        expected = tree.expected_accepted()
        self.last_expected = expected
        if expected < self.min_expected:
            return None
        if tree.value(base_ms=self.verify_base_ms,
                      per_node_ms=self.verify_per_node_ms) <= alternative_value:
            return None
        self.stats["fired"] += 1
        self.stats["nodes"] += tree.n_draft
        self.stats["expected"] += expected
        self.stats["order_hist"][match_len] = self.stats["order_hist"].get(match_len, 0) + 1
        return tree
