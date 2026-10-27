"""The suffix memory: does it find what is there, decline what is not, and stay cheap.

Token ids here are arbitrary small integers; nothing in the drafter knows or cares what they mean.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.drafters.engram import EngramDrafter  # noqa: E402
from engine.drafters.ngram import (CorpusSuffixStore, LocalSuffixIndex,  # noqa: E402
                                   NgramDrafter)


def test_local_index_finds_the_longest_match():
    ix = LocalSuffixIndex()
    ix.prime([1, 2, 3, 4, 5, 1, 2, 3, 9, 9])
    n, pos = ix.lookup([1, 2, 3], min_order=2)
    assert n == 3
    assert sorted(ix.continuation(p, 1)[0] for p in pos) == [4, 9]


def test_local_index_positions_are_capped():
    ix = LocalSuffixIndex(orders=(3,), max_positions=4)
    ix.prime([7, 7, 7] * 20)
    _, pos = ix.lookup([7, 7, 7], min_order=3)
    assert len(pos) <= 4


def test_drafter_declines_on_a_context_it_has_never_seen():
    d = NgramDrafter(corpus_path="")
    d.prime([1, 2, 3, 4, 5, 6, 7, 8])
    assert d.propose_tree([1, 2, 3, 4, 5, 6, 7, 8]) is None
    assert d.propose([1, 2, 3, 4, 5, 6, 7, 8], 8) == []


def test_drafter_proposes_a_repeat():
    seq = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * 3
    d = NgramDrafter(corpus_path="", min_expected=0.3)
    d.prime(seq)
    tree = d.propose_tree(seq, depth=8)
    assert tree is not None
    tree.check()
    # the sequence is perfectly periodic, so the continuation is known exactly
    assert tree.accepted_against([10, 11, 12, 13]) == 4


def test_ambiguity_becomes_branches_not_a_wrong_chain():
    # "a b" is followed by c twice and by d once
    seq = []
    for nxt in (30, 30, 31):
        seq += [1, 2, nxt, 99, 98, 97, 96, 95]
    d = NgramDrafter(corpus_path="", min_order=2, min_expected=0.1, branch_top_k=3)
    d.prime(seq)
    tree = d.propose_tree(seq[:-6] + [1, 2], depth=4)
    assert tree is not None
    first = [tree.tokens[c] for c in tree.children_of(0)]
    assert set(first) >= {30, 31}, "both observed continuations belong in the tree"
    assert tree.accepted_against([30]) == 1
    assert tree.accepted_against([31]) == 1


def test_scores_are_path_probabilities():
    seq = ([1, 2, 40, 50] + [77] * 6) * 4 + [1, 2, 40, 60] + [77] * 6
    d = NgramDrafter(corpus_path="", min_order=2, min_expected=0.1)
    d.prime(seq)
    tree = d.propose_tree(seq + [1, 2], depth=3)
    assert tree is not None
    for i in range(1, len(tree)):
        assert 0.0 < tree.scores[i] <= 1.0
        assert tree.scores[i] <= tree.scores[tree.parents[i]] + 1e-9


def test_expected_accepted_tracks_reality_on_a_repetitive_stream():
    rng = random.Random(7)
    vocab = [rng.randrange(1000) for _ in range(40)]
    stream = []
    for _ in range(60):
        start = rng.randrange(0, 30)
        stream += vocab[start:start + 8]
    d = NgramDrafter(corpus_path="", min_expected=0.3)
    d.prime(stream[:200])
    got, want = 0.0, 0
    fired = 0
    for i in range(200, len(stream) - 16):
        tree = d.propose_tree(stream[:i], depth=8)
        if tree is not None:
            fired += 1
            got += tree.expected_accepted()
            want += tree.accepted_against(stream[i:i + 16])
        d.observe([stream[i]])
    assert fired > 0
    # the estimate should be the right order of magnitude, not exact
    assert 0.4 < (got / max(want, 1)) < 2.5, f"expected {got:.1f} vs actual {want}"


def test_corpus_store_round_trip():
    import numpy as np
    from tools.build_corpus import build_suffix_array
    toks = np.array([5, 6, 7, 5, 6, 8, 5, 6, 7, 9], dtype=np.int32)
    sa = build_suffix_array(toks, max_order=4)
    store = CorpusSuffixStore(toks, sa, max_order=4)
    n, pos = store.lookup([1, 1, 5, 6, 7], min_order=2)
    assert n == 3
    assert sorted(store.continuation(p, 1)[0] for p in pos) == [5, 9]
    n2, pos2 = store.lookup([4, 4, 4], min_order=2)
    assert n2 == 0 and pos2 == []


def test_corpus_store_loads_from_disk():
    import numpy as np
    from tools.build_corpus import build_suffix_array, write_store
    with tempfile.TemporaryDirectory() as d:
        toks = np.array([1, 2, 3, 1, 2, 4, 1, 2, 3, 5], dtype=np.int32)
        write_store(d, toks, build_suffix_array(toks, 4), {"max_order": 4, "sources": []})
        store = CorpusSuffixStore.load(d)
        assert store is not None and store.n == 10
        n, pos = store.lookup([9, 1, 2], min_order=2)
        assert n == 2 and len(pos) == 3
        assert CorpusSuffixStore.load(os.path.join(d, "nope")) is None


def test_corpus_raises_coverage_over_the_sequence_alone():
    import numpy as np
    from tools.build_corpus import build_suffix_array
    rng = random.Random(3)
    corpus = [rng.randrange(500) for _ in range(4000)]
    store = CorpusSuffixStore(np.array(corpus, dtype=np.int32),
                              build_suffix_array(np.array(corpus, dtype=np.int32), 8),
                              max_order=8)
    # a fresh sequence that happens to quote the corpus
    quote = corpus[1000:1060]
    local_only = NgramDrafter(corpus_path="", min_expected=0.2, min_corpus_order=4)
    with_corpus = NgramDrafter(corpus_path="", min_expected=0.2, min_corpus_order=4)
    with_corpus.corpus = store
    hits = [0, 0]
    for d, slot in ((local_only, 0), (with_corpus, 1)):
        d.prime(quote[:10])
        for i in range(10, 40):
            if d.propose_tree(quote[:i], depth=8) is not None:
                hits[slot] += 1
            d.observe([quote[i]])
    assert hits[1] > hits[0], f"corpus should fire more often: {hits}"


def test_corpus_lookup_agrees_with_brute_force():
    """The binary search must find exactly the occurrences a linear scan would."""
    import numpy as np
    from tools.build_corpus import build_suffix_array
    rng = random.Random(19)
    # a skewed vocabulary, so short n-grams recur the way they do in real text
    corpus = [min(int(rng.paretovariate(1.2)), 200) for _ in range(6000)]
    arr = np.array(corpus, dtype=np.int32)
    store = CorpusSuffixStore(arr, build_suffix_array(arr, 8), max_order=8)
    for _ in range(200):
        start = rng.randrange(0, len(corpus) - 10)
        length = rng.randrange(1, 7)
        pattern = corpus[start:start + length]
        n, pos = store.lookup(pattern, min_order=1)
        assert n == length, "a pattern copied out of the corpus must match at its full length"
        brute = [i + length for i in range(len(corpus) - length + 1)
                 if corpus[i:i + length] == pattern]
        if len(brute) <= 64:
            assert sorted(pos) == sorted(brute), pattern
        else:
            assert set(pos) <= set(brute) and len(pos) == 64


def test_corpus_continuation_stops_at_a_document_boundary():
    import numpy as np
    from tools.build_corpus import DOC_SEP, build_suffix_array
    toks = np.array([1, 2, 3, DOC_SEP, 4, 5, 6], dtype=np.int64)
    store = CorpusSuffixStore(toks, build_suffix_array(toks, 4), max_order=4,
                              meta={"doc_sep": DOC_SEP})
    assert store.continuation(1, 8) == [2, 3]


def test_v2_never_costs_more_per_call_than_a_verify_step():
    rng = random.Random(11)
    stream = [rng.randrange(2000) for _ in range(4000)]
    d = NgramDrafter(corpus_path="")
    d.prime(stream)
    t0 = time.perf_counter()
    for i in range(500):
        d.propose_tree(stream[:3000 + i], depth=16)
    per_call_ms = (time.perf_counter() - t0) / 500 * 1000
    # the whole point of a lookup drafter is that it is free against a 151 ms verify
    assert per_call_ms < 5.0, f"{per_call_ms:.2f} ms per proposal is not free"


def test_v1_and_v2_agree_where_v1_is_confident():
    """V2 must not lose what v1 found: a unique long match gives the same chain."""
    seq = list(range(100, 160)) + list(range(100, 130))
    v1 = EngramDrafter()
    v1.prime(seq)
    v2 = NgramDrafter(corpus_path="", min_expected=0.3)
    v2.prime(seq)
    a = v1.propose(seq, 8)
    b = v2.propose(seq, 8)
    assert a and b
    assert a[:4] == b[:4]


# ------------------------------------------------------------------ the incremental prime

def _code_like(n, seed=0, vocab=60):
    """A token stream with the repeats code has: a few hundred distinct lines, reused."""
    import random
    rng = random.Random(seed)
    lines = [[rng.randrange(vocab) for _ in range(rng.randrange(3, 12))] for _ in range(300)]
    out = []
    while len(out) < n:
        out += lines[min(len(lines) - 1, int(rng.expovariate(1 / 40)))]
    return out[:n]


def _same(a, b):
    assert a.tokens == b.tokens
    for n in a.orders:
        assert a.index[n] == b.index[n], n


def _fresh(tokens, **kw):
    ix = LocalSuffixIndex(**kw)
    ix.prime(tokens)
    return ix


def test_an_extended_prompt_keeps_the_index_and_equals_a_rebuild():
    seq = _code_like(6000)
    ix = LocalSuffixIndex(max_positions=4)
    ix.prime(seq[:5000])
    ix.extend(seq[5000:5040])                  # the answer, observed during the decode
    nxt = seq[:5000] + seq[5000:5030] + _code_like(400, seed=1)
    ix.prime(nxt)
    assert ix.stats["kept"] == 5030 and ix.stats["rebuilt"] == 1
    _same(ix, _fresh(nxt, max_positions=4))


def test_a_prompt_that_diverges_inside_the_old_one():
    seq = _code_like(6000, seed=2)
    ix = _fresh(seq, max_positions=4)
    nxt = seq[:4100] + _code_like(900, seed=3)
    ix.prime(nxt)
    assert ix.stats["kept"] == 4100
    _same(ix, _fresh(nxt, max_positions=4))


def test_a_prompt_that_shares_little_is_rebuilt():
    seq = _code_like(6000, seed=4)
    ix = _fresh(seq)
    nxt = seq[:100] + _code_like(3000, seed=5)
    ix.prime(nxt)
    assert ix.stats["rebuilt"] == 2
    _same(ix, _fresh(nxt))
    ix.prime([9999] + nxt)
    assert ix.stats["rebuilt"] == 3
    _same(ix, _fresh([9999] + nxt))


def test_lookups_after_an_incremental_prime_are_a_rebuilds():
    seq = _code_like(8000, seed=6)
    a = LocalSuffixIndex()
    for end in (3000, 3500, 5200, 5100, 8000):  # growing, one step back, growing again
        a.prime(seq[:end])
    b = _fresh(seq)
    for i in range(10, 8000, 37):
        assert a.lookup(seq[:i], 2) == b.lookup(seq[:i], 2), i


def test_the_drafter_proposes_the_same_trees_after_an_incremental_prime():
    seq = _code_like(5000, seed=7, vocab=30)
    d1 = NgramDrafter(corpus_path="", min_expected=0.3)
    d1.prime(seq[:4000])
    d1.observe(seq[4000:4050])
    d1.prime(seq[:4600])
    d2 = NgramDrafter(corpus_path="", min_expected=0.3)
    d2.prime(seq[:4600])
    for i in range(4000, 4600, 11):
        t1 = d1.propose_tree(seq[:i], 8)
        t2 = d2.propose_tree(seq[:i], 8)
        assert (t1 is None) == (t2 is None), i
        if t1 is not None:
            assert t1.tokens == t2.tokens and t1.parents == t2.parents, i
            assert t1.scores == t2.scores, i


def test_a_reset_then_prime_is_still_incremental_and_a_reset_alone_empties():
    """The served length router's arms reset the shared index on every request, just before it
    is primed: the prime must still reuse what the index held. Read after a reset without a
    prime, the index is empty."""
    seq = _code_like(6000, seed=8)
    ix = _fresh(seq[:5000], max_positions=4)
    ix.reset()
    ix.prime(seq)
    assert ix.stats["kept"] == 5000
    _same(ix, _fresh(seq, max_positions=4))
    ix.reset()
    assert ix.lookup(seq[:100], 2) == (0, []) and ix.tokens == []
    d = NgramDrafter(corpus_path="")
    d.prime(seq[:3000])
    d.reset()
    assert d.local.lookup(seq[:50], 2) == (0, [])


def test_a_reset_then_prime_is_still_incremental_and_a_reset_alone_empties():
    """The served length router's arms reset the shared index on every request, just before it
    is primed: the prime must still reuse what the index held. Read after a reset without a
    prime, the index is empty."""
    seq = _code_like(6000, seed=8)
    ix = _fresh(seq[:5000], max_positions=4)
    ix.reset()
    ix.prime(seq)
    assert ix.stats["kept"] == 5000
    _same(ix, _fresh(seq, max_positions=4))
    ix.reset()
    assert ix.lookup(seq[:100], 2) == (0, []) and ix.tokens == []
    d = NgramDrafter(corpus_path="")
    d.prime(seq[:3000])
    d.reset()
    assert d.local.lookup(seq[:50], 2) == (0, [])


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
