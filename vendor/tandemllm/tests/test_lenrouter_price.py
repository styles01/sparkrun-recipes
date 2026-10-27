"""The latch priced on the tree each arm will run (`QWEN38_LATCH_PRICE`), driven by stub drafters.

The served configuration is modelled as it is configured: the narrow arm verifies a 16-node tree
(16 rows with the anchor), the wide arm a 16-row chain until the request has committed
`tree_wide_after` = 32 tokens and a 24-row tree after that, priced on the served curve
8:77.9, 16:78.9, 24:87.0, 32:97.0. Acceptance comes from fixed run lengths, as in
tests/test_lenrouter.py, so every assertion is about the policy and none about a board.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.lenrouter import LengthRouter  # noqa: E402
from tests.test_lenrouter import FakeDrafter, FakeEng, FakeNgram, FakeTree  # noqa: E402

SERVED = {8: 77.9, 16: 78.9, 24: 87.0, 32: 97.0}


class ServedArm:
    """A MergedRouter stand-in that builds exactly `node_budget` nodes, as the served arm does
    once its lattice and the lookup tree fill the budget. The router rewrites the wide arm's
    budget per block, and this arm follows it."""

    wants_rows = True

    def __init__(self, head, ngram, budget):
        self.mtp = head
        self.ngram = ngram
        self.node_budget = self.head_budget = budget
        self.calls = 0

    def propose_tree(self, context, k):
        self.calls += 1
        return FakeTree(context[-1], [9000 + i for i in range(self.node_budget)])

    def observe(self, tokens):
        self.ngram.observe(tokens)

    def reset(self):
        pass

    def prime(self, tokens):
        self.ngram.prime(tokens)

    def sync(self, tokens, hidden, first_pos, rows=None):
        self.mtp.sync(tokens, hidden, first_pos, rows=rows)

    def state_snapshot(self):
        return self.mtp.state_snapshot()

    def state_restore(self, snap):
        self.mtp.state_restore(snap)


def build(price=True, **kw):
    eng = FakeEng()
    small, large, ng = FakeDrafter(eng, 8), FakeDrafter(eng, 16), FakeNgram()
    narrow, wide = ServedArm(small, ng, 15), ServedArm(large, ng, 23)
    kw.setdefault("learn_cost", False)
    kw.setdefault("latch", True)
    kw.setdefault("drop_idle", True)
    kw.setdefault("tree_wide_after", 32)
    r = LengthRouter(narrow, wide, tree=True, ngram=ng, latch_price=price,
                     latch_table=SERVED, **kw)
    return r, small, large


def run(router, blocks, runs, k=31):
    """`runs[key]` is a cycle of how many of that arm's proposals the target agrees with. A narrow
    block is capped at its lattice's 7 slots, a wide one at 15, whatever the tree's node count."""
    idx = {"s": 0, "l": 0}
    keys = []
    for _ in range(blocks):
        router.propose_tree(list(range(50)), k)
        key = router.last_key
        keys.append(key)
        seq = runs[key]
        n = min(seq[idx[key] % len(seq)], 7 if key == "s" else 15)
        idx[key] += 1
        router.observe([9000 + i for i in range(n)] + [12345])
    return keys


# --- what the latch is priced on -----------------------------------------------------------------

def test_the_wide_arm_is_priced_at_the_tree_it_grows_into():
    r, _, _ = build()
    r.req_tokens, r.k_left = 1, 31
    assert r._latch_rows("s") == 16
    assert r._latch_rows("l") == 24, "24 rows once the request can reach 32 committed tokens"
    # the narrow arm pays 78.9 ms of verify, the wide one 87.0, plus draft and commit
    assert abs(r._latch_cost_ms("s", 3.0) - (78.9 + 25.3 + r.commit_ms)) < 1e-6
    assert abs(r._latch_cost_ms("l", 3.0) - (87.0 + 26.0 + r.commit_ms)) < 1e-6


def test_a_request_too_short_to_grow_the_wide_tree_is_priced_at_sixteen_rows():
    r, _, _ = build()
    r.req_tokens, r.k_left = 6, 20                 # 26 tokens in all: never reaches 32
    assert r._latch_rows("l") == 16
    r.req_tokens, r.k_left = 12, 20                # 32: the tree grows before the end
    assert r._latch_rows("l") == 24
    r2, _, _ = build(tree_wide_after=0)            # growth off: the configured budget always
    r2.k_left = 3
    assert r2._latch_rows("l") == 24


def test_the_latch_ignores_the_learned_verify_launch_time():
    """On the served loop `on_verify` times the launch of a graphed verify: 0.1-0.3 ms in the
    published row's reports. The flag prices on the curve, so a zero learned verify changes
    nothing about the decision."""
    a, _, _ = build(learn_cost=True)
    b, _, _ = build(learn_cost=True)
    for _ in range(12):
        a.on_verify(16, 0.2)
        a.on_verify(24, 0.1)
    fixture = {"s": [3], "l": [3, 3, 4, 3]}
    run(a, 30, fixture)
    run(b, 30, fixture)
    assert a.latched == b.latched == "s"
    assert a.stats["price"] == b.stats["price"]


def test_a_tie_goes_to_the_cheaper_arm():
    """Same tokens a block on both arms: the narrow arm verifies 16 rows, the wide one 24."""
    on, _, _ = build(price=True)
    run(on, 30, {"s": [3], "l": [3]})
    assert on.latched == "s" and on.idle == "l", on.report()
    # without the flag the same text latched wide: the tie rule was written for a 1 % price gap
    off, _, _ = build(price=False)
    run(off, 30, {"s": [3], "l": [3]})
    assert off.latched == "l", off.report()


def test_the_wide_arm_still_wins_where_it_commits_clearly_more():
    r, _, _ = build()
    run(r, 30, {"s": [3], "l": [6]})
    assert r.latched == "l", r.report()


def test_a_short_answer_prices_the_wide_arm_at_sixteen_rows():
    """A 25-token answer never reaches 32 committed tokens, so the wide arm verifies 16 rows for
    the whole request and costs what the narrow arm costs, bar its draft."""
    r, _, _ = build()
    produced, keys = 0, []
    while produced < 25:
        r.propose_tree(list(range(50)), min(31, 25 - produced))
        keys.append(r.last_key)
        n = 0
        r.observe([9000 + i for i in range(n)] + [12345])
        produced += n + 1
    assert r.latched is not None
    assert r.stats["price"] == "110.8/111.5", r.stats["price"]   # 78.9 + draft + 6.6 on both


# --- the probe skip ------------------------------------------------------------------------------

def test_one_full_opening_block_does_not_skip_the_probes():
    wide = [7, 2, 2, 2]                            # the opening line fills the narrow width once
    on, _, _ = build(price=True)
    keys = run(on, 30, {"s": [2], "l": wide})
    assert on.stats["probes"] > 0 and "s" in keys[4:8], keys
    off, _, _ = build(price=False)
    run(off, 30, {"s": [2], "l": wide})
    assert off.stats["probes"] == 0 and off.latched == "l", "today: one hit in four skips them"


def test_two_full_blocks_skip_the_probes_and_latch_wide():
    r, _, _ = build()
    keys = run(r, 30, {"s": [2], "l": [7, 7, 2, 2]})
    assert r.stats["probes"] == 0 and r.latched == "l" and set(keys) == {"l"}, keys


def test_probes_that_fill_the_narrow_width_latch_wide():
    """An edit whose copy starts after the opening: no full wide block in the first four, then the
    narrow probes fill all 7 slots. Their number is censored, so the latch goes wide."""
    r, _, _ = build()
    keys = run(r, 30, {"s": [7], "l": [0, 2, 5, 6, 15]})
    assert r.stats["probes"] >= 2 and r.latched == "l", r.report()
    assert keys[-1] == "l"


def test_a_quotation_latches_wide_without_probing():
    r, small, _ = build()
    keys = run(r, 30, {"s": [7], "l": [15]})
    assert r.latched == "l" and r.stats["probes"] == 0 and set(keys) == {"l"}
    assert small.calls == 0


# --- what must not change ------------------------------------------------------------------------

def test_the_loser_is_released_once():
    r, small, large = build()
    run(r, 30, {"s": [3], "l": [3]})
    assert r.latched == "s" and r.idle == "l" and large.released and not small.released
    before = len(large.synced)
    r.sync([1, 2], None, 100)
    assert len(large.synced) == before, "a released arm is not synced"


def test_a_restored_one_armed_snapshot_still_inherits_the_latch():
    r, _, _ = build()
    run(r, 30, {"s": [3], "l": [3]})
    snap = r.state_snapshot()
    r2, _, _ = build()
    r2.state_restore(snap)
    assert r2.latched == "s" and r2.idle == "l"
    keys = run(r2, 5, {"s": [3], "l": [3]})
    assert set(keys) == {"s"} and r2.stats["probes"] == 0


def test_the_latch_does_not_survive_the_request():
    r, _, _ = build()
    run(r, 30, {"s": [3], "l": [3]})
    r.reset()
    assert r.latched is None and r.idle is None and r.stats["price"] == "-"


def test_a_compile_is_not_a_draft_price():
    r, _, _ = build(learn_cost=True)
    r._note_draft("s", 782.6)
    assert r.dms_latch["s"].n == 0
    assert r._latch_draft_ms("s") == r.dms_latch["s"].value
    r._note_draft("l", 12.7)
    assert r._latch_draft_ms("s") == 12.7, "the other arm's clean timing before this arm has one"
    r._note_draft("s", 12.4)
    assert r._latch_draft_ms("s") == 12.4
    off, _, _ = build(price=False, learn_cost=True)
    off._note_draft("s", 12.4)
    assert off.dms_latch["s"].n == 0


def test_off_by_default_and_on_from_the_environment():
    old = os.environ.pop("QWEN38_LATCH_PRICE", None)
    try:
        r, _, _ = build(price=None)
        assert r.latch_price is False and "price" not in r.report()
        assert "price" not in r.stats
        os.environ["QWEN38_LATCH_PRICE"] = "1"
        r, _, _ = build(price=None)
        assert r.latch_price is True and "price" in r.report()
    finally:
        os.environ.pop("QWEN38_LATCH_PRICE", None)
        if old is not None:
            os.environ["QWEN38_LATCH_PRICE"] = old


def test_off_decides_exactly_as_before_on_the_same_fixtures():
    """The flag off must be the router it was: same arm on every block, same report, whatever the
    extra state the flag keeps."""
    fixtures = [{"s": [3], "l": [3]}, {"s": [2], "l": [7, 2, 2, 2]}, {"s": [7], "l": [15]},
                {"s": [3, 2, 5], "l": [4, 1, 6, 2]}, {"s": [2], "l": [7, 7, 2, 2]}]
    for fx in fixtures:
        a, _, _ = build(price=False)
        b, _, _ = build(price=False)
        del b.dms_latch, b.draft_prior               # state only the flag reads
        b.dms_latch = {"s": None, "l": None}
        ka, kb = run(a, 40, fx), run(b, 40, fx)
        assert ka == kb and a.report() == b.report(), fx


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception:
                fails += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{fails} failed")
    sys.exit(1 if fails else 0)
