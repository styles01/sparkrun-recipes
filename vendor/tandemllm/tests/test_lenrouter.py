"""The length router's policy, driven by simulated acceptance instead of a board.

What matters is not that it picks a length but that the picks follow the measured curve and the
evidence the loop can actually give it. Each test fixes an acceptance behaviour -- the acceptance a
workload would produce -- and checks the choice that behaviour implies, including the two properties
the design turns on: coming DOWN from the wide block needs no experiment, and going UP needs one.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.lenrouter import LengthRouter, verify_ms_b  # noqa: E402


class FakeCfg:
    def __init__(self, block):
        self.block_size = block


class FakeEng:
    def __init__(self):
        self.tap = None


class FakeDrafter:
    """A block drafter of a fixed width that always proposes, and records what it was asked."""

    wants_rows = True
    tree_temp = 1.0

    def __init__(self, eng, block, base=9000):
        self.eng = eng
        self.cfg = FakeCfg(block)
        self.base = base
        self.calls = 0
        self.taps = 0
        self.synced = []
        self._lattice = None
        self.released = False
        self.freed = None
        self.refilled = 0
        self.restored = None
        self.ctx = 128

    def propose(self, context, k):
        self.calls += 1
        return [self.base + i for i in range(min(k, self.cfg.block_size - 1))]

    def _on_tap(self, h):
        self.taps += 1

    def reset(self):
        pass

    def prime(self, tokens):
        pass

    def sync(self, tokens, hidden, first_pos, rows=None):
        if self.released:
            # `release` frees the draft KV; the real drafter reallocates it EMPTY on the next
            # `_build`, so a sync after a release is a sync into a cache with a hole in it.
            self.refilled += 1
        self.synced.append((len(tokens), first_pos))

    # what the phase-10 idle drop uses
    def release(self, free_cache=False):
        self.released = True
        self.freed = free_cache
        self.ctx = 0

    def state_snapshot(self):
        return ("fake", self.cfg.block_size, self.ctx)

    def state_restore(self, snap):
        self.restored = snap
        self.ctx = snap[2]


def build(**kw):
    """A router over two instant drafters.

    `learn_cost=False` matters: these fakes return in microseconds, so a router that timed them
    would price the draft at zero and the wide block would look free. The costs under test are the
    measured ones.
    """
    eng = FakeEng()
    small = FakeDrafter(eng, 8, base=9000)
    large = FakeDrafter(eng, 16, base=9000)
    kw.setdefault("learn_cost", False)
    return LengthRouter(small, large, **kw), small, large


def run(router, blocks, runs):
    """Drive `blocks` steps against a fixed acceptance behaviour.

    `runs` maps a drafter key -- "s" or "l" -- to a cycle of RUN LENGTHS: how many of that
    drafter's proposals the target would agree with before the block width truncates them. Stating
    the fixture this way rather than as an accepted count per width is what makes the truncation
    consistent: a narrow block is a prefix of a wide one, so the same run length has to produce
    `min(run, 7)` at eight and `min(run, 15)` at sixteen. A fixture that let those two be chosen
    independently would let a test assert something the engine cannot do.
    """
    idx = {"s": 0, "l": 0}
    widths = []
    for _ in range(blocks):
        draft = router.propose(list(range(50)), 15)
        key, width = router.last_key, router.last_width
        widths.append(width)
        seq = runs[key]
        r = seq[idx[key] % len(seq)]
        idx[key] += 1
        n = min(r, max(width - 1, 0), len(draft))
        router.observe(draft[:n] + [12345])
    return widths


# --- the curve ---------------------------------------------------------------------------------

def test_verify_curve_is_read_at_the_measured_points():
    assert verify_ms_b(8) == 99.1
    assert verify_ms_b(16) == 99.8
    # below the first point and above the last, flat and linear respectively, never a negative cost
    assert verify_ms_b(1) == 95.56
    assert verify_ms_b(32) > verify_ms_b(16)


def test_expected_prefix_is_the_running_product():
    p = [0.9, 0.8, 0.5]
    got = LengthRouter._expected_prefix(p, 3)
    assert abs(got - (0.9 + 0.72 + 0.36)) < 1e-9
    # a prefix expectation never exceeds the number of slots it is taken over
    assert LengthRouter._expected_prefix([1.0] * 15, 7) == 7.0


def test_the_curve_the_policy_is_priced_on_has_gone_flat():
    """Phase 5: a wide block cost 13.9 % more and had to commit 15.6 % more to break even. Phase 8
    deleted both terms that were linear in the row count, and the same pricing now reads about one
    per cent -- which is the whole reason the default arm changed."""
    r, _, _ = build()
    narrow = r._cost_ms("s", 8, 7.0)
    wide = r._cost_ms("l", 16, 15.0)
    assert 1.00 < wide / narrow < 1.03, wide / narrow
    # and a wide block carries more than twice the slots for it
    assert (r.w_large - 1) / (r.w_small - 1) > 2.0


# --- the tap ------------------------------------------------------------------------------------

def test_one_tap_feeds_both_drafters():
    """One bound callback, held: `self._on_tap is self._on_tap` is False in CPython, so a detach
    that compares them with `is` silently leaves the tap installed."""
    r, small, large = build()
    assert r.eng.tap is r._tap_cb
    for _ in range(5):
        r.eng.tap(object())
    assert small.taps == 5 and large.taps == 5


def test_both_caches_are_kept_current():
    """The drafter that did not propose still has to be synced or its cache falls behind."""
    r, small, large = build()
    r.sync([1, 2, 3], None, 10)
    assert small.synced == [(3, 10)] and large.synced == [(3, 10)]


# --- the regimes ---------------------------------------------------------------------------------

def test_reproduction_text_converges_on_the_wide_block():
    """`quote`: the narrow block accepts every slot it has, so its own number is censored."""
    r, _, _ = build(explore_period=32)
    widths = run(r, 40, {"s": [15], "l": [15]})
    tail = widths[-20:]
    assert sum(1 for w in tail if w == 16) >= 18, widths


def test_fresh_prose_comes_down_to_the_narrow_block():
    """`prose`: the wide block buys slots the drafter cannot fill, so the narrow one wins.

    The fixture is the measured one and the measurement is the point. Both arms accept about two
    slots, and the NARROW arm accepts slightly more of them -- 2.8 committed a block against 2.6 --
    because the two arms are different checkpoints and the eight-wide one was fine-tuned at the
    length it is being asked about. That 7 % is the whole of the gap between the two fixed
    baselines on prose, it is worth 5 % of the row, and it is invisible in the free counterfactual,
    which can only ever price the WIDE drafter's draft cut short. Finding it is what the bounded
    downward probe is for.
    """
    r, small, _ = build(explore_period=32)
    widths = run(r, 96, {"s": [2, 2, 2, 2, 1], "l": [2, 2, 1, 1, 2]})
    assert widths[-1] == 8, widths
    assert sum(1 for w in widths[-30:] if w == 8) >= 25, widths
    # and it cost a handful of probes to find out, not a policy of running narrow blocks blind
    assert r.stats["probes"] <= 4, r.report()


def test_the_downward_probe_is_bounded_and_only_taken_where_it_can_pay():
    """`quote`: the narrow block accepts every slot it has, so it is at its ceiling and there is
    nothing to learn down there. The probe is never taken and the router never leaves the wide
    block."""
    r, small, _ = build()
    widths = run(r, 40, {"s": [15], "l": [15]})
    assert r.stats["probes"] == 0, r.report()
    assert small.calls == 0
    assert set(widths[-20:]) == {16}, widths


def test_the_first_block_goes_wide():
    """A wide block prices both options and a narrow one prices only itself, so the first block --
    the one with no evidence behind it at all -- is taken at the width that learns the most."""
    r, small, large = build()
    run(r, 1, {"s": [3], "l": [3]})
    assert large.calls == 1 and small.calls == 0
    t, _, _, _ = build_tree()
    run_tree(t, 1, {"s": [3], "l": [3]})
    assert t.stats["large"] == 1 and t.stats["small"] == 0


def test_code_like_gain_now_clears_the_flattened_price():
    """The same fixture that used to settle narrow, and the opposite answer.

    Phase 5 measured `code` at 5.45 committed a block at eight against 6.10 at sixteen: a 12 %
    gain against a 15.6 % price, so the wide block lost and this test asserted that the router saw
    it lose. The price is now about one per cent, the same 12 % gain clears it several times over,
    and the router has to change its mind. It is the same measurement pointing the other way, and
    on the board the two fixed baselines say the same thing: 36.81 tok/s at eight against 46.98 at
    sixteen.
    """
    r, _, _ = build(explore_period=16)
    widths = run(r, 80, {"s": [4, 5, 4, 5, 4, 4, 5],
                         "l": [5, 5, 6, 5, 5, 5, 6]})
    assert sum(1 for w in widths[-30:] if w == 16) >= 26, widths
    # it bought the answer with the bounded probe and then stopped paying for it
    assert r.stats["probes"] <= 4, r.report()


def test_a_tie_resolves_wide():
    """Equal acceptance at both widths. The narrow arm is a shade cheaper and the margin is what
    stops a shade from becoming a policy -- because the narrow number is the optimistic one: it is
    measured by truncating wide blocks, which for a tree holds the accepted path only when that
    path's nodes ranked inside the smaller budget."""
    r, _, _ = build()
    widths = run(r, 60, {"s": [3], "l": [3]})
    assert sum(1 for w in widths[-20:] if w == 16) >= 18, widths


def test_a_chat_like_gain_keeps_the_wide_block():
    """The regression the phase-8 gate caught, as a fixture.

    On `chat` the wide block commits 4.33 a block against the narrow one's 3.92, and the old rule
    declined it 59 blocks out of 66 and finished 8.04 % behind a fixed sixteen. A 10 % gain against
    a 1 % price is not a close call, and the router must not treat it as one.
    """
    r, _, _ = build()
    widths = run(r, 66, {"s": [3, 3, 2, 3, 4], "l": [3, 3, 3, 4, 4]})
    wide = sum(1 for w in widths if w == 16)
    assert wide >= 60, (wide, widths)


def test_a_rejection_rate_is_counted_rather_than_inferred_from_how_much_was_wasted():
    """Four of seven and four of fifteen have both rejected, and the old model said 41 % against
    73 %. With the verify curve flat that difference was the largest term left in the pricing and
    it was handing the narrow arm a discount it had not earned."""
    r, _, _ = build()
    run(r, 20, {"s": [3], "l": [3]})                 # never accepts a whole block, either width
    assert r.rej.value > 0.9
    assert r._p_reject(16, 3.0) > 0.9
    assert r._p_reject(8, 3.0) > 0.9                 # and the narrow arm is not let off
    # a fixture that DOES fill the narrow block prices the narrow rollback away, as it should
    r2, _, _ = build()
    r2.fixed = 8
    run(r2, 20, {"s": [15], "l": [15]})
    assert r2._p_reject(8, 7.0) < 0.1


# --- the asymmetry ---------------------------------------------------------------------------------

def test_the_narrow_option_is_priced_from_wide_blocks_for_free():
    """A wide block that accepted 3 says exactly what a narrow one would have committed: 4."""
    r, _, _ = build()
    r.fixed = 16
    run(r, 6, {"s": [3], "l": [3]})
    assert abs(r.acc[("l", 8)].value - 4.0) < 1e-9


def test_coming_back_down_costs_at_most_a_handful_of_blocks():
    """Pinned wide, then released, on text where the narrow arm is the better one.

    Under phase 5's curve this needed no experiment at all: the truncation of the router's own
    wide blocks priced the narrow option exactly, and coming down was free. Under `wide_default`
    it costs a bounded few blocks, and the reason is worth stating because it is the one thing the
    free counterfactual cannot do. The counterfactual prices the WIDE drafter's draft cut short.
    The narrow arm is a different checkpoint and drafts its own seven slots better, and that
    difference -- 4.6 % on `prose`, 5 % of the row -- is only visible by running it.
    """
    r, small, _ = build(explore_period=1000)
    r.fixed = 16
    run(r, 12, {"s": [2, 2, 2, 2, 1], "l": [2, 2, 1, 1, 2]})     # wide blocks, badly accepted
    calls_before = small.calls
    r.fixed = 0
    widths = run(r, 16, {"s": [2, 2, 2, 2, 1], "l": [2, 2, 1, 1, 2]})
    assert small.calls > calls_before       # it did come back down
    assert widths[-1] == 8, widths
    assert r.stats["probes"] <= 4, r.report()


def test_going_up_is_forced_because_the_narrow_number_is_censored():
    """With the ceiling signal suppressed the router still probes, on the slow schedule: a policy
    that only learns about the option it took keeps taking it."""
    r, _, large = build(explore_period=8, ceiling_trigger=2.0)
    widths = run(r, 40, {"s": [2], "l": [2]})
    assert 16 in widths
    assert large.calls >= 4


def test_a_saturating_narrow_block_shortens_the_probe_schedule():
    """Once the router is on the narrow block, the ceiling rate is what buys the next wide probe.

    Both routers are given a bad wide history first, so the expected-value rule keeps them narrow,
    and then a narrow stretch that accepts every slot it has. That is the censored observation: the
    truth is "at least seven" and what sixteen would have committed is not in the data. The router
    that reads the ceiling rate buys the answer within four blocks; the one with the trigger
    switched off waits for its slow schedule and never finds out.
    """
    slow, _, _ = build(explore_period=64, ceiling_trigger=2.0)   # never triggered
    fast, _, _ = build(explore_period=64, ceiling_period=4, ceiling_trigger=0.25)
    for r in (slow, fast):
        r.fixed = 16
        run(r, 4, {"s": [2], "l": [2]})          # the wide arm looks bad
        r.fixed = 8
        run(r, 8, {"s": [15], "l": [15]})        # and the narrow one saturates
        r.fixed = 0
    w_slow = run(slow, 16, {"s": [15], "l": [15]})
    w_fast = run(fast, 16, {"s": [15], "l": [15]})
    assert sum(1 for w in w_fast if w == 16) > sum(1 for w in w_slow if w == 16), (w_slow, w_fast)
    # Not exactly 1.0 any more, and that is the phase-9 change showing up in an old test: the
    # ceiling rate is now measured on WIDE blocks as well, and the four badly-accepted wide blocks
    # at the top of this fixture are four observations that the narrow width would have had slots
    # to spare. Under the old code the signal existed only on an arm `wide_default` rarely runs.
    assert slow.ceiling.value > 0.99 and fast.ceiling.value > 0.0


# --- the tree arms -------------------------------------------------------------------------------

class FakeTree:
    def __init__(self, anchor, toks):
        self.tokens = [anchor] + list(toks)
        self.parents = [-1] + list(range(len(toks)))

    @property
    def n_draft(self):
        return len(self.tokens) - 1

    def depths(self):
        d = [0] * len(self.tokens)
        for i in range(1, len(self.tokens)):
            d[i] = d[self.parents[i]] + 1
        return d


class FakeNgram:
    """The lookup drafter both arms share. It counts how often it is told about a block."""

    def __init__(self):
        self.primed = 0
        self.observed = 0

    def prime(self, tokens):
        self.primed += 1

    def observe(self, tokens):
        self.observed += 1

    def reset(self):
        pass


class FakeArm:
    """A MergedRouter stand-in: it wraps a block drafter as `.mtp` and extends the shared index."""

    wants_rows = True

    def __init__(self, head, ngram, budget):
        self.mtp = head
        self.ngram = ngram
        self.budget = budget
        self.calls = 0

    def propose_tree(self, context, k):
        self.calls += 1
        n = min(k, self.budget)
        return FakeTree(context[-1], [9000 + i for i in range(n)]) if n else None

    def observe(self, tokens):
        self.ngram.observe(tokens)

    def reset(self):
        pass

    def prime(self, tokens):
        self.ngram.prime(tokens)

    def sync(self, tokens, hidden, first_pos, rows=None):
        self.mtp.sync(tokens, hidden, first_pos, rows=rows)


def build_tree(**kw):
    eng = FakeEng()
    small = FakeDrafter(eng, 8)
    large = FakeDrafter(eng, 16)
    ng = FakeNgram()
    kw.setdefault("learn_cost", False)
    r = LengthRouter(FakeArm(small, ng, 7), FakeArm(large, ng, 15),
                     tree=True, ngram=ng, **kw)
    return r, small, large, ng


def run_tree(router, blocks, runs):
    idx = {"s": 0, "l": 0}
    widths = []
    for _ in range(blocks):
        tree = router.propose_tree(list(range(50)), 15)
        key, width = router.last_key, router.last_width
        widths.append(width)
        seq = runs[key]
        n = min(seq[idx[key] % len(seq)], max(width - 1, 0))
        idx[key] += 1
        router.observe([9000 + i for i in range(n)] + [12345])
    return widths


def test_the_tree_arms_share_one_lookup_index():
    """Two arms, one suffix memory, one update a block. Indexing every token twice would make the
    store disagree with the text it is a memory of."""
    r, _, _, ng = build_tree()
    r.prime(list(range(20)))
    assert ng.primed == 1
    run_tree(r, 10, {"s": [3], "l": [3]})
    assert ng.observed == 10


def test_the_tree_path_prices_on_the_tree_curve():
    """A tree pays its commit on every block and a chain pays a rollback only on a rejection."""
    chain, _, _ = build()
    tree, _, _, _ = build_tree()
    assert tree._cost_ms("l", 16, 2.0) > tree.vms[16].value + tree.dms["l"].value
    # the commit is flat in the accepted length; the chain's rollback is not
    assert abs(tree._cost_ms("l", 16, 2.0) - tree._cost_ms("l", 16, 15.0)) < 1e-9
    assert chain._cost_ms("l", 16, 2.0) > chain._cost_ms("l", 16, 15.0)


def test_the_tree_router_still_finds_the_wide_budget():
    r, _, _, _ = build_tree(explore_period=32)
    widths = run_tree(r, 40, {"s": [15], "l": [15]})
    assert sum(1 for w in widths[-20:] if w == 16) >= 18, widths


# --- the pins ------------------------------------------------------------------------------------

def test_fixed_pins_the_width_through_the_same_code():
    r, small, large = build(fixed=8)
    assert set(run(r, 10, {"s": [3], "l": [3]})) == {8}
    assert large.calls == 0
    r2, small2, large2 = build(fixed=16)
    assert set(run(r2, 10, {"s": [3], "l": [3]})) == {16}
    assert small2.calls == 0


def test_a_short_budget_never_asks_for_a_wide_block():
    """Near the end of a generation the loop asks for fewer tokens than a wide block proposes.

    The short block's evidence still has to land somewhere: it belongs to the narrow arm, not to a
    width of five that the router could never choose on purpose.
    """
    r, _, large = build()
    draft = r.propose(list(range(50)), 4)
    assert len(draft) <= 4 and r.last_width <= 5 and large.calls == 0
    r.observe(draft[:2] + [1])
    assert r.acc[("s", 8)].n == 1 and (("s", 5) not in r.acc)


def test_the_loop_teaches_the_router_what_a_block_cost():
    r, _, _ = build(learn_cost=True)
    before = r.vms[16].value
    for _ in range(8):
        r.on_verify(16, 200.0)
    assert r.vms[16].value > before


# --- the latch: one decision a request -----------------------------------------------------------

def test_the_latch_measures_then_stops_deciding():
    """Four wide blocks, up to four narrow probes, one decision, and no switch after it.

    `mix3` measured what a switch costs -- 3 % on chat, 10 % on code, 18 % on quote, with no policy
    involved at all -- so the number of switches is itself a thing to minimise, and this shape has
    two of them however long the generation runs.
    """
    r, small, large = build(latch=True)
    widths = run(r, 60, {"s": [2, 2, 2, 2, 1], "l": [2, 2, 1, 1, 2]})
    assert r.latched == "s", r.report()
    assert set(widths[-40:]) == {8}, widths[-40:]
    # exactly two transitions: wide -> narrow probes -> latched
    flips = sum(1 for a, b in zip(widths, widths[1:]) if a != b)
    assert flips <= 2, widths


def test_the_latch_settles_wide_where_the_wide_arm_earns_it():
    r, _, _ = build(latch=True)
    widths = run(r, 60, {"s": [3, 3, 2, 3, 4], "l": [3, 3, 3, 4, 4]})
    assert r.latched == "l", r.report()
    assert set(widths[-40:]) == {16}, widths[-40:]


def test_a_saturating_narrow_width_is_never_probed_and_latches_at_once():
    """`quote`: the narrow block accepts every slot it has, so there is nothing down there to
    learn and the decision is taken at block four without spending anything on it."""
    r, small, _ = build(latch=True)
    widths = run(r, 30, {"s": [15], "l": [15]})
    assert r.stats["probes"] == 0 and small.calls == 0, r.report()
    assert r.latched == "l" and set(widths) == {16}, widths


def test_the_latch_is_a_belief_about_the_text_and_does_not_survive_the_request():
    r, _, _ = build(latch=True)
    run(r, 30, {"s": [2, 2, 2, 2, 1], "l": [2, 2, 1, 1, 2]})
    assert r.latched == "s"
    r.reset()
    assert r.latched is None
    # and the COSTS do survive it, because they are properties of the board
    assert r.vms[16].n == 0 or r.vms[16].value > 0


# --- phase 10: the arm that loses the latch --------------------------------------------------------

def test_the_loser_is_released_once_and_stops_being_synced():
    """The latch decides, and from that block on the router stops paying for the other arm."""
    r, small, large = build(latch=True, drop_idle=True)
    run(r, 60, {"s": [3, 3, 2, 3, 4], "l": [3, 3, 3, 4, 4]})       # latches wide
    assert r.latched == "l" and r.idle == "s"
    assert small.released and not large.released
    before = len(small.synced)
    r.sync([1, 2, 3], None, 500)
    assert len(small.synced) == before                              # the released arm is not synced
    assert large.synced[-1] == (3, 500)


def test_the_release_keeps_the_draft_kv_by_default():
    """Freeing it returns 655 MB at the shipped max-len to a pool nothing else can spend, and
    charges the next request about 6 ms of its first token to zero it again."""
    r, small, large = build(latch=True, drop_idle=True)
    run(r, 60, {"s": [3, 3, 2, 3, 4], "l": [3, 3, 3, 4, 4]})
    assert small.released and small.freed is False


def test_the_tap_stops_going_to_the_released_arm():
    r, small, large = build(latch=True, drop_idle=True)
    run(r, 60, {"s": [3, 3, 2, 3, 4], "l": [3, 3, 3, 4, 4]})
    a, b = small.taps, large.taps
    r._on_tap(None)
    assert small.taps == a and large.taps == b + 1


def test_the_tail_of_a_generation_stays_on_the_arm_that_can_draft_it():
    """The rule that sent a short block to the narrow arm was written when a wide verify cost
    13.9 % more. With the narrow arm released it would decline, and the last fifteen tokens would
    come out one a step."""
    r, small, large = build(latch=True, drop_idle=True)
    run(r, 60, {"s": [3, 3, 2, 3, 4], "l": [3, 3, 3, 4, 4]})
    assert r._choose(4) == "l"                                      # not "s"
    calls = small.calls
    r.propose(list(range(50)), 4)
    assert small.calls == calls and r.last_key == "l"


def test_a_release_that_can_never_fire_is_refused():
    """`drop_idle` without the latch would accept a flag and never act on it."""
    try:
        build(drop_idle=True)
    except ValueError as exc:
        assert "latch" in str(exc)
    else:
        raise AssertionError("drop_idle without latch was accepted")


def test_without_the_flag_nothing_is_released():
    r, small, large = build(latch=True)
    run(r, 60, {"s": [3, 3, 2, 3, 4], "l": [3, 3, 3, 4, 4]})
    assert r.idle is None and not small.released
    assert r._choose(4) == "s"                                      # the old rule, unchanged


def test_a_released_arm_is_absent_from_the_snapshot_rather_than_stale():
    r, small, large = build(latch=True, drop_idle=True)
    run(r, 60, {"s": [3, 3, 2, 3, 4], "l": [3, 3, 3, 4, 4]})
    snap = r.state_snapshot()
    assert snap[0] == "lenrouter" and snap[1] is None and snap[3] == "s"
    assert snap[2] is not None


def test_restoring_a_one_armed_snapshot_latches_to_the_arm_it_carries():
    """The missing arm cannot be made current without forwarding the prefix the restore skipped,
    so the request inherits the decision the snapshot was taken under."""
    r, small, large = build(latch=True, drop_idle=True)
    run(r, 60, {"s": [3, 3, 2, 3, 4], "l": [3, 3, 3, 4, 4]})
    snap = r.state_snapshot()
    r2, small2, large2 = build(latch=True, drop_idle=True)
    r2.state_restore(snap)
    assert r2.latched == "l" and r2.idle == "s"
    assert small2.restored is None and large2.restored is not None
    assert r2._choose(15) == "l" and r2._choose(3) == "l"


def test_an_old_three_element_snapshot_still_restores():
    r, small, large = build(latch=True, drop_idle=True)
    r.state_restore(("lenrouter", small.state_snapshot(), large.state_snapshot()))
    assert r.idle is None and r.latched is None


def test_the_release_does_not_survive_the_request():
    r, small, large = build(latch=True, drop_idle=True)
    run(r, 60, {"s": [3, 3, 2, 3, 4], "l": [3, 3, 3, 4, 4]})
    assert r.idle == "s"
    r.reset()
    assert r.idle is None
    r.sync([1, 2], None, 0)
    assert small.synced[-1] == (2, 0)                               # both arms current again


def test_the_narrow_latch_releases_the_wide_arm():
    r, small, large = build(latch=True, drop_idle=True)
    run(r, 60, {"s": [2, 2, 2, 2, 1], "l": [2, 2, 1, 1, 2]})        # latches narrow
    assert r.latched == "s" and r.idle == "l"
    assert large.released and not small.released
    # and the wide arm is never asked again, including where the wide fallback used to run
    calls = large.calls
    r.propose(list(range(50)), 15)
    assert large.calls == calls


# --- the commit histogram ------------------------------------------------------------------------

def test_the_commit_histogram_counts_every_block_by_arm():
    """Five full sixteen-wide blocks and five that commit four: the histogram says so, the arm cap
    counts exactly the full ones, and the report carries it on one space-free token."""
    r, _, _ = build(fixed=16)
    run(r, 10, {"s": [15], "l": [15, 3]})
    assert r.stats["commit_hist"] == {16: {16: 5, 4: 5}}, r.stats["commit_hist"]
    assert r.stats["cap_arm"] == 5 and r.stats["cap_depth"] == 5
    assert "commits 16:4x5,16x5 cap arm 5 depth 5" in r.report(), r.report()


def test_a_short_last_block_is_capped_by_its_depth_not_by_its_arm():
    """The last block of a generation proposes what is left of the budget. Committing all of it is
    a depth cap; it is not the arm running out of width, and a deeper draft could not have helped."""
    r, _, _ = build(fixed=8)
    r.propose(list(range(50)), 3)                              # three drafted, width four
    r.observe([9000, 9001, 9002, 12345])
    assert r.stats["commit_hist"] == {8: {4: 1}}
    assert r.stats["cap_arm"] == 0 and r.stats["cap_depth"] == 1


def test_a_bushy_tree_is_capped_at_its_depth():
    """A tree of fifteen nodes that is only three deep can commit at most four; doing so is a depth
    cap on the wide arm, not an arm cap."""
    r, _, _, _ = build_tree(fixed=16)
    tree = FakeTree(7, [9000 + i for i in range(15)])
    tree.parents = [-1, 0, 1, 2] + [0] * 12                    # one chain of three, twelve leaves
    r.large.propose_tree = lambda context, k: tree
    got = r.propose_tree(list(range(50)), 15)
    assert got is tree and r.last_depth == 3
    r.observe([9000, 9001, 9002, 12345])
    assert r.stats["commit_hist"] == {16: {4: 1}}
    assert r.stats["cap_arm"] == 0 and r.stats["cap_depth"] == 1


def test_the_commit_histogram_is_per_request():
    r, _, _ = build(fixed=16)
    run(r, 4, {"s": [15], "l": [15]})
    r.reset()
    assert r.stats["commit_hist"] == {} and r.stats["cap_arm"] == 0 and r.stats["cap_depth"] == 0
    assert "commits - cap arm 0 depth 0" in r.report()


# --- an arm's tree past its block size -----------------------------------------------------

class FakeBudgetArm(FakeArm):
    """A MergedRouter with its own node budget: it builds `budget` nodes whatever depth it is asked
    for, as the real one does when QWEN38_TREE_NODES(_NARROW) gives it more than its block."""

    def propose_tree(self, context, k):
        self.calls += 1
        return FakeTree(context[-1], [9000 + i for i in range(self.budget)])


def test_a_narrow_tree_as_wide_as_the_wide_arm_is_still_the_narrow_arms_evidence():
    """With the narrow arm's budget at 16 its tree has 16 rows. The router used to read the
    arm off the width, so a 16-row narrow tree was booked to the wide arm -- and the narrow arm's
    acceptance key for a wide width does not exist (KeyError ('s', 16) in the bench, 2026-09-24)."""
    eng = FakeEng()
    small, large, ng = FakeDrafter(eng, 8), FakeDrafter(eng, 16), FakeNgram()
    r = LengthRouter(FakeBudgetArm(small, ng, 15), FakeBudgetArm(large, ng, 15), tree=True,
                     ngram=ng, learn_cost=True, fixed=8)
    vs, vl = r.vms[8].value, r.vms[16].value
    tree = r.propose_tree(list(range(50)), 15)
    assert r.last_key == "s" and tree.n_draft == 15
    r.on_verify(tree.n_draft + 1, 55.0)
    r.observe([9000, 9001, 9002, 12345])
    assert r.acc[("s", 8)].n == 1 and r.acc[("s", 8)].value == 4.0
    assert r.vms[16].value == vl, "the wide arm's price is not the narrow arm's block"
    assert r.vms[8].value != vs or r.vms[8].n == 1
    assert r.stats["commit_hist"][8] == {4: 1}
    assert r.since[8] == 0 and r.since[16] == 1
    assert r.ceiling.n == 1 and r.ceiling.value == 0.0, "3 of 7 slots: the narrow arm had room"
    r.propose_tree(list(range(50)), 15)
    r.observe([9000 + i for i in range(7)] + [12345])        # all seven slots of its lattice
    assert r.ceiling.n == 2 and r.stats["ceiling_hits"] == 1


def test_the_wide_tree_waits_for_the_request_to_have_committed_enough():
    """The delayed wide tree: until `tree_wide_after` tokens of the request are committed the
    wide arm builds its served 16-node chain, from then on its configured budget. A short answer --
    the row's 41-token max request -- never pays a 24-row verify for branches it cannot use; a new
    request starts short again."""
    eng = FakeEng()
    small, large, ng = FakeDrafter(eng, 8), FakeDrafter(eng, 16), FakeNgram()
    wide = FakeBudgetArm(large, ng, 23)
    wide.node_budget = wide.head_budget = 23
    r = LengthRouter(FakeArm(small, ng, 7), wide, tree=True, ngram=ng, learn_cost=False, fixed=16,
                     tree_wide_after=32)
    budgets = []

    def one(commit):
        t = r.propose_tree(list(range(50)), 15)
        budgets.append(wide.head_budget)
        r.observe([9000 + i for i in range(commit - 1)] + [12345])
        return t
    for _ in range(3):
        one(10)                                    # 30 tokens: still the chain
    assert budgets == [15, 15, 15], budgets
    one(10)                                        # 40 committed before the 5th block
    one(10)
    assert budgets[-1] == 23, budgets
    r.reset()
    one(4)
    assert budgets[-1] == 15, "a new request starts on the chain"
    wide.node_budget = wide.head_budget = 23
    r2 = LengthRouter(FakeArm(small, ng, 7), wide, tree=True, ngram=ng, learn_cost=False, fixed=16)
    r2.propose_tree(list(range(50)), 15)
    assert wide.head_budget == 23 and r2.tree_wide_after == 0, "off: the configured budget always"


# --- the deep chain ------------------------------------------------------------------------------

class FakeLocal:
    def __init__(self, n):
        self.n = n

    def lookup(self, context, min_order):
        return self.n, [0]


class FakeNgramDeep(FakeNgram):
    """A lookup drafter that has a long exact continuation: `n` tokens of local match and the
    candidate list `cands`, as `NgramDrafter.candidates` returns it."""

    min_order = 3

    def __init__(self, n, cands):
        super().__init__()
        self.local = FakeLocal(n)
        self.cands = cands
        self.asked = 0

    def candidates(self, context, depth):
        self.asked += 1
        return self.local.n, [(c[:depth], w) for c, w in self.cands]


def build_deep(n=12, cands=None, deep=32, deep_after=1):
    eng = FakeEng()
    small = FakeDrafter(eng, 8)
    large = FakeDrafter(eng, 16)
    ng = FakeNgramDeep(n, cands if cands is not None else [(list(range(500, 540)), 3.0),
                                                            ([7, 7, 7], 1.0)])
    r = LengthRouter(FakeArm(small, ng, 7), FakeArm(large, ng, 15), tree=True, ngram=ng,
                     learn_cost=False, fixed=16, deep=deep, deep_after=deep_after)
    return r, ng


def _full_wide_block(r):
    tree = r.propose_tree(list(range(50)), 31)
    assert not r.last_deep and r.last_width == 16
    r.observe([9000 + i for i in range(15)] + [12345])


def test_the_deep_chain_follows_a_full_wide_block():
    r, ng = build_deep()
    _full_wide_block(r)
    tree = r.propose_tree(list(range(50)), 31)
    assert r.last_deep and tree.n_draft == 31 and tree.tokens[1:4] == [500, 501, 502]
    assert tree.parents == [-1] + list(range(31))
    vms = r.vms[16].value
    r.on_verify(32, 999.0)
    assert r.vms[16].value == vms                  # the arm's price is not the deep block's
    r.observe(list(range(500, 531)) + [12345])
    assert r.stats["commit_hist"][32] == {32: 1} and r.stats["deep"] == 1
    assert r.propose_tree(list(range(50)), 31).n_draft == 31   # full again: deep again
    assert r.stats["deep"] == 2 and "deep 2 (32 tok)" in r.report(), r.report()


def test_a_deep_chain_that_breaks_hands_back_to_the_arms():
    r, _ = build_deep()
    _full_wide_block(r)
    r.propose_tree(list(range(50)), 31)
    r.observe([500, 501, 502, 12345])              # rejected at slot 4
    r.propose_tree(list(range(50)), 31)
    assert not r.last_deep and r.last_width == 16


def test_no_deep_chain_without_a_long_local_match_or_a_clear_winner():
    for n, cands in ((5, None), (12, [(list(range(40)), 1.0), (list(range(1, 41)), 1.5)]),
                     (12, [(list(range(10)), 3.0)])):
        r, _ = build_deep(n=n, cands=cands)
        _full_wide_block(r)
        r.propose_tree(list(range(50)), 31)
        assert not r.last_deep, (n, cands)


def test_the_policy_waits_for_two_full_blocks_in_a_row():
    """One full wide block happens on new text; two in a row is a copy."""
    r, ng = build_deep(deep_after=2)
    _full_wide_block(r)
    r.propose_tree(list(range(50)), 31)
    assert not r.last_deep and ng.asked == 0, "one full block is not enough"
    r.observe([9000 + i for i in range(15)] + [12345])          # the second full block
    tree = r.propose_tree(list(range(50)), 31)
    assert r.last_deep and tree.n_draft == 31
    r.observe(list(range(500, 531)) + [12345])                  # the deep block commits in full
    r.propose_tree(list(range(50)), 31)
    assert r.last_deep, "a full deep block keeps the run"
    r.observe([500, 12345])                                     # it breaks
    r.propose_tree(list(range(50)), 31)
    assert not r.last_deep
    r.observe([9000 + i for i in range(15)] + [12345])          # one full block after the break
    r.propose_tree(list(range(50)), 31)
    assert not r.last_deep, "the run starts over after a break"


def test_a_partial_block_between_two_full_ones_resets_the_run():
    r, _ = build_deep(deep_after=2)
    _full_wide_block(r)
    r.propose_tree(list(range(50)), 31)
    r.observe([9000, 9001, 12345])
    _full_wide_block(r)
    r.propose_tree(list(range(50)), 31)
    assert not r.last_deep


def test_off_never_asks_the_lookup_for_a_deep_chain():
    r, ng = build_deep(deep=0)
    _full_wide_block(r)
    r.propose_tree(list(range(50)), 31)
    assert ng.asked == 0 and not r.last_deep
    assert "deep" not in r.report()


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
