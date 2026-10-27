"""The lookup's continuation rate (the copy estimator of the staircase cut, QWEN38_STAIR_RHO).

The lookup drafter records its top continuation every time it looks, and the arm scores that line
against the block the target committed, whichever candidate the arm submitted. These tests check
that the counts update, that the rate converges on a copy stream of known rate, and that nothing
changes with the estimator off.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.drafters.ngram import NgramDrafter  # noqa: E402
from engine.router import MergedRouter  # noqa: E402
from tests.test_lenrouter import FakeDrafter, FakeEng  # noqa: E402


def _arm(rho=True, online=True):
    ng = NgramDrafter(corpus_path="", min_order=3, max_depth=16, node_budget=31,
                      min_expected=0.0)
    head = FakeDrafter(FakeEng(), 16)
    arm = MergedRouter(ng, head, mtp_depth=15, node_budget=31, mtp_ms_per_token=0.0,
                       head_fixed_ms=15.0, adaptive_depth=False)
    arm.stair = True
    arm.stair_rho = rho
    arm.rho_online = online
    arm.rho_min_m = 3
    return arm, ng


PASSAGE = list(range(1000, 1060))


def test_the_lookup_records_its_top_line():
    _, ng = _arm()
    ng.prime(PASSAGE + [7, 8] + PASSAGE[:10])
    m, cands = ng.candidates(PASSAGE + [7, 8] + PASSAGE[:10], 16)
    assert m >= 8 and ng.last_top == PASSAGE[10:26]
    ng.candidates([1, 2, 3], 16)                          # nothing matches: no stale line
    assert ng.last_top == []


def test_the_counts_update_from_every_committed_block():
    arm, ng = _arm()
    ctx = PASSAGE + [7, 8] + PASSAGE[:10]
    ng.prime(ctx)
    arm.ngram.rho_fn = arm._rho
    ng.propose_tree(ctx, 16)
    arm.ngram.rho_fn = None
    arm._rho_top = list(ng.last_top)
    assert arm._rho_key is not None and arm._rho_key[1] == 8
    arm._rho_observe(PASSAGE[10:15] + [9999])             # five followed, then a miss
    assert arm.rho_counts[arm._rho_key] == [5.0, 1.0]
    assert arm.copy_run == 0
    arm._rho_top = list(PASSAGE[15:31])
    arm._rho_observe(PASSAGE[15:27])                      # the whole block followed the line
    assert arm.rho_counts[arm._rho_key] == [17.0, 1.0], "censored: no failure"
    assert arm.copy_run == 12
    arm._rho_top = None                                   # a round this arm did not propose
    arm._rho_observe(PASSAGE[27:40])
    assert arm.copy_run == 25 and arm.rho_counts[arm._rho_key] == [17.0, 1.0]


def test_the_rate_converges_on_a_copy_stream():
    """Each round the committed block follows the lookup's 16-token line for a geometric number
    of tokens with continuation rate `rho`, then misses (or the line runs out)."""
    for rho in (0.6, 0.9, 0.97):
        arm, _ = _arm()
        arm.rho_prior = {("local", 8, 0): [5.0, 5.0]}
        rnd = random.Random(7)
        key = ("local", 8, 0)
        for _ in range(3000):
            arm._rho_key = key
            top = [rnd.randrange(10, 99) for _ in range(16)]
            n = 0
            while n < 16 and rnd.random() < rho:
                n += 1
            tokens = top[:n] + ([1] if n < 16 else [])
            arm._rho_top = top
            arm._rho_observe(tokens)
            arm.copy_run = 0
        got = arm._rho("local", 8)
        assert abs(got - rho) < 0.02, (rho, got)


def test_the_counts_forget_slowly():
    arm, _ = _arm()
    arm.rho_cap = 100.0
    key = ("local", 8, 0)
    for _ in range(60):
        arm._rho_key = key
        arm._rho_top = [1, 2, 3, 4]
        arm._rho_observe([1, 2, 9])
    s_, f_ = arm.rho_counts[key]
    assert s_ + f_ <= 100.0 and abs(s_ / (s_ + f_) - 2 / 3) < 0.02


def test_off_counts_nothing_and_the_tree_is_the_drafters_own():
    ctx = PASSAGE + [7, 8] + PASSAGE[:10]
    plain, ngp = _arm(rho=False)
    ngp.prime(ctx)
    t_plain = ngp.propose_tree(ctx, 16)
    ref = NgramDrafter(corpus_path="", min_order=3, max_depth=16, node_budget=31,
                       min_expected=0.0)
    ref.prime(ctx)
    t_ref = ref.propose_tree(ctx, 16)
    assert t_plain.tokens == t_ref.tokens and t_plain.scores == t_ref.scores
    assert t_plain.parents == t_ref.parents
    plain._rho_top = list(ngp.last_top)
    plain.observe(PASSAGE[10:15] + [9999])
    assert plain.rho_counts == {} and plain.copy_run == 0
    prior_only, _ = _arm(online=False)
    prior_only._rho_key = ("local", 8, 0)
    prior_only._rho_top = list(PASSAGE[10:26])
    prior_only.observe(PASSAGE[10:15] + [9999])
    assert prior_only.rho_counts == {}, "the prior-only setting measured before the fix"


def test_no_head_skip_before_a_copy_run_exists():
    """An 8-token local match with no committed block behind it is priced by its rate but does not
    replace the head: on fresh text such a match is often a coincidence (row09, forced bench)."""
    from engine.lenrouter import LengthRouter
    from tests.test_router_stair import StairArm
    eng = FakeEng()
    r = LengthRouter(StairArm(FakeDrafter(eng, 8), 15), StairArm(FakeDrafter(eng, 16), 23),
                     tree=True, latch=True, learn_cost=False, switch=True, switch_mode="wide",
                     rho_prior={("local", 8, 0): [49.0, 1.0]})
    assert r.large.skip_min_bin == 1 and r.large.stair_rho
    arm, ng = _arm()
    arm.stair_skip = True
    arm.skip_min_bin = 1
    arm.rho_prior = {("local", 8, 0): [49.0, 1.0], ("local", 8, 1): [49.0, 1.0]}
    arm.rho_min_m = 8
    ctx = PASSAGE + [7, 8] + PASSAGE[:10]
    ng.prime(ctx)
    arm.propose_tree(ctx, 16)
    assert arm.stats.get("head_skipped", 0) == 0, "run 0: the head drafts"
    arm.copy_run = 5                                     # run bin 1
    arm.propose_tree(ctx, 16)
    assert arm.stats.get("head_skipped", 0) == 1, "a copy run behind the line: the lookup alone"


def test_a_rate_turns_the_decay_into_a_line_that_keeps_its_probability():
    ctx = PASSAGE + [7, 8] + PASSAGE[:10]
    ng = NgramDrafter(corpus_path="", min_order=3, max_depth=16, node_budget=31,
                      min_expected=0.0)
    ng.prime(ctx)
    decayed = ng.propose_tree(ctx, 16)
    ng.rho_fn = lambda src, m: 0.98
    kept = ng.propose_tree(ctx, 16)
    assert kept.scores[-1] > 0.7 > decayed.scores[-1]


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
