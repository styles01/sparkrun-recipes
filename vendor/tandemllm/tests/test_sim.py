"""The simulator's accounting, checked against cases whose answer can be worked out by hand.

The simulator is the instrument every tuning decision is read off, so the thing worth testing is
not that it runs but that its numbers mean what the ledger will say they mean.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tree import DraftTree  # noqa: E402
from tools.sim_draft import (MTPPolicy, NoDrafter, Policy, Trace, TreePolicy,  # noqa: E402
                             nvfp4_verify_ms, simulate)

BASE, PER_NODE, ROLLBACK = 149.1, 1.896, 22.0


def _trace(out, prompt=(1, 2, 3), mtp=None):
    return Trace(name="t", klass="test", prompt_ids=list(prompt), output_ids=list(out),
                 mtp_at=dict(mtp or {}))


class _Oracle(Policy):
    """Proposes exactly what the target will write. The upper bound on any drafter of this depth."""

    name = "oracle"

    def __init__(self, depth):
        self.depth = depth

    def block(self, ctx, i, trace):
        nxt = trace.output_ids[i:i + self.depth]
        if not nxt:
            return None, 0.0
        return DraftTree.chain(ctx[-1] if ctx else 0, list(nxt), source="oracle"), 0.0


def test_no_drafter_is_one_token_per_base_step():
    t = _trace(list(range(40)))
    r = simulate(NoDrafter(), t, BASE, PER_NODE, ROLLBACK)
    assert r.tokens == 40 and r.steps == 40 and r.fired == 0
    assert abs(r.cost_ms - 40 * BASE) < 1e-6
    assert abs(r.tok_s - 1000.0 / BASE) < 1e-6


def test_every_target_token_is_emitted_exactly_once():
    t = _trace(list(range(100)))
    for p in (NoDrafter(), _Oracle(8)):
        r = simulate(p, t, BASE, PER_NODE, ROLLBACK)
        assert r.tokens == 100, p.name


def test_oracle_matches_the_cost_model_by_hand():
    t = _trace(list(range(81)))   # nine blocks of nine tokens, no short block at the end
    r = simulate(_Oracle(8), t, BASE, PER_NODE, ROLLBACK)
    # every block accepts all 8 drafted tokens plus the model's own: 9 tokens per step, no rollback
    assert r.rollbacks == 0
    assert r.tau == 8.0
    assert r.saturation == 1.0
    expected = 9.0 / ((BASE + 8 * PER_NODE) / 1000.0)
    assert abs(r.tok_s - expected) < 0.05


def test_the_cost_model_reproduces_the_board_baseline():
    """A model of a step is worth nothing if it does not predict the step the board measured.

    The board read 8.01 tok/s with no drafter on the NVFP4 weight set. The
    curve in this module has to land on that number, or every tok/s it prints downstream is a
    number about the model rather than about the engine.
    """
    t = _trace(list(range(100)))
    r = simulate(NoDrafter(), t, BASE, PER_NODE, ROLLBACK, verify=nvfp4_verify_ms)
    assert abs(r.tok_s - 8.01) / 8.01 < 0.05, f"{r.tok_s:.2f} against a measured 8.01"


def test_rollback_is_charged_only_when_a_node_is_rejected():
    # the oracle is never wrong; a drafter that is always wrong pays a rollback every block
    class _Wrong(Policy):
        name = "wrong"

        def block(self, ctx, i, trace):
            return DraftTree.chain(0, [-1, -2, -3], source="x"), 0.0

    t = _trace(list(range(30)))
    r = simulate(_Wrong(), t, BASE, PER_NODE, ROLLBACK)
    assert r.rollbacks == r.steps == 30
    assert r.accepted == 0
    assert abs(r.cost_ms - 30 * (BASE + 3 * PER_NODE + ROLLBACK)) < 1e-6
    # and it is strictly worse than not drafting at all, which is the point of a firing policy
    assert r.tok_s < simulate(NoDrafter(), t, BASE, PER_NODE, ROLLBACK).tok_s


def test_a_tree_can_accept_a_branch_a_chain_would_miss():
    class _TwoBranch(Policy):
        name = "two"

        def block(self, ctx, i, trace):
            right = trace.output_ids[i:i + 2]
            return DraftTree.from_sequences(0, [([-1, -2], 0.6), (list(right), 0.4)]), 0.0

    t = _trace(list(range(40)))
    r = simulate(_TwoBranch(), t, BASE, PER_NODE, ROLLBACK)
    # every full block accepts both tokens of the right branch; only the last one is short
    assert r.tau > 1.9, "the correct branch is accepted even though it is not the best-scored one"


def test_mtp_replay_is_exact_where_it_was_recorded():
    out = list(range(50))
    # at position 0 the head proposed the next three tokens correctly, at 4 it proposed rubbish
    mtp = {0: [0, 1, 2], 4: [-1, -1, -1]}
    t = _trace(out, mtp=mtp)
    r = simulate(MTPPolicy(3, random.Random(0)), t, BASE, PER_NODE, ROLLBACK)
    assert r.known >= 2
    assert r.coverage < 1.0, "the rest of the stream is off the recorded grid and is drawn"


def test_lookup_drafter_fires_on_a_repeat_and_declines_on_noise():
    from engine.drafters.ngram import NgramDrafter
    rng = random.Random(5)
    noise = [rng.randrange(5000) for _ in range(120)]
    repeated = list(range(7000, 7040))
    out = noise + repeated + repeated
    t = _trace(out, prompt=[9999])
    p = TreePolicy(lambda: NgramDrafter(corpus_path="", min_expected=0.5), "ngram", 16, 16)
    r = simulate(p, t, BASE, PER_NODE, ROLLBACK)
    assert r.tokens == len(out)
    assert 0.0 < r.fire_rate < 1.0, "it must decline on the noise and fire on the repeat"
    assert r.tau_fire > 3.0, "when it fires on an exact repeat it should get most of the block"
    assert r.tok_s > simulate(NoDrafter(), t, BASE, PER_NODE, ROLLBACK).tok_s


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
