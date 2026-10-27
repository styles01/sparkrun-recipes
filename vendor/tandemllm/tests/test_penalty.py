"""The anti-repetition penalties, unit-tested on a CPU.

What is being tested is the property the whole design rests on: the penalties are a deterministic
function of the token history, applied to the target's row before the argmax -- so the four claims
below are about *which history each row sees* and *what the transform does to a row*, both of which
are pure tensor logic and need no model at all.

  1. the three formulas (multiplicative rep on negative/positive logits, flat presence,
     per-occurrence frequency) applied only to tokens the history contains;
  2. chain rows see committed history PLUS the drafts above them (vLLM's per-position history);
  3. tree rows see their ANCESTOR PATH, and a sibling leaves nothing behind;
  4. an off spec is a byte-identical no-op, and `seed` resets (the gate tools reuse one state).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from engine.penalty import PatternStop, PenaltySpec, PenaltyState  # noqa: E402
from engine.tree import DraftTree  # noqa: E402

V = 50


def state(rep=1.0, presence=0.0, freq=0.0):
    return PenaltyState(PenaltySpec(rep=rep, presence=presence, freq=freq), V, "cpu")


def row():
    # logits centred on zero so rep's two branches are both exercised
    r = torch.linspace(-2.0, 2.0, V, dtype=torch.float32)
    return r


def test_off_spec_is_a_no_op():
    ps = state()
    assert not ps.spec.on
    r = row()
    before = r.clone()
    ps.seed([3, 5, 3])
    ps.commit([7])
    ps.apply_single(r)
    ps.apply_chain(r.unsqueeze(0), [1, 2])
    assert torch.equal(r, before), "an off spec must not touch a row"
    assert ps.counts is None, "an off spec must not allocate the count vector"


def test_seed_resets_between_runs():
    ps = state(rep=1.05)
    ps.seed([1, 2, 3])
    ps.commit([4, 5])
    assert int(ps.counts.sum()) == 5
    ps.seed([9])
    assert int(ps.counts.sum()) == 1, "seed must reset, or a reused state double-counts"


def test_rep_formula_only_on_history_tokens():
    ps = state(rep=2.0)
    ps.seed([10])                       # token 10 occurs once
    r = row()
    ps.apply_single(r)
    ref = row()
    i = 10
    if ref[i] < 0:
        assert torch.isclose(r[i], ref[i] * 2.0)
    else:
        assert torch.isclose(r[i], ref[i] / 2.0)
    others = torch.cat([r[:i], r[i + 1:]])
    ref_others = torch.cat([ref[:i], ref[i + 1:]])
    assert torch.equal(others, ref_others), "fresh vocabulary must be untouched"


def test_presence_and_frequency():
    ps = state(presence=0.5, freq=0.1)
    ps.seed([10, 10, 11])                # 10 occurs twice, 11 once
    r = row()
    ps.apply_single(r)
    ref = row()
    assert torch.isclose(r[10], ref[10] - 0.5 - 0.1 * 2)
    assert torch.isclose(r[11], ref[11] - 0.5 - 0.1 * 1)
    assert torch.isclose(r[0], ref[0]), "absent tokens pay nothing"


def test_chain_rows_see_the_drafts_above_them():
    ps = state(presence=1.0)
    ps.seed([40])                        # committed history: token 40
    d0, d1 = 41, 42
    lg = torch.stack([row(), row(), row()])     # rows: anchor, d0's row, d1's row
    ps.apply_chain(lg, [d0, d1])
    ref = row()
    # row 0: history {40}
    assert torch.isclose(lg[0][40], ref[40] - 1.0)
    assert torch.isclose(lg[0][41], ref[41]), "a draft below is NOT in row 0's history"
    # row 1: history {40, d0}
    assert torch.isclose(lg[1][40], ref[40] - 1.0)
    assert torch.isclose(lg[1][41], ref[41] - 1.0), "d0 entered after row 0 was penalized"
    # row 2: history {40, d0, d1}
    assert torch.isclose(lg[2][42], ref[42] - 1.0)
    assert torch.isclose(lg[1][42], ref[42]), "d1 is not in row 1's history yet"


def test_tree_rows_see_their_ancestor_path_only():
    # anchor 30 (committed, in the base) with two children and one grandchild, in VALID DFS
    # pre-order (each subtree contiguous):  0:anchor  1:b(31)  2:d(33, child of b)  3:c(32)
    tree = DraftTree(tokens=[30, 31, 33, 32], parents=[-1, 0, 1, 0])
    tree.check()
    ps = state(presence=1.0)
    ps.seed([30])                        # the anchor is committed and already counted
    lg = torch.stack([row(), row(), row(), row()])
    ps.apply_tree(lg, tree)
    ref = row()
    # node 1 (b): history {30}
    assert torch.isclose(lg[1][30], ref[30] - 1.0)
    assert torch.isclose(lg[1][31], ref[31]), "a node's own token is not in its history"
    # node 2 (d, child of b): history {30, 31}
    assert torch.isclose(lg[2][31], ref[31] - 1.0), "the ancestor b IS in d's history"
    assert torch.isclose(lg[2][32], ref[32]), "the uncle c must not be in d's history"
    # node 3 (c, SIBLING of b): history {30} -- b and d must not have leaked
    assert torch.isclose(lg[3][31], ref[31]), "a sibling's token must not be in node 3's history"
    assert torch.isclose(lg[3][33], ref[33]), "the cousin d must not be in node 3's history"
    assert torch.isclose(lg[3][30], ref[30] - 1.0)


def test_deep_tree_walk_pops_correctly():
    # a two-branch tree: DFS visits 1,2 (the chain under b) then 3 (a second child of the anchor).
    # The anchor (10) is a COMMITTED token and lives in the base counts, exactly as the loops
    # guarantee: the tree drafter is handed `ctx` whose last token is the anchor it re-states as
    # node 0, and that token was committed by the previous block.
    tree = DraftTree(tokens=[10, 11, 12, 13], parents=[-1, 0, 1, 0])
    ps = state(presence=1.0)
    ps.seed([10])
    lg = torch.stack([row(), row(), row(), row()])
    ps.apply_tree(lg, tree)
    ref = row()
    assert torch.isclose(lg[2][11], ref[11] - 1.0), "node 2's parent b is in its history"
    assert torch.isclose(lg[3][11], ref[11]), "after popping back, b must be gone for node 3"
    assert torch.isclose(lg[3][10], ref[10] - 1.0), "the committed anchor is in node 3's history"


def test_ranges_are_validated():
    for bad in (lambda: PenaltySpec(rep=0.0), lambda: PenaltySpec(presence=2.5),
                lambda: PenaltySpec(freq=-3.0)):
        try:
            bad()
            raise AssertionError("out-of-range penalty must raise")
        except ValueError:
            pass


def test_pattern_stop_detects_a_single_token_loop():
    ps = PatternStop(max_size=8, min_size=1, count=4)
    assert not ps.observe([1, 2, 3])
    assert not ps.observe([4, 5, 6])
    assert ps.observe([7, 7, 7, 7]), "four repeats of one token is the pattern"
    assert ps.label.startswith("size=1 count=4"), ps.label
    assert "tok=[7]" in ps.label, "the label carries the pattern tokens for evidence"


def test_pattern_stop_detects_a_multi_token_cycle():
    ps = PatternStop(max_size=4, min_size=1, count=3)
    assert not ps.observe([10, 11, 10, 11])
    assert ps.observe([10, 11]), "two-token block, three times"


def test_pattern_stop_ignores_ordinary_text():
    ps = PatternStop(max_size=8, min_size=1, count=4)
    # no 4-fold repeat anywhere in this
    assert not ps.observe([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert not ps.hit


def test_pattern_stop_is_off_unless_asked_for():
    ps = PatternStop()                      # defaults: max 0, count 0
    assert not ps.on
    assert not ps.observe([5] * 100), "an off detector never fires"


def test_no_repeat_masks_the_completing_token():
    ps = state()
    ps.spec = PenaltySpec(no_repeat=3)          # a 3-gram may not repeat
    ps.seed([1, 2, 3, 4])                       # (1,2)->3, (2,3)->4
    ps.commit([1, 2, 3, 9])                     # (2,3)->9 as well
    ps.commit([2, 3])                           # the history now ends in (2,3)
    row = torch.zeros(V)
    ps._mask(row, [])
    # the tail suffix is (2,3); it was followed by 4 and by 9 before
    assert float(row[4]) == -ps.mask_penalty and float(row[9]) == -ps.mask_penalty
    assert float(row[1]) == 0.0, "unseen completions are untouched"


def test_no_repeat_chain_rows_carry_the_prefix():
    ps = state()
    ps.spec = PenaltySpec(no_repeat=2)          # a bigram may not repeat
    ps.seed([5, 7])                             # (5)->7
    lg = torch.zeros(3, V)
    ps.apply_chain(lg, [5, 7])                  # rows: anchor(7), then 5, then 7
    assert float(lg[0][7]) == 0.0, "row 0's tail is (7): nothing followed it before"
    assert float(lg[1][7]) == -ps.mask_penalty, "row 1's tail is (5): 7 was its completion"
    assert float(lg[2][7]) == 0.0, "row 2's tail is (7) again: nothing followed it"


def test_no_repeat_off_is_a_no_op():
    ps = state(rep=1.0)                          # no_repeat defaults to 0
    assert ps.spec.no_repeat == 0
    ps.seed([1, 2, 3])
    row = torch.zeros(V)
    ps.apply_single(row)
    assert float(row.min()) == 0.0, "nothing masked when the rule is off"
    assert ps.history == [], "no history is kept for an off rule"


def test_no_repeat_window_rebuilds():
    ps = state()
    ps.spec = PenaltySpec(no_repeat=2)
    ps.window = 8
    ps.seed(list(range(30)))                     # forces at least one eviction + rebuild
    assert len(ps.history) <= 8
    row = torch.zeros(V)
    # the indexed window is the tail; the last bigram is (28,29) and it was NOT followed
    ps._mask(row, [])
    assert float(row.min()) == 0.0


def test_no_repeat_seed_clears_the_previous_run():
    ps = state()
    ps.spec = PenaltySpec(no_repeat=3)
    ps.seed([1, 2, 3, 4])                        # first run indexes (2,3)->4
    ps.seed([9, 8, 7])                           # a new run must not carry it
    assert ps.history == [9, 8, 7]
    row = torch.zeros(V)
    ps._mask(row, [])
    assert float(row.min()) == 0.0, "the old run's n-grams must be gone"


def test_no_repeat_sees_repetitions_inside_the_block():
    # n=3 means a repeated bigram is masked. seed(5,6); draft 1,2,1,2: at row 4 the
    # bigram (1,2) was already written inside this block, so its completion 1 is masked.
    ps = state()
    ps.spec = PenaltySpec(no_repeat=3)
    ps.seed([5, 6])
    lg = torch.zeros(5, V)
    ps.apply_chain(lg, [1, 2, 1, 2])
    assert float(lg[2][1]) == 0.0, "at row 2 no earlier (1,2) exists"
    assert float(lg[4][1]) == -ps.mask_penalty, "row 4 sees the in-block (1,2) -> 1"


def test_no_repeat_breaks_a_period_two_cycle():
    # The retry evidence: with n=4 armed, a period-2 tail repeated 8 times still reached the
    # pattern stop. Simulate the greedy loop: decide the cycle's next token each step through the
    # mask, and require the mask to break it within a few iterations.
    ps = state()
    ps.spec = PenaltySpec(no_repeat=4)
    ps.seed([])
    a, b = 7, 9
    broke = False
    for step in range(24):
        row = torch.zeros(V)
        ps._mask(row, [])
        want = a if step % 2 == 0 else b
        if float(row[want]) < 0.0:
            broke = True
            break
        ps.commit([want])
    assert broke, "an exact a,b,a,b… cycle must be masked within a few steps"


def test_no_repeat_breaks_a_period_two_cycle_through_a_chain_block():
    ps = state()
    ps.spec = PenaltySpec(no_repeat=4)
    ps.seed([1, 2, 3])
    a, b = 7, 9
    draft = [a, b] * 6                      # 12 rows of the cycle inside one block
    lg = torch.zeros(len(draft) + 1, V)
    ps.apply_chain(lg, draft)
    blocked = [i for i in range(lg.shape[0])
               if float(lg[i][a if i % 2 == 0 else b]) < 0.0]
    assert blocked, "the mask must fire somewhere inside a 12-token period-2 block"


def test_no_repeat_mask_can_be_disabled_for_the_answer_phase():
    # Serving sets pen.mask = think.inside: the hard mask is a thinking-phase tool (the answer
    # keeps its drafts, boilerplate and repeated data). With mask off the rule must not touch a row.
    ps = state()
    ps.spec = PenaltySpec(no_repeat=2)
    ps.seed([5, 7])                              # (5) -> 7
    ps.mask = False
    row = torch.zeros(V)
    ps._mask(row, [5])
    assert float(row.min()) == 0.0, "masked-off rows must be untouched"
    ps.mask = True
    ps._mask(row, [5])
    assert float(row[7]) == -ps.mask_penalty, "masked-on still fires"


def test_the_mask_is_a_penalty_not_a_prohibition():
    # A token that is confidently preferred must still be reachable: the mask subtracts a finite
    # penalty and can never force an early EOS (chat efa916ed, 06:48). A tight loop pays it on
    # every step, which the stall detector closes properly.
    ps = state()
    ps.spec = PenaltySpec(no_repeat=2)
    ps.seed([5, 7])                              # (5) -> 7
    row = torch.zeros(V)
    row[7] = 10.0                                # the model really wants 7
    ps._mask(row, [5])
    assert float(row[7]) == 10.0 - ps.mask_penalty, "penalised, not forbidden"
    assert float(row[7]) > 0.0, "a confident token survives the mask"


def test_no_repeat_validates():
    try:
        PenaltySpec(no_repeat=1)
        raise AssertionError("n=1 must be rejected")
    except ValueError:
        pass


class _CountingDict(dict):
    def __init__(self):
        super().__init__()
        self.inserts = 0

    def setdefault(self, key, default=None):
        self.inserts += 1
        return super().setdefault(key, default)


def test_the_follower_index_is_built_over_the_window_only():
    """A 262k-token prompt was indexed in full and then cut to the last 1024 tokens.
    The work must be bounded by the window, whatever the prompt length -- and the index it leaves
    must be the one the retained window implies."""
    work = {}
    for n_prompt in (3_000, 60_000):
        ps = PenaltyState(PenaltySpec(no_repeat=3), V, "cpu")
        ps.followers = _CountingDict()
        ids = [(i * 7) % V for i in range(n_prompt)]
        ps.seed(ids)
        work[n_prompt] = ps.followers.inserts
        ref = PenaltyState(PenaltySpec(no_repeat=3), V, "cpu")
        ref.history = list(ps.history)
        ref._index_history()
        assert ps.history == ids[-ps.window // 2:]
        assert dict(ps.followers) == ref.followers
    assert work[3_000] == work[60_000] <= ps.window, work

# --- logit_bias rides on the same machinery ----------------------------------------------

def test_logit_bias_is_added_to_every_row_at_every_site():
    ps = PenaltyState(PenaltySpec(bias={3: 5.0, 7: -100.0}), V, "cpu")
    assert ps.spec.on and not ps.spec.penalizes
    ps.seed([1, 2, 3])
    assert ps.counts is None, "a bias alone reads no history and allocates no count vector"
    r = row()
    ref = r.clone()
    ps.apply_single(r)
    ref[3] += 5.0
    ref[7] -= 100.0
    assert torch.equal(r, ref)
    lg = torch.stack([row(), row() * 2, row() * 3])
    want = lg.clone()
    want[:, 3] += 5.0
    want[:, 7] -= 100.0
    ps.apply_chain(lg, [4, 5])
    assert torch.equal(lg, want), "every chain row gets the same addition"
    tree = DraftTree.from_sequences(0, [([5, 6], 0.9), ([8], 0.5)])
    lg = torch.stack([row() * (i + 1) for i in range(len(tree))])
    want = lg.clone()
    want[:, 3] += 5.0
    want[:, 7] -= 100.0
    ps.apply_tree(lg, tree)
    assert torch.equal(lg, want), "and every tree row"


def test_logit_bias_comes_before_the_penalties():
    # vLLM's order: the bias is a logits processor, the penalties read the processed row
    ps = PenaltyState(PenaltySpec(rep=2.0, bias={3: 4.0}), V, "cpu")
    ps.seed([3])
    r = row()
    base = float(r[3])
    ps.apply_single(r)
    assert abs(float(r[3]) - ((base + 4.0) / 2.0 if base + 4.0 > 0 else (base + 4.0) * 2.0)) < 1e-6


def test_logit_bias_joins_the_cache_key_only_when_present():
    assert PenaltySpec().key() == (1.0, 0.0, 0.0, 0), "a request without a bias keeps its key"
    assert PenaltySpec(bias={}).key() == PenaltySpec().key() and not PenaltySpec(bias={}).on
    assert PenaltySpec(bias={5: 0.0}).bias is None, "a zero bias is no bias"
    a, b = PenaltySpec(bias={5: 1.0}), PenaltySpec(bias={5: 2.0})
    assert a.key() != b.key() and a.key() != PenaltySpec().key()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:48s} ok")
            passed += 1
    print(f"{passed} passed")
