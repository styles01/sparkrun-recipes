"""SRV-425 / SPD-107: the host side of LFM2.5 in the server loop (engine/models/lfm2_moe/serving.py) and the
shared tree drafter with no draft head (engine/methods/mtp_drafter.py `NoHead`), no GPU.

  * with no head the round's tree is the lookup's copy of the request's own text alone, and a round
    with no match takes the plain step; the head's bookkeeping follows the rows synced;
  * `kv.length` reads the length after a pending chain commit, zero restarts the sequence (conv states
    zeroed, KV rows kept), another length keeps the rows by length; a forward not at the length is
    refused;
  * the cold verbs: `cost()` from the conv states and the KV bytes a token, the cache state's sections
    (the conv states, in layer order) and views (K and V, token axis 1), `trim_pending(n)` cuts a
    pending chain and refuses a cut past it, `rollback_to(keep)` commits the kept rows.
"""

from __future__ import annotations

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.ngram import NgramDrafter  # noqa: E402
from engine.methods.mtp_drafter import MTPTreeDrafter, NoHead  # noqa: E402
from engine.methods.staircut import PriceModel, StairCut  # noqa: E402
from engine.models.lfm2_moe.config import config_from_dict  # noqa: E402
from engine.models.lfm2_moe.ref import LfmState  # noqa: E402
from engine.models.lfm2_moe.serving import Lfm2MoeServing  # noqa: E402
from tests.test_lfm2_moe_ref import TINY  # noqa: E402


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"no ValueError (wanted {needles})")


class _Eng:
    """What the adapter reads of Lfm2MoeEngine, on the host: a state, a verify that records its block,
    a commit that moves the length."""

    device, dtype = torch.device("cpu"), torch.float32

    def __init__(self):
        self.cfg = config_from_dict(TINY)
        self.blocks, self.commits = [], []
        self._rb = {}
        self.W, self.experts = {}, {}

    def new_state(self, cap=None, paged=False):
        c = self.cfg
        st = LfmState(cap=cap or 64)
        for li in c.conv_layers:
            st.conv[li] = torch.ones(c.conv_state_len, c.hidden_size)
        for li in c.attention_layers:
            st.k[li] = torch.zeros(c.num_key_value_heads, st.cap, c.head_dim)
            st.v[li] = torch.zeros(c.num_key_value_heads, st.cap, c.head_dim)
        return st

    @staticmethod
    def _key(slot, M):
        return M if slot is None else (slot, M)

    def verify(self, toks, parents, st, graphs=None, slot=None):
        M = len(toks)
        self.blocks.append((list(toks), list(parents), st.length, slot))
        self._rb[self._key(slot, M)] = {"next": torch.arange(M), "logits": torch.zeros(M, 8), "hidden": torch.zeros(M, 4)}
        return list(range(M))

    def decode_multi(self, items, graphs=None):
        self.decoded = [(t, st.length, slot) for t, st, slot in items]
        return [{"next": torch.tensor([t + 1]), "logits": torch.full((1, 8), float(t)), "hidden": torch.zeros(1, 4)}
                for t, _, _ in items]

    def verify_multi(self, items):
        self.joint = [(list(t), list(p), st.length) for t, p, st in items]
        return [{"next": torch.arange(len(t)) + 100 * j, "logits": torch.full((len(t), 8), float(j)),
                 "hidden": torch.zeros(len(t), 4)} for j, (t, _, _) in enumerate(items)]

    def commit(self, st, path, graphs=None):
        self.commits.append(list(path))
        st.length += len(path)


def _rows(pos, n):
    return torch.arange(pos, pos + n, dtype=torch.float32)[:, None].repeat(1, 3)


def test_no_head_drafts_the_lookup_alone():
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0)
    ng = NgramDrafter(corpus_path="", min_order=3, max_depth=16)
    head = NoHead()
    d = MTPTreeDrafter(eng, head, StairCut(PriceModel(), siblings=False), ng)
    text = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 600, 601]
    prompt = text + [700] + text[:4]
    d.reset()
    d.prime(prompt)
    d.sync(prompt, _rows(0, len(prompt)), 0)
    tree = d.propose_tree(prompt, 32)
    tree.check()
    copy = text[4:] + [700] + text[:4]
    assert tree.tokens == [prompt[-1]] + copy and tree.parents == list(range(-1, len(copy))), tree.tokens
    assert head.st.length == len(prompt), head.st.length              # every synced row went in
    d.reset()
    d.prime([1, 2, 3, 4])
    d.sync([1, 2, 3, 4], _rows(0, 4), 0)
    tree = d.propose_tree([1, 2, 3, 4], 32)
    assert tree is None, tree.tokens                                   # no match: the plain step
    return "a copy of the request's text as the whole block; no match, no draft"


def test_no_head_rounds_without_a_match_are_plain_steps():
    """A NoHead round the lookup has nothing for is priced as a plain step, so a run of them does not
    teach StairCut that drafting loses: the copy that comes after them still drafts."""
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0)
    d = MTPTreeDrafter(eng, NoHead(), StairCut(PriceModel(), siblings=False),
                       NgramDrafter(corpus_path="", min_order=3, max_depth=16))
    d.reset()
    ctx = [1, 2, 3, 4]
    d.prime(ctx)
    d.sync(ctx, _rows(0, len(ctx)), 0)
    for t in range(10, 40):                          # thirty rounds with nothing to copy
        assert d.propose_tree(ctx, 32) is None
        d.on_plain(7.3)
        ctx = ctx + [t]
        d.observe([t])
        d.sync([t], _rows(len(ctx) - 1, 1), len(ctx) - 1)
    assert d.cut.spec_ms(0) is None, d.cut.spec_ms(0)   # nothing learned about drafted rounds
    assert d.stats["rounds"] == 0 and d.stats["plain"] == 30
    text = list(range(500, 512))
    ctx = ctx + text + [700] + text[:4]
    d.observe(text + [700] + text[:4])
    d.sync(text + [700] + text[:4], _rows(len(ctx) - 17, 17), len(ctx) - 17)
    tree = d.propose_tree(ctx, 32)
    assert tree is not None and tree.n_draft >= 8, tree and tree.tokens
    # a class whose drafted rounds were priced above the plain step (rounds that matched little):
    # with no head the round's own match still decides, so a copy drafts
    d.cut.spec[0], d.cut.plain[0] = 50.0, 7.3
    assert d.cut.skip(0)                                      # what a head's round would take
    d.reset()
    d.prime(ctx)
    d.sync(ctx, _rows(0, len(ctx)), 0)
    again = d.propose_tree(ctx, 32)
    assert again is not None and again.n_draft >= 8, again and again.tokens
    return (f"30 rounds with no match priced as plain steps; then a copy of {tree.n_draft} tokens, and "
            f"{again.n_draft} with drafting priced above the plain step")


def test_kv_length_and_restart():
    s = Lfm2MoeServing(_Eng(), 64)
    s.st.length = 10
    s.kv.length = 7
    assert s.st.length == 7 and s.kv.max_len == 64
    s.st.k[2][:, 3] = 5.0
    s.kv.length = 0
    assert s.st.length == 0 and all(float(t.abs().sum()) == 0 for t in s.st.conv.values())
    assert float(s.st.k[2][:, 3].sum()) != 0                         # KV rows stay
    _raises(lambda: setattr(s.kv, "length", 65), "capacity 64")
    _raises(lambda: s.forward_tree(torch.tensor([1, 2]), [-1, 0], start=3), "forward at 3", "holds 0 rows")


def test_no_head_long_matches_calibrate_apart():
    """With no draft head a copy (a match at the longest order) has a higher confidence than a passing
    phrase (the shortest), so misses of short matches leave the copy's calibration as it was."""
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0)
    d = MTPTreeDrafter(eng, NoHead(), StairCut(PriceModel(), siblings=False),
                       NgramDrafter(corpus_path="", min_order=3, max_depth=16))
    text = list(range(500, 520))
    d.reset()
    d.prime(text + [700] + text[:10])
    _, long_conf = d._look(text + [700] + text[:10])
    d.reset()
    d.prime([1, 2, 3, 9, 1, 2, 3])
    _, short_conf = d._look([1, 2, 3, 9, 1, 2, 3])
    assert long_conf > 0.8 and short_conf < 0.7 and int(long_conf * 10) != int(short_conf * 10), (long_conf, short_conf)
    for _ in range(8):
        d.cut.cal.learn("look", short_conf, False)
    assert d.cut.cal.q("look", long_conf) == long_conf          # the copy's bin learned nothing
    return f"conf {long_conf:.2f} (a copy) against {short_conf:.2f} (a short match)"


def test_reading_the_length_commits_nothing():
    """kv.length while a joint plain step waits for its slot's commit_tree (another thread's read, the
    health report): the length it will hold, and the commit still counts the row once."""
    s = Lfm2MoeServing(_Eng(), 64)
    s.st.length = 10
    s._plain_pending = True
    assert s.kv.length == 11 and s._plain_pending and s.st.length == 10
    s.commit_tree([0])
    assert s.kv.length == 11 and s.st.length == 11 and not s._plain_pending


def test_cost_and_cache_state():
    s = Lfm2MoeServing(_Eng(), 64)
    c = s.cfg
    cost = s.cost()
    assert cost.view_bytes_per_token == len(c.attention_layers) * 2 * c.kv_dim * 2
    assert cost.section_bytes == len(c.conv_layers) * c.conv_state_len * c.hidden_size * 4
    secs = s.cache_state().sections()
    assert list(secs) == [f"conv.{li}" for li in c.conv_layers]
    assert all(secs[f"conv.{li}"] is s.st.conv[li] for li in c.conv_layers)   # the live tensors
    views = s.cache_state().views()
    assert [(n, ax) for n, _, ax in views] == [(f"{kv}.{li}", 1) for li in c.attention_layers for kv in "kv"]
    snap = s.cache_state().clone()
    for t in s.st.conv.values():
        t.zero_()
    s.cache_state().restore(snap)
    assert all(float(t.min()) == 1.0 for t in s.st.conv.values())


def test_chain_block_pending_and_trim():
    e = _Eng()
    s = Lfm2MoeServing(e, 64)
    s.forward_block(torch.tensor([7, 8, 9, 10]), start=0)
    assert s.kv.length == 4 and e.commits == []                       # a read commits nothing
    s.forward_block(torch.tensor([1, 2, 3]), start=4)
    assert e.commits == [[0, 1, 2, 3]]                                # the next forward lands it
    assert not s.trim_pending(9) and s.trim_pending(6) and e.commits[-1] == [0, 1]
    assert s.st.length == 6
    s.forward_block(torch.tensor([1, 2, 3]), start=6)
    s.rollback_to(1)
    assert e.commits[-1] == [0] and s.st.length == 7
    assert s.trim_pending(7) and not s.trim_pending(8)


def test_side_slots_and_a_joint_pass():
    """A side slot shares the engine (the one copy of the weights) and has its own state and row
    buffers; a joint pass leaves each slot its picks and logits and commits nothing."""
    e = _Eng()
    s0 = Lfm2MoeServing(e, 64)
    s1, d1 = s0.side_slot(32)
    assert s1.fn is e and s1.st is not s0.st and s1.slot is not None and d1 is None and s1.max_len == 32
    s1.forward_tree(torch.tensor([1, 2]), [-1, 0], start=0)
    assert e.blocks[-1][3] == s1.slot and (s1.slot, 2) in e._rb
    s0.st.length, s1.st.length = 5, 3
    out = Lfm2MoeServing.forward_tree_multi([(s0, torch.tensor([9, 8, 7]), [-1, 0, 1], 5), (s1, torch.tensor([4, 5]), [-1, 0], 3)])
    assert e.joint == [([9, 8, 7], [-1, 0, 1], 5), ([4, 5], [-1, 0], 3)]
    assert s0.picks.tolist() == [0, 1, 2] and s1.picks.tolist() == [100, 101]
    assert float(out[1][0, 0]) == 1.0 and s0.st.length == 5 and e.commits == []
    assert s0.tree_batchable(list(range(32))) and not s0.tree_batchable(list(range(33)))
    per_token, const = s0.side_slot_bytes(32, paged=True)
    assert per_token == 0 and const > s0.cost().section_bytes


def test_a_joint_plain_step():
    """Every slot with a one-row block: one joint decode step; each slot's commit_tree([0]) moves its
    length (the pass advanced the state), and a pending step is landed before anything reads the length."""
    e = _Eng()
    s0 = Lfm2MoeServing(e, 64)
    s1, _ = s0.side_slot(32)
    s0.st.length, s1.st.length = 5, 3
    out = Lfm2MoeServing.forward_tree_multi([(s0, torch.tensor([9]), [-1], 5), (s1, torch.tensor([4]), [-1], 3)])
    assert e.decoded == [(9, 5, None), (4, 3, s1.slot)] and float(out[1][0, 0]) == 4.0
    assert s0.picks.tolist() == [10] and s1.picks.tolist() == [5]
    s0.commit_tree([0])
    assert s0.st.length == 6 and e.commits == []
    assert s1.kv.length == 4                                    # landed on read
    _raises(lambda: (setattr(s0, "_plain_pending", True), s0.commit_tree([0, 1])), "keeps its one row")


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
