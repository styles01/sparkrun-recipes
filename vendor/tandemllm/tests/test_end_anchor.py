"""The resident prefix's end anchor, on the random 4-layer model.

THE MEASUREMENT. rrr2 (2026-10-02, opencode): turn 1 prefilled 34,038 tokens and thought 32,773;
turn 2's 66,964-token prompt resumed at 33,792, the previous prompt's last 1,024 line, and re-read
33,172 tokens in about 44 s. Anchors come from prefill chunks only, and the session snapshot that
would hold the state after the generation (about 7 GB) is over the store's entry cap and the whole
store (tests/test_reread_after_think.py).

THE FIX. At the end of every generation the resident prefix keeps one more anchor: the recurrent
state and convolution tails at the last position the engine forwarded, with the rows in place valid
up to it (engine/cache.py `ResidentPrefix.keep_end`, server/app.py `_remember`). A prompt that
starts with all of those tokens resumes there and forwards only its new messages. Verify blocks
wrote that state, so the resume is held to the session cache's gate -- the same greedy tokens --
not to bit equality with a cold prefill. The tokens the session monitor hid in the answer are put
back where they sit (`replay`), as the router replays its notes.

  1  the served loop (`generate_stream`, `_remember`) with no drafter, a chain and a tree, drafts
     right and wrong: after a long generation the next turn resumes at the end, forwards only its
     new tokens, and writes the same greedy tokens as the same turn without the anchor (the gate)
  2  a generation that stopped inside an accepted path: with the commit still pending (the served
     QWEN38_COMMIT_IN_VERIFY) it is cut back and kept, bit for bit the shorter commit's state;
     without one it is declined, and the next turn resumes at the prompt's line
  3  hidden tokens: a hidden close's span (`_force_close`) and SP4's note (`_inserted`); the
     client's copy without them resumes at the end only through the replay, the replay carries
     over to the turn after, and the in-context stage's prompt gets it as well; the copy the
     template re-renders: whitespace next to the phrase aligned to the rows' own
     (`glue`), anything else served as sent
  4  a copy that parts inside the generation resumes at the prompt's last line, bit for bit as a
     cold prefill, and the end anchor goes
  5  the session-grade line (`taint`): after an end resume the anchors above it are a session's,
     the store takes no prefix snapshot and the disk writes nothing; an anchor at or below it, or
     a cold prefill, is a cold grid prefill's again, bit for bit
  6  a guest between the turns keeps the end anchor; a guest's own generation keeps none
  7  memory: one buffer pair, asked of the `Budget` at level `end`, given back after the body
     anchors and before the stash; a safe shed keeps a held anchor
  8  the pin and the park carry their rows' line; the end anchor goes to the park with its rows
  9  off: nothing is kept, and every prefill restores what it restored without the anchor
  10 the length router's mark: the arm the latch released is inherited, as a session snapshot's
  11 through the handler: the replay, `usage.prompt_tokens` the client's, and the request after
     it, whose copy lacks the same tokens

Run: python tests/test_end_anchor.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402  (sets the CPU environment before the engine imports)

import torch  # noqa: E402

import engine.model as M  # noqa: E402
from engine import cache, disk_cache  # noqa: E402
from engine.penalty import PenaltySpec  # noqa: E402
from engine.spec import Relax  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from server import app  # noqa: E402
from server.conductor.monitor.policy import CLOSE_REASONING, Action  # noqa: E402
from test_app_loop import FakeTok, Req  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402
from test_resident import same_state  # noqa: E402
from test_window_edge import FixedDrafter  # noqa: E402

CH = 16
BIG = 1 << 30
TOOL = tokens(25, seed=44)                       # turn 2's new message
TOOL2 = tokens(19, seed=45)


def _state(end=True, drafter=None, tree=False, k=4, store=None, disk=None, mem=None, **res_kw):
    """The server's state around `generate_stream` on the random model: the prefix cache's grid
    and the resident prefix, with its end anchor on or off."""
    res_kw.setdefault("stash_bytes", BIG)
    res = cache.ResidentPrefix(res_kw.pop("budget", BIG), CH, end=end, mem=mem, **res_kw)
    app.STATE.clear()
    app.STATE.update(engine=fresh(), drafter=drafter, k=k, tree=tree, sampled_tree=False,
                     relax=Relax(1.0, 1), verbose=False, state_store=store, prefix_cache=True,
                     prefix_chunk=CH, session_cache=store is not None, tok=FakeTok(),
                     device="cpu", max_len=512, default_max_tokens=32, pen_spec=PenaltySpec(),
                     model="t", cfg_eos=None, think_budget=0, think_stall=False,
                     reasoning_format="tags", request_timeout=0.0, max_queue=8,
                     queue_timeout=5.0, pattern_stop=None, response_cache=None,
                     suffix_store=None, resident=res, disk_cache=disk, mem=mem)
    return res


def turn(ids, n, session="ses", hidden=()):
    """One request as the handler runs it: the stream, then `_remember`."""
    out = list(app.generate_stream(torch.tensor(ids), n, set(), None, session))
    pre = dict(app.STATE["last_prefill"])
    ctx = list(app.STATE["last_ctx"])
    app._remember(list(ids), out, session, hidden=hidden)
    return out, pre, ctx


class Oracle:
    """Drafts what a plain greedy run wrote, so every block is accepted whole and a generation can
    stop part way through one."""

    def __init__(self, seq, width=4):
        self.seq, self.width = list(seq), width

    def reset(self):
        pass

    def observe(self, tokens):
        pass

    def propose(self, ctx, count):
        return self.seq[len(ctx):len(ctx) + self.width]

    def propose_tree(self, ctx, count):
        toks = self.seq[len(ctx):len(ctx) + self.width]
        return DraftTree.chain(ctx[-1], toks) if toks else None


# ------------------------------------------------------------------ 1. the served loop

def _two_turns(end, drafter=None, tree=False, n1=150):
    res = _state(end=end, drafter=drafter, tree=tree)
    prompt = tokens(100, seed=60)
    out1, pre1, ctx1 = turn(prompt, n1)
    e1 = res.end["at"] if res.end is not None else None
    # the client's copy: the prompt, every token it received, then the tool result
    out2, pre2, ctx2 = turn(prompt + out1 + TOOL, 24)
    return e1, out1, out2, pre2, ctx1


def test_the_next_turn_resumes_at_the_end_and_writes_the_same_tokens():
    """After a 150-token generation the next turn forwards its tool result and the token the
    generation decided last (never forwarded), not the 150 tokens again; its greedy tokens are
    the ones the same turn writes from the prompt's line without the anchor."""
    plain = _two_turns(end=False)[1]
    seq = tokens(100, seed=60) + plain + tokens(40, seed=9)
    cases = [("none", None, False), ("chain wrong", FixedDrafter(97, 5, tree_mode=False), False),
             ("tree wrong", FixedDrafter(97, 5, tree_mode=True), True),
             ("chain right", Oracle(seq), False), ("tree right", Oracle(seq), True)]
    got = []
    for label, dr, tree in cases:
        e, out1, out2, pre2, ctx1 = _two_turns(end=True, drafter=dr, tree=tree)
        res = app.STATE["resident"]
        assert out1 == plain, label
        n2 = 100 + len(out1) + len(TOOL)
        assert pre2["kind"] == "end" and pre2["reused"] == e, (label, pre2)
        # the last decided token is forwarded by the next turn, or by its block when the stream
        # ended on a block's last accepted token
        assert e in (len(ctx1) - 1, len(ctx1)) and pre2["forwarded"] == n2 - e, (label, e, pre2)
        _, _, off2, pre_off, _ = _two_turns(end=False, drafter=dr, tree=tree)
        assert pre_off["kind"] == "resident" and pre_off["reused"] == 96, (label, pre_off)
        assert out2 == off2 and len(out2) == 24, (label, out2, off2)
        assert res.stats["end_hits"] == 1 and res.stats["end_saved"] == e - 96, res.stats
        got.append(f"{label} {e}")
    return f"resumed at the end ({', '.join(got)} of 275) instead of 96; 24 greedy tokens equal"


def test_the_end_follows_the_conversation_turn_after_turn():
    """Each turn keeps its own end: the third turn resumes at the second's."""
    res = _state()
    prompt = tokens(100, seed=60)
    out1, _, _ = turn(prompt, 60)
    out2, pre2, ctx2 = turn(prompt + out1 + TOOL, 40)
    _, pre3, _ = turn(prompt + out1 + TOOL + out2 + TOOL2, 8)
    assert pre2["kind"] == pre3["kind"] == "end", (pre2, pre3)
    assert pre3["reused"] == len(ctx2) - 1 == res.stats["end_tokens_reused"] - pre2["reused"]


# ------------------------------------------------------------------ 2. inside an accepted path

class folded:
    """The served commit, folded into the next verify (tests/test_session_trim.py)."""

    def __enter__(self):
        self.old = (M.FUSED_COMMIT, M.COMMIT_IN_VERIFY)
        M.FUSED_COMMIT, M.COMMIT_IN_VERIFY = True, True

    def __exit__(self, *exc):
        M.FUSED_COMMIT, M.COMMIT_IN_VERIFY = self.old
        return False


def _block(res, path, prompt):
    """A prompt prefilled through the resident prefix, then one verified tree block committed
    along `path` (the first generated token is its anchor)."""
    eng = fresh()
    tree = T.BRANCHY
    with torch.no_grad():
        cache.prefill(eng, None, prompt, DEV, chunk=CH, resident=res)
        eng.forward_tree(torch.tensor(tree.tokens), tree.parents, start=len(prompt))
        eng.commit_tree(path)
    return eng, [tree.tokens[i] for i in path]


def test_a_generation_stopped_inside_a_pending_block_is_cut_back_and_kept():
    """The loop yields a block one token at a time and returns at the end token, a stop or its
    budget: `kv.length` is then ahead of the committed tokens. With the commit pending, the end
    anchor cuts it back (`trim_pending`) and holds exactly the shorter commit's state."""
    tree = T.BRANCHY
    prompt = tokens(40, seed=61)
    n = 0
    with folded():
        for leaf in tree.leaves():
            path = tree.path(leaf)
            for m in range(1, len(path)):
                res = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True)
                eng, toks = _block(res, path, prompt)
                assert eng._pend is not None and eng.kv.length == len(prompt) + len(path)
                why = res.keep_end(eng, None, prompt + toks[:m], len(prompt))
                assert why == "kept" and res.end["at"] == len(prompt) + m, why
                ref, _ = _block(cache.ResidentPrefix(BIG, CH), path[:m], prompt)
                ref._settle()
                S, conv = res._end_bufs
                assert torch.equal(S, ref.state.S) and torch.equal(conv, ref.state.conv), (leaf, m)
                n += 1
    return f"{n} (path, cut) pairs: the end anchor is the shorter commit's state"


def _remember_ahead(store, m: int = 1, conv="ses"):
    """`_remember` as the handler runs it after a generation that stopped `m` tokens into a pending
    block, with the session cache on (the served flags) and the resident prefix's end anchor."""
    tree = T.BRANCHY
    path = tree.path(tree.leaves()[0])
    prompt = tokens(40, seed=61)
    res = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True)
    eng, toks = _block(res, path, prompt)
    assert eng._pend is not None and eng.kv.length > len(prompt) + m
    other = cache.StateStore(BIG, chunk=CH)
    app.STATE.clear()
    app.STATE.update(engine=eng, drafter=None, state_store=store, session_cache=True,
                     verbose=True, last_ctx=prompt + toks[:m], last_prompt_n=len(prompt),
                     session_trim=False, suffix_store=None, resident=res, mem=None)
    out = io.StringIO()
    from contextlib import redirect_stdout
    with redirect_stdout(out):
        app._remember(list(prompt), list(toks[:m]), conv)
    del other
    return res, eng, out.getvalue(), prompt, toks


def test_srv253_a_session_put_the_engine_is_ahead_of_leaves_the_end_anchor_kept():
    """SRV-253 (5 of 45 non-streamed turns): the session put came first, and for a
    conversation under the store's per-entry cap it cloned a snapshot -- `capture` settles the
    block's pending commit -- and then declined it as ahead of its key. The end anchor after it
    found the commit absorbed and was not kept. The put now declines before anything is cloned."""
    with folded():
        store = cache.StateStore(BIG, chunk=CH)         # the snapshot fits: the served < 7.4k case
        res, eng, log, prompt, toks = _remember_ahead(store)
    assert store.stats["puts"] == 0 and store.stats.get("declined_short") == 1, store.stats
    assert "ahead of the committed tokens" in log, log
    assert "end anchor not kept" not in log, log
    assert res.end is not None and res.end["at"] == len(prompt) + 1, res.end
    assert res.stats["end_kept"] == 1 and res.stats["end_ahead"] == 0, res.stats
    # the next turn resumes at the end, as a streamed one over the cap always did
    nxt = prompt + toks[:1] + TOOL
    info = {}
    with folded():
        cache.prefill(eng, None, nxt, DEV, chunk=CH, resident=res, info=info)
    assert info["kind"] == "end" and info["start"] == len(prompt) + 1, info
    return "declined before the clone; end anchor kept; the next turn resumes at the end"


def test_srv253_the_doomed_put_evicts_nothing_from_the_store():
    """The decline used to make room first (`StateStore.reserve`): other conversations' entries
    were evicted for a snapshot that was then thrown away."""
    with folded():
        eng0, _ = _block(cache.ResidentPrefix(BIG, CH), T.BRANCHY.path(T.BRANCHY.leaves()[1]),
                         tokens(40, seed=70))
        eng0._settle()
        snap = cache.capture(eng0)
        store = cache.StateStore(snap.nbytes + snap.nbytes // 2, chunk=CH)
        store.put(list(range(1, eng0.kv.length + 1)), snap, "other", kind="session")
        assert store.stats["puts"] == 1
        _remember_ahead(store)
    assert store.stats["evictions"] == 0 and len(store._d) == 1, store.stats
    return "the other conversation's entry stays"


def test_without_a_pending_commit_an_engine_ahead_is_declined():
    tree = T.BRANCHY
    path = tree.path(tree.leaves()[0])
    prompt = tokens(40, seed=61)
    res = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True)
    eng, toks = _block(res, path, prompt)              # the flags off: committed at once
    assert res.keep_end(eng, None, prompt + toks[:1], len(prompt)) == "ahead"
    assert res.end is None and res.stats["end_ahead"] == 1
    # the next turn resumes at the prompt's line, as without the anchor
    nxt = prompt + toks[:1] + TOOL
    info = {}
    cache.prefill(eng, None, nxt, DEV, chunk=CH, resident=res, info=info)
    assert info["kind"] == "resident" and info["start"] == 32, info


# ------------------------------------------------------------------ 3. hidden tokens

class Sup:
    """What `_force_close` touches of a supervisor."""

    def __init__(self):
        self.forwarding, self.forwarded_at, self.resume_ms = None, None, []


PHRASE = tokens(6, seed=41)                      # the vendor's close, in the KV only
CLOSER = tokens(2, seed=42)                      # `</think>` and the blank line, streamed


def _hidden_turn(prompt, before=30, after=20, session="ses", hidden_prompt=(), phrase=None):
    """A generation with a hidden close in it, run through `_force_close` as the loop runs it:
    `before` tokens decoded, the phrase hidden and the closer shown, `after` more; then
    `_remember`. Returns what the client received."""
    eng = app.STATE["engine"]
    app.STATE["hidden_spans"], app.STATE["last_prompt_n"] = [], len(prompt)
    with torch.no_grad():
        lg, _, _ = cache.prefill(eng, None, prompt, DEV, chunk=CH, conv_id=session,
                                 resident=app.STATE["resident"])
        ctx = list(prompt) + [int(lg[0, -1].argmax())]
        app.STATE["last_ctx"] = ctx
        out = [ctx[-1]]

        def step(k):
            for _ in range(k):
                lg = eng.forward(torch.tensor([ctx[-1]]), start=len(ctx) - 1, last_only=True)
                ctx.append(int(lg[0, -1].argmax()))
                out.append(ctx[-1])
        step(before)
        act = Action(CLOSE_REASONING, reason="budget(30)")
        act.ids, act.visible = list(phrase or PHRASE) + CLOSER, list(CLOSER)
        out += list(app._force_close(eng, None, None, ctx, len(ctx) - 1, DEV, act=act,
                                     sup=Sup()))
        step(after)
    app._remember(list(prompt), out, session, hidden=hidden_prompt)
    return out, ctx


def test_the_copy_without_the_hidden_phrase_resumes_at_the_end_through_the_replay():
    res = _state()
    prompt = tokens(100, seed=62)
    out, ctx = _hidden_turn(prompt)
    q = len(prompt) + 1 + 30                          # where the phrase sits in the context
    assert app.STATE["hidden_spans"] == [(q, PHRASE)]
    assert res.end["hidden"] == [(q, PHRASE)] and res.end["at"] == len(ctx) - 1
    copy = prompt + out + TOOL                        # what the client sends: no phrase
    assert copy != ctx[:len(copy)]
    ids, spans = res.replay(copy)
    assert spans == [(q, PHRASE)] and ids == ctx + TOOL, "the phrase goes back where it sat"
    _, pre, _ = turn(ids, 8, hidden=spans)
    assert pre["kind"] == "end" and pre["reused"] == len(ctx) - 1, pre
    # without the replay the same copy parts at the phrase: the prompt's line, the end dropped
    res = _state()
    out, ctx = _hidden_turn(prompt)
    _, pre, _ = turn(copy, 8)
    assert pre["kind"] == "resident" and pre["reused"] == 96 and res.end["at"] != q, pre


GLUE = 95                                        # `\n` in these tests: whitespace only


def test_the_templates_newline_after_the_phrase_is_aligned():
    """0.3.14's live check (2026-10-02 22:07): the hidden close's phrase ends in its newline, and
    the client's copy, which the template re-renders as `reasoning|trim + "\n</think>"`, holds
    one newline more right where the phrase sat. With the tokenizer's whitespace tokens known
    (`glue`, the server's `glue_ids`) the copy resumes at the end anchor and is served with the
    rows' own tokens; without them it parts there and resumes at the prompt's line (0.3.14)."""
    phrase = PHRASE[:-1] + [GLUE]
    prompt = tokens(100, seed=62)
    for glue, want in ((frozenset({GLUE}), "end"), (frozenset(), "resident")):
        res = _state()
        res.glue = glue
        out, ctx = _hidden_turn(prompt, phrase=phrase)
        k = 1 + 30                                    # the visible tokens before the phrase
        copy = prompt + out[:k] + [GLUE] + out[k:] + TOOL
        ids, spans = res.replay(copy)
        if want == "end":
            assert ids == ctx + TOOL and spans == [(len(prompt) + k, phrase)], spans
            assert res.stats["end_replayed"] == len(phrase) - 1
        else:
            assert ids == copy and spans == []
        _, pre, _ = turn(ids, 8, hidden=spans)
        assert pre["kind"] == want, pre
        if want == "end":
            assert pre["reused"] == len(ctx) - 1, pre
        else:
            assert pre["reused"] == 96 and res.end["at"] != len(prompt) + k, pre


def test_the_rows_whitespace_the_copy_trimmed_before_the_phrase_is_aligned():
    """The visible reasoning ended in whitespace right before the phrase; the template trims it
    and writes its own newline: the copy has another whitespace token there, and still resumes
    at the end with the rows' tokens."""
    res = _state()
    prompt = tokens(100, seed=62)
    out, ctx = _hidden_turn(prompt)
    k = 1 + 30
    res.glue = frozenset({GLUE, out[k - 1]})          # the token before the phrase: whitespace
    copy = prompt + out[:k - 1] + [GLUE] + out[k:] + TOOL
    ids, spans = res.replay(copy)
    assert ids == ctx + TOOL and spans == [(len(prompt) + k, PHRASE)], spans
    _, pre, _ = turn(ids, 8, hidden=spans)
    assert pre["kind"] == "end" and pre["reused"] == len(ctx) - 1, pre


def test_anything_but_whitespace_next_to_the_phrase_is_the_clients():
    """Only whitespace is aligned: a copy with another token next to where the phrase sat (the
    client edited the answer) is served as it was sent and parts there."""
    res = _state()
    res.glue = frozenset({GLUE})
    prompt = tokens(100, seed=62)
    out, ctx = _hidden_turn(prompt)
    k = 1 + 30
    other = next(t for t in range(1, 95) if t != out[k] and t != GLUE)
    for copy in (prompt + out[:k] + [other] + out[k:] + TOOL,
                 prompt + out[:k - 1] + [other] + out[k:] + TOOL,
                 prompt + out[:k] + [GLUE, other] + out[k:] + TOOL):
        assert res.replay(copy) == (copy, []), copy[len(prompt) + k - 2:len(prompt) + k + 3]
    # the copy must go past the end anchor: up to it alone is not a prompt to resume
    assert res.replay(prompt + out[:k] + [GLUE] + out[k:-1])[1] == []
    assert res.replay(prompt + out[:k] + [GLUE] + out[k:])[1] == [(len(prompt) + k, PHRASE)]


def test_the_servers_glue_is_the_tokenizers_single_whitespace_tokens():
    """`glue_ids`: of `GLUE_TEXTS`, the ones the tokenizer writes as one token (the FakeTok writes
    one id a character: newline, space and tab)."""
    assert app.glue_ids(FakeTok()) == frozenset({1 + ord("\n") % 50, 1 + ord(" ") % 50,
                                                 1 + ord("\t") % 50})


def test_the_replay_carries_over_to_the_turn_after():
    """Turn 2 resumed with the phrase put back: its end anchor's rows hold the phrase too, so
    turn 3's copy (still without it) is replayed again, and its own hidden spans with it."""
    res = _state()
    prompt = tokens(100, seed=62)
    out1, ctx1 = _hidden_turn(prompt)
    ids2, spans2 = res.replay(prompt + out1 + TOOL)
    out2, pre2, ctx2 = turn(ids2, 30, hidden=spans2)
    assert pre2["kind"] == "end"
    ids3, spans3 = res.replay(prompt + out1 + TOOL + out2 + TOOL2)
    assert spans3 == spans2 and ids3 == ctx2 + TOOL2
    _, pre3, _ = turn(ids3, 4, hidden=spans3)
    assert pre3["kind"] == "end" and pre3["reused"] == len(ctx2) - 1, pre3


def test_sp4s_note_is_a_span_of_its_own():
    """SP4's hidden note at the prompt's tail: one run inserted, found by `_inserted`."""
    a = [1, 2, 3, 4, 5]
    assert app._inserted(a, [1, 2, 9, 9, 3, 4, 5]) == (2, [9, 9])
    assert app._inserted(a, a + [7]) == (5, [7])
    assert app._inserted(a, [9] + a) == (0, [9])
    assert app._inserted(a, [1, 9, 3, 9, 4, 5]) is None          # not one run
    assert app._inserted(a, a) is None
    res = _state()
    prompt = tokens(100, seed=63)
    note = tokens(5, seed=64)
    served = prompt[:90] + note + prompt[90:]
    out, ctx = _hidden_turn(served, hidden_prompt=[(90, note)])
    assert res.end["hidden"] == [(90, note), (len(served) + 31, PHRASE)]
    ids, spans = res.replay(prompt + out + TOOL)
    assert ids == ctx + TOOL and len(spans) == 2


def test_the_in_context_stage_resumes_at_the_end_with_the_replay():
    """The stage prefills the prompt up to its last message plus a question before the turn:
    its prompt gets the phrase back too, so the stage starts at the end anchor, and the end
    anchor is still there for the turn."""
    res = _state()
    prompt = tokens(100, seed=62)
    out, ctx = _hidden_turn(prompt)
    stage = prompt + out + TOOL + tokens(7, seed=66)  # the last message, then the question
    at = len(prompt + out + TOOL)
    picked, info = app.context_choice(stage, [3, 4], at, "ses")
    assert info["reused"] == len(ctx) - 1 and len(picked) == 2, info
    assert res.end is not None and res.end["at"] == len(ctx) - 1
    ids, spans = res.replay(prompt + out + TOOL)
    _, pre, _ = turn(ids, 4, hidden=spans)
    assert pre["kind"] in ("end", "resident") and pre["reused"] >= len(ctx) - 1, pre


# ------------------------------------------------------------------ 4. a copy that parts inside

def test_a_copy_that_parts_inside_the_generation_resumes_at_the_prompts_line_bit_for_bit():
    res = _state()
    prompt = tokens(100, seed=60)
    out1, _, _ = turn(prompt, 80)
    copy = prompt + out1[:40] + [(out1[40] % 90) + 1] + out1[41:] + TOOL   # one token re-encoded
    eng = app.STATE["engine"]
    info = {}
    lg, reused, _ = cache.prefill(eng, None, copy, DEV, chunk=CH, resident=res, info=info)
    assert info["kind"] == "resident" and reused == 96 and res.end is None, info
    ref = fresh()
    lr, _, _ = cache.prefill(ref, None, copy, DEV, chunk=CH)
    assert torch.equal(lg, lr)
    same_state(eng, ref, len(copy))
    assert res.taint is None                          # a cold grid's rows again


# ------------------------------------------------------------------ 5. the session-grade line

def test_after_an_end_resume_the_rows_are_a_sessions_and_a_cold_grid_clears_it():
    store = cache.StateStore(BIG, chunk=CH)
    d = tempfile.mkdtemp(prefix="qse-end-disk-")
    try:
        t = disk_cache.DiskTier(d, chunk=CH, fingerprint="t", min_tokens=CH, head_anchors=2,
                                free_bytes=lambda p: 10 ** 15, settings={"admit_after": 1}).open()
        res = _state(store=store, disk=t)
        app.STATE["session_cache"] = False            # the prefix snapshots alone
        prompt = tokens(100, seed=60)
        out1, _, ctx1 = turn(prompt, 60)
        assert t.flush()
        written = t.report()["entries"]
        assert res.taint == 100 and res.end["at"] == len(ctx1) - 1
        puts = store.stats["puts"]
        nxt = prompt + out1 + TOOL + tokens(40, seed=67)
        _, pre, _ = turn(nxt, 4)
        assert pre["kind"] == "end"
        assert t.flush() and t.report()["entries"] == written, "nothing of it on the disk"
        assert store.stats["puts"] == puts > 0, "no prefix snapshot of a session's rows"
        assert res.taint == 100 and max(res.anchors) > len(ctx1) - 1, sorted(res.anchors)
        # an anchor at the prompt's line is a cold grid's: what follows it is again, bit for bit
        eng = app.STATE["engine"]
        other = prompt + tokens(30, seed=68)
        lg, reused, _ = cache.prefill(eng, None, other, DEV, chunk=CH, resident=res)
        assert reused == 96 and res.taint is None and res.end is None
        ref = fresh()
        lr, _, _ = cache.prefill(ref, None, other, DEV, chunk=CH)
        assert torch.equal(lg, lr)
        same_state(eng, ref, len(other))
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_an_anchor_above_the_line_is_a_resume_under_the_gate():
    """A divergence after the end, inside turn 2's prompt: it resumes at an anchor that prefill
    took over the end anchor's state, and the line stays."""
    res = _state()
    prompt = tokens(100, seed=60)
    out1, _, ctx1 = turn(prompt, 60)
    nxt = prompt + out1 + TOOL + tokens(40, seed=67)
    turn(nxt, 4)
    b = max(x for x in res.anchors if x < len(nxt))
    assert b > len(ctx1)
    info = {}
    cache.prefill(app.STATE["engine"], None, nxt[:b + 3] + [5, 6, 7], DEV, chunk=CH,
                  resident=res, info=info)
    assert info["kind"] == "resident" and info["start"] == b and res.taint == 100, info


# ------------------------------------------------------------------ 6. guests

def test_a_guest_between_the_turns_keeps_the_end_and_keeps_none_of_its_own():
    res = _state()
    prompt = tokens(120, seed=60)
    out1, _, ctx1 = turn(prompt, 60)
    e = res.end["at"]
    title = tokens(20, seed=69)                      # opencode's title call: a guest
    _, pre_g, _ = turn(title, 8, session="side")
    assert res.stats["guests"] == 1 and res.end["at"] == e, res.report()
    _, pre, _ = turn(prompt + out1 + TOOL, 8)
    assert pre["kind"] == "end" and pre["reused"] == e, pre


# ------------------------------------------------------------------ 7. memory

class FakeAlloc:
    def __init__(self, alloc):
        self.alloc = alloc

    def __call__(self):
        return self.alloc, self.alloc


def test_the_buffers_are_asked_for_once_and_given_back_in_their_order():
    assert cache.Budget.LEVELS == ("store", "stash_idle", "anchors", "end", "stash", "park", "pin",
                                   "roles")
    st = fresh().state
    nbytes = st.S.numel() * st.S.element_size() + st.conv.numel() * st.conv.element_size()
    full = cache.Budget(10 * BIG, 12 * BIG, 0, probe=FakeAlloc(10 * BIG), empty=lambda: None)
    res = _state(mem=full)
    turn(tokens(100, seed=60), 20)
    assert res.end is None and res.stats["end_refused"] == 1, res.stats
    room = cache.Budget(10 * BIG, 12 * BIG, 0, probe=FakeAlloc(0), empty=lambda: None)
    res = _state(mem=room)
    prompt = tokens(100, seed=60)
    out1, _, _ = turn(prompt, 20)
    out2, pre2, _ = turn(prompt + out1 + TOOL, 20)
    assert pre2["kind"] == "end" and res.stats["end_kept"] == 2 and res.end_bytes == nbytes
    assert not [e for e in room.events if e["kind"] == "admit_refused"]
    assert res.report()["bytes"] == res.anchor_bytes + res.stash_bytes + nbytes
    assert res.shed_end(safe=True) == 0 and res.end is not None   # a held anchor stays
    assert res.shed_end() == nbytes and res.end is None and res.end_bytes == 0
    _, pre, _ = turn(prompt + out1 + TOOL + out2 + TOOL2, 4)
    assert pre["kind"] == "resident", pre                          # back to the prompt's line


# ------------------------------------------------------------------ 8. pin and park

def test_the_park_carries_the_line_and_the_end_goes_with_its_rows():
    res = _state(park_bytes=BIG)
    head = tokens(64, seed=70)
    a = head + tokens(60, seed=71)
    out_a, _, _ = turn(a, 30, session="A")
    nxt_a = a + out_a + TOOL + tokens(40, seed=72)
    turn(nxt_a, 10, session="A")                      # rows past 100+: a session's
    assert res.taint == len(a)
    b = head + tokens(70, seed=73)
    turn(b, 10, session="B")                          # A is parked, B in place
    assert res.park_owner == "A" and res.park_taint == len(a) and res.taint == len(b)
    assert res.end is not None and res.end["at"] == len(b) + 9  # B's own end
    _, pre, _ = turn(nxt_a + tokens(5, seed=74), 4, session="A")
    assert pre["kind"] == "resident" and res.taint == len(a), (pre, res.taint)
    # B is parked up to its end, with it: the rows its generation wrote are a session's
    assert res.park_owner == "B" and res.park_taint == len(b), res.park_taint
    assert res.park_end is not None and res.park_end["at"] == len(b) + 9


def test_the_pin_carries_the_line():
    res = _state(pin_bytes=BIG)
    prompt = tokens(100, seed=60)
    out1, _, _ = turn(prompt, 30)
    turn(prompt + out1 + TOOL + tokens(80, seed=75), 4)
    assert res.taint == 100
    compaction = tokens(200, seed=76)                 # shares nothing: the head is pinned
    turn(compaction, 4)
    assert res.pin_rows and res.pin_taint == 100 and res.end["at"] > 199
    _, pre, _ = turn(prompt + out1 + TOOL + tokens(80, seed=75) + [9], 4)
    assert res.stats["pin_hits"] == 1 and res.taint == 100, res.report()


# ------------------------------------------------------------------ 9. off

def test_off_keeps_nothing_and_every_prefill_is_the_one_without_it():
    res = _state(end=False)
    prompt = tokens(100, seed=60)
    out1, _, _ = turn(prompt, 60)
    assert res.end is None and res.report()["end_on"] is False and res.taint is None
    nxt = prompt + out1 + TOOL
    eng = app.STATE["engine"]
    lg, reused, _ = cache.prefill(eng, None, nxt, DEV, chunk=CH, resident=res)
    ref = fresh()
    lr, _, _ = cache.prefill(ref, None, nxt, DEV, chunk=CH)
    assert reused == 96 and torch.equal(lg, lr)
    same_state(eng, ref, len(nxt))
    assert res.replay(nxt) == (nxt, [])
    assert all(v == 0 for k, v in res.stats.items() if k.startswith("end_"))


# ------------------------------------------------------------------ 10. the length router

def test_the_length_router_hands_on_its_released_arm():
    from test_lenrouter import build

    class Arm:
        def __init__(self, inner):
            self.inner, self.resumed = inner, []

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def state_resume(self, n):
            self.resumed.append(n)

    r, small, large = build(latch=True, drop_idle=True)
    r.small, r.large = Arm(small), Arm(large)
    r.idle, r.latched = "s", "l"                      # the latch released the narrow arm
    assert r.end_mark() == "s"
    r.reset()
    assert r.idle is None and r.latched is None
    r.resume_end(140, "s")
    assert (r.idle, r.latched) == ("s", "l") and r.large.resumed == [140] and not r.small.resumed
    r.reset()
    r.resume_end(140, None)
    assert r.small.resumed == [140] and r.large.resumed == [140, 140] and r.idle is None


# ------------------------------------------------------------------ 11. through the handler

class ChatTok(FakeTok):
    """A template whose ids are the content's characters (offset into the vocabulary), with a
    marker per role: an assistant message is rendered the way the generation prompt opened it,
    so a client's copy of an answer is the generated ids again."""

    BASE = 0x100

    def enc(self, text):
        return [ord(c) - self.BASE for c in text]

    def decode(self, seq, skip_special_tokens=True):
        return "".join(chr(self.BASE + int(t)) for t in seq)

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        ids = []
        for m in messages:
            ids += [1 if m["role"] == "user" else 2] + self.enc(m["content"]) + [4]
        ids += [2] if add_generation_prompt else []
        if kw.get("tokenize") is False:
            return self.decode(ids)
        return {"input_ids": torch.tensor([ids])}


def _chat(messages):
    body = {"messages": messages, "max_tokens": 40, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    head, raw = Req("/v1/chat/completions", body).response()
    assert head.startswith("HTTP/1.1 200"), (head, raw[:300])
    return json.loads(raw)


def test_through_the_handler_the_replay_usage_and_the_resume():
    res = _state()
    tok = app.STATE["tok"] = ChatTok()
    user = tok.decode(tokens(90, seed=80))
    _chat([{"role": "user", "content": user}])
    n1 = 93                                           # the marker, 90 ids, the end, the answer's
    out = list(app.STATE["last_ctx"][n1:])
    e = res.end["at"]
    assert e == n1 + len(out) - 1 and len(out) == 40, (e, len(out))
    # as if the monitor had hidden 3 of the answer's tokens: the client's copy lacks them
    q = n1 + 10
    res.end["hidden"] = [(q, out[10:13])]
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": tok.decode(out[:10] + out[13:])},
            {"role": "user", "content": tok.decode(TOOL)}]
    sent = len(tok.apply_chat_template(msgs)["input_ids"][0])
    r2 = _chat(msgs)
    last = app.STATE["last_prefill"]
    assert last["kind"] == "end" and last["reused"] == e, last
    assert r2["usage"]["prompt_tokens"] == sent, (r2["usage"], sent)
    # its own end anchor carries the span on, and the next copy resumes at it the same way
    assert res.end["hidden"] == [(q, out[10:13])], res.end["hidden"]
    e2 = res.end["at"]
    out2 = list(app.STATE["last_ctx"][sent + 3:])
    msgs += [{"role": "assistant", "content": tok.decode(out2)},
             {"role": "user", "content": tok.decode(TOOL2)}]
    _chat(msgs)
    last = app.STATE["last_prefill"]
    assert last["kind"] == "end" and last["reused"] == e2, (last, e2)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:80s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
