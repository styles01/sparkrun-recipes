"""Qwen decides with the conversation it already holds: the in-context stage (2026-10-02).

A long prompt or a follow-up is classified by the engine itself, on the turn's own prompt up to its
last message plus a hidden question, thinking off, one letter; Laya keeps short first messages.

The test matrix:

  the cache (the random 4-layer model, engine/cache.py as the server runs it):
    1  the turn after the stage computes the same logits, recurrent state, convolution tails, KV
       rows and anchors as the turn without it, bit for bit, and re-reads fewer tokens
                                                test_the_turn_after_the_stage_*
    2  a cold first turn (opencode's 24k head): the stage reads it once, the turn resumes at the
       grid line below the insertion point            test_a_cold_first_turn_*
    3  the stage's state-store snapshots are the turn's own (lines below the parting point)
                                                      test_the_stores_snapshots_are_the_turns
    4  a drafter's position-indexed cache is the turn's own after the stage
                                                      test_the_drafters_rows_*
    5  the server's runner (`app.context_choice`) answers the letters' logits of a cold prefill
                                                      test_the_runner_*
  the decision path (the router on the checkpoint's template, the classifiers faked):
    6  off (the seed, and every router stored before the stage): the cascade as before, the
       prompt is never counted                        test_off_*
    7  a short first message: the decider; a follow-up or a long prompt: the stage, no decider
       and no digest question                         test_the_order_*
    8  the stage's prompt is the turn's up to its last message (earlier notes replayed) plus the
       hidden turn; the turn's prompt has none of it and shares every token up to the note
                                                      test_the_stage_prompt_*
    9  below its threshold, failing, or never run: the session's class, journaled
                                                      test_below_*, test_a_failing_*,
                                                      test_a_stage_that_never_ran_*
   10  classify off, the test box, a tool turn: no stage   test_no_stage_*
   12  an image in the history: the stage expands it as the turn does, and the turn after it is
       bit-identical to the turn without the stage       test_an_image_in_history_*
  over HTTP (the fake engine):
   11  a follow-up's class from the stage, the journal's `qwen_context` stage with its prompt
       tokens, prefill and confidence, and nothing of the question in the stream or the usage
                                                      test_over_http

Run: python tests/test_context_stage.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine import cache  # noqa: E402
from server import app  # noqa: E402
from server.conductor import incontext, inject, schema  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402
from test_conductor_router import SYSTEM, QTok, body, setup, text_of, turn  # noqa: E402
from test_resident import Pos, decode, resident, same_state  # noqa: E402

CH = 16


# ------------------------------------------------------------------ the cache

def _two_turns(stage_at: int | None, *, store=None, drafter=None, n1=100, seed=1):
    """Turn 1, its answer, then turn 2, with or without the stage first. The stage's prompt is
    turn 2 up to `stage_at` (its insertion point) plus a 9-token question."""
    turn1 = tokens(n1, seed=seed)
    res, eng = resident(), fresh()
    cache.prefill(eng, drafter, turn1, DEV, chunk=CH, resident=res, store=store, checkpoint=True)
    answer = decode(eng, len(turn1))
    turn2 = turn1 + answer + tokens(37, seed=seed + 1)
    stage = None
    if stage_at is not None:
        c = turn2[:stage_at] + tokens(9, seed=seed + 2)
        if drafter is not None:
            drafter.reset()
        stage = cache.prefill(eng, drafter, c, DEV, chunk=CH, resident=res, store=store,
                              checkpoint=True, checkpoint_upto=stage_at)
    if drafter is not None:
        drafter.reset()
    info = {}
    lg, reused, fwd = cache.prefill(eng, drafter, turn2, DEV, chunk=CH, resident=res, store=store,
                                    checkpoint=True, info=info)
    return eng, res, lg, reused, fwd, turn2, stage


def test_the_turn_after_the_stage_is_the_turn_without_it():
    at = 100 + 9 + 30                     # the insertion point: 7 tokens before turn 2's end
    a, res_a, lg_a, re_a, fwd_a, t2, _ = _two_turns(None)
    b, res_b, lg_b, re_b, fwd_b, _, stage = _two_turns(at)
    assert torch.equal(lg_a, lg_b), (lg_a - lg_b).abs().max().item()
    same_state(a, b, len(t2))
    # the anchors the turn leaves: the same lines, the same recurrent state at each
    assert sorted(res_a.anchors) == sorted(res_b.anchors)
    for k in res_a.anchors:
        assert torch.equal(res_a.anchors[k][0], res_b.anchors[k][0]), k
        assert torch.equal(res_a.anchors[k][1], res_b.anchors[k][1]), k
    assert res_a.tokens == res_b.tokens == t2 and res_a.valid == res_b.valid
    # what each re-reads: without the stage the answer and the new messages from line 96; with
    # it, the stage read those (and its question) and the turn resumes at line 128 <= 139
    assert (re_a, fwd_a) == (96, len(t2) - 96), (re_a, fwd_a)
    assert (re_b, fwd_b) == (128, len(t2) - 128), (re_b, fwd_b)
    s_lg, s_reused, s_fwd = stage
    assert (s_reused, s_fwd) == (96, at + 9 - 96)
    print(f"    re-read by the turn: {fwd_a} without the stage, {fwd_b} with it "
          f"(the stage itself prefilled {s_fwd}, {at + 9} tokens of prompt)")


def test_a_cold_first_turn_is_read_once():
    """opencode's first message: nothing resident. The stage prefills it cold, the turn resumes
    at the grid line below the insertion point."""
    ids = tokens(200, seed=11)
    at = 190
    res, eng = resident(), fresh()
    _, s_re, s_fwd = cache.prefill(eng, None, ids[:at] + tokens(5, seed=12), DEV, chunk=CH,
                                   resident=res, checkpoint=False)
    lg, reused, fwd = cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
    ref = fresh()
    lg_ref, _, _ = cache.prefill(ref, None, ids, DEV, chunk=CH)
    assert torch.equal(lg, lg_ref)
    same_state(eng, ref, len(ids))
    assert (s_re, s_fwd, reused, fwd) == (0, at + 5, 176, 24), (s_re, s_fwd, reused, fwd)


def _lines(st) -> list[int]:
    return sorted(e.snap.length for e in st._d.values())


def test_the_stores_snapshots_are_the_turns():
    """A short session lives in the state store. The stage takes prefix snapshots only at lines
    up to where the turn's prompt parts from it (those tokens are the turn's) and no session
    snapshot, so the store holds what the turn alone leaves; with the served one snapshot a
    prefill, one more at most, when the turn's tail crosses a grid line."""
    for per, at, extra in ((1, 60 + 9 + 31, []), (1, 60 + 9 + 20, [80]), (0, 60 + 9 + 20, [])):
        st_a = cache.StateStore(1 << 30, chunk=CH, per_prefill=per)
        st_b = cache.StateStore(1 << 30, chunk=CH, per_prefill=per)
        a, _, lg_a, _, _, t2, _ = _two_turns(None, store=st_a, n1=60, seed=5)
        b, _, lg_b, _, _, _, _ = _two_turns(at, store=st_b, n1=60, seed=5)
        assert torch.equal(lg_a, lg_b)
        same_state(a, b, len(t2))
        la, lb = _lines(st_a), _lines(st_b)
        assert sorted(set(lb) - set(la)) == extra and set(la) <= set(lb), (per, at, la, lb)
        assert all(x <= at for x in extra)
        for e in st_b._d.values():                      # every entry is a prefix of the turn
            assert list(e.tokens[:e.snap.length]) == t2[:e.snap.length]


def test_the_drafters_rows_are_the_turns_own():
    at = 100 + 9 + 30
    _, _, _, _, _, t2, _ = _two_turns(at, drafter=(d := Pos()))
    ref, refd = fresh(), Pos()
    cache.prefill(ref, refd, t2, DEV, store=None, chunk=CH)
    assert d.ctx_len == len(t2)
    assert torch.equal(d.buf[:len(t2)], refd.buf[:len(t2)])


def test_the_runner_answers_a_cold_prefills_letter_logits():
    ids = tokens(70, seed=21)
    letters = [3, 9, 27, 40]
    eng, res = fresh(), resident()
    app.STATE.clear()
    app.STATE.update(engine=eng, drafter=None, resident=res, state_store=None, prefix_chunk=CH,
                     device=DEV, model="tiny")
    got, info = app.context_choice(ids, letters, 60, "ses")
    ref = fresh()
    lg, _, _ = cache.prefill(ref, None, ids, DEV, chunk=CH)
    want = lg[0, -1].float()[letters].tolist()
    assert got == want and info == {"reused": 0, "forwarded": 70, "model": "tiny"}, (got, info)
    # nothing was decoded: the rows end at the prompt, and the anchors are a prefill's
    assert eng.kv.length == 70 and sorted(res.anchors) == list(range(16, 70, 16))


def test_an_image_in_history_goes_through_the_stage_bit_for_bit():
    """opencode's photo3d and spiral sessions carry screenshots in their history. The stage
    expands the images as the turn does (the same rows, rotary coordinates and `key_ids`), so the
    follow-up's turn after it resumes from its anchors and computes what it computes without it."""
    import test_vision as TV
    if TV.need_hf():
        print(f"    {TV.need_hf()}")
        return
    _, d = TV.reference()
    img = TV.image(41, (1, 4, 6))
    turn1 = TV.prompt([img], before=tuple(range(20, 40)), after=tuple(range(40, 60)))
    turn2 = turn1 + list(range(60, 70)) + list(range(70, 83))   # the answer, a follow-up
    at = len(turn2) - 4                                          # where the turn's note goes
    letters = [11, 12, 13]

    def run(stage: bool):
        eng, tower = TV.engine_for(d)
        res = cache.ResidentPrefix(1 << 26, 8, stash_bytes=1 << 26)
        full, mm = TV.mm_context(eng, tower, turn1, [img])
        eng.mm, eng.pos_delta = mm, mm.delta
        with torch.no_grad():
            cache.prefill(eng, None, full, "cpu", chunk=8, resident=res, key_ids=mm.key_ids)
            eng.forward(torch.tensor(list(range(60, 65))), start=len(full), last_only=True)
        got = None
        if stage:
            app.STATE.clear()
            app.STATE.update(engine=eng, drafter=None, vision=tower, image_cache=None,
                             resident=res, state_store=None, prefix_chunk=8, device="cpu",
                             model="tiny-vl")
            got = app.context_choice(turn2[:at] + [5, 6, 7, 8, 9, 10], letters, at, "ses", None,
                                     [img])
            assert eng.mm is None and eng.pos_delta == 0
        full2, mm2 = TV.mm_context(eng, tower, turn2, [img])
        eng.mm, eng.pos_delta = mm2, mm2.delta
        with torch.no_grad():
            lg, reused, fwd = cache.prefill(eng, None, full2, "cpu", chunk=8, resident=res,
                                            key_ids=mm2.key_ids)
        return eng, res, lg, reused, fwd, got, full2

    a, res_a, lg_a, re_a, fwd_a, _, full2 = run(False)
    b, res_b, lg_b, re_b, fwd_b, got, _ = run(True)
    assert torch.equal(lg_a, lg_b), (lg_a - lg_b).abs().max().item()
    same_state(a, b, len(full2))
    assert sorted(res_a.anchors) == sorted(res_b.anchors)
    for k in res_a.anchors:
        assert torch.equal(res_a.anchors[k][0], res_b.anchors[k][0]), k
    logits, info = got
    assert len(logits) == 3 and info["reused"] > 0, info       # it resumed past the image rows
    assert fwd_b < fwd_a and re_b > re_a, (re_a, fwd_a, re_b, fwd_b)
    # the stage's letters are a cold prefill's of the same expanded prompt
    eng, tower = TV.engine_for(d)
    cfull, cmm = TV.mm_context(eng, tower, turn2[:at] + [5, 6, 7, 8, 9, 10], [img])
    eng.mm, eng.pos_delta = cmm, cmm.delta
    with torch.no_grad():
        clg, _, _ = cache.prefill(eng, None, cfull, "cpu", chunk=8, key_ids=cmm.key_ids)
    assert logits == clg[0, -1].float()[letters].tolist()
    print(f"    with an image in history the turn re-read {fwd_a} without the stage, {fwd_b} "
          f"with it (the stage prefilled {info['forwarded']})")


# ------------------------------------------------------------------ the decision path

FOLLOW = "Now make the parser report the column too."


def _on(cond, **kw):
    """A router version with the stage on (and the session's switch threshold low enough that a
    confident stage answer moves the class)."""
    _, doc = cond.store.get("router")
    doc = json.loads(json.dumps(doc))
    doc["classifier"]["context_stage"] = dict(schema.CONTEXT_STAGE, mode="on", **kw)
    cond.store.put("router", None, doc, note="the in-context stage on", author="test")
    return doc


class Runner:
    """The engine's part, faked: the letter of `pick` at logit `hi` against 0."""

    def __init__(self, pick="code_edit", hi=4.0, fail=None):
        self.pick, self.hi, self.fail, self.calls = pick, hi, fail, []

    def __call__(self, ids, letters, at):
        self.calls.append((list(ids), list(letters), at))
        if self.fail:
            raise self.fail
        text = QTok().decode(ids, skip_special_tokens=False)
        listed = dict((cid, c) for c, cid in
                      __import__("re").findall(r"^([A-Z]): ([a-z_]+) - ", text, flags=2 | 8))
        want = listed.get(self.pick)
        return ([self.hi if chr(t) == want else 0.0 for t in letters],
                {"reused": 64, "forwarded": len(ids) - 64})


def _session(cond, fakes):
    """Turn 1 (a code request: the decider), its answer, and turn 2 (a follow-up)."""
    m1 = [{"role": "system", "content": SYSTEM},
          {"role": "user", "content": "Refactor the lexer in src/lexer.ts so tokens carry lines."}]
    p1, ids1, _ = turn(cond, body(m1))
    assert p1.row["source"] == "decider" and p1.context is None
    m2 = m1 + [{"role": "assistant", "content": "Done: every token carries its line."},
               {"role": "user", "content": FOLLOW}]
    return m2, p1, ids1


def _turn_in_context(cond, b, runner, headers=None):
    """The server's order: plan, render, the stage under the lock, apply."""
    plan = cond.plan(b, headers or {"x-session-id": "ses-1"}, client_kind="opencode",
                     request_id="req", chat=True,
                     count_tokens=lambda: app.build_prompt(b, count=True))
    prompt, _, in_think = app.build_prompt(b)
    if plan.context is not None:
        cond.decide_in_context(plan, prompt.tolist(), runner)
    out, _ = plan.apply_prompt(prompt, app.STATE["tok"], in_think)
    return plan, out.tolist(), plan.clean(prompt.tolist())


def test_off_is_the_cascade_and_never_counts_the_prompt():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    assert schema.context_stage(cond.store.get("router")[1])["mode"] == "off"
    m2, _, _ = _session(cond, fakes)
    counted = []
    b = body(m2)
    plan = cond.plan(b, {"x-session-id": "ses-1"}, client_kind="opencode", request_id="r",
                     chat=True, count_tokens=lambda: counted.append(1) or 10 ** 6)
    assert plan.context is None and not counted
    assert plan.row["source"] == "decider" and "context_stage" not in plan.row["extra"]
    # a router stored before the stage existed has no key at all, and is off
    _, doc = cond.store.get("router")
    doc = json.loads(json.dumps(doc))
    doc["classifier"].pop("context_stage")
    assert not schema.check("router", doc)
    assert schema.context_stage(doc)["mode"] == "off"


def test_the_order_short_first_message_decider_followup_stage():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93),
                        qwen=lambda t, o: ("chat", 0.99))
    _on(cond)
    m2, p1, _ = _session(cond, fakes)
    assert len(fakes["decider"].calls) == 1           # turn 1: short, first: the decider
    runner = Runner("debug")
    p2, ids2, clean2 = _turn_in_context(cond, body(m2), runner)
    assert len(fakes["decider"].calls) == 1 and not fakes["qwen"].calls, "no digest question"
    assert len(runner.calls) == 1
    row = p2.row
    st = row["stages"][0]
    assert (row["class_id"], row["source"], row["effort"]) == ("debug", "qwen_context", "xhigh")
    assert (st["stage"], st["label"], st["reason"], st["error"]) == \
        ("qwen_context", "debug", "followup", None), st
    assert st["confidence"] > 0.85 and st["prompt_tokens"] == len(runner.calls[0][0])
    assert st["prefilled"] == len(runner.calls[0][0]) - 64 and st["reused"] == 64
    assert row["extra"]["context_stage"] == {"due": "followup"}


def test_the_order_a_long_first_message_goes_to_the_stage():
    cond, fakes = setup(decider=lambda t, o: ("chat", 0.99))
    _on(cond, min_tokens=400)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Plan the migration of the store to SQLite. " * 20}]
    runner = Runner("plan")
    p, _, _ = _turn_in_context(cond, body(msgs), runner)
    assert not fakes["decider"].calls and len(runner.calls) == 1
    assert (p.row["class_id"], p.row["source"]) == ("plan", "qwen_context")
    assert p.row["extra"]["context_stage"]["due"] == "long"
    assert p.row["extra"]["context_stage"]["tokens"] >= 400
    # the same message under the threshold: the decider
    cond, fakes = setup(decider=lambda t, o: ("chat", 0.99))
    _on(cond, min_tokens=10 ** 5)
    p, _, _ = _turn_in_context(cond, body(msgs), Runner())
    assert p.row["source"] == "decider" and p.context is None
    # followups off: a follow-up under the threshold is the decider's too
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    _on(cond, followups=False, min_tokens=10 ** 5)
    m2, _, _ = _session(cond, fakes)
    p, _, _ = _turn_in_context(cond, body(m2), Runner())
    assert p.row["source"] == "decider" and len(fakes["decider"].calls) == 2


def test_the_stage_prompt_is_the_turns_prompt_plus_a_hidden_turn():
    cond, fakes = setup(decider=lambda t, o: ("design_ui", 0.95))
    _on(cond)
    m1 = [{"role": "system", "content": SYSTEM},
          {"role": "user", "content": "Design a settings page for the dashboard."}]
    p1, ids1, _ = turn(cond, body(m1))
    note1 = p1.injected[0]["text"]                     # the ui-design note, in turn 1's tail
    m2 = m1 + [{"role": "assistant", "content": "Here is the page."},
               {"role": "user", "content": FOLLOW}]
    runner = Runner("code_edit")
    p2, ids2, clean2 = _turn_in_context(cond, body(m2), runner)
    c = runner.calls[0][0]
    c_text, p_text = text_of(c), text_of(ids2)
    # the earlier note is replayed in the stage's prompt where it sits in the turn's
    assert c_text.count(note1) == 1 and p_text.count(note1) == 1
    # the two part exactly where the turn's own note goes (before the last <|im_end|>): every
    # token before it is shared, so the turn resumes from what the stage prefilled
    assert p2.injected, "code_edit's note"
    common = next(i for i, (x, y) in enumerate(zip(c, ids2)) if x != y)
    assert common == p2.injected[0]["at"], (common, p2.injected[0]["at"])
    assert c_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert FOLLOW + "<|im_end|>\n<|im_start|>user\nThe runtime that serves you" in c_text
    assert "A: code_edit - " in c_text and "I: chat - " in c_text
    # the turn's prompt has nothing of the question, and the client's prompt nothing of either
    assert "The runtime that serves you asks" not in p_text
    assert "The runtime that serves you asks" not in text_of(clean2)
    assert note1 not in text_of(clean2)


def test_below_the_threshold_keeps_the_sessions_class():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    _on(cond)
    m2, _, _ = _session(cond, fakes)
    p, _, _ = _turn_in_context(cond, body(m2), Runner("chat", hi=0.1))
    st = p.row["stages"][0]
    assert st["below"] and st["label"] == "chat" and st["confidence"] < 0.55
    assert (p.row["class_id"], p.row["source"], p.row["kept"]) == \
        ("code_edit", "fallback", "no_answer")


def test_a_failing_stage_fails_open():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    _on(cond)
    m2, _, _ = _session(cond, fakes)
    p, ids, clean = _turn_in_context(cond, body(m2), Runner(fail=RuntimeError("boom")))
    st = p.row["stages"][0]
    assert st["error"] == "qwen_context_RuntimeError", st
    assert (p.row["class_id"], p.row["source"], p.row["kept"]) == \
        ("code_edit", "fallback", "fail_open")
    assert p.row["effort"] == "xhigh" and ids         # the turn is served on


def test_a_stage_that_never_ran_settles_the_row():
    """The request ended before the engine lock (queue full, the client left): the row still
    gets a class when the conductor finishes it."""
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    _on(cond)
    m2, _, _ = _session(cond, fakes)
    b = body(m2)
    plan = cond.plan(b, {"x-session-id": "ses-1"}, client_kind="opencode", request_id="r2",
                     chat=True, count_tokens=lambda: 10)
    assert plan.context is not None and plan.row.get("class_id") is None

    class Rec:
        status, finish_reason, prompt_tokens = 503, "refused", None

    cond.finish(plan, Rec())
    assert plan.context is None
    assert (plan.row["class_id"], plan.row["kept"], plan.row["stages"][0]["error"]) == \
        ("code_edit", "fail_open", "qwen_context_not_run")


def test_no_stage_for_classify_off_the_test_box_and_tool_turns():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    _on(cond)
    m2, _, _ = _session(cond, fakes)
    b = body(m2)
    # a tool-result turn keeps the class without any classifier
    m3 = m2 + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    p = cond.plan(body(m3), {"x-session-id": "ses-1"}, client_kind="opencode", request_id="r",
                  chat=True, count_tokens=lambda: 10)
    assert p.context is None and p.row["source"] == "sticky"
    # the test box explains with the cascade (it has no engine to ask)
    out = cond.router.explain(FOLLOW, "coding", messages=m2)
    assert out["source"] == "decider", out
    # classify off: the profile's default class
    _, prof = cond.store.get("profiles", "coding")
    prof = json.loads(json.dumps(prof))
    prof["decisions"]["classify"] = "off"
    cond.store.put("profiles", "coding", prof, note="classify off", author="test")
    p = cond.plan(b, {"x-session-id": "ses-2"}, client_kind="opencode", request_id="r",
                  chat=True, count_tokens=lambda: 10 ** 6)
    assert p.context is None and p.row["source"] == "profile"


def test_the_count_is_the_prompts_length_without_a_device_tensor():
    """The router counts before the engine lock, where nothing may touch the GPU: the count is
    the rendered prompt's length on both render paths, and an int."""
    from server.tokcache import SegmentCache

    class DictTok(QTok):
        """QTok as a fast tokenizer answers a plain call: a mapping (the segment cache reads it)."""

        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            return {"input_ids": super().__call__(text, add_special_tokens, return_tensors).input_ids}

    setup()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FOLLOW},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "and the tests?"}]
    for seg in (None, SegmentCache(DictTok(), 10 ** 6)):
        app.STATE["segments"] = seg
        app.STATE["tok"] = DictTok() if seg is not None else QTok()
        b = body(msgs)
        n = app.build_prompt(b, count=True)
        assert isinstance(n, int) and n == int(app.build_prompt(b)[0].numel()), (seg, n)
    app.STATE.update(segments=None, tok=QTok())


def test_the_letters_and_the_answer():
    text, letters = incontext.listing({"a": "first", "b": "second"})
    assert text == "A: a - first\nB: b - second" and letters == ["A", "B"]
    assert incontext.letter_ids(QTok(), letters) == [ord("A"), ord("B")]
    label, conf, probs = incontext.answer([0.0, 2.0], ["a", "b"])
    assert label == "b" and abs(conf - 0.8808) < 1e-3 and abs(sum(probs.values()) - 1) < 1e-9

    class Two:
        def __call__(self, text, add_special_tokens=False):
            return type("R", (), {"input_ids": [1, 2]})()

    try:
        incontext.letter_ids(Two(), ["A"])
        raise AssertionError("a letter of two tokens must refuse")
    except incontext.ContextError as exc:
        assert exc.why == "letter_not_one_token"


# ------------------------------------------------------------------ over HTTP

def _row(e2e, port, key):
    import time
    for _ in range(50):
        rows = [r for r in e2e._decisions(port, limit=50)
                if r.get("session_key") == key and r.get("class_id")]
        if rows:
            return rows[0]
        time.sleep(0.1)
    raise AssertionError(f"no route row for {key}")


def test_over_http():
    import test_conductor_e2e as e2e
    port = e2e._port()
    cdir = tempfile.mkdtemp(prefix="qse-ctx-")
    srv, _ = e2e._server(port, cdir)
    try:
        e2e._wait(port, srv)
        st, _, cur = e2e._call(port, "GET", "/v1/conductor/router")
        assert st == 200, cur
        doc = cur["doc"]
        doc["classifier"]["context_stage"] = dict(schema.CONTEXT_STAGE, mode="on")
        st, _, w = e2e._call(port, "PUT", "/v1/conductor/router",
                             {"doc": doc, "note": "the in-context stage on"})
        assert st == 200, w
        hist = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Write a lexer."},
                {"role": "assistant", "content": "Here it is."}]
        msgs = hist + [{"role": "user", "content": "FAKE_CLASS design_ui make the page nicer"}]
        text, hdrs, usage = e2e._stream(port, msgs, headers={"x-session-id": "ctx-http"},
                                        stream_options={"include_usage": True})
        assert hdrs["X-Tandem"] == "on"
        assert "runtime that serves you asks" not in text and "A: code_edit" not in text, text
        row = _row(e2e, port, "c:ctx-http")
        st0 = row["stages"][0]
        assert (row["class_id"], row["source"], st0["stage"], st0["label"]) == \
            ("design_ui", "qwen_context", "qwen_context", "design_ui"), row
        assert st0["prompt_tokens"] > 0 and st0["prefilled"] > 0 and st0["confidence"] > 0.9
        # the class the stage picks serves the turn exactly as the same class picked without it:
        # the same stream, byte for byte, and the same usage (chat is the chat profile's
        # default, which the turn gets with the stage off and no classifier answering)
        same = hist + [{"role": "user", "content": "FAKE_CLASS chat add a test"}]
        on_text, _, on_usage = e2e._stream(port, same, headers={"x-session-id": "ctx-on"},
                                           stream_options={"include_usage": True})
        assert _row(e2e, port, "c:ctx-on")["source"] == "qwen_context"
        doc["classifier"]["context_stage"]["mode"] = "off"
        assert e2e._call(port, "PUT", "/v1/conductor/router",
                         {"doc": doc, "note": "off"})[0] == 200
        off_text, _, off_usage = e2e._stream(port, same, headers={"x-session-id": "ctx-off"},
                                             stream_options={"include_usage": True})
        off_row = _row(e2e, port, "c:ctx-off")
        assert off_row["class_id"] == "chat" and off_row["source"] != "qwen_context", off_row
        assert on_text == off_text, (on_text, off_text)
        assert on_usage["prompt_tokens"] == off_usage["prompt_tokens"]
        assert on_usage["completion_tokens"] == off_usage["completion_tokens"]
        # a failing stage: the request is served, the row says why
        doc["classifier"]["context_stage"]["mode"] = "on"
        assert e2e._call(port, "PUT", "/v1/conductor/router",
                         {"doc": doc, "note": "on"})[0] == 200
        bad = hist + [{"role": "user", "content": "FAKE_CONTEXT_ERROR now what"}]
        text, _, _ = e2e._stream(port, bad, headers={"x-session-id": "ctx-bad"})
        assert text
        assert _row(e2e, port, "c:ctx-bad")["stages"][0]["error"] == "qwen_context_RuntimeError"
    finally:
        srv.terminate()
        srv.wait(10)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
