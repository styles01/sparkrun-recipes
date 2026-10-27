"""The background memory writer (2026-10-05, P-M2, server/conductor/memwriter.py and
`app.memory_once`), on the random 4-layer model as the preparer's tests (CPU torch).

  1  idle and due (the session's end): the writer's hidden turn runs in scratch rows, its blocks
     are written as `qwen-writer` provisional notes, and the conversation's rows, anchors and end
     anchor are as they were (the next turn resumes at the end and writes the same tokens)
                                                        test_the_writer_writes_and_leaves_the_rows
  2  a request that waits stops it; nothing is written and nothing moves
                                                        test_a_waiting_request_stops_the_writer
  3  not due: too few new tokens since the last pass, mode off, no session seen
                                                        test_not_due
  4  the gate: Laya's no skips the turn (journaled), a mark (compaction, give_up) makes it due
     after `due_idle_s`                                 test_gate_and_marks
  5  shadow: the turn runs, the ops are counted, nothing is stored
                                                        test_shadow_stores_nothing
  6  a held box lock (a hold's): the look is "hold", kept in the stats and printed once; with the
     lock free the session is written                  test_a_held_box_lock_is_reported
  7  a compaction whose record is not in place (a short compaction prompt served as a guest, on the
     box 2026-10-05): its mark stays and the idle look writes the session with trigger compaction
                                                        test_a_compaction_without_its_record_waits
  8  the record from the rows' last prompt stops before its last `<|im_end|>` (the writer's tail
     closes that message; the generation prompt after it is left out)
                                                        test_the_prompt_record_stops_at_its_last_end
  9  the tool form (`<tool_call><function=tandem_memory>...`, what Qwen wrote on the box with the
     session's tools in the record) is written as its block; one without a title is refused
                                                        test_the_tool_form_is_a_block
 10  a tool call that is not a memory block (after a compaction Qwen answered an empty or a `bash`
     call, MEM2 2026-10-05) ends the turn at its function's name and the run says NONE (`none`,
     `answer: tool_call`, `cut`); an empty answer says NONE too; prose without a block does not
                                                        test_a_foreign_tool_call_is_none
 11  a resident conversation the memory never saw (no session id, raw, memory off) is "no
     session", printed once as such (it printed "nothing" with the record's numbers)
                                                        test_no_session_is_said_once
 12  M3: `ban_tool_call` gives the writer's turn a logit bias of -100 on the `<tool_call>` token
     (here the token the greedy turn chose first without it), and the turn never chooses it; a
     tokenizer without the token bans nothing                   test_the_tool_call_token_is_banned
 13  M3: `prefill` follows the generation prompt and starts the answer the parser reads (a
     prefilled NONE is the run's none)                  test_the_prefill_starts_the_answer

Run: python tests/test_memory_writer.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_compaction_continue as C  # noqa: E402  (sets the CPU environment first)

from server import app, compaction  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from test_cache import tokens  # noqa: E402
from test_end_anchor import _state  # noqa: E402

BLOCK = ('<tandem_memory op="add" scope="project" kind="gotcha">\ntitle: the checker wants node 20'
         '\n\nThe build fails on node 18 with an import attribute error; use node 20.\n'
         '</tandem_memory>')


class _Tok(C._Tok):
    """The preparer's tokenizer, whose decode of the writer's output is `out` (the random model
    writes no block of its own)."""

    out = BLOCK

    def decode(self, ids, skip_special_tokens=False):
        return self.out


def _setup(mode="on", gate=False, idle_s=0.0, min_new=1, decider=None, after="idle"):
    res = C._prep_state()
    app.STATE["compaction"]["prepare"] = "off"
    cond = Conductor(tempfile.mkdtemp(prefix="cond-mw-"), state=app.STATE, journal=False,
                     factory=lambda kind, role, timeout: decider)
    v, st = cond.store.get("settings")
    st.setdefault("memory", {})["notes"] = {"writer": {
        "mode": mode, "gate": gate, "idle_s": max(10.0, idle_s), "due_idle_s": 0.0,
        "min_new_tokens": min_new, "max_tokens": 64, "after_compaction": after}}
    cond.store.put("settings", None, st, note="test", author="test", base=v)
    if decider is not None:
        v, roles = cond.store.get("roles")
        roles["decider"] = dict(roles.get("decider") or {}, mode="on", endpoint="self")
        cond.store.put("roles", None, roles, note="test", author="test", base=v)
    app.STATE["conductor"] = cond
    ctx = {"mode": "on", "user": "owner", "project": "/home/user/app", "session": "c:ses",
           "from_user": True}
    cond.memory.writer.seen("ses", ctx, "Remember that the build needs node 20.")
    return res, cond


def _idle(seconds=1000.0):
    app.STATE["last_end_mono"] = app.time.monotonic() - seconds


def test_the_writer_writes_and_leaves_the_rows():
    res, cond = _setup()
    prompt = tokens(90, seed=81)
    out1, _ = C._turn(prompt, 30)
    before = (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint)
    app.STATE["tok"] = _Tok()
    _idle()
    got = app.memory_once()
    assert got == "wrote", got
    assert (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint) == before
    ns = cond.notes.notes("owner")
    assert [(n.title, n.writer, n.status, n.scope) for n in ns] == [
        ("the checker wants node 20", "qwen-writer", "provisional", "project")], ns
    run = cond.memory.writer.runs[0]
    assert run["trigger"] == "idle" and run["ops"][0]["op"] == "add" and run["tokens"] > 0, run
    assert app.memory_once() == "not_due"                 # done at this record
    tool = tokens(17, seed=82)
    nxt, pre = C._turn(prompt + out1 + tool, 10)
    assert pre["kind"] == "end", pre
    _state(end=True)
    C._turn(prompt, 30)
    assert nxt == C._turn(prompt + out1 + tool, 10)[0]


def test_a_waiting_request_stops_the_writer():
    res, cond = _setup()
    prompt = tokens(90, seed=83)
    out1, _ = C._turn(prompt, 30)
    before = (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint)
    app.STATE["tok"] = _Tok()
    _idle()
    calls = {"n": 0}
    real = app._waiting

    def waiting():
        calls["n"] += 1
        return calls["n"] > 6
    app._waiting = waiting
    try:
        got = app.memory_once()
    finally:
        app._waiting = real
    assert got == "stopped", got
    assert (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint) == before
    assert cond.notes.notes("owner") == [] and not app.LOCK.locked()
    assert cond.memory.writer.runs[0]["skipped"] == "stopped"


def test_not_due():
    res, cond = _setup(min_new=10_000)
    C._turn(tokens(90, seed=84), 30)
    _idle()
    assert app.memory_once() == "not_due"
    res, cond = _setup(mode="off")
    C._turn(tokens(90, seed=85), 30)
    _idle()
    assert app.memory_once() == "off"
    res, cond = _setup()
    C._turn(tokens(90, seed=86), 30, session="other")
    _idle()
    assert app.memory_once() == "no_session"             # a session the memory never saw


class _Decider:
    def __init__(self, label):
        self.label = label

    def choice(self, text, question, options, max_chars=None):
        from server.conductor.classify import Answer
        return Answer("decider", self.label, 0.95, {})

    def state_chars(self):
        return 4000


def test_gate_and_marks():
    res, cond = _setup(gate=True, min_new=10_000, decider=_Decider("no"))
    C._turn(tokens(90, seed=87), 30)
    _idle(1.0)                                           # not the session's end yet
    assert app.memory_once() == "not_due"
    cond.memory.writer.mark("ses", "compaction")
    assert app.memory_once() == "gate_no"
    assert cond.memory.writer.runs[0]["skipped"] == "gate_no"
    assert app.memory_once() == "not_due"                # the mark was used
    res, cond = _setup(gate=True, min_new=10_000, decider=_Decider("yes"))
    C._turn(tokens(90, seed=88), 30)
    app.STATE["tok"] = _Tok()
    _idle(1.0)
    cond.memory.writer.mark("ses", "give_up")
    assert app.memory_once() == "wrote"
    assert cond.memory.writer.runs[0]["trigger"] == "give_up"


def test_shadow_stores_nothing():
    res, cond = _setup(mode="shadow")
    C._turn(tokens(90, seed=89), 30)
    app.STATE["tok"] = _Tok()
    _idle()
    assert app.memory_once() == "wrote"
    run = cond.memory.writer.runs[0]
    assert run["mode"] == "shadow" and run["ops"] == [{"op": "add", "kind": "gotcha",
                                                       "scope": "project", "stored": False}], run
    assert cond.notes.notes("owner") == []


def test_a_held_box_lock_is_reported():
    import contextlib
    import fcntl
    import io
    res, cond = _setup()
    C._turn(tokens(90, seed=90), 30)
    app.STATE["tok"] = _Tok()
    _idle()
    fd = os.open(os.environ["BOX_LOCK"], os.O_RDONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for _ in range(3):
                got = app.memory_once()
                app._writer_look(got)
        assert got == "hold", got
        assert cond.memory.writer.stats["last_look"] == "hold"
        assert out.getvalue().count("[memory] writer look: hold") == 1, out.getvalue()
        assert cond.notes.notes("owner") == [] and not cond.memory.writer.runs
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    got = app.memory_once()
    app._writer_look(got)
    assert got == "wrote" and cond.memory.writer.stats["last_look"] == "wrote", got
    assert [n.writer for n in cond.notes.notes("owner")] == ["qwen-writer"]


def test_a_compaction_without_its_record_waits():
    import types
    res, cond = _setup(min_new=10_000, after="turn")
    C._turn(tokens(90, seed=91), 30)
    app.STATE["tok"] = _Tok()
    plan = types.SimpleNamespace(compaction=True)
    real = app._writer_record
    app._writer_record = lambda res, conv: None          # the guest's rows: no record in place
    try:
        app._writer_after_compaction(plan, "ses")
    finally:
        app._writer_record = real
    assert not cond.memory.writer.runs and cond.notes.notes("owner") == []
    assert cond.memory.writer.state("ses")["due"] == "compaction"
    _idle(1.0)                                           # due_idle_s 0: the next idle look
    assert app.memory_once() == "wrote"
    assert cond.memory.writer.runs[0]["trigger"] == "compaction"
    assert [n.writer for n in cond.notes.notes("owner")] == ["qwen-writer"]
    # with the record in place the writer runs at once, before the next request
    res, cond = _setup(min_new=10_000, after="turn")
    C._turn(tokens(90, seed=92), 30)
    app.STATE["tok"] = _Tok()
    app._writer_after_compaction(plan, "ses")
    assert cond.memory.writer.runs[0]["trigger"] == "compaction", cond.memory.writer.runs


def test_the_prompt_record_stops_at_its_last_end():
    class T:
        def convert_tokens_to_ids(self, t):
            return 7 if t == "<|im_end|>" else None
    keep = app.STATE.get("tok")
    app.STATE["tok"] = T()
    try:
        assert app._last_im_end([1, 2, 7, 3, 7, 4, 5]) == 4
        assert app._last_im_end([1, 2, 3]) is None
    finally:
        app.STATE["tok"] = keep
    app.STATE["tok"] = object()                          # a tokenizer without the method
    try:
        assert app._last_im_end([1, 7]) is None
    finally:
        app.STATE["tok"] = keep


CALL = ("I'll keep the build fact.\n<tool_call>\n<function=tandem_memory>\n<parameter=op>\nadd\n"
        "</parameter>\n<parameter=scope>\nproject\n</parameter>\n<parameter=kind>\ngotcha\n"
        "</parameter>\n<parameter=title>\nthe checker wants node 20\n</parameter>\n"
        "<parameter=body>\nThe build fails on node 18; use node 20.\n</parameter>\n</function>\n"
        "</tool_call>")


def test_the_tool_form_is_a_block():
    res, cond = _setup()
    C._turn(tokens(90, seed=93), 30)
    tok = _Tok()
    tok.out = CALL + CALL.replace("<parameter=title>\nthe checker wants node 20\n</parameter>\n",
                                  "")
    app.STATE["tok"] = tok
    _idle()
    assert app.memory_once() == "wrote"
    run = cond.memory.writer.runs[0]
    assert run["tool_form"] == 2 and len(run["ops"]) == 1 and len(run["refused"]) == 1, run
    assert [(n.title, n.writer) for n in cond.notes.notes("owner")] == [
        ("the checker wants node 20", "qwen-writer")]


def test_a_foreign_tool_call_is_none():
    import contextlib
    import io
    from server.conductor import memwriter as mw
    bash = ("<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n"
            "</tool_call>")
    assert mw.foreign_call(bash) == "bash"
    assert mw.foreign_call("<tool_call>\n</tool_call>") == "empty"
    assert mw.foreign_call("<tool_call>\n<function=") is None          # still open
    assert mw.foreign_call("<tool_call>\n<function=ba") is None
    assert mw.foreign_call(CALL) is None and mw.foreign_call(BLOCK) is None
    assert mw.foreign_call(CALL + "\n" + bash) == "bash"                # after a memory call
    assert mw.foreign_call("<function=read>\n") == "read"               # the opener dropped
    assert mw.answer_form("NONE", []) == "none" and mw.answer_form("  ", []) == "empty"
    assert mw.answer_form(bash, []) == "tool_call" and mw.answer_form("ok then", []) == "text"
    assert mw.answer_form("x", ["b"]) == "blocks"
    for out, form, cut in ((bash, "tool_call", "bash"),
                           ("<tool_call>\n</tool_call>", "tool_call", "empty"),
                           ("", "empty", None), ("NONE", "none", None),
                           ("The session is about the build.", "text", None)):
        res, cond = _setup()
        C._turn(tokens(90, seed=94), 30)
        tok = _Tok()
        tok.out = out
        app.STATE["tok"] = tok
        _idle()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert app.memory_once() == "wrote"
        run = cond.memory.writer.runs[0]
        assert run["answer"] == form and run["none"] == (form != "text"), (out, run)
        assert run.get("cut") == cut, (out, run)
        if cut is not None:
            # the turn stops at the call's function: one token here, as the fake decodes the
            # whole call from the first
            assert run["finish"] == "tool_call" and run["tokens"] == 1, run
            assert f"NONE (tool_call: {cut}, cut)" in buf.getvalue(), buf.getvalue()
        assert run["ops"] == [] and cond.notes.notes("owner") == []


def test_no_session_is_said_once():
    import contextlib
    import io
    res, cond = _setup()
    C._turn(tokens(90, seed=95), 30, session="other")
    _idle()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for _ in range(3):
            got = app.memory_once()
            app._writer_look(got)
    assert got == "no_session" and cond.memory.writer.stats["last_look"] == "no_session"
    text = buf.getvalue()
    assert text.count("[memory] writer look: no session") == 1, text
    assert "writer look: nothing" not in text, text


class _BanTok(_Tok):
    """A tokenizer whose `<tool_call>` is the id `ban`."""

    ban = None

    def convert_tokens_to_ids(self, t):
        return self.ban if t == "<tool_call>" else None


def _spy():
    """generate_stream wrapped: each call's prompt, penalty state and yielded ids."""
    seen = []
    real = app.generate_stream

    def spy(prompt, *a, **kw):
        seen.append({"prompt": prompt.tolist(), "pen": kw.get("pen"), "ids": []})
        for t in real(prompt, *a, **kw):
            seen[-1]["ids"].append(int(t))
            yield t
    app.generate_stream = spy
    return seen, real


def test_the_tool_call_token_is_banned():
    from server.conductor import memwriter as mw
    assert mw.ban_ids(C._Tok()) == {} and mw.ban_ids(_BanTok()) == {}
    t = _BanTok()
    t.ban = 7
    assert mw.ban_ids(t) == {7: -100.0}
    res, cond = _setup()
    C._turn(tokens(90, seed=96), 30)
    seen, real = _spy()
    try:
        app.STATE["tok"] = _Tok()
        _idle()
        assert app.memory_once() == "wrote"
        assert seen[-1]["pen"] is None and cond.memory.writer.runs[0]["ban"] == "unavailable"
        first = seen[-1]["ids"][0]
        cond.memory.writer.by_conv["ses"]["done_at"] = 0
        tok = _BanTok()
        tok.ban = first
        app.STATE["tok"] = tok
        _idle()
        assert app.memory_once() == "wrote"
        pen = seen[-1]["pen"]
        assert pen is not None and pen.spec.bias == {first: -100.0}, pen
        assert first not in seen[-1]["ids"], (first, seen[-1]["ids"])
        assert cond.memory.writer.runs[0]["ban"] == [first]
        # the switch off: nothing banned
        v, st = cond.store.get("settings")
        st["memory"]["notes"]["writer"]["ban_tool_call"] = False
        cond.store.put("settings", None, st, note="test", author="test", base=v)
        cond.memory.writer.by_conv["ses"]["done_at"] = 0
        _idle()
        assert app.memory_once() == "wrote"
        assert seen[-1]["pen"] is None and seen[-1]["ids"][0] == first
        assert cond.memory.writer.runs[0]["ban"] is None
    finally:
        app.generate_stream = real


def test_the_prefill_starts_the_answer():
    for prefill, out, form in (("Notes:\n", BLOCK, "blocks"), ("NONE", "", "none")):
        res, cond = _setup()
        v, st = cond.store.get("settings")
        st["memory"]["notes"]["writer"]["prefill"] = prefill
        cond.store.put("settings", None, st, note="test", author="test", base=v)
        C._turn(tokens(90, seed=97), 30)
        seen, real = _spy()
        try:
            tok = _Tok()
            tok.out = out
            app.STATE["tok"] = tok
            _idle()
            assert app.memory_once() == "wrote"
        finally:
            app.generate_stream = real
        pre = C._Tok()(prefill).input_ids
        assert seen[-1]["prompt"][-len(pre):] == pre
        run = cond.memory.writer.runs[0]
        assert run["prefill"] == len(pre) and run["answer"] == form, run
        assert len(cond.notes.notes("owner")) == (1 if form == "blocks" else 0)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
