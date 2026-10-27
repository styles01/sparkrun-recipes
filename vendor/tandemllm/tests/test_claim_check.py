"""E2, the claim check (2026-10-02): a coding turn that ends on a done claim, checked by Qwen
against its own tool evidence before the end token reaches the client.

The matrix (the orchestrator's brief and the design note "Coding package: how programming work
must happen", section 3):

  1  the trigger is structural: a coding class, tools offered, a non-empty answer with no tool
     call at the end token, a write in the session; nothing else fires it, and once a request
  2  Q1 then Q2, each answer to its note; supported, no claim and answers below their thresholds
     take the end token
  3  shadow journals what act would do and takes the end token; act continues hidden
  4  the caps: two acts a request, six a session hour, `per_session` notes, then
     claim_check_exhausted; a request with logprobs is never asked
  5  fail-open: a failing question, a letter that is not one token, no room
  6  identity: E2 off is the release's output byte for byte (and so is a turn it does not fire
     on); in shadow, and in act on a supported claim, the client gets what it gets without it on
     every verify path, and the engine is back where it was after each question
  7  act on the served loop: the note is forwarded hidden, the end token never reaches the
     client, the KV holds the context, and the answer goes on as a fresh decode of it would
  8  the outcome label: fixed, ran_no_fix, no_run, after `label_calls` calls or a new claim
  9  the replay of two Mario sessions of 2026-10-02 (tests/fixtures/claim_check_mario.json): the
     check fires on both final messages, the twin and the `since` list are the transcripts', and
     a scripted Qwen answer gives the matching note
 10  the policy texts on the real tokenizer: every letter one token, the turns' sizes

Run: python tests/test_claim_check.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402
import yaml  # noqa: E402

from engine.spec import ThinkBudget  # noqa: E402
from server import app  # noqa: E402
from server.conductor import incontext  # noqa: E402
from server.conductor.monitor import (  # noqa: E402
    FinishEvent, Journal, Monitor, Policy, RequestEvent, claimcheck,
)
from server.conductor.monitor import policy as policy_mod  # noqa: E402
from server.conductor.monitor.policy import CONTINUE_HIDDEN, PASS  # noqa: E402
from test_app_loop import Req, serve  # noqa: E402
from test_serve_identity import Oracle  # noqa: E402
from test_supervisor_frame import MonTok, _fields, quiet  # noqa: E402
from test_window_edge import FixedDrafter  # noqa: E402

YAML = yaml.safe_load(open(policy_mod.DEFAULT_PATH))
E2 = YAML["tasks"]["E2"]
TOOLS = [{"type": "function", "function": {"name": n}} for n in ("write", "bash", "read")]
WROTE = [{"role": "user", "content": "make a game"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "c1", "type": "function",
              "function": {"name": "write", "arguments": '{"filePath": "g.html"}'}}]},
         {"role": "tool", "tool_call_id": "c1", "content": "Wrote file successfully."}]
CLAIM = "Done. Verified in the browser: everything works."
# the served loop's tests: the random model reads MonTok's one token a character, so the texts
# are short enough for its 1,024-row context
SHORT = dict(E2, claim_question="Q1 claim?", evidence_question="Q2 {since}?",
             claim_options={"claim": "c", "no_claim": "n"},
             evidence_options={k: k[:2] for k in claimcheck.ANSWERS},
             notes={k: f"note {k[:6]}" for k in claimcheck.NOTED})
PROMPT = [2, 40, 17, 66, 30]               # thinking off: the answer is everything written


def _policy(task=None, mode="act", caps=None, **over) -> Policy:
    d = copy.deepcopy(YAML)
    d["tasks"] = {"E2": dict(task or E2, mode=mode, **over)}
    if caps:
        d["caps"] = dict(d["caps"], **caps)
    return Policy(d)


def _monitor(mode="on", **kw) -> Monitor:
    return Monitor(mode, _policy(**kw), MonTok(), Journal(echo=False))


def _begin(mon, cls="code_edit", messages=WROTE, tools=TOOLS, n=1, constrained=False,
           tool_choice="auto", rid="r1", conv="s", now=None):
    req = RequestEvent(rid, messages=messages, tools=list(tools), n=n, max_tokens=64,
                       constrained=constrained, tool_choice=tool_choice, cls=cls)
    sup = mon.begin(mon.mode_for(None, n), req, [5, 6, 7], conv, now=now)
    sup.can_ask = True
    return sup


def _finish(sup, answer=CLAIM, inside=False, room=1 << 20):
    return sup.at_finish(FinishEvent("eos", inside, answer, 0, 40, pos=100), room)


def _asker(claim=("claim", 0.99), evidence=("test_faked", 0.9), log=None, fail=None):
    """A scripted Qwen: the option it names at the probability it names, the rest shared out
    (Q1 has two letters, Q2 five)."""

    def ask(ids, letters):
        text = MonTok().decode(ids, skip_special_tokens=False)
        if log is not None:
            log.append(text)
        if fail is not None:
            raise fail
        label, p = claim if len(letters) == 2 else evidence
        opts = list(SHORT["claim_options"]) if len(letters) == 2 else list(claimcheck.ANSWERS)
        rest = (1.0 - p) / (len(opts) - 1)
        return [torch.log(torch.tensor(p if o == label else rest)).item() for o in opts]
    return ask


# ------------------------------------------------------------------ 1: the trigger

def test_the_trigger_fires_on_a_coding_stop_after_a_write():
    mon = _monitor()
    sup = _begin(mon)
    act = _finish(sup)
    assert act is not None and act.kind == CONTINUE_HIDDEN and act.detail["ask"], act
    assert act.ids == [] and act.visible == [], "nothing to forward until Qwen answered"
    assert act.detail["twin"] == "no_run_after_last_write" and act.detail["writes"] == 1
    assert act.detail["since"] == "none" and act.detail["cls"] == "code_edit"


def test_nothing_else_fires_it():
    cases = {
        "a class that is not coding": dict(cls="chat"),
        "no class (no router)": dict(cls=""),
        "no write in the session": dict(messages=[{"role": "user", "content": "hi"}]),
        "no tools offered": dict(tools=()),
        "tool_choice none": dict(tool_choice="none"),
        "a constrained request": dict(constrained=True),
        "n > 1": dict(n=2),
    }
    for why, kw in cases.items():
        mon = _monitor()
        assert _finish(_begin(mon, **kw)) is None, why
        assert mon.journal.events() == [], why
    for why, answer, inside in (("an empty answer (E1's)", "  ", False),
                                ("the end token inside the reasoning", CLAIM, True),
                                ("an answer with a tool call",
                                 CLAIM + "\n<tool_call>\n<function=bash>\n<parameter=command>ls"
                                 "</parameter>\n</function>\n</tool_call>", False)):
        mon = _monitor()
        assert _finish(_begin(mon), answer=answer, inside=inside) is None, why
    # a write the session recorded counts when a compaction took it out of the history
    mon = _monitor()
    sup = _begin(mon, messages=[{"role": "user", "content": "summary"}])
    sup.session.record_call("edit", "{}", 1.0)
    assert _finish(sup) is not None
    # once a request: the next end token of the same request is not asked again
    mon = _monitor()
    sup = _begin(mon)
    assert _finish(sup) is not None
    sup.settle_claim(_asker(claim=("no_claim", 0.9)))
    assert _finish(sup) is None


# ------------------------------------------------------------------ 2: the questions

def test_each_answer_gets_its_note_and_q2_names_the_calls_since_the_write():
    for ans in claimcheck.NOTED:
        mon = _monitor()
        msgs = WROTE + [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "bash",
                                                              "arguments": '{"command": "x"}'}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "ok"}]
        sup = _begin(mon, messages=msgs)
        assert _finish(sup) is not None
        log = []
        act = sup.settle_claim(_asker(evidence=(ans, 0.9), log=log))
        assert act is not None and act.kind == CONTINUE_HIDDEN, ans
        assert act.text == E2["notes"][ans] and act.reason == f"claim_check({ans})"
        hidden = YAML["templates"]["hidden_turn"].replace("{text}", E2["notes"][ans])
        assert act.ids == MonTok().encode(hidden) and act.visible == [], "a hidden user turn"
        assert len(log) == 2 and E2["claim_question"] in log[0]
        assert "bash; 1 calls" in log[1] and "A: supported - " in log[1], log[1][-400:]
        ev = mon.journal.actions()[-1]
        assert ev["task"] == "E2" and ev["tier"] == 1 and ev["mode"] == "act", ev
        d = ev["detail"]
        assert d["answer"] == ans and d["claim"]["label"] == "claim"
        assert d["evidence"]["label"] == ans and d["twin"] == "ran", d
        assert sup.acts == 1 and sup.hidden_tokens == len(act.ids)


def test_supported_no_claim_and_below_threshold_take_the_end_token():
    for asker, why in ((_asker(evidence=("supported", 0.95)), "supported"),
                       (_asker(claim=("no_claim", 0.9)), "no_claim"),
                       (_asker(claim=("claim", 0.7)), "claim_below"),
                       (_asker(evidence=("test_faked", 0.5)), "evidence_below")):
        mon = _monitor()
        sup = _begin(mon)
        assert _finish(sup) is not None
        assert sup.settle_claim(asker) is None, why
        ev = mon.journal.events()[-1]
        assert ev["decision"] == PASS and ev["reason"] == f"claim_check({why})", ev
        assert not sup.acted and sup.acts == 0


# ------------------------------------------------------------------ 3: shadow and act

def test_shadow_journals_the_note_and_acts_never():
    mon = _monitor(mode="shadow")
    sup = _begin(mon)
    act = _finish(sup)
    assert act is not None and act.detail["ask"], "shadow asks too: its answers are journaled"
    assert sup.settle_claim(_asker()) is None
    ev = mon.journal.events()[-1]
    assert ev["mode"] == "shadow" and ev["decision"] == CONTINUE_HIDDEN, ev
    assert ev["text"] == E2["notes"]["test_faked"] and mon.journal.actions() == []
    assert sup.acts == 0 and not sup.acted and sup.shadows == 1
    # the task's own shadow under an acting server is the same
    mon = Monitor("on", _policy(mode="shadow"), MonTok(), Journal(echo=False))
    sup = _begin(mon)
    assert _finish(sup) is not None and sup.settle_claim(_asker()) is None
    assert mon.journal.events()[-1]["mode"] == "shadow"


# ------------------------------------------------------------------ 4: the caps

def test_the_caps():
    # two acts this request already: not asked, journaled as a cap
    mon = _monitor()
    sup = _begin(mon)
    sup.acts = 2
    assert _finish(sup) is None
    ev = mon.journal.events()[-1]
    assert ev["decision"] == PASS and ev["cap"] == "per_request", ev
    # six acts in the session's hour
    mon = _monitor(caps={"acts_per_request": 10})
    sup = _begin(mon)
    for _ in range(6):
        sup.session.record_act(time.time())
    assert _finish(sup) is None and mon.journal.events()[-1]["cap"] == "per_session_hour"
    # `per_session` notes, then claim_check_exhausted and the message goes out as written
    mon = _monitor(caps={"acts_per_session_hour": 100})
    for i in range(3):
        sup = _begin(mon, rid=f"r{i}")
        act = _finish(sup)
        if i < 2:
            assert sup.settle_claim(_asker()) is not None, i
        else:
            assert act is None
            ev = mon.journal.events()[-1]
            assert ev["decision"] == PASS and ev["reason"] == "claim_check_exhausted", ev
    # a request that records logprobs is never asked
    mon = _monitor()
    sup = _begin(mon)
    sup.can_ask = False
    assert _finish(sup) is None
    assert mon.journal.events()[-1]["reason"] == "claim_check(not_asked)"


# ------------------------------------------------------------------ 5: fail-open

def test_fail_open():
    for fail, why in ((incontext.ContextError("no_room"), "no_room"),
                      (RuntimeError("boom"), "RuntimeError")):
        mon = _monitor()
        sup = _begin(mon)
        assert _finish(sup) is not None
        assert sup.settle_claim(_asker(fail=fail)) is None
        ev = mon.journal.events()[-1]
        assert ev["decision"] == PASS and ev["reason"] == f"claim_check(error:{why})", ev

    class TwoTok(MonTok):
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            r = super().__call__(text, add_special_tokens, return_tensors)
            if text in ("A", "B"):
                r.input_ids = r.input_ids * 2
            return r

    mon = Monitor("on", _policy(), TwoTok(), Journal(echo=False))
    sup = _begin(mon)
    assert _finish(sup) is not None and sup.settle_claim(_asker()) is None
    assert mon.journal.events()[-1]["reason"] == "claim_check(error:letter_not_one_token)"
    # the note does not fit: PASS, no_room
    mon = _monitor()
    sup = _begin(mon)
    assert _finish(sup, room=10) is not None
    assert sup.settle_claim(_asker()) is None and mon.journal.events()[-1]["cap"] == "no_room"
    assert sup.session.tasks["E2"]["notes"] == 0, "a note that was not sent does not count"


# ------------------------------------------------------------------ 6, 7: the served loop

def _natural(n=30):
    serve(None, max_len=1024)
    app.STATE["tok"] = MonTok()
    with quiet():
        return list(app.generate_stream(torch.tensor(PROMPT), n, set(), ThinkBudget(MonTok())))


def _stop_after(out, lo=4):
    """An end token the model draws after a non-empty answer: the first token from `lo` on that
    did not occur before it."""
    i = next(i for i in range(lo, len(out)) if out[i] not in out[:i])
    return i, out[i]


def _state(eng):
    n = eng.kv.length
    return (n, eng.state.S.clone(), eng.state.conv.clone(), eng.kv.k[..., :n, :].clone(),
            eng.kv.v[..., :n, :].clone())


def _same(a, b) -> bool:
    return a[0] == b[0] and all(torch.equal(x, y) for x, y in zip(a[1:], b[1:]))


class Spy:
    """`app._ask_in_place` as served, with the engine's state compared before and after, and
    Qwen's answer scripted on top of the real forward."""

    def __init__(self, asker):
        self.real, self.asker, self.calls, self.restored = app._ask_in_place, asker, 0, []

    def __call__(self, eng, pos, device, ids, letters):
        before = _state(eng)
        logits = self.real(eng, pos, device, ids, letters)
        assert len(logits) == len(letters) and all(isinstance(x, float) for x in logits)
        self.restored.append(_same(before, _state(eng)))
        self.calls += 1
        return self.asker(ids, letters)

    def __enter__(self):
        app._ask_in_place = self
        return self

    def __exit__(self, *exc):
        app._ask_in_place = self.real


def _loop(make, tree, stop, mode, task_mode, asker=None):
    eng = serve(make(), tree=tree, max_len=1024)
    app.STATE["tok"] = MonTok()
    mon = Monitor(mode, _policy(task=SHORT, mode=task_mode), MonTok(), Journal(echo=False))
    sup = _begin(mon)
    with Spy(asker or _asker()) as spy, quiet():
        out = list(app.generate_stream(torch.tensor(PROMPT), 40, {stop}, ThinkBudget(MonTok()),
                                       sup=sup))
    return out, list(app.STATE["last_ctx"]), eng, mon, sup, spy


def test_identity_off_shadow_and_a_supported_claim_on_every_path():
    """E2 off: the release's tokens, context and engine state. Shadow, and act on a supported
    claim: the same tokens and context, the questions asked (on the real forward) and the engine
    put back exactly after each one, on the chain, the tree and the single-token step, with the
    end token inside an accepted block and at its end."""
    ref = _natural()
    inside = 0
    for lo in (4, 7):
        at, stop = _stop_after(ref, lo)
        inside += at % 6 not in (0, 1)
        for name, make, tree in (
                ("chain", lambda: Oracle(ref, len(PROMPT), 5, False), False),
                ("tree", lambda: Oracle(ref, len(PROMPT), 5, True), True),
                ("single", lambda: None, False)):
            off = _loop(make, tree, stop, "on", "off")
            assert off[0][-1] == stop and off[5].calls == 0, (name, off[0])
            assert off[3].journal.events() == [], "off runs nothing"
            for mode, task_mode, asker, want in (
                    ("shadow", "act", _asker(), "would"),
                    ("on", "shadow", _asker(), "would"),
                    ("on", "act", _asker(evidence=("supported", 0.95)), "supported")):
                got = _loop(make, tree, stop, mode, task_mode, asker)
                out, ctx, eng, mon, sup, spy = got
                assert out == off[0] and ctx == off[1], (name, lo, mode, task_mode, out, off[0])
                assert spy.calls == 2 and all(spy.restored), (name, mode, spy.restored)
                assert not sup.acted and mon.journal.actions() == [], (name, mode)
                ev = [e for e in mon.journal.events() if e.get("task") == "E2"][-1]
                assert (ev["decision"] == CONTINUE_HIDDEN and ev["mode"] == "shadow") \
                    if want == "would" else ev["reason"] == "claim_check(supported)", ev
                assert eng.kv.length == len(ctx) - 1, "the KV holds the context but its end"
    assert inside >= 1, "an end token inside an accepted block"


def test_act_forwards_the_note_hidden_and_decodes_on():
    ref = _natural()
    at, stop = _stop_after(ref, 4)
    note = SHORT["notes"]["test_faked"]
    hidden = MonTok().encode(YAML["templates"]["hidden_turn"].replace("{text}", note))
    for name, make, tree in (
            ("chain", lambda: Oracle(ref, len(PROMPT), 5, False), False),
            ("tree", lambda: Oracle(ref, len(PROMPT), 5, True), True),
            ("single", lambda: None, False)):
        out, ctx, eng, mon, sup, spy = _loop(make, tree, stop, "on", "act")
        assert spy.calls == 2 and all(spy.restored) and sup.acted, name
        k = next(i for i in range(len(ctx)) if ctx[i:i + len(hidden)] == hidden)
        assert k == len(PROMPT) + at, (name, k, at)
        assert out[:at] == ref[:at] and stop not in out[:at + 1], "the end token was not sent"
        assert MonTok().decode(out).find("note") < 0, "the note never reaches the client"
        # the answer after the note: a fresh greedy decode of the context up to it
        serve(None, max_len=1024)
        app.STATE["tok"] = MonTok()
        with quiet():
            fresh = list(app.generate_stream(torch.tensor(ctx[:k + len(hidden)]), 8, set(),
                                             ThinkBudget(MonTok())))
        assert out[at:at + 8] == fresh[:len(out[at:at + 8])], (name, out[at:], fresh)
        acts = mon.journal.actions()
        assert [a["reason"] for a in acts] == ["claim_check(test_faked)"], acts


def test_over_the_handler_off_is_byte_identical_and_act_hides_the_note():
    # what the model writes for this chat, raw: the end token is one it draws after an answer
    prompt = MonTok().apply_chat_template(WROTE, enable_thinking=False)["input_ids"][0].tolist()
    serve(FixedDrafter(97, 5, tree_mode=False), max_len=1024)
    app.STATE["tok"] = MonTok()
    with quiet():
        ref = list(app.generate_stream(torch.tensor(prompt), 30, set(), ThinkBudget(MonTok())))
    _, stop = _stop_after(ref, 4)
    real = app.conductor.RequestEvent

    def coding(*a, **kw):
        ev = real(*a, **kw)
        ev.cls = "code_edit"
        return ev

    def ask(task_mode, stream, asker=None, task=None):
        serve(FixedDrafter(97, 5, tree_mode=False), max_len=1024)
        app.STATE["tok"] = MonTok(ends=[stop])
        app.STATE["cfg_eos"] = stop
        mon = app.STATE["monitor"] = Monitor(
            "on", _policy(task=SHORT, mode=task_mode) if task is None else task, MonTok(),
            Journal(echo=False))
        app.conductor.RequestEvent = coding
        try:
            with Spy(asker or _asker()) as spy, quiet():
                _, raw = Req("/v1/chat/completions", {
                    "messages": WROTE, "tools": TOOLS, "stream": stream, "max_tokens": 30,
                    "chat_template_kwargs": {"enable_thinking": False}}).response()
        finally:
            app.conductor.RequestEvent = real
        return mon, spy, raw

    def strip(raw, stream):
        """The response without its id and times."""
        if not stream:
            d = json.loads(raw)
            return json.dumps({"choices": d["choices"], "usage": {
                k: v for k, v in d["usage"].items() if "ms" not in k and "time" not in k}})
        return [json.loads(ln[6:])["choices"] for ln in raw.splitlines()
                if ln.startswith("data: {")]

    for stream in (True, False):
        base = Policy(dict(copy.deepcopy(YAML), tasks={}))
        _, _, none = ask("off", stream, task=base)
        mon, spy, off = ask("off", stream)
        assert strip(off, stream) == strip(none, stream) and spy.calls == 0, stream
        mon, spy, shadow = ask("shadow", stream)
        assert strip(shadow, stream) == strip(none, stream) and spy.calls == 2, stream
        assert mon.journal.events()[0]["decision"] == CONTINUE_HIDDEN
        mon, spy, act = ask("act", stream)
        content, _ = _fields(act, stream)
        assert "note" not in content and spy.calls == 2 and all(spy.restored), stream
        assert _hidden_in_ctx(SHORT["notes"]["test_faked"]), "the note was forwarded"
        assert [a["reason"] for a in mon.journal.actions()] == ["claim_check(test_faked)"]
        end = mon.journal.events()[-1]
        assert end["decision"] == "OUTCOME" and end["acts"] == 1, end


def _hidden_in_ctx(text: str) -> bool:
    ids = MonTok().encode(text)
    ctx = list(app.STATE["last_ctx"])
    return any(ctx[i:i + len(ids)] == ids for i in range(len(ctx) - len(ids) + 1))


def test_the_gpu_check_tool_on_the_cpu_reference_path():
    """tools/claim_identity.py, the hold's check on the served kernels, run here on the random
    model's reference path: the questions asked at each moment SP2 can meet (a partial accept and
    its rollback, a full accept, a tree commit, a decode step) leave every later logit and state
    bit-identical."""
    from test_window_edge import _engine
    from tools import claim_identity as ci
    eng = _engine(1024)
    ask = ci.asker([32, 33, 34, 35, 36])
    with quiet():
        for moment in ci.MOMENTS:
            bad, det, got = ci.check(eng, 64, moment, ask)
            assert not det, (moment, det)
            assert not bad, (moment, bad)
            assert len(got["asked"]["q1"]) == 2 and len(got["asked"]["q2"]) == 5


# ------------------------------------------------------------------ 8: the outcome label

def test_the_outcome_label():
    mon = _monitor(caps={"acts_per_session_hour": 100})
    sup = _begin(mon)
    assert _finish(sup) is not None and sup.settle_claim(_asker()) is not None
    t = time.time()
    sup.session.record_call("bash", "{}", t + 1)
    _begin(mon, rid="r2", now=t + 2)                 # one call after the note: not yet
    assert not [e for e in mon.journal.events() if e.get("sp") == "label"]
    for i in range(7):
        sup.session.record_call("write" if i == 3 else "read", "{}", t + 3 + i)
    _begin(mon, rid="r3", now=t + 60)
    lab = [e for e in mon.journal.events() if e.get("sp") == "label"]
    assert len(lab) == 1 and lab[0]["request_id"] == "r1", lab
    assert lab[0]["reason"] == "claim_outcome(fixed)" and lab[0]["detail"]["run"], lab
    assert lab[0]["detail"]["calls"] == 8 and lab[0]["detail"]["answer"] == "test_faked"
    # a run, no fix, and the claim again: labelled at the next request
    mon = _monitor(mode="shadow")
    sup = _begin(mon)
    assert _finish(sup) is not None and sup.settle_claim(_asker()) is None
    sup.session.record_call("bash", "{}", time.time() + 1)
    sup = _begin(mon, rid="r2")
    assert _finish(sup) is not None                  # the claim again
    _begin(mon, rid="r3")
    lab = [e for e in mon.journal.events() if e.get("sp") == "label"]
    assert lab and lab[0]["reason"] == "claim_outcome(ran_no_fix)", lab
    assert lab[0]["detail"]["reclaimed"] and lab[0]["mode"] == "shadow"


# ------------------------------------------------------------------ 9: the Mario replay

def test_the_mario_sessions_replay():
    """The two Mario sessions of 2026-10-02 (anonymised from their SRV-137 transcripts): the
    final request of each, as opencode sent it, ends on "Done ... Verified ...". The check must
    fire on both, with the calls since the write as the transcripts have them; Qwen's answers
    are scripted here (the questions on the 27B are the live check), and each gets its note."""
    fx = json.load(open(os.path.join(HERE, "fixtures", "claim_check_mario.json")))
    want = {0: "test_faked", 1: "surprise_unexplained"}
    assert len(fx["sessions"]) == 2
    for i, s in enumerate(fx["sessions"]):
        assert s["final"].startswith("Done.") and "erified" in s["final"], s["final"][:80]
        tools = [{"type": "function", "function": {"name": n}} for n in s["tools"]]
        mon = _monitor()
        sup = _begin(mon, cls=s["class"], messages=s["messages"], tools=tools)
        act = _finish(sup, answer=s["final"])
        assert act is not None and act.detail["ask"], (i, "the claim check must fire")
        assert act.detail["writes"] == 1 and act.detail["twin"] == "ran", act.detail
        since = act.detail["since"]
        calls = [c["function"]["name"] for m in s["messages"] for c in m.get("tool_calls") or []]
        after = calls[calls.index("write") + 1:]
        assert since == claimcheck.since_text(after) and "chrome-devtools_evaluate_script" in since
        assert after[0] == "bash", after
        log = []
        got = sup.settle_claim(_asker(evidence=(want[i], 0.9), log=log))
        assert got is not None and got.text == E2["notes"][want[i]], i
        assert since in log[1], "Q2 names the calls since the write"
        print(f"    session {i + 1}: fired, {since}; note {want[i]} ({len(got.ids)} tokens "
              f"on the test tokenizer)")


# ------------------------------------------------------------------ 10: the real tokenizer

def test_the_texts_on_the_real_tokenizer():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("  (skipped: no tokenizers)")
        return
    snap = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots")
    tj = next((os.path.join(snap, d, "tokenizer.json") for d in sorted(os.listdir(snap))
               if os.path.exists(os.path.join(snap, d, "tokenizer.json"))), None) \
        if os.path.isdir(snap) else None
    if tj is None:
        print("  (skipped: no Qwen3.8 tokenizer in the HF cache)")
        return
    tk = Tokenizer.from_file(tj)

    class Tok:
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            return type("R", (), {"input_ids": tk.encode(text, add_special_tokens=False).ids})()

    tok = Tok()
    assert len(incontext.letter_ids(tok, list("ABCDE"))) == 5
    turn = YAML["templates"]["hidden_turn"]
    fx = json.load(open(os.path.join(HERE, "fixtures", "claim_check_mario.json")))
    since = claimcheck.since_text(claimcheck.evidence(
        fx["sessions"][0]["messages"], E2["write_tools"], E2["run_tools"])["since"])
    q1 = E2["claim_question"] + "\n\n" + incontext.listing(E2["claim_options"])[0]
    q2 = E2["evidence_question"].replace("{since}", since) + "\n\n" + \
        incontext.listing(E2["evidence_options"])[0]
    sizes = {"Q1": len(tok(turn.replace("{text}", q1)).input_ids),
             "Q2": len(tok(turn.replace("{text}", q2)).input_ids)}
    for k, v in E2["notes"].items():
        sizes[k] = len(tok(turn.replace("{text}", v)).input_ids)
    assert all(v < 400 for v in sizes.values()), sizes
    print(f"    tokens with the hidden turn: {sizes}")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
