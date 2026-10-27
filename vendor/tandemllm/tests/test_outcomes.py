"""Turn outcomes, Tier 0 (J1 steps 2 and 3, the plan of 2026-10-02, section 2): every request
whose newest message is a tool result gets one outcome from the facts of its snapshot, journaled
as `kind: outcome` with its evidence (shadow); in act the outcome sets the effort of the turn
(no visible change and regression xhigh, error high, never below the class's, held for
`raise_turns` tool turns or until a progress) and adds its note once (the E-a note with the
snapshot's digest), replayed at the same token position, under the caps.

The test matrix:

  1  each label's Tier 0 facts, and none of them from a read or from words
                                                        test_each_label
  2  one outcome a request: the first label of the config's order, progress only when nothing
     else matched, none with nothing to read; the order is the document's
                                                        test_one_outcome_in_the_documents_order
  3  shadow: a `kind: outcome` row per tool turn with its evidence, the route row's
     extra.outcome, the effort and the note it would set in row.shadow; the prompt ids and the
     effort are those of mode off                       test_shadow_changes_nothing_served
  4  the two Mario sessions of 2026-10-02 replayed in shadow: their outcome rows (the faked test
     found in both)                                     test_the_mario_sessions
  5  act: the effort of no change, regression and error, never below the class's, held two
     turns, dropped after a progress                    test_act_raises_the_effort_and_falls_back
  6  act: the E-a note goes in once per signature, replays at the same position on the next
     request (the prompt up to it is the previous turn's), counts as an act (the monitor adopts
     it), and the caps drop the note but keep the effort
                                                        test_act_note_replay_and_caps
  7  the eval counts the outcomes per session (tools/tandem_eval.py)
                                                        test_the_eval_counts_outcomes
  8  the schemas: the profile's switches and the router's section validate; anything else is
     refused                                            test_the_schemas

Run: python tests/test_outcomes.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import QTok  # noqa: E402
from test_snapshots import S, TOOLS, body, fresh, mario_sessions, plan, png, served  # noqa: E402

from server import app  # noqa: E402
from server.conductor import outcomes, schema, signals  # noqa: E402
from server.conductor.monitor.session import SessionState  # noqa: E402


def observe(msgs, ev=None, cfg=None, now=100.0, system=None):
    ev = ev or outcomes.Evidence(0.0)
    cfg = cfg or schema.outcomes({})
    b = body(msgs)
    if system is not None:
        b["messages"][0] = {"role": "system", "content": system}
    snap = outcomes.observe(b, signals.shape(b), ev, cfg, now=now)
    label, why = outcomes.classify(snap, ev, cfg)
    outcomes.settle(ev, label, why)
    return label, why, snap, ev


def drive(steps, cfg=None, system=None):
    """A session through `steps` (each a list of calls and an optional shot): the outcomes."""
    s, ev, out = S(), outcomes.Evidence(0.0), []
    for i, st in enumerate(steps):
        calls, shot = st if isinstance(st, tuple) else (st, None)
        msgs = s.step(*calls, shot=shot)
        out.append(observe(msgs, ev, cfg, now=100.0 + i, system=system)[:2])
    return out, ev


W = ("write", {"filePath": "/work/game/game.js", "content": "let a = 1"},
     "Wrote file successfully.")
E = ("edit", {"filePath": "/work/game/game.js", "oldString": "1", "newString": "2"},
     "Edit applied successfully.")
SHOT = ("chrome-devtools_take_screenshot", {}, "Took a screenshot.")


def bash(cmd, out):
    return ("bash", {"command": cmd}, out)


def labels(out):
    return [x[0] for x in out]


# ------------------------------------------------------------------ 1

def test_each_label():
    a, b = png(64, 32, b"a"), png(64, 32, b"b")
    # no visible change: the same screenshot, the same output, the same state after a write
    out, _ = drive([([W, SHOT], a), ([E, SHOT], a)])
    assert labels(out) == ["none", "no_visible_change"], out
    assert out[1][1] == ["the same screenshot as before the last write"]
    out, _ = drive([([W, SHOT], a), ([E, SHOT], b)])
    assert labels(out)[1] == "progress", out
    out, _ = drive([[bash("node run.js", "score 10")], [E], [bash("node run.js", "score 10")]])
    assert labels(out) == ["none", "none", "no_visible_change"], out
    out, _ = drive([[bash("node run.js", "score 10")], [bash("node run.js", "score 10")]])
    assert labels(out)[1] == "none", "the same output without a write in between is no finding"
    st = json.dumps({"tandem_state": 1, "x": 4})
    out, _ = drive([[bash("node dump.js", st)], [E], [bash("node dump.js", st)]])
    assert labels(out)[2] == "no_visible_change" and "state" in out[2][1][0], out
    # regression: a checker number worse than the best, more failures, a state key gone
    ck = "calccheck src/c.py\nRESULT: FAIL <- (a) error {} > 1.0"
    out, _ = drive([[bash("python check.py", ck.format(1.3))], [E],
                    [bash("python check.py", ck.format(1.2))], [E],
                    [bash("python check.py", ck.format(1.4))]])
    assert labels(out) == ["none", "none", "progress", "none", "regression"], out
    out, _ = drive([[bash("python check.py", ck.format(3.0))]])
    assert labels(out) == ["low_quality"] and out[0][1] == ["gate (a) 3 > 1"], out
    out, _ = drive([[bash("npm test", "Tests: 1 failed, 4 passed")], [E],
                    [bash("npm test", "Tests: 3 failed, 2 passed")]])
    assert labels(out)[2] == "regression" and out[2][1][0] == "3 failed, was 1", out
    out, _ = drive([[bash("node d.js", json.dumps({"tandem_state": 1, "x": 1, "hp": 3}))],
                    [bash("node d.js", json.dumps({"tandem_state": 1, "x": 2}))]])
    assert labels(out)[1] == "regression" and "hp" in out[1][1][0], out
    # error: an exit code, failed tests, a console error, a traceback
    for res in ("exit code 2", '{"output": "boom", "exit_code": 1}',
                "Traceback (most recent call last):\n  File \"a.py\", line 1\nValueError: x",
                "Tests: 2 failed, 3 passed"):
        assert labels(drive([[bash("python a.py", res)]])[0]) == ["error"], res
    key = "sk-" + "Q7v" * 10
    out, _ = drive([[bash("python a.py", f"ValueError: the key {key} was refused")]])
    assert out[0][0] == "error" and key not in out[0][1][0] and "[redacted:" in out[0][1][0]
    out, _ = drive([[("chrome-devtools_list_console_messages", {},
                      "msgid=1 [error] Uncaught TypeError: x (0 args)")]])
    assert labels(out) == ["error"] and "1 console error" in out[0][1][0], out
    # a read of a file that contains the same words is not a finding
    out, _ = drive([[("read", {"filePath": "/work/a.py"},
                      "raise ValueError('x')\nTraceback (most recent call last):\nexit code 1")]])
    assert labels(out) == ["none"], out
    # low quality: far from the gate, or the same gate failing three times
    out, _ = drive([[bash("python check.py", ck.format(1.2))] for _ in range(3)])
    assert labels(out) == ["none", "none", "low_quality"], out
    # surprise: lives 0 in a state dump, a test run of 0 tests, an empty write
    out, _ = drive([[bash("node d.js", json.dumps({"tandem_state": 1, "lives": 0}))]])
    assert labels(out) == ["surprise"], out
    assert labels(drive([[bash("pytest -q", "0 passed in 0.01s")]])[0]) == ["surprise"]
    out, _ = drive([[("write", {"filePath": "/w/a.js", "content": ""}, "Wrote file successfully.")]])
    assert labels(out) == ["surprise"], out
    # stuck: no write and no better number for `turns` tool turns
    cfg = schema.outcomes({"outcomes": {"stuck": {"turns": 4, "minutes": 999}}})
    out, ev = drive([[bash("ls", f"{i}")] for i in range(6)], cfg=cfg)
    assert labels(out) == ["none", "none", "none", "stuck", "none", "none"], out
    # tool misuse: an unknown tool, a required argument missing, a call the client rejected
    for call in (("frobnicate", {"x": 1}, "ok"), ("bash", {"cmd": "ls"}, "ok"),
                 ("bash", {"command": "ls"}, "Error: invalid arguments for tool bash")):
        assert labels(drive([[call]])[0]) == ["tool_misuse"], call
    # test faked: a script that sets the state it then reads
    for script in ("() => { keys['ArrowRight'] = true; return 'moving right'; }",
                   "() => { mario.x = flagX - 40; return 'teleported'; }"):
        out, _ = drive([[("chrome-devtools_evaluate_script", {"function": script}, "ok")]])
        assert labels(out) == ["test_faked"], (script, out)
    for script in ("() => window.__state()", "() => { const s = {}; s.x = 1; return s; }",
                   "() => document.title === 'x'", "() => { let n = 0; n += 1; return n; }"):
        out, _ = drive([[("chrome-devtools_evaluate_script", {"function": script}, "ok")]])
        assert labels(out) == ["none"], (script, out)
    # environment
    out, _ = drive([[bash("curl localhost:3000", "curl: (7) Failed to connect: Connection refused")]])
    assert labels(out) == ["environment"], out
    # lost context: after a compaction two calls read none of the state files
    sysp = "You are opencode. Keep PROGRESS.yaml current; read AGENTS.md."
    ev = outcomes.Evidence(0.0)
    s = S()
    observe(s.step(bash("ls", "a")), ev, system=sysp)
    outcomes.compaction(ev)
    got = [observe(s.step(bash("ls", f"{i}")), ev, system=sysp)[0] for i in range(2)]
    assert got == ["none", "lost_context"], got
    ev = outcomes.Evidence(0.0)
    observe(s.step(bash("ls", "b")), ev, system=sysp)
    outcomes.compaction(ev)
    got = [observe(s.step(("read", {"filePath": "/w/PROGRESS.yaml"}, "next: s2")), ev,
                   system=sysp)[0], observe(s.step(bash("ls", "c")), ev, system=sysp)[0]]
    assert got == ["none", "none"], "a re-read ends the watch"


# ------------------------------------------------------------------ 2

def test_one_outcome_in_the_documents_order():
    steps = [[bash("npm test", "Tests: 1 failed, 4 passed\nexit code 1"),
              bash("curl localhost:1", "Connection refused")]]
    out, _ = drive(steps)
    assert labels(out) == ["error"], out
    cfg = schema.outcomes({"outcomes": {"order": ["environment", "error"]}})
    assert labels(drive(steps, cfg=cfg)[0]) == ["environment"]
    cfg = schema.outcomes({"outcomes": {"order": ["regression"]}})
    assert labels(drive(steps, cfg=cfg)[0]) == ["none"], "a label out of the order is not taken"
    assert labels(drive([[bash("npm test", "Tests: 4 passed")]])[0]) == ["progress"]
    assert labels(drive([[("read", {"filePath": "/w/a"}, "text")]])[0]) == ["none"]
    assert list(schema.OUTCOMES["order"]) == list(schema.OUTCOME_LABELS)


# ------------------------------------------------------------------ 3

def _rows(cond):
    got = []
    real = cond.journal.note

    def note(kind, task, **kw):
        got.append(dict(kw, kind=kind, task=task))
        return real(kind, task, **kw)
    cond.journal.note = note
    cond.router.journal = cond.journal
    return got


def test_shadow_changes_nothing_served():
    base, sh = fresh(None), fresh({"snapshots": "on", "mode": "shadow"})
    rows = _rows(sh)
    s = S()
    turns = [list(s.msgs), s.step(W), s.step(bash("npm test", "Tests: 1 failed, 4 passed")),
             s.step(E), s.step(bash("npm test", "Tests: 1 failed, 4 passed")),
             s.step(bash("npm test", "Tests: 5 passed"))]
    want = [None, "none", "error", "none", "no_visible_change", "progress"]
    for i, msgs in enumerate(turns):
        pa, ia = served(base, msgs)
        pb, ib = served(sh, msgs)
        assert ia == ib, f"turn {i}: the same prompt ids in shadow"
        assert (pa.row["effort"], pa.budget, pa.thinking_off) == \
               (pb.row["effort"], pb.budget, pb.thinking_off), i
        assert pb.extras == [] and pb.acts == []
        o = pb.row["extra"].get("outcome")
        assert (o or {}).get("label") == want[i], (i, o)
    outs = [r for r in rows if r["kind"] == "outcome"]
    assert [r["decision"] for r in outs] == want[1:], outs
    err = outs[1]
    assert err["detail"]["mode"] == "shadow" and err["detail"]["why"] == ["1 test failed"]
    assert err["detail"]["snapshot"]["tests"] == [{"failed": 1, "passed": 4, "call": "bash"}]
    assert err["session_key"] == "c:ses-snap"
    # code_edit thinks at xhigh already: error's high would not raise it; the note would go in
    assert err["detail"]["effort"] is None and err["detail"]["note"]["due"] is True
    nvc = outs[3]
    assert nvc["detail"]["note"]["chars"] > 100 and not nvc["detail"]["note"]["applied"]


def _debug_class(cond, effort="medium"):
    """code_edit at a lower effort, so an outcome's level shows."""
    _, doc = cond.store.get("profiles", "coding")
    doc["routing"]["code_edit"] = {"effort": effort}
    cond.store.put("profiles", "coding", doc, note="code_edit medium", author="t")


def test_act_raises_the_effort_and_falls_back():
    for mode, applied in (("shadow", False), ("act", True)):
        cond = fresh({"mode": mode})
        _debug_class(cond)
        s = S()
        steps = [list(s.msgs), s.step(bash("npm test", "Tests: 1 failed")),   # error -> high
                 s.step(bash("ls", "a")),                                     # held: high
                 s.step(bash("ls", "b")),                                     # back: medium
                 s.step(W, SHOT, shot=png(8, 8, b"1")),
                 s.step(E, SHOT, shot=png(8, 8, b"1")),                       # no change: xhigh
                 s.step(bash("npm test", "Tests: 2 passed")),                 # progress: medium
                 s.step(bash("ls", "c"))]
        got = []
        for msgs in steps:
            p = plan(cond, msgs)
            got.append((p.row["effort"], (p.row.get("shadow") or {}).get("outcome", {})
                        .get("effort")))
        if applied:
            assert [g[0] for g in got] == ["medium", "high", "high", "medium", "medium", "xhigh",
                                           "medium", "medium"], got
        else:
            assert [g[0] for g in got] == ["medium"] * 8, got
            assert [g[1] for g in got] == [None, "high", "high", None, None, "xhigh", None,
                                           None], got


def test_act_note_replay_and_caps():
    cond = fresh({"mode": "act"})
    s = S()
    t1 = served(cond, list(s.msgs))[1]
    t2 = served(cond, s.step(W, bash("node run.js", "score 10")))[1]
    p3, t3 = served(cond, s.step(E, bash("node run.js", "score 10")))
    assert p3.row["extra"]["outcome"]["label"] == "no_visible_change"
    assert p3.row["extra"]["outcome"]["note"]["applied"] is True and len(p3.acts) == 1
    inj = [n for n in p3.injected if n["parts"][0]["kind"] == "outcome"]
    assert len(inj) == 1 and "had no visible effect" in inj[0]["text"]
    assert "Runtime snapshot of your last tool turn" in inj[0]["text"]
    note_ids = QTok().encode(inj[0]["text"])
    at = inj[0]["at"]
    assert t3[at:at + len(note_ids)] == note_ids, "the note sits at the end of the last message"
    # the next request: its prompt up to the end of the previous one is the served prompt of
    # turn 3 (the note replayed at its place), so the resident prefix holds
    p4, t4 = served(cond, s.step(bash("ls", "x")))
    assert t4[:at + len(note_ids)] == t3[:at + len(note_ids)]
    assert p4.row["replayed"]["count"] >= 2 and not [
        n for n in p4.injected if n["parts"][0]["kind"] == "outcome"]
    # the same signature again: no second note; the monitor adopts the act of turn 3
    p5, _ = served(cond, s.step(E, bash("node run.js", "score 10")))
    assert p5.row["extra"]["outcome"]["label"] == "no_visible_change"
    assert p5.row["extra"]["outcome"]["note"]["skipped"] == "noted" and p5.acts == []
    from server.conductor.monitor import Monitor, load
    from server.conductor.monitor.observe import RequestEvent
    mon = Monitor("on", load(), QTok())
    sup = mon.begin("on", RequestEvent("r3", messages=s.msgs, tools=TOOLS), [1, 2, 3], None,
                    session_key="c:ses-snap")
    sup.adopt(p3.acts)
    assert sup.acts == 1 and sup.session.acts_last_hour(p3.acts[0] + 1) == 1
    # the caps: six acts in the session's hour (the monitor's count) keep the effort, drop the note
    cond2 = fresh({"mode": "act"})
    _debug_class(cond2)
    ms = SessionState("c:ses-cap", 0.0)
    import time as _t
    for _ in range(6):
        ms.record_act(_t.time())
    cond2.router.monitor_session = lambda key: ms if key == "c:ses-cap" else None
    s2 = S()
    plan(cond2, s2.msgs, sid="ses-cap")
    p = plan(cond2, s2.step(bash("npm test", "Tests: 1 failed")), sid="ses-cap")
    o = p.row["extra"]["outcome"]
    assert o["label"] == "error" and o["effort"]["applied"] and p.row["effort"] == "high"
    assert o["note"]["cap"] == "per_session_hour" and not o["note"]["applied"] and p.extras == []
    assert t1 and t2


# ------------------------------------------------------------------ 4

def test_the_mario_sessions():
    want_faked = 0
    for ses, msgs in mario_sessions():
        cond = fresh({"snapshots": "on", "mode": "shadow"})
        rows = _rows(cond)
        tools = [{"type": "function", "function": {"name": n, "parameters": {}}}
                 for n in ses["tools"]]
        sid = ses["name"]
        for i in range(2, len(msgs) + 1):
            if msgs[i - 1]["role"] not in ("tool", "user") or (
                    i < len(msgs) and msgs[i]["role"] in ("tool", "user")):
                continue
            b = {"model": "TandemLLM", "messages": msgs[:i], "tools": tools}
            cond.plan(b, {"x-session-id": sid}, client_kind="opencode", request_id=f"r{i}",
                      chat=True)
        seq = [r["decision"] for r in rows if r["kind"] == "outcome"]
        print(f"    {sid}: {len(seq)} tool turns: " + ", ".join(
            f"{k} {seq.count(k)}" for k in dict.fromkeys(seq)))
        assert len(seq) >= 10, seq
        want_faked += "test_faked" in seq
        assert "no_visible_change" not in seq[:2]
    assert want_faked == 2, "the keys[...] = true test is found in both sessions"


# ------------------------------------------------------------------ 7

def test_the_eval_counts_outcomes():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
    import tandem_eval as te

    class Api:
        def get(self, path):
            return 200, {"decisions": [
                {"kind": "route", "effort": "xhigh", "extra": {"outcome": {"label": "error"}}},
                {"kind": "route", "effort": "high", "extra": {"outcome": {"label": "error"}}},
                {"kind": "route", "effort": "high", "extra": {"outcome": {"label": "progress"}}},
                {"kind": "route", "effort": "high", "extra": {}}]}
    got = te.conductor_efforts(Api(), "ses")
    assert got["outcomes"] == {"error": 2, "progress": 1}, got


# ------------------------------------------------------------------ 8

def test_the_schemas():
    p = schema.seed("profiles", "coding")
    p["outcomes"] = {"snapshots": "on", "mode": "shadow"}
    assert schema.check("profiles", p, name="coding") == []
    for k, bad in (("mode", "on"), ("snapshots", "shadow"), ("judge", "act"), ("nope", "off")):
        p["outcomes"] = {k: bad}
        assert schema.check("profiles", p, name="coding"), (k, bad)
    r = schema.seed("router")
    r["outcomes"] = {"order": ["error", "regression"], "effort": {"error": "xhigh"},
                     "acting": ["error"], "notes": {"error": "x {digest}"}, "raise_turns": 1}
    assert schema.check("router", r) == []
    got = schema.outcomes(r)
    assert got["order"] == ["error", "regression"] and got["notes"]["error"] == "x {digest}"
    assert got["notes"]["regression"] == schema.OUTCOMES["notes"]["regression"]
    assert got["limits"] == schema.OUTCOMES["limits"]
    for k, bad in (("order", ["progress"]), ("effort", {"error": "max"}), ("raise_turns", 11),
                   ("acting", ["nope"])):
        r["outcomes"] = {k: bad}
        assert schema.check("router", r), (k, bad)
    assert schema.profile_outcomes({}) == schema.PROFILE_OUTCOMES


if __name__ == "__main__":
    app.STATE.setdefault("tok", QTok())
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
