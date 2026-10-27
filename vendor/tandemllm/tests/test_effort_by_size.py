"""Effort by task size (SRV-157, QA SRV-158): the class sets the ceiling, the session's size picks.

SRV-157's Gherkin, row by row:
  1. a small task thinks at high, not xhigh: a code_edit session that added 12k prompt tokens and
     touched 2 files gets the high sentence, and its journal row holds the size, the score, the
     signals and the ceiling;
  2. a growing task: the same session at 70k added tokens and 9 files is xhigh on the next turn,
     and stays xhigh on later turns whose own signals read small again, until a compaction
     starts the size over;
  3. shadow, end to end over HTTP (the real server with the fake engine and the fake decider):
     the journal holds the size and the effort it would set, and the turn runs on the class's
     effort (the xhigh sentence reaches the model);
  4. measured: the eval harness's report carries turns per effort level and the size mode per
     run and per suite and arm (tests/test_tandem_eval.py reads them; here the columns' text).
Plus: the class's effort is the ceiling (a size level above it is cut to it), a client's own
budget still wins, a profile without `thinking.by_size` (every stored version before SRV-157)
journals and changes nothing, off journals nothing, the raw path is untouched, a class without a
map keeps its effort, and the thresholds are validated (medium below small is refused).

Run: python tests/test_effort_by_size.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_conductor_router import CODE, SYSTEM, body, by_label, setup, text_of, turn  # noqa: E402

from server.conductor import schema  # noqa: E402
from tools import tandem_eval  # noqa: E402


class Rec:
    """What the server hands Plan.finish at the end of a turn."""

    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens
        self.status, self.finish_reason, self.tool_calls = 200, "tool_calls", 1


def by_size(cond, profile="coding", mode="on"):
    _, doc = cond.store.get("profiles", profile)
    doc["thinking"]["by_size"] = mode
    cond.store.put("profiles", profile, doc, note=f"by_size {mode}", author="t")


def tool_turn(msgs, i, files=(), command=None):
    """One assistant step with calls naming `files` (and a bash call), and their results."""
    calls = [{"id": f"c{i}-{k}", "type": "function", "function": {
        "name": "edit" if k % 2 else "read", "arguments": json.dumps({"filePath": f})}}
        for k, f in enumerate(files)]
    if command:
        calls.append({"id": f"c{i}-b", "type": "function", "function": {
            "name": "bash", "arguments": json.dumps({"command": command})}})
    msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
    for c in calls:
        msgs.append({"role": "tool", "tool_call_id": c["id"], "content": "ok"})
    return msgs


def step(cond, msgs, prompt_tokens, headers=None):
    """A turn and its end: plan, render, apply, then the prompt tokens the engine read."""
    plan, ids, clean = turn(cond, body(list(msgs)), headers=headers)
    plan.finish(Rec(prompt_tokens), cond.journal)
    return plan, ids, clean


# ------------------------------------------------------------------ rows 1 and 2

def test_a_small_task_thinks_at_high():
    cond, _ = setup(decider=by_label("code_edit", 0.91))
    by_size(cond)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE}]
    p1, ids1, _ = step(cond, msgs, 24_000)              # opencode's own head: not the task
    assert p1.row["effort"] == "high", p1.row["extra"]
    assert schema.HIGH_NOTE in p1.injected[0]["text"] and \
        schema.XHIGH_NOTE not in p1.injected[0]["text"]
    tool_turn(msgs, 1, ["src/lexer.ts", "src/token.ts"])
    step(cond, msgs, 36_000)
    tool_turn(msgs, 2, command="npm test")
    p3, _, _ = step(cond, msgs, 37_000)
    size = p3.row["extra"]["size"]
    assert size == {"mode": "on", "size": "small", "score": 0, "now": "small",
                    "signals": {"session_tokens": {"value": 12_000, "size": "small"},
                                "files": {"value": 2, "size": "small"},
                                "turns": {"value": 3, "size": "small"}},
                    "ceiling": "xhigh", "effort": "high", "applied": True}, size
    # high has a real cap since SRV-159: a small task thinks at most 16,384 tokens a turn
    assert p3.row["effort"] == "high" and p3.think_budget(0) == 16384


def test_a_growing_task_rises_and_does_not_fall():
    cond, _ = setup(decider=by_label("code_edit", 0.91))
    by_size(cond)
    h = {"x-session-id": "grow"}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE}]
    p1, _, _ = step(cond, msgs, 24_000, h)
    assert p1.row["effort"] == "high"
    tool_turn(msgs, 1, [f"src/m{k}.py" for k in range(9)])
    p2, _, _ = step(cond, msgs, 94_000, h)               # 9 files: large already
    sz = p2.row["extra"]["size"]
    assert (sz["size"], sz["signals"]["files"]["size"], p2.row["effort"]) == \
        ("large", "large", "xhigh"), sz
    assert schema.XHIGH_NOTE in p2.injected[0]["text"], "the new level's sentence, at this turn"
    tool_turn(msgs, 2, command="pytest -q")
    p, ids, _ = step(cond, msgs, 95_000, h)              # and 70k added tokens
    sz = p.row["extra"]["size"]
    assert (sz["size"], sz["effort"], p.row["effort"]) == ("large", "xhigh", "xhigh"), sz
    assert sz["signals"]["session_tokens"] == {"value": 70_000, "size": "large"}
    assert not p.injected and p.row["replayed"]["count"] == 2, "both notes replayed in place"
    assert text_of(ids).count(schema.HIGH_NOTE) == 1 and text_of(ids).count(schema.XHIGH_NOTE) == 1
    # a later turn whose own signals read small (the client trimmed its prompt, a short new
    # message) stays at the size the task reached
    short = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE},
             {"role": "assistant", "content": "done"},
             {"role": "user", "content": "Also rename the helper."}]
    cond.router.session("c:grow").prompt_last = 30_000
    p, _, _ = turn(cond, body(short), headers=h)
    sz = p.row["extra"]["size"]
    assert (sz["now"], sz["size"], p.row["effort"]) == ("small", "large", "xhigh"), sz
    # a compaction starts the size over: the next turn is small again
    summary = [{"role": "user", "content": "Summarize the conversation so far. " * 80}]
    p, _, _ = turn(cond, {"model": "TandemLLM", "messages": summary}, headers=h)
    assert p.row["class_id"] == "summarize" and "size" not in p.row["extra"]
    p.finish(Rec(80_000), cond.journal)
    assert cond.router.session("c:grow").prompt_last == 30_000, "a summary's prompt is not counted"
    after = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "The summary is above. Continue with the next step."}]
    p, _, _ = turn(cond, body(after), headers=h)
    assert p.row["extra"]["size"]["size"] == "small" and p.row["effort"] == "high"


def test_the_class_is_the_ceiling_and_a_class_without_a_map_keeps_its_effort():
    cond, _ = setup(decider=by_label("code_edit", 0.91))
    by_size(cond)
    _, doc = cond.store.get("profiles", "coding")
    doc["routing"]["code_edit"] = {"effort": "medium",
                                   "effort_by_size": {"small": "xhigh", "large": "xhigh"}}
    cond.store.put("profiles", "coding", doc, note="ceiling medium", author="t")
    p, _, _ = turn(cond, body([{"role": "user", "content": CODE}]))
    assert p.row["extra"]["size"]["effort"] == "medium" and p.row["effort"] == "medium"
    cond, _ = setup(decider=by_label("design_ui", 0.91))
    by_size(cond)
    p, _, _ = turn(cond, body([{"role": "user", "content": "Restyle the settings page."}]))
    sz = p.row["extra"]["size"]
    assert p.row["effort"] == "high" and "effort" not in sz and sz["size"] == "small"


def test_a_client_budget_still_wins():
    cond, _ = setup(decider=by_label("code_edit", 0.91))
    by_size(cond)
    p, _, _ = turn(cond, body([{"role": "user", "content": CODE}], max_reasoning_tokens=777))
    assert p.row["effort"] == "high" and p.row["budget_source"] == "client"
    assert p.think_budget(777) == 777


def test_shadow_off_and_the_profiles_stored_before():
    # a profile without by_size (every stored version before SRV-157): shadow
    cond, _ = setup(decider=by_label("code_edit", 0.91))
    _, doc = cond.store.get("profiles", "coding")
    del doc["thinking"]["by_size"]
    cond.store.put("profiles", "coding", doc, note="as stored before SRV-157", author="t")
    p, ids, clean = turn(cond, body([{"role": "user", "content": CODE}]))
    assert p.row["effort"] == "xhigh" and schema.XHIGH_NOTE in p.injected[0]["text"]
    assert p.row["shadow"]["size"] == {"size": "small", "score": 0, "effort": "high",
                                       "ceiling": "xhigh"}, p.row["shadow"]
    assert p.row["extra"]["size"]["applied"] is False
    # off: nothing journaled, nothing changed
    by_size(cond, mode="off")
    p, _, _ = turn(cond, body([{"role": "user", "content": CODE + " "}]),
                   headers={"x-session-id": "off"})
    assert "size" not in p.row["extra"] and p.row["effort"] == "xhigh"
    assert not (p.row.get("shadow") or {}).get("size")
    # the effort decision off: no size either
    by_size(cond)
    _, doc = cond.store.get("profiles", "coding")
    doc["decisions"]["effort"] = "off"
    cond.store.put("profiles", "coding", doc, note="effort off", author="t")
    p, _, _ = turn(cond, body([{"role": "user", "content": CODE + "  "}]),
                   headers={"x-session-id": "eoff"})
    assert "size" not in p.row["extra"] and p.row["effort"] is None


def test_the_raw_path_is_untouched():
    cond, fakes = setup(decider=by_label("code_edit", 0.91))
    by_size(cond)
    p, ids, clean = turn(cond, body([{"role": "user", "content": CODE}]),
                         headers={"X-Tandem": "raw", "x-session-id": "raw"})
    assert not p.active and ids == clean and p.row is None and not fakes["decider"].calls


def test_the_thresholds_are_validated_and_editable():
    r = schema.seed("router")
    r["effort"]["size"]["signals"]["files"]["medium"] = 1
    errs = schema.check("router", r)
    assert errs == ["$.effort.size.signals.files: medium (1) is below small (3)"], errs
    r = schema.seed("router")
    r["classes"][0]["policy"]["effort_by_size"]["small"] = "huge"
    assert any("effort_by_size.small" in e for e in schema.check("router", r))
    cond, _ = setup(decider=by_label("code_edit", 0.91))
    by_size(cond)
    _, r = cond.store.get("router")
    r["effort"]["size"]["signals"]["turns"] = {"enabled": True, "small": 1, "medium": 1}
    cond.store.put("router", None, r, note="every second turn is large", author="t")
    msgs = [{"role": "user", "content": CODE}]
    p, _, _ = turn(cond, body(msgs), headers={"x-session-id": "t"})
    assert p.row["extra"]["size"]["size"] == "medium"          # turn 1: up to medium
    tool_turn(msgs, 1, command="ls")
    p, _, _ = turn(cond, body(msgs), headers={"x-session-id": "t"})
    assert p.row["extra"]["size"]["size"] == "large" and p.row["effort"] == "xhigh"
    # a signal switched off does not vote
    r["effort"]["size"]["signals"]["turns"]["enabled"] = False
    cond.store.put("router", None, r, note="turns off", author="t")
    p, _, _ = turn(cond, body([{"role": "user", "content": CODE}]), headers={"x-session-id": "u"})
    assert "turns" not in p.row["extra"]["size"]["signals"]
    # the stored routers before SRV-157 have no size block: the built-in thresholds apply
    del r["effort"]["size"]
    assert schema.check("router", r) == []


def test_the_explain_box_shows_the_size():
    cond, _ = setup(decider=by_label("code_edit", 0.91))
    by_size(cond)
    got = cond.router.explain(CODE, "coding")
    assert got["effort"] == "high" and got["extra"]["size"]["ceiling"] == "xhigh"


# ------------------------------------------------------------------ row 3: shadow over HTTP

def test_shadow_end_to_end_runs_on_the_class_effort():
    """The real server with the fake engine and the fake decider: the coding profile's seed has
    the size in shadow. The turn runs on code_edit's xhigh (its sentence reaches the model, not the
    client), the journal holds the size and the high it would set; with by_size on, the next
    session gets high."""
    import re
    import test_conductor_e2e as e2e
    port, arb_port = e2e._port(), e2e._port()
    arb = e2e._decider(arb_port)
    srv, _ = e2e._server(port, tempfile.mkdtemp(prefix="size-e2e-"))
    try:
        e2e._wait(arb_port, arb, "/readyz")
        e2e._wait(port, srv)
        st, _, roles = e2e._call(port, "GET", "/v1/conductor/roles")
        doc = roles["doc"]
        doc["decider"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        doc["classifier"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        assert e2e._call(port, "PUT", "/v1/conductor/roles", {"doc": doc, "note": "fake"})[0] == 200
        user = "Refactor the lexer so every token carries its line. FAKE_CLASS=code_edit"
        msgs = [{"role": "system", "content": "You are opencode."},
                {"role": "user", "content": user}]
        hdr = {"X-Tandem-Profile": "coding", "x-session-id": "size-shadow"}
        text, h, usage = e2e._stream(port, msgs, headers=hdr)
        assert h["X-Tandem"] == "on" and "runtime-note" not in text and "xhigh" not in text
        row = e2e._decisions(port, session="c:size-shadow")[0]
        assert (row["class_id"], row["effort"]) == ("code_edit", "xhigh"), row
        assert row["shadow"]["size"] == {"size": "small", "score": 0, "effort": "high",
                                         "ceiling": "xhigh"}, row["shadow"]
        assert row["extra"]["size"]["mode"] == "shadow" and not row["extra"]["size"]["applied"]
        note = row["injected"][0]["text"]
        assert schema.XHIGH_NOTE in note and schema.HIGH_NOTE not in note
        seen = int(re.search(r"The question has (\d+) characters", text).group(1))
        assert seen == len(user) + len(note), "the model read the xhigh note"
        # on: the same request in a new session thinks at high
        st, _, prof = e2e._call(port, "GET", "/v1/conductor/profiles/coding")
        pdoc = prof["doc"]
        assert pdoc["thinking"]["by_size"] == "shadow", "the seed"
        pdoc["thinking"]["by_size"] = "on"
        st, _, w = e2e._call(port, "PUT", "/v1/conductor/profiles/coding",
                             {"doc": pdoc, "note": "size on"})
        assert st == 200, w
        e2e._stream(port, msgs, headers=dict(hdr, **{"x-session-id": "size-on"}))
        row = e2e._decisions(port, session="c:size-on")[0]
        assert row["effort"] == "high" and row["extra"]["size"]["applied"] is True
        assert schema.HIGH_NOTE in row["injected"][0]["text"] and not row.get("shadow")
    finally:
        for p in (srv, arb):
            if p.poll() is None:
                p.kill()


# ------------------------------------------------------------------ row 4: the report's columns

def test_the_report_columns():
    assert tandem_eval.efforts({"xhigh": 6, "high": 1}, "on") == "high 1, xhigh 6 (size on)"
    assert tandem_eval.efforts(None) == "—"
    rows = [{"suite": "minicalc", "arm": "tandem", "error": None, "passed": True,
             "gates": {"total": 1, "passed": 1}, "minutes_to_pass": 5.0, "wall_minutes": 5.5,
             "tokens": None, "turns": 7, "continuations": {"needed": 0},
             "pristine": {"ok": True}, "empty_stops": 0,
             "conductor": {"acts": 0, "by_task": {}, "gave_ups": 0, "shadow": 0,
                           "efforts": {"by_level": {"high": 2, "xhigh": 5},
                                       "by_size": {"small": 2, "medium": 5},
                                       "size_mode": "on", "size_applied": 7,
                                       "size_effort": {"high": 2, "xhigh": 5}}}}] * 2
    suites = [{"id": "minicalc", "title": "minicalc"}]
    got = tandem_eval.suite_rows(rows, suites, "b", "n")[0]
    assert got["efforts"] == {"high": 4, "xhigh": 10} and got["size_mode"] == "on"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
