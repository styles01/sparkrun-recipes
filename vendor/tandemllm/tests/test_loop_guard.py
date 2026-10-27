"""SRV-473 (QA SRV-474): the same-call guard (C3) on by default in the agent profiles.

The test matrix is SRV-473's Gherkin, one test or more per row: C3's rule on its own (a supervisor
stub), through the real handler with the fake engine in this process (tests/test_toolcheck.py's
harness), and the stored-profile migration on a temporary store.

  1  a loop of identical calls is stopped at the third call      test_1_*
  2  a test re-run after an edit is not stopped                   test_2_*
  3  polling a running job is not stopped (a wait, or new results) test_3_*
  4  a user message starts the streak over                        test_4_*
  5  a turn that is not a repeat streams its calls as before      test_5_*
  6  a guard turned off explicitly stays off                      test_6_*
  7  stored profiles get the guard by a one-time migration        test_7_*
  8  general keeps the release's request                          test_8_*

Run: python tests/test_loop_guard.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import test_toolcheck as tc  # noqa: E402

from server.conductor import migrations, schema  # noqa: E402
from server.conductor.monitor import toolcheck  # noqa: E402
from server.conductor.monitor.observe import ToolCallEvent  # noqa: E402
from server.conductor.monitor.policy import REGENERATE_CALL  # noqa: E402
from server.conductor.monitor.session import SessionState  # noqa: E402
from server.conductor.store import Store  # noqa: E402

PYTEST = {"command": "pytest -q"}
FAIL = "1 failed, 4 passed in 0.31s"
USER = {"role": "user", "content": "fix the failing test"}


def turn(*calls) -> list:
    """Assistant/tool pairs: each `(name, args, result)` as one call with its result."""
    out = []
    for i, (name, args, result) in enumerate(calls):
        out.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]})
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": result})
    return out


def c3(**over):
    t = tc.policy().tasks["C3"]
    t.params.update(over)
    return t


def sup(messages, waits=None):
    s = SessionState("k", 0.0)
    s.waits.update(waits or {})
    return types.SimpleNamespace(req=types.SimpleNamespace(messages=messages), session=s,
                                 memo={}, task_acts={})


def decide(messages, name="bash", args=PYTEST, waits=None, **over):
    ev = ToolCallEvent(name, json.dumps(args), 0)
    return toolcheck.repeated_call(ev, c3(**over), sup(messages, waits))


def held_policy():
    """ops/supervisor.yaml with C3 the only tool-call task that acts (coding's live shape: the
    risk and project packs in shadow, malformed off)."""
    return tc.policy(C4="off", C1="shadow", C2="shadow")


# ---------------------------------------------------------------------- 1
def test_1_a_loop_of_identical_calls_is_stopped_at_the_third_call():
    msgs = [USER] + turn(("bash", PYTEST, FAIL), ("bash", PYTEST, FAIL))
    act = decide(msgs)
    assert act is not None and act.kind == REGENERATE_CALL, act
    assert "You ran `bash` with these arguments 2 times" in act.text and FAIL in act.text
    # key order and whitespace do not make another call
    assert decide(msgs, args={"command": "pytest -q"}) is not None
    # through the handler: the third call never reaches the client, the model writes another
    same = tc.history("bash", PYTEST, [FAIL, FAIL])
    for stream in (True, False):
        mon = tc.serve(pol=held_policy())
        got = tc.ask(tc.call_json("bash", **PYTEST), history=same, stream=stream)
        assert list(got["calls"]) == [0], got
        assert json.loads(got["calls"][0]["args"]) != PYTEST, got
        rows = tc.rows(mon, task="C3", decision="REGENERATE_CALL", mode="act")
        assert len(rows) == 1, mon.journal.events()
        if stream:
            assert got["deltas"] == 1, "a held call goes out whole"


# ---------------------------------------------------------------------- 2
def test_2_a_test_rerun_after_an_edit_is_not_stopped():
    edit = ("edit", {"filePath": "src/x.py", "oldString": "a", "newString": "b"}, "Edit applied.")
    msgs = [USER] + turn(("bash", PYTEST, FAIL), edit, ("bash", PYTEST, FAIL))
    assert decide(msgs) is None
    hist = tc.history("bash", PYTEST, [FAIL]) + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "e1", "type": "function",
         "function": {"name": "edit", "arguments": json.dumps({"filePath": "src/x.py"})}}]},
        {"role": "tool", "tool_call_id": "e1", "content": "Edit applied."}]
    hist += [dict(m) for m in tc.history("bash", PYTEST, [FAIL])]
    hist[-2]["tool_calls"][0]["id"] = hist[-1]["tool_call_id"] = "c9"
    for stream in (True, False):
        mon = tc.serve(pol=held_policy())
        got = tc.ask(tc.call_json("bash", **PYTEST), history=hist, stream=stream)
        assert [json.loads(c["args"]) for c in got["calls"].values()] == [PYTEST], got
        assert not tc.rows(mon, task="C3"), mon.journal.events()


# ---------------------------------------------------------------------- 3
def test_3_polling_a_running_job_is_not_stopped():
    status = {"command": "make status"}
    msgs = [USER] + turn(("bash", status, "running"), ("bash", status, "running"))
    # the client took 30 s on each result: a wait, not a loop
    assert decide(msgs, args=status, waits={"c0": 30.0, "c1": 31.5}) is None
    # one wait in the streak is enough
    assert decide(msgs, args=status, waits={"c0": 1.0, "c1": 12.0}) is None
    # a hot poll (1.8 s a round) is a loop
    assert decide(msgs, args=status, waits={"c0": 1.8, "c1": 1.7}) is not None
    # without the policy's wait_s (a stored policy from before SRV-473): no exemption
    assert decide(msgs, args=status, waits={"c0": 30.0, "c1": 30.0}, wait_s=0) is not None
    # results that change pass, waiting or not
    moving = [USER] + turn(("bash", status, "12 %"), ("bash", status, "48 %"))
    assert decide(moving, args=status) is None
    # the session records the client's time per call id, from the end of the last answer
    s = SessionState("k", 0.0)
    s.ended_at = 100.0
    s.record_results([USER] + turn(("bash", status, "running"))[:2], 130.0)
    assert s.waits == {"c0": 30.0}
    s.record_results([USER] + turn(("bash", status, "running"))[:2], 200.0)
    assert s.waits == {"c0": 30.0}, "the first time a result arrives is its time"


# ---------------------------------------------------------------------- 4
def test_4_a_user_message_starts_the_streak_over():
    again = {"role": "user", "content": "run the tests again"}
    msgs = [USER] + turn(("bash", PYTEST, FAIL), ("bash", PYTEST, FAIL)) + [again]
    assert decide(msgs) is None
    assert toolcheck.turn_history(msgs) == []
    # within the new turn the streak counts again
    msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "n0", "type": "function",
              "function": {"name": "bash", "arguments": json.dumps(PYTEST)}}]},
             {"role": "tool", "tool_call_id": "n0", "content": FAIL}]
    assert decide(msgs) is None
    msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "n1", "type": "function",
              "function": {"name": "bash", "arguments": json.dumps(PYTEST)}}]},
             {"role": "tool", "tool_call_id": "n1", "content": FAIL}]
    assert decide(msgs) is not None


# ---------------------------------------------------------------------- 5
def test_5_a_turn_that_is_not_a_repeat_streams_its_calls_as_before():
    """C3 the only acting tool-call task: the calls stream live unless the turn already ends
    with the streak. The control is the guard off (C3 off): the same deltas."""
    big = {"command": "python3 - <<'EOF'\n" + "print('x')\n" * 40 + "EOF"}
    live = {}
    for label, pol in (("guard on", held_policy()),
                       ("guard off", tc.policy(C4="off", C3="off", C1="shadow", C2="shadow"))):
        mon = tc.serve(pol=pol)
        got = tc.ask(tc.call_json("bash", **big), history=tc.history("read", {"filePath": "a"},
                                                                     ["one"]))
        assert [json.loads(c["args"]) for c in got["calls"].values()] == [big], got
        live[label] = got["deltas"]
        assert not tc.rows(mon, task="C3"), mon.journal.events()
    assert live["guard on"] == live["guard off"] and live["guard on"] > 1, live
    # the hold decision itself: only on a request whose turn ends with the streak
    task = c3()
    assert not toolcheck.can_repeat(sup([USER] + turn(("bash", PYTEST, FAIL))), task)
    assert toolcheck.can_repeat(sup([USER] + turn(("bash", PYTEST, FAIL),
                                                  ("bash", PYTEST, FAIL))), task)
    assert not toolcheck.can_repeat(sup([USER] + turn(("bash", PYTEST, FAIL),
                                                      ("bash", PYTEST, "2 failed"))), task)


# ---------------------------------------------------------------------- 6, 7: the migration
GUARDED = ("coding", "chat-app", "hermes", "overnight", "chat")
LOOP = [m for m in migrations.MIGRATIONS if m[0].endswith("-loop-guard")]


class _Journal:
    def __init__(self):
        self.rows = []

    def note(self, kind, task, *, decision=None, detail=None, **_):
        self.rows.append((kind, task, decision, detail))


def old_doc(name: str) -> dict:
    """The profile as a box stored it before SRV-473: the guard off, no sampling section, chat's
    C3 off."""
    d = copy.deepcopy(schema.SHIPPED_PROFILES[name])
    d["toolgate"]["same_call"] = False
    d.pop("sampling", None)
    if name == "chat":
        d["tasks"]["C3"]["mode"] = "off"
    return d


def box(*names, versions=1) -> Store:
    st = Store(tempfile.mkdtemp(prefix="cond-loop-"))
    for name in names:
        for i in range(versions):
            st.put("profiles", name, old_doc(name), note=f"as stored {i}", author="owner")
    return st


def test_6_a_guard_turned_off_explicitly_stays_off():
    st = box("coding")
    on = old_doc("coding")
    on["toolgate"]["same_call"] = True
    st.put("profiles", "coding", on, note="guard on", author="owner")
    st.put("profiles", "coding", old_doc("coding"), note="guard off again", author="owner")
    got = migrations.run(st, migrations=LOOP)
    rec = [r for r in got if r["name"] == "coding"][0]
    # the guard stays off; the sampling section, a new key, is still added
    assert rec["result"] == "changed" and rec["changes"] == [
        "sampling={agentic_default: greedy, loop_escape: on}"], rec
    v, doc = st.get("profiles", "coding")
    assert doc["toolgate"]["same_call"] is False and v == 4
    # after the migration ran, an owner's off is never touched again
    st2 = box("hermes")
    migrations.run(st2, migrations=LOOP)
    v, doc = st2.get("profiles", "hermes")
    assert doc["toolgate"]["same_call"] is True
    doc["toolgate"]["same_call"] = False
    st2.put("profiles", "hermes", doc, note="off", author="owner", base=v)
    assert migrations.run(st2, migrations=LOOP) == []
    assert st2.get("profiles", "hermes")[1]["toolgate"]["same_call"] is False


def test_7_stored_profiles_get_the_guard_by_a_one_time_migration():
    st = box(*GUARDED, "general", versions=2)
    j = _Journal()
    got = {r["name"]: r for r in migrations.run(st, j, LOOP)}
    assert set(got) == set(GUARDED), got
    for name in GUARDED:
        r = got[name]
        assert (r["result"], r["version"]) == ("changed", 3), r
        want = ["toolgate.same_call=true"] + (["tasks.C3.mode=act"] if name == "chat" else []) + \
            ["sampling={agentic_default: greedy, loop_escape: on}"]
        assert r["changes"] == want, r
        v, doc = st.get("profiles", name)
        assert doc["toolgate"]["same_call"] is True and doc["tasks"]["C3"]["mode"] == "act"
        assert doc["sampling"] == {"agentic_default": "greedy", "loop_escape": "on"}
        meta = st.versions("profiles", name)[0]
        assert meta["note"] == f"migration 2026-10-14-{name}-loop-guard"
        assert meta["author"] == "migration"
        # nothing else in the document moved
        rest = {k: v_ for k, v_ in doc.items() if k not in ("toolgate", "tasks", "sampling")}
        old = {k: v_ for k, v_ in old_doc(name).items() if k not in ("toolgate", "tasks")}
        assert rest == old, name
    assert st.get("profiles", "general")[0] == 2, "general is never touched"
    assert len(j.rows) == len(GUARDED) and all(r[3]["changes"] for r in j.rows)
    with open(os.path.join(st.root, migrations.STATE)) as f:
        applied = json.load(f)["applied"]
    assert applied["2026-10-14-coding-loop-guard"]["changes"][0] == "toolgate.same_call=true"
    assert migrations.run(st, j, LOOP) == [], "runs once"
    # a box whose profiles have it already (the live PUT) only gets the sampling section
    st = box("coding")
    v, doc = st.get("profiles", "coding")
    doc["toolgate"]["same_call"] = True
    st.put("profiles", "coding", doc, note="SRV-473 loop guard on", author="owner", base=v)
    rec = [r for r in migrations.run(st, migrations=LOOP) if r["name"] == "coding"][0]
    assert rec["changes"] == ["sampling={agentic_default: greedy, loop_escape: on}"], rec
    # a shipped document (same_call on, sampling set) is unchanged; read-only runs nothing
    st = Store(tempfile.mkdtemp(prefix="cond-loop-"))
    st.put("profiles", "coding", copy.deepcopy(schema.SHIPPED_PROFILES["coding"]),
           note="shipped", author="shipped")
    rec = [r for r in migrations.run(st, migrations=LOOP) if r["name"] == "coding"][0]
    assert (rec["result"], rec["version"]) == ("unchanged", 1)
    assert migrations.run(Store(box("coding").root, readonly=True)) == []


# ---------------------------------------------------------------------- 8
def test_8_general_keeps_the_releases_request():
    g = schema.SHIPPED_PROFILES["general"]
    assert g["toolgate"]["same_call"] is False and g["tasks"]["C3"]["mode"] == "off"
    # the hook turns a task whose pack is off to off, whatever its mode says
    from server.conductor.hook import Conductor
    assert Conductor.PACKS["C3"] == "same_call"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:                               # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, exc)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
