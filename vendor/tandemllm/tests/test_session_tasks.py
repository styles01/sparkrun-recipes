"""SRV-121 / SRV-122: the session-level tasks S2 (context health), S1 (gate stagnation) and S4
(stall watchdog, with its out-of-band watch and NOTIFY).

The test matrix is SRV-121's Gherkin, one test or more per scenario:

  1  no re-read after a compaction          test_no_reread_after_a_compaction_*
  2  a re-read after a compaction           test_a_reread_after_a_compaction_*
  3  a stuck gate ends the stage            test_a_stuck_gate_*
  4  an improving gate is left alone        test_an_improving_gate_*
  5  no progress for 20 minutes             test_no_progress_for_20_minutes_*
  6  the client has gone, NOTIFY off        test_the_client_has_gone_*, test_notify_*
  7  the prefix stays intact                test_the_prefix_stays_intact_over_http (fake engine)

plus the edge cases of the ticket (a project without state files, several stuck gates, a
compaction during a stall, the midnight date line, a retried request) and the checker parsers on
the real output lines of pagodacheck, scenecheck, sketchcheck and calccheck. The unit tests run
on a fake clock with no torch; the HTTP tests start the server with the fake engine.

Run: python tests/test_session_tasks.py [name-filter]
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from server.conductor import signals  # noqa: E402
from server.conductor.monitor import Journal, Monitor, Policy, RequestEvent  # noqa: E402
from server.conductor.monitor import notify, policy as policy_mod, sessiontasks  # noqa: E402
from server.conductor.monitor.policy import PASS, PROMPT_TAIL_NOTE  # noqa: E402

SPECIAL = {"<|im_start|>": 1_000_001, "<|im_end|>": 1_000_002, "<think>": 1_000_003,
           "</think>": 1_000_004}
SYSTEM = ("You are opencode.\nWorking directory: /Users/k/demos/run\n"
          "Instructions from: /Users/k/demos/run/AGENTS.md\n# Start of every turn\n"
          "1. Read `PROGRESS.yaml`. Work on the stage named in `next`.\n"
          "Today's date: Tue Sep 30 2026")
TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
         for n in ("read", "bash", "edit", "write", "skill")]
T0 = 1_790_000_000.0
MIN = 60.0


class CharTok:
    """One id a character, the chat template's markers one id each."""

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for s, t in SPECIAL.items():
                if text.startswith(s, i):
                    ids.append(t)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return type("R", (), {"input_ids": ids})()


def policy(only=("S2", "S1", "S4"), **changes) -> Policy:
    """ops/supervisor.yaml with only the session tasks (or `only`), each changed by
    `changes[task]` (a dict merged into it)."""
    d = yaml.safe_load(open(policy_mod.DEFAULT_PATH))
    d["tasks"] = {k: v for k, v in d["tasks"].items() if k in only}
    for tid, ch in changes.items():
        d["tasks"][tid] = _merge(d["tasks"][tid], ch)
    return Policy(d)


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def monitor(mode="on", pol=None) -> Monitor:
    mon = Monitor(mode, pol or policy(), CharTok(), Journal(echo=False))
    mon.notifier = _no_network
    return mon


def _no_network(cfg, title, message):
    raise AssertionError("a push was attempted")


class Sim:
    """One opencode-shaped session against a monitor, on a clock the test moves."""

    def __init__(self, mon: Monitor, conv="ses_1", system=SYSTEM):
        self.mon, self.conv = mon, conv
        self.messages = [{"role": "system", "content": system},
                         {"role": "user", "content": "Build the next stage."}]
        self.n = 0
        self.now = T0

    def step(self, *calls):
        """One assistant step: `(name, args, result)` per call, all in one message."""
        tc = []
        for name, args, _ in calls:
            self.n += 1
            tc.append({"id": f"call_{self.n}", "type": "function",
                       "function": {"name": name, "arguments": json.dumps(args)}})
        self.messages.append({"role": "assistant", "content": "", "tool_calls": tc})
        for c, (_, _, result) in zip(tc, calls):
            self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    def checker(self, output: str, cmd="pagodacheck . --perf x"):
        self.step(("bash", {"command": cmd}, output))

    def request(self, minutes: float | None = None, tools=True, messages=None, apply=True):
        """One request `minutes` after the session's start: (supervisor, actions, the prompt
        with the actions applied)."""
        if minutes is not None:
            self.now = T0 + minutes * MIN
        msgs = copy.deepcopy(messages if messages is not None else self.messages)
        req = RequestEvent(f"r{self.now:.0f}", messages=msgs, tools=TOOLS if tools else [],
                           tool_choice="auto", max_tokens=4096, client="opencode",
                           profile="overnight")
        sup = self.mon.begin(self.mon.mode_for("on"), req, [1, 2, 3], self.conv, now=self.now)
        acts = sup.at_request(req)
        prompt = [SPECIAL["<|im_start|>"], 117, SPECIAL["<|im_end|>"],
                  SPECIAL["<|im_start|>"], 97, SPECIAL["<think>"]]
        if apply and acts:
            prompt, _ = sup.apply_request(prompt, True, acts)
        sup.finish("tool_call")
        self.mon.turn_ended(sup, now=self.now + 1.0)
        return sup, acts, prompt

    def compaction(self, minutes: float):
        """opencode's summary request (no tools, the history as one user message), then the
        conversation goes on from the summary."""
        hist = "\n\n".join(json.dumps(m) for m in self.messages)[:6000].ljust(3000, ".")
        sup, acts, _ = self.request(minutes, tools=False,
                                    messages=[{"role": "user", "content": hist}])
        assert sup.shape.compaction and not acts
        self.messages = [self.messages[0], {"role": "user", "content": "Summary: S1 is next. "
                                                                      "Continue."}]


def notes(acts):
    return [a for a in acts if a.kind == PROMPT_TAIL_NOTE]


def events(mon, task=None):
    return [e for e in mon.journal.events() if (task is None or e.get("task") == task)
            and e.get("decision") not in ("OUTCOME", "LABEL", "COUNTS")]


PAGODA_FAIL = ("pagodacheck  /Users/k/demos/run\n"
               "  (a) summer vs night 1   ssim 0.981 (>= 0.95)   island colour diff {v} (<= 1.0)\n"
               "  heatmap /Users/k/demos/run/shots/check/pagodacheck.png\n"
               "RESULT: FAIL  <- (a) summer island colour differs by {v} > 1.0: summer must keep "
               "the night-1 colours")
PAGODA_PASS = ("pagodacheck  /Users/k/demos/run\n  (a) summer vs night 1   ssim 1.000\n"
               "RESULT: PASS")


def gate_a(v):
    return PAGODA_FAIL.replace("{v}", str(v))


# ------------------------------------------------------------------ S2 context health

def test_no_reread_after_a_compaction_gets_the_note_once():
    mon = monitor()
    s = Sim(mon)
    s.step(("read", {"filePath": "/Users/k/demos/run/PROGRESS.yaml"}, "next: S1"))
    s.request(1)
    s.compaction(10)
    _, acts, _ = s.request(11)                       # the first request after the summary
    assert not acts
    s.step(("bash", {"command": "ls js/seasons"}, "palette.js"))
    _, acts, _ = s.request(12)                       # one step, no re-read yet
    assert not notes(acts)
    s.step(("edit", {"filePath": "js/seasons/palette.js", "oldString": "a",
                     "newString": "b"}, "Edit applied successfully."))
    _, acts, prompt = s.request(13)                  # the second step read nothing either
    assert [a.task for a in notes(acts)] == ["S2"]
    text = mon.policy.tasks["S2"]["text"]
    assert notes(acts)[0].text == text
    # the note sits at the end of the last message, before its turn end, nothing after it moved
    note_ids = CharTok()("\n\n" + text).input_ids
    i = prompt.index(SPECIAL["<|im_end|>"])
    assert prompt[i - len(note_ids):i] == note_ids and prompt[-3:] == [SPECIAL["<|im_start|>"],
                                                                      97, SPECIAL["<think>"]]
    ev = [e for e in events(mon, "S2") if e["decision"] == PROMPT_TAIL_NOTE]
    assert len(ev) == 1 and ev[0]["mode"] == "act" and ev[0]["detail"]["steps"] == 2
    assert ev[0]["text"] == text and ev[0]["hidden_tokens"] == len(note_ids)
    s.step(("bash", {"command": "ls"}, "x"))
    _, acts, _ = s.request(14)
    assert not notes(acts), "once per compaction"
    # a second compaction watches again
    s.compaction(30)
    s.request(31)
    s.step(("bash", {"command": "ls"}, "x"))
    s.request(32)
    s.step(("bash", {"command": "pwd"}, "x"))
    _, acts, _ = s.request(33)
    assert [a.task for a in notes(acts)] == ["S2"]


def test_a_reread_after_a_compaction_adds_no_note():
    for first in (("read", {"filePath": "/Users/k/demos/run/PROGRESS.yaml"}, "next: S2-spring"),
                  ("bash", {"command": "cat AGENTS.md"}, "# Voxel Pagoda"),
                  ("skill", {"name": "voxel-seasons"}, "skill loaded")):
        mon = monitor()
        s = Sim(mon)
        s.compaction(5)
        s.request(6)
        s.step(first)
        _, acts, _ = s.request(7)
        assert not notes(acts), first
        s.step(("bash", {"command": "ls"}, "x"))
        s.step(("bash", {"command": "pwd"}, "x"))
        _, acts, _ = s.request(8)
        assert not notes(acts), first
        obs = [e for e in events(mon, "S2") if e["decision"] == PASS]
        assert len(obs) == 1 and obs[0]["reason"].startswith("reread(step 1:"), obs


def test_no_reread_counts_only_steps_after_the_summary():
    """A client that keeps its last steps in the request after the summary: they are not
    steps after the compaction."""
    mon = monitor()
    s = Sim(mon)
    s.step(("bash", {"command": "ls"}, "x"))
    s.request(1)
    hist = [{"role": "user", "content": "h" * 3000}]
    s.request(2, tools=False, messages=hist)
    s.request(3)                                     # still carries call_1: the baseline
    s.step(("bash", {"command": "pwd"}, "x"))
    _, acts, _ = s.request(4)
    assert not notes(acts), "one step after the compaction, not two"


def test_the_midnight_date_line_is_not_a_compaction():
    mon = monitor()
    s = Sim(mon)
    s.request(1)
    s.messages[0]["content"] = SYSTEM.replace("Tue Sep 30 2026", "Wed Oct 01 2026")
    sup, _, _ = s.request(2)
    assert not sup.shape.compaction and sup.session.last_compaction is None
    for i in range(3):
        s.step(("bash", {"command": f"ls {i}"}, "x"))
        _, acts, _ = s.request(3 + i)
        assert not notes(acts)


def test_a_compaction_without_a_conversation_id_carries_to_the_next_session():
    """A client without an id starts a new key after the summary (the router hashes the first
    messages): the compaction mark goes with it."""
    mon = monitor()
    # opencode's bare compaction shape (SRV-175: the shape is opencode's, by its client kind)
    req = RequestEvent("c", messages=[{"role": "user", "content": "h" * 3000}], tools=[],
                       client="opencode")
    mon.begin("on", req, [1], None, now=T0, session_key="h:old")
    s = Sim(mon, conv=None)
    req = RequestEvent("n", messages=s.messages, tools=TOOLS)
    sup = mon.begin("on", req, [1], None, now=T0 + 60, session_key="h:new")
    assert sup.session.last_compaction == T0


# ------------------------------------------------------------------ S1 gate stagnation

def test_a_stuck_gate_gets_the_note_naming_the_readings_and_the_rule():
    mon = monitor()
    s = Sim(mon)
    for i, v in enumerate((3.5, 12.3, 3.5)):
        s.checker(gate_a(v))
        _, acts, _ = s.request(1 + i)
    assert [a.task for a in notes(acts)] == ["S1"]
    assert notes(acts)[0].text == ("Gate (a) read 3.5, 12.3, 3.5. Record this stage as blocked "
                                   "in PROGRESS.yaml with the reason and take `next`.")
    ev = [e for e in events(mon, "S1") if e["decision"] == PROMPT_TAIL_NOTE][0]
    assert ev["detail"]["values"] == [3.5, 12.3, 3.5] and ev["detail"]["limit"] == 1.0
    # once per gate and stage: a fourth failure in the same stage adds nothing
    s.checker(gate_a(3.4))
    _, acts, _ = s.request(5)
    assert not notes(acts)


def test_an_improving_gate_is_left_alone():
    mon = monitor()
    s = Sim(mon)
    for i, v in enumerate((15.6, 12.2, 3.5)):
        s.checker(gate_a(v))
        _, acts, _ = s.request(1 + i)
        assert not notes(acts), v


def test_the_night_of_2026_09_30_notes_at_the_fifth_reading():
    """15.6, 12.2, 3.5, 12.3, 3.5 (18:53Z to 19:29:44Z): the note comes with the second 3.5."""
    mon = monitor()
    s = Sim(mon)
    got = []
    for i, v in enumerate((15.6, 12.2, 3.5, 12.3, 3.5)):
        s.checker(gate_a(v))
        _, acts, _ = s.request(1 + i * 8)
        got.append(bool(notes(acts)))
    assert got == [False, False, False, False, True]


def test_a_pass_or_a_new_stage_starts_the_gate_again():
    mon = monitor()
    s = Sim(mon)
    for i, v in enumerate((3.5, 3.5, 3.5)):
        s.checker(gate_a(v))
        _, acts, _ = s.request(1 + i)
    assert notes(acts)
    s.checker(PAGODA_PASS)
    s.request(4)
    for i, v in enumerate((3.5, 3.5)):
        s.checker(gate_a(v))
        _, acts, _ = s.request(5 + i)
        assert not notes(acts), "a pass starts a new streak"
    s.checker(gate_a(3.5))
    _, acts, _ = s.request(7)
    assert notes(acts), "a third failure after the pass is a new streak"
    # the same gate stuck again in the same stage: no second note; a new stage: a note
    mon2 = monitor()
    s2 = Sim(mon2)
    s2.step(("read", {"filePath": "PROGRESS.yaml"}, "status: running\nnext: S1-palette\n"))
    for i in range(3):
        s2.checker(gate_a(3.5))
        _, acts, _ = s2.request(1 + i)
    assert notes(acts) and notes(acts)[0].detail["stage"] == "S1-palette"
    s2.step(("edit", {"filePath": "/x/PROGRESS.yaml", "oldString": "next: S1-palette",
                      "newString": "next: S2-spring"}, "Edit applied successfully."))
    s2.checker(gate_a(3.5))
    _, acts, _ = s2.request(4)
    assert notes(acts) and notes(acts)[0].detail["stage"] == "S2-spring"


def test_several_stuck_gates_get_one_note_naming_the_worst():
    out = ("pagodacheck  /x\nRESULT: FAIL  <- (a) summer island colour differs by {a} > 1.0: x; "
           "(b) autumn: warm +{b} < 0.045 (leaves)")
    mon = monitor()
    s = Sim(mon)
    for i, (a, b) in enumerate(((3.5, 0.011), (3.6, 0.011), (3.5, 0.012))):
        s.checker(out.replace("{a}", str(a)).replace("{b}", str(b)))
        _, acts, _ = s.request(1 + i)
    assert len(notes(acts)) == 1
    n = notes(acts)[0]
    assert n.text.startswith("Gate (a) read 3.5, 3.6, 3.5."), n.text   # 2.5x the limit off
    assert sorted(n.detail["stuck"]) == ["pagodacheck:/x:a", "pagodacheck:/x:b"]
    s.checker(out.replace("{a}", "3.5").replace("{b}", "0.011"))
    _, acts, _ = s.request(5)
    assert not notes(acts), "the other stuck gate was named in the same decision"


def test_a_retried_request_counts_nothing_twice():
    mon = monitor()
    s = Sim(mon)
    s.checker(gate_a(3.5))
    s.request(1)
    s.checker(gate_a(3.5))
    for k in range(4):                               # opencode retries the same request
        _, acts, _ = s.request(2 + k * 0.1)
        assert not notes(acts)
    assert [len(g["runs"]) for g in s.mon.sessions.states()[0].tasks["S1"]["gates"].values()] \
        == [2]


def test_the_checker_parsers_read_every_demo_checker():
    pagoda = ("pagodacheck  /Users/k/demos/pagoda-seasons\n"
              "  (c) fps 41.0  calls 900  instances 1  lights 3  errors 0  season 1 t 12  sig ok\n"
              "RESULT: FAIL  <- (a) summer SSIM 0.912 < 0.95: the island or camera changed; "
              "(b) autumn: warm +0.011 < 0.045 (leaves and grass must turn red/orange/gold); "
              "(c) fps 41.0 < 50.0 at season 1 t 12; (d) files over 400 lines: a.js (split)")
    scene = ("scenecheck  shots/s2-far.png  vs  ref/photo.jpg   (stage: far)\n"
             "  de8 14.2\nRESULT: FAIL  <- de8 14.2 > 9.0; skyline_iou 0.71 < 0.8")
    sketch = ("sketchcheck  shots/s3-ribbon.png   (style: spiral)\n"
              "RESULT: FAIL  <- tone_corr 0.912 < 0.95; band 0.031 > 0.02 (stripes)")
    calc = "calccheck minicalc\n  [arith] '1 + 2' -> 4\nRESULT: FAIL <- (a) arith failed 2 > 0"
    t = policy().tasks["S1"]
    got = {c: sessiontasks.checker_runs(x, t)[0] for c, x in
           (("pagodacheck", pagoda), ("scenecheck", scene), ("sketchcheck", sketch),
            ("calccheck", calc))}
    assert {c: (r["checker"], r["verdict"]) for c, r in got.items()} == \
        {c: (c, "FAIL") for c in got}
    assert [(r["gate"], r["value"], r["op"], r["limit"]) for r in got["pagodacheck"]["readings"]] \
        == [("a", 0.912, "<", 0.95), ("b", 0.011, "<", 0.045), ("c", 41.0, "<", 50.0)]
    assert [(r["gate"], r["value"]) for r in got["scenecheck"]["readings"]] == \
        [("de8", 14.2), ("skyline_iou", 0.71)]
    assert got["sketchcheck"]["target"] == "shots/s3-ribbon.png"
    assert [(r["gate"], r["value"]) for r in got["sketchcheck"]["readings"]] == \
        [("tone_corr", 0.912), ("band", 0.031)]
    assert [(r["gate"], r["value"], r["limit"]) for r in got["calccheck"]["readings"]] == \
        [("a", 2.0, 0.0)]
    # the RESULT line a model quotes in its own text is not a result: only tool messages count
    mon = monitor()
    s = Sim(mon)
    for i in range(3):
        s.messages.append({"role": "assistant", "content": gate_a(3.5)})
        _, acts, _ = s.request(1 + i)
    assert not notes(acts)


# ------------------------------------------------------------------ S4 stall watchdog

def test_no_progress_for_20_minutes_gets_the_note():
    mon = monitor()
    s = Sim(mon)
    got = []
    for m in (0, 5, 10, 15, 19.5, 20.5, 25, 39, 41):
        s.step(("bash", {"command": f"sleep {m}"}, "ok"))
        _, acts, _ = s.request(m)
        got.append([a.task for a in notes(acts)])
    assert got == [[], [], [], [], [], ["S4"], [], [], ["S4"]], got
    ev = [e for e in events(mon, "S4") if e["decision"] == PROMPT_TAIL_NOTE]
    assert ev[0]["detail"]["minutes"] == 20.5 and ev[0]["detail"]["last_progress"] is None
    assert notes(acts)[0].text == ("No progress for 20 minutes. Re-read PROGRESS.yaml and take "
                                   "the next stage.")


def test_progress_resets_the_watchdog():
    for progress in (("edit", {"filePath": "a.js"}, "Edit applied successfully."),
                     ("write", {"filePath": "b.js"}, "Wrote file successfully."),
                     ("bash", {"command": "git commit -am x"}, "[main 1a2b3c4] x\n 1 file"),
                     ("bash", {"command": "pagodacheck ."}, gate_a(2.0)),
                     ("bash", {"command": "pagodacheck ."}, PAGODA_PASS)):
        mon = monitor()
        s = Sim(mon)
        s.checker(gate_a(3.5))
        s.request(0)
        s.step(progress)
        s.request(15)
        _, acts, _ = s.request(25)
        assert not notes(acts), progress
        _, acts, _ = s.request(36)
        assert [a.task for a in notes(acts)] == ["S4"], progress
    # what is not progress: a failed edit, a worse or equal checker number
    for nothing in (("edit", {"filePath": "a.js"}, "Could not find oldString in the file."),
                    ("bash", {"command": "pagodacheck ."}, gate_a(3.4))):
        mon = monitor()
        s = Sim(mon)
        s.checker(gate_a(3.5))
        s.request(0)
        s.step(nothing)
        _, acts, _ = s.request(21)
        assert [a.task for a in notes(acts)] == ["S4"], nothing


def test_a_compaction_during_a_stall_gets_s2_first_and_s4_next():
    mon = monitor()
    s = Sim(mon)
    s.request(0)
    s.compaction(18)
    s.request(19)
    s.step(("bash", {"command": "ls"}, "x"))
    s.request(19.5)
    s.step(("bash", {"command": "pwd"}, "x"))
    _, acts, _ = s.request(21)                       # S2 and S4 both due
    assert [a.task for a in notes(acts)] == ["S2"]
    s.step(("bash", {"command": "ls"}, "x"))
    _, acts, _ = s.request(22)
    assert not notes(acts), "a note was just added: S4 waits its minutes from it"
    _, acts, _ = s.request(42)
    assert [a.task for a in notes(acts)] == ["S4"]


def test_a_project_without_state_files_stays_silent():
    mon = monitor()
    s = Sim(mon, system="You are opencode.\nWorking directory: /tmp/minicalc")
    s.compaction(1)
    for i, v in enumerate((3.5, 3.5, 3.5, 3.5)):
        s.checker(gate_a(v))
        _, acts, _ = s.request(2 + i * 10)
        assert not acts
    assert not [e for e in events(mon) if e["decision"] != PASS]
    # without `requires` the same session gets its notes
    mon = monitor(pol=policy(S1={"requires": ""}))
    s = Sim(mon, system="You are opencode.")
    for i in range(3):
        s.checker(gate_a(3.5))
        _, acts, _ = s.request(1 + i)
    assert notes(acts)


def test_shadow_journals_and_adds_nothing():
    mon = monitor(mode="shadow")
    s = Sim(mon)
    for i in range(3):
        s.checker(gate_a(3.5))
        sup, acts, prompt = s.request(1 + i)
    ev = [e for e in events(mon, "S1") if e["decision"] == PROMPT_TAIL_NOTE]
    assert len(ev) == 1 and ev[0]["mode"] == "shadow" and not sup.acted
    assert prompt == [SPECIAL["<|im_start|>"], 117, SPECIAL["<|im_end|>"],
                      SPECIAL["<|im_start|>"], 97, SPECIAL["<think>"]]


def test_the_client_has_gone_with_notify_off():
    mon = monitor()
    s = Sim(mon)
    s.request(0)
    assert mon.sweep(T0 + 1 + 19 * MIN) == []
    got = mon.sweep(T0 + 1 + 20.2 * MIN)
    assert len(got) == 1
    e = got[0]
    assert (e["task"], e["sp"], e["decision"], e["mode"]) == ("S4", "watch", PASS, "act")
    assert e["reason"] == "client_gone(20.2 min)" and e["detail"]["notify"] == "off"
    assert mon.sweep(T0 + 60 * MIN) == [], "once a turn end"
    # a request in flight is not a gone client; a new turn end arms it again
    sup, _, _ = s.request(70, apply=False)
    sup.session.ended_at = None
    assert mon.sweep(T0 + 200 * MIN) == []
    mon.turn_ended(sup, now=T0 + 71 * MIN)
    assert len(mon.sweep(T0 + 92 * MIN)) == 1


def test_notify_off_never_touches_the_network():
    import urllib.request as ur
    real = ur.urlopen

    def boom(*a, **k):
        raise AssertionError("network")
    ur.urlopen = boom
    try:
        for mode, enabled in (("on", False), ("shadow", True)):
            mon = Monitor(mode, policy(S4={"notify": {"enabled": enabled}}), CharTok(),
                          Journal(echo=False))      # the real sender, not a stub
            s = Sim(mon)
            s.request(0)
            got = mon.sweep(T0 + 30 * MIN)
            assert len(got) == 1, mode
            assert got[0]["detail"]["notify"] == ("off" if not enabled else "shadow: not sent")
    finally:
        ur.urlopen = real


def test_notify_on_sends_one_push_with_the_token_from_its_file():
    sent = []

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None, context=None):
        sent.append((req.full_url, dict(req.header_items()), req.data.decode(), timeout))
        return Resp()

    with tempfile.TemporaryDirectory() as d:
        tok = os.path.join(d, "ntfy.token")
        with open(tok, "w") as f:
            f.write("tk_test\n")
        pol = policy(S4={"notify": {"enabled": True, "token_file": tok,
                                    "url": "https://ntfy.example/alerts"}})
        mon = Monitor("on", pol, CharTok(), Journal(echo=False))
        mon.notifier = lambda cfg, t, m: notify.send(cfg, t, m, opener=opener)
        s = Sim(mon)
        s.request(0)
        got = mon.sweep(T0 + 25 * MIN)
        assert got[0]["decision"] == "NOTIFY" and got[0]["detail"]["notify"] == "sent 200"
        assert len(sent) == 1
        url, headers, body, timeout = sent[0]
        assert url == "https://ntfy.example/alerts" and timeout == 10
        assert headers["Authorization"] == "Bearer tk_test" and headers["Tags"] == "hourglass"
        assert headers["Title"] == "TandemLLM: opencode session idle"
        assert "No request for 25 minutes" in body and "profile overnight" in body
        assert "tk_test" not in json.dumps(got[0]), "the token never reaches the journal"
        assert mon.sweep(T0 + 90 * MIN) == [] and len(sent) == 1
        # no token file: nothing leaves, the journal says why
        pol = policy(S4={"notify": {"enabled": True, "token_file": os.path.join(d, "none"),
                                    "url": "https://ntfy.example/alerts"}})
        mon = Monitor("on", pol, CharTok(), Journal(echo=False))
        s = Sim(mon)
        s.request(0)
        assert mon.sweep(T0 + 25 * MIN)[0]["detail"]["notify"] == "failed: no token file"


def test_notify_url_comes_from_the_environment_when_the_policy_names_none():
    """The shipped policy names no topic URL (a public repository holds no private host):
    `$TANDEM_NOTIFY_URL` gives it, and with neither nothing leaves."""
    sent = []

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None, context=None):
        sent.append(req.full_url)
        return Resp()

    assert yaml.safe_load(open(policy_mod.DEFAULT_PATH))["tasks"]["S4"]["notify"]["url"] == ""
    with tempfile.TemporaryDirectory() as d:
        tok = os.path.join(d, "ntfy.token")
        with open(tok, "w") as f:
            f.write("tk_test\n")
        cfg = {"token_file": tok, "url": ""}
        old = os.environ.pop(notify.URL_ENV, None)
        try:
            assert notify.send(cfg, "t", "m", opener=opener) == "failed: no url" and sent == []
            os.environ[notify.URL_ENV] = "https://push.example/topic"
            assert notify.send(cfg, "t", "m", opener=opener) == "sent 200"
            assert sent == ["https://push.example/topic"]
            # the policy's own URL wins over the environment
            assert notify.send(dict(cfg, url="https://a.example/b"), "t", "m",
                               opener=opener) == "sent 200"
            assert sent[-1] == "https://a.example/b"
        finally:
            os.environ.pop(notify.URL_ENV, None)
            if old is not None:
                os.environ[notify.URL_ENV] = old


def test_the_watch_lists_what_the_session_tasks_hold():
    mon = monitor()
    s = Sim(mon)
    for i, v in enumerate((15.6, 12.2)):
        s.checker(gate_a(v))
        s.request(1 + i)
    s.compaction(5)
    w = mon.watching(T0 + 30 * MIN)
    assert len(w) == 1
    row = w[0]
    assert row["gates"] == [{"task": "S1", "checker": "pagodacheck", "gate": "a", "op": ">",
                             "limit": 1.0, "values": [15.6, 12.2], "noted": False}]
    assert row["compaction"] == {"task": "S2", "count": 1, "watching": True, "steps": 0}
    assert row["watch"]["minutes"] == 20 and row["watch"]["notify"] is False
    assert row["idle_minutes"] == round(25 - 1 / 60, 1) and not row["running"]
    schema = json.load(open(os.path.join(ROOT, "docs", "contract", "conductor-v1",
                                         "watch.schema.json")))
    from tools import contract_check
    errs = contract_check.validate({"contract_version": "1.0", "monitor": True, "sessions": w},
                                   schema)
    assert not errs, errs


def test_the_profile_context_sets_the_session_tasks():
    from server.conductor import schema
    from server.conductor.hook import Conductor
    from server.conductor.router import Plan
    with tempfile.TemporaryDirectory() as d:
        cond = Conductor(d, journal=False)
        base = policy_mod.load()
        for name, want in (("overnight", (3, 20, True, "act")),
                           ("coding", (3, 20, False, "off"))):
            doc = schema.seed("profiles", name)
            plan = Plan(type("Res", (), {"profile": name, "version": 0, "doc": doc,
                                         "mode": "on"})())
            plan.active = True
            pol = cond.monitor_policy(plan, base)
            got = (pol.tasks["S1"]["fails"], pol.tasks["S4"]["minutes"],
                   pol.tasks["S4"]["notify"]["enabled"], pol.tasks["S2"].mode)
            assert got == want, (name, got)
        doc = schema.seed("profiles", "overnight")
        doc["context"] = {"s1_runs": 4, "s4_minutes": 45, "notify": False}
        plan = Plan(type("Res", (), {"profile": "overnight", "version": 7, "doc": doc,
                                     "mode": "on"})())
        plan.active = True
        pol = cond.monitor_policy(plan, base)
        assert (pol.tasks["S1"]["fails"], pol.tasks["S4"]["minutes"],
                pol.tasks["S4"]["notify"]["enabled"]) == (4, 45, False)
        cond.close()
    # the policy file: off for NOTIFY, act for the four, the order S2, S1, S4, S5
    p = policy_mod.load()
    assert [t for t in p.tasks if t.startswith("S")] == ["S2", "S1", "S4", "S5"]   # S5: SRV-179
    assert p.tasks["S4"]["notify"]["enabled"] is False
    assert schema.seed("profiles", "chat")["context"]["notify"] is False


# ------------------------------------------------------------------ over HTTP (fake engine)

def _server(policy_path: str, config_dir: str | None = None, extra=()):
    from test_conductor_e2e import _env, _port, _wait, LAUNCH
    import subprocess
    port = _port()
    ledger = os.path.join(tempfile.mkdtemp(prefix="qse-s121-"), "ledger.sqlite3")
    args = [sys.executable, "-c", LAUNCH, "--fake-engine", "--port", str(port),
            "--served-model", "qwen38-spark-engine", "--usage-ledger", ledger,
            "--fake-tps", "4000", "--conductor", "on", "--conductor-policy", policy_path,
            "--conductor-watch-s", "0.2", *extra]
    if config_dir:
        args += ["--conductor-config", config_dir]
    proc = subprocess.Popen(args, cwd=ROOT, env=_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    _wait(port, proc)
    return port, proc, ledger


def _post(port, messages, session="ses_http", stream=False, headers=None):
    body = {"model": "TandemLLM", "messages": messages, "max_tokens": 200, "stream": stream,
            "tools": TOOLS}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "x-session-id": session, **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode()
    if stream:
        return raw
    return json.loads(raw)


def _journal(ledger: str, task: str, wait_s: float = 10.0, until=None) -> list[dict]:
    """The monitor's rows of `task` in the journal table of the server's ledger file."""
    import sqlite3
    end = time.time() + wait_s
    rows = []
    while time.time() < end:
        try:
            con = sqlite3.connect(ledger)
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute(
                "SELECT * FROM conductor_events WHERE kind = 'monitor' AND task = ? "
                "ORDER BY ts_ms", (task,))]
            con.close()
        except sqlite3.Error:
            rows = []
        if rows and (until is None or until(rows)):
            break
        time.sleep(0.3)
    for r in rows:
        r["detail"] = json.loads(r["detail"]) if r.get("detail") else None
    return rows


def _policy_file(**changes) -> str:
    d = yaml.safe_load(open(policy_mod.DEFAULT_PATH))
    d["tasks"] = {k: v for k, v in d["tasks"].items() if k in ("S2", "S1", "S4")}
    for tid, ch in changes.items():
        d["tasks"][tid] = _merge(d["tasks"][tid], ch)
    path = os.path.join(tempfile.mkdtemp(prefix="qse-s121-pol-"), "policy.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(d, f)
    return path


def _http_session():
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Build it."}]
    n = [0]

    def step(name, args, result, content=""):
        n[0] += 1
        cid = f"call_{n[0]}"
        msgs.append({"role": "assistant", "content": content, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name,
                                                         "arguments": json.dumps(args)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": result})
    return msgs, step


def test_the_session_tasks_over_http_note_hidden_and_counted():
    """Scenario 3 over HTTP: the note reaches the model, never the client; usage counts what
    the client sent; the journal holds the note's exact text and the ledger its hidden tokens."""
    port, proc, ledger = _server(_policy_file(S4={"minutes": 600}))
    try:
        msgs, step = _http_session()
        for v in (3.5, 12.3):
            step("bash", {"command": "pagodacheck ."}, gate_a(v))
            _post(port, msgs)
        step("bash", {"command": "pagodacheck ."}, gate_a(3.5))
        plain = copy.deepcopy(msgs)
        got = _post(port, msgs, stream=True)
        note = "Gate (a) read 3.5, 12.3, 3.5"
        assert note not in got and "Record this stage" not in got
        usage = [json.loads(line[6:]).get("usage") for line in got.splitlines()
                 if line.startswith("data: {")]
        usage = [u for u in usage if u][-1]
        # the same request on a server whose session tasks are off: the prompt it counts
        port2, proc2, _ = _server(_policy_file(S1={"mode": "off"}, S2={"mode": "off"},
                                               S4={"mode": "off"}))
        try:
            base = _post(port2, plain)["usage"]
        finally:
            proc2.terminate()
            proc2.wait(10)
        assert usage["prompt_tokens"] == base["prompt_tokens"], (usage, base)
        acted = [e for e in _journal(ledger, "S1") if e["decision"] == PROMPT_TAIL_NOTE]
        assert len(acted) == 1 and acted[0]["mode"] == "act" and acted[0]["hidden_tokens"] > 0
        assert acted[0]["detail"]["values"] == [3.5, 12.3, 3.5]
        import sqlite3
        con = sqlite3.connect(ledger)
        text = con.execute("SELECT text FROM conductor_texts WHERE id = ?",
                           (acted[0]["hidden_text_id"],)).fetchone()[0]
        assert note in text
        end = time.time() + 10
        row = None
        while time.time() < end:
            row = con.execute("SELECT conductor_acts, conductor_hidden_tokens, conductor_acted "
                              "FROM requests ORDER BY id DESC LIMIT 1").fetchone()
            if row and row[1]:
                break
            time.sleep(0.3)
        assert row[1] == acted[0]["hidden_tokens"] and row[2] == 1, row
    finally:
        proc.terminate()
        proc.wait(10)


def test_the_watch_api_under_the_overnight_profile_over_http():
    """Under the conductor's profiles the session tasks run in `overnight` and nowhere else;
    GET /v1/conductor/watch shows what they hold, and validates."""
    from test_conductor_e2e import _call, _check
    port, proc, _ = _server(policy_mod.DEFAULT_PATH, config_dir=tempfile.mkdtemp(
        prefix="qse-s121-cond-"))
    try:
        for profile, session in (("overnight", "ses_night"), ("coding", "ses_day")):
            msgs, step = _http_session()
            for v in (3.5, 12.3, 3.5):
                step("bash", {"command": "pagodacheck ."}, gate_a(v))
                _post(port, msgs, session=session, headers={"X-Tandem-Profile": profile})
        st, _, w = _call(port, "GET", "/v1/conductor/watch", token=None)
        assert st == 200, w
        _check(w, "watch")
        by = {x["session"]: x for x in w["sessions"]}
        night = by["c:ses_night"]
        assert night["gates"][0]["values"] == [3.5, 12.3, 3.5] and night["gates"][0]["noted"]
        assert night["watch"]["profile"] == "overnight" and night["watch"]["notify"] is True
        assert night["watch"]["minutes"] == 20
        # coding runs none of S1, S2, S4 (S5, the turn budget, runs under every profile: its
        # state is the turn's, no gate and no watch)
        day = by.get("c:ses_day")
        assert day is None or (not day.get("gates") and not day.get("watch")), day
        st, _, evs = _call(port, "GET", "/v1/conductor/events?task=S1", token=None)
        _check(evs, "events")
        assert [e["session_key"] for e in evs["events"] if e["decision"] == PROMPT_TAIL_NOTE] \
            == ["c:ses_night"]
    finally:
        proc.terminate()
        proc.wait(10)


def test_the_prefix_stays_intact_over_http():
    """Scenario 7: the request after the note, without it, resumes at the last grid line before
    the note (the fake engine's prefix grid is 64 tokens, the resident prefix's 1,024)."""
    port, proc, _ = _server(_policy_file(S4={"minutes": 600}))
    try:
        msgs, step = _http_session()
        for v in (3.5, 12.3, 3.5):
            step("bash", {"command": "pagodacheck ."}, gate_a(v))
            r1 = _post(port, msgs)
        prompt_n = r1["usage"]["prompt_tokens"]
        # the fake template's tail after the note: <|im_end|>, "\n", the generation prompt
        tail = 1 + 1 + 1 + len("assistant\n") + 1 + 1
        at = prompt_n - tail
        step("bash", {"command": "ls"}, "x")
        r2 = _post(port, msgs)
        assert r2["timings"]["cache_n"] == (at // 64) * 64, (r2["timings"], at)
        # without a note the next turn reuses the whole previous prompt's grid
        step("bash", {"command": "pwd"}, "y")
        r3 = _post(port, msgs)
        assert r3["timings"]["cache_n"] > (at // 64) * 64
    finally:
        proc.terminate()
        proc.wait(10)


def test_the_client_has_gone_over_http():
    """Scenario 6: no request after a turn for S4's minutes: client_gone in the journal, and
    nothing sent (NOTIFY off)."""
    port, proc, ledger = _server(_policy_file(S4={"minutes": 0.03}))
    try:
        msgs, _ = _http_session()
        _post(port, msgs)
        gone = _journal(ledger, "S4", wait_s=15,
                        until=lambda rows: any(r["sp"] == "watch" for r in rows))
        gone = [r for r in gone if r["sp"] == "watch"]
        assert len(gone) == 1 and gone[0]["decision"] == PASS, gone
        assert gone[0]["trigger"].startswith("client_gone(")
        assert gone[0]["detail"]["notify"] == "off"
    finally:
        proc.terminate()
        proc.wait(10)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.path.insert(0, HERE)
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and only in name:
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
