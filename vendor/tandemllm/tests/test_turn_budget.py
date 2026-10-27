"""SRV-179 / SRV-181 (QA: SRV-180, SRV-182): S5, the turn budget (server/conductor/monitor/
sessiontasks.py `turn_budget`), on a fake clock with no torch.

The test matrix, from the two tickets' Gherkin:

  SRV-179 (an agent task ends with an answer)
    1  the deliverable works, the model states the same failure of its own test bot 3 times
       -> the repeat note, naming the problem; journaled with the arm and the count
                                                  test_the_same_problem_three_times_gets_the_note
    2  a chat client's turn at the hard cap -> the wrap-up note, later the shorter one
                                                  test_a_chat_turn_past_its_minutes_*
    3  a long task that keeps writing is not cut before the hard cap; a checker-like failure
       whose numbers change is progress, not a repeat
                                                  test_progress_is_not_cut,
                                                  test_changing_numbers_are_progress
    4  opencode (no one waits in a chat) has no wall clock; a new user message starts over
                                                  test_opencode_has_no_wall_clock,
                                                  test_a_new_user_message_starts_a_new_turn
  SRV-181 (a small page is written before it is tested)
    5  a build turn 3 minutes and 5 steps in with no product file -> the first-write note, once
                                                  test_setup_before_the_first_file_gets_the_note
    6  scratch writes (.tandem, /tmp, logs) are not the deliverable; a heredoc into the project is
                                                  test_what_counts_as_a_write
    7  a debug turn (reads first) gets no first-write note
                                                  test_a_debug_turn_has_no_first_write_note
  SRV-195 (QA: SRV-196; the deliverable exists, then a test rig of the model's own)
    9  FX2's opencode pomodoro (index.html at 176 s, its last product edit at 257 s, then a
       Firefox Marionette driver in .tandem until 600 s), replayed step by step on its own
       timings -> the verify note before 500 s, once; UT1's todo (a bun download and test
       scripts after the page) gets it too        test_a_rig_after_the_first_write_gets_the_verify_note
   10  real progress is left alone: the clean opencode runs (UT2 bug and refactor, UT1 bug fix
       and duplicate finder: its test logs, test data in /tmp and one Laya helper are not a rig)
       get no S5 note; edits to the product reset the count; a debug turn (not a build) gets none
                                                test_clean_runs_and_progress_get_no_verify_note
  the policy and the box
    8  ops/supervisor.yaml validates with S5, the JSON copy matches, the box's stored document
       gains S5 through tools/monitor_migrate.py and keeps its own tasks; a box whose S5 lacks
       the verify parameters (0.3.43, policy 6) gains them and keeps its own values
                                                  test_policy_and_migration

The texts the leads use are the Mario chat's own (UT2, 2026-10-06, chat 41a8fc6c), the timings
the replay's (`replay179.py` in the FX2 note: the first-write note at 3.1 min, the wall clock at
18.0 min, the repeat note at the third "stuck at x=436" lead, 36.0 min; no note on four clean
opencode runs).

Run: python tests/test_turn_budget.py [name-filter]
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from server.conductor.monitor import RequestEvent  # noqa: E402
from server.conductor.monitor import policy as policy_mod, sessiontasks  # noqa: E402
from server.conductor.monitor.policy import PROMPT_TAIL_NOTE  # noqa: E402
from test_session_tasks import SPECIAL, events, monitor, policy  # noqa: E402

T0 = 1_790_000_000.0
MIN = 60.0
TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
         for n in ("run_command", "get_process_status", "write_file", "replace_file_content",
                   "bash", "write", "edit", "read")]
STUCK = ["The bot is stuck at x=436 because it's pushing into a pipe wall and only jumps for pits "
         "and enemies. I'll add a blocked-wall jump condition to fix this.",
         "The bot is stuck at x=436, right against the pipe wall. When it jumps, it can't move "
         "right because `moveX` clamps its position to the wall.",
         "The bot is stuck at x=436, right before the pipe at x=448. It's jumping too late.",
         "The bot is stuck at x=436, right against the pipe. It's jumping but can't move right."]


class Chat:
    """One agent turn: the user's message, then tool steps, each a request on the test's clock."""

    def __init__(self, mon, client="open-webui", cls="code_edit", conv="chat-1",
                 text="Write a Mario game very classic on with Mobil view html"):
        self.mon, self.client, self.cls, self.conv = mon, client, cls, conv
        self.messages = [{"role": "system", "content": "You are an agent with a terminal."},
                         {"role": "user", "content": text}]
        self.n = 0

    def step(self, name="run_command", args=None, result="ok", content=""):
        self.n += 1
        cid = f"call_{self.n}"
        self.messages.append({"role": "assistant", "content": content, "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args or {"command": "ls"})}}]})
        self.messages.append({"role": "tool", "tool_call_id": cid, "content": result})

    def user(self, text):
        self.messages.append({"role": "assistant", "content": "Done."})
        self.messages.append({"role": "user", "content": text})

    def request(self, minutes):
        req = RequestEvent(f"r{self.n}-{minutes}", messages=json.loads(json.dumps(self.messages)),
                           tools=TOOLS, tool_choice="auto", max_tokens=32768, client=self.client,
                           profile="chat", cls=self.cls)
        sup = self.mon.begin("on", req, [1, 2, 3], self.conv, now=T0 + minutes * MIN)
        acts = [a for a in sup.at_request(req) if a.kind == PROMPT_TAIL_NOTE]
        if acts:
            prompt = [SPECIAL["<|im_start|>"], 117, SPECIAL["<|im_end|>"],
                      SPECIAL["<|im_start|>"], 97, SPECIAL["<think>"]]
            sup.apply_request(prompt, True, acts)           # journals each note as applied
        sup.finish("tool_call")
        self.mon.turn_ended(sup, now=T0 + minutes * MIN + 1)
        return acts


def s5(**changes):
    return monitor(pol=policy(only=("S5",), **({"S5": changes} if changes else {})))


def _write(c, path="/home/user/mario-mobile/index.html"):
    c.step("write_file", {"path": path, "content": "<html></html>"}, '{"size": 13}')


# ------------------------------------------------------------------ SRV-179

def test_the_same_problem_three_times_gets_the_note():
    mon = s5(verify_scratch_writes=99)      # the repeat arm alone (the verify arm: test_a_rig_*)
    c = Chat(mon)
    assert c.request(0) == []
    _write(c)
    for i, lead in enumerate(STUCK[:2]):
        c.step("replace_file_content", {"path": "/home/user/mario-mobile/.tandem/playtest.js"},
               '{"size": 2443}', content="\n\n" + lead + "\n\n")
        assert c.request(1 + i) == [], i
    c.step("run_command", {"command": "node .tandem/playtest.js"}, "max x reached: 436",
           content="\n\nI need to expose `__tileAt` in the game:\n\n")   # not a problem lead
    assert c.request(3) == []
    c.step("replace_file_content", {"path": "/home/user/mario-mobile/.tandem/playtest.js"},
           '{"size": 2909}', content="\n\n" + STUCK[2] + "\n\n")
    acts = c.request(4)
    assert len(acts) == 1 and acts[0].reason == "turn_budget(repeat 3)", acts
    assert "stated the same problem 3 times" in acts[0].text
    assert "The bot is stuck at x=436, right before the pipe" in acts[0].text
    assert "Do not change the product only to make your own test pass" in acts[0].text
    ev = [e for e in events(mon, "S5") if e["decision"] == PROMPT_TAIL_NOTE]
    assert ev[-1]["detail"]["arm"] == "repeat" and ev[-1]["detail"]["count"] == 3
    assert ev[-1]["detail"]["numbers"] == ["436", "448"]
    # the count starts over after the note
    c.step(content=STUCK[3])
    assert c.request(5) == []


def test_a_chat_turn_past_its_minutes_gets_the_wrap_up_then_the_short_note():
    mon = s5()
    c = Chat(mon)
    c.request(0)
    _write(c)
    for m in (2, 6, 10, 14, 17):
        _write(c, f"/home/user/mario-mobile/part{m}.js")      # progress all along
        assert c.request(m) == [], m
    c.step()
    acts = c.request(18.0)
    assert len(acts) == 1 and acts[0].reason.startswith("turn_budget(wall_clock 18.0"), acts
    assert "the user is waiting in a chat" in acts[0].text and "18 minutes" in acts[0].text
    c.step()
    assert c.request(20) == []                              # again only after again_minutes
    c.step()
    acts = c.request(22.5)
    assert acts and acts[0].reason.startswith("turn_budget(wall_clock again"), acts
    assert acts[0].text.startswith("Runtime note: 22 minutes. Answer the user now")
    ev = [e for e in events(mon, "S5") if e["decision"] == PROMPT_TAIL_NOTE]
    assert ev[0]["detail"]["arm"] == "wall_clock" and ev[0]["detail"]["hard"] is True


def test_a_chat_turn_without_progress_wraps_up_at_the_soft_cap():
    mon = s5()
    c = Chat(mon)
    c.request(0)
    _write(c)
    assert c.request(1) == []
    for m in (5, 9, 11.5):
        c.step(content="")
        assert c.request(m) == [], m
    c.step()
    acts = c.request(12.5)                                  # 11.5 min since the last write
    assert acts and "wall_clock 12.5" in acts[0].reason, acts
    ev = [e for e in events(mon, "S5") if e["decision"] == PROMPT_TAIL_NOTE]
    assert ev[-1]["detail"]["quiet"] is True and ev[-1]["detail"]["hard"] is False


def test_progress_is_not_cut():
    mon = s5()
    c = Chat(mon)
    c.request(0)
    for m in range(1, 18):
        _write(c, f"/home/user/app/file{m}.js")
        assert c.request(m) == [], m                        # writing all the time: no note


def test_changing_numbers_are_progress():
    mon = s5()
    c = Chat(mon, client="opencode", conv="ses_t")
    c.request(0)
    _write(c, "/home/u/p/calc.py")
    for i, n in enumerate((5, 3, 1, 1)):
        c.step("bash", {"command": "pytest -q"}, f"{n} failed",
               content=f"{n} tests still fail in test_parse.py, fixing the next one.")
        acts = c.request(1 + i)
        if i < 3:
            assert acts == [], (i, acts)
    # "1 tests still fail" twice is not three times: still no note
    assert acts == []


def test_opencode_has_no_wall_clock():
    mon = s5()
    c = Chat(mon, client="opencode", conv="ses_o")
    c.request(0)
    _write(c, "/home/u/p/index.html")
    for m in (5, 19, 30, 45):
        c.step()
        assert c.request(m) == [], m


def test_a_new_user_message_starts_a_new_turn():
    mon = s5()
    c = Chat(mon)
    c.request(0)
    _write(c)
    c.step(content=STUCK[0])
    c.request(1)
    c.step(content=STUCK[1])
    c.request(2)
    c.user("Now add a second level.")
    assert c.request(3) == []
    c.step(content=STUCK[2])                                # the old turn's leads are gone
    assert c.request(4) == []
    for m in (8, 12, 16):
        _write(c, f"/home/user/mario-mobile/level2-{m}.js")
        assert c.request(m) == [], m                        # 13 min into the new turn


# ------------------------------------------------------------------ SRV-181

def test_setup_before_the_first_file_gets_the_note():
    mon = s5()
    c = Chat(mon, client="opencode", conv="ses_pomo", text="Build a pomodoro page, one file.")
    c.request(0)
    setup = [("bash", {"command": "command -v node npx python3"}),
             ("bash", {"command": "firefox --headless --screenshot x.png file:///tmp/t.html"}),
             ("write", {"filePath": "/home/u/pomo/.tandem/test.html"}),
             ("bash", {"command": "python3 -m http.server 8765 > .tandem/server.log 2>&1 &"})]
    for i, (name, args) in enumerate(setup):
        c.step(name, args, "Error: RenderCompositorSWGL failed")
        assert c.request(0.5 + i * 0.5) == [], i
    c.step("bash", {"command": "firefox --headless --virtual-time-budget=3000"}, "timeout")
    acts = c.request(3.5)
    assert len(acts) == 1 and acts[0].reason.startswith("turn_budget(first_write 3.5"), acts
    assert "nothing the user asked for is written yet" in acts[0].text
    assert "5 tool steps" in acts[0].text
    c.step()
    assert c.request(4) == []                               # once a turn
    c.step("write", {"filePath": "/home/u/pomo/index.html"}, "Wrote file successfully")
    assert c.request(5) == []


def test_what_counts_as_a_write():
    t = policy(only=("S5",)).tasks["S5"]
    w = sessiontasks.written_paths
    assert w([("run_command", json.dumps({"command": "cat >> /home/user/m/game.js << 'EOF'\nx\nEOF"}))],
             t) == ["/home/user/m/game.js"]
    assert w([("run_command", json.dumps({"command": "mkdir -p a/.tandem && (echo '.tandem/' >> "
                                                      "\"$(git rev-parse --git-dir)/info/exclude\")"}))], t) == []
    assert w([("bash", json.dumps({"command": "npm init -y >/dev/null 2>&1"}))], t) == []
    assert w([("write", json.dumps({"filePath": "/p/index.html", "content": "x"}))], t) == \
        ["/p/index.html"]
    prod = [p for p in ("/p/.tandem/smoke.js", "/tmp/t.html", "/p/run.log", "/p/server.pid",
                        "/p/node_modules/x/y.js", "/p/index.html", "src/app.py")
            if sessiontasks._product(p, t)]
    assert prod == ["/p/index.html", "src/app.py"], prod


def test_a_debug_turn_has_no_first_write_note():
    mon = s5()
    c = Chat(mon, client="opencode", cls="debug", conv="ses_dbg", text="Fix the failing tests.")
    c.request(0)
    for m in range(1, 8):
        c.step("read", {"filePath": f"/p/mod{m}.py"}, "def f(): ...")
        assert c.request(m) == [], m


# ------------------------------------------------------------------ SRV-195

RUNS = os.path.join(HERE, "fixtures", "turn_budget_runs.json")


def _replay(name, cls="code_edit"):
    """A run of the fixture through S5 on its own timings: (the notes as (seconds, reason), the
    run's end in seconds)."""
    with open(RUNS) as f:
        run = json.load(f)[name]
    mon = s5()
    c = Chat(mon, client="opencode", cls=cls, conv=f"ses_{name}", text="Build it as asked.")
    c.request(0)
    notes = []
    for t, tool, args in run["steps"]:
        c.step(tool, args or {"command": "true"}, "ok")
        for a in c.request(t / 60.0):
            notes.append((t, a.reason, a.text))
    return notes, run["end_s"]


def test_a_rig_after_the_first_write_gets_the_verify_note():
    notes, end = _replay("pomo2")
    verify = [n for n in notes if "verify" in n[1]]
    print(f"    pomo2: {[(t, r) for t, r, _ in notes]} (run ended {end} s)")
    assert len(verify) == 1 and verify[0][0] < 500, notes
    assert "changed only test tooling" in verify[0][2] and "Stop building test tools" in verify[0][2]
    assert "kit's page check" in verify[0][2]
    notes, _ = _replay("ut1_todo")
    print(f"    ut1_todo: {[(t, r) for t, r, _ in notes]}")
    assert [n for n in notes if "verify" in n[1]], notes
    # the Mario chat's own rig (Open WebUI): the game written, then its playtest bot tuned in
    # .tandem -> the verify note comes before the third "stuck at x=436" would
    mon = s5()
    c = Chat(mon)
    c.request(0)
    _write(c)
    c.request(1)
    for i, lead in enumerate(STUCK[:2]):
        c.step("replace_file_content", {"path": "/home/user/mario-mobile/.tandem/playtest.js"},
               '{"size": 2443}', content="\n\n" + lead + "\n\n")
        assert c.request(1.5 + i * 0.5) == [], i
    c.step("run_command", {"command": "node .tandem/playtest.js"}, "max x reached: 436")
    acts = c.request(3)
    assert len(acts) == 1 and acts[0].reason.startswith("turn_budget(verify 2.0"), acts
    ev = [e for e in events(mon, "S5") if e["decision"] == PROMPT_TAIL_NOTE]
    assert ev[-1]["detail"]["arm"] == "verify"
    assert ev[-1]["detail"]["scratch_writes"] == 2 and ev[-1]["detail"]["since_steps"] == 3


def test_clean_runs_and_progress_get_no_verify_note():
    for name in ("ut2_bug", "ut2_refac", "ut1_bugfix", "ut1_dupfinder"):
        notes, end = _replay(name)
        print(f"    {name} ({end} s): {[(t, r) for t, r, _ in notes] or 'no note'}")
        assert not notes, (name, notes)
    # product edits keep the count at zero: 8 minutes of edit-and-run on the page itself
    mon = s5()
    c = Chat(mon, client="opencode", conv="ses_prog", text="Build a pomodoro page, one file.")
    c.request(0)
    c.step("write", {"filePath": "/p/index.html"}, "Wrote file successfully")
    assert c.request(1) == []
    for m in range(2, 10):
        c.step("bash", {"command": "python3 .tandem/check.py"}, "1 failed")
        c.step("write", {"filePath": "/p/.tandem/check.py"}, "ok")
        assert c.request(m - 0.5) == [], m
        c.step("edit", {"filePath": "/p/index.html"}, "Edit applied successfully")
        assert c.request(m) == [], m
    # a debug turn is not a build: no verify note however long its checks run
    mon = s5()
    c = Chat(mon, client="opencode", cls="debug", conv="ses_dbg2", text="Fix the failing test.")
    c.request(0)
    c.step("edit", {"filePath": "/p/app.py"}, "ok")
    c.request(0.5)
    for m in range(1, 9):
        c.step("write", {"filePath": f"/p/.tandem/t{m}.py"}, "ok")
        assert c.request(m) == [], m


# ------------------------------------------------------------------ the policy and the box

def test_policy_and_migration():
    p = policy_mod.load()
    assert p.tasks["S5"].rule == "turn_budget" and p.tasks["S5"].mode == "act"
    assert p.version == 8
    with open(policy_mod.DEFAULT_JSON) as f:
        assert json.load(f) == p.data, "policy.default.json is stale: tools/conductor_policy.py --json"
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import monitor_migrate
    old = json.loads(json.dumps(p.data))
    del old["tasks"]["S5"]
    old["version"] = 4
    old["tasks"]["D1"]["mode"] = "shadow"                  # the box's own edit stays
    new, ch = monitor_migrate.migrate_doc(old)
    assert ch == ["task S5 added (turn_budget, act)", "version 4 -> 8"], ch
    assert new["tasks"]["S5"] == p.data["tasks"]["S5"] and new["tasks"]["D1"]["mode"] == "shadow"
    again, ch2 = monitor_migrate.migrate_doc(new)
    assert ch2 == [] and again == new
    # A box at policy 6 has S5 without the verify arm, and its own threshold
    box = json.loads(json.dumps(p.data))
    for k in ("verify_s", "verify_scratch_writes", "verify_text"):
        del box["tasks"]["S5"][k]
    box["version"], box["tasks"]["S5"]["hard_minutes"] = 6, 25
    new, ch = monitor_migrate.migrate_doc(box)
    assert ch == ["task S5: verify_s, verify_scratch_writes, verify_text added",
                  "version 6 -> 8"], ch
    assert new["tasks"]["S5"]["hard_minutes"] == 25 and new["tasks"]["S5"]["verify_s"] == 90
    assert monitor_migrate.migrate_doc(new)[1] == []
    # a policy-6 document without the keys still runs: the arm stays silent (no text)
    policy_mod.Policy(box)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and only in name:
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
