"""X1 (2026-10-03): the playtest kit, `server/laya_kit/` (`laya_drive.py`, `laya-drive.js`).

Laya plays the software under test: each step the kit reads the program's state, asks Laya which
action comes next, applies it as input and logs it; `report` turns the log into the text Qwen
reads to fix the program. Here Laya is a fake HTTP server in this process (the route's shape: a
session per run, `/v1/laya/drive/end`), steered by a policy, so the loop and the report are
checked without the engine.

  1  the pieces: actions, compact states, positions, a UI's own choices, keys   test_1_*
  2  proc (JSON lines): a runner game; a policy that jumps at the gaps wins every episode, the log
     has every decision, the session's end carries the counts             test_2_proc_wins
  3  proc with a policy that never jumps: deaths at the gap, where (the last decision's place)
     and why, the decisions before each, deaths by place, the game over  test_3_proc_deaths
  4  a busy Laya (503 every third step): the fallback action, logged with why, counted
                                                                          test_4_busy_fallback
  5  stuck and the step budget: a policy that waits                       test_5_stuck_and_budget
  6  cmd (a state command and an action command) on a UI flow whose state lists its own actions:
     Laya picks among them, the flow completes                            test_6_cmd_ui_flow
  7  the report on a fixed log: the text, line by line                    test_7_report_text
  8  browser over the DevTools protocol (headless Chromium, when the machine has one): real key
     input (isTrusted) on a real-time page wins; the in-page kit (laya-drive.js) on the same page
     with synthetic keys, its log through `serve-log`, the report reads it
                                                                          test_8_browser_*
  9  the kit closes what it started: the game's group at a SIGTERM, a Chrome whose page never
     came up, a Chrome whose driver was killed (SIGKILL, Linux)          test_9_*
 10  `page`, the one-command check of a page (SRV-195): real clicks and keys change a timer
     page, the values read back, screenshots at 1280 and 390 px, PASS; a broken page names its
     console error, its uncaught exception, the step that found no element and its overflow at
     390 px, FAIL (exit 1); no browser on the machine says so and tells the model not to build
     one (exit 2); a check past its time limit ends with Chrome closed      test_10_page_check

Run: python tests/test_laya_drive_kit.py
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT = os.path.join(ROOT, "server", "laya_kit")
sys.path.insert(0, KIT)

import laya_drive as ld  # noqa: E402

RUNNER = textwrap.dedent('''
    import json, sys
    GAPS = [(30, 32), (60, 62)]
    GOAL = 90
    def new():
        return {"x": 0, "air": 0, "lives": 3, "score": 0}
    def show(s):
        nxt = [a for a, b in GAPS if a > s["x"]]
        out = dict(s, gap_ahead=(nxt[0] - s["x"]) if nxt else None,
                   over=s["lives"] <= 0, won=s["x"] >= GOAL)
        print(json.dumps(out), flush=True)
    s = new()
    show(s)
    for line in sys.stdin:
        a = line.strip()
        if a == "reset":
            s = new(); show(s); continue
        if a in ("run", "run_jump"):
            s["x"] += 1
        if a == "back":
            s["x"] -= 1
        if a in ("jump", "run_jump") and s["air"] == 0:
            s["air"] = 3
        elif s["air"] > 0:
            s["air"] -= 1
        s.pop("cause", None)
        if s["air"] == 0 and any(a <= s["x"] <= b for a, b in GAPS):
            s["lives"] -= 1
            s["cause"] = "fell into the gap at x=%d" % s["x"]
            s["x"] = max(0, s["x"] - 3)
        s["score"] = s["x"]
        show(s)
''')


def smart(choices, state):
    """Jump at the gap's edge, run otherwise."""
    if "run_jump" in choices and state.get("air", 0) == 0 and state.get("gap_ahead") in (1,):
        return "run_jump"
    if "todo" in state and state["todo"] in choices:
        return state["todo"]
    return "run" if "run" in choices else next(iter(choices))


class FakeLaya:
    """The Laya route's shape: /v1/laya/ask (a session per run) and /v1/laya/drive/end."""

    def __init__(self, policy=smart, busy_every=0):
        self.policy, self.busy_every = policy, busy_every
        self.calls, self.ended = [], []
        me = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code, obj, extra=()):
                data = json.dumps(obj).encode()
                self.send_response(code)
                for k, v in (("Content-Type", "application/json"), ("Content-Length", str(len(data))),
                             ("Access-Control-Allow-Origin", "*"), *extra):
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path.endswith("/v1/laya/drive/end"):
                    me.ended.append(body)
                    return self._send(200, {"session": body["session"], "ended": "end",
                                            "ms_p50": 1.0, "ms_p95": 2.0, "unsure": 0, "errors": {}})
                me.calls.append(body)
                if me.busy_every and len(me.calls) % me.busy_every == 0:
                    return self._send(503, {"error": {"type": "busy", "message": "busy"}},
                                      (("Retry-After", "1"),))
                state = json.loads(body["text"])
                label = me.policy(body["choices"], state)
                return self._send(200, {"label": label, "p": 0.93, "sure": True, "ms": 0.5})

            def log_message(self, *a):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def kit(d):
    """The kit copied into the run's scratch directory, as the playtest package has the model do
    (SRV-171: `{tandem_scratch}`, outside the project): its log is written beside it."""
    sd = os.path.join(d, "scratch")
    os.makedirs(sd, exist_ok=True)
    dst = os.path.join(sd, "laya_drive.py")
    if not os.path.exists(dst):
        shutil.copy(os.path.join(KIT, "laya_drive.py"), dst)
    return dst


def run_cli(args, laya, cwd):
    r = subprocess.run([sys.executable, kit(cwd), *args, "--url", laya.url],
                       cwd=cwd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def game_dir():
    d = tempfile.mkdtemp(prefix="drive-kit-")
    with open(os.path.join(d, "runner.py"), "w") as f:
        f.write(RUNNER)
    return d


def log_rows(d):
    assert not os.path.exists(os.path.join(d, ".tandem")), "SRV-171: nothing in the project"
    return [json.loads(ln) for ln in open(os.path.join(d, "scratch", "drive-log.jsonl"))]


ACTIONS = "run=move right,jump=jump straight up,run_jump=jump forward,back=move left,wait=stand still"


# ------------------------------------------------------------------ 1
def test_1_pieces():
    assert ld.parse_actions("run=move right,jump") == {"run": "move right", "jump": "jump"}
    assert ld.parse_actions(["a", "b=bee"]) == {"a": "a", "b": "bee"}
    big = {"player": {"x": 1.234567, "y": None}, "enemies": [{"x": i * 1.11, "d": i} for i in range(200)]}
    c = ld.compact(big, 300)
    assert len(c) <= 300 and '"x":1.2' in c and "null" not in c, c
    assert ld.position({"x": 3.14159, "y": 2}) == (3.1, 2)
    assert ld.position({"player": {"x": 5, "y": 7}}) == (5, 7)
    assert ld.position({"pos": [1, 2, 3]}) == (1, 2) and ld.position({"a": 1}) is None
    ui = {"controls": [{"id": "save", "label": "Save the form"}, {"id": "cancel"}]}
    assert ld.choices_of(ui, {"run": "x"}) == {"save": "Save the form", "cancel": "cancel"}
    assert ld.choices_of({"actions": ["a", "run"]}, {"run": "move"}) == {"a": "a", "run": "move"}
    assert ld.key_event("Space")["code"] == "Space" and ld.key_event("a")["code"] == "KeyA"
    assert ld.key_event("ArrowRight")["windowsVirtualKeyCode"] == 39
    assert ld.lost({"over": True}) and ld.won({"won": True}) and not ld.lost({"over": False})


# ------------------------------------------------------------------ 2
def test_2_proc_wins():
    laya, d = FakeLaya(), game_dir()
    try:
        out = run_cli(["proc", f"{sys.executable} runner.py", "--actions", ACTIONS,
                       "--episodes", "3", "--steps", "200", "--label", "build 1"], laya, d)
    finally:
        laya.close()
    rows = log_rows(d)
    wins = [r for r in rows if r.get("event") == "win"]
    assert len(wins) == 3 and not [r for r in rows if r.get("event") == "death"], out
    assert "3 episodes" in out and "3 won, 0 deaths" in out and "build 1" in out, out
    steps = [r for r in rows if "a" in r]
    assert len(steps) == len(laya.calls) and steps[0]["a"] == "run" and "run_jump" in {r["a"] for r in steps}
    assert all(c["session"] == laya.calls[0]["session"] for c in laya.calls)
    assert all(set(c["choices"]) == {"run", "jump", "run_jump", "back", "wait"} for c in laya.calls)
    assert laya.calls[0]["choices"]["run_jump"] == "jump forward"
    assert laya.ended[0]["summary"]["wins"] == 3 and laya.ended[0]["summary"]["steps"] == len(steps)
    assert rows[-1]["end"] and rows[-1]["summary"]["episodes"] == 3


# ------------------------------------------------------------------ 3
def test_3_proc_deaths():
    laya, d = FakeLaya(policy=lambda ch, s: "run"), game_dir()
    try:
        out = run_cli(["proc", f"{sys.executable} runner.py", "--actions", ACTIONS,
                       "--episodes", "1", "--steps", "200"], laya, d)
    finally:
        laya.close()
    rows = log_rows(d)
    deaths = [r for r in rows if r.get("event") == "death"]
    assert len(deaths) == 3, deaths                    # two lives lost, then the game over
    assert deaths[0]["cause"] == "fell into the gap at x=30" and deaths[0]["pos"] == [29, None], deaths[0]
    assert "1 episodes" in out and "0 won, 3 deaths" in out, out
    assert "death 1 (episode 1, step 30) at x=29: fell into the gap at x=30" in out, out
    assert "   step 30: run p=0.93" in out, out
    assert "deaths by place: x=29 x3" in out, out
    assert "actions: run 100%" in out


# ------------------------------------------------------------------ 4
def test_4_busy_fallback():
    laya, d = FakeLaya(busy_every=3), game_dir()
    try:
        out = run_cli(["proc", f"{sys.executable} runner.py", "--actions", ACTIONS,
                       "--episodes", "1", "--steps", "300"], laya, d)
    finally:
        laya.close()
    steps = [r for r in log_rows(d) if "a" in r]
    fb = [r for r in steps if r.get("why")]
    assert fb and all(r["why"] == "busy" and r["a"] == "wait" for r in fb), fb[:3]
    assert len(fb) == len(steps) // 3, (len(fb), len(steps))
    assert laya.ended[0]["summary"]["busy"] == len(fb) == laya.ended[0]["summary"]["fallbacks"]
    assert f"{len(fb)} fallbacks (busy)" in out, out


# ------------------------------------------------------------------ 5
def test_5_stuck_and_budget():
    laya, d = FakeLaya(policy=lambda ch, s: "wait"), game_dir()
    try:
        out = run_cli(["proc", f"{sys.executable} runner.py", "--actions", ACTIONS,
                       "--episodes", "2", "--steps", "30", "--stuck", "10"], laya, d)
    finally:
        laya.close()
    rows = log_rows(d)
    stuck = [r for r in rows if r.get("event") == "stuck"]
    assert len(stuck) == 2 and stuck[0]["pos"] == [0, None] and stuck[0]["steps"] == 10, stuck
    assert [r["event"] for r in rows if r.get("event") in ("timeout", "win", "death")] == ["timeout"] * 2
    assert "2 hit the step budget" in out and "stuck (episode 1, step 10) at x=0 for 10 steps" in out, out


# ------------------------------------------------------------------ 6
UI_STATE = textwrap.dedent('''
    import json, os, sys
    P = "ui.json"
    FLOW = {"login": ("type_user", "user"), "user": ("type_pass", "pass"), "pass": ("submit", "done")}
    s = json.load(open(P)) if os.path.exists(P) else {"screen": "login", "errors": 0}
    if len(sys.argv) > 1:
        want, nxt = FLOW.get(s["screen"], (None, None))
        if sys.argv[1] == want:
            s["screen"] = nxt
        else:
            s["errors"] += 1
        json.dump(s, open(P, "w"))
    else:
        acts = ["cancel", "type_user", "type_pass", "submit"]
        print(json.dumps(dict(s, actions=acts, todo=FLOW.get(s["screen"], ("", ""))[0],
                              won=s["screen"] == "done")))
''')


def test_6_cmd_ui_flow():
    laya, d = FakeLaya(), tempfile.mkdtemp(prefix="drive-ui-")
    with open(os.path.join(d, "ui.py"), "w") as f:
        f.write(UI_STATE)
    try:
        out = run_cli(["cmd", "--state", f"{sys.executable} ui.py", "--act",
                       f"{sys.executable} ui.py {{action}}", "--reset", "rm -f ui.json",
                       "--episodes", "2", "--steps", "10"], laya, d)
    finally:
        laya.close()
    rows = log_rows(d)
    assert [r["a"] for r in rows if "a" in r] == ["type_user", "type_pass", "submit"] * 2, rows
    assert set(laya.calls[0]["choices"]) == {"cancel", "type_user", "type_pass", "submit"}
    assert "2 won, 0 deaths" in out, out


# ------------------------------------------------------------------ 7
def test_7_report_text():
    d = tempfile.mkdtemp(prefix="drive-rep-")
    p = os.path.join(d, "log.jsonl")
    rows = [{"run": "old", "at": "x"}, {"ep": 1, "step": 1, "a": "run", "s": "{}", "ms": 9},
            {"run": "r1", "at": "2026-10-03 11:00:00", "label": "fix 2"},
            {"ep": 1, "step": 1, "t": 0, "s": '{"x":1}', "a": "run", "p": 0.95, "sure": True, "ms": 20.0},
            {"ep": 1, "step": 2, "t": 250, "s": '{"x":2}', "a": "jump", "p": 0.55, "sure": False, "ms": 30.0},
            {"ep": 1, "step": 3, "t": 500, "s": '{"x":3}', "a": "wait", "p": None, "sure": None, "ms": 900.0, "why": "busy"},
            {"ep": 1, "step": 3, "t": 520, "event": "death", "s": '{"x":0}', "pos": [3, 1], "cause": "spikes"},
            {"ep": 1, "step": 3, "t": 520, "event": "death", "s": '{"x":0}', "pos": [3, 1], "cause": None},
            {"end": "r1", "summary": {}}]
    with open(p, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    got = ld.report(p, last=2).splitlines()
    assert got == [
        "playtest r1 (2026-10-03 11:00:00, fix 2): 1 episodes, 3 steps; 0 won, 2 deaths, 0 hit the step budget",
        "Laya: 30 ms p50, 900 ms p95 a step; not sure on 1 of 3; 1 fallbacks (busy)",
        "actions: run 33%, jump 33%, wait 33%",
        "death 1 (episode 1, step 3) at x=3 y=1: spikes",
        '   step 2: jump p=0.55 (unsure)  {"x":2}',
        '   step 3: wait [busy]  {"x":3}',
        "death 2 (episode 1, step 3) at x=3 y=1",
        '   step 2: jump p=0.55 (unsure)  {"x":2}',
        '   step 3: wait [busy]  {"x":3}',
        "deaths by place: x=3 y=1 x2"], got


# ------------------------------------------------------------------ 8 browser
PAGE = textwrap.dedent('''
    <!doctype html><title>runner</title><canvas id=c width=200 height=50></canvas>
    <script>
    const GAP = [40, 45], GOAL = 100;
    let s, held = new Set(), trusted = 0, untrusted = 0;
    function reset() { s = {x: 0, air: 0, lives: 2, over: false, won: false}; }
    reset();
    addEventListener("keydown", (e) => { (e.isTrusted ? trusted++ : untrusted++);
      if ((e.key === " " || e.code === "Space") && s.air === 0 && !held.has("Space")) s.air = 14;
      held.add(e.code === "Space" ? "Space" : e.key); });
    addEventListener("keyup", (e) => held.delete(e.code === "Space" ? "Space" : e.key));
    setInterval(() => {
      if (s.over || s.won) return;
      if (held.has("ArrowRight")) s.x += 1;
      if (s.air > 0) s.air -= 1;
      if (s.air === 0 && s.x >= GAP[0] && s.x <= GAP[1]) {
        s.lives -= 1; s.cause = "fell into the gap"; s.x = GAP[0] - 6; held.clear();
        if (s.lives <= 0) s.over = true;
      }
      if (s.x >= GOAL) s.won = true;
    }, 25);
    window.tandem_state = () => ({x: s.x, air: s.air, lives: s.lives, over: s.over, won: s.won,
      gap_ahead: s.x < GAP[0] ? GAP[0] - s.x : null, cause: s.cause, trusted, untrusted});
    window.tandem_reset = () => reset();
    </script>
''')


def page_policy(choices, s):
    if s.get("air", 0) == 0 and s.get("gap_ahead") is not None and s["gap_ahead"] <= 6:
        return "run_jump"
    return "run"


def _serve_dir(d):
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=d, **k)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_7b_an_exception_says_where_it_was_thrown():
    """Pagoda replay 3 (2026-10-17): the model read "Cannot read properties of undefined" 13 times
    without opening the line; the page check names the script and the 1-based line."""
    ev = [{"method": "Runtime.exceptionThrown", "params": {"exceptionDetails": {
        "text": "Uncaught", "lineNumber": 61, "url": "http://localhost:5174/js/scene/pond.js",
        "exception": {"description": "TypeError: Cannot read properties of undefined (reading "
                                     "'set')\n    at pond.js:62:22"}}}},
          {"method": "Runtime.exceptionThrown", "params": {"exceptionDetails": {
              "text": "Uncaught", "stackTrace": {"callFrames": [
                  {"url": "http://localhost:5174/js/main.js", "lineNumber": 9}]},
              "exception": {"description": "ReferenceError: x is not defined"}}}},
          {"method": "Runtime.exceptionThrown", "params": {"exceptionDetails": {
              "text": "Uncaught SyntaxError"}}}]
    errs, _ = ld._problems(ev)
    assert errs == ["exception: TypeError: Cannot read properties of undefined (reading 'set') "
                    "(http://localhost:5174/js/scene/pond.js:62)",
                    "exception: ReferenceError: x is not defined (http://localhost:5174/js/"
                    "main.js:10)",
                    "exception: Uncaught SyntaxError"], errs


def test_7c_a_shader_error_keeps_the_compiler_lines():
    """Ornith's reef replay (2026-10-21): three.js's shader error came back clipped after its first
    `ERROR:` line, and the model read the library for 14 calls. The report keeps the first line,
    every compiler error line and the marked source line, without the source dump; other errors
    are clipped as before."""
    dump = "\n".join(f"{i}: float x{i} = 0.0;" for i in range(1700, 1766))
    msg = ("THREE.WebGLProgram: Shader Error 0 - VALIDATE_STATUS false\n\nMaterial Name: \n"
           "Material Type: MeshStandardMaterial\n\nProgram Info Log: Fragment shader is not "
           "compiled.\n\nFRAGMENT\n\nERROR: 0:1768: 'vReefW' : undeclared identifier\n"
           "ERROR: 0:1771: 'uTime' : undeclared identifier\n\n" + dump +
           "\n> 1768: \tfloat r = vReefW.y * 0.5;\n1769: }")
    ev = [{"method": "Runtime.consoleAPICalled", "params": {"type": "error", "args": [
              {"type": "string", "value": msg}]}},
          {"method": "Runtime.consoleAPICalled", "params": {"type": "error", "args": [
              {"type": "string", "value": "x" * 400}]}}]
    errs, _ = ld._problems(ev)
    assert errs[0] == ("console: THREE.WebGLProgram: Shader Error 0 - VALIDATE_STATUS false\n"
                       "ERROR: 0:1768: 'vReefW' : undeclared identifier\n"
                       "ERROR: 0:1771: 'uTime' : undeclared identifier\n"
                       "> 1768: \tfloat r = vReefW.y * 0.5;"), errs[0]
    assert errs[1] == "console: " + "x" * 231 + "...", errs[1]


def test_8_browser_cdp_and_the_in_page_kit():
    if ld.find_chrome() is None:
        print("    (skipped: no Chrome or Chromium on this machine)")
        return
    d = tempfile.mkdtemp(prefix="drive-page-")
    with open(os.path.join(d, "index.html"), "w") as f:
        f.write(PAGE)
    srv, url = _serve_dir(d)
    laya = FakeLaya(policy=page_policy)
    try:
        t0 = time.monotonic()
        out = run_cli(["browser", url + "/index.html", "--keys",
                       "run=ArrowRight,run_jump=ArrowRight+Space,wait=", "--every-ms", "100",
                       "--episodes", "2", "--steps", "150"], laya, d)
        took = time.monotonic() - t0
        rows = log_rows(d)
        ends = [r["event"] for r in rows if r.get("event") in ("win", "death", "timeout")]
        last = json.loads([r for r in rows if "a" in r][-1]["s"])
        print(f"    browser (CDP, real keys): {ends}, {took:.1f} s; {out.splitlines()[0]}")
        assert ends.count("win") >= 1, out
        steps = [r for r in rows if "a" in r]
        trusted = max(json.loads(r["s"]).get("trusted", 0) for r in steps)
        assert trusted > 0 and all(json.loads(r["s"]).get("untrusted", 0) == 0 for r in steps), last
        # the in-page kit: synthetic keys, its log through the sink, the same report
        port = _free_port()
        sink_log = os.path.join(d, "page-log.jsonl")
        sink = subprocess.Popen([sys.executable, os.path.join(KIT, "laya_drive.py"), "serve-log",
                                 "--port", str(port), "--log", sink_log], stdout=subprocess.PIPE,
                                text=True)
        sink.stdout.readline()
        b = ld.Browser(url + "/index.html")
        try:
            b.cdp.evaluate(open(os.path.join(KIT, "laya-drive.js")).read())
            summary = b.cdp.evaluate("tandem_drive.start(" + json.dumps({
                "url": laya.url, "logUrl": f"http://127.0.0.1:{port}/log", "everyMs": 100,
                "episodes": 1, "steps": 150,
                "actions": {"run": "move right", "run_jump": "jump forward", "wait": "stand still"},
                "keys": {"run": ["ArrowRight"], "run_jump": ["ArrowRight", " "], "wait": []}}) + ")")
            st = b.state()
        finally:
            b.close()
            sink.terminate()
        print(f"    in-page kit (synthetic keys): {summary}")
        assert summary["steps"] > 5 and summary["wins"] + len(summary["deaths"]) >= 1, summary
        assert st["untrusted"] > 0, st
        rep = ld.report(sink_log)
        assert rep.startswith("playtest drive-page-") and "1 episodes" in rep, rep
        assert laya.ended[-1]["session"].startswith("drive-page-")
    finally:
        laya.close()
        srv.shutdown()


def _group_alive(pg: int) -> bool:
    try:
        os.killpg(pg, 0)
        return True
    except OSError:
        return False


def _gone(pred, timeout=10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.1)
    return pred()


def _pids_with(text: str) -> list[int]:
    out = []
    for pid in os.listdir("/proc") if os.path.isdir("/proc") else ():
        if pid.isdigit():
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    if text.encode() in f.read():
                        out.append(int(pid))
            except OSError:
                pass
    return out


def test_9_the_kit_closes_what_it_started():
    """2026-10-04: 42 Chrome processes of four drives had run for a day on the box, their driver
    gone. A page that never comes up closes its Chrome; a driver killed by SIGTERM closes the game
    and the Chrome (the `finally`s run); one killed by SIGKILL takes its Chrome along on Linux
    (the parent-death signal)."""
    # proc: SIGTERM to the driver ends the game's whole group
    d = game_dir()
    laya = FakeLaya(policy=lambda c, s: "wait" if "wait" in c else next(iter(c)))
    try:
        drv = subprocess.Popen([sys.executable, kit(d), "proc",
                                f"{sys.executable} runner.py", "--actions", ACTIONS, "--steps",
                                "100000", "--episodes", "1", "--every-ms", "50", "--url", laya.url],
                               cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert _gone(lambda: len(laya.calls) >= 3), "the drive never asked Laya"
        games = [p for p in _pids_with("runner.py") if p != drv.pid]
        drv.terminate()
        drv.wait(15)
        if os.path.isdir("/proc"):
            assert games and _gone(lambda: not [p for p in games if os.path.exists(f"/proc/{p}")
                                                and b"runner.py" in open(f"/proc/{p}/cmdline",
                                                                         "rb").read()]), games
    finally:
        laya.close()
    if ld.find_chrome() is None:
        print("    (browser part skipped: no Chrome or Chromium on this machine)")
        return
    # a page without window.tandem_state(): SystemExit, its Chrome and profile gone
    pd = tempfile.mkdtemp(prefix="drive-page-")
    with open(os.path.join(pd, "blank.html"), "w") as f:
        f.write("<html><body>no state here</body></html>")
    with open(os.path.join(pd, "index.html"), "w") as f:
        f.write(PAGE)
    srv, url = _serve_dir(pd)
    try:
        before = set(_pids_with("laya-drive-chrome-"))
        try:
            ld.Browser(url + "/blank.html")
            raise AssertionError("a page without tandem_state came up")
        except SystemExit:
            pass
        assert _gone(lambda: not (set(_pids_with("laya-drive-chrome-")) - before)), \
            set(_pids_with("laya-drive-chrome-")) - before
        if not sys.platform.startswith("linux"):
            return
        # SIGKILL to the driver in the middle of a browser drive: Chrome follows it
        laya = FakeLaya(policy=page_policy)
        try:
            drv = subprocess.Popen([sys.executable, kit(pd), "browser",
                                    url + "/index.html", "--keys", "run=ArrowRight,wait=",
                                    "--every-ms", "100", "--episodes", "50", "--steps", "100000",
                                    "--url", laya.url], cwd=pd, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
            assert _gone(lambda: len(laya.calls) >= 3, 60), "the browser drive never asked Laya"
            mine = set(_pids_with("laya-drive-chrome-")) - before
            assert mine
            drv.kill()
            drv.wait(10)
            assert _gone(lambda: not (set(_pids_with("laya-drive-chrome-")) - before), 15), \
                set(_pids_with("laya-drive-chrome-")) - before
            print(f"    SIGKILL to the driver: its {len(mine)} Chrome processes gone")
        finally:
            laya.close()
    finally:
        srv.shutdown()


TIMER_PAGE = """<!doctype html><html><head><meta name="viewport" content="width=device-width">
<style>body{font:16px sans-serif;margin:0;padding:16px}</style></head><body>
<div id="t">25:00</div><button id="start">Start</button><button id="pause" disabled>Pause</button>
<input id="name"><script>
let left = 1500, run = null, keys = 0;
const show = () => document.getElementById('t').textContent =
  String(Math.floor(left / 60)).padStart(2, '0') + ':' + String(Math.floor(left % 60)).padStart(2, '0');
document.getElementById('start').onclick = (e) => { if (!e.isTrusted) return; run = setInterval(() => { left -= 0.1; show(); }, 100);
  document.getElementById('pause').disabled = false; };
document.getElementById('pause').onclick = () => { clearInterval(run); run = null; };
addEventListener('keydown', (e) => { if (e.isTrusted && e.code === 'Space') keys++; });
window.tandem_state = () => ({tandem_state: 1, running: !!run, left: Math.round(left), keys});
</script></body></html>"""

BROKEN_PAGE = """<!doctype html><html><head><meta name="viewport" content="width=device-width"></head>
<body><div style="width:900px">too wide</div><img src="missing.png"><script>
console.error('boom: the store is empty'); setTimeout(() => { undefinedThing.call(); }, 10);
</script></body></html>"""


def _page(url_or_args, *extra, env=None, laya=None):
    args = [sys.executable, os.path.join(KIT, "laya_drive.py"), "page", *url_or_args, *extra,
            *(["--url", laya.url] if laya is not None else ["--no-laya"])]
    p = subprocess.run(args, capture_output=True, text=True, timeout=180, env=env)
    return p.returncode, json.loads(p.stdout)


def test_10_page_check():
    if ld.find_chrome() is None:
        print("    (skipped: no Chrome or Chromium on this machine)")
        return
    d = tempfile.mkdtemp(prefix="drive-page-check-")
    for name, body in (("timer.html", TIMER_PAGE), ("broken.html", BROKEN_PAGE)):
        with open(os.path.join(d, name), "w") as f:
            f.write(body)
    srv, url = _serve_dir(d)
    before = set(_pids_with("laya-drive-chrome-"))
    try:
        t0 = time.monotonic()
        code, rep = _page([url + "/timer.html", "--shots", d, "--step", "click:text=Start",
                           "--step", "wait:1200", "--step", "read:document.getElementById('t').textContent",
                           "--step", "key:Space", "--step", "type:#name=Ada",
                           "--step", "expect:document.getElementById('name').value === 'Ada'",
                           "--step", "click:#pause", "--step", "shot:after-pause"])
        took = time.monotonic() - t0
        print(f"    timer page: {rep['verdict']} in {took:.1f} s; read {rep['steps'][2]['value']}, "
              f"state {rep.get('state')}")
        assert code == 0 and rep["ok"], rep
        assert rep["steps"][2]["value"] in ("24:59", "24:58"), rep["steps"][2]
        assert rep["state"]["keys"] == 1 and rep["state"]["left"] < 1500, rep["state"]
        assert [w["width"] for w in rep["widths"]] == [1280, 390]
        assert not any(w["overflow_x"] for w in rep["widths"]), rep["widths"]
        for w in rep["widths"]:
            assert os.path.getsize(os.path.join(os.getcwd(), w["shot"])) > 1000, w
        assert os.path.exists(os.path.join(d, "after-pause.png"))
        assert {c.get("id") for c in rep["controls"]} >= {"start", "pause", "name"}, rep["controls"]
        assert took < 30, took                  # the keyring wait (25 s per launch) is gone

        laya = FakeLaya(policy=lambda c, st: "failure")
        try:
            code, rep = _page([url + "/broken.html", "--shots", d, "--step", "click:text=Save",
                               "--step", "read:1"], laya=laya)
        finally:
            laya.close()
        assert rep["laya"]["label"] == "failure", rep.get("laya")
        asked = laya.calls[-1]
        assert asked["question"] == "Does this page check show a failure?" and "boom" in asked["text"], asked
        print(f"    broken page: {rep['verdict']}; errors {rep['errors']}")
        assert code == 1 and not rep["ok"]
        errs = " ".join(rep["errors"])
        assert "boom: the store is empty" in errs and "undefinedThing" in errs and "missing.png" in errs, errs
        assert rep["steps"][0]["ok"] is False and "no element" in rep["steps"][0]["error"]
        assert len(rep["steps"]) == 1, "a step that could not run ends the script"
        assert rep["widths"][1]["overflow_x"] is True and rep["widths"][0]["overflow_x"] is False

        # no browser here: the verdict says so and tells the model not to build one
        env = dict(os.environ, HOME=d, PATH="/nonexistent", CHROME="")
        code, rep = _page([url + "/timer.html"], env=env)
        assert code == 2 and rep["verdict"].startswith("NO BROWSER") and "Do not build" in rep["verdict"], rep

        # past the time limit: a failed check, and its Chrome is closed
        code, rep = _page([url + "/timer.html", "--shots", d, "--timeout", "3", "--step", "wait:8000"])
        assert code == 2 and rep["verdict"].startswith("CHECK FAILED") and "3 s" in rep["verdict"], rep
        assert _gone(lambda: not (set(_pids_with("laya-drive-chrome-")) - before), 15), \
            set(_pids_with("laya-drive-chrome-")) - before
    finally:
        srv.shutdown()
        shutil.rmtree(d, ignore_errors=True)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
