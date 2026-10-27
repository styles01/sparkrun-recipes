"""OPS-43 / OPS-44: the eval harness, tools/tandem_eval.py.

The test matrix is OPS-43's Gherkin plus the schema check:

  1  both arms on the same build and night       test_a_night_end_to_end_records_both_arms
  2  alternation and isolation                    test_the_schedule_*, test_the_store_is_cleared_*
  3  checkers stay pristine                       test_pristine_*
  4  the indicative label                         test_n_below_ten_is_indicative (and the Compare
                                                  view's fixture in dashboard/tests/conductor.test.ts)
  5  the raw row                                  test_the_raw_row_*
  -  report.json validates                        test_the_report_validates_*, the example file,
                                                  and POST/GET /v1/conductor/evals over HTTP

The end-to-end night runs the real harness against a fake `opencode` (a script that answers
`run` and `export` like opencode 1.18.33 and solves the task only on the TandemLLM arm) and a fake
engine API in this process; the HTTP test starts the server with the fake engine.

Run: python tests/test_tandem_eval.py [name-filter]
"""

from __future__ import annotations

import copy
import json
import os
import stat
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools import contract_check, tandem_eval as te  # noqa: E402

EXAMPLE = os.path.join(ROOT, "docs", "contract", "dashboard-v1", "tandem-eval.example.json")
BUILD = {"version": "0.3.2-test", "code_sha": "abc123def456"}

FAKE_OPENCODE = r'''#!/usr/bin/env python3
"""A stand-in for opencode 1.18.33's `run --format json` and `export` (tests/test_tandem_eval.py).
The TandemLLM arm writes the right answer and runs the checker; the raw arm writes a wrong one,
runs the checker once and stops (the driver would continue)."""
import json, os, subprocess, sys, time, secrets
STORE = os.environ["FAKE_OC_STORE"]
args = sys.argv[1:]
if args[0] == "export":
    print("Exporting session: " + args[1], file=sys.stderr)
    print(open(os.path.join(STORE, args[1] + ".json")).read())
    sys.exit(0)
cfg = json.load(open("opencode.json"))
headers = cfg["provider"]["tandem-eval"]["options"]["headers"]
raw = headers.get("X-Tandem") == "raw"
with open(os.path.join(STORE, "calls.jsonl"), "a") as f:
    f.write(json.dumps({"cwd": os.getcwd(), "headers": headers, "argv": args,
                        "key": os.environ.get("TANDEM_EVAL_KEY")}) + "\n")
sid = "ses_" + ("raw" if raw else "tan") + secrets.token_hex(6)
t0 = int(time.time() * 1000)
print(json.dumps({"type": "step_start", "sessionID": sid}))
if os.environ.get("FAKE_OC_TAMPER"):
    open("check.py", "a").write("\n# changed by the model\n")
open("answer.txt", "w").write("41" if raw else "42")
out = subprocess.run([sys.executable, "check.py"], capture_output=True, text=True).stdout
t1 = int(time.time() * 1000) + 60000
parts = [{"type": "tool", "tool": "bash", "callID": "c1",
          "state": {"status": "completed", "input": {"command": "python3 check.py"},
                    "output": out, "time": {"start": t1 - 100, "end": t1}}}]
msgs = [{"info": {"role": "user", "time": {"created": t0}}, "parts": []},
        {"info": {"role": "assistant", "finish": "tool-calls", "time": {"created": t0, "completed": t1},
                  "tokens": {"input": 1000, "output": 200, "reasoning": 800,
                             "cache": {"read": 3000, "write": 0}}}, "parts": parts},
        {"info": {"role": "assistant", "finish": "stop", "time": {"created": t1, "completed": t1 + 1000},
                  "tokens": {"input": 10, "output": 0, "reasoning": 50, "cache": {"read": 4000}}},
         "parts": [] if raw else [{"type": "text", "text": "Done: RESULT: PASS"}]}]
json.dump({"info": {"id": sid}, "messages": msgs}, open(os.path.join(STORE, sid + ".json"), "w"))
print(json.dumps({"type": "step_finish", "sessionID": sid}))
'''

CHECK = '''import os, sys
print("taskcheck answer")
a = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer.txt")).read().strip() \\
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer.txt")) else ""
if a == "42":
    print("RESULT: PASS")
else:
    print(f"RESULT: FAIL <- (a) answer off by {abs(42 - int(a or 0))} > 0")
    sys.exit(1)
'''


class FakeEngine:
    """The engine API the harness reads and writes, in this process."""

    def __init__(self, guard=True):
        self.calls: list[tuple[str, str]] = []
        self.published = []
        self.guard = guard
        eng = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                eng.calls.append(("GET", self.path))
                if self.path.startswith("/v1/conductor/evals"):
                    reps = [{"kind": "speed_guard", "file": "guard.json", "build": BUILD,
                             "verdict": "pass"}] if eng.guard else []
                    return self._send(200, {"contract_version": "1.0", "reports": reps,
                                            "engine": BUILD})
                if self.path.startswith("/v1/dashboard/requests"):
                    import time
                    ts = te.now_iso(time.time())
                    return self._send(200, {"requests": [
                        {"ts": ts, "client": {"kind": "opencode"}, "decode_tps": 44.0,
                         "prefill_tps": 1500.0, "completion_tokens": 1000, "reasoning_tokens": 800,
                         "prompt_tokens": 4000, "cached_tokens": 3000, "finish_reason": "tool_calls"}],
                        "next_before": None})
                if self.path.startswith("/v1/conductor/decisions"):
                    # the router's rows of the session (SRV-157's columns): one small turn at
                    # high, then xhigh once the task grew; the raw arm has none
                    if "raw" in self.path:
                        return self._send(200, {"decisions": []})
                    size = lambda z, e: {"size": {"mode": "on", "size": z, "score": 0, "now": z,
                                                  "signals": {}, "ceiling": "xhigh",
                                                  "effort": e, "applied": True}}
                    return self._send(200, {"decisions": [
                        {"kind": "route", "effort": "xhigh", "extra": size("medium", "xhigh")},
                        {"kind": "route", "effort": "high",
                         "extra": dict(size("small", "high"),
                                       outcome={"label": "error", "mode": "shadow"})}]})
                if self.path.startswith("/v1/conductor/events"):
                    return self._send(200, {"events": [
                        {"mode": "act", "decision": "PROMPT_TAIL_NOTE", "task": "S1"},
                        {"mode": "shadow", "decision": "REGENERATE_CALL", "task": "C1"},
                        {"mode": "act", "decision": "PASS", "task": "SP3"}], "monitor": True})
                return self._send(404, {"error": {"message": "no"}})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                eng.calls.append(("POST", self.path))
                if self.headers.get("Authorization") != "Bearer adm":
                    return self._send(401, {"error": {"message": "token"}})
                if self.path == "/v1/cache/clear":
                    return self._send(200, {"cleared": True})
                if self.path == "/v1/conductor/evals":
                    eng.published.append(body["report"])
                    return self._send(201, {"file": "suite-x.json"})
                return self._send(404, {})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def _setup(tmp: str, n_suites=1) -> tuple[str, str, str]:
    """A suites file with a tiny task, the fake opencode, its store."""
    tpl = os.path.join(tmp, "tpl")
    os.makedirs(tpl)
    with open(os.path.join(tpl, "check.py"), "w") as f:
        f.write(CHECK)
    with open(os.path.join(tpl, "SPEC.md"), "w") as f:
        f.write("Write 42 into answer.txt.\n")
    checker = os.path.join(tmp, "skill-checker")
    with open(checker, "w") as f:
        f.write("#!/bin/sh\n")
    suites = {"version": 1, "opencode": {"permission": {"question": "deny"}}, "suites": [
        {"id": f"answer{i}", "title": f"answer {i}", "kind": "opencode", "template": tpl,
         "timeout_min": 2, "prompt": "Read SPEC.md and do it. Run python3 check.py.",
         "continue": "Continue.",
         "gates": {"list": [{"id": "pass", "args": "check\\.py", "final": True}]},
         "final_check": [sys.executable, "check.py"],
         "pristine": ["run:check.py", checker]} for i in range(n_suites)]}
    path = os.path.join(tmp, "suites.json")
    with open(path, "w") as f:
        json.dump(suites, f)
    oc = os.path.join(tmp, "opencode")
    with open(oc, "w") as f:
        f.write(FAKE_OPENCODE)
    os.chmod(oc, os.stat(oc).st_mode | stat.S_IEXEC)
    store = os.path.join(tmp, "store")
    os.makedirs(store)
    return path, oc, store


def _night(tmp, eng, n=2, extra=(), n_suites=1, env=None):
    path, oc, store = _setup(tmp, n_suites)
    old = dict(os.environ)
    os.environ.update({"FAKE_OC_STORE": store, "TANDEM_EVAL_KEY": "k-secret", "QSE_ADMIN_TOKEN":
                       "adm", **(env or {})})
    try:
        rc = te.main(["--suites", path, "--n", str(n), "--base", eng.url + "/v1",
                      "--work", os.path.join(tmp, "work"), "--out", os.path.join(tmp, "out"),
                      "--night", "2026-10-01", "--opencode", oc, *extra])
    finally:
        os.environ.clear()
        os.environ.update(old)
    with open(os.path.join(tmp, "out", "2026-10-01", "report.json")) as f:
        rep = json.load(f)
    calls = [json.loads(x) for x in open(os.path.join(store, "calls.jsonl"))]
    return rc, rep, calls


# ------------------------------------------------------------------ 1: both arms, one night

def test_a_night_end_to_end_records_both_arms():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rc, rep, calls = _night(tmp, eng, n=2, extra=("--publish",))
            assert rc == 0, rep
            assert not te.validate(rep), te.validate(rep)[:5]
            assert rep["build"] == BUILD and rep["night"] == "2026-10-01"
            arms = {(r["suite_id"], r["arm"]): r for r in rep["rows"]}
            raw, tan = arms[("answer0", "raw")], arms[("answer0", "tandem")]
            assert (raw["n"], tan["n"]) == (2, 2)
            assert (raw["pass_rate"], tan["pass_rate"]) == (0.0, 1.0)
            assert tan["minutes_to_pass"] is not None and raw["minutes_to_pass"] is None
            assert raw["continuations"] == 2 and tan["continuations"] == 0
            assert raw["empty_stops"] == 2 and tan["empty_stops"] == 0
            assert tan["interventions"] == 2 and tan["decode_tps"] == 44.0
            assert tan["cache_hit"] == round(7000 / 8010, 2)
            assert raw["build"] == tan["build"] == BUILD["code_sha"]
            c = rep["comparison"][0]
            assert c["verdict"] == "better" and c["pass_rate_not_lower"] and c["minutes_not_higher"]
            # every run records its hour and its session; the report went to the box
            assert all(r["started_at"].startswith("20") and r["session_id"] for r in rep["runs"])
            assert len(eng.published) == 1 and eng.published[0]["kind"] == "suite"
            # the arms reached opencode as headers, the key from the environment, never a file
            assert [c_["headers"] for c_ in calls] == [{"X-Tandem": "raw"},
                                                       {"X-Tandem-Profile": "overnight"}] * 2
            assert all(c_["key"] == "k-secret" for c_ in calls)
            for r in rep["runs"]:
                txt = open(os.path.join(r["dir"], "opencode.json")).read()
                assert "k-secret" not in txt and "{env:TANDEM_EVAL_KEY}" in txt
            md = open(os.path.join(tmp, "out", "2026-10-01", "report.md")).read()
            assert "| answer 0 | raw | 2 | 0.00 |" in md and "Indicative" in md
    finally:
        eng.close()


def test_the_report_reads_the_effort_columns():
    """SRV-157's measurement (QA SRV-158 row 4): per run the turns per effort level, the sizes and
    the size mode from the router's journal; per suite and arm their sums; the markdown column."""
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rc, rep, _ = _night(tmp, eng, n=2)
            assert rc == 0 and not te.validate(rep), te.validate(rep)[:5]
            tan = [r for r in rep["runs"] if r["arm"] == "tandem"]
            assert all(r["conductor"]["efforts"] == {
                "by_level": {"xhigh": 1, "high": 1}, "by_size": {"medium": 1, "small": 1},
                "size_mode": "on", "size_applied": 2, "size_effort": {"xhigh": 1, "high": 1},
                "outcomes": {"error": 1}}
                for r in tan), tan[0]["conductor"]
            rows = {r["arm"]: r for r in rep["rows"]}
            assert rows["tandem"]["efforts"] == {"high": 2, "xhigh": 2}
            # J1: the Tier 0 outcomes of the tool turns, summed over the runs
            assert rows["tandem"]["outcomes"] == {"error": 2} and rows["raw"]["outcomes"] is None
            assert rows["tandem"]["size_mode"] == "on"
            assert rows["raw"]["efforts"] is None and rows["raw"]["size_mode"] is None
            md = open(os.path.join(tmp, "out", "2026-10-01", "report.md")).read()
            assert "efforts (turns) |" in md and "high 2, xhigh 2 (size on) |" in md
            # the A/B's second night: the TandemLLM arm alone under another profile
            rc, rep2, calls = _night(tmp + "/b", eng, n=1, extra=("--arms", "tandem",
                                                                  "--profile", "overnight-size"))
            assert [r["arm"] for r in rep2["runs"]] == ["tandem"]
            assert calls[-1]["headers"] == {"X-Tandem-Profile": "overnight-size"}
            assert rep2["comparison"][0]["verdict"] == "incomplete"
    finally:
        eng.close()


# ------------------------------------------------------------------ 2: alternation, isolation

def test_the_schedule_alternates_arms_with_unique_tags():
    suites = [{"id": "a"}, {"id": "b"}]
    plan = te.schedule(suites, 3, "raw", "2026-10-01")
    for sid in ("a", "b"):
        assert [p["arm"] for p in plan if p["suite"] == sid] == ["raw", "tandem"] * 3
    assert len({p["tag"] for p in plan}) == len(plan) == 12
    assert all(p["tag"].startswith(f"te-2026-10-01-{p['suite']}-{p['arm']}-r{p['rep']}-")
               for p in plan)
    # a colliding token draws again
    toks = iter(["aa", "aa", "bb", "cc", "dd"])
    plan = te.schedule([{"id": "a"}], 1, "tandem", "n", token=lambda: next(toks))
    assert [p["arm"] for p in plan] == ["tandem", "raw"] and len({p["tag"] for p in plan}) == 2
    plan = te.schedule(suites, 2, "raw", "n", only_arms=("tandem",))
    assert [p["arm"] for p in plan] == ["tandem"] * 4


def test_the_store_is_cleared_between_arms_and_recorded():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _, rep, calls = _night(tmp, eng, n=2, n_suites=2)
        clears = [c for c in eng.calls if c == ("POST", "/v1/cache/clear")]
        # raw a0, tandem a0, raw a1, tandem a1, ...: every run changes the arm
        assert len(clears) == len(rep["runs"]) == 8
        assert all(r["store_clear"]["status"] == 200 for r in rep["runs"])
        assert rep["isolation"]["unique_tags"] and len(rep["isolation"]["store_clears"]) == 8
        assert "suffix store" in rep["isolation"]["suffix_store"]
        # every run in its own folder, which carries its tag (opencode's system prompt names it)
        dirs = {c["cwd"] for c in calls}
        assert len(dirs) == 8 and all(os.path.basename(d).startswith("te-") for d in dirs)
    finally:
        eng.close()
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _, rep, _ = _night(tmp, eng, n=1, extra=("--no-clear-store",))
        assert not [c for c in eng.calls if c[1] == "/v1/cache/clear"]
        assert all(r["store_clear"] is None for r in rep["runs"])
    finally:
        eng.close()


# ------------------------------------------------------------------ 3: pristine checkers

def test_pristine_a_changed_checker_is_caught_and_the_final_check_uses_the_template():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _, rep, _ = _night(tmp, eng, n=1, env={"FAKE_OC_TAMPER": "1"})
            for r in rep["runs"]:
                assert not r["pristine"]["ok"] and r["pristine"]["changed"] == ["run:check.py"]
                assert r["pristine"]["restored"] == ["check.py"]
                assert open(os.path.join(r["dir"], "check.py")).read() == CHECK
            assert not any(row["pristine"] for row in rep["rows"])
            assert "Pristine: changed in" in open(os.path.join(tmp, "out", "2026-10-01",
                                                               "report.md")).read()
    finally:
        eng.close()


def test_pristine_hashes_see_outside_files_and_the_template():
    with tempfile.TemporaryDirectory() as tmp:
        path, _, _ = _setup(tmp)
        s = te.load_suites(path)[0]
        before = te.pristine_hashes(s)
        run = os.path.join(tmp, "run")
        te.prepare(s, run)
        assert te.pristine_check(s, before, run) == {"ok": True, "changed": []}
        with open(s["pristine"][1], "a") as f:
            f.write("# edited\n")
        with open(os.path.join(s["template"], "SPEC.md"), "a") as f:
            f.write("more\n")
        got = te.pristine_check(s, before, run)
        assert not got["ok"] and set(got["changed"]) == {s["pristine"][1], "template:"}


# ------------------------------------------------------------------ 4: indicative

def test_n_below_ten_is_indicative():
    runs = [_run("raw", passed=False), _run("tandem", passed=True, minutes=12.0)] * 2
    s = [{"id": "x", "title": "x"}]
    rep = te.build_report(_meta(), runs, s, {"status": "found", "source": "g", "row": {}})
    assert all(r["indicative"] for r in rep["rows"]) and rep["comparison"][0]["indicative"]
    assert rep["verdict"]["indicative"] and "indicative" in rep["verdict"]["why"]
    runs10 = [_run("raw", passed=True, minutes=20.0), _run("tandem", passed=True, minutes=12.0)] * 10
    rep = te.build_report(_meta(), runs10, s, {"status": "found", "source": "g", "row": {}})
    assert not any(r["indicative"] for r in rep["rows"]) and rep["verdict"]["better"] is True
    assert not rep["verdict"]["indicative"]


def _run(arm, passed, minutes=None, rep=0):
    return {"suite": "x", "arm": arm, "rep": rep, "tag": f"t-{arm}-{id(object())}",
            "started_at": "2026-10-01T01:00:00Z", "ended_at": "2026-10-01T01:30:00Z",
            "error": None, "store_clear": None, "dir": None, "session_id": "ses_1",
            "exit_code": 0, "timed_out": False, "wall_minutes": 30.0,
            "gates": {"total": 1, "passed": int(passed),
                      "by_gate": [{"id": "pass", "passed": passed, "minutes": minutes}]},
            "passed": passed, "minutes_to_pass": minutes, "final_check": None,
            "tokens": {"input": 1, "output": 2, "reasoning": 3, "cache_read": 4},
            "cache_hit": 0.8, "gen_tps": 40.0, "turns": 3, "tool_calls": 2, "tool_errors": 0,
            "empty_stops": 0, "last_finish": "stop",
            "continuations": {"needed": 0, "sent": 0, "driver": "watch"},
            "engine": None, "conductor": None, "pristine": {"ok": True, "changed": []}}


def _meta():
    return {"night": "2026-10-01", "build": dict(BUILD), "base": "http://x/v1",
            "profile": "overnight", "driver": {"mode": "watch", "max_continues": 0}}


def test_the_comparison_names_better_worse_and_mixed():
    s = [{"id": "x", "title": "x"}]
    row = {"status": "found", "source": "g", "row": {"verdict": "pass"}}
    cases = [
        ([_run("raw", True, 20.0), _run("tandem", True, 12.0)], "better", True),
        ([_run("raw", True, 12.0), _run("tandem", True, 12.0)], "equal", True),
        ([_run("raw", True, 12.0), _run("tandem", False)], "worse", False),
        ([_run("raw", True, 12.0), _run("tandem", True, 20.0)], "mixed", False),
        ([_run("raw", False), _run("tandem", False)], "equal", True),
    ]
    for runs, want, better in cases:
        rep = te.build_report(_meta(), runs, s, row)
        assert rep["comparison"][0]["verdict"] == want, (want, rep["comparison"])
        assert rep["verdict"]["better"] is better, (want, rep["verdict"])
    rep = te.build_report(_meta(), [_run("raw", True, 1.0)], s, row)
    assert rep["comparison"][0]["verdict"] == "incomplete" and rep["verdict"]["better"] is None


# ------------------------------------------------------------------ 5: the raw row

def test_the_raw_row_is_found_attached_or_missing():
    s = [{"id": "x", "title": "x"}]
    runs = [_run("raw", True, 20.0), _run("tandem", True, 12.0)]
    eng = FakeEngine(guard=True)
    try:
        api = te.Api(eng.url)
        got = te.raw_row(api, BUILD, None)
        assert got["status"] == "found" and got["source"] == "guard.json"
        assert te.raw_row(api, {"version": "other", "code_sha": "zzz"}, None)["status"] == \
            "missing"
    finally:
        eng.close()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "row.json")
        with open(p, "w") as f:
            json.dump({"verdict": "pass", "views": {"off": {"mean": 50.1}}}, f)
        got = te.raw_row(None, BUILD, p)
        assert got["status"] == "attached" and got["row"]["views"]["off"]["mean"] == 50.1
    rep = te.build_report(_meta(), runs, s, {"status": "missing", "source": None, "row": None})
    assert rep["verdict"]["better"] is None and "never-worse row" in rep["verdict"]["why"]
    rep = te.build_report(_meta(), runs, s, {"status": "found", "source": "g",
                                             "row": {"verdict": "worse"}})
    assert rep["verdict"]["better"] is False
    # a short gate's speed_guard report (one run a side) is neither pass nor fail
    rep = te.build_report(_meta(), runs, s, {"status": "found", "source": "g",
                                             "row": {"verdict": "no_noise_estimate"}})
    assert rep["verdict"]["better"] is None and "noise is unknown" in rep["verdict"]["why"]


# ------------------------------------------------------------------ the schema

def test_the_report_validates_and_the_example_is_current():
    s = [{"id": "x", "title": "x"}]
    rep = te.build_report(_meta(), [_run("raw", True, 20.0), _run("tandem", True, 12.0)], s,
                          {"status": "found", "source": "g", "row": {}})
    assert te.validate(rep) == []
    broken = copy.deepcopy(rep)
    broken["rows"][0]["arm"] = "TandemLLM"
    del broken["verdict"]
    errs = te.validate(broken)
    assert any("arm" in e for e in errs) and any("verdict" in e for e in errs)
    with open(EXAMPLE) as f:
        ex = json.load(f)
    assert te.validate(ex) == []
    assert {r["arm"] for r in ex["rows"]} == {"raw", "tandem"}
    from server.conductor import routes
    assert routes.eval_errors(ex) == [] and routes.eval_errors({"kind": "x"})


def test_the_suites_file_loads_and_its_prompts_resolve():
    suites = te.load_suites(te.SUITES, include_disabled=True)
    ids = [s["id"] for s in suites]
    assert ids == ["minicalc", "csvq", "spiral", "photo3d", "pagoda", "protocol"]
    assert [s["id"] for s in te.load_suites(te.SUITES)] == ["minicalc", "csvq", "spiral",
                                                            "photo3d", "protocol"]
    for s in suites:
        if s["kind"] == "opencode" and s.get("prompt"):
            p, c = te.prompt_of(s, "/RUN")
            assert "check.py" in p and c
    # the coding tasks' checkers fail an empty folder and name their checker
    for tid in ("minicalc", "csvq"):
        s = next(x for x in suites if x["id"] == tid)
        with tempfile.TemporaryDirectory() as tmp:
            te.prepare(s, os.path.join(tmp, "r"))
            got = te.final_check(s["final_check"], os.path.join(tmp, "r"))
            assert got["verdict"] == "FAIL", got
    # a prompt file's markers are lines of their own
    with tempfile.TemporaryDirectory() as tmp:
        pf = os.path.join(tmp, "P.md")
        with open(pf, "w") as f:
            f.write("Paste after the line `PROMPT START`.\n\nPROMPT START\n\nWork in ~/x now.\n"
                    "CONTINUE\n\nGo on in ~/x.\n")
        s = {"prompt_file": pf, "prompt_start": "PROMPT START", "prompt_end": "CONTINUE",
             "rewrite": ["~/x"]}
        assert te.prompt_of(s, "/R") == ("Work in /R now.", "Go on in /R.")
    # a `~/` entry also covers the folder written out under any home, and nothing longer
    assert te.rewrite("in /Users/someone/x and /home/u/x/y, not /Users/a/xy", {"rewrite": ["~/x"]},
                      "/R") == "in /R and /R/y, not /Users/a/xy"


def test_reading_an_opencode_export():
    t0 = 1_790_000_000.0
    ms = t0 * 1000

    def tool(cmd, out, end_min, status="completed"):
        return {"type": "tool", "tool": "bash", "state": {"status": status,
                                                          "input": {"command": cmd},
                                                          "output": out,
                                                          "time": {"end": ms + end_min * 60000}}}
    sk = "sketchcheck  shots/{s}.png   (style: spiral)\nRESULT: {r}"
    exp = {"messages": [
        {"info": {"role": "assistant", "finish": "tool-calls",
                  "time": {"created": ms, "completed": ms + 120000},
                  "tokens": {"input": 100, "output": 10, "reasoning": 90, "cache": {"read": 900}}},
         "parts": [tool("sketchcheck shots/s1-tone.png ref/mona.jpg",
                        sk.format(s="s1-tone", r="FAIL  <- band 0.03 > 0.02"), 3),
                   tool("sketchcheck shots/s3-ribbon.png", sk.format(s="s3-ribbon", r="FAIL  <- "
                                                                     "tone_corr 0.9 < 0.95"), 5),
                   tool("sketchcheck shots/s3-ribbon.png", sk.format(s="s3-ribbon", r="PASS"), 9),
                   tool("ls", "boom", 9.5, status="error")]},
        {"info": {"role": "assistant", "finish": "stop",
                  "time": {"created": ms + 600000, "completed": ms + 660000},
                  "tokens": {"input": 5, "output": 0, "reasoning": 0}}, "parts": []}]}
    suite = next(s for s in te.load_suites(te.SUITES) if s["id"] == "spiral")
    got = te.read_export(exp, suite, t0)
    assert [(g["id"], g["passed"], g["minutes"]) for g in got["gates"]] == \
        [("tone", True, 3.0), ("ribbon", True, 9.0), ("animated", False, None)]
    assert got["empty_stops"] == 1 and got["tool_errors"] == 1 and got["tool_calls"] == 4
    assert got["tokens"] == {"input": 105, "output": 10, "reasoning": 90, "cache_read": 900}
    assert got["wall_minutes"] == 11.0 and got["gen_tps"] == round(100 / 180, 2)
    assert got["cache_hit"] == round(900 / 1005, 4)
    # ordinal gates: the k-th PASS is gate k
    pg = "pagodacheck  .\nRESULT: {r}"
    exp2 = {"messages": [{"info": {"role": "assistant", "time": {}}, "parts": [
        tool("pagodacheck .", pg.format(r="PASS"), 30), tool("pagodacheck .", pg.format(
            r="FAIL  <- (a) x 3.5 > 1.0"), 40), tool("pagodacheck .", pg.format(r="PASS"), 55)]}]}
    g = te.gate_progress(te.tool_parts(exp2), {"ordinal": True, "total": 9}, ms)
    assert [x["minutes"] for x in g[:3]] == [30.0, 55.0, None] and len(g) == 9


def test_the_pass_rule_and_the_tool_calls_and_turns_to_pass():
    """The decided pass rule (2026-10-02): a suite with a final check passes by its verdict, the
    stage gates are progress; the PASS minute is the final gate's. Beside it, the tool calls and
    the turns up to that minute. A suite without a final check passes when every gate does."""
    t0 = 1_790_000_000.0
    ms = t0 * 1000

    def tool(cmd, out, end_min):
        return {"type": "tool", "tool": "bash", "state": {
            "status": "completed", "input": {"command": cmd}, "output": out,
            "time": {"end": ms + end_min * 60000}}}

    def turn(start_min, *parts):
        return {"info": {"role": "assistant", "finish": "tool-calls",
                         "time": {"created": ms + start_min * 60000,
                                  "completed": ms + (start_min + 0.5) * 60000},
                         "tokens": {"input": 1, "output": 1, "reasoning": 1}},
                "parts": list(parts)}
    sk = "sketchcheck  shots/{s}.png   (style: spiral)\nRESULT: {r}"
    exp = {"messages": [
        turn(0.0, tool("ls", "a", 0.2)),
        turn(1.0, tool("sketchcheck shots/s1-tone.png", sk.format(s="s1-tone", r="FAIL  <- "
                                                                      "tone_corr 0.5 < 0.85"),
                       2.0)),
        turn(3.0, tool("sketchcheck shots/s4-animated.png",
                       sk.format(s="s4-animated", r="PASS"), 4.0), tool("ls", "b", 4.5)),
        turn(6.0, tool("ls", "c", 7.0))]}
    suite = next(s for s in te.load_suites(te.SUITES) if s["id"] == "spiral")
    assert te.final_gates(suite) == ["animated"]
    got = te.read_export(exp, suite, t0)
    assert [(g["id"], g["passed"]) for g in got["gates"]] == \
        [("tone", False), ("ribbon", False), ("animated", True)]
    fin = {"verdict": "PASS", "readings": [], "tail": "", "shot": None}
    te.apply_final(got["gates"], suite, fin, "", t0, got["wall_minutes"])
    o = te.outcome(got, fin, 0, False, "watch", te.final_gates(suite))
    # the final check passed: the run passes although the tone and ribbon gates never did
    assert o["passed"] and o["pass_rule"] == "final_check" and o["minutes_to_pass"] == 4.0, o
    assert (o["tool_calls_to_pass"], o["turns_to_pass"]) == (3, 3), o
    assert o["gates"]["passed"] == 1 and o["tool_calls"] == 5 and o["turns"] == 4
    # every stage gate seen passing, the final check FAIL or no shot: not passed
    for verdict in ("FAIL", "NO_SHOT", "ERROR"):
        gates = [dict(g, passed=True, minutes=1.0) for g in got["gates"]]
        bad = dict(fin, verdict=verdict)
        te.apply_final(gates, suite, bad, "", t0, 8.0)
        o = te.outcome(dict(got, gates=gates), bad, 0, False, "watch", te.final_gates(suite))
        assert not o["passed"] and o["minutes_to_pass"] is None, (verdict, o)
        assert o["tool_calls_to_pass"] is None and o["turns_to_pass"] is None
    # no final check: every gate, as before; the minute is the latest gate's
    gates = [{"id": "a", "passed": True, "minutes": 2.0}, {"id": "b", "passed": True,
                                                           "minutes": 4.5}]
    o = te.outcome(dict(got, gates=gates), None, 0, False, "watch")
    assert o["passed"] and o["pass_rule"] == "gates" and o["minutes_to_pass"] == 4.5
    assert (o["tool_calls_to_pass"], o["turns_to_pass"]) == (4, 3)


def test_every_comparison_reads_the_records_minutes_to_pass():
    """The morning table of 2026-10-02 compared wall minutes by mistake. The rows, the comparison
    and both markdown tables read the records' minutes_to_pass (wall 30 here, PASS at 12 and 20),
    with the tool calls and turns to PASS beside them."""
    s = [{"id": "x", "title": "x"}]
    raw = _run("raw", True, 20.0)
    tan = _run("tandem", True, 12.0)
    raw.update(tool_calls_to_pass=40, turns_to_pass=30, pass_rule="final_check")
    tan.update(tool_calls_to_pass=25, turns_to_pass=18, pass_rule="final_check")
    rep = te.build_report(_meta(), [raw, tan], s, {"status": "found", "source": "g",
                                                   "row": {"verdict": "pass"}})
    assert te.validate(rep) == [], te.validate(rep)[:5]
    rows = {r["arm"]: r for r in rep["rows"]}
    assert rows["tandem"]["minutes_to_pass"] == 12.0 and rows["tandem"]["wall_minutes"] == 30.0
    assert (rows["tandem"]["tool_calls_to_pass"], rows["raw"]["tool_calls_to_pass"]) == (25, 40)
    assert (rows["tandem"]["turns_to_pass"], rows["raw"]["turns_to_pass"]) == (18, 30)
    c = rep["comparison"][0]
    assert c["minutes_to_pass"] == {"raw": 20.0, "tandem": 12.0} and c["verdict"] == "better"
    assert c["tool_calls_to_pass"] == {"raw": 40, "tandem": 25}
    assert "12 against 20 minutes" in c["why"] and "25 against 40 tool calls to PASS" in c["why"]
    assert rep["pass_rule"].startswith("a run passes by its suite's final check")
    md = te.to_md(rep)
    assert "| x | tandem | 1 | 1.00 | 1/1 | 12 | 25 | 18 | 30 |" in md, md
    assert "| x | better | 12 / 20 | 25 / 40 | 18 / 30 |" in md, md
    assert "Pass rule: a run passes by its suite's final check verdict" in md
    # a record from before has no counts: the columns say so, nothing breaks
    old = [_run("raw", True, 20.0), _run("tandem", True, 12.0)]
    rep = te.build_report(_meta(), old, s, {"status": "found", "source": "g", "row": {}})
    assert te.validate(rep) == [] and rep["rows"][0]["tool_calls_to_pass"] is None
    assert "tool calls to PASS" not in rep["comparison"][0]["why"]
    assert "| x | better | 12 / 20 | — / — | — / — |" in te.to_md(rep)


def test_the_box_lock_refuses_while_a_hold_runs():
    lk = te.BoxLock("dgx", wait_s=120)
    cmd = lk.argv()
    assert cmd[:3] == ["ssh", "-o", "BatchMode=yes"] and cmd[3] == "dgx"
    assert "pgrep -fa '[h]old.sh' && exit 3" in cmd[4] and "flock -w 120 ~/.qwen38-box.flock" \
        in cmd[4]
    assert te.night_of(1_790_820_000.0) in ("2026-09-30", "2026-10-01")   # local time


# ------------------------------------------------------------------ over HTTP (fake engine)

def test_the_report_is_kept_for_the_compare_screen_over_http():
    """POST /v1/conductor/evals keeps a report in the conductor's evals/; GET lists it."""
    from test_conductor_e2e import ADMIN, PROXY, _call, _check, _port, _server, _wait
    port = _port()
    cdir = tempfile.mkdtemp(prefix="qse-eval-cond-")
    proc, _ = _server(port, conductor_dir=cdir)
    try:
        _wait(port, proc)
        with open(EXAMPLE) as f:
            ex = json.load(f)
        st, _, got = _call(port, "POST", "/v1/conductor/evals", {"report": ex}, headers=PROXY,
                           token=None)
        assert st in (401, 403, 404), got                     # a write needs the admin token
        st, _, got = _call(port, "POST", "/v1/conductor/evals", {"report": {"kind": "x"}},
                           token=ADMIN)
        assert st == 422 and "kind" in got["error"]["message"]
        st, _, got = _call(port, "POST", "/v1/conductor/evals", {"report": ex}, token=ADMIN)
        assert st == 201, got
        _check(got, "eval-write")
        st, _, got = _call(port, "GET", "/v1/conductor/evals")
        assert st == 200
        _check(got, "evals")
        suite = [r for r in got["reports"] if r.get("kind") == "suite"]
        assert len(suite) == 1 and suite[0]["night"] == ex["night"]
        assert contract_check.validate(suite[0], json.load(open(te.SCHEMA))) == []
    finally:
        proc.terminate()
        proc.wait(10)


def test_a_run_sees_its_own_folder_as_pwd():
    """opencode reads $PWD for its project directory; the harness's own PWD must not leak in."""
    import tempfile
    from tools import tandem_eval as t
    d = tempfile.mkdtemp(prefix="te-pwd-")
    log = os.path.join(d, "log")
    code, timed_out, lines = t.run_process(["sh", "-c", 'echo "$PWD"'], d,
                                           dict(os.environ, PWD="/somewhere/else"), 10, log)
    assert code == 0 and not timed_out and lines == [os.path.abspath(d)], lines


# ------------------------------------------------------------------ OPS-63: round 1's reader bugs

# tests/fixtures/tandem-eval-round1 is round 1 (2026-10-01, n = 1 a side) without
# private content: the published run records and report, each session's export cut down to its
# numbers (message roles, times, finishes, tokens; tool names, statuses and end times; for checker
# calls the command with the folder as {run} and only the checker's header and RESULT lines -- no
# prompt, reasoning or answer text), each run folder's shot names with their times after the
# run's start, and the real checker's output on the final shot (sums.json: the full exports'
# token sums and message counts, summed by hand outside the harness).
R1 = os.path.join(HERE, "fixtures", "tandem-eval-round1")
R1_TAGS = sorted(n[:-5] for n in os.listdir(os.path.join(R1, "runs")))


def _r1(kind, tag):
    with open(os.path.join(R1, kind, tag + ".json")) as f:
        return json.load(f)


def _r1_suites():
    return {s["id"]: s for s in te.load_suites(te.SUITES)}


def _r1_t0(rec):
    return te._iso_ms(rec["started_at"]) / 1000.0


# what round 1 really did, from the full exports and a separate checker's runs at 18:00
R1_TRUTH = {
    "minicalc-raw": {"turns": 14, "reasoning": 16063, "wall": 6.61, "minutes": 6.45},
    "minicalc-tandem": {"turns": 7, "reasoning": 17802, "wall": 7.49, "minutes": 7.38},
    "csvq-raw": {"turns": 16, "reasoning": 36302, "wall": 16.78, "minutes": 16.27},
    "csvq-tandem": {"turns": 10, "reasoning": 26019, "wall": 12.2, "minutes": 12.02},
    "spiral-raw": {"turns": 90, "reasoning": 50643, "wall": 42.65, "minutes": 40.95,
                   "shot": "shots/s4-animated.png", "metrics": {"tone_corr": 0.983,
                                                                "edge_corr": 0.950}},
    "spiral-tandem": {"turns": 54, "reasoning": 7618, "wall": 9.75, "minutes": 8.82,
                      "shot": "shots/s4-animated.png", "metrics": {"tone_corr": 0.981,
                                                                   "edge_corr": 0.931}},
    "photo3d-raw": {"turns": 144, "reasoning": 104075, "wall": 66.46, "minutes": 40.46,
                    "shot": "shots/s7.png", "metrics": {"de8 coarse colour": 9.19,
                                                        "lum layout corr": 0.903,
                                                        "edge structure": 0.630}},
    "photo3d-tandem": {"turns": 192, "reasoning": 47416, "wall": 41.91, "minutes": 29.52,
                       "shot": "shots/final.png", "metrics": {"de8 coarse colour": 8.50,
                                                              "lum layout corr": 0.923,
                                                              "edge structure": 0.588}},
}


def _truth(tag):
    return R1_TRUTH["-".join(tag.split("-")[4:6])]


# opencode 1.18.33 (bun) exits before a pipe has taken more than 64 KiB of an export: a piped
# `opencode export` of every round-1 session ended at exactly 65,536 bytes (measured on the Mac,
# 2026-10-01), while the same export to a file was whole. This stand-in does the same.
FAKE_EXPORT = r'''#!/usr/bin/env python3
import os, stat, sys
args = sys.argv[1:]
assert args[0] == "export", args
print("Exporting session: " + args[1], file=sys.stderr)
data = open(os.path.join(os.environ["FAKE_OC_SESSIONS"], args[1] + ".json"), "rb").read()
if stat.S_ISFIFO(os.fstat(1).st_mode):
    data = data[:65536]
os.write(1, data)
'''


def _fake_export(tmp):
    """A fake `opencode export` over the round-1 sessions, by session id."""
    sess = os.path.join(tmp, "sessions")
    os.makedirs(sess)
    for tag in R1_TAGS:
        with open(os.path.join(sess, _r1("runs", tag)["session_id"] + ".json"), "w") as f:
            json.dump(_r1("sessions", tag), f)
    oc = os.path.join(tmp, "opencode")
    with open(oc, "w") as f:
        f.write(FAKE_EXPORT)
    os.chmod(oc, os.stat(oc).st_mode | stat.S_IEXEC)
    return oc, dict(os.environ, FAKE_OC_SESSIONS=sess)


def test_round1_the_export_is_read_whole_through_a_file_not_cut_at_a_pipe():
    """Bug 1: `_export` read the export through a pipe; opencode cut it at 64 KiB, the JSON did
    not parse, and every run read as 0 tokens, 0 turns and no gate."""
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        oc, env = _fake_export(tmp)
        for tag in ("te-2026-10-01-photo3d-raw-r0-5472f4", "te-2026-10-01-photo3d-tandem-r0-11c1c9"):
            sid = _r1("runs", tag)["session_id"]
            whole = os.path.getsize(os.path.join(tmp, "sessions", sid + ".json"))
            assert whole > 65536, whole                    # these two are past a pipe's 64 KiB
            piped = subprocess.run([oc, "export", sid], capture_output=True, env=env).stdout
            assert len(piped) == 65536                     # what the old reader got
            exp, how = te.export_session(oc, sid, env)
            assert how == {"bytes": whole, "messages": len(_r1("sessions", tag)["messages"]),
                           "error": None}, how
            assert len(exp["messages"]) == how["messages"]
        # an export that cannot be read says so, and its counts are None, never 0
        exp, how = te.export_session(oc, "ses_none", env)
        assert exp == {} and how["messages"] is None and how["error"], how
        got = te.read_export(exp, _r1_suites()["spiral"], 0.0, how)
        assert got["tokens"] is None and got["turns"] is None and got["tool_calls"] is None
        assert got["reader"] is how and not any(g["passed"] for g in got["gates"])
        exp, how = te.export_session(os.path.join(tmp, "no-such-opencode"), "ses_x", env)
        assert exp == {} and "FileNotFoundError" in how["error"]


def test_round1_the_session_stats_are_read_from_the_export():
    """Bug 2: tokens, cache, turns and wall time per run, as the full exports and the Memo note
    give them; every round-1 run had zero empty stops and ended on a stop."""
    suites = _r1_suites()
    sums = json.load(open(os.path.join(R1, "sums.json")))
    for tag in R1_TAGS:
        rec, truth = _r1("runs", tag), _truth(tag)
        got = te.read_export(_r1("sessions", tag), suites[rec["suite"]], _r1_t0(rec))
        assert got["tokens"] == sums[tag]["tokens"], (tag, got["tokens"])
        assert got["turns"] == sums[tag]["turns"] == truth["turns"], (tag, got["turns"])
        assert got["tokens"]["reasoning"] == truth["reasoning"]
        assert got["wall_minutes"] == truth["wall"], (tag, got["wall_minutes"])
        tk = got["tokens"]
        assert got["cache_hit"] == round(tk["cache_read"] / (tk["input"] + tk["cache_read"]), 4)
        assert got["empty_stops"] == 0 and got["last_finish"] == "stop"
        assert got["tool_calls"] > 0 and got["gen_tps"] > 0
        # what the published record said instead
        assert rec["tokens"] == {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0}
        assert rec["turns"] == 0
    # the session alone already shows every gate of the image suites passing
    got = te.read_export(_r1("sessions", "te-2026-10-01-spiral-raw-r0-2dddea"), suites["spiral"],
                         _r1_t0(_r1("runs", "te-2026-10-01-spiral-raw-r0-2dddea")))
    assert [(g["id"], g["passed"], g["minutes"]) for g in got["gates"]] == \
        [("tone", True, 17.7), ("ribbon", True, 37.73), ("animated", True, 40.95)]
    rec = _r1("runs", "te-2026-10-01-photo3d-tandem-r0-11c1c9")
    got = te.read_export(_r1("sessions", rec["tag"]), suites["photo3d"], _r1_t0(rec))
    assert [g["minutes"] for g in got["gates"]] == [7.28, 11.52, 14.44, 28.21, 29.52]


def _r1_folder(tmp, tag):
    """The run folder with its shots (empty files at their times after the run's start)."""
    rec, fol = _r1("runs", tag), _r1("folders", tag)
    d = os.path.join(tmp, tag)
    os.makedirs(os.path.join(d, "shots", "hourly"))
    t0 = _r1_t0(rec)
    for name, dt in fol["shots"].items():
        p = os.path.join(d, "shots", name)
        open(p, "w").close()
        os.utime(p, (t0 + dt, t0 + dt))
        if name.endswith(".png"):                   # the checker's heatmap beside it, newer
            open(p[:-4] + ".check.png", "w").close()
    open(os.path.join(d, "shots", "hourly", "16.png"), "w").close()
    return d


def test_round1_the_final_shot_is_the_last_stages_whatever_its_case():
    """Bug 3: the final RESULT is the checker on the last stage's shot. Round 1 named it
    s4-animated.png (beside s4-mid.png, a mid-animation frame), s7.png, and final.png beside
    S7.png."""
    suites = _r1_suites()
    with tempfile.TemporaryDirectory() as tmp:
        for tag in R1_TAGS:
            rec = _r1("runs", tag)
            if rec["suite"] not in ("spiral", "photo3d"):
                assert not suites[rec["suite"]].get("final_shot")
                continue
            d = _r1_folder(tmp, tag)
            assert te.final_shot(suites[rec["suite"]], d) == _truth(tag)["shot"], tag
    s = suites["photo3d"]
    for names, want in ((["S5.png", "S6.png", "S7-polish.png"], "shots/S7-polish.png"),
                        (["s5.png", "s6.png"], "shots/s6.png"),
                        (["S4.png"], None), (["s7.check.png"], None)):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "shots", "hourly"))
            open(os.path.join(d, "shots", "hourly", "17.png"), "w").close()
            for n in names:
                open(os.path.join(d, "shots", n), "w").close()
            assert te.final_shot(s, d) == want, (names, te.final_shot(s, d))
    # a command that needs a shot when there is none: NO_SHOT, which leaves the session's reading
    got = te.final_check(s["final_check"], "/nonexistent", shot=None)
    assert got["verdict"] == "NO_SHOT" and got["shot"] is None
    gates = [{"id": "full", "passed": True, "minutes": 30.0}]
    te.apply_final(gates, s, got, "/nonexistent", 0.0, 40.0)
    assert gates == [{"id": "full", "passed": True, "minutes": 30.0}]


FAKE_CHECKER = r'''#!/usr/bin/env python3
"""The suite's checker for one round-1 run: the real checker's output on the real final shot, for
that shot only (any other shot fails), with {run} the folder it was given."""
import json, os, sys
want, out, run = json.load(open(os.environ["FAKE_CHECK"]))
args = sys.argv[1:]
if args[:1] == [want] and ("--project" not in args or args[args.index("--project") + 1] == run):
    print(out.replace("{run}", run), end="")
else:
    print("scenecheck  " + " ".join(args) + "\nRESULT: FAIL  <- (a) not the final shot 1 > 0")
'''


def _fake_checker(tmp, suite):
    p = os.path.join(tmp, "fakecheck")
    with open(p, "w") as f:
        f.write(FAKE_CHECKER)
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    s = copy.deepcopy(suite)
    s["final_check"] = [p] + s["final_check"][1:]
    return s


def test_round1_the_final_check_reads_the_shot_and_decides_the_last_gate():
    """With the shot found, the final check gives the RESULT the owner got at 18:00, with the numbers
    behind the PASS, and it is the last word on its gate: a PASS keeps the session's minute; a
    FAIL fails the gate even when the session saw a PASS earlier."""
    suites = _r1_suites()
    with tempfile.TemporaryDirectory() as tmp:
        for tag in R1_TAGS:
            rec, truth = _r1("runs", tag), _truth(tag)
            if rec["suite"] not in ("spiral", "photo3d"):
                continue
            d = _r1_folder(tmp, tag)
            s = _fake_checker(tmp, suites[rec["suite"]])
            with open(os.path.join(tmp, "want.json"), "w") as f:
                json.dump([truth["shot"], _r1("folders", tag)["check_output"], d], f)
            os.environ["FAKE_CHECK"] = os.path.join(tmp, "want.json")
            try:
                fin = te.final_check(s["final_check"], d, shot=te.final_shot(s, d))
            finally:
                del os.environ["FAKE_CHECK"]
            assert fin["verdict"] == "PASS" and fin["shot"] == truth["shot"], (tag, fin)
            m = {x["name"]: x["value"] for x in fin["metrics"]}
            for k, v in truth["metrics"].items():
                assert m[k] == v, (tag, k, m)
            got = te.read_export(_r1("sessions", tag), s, _r1_t0(rec))
            te.apply_final(got["gates"], s, fin, d, _r1_t0(rec), got["wall_minutes"])
            o = te.outcome(got, fin, 0, False, "watch")
            assert o["passed"] and o["minutes_to_pass"] == truth["minutes"], (tag, o)
            assert o["continuations"] == {"needed": 0, "sent": 0, "driver": "watch"}
            # the same run with a failing last shot: the gate fails, whatever the session saw
            bad = dict(fin, verdict="FAIL")
            got = te.read_export(_r1("sessions", tag), s, _r1_t0(rec))
            te.apply_final(got["gates"], s, bad, d, _r1_t0(rec), got["wall_minutes"])
            o = te.outcome(got, bad, 0, False, "watch")
            assert not o["passed"] and got["gates"][-1] == {"id": got["gates"][-1]["id"],
                                                            "passed": False, "minutes": None}
            assert o["continuations"]["needed"] == 1
    # a PASS the session never saw takes the shot's time
    s = suites["spiral"]
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "shots"))
        p = os.path.join(d, "shots", "s4-animated.png")
        open(p, "w").close()
        os.utime(p, (1000.0 + 600, 1000.0 + 600))
        gates = [{"id": "tone", "passed": True, "minutes": 2.0},
                 {"id": "ribbon", "passed": True, "minutes": 5.0},
                 {"id": "animated", "passed": False, "minutes": None}]
        te.apply_final(gates, s, {"verdict": "PASS", "shot": "shots/s4-animated.png"}, d, 1000.0,
                       12.0)
        assert gates[-1] == {"id": "animated", "passed": True, "minutes": 10.0}


def test_round1_continuations_count_only_what_the_driver_would_send():
    """Bug 4: every round-1 run said one continuation was needed, passed or not -- the gates were
    judged before the final check and on an empty export. A continuation is needed each time
    opencode ended in time with the suite not done."""
    def o(passed, sent, timed_out):
        got = {"gates": [{"id": "pass", "passed": passed, "minutes": 1.0 if passed else None}],
               "tokens": None, "cache_hit": None, "gen_tps": None, "turns": None,
               "tool_calls": None, "tool_errors": None, "empty_stops": 0, "last_finish": None}
        return te.outcome(got, None, sent, timed_out, "continue")["continuations"]["needed"]
    assert (o(True, 0, False), o(False, 0, False), o(False, 2, False), o(True, 2, False),
            o(False, 1, True)) == (0, 1, 3, 2, 1)
    assert all(_r1("runs", t)["continuations"]["needed"] == 1 for t in R1_TAGS)


class LedgerEngine(FakeEngine):
    """The fake engine with a ledger: given request rows, paged newest first like the box."""

    def __init__(self, rows):
        super().__init__(guard=True)
        self.rows = sorted(rows, key=lambda r: r["ts"], reverse=True)
        eng = self
        base = self.httpd.RequestHandlerClass.do_GET

        def do_GET(h):
            if h.path.startswith("/v1/dashboard/requests"):
                eng.calls.append(("GET", h.path))
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(h.path).query)
                start = int((q.get("before") or ["0"])[0])
                page = eng.rows[start:start + 200]
                return h._send(200, {"requests": page, "next_before":
                                     str(start + 200) if start + 200 < len(eng.rows) else None})
            return base(h)
        self.httpd.RequestHandlerClass.do_GET = do_GET


def test_round1_the_report_is_rebuilt_from_the_folders_and_the_ledger():
    """The round-1 report rebuilt with --report --reread: every run re-read from its session, its
    folder's final shot and the engine ledger; both arms pass every suite; the night's own build
 is kept, not the build the server serves now; the published report it replaces
    is named, and the old report and run records are kept beside the new ones."""
    import shutil as sh
    suites_doc = json.load(open(te.SUITES))
    with tempfile.TemporaryDirectory() as tmp:
        oc, env = _fake_export(tmp)
        work, night = os.path.join(tmp, "work"), os.path.join(tmp, "out", "2026-10-01")
        os.makedirs(os.path.join(night, "runs"))
        sh.copy(os.path.join(R1, "report-published.json"), os.path.join(night, "report.json"))
        ledger, checks = [], {}
        for tag in R1_TAGS:
            rec = _r1("runs", tag)
            d = _r1_folder(work, tag)
            rec["dir"] = d
            with open(os.path.join(night, "runs", tag + ".json"), "w") as f:
                json.dump(rec, f)
            checks[d] = [_truth(tag).get("shot"), _r1("folders", tag)["check_output"], d]
            # the ledger's rows for this run's window, one per turn, plus one outside it
            t0 = _r1_t0(rec)
            for i in range(_truth(tag)["turns"]):
                ledger.append({"ts": te.now_iso(t0 + 5 + i * 0.5)[:19] + ".000Z",
                               "client": {"kind": "opencode"}, "decode_tps": 40.0 + i % 3,
                               "prefill_tps": 900.0, "completion_tokens": 10,
                               "reasoning_tokens": 5, "prompt_tokens": 100, "cached_tokens": 80,
                               "finish_reason": "tool_calls"})
            ledger.append({"ts": te.now_iso(t0 - 86400)[:19] + ".000Z", "client": {"kind": "opencode"},
                           "decode_tps": 1.0, "finish_reason": "stop"})
        # every suite's checker is the fake, which answers for the folder it is given
        ck = os.path.join(tmp, "fakecheck-all")
        with open(ck, "w") as f:
            f.write("#!/usr/bin/env python3\nimport json, os, sys\n"
                    "w = json.load(open(os.environ['FAKE_CHECKS']))[os.getcwd()]\n"
                    "args = sys.argv[1:]\n"
                    "assert w[0] is None or args[:1] == [w[0]], (args, w)\n"
                    "print(w[1].replace('{run}', w[2]), end='')\n")
        os.chmod(ck, 0o755)
        with open(os.path.join(tmp, "checks.json"), "w") as f:
            json.dump({os.path.realpath(k): v for k, v in checks.items()} | checks, f)
        for s in suites_doc["suites"]:
            if s.get("final_check"):
                s["final_check"] = [ck] + [x for x in s["final_check"][1:] if x != "check.py"]
            if s.get("template"):
                s["template"] = os.path.join(tmp, "tpl")
            s.pop("prompt_file", None)
            s.setdefault("prompt", "x")
        sp = os.path.join(tmp, "suites.json")
        with open(sp, "w") as f:
            json.dump(suites_doc, f)
        eng = LedgerEngine(ledger)
        old = dict(os.environ)
        os.environ.update(env, FAKE_CHECKS=os.path.join(tmp, "checks.json"))
        try:
            rc = te.main(["--suites", sp, "--report", night, "--reread", "--opencode", oc,
                          "--base", eng.url + "/v1", "--supersedes", "suite-2026-10-01-155808.json",
                          "--supersede-why", "the reader bug (OPS-63)"])
        finally:
            os.environ.clear()
            os.environ.update(old)
            eng.close()
        assert rc == 0
        rep = json.load(open(os.path.join(night, "report.json")))
        assert te.validate(rep) == [], te.validate(rep)[:5]
        assert rep["build"] == {"version": "0.3.4-test", "code_sha": "3c64a688569d2967"}
        assert rep["supersedes"] == {"file": "suite-2026-10-01-155808.json",
                                     "why": "the reader bug (OPS-63)"}
        for r in rep["runs"]:
            truth = _truth(r["tag"])
            assert r["passed"] and r["gates"]["passed"] == r["gates"]["total"], r["tag"]
            assert r["minutes_to_pass"] == truth["minutes"], (r["tag"], r["minutes_to_pass"])
            assert r["pass_rule"] == "final_check"
            assert 0 < r["tool_calls_to_pass"] <= r["tool_calls"], r["tag"]
            assert 0 < r["turns_to_pass"] <= r["turns"], r["tag"]
            assert r["turns"] == truth["turns"] and r["tokens"]["reasoning"] == truth["reasoning"]
            assert r["wall_minutes"] == truth["wall"]
            assert r["continuations"]["needed"] == 0
            assert r["final_check"]["verdict"] == "PASS"
            assert r["final_check"]["shot"] == truth.get("shot")
            assert r["engine"]["requests"] == truth["turns"]       # the window's rows only
            assert r["engine"]["decode_tps_median"] == 41.0
            assert r["reader"]["error"] is None
            assert r["store_clear"]["status"] == 200 and r["pristine"]["ok"]   # kept as taken
        rows = {(r["suite_id"], r["arm"]): r for r in rep["rows"]}
        assert all(r["pass_rate"] == 1.0 and r["continuations"] == 0 for r in rows.values())
        assert rows[("spiral", "tandem")]["minutes_to_pass"] == 8.82
        assert rows[("photo3d", "raw")]["minutes_to_pass"] == 40.46
        # the hand-read table of the Memo note: turns and thinking tokens per suite and arm
        for tag in R1_TAGS:
            rec, truth = _r1("runs", tag), _truth(tag)
            row = rows[(rec["suite"], rec["arm"])]
            assert (row["turns"], row["reasoning_tokens"]) == (truth["turns"], truth["reasoning"])
        verdicts = {c["suite_id"]: c["verdict"] for c in rep["comparison"]}
        assert verdicts == {"minicalc": "mixed", "csvq": "better", "spiral": "better",
                            "photo3d": "better"}, verdicts
        assert rep["verdict"]["better"] is False and "minicalc" in rep["verdict"]["why"]
        # nothing deleted: the published report and the records it was built from are kept
        kept = os.listdir(night)
        old_rep = [n for n in kept if n.startswith("report-superseded-") and n.endswith(".json")]
        assert len(old_rep) == 1 and [n for n in kept if n.startswith("runs-superseded-")]
        assert json.load(open(os.path.join(night, old_rep[0])))["rows"][4]["pass_rate"] == 0.0
        md = open(os.path.join(night, "report.md")).read()
        assert "PASS on shots/s4-animated.png (tone_corr 0.983" in md
        assert "Supersedes suite-2026-10-01-155808.json" in md


def test_round1_against_the_real_folders_when_present():
    """Opt-in (TANDEM_EVAL_REAL_RUNS=~/tandem-eval-runs/2026-10-01 on the Mac that ran round 1):
    the real opencode export of a real session through a pipe and through export_session, and the
    real checkers on the real final shots."""
    real = os.path.expanduser(os.environ.get("TANDEM_EVAL_REAL_RUNS", ""))
    if not real or not os.path.isdir(real):
        print("    (skipped: TANDEM_EVAL_REAL_RUNS is not a folder)")
        return
    import shutil as sh
    import subprocess
    suites = _r1_suites()
    oc = sh.which("opencode")
    sums = json.load(open(os.path.join(R1, "sums.json")))
    for tag in R1_TAGS:
        rec, truth = _r1("runs", tag), _truth(tag)
        d = os.path.join(real, tag)
        if oc:
            piped = subprocess.run([oc, "export", rec["session_id"]], capture_output=True).stdout
            exp, how = te.export_session(oc, rec["session_id"])
            assert how["bytes"] == sums[tag]["export_bytes"] and how["error"] is None, how
            assert len(piped) < how["bytes"], (len(piped), how)
            got = te.read_export(exp, suites[rec["suite"]], _r1_t0(rec), how)
            assert got["tokens"] == sums[tag]["tokens"] and got["turns"] == truth["turns"]
        if rec["suite"] in ("spiral", "photo3d"):
            s = suites[rec["suite"]]
            fin = te.final_check(s["final_check"], d, shot=te.final_shot(s, d))
            m = {x["name"]: x["value"] for x in fin["metrics"]}
            assert fin["verdict"] == "PASS" and fin["shot"] == truth["shot"], (tag, fin)
            assert all(m[k] == v for k, v in truth["metrics"].items()), (tag, m)


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
