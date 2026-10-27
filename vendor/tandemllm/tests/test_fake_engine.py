"""The fake engine (fixture) as the e2e runs it: `server/app.py --fake-engine` in a
process of its own, with triton unimportable, a temp ledger and a test admin token.

What the dashboard's e2e depends on: the server starts with no model and no GPU; the live usage
check passes; every dashboard endpoint validates against the contract; the rows reach the ledger;
the steering words give an error and a tool call; the production ledger is refused; SIGTERM drains
and exits 0. The e2e runs with the dashboard login on (its token screen is part of what it tests);
the default, off, is checked on its own: no secrets, and the dashboard answers without a token.

Run: python tests/test_fake_engine.py
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import contract_check, usage_check  # noqa: E402

ADMIN = "e2e-" + "0" * 40
LAUNCH = ("import sys, runpy; sys.modules['triton'] = None; sys.modules['triton.language'] = None;"
          "sys.argv = ['server/app.py'] + sys.argv[1:]; runpy.run_path('server/app.py', "
          "run_name='__main__')")


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _start(port, ledger, home=None, extra=(), env_extra=None, admin=ADMIN):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home or os.environ["HOME"],
           "PYTHONPATH": ROOT, "CUDA_VISIBLE_DEVICES": ""}
    if admin:
        env["QSE_ADMIN_TOKEN"] = admin
    env.update(env_extra or {})
    return subprocess.Popen([sys.executable, "-c", LAUNCH, "--fake-engine", "--port", str(port),
                             "--served-model", "qwen38-spark-engine", "--usage-ledger", ledger,
                             "--fake-tps", "400", *extra],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


def _get(port, path, token=ADMIN, headers=None):
    h = dict(headers or {}, **({"Authorization": f"Bearer {token}"} if token else {}))
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=h)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _wait(port, proc):
    end = time.time() + 60
    while time.time() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except OSError:
            if proc.poll() is not None:
                raise AssertionError(proc.stdout.read())
            time.sleep(0.3)
    raise AssertionError("the fake engine did not come up in 60 s")


def _chat(port, text, **extra):
    body = dict({"messages": [{"role": "user", "content": text}], "max_tokens": 200}, **extra)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def test_the_fake_engine_end_to_end():
    port = _port()
    ledger = os.path.join(tempfile.mkdtemp(prefix="qse-e2e-"), "ledger.sqlite3")
    proc = _start(port, ledger, extra=("--dashboard-login", "on"))
    try:
        _wait(port, proc)
        args = type("A", (), {"base": f"http://127.0.0.1:{port}", "model": "qwen38-spark-engine",
                              "token": None, "prompt": "Name three rivers.", "max_tokens": 64,
                              "timeout": 30.0})()
        bad = {c[0]: r["errors"] for c in usage_check.CASES
               if (r := usage_check.run_case(args, *c))["errors"]}
        assert not bad, bad
        try:
            _chat(port, "FAKE_ERROR please")
            raise AssertionError("FAKE_ERROR did not fail")
        except urllib.error.HTTPError as exc:
            assert exc.code == 500
        body = _chat(port, "use the tool now",
                     tools=[{"type": "function", "function": {"name": "web_search"}}])
        assert body["choices"][0]["finish_reason"] == "tool_calls", body["choices"][0]
        assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "web_search"
        # through the fake: the new fields reach its loop and come back in their shapes
        body = _chat(port, "Name three rivers.", logprobs=True, top_logprobs=1, n=2,
                     max_tokens=16)
        assert [c["index"] for c in body["choices"]] == [0, 1]
        for c in body["choices"]:
            assert len(c["logprobs"]["content"]) == 16 and c["finish_reason"] == "length", c
        try:
            _chat(port, "Name three rivers.", response_format={"type": "json_object"})
            raise AssertionError("the fake cannot honour a constraint and must say so")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400 and json.loads(exc.read())["error"]["param"] == "response_format"
        # a server without the vision tower refuses an image with a 400 naming the part
        # (before this, the chat template silently wrote one placeholder and the image was lost)
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"}, img]}], "max_tokens": 8}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=30)
            raise AssertionError("an image was accepted by a server without a vision tower")
        except urllib.error.HTTPError as exc:
            err = json.loads(exc.read())["error"]
            assert exc.code == 400 and err["param"] == "messages[0].content[1].image_url", err
            assert "vision tower" in err["message"], err
        for ep, name in (("summary", "summary"), ("usage?bucket=hour", "usage"),
                         ("requests?limit=50", "requests"), ("system", "system"),
                         ("logs?follow=0&backlog=100", "logs-json")):
            got = _get(port, f"/v1/dashboard/{ep}")
            errs = contract_check.check(got, name)
            assert not errs, (ep, errs[:5])
        # VIS-54: without --backends the system answer says the proxy is off, so the Backends view
        # never asks /v1/config (which stays a 404 here)
        assert _get(port, "/v1/dashboard/system")["proxy"] == {"mode": "off", "backends": []}
        rows = _get(port, "/v1/dashboard/requests?limit=50")["requests"]
        assert len(rows) >= 9 and {"error", "tool_calls"} <= {r["finish_reason"] for r in rows}
        try:
            _get(port, "/v1/dashboard/summary", token=None)
            raise AssertionError("the dashboard answered without a token")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
        out = proc.stdout.read()
        assert "FAKE ENGINE" in out and "[ledger] on" in out and ADMIN not in out
        n = sqlite3.connect(ledger).execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        assert n == len(rows), (n, len(rows))
    finally:
        if proc.poll() is None:
            proc.kill()


PROXY = {"X-Forwarded-For": "203.0.113.44", "X-Real-IP": "203.0.113.44"}


def test_the_memory_policy_answers_503_before_a_prefill_it_cannot_hold():
    """OPS-47 over HTTP: with the engine at 76 GiB of a 77 GiB peak no prefill fits (its workspace
    is at least 1.15 GiB), so the request gets a 503 with a Retry-After before any response has
    started; at 60 GiB it is served. The limits are in /v1/dashboard/system and in /metrics."""
    for held, want in (("76", 503), ("60", 200)):
        port = _port()
        ledger = os.path.join(tempfile.mkdtemp(prefix="qse-e2e-"), "ledger.sqlite3")
        proc = _start(port, ledger, env_extra={"FAKE_MEM_GIB": held})
        try:
            _wait(port, proc)
            try:
                body = _chat(port, "Name three rivers.", max_tokens=8)
                got, retry, err = 200, None, body
            except urllib.error.HTTPError as exc:
                got, retry, err = exc.code, exc.headers.get("Retry-After"), json.loads(exc.read())
            assert got == want, (held, got, err)
            if want == 503:
                assert retry == "30" and "peak is 77.0 GiB" in err["error"]["message"], err
                metrics = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                                 timeout=10).read().decode()
                assert 'qse_requests_refused_total{reason="memory"} 1' in metrics, metrics[-2000:]
            sysinfo = _get(port, "/v1/dashboard/system")
            lim = sysinfo["memory"]["limits"]
            row = {r["key"]: r for r in lim["rows"]}
            assert row["MAX_LEN"]["applies"] == "locked" and row["MEM_PEAK_GIB"]["value"] == 77.0
            assert lim["budget"]["refused"] == (1 if want == 503 else 0)
            errs = contract_check.check(sysinfo, "system")
            assert not errs, errs[:5]
        finally:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)


def test_the_default_dashboard_login_is_off():
    """No secrets.env and no flag (AU1: no admin password): the dashboard's read API answers
    without a token and says there is no login and no password; the admin routes (the cache clear,
    /metrics with no metrics token, the full /health) answer through the proxy too."""
    port = _port()
    ledger = os.path.join(tempfile.mkdtemp(prefix="qse-e2e-"), "ledger.sqlite3")
    proc = _start(port, ledger, admin=None)
    try:
        _wait(port, proc)
        _chat(port, "Name three rivers.")
        for ep, name in (("summary", "summary"), ("usage?bucket=hour", "usage"),
                         ("requests?limit=50", "requests"), ("system", "system"),
                         ("logs?follow=0&backlog=100", "logs-json"), ("session", "session")):
            got = _get(port, f"/v1/dashboard/{ep}", token=None, headers=PROXY)
            errs = contract_check.check(got, name)
            assert not errs, (ep, errs[:5])
        sess = _get(port, "/v1/dashboard/session", token=None, headers=PROXY)
        assert sess["login"] is False and sess["password_set"] is False, sess
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/dashboard/metrics", headers=PROXY)
        with urllib.request.urlopen(req, timeout=10) as r:
            assert b"qse_requests" in r.read()
        for method, path in (("POST", "/v1/cache/clear"), ("GET", "/metrics"),
                             ("GET", "/v1/cache/stats")):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                         data=b"{}" if method == "POST" else None, headers=PROXY)
            with urllib.request.urlopen(req, timeout=10) as r:
                assert r.status == 200, (path, r.status)
        assert len(_get(port, "/health", token=None, headers=PROXY)) > 1, "the full body"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
        assert "admin password: not set" in proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_misspelt_login_setting_stops_the_start():
    proc = _start(_port(), "off", env_extra={"QSE_DASHBOARD_LOGIN": "true"})
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode != 0 and "QSE_DASHBOARD_LOGIN is on or off" in out, out[-500:]


def test_the_fake_engine_refuses_the_production_ledger():
    home = tempfile.mkdtemp(prefix="qse-home-")
    proc = _start(_port(), "~/.qwen38-spark-engine/usage/ledger.sqlite3", home=home)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode != 0 and "production ledger" in out, out[-500:]
    assert not os.path.exists(os.path.join(home, ".qwen38-spark-engine", "usage"))


# ------------------------------------------------------------------ the conductor over HTTP
# SRV-110 and SRV-112 over the wire, ENG-178 row 1 through the served loop's arms: the fake engine
# stops at the same safe points as the real loop (server/fake_engine.py), and the child registers
# one test rule, a tool-call check that passes everything, so a policy can hold calls.

LAUNCH_RULES = ("import sys, runpy; sys.modules['triton'] = None; sys.modules['triton.language'] = None;"
                "sys.path.insert(0, '.'); from server.conductor.monitor import tier0;"
                "tier0.register('t_pass', 'SP3', {})(lambda ev, task, sup: None);"
                "sys.argv = ['server/app.py'] + sys.argv[1:]; runpy.run_path('server/app.py', "
                "run_name='__main__')")
OPENCODE_TOOLS = json.load(open(os.path.join(ROOT, "tests", "fixtures", "opencode_tools.json")))["tools"]
GIVE_UP = "No tool call after two tries. Stopping this turn."


def _policy_file(tasks_extra: dict | None = None, only: dict | None = None) -> str:
    import yaml
    with open(os.path.join(ROOT, "ops", "supervisor.yaml")) as f:
        d = yaml.safe_load(f)
    if only is not None:
        d["tasks"] = only
    d["tasks"].update(tasks_extra or {})
    path = os.path.join(tempfile.mkdtemp(prefix="qse-pol-"), "policy.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(d, f)
    return path


def _start_conductor(port, mode, policy_path=None, tps="4000"):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ["HOME"],
           "PYTHONPATH": ROOT, "CUDA_VISIBLE_DEVICES": ""}
    ledger = os.path.join(tempfile.mkdtemp(prefix="qse-e2e-"), "ledger.sqlite3")
    args = ["--fake-engine", "--port", str(port), "--served-model", "qwen38-spark-engine",
            "--usage-ledger", ledger, "--fake-tps", tps, "--conductor", mode]
    if policy_path:
        args += ["--conductor-policy", policy_path]
    return subprocess.Popen([sys.executable, "-c", LAUNCH_RULES, *args], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _post(port, body, headers=None, stream=None):
    """(response headers, parsed body): the chunks of a stream, or the JSON answer."""
    body = dict(body, **({"stream": stream} if stream is not None else {}))
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=dict({"Content-Type": "application/json"},
                                              **(headers or {})))
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode()
        head = dict(r.headers)
    if not body.get("stream"):
        return head, json.loads(raw)
    return head, [json.loads(ln[6:]) for ln in raw.splitlines() if ln.startswith("data: {")]


def _wire(chunks) -> dict:
    """What a streamed client holds: content, reasoning, the calls by index, the finish."""
    out = {"content": "", "reasoning": "", "calls": {}, "finish": None, "deltas": 0}
    for c in chunks:
        for ch in c["choices"]:
            d = ch["delta"]
            out["content"] += d.get("content") or ""
            out["reasoning"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                out["deltas"] += 1
                call = out["calls"].setdefault(tc["index"], {"id": None, "name": "", "args": ""})
                call["id"] = tc.get("id") or call["id"]
                call["name"] += (tc.get("function") or {}).get("name") or ""
                call["args"] += (tc.get("function") or {}).get("arguments") or ""
            out["finish"] = ch["finish_reason"] or out["finish"]
    return out


def _stop(proc) -> str:
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 0
    return proc.stdout.read()


#: SRV-617: the release's tags, named (an unnamed raw client's default is `reasoning_content`)
TAGS = {"reasoning_format": "tags"}


def _chat_msgs(text):
    return {"messages": [{"role": "system", "content": "You are opencode."},
                         {"role": "user", "content": text}], "max_tokens": 400}


SMOKE = [(_chat_msgs("Name three rivers."), None),
         (dict(_chat_msgs("use the tool now"), tools=OPENCODE_TOOLS), None),
         (dict(_chat_msgs("use the tool FAKE_TWO_CALLS"), tools=OPENCODE_TOOLS), None),
         (_chat_msgs("think then answer FAKE_LONG"), {"reasoning_format": "reasoning_content"}),
         (dict(_chat_msgs("plain"), chat_template_kwargs={"enable_thinking": False}), None)]


def _normal(x):
    """A response with its ids and times taken out, for comparing two servers' answers."""
    if isinstance(x, dict):
        return {k: _normal(v) for k, v in x.items()
                if k not in ("id", "created", "usage", "timings", "metrics", "system_fingerprint")}
    if isinstance(x, list):
        return [_normal(v) for v in x]
    return x


def test_conductor_raw_and_on_with_no_task_answer_the_same():
    """SRV-110 rows 1-3 over HTTP: every response says its mode; a server `on` with every task
    off answers the smoke set exactly as raw does, streamed and not; X-Tandem: off lowers."""
    answers = {}
    for mode in ("raw", "on"):
        port = _port()
        proc = _start_conductor(port, mode, _policy_file(only={}) if mode == "on" else None)
        try:
            _wait(port, proc)
            got = []
            for body, extra in SMOKE:
                for stream in (False, True):
                    head, parsed = _post(port, dict(body, **(extra or {})), stream=stream)
                    assert head.get("X-Tandem") == mode, head
                    got.append(_normal(parsed))
            head, _ = _post(port, SMOKE[0][0], headers={"X-Tandem": "off"})
            assert head.get("X-Tandem") == "raw"
            head, _ = _post(port, SMOKE[0][0], headers={"X-Tandem": "on"})
            assert head.get("X-Tandem") == mode, "a header never raises the mode"
            answers[mode] = got
            out = _stop(proc)
            assert "[conductor] " + mode in out and "decision=" not in out, out[-2000:]
        finally:
            if proc.poll() is None:
                proc.kill()
    assert answers["raw"] == answers["on"], "on with no task must answer what raw answers"


def test_conductor_shadow_never_acts():
    """SRV-110 row 4, SRV-112 row 7, ENG-178 row 7: in shadow the empty stop and the reasoning
    loop are journaled as what would have been done, and the client gets raw's answer."""
    cases = [dict(_chat_msgs("use the tool FAKE_EMPTY_THINK"), tools=OPENCODE_TOOLS),
             dict(_chat_msgs("use the tool FAKE_EMPTY_ANSWER"), tools=OPENCODE_TOOLS),
             dict(_chat_msgs("FAKE_LOOP"), max_tokens=8000)]
    answers = {}
    for mode in ("raw", "shadow"):
        port = _port()
        proc = _start_conductor(port, mode, _policy_file() if mode == "shadow" else None)
        try:
            _wait(port, proc)
            answers[mode] = [_normal(_post(port, body, stream=s)[1]) for body in cases
                             for s in (False, True)]
            out = _stop(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
    assert answers["raw"] == answers["shadow"]
    lines = [ln for ln in out.splitlines() if ln.startswith("[conductor] chatcmpl")]
    assert any("task=E1 mode=shadow decision=CONTINUE_HIDDEN reason=empty_turn(case=A)" in ln
               for ln in lines), lines
    assert any("task=E1 mode=shadow decision=CONTINUE_HIDDEN reason=empty_turn(case=B)" in ln
               for ln in lines), lines
    assert any("task=D1 mode=shadow decision=CLOSE_REASONING reason=repeat16(" in ln
               for ln in lines), lines
    assert not any("mode=act" in ln for ln in lines), lines


def test_conductor_on_over_the_wire():
    """The shipped policy (D1, D2, E1 acting) plus a tool-call check that holds calls:

      SRV-112 1  the end token inside the reasoning: `</think>` then the call, finish tool_calls,
                 no close phrase on the wire -- streamed and not, opencode's tools
      SRV-112 2  an empty answer after the reasoning: the call, and no second `<think>`
      SRV-112 3  empty again after two acts: the GIVE_UP sentence, finish stop
      SRV-112 4-6  a text answer, a request without tools, tool_choice none: nothing done
      SRV-110 5  a hidden close (D1 on a reasoning loop): the reasoning, one `</think>`, the
                 answer; in reasoning_content no tag at all; the phrase never on the wire
      SRV-110 6  two held calls: each whole in one delta, a fresh id, indexes 0 and 1
    """
    port = _port()
    proc = _start_conductor(port, "on", _policy_file({"C": {"rule": "t_pass", "mode": "act"}}))
    phrase = "Considering the limited time"
    try:
        _wait(port, proc)
        for stream in (True, False):
            for word in ("FAKE_EMPTY_THINK", "FAKE_EMPTY_ANSWER"):
                # SRV-617: the tags named, as an unnamed raw client gets `reasoning_content`
                head, got = _post(port, dict(_chat_msgs(f"use the tool {word}"), **TAGS,
                                             tools=OPENCODE_TOOLS), stream=stream)
                assert head["X-Tandem"] == "on"
                if stream:
                    w = _wire(got)
                    content, calls, finish = w["content"], w["calls"], w["finish"]
                    assert w["deltas"] == 1, "a held call arrives whole, in one delta"
                    names = [c["name"] for c in calls.values()]
                else:
                    msg = got["choices"][0]["message"]
                    content, finish = msg["content"], got["choices"][0]["finish_reason"]
                    names = [c["function"]["name"] for c in msg.get("tool_calls") or []]
                assert finish == "tool_calls" and names == ["bash"], (word, stream, got)
                assert phrase not in content and content.count("<think>") == 1, content
                assert content.count("</think>") == 1, content
        # two failures: the GIVE_UP sentence
        for stream in (True, False):
            _, got = _post(port, dict(_chat_msgs("use the tool FAKE_EMPTY_ANSWER FAKE_EMPTY_ALWAYS"),
                                      **TAGS, tools=OPENCODE_TOOLS), stream=stream)
            content, finish = ((_wire(got)["content"], _wire(got)["finish"]) if stream else
                               (got["choices"][0]["message"]["content"],
                                got["choices"][0]["finish_reason"]))
            assert content.endswith(GIVE_UP) and finish == "stop", (stream, content, finish)
        # nothing to do: a text answer, no tools, tool_choice none
        _, got = _post(port, dict(_chat_msgs("Name three rivers."), **TAGS, tools=OPENCODE_TOOLS))
        assert got["choices"][0]["finish_reason"] == "stop"
        assert got["choices"][0]["message"]["content"].split("</think>")[-1].strip()
        for body in (dict(_chat_msgs("FAKE_EMPTY_ANSWER"), **TAGS),
                     dict(_chat_msgs("use the tool FAKE_EMPTY_ANSWER"), **TAGS,
                          tools=OPENCODE_TOOLS, tool_choice="none")):
            _, got = _post(port, body)
            c = got["choices"][0]
            assert c["finish_reason"] == "stop" and not c["message"].get("tool_calls"), c
            assert c["message"]["content"].split("</think>")[-1].strip() == "", c
        # the hidden close on a reasoning loop
        for fmt in ("tags", "reasoning_content"):
            _, got = _post(port, dict(_chat_msgs("FAKE_LOOP"), reasoning_format=fmt,
                                      max_tokens=8000), stream=True)
            w = _wire(got)
            if fmt == "tags":
                assert w["content"].count("</think>") == 1 and w["content"].split("</think>")[1]
                assert len(w["content"].split("</think>")[0]) < 3200, "closed near 2,112"
            else:
                assert "</think>" not in w["content"] + w["reasoning"] and w["content"]
            assert phrase not in w["content"] + w["reasoning"]
        # two held calls
        _, got = _post(port, dict(_chat_msgs("use the tool FAKE_TWO_CALLS"), tools=OPENCODE_TOOLS),
                       stream=True)
        w = _wire(got)
        assert sorted(w["calls"]) == [0, 1] and w["deltas"] == 2, w
        assert all(c["id"].startswith("call_") for c in w["calls"].values())
        assert [c["name"] for _, c in sorted(w["calls"].items())] == [
            "bash", "chrome-devtools_navigate_page"], w
        assert all(json.loads(c["args"]) for c in w["calls"].values())
        out = _stop(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
    lines = [ln for ln in out.splitlines() if ln.startswith("[conductor] chatcmpl")]
    assert sum("reason=empty_turn(case=A)" in ln and "mode=act" in ln for ln in lines) == 2
    assert sum("reason=empty_turn(case=B)" in ln and "mode=act" in ln for ln in lines) >= 2
    assert sum("decision=GIVE_UP reason=empty_turn(tries=2)" in ln for ln in lines) == 2, lines
    assert sum("task=D1 mode=act decision=CLOSE_REASONING reason=repeat16(" in ln
               for ln in lines) == 2, lines
    assert "conductor=on acts=1 hidden=" in out and " gave_up" in out


def test_conductor_the_openai_sdk_takes_a_continued_turn():
    """SRV-112 row 1 with the official client: the stream of a turn the monitor continued
    (the end token inside the reasoning) parses, and its call assembles whole."""
    try:
        import openai
    except ImportError:
        print("  (the OpenAI SDK is not installed here: skipped)")
        return
    port = _port()
    proc = _start_conductor(port, "on", _policy_file())
    try:
        _wait(port, proc)
        client = openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="none")
        stream = client.chat.completions.create(
            model="qwen38-spark-engine", stream=True, tools=OPENCODE_TOOLS, max_tokens=400,
            messages=[{"role": "user", "content": "use the tool FAKE_EMPTY_THINK"}],
            stream_options={"include_usage": True})
        name, args, finish, content, reasoning = "", "", None, "", ""
        for ch in stream:
            for c in ch.choices:
                content += c.delta.content or ""
                reasoning += getattr(c.delta, "reasoning_content", None) or ""
                for tc in c.delta.tool_calls or []:
                    name += (tc.function.name or "") if tc.function else ""
                    args += (tc.function.arguments or "") if tc.function else ""
                finish = c.finish_reason or finish
        assert finish == "tool_calls" and name == "bash" and json.loads(args), (finish, name, args)
        # SRV-617: a plain API client's thinking is in `reasoning_content`, no tag in `content`
        assert "</think>" not in content + reasoning and reasoning, (content, reasoning)
        assert "Considering" not in content + reasoning, content
        _stop(proc)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_tool_checks_over_the_wire():
    """SRV-114 over a socket, with the shipped policy (C4 and C3 act, C1 and C2 in shadow) and
    the OpenAI SDK where it is installed: an unreadable block never reaches the client and the
    call written again arrives whole at index 0 (rows 2, 8); two unreadable regenerations end
    as content (row 3); a third identical call is replaced (row 4); a risky command arrives
    unchanged (row 6); every act is in /metrics."""
    port = _port()
    proc = _start_conductor(port, "on", _policy_file())
    try:
        _wait(port, proc)
        tools = OPENCODE_TOOLS
        for stream in (True, False):
            _, got = _post(port, dict(_chat_msgs("call it FAKE_BAD_CALL"), tools=tools,
                                      chat_template_kwargs={"enable_thinking": False}),
                           stream=stream)
            if stream:
                w = _wire(got)
                assert sorted(w["calls"]) == [0] and w["deltas"] == 1, w
                assert "<tool_call>" not in w["content"] and w["finish"] == "tool_calls", w
            else:
                m = got["choices"][0]["message"]
                assert len(m["tool_calls"]) == 1 and "<tool_call>" not in (m["content"] or "")
            _, got = _post(port, dict(_chat_msgs("call it FAKE_BAD_ALWAYS"), tools=tools,
                                      chat_template_kwargs={"enable_thinking": False}),
                           stream=stream)
            content = _wire(got)["content"] if stream else got["choices"][0]["message"]["content"]
            assert content.count("<tool_call>") == 1 and '"broken"' in content, content
        same = [{"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{i}", "type": "function", "function": {
                "name": "bash", "arguments": '{"command": "python3 pagodacheck.py"}'}}]}
            for i in range(2)]
        hist = []
        for i, m in enumerate(same):
            hist += [m, {"role": "tool", "tool_call_id": f"call_{i}", "content": "RESULT: FAIL"}]
        body = {"messages": [{"role": "user", "content": 'FAKE_CALL_JSON {"name": "bash", '
                                                         '"arguments": {"command": "python3 '
                                                         'pagodacheck.py"}}'}] + hist,
                "tools": tools, "max_tokens": 400,
                "chat_template_kwargs": {"enable_thinking": False}}
        _, got = _post(port, body, stream=True)
        w = _wire(got)
        assert len(w["calls"]) == 1 and "pagodacheck" not in w["calls"][0]["args"], w
        body["messages"] = [{"role": "system", "content": "Working directory: /home/u/proj"},
                            {"role": "user", "content": 'FAKE_CALL_JSON {"name": "bash", '
                                                        '"arguments": {"command": "rm -rf '
                                                        '~/pago"}}'}]
        _, got = _post(port, body, stream=True)
        assert json.loads(_wire(got)["calls"][0]["args"]) == {"command": "rm -rf ~/pago"}
        try:
            import openai
        except ImportError:
            openai = None
            print("  (the OpenAI SDK is not installed here: the SDK half is skipped)")
        if openai is not None:
            client = openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="none")
            calls: dict = {}
            finish = None
            for ch in client.chat.completions.create(
                    model="qwen38-spark-engine", stream=True, tools=tools, max_tokens=400,
                    messages=[{"role": "user", "content": "use the tool FAKE_TWO_CALLS "
                                                          "FAKE_SECOND_BAD"}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}):
                for c in ch.choices:
                    for tc in c.delta.tool_calls or []:
                        cur = calls.setdefault(tc.index, {"id": tc.id, "name": "", "args": ""})
                        cur["name"] += tc.function.name or ""
                        cur["args"] += tc.function.arguments or ""
                    finish = c.finish_reason or finish
            assert sorted(calls) == [0] and finish == "tool_calls", calls
            assert calls[0]["id"].startswith("call_") and json.loads(calls[0]["args"])
        scrape = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read()
        scrape = scrape.decode()
        assert 'qse_conductor_actions_total{task="C4",action="REGENERATE_CALL"}' in scrape
        assert 'qse_conductor_actions_total{task="C3",action="REGENERATE_CALL"} 1' in scrape
        out = _stop(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert "task=C1 mode=shadow decision=REGENERATE_CALL reason=floor(rm_rf_outside_root)" in \
        out, out[-3000:]


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
