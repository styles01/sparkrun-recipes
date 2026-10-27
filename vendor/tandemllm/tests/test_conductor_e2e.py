"""The conductor end to end over HTTP: the real server with the fake engine (server/fake_engine.py),
a fake decider process (server/conductor/fake_decider.py, the Jev HTTP API) as both classifier roles, and a second
server without the conductor for the raw byte check. SRV-129/130, SRV-117/118, SRV-119/120.

What a client sees and what the model saw are both checked. The fake engine answers from the last
user turn it is given, and its reasoning line counts that turn's characters, so a note injected
into the turn shows up as a larger count while the stream carries no byte of the note.

  * one model: /v1/models lists TandemLLM alone, the raw engine's old name answers on
    /v1/models/{id} as TandemLLM's card and in a request runs as TandemLLM; `X-Tandem: raw` (on
    TandemLLM or on the alias) and the locked `bench` profile stream exactly what a server without
    the conductor streams, and say `X-Tandem: raw`;
  * a design request: class design_ui from the decider, the ui-design text in the model's turn,
    none of it in the stream or the usage, the journal holding the exact text and position;
  * a tool-result turn asks the decider nothing; the decider down: Qwen decides;
  * a summary at effort medium: thinking off through the template (no reasoning in the answer);
  * `X-Tandem-Profile: nope` is a 400 naming the header; the ledger row records the profile and
    `model_served = TandemLLM` for every request, raw ones included;
  * the config API: 401 through the proxy without a token, a write with the admin token makes the
    next version, raw is 409, invalid is 422, restore, diff, the mode switch, a label;
  * the session export lists every hidden text; stats, sessions and the test box answer, and every
    response validates against docs/contract/conductor-v1;
  * any decision model works: another model name with limits 256/96 gets states of at most
    (256 - 96) x 3.5 characters.

Run: python tests/test_conductor_e2e.py
"""

from __future__ import annotations

import json
import os
import re
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

from tools import contract_check  # noqa: E402

CONTRACT = os.path.join(ROOT, "docs", "contract", "conductor-v1")
ADMIN = "e2e-" + "0" * 40
PROXY = {"X-Forwarded-For": "203.0.113.44", "X-Real-IP": "203.0.113.44"}
LAUNCH = ("import sys, runpy; sys.modules['triton'] = None; sys.modules['triton.language'] = None;"
          "sys.argv = ['server/app.py'] + sys.argv[1:]; runpy.run_path('server/app.py', "
          "run_name='__main__')")
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run a command",
                                           "parameters": {"type": "object", "properties": {
                                               "command": {"type": "string"}}}}}]


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _env():
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ["HOME"],
            "PYTHONPATH": ROOT, "CUDA_VISIBLE_DEVICES": "", "QSE_ADMIN_TOKEN": ADMIN}


def _server(port, conductor_dir=None, extra=()):
    ledger = os.path.join(tempfile.mkdtemp(prefix="qse-cond-"), "ledger.sqlite3")
    args = [sys.executable, "-c", LAUNCH, "--fake-engine", "--port", str(port),
            "--served-model", "qwen38-spark-engine", "--usage-ledger", ledger,
            "--fake-tps", "2000", *extra]
    if conductor_dir:
        args += ["--conductor-config", conductor_dir]
    proc = subprocess.Popen(args, cwd=ROOT, env=_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return proc, ledger


def _decider(port, *extra):
    return subprocess.Popen([sys.executable, os.path.join(ROOT, "server", "conductor",
                                                          "fake_decider.py"),
                             "--port", str(port), *extra], cwd=ROOT, env=_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait(port, proc, path="/health"):
    end = time.time() + 60
    while time.time() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2)
            return
        except OSError:
            if proc.poll() is not None:
                raise AssertionError(proc.stdout.read())
            time.sleep(0.2)
    raise AssertionError(f"port {port} did not come up")


def _call(port, method, path, body=None, headers=None, token=ADMIN, raw=False):
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    if data is not None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers=h)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = r.read()
            return r.status, dict(r.headers), (payload if raw else json.loads(payload or b"{}"))
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), json.loads(exc.read() or b"{}")


def _stream(port, messages, model="TandemLLM", headers=None, tools=True, **extra):
    """(the streamed text, the response headers, the usage) of one streamed chat request."""
    body = {"model": model, "messages": messages, "stream": True, "max_tokens": 400, **extra}
    if tools:
        body["tools"] = TOOLS
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=h)
    text, usage = [], None
    with urllib.request.urlopen(req, timeout=60) as r:
        hdrs = dict(r.headers)
        for line in r.read().decode().splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                text.append(d.get("content") or "")
                text.append(d.get("reasoning_content") or "")
                for tc in d.get("tool_calls") or []:
                    text.append(json.dumps(tc))
            usage = ev.get("usage") or usage
    return "".join(text), hdrs, usage


def _check(obj, name):
    errs = contract_check.validate(obj, contract_check.load(name, CONTRACT), root=CONTRACT)
    assert not errs, (name, errs[:5])


def _decisions(port, **q):
    qs = "&".join(f"{k}={v}" for k, v in q.items())
    st, _, got = _call(port, "GET", f"/v1/conductor/decisions?{qs}")
    assert st == 200, got
    _check(got, "decisions")
    return got["decisions"]


def test_the_conductor_end_to_end():
    port, arb_port, raw_port = _port(), _port(), _port()
    cdir = tempfile.mkdtemp(prefix="qse-conductor-")
    arb = _decider(arb_port)
    srv, ledger = _server(port, cdir)
    plain, _ = _server(raw_port)
    try:
        _wait(arb_port, arb, "/readyz")
        _wait(port, srv)
        _wait(raw_port, plain)
        # ---- the roles point at the fake: the Jev API and the Qwen stage's chat endpoint
        st, _, roles = _call(port, "GET", "/v1/conductor/roles")
        assert st == 200
        _check(roles, "roles")
        doc = roles["doc"]
        doc["decider"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        doc["classifier"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        st, _, w = _call(port, "PUT", "/v1/conductor/roles", {"doc": doc, "note": "the fake"})
        assert st == 200 and w["version"] == 1, w
        _check(w, "write")

        # ---- one model; raw is a header, and the raw path is the release's, byte for byte
        st, _, models = _call(port, "GET", "/v1/models", token=None)
        assert [m["id"] for m in models["data"]] == ["TandemLLM"], models
        st, _, card = _call(port, "GET", "/v1/models/qwen38-spark-engine", token=None)
        assert st == 200 and card["id"] == "TandemLLM", (st, card)
        st, _, card = _call(port, "GET", "/v1/models/TandemLLM", token=None)
        assert st == 200 and card["id"] == "TandemLLM", (st, card)
        st, _, err = _call(port, "GET", "/v1/models/gpt-4o", token=None)
        assert st == 404 and err["error"]["code"] == "model_not_found", (st, err)
        st, _, plain_models = _call(raw_port, "GET", "/v1/models", token=None)
        assert [m["id"] for m in plain_models["data"]] == ["qwen38-spark-engine"]
        msgs = [{"role": "user", "content": "Name three rivers in Europe."}]
        b, h_plain, _ = _stream(raw_port, msgs, model="qwen38-spark-engine")
        assert h_plain["X-Tandem"] == "raw" and "X-Tandem-Profile" not in h_plain
        for name, hdr in (("TandemLLM", {"X-Tandem": "raw"}),
                          ("qwen38-spark-engine", {"X-Tandem": "raw"}),
                          ("TandemLLM", {"X-Tandem-Profile": "bench"})):
            a, h_raw, _ = _stream(port, msgs, model=name, headers=hdr)
            assert a == b and h_raw["X-Tandem"] == "raw", (name, hdr, h_raw)
        # every public surface names the one model; the base engine's name is `base_model`
        st, _, h = _call(port, "GET", "/health", token=None)
        assert (h["model"], h["base_model"]) == ("TandemLLM", "qwen38-spark-engine"), h
        assert h["cache"]["model"] == "TandemLLM", h.get("cache")
        st, _, h = _call(raw_port, "GET", "/health", token=None)
        assert h["model"] == "qwen38-spark-engine", "without the conductor: the served name"
        st, _, sysd = _call(port, "GET", "/v1/dashboard/system")
        assert st == 200 and (sysd["engine"]["model"], sysd["engine"]["base_model"]) == \
            ("TandemLLM", "qwen38-spark-engine"), sysd.get("engine")
        dash = os.path.join(ROOT, "docs", "contract", "dashboard-v1")
        assert not contract_check.validate(sysd, contract_check.load("system", dash), root=dash)
        one = {"model": "qwen38-spark-engine", "max_tokens": 16, "temperature": 0,
               "messages": [{"role": "user", "content": "Say ok."}]}
        st, _, got = _call(port, "POST", "/v1/chat/completions", one, headers={"X-Tandem": "raw"})
        assert st == 200 and got["model"] == "TandemLLM", (st, got.get("model"))
        st, _, got = _call(raw_port, "POST", "/v1/chat/completions", one)
        assert st == 200 and got["model"] == "qwen38-spark-engine", "the plain server echoes"
        st, _, live = _call(port, "GET", "/v1/dashboard/live?follow=0")
        names = {r["model"] for r in live["requests"]}
        assert names == {"TandemLLM"}, names
        assert live["engine"]["model"] == "TandemLLM", live["engine"]
        # the alias without a header is TandemLLM under its profile, not the raw engine
        a, h_alias, _ = _stream(port, [{"role": "user", "content": "Plan it. FAKE_CLASS=plan"}],
                                model="qwen38-spark-engine", headers={"x-session-id": "alias"})
        assert h_alias["X-Tandem"] == "on" and h_alias["X-Tandem-Profile"] == "chat", h_alias
        assert _decisions(port, session="c:alias")[0]["class_id"] == "plan"

        # ---- a design request: the skill goes to the model, never to the client
        user = "Make the settings page calmer. FAKE_CLASS=design_ui"
        msgs = [{"role": "system", "content": "You are opencode."},
                {"role": "user", "content": user}]
        text, hdrs, usage = _stream(port, msgs, headers={"x-session-id": "ses-e2e"})
        assert hdrs["X-Tandem"] == "on" and hdrs["X-Tandem-Profile"] == "chat", hdrs
        assert "runtime-note" not in text and "UI design" not in text, text[:300]
        row = _decisions(port, session="c:ses-e2e")[0]
        assert (row["class_id"], row["source"], row["confidence"]) == ("design_ui", "decider",
                                                                       0.92)
        note = row["injected"][0]["text"]
        assert "# UI design" in note and row["injected"][0]["pos"] > 0
        seen = int(re.search(r"The question has (\d+) characters", text).group(1))
        assert seen == len(user) + len(note), (seen, len(user), len(note))
        # SRV-165 (UT1): `prompt_tokens` is what the client sent, the server without the conductor
        # counts the same messages the same; the note is `hidden_tokens`, never in the count
        raw_usage = _stream(raw_port, msgs, model="qwen38-spark-engine")[2]
        assert usage["prompt_tokens"] == raw_usage["prompt_tokens"], (usage, raw_usage)
        hidden1 = usage["prompt_tokens_details"]["hidden_tokens"]
        assert hidden1 >= row["injected"][0]["tokens"] > 0, (usage, row["injected"][0])
        assert usage["prompt_tokens_details"]["cached_tokens"] <= usage["prompt_tokens"], usage
        assert row["outcome"]["finish_reason"] in ("stop", "tool_calls"), row["outcome"]

        # ---- a tool turn asks nobody; the note is replayed
        _call(arb_port, "POST", "/__calls", {}, token=None)
        msgs += [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash",
                                                          "arguments": "{\"command\": \"ls\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "index.html\nstyle.css"}]
        _, _, usage2 = _stream(port, msgs, headers={"x-session-id": "ses-e2e"})
        raw2 = _stream(raw_port, msgs, model="qwen38-spark-engine")[2]
        assert usage2["prompt_tokens"] == raw2["prompt_tokens"], (usage2, raw2)
        assert usage2["prompt_tokens_details"]["hidden_tokens"] >= hidden1, usage2   # replayed
        st, _, calls = _call(arb_port, "GET", "/__calls", token=None)
        assert calls["systemone"] == 0 and calls["chat"] == 0, calls
        row = _decisions(port, session="c:ses-e2e")[0]
        assert row["source"] == "sticky" and row["replayed"]["count"] == 1 and not row["injected"]

        # ---- the decider down: Qwen decides
        doc["decider"]["endpoint"] = "http://127.0.0.1:9"
        _call(port, "PUT", "/v1/conductor/roles", {"doc": doc, "note": "decider gone"})
        text, _, _ = _stream(port, [{"role": "user", "content": "It crashes. FAKE_QWEN=debug"}],
                             headers={"x-session-id": "ses-2"})
        row = _decisions(port, session="c:ses-2")[0]
        assert (row["class_id"], row["source"]) == ("debug", "qwen"), row
        assert row["stages"][0]["error"] == "decider_unreachable"
        doc["decider"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        _call(port, "PUT", "/v1/conductor/roles", {"doc": doc, "note": "decider back"})

        # ---- a summary thinks off: the generation prompt's <think> is closed at the tail
        text, _, _ = _stream(port, [{"role": "user", "content": "Shorten this. FAKE_CLASS=summarize"}],
                             tools=False, headers={"x-session-id": "ses-3"})
        row = _decisions(port, session="c:ses-3")[0]
        assert row["thinking"] == "off" and "The question has" not in text, text[:200]

        # ---- an unknown profile; the ledger row
        st, _, err = _call(port, "POST", "/v1/chat/completions",
                           {"model": "TandemLLM", "messages": [{"role": "user", "content": "x"}]},
                           headers={"X-Tandem-Profile": "nope"}, token=None)
        assert st == 400 and err["error"]["param"] == "X-Tandem-Profile", err

        # ---- the config API
        st, _, err = _call(port, "PUT", "/v1/conductor/profiles/chat",
                           {"doc": {}, "note": "x"}, headers=PROXY, token=None)
        assert st == 401, (st, err)
        st, _, prof = _call(port, "GET", "/v1/conductor/profiles/chat", headers=PROXY, token=None)
        assert st == 200, "reads follow the dashboard rule (login off)"
        _check(prof, "document")
        pdoc = prof["doc"]
        pdoc["decisions"]["rules"] = "on"
        st, _, w = _call(port, "PUT", "/v1/conductor/profiles/chat",
                         {"doc": pdoc, "note": "rules on", "base_version": prof["version"]},
                         headers=PROXY)
        assert st == 200 and w["version"] == 1, w
        st, _, e409 = _call(port, "PUT", "/v1/conductor/profiles/raw",
                            {"doc": pdoc, "note": "try"})
        assert st == 409, e409
        bad = dict(pdoc, mode="loud")
        st, _, e422 = _call(port, "PUT", "/v1/conductor/profiles/chat", {"doc": bad, "note": "x"})
        assert st == 422 and "mode" in e422["error"]["message"], e422
        _check(e422, "error")
        st, _, vs = _call(port, "GET", "/v1/conductor/profiles/chat/versions")
        _check(vs, "versions")
        assert [v["version"] for v in vs["versions"]] == [1, 0]
        st, _, df = _call(port, "GET", "/v1/conductor/profiles/chat/diff?from=0&to=1")
        _check(df, "diff")
        assert [c["path"] for c in df["changes"]] == ["$.decisions.rules"]
        st, _, w = _call(port, "POST", "/v1/conductor/profiles/chat/restore", {"version": 0})
        assert st == 200 and w["version"] == 2 and w["restored_from"] == 0
        st, _, w = _call(port, "POST", "/v1/conductor/profiles/chat/mode", {"mode": "shadow"})
        assert st == 200 and w["mode"] == "shadow"
        text, hdrs, _ = _stream(port, [{"role": "user", "content": "Style it. FAKE_CLASS=design_ui"}],
                                headers={"x-session-id": "ses-4"})
        assert hdrs["X-Tandem"] == "shadow"
        row = _decisions(port, session="c:ses-4")[0]
        assert not row["injected"] and row["shadow"]["notes"], row
        st, _, profs = _call(port, "GET", "/v1/conductor/profiles")
        _check(profs, "profiles")
        assert {p["name"]: p["mode"] for p in profs["profiles"]}["chat"] == "shadow"

        # ---- labels, sessions, the export, stats, the test box, config, meta
        rid = _decisions(port, session="c:ses-e2e")[-1]["id"]
        st, _, lab = _call(port, "POST", f"/v1/conductor/decisions/{rid}/label",
                           {"label": "right"})
        assert st == 200
        time.sleep(0.8)
        st, _, one = _call(port, "GET", f"/v1/conductor/decisions/{rid}")
        assert one["decision"]["label"] == "right"
        st, _, ss = _call(port, "GET", "/v1/conductor/sessions")
        _check(ss, "sessions")
        assert "c:ses-e2e" in {s["key"] for s in ss["sessions"]}
        st, _, tl = _call(port, "GET", "/v1/conductor/sessions/c%3Ases-e2e")
        _check(tl, "session")
        assert len(tl["turns"]) == 2
        st, hdr, md = _call(port, "GET", "/v1/conductor/sessions/c%3Ases-e2e/export?format=md",
                            raw=True)
        md = md.decode()
        assert st == 200 and "1 intervention in 2 turns" in md and "# UI design" in md, md[:400]
        st, _, ex = _call(port, "GET", "/v1/conductor/sessions/c%3Ases-e2e/export")
        _check(ex, "export")
        assert ex["interventions"] == 1 and ex["injections"][0]["text"] == note
        st, _, stats = _call(port, "GET", "/v1/conductor/stats?hours=24")
        _check(stats, "stats")
        assert stats["decisions"] >= 5 and stats["by_class"]["design_ui"] >= 1
        st, _, ex = _call(port, "POST", "/v1/conductor/classify",
                          {"text": "Restyle the header. FAKE_CLASS=design_ui", "profile": "coding"},
                          headers=PROXY, token=None)
        assert st == 200 and ex["class_id"] == "design_ui" and "# UI design" in ex["note"], ex
        _check(ex, "classify")
        for ep, name in (("config", "config"), ("meta", "meta"), ("router", "document"),
                         ("packages", "document"), ("evals", "evals")):
            st, _, got = _call(port, "GET", f"/v1/conductor/{ep}")
            assert st == 200, (ep, got)
            _check(got, name)

        # ---- the ledger row records the profile and the mode
        srv.send_signal(signal.SIGTERM)
        assert srv.wait(timeout=30) == 0
        out = srv.stdout.read()
        assert "[conductor] on" in out and ADMIN not in out
        rows = sqlite3.connect(ledger).execute(
            "SELECT model, model_served, profile, conductor_mode FROM requests").fetchall()
        assert ("TandemLLM", "TandemLLM", "chat", "on") in rows, rows
        assert ("qwen38-spark-engine", "TandemLLM", "chat", "on") in rows, rows
        assert ("TandemLLM", "TandemLLM", "chat", "raw") in rows, rows
        assert ("qwen38-spark-engine", "TandemLLM", "chat", "raw") in rows, rows
        assert ("TandemLLM", "TandemLLM", "bench", "raw") in rows, rows
        assert {r[1] for r in rows} == {"TandemLLM"}, "one served name"
    finally:
        for p in (srv, plain, arb):
            if p.poll() is None:
                p.kill()


def test_any_decision_model_works():
    """The decider is model-agnostic: a model under another name with smaller limits is used unchanged,
    and every state it gets fits its limits."""
    port, arb_port = _port(), _port()
    cdir = tempfile.mkdtemp(prefix="qse-conductor-")
    arb = _decider(arb_port, "--model", "other-decider", "--context", "256", "--question", "96")
    srv, _ = _server(port, cdir)
    try:
        _wait(arb_port, arb, "/readyz")
        _wait(port, srv)
        st, _, roles = _call(port, "GET", "/v1/conductor/roles")
        doc = roles["doc"]
        doc["decider"].update(endpoint=f"http://127.0.0.1:{arb_port}", model="other-decider")
        doc["classifier"]["mode"] = "off"
        _call(port, "PUT", "/v1/conductor/roles", {"doc": doc, "note": "another model"})
        long = "FAKE_CLASS=plan " + "Plan the migration of the billing service. " * 60
        _stream(port, [{"role": "user", "content": long}], headers={"x-session-id": "x"})
        st, _, states = _call(arb_port, "GET", "/__states", token=None)
        assert states["states"] and all(len(s) <= int((256 - 96) * 3.5)
                                        for s in states["states"]), \
            [len(s) for s in states["states"]]
        row = _decisions(port, session="c:x")[0]
        assert row["class_id"] == "plan" and row["stages"][0]["model"] == "other-decider"
    finally:
        for p in (srv, arb):
            if p.poll() is None:
                p.kill()


def test_the_monitor_and_the_router_together():
    """`--conductor on` (the frame's session monitor) with the config on: `X-Tandem: raw` is
    still the plain server's stream byte for byte and says raw, while TandemLLM runs both."""
    port, raw_port, arb_port = _port(), _port(), _port()
    cdir = tempfile.mkdtemp(prefix="qse-conductor-")
    arb = _decider(arb_port)
    srv, _ = _server(port, cdir, extra=("--conductor", "on"))
    plain, _ = _server(raw_port)
    try:
        _wait(arb_port, arb, "/readyz")
        _wait(port, srv)
        _wait(raw_port, plain)
        st, _, roles = _call(port, "GET", "/v1/conductor/roles")
        doc = roles["doc"]
        doc["decider"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        doc["classifier"]["mode"] = "off"
        _call(port, "PUT", "/v1/conductor/roles", {"doc": doc, "note": "the fake"})
        msgs = [{"role": "user", "content": "Use the tool to name three rivers."}]
        a, h_a, _ = _stream(port, msgs, model="qwen38-spark-engine", headers={"X-Tandem": "raw"})
        b, _, _ = _stream(raw_port, msgs, model="qwen38-spark-engine")
        ids = re.compile(r'"call_[0-9a-f]+"')            # a fresh id per call, on any server
        assert ids.sub("ID", a) == ids.sub("ID", b), (a[:200], b[:200])
        assert h_a["X-Tandem"] == "raw" and h_a["X-Tandem-Profile"] == "chat"
        text, h, _ = _stream(port, [{"role": "user", "content": "Restyle it. FAKE_CLASS=design_ui"}],
                             headers={"x-session-id": "both"})
        assert h["X-Tandem"] == "on" and h["X-Tandem-Profile"] == "chat", h
        assert "runtime-note" not in text
        row = _decisions(port, session="c:both")[0]
        assert row["class_id"] == "design_ui" and row["injected"], row
        srv.send_signal(signal.SIGTERM)
        assert srv.wait(timeout=30) == 0
        out = srv.stdout.read()
        assert "[conductor] on: " in out and "[conductor] on" in out, out[-2000:]
    finally:
        for p in (srv, plain, arb):
            if p.poll() is None:
                p.kill()


def test_one_journal_for_the_router_and_the_monitor():
    """SRV-115/116 over HTTP: the router and the session monitor write one journal. Under the
    coding profile a TandemLLM turn gets the router's note and the monitor's hidden continue;
    the counters rise between two /metrics scrapes; the decision feed carries the monitor's
    event on the router's row; /v1/conductor/events lists it; the session export holds both
    hidden texts ("2 interventions in 1 turn"); stats has the monitor's block; the ledger row
    says acted, with the hidden tokens; everything validates against the contract."""
    port, arb_port = _port(), _port()
    cdir = tempfile.mkdtemp(prefix="qse-conductor-")
    arb = _decider(arb_port)
    srv, ledger = _server(port, cdir, extra=("--conductor", "on"))
    try:
        _wait(arb_port, arb, "/readyz")
        _wait(port, srv)
        st, _, roles = _call(port, "GET", "/v1/conductor/roles")
        doc = roles["doc"]
        doc["decider"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        doc["classifier"]["mode"] = "off"
        _call(port, "PUT", "/v1/conductor/roles", {"doc": doc, "note": "the fake"})

        def scrape(series):
            raw = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read()
            for ln in raw.decode().splitlines():
                if ln.startswith(series + " "):
                    return float(ln.rsplit(" ", 1)[1])
            return 0.0
        e1 = 'qse_conductor_actions_total{task="E1",action="CONTINUE_HIDDEN"}'
        note = 'qse_conductor_actions_total{task="route",action="PROMPT_TAIL_NOTE"}'
        before = (scrape(e1), scrape(note))
        text, h, _ = _stream(port, [{"role": "user", "content": "Fix the lexer and use the tool. "
                                                                "FAKE_CLASS=code_edit "
                                                                "FAKE_EMPTY_ANSWER"}],
                             headers={"x-session-id": "obs", "X-Tandem-Profile": "coding"})
        assert h["X-Tandem"] == "on" and h["X-Tandem-Profile"] == "coding", h
        assert "runtime-note" not in text and "You ended your turn" not in text, text
        time.sleep(1.0)
        assert (scrape(e1), scrape(note)) == (before[0] + 1, before[1] + 1), \
            (before, scrape(e1), scrape(note))
        rows = _decisions(port, session="c:obs")
        assert len(rows) == 1 and rows[0]["injected"], rows
        assert [e["task"] for e in rows[0]["events"] if e["action"]] == ["E1"], rows[0]["events"]
        st, _, evs = _call(port, "GET", "/v1/conductor/events?session=c%3Aobs&acted=1")
        _check(evs, "events")
        assert evs["monitor"] and [e["decision"] for e in evs["events"]] == ["CONTINUE_HIDDEN"]
        assert "You ended your turn" in evs["events"][0]["text"]
        # VIS-44: the row drawer asks for one request's events; the finished row carries the
        # badge's numbers; the header's indicator reads the monitor's mode
        rid = rows[0]["request_id"]
        st, _, one = _call(port, "GET", f"/v1/conductor/events?request={rid}")
        assert st == 200 and one["events"] and {e["request_id"] for e in one["events"]} == {rid}
        st, _, none = _call(port, "GET", "/v1/conductor/events?request=chatcmpl-nope")
        assert none["events"] == [], none
        st, _, reqs = _call(port, "GET", "/v1/dashboard/requests?limit=5")
        row = next(r for r in reqs["requests"] if r["request_id"] == rid)
        assert row["conductor"]["acts"] == 1 and row["conductor"]["acted"], row["conductor"]
        st, _, lv = _call(port, "GET", "/v1/dashboard/live?follow=0")
        assert lv["engine"]["conductor"] == {"mode": "on"}, lv["engine"]
        st, _, sm = _call(port, "GET", "/v1/dashboard/summary?tz=Europe/Berlin")
        assert sm["live"]["conductor"] == {"mode": "on"}, sm["live"]
        st, _, ex = _call(port, "GET", "/v1/conductor/sessions/c%3Aobs/export")
        _check(ex, "export")
        assert ex["summary"] == "2 interventions in 1 turn", ex["summary"]
        assert [i["source"] for i in ex["injections"]] == ["router", "monitor"], ex
        st, _, md = _call(port, "GET", "/v1/conductor/sessions/c%3Aobs/export?format=md",
                          raw=True)
        md = md.decode()
        assert "2 interventions in 1 turn" in md and "You ended your turn" in md, md[:600]
        assert "Reasoning effort is set to xhigh" in md, md[:1500]
        st, _, tl = _call(port, "GET", "/v1/conductor/sessions/c%3Aobs")
        _check(tl, "session")
        assert [e["task"] for e in tl["events"] if e["action"]] == ["E1"]
        st, _, ss = _call(port, "GET", "/v1/conductor/sessions")
        _check(ss, "sessions")
        one = next(x for x in ss["sessions"] if x["key"] == "c:obs")
        assert (one["acts"], one["interventions"]) == (1, 2), one
        st, _, stats = _call(port, "GET", "/v1/conductor/stats?hours=24")
        _check(stats, "stats")
        assert stats["monitor"]["acts"] >= 1 and stats["monitor"]["last_actions"][0]["task"] == \
            "E1", stats["monitor"]
        st, _, one_ev = _call(port, "POST",
                              f"/v1/conductor/decisions/{evs['events'][0]['id']}/label",
                              {"label": "right"})
        assert st == 200, one_ev
        st, _, cfg = _call(port, "GET", "/v1/conductor/config")
        _check(cfg, "config")
        assert cfg["journal"]["path"] == os.path.realpath(ledger), cfg["journal"]
        assert cfg["server"]["monitor_mode"] == "on", cfg["server"]
        srv.send_signal(signal.SIGTERM)
        assert srv.wait(timeout=30) == 0
        conn = sqlite3.connect(ledger)
        got = conn.execute("SELECT conductor_acts, conductor_hidden_tokens, conductor_acted "
                           "FROM requests WHERE model = 'TandemLLM'").fetchall()
        kinds = conn.execute("SELECT kind, task, label FROM conductor_events "
                             "WHERE action IS NOT NULL ORDER BY ts_ms").fetchall()
        conn.close()
        assert len(got) == 1 and got[0][0] == 1 and got[0][1] > 0 and got[0][2] == 1, got
        assert kinds == [("route", "route", None), ("monitor", "E1", "right")], kinds
    finally:
        for p in (srv, arb):
            if p.poll() is None:
                p.kill()


def test_the_residency_planner_over_http():
    """OPS-41/42 over HTTP: the plan endpoint answers with the limits, what fits now, the
    engine's split and the journal, against the contract; a new version of the `residency`
    document (admin token, a note) changes the plan at once; a refusal the ops script journaled
    is the plan's `last_refusal`; the System view's `memory.split` validates too."""
    port = _port()
    cdir = tempfile.mkdtemp(prefix="qse-conductor-")
    srv, _ = _server(port, cdir)
    try:
        _wait(port, srv)
        st, _, got = _call(port, "GET", "/v1/conductor/residency/plan")
        assert st == 200, got
        _check(got, "residency-plan")
        assert got["version"] == 0 and got["plan"]["need_gib"] == 21.0, got["plan"]
        # SRV-534: the narrator and the reader are off in the roles document: no claim in the plan
        assert [r["role"] for r in got["plan"]["rows"]] == ["decider"], got["plan"]
        assert got["last_refusal"] is None and got["engine"] is not None
        st, _, doc = _call(port, "GET", "/v1/conductor/residency")
        _check(doc, "document")
        new = dict(doc["doc"], floor_gib=20.0)
        st, _, w = _call(port, "PUT", "/v1/conductor/residency", {"doc": new, "note": "floor 20"},
                         token=None, headers=PROXY)
        assert st == 401, "a write needs the admin token"
        st, _, w = _call(port, "PUT", "/v1/conductor/residency", {"doc": new, "note": "floor 20"})
        assert st == 200 and w["version"] == 1, w
        bad = dict(new, order=["decider", "decider"])
        st, _, err = _call(port, "PUT", "/v1/conductor/residency", {"doc": bad, "note": "x"})
        assert st == 422 and "listed twice" in json.dumps(err), err
        from server.conductor import residency as res_mod
        assert res_mod.main(["--root", cdir, "check", "reader", "--mem-available-gib", "18",
                             "--source", "ops"]) == 3
        st, _, got = _call(port, "GET", "/v1/conductor/residency/plan")
        _check(got, "residency-plan")
        assert got["plan"]["need_gib"] == 26.0 and got["version"] == 1, got["plan"]
        assert got["last_refusal"]["reason"] == "refused_memory", got["last_refusal"]
        st, _, cfg = _call(port, "GET", "/v1/conductor/config")
        _check(cfg, "config")
        assert cfg["store"]["documents"]["residency"]["version"] == 1
        st, _, sysinfo = _call(port, "GET", "/v1/dashboard/system")
        assert contract_check.check(sysinfo, "system") == [], contract_check.check(sysinfo,
                                                                                   "system")
        assert "split" in sysinfo["memory"]
    finally:
        srv.send_signal(signal.SIGTERM)
        srv.wait(timeout=30)


def test_the_bench_tools_reach_the_raw_path():
    """One model: a bench tool reaches the raw engine through a conductor server by header only.
    row3 `--no-server` refuses a conductor server (the atlas runner sends no headers);
    session_replay and client_smoke `--x-tandem raw` send the header and get the plain server's
    answer, and the response says raw."""
    from tools import row3, session_replay
    port, raw_port = _port(), _port()
    srv, _ = _server(port, tempfile.mkdtemp(prefix="qse-conductor-"))
    plain, _ = _server(raw_port)
    try:
        _wait(port, srv)
        _wait(raw_port, plain)
        try:
            row3.refuse_conductor(port)
            raise AssertionError("row3 --no-server must refuse a TandemLLM server")
        except SystemExit as exc:
            assert "REFUSING" in str(exc) and "X-Tandem: raw" in str(exc), exc
        row3.refuse_conductor(raw_port)                 # the raw engine: no objection
        body = {"model": "replay", "stream": False, "temperature": 0, "max_tokens": 64,
                "messages": [{"role": "user", "content": "Name three rivers in Europe."}]}
        session_replay.HEADERS["X-Tandem"] = "raw"
        try:
            a = session_replay._post(f"http://127.0.0.1:{port}/v1", body)
        finally:
            session_replay.HEADERS.pop("X-Tandem", None)
        b = session_replay._post(f"http://127.0.0.1:{raw_port}/v1", body)
        assert a["choices"][0]["message"] == b["choices"][0]["message"], (a, b)
        try:
            import openai  # noqa: F401
        except ImportError:
            print("  (openai not installed here: the client_smoke half is skipped)")
            return
        from tools.client_smoke import Smoke
        s = Smoke(f"http://127.0.0.1:{port}/v1", "", None, "raw")
        assert s.model == "TandemLLM", "the list names the one model"
        r = s.client.chat.completions.with_raw_response.create(
            model=s.model, max_tokens=4, messages=[{"role": "user", "content": "Say ok."}])
        assert r.headers.get("x-tandem") == "raw", dict(r.headers)
    finally:
        for p in (srv, plain):
            if p.poll() is None:
                p.kill()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
