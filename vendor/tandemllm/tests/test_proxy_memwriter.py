"""The proxy's background memory writer (M3, 2026-10-05, proxy/memwriter.py), against fake
backends, on a CPU, in seconds:

  1  the switches: auto is on for a local backend and off for a paid cloud model, a zero price
     is free, the settings document lowers the backend's mode, the env variables and the schema
                                                        test_switches_and_env
  2  the record and the trim: the system prompt kept, the oldest messages dropped, never a tool
     result first; the body: tool_choice none with the session's tools, thinking off, the prefill
     where the backend can continue (vLLM) and not where it cannot (OpenAI)
                                                        test_trim_and_body
  3  a local vLLM session goes idle: one writer request with the session's messages as sent and
     its answer (the prefix the backend cached), the note written as qwen-writer, a ledger row of
     `harness:writer`, the run in the writer runs with via proxy; not due afterwards
                                                        test_idle_session_is_written_on_vllm
  4  shadow: the request runs, nothing is stored, the would-be block in the user's shadow log
     (0600, read by nothing), the journal without the title
                                                        test_shadow_logs_what_it_would_write
  5  a paid cloud model: auto asks nothing; on without a price is refused (no_price); a spend
     cap below the estimate is refused (cap_cost); within it the run is billed and counted; the
     runs cap a day (cap_runs); the day's spend read back from the ledger
                                                        test_paid_cloud_caps
  6  the writer's NONE: NONE is none; a `bash` call is cut at its function's name and is none
                                                        test_none_and_a_cut_call
  7  a compaction in the session: the turn before it is the record, the writer runs with trigger
     compaction after `due_idle_s` even when later turns came                 test_compaction_record
  8  a client request to the backend stops the writer's stream (local backend); the session
     stays due                                          test_a_client_request_stops_the_writer
  9  the Backends API shows the writer's mode per model and the day's runs
                                                        test_backends_api_shows_the_writer

Nothing here starts a thread of its own: `ProxyWriter.auto` is off and the test calls `tick()`.

Run: python tests/test_proxy_memwriter.py
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import Proxy, fake, run  # noqa: E402

from proxy import app as proxy_app  # noqa: E402
from proxy import config as config_mod  # noqa: E402
from proxy import memwriter as pmw  # noqa: E402
from proxy.adapters import make  # noqa: E402

UA = {"User-Agent": "opencode/1.18.34 ai-sdk/provider-utils/3"}
SYSTEM = {"role": "system", "content": "You are a coding agent. Working directory: /tmp/proj"}
TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}}}}]


def _env(kind="vllm", port=None, bid="v", extra=None):
    up = bid.upper()
    env = {"TANDEM_BACKENDS": bid, f"TANDEM_BACKEND_{up}_KIND": kind,
           f"TANDEM_BACKEND_{up}_URL": f"http://127.0.0.1:{port}",
           f"TANDEM_BACKEND_{up}_HARNESS": "on"}
    if kind in ("openai", "deepseek"):
        env[f"TANDEM_BACKEND_{up}_KEY"] = "sk-fixture-ABCDEFGH1234"
    env.update(extra or {})
    return env


def _setup(px, mode="on", memory="on", prefill=""):
    """The coding profile with memory, the writer's settings (no gate, due at once), a proxy
    user, and the writer without its thread. Returns (conductor, headers)."""
    px.sync()
    cond = proxy_app.STATE["conductor"]
    v, prof = cond.store.get("profiles", "coding")
    cond.store.put("profiles", "coding", dict(prof, memory=memory), note="m3", author="test",
                   base=v)
    v, st = cond.store.get("settings")
    st.setdefault("memory", {})["notes"] = {"writer": {
        "mode": mode, "gate": False, "idle_s": 10.0, "due_idle_s": 0.0, "min_new_tokens": 1,
        "max_tokens": 200, "after_compaction": "idle", "prefill": prefill}}
    cond.store.put("settings", None, st, note="m3", author="test", base=v)
    px.svc.harness.writer.auto = False
    _, _, body = px.request("POST", "/v1/config/users", {"label": "laptop"})
    key = json.loads(body)["key"]
    return cond, dict(UA, Authorization=f"Bearer {key}", **{"x-session-id": "ses1"})


def _turn(px, hdr, text, model="m1", tools=None, history=()):
    body = {"model": model, "stream": True,
            "messages": [SYSTEM, *history, {"role": "user", "content": text}]}
    if tools:
        body["tools"] = tools
    n = len(proxy_app.STATE["svc"].harness.writer.sessions)
    seen = {k: s.t for k, s in proxy_app.STATE["svc"].harness.writer.sessions.items()}
    st, _, out = px.post("/v1/chat/completions", body, headers=hdr)
    assert st == 200, out
    w = proxy_app.STATE["svc"].harness.writer
    _wait(lambda: len(w.sessions) > n or any(seen.get(k) != s.t for k, s in w.sessions.items()),
          2.0)
    return out


def _wait(pred, timeout=5.0):
    """The turn's bookkeeping runs after the client has its answer (Turn.finish)."""
    t_end = time.time() + timeout
    while not pred() and time.time() < t_end:
        time.sleep(0.01)
    return pred()


def _idle(px, seconds=1000.0):
    for s in px.svc.harness.writer.sessions.values():
        s.t = time.monotonic() - seconds


def _writer_bodies(srv) -> list:
    out = []
    for b in srv.fake.bodies:
        msgs = [m for m in (b.get("messages") or []) if m.get("role") == "user"]
        if msgs and str(msgs[-1].get("content") or "").startswith("Memory pass."):
            out.append(b)
    return out


# --------------------------------------------------------------------------------- 1, 2
def test_switches_and_env():
    local = {"id": "v", "cloud": False, "harness": {"mode": "on"}}
    cloud = {"id": "o", "cloud": True, "harness": {"mode": "on"}}
    assert pmw.resolve(local, None, "on") == ("on", "auto")
    assert pmw.resolve(cloud, None, "on") == ("off", "paid_cloud")
    assert pmw.resolve(cloud, {"input": 1.0, "output": 2.0}, "on") == ("off", "paid_cloud")
    assert pmw.resolve(cloud, {"input": 0, "output": 0}, "on") == ("on", "auto")
    assert pmw.resolve(dict(cloud, harness={"writer": "on"}), None, "on") == ("on", "backend")
    assert pmw.resolve(dict(cloud, harness={"writer": "on"}), None, "shadow") == \
        ("shadow", "settings")
    assert pmw.resolve(local, None, "off") == ("off", "settings")
    assert pmw.backend_cfg(local) == pmw.DEFAULTS
    assert pmw.backend_cfg({"harness": {"writer_usd_day": None}})["writer_usd_day"] is None
    env = _env("openai", 1, "o", {"TANDEM_BACKEND_O_WRITER": "shadow",
                                  "TANDEM_BACKEND_O_WRITER_USD_DAY": "0.5",
                                  "TANDEM_BACKEND_O_WRITER_RUNS_DAY": "3",
                                  "TANDEM_BACKEND_O_WRITER_PROMPT_TOKENS": "8000"})
    got = config_mod.parse_env(env)["backends"]["o"]
    assert got["harness.writer"] == "shadow" and got["harness.writer_usd_day"] == 0.5 and \
        got["harness.writer_runs_day"] == 3 and got["harness.writer_prompt_tokens"] == 8000, got
    bad = dict(env, TANDEM_BACKEND_O_WRITER="always")
    try:
        config_mod.parse_env(bad)
        raise AssertionError("an unknown writer mode passed")
    except config_mod.ConfigError as exc:
        assert "WRITER" in str(exc) or "writer" in str(exc), exc


def test_trim_and_body():
    msgs = [SYSTEM, {"role": "user", "content": "a " * 400},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {
                "name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "out " * 400},
            {"role": "user", "content": "b " * 50}, {"role": "assistant", "content": "ok"}]
    same, n = pmw.trim(msgs, 100000)
    assert same == msgs and n == 0
    cut, n = pmw.trim(msgs, 300)
    assert cut[0] == SYSTEM and cut[1]["role"] == "user" and cut[1]["content"].startswith("b "), \
        cut
    assert n == 3
    cfg = {"max_tokens": 600, "ban_tool_call": True, "prefill": ""}
    v = make({"id": "v", "kind": "vllm", "url": "http://x/v1", "harness": {}})
    body, how = pmw.ProxyWriter.body(v, "m1", msgs, TOOLS, "Memory pass. x", cfg)
    assert body["tool_choice"] == "none" and body["tools"] == TOOLS and body["stream"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}, body
    assert body["messages"][:-1] == msgs and body["messages"][-1]["role"] == "user"
    assert "tool_choice=none" in how and "enable_thinking=false" in how
    body, _ = pmw.ProxyWriter.body(v, "m1", msgs, None, "Memory pass. x", cfg)
    assert "tools" not in body and "tool_choice" not in body
    body, how = pmw.ProxyWriter.body(v, "m1", msgs, TOOLS, "Memory pass. x",
                                     dict(cfg, prefill="Notes:\n", ban_tool_call=False))
    assert "tool_choice" not in body and body["continue_final_message"] is True
    assert body["messages"][-1] == {"role": "assistant", "content": "Notes:\n"} and "prefill" in how
    o = make({"id": "o", "kind": "openai", "url": "http://x/v1", "harness": {}})
    body, how = pmw.ProxyWriter.body(o, "gpt-x", msgs, None, "Memory pass. x",
                                     dict(cfg, prefill="Notes:\n"))
    assert "continue_final_message" not in body and "prefill=unsupported" in how


# --------------------------------------------------------------------------------- 3, 4
def test_idle_session_is_written_on_vllm():
    srv, port = fake(models=["m1"])
    with Proxy(env=_env("vllm", port), conductor=True) as px:
        cond, hdr = _setup(px)
        hx = px.svc.harness
        _turn(px, hdr, "Fix the build of the checker, it fails on node 18.", tools=TOOLS)
        assert hx.writer.tick() == "not_due"                 # not idle yet
        sent = srv.fake.bodies[-1]
        _idle(px)
        assert hx.writer.tick() == "wrote", hx.writer.stats
        wb = _writer_bodies(srv)
        assert len(wb) == 1, len(wb)
        b = wb[0]
        # the session's messages as they were sent (notes in place), its answer, the writer's turn
        assert b["messages"][:len(sent["messages"])] == sent["messages"]
        assert b["messages"][len(sent["messages"])]["role"] == "assistant"
        assert b["messages"][-1]["content"].startswith("Memory pass.")
        assert b["tool_choice"] == "none" and b["tools"] == sent["tools"]
        assert b["chat_template_kwargs"] == {"enable_thinking": False}
        ns = cond.notes.notes("laptop")
        assert [(n.title, n.writer, n.scope, n.status) for n in ns] == [
            ("the build needs node 20", "qwen-writer", "project", "provisional")], ns
        run = cond.memory.writer.runs[0]
        assert run["via"] == "proxy" and run["backend"] == "v" and run["trigger"] == "idle", run
        assert run["ops"][0]["op"] == "add" and run["tokens"] and run["answer"] == "blocks", run
        rows = [r for r in px.rows() if r["client_id"] == "harness:writer"]
        assert len(rows) == 1 and rows[0]["backend"] == "v" and rows[0]["status"] == 200, rows
        assert hx.writer.tick() == "not_due"                 # written at this record
        assert any("proxy writer v/m1" in m and "1 op(s)" in m for m in px.logs), px.logs
    srv.shutdown()


def test_shadow_logs_what_it_would_write():
    srv, port = fake(models=["m1"])
    with Proxy(env=_env("vllm", port), conductor=True) as px:
        cond, hdr = _setup(px, mode="shadow")
        _turn(px, hdr, "Fix the build of the checker, it fails on node 18.")
        _idle(px)
        assert px.svc.harness.writer.tick() == "wrote"
        assert cond.notes.notes("laptop") == []
        run = cond.memory.writer.runs[0]
        assert run["mode"] == "shadow" and run["ops"] == [
            {"op": "add", "kind": "gotcha", "scope": "project", "stored": False}], run
        assert run["shadow_log"] == 1, run
        d = os.path.join(cond.notes.user_dir("laptop"), "shadow")
        files = os.listdir(d)
        assert len(files) == 1 and files[0].startswith("writer-"), files
        path = os.path.join(d, files[0])
        text = open(path).read()
        assert "title: the build needs node 20" in text and "idle backend v model m1" in text, text
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        cond.journal.flush()
        rows = [r for r in cond.journal.rows(kind="memory_notes", limit=20)
                if r.get("task") == "writer"]
        assert rows and "node 20" not in json.dumps(rows), rows
    srv.shutdown()


# --------------------------------------------------------------------------------- 5
def test_paid_cloud_caps():
    srv, port = fake(models=["m1"])
    env = _env("openai", port, "o")
    with Proxy(env=env, conductor=True) as px:
        cond, hdr = _setup(px)
        hx = px.svc.harness
        _turn(px, hdr, "Fix the build of the checker, it fails on node 18.")
        _idle(px)
        assert hx.writer.tick() == "off"                     # auto: a paid cloud model
        assert _writer_bodies(srv) == [] and hx.writer.tick() == "not_due"
    srv.shutdown()

    srv, port = fake(models=["m1"])
    env = _env("openai", port, "o", {"TANDEM_BACKEND_O_WRITER": "on"})
    with Proxy(env=env, conductor=True) as px:
        cond, hdr = _setup(px)
        hx = px.svc.harness
        _turn(px, hdr, "Fix the build of the checker, it fails on node 18.")
        _idle(px)
        assert hx.writer.tick() == "no_price"                # its cap could not be checked
        assert _writer_bodies(srv) == []
        assert cond.memory.writer.runs[0]["skipped"] == "no_price"
    srv.shutdown()

    srv, port = fake(models=["m1"])
    price = {"TANDEM_PRICE_1": "o/m1=2.0,0.2,8.0"}
    env = _env("openai", port, "o", dict(price, TANDEM_BACKEND_O_WRITER="on",
                                         TANDEM_BACKEND_O_WRITER_USD_DAY="0.000001"))
    with Proxy(env=env, conductor=True) as px:
        cond, hdr = _setup(px)
        hx = px.svc.harness
        _turn(px, hdr, "Fix the build of the checker, it fails on node 18.")
        _idle(px)
        assert hx.writer.tick() == "cap_cost"
        run = cond.memory.writer.runs[0]
        assert run["skipped"] == "cap_cost" and run["cost_est_micros"] > 1, run
        assert _writer_bodies(srv) == []
    srv.shutdown()

    srv, port = fake(models=["m1"])
    env = _env("openai", port, "o", dict(price, TANDEM_BACKEND_O_WRITER="on",
                                         TANDEM_BACKEND_O_WRITER_RUNS_DAY="1"))
    with Proxy(env=env, conductor=True) as px:
        cond, hdr = _setup(px)
        hx = px.svc.harness
        _turn(px, hdr, "Fix the build of the checker, it fails on node 18.")
        _idle(px)
        assert hx.writer.tick() == "wrote", cond.memory.writer.runs[0]
        run = cond.memory.writer.runs[0]
        assert run["cost_micros"] and run["cost_micros"] > 0, run
        assert hx.writer.today("o") == {"runs": 1, "cost_micros": run["cost_micros"]}
        assert [n.title for n in cond.notes.notes("laptop")] == ["the build needs node 20"]
        # a second session the same day: past the runs cap
        hdr2 = dict(hdr, **{"x-session-id": "ses2"})
        _turn(px, hdr2, "Now the docs: the README is out of date.")
        _idle(px)
        assert hx.writer.tick() == "cap_runs"
        assert len(_writer_bodies(srv)) == 1
        # the day's spend comes back from the ledger after a restart
        hx.writer.spend.clear()
        px.rows()
        assert hx.writer.today("o") == {"runs": 1, "cost_micros": run["cost_micros"]}
    srv.shutdown()


# --------------------------------------------------------------------------------- 6, 7, 8
def test_none_and_a_cut_call():
    for word, answer, cut in (("FAKE_WRITER_NONE", "none", None),
                              ("FAKE_WRITER_BASH", "tool_call", "bash")):
        srv, port = fake(models=["m1"])
        with Proxy(env=_env("vllm", port), conductor=True) as px:
            cond, hdr = _setup(px)
            _turn(px, hdr, f"Look at the logs. {word}", tools=TOOLS)
            _idle(px)
            assert px.svc.harness.writer.tick() == "wrote"
            run = cond.memory.writer.runs[0]
            assert run["answer"] == answer and run["none"] is True and run.get("cut") == cut, run
            assert run["ops"] == [] and cond.notes.notes("laptop") == []
            if cut:
                assert run["finish"] == "tool_call", run
                assert any("NONE (tool_call, cut at bash)" in m for m in px.logs), px.logs
        srv.shutdown()


def test_compaction_record():
    srv, port = fake(models=["m1"])
    with Proxy(env=_env("vllm", port), conductor=True) as px:
        cond, hdr = _setup(px)
        hx = px.svc.harness
        _turn(px, hdr, "Fix the build of the checker, it fails on node 18.")
        before = list(hx.writer.sessions["px:c:ses1"].msgs)
        comp = {"model": "m1", "stream": False, "messages": [
            {"role": "system", "content": "You are a context summarization agent. You are given "
                                          "a conversation between a user and an agent."},
            {"role": "user", "content": "user: fix the build\nassistant: done " * 200}]}
        st, _, _ = px.post("/v1/chat/completions", comp, headers=hdr)
        assert st == 200
        assert _wait(lambda: cond.memory.writer.state("px:c:ses1")["due"] == "compaction")
        # the session goes on after the compaction before the writer's look
        _turn(px, hdr, "Go on with the tests.")
        assert hx.writer.tick() == "wrote"           # due_idle_s 0: no idle wait for a mark
        run = cond.memory.writer.runs[0]
        assert run["trigger"] == "compaction", run
        b = _writer_bodies(srv)[-1]
        assert b["messages"][:-1] == before, (len(b["messages"]), len(before))
        assert hx.writer.sessions["px:c:ses1"].pending is None
    srv.shutdown()


def test_a_client_request_stops_the_writer():
    srv, port = fake(models=["m1"])
    with Proxy(env=_env("vllm", port), conductor=True) as px:
        cond, hdr = _setup(px)
        hx = px.svc.harness
        _turn(px, hdr, "Fix the build. FAKE_WRITER_SLOW")
        _idle(px)
        got = {}
        t = threading.Thread(target=lambda: got.update(r=hx.writer.tick()))
        t.start()
        deadline = time.time() + 5
        while not _writer_bodies(srv) and time.time() < deadline:
            time.sleep(0.02)
        time.sleep(0.3)
        # another client's request to the same backend while the writer streams
        st, _, _ = px.post("/v1/chat/completions", {"model": "m1", "messages": [
            {"role": "user", "content": "hello FAKE_SLOW_TTFT 0.5"}]}, headers=UA)
        assert st == 200
        t.join(10)
        assert got.get("r") == "yielded", got
        run = cond.memory.writer.runs[0]
        assert run["skipped"] == "yielded:request" and run.get("yielded") is True, run
        assert cond.notes.notes("laptop") == []
        assert cond.memory.writer.state("px:c:ses1")["done_at"] == 0      # still due
    srv.shutdown()


# --------------------------------------------------------------------------------- 9
def test_backends_api_shows_the_writer():
    srv, port = fake(models=["m1"])
    env = _env("openai", port, "o", {"TANDEM_BACKEND_O_WRITER_USD_DAY": "0.1"})
    with Proxy(env=env, conductor=True) as px:
        _setup(px)
        st, _, body = px.get("/v1/config")
        assert st == 200, body
        b = [x for x in json.loads(body)["backends"] if x["id"] == "o"][0]
        assert b["writer"]["config"]["writer_usd_day"] == 0.1, b["writer"]
        assert b["writer"]["today"] == {"runs": 0, "cost_micros": 0}
        m = [x for x in b["model_rows"] if x["model"] == "m1"]
        assert m and m[0]["writer"] == {"mode": "off", "why": "paid_cloud"}, b["model_rows"]
        assert "daily cap" in b["degradations"]["memory"], b["degradations"]["memory"]
        # 0.3.36: the settings document's mode and the shadow log's place, for the screen
        assert b["writer"]["settings_mode"] in ("off", "shadow", "on"), b["writer"]
        sl = b["writer"]["shadow_log"]
        assert sl and sl.endswith("/<user>/shadow/writer-<yyyy-mm-dd>.md"), sl
        # the switches are set through the backend PATCH (harness.writer*) and merge
        st, _, body = px.request("PATCH", "/v1/config/backends/o",
                                 {"harness": {"writer": "shadow", "writer_runs_day": 5}})
        assert st == 200, body
        b = json.loads(body)
        assert b["harness"]["writer"] == "shadow" and b["harness"]["mode"], b["harness"]
        assert b["writer"]["config"] == {"writer": "shadow", "writer_usd_day": 0.1,
                                         "writer_runs_day": 5,
                                         "writer_prompt_tokens": 32000}, b["writer"]
        assert [x for x in b["model_rows"] if x["model"] == "m1"][0]["writer"]["mode"] in (
            "shadow", "off"), b["model_rows"]
        # the env variable locks its field: a change is a 422 naming it
        st, _, body = px.request("PATCH", "/v1/config/backends/o",
                                 {"harness": {"writer_usd_day": 0.5}})
        assert st == 422 and "TANDEM_BACKEND_O_WRITER_USD_DAY" in body.decode(), (st, body)
        st, _, body = px.request("PATCH", "/v1/config/backends/o",
                                 {"harness": {"writer_runs_day": -1}})
        assert st == 422, (st, body)
    srv.shutdown()


if __name__ == "__main__":
    run(globals())
