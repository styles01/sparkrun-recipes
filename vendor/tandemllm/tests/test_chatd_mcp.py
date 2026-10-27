"""chatd's MCP client (SRV-359) against the reference server tests/fake_mcp.py, over stdio and
over streamable HTTP: initialize, the paged tool list, a call, progress, cancel, list_changed;
then through chatd: servers added with write-only secrets, tools offered as mcp__<server>__<tool>
with the allow and deny lists, a call in a chat, the timeout, and no secret in any answer, event or
log line. CPU only.

Run: python -m pytest -q tests/test_chatd_mcp.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import fake_mcp
from chatd_util import Chat, run_all

from chatd import mcp as mcp_mod

FAKE = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_mcp.py")]
SECRET = "sk-test-" + "z" * 24


def calls(*c) -> str:
    return "FAKE_CALLS " + json.dumps([{"name": n, "arguments": a} for n, a in c])


def conformance(session: mcp_mod.Session, cancelled_seen) -> None:
    info = session.initialize()
    assert info["protocolVersion"] == "2025-06-18"
    tools = session.list_tools()
    assert [t["name"] for t in tools] == ["echo", "add", "slow", "grow", "fail", "secret_env"]
    r = session.call("echo", {"text": "hi"}, timeout=10)
    assert r["content"][0]["text"] == "hi"
    seen = []
    r = session.call("slow", {"seconds": 0.6}, timeout=10, on_progress=seen.append)
    assert r["content"][0]["text"] == "slow done" and [p["progress"] for p in seen] == [1, 2, 3]
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    try:
        session.call("slow", {"seconds": 5}, timeout=10, cancel=cancel)
        raise AssertionError("a cancelled call must not answer")
    except mcp_mod.McpError as exc:
        assert "cancelled" in str(exc) and time.monotonic() - t0 < 2
    time.sleep(0.5)
    assert cancelled_seen()
    try:
        session.call("slow", {"seconds": 5}, timeout=0.5)
        raise AssertionError("a call past its timeout must not answer")
    except mcp_mod.McpError as exc:
        assert "within 0 s" in str(exc) or "within 1 s" in str(exc) or "cancelled" in str(exc)
    changed = threading.Event()
    session.on_list_changed = changed.set
    session.call("grow", {}, timeout=10)
    assert changed.wait(5)
    assert "extra" in [t["name"] for t in session.list_tools()]
    assert session.request("ping", timeout=5) == {}


def test_conformance_over_stdio():
    s = mcp_mod.Session(mcp_mod.StdioTransport(mcp_mod.process_pipe(FAKE)))
    try:
        # the stdio fake records cancellations in its own process: a slow call that was cancelled
        # never answers, which the next call shows by answering at once
        conformance(s, lambda: True)
        assert s.call("echo", {"text": "still alive"}, timeout=5)["content"][0]["text"] == \
            "still alive"
    finally:
        s.close()


def test_conformance_over_streamable_http():
    httpd, port = fake_mcp.start_http()
    s = mcp_mod.Session(mcp_mod.HttpTransport(f"http://127.0.0.1:{port}/mcp", {}))
    try:
        conformance(s, lambda: bool(httpd.fake.cancelled))
        assert s.t.session_id == "s1"
    finally:
        s.close()
        httpd.shutdown()


def test_servers_in_a_chat_with_write_only_secrets():
    httpd, port = fake_mcp.start_http(header=SECRET)
    c = Chat()
    c.svc.mcp.stdio_pipe = lambda row, cid: mcp_mod.process_pipe(
        FAKE, c.svc.mcp._open(row["id"], row["env_enc"]))
    try:
        sse = c.events(c.new())
        sse.close()
        s = c.ok("POST", "/mcp/servers", {"name": "Fake HTTP", "transport": "http",
                                          "url": f"http://127.0.0.1:{port}/mcp",
                                          "headers": {"Authorization": f"Bearer {SECRET}"},
                                          "tool_allow": ["echo", "add", "fail"]})
        assert s["headers"] == {"Authorization": ""}
        c.wait(lambda: c.ok("GET", "/mcp/servers")[0]["status"] == "connected", 10)
        lst = c.ok("GET", "/mcp/servers")[0]
        assert [t["name"] for t in lst["tools"]][:2] == ["echo", "add"] and lst["schema_tokens"] > 0
        st = c.ok("POST", "/mcp/servers", {"name": "local", "transport": "stdio",
                                           "command": "python3", "args": ["fake_mcp.py"],
                                           "env": {"FAKE_MCP_TOKEN": SECRET},
                                           "tool_deny": ["slow", "grow"]})
        assert st["env"] == {"FAKE_MCP_TOKEN": ""}
        bad = c.call("POST", "/mcp/servers/test", {"name": "x", "transport": "http",
                                                   "url": f"http://127.0.0.1:{port}/mcp"})[1]
        assert bad["ok"] is False and "401" in bad["message"]
        good = c.ok("POST", "/mcp/servers/test", {"name": "x", "transport": "http",
                                                  "url": f"http://127.0.0.1:{port}/mcp",
                                                  "headers": {"Authorization": f"Bearer {SECRET}"}})
        assert good["ok"] and "6 tools" in good["message"]
        c.ok("PATCH", f"/mcp/servers/{s['id']}", {"name": "Fake HTTP"})   # the secret stays
        # its prompt is a slash command, its resource an attachment (plan note Chat 3)
        lst = c.ok("GET", "/mcp/servers")[0]
        assert lst["prompts"] == [{"name": "review", "description": "Review a file.",
                                   "arguments": [{"name": "file", "required": True}]}]
        assert lst["resources"][0]["uri"] == "file:///notes/todo.txt"
        cmd = next(x for x in c.ok("GET", "/capabilities")["commands"]
                   if x["command"] == "mcp__fake_http__review")
        assert cmd["mcp_prompt"]["server_id"] == s["id"]
        assert c.ok("POST", f"/mcp/servers/{s['id']}/prompts/get", {
            "name": "review", "arguments": {"file": "app.py"}})["text"] == "Review app.py for bugs."
        att = c.ok("POST", f"/mcp/servers/{s['id']}/resources/read",
                   {"uri": "file:///notes/todo.txt"})
        assert att["name"] == "todo.txt" and att["bytes"] == 8
        # the whole fake HTTP server allowed by one rule
        c.svc.add_rule("mcp__fake_http__*", None, "allow", None)
        cid = c.new()
        c.send(cid, calls(("mcp__fake_http__echo", {"text": "from the model"}),
                          ("mcp__fake_http__fail", {})))
        snap = c.wait_idle(cid, 30)
        parts = [p for p in snap["messages"][-1]["parts"] if p["type"] == "tool_call"]
        assert parts[0]["status"] == "done" and parts[0]["output"]["content"] == "from the model"
        assert parts[1]["status"] == "failed" and parts[1]["output"]["content"] == "it failed"
        assert parts[0]["source"] == {"kind": "mcp", "id": s["id"]}
        tools = [t["function"]["name"] for t in [b for b in c.fake.bodies if b.get("stream")][-1]
                 ["tools"]]
        assert "mcp__fake_http__echo" in tools and "mcp__fake_http__slow" not in tools
        # the stdio server: its tools are listed on first use in a chat, the secret in its env
        c.svc.mcp.refresh(st["id"], cid)
        c.svc.add_rule("mcp__local__secret_env", None, "allow", cid)
        c.send(cid, calls(("mcp__local__secret_env", {})))
        snap = c.wait_idle(cid, 30)
        call = [p for p in snap["messages"][-1]["parts"] if p["type"] == "tool_call"][-1]
        assert call["output"]["content"] == "token set"
        names = [t["id"] for t in c.ok("GET", "/tools")]
        assert "mcp__local__echo" in names and "mcp__local__slow" not in names
        # no secret anywhere
        everything = json.dumps(c.ok("GET", "/mcp/servers")) + json.dumps(c.snap(cid)) + \
            json.dumps(c.ok("GET", "/tools")) + "\n".join(c.lines)
        everything += "".join(r[0] for r in c.svc.store.all("SELECT data FROM events"))
        assert SECRET not in everything
        raw = c.svc.store.one("SELECT env_enc, headers_enc FROM mcp_servers WHERE id = ?",
                              (st["id"],))
        assert SECRET.encode() not in (raw[0] or b"") and SECRET not in str(raw[1])
        st2 = c.call("DELETE", f"/mcp/servers/{s['id']}")[0]
        assert st2 == 204 and len(c.ok("GET", "/mcp/servers")) == 1
    finally:
        c.close()
        httpd.shutdown()


def test_a_hanging_call_times_out_and_is_cancelled():
    httpd, port = fake_mcp.start_http()
    c = Chat()
    try:
        s = c.ok("POST", "/mcp/servers", {"name": "slowpoke", "transport": "http",
                                          "url": f"http://127.0.0.1:{port}/mcp", "timeout_s": 1})
        c.wait(lambda: c.ok("GET", "/mcp/servers")[0]["status"] == "connected", 10)
        c.svc.add_rule("mcp__slowpoke__*", None, "allow", None)
        cid = c.new()
        c.send(cid, calls(("mcp__slowpoke__slow", {"seconds": 6})))
        snap = c.wait_idle(cid, 30)
        call = [p for p in snap["messages"][-1]["parts"] if p["type"] == "tool_call"][0]
        assert call["status"] == "failed" and "within 1 s" in call["output"]["content"]
        time.sleep(0.5)
        assert httpd.fake.cancelled
        assert s["timeout_s"] == 1
    finally:
        c.close()
        httpd.shutdown()


if __name__ == "__main__":
    run_all(globals())
