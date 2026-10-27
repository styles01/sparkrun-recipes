"""chatd's permission modes (SRV-387, SRV-391): ask, auto (the default) and yolo per chat, a default
mode, the run's mode on the live view and the thread, and every auto-approved call in the event log.
CPU only: the model is tests/fake_backend.py (FAKE_CALLS), the sandbox the fake of test_chatd_agent.

The test matrix (each mode against bash, write_file, web_fetch to a new domain, an MCP write, a deny
rule and plan mode):

  call                          ask       auto                         yolo
  bash ls                       asks      runs, auto_approved          runs, auto_approved
  bash rm -rf /workspace        asks      asks                         runs
  write_file                    runs      runs                         runs
  web_fetch, a new domain       asks      asks                         runs
  web_fetch, a domain fetched   asks      runs                         runs
  MCP tool, not read-only       asks      asks                         runs
  MCP tool, readOnlyHint        asks      runs                         runs
  a deny rule                   denied    denied                       denied
  an explicit ask policy        asks      asks                         asks
  plan mode                     read-only read-only                    read-only

Run: python tests/test_chatd_permission_modes.py
"""

from __future__ import annotations

import json

from chatd_util import Chat, is_done, run_all
from test_chatd_agent import calls, chat_with_sandbox, parts_of

from chatd import owui as owui_mod, tools as T

BASH = T.BY_NAME["bash"]
WRITE = T.BY_NAME["write_file"]
FETCH = T.BY_NAME["web_fetch"]
RENDER = T.BY_NAME["render_page"]


def _mcp(read_only: bool | None) -> T.Tool:
    ann = {} if read_only is None else {"annotations": {"readOnlyHint": read_only}}
    return T.Tool("mcp__x__post", "Post something.", {"type": "object"}, risk="network",
                  default="ask", source={"kind": "mcp", "id": "x"},
                  side_effects=(ann.get("annotations") or {}).get("readOnlyHint") is not True)


def _d(tool, args, mode, chat=(), glob=(), settings=None, pref=None, fetched=()):
    return T.decide_mode(tool, args, list(chat), list(glob), settings or {}, pref, mode,
                         frozenset(fetched))


def test_the_matrix():
    ls = {"command": "ls -la"}
    rm = {"command": "rm -rf /workspace"}
    new = {"url": "https://example.org/a"}
    deny = [{"tool": "bash", "match": "curl", "decision": "deny"}]
    for mode, want in (
            ("ask", {"ls": "ask", "rm": "ask", "write": "allow", "new": "ask", "seen": "ask",
                     "mcp_w": "ask", "mcp_r": "ask", "render": "allow"}),
            ("auto", {"ls": "allow", "rm": "ask", "write": "allow", "new": "ask", "seen": "allow",
                      "mcp_w": "ask", "mcp_r": "allow", "render": "allow"}),
            ("yolo", {"ls": "allow", "rm": "allow", "write": "allow", "new": "allow",
                      "seen": "allow", "mcp_w": "allow", "mcp_r": "allow", "render": "allow"})):
        got = {"ls": _d(BASH, ls, mode)[0], "rm": _d(BASH, rm, mode)[0],
               "write": _d(WRITE, {"path": "a", "content": "b"}, mode)[0],
               "new": _d(FETCH, new, mode)[0],
               "seen": _d(FETCH, new, mode, fetched={"example.org"})[0],
               "mcp_w": _d(_mcp(None), {}, mode)[0], "mcp_r": _d(_mcp(True), {}, mode)[0],
               "render": _d(RENDER, {"path": "a.html"}, mode)[0]}
        assert got == want, (mode, got)
        # a deny rule wins in every mode; an explicit ask policy asks in every mode
        assert _d(BASH, {"command": "curl https://x.org"}, mode, chat=deny) == ("deny", None), mode
        assert _d(BASH, {"command": "ls && curl x"}, mode, glob=deny)[0] == "deny", mode
        assert _d(BASH, ls, mode, settings={"policies": {"bash": "never"}})[0] == "deny", mode
        assert _d(BASH, ls, mode, pref="ask")[0] == "ask", mode
        assert _d(BASH, ls, mode, settings={"policies": {"bash": "ask"}})[0] == "ask", mode
    # which mode allowed it is part of the answer (the record and the event)
    assert _d(BASH, ls, "auto") == ("allow", "auto") and _d(BASH, ls, "yolo") == ("allow", "yolo")
    assert _d(WRITE, {"path": "a", "content": "b"}, "auto") == ("allow", None)   # the default


def test_the_six_prompts_of_the_clock_run():
    """SRV-391's acceptance: the six bash prompts of 01M4STKAHX6N8S5H07WRSHZFAA run in auto."""
    six = ["grep -n numeral clock.html | head -20; chromium --headless --dump-dom file:///workspace/clock.html",
           "chromium --headless --dump-dom file:///workspace/clock.html",
           "chromium --headless --dump-dom http://localhost:8080/clock.html",
           "grep -n tilt clock.html",
           "cd /tmp && chromium --headless --screenshot=/tmp/shot.png file:///workspace/clock.html",
           "cp /tmp/shot.png /workspace/shot.png"]
    assert [_d(BASH, {"command": c}, "auto") for c in six] == [("allow", "auto")] * 6
    assert [_d(BASH, {"command": c}, "ask")[0] for c in six] == ["ask"] * 6
    for c in ("rm -rf /", "rm -rf ~", "rm -r /workspace/", "cd x && rmdir $HOME", "rm -rf /home/tandem",
              "echo $(rm -rf /tmp/x)"):
        assert _d(BASH, {"command": c}, "auto")[0] == "ask", c
    for c in ("rm -rf build", "rm /workspace/old.txt", "rm -rf /tmp/x"):
        assert _d(BASH, {"command": c}, "auto") == ("allow", "auto"), c


def test_open_webui_and_mcp_side_effects():
    assert owui_mod.side_effects('"""title: t"""\nclass Tools:\n    def a(self) -> str:\n        return "x"\n') is False
    for src in ("import requests", "from urllib.request import urlopen", "import socket",
                "import subprocess", "import os\nos.system('curl x')", "x = __import__('http')",
                "def broken(:"):
        assert owui_mod.side_effects(src) is True, src
    c = Chat()
    try:
        row = {"id": "m1", "name": "x", "tool_allow": None, "tool_deny": None,
               "tools_json": json.dumps([
                   {"name": "read", "inputSchema": {"type": "object"},
                    "annotations": {"readOnlyHint": True}},
                   {"name": "write", "inputSchema": {"type": "object"},
                    "annotations": {"readOnlyHint": False, "destructiveHint": True}},
                   {"name": "plain", "inputSchema": {"type": "object"}}])}
        tools = {t.name: t for t in c.svc.mcp._tools_of(row)}
        assert [tools[n].side_effects for n in ("mcp__x__read", "mcp__x__write", "mcp__x__plain")] == \
            [False, True, True]
        assert _d(tools["mcp__x__read"], {}, "auto") == ("allow", "auto")
        assert _d(tools["mcp__x__write"], {}, "auto") == ("ask", None)
        assert _d(tools["mcp__x__write"], {}, "yolo") == ("allow", "yolo")
    finally:
        c.close()


def _frames(sse, until=is_done):
    return sse.until(until)


def test_auto_is_the_default_and_bash_runs_without_a_prompt_but_is_logged():
    c = chat_with_sandbox()
    try:
        assert c.ok("GET", "/permissions/mode") == {"default": "auto", "modes": ["ask", "auto", "yolo"]}
        cid = c.new()
        assert c.snap(cid)["settings"]["permission_mode"] == "auto"
        assert c.ok("GET", f"/conversations/{cid}/permission-mode") == \
            {"mode": "auto", "effective": "auto", "default": "auto"}
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("bash", {"command": "ls -la"})))
        c.wait_idle(cid, 20)
        frames = _frames(sse)
        types = [f[1] for f in frames]
        assert "tool.asking" not in types, types
        started = json.loads(next(f for f in frames if f[1] == "run.started")[2])
        assert started["run"]["permission_mode"] == "auto"
        ev = json.loads(next(f for f in frames if f[1] == "tool.auto_approved")[2])
        assert ev["name"] == "bash" and ev["mode"] == "auto" and ev["summary"] == "Run: ls -la", ev
        s = c.snap(cid)
        call = next(p for p in parts_of(s) if p["type"] == "tool_call")
        assert call["status"] == "done" and call["approved"] == "auto", call
        started_ev = json.loads(next(f for f in frames if f[1] == "tool.started")[2])
        assert started_ev["approved"] == "auto", started_ev
        assert s["messages"][-1]["permission_mode"] == "auto"
        assert any("ls -la" in " ".join(cmd) for cmd in c.svc.sandboxes.ran)
        # the chat's event log keeps it
        logged = [json.loads(r["data"]) for r in c.svc.store.all(
            "SELECT data FROM events WHERE conversation_id = ? AND type = 'tool.auto_approved'", (cid,))]
        assert [e["call_id"] for e in logged] == [call["id"]], logged
        # rm on the workspace still asks
        c.send(cid, calls(("bash", {"command": "rm -rf /workspace"})))
        ask = json.loads(sse.until(lambda f: f[1] == "tool.asking")[-1][2])
        assert ask["summary"] == "Run: rm -rf /workspace"
        c.ok("POST", f"/conversations/{cid}/permissions/{ask['call_id']}", {"decision": "deny"})
        c.wait_idle(cid, 20)
        sse.close()
    finally:
        c.close()


def test_ask_mode_keeps_todays_prompts():
    c = chat_with_sandbox()
    try:
        cid = c.new()
        c.ok("PUT", f"/conversations/{cid}/permission-mode", {"mode": "ask"})
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("bash", {"command": "ls"}), ("write_file", {"path": "a.txt", "content": "x"})))
        ev = json.loads(sse.until(lambda f: f[1] == "tool.asking")[-1][2])
        assert ev["summary"] == "Run: ls"
        c.ok("POST", f"/conversations/{cid}/permissions/{ev['call_id']}", {"decision": "once"})
        c.wait_idle(cid, 20)
        frames = _frames(sse)
        assert not any(f[1] == "tool.auto_approved" for f in frames)
        s = c.snap(cid)
        assert s["messages"][-1]["permission_mode"] == "ask"
        calls_ = [p for p in parts_of(s) if p["type"] == "tool_call"]
        assert [p["status"] for p in calls_] == ["done", "done"], calls_
        assert calls_[0]["approved"] == "user" and calls_[1].get("approved") is None, calls_
        sse.close()
    finally:
        c.close()


def test_web_fetch_asks_once_per_domain_in_auto_and_never_in_yolo():
    c = Chat()
    try:
        cid = c.new()
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        # a LAN address: allowed, then refused by the egress filter (no request leaves the test)
        c.send(cid, calls(("web_fetch", {"url": "http://192.168.1.1/a"})))
        ev = json.loads(sse.until(lambda f: f[1] == "tool.asking")[-1][2])
        c.ok("POST", f"/conversations/{cid}/permissions/{ev['call_id']}", {"decision": "once"})
        c.wait_idle(cid, 20)
        sse.until(is_done)
        assert c.ok("GET", "/permissions/rules") == []          # once: no rule was written
        c.send(cid, calls(("web_fetch", {"url": "http://192.168.1.1/b"})))
        c.wait_idle(cid, 20)
        frames = sse.until(is_done)
        assert not any(f[1] == "tool.asking" for f in frames), [f[1] for f in frames]
        assert any(f[1] == "tool.auto_approved" for f in frames)
        # another domain asks again
        c.send(cid, calls(("web_fetch", {"url": "http://10.0.0.1/c"})))
        ev = json.loads(sse.until(lambda f: f[1] == "tool.asking")[-1][2])
        assert ev["summary"] == "Fetch http://10.0.0.1/c"
        # yolo from the next call on: the run's badge follows (run.mode), and nothing asks
        c.ok("PUT", f"/conversations/{cid}/permission-mode", {"mode": "yolo"})
        mode_ev = json.loads(sse.until(lambda f: f[1] == "run.mode")[-1][2])
        assert mode_ev["permission_mode"] == "yolo"
        c.ok("POST", f"/conversations/{cid}/permissions/{ev['call_id']}", {"decision": "deny"})
        c.wait_idle(cid, 20)
        sse.until(is_done)
        assert c.snap(cid)["messages"][-1]["permission_mode"] == "yolo"
        c.send(cid, calls(("web_fetch", {"url": "http://172.16.0.9/d"})))
        c.wait_idle(cid, 20)
        frames = sse.until(is_done)
        assert not any(f[1] == "tool.asking" for f in frames)
        ev = json.loads(next(f for f in frames if f[1] == "tool.auto_approved")[2])
        assert ev["mode"] == "yolo" and ev["name"] == "web_fetch"
        sse.close()
    finally:
        c.close()


def test_yolo_keeps_deny_rules_and_plan_mode():
    c = chat_with_sandbox()
    try:
        cid = c.new()
        c.ok("PUT", f"/conversations/{cid}/permission-mode", {"mode": "yolo"})
        c.svc.add_rule("bash", "curl", "deny", cid)
        c.send(cid, calls(("bash", {"command": "curl https://example.org"}),
                          ("bash", {"command": "rm -rf /workspace/tmp && ls"})))
        s = c.wait_idle(cid, 20)
        st = [p["status"] for p in parts_of(s) if p["type"] == "tool_call"]
        assert st == ["denied", "done"], st
        assert not any("curl" in " ".join(cmd) for cmd in c.svc.sandboxes.ran)
        # plan mode: read-only in yolo too
        c.ok("PATCH", f"/conversations/{cid}/settings", {"plan_mode": True})
        c.send(cid, calls(("write_file", {"path": "x.txt", "content": "no"}),
                          ("bash", {"command": "ls"})))
        s = c.wait_idle(cid, 30)
        last = [p for p in s["messages"][-1]["parts"] if p["type"] == "tool_call"]
        assert all(p["status"] == "failed" for p in last), last
        import os
        assert not os.path.exists(os.path.join(c.data, "workspaces", cid, "x.txt"))
    finally:
        c.close()


def test_the_api_and_the_default():
    c = Chat()
    try:
        st, obj, _ = c.call("PUT", "/permissions/mode", {"default": "sometimes"})
        assert st == 400
        assert c.ok("PUT", "/permissions/mode", {"default": "ask"})["default"] == "ask"
        cid = c.new()
        assert c.snap(cid)["settings"]["permission_mode"] == "ask"
        assert c.ok("GET", "/capabilities")["defaults"]["permission_mode"] == "ask"
        st, obj, _ = c.call("PUT", f"/conversations/{cid}/permission-mode", {"mode": "never"})
        assert st == 400
        st, obj, _ = c.call("PATCH", f"/conversations/{cid}/settings", {"permission_mode": "x"})
        assert st == 400
        assert c.ok("PUT", f"/conversations/{cid}/permission-mode", {"mode": "yolo"}) == \
            {"mode": "yolo", "effective": "yolo", "default": "ask"}
        # a chat from before the modes (no permission_mode) runs in the default
        c.svc.update_settings(cid, {"permission_mode": None})
        assert c.ok("GET", f"/conversations/{cid}/permission-mode") == \
            {"mode": None, "effective": "ask", "default": "ask"}
        # the settings page's call (dashboard remote.ts updateDefaults): PATCH /defaults
        d = c.ok("PATCH", "/defaults", {"permission_mode": "auto"})
        assert d["permission_mode"] == "auto" and "model" in d, d
        for bad in ({"permission_mode": "never"}, {"effort": "low"}, {}, {"permission_mode": None}):
            assert c.call("PATCH", "/defaults", bad)[0] == 400, bad
        assert c.ok("GET", f"/conversations/{cid}/permission-mode")["effective"] == "auto"
        assert c.snap(c.new())["settings"]["permission_mode"] == "auto"
    finally:
        c.close()


# ---------------------------------------------------------------------- chained commands (Ask mode)
MARIO = 'cd /workspace && node tools/check.js --level 1-1 > run.log 2>&1; echo "exit:$?"; cat run.log'


def _rule(match, decision="allow"):
    return {"tool": "bash", "match": match, "decision": decision}


def test_a_chained_command_is_covered_part_by_part():
    """The live bug of 2026-10-13 (chat 01M4TJG422BS6CS816PJR837A6): eleven "always allow cd" and
    the prompt came back every time, because the suggestion was the first part's prefix only."""
    cmd = {"command": MARIO}
    # every part needs a rule, any rule per part, from either scope
    assert T.decide(BASH, cmd, [_rule("cd"), _rule("node")], [], {}) == "ask"
    assert T.decide(BASH, cmd, [_rule("cd"), _rule("node")], [_rule("echo"), _rule("cat")], {}) == "allow"
    # what "always allow" offers: the parts not covered yet, each once, redirections ignored
    assert T.suggested_rule("bash", cmd) == {"tool": "bash", "match": "cd, node, echo, cat",
                                             "matches": ["cd", "node", "echo", "cat"]}
    assert T.suggested_rule("bash", cmd, [_rule("cd"), _rule("node")])["matches"] == ["echo", "cat"]
    assert T.bash_prefixes("ls > a.txt 2>&1 && ls -la >> b.txt | grep x") == ["ls", "grep"]
    assert T.bash_prefixes("git status && git push origin main") == ["git status", "git push"]
    # a deny in one part refuses the whole command, whatever covers the rest
    rest = [_rule("cd"), _rule("node"), _rule("echo")]
    assert T.decide(BASH, cmd, rest + [_rule("cat", "deny")], [], {}) == "deny"
    assert T.decide(BASH, cmd, rest, [_rule("cat", "deny")], {}) == "deny"
    # a command that hides a command is never allowed by a rule, and no rule is offered for it
    hidden = {"command": "cd /workspace && node $(cat which.txt)"}
    assert T.decide(BASH, hidden, [_rule("cd"), _rule("node")], [], {}) == "ask"
    assert T.suggested_rule("bash", hidden)["matches"] == []
    assert T.decide(BASH, {"command": "echo `id`"}, [_rule("echo")], [], {}) == "ask"


def test_always_allow_saves_every_part_once_and_the_prompt_does_not_come_back():
    c = chat_with_sandbox()
    try:
        cid = c.new()
        c.ok("PUT", f"/conversations/{cid}/permission-mode", {"mode": "ask"})
        c.svc.add_rule("bash", "cd", "allow", cid)
        assert c.svc.add_rule("bash", "cd", "allow", cid)["id"] == c.svc.rules(cid)[0]["id"]   # once
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("bash", {"command": MARIO})))
        ev = json.loads(sse.until(lambda f: f[1] == "tool.asking")[-1][2])
        assert ev["suggested_rule"] == {"tool": "bash", "match": "node, echo, cat",
                                        "matches": ["node", "echo", "cat"]}, ev
        c.ok("POST", f"/conversations/{cid}/permissions/{ev['call_id']}", {"decision": "always"})
        c.wait_idle(cid, 20)
        sse.until(is_done)
        assert sorted(r["match"] for r in c.ok("GET", "/permissions/rules")) == ["cat", "cd", "echo", "node"]
        # the same command again: no prompt; and no rule is written twice
        c.send(cid, calls(("bash", {"command": MARIO})))
        c.wait_idle(cid, 20)
        frames = sse.until(is_done)
        assert not any(f[1] == "tool.asking" for f in frames), [f[1] for f in frames]
        assert len(c.ok("GET", "/permissions/rules")) == 4
        calls_ = [p for p in parts_of(c.snap(cid)) if p["type"] == "tool_call"]
        assert [p["status"] for p in calls_] == ["done", "done"] and calls_[-1]["approved"] == "rule"
        sse.close()
    finally:
        c.close()


def test_the_store_drops_the_duplicate_rules_once():
    """Schema 6: the eleven "cd" rows of the live chat become one; other rules stay."""
    import os
    import tempfile
    from chatd.store import MIGRATIONS, Store
    d = tempfile.mkdtemp()
    path = os.path.join(d, "chat.sqlite3")
    old = Store(path, {k: v for k, v in MIGRATIONS.items() if k <= 5})
    rows = [("r%02d" % i, "bash", "cd", "allow", "01M4TJG422BS6CS816PJR837A6", i) for i in range(11)]
    rows += [("rn", "bash", "node", "allow", "01M4TJG422BS6CS816PJR837A6", 20),
             ("rg", "bash", "cd", "allow", None, 21), ("rd", "bash", "cd", "deny", "other", 22),
             ("rw", "web_fetch", None, "allow", None, 23), ("rw2", "web_fetch", None, "allow", None, 24)]
    for r in rows:
        old.run("INSERT INTO permission_rules VALUES (?, ?, ?, ?, ?, ?)", r)
    old.db.close()
    new = Store(path)
    got = sorted(r["id"] for r in new.all("SELECT id FROM permission_rules"))
    assert got == ["r00", "rd", "rg", "rn", "rw"], got


if __name__ == "__main__":
    run_all(globals())
