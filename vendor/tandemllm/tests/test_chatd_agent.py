"""chatd phase 2 (SRV-351, SRV-353, SRV-355, SRV-357): the agent loop, tools, permissions, the
context guard and compaction, the workspace, snapshots, the egress filter, the memory guard and
the terminals. CPU only: the model is tests/fake_backend.py (`FAKE_CALLS` makes it call our tools),
the sandbox a fake with the runtime's interface (the real one is the Spark's box test).

Run: python -m pytest -q tests/test_chatd_agent.py
"""

from __future__ import annotations

import base64
import json
import os
import socket
import tempfile
import threading
import time

from chatd_util import KEY, Chat, is_done, run_all, text_of

from chatd import egress, sandbox as sb_mod, tools as T
from chatd.snapshots import Snapshots
from chatd.workspace import ChangedError, PathError, Workspace, etag


def calls(*c) -> str:
    return "FAKE_CALLS " + json.dumps([{"name": n, "arguments": a} for n, a in c])


def parts_of(snap, role="assistant"):
    return [p for m in snap["messages"] if m["role"] == role for p in m["parts"]]


def requests(c):
    return [b for b in c.fake.bodies if "model" in b and b.get("stream")]


# ---------------------------------------------------------------------- a fake sandbox runtime
class FakeBoxes(sb_mod.Sandboxes):
    """The runtime's interface with the container engine replaced: `run` answers from `script`,
    `open_exec` gives a socketpair whose far end echoes (a fake PTY)."""

    def __init__(self, data):
        super().__init__(data, engine=None, log=lambda line: None)
        self.script = {}
        self.ran = []

    def available(self):
        return True

    def _start(self, cid):
        for d in self.paths(cid):
            os.makedirs(d, exist_ok=True)
        return "fake-" + cid

    def _proxy(self, b):
        pass

    def _alive(self, b):
        return True

    def stop(self, cid, reason="user"):
        b = self.box(cid)
        b.state, b.stopped_reason = "stopped", reason
        self._changed(b)

    def run(self, cid, cmd, *, timeout=600, on_output=None, cancel=None):
        self.ran.append(cmd)
        line = cmd[-1]
        for key, (out, code, delay) in self.script.items():
            if key in line:
                for chunk in out:
                    if cancel is not None and cancel.wait(delay):
                        return -15, False
                    on_output and on_output(chunk)
                if delay * len(out) > timeout:
                    return -9, True
                return code, False
        return 0, False

    def open_exec(self, cid, cmd, *, tty=False, stdin=False, env=None, workdir="/workspace"):
        a, b = socket.socketpair()

        def echo():
            b.sendall(b"sandbox:/workspace$ ")
            while True:
                try:
                    d = b.recv(4096)
                except OSError:
                    return
                if not d:
                    return
                b.sendall(d)
        threading.Thread(target=echo, daemon=True).start()
        return "exec-1", a, b""


def chat_with_sandbox():
    c = Chat()
    c.svc.sandboxes = FakeBoxes(c.data)
    c.svc.sandboxes.on_change = c.svc._sandbox_changed
    c.svc.sandboxes.network_mode = c.svc._network_mode
    c.svc._caps = None
    return c


# ---------------------------------------------------------------------- the loop
def test_a_tool_round_writes_a_file_and_the_model_reads_the_result():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, calls(("write_file", {"path": "notes/a.txt", "content": "hello"})))
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        call = next(p for p in a["parts"] if p["type"] == "tool_call")
        assert call["name"] == "write_file" and call["status"] == "done", call
        assert "wrote /workspace/notes/a.txt (5 bytes)" in call["output"]["content"]
        assert open(os.path.join(c.data, "workspaces", cid, "notes", "a.txt")).read() == "hello"
        assert a["status"] == "done" and "done0" in text_of(a)
        assert a["stats"]["steps"] == 2
        second = requests(c)[-1]["messages"]
        assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["function"][
            "name"] == "write_file"
        assert second[-1]["role"] == "tool" and "wrote /workspace/notes/a.txt" in \
            second[-1]["content"]
        assert [t["function"]["name"] for t in requests(c)[0]["tools"]][:3] == \
            ["read_file", "write_file", "edit_file"]
        assert requests(c)[0]["parallel_tool_calls"] is True
        assert "diff" not in json.dumps(c.ok("GET", f"/conversations/{cid}/sandbox/diff"))[:0]
        d = c.ok("GET", f"/conversations/{cid}/sandbox/diff?run_id={a['run_id']}")
        assert d["files"][0]["path"] == "/workspace/notes/a.txt" and d["files"][0]["status"] == \
            "added"
    finally:
        c.close()


def test_an_unknown_tool_and_bad_arguments_are_results_not_crashes():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, calls(("rm_everything", {}), ("read_file", {"path": "../../etc/passwd"})))
        s = c.wait_idle(cid, 20)
        p = [x for x in parts_of(s) if x["type"] == "tool_call"]
        assert p[0]["status"] == "failed" and "not available" in p[0]["output"]["content"]
        assert p[1]["status"] == "failed" and "not allowed" in p[1]["output"]["content"]
        assert s["messages"][-1]["status"] == "done"
    finally:
        c.close()


def test_read_calls_run_in_parallel_and_results_keep_order():
    c = Chat()
    try:
        cid = c.new()
        ws = c.svc.workspace(cid)
        for i in range(3):
            ws.write(f"f{i}.txt", f"file {i}\n".encode())
        c.send(cid, calls(*[("read_file", {"path": f"f{i}.txt"}) for i in range(3)]))
        c.wait_idle(cid, 20)
        tool_msgs = [m for m in requests(c)[-1]["messages"] if m["role"] == "tool"]
        assert [("file %d" % i) in m["content"] for i, m in enumerate(tool_msgs)] == [True] * 3
    finally:
        c.close()


def test_the_step_cap_ends_interrupted_and_continue_goes_on():
    c = Chat()
    try:
        cid = c.new()
        c.ok("PATCH", f"/conversations/{cid}/settings", {"max_steps": 1})
        c.send(cid, calls(("glob", {"pattern": "**/*"})))
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        assert a["status"] == "interrupted" and "limit of 1 model call" in a["parts"][-1]["message"]
        # the cap is no error: the chat's status is idle, the Continue card says the rest
        assert s["status"] == "idle", s["status"]
        assert next(x for x in c.ok("GET", "/conversations")["conversations"]
                    if x["id"] == cid)["status"] == "idle"
        c.ok("POST", f"/conversations/{cid}/continue")
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        assert a["status"] == "done" and "done0" in text_of(a)
        assert len(s["messages"]) == 2
    finally:
        c.close()


def test_a_new_chat_runs_up_to_100_model_calls():
    """Live (2026-10-17, the pagoda replay): a 15-file build used all 50 calls of the old default,
    22 of them writing files, and stopped while it fixed its first errors, before any check."""
    c = Chat()
    try:
        cid = c.new()
        snap = c.ok("GET", f"/conversations/{cid}")
        assert snap["settings"]["max_steps"] == 100
        assert c.ok("GET", "/capabilities")["defaults"]["max_steps"] == 100
        c.send(cid, calls(("glob", {"pattern": "**/*"})))
        s = c.wait_idle(cid, 20)
        assert s["messages"][-1]["status"] == "done"
        run = c.svc.store.one("SELECT max_steps FROM runs WHERE conversation_id = ?", (cid,))
        assert run["max_steps"] == 100
    finally:
        c.close()


def test_a_call_the_engine_ends_with_an_error_is_retried_once_then_fails_saying_so():
    """Live (2026-10-17, pagoda replay 4): the engine failed a call after 8,192 tokens of thinking
    (finish reason `error`); chatd took it for an empty answer, nudged, got the same, and the run
    failed with "The model answered with nothing, twice." """
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "hello FAKE_REASONING FAKE_FINISH_ERROR_ONCE")
        s = c.wait_idle(cid, 30)
        a = s["messages"][-1]
        assert a["status"] == "done" and a["stats"]["steps"] == 2, a["stats"]
        assert "w0" in text_of(a)
        assert not any("(Your last turn was empty" in json.dumps(b) for b in requests(c)), \
            "no empty-turn nudge for an engine error"
        c.send(cid, "again FAKE_REASONING FAKE_FINISH_ERROR")
        s = c.wait_idle(cid, 30)
        a = s["messages"][-1]
        assert a["status"] == "error" and a["stats"]["steps"] == 2
        assert "The engine failed this model call twice (finish reason error)" in json.dumps(a)
        assert "answered with nothing" not in json.dumps(a)
    finally:
        c.close()


def test_the_same_call_three_times_stops_the_run():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, calls(("glob", {"pattern": "*.py"})) + " FAKE_CALLS_ALWAYS")
        s = c.wait_idle(cid, 30)
        a = s["messages"][-1]
        assert [p["type"] for p in a["parts"]].count("tool_call") == 3
        assert "three times" in a["parts"][-1]["message"] and a["status"] == "done"
    finally:
        c.close()


def test_an_empty_turn_is_retried_once_with_a_nudge_then_an_error():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_EMPTY_ANSWER hello")              # thinking, then nothing
        s = c.wait_idle(cid, 20)
        reqs = requests(c)
        assert len(reqs) == 2 and reqs[-1]["messages"][-1]["content"].startswith("(Your last turn")
        assert s["messages"][-1]["status"] == "done"       # the fake answers the nudge
        assert "Your last turn" not in json.dumps(c.snap(cid))       # the nudge is not stored
        c.svc.step_empty = lambda run: True                 # a model that stays empty
        c.send(cid, "again")
        s = c.wait_idle(cid, 20)
        assert s["messages"][-1]["status"] == "error"
        assert "nothing, twice" in s["messages"][-1]["parts"][-1]["message"]
        assert len(requests(c)) == 4
    finally:
        c.close()


def test_a_steer_queued_during_the_tools_goes_in_after_the_results():
    c = chat_with_sandbox()
    try:
        c.svc.sandboxes.script["sleep"] = (["working\n"] * 10, 0, 0.1)
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        c.send(cid, calls(("bash", {"command": "sleep 1"})))
        c.wait(lambda: any(p.get("status") == "running" and p["type"] == "tool_call"
                           for p in parts_of(c.snap(cid))), 10)
        c.ok("POST", f"/conversations/{cid}/queue", {"client_msg_id": "s1", "text": "also this"})
        c.wait(lambda: len(c.snap(cid)["messages"]) == 4 and c.snap(cid)["run"] is None, 30)
        sent = requests(c)[-1]["messages"]
        roles = [m["role"] for m in sent if m["role"] != "system"]
        assert roles[-3:] == ["assistant", "tool", "user"] and sent[-1]["content"] == "also this"
    finally:
        c.close()


# ---------------------------------------------------------------------- permissions
def test_a_prompt_waits_two_devices_see_it_and_deny_with_a_note_reaches_the_model():
    c = Chat()
    try:
        cid = c.new()
        a = c.events(cid)
        b = c.events(cid)
        for x in (a, b):
            x.until(lambda f: f[1] == "current")
        c.send(cid, calls(("web_fetch", {"url": "http://example.org/x"})))
        asked = [x.until(lambda f: f[1] == "tool.asking")[-1] for x in (a, b)]
        ev = json.loads(asked[0][2])
        assert ev["summary"] == "Fetch http://example.org/x" and ev["risk"] == "network"
        assert ev["suggested_rule"] == {"tool": "web_fetch", "match": "example.org"}
        s = c.snap(cid)
        assert s["status"] == "waiting" and s["run"]["phase"] == "permission"
        time.sleep(1.5)                                 # a prompt does not time out
        assert c.snap(cid)["run"]["phase"] == "permission"
        call_id = ev["call_id"]
        c.ok("POST", f"/conversations/{cid}/permissions/{call_id}",
             {"decision": "deny", "note": "use the cached copy"})
        st, obj, _ = c.call("POST", f"/conversations/{cid}/permissions/{call_id}",
                            {"decision": "once"})
        assert st == 409                                # the second device was too late
        for x in (a, b):
            done = x.until(lambda f: f[1] == "tool.done")[-1]
            assert json.loads(done[2])["status"] == "denied"
            x.close()
        c.wait_idle(cid, 20)
        tool_msg = [m for m in requests(c)[-1]["messages"] if m["role"] == "tool"][0]
        assert tool_msg["content"] == "The user denied this call. Their note: use the cached copy"
    finally:
        c.close()


def test_a_run_waiting_for_permission_gives_its_run_slot_back():
    # live (rc-0.8.0, 2026-10-13): a chat waiting for a bash prompt held one of the two run slots,
    # and two other chats ran one after the other
    c = Chat(max_runs=2)
    try:
        cid = c.new()
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("web_fetch", {"url": "http://example.org/x"})))
        ev = json.loads(sse.until(lambda f: f[1] == "tool.asking")[-1][2])
        ids = [c.new() for _ in range(2)]
        for x in ids:
            c.send(x, "FAKE_SLOW FAKE_LONG 20 x")
        c.wait(lambda: c.fake.max_streams == 2, 10)      # both at once beside the waiting chat
        for x in ids:
            c.wait_idle(x, 20)
            assert "free run slot" not in ((c.snap(x)["run"] or {}).get("now") or "")
        assert c.snap(cid)["run"]["phase"] == "permission"
        # the answer: the run takes a slot again for its next call, and ends
        c.ok("POST", f"/conversations/{cid}/permissions/{ev['call_id']}", {"decision": "deny"})
        c.wait_idle(cid, 20)
        assert is_done(sse.until(is_done)[-1])
        sse.close()
    finally:
        c.close()


def test_always_allow_writes_a_rule_and_the_next_call_runs_without_asking():
    c = Chat()
    try:
        cid = c.new()
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("web_fetch", {"url": "http://192.168.1.1/admin"})))
        ev = json.loads(sse.until(lambda f: f[1] == "tool.asking")[-1][2])
        c.ok("POST", f"/conversations/{cid}/permissions/{ev['call_id']}",
             {"decision": "always", "scope": "chat"})
        c.wait_idle(cid, 20)
        call = [p for p in parts_of(c.snap(cid)) if p["type"] == "tool_call"][0]
        assert call["status"] == "failed" and "local network address" in call["output"]["content"]
        rules = c.ok("GET", "/permissions/rules")
        assert rules[0]["tool"] == "web_fetch" and rules[0]["match"] == "192.168.1.1"
        assert rules[0]["scope"] == "chat"
        c.send(cid, calls(("web_fetch", {"url": "http://192.168.1.1/other"})))
        c.wait_idle(cid, 20)
        frames = sse.until(is_done)
        assert not any(f[1] == "tool.asking" for f in frames)
        sse.close()
        c.call("DELETE", f"/permissions/rules/{rules[0]['id']}")
        assert c.ok("GET", "/permissions/rules") == []
    finally:
        c.close()


def test_rule_matching():
    bash = T.BY_NAME["bash"]
    allow_npm = [{"tool": "bash", "match": "npm install", "decision": "allow"}]
    assert T.decide(bash, {"command": "npm install three"}, allow_npm, [], {}) == "allow"
    assert T.decide(bash, {"command": "npm installx three"}, allow_npm, [], {}) == "ask"
    assert T.decide(bash, {"command": "npm install a && rm -rf /"}, allow_npm, [], {}) == "ask"
    assert T.decide(bash, {"command": "npm install $(curl x)"}, allow_npm, [], {}) == "ask"
    both = allow_npm + [{"tool": "bash", "match": "git status", "decision": "allow"}]
    assert T.decide(bash, {"command": "npm install && git status"}, both, [], {}) == "allow"
    deny = [{"tool": "bash", "match": "npm", "decision": "deny"}]
    assert T.decide(bash, {"command": "npm install x"}, allow_npm + deny, [], {}) == "deny"
    # chat rules before global ones
    assert T.decide(bash, {"command": "npm install x"}, allow_npm,
                    [{"tool": "bash", "match": "npm", "decision": "deny"}], {}) == "allow"
    assert T.decide(bash, {"command": "ls"}, [], [], {"policies": {"bash": "always"}}) == "allow"
    w = T.BY_NAME["write_file"]
    assert T.decide(w, {}, [], [], {"policies": {"write_file": "ask"}}) == "ask"
    assert T.decide(w, {}, [], [], {}) == "allow"
    mcp = T.Tool("mcp__x1__search", "", {}, default="ask")
    assert T.decide(mcp, {}, [], [{"tool": "mcp__x1__*", "decision": "allow"}], {}) == "allow"
    assert T.bash_prefix("npm install three") == "npm install"
    assert T.bash_prefix("python3 app.py") == "python3"
    assert T.bash_prefix("ls -la && rm x") == "ls"
    wf = T.BY_NAME["web_fetch"]
    rule = [{"tool": "web_fetch", "match": "example.org", "decision": "allow"}]
    assert T.decide(wf, {"url": "https://docs.example.org/a"}, rule, [], {}) == "allow"
    assert T.decide(wf, {"url": "https://example.org.evil.com/"}, rule, [], {}) == "ask"


# ---------------------------------------------------------------------- bash in the sandbox
def test_bash_streams_its_output_and_a_stop_cancels_it():
    c = chat_with_sandbox()
    try:
        c.svc.sandboxes.script["pytest"] = (["collected 2\n", "..\n", "2 passed\n"], 0, 0.05)
        c.svc.sandboxes.script["sleep"] = (["tick\n"] * 100, 0, 0.1)
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("bash", {"command": "pytest -q"})))
        frames = sse.until(is_done, 20)
        outs = [json.loads(f[2])["delta"] for f in frames if f[1] == "tool.output"]
        assert "".join(outs) == "collected 2\n..\n2 passed\n"
        call = [p for p in parts_of(c.snap(cid)) if p["type"] == "tool_call"][0]
        assert call["status"] == "done" and call["output"]["exit_code"] == 0
        assert c.snap(cid)["sandbox"]["state"] == "running"
        # a long command, then Stop
        c.send(cid, calls(("bash", {"command": "sleep 100"})))
        c.wait(lambda: any(f[1] == "tool.started" for f in [sse.frames.get(timeout=10)]), 10)
        time.sleep(0.4)
        c.ok("POST", f"/conversations/{cid}/stop")
        s = c.wait_idle(cid, 20)
        call = [p for p in parts_of(s) if p["type"] == "tool_call"][-1]
        assert call["status"] == "cancelled" and s["messages"][-1]["status"] == "stopped"
        sse.close()
    finally:
        c.close()


def test_a_long_output_is_cut_for_the_model_and_saved_whole():
    c = chat_with_sandbox()
    try:
        big = "x" * 100_000 + "\n"
        c.svc.sandboxes.script["cat big"] = ([big], 0, 0)
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        c.send(cid, calls(("bash", {"command": "cat big"})))
        c.wait_idle(cid, 20)
        tool_msg = [m for m in requests(c)[-1]["messages"] if m["role"] == "tool"][0]["content"]
        assert len(tool_msg) < 40_000 and "the whole output is in /workspace/.tandem/outputs/" \
            in tool_msg
        call = [p for p in parts_of(c.snap(cid)) if p["type"] == "tool_call"][0]
        saved = os.path.join(c.data, "workspaces", cid, ".tandem", "outputs", call["id"] + ".txt")
        assert os.path.getsize(saved) == len(big)
        assert call["output"]["truncated"] is True and len(call["output"]["content"]) <= 30_010
    finally:
        c.close()


# ---------------------------------------------------------------------- the context guard
def small_window(c, window=3000, backend="engine"):
    real = c.svc.model_info

    def info(mid):
        out = dict(real(mid))
        out.update(max_len=window, backend=backend, max_output=None)
        return out
    c.svc.model_info = info


def test_compaction_summarizes_older_turns_and_the_model_gets_the_summary():
    c = Chat()
    try:
        small_window(c, window=5000)
        c.svc.learn_ratio = lambda *a: None                 # the fake counts words, not tokens
        cid = c.new()
        c.svc.conv(cid).tok_ratio = 0.5
        long = "lorem ipsum dolor " * 120                          # ~2,200 characters a turn
        for i in range(3):
            c.send(cid, f"turn {i}: {long}")
            c.wait_idle(cid, 20)
        c.send(cid, "and now?")
        s = c.wait_idle(cid, 30)
        assert len(s["compactions"]) >= 1
        cp = s["compactions"][-1]
        assert cp["summary"].startswith("w0") and cp["before_tokens"] > 0
        summ = [b for b in c.fake.bodies if b.get("stream") is False and
                b.get("tandem", {}).get("compaction")]
        assert summ and summ[-1]["messages"][-1]["content"].startswith("Write a summary")
        last = requests(c)[-1]
        assert last["tandem"] == {"compaction": False}
        system = last["messages"][0]
        assert system["role"] == "system" and "Summary of the earlier part" in system["content"]
        users = [m["content"] for m in last["messages"] if m["role"] == "user"]
        assert len(users) == 2 and users[-1] == "and now?" and users[0].startswith("turn 2")
        assert last["max_tokens"] < 5000
        assert len(s["messages"]) == 8                          # the store keeps every message
        # the thread's divider sits above the first message the model still reads in full
        kept = next(m for m in s["messages"] if m["id"] == cp["boundary_id"])
        assert kept["role"] == "user" and kept["parts"][0]["text"].startswith("turn 2")
    finally:
        c.close()


# ---------------------------------------------------------------------- the engine's memory refusal
# Live 2026-10-17 ("Voxel Pagoda City Build"): a run compacted at step 43 grew back to 215,247
# tokens, under 80 % of its 262,144 window, and the engine answered every call with a 503 ("a
# 215247-token prefill needs 53.3 GiB with every cache given back; the engine's peak is 52.6 GiB")
# while another chat ran on its other slot. chatd sent the same prompt every 60 s for its 10-minute
# retry window and the run ended interrupted.

def _memory_run(c, max_steps=14):
    small_window(c, window=100_000)                     # the window is never the limit here
    c.svc.learn_ratio = lambda *a: None                 # the fake counts words, not tokens
    cid = c.new()
    c.svc.conv(cid).tok_ratio = 0.5
    ws = c.svc.workspace(cid)
    ws.write("a.txt", ("alpha line\n" * 60).encode())
    ws.write("b.txt", ("beta line\n" * 60).encode())
    c.ok("PATCH", f"/conversations/{cid}/settings", {"max_steps": max_steps})
    return cid


def _prompt_words(b: dict) -> int:
    return sum(len(str(m.get("content") or "").split()) + 4 for m in b["messages"])


def test_a_memory_refusal_compacts_and_the_run_goes_on():
    c = Chat(retry_s=4)
    try:
        cid = _memory_run(c)
        c.fake.memory = 2500             # each step adds ~380 of the fake's tokens: step 8 is over
        ev = c.events(cid)
        c.send(cid, calls(("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"}))
               + " FAKE_CALLS_ALWAYS")
        frames = ev.until(is_done, 60)
        ev.close()
        s = c.wait_idle(cid, 10)
        a = s["messages"][-1]
        # one retry after the Retry-After, then a compaction; the guard compacts before the
        # refused size from then on, so the engine refuses nothing more
        assert c.fake.refused == 2, c.fake.refused
        assert a["stats"]["interrupted"] == "limit" and a["stats"]["steps"] == 14, a["stats"]
        assert len(s["compactions"]) >= 2 and "@" in s["compactions"][0]["boundary_id"]
        sent = [b for b in requests(c) if _prompt_words(b) <= 2500]
        assert len(sent) == len(requests(c)) - 2
        nows = [json.loads(f[2]).get("now") or "" for f in frames if f[1] == "run.phase"]
        assert any(n.startswith("The engine has no memory for this chat's context right now (2k "
                                "tokens, needs 53.3 GiB of 52.6): retry in 1 s, then compact")
                   for n in nows), nows
        assert not any(n.startswith("Waiting for the engine") for n in nows)
        assert c.svc.conv(cid).memory_limit                  # remembered for the chat
    finally:
        c.close()


def test_a_memory_refusal_right_after_the_compaction_ends_as_before_with_the_reason():
    c = Chat(retry_s=3)
    try:
        cid = _memory_run(c)
        c.fake.memory = 2500
        real = c.svc.compact

        def compact(*a, **kw):
            ok = real(*a, **kw)
            c.fake.memory = 50          # the other slot's chat took more meanwhile: still no room
            return ok
        c.svc.compact = compact
        ev = c.events(cid)
        c.send(cid, calls(("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"}))
               + " FAKE_CALLS_ALWAYS")
        frames = ev.until(is_done, 60)
        ev.close()
        s = c.wait_idle(cid, 10)
        a = s["messages"][-1]
        assert a["status"] == "interrupted" and a["stats"]["interrupted"] == "engine", a["stats"]
        assert len(s["compactions"]) == 1                    # no second compaction, no loop
        assert c.fake.refused <= 6, c.fake.refused           # the backoff, not a tight loop
        words = "The engine has no memory for this chat's context right now ("
        assert words in a["parts"][-1]["message"], a["parts"][-1]
        nows = [json.loads(f[2]).get("now") or "" for f in frames if f[1] == "run.phase"]
        assert any(n.startswith(words) and n.endswith(" s") and "compact" not in n
                   for n in nows), nows
        assert not any(n.startswith("Waiting for the engine") for n in nows)
    finally:
        c.close()


# ---------------------------------------------------------------------- over the window (#18)
# A tool result can grow the request past the window after the guard's estimate: a relay answered
# "request: 266,500 tokens (input 262,025 + requested output 4,475)" on a 262,144-token model and
# the run failed. The 400 compacts and the call goes once more.

def test_over_window_reads_the_engines_and_the_backends_words():
    from chatd.engine import EngineError
    from chatd.runs import over_window
    yes = ["prompt is 262200 tokens and the context is 262144; nothing is left to generate",
           "This model's maximum context length is 131072 tokens. However, you requested 140000 "
           "tokens (135000 in the messages, 5000 in the completion).",
           "The model's maximum context is 262,144 tokens. Increase the engine's context length, "
           "or lower the request size. request: 266,500 tokens (input 262,025 + requested output "
           "4,475)",
           "the request exceeds the available context size, try increasing it",
           "The input (270000 tokens) is longer than the model's context length (262144 tokens)."]
    for m in yes:
        assert over_window(EngineError(400, m, False)), m
    assert not over_window(EngineError(400, "fake 400", False))
    assert not over_window(EngineError(400, "'messages' is a required property", False))
    assert not over_window(EngineError(503, yes[0], True))      # only the 400 is this case


def test_an_over_window_400_compacts_and_the_run_goes_on():
    c = Chat()
    try:
        cid = _memory_run(c)                     # the guard's window (100,000) is never the limit
        c.fake.window = 2500                     # each step adds ~380 of the fake's tokens
        ev = c.events(cid)
        c.send(cid, calls(("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"}))
               + " FAKE_CALLS_ALWAYS")
        frames = ev.until(is_done, 60)
        ev.close()
        s = c.wait_idle(cid, 10)
        a = s["messages"][-1]
        assert c.fake.windowed >= 1
        assert a["stats"]["interrupted"] == "limit" and a["stats"]["steps"] == 14, a["stats"]
        assert len(s["compactions"]) >= c.fake.windowed, (s["compactions"], c.fake.windowed)
        # each refused call went once more, compacted and under the window
        sent = requests(c)
        over = [i for i, b in enumerate(sent) if _prompt_words(b) > 2500]
        assert len(over) == c.fake.windowed
        assert all(i + 1 < len(sent) and _prompt_words(sent[i + 1]) <= 2500 for i in over), over
        nows = [json.loads(f[2]).get("now") or "" for f in frames if f[1] == "run.phase"]
        assert "The request is over the model's context window: compacting, then one retry" \
            in nows, nows
    finally:
        c.close()


def test_an_over_window_400_that_compaction_cannot_cut_fails_at_once():
    """#18's shape: one user turn, nothing before it to summarize. The same body would get the
    same 400, so it is not sent again."""
    c = Chat()
    try:
        cid = _memory_run(c)
        c.fake.window = 30                       # the user message alone is over it
        ev = c.events(cid)
        c.send(cid, "word " * 100)
        frames = ev.until(is_done, 30)
        ev.close()
        s = c.wait_idle(cid, 10)
        a = s["messages"][-1]
        assert c.fake.windowed == 1, c.fake.windowed
        assert s["compactions"] == []
        assert a["status"] == "error", a
        assert "nothing is left to generate" in a["parts"][-1]["message"], a["parts"][-1]
        assert sum(1 for f in frames if is_done(f)) == 1
    finally:
        c.close()


def test_a_second_over_window_400_after_a_compaction_fails_the_run_once():
    c = Chat()
    try:
        cid = _memory_run(c)
        c.fake.window = 2500
        real = c.svc.compact

        def compact(*a, **kw):
            ok = real(*a, **kw)
            c.fake.window = 50                   # still over after the cut
            return ok
        c.svc.compact = compact
        ev = c.events(cid)
        c.send(cid, calls(("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"}))
               + " FAKE_CALLS_ALWAYS")
        frames = ev.until(is_done, 60)
        ev.close()
        s = c.wait_idle(cid, 10)
        a = s["messages"][-1]
        assert c.fake.windowed == 2, c.fake.windowed               # one retry, no loop
        assert len(s["compactions"]) == 1
        assert a["status"] == "error", a
        assert "and the context is 50;" in a["parts"][-1]["message"], a["parts"][-1]
        assert sum(1 for f in frames if is_done(f)) == 1
    finally:
        c.close()


def test_another_400_is_not_retried():
    c = Chat()
    try:
        cid = _memory_run(c)
        n = len(requests(c))
        c.fake.refuse = 400
        c.send(cid, "hello")
        s = c.wait_idle(cid, 10)
        a = s["messages"][-1]
        assert a["status"] == "error" and a["parts"][-1]["message"] == "fake 400", a
        assert len(requests(c)) == n + 1
        assert s["compactions"] == []
    finally:
        c.close()


# ---------------------------------------------------------------------- results sized to the room
# #18's own shape (SRV-639): one user turn, the big result in the last step. A paragraph-per-line
# file read with the default limit filled the window, nothing could be compacted, and the run died
# with an over-window 400. One step's results now share a budget of the room left, and read_file
# cuts at whole lines and names the offset to read on from.

class _Ctx:
    def __init__(self, ws, cap=None, agents=False):
        self.ws, self.cap, self.agents = ws, cap, agents


def _ws(files: dict) -> Workspace:
    ws = Workspace(tempfile.mkdtemp(prefix="rd18-"))
    for name, text in files.items():
        ws.write(name, text.encode())
    return ws


def _numbered(out: str) -> dict:
    """{line number: text} of a read's rows (a row in part where the read cut a line)."""
    rows = {}
    for row in out.splitlines():
        if row[:6].strip().isdigit() and row[6:7] == "\t":
            rows[int(row[:6])] = row[7:]
    return rows


def test_a_long_line_file_is_read_in_honest_pages():
    import re
    paras = [f"para {i}: " + " ".join(f"w{i}x{k}" for k in range(400)) for i in range(1, 301)]
    ws = _ws({"town.md": "\n".join(paras) + "\n"})
    got, off, reads = {}, 1, 0
    while True:
        out = T._read(_Ctx(ws, cap=20_000), {"path": "town.md", "offset": off}).content
        reads += 1
        rows = _numbered(out)
        assert len(out) <= 20_000 + 200, len(out)
        assert all(rows[i] == paras[i - 1] for i in rows), "whole lines only"
        got.update(rows)
        m = re.search(r"\[truncated at 20,000 characters: lines (\d+)-(\d+) of 300 shown, "
                      r"([\d,]+) more bytes; read on with offset=(\d+)\]$", out)
        if m is None:
            assert "more lines" not in out and "truncated" not in out
            break
        assert int(m.group(1)) == off and int(m.group(4)) == int(m.group(2)) + 1
        rest = sum(len(p.encode()) + 1 for p in paras[int(m.group(2)):])
        assert int(m.group(3).replace(",", "")) == rest
        off = int(m.group(4))
    assert sorted(got) == list(range(1, 301)) and reads > 10
    # no budget: the result caps (64,000 characters), not the whole 2,000 lines
    out = T._read(_Ctx(ws), {"path": "town.md"}).content
    assert len(out) <= T.MODEL_LIMIT_CHARS + 200 and out.endswith("]") and "offset=" in out
    # `limit` stopped it: the line count and the offset
    out = T._read(_Ctx(ws), {"path": "town.md", "offset": 3, "limit": 2}).content
    assert sorted(_numbered(out)) == [3, 4] and out.endswith("[296 more lines; read on with "
                                                             "offset=5]"), out[-80:]


def test_a_line_longer_than_one_read_pages_by_column():
    line = "".join(f"{k:07d} " for k in range(40_000))           # 320,000 characters
    ws = _ws({"min.js": "head\n" + line + "\ntail\n"})
    out = T._read(_Ctx(ws, cap=50_000), {"path": "min.js"}).content
    assert _numbered(out) == {1: "head"} and out.endswith("read on with offset=2]"), out[-120:]
    pieces, col = [], 1
    while True:
        out = T._read(_Ctx(ws, cap=50_000), {"path": "min.js", "offset": 2,
                                             "column": col}).content
        rows = _numbered(out)
        pieces.append(rows[2])
        assert len(out) <= 50_200
        if "goes on" not in out:
            assert rows.get(3) == "tail"
            break
        tail = out.rsplit("\n", 1)[1]
        left = len(line) - (col - 1) - len(rows[2])
        assert tail == (f"[line 2 goes on: {left:,} more bytes of it; read on with offset=2, "
                        f"column={col + len(rows[2])}]"), tail
        col += len(rows[2])
    assert "".join(pieces) == line and len(pieces) == 7
    # Agents mode: its 12,000 characters, the same paging
    out = T._read(_Ctx(ws, agents=True), {"path": "min.js", "offset": 2}).content
    assert len(_numbered(out)[2]) < 12_000 and "read on with offset=2, column=" in out


def test_step_budget_and_margin():
    from chatd import compaction as comp
    # half of what is left under the compaction line, 2,000 tokens at least, never past the
    # window less twice the answer's room
    assert comp.step_budget(262_144, 11_000) == int((209_715 - 11_000) * 0.5)
    assert comp.step_budget(262_144, 208_000) == comp.RESULT_FLOOR
    assert comp.step_budget(262_144, 259_500) == 262_144 - 259_500 - 2048
    assert comp.step_budget(262_144, 262_000) == 0
    assert comp.step_budget(0, 5) is None
    assert comp.result_chars(10_000, 4, 0.25) == int(2500 / (0.25 * 1.15))
    assert comp.result_chars(0, 11, 0.25) == comp.MIN_RESULT_CHARS
    # the margin: 15 % of the estimate of what the engine never counted, 256 at least
    assert comp.margin(700_000, 30_000, 0.25) == int(670_000 * 0.25 * 0.15)
    assert comp.margin(30_100, 30_000, 0.25) == comp.MIN_MARGIN
    assert comp.margin(10_000, 0, None) == int(10_000 * 0.3 * 0.15)


def _paragraphs(n=300, words=400) -> str:
    return "\n".join(f"para {i}: " + "lorem " * words for i in range(1, n + 1)) + "\n"


def _read_run(c, window: int, *reads: dict, strict: bool = False):
    small_window(c, window=window)
    cid = c.new()
    c.svc.workspace(cid).write("town.md", _paragraphs().encode())
    c.fake.window, c.fake.strict = window, strict
    c.send(cid, calls(*(("read_file", dict({"path": "town.md"}, **r)) for r in reads)))
    s = c.wait_idle(cid, 30)
    return s, [p for p in parts_of(s) if p.get("type") == "tool_call"]


def test_a_paragraph_file_read_fits_the_room_and_the_run_answers():
    """On main the read came back as 64,000 characters (about 10,700 of the fake's tokens) into
    a 6,000-token window: an over-window 400 that nothing could compact."""
    c = Chat()
    try:
        s, reads = _read_run(c, 6000, {})
        a = s["messages"][-1]
        assert a["status"] == "done" and c.fake.windowed == 0, (a["status"], a["parts"][-1])
        text = reads[0]["model_content"]
        assert "read on with offset=" in text and len(text) < 20_000, len(text)
        last = requests(c)[-1]
        assert c.fake.prompt_tokens(last)[0] + last["max_tokens"] <= 6000
    finally:
        c.close()


def test_many_reads_in_one_step_share_the_budget():
    c = Chat()
    try:
        s, reads = _read_run(c, 6000, *({"offset": 1 + 27 * k} for k in range(11)), strict=True)
        a = s["messages"][-1]
        assert len(reads) == 11 and all(r["status"] == "done" for r in reads)
        assert a["status"] == "done" and c.fake.windowed == 0, a["parts"][-1]
        assert all(len(r["model_content"]) < 3000 for r in reads)
        assert all("read on with offset=" in r["model_content"] for r in reads)
        for b in requests(c):                    # every request inside the window, output too
            assert c.fake.prompt_tokens(b)[0] + b["max_tokens"] <= 6000
    finally:
        c.close()


def _edge_chat(c, window: int, backend: str = "engine", max_output=None):
    real = c.svc.model_info

    def info(mid):
        return dict(real(mid), max_len=window, backend=backend, max_output=max_output)
    c.svc.model_info = info
    c.svc.learn_ratio = lambda *a: None
    cid = c.new()
    c.svc.conv(cid).tok_ratio = 0.5
    return cid


def test_at_the_edge_our_engine_counts_the_request_exactly():
    c = Chat()
    try:
        cid = _edge_chat(c, 12_000)
        c.fake.tokenize = 6000
        c.send(cid, "word " * 1800)            # about 4,700 estimated: no count needed
        s = c.wait_idle(cid, 20)
        assert s["messages"][-1]["status"] == "done" and c.fake.counted == 0
        c.send(cid, "more " * 1700)            # about 9,200 and its margin: the edge
        s = c.wait_idle(cid, 20)
        assert s["messages"][-1]["status"] == "done" and c.fake.counted == 1
        assert requests(c)[-1]["max_tokens"] == 12_000 - 6000 - 256 - 256
    finally:
        c.close()


def test_a_request_the_engine_counts_over_the_window_is_never_sent():
    c = Chat()
    try:
        cid = _edge_chat(c, 12_000)
        c.fake.tokenize = 11_500
        n = len(requests(c))
        c.send(cid, "word " * 3500)            # about 9,000 estimated: it would go
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        assert a["status"] == "error" and c.fake.counted == 1 and len(requests(c)) == n
        assert a["parts"][-1]["message"].startswith(
            "This request is 11,500 tokens (the engine's count) and the model's window is "
            "12,000: there is no room left for an answer."), a["parts"][-1]["message"]
    finally:
        c.close()


def test_a_relayed_model_keeps_the_estimate_margin_free():
    c = Chat()
    try:
        cid = _edge_chat(c, 12_000, backend="proxy", max_output=8192)
        c.fake.tokenize = 1                     # never asked: it counts our engine's model only
        c.send(cid, "word " * 2800)            # about 9,200 and its margin: the edge
        s = c.wait_idle(cid, 20)
        assert s["messages"][-1]["status"] == "done" and c.fake.counted == 0
        b = requests(c)[-1]
        chars = len(json.dumps(b["messages"], ensure_ascii=False)) + \
            len(json.dumps(b.get("tools") or [], ensure_ascii=False))
        est = int(chars * 0.5) + 16
        assert b["max_tokens"] == 12_000 - est - int(chars * 0.5 * 0.15) - 256, b["max_tokens"]
    finally:
        c.close()


def test_a_request_that_cannot_fit_fails_before_the_engine_is_asked():
    c = Chat()
    try:
        small_window(c, window=1200)
        cid = c.new()
        n = len(requests(c))
        c.send(cid, "word " * 2000)
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        assert a["status"] == "error" and "window is 1,200" in a["parts"][-1]["message"]
        assert len(requests(c)) == n
    finally:
        c.close()


def test_earlier_reasoning_is_left_out_when_nothing_else_helps():
    c = Chat()
    try:
        small_window(c, window=2700)          # read_file's `column` (SRV-639) took 2,600's slack
        cid = c.new()
        c.send(cid, "FAKE_REASONING FAKE_LONG 6 " + "w " * 300)
        c.wait_idle(cid, 20)
        c.svc.conv(cid).tok_ratio = 0.3
        # make the first answer's reasoning big in the store
        conv = c.svc.conv(cid)
        mid = c.snap(cid)["messages"][1]["id"]
        row = c.svc.store.one("SELECT parts FROM messages WHERE id = ?", (mid,))
        parts = json.loads(row[0])
        parts[0]["text"] = "thinking " * 900
        c.svc.store.run("UPDATE messages SET parts = ? WHERE id = ?", (json.dumps(parts), mid))
        c.send(cid, "next")
        s = c.wait_idle(cid, 20)
        assert s["messages"][-1]["status"] == "done", s["messages"][-1]
        last = requests(c)[-1]
        assert not any("reasoning_content" in m for m in last["messages"])
        assert conv is c.svc.conv(cid)
    finally:
        c.close()


# ---------------------------------------------------------------------- the workspace
def test_workspace_paths_refuse_escapes():
    d = tempfile.mkdtemp()
    ws = Workspace(d)
    ws.ensure()
    ws.write("a/b.txt", b"hi")
    assert ws.read("/workspace/a/b.txt") == b"hi" and ws.read("a/b.txt") == b"hi"
    for bad in ("../x", "/etc/passwd", "a/../../x", "/workspace/../etc"):
        try:
            ws.resolve(bad)
            raise AssertionError(bad)
        except PathError:
            pass
    outside = tempfile.mkdtemp()
    with open(os.path.join(outside, "secret"), "w") as fh:
        fh.write("no")
    os.symlink(outside, os.path.join(d, "link"))
    os.symlink(os.path.join(outside, "secret"), os.path.join(d, "s.txt"))
    for bad in ("link/secret", "s.txt"):
        try:
            ws.read(bad)
            raise AssertionError(bad)
        except (PathError, OSError):
            pass
    assert "/workspace/link" not in ws.glob("**/*")
    e = etag(b"hi")
    ws.write("a/b.txt", b"new", if_match=e)
    try:
        ws.write("a/b.txt", b"x", if_match=e)
        raise AssertionError("a stale etag must fail")
    except ChangedError:
        pass
    assert ws.grep("ne.") == ["/workspace/a/b.txt:1: new"]


def test_the_files_api():
    c = Chat()
    try:
        cid = c.new()
        base = f"/conversations/{cid}/sandbox"
        st, _, _ = c.call("PUT", f"{base}/file?path=/workspace/src/app.py",
                          raw=b"print('hi')\n", headers={"Content-Type": "text/plain"})
        assert st == 204
        f = c.ok("GET", f"{base}/file?path=/workspace/src/app.py")
        assert f["content"] == "print('hi')\n" and f["etag"] == etag(b"print('hi')\n")
        st, _, _ = c.call("PUT", f"{base}/file?path=src/app.py", raw=b"x",
                          headers={"If-Match": "stale", "Content-Type": "text/plain"})
        assert st == 412
        assert [x["path"] for x in c.ok("GET", f"{base}/files?path=/workspace")] == \
            ["/workspace/src"]
        st, obj, _ = c.call("GET", f"{base}/file?path=../../x")
        assert st == 400
        boundary = "B"
        raw = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"up.txt"
               f"\"\r\nContent-Type: text/plain\r\n\r\nuploaded\r\n--{boundary}--\r\n").encode()
        up = c.ok("POST", f"{base}/upload?dir=/workspace/in", raw=raw,
                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        assert up[0]["path"] == "/workspace/in/up.txt"
        st, body, h = c.call("GET", f"{base}/view/src/app.py")
        assert st == 200 and "frame-ancestors 'self'" in h["Content-Security-Policy"]
        a = c.ok("GET", f"/conversations/{cid}/artifacts/" + "file%3A%2Fworkspace%2Fsrc%2Fapp.py")
        assert a["kind"] == "file" and a["content"] == "print('hi')\n"
        st, _, _ = c.call("DELETE", f"{base}/file?path=in/up.txt")
        assert st == 204
    finally:
        c.close()


def test_snapshots_diff_and_undo_with_renames_and_deletes():
    d = tempfile.mkdtemp()
    ws = Workspace(os.path.join(d, "ws"))
    ws.ensure()
    ws.write("keep.txt", b"one\ntwo\n")
    ws.write("old_name.py", b"x = 1\n")
    ws.write("gone.txt", b"bye\n")
    snaps = Snapshots(os.path.join(d, "snaps"))
    snaps.take("c1", ws, "r1", "before")
    ws.write("keep.txt", b"one\nTWO\n")
    ws.write("new_name.py", ws.read("old_name.py"))
    ws.delete("old_name.py")
    ws.delete("gone.txt")
    snaps.take("c1", ws, "r1", "after")
    files = {f["path"]: f for f in snaps.diff("c1", "r1")["files"]}
    assert files["/workspace/keep.txt"]["status"] == "modified"
    assert files["/workspace/keep.txt"]["additions"] == 1 and \
        files["/workspace/keep.txt"]["deletions"] == 1
    assert files["/workspace/new_name.py"]["status"] == "added"
    assert files["/workspace/old_name.py"]["status"] == "deleted"
    assert files["/workspace/gone.txt"]["status"] == "deleted"
    n = snaps.undo("c1", ws, "r1", "u1")
    assert n == 4
    assert ws.read("keep.txt") == b"one\ntwo\n" and ws.read("old_name.py") == b"x = 1\n"
    assert ws.read("gone.txt") == b"bye\n"
    try:
        ws.read("new_name.py")
        raise AssertionError("the added file must be gone")
    except (PathError, OSError):
        pass
    snaps.undo("c1", ws, "u1", "u2")                      # the undo can be undone
    assert ws.read("keep.txt") == b"one\nTWO\n" and ws.read("new_name.py") == b"x = 1\n"


# ---------------------------------------------------------------------- the egress filter
def test_the_egress_filter_refuses_the_lan_and_this_box():
    for bad in ("10.0.0.1", "192.168.1.131", "172.20.0.1", "127.0.0.1", "169.254.1.1",
                "100.64.0.1", "::1", "fd00::1", "fe80::1", "::ffff:192.168.1.1", "0.0.0.0",
                "224.0.0.1"):
        assert not egress.allowed(bad), bad
    for good in ("1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"):
        assert egress.allowed(good), good
    for host in ("localhost", "127.0.0.1"):
        try:
            egress.resolve(host, 80)
            raise AssertionError(host)
        except egress.Refused:
            pass
    d = tempfile.mkdtemp()
    mode = {"m": "on"}
    p = egress.Proxy(os.path.join(d, "proxy.sock"), lambda: mode["m"])
    p.start()
    try:
        def ask(req: bytes) -> bytes:
            s = socket.socket(socket.AF_UNIX)
            s.connect(p.path)
            s.sendall(req)
            out = b""
            while True:
                x = s.recv(4096)
                if not x:
                    break
                out += x
            s.close()
            return out
        assert ask(b"CONNECT 192.168.1.1:443 HTTP/1.1\r\nHost: x\r\n\r\n").startswith(
            b"HTTP/1.1 403")
        assert b"local network" in ask(b"GET http://127.0.0.1:8000/v1/models HTTP/1.1\r\n"
                                       b"Host: 127.0.0.1\r\n\r\n")
        mode["m"] = "off"
        assert b"network is off" in ask(b"CONNECT example.org:443 HTTP/1.1\r\n\r\n")
        assert p.stats["refused"] == 3
    finally:
        p.close()


# ---------------------------------------------------------------------- the memory guard
class FakeEngineAPI:
    def __init__(self):
        self.calls = []

    def call(self, method, path, body=None, timeout=60):
        self.calls.append((method, path))
        if path.startswith("/images/"):
            return 200, {}
        if path.endswith("/json") and "containers" in path:
            return 404, None
        if path.startswith("/containers/create"):
            return 201, {"Id": "ctr-" + str(len(self.calls))}
        if path.endswith("/start"):
            return 204, None
        if path.endswith("/exec"):
            return 201, {"Id": "e"}
        return 204, None

    def hijack(self, path, body):
        a, b = socket.socketpair()
        b.close()
        return a, b""


def guarded(avail_gb, release=None):
    """Sandboxes on the fake engine API with MemAvailable from `avail_gb["v"]` (GiB)."""
    clock = {"t": 1000.0}
    sb = sb_mod.Sandboxes(tempfile.mkdtemp(), engine=FakeEngineAPI(), meminfo_fn=lambda: {
        "MemAvailable": int(avail_gb["v"] * sb_mod.GiB)}, clock=lambda: clock["t"],
        release=release)
    sb._proxy = lambda b: None
    sb.run = lambda *a, **k: (0, False)
    sb.wait_s, sb.poll_s, sb.release_every_s = 0.4, 0.05, 0.0
    return sb, clock


def test_the_memory_guard_admits_stops_and_kills_by_the_numbers():
    mem = {"v": 60}
    sb, clock = guarded(mem)
    # the tests pin 1 GiB (tests/chatd_util.py); the default is 4 (the next test)
    assert sb_mod.LIMIT_BYTES == 1 * sb_mod.GiB and sb_mod.FLOOR_GB == 4.0
    assert sb.max_running == 4 and sb.total_gb == 4.0, "the cap: max_running times 1 GiB"
    sb.max_running = 1
    assert sb.admit() is None
    mem["v"] = 20.4                                       # chat 01M4ZPKR...: waited 21 minutes
    assert sb.admit() is None
    mem["v"] = 4.9                                        # 4.9 - 1 = 3.9 < 4
    assert sb.admit() == ("MemAvailable 4.9 GiB: a 1 GiB sandbox would leave the engine under "
                          "its 4 GiB floor")
    mem["v"] = 5.0
    assert sb.admit() is None
    sb.ensure("c1")
    assert sb.box("c1").state == "running"
    mem["v"] = 4.0                                        # the sandbox took its 1 GiB
    assert sb.guard_once() is None, "the guard leaves alone what the admission let in"
    mem["v"] = 60
    sb.ensure("c2")                                       # the cap: the idle c1 makes room
    assert sb.box("c1").state == "stopped" and sb.box("c2").state == "running"
    assert sb.box("c1").stopped_reason == "idle"
    mem["v"] = 2.9
    assert sb.guard_once() == "stop c2" and sb.box("c2").stopped_reason == "memory_guard"
    mem["v"] = 60
    sb.ensure("c3")
    mem["v"] = 1.9
    assert sb.guard_once() == "kill" and sb.box("c3").state == "stopped"
    mem["v"] = 60
    sb.ensure("c4")
    clock["t"] += 31 * 60
    assert sb.guard_once() == "idle c4"
    cancel = threading.Event()
    mem["v"] = 3
    sb.wait_s = 30
    threading.Timer(0.3, cancel.set).start()
    try:
        sb.ensure("c5", cancel=cancel)
        raise AssertionError("must wait, then give up when stopped")
    except sb_mod.Unavailable as exc:
        assert sb.box("c5").state == "stopped" and str(exc) == "stopped while waiting for memory"


def test_the_old_start_line_is_gone():
    os.environ["QSE_SANDBOX_MIN_START_GB"] = "30"
    try:
        sb, _ = guarded({"v": 20.4})
        assert sb.admit() is None and not hasattr(sb, "min_start_gb")
        sb.ensure("c1")
        assert sb.box("c1").state == "running"
    finally:
        del os.environ["QSE_SANDBOX_MIN_START_GB"]


def test_a_wait_for_memory_ends_after_wait_s_with_a_reason_the_model_and_the_chat_read():
    mem = {"v": 3.2}
    sb, _ = guarded(mem, release=lambda n: 0)
    seen, changes = [], []
    sb.on_change = lambda cid, st: changes.append(st)
    for cancel in (threading.Event(), None):              # a tool call; a terminal or MCP pipe
        t0 = time.monotonic()
        try:
            sb.ensure("c1", cancel=cancel, on_wait=seen.append)
            raise AssertionError("must give up")
        except sb_mod.Unavailable as exc:
            took = time.monotonic() - t0
            msg, data = str(exc), exc.data["sandbox"]
        assert 0.4 <= took < 2.0, took
        assert msg.startswith("no sandbox for this call: after waiting 0 s it still could not "
                              "start (MemAvailable 3.2 GiB: a 1 GiB sandbox would leave the "
                              "engine under its 4 GiB floor; the engine gave back 0.0 GiB)")
        assert "The command did not run" in msg and "answer with what you already have, or " \
            "without tools" in msg and "lack of memory" in msg
        assert data == {"reason": "no_memory", "why": "MemAvailable 3.2 GiB: a 1 GiB sandbox "
                        "would leave the engine under its 4 GiB floor", "waited_s": 0,
                        "mem_available_gib": 3.2, "limit_gib": 1.0, "floor_gib": 4.0,
                        "engine_released_gib": 0.0}
        b = sb.box("c1")
        assert b.state == "stopped" and b.error.startswith("did not start after 0 s: MemAvailable")
        assert changes[-1]["state"] == "stopped" and changes[-1]["error"] == b.error
    assert seen[0].endswith("; 0 of 0 s") and seen[0].startswith("MemAvailable 3.2 GiB")
    mem["v"] = 30                                         # memory back: the next call starts it
    sb.ensure("c1")
    assert sb.box("c1").state == "running" and sb.box("c1").error is None


def test_busy_sandboxes_of_other_chats_end_the_wait_with_their_own_reason():
    sb, _ = guarded({"v": 60})
    sb.max_running = 1
    sb.ensure("c1")
    sb.box("c1").busy = 1
    try:
        sb.ensure("c2")
        raise AssertionError("must give up")
    except sb_mod.Unavailable as exc:
        assert exc.data["sandbox"]["reason"] == "busy"
        assert "1 sandbox(es) of other chats are running commands" in str(exc)
        assert "other chats' sandboxes are busy" in str(exc) and "engine gave" not in str(exc)


def test_short_of_memory_the_engine_gives_way_before_the_sandbox_waits():
    mem = {"v": 3.0}
    asked = []

    def release(n):
        asked.append(n)
        mem["v"] += n / sb_mod.GiB                        # the engine gave back all it was asked
        return n
    sb, _ = guarded(mem, release=release)
    waits = []
    sb.ensure("c1", on_wait=waits.append)
    assert asked == [2 * sb_mod.GiB], "the floor plus the limit less MemAvailable: 4 + 1 - 3"
    assert sb.box("c1").state == "running" and waits == [] and sb.released == 2 * sb_mod.GiB

    # an engine that can give back only part: the sandbox waits, the reason names what it gave
    mem["v"], asked[:] = 3.0, []
    sb2, _ = guarded(mem, release=lambda n: asked.append(n) or (mem.__setitem__(
        "v", mem["v"] + 0.5) or int(0.5 * sb_mod.GiB)))
    sb2.release_every_s = 60.0                            # asked once, not every look
    try:
        sb2.ensure("c2")
        raise AssertionError("must give up")
    except sb_mod.Unavailable as exc:
        assert len(asked) == 1 and "the engine gave back 0.5 GiB" in str(exc)
        assert exc.data["sandbox"]["engine_released_gib"] == 0.5

    # an engine without the route (0) or down (an exception) changes nothing but the wait
    for rel in (lambda n: 0, lambda n: 1 / 0):
        sb3, _ = guarded({"v": 3.0}, release=rel)
        assert sb3.admit().startswith("MemAvailable 3.0 GiB")


def test_the_guard_asks_the_engine_before_it_stops_a_sandbox():
    mem = {"v": 60}
    sb, _ = guarded(mem, release=lambda n: mem.__setitem__("v", mem["v"] + n / sb_mod.GiB) or n)
    sb.ensure("c1")
    mem["v"] = 2.5
    assert sb.guard_once() == "engine released" and sb.box("c1").state == "running"
    sb.release = lambda n: 0
    mem["v"] = 2.5
    assert sb.guard_once() == "stop c1"


def test_the_memory_limit_follows_qse_sandbox_mem_gb():
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", "from chatd import sandbox as m; "
                          "print(m.LIMIT_BYTES, m.limit_words(), m.FLOOR_GB, m.WAIT_S)"],
                         capture_output=True, text=True, cwd=root, check=True, env=dict(
                             os.environ, PYTHONPATH=root, QSE_SANDBOX_MEM_GB="2",
                             QSE_SANDBOX_FLOOR_GB="6", QSE_SANDBOX_WAIT_S="30")).stdout.split()
    assert out == [str(2 * sb_mod.GiB), "2", "GiB", "6.0", "30.0"], out
    old = sb_mod.LIMIT_BYTES
    sb_mod.LIMIT_BYTES = 2 * sb_mod.GiB
    try:
        api = FakeEngineAPI()
        bodies = []
        call = api.call
        api.call = lambda method, path, body=None, timeout=60: (bodies.append(body),
                                                                call(method, path, body))[1]
        sb = sb_mod.Sandboxes(tempfile.mkdtemp(), engine=api, meminfo_fn=lambda: {})
        sb._proxy = lambda b: None
        sb.run = lambda *a, **k: (0, False)
        assert sb.total_gb == 8.0
        sb.ensure("c1")
        hc = next(x for x in bodies if x and "HostConfig" in x)["HostConfig"]
        assert hc["Memory"] == hc["MemorySwap"] == 2 * sb_mod.GiB
        assert sb.status("c1")["memory_limit_bytes"] == 2 * sb_mod.GiB
        assert T.oom_words() == "out of memory (2 GiB sandbox limit)"
    finally:
        sb_mod.LIMIT_BYTES = old


def test_the_default_limit_is_4_gib_and_an_older_container_takes_it_at_its_start():
    """2026-10-17: headless Chromium drawing a Three.js app in software went past the 1 GiB
    default, and render_page said "out of memory (1 GiB sandbox limit), the kernel killed
    Chromium" on the app it was there to check."""
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if not k.startswith("QSE_SANDBOX_")}
    out = subprocess.run([sys.executable, "-c", "from chatd import sandbox as m; "
                          "s = m.Sandboxes('/tmp', engine=None, meminfo_fn=lambda: {}); "
                          "print(m.LIMIT_BYTES, m.limit_words(), s.total_gb)"],
                         capture_output=True, text=True, cwd=root, check=True,
                         env=dict(env, PYTHONPATH=root)).stdout.split()
    assert out == [str(4 * sb_mod.GiB), "4", "GiB", "16.0"], out

    class Old(FakeEngineAPI):
        def call(self, method, path, body=None, timeout=60):
            self.calls.append((method, path, body))
            if path.endswith("/json"):
                return 200, {"Id": "old1", "State": {"Running": False},
                             "Config": {"Labels": {sb_mod.LABEL: "c1",
                                                   sb_mod.INSTANCE_LABEL: inst}},
                             "HostConfig": {"Memory": 2 * sb_mod.GiB}}
            return 204, None
    api = Old()
    sb = sb_mod.Sandboxes(tempfile.mkdtemp(), engine=api, meminfo_fn=lambda: {})
    inst = sb.instance
    sb._proxy = lambda b: None
    sb.run = lambda *a, **k: (0, False)
    sb.ensure("c1")
    upd = [c for c in api.calls if c[1] == "/containers/old1/update"]
    assert upd == [("POST", "/containers/old1/update", {"Memory": sb_mod.LIMIT_BYTES,
                                                       "MemorySwap": sb_mod.LIMIT_BYTES})]
    assert api.calls.index(upd[0]) < api.calls.index(("POST", "/containers/old1/start", None))
    # a container made under the current limit is started as it is
    sb2 = sb_mod.Sandboxes(tempfile.mkdtemp(), engine=FakeEngineAPI(), meminfo_fn=lambda: {})
    sb2._proxy = lambda b: None
    sb2.run = lambda *a, **k: (0, False)
    sb2.ensure("c2")
    assert not [c for c in sb2.engine.calls if c[1].endswith("/update")]


def test_oom_kills_are_read_from_the_cgroup():
    d = tempfile.mkdtemp()
    sb, _ = guarded({"v": 60})
    old = sb_mod.CGROUP
    sb_mod.CGROUP = d
    try:
        sb.ensure("c1")
        ctr = sb.box("c1").container
        assert sb.oom_kills("c1") is None                 # nothing to read
        os.makedirs(os.path.join(d, "system.slice", f"docker-{ctr}.scope"))
        with open(os.path.join(d, "system.slice", f"docker-{ctr}.scope", "memory.events"),
                  "w") as fh:
            fh.write("low 0\nhigh 0\nmax 12\noom 2\noom_kill 2\noom_group_kill 0\n")
        assert sb.oom_kills("c1") == 2
        assert sb.oom_kills("nobox") is None
    finally:
        sb_mod.CGROUP = old


def test_a_command_killed_at_the_limit_says_out_of_memory():
    c = chat_with_sandbox()
    try:
        boxes = c.svc.sandboxes
        kills = {"n": 0}
        boxes.oom_kills = lambda cid: kills["n"]
        boxes.script["ffmpeg"] = (["frame 1\n"], 137, 0.0)
        real_run = boxes.run

        def run(cid, cmd, **kw):
            if "ffmpeg" in cmd[-1]:
                kills["n"] += 1                           # the kernel killed it at the limit
            return real_run(cid, cmd, **kw)
        boxes.run = run
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        c.send(cid, calls(("bash", {"command": "ffmpeg -i in.mp4 out.mp4"})))
        s = c.wait_idle(cid, 20)
        call = next(p for p in parts_of(s) if p["type"] == "tool_call")
        assert call["status"] == "failed" and call["output"]["exit_code"] == 137
        assert "[out of memory (1 GiB sandbox limit): the kernel killed a process of this " \
               "command (exit code 137)" in call["output"]["content"]
        assert call["output"]["data"] == {"sandbox": {"reason": "oom", "limit_gib": 1.0}}
        tool_msg = [m for m in requests(c)[-1]["messages"] if m["role"] == "tool"][-1]
        assert "out of memory (1 GiB sandbox limit)" in tool_msg["content"]

        # ffmpeg killed under a pipe: its writer exits 1, the cgroup still counts the kill
        boxes.script["render.py"] = (["BrokenPipeError: [Errno 32] Broken pipe\n"], 1, 0.0)
        real = boxes.run
        boxes.run = lambda cid, cmd, **kw: (kills.__setitem__("n", kills["n"] + (
            "render.py" in cmd[-1])), real(cid, cmd, **kw))[1]
        c.send(cid, calls(("bash", {"command": "python3 render.py"})))
        s = c.wait_idle(cid, 20)
        call = [p for p in parts_of(s) if p["type"] == "tool_call"][-1]
        assert call["output"]["exit_code"] == 1 and "out of memory (1 GiB sandbox limit)" in \
            call["output"]["content"] and "-threads 4" in call["output"]["content"]
        # a plain failure is not called out of memory; nor is a 137 the cgroup does not count
        boxes.script["false"] = ([], 1, 0.0)
        boxes.script["kill9"] = ([], 137, 0.0)
        for cmd in ("false", "kill9"):
            c.send(cid, calls(("bash", {"command": cmd})))
            s = c.wait_idle(cid, 20)
            call = [p for p in parts_of(s) if p["type"] == "tool_call"][-1]
            assert "out of memory" not in call["output"]["content"], cmd
        # where the cgroup cannot be read, a SIGKILL chatd did not send counts as one: 137 from
        # bash, 9 from setsid when the killed process was the command itself
        boxes.oom_kills = lambda cid: None
        boxes.script["setsid9"] = ([], 9, 0.0)
        for cmd in ("kill9 again", "setsid9"):
            c.send(cid, calls(("bash", {"command": cmd})))
            s = c.wait_idle(cid, 20)
            call = [p for p in parts_of(s) if p["type"] == "tool_call"][-1]
            assert "out of memory (1 GiB sandbox limit)" in call["output"]["content"], cmd
    finally:
        c.close()


def test_render_page_killed_at_the_limit_says_out_of_memory():
    ws = Workspace(tempfile.mkdtemp())
    ws.write("index.html", b"<p>hi</p>")

    class Ctx:
        call_id, oom = "call1", True

        def __init__(self):
            self.ws = ws

        def exec(self, cmd, timeout=600, quiet=False):
            return 9, False

        def output_text(self):
            return ""
    try:
        T._render(Ctx(), {"path": "index.html"})
        raise AssertionError("no screenshot: an error")
    except T.ToolError as exc:
        assert str(exc).startswith("the page did not render: out of memory (1 GiB sandbox "
                                   "limit), the kernel killed Chromium")
        assert exc.data == {"sandbox": {"reason": "oom", "limit_gib": 1.0}}
    ctx = Ctx()
    ctx.oom = False
    try:
        T._render(ctx, {"path": "index.html"})
    except T.ToolError as exc:
        assert str(exc) == "the page did not render: no screenshot" and exc.data is None


def test_render_page_on_a_port_dials_localhost_and_says_what_listens():
    """Live (2026-10-16): Vite on `localhost` listened on [::1]:5173 only, Chromium sent to
    127.0.0.1 was refused and the model read "Connection refused" as a network restriction."""
    ws = Workspace(tempfile.mkdtemp())
    listening = [{"port": 5174, "addresses": ["::1"], "command": "node /w/node_modules/.bin/vite"},
                 {"port": 8001, "addresses": ["127.0.0.1", "::"], "command": ""}]

    class Ctx:
        call_id, oom = "call1", False

        def __init__(self):
            self.ws, self.cmds, self.out = ws, [], ""

        def exec(self, cmd, timeout=600, quiet=False):
            self.cmds.append(cmd)
            if cmd[-1] == "ports":
                assert cmd[:2] == ["python3", "-c"] and "def listening(" in cmd[2]
                self.out += json.dumps(listening) + "\n"
            elif "chromium" in cmd[-1]:
                ws.write(".tandem/shots/call1.png", b"\x89PNG fake")
            return 0, False

        def output_text(self):
            return self.out
    ctx = Ctx()
    try:
        T._render(ctx, {"port": 5173})
        raise AssertionError("nothing on 5173: an error")
    except T.ToolError as exc:
        assert str(exc) == ("nothing listens on port 5173 in the sandbox, on any address; "
                            "listening now: 5174 on ::1 (node /w/node_modules/.bin/vite), "
                            "8001 on 127.0.0.1 and ::")
    assert not any("chromium" in c[-1] for c in ctx.cmds)        # no Chromium for a closed port
    ctx = Ctx()
    res = T._render(ctx, {"port": 5174, "path": "/app"})
    assert "http://localhost:5174/app" in ctx.cmds[-1][-1] and "127.0.0.1" not in ctx.cmds[-1][-1]
    assert res.content.startswith("rendered http://localhost:5174/app at 1280x800")
    listening.clear()
    try:
        T._render(Ctx(), {"port": 5173})
    except T.ToolError as exc:
        assert str(exc).endswith("; nothing listens on any port in the sandbox")


def test_render_page_reports_the_page_console_without_chromium_noise():
    """Live (2026-10-16, the pagoda chat): every render_page result said "console:" over eight
    lines of Chromium's own pages (chrome://, "Slow network is detected"), GCM registration errors
    and the deprecated WebGL fallback, and no line of the page's; the model sifted them on each
    of its renders."""
    import subprocess
    ws = Workspace(tempfile.mkdtemp())
    ws.write("index.html", b"<canvas></canvas>")
    chromium_log = "\n".join([
        '[86:86:1016/215118.672506:INFO:CONSOLE:422] "Slow network is detected. See https://www.'
        'chromestatus.com/feature/5636954674692096 for more details.", source: chrome://resources/'
        'cr_components/composebox/composebox_input.js (422)',
        '[86:86:1016/215118.764582:INFO:CONSOLE:0] "Slow network is detected.", source: '
        'chrome://omnibox-popup.top-chrome/ (0)',
        '[86:116:1016/215120.901110:ERROR:google_apis/gcm/engine/registration_request.cc:291] '
        'Registration response error message: PHONE_REGISTRATION_ERROR',
        '[86:86:1016/215123.040540:INFO:CONSOLE:0] "[]Automatic fallback to software WebGL has been '
        'deprecated. Please use the --enable-unsafe-swiftshader flag", source: http://localhost:'
        '8124/ (0)',
        '[634:634:1016/215214.160964:INFO:CONSOLE:61] "Uncaught TypeError: THREE.MathUtils.'
        'lerpAngle is not a function", source: http://localhost:8124/js/scene/koi.js (61)',
        '[634:634:1016/215214.3:INFO:CONSOLE:0] "[.WebGL-0x2400177600]GL Driver Message (OpenGL, '
        'Performance, GL_CLOSE_PATH_NV, High): GPU stall due to ReadPixels", source: http://'
        'localhost:8124/ (0)',
        '[634:640:1016/215214.2:ERROR:dbus/bus.cc:408] Failed to connect to the bus']) + "\n"

    class Ctx:
        call_id, oom = "call1", False

        def __init__(self, log):
            self.ws, self.cmds, self.out, self.log = ws, [], "", log

        def exec(self, cmd, timeout=600, quiet=False):
            self.cmds.append(cmd)
            if "chromium" in cmd[-1]:
                ws.write(".tandem/shots/call1.png", b"\x89PNG fake")
                # the command's own filter over Chromium's stderr, as bash runs it in the sandbox
                pipe = cmd[-1].split(" 2>&1 | ", 1)[1]
                self.out += subprocess.run(["bash", "-c", pipe], input=self.log, text=True,
                                           capture_output=True).stdout
            return 0, False

        def output_text(self):
            return self.out
    ctx = Ctx(chromium_log)
    res = T._render(ctx, {"path": "index.html"})
    assert "--use-angle=swiftshader --enable-unsafe-swiftshader" in ctx.cmds[-1][-1]
    assert res.content.split("\n", 2)[1] == "WebGPU: not available in the check browser (WebGL2 " \
        "only): a page with a WebGPU path was not checked on it"
    assert res.content.split("\n", 2)[2] == (
        'console:\n"Uncaught TypeError: THREE.MathUtils.lerpAngle is not a function", source: '
        'http://localhost:8124/js/scene/koi.js (61)')
    # only Chromium's lines: the page logged nothing
    res = T._render(Ctx(chromium_log.split("[634:634")[0]), {"path": "index.html"})
    assert res.content.endswith("\nno console errors")


def test_render_page_on_a_port_takes_a_file_path_below_the_server_and_says_an_error_page():
    """Live (2026-10-17, pagoda replay 2): render_page {"port": 8080, "path":
    "/workspace/_test.html"} on `python3 -m http.server` run from /workspace loaded
    http://localhost:8080/workspace/_test.html, the server's 404 page, and said "no console
    errors"."""
    ws = Workspace(tempfile.mkdtemp())
    listening = [{"port": 8080, "addresses": ["0.0.0.0"], "command": "python3 -m http.server 8080",
                  "cwd": "/workspace"}]

    class Ctx:
        call_id, oom = "call1", False

        def __init__(self, status):
            self.ws, self.cmds, self.out, self.status = ws, [], "", status

        def exec(self, cmd, timeout=600, quiet=False):
            self.cmds.append(cmd)
            if cmd[-1] == "ports":
                self.out += json.dumps(listening) + "\n"
            elif "chromium" in cmd[-1]:
                ws.write(".tandem/shots/call1.png", b"\x89PNG fake")
                self.out += f"HTTP_STATUS {self.status}\n"
            return 0, False

        def output_text(self):
            return self.out
    ctx = Ctx("200")
    res = T._render(ctx, {"port": 8080, "path": "/workspace/_test.html"})
    assert "http://localhost:8080/_test.html" in ctx.cmds[-1][-1]
    assert "curl -s -o /dev/null -m 10 -w '%{http_code}' http://localhost:8080/_test.html" in \
        ctx.cmds[-1][-1]
    assert res.content.endswith("\nno console errors") and "HTTP" not in res.content
    ctx = Ctx("404")
    res = T._render(ctx, {"port": 8080, "path": "/elsewhere/x.html"})
    assert "http://localhost:8080/elsewhere/x.html" in ctx.cmds[-1][-1]
    assert "the server answered HTTP 404 for http://localhost:8080/elsewhere/x.html: the " \
           "screenshot is its error page, not the app" in res.content
    # a file render asks no server
    ws.write("index.html", b"<p>hi</p>")
    ctx = Ctx("200")
    T._render(ctx, {"path": "index.html"})
    assert "curl" not in ctx.cmds[-1][-1]


def test_render_page_reports_a_failed_module_load_through_the_kit():
    """Live (2026-10-17, pagoda replay 3): js/main.js imported "./util.js", which answered 404;
    the app stayed on "Loading garden..." and Chromium's screenshot mode logged nothing, so
    render_page said "no console errors" six times over the same picture. The kit's page check
    (Chrome under DevTools) sees the failed load."""
    ws = Workspace(tempfile.mkdtemp())
    listening = [{"port": 5173, "addresses": ["0.0.0.0"], "command": "python3 -m http.server",
                  "cwd": "/workspace"}]
    rep = {"url": "http://localhost:5173/", "ok": False, "verdict": "FAIL: 1 console error(s)",
           "errors": ["network: Failed to load resource: the server responded with a status of "
                      "404 (File not found) http://localhost:5173/js/util.js"],
           "steps": [], "widths": [{"width": 1280, "text": "Loading garden...",
                                     "scroll_width": 1280}]}

    class Ctx:
        call_id, oom = "call1", False

        def __init__(self):
            self.ws, self.cmds, self.out = ws, [], ""

        def exec(self, cmd, timeout=600, quiet=False):
            self.cmds.append(cmd)
            if cmd[-1] == "ports":
                self.out += json.dumps(listening) + "\n"
            elif "laya_drive.py page" in cmd[-1]:
                ws.write(".tandem/shots/call1/page-1280.png", b"\x89PNG fake")
                self.out += json.dumps(rep, indent=1) + "\n"
            return 0, False

        def output_text(self):
            return self.out
    ctx = Ctx()
    res = T._render(ctx, {"port": 5173})
    assert not any("chromium" in c[-1] for c in ctx.cmds)
    kit = ctx.cmds[-1][-1]
    assert "python3 .tandem/laya_drive.py page http://localhost:5173/ --no-laya --widths 1280 " \
           "--shots /workspace/.tandem/shots/call1 --timeout 60" in kit
    assert ws.read(".tandem/laya_drive.py").startswith(b"#!/usr/bin/env python3")
    assert res.content.startswith("rendered http://localhost:5173/ at 1280x900 just now")
    assert 'visible text: "Loading garden..."' in res.content
    assert res.content.endswith("console errors, exceptions and failed loads:\nnetwork: Failed to "
                                "load resource: the server responded with a status of 404 (File "
                                "not found) http://localhost:5173/js/util.js")
    assert res.images and res.pane["ref"] == "file:/workspace/.tandem/shots/call1/page-1280.png"
    rep["errors"], rep["widths"][0]["text"] = [], ""
    assert T._render(Ctx(), {"port": 5173}).content.endswith(
        "no console errors and no failed loads")


def test_render_page_says_which_graphics_api_the_check_browser_has():
    """Live (2026-10-17, chat 01M556JVWDQVDCAMBMVFZCT7XF): a WebGPU ocean with a WebGL2 fallback
    was checked on its fallback only ("WebGL2" in every screenshot) while Safari ran its WebGPU
    path and drew black; render_page said "no console errors". The kit reads the page's graphics
    in the same render, and the result says what was not checked."""
    import shlex
    ws = Workspace(tempfile.mkdtemp())
    ws.write("index.html", b"<canvas></canvas>")

    class Ctx:
        call_id, oom = "call1", False

        def __init__(self, probe):
            self.ws, self.cmds, self.out, self.probe = ws, [], "", probe

        def exec(self, cmd, timeout=600, quiet=False):
            self.cmds.append(cmd)
            if "laya_drive.py page" in cmd[-1]:
                ws.write(".tandem/shots/call1/page-1280.png", b"\x89PNG fake")
                step = shlex.split(cmd[-1])[-1]
                self.out += json.dumps({"errors": [], "widths": [{"text": ""}], "steps": [
                    {"step": step, "ok": True, "value": self.probe}]}) + "\n"
            return 0, False

        def output_text(self):
            return self.out
    ctx = Ctx("no adapter true")
    res = T._render(ctx, {"path": "index.html"})
    assert shlex.split(ctx.cmds[-1][-1])[-2:] == ["--step", "read:" + T.GRAPHICS_PROBE]
    assert res.content.endswith("WebGPU: not available in the check browser (WebGL2 only): a "
                                "page with a WebGPU path was not checked on it\nno console errors "
                                "and no failed loads"), res.content
    assert "\nWebGPU and WebGL2 in the check browser\n" in T._render(
        Ctx("adapter true"), {"path": "index.html"}).content
    assert "(no WebGL2 either)" in T._render(Ctx("none false"), {"path": "index.html"}).content


def test_render_page_quotes_the_line_an_error_names():
    """Live (2026-10-17, pagoda replay 3): "Cannot read properties of undefined (reading 'set')"
    at pond.js (62) reached the model 13 times and it edited main.js for 20 minutes; line 62 was
    `petal.position.set(...)` on a geometry. The result now shows that line."""
    ws = Workspace(tempfile.mkdtemp())
    ws.write("js/scene/pond.js", ("\n" * 61 + "      petal.position.set(1, 2, 3);\n").encode())
    listening = [{"port": 5174, "addresses": ["0.0.0.0"], "command": "python3 -m http.server",
                  "cwd": "/workspace"}]
    exc = ("exception: TypeError: Cannot read properties of undefined (reading 'set') "
           "(http://localhost:5174/js/scene/pond.js:62)")

    class Ctx:
        call_id, oom = "call1", False

        def __init__(self, kit):
            self.ws, self.cmds, self.out, self.kit = ws, [], "", kit

        def exec(self, cmd, timeout=600, quiet=False):
            self.cmds.append(cmd)
            if cmd[-1] == "ports":
                self.out += json.dumps(listening) + "\n"
            elif "laya_drive.py page" in cmd[-1] and self.kit:
                ws.write(".tandem/shots/call1/page-1280.png", b"\x89PNG fake")
                self.out += json.dumps({"errors": [exc], "widths": [{"text": ""}]}) + "\n"
            elif "chromium" in cmd[-1]:
                ws.write(".tandem/shots/call1.png", b"\x89PNG fake")
                self.out += ('[1:1:1017/045852.1:INFO:CONSOLE:62] "Uncaught TypeError: Cannot '
                             'read properties of undefined (reading \'set\')", source: '
                             'http://localhost:5174/js/scene/pond.js (62)\n')
            return 0, False

        def output_text(self):
            return self.out
    quoted = "    /workspace/js/scene/pond.js:62: petal.position.set(1, 2, 3);"
    res = T._render(Ctx(True), {"port": 5174})
    assert res.content.endswith(exc + "\n" + quoted), res.content
    res = T._render(Ctx(False), {"port": 5174})            # no report from the kit: Chromium
    assert res.content.endswith("(reading 'set')\", source: http://localhost:5174/js/scene/"
                                "pond.js (62)\n" + quoted), res.content
    # another port's script, or a line past the end, is not quoted
    assert T.quote_lines(Ctx(True), [exc.replace("5174", "9999")], 5174, "/workspace") == \
        [exc.replace("5174", "9999")]
    assert T.quote_lines(Ctx(True), [exc.replace(":62", ":99")], 5174, "/workspace") == \
        [exc.replace(":62", ":99")]


def test_the_relay_dials_localhost_on_both_families_and_lists_what_listens():
    """ops/sandbox/relay.py as chatd sends it (python3 -c): a server on ::1 only and one on
    127.0.0.1 only both answer through `connect`; a closed port says what listens instead."""
    import subprocess
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = f"hi {self.server.server_address[0]}".encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class H6(ThreadingHTTPServer):
        address_family = socket.AF_INET6
    servers = [ThreadingHTTPServer(("127.0.0.1", 0), H)]
    try:
        servers.append(H6(("::1", 0), H))
    except OSError:
        print("no IPv6 loopback here: the ::1 half is skipped")
    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    def relay(*args, stdin=b""):
        return subprocess.run(sb_mod.relay_cmd(*args), input=stdin, capture_output=True,
                              timeout=30).stdout
    try:
        for srv in servers:
            got = relay("connect", str(srv.server_address[1]),
                        stdin=b"GET / HTTP/1.0\r\nHost: x\r\n\r\n")
            assert got.startswith(b"HTTP/1.0 200") and got.endswith(
                f"hi {srv.server_address[0]}".encode()), got
        free = socket.socket()
        free.bind(("127.0.0.1", 0))
        closed = free.getsockname()[1]
        free.close()
        got = relay("connect", str(closed)).decode()
        assert got.startswith("HTTP/1.1 502") and \
            f"Nothing listens on port {closed} in the sandbox (dialed localhost: ::1 and " \
            "127.0.0.1)" in got, got
    finally:
        for srv in servers:
            srv.shutdown()
            srv.server_close()


def test_the_relay_reads_the_listening_ports_from_proc():
    import importlib.util
    spec = importlib.util.spec_from_file_location("relay", sb_mod.RELAY)
    relay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relay)
    proc = tempfile.mkdtemp()
    os.makedirs(os.path.join(proc, "net"))
    head = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  " \
           "timeout inode\n"

    def row(addr, port, st, ino):
        return f"   0: {addr}:{port:04X} 00000000:0000 {st} 0:0 00:0 0 1000 0 {ino} 1 0 100 0 0\n"
    with open(os.path.join(proc, "net", "tcp"), "w") as fh:
        fh.write(head + row("0100007F", 3128, "0A", 11) + row("00000000", 8000, "0A", 12) +
                 row("0100007F", 40000, "01", 13))                 # 01: a connection, not a listener
    with open(os.path.join(proc, "net", "tcp6"), "w") as fh:
        fh.write(head + row("00000000000000000000000001000000", 5173, "0A", 21) +
                 row("0" * 32, 8000, "0A", 22))
    for pid, cmd, inos in (("1", b"python3\0/opt/tandem/relay.py\0serve\0", [11]),
                           ("40", b"node\0/w/node_modules/.bin/vite\0", [21]),
                           ("41", b"python3\0-m\0http.server\0", [12, 22])):
        os.makedirs(os.path.join(proc, pid, "fd"))
        with open(os.path.join(proc, pid, "cmdline"), "wb") as fh:
            fh.write(cmd)
        for i, ino in enumerate(inos):
            os.symlink(f"socket:[{ino}]", os.path.join(proc, pid, "fd", str(i + 3)))
    # the vite process: its working directory and its start (field 22 of stat, after the name)
    os.symlink("/workspace/pagoda-garden", os.path.join(proc, "40", "cwd"))
    with open(os.path.join(proc, "40", "stat"), "w") as fh:
        fh.write("40 (node (vite)) S 1 40 40 0 -1 4194560 " + "0 " * 12 + "98765 1 2\n")
    assert relay.listening(proc) == [
        {"port": 5173, "addresses": ["::1"], "command": "node /w/node_modules/.bin/vite",
         "cwd": "/workspace/pagoda-garden", "started": 98765},
        {"port": 8000, "addresses": ["0.0.0.0", "::"], "command": "python3 -m http.server",
         "cwd": "", "started": 0}]
    assert relay.ports_words([]) == "nothing listens on any port in the sandbox"


def test_the_relay_probes_a_page_and_its_scripts_and_style_sheets():
    """`relay.py probe PORT PATH`: the page's status, type and title, and each script and style
    sheet on the same server with its own status (a 404 module is named); another host is left
    out."""
    import importlib.util
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    spec = importlib.util.spec_from_file_location("relay", sb_mod.RELAY)
    relay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relay)
    page = (b'<!doctype html><html><head><title> Pagoda\n Garden </title>'
            b'<link rel="stylesheet" crossorigin href="/assets/a.css">'
            b'<link rel=icon href=/favicon.ico><script src="https://cdn.example/x.js"></script>'
            b'<script type="module" src="/src/main.js"></script>'
            b"<script type='module' src='missing.js'></script></head></html>")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            known = {"/app/": (page, "text/html"), "/assets/a.css": (b"b{}", "text/css"),
                     "/src/main.js": (b"1", "text/javascript")}
            body, ctype = known.get(self.path, (b"no", "text/plain"))
            self.send_response(200 if self.path in known else 404)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        got = relay.probe(port, "/app/")
    finally:
        srv.shutdown()
        srv.server_close()
    assert got["status"] == 200 and got["type"] == "text/html" and got["title"] == "Pagoda Garden"
    assert got["assets"] == [
        {"url": "/assets/a.css", "status": 200, "type": "text/css"},
        {"url": "/src/main.js", "status": 200, "type": "text/javascript"},
        {"url": "missing.js", "status": 404, "type": "text/plain"}]
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    closed = free.getsockname()[1]
    free.close()
    down = relay.probe(closed)
    assert down["status"] == 0 and "nothing listens" in down["type"] and down["assets"] == []


def test_a_sandbox_that_never_starts_fails_the_call_with_the_reason_and_the_run_goes_on():
    c = chat_with_sandbox()
    try:
        boxes = c.svc.sandboxes
        boxes.meminfo = lambda: {"MemAvailable": int(3.2 * sb_mod.GiB)}
        boxes.wait_s, boxes.poll_s = 0.5, 0.05
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        c.send(cid, calls(("bash", {"command": "make"})))
        s = c.wait_idle(cid, 20)
        call = next(p for p in parts_of(s) if p["type"] == "tool_call")
        assert call["status"] == "failed" and call["output"]["is_error"]
        assert call["output"]["content"].startswith("no sandbox for this call: after waiting")
        assert call["output"]["data"]["sandbox"]["reason"] == "no_memory"
        tool_msg = [m for m in requests(c)[-1]["messages"] if m["role"] == "tool"][-1]
        assert "answer with what you already have" in tool_msg["content"]
        assert s["messages"][-1]["status"] == "done", "the model answers after the failed call"
        assert boxes.ran == [], "nothing ran"
        sbx = c.ok("GET", f"/conversations/{cid}/sandbox")
        assert sbx["state"] == "stopped" and sbx["error"].startswith("did not start after")
        assert sbx["memory_limit_bytes"] == 1 * sb_mod.GiB
    finally:
        c.close()


# ---------------------------------------------------------------------- terminals
def ws_open(port, path):
    s = socket.create_connection(("127.0.0.1", port))
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
              f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nX-Tandem-Forward: {KEY}"
              f"\r\n\r\n".encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += s.recv(1)
    assert head.startswith(b"HTTP/1.1 101"), head
    return s


def ws_send(s, data: bytes, op=2):
    mask = os.urandom(4)
    n = len(data)
    head = bytes([0x80 | op]) + (bytes([0x80 | n]) if n < 126 else
                                 bytes([0x80 | 126]) + n.to_bytes(2, "big"))
    s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))


def ws_recv(s, timeout=5.0) -> bytes:
    s.settimeout(timeout)
    out = b""
    try:
        while True:
            h = s.recv(2)
            if len(h) < 2:
                break
            n = h[1] & 0x7F
            if n == 126:
                n = int.from_bytes(s.recv(2), "big")
            data = b""
            while len(data) < n:
                data += s.recv(n - len(data))
            out += data
            s.settimeout(0.3)
    except socket.timeout:
        pass
    return out


def test_a_terminal_keeps_its_scrollback_and_a_second_attach_sees_it():
    c = chat_with_sandbox()
    try:
        cid = c.new()
        s1 = ws_open(c.port, f"/v1/chat-app/conversations/{cid}/terminals/shell/ws")
        assert b"sandbox:/workspace$" in ws_recv(s1)
        ws_send(s1, b"echo first\n")
        assert b"echo first" in ws_recv(s1)
        s1.close()
        s2 = ws_open(c.port, f"/v1/chat-app/conversations/{cid}/terminals/shell/ws")
        back = ws_recv(s2)
        assert b"sandbox:/workspace$" in back and b"echo first" in back
        ws_send(s2, json.dumps({"type": "resize", "cols": 100, "rows": 30}).encode(), op=1)
        s2.close()
        assert c.ok("GET", f"/conversations/{cid}/terminals")[0] == {
            "session": "shell", "title": "Terminal", "owner": "user", "alive": True}
    finally:
        c.close()


def test_an_agent_call_is_a_read_only_terminal():
    c = chat_with_sandbox()
    try:
        c.svc.sandboxes.script["make"] = (["building\n", "done\n"], 0, 0)
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        c.send(cid, calls(("bash", {"command": "make all"})))
        s = c.wait_idle(cid, 20)
        call = [p for p in parts_of(s) if p["type"] == "tool_call"][0]
        sock = ws_open(c.port, f"/v1/chat-app/conversations/{cid}/terminals/{call['id']}/ws")
        out = ws_recv(sock)
        assert b"$ make all" in out and b"building\r\ndone\r\n" in out
        sock.close()
        agent = [t for t in c.ok("GET", f"/conversations/{cid}/terminals")
                 if t["owner"] == "agent"]
        assert agent and agent[0]["session"] == call["id"]
    finally:
        c.close()


def test_present_opens_a_pane_on_every_device():
    c = Chat()
    try:
        cid = c.new()
        c.svc.workspace(cid).write("site/index.html", b"<h1>hi</h1>")
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("present", {"kind": "preview", "path": "site/index.html"})))
        ev = json.loads(sse.until(lambda f: f[1] == "pane.open")[-1][2])
        assert ev["pane"] == {"kind": "preview", "ref": "file:/workspace/site/index.html",
                              "title": None}
        sse.close()
        a = c.ok("GET", f"/conversations/{cid}/artifacts/file%3A%2Fworkspace%2Fsite%2Findex.html")
        assert a["kind"] == "preview" and a["url"].endswith("/sandbox/view/site/index.html")
    finally:
        c.close()


def test_an_empty_message_never_reaches_the_engine():
    """2026-10-12: a client sent `content` instead of `text`; chatd took it as an empty message
    and the engine got a user turn with nothing in it (and answered something unrelated)."""
    c = Chat()
    try:
        cid = c.new()
        st, err, _ = c.call("POST", f"/conversations/{cid}/messages",
                            {"client_msg_id": "e1", "content": "Say hello in three words."})
        assert st == 400 and err["error"]["type"] == "empty_message"
        assert "`text`, not `content`" in err["error"]["message"]
        for body in ({"client_msg_id": "e2", "text": "   "}, {"client_msg_id": "e3"}):
            assert c.call("POST", f"/conversations/{cid}/messages", body)[0] == 400
            assert c.call("POST", f"/conversations/{cid}/queue", body)[0] == 400
        assert c.call("POST", f"/conversations/{cid}/messages",
                      {"client_msg_id": "e4", "text": 5})[0] == 400
        assert not [b for b in c.fake.bodies if "model" in b and "session" not in b]
        assert c.snap(cid)["messages"] == []
        # an empty message stored before this check (an old database): refused before the engine
        um = c.svc._insert_message(cid, None, "user", [{"type": "text", "text": ""}], "done")
        with c.svc.conv(cid).lock:
            c.svc._start_run(c.svc.conv(cid), um["id"], None)
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        assert a["status"] == "error" and not [b for b in c.fake.bodies
                                               if "model" in b and "session" not in b]
        # with a file it goes
        att = c.svc.store_blob(cid, "notes.txt", "text/plain", b"one line")
        c.ok("POST", f"/conversations/{cid}/messages", {"client_msg_id": "e5", "text": "",
                                                        "attachments": [att["id"]]})
        c.wait_idle(cid, 20)
        assert "one line" in requests(c)[-1]["messages"][-1]["content"]
    finally:
        c.close()


# ---------------------------------------------------------------------- the port preview
class RelayBoxes(FakeBoxes):
    """`relay.py connect PORT` played on the host: the exec's stdin goes raw to 127.0.0.1:PORT,
    its answer comes back in the attach stream's 8-byte frames, as the real relay's does."""

    def __init__(self, data):
        super().__init__(data)
        self.relays = 0

    def open_exec(self, cid, cmd, *, tty=False, stdin=False, env=None, workdir="/workspace"):
        if cmd[-2:-1] != ["connect"]:
            return super().open_exec(cid, cmd, tty=tty, stdin=stdin, env=env, workdir=workdir)
        assert cmd[:2] == ["python3", "-c"] and "def connect(port)" in cmd[2]  # this release's
        a, b = socket.socketpair()
        srv = socket.create_connection(("127.0.0.1", int(cmd[-1])))
        self.relays += 1

        def up():
            try:
                while True:
                    d = b.recv(65536)
                    if not d:
                        break
                    srv.sendall(d)
                srv.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        def down():
            while True:
                d = srv.recv(65536)
                if not d:
                    break
                b.sendall(bytes([1, 0, 0, 0]) + len(d).to_bytes(4, "big") + d)
            # shut down before the close: on Linux a close() while `up` blocks in b.recv leaves
            # the socket open (the blocked call holds it), the reader never sees the end and the
            # preview hung on the Spark only (macOS wakes the reader)
            try:
                b.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            b.close()
            self.relays -= 1
        threading.Thread(target=up, daemon=True).start()
        threading.Thread(target=down, daemon=True).start()
        return "exec-relay", a, b""

    def exit_code(self, exec_id):
        return False, 0, None


def app_server():
    """A dev server in the sandbox: a page with root-relative links, and a WebSocket echo."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from chatd.terminals import WsConn, ws_accept

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            if ws_accept(self):
                ws = WsConn(self)
                while True:
                    m = ws.recv()
                    if m is None:
                        return
                    ws.send(m[0], b"echo:" + m[1])
            ctype = "text/html"
            body = b'<script src="/app.js"></script><a href="/about">about</a>'
            if self.path.endswith(".js"):          # Vite's way: a module's absolute imports
                ctype = "text/javascript"
                body = (b'import "/@vite/client";\nimport { a } from "/src/a.js?v=1";\n'
                        b'import b from "./b.js"; const c = import("/c.js"); const s = "/x";')
                assert "Accept-Encoding" not in self.headers, "the rewrite reads plain bytes"
            elif self.path.endswith(".css"):
                ctype = "text/css"
                body = b'a{background:url(/img/x.png)} b{background:url(//cdn/y.png)}'
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Set-Cookie", "x=1")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_the_port_preview_proxies_pages_and_websockets():
    c = Chat()
    c.svc.sandboxes = RelayBoxes(c.data)
    c.svc.sandboxes.on_change = c.svc._sandbox_changed
    c.svc.sandboxes.network_mode = c.svc._network_mode
    app = app_server()
    port = app.server_address[1]
    try:
        cid = c.new()
        base = f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}"
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/")
        assert st == 503                                   # the sandbox is not running
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/")
        body = body.decode()
        assert st == 200 and f'src="{base}/app.js"' in body and f'href="{base}/about"' in body
        assert "frame-ancestors 'self'" in h["Content-Security-Policy"] and "Set-Cookie" not in h
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/src/main.js",
                             headers={"Accept-Encoding": "gzip"})
        assert st == 200 and body.decode() == (
            f'import "{base}/@vite/client";\nimport {{ a }} from "{base}/src/a.js?v=1";\n'
            f'import b from "./b.js"; const c = import("{base}/c.js"); const s = "/x";'), body
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/s.css")
        assert body.decode() == (f"a{{background:url({base}/img/x.png)}} "
                                 "b{background:url(//cdn/y.png)}")
        sock = ws_open(c.port, base + "/socket")
        ws_send(sock, b"hello", op=1)
        assert ws_recv(sock) == b"echo:hello"
        ws_send(sock, b"\x00\x01binary", op=2)
        assert ws_recv(sock) == b"echo:\x00\x01binary"
        sock.close()
        # the browser gone, the relay ends
        c.wait(lambda: c.svc.sandboxes.relays == 0, 10)
    finally:
        app.shutdown()
        c.close()


if __name__ == "__main__":
    run_all(globals())
