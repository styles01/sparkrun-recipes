"""One chatd's containers, and a container that is gone (live 2026-10-16, Spark): a test chatd
with its own --data dir on the same Docker adopted the live chatd's containers, found their chats
missing from its store and removed them; the live chatd kept answering `exec failed (404)` and
the model retried until the step limit.

- two chatds (two data dirs) on one engine: neither adopts, reaps, stops, kills or removes the
  other's containers; the instance id is kept in the data dir, a copied dir gets its own;
- a container from before the instance label: adopted only with its workspace folder here, and
  created again with the label when it starts after a stop;
- a container removed outside chatd: the next call starts a new one and its result says once
  that the sandbox restarted; an exec answered 404 marks the box stopped;
- the same sandbox failure three calls in a row (other commands each) ends the run.

CPU only: the container engine is a fake of Docker's API with state. Run:
python tests/test_chatd_sandbox_scope.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import tempfile
import urllib.parse

from chatd_util import Chat, run_all
from test_chatd_agent import calls, chat_with_sandbox, parts_of

from chatd import sandbox as sb_mod
from chatd.sandbox import Sandboxes, Unavailable


class Docker:
    """Docker's Engine API as far as chatd uses it, with containers that exist and run."""

    def __init__(self):
        self.ctrs: dict[str, dict] = {}      # id -> {"Name", "Labels", "Running"}
        self.calls: list[tuple[str, str]] = []
        self.n = 0

    def _find(self, ref: str) -> dict | None:
        return self.ctrs.get(ref) or next((c for c in self.ctrs.values() if c["Name"] == ref),
                                          None)

    def call(self, method, path, body=None, timeout=60.0):
        self.calls.append((method, path))
        u = urllib.parse.urlparse(path)
        parts = u.path.strip("/").split("/")
        if parts[0] == "images":
            return 200, {}
        if parts[0] == "exec":
            return 200, {"Running": False, "ExitCode": 0, "Pid": None}
        if u.path == "/containers/json":
            want = json.loads(urllib.parse.parse_qs(u.query)["filters"][0])["label"]
            return 200, [{"Id": i, "Names": ["/" + c["Name"]], "Labels": c["Labels"],
                          "State": "running" if c["Running"] else "exited"}
                         for i, c in self.ctrs.items() if all(k in c["Labels"] for k in want)]
        if u.path == "/containers/create":
            name = urllib.parse.parse_qs(u.query)["name"][0]
            if self._find(name):
                return 409, {"message": "name in use"}
            self.n += 1
            cid = f"ctr{self.n}"
            self.ctrs[cid] = {"Name": name, "Labels": dict(body["Labels"]), "Running": False}
            return 201, {"Id": cid}
        c = self._find(parts[1])
        if c is None:
            return 404, {"message": "no such container"}
        cid = next(i for i, x in self.ctrs.items() if x is c)
        act = parts[2] if len(parts) > 2 else None
        if method == "DELETE":
            del self.ctrs[cid]
            return 204, None
        if act == "json":
            return 200, {"Id": cid, "Config": {"Labels": c["Labels"]},
                         "State": {"Running": c["Running"]}}
        if act == "start":
            c["Running"] = True
            return 204, None
        if act in ("stop", "kill"):
            c["Running"] = False
            return 204, None
        if act == "exec":
            return (201, {"Id": "e-" + cid}) if c["Running"] else (409, {"message": "not running"})
        return 204, None

    def hijack(self, path, body):
        a, b = socket.socketpair()
        out = b"ran\n"
        b.sendall(bytes([1, 0, 0, 0]) + struct.pack(">I", len(out)) + out)
        b.close()
        return a, b""

    def legacy(self, cid: str, running: bool = True) -> str:
        """A container made before the instance label."""
        self.n += 1
        i = f"ctr{self.n}"
        self.ctrs[i] = {"Name": "tandem-chat-" + cid.lower(), "Labels": {sb_mod.LABEL: cid},
                        "Running": running}
        return i


def boxes(engine, data=None) -> Sandboxes:
    sb = Sandboxes(data or tempfile.mkdtemp(prefix="sbx-scope-"), engine=engine,
                   meminfo_fn=lambda: {})
    sb._proxy = lambda b: None
    return sb


def test_two_chatds_on_one_engine_leave_each_other_alone():
    eng = Docker()
    live, test = boxes(eng), boxes(eng)
    try:
        assert live.instance != test.instance
        ctr = live.ensure("01LIVECHAT")
        assert eng.ctrs[ctr]["Labels"][sb_mod.INSTANCE_LABEL] == live.instance
        # the second chatd starts: it adopts nothing of the live one's
        assert test.adopt() == 0 and test.boxes == {}
        test.kill_all()
        test.remove("01LIVECHAT", files=True)            # by name, but not its own: left alone
        assert eng.ctrs[ctr]["Running"], "the live chatd's container survives"
        assert os.path.isdir(live.paths("01LIVECHAT")[0])
        # its own container, then the live chatd restarts: each adopts its own only
        mine = test.ensure("01TESTCHAT")
        again = boxes(eng, live.data)
        assert again.instance == live.instance, "the id is kept in the data dir"
        assert again.adopt() == 1 and set(again.boxes) == {"01LIVECHAT"}
        again.kill_all()
        assert eng.ctrs[mine]["Running"] and not eng.ctrs[ctr]["Running"]
    finally:
        shutil.rmtree(live.data, ignore_errors=True)
        shutil.rmtree(test.data, ignore_errors=True)


def test_the_start_reaps_only_its_own_containers():
    """chatd's start: adopt, then reap the containers of chats gone from its store. The live
    chatd's container is not this chatd's, so it is never reaped."""
    eng = Docker()
    live = boxes(eng)
    c = Chat()
    try:
        live_ctr = live.ensure("01LIVECHATLIVECHATLIVECHAT")
        sb = boxes(eng, c.data)
        c.svc.sandboxes = sb
        gone = "01GONEGONEGONEGONEGONEGONE"
        own = sb.ensure(gone)                            # its own, of a chat not in its store
        sb.boxes.clear()
        assert sb.adopt() == 1
        assert c.svc.reap_sandboxes() == [gone]
        assert own not in eng.ctrs and eng.ctrs[live_ctr]["Running"]
        assert os.path.isdir(live.paths("01LIVECHATLIVECHATLIVECHAT")[0])
    finally:
        c.close()
        shutil.rmtree(live.data, ignore_errors=True)


def test_a_copied_data_dir_gets_its_own_instance_id():
    a = tempfile.mkdtemp(prefix="sbx-a-")
    b = a + "-copy"
    try:
        first = sb_mod.instance_id(a)
        assert sb_mod.instance_id(a) == first
        shutil.copytree(a, b)
        assert sb_mod.instance_id(b) != first
        assert sb_mod.instance_id(a) == first
    finally:
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)


def test_a_container_from_before_the_label_is_adopted_only_with_its_folder():
    eng = Docker()
    mine, other = boxes(eng), boxes(eng)
    try:
        cid = "01OLDCHAT"
        os.makedirs(mine.paths(cid)[0])
        old = eng.legacy(cid)
        assert other.adopt() == 0, "no folder of that chat here: not this chatd's"
        other.remove(cid, files=False)
        assert old in eng.ctrs
        assert mine.adopt() == 1 and mine.box(cid).container == old
        assert mine.ensure(cid) == old, "a running one is used as it is"
        # stopped, it is made again with the label (the files are mounts)
        mine.stop(cid)
        new = mine.ensure(cid)
        assert new != old and old not in eng.ctrs
        assert eng.ctrs[new]["Labels"][sb_mod.INSTANCE_LABEL] == mine.instance
        # a name held by another chatd's container is refused, never used
        theirs = other.ensure("01THEIRS")
        os.makedirs(mine.paths("01THEIRS")[0])
        try:
            mine.ensure("01THEIRS")
            raise AssertionError("used another chatd's container")
        except Unavailable as exc:
            assert "another chatd" in str(exc)
        assert eng.ctrs[theirs]["Running"]
    finally:
        shutil.rmtree(mine.data, ignore_errors=True)
        shutil.rmtree(other.data, ignore_errors=True)


def test_a_container_removed_outside_chatd_is_started_again():
    eng = Docker()
    sb = boxes(eng)
    changes = []
    sb.on_change = lambda cid, st: changes.append((st["state"], st["stopped_reason"]))
    try:
        first = sb.ensure("c1")
        assert sb.take_restarted("c1") is None
        del eng.ctrs[first]                              # docker rm -f
        second = sb.ensure("c1")
        assert second != first and eng.ctrs[second]["Running"]
        assert ("stopped", "gone") in changes and changes[-1][0] == "running"
        assert "the sandbox was restarted" in sb.take_restarted("c1")
        assert sb.take_restarted("c1") is None, "said once"
        # stopped outside chatd: the same
        eng.ctrs[second]["Running"] = False
        assert sb.ensure("c1") == second and eng.ctrs[second]["Running"]
        # an exec answered 404: the box is stopped, the next call starts a new one
        del eng.ctrs[second]
        try:
            sb.open_exec("c1", ["true"])
            raise AssertionError("no error")
        except sb_mod.Gone as exc:
            assert "gone" in str(exc)
        assert sb.box("c1").state == "stopped" and sb.box("c1").stopped_reason == "gone"
    finally:
        shutil.rmtree(sb.data, ignore_errors=True)


def test_a_bash_call_after_the_container_is_gone_runs_in_a_new_one():
    c = Chat()
    eng = Docker()
    try:
        sb = boxes(eng, c.data)
        sb.on_change, sb.network_mode = c.svc._sandbox_changed, c.svc._network_mode
        c.svc.sandboxes, c.svc._caps = sb, None
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        c.send(cid, calls(("bash", {"command": "echo one"})))
        c.wait_idle(cid, 20)
        eng.ctrs.clear()                                 # docker rm -f, outside chatd
        c.send(cid, calls(("bash", {"command": "echo two"})))
        s = c.wait_idle(cid, 20)
        call = [p for p in parts_of(s) if p["type"] == "tool_call"][-1]
        assert call["status"] == "done", call
        assert call["output"]["content"].startswith("[the sandbox was restarted") and \
            "ran" in call["output"]["content"], call["output"]
        assert len(eng.ctrs) == 1 and sb.box(cid).state == "running"
        c.send(cid, calls(("bash", {"command": "echo three"})))
        s = c.wait_idle(cid, 20)
        assert "restarted" not in [p for p in parts_of(s) if p["type"] == "tool_call"][-1][
            "output"]["content"], "said once"
    finally:
        c.close()


def test_the_same_sandbox_failure_three_times_ends_the_run():
    c = chat_with_sandbox()
    try:
        def broken(*a, **k):
            raise Unavailable("the container engine answered 500")
        c.svc.sandboxes.run = broken
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        # other arguments each time: the same-call guard does not see it
        c.send(cid, calls(*[("bash", {"command": f"sleep 30; echo ping{i}"}) for i in range(5)]))
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        tc = [p for p in a["parts"] if p["type"] == "tool_call"]
        assert [p["output"]["content"] for p in tc][:3] == [
            "the container engine answered 500"] * 3
        assert all("Not run" in p["output"]["content"] for p in tc[3:]) and len(tc) == 5
        assert a["status"] == "interrupted" and a["stats"]["interrupted"] == "sandbox"
        assert "sandbox failed three times" in a["parts"][-1]["message"]
        assert a["stats"]["steps"] == 1, "no further model call"
        # the sandbox is back: Continue goes on
        c.svc.sandboxes.run = lambda *a, **k: (0, False)
        c.ok("POST", f"/conversations/{cid}/continue")
        assert c.wait_idle(cid, 20)["messages"][-1]["status"] == "done"
    finally:
        c.close()


def test_an_interrupted_answer_records_why():
    """VIS-197: the Continue card said "The engine restarted" for a run at its limit of model
    calls. chatd records the reason in the answer's stats."""
    c = Chat()
    try:
        cid = c.new()
        c.ok("PATCH", f"/conversations/{cid}/settings", {"max_steps": 1})
        c.send(cid, calls(("glob", {"pattern": "*"})))
        a = c.wait_idle(cid, 20)["messages"][-1]
        assert a["status"] == "interrupted" and a["stats"]["interrupted"] == "limit"
    finally:
        c.close()
    # chatd stopped with a run on: `restart`, kept in the stored answer
    c = Chat()
    data = c.data
    try:
        cid = c.new()
        c.send(cid, "hello")
        mid = c.wait_idle(cid, 20)["messages"][-1]["id"]
        c.svc.store.run("UPDATE messages SET status = 'streaming' WHERE id = ?", (mid,))
        c.svc.store.run("UPDATE runs SET state = 'running' WHERE assistant_message_id = ?", (mid,))
    finally:
        c.close(keep_data=True)
    c = Chat(data=data)
    try:
        a = c.snap(cid)["messages"][-1]
        assert a["status"] == "interrupted" and a["stats"]["interrupted"] == "restart", a["stats"]
        assert a["stats"]["completion_tokens"] > 0, "the earlier figures stay"
    finally:
        c.close()


if __name__ == "__main__":
    run_all(globals())
