"""Workers: Stop, a steer, Continue and failures (VIS-E15, SRV-585; QA SRV-586). CPU only:
tests/fake_backend.py with a script for the main agent and its worker (tests/test_chatd_tasks.py),
a backend that answers 503 for the retry window or 400, a chatd restarted on the same data, and
the fake sandbox runtime of tests/test_chatd_agent.py, whose log of commands proves that no command
runs twice.

  1  Stop stops both within 2 s; the task result says "cancelled by the user at call 2" in its
     first line and keeps chatd's check            test_stop_stops_both
  2  a steer while a worker runs: a note at the end of the worker's next stored tool result, read
     in its next request; the main agent reads the message after the report
                                                    test_a_steer_reaches_the_main_agent_and_shows_to_the_worker
  3  the engine away for the retry window: worker and parent end interrupted (stats.interrupted
     task); Continue resumes the worker first, then the parent
                                                    test_an_interrupted_worker_resumes_with_continue
  4  chatd restarted while the worker's command ran: on Continue the command is not run again, its
     result says its effect is unknown              test_an_interrupted_command_is_not_run_again
  5  two Continues at once: one resume, the other answers 409
                                                    test_two_continues_start_one_resume
  6  a report stored after its parent ended: Continue delivers it, no worker runs
                                                    test_a_report_that_lands_after_the_parent_ended_is_kept
  7  the engine answers 400: the task result says failed with the engine's words
                                                    test_a_failed_worker_reports_the_failure
  8  a worker's prompt in Ask mode: tool.asking with task_id on the chat, the slot given back, the
     answer at the chat's route                     test_a_worker_asks_for_permission_in_ask_mode

Run: PYTHONPATH=.:tests python3 tests/test_chatd_task_control.py
"""

from __future__ import annotations

import json
import os
import threading
import time

from chatd_util import Chat, run_all
from test_chatd_agent import FakeBoxes
from test_chatd_tasks import (REPORT, Script, agents_chat, hdr, main_calls, task_call,
                              task_part, worker_calls)


def boxed(c: Chat) -> FakeBoxes:
    c.svc.sandboxes = FakeBoxes(c.data)
    c.svc.sandboxes.on_change = c.svc._sandbox_changed
    c.svc.sandboxes.network_mode = c.svc._network_mode
    c.svc._caps = None
    return c.svc.sandboxes


def bash_then_report(cmd: str):
    return lambda b, k, n: {"calls": [{"name": "bash", "arguments": {"command": cmd}}]} \
        if k == 0 else {"content": REPORT}


def running_worker(c, cid, step=1):
    conv = c.svc.conv(cid)
    c.wait(lambda: conv.run is not None and conv.run.child is not None and
           conv.run.child.step >= step, 15)
    return conv.run.child


# ---------------------------------------------------------------------- Stop
def test_stop_stops_both():
    c = Chat()
    try:
        def worker(b, k, n):
            if k == 0:
                return {"calls": [{"name": "write_file", "arguments": {"path": "hello.txt",
                                                                       "content": "hi"}}]}
            return {"content": "y" * 3000, "tps": 20}                 # 30 s of text
        c.fake.script = Script(worker=worker)
        cid = agents_chat(c)
        c.send(cid, "go")
        w = running_worker(c, cid, 2)
        c.wait(lambda: w.phase == "writing", 10)
        t0 = time.monotonic()
        c.ok("POST", f"/conversations/{cid}/stop")
        c.wait(lambda: c.snap(cid)["run"] is None, 5)
        assert time.monotonic() - t0 < 2.0
        s = c.snap(cid)
        p = task_part(s)
        assert p["status"] == "cancelled" and w.state == "cancelled"
        first = p["model_content"].splitlines()[0]
        assert "cancelled by the user at call 2" in first, first
        assert "- Changed in /workspace (snapshot diff): hello.txt (new, +1)" in p["model_content"]
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        assert t["status"] == "cancelled" and t["report_source"] == "chatd"
        assert s["messages"][-1]["status"] == "stopped"
    finally:
        c.close()


# ---------------------------------------------------------------------- a steer
def test_a_steer_reaches_the_main_agent_and_shows_to_the_worker():
    c = Chat()
    try:
        box = boxed(c)
        box.script["npm install"] = (["added 1 package\n"] * 15, 0, 0.1)
        seen = []

        def main(b, k):
            if b["messages"][-1].get("content") == "use WebGL2, not WebGPU":
                seen.append(b)
                return {"content": "Noted."}
            return {"calls": [task_call()]} if k == 0 else {"content": "ok"}
        c.fake.script = Script(main, bash_then_report("npm install three"))
        cid = agents_chat(c, permission_mode="auto")
        c.send(cid, "go")
        c.wait(lambda: ["bash", "-lc", "npm install three"] in box.ran, 15)
        c.ok("POST", f"/conversations/{cid}/queue", {"client_msg_id": "s1",
                                                    "text": "use WebGL2, not WebGPU"})
        c.wait(lambda: seen, 30)
        c.wait_idle(cid, 30)
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        d = c.ok("GET", f"/conversations/{cid}/tasks/{t['id']}")
        bash = next(p for p in d["messages"][1]["parts"] if p.get("name") == "bash")
        note = bash["model_content"].rsplit("\n\n", 1)[-1]
        assert note.startswith("[Note from the chat, ") and note.endswith(
            ': the user wrote to the main agent: "use WebGL2, not WebGPU". The main agent reads '
            'it after your report. If it changes or cancels your goal, stop at a safe point and '
            'report.]'), note
        nxt = worker_calls(c)[1][1]["messages"][-1]
        assert nxt["role"] == "tool" and nxt["content"].endswith(note)
        # the main agent: the task's result, then the steer as the next run's message
        msgs = seen[0]["messages"]
        assert [m["role"] for m in msgs[-3:]] == ["assistant", "tool", "user"]
        assert msgs[-2]["content"].startswith("Result of task 1") and \
            msgs[-1]["content"] == "use WebGL2, not WebGPU"
    finally:
        c.close()


# ---------------------------------------------------------------------- Continue
def test_an_interrupted_worker_resumes_with_continue():
    c = Chat(retry_s=2)
    try:
        away = [True]

        def worker(b, k, n):
            if k == 1 and away[0]:
                return {"status": 503, "message": "the engine is restarting"}
            return {"calls": [{"name": "write_file", "arguments": {"path": "hello.txt",
                                                                   "content": "hi"}}]} \
                if k == 0 else {"content": REPORT}
        c.fake.script = Script(worker=worker)
        cid = agents_chat(c)
        c.send(cid, "go")
        s = c.wait_idle(cid, 30)
        a = s["messages"][-1]
        assert a["status"] == "interrupted" and a["stats"]["interrupted"] == "task", a["stats"]
        assert task_part(s)["status"] == "interrupted"
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        assert t["status"] == "interrupted"
        n_before = len(c.fake.calls)
        away[0] = False
        c.ok("POST", f"/conversations/{cid}/continue")
        s = c.wait_idle(cid, 30)
        later = [hdr(h, "x-session-id") for h, b in c.fake.calls[n_before:] if b.get("stream")]
        assert later[0] == f"chat-{cid}-task-1" and later[-1] == f"chat-{cid}", later
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        assert t["status"] == "done" and t["attempt"] == 2 and t["steps"] == 2
        assert task_part(s)["status"] == "done" and s["messages"][-1]["status"] == "done"
        assert len(c.ok("GET", f"/conversations/{cid}/tasks")["tasks"]) == 1   # the same task
    finally:
        c.close()


def test_an_interrupted_command_is_not_run_again():
    c = Chat()
    data, fake = c.data, (c.fake_srv, c.fake_port)
    c.owned_fake = False
    c2 = None
    try:
        box = boxed(c)
        box.script["npm install"] = (["added\n"] * 40, 0, 0.1)          # 4 s
        c.fake.script = Script(worker=bash_then_report("npm install three"))
        cid = agents_chat(c, permission_mode="auto")
        c.send(cid, "go")
        c.wait(lambda: ["bash", "-lc", "npm install three"] in box.ran, 15)
        c.close(keep_data=True)                       # chatd stops while the command runs
        c2 = Chat(data=data, fake=fake)
        box2 = boxed(c2)
        s = c2.snap(cid)
        assert s["messages"][-1]["status"] == "interrupted"
        assert s["messages"][-1]["stats"]["interrupted"] == "task"
        n0 = len(c2.fake.calls)
        c2.ok("POST", f"/conversations/{cid}/continue")
        s = c2.wait_idle(cid, 30)
        assert box.ran.count(["bash", "-lc", "npm install three"]) == 1 and box2.ran == []
        t = c2.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        d = c2.ok("GET", f"/conversations/{cid}/tasks/{t['id']}")
        bash = next(p for p in d["messages"][1]["parts"] if p.get("name") == "bash")
        assert bash["status"] == "interrupted" and bash["model_content"].endswith(
            "this call was interrupted and its effect is unknown; check the file before you "
            "repeat it")
        resumed = next(b for h, b in c2.fake.calls[n0:] if b.get("stream") and
                       "-task-" in (hdr(h, "x-session-id") or ""))["messages"]
        assert resumed[-1]["role"] == "tool" and "effect is unknown" in resumed[-1]["content"]
        assert t["status"] == "done" and s["messages"][-1]["status"] == "done"
    finally:
        if c2 is not None:
            c2.close()
        fake[0].shutdown()


def test_two_continues_start_one_resume():
    c = Chat(retry_s=1)
    try:
        away = [True]
        c.fake.script = Script(worker=lambda b, k, n: {"status": 503} if away[0] else
                               {"content": REPORT})
        cid = agents_chat(c)
        c.send(cid, "go")
        c.wait_idle(cid, 30)
        away[0] = False
        got = []

        def press():
            got.append(c.call("POST", f"/conversations/{cid}/continue")[0])
        ts = [threading.Thread(target=press) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert sorted(got) == [202, 409], got
        c.wait_idle(cid, 30)
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        assert t["attempt"] == 2 and t["status"] == "done"
    finally:
        c.close()


def test_a_report_that_lands_after_the_parent_ended_is_kept():
    c = Chat()
    try:
        c.fake.script = Script()
        cid = agents_chat(c)
        c.send(cid, "go")
        s = c.wait_idle(cid, 30)
        stored = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        # the parent ended while its worker still wrote the report (a stop, a restart): its task
        # call never got the result; the worker's end is stored
        m = s["messages"][-1]
        parts = [p for p in m["parts"] if p["type"] != "text"]
        for p in parts:
            if p.get("name") == "task":
                p.update(status="running", output=None, model_content=None, ended_at=None)
        c.svc.store.run("UPDATE messages SET parts = ?, status = 'interrupted' WHERE id = ?",
                        (json.dumps(parts), m["id"]))
        n_worker = len(worker_calls(c))
        c.ok("POST", f"/conversations/{cid}/continue")
        s = c.wait_idle(cid, 30)
        p = task_part(s)
        assert p["status"] == "done" and p["model_content"].startswith("Result of task 1")
        assert len(worker_calls(c)) == n_worker                  # no worker ran again
        assert c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]["ended"] == stored["ended"]
        assert s["messages"][-1]["status"] == "done"
    finally:
        c.close()


# ---------------------------------------------------------------------- failures and prompts
def test_a_failed_worker_reports_the_failure():
    c = Chat()
    try:
        c.fake.script = Script(worker=lambda b, k, n: {"status": 400,
                                                       "message": "the prompt is malformed"})
        cid = agents_chat(c)
        c.send(cid, "go")
        s = c.wait_idle(cid, 30)
        p = task_part(s)
        assert p["status"] == "failed" and p["output"]["is_error"]
        text = p["model_content"]
        assert "the worker failed" in text and "the prompt is malformed" in text, text
        assert "Status: failed" in text
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        assert t["status"] == "failed" and "the prompt is malformed" in (t["error"] or "")
        assert s["messages"][-1]["status"] == "done"            # the main agent decides
        assert main_calls(c, cid)[-1][1]["messages"][-1]["content"] == text
    finally:
        c.close()


def test_a_worker_asks_for_permission_in_ask_mode():
    c = Chat()
    try:
        boxed(c)
        c.fake.script = Script(worker=bash_then_report("npm install three"))
        cid = agents_chat(c, permission_mode="ask")
        ev = c.events(cid)
        c.send(cid, "go")
        frames = ev.until(lambda f: f[1] == "tool.asking", 20)
        asking = json.loads(frames[-1][2])
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        assert asking["task_id"] == t["id"] and asking["summary"] == "Run: npm install three"
        c.wait(lambda: not c.svc.slots.holders, 5)              # the slot is given back
        assert c.snap(cid)["status"] == "waiting"
        rail = [json.loads(r["data"]) for r in c.svc.store.all(
            "SELECT data FROM global_events WHERE type = 'tool.asking'")]
        assert rail and rail[-1]["conversation_id"] == cid
        c.ok("POST", f"/conversations/{cid}/permissions/{asking['call_id']}", {"decision": "once"})
        frames = ev.until(lambda f: f[1] == "run.done", 30)
        ev.close()
        s = c.wait_idle(cid, 10)
        assert task_part(s)["status"] == "done"
        assert ["bash", "-lc", "npm install three"] in c.svc.sandboxes.ran
    finally:
        c.close()


if __name__ == "__main__":
    run_all(globals())
