"""Agents mode (VIS-E15, SRV-583; QA SRV-584): the main agent's tools, its caps, its read cache and
its paragraph. CPU only: tests/fake_backend.py with a script (tests/test_chatd_tasks.py), the fake
sandbox runtime of tests/test_chatd_agent.py where bash and render_page must be offered.

  1  the tool lists per mode, in their fixed order: Agents on has no bash, write_file or edit_file,
     the todo list and task, then the MCP tools; off is today's; plan mode's read tools win
                                                    test_the_tool_lists_per_mode
  2  a write tool the model calls anyway: the refusal, no file
                                                    test_a_write_tool_is_refused
  3  read_file: 200 lines or 12,000 characters and where it stopped; any result cut at 12,000
     characters, the rest saved; Agents off unchanged
                                                    test_the_reads_and_results_are_capped
  4  the read cache: the same read answers "unchanged since call <id> (step <k>)"; a read after a
     worker ran, after a file changed or after a compaction runs again
                                                    test_the_read_cache_hits_and_misses
  5  the paragraph: in Agents mode only, right after the tool rules
                                                    test_the_paragraph_is_in_agents_mode_only
  6  a switch during a run: the run keeps its tools, the next message's run has the Agents list;
     the settings answer carries reread_tokens       test_a_switch_takes_effect_at_the_next_message
  7  the defaults: off for a new chat, the new-chat default settable; the panel's last worker and
     the usage's workers                             test_the_defaults_and_the_figures
  8  Agents off is today's loop: the stored chats' requests are main's, byte for byte
                                                    test_agents_off_is_todays_loop

Run: PYTHONPATH=.:tests python3 tests/test_chatd_agents_mode.py
"""

from __future__ import annotations

import json
import os

import test_chatd_tasks
from chatd_util import Chat, run_all
from test_chatd_agent import chat_with_sandbox
from test_chatd_tasks import REPORT, Script, agents_chat, main_calls, task_call

from chatd import tools as T
from chatd.plans import PLAN_TOOLS

PARAGRAPH = ("You lead this chat. You talk with the user, plan the work, and give the hands-on "
             "work to workers with the task tool. You cannot write files or run commands yourself.")
NO_CODE = ("Never write code or file contents yourself, not in a brief and not in your thinking: "
           "describe what the result must do, and let the worker design and write it. Decide the "
           "look and behaviour in words (colours, layout, features), not in code.")


def names(body: dict) -> list[str]:
    return [t["function"]["name"] for t in body.get("tools") or []]


def main_parts(s: dict) -> list[dict]:
    return [p for p in s["messages"][-1]["parts"] if p["type"] == "tool_call"]


def test_the_tool_lists_per_mode():
    c = chat_with_sandbox()
    try:
        c.fake.script = Script(lambda b, k: {"content": "ok"})
        on, off = agents_chat(c), c.new()
        c.send(on, "hello")
        c.send(off, "hello")
        c.wait_idle(on, 20)
        c.wait_idle(off, 20)
        assert names(main_calls(c, on)[-1][1]) == [
            "read_file", "glob", "grep", "web_search", "web_fetch", "render_page", "present",
            "todo_write", "task"]
        assert names(main_calls(c, off)[-1][1]) == [
            "bash", "read_file", "write_file", "edit_file", "glob", "grep", "web_search",
            "web_fetch", "render_page", "present", "todo_write"]
        # the chat's MCP tools after task; plan mode keeps its read tools and the list
        mcp = T.Tool("mcp__notes__find", "find a note", {"type": "object", "properties": {}})
        c.svc.mcp.tools_for = lambda settings, cid: [mcp]
        settings = c.svc.settings_of(on)
        info = c.svc.model_info(settings.get("model"))
        got = [t.name for t in c.svc.tools_for(settings, info, on, agents=True)]
        assert got[-2:] == ["task", "mcp__notes__find"] and "bash" not in got
        plan = [t.name for t in c.svc.tools_for({**settings, "plan_mode": True}, info, on,
                                                agents=True)]
        # SRV-663: plan mode keeps the list (the prompt's head); the run refuses all but these
        assert plan == got and {"read_file", "todo_write"} <= set(PLAN_TOOLS)
    finally:
        c.close()


def test_a_write_tool_is_refused():
    c = chat_with_sandbox()
    try:
        c.fake.script = Script(lambda b, k: {"calls": [
            {"name": "write_file", "arguments": {"path": "x.txt", "content": "no"}},
            {"name": "bash", "arguments": {"command": "touch y.txt"}}]} if k == 0 else
            {"content": "ok"})
        cid = agents_chat(c, permission_mode="yolo")
        c.send(cid, "write it yourself")
        s = c.wait_idle(cid, 20)
        for p in main_parts(s):
            assert p["status"] == "failed" and p["output"]["content"] == \
                "not available in Agents mode: give this to a worker with task"
        assert not os.path.exists(os.path.join(c.data, "workspaces", cid, "x.txt"))
        assert c.svc.sandboxes.ran == []
    finally:
        c.close()


def test_the_reads_and_results_are_capped():
    c = Chat()
    try:
        steps = [{"calls": [{"name": "read_file", "arguments": {"path": "src/main.js"}}]},
                 {"calls": [{"name": "read_file", "arguments": {"path": "wide.txt"}}]},
                 {"calls": [{"name": "read_file", "arguments": {"path": "src/main.js",
                                                                "offset": 1801, "limit": 500}}]},
                 {"calls": [{"name": "grep", "arguments": {"pattern": "www"}}]},
                 {"content": "ok"}]
        c.fake.script = Script(lambda b, k: steps[k])
        on, off = agents_chat(c), c.new()
        for cid in (on, off):
            ws = c.svc.workspace(cid)
            ws.write("src/main.js", "".join(f"const value{i} = {i};\n"
                                            for i in range(1834)).encode())
            ws.write("wide.txt", ("w" * 500 + "\n").encode() * 100)
        c.send(on, "read")
        s = c.wait_idle(on, 30)
        r1, r2, r3, g = [p["model_content"] for p in main_parts(s)]
        assert r1.splitlines()[0] == "     1\tconst value0 = 0;"
        assert r1.splitlines()[199].startswith("   200\t") and len(r1.splitlines()) == 201
        assert r1.endswith("Showing lines 1-200 of 1,834. Read a range with offset and limit, "
                           "or give the reading to a worker.")
        assert len(r2) <= 12_200 and r2.endswith(
            "Showing lines 1-23 of 100. Read a range with offset and limit, or give the reading "
            "to a worker."), r2[-120:]
        assert r3.splitlines()[0].startswith("  1801\t") and "Showing" not in r3      # its end
        assert len(g) < 12_400 and "characters left out; the whole output is in " \
               "/workspace/.tandem/outputs/" in g
        # Agents off: as before
        c.send(off, "read")
        s = c.wait_idle(off, 30)
        r1 = main_parts(s)[0]["model_content"]
        assert len(r1.splitlines()) == 1834 and "Showing" not in r1
    finally:
        c.close()


def test_the_read_cache_hits_and_misses():
    c = Chat()
    try:
        read = {"name": "read_file", "arguments": {"path": "src/main.js"}}
        plan = {"a": [{"calls": [read]}, {"calls": [read]}, {"content": "ok"}],
                "b": [{"calls": [read]}, {"calls": [task_call()]}, {"calls": [read]},
                      {"content": "ok"}]}
        turn = ["a"]

        def worker(b, k, n):
            if k == 0:
                return {"calls": [{"name": "edit_file", "arguments": {
                    "path": "src/main.js", "old": "one", "new": "two"}}]}
            return {"content": REPORT}
        c.fake.script = Script(lambda b, k: plan[turn[0]][k], worker)
        cid = agents_chat(c)
        c.svc.workspace(cid).write("src/main.js", b"const one = 1;\n")
        c.send(cid, "read twice")
        s = c.wait_idle(cid, 30)
        first, second = main_parts(s)
        assert second["model_content"] == f"unchanged since call {first['id']} (step 1)"
        assert second["cached_from"] == first["id"]
        # a worker ran between the reads: the read runs
        turn[0] = "b"
        c.send(cid, "read, change, read")
        s = c.wait_idle(cid, 30)
        r1, task, r2 = main_parts(s)
        assert r1["model_content"] == f"unchanged since call {first['id']} (step 1)"  # last turn's
        assert task["status"] == "done" and "const two = 1;" in r2["model_content"]
        # a file changed with no worker (the files pane): the read runs
        turn[0] = "a"
        c.ok("PUT", f"/conversations/{cid}/sandbox/file?path=/workspace/src/main.js",
             raw=b"const three = 1;\n")
        c.send(cid, "and again")
        s = c.wait_idle(cid, 30)
        a, b2 = main_parts(s)
        assert "const three" in a["model_content"] and b2["model_content"].startswith(
            "unchanged since call ")
        # the earlier result compacted away: the read runs
        def compacting(b, k):
            if k == 0:
                msgs = c.svc._messages(cid)
                u = max(m for m in msgs if msgs[m]["role"] == "user")
                c.svc.store.run("INSERT INTO compactions VALUES ('cp1', ?, ?, 'summary', 1, 1, 1, "
                                "1, NULL)", (cid, u))
            return plan["a"][k]
        c.fake.script.main = compacting
        c.send(cid, "after a compaction")
        s = c.wait_idle(cid, 30)
        a, b2 = main_parts(s)
        assert "const three" in a["model_content"]
        assert b2["model_content"] == f"unchanged since call {a['id']} (step 1)"
    finally:
        c.close()


def test_the_paragraph_is_in_agents_mode_only():
    c = Chat()
    try:
        c.fake.script = Script(lambda b, k: {"content": "ok"})
        on, off = agents_chat(c), c.new()
        for cid in (on, off):
            c.send(cid, "hi")
            c.wait_idle(cid, 20)
        sys_on = main_calls(c, on)[-1][1]["messages"][0]["content"]
        sys_off = main_calls(c, off)[-1][1]["messages"][0]["content"]
        assert PARAGRAPH not in sys_off
        rules = sys_off.split("\n\n")[0]
        assert sys_on.split("\n\n")[0] == rules and sys_on.split("\n\n")[1] == PARAGRAPH
        assert sys_on.endswith("A report marked partial or failed is not done: give a better "
                               "brief, change the plan, or ask the user.")
        assert NO_CODE in sys_on.split("\n\n") and NO_CODE not in sys_off
        task = [t["function"] for t in main_calls(c, on)[-1][1]["tools"]
                if t["function"]["name"] == "task"][0]
        assert task["description"].endswith(" " + NO_CODE)
        assert ("The worker also gets the user's request word for word: name which part is this "
                "task and what is already done; you do not need to repeat the details.") in sys_on
    finally:
        c.close()


def test_a_switch_takes_effect_at_the_next_message():
    c = chat_with_sandbox()
    try:
        answers = {}

        def main(b, k):
            if k == 0 and "switch" in b["messages"][-1].get("content", ""):
                answers["patch"] = c.ok("PATCH", f"/conversations/{cid}/settings",
                                        {"agents": True})
                return {"calls": [{"name": "glob", "arguments": {"pattern": "*"}}]}
            return {"content": "ok"}
        c.fake.script = Script(main)
        cid = c.new()
        c.send(cid, "first")
        c.wait_idle(cid, 20)
        c.send(cid, "switch now")
        c.wait_idle(cid, 20)
        c.send(cid, "next")
        c.wait_idle(cid, 20)
        bodies = [b for _, b in main_calls(c, cid)]
        assert "bash" in names(bodies[1]) and "bash" in names(bodies[2])    # the run's own
        assert PARAGRAPH not in bodies[2]["messages"][0]["content"]
        assert "task" in names(bodies[3]) and "bash" not in names(bodies[3])
        assert PARAGRAPH in bodies[3]["messages"][0]["content"]
        assert answers["patch"]["agents"] is True
        assert answers["patch"]["reread_tokens"] == \
            c.snap(cid)["messages"][1]["stats"]["prompt_tokens"] > 0
        fresh = c.new()
        assert c.ok("PATCH", f"/conversations/{fresh}/settings", {"agents": False})[
            "reread_tokens"] == 0
    finally:
        c.close()


def test_the_defaults_and_the_figures():
    c = Chat()
    try:
        assert c.ok("GET", "/capabilities")["defaults"]["agents"] is False
        assert c.snap(c.new())["settings"]["agents"] is False
        assert c.ok("PATCH", "/defaults", {"agents": True})["agents"] is True
        cid = c.new()
        assert c.snap(cid)["settings"]["agents"] is True
        st, err, _ = c.call("PATCH", "/defaults", {"max_steps": 3})
        assert st == 400
        c.fake.script = Script()
        c.send(cid, "go")
        s = c.wait_idle(cid, 30)
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        lw = s["context"]["last_worker"]
        assert lw["task_id"] == t["id"] and lw["n"] == 1 and lw["status"] == "done" and \
            lw["used"] > 0
        w = s["usage"]["workers"]
        assert w["count"] == 1 and w["prompt_tokens"] == t["prompt_tokens"]
        assert s["usage"]["prompt_tokens"] >= t["prompt_tokens"]
    finally:
        c.close()


def test_agents_off_is_todays_loop():
    test_chatd_tasks.test_agents_off_requests_are_byte_identical_to_main()


if __name__ == "__main__":
    run_all(globals())
