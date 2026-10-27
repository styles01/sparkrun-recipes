"""Stored chats replayed into the requests chatd builds for them (VIS-E15, SRV-579, SRV-583): with
Agents mode off, every request must be the one origin/main built, byte for byte, headers included.

The chats are rows written straight into a fresh store (fixed ids, times and texts): a plain chat
with thinking, a tool run (a long write, a long read, a command, an edit, a todo list), a long run
compacted inside, a chat with a system prompt, an effort and thinking off, and a run in the middle of
its steps. `build(svc)` writes them and answers {name: {"body", "headers"}} for one request each,
the date frozen. The fixture tests/fixtures/chatd_replay_main.json holds what origin/main
(ca07b67c) answers, with read_file's `column` argument (SRV-639) in its tool list and a write's
content whole in the requests after it (SRV-659: no stub in its place); write it again from a
checkout of that tree:

    PYTHONPATH=.:tests python3 tests/chatd_replay.py --write <checkout>/../chatd_replay_main.json

CPU only, the standard library.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys

from chatd_util import Chat

from chatd.runs import Run

T0 = 1_792_000_000_000              # 2026-10-17, the stored rows' clock
TOOLS = ["read_file", "write_file", "edit_file", "glob", "grep", "web_search", "web_fetch",
         "present", "todo_write"]
LONG = "".join(f"const v{i} = {i} * Math.sin({i} / 7);\n" for i in range(160))


class _Day(_dt.date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 17)


class _Dt:
    date = _Day


def _chat(svc, cid: str, settings: dict, head: str) -> None:
    svc.store.run("INSERT INTO conversations (id, title, created, updated, model, settings, ui, "
                  "head_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (cid, "replay", T0, T0, "fake-a", json.dumps({"model": "fake-a",
                                                                 "tools": TOOLS, **settings}),
                   json.dumps({"panes": [], "active": None, "split": 0.6}), head))


def _msg(svc, cid: str, mid: str, parent: str | None, role: str, parts: list, t: int,
         stats: dict | None = None, attachments: list | None = None) -> None:
    svc.store.run("INSERT INTO messages (id, conversation_id, parent_id, role, created, status, "
                  "parts, attachments, run_id, client_msg_id, stats) VALUES (?, ?, ?, ?, ?, 'done', "
                  "?, ?, NULL, ?, ?)",
                  (mid, cid, parent, role, T0 + t, json.dumps(parts),
                   json.dumps(attachments) if attachments else None, f"cm-{mid}",
                   json.dumps(stats) if stats else None))


def _call(cid: str, name: str, args: dict, out: str, step: int, ended: str, **extra) -> dict:
    return {"type": "tool_call", "id": cid, "name": name, "source": {"kind": "builtin"},
            "risk": "read", "input": json.dumps(args), "status": "done",
            "output": {"content": out, "is_error": False, "bytes": len(out), "truncated": False,
                       "exit_code": extra.pop("exit_code", None), "images": []},
            "model_content": out, "started_at": ended, "ended_at": ended, "permission": None,
            "pane": None, "step": step, **extra}


def _stats(prompt: int, completion: int, steps: int = 1) -> dict:
    return {"finish": "stop", "prompt_tokens": prompt, "completion_tokens": completion,
            "reasoning_tokens": 3, "cached_tokens": 0, "steps": steps,
            "step_usage": [{"step": k, "prompt_tokens": prompt, "cached_tokens": 0,
                            "completion_tokens": completion, "reasoning_tokens": 1}
                           for k in range(1, steps + 1)]}


def write_chats(svc) -> dict:
    """The stored chats; {name: (cid, user message id, assistant message id, live parts, step)}."""
    out = {}
    # 1. a plain chat: two turns, the answer with its thinking
    _chat(svc, "RPLAIN", {}, "RPLAIN-u2")
    _msg(svc, "RPLAIN", "RPLAIN-u1", None, "user", [{"type": "text", "text": "What is a quine?"}],
         1)
    _msg(svc, "RPLAIN", "RPLAIN-a1", "RPLAIN-u1", "assistant",
         [{"type": "thinking", "text": "a program that prints itself", "tokens": 6, "done": True,
           "step": 1}, {"type": "text", "text": "A program that prints its own source.",
                        "step": 1}], 2, _stats(40, 20))
    _msg(svc, "RPLAIN", "RPLAIN-u2", "RPLAIN-a1", "user",
         [{"type": "text", "text": "Show one in Python."}], 3)
    out["plain"] = ("RPLAIN", "RPLAIN-u2", "RPLAIN-a2", [], 1)

    # 2. a tool run (a long write, a read, a command, an edit) and a todo list on the next message
    _chat(svc, "RTOOLS", {"agents": False}, "RTOOLS-u2")
    _msg(svc, "RTOOLS", "RTOOLS-u1", None, "user",
         [{"type": "text", "text": "Write sine.js and check it."}], 1)
    at = "2026-10-17T10:00:0"
    parts = [
        {"type": "text", "text": "Writing it.", "step": 1},
        _call("call_1", "write_file", {"path": "sine.js", "content": LONG},
              "wrote /workspace/sine.js (6,000 bytes)", 1, at + "1.000Z", risk="write"),
        _call("call_2", "read_file", {"path": "sine.js"},
              "\n".join(f"{i:6d}\t{ln}" for i, ln in enumerate(LONG.splitlines(), 1)), 2,
              at + "2.000Z"),
        _call("call_3", "edit_file", {"path": "sine.js", "old": "v1 =", "new": "w1 ="},
              "edited /workspace/sine.js: 1 replacement(s)", 3, at + "3.000Z", risk="write"),
        _call("call_4", "glob", {"pattern": "**/*.js"}, "/workspace/sine.js", 3, at + "3.500Z"),
        {"type": "text", "text": "Done: sine.js computes 160 values.", "step": 4},
    ]
    _msg(svc, "RTOOLS", "RTOOLS-a1", "RTOOLS-u1", "assistant", parts, 2, _stats(900, 300, 4))
    items = [{"id": "t1", "text": "write sine.js", "state": "done"},
             {"id": "t2", "text": "plot it", "state": "pending"}]
    _msg(svc, "RTOOLS", "RTOOLS-u2", "RTOOLS-a1", "user",
         [{"type": "text", "text": "Now plot it."},
          {"type": "todos", "text": "Your todo list (keep it current with todo_write):\n[x] write "
                                    "sine.js\n[ ] plot it", "items": items}], 3)
    out["tools"] = ("RTOOLS", "RTOOLS-u2", "RTOOLS-a2", [], 1)

    # 3. a long run compacted inside: the summary in the system block, its steps from 3 on
    _chat(svc, "RLONG", {}, "RLONG-u1")
    _msg(svc, "RLONG", "RLONG-u1", None, "user", [{"type": "text", "text": "Build the site."}], 1)
    live = []
    for k in range(1, 6):
        live.append({"type": "text", "text": f"step {k}", "step": k})
        live.append(_call(f"call_l{k}", "read_file", {"path": f"p{k}.html"},
                          f"     1\t<p>{k}</p>" * 400, k, f"2026-10-17T11:00:0{k}.000Z"))
    _msg(svc, "RLONG", "RLONG-a1", "RLONG-u1", "assistant", live, 2)
    svc.store.run("INSERT INTO compactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  ("RLONG-c1", "RLONG", "RLONG-a1@3", "The user wants a site; p1 and p2 read.",
                   9000, 3000, T0 + 2500, 1200, None))
    out["long_run"] = ("RLONG", "RLONG-u1", "RLONG-a1", live, 6)

    # 4. a system prompt, an effort, thinking off, a text attachment
    svc.store.run("INSERT INTO prompts VALUES (?, ?, ?, ?)", ("RP1", "terse", "Terse",
                                                              "Answer in one line."))
    _chat(svc, "RPROMPT", {"prompt_id": "RP1", "effort": "high", "thinking": False}, "RPROMPT-u1")
    svc.store.run("INSERT INTO attachments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  ("RATT", "RPROMPT", "notes.md", "text", "text/markdown", 0, "0" * 64, 5, T0))
    _msg(svc, "RPROMPT", "RPROMPT-u1", None, "user",
         [{"type": "text", "text": "Summarize the notes."}], 1,
         attachments=[{"id": "RATT", "name": "notes.md", "kind": "text", "mime": "text/markdown",
                       "bytes": 0, "tokens": 5, "inline": False, "path": "/workspace/uploads/notes.md"}])
    out["prompt"] = ("RPROMPT", "RPROMPT-u1", "RPROMPT-a1", [], 1)

    # 5. a run in the middle of its steps (the next call's request)
    _chat(svc, "RMID", {"agents": False, "plan_mode": False, "effort": "xhigh"}, "RMID-u1")
    _msg(svc, "RMID", "RMID-u1", None, "user", [{"type": "text", "text": "Fix the clock."}], 1)
    mid = [{"type": "thinking", "text": "look first", "tokens": 3, "done": True, "step": 1},
           _call("call_m1", "read_file", {"path": "clock.html"}, "     1\t<canvas></canvas>", 1,
                 "2026-10-17T12:00:01.000Z"),
           _call("call_m2", "write_file", {"path": "clock.html", "content": LONG},
                 "wrote /workspace/clock.html", 2, "2026-10-17T12:00:02.000Z", risk="write")]
    out["mid_run"] = ("RMID", "RMID-u1", "RMID-a1", mid, 3)
    return out


def build(svc) -> dict:
    """The requests of the stored chats: {name: {"body": the bytes as text, "headers": {...}}}."""
    from chatd import agent
    real = agent._dt
    agent._dt = _Dt
    try:
        out = {}
        for name, (cid, um, am, live, step) in write_chats(svc).items():
            conv = svc.conv(cid)
            conv.live = {"id": am, "parts": json.loads(json.dumps(live)), "status": "streaming"}
            run = Run(svc, conv, user_message_id=um, message_id=am, max_steps=100,
                      started_by=None)
            run.step = step
            body, headers = svc.request_for(run)
            conv.live = None
            out[name] = {"body": body.decode(), "headers": headers}
        return out
    finally:
        agent._dt = real


def chat() -> Chat:
    """A chatd whose model is our engine's (the body's `tandem` field, images inline)."""
    c = Chat()
    c.fake.name = "tandemllm"
    c.svc._caps = None
    return c


if __name__ == "__main__":
    if sys.argv[1:2] != ["--write"] or len(sys.argv) != 3:
        sys.exit("usage: chatd_replay.py --write <path>")
    c = chat()
    try:
        got = build(c.svc)
    finally:
        c.close()
    with open(sys.argv[2], "w") as fh:
        json.dump(got, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"{len(got)} requests -> {sys.argv[2]}")
