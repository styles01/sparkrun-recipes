"""A recorded Chat-tab conversation replayed through chatd in seconds, no model (SRV-667).

A chat's snapshot (`GET /v1/chat-app/conversations/<id>`, or a replay script's file that holds it
under `conv`) records what the model said at every step: its reasoning, its text and its tool
calls with their arguments. This feeds those steps back to a chatd in front of the test suite's
fake engine (tests/fake_backend.py), turn by turn with the chat's own settings, so every request
chatd builds for that conversation is built again by the code of this tree:

    replay   the conversation through chatd: file tools run on a scratch workspace, bash answers
             with the recorded output and exit code (a fake sandbox), the network and page tools
             with their recorded results; plan mode on for a turn whose message carried the plan
             rule, Approve for "Carry out the approved plan."; `cap` sets the chat's call limit
    check    what chatd must do with it: no write's text replaced by a stub, a history note sent
             as content refused with what is on disk, every request the one before it plus its
             new messages (the engine's prefix cache), the same tool list and system block in and
             out of plan mode, a call cap leaving the chat idle, bash status from the exit code

    python tools/chat_trace.py <trace.json> [--cap N] [--bodies out.jsonl]

`--bodies` writes the requests chatd built, one JSON a line, for tools/prefix_check.py (the same
requests against the live engine) and tools/chat_fork.py (one step sampled again). The methods:
docs/testing.md, "Chat traces". Standard library and the test suite's helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

#: tools whose results come from the record (the network, the page check, the panes, the library)
RECORDED = ("web_fetch", "web_search", "render_page", "present", "library_search", "library_read")
APPROVED = "Carry out the approved plan."


def load(path: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    return d.get("conv") if isinstance(d.get("conv"), dict) else d


def turns(conv: dict) -> list[dict]:
    """The conversation's branch as turns: the user's text, whether it was a plan turn or the
    Approve, and the model's steps (reasoning, text, calls with their recorded results)."""
    out, cur = [], None
    for m in conv.get("messages") or []:
        parts = m.get("parts") or []
        if m.get("role") == "user":
            text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text")
            cur = {"text": text, "plan": any(p.get("type") == "plan" for p in parts),
                   "approve": text.strip() == APPROVED, "steps": []}
            out.append(cur)
        elif m.get("role") == "assistant" and cur is not None:
            by: dict[int, dict] = {}
            for p in parts:
                s = by.setdefault(int(p.get("step") or 1), {"reasoning": "", "content": "",
                                                            "calls": []})
                if p.get("type") == "thinking":
                    s["reasoning"] += p.get("text") or ""
                elif p.get("type") == "text":
                    s["content"] += p.get("text") or ""
                elif p.get("type") == "tool_call" and p.get("name"):
                    try:
                        args = json.loads(p.get("input") or "{}")
                    except ValueError:
                        args = {}
                    out_ = p.get("output") or {}
                    s["calls"].append({"name": p["name"], "arguments": args,
                                       "result": out_.get("content") or "",
                                       "exit_code": out_.get("exit_code"),
                                       "is_error": bool(out_.get("is_error")),
                                       "status": p.get("status")})
            cur["steps"] += [by[k] for k in sorted(by)]
    return out


def _users(body: dict) -> int:
    return sum(1 for m in body.get("messages") or [] if m.get("role") == "user"
               and not str(m.get("content") or "").startswith("The images the tool calls"))


def replay(conv: dict, cap: int | None = None, window: int | None = None) -> dict:
    """The conversation through a chatd of this tree. Returns {"requests", "snapshot", "calls",
    "statuses"}: the bodies chatd sent (in order), the final snapshot, the calls as chatd ran them
    and the chat's status after each turn."""
    from chatd import tools as tools_mod
    from chatd.tools import Result
    from test_chatd_agent import chat_with_sandbox, requests, small_window

    ts = turns(conv)
    steps_by_turn = [list(t["steps"]) for t in ts]
    recorded: dict[str, list] = {n: [] for n in RECORDED}
    bash: list = []
    for t in ts:
        for s in t["steps"]:
            for c in s["calls"]:
                if c["name"] in recorded:
                    recorded[c["name"]].append(c)
                elif c["name"] == "bash":
                    bash.append(c)
    c = chat_with_sandbox()
    saved = {n: tools_mod.BY_NAME[n].executor for n in RECORDED if n in tools_mod.BY_NAME}
    try:
        for n in saved:
            queue = recorded[n]

            def answer(ctx, a, queue=queue):
                got = queue.pop(0) if queue else {"result": "(not in the record)", "is_error": True}
                return Result(got["result"], is_error=got["is_error"])
            tools_mod.BY_NAME[n].executor = answer
        boxes = c.svc.sandboxes
        boxes.run_real = boxes.run

        def run(cid, cmd, *, timeout=600, on_output=None, cancel=None):
            line = str(cmd[-1]) if cmd else ""
            k = next((i for i, x in enumerate(bash)
                      if str(x["arguments"].get("command") or "\0") in line), None)
            if k is None:
                return 0, False                     # chatd's own command (a start, a snapshot)
            got = bash.pop(k)
            text = got["result"]
            for tail in (f"\n[exit code {got.get('exit_code')}]",):
                if text.endswith(tail):
                    text = text[:-len(tail)]
            if on_output and text:
                on_output(text)
            return int(got.get("exit_code") or 0), False
        boxes.run = run
        if window:
            small_window(c, window=window)
        cid = c.new()
        settings = dict(conv.get("settings") or {})
        keep = {k: settings[k] for k in ("effort", "thinking", "tools", "permission_mode",
                                         "verify", "agents", "max_steps") if k in settings}
        keep["verify"] = False                      # the page check is the record's
        if cap:
            keep["max_steps"] = cap
        c.ok("PATCH", f"/conversations/{cid}/settings", keep)
        turn_of = {"n": 0}

        def script(body, headers):
            if headers.get("x-session-id", headers.get("X-Session-Id", "")).endswith("-task"):
                return None
            k = _users(body) - 1
            if k < 0 or k >= len(steps_by_turn):
                return {"content": "done"}
            turn_of["n"] = k
            q = steps_by_turn[k]
            if not q:
                return {"content": "done"}
            s = q.pop(0)
            out = {"content": s["content"]}
            if s["reasoning"]:
                out["reasoning"] = s["reasoning"]
            if s["calls"]:
                out["calls"] = [{"name": x["name"], "arguments": x["arguments"]}
                                for x in s["calls"]]
            return out
        c.fake.script = script
        statuses = []
        for i, t in enumerate(ts):
            if t["approve"]:
                c.ok("POST", f"/conversations/{cid}/plan/approve", {})
            else:
                if t["plan"]:
                    c.ok("PATCH", f"/conversations/{cid}/settings", {"plan_mode": True})
                c.send(cid, t["text"])
            snap = c.wait_idle(cid, 120)
            statuses.append(snap.get("status"))
            if cap and snap["messages"][-1].get("status") == "interrupted":
                break
        snap = c.snap(cid)
        made = [p for m in snap["messages"] if m["role"] == "assistant"
                for p in m["parts"] if p.get("type") == "tool_call"]
        return {"requests": requests(c), "snapshot": snap, "calls": made, "statuses": statuses}
    finally:
        for n, ex in saved.items():
            tools_mod.BY_NAME[n].executor = ex
        c.close()


def _prefix_kept(prev: dict, cur: dict) -> bool:
    n = len(prev["messages"])
    enc = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)  # noqa: E731
    return (enc(cur["messages"][:n]) == enc(prev["messages"]) and
            enc(cur.get("tools")) == enc(prev.get("tools")))


def check(res: dict, cap: int | None = None) -> list[str]:
    """What chatd must do with the conversation; a list of failures (empty: all held)."""
    from chatd import render
    bad = []
    reqs = res["requests"]
    done = {p["id"]: p for p in res["calls"]}
    for k, b in enumerate(reqs):
        for m in b["messages"]:
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                if fn["name"] != "write_file":
                    continue
                part = done.get(tc["id"])
                if part is None or part.get("status") != "done":
                    continue
                try:
                    a = json.loads(fn["arguments"])
                except ValueError:
                    bad.append(f"request {k}: write {tc['id']} arguments are not JSON")
                    continue
                c = a.get("content")
                if isinstance(c, str) and (render.WROTE_STUB.match(c.strip()) or
                                           render.FOLDED_NOTE.match(c.strip())):
                    bad.append(f"request {k}: write {tc['id']} shows a stub as its content")
    for p in res["calls"]:
        c = (json.loads(p.get("input") or "{}") if p.get("name") == "write_file" else {})
        c = c.get("content") if isinstance(c, dict) else None
        if isinstance(c, str) and (render.WROTE_STUB.match(c.strip()) or
                                   render.FOLDED_NOTE.match(c.strip())):
            out = (p.get("output") or {}).get("content") or ""
            if p.get("status") != "failed" or "nothing was written" not in out:
                bad.append(f"call {p['id']}: a note sent as content was not refused with what is "
                           f"on disk ({out[:80]!r})")
        if p.get("name") == "bash" and p.get("status") in ("done", "failed"):
            code = (p.get("output") or {}).get("exit_code")
            want = "failed" if code else "done"
            if code is not None and p["status"] != want:
                bad.append(f"call {p['id']}: bash exit {code} marked {p['status']}")
    for k in range(1, len(reqs)):
        if not _prefix_kept(reqs[k - 1], reqs[k]):
            notes = sum("is left out of this conversation" in str(m.get("content") or "")
                        for m in reqs[k]["messages"])
            summary = "Summary of the earlier part" in str(reqs[k]["messages"][0].get("content"))
            if not summary and not notes:
                bad.append(f"request {k} does not start with request {k - 1} (prefix changed)")
    if cap and res["snapshot"]["messages"][-1].get("status") == "interrupted" and \
            res["snapshot"].get("status") != "idle":
        bad.append(f"the call cap left the chat {res['snapshot'].get('status')}, not idle")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--bodies", default=None)
    a = ap.parse_args(argv)
    conv = load(a.trace)
    res = replay(conv, cap=a.cap, window=a.window)
    if a.bodies:
        with open(a.bodies, "w") as fh:
            for b in res["requests"]:
                fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    bad = check(res, a.cap)
    ts = turns(conv)
    print(f"{a.trace}: {len(ts)} turns, {sum(len(t['steps']) for t in ts)} recorded steps, "
          f"{len(res['requests'])} requests, {len(res['calls'])} calls, statuses {res['statuses']}")
    for line in bad:
        print("FAIL", line)
    print("ok" if not bad else f"{len(bad)} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
