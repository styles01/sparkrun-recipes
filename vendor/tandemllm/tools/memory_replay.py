"""Replay a real session's transcript into an engine with memory on, so the background writer
writes notes from a real session (OPS-55).

    python tools/memory_replay.py --base http://127.0.0.1:8000 \
        --transcript ~/.qwen38-spark-engine/memory/sessions/2026-10/c-ses_.../transcript.md \
        --session ops55-1 --out DIR [--wait 420]

The transcript (server/conductor/transcripts.py) is the session's record as the runtime kept it:
each turn's user message, tool results, assistant text and tool calls, the long ones clipped.
It becomes one chat request: an opencode-shaped system prompt (the working directory read from
the tool calls' paths), the turns as OpenAI messages (a clipped tool call's arguments that no
longer parse are kept as a note with their head), up to the last user or tool message. It goes
to the engine with `X-Session-Id` and the coding profile (memory on), a short answer, thinking
off. Then nothing is sent, so the session goes idle and the writer's look runs its hidden turn on
the conversation the engine holds; the script waits for that run in the writes journal and
reports it (trigger, ops, refusals, seconds).

The request carries the user's real text from the transcript, so it stays on this host; the
report holds counts and note ids only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

TURN = re.compile(r"^## turn \d+ .*$", re.M)
SECTION = re.compile(r"^### (.+)$", re.M)
PATH = re.compile(r'"(?:filePath|path|workdir)":\s*"(/[^"]+)"')


def parse(text: str) -> tuple[list, str | None]:
    """(messages, working directory) from a transcript."""
    msgs: list = []
    workdir = None
    n_call = 0
    for t, chunk in enumerate(TURN.split(text)[1:], 1):
        parts = SECTION.split(chunk)
        calls, said = [], []
        for head, body in zip(parts[1::2], parts[2::2]):
            body = body.strip("\n")
            h = head.strip()
            if h == "user":
                msgs.append({"role": "user", "content": body})
            elif h.startswith("tool result"):
                prev = next((m for m in reversed(msgs) if m["role"] == "assistant"
                             and m.get("tool_calls")), None)
                done = sum(1 for m in msgs if m["role"] == "tool")
                ids = [c["id"] for m in msgs if m["role"] == "assistant"
                       for c in m.get("tool_calls") or []]
                if prev is None or done >= len(ids):
                    continue
                msgs.append({"role": "tool", "tool_call_id": ids[done], "content": body})
            elif h.startswith("tool call"):
                name = re.match(r"tool call \(([^)]+)\)", h)
                name = name.group(1) if name else "tool"
                try:
                    args = json.loads(body)
                    arg_s = json.dumps(args)
                except ValueError:
                    arg_s = json.dumps({"note": "arguments clipped in the transcript",
                                        "head": body[:2000]})
                if workdir is None:
                    m = PATH.search(body)
                    if m:
                        workdir = os.path.dirname(m.group(1)) if "." in os.path.basename(
                            m.group(1)) else m.group(1)
                n_call += 1
                calls.append({"id": f"call_{t}_{n_call}", "type": "function",
                              "function": {"name": name, "arguments": arg_s}})
            elif h == "assistant":
                if body and body != "(no text, no tool call)":
                    said.append(body)
        if said or calls:
            m = {"role": "assistant", "content": "\n\n".join(said) or None}
            if calls:
                m["tool_calls"] = calls
            msgs.append(m)
    # every tool call answered (a call the transcript has no result for gets an empty one)
    out, pending = [], []
    for m in msgs:
        if m["role"] != "tool" and pending:
            out += [{"role": "tool", "tool_call_id": i, "content": "(no result recorded)"}
                    for i in pending]
            pending = []
        if m["role"] == "assistant":
            pending = [c["id"] for c in m.get("tool_calls") or []]
        elif m["role"] == "tool" and m["tool_call_id"] in pending:
            pending.remove(m["tool_call_id"])
        out.append(m)
    while out and out[-1]["role"] == "assistant":
        out.pop()
    return out, workdir


def _req(base, path, body=None, headers=None, timeout=900):
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    r = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                               headers=h, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return f.status, f.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait", type=float, default=420.0)
    ap.add_argument("--max-messages", type=int, default=0, help="keep the last N (0: all)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    admin = {}
    sec = os.path.expanduser("~/.qwen38-spark-engine/secrets.env")
    if os.path.exists(sec):
        for line in open(sec):
            if line.startswith("QSE_ADMIN_TOKEN="):
                admin = {"Authorization": "Bearer " + line.split("=", 1)[1].strip().strip('"')}
    msgs, workdir = parse(open(os.path.expanduser(a.transcript)).read())
    if a.max_messages and len(msgs) > a.max_messages:
        msgs = msgs[-a.max_messages:]
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
    workdir = workdir or "/home/user/project"
    system = (f"You are opencode, an interactive CLI coding agent.\nWorking directory: {workdir}\n"
              "Platform: linux\nUse the tools to read and write files and run commands.")
    tools = sorted({c["function"]["name"] for m in msgs for c in m.get("tool_calls") or []})
    body = {"model": "TandemLLM", "stream": False, "max_tokens": 32, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "tools": [{"type": "function", "function": {"name": n, "parameters": {
                "type": "object", "properties": {}}}} for n in tools],
            "messages": [{"role": "system", "content": system}, *msgs]}
    hdr = {"User-Agent": "opencode/1.18.34 ai-sdk/provider-utils/3 (ops55-replay)",
           "X-Session-Id": a.session, "X-Tandem-Profile": "coding"}
    rep = {"session": a.session, "transcript": os.path.basename(os.path.dirname(a.transcript)),
           "messages": len(msgs), "users": sum(1 for m in msgs if m["role"] == "user"),
           "tool_calls": sum(len(m.get("tool_calls") or []) for m in msgs), "workdir": workdir}
    _, before = _req(a.base, "/v1/conductor/memory/notes/writes?limit=50", headers=admin)
    seen = {json.dumps(r, sort_keys=True) for r in json.loads(before).get("rows", [])}
    t0 = time.time()
    try:
        st, out = _req(a.base, "/v1/chat/completions", body, hdr)
        u = json.loads(out).get("usage") or {}
        rep.update(status=st, prompt_tokens=u.get("prompt_tokens"),
                   completion_tokens=u.get("completion_tokens"),
                   request_s=round(time.time() - t0, 1))
    except urllib.error.HTTPError as e:
        rep.update(status=e.code, error=e.read()[:300].decode("utf-8", "replace"))
        print(json.dumps(rep), flush=True)
        json.dump(rep, open(os.path.join(a.out, f"{a.session}.json"), "w"), indent=1)
        return 1
    print(f"[replay] {a.session}: {rep['messages']} messages, prompt {rep.get('prompt_tokens')} "
          f"tokens in {rep['request_s']} s; waiting for the writer", flush=True)
    new, t_end = [], time.time() + a.wait
    while time.time() < t_end:
        time.sleep(10)
        _, body_w = _req(a.base, "/v1/conductor/memory/notes/writes?limit=50", headers=admin)
        w = json.loads(body_w)
        new = [r for r in w.get("rows", []) if json.dumps(r, sort_keys=True) not in seen]
        if any(r.get("task") == "writer" for r in new):
            break
    rep["writer"] = [r for r in new if r.get("task") == "writer"]
    rep["writes"] = len(new)
    rep["waited_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(a.out, f"{a.session}.json"), "w"), indent=1, default=str)
    print(f"[replay] {a.session}: writer runs {len(rep['writer'])}, journal rows {len(new)}, "
          f"{rep['waited_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
