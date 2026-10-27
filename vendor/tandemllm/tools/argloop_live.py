"""The tool-argument loop guard on the live engine: one opencode-shaped turn whose `write` is
asked to repeat one map row hundreds of times (the 2026-10-04 shape, on purpose), then the next
turn of the same session.

    python tools/argloop_live.py --base http://127.0.0.1:8000 --key d7-test [--admin TOKEN]

Prints JSON: the first turn's finish, the stop sentence, tokens and seconds; whether a call
reached the client; the second turn's monitor rows (A1's GIVE_UP on the first request, A1's
PROMPT_TAIL_NOTE on the second). Streams, as opencode does. The ask is a legit request whose
honest answer IS a periodic write, so the guard firing here is the mechanism, not a false
positive measurement (that is tools/argloop_fp.py).

    python tools/argloop_live.py --owui [--attempts 5] [--effort none]

`--owui`: Open WebUI's agent shape (aiohttp's user agent, the chat profile, a
`write_file` tool, `reasoning_effort` as given): the write asks for one 12-row sprite frame of about
400 characters written out 40 times (a long unit), and every answer that carries a call is
answered as Open WebUI's tool loop does -- the tool's error, the conversation sent again with the
call's arguments as streamed -- up to `--attempts` times. Prints each attempt's finish, call,
prompt tokens and seconds, the sentence, and A1's rows: the stop, the retry's note, the give-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

UA = "opencode/1.18.34 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"
ROW = '"' + " " * 94 + '",'
WRITE = {"type": "function", "function": {
    "name": "write", "description": "Write a file to the local filesystem.",
    "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string", "description": "absolute path"},
        "content": {"type": "string", "description": "the file's content"}},
        "required": ["filePath", "content"]}}}
READ = {"type": "function", "function": {
    "name": "read", "description": "Read a file.",
    "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}},
                   "required": ["filePath"]}}}


def system_text(chars: int = 24000) -> str:
    """A system text of `chars` characters (the session is keyed by `x-session-id`)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = "You are opencode, an interactive CLI coding agent.\n<env>\n  Working directory: " \
           "/tmp/d7-guard\n</env>\n\n"
    with open(os.path.join(here, "docs", "server.md")) as f:
        text += f.read()[:chars]
    return text


def stream(base, key, body, sid, headers=None):
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
         "User-Agent": UA, "x-session-id": sid, "X-Tandem-Profile": "coding"}
    if headers is not None:
        h = dict({"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, **headers)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=h)
    out = {"content": "", "reasoning": "", "calls": 0, "finish": None, "id": None, "call": None}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data: {"):
                continue
            ch = json.loads(ln[6:])
            out["id"] = out["id"] or ch.get("id")
            if ch.get("usage"):
                out["usage"] = ch["usage"]
            for c in ch.get("choices") or ():
                d = c.get("delta") or {}
                out["content"] += d.get("content") or ""
                out["reasoning"] += d.get("reasoning_content") or ""
                out["calls"] += sum(1 for tc in d.get("tool_calls") or () if tc.get("id"))
                for tc in d.get("tool_calls") or ():
                    c = out["call"] = out["call"] or {"id": None, "name": "", "arguments": ""}
                    c["id"] = tc.get("id") or c["id"]
                    fn = tc.get("function") or {}
                    c["name"] += fn.get("name") or ""
                    c["arguments"] += fn.get("arguments") or ""
                out["finish"] = c.get("finish_reason") or out["finish"]
    out["s"] = round(time.time() - t0, 1)
    return out


def events(base, admin, since_ms):
    h = {"Authorization": f"Bearer {admin}"} if admin else {}
    req = urllib.request.Request(f"{base}/v1/conductor/events?since={since_ms}&limit=500",
                                 headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        got = json.loads(r.read())
    rows = got.get("events") or got.get("rows") or []
    return [{k: e.get(k) for k in ("request_id", "task", "decision", "reason", "detail")}
            for e in rows if e.get("task") == "A1"]


WRITE_FILE = {"type": "function", "function": {
    "name": "write_file", "description": "Write a file in the workspace.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}}
OWUI_UA = "Python/3.11 aiohttp/3.12.15"


def _tool_result(arguments: str) -> str:
    """What Open WebUI's tool answers: the write when the arguments parse, else the error."""
    try:
        json.loads(arguments)
    except ValueError:
        return "Error: the tool arguments are not valid JSON"
    return "ok: the file was written"


def owui(a) -> int:
    """Open WebUI's agent loop against a forced long-unit loop (see the module's docstring)."""
    sid = f"owui-loop-{int(time.time())}"
    since = int(time.time() * 1000)
    frame = "\n".join("  \"" + "".join(".rgby"[(i * 7 + j * 3) % 5] for j in range(28)) + "\","
                      for i in range(12))
    user = ("Use the write_file tool to create sprites.js. It must start with `const FRAMES = [` and "
            "then contain this frame, exactly as given, written out in full 40 times one after the "
            "other, each copy followed by a line `],[`, then `];`. Write every copy out by hand in "
            "the content argument; do not use a loop or code to generate them. Call the tool now."
            "\n\nThe frame:\n" + frame)
    msgs = [{"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": user}]
    body = {"model": "TandemLLM", "tools": [WRITE_FILE], "stream": True, "max_tokens": 32768,
            "stream_options": {"include_usage": True}}
    if a.effort:
        body["reasoning_effort"] = a.effort
    hdr = {"User-Agent": OWUI_UA, "x-session-id": sid}
    rep = {"session": sid, "effort": a.effort, "attempts": []}
    for k in range(a.attempts):
        got = stream(a.base, a.key, dict(body, messages=msgs), sid, headers=hdr)
        u = got.get("usage") or {}
        rep["attempts"].append({
            "id": got["id"], "finish": got["finish"], "call": bool(got["call"]),
            "call_args_chars": len((got["call"] or {}).get("arguments") or ""),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "s": got["s"], "sentence": got["content"].strip()[-420:]})
        c = got["call"]
        if not c:
            break
        msgs = msgs + [{"role": "assistant", "content": got["content"], "tool_calls": [
            {"id": c["id"] or f"call_{k}", "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}]},
            {"role": "tool", "tool_call_id": c["id"] or f"call_{k}",
             "content": _tool_result(c["arguments"])}]
    try:
        rep["a1_rows"] = events(a.base, a.admin, since)
    except Exception as exc:                                       # noqa: BLE001
        rep["a1_rows"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(rep, indent=1))
    n = len(rep["attempts"])
    ok = n < a.attempts or not rep["attempts"][-1]["call"]
    print(f"[argloop owui] {'PASS' if ok else 'FAIL'}: {n} attempts, last "
          f"{'with' if rep['attempts'][-1]['call'] else 'without'} a call", file=sys.stderr)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="d7-test")
    ap.add_argument("--admin", default=os.environ.get("QSE_ADMIN_TOKEN"))
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--system-chars", type=int, default=24000,
                    help="keep it small on a box someone is using")
    ap.add_argument("--owui", action="store_true", help="Open WebUI's agent loop (0.3.40-test)")
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--effort", default="none", help="--owui: reasoning_effort ('' for none sent)")
    a = ap.parse_args()
    if a.owui:
        return owui(a)
    sid = f"d7-argloop-{int(time.time())}"
    since = int(time.time() * 1000)
    user = (f"Use the write tool to create /tmp/d7-guard/level.js. It must contain `const LEVEL "
            f"= [` on the first line, then exactly {a.rows} lines that are each exactly {ROW} "
            f"(a double quote, 94 spaces, a double quote, a comma), then `];`. Write every one "
            f"of the {a.rows} lines out in full in the content argument; do not generate them "
            f"with code. Call the tool now.")
    msgs = [{"role": "system", "content": system_text(a.system_chars)}, {"role": "user", "content": user}]
    body = {"model": "TandemLLM", "messages": msgs, "tools": [WRITE, READ], "stream": True,
            "max_tokens": 65536, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    first = stream(a.base, a.key, body, sid)
    rep = {"session": sid, "first": {k: first.get(k) for k in ("id", "finish", "calls", "s",
                                                                 "usage")},
           "stop_sentence": first["content"].strip()[-400:]}
    msgs2 = msgs + [{"role": "assistant", "content": first["content"]},
                    {"role": "user", "content": "Go on."}]
    second = stream(a.base, a.key, dict(body, messages=msgs2, max_tokens=4096), sid)
    rep["second"] = {k: second.get(k) for k in ("id", "finish", "calls", "s")}
    try:
        rep["a1_rows"] = events(a.base, a.admin, since)
    except Exception as exc:                                       # noqa: BLE001
        rep["a1_rows"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(rep, indent=1))
    # a held call never reaches the client; a call that streams live (opencode's coding profile)
    # keeps the one call it was sent part of, and the block is never sent again as
    # content
    live = any(isinstance(r, dict) and (r.get("detail") or {}).get("streamed")
               for r in rep["a1_rows"] if isinstance(rep["a1_rows"], list))
    ok = ("[TandemLLM stopped the" in first["content"] and first["finish"] == "stop"
          and first["calls"] <= (1 if live else 0) and ROW * 3 not in first["content"])
    print(f"[argloop] {'PASS' if ok else 'FAIL'}: calls {first['calls']} "
          f"({'streamed live' if live else 'held'}), finish {first['finish']}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
