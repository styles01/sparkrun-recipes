"""The memory notes on a real engine: what a write and a read cost, and whether the notes are sensible
(2026-10-05, P-M2's first step).

    python tools/memory_measure.py --base http://127.0.0.1:8011 --out ~/m2/measure.json

Against a test engine whose conductor directory is its own (never the service's): it turns the
coding profile's `memory` on and the writer's settings to the values below, then drives one
synthetic opencode-shaped session (a small calculator project, tools answered from a script):

  1  a first message that asks the model to remember two project rules, then tool turns: the
     inline blocks the model writes (count, kinds, scopes, titles: synthetic), TTFT and the
     answer's tokens per turn
  2  the session's end: no request for `idle_s`; the writer's hidden turn (its seconds, prefill
     and decode, tokens, ops) from the journal
  3  a compaction in opencode's shape, served as the continuation; with `after_compaction: turn`
     the writer runs before the next request (its seconds); then the next turn after it (prompt,
     cached and prefilled tokens)
  4  a new session in the same project that asks what the model remembers: the index and the
     matching notes that went in (tokens), TTFT, and whether the answer names the rules
  5  the same first message with memory off, for the TTFT difference of the reads
  6  the decode with memory on and off: one coding question in new sessions, `--decode-pairs`
     alternating pairs, tok/s after the first token (median each)

Synthetic text only. The output is counts, seconds and the synthetic titles; no body is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import compaction  # noqa: E402

PROJECT = "/home/user/demo-calc"
SYSTEM = (f"You are opencode, an interactive CLI coding agent.\nWorking directory: {PROJECT}\n"
          "Platform: linux\nUse the tools to read and write files and run commands.")
TOOLS = [
    {"type": "function", "function": {"name": "write", "description": "Write a file",
                                      "parameters": {"type": "object", "properties": {
                                          "filePath": {"type": "string"},
                                          "content": {"type": "string"}},
                                          "required": ["filePath", "content"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
                                      "parameters": {"type": "object", "properties": {
                                          "command": {"type": "string"}},
                                          "required": ["command"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a file",
                                      "parameters": {"type": "object", "properties": {
                                          "filePath": {"type": "string"}},
                                          "required": ["filePath"]}}}]
FIRST = ("Remember this for the whole project: every function needs a one-line docstring, and the "
         "tests run with `python -m pytest -q` from the project root, never with unittest. Now "
         "create calc.py with add(a, b) and mul(a, b), and a test file for them.")
ASK = "What do you remember about how I want tests run and functions written in this project?"
DECODE_ASK = ("Write a Python function that splits one CSV line into fields, with quoted fields that "
              "may hold commas and doubled quotes. Answer with the code only.")
UA = "opencode/1.18.34 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"
# the conductor's writes take the admin token (from the environment, never printed)
AUTH = {"Content-Type": "application/json",
        **({"Authorization": f"Bearer {os.environ['QSE_ADMIN_TOKEN']}"}
           if os.environ.get("QSE_ADMIN_TOKEN") else {})}


def call(base, method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers=AUTH)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw[:1] in (b"{", b"[") else raw


def chat(base, messages, sid, max_tokens=4000, tools=True):
    """One streamed opencode-shaped request: (content, reasoning, calls, usage, ttft_s, total_s)."""
    body = {"model": "TandemLLM", "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": max_tokens, "messages": messages, "reasoning_effort": "medium",
            "reasoning_format": "reasoning_content"}
    if tools:
        body["tools"] = TOOLS
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": UA,
                                          "x-session-id": sid})
    t0 = time.perf_counter()
    ttft = None
    content, reasoning, calls, usage = "", "", {}, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if line == "data: [DONE]":
                break                   # the client is done here; the server may work on
            if not line.startswith("data: {"):
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")) and \
                        ttft is None:
                    ttft = time.perf_counter() - t0
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "name": "",
                                                              "arguments": ""})
                    fn = tc.get("function") or {}
                    c["name"] += fn.get("name") or ""
                    c["arguments"] += fn.get("arguments") or ""
                    c["id"] = c["id"] or tc.get("id")
    return content, reasoning, list(calls.values()), usage, ttft, time.perf_counter() - t0


def tool_result(call_):
    """The scripted tools: writes succeed, the first pytest run fails once, then passes."""
    name = call_["name"]
    try:
        args = json.loads(call_["arguments"] or "{}")
    except ValueError:
        args = {}
    if name == "write":
        return f"Wrote file successfully: {args.get('filePath')}"
    if name == "read":
        return "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n"
    cmd = str(args.get("command") or "")
    tool_result.runs = getattr(tool_result, "runs", 0) + 1
    if "pytest" in cmd and tool_result.runs == 1:
        return ("F.\n=== FAILURES ===\n___ test_mul ___\n    assert mul(2, 3) == 6\nE   assert 5 == 6\n"
                "1 failed, 1 passed in 0.02s")
    if "pytest" in cmd:
        return "..\n2 passed in 0.01s"
    return "(no output)"


def notes(base):
    return call(base, "GET", "/v1/conductor/memory/notes?user=owner")


def rows(base, what):
    return call(base, "GET", f"/v1/conductor/memory/notes/{what}?limit=50")["rows"]


def setup(base, writer):
    v = call(base, "GET", "/v1/conductor/profiles/coding")
    # the memory is measured, not the reasoning: every class thinks at low effort here
    routing = {c: {"effort": "low"} for c in ("code_edit", "debug", "plan", "chat", "research",
                                               "writing", "summarize")}
    doc = dict(v["doc"], memory="on", routing=routing)
    call(base, "PUT", "/v1/conductor/profiles/coding", {"doc": doc, "note": "memory measure",
                                                        "base_version": v["version"]})
    s = call(base, "GET", "/v1/conductor/settings")
    sd = s["doc"]
    sd.setdefault("memory", {})["notes"] = {"intent": "on", "writer": writer}
    call(base, "PUT", "/v1/conductor/settings", {"doc": sd, "note": "memory measure",
                                                 "base_version": s["version"]})


def set_memory(base, mode):
    v = call(base, "GET", "/v1/conductor/profiles/coding")
    call(base, "PUT", "/v1/conductor/profiles/coding", {"doc": dict(v["doc"], memory=mode),
                                                        "note": f"memory {mode}",
                                                        "base_version": v["version"]})


def session(base, sid, first, turns, out):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first}]
    for t in range(turns):
        content, reasoning, calls, usage, ttft, total = chat(base, msgs, sid)
        out.append({"turn": t + 1, "ttft_s": round(ttft or 0, 2), "total_s": round(total, 2),
                    "prompt": (usage or {}).get("prompt_tokens"),
                    "cached": ((usage or {}).get("prompt_tokens_details") or {}).get(
                        "cached_tokens"),
                    "completion": (usage or {}).get("completion_tokens"),
                    "calls": [c["name"] for c in calls], "block_in_content": "<tandem_memory"
                    in content, "content_chars": len(content)})
        print(f"[measure] {sid} turn {t + 1}: ttft {ttft or 0:.2f} s, {total:.1f} s, "
              f"prompt {out[-1]['prompt']}, cached {out[-1]['cached']}, calls {out[-1]['calls']}",
              flush=True)
        msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        if calls:
            msg["tool_calls"] = [{"id": c["id"] or f"call_{t}_{i}", "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"]}}
                                 for i, c in enumerate(calls)]
        msgs.append(msg)
        if not calls:
            break
        for i, c in enumerate(calls):
            msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{t}_{i}",
                         "content": tool_result(c)})
    return msgs


def wait_writer(base, since_ms, trigger, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for r in rows(base, "writes"):
            d = r.get("detail") or {}
            if r.get("task") == "writer" and r["ts_ms"] >= since_ms and d.get("trigger") == trigger:
                return r
        time.sleep(2)
    try:
        st = call(base, "GET", "/v1/conductor/memory/notes/writes?limit=1").get("writer")
    except Exception as exc:                                       # noqa: BLE001
        st = f"unread: {type(exc).__name__}"
    print(f"[measure] no writer run ({trigger}); the writer's stats: {json.dumps(st)}",
          flush=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    ap.add_argument("--out", required=True)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--idle-s", type=float, default=20.0)
    ap.add_argument("--after-compaction", default="turn", choices=("turn", "idle"))
    ap.add_argument("--decode-pairs", type=int, default=3)
    ap.add_argument("--seed-dir", default="",
                    help="the test engine's memory root, to seed two notes when none was written")
    # M3 (2026-10-05): the writer's own NONE after a compaction (memwriter `ban_tool_call`,
    # `prefill`); `--stop-after compaction` ends after step 3 (the writer runs only, ~4 min)
    ap.add_argument("--ban-tool-call", default="on", choices=("on", "off"))
    ap.add_argument("--prefill", default="")
    ap.add_argument("--stop-after", default="all", choices=("all", "compaction"))
    a = ap.parse_args()
    writer = {"mode": "on", "idle_s": a.idle_s, "due_idle_s": 2.0, "min_new_tokens": 500,
              "max_tokens": 600, "gate": True, "after_compaction": a.after_compaction,
              "ban_tool_call": a.ban_tool_call == "on", "prefill": a.prefill}
    setup(a.base, writer)
    rep = {"writer": writer, "turns": [], "t0": time.time()}
    # 0 a chat without tools: an explicit request to remember, and a correction
    chat0 = []
    m0 = [{"role": "system", "content": SYSTEM},
          {"role": "user", "content": "Please remember for the future: I prefer answers of at "
                                      "most five sentences, and code comments in English."}]
    for follow in ("Also note: in this project we use tabs, not spaces, for indentation.", None):
        content, reasoning, calls, usage, ttft, total = chat(a.base, m0, "ses_mem_c", tools=False)
        chat0.append({"ttft_s": round(ttft or 0, 2), "total_s": round(total, 2),
                      "completion": (usage or {}).get("completion_tokens"),
                      "block_in_content": "<tandem_memory" in content})
        m0 = m0 + [{"role": "assistant", "content": content}]
        if follow is None:
            break
        m0.append({"role": "user", "content": follow})
    w0 = [r for r in rows(a.base, "writes") if r.get("task") == "write"]
    rep["chat_remember"] = {"turns": chat0, "blocks": len(w0),
                            "decisions": [r.get("decision") for r in w0]}
    print(f"[measure] chat remember: {chat0}; blocks {[r.get('decision') for r in w0]}",
          flush=True)
    # 1 the session
    msgs = session(a.base, "ses_mem_a", FIRST, a.turns, rep["turns"])
    ws = [r for r in rows(a.base, "writes") if r.get("task") == "write"]
    ns = notes(a.base)["notes"]
    rep["inline"] = {"blocks": len(ws), "refused": [r["decision"] for r in ws
                                                    if (r.get("detail") or {}).get("why")],
                     "notes": [{"title": n["title"], "kind": n["kind"], "scope": n["scope"],
                                "writer": n["writer"]} for n in ns]}
    print(f"[measure] inline: {len(ws)} blocks, notes {[(n['kind'], n['scope'], n['title']) for n in ns]}",
          flush=True)
    # 2 the session's end
    since = int(time.time() * 1000)
    w = wait_writer(a.base, since, "idle", a.idle_s + 120)
    rep["writer_idle"] = w and w.get("detail")
    print(f"[measure] writer at the session's end: {json.dumps(w and w.get('detail'))}", flush=True)
    # 3 a compaction, then the next turn
    hist = "\n".join(f"[User]: {m['content']}" if m["role"] == "user" else
                     f"[Assistant]: {(m.get('content') or '')[:400]}" if m["role"] == "assistant"
                     else f"[Tool result]: {m['content'][:200]}" for m in msgs[1:])
    comp_body = [{"role": "system", "content": compaction.OPENCODE_SYSTEM},
                 {"role": "user", "content": "Here is the conversation so far:\n\n<conversation>\n"
                                             + hist + "\n</conversation>\n\n"
                                             + compaction.OPENCODE_INSTRUCTIONS}]
    since = int(time.time() * 1000)
    summary, _, _, usage, ttft, total = chat(a.base, comp_body, "ses_mem_a", tools=False)
    rep["compaction"] = {"total_s": round(total, 2), "prompt": (usage or {}).get("prompt_tokens"),
                         "completion": (usage or {}).get("completion_tokens")}
    t_next = time.time()
    after = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "What did we do so far?"},
             {"role": "assistant", "content": summary},
             {"role": "user", "content": "Continue if you have next steps, or stop and ask for "
                                         "clarification if you are unsure how to proceed."}]
    nxt = []
    session_msgs = after
    content, reasoning, calls, usage, ttft, total = chat(a.base, session_msgs, "ses_mem_a")
    nxt.append({"waited_s": round(time.time() - t_next - total, 2), "ttft_s": round(ttft or 0, 2),
                "prompt": (usage or {}).get("prompt_tokens"),
                "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")})
    w = wait_writer(a.base, since, "compaction", 60)
    rep["writer_compaction"] = w and w.get("detail")
    rep["after_compaction"] = nxt
    print(f"[measure] compaction {rep['compaction']}; writer {json.dumps(w and w.get('detail'))}; "
          f"next turn {nxt}", flush=True)
    if a.stop_after == "compaction":
        for k, d in (("idle", rep["writer_idle"]), ("compaction", rep["writer_compaction"])):
            d = d or {}
            print(f"[measure] NONE check {k}: answer={d.get('answer')} none={d.get('none')} "
                  f"cut={d.get('cut')} ops={len(d.get('ops') or [])} ban={d.get('ban')} "
                  f"prefill={d.get('prefill')} tokens={d.get('tokens')} ms={d.get('ms')}",
                  flush=True)
        with open(os.path.expanduser(a.out), "w") as f:
            json.dump(rep, f, indent=1)
        return
    # 4 a new session asks what is remembered; with no note written, two synthetic notes are seeded
    # (as a person's, through the store the engine reads) so the reads are measured all the same
    if not notes(a.base)["notes"] and a.seed_dir:
        from server.conductor.notes import NoteStore
        st = NoteStore(os.path.expanduser(a.seed_dir))
        st.write("owner", {"op": "add", "scope": "project", "kind": "rule",
                           "title": "tests run with python -m pytest -q from the project root",
                           "body": "Never unittest. Run `python -m pytest -q` from the root."},
                 writer="human", project=PROJECT)
        st.write("owner", {"op": "add", "scope": "project", "kind": "rule",
                           "title": "every function has a one-line docstring",
                           "body": "Each function in this project starts with a one-line "
                                   "docstring that says what it returns."},
                 writer="human", project=PROJECT)
        rep["seeded"] = 2
        time.sleep(1.5)
    out_b = []
    content_b = None
    for mode in ("on", "off"):
        set_memory(a.base, mode)
        msgs_b = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK}]
        content, reasoning, calls, usage, ttft, total = chat(a.base, msgs_b, f"ses_mem_b_{mode}",
                                                             tools=False)
        out_b.append({"memory": mode, "ttft_s": round(ttft or 0, 2), "total_s": round(total, 2),
                      "prompt": (usage or {}).get("prompt_tokens"),
                      "completion": (usage or {}).get("completion_tokens"),
                      "decode_tps": round(((usage or {}).get("completion_tokens") or 0)
                                          / max(1e-6, total - (ttft or 0)), 1),
                      "names_pytest": "pytest" in content, "names_docstring": "docstring" in
                      content.lower()})
        if mode == "on":
            content_b = content
    # 5 the decode with memory on and off: one coding question, 3 alternating pairs in new
    # sessions (the notes' prefill is in the TTFT, not in the decode); tok/s after the first token
    pairs = []
    for i in range(a.decode_pairs):
        for mode in ("on", "off"):
            set_memory(a.base, mode)
            msgs_d = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": DECODE_ASK}]
            content, reasoning, calls, usage, ttft, total = chat(
                a.base, msgs_d, f"ses_mem_d{i}_{mode}", max_tokens=600, tools=False)
            n = (usage or {}).get("completion_tokens") or 0
            pairs.append({"memory": mode, "prompt": (usage or {}).get("prompt_tokens"),
                          "completion": n, "ttft_s": round(ttft or 0, 2),
                          "decode_tps": round(n / max(1e-6, total - (ttft or 0)), 1)})
    on = sorted(p["decode_tps"] for p in pairs if p["memory"] == "on")
    off = sorted(p["decode_tps"] for p in pairs if p["memory"] == "off")
    rep["decode"] = {"pairs": pairs, "median_on": on[len(on) // 2] if on else None,
                     "median_off": off[len(off) // 2] if off else None}
    print(f"[measure] decode {rep['decode']}", flush=True)
    inj = rows(a.base, "injected")
    rep["recall"] = out_b
    rep["injected_rows"] = len(inj)
    rep["notes_final"] = [{"title": n["title"], "kind": n["kind"], "scope": n["scope"],
                           "writer": n["writer"], "status": n["status"]}
                          for n in notes(a.base)["notes"]]
    print(f"[measure] recall {out_b}; injected rows {len(inj)}", flush=True)
    print(f"[measure] recall answer (synthetic session): {content_b[:600] if content_b else ''}",
          flush=True)
    with open(os.path.expanduser(a.out), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
