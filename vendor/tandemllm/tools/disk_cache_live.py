#!/usr/bin/env python3
"""C1's live check of the disk tier on the served engine, after the deploy (no hold; stdlib only).
The server must run with `QSE_DISK_CACHE` set and the Disk cache tab's mode on.

    python3 tools/disk_cache_live.py [--server http://127.0.0.1:8000] [--head-tokens 24000]
        [--token-file ~/tandem-staircut/ops/secrets.env] [--out results/disk/live-<time>.json]

Run it on the box. It refuses while the box lock is held (a hold runs) and sends one request at a
time, every one bounded. It empties the caches in RAM once (`POST /v1/cache/clear`, the RAM tier
only), so a session that is running re-reads its next turn -- from the disk where it can.

An opencode-shaped prompt: a synthetic head of about `--head-tokens` tokens (a system text and a
tool block, with a nonce at its start, so nothing on the box holds it yet), then one user message.
Through TandemLLM, greedy, `max_tokens` 96, opencode's user agent and a session id of its own:

  1. cold     session A's first request: nothing holds the head; the tier writes it (the first
              long request of a session) behind the request.
  2. the writer is waited for (the Disk cache tab's report lists an opencode head of this length),
     then the caches in RAM are emptied.
  3. disk     session B's first request, the same prompt: it starts from the disk.
  4. other    session C's first request, the same head with another user message: it starts from
              the disk too (the case the tier exists for); no text to compare it with.

The gates: run 1 wrote its head (an opencode-head entry written during this check, complete --
every chunk below its anchors on disk -- with no writer error since the check began); runs 1 and 3
finished without an error and wrote something; run 3's `cache_source` is `disk` and it reused at
least that entry's tokens; run 3's text and tool calls equal run 1's (greedy). Run 4 is reported, not gated: the router may give its
message another effort, and the effort's sentence is at the head of the prompt (docs/server.md), so
its head is then another one. Printed: the TTFT of each run (the client's and the server's), the
prompt and reused tokens, the restore's time and read rate from the tier's events. Exit 0 when the
gates hold.

What is compared is what the model wrote: the stream's text without TandemLLM's step lines (S1's
`› reading 16,427 tokens at ~800 tok/s, about 18 s left`, `› waiting: ...`, which say how long
something took and differ between two identical answers; server/conductor/states.py `strip_text`),
and each tool call's name and arguments, not the call id the server draws per request. On
2026-10-02 at 22:08 the gate compared the raw stream: the cold run's reading lines
and the disk run's waiting line, and two calls' ids, made `same_text` false on the same answer
and the same calls. The server's TTFT is the one that counts: the client's first piece is a step
line.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import importlib.util
import json
import os
import random
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
OPENCODE_UA = "opencode/1.18.33 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"
LOCK = "~/.qwen38-box.flock"
CHARS_PER_TOKEN = 3.6
WORDS = ("file", "edit", "read", "write", "test", "build", "check", "path", "line", "token",
         "module", "function", "class", "argument", "result", "error", "output", "input", "tool",
         "command", "project", "directory", "branch", "commit", "patch", "diff", "search", "match")


def held() -> bool:
    p = os.path.expanduser(LOCK)
    if not os.path.exists(p):
        return False
    fd = os.open(p, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def admin_token(path: str) -> str | None:
    """The admin token from secrets.env (never printed); None from the box itself works too."""
    try:
        with open(os.path.expanduser(path)) as f:
            for line in f:
                if line.startswith("QSE_ADMIN_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    except OSError:
        return None
    return None


def _states():
    """server/conductor/states.py by its path (it needs nothing but `re`): this file runs on the
    box's own python, without the engine's packages."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server",
                        "conductor", "states.py")
    spec = importlib.util.spec_from_file_location("_tandem_states", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STATES = _states()


def model_text(text: str) -> str:
    """The stream's text without TandemLLM's step lines."""
    return STATES.strip_text(text)


def call_list(deltas: list) -> list:
    """The streamed tool-call deltas as whole calls, (index, name, arguments), without the id the
    server draws for each call."""
    calls: dict = {}
    for d in deltas:
        for c in json.loads(d) if isinstance(d, str) else d:
            k = calls.setdefault(c.get("index", 0), {"name": "", "arguments": ""})
            fn = c.get("function") or {}
            k["name"] += fn.get("name") or ""
            k["arguments"] += fn.get("arguments") or ""
    return [[i, v["name"], v["arguments"]] for i, v in sorted(calls.items())]


def head(n_tokens: int, nonce: str) -> tuple[str, list]:
    """A system text and a tool block shaped like opencode's, about `n_tokens` long."""
    rng = random.Random(7)
    tools = []
    for i in range(16):
        desc = " ".join(rng.choice(WORDS) for _ in range(160))
        tools.append({"type": "function", "function": {
            "name": f"tool_{i:02d}", "description": f"Tool {i}: {desc}.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "the file path"},
                "text": {"type": "string", "description": "what to write"}},
                "required": ["path"]}}})
    tool_chars = len(json.dumps(tools))
    rules, k = [f"You are a coding agent (live check {nonce}). Follow these rules."], 0
    while sum(len(r) for r in rules) + tool_chars < n_tokens * CHARS_PER_TOKEN:
        k += 1
        rules.append(f"Rule {k}: " + " ".join(rng.choice(WORDS) for _ in range(40)) + ".")
    return "\n".join(rules), tools


def one(server: str, system: str, tools: list, user: str, session: str) -> dict:
    body = {"model": "TandemLLM", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "tools": tools, "temperature": 0, "seed": 7, "max_tokens": 96,
            "reasoning_format": "reasoning_content"}
    req = urllib.request.Request(server + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": OPENCODE_UA, "x-session-id": session})
    t0 = time.monotonic()
    first, text, calls, usage, timings, finish = None, [], [], None, None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            ln = raw.decode().strip()
            if not ln.startswith("data: {"):
                continue
            ev = json.loads(ln[6:])
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                finish = ch.get("finish_reason") or finish
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if d.get("tool_calls"):
                    calls.append(json.dumps(d["tool_calls"], sort_keys=True))
                if piece or d.get("tool_calls"):
                    first = first if first is not None else time.monotonic()
                    text.append(piece)
    t = timings or {}
    text = "".join(text)
    return {"session": session, "text": text, "model_text": model_text(text),
            "calls": calls, "call_list": call_list(calls), "finish": finish,
            "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
            "reused": int(t.get("cache_n") or 0), "cache_source": t.get("cache_source"),
            "ttft_ms_client": round((first - t0) * 1e3, 1) if first is not None else None,
            "ttft_ms_server": t.get("ttft_ms"), "prefill_tps": t.get("prompt_per_second")}


def call(server: str, method: str, path: str, token: str | None, body=None) -> dict:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(server + path, method=method, headers=h,
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=BASE)
    ap.add_argument("--head-tokens", type=int, default=24000)
    ap.add_argument("--token-file", default="~/tandem-staircut/ops/secrets.env")
    ap.add_argument("--wait-s", type=float, default=120.0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    server = a.server.rstrip("/")
    if held():
        print("[disk-live] the box lock is held (a hold runs): not now", flush=True)
        return 2
    token = admin_token(a.token_file)
    disk = call(server, "GET", "/v1/conductor/memory/disk", token)
    if not disk.get("running") or not disk.get("on"):
        print(f"[disk-live] the disk tier does not run on {server} (running {disk.get('running')}, "
              f"mode {disk.get('settings', {}).get('mode')})", flush=True)
        return 2
    nonce = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    system, tools = head(a.head_tokens, nonce)
    user1 = "Add a --verbose flag to the command line tool and print each file it reads."
    user2 = "Rename the function parse_line to parse_record everywhere in the project."
    runs = {}
    t_start = time.time()
    runs["cold"] = one(server, system, tools, user1, f"disk-live-{nonce}-a")
    print(f"[disk-live] cold: {runs['cold']['prompt_tokens']} prompt tokens, reused "
          f"{runs['cold']['reused']} ({runs['cold']['cache_source']}), TTFT "
          f"{runs['cold']['ttft_ms_client']} ms", flush=True)
    # session A's head: its anchors below where the user message starts (a note the router splices
    # in sits after it), written behind the request; the longest of them once the writer is idle
    # the writer's own event says it (the top-20 list is sorted by hits, where a new entry may not
    # show on a busy cache): a write of an opencode head since the start, nothing dropped
    deadline, entry = time.time() + a.wait_s, None
    while time.time() < deadline:
        rep = call(server, "GET", "/v1/conductor/memory/disk", token)
        done = [e for e in rep.get("events") or [] if e.get("kind") == "write"
                and e.get("ts", 0) >= t_start - 1 and "opencode-head" in (e.get("kinds") or [])
                and not e.get("dropped")]
        if done and not rep.get("open_jobs"):
            top = max(done, key=lambda e: e["tokens"])
            entry = next((r for r in rep.get("top") or [] if r["tokens"] == top["tokens"]
                          and r["kind"] == "opencode-head"),
                         {"tokens": top["tokens"], "reads_bytes": None, "event": top})
            # 2026-10-02 21:09: a writer that failed part way left anchors without their chunks
            # (`complete` false), which no lookup can read; its error is an event of its own
            entry["writer_errors"] = [str(e.get("error"))[:200] for e in rep.get("events") or []
                                      if e.get("kind") == "error"
                                      and e.get("ts", 0) >= t_start - 1]
            break
        time.sleep(2)
    print(f"[disk-live] written: {entry and entry['tokens']} tokens"
          + (f", a hit reads {round(entry['reads_bytes'] / 1e9, 2)} GB"
             if entry and entry.get("reads_bytes") else ""), flush=True)
    call(server, "POST", "/v1/cache/clear", token, {})
    runs["disk"] = one(server, system, tools, user1, f"disk-live-{nonce}-b")
    runs["other"] = one(server, system, tools, user2, f"disk-live-{nonce}-c")
    rep = call(server, "GET", "/v1/conductor/memory/disk", token)
    hits = [e for e in rep.get("events") or [] if e.get("kind") == "hit"][-2:]
    for k in ("disk", "other"):
        r = runs[k]
        print(f"[disk-live] {k}: reused {r['reused']} of {r['prompt_tokens']} "
              f"({r['cache_source']}), TTFT {r['ttft_ms_client']} ms client, "
              f"{r['ttft_ms_server']} ms server", flush=True)
    for e in hits:
        print(f"[disk-live] restore: {e.get('tokens')} tokens, {e.get('read')} read, "
              f"{round((e.get('bytes') or 0) / 1e9, 2)} GB in {e.get('ms')} ms", flush=True)
    gates = {
        "head_written": entry is not None and entry.get("complete") is not False
        and not entry.get("writer_errors"),
        "answered": all(runs[k]["finish"] not in (None, "error")
                        and bool(runs[k]["text"] or runs[k]["calls"]) for k in ("cold", "disk")),
        "disk_hit": runs["disk"]["cache_source"] == "disk"
        and entry is not None and runs["disk"]["reused"] >= entry["tokens"],
        "same_text": (runs["disk"]["model_text"], runs["disk"]["call_list"])
        == (runs["cold"]["model_text"], runs["cold"]["call_list"]),
    }
    ok = all(gates.values())
    out = {"ts": nonce, "server": server, "head_tokens": a.head_tokens, "entry": entry,
           "runs": runs, "restores": hits, "gates": gates, "pass": ok,
           "other_session_from_disk": runs["other"]["cache_source"] == "disk",
           "ttft_ratio": (round(runs["cold"]["ttft_ms_server"] / runs["disk"]["ttft_ms_server"], 1)
                          if runs["disk"]["ttft_ms_server"] and runs["cold"]["ttft_ms_server"]
                          else None)}
    path = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "results", "disk", f"live-{nonce}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[disk-live] gates {gates}; TTFT server cold {runs['cold']['ttft_ms_server']} ms -> "
          f"disk {runs['disk']['ttft_ms_server']} ms (client, a step line first: "
          f"{runs['cold']['ttft_ms_client']} -> {runs['disk']['ttft_ms_client']} ms); "
          f"{'PASS' if ok else 'FAIL'} ({path})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
