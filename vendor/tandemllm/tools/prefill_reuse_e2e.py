"""SRV-546: does a long Chat-tab run keep its prefix across screenshots and a parallel chat?

    python tools/prefill_reuse_e2e.py [--url http://127.0.0.1:8000] [--ledger PATH]
        [--history-tokens 100000] [--turns 6] [--out results/prefill-reuse/<utc>.json]

Builds a tool run the way chatd renders it (chatd/render.py: the same system block, tool schemas,
reasoning, tool results, screenshots in their tool messages), about `--history-tokens` long, and
then runs live turns against the engine, each one the previous request plus the model's own answer
plus a tool result, as chatd's agent loop does:

  A  the long history, text only (a cold prefill the first time)
  B  a render_page result with a screenshot (the first image of the run)
  C  a text tool result; the screenshot stays in the prompt
  P  a parallel chat in another session while C decodes (it must take a side slot)
  D  a second screenshot (both stay: append-only)
  E  a text tool result after it

For every turn the ledger row (by the response's id) gives what the engine reused and read: a turn
passes when it forwarded at most what is new since the previous turn's end (its tool result, the
router's note of a new user turn, the glue, the last decided token) plus 64, i.e. it resumed at the end of the previous generation.
Reports reuse = engine_cached / (engine_cached + forwarded) per turn. Text only, no personal data:
the history is generated filler, the screenshots are generated PNGs. Never run it while the box
serves someone: it takes slot 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
import threading
import time
import urllib.request
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chatd import render  # noqa: E402

TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command in /workspace and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "render_page",
        "description": "Open a page in headless Chromium; returns console output and a screenshot.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
]

WORDS = ("const", "scene", "mesh", "light", "update", "camera", "material", "return", "group",
         "vector", "pond", "petal", "roof", "tier", "beam", "water", "shader", "uniform", "time")


def png(w: int, h: int, seed: int) -> bytes:
    """A real PNG, w x h, horizontal bands whose colours follow `seed` (each seed its own image)."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    rows = []
    for y in range(h):
        c = (seed * 37 + (y // 40) * 53) % 256
        rows.append(b"\x00" + bytes((c, (c * 3) % 256, (c * 7) % 256)) * w)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def filler(n_words: int, seed: int) -> str:
    out, x = [], seed
    for i in range(n_words):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(WORDS[x % len(WORDS)] + ("();\n    " if i % 9 == 8 else " "))
    return "".join(out)


def history(tokens: int) -> list[dict]:
    """One user message and one assistant message of tool steps (chatd's parts), ~`tokens` long."""
    parts, step, n = [], 1, 0
    while n < tokens:
        body = filler(1400, step)
        parts += [{"type": "thinking", "text": f"Write module {step} of the scene.", "step": step},
                  {"type": "text", "text": f"Writing js/m{step}.js.", "step": step},
                  {"type": "tool_call", "id": f"call_h{step}", "name": "bash", "step": step,
                   "input": json.dumps({"command": f"cat > js/m{step}.js <<'EOF'\n{body}\nEOF"}),
                   "status": "done", "output": {"content": f"m{step}.js written"}}]
        n += 2400
        step += 1
    return [{"id": "u", "role": "user", "parts": [{"type": "text", "text": (
                "Build an interactive 3D garden scene in /workspace with three.js, one module "
                "per file, then check it in the browser.")}]},
            {"id": "a", "role": "assistant", "parts": parts}], step


def post(url: str, body: dict, headers: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ledger_row(path: str, rid: str, wait_s: float = 20) -> dict | None:
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            db.row_factory = sqlite3.Row
            r = db.execute("SELECT prompt_tokens, completion_tokens, engine_cached, cached_tokens,"
                           " forwarded_tokens, cache_source, slot, prompt_ms, ts_ms FROM requests"
                           " WHERE request_id = ? ORDER BY id DESC LIMIT 1", (rid,)).fetchone()
            db.close()
            if r is not None:
                return dict(r)
        except sqlite3.Error:
            pass
        time.sleep(1)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--ledger", default=os.path.expanduser(
        "~/.qwen38-spark-engine/usage/ledger.sqlite3"))
    ap.add_argument("--history-tokens", type=int, default=100_000)
    # chatd sends the window's room; a small limit cuts tool calls, which then go back as other
    # text than the model wrote (server/toolcall.py) and the copy parts from the rows there
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    conv = f"e2e-prefill-{stamp}"
    branch, step = history(a.history_tokens)
    shots = {f"img{k}": png(1280, 800, k) for k in range(4)}
    settings = {"effort": "low", "thinking": True}
    system = render.system_block(today="2026-10-17", tools=True)

    def body_of(br):
        b = render.body("TandemLLM", settings, br, system=system, vision=True,
                        blob=lambda att: shots.get(att.get("id")), tools=TOOLS,
                        inline_images=True)
        b["stream"] = False
        b.pop("stream_options", None)
        b["max_tokens"] = a.max_tokens
        return b

    plan = [("A", None), ("B", "img0"), ("C", "text"), ("D", "img1"), ("E", "text")]
    rows, prev = [], None
    parallel = {}
    for label, result in plan:
        body = body_of(branch)
        th = None
        if label == "C":
            # a parallel chat in another session, sent while this turn decodes
            def side():
                time.sleep(3)
                pb = {"model": "TandemLLM", "stream": False, "max_tokens": 64,
                      "messages": [{"role": "user", "content": "Name three Japanese garden "
                                    "elements in one line. " + filler(300, 99)}]}
                r = post(a.url, pb, render.headers(conv + "-side"))
                parallel["row"] = ledger_row(a.ledger, r["id"])
            th = threading.Thread(target=side)
            th.start()
        t0 = time.time()
        r = post(a.url, body, render.headers(conv))
        wall = time.time() - t0
        if th is not None:
            th.join()
        row = ledger_row(a.ledger, r["id"]) or {}
        msg = r["choices"][0]["message"]
        # what the engine read is the served prompt: the client's tokens and the router's hidden
        # notes (a new user turn gets one), engine_cached + forwarded
        new = None
        if prev is not None and row:
            served = int(row["engine_cached"] or 0) + int(row["forwarded_tokens"] or 0)
            served_prev = int(prev["engine_cached"] or 0) + int(prev["forwarded_tokens"] or 0)
            new = served - served_prev - int(prev["completion_tokens"])
        fwd = int(row.get("forwarded_tokens") or 0)
        ec = int(row.get("engine_cached") or 0)
        ok = None if new is None else fwd <= new + 64
        rows.append({"turn": label, "request_id": r["id"], "wall_s": round(wall, 1),
                     "prompt": row.get("prompt_tokens"), "engine_cached": ec, "forwarded": fwd,
                     "new_since_end": new, "reuse": round(ec / max(1, ec + fwd), 4),
                     "cache_source": row.get("cache_source"), "slot": row.get("slot"),
                     "prompt_ms": row.get("prompt_ms"), "resumed_at_end": ok})
        print(json.dumps(rows[-1]), flush=True)
        prev = row
        # the answer as chatd keeps it, then the next tool result (or a steer when no call)
        parts = branch[-1]["parts"]
        if msg.get("reasoning_content"):
            parts.append({"type": "thinking", "text": msg["reasoning_content"], "step": step})
        if msg.get("content"):
            parts.append({"type": "text", "text": msg["content"], "step": step})
        calls = msg.get("tool_calls") or []
        for c in calls:
            parts.append({"type": "tool_call", "id": c["id"], "name": c["function"]["name"],
                          "step": step, "input": c["function"].get("arguments") or "{}",
                          "status": "done", "output": {"content": "ok, no console errors"}})
        i = [p[0] for p in plan].index(label)
        nxt = plan[i + 1][1] if i + 1 < len(plan) else None
        if nxt and nxt.startswith("img"):
            if calls:
                parts[-1]["output"] = {"content": "rendered http://127.0.0.1:8123; no console "
                                                  "errors", "images": [{"id": nxt}]}
            else:
                branch.append({"id": f"s{step}", "role": "user", "parts": [
                    {"type": "text", "text": "Here is the page."}],
                    "attachments": [{"id": nxt, "kind": "image", "mime": "image/png"}]})
        elif not calls:
            branch.append({"id": f"s{step}", "role": "user",
                           "parts": [{"type": "text", "text": "Go on."}]})
        if branch[-1]["role"] == "user":
            branch.append({"id": f"a{step}", "role": "assistant", "parts": []})
        step += 1
    report = {"conversation": conv, "url": a.url, "history_tokens": a.history_tokens,
              "turns": rows, "parallel": parallel.get("row"),
              "pass": all(r["resumed_at_end"] for r in rows if r["resumed_at_end"] is not None)
              and (parallel.get("row") or {}).get("slot") not in (0, None)}
    print(json.dumps({"pass": report["pass"], "parallel_slot": (parallel.get("row") or {})
                      .get("slot")}), flush=True)
    out = a.out or os.path.join(ROOT, "results", "prefill-reuse", f"{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"report: {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
