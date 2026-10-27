"""The shadow reader on the live engine (SR1, 2026-10-04): a test session of opencode's shape.

Sends `--turns` turns of one opencode-shaped session (opencode's user agent, a session id, a system
text of about `--system-tokens` tokens of the repository's docs, opencode's tool list shape, each
turn a short user message, the answers kept in the history, `--pause-s` between turns) under a
test bearer (`--key`, which the idle checks do not count), then reads the shadow's status, its
journal rows for that session and, with `--ask`, one typed question over the session
(`POST /v1/conductor/shadow/ask`). Prints one JSON report; nothing of a real conversation.

    python tools/shadow_live.py --base http://127.0.0.1:8000 --key sr1-test --turns 4
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = "opencode/1.18.34 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"
TOOLS = [{"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}
    for n, d in (("read", "Read a file"), ("write", "Write a file"), ("bash", "Run a command"))]
ASKS = ["Which file of the docs explains the drafters? Answer in one sentence.",
        "Summarise the exactness rules in three bullet points.",
        "Write a Python function that parses a size like '4 GiB' into bytes.",
        "Now add a docstring and two doctests to that function.",
        "What would you change first to make prefill faster? Two sentences."]


def docs(tokens: int) -> str:
    text = ""
    for p in sorted(glob.glob(os.path.join(ROOT, "docs", "*.md"))):
        with open(p) as f:
            text += f.read() + "\n\n"
        if len(text) > tokens * 4:
            break
    return text[:tokens * 4]


def call(base, key, body, sid):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}", "User-Agent": UA,
                                          "x-session-id": sid})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def get(base, path, token=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    with urllib.request.urlopen(urllib.request.Request(base + path, headers=h), timeout=60) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="sr1-test")
    ap.add_argument("--admin", default=os.environ.get("QSE_ADMIN_TOKEN"))
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--system-tokens", type=int, default=12000)
    ap.add_argument("--pause-s", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--ask", action="store_true")
    ap.add_argument("--save-streams", default=None,
                    help="a directory: each turn's conversation up to its answer, the answer's "
                         "reasoning kept, as int64 ids (turn-<i>.bin, for shadow_measure's "
                         "narrator part); test text only")
    ap.add_argument("--tokenizer", default=os.path.expanduser(
        "~/.qwen38-spark-engine/models/qwen3.5-0.8b"))
    ap.add_argument("--journal", default=os.path.expanduser(
        "~/.qwen38-spark-engine/conductor/shadow.jsonl"))
    a = ap.parse_args()
    sid = f"sr1-live-{int(time.time())}"
    key = "c:" + sid
    hkey = hashlib.sha256(key.encode()).hexdigest()[:24]
    msgs = [{"role": "system", "content": "You are a coding agent.\n\n" + docs(a.system_tokens)}]
    turns = []
    for i in range(a.turns):
        msgs.append({"role": "user", "content": ASKS[i % len(ASKS)]})
        t0 = time.time()
        r = call(a.base, a.key, {"model": "TandemLLM", "messages": msgs, "tools": TOOLS,
                                 "max_tokens": a.max_tokens, "temperature": 0}, sid)
        m = r["choices"][0]["message"]
        msgs.append({"role": "assistant", "content": m.get("content") or "",
                     **({"reasoning_content": m["reasoning_content"]}
                        if m.get("reasoning_content") else {})})
        u = r.get("usage") or {}
        turns.append({"turn": i, "s": round(time.time() - t0, 2),
                      "prompt_tokens": u.get("prompt_tokens"),
                      "completion_tokens": u.get("completion_tokens"),
                      "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")})
        if a.save_streams:
            import numpy as np
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(a.tokenizer)
            ids = tk.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
            os.makedirs(a.save_streams, exist_ok=True)
            np.asarray(list(ids), dtype="<i8").tofile(os.path.join(a.save_streams,
                                                                   f"turn-{i}.bin"))
        time.sleep(a.pause_s)
    time.sleep(5)
    st = get(a.base, "/v1/conductor/shadow/status", a.admin)
    mine = [s for s in st.get("sessions") or [] if s["session"] == hkey]
    rows = []
    if os.path.isfile(a.journal):
        with open(a.journal) as f:
            rows = [json.loads(x) for x in f if hkey in x]
    out = {"session": key, "turns": turns, "shadow": mine, "router": st.get("router"),
           "stats": st.get("stats"), "slot": {k: (st.get("slot") or {}).get(k) for k in
                                              ("state", "load_s", "took_gib", "precision")},
           "journal": [{k: r.get(k) for k in ("label", "confidence", "agree", "router",
                                               "behind_at_begin", "lag_s", "ms", "error",
                                               "prompt_tokens")} for r in rows]}
    if a.ask:
        body = {"session": key, "question": "What is the user working on in this session?",
                "choices": {"docs": "reading or summarising documentation",
                            "code": "writing or changing code",
                            "ops": "running or deploying a service"}}
        req = urllib.request.Request(a.base + "/v1/conductor/shadow/ask",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {a.admin}"}
                                                 if a.admin else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out["ask"] = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            out["ask"] = {"http": exc.code, "body": exc.read()[:300].decode(errors="replace")}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
