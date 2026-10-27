#!/usr/bin/env python3
"""pin3 coding probe — FastAPI refactor (Loca regime).

4 runs, temperature 0.6, max_tokens 512, thinking off, streaming.
tok/s = completion_tokens / decode_s (first-token -> last-chunk).
mean_n = delta(spec_decode_num_draft_tokens_total) / delta(spec_decode_num_drafts_total).
"""
import json
import re
import statistics
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "GLM-5.3-Flash-EXL3"

PROMPT = (
    "Refactor this FastAPI service module for clarity and add error handling. "
    "Improve naming, extract helper functions, wrap I/O in try/except with proper "
    "HTTPException statuses. Keep the same endpoints and response shape.\n"
    "```python\n"
    "from fastapi import FastAPI\n"
    'app = FastAPI()\n\n'
    '@app.get("/items")\n'
    "def get_items():\n"
    "    import sqlite3\n"
    '    con = sqlite3.connect("app.db")\n'
    "    cur = con.cursor()\n"
    '    rows = cur.execute("select id,name,price from items").fetchall()\n'
    "    out = []\n"
    "    for r in rows:\n"
    '        out.append({"id": r[0], "name": r[1], "price": r[2]})\n'
    "    con.close()\n"
    "    return out\n\n"
    '@app.post("/items")\n'
    "def create_item(name: str, price: float):\n"
    "    import sqlite3\n"
    '    con = sqlite3.connect("app.db")\n'
    "    cur = con.cursor()\n"
    '    cur.execute("insert into items(name,price) values (?,?)", (name, price))\n'
    "    con.commit()\n"
    "    con.close()\n"
    '    return {"ok": True}\n'
    "```"
)

SPEC_RE = re.compile(r"^(vllm:spec_decode_[a-zA-Z0-9_]+)\{[^}]*\}\s+(\S+)$", re.M)


def spec_snapshot():
    req = urllib.request.Request(BASE + "/metrics")
    text = urllib.request.urlopen(req, timeout=10).read().decode()
    out = {}
    for m in SPEC_RE.finditer(text):
        out[m.group(1)] = float(m.group(2))
    return out


def one_run(max_tokens, temperature, phase):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    pre = spec_snapshot()
    t0 = time.perf_counter()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=600)
    t_first = None
    t_last = None
    completion_tokens = None
    text_parts = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        if d.get("usage"):
            completion_tokens = d["usage"].get("completion_tokens")
        ch = d.get("choices") or []
        if ch:
            piece = (ch[0].get("delta") or {}).get("content")
            if piece:
                if t_first is None:
                    t_first = time.perf_counter()
                t_last = time.perf_counter()
                text_parts.append(piece)
    t_end = time.perf_counter()
    post = spec_snapshot()
    decode_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
    text = "".join(text_parts)
    if completion_tokens is None:
        completion_tokens = 0
    d_drafts = post["vllm:spec_decode_num_drafts_total"] - pre["vllm:spec_decode_num_drafts_total"]
    d_draft_tok = post["vllm:spec_decode_num_draft_tokens_total"] - pre["vllm:spec_decode_num_draft_tokens_total"]
    d_acc = post["vllm:spec_decode_num_accepted_tokens_total"] - pre["vllm:spec_decode_num_accepted_tokens_total"]
    rec = {
        "phase": phase,
        "ttft_s": t_first - t0 if t_first else None,
        "decode_s": decode_s,
        "wall_s": t_end - t0,
        "completion_tokens": completion_tokens,
        "tok_s": (completion_tokens / decode_s) if decode_s > 0 else None,
        "text_head": text[:60],
        "mean_n": (d_draft_tok / d_drafts) if d_drafts > 0 else None,
        "accept_ratio": (d_acc / d_draft_tok) if d_draft_tok > 0 else None,
        "d_drafts": d_drafts,
        "d_draft_tokens": d_draft_tok,
    }
    print(json.dumps(rec), flush=True)
    return rec


def main():
    out = {"phase": "pin3-coding", "prompt": "fastapi-refactor", "thinking": False,
           "temperature": 0.6, "max_tokens": 512, "runs": []}
    warm = one_run(32, 0.6, "warmup32")
    out["warmup"] = warm
    for i in range(4):
        out["runs"].append(one_run(512, 0.6, f"run{i+1}"))
        time.sleep(0.5)
    tok = [r["tok_s"] for r in out["runs"] if r["tok_s"]]
    ns = [r["mean_n"] for r in out["runs"] if r["mean_n"] is not None]
    ars = [r["accept_ratio"] for r in out["runs"] if r["accept_ratio"] is not None]
    out["tok_s_median"] = statistics.median(tok) if tok else None
    out["mean_n_median"] = statistics.median(ns) if ns else None
    out["accept_ratio_median"] = statistics.median(ars) if ars else None
    with open("/tmp/pin3_coding.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("tok_s_median", "mean_n_median", "accept_ratio_median")}), flush=True)
    print("wrote /tmp/pin3_coding.json", flush=True)


if __name__ == "__main__":
    main()