"""Two live checks of the 2026-10-05 loose ends on a running engine (for the night pass).

    python tools/loose_ends_live.py --base http://127.0.0.1:8000 --out results/loose.json

  cut    a streamed `write` whose content cannot fit `max_tokens` (opencode's shape, raw, so the
         call streams live and nothing holds it): the client gets the call's name and part of its
         arguments, `finish_reason: length`, and none of the block's text as content (until
         0.3.31 it came a second time as content)
  queue  a long streamed request holds the engine; a second streamed request (too long for the
         side lane: thinking on, 200 tokens) arrives a second later. It must get a 200 at once
         or after `queue_wait.OPEN_AFTER_S`, `: waiting, position 1` comments while it waits
         (when it waits more than 10 s), and its answer; never a 429. The first request's
         length is chosen to make the wait about 25 s at 50 tok/s.

Exit code: the number of failed checks. Prints the JSON it writes (no conversation text).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

UA = "opencode/1.18.34 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"
WRITE = {"type": "function", "function": {
    "name": "write", "description": "Write a file to the local filesystem.",
    "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string", "description": "absolute path"},
        "content": {"type": "string", "description": "the file's content"}},
        "required": ["filePath", "content"]}}}


def raw_stream(base, body, headers, timeout=900):
    """The stream's lines as they came, with the time of each, and the status."""
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": UA,
                                          **headers})
    t0 = time.time()
    lines = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = r.status
            for raw in r:
                lines.append((round(time.time() - t0, 3), raw.decode("utf-8", "replace").rstrip()))
    except urllib.error.HTTPError as exc:
        return exc.code, [(round(time.time() - t0, 3), exc.read().decode("utf-8", "replace"))]
    return status, lines


def parse(lines):
    out = {"content": "", "names": [], "args": "", "finish": None, "comments": [],
           "first_data_s": None}
    for t, ln in lines:
        if ln.startswith(":"):
            out["comments"].append((t, ln))
            continue
        if not ln.startswith("data: {"):
            continue
        if out["first_data_s"] is None:
            out["first_data_s"] = t
        ev = json.loads(ln[6:])
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            out["content"] += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                f = tc.get("function") or {}
                if f.get("name"):
                    out["names"].append(f["name"])
                out["args"] += f.get("arguments") or ""
            out["finish"] = ch.get("finish_reason") or out["finish"]
    return out


def check_cut(base) -> dict:
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 400, "temperature": 0,
            "tools": [WRITE],
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": "You are a coding agent. Use the tools."},
                         {"role": "user", "content": "Write /tmp/loose/big.txt with the numbers "
                          "1 to 2000, one per line, spelled out in English words. Call the "
                          "write tool now; do not explain."}]}
    status, lines = raw_stream(base, body, {"X-Tandem": "raw", "x-session-id": "loose-cut"})
    p = parse(lines)
    ok = (status == 200 and p["names"] == ["write"] and p["finish"] == "length"
          and "<tool_call>" not in p["content"] and "<function=" not in p["content"]
          and "<parameter=" not in p["content"])
    return {"ok": ok, "status": status, "names": p["names"], "finish": p["finish"],
            "args_chars": len(p["args"]), "content_chars": len(p["content"]),
            "block_text_in_content": "<parameter=" in p["content"]}


def check_queue(base, hold_tokens: int) -> dict:
    long_body = {"model": "TandemLLM", "stream": True, "max_tokens": hold_tokens,
                 "temperature": 0,
                 "chat_template_kwargs": {"enable_thinking": False},
                 "messages": [{"role": "user", "content": "Count from 1 upward, one number a "
                               "line, and never stop."}]}
    second = {"model": "TandemLLM", "stream": True, "max_tokens": 200, "temperature": 0,
              "messages": [{"role": "user", "content": "In two sentences: why is the sky blue?"}]}
    box = {}
    t = threading.Thread(target=lambda: box.update(long=raw_stream(
        base, long_body, {"X-Tandem": "raw", "x-session-id": "loose-hold"})), daemon=True)
    t.start()
    time.sleep(1.0)
    status, lines = raw_stream(base, second, {"x-session-id": "loose-second"})
    t.join(900)
    p = parse(lines)
    waited = p["first_data_s"] or 0.0
    pings = [c for c in p["comments"] if c[1].startswith(": waiting")]
    ok = (status == 200 and p["finish"] in ("stop", "length") and len(p["content"]) > 0
          and (waited < 10.0 or len(pings) >= 1))
    return {"ok": ok, "status": status, "finish": p["finish"], "first_data_s": waited,
            "waiting_comments": len(pings), "first_comment": pings[0][1] if pings else None,
            "long_status": (box.get("long") or (None,))[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--hold-tokens", type=int, default=1300)
    ap.add_argument("--only", default="cut,queue")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {}
    if "cut" in a.only:
        res["cut"] = check_cut(a.base)
    if "queue" in a.only:
        res["queue"] = check_queue(a.base, a.hold_tokens)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return sum(1 for v in res.values() if not v["ok"])


if __name__ == "__main__":
    sys.exit(main())
