"""The requests of a recorded conversation against the live engine, one token each (SRV-667).

tools/chat_trace.py --bodies writes the requests chatd builds for a recorded conversation; this
sends them to the engine in order, `max_tokens` 1, under one session, with a request of another
session in the middle (a side call: another client, a title, a router question), and reads for
each how much the engine read again (the usage ledger's `forwarded_tokens` and `cache_source`).
A request that reads more than its new tail plus one prefill chunk (the turn resumes at the grid
line below the previous prompt's end: the one generated token is not the recorded answer) read
the conversation again: the prefix changed, a cache was refused or lost, or the side call took
the slot. The engine log's cache lines of the window are counted beside it. Minutes, not the hours
of a model run; on the box, beside a served engine (it adds one short request at a time).

    python tools/prefix_check.py <bodies.jsonl> [--url http://127.0.0.1:8000] [--side-at N]
        [--limit N] [--log ~/tandem-bolt/logs/engine.log]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.request
import uuid

LEDGER = os.path.expanduser("~/.qwen38-spark-engine/usage/ledger.sqlite3")
CHUNK = 4096
SLACK = 256
CACHE_LINES = re.compile(r"\[cache\] (re-read .*|session put declined.*|end anchor not kept.*|"
                         r"end alignment partial.*)")


def post(url: str, body: dict, headers: dict, timeout: float = 1800) -> dict:
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.load(f)


def ledger_row(rid: str, path: str = LEDGER, wait: float = 10.0) -> dict | None:
    t0 = time.time()
    while time.time() - t0 < wait:
        c = sqlite3.connect(path)
        try:
            cols = [r[1] for r in c.execute("pragma table_info(requests)")]
            row = c.execute("select * from requests where request_id = ?", (rid,)).fetchone()
        finally:
            c.close()
        if row is not None:
            return dict(zip(cols, row))
        time.sleep(0.3)
    return None


def side_body(seed: int, words: int = 3000) -> dict:
    r = random.Random(seed)
    text = " ".join(r.choice(["tide", "reef", "coral", "lantern", "harbor", "signal"]) +
                    str(r.randint(0, 999)) for _ in range(words))
    return {"model": "TandemLLM", "stream": False, "max_tokens": 1,
            "messages": [{"role": "user", "content": text + "\nIs this a list? Answer yes or no."}]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bodies")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--side-at", type=int, default=None, help="index of the request after which "
                    "a side request runs (default: the middle)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log", default=os.path.expanduser("~/tandem-bolt/logs/engine.log"))
    ap.add_argument("--ledger", default=LEDGER)
    a = ap.parse_args(argv)
    bodies = [json.loads(ln) for ln in open(a.bodies) if ln.strip()]
    if a.limit:
        bodies = bodies[:a.limit]
    side_at = a.side_at if a.side_at is not None else len(bodies) // 2
    session = "chat-prefixcheck-" + uuid.uuid4().hex[:8]
    head = {"x-session-id": session, "X-QSE-Client": "chat-app"}
    log0 = os.path.getsize(a.log) if os.path.exists(a.log) else 0
    prev_n, bad, rows = 0, [], []
    t_start = time.time()
    for k, b in enumerate(bodies):
        body = {kk: v for kk, v in b.items() if kk not in ("stream_options",)}
        body.update(model="TandemLLM", stream=False, max_tokens=1)
        t0 = time.time()
        out = post(a.url, body, head)
        row = ledger_row(out.get("id", ""), a.ledger) or {}
        n = int(row.get("engine_cached") or 0) + int(row.get("forwarded_tokens") or 0)
        fw = int(row.get("forwarded_tokens") or 0)
        tail = max(0, n - prev_n)
        allowed = tail + CHUNK + SLACK
        ok = k == 0 or fw <= allowed
        rows.append((k, n, fw, row.get("cache_source"), round(time.time() - t0, 1), ok))
        if not ok:
            bad.append(f"request {k}: read {fw:,} again of {n:,} (new tail {tail:,})")
        prev_n = n
        if k == side_at:
            s = post(a.url, side_body(k), {"x-session-id": "side-" + uuid.uuid4().hex[:6]})
            srow = ledger_row(s.get("id", ""), a.ledger) or {}
            rows.append(("side", srow.get("prompt_tokens"), srow.get("forwarded_tokens"),
                         srow.get("cache_source"), None, True))
    lines = []
    if os.path.exists(a.log):
        with open(a.log, errors="replace") as fh:
            fh.seek(log0)
            lines = [m.group(1) for m in (CACHE_LINES.search(ln) for ln in fh) if m]
    print(f"{a.bodies}: {len(bodies)} requests, session {session}, side request after "
          f"{side_at}, {time.time() - t_start:.0f} s")
    print(f"{'#':>5} {'rows':>8} {'re-read':>8} {'from':>9} {'s':>6}")
    for k, n, fw, src, s, ok in rows:
        print(f"{k!s:>5} {n or 0:>8,} {fw or 0:>8,} {str(src):>9} {s if s is not None else '':>6}"
              + ("" if ok else "  <-- read again"))
    print(f"re-read in all: {sum(r[2] or 0 for r in rows if r[0] != 'side'):,} tokens; "
          f"engine cache lines: {len(lines)}")
    for ln in lines[:20]:
        print("   ", ln[:200])
    for x in bad:
        print("FAIL", x)
    print("ok" if not bad else f"{len(bad)} request(s) read the conversation again")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
