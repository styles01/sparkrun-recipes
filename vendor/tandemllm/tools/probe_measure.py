#!/usr/bin/env python3
"""ENG-181's n = 1 check of the probe snapshots on the live engine, after the deploy (no hold;
stdlib only). The server must run with `QWEN38_PROBE_SNAPSHOTS=on`.

    python3 tools/probe_measure.py [--server http://127.0.0.1:8000] [--runs 3]
        [--dir ~/.qwen38-spark-engine/probe] [--out results/probe/measure-<time>.json]

Run it on the box. It never runs in a hold (it refuses while the box lock is held) and sends one
request at a time, every one bounded.

One fixed prompt with a tool, greedy, through TandemLLM, with a client reasoning budget of 1,100
tokens (so a think point at 1,024 and a finish candidate fall inside it) and 1,500 tokens in all,
`--runs` pairs alternating: with the snapshots (no header) and without (`X-Tandem-Probe: off`, the
request skips the capture; the server, its caches and its profile are the same). After one warm-up
each. Measured per run: the streamed reasoning and answer text, the server's decode rate
(`timings`), the TTFT on the client; and, from the ring's index, the points each request wrote.

The gates: the text of every run equal to every other run's (snapshots on and off, greedy); the
snapshot runs' mean decode rate within 3 % of the others'; every snapshot run wrote its prompt
point and every off run none. Exit 0 when all hold.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import glob
import json
import os
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
OPENCODE_UA = "opencode/1.18.33 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"
LOCK = "~/.qwen38-box.flock"
PROMPT = ("Plan a Python function that parses one CSV line with quoted fields (a quote inside a "
          "quoted field is doubled). Think it through, then call the tool `noop` once.")
TOOL = {"type": "function", "function": {"name": "noop", "description": "Does nothing.",
                                         "parameters": {"type": "object", "properties": {},
                                                        "required": []}}}
GATE = 0.03


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


def one(snap: bool, run: int) -> dict:
    body = {"model": "TandemLLM", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": PROMPT}], "tools": [TOOL],
            "temperature": 0, "seed": 7, "max_tokens": 1500, "max_reasoning_tokens": 1100,
            "reasoning_format": "reasoning_content"}
    hdr = {"User-Agent": OPENCODE_UA, "x-session-id": f"probe-measure-{int(snap)}-{run}"}
    if not snap:
        hdr["X-Tandem-Probe"] = "off"
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **hdr})
    t0 = time.monotonic()
    first, rid, text, calls, usage, timings = None, None, [], [], None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            ln = raw.decode().strip()
            if not ln.startswith("data: {"):
                continue
            ev = json.loads(ln[6:])
            rid = rid or ev.get("id")
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if d.get("tool_calls"):
                    calls.append(json.dumps(d["tool_calls"], sort_keys=True))
                if piece:
                    first = first if first is not None else time.monotonic()
                    text.append(piece)
    return {"snap": snap, "run": run, "request_id": rid, "text": "".join(text),
            "calls": calls, "completion_tokens": int((usage or {}).get("completion_tokens") or 0),
            "ttft_ms": round((first - t0) * 1e3, 1) if first is not None else None,
            "server_tps": (timings or {}).get("predicted_per_second")}


def points(root: str, rids: set) -> dict:
    """request id -> the points the ring's index lists for it (today's and yesterday's days)."""
    out: dict = {}
    for idx in sorted(glob.glob(os.path.join(root, "snapshots", "*", "index.jsonl")))[-2:]:
        with open(idx, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("request_id") in rids:
                    out.setdefault(r["request_id"], []).append(r["point"])
    return out


def main(argv=None) -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=BASE)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--dir", default="~/.qwen38-spark-engine/probe")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    BASE = a.server.rstrip("/")
    if held():
        print("[probe] the box lock is held (a hold runs): not now", flush=True)
        return 2
    for s in (True, False):
        w = one(s, -1)
        print(f"[probe] warm-up snapshots {'on' if s else 'off'}: {w['completion_tokens']} tok",
              flush=True)
    rows = []
    for i in range(a.runs):
        for s in ((True, False) if i % 2 == 0 else (False, True)):
            r = one(s, i)
            rows.append(r)
            print(f"[probe] snapshots {'on ' if s else 'off'} run {i}: {r['completion_tokens']} "
                  f"tok, server {r['server_tps']} tok/s, ttft {r['ttft_ms']} ms", flush=True)
    time.sleep(3)                                   # the writer appends after the request
    got = points(os.path.expanduser(a.dir), {r["request_id"] for r in rows})
    texts = {(r["text"], tuple(r["calls"])) for r in rows}
    on = [r["server_tps"] for r in rows if r["snap"] and r["server_tps"]]
    off = [r["server_tps"] for r in rows if not r["snap"] and r["server_tps"]]
    delta = statistics.mean(on) / statistics.mean(off) - 1.0 if on and off else None
    wrote = all("prompt" in got.get(r["request_id"], []) for r in rows if r["snap"])
    none = not any(got.get(r["request_id"]) for r in rows if not r["snap"])
    out = {"runs": [{k: v for k, v in r.items() if k != "text"} for r in rows],
           "points": got, "identical_text": len(texts) == 1, "decode_delta": delta,
           "decode_gate": delta is not None and abs(delta) <= GATE, "snapshot_runs_wrote": wrote,
           "off_runs_wrote_none": none}
    ok = out["identical_text"] and out["decode_gate"] and wrote and none
    print(f"\n[probe] text identical across {len(rows)} runs: {out['identical_text']}; decode on "
          f"{statistics.mean(on) if on else None} vs off {statistics.mean(off) if off else None} "
          f"tok/s, delta {delta if delta is None else f'{delta:+.2%}'} (gate {GATE:.0%}); points "
          f"{ {k: v for k, v in got.items()} }; -> {'PASS' if ok else 'FAIL'}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(dict(out, ts=datetime.datetime.now().isoformat(timespec="seconds")), f,
                      indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
