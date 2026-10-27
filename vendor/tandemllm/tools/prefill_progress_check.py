"""Is the live prefill counter where the GPU is? The box tier.

    python tools/prefill_progress_check.py --base http://127.0.0.1:8011 --token "$QSE_ADMIN_TOKEN" \\
        --sizes 8192,32768,65536 --out results/live/prefill-progress-<date>.json

The counter is stored by the prefill's chunk hook after each chunk's forward with no sync, so it
counts chunks the host has ISSUED. If the GPU ran behind the host, `done` would race ahead early
and the request would then sit at its last value for many chunks' time. For each size this sends
one cold prompt (random words, so no cache helps), polls `/v1/dashboard/live?follow=0` at 20 Hz,
and reads the chunk stamps the server reports (`activity.prefill.done` and `at_ms`, both in the
server's clock) and the final `ttft_ms`. It reports every chunk's rate, the median chunk time,
and the tail: the time from the last chunk stamp to the first token, against one chunk's time.
PASS when the tail is at most 1.5 chunk times and no chunk ran faster than 1.5x the median rate.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get(base, token):
    req = urllib.request.Request(base + "/v1/dashboard/live?follow=0",
                                 headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def one(base, token, model, n_words):
    from tools.activity_check import cold_words
    words = cold_words(n_words)
    body = {"model": model, "stream": False, "max_tokens": 4,
            "messages": [{"role": "user", "content": "Reply with ok. " + words}],
            "chat_template_kwargs": {"enable_thinking": False}}
    out = {}

    def send():
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            out["resp"] = json.loads(r.read())
    t = threading.Thread(target=send)
    t.start()
    stamps, rid = {}, None
    while t.is_alive():
        try:
            snap = _get(base, token)
        except OSError:
            time.sleep(0.05)
            continue
        for row in snap["requests"]:
            act = row.get("activity") or {}
            p = act.get("prefill") or {}
            if act.get("state") == "prefilling" and p.get("at_ms") is not None:
                rid = rid or row["request_id"]
                if row["request_id"] == rid:
                    stamps[p["done"]] = (p["at_ms"], p["total"], p.get("cached"))
        time.sleep(0.05)
    t.join()
    resp = out.get("resp") or {}
    tm = resp.get("timings") or {}
    ttft = tm.get("ttft_ms")
    pts = sorted((at, done) for done, (at, _tot, _c) in stamps.items())
    total = next(iter(stamps.values()))[1] if stamps else None
    rates, times = [], []
    for (a0, d0), (a1, d1) in zip(pts, pts[1:]):
        if a1 > a0:
            rates.append((d1 - d0) / ((a1 - a0) / 1e3))
            times.append((a1 - a0, d1 - d0))
    med_rate = statistics.median(rates) if rates else None
    chunk_rows = statistics.median([d for _ms, d in times]) if times else None
    chunk_ms = chunk_rows / med_rate * 1e3 if med_rate else None
    last_at, last_done = pts[-1] if pts else (None, None)
    tail_ms = ttft - last_at if ttft is not None and last_at is not None else None
    left = (total - last_done) if total and last_done else None
    return {"words": n_words, "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens"),
            "total": total, "stamps_seen": len(pts), "ttft_ms": ttft,
            "prompt_per_second": tm.get("prompt_per_second"),
            "median_chunk_rate": round(med_rate, 1) if med_rate else None,
            "max_chunk_rate": round(max(rates), 1) if rates else None,
            "median_chunk_ms": round(chunk_ms, 1) if chunk_ms else None,
            "last_stamp_at_ms": last_at, "rows_after_last_stamp": left,
            "tail_ms": round(tail_ms, 1) if tail_ms is not None else None,
            "tail_in_chunks": round(tail_ms / chunk_ms, 2) if tail_ms and chunk_ms else None,
            "points": pts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    ap.add_argument("--token", default=os.environ.get("QSE_ADMIN_TOKEN"))
    ap.add_argument("--model", default="qwen38-spark-engine")
    ap.add_argument("--sizes", default="8192,32768,65536", help="words (about one token each)")
    ap.add_argument("--out")
    a = ap.parse_args()
    res, ok = [], True
    for n in [int(x) for x in a.sizes.split(",")]:
        r = one(a.base, a.token, a.model, n)
        good = (r["tail_in_chunks"] is not None and r["tail_in_chunks"] <= 1.5
                and r["max_chunk_rate"] <= 1.5 * r["median_chunk_rate"])
        ok &= bool(good)
        r["pass"] = bool(good)
        res.append(r)
        print(f"{'PASS' if good else 'FAIL'} {n} words: {r['prompt_tokens']} tokens, "
              f"{r['stamps_seen']} stamps, chunk {r['median_chunk_ms']} ms at "
              f"{r['median_chunk_rate']} tok/s (max {r['max_chunk_rate']}), ttft {r['ttft_ms']} ms, "
              f"tail {r['tail_ms']} ms = {r['tail_in_chunks']} chunks "
              f"({r['rows_after_last_stamp']} rows after the last stamp)", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": res, "pass": ok},
                      f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
