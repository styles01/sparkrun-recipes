"""The proxy's own cost and its concurrency, against the fake backend on this machine.

    python tools/proxy_bench.py [--n 200] [--streams 64] [--out bench.json]

1. Added latency: `--n` streamed requests one at a time, each sent straight to the fake backend
   and through the proxy, alternating. The client times its first content frame; the added TTFT
   is the difference of the medians. The proxy's own `proxy_ms` comes from its ledger rows.
2. Concurrency: `--streams` streamed requests at once through the proxy, each 200 tokens at 50
   tokens a second (4 s alone). The fake counts the most streams it served at once; the proxy must
   not serialise them, so that count is `--streams` and the wall time stays near 4 s.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from proxy_util import Proxy, fake  # noqa: E402


def ttft(port: int, body: dict) -> tuple[float, float]:
    """(client TTFT ms, total ms) of one streamed request."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    first = None
    buf = b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        buf += chunk
        if first is None and b'"content": "w' in buf:
            first = time.perf_counter()
    c.close()
    return (first - t0) * 1e3 if first else float("nan"), (time.perf_counter() - t0) * 1e3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    srv, port = fake(models=["fake-a"], tps=2000, cache=False)
    slow, sport = fake(models=["slow-a"], tps=50, cache=False, name="slow")
    env = {"TANDEM_BACKENDS": "f1,f2", "TANDEM_BACKEND_F1_KIND": "generic",
           "TANDEM_BACKEND_F1_URL": f"http://127.0.0.1:{port}",
           "TANDEM_BACKEND_F2_KIND": "generic", "TANDEM_BACKEND_F2_URL": f"http://127.0.0.1:{sport}",
           "TANDEM_BACKEND_F2_MAX_INFLIGHT": str(max(64, a.streams))}
    out = {}
    with Proxy(env) as px:
        px.sync()
        body = {"model": "fake-a", "stream": True,
                "messages": [{"role": "user", "content": "hello FAKE_LONG 8"}]}
        for _ in range(10):                                     # warm both paths
            ttft(port, body), ttft(px.port, body)
        d, p = [], []
        for _ in range(a.n):
            d.append(ttft(port, body)[0])
            p.append(ttft(px.port, body)[0])
        rows = px.rows()[-a.n:]
        pms = sorted(r["proxy_ms"] for r in rows if r["proxy_ms"] is not None)
        out["latency"] = {"n": a.n, "direct_ttft_ms_p50": round(statistics.median(d), 3),
                          "proxy_ttft_ms_p50": round(statistics.median(p), 3),
                          "added_ttft_ms_p50": round(statistics.median(p) - statistics.median(d), 3),
                          "proxy_ms_p50": round(statistics.median(pms), 3),
                          "proxy_ms_p90": round(pms[int(0.9 * len(pms)) - 1], 3)}
        print(f"[bench] latency {json.dumps(out['latency'])}", flush=True)
        slow_body = {"model": "slow-a", "stream": True,
                     "messages": [{"role": "user", "content": "go FAKE_LONG 200"}]}
        done = []
        t0 = time.perf_counter()
        ths = [threading.Thread(target=lambda: done.append(ttft(px.port, slow_body)))
               for _ in range(a.streams)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0
        alone = ttft(sport, slow_body)[1] / 1e3
        out["concurrency"] = {"streams": a.streams, "backend_max_at_once": slow.fake.max_streams,
                              "wall_s": round(wall, 2), "one_stream_alone_s": round(alone, 2),
                              "completed": len(done)}
        print(f"[bench] concurrency {json.dumps(out['concurrency'])}", flush=True)
    srv.shutdown()
    slow.shutdown()
    ok = (out["concurrency"]["backend_max_at_once"] == a.streams
          and out["latency"]["proxy_ms_p50"] < 2.0)
    out["verdict"] = "PASS" if ok else "FAIL"
    print(f"[bench] {out['verdict']}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
