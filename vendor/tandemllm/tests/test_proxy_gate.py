"""tools/proxy_gate.py on LAN jitter (NP1's issue 4): the decode check is relative, on the median,
and a request that misses the per-request bound is sent again.

A fake engine streams 40 tokens 10 ms apart and reports its own `predicted_per_second` on the
finish chunk, as the real one does. "Jitter" holds the last token frame back by 80 ms after the
engine's clock has stopped -- what a late frame on a LAN looks like to the proxy, which times the
frames as they arrive (about -17 % on this request's rate).

On a loaded box (FX14, 2026-10-08: the CPU suite at nice 19 next to a night A/B) the scheduler
adds a few ms between the engine's clock and the proxy's, which moved requests that were NOT made
late past 2 % and the proxy's own p50 past 2 ms; with 5 ms gaps and a 20 ms late frame that was
enough to fail the PASS case and blur the FAIL cases. So: the stream is twice as long and the late
frame four times as late (the same signal, a quarter of the relative jitter), and a run whose
failure comes with load seen in the run itself -- a request that was not made late off by more
than 2 % or retried, or the proxy's p50 at 2 ms or more -- is run again, up to 3 times. A failure
on a run without that signature fails at once, so a gate that stops retrying, keeps the wrong try
or judges wrong is never retried into a pass.

    one late frame, first try only     the old gate (2 % on every request, one try) fails; the new
                                       one passes on the retry and counts it
    a request late on every try        fails on any setting (a real proxy fault is not hidden)
    judge()                            the median, the per-request bound, the strict checks

CPU only, loopback (the fake engine and the proxy on ephemeral ports), about 15 s.

Run: python tests/test_proxy_gate.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import proxy_gate as G  # noqa: E402

N_TOK, GAP_S, LATE_S = 40, 0.010, 0.080
ATTEMPTS = 3


class Engine:
    """late: {request index: tries that get the late frame} (None: every try)."""

    def __init__(self, late: dict):
        self.late = late
        self.tries: dict[int, int] = {}
        eng = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"object": "list", "data": [{"id": "fake-38",
                                                               "object": "model"}]}).encode()
                self.send_response(200 if self.path.endswith("/models") else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                t0 = time.perf_counter()
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                m = re.search(r"\(run (\d+)\)", req["messages"][-1]["content"])
                i = int(m.group(1)) if m else -1
                k = eng.tries[i] = eng.tries.get(i, 0) + 1
                tries = eng.late.get(i, ())
                late = i in eng.late and (tries is None or k in tries)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()

                def frame(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()

                sent = []
                for n in range(N_TOK):
                    if n:
                        time.sleep(GAP_S)
                    if n == N_TOK - 1 and late:
                        sent.append(time.perf_counter())     # the engine's clock stops here
                        time.sleep(LATE_S)                    # the network holds the frame back
                    else:
                        sent.append(time.perf_counter())
                    frame({"id": "c", "object": "chat.completion.chunk", "model": "fake-38",
                           "choices": [{"index": 0, "delta": {"content": f"w{n} "}}]})
                tps = (N_TOK - 1) / (sent[-1] - sent[0])
                frame({"id": "c", "object": "chat.completion.chunk", "model": "fake-38",
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 31, "completion_tokens": N_TOK,
                                 "total_tokens": 31 + N_TOK},
                       "timings": {"predicted_per_second": tps,
                                   "ttft_ms": (sent[0] - t0) * 1e3, "cache_n": 0}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _gate(late: dict, *args) -> tuple[int, dict, str]:
    eng = Engine(late)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "gate.json")
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                rc = G.main(["--url", f"http://127.0.0.1:{eng.port}/v1", "--n", "6",
                             "--max-tokens", str(N_TOK), "--out", out, *args])
            rep = json.load(open(out))
        finally:
            eng.close()
    return rc, rep, buf.getvalue()


def _load_seen(rep: dict, late) -> str:
    """Why a run's numbers are the box's and not the gate's ('' when the run is clean): a request
    that was not made late off by more than 2 % or retried, or the proxy's own p50 at 2 ms+."""
    why = []
    s = rep["summary"]
    if s.get("proxy_ms_p50") is not None and s["proxy_ms_p50"] >= 2.0:
        why.append(f"proxy_ms p50 {s['proxy_ms_p50']:.2f}")
    for r in rep["rows"]:
        if r["i"] in late or "decode_rel" not in r:
            continue
        if r.get("tries", 1) > 1 or abs(r["decode_rel"]) > 0.02:
            why.append(f"request {r['i']} (not late) {r['decode_rel']:+.1%} in {r.get('tries', 1)} tries")
    return "; ".join(why)


def _settled(late: dict, check, *args) -> str:
    """`check(rc, rep, log)` on a gate run; again (up to ATTEMPTS) only when it failed on a run
    that shows load, at once when the run was clean."""
    seen = []
    for k in range(ATTEMPTS):
        rc, rep, log = _gate(late, *args)
        try:
            out = check(rc, rep, log)
        except AssertionError:
            why = _load_seen(rep, late)
            if not why:
                raise
            seen.append(why)
            print(f"    run {k + 1} on a loaded box ({why}); again", flush=True)
            continue
        return out + (f" [run {k + 1}; load before: {' | '.join(seen)}]" if seen else "")
    raise AssertionError(f"{ATTEMPTS} runs, each with load seen: " + " | ".join(seen))


def test_one_late_frame_old_gate_fails_new_gate_retries():
    def old(rc, rep, log):
        s = rep["summary"]
        assert rc == 1 and s["verdict"] == "FAIL", (s, log[-1500:])
        r2 = [r for r in rep["rows"] if r["i"] == 2][0]
        assert r2["decode_rel"] < -0.05, r2
        return f"old gate FAIL (request 2 at {r2['decode_rel']:+.1%}, one try)"

    def new(rc, rep, log):
        s = rep["summary"]
        assert rc == 0 and s["verdict"] == "PASS", (s, log[-1500:])
        assert s["retried"] == 1 and s["loopback"] is True, s
        r2 = [r for r in rep["rows"] if r["i"] == 2][0]
        assert r2["tries"] == 2 and abs(r2["decode_rel"]) <= 0.02, r2
        assert "try 1: decode" in log, log[-1500:]
        return (f"new gate PASS on the retry (request 2 at {r2['decode_rel']:+.1%} on try 2), "
                f"median {s['decode_rel_median']:.2%}")

    return (_settled({2: (1,)}, old, "--tol-one", "0.02", "--repeats", "1") + "; "
            + _settled({2: (1,)}, new))


def test_a_request_late_every_try_fails():
    def check(rc, rep, log):
        s = rep["summary"]
        assert rc == 1 and s["verdict"] == "FAIL", (s, log[-1500:])
        r4 = [r for r in rep["rows"] if r["i"] == 4][0]
        assert r4["tries"] == 3 and r4["decode_rel"] < -0.05, r4
        assert any(i == 4 for i, _ in s["fail_list"]), s["fail_list"]
        return f"request 4 late on 3 of 3 tries: FAIL ({r4['decode_rel']:+.1%})"

    return _settled({4: None}, check)


def _row(i, rel, fail=(), pms=0.4, tries=1):
    return {"i": i, "decode_rel": rel, "fail": list(fail), "tries": tries,
            "ttft_delta_ms": 1.0, "proxy": {"proxy_ms": pms}}


def test_judge():
    ok = [_row(i, 0.004 * (-1) ** i) for i in range(20)]
    assert G.judge(ok)["verdict"] == "PASS"
    # four requests at 3 % (the LAN failures of 2026-10-04) pass: the median is what counts
    lan = ok[:16] + [_row(16 + k, -0.03) for k in range(4)]
    assert G.judge(lan)["verdict"] == "PASS", G.judge(lan)
    # past the per-request bound fails, and so does a median past 2 %
    assert G.judge(ok[:19] + [_row(19, -0.08, tries=3)])["verdict"] == "FAIL"
    assert G.judge([_row(i, 0.025) for i in range(20)])["verdict"] == "FAIL"
    # a strict check fails whatever the rates say, and so does the proxy's own cost
    assert G.judge(ok[:19] + [_row(19, 0.0, fail=["token counts differ"])])["verdict"] == "FAIL"
    assert G.judge([_row(i, 0.0, pms=2.5) for i in range(20)])["verdict"] == "FAIL"
    assert G.loopback("http://127.0.0.1:8000/v1") and G.loopback("http://localhost:8000/v1")
    assert not G.loopback("http://192.168.1.10:8000/v1")
    return "median, per-request bound, strict checks, proxy_ms, loopback"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<56} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<56} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
