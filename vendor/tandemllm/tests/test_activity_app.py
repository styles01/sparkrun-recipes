"""The live activity through the real server: `server/app.py
--fake-engine` in a process of its own, real sockets, the SSE stream read raw.

Scenario map:
   states, tool name, stop tool_calls, waiting_for_client, continues, abandon mid-prefill,
          length / stop string / error stops, recent            test_the_activity_check_passes
          (tools/activity_check.py --fake, the same checks the box tier runs)
          queue full -> refused, too long -> rejected            test_refused_and_rejected
          streamed bytes identical with the activity on and off  test_bytes_identical_on_and_off
  the fake prefill fills the counter, done grows         test_prefill_progress_grows
  VIS-40: the live prefill rate after the slow first chunk, tile field == banner field, decode
          from token 50, the finished row == timings            test_live_rates_leave_the_warm_up_out
   401 without auth, 429 on a fifth stream, a closed reader released within 1 s, `: ping`
          on an idle stream, the kill switch                     test_auth_cap_release_and_ping,
                                                                 test_the_kill_switch_on_the_wire

Run: python tests/test_activity_app.py   (CPU torch; no GPU, no triton)
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_fake_engine import ADMIN, _port, _start  # noqa: E402
from tools import contract_check  # noqa: E402


def _up(port, proc):
    end = time.time() + 60
    while time.time() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except OSError:
            if proc.poll() is not None:
                raise AssertionError(proc.stdout.read())
            time.sleep(0.2)
    raise AssertionError("the fake server did not come up")


class Server:
    def __init__(self, *extra, hz="4"):
        self.port = _port()
        self.ledger = os.path.join(tempfile.mkdtemp(prefix="qse-act-"), "ledger.sqlite3")
        os.environ["QSE_LIVE_HZ"] = hz
        try:
            self.proc = _start(self.port, self.ledger, extra=extra)
        finally:
            os.environ.pop("QSE_LIVE_HZ", None)
        _up(self.port, self.proc)
        self.base = f"http://127.0.0.1:{self.port}"

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _live_socket(port, token=ADMIN):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    auth = f"Authorization: Bearer {token}\r\n" if token else ""
    s.sendall(f"GET /v1/dashboard/live HTTP/1.1\r\nHost: x\r\n{auth}\r\n".encode())
    return s


def _read_for(s, seconds):
    s.settimeout(0.2)
    end, buf = time.time() + seconds, b""
    while time.time() < end:
        try:
            c = s.recv(65536)
        except socket.timeout:
            continue
        if not c:
            break
        buf += c
    return buf.decode(errors="replace")


def _events(text):
    return [json.loads(line[6:]) for line in text.split("\n") if line.startswith("data: {")]


def _stream(base, body, headers=None):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# ------------------------------------------------------------------ end to end
def test_the_activity_check_passes():
    srv = Server("--fake-tps", "60")
    try:
        out = os.path.join(tempfile.mkdtemp(prefix="qse-act-"), "check.json")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "activity_check.py"),
                            "--base", srv.base, "--token", ADMIN, "--fake",
                            "--abandon-tokens", "500", "--out", out],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        res = json.load(open(out))
        assert res["pass"] and len(res["checks"]) >= 14, res["checks"]
    finally:
        srv.close()


def test_prefill_progress_grows():
    srv = Server("--fake-prefill-tps", "4000")
    try:
        body = {"messages": [{"role": "user", "content": "FAKE_SLOW_PREFILL go"}],
                "max_tokens": 8, "stream": True}
        t = threading.Thread(target=_stream, args=(srv.base, body))
        t.start()
        seen = []
        end = time.time() + 10
        while time.time() < end and t.is_alive():
            with urllib.request.urlopen(urllib.request.Request(
                    srv.base + "/v1/dashboard/live?follow=0",
                    headers={"Authorization": f"Bearer {ADMIN}"}), timeout=5) as r:
                snap = json.loads(r.read())
            for row in snap["requests"]:
                p = (row["activity"] or {}).get("prefill")
                if row["activity"]["state"] == "prefilling" and p and p["pct"] is not None:
                    seen.append(p)
            time.sleep(0.15)
        t.join(20)
        done = [p["done"] for p in seen]
        assert len(done) >= 2 and done == sorted(done) and done[-1] > done[0], done
        assert all(p["progress"] == "chunked" and p["total"] >= 8192 for p in seen), seen[-1]
        assert seen[-1]["tps_avg"] and seen[-1]["eta_ms"] is not None, seen[-1]
    finally:
        srv.close()


def test_live_rates_leave_the_warm_up_out():
    """VIS-40 through the real handler: the chunk hook's marks give the prefill's now-rate
    after its (slow) first chunk, the same number in the tile's field and the banner's; decode
    counts from token 50; the finished row keeps the response's own timings."""
    srv = Server("--fake-prefill-tps", "4000", "--fake-tps", "60")
    try:
        body = {"messages": [{"role": "user", "content": "FAKE_WARM_PREFILL FAKE_LONG go"}],
                "max_tokens": 400, "stream": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        out = {}
        t = threading.Thread(target=lambda: out.update(r=_stream(srv.base, body)))
        t.start()
        snaps = []
        end = time.time() + 40
        while time.time() < end and t.is_alive():
            with urllib.request.urlopen(urllib.request.Request(
                    srv.base + "/v1/dashboard/live?follow=0",
                    headers={"Authorization": f"Bearer {ADMIN}"}), timeout=5) as r:
                snaps.append(json.loads(r.read()))
            time.sleep(0.1)
        t.join(30)
        with urllib.request.urlopen(urllib.request.Request(
                srv.base + "/v1/dashboard/live?follow=0",
                headers={"Authorization": f"Bearer {ADMIN}"}), timeout=5) as r:
            snaps.append(json.loads(r.read()))
        for s in snaps[::5]:
            assert not contract_check.check(s, "live")
        assert snaps[-1]["windows"]["decode_warmup_tokens"] == 50
        pre = [(s, r) for s in snaps for r in s["requests"] if r["phase"] == "prefill"
               and r["activity"]["prefill"]["pct"] is not None]
        warming = [p for _, r in pre for p in [r["activity"]["prefill"]]
                   if p["tps_live"] is None]
        rated = [(s, r["activity"]["prefill"]) for s, r in pre
                 if r["activity"]["prefill"]["tps_live"] is not None]
        assert warming and warming[0]["warmup"]["reached"] is True, warming[:1]
        assert len(rated) >= 3, len(rated)
        for s, p in rated:
            assert s["now"]["prefill_live_tps"] == p["tps_live"], (s["now"], p)
            assert 2500 < p["tps_live"] < 4500, p                   # the fake's 4,000 tok/s
        # the loop's own average still carries the 1.5 s first chunk
        assert rated[0][1]["tps_avg"] < 0.8 * rated[0][1]["tps_live"], rated[0][1]
        dec = [r for s in snaps for r in s["requests"] if r["phase"] == "decode"]
        early = [r for r in dec if not r["decode_warmup"]["reached"]]
        late = [r for r in dec if r["decode_live_now_tps"] is not None]
        assert early and all(r["decode_live_now_tps"] is None for r in early), early[:1]
        assert late and all(r["decode_warmup"]["tokens"] >= 50 for r in late), late[:1]
        assert 40 < late[-1]["decode_live_now_tps"] < 80, late[-1]
        code, text = out["r"]
        assert code == 200
        timings = [json.loads(ln[6:]) for ln in text.split("\n") if ln.startswith("data: {")
                   and '"timings"' in ln][-1]["timings"]
        done = next(r for r in snaps[-1]["requests"] if r["phase"] == "done")
        assert done["prefill_tps"] == timings["prompt_per_second"], (done, timings)
        assert done["decode_tps"] == timings["predicted_per_second"], (done, timings)
        assert done["decode_live_avg_tps"] is not None
    finally:
        srv.close()


def test_refused_and_rejected():
    srv = Server("--max-queue", "1", "--max-len", "4096")
    try:
        slow = {"messages": [{"role": "user", "content": "FAKE_SLOW one"}], "max_tokens": 40,
                "stream": True}
        ts = [threading.Thread(target=_stream, args=(srv.base, slow)) for _ in range(2)]
        for t in ts:
            t.start()
            time.sleep(0.4)
        code, _ = _stream(srv.base, dict(slow, stream=False))
        assert code == 503, code
        for t in ts:
            t.join(30)
        code, _ = _stream(srv.base, {"messages": [{"role": "user", "content": "x" * 5000}]})
        assert code == 400, code
        with urllib.request.urlopen(urllib.request.Request(
                srv.base + "/v1/dashboard/live?follow=0",
                headers={"Authorization": f"Bearer {ADMIN}"}), timeout=5) as r:
            snap = json.loads(r.read())
        assert not contract_check.check(snap, "live")
        stops = {(x["stop"]["reason"], x["stop"]["detail"]) for x in snap["recent"]}
        assert ("refused", "queue_full") in stops, stops
        assert ("rejected", "prompt_too_long") in stops, stops
        ref = next(x for x in snap["recent"] if x["stop"]["reason"] == "refused")
        assert ref["stop"]["state"] == "queued" and ref["stop"]["sentence"] == "refused: queue full"
    finally:
        srv.close()


# ------------------------------------------------------------------ lossless: bytes on vs off
_SCENARIOS = [
    {"messages": [{"role": "user", "content": "plain question about rivers"}]},
    {"messages": [{"role": "user", "content": "no thinking please"}],
     "chat_template_kwargs": {"enable_thinking": False}},
    {"messages": [{"role": "user", "content": "use a tool to look this up"}],
     "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {
         "type": "object", "properties": {"query": {"type": "string"}}}}}]},
    {"messages": [{"role": "user", "content": "FAKE_LONG essay"}], "max_tokens": 400},
    {"messages": [{"role": "user", "content": "FAKE_ERROR now"}]},
    {"messages": [{"role": "user", "content": "a stop string"}], "stop": [" the "]},
    {"messages": [{"role": "user", "content": "usage separate"}],
     "stream_options": {"include_usage": True}},
    {"messages": [{"role": "user", "content": "FAKE_SLOW_PREFILL long prompt"}],
     "max_tokens": 30},
]


def _normalise(text):
    """The stream minus what differs run to run: ids, `created`, and the timing numbers."""
    out = []
    for line in text.split("\n"):
        if not line.startswith("data: {"):
            out.append(line)
            continue
        c = json.loads(line[6:])
        for k in ("id", "created", "timings", "metrics"):
            c.pop(k, None)
        for ch in c.get("choices") or []:
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                tc.pop("id", None)
            for tc in (ch.get("message") or {}).get("tool_calls") or []:
                tc.pop("id", None)
        out.append("data: " + json.dumps(c, sort_keys=True))
    return "\n".join(out)


def test_bytes_identical_on_and_off():
    runs = {}
    for flag in ("on", "off"):
        srv = Server("--live-activity", flag, "--fake-prefill-tps", "20000")
        try:
            watcher = _live_socket(srv.port)             # a stream open, so the sampler runs
            got = []
            for body in _SCENARIOS:
                for stream in (True, False):
                    code, text = _stream(srv.base, dict({"max_tokens": 120}, **body,
                                                        stream=stream))
                    if not stream:
                        d = json.loads(text) if text.startswith("{") else text
                        if isinstance(d, dict):
                            for k in ("id", "created", "timings", "metrics"):
                                d.pop(k, None)
                            for ch in d.get("choices") or []:
                                for tc in (ch.get("message") or {}).get("tool_calls") or []:
                                    tc.pop("id", None)
                        text = json.dumps(d, sort_keys=True)
                    got.append((code, _normalise(text)))
            watcher.close()
            runs[flag] = got
        finally:
            srv.close()
    assert len(runs["on"]) == 2 * len(_SCENARIOS)
    for i, (a, b) in enumerate(zip(runs["on"], runs["off"])):
        assert a == b, (i, a[1][:400], b[1][:400])


# ------------------------------------------------------------------ transport
def test_auth_cap_release_and_ping():
    srv = Server("--dashboard-login", "on")          # the 401 below is the login's; off is the default
    try:
        s = _live_socket(srv.port, token=None)
        head = _read_for(s, 1.0)
        assert head.startswith("HTTP/1.0 401") or head.startswith("HTTP/1.1 401"), head[:80]
        s.close()
        four = [_live_socket(srv.port) for _ in range(4)]
        time.sleep(0.5)
        fifth = _live_socket(srv.port)
        assert " 429 " in _read_for(fifth, 1.0).split("\r\n")[0]
        fifth.close()
        four[0].close()
        t0 = time.time()
        while True:
            again = _live_socket(srv.port)
            first = _read_for(again, 0.3).split("\r\n")[0]
            if " 200 " in first:
                break
            again.close()
            assert time.time() - t0 < 1.0, "the closed reader was not released within 1 s"
        released = time.time() - t0
        for x in four[1:]:
            x.close()
        text = _read_for(again, 32.0)                    # idle: pings, events with rising seq
        again.close()
        assert text.count(": ping") >= 2, text.count(": ping")
        ids = [int(m) for m in re.findall(r"^id: (\d+)$", text, flags=re.M)]
        assert ids == sorted(set(ids)), ids[:10]
        for e in _events(text)[-3:]:
            assert not contract_check.check(e, "live")
        assert released < 1.0, released
    finally:
        srv.close()


def test_the_kill_switch_on_the_wire():
    srv = Server("--live-activity", "off", "--fake-tps", "40")
    try:
        s = _live_socket(srv.port)
        t = threading.Thread(target=_stream, args=(srv.base, {
            "messages": [{"role": "user", "content": "FAKE_SLOW x"}], "max_tokens": 30,
            "stream": True}))
        t.start()
        text = _read_for(s, 3.0)
        t.join(20)
        s.close()
        evs = _events(text)
        busy = [e for e in evs if e["counts"]["in_flight"]]
        assert busy and all(e["interval_s"] == 1.0 for e in evs), [e["interval_s"] for e in evs]
        assert len(busy) <= 4, len(busy)                 # one a second, not four
        for e in evs:
            assert e["recent"] is None and e["engine"]["waiting_for_client"] is None
            assert all(r["activity"] is None and r["timeline"] is None for r in e["requests"])
            assert not contract_check.check(e, "live")
    finally:
        srv.close()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"  {name:64s} ok ({time.time() - t0:.1f} s)", flush=True)
            passed += 1
    print(f"{passed} passed")
