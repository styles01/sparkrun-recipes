"""`tools/soak.py`'s parts: what a leak verdict and an end-against-start rate are made of.

Tested without a board:

  * /proc/meminfo reads MemFree, MemAvailable and the page cache in GB, and nothing off the host;
  * a `/health` answer flattens into one CSV row, the store counters included whatever they are;
  * the CSV carries the union of every row's columns, so a counter that appears late is kept;
  * `trend` reads a plateau as a ~0 second-half slope however much the warm-up climbed, and a
    steady climb as its rate;
  * a probe round pools tokens and milliseconds a block from the joined server lines;
  * a streamed tool call is put back together from its deltas, and the second turn sends it with
    its result as a `tool` message;
  * an idle gap stops new requests and lets traffic resume.

Run: python tests/test_soak.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import soak  # noqa: E402

MEMINFO = """MemTotal:       127535936 kB
MemFree:         10485760 kB
MemAvailable:    62914560 kB
Buffers:           102400 kB
Cached:          52428800 kB
"""


def test_meminfo_reads_free_available_and_page_cache():
    m = soak.meminfo_gb(MEMINFO)
    assert abs(m["memfree_gb"] - 10485760 * 1024 / 1e9) < 1e-9
    assert abs(m["memavail_gb"] - 62914560 * 1024 / 1e9) < 1e-9
    assert abs(m["pagecache_gb"] - 52428800 * 1024 / 1e9) < 1e-9


def test_sample_row_flattens_health_and_store_counters():
    h = {"t": 1010.0,
         "memory": {"allocated_gb": 52.4, "reserved_gb": 53.0, "max_allocated_gb": 52.5, "rss_gb": 3.0},
         "inflight": {"served": 7, "errors": 0},
         "cache": {"state_store": {"entries": 3, "bytes": 123, "puts": 4, "declined_short": 1,
                                   "boundaries": [1024, 2048], "bytes_by_part": {"kv": 1}},
                   "suffix_store": {"tokens": 99, "indexed": True, "path": "/x"},
                   "response_cache": None}}
    row = soak.sample_row(h, 1000.0, {"memfree_gb": 9.0})
    assert row["elapsed_s"] == 10.0 and row["rss_gb"] == 3.0 and row["allocated_gb"] == 52.4
    assert row["memfree_gb"] == 9.0
    assert row["req_served"] == 7 and row["req_errors"] == 0
    assert row["store_entries"] == 3 and row["store_declined_short"] == 1
    assert "store_boundaries" not in row and "store_bytes_by_part" not in row
    assert row["suffix_tokens"] == 99
    assert "suffix_indexed" not in row and "suffix_path" not in row     # not a number to chart


def test_csv_keeps_a_column_that_appears_late():
    rows = [{"t": 1, "elapsed_s": 0, "rss_gb": 3.0},
            {"t": 2, "elapsed_s": 10, "rss_gb": 3.1, "store_hits": 2}]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "s.csv")
        soak.write_csv(path, rows)
        got = list(csv.DictReader(open(path)))
    assert list(got[0]) == ["t", "elapsed_s", "rss_gb", "store_hits"]
    assert got[0]["store_hits"] == "" and got[1]["store_hits"] == "2"


def test_trend_plateau_against_a_leak():
    # warm-up climb over the first ten minutes, then flat for fifty
    flat = [{"elapsed_s": t, "rss_gb": min(3.0 + t / 600.0, 4.0)} for t in range(0, 3600, 10)]
    tr = soak.trend(flat, "rss_gb")
    assert abs(tr["slope_per_10min_2nd_half"]) < 1e-9, tr
    assert tr["first"] < tr["last"] == 4.0
    # 50 MB every ten minutes, all the way
    leak = [{"elapsed_s": t, "rss_gb": 3.0 + 0.05 * t / 600.0} for t in range(0, 3600, 10)]
    tr = soak.trend(leak, "rss_gb")
    assert abs(tr["slope_per_10min_2nd_half"] - 0.05) < 1e-6, tr
    assert soak.trend([{"elapsed_s": 0, "rss_gb": 1.0}], "rss_gb") is None
    assert soak.trend(flat, "absent") is None


def test_probe_summary_pools_blocks_per_round():
    recs = [{"probe": 0, "error": None, "tokens": 256, "tok_s": 40.0, "blocks": 60,
             "committed": 255, "decode_ms": 5700.0},
            {"probe": 0, "error": None, "tokens": 256, "tok_s": 50.0, "blocks": 40,
             "committed": 255, "decode_ms": 3800.0},
            {"probe": 1, "error": "HTTP 503", "tokens": 0, "tok_s": 0.0},
            {"probe": None, "error": None, "tokens": 9, "tok_s": 99.0}]
    s = soak.probe_summary(recs)
    assert set(s) == {0, 1}
    assert s[0]["ok"] == 2 and s[0]["tok_s_p50"] == 45.0
    assert abs(s[0]["ms_blk"] - 95.0) < 1e-9 and abs(s[0]["tok_blk"] - 5.1) < 1e-9
    assert s[1] == {"n": 1, "ok": 0, "tok_s_p50": None, "ms_blk": None, "tok_blk": None}


class _Fake(BaseHTTPRequestHandler):
    """A chat endpoint that answers a request carrying `tools` with a call streamed in pieces,
    and anything else with two words. Every body it receives is kept."""

    bodies: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.bodies.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        last = body["messages"][-1]
        if body.get("tools") and last["role"] == "user":
            deltas = [{"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                       "function": {"name": "get_weather", "arguments": ""}}]},
                      {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]},
                      {"tool_calls": [{"index": 0, "function": {"arguments": '"Hamburg"}'}}]}]
            finish = "tool_calls"
        else:
            deltas = [{"content": "light "}, {"content": "rain"}]
            finish = "stop"
        for d in deltas:
            self.wfile.write(f"data: {json.dumps({'id': 'chatcmpl-x', 'choices': [{'delta': d}]})}\n\n".encode())
        end = {"id": "chatcmpl-x", "choices": [{"delta": {}, "finish_reason": finish}],
               "usage": {"completion_tokens": 5}}
        self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())


def _serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _args(port, **kw):
    a = dict(base_url=f"http://127.0.0.1:{port}/v1", model="m", timeout=10.0, abandon_rate=0.0,
             abandon_after=0.0, long_repeat=1, gap=0.0, health_every=0.05, meminfo=False)
    a.update(kw)
    return SimpleNamespace(**a)


def test_tool_call_is_reassembled_and_answered_with_its_result():
    srv = _serve()
    try:
        _Fake.bodies = []
        rec = soak.post_stream(f"http://127.0.0.1:{srv.server_port}/v1",
                               {"messages": [{"role": "user", "content": "q"}], "tools": [{}]}, 10.0)
        assert rec["id"] == "chatcmpl-x" and rec["finish"] == "tool_calls" and rec["chars"] > 0
        assert rec["calls"] == {0: {"id": "call_a", "name": "get_weather",
                                    "arguments": '{"city": "Hamburg"}'}}
        s = soak.Soak(_args(srv.server_port))
        tool = next(w for w in soak.WORKLOADS if w[0] == "tool")

        class OnlyTool(random.Random):
            def choice(self, seq):
                return tool if tool in seq else super().choice(seq)
        _Fake.bodies = []
        s.one(OnlyTool(0))
        assert len(_Fake.bodies) == 2, _Fake.bodies
        second = _Fake.bodies[1]["messages"]
        assert [m["role"] for m in second] == ["user", "assistant", "tool"]
        assert second[1]["tool_calls"][0]["function"] == {"name": "get_weather",
                                                          "arguments": '{"city": "Hamburg"}'}
        assert second[2]["tool_call_id"] == "call_a" and json.loads(second[2]["content"])["city"]
        assert _Fake.bodies[0]["tools"][0]["function"]["name"] == "get_weather"
        assert [r["workload"] for r in s.recs] == ["tool", "tool"]
    finally:
        srv.shutdown()


def test_probe_rounds_take_the_rows_asked_for_then_the_next_unused():
    srv = _serve()
    try:
        _Fake.bodies = []
        s = soak.Soak(_args(srv.server_port, probe_rows="4,0"))
        s.probe_docs = [{"len": 2048, "domain": dom, "i": i, "text": f"{dom}{i}"}
                        for dom in ("prose", "german", "code") for i in range(6)]
        for _ in range(3):
            s.probe()
        assert s.gate.is_set()                          # traffic resumes after each round
        got = [(r["probe"], r["doc"]) for r in s.recs]
        assert got[:3] == [(0, "prose-2048-4"), (0, "german-2048-4"), (0, "code-2048-4")], got
        assert got[3:6] == [(1, "prose-2048-0"), (1, "german-2048-0"), (1, "code-2048-0")], got
        assert [d for _, d in got[6:]] == ["prose-2048-1", "german-2048-1", "code-2048-1"], got
        assert _Fake.bodies[0]["temperature"] == 0 and _Fake.bodies[0]["max_tokens"] == 256
        assert _Fake.bodies[0]["messages"][0]["content"].endswith("prose4")
    finally:
        srv.shutdown()


def test_the_next_turn_echoes_the_answer_the_server_wrote():
    srv = _serve()
    try:
        _Fake.bodies = []
        s = soak.Soak(_args(srv.server_port))
        chat = next(w for w in soak.WORKLOADS if w[0] == "chat")

        class OnlyChat(random.Random):
            def choice(self, seq):
                return chat if chat in seq else super().choice(seq)
        s.one(OnlyChat(0))
        assert len(_Fake.bodies) == chat[4]
        second = _Fake.bodies[1]["messages"]
        assert [m["role"] for m in second] == ["user", "assistant", "user"]
        assert second[1]["content"] == "light rain"          # the streamed answer, not a placeholder
        assert all("text" not in r for r in s.recs)           # answers stay out of the records
    finally:
        srv.shutdown()


def test_sampled_workload_sends_its_temperature_and_longdoc_needs_documents():
    s = soak.Soak(_args(1))
    assert "longdoc" not in [w[0] for w in s._workloads()]
    s.long_docs = [{"len": 8192, "domain": "prose", "i": 0, "text": "doc"}]
    assert "longdoc" in [w[0] for w in s._workloads()]
    # every length is asked about about as often, however few rows the longest has
    s.long_docs = ([{"len": 8192, "domain": "prose", "i": i, "text": "a"} for i in range(21)]
                   + [{"len": 32768, "domain": "prose", "i": 0, "text": "b"}])
    longdoc = next(w for w in soak.WORKLOADS if w[0] == "longdoc")
    s.a.base_url = "http://127.0.0.1:9/v1"              # nothing listens: the record is an error

    class OnlyLongdoc(random.Random):
        def choice(self, seq):
            return longdoc if longdoc in seq else super().choice(seq)
    rng = OnlyLongdoc(1)
    for _ in range(200):
        s.one(rng)
    sent = [r["doc"] for r in s.recs]
    share = sum(d.startswith("prose-32768") for d in sent) / len(sent)
    assert 0.35 < share < 0.65, share
    assert soak.EXTRA["sampled"]["temperature"] > 0


def test_a_closed_gate_holds_new_requests_and_reopens():
    srv = _serve()
    try:
        _Fake.bodies = []
        s = soak.Soak(_args(srv.server_port))
        s.gate.clear()
        t = threading.Thread(target=s.worker, args=(1,), daemon=True)
        t.start()
        time.sleep(0.3)
        assert _Fake.bodies == []                      # idle: nothing was sent
        s.gate.set()
        deadline = time.time() + 5
        while not _Fake.bodies and time.time() < deadline:
            time.sleep(0.05)
        assert _Fake.bodies, "traffic did not resume"
        s.stop.set()
        t.join(timeout=5)
    finally:
        srv.shutdown()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
