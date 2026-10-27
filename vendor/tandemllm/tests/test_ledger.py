"""The usage ledger, on a CPU, in temp directories. the Gherkin is this matrix:

  * one row per request through the real handler, with its counts and timings, within 2 s;
  * refused and rejected requests are rows with null token fields;
  * no text: a sentinel in the prompt and in the answer is nowhere in the database or its WAL;
  * the ledger never blocks a request: a stalled writer drops the 10,001st row and counts it;
  * a disk error fails no request and is logged at most once a minute;
  * retention prunes at 400 days, and a shorter retention is refused outside tests;
  * off by default, and not switched on by the environment (row3's servers stay off);
  * the SIGTERM drain flushes what is queued;
  * a test run cannot write the production ledger, symlinks included.

Run: python tests/test_ledger.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_app_loop import Req, serve  # noqa: E402  (first: the CPU environment)
from test_usage import UTok, _ids, _script  # noqa: E402

from server import app, ledger, usage  # noqa: E402

SENTINEL = "zqx-SENTINEL-7f3a91"


def _tmp() -> str:
    return os.path.join(tempfile.mkdtemp(prefix="qse-ledger-"), "usage", "ledger.sqlite3")


def _rows(path: str) -> list[dict]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute("SELECT * FROM requests ORDER BY id")]
    con.close()
    return out


def _served(led, body, engine, path="/v1/chat/completions", headers=None, before=None):
    serve()
    app.STATE.update(tok=UTok(), ledger=led, version="0.1.0-test", code_sha="c0de")
    if before:
        before()
    real = app.generate_stream
    app.generate_stream = engine
    try:
        req = Req(path, body)
        req.headers.update(headers or {})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            head, raw = req.response()
    finally:
        app.generate_stream = real
        app.STATE.pop("ledger", None)
    return head, raw


def test_one_row_per_finished_request():
    led = ledger.Ledger(_tmp()).open()
    tail = len("\n<think>\n\n</think>\n\n")
    prompt = "p" * (1959 - tail)
    gen = _ids("y" * 411) + [3]
    t0 = time.time()
    head, raw = _served(led, {"messages": [{"role": "user", "content": prompt}], "stream": True,
                              "chat_template_kwargs": {"enable_thinking": False}},
                        _script(gen), headers={"User-Agent": "Python/3.12 aiohttp/3.11.2",
                                               "Authorization": "Bearer sk-owui-connection"},
                        before=lambda: app.STATE.update(cfg_eos=3, max_len=4096))
    assert led.flush(2.0), "the row reached the database within 2 s"
    assert time.time() - t0 < 2.5
    rows = _rows(led.path)
    assert len(rows) == 1, rows
    r = rows[0]
    assert (r["prompt_tokens"], r["completion_tokens"], r["finish_reason"], r["status"]) == \
        (1959, 412, "stop", 200), r
    assert r["endpoint"] == "chat" and r["stream"] == 1 and r["client_kind"] == "open-webui"
    assert r["client_id"].startswith("k:") and len(r["client_id"]) == 14
    assert r["client_id"] != "k:sk-owui-conn", "the id is a hash, not the token"
    for k in ("queue_ms", "prompt_ms", "ttft_ms", "decode_ms", "total_ms", "decode_tps"):
        assert r[k] is not None and r[k] >= 0, (k, r)
    assert r["engine_version"] == "0.1.0-test" and r["code_sha"] == "c0de"
    assert abs(r["ts_ms"] / 1000 - t0) < 5
    led.close()


def test_refused_and_rejected_requests_are_rows():
    led = ledger.Ledger(_tmp()).open()
    # the queue is full: the ninth request is refused with a 503
    head, _ = _served(led, {"messages": [{"role": "user", "content": "q"}]}, _script([65]),
                      before=lambda: app.INFLIGHT.update(waiting=8))
    app.INFLIGHT["waiting"] = 0
    assert head.startswith("HTTP/1.1 503"), head
    # a bad parameter: a 400 before the engine
    head, _ = _served(led, {"messages": [{"role": "user", "content": "q"}], "temperature": 9},
                      _script([65]))
    assert head.startswith("HTTP/1.1 400"), head
    # and a body that is not JSON at all
    serve()
    app.STATE["ledger"] = led
    req = Req("/v1/chat/completions", {})
    req.rfile = io.BytesIO(b"{not json")
    req.headers["Content-Length"] = "9"
    with contextlib.redirect_stdout(io.StringIO()):
        head, _ = req.response()
    app.STATE.pop("ledger", None)
    assert head.startswith("HTTP/1.1 400"), head
    assert led.flush(2.0)
    rows = _rows(led.path)
    assert [(r["status"], r["finish_reason"]) for r in rows] == \
        [(503, "refused"), (400, None), (400, None)], rows
    for r in rows:
        assert r["prompt_tokens"] is None and r["completion_tokens"] is None, r
        assert r["decode_tps"] is None and r["ttft_ms"] is None, r
    led.close()


def test_no_text_in_the_ledger():
    led = ledger.Ledger(_tmp()).open()
    gen = _ids(f"The answer mentions {SENTINEL} too.")
    tools = [{"type": "function", "function": {"name": f"tool_{SENTINEL}"}}]
    for stream in (True, False):
        _served(led, {"messages": [{"role": "user", "content": f"Say {SENTINEL}"}],
                      "stream": stream, "tools": tools, "user": f"user-{SENTINEL}",
                      "metadata": {"conversation_id": f"conv-{SENTINEL}"}},
                _script(gen), headers={"Authorization": f"Bearer {SENTINEL}"})
    assert led.flush(2.0)
    raw = b""
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(led.path + suffix):
            raw += open(led.path + suffix, "rb").read()
    assert len(_rows(led.path)) == 2
    assert SENTINEL.encode() not in raw, "text reached the ledger file"
    led.close()
    raw = open(led.path, "rb").read()
    assert SENTINEL.encode() not in raw


def test_the_ledger_never_blocks_a_request():
    stall = threading.Event()
    led = ledger.Ledger(_tmp())
    led._run = lambda: stall.wait(30)               # a writer that never gets to the queue
    led.open()
    row = usage.RequestRecord("x", "chat", False).row()
    t0 = time.perf_counter()
    for _ in range(10_000):
        assert led.submit(row)
    assert led.submit(row) is False                 # the 10,001st
    assert time.perf_counter() - t0 < 2.0, "submit never waits"
    assert led.stats["dropped"] == 1
    assert led.info()["dropped"] == 1
    # and a request completes normally with the queue full
    head, raw = _served(led, {"messages": [{"role": "user", "content": "q"}]}, _script([65, 66]))
    assert head.startswith("HTTP/1.1 200") and led.stats["dropped"] == 2, head
    stall.set()


def test_a_disk_error_fails_no_request_and_logs_once_a_minute():
    now = [1_790_000_000.0]
    logged = []
    led = ledger.Ledger(_tmp(), clock=lambda: now[0], log=logged.append).open()
    led.flush(1.0)
    # the database becomes read-only under the writer
    ro = sqlite3.connect(f"file:{led.path}?mode=ro", uri=True, check_same_thread=False)
    led._conn = ro
    for i in range(3):
        head, raw = _served(led, {"messages": [{"role": "user", "content": "q"}]},
                            _script([65, 66]))
        assert head.startswith("HTTP/1.1 200"), head
        assert led.flush(2.0)
        now[0] += 20.0                               # three failures inside one minute
    assert led.stats["failed"] == 3
    assert len(logged) == 1 and "write failed" in logged[0], logged
    now[0] += 61.0
    _served(led, {"messages": [{"role": "user", "content": "q"}]}, _script([65]))
    led.flush(2.0)
    errors = [m for m in logged if "write failed" in m]
    assert len(errors) == 2, logged
    led._conn = None                                 # the writer reconnects on the next batch
    _served(led, {"messages": [{"role": "user", "content": "q"}]}, _script([65]))
    led.flush(2.0)
    assert len(_rows(led.path)) == 1 and led.info()["dropped"] == 4
    ro.close()


def test_retention_prunes_at_400_days():
    now = time.time()
    led = ledger.Ledger(_tmp(), clock=lambda: now).open()
    base = usage.RequestRecord("x", "chat", False).row()
    for days in (401, 399, 1):
        led.submit(dict(base, ts_ms=int((now - days * 86400) * 1000), request_id=f"d{days}"))
    led.flush(2.0)
    assert led.prune() == 1
    assert [r["request_id"] for r in _rows(led.path)] == ["d399", "d1"]
    led.close()


def test_the_daily_pass_runs_at_four_and_backs_up_weekly():
    import datetime as dt
    t = dt.datetime(2026, 9, 24, 3, 59, 0).astimezone().timestamp()
    assert ledger.next_local(4, t) - t == 60.0
    assert ledger.next_local(4, t + 60.0) - (t + 60.0) > 23 * 3600
    now = [t]
    led = ledger.Ledger(_tmp(), clock=lambda: now[0], backup_keep=2).open()
    led.submit(dict(usage.RequestRecord("x", "chat", False).row(), ts_ms=int(t * 1000)))
    led.flush(2.0)
    made = []
    for week in range(4):
        made.append(led.backup(now=t + week * 7 * 86400))
    assert led.backup(now=t + 3 * 7 * 86400 + 3600) is None, "one backup an ISO week"
    kept = sorted(os.listdir(os.path.join(os.path.dirname(led.path), "backup")))
    assert len(kept) == 2 and kept == sorted(os.path.basename(m) for m in made[-2:]), kept
    b = sqlite3.connect(made[-1])
    assert b.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1
    b.close()
    led.close()


def test_retention_below_the_floor_is_refused_outside_tests():
    a = app.parser().parse_args(["--usage-ledger", _tmp(), "--usage-retention-days", "30"])
    try:
        app.open_ledger(a, test=False)
        raise AssertionError("30 days was accepted")
    except SystemExit as exc:
        assert "400" in str(exc), exc
    assert app.open_ledger(a, test=True) is not None, "tests may prune sooner"


def test_off_by_default_and_not_from_the_environment():
    path = _tmp()
    old = os.environ.get("QSE_USAGE_LEDGER")
    os.environ["QSE_USAGE_LEDGER"] = path
    try:
        a = app.parser().parse_args([])
        assert a.usage_ledger == "off"
        assert app.open_ledger(a) is None
    finally:
        if old is None:
            os.environ.pop("QSE_USAGE_LEDGER")
        else:
            os.environ["QSE_USAGE_LEDGER"] = old
    assert not os.path.exists(os.path.dirname(path)), "nothing was created"
    # row3's servers, which ops/gate.sh starts with all of serve.env exported, pass no ledger
    from pathlib import Path
    from tools import row3
    from types import SimpleNamespace
    ns = SimpleNamespace(python=Path("py"), port=8011, max_len=4096, len_fixed=0, budget=16,
                         repo=Path("/r"), nvfp4="nv", head="hd", len_latch=True, server_arg=[])
    assert not any("usage-ledger" in x for x in row3.server_cmd(ns)), row3.server_cmd(ns)


def test_start_sh_passes_the_ledger_and_serve_env_names_it():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    start = open(os.path.join(root, "ops", "start.sh")).read()
    assert '--usage-ledger "${QSE_USAGE_LEDGER:-off}"' in start
    env = open(os.path.join(root, "ops", "serve.env")).read()
    assert "QSE_USAGE_LEDGER=${HOME}/.qwen38-spark-engine/usage/ledger.sqlite3" in env


_DRAIN_CHILD = r"""
import os, sys, time, threading
sys.path.insert(0, os.path.join({root!r}, "tests"))
import test_app_loop as t
from http.server import ThreadingHTTPServer
from server import app, ledger, usage

t.serve()
led = ledger.Ledger({path!r})
real_write = led._write


def slow_write(batch):
    led._closing.wait(30)          # a writer that has not caught up when the signal lands
    real_write(batch)


led._write = slow_write
app.STATE["ledger"] = led.open()
for i in range(20):
    led.submit(dict(usage.RequestRecord("r%d" % i, "chat", False).row(), request_id="r%d" % i))
httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
print(httpd.server_address[1], flush=True)
app.serve_until_drained(httpd)
"""


def test_the_drain_flushes_the_queue():
    import signal
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = _tmp()
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    child = subprocess.Popen([sys.executable, "-c", _DRAIN_CHILD.format(root=root, path=path)],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    try:
        int(child.stdout.readline())
        con = sqlite3.connect(path)
        assert con.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0, "still queued"
        con.close()
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=30) == 0
    finally:
        if child.poll() is None:
            child.kill()
    assert len(_rows(path)) == 20


def test_tests_cannot_write_the_production_ledger():
    home = tempfile.mkdtemp(prefix="qse-home-")
    old = os.environ.get("HOME")
    os.environ["HOME"] = home
    try:
        prod = os.path.expanduser(ledger.PRODUCTION)
        os.makedirs(os.path.dirname(prod))
        link = os.path.join(home, "elsewhere.sqlite3")
        os.symlink(prod, link)
        for p in (ledger.PRODUCTION, prod, link):
            try:
                ledger.check_config(p, 400, test=True)
                raise AssertionError(f"{p} was accepted")
            except SystemExit as exc:
                assert "production ledger" in str(exc)
        ledger.check_config(prod, 400, test=False)            # the service itself may
        ledger.check_config(os.path.join(home, "e2e.sqlite3"), 1, test=True)
    finally:
        os.environ["HOME"] = old


def test_client_kinds():
    ck = ledger.client_of
    assert ck({}) == ("anon", "other")
    assert ck({"User-Agent": "curl/8.7.1"})[1] == "curl"
    assert ck({"User-Agent": "OpenAI/Python 1.99.0"})[1] == "openai-sdk"
    assert ck({"User-Agent": "Python/3.12 aiohttp/3.11.2"})[1] == "open-webui"
    assert ck({"X-QSE-Client": "dashboard", "User-Agent": "Mozilla/5.0"})[1] == "dashboard"
    assert ck({"Referer": "https://your-host.example/dashboard/dev", "User-Agent": "Mozilla/5.0"})[1] \
        == "dashboard", "the Dev tab's test request, a same-origin fetch"
    assert ck({"Referer": "https://chat.example.com/c/1", "User-Agent": "Mozilla/5.0"})[1] == "other"
    a, b = ck({"Authorization": "Bearer one"})[0], ck({"Authorization": "Bearer two"})[0]
    assert a != b and a.startswith("k:") and len(a) == 14
    assert ck({"Authorization": "Basic Zm9v"})[0] == "anon"
    # VIS-60: TandemLLM's own calls (the router's question to the engine)
    assert ck({"User-Agent": "tandem-conductor/1"})[1] == "internal"


def test_schema_is_the_design_notes():
    led = ledger.Ledger(_tmp()).open()
    con = sqlite3.connect(led.path)
    cols = [r[1] for r in con.execute("PRAGMA table_info(requests)")]
    assert cols == ["id"] + list(ledger.COLUMNS), cols
    meta = dict(con.execute("SELECT key, value FROM meta"))
    # version 2 (SRV-115): the conductor's columns and its journal's two tables; version 3: the
    # proxy's columns; version 4: what the plugins inside vLLM and SGLang report (P6)
    assert meta["schema_version"] == "10" and int(meta["created_at"]) > 0
    assert "migrated_v3_to_v4" not in meta, "a new file is version 4 from the start"
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"requests", "meta", "conductor_events", "conductor_texts"} <= tables
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert con.execute("PRAGMA auto_vacuum").fetchone()[0] == 2        # incremental
    idx = {r[1] for r in con.execute("PRAGMA index_list(requests)")}
    assert {"requests_ts", "requests_dim_ts", "requests_backend_ts"} <= idx
    con.close()
    led.close()
    assert oct(os.stat(led.path).st_mode & 0o777) == "0o600"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
