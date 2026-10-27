"""The proxy inside the engine's server (`server/app.py --backends`), with the fake engine.

What must hold: without `--backends` nothing changes (the proxy is never imported); with it, the
engine's own model is served by the engine as before and a backend's model is relayed without
touching the engine's counters, lock or queue; /v1/models lists both; one ledger holds both rows;
the dashboard's answers validate against contract 1.3; /v1/config answers; a user's key gates the
engine's routes too.

Run: python tests/test_proxy_engine.py   (CPU torch, as tests/test_fake_engine.py)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_backend  # noqa: E402
from proxy_util import run  # noqa: E402
from test_fake_engine import ADMIN, _port, _start  # noqa: E402

from tools import contract_check  # noqa: E402


def _req(port, path, body=None, method=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {ADMIN}"}
    h.update(headers or {})
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=h,
                               method=method or ("POST" if body is not None else "GET"))
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _wait(port, proc, s=60):
    t0 = time.time()
    while time.time() - t0 < s:
        try:
            if _req(port, "/health")[0] == 200:
                return
        except OSError:
            time.sleep(0.3)
        if proc.poll() is not None:
            raise AssertionError(proc.stdout.read())
    raise AssertionError("the server did not start")


def test_the_engine_with_backends():
    fsrv, fport = fake_backend.start(models=["fake-a"])
    d = tempfile.mkdtemp(prefix="pxeng-")
    ledger = os.path.join(d, "ledger.sqlite3")
    port = _port()
    env = {"QSE_TEST": "1", "TANDEM_BACKENDS": "f1", "TANDEM_BACKEND_F1_KIND": "generic",
           "TANDEM_BACKEND_F1_URL": f"http://127.0.0.1:{fport}",
           "TANDEM_SECRETS_KEY": "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="}
    proc = _start(port, ledger, home=d, extra=("--backends", os.path.join(d, "config.sqlite3")),
                  env_extra=env)
    try:
        _wait(port, proc)
        time.sleep(1.0)                                           # the first health poll
        ids = [m["id"] for m in json.loads(_req(port, "/v1/models")[2])["data"]]
        assert "fake-a" in ids and ids[0] != "fake-a", ids
        before = json.loads(_req(port, "/health")[2])["inflight"]
        st, h, body = _req(port, "/v1/chat/completions", {
            "model": "fake-a", "stream": True,
            "messages": [{"role": "user", "content": "hi FAKE_REASONING"}]})
        assert st == 200 and h.get("X-Tandem-Backend") == "f1" and b"[DONE]" in body
        after = json.loads(_req(port, "/health")[2])["inflight"]
        assert before == after, ("a relayed request moved the engine's counters", before, after)
        st, _, body = _req(port, "/v1/chat/completions", {
            "model": "qwen38-spark-engine", "messages": [{"role": "user", "content": "hello"}]})
        assert st == 200, body
        assert json.loads(_req(port, "/health")[2])["inflight"]["served"] == \
            before["served"] + 1, "the engine's own model is the engine's"
        st, _, body = _req(port, "/v1/responses", {"model": "fake-a", "input": "x"})
        assert st == 200 and b"resp_fake" in body
        time.sleep(1.5)
        con = sqlite3.connect(ledger)
        rows = con.execute("SELECT model, backend, usage_source, prompt_ms FROM requests "
                           "ORDER BY id").fetchall()
        con.close()
        assert ("fake-a", "f1", "backend", None) in rows, rows
        assert any(r[1] is None and r[3] is not None for r in rows), rows
        for name, path in (("summary", "/v1/dashboard/summary"),
                           ("requests", "/v1/dashboard/requests"),
                           ("usage", "/v1/dashboard/usage?bucket=day&backend=f1"),
                           ("system", "/v1/dashboard/system")):
            errs = contract_check.validate(json.loads(_req(port, path)[2]),
                                           contract_check.load(name))
            assert not errs, (name, errs[:3])
        cfg = json.loads(_req(port, "/v1/config")[2])
        assert cfg["backends"][0]["id"] == "f1" and cfg["engine"]["embedded"]
        # a user's key gates the engine's routes too (a request from afar)
        st, _, body = _req(port, "/v1/config/users", {"label": "t"})
        key = json.loads(body)["key"]
        far = {"X-Forwarded-For": "192.0.2.9", "Authorization": "Bearer nope"}
        st, _, _ = _req(port, "/v1/chat/completions", {
            "model": "qwen38-spark-engine", "messages": [{"role": "user", "content": "x"}]},
            headers=far)
        assert st == 401
        st, _, _ = _req(port, "/v1/chat/completions", {
            "model": "qwen38-spark-engine", "messages": [{"role": "user", "content": "x"}]},
            headers=dict(far, Authorization=f"Bearer {key}"))
        assert st == 200
    finally:
        proc.terminate()
        proc.wait(30)
        fsrv.shutdown()


def test_the_harness_inside_the_engine():
    """P4 in the embedded mode: a backend with its harness on runs under the engine's own
    conductor (one journal, in the ledger's file); the class question goes to the backend; the
    engine's counters do not move."""
    fsrv, fport = fake_backend.start(models=["fake-a"])
    d = tempfile.mkdtemp(prefix="pxeng-h-")
    ledger = os.path.join(d, "ledger.sqlite3")
    port = _port()
    env = {"QSE_TEST": "1", "TANDEM_BACKENDS": "f1", "TANDEM_BACKEND_F1_KIND": "vllm",
           "TANDEM_BACKEND_F1_URL": f"http://127.0.0.1:{fport}",
           "TANDEM_BACKEND_F1_HARNESS": "on",
           "TANDEM_SECRETS_KEY": "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="}
    proc = _start(port, ledger, home=d, extra=(
        "--backends", os.path.join(d, "config.sqlite3"), "--conductor", "on",
        "--conductor-config", os.path.join(d, "conductor"), "--conductor-defaults", "shipped"),
        env_extra=env)
    try:
        _wait(port, proc)
        time.sleep(1.0)
        before = json.loads(_req(port, "/health")[2])["inflight"]
        tools = [{"type": "function", "function": {"name": "read", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}}}}]
        st, h, body = _req(port, "/v1/chat/completions", {
            "model": "fake-a", "stream": True, "tools": tools, "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Fix the failing test in utils.py FAKE_CLASS debug"}]},
            headers={"User-Agent": "opencode/1.18.34 ai-sdk/x"})
        assert st == 200 and h.get("X-Tandem-Backend") == "f1" and body.count(b"[DONE]") == 1
        assert any(b.get("structured_outputs") for b in fsrv.fake.bodies), \
            "the class question went to the backend"
        assert json.loads(_req(port, "/health")[2])["inflight"] == before
        time.sleep(1.5)
        con = sqlite3.connect(ledger)
        rows = con.execute("SELECT kind, extra FROM conductor_events WHERE request_id LIKE "
                           "'chatcmpl-px-%'").fetchall()
        con.close()
        assert any(k == "route" and '"backend": "f1"' in (x or "") for k, x in rows), rows
    finally:
        proc.terminate()
        proc.wait(30)
        fsrv.shutdown()


def test_without_backends_the_proxy_is_not_loaded():
    d = tempfile.mkdtemp(prefix="pxeng0-")
    port = _port()
    proc = _start(port, os.path.join(d, "ledger.sqlite3"), home=d,
                  env_extra={"QSE_TEST": "1", "TANDEM_BACKENDS": "f1"})
    try:
        _wait(port, proc)
        st, _, body = _req(port, "/v1/config")
        assert st == 404 and b"--backends off" in body
        ids = [m["id"] for m in json.loads(_req(port, "/v1/models")[2])["data"]]
        assert ids == ["qwen38-spark-engine"], ids
        st, _, _ = _req(port, "/v1/responses", {"model": "x", "input": "y"})
        assert st == 404
    finally:
        proc.terminate()
        out = proc.communicate(timeout=30)[0]
    assert "[proxy]" not in out, "TANDEM_* is ignored without --backends"


if __name__ == "__main__":
    run(globals())
