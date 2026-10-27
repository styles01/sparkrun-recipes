"""The relay and the router against fake backends (tests/fake_backend.py), on a CPU, in seconds:
the stream byte for byte, the usage frame asked for and dropped, fallbacks and retries only
before the first byte, max_inflight, a client that leaves, the idle timeout, /v1/models with real
ids and the prefixed form on a collision, routing rules, the users' keys, the ledger row, and no
key in any output.

The night pass (tools/proxy_e2e.sh) adds the dashboard, our engine through the proxy, and the
latency and concurrency numbers.

Run: python tests/test_proxy_relay.py
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import FIXTURE_KEYS, Proxy, fake, run  # noqa: E402

from tools import contract_check  # noqa: E402

KEY_A, KEY_B = FIXTURE_KEYS[0], FIXTURE_KEYS[1]


def _env(*backs, extra=None):
    env = {"TANDEM_BACKENDS": ",".join(b for b, _ in backs)}
    for bid, port in backs:
        up = bid.upper()
        env[f"TANDEM_BACKEND_{up}_KIND"] = "generic"
        env[f"TANDEM_BACKEND_{up}_URL"] = f"http://127.0.0.1:{port}"
    env.update(extra or {})
    return env


def _chat(text, model="fake-a", stream=False, **kw):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "stream": stream}
    body.update(kw)
    return body


def _direct(port, body, key=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    c.request("POST", "/v1/chat/completions", json.dumps(body), h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return data


def test_stream_byte_identical_and_usage_frame_dropped():
    srv, port = fake(models=["fake-a"], key=KEY_A, cache=False)
    with Proxy(_env(("f1", port), extra={"TANDEM_BACKEND_F1_KEY": KEY_A})) as px:
        px.sync()
        # a client that asked for usage: what the backend sent, byte for byte
        asked = _chat("hello FAKE_REASONING", stream=True,
                      stream_options={"include_usage": True})
        want = _direct(port, asked, KEY_A)
        st, hdrs, got = px.post("/v1/chat/completions", asked)
        assert st == 200 and got == want, (got[-300:], want[-300:])
        assert hdrs.get("X-Tandem-Backend") == "f1"
        # a client that did not ask: the same bytes less the one usage frame the proxy asked for
        plain = _chat("hello FAKE_REASONING", stream=True)
        st, _, got = px.post("/v1/chat/completions", plain)
        frames = want.split(b"\n\n")
        usage_frames = [f for f in frames if b'"choices": []' in f]
        assert len(usage_frames) == 1
        assert got == b"\n\n".join(f for f in frames if f not in usage_frames)
        assert srv.fake.last["body"]["stream_options"] == {"include_usage": True}
        assert srv.fake.last["headers"]["Authorization"] == f"Bearer {KEY_A}"
        rows = px.rows()
        r = rows[-1]
        assert r["backend"] == "f1" and r["usage_source"] == "backend"
        assert r["prompt_tokens"] == 6 and r["completion_tokens"] == 30
        assert r["reasoning_tokens"] == 6 and r["thinking"] == 1
        assert r["decode_tps"] and r["ttft_ms"] is not None and r["prompt_ms"] is None
        assert r["prefill_tps"] is None and r["blocks"] is None and r["cache_source"] is None
        assert r["queue_ms"] == 0.0 and r["proxy_ms"] is not None and r["retries"] == 0
        # every output: no fixture key anywhere
        bodies = [px.get(p)[2] for p in ("/v1/config", "/v1/config/backends/f1",
                                         "/v1/dashboard/requests", "/v1/dashboard/system",
                                         "/metrics", "/v1/models", "/health")]
        for k in FIXTURE_KEYS:
            assert not any(k.encode() in b for b in bodies), k
            for path in px.files():
                with open(path, "rb") as f:
                    assert k.encode() not in f.read(), (k, path)
            assert not any(k in line for line in px.logs)
    srv.shutdown()


def _first_byte_ms(port, body, path="/v1/chat/completions"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    t0 = time.perf_counter()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json",
                                                "Connection": "close"})
    r = c.getresponse()
    first = None
    while True:
        line = r.readline()
        if first is None and line.startswith(b"data: "):
            first = (time.perf_counter() - t0) * 1e3
        if not line:
            break
    c.close()
    return first


def test_the_first_frame_is_not_held_and_64_clients_connect_at_once():
    """Night pass 2026-10-04 on the Linux box: tools/proxy_bench.py measured 41.6 ms added to the
    first frame and 10 of 64 clients that connected at once got a reset. The 41.6 ms was the fake
    backend's (Nagle on its pooled keep-alive connection: headers and first frame as two writes,
    held for the proxy's delayed ACK); the proxy's own side adds 1 to 2 ms. The resets were the
    proxy's: a listen backlog of 5."""
    srv, port = fake(models=["fake-a"], cache=False)
    with Proxy(_env(("f1", port))) as px:
        px.sync()
        body = _chat("hello", stream=True)
        for _ in range(3):
            _first_byte_ms(port, body), _first_byte_ms(px.port, body)
        direct = sorted(_first_byte_ms(port, body) for _ in range(15))[7]
        via = sorted(_first_byte_ms(px.port, body) for _ in range(15))[7]
        assert via - direct < 15.0, (direct, via)
        ok = []
        gate = threading.Barrier(64)

        def one():
            gate.wait()                                   # 64 connects in the same moment
            try:
                s = socket.create_connection(("127.0.0.1", px.port), timeout=10)
                req = json.dumps(_chat("hi", stream=True)).encode()
                s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                          b"Content-Type: application/json\r\nConnection: close\r\n"
                          + f"Content-Length: {len(req)}\r\n\r\n".encode() + req)
                data = b""
                while True:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                s.close()
                ok.append(data.startswith(b"HTTP/1.1 200") and b"[DONE]" in data)
            except OSError:
                ok.append(False)
        ts = [threading.Thread(target=one) for _ in range(64)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        assert ok.count(True) == 64, ok.count(True)
    srv.shutdown()
    return f"first frame: direct {direct:.1f} ms, through the proxy {via:.1f} ms; 64 of 64 connected"


def test_no_usage_means_null_speeds_and_counts():
    srv, port = fake(models=["fake-a"])
    with Proxy(_env(("f1", port))) as px:
        px.sync()
        st, _, got = px.post("/v1/chat/completions", _chat("x FAKE_NO_USAGE", stream=True))
        assert st == 200 and b"[DONE]" in got
        r = px.rows()[-1]
        assert r["usage_source"] == "none" and r["completion_tokens"] is None
        assert r["decode_tps"] is None and r["ttft_ms"] is not None, \
            "a frame is not a token: no speed without usage"
    srv.shutdown()


def test_fallback_and_retries_before_the_first_byte():
    s1, p1 = fake(models=["fake-a"], refuse=503)
    s2, p2 = fake(models=["fake-a"], name="second")
    env = _env(("f1", p1), ("f2", p2), extra={"TANDEM_BACKEND_F1_FALLBACK": "f2",
                                              "TANDEM_ROUTE_1": "fake-*=f1"})
    with Proxy(env) as px:
        px.sync()
        st, _, _ = px.post("/v1/chat/completions", _chat("x"))
        assert st == 200
        r = px.rows()[-1]
        assert r["backend"] == "f2" and r["fallback_from"] == "f1" and r["retries"] == 1
        assert r["upstream_status"] == 200
        # FAKE_429 everywhere: two tries, then the 429 with its Retry-After
        t0 = time.time()
        st, h, _ = px.post("/v1/chat/completions", _chat("y FAKE_429"))
        assert st == 429 and h.get("Retry-After") == "1", (st, h)
        r = px.rows()[-1]
        assert r["retries"] == 2 and r["finish_reason"] == "refused"
        assert time.time() - t0 >= 0.9, "the same backend is tried again after Retry-After"
    s1.shutdown()
    s2.shutdown()
    s3, p3 = fake(models=["fake-a"])
    with Proxy(_env(("f3", p3))) as px:
        px.sync()
        st, _, _ = px.post("/v1/chat/completions", _chat("z FAKE_503_ONCE"))
        r = px.rows()[-1]
        assert st == 200 and r["retries"] == 1 and r["fallback_from"] is None
    s3.shutdown()
    return
    s2.shutdown()


def test_max_inflight_answers_503():
    srv, port = fake(models=["fake-a"])
    with Proxy(_env(("f1", port), extra={"TANDEM_BACKEND_F1_MAX_INFLIGHT": "1"})) as px:
        px.sync()
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("a", px.post(
            "/v1/chat/completions", _chat("slow FAKE_SLOW_TTFT 1.0"))))
        t.start()
        time.sleep(0.3)
        st, h, body = px.post("/v1/chat/completions", _chat("second"))
        t.join()
        assert out["a"][0] == 200 and st == 503 and h.get("Retry-After") == "1", (st, body)
        assert json.loads(body)["error"]["code"] == "max_inflight"
    srv.shutdown()


def test_no_limit_from_us_by_default_for_vllm_and_cloud_kinds():
    """PL1 (2026-10-06): the parallel setting is our engine's; a relayed backend (vLLM, SGLang, a
    cloud API) with no explicit max_inflight takes as many requests at once as it accepts. 80 at
    once (past the old default of 64) all reach it, and none is refused by the proxy."""
    srv, port = fake(models=["fake-a"])
    env = {"TANDEM_BACKENDS": "v1,oa",
           "TANDEM_BACKEND_V1_KIND": "vllm", "TANDEM_BACKEND_V1_URL": f"http://127.0.0.1:{port}/v1",
           "TANDEM_BACKEND_V1_MODELS": "fake-a", "TANDEM_BACKEND_V1_MODELS_MODE": "list",
           "TANDEM_BACKEND_OA_KIND": "openai", "TANDEM_BACKEND_OA_URL": f"http://127.0.0.1:{port}/v1",
           "TANDEM_BACKEND_OA_KEY": FIXTURE_KEYS[2], "TANDEM_BACKEND_OA_MODELS": "fake-b",
           "TANDEM_BACKEND_OA_MODELS_MODE": "list"}
    with Proxy(env) as px:
        px.sync()
        svc = px.svc
        for bid in ("v1", "oa"):
            assert svc.cfg.backend(bid)["max_inflight"] is None
        n, peak, out = 80, {"v1": 0}, []
        lock = threading.Lock()

        def one(i):
            r = px.post("/v1/chat/completions", _chat(f"r{i} FAKE_SLOW_TTFT 1.5"), timeout=60)
            with lock:
                out.append(r[0])

        stop = threading.Event()

        def watch():
            while not stop.is_set():
                peak["v1"] = max(peak["v1"], svc.relay.inflight.count("v1"))
                time.sleep(0.02)

        w = threading.Thread(target=watch)
        w.start()
        ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        stop.set()
        w.join()
        assert out.count(200) == n, sorted(set(out))
        assert peak["v1"] > 64, peak
        # an explicit cap still works, set from the GUI's store, and none takes it off again
        svc.cfg.update_backend("v1", {"max_inflight": 2})
        assert svc.cfg.backend("v1")["max_inflight"] == 2
        svc.cfg.update_backend("v1", {"max_inflight": None})
        assert svc.cfg.backend("v1")["max_inflight"] is None
    srv.shutdown()


def test_client_leaves_and_upstream_closes_within_a_second():
    srv, port = fake(models=["fake-a"], tps=20)
    with Proxy(_env(("f1", port))) as px:
        px.sync()
        s = socket.create_connection(("127.0.0.1", px.port))
        body = json.dumps(_chat("go FAKE_LONG 400", stream=True)).encode()
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: "
                  b"application/json\r\nContent-Length: " + str(len(body)).encode() +
                  b"\r\n\r\n" + body)
        s.recv(4096)
        time.sleep(0.3)
        s.close()
        t0 = time.time()
        while srv.fake.closed_early == 0 and time.time() - t0 < 3:
            time.sleep(0.05)
        assert srv.fake.closed_early == 1 and time.time() - t0 < 1.5, time.time() - t0
        time.sleep(0.3)
        assert px.rows()[-1]["finish_reason"] == "abandoned"
    srv.shutdown()


def test_idle_timeout_is_its_own_reason_and_a_dropped_stream_ends_in_error():
    srv, port = fake(models=["fake-a"])
    with Proxy(_env(("f1", port), extra={"TANDEM_BACKEND_F1_TIMEOUT_IDLE": "1"})) as px:
        px.sync()
        st, _, got = px.post("/v1/chat/completions", _chat("x FAKE_STALL 3", stream=True))
        assert b"timeout_idle" in got and got.endswith(b"data: [DONE]\n\n")
        r = px.rows()[-1]
        assert r["finish_reason"] == "timeout" and r["error_type"] == "timeout_idle"
        st, _, got = px.post("/v1/chat/completions", _chat("x FAKE_DROP", stream=True))
        assert b"upstream_closed" in got
        assert px.rows()[-1]["finish_reason"] == "error"
    srv.shutdown()


def test_models_real_ids_collision_rules_and_404():
    s1, p1 = fake(models=["fake-a", "shared"])
    s2, p2 = fake(models=["shared", "fake-b"])
    with Proxy(_env(("f1", p1), ("f2", p2))) as px:
        px.sync()
        ids = {m["id"]: m["owned_by"] for m in json.loads(px.get("/v1/models")[2])["data"]}
        assert ids == {"fake-a": "f1", "fake-b": "f2", "f1/shared": "f1", "f2/shared": "f2"}
        st, _, body = px.post("/v1/chat/completions", _chat("x", model="shared"))
        assert st == 404 and json.loads(body)["error"]["code"] == "model_not_found"
        assert b"f1/shared" in body
        st, h, _ = px.post("/v1/chat/completions", _chat("x", model="f2/shared"))
        assert st == 200 and h["X-Tandem-Backend"] == "f2"
        assert s2.fake.last["body"]["model"] == "shared", "the backend gets its own id"
        px.svc.cfg.set_rules([{"pattern": "shar*", "backend": "f1"}])
        st, h, _ = px.post("/v1/chat/completions", _chat("x", model="shared"))
        assert st == 200 and h["X-Tandem-Backend"] == "f1"
        st, _, body = px.post("/v1/chat/completions", _chat("x", model="nope"))
        assert st == 404 and json.loads(body)["error"]["code"] == "model_not_found"
        r = px.rows()[-1]
        assert r["status"] == 404 and r["error_type"] == "model_not_found"
    s1.shutdown()
    s2.shutdown()


def test_users_keys_401_and_label_in_the_row():
    srv, port = fake(models=["fake-a"])
    with Proxy(_env(("f1", port))) as px:
        px.sync()
        st, _, body = px.request("POST", "/v1/config/users", {"label": "laptop"})
        key = json.loads(body)["key"]
        # the test client is on loopback without proxy headers: trusted-local, so it needs no key;
        # through a reverse proxy (X-Forwarded-For) it does
        fwd = {"X-Forwarded-For": "192.0.2.7"}
        st, _, _ = px.post("/v1/chat/completions", _chat("x"), headers=fwd)
        assert st == 401
        st, _, _ = px.post("/v1/chat/completions", _chat("x"),
                           headers=dict(fwd, Authorization=f"Bearer {key}"))
        assert st == 200
        # by status, not by position: under the box suite's 4-way load the 401's row was written
        # after the 200's (2026-10-05, L2)
        rows = px.rows()
        r = [x for x in rows if x["status"] == 200][-1]
        assert r["user"] == "laptop" and r["client_id"].startswith("k:"), r
        assert [x["status"] for x in rows].count(401) == 1, rows
        users = json.loads(px.get("/v1/config/users")[2])["users"]
        assert key not in json.dumps(users)
    srv.shutdown()


def test_responses_passthrough_and_deepseek_usage_and_contract():
    srv, port = fake(models=["fake-a", "deepseek-chat"])
    env = _env(("f1", port), extra={"TANDEM_PRICE_1": "f1/deepseek-chat=0.27,0.07,1.10"})
    with Proxy(env) as px:
        px.sync()
        st, _, got = px.post("/v1/responses", {"model": "fake-a", "input": "hi FAKE_CACHED 2",
                                               "stream": True})
        assert st == 200 and b"response.completed" in got
        r = px.rows()[-1]
        assert r["endpoint"] == "responses" and r["cached_tokens"] == 2
        assert r["completion_tokens"] == 15 and r["reasoning_tokens"] == 3
        st, _, _ = px.post("/v1/chat/completions",
                           _chat("x FAKE_DEEPSEEK FAKE_CACHED 3", model="deepseek-chat"))
        r = px.rows()[-1]
        assert r["cached_tokens"] == 3 and r["cost_micros"] == round(
            (r["prompt_tokens"] - 3) * 0.27 + 3 * 0.07 + r["completion_tokens"] * 1.10)
        for name, path in (("requests", "/v1/dashboard/requests"),
                           ("usage", "/v1/dashboard/usage?bucket=hour&from=2026-10-01"),
                           ("summary", "/v1/dashboard/summary"),
                           ("system", "/v1/dashboard/system")):
            doc = json.loads(px.get(path)[2])
            errs = contract_check.validate(doc, contract_check.load(name))
            assert not errs, (name, errs[:3])
        u = json.loads(px.get("/v1/dashboard/usage?backend=f1&bucket=day")[2])
        assert u["totals"]["requests"] == 2 and u["totals"]["cost_micros"] > 0
        assert json.loads(px.get("/v1/dashboard/usage?backend=engine")[2])["totals"][
            "requests"] == 0
    srv.shutdown()


def test_cost_comment_before_done_only_when_asked():
    """`X-Tandem-Cost: 1` (chatd): one `: tandem-cost` comment just before [DONE], the price its
    ledger row has; DeepSeek's cache split priced at the cached rate. Without the header the
    stream carries none."""
    srv, port = fake(models=["deepseek-chat"])
    env = _env(("f1", port), extra={"TANDEM_PRICE_1": "f1/deepseek-chat=0.27,0.07,1.10"})
    body = _chat("x FAKE_DEEPSEEK FAKE_CACHED 3 FAKE_REASONING", model="deepseek-chat",
                 stream=True, stream_options={"include_usage": True})
    with Proxy(env) as px:
        px.sync()
        st, _, plain = px.post("/v1/chat/completions", body)
        assert st == 200 and b"tandem-cost" not in plain
        st, _, got = px.post("/v1/chat/completions", body, headers={"X-Tandem-Cost": "1"})
        assert st == 200
        events = got.split(b"\n\n")
        i = next(k for k, e in enumerate(events) if e.startswith(b": tandem-cost "))
        assert events[i + 1] == b"data: [DONE]" and sum(
            e.startswith(b": tandem-cost") for e in events) == 1
        said = json.loads(events[i][len(b": tandem-cost "):])
        row = px.rows()[-1]
        assert said == {"cost_micros": row["cost_micros"], "backend": "f1"}
        assert row["cached_tokens"] == 3 and row["cost_micros"] == round(
            (row["prompt_tokens"] - 3) * 0.27 + 3 * 0.07 + row["completion_tokens"] * 1.10)
        # the rest of the stream is the backend's, byte for byte
        assert got.replace(events[i] + b"\n\n", b"") == plain
    srv.shutdown()


def test_config_api_add_test_lock_delete():
    srv, port = fake(models=["fake-a"], key=KEY_B)
    with Proxy(_env(("f1", port))) as px:
        st, _, body = px.post("/v1/config/test", {"kind": "generic",
                                                  "url": f"http://127.0.0.1:{port}"})
        t = json.loads(body)
        assert st == 200 and not t["ok"] and "refused" in t["error"]
        st, _, body = px.post("/v1/config/test", {"kind": "generic", "key": KEY_B,
                                                  "url": f"http://127.0.0.1:{port}"})
        t = json.loads(body)
        assert t["ok"] and t["models"] == ["fake-a"] and t["latency_ms"] > 0
        st, _, body = px.post("/v1/config/backends", {"id": "g2", "kind": "generic",
                                                      "url": f"http://127.0.0.1:{port}",
                                                      "key": KEY_B})
        b = json.loads(body)
        assert st == 201 and b["key"]["key_set"] and b["key"]["last4"] == "9876"
        assert KEY_B not in body.decode() and b["health"]["state"] == "up"
        st, _, body = px.request("PATCH", "/v1/config/backends/f1", {"url": "http://h:1"})
        e = json.loads(body)["error"]
        assert st == 422 and e["variable"] == "TANDEM_BACKEND_F1_URL"
        st, _, body = px.request("DELETE", "/v1/config/backends/f1")
        assert st == 422
        st, _, _ = px.request("DELETE", "/v1/config/backends/g2")
        assert st == 204 and px.svc.cfg.backend("g2") is None
        # no admin password configured: a write from afar goes through (the admin rule waits for a
        # password; with one set it needs the token, tests/test_auth.py)
        st, _, _ = px.request("POST", "/v1/config/backends", {"id": "x", "kind": "generic",
                                                              "url": "http://h:1"},
                              headers={"X-Forwarded-For": "192.0.2.7"})
        assert st == 201, st
        st, _, _ = px.request("DELETE", "/v1/config/backends/x",
                              headers={"X-Forwarded-For": "192.0.2.7"})
        assert st == 204
    srv.shutdown()


if __name__ == "__main__":
    run(globals())
