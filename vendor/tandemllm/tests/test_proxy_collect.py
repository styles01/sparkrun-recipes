"""Layer L2 at the proxy (proxy/collect_api.py, P6) and the launcher (proxy/launch.py, L3).

The collect route and its token, the tag the relay puts on a request to vLLM (X-Request-Id) and to
SGLang (the body's rid), the join in both orders (the row first, the item first), the approximate
join and its refusal when two rows fit, the waits and their counts, the ledger amended in place,
the backend's plugin block (L2, then "L1 only" after 60 s) valid against the contract, the pull
route of vLLM, and `tandem-dash launch` from nothing to a registered backend with a stand-in
engine.

Run: python tests/test_proxy_collect.py
"""

from __future__ import annotations

import http.server
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import ROOT, Proxy, fake, run  # noqa: E402

from proxy import collect_api as C  # noqa: E402
from proxy import config as config_mod  # noqa: E402
from server import ledger as ledger_mod  # noqa: E402
from tools import contract_check  # noqa: E402

TOKEN = "collect-fixture-token-0123456789"


def _env(kind: str, port: int, bid: str = "eng1") -> dict:
    return {"TANDEM_BACKENDS": bid, f"TANDEM_BACKEND_{bid.upper()}_KIND": kind,
            f"TANDEM_BACKEND_{bid.upper()}_URL": f"http://127.0.0.1:{port}/v1",
            "TANDEM_COLLECT_TOKEN": TOKEN}


def _item(rid, **kw):
    it = {"id": rid, "finish_reason": "stop", "prompt_tokens": None, "cached_tokens": 3,
          "completion_tokens": None, "preemptions": 1, "queue_ms": 12.5, "prefill_ms": 40.0,
          "decode_ms": 900.0, "corrupted": False, "t_end": time.time()}
    it.update(kw)
    return it


def _batch(items=(), steps=(), instance="i1", seq0=0):
    reqs = [dict(it, seq=seq0 + n + 1) for n, it in enumerate(items)]
    return {"plugin": "tandemllm-vllm-plugin", "plugin_version": "0.1.0", "engine": "vllm",
            "engine_version": "0.30.0", "instance": instance, "pid": 1, "sent_at": time.time(),
            "dropped": {"requests": 0, "steps": 0}, "requests": reqs,
            "steps": [dict(s, seq=seq0 + n + 1) for n, s in enumerate(steps)]}


def _collect(px, bid, batch, token=TOKEN):
    return px.post(f"/v1/collect/{bid}", batch,
                   headers={"Authorization": f"Bearer {token}"} if token else {})


def _chat(px, model, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": "say hi please"}]}
    body.update(extra)
    st, _, out = px.post("/v1/chat/completions", body)
    assert st == 200, out
    return json.loads(out)


# ----------------------------------------------------------------------------------- the route

def test_the_route_needs_the_token_and_a_backend():
    srv, port = fake(models=["m-v"], dialect="vllm")
    with Proxy(_env("vllm", port)) as px:
        st, h, _ = _collect(px, "eng1", _batch(), token=None)
        assert st == 401 and "Bearer" in h.get("WWW-Authenticate", "")
        assert _collect(px, "eng1", _batch(), token="wrong-token-wrong-token")[0] == 401
        assert _collect(px, "nope", _batch())[0] == 404
        st, _, out = _collect(px, "eng1", b"{not json")
        assert st == 400
        st, _, out = _collect(px, "eng1", {"requests": "x"})
        assert st == 400
        st, _, out = _collect(px, "eng1", _batch(steps=[{"t": 1, "running": 2}]))
        assert st == 200 and json.loads(out)["accepted"] == {"requests": 0, "steps": 1}
        # the same batch again counts once (sequence numbers per plugin instance)
        st, _, out = _collect(px, "eng1", _batch(steps=[{"t": 1, "running": 2}]))
        assert json.loads(out)["accepted"] == {"requests": 0, "steps": 0}
        assert all(TOKEN not in line for line in px.logs)
    srv.shutdown()


def test_the_token_is_made_once_and_kept():
    d = tempfile.mkdtemp(prefix="px-collect-")
    cfg = config_mod.ConfigDB(os.path.join(d, "c.sqlite3"), env={}, use_keychain=False).open()
    t1 = C.token_of(cfg, {})
    t2 = C.token_of(cfg, {})
    assert t1 == t2 and len(t1) >= 32
    assert C.token_of(cfg, {"TANDEM_COLLECT_TOKEN": "from-the-env-0123"}) == "from-the-env-0123"
    cfg.close()


# ----------------------------------------------------------------------------------- the join

def test_vllm_row_first_then_the_plugin_item():
    srv, port = fake(models=["m-v"], dialect="vllm")
    with Proxy(_env("vllm", port)) as px:
        px.sync()
        out = _chat(px, "m-v")
        sent = srv.fake.last["headers"]
        hdr = {k.lower(): v for k, v in sent.items()}
        rid = hdr["x-request-id"]
        assert rid.startswith("px-")
        row = px.rows()[-1]
        assert row["request_id"] == "chatcmpl-" + rid and row["l2_join"] is None
        assert out["id"]                                 # the fake's own id, passed as sent
        st, _, res = _collect(px, "eng1", _batch([_item("chatcmpl-" + rid)]))
        assert st == 200 and json.loads(res)["joined"] == 1
        row = px.rows()[-1]
        assert row["l2_join"] == "exact" and row["queue_ms"] == 12.5
        assert row["prompt_ms"] == 40.0 and row["decode_ms"] == 900.0
        assert row["cached_tokens"] == 3 and row["preemptions"] == 1
        assert row["engine_finish"] == "stop" and row["corrupted"] == 0
        n_out = row["completion_tokens"]
        if n_out and n_out > 1:
            assert abs(row["decode_tps"] - (n_out - 1) / 0.9) < 0.02
        # the Dev view's row
        req = json.loads(px.get("/v1/dashboard/requests")[2])
        assert not contract_check.validate(req, contract_check.load("requests"))
        r0 = req["requests"][0]
        assert r0["l2"] == {"join": "exact", "preemptions": 1, "engine_finish": "stop",
                            "corrupted": False}
        assert r0["queue_ms"] == 12.5 and r0["prompt_ms"] == 40.0
    srv.shutdown()


def test_a_client_request_id_is_kept_and_becomes_the_key():
    srv, port = fake(models=["m-v"], dialect="vllm")
    with Proxy(_env("vllm", port)) as px:
        px.sync()
        st, _, _ = px.post("/v1/chat/completions", {
            "model": "m-v", "messages": [{"role": "user", "content": "x"}]},
            headers={"X-Request-Id": "client-7"})
        assert st == 200
        hdr = {k.lower(): v for k, v in srv.fake.last["headers"].items()}
        assert hdr["x-request-id"] == "client-7"
        px.rows()
        _collect(px, "eng1", _batch([_item("chatcmpl-client-7")]))
        assert px.rows()[-1]["l2_join"] == "exact"
    srv.shutdown()


def test_sglang_gets_a_rid_in_the_body_and_a_client_rid_is_left_alone():
    srv, port = fake(models=["m-s"], dialect="sglang")
    with Proxy(_env("sglang", port, "sg1")) as px:
        px.sync()
        _chat(px, "m-s")
        body = srv.fake.last["body"]
        assert body["rid"].startswith("chatcmpl-px-")
        rid = body["rid"]
        assert px.rows()[-1]["request_id"] == rid
        _collect(px, "sg1", _batch([_item(rid, spec_verify_ct=20, spec_accepted=40,
                                         spec_draft_tokens=60)]))
        row = px.rows()[-1]
        assert row["l2_join"] == "exact" and row["blocks"] == 20
        assert (row["draft_tokens"], row["draft_accepted"]) == (60, 40)
        _chat(px, "m-s", rid="mine-1")
        assert srv.fake.last["body"]["rid"] == "mine-1"
    srv.shutdown()


class _Rec:
    def __init__(self, rid, kind="vllm", bid="b"):
        self.request_id, self.backend, self.backend_kind = rid, bid, kind


class _Svc:
    def __init__(self):
        self.ledger = types.SimpleNamespace(amends=[], amend=lambda rid, f, since_ms=0:
                                            self.ledger.amends.append((rid, f)))
        self.logs = []
        self.log = self.logs.append
        self.cfg = types.SimpleNamespace(backends=lambda: [], setting=lambda k: None,
                                         set_setting=lambda *a, **k: None)


def test_item_first_then_the_row_fills_it_before_the_ledger():
    svc = _Svc()
    col = C.Collector(svc, env={"TANDEM_COLLECT_TOKEN": TOKEN})
    rec = _Rec("chatcmpl-px-1")
    body, hdrs = col.tag({"id": "b", "kind": "vllm"}, "/v1/chat/completions", {}, b"{}", [], rec)
    assert ("X-Request-Id", "px-1") in hdrs and rec.l2_key == "chatcmpl-px-1"
    res = col.ingest("b", _batch([_item("chatcmpl-px-1", completion_tokens=11)]))
    assert res["joined"] == 0 and svc.ledger.amends == []      # waits for its row
    row = {"ts_ms": 1, "prompt_tokens": 50, "completion_tokens": 11}
    col.on_row(rec, row)
    assert row["l2_join"] == "exact" and row["decode_ms"] == 900.0
    assert row["decode_tps"] == round(10 / 0.9, 2)
    assert row["prefill_tps"] == round((50 - 3) / 0.04, 2)
    # a completions request: vLLM reports cmpl-<id>
    rec2 = _Rec("px-2")
    col.tag({"id": "b", "kind": "vllm"}, "/v1/completions", {}, b"{}", [], rec2)
    assert rec2.l2_key == "cmpl-px-2"
    # a cloud or generic backend is never tagged
    rec3 = _Rec("chatcmpl-px-3", kind="openai")
    _, h3 = col.tag({"id": "o", "kind": "openai"}, "/v1/chat/completions", {}, b"{}", [], rec3)
    assert h3 == [] and not hasattr(rec3, "l2_key")
    # SGLang with n > 1 makes `<rid>_<i>` of a rid: such a request is not tagged
    rec4 = _Rec("chatcmpl-px-4", kind="sglang")
    b4, _ = col.tag({"id": "s", "kind": "sglang"}, "/v1/chat/completions", {"n": 2},
                    b'{"n": 2}', [], rec4)
    assert b4 == b'{"n": 2}' and not hasattr(rec4, "l2_key")


def test_approximate_join_for_an_untagged_row_and_no_guess_between_two():
    svc = _Svc()
    clock = [100.0]
    wall = [5000.0]
    col = C.Collector(svc, env={"TANDEM_COLLECT_TOKEN": TOKEN}, clock=lambda: clock[0],
                      wall=lambda: wall[0])
    a = _Rec("resp-a")                       # the Responses API: not tagged
    col.on_row(a, {"ts_ms": 1, "prompt_tokens": 70, "completion_tokens": 9})
    res = col.ingest("b", _batch([_item("resp_engine_1", prompt_tokens=70,
                                        completion_tokens=9, t_end=5001.0)]))
    assert res["joined"] == 1 and svc.ledger.amends[-1][0] == "resp-a"
    assert svc.ledger.amends[-1][1]["l2_join"] == "approximate"
    assert any("join approximate" in m for m in svc.logs)
    # two rows that fit: no guess; the item waits, then is dropped and counted
    for rid in ("r1", "r2"):
        col.on_row(_Rec(rid), {"ts_ms": 1, "prompt_tokens": 5, "completion_tokens": 5})
    res = col.ingest("b", _batch([_item("e9", prompt_tokens=5, completion_tokens=5,
                                        t_end=5000.5)], seq0=1))
    assert res["joined"] == 0
    clock[0] += 61
    col.tick()
    assert col.backends["b"].counts["item_untagged"] == 1
    # a tagged row becomes a candidate only after 5 s without its item
    t = _Rec("chatcmpl-px-t")
    col.tag({"id": "b", "kind": "vllm"}, "/v1/chat/completions", {}, b"{}", [], t)
    col.on_row(t, {"ts_ms": 1, "prompt_tokens": 8, "completion_tokens": 8})
    col.ingest("b", _batch([_item("zzz", prompt_tokens=8, completion_tokens=8,
                                  t_end=5000.2)], seq0=2))
    assert svc.ledger.amends[-1][0] == "resp-a"                  # not yet
    clock[0] += 6
    col.tick()
    assert svc.ledger.amends[-1][0] == "chatcmpl-px-t"


def test_rows_and_items_wait_ten_minutes():
    svc = _Svc()
    clock = [0.0]
    col = C.Collector(svc, env={"TANDEM_COLLECT_TOKEN": TOKEN}, clock=lambda: clock[0])
    r = _Rec("chatcmpl-px-w")
    col.tag({"id": "b", "kind": "vllm"}, "/v1/chat/completions", {}, b"{}", [], r)
    col.on_row(r, {"ts_ms": 1})
    clock[0] = 599.0
    col.tick()
    assert ("b", "chatcmpl-px-w") in col.done
    clock[0] = 601.0
    col.tick()
    assert col.backends["b"].counts["row_unmatched"] == 1 and not col.done


def test_the_ledger_amends_a_written_row_in_place():
    d = tempfile.mkdtemp(prefix="px-led-")
    led = ledger_mod.Ledger(os.path.join(d, "ledger.sqlite3")).open()
    now = int(time.time() * 1000)
    led.submit({"ts_ms": now, "request_id": "chatcmpl-px-a", "model": "m", "client_id": "anon",
                "client_kind": "curl", "endpoint": "chat", "stream": 1, "status": 200,
                "backend": "b", "tool_calls": 0, "thinking": 0})
    assert led.amend("chatcmpl-px-a", {"queue_ms": 5.0, "l2_join": "exact", "model": "nope"},
                     since_ms=now - 60_000)
    assert not led.amend("chatcmpl-px-a", {"model": "nope"})       # not an amendable column
    led.flush(5)
    con = sqlite3.connect(led.path)
    got = con.execute("SELECT queue_ms, l2_join, model FROM requests").fetchone()
    con.close()
    led.close()
    assert got == (5.0, "exact", "m")
    assert ledger_mod.SCHEMA_VERSION >= 4          # 5: wait_own_ms (SRV-177), additive


# ----------------------------------------------------------------------------------- the view

def test_backend_block_says_l2_then_l1_only_and_validates():
    srv, port = fake(models=["m-v"], dialect="vllm")
    with Proxy(_env("vllm", port)) as px:
        px.sync()
        px.svc.engines.tick()
        steps = [{"t": time.time(), "steps": 20, "running": 2, "waiting": 1, "kv_usage": 0.3,
                  "prefix_queries": 200, "prefix_hits": 80, "preempted": 1, "evictions": 2,
                  "spec": {"drafts": 10, "draft_tokens": 30, "accepted": 15,
                           "accepted_per_pos": [8, 5, 2]}}]
        _collect(px, "eng1", _batch(steps=steps))
        sysb = json.loads(px.get("/v1/dashboard/system")[2])
        b = sysb["proxy"]["backends"][0]
        e = b["engine"]
        assert e["layer"] == "L2" and b["plugin_age_s"] is not None
        p = e["plugin"]
        assert p["state"] == "l2" and p["plugin_version"] == "0.1.0"
        assert p["engine_version"] == "0.30.0" and p["via"] == "push"
        w = p["window"]
        assert w["prefix_hit_rate"] == 0.4 and w["preempted"] == 1 and w["evictions"] == 2
        assert w["spec"] == {"acceptance": 0.5, "tokens_per_step": 2.5,
                             "per_position": [0.8, 0.5, 0.2]}
        for name, path in (("system", "/v1/dashboard/system"),
                           ("summary", "/v1/dashboard/summary")):
            errs = contract_check.validate(json.loads(px.get(path)[2]),
                                           contract_check.load(name))
            assert not errs, (name, errs[:3])
        page = px.get("/metrics")[2].decode()
        assert 'qse_backend_plugin_age_seconds{backend="eng1"}' in page
        assert 'qse_backend_plugin_joins{backend="eng1",join="exact"} 0' in page
        px.svc.collect.backends["eng1"].last_batch -= 61
        e = json.loads(px.get("/v1/dashboard/system")[2])["proxy"]["backends"][0]["engine"]
        assert e["layer"] == "L1" and e["plugin"]["state"] == "l1_only"
    srv.shutdown()


# ----------------------------------------------------------------------------------- pull (vLLM)

def _stats_server(answer):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            code, body = answer(self)
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_pull_reads_the_route_and_backs_off_on_404():
    seen = []

    def answer(h):
        seen.append((h.path, h.headers.get("Authorization")))
        if h.path.startswith("/tandem/stats"):
            return 200, _batch(steps=[{"t": 1, "running": 4}], instance="pull1")
        return 404, {}
    srv, port = _stats_server(answer)
    svc = _Svc()
    svc.cfg.backends = lambda: [{"id": "v", "kind": "vllm", "enabled": True,
                                 "url": f"http://127.0.0.1:{port}/v1"}]
    col = C.Collector(svc, env={"TANDEM_COLLECT_TOKEN": TOKEN})
    col._pull_round()
    assert seen and seen[0][0] == "/tandem/stats?after_requests=0&after_steps=0"
    assert seen[0][1] == f"Bearer {TOKEN}"
    v = col.view("v")
    assert v["via"] == "pull" and v["last"] == {"running": 4}
    col.backends["v"].pull["next"] = 0
    col._pull_round()
    assert seen[-1][0].endswith("after_requests=0&after_steps=1")
    srv.shutdown()

    def missing(h):
        return 404, {}
    srv2, port2 = _stats_server(missing)
    svc.cfg.backends = lambda: [{"id": "w", "kind": "vllm", "enabled": True,
                                 "url": f"http://127.0.0.1:{port2}/v1"}]
    col._pull_round()
    st = col.backends["w"].pull
    assert st["state"] == "off" and st["next"] - col.clock() > 100
    srv2.shutdown()


# ----------------------------------------------------------------------------------- the launcher

STAND_IN = ("import json, os, sys; print('STANDIN ' + json.dumps({'pid': os.getpid(), "
            "'argv': sys.argv[1:], 'env': {k: v for k, v in os.environ.items() "
            "if k.startswith('TANDEM_COLLECT') or k.endswith('_PLUGINS')}}))")


def test_launch_registers_the_backend_and_execs_the_engine():
    d = tempfile.mkdtemp(prefix="px-launch-")
    db = os.path.join(d, "config.sqlite3")
    stand_in = f"{sys.executable} -c \"{STAND_IN}\""
    env = dict(os.environ, VLLM_PLUGINS="other", TANDEM_PORT="8188")
    env.pop("TANDEM_COLLECT_TOKEN", None)
    r = subprocess.run([sys.executable, "-m", "proxy", "launch", "vllm", "--model", "Qwen/Q",
                        "--port", "8111", "--config-db", db, "--engine-cmd", stand_in,
                        "--python", sys.executable, "--", "--max-model-len", "4096"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    line = next(x for x in r.stdout.splitlines() if x.startswith("STANDIN "))
    got = json.loads(line[len("STANDIN "):])
    assert got["argv"][:1] == ["Qwen/Q"]
    assert "--enable-prompt-tokens-details" in got["argv"]
    assert got["argv"][-2:] == ["--max-model-len", "4096"]
    assert got["env"]["TANDEM_COLLECT_URL"] == "http://127.0.0.1:8188/v1/collect/vllm8111"
    assert got["env"]["VLLM_PLUGINS"] == "other,tandem_stat_logger"
    assert "the plugin is not installed" in r.stderr           # not in this Python
    cfg = config_mod.ConfigDB(db, env={}, use_keychain=False).open()
    b = cfg.backend("vllm8111")
    assert b["kind"] == "vllm" and b["url"] == "http://127.0.0.1:8111/v1"
    assert b["host"]["mode"] == "same"
    assert cfg.setting("launch:vllm8111")["pid"] == got["pid"]     # exec kept the pid
    assert got["env"]["TANDEM_COLLECT_TOKEN"] == cfg.setting("collect_token")
    said = [x for x in (r.stdout + r.stderr).splitlines() if not x.startswith("STANDIN ")]
    assert not any(got["env"]["TANDEM_COLLECT_TOKEN"] in x for x in said)  # the launcher's lines
    cfg.close()
    # SGLang: the flags it needs, its plugin list left unset when it was unset
    env.pop("VLLM_PLUGINS")
    r = subprocess.run([sys.executable, "-m", "proxy", "launch", "sglang", "--model", "Qwen/Q",
                        "--config-db", db, "--engine-cmd", stand_in, "--python", sys.executable,
                        "--id", "sg"], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    got = json.loads(next(x for x in r.stdout.splitlines()
                          if x.startswith("STANDIN "))[len("STANDIN "):])
    assert got["argv"][:2] == ["--model-path", "Qwen/Q"]
    assert "--enable-metrics" in got["argv"] and "--enable-cache-report" in got["argv"]
    assert "SGLANG_PLUGINS" not in got["env"]
    # a dry run starts nothing and registers nothing
    r = subprocess.run([sys.executable, "-m", "proxy", "launch", "vllm", "--model", "Q",
                        "--config-db", db, "--engine-cmd", stand_in, "--python", sys.executable,
                        "--id", "dry", "--dry-run"], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and "[launch] exec" in r.stdout
    assert not any(x.startswith("STANDIN ") for x in r.stdout.splitlines())
    cfg = config_mod.ConfigDB(db, env={}, use_keychain=False).open()
    assert cfg.backend("dry") is None
    cfg.close()


if __name__ == "__main__":
    run(globals())
