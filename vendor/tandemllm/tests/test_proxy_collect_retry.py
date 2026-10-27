"""The pull round of proxy/collect_api.py survives a sample that fails (audit P2 #6).

One backend's sample failing — transient or unexpected — must not leave that backend
unsampled forever, and must not cost the other backends their round.

Run: python tests/test_proxy_collect_retry.py
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import run  # noqa: E402

from proxy import collect_api as C  # noqa: E402

TOKEN = "collect-fixture-token-0123456789"


class _Svc:
    def __init__(self):
        self.ledger = None
        self.logs = []
        self.log = self.logs.append
        self.cfg = types.SimpleNamespace(backends=lambda: [], setting=lambda k: None,
                                         set_setting=lambda *a, **k: None)


def _batch(steps=()):
    return {"plugin": "p", "instance": "i1", "pid": 1, "requests": [],
            "steps": [dict(s, seq=n + 1) for n, s in enumerate(steps)]}


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


def test_a_transient_pull_error_is_asked_again_after_the_backoff():
    calls = {"n": 0}

    def answer(h):
        calls["n"] += 1
        if calls["n"] == 1:
            return 500, {}                                   # the transient failure
        return 200, _batch(steps=[{"t": 1, "running": 4}])
    srv, port = _stats_server(answer)
    svc = _Svc()
    svc.cfg.backends = lambda: [{"id": "v", "kind": "vllm", "enabled": True,
                                 "url": f"http://127.0.0.1:{port}/v1"}]
    col = C.Collector(svc, env={"TANDEM_COLLECT_TOKEN": TOKEN})
    col._pull_round()
    assert col.backends["v"].pull["state"] == "error"
    col._pull_round()                                        # backing off: not asked again
    assert calls["n"] == 1
    col.backends["v"].pull["next"] = 0.0                     # a later round, once it is due
    col._pull_round()
    assert calls["n"] == 2
    assert col.backends["v"].pull["state"] == "on"
    assert col.view("v")["last"] == {"running": 4}           # the window recovered
    srv.shutdown()


def test_one_failing_backend_does_not_skip_the_others_round():
    srv, port = _stats_server(lambda h: (200, _batch(steps=[{"t": 1, "running": 2}])))
    svc = _Svc()
    svc.cfg.backends = lambda: [
        {"id": "x", "kind": "vllm", "enabled": True},        # no url: the sample itself raises
        {"id": "v", "kind": "vllm", "enabled": True, "url": f"http://127.0.0.1:{port}/v1"}]
    col = C.Collector(svc, env={"TANDEM_COLLECT_TOKEN": TOKEN})
    col._pull_round()
    assert col.backends["x"].pull["state"] == "error"        # the failure is accounted
    assert col.backends["x"].pull["next"] > col.clock()      # ... and it backs off
    assert any("pull x failed" in s for s in svc.logs)       # ... and it is named, once
    assert col.view("v")["last"] == {"running": 2}           # the other backend got its round
    col.backends["x"].pull["next"] = 0.0                     # due again, still broken
    col._pull_round()
    assert sum("pull x failed" in s for s in svc.logs) == 1  # no spam while it stays error
    srv.shutdown()


if __name__ == "__main__":
    run(globals())
