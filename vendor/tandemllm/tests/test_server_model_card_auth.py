"""The engine's single model card answers only to a key, as its list route does.

server/app.py gated /v1/models on the users' keys (with `--backends`) while /v1/models/{id}
answered anyone, the 404-vs-200 difference naming the models. The proxy's twin is
tests/test_proxy_model_card_auth.py.

Run: python tests/test_server_model_card_auth.py   (CPU torch, as tests/test_fake_engine.py)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fake_engine import _port, _start  # noqa: E402
from test_proxy_engine import _req, _wait  # noqa: E402

MODEL = "qwen38-spark-engine"
FAR = {"X-Forwarded-For": "192.0.2.9", "Authorization": "Bearer nope"}


def test_card_needs_the_key_the_list_needs():
    d = tempfile.mkdtemp(prefix="srvcard-")
    port = _port()
    env = {"QSE_TEST": "1", "TANDEM_SECRETS_KEY": "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="}
    proc = _start(port, os.path.join(d, "ledger.sqlite3"), home=d,
                  extra=("--backends", os.path.join(d, "config.sqlite3")), env_extra=env)
    try:
        _wait(port, proc)
        # no users yet: nobody is refused
        assert _req(port, f"/v1/models/{MODEL}", headers=FAR)[0] == 200
        assert _req(port, "/v1/models/nope", headers=FAR)[0] == 404
        st, _, body = _req(port, "/v1/config/users", {"label": "t"})
        assert st in (200, 201), body
        key = json.loads(body)["key"]
        assert _req(port, "/v1/models", headers=FAR)[0] == 401, "the list was already gated"
        assert _req(port, f"/v1/models/{MODEL}", headers=FAR)[0] == 401
        assert _req(port, "/v1/models/nope", headers=FAR)[0] == 401, \
            "an unknown name answers as a known one does: no enumeration"
        with_key = dict(FAR, Authorization=f"Bearer {key}")
        st, _, body = _req(port, f"/v1/models/{MODEL}", headers=with_key)
        assert st == 200 and json.loads(body)["id"] == MODEL, body
        assert _req(port, "/v1/models/nope", headers=with_key)[0] == 404
    finally:
        proc.terminate()
        proc.wait(30)


def test_without_backends_unchanged():
    d = tempfile.mkdtemp(prefix="srvcard-")
    port = _port()
    proc = _start(port, os.path.join(d, "ledger.sqlite3"), home=d, env_extra={"QSE_TEST": "1"})
    try:
        _wait(port, proc)
        time.sleep(0.2)
        assert _req(port, f"/v1/models/{MODEL}", headers=FAR)[0] == 200
        assert _req(port, "/v1/models/nope", headers=FAR)[0] == 404
    finally:
        proc.terminate()
        proc.wait(30)


if __name__ == "__main__":
    test_card_needs_the_key_the_list_needs()
    test_without_backends_unchanged()
    print("2 passed")
