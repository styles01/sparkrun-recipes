"""chatd's operations (OPS-103, phase 1 part): the forward key in secrets.env and in chatd's own
file, the same key on both sides, and `python -m chatd` starting with nothing beyond the standard
library, making its store and answering /health and the session route.

Run: python -m pytest -q tests/test_chatd_ops.py
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import time

from chatd_util import ROOT, run_all

from chatd.__main__ import load_key
from server import chat_forward


def test_make_secrets_writes_the_forward_key_and_adds_it_to_an_old_file():
    d = tempfile.mkdtemp()
    env = dict(os.environ, QSE_STATE_DIR=d)
    subprocess.run(["bash", os.path.join(ROOT, "ops", "make-secrets.sh")], env=env, check=True,
                   capture_output=True)
    text = open(os.path.join(d, "secrets.env")).read()
    key = [ln.split("=", 1)[1] for ln in text.splitlines() if ln.startswith("QSE_CHAT_FORWARD_KEY=")]
    assert len(key) == 1 and len(key[0]) == 64
    assert stat.S_IMODE(os.stat(os.path.join(d, "secrets.env")).st_mode) == 0o600
    # a file from before OPS-103 gets the line, once, and keeps what it had
    with open(os.path.join(d, "secrets.env"), "w") as fh:
        fh.write("QSE_METRICS_TOKEN=" + "m" * 64 + "\n")
    for _ in range(2):
        subprocess.run(["bash", os.path.join(ROOT, "ops", "make-secrets.sh")], env=env,
                       check=True, capture_output=True)
    text = open(os.path.join(d, "secrets.env")).read()
    assert text.startswith("QSE_METRICS_TOKEN=" + "m" * 64)
    assert text.count("QSE_CHAT_FORWARD_KEY=") == 1


def test_both_sides_read_the_same_key():
    d = tempfile.mkdtemp()
    saved = os.environ.pop("QSE_CHAT_FORWARD_KEY", None)
    try:
        k = load_key(d)                                  # chatd's first start makes it
        assert len(k) == 64
        assert stat.S_IMODE(os.stat(os.path.join(d, "forward.key")).st_mode) == 0o600
        assert load_key(d) == k                          # and keeps it
        assert chat_forward.forward_key({"QSE_CHAT_DATA": d}) == k
        assert chat_forward.forward_key({"QSE_CHAT_DATA": d, "QSE_CHAT_FORWARD_KEY": "e" * 40}) \
            == "e" * 40
        os.environ["QSE_CHAT_FORWARD_KEY"] = "s" * 40
        assert load_key(d) == "s" * 40                   # secrets.env wins over the file
    finally:
        os.environ.pop("QSE_CHAT_FORWARD_KEY", None)
        if saved is not None:
            os.environ["QSE_CHAT_FORWARD_KEY"] = saved


def test_python_m_chatd_starts_and_stops_cleanly():
    d = tempfile.mkdtemp()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("QSE_CHAT_FORWARD_KEY", None)
    p = subprocess.Popen([sys.executable, "-m", "chatd", "--port", str(port), "--data", d,
                          "--engine", "http://127.0.0.1:9", "--no-titles"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(100):
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                c.request("GET", "/health")
                r = c.getresponse()
                assert json.loads(r.read())["service"] == "chatd"
                break
            except OSError:
                time.sleep(0.1)
        key = open(os.path.join(d, "forward.key")).read().strip()
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("GET", "/v1/chat-app/session", headers={"X-Tandem-Forward": key})
        r = c.getresponse()
        assert r.status == 200 and json.loads(r.read())["password_set"] is False
        assert os.path.exists(os.path.join(d, "chat.sqlite3"))
    finally:
        p.terminate()
        out, _ = p.communicate(timeout=20)
    assert p.returncode == 0, out
    assert "[chatd] stopping" in out


if __name__ == "__main__":
    run_all(globals())
