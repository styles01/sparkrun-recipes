"""Setup mode (OPS-93): `server/app.py --setup` and `ops/start.sh --setup`, on a CPU.

The test matrix:

  1  the setup-mode server's routes: /health says mode setup and Wren's state; the dashboard's
     files and session answer; /v1/setup/* answers; /v1/models is an empty list; a chat request or
     any other route is 503 setup_mode with Retry-After          test_the_setup_routes
  2  ops/start.sh --setup starts `server/app.py --setup` on 127.0.0.1 with QSE_SETUP_MODE and
     nothing of the engine's flags; --host names another address        test_start_setup
  3  while run/setup-pending exists a plain start is setup mode, on the host the marker names;
     without it the configured engine starts as before (no --setup)     test_the_marker
  2b setup mode reads secrets.env like the engine does: QSE_ADMIN_TOKEN reaches the server
                                                                  test_start_setup
  4  start.sh refuses an unknown argument, a host that is not an address, and --host with no
     address after it (it looped forever)                         test_bad_arguments

Run: python tests/test_setup_mode.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app, auth  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Req(app.SetupHandler):
    def __init__(self, method, path, body=None):
        raw = json.dumps(body if body is not None else {}).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json"}
        self.path, self.command, self.request_version = path, method, "HTTP/1.1"
        self.requestline, self.client_address = f"{method} {path}", ("127.0.0.1", 0)
        self.close_connection = True

    def go(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            getattr(self, f"do_{self.command}")()
        head, _, raw = self.wfile.getvalue().partition(b"\r\n\r\n")
        return int(head.decode().split(" ", 2)[1]), head.decode(), raw


def test_the_setup_routes():
    app.STATE.clear()
    app.STATE.update(auth=auth.Auth(None, None), started=time.time(), setup_mode=True,
                     version="0.6.2", dashboard_dir=None)
    code, _, raw = Req("GET", "/health").go()
    h = json.loads(raw)
    assert code == 200 and h["mode"] == "setup" and h["status"] == "ok" and h["version"] == "0.6.2"
    assert h["assistant"] == "off"
    code, _, raw = Req("GET", "/v1/setup/device").go()
    d = json.loads(raw)
    assert code == 200 and d["mode"] == "setup" and d["engine"]["profile"] is None, d
    code, _, raw = Req("GET", "/v1/setup/models").go()
    assert code == 200 and json.loads(raw)["models"][0]["id"] == "qwen3.8-27b"
    code, _, raw = Req("GET", "/v1/models").go()
    assert code == 200 and json.loads(raw) == {"object": "list", "data": []}
    code, _, raw = Req("GET", "/v1/dashboard/session").go()
    assert code == 200 and json.loads(raw)["login"] is False
    code, _, _ = Req("GET", "/dashboard/").go()
    assert code == 200
    for method, path in (("POST", "/v1/chat/completions"), ("POST", "/v1/completions"),
                         ("GET", "/v1/dashboard/summary"), ("GET", "/metrics"),
                         ("PUT", "/v1/conductor/settings"), ("DELETE", "/v1/config/x")):
        code, head, raw = Req(method, path, {"model": "TandemLLM", "messages": []}).go()
        assert code == 503 and json.loads(raw)["error"]["code"] == "setup_mode", (path, raw)
        assert "Retry-After: 60" in head
    code, _, raw = Req("POST", "/v1/setup/estimate", {"model": "qwen3.8-27b"}).go()
    assert code == 200 and json.loads(raw)["fits"] is True


class Box:
    """A checkout with ops/ (the real start.sh and engines.sh), a serve.env whose PY is a stub that
    records how it was started, and PATH shims for setsid, curl (healthy once the stub ran) and ss
    (no port taken: on a box with a live engine the real one sees its port)."""

    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="setup-mode-")
        self.root = os.path.join(self.tmp, "repo")
        ops = os.path.join(self.root, "ops")
        os.makedirs(ops)
        for f in ("start.sh", "engines.sh"):
            shutil.copy(os.path.join(REPO, "ops", f), ops)
        shutil.copytree(os.path.join(REPO, "ops", "profiles"), os.path.join(ops, "profiles"))
        bin_ = os.path.join(self.tmp, "bin")
        os.makedirs(bin_)
        self.py = os.path.join(bin_, "fakepy")
        self.args = os.path.join(self.root, "py.args")
        with open(self.py, "w") as fh:
            fh.write('#!/bin/bash\nprintf "%s|SETUP=%s|OPS=%s|ADMIN=%s\\n" "$*" '
                     '"${QSE_SETUP_MODE:-}" "${QSE_OPS_DIR:-}" "${QSE_ADMIN_TOKEN:+set}" >> "'
                     + self.args + '"\n'
                     'touch "' + os.path.join(self.root, "healthy") + '"\nsleep 1\n')
        with open(os.path.join(bin_, "setsid"), "w") as fh:
            fh.write('#!/bin/bash\nexec "$@"\n')
        with open(os.path.join(bin_, "curl"), "w") as fh:
            fh.write('#!/bin/bash\n[ -f "' + os.path.join(self.root, "healthy") + '" ]\n')
        with open(os.path.join(bin_, "ss"), "w") as fh:
            fh.write('#!/bin/bash\nexit 0\n')
        for f in ("fakepy", "setsid", "curl", "ss"):
            os.chmod(os.path.join(bin_, f), 0o755)
        with open(os.path.join(REPO, "ops", "serve.env")) as fh:
            env = fh.read()
        env = env.replace("REPO=${HOME}/qwen38-spark-engine", f"REPO={self.root}")
        env = env.replace("PY=${HOME}/recipes/ling3-flash-dgx-spark/.venv/bin/python",
                          f"PY={self.py}")
        with open(os.path.join(ops, "serve.env"), "w") as fh:
            fh.write(env)
        self.env = {"PATH": bin_ + os.pathsep + os.environ["PATH"], "HOME": self.tmp,
                    "ENGINE_PROC": os.path.join(self.tmp, "noproc"), "STATUS_PY": "true"}
        os.makedirs(self.env["ENGINE_PROC"])

    def start(self, *args):
        r = subprocess.run(["bash", os.path.join(self.root, "ops", "start.sh"), *args],
                           capture_output=True, text=True, timeout=60, env=self.env)
        time.sleep(1.2)                                  # the stub's own exit
        launched = open(self.args).read().splitlines() if os.path.exists(self.args) else []
        return r, launched

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.tmp, ignore_errors=True)


def test_start_setup():
    with Box() as box:
        r, launched = box.start("--setup")
        assert r.returncode == 0, r.stdout + r.stderr
        argv, setup, ops, admin = launched[0].split("|")
        assert argv == "-u server/app.py --setup --host 127.0.0.1 --port 8000", argv
        assert setup == "SETUP=1" and ops == f"OPS={os.path.join(box.root, 'ops')}"
        assert admin == "ADMIN=", admin
        assert "setup mode answers" in r.stdout and len(launched) == 1, r.stdout
    with Box() as box:
        # an admin password set: setup mode reads secrets.env too, so its routes are guarded
        sec = os.path.join(box.tmp, ".qwen38-spark-engine", "secrets.env")
        os.makedirs(os.path.dirname(sec))
        with open(sec, "w") as fh:
            fh.write("QSE_ADMIN_TOKEN=" + "a" * 40 + "\n")
        r, launched = box.start("--setup")
        assert r.returncode == 0 and launched[0].split("|")[3] == "ADMIN=set", launched
    with Box() as box:
        r, launched = box.start("--setup", "--host", "0.0.0.0")
        assert r.returncode == 0 and launched[0].startswith(
            "-u server/app.py --setup --host 0.0.0.0 --port 8000"), launched


def test_the_marker():
    with Box() as box:
        os.makedirs(os.path.join(box.root, "run"))
        with open(os.path.join(box.root, "run", "setup-pending"), "w") as fh:
            fh.write("HOST=0.0.0.0\n")
        r, launched = box.start()
        assert r.returncode == 0, r.stdout + r.stderr
        assert launched[0].startswith("-u server/app.py --setup --host 0.0.0.0 "), launched
    with Box() as box:
        r, launched = box.start()
        assert r.returncode == 0, r.stdout + r.stderr
        first = launched[0].split("|")[0]
        assert first.startswith("-u server/app.py --host 0.0.0.0 --port 8000") and \
            "--setup" not in first and "--conductor-config" in first, first
        assert launched[0].split("|")[1] == "SETUP="


def test_bad_arguments():
    with Box() as box:
        r, launched = box.start("--bogus")
        assert r.returncode == 2 and "unknown argument" in r.stdout and not launched
        r, launched = box.start("--setup", "--host", "a;reboot")
        assert r.returncode == 2 and not launched, r.stdout
        r, launched = box.start("--setup", "--host")
        assert r.returncode == 2 and "needs an address" in r.stdout and not launched, r.stdout


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
