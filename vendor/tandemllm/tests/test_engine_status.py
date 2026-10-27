"""OPS-65 (the UT1 self-test, 2026-10-06): a restarting engine is explained, not a blank page with a
JSON text about vLLM. ops/engine_status.py, CPU only, loopback, a few seconds.

  1  the status file: each state with its time; a start's duration joins the last five; the
     estimate is their median (300 s before any)                 test_the_file_and_the_estimate
  2  the responder, the engine stopped for a hold: GET /dashboard/ is a page (503, Retry-After)
     that names the state, the reason and the time left, and polls /health; an API request is the
     OpenAI error shape with Retry-After and the facts under `engine`; nothing says vLLM
                                                                 test_the_responder_explains_a_hold
  3  a file that says `up` while the engine does not answer reads "not answering"
                                                                 test_up_but_not_answering
  4  ops/stop.sh writes stopping then down; under a hold (QSE_STATUS_KEEP=1) it leaves `held`
                                                                 test_the_scripts_write_it

Run: python tests/test_engine_status.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ops"))

import engine_status as ES  # noqa: E402


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_the_file_and_the_estimate():
    d = tempfile.mkdtemp(prefix="qse-status-")
    p = os.path.join(d, "engine-status.json")
    t0 = 1_760_000_000.0
    assert ES.typical_start(ES.read(p)) == ES.DEFAULT_START_S
    ES.write("starting", reason="loading the model", p=p, now=t0)
    for i, took in enumerate((200, 260, 240, 1300, 230, 250)):
        ES.write("up", took_s=took, p=p, now=t0 + i)
    st = ES.read(p)
    assert st["state"] == "up" and len(st["starts"]) == ES.KEEP_STARTS
    assert ES.typical_start(st) == 250.0, st["starts"]          # 260 240 1300 230 250
    ES.write("held", reason="deploy 0.3.41-test", expected_s=600, p=p, now=t0 + 100)
    f = ES.facts(ES.read(p), t0 + 160)
    assert f["state"] == "held" and f["words"] == "stopped for maintenance"
    assert f["left_s"] == 600 + 250 - 60 and f["reason"] == "deploy 0.3.41-test"
    assert "vLLM" not in ES.message(f) and "back in about 13 min" in ES.message(f), ES.message(f)
    # OPS-99: the profile the scripts start (serve.env's QSE_WEIGHTS) is in the file, the facts and the page
    old = os.environ.get("QSE_WEIGHTS")
    os.environ["QSE_WEIGHTS"] = "flashnext-3.05"
    try:
        ES.write("starting", reason="loading the model", p=p, now=t0 + 200)
    finally:
        if old is None:
            os.environ.pop("QSE_WEIGHTS", None)
        else:
            os.environ["QSE_WEIGHTS"] = old
    f = ES.facts(ES.read(p), t0 + 210)
    assert f["profile"] == "flashnext-3.05" and 'id="profile">flashnext-3.05<' in ES.page(f), f
    return "5 starts kept, median 250 s; a hold's time left = its budget + a start; the profile named"


def _get(url, accept=None):
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode()


def _serve(status_file, engine_port):
    port = _port()
    env = dict(os.environ, QSE_STATUS_FILE=status_file)
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "ops", "engine_status.py"), "serve",
                             "--port", str(port), "--engine-port", str(engine_port)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return port, proc
        except OSError:
            time.sleep(0.05)
    proc.kill()
    raise AssertionError("the responder did not come up")


def test_the_responder_explains_a_hold():
    d = tempfile.mkdtemp(prefix="qse-status-")
    p = os.path.join(d, "engine-status.json")
    ES.write("held", reason="deploy 0.3.41-test", expected_s=1200, p=p)
    port, proc = _serve(p, _port())
    try:
        st, h, body = _get(f"http://127.0.0.1:{port}/dashboard/", "text/html")
        assert st == 503 and h["Content-Type"].startswith("text/html") and int(h["Retry-After"]) >= 5
        assert "TandemLLM is stopped for maintenance" in body and "deploy 0.3.41-test" in body
        assert "fetch('/health'" in body and "vLLM" not in body
        assert 'name="viewport"' in body, "fits a phone"
        st, h, body = _get(f"http://127.0.0.1:{port}/v1/chat/completions")
        j = json.loads(body)
        assert st == 503 and h["Content-Type"] == "application/json" and int(h["Retry-After"]) >= 5
        assert j["error"]["type"] == "provider_unavailable" and j["error"]["code"] == "engine_held"
        assert "deploy 0.3.41-test" in j["error"]["message"] and "vLLM" not in body
        assert j["engine"]["left_s"] > 1200, j["engine"]
        st, _, body = _get(f"http://127.0.0.1:{port}/health")
        assert st == 503 and json.loads(body)["status"] == "held"
        ES.write("starting", reason="restarting after a hold", p=p)
        st, _, body = _get(f"http://127.0.0.1:{port}/health")
        assert json.loads(body)["engine"]["words"] == "starting"
    finally:
        proc.kill()
    return "page and JSON both name the hold, its reason and its time; Retry-After; no vLLM"


def test_up_but_not_answering():
    d = tempfile.mkdtemp(prefix="qse-status-")
    p = os.path.join(d, "engine-status.json")
    ES.write("up", took_s=240, p=p)
    port, proc = _serve(p, _port())              # nothing listens on the engine's port
    try:
        st, _, body = _get(f"http://127.0.0.1:{port}/v1/completions")
        e = json.loads(body)["engine"]
        assert st == 503 and e["state"] == "not_answering" and e["words"] == "not answering", e
        assert e["left_s"] == 240
        # OPS-67: the model list stays (Open WebUI keeps TandemLLM), the state rides along
        st, _, body = _get(f"http://127.0.0.1:{port}/v1/models")
        m = json.loads(body)
        assert st == 200 and m["data"][0]["id"] == "TandemLLM", m
        assert m["tandem_engine"]["state"] == "not_answering"
    finally:
        proc.kill()
    return "a file that says up while nothing answers: not answering, a start's time"


def test_the_scripts_write_it():
    repo = tempfile.mkdtemp(prefix="qse-status-repo-")
    os.makedirs(os.path.join(repo, "ops"))
    for f in ("stop.sh", "engines.sh", "engine_status.py"):
        shutil.copy(os.path.join(ROOT, "ops", f), os.path.join(repo, "ops", f))
    with open(os.path.join(repo, "ops", "serve.env"), "w") as fh:
        fh.write(f"REPO={repo}\nPORT=18997\n")
    proc_dir = tempfile.mkdtemp(prefix="qse-proc-")      # an empty process table: nothing to stop
    env = dict(os.environ, ENGINE_PROC=proc_dir, QSE_STATUS_FILE=os.path.join(repo, "st.json"))
    out = subprocess.run(["bash", os.path.join(repo, "ops", "stop.sh"), "1"], env=env,
                         capture_output=True, text=True, timeout=30).stdout
    assert "nothing on :18997" in out and not os.path.exists(env["QSE_STATUS_FILE"])
    # a fake engine process in the table: stop.sh signals it (it is gone at once) and writes
    eng = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    os.makedirs(os.path.join(proc_dir, str(eng.pid)))
    with open(os.path.join(proc_dir, str(eng.pid), "cmdline"), "wb") as fh:
        fh.write(b"python\0-u\0server/app.py\0--port\00018997\0")
    os.symlink(sys.executable, os.path.join(proc_dir, str(eng.pid), "exe"))
    try:
        out = subprocess.run(["bash", os.path.join(repo, "ops", "stop.sh"), "5"], env=env,
                             capture_output=True, text=True, timeout=60)
        st = ES.read(env["QSE_STATUS_FILE"])
        assert st["state"] == "down", (st, out.stdout, out.stderr)
        ES.write("held", reason="a hold", p=env["QSE_STATUS_FILE"])
        subprocess.run(["bash", os.path.join(repo, "ops", "stop.sh"), "1"],
                       env=dict(env, QSE_STATUS_KEEP="1"), capture_output=True, timeout=60)
        assert ES.read(env["QSE_STATUS_FILE"])["state"] == "held"
    finally:
        eng.kill()
    return "stop.sh: stopping -> down; under a hold it keeps held"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:45s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
