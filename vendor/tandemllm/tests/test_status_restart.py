"""The status responder belongs to its tree and carries a Setup Guide restart (OPS-173 and OPS-175;
QA OPS-174 and OPS-176). ops/engine_status.py, CPU only, loopback, free ports, temporary trees.

On 2026-10-17 a Setup Guide install restarted the engine and the page, through the reverse proxy,
reached a responder of another tree (~/tandem-lfm25, started the day before): it read that tree's
status file of 10-15 ("switching to Nemotron 3.5 Bolt") and answered the job's route with 404 ("This
install and its log are no longer on the box"). `ensure` only asked whether something answered.

The test matrix, the Gherkin of OPS-173 and OPS-175:

  1  a responder of another tree is replaced, one of this tree kept; an older one without the
     identity route is found by its socket and replaced; another program on the port is left alone
                                                     test_ensure_replaces_only_a_foreign_responder
  2  an old reason ages out; a start that should long be over reads "not answering"
                                                     test_an_old_reason_ages_out
  3  the job through the responder: JSON, and SSE with the log, the snapshot (no log line's id),
     each new event and done; CORS on the read-only routes
                                                     test_the_job_through_the_responder
  4  the load's progress from the engine log (the 27B's, NemotronH's and the fake engine's lines)
     and /v1/setup/status's step n of N               test_the_load_progress, test_the_setup_status
  5  the server answers /v1/setup/status in the same shape while it runs
                                                     test_the_server_answers_the_same_shape
  6  an installed box has a responder: install.sh writes QSE_STATUS_PORT = the API port + 9; the
     dashboard's CSP lets the page ask it                  test_an_install_sets_the_status_port,
                                                           test_the_page_may_ask_the_responder

Run: python tests/test_status_restart.py   (1's socket lookup and 6 need Linux)
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ops"))
sys.path.insert(0, ROOT)

import engine_status as ES  # noqa: E402

SCRIPT = os.path.join(ROOT, "ops", "engine_status.py")
LINUX = platform.system() == "Linux"


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, accept: str | None = None, timeout: float = 5):
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode()


def _get_m(url: str, method: str):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=5) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode()


def _up(port: int, wait: float = 5.0) -> bool:
    t = time.time() + wait
    while time.time() < t:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


def _tree(name: str) -> str:
    """A tree of its own: VERSION, logs/, run/setup-jobs/."""
    t = os.path.realpath(tempfile.mkdtemp(prefix=f"status-{name}-"))
    os.makedirs(os.path.join(t, "logs"))
    os.makedirs(os.path.join(t, "run", "setup-jobs"))
    with open(os.path.join(t, "VERSION"), "w") as fh:
        fh.write("0.11.3\n")
    return t


def _ensure(t: str, port: int, script: str = SCRIPT) -> str:
    env = dict(os.environ, QSE_STATUS_TREE=t, QSE_STATUS_FILE=os.path.join(t, "logs", "engine-status.json"))
    env.pop("REPO", None)
    r = subprocess.run([sys.executable, script, "ensure", "--port", str(port), "--engine-port", "1",
                        "--host", "127.0.0.1"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    _up(port)
    return r.stdout.strip()


def _stop_ours(t: str, port: int) -> None:
    env = dict(os.environ, QSE_STATUS_TREE=t)
    subprocess.run([sys.executable, SCRIPT, "stop", "--port", str(port)], env=env, capture_output=True,
                   timeout=30)


def test_ensure_replaces_only_a_foreign_responder():
    a, b = _tree("a"), _tree("b")
    port = _port()
    try:
        assert "started" in _ensure(a, port)
        _, _, body = _get(f"http://127.0.0.1:{port}/_tandem/status")
        ida = json.loads(body)
        assert ida["tree"] == a and ida["version"] == "0.11.3" and ida["host"] == "127.0.0.1", ida
        assert ida["status_file"] == os.path.join(a, "logs", "engine-status.json")
        assert "running" in _ensure(a, port)                   # the same tree: kept
        if not LINUX:
            print("    (the replacement checks the process's command line in /proc: run it on the box)")
            return
        out = _ensure(b, port)                                 # another tree: replaced
        assert f"replaced (it served {a})" in out, out
        _, _, body = _get(f"http://127.0.0.1:{port}/_tandem/status")
        assert json.loads(body)["tree"] == b
        with open(os.path.join(b, "VERSION"), "w") as fh:     # a new release of the same tree
            fh.write("0.11.4\n")
        assert f"replaced (it served {b})" in _ensure(b, port)
    finally:
        _stop_ours(b, port)
        _stop_ours(a, port)
    if not LINUX:
        return
    # an older responder: no identity route; found by its listening socket and its command line
    old = os.path.join(tempfile.mkdtemp(prefix="status-old-"), "engine_status.py")
    text = open(SCRIPT).read()
    assert text.count('path == "/_tandem/status"') == 1
    with open(old, "w") as fh:
        fh.write(text.replace('path == "/_tandem/status"', 'path == "/_not_in_this_release"'))
    port = _port()
    p = subprocess.Popen([sys.executable, old, "serve", "--port", str(port)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        assert _up(port)
        assert ES.identity(port) is None and ES.listener_pid(port) == p.pid
        out = _ensure(a, port)
        assert "replaced (it served an older release)" in out, out
        p.wait(timeout=10)
        assert json.loads(_get(f"http://127.0.0.1:{port}/_tandem/status")[2])["tree"] == a
    finally:
        p.kill()
        _stop_ours(a, port)
    # another program on the port: reported, never stopped
    port = _port()
    other = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=a)
    try:
        assert _up(port)
        out = _ensure(a, port)
        assert "held by another program" in out, out
        time.sleep(0.3)
        assert other.poll() is None, "the other program was stopped"
    finally:
        other.kill()


def test_an_old_reason_ages_out():
    p = os.path.join(_tree("age"), "logs", "engine-status.json")
    t0 = 1_760_000_000.0
    ES.write("down", reason="switching to Nemotron 3.5 Bolt", p=p, now=t0)
    assert ES.facts(ES.read(p), t0 + 60)["reason"] == "switching to Nemotron 3.5 Bolt"
    f = ES.facts(ES.read(p), t0 + 2 * 86400)
    assert f["reason"] is None and f["state"] == "down" and "Bolt" not in ES.message(f), f
    # a start that should be over long ago: a silence, not a promise
    ES.write("starting", reason="loading the model", p=p, now=t0)
    assert ES.facts(ES.read(p), t0 + 100)["state"] == "starting"
    f = ES.facts(ES.read(p), t0 + 2 * ES.DEFAULT_START_S + 601)
    assert f["state"] == "not_answering" and f["reason"] is None, f
    # a hold keeps its reason for as long as it says it runs
    ES.write("held", reason="deploy", expected_s=86400, p=p, now=t0)
    assert ES.facts(ES.read(p), t0 + 7 * 3600)["reason"] == "deploy"


def _job(t: str, jid: str = "0123456789abcdef", steps=None, state: str = "running") -> str:
    d = os.path.join(t, "run", "setup-jobs", jid)
    os.makedirs(d, exist_ok=True)
    ids = ("check", "unload_assistant", "fetch", "preflight", "write_config", "switch", "smoke", "rollback")
    st = dict(zip(ids, steps or ("done", "done", "skipped", "done", "done", "running", "pending", "pending")))
    job = {"id": jid, "kind": "apply", "state": state, "created_at": time.time() - 60, "ended_at": None,
           "profile": "exl3-6.0", "model": "qwen3.8-27b", "message": None, "changes": [],
           "summary": {"model": "Qwen3.8-27B", "profile": "exl3-6.0", "profile_label": "EXL3 6.0",
                       "tools": [], "slots": 1, "window": 262144, "side_window": None},
           "steps": [{"id": i, "label": {"switch": "Restart the engine with the new setup"}.get(i, i),
                      "state": st[i], "started_at": None, "ended_at": None, "detail": None} for i in ids]}
    with open(os.path.join(d, "state.json"), "w") as fh:
        json.dump(job, fh)
    return d


def _event(d: str, seq: int, ev: dict) -> None:
    with open(os.path.join(d, "events.jsonl"), "a") as fh:
        fh.write(json.dumps({"seq": seq, "event": ev}) + "\n")


def _serve(t: str):
    port = _port()
    env = dict(os.environ, QSE_STATUS_TREE=t, QSE_STATUS_FILE=os.path.join(t, "logs", "engine-status.json"))
    p = subprocess.Popen([sys.executable, SCRIPT, "serve", "--port", str(port), "--engine-port", "1"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert _up(port)
    return port, p


def test_the_job_through_the_responder():
    t = _tree("job")
    d = _job(t)
    _event(d, 1, {"type": "job", "job": {"id": "0123456789abcdef"}})
    _event(d, 2, {"type": "step", "step": {"id": "switch", "state": "running"}})
    _event(d, 3, {"type": "log", "line": "[stop] stopped", "ts": 1})
    ES.write("starting", reason="loading the model", p=os.path.join(t, "logs", "engine-status.json"))
    port, p = _serve(t)
    try:
        st, h, body = _get(f"http://127.0.0.1:{port}/v1/setup/jobs/0123456789abcdef")
        assert st == 200 and json.loads(body)["state"] == "running" and h["Access-Control-Allow-Origin"] == "*"
        st, _, _ = _get(f"http://127.0.0.1:{port}/v1/setup/jobs/fedcba9876543210")
        assert st == 404
        # SSE: the log, the snapshot without the log line's id, then what the runner writes next
        frames: list[str] = []
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/setup/jobs/0123456789abcdef",
                                     headers={"Accept": "text/event-stream"})

        def read():
            with urllib.request.urlopen(req, timeout=20) as r:
                buf = ""
                for raw in r:
                    buf += raw.decode()
                    if buf.endswith("\n\n"):
                        frames.append(buf.strip())
                        buf = ""
        th = threading.Thread(target=read)
        th.start()
        time.sleep(1.0)
        _event(d, 4, {"type": "log", "line": "[start] healthy after 200s", "ts": 2})
        _event(d, 5, {"type": "step", "step": {"id": "switch", "state": "done"}})
        _event(d, 6, {"type": "done", "state": "done", "message": "exl3-6.0 is live on port 8000."})
        th.join(15)
        assert not th.is_alive(), "the stream did not end at done"
        assert frames[0].startswith("id: 3\n") and "[stop] stopped" in frames[0], frames
        assert frames[1].startswith("data: ") and '"type": "job"' in frames[1], frames   # no id: VIS-211
        assert [f.split("\n")[0] for f in frames[2:]] == ["id: 4", "id: 5", "id: 6"], frames
        assert '"type": "done"' in frames[-1]
        # Last-Event-ID resumes after it
        st, _, body = _get(f"http://127.0.0.1:{port}/v1/setup/jobs/0123456789abcdef?after=5", "text/event-stream")
        heads = [f.split("\n")[0] for f in body.strip().split("\n\n")]
        assert len(heads) == 2 and heads[0].startswith('data: {"type": "job"') and heads[1] == "id: 6", heads
        # a browser's preflight on a read-only route: allowed from any origin; on a chat route: not
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/setup/status", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 204 and r.headers["Access-Control-Allow-Origin"] == "*"
        st, h, _ = _get_m(f"http://127.0.0.1:{port}/v1/chat/completions", "OPTIONS")
        assert st == 503 and "Access-Control-Allow-Origin" not in h
        # the identity and /health answer any origin; a chat request does not
        assert _get(f"http://127.0.0.1:{port}/health")[1].get("Access-Control-Allow-Origin") == "*"
        assert "Access-Control-Allow-Origin" not in _get(f"http://127.0.0.1:{port}/v1/chat/completions")[1]
    finally:
        p.kill()


FIXTURE_27B = """[exl3] /home/u/exl3-quants/6.0 replaces the projections and the head
[exl3] 401 tensors from /home/u/exl3-quants/6.0: 19.53 GB (K6: 400, K8: 1)
[server] 400 quantised projections (17.00 GiB), 451 plain tensors (3.60 GiB), 20.61 GiB resident
[server] drafter=lenrouter depth=31 k=31 nvfp4=off fp8_head=off  loaded in 110.6s
[server] warm-up prefill of 2600 tokens in 3.6s
[server] verify graphs: 121 captured in 15.8s
[server] listening on http://0.0.0.0:8000  model TandemLLM
""".splitlines()
FIXTURE_NH = """[nh-load] experts through layer 17: 11.5 s
[nh-load] experts through layer 52: 34.7 s
[nh-load] /home/u/models/nemotron-3.5-super-vl-120b-a12b-exl3-4.0: {'index_s': 9.24}
[vision] tower 1.68 GB, 32 blocks, 392 tensors; processor Preprocessor (native backend, pinned); loaded in 11.1s
[server] NemotronH /home/u/models/nemotron (exl3): load 71.0 s, drafter LiLiCorr + lookup + StairCut
[server] warm-up prefill of 2600 tokens in 4.8s
[server] verify graphs: 1 captured in 0.0s
[server] listening on http://0.0.0.0:8000  model TandemLLM
""".splitlines()


def _progress(lines: list[str]) -> dict:
    t = _tree("load")
    with open(os.path.join(t, "logs", "engine.log"), "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return ES.load_progress(t)


def test_the_load_progress():
    for fixture, cut, current, detail in (
            (FIXTURE_27B, 2, "weights", "401 tensors from /home/u/exl3-quants/6.0: 19.53 GB (K6: 400, K8: 1)"),
            (FIXTURE_27B, 3, "drafters", None), (FIXTURE_27B, 4, "warmup", None),
            (FIXTURE_27B, 5, "graphs", None), (FIXTURE_27B, 6, "ready", None), (FIXTURE_27B, 7, None, None),
            (FIXTURE_NH, 2, "weights", "experts through layer 52"), (FIXTURE_NH, 4, "drafters", None),
            (FIXTURE_NH, 5, "warmup", None), (FIXTURE_NH, 8, None, None)):
        g = _progress(fixture[:cut])
        assert g["current"] == current, (cut, g)
        if detail:
            assert g["weights_detail"] == detail, g
    # the fake engine's load writes the same phases
    from server import fake_engine
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fake_engine.fake_load(10, sleep=lambda s: None)
    lines = buf.getvalue().splitlines()
    assert _progress(lines[:2])["current"] == "weights" and _progress(lines[:2])["weights_detail"] == \
        "16.0 GB of 20.0 GB of weights read"
    assert _progress(lines)["current"] == "ready"
    assert _progress(lines + ["[server] listening on http://127.0.0.1:1  model x"])["current"] is None
    assert ES.load_progress(_tree("nolog")) is None


def test_the_setup_status():
    t = _tree("status")
    _job(t)
    with open(os.path.join(t, "logs", "engine.log"), "w") as fh:
        fh.write("\n".join(FIXTURE_27B[:4]) + "\n")
    s = ES.setup_status(t, engine={"state": "starting", "since": int(time.time()) - 30})
    j = s["job"]
    assert j["id"] == "0123456789abcdef" and j["model"] == "qwen3.8-27b"
    assert j["step"]["index"] == 6 and j["step"]["total"] == 7 and \
        j["step"]["label"] == "Restart the engine with the new setup", j["step"]       # rollback not shown
    assert s["load"]["current"] == "warmup", s["load"]
    # an engine that is up: no load; a tree with no job: none
    assert ES.setup_status(t, engine={"state": "up"})["load"] is None
    assert ES.setup_status(_tree("empty"), engine={"state": "up"})["job"] is None
    # a job that ended an hour ago is not the restart's
    old = _tree("old")
    _job(old, state="done")
    assert ES.active_job(old) is None


def test_the_server_answers_the_same_shape():
    from server import setup_api
    from server.setup import apply
    t = _tree("server")
    _job(t)

    class App:
        STATE = {"started": time.time() - 5, "version": "0.11.3", "setup_mode": False}
    real = apply.ROOT
    apply.ROOT = t
    try:
        s = setup_api.setup_status(App)
    finally:
        apply.ROOT = real
    r = ES.setup_status(t, engine={"state": "starting"})
    assert set(s) - {"contract_version"} == set(r), (sorted(s), sorted(r))
    assert s["engine"]["state"] == "up" and s["load"] is None and s["job"]["id"] == "0123456789abcdef"


def test_the_page_may_ask_the_responder():
    """The dashboard's CSP lets the page ask the status responder's port (VIS-209), only with one."""
    from server import static
    old = os.environ.pop("QSE_STATUS_PORT", None)
    try:
        assert "connect-src 'self';" in static.csp_for(b"<html></html>")
        os.environ["QSE_STATUS_PORT"] = "8009"
        assert "connect-src 'self' http://*:8009;" in static.csp_for(b"<html></html>")
        os.environ["QSE_STATUS_PORT"] = "8009; script-src *"
        assert "script-src *" not in static.csp_for(b"<html></html>")
    finally:
        os.environ.pop("QSE_STATUS_PORT", None)
        if old is not None:
            os.environ["QSE_STATUS_PORT"] = old


def test_an_install_sets_the_status_port():
    text = open(os.path.join(ROOT, "install.sh")).read()
    assert '[ "$PORT" -le 65526 ] && setv QSE_STATUS_PORT "$((PORT + 9))"' in text
    if not LINUX:
        print("    (the install itself runs on Linux only -- run it on the box)")
        return
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_install_control as I
    port = I.free_port()
    box = I.install("status-port", True, port)
    env = open(os.path.join(box["dir"], "run", "ops", "serve.env")).read()
    assert f"\nQSE_STATUS_PORT={port + 9}\n" in env, env[-2000:]


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"  {name:55s} ok ({time.time() - t0:.1f} s)")
            passed += 1
    print(f"{passed} passed")
