"""SRV-131 / SRV-132: typed decisions on the engine port, over HTTP.

The real server with the fake engine (server/fake_engine.py) and the fake decision model
"other-decider" (engine/decision/fake.py, limits 256 and 96), the conductor on a temp directory.
The test matrix is SRV-131's Gherkin (SRV-132's table); rows 7, 8 and 10 (the conductor's
in-process client) are tests/test_decider_conductor.py.

  1  the Jev contract on :8000        test_1_* (and /v1/predict, /v1/models/{id}, the contract)
  2  behind the engine's bearer auth  test_2_* (proxied 401, token 200, local 200; no admin
                                      password set: open, AU1)
  3  invalid requests                 test_3_* (each 422; the model saw no question)
  4  not ready yet                    test_4_* (503 loading + Retry-After, /readyz 503, then ready)
  5  one chat model stays one         test_5_*
  6  off is the release               test_6_* (a second server with the flag empty)
  9  a busy engine                    test_9_* (a completion holds the engine: 503 busy_decoding
                                      after the caller's 200 ms; in_decode refused)

plus the Deciders screen's routes (status, load, unload, autoload, the test box, a refusal).

Run: python tests/test_decider_api.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_conductor_e2e import ADMIN, LAUNCH, PROXY, _call, _port, _wait  # noqa: E402
from tools import contract_check  # noqa: E402

CONTRACT = os.path.join(ROOT, "docs", "contract", "conductor-v1")
CHOICE = {"cls": {"type": "choice", "instructions": "Which kind of work is this?",
                  "criteria": {"debug": "fix a bug", "plan": "plan the work", "chat": "talk"}}}
MIXED = {**CHOICE, "urgent": {"type": "noul", "instructions": "The user is in a hurry."},
         "level": {"type": "score", "instructions": "How hard is it?",
                   "criteria": ["easy", "medium", "hard"]}}
LOCK_DIR = tempfile.mkdtemp(prefix="dec-api-lock-")


def _env(models: str):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ["HOME"],
            "PYTHONPATH": ROOT, "CUDA_VISIBLE_DEVICES": "", "QSE_ADMIN_TOKEN": ADMIN,
            "QWEN38_DECIDER_MODELS": models, "FAKE_MEM_AVAILABLE_GIB": "64",
            "BOX_LOCK": os.path.join(LOCK_DIR, "no-box.flock")}


def _server(port, models: str, tps: str = "2000"):
    cdir = tempfile.mkdtemp(prefix="dec-cond-")
    ledger = os.path.join(tempfile.mkdtemp(prefix="dec-led-"), "ledger.sqlite3")
    args = [sys.executable, "-c", LAUNCH, "--fake-engine", "--port", str(port),
            "--served-model", "qwen38-spark-engine", "--usage-ledger", ledger,
            "--fake-tps", tps, "--conductor-config", cdir]
    proc = subprocess.Popen(args, cwd=ROOT, env=_env(models), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return proc


PORT, OFF = _port(), _port()
SRV = _server(PORT, "other-decider", tps="60")
SRV_OFF = _server(OFF, "")


def _ok(obj, name):
    errs = contract_check.validate(obj, contract_check.load(name, CONTRACT), root=CONTRACT)
    assert not errs, f"{name}: {errs[:5]}"


def _ask(body, headers=None, token=ADMIN, port=None, path="/v1/systemone"):
    return _call(port or PORT, "POST", path, body, headers=headers, token=token)


def _status(port=None):
    st, _, out = _call(port or PORT, "GET", "/v1/conductor/deciders/status")
    assert st == 200, out
    return out


def _until_loaded(name="other-decider", timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if any(m["name"] == name and m["state"] == "loaded" for m in _status()["models"]):
            return
        time.sleep(0.1)
    raise AssertionError(f"{name} did not load: {_status()}")


# ------------------------------------------------------------------ 4 first: nothing loaded yet
def test_1_4_not_ready_then_the_contract():
    """Row 4 then row 1: the model is not loaded at start (other-decider is not in the seed
    document's on_start list), so the first question starts its load and answers 503 loading."""
    st, _, out = _call(PORT, "GET", "/readyz")
    assert st == 503 and out["status"] == "not_loaded" and out["models"] == [], out
    st, hdr, out = _ask({"state": "It crashes. debug please", "questions": CHOICE,
                         "model": "other-decider"})
    assert st == 503 and out["error"]["type"] == "loading" and hdr.get("Retry-After"), (st, out)
    assert out["type"] == "error"
    _until_loaded()
    st, _, out = _call(PORT, "GET", "/readyz")
    assert st == 200 and out["status"] == "ready" and out["models"] == ["other-decider"], out
    # row 1: the Jev contract
    st, _, out = _ask({"state": "It crashes on start. debug please FAKE_TRUE", "questions": MIXED})
    assert st == 200, out
    _ok(out, "systemone")
    assert out["model"] == "other-decider" and out["path"] == "eager"
    assert out["answers"]["cls"]["choice"] == "debug" and out["answers"]["urgent"]["noul"] == 0.9
    assert out["answers"]["level"]["type"] == "score" and set(out["answers"]) == set(MIXED)
    assert out["usage"]["output_tokens"] == 0 and out["usage"]["input_tokens"] > 0
    assert out["routing"]["model"] == "other-decider" and out["latency_ms"] > 0
    st, _, out2 = _ask({"state": "plan it", "questions": CHOICE, "model": "jev-latest"},
                       path="/v1/predict")
    assert st == 200 and out2["answers"]["cls"]["choice"] == "plan", out2
    st, _, card = _call(PORT, "GET", "/v1/models/other-decider")
    assert st == 200 and card["object"] == "decision_model" and card["loaded"], card
    assert card["limits"] == {"context_tokens": 256, "question_tokens": 96, "max_options": 20}


def test_2_behind_the_engines_bearer_auth():
    _until_loaded()
    body = {"state": "debug", "questions": CHOICE}
    st, hdr, out = _ask(body, headers=PROXY, token=None)
    assert st == 401 and "Bearer" in hdr.get("WWW-Authenticate", ""), (st, out)
    assert _ask(body, headers=PROXY)[0] == 200, "the admin token through the proxy"
    assert _ask(body, token=None)[0] == 200, "a tool on the box itself (trusted local)"
    from server.auth import Auth
    assert Auth(None, login=False).decide("decider", "203.0.113.44", PROXY) == "ok", \
        "AU1: no admin password set, the route is open"
    assert Auth(None, login=False).decide("decider", "127.0.0.1", {}) == "ok"


def test_3_invalid_requests_are_422_and_reach_no_model():
    _until_loaded()
    before = next(m for m in _status()["models"] if m["name"] == "other-decider")["questions"]
    too_many = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(65)}
    cases = [
        {"state": "x", "questions": {}},
        {"state": "x", "questions": {"q": {"type": "maybe", "instructions": "x"}}},
        {"state": "x", "questions": {"q": {"type": "choice", "instructions": "x",
                                           "criteria": {"only": None}}}},
        {"state": "x", "questions": {"q": {"type": "score", "instructions": "x",
                                           "criteria": [str(i) for i in range(11)]}}},
        {"state": "x", "questions": CHOICE, "model": "laya-giant"},
        {"state": "x", "questions": too_many},
        {"state": "x", "questions": {"q": {"type": "noul", "instructions": "  "}}},
        {"state": "x", "questions": {"q": {"type": "noul", "instructions": "x",
                                           "criteria": {"maybe": "y"}}}},
        {"state": 7, "questions": CHOICE},
    ]
    for body in cases:
        st, _, out = _ask(body)
        assert st == 422 and out["error"]["type"] == "invalid_request_error", (body, st, out)
    after = next(m for m in _status()["models"] if m["name"] == "other-decider")["questions"]
    assert after == before, "a refused request must not reach the model"


def test_5_one_chat_model_stays_one():
    st, _, out = _call(PORT, "GET", "/v1/models")
    assert st == 200 and [m["id"] for m in out["data"]] == ["TandemLLM"], out["data"]
    dm = {m["id"]: m for m in out["decision_models"]}
    assert set(dm) == {"other-decider", "jev-latest"}, dm
    assert dm["other-decider"]["limits"]["question_tokens"] == 96
    assert dm["jev-latest"]["limits"] == dm["other-decider"]["limits"]
    assert all(m["object"] == "decision_model" for m in dm.values())


def test_6_off_is_the_release():
    _wait(OFF, SRV_OFF)
    st, _, raw = _call(OFF, "GET", "/v1/models", raw=True)
    out = json.loads(raw)
    assert st == 200 and set(out) == {"object", "data"}, out
    assert raw == json.dumps({"object": "list", "data": out["data"]}, ensure_ascii=False).encode(), \
        "the release's body: the same dict, serialised the same way"
    assert [m["id"] for m in out["data"]] == ["TandemLLM"]
    for method, path in (("POST", "/v1/systemone"), ("POST", "/v1/predict"), ("GET", "/readyz")):
        st, _, body = _call(OFF, method, path, {"state": "x", "questions": CHOICE}
                            if method == "POST" else None)
        assert st == 404, (path, st, body)
    off = _status(OFF)
    assert off["enabled"] is False and "QWEN38_DECIDER_MODELS is empty" in off["why"]
    _ok(off, "deciders-status")
    st, _, out = _call(OFF, "POST", "/v1/conductor/deciders/models/other-decider/load", {})
    assert st == 404


def test_9_a_busy_engine_answers_after_the_callers_deadline():
    _until_loaded()
    done = threading.Event()

    def chat():
        body = {"model": "TandemLLM", "stream": False, "max_tokens": 240,
                "messages": [{"role": "user", "content": "write a long story"}]}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "X-Tandem": "raw"})
        try:
            urllib.request.urlopen(req, timeout=60).read()
        finally:
            done.set()

    threading.Thread(target=chat, daemon=True).start()
    end = time.time() + 10
    while time.time() < end:                        # until the completion holds the engine
        _, _, h = _call(PORT, "GET", "/health")
        if h.get("engine_busy"):
            break
        time.sleep(0.05)
    else:
        raise AssertionError("the completion never took the engine")
    t0 = time.time()
    st, hdr, out = _ask({"state": "debug", "questions": CHOICE},
                        headers={"X-Decider-Deadline-Ms": "200"})
    took = time.time() - t0
    assert st == 503 and out["error"]["type"] == "busy_decoding", (st, out)
    assert hdr.get("Retry-After") and 0.18 <= took < 2.0, (hdr, took)
    assert not done.is_set(), "the completion was still running: the question never ran beside it"
    st, _, out = _ask({"state": "debug", "questions": CHOICE},
                      headers={"X-Decider-Window": "in_decode"})
    assert st == 422 and "stage C" in out["error"]["message"], out
    done.wait(60)
    assert _ask({"state": "debug", "questions": CHOICE})[0] == 200, "idle again: answered"


def test_raw_answers_are_the_same_with_the_deciders_loaded():
    """The fake-engine form of the raw identity gate: one raw request (`X-Tandem: raw`) to the
    server with a decision model loaded and asked, and to the server without the runtime, streams
    the same text and the same usage. (The real gate, flagoff_identity and the PASS identity on
    the model, runs on the box under a hold.) Since SRV-617 the raw path gives a plain client
    the thinking in reasoning_content and the answer alone in content, so the budget reaches past
    the reasoning and both fields must hold text."""
    _until_loaded()
    assert _ask({"state": "debug", "questions": CHOICE})[0] == 200
    _wait(OFF, SRV_OFF)

    def raw(port):
        body = {"model": "TandemLLM", "stream": False, "max_tokens": 96, "temperature": 0,
                "messages": [{"role": "user", "content": "Name three prime numbers."}]}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "X-Tandem": "raw"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
            assert r.headers.get("X-Tandem") == "raw"
        m = out["choices"][0]["message"]
        return m.get("content"), m.get("reasoning_content"), out["usage"]

    on, off = raw(PORT), raw(OFF)
    assert on == off and on[0] and on[1] and "<think>" not in on[0], (on, off)


# ------------------------------------------------------------------ the Deciders screen's routes
def test_screen_routes_status_unload_load_autoload_test_box():
    _until_loaded()
    s = _status()
    _ok(s, "deciders-status")
    assert s["enabled"] and s["window"] == "idle" and s["ready"]
    m = s["models"][0]
    assert m["name"] == "other-decider" and m["state"] == "loaded" and m["stats"]["forwards"] >= 1
    assert s["planner"]["allowed"] and s["planner"]["claim_gib"] == 3.0, s["planner"]
    # L1 adds the judge's and the Laya route's question ids (both `auto` in the seed)
    assert {q["id"] for q in s["questions"]} == {"classify", "packages", "judge", "laya_api"}
    assert s["api"]["mode"] == "on" and s["api"]["stats"]["calls"] == 0, s["api"]
    assert set(s["unserved"]) == {"laya-english", "laya-multilingual", "laya-typed-decisions"}
    # writes need the token through the proxy
    st, _, _ = _call(PORT, "POST", "/v1/conductor/deciders/models/other-decider/unload", {},
                     headers=PROXY, token=None)
    assert st == 401
    st, _, out = _call(PORT, "POST", "/v1/conductor/deciders/models/other-decider/unload", {})
    assert st == 200 and out["state"] == "not_loaded", out
    _ok(out, "decider-action")
    st, _, out = _call(PORT, "POST", "/v1/conductor/deciders/models/other-decider/unload", {})
    assert st == 409
    st, _, out = _call(PORT, "POST", "/v1/conductor/deciders/models/other-decider/load", {})
    assert st == 202 and out["decision"]["allowed"] and out["state"] in ("loading", "warming",
                                                                         "loaded"), out
    _ok(out, "decider-action")
    _until_loaded()
    st, _, out = _call(PORT, "POST", "/v1/conductor/deciders/models/laya-giant/load", {})
    assert st == 404
    # the deciders document: on_start for other-decider, then the autoload start.sh asks for
    st, _, doc = _call(PORT, "GET", "/v1/conductor/deciders")
    assert st == 200 and doc["kind"] == "deciders" and doc["version"] == 0, doc
    d = doc["doc"]
    d["models"].append({"name": "other-decider", "load": "on_start"})
    d["questions"]["classify"] = "other-decider"
    st, _, w = _call(PORT, "PUT", "/v1/conductor/deciders", {"doc": d, "note": "the test model"})
    assert st == 200 and w["version"] == 1, w
    _call(PORT, "POST", "/v1/conductor/deciders/models/other-decider/unload", {})
    st, _, out = _call(PORT, "POST", "/v1/conductor/deciders/autoload",
                       {"service_start": True}, token=None)
    assert st == 202 and out["queued"] == ["other-decider"], out
    _until_loaded()
    assert next(q for q in _status()["questions"] if q["id"] == "classify")["model"] == \
        "other-decider"
    # the test box follows the dashboard's rule (no token with the login off) and answers the Jev
    st, _, out = _call(PORT, "POST", "/v1/conductor/deciders/test",
                       {"state": "please plan", "questions": CHOICE}, headers=PROXY, token=None)
    assert st == 200 and out["answers"]["cls"]["choice"] == "plan", out
    _ok(out, "systemone")
    st, _, out = _call(PORT, "POST", "/v1/conductor/deciders/test", {"state": "x", "questions": {}})
    assert st == 422 and out["error"]["type"] == "invalid_request_error", out


def test_a_refused_load_is_a_409_with_the_planners_numbers():
    """The route over a registry whose planner says no (MemAvailable 20 GiB): 409, the decision,
    nothing loaded."""
    from engine.decision.fake import FakeModel
    from engine.decision.registry import Registry
    from server.conductor import residency, routes, schema

    def plan(name, *, loading, service_start, hold_restart):
        return residency.check("decider", 20.0, schema.seed("residency"), schema.seed("roles"),
                               loading=loading)

    reg = Registry([FakeModel("other-decider")], gate=threading.Lock(), planner=plan,
                   log=lambda s: None)

    class Cond:
        state = {"deciders": reg}

    st, out, _ = routes._deciders(Cond, "POST", ["models", "other-decider", "load"], {})
    assert st == 409 and out["decision"]["reason"] == "refused_memory", out
    assert out["decision"]["after_gib"] == 17.0 and out["state"] == "not_loaded"
    assert not contract_check.validate(dict(out, contract_version="1.0"),
                                       contract_check.load("decider-action", CONTRACT),
                                       root=CONTRACT)


def _main():
    try:
        _wait(PORT, SRV)
        passed = 0
        order = ["test_1_4_not_ready_then_the_contract"] + sorted(
            n for n in globals() if n.startswith("test_") and n != "test_1_4_not_ready_then_the_contract")
        for name in order:
            globals()[name]()
            print(f"  {name:72s} ok")
            passed += 1
        print(f"{passed} passed")
    finally:
        for p in (SRV, SRV_OFF):
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    _main()
