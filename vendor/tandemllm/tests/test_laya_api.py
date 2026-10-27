"""L1 (2026-10-03): Laya for the model's own scripts, `POST /v1/laya/ask` and `GET /v1/laya`.

The real server with the fake engine and the fake decision model "other-decider" (limits 256 and
96, so Laya's window here is (256 - 96) x 3.5 = 560 characters), the conductor on a temp
directory. The fake model is steered by words in the state (engine/decision/fake.py): a label
that appears as a word is picked at 0.91, else the first at 0.2; FAKE_TRUE makes a yes-no 0.9.

The test matrix (the requirement: the model reaches Laya through the provider from its scripts):

  1  the card: GET /v1/laya says the route, the limits and the window   test_1_the_card
  2  open like the chat API: through the proxy, no token, a bearer ignored; a choice as a list
     and as {label: meaning}; label, p, sure, probs; the contract             test_2_choice_*
  3  a yes-no question (neither choices nor scale): yes or no, `yes`         test_3_yes_no
  4  a scale: the level, its index, the expected level                       test_4_scale
  5  texts: one answer a text, in order, one forward each; the contract      test_5_texts
  6  a long text: Laya reads its head and tail (the tail's label wins, the middle's is lost),
     `clipped` and `read_chars` say so                                       test_6_long_text
  7  invalid requests: 422 with `param`, the OpenAI error shape, no model reached
                                                                              test_7_invalid
  8  a busy engine: a completion holds it, the caller's deadline passes: 503 busy with
     Retry-After; idle again: answered                                      test_8_busy
  9  the journal: one `kind: laya_api` row a call, names and numbers, never the question's or
     the text's words; the Deciders status counts them                      test_9_journal
 10  the switch and the rate: the deciders document's api.mode off: 404 on both routes; on with
     rate_per_min 3: the fourth text in a minute is 429 with Retry-After     test_z_switch_and_rate
 11  off is the release: a server with no decision model has no route (404)  test_11_off
 12  the pieces: clip, the rate window, answer_of                            test_12_units

Run: python tests/test_laya_api.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_conductor_e2e import ADMIN, LAUNCH, PROXY, _call, _port, _wait  # noqa: E402
from tools import contract_check  # noqa: E402
from server import laya_api  # noqa: E402

CONTRACT = os.path.join(ROOT, "docs", "contract", "conductor-v1")
LOCK_DIR = tempfile.mkdtemp(prefix="laya-api-lock-")
CDIR = tempfile.mkdtemp(prefix="laya-cond-")
LEDGER = os.path.join(tempfile.mkdtemp(prefix="laya-led-"), "ledger.sqlite3")


def _env(models: str):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ["HOME"],
            "PYTHONPATH": ROOT, "CUDA_VISIBLE_DEVICES": "", "QSE_ADMIN_TOKEN": ADMIN,
            "QWEN38_DECIDER_MODELS": models, "FAKE_MEM_AVAILABLE_GIB": "64",
            "BOX_LOCK": os.path.join(LOCK_DIR, "no-box.flock")}


def _server(port, models: str, cdir: str, ledger: str):
    args = [sys.executable, "-c", LAUNCH, "--fake-engine", "--port", str(port),
            "--served-model", "qwen38-spark-engine", "--usage-ledger", ledger,
            "--fake-tps", "60", "--conductor-config", cdir]
    return subprocess.Popen(args, cwd=ROOT, env=_env(models), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


PORT, OFF = _port(), _port()
SRV = _server(PORT, "other-decider", CDIR, LEDGER)
SRV_OFF = _server(OFF, "", tempfile.mkdtemp(prefix="laya-cond-off-"),
                  os.path.join(tempfile.mkdtemp(prefix="laya-led-off-"), "ledger.sqlite3"))
MODEL = "other-decider"


def _ask(body, port=None, headers=None, token=None):
    """A script's call: through the proxy (as a reverse proxy in front of the engine sends it), no admin token."""
    return _call(port or PORT, "POST", "/v1/laya/ask", body,
                 headers=PROXY if headers is None else headers, token=token)


def _ok(obj, name):
    errs = contract_check.validate(obj, contract_check.load(name, CONTRACT), root=CONTRACT)
    assert not errs, f"{name}: {errs[:5]}"


def _questions() -> int:
    st, _, out = _call(PORT, "GET", "/v1/conductor/deciders/status")
    assert st == 200, out
    return next(m for m in out["models"] if m["name"] == MODEL)["questions"]


def _loaded():
    st, _, out = _call(PORT, "POST", f"/v1/conductor/deciders/models/{MODEL}/load", {})
    end = time.time() + 20
    while time.time() < end:
        st, _, out = _call(PORT, "GET", "/v1/conductor/deciders/status")
        if any(m["name"] == MODEL and m["state"] == "loaded" for m in out["models"]):
            return
        time.sleep(0.1)
    raise AssertionError(f"{MODEL} did not load: {out}")


def _deciders_put(api: dict) -> None:
    st, _, cur = _call(PORT, "GET", "/v1/conductor/deciders")
    assert st == 200, cur
    doc = dict(cur["doc"], questions=dict(cur["doc"]["questions"], laya_api=MODEL), api=api)
    st, _, out = _call(PORT, "PUT", "/v1/conductor/deciders",
                       {"doc": doc, "note": "test", "base_version": cur["version"]})
    assert st == 200, out


# ------------------------------------------------------------------ 1 the card
def test_1_the_card():
    _loaded()
    _deciders_put({})                       # the defaults, with this server's model for the route
    st, _, card = _call(PORT, "GET", "/v1/laya", headers=PROXY, token=None)
    assert st == 200, card
    assert card["route"] == "POST /v1/laya/ask" and card["on"] is True and card["ready"] is True
    assert card["model"] == MODEL and card["limits"]["read_chars"] == 560, card
    assert card["limits"]["max_texts"] == 16 and card["limits"]["rate_per_min"] == 240
    assert "none at the engine" in card["auth"]


# ------------------------------------------------------------------ 2 choice
def test_2_choice_open_like_chat():
    body = {"question": "Does this output show a failure?", "choices": ["failure", "clean"],
            "text": "Traceback (most recent call last):\n  ... failure in parse()"}
    st, hdr, out = _ask(body)
    assert st == 200, (st, out)
    _ok(out, "laya-ask")
    assert out["label"] == "failure" and out["p"] == 0.91 and out["sure"] is True, out
    assert out["probs"] == {"failure": 0.91, "clean": 0.09} and out["model"] == MODEL
    assert out["chars"] == len(body["text"]) and out["clipped"] is False
    # a page's fetch (the browser tool's evaluate script) may read the answer: CORS, preflight
    assert hdr.get("Access-Control-Allow-Origin") == "*", hdr
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/laya/ask", method="OPTIONS",
                                 headers={"Origin": "http://localhost:5173",
                                          "Access-Control-Request-Method": "POST"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 204 and r.headers["Access-Control-Allow-Origin"] == "*"
        assert "POST" in r.headers["Access-Control-Allow-Methods"]
    # a bearer the script sends anyway changes nothing, and the box itself is the same
    assert _ask(body, token="sk-anything-at-all")[2]["label"] == "failure"
    assert _ask(body, headers={})[0] == 200
    # the meanings, and a threshold above the answer: not sure
    st, _, out = _ask({"question": "Is the build ok?", "threshold": 0.95,
                       "choices": {"ok": "it compiled and linked", "broken": "it failed"},
                       "text": "linker: undefined symbol; build broken"})
    assert st == 200 and out["label"] == "broken" and out["sure"] is False, out
    # nothing fits: the first label at 0.2, not sure
    st, _, out = _ask({"question": "q?", "choices": ["red", "green"], "text": "blue"})
    assert out["label"] == "red" and out["p"] == 0.2 and out["sure"] is False, out


def test_3_yes_no():
    st, _, out = _ask({"question": "The page loaded with zero console errors.",
                       "text": "console: 0 errors, 2 warnings FAKE_TRUE"})
    assert st == 200 and out["label"] == "yes" and out["yes"] == 0.9 and out["sure"], out
    _ok(out, "laya-ask")
    st, _, out = _ask({"question": "The page loaded with zero console errors.",
                       "text": "console: 3 errors"})
    assert out["label"] == "no" and out["p"] == 0.9 and out["yes"] == 0.1, out
    assert out["probs"] == {"yes": 0.1, "no": 0.9}


def test_4_scale():
    st, _, out = _ask({"question": "How close is the output to the goal?",
                       "scale": ["far off", "partly", "close", "exact"],
                       "text": "matched 9 of 10 rows: 2"})
    assert st == 200, out
    _ok(out, "laya-ask")
    assert out["level"] == 2 and out["label"] == "close" and out["score"] == 2.0, out
    assert set(out["probs"]) == {"0", "1", "2", "3"} and out["p"] == 0.91


def test_5_texts():
    before = _questions()
    texts = ["run 1: ok", "run 2: Error: boom", "run 3: ok"]
    st, _, out = _ask({"question": "Which run failed?", "choices": ["ok", "Error"],
                       "texts": texts})
    assert st == 200, out
    _ok(out, "laya-ask-texts")
    assert [a["label"] for a in out["answers"]] == ["ok", "Error", "ok"], out
    assert [a["chars"] for a in out["answers"]] == [len(t) for t in texts]
    assert _questions() - before == 3, "one forward a text"


def test_6_long_text():
    tail = "x " * 1500 + "\nfinal: failure\n"
    st, _, out = _ask({"question": "Did it fail?", "choices": ["failure", "clean"],
                       "text": tail})
    assert st == 200 and out["label"] == "failure", out
    assert out["clipped"] is True and out["read_chars"] <= 560 and out["chars"] == len(tail)
    middle = "x " * 700 + " failure " + "x " * 700
    st, _, out = _ask({"question": "Did it fail?", "choices": ["failure", "clean"],
                       "text": middle})
    assert out["clipped"] is True and out["p"] == 0.2, "the middle of a long text is not read"
    assert laya_api.clip(middle, 560).endswith(middle[-300:])


def test_7_invalid():
    before = _questions()
    cases = [
        ({"choices": ["a", "b"], "text": "x"}, "question"),
        ({"question": "q", "choices": ["a"], "text": "x"}, "choices"),
        ({"question": "q", "choices": ["a", "b c"], "text": "x"}, "choices"),
        ({"question": "q", "choices": ["a", "a"], "text": "x"}, "choices"),
        ({"question": "q", "choices": [f"l{i}" for i in range(21)], "text": "x"}, "choices"),
        ({"question": "q", "choices": ["a", "b"], "scale": ["x", "y"], "text": "x"}, "scale"),
        ({"question": "q", "scale": ["only"], "text": "x"}, "scale"),
        ({"question": "q", "text": ""}, "texts"),
        ({"question": "q"}, "text"),
        ({"question": "q", "text": "x", "texts": ["y"]}, "texts"),
        ({"question": "q", "texts": ["x"] * 17}, "texts"),
        ({"question": "q", "text": "x" * 100001}, "texts"),
        ({"question": "q" * 401, "text": "x"}, "question"),
        ({"question": "q", "text": "x", "threshold": 2}, "threshold"),
        ({"question": "q", "text": "x", "deadline_ms": 0}, "deadline_ms"),
        ({"question": "q", "text": "x", "model": "laya-giant"}, "model"),
    ]
    for body, param in cases:
        st, _, out = _ask(body)
        assert st == 422 and out["error"]["type"] == "invalid_request_error", (body, st, out)
        assert out["error"].get("param") == param, (body, out)
    assert _questions() == before, "a refused request reaches no model"


def test_8_busy():
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
    while time.time() < end:
        if _call(PORT, "GET", "/health")[2].get("engine_busy"):
            break
        time.sleep(0.05)
    else:
        raise AssertionError("the completion never took the engine")
    t0 = time.time()
    st, hdr, out = _ask({"question": "q?", "choices": ["a", "b"], "text": "a",
                         "deadline_ms": 200})
    took = time.time() - t0
    assert st == 503 and out["error"]["type"] == "busy" and hdr.get("Retry-After"), (st, out)
    assert 0.18 <= took < 2.0 and not done.is_set(), took
    done.wait(60)
    assert _ask({"question": "q?", "choices": ["a", "b"], "text": "a"})[0] == 200


def test_9_journal():
    time.sleep(1.0)                          # the journal's writer thread
    conn = sqlite3.connect(LEDGER)
    rows = conn.execute("SELECT decision, detail FROM conductor_events WHERE kind='laya_api' "
                        "AND task='ask'").fetchall()
    conn.close()
    assert len(rows) >= 10, rows
    blob = json.dumps(rows)
    for words in ("Traceback", "Does this output", "console:", "final: failure", "far off"):
        assert words not in blob, f"the journal holds text: {words!r}"
    details = [json.loads(d) for _, d in rows]
    ok = [d for d in details if "error" not in d]
    assert {d["type"] for d in ok} == {"choice", "noul", "score"}, ok
    three = next(d for d in ok if d["texts"] == 3)
    assert three["labels"] == ["ok", "Error", "ok"] and three["model"] == MODEL
    assert all(len(d["q"]) == 12 and d["q_chars"] for d in details if d.get("q"))
    errs = [d for d in details if d.get("error")]
    assert {d["error"] for d in errs} >= {"invalid_request_error", "busy"}, errs
    st, _, s = _call(PORT, "GET", "/v1/conductor/deciders/status")
    _ok(dict(s, contract_version=s.get("contract_version", "1.0")), "deciders-status")
    assert s["api"]["mode"] == "on" and s["api"]["stats"]["calls"] == len(rows), s["api"]
    assert s["api"]["stats"]["errors"] == len(errs) and s["api"]["stats"]["last_ms"] is not None


def test_11_off():
    _wait(OFF, SRV_OFF)
    st, _, out = _call(OFF, "POST", "/v1/laya/ask", {"question": "q", "text": "x"},
                       headers=PROXY, token=None)
    assert st == 404, (st, out)
    assert _call(OFF, "GET", "/v1/laya", headers=PROXY, token=None)[0] == 404
    for port, path in ((OFF, "/v1/laya/ask"), (PORT, "/v1/chat/completions")):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="OPTIONS")
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("OPTIONS answered")
        except urllib.error.HTTPError as exc:
            assert exc.code == 501, (path, exc.code)      # as before L1


def test_12_units():
    assert laya_api.clip("abc", 10) == "abc"
    long = "H" * 100 + "M" * 1000 + "T" * 300
    c = laya_api.clip(long, 400)
    assert len(c) == 400 and c.startswith("H" * 100) and c.endswith("T" * 293) and "[...]" in c
    t = [0.0]
    r = laya_api.Rate(clock=lambda: t[0])
    assert r.take(2, 3) is None and r.take(1, 3) is None
    assert r.take(1, 3) == 61, "the window is full until the first stamp is a minute old"
    t[0] = 30.0
    assert r.take(1, 3) == 31
    t[0] = 60.0
    assert r.take(3, 3) is None
    assert r.take(5, 3) == 60, "more than the rate in one call"
    req = {"type": "choice", "threshold": 0.8, "criteria": {"a": "a", "b": "b"}}
    a = laya_api.answer_of({"choice": "b", "probabilities": {"a": 0.15, "b": 0.85}}, req)
    assert a == {"label": "b", "p": 0.85, "sure": True, "probs": {"a": 0.15, "b": 0.85}}
    req = {"type": "noul", "threshold": 0.8, "criteria": None}
    assert laya_api.answer_of({"noul": 0.3}, req)["label"] == "no"
    req = {"type": "score", "threshold": 0.5, "criteria": ["lo", "mid", "hi"]}
    a = laya_api.answer_of({"score": 1.6, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}, req)
    assert (a["label"], a["level"], a["score"], a["sure"]) == ("hi", 2, 1.6, True)


def test_z_switch_and_rate():
    _deciders_put({"mode": "off"})
    assert _ask({"question": "q", "choices": ["a", "b"], "text": "a"})[0] == 404
    assert _call(PORT, "GET", "/v1/laya", headers=PROXY, token=None)[0] == 404
    _deciders_put({"mode": "on", "rate_per_min": 3})
    # the calls of the last minute count too: at most 3 of these 5 pass, then 429
    got = [_ask({"question": "q", "choices": ["a", "b"], "text": "a"}) for _ in range(5)]
    codes = [st for st, _, _ in got]
    assert codes.count(200) <= 3 and codes[-1] == 429, codes
    st, hdr, out = got[-1]
    assert out["error"]["type"] == "rate_limited" and 1 <= int(hdr["Retry-After"]) <= 61, out
    st, _, out = _ask({"question": "q", "choices": ["a", "b"], "texts": ["a"] * 4})
    assert st == 429, "more texts in one call than the rate"
    _deciders_put({})


def _main():
    try:
        _wait(PORT, SRV)
        passed = 0
        for name in sorted(n for n in globals() if n.startswith("test_")):
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
