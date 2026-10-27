"""X1 (2026-10-03): Laya's driver sessions on `POST /v1/laya/ask` and the playtest kit's routes.

The intended use of Laya: it plays the software under test. A driver asks every ~250 ms which action
comes next; it needs its own rate (a session's, 10 texts a second), not the scripts' minute, one
journal row a session instead of one a call, and the kit the model copies into a project.

The real server with the fake engine and the fake decision model "other-decider" (as
tests/test_laya_api.py), the conductor on a temp directory.

  1  the card names `session`, the driver limits and the kit's files       test_1_the_card
  2  a session's rate: 10 texts in a second pass, the 11th is 429 with Retry-After 1, the next
     second passes again; another session has its own; the scripts' minute is not touched
     (rate_per_min 3 and 30 session texts, then 3 script calls pass)       test_2_rates
  3  sessions at once: drive_sessions 2, a third is 429 until one ends     test_3_sessions
  4  the end: the session's numbers back (calls, texts, labels, p, unsure, ms), one `laya_drive`
     row in the journal with the driver's summary (numbers only), no state's or question's words;
     no `laya_api` row a session call; a second end is 404               test_4_end_and_journal
  5  an idle session's row: drive_idle_s 5, then any call of the routes writes it (`idle`)
                                                                         test_5_idle
  6  invalid: a bad session name 422 (no model reached), an end without a name 422
                                                                         test_6_invalid
  7  the kit: GET /v1/laya/kit/laya_drive.py and laya-drive.js are the repo's files, CORS, open
     through the proxy; an unknown file 404; the preflight of drive/end    test_7_kit
  8  the latency of the server path: 200 sequential session calls through HTTP, one connection
     each (the fake model's forward is free): p50 and p95 printed, p50 under 50 ms
                                                                         test_8_latency
  9  the status: deciders-status keeps its contract with the drive fields  test_9_status

Run: python tests/test_laya_drive_api.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_conductor_e2e import ADMIN, LAUNCH, PROXY, _call, _port, _wait  # noqa: E402
from tools import contract_check  # noqa: E402

CONTRACT = os.path.join(ROOT, "docs", "contract", "conductor-v1")
LEDGER = os.path.join(tempfile.mkdtemp(prefix="drive-led-"), "ledger.sqlite3")
MODEL = "other-decider"
LOCK_DIR = tempfile.mkdtemp(prefix="drive-lock-")


def _env(models: str):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ["HOME"],
            "PYTHONPATH": ROOT, "CUDA_VISIBLE_DEVICES": "", "QSE_ADMIN_TOKEN": ADMIN,
            "QWEN38_DECIDER_MODELS": models, "FAKE_MEM_AVAILABLE_GIB": "64",
            "BOX_LOCK": os.path.join(LOCK_DIR, "no-box.flock")}


PORT = _port()
SRV = subprocess.Popen(
    [sys.executable, "-c", LAUNCH, "--fake-engine", "--port", str(PORT), "--served-model",
     "qwen38-spark-engine", "--usage-ledger", LEDGER, "--fake-tps", "60", "--conductor-config",
     tempfile.mkdtemp(prefix="drive-cond-")],
    cwd=ROOT, env=_env(MODEL), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
STATE = '{"x":412,"gap_ahead":38,"lives":3,"secretword":"zebra-kumquat"}'
Q = "Which action keeps the runner alive? QUESTION-WORDS-NOT-KEPT"


def _ask(body, headers=None):
    return _call(PORT, "POST", "/v1/laya/ask", body, headers=PROXY if headers is None else headers,
                 token=None)


def _step(sid, text=STATE, **kw):
    return _ask({"question": Q, "choices": {"run": "move right", "jump": "jump"}, "text": text,
                 "session": sid, **kw})


def _end(sid, summary=None):
    return _call(PORT, "POST", "/v1/laya/drive/end",
                 {"session": sid, **({"summary": summary} if summary is not None else {})},
                 headers=PROXY, token=None)


def _deciders_put(api: dict) -> None:
    st, _, cur = _call(PORT, "GET", "/v1/conductor/deciders")
    assert st == 200, cur
    doc = dict(cur["doc"], questions=dict(cur["doc"]["questions"], laya_api=MODEL), api=api)
    st, _, out = _call(PORT, "PUT", "/v1/conductor/deciders",
                       {"doc": doc, "note": "test", "base_version": cur["version"]})
    assert st == 200, out


def _rows(kind):
    time.sleep(1.0)                                  # the journal's writer thread
    conn = sqlite3.connect(LEDGER)
    rows = conn.execute("SELECT decision, detail FROM conductor_events WHERE kind=?",
                        (kind,)).fetchall()
    conn.close()
    return [(d, json.loads(x)) for d, x in rows]


def _loaded():
    _call(PORT, "POST", f"/v1/conductor/deciders/models/{MODEL}/load", {})
    end = time.time() + 20
    while time.time() < end:
        st, _, out = _call(PORT, "GET", "/v1/conductor/deciders/status")
        if any(m["name"] == MODEL and m["state"] == "loaded" for m in out["models"]):
            return
        time.sleep(0.1)
    raise AssertionError(f"{MODEL} did not load")


def _second():
    """Wait for the start of a fresh second of the session window."""
    time.sleep(1.05)


# ------------------------------------------------------------------ 1
def test_1_the_card():
    _loaded()
    _deciders_put({})
    st, _, card = _call(PORT, "GET", "/v1/laya", headers=PROXY, token=None)
    assert st == 200 and "session" in card["request"], card
    assert card["limits"]["drive_rate_per_s"] == 10 and card["limits"]["drive_sessions"] == 4
    assert set(card["kit"]) == {"laya_drive.py", "laya-drive.js"}, card["kit"]


# ------------------------------------------------------------------ 2
def test_2_rates():
    _deciders_put({"rate_per_min": 3})
    _second()
    codes = [_step("rate-a")[0] for _ in range(10)]
    assert codes == [200] * 10, codes
    st, hdr, out = _step("rate-a")
    assert st == 429 and out["error"]["type"] == "rate_limited" and hdr["Retry-After"] == "1", out
    assert out["error"]["param"] == "session"
    assert _step("rate-b")[0] == 200, "another session has its own second"
    _second()
    assert _step("rate-a")[0] == 200, "a new second"
    for _ in range(2):
        _second()
        assert all(_step("rate-a")[0] == 200 for _ in range(10))
    # 30 and more session texts this minute, and the scripts' 3 a minute are all still there
    codes = [_ask({"question": "q", "choices": ["a", "b"], "text": "a"})[0] for _ in range(4)]
    assert codes == [200, 200, 200, 429], codes
    # texts count: 4 at once is 4 of the 10
    _second()
    st, _, out = _ask({"question": Q, "choices": ["run", "jump"], "texts": ["run"] * 4,
                       "session": "rate-c"})
    assert st == 200 and [a["label"] for a in out["answers"]] == ["run"] * 4, out
    assert [_step("rate-c")[0] for _ in range(7)] == [200] * 6 + [429]
    for sid in ("rate-a", "rate-b", "rate-c"):
        assert _end(sid)[0] == 200
    _deciders_put({})


# ------------------------------------------------------------------ 3
def test_3_sessions():
    _deciders_put({"drive_sessions": 2})
    assert _step("s1")[0] == 200 and _step("s2")[0] == 200
    st, hdr, out = _step("s3")
    assert st == 429 and "at most 2 driver sessions" in out["error"]["message"], out
    assert _step("s1")[0] == 200, "a live session goes on"
    assert _end("s2")[0] == 200
    assert _step("s3")[0] == 200
    for sid in ("s1", "s3"):
        _end(sid)
    _deciders_put({})


# ------------------------------------------------------------------ 4
def test_4_end_and_journal():
    before_api = len(_rows("laya_api"))
    _second()
    labels = []
    for text in ('{"x":1,"gap_ahead":30} jump', '{"x":2} run', '{"x":3} run', '{"x":4} neither'):
        st, _, out = _step("end-me", text=text)
        assert st == 200, out
        labels.append(out["label"])
    st, _, row = _end("end-me", {"episodes": 2, "deaths": 1, "wins": 1, "steps": 4,
                                 "note": "a string is dropped", "busy": True, "stuck": -3})
    assert st == 200, row
    assert row["session"] == "end-me" and row["ended"] == "end" and row["calls"] == 4
    assert row["texts"] == 4 and row["labels"] == {"jump": 1, "run": 3}, row
    assert row["unsure"] == 1 and row["p_min"] == 0.2 and 0.2 < row["p_mean"] < 0.91, row
    assert row["ms_p50"] is not None and row["ms_max"] >= row["ms_p50"]
    assert len(row["q"]) == 1 and len(row["q"][0]) == 12
    assert _end("end-me")[0] == 404, "ended once"
    drive = [d for _, d in _rows("laya_drive") if d["session"] == "end-me"]
    assert len(drive) == 1, drive
    assert drive[0]["summary"] == {"episodes": 2, "steps": 4, "deaths": 1, "wins": 1}, drive[0]
    blob = json.dumps(_rows("laya_drive")) + json.dumps(_rows("laya_api"))
    for words in ("zebra-kumquat", "QUESTION-WORDS", "gap_ahead", "a string is dropped"):
        assert words not in blob, f"the journal holds text: {words!r}"
    assert len(_rows("laya_api")) == before_api, "no row a session call"


# ------------------------------------------------------------------ 5
def test_5_idle():
    _deciders_put({"drive_idle_s": 5})
    assert _step("idle-1")[0] == 200
    time.sleep(5.3)
    assert _call(PORT, "GET", "/v1/laya", headers=PROXY, token=None)[0] == 200
    assert _step("other")[0] == 200            # any call of the ask route sweeps
    rows = [(d, x) for d, x in _rows("laya_drive") if x["session"] == "idle-1"]
    assert rows and rows[0][0] == "idle" and rows[0][1]["calls"] == 1, rows
    assert _end("idle-1")[0] == 404
    _end("other")
    _deciders_put({})


# ------------------------------------------------------------------ 6
def test_6_invalid():
    for sid in ("", "has space", "x" * 65, 12, "a/b"):
        st, _, out = _ask({"question": "q", "choices": ["a", "b"], "text": "a", "session": sid})
        assert st == 422 and out["error"]["param"] == "session", (sid, out)
    assert _call(PORT, "POST", "/v1/laya/drive/end", {}, headers=PROXY, token=None)[0] == 422
    assert _end("never-started")[0] == 404


# ------------------------------------------------------------------ 7
def test_7_kit():
    for name, ctype in (("laya_drive.py", "text/x-python"), ("laya-drive.js", "text/javascript")):
        st, hdr, body = _call(PORT, "GET", f"/v1/laya/kit/{name}", headers=PROXY, token=None,
                              raw=True)
        assert st == 200 and hdr["Content-Type"].startswith(ctype), hdr
        assert hdr["Access-Control-Allow-Origin"] == "*"
        with open(os.path.join(ROOT, "server", "laya_kit", name), "rb") as f:
            assert body == f.read()
    assert _call(PORT, "GET", "/v1/laya/kit/../app.py", headers=PROXY, token=None, raw=True)[0] == 404
    assert _call(PORT, "GET", "/v1/laya/kit/nothing.py", headers=PROXY, token=None, raw=True)[0] == 404
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/laya/drive/end", method="OPTIONS",
                                 headers={"Origin": "http://localhost:5173",
                                          "Access-Control-Request-Method": "POST"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 204 and r.headers["Access-Control-Allow-Origin"] == "*"


# ------------------------------------------------------------------ 8
def test_8_latency():
    _deciders_put({"drive_rate_per_s": 50})
    ms = []
    for i in range(200):
        t0 = time.perf_counter()
        st, _, out = _step("lat", text=f'{{"x":{i},"gap_ahead":{40 - i % 40}}}')
        ms.append((time.perf_counter() - t0) * 1e3)
        assert st == 200, out
        if i % 45 == 44:
            time.sleep(1.0)                    # under 50 a second
    _, _, row = _end("lat")
    ms.sort()
    p50, p95 = ms[len(ms) // 2], ms[int(len(ms) * 0.95)]
    print(f"    server path, 200 session steps (HTTP, the fake model): p50 {p50:.1f} ms, "
          f"p95 {p95:.1f} ms; the server's own ms p50 {row['ms_p50']}, p95 {row['ms_p95']}")
    assert p50 < 50, (p50, p95)
    _deciders_put({})


# ------------------------------------------------------------------ 9
def test_9_status():
    st, _, s = _call(PORT, "GET", "/v1/conductor/deciders/status")
    errs = contract_check.validate(dict(s, contract_version=s.get("contract_version", "1.0")),
                                   contract_check.load("deciders-status", CONTRACT), root=CONTRACT)
    assert not errs, errs[:5]
    assert s["api"]["drive_rate_per_s"] == 10 and s["api"]["stats"]["drive_sessions"] == 0, s["api"]


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
        SRV.terminate()
        try:
            SRV.wait(10)
        except subprocess.TimeoutExpired:
            SRV.kill()


if __name__ == "__main__":
    _main()
