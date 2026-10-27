"""VIS-44 / VIS-45, the server's half: what the dashboard's supervisor views read.

The test matrix is VIS-44's Gherkin; this file holds the rows the server answers for, the
dashboard's vitest and Playwright specs hold the rest:

  1  the banner shows a live action          test_1_* (the action, its task, its score and the
                                             reasoning tokens it was taken at, in `last_action`)
  2  request rows carry the actions          test_2_* (the finished row's `conductor` block from
                                             the ledger, the drawer's `events?request=`)
  3  shadow decisions look different         test_3_* (`shadow` counts them, live and finished)
  5  export                                  tests/test_conductor_journal.py test_5_*
  6  an older server                         test_6_* (no monitor: no `conductor` anywhere; an
                                             older ledger: the row's block is null)
     the header's mode indicator             test_mode_* (live engine block and the summary)

Run: python tests/test_supervisor_views.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import activity, app, dashboard_api, ledger as ledger_mod  # noqa: E402
from server.conductor.monitor import Action, Journal, Monitor, RequestEvent  # noqa: E402
from server.conductor.monitor.policy import CLOSE_REASONING, PASS  # noqa: E402
from test_toolcheck import TOOLS, policy  # noqa: E402
from tools import contract_check  # noqa: E402


def _sup(mode: str = "on"):
    mon = Monitor(mode, policy(), None, Journal(echo=False))
    return mon, mon.begin(mode, RequestEvent("chatcmpl-v44", tools=TOOLS), [1, 2], None)


def _task(sup, tid: str):
    return next(t for sp in sup._at.values() for t in sp if t.id == tid)


# ------------------------------------------------------------------ 1: the banner

def test_1_the_last_action_carries_its_task_score_and_tokens():
    """"Supervisor: D1 closed the reasoning at 2,880 tokens (repeat 0.93)": the live block's
    `last_action` holds what the banner needs, the score and the reasoning tokens included."""
    _, sup = _sup("on")
    act = Action(CLOSE_REASONING, reason="repeat(0.93)", score=0.93, task="D1")
    sup._write("SP2", _task(sup, "D1"), act, "act", hidden=12, at=2880)
    c = activity.conductor_of(sup, time.perf_counter())
    la = c["last_action"]
    assert (la["task"], la["decision"], la["score"], la["at_tokens"], la["tier"]) == \
        ("D1", "CLOSE_REASONING", 0.93, 2880, 0), la
    assert la["ago_ms"] is not None and la["ago_ms"] >= 0
    assert c["last"] == la and c["shadow"] == 0


def test_1_a_pass_is_the_newest_decision_but_not_the_last_action():
    _, sup = _sup("on")
    sup._write("SP2", _task(sup, "D1"), Action(CLOSE_REASONING, reason="repeat(0.91)",
                                                 score=0.91, task="D1"), "act", at=2048)
    sup._write("SP2", _task(sup, "D1"), Action(PASS, reason="below(0.4)", score=0.4,
                                                 task="D1"), "act", at=4096)
    c = activity.conductor_of(sup, time.perf_counter())
    assert c["last"]["decision"] == "PASS" and c["last"]["at_tokens"] == 4096
    assert c["last_action"]["decision"] == "CLOSE_REASONING" and \
        c["last_action"]["at_tokens"] == 2048


def test_1_the_block_validates_against_live_1_2():
    """The new fields are in the schema (additive), and a 1.2 message without them still
    validates: an older 1.2 server's rows."""
    from test_live import Clock, _reg, _rec, _row, _valid
    c = Clock()
    reg = _reg(c)
    rec = _rec(c)
    reg.register(rec)
    rec.lock_acquired()
    _, sup = _sup("on")
    rec.sup = sup
    sup._write("SP2", _task(sup, "D1"), Action(CLOSE_REASONING, reason="repeat(0.93)",
                                                 score=0.93, task="D1"), "act", at=2880)
    snap = _valid(reg.snapshot())
    assert _row(snap, "chatcmpl-1")["activity"]["conductor"]["last_action"]["score"] == 0.93
    old = json.loads(json.dumps(snap))
    blk = _row(old, "chatcmpl-1")["activity"]["conductor"]
    blk.pop("shadow")
    for k in ("last", "last_action"):
        for f in ("score", "tier", "at_tokens"):
            blk[k].pop(f)
    assert not contract_check.check(old, "live")


# ------------------------------------------------------------------ 3: shadow decisions

def test_3_shadow_decisions_are_counted_apart_from_acts():
    _, sup = _sup("shadow")
    act = Action(CLOSE_REASONING, reason="repeat(0.88)", score=0.88, task="D1")
    sup._write("SP2", _task(sup, "D1"), act, "shadow", at=3000)
    sup._write("SP2", _task(sup, "D1"), Action(PASS, task="D1"), "shadow", at=3100)
    c = activity.conductor_of(sup, time.perf_counter())
    assert (c["acts"], c["shadow"]) == (0, 1), c
    assert c["last_action"] is None, "a shadow decision is not an action"
    assert c["last"]["mode"] == "shadow" and c["last"]["decision"] == "PASS"


# ------------------------------------------------------------------ 2: finished rows

def _ledger():
    path = os.path.join(tempfile.mkdtemp(prefix="qse-v44-"), "ledger.sqlite3")
    return ledger_mod.Ledger(path, retention_days=400).open()


def _req(led, rid, **cols):
    base = {"ts_ms": int(time.time() * 1000), "request_id": rid, "model": "TandemLLM",
            "client_id": "k:a", "client_kind": "opencode", "endpoint": "chat", "stream": 1,
            "status": 200, "finish_reason": "tool_calls", "completion_tokens": 40,
            "decode_tps": 40.0, "tool_calls": 1, "thinking": 1}
    led.submit(dict(base, **cols))


def _event(path, rid, mode, decision, n=1):
    from server.conductor import decisions
    jr = decisions.Journal(path)
    for i in range(n):
        jr.monitor_sink({"request_id": rid, "session": "s1", "sp": "SP2", "task": "D1",
                         "tier": 0, "mode": mode, "decision": decision, "reason": "x",
                         "score": 0.9, "ts": time.time() + i * 0.001, "text": ""})
    jr.flush()


def test_2_a_finished_row_carries_its_acts_and_the_drawer_reads_its_events():
    """Given a finished request with two actions, its row says 2 (the badge), acted, with the
    hidden tokens; another with one shadow decision says shadow 1; a raw one carries null."""
    led = _ledger()
    _req(led, "chatcmpl-a", conductor_mode="on", conductor_acts=2, conductor_hidden_tokens=31,
         conductor_acted=1)
    _req(led, "chatcmpl-s", conductor_mode="shadow", conductor_acts=0,
         conductor_hidden_tokens=0, conductor_acted=0)
    _req(led, "chatcmpl-r", conductor_mode="raw")
    led.flush()
    _event(led.path, "chatcmpl-a", "act", "CLOSE_REASONING", 2)
    _event(led.path, "chatcmpl-s", "shadow", "CLOSE_REASONING")
    _event(led.path, "chatcmpl-s", "shadow", "PASS")
    out = dashboard_api.DashboardAPI(led).requests({"limit": "10"})
    assert not contract_check.check(json.loads(json.dumps(out)), "requests")
    by = {r["request_id"]: r["conductor"] for r in out["requests"]}
    assert by["chatcmpl-a"] == {"mode": "on", "acts": 2, "shadow": 0, "hidden_tokens": 31,
                                "acted": True}, by
    assert by["chatcmpl-s"] == {"mode": "shadow", "acts": 0, "shadow": 1, "hidden_tokens": 0,
                                "acted": False}, by
    assert by["chatcmpl-r"] is None, by
    led.close()


# ------------------------------------------------------------------ 6: an older server

def test_6_an_older_ledger_gives_null_and_no_error():
    """A ledger without the journal table (schema version 1, or the conductor never ran):
    every row's block is null, and the query does not fail."""
    led = _ledger()
    _req(led, "chatcmpl-x")
    led.flush()
    conn = sqlite3.connect(led.path)
    conn.execute("DROP TABLE conductor_events")
    conn.commit()
    conn.close()
    out = dashboard_api.DashboardAPI(led).requests({"limit": "5"})
    assert [r["conductor"] for r in out["requests"]] == [None]
    led.close()


def test_6_no_monitor_no_conductor_anywhere():
    """A server without a monitor policy: neither the live engine block nor the summary's
    status carries `conductor`, so the dashboard renders nothing supervisor-related."""
    app.STATE["monitor"] = None
    assert "conductor" not in app._live_engine()
    assert "conductor" not in app._live()


# ------------------------------------------------------------------ the mode indicator

def test_mode_the_header_reads_off_shadow_or_on():
    for mode, word in (("raw", "off"), ("shadow", "shadow"), ("on", "on")):
        app.STATE["monitor"] = Monitor(mode, policy(), None, Journal(echo=False))
        assert app._live_engine()["conductor"] == {"mode": word}
        assert app._live()["conductor"] == {"mode": word}
    app.STATE["monitor"] = None


def test_mode_the_live_engine_block_validates():
    from test_live import Clock, _reg, _valid
    app.STATE["monitor"] = Monitor("shadow", policy(), None, Journal(echo=False))
    try:
        c = Clock()
        reg = _reg(c, engine=lambda: {"model": "TandemLLM", "version": "t", "draining": False,
                                      "kv": None, "memory": None, "store": None,
                                      **app._monitor_mode()})
        snap = _valid(reg.snapshot())
        assert snap["engine"]["conductor"] == {"mode": "shadow"}
    finally:
        app.STATE["monitor"] = None


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
