"""SRV-115 / SRV-116: the conductor's observability, on a CPU -- one journal for the router and the
session monitor, the ledger's schema version 2, the Prometheus counters, the live fields, the
session export and the 30-day retention.

The test matrix is SRV-115's Gherkin (SRV-116's table), through the real handler with the fake
engine in this process and the real journal on a ledger file of its own:

  1  an action is journaled, counted and marked     test_1_*
  2  a pass is counted too                          test_2_*
  3  the outcome is filled in later                 test_3_* (and the three checkers' formats)
  4  the live feed shows the action within a second test_4_*
  5  the session export lists every hidden text     test_5_*
  6  the ledger migrates in place                   test_6_* (a version-1 fixture; the box runs
                                                    tools/ledger_migrate_check.py on a copy of
                                                    its own ledger)
  7  off leaves no trace                            test_7_*

plus the retention, the weekly backup without the journal, the router's old file imported once,
and the speed views leaving acted requests out. The HTTP e2e (the /metrics scrape before and
after, the export over the API, the ledger row) is tests/test_conductor_e2e.py
(`test_one_journal_for_the_router_and_the_monitor`).

Run: python tests/test_conductor_journal.py
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

from server import activity, app, ledger as ledger_mod, live, metrics  # noqa: E402
from server.conductor import decisions, signals  # noqa: E402
from server.conductor.monitor import Action, Journal, Monitor, RequestEvent  # noqa: E402
from server.conductor.monitor.policy import CONTINUE_HIDDEN  # noqa: E402
from test_toolcheck import (TOOLS, Req, ask, policy, quiet, serve,  # noqa: E402
                            call_json)

SESSION = {"x-session-id": "ses_obs"}


def wired(mode: str = "on", pol=None, ledger: bool = True):
    """The fake engine with the monitor, the shared journal on a ledger file and the metrics
    sink, wired as `_serve` wires them; (monitor, journal, ledger)."""
    mon = serve(mode, pol)
    metrics.REGISTRY.reset()
    led = jr = None
    if ledger:
        path = os.path.join(tempfile.mkdtemp(prefix="qse-journal-"), "ledger.sqlite3")
        led = app.STATE["ledger"] = ledger_mod.Ledger(path, retention_days=400).open()
        jr = app.STATE["journal"] = decisions.Journal(path)
    if mon is not None:
        mon.journal.sinks.append(metrics.on_conductor_event)
        if jr is not None:
            mon.journal.sinks.append(jr.monitor_sink)
    return mon, jr, led


def chat(text, headers=None, tools=TOOLS, stream=True, msgs=None, **extra):
    body = dict({"messages": msgs or [{"role": "user", "content": text}], "stream": stream,
                 "max_tokens": 2000}, **extra)
    if tools:
        body["tools"] = tools
    return Req(body, headers).response()


def flush(jr, led):
    if led is not None:
        led.flush()
    if jr is not None:
        jr.flush()


def sample(name: str, **labels) -> float:
    """One sample of /metrics, 0 when the series has none."""
    want = name + ("{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}" if labels
                   else "")
    for ln in metrics.render().splitlines():
        if ln.startswith(want + " ") or (not labels and ln.startswith(name + " ")):
            return float(ln.rsplit(" ", 1)[1])
    return 0.0


def db_rows(led, sql, *args):
    conn = sqlite3.connect(led.path)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


# ------------------------------------------------------------------ 1: journaled, counted, marked

def test_1_an_action_is_journaled_counted_and_marked():
    """E1 continues an empty turn: a journal row with task E1, action CONTINUE_HIDDEN and its
    hidden tokens; the counter rises by one; the ledger row says acted."""
    mon, jr, led = wired()
    before = sample("qse_conductor_actions_total", task="E1", action="CONTINUE_HIDDEN")
    with quiet():
        chat("use the tool FAKE_EMPTY_ANSWER", headers=SESSION)
    flush(jr, led)
    after = sample("qse_conductor_actions_total", task="E1", action="CONTINUE_HIDDEN")
    assert after == before + 1, (before, after)
    rows = db_rows(led, "SELECT task, kind, action, mode, hidden_tokens, hidden_text_id, "
                        "outcome, session_key FROM conductor_events WHERE task = 'E1'")
    assert len(rows) == 1, rows
    task, kind, action, mode, hidden, tid, outcome, skey = rows[0]
    assert (task, kind, action, mode) == ("E1", "monitor", "CONTINUE_HIDDEN", "act")
    assert hidden > 0 and tid and json.loads(outcome)["kind"] == "tool_call", rows
    text = db_rows(led, "SELECT text FROM conductor_texts WHERE id = ?", tid)[0][0]
    assert "You ended your turn without a tool call" in text, text
    req = db_rows(led, "SELECT conductor_acts, conductor_hidden_tokens, conductor_acted "
                       "FROM requests")
    assert req == [(1, hidden, 1)], req
    assert sample("qse_conductor_hidden_tokens_total") == hidden


def test_1_a_text_is_stored_once():
    mon, jr, led = wired()
    with quiet():
        for _ in range(2):
            chat("use the tool FAKE_EMPTY_ANSWER", headers=SESSION)
    flush(jr, led)
    ids = db_rows(led, "SELECT hidden_text_id FROM conductor_events WHERE task = 'E1'")
    assert len(ids) == 2 and ids[0] == ids[1], ids
    used = db_rows(led, "SELECT COUNT(DISTINCT hidden_text_id) FROM conductor_events "
                        "WHERE hidden_text_id IS NOT NULL")[0][0]
    assert db_rows(led, "SELECT COUNT(*) FROM conductor_texts")[0][0] == used, "each text once"


# ------------------------------------------------------------------ 2: a pass is counted

def test_2_a_pass_is_counted_too():
    """D1 checks every block of the reasoning and finds nothing: counted, never a row."""
    mon, jr, led = wired()
    with quiet():
        chat("Name three rivers.", tools=[], headers=SESSION)
    flush(jr, led)
    n = sample("qse_conductor_decisions_total", task="D1", tier="0", mode="act", decision="pass")
    assert n >= 1, metrics.render()
    assert db_rows(led, "SELECT COUNT(*) FROM conductor_events WHERE task = 'D1'")[0][0] == 0
    # a held answer whose calls all pass: counted per task, one row for the answer
    with quiet():
        chat(call_json("read", filePath="js/a.js"), headers=SESSION,
             chat_template_kwargs={"enable_thinking": False})
    flush(jr, led)
    assert sample("qse_conductor_decisions_total", task="C3", tier="0", mode="act",
                  decision="pass") >= 1
    assert db_rows(led, "SELECT decision FROM conductor_events WHERE task = 'SP3'") == \
        [("PASS",)]


# ------------------------------------------------------------------ 3: the outcome, later

# the checkers' own formats (pagodacheck's from the 2026-09-30 session, as SRV-123's fixtures)
PAGODA = "pagodacheck  /demo\nRESULT: FAIL  <- (a) summer island colour differs by 3.5 > 1.0"
SKETCH = "sketchcheck a.png (style: spiral)\nRESULT: PASS"
SCENE = "scenecheck shot.png\nRESULT: FAIL <- (b) horizon error 4.2 > 2.0"


def test_3_the_outcome_is_filled_in_later():
    """Turn N acts; turn N+1 brings the checker's RESULT line in a tool result: the act's row
    gets it as next_checker, with the gate numbers."""
    mon, jr, led = wired()
    with quiet():
        chat("use the tool FAKE_EMPTY_ANSWER", headers=SESSION)
        msgs = [{"role": "user", "content": "use the tool FAKE_EMPTY_ANSWER"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "pagodacheck"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": PAGODA}]
        chat(None, msgs=msgs, headers=SESSION)
    flush(jr, led)
    got = db_rows(led, "SELECT next_checker FROM conductor_events WHERE task = 'E1' "
                       "ORDER BY ts_ms")
    first = json.loads(got[0][0])
    assert first[0]["verdict"] == "FAIL" and first[0]["checker"] == "pagodacheck", first
    assert first[0]["readings"][0]["gate"] == "a" and first[0]["readings"][0]["value"] == 3.5
    assert got[-1][0] is None, "the newest request has no next turn yet"


def test_3_the_three_checkers_formats():
    for text, checker, verdict, gate in ((PAGODA, "pagodacheck", "FAIL", "a"),
                                         (SKETCH, "sketchcheck", "PASS", None),
                                         (SCENE, "scenecheck", "FAIL", "b")):
        r = signals.checker_results(text)
        assert [(x["checker"], x["verdict"]) for x in r] == [(checker, verdict)], r
        assert (r[0]["readings"][0]["gate"] if r[0]["readings"] else None) == gate


# ------------------------------------------------------------------ 4: live within a second

def test_4_the_live_feed_shows_the_action_within_a_second():
    """While a hidden forward runs the running request says `supervising` with the action and
    its reason, and keeps saying it for a second, so a reader at one tick a second sees it."""
    from test_live import Clock, _reg, _rec, _row, _valid
    c = Clock()
    reg = _reg(c)
    rec = _rec(c)
    reg.register(rec)
    rec.lock_acquired()
    src = rec.track(iter(range(5)))
    next(src)
    mon = Monitor("on", policy(), None, Journal(echo=False))
    sup = rec.sup = mon.begin("on", RequestEvent("chatcmpl-1", tools=TOOLS), [1, 2], None)
    act = Action(CONTINUE_HIDDEN, reason="empty_turn(case=B)", task="E1")
    sup.forwarding = act
    a = _row(_valid(reg.snapshot()), "chatcmpl-1")["activity"]
    assert a["conductor"]["supervising"] and a["conductor"]["action"] == {
        "task": "E1", "action": "CONTINUE_HIDDEN", "reason": "empty_turn(case=B)"}, a
    assert a["label"] == "Supervising: E1 continuing an empty turn (empty_turn(case=B))"
    sup.forwarding, sup.forwarded_at = None, c.perf()
    sup.last_action = {"task": "E1", "decision": "CONTINUE_HIDDEN", "mode": "act",
                       "reason": "empty_turn(case=B)", "at": time.time()}
    c.tick(0.9)
    a = _row(_valid(reg.snapshot()), "chatcmpl-1")["activity"]
    assert a["conductor"]["supervising"], "held for a second after the forward"
    c.tick(0.2)
    a = _row(_valid(reg.snapshot()), "chatcmpl-1")["activity"]
    assert not a["conductor"]["supervising"] and a["conductor"]["last_action"]["task"] == "E1"
    rec2 = _rec(c, "chatcmpl-2")
    reg.register(rec2)
    assert _row(_valid(reg.snapshot()), "chatcmpl-2")["activity"]["conductor"] is None, \
        "a request the monitor does not watch carries null"
    assert reg.snapshot()["contract_version"] == "1.2"


def test_4_the_label_words():
    a = {"state": "writing", "conductor": {"supervising": True, "action": {
        "task": "C4", "action": "REGENERATE_CALL", "reason": "unreadable(x)"}}}
    assert activity.label(a) == ("Supervising: C4 having the tool calls written again "
                                 "(unreadable(x))")
    a["conductor"]["supervising"] = False
    assert activity.label(a) == "Writing"


# ------------------------------------------------------------------ 5: the export

def test_5_the_session_export_lists_every_hidden_injection():
    """Three hidden injections in a session (two hidden turns, one regeneration): the export
    says "3 interventions in 3 turns" and lists each exact text, its position, the reason and
    the outcome, in JSON and Markdown."""
    mon, jr, led = wired()
    with quiet():
        chat("use the tool FAKE_EMPTY_ANSWER", headers=SESSION,
             chat_template_kwargs={"enable_thinking": False})
        chat("call it FAKE_BAD_CALL", headers=SESSION,
             chat_template_kwargs={"enable_thinking": False})
        chat("use the tool FAKE_EMPTY_ANSWER again", headers=SESSION,
             chat_template_kwargs={"enable_thinking": False})
    flush(jr, led)
    key = mon.journal.events()[0]["session"]
    evs = jr.rows(kind="monitor", session=key, limit=100)
    exp = decisions.export(key, [], evs)
    assert exp["summary"] == "3 interventions in 3 turns", exp["summary"]
    assert [i["action"] for i in exp["injections"]] == ["CONTINUE_HIDDEN", "REGENERATE_CALL",
                                                       "CONTINUE_HIDDEN"]
    for i in exp["injections"]:
        assert i["hidden"] and i["text"] and i["tokens"] > 0 and i["pos"] > 0, i
        assert i["outcome"]["kind"] == "tool_call", i
    md = decisions.export_md(key, [], evs)
    assert "3 interventions in 3 turns" in md
    for i in exp["injections"]:
        assert i["text"] in md and f"Hidden text at token {i['pos']}" in md
    # the file alone (a restart: the rings are empty) gives the same record
    cold = decisions.Journal(led.path)
    again = decisions.export(key, [], cold.rows(kind="monitor", session=key, limit=100))
    assert [i["text"] for i in again["injections"]] == [i["text"] for i in exp["injections"]]


# ------------------------------------------------------------------ 6: the migration

V1_DDL = ledger_mod.DDL     # the requests table as version 1 created it, before ADDED


def _v1(path: str, n: int = 25) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(V1_DDL)
    conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?)",
                     [("schema_version", "1"), ("created_at", "1790000000000")])
    conn.executemany("INSERT INTO requests (ts_ms, request_id, model, client_id, client_kind, "
                     "endpoint, stream, status, finish_reason, completion_tokens, decode_tps) "
                     "VALUES (?, ?, 'qwen38-spark-engine', 'anon', 'curl', 'chat', 1, 200, "
                     "'stop', 40, 50.0)", [(1790000000000 + i, f"r{i}") for i in range(n)])
    conn.commit()
    conn.close()


def test_6_the_ledger_migrates_in_place():
    path = os.path.join(tempfile.mkdtemp(prefix="qse-v1-"), "ledger.sqlite3")
    _v1(path)
    led = ledger_mod.Ledger(path).open()
    try:
        assert led.schema_was == 1
        conn = sqlite3.connect(path)
        assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == \
            (str(ledger_mod.SCHEMA_VERSION),)
        assert conn.execute("SELECT COUNT(*), SUM(completion_tokens) FROM requests"
                            ).fetchone() == (25, 1000), "every old row is kept"
        cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
        assert {"conductor_acts", "conductor_hidden_tokens", "conductor_acted",
                "profile", "model_served"} <= cols
        assert conn.execute("SELECT conductor_acted FROM requests LIMIT 1").fetchone() == (None,)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"conductor_events", "conductor_texts"} <= tables
        assert conn.execute("SELECT COUNT(*) FROM meta WHERE key = ?",
                            (f"migrated_v1_to_v{ledger_mod.SCHEMA_VERSION}",)).fetchone() == (1,)
        conn.close()
    finally:
        led.close()
    led = ledger_mod.Ledger(path).open()            # idempotent: a second start changes nothing
    assert led.schema_was == ledger_mod.SCHEMA_VERSION
    led.close()


def test_6_the_box_check_runs_on_a_copy_and_never_writes_the_source():
    from tools import ledger_migrate_check
    src = os.path.join(tempfile.mkdtemp(prefix="qse-v1-"), "ledger.sqlite3")
    _v1(src, 40)
    before = open(src, "rb").read()
    rep = ledger_migrate_check.check(src)
    assert rep["ok"] and rep["rows_before"] == rep["rows_after"] == 40, rep
    assert rep["schema_before"] == 1 and rep["schema_after"] == ledger_mod.SCHEMA_VERSION
    assert open(src, "rb").read() == before, "the source file is read, never written"


# ------------------------------------------------------------------ 7: off leaves no trace

def test_7_off_leaves_no_trace():
    mon, jr, led = wired("raw")
    jr = app.STATE["journal"]
    with quiet():
        chat("use the tool FAKE_EMPTY_ANSWER", headers=SESSION)
        chat("call it FAKE_BAD_CALL", headers=SESSION)
    flush(jr, led)
    assert db_rows(led, "SELECT COUNT(*) FROM conductor_events")[0][0] == 0
    assert db_rows(led, "SELECT conductor_acts, conductor_hidden_tokens, conductor_acted "
                        "FROM requests") == [(None, None, None)] * 2
    samples = [ln for ln in metrics.render().splitlines()
               if ln.startswith("qse_conductor_") and not ln.endswith(" 0")]
    assert samples == [], samples


# ------------------------------------------------------------------ retention, backup, import

def test_retention_drops_old_rows_and_their_texts():
    path = os.path.join(tempfile.mkdtemp(prefix="qse-ret-"), "ledger.sqlite3")
    led = ledger_mod.Ledger(path).open()
    jr = decisions.Journal(path, retention_days=30)
    old = time.time() - 31 * 86400
    jr.monitor_sink({"ts": old, "request_id": "a", "session": "s", "task": "E1", "mode": "act",
                     "decision": "CONTINUE_HIDDEN", "text": "old nudge", "hidden_tokens": 3})
    jr.monitor_sink({"ts": time.time(), "request_id": "b", "session": "s", "task": "E1",
                     "mode": "act", "decision": "CONTINUE_HIDDEN", "text": "new nudge",
                     "hidden_tokens": 3})
    jr.flush()
    conn = jr._connect()
    jr.prune(conn)                    # the writer's pass at start may have been first
    assert [r[0] for r in conn.execute("SELECT request_id FROM conductor_events")] == ["b"]
    assert [r[0] for r in conn.execute("SELECT text FROM conductor_texts")] == ["new nudge"]
    assert jr._next_prune > time.time() + 60, "the next pass is tonight at 04:00"
    conn.close()
    led.close()


def test_the_weekly_backup_holds_no_journal():
    path = os.path.join(tempfile.mkdtemp(prefix="qse-bak-"), "ledger.sqlite3")
    led = ledger_mod.Ledger(path).open()
    jr = decisions.Journal(path)
    jr.monitor_sink({"ts": time.time(), "request_id": "a", "session": "s", "task": "E1",
                     "mode": "act", "decision": "CONTINUE_HIDDEN", "text": "t"})
    jr.flush()
    dest = led.backup()
    conn = sqlite3.connect(dest)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "requests" in tables and not tables & {"conductor_events", "conductor_texts"}
    led.close()


def test_the_routers_old_file_is_imported_once():
    d = tempfile.mkdtemp(prefix="qse-legacy-")
    old = os.path.join(d, "decisions.sqlite3")
    conn = sqlite3.connect(old)
    conn.execute("CREATE TABLE route_decisions (id TEXT PRIMARY KEY, ts_ms INTEGER, request_id "
                 "TEXT, session_key TEXT, class_id TEXT, source TEXT, confidence REAL, "
                 "injected TEXT, label TEXT, acted INTEGER)")
    conn.execute("INSERT INTO route_decisions VALUES ('r1', 1790000000000, 'req1', 'c:s', "
                 "'code_edit', 'decider', 0.9, ?, 'right', 1)",
                 (json.dumps([{"pos": 10, "tokens": 4, "text": "a note", "reason": "x"}]),))
    conn.commit()
    conn.close()
    path = os.path.join(d, "ledger.sqlite3")
    ledger_mod.Ledger(path).open().close()
    jr = decisions.Journal(path)
    assert jr.import_legacy(old) == 1 and jr.import_legacy(old) == 0
    r = jr.get("r1")
    assert r["kind"] == "route" and r["task"] == "route" and r["label"] == "right"
    assert r["action"] == "PROMPT_TAIL_NOTE" and r["hidden_tokens"] == 4 and r["tier"] == 1


def test_the_speed_views_leave_acted_requests_out():
    from server import dashboard_api
    assert "COALESCE(conductor_acted, 0) = 0" in dashboard_api.DECODED
    path = os.path.join(tempfile.mkdtemp(prefix="qse-speed-"), "ledger.sqlite3")
    led = ledger_mod.Ledger(path).open()
    base = {"request_id": "r", "model": "TandemLLM", "client_id": "anon", "client_kind": "curl",
            "endpoint": "chat", "stream": 1, "status": 200, "finish_reason": "stop",
            "completion_tokens": 50, "decode_ms": 1000.0, "tool_calls": 0, "thinking": 0}
    now = int(time.time() * 1000)
    led.submit(dict(base, ts_ms=now - 2000, decode_tps=40.0, conductor_acted=0))
    led.submit(dict(base, ts_ms=now - 1000, decode_tps=4.0, conductor_acted=1))
    led.flush()
    conn = sqlite3.connect(path)
    got = conn.execute(f"SELECT decode_tps FROM requests WHERE {dashboard_api.DECODED}"
                       ).fetchall()
    conn.close()
    assert got == [(40.0,)], got
    led.close()


def test_the_tool_check_rows_keep_no_arguments():
    """A repaired call's arguments (a user's file, a command) never go into the journal."""
    mon, jr, led = wired()
    with quiet():
        ask("FAKE_DROP_CLOSER " + call_json("bash", command="cat secret.txt"))
    flush(jr, led)
    detail = db_rows(led, "SELECT detail FROM conductor_events WHERE task = 'C4'")
    assert detail and "secret.txt" not in detail[0][0], detail


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
