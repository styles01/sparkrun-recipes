"""The idle-gap histogram (OPS-49, QA OPS-50) on a planted ledger and planted opencode steps.

OPS-49's Gherkin, row by row:
  1. gaps per tool kind: the report prints the histogram, p50 / p90 / p99 per kind and overall,
     and the gaps above 5 s and above 20 s per hour;
  2. a planted fixture reproduces: ten requests in two sessions with known gaps of 0.3, 1.5, 4,
     12 and 45 s give every bucket count and percentile exactly;
  3. compaction and session end: the compaction's gap is flagged compaction (its request's own
     length kept beside it) and the last request of a session produces no gap.
Plus: the ledger is opened read-only (its mtime and size do not move, no -wal or -shm appears,
the live ledger's path is refused), a request of another session inside a gap is busy time and
not idle, the clock offset between opencode and the ledger is found, the steps reader keeps
names and times only (no output, no command) and marks a checker run from its RESULT line, and
without opencode's steps the sessions are the conductor's session keys.

Run: python tests/test_idle_gaps.py
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import ledger  # noqa: E402
from tools import idle_gaps  # noqa: E402

T0 = 1_790_000_000_000          # ms, the planted night's first request
OFFSET_MS = 200                 # opencode's clock runs 0.2 s behind the ledger's

# (session, start ms after T0, duration ms, the step's tools, compaction)
A = [("ses_a", 0, 2000, [("read", None)], False),
     ("ses_a", 2300, 2000, [("edit", None)], False),            # gap 0.3 after read
     ("ses_a", 5800, 2000, [("bash", None)], False),            # gap 1.5 after edit
     ("ses_a", 11800, 2000, [("bash", "calccheck")], False),    # gap 4 after bash
     ("ses_a", 25800, 2000, [("chrome-devtools_navigate_page", None)], False),  # 12 after checker
     ("ses_a", 72800, 2000, [], False)]                         # 45 after browser; last
B = [("ses_b", 1_000_000, 2000, [("write", None)], False),
     ("ses_b", 1_003_500, 30000, [], True),                     # gap 1.5 after write
     ("ses_b", 1_037_500, 2000, [("chrome-devtools_take_screenshot", None)], False),  # 4, compaction
     ("ses_b", 1_051_500, 2000, [], False)]                     # 12 after screenshot; last


def make_ledger(path: str, rows, events=()) -> None:
    conn = sqlite3.connect(path)
    ledger.migrate(conn)
    for i, (sid, start, dur, _tools, _c) in enumerate(rows):
        conn.execute("INSERT INTO requests (ts_ms, request_id, model, client_id, client_kind, "
                     "endpoint, stream, status, prompt_tokens, total_ms) VALUES "
                     "(?, ?, 'TandemLLM', 'k1', 'opencode', 'chat', 1, 200, 1000, ?)",
                     (T0 + start, f"req-{sid}-{i}", float(dur)))
    for rid, key in events:
        conn.execute("INSERT INTO conductor_events (id, ts_ms, request_id, session_key, kind) "
                     "VALUES (?, ?, ?, ?, 'route')", ("ev-" + rid, T0, rid, key))
    conn.commit()
    conn.close()


def make_steps(rows) -> list[dict]:
    out = []
    for i, (sid, start, dur, tools, comp) in enumerate(rows):
        c = T0 + start - OFFSET_MS
        out.append({"session": sid, "title": sid, "id": f"msg-{sid}-{i}", "created": c,
                    "completed": c + dur, "compaction": comp,
                    "finish": "tool-calls" if tools else "stop",
                    "tools": [{"tool": name, "start": c + dur + 10, "end": c + dur + 20 + k,
                               "status": "completed", "checker": chk}
                              for k, (name, chk) in enumerate(tools)]})
    return out


def planted(tmp: str):
    path = os.path.join(tmp, "ledger-copy.sqlite3")
    make_ledger(path, A + B)
    return path, make_steps(A + B)


def run(path: str, steps: list[dict], groups=None) -> dict:
    rows = idle_gaps.read_ledger(path, T0 - 600_000, T0 + 2_000_000)
    return idle_gaps.analyse(rows, steps, groups)


# ------------------------------------------------------------------ row 2: the planted fixture

def test_a_planted_fixture_reproduces():
    tmp = tempfile.mkdtemp(prefix="idle-")
    path, steps = planted(tmp)
    rep = run(path, steps, {"both": ["ses_a", "ses_b"]})
    o = rep["overall"]
    assert (o["requests"], o["gaps"], o["sessions"]) == (10, 8, 2), o
    got = sorted(g["gap_s"] for gs in rep["gaps"].values() for g in gs)
    assert got == [0.3, 1.5, 1.5, 4.0, 4.0, 12.0, 12.0, 45.0], got
    assert [b["count"] for b in o["histogram"]] == [1, 2, 2, 2, 1, 0], o["histogram"]
    assert [b["idle_s"] for b in o["histogram"]] == [0.3, 3.0, 8.0, 24.0, 45.0, 0.0]
    assert o["overall"] == {"p50": 4.0, "p90": 45.0, "p99": 45.0}, o["overall"]
    pk = o["per_kind"]
    want = {"read": 0.3, "edit": 1.5, "bash": 4.0, "checker": 12.0, "browser": 45.0,
            "write": 1.5, "compaction": 4.0, "screenshot": 12.0}
    assert {k: (v["n"], v["p50"], v["p90"], v["p99"]) for k, v in pk.items()} == \
        {k: (1, v, v, v) for k, v in want.items()}, pk
    assert (o["over_5s"], o["over_20s"]) == (3, 1)
    assert rep["groups"]["both"]["gaps"] == 8 and rep["groups"]["both"]["members"] == ["ses_a",
                                                                                       "ses_b"]
    # opencode's clock is found and taken out before matching
    assert rep["sessions"]["ses_a"]["clock_offset_s"] == OFFSET_MS / 1000
    assert rep["sessions"]["ses_a"]["steps_unmatched"] == 0


def test_percentiles_are_nearest_rank():
    v = [0.3, 1.5, 4.0, 12.0, 45.0]
    assert [idle_gaps.pct(v, q) for q in (0.5, 0.9, 0.99)] == [4.0, 45.0, 45.0]
    assert idle_gaps.pct([], 0.5) is None and idle_gaps.pct([7.0], 0.99) == 7.0


# ------------------------------------------------------------------ row 3: compaction, session end

def test_compaction_and_session_end_are_labelled():
    tmp = tempfile.mkdtemp(prefix="idle-")
    path, steps = planted(tmp)
    rep = run(path, steps)
    a, b = rep["gaps"]["ses_a"], rep["gaps"]["ses_b"]
    assert len(a) == len(A) - 1 and len(b) == len(B) - 1, "the last request has no gap"
    assert a[-1]["next_request_id"] == "req-ses_a-5", "the last gap ends at the last request"
    assert not any(g["request_id"] == "req-ses_a-5" for g in a)
    comp = [g for g in b if g["compaction"]]
    assert len(comp) == 1 and comp[0]["kind"] == "compaction" and comp[0]["gap_s"] == 4.0
    assert comp[0]["request_s"] == 30.0, "the compaction request's own length is kept"
    assert rep["overall"]["compaction_gaps"] == [4.0]
    assert rep["overall"]["compaction_requests_s"] == [30.0]
    assert not any(g["compaction"] for g in a)


def test_a_turn_end_and_an_unmatched_step_have_their_own_kind():
    tmp = tempfile.mkdtemp(prefix="idle-")
    rows = [("ses_c", 0, 1000, [], False), ("ses_c", 61_000, 1000, [("read", None)], False),
            ("ses_c", 62_500, 1000, [], False)]
    path = os.path.join(tmp, "l.sqlite3")
    make_ledger(path, rows)
    steps = make_steps(rows)
    steps[1]["created"] += 9_000                      # no ledger row near it: unknown
    rep = run(path, steps)
    kinds = [g["kind"] for g in rep["gaps"]["ses_c"]]
    assert kinds == ["turn_end"], kinds                # the unmatched step's row is in no session
    assert rep["sessions"]["ses_c"]["steps_unmatched"] == 1


# ------------------------------------------------------------------ row 1: the report

def test_gaps_per_tool_kind_in_the_report():
    tmp = tempfile.mkdtemp(prefix="idle-")
    path, steps = planted(tmp)
    sp = os.path.join(tmp, "steps.jsonl")
    with open(sp, "w") as f:
        for s in steps:
            f.write(json.dumps(s) + "\n")
    js, md = os.path.join(tmp, "r.json"), os.path.join(tmp, "r.md")
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert idle_gaps.main(["report", "--ledger", path, "--steps", sp, "--group",
                               "planted=ses_a,ses_b", "--json", js, "--md", md]) == 0
    out = buf.getvalue()
    assert out == open(md).read() + "\n"
    for want in ("| planted | 2 | 10 | 8 |", "0-0.5 s", "over 60 s", "| 1 / 0 | 2 / 3 | 2 / 8 | "
                 "2 / 24 | 1 / 45 | 0 / 0 |", "| browser | 1 | 45.00 | 45.00 | 45.00 | 45.00 | 45 | 1: 44.0 / 44.0 s |",
                 "gaps > 5 s per h", "Compaction requests", "after 5 s", "note writer 20 s"):
        assert want in out, (want, out)
    rep = json.load(open(js))
    hours = rep["groups"]["planted"]["per_clock_hour"]
    assert sum(h["gaps"] for h in hours.values()) == 8
    assert sum(h["over_5s"] for h in hours.values()) == 3
    assert sum(h["over_20s"] for h in hours.values()) == 1
    g = rep["groups"]["planted"]
    wall_h = sum(s["wall_s"] for s in rep["sessions"].values()) / 3600
    assert g["over_5s_per_session_hour"] == round(3 / wall_h, 1)
    assert g["idle_min_per_session_hour"] == round(80.3 / 60 / wall_h, 2)
    # a gap that has lasted 5 s: the 12 s gaps have 7 more, the 45 s gap 40 more
    assert g["after"]["5"]["gaps"] == 3 and g["after"]["5"]["p50_more_s"] == 7.0
    assert g["fits"]["note writer 20 s"]["gaps"] == 1 and g["fits"]["decider"]["gaps"] == 8
    # per kind, once the gap has lasted 1 s: the browser gap (45 s) has 44 more
    assert g["per_kind"]["browser"]["after_1s"] == {"gaps": 1, "p50_more_s": 44.0,
                                                   "p10_more_s": 44.0}
    assert g["per_kind"]["read"]["after_1s"]["gaps"] == 0


# ------------------------------------------------------------------ read-only, other sessions

def test_the_ledger_is_read_only():
    tmp = tempfile.mkdtemp(prefix="idle-")
    path, steps = planted(tmp)
    before = os.stat(path)
    listing = sorted(os.listdir(tmp))
    run(path, steps)
    after = os.stat(path)
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)
    assert sorted(os.listdir(tmp)) == listing, "no -wal or -shm next to the copy"
    try:
        sqlite3.connect(f"file:{path}?mode=ro", uri=True).execute(
            "INSERT INTO meta(key, value) VALUES ('x', 'y')")
        raise AssertionError("a read-only connection wrote")
    except sqlite3.OperationalError:
        pass
    old = idle_gaps.PRODUCTION
    idle_gaps.PRODUCTION = path
    try:
        idle_gaps.read_ledger(path, 0, 1)
        raise AssertionError("the live ledger's path must be refused")
    except SystemExit as exc:
        assert "copy" in str(exc)
    finally:
        idle_gaps.PRODUCTION = old


def test_another_session_inside_a_gap_is_busy_not_idle():
    tmp = tempfile.mkdtemp(prefix="idle-")
    path = os.path.join(tmp, "l.sqlite3")
    # ses_a's 45 s gap runs from 27.8 s to 72.8 s; a 10 s request of another client sits in it
    make_ledger(path, A + [("other", 40_000, 10_000, [], False)])
    rep = run(path, make_steps(A))
    g45 = next(g for g in rep["gaps"]["ses_a"] if g["gap_s"] == 45.0)
    assert (g45["busy_other_s"], g45["idle_s"]) == (10.0, 35.0), g45
    assert rep["overall"]["busy_other_s"] == 10.0
    assert rep["overall"]["requests"] == 6, "the other request belongs to no session here"


def test_without_steps_the_sessions_are_the_conductors_keys():
    tmp = tempfile.mkdtemp(prefix="idle-")
    path = os.path.join(tmp, "l.sqlite3")
    make_ledger(path, A + B, events=[(f"req-ses_a-{i}", "c:ses_a") for i in range(len(A))])
    rep = run(path, [])
    assert set(rep["sessions"]) == {"c:ses_a", "client:k1"}
    assert {g["kind"] for g in rep["gaps"]["c:ses_a"]} == {"unknown"}
    assert sorted(g["gap_s"] for g in rep["gaps"]["c:ses_a"]) == [0.3, 1.5, 4.0, 12.0, 45.0]


def test_the_steps_reader_keeps_names_and_times_only():
    tmp = tempfile.mkdtemp(prefix="idle-")
    db = os.path.join(tmp, "opencode.db")
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE session (id TEXT PRIMARY KEY, title TEXT, directory TEXT);
        CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER,
                              data TEXT);
        CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT);""")
    conn.execute("INSERT INTO session VALUES ('ses_x', 'minicalc', '/tmp/x')")
    msgs = [("m1", {"role": "user", "time": {"created": 1000}}),
            ("m2", {"role": "assistant", "time": {"created": 1010, "completed": 3000},
                    "finish": "tool-calls"}),
            ("m3", {"role": "assistant", "mode": "compaction", "summary": True,
                    "time": {"created": 4000, "completed": 9000}, "finish": "stop"})]
    for mid, d in msgs:
        conn.execute("INSERT INTO message VALUES (?, 'ses_x', ?, ?)",
                     (mid, d["time"]["created"], json.dumps(d)))
    secret = "python3 check.py --token SECRET-PLAN"
    parts = [("p1", "m2", {"type": "tool", "tool": "bash", "state": {
        "status": "completed", "input": {"command": secret},
        "output": "calccheck minicalc\n  [vars] oops\nRESULT: FAIL <- (a) vars failed 1 > 0",
        "time": {"start": 3010, "end": 3500}}}),
        ("p2", "m2", {"type": "tool", "tool": "read", "state": {
            "status": "completed", "input": {"filePath": "/tmp/x/SPEC.md"},
            "output": "the spec text", "time": {"start": 3001, "end": 3002}}}),
        ("p3", "m2", {"type": "text", "text": "I will run the checker"})]
    for pid, mid, d in parts:
        conn.execute("INSERT INTO part VALUES (?, ?, 'ses_x', ?)", (pid, mid, json.dumps(d)))
    conn.commit()
    conn.close()
    before = os.stat(db)
    out = os.path.join(tmp, "steps.jsonl")
    with redirect_stdout(io.StringIO()):
        assert idle_gaps.main(["steps", "--opencode-db", db, "--session", "ses_x",
                               "--out", out]) == 0
    assert os.stat(db).st_mtime_ns == before.st_mtime_ns
    raw = open(out).read()
    for leak in ("SECRET", "spec text", "RESULT", "SPEC.md", "I will run"):
        assert leak not in raw, leak
    steps = [json.loads(x) for x in raw.splitlines()]
    assert [s["id"] for s in steps] == ["m2", "m3"], "assistant messages only"
    s2, s3 = steps
    assert [t["tool"] for t in s2["tools"]] == ["bash", "read"]
    assert s2["tools"][0]["checker"] == "calccheck" and s2["tools"][1]["checker"] is None
    assert idle_gaps.step_kind(s2) == "checker", "the longest tool decides; a checker run"
    assert s3["compaction"] and idle_gaps.step_kind(s3) == "compaction"


def test_tool_kinds():
    k = idle_gaps.tool_kind
    assert k({"tool": "multiedit"}) == "edit" and k({"tool": "write"}) == "write"
    assert k({"tool": "chrome-devtools_take_screenshot"}) == "screenshot"
    assert k({"tool": "chrome-devtools_evaluate_script"}) == "browser"
    assert k({"tool": "webfetch"}) == "browser" and k({"tool": "todowrite"}) == "other"
    assert k({"tool": "chrome-devtools_evaluate_script", "checker": "pagodacheck"}) == "checker"
    assert idle_gaps.step_kind(None) == "unknown"
    assert idle_gaps.step_kind({"tools": []}) == "turn_end"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
