"""The sessions page and the Overview tiles read what they answer with (deep audit
2026-10-16, P2).

Both conductor reads used to pull the window whole: up to 100k of the router's rows
with their JSON and 200k of the monitor's — every hidden text included — 300k
materialised dicts per request, for a page of 200 lines and a set of counts. Now the
sessions page first groups the session keys (`recent_sessions`, a GROUP BY of two
columns) and reads only those rows, and neither read attaches a hidden text the
response never shows.

  1  the page says exactly what the old global read said            test_the_page_says_*
  2  a page of 200 reads 200 sessions, not the week's 260            test_the_page_reads_*
  3  no hidden text is read for the page or the tiles               ...no hidden text*
  4  the null session keys as one group ("-")                       test_the_null_*

Run: python tests/test_conductor_page_reads.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor import decisions, routes  # noqa: E402

NOW = int(time.time() * 1000)
WEEK = int((time.time() - 7 * 86400) * 1000)


def seed(jr, key, i, *, kind="route", text=None, ts=None):
    ts = ts if ts is not None else NOW - 1000 * i
    if kind == "monitor":
        jr.monitor_sink({"decision": "CONTINUE", "mode": "act", "task": "S4", "session": key,
                         "request_id": f"req-{key}-{i}", "ts": ts / 1000.0, "text": text,
                         "hidden_tokens": 42, "client": "cli", "score": 0.9,
                         "detail": {"why": "idle"}, "profile": "default"})
    elif kind == "side":
        r = {"id": f"side-{key}-{i}", "ts_ms": ts, "kind": "side", "task": "route",
             "session_key": key, "request_id": f"req-{key}-{i}", "client_kind": "cli"}
        jr.ring[r["id"]] = r          # ring only: a side row is no page line
    else:
        r = {"id": f"row-{key}-{i}", "ts_ms": ts, "session_key": key,
             "request_id": f"req-{key}-{i}", "client_kind": "cli", "profile": "default",
             "class_id": "chat", "source": "rules"}
        if text:
            r["injected"] = [{"text": text, "tokens": 12}]
        jr.open_row(r)
        jr.close_row(r)


def make_journal(n_sessions=6, per=3, *, with_monitor=True, with_text=True, extra=None):
    jr = decisions.Journal(os.path.join(tempfile.mkdtemp(prefix="cond-page-"),
                                        "decisions.sqlite3"))
    for s in range(n_sessions):
        key = f"ses_{s:03d}"
        for i in range(per):
            seed(jr, key, s * 100 + i)
            if with_monitor:
                seed(jr, key, s * 100 + i, kind="monitor",
                     text="the hidden words" if with_text else None)
    for what in extra or ():
        what(jr)
    jr.flush(5.0)
    return jr


class FakeRouter:
    def sessions_live(self):
        return []


class FakeConductor:
    def __init__(self, jr):
        self.store = None
        self.journal = jr
        self.state = {"monitor": object()}
        self.router = FakeRouter()

    def roles_health(self):
        return {}


def page(jr):
    return routes._route(None, FakeConductor(jr), "GET", ["sessions"], {}, {}, None)


def old_page(jr):
    """The page as the global reads computed it (routes.py at v0.11.0)."""
    return decisions.sessions(jr.rows(since_ms=WEEK, limit=100000),
                              jr.rows(since_ms=WEEK, limit=200000, kind="monitor"))[:200]


def texts_spy(jr):
    """(stop, n): count every monitor row a hidden text was looked up for."""
    seen = [0]
    orig = jr._texts

    def wrapped(rows):
        seen[0] += sum(1 for r in rows
                       if r.get("kind") == "monitor" and r.get("hidden_text_id"))
        return orig(rows)

    jr._texts = wrapped
    return lambda: seen[0]


def test_the_page_says_what_the_old_read_said():
    jr = make_journal()
    status, out, _ = page(jr)
    assert status == 200
    assert out["sessions"] == old_page(jr), "the page moved"
    assert [s["key"] for s in out["sessions"]][0] == "ses_000"   # newest first
    assert out["sessions"][0]["turns"] == 3 and out["sessions"][0]["acts"] == 3


def test_the_page_reads_its_own_sessions_only():
    jr = make_journal(n_sessions=260, per=1)      # 260 lines of history, a page of 200
    n = [0]
    orig = jr.rows

    def counted(**kw):
        got = orig(**kw)
        n[0] += len(got)
        return got

    jr.rows = counted
    _, out, _ = page(jr)
    new_read = n[0]
    jr.rows = orig                              # unspied: the expected page reads too
    assert out["sessions"] == old_page(jr)[:200], "the page moved"
    assert len(out["sessions"]) == 200
    assert new_read == 400, f"the page read {new_read} rows, not its own 200 + 200"


def test_no_hidden_text_is_read_for_the_page():
    jr = make_journal(n_sessions=260, per=1)
    counted = texts_spy(jr)
    page(jr)
    assert counted() == 0, "a page of 200 sessions read hidden texts for the week"


def test_the_null_session_keys_as_one_group():
    def null_group(jr):                    # an open row with no session: the "-" line
        r = {"id": "row-null", "ts_ms": NOW - 500, "session_key": None,
             "request_id": "req-null", "client_kind": "cli", "class_id": "chat"}
        jr.ring[r["id"]] = r

    jr = make_journal(n_sessions=3, extra=[null_group])
    _, out, _ = page(jr)
    assert out["sessions"] == old_page(jr)
    assert "-" in [s["key"] for s in out["sessions"]]


def test_a_side_row_is_no_page_line():
    def a_side(jr):
        seed(jr, "ses_s1", 0, kind="side")     # S1 only: the page never saw these

    jr = make_journal(n_sessions=3, extra=[a_side])
    _, out, _ = page(jr)
    assert out["sessions"] == old_page(jr)     # identity again, side rows excluded
    assert "ses_s1" not in [s["key"] for s in out["sessions"]]


def test_the_tiles_answer_the_same_without_the_texts():
    jr = make_journal()
    rows = jr.rows(since_ms=0, limit=100000)
    evs = jr.rows(since_ms=0, limit=200000, kind="monitor")          # texts attached
    expected = decisions.stats(rows, hours=24, events=evs)
    counted = texts_spy(jr)
    _, out, _ = routes._route(None, FakeConductor(jr), "GET", ["stats"],
                              {"hours": "24"}, {}, None)
    got = dict(out)
    got.pop("roles")
    # the ribbon's buckets are stamped from each call's own `now`: compare the counts,
    # not the millisecond each call drew its window at
    for hourly in (expected["hourly"], got["hourly"],
                   expected["monitor"]["hourly"], got["monitor"]["hourly"]):
        for bucket in hourly:
            bucket.pop("start_ms")
    assert got == expected, "the tiles moved"
    assert counted() == 0, "the tiles read hidden texts they never show"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
