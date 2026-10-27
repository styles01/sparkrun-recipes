"""One bad query number is a 400, not a dead connection (deep audit 2026-10-16, P2).

The conductor's live-feed routes (events, decisions, stats) read `since`, `limit`, `before`
and `hours` straight through `int()`. A value that is not a number raised there, and an
exception out of the route answers nothing at all — the connection dies mid-request and
every retry spams a traceback (the sibling routes — versions/diff, transcripts, memory
search, packages/usage — already answer 400 with the param's name; these three missed).

  1  a non-numeric since/limit/before/hours is a 400 naming the param    test_a_bad_*
  2  a good value still parses and clamps as before                      test_a_good_*

Run: python tests/test_conductor_int_params.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor import routes  # noqa: E402


class FakeJournal:
    def __init__(self):
        self.calls = []

    def rows(self, **kw):
        self.calls.append(kw)
        return []

    def with_events(self, rows, events=None):
        return list(rows)


class FakeConductor:
    def __init__(self):
        self.store = None
        self.journal = FakeJournal()
        self.state = {}
        self.router = type("R", (), {"sessions_live": lambda s: []})()

    def roles_health(self):
        return {}


def _route(parts, q, method="GET"):
    return routes._route(None, FakeConductor(), method, parts, q, {}, None)


def _bad(parts, **q):
    try:
        _route(parts, q)
    except routes.ApiError as exc:
        return exc
    raise AssertionError(f"{parts} {q}: expected a 400, got a response")


def test_a_bad_since_is_400_on_both_feeds():
    for parts in (["events"], ["decisions"]):
        exc = _bad(parts, since="yesterday")
        assert exc.status == 400 and "since" in str(exc), exc


def test_a_bad_limit_is_400_on_both_feeds():
    for parts in (["events"], ["decisions"]):
        exc = _bad(parts, limit="many")
        assert exc.status == 400 and "limit" in str(exc), exc


def test_a_bad_before_is_400_on_both_feeds():
    for parts in (["events"], ["decisions"]):
        exc = _bad(parts, before="an-hour-ago")
        assert exc.status == 400 and "before" in str(exc), exc


def test_a_bad_hours_is_400():
    exc = _bad(["stats"], hours="two")
    assert exc.status == 400 and "hours" in str(exc), exc


def test_a_good_value_still_parses():
    cond = FakeConductor()
    routes._route(None, cond, "GET", ["events"], {"since": "123", "limit": "50",
                                                  "before": "456"}, {}, None)
    call = cond.journal.calls[0]
    assert call["since_ms"] == 123 and call["limit"] == 50 and call["before_ms"] == 456, call


def test_a_default_and_a_clamp_stand():
    cond = FakeConductor()
    routes._route(None, cond, "GET", ["events"], {}, {}, None)
    assert cond.journal.calls[0]["since_ms"] == 0
    assert cond.journal.calls[0]["limit"] == 100  # the default, not yet the 2000 cap
    assert cond.journal.calls[0]["before_ms"] is None
    status, out, _ = routes._route(None, cond, "GET", ["stats"], {"hours": "99999"}, {}, None)
    assert status == 200 and out["hours"] == 168, out


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
