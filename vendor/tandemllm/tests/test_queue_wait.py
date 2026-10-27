"""2026-10-05: a turn that waits for the engine is told so and never refused after minutes.

On 2026-10-04 a second opencode session's turn waited behind a running turn and got a 429 after
`--queue-timeout` (240 s). Here:

  1  the estimate (server/queue_wait.py): no basis below MIN_SAMPLES; the holder's remaining time
     is the median of what requests that ran as long went on for; a holder longer than nearly all
     has none; waiters ahead add their rank's median; the prepare and shadow jobs add nothing;
     the rounding of the line and the comment's text
  2  through the handler with the fake engine: a streamed turn behind a running one waits past
     `--queue-timeout`, its stream opens early and carries `: waiting, position 1` comments, and
     it is served (no 429); a request without a stream behind the same turn still gets the 429
  3  with the profile's progress lines on and hold times recorded, the waiting line carries the
     estimate ("waiting: another request is running, position 1, about 10 s")
  4  the clock records each request's hold time by rank
  5  the waiting line names TandemLLM's own work when that holds the engine (the conductor's
     question to Qwen, the shadow reader, the summary written ahead, the drain), not "another
     request"
  7  SRV-249: with the holder's live work (`Live`) the estimate is its prefill still to read plus
     the tokens still to write (the `TOKENS_Q` quantile of what earlier requests of its kind
     wrote past what it has written) at its own rate, capped by its max_tokens; the W18 shape
     (35.8 s behind a thinking turn 14.8 s in) is no longer read as 20 s; no basis falls back
     to the hold times; the handler's `_live_of` reads a record in both phases

Run: python tests/test_queue_wait.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import queue_wait as qw  # noqa: E402


# ------------------------------------------------------------------ 1: the estimate
def test_1_no_estimate_without_a_basis():
    c = qw.TurnClock()
    for x in (10, 20, 30, 40):
        c.done(1, x)
    assert c.remaining(1, 0.0) is None and c.typical(1) is None
    assert c.estimate((1, 0.0), []) is None
    assert c.estimate(None, []) == 0.0          # nobody holds it and nobody waits ahead


def test_1_the_holder_s_remaining_time_is_conditional_on_how_long_it_ran():
    c = qw.TurnClock()
    for x in (2, 3, 4, 5, 60, 120, 300):
        c.done(1, x)
    # at 0 s: the median of all 7
    assert c.remaining(1, 0.0) == 5.0
    # 50 s in: only 60, 120 and 300 ran that long; they went on for 10, 70 and 250 s
    assert c.remaining(1, 50.0) == 70.0
    # 200 s in: one recorded time ran longer, too few to say
    assert c.remaining(1, 200.0) is None
    # waiters ahead add their rank's median; a rank that yields adds nothing
    for x in (1, 1, 2, 2, 3):
        c.done(0, x)
    assert c.estimate((1, 50.0), [0, 1]) == 70.0 + 2.0 + 5.0
    # SRV-177: a yielding holder (the prepare or shadow job) has no estimate until waits behind
    # it were measured, then the median of those longer than the wait so far; a yielding waiter
    # ahead still adds nothing (it yields to this request too)
    assert c.estimate((3, 0.5), [4], yields=frozenset((3, 4))) is None
    for x in (0.4, 1.2, 1.8, 2.6, 3.0):
        c.yielded(3, x)
    # the waits longer than 0.5 s: 1.2 1.8 2.6 3.0, less 0.5, median 1.7
    assert abs(c.estimate((3, 0.5), [4], yields=frozenset((3, 4))) - 1.7) < 1e-9
    assert c.estimate((3, 2.0), [], yields=frozenset((3, 4))) is None   # 2 longer: too few
    assert c.estimate((1, 50.0), [2]) is None   # no basis for rank 2


def test_1_the_line_and_the_comment():
    # SRV-177: below 10 s the seconds themselves (every estimate used to round up to "10 s")
    assert [qw.about(s) for s in (0, 4, 14, 26, 89, 95, 400)] == \
        ["1 s", "4 s", "10 s", "30 s", "90 s", "2 min", "7 min"]
    assert qw.comment(2, 41.0, 12.3) == b": waiting, position 2, about 40 s, 12 s so far\n\n"
    assert qw.comment(1, None, 3.0) == b": waiting, position 1, 3 s so far\n\n"


def test_1_the_clock_keeps_a_bounded_window():
    c = qw.TurnClock(keep=10)
    for x in range(100):
        c.done(1, float(x))
    assert c.typical(1) == 94.5                  # the last 10: 90..99


# ------------------------------------------------------------------ 7: the holder's live work
def test_7_the_holder_s_remaining_tokens_at_its_rate():
    c = qw.TurnClock()
    assert c.live_remaining(qw.Live(written=0, rate=30.0, thinking=True)) is None   # no basis
    for n in (300, 800, 1200, 1500, 1500, 1500, 2000):
        c.wrote(True, n, 30.0)
    for n in (20, 40, 60, 80, 100):
        c.wrote(False, n, 40.0)
    # the ticket's case: 1,000 tokens left (the quantile of 1500-500 ... 2000-500) at 30 tok/s
    est = c.live_remaining(qw.Live(written=500, rate=30.0, thinking=True))
    assert est is not None and est >= 1000 / 30.0, est
    # the holder's max_tokens caps what is left
    capped = c.live_remaining(qw.Live(written=500, rate=30.0, thinking=True, max_tokens=800))
    assert abs(capped - 300 / 30.0) < 1e-9, capped
    # by kind: a turn that does not think reads the short ones
    short = c.live_remaining(qw.Live(written=10, rate=40.0, thinking=False))
    assert short is not None and short <= 100 / 40.0, short
    # while it prefills: the prefill left at its rate, then the tokens at the median rate
    pre = c.live_remaining(qw.Live(pf_left=9000, pf_rate=900.0, thinking=True))
    assert pre is not None and pre >= 10.0 + 1500 / 30.0 - 1e-9, pre
    assert c.live_remaining(qw.Live(pf_left=9000, pf_rate=None, thinking=True)) is None
    # past nearly every recorded length: too few longer, no basis (the hold times take over)
    assert c.live_remaining(qw.Live(written=1900, rate=30.0, thinking=True)) is None
    for x in (2, 3, 4, 5, 60, 120, 300):
        c.done(1, x)
    assert c.estimate((1, 50.0), [], live=qw.Live(written=1900, rate=30.0, thinking=True)) \
        == c.remaining(1, 50.0)
    # a yielding holder never reads a live view
    assert c.estimate((3, 0.5), [], yields=frozenset((3,)),
                      live=qw.Live(written=500, rate=30.0, thinking=True)) is None


def test_7_the_w18_shape_is_not_read_as_twenty_seconds():
    """W18: tab B waited 35.8 s behind a thinking turn (1,479 tokens, 48 s of
    decode at ~30 tok/s) that had run 14.8 s (1.5 s of prefill, ~400 tokens written). The hold
    times said "about 20 s"."""
    c = qw.TurnClock()
    # thinking chat turns of a session like UX2's: a few hundred to a few thousand tokens
    for n in (52, 180, 420, 650, 900, 1200, 1479, 1800, 2600, 3400):
        c.wrote(True, n, 30.0)
    est = c.live_remaining(qw.Live(written=400, rate=30.6, thinking=True, max_tokens=32768))
    assert est is not None and est >= 30.0, est
    assert qw.about(est) not in ("20 s", "10 s"), qw.about(est)


def test_7_the_handler_reads_the_holder_s_record():
    from server import app
    from server.usage import RequestRecord as Record
    r = Record("chatcmpl-x", "chat", True)
    r.thinking, r.max_tokens = True, 4000
    assert app._live_of(r) is None                     # before its first prefill chunk
    t = time.perf_counter()
    r.prefill_info = {"start": 0, "t0": t - 4.0, "chunk": 1024}
    r.pf_marks = [(1024, 8192, t - 3.0), (2048, 8192, t - 2.0), (3072, 8192, t - 1.0)]
    live = app._live_of(r)
    assert live.pf_left == 8192 - 3072 and live.pf_rate and 900 <= live.pf_rate <= 1100, \
        (live.pf_left, live.pf_rate)
    r.t_first, r.n_live = t - 10.0, 300
    live = app._live_of(r)
    assert live.pf_left == 0 and live.written == 300 and 29.0 <= live.rate <= 31.0, live.rate
    assert live.thinking and live.max_tokens == 4000
    app.CLOCK = qw.TurnClock()
    r.t_last = r.t_first + 9.0
    app._clock_wrote(r)
    assert app.CLOCK._wrote[True][-1] == 300 and abs(app.CLOCK._rates[-1] - 299 / 9.0) < 1e-9


# ------------------------------------------------------------------ 2-4: the handler
def _serve(ui):
    import test_side_requests as sr
    cond, order = sr.serve(ui)
    from server import app
    app.STATE["queue_timeout"] = 1.0
    app.STATE["request_timeout"] = 60.0
    app.CLOCK = qw.TurnClock()
    return app, sr, order


def _patched(open_after=0.3, ping=0.3):
    old = (qw.OPEN_AFTER_S, qw.PING_EVERY_S)
    qw.OPEN_AFTER_S, qw.PING_EVERY_S = open_after, ping
    return old


def test_2_a_streamed_turn_waits_past_the_queue_timeout_and_is_served():
    app, sr, order = _serve(None)
    import test_ui_states as tu
    old = _patched()
    try:
        with sr.threads_quiet():
            # FAKE_SLOW: ten tokens a second, 30 tokens: the engine is held about 3 s
            busy, _ = sr._bg(sr.tu_body("FAKE_SLOW one", tu.TOOLS, 30), {"x-session-id": "ses_a"})
            sr._wait_until(lambda: len(order) == 1)
            turn, box = sr._bg(sr.tu_body("the waiting turn", tu.TOOLS, 200),
                               {"x-session-id": "ses_b"})
            sr._wait_until(lambda: app.INFLIGHT["waiting"] == 1)
            pbox = {}
            plain = threading.Thread(target=lambda: pbox.update(out=tu.Req(
                dict(sr.tu_body("not streamed", tu.TOOLS, 50), stream=False),
                {"User-Agent": sr.OPENCODE_UA, "x-session-id": "ses_c"}).response()), daemon=True)
            plain.start()
            for t in (busy, turn, plain):
                t.join(60)
    finally:
        qw.OPEN_AFTER_S, qw.PING_EVERY_S = old
    out = box["out"]
    assert " 200 " in out["head"].split("\r\n")[0], out["head"][:80]
    assert ": waiting, position 1" in out["raw"], out["raw"][:300]
    assert out["raw"].count(": waiting, position") >= 2, out["raw"][:400]   # one a slice (1 s)
    assert out["usage"] and out["usage"]["completion_tokens"] > 0, out["raw"][-300:]
    assert any("the waiting turn" in o[1] for o in order)
    assert '"finish_reason": "error"' not in out["raw"]
    # without a stream: the old bound, a 429 after `--queue-timeout`
    head, raw = pbox["out"]
    assert " 429 " in head.split("\r\n")[0] and "(1 s)" in raw, (head[:80], raw[:200])
    assert not any("not streamed" in o[1] for o in order)


def test_3_the_waiting_line_carries_the_estimate():
    app, sr, order = _serve({"states": "on", "states_channel": "content"})
    import test_ui_states as tu
    from server.conductor import states
    for x in (2.0, 2.5, 3.0, 4.0, 20.0, 30.0, 40.0):
        app.CLOCK.done(app.RANK_TURN, x)
    old = _patched(open_after=30.0)              # no comment before the line
    try:
        with sr.threads_quiet():
            busy, _ = sr._bg(sr.tu_body("FAKE_SLOW one", tu.TOOLS, 30), {"x-session-id": "ses_a"})
            sr._wait_until(lambda: len(order) == 1)
            turn, box = sr._bg(sr.tu_body("the waiting turn", tu.TOOLS, 200),
                               {"x-session-id": "ses_b"})
            for t in (busy, turn):
                t.join(60)
    finally:
        qw.OPEN_AFTER_S, qw.PING_EVERY_S = old
    got = [ln[len(states.MARKER):] for ln in box["out"]["content"].split("\n")
           if ln.startswith(states.MARKER)]
    # SRV-177: no line in the first 2 s of the wait; then the median of the recorded times
    # longer than the holder has run (about 2 s: 2.5 3 4 20 30 40 -> 12 s) less what it has run
    assert got[0].startswith("waiting: another request is running, position 1, about "), got
    assert got[0].endswith(" s"), got


def test_4_the_clock_records_each_hold_by_rank():
    app, sr, order = _serve(None)
    import test_ui_states as tu
    with sr.threads_quiet():
        sr.go(sr.tu_body("one", tu.TOOLS, 20))
        sr.go(sr.tu_body("two", tu.TOOLS, 20))
    assert len(app.CLOCK._samples(app.RANK_TURN)) == 2
    assert all(0 <= x < 10 for x in app.CLOCK._samples(app.RANK_TURN))


def test_5_the_waiting_line_names_tandemllm_s_own_work():
    """NP1's night pass (2026-10-04, open issue 2): a turn's first line read "waiting for another
    request to finish" while the request that held the engine was the conductor's own question
    to Qwen. The line names what holds the engine when it is TandemLLM's own work."""
    app, sr, order = _serve({"states": "on", "states_channel": "content"})
    import test_ui_states as tu
    from server.conductor import classify, states, voice
    old = _patched(open_after=30.0)
    try:
        with sr.threads_quiet():
            busy, _ = sr._bg(sr.tu_body("FAKE_SLOW one", tu.TOOLS, 30),
                             {"x-session-id": "ses_a", "User-Agent": classify.USER_AGENT})
            sr._wait_until(lambda: len(order) == 1)
            turn, box = sr._bg(sr.tu_body("the waiting turn", tu.TOOLS, 200),
                               {"x-session-id": "ses_b"})
            for t in (busy, turn):
                t.join(60)
    finally:
        qw.OPEN_AFTER_S, qw.PING_EVERY_S = old
    got = [ln[len(states.MARKER):] for ln in box["out"]["content"].split("\n")
           if ln.startswith(states.MARKER)]
    assert got[0] == "waiting: TandemLLM decides how to handle the request first, position 1", got
    assert "shadow" not in " ".join(app.OWN_WAIT.values()), "SRV-177: no internal role names"
    # the holder kinds without a request: the shadow's burst, the summary written ahead, a lock
    # taken by nobody's request (the drain, a cache clear); a client's request stays "another"
    me = {"rank": app.RANK_TURN, "t": time.monotonic(), "side": None}
    try:
        for who, want in (({"rank": app.RANK_SHADOW, "t": 0.0, "side": None}, "shadow"),
                          ({"rank": app.RANK_PREP, "t": 0.0, "side": None}, "prepare"),
                          ({"rank": app.RANK_TURN, "t": 0.0, "side": None, "own": True},
                           "question")):
            app.HOLDER["who"] = who
            assert app._wait_reason(me) == (app.OWN_WAIT[want], 1), want
        app.HOLDER["who"] = {"rank": app.RANK_TURN, "t": 0.0, "side": None, "own": False}
        assert app._wait_reason(me) == ("another request is running", 1)
        app.HOLDER["who"] = None
        with app.LOCK:
            assert app._wait_reason(me) == (app.OWN_WAIT["engine"], 1)
        assert app._wait_reason(me) == ("another request is running", 1)
    finally:
        app.HOLDER["who"] = None
    # the voice says each in words, in English and German
    for reason in app.OWN_WAIT.values():
        en = voice.natural("queue", {"reason": reason, "n": 1}, "en")
        assert en.startswith("waiting while ") and " first," not in en, en
        assert reason in voice.OWN_WAIT_DE
        de = voice.natural("queue", {"reason": reason, "n": 1}, "de")
        assert de == f"wartet: {voice.OWN_WAIT_DE[reason]}, Platz 1", de


def test_6_a_short_wait_shows_no_line_and_own_work_is_measured():
    """SRV-177 (UT1, 2026-10-06): "waiting while the shadow model reads the conversation, 1st in
    line, about 10 s left" was the line 9 times of 9, while the ledger's waits were 0.1 to 3 s.
    A request that waits less than `QUIET_S` gets no line; one that waited behind TandemLLM's own
    work records how long (`wait_own_ms`, the ledger), and those waits are the estimate's basis."""
    app, sr, order = _serve({"states": "on", "states_channel": "content"})
    import test_ui_states as tu
    from server.conductor import states
    old = _patched(open_after=30.0)
    try:
        with sr.threads_quiet():
            # FAKE_SLOW with 8 tokens: the engine is held under a second
            busy, _ = sr._bg(sr.tu_body("FAKE_SLOW one", tu.TOOLS, 8), {"x-session-id": "ses_a"})
            sr._wait_until(lambda: len(order) == 1)
            turn, box = sr._bg(sr.tu_body("the waiting turn", tu.TOOLS, 50),
                               {"x-session-id": "ses_b"})
            for t in (busy, turn):
                t.join(60)
    finally:
        qw.OPEN_AFTER_S, qw.PING_EVERY_S = old
    got = [ln for ln in box["out"]["content"].split("\n") if ln.startswith(states.MARKER)
           and "waiting" in ln]
    assert got == [], got
    # the time behind TandemLLM's own work: the holder kinds, and the record's field
    try:
        for who, want in (({"rank": app.RANK_SHADOW, "t": 0.0, "side": None}, app.RANK_SHADOW),
                          ({"rank": app.RANK_PREP, "t": 0.0, "side": None}, app.RANK_PREP),
                          ({"rank": app.RANK_TURN, "t": 0.0, "side": None, "own": True}, -1),
                          ({"rank": app.RANK_TURN, "t": 0.0, "side": None}, None)):
            app.HOLDER["who"] = who
            assert app._own_holder() == want, (who, want)
    finally:
        app.HOLDER["who"] = None
    from server import ledger, usage
    assert "wait_own_ms" in ledger.COLUMNS and "wait_own_ms" in usage.RequestRecord(
        "x", "chat", False).row()


def _needs_torch() -> bool:
    try:
        import torch  # noqa: F401
        return False
    except ImportError:
        return True


if __name__ == "__main__":
    passed = skipped = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if not name.startswith("test_1") and _needs_torch():
                print(f"  {name:72s} skipped (no torch)")
                skipped += 1
                continue
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed" + (f", {skipped} skipped" if skipped else ""))
    sys.exit(0)
