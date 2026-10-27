"""Parallel requests through the real handler, on the fake engine (`server/app.py --fake-engine
--parallel N`): no model, no GPU, every line of the HTTP path the real.

  * N streams at once: three slow requests run together on slot 0 and two side slots (they
    overlap in time, the wall clock is well under three times one request), each answer is the
    answer its prompt gets alone, and /health counts the side grants and the three running;
  * a fourth request waits for a slot and is served when one frees;
  * the admission counts bytes: with MemAvailable just over the floor, the second request waits
    for the first one's reservation and runs after it;
  * the Parallel view shows each engine lane's slot and the engine's slots;
  * without --parallel nothing of this is there: /health has no `parallel`, and two requests run
    one after the other, as always.

Run: python tests/test_parallel_app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fake_engine import _get, _port, _start, _wait  # noqa: E402

TEXTS = ["FAKE_SLOW Name three rivers.", "FAKE_SLOW Name three mountains.",
         "FAKE_SLOW Name three lakes.", "FAKE_SLOW Name three deserts."]


def _chat(port, text, max_tokens=24):
    """A streamed chat: (the answer, its first token's time, its end). The span between the two
    is when the engine served it; a request that waited in the queue starts late."""
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    parts, t_first = [], None
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            for c in d.get("choices") or []:
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    if t_first is None:
                        t_first = time.monotonic()
                    parts.append(piece)
    return "".join(parts), t_first, time.monotonic()


def _together(port, texts):
    res: dict = {}

    def go(i):
        res[i] = _chat(port, texts[i])

    ths = [threading.Thread(target=go, args=(i,)) for i in range(len(texts))]
    t0 = time.monotonic()
    for th in ths:
        th.start()
        time.sleep(0.05)
    for th in ths:
        th.join(120)
    return [res[i] for i in range(len(texts))], time.monotonic() - t0


def _overlap(spans) -> int:
    """The most requests in flight at one moment."""
    ev = sorted([(a, 1) for _, a, _ in spans] + [(b, -1) for _, _, b in spans])
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def _server(extra=(), env=None):
    port = _port()
    ledger = os.path.join(tempfile.mkdtemp(prefix="qse-par-"), "ledger.sqlite3")
    # the side lane off: these short raw requests would otherwise be served as lane guests at the
    # holder's safe points (server/preempt.py), which is a different overlap
    proc = _start(port, ledger, extra=("--side-lane", "off", *extra), env_extra=env)
    _wait(port, proc)
    return port, proc


def _stop(proc):
    proc.terminate()
    try:
        proc.wait(20)
    except Exception:                                              # noqa: BLE001
        proc.kill()


def test_three_streams_at_once_on_three_slots():
    port, proc = _server(("--parallel", "3", "--side-max-len", "8192"))
    try:
        alone = [_chat(port, t)[0] for t in TEXTS[:3]]
        t_one = max(b - a for _, a, b in (_chat(port, TEXTS[0]),))
        spans, wall = _together(port, TEXTS[:3])
        assert [s[0] for s in spans] == alone, "each answer is the one its prompt gets alone"
        assert _overlap(spans) == 3, "the three ran at once"
        assert wall < 2.0 * t_one, f"three at once took {wall:.1f} s, one alone {t_one:.1f} s"
        h = _get(port, "/health")["parallel"]
        assert h["n"] == 3 and h["side_grants"] >= 2 and h["max_running"] == 3, h
        assert all(s["phase"] == "" and not s["busy"] for s in h["slots"]), h["slots"]
        # a fourth waits for a slot
        spans, _ = _together(port, TEXTS)
        assert [s[0] for s in spans[:3]] == alone
        assert _overlap(spans) == 3, "at most three at once on three slots"
    finally:
        _stop(proc)


def test_the_parallel_view_shows_the_slots():
    port, proc = _server(("--parallel", "2", "--side-max-len", "8192"))
    try:
        seen: list = []
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                try:
                    d = _get(port, "/v1/dashboard/parallel")
                    seen.append(d)
                except Exception:                                  # noqa: BLE001
                    pass
                time.sleep(0.2)

        th = threading.Thread(target=watch)
        th.start()
        _together(port, TEXTS[:2])
        stop.set()
        th.join()
        both = [d for d in seen if len(d.get("lanes") or []) >= 2]
        assert both, "the view saw both requests in flight"
        assert sorted(l["slot"] for l in both[-1]["lanes"]) == [0, 1], both[-1]["lanes"]
        es = seen[-1]["engine_slots"]
        assert es["n"] == 2 and [s["window"] for s in es["slots"]][1] == 8192, es
    finally:
        _stop(proc)


def test_admission_holds_a_request_until_the_bytes_are_there():
    # 15 GiB floor, MemAvailable 15 GiB + 1 MiB: a second request's worst case cannot fit beside
    # the first one's reservation, so it waits for it (a lone request is always admitted)
    port, proc = _server(("--parallel", "2", "--side-max-len", "8192"),
                         env={"FAKE_MEMAVAILABLE_GIB": str(15 + 1 / 1024)})
    try:
        spans, _ = _together(port, TEXTS[:2])
        assert _overlap(spans) == 1, "the second request waited for the first one's bytes"
        h = _get(port, "/health")["parallel"]
        assert h["admission"]["waits"] >= 1 and h["admission"]["refused"] == 0, h["admission"]
        assert h["admission"]["in_flight"] == 0, "every reservation released"
    finally:
        _stop(proc)


def test_without_parallel_one_at_a_time_as_always():
    port, proc = _server()
    try:
        assert "parallel" not in _get(port, "/health")
        spans, _ = _together(port, TEXTS[:2])
        assert _overlap(spans) == 1
    finally:
        _stop(proc)


def test_side_graphs_off_builds_no_graph_pool_and_estimates_less():
    """PL1 (2026-10-06): `--side-graphs off` makes a side slot's engine verify eager (no graph pool,
    the slot's 7.5 GiB constant mostly gone) and the build's estimate follows; slot 0 is untouched.
    QSE_SIDE_SLOT_CONST_GB overrides the estimate either way."""
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.models.qwen3_5.adapter import Qwen38Engine
    from server import app as A
    e = Qwen38Engine.__new__(Qwen38Engine)
    e._graphs, e.graphs_off = None, True
    assert e._graphs_for(2, 0) is None and e._graphs is None
    ns = argparse.Namespace
    assert A._side_const_gb(ns(side_graphs="on", side_slot_const_gb=0.0)) == 7.5
    assert A._side_const_gb(ns(side_graphs="off", side_slot_const_gb=0.0)) == \
        A.SIDE_CONST_NO_GRAPHS_GB < 7.5
    assert A._side_const_gb(ns(side_graphs="off", side_slot_const_gb=2.25)) == 2.25
    from server import slots as S
    slot = S.Slot(1, 65536, values={"engine": e})
    assert slot.report()["graphs"] is False and S.Slot(0, 262144).report()["graphs"] is None


def test_a_short_resident_conversation_does_not_hold_every_request_on_slot_0():
    """PL1 (2026-10-06, live on the box): after a 450-token raw request, both requests of a c2 pair
    were counted `resident_waits` and decoded one after the other. A request continues slot 0's
    conversation only when it shares the rows within a chunk AND at least one chunk of them."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from server import app as A
    c = A._continues_resident
    assert not c(450, 5, 1024), "a short resident and an unrelated prompt: take the side slot"
    assert not c(450, 450, 1024), "a short conversation continued: cheap to prefill elsewhere"
    assert c(100_000, 99_500, 1024), "a long conversation continued: wait and resume in place"
    assert not c(100_000, 40_000, 1024), "a long shared head but not the conversation"
    assert c(2048, 1100, 1024) and not c(2048, 900, 1024)
    assert not c(0, 0, 1024)


def test_a_window_offers_the_turn_between_layers_only_to_a_waiting_slot():
    """SRV-321: the window's layer yield pauses the turn when a slot waits for it, and does nothing
    alone (no hand-over, no device sync); QSE_PARALLEL_LAYER_YIELD defaults to on."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import types
    from server import app as A
    calls = []
    turn = types.SimpleNamespace(n=0, waiting=lambda: turn.n, pause=lambda who: calls.append(who))
    pool = types.SimpleNamespace(turn=turn)
    old = dict.get(A.STATE, "pool")
    dict.__setitem__(A.STATE, "pool", pool)
    try:
        A._par_window_yield()
        assert calls == [], "alone: no pause"
        turn.n = 2
        A._par_window_yield()
        assert calls == [0], "a slot waits: the turn is offered by this thread's slot"
        pool.turn = None
        A._par_window_yield()
        assert calls == [0]
    finally:
        dict.__setitem__(A.STATE, "pool", old)
    assert A.PAR_LAYER_YIELD is True


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed")
