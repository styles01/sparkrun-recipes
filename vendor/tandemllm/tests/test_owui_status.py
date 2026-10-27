"""Open WebUI's "Thought for N seconds" times the thinking, not the wait (SRV-237, SRV-249).

Open WebUI 0.11.4 times a reasoning block from its first `reasoning_content` delta to the first
content delta. The reading and waiting lines went out as reasoning, so a thinking-off summary that
read 12,874 tokens showed "Thought for 14 seconds" and a one-line answer that waited 35.8 s for
the other request "Thought for 34 seconds". For the client kinds in `QWEN38_STATUS_CLIENTS`
(default `open-webui`) every line said before the first generated token goes out as an Open WebUI
`status` event (`{"choices": [], "event": {"type": "status", ...}}`), and the reasoning opens
with the model's first token (server/conductor/states.py StatusBar).

The test matrix (the fake engine in process, a conductor, Open WebUI's User-Agent):

  1  the bar: plain text from a framed line, events while live, one `done` event that says the
     tokens read and the time waited, nothing after it          test_1_*
  2  a thinking-off turn with a long prefill: the reading lines are status events, before the
     first content; no reasoning delta at all (no "Thought for"); the last event is done and
     names the tokens read; the content equals the run with the bar off  test_2_*
  3  the same turn with `QWEN38_STATUS_CLIENTS` empty: the lines in the reasoning, as before
                                                                    test_3_*
  4  a thinking turn: the reasoning starts with the model's first token, no line in it before
                                                                    test_4_*
  5  a waiting request: its waiting line is a status event, never reasoning; the queue timeout
     closes the status line (done) before the error chunk        test_5_*
  6  opencode (not in the list): its lines unchanged, no event      test_6_*

Run: python tests/test_owui_status.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import app, queue_wait as qwait  # noqa: E402
from server.conductor import states  # noqa: E402
import test_ui_states as tu  # noqa: E402

M = states.MARKER
# what Open WebUI 0.11.4 sends from its backend (aiohttp); ledger.client_of reads it as open-webui
OWUI_UA = "Python/3.12 aiohttp/3.11.18"


def serve(ui):
    cond, seen = tu.serve(None)
    for name in ("coding", "chat"):
        _, doc = cond.store.get("profiles", name)
        doc["ui"] = ui
        cond.store.put("profiles", name, doc, note="ui", author="test")
    return cond, seen


def owui(text, effort=None, **extra):
    """An Open WebUI-shaped request: no reasoning_format, no tools, streamed."""
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 2000,
            "messages": [{"role": "user", "content": text}], **extra}
    if effort is not None:
        body["reasoning_effort"] = effort
    return parse(*tu.Req(body, {"User-Agent": OWUI_UA}).response())


def parse(head, raw):
    out = {"head": head, "raw": raw, "content": "", "reasoning": "", "events": [], "order": []}
    for ln in raw.splitlines():
        if not ln.startswith("data: {"):
            continue
        ev = json.loads(ln[6:])
        if "event" in ev:
            assert ev.get("choices") == [], ev          # every OpenAI client skips it
            out["events"].append(ev["event"])
            out["order"].append(("event", ev["event"]))
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                out["content"] += d["content"]
                out["order"].append(("content", d["content"]))
            if d.get("reasoning_content"):
                out["reasoning"] += d["reasoning_content"]
                out["order"].append(("reasoning", d["reasoning_content"]))
    return out


@contextlib.contextmanager
def clients(spec):
    real = states.status_clients
    states.status_clients = lambda s=None: real(spec)
    try:
        yield
    finally:
        states.status_clients = real


PROG = {"mode": "on", "prefill_every_s": 2}
REASONING = {"states": "on", "states_channel": "reasoning", "progress": PROG}


# ------------------------------------------------------------------ 1. the bar


def test_1_the_bar_says_plain_lines_and_closes_once():
    assert states.plain(states.frame("reading 12,874 tokens, about 14 s left")) == \
        "Reading 12,874 tokens, about 14 s left"
    assert states.plain("\n› reading 12,874 tokens, about 3 s left⁠\n\n") == \
        "Reading 12,874 tokens, about 3 s left"
    assert states.plain(states.inline(states.frame("deciding: chat, low effort"))) == \
        "Deciding: chat, low effort"
    assert states.status_clients("open-webui, hermes,") == frozenset({"open-webui", "hermes"})
    assert states.status_clients("") == frozenset()
    bar = states.StatusBar()
    assert bar.close() is None and not bar.live, "no line: nothing to close"
    bar = states.StatusBar()
    e = bar.line("queue_eta", states.frame("waiting: another request is running, position 1, "
                                           "about 40 s"))
    assert e == {"type": "status", "data": {"description": "Waiting: another request is "
                                            "running, position 1, about 40 s", "done": False}}
    bar.line("prefill", states.frame("reading 12,874 tokens at ~900 tok/s, about 14 s left"))
    done = bar.close(read=(12874, 14.2), waited=31.0)
    assert done == {"type": "status", "data": {"description": "Waited 31 s, then read 12,874 "
                                               "tokens in 14 s", "done": True}}, done
    assert bar.close() is None and bar.line("prefill", "x") is None
    # only the route line: the newest line, done
    bar = states.StatusBar()
    bar.line("route", states.frame("deciding: summarize, thinking off"))
    assert bar.close(read=None, waited=0.0)["data"] == {"description": "Deciding: summarize, "
                                                        "thinking off", "done": True}


# ------------------------------------------------------------------ 2-4. the stream


def test_2_a_thinking_off_turn_shows_no_thought_for():
    serve(REASONING)
    on = owui("FAKE_SLOW_PREFILL summarize it", effort="none")
    descs = [e["data"]["description"] for e in on["events"]]
    assert sum(d.startswith("Reading ") for d in descs) >= 2, descs
    assert all(e["type"] == "status" for e in on["events"])
    assert [e["data"]["done"] for e in on["events"]][-1] is True
    assert all(not e["data"]["done"] for e in on["events"][:-1])
    last = descs[-1]
    assert last.startswith("Read ") and " tokens in " in last, last
    # nothing timed as thinking: no reasoning delta, and every event before the first content
    assert on["reasoning"] == "", on["reasoning"][:300]
    kinds = [k for k, _ in on["order"]]
    assert "event" not in kinds[kinds.index("content"):], kinds
    assert M not in on["content"]
    with clients(""):
        serve(REASONING)
        off = owui("FAKE_SLOW_PREFILL summarize it", effort="none")
    assert on["content"] == states.strip_text(off["content"]) or on["content"] == off["content"]


def test_3_without_the_switch_the_lines_stay_in_the_reasoning():
    with clients(""):
        serve(REASONING)
        r = owui("FAKE_SLOW_PREFILL summarize it", effort="none")
    assert not r["events"]
    assert any(ln.startswith("reading") for ln in tu.lines(r["reasoning"])), r["reasoning"][:300]


def test_4_a_thinking_turn_opens_its_reasoning_with_the_model():
    serve(REASONING)
    on = owui("FAKE_SLOW_PREFILL think about it", effort="medium")
    assert on["events"] and on["events"][-1]["data"]["done"] is True
    assert on["reasoning"], "the fake engine thinks on this turn"
    first = next(v for k, v in on["order"] if k == "reasoning")
    assert M not in first, first               # the model's own first piece, not a line
    kinds = [k for k, _ in on["order"]]
    assert kinds.index("event") < kinds.index("reasoning")
    assert all(k != "event" for k in kinds[kinds.index("reasoning"):]), kinds
    assert not [ln for ln in tu.lines(on["reasoning"]) if ln.startswith("reading")]


# ------------------------------------------------------------------ 5. waiting


def _bg(body, headers):
    box = {}

    def run():
        r = tu.Req(body, headers)
        box["head"], box["raw"] = r.response()
        box["status"] = r._status

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def test_5_a_waiting_request_waits_in_the_status_line():
    serve(REASONING)
    real = tu.quiet
    tu.quiet = contextlib.nullcontext
    rt = app.STATE.get("request_timeout")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            busy_body = {"model": "TandemLLM", "stream": True, "max_tokens": 40,
                         "messages": [{"role": "user", "content": "FAKE_SLOW hold it"}]}
            busy, _ = _bg(busy_body, {"User-Agent": OWUI_UA})
            t0 = time.monotonic()
            while not app.LOCK.locked():
                assert time.monotonic() - t0 < 10
                time.sleep(0.02)
            time.sleep(0.2)
            app.STATE["queue_timeout"] = 1.5
            app.STATE["request_timeout"] = 1.5
            qwait.QUIET_S = 0.0
            wait_body = dict(busy_body, messages=[{"role": "user", "content": "17 x 23"}])
            waiter, box = _bg(wait_body, {"User-Agent": OWUI_UA})
            waiter.join(30)
            busy.join(60)
    finally:
        tu.quiet = real
        app.STATE["queue_timeout"] = 120.0
        app.STATE["request_timeout"] = rt
        qwait.QUIET_S = 2.0
    out = parse(box["head"], box["raw"])
    assert "200 OK" in box["head"].splitlines()[0], box["head"]
    assert out["reasoning"] == "" and out["content"] == "", out["order"]
    descs = [e["data"]["description"] for e in out["events"]]
    assert descs[0] == "Waiting: another request is running, position 1", descs
    assert out["events"][-1]["data"]["done"] is True, out["events"]
    events = [json.loads(ln[6:]) for ln in box["raw"].splitlines() if ln.startswith("data: {")]
    last = events[-1]
    assert last["choices"][0]["finish_reason"] == "error" and last["error"]["code"] == 429, last
    assert box["raw"].rstrip().endswith("data: [DONE]")


# ------------------------------------------------------------------ 6. other clients


def test_6_opencode_keeps_its_lines():
    serve({"states": "on", "states_channel": "content", "progress": PROG})
    on = tu.opencode("FAKE_SLOW_PREFILL read it all")
    assert '"event"' not in on["raw"]
    assert any(ln.startswith("reading") for ln in tu.lines(on["content"]))


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
