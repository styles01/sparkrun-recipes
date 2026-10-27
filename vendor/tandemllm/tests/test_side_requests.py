"""A client's side requests (S1, 2026-10-02): opencode's title and summary calls, Hermes' title.

On 2026-10-02 at 18:08 opencode's title request reached the router as a turn: no tools, temperature
0.5, its fixed title instruction around the user's message. Qwen classed it code_edit at xhigh, it
got the coding package and wrote 5,567 tokens, and the session's real turn waited 54 s behind it.
The server now knows such a request by its shape (no tools, the client's own instruction opening
the system message; server/conductor/signals.py `SIDE_SHAPES`) and serves it as what it is.

The test matrix:

  the shape (signals.side_shape), on the instructions the clients ship
  (tests/fixtures/side/: opencode 1.18.34's title and summary agents, Hermes 0.21.0's title):
    1  each client's request is its shape; with tools, another system text, no system message, or
       the instruction later in the message it is not; a rewrapped instruction still is
                                                          test_the_shapes_*
  the router (the checkpoint's template, the classifiers faked):
    2  opencode's real title request: no classifier asked, no note, thinking off (the template's
       closed block at the tail), the cap 64, the journal row `kind: side`; the session's class,
       turns and notes are what the turn before left           test_opencode_title_*
    3  a numeric reasoning budget from the client keeps thinking on, under its number
                                                          test_a_numeric_budget_*
    4  `structural.side_requests: off` serves it as a turn, as before    test_off_*
    5  a profile in shadow journals the side request and changes nothing  test_shadow_*
    6  Hermes' title request; a router's own `side_shapes` replace the built-in ones
                                                          test_hermes_*, test_the_router_shapes_*
    7  the decisions feed lists the side rows with the router's turns; the Overview's and the
       sessions' figures read the turns alone                  test_the_journal_*
  the handler (the fake engine in process):
    8  the title request through the server: at most 64 tokens, thinking closed in the prompt
       the engine got, no state line even with `ui.states` on   test_served_*
    9  the queue: a side request that arrives while a turn waits runs before it; the waiting turn's
       stream says so ("waiting: opencode's title request runs first"); a long side request waits
       after the turns                                       test_the_queue_*

Run: python tests/test_side_requests.py
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

from server import app  # noqa: E402
from server.conductor import schema, signals, states  # noqa: E402
from test_conductor_router import TOOLS, setup, text_of  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "side")
OC_TITLE = open(os.path.join(FIX, "opencode-1.18.34-title-system.txt")).read()
OC_SUMMARY = open(os.path.join(FIX, "opencode-1.18.34-summary-system.txt")).read()
HERMES_TITLE = open(os.path.join(FIX, "hermes-0.21.0-title-system.txt")).read()
USER = "Build a voxel pagoda viewer in three.js with orbit controls and a day/night toggle."
OPENCODE_UA = "opencode/1.18.34 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"


def opencode_title(user=USER, **extra):
    """opencode 1.18.34's title request (SessionPrompt.ensureTitle through the openai-compatible
    provider): the title agent's prompt as the one system message, then "Generate a title for this
    conversation:" and the first user message; no tools, the agent's temperature 0.5, the small
    model's options (`reasoning_effort`), streamed with usage."""
    return {"model": "TandemLLM", "stream": True, "stream_options": {"include_usage": True},
            "temperature": 0.5, "max_tokens": 32000, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": OC_TITLE},
                         {"role": "user", "content": "Generate a title for this conversation:\n"},
                         {"role": "user", "content": user}], **extra}


def hermes_title(user=USER):
    """Hermes 0.21.0's title_generator: its prompt, the user's opening message, 64 tokens,
    temperature 0.3, a JSON schema for the answer."""
    return {"model": "TandemLLM", "max_tokens": 64, "temperature": 0.3,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "session_title", "strict": True, "schema": {
                    "type": "object", "properties": {"title": {"type": "string"}},
                    "required": ["title"], "additionalProperties": False}}},
            "messages": [{"role": "system", "content": HERMES_TITLE},
                         {"role": "user", "content": user}]}


# ------------------------------------------------------------------ 1. the shapes

def test_the_shapes_are_the_clients_instructions():
    assert signals.side_shape(opencode_title())["id"] == "opencode_title"
    summary = {"messages": [{"role": "system", "content": OC_SUMMARY},
                            {"role": "user", "content": USER},
                            {"role": "assistant", "content": "I added the viewer."}]}
    assert signals.side_shape(summary)["id"] == "opencode_summary"
    assert signals.side_shape(hermes_title())["id"] == "hermes_title"
    assert [s["max_tokens"] for s in signals.SIDE_SHAPES] == [64, 256, 64]
    # rewrapped by a client (other line breaks, other spaces): the same shape
    wrapped = opencode_title()
    wrapped["messages"][0]["content"] = "  " + OC_TITLE.replace(". ", ".\n   ", 2)
    assert signals.side_shape(wrapped)["id"] == "opencode_title"
    # list content with the text in a part
    parts = opencode_title()
    parts["messages"][0]["content"] = [{"type": "text", "text": OC_TITLE}]
    assert signals.side_shape(parts)["id"] == "opencode_title"


def test_the_shapes_are_not_anything_else():
    assert signals.side_shape(opencode_title(tools=TOOLS)) is None, "a request with tools is a turn"
    b = opencode_title()
    b["messages"][0]["content"] = "You are opencode. " + OC_TITLE      # later in the text
    assert signals.side_shape(b) is None
    b = opencode_title()
    b["messages"] = b["messages"][1:]                                   # no system message
    assert signals.side_shape(b) is None
    b = opencode_title()
    b["messages"][1:1] = [{"role": "system", "content": "x"}]
    b["messages"][0] = {"role": "user", "content": OC_TITLE}            # the text as a user's
    assert signals.side_shape(b) is None
    assert signals.side_shape({"messages": [{"role": "system", "content": "Write a title."}]}) \
        is None, "the words are not the shape: only the client's fixed instruction is"
    assert signals.side_shape({"messages": []}) is None


# ------------------------------------------------------------------ 2-7. the router

def _turn(cond, b, headers, kind="opencode"):
    plan = cond.plan(b, headers, client_kind=kind, request_id="req", chat=True)
    prompt, _, in_think = app.build_prompt(b)
    out, after = plan.apply_prompt(prompt, app.STATE["tok"], in_think)
    return plan, out.tolist(), after


def test_opencode_title_asks_no_classifier_and_leaves_the_session_alone():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.95))
    hdr = {"x-session-id": "ses-side"}
    work = {"model": "TandemLLM", "tools": TOOLS,
            "messages": [{"role": "system", "content": "You are opencode."},
                         {"role": "user", "content": USER}]}
    first, _, _ = _turn(cond, work, hdr)
    sess = cond.router.session("c:ses-side")
    before = (sess.cls, sess.conf, sess.turns, sess.user_hash, len(sess.notes),
              [n.ids for n in sess.notes])
    asked = len(fakes["decider"].calls) + len(fakes["qwen"].calls)
    plan, ids, in_think = _turn(cond, opencode_title(), hdr)
    assert len(fakes["decider"].calls) + len(fakes["qwen"].calls) == asked, "no classifier"
    assert plan.side == {"shape": "opencode_title", "client": "opencode", "kind": "title",
                         "max_tokens": 64, "thinking": "off"}, plan.side
    assert plan.block is None and not plan.injected, "no package, no rules, no note"
    assert plan.thinking_off and not in_think
    assert text_of(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), \
        text_of(ids)[-80:]
    assert "<runtime-note>" not in text_of(ids)
    row = plan.row
    assert row["kind"] == "side" and row["class_id"] is None and row["source"] == "structural"
    assert row["session_key"] == "c:ses-side" and row["extra"]["side"]["shape"] == "opencode_title"
    assert row["thinking"] == "off" and row["acted"] == 1
    after = (sess.cls, sess.conf, sess.turns, sess.user_hash, len(sess.notes),
             [n.ids for n in sess.notes])
    assert after == before, (before, after)
    assert first.row["kind"] == "route"
    # the next real turn of the session replays its note exactly as without the side request
    nxt, ids2, _ = _turn(cond, dict(work, messages=work["messages"] + [
        {"role": "assistant", "content": "On it."}, {"role": "user", "content": "Go on."}]), hdr)
    assert nxt.row["kind"] == "route" and nxt.row.get("replayed")


def test_a_numeric_budget_keeps_thinking_on():
    cond, _ = setup()
    plan, ids, in_think = _turn(cond, opencode_title(max_reasoning_tokens=200),
                                {"x-session-id": "s"})
    assert plan.side["thinking"] == "client" and not plan.thinking_off and in_think
    assert text_of(ids).endswith("<think>\n")
    # an effort word, a switch or 0 (no cap) are not a number of tokens: thinking off
    for extra in ({"reasoning_effort": "xhigh"}, {"reasoning": {"enabled": True}},
                  {"max_reasoning_tokens": 0}):
        plan, _, in_think = _turn(cond, opencode_title(**extra), {"x-session-id": "s"})
        assert plan.side["thinking"] == "off" and not in_think, extra


def _router(cond, **structural):
    _, doc = cond.store.get("router")
    doc["structural"].update(structural)
    cond.store.put("router", None, doc, note="side", author="test")


def test_off_serves_it_as_a_turn():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.95))
    _router(cond, side_requests="off")
    plan, ids, in_think = _turn(cond, opencode_title(), {"x-session-id": "s"})
    assert plan.side is None and plan.row["kind"] == "route" and fakes["decider"].calls
    assert plan.row["class_id"] == "code_edit"
    # a router stored before S1 (no key) has it on
    cond, _ = setup()
    _, doc = cond.store.get("router")
    doc["structural"].pop("side_requests")
    cond.store.put("router", None, doc, note="old", author="test")
    assert schema.side_requests(cond.store.get("router")[1])["mode"] == "on"
    assert _turn(cond, opencode_title(), {"x-session-id": "s"})[0].side is not None


def test_shadow_journals_and_changes_nothing():
    cond, fakes = setup()
    _, prof = cond.store.get("profiles", "coding")
    prof["mode"] = "shadow"
    cond.store.put("profiles", "coding", prof, note="shadow", author="test")
    b = opencode_title()
    plan, ids, in_think = _turn(cond, b, {"x-session-id": "s", "User-Agent": OPENCODE_UA})
    prompt, _, _ = app.build_prompt(b)
    assert plan.side is None and in_think and ids == prompt.tolist()
    assert plan.row["kind"] == "side" and plan.row["shadow"]["side"]["shape"] == "opencode_title"
    assert not fakes["decider"].calls and not fakes["qwen"].calls


def test_hermes_title_is_a_side_request():
    cond, fakes = setup()
    plan, ids, in_think = _turn(cond, hermes_title(), {}, kind="openai-sdk")
    assert plan.side["shape"] == "hermes_title" and plan.side["client"] == "Hermes"
    assert plan.side["max_tokens"] == 64 and not in_think
    assert not fakes["decider"].calls and not fakes["qwen"].calls


def test_the_router_shapes_replace_the_built_in_ones():
    cond, _ = setup()
    mine = {"id": "my_title", "client": "myclient", "kind": "title", "max_tokens": 24,
            "match": "You title chats for MyClient. Reply with the title only."}
    _router(cond, side_shapes=[mine])
    b = opencode_title()
    b["messages"][0]["content"] = mine["match"] + "\nBe brief."
    plan, _, _ = _turn(cond, b, {"x-session-id": "s"})
    assert plan.side["shape"] == "my_title" and plan.side["max_tokens"] == 24
    plan, _, _ = _turn(cond, opencode_title(), {"x-session-id": "s"})
    assert plan.side is None, "the document's list replaces the built-in one"
    # two shapes with one id, a match too short to be a client's fixed text: refused
    for bad in ([mine, mine], [dict(mine, match="Write a title.")]):
        _, doc = cond.store.get("router")
        doc["structural"]["side_shapes"] = bad
        assert schema.check("router", doc), bad


def test_the_journal_lists_side_rows_with_the_routers():
    cond, _ = setup()
    plan, _, _ = _turn(cond, opencode_title(), {"x-session-id": "s"})
    turn, _, _ = _turn(cond, {"model": "TandemLLM", "tools": TOOLS, "messages": [
        {"role": "user", "content": USER}]}, {"x-session-id": "s"})
    ids = lambda kind: {r["id"] for r in cond.journal.rows(kind=kind)}       # noqa: E731
    assert ids("router") == {plan.row["id"], turn.row["id"]}, "the feed lists both"
    assert ids("side") == {plan.row["id"]} and ids("route") == {turn.row["id"]}, \
        "the Overview's and the sessions' figures read the turns alone"
    from server.conductor import decisions
    assert decisions.sessions(cond.journal.rows(kind="route"))[0]["class_id"] == "code_edit"
    from server.conductor.decisions import route_fields
    f = route_fields(plan.row)
    assert f["kind"] == "side" and f["decision"] == "opencode_title" and f["cap"] == 64
    assert f["action"] == "SIDE_REQUEST"


# ------------------------------------------------------------------ 8-9. the handler

def serve(ui=None):
    """The fake engine with the monitor and a conductor (test_ui_states.serve), the coding and chat
    profiles' `ui` as given; every prompt the engine got, with when it started."""
    import test_ui_states as tu
    cond, seen = tu.serve(None)
    for name in ("coding", "chat"):
        _, doc = cond.store.get("profiles", name)
        if ui is not None:
            doc["ui"] = ui
        cond.store.put("profiles", name, doc, note="ui", author="test")
    order = []
    real = app.generate_stream

    def recording(prompt, *a, **k):
        order.append((time.monotonic(), app.STATE["tok"].decode(prompt.tolist(),
                                                                skip_special_tokens=False)))
        return real(prompt, *a, **k)

    app.generate_stream = recording
    return cond, order


def go(b, headers=None):
    import test_ui_states as tu
    return tu._go(b, dict({"User-Agent": OPENCODE_UA, "x-session-id": "ses_s1"},
                          **(headers or {})))


def test_served_title_is_short_and_thinks_not():
    cond, order = serve({"states": "on", "states_channel": "content"})
    out = go(opencode_title())
    assert out["usage"]["completion_tokens"] <= 64, out["usage"]
    assert order[-1][1].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), \
        order[-1][1][-60:]
    assert states.MARKER not in out["raw"], "no state line in a title"
    rid = json.loads(out["raw"].splitlines()[0][6:])["id"]
    row = [r for r in cond.journal.ring.values() if r.get("request_id") == rid][0]
    assert row["kind"] == "side" and row["extra"]["side"]["first"] is True
    # the same request with the switch off: a turn, thinking, and far longer
    _, doc = cond.store.get("router")
    doc["structural"]["side_requests"] = "off"
    cond.store.put("router", None, doc, note="off", author="test")
    off = go(opencode_title())
    assert off["usage"]["completion_tokens"] > 64, off["usage"]
    assert order[-1][1].endswith("<think>\n"), order[-1][1][-40:]


@contextlib.contextmanager
def threads_quiet():
    """Requests on threads: `redirect_stdout` swaps the process's stdout, so per-request
    redirects on two threads at once leave it pointing at a closed buffer. One redirect here, on
    the test's own thread, for all of them."""
    import test_ui_states as tu
    real = tu.quiet
    tu.quiet = contextlib.nullcontext
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        tu.quiet = real


def _bg(b, headers=None):
    box = {}
    t = threading.Thread(target=lambda: box.update(out=go(b, headers)), daemon=True)
    t.start()
    return t, box


def _wait_until(pred, timeout=20.0):
    t0 = time.monotonic()
    while not pred():
        assert time.monotonic() - t0 < timeout, "timed out"
        time.sleep(0.02)


def test_the_queue_puts_a_short_side_request_first_and_says_so():
    import test_ui_states as tu
    cond, order = serve({"states": "on", "states_channel": "content"})
    tools = tu.TOOLS
    with threads_quiet():
        # a slow turn holds the engine; a second turn of the session waits; then the title request
        busy, _ = _bg(tu_body("FAKE_SLOW one", tools, 40), {"x-session-id": "ses_other"})
        _wait_until(lambda: len(order) == 1)
        turn, turn_box = _bg(tu_body("the waiting turn", tools))
        _wait_until(lambda: app.INFLIGHT["waiting"] == 1)
        time.sleep(2.4)               # its first waiting line goes out (after QUIET_S, SRV-177)
        side, side_box = _bg(opencode_title())
        for t in (busy, turn, side):
            t.join(60)
    texts = [o[1] for o in order]
    assert "the waiting turn" in texts[2] and OC_TITLE[:40] in texts[1], \
        [x[-60:] for x in texts]
    got = [ln[len(states.MARKER):] for ln in turn_box["out"]["content"].split("\n")
           if ln.startswith(states.MARKER)]
    assert got[0] == "waiting: another request is running, position 1", got
    assert "waiting: opencode's title request runs first, position 2" in got, got
    assert "waiting: opencode's title request runs first, position 1" in got, got
    assert got.index("waiting: opencode's title request runs first, position 1") < \
        got.index("deciding: code_edit, xhigh effort"), got
    # the waiting lines are content, outside the reasoning, and none reached the title
    assert states.MARKER not in turn_box["out"]["reasoning"]
    assert states.MARKER not in side_box["out"]["raw"]


def tu_body(text, tools, max_tokens=2000):
    import test_ui_states as tu
    return {"model": "TandemLLM", "stream": True, "max_tokens": max_tokens, "tools": tools,
            "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": tu.SYSTEM},
                         {"role": "user", "content": text}]}


def test_the_queue_puts_a_long_side_request_after_the_turns():
    import test_ui_states as tu
    cond, order = serve(None)
    with threads_quiet():
        busy, _ = _bg(tu_body("FAKE_SLOW one", tu.TOOLS, 30), {"x-session-id": "ses_other"})
        _wait_until(lambda: len(order) == 1)
        long_side, _ = _bg(opencode_title(user="x" * (schema.SIDE_FIRST_MAX_TOKENS + 100)))
        _wait_until(lambda: app.INFLIGHT["waiting"] == 1)
        turn, _ = _bg(tu_body("the waiting turn", tu.TOOLS))
        for t in (busy, long_side, turn):
            t.join(60)
    texts = [o[1] for o in order]
    assert "the waiting turn" in texts[1] and OC_TITLE[:40] in texts[2], \
        [x[-60:] for x in texts]


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
