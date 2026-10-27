"""The state lines outside the thinking, and their progress numbers (S1, 2026-10-02).

The owner: the `›` lines "must be outside the think". A profile's `ui.states_channel: content` (the
default of a new profile; one stored before keeps `reasoning`) sends them as content deltas:
before the reasoning starts, between its parts, after it. The progress lines say what a wait is
made of, templated: the queue ("waiting: <reason>, position N"), a long prefill ("reading 35,189
tokens at ~750 tok/s, about 47 s left", every `prefill_every_s`), a long thinking ("thinking 4 min,
9,800 tokens", every `thinking_every_s` or `thinking_every_tokens`).

The test matrix (the fake engine in process, the monitor on, a conductor):

  the content channel:
    1  opencode's shape (`reasoning_content`): every line is a content delta, the router's before
       any reasoning, the tool checks' after it; no line in the reasoning; the client's text with
       the lines taken out, its reasoning, calls and usage equal the run with the lines off
                                                          test_opencode_*
    2  turn 2 sends turn 1's content back with its lines (as opencode keeps them in its text part,
       and trimmed): the engine gets the prompt it gets without them, token for token, and the
       client the same usage                              test_a_replayed_content_*
    3  Hermes' shape (`tags`): the router's line before `<think>`; a line said inside the block
       waits for `</think>` and goes out right after it; nothing inside the block
                                                          test_hermes_*
    4  a non-streamed answer: the lines at the head of the content       test_non_streamed_*
    5  on the checkpoint's template, an assistant content with the lines (opencode's text part,
       trimmed, Hermes' tags) renders the prompt without them      test_the_prompt_*
  the progress lines:
    6  a long prefill says its tokens, rate and time left, more than once
                                                          test_a_long_prefill_*
    7  a long thinking says its time and tokens, between reasoning parts (content), inside the
       reasoning (reasoning), once after `</think>` (tags: only the newest waits)
                                                          test_a_long_thinking_*
    8  `ui.progress.mode: off`: none of them             test_progress_off_*
    9  a request that waits: its stream opens with "waiting: another request is running, position
       1"; a queue timeout then ends the stream with an error (429) and `[DONE]`
                                                          test_a_waiting_request_*

Run: python tests/test_state_channel.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import app, ledger, queue_wait as qwait  # noqa: E402
from server.conductor import states  # noqa: E402
import test_ui_states as tu  # noqa: E402

M = states.MARKER
CONTENT = {"states": "on", "states_channel": "content"}
CALL = "FAKE_DROP_CLOSER FAKE_CALL_JSON " + json.dumps({"name": "bash",
                                                         "arguments": {"command": "ls -la"}})


def serve(ui):
    cond, seen = tu.serve(None)
    for name in ("coding", "chat"):
        _, doc = cond.store.get("profiles", name)
        doc["ui"] = ui
        cond.store.put("profiles", name, doc, note="ui", author="test")
    return cond, seen


def lines(text):
    return [ln[len(M):] for ln in text.split("\n") if ln.startswith(M)]


def same_usage(a, b):
    for k in ("prompt_tokens", "completion_tokens"):
        assert a["usage"][k] == b["usage"][k], (k, a["usage"], b["usage"])


# ------------------------------------------------------------------ 1-5. the content channel

def test_opencode_lines_are_content_and_the_rest_is_unchanged():
    serve(CONTENT)
    on = tu.opencode(CALL)
    serve({"states": "off"})
    off = tu.opencode(CALL)
    got = lines(on["content"])
    assert got[0] == "deciding: code_edit, xhigh effort", got
    assert "checking the tool call" in got and "repairing the tool call" in got, got
    assert M not in on["reasoning"], "nothing in the reasoning"
    assert on["order"][0] == ("content", states.frame("deciding: code_edit, xhigh effort"))
    first_r = next(i for i, (k, _) in enumerate(on["order"]) if k == "reasoning")
    assert first_r > 0, "the router's line comes before the reasoning starts"
    assert states.strip_text(on["content"]) == off["content"]
    assert on["reasoning"] == off["reasoning"]
    assert on["calls"] == [dict(c, id=on["calls"][i]["id"]) if "id" in c else c
                           for i, c in enumerate(off["calls"])]
    same_usage(on, off)


def test_a_replayed_content_reaches_the_engine_without_its_lines():
    def second(shape):
        cond, seen = serve(CONTENT)
        first = tu.opencode("Write the lexer.")
        c = first["content"]
        assert lines(c)
        sent = {"lines": c, "trimmed": c.strip(), "clean": states.strip_text(c)}[shape]
        asst = {"role": "assistant", "content": sent, "reasoning_content": first["reasoning"]}
        seen.clear()
        out = tu.opencode("Now add a test.",
                          history=[{"role": "user", "content": "Write the lexer."}, asst])
        return seen[-1], out

    p_clean, o_clean = second("clean")
    for shape in ("lines", "trimmed"):
        p, o = second(shape)
        assert p == p_clean, f"{shape}: the engine's prompt is the one without the lines"
        same_usage(o, o_clean)


def test_hermes_line_before_think_and_one_held_to_the_close():
    serve(CONTENT)
    on = tu.hermes("Explain the lexer.", max_reasoning_tokens=12)
    serve({"states": "off"})
    off = tu.hermes("Explain the lexer.", max_reasoning_tokens=12)
    head, sep, rest = on["content"].partition("<think>\n")
    assert sep and lines(head) == ["deciding: chat, low effort"], on["content"][:200]
    block, close, after = rest.partition("</think>")
    assert close and M not in block, "nothing inside the block"
    assert lines(after)[0] == "thinking closed at the 12 cap", after[:200]
    assert after.startswith(states.frame("thinking closed at the 12 cap")), after[:80]
    assert states.strip_text(on["content"]) == off["content"]
    same_usage(on, off)


def test_non_streamed_lines_lead_the_content():
    serve(CONTENT)
    on = tu.opencode("Write the lexer.", stream=False)
    serve({"states": "off"})
    off = tu.opencode("Write the lexer.", stream=False)
    assert on["content"].startswith(states.frame("deciding: code_edit, xhigh effort"))
    assert M not in on["reasoning"] and on["reasoning"] == off["reasoning"]
    assert states.strip_text(on["content"]) == off["content"]
    same_usage(on, off)


def test_the_prompt_from_content_with_lines_is_the_prompt_without():
    """On the checkpoint's template: an assistant message whose content carries the lines (and
    the progress lines' numbers), as opencode's text part and Hermes' tags content keep them,
    renders the prompt it renders without them, token for token."""
    r = "The user wants a lexer. I read src/lexer.ts first."
    wait = states.frame("waiting: opencode's title request runs first, position 1")
    read = states.frame("reading 35,189 tokens at ~750 tok/s, about 47 s left")
    route = states.frame("deciding: code_edit, xhigh effort")
    think = states.frame("thinking 4 min, 9,800 tokens")
    check = states.frame("checking the tool call")
    sysm = {"role": "system", "content": "You are opencode."}
    user = {"role": "user", "content": "Refactor the lexer."}
    call = [{"id": "c1", "type": "function", "function": {"name": "read",
                                                           "arguments": '{"path": "a"}'}}]
    tool = {"role": "tool", "tool_call_id": "c1", "content": "file text"}
    nxt = {"role": "user", "content": "Go on."}

    def conv(asst):
        return [sysm, user, asst, tool, nxt]

    for answer in ("", "I will read the file first."):
        plain = {"role": "assistant", "content": answer, "reasoning_content": r,
                 "tool_calls": call}
        want = tu._render(conv(plain))
        for content in (wait + read + route + think + answer + check,
                        (wait + read + route + think + answer + check).strip()):
            msgs, n = states.strip_messages(conv(dict(plain, content=content)))
            assert n > 0 and tu._render(msgs) == want, (answer, content[:60])
            assert tu._render(conv(dict(plain, content=content))) != want
    # tags: a line before the block, one held to its close
    tagged = {"role": "assistant",
              "content": route + "<think>\n" + r + "</think>" + think + "\n\nDone."}
    plain_t = {"role": "assistant", "content": "<think>\n" + r + "</think>\n\nDone."}
    msgs, n = states.strip_messages(conv(tagged))
    assert n > 0 and tu._render(msgs) == tu._render(conv(plain_t))


# ------------------------------------------------------------------ 6-9. the progress lines

PREFILL = re.compile(r"^reading ([\d,]+) tokens at ~([\d,]+) tok/s, about (\d+ s|\d+ min) left$")
THINKING = re.compile(r"^thinking (\d+ s|\d+ min), ([\d,]+) tokens$")


def test_a_long_prefill_says_its_numbers():
    serve(dict(CONTENT, progress={"mode": "on", "prefill_every_s": 2}))
    on = tu.opencode("FAKE_SLOW_PREFILL read it all")
    got = [g for g in lines(on["content"]) if g.startswith("reading")]
    assert len(got) >= 2, lines(on["content"])
    for g in got:
        m = PREFILL.match(g)
        assert m, g
        assert 8000 <= int(m.group(1).replace(",", "")) <= 9000, g
        assert 1500 <= int(m.group(2).replace(",", "")) <= 2100, g       # 2,000 tok/s, slept
    left = [int(PREFILL.match(g).group(3).split()[0]) for g in got]
    assert left == sorted(left, reverse=True), left
    # before the reasoning, and taken out like any line
    first_r = next(i for i, (k, _) in enumerate(on["order"]) if k == "reasoning")
    assert any(k == "content" and "reading" in v for k, v in on["order"][:first_r])
    serve({"states": "off"})
    off = tu.opencode("FAKE_SLOW_PREFILL read it all")
    assert states.strip_text(on["content"]) == off["content"]


def test_a_long_thinking_says_its_time_and_tokens():
    prog = {"mode": "on", "thinking_every_tokens": 1000}
    serve(dict(CONTENT, progress=prog))
    on = tu.opencode("FAKE_LOOP think on")
    got = [g for g in lines(on["content"]) if g.startswith("thinking ") and "cap" not in g]
    assert got and all(THINKING.match(g) for g in got), lines(on["content"])
    assert [int(THINKING.match(g).group(2).replace(",", "")) for g in got][:2] == \
        [1000, 2000][:len(got)], got
    # between two parts of the reasoning: content after some reasoning and before more
    i = next(i for i, (k, v) in enumerate(on["order"]) if k == "content" and "thinking " in v
             and "cap" not in v)
    kinds = [k for k, _ in on["order"]]
    assert "reasoning" in kinds[:i] and "reasoning" in kinds[i + 1:], kinds[max(0, i - 3):i + 3]
    serve({"states": "off"})
    off = tu.opencode("FAKE_LOOP think on")
    assert on["reasoning"] == off["reasoning"]
    # in the reasoning channel, inside the reasoning
    serve({"states": "on", "progress": prog})
    r = tu.opencode("FAKE_LOOP think on")
    assert [THINKING.match(g).group(2) for g in lines(r["reasoning"]) if THINKING.match(g)] == \
        [THINKING.match(g).group(2) for g in got]
    # tags in the content channel: only the newest waits for the block to close
    serve(dict(CONTENT, progress=prog))
    # the Hermes agent itself (tags, its lines in the content); a plain SDK client gets none
    # there since SRV-245
    h = tu.hermes("FAKE_LOOP think on", system=ledger.HERMES_IDENTITY)
    _, _, rest = h["content"].partition("<think>\n")
    block, _, after = rest.partition("</think>")
    assert M not in block
    held = [g for g in lines(after) if THINKING.match(g)]
    assert len(held) == 1, lines(after)


def test_progress_off_says_none_of_them():
    serve(dict(CONTENT, progress={"mode": "off", "thinking_every_tokens": 1000,
                                  "prefill_every_s": 2}))
    a = tu.opencode("FAKE_SLOW_PREFILL FAKE_LOOP both")
    assert not [g for g in lines(a["content"]) if PREFILL.match(g) or THINKING.match(g)], \
        lines(a["content"])
    assert lines(a["content"])[0] == "deciding: code_edit, xhigh effort"


class Req(tu.Req):
    pass


def _bg(b, headers):
    box = {}

    def run():
        r = Req(b, headers)
        box["head"], box["raw"] = r.response()
        box["status"] = r._status

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def test_a_waiting_request_opens_its_stream_and_a_timeout_closes_it():
    serve(CONTENT)
    real = tu.quiet
    tu.quiet = contextlib.nullcontext
    rt = app.STATE.get("request_timeout")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            busy_body = {"model": "TandemLLM", "stream": True, "max_tokens": 40, "tools": tu.TOOLS,
                         "reasoning_format": "reasoning_content",
                         "messages": [{"role": "user", "content": "FAKE_SLOW hold it"}]}
            busy, _ = _bg(busy_body, {"User-Agent": tu.OPENCODE_UA, "x-session-id": "a"})
            t0 = time.monotonic()
            while not app.LOCK.locked():
                assert time.monotonic() - t0 < 10
                time.sleep(0.02)
            time.sleep(0.2)                  # the busy request's own deadline is set by now
            # a streamed request waits as long as one request may run (2026-10-05): that limit
            # closes the open stream here
            app.STATE["queue_timeout"] = 1.5
            rt = app.STATE.get("request_timeout")
            app.STATE["request_timeout"] = 1.5
            # the line at once (an earlier build waited QUIET_S, 2 s, before the first one: longer than
            # this test's whole limit)
            qwait.QUIET_S = 0.0
            wait_body = dict(busy_body, messages=[{"role": "user", "content": "me next"}])
            waiter, box = _bg(wait_body, {"User-Agent": tu.OPENCODE_UA, "x-session-id": "b"})
            waiter.join(30)
            busy.join(60)
    finally:
        tu.quiet = real
        app.STATE["queue_timeout"] = 120.0
        app.STATE["request_timeout"] = rt
        qwait.QUIET_S = 2.0
    assert "200 OK" in box["head"].splitlines()[0], box["head"]
    events = [json.loads(ln[6:]) for ln in box["raw"].splitlines() if ln.startswith("data: {")]
    text = "".join((c.get("delta") or {}).get("content") or "" for e in events
                   for c in e.get("choices") or [])
    assert lines(text) == ["waiting: another request is running, position 1"], text
    last = events[-1]
    assert last["choices"][0]["finish_reason"] == "error" and last["error"]["code"] == 429, last
    assert box["raw"].rstrip().endswith("data: [DONE]")
    assert box["status"] == 429, "the ledger keeps the refusal"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
