"""The state lines in time order in opencode: `ui.states_channel: inline` (X1, 2026-10-03).

A user report's screenshot of 09:23 (opencode 1.18.34, coding profile, channel `content`): every `›` line of
the turn in one text block at its top, the reasoning split into "Thought" blocks below it. The
cause, read in opencode's binary (states.INLINE): its @ai-sdk/openai-compatible opens one text part
per response at the first content delta and appends every later one to it, while each reasoning
delta after a content or `tool_calls` delta opens a new reasoning part. `inline` sends each line as
an empty `tool_calls` delta and a reasoning delta `\\n**› text**\\n\\n`: a new reasoning part whose
title (the TUI's `**title**`) is the line, at the moment it happens.

The test matrix (the fake engine in process, the monitor on, a conductor):

  1  the strip: an inline title anywhere in the reasoning (head, end, between two characters,
     parts joined with nothing, trimmed ends, a narrated line) comes out exactly
                                                          test_1_the_strip_*
  2  opencode's shape: every line is an empty `tool_calls` delta then a reasoning title, none in
     the content; through opencode's part rules (`parts()`, the port of its provider and
     processor) each line is the title of a reasoning part, in the order said, the answer's text
     part after them; content, reasoning without the titles, calls and usage equal the run with
     the lines off                                        test_2_opencode_*
  3  turn 2 sends turn 1 back as opencode builds it from its parts (reasoning parts joined, and
     trimmed): the engine gets the prompt it gets without the lines, token for token
                                                          test_3_replayed_*
  4  Hermes' shape (`tags`) under inline is the content channel's, byte for byte
                                                          test_4_hermes_*
  5  a non-streamed answer: the lines at the head of the content, as content does
                                                          test_5_non_streamed_*
  6  a waiting request opens its stream with the queue line as a reasoning title
                                                          test_6_a_waiting_*
  2b a line, then thinking, then the answer (the screenshot of 10:03): inline gives [the thinking
     titled by the line, the answer]; content gave [the line and the answer, the thinking]
                                                          test_2_a_line_then_*
  2c a follow-up after a tool result whose prefill says "reading ... left" twice, then thinks and
     answers (the case of 10:46): inline gives [reading, reading + thinking, the answer]; content
     gave [the lines and the answer, the thinking]           test_2_progress_lines_*
  7  opencode's own parser in node, when the box has node and an opencode binary: the extracted
     provider (tools/opencode_bundle.py, tools/opencode_parts.mjs) gives the parts the port gives,
     for inline (the lines in order, titled; the answer after the thinking) and for content (the
     screenshots: one text part on top with every line and the answer)    test_7_node_*

Run: python tests/test_state_inline.py
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from server import app, ledger, queue_wait as qwait  # noqa: E402
from server.conductor import narrator, states  # noqa: E402
import test_ui_states as tu  # noqa: E402

M = states.MARKER
INLINE = {"states": "on", "states_channel": "inline"}
CONTENT = {"states": "on", "states_channel": "content"}
PROG = {"mode": "on", "thinking_every_tokens": 1000}
CALL = "FAKE_DROP_CLOSER FAKE_CALL_JSON " + json.dumps({"name": "bash",
                                                         "arguments": {"command": "ls -la"}})
TITLE = re.compile(r"^\*\*([^*\n]+)\*\*(?:\r?\n\r?\n|$)")        # opencode's TUI (`ma`)


def serve(ui):
    cond, seen = tu.serve(None)
    for name in ("coding", "chat"):
        _, doc = cond.store.get("profiles", name)
        doc["ui"] = ui
        cond.store.put("profiles", name, doc, note="ui", author="test")
    return cond, seen


def deltas(raw):
    for ln in raw.splitlines():
        if ln.startswith("data: {"):
            ev = json.loads(ln[6:])
            for ch in ev.get("choices") or []:
                yield ch.get("delta") or {}


def parts(raw):
    """opencode's parts for a raw stream: its openai-compatible provider (chunk-9h4pcghr.js,
    doStream) and its SessionProcessor, as tools/opencode_parts.mjs runs the real ones."""
    out, text, rsn = [], None, None
    for d in deltas(raw):
        r = d.get("reasoning_content") or d.get("reasoning")
        if r:
            if rsn is None:
                rsn = {"type": "reasoning", "text": ""}
                out.append(rsn)
            rsn["text"] += r
        if d.get("content"):
            rsn = None
            if text is None:
                text = {"type": "text", "text": ""}
                out.append(text)
            text["text"] += d["content"]
        if d.get("tool_calls") is not None:
            rsn = None
            for tc in d["tool_calls"]:
                if tc.get("id") is not None:
                    out.append({"type": "tool", "name": tc["function"]["name"], "id": tc["id"]})
    for p in out:
        if p["type"] == "reasoning":
            m = TITLE.match(p["text"].strip())
            p["title"] = m.group(1).strip() if m else None
    return out


def message(ps):
    """The assistant message opencode sends back: text parts and reasoning parts, each joined."""
    return {"role": "assistant",
            "content": "".join(p["text"] for p in ps if p["type"] == "text"),
            "reasoning_content": "".join(p["text"] for p in ps if p["type"] == "reasoning")}


def said(raw):
    """The lines the stream said, in order (the inline titles' texts)."""
    return [r.strip("\n")[2 + len(M):-2] for d in deltas(raw)
            for r in [d.get("reasoning_content") or ""] if r.startswith("\n**" + M)]


def same_usage(a, b):
    for k in ("prompt_tokens", "completion_tokens"):
        assert a["usage"][k] == b["usage"][k], (k, a["usage"], b["usage"])


# ------------------------------------------------------------------ 1 the strip
def test_1_the_strip_is_exact_for_inline_titles():
    a = states.inline(states.frame("deciding: code_edit, xhigh effort"))
    b = states.inline(states.frame("thinking 2 min, 4,096 tokens"))
    n = states.inline(narrator.frame("Reading the lexer before the edit"))
    assert a == "\n**› deciding: code_edit, xhigh effort**\n\n", repr(a)
    assert states.inline(states.frame("a *bold* move")) == "\n**› a bold move**\n\n"
    for r in tu.REASONINGS:
        cases = [a + r, r + a, a + r + b, a + b + r, n + r + a, r + n]
        for i in range(1, len(r)):
            cases.append(r[:i] + b + r[i:])
        cases.append(a + r[:len(r) // 2] + b + r[len(r) // 2:] + n)    # parts joined with ""
        for c in cases:
            assert states.strip_text(c) == r, (r, c)
            assert states.strip_text(c.strip()).strip() == r.strip(), (r, c)
    # the title is opencode's
    assert TITLE.match(a.strip()).group(1) == "› deciding: code_edit, xhigh effort"


# ------------------------------------------------------------------ 2 opencode's shape
def test_2_opencode_lines_are_reasoning_titles_in_order():
    serve(dict(INLINE, progress=PROG))
    on = tu.opencode("FAKE_LOOP think on")
    serve({"states": "off"})
    off = tu.opencode("FAKE_LOOP think on")
    lines = said(on["raw"])
    assert lines[0] == "deciding: code_edit, xhigh effort", lines
    assert any(re.match(r"thinking \d+ (s|min), [\d,]+ tokens$", x) for x in lines), lines
    assert M not in on["content"] and on["content"] == off["content"]
    # every line: an empty tool_calls delta right before it
    ds = list(deltas(on["raw"]))
    for i, d in enumerate(ds):
        if (d.get("reasoning_content") or "").startswith("\n**" + M):
            assert ds[i - 1] == {"tool_calls": []}, ds[i - 1]
    ps = parts(on["raw"])
    titles = [p["title"] for p in ps if p["type"] == "reasoning" and p["title"]]
    assert titles == [M + x for x in lines], (titles, lines)
    # each part opens with its title; the answer's text part comes after the reasoning parts
    kinds = [p["type"] for p in ps]
    assert "text" not in kinds or kinds.index("text") > max(
        i for i, k in enumerate(kinds) if k == "reasoning"), kinds
    assert states.strip_text(message(ps)["reasoning_content"]) == off["reasoning"]
    same_usage(on, off)


def test_2_opencode_tool_call_with_checks_in_order():
    serve(INLINE)
    on = tu.opencode(CALL)
    serve({"states": "off"})
    off = tu.opencode(CALL)
    lines = said(on["raw"])
    assert lines[0] == "deciding: code_edit, xhigh effort", lines
    assert "checking the tool call" in lines and "repairing the tool call" in lines, lines
    assert M not in on["content"] and states.strip_text(on["reasoning"]) == off["reasoning"]
    assert [c.get("function", {}).get("name") for c in on["calls"]] == \
        [c.get("function", {}).get("name") for c in off["calls"]]
    ps = parts(on["raw"])
    assert [p["title"] for p in ps if p["type"] == "reasoning" and p["title"]] == \
        [M + x for x in lines]
    assert [p for p in ps if p["type"] == "tool"], ps
    same_usage(on, off)


def test_2_a_line_then_thinking_then_the_answer():
    """A user report's screenshot of 10:03 ("write an essay short about qwen 27B"): `› deciding: chat,
    low effort`, ~20 s of thinking, the answer. With `content` opencode drew the line and the
    WHOLE answer in one text part on top and "Thought: 20.1s" under it; inline draws the
    thinking (titled by the line) and then the answer."""
    serve(INLINE)
    on = tu.opencode("Write the lexer.")
    ps = parts(on["raw"])
    assert [p["type"] for p in ps] == ["reasoning", "text"], [(p["type"], p.get("title")) for p in ps]
    assert ps[0]["title"] == M + "deciding: code_edit, xhigh effort" and ps[1]["text"] == on["content"]
    assert len(ps[0]["text"]) > len(states.inline(states.frame("deciding: code_edit, xhigh effort")))
    serve(CONTENT)
    bug = parts(tu.opencode("Write the lexer.")["raw"])
    assert [p["type"] for p in bug] == ["text", "reasoning"], "the screenshot's order"
    assert bug[0]["text"].startswith(states.frame("deciding: code_edit, xhigh effort"))


# the fake engine reads its switches in the last user message, which is the essay request here
# (the template renders the tool result in a turn of its own)
FOLLOW = [{"role": "user", "content": "Write a short essay about Qwen 27B. FAKE_SLOW_PREFILL"},
          {"role": "assistant", "content": "", "reasoning_content": "I fetch the model card first.",
           "tool_calls": [{"id": "c1", "type": "function", "function": {
               "name": "bash", "arguments": '{"command": "curl -s hf.co/qwen"}'}}]},
          {"role": "tool", "tool_call_id": "c1", "content": "model card. " * 300}]
SLOW = {"mode": "on", "prefill_every_s": 2}


def follow():
    """opencode's next request after the tool ran: the messages end with the tool result."""
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 2000, "tools": tu.TOOLS,
            "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": tu.SYSTEM}, *FOLLOW]}
    return tu._go(body, {"User-Agent": tu.OPENCODE_UA, "x-session-id": "ses_ui1"})


def test_2_progress_lines_after_a_tool_result_then_the_answer():
    """The report's third case (10:46, live 0.3.19, `content`): after two web fetches the follow-up
    request said `› reading 11,426 tokens at ~820 tok/s, about 12 s left` twice during its prefill,
    then thought 1.1 s and wrote the essay; opencode drew the two lines, the essay, and the
    thought last. With inline the reading lines title reasoning parts, the thinking follows, the
    essay comes last."""
    serve(dict(INLINE, progress=SLOW))
    on = follow()
    lines = said(on["raw"])
    reading = [x for x in lines if x.startswith("reading ")]
    assert len(reading) >= 2, lines
    ps = parts(on["raw"])
    kinds = [p["type"] for p in ps]
    assert kinds[-1] == "text" and set(kinds[:-1]) == {"reasoning"}, kinds
    titles = [p["title"] for p in ps if p["type"] == "reasoning"]
    assert [t for t in titles if t and t.startswith(M + "reading ")] == [M + x for x in reading]
    assert ps[-1]["text"] == on["content"] and M not in on["content"]
    serve(dict(CONTENT, progress=SLOW))
    bug = parts(follow()["raw"])
    assert [p["type"] for p in bug][:1] == ["text"] and bug[-1]["type"] == "reasoning", \
        "the third screenshot's order: the reading lines and the essay on top, the thought last"
    assert len(lines_of(bug[0]["text"])) >= 2


# ------------------------------------------------------------------ 3 replay
def test_3_replayed_parts_reach_the_engine_without_their_lines():
    def second(shape):
        cond, seen = serve(dict(INLINE, progress=PROG))
        first = tu.opencode("FAKE_LOOP Write the lexer.")
        msg = message(parts(first["raw"]))
        assert states.MARKER in msg["reasoning_content"]
        r = {"parts": msg["reasoning_content"], "trimmed": msg["reasoning_content"].strip(),
             "clean": states.strip_text(msg["reasoning_content"])}[shape]
        asst = dict(msg, reasoning_content=r)
        seen.clear()
        out = tu.opencode("Now add a test.",
                          history=[{"role": "user", "content": "FAKE_LOOP Write the lexer."}, asst])
        return seen[-1], out

    p_clean, o_clean = second("clean")
    for shape in ("parts", "trimmed"):
        p, o = second(shape)
        assert p == p_clean, f"{shape}: the engine's prompt is the one without the lines"
        same_usage(o, o_clean)


# ------------------------------------------------------------------ 4-6 the other shapes
def _norm(raw):
    raw = re.sub(r'"id": "[^"]*"', '"id": "x"', raw)
    raw = re.sub(r'"created": \d+', '"created": 0', raw)
    return re.sub(r'"(timings|metrics|usage)": \{.*?\}(?=[,}])', '"x": 0', raw)


def test_4_hermes_inline_is_the_content_channel():
    serve(INLINE)
    # the Hermes agent itself (tags); a plain SDK client gets reasoning_content since SRV-245
    on = tu.hermes("Explain the lexer.", max_reasoning_tokens=12, system=ledger.HERMES_IDENTITY)
    serve(CONTENT)
    ref = tu.hermes("Explain the lexer.", max_reasoning_tokens=12,
                    system=ledger.HERMES_IDENTITY)
    assert on["content"] == ref["content"] and lines_of(on["content"]), on["content"][:200]
    assert '"tool_calls": []' not in on["raw"]


def lines_of(text):
    return [ln[len(M):] for ln in text.split("\n") if ln.startswith(M)]


def test_5_non_streamed_lines_lead_the_content():
    serve(INLINE)
    on = tu.opencode("Write the lexer.", stream=False)
    serve({"states": "off"})
    off = tu.opencode("Write the lexer.", stream=False)
    assert on["content"].startswith(states.frame("deciding: code_edit, xhigh effort"))
    assert on["reasoning"] == off["reasoning"]
    assert states.strip_text(on["content"]) == off["content"]


def test_6_a_waiting_request_opens_with_a_titled_part():
    import test_state_channel as sc
    serve(INLINE)
    real = tu.quiet
    tu.quiet = contextlib.nullcontext
    rt = app.STATE.get("request_timeout")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            busy_body = {"model": "TandemLLM", "stream": True, "max_tokens": 40, "tools": tu.TOOLS,
                         "reasoning_format": "reasoning_content",
                         "messages": [{"role": "user", "content": "FAKE_SLOW hold it"}]}
            busy, _ = sc._bg(busy_body, {"User-Agent": tu.OPENCODE_UA, "x-session-id": "a"})
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
            waiter, box = sc._bg(wait_body, {"User-Agent": tu.OPENCODE_UA, "x-session-id": "b"})
            waiter.join(30)
            busy.join(60)
    finally:
        tu.quiet = real
        app.STATE["queue_timeout"] = 120.0
        app.STATE["request_timeout"] = rt
        qwait.QUIET_S = 2.0
    ds = list(deltas(box["raw"]))
    assert ds[0].get("role") == "assistant" and ds[1] == {"tool_calls": []}, ds[:3]
    assert ds[2] == {"reasoning_content": states.inline(
        states.frame("waiting: another request is running, position 1"))}, ds[2]
    ps = parts(box["raw"])
    assert ps[0]["title"] == M + "waiting: another request is running, position 1", ps


# ------------------------------------------------------------------ 7 opencode's own parser
def _node():
    for c in [os.environ.get("NODE"), shutil.which("node"),
              *sorted(glob.glob(os.path.expanduser("~/.local/node-*/bin/node")))]:
        if c and os.access(c, os.X_OK):
            return c
    return None


def test_7_node_runs_opencodes_parser_on_both_channels():
    node = _node()
    binary = os.environ.get("OPENCODE_BIN") or os.path.expanduser("~/.opencode/bin/opencode")
    if node is None or not os.path.exists(binary):
        print("    (skipped: no node or no opencode binary on this machine)")
        return
    from tools import opencode_bundle
    work = tempfile.mkdtemp(prefix="oc-parts-")
    entry = opencode_bundle.extract(binary, os.path.join(work, "provider"))
    runs = {}
    for name, ui in (("inline", INLINE), ("content", CONTENT)):
        serve(dict(ui, progress=PROG))
        runs[name] = tu.opencode("FAKE_LOOP think on")["raw"]
        runs[name + "-answer"] = tu.opencode("Write the lexer.")["raw"]
        serve(dict(ui, progress=SLOW))
        runs[name + "-follow"] = follow()["raw"]
    for name, raw in runs.items():
        sse = os.path.join(work, f"{name}.sse")
        with open(sse, "w") as f:
            f.write(raw)
        r = subprocess.run([node, os.path.join(ROOT, "tools", "opencode_parts.mjs"), entry, sse],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-2000:]
        got = json.loads(r.stdout)
        want = parts(raw)
        assert [(p["type"], p.get("title"), p.get("text")) for p in got["parts"]] == \
            [(p["type"], p.get("title"), p.get("text")) for p in want], name
        assert got["message"] == message(want), name
        kinds = [p["type"] for p in got["parts"]]
        if name.endswith("-follow"):
            # after a tool result: the prefill's reading lines, the thinking, the answer
            want = (kinds[-1] == "text" and set(kinds[:-1]) == {"reasoning"}
                    and sum(1 for p in got["parts"] if (p.get("title") or "").startswith(M + "reading")) >= 2
                    if name == "inline-follow" else kinds[0] == "text" and kinds[-1] == "reasoning")
            assert want, (name, kinds, [p.get("title") for p in got["parts"]])
            print(f"    node: {name}: {kinds}, titles {[p.get('title') for p in got['parts'] if p.get('title')]}")
            continue
        if name.endswith("-answer"):
            # a line, the thinking, the answer: inline keeps opencode's order, content puts the
            # line and the answer on top (the screenshot of 10:03)
            assert kinds == (["reasoning", "text"] if name == "inline-answer"
                             else ["text", "reasoning"]), (name, kinds)
            print(f"    node: {name}: {kinds}, first title {got['parts'][0].get('title')!r}")
            continue
        lines = said(raw) if name == "inline" else lines_of(
            "".join(d.get("content") or "" for d in deltas(raw)))
        if name == "inline":
            assert [p["title"] for p in got["parts"] if p.get("title")] == [M + x for x in lines]
        else:
            # the screenshot: one text part, made at the first line, on top, with every line
            assert got["parts"][0]["type"] == "text"
            assert lines_of(got["parts"][0]["text"]) == lines and len(lines) >= 2, lines
            assert sum(p["type"] == "reasoning" for p in got["parts"]) >= 2
        print(f"    node: {name}: {len(got['parts'])} parts, "
              f"{sum(p['type'] == 'reasoning' for p in got['parts'])} reasoning, {len(lines)} lines")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
