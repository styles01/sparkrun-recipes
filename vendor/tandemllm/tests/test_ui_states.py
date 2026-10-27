"""TandemLLM's own steps shown to the user: the state lines (`ui.states`, 2026-10-02).

Each step the conductor takes is one framed line (`\\n› text\\n\\n`) in the reasoning the client
shows, at the moment it happens; the server takes every framed line out of the reasoning a client
sends back, so Qwen's prompt is the one it would be without them.

The test matrix:

  the frame and the strip (server/conductor/states.py):
    1  a line anywhere in the reasoning (head, end, between two characters of Qwen's text, in two
       parts a client joined) comes out exactly; trimmed by a client, it comes out up to
       whitespace at the ends                         test_the_strip_is_exact_*
    2  the prompt rendered on the checkpoint's template from history with lines equals the one
       without them, token for token: opencode's `reasoning_content`, a client that trims it, the
       `<think>` block in the content of a `tags` client      test_the_prompt_*
  the handler (the fake engine in process, the monitor on, a conductor):
    3  an opencode-shaped stream (`reasoning_content`) carries the router's line before the
       reasoning, and the tool checks' lines when the monitor checks and repairs a call; the
       client's text and usage equal the same request with the lines off
                                                      test_opencode_*
    4  a Hermes-shaped stream (`tags`, the server's default) carries the lines inside `<think>`,
       none after the block closed                    test_hermes_*
    5  a cap close says so: "thinking closed at the N cap"    test_a_cap_*
    6  the reasoning sent back with its lines reaches the engine as the prompt without them, and
       the journal counts what was stripped            test_a_replayed_*
    7  a non-streamed answer: the lines in `reasoning_content`, the content unchanged
                                                      test_non_streamed_*
    8  off (the seed and every profile stored before it), raw, shadow, per-kind switches: no lines
                                                      test_off_*, test_raw_*, test_a_kind_*
    9  the journal row keeps them as `extra.states`, the transcript as a `### states` block
                                                      test_the_journal_and_the_transcript

Run: python tests/test_ui_states.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import app, fake_engine  # noqa: E402
from server.conductor import states, transcripts  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from server.conductor.monitor import Journal, Monitor  # noqa: E402

M = states.MARKER
OPENCODE_UA = "opencode/1.18.33 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"
HERMES_UA = "OpenAI/Python 2.8.1"


# ------------------------------------------------------------------ 1. the strip

REASONINGS = [
    "The user wants a lexer.\nI will read the file first.",
    "Short.",
    "Ends mid-line without a newline",
    "Ends with a newline\n",
    "\nStarts with a newline and has a › sign and ›› in it\nand a line\n› that starts like one",
    "",
]


def test_the_strip_is_exact_wherever_the_frame_sits():
    a, b = states.frame("deciding: code_edit, xhigh effort"), states.frame("checking the tool call")
    for r in REASONINGS:
        cases = [a + r, r + a, a + r + b, a + b + r, r + a + b]
        for i in range(1, len(r)):
            cases.append(r[:i] + a + r[i:])               # between two characters of Qwen's text
        cases.append((a + r[:len(r) // 2]) + (r[len(r) // 2:] + b))  # two parts, joined
        for c in cases:
            assert states.strip_text(c) == r, (r, c)
        # a client that trims the whole reasoning: equal up to whitespace at the ends (the
        # template trims the reasoning it renders)
        for c in (a + r, r + b, a + r + b, a + b):
            assert states.strip_text(c.strip()).strip() == (r if c != a + b else "").strip(), c


def test_the_strip_leaves_text_without_a_frame_alone():
    for r in REASONINGS:
        assert states.strip_text(r) == r
    # a line of Qwen's own that starts with the marker is not a state line: only the templates
    for r in ("x\n› a quote of mine\n\ny", "› my own first line\n\nrest", "end\n› mine"):
        assert states.strip_text(r) == r, r
    msgs = [{"role": "user", "content": M + "a user may write this\n\n"},
            {"role": "assistant", "content": "plain", "reasoning_content": "no frame"}]
    out, n = states.strip_messages(msgs)
    assert out is msgs and n == 0


def test_the_frame_is_one_line():
    f = states.frame("two\nlines and   spaces " + "x" * 400)
    assert f.startswith("\n" + M) and f.endswith("\n\n") and f.count("\n") == 3
    assert len(f) == 3 + len(M) + states.MAX_CHARS


def test_every_step_says_its_template():
    ui = states.States()
    row = {"stages": [{"stage": "qwen_context"}], "class_id": "three_d", "effort": "high"}
    inj = [{"parts": [{"kind": "effort", "name": "high"},
                      {"kind": "package", "name": "photo-to-3d"}]}]
    ui.route(row, inj, {"photo-to-3d": {"kind": "skill"}}, compaction=True)
    ui.route({"stages": [{"stage": "decider"}], "class_id": "summarize", "effort": "off"}, [],
             {}, False)
    ui.route({"stages": [], "class_id": "code_edit", "effort": "xhigh"}, [], {}, False)  # kept
    for ev in ({"decision": "PROMPT_TAIL_NOTE", "task": "S1"},
               {"decision": "PROMPT_TAIL_NOTE", "task": "S2"},
               {"decision": "CLOSE_REASONING", "task": "route", "reason": "budget(16384)"},
               {"decision": "CLOSE_REASONING", "task": "reserve", "reason": "budget(2048)"},
               {"decision": "CLOSE_REASONING", "task": "D2", "reason": "backstop(32768)"},
               {"decision": "CLOSE_REASONING", "task": "D1", "reason": "repeat(0.81)"},
               {"decision": "CLOSE_REASONING", "task": "X9", "reason": "?"},
               {"decision": "CONTINUE_HIDDEN", "task": "E1"},
               {"decision": "CHECK", "task": "SP3", "detail": {"calls": 1}},
               {"decision": "CHECK", "task": "SP3", "detail": {"calls": 3}},
               {"decision": "REPAIR_CALL", "task": "C4"},
               {"decision": "REGENERATE_CALL", "task": "C4"},
               {"decision": "GIVE_UP", "task": "C4"},
               {"decision": "SET_THINK_BUDGET", "task": "D4"}):
        ui.monitor(ev)
    assert [e["text"] for e in ui.log] == [
        "compaction seen", "deciding: three_d, high effort", "retrieving: skill photo-to-3d",
        "deciding: summarize, thinking off",
        "steering: gate progress", "steering: context health",
        "thinking closed at the 16,384 cap", "thinking closed at the 2,048 cap",
        "thinking closed at the 32,768 backstop", "loop detected, steering", "closing reasoning",
        "empty answer, continuing", "checking the tool call", "checking 3 tool calls",
        "repairing the tool call", "asking for the tool call again", "giving up"]
    assert all(e["summary"] is None for e in ui.log), "the clerk's field, not written yet"
    assert ui.take() == [states.frame(e["text"]) for e in ui.log] and ui.take() == []
    # every text the server writes is one the strip knows
    for e in ui.log:
        assert states.strip_text("a" + states.frame(e["text"]) + "b") == "ab", e["text"]
    assert states.TEXTS["memory"].format(n=2) == "reading memory: 2 notes"


def test_a_line_goes_where_the_reasoning_goes():
    from server.stream import Reasoning
    f = states.frame("checking the tool call")
    assert Reasoning("tags").state(f) == [("tagged", f)]
    assert Reasoning("reasoning_content").state(f) == [("reasoning", f)]
    assert Reasoning("both").state(f) == [("reasoning", f), ("tagged", f)]
    for fmt, after in (("tags", []), ("reasoning_content", [("reasoning", f)]),
                       ("both", [("reasoning", f)])):
        r = Reasoning(fmt)
        r.push("thought</think>\n\nthe answer")
        assert not r.in_think and r.state(f) == after, fmt
        assert Reasoning(fmt, in_think=False).state(f) == after, fmt     # thinking off


# ------------------------------------------------------------------ 2. the prompt

def _render(messages):
    from test_conductor_router import setup
    setup()
    b = {"model": "TandemLLM", "messages": messages}
    return app.build_prompt(b)[0].tolist()


def test_the_prompt_from_history_with_lines_is_the_prompt_without():
    r = "The user wants a lexer. I read src/lexer.ts first."
    lines = states.frame("deciding: code_edit, xhigh effort") + \
        states.frame("retrieving: skill ui-design")
    tail = states.frame("thinking closed at the 16,384 cap")
    check = states.frame("checking the tool call")
    sysm = {"role": "system", "content": "You are opencode."}
    user = {"role": "user", "content": "Refactor the lexer."}
    call = [{"id": "c1", "type": "function", "function": {"name": "read",
                                                           "arguments": '{"path": "a"}'}}]
    tool = {"role": "tool", "tool_call_id": "c1", "content": "file text"}
    nxt = {"role": "user", "content": "Go on."}

    def conv(asst):
        return [sysm, user, asst, tool, nxt]

    plain = {"role": "assistant", "content": "", "reasoning_content": r, "tool_calls": call}
    want = _render(conv(plain))
    shapes = {
        # opencode: its reasoning parts joined with nothing between, as `reasoning_content`
        "opencode": dict(plain, reasoning_content=lines + r + tail + check),
        # a client that trims the reasoning it keeps
        "trimmed": dict(plain, reasoning_content=(lines + r + tail).strip()),
        # Hermes keeps the reasoning as `reasoning` beside it (not rendered by the template)
        "hermes": dict(plain, reasoning_content=lines + r, reasoning=lines + r),
    }
    for name, asst in shapes.items():
        msgs, n = states.strip_messages(conv(asst))
        assert n > 0, name
        assert _render(msgs) == want, name
        assert _render(conv(asst)) != want, f"{name}: the lines would have reached the prompt"
    # a `tags` client that sends the content back as it received it
    tagged = {"role": "assistant", "content": "<think>\n" + lines + r + tail + "</think>\n\nDone."}
    plain_t = {"role": "assistant", "content": "<think>\n" + r + "</think>\n\nDone."}
    msgs, n = states.strip_messages(conv(tagged))
    assert n > 0 and _render(msgs) == _render(conv(plain_t))
    # list content with text parts
    parts = {"role": "assistant", "content": [{"type": "text", "text": lines + "Done."}]}
    msgs, n = states.strip_messages(conv(parts))
    assert n > 0 and msgs[2]["content"][0]["text"] == "Done."


# ------------------------------------------------------------------ the handler

TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
SYSTEM = "You are opencode.\n<env>\n  Working directory: /home/user/demos/x\n</env>"


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def serve(ui: str | None = "on", kinds: dict | None = None, mode: str | None = None):
    """The fake engine in this process, the session monitor on, and a conductor whose coding and
    chat profiles have `ui.states` as given (None: the seed's, which has it off)."""
    a = app.parser().parse_args(["--fake-engine", "--fake-tps", "200000", "--max-len", "65536",
                                 "--default-max-tokens", "4000"])
    app.STATE.clear()
    app.CLOCK = app.qwait.TurnClock()          # no hold times from an earlier test
    with quiet():
        fake_engine.load(app, a)
    from test_toolcheck import policy
    app.STATE["monitor"] = Monitor("on", policy(), app.STATE["tok"], Journal(echo=False))
    cond = Conductor(tempfile.mkdtemp(prefix="cond-ui-"), state=app.STATE, journal=False)
    app.STATE["conductor"] = cond
    for name in ("coding", "chat"):
        _, doc = cond.store.get("profiles", name)
        if ui is not None:
            doc["ui"] = {"states": ui, **({"kinds": kinds} if kinds else {})}
        if mode is not None:
            doc["mode"] = mode
        cond.store.put("profiles", name, doc, note="ui states", author="test")
    seen = []
    real = app.generate_stream

    def recording(prompt, *a, **k):
        seen.append(prompt.tolist())
        return real(prompt, *a, **k)

    app.generate_stream = recording
    return cond, seen


class Req(app.Handler):
    def __init__(self, body, headers=None):
        raw = json.dumps(body).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = dict({"Content-Length": str(len(raw))}, **(headers or {}))
        self.path, self.command, self.request_version = "/v1/chat/completions", "POST", "HTTP/1.1"
        self.requestline, self.client_address = "POST /v1/chat/completions", ("127.0.0.1", 0)
        self.close_connection = True

    def response(self):
        with quiet():
            self.do_POST()
        head, _, body = self.wfile.getvalue().partition(b"\r\n\r\n")
        return head.decode(), body.decode()


def opencode(text, history=(), stream=True, **extra):
    body = {"model": "TandemLLM", "stream": stream, "max_tokens": 2000, "tools": TOOLS,
            "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
            "messages": [{"role": "system", "content": SYSTEM}, *history,
                         {"role": "user", "content": text}], **extra}
    if stream:
        body["stream_options"] = {"include_usage": True}
    return _go(body, {"User-Agent": OPENCODE_UA, "x-session-id": "ses_ui1"})


def hermes(text, history=(), system=None, **extra):
    """The Hermes agent: an OpenAI SDK client whose system message opens with its identity
    (server/ledger.py HERMES_IDENTITY), which the server reads as client kind `hermes`. Before
    SRV-245 this sent "You are Hermes." (read as `openai-sdk`), which got the same `tags` then."""
    system = system or ("You are Hermes Agent, built by Nous Research. You are Hermes.")
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 24000, "tools": TOOLS,
            "reasoning_effort": "medium",
            "messages": [{"role": "system", "content": system}, *history,
                         {"role": "user", "content": text}], **extra}
    return _go(body, {"User-Agent": HERMES_UA})


def _go(body, headers):
    head, raw = Req(body, headers).response()
    out = {"content": "", "reasoning": "", "calls": [], "usage": None, "head": head, "raw": raw,
           "order": []}
    if not body.get("stream"):
        m = json.loads(raw)
        msg = m["choices"][0]["message"]
        out.update(content=msg.get("content") or "", reasoning=msg.get("reasoning_content") or "",
                   calls=msg.get("tool_calls") or [], usage=m.get("usage"))
        return out
    for ln in raw.splitlines():
        if not ln.startswith("data: {"):
            continue
        ev = json.loads(ln[6:])
        out["usage"] = ev.get("usage") or out["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                out["content"] += d["content"]
                out["order"].append(("content", d["content"]))
            if d.get("reasoning_content"):
                out["reasoning"] += d["reasoning_content"]
                out["order"].append(("reasoning", d["reasoning_content"]))
            for tc in d.get("tool_calls") or []:
                out["calls"].append(tc)
                out["order"].append(("call", tc))
    return out


def lines(text):
    return [ln[len(M):] for ln in text.split("\n") if ln.startswith(M)]


def same_usage(a, b):
    for k in ("prompt_tokens", "completion_tokens"):
        assert a["usage"][k] == b["usage"][k], (k, a["usage"], b["usage"])


def test_opencode_shaped_stream_shows_the_router_and_the_tool_checks():
    serve("on")
    on = opencode("FAKE_DROP_CLOSER FAKE_CALL_JSON " +
                  json.dumps({"name": "bash", "arguments": {"command": "ls -la"}}))
    serve("off")
    off = opencode("FAKE_DROP_CLOSER FAKE_CALL_JSON " +
                   json.dumps({"name": "bash", "arguments": {"command": "ls -la"}}))
    got = lines(on["reasoning"])
    assert got[0] == "deciding: code_edit, xhigh effort", got
    assert "checking the tool call" in got and "repairing the tool call" in got, got
    # the router's line is the first thing the client gets, before any of Qwen's reasoning
    assert on["order"][0] == ("reasoning", states.frame("deciding: code_edit, xhigh effort"))
    # the check came after the answer's text, as it happened
    kinds = [k for k, _ in on["order"]]
    assert kinds.index("call") > max(i for i, (k, v) in enumerate(on["order"])
                                     if k == "reasoning" and "checking" in v)
    # everything else is the stream without the lines, and the usage is the same
    assert states.strip_text(on["reasoning"]) == off["reasoning"] and M not in off["reasoning"]
    assert on["content"] == off["content"] and on["calls"] == [
        dict(c, id=on["calls"][i]["id"]) if "id" in c else c for i, c in enumerate(off["calls"])]
    same_usage(on, off)
    assert json.loads(on["calls"][0]["function"]["arguments"]) == {"command": "ls -la"}


def test_a_lifted_call_says_so_after_its_own_text():
    """E1's lift (SRV-199): the line goes out after the reasoning that holds the call, never in the
    middle of it. The end token sits in a block with the call's last characters; the decision is
    made before those reach the client, so its line waits for them (live, 2026-10-06: the line
    landed between `<parameter=wait>` and its `</parameter>`)."""
    serve("on")
    out = opencode("list the files FAKE_CALL_IN_THINK")
    line = states.frame("tool call written in the thinking, sending it")
    assert line in out["reasoning"], lines(out["reasoning"])
    r = out["reasoning"]
    assert r.rindex("</tool_call>") < r.index(line), r[-400:]
    assert out["calls"] and out["calls"][0]["function"]["name"] == "bash", out["calls"]
    # the line is the only thing between the call's text and the end of the reasoning
    assert states.strip_text(r).rstrip().endswith("</tool_call>"), r[-200:]


def test_hermes_shaped_stream_puts_the_lines_inside_think():
    serve("on")
    on = hermes("FAKE_DROP_CLOSER FAKE_CALL_JSON " +
                json.dumps({"name": "bash", "arguments": {"command": "ls"}}))
    serve("off")
    off = hermes("FAKE_DROP_CLOSER FAKE_CALL_JSON " +
                 json.dumps({"name": "bash", "arguments": {"command": "ls"}}))
    assert on["reasoning"] == "" and on["content"].startswith("<think>\n" + states.frame(
        "deciding: chat, low effort")), on["content"][:120]
    block, _, after = on["content"].partition("</think>")
    assert lines(block) == ["deciding: chat, low effort"], lines(block)
    assert M not in after, "the checks came after the block closed: no place for them in tags"
    assert states.strip_text(on["content"]) == off["content"]
    same_usage(on, off)


def test_a_cap_close_says_so():
    serve("on")
    on = opencode("Explain the lexer.", max_reasoning_tokens=12)
    got = lines(on["reasoning"])
    assert "thinking closed at the 12 cap" in got, got
    # it is the end of the reasoning: nothing of Qwen's follows it (the call's check may)
    cap = states.frame("thinking closed at the 12 cap")
    i = on["order"].index(("reasoning", cap))
    assert all(v.startswith("\n" + M) for k, v in on["order"][i:] if k == "reasoning"), \
        on["order"][i:]


def test_a_replayed_reasoning_reaches_the_engine_without_its_lines():
    """Turn 1 streams with its lines; turn 2 sends turn 1's reasoning back. With the lines (as
    opencode joins its parts), trimmed by the client, or already without them, the engine gets
    the same prompt, token for token, and the client the same usage."""
    def second(shape):
        cond, seen = serve("on")
        first = opencode("Write the lexer.")
        r = first["reasoning"]
        assert lines(r)
        sent = {"lines": r, "trimmed": r.strip(), "clean": states.strip_text(r)}[shape]
        asst = {"role": "assistant", "content": first["content"], "reasoning_content": sent}
        seen.clear()
        out = opencode("Now add a test.",
                       history=[{"role": "user", "content": "Write the lexer."}, asst])
        rid = json.loads(out["raw"].splitlines()[0][6:])["id"]
        row = [x for x in cond.journal.ring.values() if x.get("request_id") == rid][0]
        return seen[-1], out, row, len(r) - len(states.strip_text(r))

    p_clean, o_clean, row_clean, _ = second("clean")
    for shape in ("lines", "trimmed"):
        p, o, row, n = second(shape)
        assert p == p_clean, f"{shape}: the engine's prompt is the one without the lines"
        same_usage(o, o_clean)
        assert row["extra"].get("states_stripped", 0) > 0, (shape, row["extra"])
        if shape == "lines":
            assert row["extra"]["states_stripped"] == n
    assert "states_stripped" not in row_clean["extra"]


def test_non_streamed_answers_carry_the_lines_in_reasoning_content():
    serve("on")
    on = opencode("Write the lexer.", stream=False)
    serve("off")
    off = opencode("Write the lexer.", stream=False)
    assert lines(on["reasoning"])[0] == "deciding: code_edit, xhigh effort"
    assert on["reasoning"].startswith(states.frame("deciding: code_edit, xhigh effort"))
    assert states.strip_text(on["reasoning"]) == off["reasoning"]
    assert on["content"] == off["content"]
    same_usage(on, off)
    # tags: inside the think block of the content
    serve("on")
    on = opencode("Write the lexer.", stream=False, reasoning_format="tags")
    assert on["content"].startswith("<think>\n" + states.frame("deciding: code_edit, xhigh effort"))


def test_off_is_the_stream_byte_for_byte():
    """The seed (no `ui` in the profile, as every profile stored before it) and `off` stream the
    same bytes, and no line."""
    serve(None)
    a = opencode("Write the lexer.")
    serve("off")
    b = opencode("Write the lexer.")
    assert M not in a["raw"] and M not in b["raw"]
    norm = [_strip_ids(x["raw"]) for x in (a, b)]
    assert norm[0] == norm[1]


def _strip_ids(raw: str) -> str:
    import re
    raw = re.sub(r'"id": "chatcmpl-[0-9a-f]+"', '"id": "x"', raw)
    raw = re.sub(r'"created": \d+', '"created": 0', raw)
    raw = re.sub(r'"(timings|metrics)": \{[^{}]*(\{[^{}]*\}[^{}]*)*\}', '"\\1": {}', raw)
    raw = re.sub(r'"id": "call_[0-9a-zA-Z]+"', '"id": "c"', raw)
    return raw


def test_raw_and_shadow_show_no_line():
    serve("on")
    head, body = Req({"model": "TandemLLM", "stream": True, "max_tokens": 200,
                      "reasoning_format": "reasoning_content",
                      "messages": [{"role": "user", "content": "Write the lexer."}]},
                     {"User-Agent": OPENCODE_UA, "X-Tandem": "raw"}).response()
    assert M not in body and "X-Tandem: raw" in head
    serve("on", mode="shadow")
    assert M not in opencode("Write the lexer.")["raw"]


def test_a_kind_can_be_switched_off():
    serve("on", kinds={"route": False})
    on = opencode("FAKE_DROP_CLOSER FAKE_CALL_JSON " +
                  json.dumps({"name": "bash", "arguments": {"command": "ls -la"}}))
    got = lines(on["reasoning"])
    assert "repairing the tool call" in got and not any(g.startswith("deciding") for g in got)


def test_the_journal_and_the_transcript():
    cond, _ = serve("on")
    out = opencode("Write the lexer.")
    rid = json.loads(out["raw"].splitlines()[0][6:])["id"]
    row = [r for r in cond.journal.ring.values() if r.get("request_id") == rid][0]
    assert [s["text"] for s in row["extra"]["states"]][0] == "deciding: code_edit, xhigh effort"
    assert all(set(s) == {"kind", "text", "at_ms"} for s in row["extra"]["states"])
    job = {"ts": 1790000000, "request_id": rid, "messages": [{"role": "user", "content": "x"}],
           "content": "ok", "calls": [], "states": ["deciding: code_edit, xhigh effort"]}
    text, _ = transcripts.block(job, 1, {"clip_chars": 4000, "loop_chars": 200})
    assert "### states\n› deciding: code_edit, xhigh effort" in text
    job.pop("states")
    text, _ = transcripts.block(job, 1, {"clip_chars": 4000, "loop_chars": 200})
    assert "### states" not in text


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
