"""SRV-199 (QA SRV-200): a tool call written inside the thinking does not end the turn silently.

On the fake engine in this process (server/fake_engine.py), through the real handler:

  1  the reasoning ends with a complete call to an offered tool and the end token follows inside
     it: E1's case C forwards the model's own call tokens as the answer -- the client gets that
     call (finish `tool_calls`), every reasoning format, streamed and not
  2  the same with the reasoning closed after the call and an empty answer
  3  a call to a tool the request does not offer is not lifted: case A as before
  4  no monitor (raw) and a shadow monitor: nothing changes (finish `stop`, no call)
  5  the scan itself (`_trailing_calls`): calls followed by text, a call cut short, a closer
     without its opener, two calls in a row

Run: python tests/test_call_in_think.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import app, fake_engine  # noqa: E402
from test_toolcheck import Req, policy, quiet, serve  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "run_command", "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}]


def ask(text: str, stream: bool = True, fmt: str = "tags") -> dict:
    """One chat request with thinking on: what the client holds."""
    app.STATE["reasoning_format"] = fmt
    # SRV-617: the format named, as an unnamed raw client otherwise gets `reasoning_content`
    body = {"messages": [{"role": "user", "content": text}], "stream": stream,
            "tools": TOOLS, "max_tokens": 2000, "reasoning_format": fmt}
    head, raw = Req(body).response()
    out = {"content": "", "reasoning": "", "calls": {}, "finish": None, "raw": raw}
    if not stream:
        m = json.loads(raw)["choices"][0]
        out["content"] = m["message"].get("content") or ""
        out["reasoning"] = m["message"].get("reasoning_content") or ""
        for i, c in enumerate(m["message"].get("tool_calls") or []):
            out["calls"][i] = {"name": c["function"]["name"], "args": c["function"]["arguments"]}
        out["finish"] = m["finish_reason"]
        return out
    for ln in raw.splitlines():
        if not ln.startswith("data: {"):
            continue
        for ch in json.loads(ln[6:])["choices"]:
            d = ch["delta"]
            out["content"] += d.get("content") or ""
            out["reasoning"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                c = out["calls"].setdefault(tc["index"], {"name": "", "args": ""})
                c["name"] += (tc.get("function") or {}).get("name") or ""
                c["args"] += (tc.get("function") or {}).get("arguments") or ""
            out["finish"] = ch["finish_reason"] or out["finish"]
    return out


def _e1_policy():
    return policy(only=("E1",))


def test_1_a_call_inside_the_open_reasoning_is_lifted():
    for fmt in ("tags", "reasoning_content"):
        for stream in (True, False):
            mon = serve("on", _e1_policy())
            out = ask("list the files FAKE_CALL_IN_THINK", stream=stream, fmt=fmt)
            acts = mon.journal.actions()
            assert [a["reason"] for a in acts] == ["empty_turn(case=C)"], (fmt, stream, acts)
            assert out["finish"] == "tool_calls", (fmt, stream, out["finish"], out["raw"][-400:])
            assert list(out["calls"]) == [0] and out["calls"][0]["name"] == "run_command"
            assert json.loads(out["calls"][0]["args"])["query"].startswith("list the files")
            # the answer carries the call as a call, never as text (with `tags` the reasoning,
            # which still holds the model's thought, is in the content before `</think>`)
            answer = out["content"].split("</think>")[-1]
            assert "<tool_call>" not in answer, (fmt, stream, out["content"])
            if fmt == "tags":
                assert out["content"].count("</think>") == 1, out["content"]
            else:
                assert "<tool_call>" in out["reasoning"] and "</think>" not in out["content"]


def test_2_a_call_before_the_close_with_an_empty_answer_is_lifted():
    for stream in (True, False):
        mon = serve("on", _e1_policy())
        out = ask("list the files FAKE_CALL_IN_THINK FAKE_CALL_IN_THINK_CLOSED", stream=stream,
                  fmt="reasoning_content")
        acts = mon.journal.actions()
        assert [a["reason"] for a in acts] == ["empty_turn(case=C)"], acts
        assert out["finish"] == "tool_calls" and out["calls"][0]["name"] == "run_command"


def test_3_a_tool_the_request_does_not_offer_is_not_lifted():
    mon = serve("on", _e1_policy())
    out = ask("list the files FAKE_CALL_IN_THINK FAKE_CALL_IN_THINK_OTHER", fmt="tags")
    acts = mon.journal.actions()
    assert acts and acts[0]["reason"] == "empty_turn(case=A)", acts
    # case A's continuation: the fake writer answers a hidden action with a call to the first tool
    assert out["finish"] == "tool_calls" and out["calls"][0]["name"] == "run_command"


def test_4_without_an_acting_monitor_nothing_changes():
    for mode in ("raw", "shadow"):
        mon = serve(mode, _e1_policy() if mode != "raw" else None)
        out = ask("list the files FAKE_CALL_IN_THINK", fmt="tags")
        assert out["finish"] == "stop" and not out["calls"], (mode, out["finish"])
        assert "<tool_call>" in out["content"] and "</think>" not in out["content"]
        if mon is not None:
            rows = [e for e in mon.journal.events() if e.get("task") == "E1"]
            assert rows and rows[0]["mode"] == "shadow", rows


def _scan(text: str, close_id=None):
    tok = app.STATE["tok"]
    ids = tok.encode_text(text)
    got = app._trailing_calls(ids, close_id)
    return None if got is None else tok.decode(got, skip_special_tokens=False)


def test_5_the_scan():
    serve("raw")
    app._CALL_MARKS.clear()
    call = fake_engine._call("run_command", "ls")
    assert _scan("thinking about it\n" + call + "\n") == call
    assert _scan("a\n" + call + "\n\n" + call + "  \n") == call + "\n\n" + call
    assert _scan("a " + call + "\n</think>\n\n", fake_engine.THINK_CLOSE) == call
    assert _scan(call + "\nand then I will check the output") is None, "text after the call"
    assert _scan("<tool_call>\n<function=run_command>\n<parameter=query>\nls") is None, "cut"
    assert _scan("no call here </tool_call>\n") is None, "a closer without its opener"
    assert _scan("nothing at all") is None and _scan("") is None
    # a call the model only discussed, then an answer: the answer is not empty, so E1 is not
    # asked at all (SP2 scans only an empty answer); the scan alone also says no
    assert _scan(call + "\n</think>\n\nHere is the answer.", fake_engine.THINK_CLOSE) is None


def test_6_the_state_line_says_what_happened():
    """The lift's line, under the `continue` switch: not "asking it to go on"."""
    from server.conductor import states as S, voice as V
    ev = {"decision": "CONTINUE_HIDDEN", "task": "E1", "reason": "empty_turn(case=C)",
          "detail": {"case": "C", "calls": ["run_command"], "act": 1}}
    st = S.States()
    st.monitor(ev)
    assert [(e["kind"], e["text"]) for e in st.log] == [
        ("continue", "tool call written in the thinking, sending it")], st.log
    assert "lift" not in S.KINDS, "the switch is continue's"
    off = S.States(kinds={"continue": False})
    off.monitor(ev)
    assert off.log == [], "continue off: no line for the lift either"
    f = st.log[0]["f"] if "f" in st.log[0] else {"lift": True, "tools": ["run_command"]}
    assert V.natural("continue", f, "en") == "the run_command call was written in the thinking, sending it"
    assert V.natural("continue", f, "de").startswith("der run_command-Aufruf stand im Denken")
    assert V.natural("continue", {}, "en") == "stopped without an answer, asking it to go on"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with quiet():
                fn()
            n += 1
            print("ok", name)
    print(f"{n} passed")
