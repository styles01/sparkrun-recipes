"""SRV-426 (QA of SRV-425): LFM2.5's formats through the real chat handler, on the fake engine in this
process (server/fake_engine.py) with the server's `tool_format` "pythonic", as the LFM2.5 adapter sets
it (engine/models/lfm2_moe/serving.py). The parser and its stream machine alone are
tests/test_toolcall_pythonic.py; this file is what a client holds.

  1  a tool call: the model writes `<|tool_call_start|>[get_weather(city='Berlin', days=3)]<|tool_call_end|>`;
     the response carries one tool_call named get_weather with arguments {"city": "Berlin", "days": 3},
     finish `tool_calls`, and no call text in content                  test_a_tool_call
  2  a streamed tool call: the call arrives as tool_calls deltas, after the content before the
     block, and the stream ends with finish `tool_calls`                test_a_streamed_tool_call
  3  text that is not a call: a block whose list never closes is released as content, finish
     `stop`, no tool_calls, streamed and not                            test_a_malformed_block_is_text
  4  reasoning: the thinking is in reasoning_content (or inline <think> tags when the server's
     reasoning format is tags) and the answer in content, with the call beside it
                                                                        test_reasoning_and_answer_apart
  5  the Qwen path is unchanged: with no tool_format a Pythonic block is text
                                                                        test_without_the_format_the_block_is_text

Run: python tests/test_lfm25_api.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import app  # noqa: E402
from test_toolcheck import Req, serve  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "the weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
OPEN, CLOSE = "<|tool_call_start|>", "<|tool_call_end|>"


def ask(text: str, stream: bool, fmt: str = "reasoning_content", tool_format: str | None = "pythonic",
        think: bool = True) -> dict:
    """One chat request with tools on the fake engine: what the client holds."""
    serve("raw")
    if tool_format:
        app.STATE["tool_format"] = tool_format
    app.STATE["reasoning_format"] = fmt
    # SRV-617: the format named, as an unnamed raw client otherwise gets `reasoning_content`
    body = {"messages": [{"role": "user", "content": text}], "stream": stream, "tools": TOOLS,
            "max_tokens": 2000, "reasoning_format": fmt,
            "chat_template_kwargs": {"enable_thinking": think}}
    head, raw = Req(body).response()
    out = {"content": "", "reasoning": "", "calls": {}, "finish": None, "raw": raw, "order": []}
    if not stream:
        m = json.loads(raw)["choices"][0]
        out["content"] = m["message"].get("content") or ""
        out["reasoning"] = m["message"].get("reasoning_content") or ""
        for i, c in enumerate(m["message"].get("tool_calls") or []):
            out["calls"][i] = {"name": c["function"]["name"], "args": c["function"]["arguments"], "id": c.get("id")}
        out["finish"] = m["finish_reason"]
        return out
    for ln in raw.splitlines():
        if not ln.startswith("data: {"):
            continue
        for ch in json.loads(ln[6:])["choices"]:
            d = ch["delta"]
            if d.get("content"):
                out["content"] += d["content"]
                out["order"].append("content")
            out["reasoning"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                c = out["calls"].setdefault(tc["index"], {"name": "", "args": ""})
                c["name"] += (tc.get("function") or {}).get("name") or ""
                c["args"] += (tc.get("function") or {}).get("arguments") or ""
                out["order"].append("call")
            out["finish"] = ch["finish_reason"] or out["finish"]
    return out


def test_a_tool_call():
    out = ask("what is the weather in Berlin FAKE_PYTHONIC", stream=False)
    assert out["finish"] == "tool_calls", (out["finish"], out["raw"][-400:])
    assert list(out["calls"]) == [0] and out["calls"][0]["name"] == "get_weather"
    assert json.loads(out["calls"][0]["args"]) == {"city": "Berlin", "days": 3}
    assert out["calls"][0]["id"]
    for leak in (OPEN, CLOSE, "get_weather(", "<tool_call>"):
        assert leak not in out["content"], (leak, out["content"])
    assert out["content"].strip() in ("", "Let me look that up."), out["content"]


def test_a_streamed_tool_call():
    out = ask("what is the weather in Berlin FAKE_PYTHONIC", stream=True)
    assert out["finish"] == "tool_calls", (out["finish"], out["raw"][-400:])
    assert list(out["calls"]) == [0] and out["calls"][0]["name"] == "get_weather"
    assert json.loads(out["calls"][0]["args"]) == {"city": "Berlin", "days": 3}
    for leak in (OPEN, CLOSE, "get_weather(", "<tool_call>"):
        assert leak not in out["content"], (leak, out["content"])
    # the content before the block first, the call after it
    assert "call" in out["order"]
    first_call = out["order"].index("call")
    assert "content" not in out["order"][first_call:], out["order"]
    assert out["raw"].rstrip().endswith("data: [DONE]")


def test_a_malformed_block_is_text():
    for stream in (False, True):
        out = ask("what is the weather in Berlin FAKE_PYTHONIC FAKE_PYTHONIC_BAD", stream=stream)
        assert out["finish"] == "stop", (stream, out["finish"])
        assert out["calls"] == {}, (stream, out["calls"])
        assert f"{OPEN}[get_weather(city='Berlin', days=3{CLOSE}" in out["content"], (stream, out["content"])


def test_reasoning_and_answer_apart():
    for stream in (False, True):
        out = ask("what is the weather in Berlin FAKE_PYTHONIC", stream=stream)
        assert out["reasoning"].startswith("The question has"), (stream, out["reasoning"])
        assert "<think>" not in out["content"] and "The question has" not in out["content"]
        assert out["finish"] == "tool_calls"
        out = ask("say something", stream=stream)
        assert out["reasoning"].startswith("The question has") and out["content"].strip()
        assert "The question has" not in out["content"] and out["finish"] == "stop"
        # the tags format: the thinking inline in content, closed before the answer
        out = ask("say something", stream=stream, fmt="tags")
        assert out["reasoning"] == "" and out["content"].lstrip().startswith("<think>"), out["content"][:80]
        assert "</think>" in out["content"] and out["content"].split("</think>", 1)[1].strip()
        # thinking off: no reasoning at all
        out = ask("say something", stream=stream, think=False)
        assert out["reasoning"] == "" and "<think>" not in out["content"] and out["content"].strip()


def test_without_the_format_the_block_is_text():
    for stream in (False, True):
        out = ask("what is the weather in Berlin FAKE_PYTHONIC", stream=stream, tool_format=None)
        assert out["calls"] == {} and out["finish"] == "stop", (stream, out["finish"], out["calls"])
        assert OPEN in out["content"]


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
