"""The real-client smoke set's own logic, on a CPU: what a validating client refuses.

Run: python tests/test_client_smoke.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.client_smoke import OC_TOOLS, WEATHER, filler, read_heartbeat, validate  # noqa: E402,E501


def _oc():
    with open(OC_TOOLS) as f:
        return json.load(f)["tools"]


def test_validate_refuses_what_opencode_refused():
    # the exact 2026-09-26 shapes: a string where the schema asks for an integer / an array
    tools = _oc()
    bad = validate([(0, "c", "read", json.dumps({"filePath": "/a", "offset": "150"}), 1)], tools)
    assert bad and "offset" in bad[0], bad
    bad = validate([(0, "c", "todowrite", json.dumps({"todos": "[]"}), 1)], tools)
    assert bad and "todos" in bad[0], bad
    assert validate([(0, "c", "read", json.dumps({"filePath": "/a", "offset": 150}), 1)],
                    tools) == []


def test_validate_names_json_and_unknown_tools():
    assert "not JSON" in validate([(0, "c", "get_time", '{"city": "Ro', 1)], WEATHER)[0]
    assert "not a tool" in validate([(0, "c", "rm", "{}", 1)], WEATHER)[0]
    assert "required" in validate([(0, "c", "get_weather", '{"city": "Oslo"}', 1)], WEATHER)[0]


def test_filler_is_roughly_the_asked_size_and_never_repeats_a_line():
    text = filler(12000)
    lines = text.splitlines()
    assert len(set(lines)) == len(lines) and 300 < len(lines) < 340


def _ev(**delta):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]})


def _sse(*items):
    """Raw stream lines: every event and comment closed by its blank line, as the server writes."""
    out = []
    for it in items:
        out += [it, ""]
    return out + ["data: [DONE]", ""]


def test_heartbeat_comments_before_the_first_event_pass():
    got = read_heartbeat(_sse(": prefill 2048/24016", ": prefill 4096/24016",
                              _ev(role="assistant", content=""), _ev(content="ok")))
    assert got == {"comments": 2, "events": 2, "done": True, "problems": []}, got


def test_heartbeat_comments_after_step_lines_pass():
    # a test build's smoke (2026-10-02 23:28): S1's progress line opened the stream before the
    # prefill ended, and the old check failed on "a comment after the first event"
    got = read_heartbeat(_sse(_ev(role="assistant", content="\n› deciding: chat, low\n\n"),
                              _ev(content="\n› reading 24,016 tokens at ~750 tok/s, about 30 s "
                                          "left\n\n"),
                              ": prefill 8192/24016", ": prefill 9216/24016",
                              _ev(reasoning_content="\n› thinking 1 min, 900 tokens\n\n"),
                              _ev(content="ok"), _ev(content=".")))
    assert got["problems"] == [] and got["comments"] == 2 and got["done"], got


def test_heartbeat_refuses_a_comment_after_a_token_or_inside_an_event():
    got = read_heartbeat(_sse(_ev(content="ok"), ": prefill 1/2"))
    assert got["problems"] == ["a comment after the model's first token"], got
    got = read_heartbeat(_sse(_ev(content="\n› reading\n\n"),
                              _ev(tool_calls=[{"index": 0, "function": {"name": "read"}}]),
                              ": prefill 1/2"))
    assert got["problems"] == ["a comment after the model's first token"], got
    split = [": prefill 1/2", "", _ev(content="\n› reading\n\n"), ": prefill 2/2", "",
             _ev(content="ok"), "", "data: [DONE]", ""]
    assert read_heartbeat(split)["problems"] == ["a comment inside an event"]
    # a line of the model's own that only contains the marker later is a token
    got = read_heartbeat(_sse(_ev(content="see › here"), ": prefill 1/2"))
    assert got["problems"] == ["a comment after the model's first token"], got


def test_heartbeat_still_wants_an_intact_stream():
    got = read_heartbeat([": prefill 1/2", "", _ev(content="ok"), "", "data: {bad", "",
                          "event: x", ""])
    assert not got["done"] and got["events"] == 1
    assert got["problems"] == ["not JSON 'data: {bad'", "unexpected line 'event: x'"], got


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:62s} ok")
            passed += 1
    print(f"{passed} passed")
