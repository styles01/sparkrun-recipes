"""SRV-477 (QA SRV-478): `--reasoning-effort none|off` starts the server and turns thinking off for
a request that says nothing about thinking; one that says anything keeps it; low, medium and
xhigh are unchanged.

  1  the server starts with none or off                 test_1_*
  2  a request that says nothing gets thinking off       test_2_*
  3  a request that says something keeps it             test_3_*
  4  the old values are unchanged                        test_4_*

The requests run through the real handler with the fake engine in this process; what the model saw
is the engine's context (the template's thinking off renders `<think>\\n\\n</think>`).

Run: python tests/test_reasoning_effort_off.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import app, fake_engine, thinking  # noqa: E402

OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
ON = "<|im_start|>assistant\n<think>\n"


def args(*extra):
    a = app.parser().parse_args(["--fake-engine", "--fake-tps", "200000", *extra])
    app._thinking_default(a)
    return a


def serve(*extra):
    a = args(*extra)
    app.STATE.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        fake_engine.load(app, a)
    app.STATE["monitor"] = None
    app.STATE["thinking_default"] = a.thinking_default
    return a


class Req(app.Handler):
    def __init__(self, body):
        raw = json.dumps(body).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = {"Content-Length": str(len(raw))}
        self.path, self.command, self.request_version = "/v1/chat/completions", "POST", "HTTP/1.1"
        self.requestline, self.client_address = "POST /v1/chat/completions", ("127.0.0.1", 0)
        self.close_connection = True


def seen(**extra) -> str:
    """The prompt the model read for one request, its generation prompt's tail."""
    body = dict({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8}, **extra)
    with contextlib.redirect_stdout(io.StringIO()):
        Req(body).do_POST()
    text = app.STATE["tok"].decode(app.STATE["last_ctx"], skip_special_tokens=False)
    return text[text.rfind("<|im_start|>assistant"):]


def test_1_the_server_starts_with_none_or_off():
    for word in ("none", "off"):
        a = args("--reasoning-effort", word)
        assert (a.reasoning_effort, a.thinking_default) == (None, "off"), word
    for word in ("high", "minimal", "max"):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                args("--reasoning-effort", word)
        except SystemExit:
            continue
        raise AssertionError(f"{word} was accepted")


def test_2_a_request_that_says_nothing_gets_thinking_off():
    serve("--reasoning-effort", "none")
    assert seen().startswith(OFF)
    # a `reasoning` object that says nothing about the switch says nothing
    assert seen(reasoning={"exclude": True}).startswith(OFF)
    assert not thinking.said({"messages": [], "reasoning_effort": ""})
    body = {"messages": [], "chat_template_kwargs": {"foo": 1}}
    out = thinking.server_default_off(body)
    assert out["chat_template_kwargs"] == {"foo": 1, "enable_thinking": False}
    assert body["chat_template_kwargs"] == {"foo": 1}, "the client's body is not changed"


def test_3_a_request_that_says_something_keeps_it():
    serve("--reasoning-effort", "off")
    for extra in ({"reasoning_effort": "medium"},
                  {"chat_template_kwargs": {"enable_thinking": True}},
                  {"enable_thinking": True}, {"max_reasoning_tokens": 512},
                  {"reasoning": {"effort": "low"}}, {"thinking": {"type": "enabled"}}):
        assert thinking.said(extra), extra
        assert seen(**extra).startswith(ON), extra
    assert seen(reasoning_effort="none").startswith(OFF)


def test_4_the_old_values_are_unchanged():
    for word in ("low", "medium", "xhigh"):
        a = serve("--reasoning-effort", word)
        assert (a.reasoning_effort, a.thinking_default) == (word, "on")
        assert seen().startswith(ON), word
    a = serve()
    assert (a.reasoning_effort, a.thinking_default) == (None, "on")
    assert seen().startswith(ON)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:                               # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, exc)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
