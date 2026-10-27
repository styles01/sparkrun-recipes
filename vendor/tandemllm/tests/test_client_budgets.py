"""A client's thinking budget in every request shape, and the answer reserve, on a CPU.

The Hermes agent (Nous Research's hermes-agent, its `custom` provider for an OpenAI-compatible
endpoint) sends top-level `reasoning_effort` -- `medium` when the user set none, `none` for
thinking off -- and `max_tokens` when one is configured. The server read that word as a template
sentence only, so the reasoning ran until the output limit ended the turn inside the block, and
Hermes showed "Thinking Budget Exhausted" with no answer. server/thinking.py reads every shape
into one budget; server/app.py closes the reasoning early enough to leave the answer a reserve.

The test matrix:

  1  every shape reads into one budget                 test_each_shape_*
  2  the exact Hermes requests                         test_hermes_*
  3  the precedence                                    test_precedence_*
  4  thinking off, in every form                       test_off_*
  5  bad values are refused, naming the field          test_bad_values_*
  6  the answer reserve                                test_the_reserve_*
  7  a request with no budget field is untouched       test_no_field_*, and
                                                       tests/test_serve_identity.py (the loop)
  8  the close is graceful on every path               test_over_the_handler_*
  9  the router: a number wins, a word yields          test_the_router_*

Run: python tests/test_client_budgets.py
"""

from __future__ import annotations

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import test_supervisor_frame as frame  # noqa: E402  (sets the CPU kernel switches first)
from test_app_loop import Req, _deltas, serve  # noqa: E402
from test_conductor_router import CODE, SYSTEM, QTok, by_label, setup, turn  # noqa: E402
from test_conductor_router import body as rbody  # noqa: E402

from engine.spec import ThinkBudget  # noqa: E402
from server import app, compat, logbuf, thinking  # noqa: E402

PHRASE = "Considering the limited time by the user"
P = len(frame.MonTok().encode(ThinkBudget.PHRASE))     # the closing phrase, in test tokens
TOOLS = [{"type": "function", "function": {
    "name": "terminal", "description": "Run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]


def hermes(effort: str | None = "medium", max_tokens: int | None = None, **extra) -> dict:
    """What hermes-agent sends to a `custom` (OpenAI-compatible) provider, built as its
    `CustomProfile.build_api_kwargs_extras` and `ChatCompletionsTransport` build it (main,
    2026-10-01): model, messages, tools, top-level `reasoning_effort` (the agent's level, `medium`
    when unset, `none` when disabled), `max_tokens` only when the agent has one, streamed."""
    b = {"model": "TandemLLM", "stream": True,
         "messages": [{"role": "system", "content": "You are Hermes."},
                      {"role": "user", "content": "hello"}],
         "tools": TOOLS}
    if effort is not None:
        b["reasoning_effort"] = effort
    if max_tokens is not None:
        b["max_tokens"] = max_tokens
    b.update(extra)
    return b


def _read(b: dict) -> thinking.Client:
    return thinking.read(b)


# ------------------------------------------------------------------ 1. every shape

def test_each_shape_reads_into_one_budget():
    want = [
        ({"max_reasoning_tokens": 777}, 777, "number", "max_reasoning_tokens"),
        ({"thinking_budget": 500}, 500, "number", "thinking_budget"),
        ({"chat_template_kwargs": {"thinking_budget": 300}}, 300, "number",
         "chat_template_kwargs.thinking_budget"),
        ({"chat_template_kwargs": {"max_thinking_tokens": 301}}, 301, "number",
         "chat_template_kwargs.max_thinking_tokens"),
        ({"chat_template_kwargs": {"reasoning_budget": 302}}, 302, "number",
         "chat_template_kwargs.reasoning_budget"),
        ({"chat_template_kwargs": {"thinking_token_budget": 303}}, 303, "number",
         "chat_template_kwargs.thinking_token_budget"),
        ({"reasoning": {"max_tokens": 2000}}, 2000, "number", "reasoning.max_tokens"),
        ({"reasoning": {"max_tokens": 2000, "exclude": True}}, 2000, "number",
         "reasoning.max_tokens"),
        ({"thinking": {"type": "enabled", "budget_tokens": 4000}}, 4000, "number",
         "thinking.budget_tokens"),
        ({"reasoning_budget": 11}, 11, "number", "reasoning_budget"),
        ({"max_thinking_tokens": 12}, 12, "number", "max_thinking_tokens"),
        ({"thinking_token_budget": 13}, 13, "number", "thinking_token_budget"),
        ({"thinking_budget_tokens": 14}, 14, "number", "thinking_budget_tokens"),
        ({"reasoning": {"enabled": True}}, 8192, "switch", "reasoning.enabled"),
        ({"thinking": {"type": "adaptive"}}, 0, "switch", "thinking.type"),
        ({"thinking": {"type": "enabled"}}, 0, "switch", "thinking.type"),
    ]
    for b, budget, kind, source in want:
        c = _read(b)
        assert (c.budget, c.kind, c.source) == (budget, kind, source), (b, c)
        assert c.asked and not c.off, b
    assert _read({"reasoning": {"max_tokens": 1, "exclude": True}}).exclude


def test_each_effort_word_is_its_level_and_cap():
    want = {"minimal": ("low", 1024), "low": ("low", 1024), "medium": ("medium", 8192),
            "high": ("high", 16384), "xhigh": ("xhigh", 0), "max": ("xhigh", 0),
            " High ": ("high", 16384)}
    for where in ("top", "reasoning", "kwargs"):
        for word, (level, cap) in want.items():
            b = ({"reasoning_effort": word} if where == "top" else
                 {"reasoning": {"effort": word}} if where == "reasoning" else
                 {"chat_template_kwargs": {"reasoning_effort": word}})
            c = _read(b)
            assert (c.effort, c.budget, c.kind) == (level, cap, "effort"), (b, c)
    # the template knows low, medium and xhigh; `high` leaves it at the server's default
    assert [_read({"reasoning_effort": w}).template_effort
            for w in ("minimal", "low", "medium", "high", "xhigh", "max")] == \
        ["low", "low", "medium", None, "xhigh", "xhigh"]


def test_each_shape_follows_the_router_documents_levels():
    levels = {"low": {"thinking": True, "budget": 500}, "medium": {"thinking": True,
                                                                   "budget": 6000},
              "off": {"thinking": False, "budget": 0}}
    caps = thinking.caps(levels)
    assert caps == {"low": 500, "medium": 6000}
    assert thinking.read({"reasoning_effort": "medium"}, caps).budget == 6000
    assert thinking.caps() == {"low": 1024, "medium": 8192, "high": 16384, "xhigh": 0}


# ------------------------------------------------------------------ 2. Hermes

def test_hermes_default_is_medium_and_its_cap():
    c = _read(hermes())
    assert (c.budget, c.kind, c.source, c.effort) == (8192, "effort", "reasoning_effort",
                                                      "medium")
    for word, cap in (("low", 1024), ("minimal", 1024), ("high", 16384), ("xhigh", 0),
                      ("max", 0)):
        assert _read(hermes(word)).budget == cap, word


def test_hermes_none_is_thinking_off():
    c = _read(hermes("none"))
    assert c.off and not c.asked and c.source == "reasoning_effort"


def test_hermes_renders_on_the_real_template():
    """`high`, `none`, `minimal` and `max` used to reach the template, which raises on any word
    but xhigh, medium and low: a 500. They render now."""
    setup()
    seen = {}
    for word in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        ids, _, in_think = app.build_prompt(hermes(word))
        text = QTok().decode(ids.tolist(), skip_special_tokens=False)
        seen[word] = (in_think, "Reasoning effort is set to low" in text,
                      "Reasoning effort is set to xhigh" in text)
    assert seen == {"none": (False, False, False), "minimal": (True, True, False),
                    "low": (True, True, False), "medium": (True, False, False),
                    "high": (True, False, False), "xhigh": (True, False, True),
                    "max": (True, False, True)}, seen


# ------------------------------------------------------------------ 3. precedence

def test_precedence_a_number_beats_a_word_beats_a_switch():
    c = _read({"reasoning_effort": "high", "reasoning": {"max_tokens": 300, "enabled": True}})
    assert (c.budget, c.source) == (300, "reasoning.max_tokens")
    c = _read({"reasoning": {"effort": "low", "enabled": True}})
    assert (c.budget, c.source) == (1024, "reasoning.effort")


def test_precedence_among_numbers_and_words():
    c = _read({"max_reasoning_tokens": 1, "thinking_budget": 2, "reasoning": {"max_tokens": 3},
               "thinking": {"type": "enabled", "budget_tokens": 4},
               "chat_template_kwargs": {"thinking_budget": 5}, "reasoning_budget": 6})
    assert (c.budget, c.source) == (1, "max_reasoning_tokens")
    c = _read({"thinking_budget": 2, "reasoning": {"max_tokens": 3},
               "chat_template_kwargs": {"thinking_budget": 5}})
    assert (c.budget, c.source) == (2, "thinking_budget")
    c = _read({"reasoning": {"max_tokens": 3}, "chat_template_kwargs": {"thinking_budget": 5},
               "thinking": {"type": "enabled", "budget_tokens": 4}})
    assert (c.budget, c.source) == (5, "chat_template_kwargs.thinking_budget")
    c = _read({"reasoning": {"max_tokens": 3}, "thinking": {"type": "enabled",
                                                            "budget_tokens": 4},
               "reasoning_budget": 6})
    assert (c.budget, c.source) == (3, "reasoning.max_tokens")
    c = _read({"reasoning": {"effort": "high"}, "reasoning_effort": "medium",
               "chat_template_kwargs": {"reasoning_effort": "low"}})
    assert (c.budget, c.source) == (1024, "chat_template_kwargs.reasoning_effort")
    c = _read({"reasoning": {"effort": "high"}, "reasoning_effort": "medium"})
    assert (c.budget, c.source) == (8192, "reasoning_effort")


def test_precedence_the_templates_switch_decides_on_or_off():
    c = _read({"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "none",
               "max_reasoning_tokens": 50})
    assert c.thinking is True and (c.budget, c.source) == (50, "max_reasoning_tokens")
    c = _read({"chat_template_kwargs": {"enable_thinking": False}, "max_reasoning_tokens": 50})
    assert c.off and not c.asked
    c = _read({"enable_thinking": True, "reasoning": {"enabled": False}})
    assert c.thinking is True and not c.asked
    # an off form beats a number
    c = _read({"reasoning": {"enabled": False, "max_tokens": 900}})
    assert c.off and not c.asked


def test_precedence_zero_is_no_cap_but_asked():
    c = _read({"max_reasoning_tokens": 0})
    assert c.asked and c.budget == 0 and c.kind == "number"


# ------------------------------------------------------------------ 4. thinking off

OFF_FORMS = ({"chat_template_kwargs": {"enable_thinking": False}}, {"enable_thinking": False},
             {"reasoning": {"enabled": False}}, {"reasoning": {"effort": "none"}},
             {"reasoning_effort": "none"}, {"reasoning_effort": "NONE"},
             {"thinking": {"type": "disabled"}}, {"thinking": False},
             {"chat_template_kwargs": {"reasoning_effort": "none"}})


def test_off_every_form_turns_thinking_off():
    for b in OFF_FORMS:
        c = _read(b)
        assert c.off and not c.asked and c.budget is None, (b, c)


def test_off_every_form_renders_the_templates_thinking_off():
    setup()
    msgs = [{"role": "user", "content": "hello"}]
    off_ids = app.build_prompt({"messages": msgs, "chat_template_kwargs": {
        "enable_thinking": False}})[0].tolist()
    for b in OFF_FORMS:
        ids, _, in_think = app.build_prompt(dict(b, messages=msgs))
        assert not in_think and ids.tolist() == off_ids, b


# ------------------------------------------------------------------ 5. refusals

def test_bad_values_are_refused_naming_the_field():
    bad = [({"reasoning_effort": "turbo"}, "reasoning_effort"),
           ({"reasoning_effort": 3}, "reasoning_effort"),
           ({"reasoning": {"effort": "fast"}}, "reasoning.effort"),
           ({"reasoning": {"max_tokens": -1}}, "reasoning.max_tokens"),
           ({"reasoning": {"max_tokens": "many"}}, "reasoning.max_tokens"),
           ({"reasoning": {"enabled": "yes"}}, "reasoning.enabled"),
           ({"reasoning": "high"}, "reasoning"),
           ({"thinking": {"type": "sometimes"}}, "thinking.type"),
           ({"thinking": {"type": "enabled", "budget_tokens": True}}, "thinking.budget_tokens"),
           ({"max_reasoning_tokens": -5}, "max_reasoning_tokens"),
           ({"max_reasoning_tokens": 1 << 21}, "max_reasoning_tokens"),
           ({"chat_template_kwargs": {"enable_thinking": "no"}},
            "chat_template_kwargs.enable_thinking")]
    for b, param in bad:
        try:
            _read(b)
        except compat.Refusal as exc:
            assert exc.param == param, (b, exc.param)
        else:
            raise AssertionError(f"not refused: {b}")
        # the router reads leniently, after the handler refused
        thinking.read(b, strict=False)
    # what used to work still does: a numeric string, an empty word
    assert _read({"max_reasoning_tokens": "1536"}).budget == 1536
    assert not _read({"reasoning_effort": ""}).asked


def test_bad_values_are_a_400_over_the_handler():
    serve(None)
    app.STATE["tok"] = frame.MonTok()
    with frame.quiet():
        head, raw = Req("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "turbo"}).response()
    assert " 400 " in head.splitlines()[0], head
    assert json.loads(raw)["error"]["param"] == "reasoning_effort"


# ------------------------------------------------------------------ 6. the reserve

def test_the_reserve_is_min_2048_and_a_quarter():
    assert thinking.reserve(24000) == 2048 and thinking.reserve(4000) == 1000
    assert thinking.reserve(32768, 0) == 0 and thinking.reserve(32768, 512) == 512
    assert thinking.parse_reserve("auto") == thinking.AUTO == -1
    assert thinking.parse_reserve("700") == 700 and thinking.parse_reserve(" Auto ") == -1
    for bad in ("-3", "lots"):
        try:
            thinking.parse_reserve(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_the_reserve_lowers_a_cap_that_would_eat_the_answer():
    assert thinking.with_reserve(8192, 24000) == (8192, 2048)       # leaves the reserve
    assert thinking.with_reserve(0, 24000) == (21952, 2048)         # no cap: limit - reserve
    assert thinking.with_reserve(23000, 24000) == (21952, 2048)
    assert thinking.with_reserve(8192, 4000) == (3000, 1000)
    assert thinking.with_reserve(500, 4000, 0) == (500, 0)          # reserve off
    assert thinking.with_reserve(0, 4000, 0) == (0, 0)
    assert thinking.with_reserve(0, 3, 10) == (1, 10)               # never below one token
    # the closing phrase is the client's too where it streams: the answer's room is after it
    assert thinking.with_reserve(0, 24000, closer=25) == (21927, 2048)
    assert thinking.with_reserve(8192, 9000, closer=25) == (6927, 2048)
    assert thinking.with_reserve(500, 4000, 0, closer=25) == (500, 0)  # reserve off: no change


def test_the_reserve_flag_parses():
    a = app.parser().parse_args([])
    assert a.answer_reserve == "auto"
    a = app.parser().parse_args(["--answer-reserve", "1024"])
    assert thinking.parse_reserve(a.answer_reserve) == 1024


# ------------------------------------------------------------------ 7. no field, no change

def _legacy_kwargs(b: dict, server_effort) -> dict:
    """build_prompt's template arguments before server/thinking.py, line for line."""
    kwargs = dict(b.get("chat_template_kwargs") or {})
    kwargs.setdefault("enable_thinking", True)
    if b.get("tools"):
        kwargs["tools"] = b["tools"]
    effort = b.get("reasoning_effort") or server_effort
    if effort:
        kwargs.setdefault("reasoning_effort", effort)
    return kwargs


def test_no_field_renders_byte_identical_on_the_real_template():
    """Every body a client could send before -- no budget field, or the template's own words
    and switch -- renders the same ids as before, at each server effort."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE}]
    bodies = [{}, {"tools": TOOLS}, {"chat_template_kwargs": {"enable_thinking": False}},
              {"chat_template_kwargs": {"enable_thinking": True}},
              {"reasoning_effort": "low"}, {"reasoning_effort": "medium"},
              {"reasoning_effort": "xhigh", "tools": TOOLS},
              {"chat_template_kwargs": {"reasoning_effort": "low"}},
              {"chat_template_kwargs": {"reasoning_effort": "medium"}, "reasoning_effort": "low"},
              {"max_tokens": 100, "temperature": 0.3}]
    for server_effort in (None, "low", "medium", "xhigh"):
        setup(effort=server_effort)
        tok = app.STATE["tok"]
        for extra in bodies:
            b = dict(extra, messages=msgs)
            legacy = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             **_legacy_kwargs(b, server_effort))["input_ids"][0]
            assert app.build_prompt(b)[0].tolist() == legacy.tolist(), (server_effort, extra)


def test_no_field_reads_as_nothing_asked():
    for b in ({}, {"messages": []}, {"max_tokens": 50}, {"chat_template_kwargs": {}},
              {"chat_template_kwargs": {"enable_thinking": True}}, {"enable_thinking": True},
              {"reasoning": {"exclude": True}}, {"reasoning_effort": None}):
        c = _read(b)
        assert not c.asked and c.budget is None and not c.off, (b, c)


class _Spy:
    """Wraps generate_stream: the budget and limit the loop was handed, the ids it yielded."""

    def __init__(self):
        self.real, self.calls = app.generate_stream, []

    def __call__(self, prompt, max_new, eos, think, *a, **k):
        call = {"budget": think.budget, "max_new": max_new, "prompt": prompt.tolist(),
                "sup": k.get("sup"), "pen": k.get("pen"), "ids": []}
        self.calls.append(call)
        for t in self.real(prompt, max_new, eos, think, *a, **k):
            call["ids"].append(t)
            yield t

    def __enter__(self):
        app.generate_stream = self
        return self

    def __exit__(self, *exc):
        app.generate_stream = self.real


def _handler(b: dict, mode=None, cond=None, reserve=None, headers=None, think_budget=0):
    serve(None)
    app.STATE.update(tok=frame.MonTok(), think_budget=think_budget)
    if reserve is not None:
        app.STATE["answer_reserve"] = reserve
    if cond is not None:
        app.STATE["conductor"] = cond
    if mode is not None:
        app.STATE["monitor"] = frame.monitor(mode)
    req = Req("/v1/chat/completions", b)
    req.headers.update(headers or {})
    out = io.StringIO()
    with _Spy() as spy, logbuf_capture(out):
        _, raw = req.response()
    return raw, spy.calls, out.getvalue()


def logbuf_capture(out):
    import contextlib
    return contextlib.redirect_stdout(out)


def test_no_field_hands_the_loop_what_it_had():
    """No budget field: the loop gets the server's budget and the request's limit, no monitor
    cap, no client-budget line -- on raw, with a monitor on, and with the server's own --think-budget."""
    msgs = [{"role": "user", "content": "hello"}]
    for mode, tb in ((None, 0), ("on", 0), (None, 7), ("on", 7)):
        for b in ({"messages": msgs, "max_tokens": 40},
                  {"messages": msgs, "max_tokens": 40,
                   "chat_template_kwargs": {"enable_thinking": True}}):
            raw, calls, log = _handler(b, mode=mode, think_budget=tb)
            assert len(calls) == 1 and calls[0]["budget"] == tb and calls[0]["max_new"] == 40
            sup = calls[0]["sup"]
            assert sup is None or sup.think_budget is None, mode
            assert "client budget" not in log, log


def test_no_field_x_tandem_raw_is_the_raw_loop():
    msgs = [{"role": "user", "content": "hello"}]
    raw, calls, _ = _handler({"messages": msgs, "max_tokens": 40}, mode="on",
                             headers={"X-Tandem": "raw"})
    assert calls[0]["sup"] is None and calls[0]["budget"] == 0
    # the same request with no monitor at all yields the same tokens
    raw2, calls2, _ = _handler({"messages": msgs, "max_tokens": 40})
    assert calls[0]["ids"] == calls2[0]["ids"]


# ------------------------------------------------------------------ 8. the close, over the handler

def _reasoning_count(ids):
    return ids.index(frame.CLOSE) + 1 if frame.CLOSE in ids else len(ids)


def test_over_the_handler_raw_hermes_closes_with_the_phrase_and_answers():
    """No monitor (raw): ThinkBudget's phrase streams, then `</think>` and the answer. A reserve
    of 20 out of 160, after the phrase's P tokens, lowers the effort word's 8,192."""
    # SRV-617: the tags named (this body reads as an unnamed client, not as `hermes`)
    b = hermes("medium", 160, stream=False, reasoning_format="tags")
    raw, calls, log = _handler(b, reserve=20)
    got = json.loads(raw)
    cap = 160 - 20 - P
    assert calls[0]["budget"] == cap, (calls[0]["budget"], cap)
    text = got["choices"][0]["message"]["content"]
    assert PHRASE in text and "</think>" in text, text
    answer = text.split("</think>", 1)[1]
    assert answer.strip(), "the client gets an answer, not a truncated thinking block"
    ids = calls[0]["ids"]
    u = got["usage"]
    assert u["completion_tokens"] == len(ids) <= 160
    assert u["completion_tokens_details"]["reasoning_tokens"] == _reasoning_count(ids)
    assert u["completion_tokens"] > u["completion_tokens_details"]["reasoning_tokens"]
    assert "client budget 8192 (reasoning_effort)" in log and f"cap {cap}" in log, log


def test_over_the_handler_hermes_streamed_gets_an_answer():
    b = hermes("medium", 160, stream_options={"include_usage": True},
               reasoning_format="tags")                                         # SRV-617
    raw, calls, _ = _handler(b, reserve=20)
    ds = _deltas(raw)
    text = "".join(d.get("content") or "" for d in ds)
    assert PHRASE in text and text.split("</think>", 1)[1].strip(), text


class _Res:
    mode, profile = "on", "coding"
    doc: dict = {}                                   # the profile: no `ui` (state lines off)

    def headers(self):
        return ()


class _Plan:
    """The handler's view of a router plan: a class cap (or None) and whether it decides."""

    def __init__(self, budget, decides=True):
        self.res, self.active, self.session, self.changed = _Res(), True, None, True
        self.budget, self.clean_ids, self.decides_effort = budget, None, decides
        self.client_word = _Conductor.client_word    # Ceiling, or ignore
        self.row = {"extra": {}}
        self.transcript = None                       # SRV-137: no session transcript here
        self.context = None                          # no in-context stage due
        self.ask, self.acts, self.outcome = None, [], None   # J1: no outcome, no act
        self.side = None                             # S1: not a client's side request

    def think_budget(self, budget):
        return self.budget if self.budget is not None else budget

    def apply_prompt(self, prompt, tok, in_think):
        return prompt, in_think

    def headers(self):
        return ()


class _Conductor:
    client_word = "ceiling"

    def __init__(self, budget, decides=True):
        self.budget, self.decides, self.plans = budget, decides, []

    def plan(self, body, headers, **kw):
        p = _Plan(self.budget, self.decides)
        self.plans.append(p)
        return p

    def monitor_mode(self, plan, server_mode):
        return server_mode

    def monitor_policy(self, plan, base):
        return None

    def finish(self, plan, rec):
        pass


def _acts():
    return [(e["task"], e["reason"]) for e in app.STATE["monitor"].journal.actions()]


def test_over_the_handler_a_client_number_closes_hidden_with_the_monitor():
    for b in ({"reasoning": {"max_tokens": 4}}, {"thinking": {"type": "enabled",
                                                             "budget_tokens": 4}},
              {"max_reasoning_tokens": 4},
              {"chat_template_kwargs": {"max_thinking_tokens": 4}}):
        body = dict(b, messages=[{"role": "user", "content": "hello"}], max_tokens=160,
                    reasoning_format="reasoning_content")
        raw, calls, _ = _handler(body, mode="on", cond=_Conductor(None))
        got = json.loads(raw)
        msg = got["choices"][0]["message"]
        assert PHRASE not in json.dumps(msg) and msg["content"], (b, msg)
        assert _acts() == [("client", "budget(4)")], (b, _acts())
        assert calls[0]["budget"] == 0 and calls[0]["sup"].think_budget == 4
        ids = calls[0]["ids"]
        assert got["usage"]["completion_tokens_details"]["reasoning_tokens"] == \
            _reasoning_count(ids)
        # the hidden phrase is not the client's: the cap's 4 reasoning tokens (SP1 reads them
        # after the block that reached them, one more here), then `</think>`
        assert 4 <= ids.index(frame.CLOSE) <= 5, ids[:8]


def test_over_the_handler_the_reserve_closes_hidden_and_is_journaled():
    """xhigh asks for no cap; the limit of 160 keeps a reserve of 40 after the phrase, so the
    reasoning closes with the monitor's hidden close, and the journal says it was the reserve."""
    cond = _Conductor(None, decides=False)
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 160,
            "reasoning_effort": "xhigh", "reasoning_format": "reasoning_content"}
    raw, calls, _ = _handler(body, mode="on", cond=cond)
    got = json.loads(raw)
    cap = 160 - 40 - P
    assert _acts() == [("reserve", f"budget({cap})")], _acts()
    ids = calls[0]["ids"]
    assert cap <= ids.index(frame.CLOSE) <= cap + 1, ids
    assert len(got["choices"][0]["message"]["content"]) >= 40
    assert cond.plans[0].row["extra"]["client_budget"] == {
        "field": "reasoning_effort", "kind": "effort", "asked": 0, "reserve": 40,
        "max_tokens": 160, "cap": cap, "lowered": True}


def test_over_the_handler_n2_and_shadow_close_with_the_phrase():
    """`n` > 1 and a monitor in shadow cannot close hidden: ThinkBudget's phrase, per choice."""
    for mode, n in (("shadow", 1), ("on", 2), (None, 2)):
        # tags asked for: since SRV-245 a client that names no format gets reasoning_content
        body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 160,
                "reasoning": {"max_tokens": 4}, "n": n, "reasoning_format": "tags"}
        raw, calls, _ = _handler(body, mode=mode, cond=_Conductor(None))
        got = json.loads(raw)
        assert len(got["choices"]) == n
        for ch in got["choices"]:
            text = ch["message"]["content"]
            assert PHRASE in text and text.split("</think>", 1)[1].strip(), (mode, n, text)
        assert all(c["budget"] == 4 for c in calls)


def test_over_the_handler_a_word_yields_to_the_routers_level():
    """Under a router that decides the effort and a profile whose `thinking.client_word` is
    `ignore` (opencode sends `medium` on every turn), a client's word does not override the
    class's cap; the class cap closes as SRV-159's does."""
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 160,
            "reasoning_effort": "medium"}
    _Conductor.client_word = "ignore"
    try:
        _handler(body, mode="on", cond=_Conductor(4))
        assert _acts() == [("route", "budget(4)")], _acts()
        # an xhigh class (no cap) with a word: the server's budget, then the reserve
        _, calls, _ = _handler(body, mode="on", cond=_Conductor(None))
        assert calls[0]["sup"].think_budget == 160 - 40 - P and calls[0]["budget"] == 0
    finally:
        _Conductor.client_word = "ceiling"
    # where the router does not decide the effort, the word is the cap (8,192, then the reserve)
    _, calls, _ = _handler(dict(body, max_tokens=200), cond=_Conductor(None, decides=False))
    assert calls[0]["budget"] == 200 - 50 - P


def test_over_the_handler_a_word_is_a_ceiling_by_default():
    """A design decision (2026-10-03 12:30): under a router that decides the effort, a client's
    word is a ceiling: a class below it keeps its own cap, a class above it (high, or xhigh with
    no cap) is held to the word's cap; the answer reserve then applies as before. Hermes sends
    `medium` with 24,000 tokens. The journal row says when the word held the router down."""
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 160,
            "reasoning_effort": "medium"}
    for router, word, want in ((1024, "medium", None), (16384, "medium", 8192),
                               (None, "medium", 8192), (16384, "low", 1024)):
        cond = _Conductor(router)
        _handler(dict(body, reasoning_effort=word), mode="on", cond=cond)
        got = cond.plans[-1].row["extra"].get("client_word")
        if want is None:
            assert got is None, (router, word, got)          # the router went lower: its own
        else:
            assert got == {"word": word, "cap": want, "router": router or 0, "budget": want}, \
                (router, word, got)
    # opencode's profile ignores the word: the router's level alone
    _Conductor.client_word = "ignore"
    try:
        cond = _Conductor(16384)
        _handler(body, mode="on", cond=cond)
        assert "client_word" not in cond.plans[-1].row["extra"]
    finally:
        _Conductor.client_word = "ceiling"


def test_over_the_handler_thinking_off_has_no_budget():
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 40,
            "reasoning_effort": "none"}
    raw, calls, log = _handler(body, mode="on", cond=_Conductor(None))
    tail = frame.MonTok().encode("<think>\n\n</think>\n\n")
    assert calls[0]["prompt"][-len(tail):] == tail
    assert calls[0]["sup"].think_budget is None and "client budget" not in log
    assert json.loads(raw)["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0


# ------------------------------------------------------------------ 9. the router

def _plan(label, **kw):
    cond, _ = setup(decider=by_label(label, 0.9))
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE}]
    plan, _, _ = turn(cond, rbody(msgs, **kw), headers={"x-session-id": "cb-" + label})
    return plan


def test_the_router_a_client_number_in_any_shape_wins():
    for kw in ({"max_reasoning_tokens": 777}, {"reasoning": {"max_tokens": 777}},
               {"thinking": {"type": "enabled", "budget_tokens": 777}},
               {"chat_template_kwargs": {"thinking_budget": 777}}):
        plan = _plan("plan", **kw)
        assert plan.row["budget_source"] == "client" and plan.budget is None, kw
        assert plan.decides_effort


def test_the_router_a_client_word_yields_to_the_class():
    for kw in ({"reasoning_effort": "medium"}, {"reasoning": {"effort": "low"}},
               {"reasoning": {"enabled": True}}):
        plan = _plan("plan", **kw)
        assert (plan.row["effort"], plan.budget, plan.row["budget_source"]) == \
            ("high", 16384, "class"), kw


def test_the_router_a_client_off_form_is_the_clients():
    """An explicit off is the client's (the template closes the block), and the row
    says so: `off_client`, no budget, `extra.client_thinking` -- never the class's level as if it
    thought (Open WebUI's `none` against xhigh's row and note, 2026-10-06 06:11)."""
    for kw in ({"reasoning_effort": "none"}, {"thinking": {"type": "disabled"}},
               {"reasoning": {"enabled": False}},
               {"chat_template_kwargs": {"enable_thinking": False}}):
        for label in ("summarize", "plan"):
            plan = _plan(label, **kw)
            assert plan.row["thinking"] == "off_client" and not plan.thinking_off, (kw, label)
            assert plan.row["budget"] is None and plan.budget is None, (kw, label)
            ct = plan.row["extra"]["client_thinking"]
            assert ct["off"] and ct["why"] == "client turned thinking off", ct
    assert _plan("summarize").thinking_off


def test_the_router_no_effort_sentence_in_a_turn_the_client_turned_off():
    """The class's effort sentence ("Reasoning effort is set to xhigh. Please think carefully
    ...") is not added to a turn that cannot think; with thinking on it is."""
    on = _plan("plan")
    off = _plan("plan", reasoning_effort="none")
    kinds = lambda p: [x["kind"] for x in (p.block or {}).get("parts", [])]   # noqa: E731
    assert "effort" in kinds(on), on.block
    assert "effort" not in kinds(off), off.block
    assert "effort_note_skipped:client_off" in off.row["extra"].get("notes", []), off.row["extra"]


class _ClassConductor(_Conductor):
    """A plan whose row carries a router class (the penalty rule reads it)."""

    def __init__(self, cls):
        super().__init__(None)
        self.cls = cls

    def plan(self, body, headers, **kw):
        p = super().plan(body, headers, **kw)
        p.row["class_id"] = self.cls
        return p


def test_over_the_handler_no_repeat_is_dropped_on_code_and_tool_turns():
    """A client's `no_repeat_ngram_size` is not for code (every repeated n-gram pays
    4 nats; repeated n-grams are code). Under the conductor it is dropped on the code classes'
    turns and on turns that offer tools, and the row and the log say so; a chat turn without
    tools and a raw request keep it."""
    base = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 20,
            "no_repeat_ngram_size": 4, "chat_template_kwargs": {"enable_thinking": False}}
    for cls, tools, kept in (("code_edit", False, 0), ("chat", True, 0), ("chat", False, 4),
                             ("design_ui", False, 0)):
        cond = _ClassConductor(cls)
        b = dict(base, **({"tools": TOOLS} if tools else {}))
        raw, calls, log = _handler(b, mode="on", cond=cond)
        pen = calls[0]["pen"]
        got = pen.spec.no_repeat if pen is not None else 0
        assert got == kept, (cls, tools, got)
        extra = cond.plans[0].row["extra"]
        if kept:
            assert "penalty_dropped" not in extra and "[penalty]" not in log, extra
        else:
            assert extra["penalty_dropped"] == {"no_repeat_ngram_size": 4,
                                                "why": "code" if cls != "chat" else "tools"}
            assert "no_repeat_ngram_size=4 dropped" in log, log[-300:]
    raw, calls, log = _handler(dict(base, tools=TOOLS))          # raw: the client's mask
    assert calls[0]["pen"] is not None and calls[0]["pen"].spec.no_repeat == 4


def test_the_request_keys_line_shows_the_shapes():
    line = logbuf.request_keys_line("c", {"reasoning": {"max_tokens": 9, "effort": "x"},
                                          "thinking": {"budget_tokens": 7},
                                          "enable_thinking": False})
    assert '"reasoning": {"max_tokens": 9}' in line and '"budget_tokens": 7' in line, line
    assert '"enable_thinking": false' in line, line


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
