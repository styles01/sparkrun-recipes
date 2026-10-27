"""The tool-argument loop guard (server/argloop.py).

The 2026-10-04 16:18 shape: a `write` whose content falls into one 97-character tile-map row
repeated until the token limit (65,536 tokens, 327 s, nothing written). Here:

  1  the guard's rules on text: a periodic tail fires within `span_tokens` of the loop's start;
     legit shapes do not (a tile map with a few blank rows, a data table, minified code, a long
     varied file); the size rule needs size, the context multiple and the compression together
  2  through the handler with the fake engine (one token a character): the looping call is
     stopped within a few thousand tokens, the client gets no call and no block text, one
     sentence, finish stop; the journal has A1's GIVE_UP
  3  the session's next request carries the hidden note at the prompt's tail, once
  4  a long legit write streams through whole
  5  without held calls (no acting monitor) the guard only logs: nothing of the stream changes
  6  `--argloop off` and `log`

2026-10-05: a live call the token limit cuts (no guard involved) is not sent again as
content (5).

The case (the Open WebUI retry storm, 2026-10-06 06:11: the stopped 6,400-token arguments
replayed in the history, five byte-identical attempts, the prompt 17k -> 46k):

  7  a long unit fires at about three copies, a short one as before; two identical frames do not
  8  the history: a stopped call's arguments are cut to a head and a marker; a closed call never
  9  the client's retry loop through the handler: the first retry carries the collapsed history
     and the named note, the identical second stop ends the loop with the give-up sentence, no
     generation; five client attempts become three; a new user message starts over
 10  the router: a retry thinks one level up, at least medium, unless the client turned it off

Run: python tests/test_argloop.py
"""

from __future__ import annotations

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import app, argloop  # noqa: E402
from test_toolcheck import SYSTEM, ask, ctx_text, policy, rows, serve  # noqa: E402

# opencode's system prompt is longer than the monitor's lineage window: one session
LONG = SYSTEM + "\n" + "Follow the project's rules. " * 200

ROW = '"' + " " * 94 + '",\n'


def _stream(text: str, cfg=None, context_chars: int = 0, per_token: int = 1):
    """Feed `text` as tokens of `per_token` characters; the verdict and the tokens fed."""
    g = argloop.ArgLoopGuard(cfg, context_chars)
    n = 0
    for i in range(per_token, len(text) + per_token, per_token):
        n += 1
        v = g.feed(n, min(i, len(text)), lambda i=i: text[:i])
        if v is not None:
            return v, n
    return None, n


def _code(n_lines: int, seed: int = 1) -> str:
    r = random.Random(seed)
    names = ["player", "enemy", "coin", "tile", "camera", "score", "level", "vx", "vy", "dt"]
    out = []
    for i in range(n_lines):
        a, b = r.choice(names), r.choice(names)
        out.append(f"  if ({a}.x > {b}.y + {r.randint(0, 999)}) {{ {a}.v = {r.random():.3f}; }}"
                   f" // {i}")
    return "\n".join(out)


# ------------------------------------------------------------------ 1: the rules on text
def test_1_the_incident_shape_fires_soon_after_the_loop_starts():
    head = _code(60)
    text = head + "\nconst LEVEL = [\n" + ROW * 3000
    # the served tokenizer reads the row as ~5 tokens: 20 characters a token
    v, n = _stream(text, per_token=20)
    assert v is not None and v.rule == "periodic" and v.period == len(ROW), v
    onset = len(head) // 20
    assert n - onset <= argloop.DEFAULTS["span_tokens"] + 2 * argloop.DEFAULTS["every"] + 64, \
        (n, onset)


def test_1_a_short_unit_loop_fires_too():
    v, _ = _stream(_code(30) + "\n" + "0, " * 20000, per_token=2)
    assert v is not None and v.rule == "periodic" and v.period in (3, 6, 9), v


def test_1_legit_shapes_do_not_fire():
    blank = "  " + '"' + " " * 60 + '",\n'
    tilemap = "const MAP = [\n" + blank * 12 + ('"' + "#" * 60 + '",\n') * 2 + "];\n"
    table = "".join(f"  {{id: {i}, name: 'item{i}', price: {i * 3 % 17}.99}},\n"
                    for i in range(3000))
    minified = "".join(f"function f{i}(a,b){{return a*{i}+b-{i % 7}}};" for i in range(4000))
    levels = "".join(f"// level {k}\n" + tilemap.replace("#" * 60, "#" * k + "?" * (60 - k))
                     for k in range(1, 30))
    for text in (_code(5000), levels + _code(200), table, minified,
                 _code(300) + tilemap + _code(300)):
        for per in (1, 4):
            v, _ = _stream(text, per_token=per)
            assert v is None, (v, text[:80])


def test_1_the_size_rule_needs_size_context_and_repetition():
    big = _code(30000)                  # ~2 MB, varied: compresses ~4x
    cfg = {"size_tokens": 1000, "min_tokens": 64, "every": 256}
    assert _stream(big, cfg, per_token=4)[0] is None
    # near-repetition no period catches: a counter in every row
    rows_ = "".join(f'"{" " * 90}{i:06d}",\n' for i in range(60000))
    v, _ = _stream(rows_, cfg, per_token=8)
    assert v is not None and v.rule == "size" and v.ratio >= 16, v
    # the same text, with a message in context bigger than it can grow to: no fire
    v, _ = _stream(rows_, cfg, context_chars=10 ** 8, per_token=8)
    assert v is None


def test_1_periodic_tail_and_the_messages():
    p, span = argloop.periodic_tail("x" * 50 + "abc" * 400, 4096)
    assert p == 3 and span == 1200
    assert argloop.periodic_tail(_code(50), 4096) == (0, 0)
    msgs = [{"role": "system", "content": "s" * 10},
            {"role": "tool", "content": [{"type": "text", "text": "t" * 500}]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "write", "arguments": "a" * 900}}]}]
    assert argloop.longest_message_chars(msgs) == 900
    note = argloop.note_text({"rule": "periodic", "period": 98, "repeats": 211, "tokens": 2176})
    assert "98-character" in note and "211 times" in note and "{why}" not in note


# ------------------------------------------------------------------ 2, 3: through the handler
def test_2_the_looping_write_is_stopped_with_one_sentence():
    mon = serve()
    got = ask("FAKE_ARG_LOOP write the game", max_tokens=60000)
    assert got["calls"] == {} and "<tool_call>" not in got["content"], got["content"][:200]
    assert "LEVEL" not in got["content"]
    assert got["content"].strip().startswith("[TandemLLM stopped the `write` call"), got["content"]
    assert got["finish"] == "stop", got["finish"]
    n = len(app.STATE["last_ctx"]) - len(app.STATE["tok"].encode_text(""))
    g = rows(mon, task="A1", decision="GIVE_UP")
    assert g and g[0]["detail"]["rule"] == "periodic" and g[0]["detail"]["name"] == "write", g
    # the fake writes one token a character: the loop starts ~750 tokens into the block
    assert g[0]["detail"]["tokens"] < 3000, g[0]["detail"]
    assert n < 10000, n


def test_3_the_next_request_carries_the_hidden_note_once():
    mon = serve()
    ask("FAKE_ARG_LOOP write the game", max_tokens=60000, system=LONG)
    ask("FAKE_ARG_LOOP write the game", max_tokens=200, system=LONG,
        history=[{"role": "assistant", "content": "[TandemLLM stopped the `write` call]"},
                 {"role": "user", "content": "go on"}])
    notes = rows(mon, task="A1", decision="PROMPT_TAIL_NOTE")
    assert len(notes) == 1, mon.journal.events()
    assert "Runtime note: your last tool call was stopped" in ctx_text()
    ask("go on, read the tool docs", max_tokens=200, system=LONG,
        history=[{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}])
    assert len(rows(mon, task="A1", decision="PROMPT_TAIL_NOTE")) == 1
    assert "Runtime note: your last tool call was stopped" not in ctx_text()


def test_4_a_long_legit_write_goes_through_whole():
    mon = serve()
    content = _code(400)
    import json as _j
    got = ask("FAKE_CALL_JSON " + _j.dumps({"name": "write", "arguments": {
        "filePath": "/tmp/a.js", "content": content}}), max_tokens=60000)
    assert list(got["calls"]) == [0], got["content"][:200]
    assert _j.loads(got["calls"][0]["args"])["content"].strip() == content.strip()
    assert not rows(mon, task="A1")


def test_4_a_call_streamed_live_is_stopped_too():
    """The live coding profile's tool gate is off (no SP3 task acts): calls stream as they are
    written. The client keeps the call it was sent part of (never closed, as the token limit
    left it on 2026-10-04), gets no block text as content, one sentence, finish stop; the
    session's next request has the note."""
    mon = serve(pol=policy(C4="off", C3="off", C1="off", C2="off"))
    got = ask("FAKE_ARG_LOOP write the game", max_tokens=60000, system=LONG)
    assert list(got["calls"]) == [0] and got["calls"][0]["name"] == "write", got["calls"].keys()
    args = got["calls"][0]["args"]
    assert args.startswith("{") and 1000 < len(args) < 6000, len(args)
    assert "<tool_call>" not in got["content"] and "LEVEL" not in got["content"]
    assert "[TandemLLM stopped the `write` call" in got["content"], got["content"][-300:]
    assert got["finish"] == "stop", got["finish"]
    g = rows(mon, task="A1", decision="GIVE_UP")
    assert g and g[0]["detail"]["streamed"] is True, g
    ask("FAKE_ARG_LOOP write the game", max_tokens=200, system=LONG,
        history=[{"role": "assistant", "content": "stopped"}, {"role": "user", "content": "go"}])
    assert len(rows(mon, task="A1", decision="PROMPT_TAIL_NOTE")) == 1


def test_5_without_a_monitor_the_guard_only_logs():
    serve(mode="raw")
    got = ask("FAKE_ARG_LOOP write the game", max_tokens=6000)
    assert got["finish"] == "length", got["finish"]
    assert "stopped the" not in got["content"]
    # 2026-10-05: the call streamed live and the token limit cut it; the client keeps the call as it was
    # sent and gets none of the block's text as content (it came twice until 2026-10-05)
    assert list(got["calls"]) == [0] and got["calls"][0]["name"] == "write", got["calls"].keys()
    assert "<tool_call>" not in got["content"] and "LEVEL" not in got["content"], \
        got["content"][:200]


def test_6_off_and_log():
    for flag in ("off", "log"):
        mon = serve()
        app.STATE["argloop"] = flag
        got = ask("FAKE_ARG_LOOP write the game", max_tokens=6000)
        assert "stopped the" not in got["content"] and not rows(mon, task="A1"), flag


# ------------------------------------------------------------------ 7: the long tier (0.3.40)
def _unit(period: int, seed: int) -> str:
    r = random.Random(seed)
    u = ""
    while len(u) < period:
        u += '  "' + "".join(r.choice(".rgbk") for _ in range(30)) + '",\n'
    return u[:period]


def test_7_a_long_unit_fires_at_about_three_copies():
    head = _code(60) + "\nconst DATA = [\n"
    for period in (405, 1000):
        unit = _unit(period, period)
        v, n = _stream(head + unit * 40, per_token=3)
        assert v is not None and v.rule == "periodic" and v.period == period, (period, v)
        assert v.detail["tier"] == "long" and 3.0 <= v.repeats < 4.0, (period, v.as_dict())
        assert v.piece and v.piece in unit + unit, v.piece
        old, _ = _stream(head + unit * 40, cfg={"long_period": 10 ** 6}, per_token=3)
        assert old.repeats >= 6.0 and old.tokens > v.tokens, (old.as_dict(), v.as_dict())
    # a short unit keeps the six-copy rule
    v, _ = _stream(head + _unit(98, 3) * 400, per_token=3)
    assert v.detail["tier"] == "short" and v.repeats >= 6.0, v.as_dict()


def test_7_two_identical_frames_and_pixel_rows_do_not_fire():
    frame = "  [\n" + "".join(f'    "{_unit(16, k)[3:19]}",\n' for k in range(16)) + "  ],\n"
    other = frame.replace("r", "g")
    off = [random.Random(k).randint(0, 20) for k in range(60)]   # pixel art: no cycle
    rows_ = "".join("  '" + "." * o + "r" * 40 + "." * (24 - o) + "',\n" for o in off)
    for text in (_code(80) + "const FRAMES = [\n" + frame * 2 + other + "];\n" + _code(40),
                 _code(80) + "const PX = [\n" + rows_ + "];\n" + _code(40)):
        for per in (1, 3, 4):
            v, _ = _stream(text, per_token=per)
            assert v is None, (per, v)


# ------------------------------------------------------------------ 8: the replayed history
def _stopped_args(period: int = 405) -> str:
    """A call's JSON arguments as a client keeps them after the guard stopped it: never closed."""
    import json as _j
    text = _code(40) + "\nconst DATA = [\n" + _unit(period, 7) * 4
    return _j.dumps({"filePath": "/tmp/game.html", "content": text})[:-2]


def _call(args: str, name: str = "write", cid: str = "c1") -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]}


def test_8_a_stopped_call_in_the_history_is_cut_to_its_head_and_a_marker():
    import json as _j
    args = _stopped_args()
    msgs = [{"role": "user", "content": "write it"}, _call(args),
            {"role": "tool", "tool_call_id": "c1", "content": "invalid arguments"}]
    out, cut = argloop.collapse_history(msgs)
    assert len(cut) == 1 and cut[0]["name"] == "write" and cut[0]["chars"] == len(args), cut
    new = out[1]["tool_calls"][0]["function"]["arguments"]
    assert len(new) < 2000 < len(args) and "TandemLLM stopped this `write` call" in new, len(new)
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == args     # the client's list kept
    assert out[0] is msgs[0] and out[2] is msgs[2]
    stops = argloop.trailing_stops(out)
    assert [(x["name"], x["period"]) for x in stops] == [("write", cut[0]["period"])], stops
    # idempotent: the marker is never cut again
    again, cut2 = argloop.collapse_history(out)
    assert not cut2 and again[1]["tool_calls"][0]["function"]["arguments"] == new
    # a call that closed is never touched, however repetitive its content
    closed = _j.dumps({"filePath": "/tmp/a.txt", "content": _unit(405, 7) * 6})
    out, cut = argloop.collapse_history([_call(closed)])
    assert not cut and out[0]["tool_calls"][0]["function"]["arguments"] == closed
    # opencode's stand-in for a call that did not parse: its value is cut
    inv = _j.dumps({"tool": "write", "error": "bad JSON: " + _unit(405, 7) * 6})
    out, cut = argloop.collapse_history([_call(inv, name="invalid")])
    assert len(cut) == 1 and _j.loads(out[0]["tool_calls"][0]["function"]["arguments"])["tool"] \
        == "write"


def test_8_the_real_template_renders_the_marker_not_the_loop():
    from test_conductor_router import setup
    setup()
    a = _stopped_args(405)
    unit = _unit(405, 7)
    msgs = [{"role": "user", "content": "write it"}, _call(a),
            {"role": "tool", "tool_call_id": "c1", "content": "Error: not valid JSON"}]
    before = app.STATE["tok"].decode(app.build_prompt({"messages": msgs})[0].tolist())
    out, _ = argloop.collapse_history(msgs)
    after = app.STATE["tok"].decode(app.build_prompt({"messages": out})[0].tolist())
    import json as _j
    row = _j.dumps(unit.split("\n")[1])[1:-1]       # as the cut call's raw arguments hold it
    assert before.count(row) >= 3 and after.count(row) <= 1, (before.count(row), after.count(row))
    assert "TandemLLM stopped this `write` call" in after and len(after) < len(before) - 1000


def test_8_the_streak_and_the_give_up_rule():
    a, b = _stopped_args(405), _stopped_args(600)
    tool = {"role": "tool", "tool_call_id": "c1", "content": "invalid arguments"}
    user = {"role": "user", "content": "write it"}
    out, _ = argloop.collapse_history([user, _call(a), tool, _call(a), tool])
    st = argloop.trailing_stops(out)
    assert len(st) == 2 and argloop.identical(st) == 2 and argloop.give_up_due(st)
    out, _ = argloop.collapse_history([user, _call(a), tool, _call(b), tool])
    st = argloop.trailing_stops(out)
    assert len(st) == 2 and argloop.identical(st) == 1 and not argloop.give_up_due(st)
    out, _ = argloop.collapse_history([user, _call(a), tool, _call(b), tool, _call(a), tool])
    assert argloop.give_up_due(argloop.trailing_stops(out))        # three of one tool
    # a new user message starts over; a call that ran ends the streak
    out, _ = argloop.collapse_history([user, _call(a), tool, _call(a), tool,
                                       {"role": "user", "content": "try again"}])
    assert argloop.trailing_stops(out) == []
    out, _ = argloop.collapse_history([user, _call(a), tool, _call('{"filePath": "a"}', "read"),
                                       tool, _call(a), tool])
    assert len(argloop.trailing_stops(out)) == 1
    note = argloop.note_text({}, argloop.trailing_stops(
        argloop.collapse_history([user, _call(a), tool, _call(a), tool])[0]))
    assert "2 `write` calls were stopped" in note and "identical loop" in note, note
    assert "beginning" in note


# ------------------------------------------------------------------ 9: the client's retry loop
def _owui_loop(attempts: int = 5):
    """What Open WebUI's native tool loop did on 2026-10-06: every answer with a call is
    answered with the tool's error and the conversation sent again, up to `attempts` times."""
    history, out = [], []
    for k in range(attempts):
        got = ask("FAKE_ARG_LOOP write the game", max_tokens=60000, system=LONG,
                  history=list(history))
        out.append((got, ctx_text() if app.STATE.get("last_ctx") is not None else ""))
        app.STATE["last_ctx"] = None
        if not got["calls"]:
            break
        c = got["calls"][0]
        history += [{"role": "assistant", "content": got["content"], "tool_calls": [
            {"id": c["id"] or f"c{k}", "type": "function",
             "function": {"name": c["name"], "arguments": c["args"]}}]},
            {"role": "tool", "tool_call_id": c["id"] or f"c{k}",
             "content": "Error: the tool arguments are not valid JSON"}]
    return out


def test_9_five_client_retries_become_three_attempts_and_a_sentence():
    mon = serve(pol=policy(C4="off", C3="off", C1="off", C2="off"))
    cuts, orig = [], argloop.collapse_history

    def spy(messages):
        out = orig(messages)
        cuts.append(out[1])
        return out
    app.argloop_mod.collapse_history = spy
    try:
        runs = _owui_loop(5)
    finally:
        app.argloop_mod.collapse_history = orig
    assert len(runs) == 3, [r[0]["content"][-120:] for r in runs]
    first, second, third = runs
    assert "[TandemLLM stopped the `write` call" in first[0]["content"]
    # each retry's history: every stopped call cut down before anything read it (the fake's
    # template renders no calls; test_8_..._renders_the_marker is the real template's prompt)
    assert [len(c) for c in cuts] == [0, 1, 2], [len(c) for c in cuts]
    assert all(c["chars"] - c["cut"] < 2000 < c["chars"] or c["chars"] < 2000
               for c in cuts[1] + cuts[2]), cuts
    # the retry's note names the tool and the piece (its spaces kept)
    p2 = second[1]
    assert "Runtime note: your last `write` call was stopped" in p2, p2[-600:]
    note = p2[p2.find("Runtime note"):][:400]
    assert "101-character piece" in note and '\\"' + " " * 20 in note, note
    assert "[TandemLLM stopped the `write` call" in second[0]["content"]
    # the identical second stop: no generation, one sentence, no call, finish stop
    assert third[1] == "", "the give-up runs no generation"
    assert third[0]["calls"] == {} and third[0]["finish"] == "stop", third[0]
    assert third[0]["content"].startswith("[TandemLLM stopped retrying: the last 2 attempts at "
                                          "the `write` call"), third[0]["content"]
    g = rows(mon, task="A1", decision="GIVE_UP")
    assert [x["reason"] for x in g] == ["tool_arg_loop(periodic)", "tool_arg_loop(periodic)",
                                        "tool_arg_loop_retries(2)"], [x["reason"] for x in g]
    notes = rows(mon, task="A1", decision="PROMPT_TAIL_NOTE")
    assert len(notes) == 1 and notes[0]["detail"]["stops"] == 1, notes


def test_9_a_new_user_message_starts_over():
    serve(pol=policy(C4="off", C3="off", C1="off", C2="off"))
    a = _stopped_args(98)
    tool = {"role": "tool", "tool_call_id": "c1", "content": "Error: not valid JSON"}
    got = ask("FAKE_ARG_LOOP write the game", max_tokens=60000, system=LONG,
              history=[_call(a), tool, _call(a), tool,
                       {"role": "user", "content": "FAKE_ARG_LOOP try again, smaller"}])
    assert "stopped retrying" not in got["content"] and app.STATE.get("last_ctx") is not None


def test_9_raw_without_a_monitor_still_cuts_the_history():
    serve(mode="raw")
    a = _stopped_args(405)
    cuts, orig = [], argloop.collapse_history

    def spy(messages):
        out = orig(messages)
        cuts.append(len(out[1]))
        return out
    app.argloop_mod.collapse_history = spy
    try:
        hist = [_call(a), {"role": "tool", "tool_call_id": "c1", "content": "Error"}]
        ask("hello", max_tokens=50, history=hist)
        assert cuts == [1], cuts
        app.STATE["argloop"] = "off"                 # off: the history as the client sent it
        ask("hello", max_tokens=50, history=hist)
        assert cuts == [1], cuts
    finally:
        app.argloop_mod.collapse_history = orig


# ------------------------------------------------------------------ 10: the router's retry
def test_10_a_retry_thinks_one_level_up_unless_the_client_turned_it_off():
    from test_conductor_router import SYSTEM as RSYS, body as rbody, by_label, setup, turn
    tool = {"role": "tool", "tool_call_id": "c1", "content": "Error: not valid JSON"}
    a = _stopped_args(405)
    for label, want in (("summarize", "medium"), ("plan", "xhigh")):
        cond, _ = setup(decider=by_label(label, 0.9))
        msgs = [{"role": "system", "content": RSYS}, {"role": "user", "content": "write it"},
                _call(a), tool]
        msgs, _ = argloop.collapse_history(msgs)
        plan, _, _ = turn(cond, rbody(msgs), headers={"x-session-id": "ar-" + label})
        r = plan.row["extra"]["argloop_retry"]
        assert r["stops"] == 1 and r["effort"]["to"] == want, (label, r)
        assert plan.row["effort"] == want, (label, plan.row["effort"])
        cond, _ = setup(decider=by_label(label, 0.9))
        plan, _, _ = turn(cond, rbody(msgs, reasoning_effort="none"),
                          headers={"x-session-id": "ar-off-" + label})
        r = plan.row["extra"]["argloop_retry"]
        assert r["effort"]["blocked"] == "client turned thinking off", r
        assert plan.row["thinking"] == "off_client"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
