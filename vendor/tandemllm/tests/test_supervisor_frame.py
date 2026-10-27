"""SRV-109 / SRV-110: the conductor's session monitor frame, on a CPU.

The test matrix is SRV-109's Gherkin, one test or more per scenario:

  1  off is byte-identical to the release              tests/test_serve_identity.py (golden),
                                                       and here: raw vs on-with-no-task
  2  on with no task passes everything                 test_on_with_no_task_*
  3  a header lowers the mode and cannot raise it      test_the_header_*
  4  shadow never acts                                 test_shadow_*
  5  a hidden close streams only the closer            test_a_hidden_close_*
  6  a held tool call is released whole                test_held_calls_* (and tests/test_toolcall.py)
  7  the end token is dropped cleanly                  test_the_end_token_is_dropped_*
  8  caps end in GIVE_UP                               test_caps_*
  9  one session across requests without an id         test_the_session_*

plus the policy file and its schema, the SP4 splices (a tail note, thinking off) against the
real chat template, and the accounting of hidden tokens. The HTTP e2e over the fake engine is in
tests/test_fake_engine.py.

Run: python tests/test_supervisor_frame.py
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine.spec import ThinkBudget  # noqa: E402
from server import app  # noqa: E402
from server.conductor.monitor import (  # noqa: E402
    Action, BlockEvent, FinishEvent, Journal, Monitor, Policy, PolicyError, RequestEvent,
    ToolCallEvent, lower, parse_mode, schema, tier0,
)
from server.conductor.monitor import policy as policy_mod  # noqa: E402
from server.conductor.monitor.executor import tail_note, think_off  # noqa: E402
from server.conductor.monitor.policy import (  # noqa: E402
    CONTINUE_HIDDEN, GIVE_UP, PASS, PROMPT_TAIL_NOTE, REGENERATE_CALL, SET_THINK_BUDGET,
)
from server.conductor.monitor.session import Sessions  # noqa: E402
from test_app_loop import Req, serve  # noqa: E402
from test_serve_identity import Oracle  # noqa: E402
from test_window_edge import FixedDrafter  # noqa: E402

OPEN, CLOSE, IM_END, IM_START = 90, 91, 92, 93
ALPHA = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,\n0123456789:;!?'-()<>=/_\"{}[]|"
SPECIAL = {"<think>": OPEN, "</think>": CLOSE, "<|im_end|>": IM_END, "<|im_start|>": IM_START}
CLOSE_TEXT = "\n\nact now.\n</think>\n\n"


class MonTok:
    """A tokenizer for the random 4-layer model (97 ids): one id a character of ALPHA, and the
    chat template's special tokens as single ids, so hidden texts forward through the model and
    decode back to what they were."""

    unk_token_id = -1
    eos_token_id = None

    def __init__(self, ends=()):
        # the ids a test made end tokens: special, so decoding skips them as it skips <|im_end|>
        self.ends = set(ends)

    def convert_tokens_to_ids(self, text):
        return SPECIAL.get(text, -1)

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for s, t in SPECIAL.items():
                if text.startswith(s, i):
                    out.append(t)
                    i += len(s)
                    break
            else:
                k = ALPHA.find(text[i])
                out.append(k + 1 if k >= 0 else ALPHA.index("?") + 1)
                i += 1
        return out

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.encode(text or "")
        if return_tensors == "pt":
            ids = torch.tensor([ids])
        return type("R", (), {"input_ids": ids})()

    def decode(self, seq, skip_special_tokens=True):
        inv = {v: k for k, v in SPECIAL.items()}
        out = []
        for t in seq:
            t = int(t)
            if skip_special_tokens and t in self.ends:
                continue
            if t in inv:
                if not (skip_special_tokens and t in (IM_END, IM_START)):
                    out.append(inv[t])
            elif 1 <= t <= len(ALPHA):
                out.append(ALPHA[t - 1])
            else:
                out.append("#")
        return "".join(out)

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=True,
                            tools=None, **kw):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        text += "<|im_start|>assistant\n" + ("<think>\n" if enable_thinking
                                              else "<think>\n\n</think>\n\n")
        return {"input_ids": torch.tensor([self.encode(text)])}


def policy(tasks: dict | None = None, **top) -> Policy:
    """ops/supervisor.yaml with its tasks (and any top-level section) replaced."""
    d = yaml.safe_load(open(policy_mod.DEFAULT_PATH))
    if tasks is not None:
        d["tasks"] = tasks
    for k, v in top.items():
        d[k] = dict(d[k], **v)
    return Policy(d)


def backstop(cap: int, mode: str = "act") -> dict:
    """A test policy that closes the reasoning at `cap` tokens."""
    return {"B": {"rule": "reasoning_backstop", "mode": mode, "close": CLOSE_TEXT, "cap": cap,
                  "reserve": 1}}


def monitor(mode: str, tasks: dict | None = None, **top) -> Monitor:
    return Monitor(mode, policy(tasks if tasks is not None else {}, **top), MonTok(),
                   Journal(echo=False))


def begin(mon: Monitor, mode: str = "on", tools=(), max_tokens: int = 64, n: int = 1,
          messages=None, prompt=(5, 6, 7), conv=None):
    req = RequestEvent("r", messages=messages or [{"role": "user", "content": "q"}],
                       tools=list(tools), n=n, max_tokens=max_tokens)
    return mon.begin(mon.mode_for(mode, n), req, list(prompt), conv)


@contextlib.contextmanager
def rule(name: str, sp: str, fn):
    """A rule registered for one test, and gone after it."""
    tier0.register(name, sp, {})(fn)
    try:
        yield
    finally:
        del tier0.RULES[name]


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def run(drafter=None, tree=False, prompt=(5, 6, 7, 8, OPEN), max_new=40, eos=(), budget=0,
        sup=None):
    """One generation on the random model: (tokens, engine, published context)."""
    eng = serve(drafter, tree=tree)
    app.STATE["tok"] = MonTok()
    think = ThinkBudget(MonTok(), budget=budget, arms=sup.arms if sup is not None else None)
    with quiet():
        out = list(app.generate_stream(torch.tensor(list(prompt)), max_new, set(eos), think,
                                       sup=sup))
    return out, eng, app.STATE["last_ctx"]


PATHS = (("chain", lambda: FixedDrafter(97, 5, tree_mode=False), False),
         ("tree", lambda: FixedDrafter(97, 5, tree_mode=True), True),
         ("single", lambda: None, False))


# ------------------------------------------------------------------ the policy and its schema

def test_the_policy_file_validates_and_its_schema_file_is_current():
    p = policy_mod.load()
    assert set(p.tasks) >= {"D1", "D2", "E1"} and p.version >= 1
    with open(policy_mod.SCHEMA_PATH) as f:
        assert json.load(f) == schema(), \
            "ops/supervisor.schema.json is stale: python tools/conductor_policy.py --schema"


def test_a_bad_policy_is_refused_naming_the_field():
    base = yaml.safe_load(open(policy_mod.DEFAULT_PATH))
    for change, where in (
            (lambda d: d["tasks"]["D1"]["arms"]["repeat"].update(act=1.5),
             "tasks.D1.arms.repeat.act"),
            (lambda d: d["tasks"]["E1"].update(mode="sometimes"), "tasks.E1.mode"),
            (lambda d: d["tasks"]["D2"].update(rule="no_such_rule"), "unknown rule"),
            (lambda d: d["tasks"]["D2"].pop("cap"), "tasks.D2.cap"),
            (lambda d: d["caps"].update(acts_per_request=-1), "caps.acts_per_request"),
            (lambda d: d.update(extra=1), "policy.extra"),
            (lambda d: d["templates"].pop("think_close"), "templates.think_close")):
        d = copy.deepcopy(base)
        change(d)
        try:
            Policy(d)
            raise AssertionError(f"accepted a policy with {where} wrong")
        except PolicyError as exc:
            assert where in str(exc), (where, str(exc))


def test_the_server_refuses_to_start_on_a_bad_policy_and_ignores_it_raw():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "bad.yaml")
    with open(path, "w") as f:
        f.write("version: 1\ncaps: {}\n")
    args = type("A", (), {"conductor": "on", "conductor_policy": path})()
    try:
        app._conductor_policy(args)
        raise AssertionError("a policy that does not validate must stop the start")
    except SystemExit as exc:
        assert "does not load: policy.session: missing" in str(exc), exc
    args.conductor = "raw"
    assert app._conductor_policy(args) is None


def test_the_modes():
    assert [parse_mode(x) for x in ("raw", "off", "OFF", "shadow", "on", "bogus", None)] == \
        ["raw", "raw", "raw", "shadow", "on", None, None]
    assert lower("on", "shadow") == "shadow" and lower("on", "off") == "raw"
    assert lower("raw", "on") == "raw" and lower("shadow", "on") == "shadow"
    assert lower("on", None) == "on" and lower("on", "nonsense") == "on"


# ------------------------------------------------------------------ 2: on with no task passes

def test_on_with_no_task_writes_what_raw_writes():
    """Every path, with and without a reasoning budget: the supervised loop with no task is the
    raw loop token for token, and the journal holds nothing."""
    for name, make, tree in PATHS:
        for budget in (0, 3):
            ref = run(make(), tree, budget=budget)
            mon = monitor("on")
            sup = begin(mon)
            got = run(make(), tree, budget=budget, sup=sup)
            assert got[0] == ref[0], (name, budget)
            assert got[1].kv.length == ref[1].kv.length and list(got[2]) == list(ref[2])
            assert mon.journal.events() == [] and not sup.acted, (name, mon.journal.events())


def test_on_with_no_task_answers_what_raw_answers_over_the_handler():
    """The whole streamed response, raw and on with every task off: the same chunks, and the
    header says which mode ran. The budget is the server's own (--think-budget); a client's
    budget is the client's request and closes hidden under an acting monitor
    (tests/test_client_budgets.py)."""
    bodies = {}
    for mode in ("raw", "on"):
        serve(FixedDrafter(97, 5, tree_mode=False))
        app.STATE["tok"] = MonTok()
        app.STATE["think_budget"] = 4
        app.STATE["monitor"] = monitor(mode, {}) if mode == "on" else None
        with quiet():
            head, body = Req("/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hello"}], "stream": True,
                              "max_tokens": 24}).response()
        assert f"X-Tandem: {mode}" in head, head
        bodies[mode] = [(c["choices"][0]["delta"] if c["choices"] else None,
                         c["choices"][0]["finish_reason"] if c["choices"] else None)
                        for c in (json.loads(ln[6:]) for ln in body.splitlines()
                                  if ln.startswith("data: {"))]
    assert bodies["raw"] == bodies["on"], bodies


# ------------------------------------------------------------------ 3: the header

def test_the_header_lowers_the_mode_and_never_raises_it():
    mon = monitor("on", backstop(4))
    assert mon.mode_for("off") == "raw" and mon.mode_for("shadow") == "shadow"
    assert mon.mode_for("on") == "on" and mon.mode_for(None) == "on"
    assert mon.mode_for(None, n=2) == "shadow", "n > 1 never acts"
    assert monitor("shadow").mode_for("on") == "shadow"
    # over the handler: X-Tandem: off runs nothing, even under a policy that would act
    for mode, header, want in (("on", "off", "raw"), ("on", "raw", "raw"),
                               ("shadow", "on", "shadow"), ("on", None, "on")):
        serve(None)
        app.STATE["tok"] = MonTok()
        mon = app.STATE["monitor"] = monitor(mode, backstop(4))
        req = Req("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                           "max_tokens": 16})
        if header is not None:
            req.headers["X-Tandem"] = header
        with quiet():
            head, _ = req.response()
        assert f"X-Tandem: {want}" in head, (mode, header, head)
        assert (mon.journal.events() == []) == (want == "raw"), (mode, header)
    # a raw server has no monitor at all, and a header cannot make one
    serve(None)
    app.STATE["tok"] = MonTok()
    app.STATE["monitor"] = None
    req = Req("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                       "max_tokens": 8})
    req.headers["X-Tandem"] = "on"
    with quiet():
        head, _ = req.response()
    assert "X-Tandem: raw" in head, head


# ------------------------------------------------------------------ 4: shadow

def test_shadow_journals_the_close_and_writes_what_raw_writes():
    """A test policy that would close the reasoning at token 6: in shadow the journal holds the
    decision with mode shadow, and the output is the raw one, on every path."""
    for name, make, tree in PATHS:
        ref = run(make(), tree)
        mon = monitor("shadow", backstop(6))
        sup = begin(mon, mode="shadow")
        got = run(make(), tree, sup=sup)
        assert got[0] == ref[0] and list(got[2]) == list(ref[2]), name
        evs = mon.journal.events()
        assert len(evs) == 1, "a decision that stands is journaled once, not at every block"
        assert evs[0]["mode"] == "shadow" and evs[0]["decision"] == "CLOSE_REASONING", evs
        assert mon.journal.actions() == [] and not sup.acted and sup.hidden_tokens == 0


# ------------------------------------------------------------------ 5: the hidden close

def _chat(fmt: str, stream: bool, mode: str = "on", tasks=None, **extra):
    serve(FixedDrafter(97, 5, tree_mode=False))
    app.STATE["tok"] = MonTok()
    app.STATE["reasoning_format"] = fmt
    mon = app.STATE["monitor"] = monitor(mode, backstop(6) if tasks is None else tasks)
    # SRV-617: the format named, as an unnamed raw client otherwise gets `reasoning_content`
    body = dict({"messages": [{"role": "user", "content": "hi"}], "stream": stream,
                 "max_tokens": 30, "reasoning_format": fmt}, **extra)
    with quiet():
        head, raw = Req("/v1/chat/completions", body).response()
    return mon, head, raw


def _fields(raw: str, stream: bool) -> tuple[str, str]:
    """(content, reasoning_content) of a response."""
    if not stream:
        m = json.loads(raw)["choices"][0]["message"]
        return m.get("content") or "", m.get("reasoning_content") or ""
    c = r = ""
    for ln in raw.splitlines():
        if ln.startswith("data: {"):
            for ch in json.loads(ln[6:])["choices"]:
                c += ch["delta"].get("content") or ""
                r += ch["delta"].get("reasoning_content") or ""
    return c, r


def test_a_hidden_close_streams_only_the_closer():
    """The close phrase is forwarded (it is in the context the engine holds) and never reaches
    the client: in `tags` the client sees the reasoning, one `</think>`, then the answer; in
    `reasoning_content` the reasoning channel simply ends; `both` is `tags` plus the channel;
    the JSON answer is the stream."""
    phrase = MonTok().encode(CLOSE_TEXT)
    for fmt in ("tags", "reasoning_content", "both"):
        for stream in (True, False):
            mon, head, raw = _chat(fmt, stream)
            content, reasoning = _fields(raw, stream)
            assert "act now" not in content + reasoning, (fmt, stream, content, reasoning)
            ctx = list(app.STATE["last_ctx"])
            assert any(ctx[i:i + len(phrase) - 3] == phrase[:-3] for i in range(len(ctx))), \
                "the hidden phrase is in the engine's context"
            if fmt == "reasoning_content":
                assert "</think>" not in content + reasoning, (stream, content, reasoning)
                assert reasoning, "the reasoning before the close still goes out"
            else:
                assert content.startswith("<think>\n") and content.count("</think>") == 1, content
            ev = mon.journal.actions()
            assert [e["decision"] for e in ev] == ["CLOSE_REASONING"], ev
            assert ev[0]["hidden_tokens"] == len(MonTok().encode("\n\nact now.\n")), ev


def test_a_hidden_close_is_not_counted_and_not_cached():
    """`completion_tokens` counts the streamed tokens only; the response cache does not keep an
    acted answer; the suffix store never receives a hidden token."""
    from engine.cache import ResponseCache

    class Suffix:
        def __init__(self):
            self.got = []

        def append_turn(self, prompt, out):
            self.got.append((list(prompt), list(out)))

        def append(self, out):
            self.got.append(([], list(out)))

    serve(FixedDrafter(97, 5, tree_mode=False))
    app.STATE["tok"] = MonTok()
    app.STATE["monitor"] = monitor("on", backstop(6))
    app.STATE["response_cache"] = rc = ResponseCache(1 << 20, 3600)
    app.STATE["suffix_store"] = sfx = Suffix()
    app.STATE["suffix_scope"] = "all"
    with quiet():
        _, raw = Req("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                              "max_tokens": 30}).response()
    body = json.loads(raw)
    hidden = MonTok().encode("\n\nact now.\n")
    out = sfx.got[0][1]
    assert body["usage"]["completion_tokens"] == len(out), (body["usage"], len(out))
    assert not any(out[i:i + len(hidden)] == hidden for i in range(len(out))), out
    assert rc.report()["entries"] == 0, rc.report()


# ------------------------------------------------------------------ 6: held tool calls

CALLS = ("Hmm.</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n"
         "</function>\n</tool_call>\n<tool_call>\n<function=bash>\n<parameter=cmd>\nls -la\n"
         "</parameter>\n</function>\n</tool_call>")


def _scripted(text: str):
    return lambda *a, **k: iter(MonTok().encode(text))


def _stream_calls(tasks, stream=True):
    serve(None)
    app.STATE["tok"] = MonTok()
    mon = app.STATE["monitor"] = monitor("on", tasks)
    real = app.generate_stream
    app.generate_stream = _scripted(CALLS)
    try:
        with quiet():
            head, raw = Req("/v1/chat/completions", {
                "messages": [{"role": "user", "content": "go"}], "stream": stream,
                "tools": [{"type": "function", "function": {"name": "read"}},
                          {"type": "function", "function": {"name": "bash"}}]}).response()
    finally:
        app.generate_stream = real
    return mon, raw


def test_held_calls_are_released_whole_with_fresh_ids_and_indexes_from_zero():
    """With a tool-call task that can act, a call's arguments do not stream as the model writes
    them: each call arrives in ONE delta at its `</tool_call>`, whole, with an id and the next
    index; the finish reason is tool_calls."""
    with rule("t_pass", "SP3", lambda ev, task, sup: None):
        mon, raw = _stream_calls({"C": {"rule": "t_pass", "mode": "act"}})
    deltas = [d for ln in raw.splitlines() if ln.startswith("data: {")
              for c in json.loads(ln[6:])["choices"] for d in c["delta"].get("tool_calls") or []]
    assert [d["index"] for d in deltas] == [0, 1], deltas
    assert [d["function"]["name"] for d in deltas] == ["read", "bash"]
    assert [json.loads(d["function"]["arguments"]) for d in deltas] == [{"path": "a.txt"},
                                                                          {"cmd": "ls -la"}]
    assert all(d["id"].startswith("call_") and len(d["id"]) == 29 for d in deltas)
    assert len({d["id"] for d in deltas}) == 2


def test_held_calls_a_refused_one_takes_the_whole_answer_with_it():
    """SRV-113: a call the SP3 check refuses takes every call of its answer with it -- none is
    sent. This answer ended without its end token, so the model cannot write the set again
    (that is tests/test_toolcheck.py): the turn gives up with the policy's sentence instead."""
    def refuse_read(ev, task, sup):
        return Action(REGENERATE_CALL, reason="test", text="no") if ev.name == "read" else None

    with rule("t_refuse", "SP3", refuse_read):
        mon, raw = _stream_calls({"C": {"rule": "t_refuse", "mode": "act"}})
        deltas = [d for ln in raw.splitlines() if ln.startswith("data: {")
                  for c in json.loads(ln[6:])["choices"] for d in c["delta"].get("tool_calls") or []]
        assert deltas == [] and '"read"' not in raw and '"bash"' not in raw, deltas
        assert mon.journal.actions()[-1]["decision"] == GIVE_UP
        assert mon.journal.actions()[-1]["cap"] == "ended"
        # the same answer not streamed
        mon, raw = _stream_calls({"C": {"rule": "t_refuse", "mode": "act"}}, stream=False)
        msg = json.loads(raw)["choices"][0]["message"]
        assert not msg.get("tool_calls") and msg["content"].endswith(
            mon.policy.texts["give_up"]), msg


def test_without_a_tool_task_calls_stream_live_as_before():
    mon, raw = _stream_calls({})
    frags = [d for ln in raw.splitlines() if ln.startswith("data: {")
             for c in json.loads(ln[6:])["choices"] for d in c["delta"].get("tool_calls") or []]
    assert len(frags) > 2, "arguments stream in fragments when nothing holds them"


# ------------------------------------------------------------------ 7: the end token dropped

def _continue_once(ev, task, sup):
    if sup.memo.get("done"):
        return None
    sup.memo["done"] = True
    return Action(CONTINUE_HIDDEN, reason="test", text="go on", detail={"case": "B"})


def test_the_end_token_is_dropped_cleanly_on_every_path():
    """An end token inside an accepted verify block (chain and tree, a drafter that is always
    right), and on the single-token step: the supervisor continues past it once. Right after the
    hidden forward the loop's invariant holds -- the KV holds every token of the context but the
    last -- and decoding goes on exactly as a fresh greedy decode of that context would."""
    prompt = [2, 40, 17, 66, 30]            # its first twelve tokens are all different
    ref, _, _ = run(None, prompt=prompt)
    inside = 0
    with rule("t_cont", "SP2", _continue_once):
        for stop_at in (3, 6, 8, 9, 12):
            stop = ref[stop_at]
            if stop in ref[:stop_at]:
                continue
            # an oracle block is the anchor and five drafts, all accepted, and one bonus token:
            # out[1..6], out[7..12], ... -- a stop at 3, 8 or 9 is a draft INSIDE a block
            inside += stop_at % 6 not in (0, 1)
            for name, make, tree in (
                    ("chain", lambda: Oracle(ref, len(prompt), 5, False), False),
                    ("tree", lambda: Oracle(ref, len(prompt), 5, True), True),
                    ("single", lambda: None, False)):
                eng = serve(make(), tree=tree)
                app.STATE["tok"] = MonTok()
                mon = monitor("on", {"T": {"rule": "t_cont", "mode": "act"}})
                sup = begin(mon)
                think = ThinkBudget(MonTok())
                gen = app.generate_stream(torch.tensor(prompt), 30, {stop}, think, sup=sup)
                hidden = MonTok().encode(policy().templates["hidden_turn"].replace(
                    "{text}", "go on"))
                out, snap = [], None
                with quiet():
                    for t in gen:
                        out.append(t)
                        ctx = app.STATE["last_ctx"]
                        if snap is None and ctx[len(ctx) - 1 - len(hidden):-1] == hidden:
                            # the first token after the hidden forward
                            assert eng.kv.length == len(ctx) - 1, (name, eng.kv.length, len(ctx))
                            snap = (list(ctx), len(out))
                assert snap is not None and sup.acted, (name, stop_at, out)
                assert stop not in out[:snap[1]], "the end token never reaches the client"
                ctx = snap[0]
                # the continuation: a fresh greedy decode of the context before its last token
                fresh, _, _ = run(None, prompt=ctx[:-1], max_new=12)
                got = out[snap[1] - 1:][:12]
                assert len(got) >= 6 and got == fresh[:len(got)], (name, stop_at, got, fresh)
    assert inside == 3, "the end token must fall inside an accepted block three times"


def test_a_hidden_forward_that_does_not_fit_is_not_made():
    """No room in the context for the hidden tokens: PASS, journaled as no_room."""
    mon = monitor("on", {"E": {"rule": "empty_turn", "mode": "act", "close": CLOSE_TEXT,
                               "nudges": ["go"], "tries": 2}})
    sup = begin(mon, tools=[{"type": "function", "function": {"name": "f"}}])
    assert sup.at_finish(FinishEvent("eos", True, "", 10, 11), room=5) is None
    assert mon.journal.events()[-1]["cap"] == "no_room"


# ------------------------------------------------------------------ 8: caps

def _always(ev, task, sup):
    return Action(CONTINUE_HIDDEN, reason="always", text=f"n{sup.acts}", detail={"case": "B"})


def test_caps_two_acts_a_request_then_give_up():
    with rule("t_always", "SP2", _always):
        mon = monitor("on", {"A": {"rule": "t_always", "mode": "act"}})
        sup = begin(mon)
        ev = FinishEvent("eos", False, "", 0, 1)
        kinds = [sup.at_finish(ev, room=1 << 20) for _ in range(3)]
        assert [a.kind for a in kinds] == [CONTINUE_HIDDEN, CONTINUE_HIDDEN, GIVE_UP], kinds
        assert kinds[2].text == policy().texts["give_up"] == sup.gave_up
        assert mon.journal.events()[-1]["cap"] == "per_request"


def test_caps_six_acts_a_session_hour():
    with rule("t_always", "SP2", _always):
        mon = monitor("on", {"A": {"rule": "t_always", "mode": "act"}})
        for i in range(3):
            sup = begin(mon)
            a, b = (sup.at_finish(FinishEvent("eos", False, "", 0, 1), room=1 << 20)
                    for _ in range(2))
            assert (a.kind, b.kind) == (CONTINUE_HIDDEN, CONTINUE_HIDDEN), i
        sup = begin(mon)                    # the same session: 6 acts in the last hour
        a = sup.at_finish(FinishEvent("eos", False, "", 0, 1), room=1 << 20)
        assert a.kind == GIVE_UP and mon.journal.events()[-1]["cap"] == "per_session_hour"


def test_caps_never_the_same_hidden_action_twice_in_a_row():
    tasks = {"E": {"rule": "empty_turn", "mode": "act", "close": CLOSE_TEXT,
                   "nudges": ["same", "same"], "tries": 2}}
    mon = monitor("on", tasks)
    sup = begin(mon, tools=[{"type": "function", "function": {"name": "f"}}])
    ev = FinishEvent("eos", False, "", 0, 1)
    a, b = sup.at_finish(ev, room=1 << 20), sup.at_finish(ev, room=1 << 20)
    assert (a.kind, b.kind) == (CONTINUE_HIDDEN, GIVE_UP)
    assert mon.journal.events()[-1]["cap"] == "same_in_a_row"
    # allowed by the policy, it acts twice
    mon = monitor("on", tasks, caps={"same_action_in_a_row": True})
    sup = begin(mon, tools=[{"type": "function", "function": {"name": "f"}}])
    assert [sup.at_finish(ev, room=1 << 20).kind for _ in range(2)] == [CONTINUE_HIDDEN] * 2


def test_caps_give_up_on_the_wire():
    """An always-acting policy on the real loop and the handler: after two acts the response
    ends with the policy's GIVE_UP sentence as content and finish stop."""
    with rule("t_always", "SP2", _always):
        for stream in (True, False):
            serve(None)
            app.STATE["tok"] = MonTok()
            app.STATE["monitor"] = monitor("on", {"A": {"rule": "t_always", "mode": "act"}})
            app.STATE["cfg_eos"] = list(range(97))      # every token the model draws ends it
            with quiet():
                _, raw = Req("/v1/chat/completions", {
                    "messages": [{"role": "user", "content": "hi"}], "stream": stream,
                    "max_tokens": 60, "chat_template_kwargs": {"enable_thinking": False}}
                ).response()
            content, _ = _fields(raw, stream)
            assert content.endswith(policy().texts["give_up"]), (stream, content)
            fin = (json.loads(raw)["choices"][0]["finish_reason"] if not stream else
                   [json.loads(ln[6:])["choices"][0]["finish_reason"] for ln in raw.splitlines()
                    if ln.startswith("data: {") and json.loads(ln[6:])["choices"]][-1])
            assert fin == "stop", fin


# ------------------------------------------------------------------ 9: sessions

def test_the_session_of_two_requests_without_an_id():
    s = Sessions(max_sessions=32, join_window_s=1800, lineage_tokens=4096)
    head = list(range(5000))
    k1, how1 = s.key_for(None, head + [1, 2], True, [{"role": "user", "content": "a"}], now=0)
    s.touch(k1, now=0)
    k2, how2 = s.key_for(None, head + [3, 4, 5], True, [{"role": "user", "content": "a"},
                                                        {"role": "user", "content": "b"}],
                         now=300)
    assert k1 == k2 and how1 == how2 == "lineage"
    st = s.touch(k2, now=300)
    assert st.requests == 2
    # a different head is a different session; 31 minutes later the same head is a fresh one
    k3, _ = s.key_for(None, [9] * 5000, True, [], now=301)
    assert k3 != k1
    st = s.touch(k1, now=300 + 1860)
    assert st.requests == 1, "outside the join window the state starts again"


def test_the_session_a_conversation_id_wins_and_a_compaction_joins():
    s = Sessions(max_sessions=32, join_window_s=1800, lineage_tokens=4096)
    assert s.key_for("abc", [1], True, [], now=0)[1] == "conversation"
    assert s.key_for("abc", [1], True, [], now=0)[0] == s.key_for("abc", [2], True, [], now=5)[0]
    k1, _ = s.key_for(None, [1] * 50, True, [{"role": "user", "content": "build"}], now=0)
    s.touch(k1, now=0)
    # opencode's compaction request: no tools, one user message -- it joins the last session
    kc, how = s.key_for(None, [7] * 50, False, [{"role": "system", "content": "s"},
                                                {"role": "user", "content": "summarise"}],
                        now=60)
    assert (kc, how) == (k1, "compaction") and s.get(k1).last_compaction == 60
    # and the next request with a prefix nobody has seen joins it too
    kn, how = s.key_for(None, [8] * 50, True, [{"role": "user", "content": "go on"}], now=90)
    assert (kn, how) == (k1, "compaction")


def test_the_session_state_is_an_lru_of_hashes():
    s = Sessions(max_sessions=2, join_window_s=1800, lineage_tokens=8)
    for i in range(3):
        s.touch(f"k{i}", now=i)
    assert len(s) == 2 and s.get("k0") is None
    st = s.touch("k2", now=5)
    st.record_call("read", '{"b": 1, "a": 2}', 5)
    st.record_call("read", '{"a":2,"b":1}', 6)
    assert st.calls[0][2] == st.calls[1][2], "the canonical arguments hash the same"
    st.record_results([{"role": "assistant", "content": ""},
                       {"role": "tool", "tool_call_id": "c1", "content": "ok"}], 7)
    assert len(st.results) == 1 and "ok" not in repr(st.results)


# ------------------------------------------------------------------ SP4: the prompt's tail

def test_sp4_a_tail_note_and_thinking_off_touch_only_the_tail():
    tok = MonTok()
    prompt = tok.encode("<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n"
                        "<|im_start|>assistant\n<think>\n")
    note = tok.encode("\n\nnote")
    got = tail_note(prompt, note, [IM_START], [IM_END])
    assert tok.decode(got, skip_special_tokens=False).endswith(
        "user\nhello\n\nnote<|im_end|>\n<|im_start|>assistant\n<think>\n")
    k = prompt.index(IM_END, prompt.index(IM_END) + 1)
    assert got[:k] == prompt[:k], "everything before the last message's end is untouched"
    off = think_off(prompt, tok.encode("<think>\n"), tok.encode("<think>\n\n</think>\n\n"))
    assert off == tok.encode("<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhello"
                             "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert think_off(off, tok.encode("<think>\n"), [1]) is None, "only an open block"
    assert tail_note([1, 2, 3], note, [IM_START], [IM_END]) is None


def test_sp4_on_the_real_chat_template():
    """With the Qwen3.8 template and tokenizer: thinking off by splice is exactly the prompt the
    template renders with `enable_thinking: false` (effort medium, as served), and a note lands
    in the last user message."""
    try:
        import jinja2
        from tokenizers import Tokenizer
    except ImportError:
        print("  (skipped: no jinja2/tokenizers)")
        return
    snap = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots")
    tj = next((os.path.join(snap, d, "tokenizer.json") for d in sorted(os.listdir(snap))
               if os.path.exists(os.path.join(snap, d, "tokenizer.json"))), None) \
        if os.path.isdir(snap) else None
    if tj is None:
        print("  (skipped: no Qwen3.8 tokenizer in the HF cache)")
        return
    tk = Tokenizer.from_file(tj)
    env = jinja2.Environment()
    env.filters["tojson"] = lambda v, **k: json.dumps(v)
    tpl = env.from_string(open(os.path.join(HERE, "fixtures", "qwen38_chat_template.jinja")).read())

    def render(**kw):
        return tpl.render(messages=[{"role": "system", "content": "Be brief."},
                                    {"role": "user", "content": "hello"}],
                          tools=[{"type": "function", "function": {"name": "read"}}],
                          add_generation_prompt=True, reasoning_effort="medium", **kw)

    def ids(text):
        return tk.encode(text, add_special_tokens=False).ids

    t = policy().templates
    on, off = ids(render(enable_thinking=True)), ids(render(enable_thinking=False))
    assert think_off(on, ids(t["think_open"]), ids(t["think_off"])) == off
    got = tail_note(on, ids(t["tail_note"].replace("{text}", "Use the skill.")),
                    ids(t["turn_start"]), ids(t["turn_end"]))
    text = tk.decode(got, skip_special_tokens=False)
    assert text.endswith("hello\n\nUse the skill.<|im_end|>\n<|im_start|>assistant\n<think>\n")


def test_sp4_actions_over_the_handler():
    """A note at the tail and thinking off from SP4 rules: the prompt the engine gets carries
    them, `prompt_tokens` is what the client sent, and a budget closes the reasoning hidden."""
    seen = {}

    def spy(prompt, *a, **k):
        seen["prompt"] = prompt.tolist()
        return iter(MonTok().encode("Hello."))

    def note(ev, task, sup):
        return Action(PROMPT_TAIL_NOTE, reason="t", text="use skill")

    def off(ev, task, sup):
        return Action(SET_THINK_BUDGET, reason="t", budget=0)

    with rule("t_note", "SP4", note), rule("t_off", "SP4", off):
        serve(None)
        app.STATE["tok"] = MonTok()
        mon = app.STATE["monitor"] = monitor("on", {"N": {"rule": "t_note", "mode": "act"},
                                                    "O": {"rule": "t_off", "mode": "act"}})
        real = app.generate_stream
        app.generate_stream = spy
        try:
            with quiet():
                _, raw = Req("/v1/chat/completions", {"messages": [
                    {"role": "user", "content": "hello"}], "max_tokens": 8}).response()
        finally:
            app.generate_stream = real
    text = MonTok().decode(seen["prompt"], skip_special_tokens=False)
    assert text.endswith("hello\n\nuse skill<|im_end|>\n<|im_start|>assistant\n"
                         "<think>\n\n</think>\n\n"), text
    body = json.loads(raw)
    client = len(MonTok().encode("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
                                 "<think>\n"))
    assert body["usage"]["prompt_tokens"] == client, body["usage"]
    assert body["choices"][0]["message"]["content"] == "Hello.", "thinking off: no <think> tag"
    assert [e["decision"] for e in mon.journal.actions()] == [PROMPT_TAIL_NOTE, SET_THINK_BUDGET]


def test_sp4_a_budget_closes_the_reasoning_hidden():
    def budget(ev, task, sup):
        return Action(SET_THINK_BUDGET, reason="t", budget=4, text=CLOSE_TEXT)

    with rule("t_budget", "SP4", budget):
        mon = monitor("on", {"T": {"rule": "t_budget", "mode": "act"}})
        sup = begin(mon)
        acts = sup.at_request(sup.req)
        assert [a.kind for a in acts] == [SET_THINK_BUDGET] and sup.think_budget == 4
        out, eng, ctx = run(FixedDrafter(97, 5, tree_mode=False), sup=sup)
        text = MonTok().decode(out, skip_special_tokens=False)
        assert "</think>" in text and "act now" not in text, text
        assert [e["reason"] for e in mon.journal.actions()][-1] == "budget(4)"


# ------------------------------------------------------------------ the rules' own contract

def test_a_pass_with_a_reason_is_an_observation_not_an_action():
    with rule("t_obs", "SP1", lambda ev, task, sup: Action(PASS, reason="seen")):
        mon = monitor("on", {"O": {"rule": "t_obs", "mode": "act"}})
        sup = begin(mon)
        assert sup.at_block(BlockEvent(5, True, None, None, 5), room=1 << 20) is None
        assert mon.journal.events()[0]["decision"] == PASS and not sup.acted


def test_an_sp3_event_and_the_session_record():
    with rule("t_pass", "SP3", lambda ev, task, sup: None):
        mon = monitor("on", {"C": {"rule": "t_pass", "mode": "act"}})
        sup = begin(mon)
        assert sup.holds_calls and sup.at_tool_call(ToolCallEvent("read", "{}", 0)) is None
        assert not begin(monitor("shadow", {"C": {"rule": "t_pass", "mode": "act"}}),
                         mode="shadow").holds_calls, "shadow never holds"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
