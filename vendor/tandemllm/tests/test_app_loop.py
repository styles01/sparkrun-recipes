"""The serving loop and its HTTP handler, on a CPU, with the random 4-layer model.

`server/app.py`'s `generate_stream` is its own copy of the decode loop -- a generator, so a token
reaches the socket the moment it is accepted -- and the gate tools never run it: they drive
`engine/spec.py`. So the claims about the SERVED loop are tested here, against the loop itself:

  * a pattern-stop hit inside the forced reasoning close ends the stream there, on every verify
    path, with the KV holding exactly what the stream says was written;
  * `X-Engine-Stop` reaches a non-streamed client, and a streamed one gets the marker in band
    instead, since its headers left before the guard fired;
  * the single-token step closes the reasoning block like the verified ones, and brings
    the drafter current like engine/spec.py's does;
  * a seed reproduces a sampled request whatever the drafter and the width choices do;
  * a graceful stop lets the generation in flight finish;
  * a failed non-streamed request is logged and counted like a streamed one;
  * a tool call written inside the reasoning block is not a call;
  * with thinking on, nothing a client can time as a first token goes out before the engine's
    first token: the synthetic `<think>` rides on the first text, and thinking off is untouched;
  * nor does the role chunk, thinking on or off: it goes out with the first text;
  * a streamed stop string never reaches the client, not even the part of it that arrives before
    the match, and a streamed answer with `stop` is the non-streamed one;
  * the `both` format closes the `<think>` it opens in `content`;
  * the `[req]` line carries the loop's own block count, decode time and first-miss histogram,
    for the chain and the tree, and a request that never generated carries none;

Run: python tests/test_app_loop.py
"""

from __future__ import annotations

import io
import json
import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.penalty import PenaltySpec  # noqa: E402
from engine.spec import Relax, ThinkBudget  # noqa: E402
from server import app  # noqa: E402
from test_window_edge import FixedDrafter, _engine  # noqa: E402

OPEN, CLOSE = 90, 91
CLOSING = [80, 81, CLOSE]          # the forced phrase; like the real one it ends with the closer


class FakeTok:
    """Just enough tokenizer for the loop, the think budget and the handler."""

    unk_token_id = -1
    eos_token_id = None

    def convert_tokens_to_ids(self, text):
        return {"<think>": OPEN, "</think>": CLOSE}.get(text, -1)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        if text == "<think>":
            ids = [60, 61]
        elif text == "</think>":
            ids = [62, 63]
        elif text == ThinkBudget.PHRASE:
            ids = list(CLOSING)
        else:
            ids = [1 + (ord(c) % 50) for c in text]
        if return_tensors == "pt":
            ids = torch.tensor([ids])
        return type("R", (), {"input_ids": ids})()

    def decode(self, seq, skip_special_tokens=True):
        return "".join(chr(ord("a") + int(t) % 26) for t in seq)


def serve(drafter=None, tree=False, max_len=256, k=7):
    eng = _engine(max_len)
    app.STATE.clear()
    app.STATE.update(engine=eng, drafter=drafter, k=k, tree=tree, sampled_tree=False,
                     relax=Relax(1.0, 1), verbose=False, state_store=None, prefix_cache=False,
                     prefix_chunk=0, session_cache=False, tok=FakeTok(), device="cpu",
                     max_len=max_len, default_max_tokens=64, pen_spec=PenaltySpec(), model="t",
                     cfg_eos=None, think_budget=0, think_stall=False, reasoning_format="tags",
                     request_timeout=0.0, max_queue=8, queue_timeout=5.0, pattern_stop=None,
                     response_cache=None, suffix_store=None)
    return eng


class StopOn:
    """A pattern guard that fires on one exact block -- the forced close, here."""

    def __init__(self, block):
        self.block, self.hit, self.label = list(block), False, "stub"

    def observe(self, new):
        if list(new) == self.block:
            self.hit = True
        return self.hit


# ------------------------------------------------------------------

def _forced_close_with_guard(drafter, tree):
    eng = serve(drafter, tree=tree)
    prompt = torch.tensor([5, 6, 7, 8, OPEN])
    think = ThinkBudget(FakeTok(), budget=3)
    guard = StopOn(CLOSING)
    out = list(app.generate_stream(prompt, 60, set(), think, pstop=guard))
    ctx = app.STATE["last_ctx"]
    return eng, out, ctx, guard


def test_a_guard_hit_in_the_forced_close_ends_the_stream():
    """`_force_close` returned early when the closing phrase tripped the pattern guard,
    and the caller carried on decoding as if nothing had happened: the guard's stop was lost, and
    the next step forwarded `ctx[-1]` -- the phrase's last token, already in the KV -- a second
    time, one row further on. The stream must end on the phrase, with every token in `ctx` in the
    KV exactly once."""
    for label, drafter, tree in (("chain", FixedDrafter(97, 5, tree_mode=False), False),
                                 ("tree", FixedDrafter(97, 5, tree_mode=True), True)):
        eng, out, ctx, guard = _forced_close_with_guard(drafter, tree)
        assert guard.hit, label
        assert out[-len(CLOSING):] == CLOSING, f"{label}: the stream must end on the phrase: {out}"
        assert out.count(CLOSING[0]) >= 1 and len(out) < 30, f"{label}: decoding went on: {out}"
        assert eng.kv.length == len(ctx), (
            f"{label}: kv.length {eng.kv.length} vs len(ctx) {len(ctx)} -- the forced forward "
            f"wrote every token of ctx, and nothing may be written after it")


# ------------------------------------------------------------------ the single-token step

def test_the_reasoning_close_fires_on_the_single_token_path():
    """The budget and the stall signal closed the block only after a VERIFIED block; the
    single-token step -- every step of a drafter-less server, and a declined step of any other --
    observed the token and never looked at `think.hit`. A drafter-less server never closed a
    reasoning block at all. With the guard on the phrase, the stream also ends there."""
    serve(None)
    think = ThinkBudget(FakeTok(), budget=3)
    out = list(app.generate_stream(torch.tensor([5, 6, 7, 8, OPEN]), 30, set(), think))
    assert think.done, f"the block was never closed: {out}"
    i = next(i for i in range(len(out)) if out[i:i + len(CLOSING)] == CLOSING)
    assert i == 4, f"closed after the 4th token (3 in the budget + the one that crossed it): {out}"
    assert len(out) > i + len(CLOSING), "the answer follows the phrase"
    eng, out, ctx, guard = _forced_close_with_guard(None, False)
    assert guard.hit and out[-len(CLOSING):] == CLOSING and eng.kv.length == len(ctx), out


class Cursor:
    """A block drafter's position discipline, as DFlash2 has it: its cache covers the positions it
    was synced for, and it declines whenever the anchor is not the first position it lacks."""

    def __init__(self, decline_on):
        self.ctx_len, self.calls, self.decline_on = 0, 0, decline_on
        self.covered, self.after = set(), 0

    def reset(self):
        pass

    def observe(self, tokens):
        pass

    def sync(self, tokens, hidden, first_pos, rows=None):
        self.covered.update(range(first_pos, first_pos + len(tokens)))
        self.ctx_len = first_pos + len(tokens)

    def propose(self, ctx, k):
        self.calls += 1
        if self.calls == self.decline_on or len(ctx) - 1 != self.ctx_len:
            return []
        if self.calls > self.decline_on:
            self.after += 1
        return [(len(ctx) * 7 + i) % 97 for i in range(3)]


def test_a_declined_step_keeps_the_drafter_current():
    """A declined step forwards one token and did not sync the drafter, as engine/spec.py
    does. A drafter whose cache is indexed by position -- the block drafter -- was then one
    position behind for good, declined every later step, and the request finished one token a
    forward. After one decline the drafter must be current again and proposing."""
    dr = Cursor(decline_on=3)
    eng = serve(dr)
    out = list(app.generate_stream(torch.tensor([5, 6, 7, 8, 9]), 40, set()))
    assert len(out) == 40
    assert dr.after > 0, "the drafter never proposed again after its one decline"
    committed = len(app.STATE["last_ctx"]) - 1         # the last token is decided, not forwarded
    missing = set(range(committed)) - dr.covered
    assert not missing, f"positions the drafter was never synced for: {sorted(missing)}"


# ------------------------------------------------------------------

class QDrafter:
    """A drafter that SAMPLES its proposals from its own distribution, as DFlash2 does under a
    sampled request, and carries q for the accept. Its q is deliberately unlike the target's."""

    def __init__(self, width, seed=4):
        self.width, self.sampler, self.last_q = width, None, None
        g = torch.Generator().manual_seed(seed)
        self.q = torch.softmax(torch.randn(97, generator=g) * 3.0, dim=-1)

    def reset(self):
        pass

    def observe(self, tokens):
        pass

    def set_sampling(self, sampler):
        self.sampler = sampler if sampler is not None and sampler.on else None

    def propose(self, ctx, k):
        s, n = self.sampler, min(k, self.width)
        if s is None:
            return [int(self.q.argmax())] * n
        self.last_q = [self.q] * n
        if getattr(s, "coupled", False):
            return [s.pick_at(self.q, len(ctx) + i) for i in range(n)]
        return [s.pick(self.q) for _ in range(n)]


class Wobble(FixedDrafter):
    """Changes its block width from step to step, the way the length router's latch lands on a
    different width when the box runs at a different speed."""

    def __init__(self, seed):
        super().__init__(97, 3, tree_mode=False)
        self.g = torch.Generator().manual_seed(seed)

    def propose(self, ctx, count):
        self.width = int(torch.randint(1, 9, (1,), generator=self.g))
        return super().propose(ctx, count)


def _sampled(drafter, tree=False, seed=42):
    serve(drafter, tree=tree)
    app.STATE["sampled_tree"] = tree
    s = app.Sampler(temperature=0.9, top_p=0.95, seed=seed)
    return list(app.generate_stream(torch.tensor([5, 6, 7, 8, 9]), 48, set(), sampler=s))


def test_a_seed_reproduces_the_request_whatever_the_drafter_does():
    """One generator stream fed both the drafter's proposals and the accept's draws, in an
    order the drafter decided -- and the served length router decides its width from wall-clock
    timing. Two runs of one seeded request under different load drew differently and diverged.
    With position-keyed draws the output is a function of (prompt, params, seed) alone: no
    drafter, two fixed widths, two different width schedules, a sampling drafter at two widths,
    and the tree walk all emit the same tokens."""
    ref = _sampled(None)
    runs = {
        "chain width 3": _sampled(FixedDrafter(97, 3, tree_mode=False)),
        "chain width 7": _sampled(FixedDrafter(97, 7, tree_mode=False)),
        "width schedule A": _sampled(Wobble(1)),
        "width schedule B": _sampled(Wobble(2)),
        "sampling drafter 4": _sampled(QDrafter(4)),
        "sampling drafter 8": _sampled(QDrafter(8)),
        "tree walk": _sampled(FixedDrafter(97, 5, tree_mode=True), tree=True),
    }
    bad = {label: out for label, out in runs.items() if out != ref}
    assert not bad, f"diverged from the drafter-less run {ref}: {bad}"
    assert _sampled(None, seed=43) != ref, "and the seed is what the output depends on"


# ------------------------------------------------------------------ nit 5

class Req(app.Handler):
    """The handler without a socket: the body in, the raw response bytes out."""

    def __init__(self, path, body):
        raw = json.dumps(body).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = {"Content-Length": str(len(raw))}
        self.path, self.command, self.request_version = path, "POST", "HTTP/1.1"
        self.requestline, self.client_address = "POST " + path, ("127.0.0.1", 0)
        self.close_connection = True

    def response(self):
        self.do_POST()
        head, _, body = self.wfile.getvalue().partition(b"\r\n\r\n")
        return head.decode(), body.decode()


class HitAfter:
    """Stands in for `PatternStop`: fires on its third block."""

    def __init__(self, *_):
        self.n, self.hit, self.label = 0, False, "size=1 count=2 tok=[1]"

    def observe(self, new):
        self.n += 1
        self.hit = self.hit or self.n >= 3
        return self.hit


def test_the_stop_header_is_non_streaming_only_and_the_stream_says_it_in_band():
    real = app.PatternStop
    app.PatternStop = HitAfter
    try:
        serve()
        app.STATE["pattern_stop"] = (1, 1, 2)
        head, body = Req("/v1/completions", {"prompt": "hello", "max_tokens": 20}).response()
        assert "X-Engine-Stop: pattern-stop(" in head, head
        assert app.GUARD_MARKER in json.loads(body)["choices"][0]["text"]
        head, body = Req("/v1/completions", {"prompt": "hello", "max_tokens": 20,
                                             "stream": True}).response()
        assert "X-Engine-Stop" not in head, "a streamed response's headers precede the guard"
        text = "".join(json.loads(line[6:])["choices"][0]["text"]
                       for line in body.splitlines()
                       if line.startswith("data: {") and json.loads(line[6:])["choices"])
        assert text.endswith(app.GUARD_MARKER), "the streamed client gets the stop in band"
    finally:
        app.PatternStop = real



# ------------------------------------------------------------------

def test_a_failed_non_streamed_request_is_logged_and_counted():
    """`_log_request` is "one line per generation, always, whatever happened to it", and
    it is what counts `errors` for /health and /metrics. The streamed path calls it on an
    exception; the non-streamed one let the exception go straight to `do_POST`'s 500, so a failed
    JSON request left no [req] line and no count -- /health said errors: 0 over a failing server."""
    import contextlib

    def broken(*a, **k):
        yield 5
        raise RuntimeError("the engine fell over")

    serve()
    real = app.generate_stream
    app.generate_stream = broken
    before = app.INFLIGHT["errors"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            head, body = Req("/v1/completions", {"prompt": "hello", "max_tokens": 8}).response()
    finally:
        app.generate_stream = real
    assert head.startswith("HTTP/1.1 500"), head
    assert app.INFLIGHT["errors"] == before + 1, "the failure must be counted"
    assert "finish=error" in out.getvalue() and "RuntimeError" in out.getvalue(), out.getvalue()


# ------------------------------------------------------------------

class CharTok(FakeTok):
    """One token per character, so a scripted generation can spell any text -- tags included."""

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        text = "".join(m["content"] for m in messages) + "\n<think>\n"
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}

    def decode(self, seq, skip_special_tokens=True):
        return "".join(chr(int(t)) for t in seq)


THOUGHT = ("I could call <tool_call>\n<function=delete_file>\n<parameter=path>\n/home/user/a.txt"
           "\n</parameter>\n</function>\n</tool_call> but I should ask first.\n</think>\n\n")


def _chat(answer, stream):
    serve()
    app.STATE["tok"] = CharTok()
    real = app.generate_stream
    app.generate_stream = lambda *a, **k: iter([ord(c) for c in THOUGHT + answer])
    try:
        return Req("/v1/chat/completions",
                   {"messages": [{"role": "user", "content": "tidy up"}], "stream": stream,
                    "reasoning_format": "tags",                                 # SRV-617
                    "tools": [{"type": "function", "function": {"name": "delete_file"}}]}
                   ).response()
    finally:
        app.generate_stream = real


def test_a_tool_call_inside_the_reasoning_is_not_a_call():
    """In `tags` format the reasoning travels in `content`, and both the streamed tool-call
    buffer and the non-streamed parser read all of it: a call the model only DELIBERATES about --
    "I could call delete_file ... but I should ask first" -- went out as a real `tool_calls` entry
    with finish_reason tool_calls, for a client to execute. Only the answer can call a tool; the
    thought stays visible as text."""
    head, body = _chat("Should I delete a.txt?", stream=False)
    msg = json.loads(body)["choices"][0]
    assert "tool_calls" not in msg["message"], msg
    assert msg["finish_reason"] == "length" and "<tool_call>" in msg["message"]["content"]
    head, body = _chat("Should I delete a.txt?", stream=True)
    assert '"tool_calls"' not in body and "finish_reason\": \"tool_calls" not in body, body[-400:]
    # and a call in the ANSWER still is one, on both paths
    call = ("<tool_call>\n<function=delete_file>\n<parameter=path>\n/home/user/a.txt\n"
            "</parameter>\n</function>\n</tool_call>")
    head, body = _chat(call, stream=False)
    msg = json.loads(body)["choices"][0]["message"]
    assert [c["function"]["name"] for c in msg["tool_calls"]] == ["delete_file"], msg
    head, body = _chat(call, stream=True)
    assert body.count('"name": "delete_file"') == 1, body[-600:]


# ------------------------------------------------------------------

class ThinkTok(CharTok):
    """`CharTok` with both of the template's endings: thinking on leaves the block open, thinking
    off opens and closes it inside the prompt."""

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=True,
                            **kw):
        tail = "\n<think>\n" if enable_thinking else "\n<think>\n\n</think>\n\n"
        text = "".join(m["content"] for m in messages) + tail
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}


def _deltas(raw: str) -> list[dict]:
    """The `delta` of every chunk in a raw event stream, finish chunks as `{"finish": ...}`."""
    out = []
    for line in raw.splitlines():
        if not line.startswith("data: {"):
            continue
        for c in json.loads(line[6:])["choices"]:
            out.append(dict(c["delta"], **({"finish": c["finish_reason"]}
                                          if c["finish_reason"] else {})))
    return out


def _text(d: dict) -> str:
    return d.get("content") or d.get("reasoning_content") or ""


def _stream_chat(engine, think=True, fmt="tags", steps=None, **extra):
    """One streamed chat request, and what the client already held when the engine was first
    asked for a token. That call is where the prefill runs -- `generate_stream` is a generator, so
    nothing of it executes before the handler's first `next()` -- and anything already on the
    wire then is something the client can time as a first token the model has not produced.

    `engine(*args)` is the generator the handler's call is forwarded to. `steps`, a list, gets
    the deltas on the wire at every `next()`: [0] before the prefill, [1] once the first token
    has been handled, and so on."""
    serve()
    app.STATE["tok"] = ThinkTok()
    app.STATE["reasoning_format"] = fmt
    # SRV-617: named, since an unnamed raw client gets `reasoning_content` over the server's tags
    body = dict({"messages": [{"role": "user", "content": "q"}], "stream": True,
                 "reasoning_format": fmt,
                 "chat_template_kwargs": {"enable_thinking": think}}, **extra)
    req = Req("/v1/chat/completions", body)
    held = []

    def recorded(*a, **k):
        it = engine(*a, **k)
        while True:
            held.append(req.wfile.getvalue().partition(b"\r\n\r\n")[2].decode())
            try:
                t = next(it)
            except StopIteration:
                return
            yield t

    real = app.generate_stream
    app.generate_stream = recorded
    try:
        head, raw = req.response()
    finally:
        app.generate_stream = real
    assert head.startswith("HTTP/1.1 200"), head
    if steps is not None:
        steps.extend(_deltas(h) for h in held)
    return _deltas(held[0]), _deltas(raw)


def _script(text):
    return lambda *a, **k: iter([ord(c) for c in text])


def test_no_content_goes_out_before_the_first_generated_token():
    """The synthetic `<think>` went out before the loop started -- before the prefill --
    so a client timing its first content delta read the HTTP round trip (0.003 s on the box
    against a 551 ms request). With thinking on, in every format, the client may hold nothing
    with text in it until the engine has produced a token (and since, nothing more)."""
    for fmt in ("tags", "reasoning_content", "both"):
        before, _ = _stream_chat(_script("Hmm.</think>\n\nYes."), fmt=fmt)
        assert not any(_text(d) for d in before), (fmt, before)


# ------------------------------------------------------------------

ROLE = {"role": "assistant", "content": ""}


def test_nothing_with_a_choices_array_goes_out_before_the_prefill():
    """The role chunk went out before the loop, i.e. before the prefill. vLLM's bench
    client stamps TTFT on the first chunk that has a `choices` array, whatever is in it, so against
    this server it read the HTTP round trip as TTFT on every request. Before the engine's first
    `next()` the wire holds the headers and nothing else, thinking on or off, in every format."""
    for think in (True, False):
        for fmt in ("tags", "reasoning_content", "both"):
            before, _ = _stream_chat(_script("Hmm.</think>\n\nYes."), think=think, fmt=fmt)
            assert before == [], (think, fmt, before)


def test_the_role_chunk_goes_out_with_the_first_text():
    """The role chunk is still the first chunk of every stream, byte for byte what it was,
    and it reaches the client in the same write as the first text: once the first token has been
    handled, the client holds the role chunk and the first text delta."""
    steps = []
    _stream_chat(_script("Hmm.</think>\n\nYes."), steps=steps)
    assert steps[1] == [ROLE, {"content": "<think>\nH"}], steps[1]
    steps = []
    _stream_chat(_script("Plain."), think=False, steps=steps)
    assert steps[1] == [ROLE, {"content": "P"}], steps[1]
    steps = []
    _stream_chat(_script("Hmm.</think>\n\nYes."), fmt="reasoning_content", steps=steps)
    assert steps[1] == [ROLE, {"reasoning_content": "H"}], steps[1]
    # a first token that is not a character yet (the detokeniser holds a trailing U+FFFD back)
    # sends nothing, the role chunk included; it goes out with the first text there is
    steps = []
    _stream_chat(lambda *a, **k: iter([0xFFFD, ord("x")]), think=False, steps=steps)
    assert steps[1] == [] and steps[2] == [ROLE, {"content": "\ufffdx"}], steps


def test_a_stream_with_no_text_still_starts_with_the_role_chunk():
    """A stop string at the first character, an end of sequence as the first token, a
    failure before the first token: the role chunk still goes out, first, before the finish."""
    _, deltas = _stream_chat(_script("Stop here."), think=False, stop=["S"])
    assert deltas == [ROLE, {"finish": "stop"}], deltas
    _, deltas = _stream_chat(_script("Stop here."), stop=["S"])
    assert deltas == [ROLE, {"content": "<think>\n"}, {"finish": "stop"}], deltas

    def broken(*a, **k):
        raise RuntimeError("the prefill fell over")
        yield  # noqa: unreachable -- a generator, like the engine's

    import contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _, deltas = _stream_chat(broken, think=False)
    assert deltas[0] == ROLE and deltas[-1] == {"finish": "error"} and len(deltas) == 2, deltas


def test_the_role_chunk_precedes_a_separate_usage_chunk():
    """With the `include_usage`: the usage chunk (`choices: []`) still comes after the
    finish chunk, exactly once, and the role chunk is still first."""
    serve()
    app.STATE["tok"] = ThinkTok()
    real = app.generate_stream
    app.generate_stream = _script("Plain.")
    try:
        head, raw = Req("/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "q"}], "stream": True,
                         "stream_options": {"include_usage": True},
                         "chat_template_kwargs": {"enable_thinking": False}}).response()
    finally:
        app.generate_stream = real
    chunks = [json.loads(ln[6:]) for ln in raw.splitlines() if ln.startswith("data: {")]
    assert chunks[0]["choices"][0]["delta"] == ROLE, chunks[0]
    usage = [c for c in chunks if c.get("usage")]
    assert len(usage) == 1 and usage[0]["choices"] == [] and usage[0] is chunks[-1], usage


def test_the_first_content_chunk_carries_the_tag_and_the_first_text():
    """The tag is not dropped (Open WebUI folds on it and the model never writes it): it
    goes out WITH the first generated text, in one chunk, and is still the first thing in
    `content`. The whole of `content` is what it was before the fix."""
    _, deltas = _stream_chat(_script("Hmm.</think>\n\nYes."))
    first = next(d for d in deltas if _text(d))
    assert first == {"content": "<think>\nH"}, first
    content = "".join(d.get("content") or "" for d in deltas)
    assert content == "<think>\nHmm.</think>\n\nYes.", content
    # `both`: the reasoning delta is the first text on the wire, and the content copy of it
    # carries the tag
    _, deltas = _stream_chat(_script("Hmm.</think>\n\nYes."), fmt="both")
    texts = [d for d in deltas if _text(d)]
    assert texts[0] == {"reasoning_content": "H"}, texts[:2]
    assert texts[1] == {"content": "<think>\nH"}, texts[:2]
    # `reasoning_content` never had a tag to send
    _, deltas = _stream_chat(_script("Hmm.</think>\n\nYes."), fmt="reasoning_content")
    assert "<think>" not in json.dumps(deltas), deltas
    assert "".join(d.get("reasoning_content") or "" for d in deltas) == "Hmm."
    assert "".join(d.get("content") or "" for d in deltas) == "Yes."


def test_thinking_off_streams_exactly_what_it_did():
    """The row's path (thinking off) had no tag to hold, and it must not change by a byte:
    the role chunk, one chunk per generated character, the finish chunk. (The role chunk now
    waits for the first text,; the stream's chunks are the same.)"""
    for fmt in ("tags", "reasoning_content", "both"):
        before, deltas = _stream_chat(_script("Plain."), think=False, fmt=fmt)
        assert before == [], (fmt, before)
        assert deltas == ([{"role": "assistant", "content": ""}]
                          + [{"content": c} for c in "Plain."]
                          + [{"finish": "length"}]), (fmt, deltas)


def test_a_stream_with_no_generated_text_still_opens_the_block():
    """The tag waits for the first text, and some streams have none: a stop string that
    matches at the first character, or a generation that fails before its first token. The tag
    then goes out at the end, before the finish chunk, so `content` is still `<think>\\n` exactly
    as the non-streamed answer's is."""
    _, deltas = _stream_chat(_script("Stop here."), stop=["S"])
    assert "".join(_text(d) for d in deltas) == "<think>\n", deltas
    assert deltas[-1] == {"finish": "stop"}, deltas

    def broken(*a, **k):
        raise RuntimeError("the prefill fell over")
        yield  # noqa: unreachable -- a generator, like the engine's

    import contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        before, deltas = _stream_chat(broken)
    assert "".join(_text(d) for d in deltas) == "<think>\n", deltas
    assert deltas[-1] == {"finish": "error"}, deltas


class BudgetTok(FakeTok):
    """`FakeTok` with a template, so the handler can drive the random model through a forced
    close: the prompt ends on the `<think>` token, and the tags decode as tags."""

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        return {"input_ids": torch.tensor([[5, 6, 7, 8, OPEN]])}

    def decode(self, seq, skip_special_tokens=True):
        return "".join({OPEN: "<think>", CLOSE: "</think>"}.get(int(t), chr(ord("a") + int(t) % 26))
                       for t in seq)


def test_the_forced_close_and_the_budget_keep_the_tag_first():
    """On the real loop: a reasoning budget of three tokens, closed by the engine. The tag
    still waits for the first token and still leads `content`, and the forced phrase and the answer
    follow it."""
    for fmt in ("tags", "both"):
        serve()
        app.STATE["tok"] = BudgetTok()
        app.STATE["reasoning_format"] = fmt
        req = Req("/v1/chat/completions", {"messages": [{"role": "user", "content": "q"}],
                                           "stream": True, "max_tokens": 20,
                                           "max_reasoning_tokens": 3,
                                           "reasoning_format": fmt})            # SRV-617
        held, real = [], app.generate_stream

        def recorded(*a, **k):
            held.append(req.wfile.getvalue().partition(b"\r\n\r\n")[2].decode())
            yield from real(*a, **k)

        app.generate_stream = recorded
        import contextlib
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                head, raw = req.response()
        finally:
            app.generate_stream = real
        before, deltas = _deltas(held[0]), _deltas(raw)
        assert before == [], (fmt, before)
        assert deltas[0] == {"role": "assistant", "content": ""}, (fmt, deltas[:2])
        content = "".join(d.get("content") or "" for d in deltas)
        first = next(d["content"] for d in deltas if d.get("content"))
        assert first.startswith("<think>\n") and len(first) > len("<think>\n"), (fmt, first)
        assert content.startswith("<think>\n"), (fmt, content)
        if fmt == "tags":
            assert "cd</think>" in content, content      # the forced phrase, CLOSING, decoded


def test_a_streamed_tool_call_after_the_reasoning_is_unchanged():
    """With the tool-call buffer on the path: the tag no longer goes through it on its own,
    and the call in the answer is still exactly one call, with the reasoning still text."""
    call = ("<tool_call>\n<function=delete_file>\n<parameter=path>\n/home/user/a.txt\n"
            "</parameter>\n</function>\n</tool_call>")
    before, deltas = _stream_chat(_script(THOUGHT + call),
                                  tools=[{"type": "function", "function": {"name": "delete_file"}}])
    assert before == [], before
    calls = [c for d in deltas for c in d.get("tool_calls") or []]
    assert [c["function"]["name"] for c in calls if c.get("function", {}).get("name")] == \
        ["delete_file"], calls
    content = "".join(d.get("content") or "" for d in deltas)
    assert content == "<think>\n" + THOUGHT, content
    assert next(d for d in deltas if _text(d)) == {"content": "<think>\nI"}


def test_the_non_streamed_answer_is_unchanged():
    """Touches the stream only: the JSON answer puts the tag back as it always did."""
    serve()
    app.STATE["tok"] = ThinkTok()
    real = app.generate_stream
    try:
        for think, want in ((True, "<think>\nHmm.</think>\n\nYes."),
                            (False, "Hmm.</think>\n\nYes.")):
            app.generate_stream = _script("Hmm.</think>\n\nYes.")
            head, body = Req("/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "q"}],
                              "reasoning_format": "tags",                       # SRV-617
                              "chat_template_kwargs": {"enable_thinking": think}}).response()
            assert json.loads(body)["choices"][0]["message"]["content"] == want, body
    finally:
        app.generate_stream = real


# ------------------------------------------------------------------

def _content(deltas) -> str:
    return "".join(d.get("content") or "" for d in deltas)


def _both_ways(text, think, fmt="tags", **extra):
    """The same scripted generation as one streamed and one non-streamed chat request:
    `(content, reasoning_content, finish_reason, tool names)` for each."""
    _, deltas = _stream_chat(_script(text), think=think, fmt=fmt, **extra)
    streamed = (_content(deltas),
                "".join(d.get("reasoning_content") or "" for d in deltas) or None,
                next(d["finish"] for d in deltas if "finish" in d),
                [c["function"]["name"] for d in deltas for c in d.get("tool_calls") or []
                 if c.get("function", {}).get("name")])
    serve()
    app.STATE["tok"] = ThinkTok()
    app.STATE["reasoning_format"] = fmt
    real = app.generate_stream
    app.generate_stream = _script(text)
    try:
        head, body = Req("/v1/chat/completions",
                         dict({"messages": [{"role": "user", "content": "q"}],
                               "reasoning_format": fmt,                         # SRV-617
                               "chat_template_kwargs": {"enable_thinking": think}},
                              **extra)).response()
    finally:
        app.generate_stream = real
    choice = json.loads(body)["choices"][0]
    msg = choice["message"]
    # an empty reasoning field is "" in the JSON answer and no delta at all in a stream
    whole = (msg["content"], msg.get("reasoning_content") or None, choice["finish_reason"],
             [c["function"]["name"] for c in msg.get("tool_calls") or []])
    return streamed, whole


def test_a_stop_string_split_across_pieces_leaks_nothing():
    """Only the piece that completed the match was cut; the pieces before it had gone out.
    `stop: ["Stop"]` on "Stop here." streamed S, t, o and then finished with `stop`, where the
    non-streamed answer is empty. One token per character here, so every prefix of the stop string
    arrives as a piece of its own."""
    _, deltas = _stream_chat(_script("Stop here."), think=False, stop=["Stop"])
    assert _content(deltas) == "" and deltas[-1] == {"finish": "stop"}, deltas
    _, deltas = _stream_chat(_script("Stop here."), stop=["Stop"])
    assert _content(deltas) == "<think>\n" and deltas[-1] == {"finish": "stop"}, deltas
    _, deltas = _stream_chat(_script("Go on. Stop here."), think=False, stop="Stop")
    assert _content(deltas) == "Go on. " and deltas[-1] == {"finish": "stop"}, deltas


def test_a_held_prefix_that_does_not_complete_is_released():
    """What is held back because it could still become a stop string goes out the moment
    it cannot, and at the end of the generation if it never decided: nothing is lost."""
    steps = []
    _, deltas = _stream_chat(_script("Storm"), think=False, steps=steps, stop=["Stop"])
    assert _content(deltas) == "Storm" and deltas[-1] == {"finish": "length"}, deltas
    assert _content(steps[3]) == "", steps[3]          # "Sto" is held
    assert _content(steps[4]) == "Stor", steps[4]      # "r" decides it
    _, deltas = _stream_chat(_script("Hello Sto"), think=False, stop=["Stop"])
    assert _content(deltas) == "Hello Sto" and deltas[-1] == {"finish": "length"}, deltas


CALL = ("<tool_call>\n<function=delete_file>\n<parameter=path>\n/home/user/a.txt\n"
        "</parameter>\n</function>\n</tool_call>")
STOP_CASES = [
    ("Stop here.", ["Stop"]),
    ("Storm and then Stop.", ["Stop"]),
    ("Hello Sto", ["Stop"]),
    ("Hmm.</think>\n\nYes. Stop now.", ["Stop"]),
    ("Hmm, Stop.</think>\n\nYes.", ["Stop"]),
    ("Hmm.</think>\n\nYes.", ["</think>"]),
    ("Hmm.</think>\n\nYes.", ["\n\nY"]),
    # the earliest match wins, as in the non-streamed answer, even when a shorter stop string
    # completes first inside a longer one that is still arriving
    ("abcdef", ["abcd", "bc"]),
    ("abcxef", ["abcd", "bc"]),
    ("xaab", ["ab", "aab"]),
    ("no match at all", ["zz", "q"]),
    ("Hmm.</think>\n\n" + CALL + " Stop", ["Stop"]),
    ("Hmm.</think>\n\nCalling. " + CALL + " Stop", ["Stop"]),
]


def test_a_streamed_answer_with_stop_is_the_non_streamed_one():
    """For every case, thinking on and off, both split formats' fields and the tool calls:
    the concatenated stream is the JSON answer."""
    tools = [{"type": "function", "function": {"name": "delete_file"}}]
    for text, stop in STOP_CASES:
        for think in (True, False):
            for fmt in ("tags", "reasoning_content", "both"):
                streamed, whole = _both_ways(text, think, fmt, stop=stop, tools=tools)
                assert streamed == whole, (text, stop, think, fmt, streamed, whole)


def test_the_completions_stream_holds_a_stop_string_back_too():
    """On `/v1/completions`, which shares the loop."""
    for text, stop, want in (("Stop here.", ["Stop"], ""), ("Storm", ["Stop"], "Storm"),
                             ("a Sto b Stop c", ["Stop"], "a Sto b ")):
        serve()
        app.STATE["tok"] = CharTok()
        real = app.generate_stream
        try:
            app.generate_stream = _script(text)
            _, raw = Req("/v1/completions", {"prompt": "q", "stop": stop, "stream": True,
                                             "max_tokens": 64}).response()
            app.generate_stream = _script(text)
            _, body = Req("/v1/completions", {"prompt": "q", "stop": stop,
                                              "max_tokens": 64}).response()
        finally:
            app.generate_stream = real
        streamed = "".join(c["text"] for ln in raw.splitlines() if ln.startswith("data: {")
                           for c in json.loads(ln[6:])["choices"])
        assert streamed == want == json.loads(body)["choices"][0]["text"], (text, streamed, body)


# ------------------------------------------------------------------

def test_both_closes_the_block_it_opens():
    """`both` put `<think>\n` + the reasoning into `content` and never the `</think>`:
    streamed content was `<think>\nHmm.Yes.`, and a client that folds on tags (Open WebUI) folded
    the answer into the reasoning. Content is now the `tags` text, reasoning_content the reasoning
    alone, and both are the non-streamed answer's."""
    _, deltas = _stream_chat(_script("Hmm.</think>\n\nYes."), fmt="both")
    content = _content(deltas)
    assert content == "<think>\nHmm.</think>\n\nYes.", content
    assert content.count("<think>") == 1 and content.count("</think>") == 1, content
    assert "".join(d.get("reasoning_content") or "" for d in deltas) == "Hmm."
    for text in ("Hmm.</think>\n\nYes.", "Hmm.", "Hmm.</think>Yes.</think>", THOUGHT + "Ok."):
        streamed, whole = _both_ways(text, True, "both")
        assert streamed == whole, (text, streamed, whole)


def test_both_does_not_read_a_call_in_the_reasoning_as_a_call():
    """Found it: `both`'s content copy of the reasoning went through the tool-call buffer, so
    a call the model only deliberated about was sent as a real `tool_calls` entry (had
    fixed it for `tags` only). The copy is the `tags` block now, and is not read for calls."""
    tools = [{"type": "function", "function": {"name": "delete_file"}}]
    streamed, whole = _both_ways(THOUGHT + "Should I delete a.txt?", True, "both", tools=tools)
    assert streamed == whole and streamed[3] == [] and streamed[2] == "length", (streamed, whole)
    streamed, whole = _both_ways(THOUGHT + CALL, True, "both", tools=tools)
    assert streamed == whole and streamed[3] == ["delete_file"], (streamed, whole)


# ------------------------------------------------------------------

_DRAIN_CHILD = r"""
import os, sys, time
sys.path.insert(0, os.path.join({root!r}, "tests"))
import test_app_loop as t
from http.server import ThreadingHTTPServer
from server import app

t.serve()


def slow(*a, **k):
    for tok in (5, 6, 7, 8, 9, 10, 11, 12):
        time.sleep(0.3)
        yield tok


app.generate_stream = slow
httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
print(httpd.server_address[1], flush=True)
app.serve_until_drained(httpd)
"""


def test_a_graceful_stop_lets_the_generation_in_flight_finish():
    """SIGTERM set `draining` and shut the listener down -- and then `main()` returned.
    ThreadingHTTPServer's handler threads are DAEMONS, so the interpreter exited under the stream
    it had promised to finish: the operations guide's "it drains first", stop.sh's grace period and every
    hold's stop cut the generation in flight instead. A streamed request that is running when the
    signal lands must end with its finish chunk and [DONE], and the process must exit 0."""
    import signal
    import socket
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    child = subprocess.Popen([sys.executable, "-c", _DRAIN_CHILD.format(root=root)],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                             env=env)
    try:
        port = int(child.stdout.readline())
        body = json.dumps({"prompt": "hello", "max_tokens": 8, "stream": True}).encode()
        s = socket.create_connection(("127.0.0.1", port), timeout=30)
        s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        got = s.recv(65536)
        while b"data: {" not in got:
            got += s.recv(65536)
        child.send_signal(signal.SIGTERM)             # mid-stream: the first token is out
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            got += chunk
        assert b"data: [DONE]" in got, f"the stream was cut: {got[-300:]!r}"
        assert got.count(b'"text"') >= 8, "every token reached the client"
        assert child.wait(timeout=30) == 0
    finally:
        if child.poll() is None:
            child.kill()


def test_the_engine_server_takes_64_connects_at_once_and_answers_a_kept_connection_at_once():
    """NP1's night pass (2026-10-04, open issue 1): the engine's server kept Python's listen
    backlog of 5 and Nagle on. The proxy had the same defaults: 10 of 64 clients that connected in
    the same moment got a reset, and a keep-alive connection's answer waited for the
    reader's delayed ACK (40 ms on Linux). The engine's server now has a backlog of 128 and no
    Nagle, as the proxy and uvicorn's servers."""
    import socket
    import threading
    import time
    serve()

    httpd = app.EngineServer(("127.0.0.1", 0), app.Handler)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    try:
        assert app.EngineServer.request_queue_size == 128
        assert app.Handler.disable_nagle_algorithm is True
        # a keep-alive connection's later answers (a stream closes its connection; a JSON answer
        # keeps it, and its headers and body are two writes): no 40 ms delayed-ACK stall
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        ms = []
        for _ in range(8):
            t0 = time.perf_counter()
            conn.request("GET", "/health")
            r = conn.getresponse()
            r.read()
            assert r.status == 200 and not r.will_close
            ms.append((time.perf_counter() - t0) * 1e3)
        conn.close()
        assert sorted(ms[1:])[3] < 20.0, ms              # the median of the later seven
        # 64 clients that connect in the same moment
        gate = threading.Barrier(64)
        ok = []

        def one():
            gate.wait()
            try:
                c = socket.create_connection(("127.0.0.1", port), timeout=10)
                c.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                data = b""
                while True:
                    chunk = c.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                c.close()
                ok.append(data.startswith(b"HTTP/1.1 200"))
            except OSError:
                ok.append(False)
        ts = [threading.Thread(target=one) for _ in range(64)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        assert ok.count(True) == 64, ok.count(True)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ------------------------------------------------------------------ /

def test_the_req_line_carries_both_factors_of_the_speed():
    """Tok/s = committed tokens a block / block time, and the row reported only the quotient. The
    served loop counts every forward it pays and each block's draft depth and accepted length; the
    `[req]` line carries them and `tools/rowlog.py` reads them back."""
    import contextlib
    import time
    from tools.rowlog import parse_requests
    for tree in (False, True):
        serve(FixedDrafter(97, 5, tree_mode=tree), tree=tree)
        out = list(app.generate_stream(torch.tensor([5, 6, 7, 8]), 40, set()))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            app._log_request("c", 4, len(out), "length", time.perf_counter(), stream=False)
        r = parse_requests(buf.getvalue())[0]
        assert r["committed"] == len(out) - 1 == 39, (tree, r)
        # every block had a five-token draft (a chain, or a chain-shaped tree of depth five)
        assert set(r["accept"]) == {5}, (tree, r["accept"])
        assert sum(r["accept"][5].values()) == r["blocks"], (tree, r)
        # a block yields its accepted draft and the target's own token; the last one may be cut
        # by the token budget, never extended
        yielded = sum((a + 1) * n for a, n in r["accept"][5].items())
        assert yielded >= r["committed"] and yielded - r["committed"] <= 5, (tree, r)
        assert r["decode_ms"] >= 0.0
        # the stats are the request's: a request logged without a generation carries none
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            app._log_request("d", 4, 0, "error", time.perf_counter(), stream=False)
        assert parse_requests(buf.getvalue())[0]["blocks"] is None


def test_a_sampled_request_says_its_temperature_and_a_greedy_line_is_unchanged():
    """Step 0: how much served traffic samples decides whether the sampled tree is worth
    building, and nothing recorded it. A sampled request's `[req]` line carries `temp=`; a greedy
    one carries nothing new, so every parser of the old line reads it as before."""
    import contextlib
    import time
    from tools.rowlog import parse_requests
    lines = {}
    for t in (0.0, 0.7):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            app._log_request("c", 4, 9, "stop", time.perf_counter(), stream=True, temp=t)
        lines[t] = buf.getvalue()
    assert " temp=" not in lines[0.0], lines[0.0]
    assert lines[0.7].rstrip().endswith(" tok/s temp=0.7"), lines[0.7]
    assert [r["temp"] for r in parse_requests(lines[0.0] + lines[0.7])] == [0.0, 0.7]


def test_blocking_sync_sets_the_primary_context_flag_or_refuses_to_start():
    """Hypothesis 3: QWEN38_BLOCKING_SYNC=1 asks the driver for CU_CTX_SCHED_BLOCKING_SYNC on
    the primary context before torch creates it, and a start where the flag did not take stops
    instead of measuring the wrong thing. A fake driver stands in for libcuda."""
    import ctypes

    class FakeCu:
        def __init__(self, keeps):
            self.keeps, self.calls, self.flags = keeps, [], 0

        def cuInit(self, f):
            self.calls.append(("init", f)); return 0

        def cuDeviceGet(self, ref, i):
            self.calls.append(("get", i)); return 0

        def cuDevicePrimaryCtxSetFlags(self, dev, flags):
            self.calls.append(("set", flags))
            if self.keeps:
                self.flags = flags
            return 0

        def cuDevicePrimaryCtxGetState(self, dev, flags, active):
            flags._obj.value = self.flags; return 0

    real = ctypes.CDLL
    try:
        cu = FakeCu(keeps=True)
        ctypes.CDLL = lambda name: cu
        app._blocking_sync()
        assert ("set", 0x04) in cu.calls and cu.calls[0] == ("init", 0), cu.calls
        cu = FakeCu(keeps=False)
        ctypes.CDLL = lambda name: cu
        try:
            app._blocking_sync()
            raise AssertionError("a flag that did not take must stop the start")
        except SystemExit:
            pass
    finally:
        ctypes.CDLL = real


def test_a_declined_step_is_a_block_without_a_draft():
    serve(None)
    out = list(app.generate_stream(torch.tensor([5, 6, 7, 8]), 9, set()))
    import contextlib
    import time
    from tools.rowlog import parse_requests
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        app._log_request("c", 4, len(out), "length", time.perf_counter(), stream=False)
    r = parse_requests(buf.getvalue())[0]
    assert r["blocks"] == 8 and r["committed"] == 8 and r["accept"] == {}, r

if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
