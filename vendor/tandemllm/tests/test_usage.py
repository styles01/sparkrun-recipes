"""Per-response usage and timings, on a CPU, through the real handler.

The Gherkin of is this file's matrix. What it pins down:

  * exactly ONE chunk of a stream carries `usage` / `timings` / `metrics`: the finish chunk when
    the client sent no `stream_options` (Open WebUI's base models), the separate `choices: []`
    chunk when it asked with `include_usage: true`, none when it said false or the server runs
    `--usage-default off`;
  * replayed through a port of Open WebUI 0.11.3's own merge, the counts are not doubled;
  * the counts are exact: committed ids (EOS included), blocks and draft sums from the loop's own
    histogram, reasoning through the closing token, cached/forwarded from the prefill, a replay
    from the response cache as a replay;
  * the timings are consistent with each other: ttft = queue + prompt, and the decode rate is the
    atlas row's `(completion - 1) / predicted_ms`.

Run: python tests/test_usage.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# first: it sets the CPU reference paths' environment before the engine modules read it
from test_app_loop import Req, serve  # noqa: E402

import torch  # noqa: E402

from engine import cache  # noqa: E402
from server import app, usage  # noqa: E402

FIELDS = ("usage", "timings", "metrics")


# ------------------------------------------------------------------ tokenizers

class UTok:
    """One token per character, and a template with both of the real one's endings. `</think>`
    is NOT a special token here, so the block closes on the literal characters."""

    unk_token_id = -1
    eos_token_id = None
    SPECIAL: dict = {}

    def convert_tokens_to_ids(self, text):
        return self.SPECIAL.get(text, -1)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [self.SPECIAL[text]] if text in self.SPECIAL else [ord(c) for c in text]
        if return_tensors == "pt":
            ids = torch.tensor([ids])
        return type("R", (), {"input_ids": ids})()

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=True,
                            **kw):
        tail = "\n<think>\n" if enable_thinking else "\n<think>\n\n</think>\n\n"
        text = "".join(m["content"] for m in messages) + tail
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}

    def decode(self, seq, skip_special_tokens=True):
        inv = {v: k for k, v in self.SPECIAL.items()}
        return "".join(inv.get(int(t), chr(int(t))) for t in seq)


class STok(UTok):
    """`UTok` where `</think>` IS one special token, as in the real Qwen vocabulary."""

    SPECIAL = {"</think>": 1000}


def _ids(text: str, tok=None) -> list[int]:
    return list((tok or UTok())(text).input_ids)


def _script(ids):
    return lambda *a, **k: iter(list(ids))


def _run(body, *, tok=None, engine=None, path="/v1/chat/completions", state=None):
    """One request through the handler; `engine` replaces `generate_stream`."""
    serve()
    app.STATE["tok"] = tok or UTok()
    app.STATE.update(state or {})
    real = app.generate_stream
    if engine is not None:
        app.generate_stream = engine
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            head, raw = Req(path, body).response()
    finally:
        app.generate_stream = real
    return head, raw, out.getvalue()


def _chunks(raw: str) -> list[dict]:
    return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]


def _carriers(chunks):
    return [i for i, c in enumerate(chunks) if any(k in c for k in FIELDS)]


def _chat(text="q", stream=True, **extra):
    return dict({"messages": [{"role": "user", "content": text}], "stream": stream,
                 "chat_template_kwargs": {"enable_thinking": False}}, **extra)


# ------------------------------------------------------------------ placement

def test_the_finish_chunk_carries_it_when_the_client_did_not_ask():
    """Open WebUI on a base model: no stream_options. The finish chunk -- a non-empty `choices` --
    carries all three, and no other chunk carries any of them."""
    head, raw, _ = _run(_chat(), engine=_script(_ids("Plain answer.")))
    chunks = _chunks(raw)
    where = _carriers(chunks)
    assert len(where) == 1, f"usage on {len(where)} chunks: {where}"
    fin = chunks[where[0]]
    assert fin["choices"] and fin["choices"][0]["finish_reason"] == "length", fin
    assert all(k in fin for k in FIELDS), fin.keys()
    assert fin["usage"]["completion_tokens"] == len("Plain answer.")
    assert raw.rstrip().endswith("data: [DONE]")


def test_include_usage_true_is_the_separate_chunk():
    head, raw, _ = _run(_chat(stream_options={"include_usage": True}),
                        engine=_script(_ids("Plain answer.")))
    chunks = _chunks(raw)
    where = _carriers(chunks)
    assert len(where) == 1 and chunks[where[0]]["choices"] == [], where
    assert where[0] == len(chunks) - 1, "the usage chunk is the last before [DONE]"
    fin = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert len(fin) == 1 and not any(k in fin[0] for k in FIELDS), fin


def test_include_usage_false_and_the_default_off_send_nothing():
    head, raw, _ = _run(_chat(stream_options={"include_usage": False}),
                        engine=_script(_ids("Plain.")))
    assert _carriers(_chunks(raw)) == [], raw[-400:]
    head, raw, _ = _run(_chat(), engine=_script(_ids("Plain.")),
                        state={"usage_default": False})
    assert _carriers(_chunks(raw)) == [], "--usage-default off: as before "
    # the flag leaves an explicit request alone
    head, raw, _ = _run(_chat(stream_options={"include_usage": True}),
                        engine=_script(_ids("Plain.")), state={"usage_default": False})
    assert len(_carriers(_chunks(raw))) == 1


def test_placement_rule():
    P = usage.placement
    assert P({}, False, True) == "body" and P({}, False, False) == "body"
    assert P({}, True, True) == "finish" and P({}, True, False) == "none"
    assert P({"stream_options": {}}, True, True) == "finish"
    assert P({"stream_options": {"include_usage": None}}, True, True) == "finish"
    assert P({"stream_options": {"include_usage": True}}, True, False) == "separate"
    assert P({"stream_options": {"include_usage": False}}, True, True) == "none"


def test_the_text_completion_endpoint_gets_the_same_fields():
    head, raw, _ = _run({"prompt": "hello", "stream": True, "max_tokens": 5},
                        engine=_script(_ids("abcde")), path="/v1/completions")
    chunks = _chunks(raw)
    where = _carriers(chunks)
    assert len(where) == 1 and chunks[where[0]]["object"] == "text_completion", where
    assert chunks[where[0]]["usage"]["completion_tokens"] == 5
    head, raw, _ = _run({"prompt": "hello", "max_tokens": 5}, engine=_script(_ids("abcde")),
                        path="/v1/completions")
    body = json.loads(raw)
    assert all(k in body for k in FIELDS) and body["usage"]["prompt_tokens"] == 5, body


# ------------------------------------------------------------------ Open WebUI's merge

# Open WebUI 0.11.3's own usage merge, ported in tools/usage_check.py (file:line in its header):
# the live check and these tests read a stream the same way.
from tools.usage_check import consistent, owui_replay as _owui_replay  # noqa: E402


def test_open_webui_does_not_double_the_counts():
    for extra in ({}, {"stream_options": {"include_usage": True}}):
        head, raw, _ = _run(_chat(**extra), engine=_script(_ids("Twelve chars")))
        chunks = _chunks(raw)
        u = _owui_replay(chunks)
        mine = chunks[_carriers(chunks)[0]]["usage"]
        assert u["input_tokens"] == mine["prompt_tokens"], (extra, u)
        assert u["output_tokens"] == mine["completion_tokens"] == 12, (extra, u)
        assert u["predicted_per_second"] == chunks[_carriers(chunks)[0]]["timings"][
            "predicted_per_second"], "the tooltip shows the engine's speed"
    # and the port is faithful enough to see the failure it guards against: usage twice doubles
    doubled = _owui_replay(chunks + [chunks[_carriers(chunks)[0]]])
    assert doubled["output_tokens"] == 24


# ------------------------------------------------------------------ exact counts

def _engine_with_blocks(n_first_plus_blocks, accept, prefill=None):
    """A fake engine: yields ids 97.. and publishes the loop's own counters as the real one does."""
    def gen(*a, **k):
        bs = app.STATE["blocks"] = app.BlockStats()
        if prefill is not None:
            app.STATE["last_prefill"] = dict(prefill)
        yield 97
        bs.first()
        for depth, acc, n in accept:
            for _ in range(n):
                bs.block(depth, acc)
        for i in range(n_first_plus_blocks - 1):
            yield 98 + (i % 20)
    return gen


def test_exact_counts_under_speculation():
    """40 tokens: the prefill's one and 39 committed in 9 blocks of depth 15 (3 blocks kept 4
    drafted tokens, 6 kept 3: 3*5 + 6*4 = 39)."""
    head, raw, log = _run(_chat(max_tokens=40),
                          engine=_engine_with_blocks(40, [(15, 4, 3), (15, 3, 6)]))
    chunks = _chunks(raw)
    fin = chunks[_carriers(chunks)[0]]
    t, u = fin["timings"], fin["usage"]
    assert u["completion_tokens"] == 40 and t["blocks"] == 9, (u, t)
    assert t["tokens_per_block"] == round(39 / 9, 2), t
    assert t["draft_n"] == 9 * 15 and t["draft_n_accepted"] == 3 * 4 + 6 * 3, t
    assert fin["metrics"]["speculative_decoding"] == {
        "mean_acceptance_length": round(39 / 9, 2), "draft_acceptance_rate": round(30 / 135, 4)}
    # the [req] line carries the same histogram, so both reports agree
    from tools.rowlog import parse_requests
    r = parse_requests(log)[0]
    assert r["blocks"] == 9 and r["committed"] == 39, r


def test_the_real_loop_counts_match_its_own_histogram():
    """The served loop, the random 4-layer model and a fixed-width drafter: the timings' draft
    sums are the sums over the `[req]` line's first-miss histogram."""
    from test_app_loop import BudgetTok
    from test_window_edge import FixedDrafter
    from tools.rowlog import parse_requests
    for tree in (False, True):
        serve(FixedDrafter(97, 5, tree_mode=tree), tree=tree)
        app.STATE["tok"] = BudgetTok()
        app.STATE["pen_spec"] = app.PenaltySpec()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            head, raw = Req("/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "q"}], "max_tokens": 30,
                             "chat_template_kwargs": {"enable_thinking": False}}).response()
        body = json.loads(raw)
        t = body["timings"]
        r = parse_requests(out.getvalue())[0]
        assert body["usage"]["completion_tokens"] == 30 and t["predicted_n"] == 30
        assert t["blocks"] == r["blocks"], (t, r)
        assert t["draft_n"] == sum(d * n for d, h in r["accept"].items() for n in h.values())
        assert t["draft_n_accepted"] == sum(a * n for h in r["accept"].values()
                                            for a, n in h.items())


def test_reasoning_tokens_through_the_special_closer():
    tok = STok()
    gen = [ord(c) for c in "eleven toks"] + [1000] + [ord("x")] * 18
    assert len(gen) == 30 and gen.index(1000) == 11
    head, raw, _ = _run({"messages": [{"role": "user", "content": "q"}], "stream": True,
                         "max_tokens": 30}, tok=tok, engine=_script(gen))
    fin = _chunks(raw)[_carriers(_chunks(raw))[0]]
    assert fin["usage"]["completion_tokens_details"]["reasoning_tokens"] == 12, fin["usage"]
    assert fin["usage"]["completion_tokens"] == 30 and fin["timings"]["reasoning_n"] == 12


def test_reasoning_through_the_literal_closer_and_thinking_off():
    gen = _ids("Hmm, yes.</think>\n\nNo.")
    head, raw, _ = _run({"messages": [{"role": "user", "content": "q"}], "stream": True},
                        engine=_script(gen))
    fin = _chunks(raw)[_carriers(_chunks(raw))[0]]
    assert fin["usage"]["completion_tokens_details"]["reasoning_tokens"] == \
        len("Hmm, yes.</think>"), fin["usage"]
    head, raw, _ = _run(_chat(), engine=_script(gen))           # thinking off
    fin = _chunks(raw)[_carriers(_chunks(raw))[0]]
    assert fin["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0


def test_a_block_that_never_closes_is_all_reasoning():
    head, raw, _ = _run({"messages": [{"role": "user", "content": "q"}], "stream": True,
                         "max_tokens": 9}, engine=_script(_ids("still thinking")[:9]))
    fin = _chunks(raw)[_carriers(_chunks(raw))[0]]
    assert fin["choices"][0]["finish_reason"] == "length"
    assert fin["usage"]["completion_tokens_details"]["reasoning_tokens"] == \
        fin["usage"]["completion_tokens"] == 9


def test_a_tool_call_is_counted_and_timed():
    call = ("<tool_call>\n<function=delete_file>\n<parameter=path>\n/tmp/a.txt\n"
            "</parameter>\n</function>\n</tool_call>")
    gen = _ids("Ok.</think>\n\n" + call) + [3]              # 3 is the EOS below
    tools = [{"type": "function", "function": {"name": "delete_file"}}]
    for stream in (True, False):
        head, raw, _ = _run({"messages": [{"role": "user", "content": "q"}], "stream": stream,
                             "tools": tools}, engine=_script(gen), state={"cfg_eos": 3})
        if stream:
            chunks = _chunks(raw)
            fin = chunks[_carriers(chunks)[0]]
            finish = fin["choices"][0]["finish_reason"]
        else:
            fin = json.loads(raw)
            finish = fin["choices"][0]["finish_reason"]
        assert finish == "tool_calls", (stream, finish)
        assert fin["usage"]["completion_tokens"] == len(gen), (stream, fin["usage"])
        assert fin["usage"]["completion_tokens_details"]["reasoning_tokens"] == len("Ok.</think>")
        assert fin["timings"]["predicted_n"] == len(gen)


def test_a_stop_string_counts_what_was_generated_to_the_cut():
    gen = _ids("abcSTOPdefghij")
    head, raw, _ = _run(_chat(stop=["STOP"]), engine=_script(gen))
    fin = _chunks(raw)[_carriers(_chunks(raw))[0]]
    assert fin["choices"][0]["finish_reason"] == "stop"
    # the stream breaks when the stop string is complete: 7 ids were generated to that point
    assert fin["usage"]["completion_tokens"] == len("abcSTOP"), fin["usage"]


def test_hidden_tokens_are_reported_apart_from_the_prompt():
    """SRV-165 (UT1, 2026-10-06): a 'hi' under the conductor reported 238 prompt tokens (raw: 11):
    the router's note was counted. The record keeps the client's count in `prompt_tokens`, the
    note's in `prompt_tokens_details.hidden_tokens`, and `cached_tokens` within the client's
    count (the cache holds the replayed notes too); without hidden tokens nothing changes."""
    r = usage.RequestRecord("chatcmpl-x", "chat", False)
    r.prompt_tokens, r.hidden_prompt, r.cached_tokens = 11, 227, 230
    u = r.usage()
    assert u["prompt_tokens"] == 11 and u["total_tokens"] == 11, u
    assert u["prompt_tokens_details"] == {"cached_tokens": 11, "hidden_tokens": 227}, u
    assert r.row()["cached_tokens"] == 11 and r.row()["prompt_tokens"] == 11
    plain = usage.RequestRecord("chatcmpl-y", "chat", False)
    plain.prompt_tokens, plain.cached_tokens = 11, 8
    assert plain.usage()["prompt_tokens_details"] == {"cached_tokens": 8}, plain.usage()
    return "11 prompt + 227 hidden, cached 11; a raw request's usage keeps its old shape"


# ------------------------------------------------------------------ caches

def test_a_prefix_hit_reports_cached_and_forwarded():
    """The Gherkin's numbers, as the prefill publishes them."""
    head, raw, _ = _run(_chat(), engine=_engine_with_blocks(
        5, [(15, 3, 1)], prefill={"reused": 1536, "forwarded": 188, "ms": 1.0}))
    t = _chunks(raw)[_carriers(_chunks(raw))[0]]
    # the fake's prefill reuses more rows than its 21-token prompt holds: `usage` counts at most
    # the client's prompt (usage.client_cached_of, 2026-10-09), `timings` keeps the engine's own
    u = t["usage"]
    assert u["prompt_tokens_details"]["cached_tokens"] == min(1536, u["prompt_tokens"]), u
    assert t["timings"]["cache_n"] == 1536 and t["timings"]["prompt_n"] == 188
    assert t["timings"]["cache_source"] == "prefix"


def test_the_real_prefix_cache():
    """The real prefill and state store on the random model: a second prompt sharing 32 tokens
    with the first resumes from the 32-token checkpoint."""
    serve(None, max_len=256)
    from test_app_loop import FakeTok
    app.STATE.update(tok=FakeTok(), state_store=cache.StateStore(1 << 30, chunk=16),
                     prefix_cache=True, prefix_chunk=16)
    shared = "abcdefghijklmnopqrstuvwxyzABCDEF"             # 32 tokens under FakeTok
    bodies = []
    for tail in ("0123456789", "zyxwvu"):
        with contextlib.redirect_stdout(io.StringIO()):
            head, raw = Req("/v1/completions", {"prompt": shared + tail,
                                                "max_tokens": 3}).response()
        bodies.append(json.loads(raw))
    t1, t2 = bodies[0]["timings"], bodies[1]["timings"]
    assert t1["cache_n"] == 0 and t1["cache_source"] == "none" and t1["prompt_n"] == 42, t1
    assert t2["cache_n"] == 32 and t2["prompt_n"] == 6 and t2["cache_source"] == "prefix", t2
    assert bodies[1]["usage"]["prompt_tokens"] == 38


def test_a_response_cache_hit_is_a_replay():
    rc = {"response_cache": cache.ResponseCache(1 << 20, 3600.0)}
    serve()
    bodies = []
    real = app.generate_stream
    app.generate_stream = _script(_ids("same answer"))
    app.STATE["tok"] = UTok()
    app.STATE.update(rc)
    try:
        for _ in range(2):
            with contextlib.redirect_stdout(io.StringIO()):
                head, raw = Req("/v1/chat/completions", _chat(stream=False)).response()
            bodies.append(json.loads(raw))
    finally:
        app.generate_stream = real
    first, again = bodies[0], bodies[1]
    assert first["timings"]["cache_source"] == "none"
    assert again["timings"]["cache_source"] == "response", again["timings"]
    assert again["usage"]["prompt_tokens_details"]["cached_tokens"] == again["usage"]["prompt_tokens"]
    assert again["timings"]["prompt_n"] == 0
    assert again["usage"]["completion_tokens"] == first["usage"]["completion_tokens"]


def _replays(bodies: list[dict]) -> list[str]:
    """The cache source of each request, in order, through one response cache."""
    serve()
    real = app.generate_stream
    app.generate_stream = _script(_ids("thought</think>same answer"))
    app.STATE["tok"] = UTok()
    app.STATE["response_cache"] = cache.ResponseCache(1 << 20, 3600.0)
    out = []
    try:
        for body in bodies:
            with contextlib.redirect_stdout(io.StringIO()):
                head, raw = Req("/v1/chat/completions", body).response()
            out.append(json.loads(raw)["timings"]["cache_source"])
    finally:
        app.generate_stream = real
    return out


def test_the_response_cache_keys_on_the_effort_level():
    """`high` and `xhigh` render one prompt (the template has one sentence for both) and, at a
    16,384-token limit, one cap after the answer reserve: before the key named the level, a run
    at `xhigh` replayed a `high` run's answers. The same level still replays."""
    def body(effort):
        return {"messages": [{"role": "user", "content": "q"}], "stream": False,
                "max_tokens": 16384, "reasoning_effort": effort}

    assert _replays([body("high"), body("xhigh")]) == ["none", "none"]
    assert _replays([body("xhigh"), body("xhigh")]) == ["none", "response"]
    assert _replays([body("medium"), body("high"), body("high")]) == ["none", "none", "response"]


# ------------------------------------------------------------------ failures and bodies

def test_a_failed_stream_carries_its_partial_counts():
    def broken(*a, **k):
        for c in "abcde":
            yield ord(c)
        raise RuntimeError("the engine fell over")

    head, raw, _ = _run(_chat(), engine=broken)
    chunks = _chunks(raw)
    where = _carriers(chunks)
    assert len(where) == 1
    fin = chunks[where[0]]
    assert fin["choices"][0]["finish_reason"] == "error" and fin["error"]["type"] == "RuntimeError"
    assert fin["usage"]["completion_tokens"] == 5, fin["usage"]


def test_the_non_streamed_body():
    head, raw, _ = _run(_chat(stream=False), engine=_script(_ids("An answer.")))
    body = json.loads(raw)
    u = body["usage"]
    # the OpenAI fields, and our engine's split beside them (SRV-349; clients ignore the key)
    assert set(u) == {"prompt_tokens", "completion_tokens", "total_tokens",
                      "prompt_tokens_details", "completion_tokens_details", "tandem"}, u
    assert sum(u["tandem"]["parts"].values()) == u["prompt_tokens"], u["tandem"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert set(body["timings"]) == set(usage.LLAMA_KEYS + usage.ENGINE_KEYS), body["timings"]
    assert set(body["metrics"]) == {"time_to_first_token_ms", "generation_time_ms",
                                    "queue_time_ms", "mean_itl_ms", "tokens_per_second",
                                    "speculative_decoding"}


def test_the_timings_are_consistent():
    def slow(*a, **k):
        time.sleep(0.03)                                   # the "prefill"
        for c in "abcdefgh":
            time.sleep(0.005)
            yield ord(c)

    for stream in (True, False):
        head, raw, _ = _run(_chat(stream=stream), engine=slow)
        fin = _chunks(raw)[_carriers(_chunks(raw))[0]] if stream else json.loads(raw)
        t, u = fin["timings"], fin["usage"]
        assert abs(t["ttft_ms"] - (t["queue_ms"] + t["prompt_ms"])) <= 0.1, t
        assert t["prompt_ms"] >= 25.0, t
        rate = (u["completion_tokens"] - 1) * 1000.0 / t["predicted_ms"]
        assert abs(rate - t["predicted_per_second"]) <= 0.02 * rate + 0.01, (rate, t)
        assert t["total_ms"] >= t["ttft_ms"] + t["predicted_ms"] - 0.1, t
        assert fin["metrics"]["time_to_first_token_ms"] == t["ttft_ms"]
        assert consistent({k: fin[k] for k in FIELDS}) == [], consistent(fin)


def test_the_req_line_is_unchanged():
    """The `[req]` line keeps rc4's format: row3 and accept_hist parse it."""
    from tools.rowlog import parse_requests
    head, raw, log = _run(_chat(max_tokens=10), engine=_engine_with_blocks(10, [(7, 2, 3)]))
    line = [x for x in log.splitlines() if x.startswith("[req]")][0]
    import re
    assert re.fullmatch(r"\[req\] chatcmpl-[0-9a-f]{24} stream prompt=\d+ completion=10 "
                        r"finish=length \d+ ms [0-9.]+ tok/s blocks=3 committed=9 "
                        r"decode_ms=[0-9.]+ accept=7:2x3", line), line
    assert parse_requests(log)[0]["completion"] == 10


def test_the_live_checker_passes_against_the_handler():
    """Tools/usage_check.py, the check run against :8011 and through the reverse proxy, on a real socket
    over the real handler: every case passes, and it fails a server that sends usage twice."""
    import threading
    from http.server import ThreadingHTTPServer
    from tools import usage_check as uc
    serve()
    app.STATE["tok"] = UTok()
    real = app.generate_stream
    app.generate_stream = lambda *a, **k: iter(_ids("Hmm.</think>\n\nDanube, Rhine, Elbe."))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    args = type("A", (), {"base": f"http://127.0.0.1:{httpd.server_address[1]}", "model": "t",
                          "token": None, "prompt": "q", "max_tokens": 64, "timeout": 30.0})()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            results = [uc.run_case(args, *case) for case in uc.CASES]
            bad = {r["case"]: r["errors"] for r in results if r["errors"]}
            assert not bad, bad
            think = next(r for r in results if r["case"] == "stream-thinking")
            assert think["usage"]["completion_tokens_details"]["reasoning_tokens"] == len(
                "Hmm.</think>"), think["usage"]
            # a server that sends usage on the finish chunk AND a separate chunk is caught
            real_chunk = app._chunk

            def twice(cid, model, created, delta, finish=None, usage=None, error=None,
                      extra=None):
                out = real_chunk(cid, model, created, delta, finish, usage, error, extra)
                if extra and finish is not None:
                    out += real_chunk(cid, model, created, None, extra=extra)
                return out

            app._chunk = twice
            try:
                r = uc.run_case(args, *uc.CASES[0])
            finally:
                app._chunk = real_chunk
            assert r["errors"] and "exactly 1" in r["errors"][0], r
    finally:
        httpd.shutdown()
        app.generate_stream = real


def test_the_record_on_its_own():
    r = usage.RequestRecord("x", "chat", True, t_arrival=10.0)
    r.t_lock, r.t_first, r.t_last, r.t_end = 10.0004, 12.2401, 18.2611, 18.2612
    r.prompt_tokens, r.completion_tokens, r.reasoning_tokens = 1959, 412, 230
    r.cached_tokens, r.prompt_n, r.cache_source = 1536, 423, "prefix"
    r.blocks, r.draft_n, r.draft_accepted = 90, 1350, 322
    f = r.fields()
    t = f["timings"]
    assert t["prompt_ms"] == 2239.7 and t["queue_ms"] == 0.4 and t["ttft_ms"] == 2240.1, t
    assert t["predicted_ms"] == 6021.0 and t["predicted_per_second"] == 68.26, t
    assert t["prompt_per_second"] == 188.86 and t["prompt_per_token_ms"] == 5.29, t
    assert t["predicted_per_token_ms"] == 14.65 and t["tokens_per_block"] == 4.57, t
    assert f["metrics"]["tokens_per_second"] == 49.87, f["metrics"]
    assert f["metrics"]["speculative_decoding"]["draft_acceptance_rate"] == 0.2385
    assert f["usage"]["total_tokens"] == 2371
    # refused before the lock: renders, all zeros
    z = usage.RequestRecord("y", "chat", True).fields()
    assert z["timings"]["ttft_ms"] == 0.0 and z["usage"]["completion_tokens"] == 0


def test_reasoning_count():
    rc = usage.reasoning_count
    assert rc([1, 2, 9, 4], 9, [7, 8]) == 3
    assert rc([1, 7, 8, 4], 9, [7, 8]) == 3
    assert rc([1, 2, 3], 9, [7, 8]) == 3
    assert rc([], 9, [7, 8]) == 0
    assert rc([7, 1, 7, 8], None, [7, 8]) == 4


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
