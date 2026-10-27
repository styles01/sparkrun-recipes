"""A fake OpenAI-compatible backend for the proxy's tests: the standard library, no model.

    python tests/fake_backend.py --port 9101 --models fake-a,fake-b [--key sk-fake-...]
    python tests/fake_backend.py --port 9102 --dialect vllm        (a /metrics page for P2)

or in a test: `srv, port = fake_backend.start(models=["fake-a"])`, `srv.shutdown()`.

It answers `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses` and
`/v1/models` the way OpenAI does: a stream asked for `stream_options.include_usage` gets
`"usage": null` in every chunk and a last `choices: []` chunk with the usage; a stream that did
not ask gets no usage at all. Words in the last user message steer it:

    FAKE_NO_USAGE        never any usage
    FAKE_USAGE_FINISH    usage on the finish chunk, unasked (our engine's default placement)
    FAKE_CACHED <n>      usage says <n> cached prompt tokens
    FAKE_TANDEM          usage carries our engine's `tandem` object (SRV-349: parts, harness)
    FAKE_REASONING       a few `reasoning_content` deltas before the answer
    FAKE_THINK_TAGS      the reasoning as <think>...</think> inside `content`
    FAKE_MEMORY          a memory block (a user rule) after the answer, in 7-character pieces
    FAKE_TOOL            two tool calls, their arguments streamed in pieces
    FAKE_CALLS [...]     these calls (JSON list of {name, arguments, id?}, the rest of the line),
                         when the request offers tools and its last message is not a tool result
                         (FAKE_CALLS_ALWAYS: after tool results too, for loop guards)
    FAKE_SLOW_TTFT <s>   wait <s> seconds before the first token
    FAKE_SLOW            ten tokens a second
    FAKE_TPS <n>         <n> tokens a second (the parallel load: backends of different speeds)
    FAKE_STALL <s>       stop for <s> seconds after the third token
    FAKE_DROP            close the connection after five tokens, no [DONE] (FAKE_DROP_ONCE: only
                         the first time per text, so a client's retry gets through)
    FAKE_429             429 with Retry-After: 1 (FAKE_429_ONCE: only the first time per text)
    FAKE_503             503 (FAKE_503_ONCE: only the first time per text)
    FAKE_500             500
    FAKE_DEEPSEEK        DeepSeek's usage shape: prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                         and a `system_fingerprint` in every chunk
    FAKE_LONG <n>        <n> answer tokens (default 24)
    FAKE_SAY <text>      this answer (the rest of the line, `\\n` a line break)
    FAKE_ERROR_FRAME     an `error` object inside the stream after three tokens
    FAKE_FINISH_ERROR    reasoning, then a finish chunk with `finish_reason: "error"` (our engine's
                         failure mid-stream; FAKE_FINISH_ERROR_ONCE: only the first time per text)

For the harness (P4):

    FAKE_EMPTY_ANSWER    reasoning, then no content and no tool call (E1's case)
    FAKE_BAD_CALL        one tool call whose arguments are not JSON and cannot be repaired
    FAKE_ALMOST_JSON     one tool call whose arguments have a trailing comma (repairable)
    FAKE_LOOP            reasoning that repeats one sentence (D1's case)
    FAKE_ARGLOOP         a `write` call whose content repeats one map row (the argument guard)
    FAKE_CLASS <id>      the class question's answer (else its first option)

The memory writer's request (its last user message starts "Memory pass.") is steered by the
session's earlier user messages: FAKE_WRITER_NONE, FAKE_WRITER_BASH, FAKE_WRITER_SLOW
(`_writer_answer`).

A request whose last message is a tool result answers plainly. The class question (a body with
`structured_outputs.choice`, `regex` or a `json_schema` response format, or a system prompt that
lists task classes) is answered without a stream, with logprobs. A body with
`continue_final_message` continues: it streams `closed` words as content. Every request body is
kept in `fake.bodies`.

A script (in a test): `fake.script = fn(body, headers)` answers a chat completion it returns a
dict for (None: the words above decide). The dict: `content` (the answer's text), `reasoning`,
`calls` ([{name, arguments, id?}], as FAKE_CALLS), `status` (an error answer with that status and
`message`), `drop` (the stream breaks after five pieces), `finish` (`error`: our engine's failure
mid-stream), `tps` (pieces a second; a piece is five characters), `ttft` (seconds before the first
chunk). The tests of the workers (VIS-E15) steer the main agent and each worker by the request's
`x-session-id`. `fake.live = fn()` serves the engine's live rows (`GET /v1/dashboard/live?follow=0`;
the stream answers 429, so chatd's reader polls).

A prefix cache: a request whose messages begin with the messages of an earlier request reports
those messages' tokens as `cached_tokens` (a backend's prefix cache, for P4's replay test). A token
here is a whitespace-separated word (plus 4 per message). `GET /fake/last` returns the last
request's path, headers and body; `GET /fake/stats` the counts and the most streams it served at
once. `refuse=<status>` (in a test) makes every completion answer that status. `memory=<n>` (in a
test) answers a completion of more than n prompt tokens with our engine's memory refusal: a 503
"a <n>-token prefill needs ... GiB with every cache given back", Retry-After 1 (`fake.refused`
counts them). `window=<n>` (in a test) answers a completion of more than n prompt tokens with our
engine's 400 for a prompt past the context window (`fake.windowed` counts them); with
`strict = True` the prompt and `max_tokens` together count, and the 400 is a strict
OpenAI-compatible backend's ("maximum context length is n tokens", a relay's backend). `tokenize` (in a test) answers `POST
/v1/tokenize` (server/tokenize_api.py) with the prompt's tokens, or this number when it is an int;
None: 404, a server without the route (`fake.counted` counts the calls). `--key` makes
every route need `Authorization: Bearer <key>`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _words(text) -> int:
    if isinstance(text, list):
        return sum(_words(p.get("text", "")) for p in text if isinstance(p, dict))
    return len(str(text or "").split())


WRITER_BLOCK = ('<tandem_memory op="add" scope="project" kind="gotcha">\ntitle: the build needs '
                'node 20\n\nThe build fails on node 18 with an import attribute error; use node '
                '20.\n</tandem_memory>')
WRITER_BASH = ("<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n"
               "</tool_call>")


def _text_of(c) -> str:
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


def _writer_answer(body: dict, text: str) -> list:
    """The memory writer's answer (a request whose last user message starts "Memory pass."):

        FAKE_WRITER_NONE    NONE
        FAKE_WRITER_BASH    a `bash` tool call as text (what Qwen wrote after a compaction)
        FAKE_WRITER_SLOW    the block at 5 pieces a second (a client request stops it)
        otherwise           one project gotcha block, in 9-character pieces
    """
    if "FAKE_WRITER_NONE" in text:
        return ["NONE"]
    if "FAKE_WRITER_BASH" in text:
        return [WRITER_BASH[i:i + 5] for i in range(0, len(WRITER_BASH), 5)]
    head = [] if body.get("continue_final_message") else ["Notes:\n"]
    return head + [WRITER_BLOCK[i:i + 9] for i in range(0, len(WRITER_BLOCK), 9)]


def _last_user(body: dict) -> str:
    if isinstance(body.get("input"), str):
        return body["input"]
    for m in reversed(body.get("messages") or body.get("input") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in c or [] if isinstance(p, dict))
    p = body.get("prompt")
    return p if isinstance(p, str) else ""


class Fake:
    def __init__(self, models=("fake-a",), key=None, tps=400.0, dialect=None, name="fake",
                 refuse=None, cache=True):
        self.models = list(models)
        # extra fields of a model in /v1/models, by id (DeepSeek's name, context_window, ...)
        self.cards: dict[str, dict] = {}
        self.cache = cache              # the prefix cache (off: every request is cold)
        self.refuse = refuse            # a status every completion answers (a backend that is down)
        self.memory: int | None = None  # the prompt tokens over which the engine has no memory
        self.refused = 0
        self.window: int | None = None  # the prompt tokens over which the request is too long
        self.windowed = 0
        self.strict = False             # the window holds the prompt and max_tokens (strict)
        self.tokenize = None            # POST /v1/tokenize: True (count) or an int; None: 404
        self.counted = 0
        self.key = key
        self.tps = float(tps)
        self.dialect = dialect
        self.name = name
        self.lock = threading.Lock()
        self.last = None
        self.requests = 0
        self.streams = 0
        self.max_streams = 0
        self.prefixes: dict[str, int] = {}
        self.once: set = set()
        self.closed_early = 0
        self.bodies: list = []
        # SRV-349: what POST /v1/cache/peek answers (None: 404, a server without the route) and
        # how many prewarms came
        self.peek: dict | None = None
        self.prewarms = 0
        self.script = None              # fn(body, headers) -> dict | None (see the module doc)
        self.calls: list = []           # (headers, body) of every chat completion, in order
        # fn() -> the rows of GET /v1/dashboard/live?follow=0 (None: 404, a server without the
        # route); the stream itself answers 429, so a reader polls
        self.live = None

    def prompt_tokens(self, body: dict) -> tuple[int, int]:
        """(prompt tokens, cached) with the prefix cache: the longest earlier message prefix."""
        msgs = body.get("messages")
        if not isinstance(msgs, list):
            text = body.get("prompt") or body.get("input") or ""
            if isinstance(text, list):
                text = json.dumps(text)
            return _words(text) + 4, 0
        total, cached, h = 0, 0, hashlib.sha256()
        hit = True
        for m in msgs:
            n = _words((m or {}).get("content")) + 4
            total += n
            h.update(json.dumps(m, sort_keys=True).encode())
            key = h.hexdigest()
            with self.lock:
                if hit and self.cache and key in self.prefixes:
                    cached = total
                else:
                    hit = False
                self.prefixes[key] = total
        return total, cached


def make_handler(fake: Fake):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "FakeBackend"
        # as common OpenAI-compatible servers (uvicorn) do: without it, a keep-alive connection's
        # headers and first frame wait 40 ms for the reader's delayed ACK on Linux, and the
        # bench measured that instead of the proxy
        disable_nagle_algorithm = True

        def log_message(self, *a):
            pass

        def _json(self, code, obj, headers=()):
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _authed(self) -> bool:
            if not fake.key:
                return True
            if self.headers.get("Authorization") == f"Bearer {fake.key}":
                return True
            self._json(401, {"error": {"message": "Incorrect API key provided",
                                       "type": "invalid_request_error",
                                       "code": "invalid_api_key"}})
            return False

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/fake/last":
                return self._json(200, fake.last or {})
            if p == "/fake/stats":
                return self._json(200, {"requests": fake.requests,
                                        "max_streams": fake.max_streams,
                                        "streams": fake.streams,
                                        "closed_early": fake.closed_early})
            if p == "/v1/dashboard/live" and fake.live is not None:
                if "follow=0" not in self.path:
                    return self._json(429, {"error": {"message": "too many live streams"}})
                return self._json(200, {"requests": fake.live(), "engine": {}})
            if p in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok"})
            if p == "/version" and fake.dialect == "vllm":
                return self._json(200, {"version": "0.30.0"})
            if p == "/server_info" and (fake.dialect or "").startswith("sglang"):
                return self._json(200, {"version": "0.5.21", "speculative_algorithm": None,
                                        "enable_metrics": fake.dialect == "sglang",
                                        "enable_cache_report": True, "context_length": 32768})
            if p == "/metrics" and fake.dialect in ("vllm", "sglang"):
                raw = metrics_page(fake).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if p in ("/v1/models", "/models"):
                if not self._authed():
                    return
                return self._json(200, {"object": "list", "data": [
                    {"id": m, "object": "model", "created": 1, "owned_by": fake.name,
                     "max_model_len": 32768, **fake.cards.get(m, {})} for m in fake.models]})
            return self._json(404, {"error": {"message": "no route", "type": "not_found"}})

        def do_POST(self):
            p = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._json(400, {"error": {"message": "bad json"}})
            with fake.lock:
                fake.requests += 1
                if not p.startswith("/v1/cache/"):
                    # `last` is the last completion: chatd's cache peek runs on its own thread
                    # and would race the request a test looks at
                    fake.last = {"path": p, "headers": dict(self.headers.items()),
                                 "body": body, "raw": raw.decode("utf-8", "replace")}
                fake.bodies.append(body)
            if not self._authed():
                return
            if p in ("/v1/cache/peek", "/v1/cache/prewarm"):
                if fake.peek is None:
                    return self._json(404, {"error": {"message": "no route", "type": "not_found"}})
                if p.endswith("prewarm"):
                    with fake.lock:
                        fake.prewarms += 1
                    return self._json(200, {"prewarmed": True, **fake.peek})
                return self._json(200, fake.peek)
            text = _last_user(body)
            if text.startswith("Memory pass."):
                # the memory writer's request (proxy/memwriter.py): steered by the session's
                # earlier user messages
                hist = " ".join(_text_of(m.get("content")) for m in (body.get("messages") or [])
                                [:-1] if isinstance(m, dict) and m.get("role") == "user")
                text = "FAKE_WRITER " + hist + (" FAKE_TPS 5" if "FAKE_WRITER_SLOW" in hist
                                                else "")
            if fake.refuse:
                return self._json(fake.refuse, {"error": {"message": f"fake {fake.refuse}",
                                                          "type": "server_error"}},
                                  (("Retry-After", "1"),))
            if fake.memory is not None and p.endswith("/completions"):
                n_prompt = sum(_words((m or {}).get("content")) + 4
                               for m in body.get("messages") or [])
                if n_prompt > fake.memory:
                    # server/app.py `_busy` with engine/cache.py MemoryRefused's words
                    with fake.lock:
                        fake.refused += 1
                    return self._json(503, {"error": {
                        "message": f"a {n_prompt}-token prefill needs 53.3 GiB with every cache "
                                   f"given back; the engine's peak is 52.6 GiB; retry when the "
                                   f"engine is idle", "type": "server_busy", "code": 503}},
                        (("Retry-After", "1"),))
            if p == "/v1/tokenize":
                if fake.tokenize is None:
                    return self._json(404, {"error": {"message": "no route", "type": "not_found"}})
                with fake.lock:
                    fake.counted += 1
                n = fake.tokenize if isinstance(fake.tokenize, int) and \
                    not isinstance(fake.tokenize, bool) else \
                    sum(_words((m or {}).get("content")) + 4 for m in body.get("messages") or [])
                return self._json(200, {"tokens": n, "parts": {"system": 0, "tools": 0,
                                                               "messages": n}})
            if fake.window is not None and fake.strict and p.endswith("/completions"):
                n_prompt = sum(_words((m or {}).get("content")) + 4
                               for m in body.get("messages") or [])
                n_out = int(body.get("max_tokens") or 0)
                if n_prompt + n_out > fake.window:
                    with fake.lock:
                        fake.windowed += 1
                    return self._json(400, {"error": {
                        "message": f"This model's maximum context length is {fake.window} "
                                   f"tokens. However, you requested {n_prompt + n_out} tokens "
                                   f"({n_prompt} in the messages, {n_out} in the completion).",
                        "type": "BadRequestError", "param": None, "code": 400}})
            if fake.window is not None and p.endswith("/completions"):
                n_prompt = sum(_words((m or {}).get("content")) + 4
                               for m in body.get("messages") or [])
                if n_prompt > fake.window:
                    # server/app.py's `room <= 0` refusal
                    with fake.lock:
                        fake.windowed += 1
                    return self._json(400, {"error": {
                        "message": f"prompt is {n_prompt} tokens and the context is "
                                   f"{fake.window}; nothing is left to generate",
                        "type": "invalid_request_error", "param": "messages"}})
            for word, code in (("FAKE_429", 429), ("FAKE_503", 503), ("FAKE_500", 500)):
                if word + "_ONCE" in text:
                    with fake.lock:
                        k = (word, text)
                        first = k not in fake.once
                        fake.once.add(k)
                    if first:
                        return self._json(code, {"error": {"message": f"fake {code}",
                                                           "type": "rate_limit_error"}},
                                          (("Retry-After", "1"),))
                elif word in text:
                    return self._json(code, {"error": {"message": f"fake {code}",
                                                       "type": "rate_limit_error"}},
                                      (("Retry-After", "1"),))
            if p in ("/v1/models",):
                return self._json(405, {})
            if p == "/v1/embeddings":
                pt = _words(body.get("input"))
                return self._json(200, {"object": "list", "model": body.get("model"),
                                        "data": [{"object": "embedding", "index": 0,
                                                  "embedding": [0.1, 0.2, 0.3]}],
                                        "usage": {"prompt_tokens": pt, "total_tokens": pt}})
            if p == "/v1/responses":
                return self._responses(body, text)
            if p not in ("/v1/chat/completions", "/v1/completions"):
                return self._json(404, {"error": {"message": "no route"}})
            if p.endswith("chat/completions") and _is_question(body):
                return self._question(body, text)
            if p.endswith("chat/completions"):
                with fake.lock:
                    fake.calls.append((dict(self.headers.items()), body))
                got = fake.script(body, dict(self.headers.items())) if fake.script else None
                if got is not None:
                    return self._scripted(body, got)
            return self._complete(p, body, text)

        def _scripted(self, body, got: dict):
            """A completion the test's script wrote (`fake.script`)."""
            if got.get("status"):
                return self._json(int(got["status"]), {"error": {
                    "message": got.get("message") or f"fake {got['status']}",
                    "type": "invalid_request_error"}})
            content = got.get("content") or ""
            reasoning = [got["reasoning"]] if got.get("reasoning") else []
            answer = [content[i:i + 5] for i in range(0, len(content), 5)]
            self._special, self._calls = None, got.get("calls")
            self._n_msgs = len(body.get("messages") or [])
            tools = bool(self._calls)
            usage = self._usage(body, "", len(reasoning) + len(answer) + (8 if tools else 0),
                                len(reasoning))
            words = ("FAKE_DROP " if got.get("drop") else "") + \
                ("FAKE_FINISH_ERROR " if got.get("finish") == "error" else "") + \
                (f"FAKE_TPS {got['tps']} " if got.get("tps") else "")
            if not body.get("stream"):
                msg = {"role": "assistant", "content": content}
                return self._json(200, {"id": "chatcmpl-s", "object": "chat.completion",
                                        "created": 1, "model": body.get("model"), "choices": [
                                            {"index": 0, "message": msg, "finish_reason": "stop"}],
                                        "usage": usage})
            opts = body.get("stream_options") or {}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                self._stream(True, "chatcmpl-s" + ulid_ish(), body.get("model"), reasoning, answer,
                             tools, usage, opts.get("include_usage") is True, False, words,
                             float(got.get("ttft") or 0.0))
            except (BrokenPipeError, ConnectionResetError, OSError):
                with fake.lock:
                    fake.closed_early += 1

        def _question(self, body, text):
            """The router's class question: an option, greedy, with logprobs."""
            opts = ((body.get("structured_outputs") or {}).get("choice")
                    or (((body.get("response_format") or {}).get("json_schema") or {})
                        .get("schema", {}).get("properties", {}).get("choice", {}).get("enum")))
            if not opts and body.get("regex"):
                opts = body["regex"].strip("()").replace("\\", "").split("|")
            if not opts:
                sysp = (body.get("messages") or [{}])[0].get("content") or ""
                opts = re.findall(r"^- ([\w-]+):", sysp, re.M)
            m = re.search(r"FAKE_CLASS ([\w-]+)", text)
            pick = m.group(1) if m and m.group(1) in opts else (opts[0] if opts else "none")
            content = json.dumps({"choice": pick}) if body.get("response_format") else pick
            pt, _ = fake.prompt_tokens(body)
            return self._json(200, {"id": "chatcmpl-q", "object": "chat.completion",
                                    "created": 1, "model": body.get("model"), "choices": [{
                                        "index": 0, "message": {"role": "assistant",
                                                                "content": content},
                                        "logprobs": {"content": [{"token": pick,
                                                                  "logprob": -0.05}]},
                                        "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": pt, "completion_tokens": 2,
                                              "total_tokens": pt + 2}})

        # ------------------------------------------------------------- chat and completions
        def _plan(self, body, text):
            if text.startswith("FAKE_WRITER "):
                return [], _writer_answer(body, text)
            m = re.search(r"FAKE_LONG (\d+)", text)
            n_answer = int(m.group(1)) if m else 24
            reasoning = []
            if body.get("continue_final_message"):
                return [], [f"closed{i} " for i in range(6)]
            msgs = body.get("messages") or []
            if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "tool":
                return [], [f"done{i} " for i in range(6)]
            m = re.search(r"FAKE_SAY (.+)", text)
            if m:
                say = m.group(1).replace("\\n", "\n")
                return [], [say[i:i + 5] for i in range(0, len(say), 5)]
            if "FAKE_REASONING" in text or "FAKE_EMPTY_ANSWER" in text:
                reasoning = [f"think{i} " for i in range(6)]
            if "FAKE_LOOP" in text:
                reasoning = [f"step {i} " for i in range(4)] + \
                    ["I should check the map again and then decide what to do next. "] * 120
            answer = [f"w{i} " for i in range(n_answer)]
            if "FAKE_EMPTY_ANSWER" in text:
                answer = []
            if "FAKE_THINK_TAGS" in text:
                answer = ["<think>", "plan ", "more ", "</think>"] + answer
            if "FAKE_MEMORY" in text:
                # 2026-10-05: a memory block after the answer, streamed in small pieces
                blk = ('\n\n<tandem_memory op="add" scope="user" kind="rule">\ntitle: short '
                       'answers\n\nThe user wants answers of at most five sentences.\n'
                       '</tandem_memory>')
                answer = answer + [blk[i:i + 7] for i in range(0, len(blk), 7)]
            return reasoning, answer

        def _usage(self, body, text, n_out, n_reason):
            pt, cached = fake.prompt_tokens(body)
            m = re.search(r"FAKE_CACHED (\d+)", text)
            if m:
                cached = min(pt, int(m.group(1)))
            if "FAKE_DEEPSEEK" in text:
                return {"prompt_tokens": pt, "completion_tokens": n_out,
                        "total_tokens": pt + n_out, "prompt_cache_hit_tokens": cached,
                        "prompt_cache_miss_tokens": pt - cached,
                        "completion_tokens_details": {"reasoning_tokens": n_reason}}
            out = {"prompt_tokens": pt, "completion_tokens": n_out, "total_tokens": pt + n_out,
                   "prompt_tokens_details": {"cached_tokens": cached},
                   "completion_tokens_details": {"reasoning_tokens": n_reason}}
            if "FAKE_TANDEM" in text:
                # our engine's split and the harness's choices (SRV-349)
                out["tandem"] = {"parts": {"system": 2, "tools": 3, "history": pt - 5,
                                           "injected": 0, "images": 0},
                                 "harness": {"class": "code_edit", "effort": "xhigh",
                                             "packages": [{"name": "coding", "tokens": 7}]},
                                 "cache": {"tier": "resident", "source": "end",
                                           "cached": cached, "reread": pt - cached}}
            return out

        def _complete(self, p, body, text):
            chat = p.endswith("chat/completions")
            reasoning, answer = self._plan(body, text)
            last_tool = bool(body.get("messages")) and isinstance(body["messages"][-1], dict) \
                and body["messages"][-1].get("role") == "tool"
            special = None
            for w in ("FAKE_BAD_CALL", "FAKE_ALMOST_JSON", "FAKE_ARGLOOP"):
                if w in text and bool(body.get("tools")) and not last_tool and \
                        not body.get("continue_final_message"):
                    special = w
            self._special = special
            m = re.search(r"FAKE_CALLS (\[.*\])", text)
            self._calls = json.loads(m.group(1)) if m else None
            self._n_msgs = len(body.get("messages") or [])
            tools = ("FAKE_TOOL" in text and bool(body.get("tools")) and not last_tool) or \
                special is not None or (self._calls is not None and bool(body.get("tools")) and
                                        (not last_tool or "FAKE_CALLS_ALWAYS" in text))
            n_out = len(reasoning) + len(answer) + (8 if tools else 0)
            model = body.get("model")
            cid = ("chatcmpl-" if chat else "cmpl-") + hashlib.sha1(
                text.encode()).hexdigest()[:20]
            usage = self._usage(body, text, n_out, len(reasoning))
            if "FAKE_NO_USAGE" in text:
                usage = None
            m = re.search(r"FAKE_SLOW_TTFT ([0-9.]+)", text)
            ttft = float(m.group(1)) if m else 0.0
            if not body.get("stream"):
                if ttft:
                    time.sleep(ttft)
                content = "".join(answer)
                if chat:
                    msg = {"role": "assistant", "content": content}
                    if reasoning:
                        msg["reasoning_content"] = "".join(reasoning)
                    if tools:
                        msg["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
                            "name": "read", "arguments": '{"path": "a.txt"}'}}]
                    choice = {"index": 0, "message": msg,
                              "finish_reason": "tool_calls" if tools else "stop"}
                else:
                    choice = {"index": 0, "text": content, "finish_reason": "stop"}
                out = {"id": cid, "object": "chat.completion" if chat else "text_completion",
                       "created": 1, "model": model, "choices": [choice]}
                if usage is not None:
                    out["usage"] = usage
                return self._json(200, out)
            opts = body.get("stream_options") or {}
            asked = isinstance(opts, dict) and opts.get("include_usage") is True
            on_finish = "FAKE_USAGE_FINISH" in text
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            with fake.lock:
                fake.streams += 1
                fake.max_streams = max(fake.max_streams, fake.streams)
            try:
                self._stream(chat, cid, model, reasoning, answer, tools, usage, asked, on_finish,
                             text, ttft)
            except (BrokenPipeError, ConnectionResetError, OSError):
                with fake.lock:
                    fake.closed_early += 1
            finally:
                with fake.lock:
                    fake.streams -= 1

        def _error_now(self, text: str) -> bool:
            """FAKE_FINISH_ERROR: every call of the turn; FAKE_FINISH_ERROR_ONCE: its first."""
            if "FAKE_FINISH_ERROR_ONCE" not in text:
                return True
            with fake.lock:
                k = ("FAKE_FINISH_ERROR", text)
                first = k not in fake.once
                fake.once.add(k)
            return first

        def _drop_now(self, text: str) -> bool:
            if "FAKE_DROP_ONCE" not in text:
                return True
            with fake.lock:
                k = ("FAKE_DROP", text)
                first = k not in fake.once
                fake.once.add(k)
            return first

        def _send_chunk(self, data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _event(self, obj):
            self._send_chunk(b"data: " + json.dumps(obj).encode() + b"\n\n")

        def _stream(self, chat, cid, model, reasoning, answer, tools, usage, asked, on_finish,
                    text, ttft):
            gap = 1.0 / (10.0 if "FAKE_SLOW" in text and "FAKE_SLOW_TTFT" not in text
                         else fake.tps)
            m = re.search(r"FAKE_TPS ([0-9.]+)", text)
            if m and float(m.group(1)) > 0:
                gap = 1.0 / float(m.group(1))
            m = re.search(r"FAKE_STALL ([0-9.]+)", text)
            stall = float(m.group(1)) if m else 0.0
            if ttft:
                time.sleep(ttft)

            def chunk(delta=None, finish=None, text_piece=None):
                if chat:
                    ch = {"index": 0, "delta": delta or {}, "finish_reason": finish}
                    obj = {"id": cid, "object": "chat.completion.chunk", "created": 1,
                           "model": model, "choices": [ch]}
                else:
                    obj = {"id": cid, "object": "text_completion", "created": 1, "model": model,
                           "choices": [{"index": 0, "text": text_piece or "",
                                        "finish_reason": finish}]}
                if asked:
                    obj["usage"] = None
                if "FAKE_DEEPSEEK" in text:      # DeepSeek names its configuration per chunk
                    obj["system_fingerprint"] = "fp_fake_e72d3c7f"
                return obj

            if chat:
                self._event(chunk({"role": "assistant", "content": ""}))
            k = 0
            for piece in reasoning:
                self._event(chunk({"reasoning_content": piece}))
                k += 1
                time.sleep(gap)
            if "FAKE_FINISH_ERROR" in text and self._error_now(text):
                # our engine's failure mid-stream: a finish chunk that says `error`, then [DONE]
                self._event(chunk({}, finish="error"))
                self._send_chunk(b"data: [DONE]\n\n")
                self._send_chunk(b"")
                return
            for piece in answer:
                self._event(chunk({"content": piece}) if chat else chunk(text_piece=piece))
                k += 1
                if k == 3 and stall:
                    time.sleep(stall)
                if k == 3 and "FAKE_ERROR_FRAME" in text:
                    self._event({"error": {"message": "fake upstream failure",
                                           "type": "server_error"}})
                    self._send_chunk(b"data: [DONE]\n\n")
                    self._send_chunk(b"")
                    return
                if k == 5 and "FAKE_DROP" in text and self._drop_now(text):
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.shutdown(2)
                    return
                time.sleep(gap)
            special = getattr(self, "_special", None)
            if tools and special is not None:
                if special == "FAKE_ARGLOOP":
                    name = "write"
                    row = "#" * 40 + ".." + "=" * 40 + "\\n"
                    parts = ['{"filePath": "map.txt", "content": "'] + [row] * 400 + ['"}']
                elif special == "FAKE_ALMOST_JSON":
                    name, parts = "read", ['{"path": ', '"a.txt",', '}']
                else:
                    name, parts = "read", ['{"path" ', '"a.txt" ', 'oops']
                self._event(chunk({"tool_calls": [{"index": 0, "id": "call_0",
                                                   "type": "function", "function": {
                                                       "name": name, "arguments": ""}}]}))
                for part in parts:
                    self._event(chunk({"tool_calls": [{"index": 0, "function": {
                        "arguments": part}}]}))
            elif tools and getattr(self, "_calls", None) is not None:
                for i, call in enumerate(self._calls):
                    args = json.dumps(call.get("arguments") or {})
                    self._event(chunk({"tool_calls": [{"index": i, "id": call.get("id") or
                                                       f"call_{i}_{self._n_msgs}",
                                                       "type": "function", "function": {
                                                           "name": call["name"], "arguments": ""}}]}))
                    for k in range(0, len(args), 7):
                        self._event(chunk({"tool_calls": [{"index": i, "function": {
                            "arguments": args[k:k + 7]}}]}))
            elif tools:
                for i, name in enumerate(("read", "write")):
                    self._event(chunk({"tool_calls": [{"index": i, "id": f"call_{i}",
                                                       "type": "function", "function": {
                                                           "name": name, "arguments": ""}}]}))
                    for part in ('{"path"', ': "a', '.txt"}'):
                        self._event(chunk({"tool_calls": [{"index": i, "function": {
                            "arguments": part}}]}))
            fin = chunk(finish="tool_calls" if tools else "stop")
            if on_finish and usage is not None:
                fin["usage"] = usage
            self._event(fin)
            if asked and usage is not None and not on_finish:
                self._event({"id": cid, "object": "chat.completion.chunk", "created": 1,
                             "model": model, "choices": [], "usage": usage})
            self._send_chunk(b"data: [DONE]\n\n")
            self._send_chunk(b"")

        # ------------------------------------------------------------- the Responses API
        def _responses(self, body, text):
            pt, cached = fake.prompt_tokens(body)
            m = re.search(r"FAKE_CACHED (\d+)", text)
            if m:
                cached = min(pt, int(m.group(1)))
            answer = [f"r{i} " for i in range(12)]
            usage = {"input_tokens": pt, "input_tokens_details": {"cached_tokens": cached},
                     "output_tokens": len(answer) + 3,
                     "output_tokens_details": {"reasoning_tokens": 3},
                     "total_tokens": pt + len(answer) + 3}
            resp = {"id": "resp_fake", "object": "response", "created_at": 1,
                    "model": body.get("model"), "status": "completed",
                    "output": [{"type": "message", "role": "assistant", "content": [
                        {"type": "output_text", "text": "".join(answer)}]}], "usage": usage}
            if not body.get("stream"):
                return self._json(200, resp)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            seq = 0

            def ev(kind, obj):
                nonlocal seq
                obj = dict(obj, type=kind, sequence_number=seq)
                seq += 1
                self._send_chunk(f"event: {kind}\ndata: {json.dumps(obj)}\n\n".encode())

            ev("response.created", {"response": dict(resp, status="in_progress", output=[],
                                                     usage=None)})
            ev("response.output_item.added", {"output_index": 0, "item": {
                "type": "message", "role": "assistant", "content": []}})
            for piece in answer:
                ev("response.output_text.delta", {"output_index": 0, "content_index": 0,
                                                  "delta": piece})
                time.sleep(1.0 / fake.tps)
            ev("response.completed", {"response": resp})
            self._send_chunk(b"")

    return H


_N = [0]


def ulid_ish() -> str:
    _N[0] += 1
    return f"{_N[0]:06d}"


def _is_question(body: dict) -> bool:
    if body.get("structured_outputs") or body.get("regex"):
        return True
    rf = body.get("response_format") or {}
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        return True
    msgs = body.get("messages") or []
    first = msgs[0] if msgs and isinstance(msgs[0], dict) else {}
    return first.get("role") == "system" and "task classes" in str(first.get("content") or "") \
        and not body.get("stream")


def metrics_page(fake: Fake) -> str:
    """A small /metrics page in vLLM's or SGLang's dialect (P2, P3): the real names and labels
    (proxy/engines.py), counters that move with the requests served. `sglang-off` answers 404,
    as SGLang does without --enable-metrics."""
    n = fake.requests
    m = fake.models[0]
    if fake.dialect == "vllm":
        lab = f'model_name="{m}",engine="0"'
        return (f'vllm:num_requests_running{{{lab}}} {fake.streams}\n'
                f'vllm:num_requests_waiting{{{lab}}} 0\n'
                f'vllm:kv_cache_usage_perc{{{lab}}} 0.12\n'
                f'vllm:prompt_tokens_total{{{lab}}} {n * 100}\n'
                f'vllm:generation_tokens_total{{{lab}}} {n * 20}\n'
                f'vllm:prefix_cache_queries_total{{{lab}}} {n * 100}\n'
                f'vllm:prefix_cache_hits_total{{{lab}}} {n * 40}\n')
    lab = f'model_name="{m}",engine_type="unified",tp_rank="0",pp_rank="0",moe_ep_rank="0"'
    tl = f'model_name="{m}",engine_type="unified"'
    return (f'sglang:num_running_reqs{{{lab}}} {fake.streams}\n'
            f'sglang:num_queue_reqs{{{lab}}} 0\n'
            f'sglang:token_usage{{{lab}}} 0.12\n'
            f'sglang:cache_hit_rate{{{lab}}} 0.4\n'
            f'sglang:prompt_tokens_total{{{tl},is_streaming="true"}} {n * 100}\n'
            f'sglang:cached_tokens_total{{{tl},cache_source="device"}} {n * 40}\n'
            f'sglang:generation_tokens_total{{{tl},is_streaming="true"}} {n * 20}\n')


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def start(port: int = 0, host: str = "127.0.0.1", **kw):
    """(server, port) running in a daemon thread; `server.fake` is the state."""
    fake = Fake(**kw)
    srv = Server((host, port), make_handler(fake))
    srv.fake = fake
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1},
                     daemon=True).start()
    return srv, srv.server_address[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--models", default="fake-a")
    ap.add_argument("--key", default=None)
    ap.add_argument("--tps", type=float, default=400.0)
    ap.add_argument("--dialect", choices=("vllm", "sglang", "sglang-off"), default=None)
    ap.add_argument("--name", default="fake")
    a = ap.parse_args(argv)
    srv, port = start(a.port, a.host, models=[m for m in a.models.split(",") if m], key=a.key,
                      tps=a.tps, dialect=a.dialect, name=a.name)
    print(f"[fake-backend] {a.name} on {a.host}:{port}, models {a.models}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    sys.exit(main())
