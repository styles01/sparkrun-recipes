"""One record per request: the single source of its usage, its timings and its ledger row.

Open WebUI shows nothing under an answer unless the stream carries `usage`, and it shows speed only
when it carries llama.cpp's `timings` (Open WebUI 0.11.3). The numbers were all in
the server already -- the `[req]` line printed most of them -- but in four places and computed
four ways. `RequestRecord` is filled as the request runs and everything that reports on a request
reads it: the usage JSON here, the Prometheus observations and the ledger row.

THE TIME POINTS, all `time.perf_counter()`:

    t_arrival   the handler entered `_complete`
    t_lock      the engine lock was acquired
    t_first     the first token reached the handler (the prefill ran inside that `next()`)
    t_last      the last token reached the handler
    t_end       the request was logged, just before its finish chunk is written

so `queue_ms = t_lock - t_arrival`, `prompt_ms = t_first - t_lock` and `ttft_ms` is exactly their
sum. `predicted_per_second` is `(completion_tokens - 1) / predicted_ms`: the atlas row's divisor
and the `[req]` line's `committed/decode_ms`. llama.cpp divides by `predicted_n` instead; the
first token is the prefill's, not decode's, so this engine does not.

EXACTNESS. `completion_tokens` counts the ids the engine committed -- the generator yields
committed tokens only, so a drafted-and-rejected token can never be counted -- with the EOS
included, as the non-streamed body always counted it. Reasoning tokens are the committed ids up to
and including the one that closes the block (the special id or the literal text, as ThinkBudget
watches both), all of them if it never closed; the tokens a forced close writes are reasoning.

Nothing here imports torch.
"""

from __future__ import annotations

import time

# The timings keys, in order: llama.cpp's names and meanings first (tools/server/server-common.cpp
# in llama.cpp), then this engine's own. Open WebUI prints every one of them in its tooltip.
LLAMA_KEYS = ("cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
              "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second",
              "draft_n", "draft_n_accepted")
#: The cache sources whose answer came out of memory with no decode: the response cache
#: (`response`, an identical greedy request) and a summary written ahead (`prepared`).
#: Their "decode rate" is the speed of a replay; no speed view, average or histogram reads it, and
#: the ledger records none.
REPLAY = ("response", "prepared")

ENGINE_KEYS = ("ttft_ms", "queue_ms", "total_ms", "blocks", "tokens_per_block", "reasoning_n",
               "cache_source")


def _ms(a: float | None, b: float | None) -> float | None:
    return (b - a) * 1e3 if a is not None and b is not None else None


def _r(x: float | None, nd: int = 2) -> float:
    return round(float(x), nd) if x is not None else 0.0


def special_id(tokenizer, text: str) -> int | None:
    """The id of `text` when the tokenizer holds it as ONE token, else None.

    engine/spec.py's `_special_id` falls back to the last id of the text's own tokens, which is
    right for a budget that only needs something to watch and wrong for a count: on a tokenizer
    without the special token it would end the reasoning block at the first `>`.
    """
    i = tokenizer.convert_tokens_to_ids(text)
    if isinstance(i, int) and i >= 0 and i != getattr(tokenizer, "unk_token_id", None):
        return i
    return None


def reasoning_count(ids: list[int], end_id: int | None, end_text: list[int] | None) -> int:
    """Committed ids through the one that closes the reasoning block; all of them if none does.

    The model closes the block with the special `</think>` id or, on prompts that look like raw
    text, with the literal characters (engine/spec.py's ThinkBudget watches both, for the same
    reason), so both end it here.
    """
    seq = [int(t) for t in (end_text or [])]
    k = len(seq)
    for i, t in enumerate(ids):
        if end_id is not None and t == end_id:
            return i + 1
        if k and i + 1 >= k and [int(x) for x in ids[i + 1 - k:i + 1]] == seq:
            return i + 1
    return len(ids)


def client_cached_of(rec, cached) -> int:
    """`cached` (the engine's rows read from a cache, in served positions) as the client's
    tokens: the one definition every view of a request uses (`usage`, the ledger row, the live
    view, the parallel view), so `cached_tokens <= prompt_tokens` holds wherever both are shown.

    `prompt_tokens` is the client's count (SRV-165); the engine's figure also counts the prompt's
    hidden tokens (the router's notes, the end anchor's replay, SP4's note). With their positions
    known (`hidden_at`, SRV-233) the hidden tokens inside the cached span come off; without them
    the clamp to `prompt_tokens` holds. A prompt with no hidden tokens is the client's own, and
    its cached rows never exceed it; the clamp holds there too (a prompt the client sent is all
    a cache can hold of it)."""
    c = int(cached or 0)
    p = getattr(rec, "prompt_tokens", None)
    if p is None:
        return c
    p = int(p)
    hidden_at = getattr(rec, "hidden_at", None)
    if (getattr(rec, "hidden_prompt", 0) and hidden_at is not None
            and getattr(rec, "prompt_client", None) is None):
        below = sum(max(0, min(int(b), c) - int(a)) for a, b in hidden_at if int(a) < c)
        c -= below
    return max(0, min(c, p))


class RequestRecord:
    """Everything one request did, filled in as it happens.

    The handler sets the fields it knows; `track()` stamps the token times; `absorb_*` read the
    loop's own counters. The derived values are properties, so a record that was refused before
    the lock -- no prompt, no tokens -- still renders, with zeros where nothing happened.
    """

    def __init__(self, request_id: str, endpoint: str, stream: bool,
                 t_arrival: float | None = None, ts: float | None = None):
        self.request_id = request_id
        self.endpoint = endpoint                  # "chat" | "completions"
        self.stream = bool(stream)
        self.ts = time.time() if ts is None else ts
        self.t_arrival = time.perf_counter() if t_arrival is None else t_arrival
        self.t_lock = self.t_first = self.t_last = self.t_end = None
        self.slot: int | None = None              # the engine's sequence slot (`--parallel N`)
        self.model = ""
        self.status = 200
        self.finish_reason: str | None = None
        self.error_type: str | None = None
        self.prompt_tokens: int | None = None
        # SRV-165: the prompt's tokens the client did not send (the router's notes, new and
        # replayed; the monitor's SP4 note; an earlier answer's hidden tokens put back), which
        # `prompt_tokens` leaves out; `usage.prompt_tokens_details.hidden_tokens` when not 0
        self.hidden_prompt = 0
        self.cached_tokens = 0
        self.prompt_n: int | None = None          # forwarded; None = the prompt's whole length
        # A compaction served as a continuation reads a prompt longer than the one
        # the client sent; its cached tokens are the client's prompt less what was forwarded
        self.prompt_client: int | None = None
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.blocks = 0
        self.draft_n = 0
        self.draft_accepted = 0
        self.cache_source = "none"
        self.tool_calls = 0
        self.thinking = False
        self.max_tokens: int | None = None
        self.client_id = "anon"
        self.client_kind = "other"
        self.temperature: float | None = None
        # tokens handed to the handler so far, read once a second by server/live.py.
        # Equals `completion_tokens` at the end; the one per-token cost of the live view.
        self.n_live = 0
        # the live activity (server/activity.py). Each is written once per request or once
        # per transition by the handler, never per token, and read by the sampler thread.
        self.live_refs = None          # (ThinkBudget, ToolCallBuffer | None) of the running choice
        self.prefill_info = None       # the prefill's own info dict: start, t0, chunk, kind
        self.pf_marks = []             # per prefill chunk, one (done, total, t), appended by the hook
        self.decode_anchor = None      # (t, tokens) where the live decode window starts (the sampler)
        self.sock = None               # the client's socket, for the sampler's close check
        self.conv = None               # a conversation label (body, X-Conversation-Id, x-session-id)
        self.constrained = None        # "response_format" | "tool_choice" | None
        self.step = None               # after the token loop: flush, saving_state, final_chunk
        self.t_finishing = None
        self.end_state = None          # the state the token loop ended in
        self.stop_detail = None        # refines the stop reason (eos, stop_string, queue_full, ...)
        self.argloop = None            # the tool-argument loop guard's verdict (server/argloop.py)
        self.argloop_history = None    # The stopped calls cut from the history
        self.tool_names = None
        self.client_gone_at = None     # perf time the server first saw the socket closed
        self.continues = None          # the request this one follows after a tool_calls finish
        self.stop = None               # the stop block, made once at the end
        # the conductor (server/conductor/): its plan, and what the ledger records of it
        self.plan = None
        # SRV-137: the output's ids and calls, kept for the session transcript (TandemLLM only)
        self.out = None                # (ids, in_think)
        self.out_calls = None
        self.probe = None              # ENG-181: the probe snapshots' capture (server/probe.py)
        self.model_served = None       # TandemLLM, or the raw engine's name
        self.profile = None
        self.profile_version = None
        self.conductor_mode = None     # raw | shadow | on
        self.sup = None                # the monitor's supervisor of this request (the live view)
        self.conductor_acts = None     # the monitor's executed acts (SRV-115)
        self.conductor_hidden_tokens = None   # tokens the model saw that the client did not send
        self.conductor_acted = None    # 1: the conductor changed what the model saw or wrote
        # The side lane (server/preempt.py): the job, and where it was served
        self.lane_job = None
        self.lane = None
        self.queue_limit_s = None   # how long this request may wait for the engine
        # SRV-177: ms this request waited behind TandemLLM's own work (the shadow or prepare job,
        # the conductor's question); None when it waited behind none
        self.wait_own_ms = None
        # SRV-250: ms the parallel memory admission held this request after it took its slot
        # (inside `queue_ms`, never `prompt_ms`)
        self.wait_mem_ms: float | None = None
        # SRV-305: why the request waited ("memory" while the admission held it for memory) and
        # whether it is waiting for memory right now (the live and parallel views' lane state)
        self.wait_reason: str | None = None
        self.waiting_memory = False
        # SRV-241: how the request sampled -- None greedy, `client` the client's own sampling,
        # `regenerate` the regenerate's (server/regenerate.py); `regen` the detection's decision
        self.sampling: str | None = None
        # SRV-475: `loop_escape` / `card` sampling's detail for the usage frame (`tandem.sampling`)
        # and the escape record the session monitor's note reads (SP4)
        self.sampling_detail: dict | None = None
        self.loop_escape: dict | None = None
        self.regen = None
        # SRV-233: where the served prompt holds tokens the client did not send, as
        # `[(start, end)]` in served positions (the router's notes, the end anchor's replay, a
        # reasoning the client left out, SP4's note); None: not known (the clamp of SRV-165)
        self.hidden_at: list | None = None
        # FX16: the same ranges without the usage count check, for the end anchor's record
        self.note_at: list | None = None
        # VIS-60: an internal row's request (the turn the router's question served)
        self.parent_request = None
        # ledger v3 (the proxy, P1): a request relayed to another backend (proxy/relay.py fills
        # these and overrides the fields it cannot see); `user` is the proxy key's label, also on
        # an engine row
        self.backend = None
        self.backend_kind = None
        self.backend_model = None
        self.public_model = None
        self.usage_source = None
        self.cache_write_tokens = None
        self.cost_micros = None
        self.proxy_ms = None
        self.retries = None
        self.fallback_from = None
        self.upstream_status = None
        self.user = None
        # SRV-349: the prompt's head by part ({system, tools, images}, from the renderer) and the
        # router's choices for it ({class, effort, packages}); `usage.tandem` when `parts` is set
        self.parts: dict | None = None
        self.harness: dict | None = None
        # the park or the pin the rows came back from before the prefill resumed (None: in place)
        self.cache_via: str | None = None

    # ----------------------------------------------------------------- filling it in
    def track(self, source):
        """Pass the token source through, stamping the first and the last token as they arrive.

        One integer add per token beside the two stamps: no lock, no allocation."""
        for t in source:
            now = time.perf_counter()
            if self.t_first is None:
                self.t_first = now
            self.t_last = now
            self.n_live += 1
            yield t

    @property
    def replay(self) -> bool:
        """Answered from memory, nothing decoded (`REPLAY`)."""
        return self.cache_source in REPLAY

    def lock_acquired(self) -> None:
        self.t_lock = time.perf_counter()

    def absorb_prefill(self, info: dict | None) -> None:
        """`STATE["last_prefill"]` of this request's own prefill: tokens restored and forwarded."""
        if not info:
            return
        self.cached_tokens = int(info.get("reused") or 0)
        self.prompt_n = int(info.get("forwarded") or 0)
        if self.cached_tokens:
            self.cache_source = str(info.get("kind") or "prefix")
            self.cache_via = info.get("via")
        if self.prompt_client is not None:
            self.cached_tokens = max(0, int(self.prompt_client) - self.prompt_n)
            self.cache_source = "continuation"

    def absorb_response_cache(self) -> None:
        """A replay: every prompt token came out of a cache and none was forwarded."""
        self.cache_source = "response"
        self.cached_tokens = int(self.prompt_tokens or 0)
        self.prompt_n = 0

    def absorb_blocks(self, bs) -> None:
        """The decode loop's `BlockStats`: forwards paid, and the first-miss histogram.

        `accept` maps a block's draft depth to {accepted: count}, so the tokens proposed and kept
        are exact sums over it. A step the drafter declined is a block with no draft.
        """
        if bs is None:
            return
        # added, not set: an `n > 1` request runs one generation a choice
        self.blocks += int(bs.blocks)
        self.draft_n += sum(d * n for d, h in bs.accept.items() for n in h.values())
        self.draft_accepted += sum(a * n for h in bs.accept.values() for a, n in h.items())

    def end(self) -> None:
        if self.t_end is None:
            self.t_end = time.perf_counter()

    # ----------------------------------------------------------------- derived
    @property
    def queue_ms(self) -> float | None:
        return _ms(self.t_arrival, self.t_lock)

    @property
    def prompt_ms(self) -> float | None:
        return _ms(self.t_lock, self.t_first)

    @property
    def ttft_ms(self) -> float | None:
        q, p = self.queue_ms, self.prompt_ms
        return q + p if q is not None and p is not None else None

    @property
    def predicted_ms(self) -> float | None:
        return _ms(self.t_first, self.t_last)

    @property
    def total_ms(self) -> float | None:
        return _ms(self.t_arrival, self.t_end)

    @property
    def forwarded(self) -> int:
        if self.prompt_n is not None:
            return self.prompt_n
        return int(self.prompt_tokens or 0) - self.cached_tokens

    @property
    def committed(self) -> int:
        """Decode's tokens: the first one is the prefill's."""
        return max(0, self.completion_tokens - 1)

    @property
    def decode_tps(self) -> float | None:
        ms = self.predicted_ms
        return self.committed / (ms / 1e3) if self.committed and ms else None

    @property
    def prefill_tps(self) -> float | None:
        ms = self.prompt_ms
        return self.forwarded / (ms / 1e3) if self.forwarded and ms else None

    @property
    def tokens_per_block(self) -> float | None:
        return self.committed / self.blocks if self.blocks else None

    @property
    def draft_acceptance(self) -> float | None:
        return self.draft_accepted / self.draft_n if self.draft_n else None

    # ----------------------------------------------------------------- the three objects
    def usage(self) -> dict:
        """VLLM's usage, the details included."""
        p, c = int(self.prompt_tokens or 0), int(self.completion_tokens)
        details = {"cached_tokens": self.client_cached}
        if self.hidden_prompt:
            details["hidden_tokens"] = int(self.hidden_prompt)
        out = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
               "prompt_tokens_details": details,
               "completion_tokens_details": {"reasoning_tokens": int(self.reasoning_tokens)}}
        tandem = self.tandem()
        if tandem is not None:
            out["tandem"] = tandem
        return out

    def tandem(self) -> dict | None:
        """SRV-349, the Context panel's split (an extra key: OpenAI clients ignore it). `parts`
        are the served prompt's: `system`, `tools` and `images` from the renderer, `injected` the
        tokens the client did not send (`hidden_tokens`: the router's notes, new and replayed, the
        end anchor's replay), `history` the rest -- so system + tools + history + images is
        `prompt_tokens` and the five add up to the prompt the engine read, which `cache` splits
        into what came from a tier (`cached`, the engine's `cache_n`) and what was forwarded
        (`reread`). `harness`: the router's class, effort level and the packages in the prompt, null
        for a request no router plan served. None for a request with no rendered chat prompt."""
        if self.parts is None or self.prompt_tokens is None:
            return None
        from server.cache_peek import tier_of
        system, tools = int(self.parts.get("system") or 0), int(self.parts.get("tools") or 0)
        images = int(self.parts.get("images") or 0)
        p = int(self.prompt_tokens)
        return {"parts": {"system": system, "tools": tools,
                          "history": max(0, p - system - tools - images),
                          "injected": int(self.hidden_prompt or 0), "images": images},
                "harness": self.harness,
                "cache": {"tier": tier_of(self.cache_via or self.cache_source),
                          "source": self.cache_source,
                          "cached": int(self.cached_tokens), "reread": int(self.forwarded)},
                **({"sampling": self.sampling_detail} if self.sampling_detail else {})}

    @property
    def client_cached(self) -> int:
        """SRV-165: `cached_tokens` as a share of what the client sent: the prompt's tokens read
        from a cache, at most `prompt_tokens` (the cache also holds the hidden notes, which the
        client's count leaves out). The engine's own figure stays in `timings.cache_n`.

        SRV-233 (UX2, 2026-10-07): the clamp alone counted the hidden rows INSIDE the cached span
        as client tokens: an opencode turn whose engine prompt was 24,706 client + 5,745 note
        tokens, resumed at 26,624 and forwarded 3,827, reported 24,706 of 24,706 cached. With the
        hidden tokens' positions known (`hidden_at`), the client's cached tokens are the cached
        span less the hidden tokens in it: 26,624 - 5,745 = 20,879."""
        return client_cached_of(self, self.cached_tokens)

    def timings(self) -> dict:
        """Llama.cpp's `timings`, the one speed shape Open WebUI displays, plus the engine's keys."""
        n = self.forwarded
        pms = self.prompt_ms or 0.0
        dms = self.predicted_ms or 0.0
        return {
            "cache_n": int(self.cached_tokens),
            "prompt_n": int(n),
            "prompt_ms": _r(pms),
            "prompt_per_token_ms": _r(pms / n if n else 0.0),
            "prompt_per_second": _r(self.prefill_tps),
            "predicted_n": int(self.completion_tokens),
            "predicted_ms": _r(dms),
            "predicted_per_token_ms": _r(dms / self.committed if self.committed else 0.0),
            "predicted_per_second": _r(self.decode_tps),
            "draft_n": int(self.draft_n),
            "draft_n_accepted": int(self.draft_accepted),
            "ttft_ms": _r(self.ttft_ms),
            "queue_ms": _r(self.queue_ms),
            "total_ms": _r(self.total_ms),
            "blocks": int(self.blocks),
            "tokens_per_block": _r(self.tokens_per_block),
            "reasoning_n": int(self.reasoning_tokens),
            "cache_source": self.cache_source,
        }

    def metrics(self) -> dict:
        """VLLM 0.29's per-response `metrics` object. Open WebUI does not read it; vLLM-aware
        clients do."""
        total = self.total_ms or 0.0
        return {
            "time_to_first_token_ms": _r(self.ttft_ms),
            "generation_time_ms": _r(self.predicted_ms),
            "queue_time_ms": _r(self.queue_ms),
            "mean_itl_ms": _r((self.predicted_ms or 0.0) / self.committed
                              if self.committed else 0.0),
            "tokens_per_second": _r(self.completion_tokens / (total / 1e3) if total else 0.0),
            "speculative_decoding": {
                "mean_acceptance_length": _r(self.tokens_per_block),
                "draft_acceptance_rate": _r(self.draft_acceptance, 4),
            },
        }

    def fields(self) -> dict:
        """`usage`, `timings` and `metrics` together, for the one chunk or body that carries them."""
        return {"usage": self.usage(), "timings": self.timings(), "metrics": self.metrics()}

    def row(self, engine_version: str = "", code_sha: str = "") -> dict:
        """The ledger row: counts and times, nothing a person wrote or read.

        A request that never reached the engine -- refused at the queue, rejected with a 400 --
        has null token and speed fields rather than zeros, so a sum over the ledger is honest and
        a percentile over it does not see a refusal as an infinitely slow request.
        """
        ran = self.prompt_tokens is not None
        decoded = ran and self.t_first is not None

        def r2(x):
            return round(x, 2) if x is not None else None

        return {
            "ts_ms": int(self.ts * 1000), "request_id": self.request_id, "model": self.model,
            "client_id": self.client_id, "client_kind": self.client_kind,
            "endpoint": self.endpoint, "stream": int(self.stream), "status": int(self.status),
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens if ran else None,
            "cached_tokens": self.client_cached if ran else None,
            "completion_tokens": self.completion_tokens if ran else None,
            "reasoning_tokens": self.reasoning_tokens if ran else None,
            "queue_ms": r2(self.queue_ms), "prompt_ms": r2(self.prompt_ms),
            "ttft_ms": r2(self.ttft_ms), "decode_ms": r2(self.predicted_ms),
            "total_ms": r2(self.total_ms),
            "decode_tps": r2(self.decode_tps) if decoded and not self.replay else None,
            "prefill_tps": r2(self.prefill_tps) if decoded and not self.replay else None,
            "blocks": self.blocks if ran else None,
            "draft_tokens": self.draft_n if ran else None,
            "draft_accepted": self.draft_accepted if ran else None,
            "tool_calls": int(self.tool_calls), "thinking": int(self.thinking),
            "cache_source": self.cache_source if ran else None,
            "max_tokens": self.max_tokens, "error_type": self.error_type,
            "engine_version": engine_version, "code_sha": code_sha,
            "model_served": self.model_served, "profile": self.profile,
            "profile_version": self.profile_version, "conductor_mode": self.conductor_mode,
            "conductor_acts": self.conductor_acts,
            "conductor_hidden_tokens": self.conductor_hidden_tokens,
            "conductor_acted": self.conductor_acted,
            "lane": self.lane,
            "backend": self.backend, "backend_kind": self.backend_kind,
            "backend_model": self.backend_model, "public_model": self.public_model,
            "usage_source": self.usage_source, "cache_write_tokens": self.cache_write_tokens,
            "cost_micros": self.cost_micros, "proxy_ms": r2(self.proxy_ms),
            "retries": self.retries, "fallback_from": self.fallback_from,
            "upstream_status": self.upstream_status, "user": self.user,
            "wait_own_ms": self.wait_own_ms,
            "parent_request": self.parent_request,
            "slot": self.slot,
            "engine_cached": int(self.cached_tokens or 0) if ran else None,
            "forwarded_tokens": int(self.forwarded) if ran else None,
            "wait_mem_ms": r2(self.wait_mem_ms),
            "sampling": self.sampling,
        }


def placement(body: dict, stream: bool, default_on: bool) -> str:
    """Where a response's usage goes: `body`, `separate`, `finish` or `none`. Exactly one place.

    Open WebUI's merge ADDS the token counts of every chunk that carries usage (utils/response.py
    `merge_usage`), so a stream that carried it twice would show doubled counts. It asks for
    `include_usage` only when a model's "Usage" capability is ticked, which it is not on the base
    models -- hence the default: with no `stream_options` the finish chunk carries it (a chunk
    with a non-empty `choices`, so a client that indexes `choices[0]` is safe), with
    `include_usage: true` the separate `choices: []` chunk does, as vLLM sends it, and an explicit
    `include_usage: false` gets nothing.
    """
    if not stream:
        return "body"
    opts = body.get("stream_options")
    asked = opts.get("include_usage") if isinstance(opts, dict) else None
    if asked is not None:
        return "separate" if asked else "none"
    return "finish" if default_on else "none"
