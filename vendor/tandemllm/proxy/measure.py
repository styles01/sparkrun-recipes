"""The record of one relayed request: the engine's `RequestRecord`, filled from what the proxy saw.

The proxy sees every byte of the request and the answer, so it measures what a client could
measure, and it says null for the rest:

    ttft_ms        arrival at the proxy to the first frame that carries a token (content,
                   reasoning or a tool-call delta). The network and the backend's own queue are
                   inside it. Null for an answer that was not streamed.
    decode_tps     (usage.completion_tokens - 1) / (last token frame - first token frame), the
                   engine's formula (server/usage.py). Without `usage` it is null: a frame is not a
                   token, and a frame count is never written as a token count.
    queue_ms       the proxy's own queue: 0 (a full backend answers 503, it does not hold)
    prompt_ms, prefill_tps, blocks, draft_tokens, draft_accepted, cache_source
                   null: they come off an engine's loop, and the proxy has none
    tokens         from `usage`; `usage_source` says `backend` or `none`
    cached_tokens, cache_write_tokens, reasoning_tokens
                   the provider's own fields when it sends them, else null
    cost_micros    the model's price at the time the row is written, else null
    proxy_ms       the proxy's own cost: arrival to the upstream send, less the time spent opening
                   a connection (DNS, TCP, TLS: the network's, logged as connect_ms), plus the
                   time between the first frame arriving and the proxy writing it to the client
"""

from __future__ import annotations

import time

from server import usage as usage_mod


class RelayRecord(usage_mod.RequestRecord):
    """A `RequestRecord` whose engine-only fields stay null. The live view and the ledger read
    it as they read an engine request; `track` is never used (a frame is not a token)."""

    def __init__(self, request_id: str, endpoint: str, stream: bool, t_arrival=None):
        super().__init__(request_id, endpoint, stream, t_arrival=t_arrival)
        self.t_sent = None             # the upstream request was written
        self.t_first_frame = None      # the first token frame arrived from upstream
        self.t_first_out = None        # ... and was written to the client
        self.cached_tokens = None
        self.completion_tokens = None
        self.reasoning_tokens = None
        self.cache_source = None
        self.prompt_n = None
        self.retries = 0
        self.frames = 0
        self.connect_ms = 0.0          # opening upstream connections: the network's, not the proxy's
        # the parallel view (server/parallel.py), written once per token frame by the relay's
        # own thread: the frames that carried reasoning, what the newest one carried, the call
        self.frames_reasoning = 0
        self.frames_tool = 0
        self.tool_chars = 0            # the calls' argument characters so far
        self.live_kind = None          # "reasoning" | "content" | "tool"
        self.live_tool = None
        self.system_fingerprint = None  # the answer's `system_fingerprint` (ledger schema 10)

    # the engine's split does not exist here
    @property
    def queue_ms(self):
        return 0.0 if self.t_sent is not None else None

    @property
    def prompt_ms(self):
        return None

    @property
    def ttft_ms(self):
        if self.t_first is None:
            return None
        return (self.t_first - self.t_arrival) * 1e3

    @property
    def prefill_tps(self):
        return None

    @property
    def committed(self) -> int:
        return max(0, int(self.completion_tokens or 0) - 1)

    @property
    def decode_tps(self):
        if self.usage_source != "backend" or not self.completion_tokens:
            return None
        ms = self.predicted_ms
        return self.committed / (ms / 1e3) if self.committed and ms and ms > 0 else None

    @property
    def forwarded(self) -> int:
        return 0

    def set_usage(self, u: dict) -> None:
        """The adapter's normalised usage. Every None stays None."""
        if u.get("prompt_tokens") is None and u.get("completion_tokens") is None:
            self.usage_source = "none"
            return
        self.usage_source = "backend"
        self.prompt_tokens = u.get("prompt_tokens")
        self.completion_tokens = u.get("completion_tokens")
        self.cached_tokens = u.get("cached_tokens")
        self.cache_write_tokens = u.get("cache_write_tokens")
        self.reasoning_tokens = u.get("reasoning_tokens")
        if self.prompt_tokens is not None and self.cached_tokens is not None:
            self.prompt_n = max(0, self.prompt_tokens - self.cached_tokens)

    def stamp_sent(self) -> None:
        self.t_sent = time.perf_counter()
        self.t_lock = self.t_sent              # the live view: "waiting for the first token"

    def stamp_token(self, t_arrived: float, kind: str | None = None,
                    tool: str | None = None, tool_chars: int | None = None) -> None:
        if self.t_first is None:
            self.t_first = t_arrived
            self.t_first_frame = t_arrived
        self.t_last = t_arrived
        self.frames += 1
        if kind is not None:
            self.live_kind = kind
            if kind == "reasoning":
                self.frames_reasoning += 1
            elif kind == "tool":
                self.frames_tool += 1
                if tool is not None:
                    self.live_tool = tool
                if tool_chars is not None:
                    self.tool_chars = tool_chars

    @property
    def proxy_ms_value(self) -> float | None:
        if self.t_sent is None:
            return None
        ms = max(0.0, (self.t_sent - self.t_arrival) * 1e3 - self.connect_ms)
        if self.t_first_out is not None and self.t_first_frame is not None:
            ms += max(0.0, (self.t_first_out - self.t_first_frame) * 1e3)
        return ms

    def row(self, engine_version: str = "", code_sha: str = "") -> dict:
        self.proxy_ms = self.proxy_ms_value
        out = super().row(engine_version, code_sha)
        ran = self.usage_source == "backend"
        decoded = ran and self.t_first is not None and self.decode_tps is not None

        def r2(x):
            return round(x, 2) if x is not None else None

        out.update({
            "prompt_tokens": self.prompt_tokens if ran else None,
            "cached_tokens": self.cached_tokens if ran else None,
            "completion_tokens": self.completion_tokens if ran else None,
            "reasoning_tokens": self.reasoning_tokens if ran else None,
            "queue_ms": r2(self.queue_ms), "prompt_ms": None, "ttft_ms": r2(self.ttft_ms),
            "decode_ms": r2(self.predicted_ms) if self.t_first is not None else None,
            "decode_tps": r2(self.decode_tps) if decoded else None,
            "prefill_tps": None, "blocks": None, "draft_tokens": None, "draft_accepted": None,
            "cache_source": None,
            "system_fingerprint": self.system_fingerprint,
        })
        return out
