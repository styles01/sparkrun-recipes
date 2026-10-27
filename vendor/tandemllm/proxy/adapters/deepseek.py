"""DeepSeek's API (OpenAI-shaped chat completions).

Its usage carries the cache split as `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`
(verified on api-docs.deepseek.com on 2026-10-04): the hits fill `cached_tokens`. The misses are
checked against `prompt_tokens - cached_tokens`; a mismatch is one log line, never a correction.
Reasoning arrives as `delta.reasoning_content`. The lineup read on api-docs.deepseek.com on
2026-10-04: `deepseek-flash` (legacy names `deepseek-v4-flash`, `deepseek-v4-flash-vision-exp`)
and `deepseek-v4-pro`, both with a 1M context and thinking on by default, switched per request by
`thinking: {"type": "enabled" | "disabled"}` (the older `deepseek-chat` and `deepseek-reasoner`
pick is gone from the page). It cannot prefill an assistant message except in its beta prefix
mode, so `continue_final_message` is off.

PRICES BY THE HOUR. DeepSeek charges its list price in peak hours (01:00-04:00 and 06:00-10:00 UTC,
Monday to Friday) and half of it otherwise. A model's price row may carry an `offpeak` block; the
cost of a request that arrived outside peak hours uses it. Chinese public holidays are off-peak at
DeepSeek and peak here, so a weekday holiday's cost reads high by the difference.
"""

from __future__ import annotations

import datetime as _dt

from proxy.adapters.base import CHAT, COMPLETIONS, Adapter, _int


class DeepSeek(Adapter):
    kind = "deepseek"
    cloud = True
    # the V4 names; `deepseek-v4-flash` is the legacy id of `deepseek-flash`
    names = {"deepseek-flash": "DeepSeek V4 Flash", "deepseek-v4-flash": "DeepSeek V4 Flash",
             "deepseek-v4-pro": "DeepSeek V4 Pro"}
    # a larger max_tokens is refused: "the valid range of max_tokens is [1, 393216]" (2026-10-16)
    max_output = dict.fromkeys(names, 393216)

    # `reasoning_effort` (api-docs.deepseek.com, 2026-10-17): none | low | high | max, high by
    # default; `none` turns thinking off. The model list names the levels per model.
    effort_off = "none"

    @staticmethod
    def display_name(name) -> str | None:
        """"DeepSeek-V4.1-Flash" from the model list reads "DeepSeek V4.1 Flash"."""
        return name.strip().replace("-", " ") if isinstance(name, str) and name.strip() else None

    def __init__(self, backend: dict, log=None):
        super().__init__(backend)
        self.log = log or (lambda msg: None)

    def routes(self) -> tuple[str, ...]:
        return (CHAT, COMPLETIONS)

    def capabilities(self) -> dict:
        return {"continue_final_message": False, "choice": "text",
                "reasoning_field": "reasoning_content",
                "cache_counts": ["prompt_cache_hit_tokens", "prompt_cache_miss_tokens"],
                "thinking_switch": "thinking_type"}

    @staticmethod
    def peak(at: float) -> bool:
        t = _dt.datetime.fromtimestamp(at, _dt.timezone.utc)
        return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)

    def price(self, usage: dict, price: dict | None, at: float | None = None) -> int | None:
        if price and price.get("offpeak") and at is not None and not self.peak(at):
            off = {k: v for k, v in price["offpeak"].items() if v is not None}
            price = dict(price, **off)
        return self.cost(usage, price)

    def read_usage(self, obj, path: str = CHAT) -> dict:
        out = super().read_usage(obj, path)
        if not isinstance(obj, dict):
            return out
        hit = _int(obj.get("prompt_cache_hit_tokens"))
        miss = _int(obj.get("prompt_cache_miss_tokens"))
        if hit is not None:
            out["cached_tokens"] = hit
        if hit is not None and miss is not None and out["prompt_tokens"] is not None and \
                out["prompt_tokens"] - hit != miss:
            self.log(f"[proxy] {self.backend.get('id')}: prompt_cache_miss_tokens {miss} != "
                     f"prompt_tokens {out['prompt_tokens']} - hits {hit}; kept as reported")
        return out
