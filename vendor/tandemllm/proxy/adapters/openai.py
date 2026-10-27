"""OpenAI's API: chat completions, completions, embeddings, and the Responses API relayed as is.

The Responses API is never translated to chat completions and back: a translation would have to
invent a mapping for tools, built-in tools and reasoning items that the client already handles.
Its events are read for the ledger only (`response.output_text.delta` for the first token,
`response.completed` for `usage` with `input_tokens_details.cached_tokens` and
`output_tokens_details.reasoning_tokens`). Chat models hide their reasoning text; the count comes
from `completion_tokens_details.reasoning_tokens`.
"""

from __future__ import annotations

from proxy.adapters.base import CHAT, COMPLETIONS, EMBEDDINGS, RESPONSES, Adapter


class OpenAI(Adapter):
    kind = "openai"
    cloud = True

    def routes(self) -> tuple[str, ...]:
        return (CHAT, COMPLETIONS, EMBEDDINGS, RESPONSES)

    def capabilities(self) -> dict:
        return {"continue_final_message": False, "choice": "json_schema_enum",
                "reasoning_field": "none",
                "cache_counts": ["prompt_tokens_details.cached_tokens",
                                 "input_tokens_details.cached_tokens"],
                "thinking_switch": "reasoning_effort"}

    def forward(self, client_headers) -> list[tuple[str, str]]:
        try:
            beta = client_headers.get("OpenAI-Beta")
        except Exception:                                          # noqa: BLE001
            beta = None
        return [("OpenAI-Beta", beta)] if beta else []
