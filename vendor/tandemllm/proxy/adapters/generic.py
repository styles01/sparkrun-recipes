"""Any OpenAI-compatible server reached by URL: llama.cpp, LM Studio, Ollama, vLLM and SGLang (their
L1 tiles come in P2 and P3), another TandemLLM. Chat, completions and embeddings are relayed; the
usage the server sends is read as the base adapter reads it.

`vllm` and `sglang` are the generic adapter with what the harness (P4) may ask of each, read from
their docs and code (vLLM v0.30.0, SGLang v0.5.21): both prefill an assistant message
(`continue_final_message`), both switch Qwen-family thinking through `chat_template_kwargs`, and
each constrains the class question its own way (vLLM `structured_outputs.choice`, SGLang
`regex`). A server of either kind registered as `generic` relays the same and loses only those
three."""

from __future__ import annotations

from proxy.adapters.base import CHAT, COMPLETIONS, EMBEDDINGS, RESPONSES, Adapter


class Generic(Adapter):
    kind = "generic"
    cloud = False

    def routes(self) -> tuple[str, ...]:
        # vLLM and SGLang serve the Responses API too; a server that does not answers its own 404
        return (CHAT, COMPLETIONS, EMBEDDINGS, RESPONSES)


class TandemLLM(Generic):
    """Our own engine at a URL (the standalone proxy in front of it). It puts `usage` on the finish
    chunk unasked, so the stream is not asked for a separate usage frame, and it reads `X-Tandem`
    and the session headers, which the relay passes on."""

    kind = "tandemllm"

    def capabilities(self) -> dict:
        caps = super().capabilities()
        caps["choice"] = "grammar"
        return caps

    def choice_headers(self) -> list[tuple[str, str]]:
        return [("X-Tandem", "raw")]


class VLLM(Generic):
    kind = "vllm"

    def capabilities(self) -> dict:
        caps = super().capabilities()
        caps.update(choice="guided_choice", continue_final_message=True)
        return caps


class SGLang(Generic):
    kind = "sglang"

    def capabilities(self) -> dict:
        caps = super().capabilities()
        caps.update(choice="regex", continue_final_message=True)
        return caps
