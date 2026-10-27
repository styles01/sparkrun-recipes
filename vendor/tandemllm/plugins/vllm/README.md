# tandemllm-vllm-plugin

Per-request numbers from inside vLLM for TandemLLM's dashboard: queue, prefill and decode time, cached tokens, preemptions and the engine's finish reason for each request, and once a second the scheduler's state (running and waiting requests, KV use, prefix-cache hits, speculation per draft position).

    pip install tandemllm-vllm-plugin        # into vLLM's environment; pulls nothing
    TANDEM_COLLECT_URL=http://127.0.0.1:8080/v1/collect/vllm1 \
    TANDEM_COLLECT_TOKEN=... vllm serve Qwen/Qwen3.5-4B

vLLM loads the stat logger from the `vllm.stat_logger_plugins` entry point by itself. The plugin posts to the proxy once a second and never stops vLLM: when it does not fit the installed vLLM, vLLM starts without it. Tested versions and details: `plugins/README.md` in the TandemLLM repository. License: AGPL-3.0-only.
