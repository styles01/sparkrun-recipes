# tandemllm-sglang-plugin

Per-request numbers from inside SGLang for TandemLLM's dashboard: queue, prefill and decode time, cached tokens by tier, retractions and speculation (verify passes, accepted drafts) for each request, and once a second each scheduler's state.

    pip install tandemllm-sglang-plugin      # into SGLang's environment; pulls nothing
    TANDEM_COLLECT_URL=http://127.0.0.1:8080/v1/collect/sglang1 \
    TANDEM_COLLECT_TOKEN=... sglang serve --model-path Qwen/Qwen3.5-4B --enable-metrics

SGLang loads it from the `sglang.srt.plugins` entry point in each of its processes and applies its two hooks before serving. It needs `--enable-metrics`. It never stops SGLang: a hook that does not fit is skipped. Tested versions and details: `plugins/README.md` in the TandemLLM repository. License: AGPL-3.0-only.
