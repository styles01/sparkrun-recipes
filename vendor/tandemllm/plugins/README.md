# Plugins for vLLM and SGLang

The dashboard reads a vLLM or SGLang server from the outside: its `/metrics` page and the `usage` of each answer (layer L1, docs/proxy.md). That gives totals over five minutes. Per request it knows the tokens and its own time to first token, but not how long the request waited in the engine's queue, how long the prefill took, or whether the scheduler preempted it.

The two packages here run inside the engine and send exactly those numbers to the proxy (layer L2). Each is a small Python package with no dependencies of its own. Install it into the Python environment the engine runs in:

    pip install plugins/vllm       # tandemllm-vllm-plugin
    pip install plugins/sglang     # tandemllm-sglang-plugin

Then start the engine with two variables, or let `tandem-dash launch` do it (below):

    TANDEM_COLLECT_URL=http://127.0.0.1:8080/v1/collect/<backend id>
    TANDEM_COLLECT_TOKEN=<the proxy's collect token>

The token is `TANDEM_COLLECT_TOKEN` in the proxy's environment, or the one the proxy made and stored in its config database at the first need (`tandem-dash launch` reads it from there).

## What each one reads

| | vLLM | SGLang |
|-|-|-|
| how the engine loads it | the `vllm.stat_logger_plugins` entry point; vLLM builds the logger once per data-parallel engine in its API server process | the `sglang.srt.plugins` entry point; SGLang runs it in every process it starts and applies its two hooks before serving |
| when it runs | `record()` once per engine step | after `TokenizerManager.collect_metrics` (once per output batch of a request) and after `SchedulerMetricsCollector.log_stats` (once per logging interval) |
| per request | the request id, finish reason, prompt, cached and generated tokens, preemptions, queue, prefill, decode and end-to-end time, the NaN check | the request id (`rid`), finish reason, prompt, cached (device, host, storage), reasoning and generated tokens, retractions, queue time, prefill (forward entry to the end of prefill), decode, speculation verify passes and accepted drafts |
| per second | running and waiting requests, KV use, prefix-cache queries and hits, evictions, preemptions, speculation drafts and accepted tokens per draft position | running and queued requests, KV use, cache hit rate, generation rate, retractions, speculation accept length and rate |
| needs | nothing; `--enable-prompt-tokens-details` makes the answer's own usage carry cached tokens too | `--enable-metrics` (without it SGLang calls neither hook) |
| pull instead of push | `GET /tandem/stats` on vLLM's own port, when `VLLM_PLUGINS` names `tandem_stats_endpoint` | no |

Neither reads a prompt, an answer or a token id. Each item is counts and times; the proxy joins it to the request's ledger row by the id the proxy sent upstream (vLLM: the `X-Request-Id` header, SGLang: the body's `rid`).

## Tested versions

| plugin | engine | checked against | how |
|-|-|-|-|
| `tandemllm-vllm-plugin` 0.1.0 | vLLM 0.30 | v0.30.0, tag ced6857 (2026-09-22) | unit tests on stand-ins built from the tagged `vllm/v1/metrics/loggers.py`, `stats.py` and `vllm/plugins/__init__.py`; the real server in the night pass (`tools/proxy_engines_check.sh --plugins`) |
| `tandemllm-sglang-plugin` 0.1.0 | SGLang 0.5 | v0.5.21, tag e00930c (2026-10-02) | unit tests on stand-ins built from the tagged `srt/plugins/hook_registry.py`, `srt/managers/tokenizer_manager.py`, `io_struct.py` and `srt/observability/`; the real server in the night pass |

The plan is the last two minors of each engine. I checked one of each until now. A newer release still loads: the plugins read every field with `getattr`, and a missing or renamed one becomes a null in the ledger, not an error.

## When the plugin does not fit the engine

The engine always starts. The plugin never makes it fail:

- vLLM: if `StatLoggerBase` does not import, the plugin's module raises at import, and vLLM's plugin loader catches it, logs it and starts without the logger. An exception inside `record()` is caught and logged at most once a minute; after 100 in a row the logger switches itself off for that process.
- SGLang: if the hook registry does not import, the plugin logs one line and registers nothing. A hook target that a release renamed is logged and skipped by SGLang's own `apply_hooks`. A failing hook gets the same treatment as vLLM's `record()`.
- The proxy: a backend whose plugin has posted nothing for 60 s reads "L1 only" on its card, with the plugin and engine versions it last saw. An idle plugin posts an empty batch every 10 s, so an idle engine still reads L2.

## License

Both packages are part of TandemLLM and under the same license, AGPL-3.0-only (the `LICENSE` file in each directory is the repository's own). vLLM and SGLang are Apache-2.0 projects; the plugins import them at run time and contain none of their code.
