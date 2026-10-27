"""tandemllm-sglang-plugin: per-request internals from inside SGLang for TandemLLM's dashboard (layer L2).

SGLang v0.5.21 loads general plugins from the `sglang.srt.plugins` entry point group early in every
process it starts (`sglang.srt.plugins.load_plugins`: the HTTP server's process and each
scheduler's), unless `SGLANG_PLUGINS` names others only. A general plugin registers hooks with
`HookRegistry`; SGLang applies them before it serves. This package registers two AFTER hooks:

    sglang.srt.managers.tokenizer_manager.TokenizerManager.collect_metrics
        the HTTP process, once per output batch of a request; on its last one the hook reads the
        request's id (`rid`), its counts (prompt, completion, cached with the device, host and
        storage split, reasoning), its retractions, its speculation (verify passes, accepted
        drafts) and its times (the scheduler's queue time, prefill from forward entry to the end
        of prefill, decode from the first token to the end)
    sglang.srt.observability.metrics_collector.SchedulerMetricsCollector.log_stats
        each scheduler process, once per logging interval: running and queued requests, KV use,
        the cache hit rate, the generation rate, speculation acceptance, retractions

Both run only with `--enable-metrics` (SGLang calls neither without it). A hook never raises into
SGLang: a failure is counted and logged at most once a minute, and after 100 in a row the hook
turns itself off for the process. A target SGLang renamed is logged by SGLang's own `apply_hooks`
and skipped; the server starts the same.

Why hooks and not `ServerArgs.stat_loggers`: the tokenizer collector's
`observe_one_finished_request` receives no request id in v0.5.21, so a subclass of it could not
join its numbers to the proxy's row; `collect_metrics` has the request's state. The `stat_loggers`
map also holds classes and has no CLI flag, so it needs a launcher; the entry point does not.
"""

__version__ = "0.1.0"

PLUGIN = "tandemllm-sglang-plugin"

# SGLang minor versions whose structures this plugin reads, checked against the tagged sources
# (v0.5.21, tag e00930c: srt/plugins, srt/managers/tokenizer_manager.py and io_struct.py,
# srt/observability/metrics_collector.py and req_time_stats.py).
TESTED = ("0.5",)

TOKENIZER_TARGET = "sglang.srt.managers.tokenizer_manager.TokenizerManager.collect_metrics"
SCHEDULER_TARGET = "sglang.srt.observability.metrics_collector.SchedulerMetricsCollector.log_stats"


def register() -> None:
    """The `sglang.srt.plugins` entry point: register the two hooks. Never raises."""
    import logging
    log = logging.getLogger("sglang.tandem")
    try:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    except Exception as exc:                                       # noqa: BLE001
        log.warning("tandem: the plugin is off: sglang.srt.plugins.hook_registry did not import "
                    "(%s: %s); SGLang starts without it", type(exc).__name__, exc)
        return
    from tandem_sglang_plugin import hooks
    HookRegistry.register(TOKENIZER_TARGET, hooks.after_collect_metrics, HookType.AFTER)
    HookRegistry.register(SCHEDULER_TARGET, hooks.after_log_stats, HookType.AFTER)
