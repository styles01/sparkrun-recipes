"""tandemllm-vllm-plugin: per-request internals from inside vLLM for TandemLLM's dashboard (layer L2).

Two entry points, both optional, both loaded by vLLM itself:

    vllm.stat_logger_plugins  tandem_stat_logger    -> tandem_vllm_plugin.logger:TandemStatLogger
    vllm.endpoint_plugins     tandem_stats_endpoint -> tandem_vllm_plugin.endpoint:TandemStatsEndpoint

The stat logger runs in vLLM's API server process, is called once per engine step, and pushes
what it saw to the proxy (`TANDEM_COLLECT_URL`, `TANDEM_COLLECT_TOKEN`). The endpoint adds
`GET /tandem/stats` to vLLM's own server for a proxy that pulls instead; vLLM loads an endpoint
plugin only when `VLLM_PLUGINS` names it.

See plugins/README.md in the TandemLLM repository for the versions it was checked against.
"""

__version__ = "0.1.0"

PLUGIN = "tandemllm-vllm-plugin"

# vLLM minor versions whose stats objects this plugin reads, checked against the tagged sources
# (v0.30.0, tag ced6857: vllm/v1/metrics/loggers.py and stats.py, vllm/plugins/__init__.py).
# A version outside it still loads: every field is read with getattr and a missing one is null.
TESTED = ("0.30",)
