"""Generate the Grafana dashboard for the engine's /metrics, and its ConfigMap.

    python tools/grafana_dashboard.py        # writes ops/monitoring/qse-engine-dashboard.json
                                             #    and ops/monitoring/qse-engine-dashboard-configmap.yaml

The panels are server/METRICS.md's fifteen, in its order (tokens a second served, tokens per
block, draft acceptance, the block split, TTFT, queue, refusals, finish reasons, errors, prompt
tokens reused, cache bytes against the budget, memory, decode tok/s per request, reasoning share,
cached share), plus HTTP codes by route, prefill tok/s, suffix-store hits, the usage ledger, and the
GB10 row from dgx-exporter. A restart is an annotation (qse_engine_start_time_seconds changes).
Every query names only series this server or dgx-exporter publishes -- tests/test_monitoring.py
checks each name against the metric registry. Edit this script, not the JSON.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ops", "monitoring")
DS = {"type": "prometheus", "uid": "prometheus"}
UID = "qse-engine"
TITLE = "Qwen3.8 Spark engine"


def _target(expr: str, legend: str = "", ref: str = "A") -> dict:
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref}


def panel(title: str, exprs, unit: str = "short", *, desc: str = "", kind: str = "timeseries",
          stack: bool = False, w: int = 8, h: int = 8, decimals: int | None = None) -> dict:
    targets = [_target(e, lg, chr(65 + i)) for i, (e, lg) in enumerate(exprs)]
    p = {"type": kind, "title": title, "description": desc, "datasource": DS, "targets": targets,
         "gridPos": {"w": w, "h": h},
         "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
         "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                     "tooltip": {"mode": "multi", "sort": "none"}}}
    if decimals is not None:
        p["fieldConfig"]["defaults"]["decimals"] = decimals
    if kind == "timeseries":
        p["fieldConfig"]["defaults"]["custom"] = {
            "drawStyle": "line", "lineWidth": 1, "fillOpacity": 25 if stack else 8,
            "stacking": {"mode": "normal" if stack else "none", "group": "A"},
            "showPoints": "never", "spanNulls": True}
    if kind == "stat":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "value", "graphMode": "none", "textMode": "auto",
                        "justifyMode": "auto", "orientation": "auto"}
    return p


def row(title: str) -> dict:
    return {"type": "row", "title": title, "collapsed": False, "panels": [],
            "gridPos": {"w": 24, "h": 1}}


def q(p: float, metric: str, window: str = "5m") -> str:
    return f"histogram_quantile({p}, sum by (le) (rate({metric}_bucket[{window}])))"


def panels() -> list[dict]:
    return [
        row("Now"),
        panel("Engine up", [('up{job="qse-engine-health"}', "")], "none", kind="stat", w=4, h=4,
              desc="the public liveness page answers (1) or not (0)"),
        panel("Metrics scrape", [('up{job="qse-engine"}', "")], "none", kind="stat", w=4, h=4,
              desc="0 while the engine is up means the metrics token is wrong"),
        panel("Build", [("qse_build_info", "{{version}} {{git_sha}}")], "none", kind="stat",
              w=8, h=4, desc="version and commit; the code hash is a label"),
        panel("Uptime", [("qse_engine_uptime_seconds", "")], "s", kind="stat", w=4, h=4),
        panel("Hold on the box", [("dgx_qse_hold_active", "")], "none", kind="stat", w=4, h=4,
              desc="1 while ops/hold.sh has :8000 stopped for a measurement"),

        row("Speed"),
        panel("1. Tokens a second, served", [("rate(qse_generation_tokens_total[1m])", "tok/s")],
              "short", desc="the headline: every token written, idle time included"),
        panel("13. Decode tok/s per request",
              [(q(0.5, "qse_request_decode_tokens_per_second", "15m"), "p50"),
               (q(0.9, "qse_request_decode_tokens_per_second", "15m"), "p90")], "short",
              desc="how fast a request decoded once it was decoding (a response's "
                   "predicted_per_second)"),
        panel("2. Tokens per block",
              [("rate(qse_spec_accept_per_block_sum[5m]) / rate(qse_spec_accept_per_block_count[5m])",
                "tokens/block")], "short", decimals=2,
              desc="the number the engine is tuned on"),
        panel("3. Draft acceptance",
              [("rate(qse_spec_decode_num_accepted_tokens_total[5m]) / "
                "rate(qse_spec_decode_num_draft_tokens_total[5m])", "accepted / drafted")],
              "percentunit", desc="falls first when the drafter loses the workload"),
        panel("4. The block, split",
              [("1000 * rate(qse_verify_seconds_sum[5m]) / rate(qse_verify_seconds_count[5m])",
                "verify ms"),
               ("1000 * rate(qse_draft_seconds_sum[5m]) / rate(qse_draft_seconds_count[5m])",
                "draft ms")], "ms"),
        panel("5. Time to first token",
              [(q(0.5, "qse_time_to_first_token_seconds"), "p50"),
               (q(0.99, "qse_time_to_first_token_seconds"), "p99")], "s"),
        panel("Prefill tok/s per request",
              [(q(0.5, "qse_request_prefill_tokens_per_second", "15m"), "p50")], "short"),

        row("Load"),
        panel("6. Queue", [("qse_requests_waiting", "waiting"), ("qse_requests_running", "running")],
              "short", desc="one sequence at a time; the ninth waiting caller gets a 503"),
        panel("7. Refusals a minute", [("60 * rate(qse_requests_refused_total[5m])", "{{reason}}")],
              "short"),
        panel("8. Finish reasons",
              [("sum by (finish_reason) (rate(qse_requests_total[5m]))", "{{finish_reason}}")],
              "reqps", stack=True),
        panel("9. Errors", [("sum by (type) (rate(qse_errors_total[5m]))", "{{type}}")], "reqps"),
        panel("HTTP responses by route",
              [("sum by (route, code) (rate(qse_http_requests_total[5m]))", "{{route}} {{code}}")],
              "reqps", stack=True),
        panel("Usage ledger",
              [("3600 * rate(qse_usage_ledger_rows_total[1h])", "rows an hour"),
               ("increase(qse_usage_ledger_dropped_total[1h])", "dropped an hour")], "short"),

        row("Tokens and caches"),
        panel("10. Prompt tokens reused",
              [("rate(qse_prefill_tokens_reused_total[10m]) / (rate(qse_prefill_tokens_reused_total"
                "[10m]) + rate(qse_prefill_tokens_forwarded_total[10m]))", "reused")],
              "percentunit"),
        panel("15. Cached share of the prompt",
              [("rate(qse_prompt_tokens_cached_total[1h]) / rate(qse_prompt_tokens_total[1h])",
                "cached")], "percentunit"),
        panel("14. Reasoning share",
              [("rate(qse_reasoning_tokens_total[1h]) / rate(qse_generation_tokens_total[1h])",
                "reasoning")], "percentunit"),
        panel("11. Cache bytes against the budget",
              [("qse_cache_bytes", "{{cache}}"), ("qse_state_store_budget_bytes", "state budget")],
              "bytes"),
        panel("State store hits",
              [("sum by (kind) (rate(qse_cache_hits_total{cache=\"state\"}[15m]))", "{{kind}}"),
               ("rate(qse_cache_evictions_total{cache=\"state\"}[15m])", "evictions")], "reqps"),
        panel("Suffix store: blocks it matched",
              [("sum(rate(qse_suffix_store_drafts_total{outcome=\"hit\"}[15m])) / "
                "sum(rate(qse_suffix_store_drafts_total[15m]))", "hit share")], "percentunit"),

        row("The board"),
        panel("12. Memory",
              [("qse_gpu_memory_used_bytes", "allocated"),
               ("qse_gpu_memory_reserved_bytes", "reserved"),
               ("qse_unified_memory_free_bytes", "unified free"),
               ("qse_process_resident_memory_bytes", "process RSS")], "bytes",
              desc="allocated climbing across requests is a leak; reserved climbing alone is "
                   "fragmentation"),
        panel("GB10 temperature", [("dgx_gpu_temperature_celsius", "°C")], "celsius"),
        panel("GB10 power and SM clock",
              [("dgx_gpu_power_watts", "W"), ("dgx_gpu_sm_clock_mhz", "SM MHz")], "short"),
    ]


def layout(ps: list[dict]) -> list[dict]:
    """Assign ids and positions: rows full width, panels left to right, wrapping at 24."""
    x = y = 0
    row_h = 0
    for i, p in enumerate(ps, 1):
        p["id"] = i
        w, h = p["gridPos"]["w"], p["gridPos"]["h"]
        if p["type"] == "row" or x + w > 24:
            y += row_h
            x, row_h = 0, 0
        p["gridPos"].update(x=x, y=y)
        if p["type"] == "row":
            y += 1
            continue
        x += w
        row_h = max(row_h, h)
    return ps


def dashboard() -> dict:
    return {
        "uid": UID, "title": TITLE,
        "description": "The qwen38-spark-engine on DGX :8000 (/metrics contract 0.2.0, "
                       "server/METRICS.md): speed, load, caches, the board. A year of usage is the "
                       "engine's own dashboard at /dashboard/; Prometheus keeps 15 days.",
        "tags": ["qwen38-spark-engine", "dgx", "ai"], "timezone": "browser",
        "schemaVersion": 39, "version": 1, "editable": True, "graphTooltip": 1,
        "refresh": "30s", "time": {"from": "now-6h", "to": "now"}, "timepicker": {},
        "annotations": {"list": [
            {"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
             "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
             "name": "Annotations & Alerts", "type": "dashboard"},
            {"datasource": DS, "enable": True, "iconColor": "rgba(255, 120, 40, 1)",
             "name": "Engine restarts", "expr": "changes(qse_engine_start_time_seconds[2m]) > 0",
             "titleFormat": "engine restarted", "step": "60s"}]},
        "templating": {"list": []},
        "panels": layout(panels()),
    }


def configmap(dash: dict) -> dict:
    return {"apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": "qse-engine-dashboard", "namespace": "monitoring",
                         "labels": {"grafana_dashboard": "1",
                                    "app.kubernetes.io/part-of": "strata-alerting"},
                         "annotations": {"description": TITLE}},
            "data": {"qse-engine.json": json.dumps(dash, indent=2)}}


def main() -> int:
    dash = dashboard()
    with open(os.path.join(OUT, "qse-engine-dashboard.json"), "w") as f:
        json.dump(dash, f, indent=2)
        f.write("\n")
    with open(os.path.join(OUT, "qse-engine-dashboard-configmap.yaml"), "w") as f:
        f.write("# GENERATED by tools/grafana_dashboard.py -- edit the script, not this file.\n")
        json.dump(configmap(dash), f, indent=2)
        f.write("\n")
    print(f"{len([p for p in dash['panels'] if p['type'] != 'row'])} panels -> "
          f"ops/monitoring/qse-engine-dashboard.json, qse-engine-dashboard-configmap.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
