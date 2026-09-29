#!/usr/bin/env python3
"""sparkmon — SparkDash-in-a-terminal for the DGX Spark (GB10). v2.

Single-file, pure-stdlib curses TUI. Project:
  ~/.hermes/profiles/oracle/workspace/sparkmon/  (README/TODO/CHANGELOG/ADRs)

Shows the SparkDash lane telemetry plus a LANES panel:
  * LLM lane: decode + prefill tok/s as rolling sparkline GRAPHS + stats,
    session avg, MTP acceptance overall + per-position bars, KV cache,
    prefix hit, context info; RIGHT COLUMN (btop label-left/value-right):
    ttft/itl/e2e avgs (6s window else lifetime 'avg'), reqs running/
    waiting + completed/failed, prefill/decode tok totals, prefix hit
    % + raw queries/hits, KV usage % + fp8 badge, MTP accepted/drafted.
  * LANES: which models are actually serving — LLM (model + docker image +
    running/waiting) and decision lanes (gate checkpoint, decider version,
    router shadow), each with liveness.
  * Memory: honest GB10 unified-memory accounting from /proc/meminfo
    (NVML 'used' lies on coherent UMA) + engine GMU view.

Backends auto-detected (ADR-0003):
  * vLLM: /health non-JSON -> /metrics (Prometheus) normalized in-process.
  * EXL3 shim: /health JSON counters (the v1 path, unchanged).

Visuals per ADR-0002: btop language — box-drawn panels, solid bars
(█ fill, spaces for remainder; never ░/diamonds), block sparklines.
ASCII fallback (--ascii, auto on TERM=linux) for the bare tty1 console.

Usage:
  sparkmon                     full TUI (default URLs, lanes on)
  sparkmon -i 2                2s poll
  sparkmon --once [--sample 3] snapshot (verification without a terminal)
  sparkmon --no-lanes          LLM lane only (v1-ish)
  sparkmon --remote jaita@larryspark.local   probe a remote Spark from a Mac

Env overrides: SPARKMON_URL (default http://localhost:8000),
SPARKMON_CTX (262144), SPARKMON_MAX_SEQS (8), SPARKMON_GATE
(http://localhost:8710), SPARKMON_DECI (http://localhost:8712),
SPARKMON_ROUTER (:8711 TCP only). Launch-args precedence (v2.2): env
SPARKMON_MAX_SEQS/SPARKMON_CTX > docker-inspect captured values > defaults.
"""

from __future__ import annotations

import argparse
import curses
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime

DEFAULT_URL = os.environ.get("SPARKMON_URL", "http://localhost:8000")
DEFAULT_GATE = os.environ.get("SPARKMON_GATE", "http://localhost:8710")
DEFAULT_DECI = os.environ.get("SPARKMON_DECI", "http://localhost:8712")
ROUTER_PORT = 8711
DEFAULT_CTX = int(os.environ.get("SPARKMON_CTX", "262144"))
DEFAULT_SEQS = int(os.environ.get("SPARKMON_MAX_SEQS", "8"))  # --max-num-seqs
# Provenance/metadata parity with sparkDash recipeInfo (v2.2): the launch args
# from `docker inspect <vllm container>` ARE the authoritative values. Fetched
# ONCE at startup (probe_lanes), stored in cfg["vllm"]; env overrides win:
#   SPARKMON_MAX_SEQS / SPARKMON_CTX, then captured, then these defaults.
VLLM_DEFAULTS = {
    "max_model_len": 262144, "max_num_seqs": 8,
    "max_num_batched_tokens": None, "kv_cache_dtype": None,
    "gpu_memory_utilization": None, "spec_tokens": None,
    "reasoning_parser": None, "tool_call_parser": None,
}
BAR_WIDTH = 14
HIST_N = 48          # sparkline window (samples)
MTP_WINDOW = 6.0     # seconds for per-position acceptance window
AVG_WINDOW = 6.0     # seconds for ttft/itl/e2e window averages

# ---------------------------------------------------------------- utilities


def bar_solid(frac, width=BAR_WIDTH, ascii_mode=False):
    """btop-style bar: [████████    ] — solid fill, spaces, never diamonds."""
    frac = max(0.0, min(1.0, float(frac))) if frac is not None else 0.0
    fill = "#" if ascii_mode else "\u2588"   # █
    n = int(round(frac * width))
    return "[" + fill * n + " " * (width - n) + "]"


_SPARK = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"   # space=empty .. full
_SPARK_ASCII = "_.-~=*#"


def spark(vals, width=24, ascii_mode=False):
    """Rolling sparkline; None-valued samples render as blank.

    Idle line rule: when every sample is 0/None the line renders BLANK —
    a full bed of ▁ sliver glyphs read as "diamonds" in Menlo/mono at
    1-cell height (James's complaint); btop does the same (no baseline
    noise). Glyphs only appear once samples carry signal."""
    glyphs = _SPARK_ASCII if ascii_mode else _SPARK
    vals = list(vals)[-width:]
    if not vals:
        return " " * width
    known = [v for v in vals if v is not None]
    if not known or max(known) <= 0.0:
        return " " * width
    peak = (max(known) if known else 0.0) or 1.0
    out = []
    for v in vals:
        if v is None:
            out.append(" ")
        else:
            idx = min(len(glyphs) - 1, 1 + int((v / peak) * (len(glyphs) - 2)))
            out.append(glyphs[idx] if idx > 1 or v > 0 else " ")
    return "".join(out)


def fmt_num(v):
    if v is None:
        return "--"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e4:
        return f"{v / 1e3:.1f}K"
    return f"{v:,.0f}"


def fmt_rate(v):
    return "-- tok/s" if v is None else f"{v:,.1f} tok/s"


def _dash(v, suffix=""):
    """Compact 'value+unit or em-dash' cell text (v3.0 GPU/vitals rows)."""
    return f"{v}{suffix}" if v is not None else "\u2014"


def fmt_pct(v, digits=1):
    return "--" if v is None else f"{v * 100:.{digits}f}%"


def fmt_gib(kib):
    return f"{kib / 1024.0 / 1024.0:.1f} GiB"


def fmt_uptime(seconds):
    if seconds is None:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def read_meminfo():
    try:
        with open("/proc/meminfo", "r") as fh:
            info = {}
            for line in fh:
                parts = line.split(":")
                if len(parts) == 2:
                    info[parts[0].strip()] = int(parts[1].strip().split()[0])
        return info.get("MemTotal"), info.get("MemAvailable")
    except (OSError, ValueError, IndexError):
        return None, None


def _num(s):
    """float or None ('[N/A]' and friends on GB10's driver)."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


_THROTTLE_BITS = (
    (0x02, "idle"), (0x04, "appClks"), (0x08, "swPwrCap"),
    (0x10, "hwSlowdn"), (0x40, "swTherm"), (0x80, "hwTherm"),
    (0x20, "syncBoost"), (0x100, "hwPwrBrake"),
)


def read_hardware():
    """Local GB10 hardware snapshot: nvidia-smi only — no network, no docker.

    Missing fields stay None (render '—'); on this driver power.limit and
    memory.total report [N/A], so limit/total bars fall back to UMA total.
    """
    out = {"temp_c": None, "power_w": None, "power_limit_w": None,
           "sm_mhz": None, "sm_max_mhz": None, "throttle": None,
           "util_pct": None, "vram_used_mib": None, "vram_procs": []}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,"
             "clocks.sm,clocks.max.sm,memory.used,"
             "clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        if r.returncode == 0 and r.stdout.strip():
            v = [x.strip() for x in r.stdout.strip().split(",")]
            if len(v) >= 7:
                out["temp_c"] = _num(v[0])
                out["power_w"] = _num(v[1])
                out["util_pct"] = _num(v[2])
                out["sm_mhz"] = _num(v[3])
                out["sm_max_mhz"] = _num(v[4])
                out["vram_used_mib"] = _num(v[5])
                out["throttle"] = v[6]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0] and parts[1]:
                    out["vram_procs"].append(
                        {"pid": parts[0], "mib": _num(parts[1])})
            # memory.used flakes to [N/A] on this driver — procs sum is the
            # same reservation and matches NVML exactly when it reports.
            if out["vram_used_mib"] is None and out["vram_procs"]:
                s_mib = sum(p["mib"] or 0 for p in out["vram_procs"])
                if s_mib > 0:
                    out["vram_used_mib"] = s_mib
                    out["vram_from"] = "procs"
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def _thr_text(throttle):
    """Compact throttle tag: bitmask names or '—' when none/unknown."""
    thr = (throttle or "").strip().lower()
    try:
        val = int(thr, 0)              # accepts '0x...' hex and decimal
    except ValueError:
        return thr[:16] if thr else "—"   # driver gave text, not a bitmask
    bits = [name for bit, name in _THROTTLE_BITS if val & bit]
    if bits:
        return "ACTIVE:" + "+".join(bits)
    return "—"


def _proc_name(n):
    """sparkDash proc label -> compact display name (VLLM::Worker etc.)."""
    if not n:
        return "?"
    if n.startswith("VLLM::"):
        return n
    base = n.rsplit("/", 1)[-1]
    if base == "python" and "laya" in n:
        return "laya python"
    return base


_DASH_CACHE = {"ts": 0.0, "data": None}
DASH_METRICS_URL = "http://127.0.0.1:5555/api/sparks/spark-001/metrics"


def _dash_poll(ttl=3.0):
    """Full-but-compact sparkDash payload, cached ~3s (v3.0: ONE curl feeds
    VITALS (CPU %/temp/draw, oomRisk, UMA split) + PROCESSES per-proc VRAM;
    reuses this helper's cheap-curl pattern). Dict or None when down."""
    now = time.time()
    if now - _DASH_CACHE["ts"] < ttl:
        return _DASH_CACHE["data"]
    _DASH_CACHE["ts"] = now
    data = None
    try:
        r = subprocess.run(
            ["curl", "-sf", "-m", "2", DASH_METRICS_URL],
            capture_output=True, text=True, timeout=4)
        if r.returncode == 0 and r.stdout.strip():
            m = (json.loads(r.stdout).get("metrics") or {})
            um = m.get("unifiedMemory") or {}
            gp = m.get("gpu") or {}
            cpu = m.get("cpu") or {}
            procs = []
            for p in (gp.get("vram") or {}).get("processes") \
                    if isinstance(gp.get("vram"), dict) else \
                    (gp.get("processes") or []):
                mib = p.get("vramMB") if p.get("vramMB") is not None \
                    else p.get("vram_mb") or p.get("memMB") or 0
                if not mib:
                    continue
                procs.append({"name": _proc_name(p.get("name") or p.get("process")),
                              "pid": p.get("pid"), "mib": float(mib)})
            procs.sort(key=lambda p: -p["mib"])
            data = {
                "um_gpu": float(um["gpuUsed"]) if um.get("gpuUsed") is not None else None,
                "um_cpu": float(um["cpuUsed"]) if um.get("cpuUsed") is not None else None,
                "um_oom": um.get("oomRisk"),
                "cpu_pct": float(cpu["usage"]) if cpu.get("usage") is not None else None,
                "cpu_temp": float(cpu["temp"]) if cpu.get("temp") is not None else None,
                "cpu_draw": float(cpu["draw"]) if cpu.get("draw") is not None else None,
                "cpu_tdp": float(cpu["tdp"]) if cpu.get("tdp") is not None else None,
                "procs": procs,
            }
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError,
            TypeError):
        data = None
    _DASH_CACHE["data"] = data
    return data


def _dash_um():
    """(gpuGB, cpuGB, oomRisk) — now a thin wrapper over the cached
    _dash_poll so the whole TUI shares ONE sparkDash curl per ~3s."""
    d = _dash_poll()
    if not d:
        return None, None, None
    return d["um_gpu"], d["um_cpu"], d["um_oom"]


def read_host_uptime():
    try:
        with open("/proc/uptime", "r") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


# ------------------------------------------------------------ fetching (ADR-0003)


def fetch_any(url, timeout=2.0):
    """GET telemetry from a base URL. JSON at /health -> EXL3 dict;
    otherwise -> vLLM /metrics normalized (ADR-0003)."""
    base = url.rstrip("/")
    probe = base if base.endswith("/health") else base + "/health"
    req = urllib.request.Request(probe, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace").strip()
        if body.startswith("{"):
            return json.loads(body)
    except urllib.error.HTTPError:
        pass  # no /health route — go straight to metrics
    req2 = urllib.request.Request(base + "/metrics")
    with urllib.request.urlopen(req2, timeout=timeout) as resp2:
        return parse_vllm_metrics(resp2.read().decode("utf-8", "replace"))


_METRIC_RE = re.compile(r"^(vllm:[a-zA-Z0-9_:]+)(\{[^}]*\})?\s+([-+eE0-9.]+)\s*$")

# Prometheus histogram families sparkmon derives P95 quantiles from
# (matches sparkDash ttftP95/e2eP95/itlP95 semantics; vLLM /metrics):
_BUCKET_FAMS = {
    "time_to_first_token_seconds": "ttft",
    "e2e_request_latency_seconds": "e2e",
    "inter_token_latency_seconds": "itl",
    # deprecated alias family: fallback only — merging its rows into the
    # same list would rewind the quantile walk (kept under a separate key)
    "request_time_per_output_token_seconds": "itl_f",
}


def _hist_p95(rows):
    """P95 quantile from Prometheus _bucket rows [(cum, le)], as sparkDash
    computes it: histogram_quantile(0.95) — le edge where cumulative count
    reaches 0.95·N, linearly interpolated inside the bucket (le itself, not
    a midpoint, is the bucket's implicit upper bound). None when empty."""
    if not rows:
        return None
    pairs, n = [], 0.0
    for c, le in rows:
        try:
            c = float(c)
        except (TypeError, ValueError):
            return None
        n = max(n, c)
        try:
            le_f = float(le)
        except (TypeError, ValueError):
            continue                       # "+Inf": total-N row only
        if 0.0 <= le_f < float("inf"):
            pairs.append((le_f, c))        # sort by bucket edge, not count
    if n <= 0 or not pairs:
        return None
    pairs.sort()
    thr = 0.95 * n
    prev_le, prev_c = 0.0, 0.0
    for le_f, c in pairs:
        if c >= thr:
            span = c - prev_c
            return le_f if span <= 0 else \
                prev_le + (le_f - prev_le) * (thr - prev_c) / span
        prev_le, prev_c = le_f, c
    return None


def parse_vllm_metrics(text):
    """Normalize vLLM Prometheus /metrics into the lane dict (ADR-0003 map)."""
    mtp_pos = {}
    h = {
        "engine": "vLLM", "backend": "vllm", "context_length": DEFAULT_CTX,
        "busy": False, "is_prefilling": False,
        "requests_running": 0.0, "requests_waiting": 0.0,
        "requests_completed_total": 0.0,
        "requests_failed_total": 0.0,
        "mtp_accepted_tokens_total": None, "mtp_drafted_tokens_total": None,
        "mtp_draft_rounds_total": None,
    }
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name = m.group(1)[len("vllm:"):]
        labels = dict(re.findall(r'([a-zA-Z0-9_]+)="([^"]*)"', m.group(2) or ""))
        if labels.get("model_name") and not h.get("model_name"):
            h["model_name"] = labels["model_name"]
        val = float(m.group(3))
        if name == "num_requests_running":
            h["busy"] = h["busy"] or val > 0
            h["requests_running"] += val
        elif name == "num_requests_waiting":
            h["requests_waiting"] += val
        elif name == "generation_tokens_total":
            h["completion_tokens_total"] = val
        elif name == "prompt_tokens_total":
            h["prompt_tokens_total"] = val
        elif name == "spec_decode_num_drafts_total":
            h["mtp_draft_rounds_total"] = val
        elif name == "spec_decode_num_draft_tokens_total":
            h["mtp_drafted_tokens_total"] = val
        elif name == "spec_decode_num_accepted_tokens_total":
            h["mtp_accepted_tokens_total"] = val
        elif name == "spec_decode_num_accepted_tokens_per_pos_total":
            pos = labels.get("position")
            if pos is not None:
                mtp_pos.setdefault(int(pos), 0.0)
                mtp_pos[int(pos)] += val
        elif name == "request_success_total":
            reason = labels.get("finished_reason")
            if val > 0 and reason == "stop":
                h["requests_completed_total"] += val
            if val > 0 and reason in ("abort", "error"):
                h["requests_failed_total"] += val
        elif name == "num_preemptions_total":
            h["preemptions_total"] = h.get("preemptions_total", 0.0) + val
        elif name == "kv_cache_usage_perc":
            h["kv_cache_usage"] = val
        elif name == "prefix_cache_queries_total":
            h["prefix_queries"] = val
        elif name == "prefix_cache_hits_total":
            h["prefix_hits"] = val
        elif name.endswith("_bucket"):
            stem = name[:-len("_bucket")]
            key = _BUCKET_FAMS.get(stem)
            if key and val >= 0.0:
                le = labels.get("le")
                if le is not None:
                    h.setdefault(key, []).append((val, le))
        elif name == "cache_config_info":
            try:
                if int(float(labels.get("kv_cache_size_tokens", "0"))):
                    h["kv_capacity_tok"] = int(
                        float(labels["kv_cache_size_tokens"]))
                if labels.get("gpu_memory_utilization"):
                    h["gpu_memory_utilization"] = float(
                        labels["gpu_memory_utilization"])
                if labels.get("cache_dtype"):
                    h["kv_cache_dtype"] = labels["cache_dtype"]
            except ValueError:
                pass
        elif name.endswith(("_created", "_count", "_sum")):
            pair = {
                "time_to_first_token_seconds": "ttft",
                "time_per_output_token_seconds": "itl",
                "inter_token_latency_seconds": "itl",
                "request_time_per_output_token_seconds": "itl",
                "e2e_request_latency_seconds": "e2e",
            }
            stem = name.rsplit("_", 1)[0]
            key = pair.get(stem)
            if key:
                h[f"_{key}_{name.rsplit('_', 1)[1]}"] = val
            continue
    if mtp_pos:
        h["mtp_accept_by_position"] = [
            {"position": p, "rate": None, "tested": c}
            for p, c in sorted(mtp_pos.items())]
    pq, ph = h.get("prefix_queries"), h.get("prefix_hits")
    if pq and ph is not None:
        h["prefix_cache_hit_rate"] = ph / pq
    for key in ("ttft", "itl", "e2e"):
        if h.get(f"_{key}_count"):
            h[f"{key}_seconds_count"] = h[f"_{key}_count"]
            h[f"{key}_seconds_sum"] = h[f"_{key}_sum"]
        rows = h.get(key) or []
        if not rows and key == "itl":     # legacy engines: deprecated alias
            rows = h.get("itl_f") or []
        p95 = _hist_p95(rows)
        if p95 is not None:
            h[f"{key}_p95_seconds"] = p95
    return h


# ------------------------------------------------------------ rate math

COUNTERS = (
    "completion_tokens_total", "prompt_tokens_total",
    "mtp_accepted_tokens_total", "mtp_drafted_tokens_total",
    "mtp_draft_rounds_total",
    "requests_completed_total", "requests_failed_total",
    "prefix_queries", "prefix_hits",
    "ttft_seconds_sum", "ttft_seconds_count",
    "itl_seconds_sum", "itl_seconds_count",
    "e2e_seconds_sum", "e2e_seconds_count",
    "preemptions_total",
)

# histogram sum/count keys -> (label, fmt) for the right column
_HIST_KEYS = (
    ("ttft", "TTFT", "s"), ("itl", "ITL", "ms"), ("e2e", "e2e", "s"),
)


def _fmt_lat(v, unit):
    """Latency value for the right column: ms for itl, s for ttft/e2e."""
    if v is None:
        return "—"
    return f"{v * 1000:.1f}ms" if unit == "ms" else f"{v:.2f}s"


def compute_window_avgs(cur, prev, prev_t, now):
    """Delta-over-window avgs for ttft/itl/e2e; None when no prior sample."""
    out = {}
    if prev is None or prev_t is None or now is None:
        return out
    dt = now - prev_t
    if dt <= 0:
        return out
    for key, _, _ in _HIST_KEYS:
        s0, c0 = prev.get(f"{key}_seconds_sum"), prev.get(f"{key}_seconds_count")
        s1, c1 = cur.get(f"{key}_seconds_sum"), cur.get(f"{key}_seconds_count")
        try:
            ds = float(s1) - float(s0)
            dc = float(c1) - float(c0)
        except (TypeError, ValueError):
            continue
        if ds < 0 or dc < 0:          # engine restart / counter reset
            continue
        if dc > 0:
            out[key] = {"avg": ds / dc, "kind": f"{dt:.0f}s window"}
    return out


def compute_rates(cur, prev, prev_t, now):
    """Counter deltas between polls; silent resync on counter reset."""
    out = {"decode_tps": None, "prefill_tps": None, "prefill_src": None}
    if prev is None or prev_t is None:
        return out
    dt = now - prev_t
    if dt <= 0:
        return out
    try:
        reset = any(cur.get(c) is not None and prev.get(c) is not None
                    and cur[c] < prev[c] for c in COUNTERS)
    except TypeError:
        reset = True
    if reset:
        return out

    def _delta(c):
        try:
            return float(cur.get(c, 0)) - float(prev.get(c, 0))
        except (TypeError, ValueError):
            return 0.0

    out["decode_tps"] = _delta("completion_tokens_total") / dt
    live = cur.get("prefill_rate") or cur.get("prefill_tps_live") or 0.0
    if live:
        out["prefill_tps"] = float(live)
        out["prefill_src"] = "live"
    else:
        out["prefill_tps"] = _delta("prompt_tokens_total") / dt
        out["prefill_src"] = "delta"
    return out


class MTPWindow:
    """Rolling acceptance window so per-position rates are live, not lifetime."""

    def __init__(self, k=3, seconds=MTP_WINDOW):
        self.k = k
        self.seconds = seconds
        self.last = None      # (t, rounds, accepted, {pos: count})
        self.frame = {}

    def update(self, h):
        r = h.get("mtp_draft_rounds_total")
        a = h.get("mtp_accepted_tokens_total")
        pos = {p["position"]: p["tested"]
               for p in (h.get("mtp_accept_by_position") or [])}
        if r is None or a is None:
            self.frame = {}
            return self.frame
        now = time.time()
        if self.last is None:
            self.last = (now, r, a, pos)
            self.frame = {}
            return self.frame
        t0, r0, a0, pos0 = self.last
        dt = now - t0
        dr, da = r - r0, a - a0
        if dr < 0 or da < 0 or dt <= 0:      # engine restart / counter reset
            self.last = (now, r, a, pos)
            self.frame = {}
            return self.frame
        self.frame = {
            "overall": (da / max(1.0, dr * self.k)) if dr else None,
            "per_pos": {p: (pos.get(p, 0) - pos0.get(p, 0)) / dr
                        for p in pos} if dr else {},
            "window_s": round(dt, 1),
        }
        if dt >= self.seconds or not self.frame["per_pos"]:
            self.last = (now, r, a, pos)
        return self.frame


# ------------------------------------------------------------ lanes probing


def _tcp_up(port, host="127.0.0.1", t=0.4):
    try:
        with socket.create_connection((host, port), timeout=t):
            return True
    except OSError:
        return False


def _http_json(url, t=1.2):
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=t) as resp:
            body = resp.read().decode("utf-8", "replace").strip()
        if body.startswith("{"):
            return json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        pass
    return None


_docker_image_cache = {"ts": 0.0, "value": None}


def _docker_serving_image():
    """(container, image) for the vLLM serving container, cached 30s."""
    now = time.time()
    if now - _docker_image_cache["ts"] < 30:
        return _docker_image_cache["value"]
    val = None
    try:
        out = subprocess.run(
            ["docker", "ps", "--format",
             "{{.Names}}|{{.Image}}|{{.Status}}"],
            capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            name, image, status = (line.split("|") + ["", "", ""])[:3]
            if "vllm" in name.lower() or "vllm" in image.lower():
                short = image.split("/")[-1]
                val = {"container": name, "image": short,
                       "up": "Up" in status}
                break
    except (OSError, subprocess.SubprocessError):
        val = None
    _docker_image_cache.update(ts=now, value=val)
    return val


def _docker_launch_args(container):
    """Parsed vLLM launch args from docker inspect — the authoritative
    provenance (ADR-0004). One call at startup/lanes-refresh; never per-frame."""
    try:
        out = subprocess.run(
            ["docker", "inspect", container, "--format", "{{json .Args}}"],
            capture_output=True, text=True, timeout=5).stdout
        argv = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    flags, cfg2 = {}, {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("--") or "=" in a:
            i += 1
            continue
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if nxt is not None and not nxt.startswith("--"):
            flags[a[2:]] = nxt
            i += 2
        else:                      # boolean flag (e.g. --enable-chunked-prefill)
            flags[a[2:]] = True
            i += 1
    if "max-model-len" in flags:
        try:
            cfg2["max_model_len"] = int(flags["max-model-len"])
        except ValueError:
            pass
    if "max-num-seqs" in flags:
        try:
            cfg2["max_num_seqs"] = int(flags["max-num-seqs"])
        except ValueError:
            pass
    if "max-num-batched-tokens" in flags:
        try:
            cfg2["max_num_batched_tokens"] = int(
                flags["max-num-batched-tokens"])
        except ValueError:
            pass
    if "gpu-memory-utilization" in flags:
        try:
            cfg2["gpu_memory_utilization"] = float(
                flags["gpu-memory-utilization"])
        except ValueError:
            pass
    if "kv-cache-dtype" in flags:
        cfg2["kv_cache_dtype"] = flags["kv-cache-dtype"]
    if "tensor-parallel-size" in flags:      # sparkDash recipeInfo.maxLanes
        try:
            cfg2["max_lanes"] = int(flags["tensor-parallel-size"])
        except ValueError:
            pass
    if "reasoning-parser" in flags:
        cfg2["reasoning_parser"] = flags["reasoning-parser"]
    if "tool-call-parser" in flags:
        cfg2["tool_call_parser"] = flags["tool-call-parser"]
    sc = flags.get("speculative-config")
    if isinstance(sc, str) and sc.startswith("{"):
        try:
            scj = json.loads(sc)
            st = scj.get("num_speculative_tokens")
            if st is not None:
                cfg2["spec_tokens"] = int(st)
            m = scj.get("method")
            if m:
                cfg2["spec_method"] = str(m).upper()
        except ValueError:
            pass
    return cfg2


def _posture_open():
    """sparkDash posture parity (posture.auth/scope/label): 'Open · Local'
    when the sparkDash probe is unauthenticated (authMode required-missing);
    sparkmon probes the loopback target, matching dash scope=local. None
    when dash is unreachable or auth is enforced — badge omitted, not faked."""
    h = _http_json("http://127.0.0.1:5555/api/health", t=1.0)
    if isinstance(h, dict) and h.get("authMode") == "required-missing":
        return {"auth": "open", "scope": "local", "label": "Open · Local"}
    return None


def probe_lanes(cfg):
    """Lanes panel data: liveness for all, rich health where cheap."""
    lanes = {}
    llm_up = _tcp_up(8000)
    lanes["llm"] = {
        "up": llm_up, "port": 8000,
        "model": cfg.get("llm_model") or "vLLM",
        "detail": None,
    }
    docker = _docker_serving_image() if llm_up else None
    if docker:
        lanes["llm"]["image"] = f"{docker['container']} \u00b7 {docker['image']}"
        if not cfg["vllm"].get("captured"):        # capture once at startup
            cap = _docker_launch_args(docker["container"])
            if cap:
                merged = dict(cfg["vllm"])         # keep defaults/quantization
                merged.update(cap)
                merged["captured"] = True
                # precedence: env SPARKMON_MAX_SEQS/SPARKMON_CTX WIN over
                # docker-inspect captured values (docstring + ADR-0004)
                cfg["vllm_src"] = {}
                for k2, env_var in (("max_num_seqs", "SPARKMON_MAX_SEQS"),
                                    ("max_model_len", "SPARKMON_CTX")):
                    if env_var in os.environ:
                        merged.pop(k2, None)       # keep env/default value
                        cfg["vllm_src"][k2] = "env"
                    elif merged.get(k2) is not None:
                        cfg["vllm_src"][k2] = "docker"
                for k2 in ("max_num_batched_tokens", "kv_cache_dtype",
                           "gpu_memory_utilization", "spec_tokens",
                           "reasoning_parser", "tool_call_parser",
                           "max_lanes"):
                    if merged.get(k2) is not None:
                        cfg["vllm_src"][k2] = "docker"
                cfg["vllm"] = merged
                if "NVFP4" in (lanes["llm"].get("model") or "").upper():
                    cfg["quantization"] = "NVFP4"
    elif llm_up and cfg.get("remote"):
        lanes["llm"]["image"] = "docker n/a (remote probe)"
    # sparkDash posture: auth/scope parity (posture.* fields). Label string
    # when unauthenticated; None when dash down or auth enforced (no fake).
    po = _posture_open()
    if po:
        lanes["llm"]["posture"] = po["label"]
    gate = _http_json(cfg["gate_url"] + "/health")
    lanes["gate"] = {
        "up": gate is not None or _tcp_up(8710), "port": 8710,
        "model": (gate or {}).get("model") or "deciserv gate",
        "detail": f'{(gate or {}).get("precision", "?")}'
                  f'  load {(gate or {}).get("load_s", "--")}s',
    }
    deci = _http_json(cfg["deci_url"] + "/healthz")
    if deci is None:
        deci = _http_json(cfg["deci_url"] + "/health")
    # pool_used_gb is UMA-polluted on GB10 (torch mem_get_info sees the whole
    # pool) — show it only when it looks like a real allocation (<100GB).
    pool = (deci or {}).get("pool_used_gb")
    pool_txt = f"  pool {pool}GB" if pool and pool < 100 else ""
    lanes["deci"] = {
        "up": deci is not None or _tcp_up(8712), "port": 8712,
        "model": (deci or {}).get("model") or "decider-2b",
        "detail": f'{(deci or {}).get("precision", "?")}{pool_txt}',
    }
    lanes["router"] = {
        "up": _tcp_up(ROUTER_PORT), "port": ROUTER_PORT,
        "model": "intent-router (shadow)", "detail": "CPU-only",
    }
    # sparkDash web dashboard (viewer, not a lane) — shown for provenance
    dash = _http_json("http://127.0.0.1:5555/api/health")
    lanes["dash"] = {
        "up": dash is not None or _tcp_up(5555), "port": 5555,
        "model": "sparkDash (web UI)", "detail": "docker · host-net",
    }
    return lanes


# ------------------------------------------------------------ rendering
#
# v3.0 frame (Hailey's approved redesign, task t_14bc01ee):
#   HEADER -> VITALS strip -> full-width THROUGHPUT -> MEMORY || PROCESSES
#   -> LLM LANE (3 fixed-gutter columns) -> LANES (right-aligned posture
#   badge + ╰ recipe footnote) -> single-row GPU panel -> footer.
# JAMES'S COLOUR/BAR DIRECTIVE (binding, applies to every panel): colour is
#   a state channel, not decoration - every styled token exercised, every
#   bar in the bar_solid() solid-█ contract (never ░/diamonds), thresholds
#   noted per panel; --ascii keeps bracket/state badges so semantics
#   survive without colour. Palette assumes a dark background (btop does).

S = {"plain": None, "title": "title", "dim": "dim", "good": "good",
     "warn": "warn", "bad": "bad", "accent": "accent", "badge": "badge",
     "rev_bad": "rev_bad", "rev_good": "rev_good", "rev_acc": "rev_acc"}


def seg(text, style="plain"):
    return (text, S.get(style, None))


def box_top(title, width, ascii_mode):
    bar = "-" if ascii_mode else "\u2500"
    if not ascii_mode:
        return [seg("\u250c\u2500 " + title + " ", "title"),
                seg("\u2500" * max(3, width - len(title) - 5) + "\u2510",
                    "dim")]
    # ascii: '+-- TITLE --… --+' — title INSIDE the bar (matches unicode)
    dash_len = max(3, width - len(title) - 6)
    return [seg("+" + "-- " + title + " " + "-" * dash_len + "+", "dim")]


def box_bottom(width, ascii_mode):
    return [seg("+" + ("-" * max(3, width - 2)) + "+" if ascii_mode
                else "\u2514" + "\u2500" * max(3, width - 2) + "\u2518", "dim")]


def _rcol_row(label, value, style="plain", tail=None, lab_w=8):
    """Aligned cell (P0-2/P1-2): fixed label pad + value + dim tail."""
    row = [seg(f"{label:<{lab_w}} "), seg(value, style)]
    if tail:
        row.append(seg(f" {tail}", "dim"))
    return row


def _row_text(row):
    return "".join(t for t, _ in row)


def _dq_valid(dq):
    return [v for v in dq if v is not None]


def _dq_mean(dq):
    v = _dq_valid(dq)
    return sum(v) / len(v) if v else None


def _dq_max(dq):
    v = _dq_valid(dq)
    return max(v) if v else None


def _dq_wmean(dq, interval):
    """10s window mean from the deque tail (sample rate = poll interval)."""
    n = max(1, int(round(10.0 / max(interval, 0.1))))
    v = _dq_valid(dq)[-n:]
    return sum(v) / len(v) if v else None


def _proc_name(n):
    """sparkDash proc label -> compact display name (VLLM::Worker etc.)."""
    if not n:
        return "?"
    if n.startswith("VLLM::"):
        return n
    base = n.rsplit("/", 1)[-1]
    if base == "python" and "laya" in n:
        return "laya python"
    return base


def _oom_style(risk):
    """oomRisk badge style: HIGH=red-reverse, mod*=warn, low=green-reverse."""
    r = (risk or "").lower()
    if not r:
        return "dim"
    if "high" in r:
        return "rev_bad"
    if "mod" in r or "risk" in r:
        return "warn"
    return "rev_good" if r == "low" else "good"


def _thr_pct_style(frac):
    """VITALS utilisation cells (RAM/CPU): warn >=0.6, bad >=0.8 (steer)."""
    return "good" if frac < 0.6 else ("warn" if frac < 0.8 else "bad")


def _mem_style(frac):
    """Big memory bars (system/UMA/VRAM): 0.7/0.9 thresholds (deliberate,
    coarse memory bars stay calm longer than VITALS alert cells)."""
    return "good" if frac < 0.7 else ("warn" if frac < 0.9 else "bad")


def _temp_style(t):
    if t is None:
        return "dim"
    return "bad" if t >= 90 else ("warn" if t >= 80 else "good")


def build_vitals_rows(st, width, ascii_mode):
    """VITALS as TWO rows (dec/pre | RAM/CPU/OOM) so it pairs with the
    THROUGHPUT box (2 sparkline rows) side-by-side — every band is now
    two columns; heights match, borders share one grid."""
    r = st.get("rates") or {}
    avg = st.get("avg") or {}
    dec, pre = r.get("decode_tps"), r.get("prefill_tps")
    dec_avg, pre_avg = avg.get("decode"), avg.get("prefill")
    compact = width < 200
    bar_w = 6 if compact else 10
    insep = " \u2502 " if compact else " \u00b7 "
    sep = seg(" \u2502 ", "dim")

    def _num1(v, nd=1):
        return "--" if v is None else f"{v:,.{nd}f}"

    cells = []
    dec_txt = "--" if dec is None else f"{_num1(dec)}{'t/s' if compact else ' tok/s'}"
    c = [seg("dec ", "dim"), seg(dec_txt, "good" if dec else "dim")]
    if dec_avg is not None:
        c.append(seg(f"{insep}avg {_num1(dec_avg)}", "dim"))
    cells.append(c)
    pre_txt = "--" if pre is None else f"{_num1(pre)}{'t/s' if compact else ' tok/s'}"
    c = [seg("pre ", "dim"), seg(pre_txt, "accent" if pre else "dim")]
    if pre_avg is not None:
        c.append(seg(f"{insep}avg {_num1(pre_avg, 0)}", "dim"))
    cells.append(c)
    # fixed cell anchors so VITALS row1 (dec/pre) and row2 (RAM/CPU/OOM)
    # share the same columns: cell1@3, cell2@31, cell3@59
    def _at(cells_list, xs=(3, 31, 59)):
        out_c, x = [], 0
        for k_i, cc in enumerate(cells_list):
            x_t = xs[k_i] if k_i < len(xs) else x + 2
            txt = _row_text(cc)
            if x_t > x:
                out_c.append(seg(" " * (x_t - x)))
                x = x_t
            elif x > x_t:
                pass
            out_c += cc
            x += len(txt)
        return out_c

    row_rate = _at(cells)

    cells2 = []
    total, avail = st.get("mem_total"), st.get("mem_avail")
    if total and avail:
        frac = (total - avail) / total
        stl = _thr_pct_style(frac)
        cells2.append([seg(f"RAM {frac * 100:.0f}%{insep if compact else ' '}", stl),
                       seg(bar_solid(frac, width=bar_w, ascii_mode=ascii_mode), stl)])
    cpu = st.get("cpu_pct")
    if cpu is not None:
        stl2 = _thr_pct_style(cpu / 100.0)
        cells2.append([seg(f"CPU {cpu:.0f}%{insep if compact else ' '}", stl2),
                       seg(bar_solid(cpu / 100.0, width=bar_w,
                                     ascii_mode=ascii_mode), stl2)])
    oom = st.get("um_oom")
    if oom:
        cells2.append([seg("OOM ", "dim"),
                       seg(f"[{oom.upper()}]", _oom_style(oom))])
    else:
        cells2.append([seg("OOM \u2014", "dim")])
    row_res = _at(cells2)
    return [row_rate, row_res]


def build_throughput_rows(st, width, ascii_mode):
    """THROUGHPUT panel: decode+prefill sparklines (48 wide / 30 compact),
    rate cell coloured per steer, instant|10s|avg|peak all from deques."""
    r = st.get("rates") or {}
    avg = st.get("avg") or {}
    interval = st.get("interval") or 1.0
    compact = width < 200

    def _row(label, key, hist, rate, avg_v, label_style, rate_style):
        sw = 30 if compact else 48
        dec10 = _dq_wmean(hist, interval)
        peak = _dq_max(hist)
        inst = "--" if rate is None else f"{rate:,.1f}"
        stats = []
        if compact:
            stats.append(f"avg {_dq_mean(hist) if avg_v is None else avg_v:,.1f}"
                         if (avg_v is not None or _dq_mean(hist) is not None)
                         else "avg --")
            if peak is not None:
                stats.append(f"pk {fmt_num(peak)}")
        else:
            if dec10 is not None:
                stats.append(f"10s {dec10:,.1f}")
            if avg_v is not None:
                stats.append(f"avg {avg_v:,.1f}")
            elif _dq_mean(hist) is not None:
                stats.append(f"avg {_dq_mean(hist):,.1f}")
            if peak is not None:
                stats.append(f"peak {fmt_num(peak)}")
        # stats column snaps to the shared x=78 anchor (column parity with
        # LLM LANE mid col + PROCESSES left edge — one grid, top to bottom)
        head = [seg(f"  {label} "),
                seg(spark(hist, width=sw, ascii_mode=ascii_mode),
                    label_style),
                seg("  "),
                seg(f"{inst} {'t/s' if compact else 'tok/s'}",
                    rate_style if rate else "dim")]
        used = sum(len(t) for t, _ in head)
        pad = max(1, 78 - used)
        row = head + [seg(" " * pad), seg(" \u2502 ".join(stats), "dim")]
        return row

    return [
        _row("dec", "decode", st["hist"].get("decode") or [],
             r.get("decode_tps"), avg.get("decode"), "good", "good"),
        _row("pre", "prefill", st["hist"].get("prefill") or [],
             r.get("prefill_tps"), avg.get("prefill"), "accent", "accent"),
    ]


def _llm_left_rows(st, h, bar_w, ascii_mode):
    """LLM LANE column 1: kv / prefix / mtp / mtp-pos bars (18 wide/12)."""
    win = st.get("mtp_win") or {}
    spec_k = (st.get("cfg") or {}).get("vllm", {}).get("spec_tokens") or 3
    rows = []

    # kv cache: usage% bar + state colour (0.6/0.85), capacity + fp8 badge
    kv = h.get("kv_cache_usage")
    kvc = h.get("kv_capacity_tok")
    kv_lbl = "kv" if bar_w < 14 else "kv cache"
    stl = "good" if (kv or 0) < 0.6 else ("warn" if (kv or 0) < 0.85 else "bad")
    row = [seg(f"{kv_lbl:<9} "),
           seg(bar_solid(kv, width=bar_w, ascii_mode=ascii_mode)
               if kv is not None else "[" + " " * bar_w + "]", stl),
           seg(" "), seg(fmt_pct(kv), stl if kv else "plain")]
    if kvc:
        row.append(seg(f" {fmt_num(kvc)} tok", "dim"))
    dtype = h.get("kv_cache_dtype")
    if dtype and dtype != "auto":
        row.append(seg(f" [fp8]" if bar_w < 14 else f" [kv {dtype}]",
                       "accent" if dtype == "fp8" else "dim"))
    rows.append(row)

    # prefix cache: hit-rate bar (cyan family), queries/hits tail
    pfx = h.get("prefix_cache_hit_rate")
    pq, ph = h.get("prefix_queries"), h.get("prefix_hits")
    row = [seg(f"{'prefix' if bar_w >= 14 else 'pfx':<9} "),
           seg(bar_solid(pfx, width=bar_w, ascii_mode=ascii_mode)
               if pfx is not None else "[" + " " * bar_w + "]", "accent"),
           seg(" "), seg(fmt_pct(pfx), "accent" if pfx else "dim")]
    if pq is not None:
        row.append(seg(f" {fmt_num(pq)}q/{fmt_num(ph or 0)}h", "dim"))
    rows.append(row)

    # MTP: acceptance bar + live window + accepted/drafted tokens
    acc = h.get("mtp_accepted_tokens_total")
    drf = h.get("mtp_drafted_tokens_total")
    overall = (acc / drf) if (acc is not None and drf) else None
    w_over = win.get("overall")
    mst = ("good" if (w_over or overall or 0) >= 0.5
           else ("warn" if (w_over or overall or 0) >= 0.3 else "dim"))
    row = [seg(f"{'mtp' if bar_w >= 14 else 'mtp':<9} "),
           seg(bar_solid(overall, width=bar_w, ascii_mode=ascii_mode)
               if overall is not None else "[" + " " * bar_w + "]", mst),
           seg(" "), seg(fmt_pct(overall), "plain")]
    if w_over is not None:
        row.append(seg(f" live {fmt_pct(w_over, 0)}"
                       f" ({win.get('window_s', '--')}s)", "dim"))
    if acc is not None:
        row.append(seg(f" {fmt_num(acc)}/{fmt_num(drf)} tok"
                       + (f" \u00b7 k={spec_k}" if bar_w < 14 else ""), "dim"))
    else:
        row.append(seg(f" k={spec_k}", "dim"))
    rows.append(row)

    # per-position, ONE shared row (live window preferred, lifetime fallback)
    # - always rendered so the panel never gains/loses a line (P3-2 fix)
    pp = win.get("per_pos") or {}
    cells = []
    if pp:
        for p in sorted(pp)[:6]:
            v = pp[p]
            cells.append(f"p{p} {bar_solid(v, width=8, ascii_mode=ascii_mode)}"
                         f" {fmt_pct(v, 0)}")
        tail = f"k={spec_k} (live window)"
        style = "plain"
    else:
        pos_h = h.get("mtp_accept_by_position") or []
        mx = max((p.get("tested") or 0) for p in pos_h[:6]) or 1
        for p in pos_h[:6]:
            b = bar_solid((p.get("tested") or 0) / mx, width=4,
                          ascii_mode=ascii_mode)
            cells.append(f"p{p.get('position')} {b}"
                         f" {fmt_num(p.get('tested'))}")
        tail = (f"k={spec_k} (lifetime)" if bar_w < 14
                else f"k={spec_k} (lifetime \u2014 window warming up)")
        style = "dim"
    pos_row = [seg(f"{'mtp pos' if bar_w >= 14 else 'mpos':<9} "),
               seg(" ".join(cells) if cells else "\u2014 no by-position data",
                   style)]
    rows.append(pos_row)
    return rows


def _llm_mid_rows(st, h):
    """LLM LANE column 2: ttft / itl / e2e mean + p95 (p95 cyan)."""
    w_avgs = st.get("win_avgs") or {}
    out = []

    def lat_avg(key):
        wv = w_avgs.get(key)
        if wv:
            return wv["avg"], wv["kind"]
        c = h.get(f"{key}_seconds_count")
        s = h.get(f"{key}_seconds_sum")
        if c and s is not None:
            return s / c, "avg"
        return None, "avg"

    for key, label, unit in _HIST_KEYS:
        v, kind = lat_avg(key)
        lab = f"{label.lower()}  "
        if v is None:
            out.append(_rcol_row(lab, "\u2014", "dim", lab_w=5))
            continue
        row = _rcol_row(lab, _fmt_lat(v, unit), "plain", lab_w=5)
        row.append(seg(" (6s)" if kind.startswith("6s") or
                       "window" in kind else f" ({kind})", "dim"))
        p95 = h.get(f"{key}_p95_seconds")
        if p95 is not None:
            row.append(seg(f" p95 {_fmt_lat(p95, unit)}", "accent"))
        out.append(row)
    return out


def _llm_right_rows(st, h, cfg):
    """LLM LANE column 3: reqs / seqs / preempt / eng tok (+ source tags)."""
    rows = []
    rr = h.get("requests_running") or 0
    rw = h.get("requests_waiting") or 0
    tail = (f"{fmt_num(h.get('requests_completed_total'))}d/"
            f"{fmt_num(h.get('requests_failed_total'))}f"
            + (" QUEUE" if rw else ""))
    rows.append(_rcol_row("reqs", f"{rr:.0f}r/{rw:.0f}w", "plain", tail))
    seqs_src = cfg.get("vllm_src", {}).get("max_num_seqs", "default")
    run = h.get("requests_running") or 0
    seqs = cfg.get("vllm", {}).get("max_num_seqs") or DEFAULT_SEQS
    ctx_src = cfg.get("vllm_src", {}).get("max_model_len", "default")
    cl = cfg.get("vllm", {}).get("max_model_len") or \
        h.get("context_length") or DEFAULT_CTX
    rows.append(_rcol_row("seqs", f"{run:.0f}/{seqs:.0f}", "plain",
                          f"\u00b7 ctx {cl / 1000:.1f}K ({ctx_src})"))
    mbt = (cfg.get("vllm") or {}).get("max_num_batched_tokens")
    if mbt and cfg.get("vllm_src", {}).get("max_num_batched_tokens") == "docker":
        rows[-1].append(seg(f" \u00b7 batched {fmt_num(mbt)}", "dim"))
    pre = h.get("preemptions_total")
    stl = "good" if pre == 0 else ("plain" if pre is None else "warn")
    tail2 = (f"\u00b7 eng {fmt_num(h.get('prompt_tokens_total'))}/"
             f"{fmt_num(h.get('completion_tokens_total'))}")
    rows.append(_rcol_row("preempt",
                          f"{pre:.0f}" if pre is not None else "\u2014",
                          stl, tail2))
    return rows


def _merge_columns(rows_per_col, gutters, width):
    """Zip columns of rows into one band.

    Columns 0..n-2 sit LEFT-ANCHORED at their gutter x; the FINAL column
    is RIGHT-ANCHORED — its label block ends 2 cols inside the right
    border. Every row reads as bars | values | right rail: no long dead
    tail after the last column, no mid-row gaps that read as breakage.
    Content longer than its span sheds dim tails (values never)."""
    out = []
    single = any(g is None for g in gutters)
    if single:
        flat = [r for gi, col in enumerate(rows_per_col)
                if gutters is not None for r in col]
        return [list(r) for r in flat if isinstance(r, list)]
    nrows = max((len(c) for c in rows_per_col), default=0)
    xs = list(gutters)
    last_ci = len(rows_per_col) - 1

    def _shed(cell, span):
        c = [t for t in cell
             if isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], str)]
        while c and sum(len(t) for t, _ in c) > span:
            c.pop()
        return c

    for i in range(nrows):
        row, x = [], 0
        for ci, col in enumerate(rows_per_col):
            cell = col[i] if i < len(col) else []
            if ci < last_ci:
                nxt = xs[ci + 1]
                ccell = _shed(cell, nxt - xs[ci] - 2)
                text = _row_text(ccell)
                row += ccell
                x += len(text)
                if x < xs[ci + 1]:
                    row.append(seg(" " * (xs[ci + 1] - x)))
                    x = xs[ci + 1]
            else:
                span = (width - 3) - xs[ci]
                ccell = _shed(cell, span)
                text = _row_text(ccell)
                lead = max(0, span - len(text))
                if lead:
                    row.append(seg(" " * lead))
                row += ccell
                x += lead + len(text)
        out.append(row)
    return out


def build_mem_proc(st, cfg, width, ascii_mode):
    """MEMORY panel rows, COMPACT variant (per A.2): system/cpu, uma+OOM,
    procs one-liner. Wide mode uses _split_memory + _proc_rows side-by-side."""
    h = st.get("health") or {}
    total, avail = st.get("mem_total"), st.get("mem_avail")
    mem_rows = []

    if total and avail:
        used = total - avail
        frac = used / total
        stl = _thr_pct_style(frac)
        mem_rows.append(
            [seg("system "),
             seg(bar_solid(frac, width=14, ascii_mode=ascii_mode), stl),
             seg(f" {frac * 100:.1f}%", stl),
             seg(f"  {fmt_gib(used)} / {fmt_gib(total)}", "dim")])
    else:
        mem_rows.append([seg("system   meminfo unavailable", "dim")])
    gmu = h.get("gpu_memory_utilization")
    if gmu is not None:
        stl = _mem_style(gmu)
        mem_rows.append(
            [seg("engine "),
             seg(bar_solid(gmu, width=14, ascii_mode=ascii_mode), stl),
             seg(f" {gmu * 100:.1f}%", stl),
             seg("   gpu_memory_utilization (engine view)", "dim")])
    else:
        mem_rows.append([seg("engine   \u2014", "dim")])
    gpuu = st.get("um_gpu")
    if gpuu is not None and total:
        cf = gpuu / (total / 1024.0)          # sparkDash unifiedMemory = MiB
        stl = _mem_style(cf)
        uma = [seg("uma gpu "),
               seg(bar_solid(cf, width=14, ascii_mode=ascii_mode), stl),
               seg(f" {fmt_gib(gpuu * 1024)}", stl)]
        uma.append(seg(f"  split cpu {fmt_gib((st.get('um_cpu') or 0) * 1024)}"
                       "  (d)", "dim"))
        mem_rows.append(uma)
    else:
        mem_rows.append([seg("uma gpu  \u2014", "dim")])
    # compact folds: oomRisk badge row (P0-3 style) + procs one-liner
    oom = st.get("um_oom")
    mem_rows.append(
        [seg("oom   ", "dim"),
         seg(f"[{oom.upper()}]" if oom else "\u2014",
             _oom_style(oom) if oom else "dim"),
         seg("   oomRisk (sparkDash)" if oom
             else "\u2014 oomRisk unknown (sparkDash offline)", "dim")])
    procs = st.get("procs") or []
    if procs:
        cells = " \u00b7 ".join(
            f"{p['name']} {fmt_gib(p['mib'] * 1024)}" for p in procs[:4])
        s_mib = sum(p["mib"] for p in procs)
        mem_rows.append([seg("procs ", "dim"),
                         seg(cells, "plain"),
                         seg(f"  \u03a3 {fmt_gib(s_mib * 1024)} (d)", "dim")])
    else:
        mem_rows.append([seg("procs  (sparkDash offline \u2014 no per-proc data)",
                             "dim")])
    return mem_rows


def build_gpu_row(st, width, ascii_mode):
    """Single-row GPU panel: temp · watts · SM · util · throttle."""
    hw = st.get("hw") or {}
    under_load = (hw.get("util_pct") or 0) > 5
    dash = _dash(hw.get("temp_c"), "C")
    power = (f"{hw.get('power_w'):g}W" if hw.get("power_w") is not None
             else "\u2014")
    row = [seg(f"  {dash} \u00b7 {power}", _temp_style(hw.get("temp_c"))),
           seg(f" \u00b7 SM {_dash(hw.get('sm_mhz'), 'MHz')}", "dim")]
    if hw.get("sm_mhz") is not None and hw.get("sm_max_mhz"):
        row.append(seg(f" ({hw['sm_mhz'] / hw['sm_max_mhz'] * 100:.0f}%)", "dim"))
    if hw.get("util_pct") is not None:
        row.append(seg(f" \u00b7 util {hw['util_pct']:.0f}%",
                       "accent" if under_load else "dim"))
    thr_txt = _thr_text(hw.get("throttle"))
    row.append(seg(f" \u00b7 thr {thr_txt}",
                   "bad" if thr_txt.startswith("ACTIVE:") else "dim"))
    row.append(seg(f"   detail: btop \u00b7 sparkDash"
                   + ("  (idle)" if not under_load else ""), "dim"))
        # right half: system quick stats — wide standalone GPU panel only
    if width >= 200:
        total, avail = st.get("mem_total"), st.get("mem_avail")
        oom = st.get("um_oom")
        nproc = len(st.get("procs") or [])
        quick = []
        if total and avail:
            frac = (total - avail) / total
            quick.append(f"RAM {frac * 100:.0f}%")
        if oom:
            quick.append(f"OOM {oom.upper()}")
        if nproc:
            quick.append(f"{nproc} proc")
        if quick:
            txt = " · ".join(quick)
            pad = width - 3 - len(_row_text(row)) - len(txt)
            if pad >= 2:
                row.append(seg(" " * pad, "dim"))
                row.append(seg(txt,
                               _oom_style(oom) if oom == "high" else "dim"))
    return row


def build_lanes_rows(st, width, cfg, ascii_mode):
    """LANES panel: 5 lane rows + right-aligned posture + ╰ recipe footnote."""
    lanes = st.get("lanes") or {}
    h = st.get("health") or {}
    eng = h.get("engine", "?")
    model = h.get("model_name") or eng
    quant = cfg.get("quantization")
    rows = []
    posture = (lanes.get("llm") or {}).get("posture")
    badge = f"[{posture}]" if posture else ""
    for key, label, lcol in (("llm", "LLM", "good"), ("gate", "gate", "good"),
                             ("deci", "deci", "good"), ("router", "route", "dim"),
                             ("dash", "dash", "dim")):
        ln = lanes.get(key) or {}
        up = ln.get("up")
        dot, dstyle = (("\u25cf", "good") if up else ("\u25cb", "bad")) \
            if not ascii_mode else (("*" if up else "x"),
                                    "good" if up else "bad")
        row = [seg(f"  {label:<5} "), seg(dot, dstyle),
               seg(f" :{ln.get('port', '--')}", "dim"),
               seg(f"  {ln.get('model') or '?'}",
                   "plain" if up else "dim")]
        if ln.get("image"):
            row.append(seg(f"  [{ln['image']}]", "dim"))
        detail = ln.get("detail") if up else None
        if detail:
            # right-align the detail tag at the panel's right edge — the
            # right half of LANES stops being dead space (P1-4 extension)
            pad = width - 5 - len(_row_text(row)) - len(detail)
            if pad >= 1:
                row.append(seg(" " * pad, "dim"))
            row.append(seg(detail, "dim"))
        rows.append(row)
        if key == "llm" and posture:
            # right-align the posture badge inside the box (P1-4)
            pad = width - 3 - len(_row_text(row)) - len(badge) - 2
            if pad >= 1:
                row.append(seg(" " * pad))
            row.append(seg(badge, "rev_acc" if not ascii_mode else "plain"))

    # recipe footnote: merged recipe row (rs/tc/spec/quant/lanes) + author /
    # engine provenance, '╰' marker, indent 2 (P1-3). Recipe row content
    # moved here from the LLM-lane right column (its natural home).
    vllm_cfg = cfg.get("vllm") or {}
    parts = []
    author = cfg.get("recipe_author")
    if cfg.get("recipe_model"):
        parts.append(cfg["recipe_model"])
    if author:
        parts.append(f"author {author}")
    parts.append(f"engine {eng}")
    if quant:
        parts.append(quant)
    rp = vllm_cfg.get("reasoning_parser")
    tp = vllm_cfg.get("tool_call_parser")
    sk = vllm_cfg.get("spec_tokens")
    if rp:
        parts.append(f"rs {rp}")
    if tp:
        parts.append(f"tc {tp}")
    parts.append(f"spec {vllm_cfg.get('spec_method') or 'MTP'}"
                 + (f" k={sk}" if sk else ""))
    ml = vllm_cfg.get("max_lanes")
    if ml:
        parts.append(f"lanes {ml}")
    rows.append([seg("  \u2570 recipe: " + " \u00b7 ".join(parts), "dim")])
    return rows





def _fmt_t(t):
    return f"{t:g}°C " if t is not None else "—  "


def _fmt_p(p):
    return f"{p:g}W  " if p is not None else "—  "


def _gpu_zone_bar(v, soft, hot, width=8, ascii_mode=False):
    """Zone-coloured bracket bar: green below soft, yellow to hot, red
    above hot. v can be absolute temp (with soft/hot thresholds) or a
    frac (soft/hot as fractions)."""
    br = "]" if not ascii_mode else "]"
    if v is None:
        return "[" + " " * (width - 1) + br
    frac = max(0.0, min(1.0, v / 100.0)) if v > 1.5 and soft > 1.5 else \
        max(0.0, min(1.0, float(v)))
    if soft > 1.5:
        soft, hot = soft / 100.0, hot / 100.0
    n = int(round(frac * width))
    return "[" + "█" * n + " " * (width - n) + br


def _our_band(st, cfg, width, ascii_mode):
    """'Our' band on the SAME two-box grid as MEMORY∥PROCESSES: left box
    x0..113, gap 114-115, right box x116..239. LLM LANE fills the left
    box full-height on a 2-column interior (bars+tails | latency);
    counters render as one dim line under the lanes; right box stacks
    the 5 LANES rows then a GPU strip row. No dead columns: every row
    either spans or right-anchors its tail."""
    h = st.get("health") or {}
    lw = 114                       # ┌ 0..┐ 113 — matches _merge_sideboxes
    gap = 2
    inner_l = max(1, lw - 5)       # content x=3..111 (109)
    rx = lw + gap                  # right box border x=116
    inner_r = width - rx - 4       # content x=118..~237
    bar_w = 18

    lane_rows = (_llm_left_rows(st, h, bar_w, ascii_mode)
                 if h else [[seg("  (vLLM telemetry offline)", "dim")]])
    mid_rows = _llm_mid_rows(st, h) if h else []

    counters = ""
    if h:
        rr = _llm_right_rows(st, h, cfg)
        counters = " · ".join(_row_text(r).strip() for r in rr
                              if _row_text(r).strip())
        win = st.get("mtp_win") or {}
        pp = win.get("per_pos") or {}
        hot = sum(1 for v in pp.values() if v > 0.5)
        wn = (f"win {win.get('window_s', '--')}s · {hot}/{len(pp)}"
              f" pos >50%" if pp else "— window warming")
        mid_rows = (list(mid_rows) + [None, None, None])[:3] \
            + [[seg(wn, "dim")]]
        sk = (st.get("cfg") or {}).get("vllm", {}).get("spec_tokens") or 3
        counters += f" · spec MTP k={sk}"
    # left rows total: title + lanes + counters (+1 spare)
    n_left = 1 + len(lane_rows) + (1 if counters else 0)

    lanes = st.get("lanes") or {}
    posture = (lanes.get("llm") or {}).get("posture")
    badge = f"[{posture}]" if posture else ""
    right_stack = []
    for key, label in (("llm", "LLM"), ("gate", "gate"), ("deci", "deci"),
                       ("router", "route"), ("dash", "dash")):
        ln = lanes.get(key) or {}
        up = ln.get("up")
        dot, dstyle = (("●", "good") if up else ("○", "bad")) \
            if not ascii_mode else (("*" if up else "x"),
                                    "good" if up else "bad")
        row = [seg("  "), seg(f"{label:<6}"), seg(dot, dstyle),
               seg(f" :{ln.get('port', '--')}", "dim"),
               seg(f" {ln.get('model') or '?'}",
                   "plain" if up else "dim")]
        if ln.get("image"):
            row.append(seg(f"  [{ln['image']}]", "dim"))
        det = ln.get("detail") if up else None
        right_stack.append((row, det, badge if key == "llm" else ""))
    hw = st.get("hw") or {}
    gpu_cell = [seg("  GPU  ", "dim")]
    t_c = hw.get("temp_c")
    p_w = hw.get("power_w")
    util = (hw.get("util_pct") or 0) / 100.0
    t_st = _temp_style(t_c)
    p_st = "warn" if (p_w or 0) > 180 else "good"
    u_st = "accent" if util > 0.05 else "dim"
    gpu_cell += [seg(" temp "), seg(_fmt_t(t_c), t_st),
                 seg(_gpu_zone_bar(t_c, 80, 90, 8), t_st),
                 seg("  pwr "), seg(_fmt_p(p_w), p_st),
                 seg(_gpu_zone_bar((p_w or 0) / 240.0, 0.75, 0.92, 8), p_st),
                 seg("  util "), seg(f"{util * 100:.0f}%", u_st),
                 seg(_gpu_zone_bar(util, 0.05, 0.9, 8), u_st)]
    thr = _thr_text(hw.get("throttle"))
    if thr != "—":
        gpu_cell.append(seg(" · thr ", "dim"),
                        )
        gpu_cell.append(seg(thr, "bad" if "ACTIVE" in thr else "dim"))
    right_stack.append((gpu_cell, None, ""))
    n_right = 1 + len(right_stack)

    def _cl(segs, span):
        cl, used = [], 0
        for t, s in segs:
            if used + len(t) > span:
                t = t[: span - used]
            if not t:
                break
            cl.append(seg(t, s))
            used += len(t)
        if used < span:
            cl.append(seg(" " * (span - used)))
        return cl

    def _ra(clamped, span):
        """right-ALIGN a clamped cell inside span (lead pad)."""
        txt = _row_text(clamped)
        lead = max(0, span - len(txt))
        return [seg(" " * lead)] + clamped

    out = []
    nrows = max(n_left, n_right)
    for i in range(nrows):
        vbar = "|" if ascii_mode else "│"
        if i == 0:
            lcell = [seg("LLM LANE · " + (h.get("model_name")
                        or h.get("engine", "")), "title")]
            rcell = [seg("  "), seg("LANES — what is serving", "title")]
        else:
            li = i - 1
            lcell = []
            if li < len(lane_rows):
                l1 = lane_rows[li]
                lm = mid_rows[li] if li < len(mid_rows) else []
                c1 = _cl([t for t in l1 if isinstance(t, tuple)], 61)
                c2 = [t for t in lm if isinstance(t, tuple)]
                t2 = _row_text(c2)
                lead2 = max(1, 61 - len(_row_text(c1)))
                lcell = [*c1, seg(" " * lead2), *c2]
            elif counters and li == len(lane_rows):
                span = inner_l - 2
                if len(counters) > span:
                    counters = counters[:span].rsplit(" · ", 1)[0]
                lcell = [seg(counters, "dim")]
            rsi = li - (len(lane_rows) + (1 if counters else 0) - 0) \
                if li >= len(lane_rows) + (1 if counters else 0) else None
        # RIGHT: stack index = i-1 (title=0). GPU row = last.
        si = i - 1
        rcell = []
        if i == 0:
            rcell = [seg("  "), seg("LANES — what is serving", "title")]
        elif si < len(right_stack):
            row, det, bd = right_stack[si]
            rcell = list(row)
            if det:
                pad = inner_r - len(_row_text(rcell)) - len(det)
                if pad >= 1:
                    rcell.append(seg(" " * pad, "dim"))
                rcell.append(seg(det, "dim"))
            if bd:
                pad2 = inner_r - len(_row_text(rcell)) - len(bd)
                if pad2 >= 1:
                    rcell.append(seg(" " * pad2, "dim"))
                rcell.append(seg(bd, "rev_acc" if not ascii_mode
                                 else "plain"))
        # shared row skeleton — fixed border columns
        out.append([seg(" "), seg(vbar), seg(" "),
                    *_cl(lcell, inner_l), seg(" "), seg(vbar),
                    seg(" " * gap), seg(" "), *_cl(rcell, inner_r),
                    seg(" "), seg(vbar)])
    return out


def build_lines(st, width, cfg=None):
    """v3.0 frame per approved redesign: zero data removal, reorganize/promote
    (James's constraints: reorg + demote only; solid █ bars; colour=state)."""
    cfg = cfg if cfg is not None else st.get("cfg") or {}
    L = []
    h = st.get("health") or {}
    off = st.get("offline", False)
    ascii_mode = st.get("ascii", False)
    now = st.get("now")
    r = st.get("rates") or {}
    tstr = now.strftime("%H:%M:%S") if now else "--:--:--"
    compact = width < 200

    # ---- header line
    model = h.get("model_name") or h.get("engine", "?")
    eng = h.get("engine", "?")
    quant = cfg.get("quantization")
    if quant is None and model and "NVFP4" in model.upper():
        quant = "NVFP4"
        cfg["quantization"] = "NVFP4"
    if off:
        L.append([seg("  \u25cf OFFLINE", "bad" if not ascii_mode else "plain"),
                  seg(f"  (last ok {st.get('last_ok_age')}s ago)"
                      if st.get("last_ok_age") is not None
                      else "  (never connected)", "dim"),
                  seg(f"   {tstr}", "dim")])
    else:
        waiting = h.get("requests_waiting") or 0
        if waiting:
            badge, bstyle = "PREFILLING", "accent"
        elif h.get("busy"):
            badge, bstyle = "BUSY", "warn"
        else:
            badge, bstyle = "IDLE", "dim"
        L.append([seg("  \u25cf ONLINE", "good" if not ascii_mode else "plain"),
                  seg("  "), seg(f"{model}", "accent"),
                  seg(f" ({eng})", "dim"),
                  seg(f" [{quant}]" if quant else "",
                      "badge" if quant else "dim"),
                  seg("  "), seg(badge, bstyle),
                  seg(f"   {tstr}", "dim")])

    # ---- VITALS || THROUGHPUT side-by-side (shared two-box grid)
    if not compact:
        vit_rows = build_vitals_rows(st, width, ascii_mode)
        thr_rows = build_throughput_rows(st, width, ascii_mode)
        L += _merge_sideboxes("VITALS", "THROUGHPUT \u00b7 instant \u00b7 rolling"
                              " 48s \u00b7 avg \u00b7 peak", vit_rows, thr_rows,
                              width, ascii_mode)
    else:
        L.append(box_top("VITALS", width, ascii_mode))
        L.append(build_vitals_rows(st, width, ascii_mode)[0])
        L.append(box_bottom(width, ascii_mode))

        # ---- THROUGHPUT panel
        L.append(box_top("THROUGHPUT", width, ascii_mode))
        L += build_throughput_rows(st, width, ascii_mode)
        L.append(box_bottom(width, ascii_mode))

    # ---- MEMORY || PROCESSES (wide: side-by-side; compact: stacked box)
    if compact:
        L.append(box_top("MEMORY", width, ascii_mode))
        L += build_mem_proc(st, cfg, width, ascii_mode)
        L.append(box_bottom(width, ascii_mode))
    else:
        mem_rows = _split_memory(st, cfg, width, ascii_mode)
        proc_rows = _proc_rows(st, cfg, width, ascii_mode)
        L += _merge_sideboxes("MEMORY", "PROCESSES \u00b7 VRAM by reservation"
                              " (sparkDash)", mem_rows, proc_rows, width,
                              ascii_mode)

    # ---- LLM LANE / LANES / GPU composite band (wide)
    if width >= 200 and not compact:
        band = _our_band(st, cfg, width, ascii_mode)
        # frame: one shared top border, divider rows already carry │;
        # bottom border ends the composite
        L.append(box_top("LLM LANE │ LANES", width, ascii_mode))
        L += [r for jx, r in enumerate(band) if jx > 0]   # skip band title row
        L.append(box_bottom(width, ascii_mode))
    else:
        # ---- LLM LANE: 3 fixed-gutter columns (compact path preserved)
        bar_w = 18 if not compact else 12
        lane_title = f" \u00b7 {model}" if model and model != eng else ""
        L.append(box_top("LLM LANE" + lane_title, width, ascii_mode))
        lane_rows = _llm_left_rows(st, h, bar_w, ascii_mode) if h else \
            [[seg("  (vLLM telemetry offline)", "dim")]]
        mid_rows = _llm_mid_rows(st, h) if h else [[] for _ in range(0)]
        right_rows = _llm_right_rows(st, h, cfg) if h else []
        if h and lane_rows and len(lane_rows) == 4:
            win = st.get("mtp_win") or {}
            pp = win.get("per_pos") or {}
            hot = sum(1 for v in pp.values() if v > 0.5)
            wn = (f"win {win.get('window_s', '--')}s \u00b7 {hot}/{len(pp)}"
                  f" pos >50%" if pp else "\u2014 window warming")
            sk = (st.get("cfg") or {}).get("vllm", {}).get("spec_tokens") or 3
            mid_rows = mid_rows[:3] + [[seg(wn, "dim")]]
            right_rows = right_rows[:3] + [[seg(f"k={sk}", "dim")]]
        if compact:
            if width >= 116:
                gutters = (2, 44, 74)
            else:
                gutters = (2, None, None)      # single: stack verbatim
                mid_rows = []
                right_rows = []
            merged = _merge_columns([lane_rows, mid_rows, right_rows],
                                    gutters, width)
        else:
            merged = _merge_columns([lane_rows, mid_rows, right_rows],
                                    (2, 66, 130), width)
        L += merged[:4] if h else merged
        L.append(box_bottom(width, ascii_mode))
        # ---- LANES panel
        L.append(box_top("LANES \u2014 what is serving", width, ascii_mode))
        L += build_lanes_rows(st, width, cfg, ascii_mode)
        L.append(box_bottom(width, ascii_mode))
        # ---- GPU panel: single honest row
        L.append(box_top("GPU", width, ascii_mode))
        L.append(build_gpu_row(st, width, cfg, ascii_mode) if False
                 else build_gpu_row(st, width, ascii_mode))
        L.append(box_bottom(width, ascii_mode))
    # ---- footer (wide)
    L.append([seg(f"  poll {st.get('interval', 1):g}s   "
                      f"errors {st.get('errors', 0)}   "
                      f"lanes {'on' if st.get('lanes') else 'off'}   "
                      f"{'[q] quit' if not st.get('once') else '--once snapshot'}"
                      f"   {tstr}", "dim")])
    return L


def _gpu_footer_compact(st, tstr):
    """Compact LANES bottom row: gpu strip + footer folded (A.2 row 24)."""
    hw = st.get("hw") or {}
    t = hw.get("temp_c")
    p = hw.get("power_w")
    sm = hw.get("sm_mhz")
    smx = hw.get("sm_max_mhz")
    u = hw.get("util_pct")
    thr = _thr_text(hw.get("throttle"))
    parts = [f"{_dash(t, 'C')}", f"{p:g}W" if p is not None else "\u2014"]
    if sm is not None:
        pct = f"({sm / smx * 100:.0f}%)" if smx else ""
        parts.append(f"SM{sm:g}MHz{pct}")
    if u is not None:
        parts.append(f"util{u:.0f}%")
    parts.append(f"thr{thr}")
    foot = (f"poll {st.get('interval', 1):g}s err {st.get('errors', 0)}"
            f" [q] quit {tstr}")
    return [[seg("  gpu " + " \u00b7 ".join(parts) + "   " + foot, "dim")]]



def _pad_to(x_abs):
    """seg of spaces up to absolute x — rows start at 0 in sideboxes."""
    return seg(" " * x_abs)


def _merge_sideboxes(title_l, title_r, left_rows, right_rows, width,
                     ascii_mode):
    """MEMORY | PROCESSES side-by-side, SEG-AWARE (colours preserved per
    James's colour steer — string-concat rows would drop them). Left box
    spans x=0..lw (borders incl.), gap, right box to edge; only text is
    clamped, structure kept."""
    lw = 114                       # left box: ┌ at 0, ┐ at 113
    gap = 2                        # blank columns between the two boxes
    inner_l = max(1, lw - 5)       # content x=3..111 (109) with ' │ ' pre
    rw = width - lw - gap          # right box: ┌ at 116, ┐ at 239
    inner_r = max(1, rw - 4)       # content x=118..237 (120)
    out = []
    dash = "-" if ascii_mode else "\u2500"
    vbar = "|" if ascii_mode else "\u2502"

    corner_l = "+" if ascii_mode else "\u250c"
    corner_r = "+" if ascii_mode else "\u2510"
    corner_b_l = "+" if ascii_mode else "\u2514"
    corner_b_r = "+" if ascii_mode else "\u2518"

    def _top(ttl, w):
        pad = max(3, w - len(ttl) - 6)
        return (corner_l + dash + " " + ttl + " "
                + dash * (pad + 1) + corner_r)

    out.append([seg(_top(title_l, lw) + " " * gap + _top(title_r, rw),
                    "dim")])
    nl, nr = len(left_rows), len(right_rows)
    for i in range(max(nl, nr)):
        lsegs = [t for t in (left_rows[i] if i < nl else [])
                 if isinstance(t, tuple) and len(t) == 2]
        rsegs = [t for t in (right_rows[i] if i < nr else [])
                 if isinstance(t, tuple) and len(t) == 2]

        def _clamp(segs, span):
            cl, used = [], 0
            for t, s in segs:
                if used + len(t) > span:
                    t = t[: span - used]
                if not t:
                    break
                cl.append(seg(t, s))
                used += len(t)
            if used < span:
                cl.append(seg(" " * (span - used)))
            return cl

        # FIXED column map (240-wide): left │ at x=1, content 3..112,
        # left-right │ x=114; gap 115-116; right box border+content to
        # right │ at x=238. Every part is width-fixed, zero drift.
        row = [seg(" "), seg(vbar), seg(" "), *_clamp(lsegs, inner_l),
               seg(" "), seg(vbar), seg(" " * gap), seg(vbar), seg(" "),
               *_clamp(rsegs, inner_r), seg(" "), seg(vbar)]
        out.append(row)
    # mirror the top border exactly: left └..┘ at 0..113, gap 2,
    # right └ at 116 .. ┘ at width-1 (239)
    bot_l = corner_b_l + dash * (lw - 2) + corner_b_r          # 0..113
    bot_r = corner_b_l + dash * (width - 117 - 1) + corner_b_r
    out.append([seg(bot_l + " " * gap + bot_r, "dim")])
    return out


def _split_memory(st, cfg, width, ascii_mode):
    h = st.get("health") or {}
    total, avail = st.get("mem_total"), st.get("mem_avail")
    rows = []
    if total and avail:
        used = total - avail
        frac = used / total
        stl = _thr_pct_style(frac)
        rows.append(
            [seg("system  "),
             seg(bar_solid(frac, width=22, ascii_mode=ascii_mode), stl),
             seg(f" {frac * 100:.1f}%", stl),
             seg(f"  {fmt_gib(used)} / {fmt_gib(total)}"
                 f"  ({fmt_gib(avail)} avail)", "dim")])
    else:
        rows.append([seg("system    meminfo unavailable", "dim")])
    gmu = h.get("gpu_memory_utilization")
    if gmu is not None:
        stl = _mem_style(gmu)
        rows.append(
            [seg("engine  "),
             seg(bar_solid(gmu, width=22, ascii_mode=ascii_mode), stl),
             seg(f" {gmu * 100:.1f}%", stl),
             seg("   gpu_memory_utilization", "dim")])
    else:
        rows.append([seg("engine    \u2014", "dim")])
    gpuu = st.get("um_gpu")
    if gpuu is not None and st.get("mem_total"):
        cf = gpuu / (st["mem_total"] / 1024.0)
        stl = _mem_style(cf)
        rows.append(
            [seg("uma gpu "),
             seg(bar_solid(cf, width=22, ascii_mode=ascii_mode), stl),
             seg(f" {fmt_gib(gpuu * 1024)}", stl),
             seg(f"  split cpu {fmt_gib((st.get('um_cpu') or 0) * 1024)}"
                 " (d)", "dim")])
        # demoted v2 'note:' content rides here (P1-5: content kept, dim)
        rows[-1].append(seg(
            "  \u00b7 meminfo avail honest, NVML used = reservation", "dim"))
    else:
        rows.append([seg("uma gpu  \u2014", "dim")])
    cpu = st.get("cpu_pct")
    if cpu is not None:
        stl = _thr_pct_style(cpu / 100.0)
        draw = st.get("cpu_draw")
        tdp = st.get("cpu_tdp")
        temp = st.get("cpu_temp")
        pw = (f"{draw:g}W" if draw is not None else "\u2014") \
            + (f"/{tdp:g}W" if tdp else "")
        rows.append(
            [seg(f"cpu {cpu:.0f}% "),
             seg(bar_solid(cpu / 100.0, width=10, ascii_mode=ascii_mode),
                 stl),
             seg(f" {pw}", "plain"),
             seg(f" {temp:g}C" if temp is not None else "", "dim")])
    return rows


def _proc_rows(st, cfg, width, ascii_mode):
    """PROCESSES right box rows (wide): per-proc VRAM bars + footnote.

    Bar fraction = proc's VRAM / UMA total (meminfo KiB -> MiB)."""
    procs = st.get("procs") or []
    total = st.get("mem_total")
    total_mib = (total / 1024.0) if total else None
    rows = []
    for p in procs[:4]:
        frac = (p["mib"] / total_mib
                if (total_mib and p.get("mib")) else None)
        stl = _mem_style(frac or 0)
        rows.append(
            [seg(f"{p['name']:<12} "),
             seg(bar_solid(frac, width=18, ascii_mode=ascii_mode), stl),
             seg(f" {fmt_gib(p['mib'] * 1024)}", "plain"),
             seg(f"  pid {p.get('pid', '?')}", "dim")])
    if not rows:
        rows.append([seg("(sparkDash offline \u2014 no per-proc VRAM)",
                         "dim")])
    hw = st.get("hw") or {}
    vram = hw.get("vram_used_mib")
    s_mib = sum(p["mib"] for p in procs)
    if vram is not None:
        agree = ("sources agree" if s_mib and abs(s_mib - vram) < vram * 0.05
                 else "NVML [N/A] \u2014 procs sum")
        rows.append(
            [seg("\u2570 procs-sum ", "dim"),
             seg(fmt_gib(s_mib * 1024) if s_mib else fmt_gib(vram * 1024),
                 "plain"),
             seg(f" \u2248 NVML used \u2014 {agree}", "dim")])
    return rows


def _merge_str_pad(left, lw, right, width):
    """Two border strings on one row: left at lw, right hugs the edge."""
    line = left.ljust(lw + 2) + right
    return line[:max(0, width)]


# ------------------------------------------------------------ snapshot mode


def run_once(args, cfg):
    now = time.time()
    prev = None
    prev_t = None
    health = None
    try:
        health = fetch_any(args.url, timeout=3.0)
        if args.sample and args.sample > 0:
            prev = {c: health.get(c) for c in COUNTERS}
            prev_t = now
            time.sleep(args.sample)
            now = time.time()
            health = fetch_any(args.url, timeout=3.0)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError) as exc:
        print(f"sparkmon: telemetry unreachable ({exc})", file=sys.stderr)
    rates = compute_rates(health, prev, prev_t, now) if health else {
        "decode_tps": None, "prefill_tps": None, "prefill_src": None}
    w_avgs = compute_window_avgs(health, prev, prev_t, now) if health else {}
    total, avail = read_meminfo()
    dash = _dash_poll()
    d0 = dash or {}
    gpuu, cpuu, oom = d0.get("um_gpu"), d0.get("um_cpu"), d0.get("um_oom")
    pa = {"sum": rates.get("prefill_tps") or 0.0,
          "n": 1 if rates.get("prefill_tps") is not None else 0}
    st = {"health": health, "rates": rates, "offline": health is None,
          "last_ok_age": None, "errors": 0, "interval": args.interval,
          "now": datetime.now(), "ascii": args.ascii,
          "mem_total": total, "mem_avail": avail, "hw": read_hardware(),
          "um_gpu": gpuu, "um_cpu": cpuu, "um_oom": oom,
          "cpu_pct": d0.get("cpu_pct"), "cpu_temp": d0.get("cpu_temp"),
          "cpu_draw": d0.get("cpu_draw"), "cpu_tdp": d0.get("cpu_tdp"),
          "procs": d0.get("procs") or [],
          "host_uptime": read_host_uptime(),
          "once": True,
          "hist": {"decode": deque(maxlen=HIST_N),
                   "prefill": deque(maxlen=HIST_N)},
          "avg": {"decode": None, "dec_tot": None, "pre_tot": None,
                  "prefill": (pa["sum"] / pa["n"]) if pa["n"] else None},
          "win_avgs": w_avgs,
          "mtp_win": {}, "lanes": probe_lanes(cfg) if cfg["lanes"] else {}}
    lines = build_lines(st, 120, cfg)
    print("\n".join("".join(t for t, _ in line).rstrip() for line in lines))


# ------------------------------------------------------------ main / curses


def make_cfg(args):
    # launch-arg defaults; env SPARKMON_* (read in the constants region)
    # already carries the user override — seed it into cfg so the render
    # fallback (env > docker > default) holds even with --no-lanes
    return {"lanes": not args.no_lanes, "gate_url": DEFAULT_GATE,
            "deci_url": DEFAULT_DECI, "llm_model": None, "remote": args.remote,
            # sparkDash recipeInfo parity (v2.2): filled here, refined by the
            # docker-inspect capture in probe_lanes once lanes probe runs.
            "recipe_model": "Mia-AiLab/Qwen3.8-Flash-Next-NVFP4",
            "recipe_author": "styles01",
            "quantization": "NVFP4",
            "vllm": {"max_num_seqs": DEFAULT_SEQS,
                     "max_model_len": DEFAULT_CTX},
            # source tags: env SPARKMON_* always beats docker-inspect capture
            "vllm_src": {"max_num_seqs": "env" if "SPARKMON_MAX_SEQS"
                         in os.environ else "default",
                         "max_model_len": "env" if "SPARKMON_CTX"
                         in os.environ else "default"}}


class Avg:
    """Session-average tok/s + totals."""

    def __init__(self):
        self.sum = 0.0
        self.n = 0
        self.dec_tot = None
        self.pre_tot = None

    def update(self, rates, health):
        dec = rates.get("decode_tps")
        if dec is not None:
            self.sum += dec
            self.n += 1
        for key, attr in (("completion_tokens_total", "dec_tot"),
                          ("prompt_tokens_total", "pre_tot")):
            v = (health or {}).get(key)
            if v is not None:
                setattr(self, attr, v)

    @property
    def decode(self):
        return self.sum / self.n if self.n else None


def _poll_once(args, cfg, state):
    now = time.time()
    health = None
    try:
        health = fetch_any(args.url, timeout=max(0.5, args.interval))
        state["errors"] = 0
        state["last_ok"] = now
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError):
        state["errors"] += 1
    rates = {"decode_tps": None, "prefill_tps": None, "prefill_src": None}
    w_avgs = {}
    if health is not None:
        rates = compute_rates(health, state["prev"], state["prev_t"], now)
        w_avgs = compute_window_avgs(health, state["prev"], state["prev_t"],
                                     now)
        state["prev"] = {c: health.get(c) for c in COUNTERS}
        state["prev_t"] = now
        state["hist"]["decode"].append(rates.get("decode_tps"))
        state["hist"]["prefill"].append(rates.get("prefill_tps"))
        state["avg"].update(rates, health)
        pa = state.setdefault("pre_avg", {"sum": 0.0, "n": 0})
        if rates.get("prefill_tps") is not None:
            pa["sum"] += rates["prefill_tps"]
            pa["n"] += 1
        if not cfg.get("llm_model") and health.get("model_name"):
            cfg["llm_model"] = health["model_name"]
    if cfg["lanes"] and (now - state["lanes_ts"] > 3.0 or state["lanes"] is None):
        state["lanes"] = probe_lanes(cfg)
        state["lanes_ts"] = now
    mtp_win = state["mtp_win"].update(health) if health else {}
    total, avail = read_meminfo()
    if not state.get("host_uptime"):
        state["host_uptime"] = read_host_uptime()
    dash = _dash_poll()
    d0 = dash or {}
    gpuu, cpuu, oom = d0.get("um_gpu"), d0.get("um_cpu"), d0.get("um_oom")
    avg = state["avg"]
    avg_d = {"decode": avg.decode, "dec_tot": avg.dec_tot, "pre_tot": avg.pre_tot,
             "prefill": (state["pre_avg"]["sum"] / state["pre_avg"]["n"])
             if state.get("pre_avg", {}).get("n") else None}
    return {"health": health, "rates": rates, "offline": health is None,
            "last_ok_age": int(now - state["last_ok"])
            if state.get("last_ok") else None,
            "errors": state["errors"], "interval": args.interval,
            "now": datetime.now(), "ascii": state["ascii"],
            "mem_total": total, "mem_avail": avail, "hw": read_hardware(),
            "um_gpu": gpuu, "um_cpu": cpuu, "um_oom": oom, "cpu_pct": d0.get("cpu_pct"), "cpu_temp": d0.get("cpu_temp"), "cpu_draw": d0.get("cpu_draw"), "cpu_tdp": d0.get("cpu_tdp"), "procs": d0.get("procs") or [],
            "host_uptime": state.get("host_uptime"),
            "hist": state["hist"], "avg": avg_d, "win_avgs": w_avgs,
            "mtp_win": mtp_win, "lanes": state["lanes"] or {},
            "cfg": cfg}


def _tui_loop(stdscr, args, cfg):
    curses.curs_set(0)
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK
        curses.init_pair(1, curses.COLOR_GREEN, bg)
        curses.init_pair(2, curses.COLOR_YELLOW, bg)
        curses.init_pair(3, curses.COLOR_RED, bg)
        curses.init_pair(4, curses.COLOR_CYAN, bg)
        curses.init_pair(5, curses.COLOR_WHITE, bg)
    stdscr.timeout(max(50, int(args.interval * 1000)))

    state = {"prev": None, "prev_t": None, "errors": 0, "last_ok": None,
             "ascii": args.ascii or os.environ.get("TERM") == "linux",
             "hist": {"decode": deque(maxlen=HIST_N),
                      "prefill": deque(maxlen=HIST_N)},
             "avg": Avg(), "mtp_win": MTPWindow(),
             "lanes": None, "lanes_ts": 0.0}

    while True:
        st = _poll_once(args, cfg, state)
        _draw(stdscr, st)
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            return
        if key == curses.KEY_RESIZE:
            stdscr.erase()


def _draw(stdscr, st):
    maxy, maxx = stdscr.getmaxyx()
    st["compact"] = maxy < 26
    width = max(40, maxx - 1)
    lines = build_lines(st, width, st.get("cfg"))
    styles = {
        "title": curses.A_BOLD | curses.color_pair(5),
        "dim": curses.A_DIM,
        "good": curses.color_pair(1),
        "warn": curses.color_pair(2),
        "bad": curses.color_pair(3),
        "accent": curses.color_pair(4),
        "badge": curses.A_REVERSE,
        "rev_bad": curses.A_REVERSE | curses.color_pair(3),
        "rev_good": curses.A_REVERSE | curses.color_pair(1),
        "rev_acc": curses.A_REVERSE | curses.color_pair(4),
        None: curses.A_NORMAL,
    }
    stdscr.erase()
    try:
        for y, line in enumerate(lines[: maxy - 1]):
            x = 0
            for text, style in line:
                if not text:
                    continue
                stdscr.addnstr(y, x, text, max(1, width - x),
                               styles.get(style, curses.A_NORMAL))
                x += len(text)
                if x >= width:
                    break
        stdscr.refresh()
    except curses.error:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="sparkmon",
        description="SparkDash-in-a-terminal: LLM lane + lanes + memory TUI "
                    "for NVIDIA DGX Spark (pure stdlib).")
    ap.add_argument("-i", "--interval", type=float, default=1.0,
                    help="poll interval seconds (default 1.0)")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"LLM base url (default {DEFAULT_URL})")
    ap.add_argument("--once", action="store_true",
                    help="print one frame as plain text and exit")
    ap.add_argument("--sample", type=float, default=0.0, metavar="SECONDS",
                    help="with --once: sample counter deltas over SECONDS")
    ap.add_argument("--ascii", action="store_true",
                    help="pure-ASCII bars (auto on TERM=linux)")
    ap.add_argument("--no-lanes", action="store_true",
                    help="skip the LANES panel probes")
    ap.add_argument("--remote", default=None, metavar="USER@HOST",
                    help="probe lanes over ssh from another machine")
    args = ap.parse_args(argv)
    cfg = make_cfg(args)

    if args.once:
        run_once(args, cfg)
        return 0
    try:
        curses.wrapper(_tui_loop, args, cfg)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())