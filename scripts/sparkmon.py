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
SPARKMON_CTX (262144), SPARKMON_GATE (http://localhost:8710),
SPARKMON_DECI (http://localhost:8712), SPARKMON_ROUTER (:8711 TCP only).
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
    """Rolling sparkline; None-valued samples render as blank."""
    glyphs = _SPARK_ASCII if ascii_mode else _SPARK
    vals = list(vals)[-width:]
    if not vals:
        return " " * width
    known = [v for v in vals if v is not None]
    peak = (max(known) if known else 0.0) or 1.0
    out = []
    for v in vals:
        if v is None:
            out.append(" ")
        else:
            idx = min(len(glyphs) - 1, 1 + int((v / peak) * (len(glyphs) - 2)))
            out.append(glyphs[idx])
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
        elif name == "kv_cache_usage_perc":
            h["kv_cache_usage"] = val
        elif name == "prefix_cache_queries_total":
            h["prefix_queries"] = val
        elif name == "prefix_cache_hits_total":
            h["prefix_hits"] = val
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
)

# histogram sum/count keys -> (label, fmt) for the right column
_HIST_KEYS = (
    ("ttft", "TTFT", "s"), ("itl", "ITL", "ms"), ("e2e", "e2e", "s"),
)


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
    elif llm_up and cfg.get("remote"):
        lanes["llm"]["image"] = "docker n/a (remote probe)"
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
    return lanes


# ------------------------------------------------------------ rendering

S = {"plain": None, "title": "title", "dim": "dim", "good": "good",
     "warn": "warn", "bad": "bad", "accent": "accent", "badge": "badge"}


def seg(text, style="plain"):
    return (text, S.get(style, None))


def box_top(title, width, ascii_mode):
    if ascii_mode:
        return [seg("+" + ("-" * max(3, width - 2)) + "+", "dim"),
                seg(f"| {title} ", "title")]
    return [seg("\u250c\u2500 " + title + " ", "title"),
            seg("\u2500" * max(3, width - len(title) - 5) + "\u2510", "dim")]


def box_bottom(width, ascii_mode):
    return [seg("+" + ("-" * max(3, width - 2)) + "+" if ascii_mode
                else "\u2514" + "\u2500" * max(3, width - 2) + "\u2518", "dim")]


def _rcol_row(label, value, style="plain"):
    """Right-column row: fixed label pad + value, btop label-left/value-right."""
    return [seg(f"  {label:<10}"), seg(value, style)]


def build_llm_right_col(st, h):
    """Right column rows: label left, value right, btop-style. '—' = missing."""
    w_avgs = st.get("win_avgs") or {}
    rows = []

    def lat_avg(key):
        wv = w_avgs.get(key)
        if wv:
            return wv["avg"], wv["kind"]
        c = h.get(f"{key}_seconds_count")
        s = h.get(f"{key}_seconds_sum")
        if c and s is not None:
            return s / c, "avg"
        return None, "avg"

    # ttft/itl/e2e (window avg when deltas exist, else lifetime 'avg'),
    # kind tag inline — one row per metric
    for key, label, unit in _HIST_KEYS:
        v, kind = lat_avg(key)
        if v is None:
            rows.append(_rcol_row(label, "—", "dim"))
            continue
        val = f"{v * 1000:.1f}ms" if unit == "ms" else f"{v:.2f}s"
        row = _rcol_row(label, val, "plain")
        row.append(seg(f"  ({kind})", "dim"))
        rows.append(row)

    # requests: live + lifetime on one row
    rows.append(_rcol_row(
        "reqs",
        f"{h.get('requests_running') or 0:.0f} run / "
        f"{h.get('requests_waiting') or 0:.0f} wait", "plain"))
    rows[-1].append(seg(
        f"   {fmt_num(h.get('requests_completed_total'))} done / "
        f"{fmt_num(h.get('requests_failed_total'))} fail", "dim"))

    # engine lifetime token totals on one row (session tots live on the left)
    rows.append(_rcol_row(
        "eng tok",
        f"{fmt_num(h.get('prompt_tokens_total'))} pre / "
        f"{fmt_num(h.get('completion_tokens_total'))} gen", "plain"))

    # prefix cache: rate + raw queries/hits on one row
    pfx = h.get("prefix_cache_hit_rate")
    pq, ph = h.get("prefix_queries"), h.get("prefix_hits")
    if pfx is not None and pq is not None:
        rows.append(_rcol_row("prefix", f"{pfx * 100:.1f}%", "plain"))
        rows[-1].append(seg(f"   {fmt_num(pq)}q / {fmt_num(ph)} hits", "dim"))
    else:
        rows.append(_rcol_row("prefix", "—", "dim"))

    # kv cache usage + dtype badge on one row
    kv = h.get("kv_cache_usage")
    if kv is not None:
        rows.append(_rcol_row("kv cache", f"{kv * 100:.1f}%", "plain"))
        dtype = h.get("kv_cache_dtype")
        if dtype and dtype != "auto":
            rows[-1].append(seg(f"   [kv {dtype}]", "accent" if dtype == "fp8"
                                else "dim"))
    else:
        rows.append(_rcol_row("kv cache", "—", "dim"))

    # MTP accepted/drafted totals
    acc = h.get("mtp_accepted_tokens_total")
    drf = h.get("mtp_drafted_tokens_total")
    if acc is not None or drf is not None:
        rows.append(_rcol_row(
            "MTP", f"{fmt_num(acc)} acc / {fmt_num(drf)} draft", "plain"))
    else:
        rows.append(_rcol_row("MTP", "—", "dim"))
    return rows


def build_lines(st, width):
    """v2 frame: header, LLM lane box, LANES box, memory box, footer."""
    L = []
    h = st.get("health") or {}
    off = st.get("offline", False)
    ascii_mode = st.get("ascii", False)
    now = st.get("now")
    r = st.get("rates") or {}
    tstr = now.strftime("%H:%M:%S") if now else "--:--:--"

    # ---- header line
    model = h.get("model_name") or h.get("engine", "?")
    eng = h.get("engine", "?")
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
                  seg("  "), seg(badge, bstyle),
                  seg(f"   {tstr}", "dim")])

    # ---- LLM lane box
    L.append(box_top("LLM lane", width, ascii_mode))
    dec = r.get("decode_tps")
    pre = r.get("prefill_tps")
    src = r.get("prefill_src") or ""
    dec_h = st["hist"].get("decode")
    pre_h = st["hist"].get("prefill")
    avg = st["avg"].get("decode")
    lane = []
    lane.append([seg("  decode   "), seg(spark(dec_h, ascii_mode=ascii_mode), "accent"),
                 seg("  "), seg(fmt_rate(dec), "good" if dec else "dim")])
    lane.append([seg("  prefill  "), seg(spark(pre_h, ascii_mode=ascii_mode), "accent"),
                 seg("  "), seg(fmt_rate(pre), "accent" if pre else "dim"),
                 seg(f"  [{src}]" if src else "", "dim")])

    # MTP overall (token-accepted / token-drafted)
    acc, drf = h.get("mtp_accepted_tokens_total"), h.get("mtp_drafted_tokens_total")
    overall = (acc / drf) if (acc is not None and drf) else None
    win = st.get("mtp_win") or {}
    w_over = win.get("overall")
    lane.append([seg("  MTP      "),
              seg(bar_solid(overall, ascii_mode=ascii_mode) if overall is not None
                  else "[" + " " * BAR_WIDTH + "]",
                  "good" if (w_over or overall or 0) >= 0.5
                  else ("warn" if (w_over or overall or 0) >= 0.3 else "dim")),
              seg(" "), seg(fmt_pct(overall), "plain"),
              seg(f"   live window {fmt_pct(w_over)}"
                  f" ({win.get('window_s', '--')}s)" if w_over is not None else "",
                  "dim"),
              seg(f"   {fmt_num(acc)}/{fmt_num(drf)} tok", "dim")])
    # per-position compact bars
    pp = win.get("per_pos") or {}
    if pp:
        cells = []
        for p in sorted(pp)[:6]:
            v = pp[p]
            cells.append(f"p{p} {fmt_pct(v)}"
                         f" {bar_solid(v, width=6, ascii_mode=ascii_mode)}")
        lane.append([seg("  MTP pos  "), seg("  ".join(cells), "plain"),
                  seg(f"   ({len(pp)} positions, k=3)", "dim")])
    else:
        pos_h = h.get("mtp_accept_by_position") or []
        if pos_h:
            cells = [f"p{p.get('position')} {fmt_num(p.get('tested'))} tok"
                     for p in pos_h[:6]]
            lane.append([seg("  MTP pos  "), seg("  ".join(cells), "dim"),
                         seg("  (lifetime — window warming up)", "dim")])

    kv = h.get("kv_cache_usage")
    kvc = h.get("kv_capacity_tok")
    pfx = h.get("prefix_cache_hit_rate")
    lane.append([seg("  kv cache  "),
                 seg(bar_solid(kv, ascii_mode=ascii_mode) if kv is not None
                     else "[" + " " * BAR_WIDTH + "]",
                     "good" if (kv or 0) < 0.6
                     else ("warn" if (kv or 0) < 0.85 else "bad")),
                 seg(" "), seg(fmt_pct(kv), "plain"),
                 seg(f"   cap {fmt_num(kvc)} tok" if kvc else "", "dim")])
    lane.append([seg("  prefix    "),
                 seg(bar_solid(pfx, ascii_mode=ascii_mode) if pfx is not None
                     else "[" + " " * BAR_WIDTH + "]", "accent"),
                 seg(" "), seg(fmt_pct(pfx), "plain")])
    if not h:
        lane.append([seg("  (vLLM telemetry offline)", "dim")])
    ctx_len = h.get("context_length") or DEFAULT_CTX

    # right column: merge lane rows and metric rows side by side.
    # Wide terminal: right column hugs the box's right edge (btop-style).
    # Narrow terminal (<100 cols): fixed offset at col 62, drop rows that
    # would overflow.
    rrows = build_llm_right_col(st, h) if h else []
    if width >= 100:
        rcol_w = max(len("".join(t for t, _ in rr)) for rr in rrows) if rrows else 0
        rcol_x = max(66, width - 2 - rcol_w)   # 2 = right border + margin
        narrow = False
    else:
        rcol_x, narrow = 62, True
    merged = []
    for i in range(max(len(lane), len(rrows))):
        left = list(lane[i]) if i < len(lane) else [seg(" " * 12)]
        ltxt = sum(len(t) for t, _ in left)
        pad = rcol_x - ltxt if not narrow else max(1, rcol_x - ltxt)
        rrow = rrows[i] if i < len(rrows) else []
        if narrow and ltxt + len("".join(t for t, _ in rrow)) > width - 2:
            rrow = []                          # would overflow: drop
        merged.append(left + [seg(" " * max(1, pad))] + rrow)
    for row in merged:
        L.append(row)
    L.append(box_bottom(width, ascii_mode))

    # ---- LANES box
    L.append(box_top("LANES — what is serving", width, ascii_mode))
    lanes = st.get("lanes") or {}
    for key, label, lcol in [("llm", "LLM  ", "good"),
                             ("gate", "gate ", "good"),
                             ("deci", "deci ", "good"),
                             ("router", "route", "dim")]:
        ln = lanes.get(key) or {}
        up = ln.get("up")
        dot, dstyle = (("\u25cf", "good") if up else ("\u25cb", "bad")) \
            if not ascii_mode else (("*" if up else "x"),
                                    "good" if up else "bad")
        row = [seg(f"  {label} "), seg(dot, dstyle),
               seg(f" :{ln.get('port', '--')}", "dim"),
               seg(f"  {ln.get('model') or '?'}", "plain" if up else "dim")]
        if ln.get("image"):
            row.append(seg(f"   [{ln['image']}]", "dim"))
        if ln.get("detail") and up:
            row.append(seg(f"   {ln['detail']}", "dim"))
        L.append(row)
    if h.get("busy") and h.get("requests_waiting") is not None:
        L.append([seg(f"        LLM queue: running "
                      f"{'yes' if h.get('busy') else 'no'}"
                      f", waiting {h.get('requests_waiting', 0):.0f}", "dim")])
    L.append(box_bottom(width, ascii_mode))

    # ---- memory box
    L.append(box_top("Memory (GB10 unified)", width, ascii_mode))
    total, avail = st.get("mem_total"), st.get("mem_avail")
    if total and avail:
        used = total - avail
        frac = used / total
        style = "good" if frac < 0.7 else ("warn" if frac < 0.9 else "bad")
        L.append([seg("  system    "),
                  seg(bar_solid(frac, width=24, ascii_mode=ascii_mode), style),
                  seg(f" {frac * 100:.1f}%", "plain"),
                  seg(f"  {fmt_gib(used)} / {fmt_gib(total)}"
                      f"   ({fmt_gib(avail)} avail)", "dim")])
    else:
        L.append([seg("  system      meminfo unavailable", "dim")])
    gmu = h.get("gpu_memory_utilization")
    if gmu is not None:
        style = "good" if gmu < 0.7 else ("warn" if gmu < 0.9 else "bad")
        L.append([seg("  engine     "),
                  seg(bar_solid(gmu, width=24, ascii_mode=ascii_mode), style),
                  seg(f" {gmu * 100:.1f}%", "plain"),
                  seg("   gpu_memory_utilization (engine view)", "dim")])
    L.append([seg("  note: system row = /proc/meminfo available (honest on"
                  " coherent UMA); NVML 'used' reports reservation.", "dim")])
    L.append(box_bottom(width, ascii_mode))

    # ---- footer
    L.append([seg(f"  poll {st.get('interval', 1):g}s   "
                  f"errors {st.get('errors', 0)}   "
                  f"lanes {'on' if st.get('lanes') else 'off'}   "
                  f"{'[q] quit' if not st.get('once') else '--once snapshot'}"
                  f"   {tstr}", "dim")])
    return L


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
    st = {"health": health, "rates": rates, "offline": health is None,
          "last_ok_age": None, "errors": 0, "interval": args.interval,
          "now": datetime.now(), "ascii": args.ascii,
          "mem_total": total, "mem_avail": avail,
          "host_uptime": read_host_uptime(), "once": True,
          "hist": {"decode": deque(maxlen=HIST_N),
                   "prefill": deque(maxlen=HIST_N)},
          "avg": {"decode": None, "dec_tot": None, "pre_tot": None},
          "win_avgs": w_avgs,
          "mtp_win": {}, "lanes": probe_lanes(cfg) if cfg["lanes"] else {}}
    lines = build_lines(st, 110)
    print("\n".join("".join(t for t, _ in line).rstrip() for line in lines))


# ------------------------------------------------------------ main / curses


def make_cfg(args):
    return {"lanes": not args.no_lanes, "gate_url": DEFAULT_GATE,
            "deci_url": DEFAULT_DECI, "llm_model": None, "remote": args.remote}


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
        if not cfg.get("llm_model") and health.get("model_name"):
            cfg["llm_model"] = health["model_name"]
    if cfg["lanes"] and (now - state["lanes_ts"] > 3.0 or state["lanes"] is None):
        state["lanes"] = probe_lanes(cfg)
        state["lanes_ts"] = now
    mtp_win = state["mtp_win"].update(health) if health else {}
    total, avail = read_meminfo()
    if not state.get("host_uptime"):
        state["host_uptime"] = read_host_uptime()
    avg = state["avg"]
    avg_d = {"decode": avg.decode, "dec_tot": avg.dec_tot, "pre_tot": avg.pre_tot}
    return {"health": health, "rates": rates, "offline": health is None,
            "last_ok_age": int(now - state["last_ok"])
            if state.get("last_ok") else None,
            "errors": state["errors"], "interval": args.interval,
            "now": datetime.now(), "ascii": state["ascii"],
            "mem_total": total, "mem_avail": avail,
            "host_uptime": state.get("host_uptime"),
            "hist": state["hist"], "avg": avg_d, "win_avgs": w_avgs,
            "mtp_win": mtp_win, "lanes": state["lanes"] or {}}


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
    lines = build_lines(st, width)
    styles = {
        "title": curses.A_BOLD | curses.color_pair(5),
        "dim": curses.A_DIM,
        "good": curses.color_pair(1),
        "warn": curses.color_pair(2),
        "bad": curses.color_pair(3),
        "accent": curses.color_pair(4),
        "badge": curses.A_REVERSE,
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