#!/usr/bin/env python3
"""sparkllm — LLM engine lane monitor for NVIDIA DGX Spark (GB10).

Single-file, pure-stdlib curses TUI. Renders the serving telemetry exposed by
the EXL3 OpenAI-compatible engine's /health endpoint (default
http://localhost:8000/health): decode/prefill tok/s (counter deltas), MTP
draft acceptance overall + per position, KV/prefix cache usage, request
counters, and honest GB10 unified-memory usage from /proc/meminfo
(available-based, since NVML "used" on coherent UMA reports reservation, not
allocation).

Pairs with a hardware monitor (sparkview / dgxtop / sparktop) — this tool
deliberately shows only the LLM lane plus just enough memory context.

Usage:
    sparkllm                    # live TUI, poll every 1s
    sparkllm -i 2               # poll every 2s
    sparkllm --once             # print one frame as plain text and exit
    sparkllm --once --sample 2  # --once, but sample deltas over 2s first
    sparkllm --url http://localhost:8000/health
    sparkllm --ascii            # pure-ASCII bars (for tty1 console fonts)

Designed for the HDMI console (tty1) or GNOME Terminal on the Spark; also
works fine over SSH and inside tmux.
"""
from __future__ import annotations

import argparse
import curses
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

DEFAULT_URL = "http://localhost:8000/health"
BAR_WIDTH = 12

# ---------------------------------------------------------------- utilities


def bar(frac, width=BAR_WIDTH, ascii_mode=False):
    """ASCII/unicode progress bar like [██████░░░░]."""
    frac = max(0.0, min(1.0, float(frac)))
    fill = "#" if ascii_mode else "\u2588"   # █
    empty = "-" if ascii_mode else "\u2591"  # ░
    n = int(round(frac * width))
    return "[" + fill * n + empty * (width - n) + "]"


def fmt_num(v):
    if v is None:
        return "--"
    return f"{v:,.0f}"


def fmt_rate(v):
    return "-- tok/s" if v is None else f"{v:,.1f} tok/s"


def fmt_pct(v, digits=1):
    return "--" if v is None else f"{v * 100:.{digits}f}%"


def fmt_gib(kib):
    return f"{kib / 1024.0 / 1024.0:.1f} GiB"


def fmt_uptime(seconds):
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def read_meminfo():
    """Return (total_kib, available_kib) from /proc/meminfo. None if unreadable."""
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
    """Host uptime in seconds from /proc/uptime, or None."""
    try:
        with open("/proc/uptime", "r") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def fetch_health(url, timeout=2.0):
    """GET the health JSON. Raises on any failure."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ------------------------------------------------------------ rate math

COUNTERS = (
    "completion_tokens_total",
    "prefill_tokens_processed_total",
    "prompt_tokens_total",
    "mtp_accepted_tokens_total",
    "mtp_drafted_tokens_total",
    "requests_completed_total",
    "requests_failed_total",
)


def compute_rates(cur, prev, prev_t, now):
    """Counter deltas between polls; resync silently if counters reset.

    Returns dict with decode_tps / prefill_tps (None = unknown, e.g. first
    poll) plus a prefill_src tag ('live' or 'delta').
    """
    out = {"decode_tps": None, "prefill_tps": None, "prefill_src": None}
    if prev is None or prev_t is None:
        return out
    dt = now - prev_t
    if dt <= 0:
        return out
    try:
        reset = any(c in cur and c in prev and cur[c] < prev[c] for c in COUNTERS)
    except TypeError:
        reset = True
    if reset:
        return out  # resync: caller stores new counters; show -- this frame

    def _delta(c):
        try:
            return float(cur.get(c, 0)) - float(prev.get(c, 0))
        except (TypeError, ValueError):
            return 0.0

    out["decode_tps"] = _delta("completion_tokens_total") / dt
    # Prefill: prefer the engine's live gauge, fall back to counter delta.
    live = cur.get("prefill_rate") or 0.0
    if not live:
        live = cur.get("prefill_tps_live") or 0.0
    if live:
        out["prefill_tps"] = float(live)
        out["prefill_src"] = "live"
    else:
        out["prefill_tps"] = _delta("prefill_tokens_processed_total") / dt
        out["prefill_src"] = "delta"
    return out


# ------------------------------------------------------------ rendering
# Lines are lists of (text, style) segments; style is a logical name mapped
# to curses pairs at draw time, or None for plain (used by --once).

S = {"plain": None, "title": "title", "dim": "dim", "good": "good",
     "warn": "warn", "bad": "bad", "accent": "accent", "badge": "badge"}


def seg(text, style="plain"):
    return (text, S.get(style, None))


def rate_color(v, good=0.5, mid=0.3):
    if v is None:
        return "dim"
    if v >= good:
        return "good"
    if v >= mid:
        return "warn"
    return "bad"


def build_lines(st, width):
    """Build the frame as a list of segment-lists from state dict `st`."""
    L = []
    h = st.get("health") or {}
    off = st.get("offline", False)
    ascii_mode = st.get("ascii", False)
    now = st.get("now")

    # ---- header
    engine = h.get("engine", "?")
    backend = h.get("backend", "?")
    ctx_len = h.get("context_length")
    host_up = st.get("host_uptime")
    L.append([seg("sparkllm", "title"), seg("  "),
              seg(f"{engine}", "accent"), seg(f"  backend {backend}", "plain"),
              seg(f"   ctx {ctx_len:,}" if ctx_len else "   ctx ?", "dim"),
              seg(f"   host-up {fmt_uptime(host_up)}" if host_up else "", "dim")])
    tstr = now.strftime("%H:%M:%S") if now else "--:--:--"
    if off:
        L.append([seg("  \u25cf OFFLINE", "bad" if not ascii_mode else "plain"),
                  seg(f"  (last ok {st.get('last_ok_age')}s ago)", "dim")
                  if st.get("last_ok_age") is not None else seg("  (never connected)", "dim"),
                  seg(f"   {tstr}", "dim")])
    else:
        busy = h.get("busy")
        pre = h.get("is_prefilling")
        badge = "IDLE"
        bstyle = "dim"
        if pre:
            badge, bstyle = "PREFILLING", "accent"
        elif busy:
            badge, bstyle = "BUSY", "warn"
        L.append([seg("  \u25cf ONLINE", "good" if not ascii_mode else "plain"),
                  seg("  "), seg(badge, bstyle),
                  seg(f"   {tstr}", "dim")])

    # ---- LLM lane
    L.append([seg("  LLM lane " + "\u2500" * max(4, width - 13), "dim")])
    r = st.get("rates") or {}
    dec = r.get("decode_tps")
    pre_tps = r.get("prefill_tps")
    src = r.get("prefill_src") or ""
    L.append([seg("  decode "), seg(fmt_rate(dec), "good" if dec else "dim"),
              seg("      prefill "), seg(fmt_rate(pre_tps), "accent" if pre_tps else "dim"),
              seg(f"  [{src}]" if src else "", "dim")])

    acc = h.get("mtp_accepted_tokens_total")
    drf = h.get("mtp_drafted_tokens_total")
    overall = (acc / drf) if (acc is not None and drf) else None
    L.append([seg("  MTP overall "),
              seg(bar(overall, ascii_mode=ascii_mode) if overall is not None else "[" + " " * BAR_WIDTH + "]",
                  "good" if (overall or 0) >= 0.5 else ("warn" if (overall or 0) >= 0.3 else "dim")),
              seg(" "), seg(fmt_pct(overall), "plain"),
              seg(f"  ({fmt_num(acc)}/{fmt_num(drf)} accepted)", "dim")])

    L.append([seg(f"  waiting {h.get('requests_waiting', '?')}   "
                  f"completed {fmt_num(h.get('requests_completed_total'))}   "
                  f"failed {fmt_num(h.get('requests_failed_total'))}", "plain")])

    ttft_c = h.get("ttft_seconds_count") or 0
    itl_c = h.get("itl_seconds_count") or 0
    e2e_c = h.get("e2e_seconds_count") or 0
    ttft = h.get("ttft_seconds_sum") / ttft_c if ttft_c else None
    itl = h.get("itl_seconds_sum") / itl_c if itl_c else None
    e2e = h.get("e2e_seconds_sum") / e2e_c if e2e_c else None
    L.append([seg(f"  ttft avg {ttft:.2f}s   itl avg {itl:.3f}s   e2e avg {e2e:.1f}s"
                  if itl is not None else "  ttft/itl/e2e --", "dim")])

    kv = h.get("kv_cache_usage")
    pfx = h.get("prefix_cache_hit_rate")
    ctx_last = h.get("context_last")
    L.append([seg("  kv cache    "), seg(bar(kv, ascii_mode=ascii_mode) if kv is not None else "[--]",
              "good" if (kv or 0) < 0.6 else ("warn" if (kv or 0) < 0.85 else "bad")),
              seg(" "), seg(fmt_pct(kv), "plain")])
    L.append([seg("  prefix hit  "), seg(bar(pfx, ascii_mode=ascii_mode) if pfx is not None else "[--]", "accent"),
              seg(" "), seg(fmt_pct(pfx), "plain")])
    ctx_frac = None
    if ctx_last is not None and ctx_len:
        ctx_frac = min(1.0, ctx_last / ctx_len)
    L.append([seg("  context     "),
              seg(bar(ctx_frac, ascii_mode=ascii_mode) if ctx_frac is not None else "[--]",
                  "good" if (ctx_frac or 0) < 0.7 else ("warn" if (ctx_frac or 0) < 0.9 else "bad")),
              seg(" "), seg(f"{fmt_num(ctx_last)} / {fmt_num(ctx_len)} tok"
                            if ctx_last is not None else "--", "plain")])

    # ---- MTP per-position
    positions = h.get("mtp_accept_by_position") or []
    if positions:
        if st.get("compact"):
            cells = []
            for p in positions[:6]:
                cells.append(f"p{p.get('position')} {fmt_pct(p.get('rate'))}")
            L.append([seg("  MTP pos  ", "dim"), seg("   ".join(cells), "plain")])
        else:
            L.append([seg("  MTP accept by position:", "dim")])
            for p in positions[:8]:
                rate = p.get("rate")
                L.append([seg(f"    pos{p.get('position')}       "),
                          seg(bar(rate, ascii_mode=ascii_mode) if rate is not None else "[--]",
                              "good" if (rate or 0) >= 0.5 else ("warn" if (rate or 0) >= 0.3 else "dim")),
                          seg(" "), seg(fmt_pct(rate), "plain"),
                          seg(f"  ({fmt_num(p.get('tested'))} tested)", "dim")])

    # ---- memory
    L.append([seg("  Memory (GB10 unified) " + "\u2500" * max(4, width - 24), "dim")])
    total, avail = st.get("mem_total"), st.get("mem_avail")
    if total and avail:
        used = total - avail
        frac = used / total
        style = "good" if frac < 0.7 else ("warn" if frac < 0.9 else "bad")
        L.append([seg("  system      "),
                  seg(bar(frac, ascii_mode=ascii_mode), style),
                  seg(" "), seg(f"{frac * 100:.1f}%", "plain"),
                  seg(f"  {fmt_gib(used)} / {fmt_gib(total)} used"
                      f"  ({fmt_gib(avail)} avail)", "dim")])
    else:
        L.append([seg("  system      meminfo unavailable", "dim")])
    gmu = h.get("gpu_memory_utilization")
    if gmu is not None:
        style = "good" if gmu < 0.7 else ("warn" if gmu < 0.9 else "bad")
        L.append([seg("  engine      "),
                  seg(bar(gmu, ascii_mode=ascii_mode), style),
                  seg(" "), seg(f"{gmu * 100:.1f}%", "plain"),
                  seg("  gpu_memory_utilization (engine view)", "dim")])
    L.append([seg("  note: system row uses /proc/meminfo available (honest on coherent UMA);", "dim")])
    L.append([seg("        NVML 'used' on GB10 reports reservation, not allocation.", "dim")])

    # ---- footer
    L.append([seg(f"  poll {st.get('interval', 1):g}s   "
                  f"errors {st.get('errors', 0)}   "
                  f"{'[q] quit' if not st.get('once') else '--once snapshot'}", "dim")])
    return L


# ------------------------------------------------------------ snapshot mode


def run_once(args):
    now = time.time()
    prev = None
    prev_t = None
    errors = 0
    health = None
    try:
        health = fetch_health(args.url, timeout=3.0)
        if args.sample and args.sample > 0:
            prev = {c: health.get(c) for c in COUNTERS}
            prev_t = now
            time.sleep(args.sample)
            now = time.time()
            health = fetch_health(args.url, timeout=3.0)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        print(f"sparkllm: health endpoint unreachable ({exc})", file=sys.stderr)
        health = health  # may be None
    rates = compute_rates(health, prev, prev_t, now) if health else {
        "decode_tps": None, "prefill_tps": None, "prefill_src": None}
    total, avail = read_meminfo()
    st = {"health": health, "rates": rates, "offline": health is None,
          "last_ok_age": None, "errors": errors, "interval": args.interval,
          "now": datetime.now(), "ascii": args.ascii,
          "mem_total": total, "mem_avail": avail,
          "host_uptime": read_host_uptime(), "once": True}
    lines = build_lines(st, 100)
    out = []
    for line in lines:
        out.append("".join(text for text, _ in line).rstrip())
    print("\n".join(out))


# ------------------------------------------------------------ main


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="sparkllm",
        description="LLM engine lane TUI for NVIDIA DGX Spark (pure stdlib).")
    ap.add_argument("-i", "--interval", type=float, default=1.0,
                    help="poll interval in seconds (default 1.0)")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"health endpoint (default {DEFAULT_URL})")
    ap.add_argument("--once", action="store_true",
                    help="print one frame as plain text and exit")
    ap.add_argument("--sample", type=float, default=0.0, metavar="SECONDS",
                    help="with --once: sample counter deltas over SECONDS")
    ap.add_argument("--ascii", action="store_true",
                    help="pure-ASCII bars (for tty1 console fonts)")
    args = ap.parse_args(argv)

    if args.once:
        run_once(args)
        return 0
    try:
        curses.wrapper(curses_main(args))
    except KeyboardInterrupt:
        pass
    return 0


def curses_main(args):
    def inner(stdscr):
        _tui_loop(stdscr, args)
    return inner


def _tui_loop(stdscr, args):
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

    prev = None
    prev_t = None
    errors = 0
    last_ok = None
    host_uptime = read_host_uptime()

    while True:
        now = time.time()
        health = None
        try:
            health = fetch_health(args.url, timeout=max(0.5, args.interval))
            errors = 0
            last_ok = now
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            errors += 1
        rates = {"decode_tps": None, "prefill_tps": None, "prefill_src": None}
        if health is not None:
            rates = compute_rates(health, prev, prev_t, now)
            prev = {c: health.get(c) for c in COUNTERS}
            prev_t = now
        total, avail = read_meminfo()
        st = {"health": health, "rates": rates, "offline": health is None,
              "last_ok_age": int(now - last_ok) if last_ok else None,
              "errors": errors, "interval": args.interval,
              "now": datetime.now(), "ascii": args.ascii,
              "mem_total": total, "mem_avail": avail,
              "host_uptime": host_uptime}
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


if __name__ == "__main__":
    sys.exit(main())