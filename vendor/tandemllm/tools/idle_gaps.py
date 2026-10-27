#!/usr/bin/env python3
"""How long the engine sits idle between a session's requests, and after which tool (OPS-49).

    # on the Mac, where opencode keeps its sessions: the steps of the chosen sessions, names and
    # times only (no prompt, no output, no command), read-only
    python tools/idle_gaps.py steps --opencode-db ~/.local/share/opencode/opencode.db \\
        --session ses_... [--session ...] --out steps.jsonl

    # anywhere with a COPY of the usage ledger (sqlite3 ledger.sqlite3 ".backup copy.sqlite3")
    python tools/idle_gaps.py report --ledger copy.sqlite3 --steps steps.jsonl \\
        [--group "pagoda night=ses_a" --group "eval round 1=ses_b,ses_c"] [--json OUT] [--md OUT]

Harmony R0 (Harmony 3, "How much idle time a real session has"). A gap is the next request's
arrival minus this request's end (`ts_ms + total_ms`, the ledger's own clock), per session, so
the gap is what opencode on the Mac spends running the tool the model called and sending the
next request. The gap's kind is the previous request's step in opencode: the tool that ran
longest in it, mapped to read, edit, write, bash, browser, screenshot, checker or other;
`turn_end` when the step called no tool (the next request is a new user message, a person or
the eval's driver); `compaction` when the step was opencode's compaction request; `unknown`
when no opencode step matched the ledger row. A bash or browser call whose output carries a
known checker's RESULT line (server/conductor/signals.py) is a `checker`.

Matching: opencode's clock (the Mac) and the ledger's (the box) differ by a few hundred ms; the
offset is the median difference of the nearest pairs, then every step takes the nearest
unmatched ledger row within --tolerance seconds. Ledger rows in a session's window that no step
claimed (opencode's title request, another client) belong to no session: when one runs inside a
session's gap, that part of the gap is counted as busy (`busy_other_s`), not idle.

Read-only by construction: both databases are opened with `mode=ro` and never written; the
production ledger itself is refused (read a `.backup` copy). Standard library only.

Output: per group and overall, the histogram (0-0.5, 0.5-2, 2-5, 5-15, 15-60, over 60 s; count
and idle seconds), p50 / p90 / p99 per kind and overall (nearest rank), the gaps above 5 s and
above 20 s per clock hour and per session hour, idle seconds per session hour, the long-gap
share, and how many gaps each Harmony job would fit (decider 21 ms, clerk 1 / 5 / 10 s, note
writer 6 / 12 / 20 s).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sqlite3
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import signals  # noqa: E402

PRODUCTION = "~/.qwen38-spark-engine/usage/ledger.sqlite3"
BUCKETS = ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 15.0), (15.0, 60.0), (60.0, math.inf))
KINDS = ("read", "edit", "write", "bash", "browser", "screenshot", "checker", "other",
         "turn_end", "compaction", "unknown")
LONG = (5.0, 20.0)
AFTER = (1.0, 2.0, 5.0, 10.0, 20.0, 60.0)
# Harmony 3's jobs: what each needs of one gap, in seconds
JOBS = (("decider", 0.021), ("clerk 1 s", 1.0), ("clerk 5 s", 5.0), ("clerk 10 s", 10.0),
        ("note writer 6 s", 6.0), ("note writer 12 s", 12.0), ("note writer 20 s", 20.0))
EDIT_TOOLS = ("edit", "multiedit", "apply_patch", "patch")
BROWSER_PREFIX = ("chrome-devtools_", "playwright_", "browser_", "puppeteer_")


def bucket_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g} s" if hi != math.inf else f"over {lo:g} s"


def pct(vals: list[float], q: float) -> float | None:
    """Nearest rank: the smallest value with at least q of the values at or below it."""
    if not vals:
        return None
    v = sorted(vals)
    return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))]


def ro(path: str) -> sqlite3.Connection:
    """A read-only connection; nothing this tool does can write the file."""
    p = os.path.realpath(os.path.expanduser(path))
    if not os.path.exists(p):
        raise SystemExit(f"no such file: {p}")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


# ------------------------------------------------------------------ opencode's steps

def tool_kind(tool: dict) -> str:
    name = str(tool.get("tool") or "").lower()
    if tool.get("checker"):
        return "checker"
    if "screenshot" in name:
        return "screenshot"
    if name.startswith(BROWSER_PREFIX) or name == "webfetch":
        return "browser"
    if name in EDIT_TOOLS:
        return "edit"
    if name in ("read", "write", "bash"):
        return name
    return "other"


def step_kind(step: dict | None) -> str:
    """The kind of the gap after this step: what opencode did before its next request."""
    if step is None:
        return "unknown"
    if step.get("compaction"):
        return "compaction"
    tools = step.get("tools") or []
    if not tools:
        return "turn_end"
    longest = max(tools, key=lambda t: (t.get("end") or 0) - (t.get("start") or 0))
    return tool_kind(longest)


def read_steps(db: str, sessions: list[str]) -> list[dict]:
    """Every assistant message (one engine request each) of the sessions, with its tools' names
    and times; a tool's output is read only to see whether a checker's RESULT line is in it."""
    conn = ro(db)
    out = []
    for sid in sessions:
        srow = conn.execute("SELECT title, directory FROM session WHERE id = ?", (sid,)).fetchone()
        if srow is None:
            raise SystemExit(f"no opencode session {sid}")
        tools: dict[str, list] = {}
        for mid, data in conn.execute(
                "SELECT message_id, data FROM part WHERE session_id = ? ORDER BY id", (sid,)):
            p = json.loads(data)
            if p.get("type") != "tool":
                continue
            st = p.get("state") or {}
            t = st.get("time") or {}
            res = signals.checker_results(str(st.get("output") or ""))
            tools.setdefault(mid, []).append({
                "tool": p.get("tool"), "start": t.get("start"), "end": t.get("end"),
                "status": st.get("status"),
                "checker": next((r["checker"] for r in res if r["checker"] != "unknown"), None)})
        for mid, created, data in conn.execute(
                "SELECT id, time_created, data FROM message WHERE session_id = ? "
                "ORDER BY time_created, id", (sid,)):
            m = json.loads(data)
            if m.get("role") != "assistant":
                continue
            t = m.get("time") or {}
            out.append({"session": sid, "title": srow[0], "id": mid,
                        "created": t.get("created") or created, "completed": t.get("completed"),
                        "compaction": m.get("mode") == "compaction" or m.get("summary") is True,
                        "finish": m.get("finish"), "tools": tools.get(mid, [])})
    conn.close()
    return out


# ------------------------------------------------------------------ the ledger

def read_ledger(path: str, t0_ms: int, t1_ms: int) -> list[dict]:
    if os.path.realpath(os.path.expanduser(path)) == os.path.realpath(
            os.path.expanduser(PRODUCTION)):
        raise SystemExit("refusing the live ledger: read a copy (sqlite3 ledger.sqlite3 "
                         "\".backup copy.sqlite3\")")
    conn = ro(path)
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    keys = {}
    if "conductor_events" in have:
        keys = dict(conn.execute("SELECT request_id, session_key FROM conductor_events WHERE "
                                 "kind = 'route' AND ts_ms BETWEEN ? AND ?", (t0_ms, t1_ms)))
    rows = []
    for rid, ts, total, client, kind, ptok in conn.execute(
            "SELECT request_id, ts_ms, total_ms, client_id, client_kind, prompt_tokens "
            "FROM requests WHERE ts_ms BETWEEN ? AND ? ORDER BY ts_ms", (t0_ms, t1_ms)):
        rows.append({"request_id": rid, "ts": ts / 1000.0,
                     "end": (ts + float(total or 0.0)) / 1000.0, "client_id": client,
                     "client_kind": kind, "prompt_tokens": ptok, "session_key": keys.get(rid)})
    conn.close()
    return rows


def match(steps: list[dict], rows: list[dict], tolerance_s: float = 1.5) -> tuple[dict, float]:
    """step id -> ledger row index, and the clock offset (ledger minus opencode, seconds)."""
    ts = [r["ts"] for r in rows]
    diffs = []
    for s in steps:
        c = s["created"] / 1000.0
        near = min(ts, key=lambda x: abs(x - c), default=None)
        if near is not None and abs(near - c) < 5.0:
            diffs.append(near - c)
    off = statistics.median(diffs) if diffs else 0.0
    used, out = set(), {}
    for s in steps:
        c = s["created"] / 1000.0 + off
        best = None
        for i, r in enumerate(rows):
            if i in used or abs(r["ts"] - c) > tolerance_s:
                continue
            if best is None or abs(r["ts"] - c) < abs(rows[best]["ts"] - c):
                best = i
        if best is not None:
            used.add(best)
            out[s["id"]] = best
    return out, off


def _overlap(a0: float, a1: float, spans: list[tuple[float, float]]) -> float:
    """Seconds of [a0, a1] covered by the union of `spans`."""
    cut = sorted((max(a0, s), min(a1, e)) for s, e in spans if e > a0 and s < a1)
    tot, cur0, cur1 = 0.0, None, None
    for s, e in cut:
        if cur1 is None or s > cur1:
            if cur1 is not None:
                tot += cur1 - cur0
            cur0, cur1 = s, e
        else:
            cur1 = max(cur1, e)
    if cur1 is not None:
        tot += cur1 - cur0
    return tot


def session_gaps(reqs: list[dict], others: list[dict], step_of: dict) -> list[dict]:
    """The gaps of one session: between each request's end and the next one's arrival. The last
    request has no successor and so no gap. `others`: ledger rows of no session or of another
    session, whose time inside a gap is busy, not idle."""
    reqs = sorted(reqs, key=lambda r: r["ts"])
    spans = [(o["ts"], o["end"]) for o in others]
    out = []
    for a, b in zip(reqs, reqs[1:]):
        gap = b["ts"] - a["end"]
        busy = _overlap(a["end"], b["ts"], spans) if gap > 0 else 0.0
        step = step_of.get(a["request_id"])
        kind = step_kind(step) if step_of else "unknown"
        out.append({"at": a["end"], "gap_s": round(max(gap, 0.0), 3),
                    "idle_s": round(max(gap, 0.0) - busy, 3), "busy_other_s": round(busy, 3),
                    "overlap": gap < 0, "kind": kind, "compaction": kind == "compaction",
                    # the compaction request itself (cold prefill plus the summary) is busy time;
                    # its length is kept beside the gap after it
                    "request_s": round(a["end"] - a["ts"], 3),
                    "request_id": a["request_id"], "next_request_id": b["request_id"]})
    return out


# ------------------------------------------------------------------ the figures

def figures(gaps: list[dict], sessions: list[dict]) -> dict:
    """The histogram, percentiles, per-hour counts, idle per session hour and what fits."""
    g = [x["gap_s"] for x in gaps]
    idle = [x["idle_s"] for x in gaps]
    hist = []
    for lo, hi in BUCKETS:
        sel = [x for x in gaps if lo <= x["gap_s"] < hi]
        hist.append({"bucket": bucket_label(lo, hi), "lo": lo, "hi": None if hi == math.inf else hi,
                     "count": len(sel), "idle_s": round(sum(x["idle_s"] for x in sel), 1)})
    per_kind = {}
    for k in KINDS:
        v = [x["gap_s"] for x in gaps if x["kind"] == k]
        if v:
            rest = [x - 1.0 for x in v if x > 1.0]
            per_kind[k] = {"n": len(v), "p50": pct(v, 0.5), "p90": pct(v, 0.9),
                           "p99": pct(v, 0.99), "max": max(v), "idle_s": round(sum(
                               x["idle_s"] for x in gaps if x["kind"] == k), 1),
                           # the broker's rule per kind: once such a gap has lasted 1 s
                           "after_1s": {"gaps": len(rest), "p50_more_s": pct(rest, 0.5),
                                        "p10_more_s": pct(rest, 0.1)}}
    wall_s = sum(s["wall_s"] for s in sessions)
    busy_s = sum(s["busy_s"] for s in sessions)
    hours = wall_s / 3600.0 if wall_s else 0.0
    idle_total = sum(idle)
    clock: dict[str, dict] = {}
    for x in gaps:
        h = _dt.datetime.fromtimestamp(x["at"]).strftime("%Y-%m-%d %H:00")
        c = clock.setdefault(h, {"gaps": 0, "over_5s": 0, "over_20s": 0, "idle_s": 0.0})
        c["gaps"] += 1
        c["over_5s"] += x["gap_s"] > LONG[0]
        c["over_20s"] += x["gap_s"] > LONG[1]
        c["idle_s"] = round(c["idle_s"] + x["idle_s"], 1)
    fits = {}
    for name, need in JOBS:
        sel = [x for x in gaps if x["idle_s"] >= need]
        fits[name] = {"need_s": need, "gaps": len(sel),
                      "per_session_hour": round(len(sel) / hours, 1) if hours else None,
                      "share_of_gaps": round(len(sel) / len(gaps), 4) if gaps else None}
    # the broker's question: a gap has lasted t seconds; how much more is it likely to last?
    after = {}
    for t in AFTER:
        rest = [x["gap_s"] - t for x in gaps if x["gap_s"] > t]
        after[f"{t:g}"] = {"gaps": len(rest), "p50_more_s": pct(rest, 0.5),
                           "p10_more_s": pct(rest, 0.1),
                           "share_lasting_20s_more": round(sum(1 for v in rest if v >= 20) /
                                                           len(rest), 4) if rest else None}
    return {
        "sessions": len(sessions), "requests": sum(s["requests"] for s in sessions),
        "gaps": len(gaps), "wall_hours": round(hours, 3), "busy_s": round(busy_s, 1),
        "idle_s": round(idle_total, 1), "busy_other_s": round(sum(x["busy_other_s"]
                                                                  for x in gaps), 1),
        "idle_min_per_session_hour": round(idle_total / 60.0 / hours, 2) if hours else None,
        "mean_gap_s": round(statistics.mean(g), 3) if g else None,
        "overall": {"p50": pct(g, 0.5), "p90": pct(g, 0.9), "p99": pct(g, 0.99)},
        "histogram": hist, "per_kind": per_kind,
        "over_5s": sum(1 for v in g if v > LONG[0]), "over_20s": sum(1 for v in g if v > LONG[1]),
        "over_5s_per_session_hour": round(sum(1 for v in g if v > LONG[0]) / hours, 1)
        if hours else None,
        "over_20s_per_session_hour": round(sum(1 for v in g if v > LONG[1]) / hours, 1)
        if hours else None,
        "long_share": {f"over_{int(t)}s": {
            "gaps": round(sum(1 for v in g if v > t) / len(g), 4) if g else None,
            "idle_time": round(sum(x["idle_s"] for x in gaps if x["gap_s"] > t) / idle_total, 4)
            if idle_total else None} for t in LONG},
        "per_clock_hour": dict(sorted(clock.items())),
        "compaction_gaps": [x["gap_s"] for x in gaps if x["compaction"]],
        "compaction_requests_s": [x["request_s"] for x in gaps if x["compaction"]],
        "after": after,
        "overlaps": sum(1 for x in gaps if x["overlap"]),
        "fits": fits}


def analyse(rows: list[dict], steps: list[dict], groups: dict[str, list[str]] | None = None,
            tolerance_s: float = 1.5) -> dict:
    """Ledger rows and opencode steps -> the report. Without steps, the sessions are the
    conductor's session keys (else the client id) and every kind is unknown."""
    by_sess: dict[str, list[dict]] = {}
    for s in steps:
        by_sess.setdefault(s["session"], []).append(s)
    sess_rows: dict[str, list[dict]] = {}
    step_of: dict[str, dict] = {}
    titles, offsets = {}, {}
    if steps:
        claimed = set()
        for sid, ss in by_sess.items():
            ss.sort(key=lambda s: s["created"])
            t0 = ss[0]["created"] / 1000.0 - 120
            t1 = max((s.get("completed") or s["created"]) for s in ss) / 1000.0 + 120
            cand = [i for i, r in enumerate(rows) if t0 <= r["ts"] <= t1 and i not in claimed]
            got, off = match(ss, [rows[i] for i in cand], tolerance_s)
            offsets[sid] = round(off, 3)
            titles[sid] = ss[0].get("title")
            idx = {s["id"]: s for s in ss}
            for step_id, j in got.items():
                i = cand[j]
                claimed.add(i)
                sess_rows.setdefault(sid, []).append(rows[i])
                step_of[rows[i]["request_id"]] = idx[step_id]
    else:
        for r in rows:
            sess_rows.setdefault(r["session_key"] or f"client:{r['client_id']}", []).append(r)
    per_session, all_gaps = {}, {}
    for sid, reqs in sess_rows.items():
        mine = {r["request_id"] for r in reqs}
        lo, hi = min(r["ts"] for r in reqs), max(r["end"] for r in reqs)
        others = [r for r in rows if r["request_id"] not in mine and r["end"] > lo and r["ts"] < hi]
        gaps = session_gaps(reqs, others, step_of)
        all_gaps[sid] = gaps
        unmatched = len(by_sess.get(sid, [])) - len(reqs) if steps else None
        per_session[sid] = {"title": titles.get(sid), "requests": len(reqs),
                            "steps_unmatched": unmatched, "clock_offset_s": offsets.get(sid),
                            "start": _dt.datetime.fromtimestamp(lo).isoformat(timespec="seconds"),
                            "wall_s": round(hi - lo, 1),
                            "busy_s": round(sum(r["end"] - r["ts"] for r in reqs), 1)}
    out = {"sessions": {}, "groups": {}, "overall": None}
    for sid, s in per_session.items():
        out["sessions"][sid] = dict(s, **figures(all_gaps[sid], [s]))
    for name, sids in (groups or {}).items():
        sids = [x for x in sids if x in per_session]
        out["groups"][name] = dict(figures([g for x in sids for g in all_gaps[x]],
                                           [per_session[x] for x in sids]), members=sids)
    out["overall"] = figures([g for gs in all_gaps.values() for g in gs],
                             list(per_session.values()))
    out["gaps"] = {sid: gs for sid, gs in all_gaps.items()}
    return out


def to_md(rep: dict) -> str:
    """The tables for the Memo note: one histogram and one percentile table per group."""
    def f(v, nd=2):
        return "—" if v is None else f"{v:,.{nd}f}" if isinstance(v, float) else f"{v:,}"
    lines = []
    groups = list(rep["groups"].items()) or [("all sessions", rep["overall"])]
    lines += ["| group | sessions | requests | gaps | wall h | idle min per session h | mean gap s "
              "| p50 | p90 | p99 | gaps > 5 s per h | gaps > 20 s per h | idle time in gaps > 5 s "
              "| idle time in gaps > 20 s |", "|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|"]
    for name, g in groups:
        ls = g["long_share"]
        lines.append(f"| {name} | {g['sessions']} | {g['requests']:,} | {g['gaps']:,} | "
                     f"{f(g['wall_hours'])} | {f(g['idle_min_per_session_hour'], 1)} | "
                     f"{f(g['mean_gap_s'])} | {f(g['overall']['p50'])} | {f(g['overall']['p90'])} "
                     f"| {f(g['overall']['p99'])} | {f(g['over_5s_per_session_hour'], 1)} | "
                     f"{f(g['over_20s_per_session_hour'], 1)} | "
                     f"{f((ls['over_5s']['idle_time'] or 0) * 100, 1)} % | "
                     f"{f((ls['over_20s']['idle_time'] or 0) * 100, 1)} % |")
    lines += ["", "Histogram (count / idle seconds):", "",
              "| group | " + " | ".join(b["bucket"] for b in groups[0][1]["histogram"]) + " |",
              "|-|" + "-:|" * len(BUCKETS)]
    for name, g in groups:
        lines.append(f"| {name} | " + " | ".join(f"{b['count']:,} / {b['idle_s']:,.0f}"
                                                 for b in g["histogram"]) + " |")
    for name, g in groups:
        lines += ["", f"Per kind, {name} (gap seconds):", "",
                  "| kind | n | p50 | p90 | p99 | max | idle s | lasted 1 s: n, median / p10 more |",
                  "|-|-:|-:|-:|-:|-:|-:|-:|"]
        for k, v in g["per_kind"].items():
            a1 = v["after_1s"]
            more = (f"{a1['gaps']:,}: {f(a1['p50_more_s'], 1)} / {f(a1['p10_more_s'], 1)} s"
                    if a1["gaps"] else "—")
            lines.append(f"| {k} | {v['n']:,} | {f(v['p50'])} | {f(v['p90'])} | {f(v['p99'])} | "
                         f"{f(v['max'])} | {v['idle_s']:,.0f} | {more} |")
        if g["compaction_requests_s"]:
            lines.append(f"\nCompaction requests (busy, cold prefill plus the summary): "
                         f"{len(g['compaction_requests_s'])}, "
                         f"{sum(g['compaction_requests_s']) / 60:.1f} min in all; the gap after "
                         f"each: {', '.join(f'{v:g}' for v in g['compaction_gaps'])} s")
        if g["wall_hours"] >= 2:
            lines += ["", f"Per clock hour, {name}:", "",
                      "| hour | gaps | over 5 s | over 20 s | idle s |", "|-|-:|-:|-:|-:|"]
            for h, c in g["per_clock_hour"].items():
                lines.append(f"| {h} | {c['gaps']:,} | {c['over_5s']:,} | {c['over_20s']:,} | "
                             f"{c['idle_s']:,.0f} |")
    lines += ["", "A gap has lasted t seconds: how much longer (median, p10) and the share lasting 20 s "
              "more:", "", "| group | " + " | ".join(f"after {t:g} s" for t in AFTER) + " |",
              "|-|" + "-:|" * len(AFTER)]
    for name, g in groups:
        cells = []
        for t in AFTER:
            v = g["after"][f"{t:g}"]
            cells.append("—" if not v["gaps"] else
                         f"{v['gaps']:,}: {f(v['p50_more_s'], 1)} / {f(v['p10_more_s'], 1)} s, "
                         f"{(v['share_lasting_20s_more'] or 0) * 100:.0f} %")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "What fits (gaps whose idle time is at least the job's):", "",
              "| group | " + " | ".join(n for n, _ in JOBS) + " |", "|-|" + "-:|" * len(JOBS)]
    for name, g in groups:
        lines.append(f"| {name} | " + " | ".join(
            f"{g['fits'][n]['gaps']:,} ({f(g['fits'][n]['per_session_hour'], 1)}/h)"
            for n, _ in JOBS) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ CLI

def _groups(specs: list[str]) -> dict[str, list[str]]:
    out = {}
    for s in specs or ():
        name, _, ids = s.partition("=")
        out[name.strip()] = [x.strip() for x in ids.split(",") if x.strip()]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("steps", help="opencode's steps of the sessions, as JSON lines")
    s.add_argument("--opencode-db", default="~/.local/share/opencode/opencode.db")
    s.add_argument("--session", action="append", required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("report", help="the gaps from a ledger copy and the steps")
    r.add_argument("--ledger", required=True, help="a copy of the usage ledger (never the live file)")
    r.add_argument("--steps", default=None, help="the steps file; without it, kinds are unknown")
    r.add_argument("--since", default=None, help="ledger only: ISO start (local time)")
    r.add_argument("--until", default=None, help="ledger only: ISO end (local time)")
    r.add_argument("--group", action="append", default=[], help='"name=ses_a,ses_b"')
    r.add_argument("--tolerance", type=float, default=1.5)
    r.add_argument("--json", default=None)
    r.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "steps":
        steps = read_steps(a.opencode_db, a.session)
        with open(a.out, "w") as f:
            for st in steps:
                f.write(json.dumps(st) + "\n")
        print(f"{len(steps)} steps of {len(a.session)} sessions -> {a.out}")
        return 0
    steps = []
    if a.steps:
        with open(a.steps) as f:
            steps = [json.loads(x) for x in f if x.strip()]
    if steps:
        t0 = min(s["created"] for s in steps) - 600_000
        t1 = max((s.get("completed") or s["created"]) for s in steps) + 600_000
    else:
        if not (a.since and a.until):
            raise SystemExit("without --steps, give --since and --until")
        t0 = int(_dt.datetime.fromisoformat(a.since).timestamp() * 1000)
        t1 = int(_dt.datetime.fromisoformat(a.until).timestamp() * 1000)
    rows = read_ledger(a.ledger, t0, t1)
    rep = analyse(rows, steps, _groups(a.group), a.tolerance)
    rep["ledger"] = os.path.basename(a.ledger)
    rep["created_at"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    md = to_md(rep)
    print(md)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    if a.md:
        with open(a.md, "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
