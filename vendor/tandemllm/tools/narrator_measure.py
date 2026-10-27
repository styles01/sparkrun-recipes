#!/usr/bin/env python3
"""P1's n = 1 check of the narrator on the live engine, after the deploy (no hold; stdlib only).

    python3 tools/narrator_measure.py [--server http://127.0.0.1:8000] [--events 20]
        [--since "2026-10-01 21:00"] [--until "2026-10-02 09:30"] [--lifted-ms 3000]
        [--runs 3] [--keep-loaded] [--out results/narrator/measure-<time>.json]

Run it on the box: the conductor routes it calls are writes, which the box itself may make without a
token. It never runs in a hold (it refuses while the box lock is held) and sends one request at a
time, every one bounded.

1 THE LINES. About 20 state events recorded on night 2 (2026-10-01 21:00 to 2026-10-02 09:30, the
  eval night; `--since`/`--until`), read read-only from the usage ledger's `conductor_events`: the
  router's rows (a class and its effort, the packages of its note, the tools its answer called,
  which the tool gate checks) and the session monitor's acts (steering notes, loop and cap closes).
  The state lines were off that night, so the events are rebuilt from those rows as states.py makes
  them. They go through `POST /v1/conductor/narrator/try` twice: at the role's budget (what a user
  would see; a line not finished in time is the template) and at `--lifted-ms` (what the model
  writes when it may finish: the lines to read by eye, and the latency a line needs). Printed: the
  template, the line, its milliseconds and why it fell back; then p50 and p90 per pass.
2 QWEN'S DECODE. One fixed prompt, greedy, a client reasoning budget of 384 tokens (so a cap line
  goes out in the middle of the decode) and 1,200 tokens in all, through two header-only profiles
  (copies of `coding` without clients): `narr-off` (state lines on, narrator off) and `narr-on`
  (state lines and narrator on). `--runs` runs each, alternating, after one warm-up each. The
  decode rate is taken on the client (completion tokens after the first, over the time from the
  first token that is not a state line to the last) and from the server's `timings`. The gate: the
  narrator's mean decode rate within 3 % of the template's; the TTFT difference is printed (the
  router's line is narrated before the prefill).

The narrator is loaded first (`POST /v1/conductor/narrator/load`) and unloaded at the end unless
--keep-loaded; the two profiles stay (reachable only by the header). Exit 0 when the decode gate
holds and the narrator wrote at least one line in each pass that allows it.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import re
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request
import zoneinfo

BASE = "http://127.0.0.1:8000"
OPENCODE_UA = "opencode/1.18.33 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"
LEDGER = "~/.qwen38-spark-engine/usage/ledger.sqlite3"
LOCK = "~/.qwen38-box.flock"
SENTINEL = "⁠"
FRAME = re.compile(r"\n› ([^\n]*)\n\n")
PROMPT = ("Write a Python function that parses one CSV line with quoted fields (a quote inside a "
          "quoted field is doubled), and three short tests for it. Think it through first.")
GATE = 0.03
EFFORT_WORDS = {"off": "thinking off"}


def call(method, path, body=None, headers=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def held() -> bool:
    p = os.path.expanduser(LOCK)
    if not os.path.exists(p):
        return False
    fd = os.open(p, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


# ------------------------------------------------------------------ 1 the recorded events
def _ms(text: str) -> int:
    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    return int(datetime.datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=tz)
               .timestamp() * 1000)


def _j(v):
    try:
        return json.loads(v) if isinstance(v, str) and v else (v or None)
    except ValueError:
        return None


def recorded(since: str, until: str) -> list[dict]:
    """Every state event night 2's rows imply, in time order: {kind, facts, extra, ts_ms}."""
    db = sqlite3.connect(f"file:{os.path.expanduser(LEDGER)}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM conductor_events WHERE ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
                      (_ms(since), _ms(until))).fetchall()
    out = []
    for r in rows:
        r = dict(r)
        if r.get("task") == "route" and r.get("kind") != "monitor":
            if _j(r.get("stages")) and r.get("class_id"):
                eff = r.get("effort")
                out.append({"kind": "route", "ts_ms": r["ts_ms"], "facts": {
                    "cls": r["class_id"],
                    "effort": EFFORT_WORDS.get(eff, f"{eff} effort") if eff else "default"}})
            for n in _j(r.get("injected")) or []:
                for p in (n.get("parts") or []) if isinstance(n, dict) else []:
                    if p.get("kind") == "package":
                        out.append({"kind": "package", "ts_ms": r["ts_ms"],
                                    "facts": {"pkind": "skill", "name": p.get("name")}})
            names = (_j(r.get("outcome")) or {}).get("tool_names") or []
            if names:
                k = "check" if len(names) == 1 else "checks"
                out.append({"kind": k, "ts_ms": r["ts_ms"],
                            "facts": {} if k == "check" else {"n": len(names)},
                            "extra": {"tools": names[:4]}})
            continue
        if r.get("mode") != "act":
            continue
        d, task, reason = r.get("decision"), r.get("task") or "", r.get("trigger") or ""
        detail = _j(r.get("detail")) or {}
        reason = detail.get("reason") or reason
        if d == "PROMPT_TAIL_NOTE":
            label = {"S1": "gate progress", "S2": "context health", "S3": "skill routing",
                     "S4": "stall watchdog"}.get(task, task)
            out.append({"kind": "steer", "ts_ms": r["ts_ms"], "facts": {"label": label}})
        elif d == "CLOSE_REASONING":
            m = re.search(r"budget\((\d+)\)", json.dumps(r))
            if task in ("route", "client", "reserve") and m:
                out.append({"kind": "cap", "ts_ms": r["ts_ms"],
                            "facts": {"cap": f"{int(m.group(1)):,}"}})
            elif task in ("D1", "D1b"):
                out.append({"kind": "loop", "ts_ms": r["ts_ms"], "facts": {},
                            "extra": {"check": "reasoning loop"}})
            else:
                out.append({"kind": "close", "ts_ms": r["ts_ms"], "facts": {}})
        elif d == "CONTINUE_HIDDEN":
            out.append({"kind": "continue", "ts_ms": r["ts_ms"], "facts": {}})
        elif d in ("REPAIR_CALL", "REGENERATE_CALL", "GIVE_UP"):
            out.append({"kind": {"REPAIR_CALL": "repair", "REGENERATE_CALL": "regenerate",
                                 "GIVE_UP": "give_up"}[d], "ts_ms": r["ts_ms"], "facts": {}})
    return out


def pick(events: list[dict], n: int) -> list[dict]:
    """Up to `n` distinct events, round-robin over the kinds (each kind's most frequent first)."""
    count, first = {}, {}
    for e in events:
        key = json.dumps([e["kind"], e["facts"], e.get("extra")], sort_keys=True)
        count[key] = count.get(key, 0) + 1
        first.setdefault(key, e)
    by_kind: dict[str, list] = {}
    for key in sorted(count, key=lambda k: -count[k]):
        by_kind.setdefault(first[key]["kind"], []).append(first[key])
    out = []
    while len(out) < n and any(by_kind.values()):
        for k in list(by_kind):
            if by_kind[k] and len(out) < n:
                out.append(by_kind[k].pop(0))
    return [{k: v for k, v in e.items() if k in ("kind", "facts", "extra")} for e in out]


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * q))], 1) if xs else None


def lines(events: list[dict], budget_ms: int | None, label: str) -> dict:
    body = {"events": events, "max_tokens": 24}
    if budget_ms:
        body["budget_ms"] = budget_ms
    st, out = call("POST", "/v1/conductor/narrator/try", body, timeout=600)
    if st != 200:
        raise SystemExit(f"[narr] try ({label}): {st} {out}")
    got = out["lines"]
    print(f"\n[narr] {label}: {len(got)} events", flush=True)
    for x in got:
        print(f"  {x.get('kind', '?'):10s} {x.get('ms') or 0:7.1f} ms  {x.get('tokens') or 0:2d} tok "
              f"{(x.get('why') or 'line'):8s} | {x.get('template')}\n"
              f"  {'':10s} -> {x.get('line') or ('(template) ' + str(x.get('raw') or ''))}",
              flush=True)
    ms = [x["ms"] for x in got if x.get("ms") is not None]
    done = [x for x in got if x.get("line")]
    summary = {"events": len(got), "lines": len(done), "ms_p50": pct(ms, 0.5),
               "ms_p90": pct(ms, 0.9), "why": {}}
    for x in got:
        summary["why"][x.get("why") or "line"] = summary["why"].get(x.get("why") or "line", 0) + 1
    print(f"[narr] {label}: {summary['lines']} of {len(got)} written by the narrator; per line "
          f"p50 {summary['ms_p50']} ms, p90 {summary['ms_p90']} ms; {summary['why']}", flush=True)
    return {"summary": summary, "lines": got}


# ------------------------------------------------------------------ 2 Qwen's decode
def profiles() -> None:
    st, cur = call("GET", "/v1/conductor/profiles/coding")
    if st != 200:
        raise SystemExit(f"[narr] GET profile coding: {st}")
    for name, narr in (("narr-off", "off"), ("narr-on", "on")):
        doc = json.loads(json.dumps(cur["doc"]))
        doc.update(description=f"narrator_measure: coding, state lines on, narrator {narr}; "
                               f"reachable by header only", clients=[],
                   ui={"states": "on", "narrator": narr})
        st, w = call("PUT", f"/v1/conductor/profiles/{name}",
                     {"doc": doc, "note": f"narrator_measure: narrator {narr}"})
        if st != 200:
            raise SystemExit(f"[narr] PUT profile {name}: {st} {w}")
        print(f"[narr] profile {name} v{w['version']}", flush=True)


def one(profile: str, run: int) -> dict:
    body = {"model": "TandemLLM", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": PROMPT}], "temperature": 0, "seed": 7,
            "max_tokens": 1200, "max_reasoning_tokens": 384,
            "reasoning_format": "reasoning_content"}
    hdr = {"User-Agent": OPENCODE_UA, "X-Tandem-Profile": profile,
           "x-session-id": f"narr-{profile}-{run}-{int(time.time())}"}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **hdr})
    t0 = time.monotonic()
    first = last = None
    reasoning, usage, timings = [], None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            ln = raw.decode().strip()
            if not ln.startswith("data: {"):
                continue
            ev = json.loads(ln[6:])
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if not piece:
                    continue
                reasoning.append(d.get("reasoning_content") or "")
                if FRAME.fullmatch(piece):
                    continue                       # a state line, not one of Qwen's tokens
                now = time.monotonic()
                first = first if first is not None else now
                last = now
    said = FRAME.findall("".join(reasoning))
    n = int((usage or {}).get("completion_tokens") or 0)
    span = (last - first) if first is not None and last is not None else 0.0
    return {"profile": profile, "run": run, "completion_tokens": n,
            "ttft_ms": round((first - t0) * 1e3, 1) if first is not None else None,
            "client_tps": round((n - 1) / span, 3) if span > 0 and n > 1 else None,
            "server_tps": (timings or {}).get("predicted_per_second"),
            "lines": said, "narrated": sum(1 for s in said if s.endswith(SENTINEL))}


def decode(runs: int) -> dict:
    profiles()
    rows = []
    for p in ("narr-off", "narr-on"):
        w = one(p, -1)
        print(f"[narr] warm-up {p}: {w['completion_tokens']} tok, lines {w['lines']}", flush=True)
    for i in range(runs):
        for p in (("narr-off", "narr-on") if i % 2 == 0 else ("narr-on", "narr-off")):
            r = one(p, i)
            rows.append(r)
            print(f"[narr] {p:8s} run {i}: {r['completion_tokens']} tok, client "
                  f"{r['client_tps']} tok/s, server {r['server_tps']} tok/s, ttft {r['ttft_ms']} "
                  f"ms, lines {len(r['lines'])} ({r['narrated']} narrated)", flush=True)
    out = {"rows": rows}
    for p in ("narr-off", "narr-on"):
        xs = [r for r in rows if r["profile"] == p]
        out[p] = {k: round(statistics.mean(r[k] for r in xs if r[k] is not None), 3)
                  if any(r[k] is not None for r in xs) else None
                  for k in ("client_tps", "server_tps", "ttft_ms")}
    off, on = out["narr-off"]["client_tps"], out["narr-on"]["client_tps"]
    out["delta"] = round(on / off - 1.0, 4) if off and on else None
    out["gate"] = out["delta"] is not None and out["delta"] >= -GATE
    out["narrated"] = sum(r["narrated"] for r in rows if r["profile"] == "narr-on")
    print(f"\n[narr] decode: off {out['narr-off']}, on {out['narr-on']}; client delta "
          f"{out['delta']:+.2%} (gate: within {GATE:.0%}) -> {'PASS' if out['gate'] else 'FAIL'}; "
          f"ttft on-off {(out['narr-on']['ttft_ms'] or 0) - (out['narr-off']['ttft_ms'] or 0):+.0f} "
          f"ms; {out['narrated']} narrated lines in the on runs", flush=True)
    return out


def main(argv=None) -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=BASE)
    ap.add_argument("--events", type=int, default=20)
    ap.add_argument("--since", default="2026-10-01 21:00")
    ap.add_argument("--until", default="2026-10-02 09:30")
    ap.add_argument("--lifted-ms", type=int, default=3000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--keep-loaded", action="store_true")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    BASE = a.server.rstrip("/")
    if held():
        print("[narr] the box lock is held (a hold runs): not now", flush=True)
        return 2
    st, s = call("GET", "/v1/conductor/narrator/status")
    if st != 200 or not s.get("enabled"):
        print(f"[narr] no narrator on this engine: {st} {s}", flush=True)
        return 2
    events = pick(recorded(a.since, a.until), a.events)
    print(f"[narr] {len(events)} events from {a.since} to {a.until}: "
          f"{sorted({e['kind'] for e in events})}", flush=True)
    st, d = call("POST", "/v1/conductor/narrator/load")
    if st not in (200, 202):
        print(f"[narr] load refused: {st} {d}", flush=True)
        return 3
    t0 = time.time()
    while time.time() - t0 < 300:
        st, s = call("GET", "/v1/conductor/narrator/status")
        if s["slot"]["state"] in ("loaded", "failed"):
            break
        time.sleep(2)
    slot = s["slot"]
    print(f"[narr] {slot['name']}: {slot['state']} in {slot.get('load_s')} s, "
          f"{(slot.get('device_bytes') or 0) / 2**30:.2f} GiB on the device, MemAvailable "
          f"-{slot.get('took_gib')} GiB, {slot.get('precision')}", flush=True)
    if slot["state"] != "loaded":
        return 3
    report = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "slot": slot, "events": events,
              "budget": lines(events, None, f"at the role's budget ({s['settings']['budget_ms']} ms)"),
              "lifted": lines(events, a.lifted_ms, f"lifted to {a.lifted_ms} ms")}
    if not a.skip_decode:
        report["decode"] = decode(a.runs)
    if not a.keep_loaded:
        call("POST", "/v1/conductor/narrator/unload")
    out = a.out or os.path.join("results", "narrator",
                                f"measure-{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[narr] report: {out}", flush=True)
    ok = report["lifted"]["summary"]["lines"] > 0 and (a.skip_decode or report["decode"]["gate"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
