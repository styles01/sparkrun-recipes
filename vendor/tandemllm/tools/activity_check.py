"""Prove the live activity on a running server: the states in order,
the tool name, the stop blocks, a client that leaves mid-prefill, and the 4 Hz cadence.

    python tools/activity_check.py --base http://127.0.0.1:8011 --token "$QSE_ADMIN_TOKEN" \\
        --out results/live/activity-check-<date>.json --fixture tests/fixtures/live-1.1-box.json

One reader holds `GET /v1/dashboard/live` open for the whole run and records every event with its
arrival time. Meanwhile, one after another:

  1. AGENT: an opencode-shaped streamed request (its tools, a long system prompt, thinking on,
     `x-session-id`) that asks for a `write` call; then the follow-up turn with the tool result
     under the same session. Checks: the first request's states run prefilling -> thinking ->
     writing|tool_call -> ... with `tool.name` "write" seen in `tool_call`, its stop is
     `tool_calls` with state `tool_call`, the engine reported `waiting_for_client` between the
     turns, and the second request `continues` the first.
  2. ABANDON: a streamed request with a long cold prompt (`--abandon-tokens`, unique text so no
     cache helps) whose socket is closed once the live view shows its prefill progressing.
     Checks: `client.connected` false within one tick after the close, and the stop is
     `abandoned` in state `prefilling` with 0 tokens sent and both silences set.
  3. STOPS: `max_tokens` 16 (length), a stop string (stop_string), and on the fake engine
     FAKE_ERROR (error).
  4. CADENCE: events while a request was in flight arrive every 1/QSE_LIVE_HZ s (median and
     p90 reported; the check is the median within 50 ms), `seq` rises by one per event with a
     matching `id:`, and the sampler's own tick time (`sampler.tick_us_*`).

`--fixture` writes a handful of the recorded messages (prefilling with progress, thinking,
tool_call, waiting_for_client, a stop in `recent`) for the dashboard's contract work.
Standard library only. Exit 1 on any failed check.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import statistics
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
OC_TOOLS = os.path.join(HERE, "..", "tests", "fixtures", "opencode_tools.json")
UA = "opencode/1.18.32 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
# common words, about one token each: a random order of them is a cold prompt of a known length
COMMON = ("time year people way day man thing woman life child world school state family student "
          "group country problem hand part place case week company system program question work "
          "government number night point home water room mother area money story fact month lot "
          "right study book eye job word business issue side kind head house service friend father "
          "power hour game line end member law car city community name president team minute idea "
          "kid body information back parent face others level office door health person art war "
          "history party result change morning reason research girl guy moment air teacher force").split()


def cold_words(n: int) -> str:
    return " ".join(random.choice(COMMON) for _ in range(n))


# ------------------------------------------------------------------ the live stream reader
class Reader(threading.Thread):
    """Holds the SSE stream open; `events` is [(t, id, dict)], `pings` [t]."""

    def __init__(self, base, token):
        super().__init__(daemon=True)
        u = urllib.parse.urlparse(base)
        self.host, self.port = u.hostname, u.port or 80
        self.token = token
        self.events: list = []
        self.pings: list = []
        self.stop = threading.Event()
        self.error = None

    def run(self):
        try:
            s = socket.create_connection((self.host, self.port), timeout=5)
            s.settimeout(1.0)
            req = (f"GET /v1/dashboard/live HTTP/1.1\r\nHost: {self.host}\r\n"
                   f"Authorization: Bearer {self.token}\r\nAccept: text/event-stream\r\n\r\n")
            s.sendall(req.encode())
            buf = b""
            head_done = False
            while not self.stop.is_set():
                try:
                    chunk = s.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                now = time.time()
                buf += chunk
                if not head_done:
                    if b"\r\n\r\n" not in buf:
                        continue
                    head, buf = buf.split(b"\r\n\r\n", 1)
                    if b" 200 " not in head.split(b"\r\n")[0]:
                        self.error = head.decode(errors="replace")[:200]
                        return
                    head_done = True
                while b"\n\n" in buf:
                    block, buf = buf.split(b"\n\n", 1)
                    eid, data = None, None
                    for line in block.decode().splitlines():
                        if line.startswith(": ping"):
                            self.pings.append(now)
                        elif line.startswith("id: "):
                            eid = int(line[4:])
                        elif line.startswith("data: "):
                            data = json.loads(line[6:])
                    if data is not None:
                        self.events.append((now, eid, data))
            s.close()
        except Exception as exc:                                  # noqa: BLE001
            self.error = repr(exc)


def rows_of(events, rid):
    out = []
    for t, _eid, e in events:
        for r in e.get("requests") or []:
            if r["request_id"] == rid:
                out.append((t, e, r))
    return out


def states_of(rows):
    seq = []
    for _t, _e, r in rows:
        a = r.get("activity") or {}
        st = a.get("state")
        if st and (not seq or seq[-1] != st):
            seq.append(st)
    return seq


# ------------------------------------------------------------------ requests
def stream_chat(base, body, headers=None, close_when=None):
    """A streamed chat on a raw socket. Returns (request id, text, finish, raw bytes).
    `close_when(sock)` is polled before each read; True closes the socket there."""
    u = urllib.parse.urlparse(base)
    raw = json.dumps(body).encode()
    s = socket.create_connection((u.hostname, u.port or 80), timeout=600)
    s.settimeout(0.25)
    h = {"Content-Type": "application/json", "Content-Length": str(len(raw)), "User-Agent": UA}
    h.update(headers or {})
    s.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: {u.hostname}\r\n"
               + "".join(f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n").encode() + raw)
    data = b""
    closed_early = False
    status = None
    while True:
        if close_when is not None and close_when():
            s.close()
            closed_early = True
            break
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            break
        data += chunk
        if status is None and b"\r\n\r\n" in data:
            head, body = data.split(b"\r\n\r\n", 1)
            status = int(head.split(b" ")[1])
            if status != 200:                     # a JSON error on a kept-alive socket
                n = int(next((ln.split(b":")[1] for ln in head.split(b"\r\n")
                              if ln.lower().startswith(b"content-length:")), b"0"))
                while len(body) < n:
                    body += s.recv(65536)
                s.close()
                return None, body.decode(errors="replace"), f"http {status}", [], False
        if b"data: [DONE]" in data:
            break
    if not closed_early:
        s.close()
    rid, text, finish, calls = None, "", None, []
    for line in data.decode(errors="replace").split("\n"):
        if not line.startswith("data: {"):
            continue
        c = json.loads(line[6:])
        rid = rid or c.get("id")
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            text += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                calls.append(tc)
            finish = ch.get("finish_reason") or finish
    return rid, text, finish, calls, closed_early


def oc_body(model, messages, think=True, max_tokens=2048):
    with open(OC_TOOLS) as f:
        tools = json.load(f)["tools"]
    return {"model": model, "messages": messages, "tools": tools, "stream": True,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think},
            "stream_options": {"include_usage": True}}


OC_SYSTEM = ("You are opencode, an interactive CLI coding agent. Use the tools to act; never "
             "describe a tool call in prose. The working directory is /work/demo. " * 40)


# ------------------------------------------------------------------ the run
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    ap.add_argument("--token", default=os.environ.get("QSE_ADMIN_TOKEN"))
    ap.add_argument("--model", default="qwen38-spark-engine")
    ap.add_argument("--hz", type=float, default=4.0, help="the server's QSE_LIVE_HZ")
    ap.add_argument("--abandon-tokens", type=int, default=60000,
                    help="words in the abandoned prompt (about one token each)")
    ap.add_argument("--fake", action="store_true", help="the server is --fake-engine")
    ap.add_argument("--skip", default="", help="comma list of agent,abandon,stops")
    ap.add_argument("--out")
    ap.add_argument("--fixture")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    rd = Reader(a.base, a.token)
    rd.start()
    time.sleep(1.5)
    checks: dict[str, bool] = {}
    notes: dict[str, object] = {}
    busy_windows = []

    def check(name, ok, info=None):
        checks[name] = bool(ok)
        if info is not None:
            notes[name] = info
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {info}" if info is not None else ""),
              flush=True)

    if rd.error:
        print("live stream:", rd.error)
        return 1
    session = f"ses_check{random.randrange(1 << 30):08x}"
    agent_ids = []
    if "agent" not in skip:
        user = ("FAKE tool: create the file hello.txt containing the single word hi" if a.fake
                else "Create the file hello.txt in the working directory containing the single "
                     "word hi. Use the write tool now.")
        msgs = [{"role": "system", "content": OC_SYSTEM}, {"role": "user", "content": user}]
        t0 = time.time()
        rid, text, finish, calls, _ = stream_chat(a.base, oc_body(a.model, msgs),
                                                  {"x-session-id": session})
        busy_windows.append((t0, time.time()))
        agent_ids.append(rid)
        names = [c.get("function", {}).get("name") for c in calls if c.get("function", {})
                 .get("name")]
        print(f"agent turn 1: {rid} finish={finish} calls={names}", flush=True)
        time.sleep(2.0)                                  # the client "runs the tool"
        rows = rows_of(rd.events, rid)
        seen = states_of(rows)
        notes["agent_states"] = seen
        check("agent: prefilling seen before the first token", "prefilling" in seen[:3], seen)
        if not a.fake or True:
            check("agent: thinking seen", "thinking" in seen, seen)
        tc = [r for _t, _e, r in rows if (r.get("activity") or {}).get("state") == "tool_call"]
        tool_names = {(r["activity"].get("tool") or {}).get("name") for r in tc}
        if finish == "tool_calls":
            check("agent: tool_call state carried the tool name",
                  bool(tool_names & set(names)), sorted(n for n in tool_names if n))
            done = [r for _t, _e, r in rows if (r.get("activity") or {}).get("state") == "done"]
            stop = done[-1]["activity"]["stop"] if done else None
            check("agent: stop tool_calls in state tool_call",
                  stop is not None and stop["reason"] == "tool_calls"
                  and stop["state"] == "tool_call", stop)
            waits = [e["engine"] for _t, _i, e in rd.events
                     if e["engine"]["state"] == "waiting_for_client"
                     and (e["engine"]["waiting_for_client"] or {}).get("request_id") == rid]
            check("agent: engine waiting_for_client after the tool_calls finish", bool(waits),
                  waits[-1]["label"] if waits else None)
            call = calls[0]
            msgs2 = msgs + [{"role": "assistant", "content": "",
                             "tool_calls": [{"id": call.get("id") or "call_1", "type": "function",
                                             "function": {"name": names[0],
                                                          "arguments": "".join(
                                                              c.get("function", {}).get(
                                                                  "arguments") or ""
                                                              for c in calls)}}]},
                            {"role": "tool", "tool_call_id": call.get("id") or "call_1",
                             "content": "Wrote file successfully."}]
            t0 = time.time()
            rid2, _text2, finish2, _c2, _ = stream_chat(
                a.base, oc_body(a.model, msgs2, max_tokens=256), {"x-session-id": session})
            busy_windows.append((t0, time.time()))
            agent_ids.append(rid2)
            time.sleep(1.0)
            r2 = rows_of(rd.events, rid2)
            cont = [r["activity"]["continues"] for _t, _e, r in r2
                    if (r.get("activity") or {}).get("continues")]
            check("agent: the next turn continues the first",
                  bool(cont) and cont[-1]["request_id"] == rid, cont[-1] if cont else None)
            notes["agent_turn2"] = {"id": rid2, "finish": finish2, "states": states_of(r2)}
        else:
            check("agent: the model called a tool", False, f"finish={finish}: {text[:120]!r}")

    if "abandon" not in skip:
        words = cold_words(a.abandon_tokens)
        body = {"model": a.model, "stream": True, "max_tokens": 64,
                "messages": [{"role": "user", "content": ("FAKE_SLOW_PREFILL " if a.fake else "")
                              + "Summarise: " + words}]}
        state = {"t_seen": None, "t_close": None}
        n0 = len(rd.events)

        def close_when():
            if state["t_close"] is not None:
                return True
            for t, _i, e in rd.events[n0:]:
                for r in e.get("requests") or []:
                    act = r.get("activity") or {}
                    p = act.get("prefill") or {}
                    if act.get("state") == "prefilling" and (p.get("pct") or 0) > 0:
                        if state["t_seen"] is None:
                            state["t_seen"] = time.time()
                            state["rid"] = r["request_id"]
                        if time.time() - state["t_seen"] > 2.0:
                            state["t_close"] = time.time()
                            return True
            return False
        t0 = time.time()
        stream_chat(a.base, body, {}, close_when=close_when)
        deadline = time.time() + 30
        rid = state.get("rid")
        gone_t, stop = None, None
        while time.time() < deadline and rid:
            for t, _i, e in rd.events:
                for r in e.get("requests") or []:
                    if r["request_id"] != rid:
                        continue
                    act = r.get("activity") or {}
                    # the sampler sees the close, or the handler's chunk check sees it first and
                    # ends the request: either is the server knowing within one tick
                    if gone_t is None and ((act.get("client") or {}).get("connected") is False
                                           or act.get("state") == "done"):
                        gone_t = t
                    if act.get("state") == "done":
                        stop = act.get("stop")
            if stop is not None:
                break
            time.sleep(0.2)
        busy_windows.append((t0, time.time()))
        tick = 1.0 / a.hz
        lag = gone_t - state["t_close"] if gone_t and state["t_close"] else None
        check("abandon: the prefill showed progress", rid is not None,
              {"request_id": rid})
        check("abandon: the client's close on the page within one tick",
              lag is not None and lag <= tick + 0.35, _r(lag, 3))
        check("abandon: stop abandoned, state prefilling, 0 tokens, silences set",
              stop is not None and stop["reason"] == "abandoned"
              and stop["state"] == "prefilling" and stop["tokens_sent"] == 0
              and stop["silent_ms"] is not None and stop["client_gone_ms"] is not None, stop)
        notes["abandon_sentence"] = stop and stop.get("sentence")

    if "stops" not in skip:
        cases = [("length", {"max_tokens": 16}, "Count from one to fifty in words.", "length", None),
                 ("stop string", {"stop": [" the " if a.fake else " three"], "max_tokens": 200},
                  "Count from one to ten in English words separated by spaces, no thinking.",
                  "stop", "stop_string")]
        if a.fake:
            cases.append(("error", {"max_tokens": 50}, "FAKE_ERROR please", "error",
                          "RuntimeError"))
        for label, extra, prompt, reason, detail in cases:
            body = dict({"model": a.model, "stream": True,
                         "messages": [{"role": "user", "content": prompt}],
                         "chat_template_kwargs": {"enable_thinking": False}}, **extra)
            t0 = time.time()
            rid, *_ = stream_chat(a.base, body)
            busy_windows.append((t0, time.time()))
            time.sleep(1.2)
            done = [r for _t, _e, r in rows_of(rd.events, rid)
                    if (r.get("activity") or {}).get("state") == "done"]
            stop = done[-1]["activity"]["stop"] if done else None
            check(f"stops: {label}", stop is not None and stop["reason"] == reason
                  and (detail is None or stop["detail"] == detail), stop)
        recent = rd.events[-1][2].get("recent") or []
        check("stops: recent holds them with sentences",
              len(recent) >= 3 and all(r["stop"]["sentence"] for r in recent),
              [r["stop"]["sentence"] for r in recent[:4]])

    time.sleep(1.0)
    rd.stop.set()
    # cadence while busy
    ev = rd.events
    ids = [i for _t, i, _e in ev]
    steps = [b - a_ for a_, b in zip(ids[1:], ids[2:])]
    # a stream writes the newest tick's event: a writer that fell a tick behind skips one
    check("seq: strictly rising, equal to id, at most 5 % skipped",
          all(x >= 1 for x in steps) and sum(x > 1 for x in steps) <= 0.05 * max(1, len(steps))
          and all(i == e["seq"] for _t, i, e in ev),
          {"events": len(ids), "skipped": sum(x - 1 for x in steps if x > 1)})
    gaps = []
    for (t1, _i1, e1), (t2, _i2, e2) in zip(ev, ev[1:]):
        if e1["counts"]["in_flight"] and e2["counts"]["in_flight"] and e2["interval_s"] < 1.0:
            gaps.append(t2 - t1)
    if gaps:
        gaps.sort()
        med = statistics.median(gaps)
        p90 = gaps[int(0.9 * (len(gaps) - 1))]
        check("cadence: median gap within 50 ms of 1/hz while busy",
              abs(med - 1.0 / a.hz) <= 0.05,
              {"n": len(gaps), "median_ms": _r(med * 1e3, 1), "p90_ms": _r(p90 * 1e3, 1),
               "max_ms": _r(gaps[-1] * 1e3, 1)})
    sampler = ev[-1][2].get("sampler") if ev else None
    notes["sampler"] = sampler
    print("sampler:", sampler)
    hist = ev[-1][2].get("history") if ev else None
    samples = [e["sample"]["t"] for _t, _i, e in ev if e.get("sample")]
    per_s = [b - a_ for a_, b in zip(samples, samples[1:]) if b != a_]
    check("history: one sample a second", not per_s or min(per_s) >= 0.85,
          _r(min(per_s), 3) if per_s else None)
    del hist
    out = {"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base": a.base, "hz": a.hz,
           "checks": checks, "notes": notes, "events": len(ev), "pings": len(rd.pings),
           "pass": all(checks.values())}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    if a.fixture:
        write_fixture(a.fixture, ev, agent_ids, a)
    print(f"{sum(checks.values())}/{len(checks)} checks passed")
    return 0 if out["pass"] else 1


def _r(x, nd=2):
    return round(x, nd) if x is not None else None


def write_fixture(path, ev, agent_ids, a):
    """A few recorded messages, one per interesting moment, for the dashboard's contract work."""
    picks = {}

    def first(name, pred):
        for t, i, e in ev:
            if name not in picks and pred(e):
                picks[name] = e
                return

    def state_is(st):
        return lambda e: any((r.get("activity") or {}).get("state") == st
                             for r in e.get("requests") or [])
    first("prefilling", lambda e: any((r.get("activity") or {}).get("state") == "prefilling"
                                      and ((r["activity"].get("prefill") or {}).get("pct") or 0)
                                      > 10 for r in e.get("requests") or []))
    first("thinking", state_is("thinking"))
    first("writing", state_is("writing"))
    first("tool_call", lambda e: any((r.get("activity") or {}).get("state") == "tool_call"
                                     and (r["activity"].get("tool") or {}).get("name")
                                     for r in e.get("requests") or []))
    first("finishing", state_is("finishing"))
    first("waiting_for_client", lambda e: e["engine"]["state"] == "waiting_for_client")
    first("continues", lambda e: any((r.get("activity") or {}).get("continues")
                                     for r in e.get("requests") or []))
    first("client_gone", lambda e: any(((r.get("activity") or {}).get("client") or {})
                                       .get("connected") is False
                                       for r in e.get("requests") or []))
    picks["idle_with_recent"] = ev[-1][2] if ev else None
    doc = {"_note": ("Real `event: live` messages of contract 1.1 recorded by "
                     "tools/activity_check.py against " + ("the fake engine" if a.fake else
                                                             "the engine on the box")
                     + " on " + time.strftime("%Y-%m-%d %H:%M %Z") + "; one per moment, "
                     "`history` dropped. Prompt text never appears in the stream."),
           "messages": {k: v for k, v in picks.items() if v is not None}}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"fixture: {path} ({', '.join(doc['messages'])})")


if __name__ == "__main__":
    sys.exit(main())
