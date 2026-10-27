#!/usr/bin/env python3
"""N agent-like sessions at once through one OpenAI-compatible server (the proxy, usually),
spread over its backends, and what they measured: per backend and in total.

    python tools/parallel_load.py --base http://127.0.0.1:18700 \\
        --mix deepseek-flash=8,fake-a=20,TandemLLM=2 --turns 3 --budget-usd 1.0 \\
        --board --out /tmp/run.json

Each session is a small coding agent: a system prompt, two tools (`read_file`, `run_tests`), a
task, and up to `--turns` streamed turns. A turn that ends in tool calls gets a short tool result
and goes on; a turn that ends in text gets the next instruction. Sessions run at once (one thread
each) and start `--stagger` seconds apart.

A model whose id starts with `fake` is the fake backend (tests/fake_backend.py): its user turns
carry the words that steer it (a speed between 25 and 120 tokens a second, an answer length, a
tool call every other turn, some reasoning, a TTFT), so its lanes look like real ones.

MEASURED HERE, per turn, off the stream the client got: TTFT (send to the first token frame),
token frames, the decode rate ((completion tokens - 1) / (last frame - first frame), the engine's
formula), `usage` (prompt, cached, completion), the cost at the price list's rates
(proxy/prices.seed.json, DeepSeek's off-peak half by the UTC hour). Totals: completion tokens over
the run's wall time per backend and in all; the most turns in flight at once.

BUDGET: `--budget-usd` stops new turns once the cost so far plus the worst case of the turns in
flight (max_tokens at the output price, the prompt at the input price) would pass it.

`--board`: reads `/v1/dashboard/parallel` once a second during the run and compares its totals
with the client's own count: per second, the token frames the client received (all streams) beside
the board's series for the same second.

Keys: `--key-env NAME` reads a bearer token from the environment (never an argument, never
printed). The report holds no prompt text and no key.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import http.client
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = [
    "Add input validation to the `parse_port` function in config.py and a test for it.",
    "The test `test_retry_backoff` is flaky. Find out why and fix it.",
    "Rename the module `utils/strings.py` to `utils/text.py` and update every import.",
    "Write a function that merges two sorted lists of intervals, with tests.",
    "Explain what the `Inflight` class in relay.py does and suggest one improvement.",
    "Add a `--dry-run` flag to the CLI that prints what would be deleted.",
    "Find the off-by-one error in the pagination code and fix it.",
    "Make the cache eviction LRU instead of FIFO; keep the public API.",
    "Add type hints to every function in scheduler.py.",
    "Write a small benchmark for the JSON encoder and report the numbers.",
]
FOLLOW_UPS = [
    "Good. Now run the tests and tell me what fails.",
    "Keep it shorter. What is the one change you would make first?",
    "Show only the diff of that change.",
    "Check the edge case where the input is empty.",
]
SYSTEM = ("You are a coding agent working in a small Python repository. Use the tools when you "
          "need a file's content or a test run. Keep answers short: a plan of at most three "
          "steps, then the change.")
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite, or one test file.",
        "parameters": {"type": "object", "properties": {"target": {"type": "string"}}}}},
]
FILE_TEXT = ("def parse_port(value):\n    return int(value)\n\n\ndef retry(fn, tries=3):\n"
             "    for i in range(tries):\n        try:\n            return fn()\n"
             "        except OSError:\n            time.sleep(2 ** i)\n")
TEST_TEXT = "3 passed, 1 failed: test_parse_port_rejects_zero (AssertionError: 0 accepted)"


def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[k]


def _r(x, nd=2):
    return round(x, nd) if x is not None else None


def load_prices(path: str) -> dict:
    """{model id: price row} from the seed file (every provider)."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except OSError:
        return {}
    out = {}
    for _prov, models in (doc.get("providers") or {}).items():
        for mid, row in models.items():
            out[mid] = row
    return out


def peak(at: float) -> bool:
    t = _dt.datetime.fromtimestamp(at, _dt.timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def cost_usd(price: dict | None, usage: dict, at: float, deepseek: bool) -> float | None:
    if not price or usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
        return None
    if deepseek and price.get("offpeak") and not peak(at):
        price = dict(price, **{k: v for k, v in price["offpeak"].items() if v is not None})
    pin, pout = price.get("input"), price.get("output")
    if pin is None or pout is None:
        return None
    cached = usage.get("cached") or 0
    pc = price.get("cached_input")
    pc = pin if pc is None else pc
    return ((usage["prompt_tokens"] - cached) * pin + cached * pc
            + usage["completion_tokens"] * pout) / 1e6


class Budget:
    def __init__(self, usd: float | None):
        self.cap = usd
        self.spent = 0.0
        self.reserved = 0.0
        self.lock = threading.Lock()
        self.stopped = False

    def take(self, worst: float) -> bool:
        if self.cap is None:
            return True
        with self.lock:
            if self.spent + self.reserved + worst > self.cap:
                self.stopped = True
                return False
            self.reserved += worst
            return True

    def settle(self, worst: float, cost: float | None) -> None:
        with self.lock:
            if self.cap is not None:
                self.reserved = max(0.0, self.reserved - worst)
            self.spent += cost or 0.0


class Run:
    def __init__(self, a):
        self.a = a
        u = urllib.parse.urlsplit(a.base)
        self.host, self.port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
        self.https = u.scheme == "https"
        self.prefix = u.path.rstrip("/")
        self.key = os.environ.get(a.key_env) if a.key_env else None
        self.prices = load_prices(a.prices)
        self.budget = Budget(a.budget_usd)
        self.turns: list[dict] = []
        self.frames: list[tuple] = []          # (wall time, model) of every token frame
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.deadline = None
        self.board: list[dict] = []
        self.board_series: dict = {}
        self.stop_board = threading.Event()

    def conn(self, timeout=600.0):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout)

    def headers(self, session: str) -> dict:
        h = {"Content-Type": "application/json", "User-Agent": "parallel-load/1",
             "x-session-id": session}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    # ----------------------------------------------------------------- one turn
    def turn(self, model: str, session: str, messages: list, rnd: random.Random, i: int,
             max_tokens: int | None = None, tools: bool = True) -> dict:
        a = self.a
        mt = max_tokens or a.max_tokens
        body = {"model": model, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True}, "max_tokens": mt}
        if tools:
            body["tools"] = TOOLS
        if model in a.no_think:
            body["thinking"] = {"type": "disabled"}
        price = self.prices.get(model)
        prompt_est = sum(len(json.dumps(m)) for m in messages) // 3 + 200
        worst = 0.0
        if price:
            worst = (prompt_est * (price.get("input") or 0) + mt *
                     (price.get("output") or 0)) / 1e6
        out = {"model": model, "session": session, "turn": i, "ok": False, "status": None,
               "ttft_ms": None, "frames": 0, "decode_tps": None, "usage": None, "cost": None,
               "finish": None, "tool_calls": 0, "reasoning_frames": 0, "error": None}
        if not self.budget.take(worst):
            out["error"] = "budget"
            return out
        with self.lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        t0 = time.perf_counter()
        t_wall0 = time.time()
        first = last = None
        content, reasoning, calls = [], [], {}
        usage = None
        try:
            c = self.conn()
            c.request("POST", self.prefix + "/v1/chat/completions", json.dumps(body),
                      self.headers(session))
            r = c.getresponse()
            out["status"] = r.status
            if r.status != 200:
                out["error"] = f"http {r.status}: {r.read()[:200].decode('utf-8', 'replace')}"
                c.close()
                return out
            while True:
                line = r.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                if obj.get("error"):
                    out["error"] = str(obj["error"])[:200]
                token = False
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content"):
                        reasoning.append(d["reasoning_content"])
                        token = True
                        out["reasoning_frames"] += 1
                    if d.get("content"):
                        content.append(d["content"])
                        token = True
                    for tc in d.get("tool_calls") or []:
                        token = True
                        k = tc.get("index", 0)
                        slot = calls.setdefault(k, {"id": tc.get("id") or f"call_{k}",
                                                    "name": "", "args": ""})
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
                    if ch.get("finish_reason"):
                        out["finish"] = ch["finish_reason"]
                if token:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    out["frames"] += 1
                    with self.lock:
                        self.frames.append((time.time(), model))
            c.close()
        except (OSError, http.client.HTTPException) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        finally:
            with self.lock:
                self.inflight -= 1
            cost = None
            if usage is not None:
                det = usage.get("prompt_tokens_details") or {}
                cached = usage.get("prompt_cache_hit_tokens", det.get("cached_tokens"))
                u = {"prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"), "cached": cached,
                     "reasoning": (usage.get("completion_tokens_details") or {}).get(
                         "reasoning_tokens")}
                out["usage"] = u
                cost = cost_usd(price, u, t_wall0, model.startswith("deepseek"))
                out["cost"] = cost
            self.budget.settle(worst, cost)
        if first is not None:
            out["ttft_ms"] = _r((first - t0) * 1e3, 1)
            n = (out["usage"] or {}).get("completion_tokens")
            if n and last > first:
                out["decode_tps"] = _r((n - 1) / (last - first))
        out["tool_calls"] = len(calls)
        out["ok"] = out["error"] is None and out["finish"] is not None
        out["elapsed_ms"] = _r((time.perf_counter() - t0) * 1e3, 1)
        out["_assistant"] = ("".join(content), [calls[k] for k in sorted(calls)])
        return out

    # ----------------------------------------------------------------- one session
    def session(self, idx: int, model: str) -> None:
        a = self.a
        rnd = random.Random(a.seed * 1000 + idx)
        sid = f"load-{idx:02d}"
        conv = 0
        while True:
            task = TASKS[(idx + conv) % len(TASKS)]
            fake = model.startswith("fake")
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": task + (self.fake_words(rnd, 0) if fake else "")}]
            for i in range(a.turns):
                if self.deadline is not None and time.time() > self.deadline and i == 0:
                    return
                res = self.turn(model, f"{sid}-{conv}", msgs, rnd, i)
                text, calls = res.pop("_assistant", ("", []))
                with self.lock:
                    self.turns.append(res)
                if res["error"] == "budget" or not res["ok"]:
                    if res["error"] == "budget":
                        return
                    break
                if calls:
                    msgs.append({"role": "assistant", "content": text or None, "tool_calls": [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                        for c in calls]})
                    for c in calls:
                        msgs.append({"role": "tool", "tool_call_id": c["id"],
                                     "content": FILE_TEXT if c["name"] == "read_file"
                                     else TEST_TEXT})
                else:
                    msgs.append({"role": "assistant", "content": text[:4000]})
                    nxt = FOLLOW_UPS[(i + idx) % len(FOLLOW_UPS)]
                    msgs.append({"role": "user", "content": nxt + (self.fake_words(rnd, i + 1)
                                                                   if fake else "")})
            conv += 1
            if self.deadline is None or time.time() > self.deadline:
                return

    def fake_words(self, rnd: random.Random, i: int) -> str:
        w = [f"FAKE_TPS {rnd.choice((25, 40, 60, 80, 120))}",
             f"FAKE_LONG {rnd.randint(60, 360)}",
             f"FAKE_SLOW_TTFT {rnd.choice((0.05, 0.2, 0.6, 1.2))}"]
        if i % 2 == 1:
            w.append("FAKE_TOOL")
        if rnd.random() < 0.5:
            w.append("FAKE_REASONING")
        return " " + " ".join(w)

    # ----------------------------------------------------------------- the board
    def watch(self) -> None:
        since = None
        while not self.stop_board.is_set():
            try:
                c = self.conn(timeout=10)
                q = f"?since={since}" if since is not None else ""
                h = {"User-Agent": "parallel-load/1"}
                if self.key:
                    h["Authorization"] = f"Bearer {self.key}"
                c.request("GET", self.prefix + "/v1/dashboard/parallel" + q, headers=h)
                r = c.getresponse()
                body = r.read()
                c.close()
                if r.status == 200:
                    doc = json.loads(body)
                    t = doc["totals"]
                    self.board.append({"t": doc["at"], "in_flight": t["in_flight"],
                                       "decode_tps": t["decode_tps"],
                                       "decode_unreported": t["decode_unreported"],
                                       "build_us": doc.get("build_us", {}).get("last"),
                                       "bytes": len(body),
                                       "backends": {b["id"]: b["decode_tps"]
                                                    for b in doc["backends"]}})
                    for s in doc.get("series") or []:
                        self.board_series[s["t"]] = s["by"]
                        since = s["t"] if since is None or s["t"] > since else since
            except (OSError, ValueError, http.client.HTTPException):
                pass
            self.stop_board.wait(1.0)

    # ----------------------------------------------------------------- the run
    def go(self) -> dict:
        a = self.a
        plan = []
        for part in a.mix.split(","):
            model, _, n = part.partition("=")
            plan += [model.strip()] * int(n or 1)
        for model in dict.fromkeys(plan):
            # two short answers per model first: the board checks that a backend's frames are
            # tokens before it counts them live (server/parallel.py)
            for k in range(2 * a.warmup):
                tool = k >= a.warmup          # then two that call a tool (checked on their own)
                fake = model.startswith("fake")
                text = ("Read the file config.py with the tool, nothing else." if tool else
                        "Say hello in five words.")
                if fake:
                    text += " FAKE_LONG 12" + (" FAKE_TOOL" if tool else "")
                msgs = ([{"role": "system", "content": SYSTEM}] if tool else []) + \
                    [{"role": "user", "content": text}]
                res = self.turn(model, f"warmup-{k}", msgs, random.Random(k), 0,
                                max_tokens=160 if tool else 64, tools=tool)
                res.pop("_assistant", None)
                res["warmup"] = True
                self.turns.append(res)
        if a.duration:
            self.deadline = time.time() + a.duration
        wt = None
        if a.board:
            wt = threading.Thread(target=self.watch, daemon=True)
            wt.start()
        t0 = time.time()
        ths = []
        for idx, model in enumerate(plan):
            th = threading.Thread(target=self.session, args=(idx, model), daemon=True)
            th.start()
            ths.append(th)
            time.sleep(a.stagger)
        for th in ths:
            th.join()
        wall = time.time() - t0
        time.sleep(1.5 if a.board else 0)
        self.stop_board.set()
        if wt is not None:
            wt.join(timeout=5)
        return self.report(plan, wall, t0)

    def report(self, plan, wall, t0) -> dict:
        by: dict[str, dict] = {}
        warm = [t for t in self.turns if t.get("warmup")]
        for t in self.turns:
            if t["error"] == "budget" or t.get("warmup"):
                continue
            b = by.setdefault(t["model"], {"turns": 0, "ok": 0, "errors": 0, "tool_turns": 0,
                                           "ttft": [], "tps": [], "completion": 0,
                                           "prompt": 0, "cached": 0, "cost": 0.0,
                                           "cost_known": 0, "frames": 0,
                                           "frames_vs_tokens": []})
            b["turns"] += 1
            b["ok"] += int(t["ok"])
            b["errors"] += int(not t["ok"])
            b["tool_turns"] += int(t["tool_calls"] > 0)
            b["ttft"].append(t["ttft_ms"])
            b["tps"].append(t["decode_tps"])
            b["frames"] += t["frames"]
            u = t["usage"] or {}
            b["completion"] += u.get("completion_tokens") or 0
            b["prompt"] += u.get("prompt_tokens") or 0
            b["cached"] += u.get("cached") or 0
            if t["cost"] is not None:
                b["cost"] += t["cost"]
                b["cost_known"] += 1
            if u.get("completion_tokens"):
                b["frames_vs_tokens"].append((t["frames"], u["completion_tokens"]))
        per = {}
        for m, b in by.items():
            fv = b.pop("frames_vs_tokens")
            per[m] = {"sessions": plan.count(m), "turns": b["turns"], "ok": b["ok"],
                      "errors": b["errors"], "tool_turns": b["tool_turns"],
                      "ttft_p50_ms": _pct(b["ttft"], 0.5), "ttft_p90_ms": _pct(b["ttft"], 0.9),
                      "decode_tps_p50": _pct(b["tps"], 0.5),
                      "decode_tps_mean": _r(statistics.mean([x for x in b["tps"] if x])
                                            if any(b["tps"]) else None),
                      "completion_tokens": b["completion"], "prompt_tokens": b["prompt"],
                      "cached_tokens": b["cached"],
                      "cache_hit": _r(b["cached"] / b["prompt"], 4) if b["prompt"] else None,
                      "throughput_tps": _r(b["completion"] / wall),
                      "cost_usd": _r(b["cost"], 6) if b["cost_known"] else None,
                      "frames": b["frames"],
                      "frames_eq_tokens": (sum(1 for f, n in fv if abs(n - f) <= 2 + 0.01 * n),
                                           len(fv))}
        total_c = sum(p["completion_tokens"] for p in per.values())
        out = {"base": self.a.base, "mix": self.a.mix, "sessions": len(plan),
               "turns_per_session": self.a.turns, "max_tokens": self.a.max_tokens,
               "started": _dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
               "wall_s": _r(wall, 1), "peak_in_flight": self.peak,
               "turns": sum(p["turns"] for p in per.values()),
               "errors": sum(p["errors"] for p in per.values()),
               "completion_tokens": total_c, "throughput_tps": _r(total_c / wall),
               "cost_usd": _r(sum(p["cost_usd"] or 0 for p in per.values())
                              + sum(t["cost"] or 0 for t in warm), 6),
               "warmup_turns": len(warm),
               "budget_usd": self.a.budget_usd, "budget_stopped": self.budget.stopped,
               "per_model": per}
        # the client's own per-second count beside the board's series
        if self.a.board:
            # each board sample counts the tokens since the sample before it: the client's frames
            # in the same interval, per second
            import bisect
            ft = sorted(t for t, _m in self.frames)
            ts = sorted(self.board_series)
            pairs = []
            for t_prev, t in zip(ts, ts[1:]):
                if not 0.5 < t - t_prev < 2.0:
                    continue
                n = bisect.bisect_right(ft, t) - bisect.bisect_right(ft, t_prev)
                pairs.append((n / (t - t_prev), sum(self.board_series[t].values())))
            top = max((x for x, _ in pairs), default=0)
            steady = [p for p in pairs if top and p[0] >= 0.5 * top]
            rates = [b["decode_tps"] for b in self.board if b["decode_tps"] is not None]
            out["board"] = {
                "polls": len(self.board),
                "peak_in_flight": max((b["in_flight"] for b in self.board), default=None),
                "decode_tps_mean": _r(statistics.mean(rates)) if rates else None,
                "decode_tps_max": max(rates) if rates else None,
                "build_us_max": max((b["build_us"] or 0 for b in self.board), default=None),
                "unreported_max": max((b["decode_unreported"] for b in self.board), default=None),
                "bytes_max": max((b["bytes"] for b in self.board), default=None),
                "series_vs_client": {
                    "seconds": len(steady),
                    "client_frames_per_s_mean": _r(statistics.mean(x for x, _ in steady))
                    if steady else None,
                    "board_tps_mean": _r(statistics.mean(y for _, y in steady)) if steady else None,
                }}
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--mix", required=True, help="model=sessions,... e.g. deepseek-flash=8,fake-a=20")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds: sessions start new conversations until then (0: one each)")
    ap.add_argument("--stagger", type=float, default=0.15)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--prices", default=os.path.join(ROOT, "proxy", "prices.seed.json"))
    ap.add_argument("--no-think", default="", help="models sent thinking disabled (comma list)")
    ap.add_argument("--key-env", default=None, help="an environment variable with a bearer key")
    ap.add_argument("--board", action="store_true", help="read /v1/dashboard/parallel meanwhile")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2,
                    help="short answers per model before the run (the board's frame check)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    a.no_think = {x.strip() for x in a.no_think.split(",") if x.strip()}
    rep = Run(a).go()
    text = json.dumps(rep, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0 if rep["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
