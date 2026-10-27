"""Hours of mixed traffic against a running server, watching for the things a bench cannot see.

The atlas row sends fifty identical-shaped requests to a freshly started process with the caches
off, and it answers one question very well: how fast is a generation. It cannot answer any of the
questions that decide whether this is a thing the operator can leave running:

  * does memory climb -- the state cache, the suffix store, the response cache, the allocator;
  * does the cache budget actually evict, or does it grow until the board is full;
  * do the rates drift as the process ages, and does time-to-first-token drift with them;
  * what happens to a client that arrives while a long generation is in flight;
  * what happens to a client that hangs up half way through a stream;
  * does anything raise, and if it does, does the caller find out.

So the mix here is deliberately unlike the row: five languages of workload, thinking on and off,
single-shot and multi-turn sessions, prompts from forty tokens to twenty-four thousand, several
connections at once, and a fraction of requests that disconnect on purpose. Every request's
outcome is recorded, `/health` is sampled on a timer, and both are reported as a table per
interval so a drift shows up as a trend rather than as a final average.

    python tools/soak.py --base-url http://127.0.0.1:8000/v1 --minutes 120 --concurrency 3

added what a leak verdict needs beyond that. `--csv` writes one row per `/health` sample
(every `--health-every` seconds): the process's RSS, the allocator, the state store's and the
suffix store's counters, and -- when the server runs on this host -- MemFree, MemAvailable and the
page cache from /proc/meminfo. Tool calls (a call, then its result as a `tool` message) and sampled
requests join the mix, and `--long-docs` adds real held-out documents of 8k-32k tokens asked about
twice in one conversation. `--idle-every`/`--idle-for` stop all traffic for a while, as a real day
has gaps. `--probe-docs` measures the rate the same way at the start, every `--probe-every`
seconds and at the end, with the traffic paused and a 2k document the server has never seen
("continue the text", greedy), so an end-against-start comparison is not a comparison of what the
suffix store has learnt; `--server-log` adds each probe's tokens and milliseconds a block from the
server's own `[req]` line.

    python tools/soak.py --base-url http://127.0.0.1:8011/v1 --minutes 35 --concurrency 3 \\
        --health-every 10 --csv results/soak/srv8-h1.csv --json-out results/soak/srv8-h1.json \\
        --long-docs bench/longprompts --probe-docs bench/longprompts --probe-every 600 \\
        --idle-every 420 --idle-for 75 --server-log results/soak/srv8-h1-server.log
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FILLER = (
    "The engine holds one sequence at a time, so the server is honest about that and serialises "
    "requests behind a lock. A block of sixteen rows verifies for about a hundred milliseconds, "
    "and the drafter's own forward costs a quarter of that again. "
)

GERMAN = (
    "Schreibe eine kurze, sachliche Zusammenfassung der folgenden Idee: ein Sprachmodell, das "
    "seine eigenen naechsten Tokens vorschlaegt und sie in einem einzigen Durchlauf pruefen "
    "laesst, spart Bandbreite und nicht Rechenzeit. Erklaere, warum das auf dieser Hardware der "
    "entscheidende Unterschied ist."
)

#: (name, prompt, thinking, max_tokens, turns). `turns` > 1 continues the conversation, which is
#: what exercises the session cache: the second turn's prompt is the first turn's prefix plus glue.
WORKLOADS = [
    ("prose", "Write three paragraphs about why a slow river is a bad metaphor for time.",
     False, 320, 1),
    ("chat", "I have two hours and a tired brain. What should I do with them? Be concrete.",
     False, 256, 3),
    ("code", "Write a Python function that merges overlapping intervals, with tests. Explain the "
     "invariant it maintains.", False, 700, 1),
    ("edit", "Rewrite this to be shorter and less pompous, keeping every fact: 'It is incumbent "
     "upon us to acknowledge that the aforementioned methodology, in its current instantiation, "
     "may be characterised as suboptimal with respect to the efficiency criteria.'", False, 220, 1),
    ("german", GERMAN, False, 400, 2),
    ("think", "A train leaves at 09:14 and arrives at 11:02, having stopped twice for four "
     "minutes each. What was its moving time, and what is the trap in this question?", True, 900, 1),
    ("long", None, False, 300, 1),          # prompt built to ~24k tokens at run time
    ("emoji", "Reply with one line: three emoji with skin-tone modifiers, two flags, and the "
     "word 'Grüße'. Nothing else.", False, 80, 1),
    # a function call and its result, the way an agent client sends them
    ("tool", "What is the weather in Hamburg right now? Use the tool, then answer in one sentence.",
     False, 200, 2),
    # a sampled request (the served default is greedy; clients such as Open WebUI sample)
    ("sampled", "Write a short poem about a lighthouse keeper who collects clocks.", False, 256, 1),
    ("longdoc", None, False, 256, 2),       # a real 8k-32k document, only with --long-docs
]

#: per-workload request fields beyond the defaults
EXTRA = {
    "tool": {"tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]},
    "sampled": {"temperature": 0.7, "top_p": 0.95},
}
TOOL_RESULT = json.dumps({"city": "Hamburg", "temp_c": 14, "sky": "light rain", "wind_kmh": 22})


def post_stream(url: str, body: dict, timeout: float, abandon_after: float = 0.0) -> dict:
    """One streamed chat completion, measured. Returns a record; never raises for a bad answer."""
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    rec = {"t0": time.time(), "ttft": None, "tokens": 0, "chars": 0, "finish": None,
           "status": None, "error": None, "usage": None, "abandoned": False}
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        rec["status"] = resp.status
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                rec["usage"] = d["usage"]
            if d.get("error"):
                rec["error"] = str(d["error"])[:200]
            rec.setdefault("id", d.get("id"))
            for c in d.get("choices", []):
                piece = c["delta"].get("content") or c["delta"].get("reasoning_content")
                for call in c["delta"].get("tool_calls") or []:
                    # A call streams as its arguments, not as content; it is the answer all the
                    # same, and the second turn needs it back whole.
                    calls = rec.setdefault("calls", {})
                    got = calls.setdefault(call.get("index", 0), {"id": None, "name": "",
                                                                   "arguments": ""})
                    got["id"] = call.get("id") or got["id"]
                    fn = call.get("function") or {}
                    got["name"] += fn.get("name") or ""
                    got["arguments"] += fn.get("arguments") or ""
                    piece = piece or fn.get("arguments") or fn.get("name")
                if c["delta"].get("content"):
                    # the answer as the client saw it, for the next turn's assistant message
                    rec["text"] = rec.get("text", "") + c["delta"]["content"]
                if piece:
                    if rec["ttft"] is None:
                        rec["ttft"] = time.time() - rec["t0"]
                    rec["chars"] += len(piece)
                if c.get("finish_reason"):
                    rec["finish"] = c["finish_reason"]
            if abandon_after and time.time() - rec["t0"] > abandon_after:
                # A reader that goes away mid-stream. The server should log it as `abandoned`,
                # release the engine, and be ready for the next request -- not leak the lock.
                rec["abandoned"] = True
                resp.close()
                break
    except urllib.error.HTTPError as exc:
        rec["status"] = exc.code
        try:
            rec["error"] = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            rec["error"] = f"HTTP {exc.code}"
    except Exception as exc:                                       # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["wall"] = time.time() - rec["t0"]
    if rec["usage"]:
        rec["tokens"] = rec["usage"].get("completion_tokens", 0)
    if rec["tokens"] > 1 and rec["ttft"] is not None and rec["wall"] > rec["ttft"]:
        rec["tok_s"] = (rec["tokens"] - 1) / (rec["wall"] - rec["ttft"])
    else:
        rec["tok_s"] = 0.0
    return rec


def meminfo_gb(text: str | None = None) -> dict:
    """MemFree, MemAvailable and the page cache from /proc/meminfo, in GB; {} off this host.

    The GPU allocates from the same pool, and found the page cache is what fills it: the
    weight files the engine read once at load. MemAvailable counts that cache as free."""
    if text is None:
        try:
            text = open("/proc/meminfo").read()
        except OSError:
            return {}
    raw = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemFree", "MemAvailable", "Cached"):
            raw[key] = int(rest.split()[0]) * 1024 / 1e9
    return {"memfree_gb": raw.get("MemFree"), "memavail_gb": raw.get("MemAvailable"),
            "pagecache_gb": raw.get("Cached")}


def sample_row(h: dict, t0: float, mem: dict | None = None) -> dict:
    """One `/health` answer as a flat row: the columns a chart is drawn from.

    The store counters are whatever the store reports (`puts`, `hits`, `declined_short`, ...), so a
    counter added to the server later lands in the CSV without a change here."""
    m = h.get("memory") or {}
    row = {"t": round(h.get("t", time.time()), 1), "elapsed_s": round(h.get("t", time.time()) - t0, 1),
           "rss_gb": m.get("rss_gb"), "allocated_gb": m.get("allocated_gb"),
           "reserved_gb": m.get("reserved_gb"), "max_allocated_gb": m.get("max_allocated_gb")}
    row.update(mem or {})
    for k, v in (h.get("inflight") or {}).items():
        row[f"req_{k}"] = v
    cache = h.get("cache") or {}
    for part, prefix in (("state_store", "store_"), ("suffix_store", "suffix_"),
                         ("response_cache", "resp_")):
        for k, v in (cache.get(part) or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[prefix + k] = v
    return row


def write_csv(path: str, rows: list[dict]) -> None:
    """Every sample, the union of their columns in first-seen order (a counter may appear late)."""
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def trend(rows: list[dict], key: str, window_s: float = 300.0) -> dict | None:
    """Did `key` keep growing? The first and last `window_s` medians, and the least-squares slope
    over the second half of the run in units a ten minutes -- a plateau reads ~0 there however
    much the warm-up climbed, a leak does not."""
    pts = [(r["elapsed_s"], float(r[key])) for r in rows if r.get(key) not in (None, "")]
    if len(pts) < 4:
        return None
    end = pts[-1][0]
    first = [v for t, v in pts if t <= pts[0][0] + window_s]
    last = [v for t, v in pts if t >= end - window_s]
    half = [(t, v) for t, v in pts if t >= end / 2] or pts
    n = len(half)
    mt = sum(t for t, _ in half) / n
    mv = sum(v for _, v in half) / n
    var = sum((t - mt) ** 2 for t, _ in half)
    slope = (sum((t - mt) * (v - mv) for t, v in half) / var * 600.0) if var else 0.0
    return {"first": statistics.median(first), "last": statistics.median(last),
            "min": min(v for _, v in pts), "max": max(v for _, v in pts),
            "slope_per_10min_2nd_half": slope}


def probe_summary(recs: list[dict]) -> dict:
    """Per probe round: the median rate and, where the server log was read, the pooled tokens and
    milliseconds a block -- the block time is the engine's own speed whatever the text accepts."""
    out: dict = {}
    for r in recs:
        if r.get("probe") is None:
            continue
        out.setdefault(r["probe"], []).append(r)
    summary = {}
    for rnd, rs in sorted(out.items()):
        ok = [r for r in rs if not r["error"] and r["tokens"] > 1]
        blk = [r for r in ok if r.get("blocks")]
        summary[rnd] = {
            "n": len(rs), "ok": len(ok),
            "tok_s_p50": statistics.median([r["tok_s"] for r in ok]) if ok else None,
            "ms_blk": (sum(r["decode_ms"] for r in blk) / sum(r["blocks"] for r in blk)
                       if blk else None),
            "tok_blk": (sum(r["committed"] for r in blk) / sum(r["blocks"] for r in blk)
                        if blk else None)}
    return summary


def load_docs(path: str, tokenizer: str, lens: tuple[int, ...]) -> list[dict]:
    """The held-out documents of `tools/longprompts.py` as text, one record per row and length."""
    import numpy as np
    from transformers import AutoTokenizer

    man = json.load(open(os.path.join(path, "manifest.json")))
    tok = AutoTokenizer.from_pretrained(tokenizer)
    docs = []
    for L in lens:
        if str(L) not in man["prompts"]:
            continue
        ids = np.load(os.path.join(path, f"ids-{L}.npy"))
        for i, m in enumerate(man["prompts"][str(L)]):
            docs.append({"len": L, "domain": m["domain"], "i": m.get("i", i),
                         "text": tok.decode(ids[i].tolist())})
    return docs


class Soak:
    def __init__(self, a):
        self.a = a
        self.recs: list[dict] = []
        self.health: list[dict] = []
        self.samples: list[dict] = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        # set = traffic may run; cleared for an idle gap or a probe round
        self.gate = threading.Event()
        self.gate.set()
        self.busy = 0
        self.t0 = time.time()
        self.long_prompt = ("Read this and then answer the question at the end.\n\n"
                            + FILLER * a.long_repeat
                            + "\n\nQuestion: in two sentences, what is the passage claiming?")
        self.long_docs: list[dict] = []
        self.probe_docs: list[dict] = []
        self.probe_round = 0
        # the document row each probe round takes, in order; past the list, the next unused one
        self.probe_rows = [int(x) for x in getattr(a, "probe_rows", "").split(",") if x.strip()]

    def _workloads(self):
        return [w for w in WORKLOADS if w[0] != "longdoc" or self.long_docs]

    def one(self, rng: random.Random) -> None:
        name, prompt, think, max_tok, turns = rng.choice(self._workloads())
        if name == "long":
            prompt = self.long_prompt
        doc = None
        if name == "longdoc":
            # a length first, then a document of it: the set has 21 rows at 8k and 16k and 3 at 32k,
            # and a uniform pick over rows would all but never send the longest
            lens = sorted({d["len"] for d in self.long_docs})
            want = rng.choice(lens)
            doc = rng.choice([d for d in self.long_docs if d["len"] == want])
            prompt = ("Read this document and then answer the question at the end.\n\n"
                      + doc["text"] + "\n\nQuestion: summarise the document in three sentences.")
        messages = []
        for turn in range(turns):
            if turn == 0:
                messages = messages + [{"role": "user", "content": prompt}]
            elif name != "tool":
                follow = ("Now name the three terms it depends on most." if name == "longdoc"
                          else "Now say the same thing in one sentence.")
                messages = messages + [{"role": "user", "content": follow}]
            body = {"model": self.a.model, "stream": True, "temperature": 0,
                    "max_tokens": max_tok, "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": bool(think)},
                    "conversation_id": f"soak-{name}-{rng.randrange(10**6)}"}
            body.update(EXTRA.get(name, {}))
            body["messages"] = messages
            abandon = self.a.abandon_after if rng.random() < self.a.abandon_rate else 0.0
            with self.lock:
                self.busy += 1
            try:
                rec = post_stream(self.a.base_url, body, self.a.timeout, abandon)
            finally:
                with self.lock:
                    self.busy -= 1
            text = rec.pop("text", "")              # kept out of the records: no answers in the JSON
            rec["workload"] = name
            rec["turn"] = turn + 1
            if doc is not None:
                rec["doc"] = f"{doc['domain']}-{doc['len']}-{doc['i']}"
            with self.lock:
                self.recs.append(rec)
            if rec["error"] or rec["abandoned"] or not rec["chars"]:
                return
            if name == "tool":
                calls = [c for _, c in sorted((rec.get("calls") or {}).items())]
                if not calls:
                    return                     # it answered without the tool: nothing to return
                messages = messages + [
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": c["id"] or f"call_{k}", "type": "function",
                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for k, c in enumerate(calls)]}] + [
                    {"role": "tool", "tool_call_id": c["id"] or f"call_{k}", "content": TOOL_RESULT}
                    for k, c in enumerate(calls)]
            else:
                # The answer the server wrote, as a client echoes it: that is what lets the next turn
                # resume from the session entry rather than from a prefix checkpoint (soak 1
                # echoed a placeholder and read 5 session hits in 395 requests).
                messages = messages + [{"role": "assistant", "content": text}]
            if not self.gate.is_set() or self.stop.is_set():
                return                         # an idle gap or a probe round began: stop here

    def worker(self, seed: int) -> None:
        rng = random.Random(seed)
        while not self.stop.is_set():
            self.gate.wait()
            if self.stop.is_set():
                break
            try:
                self.one(rng)
            except Exception as exc:                               # noqa: BLE001
                with self.lock:
                    self.recs.append({"workload": "?", "error": f"driver: {exc}", "tok_s": 0.0,
                                      "wall": 0.0, "tokens": 0, "ttft": None, "finish": None,
                                      "status": None, "abandoned": False, "turn": 0,
                                      "t0": time.time()})
            time.sleep(self.a.gap)

    def sampler(self) -> None:
        while not self.stop.is_set():
            try:
                with urllib.request.urlopen(self.a.base_url.rsplit("/v1", 1)[0] + "/health",
                                            timeout=10) as r:
                    h = json.loads(r.read())
                h["t"] = time.time()
                row = sample_row(h, self.t0, meminfo_gb() if self.a.meminfo else None)
                with self.lock:
                    self.health.append(h)
                    self.samples.append(row)
            except Exception:
                pass
            self.stop.wait(self.a.health_every)

    def _drain(self, limit: float) -> None:
        """Wait until no soak request is in flight (a probe measures the engine alone)."""
        t = time.time()
        while time.time() - t < limit:
            with self.lock:
                if self.busy == 0:
                    return
            time.sleep(0.5)

    def probe(self) -> None:
        """One probe round: traffic paused, one unseen 2k document per domain, greedy, 256 out."""
        take = []
        rows = self.probe_rows[self.probe_round] if self.probe_round < len(self.probe_rows) else None
        for dom in ("prose", "german", "code"):
            d = next((d for d in self.probe_docs if d["domain"] == dom and not d.get("used")
                      and (rows is None or d["i"] == rows)), None)
            if d is not None:
                d["used"] = True
                take.append(d)
        if not take:
            return
        self.gate.clear()
        self._drain(self.a.timeout)
        rnd = self.probe_round
        self.probe_round += 1
        for d in take:
            body = {"model": self.a.model, "stream": True, "temperature": 0, "max_tokens": 256,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": "Continue the following text exactly "
                                  "where it stops, in the same style.\n\n" + d["text"]}]}
            rec = post_stream(self.a.base_url, body, self.a.timeout)
            rec.update(workload="probe", turn=1, probe=rnd, doc=f"{d['domain']}-{d['len']}-{d['i']}")
            with self.lock:
                self.recs.append(rec)
            print(f"[probe {rnd}] {rec['doc']} tok/s {rec['tok_s']:.2f} ttft "
                  f"{(rec['ttft'] or 0) * 1e3:.0f} ms tokens {rec['tokens']}"
                  + (f" error {rec['error']}" if rec["error"] else ""), flush=True)
        if not self.stop.is_set():
            self.gate.set()

    def report(self, since: float, label: str) -> None:
        with self.lock:
            recs = [r for r in self.recs if r.get("t0", 0) >= since and r.get("probe") is None]
            health = list(self.health)
        if not recs:
            print(f"[{label}] no requests"); return
        ok = [r for r in recs if not r["error"] and not r["abandoned"] and r["tokens"]]
        rates = sorted(r["tok_s"] for r in ok) or [0.0]
        ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None) or [0.0]
        fins: dict = {}
        for r in recs:
            key = ("abandoned" if r["abandoned"] else r["error"] and "error" or r["finish"])
            fins[key] = fins.get(key, 0) + 1
        mem = health[-1].get("memory", {}) if health else {}
        cache = (health[-1].get("cache") or {}) if health else {}
        store = (cache.get("state_store") or {}) if isinstance(cache, dict) else {}
        store = {k: v for k, v in store.items() if k not in ("boundaries", "bytes_by_part")}
        print(f"[{label}] n={len(recs)} ok={len(ok)} "
              f"tok/s p50={statistics.median(rates):.2f} p10={rates[len(rates) // 10]:.2f} "
              f"ttft p50={statistics.median(ttfts) * 1e3:.0f} ms "
              f"finish={fins} "
              f"alloc={mem.get('allocated_gb')} reserved={mem.get('reserved_gb')} "
              f"rss={mem.get('rss_gb')} store={store}", flush=True)

    def _join_server_log(self, recs: list[dict]) -> None:
        """Each request's blocks, committed tokens and decode time from the server's `[req]` line."""
        if not self.a.server_log or not os.path.exists(self.a.server_log):
            return
        from tools.rowlog import parse_requests
        by_id = {r["cid"]: r for r in parse_requests(open(self.a.server_log, errors="replace").read())}
        for r in recs:
            s = by_id.get(r.get("id"))
            if s is not None and s.get("blocks"):
                r.update(blocks=s["blocks"], committed=s["committed"], decode_ms=s["decode_ms"])

    def run(self) -> int:
        a = self.a
        if a.long_docs:
            self.long_docs = load_docs(a.long_docs, a.tokenizer, (8192, 16384, 32768))
            print(f"[soak] long documents: {len(self.long_docs)} "
                  f"({sorted({d['len'] for d in self.long_docs})} tokens)", flush=True)
        if a.probe_docs:
            self.probe_docs = load_docs(a.probe_docs, a.tokenizer, (2048,))
        self.t0 = time.time()
        threads = [threading.Thread(target=self.sampler, daemon=True)]
        threads[0].start()
        if self.probe_docs:
            self.probe()
        workers = [threading.Thread(target=self.worker, args=(i,), daemon=True)
                   for i in range(a.concurrency)]
        for t in workers:
            t.start()
        t0 = time.time()
        last = last_probe = last_idle = t0
        try:
            while time.time() - t0 < a.minutes * 60:
                time.sleep(min(a.report_every, 5))
                now = time.time()
                if now - last >= a.report_every:
                    self.report(last, time.strftime("%H:%M:%S"))
                    last = now
                    if a.csv:
                        with self.lock:
                            write_csv(a.csv, list(self.samples))
                if self.probe_docs and a.probe_every and now - last_probe >= a.probe_every:
                    self.probe()
                    last_probe = time.time()
                if a.idle_every and now - last_idle >= a.idle_every:
                    # A gap in the day: nothing new starts for `idle_for` seconds. What is in
                    # flight finishes; the sampler keeps sampling an idle server.
                    print(f"[idle] {time.strftime('%H:%M:%S')} {a.idle_for:.0f} s", flush=True)
                    self.gate.clear()
                    self.stop.wait(a.idle_for)
                    self.gate.set()
                    last_idle = time.time()
        except KeyboardInterrupt:
            pass
        if self.probe_docs:
            self.probe()                       # the end round, with the same drain
        self.stop.set()
        self.gate.set()
        for t in workers + threads:
            t.join(timeout=a.timeout + 10)
        print("\n" + "=" * 96)
        self.report(0.0, "TOTAL")
        with self.lock:
            recs, health, samples = list(self.recs), list(self.health), list(self.samples)
        self._join_server_log(recs)
        bad = [r for r in recs if r["error"]]
        if bad:
            print(f"\n{len(bad)} requests with an error:")
            seen: dict = {}
            for r in bad:
                seen[r["error"][:120]] = seen.get(r["error"][:120], 0) + 1
            for msg, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"  {n:4d}  {msg}")
        if health:
            first, last_h = health[0].get("memory", {}), health[-1].get("memory", {})
            print(f"\nmemory first -> last: allocated {first.get('allocated_gb')} -> "
                  f"{last_h.get('allocated_gb')} GB, reserved {first.get('reserved_gb')} -> "
                  f"{last_h.get('reserved_gb')} GB, rss {first.get('rss_gb')} -> "
                  f"{last_h.get('rss_gb')} GB")
        trends = {k: trend(samples, k) for k in ("rss_gb", "allocated_gb", "reserved_gb",
                                                 "pagecache_gb", "memfree_gb", "store_bytes",
                                                 "store_entries", "suffix_tokens")}
        for k, v in trends.items():
            if v is not None:
                print(f"[trend] {k:14s} first {v['first']:.4g} last {v['last']:.4g} "
                      f"min {v['min']:.4g} max {v['max']:.4g} "
                      f"slope (2nd half) {v['slope_per_10min_2nd_half']:+.4g} / 10 min")
        probes = probe_summary(recs)
        for rnd, p in probes.items():
            print(f"[probe] round {rnd}: ok {p['ok']}/{p['n']} tok/s p50 "
                  f"{p['tok_s_p50'] if p['tok_s_p50'] is None else round(p['tok_s_p50'], 2)} "
                  f"ms/blk {p['ms_blk'] if p['ms_blk'] is None else round(p['ms_blk'], 2)} "
                  f"tok/blk {p['tok_blk'] if p['tok_blk'] is None else round(p['tok_blk'], 2)}")
        if a.csv:
            write_csv(a.csv, samples)
            print(f"[soak] -> {a.csv} ({len(samples)} samples)")
        if a.json_out:
            json.dump({"args": vars(a), "requests": recs, "health": health,
                       "trends": trends, "probes": probes}, open(a.json_out, "w"))
            print(f"[soak] -> {a.json_out}")
        return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen38-spark-engine")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--concurrency", type=int, default=3,
                    help="connections at once. The engine serves one; the rest queue, which is "
                         "the point -- a queued client must get an answer or an honest refusal")
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--report-every", type=float, default=300.0)
    ap.add_argument("--health-every", type=float, default=30.0)
    ap.add_argument("--abandon-rate", type=float, default=0.08,
                    help="fraction of requests that hang up mid-stream on purpose")
    ap.add_argument("--abandon-after", type=float, default=3.0)
    ap.add_argument("--long-repeat", type=int, default=380,
                    help="copies of the filler paragraph in the 'long' workload. The filler is "
                         "about 60 tokens, so 380 is ~23k and fits a 32k context with room for "
                         "the answer; 900 was 54k and the server correctly refused every one of "
                         "them with a 400, which is a fine thing to have learnt once")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--csv", default="",
                    help="one row per /health sample: memory, store counters, page cache")
    ap.add_argument("--meminfo", action=argparse.BooleanOptionalAction, default=True,
                    help="add /proc/meminfo to every sample; only meaningful on the server's host")
    ap.add_argument("--long-docs", default="",
                    help="a tools/longprompts.py set: its 8k/16k/32k rows become the longdoc workload")
    ap.add_argument("--probe-docs", default="",
                    help="a tools/longprompts.py set with 2,048-token rows: one unseen row per "
                         "domain per probe round")
    ap.add_argument("--probe-every", type=float, default=0.0,
                    help="seconds between probe rounds; 0 = only at the start and the end")
    ap.add_argument("--probe-rows", default="",
                    help="the document row each probe round takes, in order (e.g. 4,0): a fresh "
                         "server given the end round's row is the control for an aged one")
    ap.add_argument("--idle-every", type=float, default=0.0,
                    help="seconds between idle gaps (no new requests); 0 = none")
    ap.add_argument("--idle-for", type=float, default=60.0)
    ap.add_argument("--server-log", default="",
                    help="the server's log: joins each request's [req] line (blocks, decode ms)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    return Soak(ap.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
