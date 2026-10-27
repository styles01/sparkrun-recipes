"""Prove `/v1/dashboard/live` is live, and that its numbers are the response's.

    python tools/live_check.py --base http://127.0.0.1:8011 --token "$QSE_ADMIN_TOKEN" \\
        --out results/live/live-check-<date>.json

Fires streamed chat requests at a server -- two at once, so one queues behind the other -- and
polls `GET /v1/dashboard/live?follow=0` every 0.5 s from a thread while they run. Their text is
new on every run (`fresh_prompts`, a seeded story with the run's nonce, recorded in the report),
so a server whose suffix store learned an earlier run still decodes at its novel-text rate.
Then it checks:

  1. LIVE: for every request that decoded for more than two seconds, the polled `tokens` grew
     between snapshots while its phase was `decode`, and `decode_tps_now` was set;
  2. EQUAL: every request's last `done` row equals its own response `timings` -- `ttft_ms`,
     `queue_ms`, `prompt_per_second`, `predicted_per_second`, `predicted_n`, `predicted_ms`,
     `tokens_per_block`, `total_ms` -- to the cent (they are the same record);
  3. COUNTS: `counts.in_flight` reached 2 (one decoding, one queued or prefilling) and
     `counts.served` grew by the number of requests sent;
  4. QUEUE: the second request of a pair was seen as `queued` at least once.
  5. LIVE RATES (VIS-40): the event reports its `windows`; a request that decoded past the
     warm-up got `decode_live_now_tps` only after `decode_warmup.reached`; and with
     `--long-prompt-tokens N` one more request prefills a synthetic prompt of about N tokens,
     during which every polled snapshot has `now.prefill_live_tps` equal to that row's
     `activity.prefill.tps_live` (the tile and the banner read the same number), at least one of
     them set.
  6. MONITOR (SRV-115, contract 1.2): with `--expect-monitor` (a server running `--conductor
     shadow` or `on`), every decoding row of a polled snapshot carries `activity.conductor` with
     its mode, and the report lists every `supervising` sighting with its action and reason (a
     hidden forward shows there within one poll, and for a second after it).

Standard library only, so it runs from the Mac against the box. Exit 1 on any failed check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

# The requests' text is new on every run. Until 2026-10-01 they were four fixed prompts; after three
# runs in a day the persistent suffix store had learned their answers, the 98-token one decoded at
# 186 tok/s with 15.9 tokens a block, every request ended in about 1.3 s, and "a live decode rate
# from token 50 on" failed with no request decoding long enough for a "now" rate. Each prompt is
# now a story drawn from these lists by a seeded generator, with the run's nonce in it and asked
# for in the answer, so neither the prompt nor its continuation is one the store has seen.
KEEPERS = ["lighthouse keeper", "ferry pilot", "glassblower", "night baker", "bee keeper",
           "clock mender", "salt farmer", "map engraver", "bell founder", "rope maker",
           "orchard warden", "canal lock tender", "kite maker", "tide gauge reader"]
PLACES = ["a fjord village", "a desert railway halt", "an island observatory", "a river delta",
          "a mountain pass", "a harbour market", "a salt marsh", "a quarry town",
          "a frozen lake camp", "a terraced vineyard", "a border bridge", "a reed-thatched hamlet"]
THINGS = ["a brass compass", "a cracked violin", "a ledger of tides", "a blue lantern",
          "a broken barometer", "a jar of seeds", "a carved chess piece", "a folded sail",
          "a letter with no stamp", "a copper kettle", "a spool of red thread", "a weather vane"]
TURNS = ["a storm closes the road", "a stranger asks for directions", "the power fails at dusk",
         "a child finds a hidden door", "the river freezes overnight", "a festival is cancelled",
         "an old debt is repaid", "a bird flies in through the window", "the last train is late"]
FORMS = ["a short story of about {n} words", "a diary entry of about {n} words",
         "a letter of about {n} words", "a radio report of about {n} words"]


def fresh_prompts(seed: int, k: int = 4) -> list[str]:
    """`k` prompts no run has sent before: each a different story drawn from the lists by
    `random.Random(seed)`, with `seed` as the run's nonce in the text and in the answer (the
    register number), so the continuation is new as well. The same seed gives the same prompts
    (a run is reproducible from its report); a new seed every run is the point."""
    import random
    rng = random.Random(seed)
    out = []
    for i in range(k):
        keeper, place, thing, turn = (rng.choice(KEEPERS), rng.choice(PLACES), rng.choice(THINGS),
                                      rng.choice(TURNS))
        form = rng.choice(FORMS).format(n=rng.choice((180, 200, 220, 250)))
        reg = f"{seed % 100000:05d}-{i}{rng.randrange(100, 1000)}"
        out.append(f"Write {form} about a {keeper} in {place} who keeps {thing} on the day "
                   f"{turn}. The {keeper.split()[-1]}'s register number is {reg}: name it in the "
                   f"first sentence and again in the last one. Give the {keeper.split()[-1]} a "
                   f"name that starts with the letter {chr(65 + rng.randrange(26))}.")
    return out


def _req(base, path, token=None, data=None, timeout=300):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(base + path, data=json.dumps(data).encode() if data else None,
                                  headers=headers)


def snapshot(base, token):
    with urllib.request.urlopen(_req(base, "/v1/dashboard/live?follow=0", token),
                                timeout=10) as r:
        return json.loads(r.read())


def _composite(base: str) -> bool:
    """True when the server serves TandemLLM, the one model of a server with the conductor."""
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            return any(m.get("id") == "TandemLLM" for m in json.loads(r.read()).get("data") or [])
    except Exception:                                              # noqa: BLE001
        return False


def long_prompt(tokens: int) -> str:
    """About `tokens` tokens of plain numbered sentences, different on every run so no cache serves
    it. ~2.9 tokens a word on the served tokenizer: the run's salt and the line number are digits,
    one token each (at 1.3 a word, 16,384 asked read 36,749 tokens on 2026-10-01)."""
    salt = f"{time.time():.6f}"
    lines, i, words = [], 0, 0
    # count the words of each sentence, not the sentences: a sentence is ~13 words, and counting it
    # as one asked for a 16k prompt at 455k tokens, past the context (2026-10-01 on the box)
    while words * 2.9 < tokens:
        line = f"Line {i} of run {salt}: the keeper counts ships passing the lighthouse."
        lines.append(line)
        words += len(line.split())
        i += 1
    return " ".join(lines) + "\nSummarise the text above in one sentence."


def chat(base, model, prompt, max_tokens, think):
    """One streamed chat; returns (request id, timings, usage, wall seconds)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True,
            "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.perf_counter()
    rid, timings, usage = None, None, None
    with urllib.request.urlopen(_req(base, "/v1/chat/completions", data=body),
                                timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            rid = chunk.get("id", rid)
            if "timings" in chunk:
                timings, usage = chunk["timings"], chunk.get("usage")
    return rid, timings, usage, time.perf_counter() - t0


class Poller(threading.Thread):
    def __init__(self, base, token, period):
        super().__init__(daemon=True)
        self.base, self.token, self.period = base, token, period
        self.snaps: list[dict] = []
        self.errors: list[str] = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                self.snaps.append(snapshot(self.base, self.token))
            except Exception as exc:                                   # noqa: BLE001
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self.stop.wait(self.period)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    ap.add_argument("--token", default=os.environ.get("QSE_ADMIN_TOKEN"))
    ap.add_argument("--model", default="qwen38-spark-engine")
    ap.add_argument("--pairs", type=int, default=2, help="pairs of concurrent requests")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--period", type=float, default=0.5)
    ap.add_argument("--long-prompt-tokens", type=int, default=0,
                    help="one more request with a synthetic prompt of about this many tokens, for "
                         "the live prefill rate (0 = none; 16384 prefills for ~20 s on the box)")
    ap.add_argument("--expect-monitor", action="store_true",
                    help="the server's session monitor watches these requests: check the live "
                         "`activity.conductor` block (contract 1.2)")
    ap.add_argument("--seed", type=int, default=None,
                    help="the prompts' seed (default: new on every run; give one only to repeat a "
                         "run's prompts, which a store-on server may have learned)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if not a.token:
        print("a token is needed (--token or QSE_ADMIN_TOKEN)")
        return 2
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(4), "big") ^ time.time_ns()
    prompts = fresh_prompts(seed, 2 * a.pairs)
    print(f"  prompts: seed {seed}, {len(prompts)} new stories")

    before = snapshot(a.base, a.token)
    poller = Poller(a.base, a.token, a.period)
    poller.start()
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def one(prompt):
        rid, timings, usage, wall = chat(a.base, a.model, prompt, a.max_tokens, a.think)
        with lock:
            results[rid] = {"timings": timings, "usage": usage, "wall_s": round(wall, 3)}
        return rid

    sent = 0
    for i in range(a.pairs):
        ts = [threading.Thread(target=one, args=(prompts[2 * i + j],)) for j in range(2)]
        for t in ts:
            t.start()
            time.sleep(0.3)                      # the second lands while the first prefills
        for t in ts:
            t.join()
        sent += 2
    long_rid = None
    if a.long_prompt_tokens > 0:
        long_rid = one(long_prompt(a.long_prompt_tokens))
        sent += 1
    time.sleep(2.5)                              # two more snapshots with every row done
    poller.stop.set()
    poller.join(timeout=5)
    after = snapshot(a.base, a.token)

    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}")

    # per request: the polled rows, in order
    rows: dict[str, list[dict]] = {}
    for s in poller.snaps:
        for r in s["requests"]:
            rows.setdefault(r["request_id"], []).append(r)
    missing = [rid for rid in results if rid not in rows]
    check("every request was seen by the poller", not missing, f"missing {missing}")

    for rid, res in results.items():
        seen = rows.get(rid, [])
        dec = [r for r in seen if r["phase"] == "decode"]
        t = res["timings"] or {}
        if (t.get("predicted_ms") or 0) > 2000:
            grew = len(dec) >= 2 and dec[-1]["tokens"] > dec[0]["tokens"]
            now = any(r["decode_tps_now"] is not None for r in dec)
            check(f"{rid[:20]} tokens grew while decoding",
                  grew and now, f"{[r['tokens'] for r in dec]} now={[r['decode_tps_now'] for r in dec]}")
        done = [r for r in seen if r["phase"] == "done"]
        if not done:
            check(f"{rid[:20]} final row present", False, "no done row polled")
            continue
        d = done[-1]
        pairs = [("ttft_ms", "ttft_ms"), ("queue_ms", "queue_ms"),
                 ("prefill_tps", "prompt_per_second"), ("decode_tps", "predicted_per_second"),
                 ("tokens", "predicted_n"), ("decode_ms", "predicted_ms"),
                 ("tokens_per_block", "tokens_per_block"), ("elapsed_ms", "total_ms")]
        diffs = {}
        for lk, tk in pairs:
            lv, tv = d.get(lk), t.get(tk)
            if lk in ("prefill_tps", "decode_tps", "tokens_per_block") and lv is None and tv == 0.0:
                continue                         # the response prints 0.0 where the row says null
            if lv != tv and not (isinstance(lv, (int, float)) and isinstance(tv, (int, float))
                                 and abs(lv - tv) <= 0.011):
                diffs[lk] = (lv, tv)
        check(f"{rid[:20]} final row == timings", not diffs,
              f"{diffs}" if diffs else f"tps {d['decode_tps']} ttft {d['ttft_ms']} tok/blk {d['tokens_per_block']}")

    peak = max((s["counts"]["in_flight"] for s in poller.snaps), default=0)
    check("in_flight reached 2", peak >= 2, f"peak {peak}")
    queued = any(r["phase"] == "queued" for rs in rows.values() for r in rs)
    check("a request was seen queued", queued, "")
    # A server with the conductor serves its classifier's own calls too (Qwen with X-Tandem: raw, one
    # per new user message), so there the count may only grow by more than this tool sent.
    grew = after["counts"]["served"] - before["counts"]["served"]
    conducted = _composite(a.base)
    check("served grew by the requests sent",
          grew == sent or (conducted and grew > sent),
          f"{before['counts']['served']} -> {after['counts']['served']} ({sent} sent"
          + (f", {grew - sent} the conductor's own" if conducted and grew > sent else "") + ")")
    nowv = [s["now"]["decode_tps"] for s in poller.snaps if s["now"]["decode_tps"]]
    check("now.decode_tps was set while decoding", bool(nowv),
          f"{len(nowv)} of {len(poller.snaps)} snapshots, max {max(nowv) if nowv else None}")
    check("poller saw no errors", not poller.errors, f"{poller.errors[:3]}")

    # VIS-40: the live rates without the warm-up
    win = (after.get("windows") or {})
    check("the event reports its windows", bool(win), f"{win}")
    warm = int(win.get("decode_warmup_tokens", 50))
    early = [r for rs in rows.values() for r in rs if r["phase"] == "decode"
             and r.get("decode_live_now_tps") is not None
             and not (r.get("decode_warmup") or {}).get("reached")]
    check("no live decode rate before the warm-up", not early, f"{early[:1]}")
    past = [r for rs in rows.values() for r in rs if r["phase"] == "decode"
            and r.get("decode_live_now_tps") is not None]
    check(f"a live decode rate from token {warm} on", bool(past),
          f"{len(past)} rows, max {max((r['decode_live_now_tps'] for r in past), default=None)}")
    if long_rid is not None:
        pairs = [(s["now"].get("prefill_live_tps"), r["activity"]["prefill"].get("tps_live"))
                 for s in poller.snaps for r in s["requests"]
                 if r["request_id"] == long_rid and r["phase"] == "prefill"
                 and r.get("activity") and r["activity"].get("prefill")]
        rated = [p for p in pairs if p[1] is not None]
        check("prefill tile == banner in every snapshot", all(x == y for x, y in pairs),
              f"{len(pairs)} snapshots, {[p for p in pairs if p[0] != p[1]][:3]}")
        check("a live prefill rate after the warm-up chunk", bool(rated),
              f"{len(rated)} of {len(pairs)}, last {rated[-1][1] if rated else None}")

    supervising = []
    if a.expect_monitor:
        # this tool's requests only: the conductor's own classifier calls run raw (no monitor)
        dec = [r for rs in rows.values() for r in rs
               if r["phase"] == "decode" and r.get("activity") and r["request_id"] in results]
        check("live contract 1.2", after.get("contract_version") == "1.2",
              f"{after.get('contract_version')}")
        missing = [r["request_id"] for r in dec if not (r["activity"].get("conductor") or {})
                   .get("mode")]
        check("activity.conductor on every decoding row", bool(dec) and not missing,
              f"{len(dec)} rows, {len(missing)} without it")
        supervising = [{"request_id": r["request_id"], **(r["activity"]["conductor"]["action"] or {}),
                        "label": r["activity"].get("label")}
                       for r in dec if (r["activity"].get("conductor") or {}).get("supervising")]
        print(f"  supervising seen in {len(supervising)} polled rows"
              + (f", first: {supervising[0]}" if supervising else ""))

    ok = all(c["ok"] for c in checks)
    out = {"base": a.base, "supervising": supervising, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "seed": seed, "prompts": prompts, "sent": sent,
           "snapshots": len(poller.snaps), "period_s": a.period, "checks": checks,
           "results": results, "before": before["counts"], "after": after["counts"],
           "polled_rows": rows, "ok": ok}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.out}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
