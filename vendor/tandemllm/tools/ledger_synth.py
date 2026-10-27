"""A synthetic usage ledger: a seeded year of requests, for tests, e2e and the query budget.

    python tools/ledger_synth.py /tmp/e2e/ledger.sqlite3 --rows 500000 --days 365 --seed 7

The rows have the shape real traffic has, roughly: a daily rhythm (quiet nights), quieter
weekends, a few quiet weeks, a few very heavy days, two clients (Open WebUI and a curl script) and
the dashboard's own test requests, every finish reason including refusals and errors, tool calls,
thinking, and response-cache replays. It never writes the production ledger (the same guard as a
test run), and it writes through the schema the server creates.
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import ledger  # noqa: E402

CLIENTS = (("k:3f9a1c0e2b7d", "open-webui", 0.78), ("k:91c2e4aa0b11", "curl", 0.17),
           ("k:5d0e7fb2c9a4", "dashboard", 0.05))


def rows(n: int, days: int, seed: int, end: float) -> list[tuple]:
    rnd = random.Random(seed)
    start = end - days * 86400
    quiet_weeks = {rnd.randrange(days // 7) for _ in range(3)}
    heavy_days = {rnd.randrange(days) for _ in range(4)}
    weights = []
    for d in range(days):
        w = 0.0 if d // 7 in quiet_weeks else (0.45 if (d % 7) in (5, 6) else 1.0)
        weights.append(w * (6.0 if d in heavy_days else 1.0))
    total = sum(weights) or 1.0
    out = []
    for d, w in enumerate(weights):
        for _ in range(int(round(n * w / total))):
            hour = min(23, max(0, int(rnd.gauss(14, 4))))
            ts = start + d * 86400 + hour * 3600 + rnd.random() * 3600
            r = rnd.random()
            cid, kind = next((c, k) for c, k, p in _cum(CLIENTS) if r <= p)
            fin = rnd.choices(("stop", "length", "tool_calls", "abandoned", "timeout", "error",
                               "refused"), (70, 12, 9, 4, 1, 2, 2))[0]
            if fin == "refused":
                out.append(_row(ts, cid, kind, fin, 503 if rnd.random() < 0.7 else 429))
                continue
            prompt = int(rnd.lognormvariate(6.8, 1.1)) + 20
            cached = int(prompt * rnd.choice((0.0, 0.0, 0.6, 0.9)))
            replay = rnd.random() < 0.03
            comp = int(rnd.lognormvariate(5.6, 1.0)) + 2
            think = rnd.random() < 0.7
            reason = int(comp * rnd.uniform(0.2, 0.7)) if think else 0
            blocks = max(1, int((comp - 1) / rnd.uniform(2.5, 5.5)))
            drafted = blocks * 15
            accepted = max(0, min(drafted, comp - 1 - blocks))
            queue = rnd.expovariate(1 / 30.0)
            pms = 0.4 if replay else (prompt - cached) / rnd.uniform(1500, 2600) * 1000 + 150
            tps = rnd.gauss(45, 18) if not replay else 0
            tps = max(8.0, tps)
            dms = (comp - 1) / tps * 1000 if not replay else 2.0
            out.append((int(ts * 1000), f"chatcmpl-{rnd.getrandbits(96):024x}",
                        "qwen38-spark-engine", cid, kind, "chat", 1, 500 if fin == "error" and
                        rnd.random() < 0.3 else 200, fin, prompt, prompt if replay else cached,
                        comp, reason, round(queue, 2), round(pms, 2), round(queue + pms, 2),
                        round(dms, 2), round(queue + pms + dms + 3, 2),
                        None if replay else round(tps, 2),
                        None if replay else round((prompt - cached) / (pms / 1000), 2),
                        0 if replay else blocks, 0 if replay else drafted,
                        0 if replay else accepted, int(fin == "tool_calls"), int(think),
                        "response" if replay else ("prefix" if cached else "none"),
                        32768, "RuntimeError" if fin == "error" else None, "0.1.0-synthetic",
                        "synthetic"))
    out.sort()
    return out


def _cum(clients):
    acc = 0.0
    for c, k, p in clients:
        acc += p
        yield c, k, acc


def _row(ts, cid, kind, fin, status):
    return (int(ts * 1000), "-", "qwen38-spark-engine", cid, kind, "chat", 1, status, fin,
            None, None, None, None, None, None, None, None, 0.5, None, None, None, None, None, 0, 0,
            None, None, None, "0.1.0-synthetic", "synthetic")


def write(path: str, n: int, days: int, seed: int, end: float | None = None) -> int:
    ledger.check_config(path, 400, test=True)
    led = ledger.Ledger(path).open()
    led.close()
    con = sqlite3.connect(led.path)
    data = rows(n, days, seed, end if end is not None else time.time())
    # a synthetic row predates the conductor's columns (model_served, profile, ...): null in each
    pad = (None,) * (len(ledger.COLUMNS) - len(data[0])) if data else ()
    data = [r + pad for r in data]
    con.executemany(f"INSERT INTO requests ({', '.join(ledger.COLUMNS)}) VALUES "
                    f"({', '.join('?' * len(ledger.COLUMNS))})", data)
    con.commit()
    con.close()
    return len(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    n = write(a.path, a.rows, a.days, a.seed)
    print(f"{n} rows over {a.days} days -> {ledger.resolve(a.path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
