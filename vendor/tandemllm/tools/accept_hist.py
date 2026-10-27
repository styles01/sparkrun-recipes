"""What a verified block commits, as a distribution, per workload and over the row.

The row's mean is a statement about its reproduction workloads and its median about fresh text
(docs/measurement.md). Neither says whether a block ever runs out of WIDTH, which is the one
question a deeper draft answers: a block that commits all sixteen tokens its arm could hold would
have committed more from a longer draft, and a block that commits four would not. Since 2026-09-23
the length router counts that per request -- `commits 16:4x5,16x5 cap arm 5 depth 5` on the
`[drafter]` line the server prints at the start of the NEXT request -- and this reads it back.

    # the row: every [drafter] line of a row3 server log, summed
    python tools/accept_hist.py results/row3/<label>/server.log

    # the five bench workloads through the served stack, one request each, on a test server
    # (inside ops/hold.sh: it starts an engine)
    python tools/accept_hist.py --serve --port 8011 --json results/accept-five.json

`cap arm` is a block that committed its arm's whole width (8 or 16). `cap depth` is a block that
committed everything it was handed: the whole width, or a tree the prune stopped short of the arm,
or the budget left at the end of a generation.

`--curve` reads where in a block the draft went wrong instead: the `accept=` histogram on
each request's own `[req]` line (depth the block offered, draft tokens it accepted), rebuilt into
a_i = P(slot i accepted | slots < i were), censored where a block had no slot i, beside the sample
count per slot and the first-miss histogram. With `--serve` it is per workload; `--chain` serves
without the tree (`--len-fixed 16` in `--server-arg` for the wide arm alone) for the clean chain
curve.

    python tools/accept_hist.py --curve results/row3/<label>/server.log
    python tools/accept_hist.py --serve --curve --port 8011 --repeat 3 --json results/curve.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import rowlog  # noqa: E402

LINE = re.compile(r"\[drafter\] .*?commits (\S+) cap arm (\d+) depth (\d+)")


parse_hist = rowlog.parse_hist


def parse_log(text: str) -> list[dict]:
    """One record per `[drafter]` line that carries the histogram, in log order."""
    recs = []
    for m in LINE.finditer(text):
        recs.append({"commits": parse_hist(m.group(1)), "cap_arm": int(m.group(2)),
                     "cap_depth": int(m.group(3))})
    return recs


def summarize(recs: list[dict]) -> dict:
    """Blocks, the mean commit and the two cap rates, per arm and over both."""
    arms: dict[int, dict[int, int]] = {}
    cap_arm = cap_depth = 0
    for r in recs:
        cap_arm += r["cap_arm"]
        cap_depth += r["cap_depth"]
        for arm, h in r["commits"].items():
            dst = arms.setdefault(arm, {})
            for c, n in h.items():
                dst[c] = dst.get(c, 0) + n
    out = {"arms": {}, "cap_arm": cap_arm, "cap_depth": cap_depth}
    total_blocks = total_tokens = 0
    for arm, h in sorted(arms.items()):
        blocks = sum(h.values())
        tokens = sum(c * n for c, n in h.items())
        total_blocks += blocks
        total_tokens += tokens
        out["arms"][arm] = {"blocks": blocks, "mean": tokens / blocks if blocks else 0.0,
                            "full": h.get(arm, 0), "full_pct": 100.0 * h.get(arm, 0) / blocks
                            if blocks else 0.0, "hist": dict(sorted(h.items()))}
    out["blocks"] = total_blocks
    out["mean"] = total_tokens / total_blocks if total_blocks else 0.0
    out["cap_arm_pct"] = 100.0 * cap_arm / total_blocks if total_blocks else 0.0
    out["cap_depth_pct"] = 100.0 * cap_depth / total_blocks if total_blocks else 0.0
    return out


def show(label: str, s: dict) -> str:
    rows = [f"{label}: {s['blocks']} blocks, {s['mean']:.2f} committed a block, "
            f"arm cap {s['cap_arm']} ({s['cap_arm_pct']:.1f} %), "
            f"depth cap {s['cap_depth']} ({s['cap_depth_pct']:.1f} %)"]
    for arm, a in s["arms"].items():
        hist = " ".join(f"{c}:{n}" for c, n in a["hist"].items())
        rows.append(f"    arm {arm:2d}: {a['blocks']:5d} blocks  mean {a['mean']:5.2f}  "
                    f"full {a['full']:4d} ({a['full_pct']:5.1f} %)   {hist}")
    return "\n".join(rows)


def show_curve(label: str, acc: dict) -> str:
    """The per-slot acceptance of one workload, and where the first misses landed."""
    cv = rowlog.curve(acc)
    blocks = sum(sum(h.values()) for h in acc.values())
    rows = [f"{label}: {blocks} blocks with a draft"]
    rows.append("    slot  " + " ".join(f"{c['slot']:>5d}" for c in cv))
    rows.append("    a_i   " + " ".join(f"{c['rate']:5.3f}" if c["rate"] is not None else "    -"
                                   for c in cv))
    rows.append("    n     " + " ".join(f"{c['n']:>5d}" for c in cv))
    fm = rowlog.first_miss(acc)
    rows.append("    first miss at slot: " + " ".join(f"{s}:{n}" for s, n in fm.items()))
    return "\n".join(rows)


def curves_by(names: list[str], reqs: list[dict]) -> dict:
    """Per workload, the accept histograms of its requests summed; `reqs` are the `[req]` records
    of `names`, in order."""
    by: dict[str, dict] = {}
    for name, r in zip(names, reqs):
        if name in ("warm", "flush") or not r.get("blocks"):
            continue
        rowlog.add_hist(by.setdefault(name, {}), r["accept"])
    by["ALL"] = {}
    for name, acc in list(by.items()):
        if name != "ALL":
            rowlog.add_hist(by["ALL"], acc)
    return by


# ---------------------------------------------------------------- the five workloads, served


def factors_by(names: list[str], reqs: list[dict]) -> dict:
    """Per workload, both factors of its speed from its requests' `[req]` lines -- tokens
    a round and ms a round pooled (`rowlog.factors`), tok/s as committed tokens over decode time,
    and each request's own tokens a round, for a paired or a spread comparison."""
    by: dict[str, list[dict]] = {}
    for name, r in zip(names, reqs):
        if name in ("warm", "flush") or not r.get("blocks"):
            continue
        by.setdefault(name, []).append(r)
    by["ALL"] = [r for name, rs in list(by.items()) for r in rs]
    out = {}
    for name, rs in by.items():
        f = rowlog.factors(rs)
        f.pop("accept", None)
        f["tok_s"] = 1e3 * sum(r["committed"] for r in rs) / sum(r["decode_ms"] for r in rs)
        f["per_request_tok_blk"] = [r["committed"] / r["blocks"] for r in rs]
        out[name] = f
    return out


def show_factors(name: str, f: dict) -> str:
    return (f"{name:6s} {f['requests']:3d} requests {f['blocks']:5d} rounds  "
            f"{f['tok_blk']:5.2f} tokens a round  {f['ms_blk']:6.1f} ms a round  "
            f"{f['tok_s']:6.2f} tok/s")


def _post(port: int, prompt: str, max_tokens: int, temperature: float = 0.0,
          draft_temperature: float | None = None) -> dict:
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    if draft_temperature is not None:
        body["draft_temperature"] = draft_temperature
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as fh:
        out = json.load(fh)
    out["_wall_s"] = time.perf_counter() - t0
    return out


def serve(a) -> dict:
    """Start the served configuration on a test port, one request per workload, and read each
    request's histogram off the `[drafter]` line the NEXT request prints."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import row3
    from tools.bench_decode import PROMPTS
    ns = argparse.Namespace(port=a.port, python=Path(a.python), pythonpath=Path(a.pythonpath),
                            repo=Path(a.repo), max_len=a.max_len, len_fixed=0, budget=16,
                            nvfp4=row3.DEFAULT_NV, head=row3.DEFAULT_HEAD,
                            server_arg=["--drop-idle"] + a.server_arg, len_latch=True,
                            start_timeout=600, tree=not a.chain)
    env = dict(kv.split("=", 1) for kv in a.env)
    log = Path(a.log)
    proc = row3.start_server(ns, env, log)
    order = ["warm"] + [n for n in PROMPTS for _ in range(a.repeat)] + ["flush"]
    walls = []
    try:
        for name in order:
            text = PROMPTS["chat"] if name in ("warm", "flush") else PROMPTS[name]
            r = _post(a.port, text, 16 if name == "flush" else a.max_tokens,
                      temperature=0.0 if name in ("warm", "flush") else a.temperature,
                      draft_temperature=a.draft_temperature)
            walls.append((name, r["_wall_s"], r.get("usage", {}).get("completion_tokens")))
    finally:
        row3.stop_server(proc)
    names = order[:-1]
    text = log.read_text()
    if a.factors:
        return factors_by(order, rowlog.parse_requests(text)[-len(order):])
    if a.curve:
        # `[req]` lines are printed at the END of each request, so the last len(order) of them
        # are this run's requests, flush included, in order
        reqs = rowlog.parse_requests(text)[-len(order):]
        return curves_by(order, reqs)
    per_req = per_request(parse_log(text), len(names))
    if per_req is None:
        raise SystemExit(f"fewer [drafter] lines than requests; see {log}")
    return summarize_by(names, per_req, walls)


def per_request(recs: list[dict], n: int) -> list[dict] | None:
    """The records of the last `n` requests before the flush.

    A `[drafter]` line is printed at the START of a request and describes the one before it, so
    the flush request's line is the last request's. The server also runs requests of its own at
    startup (the warm-up that pays for the autotuning), whose lines come first; counting from the
    end is what makes the mapping independent of how many there were.
    """
    if len(recs) < n:
        return None
    return recs[len(recs) - n:]


def summarize_by(names: list[str], per_req: list[dict], walls: list) -> dict:
    by: dict[str, list[dict]] = {}
    for name, rec in zip(names, per_req):
        if name != "warm":
            by.setdefault(name, []).append(rec)
    out = {name: summarize(rs) for name, rs in by.items()}
    out["ALL"] = summarize([r for rs in by.values() for r in rs])
    out["_walls"] = walls
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="server logs to sum")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--python", default=os.path.expanduser(
        "~/recipes/ling3-flash-dgx-spark/.venv/bin/python"))
    ap.add_argument("--pythonpath", default=os.path.expanduser("~/pylibs"))
    ap.add_argument("--log", default="results/accept-serve.log")
    ap.add_argument("--json", default="")
    ap.add_argument("--curve", action="store_true",
                    help="per-slot acceptance a1..a15 and the first-miss histogram from the "
                         "[req] lines")
    ap.add_argument("--chain", action="store_true",
                    help="with --serve: no tree, for the clean chain curve")
    ap.add_argument("--names", default="",
                    help="with logs: the request order of a finished --serve run (warm,prose,...), "
                         "to re-read its server log per workload without running it again")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="with --serve: the workloads' sampling temperature (; the warm-up "
                         "and the flush stay greedy)")
    ap.add_argument("--draft-temperature", type=float, default=None,
                    help="with --serve: the request's draft_temperature (default: the server's)")
    ap.add_argument("--factors", action="store_true",
                    help="with --serve: per workload tokens a round, ms a round and tok/s from the "
                         "[req] lines")
    a = ap.parse_args()

    if a.factors and a.serve:
        out = serve(a)
        for name, f in out.items():
            print(show_factors(name, f))
    elif a.curve:
        if a.serve:
            by = serve(a)
        else:
            acc: dict = {}
            for p in a.logs:
                for r in rowlog.parse_requests(Path(p).read_text()):
                    if r.get("blocks"):
                        rowlog.add_hist(acc, r["accept"])
            by = {"ALL": acc}
        for name, acc in by.items():
            print(show_curve(name, acc))
        out = {name: {"curve": rowlog.curve(acc), "first_miss": rowlog.first_miss(acc),
                      "accept": acc} for name, acc in by.items()}
    elif a.serve:
        out = serve(a)
        for name, s in out.items():
            if not name.startswith("_"):
                print(show(name, s))
    elif a.names:
        names = a.names.split(",")
        per_req = per_request(parse_log(Path(a.logs[0]).read_text()), len(names))
        out = summarize_by(names, per_req, [])
        for name, s in out.items():
            if not name.startswith("_"):
                print(show(name, s))
    else:
        recs = [r for p in a.logs for r in parse_log(Path(p).read_text())]
        out = {"ALL": summarize(recs), "requests": len(recs)}
        print(show(f"{len(recs)} requests", out["ALL"]))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
