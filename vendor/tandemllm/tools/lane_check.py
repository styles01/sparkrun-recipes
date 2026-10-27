"""The side lane against a real engine (server/preempt.py): short requests served in
the middle of a long generation, the generation's text held to an uninterrupted run's, byte for
byte.

One server (started here with row3's command -- every cache off, so each run of the long request
is the same cold prefill -- or one already up with --no-server), greedy, `--no-learn-cost` among
the server arguments so the drafter's block widths do not drift between runs:

  W   the long request alone `--warmups` times (2): the first requests of a process write other
      texts even with the price table held (D6, 15:33), so they only warm it
  A1  the long request alone
  B   the long request again, with short requests fired into it: two while it prefills, then one
      every `--every` seconds while it decodes, alternating opencode's title shape (64 tokens,
      thinking off, streamed) and the router's question shape (24 tokens under a choice grammar
      with log-probabilities, as server/conductor/classify.py sends it); and one raw request over
      the lane's output cap, which waits in the queue as every short request did before
  A2  the long request alone again (the timing the B run is compared to)

It passes when every short request in the lane came back and B and A2 each agree with A1 under
the near-tie rule of docs/exactness.md: their tokens (reasoning and answer, from the streamed
log-probabilities) are A1's up to the first fork, and the fork, if there is one, is a near tie --
the two runs took each other's runner-up and the gap between the two best tokens is at most
`--tie-nats` in both rows. The length router sizes blocks from timings inside the request, so two
runs of the same request can part at a near tie with no bug behind it; a byte compare of the texts
fails there. The fork's position and both gaps are printed; `identical` still says
whether the texts are equal byte for byte. It reports each short request's latency and where it was served (the
server log's `[req] ... lane=` field), the queued one's latency, and the long request's TTFT, decode
rate and wall time in B against A2.

    python tools/lane_check.py --port 8011 --nvfp4 $NV --head $HEAD \\
        --server-arg=--no-learn-cost --server-arg=--narrow-arm=on --out ~/d8/lane.json
    python tools/lane_check.py --no-server --port 8011 --server-log <its log> --out ...
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools import row3  # noqa: E402

TITLE_SYSTEM = (REPO / "tests/fixtures/side/opencode-1.18.34-title-system.txt").read_text()
CLASSES = {
    "code_edit": "change or add code in the user's project",
    "debug": "find and fix a bug or a failing test",
    "design_ui": "design or restyle a user interface",
    "explain": "explain code or a concept without changing anything",
    "chat": "small talk, a question about anything else",
    "plan": "plan a larger piece of work before writing code",
}
QUESTION = ("You sort a user's message into one of these classes. Answer with the class id only.\n"
            + "\n".join(f"- {k}: {v}" for k, v in CLASSES.items()))
USERS = [
    "Build a voxel pagoda viewer in three.js with orbit controls and a day/night toggle.",
    "The checkout test fails with a timeout since the last merge; find out why.",
    "Make the settings page look like the rest of the app, it is still the old grey one.",
    "What does the resident prefix do when a short request arrives between two turns?",
    "Plan the migration of the ledger to a second table before we write any code.",
    "Add a dark mode switch to the header and remember the choice.",
]
PARA = ("The engine serves one sequence at a time. A turn of an agent conversation can think for "
        "minutes, and everything short that arrives meanwhile has to wait unless it is served in "
        "between. This paragraph is filler that makes the prompt long enough for a prefill of "
        "several chunks, so that short requests can arrive while it runs. ")


def _post(port: int, body: dict, timeout: float = 900.0,
          on_first=None) -> tuple[dict, list[float], float]:
    """(the final message or the merged stream, the arrival time of each streamed chunk, t0)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps: list[float] = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not body.get("stream"):
            out = json.loads(r.read())
            stamps.append(time.perf_counter())
            return out, stamps, t0
        merged = {"id": None, "content": "", "reasoning": "", "usage": None, "finish": None,
                  "lp": []}
        for raw in r:
            ln = raw.decode().strip()
            if not ln.startswith("data: {"):
                continue
            ch = json.loads(ln[6:])
            merged["id"] = merged["id"] or ch.get("id")
            if ch.get("usage"):
                merged["usage"] = ch["usage"]
            for c in ch.get("choices") or []:
                merged["lp"] += ((c.get("logprobs") or {}).get("content") or [])
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    stamps.append(time.perf_counter())
                    if len(stamps) == 1 and on_first is not None:
                        on_first()
                merged["content"] += d.get("content") or ""
                merged["reasoning"] += d.get("reasoning_content") or ""
                merged["finish"] = c.get("finish_reason") or merged["finish"]
        return merged, stamps, t0


def main_body(a) -> dict:
    filler = PARA * max(1, a.filler_paras)
    return {"model": "qwen38-spark-engine", "stream": True, "temperature": 0,
            "max_tokens": a.main_tokens, "reasoning_format": "reasoning_content",
            "logprobs": True, "top_logprobs": 2,
            "messages": [{"role": "system", "content": "You are a careful senior engineer.\n\n"
                          + filler},
                         {"role": "user", "content": "Write a small but complete Python module "
                          "that implements an LRU cache with a TTL per entry, thread-safe, with "
                          "unit tests. Think it through first."}]}


def title_body(i: int) -> dict:
    return {"model": "qwen38-spark-engine", "stream": True, "temperature": 0, "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": TITLE_SYSTEM},
                         {"role": "user", "content": "Generate a title for this conversation:\n"},
                         {"role": "user", "content": USERS[i % len(USERS)]}]}


def question_body(i: int, max_tokens: int = 24) -> dict:
    return {"model": "qwen38-spark-engine", "stream": False, "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"choice": list(CLASSES)}, "logprobs": True, "top_logprobs": 3,
            "messages": [{"role": "system", "content": QUESTION},
                         {"role": "user", "content": USERS[i % len(USERS)]}]}


def run_main(a, tag: str, sides: list | None = None) -> dict:
    """The long request; `sides` (shape, at, phase) fire on threads at `at` seconds into the
    prefill (phase 'prefill') or after the first token (phase 'decode')."""
    box: dict = {}
    first = threading.Event()
    shots: list[dict] = []

    def at_first():
        box["t_first"] = time.perf_counter()
        first.set()

    def main():
        out, stamps, t0 = _post(a.port, main_body(a), on_first=at_first)
        box.update(out=out, stamps=stamps, t0=t0, t_end=time.perf_counter())

    def fire(shape: str, i: int):
        body = (title_body(i) if shape == "title" else question_body(i) if shape == "question"
                else question_body(i, max_tokens=a.queued_tokens))
        rec = {"shape": shape, "i": i}
        try:
            out, stamps, t0 = _post(a.port, body)
            rec.update(ms=round((time.perf_counter() - t0) * 1e3, 1),
                       id=out.get("id"), ok=True,
                       text=(out.get("content") if shape == "title" else
                             ((out.get("choices") or [{}])[0].get("message") or {}).get("content")))
        except Exception as exc:                                   # noqa: BLE001
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        rec["main_done"] = "t_end" in box
        shots.append(rec)

    th = threading.Thread(target=main)
    th.start()
    t_start = time.perf_counter()
    side_threads = []
    if sides:
        for shape, at, phase, i in sides:
            if phase == "prefill":
                while time.perf_counter() - t_start < at:
                    time.sleep(0.02)
            else:
                first.wait(600)
                t_first = box.get("t_first", time.perf_counter())
                while time.perf_counter() - t_first < at:
                    time.sleep(0.02)
            t = threading.Thread(target=fire, args=(shape, i))
            t.start()
            side_threads.append(t)
    th.join()
    for t in side_threads:
        t.join()
    out, stamps, t0 = box["out"], box["stamps"], box["t0"]
    usage = out.get("usage") or {}
    n = int(usage.get("completion_tokens") or 0)
    ttft = (stamps[0] - t0) if stamps else None
    dec_s = (box["t_end"] - stamps[0]) if stamps else None
    return {"tag": tag, "id": out.get("id"), "finish": out.get("finish"),
            "completion_tokens": n, "prompt_tokens": usage.get("prompt_tokens"),
            "ttft_s": round(ttft, 3) if ttft else None,
            "wall_s": round(box["t_end"] - t0, 3),
            "decode_tps": round((n - 1) / dec_s, 2) if dec_s and n > 1 else None,
            "text": out.get("content", ""), "reasoning": out.get("reasoning", ""),
            "lp": out.get("lp") or [], "sides": shots}


def _tok(e: dict):
    return tuple(e.get("bytes") or ()) or e.get("token")


def _gap(e: dict) -> float | None:
    tops = sorted((t.get("logprob") for t in e.get("top_logprobs") or []), reverse=True)
    return round(tops[0] - tops[1], 4) if len(tops) >= 2 else None


def _second(e: dict):
    """The runner-up of a row: the best alternative that is not the token taken."""
    tops = sorted(e.get("top_logprobs") or [], key=lambda t: -t.get("logprob", -1e9))
    alts = [_tok(t) for t in tops if _tok(t) != _tok(e)]
    return alts[0] if alts else None


def fork(ref: list, run: list, tie_nats: float) -> dict:
    """Two runs' streamed tokens (`logprobs.content` entries) under the near-tie rule: equal up to
    the first fork, and at the fork each run took the other's runner-up with a top-2 gap within
    `tie_nats` in both rows. `at` is the fork's token index (None: no fork)."""
    n = min(len(ref), len(run))
    at = next((i for i in range(n) if _tok(ref[i]) != _tok(run[i])), None)
    if at is None:
        same_len = len(ref) == len(run)
        return {"at": None if same_len else n, "agrees": True, "near_tie": None,
                "tokens": [len(ref), len(run)],
                **({} if same_len else {"why": "one run stopped earlier with the same tokens"})}
    a, b = ref[at], run[at]
    ga, gb = _gap(a), _gap(b)
    crossed = _second(a) == _tok(b) and _second(b) == _tok(a)
    tie = (crossed and ga is not None and gb is not None and ga <= tie_nats and gb <= tie_nats)
    return {"at": at, "agrees": tie, "near_tie": tie, "gap_nats": [ga, gb],
            "took": [a.get("token"), b.get("token")], "crossed": crossed,
            "tokens": [len(ref), len(run)]}


def lanes_from_log(log: Path | None) -> dict:
    """request id -> (lane, wait_ms, lent_ms) from the server log's `[req]` lines."""
    out: dict = {}
    if log is None or not log.exists():
        return out
    pat = re.compile(r"\[req\] (\S+) .* lane=(\w+) wait_ms=(\d+) lent_ms=(\d+)")
    for ln in log.read_text(errors="replace").splitlines():
        m = pat.search(ln)
        if m:
            out[m.group(1)] = {"lane": m.group(2), "wait_ms": int(m.group(3)),
                               "lent_ms": int(m.group(4))}
    return out


def plan_sides(a, prefill_s: float) -> list:
    """Two shots inside the prefill (at a third and two thirds of A1's TTFT), one queued shot and
    `--decode-sides` lane shots in the decode."""
    sides = [("title", prefill_s / 3, "prefill", 0), ("question", 2 * prefill_s / 3, "prefill", 1)]
    for k in range(a.decode_sides):
        sides.append(("title" if k % 2 == 0 else "question", a.first_at + k * a.every, "decode",
                      k + 2))
        if k == 1:
            sides.append(("queued", a.first_at + k * a.every + 0.5, "decode", 99))
    return sides


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--no-server", action="store_true")
    p.add_argument("--server-log", type=Path, default=None)
    p.add_argument("--server-arg", action="append", default=[])
    p.add_argument("--env", action="append", default=[], metavar="K=V")
    p.add_argument("--repo", type=Path, default=REPO)
    p.add_argument("--python", type=Path, default=row3.DEFAULT_PY)
    p.add_argument("--pythonpath", type=Path, default=row3.HOME / "pylibs")
    p.add_argument("--nvfp4", default=row3.DEFAULT_NV)
    p.add_argument("--head", default=row3.DEFAULT_HEAD)
    p.add_argument("--max-len", type=int, default=262144)
    p.add_argument("--start-timeout", type=int, default=600)
    p.add_argument("--main-tokens", type=int, default=1024)
    p.add_argument("--filler-paras", type=int, default=110,
                   help="filler paragraphs in the long request's system text (~60 tokens each)")
    p.add_argument("--decode-sides", type=int, default=8)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--first-at", type=float, default=2.0)
    p.add_argument("--every", type=float, default=2.5)
    p.add_argument("--queued-tokens", type=int, default=100)
    p.add_argument("--tie-nats", type=float, default=0.125,
                   help="the largest top-2 gap (nats) a fork may have and still count as a near "
                        "tie: one bf16 ulp at a top logit of 16 to 32 (tools/verify_spec.py "
                        "judges in ulps of the logit itself)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keep-server", action="store_true",
                   help="leave the started server up (a row runs against it next, --no-server)")
    a = p.parse_args()

    proc = None
    log = a.server_log
    if not a.no_server:
        ra = argparse.Namespace(
            python=a.python, port=a.port, max_len=a.max_len, len_fixed=0, tree=True, budget=16,
            repo=a.repo, nvfp4=a.nvfp4, head=a.head, served_caches=False, cache_gb=0,
            store="off", with_suffix_store=False, clean_store="", len_latch=True,
            server_arg=list(a.server_arg), start_timeout=a.start_timeout,
            pythonpath=a.pythonpath, mem_floor_gb=10.0)
        log = a.out.with_suffix(".server.log")
        proc = row3.start_server(ra, dict(kv.split("=", 1) for kv in a.env), log)
        # the caller stops a kept server by this pid (its session's group), never by a pattern
        a.out.with_suffix(".server.pid").write_text(str(proc.pid))
    try:
        for w in range(a.warmups):
            wr = run_main(a, f"W{w + 1}")
            print(f"[lane] W{w + 1} ttft {wr['ttft_s']} s, {wr['completion_tokens']} tokens, "
                  f"{wr['decode_tps']} tok/s", flush=True)
        a1 = run_main(a, "A1")
        print(f"[lane] A1 ttft {a1['ttft_s']} s, {a1['completion_tokens']} tokens, "
              f"{a1['decode_tps']} tok/s, wall {a1['wall_s']} s", flush=True)
        sides = plan_sides(a, a1["ttft_s"] or 6.0)
        b = run_main(a, "B", sides)
        print(f"[lane] B  ttft {b['ttft_s']} s, {b['completion_tokens']} tokens, "
              f"{b['decode_tps']} tok/s, wall {b['wall_s']} s, {len(b['sides'])} short requests",
              flush=True)
        a2 = run_main(a, "A2")
        print(f"[lane] A2 ttft {a2['ttft_s']} s, {a2['completion_tokens']} tokens, "
              f"{a2['decode_tps']} tok/s, wall {a2['wall_s']} s", flush=True)
    finally:
        if proc is not None and not a.keep_server:
            row3.stop_server(proc)
    time.sleep(1.0)
    lanes = lanes_from_log(log)
    for s in b["sides"]:
        s.update(lanes.get(s.get("id"), {"lane": None}))
    same = all(x["text"] == a1["text"] and x["reasoning"] == a1["reasoning"] for x in (b, a2))
    alone_same = a2["text"] == a1["text"] and a2["reasoning"] == a1["reasoning"]
    in_lane = [s for s in b["sides"] if s["shape"] != "queued"]
    queued = [s for s in b["sides"] if s["shape"] == "queued"]
    lat = [s["ms"] for s in in_lane if s.get("ok")]
    forks = {"B": fork(a1["lp"], b["lp"], a.tie_nats),
             "A2": fork(a1["lp"], a2["lp"], a.tie_nats)}
    for k, f in forks.items():
        if "took" in f:
            msg = (f"fork at token {f['at']}, took {f['took']!r}, top-2 gap {f['gap_nats']} "
                   f"nats, {'a near tie' if f['near_tie'] else 'NOT a near tie'}")
        elif f["at"] is None:
            msg = f"the same {f['tokens'][0]} tokens"
        else:
            msg = f"the same tokens up to {f['at']}, lengths {f['tokens']}"
        print(f"[lane] {k} vs A1: {msg}", flush=True)
    verdict = {
        "agrees": all(f["agrees"] for f in forks.values()) and bool(a1["lp"]),
        "forks": forks,
        "identical": same,
        "alone_runs_identical": alone_same,
        "b_vs_a1": {"text": b["text"] == a1["text"], "reasoning": b["reasoning"] == a1["reasoning"]},
        "lane_ok": all(s.get("ok") and s.get("lane") in ("decode", "prefill") for s in in_lane),
        "lane_where": {w: sum(1 for s in in_lane if s.get("lane") == w)
                       for w in ("prefill", "decode", "self", None)},
        "lane_ms_p50": round(statistics.median(lat), 1) if lat else None,
        "lane_ms_max": max(lat) if lat else None,
        "queued_ms": [s.get("ms") for s in queued],
        "lent_ms_sum": sum(s.get("lent_ms") or 0 for s in in_lane),
        "main_wall_delta_s": round(b["wall_s"] - a2["wall_s"], 3),
        "main_decode_tps": {"A1": a1["decode_tps"], "B": b["decode_tps"], "A2": a2["decode_tps"]},
        "main_ttft_s": {"A1": a1["ttft_s"], "B": b["ttft_s"], "A2": a2["ttft_s"]},
    }
    verdict["pass"] = verdict["agrees"] and verdict["lane_ok"]
    doc = {"args": {k: str(v) for k, v in vars(a).items()}, "verdict": verdict,
           "runs": [{k: v for k, v in r.items() if k not in ("text", "reasoning", "lp")}
                    for r in (a1, b, a2)],
           "main_chars": {"text": len(a1["text"]), "reasoning": len(a1["reasoning"])}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1))
    print("[lane] " + json.dumps(verdict), flush=True)
    print(f"[lane] {'PASS' if verdict['pass'] else 'FAIL'} -> {a.out}", flush=True)
    sys.exit(0 if verdict["pass"] else 1)


if __name__ == "__main__":
    main()
