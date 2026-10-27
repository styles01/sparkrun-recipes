"""How fresh the shadow reader would be on real sessions (SR1, 2026-10-04): a replay of the usage
ledger's timeline with a measured read rate.

The ledger has every request's start, duration and token counts (no text). A session here is one
client's run of `--profile` requests (opencode's coding by default) without a gap longer than
`--split-min` minutes and without the prompt falling below half of the one before (a compaction or
a new conversation). The replay feeds the shadow as the service does: at a turn's start the prompt
(the new ids since what was read: the prompt's growth over the last prompt, since the template
re-renders the previous answer and the shadow goes back to the checkpoint at the last prompt's
end), at its end Qwen's output; it reads only in the idle gaps, after a poll delay, at `--ms-per-1k`
(tools/shadow_measure.py's replay rate), and stops when the next request starts.

Per turn it reports what the shadow still had to read when the turn began (`behind`, the backlog
before the new prompt; `behind + new prompt` is what a live question at the turn's start would
have had to read first, and `* ms_per_1k` the TTFT that would add), and the time the gap left over.

    python tools/shadow_fresh.py --ms-per-1k 25 [--since 2026-10-02] [--client-kind opencode]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import time

LEDGER = os.path.expanduser("~/.qwen38-spark-engine/usage/ledger.sqlite3")


def sessions(rows, split_min: float):
    out, cur, last = [], [], {}
    for r in rows:
        prev = last.get(r["client_id"])
        new = prev is None or (r["ts_ms"] - (prev["ts_ms"] + (prev["total_ms"] or 0))
                               > split_min * 60000) or \
            (r["prompt_tokens"] or 0) < 0.5 * (prev["prompt_tokens"] or 0)
        if new:
            out.append([])
            last[r["client_id"]] = None
        out[-1].append(r)
        last[r["client_id"]] = r
    return [s for s in out if len(s) >= 3]


def replay(sess, ms_per_1k: float, poll_s: float, grid: int, share: float = 0.0):
    """`read`: the shadow's committed position; `target`: what it was fed. A prompt shorter than
    the last target (the template dropped the last answer's reasoning, or re-rendered it) sends
    the shadow back to its checkpoint at the last prompt's end."""
    read, target, per = 0, 0, []
    prev_prompt = 0
    for i, r in enumerate(sess):
        p = int(r["prompt_tokens"] or 0)
        c = int(r["completion_tokens"] or 0)
        if p < target:                         # the last answer did not come back as written
            read = min(read, prev_prompt // grid * grid)
        behind = max(0, p - read)
        per.append({"behind": max(0, target - read) if p >= target else max(0, prev_prompt - read),
                    "to_prompt": behind, "prompt": p,
                    "live_ttft_ms": round(behind * ms_per_1k / 1000, 1)})
        target = p + c
        prev_prompt = p
        if share > 0:
            # inside the turn: rationed slices, `share` of the turn's time
            # the share counts from Qwen's first token: the decode's time
            cap = int(share * float(r["decode_ms"] or 0) / ms_per_1k * 1000) // grid * grid
            inl = min(max(0, (target - read) // grid * grid), cap)
            per[-1]["read_in_turn"] = inl
            read += inl
        if i + 1 < len(sess):
            end = r["ts_ms"] + (r["total_ms"] or 0)
            gap = max(0.0, (sess[i + 1]["ts_ms"] - end) / 1000.0 - poll_s)
            can = int(gap * 1000.0 / ms_per_1k * 1000) // grid * grid
            done = min(max(0, (target - read) // grid * grid), can)
            per[-1]["gap_s"] = round(gap + poll_s, 2)
            per[-1]["read_in_gap"] = done
            read += done
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms-per-1k", type=float, required=True)
    ap.add_argument("--since", default="2026-10-02")
    ap.add_argument("--client-kind", default="opencode")
    ap.add_argument("--profile", default="coding")
    ap.add_argument("--split-min", type=float, default=20.0)
    ap.add_argument("--poll-s", type=float, default=0.5)
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--inline-share", type=float, default=0.0,
                    help="the share of each turn's time the shadow may read inside it")
    ap.add_argument("--min-prompt", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    since = int(time.mktime(time.strptime(a.since, "%Y-%m-%d")) * 1000)
    c = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "select ts_ms, client_id, prompt_tokens, completion_tokens, total_ms, decode_ms "
        "from requests "
        "where ts_ms >= ? and client_kind = ? and profile = ? and status = 200 order by ts_ms",
        (since, a.client_kind, a.profile))]
    ss = sessions(rows, a.split_min)
    allp, summary = [], []
    for s in ss:
        per = replay(s, a.ms_per_1k, a.poll_s, a.grid, a.inline_share)
        per = [p for p in per if p["prompt"] >= a.min_prompt]
        if not per:
            continue
        allp += per[1:]                                  # the first turn: nothing read before
        summary.append({"turns": len(s), "max_prompt": max(int(r["prompt_tokens"] or 0)
                                                             for r in s),
                        "behind_median": statistics.median(p["behind"] for p in per[1:])
                        if len(per) > 1 else None})
    def q(xs, f):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None
    beh = [p["behind"] for p in allp]
    top = [p["to_prompt"] for p in allp]
    ttft = [p["live_ttft_ms"] for p in allp]
    gaps = sorted(p["gap_s"] for p in allp if "gap_s" in p)
    out = {"ms_per_1k": a.ms_per_1k, "inline_share": a.inline_share, "poll_s": a.poll_s,
           "gap_s_p50": q(gaps, 0.5), "gap_s_p90": q(gaps, 0.9), "sessions": len(summary), "turns": len(allp),
           "fresh_at_start": round(sum(1 for b in beh if b < a.grid) / len(beh), 4) if beh else None,
           "behind_p50": q(beh, 0.5), "behind_p90": q(beh, 0.9), "behind_max": max(beh or [0]),
           "to_prompt_p50": q(top, 0.5), "to_prompt_p90": q(top, 0.9),
           "live_ttft_ms_p50": q(ttft, 0.5), "live_ttft_ms_p90": q(ttft, 0.9),
           "live_ttft_ms_max": max(ttft or [0]), "per_session": summary}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
