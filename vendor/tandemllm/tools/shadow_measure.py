"""The shadow reader's measurements on the GPU (SR1, 2026-10-04): cost, memory, exactness, forks.

Runs the small model on its own (beside an idle engine, under the box lock and the idle rule, as
the B2/B3 measurements ran), with the served process's arithmetic switches (`--env-from PID`
copies its QWEN38_* flags, nothing else), over a real conversation's token ids (`--tokens`: an
int64 file, e.g. a spill entry's tokens.bin) split into turns at each generation prompt
(`<|im_start|>assistant`), so turn k's prompt is the stream up to its assistant header and its
output the ids up to the next `<|im_end|>`.

    python tools/shadow_measure.py --model ~/.qwen38-spark-engine/models/qwen3.5-0.8b \\
        --tokens <entry>/tokens.bin --engine-dir <27B dir> --out shadow-0.8b.json

What it measures (each part's numbers in the JSON, nothing of the conversation's text):
  cost      ms per grid chunk and per 1,000 tokens at several grids and depths (cold weights
            excluded: one warm-up chunk per shape), the longest chunk (what a request that
            arrives during a burst waits at most)
  replay    the conversation read turn by turn as the service reads it: rows read per turn,
            ms per turn, rollbacks (none expected: the stream only grows)
  exact     the replayed state against a fresh prefill of the same ids on the same grid (bit for
            bit: KV rows, recurrent and convolution state), and against a fresh prefill on
            another grid (max difference; top-1 of question forks)
  memory    bytes per session, per token of KV, the recurrent state, the checkpoints
  fork      the router question at the end of the session (ms), the narrator's thought summary
            (prefill of its question and 40 eager greedy steps: ms per step)
  abort     between parts it reads the usage ledger: a request after the start stops the run
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

LEDGER = os.path.expanduser("~/.qwen38-spark-engine/usage/ledger.sqlite3")
KEEP_ENV = ("QWEN38_FUSED", "QWEN38_DECODE_ATTN", "QWEN38_GDN", "QWEN38_GQA_FROM",
            "QWEN38_LOWER_RIGHT_FROM", "QWEN38_PREFILL", "QWEN38_TWO_STREAM", "QWEN38_KV_FP8",
            "QWEN38_HOST_ASYNC", "QWEN38_CHUNK")


def env_from(pid: int) -> dict:
    """The served process's arithmetic switches (QWEN38_* that change the forward), no secrets."""
    out = {}
    with open(f"/proc/{pid}/environ", "rb") as f:
        for kv in f.read().split(b"\0"):
            k, _, v = kv.decode(errors="replace").partition("=")
            if k.startswith(KEEP_ENV):
                out[k] = v
    return out


def last_request_ms() -> int:
    try:
        c = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
        return int(c.execute("select max(ts_ms) from requests").fetchone()[0] or 0)
    except Exception:                                              # noqa: BLE001
        return 0


class Abort(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-from", type=int, default=None)
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--grids", default="256,512,1024,2048")
    ap.add_argument("--depths", default="0,16384,40960")
    ap.add_argument("--other-grid", type=int, default=2048)
    ap.add_argument("--parts", default="cost,replay,exact,memory,fork")
    ap.add_argument("--narr", type=int, default=8, help="the last N thoughts the narrator part sums up")
    a = ap.parse_args()
    if a.env_from:
        os.environ.update(env_from(a.env_from))
    import numpy as np
    import torch
    from engine import shadow as S
    from engine.decision.device import Device, Placement
    from engine.generative.qwen35 import Qwen35Model

    t_start = int(time.time() * 1000)

    def check():
        if last_request_ms() > t_start:
            raise Abort("a request came: stopped")

    res: dict = {"model": os.path.basename(a.model.rstrip("/")), "grid": a.grid,
                 "env": sorted(k for k in os.environ if k.startswith("QWEN38_")),
                 "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    dev = Device("cuda")
    where = Placement(dev, None)
    m = Qwen35Model(os.path.basename(a.model.rstrip("/")), a.model, engine_dir=a.engine_dir,
                    graph=False)
    t0 = time.perf_counter()
    m.load(where)
    res["load_s"] = round(time.perf_counter() - t0, 2)
    res["same_ids"] = m.same_ids
    ids_all = np.fromfile(a.tokens, dtype="<i8").tolist()

    def sync():
        torch.cuda.synchronize()

    try:
        with where.scope():
            rd = S.Reader(m, grid=a.grid)
            stream = rd.map_ids(ids_all)
            res["tokens"] = {"raw": len(ids_all), "mapped": len(stream),
                             "dropped": len(ids_all) - len(stream)}
            # the turns: prompt ends after each "<|im_start|>assistant\n"
            tok = m.tok
            im_start = tok.convert_tokens_to_ids("<|im_start|>")
            im_end = tok.convert_tokens_to_ids("<|im_end|>")
            asst = tok("assistant\n", add_special_tokens=False)["input_ids"]
            ends = []
            for i, t in enumerate(stream):
                if t == im_start and stream[i + 1:i + 1 + len(asst)] == asst:
                    ends.append(i + 1 + len(asst))
            turns = []
            for k, pe in enumerate(ends):
                nxt = next((j for j in range(pe, len(stream)) if stream[j] == im_end), None)
                oe = (nxt + 1) if nxt is not None else len(stream)
                turns.append((pe, oe))
            res["turns"] = len(turns)
            parts = set(a.parts.split(","))

            if "cost" in parts:
                check()
                cost = []
                for g in [int(x) for x in a.grids.split(",")]:
                    for d in [int(x) for x in a.depths.split(",")]:
                        if d + 4 * g > len(stream):
                            continue
                        r2 = S.Reader(m, grid=g)
                        s = r2.new("cost")
                        s.target = stream[:d + 4 * g]
                        # reach the depth in large chunks (not timed), then time 4 chunks
                        if d:
                            big = S.Reader(m, grid=g)
                            s.target = stream[:d]
                            big.advance(s)
                            s.target = stream[:d + 4 * g]
                        r2.advance(s, max_chunks=1)            # warm the shape
                        sync()
                        ms = []
                        for _ in range(3):
                            sync()
                            t = time.perf_counter()
                            r2.advance(s, max_chunks=1)
                            sync()
                            ms.append((time.perf_counter() - t) * 1e3)
                        cost.append({"grid": g, "depth": s.pos - 3 * g,
                                     "chunk_ms_median": round(statistics.median(ms), 2),
                                     "chunk_ms_max": round(max(ms), 2),
                                     "ms_per_1k": round(statistics.median(ms) * 1000 / g, 2)})
                        del s
                        torch.cuda.empty_cache()
                res["cost"] = cost

            s = None
            if parts & {"replay", "exact", "memory", "fork"}:
                check()
                s = rd.new("replay")
                per = []
                for (pe, oe) in turns:
                    t = time.perf_counter()
                    r_prompt = rd.sync(s, stream[:pe])
                    rd.mark(s, pe)
                    n1 = rd.advance(s)
                    rd.sync(s, stream[:oe])
                    n2 = rd.advance(s)
                    sync()
                    per.append({"prompt_end": pe, "read": n1 + n2,
                                "ms": round((time.perf_counter() - t) * 1e3, 2),
                                "rolled_back": r_prompt})
                    check()
                rd.sync(s, stream)
                rd.advance(s)
                sync()
                reads = [p["read"] for p in per if p["read"]]
                res["replay"] = {"turns": len(per), "read_total": s.pos,
                                 "ms_total": round(sum(p["ms"] for p in per), 1),
                                 "ms_per_1k": round(sum(p["ms"] for p in per) * 1000
                                                    / max(1, sum(reads)), 2),
                                 "rollbacks": s.stats["rollbacks"],
                                 "checkpoints": [c[0] for c in s.checks],
                                 "per_turn_sample": per[:3] + per[-3:]}

            if "exact" in parts:
                check()
                t = time.perf_counter()
                f = rd.fresh(s.ids)
                sync()
                fresh_ms = (time.perf_counter() - t) * 1e3
                same = S.same_state(s, f)
                del f
                torch.cuda.empty_cache()
                other = S.Reader(m, grid=a.other_grid)
                n_other = len(s.ids) // a.other_grid * a.other_grid
                f2 = other.fresh(s.ids[:n_other])
                # bring the other grid's state to the same position with the reader's grid
                f2.target = list(s.ids)
                rd.advance(f2)
                diff = S.same_state(s, f2)
                # top-1 of question forks on both: the next real tokens as probes
                tail_q = tok("\n<|im_start|>user\nWhat is the next step?<|im_end|>\n"
                             "<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
                s.target, f2.target = stream, stream
                probes, agree = 0, 0
                kl = []
                for cut in range(max(s.pos - 8 * 64, 0), s.pos, 64):
                    lg_a = rd.fork_logits(s, s.pos, tail_q + stream[cut:cut + 32])
                    lg_b = rd.fork_logits(f2, f2.pos, tail_q + stream[cut:cut + 32])
                    probes += 1
                    agree += int(int(lg_a.argmax()) == int(lg_b.argmax()))
                    pa = torch.log_softmax(lg_a, -1)
                    pb = torch.log_softmax(lg_b, -1)
                    kl.append(float((pa.exp() * (pa - pb)).sum()))
                res["exact"] = {"fresh_same_grid": same, "fresh_ms": round(fresh_ms, 1),
                                "other_grid": a.other_grid, "vs_other_grid": diff,
                                "probes": probes, "top1_agree": agree,
                                "kl_max": round(max(kl), 6) if kl else None}
                del f2
                torch.cuda.empty_cache()
                check()

            if "memory" in parts:
                n = s.pos
                res["memory"] = {"tokens": n, "session_bytes": s.nbytes(),
                                 "kv_bytes_per_token": s.kv.row_bytes(),
                                 "kv_cap_rows": s.kv.cap, "gdn_bytes": s.gdn.nbytes,
                                 "checkpoints": len(s.checks),
                                 "gib": round(s.nbytes() / 2**30, 3),
                                 "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                                 "weights_gib": round(m.device_bytes() / 2**30, 3)}

            if "fork" in parts:
                check()
                from server.conductor import incontext, schema
                router = schema.seed("router")
                classes = {c["id"]: c["description"] for c in router["classes"]}
                text, letters = incontext.listing(classes)
                q = schema.context_stage(router)["question"].replace("{classes}", text)
                tail = tok(incontext.TURN.replace("{question}", q),
                           add_special_tokens=False)["input_ids"]
                lids = incontext.letter_ids(tok, letters)
                s.target = stream
                ms = []
                for _ in range(3):
                    sync()
                    t = time.perf_counter()
                    lp = rd.fork_logits(s, len(stream), tail, rows=lids)
                    sync()
                    ms.append((time.perf_counter() - t) * 1e3)
                from server.conductor import shadow as svc_mod
                ttail = tok(svc_mod.THOUGHT_TAIL.format(q=svc_mod.THOUGHT_QUESTION),
                            add_special_tokens=False)["input_ids"]
                sync()
                t = time.perf_counter()
                fk = rd.fork(s, len(stream), ttail)
                fk.prefill()
                sync()
                t_pre = (time.perf_counter() - t) * 1e3
                t = time.perf_counter()
                fk.decode(40, set(), None)
                sync()
                t_dec = (time.perf_counter() - t) * 1e3
                fk.close()
                res["fork"] = {"router_question_tokens": len(tail),
                               "pending_tail": len(stream) - s.pos,
                               "router_ms_median": round(statistics.median(ms), 2),
                               "router_classes": len(classes),
                               "thought_question_tokens": len(ttail),
                               "thought_prefill_ms": round(t_pre, 2),
                               "thought_decode_ms_per_token": round(t_dec / 40, 2)}
            if "narr" in parts:
                # the narrator's summary of each thought read from the shadow (SR1): one pass over
                # the conversation, a fork at the end of every thought of at least 128 tokens
                check()
                from server.conductor import narrator as narr_mod, shadow as svc_mod
                o_id = tok.convert_tokens_to_ids("<think>")
                c_id = tok.convert_tokens_to_ids("</think>")
                spans, open_at = [], None
                for i, t in enumerate(stream):
                    if t == o_id:
                        open_at = i + 1
                    elif t == c_id and open_at is not None:
                        if i - open_at >= narr_mod.THOUGHT_MIN_TOKENS:
                            spans.append((open_at, i))
                        open_at = None
                spans = spans[-a.narr:]
                ttail = tok(svc_mod.THOUGHT_TAIL.format(q=svc_mod.THOUGHT_QUESTION),
                            add_special_tokens=False)["input_ids"]
                ns = rd.new("narr")
                outs = []
                for (lo, hi) in spans:
                    rd.sync(ns, stream[:hi])
                    rd.advance(ns)
                    sync()
                    t = time.perf_counter()
                    ids_out, fin = rd.generate(ns, hi, ttail, narr_mod.THOUGHT_MAX_TOKENS,
                                               m.eos)
                    sync()
                    raw = m.decode(ids_out)
                    line, why = narr_mod.clean_summary(raw, fin, 400)
                    outs.append({"thought_tokens": hi - lo, "at": hi, "ms": round(
                        (time.perf_counter() - t) * 1e3, 1), "tokens": len(ids_out),
                        "finish": fin, "why": why, "line": line, "raw": raw[:300]})
                    check()
                res["narr"] = outs
    except Abort as exc:
        res["aborted"] = str(exc)
    res["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("env",)}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
