"""The shadow reader's router question, asked again offline with other small models (2026-10-05).

The live shadow (Qwen3.5-0.8B) answered the router's class question on real opencode turns and
journaled each answer (`~/.qwen38-spark-engine/conductor/shadow.jsonl`, kind `router`): its
label, its probabilities and the router's own class. On the first test session it agreed 0 of 5
times. This asks the same question at the same places with each model given and reports the
agreement with the router.

Where the question was asked: after the last `<|im_end|>` of the turn's prompt (server/conductor/
shadow.py `router_ask`). The session's ids come from its parked shadow file (`target`: the served
prompts and Qwen's outputs in the small model's ids). A row is placed at the generation prompt
whose end is the row's `prompt_tokens`; a row whose prompt does not end at such a header in the
parked stream (an earlier turn re-rendered since, or ids the mapping dropped) is `unplaced` and
skipped. The live model is the control: its offline answer must give the journaled probabilities
again (`control_max_dp`), else the place is not the live one and the row is reported apart.

Two questions: `live` (the stored router's context-stage question, the classes with their
descriptions, as the live shadow asks) and `labels` (the same question, the classes with their
labels only, a shorter listing).

    python tools/shadow_router_eval.py --models qwen3.5-0.8b,qwen3.5-2b,qwen3.5-4b \\
        --out ~/d9/shadow-router-eval.json

Runs on the GPU beside the engine, under the box lock (the caller's), and stops when a request
from a person's client arrives (the usage ledger; scripts and checks do not stop it). The JSON holds labels, probabilities, counts and ids of rows, nothing
of the conversations' text.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

ROOT = os.path.expanduser("~/.qwen38-spark-engine")
LEDGER = os.path.join(ROOT, "usage", "ledger.sqlite3")


class Abort(Exception):
    pass


#: requests that do not stop the run: scripts and checks (a person's client does)
SCRIPT_KINDS = ("other", "curl", "openai-sdk")


def last_request_ms() -> int:
    """The newest request from a person's client (opencode, Open WebUI, Hermes, the dashboard)."""
    try:
        c = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
        q = ("select max(ts_ms) from requests where client_kind is null or client_kind not in ("
             + ",".join("?" * len(SCRIPT_KINDS)) + ")")
        return int(c.execute(q, SCRIPT_KINDS).fetchone()[0] or 0)
    except Exception:                                              # noqa: BLE001
        return 0


def router_doc(conductor: str) -> dict:
    d = os.path.join(conductor, "router")
    with open(os.path.join(d, "current")) as f:
        v = int(f.read().strip())
    with open(os.path.join(d, f"{v:04d}.json")) as f:
        return json.load(f)["doc"]


def questions(router: dict) -> dict:
    from server.conductor import schema
    q = schema.context_stage(router)["question"]
    live = {c["id"]: c.get("description") or c.get("label") or c["id"]
            for c in router.get("classes") or []}
    labels = {c["id"]: c.get("label") or c["id"] for c in router.get("classes") or []}
    return {"live": (q, live), "labels": (q, labels)}


def tail_of(tok, question: str, choices: dict):
    from server.conductor import incontext
    text, letters = incontext.listing(choices)
    q = question.replace("{classes}", text) if "{classes}" in question else \
        question + "\n\n" + text + "\n\nAnswer with the letter only."
    tail = tok(incontext.TURN.replace("{question}", q), add_special_tokens=False)["input_ids"]
    return list(tail), incontext.letter_ids(tok, letters)


def place(target: list[int], prompt_tokens: int, tok) -> int | None:
    """The question's position for a turn whose served prompt had `prompt_tokens` ids: just after
    the last `<|im_end|>` before the generation prompt that ends there; None when no generation
    prompt ends there in this stream."""
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    asst = tok("assistant\n", add_special_tokens=False)["input_ids"]
    p = int(prompt_tokens)
    if p > len(target):
        return None
    # the generation prompt may carry the opened or closed think block after the header
    for back in range(0, 12):
        h = p - back - len(asst) - 1
        if h >= 0 and target[h] == im_start and target[h + 1:h + 1 + len(asst)] == asst:
            break
    else:
        return None
    for i in range(h - 1, -1, -1):
        if target[i] == im_end:
            return i + 1
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3.5-0.8b,qwen3.5-2b,qwen3.5-4b")
    ap.add_argument("--live-model", default="qwen3.5-0.8b",
                    help="the model the journal's answers came from (the control)")
    ap.add_argument("--journal", default=os.path.join(ROOT, "conductor", "shadow.jsonl"))
    ap.add_argument("--parked", default=os.path.join(ROOT, "cache", "shadow"))
    ap.add_argument("--conductor", default=os.path.join(ROOT, "conductor"))
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-abort", action="store_true")
    a = ap.parse_args()

    import torch
    from safetensors import safe_open
    from engine import shadow as S
    from engine.config import resolve_snapshot
    from engine.decision.device import Device, Placement
    from engine.generative.qwen35 import Qwen35Model
    from server.conductor import incontext

    t_start = int(time.time() * 1000)

    def check():
        if not a.no_abort and last_request_ms() > t_start:
            raise Abort("a request came: stopped")

    rows = [json.loads(ln) for ln in open(a.journal)]
    rows = [r for r in rows if r.get("kind") == "router" and not r.get("error")]
    by_sess = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_sess[r["session"]].append(i)
    targets = {}
    for h in by_sess:
        path = os.path.join(a.parked, h + ".safetensors")
        if os.path.isfile(path):
            with safe_open(path, framework="pt", device="cpu") as f:
                targets[h] = f.get_tensor("target").tolist()
    qs = questions(router_doc(a.conductor))
    engine_dir = resolve_snapshot()
    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": len(rows),
           "sessions": {h: {"rows": len(v), "parked_tokens": len(targets.get(h) or [])}
                        for h, v in by_sess.items()},
           "questions": list(qs), "models": {}, "per_row": []}
    per = [{"i": i, "session": r["session"], "request_id": r.get("request_id"),
            "prompt_tokens": r.get("prompt_tokens"), "router": r.get("router"),
            "journal": {"label": r.get("label"), "confidence": r.get("confidence")},
            "answers": {}} for i, r in enumerate(rows)]
    dev = Device("cuda")
    where = Placement(dev, None)
    try:
        for name in [m.strip() for m in a.models.split(",") if m.strip()]:
            check()
            m = Qwen35Model(name, os.path.join(a.models_dir, name), engine_dir=engine_dir,
                            graph=False)
            t0 = time.perf_counter()
            m.load(where)
            stat = {"load_s": round(time.perf_counter() - t0, 2), "read_rows": 0, "read_ms": 0.0}
            tails = {k: tail_of(m.tok, q, ch) for k, (q, ch) in qs.items()}
            with where.scope():
                rd = S.Reader(m, grid=a.grid)
                for h, idx in by_sess.items():
                    tgt = targets.get(h)
                    if not tgt:
                        continue
                    placed = []
                    for i in idx:
                        pos = place(tgt, rows[i]["prompt_tokens"], m.tok)
                        per[i]["pos"] = pos
                        if pos is not None:
                            placed.append((pos, i))
                    placed.sort()
                    if not placed:
                        continue
                    s = rd.new("eval:" + h)
                    s.target = tgt[:placed[-1][0]]
                    for pos, i in placed:
                        check()
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        stat["read_rows"] += rd.advance(s, stop=pos)
                        torch.cuda.synchronize()
                        stat["read_ms"] += (time.perf_counter() - t) * 1e3
                        for k, (q, ch) in qs.items():
                            tail, lids = tails[k]
                            lp = rd.fork_logits(s, pos, tail, rows=lids)
                            lp = (lp - lp.logsumexp(0)).tolist()
                            label, conf, probs = incontext.answer(lp, list(ch))
                            per[i]["answers"].setdefault(name, {})[k] = {
                                "label": label, "confidence": round(conf, 4),
                                "probs": {c: round(p, 4) for c, p in probs.items()}}
                    del s
                    torch.cuda.empty_cache()
                del rd
            stat["ms_per_1k"] = round(stat["read_ms"] * 1000 / max(1, stat["read_rows"]), 2)
            out["models"][name] = stat
            m.unload() if hasattr(m, "unload") else None
            del m
            torch.cuda.empty_cache()
    except Abort as exc:
        out["aborted"] = str(exc)
    # the control: the live model's offline answer against the journal's
    for p, r in zip(per, rows):
        live = (p["answers"].get(a.live_model) or {}).get("live")
        if live is not None:
            p["control_max_dp"] = round(max(abs(live["probs"].get(c, 0.0) - v)
                                            for c, v in (r.get("probs") or {}).items()), 4)
            p["control_same_label"] = live["label"] == r.get("label")
    out["per_row"] = per
    out["summary"] = summarize(per, a.live_model)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["summary"], indent=1))
    return 0


def summarize(per: list[dict], live_model: str) -> dict:
    """Agreement with the router per model and question: every placed row, the rows the control
    reproduces (max |dp| <= 0.02), and those whose router class Qwen decided on that turn."""
    out = {}
    exact = [p for p in per if p.get("control_max_dp") is not None and p["control_max_dp"] <= 0.02]
    sets = {"placed": [p for p in per if p.get("answers")], "control_exact": exact,
            "router_fresh": [p for p in per if p.get("answers")
                             and (p.get("router") or {}).get("source") == "qwen"]}
    models = sorted({m for p in per for m in p.get("answers") or {}})
    for m in models:
        for q in ("live", "labels"):
            row = {}
            for sname, ps in sets.items():
                got = [(p["answers"][m][q]["label"], (p.get("router") or {}).get("class_id"))
                       for p in ps if m in p["answers"] and q in p["answers"][m]]
                row[sname] = {"n": len(got), "agree": sum(1 for x, y in got if x == y)}
            out[f"{m}/{q}"] = row
    out["journal_live"] = {"n": len(per), "agree": sum(
        1 for p in per if p["journal"]["label"] == (p.get("router") or {}).get("class_id"))}
    out["control"] = {"rows": sum(1 for p in per if "control_max_dp" in p),
                      "exact": len(exact),
                      "same_label": sum(1 for p in per if p.get("control_same_label"))}
    return out


if __name__ == "__main__":
    sys.exit(main())
