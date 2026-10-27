#!/usr/bin/env python3
"""N2's measurement: Qwen3.5 sums up real thoughts, and what that costs Qwen's decode.

    # in a hold (ops/hold.sh: the service stopped), with serve.env's QWEN38_* exported:
    python tools/thought_narrator_measure.py --part summaries [--model qwen3.5-0.8b] [--n 20]
    python tools/thought_narrator_measure.py --part impact [--runs 3]
    # a smoke run of part 1 on the CPU (no hold needed; nice it):
    python tools/thought_narrator_measure.py --part summaries --device cpu --n 3 --hf-check 0

1 SUMMARIES (the small model alone, in this process, through the narrator's own code: the slot,
  `Narrator.around`, `clean_summary`). 20 real thoughts: the persistent suffix store keeps every
  answer the engine wrote as token ids (the transcripts under ~/.qwen38-spark-engine/memory/sessions
  keep a turn's reasoning only as its token count), so the thoughts are the spans between `<think>`
  and `</think>` there, the newest first, one per distinct opening, at least THOUGHT_MIN_TOKENS
  long. Each goes to the model as the live path sends it (the last THOUGHT_WINDOW ids, Qwen's ids
  as they are) and the line is printed for reading: "Thought <n> tokens: <summary>". Per thought:
  the summary's whole time (what Qwen would wait), the prefill of the thought (a job of one token
  on the same prompt) and the decode step (the rest over the tokens). p50/p90 of each. Then the
  decode step from the CUDA graph against the eager step (the same ids?) on the first
  `--graph-check` thoughts, and transformers' Qwen3_5ForCausalLM greedy on the first `--hf-check`
  (the same ids, or the first difference and the reference's top-2 gap there).
2 IMPACT (the candidate server on :8011 with the conductor, the served engine flags from the
  environment, QWEN38_NARRATOR_MODEL set): two header-only profiles copied from `coding`, state
  lines on, `n2-off` (narrator off) and `n2-on` (narrator thoughts), one fixed prompt, greedy,
  `--runs` each alternating after a warm-up each. Qwen's decode rate on the client (completion
  tokens after the first, over the first to the last token that is not a state line: the slices
  between rounds are inside it) and the server's `predicted_per_second`; the end delay (the last
  token to [DONE]: the turn-end allowance); the thought lines the stream carried. The gate: n2-on's
  mean client rate within 3 % of n2-off's.

Writes a JSON report (`--out`, outside the repository by default: it holds summaries of real
thoughts). Exit 0 when part 1 wrote a line for most thoughts and part 2's gate holds.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

DOC_SEP = 1 << 30
GATE = 0.03
PROGRESS_TOKENS = 512
STORE = "~/.qwen38-spark-engine/suffix"
PROMPT = ("Plan a small browser platformer in one HTML file: a canvas, a player with gravity and "
          "jumping, three levels, a fixed-timestep game loop. Think the design through first, "
          "then write the file.")


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(len(xs) * q))], 2) if xs else None


# ------------------------------------------------------------------ 1 the summaries
def thoughts(store: str, open_id: int, close_id: int, n: int, least: int,
             most: int = 40000) -> list[list[int]]:
    """The newest `n` thoughts of the suffix store (ids between `<think>` and `</think>`), one per
    distinct first 64 ids, `least` to `most` ids long."""
    import numpy as np
    a = np.fromfile(os.path.join(os.path.expanduser(store), "tokens.bin"), dtype="<i4")
    opens = np.nonzero(a == open_id)[0][::-1]
    out, seen = [], set()
    for o in opens.tolist():
        end = o + 1 + most
        seg = a[o + 1:end]
        c = np.nonzero((seg == close_id) | (seg == DOC_SEP) | (seg == open_id))[0]
        if not len(c) or seg[c[0]] != close_id:
            continue
        t = seg[:c[0]].tolist()
        if len(t) < least or tuple(t[:64]) in seen:
            continue
        seen.add(tuple(t[:64]))
        out.append(t)
        if len(out) >= n:
            break
    return out


def hf_greedy(path: str, device: str, ids: list[int], n: int, eos: set):
    """transformers' Qwen3_5ForCausalLM on the checkpoint's own tensors: greedy ids, and a function
    for the top-2 gap after a prefix."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5ForCausalLM,
                                                              Qwen3_5TextRotaryEmbedding)
    from engine.loader import Layout
    hf = AutoConfig.from_pretrained(path).text_config
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(hf)
    sd = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(path, name), framework="pt", device=device) as f:
                for key in f.keys():
                    c = Layout.canonical(key)
                    if c is not None and not c.startswith(Layout.mtp_prefix):
                        sd[c if c == "lm_head.weight" else "model." + c] = f.get_tensor(key)
    sd.setdefault("lm_head.weight", sd["model.embed_tokens.weight"])
    model.load_state_dict(sd, strict=False, assign=True)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(hf, device=device)
    model.eval()

    def gap(prefix):
        with torch.inference_mode():
            lg = model(input_ids=torch.tensor(prefix, device=device)[None]).logits[0, -1].float()
        v = lg.topk(2).values
        return float(v[0] - v[1])

    with torch.inference_mode():
        g = model.generate(input_ids=torch.tensor(ids, device=device)[None], max_new_tokens=n,
                           do_sample=False, eos_token_id=sorted(eos))
    out = g[0, len(ids):].tolist()
    while out and out[-1] in eos:
        out.pop()
    return out, gap, model


def summaries(a) -> dict:
    import torch
    from engine.decision.device import Device
    from engine.generative.qwen35 import Qwen35Model
    from engine.generative.slot import Slot
    from server.conductor import narrator as N
    from server.conductor.states import NARRATED_MAX_CHARS
    path = os.path.expanduser(a.dir or f"~/.qwen38-spark-engine/models/{a.model}")
    dev = Device(a.device)
    m = Qwen35Model(a.model, path, engine_dir=a.engine_dir, graph=not a.no_graph)
    slot = Slot(m, role="narrator", gate=threading.Lock(), device=dev, log=print)
    t0 = time.time()
    slot.load(wait=True)
    if slot.state != "loaded":
        raise SystemExit(f"[n2] load failed: {slot.error}")
    st = slot.status()
    print(f"[n2] {a.model} on {dev}: loaded in {time.time() - t0:.1f} s, "
          f"{st['device_bytes'] / 2**30:.2f} GiB, {st['precision']}; Qwen's ids "
          f"{'as they are' if m.same_ids is None else 'through the text: ' + str(m.same_ids)}",
          flush=True)
    narr = N.Narrator(slot, role=lambda: {"mode": "on", "latency_budget_ms": 150})
    head, tail = narr.around()
    th = thoughts(a.store, m.tok.convert_tokens_to_ids("<think>"),
                  m.tok.convert_tokens_to_ids("</think>"), a.n, N.THOUGHT_MIN_TOKENS)
    print(f"[n2] {len(th)} thoughts from the suffix store ({a.store}), {len(head)}-token head, "
          f"{len(tail)}-token tail, window {N.THOUGHT_WINDOW}", flush=True)
    rows = []
    for i, t in enumerate(th):
        body = m.text_ids(t)[-N.THOUGHT_WINDOW:]
        ids = head + body + tail
        r = slot.begin(ids, max_new=N.THOUGHT_MAX_TOKENS, keep=len(head))
        job = r["job"]
        slot.advance(job, budget_ms=60_000)
        pre = slot.begin(ids, max_new=1, keep=len(head))["job"]
        slot.advance(pre, budget_ms=60_000)
        raw = m.job_text(job)
        prefix = f"Thought {len(t):,} tokens: "
        summary, why = N.clean_summary(raw, job.finish, NARRATED_MAX_CHARS - len(prefix))
        n_out = len(job.out)
        step = (job.ms - pre.ms) / max(1, n_out - 1) if n_out > 1 else None
        row = {"thought_tokens": len(t), "window": len(body), "prompt": len(ids),
               "reused": job.reused, "ms": round(job.ms, 1), "prefill_ms": round(pre.ms, 1),
               "step_ms": round(step, 2) if step else None, "tokens": n_out,
               "finish": job.finish, "why": why, "ids": job.out,
               "line": prefix + summary if summary else None, "raw": raw}
        rows.append(row)
        print(f"\n[n2] {i:2d}  {len(t):6,d} tok thought | {row['ms']:7.1f} ms (prefill "
              f"{row['prefill_ms']:.1f}, step {row['step_ms']}) {n_out} tok {job.finish} "
              f"{why or ''}\n      {row['line'] or '(template) ' + raw[:160]}", flush=True)
    out = {"model": a.model, "device": str(dev), "status": slot.status(), "rows": rows,
           "ms_p50": pct([r["ms"] for r in rows], 0.5), "ms_p90": pct([r["ms"] for r in rows], 0.9),
           "prefill_p50": pct([r["prefill_ms"] for r in rows], 0.5),
           "prefill_p90": pct([r["prefill_ms"] for r in rows], 0.9),
           "step_p50": pct([r["step_ms"] for r in rows], 0.5),
           "step_p90": pct([r["step_ms"] for r in rows], 0.9),
           "lines": sum(1 for r in rows if r["line"])}
    print(f"\n[n2] {out['lines']} of {len(rows)} written; a summary p50 {out['ms_p50']} ms p90 "
          f"{out['ms_p90']} ms; the thought's prefill p50 {out['prefill_p50']} p90 "
          f"{out['prefill_p90']} ms; a decode step p50 {out['step_p50']} p90 {out['step_p90']} ms",
          flush=True)
    if m.gs is not None and a.graph_check:
        gs, same, eager = m.gs, 0, []
        m.gs = None
        try:
            for r, t in zip(rows[:a.graph_check], th):
                ids = head + m.text_ids(t)[-N.THOUGHT_WINDOW:] + tail
                job = slot.begin(ids, max_new=N.THOUGHT_MAX_TOKENS, keep=len(head))["job"]
                slot.advance(job, budget_ms=60_000)
                pre = slot.begin(ids, max_new=1, keep=len(head))["job"]
                slot.advance(pre, budget_ms=60_000)
                same += job.out == r["ids"]
                if len(job.out) > 1:
                    eager.append((job.ms - pre.ms) / (len(job.out) - 1))
        finally:
            m.gs = gs
        out["graph_check"] = {"thoughts": min(a.graph_check, len(rows)), "same_ids": same,
                              "eager_step_p50": pct(eager, 0.5)}
        print(f"[n2] the graph against the eager step: {same} of {out['graph_check']['thoughts']} "
              f"the same ids; eager step p50 {out['graph_check']['eager_step_p50']} ms", flush=True)
    elif a.graph_check:
        out["graph_check"] = {"error": m.graph_error or "no graph (the CPU)"}
    if a.hf_check:
        checks = []
        for r, t in list(zip(rows, th))[:a.hf_check]:
            ids = head + m.text_ids(t)[-N.THOUGHT_WINDOW:] + tail
            theirs, gap, model = hf_greedy(path, str(dev.device), ids, N.THOUGHT_MAX_TOKENS, m.eos)
            mine = r["ids"]
            first = next((k for k, (x, y) in enumerate(zip(mine, theirs)) if x != y), None)
            if first is None and len(mine) != len(theirs):
                first = min(len(mine), len(theirs))
            checks.append({"same": first is None, "first_diff": first,
                           "gap_there": gap(ids + theirs[:first]) if first is not None else None})
            print(f"[n2] transformers' greedy: {checks[-1]}", flush=True)
            del model
            if dev.cuda:
                torch.cuda.empty_cache()
        out["hf_check"] = checks
    for r in rows:
        r.pop("ids")
    slot.unload()
    return out


# ------------------------------------------------------------------ 2 the impact
def _profiles(root: str, on_mode: str = "thoughts") -> None:
    """`on_mode`: the on arm's `ui.narrator` (`thoughts`, N2; `live`, v3 the voice, with its
    shipped timings)."""
    from server.conductor import schema
    from server.conductor.store import Store
    s = Store(root)
    base = schema.seed("profiles", "coding")
    for name, narr in (("n2-off", "off"), ("n2-on", on_mode)):
        doc = json.loads(json.dumps(base))
        doc.update(description=f"thought_narrator_measure: coding, state lines on, narrator "
                               f"{narr}; reachable by header only", clients=[])
        doc["ui"] = dict(doc.get("ui") or {}, states="on", narrator=narr)
        # A progress mark every 512 tokens of thinking, on both sides, so the narrator
        # sums up the open thought three times in a 2,000-token run (0.3.23's runs never closed
        # their thought and the narrator wrote nothing before the turn's end)
        doc["ui"]["progress"] = {"mode": "on", "thinking_every_tokens": PROGRESS_TOKENS,
                                 "thinking_every_s": 3600}
        s.put("profiles", name, doc, note=f"N2 measure: narrator {narr}", author="n2-measure")


def one(port: int, profile: str, run: int, max_tokens: int) -> dict:
    from server.conductor.narrator import SENTINEL
    import re
    frame = re.compile(r"\n› ([^\n]*)\n\n")
    body = {"model": "TandemLLM", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": PROMPT}], "temperature": 0, "seed": 7,
            "max_tokens": max_tokens, "reasoning_format": "reasoning_content"}
    hdr = {"Content-Type": "application/json", "X-Tandem-Profile": profile,
           "User-Agent": "opencode/1.18.34 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5",
           "x-session-id": f"n2-{profile}-{run}-{int(time.time())}"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=hdr)
    t0 = time.monotonic()
    first = last = None
    text, usage, timings = [], None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            ln = raw.decode().strip()
            if not ln.startswith("data: {"):
                continue
            ev = json.loads(ln[6:])
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if not piece:
                    continue
                text.append(piece)
                if frame.fullmatch(piece):
                    continue                       # a state line, not one of Qwen's tokens
                now = time.monotonic()
                first = first if first is not None else now
                last = now
    done = time.monotonic()
    said = frame.findall("".join(text))
    n = int((usage or {}).get("completion_tokens") or 0)
    span = (last - first) if first is not None and last is not None else 0.0
    import hashlib
    qwen = frame.sub("", "".join(text))
    tm = timings or {}
    blocks, pms = tm.get("blocks"), tm.get("predicted_ms")
    return {"profile": profile, "run": run, "completion_tokens": n,
            "client_tps": round((n - 1) / span, 3) if span > 0 and n > 1 else None,
            "server_tps": tm.get("predicted_per_second"),
            "blocks": blocks, "predicted_ms": pms,
            "ms_per_block": round(pms / blocks, 3) if blocks and pms else None,
            "text_sha": hashlib.sha256(qwen.encode()).hexdigest()[:16],
            "end_ms": round((done - last) * 1e3, 1) if last is not None else None,
            "lines": said, "thoughts": [s for s in said if s.endswith(SENTINEL)]}


def impact(a) -> dict:
    import row3
    root = tempfile.mkdtemp(prefix="n2-cond-")
    _profiles(root, a.on_mode)
    ns = argparse.Namespace(
        python=Path(a.python), port=a.port, max_len=32768, len_fixed=0, budget=16,
        nvfp4=os.environ.get("NV") or row3.DEFAULT_NV, head=os.environ.get("HEAD") or row3.DEFAULT_HEAD,
        repo=Path(ROOT), pythonpath=Path(a.pythonpath), start_timeout=900, len_latch=True,
        store="off", server_arg=["--conductor=on", f"--conductor-config={root}",
                                 "--reasoning-format=reasoning_content", "--drop-idle",
                                 "--narrow-arm=on",
                                 # The served price table, not learned: with
                                 # learning on, one text took 366 to 504 blocks across runs
                                 *([] if a.learn_cost else ["--no-learn-cost"])],
        mem_floor_gb=10.0, tree=True, with_suffix_store=False)
    # the hold that runs this holds the box lock, which the residency planner reads as "a hold
    # owns the board" and refuses the narrator's load; this server is the hold's own
    env = {"QWEN38_NARRATOR_MODEL": a.model, "BOX_LOCK": os.path.join(root, "no-box-lock"),
           "QWEN38_NARRATOR_DIR": os.path.expanduser(a.dir or f"~/.qwen38-spark-engine/models/"
                                                              f"{a.model}")}
    log = Path(tempfile.mkdtemp(prefix="n2-server-")) / "server.log"
    proc = row3.start_server(ns, env, log)
    rows, out_slot = [], None
    try:
        base = f"http://127.0.0.1:{a.port}/v1/conductor/narrator"
        try:
            urllib.request.urlopen(urllib.request.Request(base + "/load", data=b"{}", headers={
                "Content-Type": "application/json"}), timeout=60).read()
        except urllib.error.HTTPError as exc:
            print(f"[n2] load refused: {exc.code} {exc.read()[:300]!r}", flush=True)
        for _ in range(150):
            with urllib.request.urlopen(base + "/status", timeout=30) as r:
                slot = json.loads(r.read())["slot"]
            if slot["state"] != "loading":
                break
            time.sleep(2)
        out_slot = {k: slot.get(k) for k in ("state", "precision", "error", "refusal", "load_s",
                                              "took_gib", "device_bytes")}
        print(f"[n2] the narrator on :{a.port}: {out_slot}", flush=True)
        if slot["state"] != "loaded":
            # An A/B without the model loaded measures nothing of the narrator
            # (0.3.22's hold: refused_hold, because the server ran from the served tree whose
            # .watchdog.off is the hold's own pause file; run this tool from another tree)
            print("[n2] NOT LOADED: no A/B", flush=True)
            return {"rows": [], "server_log": str(log), "slot": out_slot, "loaded": False,
                    "delta": None, "gate": False, "thoughts": 0}
        for _ in range(a.warmups):
            for p in ("n2-off", "n2-on"):
                w = one(a.port, p, -1, a.max_tokens)
                print(f"[n2] warm-up {p}: {w['completion_tokens']} tok, blocks {w['blocks']}, "
                      f"text {w['text_sha']}", flush=True)
        for i in range(a.runs):
            for p in (("n2-off", "n2-on") if i % 2 == 0 else ("n2-on", "n2-off")):
                r = one(a.port, p, i, a.max_tokens)
                rows.append(r)
                print(f"[n2] {p:6s} run {i}: {r['completion_tokens']} tok, client "
                      f"{r['client_tps']} tok/s, server {r['server_tps']} tok/s, blocks "
                      f"{r['blocks']}, {r['ms_per_block']} ms a block, end {r['end_ms']} ms, "
                      f"text {r['text_sha']}, lines {r['lines']}", flush=True)
    finally:
        row3.stop_server(proc)
    out = {"rows": rows, "server_log": str(log), "slot": out_slot, "loaded": True}
    # the off runs against each other: the noise the delta has to beat:
    # the gate is read only when that noise is under it, on the runs that wrote the same text (the
    # most common one): a different text is different work. 0.3.24's hold: the first two runs of
    # the process wrote other texts (398 and 505 blocks), the next six one text in 514 blocks each
    from collections import Counter
    mode = Counter(r["text_sha"] for r in rows).most_common(1)[0][0] if rows else None
    out["text_runs"] = {p: sum(1 for r in rows if r["profile"] == p and r["text_sha"] == mode)
                        for p in ("n2-off", "n2-on")}
    all_rows, rows = rows, [r for r in rows if r["text_sha"] == mode]

    def spread(p, k):
        xs = [r[k] for r in rows if r["profile"] == p and r[k]]
        return round(max(xs) / min(xs) - 1.0, 4) if len(xs) > 1 else None
    out["off_spread"] = spread("n2-off", "client_tps")
    out["on_spread"] = spread("n2-on", "client_tps")
    out["off_spread_server"] = spread("n2-off", "server_tps")
    out["blocks"] = {p: [r["blocks"] for r in all_rows if r["profile"] == p]
                     for p in ("n2-off", "n2-on")}
    out["same_text"] = len({r["text_sha"] for r in all_rows}) == 1
    out["learn_cost"] = bool(a.learn_cost)
    out["readable"] = (out["off_spread"] is not None and out["off_spread"] <= GATE
                       and min(out["text_runs"].values()) >= 2)
    for p in ("n2-off", "n2-on"):
        xs = [r for r in rows if r["profile"] == p]
        out[p] = {k: round(statistics.mean(r[k] for r in xs if r[k] is not None), 3)
                  if any(r[k] is not None for r in xs) else None
                  for k in ("client_tps", "server_tps", "end_ms")}
    off, on = out["n2-off"]["client_tps"], out["n2-on"]["client_tps"]
    out["delta"] = round(on / off - 1.0, 4) if off and on else None
    out["gate"] = out["readable"] and out["delta"] is not None and out["delta"] >= -GATE
    out["thoughts"] = sum(len(r["thoughts"]) for r in all_rows if r["profile"] == "n2-on")
    out["lines_on"] = sum(len(r["lines"]) for r in all_rows if r["profile"] == "n2-on")
    print(f"\n[n2] decode: off {out['n2-off']}, on {out['n2-on']}; client delta "
          f"{(out['delta'] or 0):+.2%} (gate: within {GATE:.0%}; off runs spread "
          f"{out['off_spread']}, on {out['on_spread']}, "
          f"{'readable' if out['readable'] else 'NOT readable'}) -> "
          f"{'PASS' if out['gate'] else 'FAIL'}; blocks {out['blocks']}, runs with the common text "
          f"{out['text_runs']}; {out['thoughts']} thought lines in the on runs", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="summaries,impact")
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--dir", default=None, help="the checkpoint (default: models/<model>)")
    ap.add_argument("--engine-dir", default=None,
                    help="the served checkpoint, whose tokenizer.json the model's is compared with "
                         "(default: the engine's, engine/config.py)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--graph-check", type=int, default=5)
    ap.add_argument("--hf-check", type=int, default=3)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=2,
                    help="warm-up runs of each arm before the measured ones (0.3.24-test: the "
                         "first two requests of a process wrote other texts)")
    ap.add_argument("--learn-cost", action="store_true",
                    help="the impact server learns its costs as :8000 does (the A/B then does "
                         "not repeat; 0.3.24-test)")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--on-mode", default="thoughts", choices=("thoughts", "live"),
                    help="the on arm's ui.narrator (v3: live, the voice)")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pythonpath", default=os.path.expanduser("~/pylibs"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.engine_dir is None:
        from engine.config import resolve_snapshot
        try:
            a.engine_dir = resolve_snapshot(None)
        except FileNotFoundError:
            a.engine_dir = None
    parts = set(a.part.split(","))
    if a.device == "cpu":
        # the CPU forward: the fused kernels are Triton's (read at engine.model's import)
        for k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE", "GDNPREFILL",
                  "ADDNORM"):
            os.environ[f"QWEN38_FUSED_{k}"] = "0"
        os.environ["QWEN38_DECODE_ATTN"] = "0"
    if a.device != "cpu" or "impact" in parts:
        import row3
        row3._refuse_if_service_is_up()           # one engine on the board: a hold only
    report = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": vars(a)}
    if "summaries" in parts:
        report["summaries"] = summaries(a)
    if "impact" in parts:
        report["impact"] = impact(a)
    out = a.out or os.path.expanduser(f"~/n2-measure/thoughts-{time.strftime('%Y%m%d-%H%M%S')}"
                                      f".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[n2] report: {out}", flush=True)
    s, i = report.get("summaries"), report.get("impact")
    ok = (s is None or s["lines"] * 2 >= len(s["rows"])) and (i is None or i["gate"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
