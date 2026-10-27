#!/usr/bin/env python3
"""Narrator v3 (the voice) on real opencode sessions: what a reader of the lines alone would see.

    # on the machine with opencode (reads its database, writes a private JSON; stdlib only):
    python tools/voice_replay.py export --session ses_... [--db ~/.local/share/opencode/opencode.db]
           --out /private/dir/session.json
    # on the box (the small model on the CPU or a GPU, the served tokenizer; nice it):
    python tools/voice_replay.py run session.json [--device cpu] [--model qwen3.5-0.8b]
           [--lang auto|en|de] [--out lines.json]

EXPORT reads one session's assistant steps from opencode's database: each step's reasoning parts
(the state-line titles taken out with states.strip_text), its text parts and its tool calls (name
and input), each with the times opencode stamped (a part's creation, its end). The user's message
before each step goes along (its language). The output holds the user's own text: keep it out
of the repository, delete it after.

RUN replays each step through the voice's own code (server/conductor/voice.py `Voice`, the
narrator's slot, states.States) on a simulated clock: the reasoning's tokens (the served 27B
tokenizer) spread evenly over the part's recorded time, the answer's text pieces likewise, each
tool call written as Qwen writes it (`<function=NAME><parameter=KEY>...`) over the time from its
part's creation to the next part. The small model's own time does not move the simulated clock
(the CPU is ~15 times slower than the GPU); the share is given at the GPU's measured cost per
line (`--gpu-ms`, the clock is advanced by it per job). Prints every line in order with its
simulated time, the task and what the model was given; writes them as JSON (`--out`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------- export
def export(db: str, session: str) -> dict:
    from server.conductor import states
    con = sqlite3.connect(os.path.expanduser(db))
    msgs = con.execute("select id, data from message where session_id=? order by time_created, id",
                       (session,)).fetchall()
    parts = con.execute("select message_id, time_created, data from part where session_id=? "
                        "order by id", (session,)).fetchall()
    by_msg: dict[str, list] = {}
    for mid, tc, data in parts:
        by_msg.setdefault(mid, []).append((tc, json.loads(data)))
    steps, user = [], ""
    for mid, data in msgs:
        m = json.loads(data)
        ps = by_msg.get(mid, [])
        if m.get("role") == "user":
            user = " ".join(p.get("text") or "" for _, p in ps if p.get("type") == "text")
            continue
        step = None
        for tc, p in ps:
            t = p.get("type")
            if t == "step-start":
                step = {"user": user, "events": []}
                steps.append(step)
                continue
            if step is None:
                continue
            tm = p.get("time") or {}
            if t == "reasoning":
                text = states.strip_text(p.get("text") or "")
                text = re.sub(r"\A\s+", "", text)
                if text.strip():
                    step["events"].append({"type": "reasoning", "text": text,
                                           "t0": tm.get("start") or tc, "t1": tm.get("end") or tc})
            elif t == "text":
                if (p.get("text") or "").strip():
                    step["events"].append({"type": "text", "text": p["text"],
                                           "t0": tm.get("start") or tc, "t1": tm.get("end") or tc})
            elif t == "tool":
                st = p.get("state") or {}
                step["events"].append({"type": "tool", "name": p.get("tool"),
                                       "input": st.get("input") or {}, "t0": tc, "t1": None})
            elif t == "step-finish":
                step["t_end"] = tc
    # a tool call streams until the next event of its step (or the step's end); a text part
    # opencode keeps open across the calls after it ends where the next event begins
    for s in steps:
        ev = s["events"]
        for i, e in enumerate(ev):
            nxt = next((x["t0"] for x in ev[i + 1:]), None)
            if e["type"] == "tool":
                e["t1"] = nxt or s.get("t_end") or e["t0"] + 1000
            elif nxt is not None and e["t1"] > nxt:
                e["t1"] = max(e["t0"], nxt)
    return {"session": session, "steps": [s for s in steps if s["events"]]}


# ---------------------------------------------------------------------- run
def call_text(name: str, args: dict) -> str:
    out = f"<tool_call>\n<function={name}>\n"
    for k, v in args.items():
        v = v if isinstance(v, str) else json.dumps(v)
        out += f"<parameter={k}>\n{v}\n</parameter>\n"
    return out + "</function>\n</tool_call>"


# Qwen's decode rate for placing a part's tokens: a reasoning part opens at its title (a queue or
# a prefill line), long before its first token, so its tokens end at the part's end at this rate
RATE = 45.0


def spread(a: float, b: float, n: int) -> tuple[float, float]:
    """(start, span) of `n` tokens that end at `b`, at most RATE a second, not before `a`."""
    span = max(0.001, min(b - a, n / RATE))
    return b - span, span


class SimClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def run(path: str, *, device: str, model: str, lang: str, gpu_ms: float, only: int | None,
        out: str | None, settings: dict | None = None, skip: int = 0) -> int:
    from transformers import AutoTokenizer
    from engine import generative
    from engine.generative.slot import Slot
    from server.conductor import narrator as narr_mod, states, voice as voice_mod
    from engine.settings import SETTINGS as _S

    data = json.load(open(path))
    from engine.config import resolve_snapshot
    engine_dir = resolve_snapshot()
    tok = AutoTokenizer.from_pretrained(engine_dir)
    m = generative.model(model, _S.get("NARRATOR_DIR") or None, engine_dir=engine_dir)
    where = None
    if device != "cpu":
        from engine.decision.device import Device, Placement
        where = Placement(Device(device), None)
    t = time.time()
    m.load(where)
    print(f"[replay] {model} on {device}: loaded in {time.time() - t:.1f}s, same ids: "
          f"{m.same_ids is None}", flush=True)
    clock = SimClock()

    # the slot without a planner or a lock; the model's own clock is wall time (its deadline),
    # the voice's is simulated
    slot = Slot(m, role="narrator", gate=None, log=lambda s: None)
    slot.state, slot.loaded_at, slot._where = "loaded", time.time(), where
    budget = {"ms": gpu_ms}
    real_advance = slot.advance

    def advance(job, *, budget_ms):
        if device != "cpu":
            # a GPU: the real slices, and the simulated clock moves by what they took
            r = real_advance(job, budget_ms=budget_ms)
            clock.t += float(r.get("ms") or 0.0) / 1000.0
            return r
        # the CPU writes the whole phrase in one go (no wall deadline), and the simulated clock
        # moves by the GPU's cost of it
        r = real_advance(job, budget_ms=1e9)
        r["ms"] = budget["ms"] if r.get("done") else 1.0
        clock.t += r["ms"] / 1000.0
        return r

    slot.advance = advance
    narr = narr_mod.Narrator(slot)
    open_id, close_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    lines = []
    steps = data["steps"][skip:skip + only] if only else data["steps"][skip:]
    t_base = None
    for si, step in enumerate(steps):
        lg = lang if lang != "auto" else voice_mod.lang_of([{"role": "user",
                                                             "content": step["user"]}])
        ui = states.States({}, voice=lg)
        v = voice_mod.Voice(narr, ui, lang=lg, opened=False, close_id=close_id,
                            settings=settings,
                            decode=lambda xs: tok.decode(xs, skip_special_tokens=True),
                            clock=clock)
        ids: list[int] = []
        clock.t = 0.0
        t0 = step["events"][0]["t0"] / 1000.0
        if t_base is None:
            t_base = t0
        in_think = False
        answer_open = None

        def drain(at: float):
            for e in ui._out():
                text = e.get("summary") or e.get("shown") or e["text"]
                info = e.get("narrator") or {}
                lines.append({"step": si, "t": round(at, 1), "kind": e["kind"], "text": text,
                              "task": info.get("task"), "why": info.get("why"),
                              "ms": info.get("ms"), "slices": info.get("slices"),
                              "tokens": info.get("tokens"), "prompt_tokens": info.get("prompt_tokens")})
                print(f"  {at:8.1f}s  step {si:3d}  › {text}", flush=True)

        for e in step["events"]:
            a, b = e["t0"] / 1000.0, (e["t1"] or e["t0"]) / 1000.0
            if e["type"] == "reasoning":
                toks = tok(e["text"], add_special_tokens=False)["input_ids"]
                if not in_think:
                    ids.append(open_id)
                    in_think = True
                a, span = spread(a, b, len(toks))
                for i, x in enumerate(toks):
                    clock.t = max(clock.t, a - t0 + span * (i + 1) / len(toks))
                    ids.append(x)
                    v.token(ids, True)
                    drain(clock.t + t0 - t_base)
            else:
                if in_think:
                    ids.append(close_id)
                    in_think = False
                    v.token(ids, False)
                    drain(clock.t + t0 - t_base)
                if e["type"] == "text":
                    toks = tok(e["text"], add_special_tokens=False)["input_ids"]
                    a, span = spread(a, b, len(toks))
                    for i, x in enumerate(toks):
                        clock.t = max(clock.t, a - t0 + span * (i + 1) / len(toks))
                        ids.append(x)
                        v.token(ids, False)
                        v.content(tok.decode([x]))
                        v.tool(ids, None)
                        drain(clock.t + t0 - t_base)
                else:
                    text = call_text(e["name"], e["input"])
                    body = text[len("<tool_call>"):]
                    toks = tok(text, add_special_tokens=False)["input_ids"]
                    a, span = spread(a, b, len(toks))
                    acc = ""
                    for i, x in enumerate(toks):
                        clock.t = max(clock.t, a - t0 + span * (i + 1) / len(toks))
                        ids.append(x)
                        acc += tok.decode([x])
                        v.token(ids, False)
                        blk = acc[len("<tool_call>"):] if acc.startswith("<tool_call>") else None
                        if blk is not None and "</tool_call>" in blk:
                            blk = None
                        v.tool(ids, blk)
                        drain(clock.t + t0 - t_base)
                    v.tool(ids, None)
                    del body
        v.finish(ids)
        drain(clock.t + t0 - t_base)
        for tr in v.trace:
            if tr.get("phrase") is None:
                print(f"    rejected {tr['task']}: {tr.get('why')} {tr.get('raw', '')!r}", flush=True)
        if os.environ.get("VOICE_DEBUG"):
            print("    calls:", [{k: c.get(k) for k in ("name", "args", "lines", "t_target",
                                                         "t_closed", "said")}
                              for c in v.tools.calls], flush=True)
        tot = v.totals()
        print(f"  [step {si}] {len(ids)} ids, lines {tot['lines']} (phrases {tot['phrases']}, "
              f"plain {tot['plain']}), why {tot['why']}", flush=True)
    ms = sorted(x["ms"] for x in lines if x.get("ms"))
    if ms:
        print(f"[replay] {len(lines)} lines, {len(ms)} phrases: ms p50 {ms[len(ms) // 2]:.0f}, "
              f"p90 {ms[min(len(ms) - 1, int(len(ms) * 0.9))]:.0f}, max {ms[-1]:.0f} "
              f"(the model's time per phrase, all slices)", flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(lines, f, indent=1, ensure_ascii=False)
        os.chmod(out, 0o600)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--db", default="~/.local/share/opencode/opencode.db")
    e.add_argument("--session", required=True)
    e.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("path")
    r.add_argument("--device", default="cpu")
    r.add_argument("--model", default="qwen3.5-0.8b")
    r.add_argument("--lang", default="auto")
    r.add_argument("--gpu-ms", type=float, default=200.0)
    r.add_argument("--only", type=int, default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--skip", type=int, default=0)
    r.add_argument("--set", action="append", default=[],
                   help="a voice setting, e.g. every_s=30 (repeatable)")
    a = ap.parse_args()
    if a.cmd == "run" and a.device == "cpu":
        for k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
            os.environ[f"QWEN38_FUSED_{k}"] = "0"          # the CPU forward
    if a.cmd == "export":
        d = export(a.db, a.session)
        with open(a.out, "w") as f:
            json.dump(d, f, ensure_ascii=False)
        os.chmod(a.out, 0o600)
        print(f"[export] {len(d['steps'])} steps -> {a.out}")
        return 0
    settings = {k: float(v) for k, v in (x.split("=", 1) for x in a.set)}
    return run(a.path, device=a.device, model=a.model, lang=a.lang, gpu_ms=a.gpu_ms,
               only=a.only, out=a.out, settings=settings, skip=a.skip)


if __name__ == "__main__":
    raise SystemExit(main())
