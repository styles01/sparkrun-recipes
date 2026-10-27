"""How well the conductor's classifiers pick the task class: accuracy, confusion and latency per stage.

    python tools/conductor_eval.py --decider http://127.0.0.1:8010 --qwen http://127.0.0.1:8000 \\
        [--set tests/fixtures/conductor/task_prompts.jsonl --set ...] [--digest today,sized,line] \\
        [--config ~/.qwen38-spark-engine/conductor] [--out ~/.qwen38-spark-engine/conductor/evals/...]

    # Laya in this process on the box CPU, no server touched (SRV-153's measurement):
    python tools/conductor_eval.py --local-deciders laya-english --device cpu --decider self ...

Every prompt of each labelled set goes to each stage once, with the class descriptions, question
and thresholds of the router document the server loads (`--config`, else the built-in one). The
cascade is then derived the way `Router.cascade` runs it: the first stage whose answer reaches its
threshold decides; nobody sure is "no answer". `router` is the session's class after the turn:
the cascade's answer, kept at the previous class when it is below the stickiness threshold, the
previous class when nobody is sure (a prompt with a `context` carries its session's previous
class; one without is a session's first message).

The digest, per `--digest` variant (SRV-153):
  today   the 2,000-character head-and-tail digest for both stages, which the decider then cuts to
          its own limits from the front (the router before SRV-153)
  sized   each stage's own digest: the decider's sized to its `limits()`, Qwen's to digest_chars
  line    sized, plus the context line built from the prompt's `context` (prev, conf, tools,
          exts, checker, compaction); `prev=none ...` for a prompt without one
  router  what the router does under the router document: `line` when its classifier has
          `context_line` on, else `sized` (the default)

Per set, variant and stage: `accuracy` (right answers over all prompts; an answer below the
threshold counts as none), `answered`, `accuracy_answered`, `top1` (the label with the highest
probability, threshold ignored), p50 and p95 latency, and the confusion table (true class ->
answer, "—" for none).

One counting, four numbers (`counting`, a model stage's): `all_right` (its best guess right, over
all prompts, the threshold ignored: `top1`), `confident_right` (right AND at or above the
threshold, over all prompts: `accuracy`), `coverage` (the share at or above the threshold) and
`confident_precision` (right among the confident: `accuracy_answered`), with the `threshold` they
were counted at, and `errors` (answers that failed, a timeout among them: they are neither right
nor confident, so a run with errors is not a count of the model). A percentage without its
counting is not a number: on 2026-10-01 the same
Laya answers read 63.9 % (confident_right at 0.60) in the morning and 50.0 % (confident_right at
0.80) at night, both 78.2 % all_right.

    # re-count a report's rows at other thresholds, no model asked
    python tools/conductor_eval.py --rescore evals/classifier-....json --at 0.6,0.8 The first set under the first variant is also the report's top-level
`set` and `stages`, which the dashboard's Compare screen reads from the box's `evals/`
directory; `variants` holds every set and variant. The labelled sets are written from task
types, not from anyone's chats.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import schema, signals  # noqa: E402
from server.conductor.classify import Qwen, decider_for  # noqa: E402
from server.conductor.store import Store  # noqa: E402

NONE = "—"
VARIANTS = ("today", "sized", "line", "router")


def pct(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    return round(v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))], 1)


def counting(rows: list[dict], key: str, threshold: float) -> dict:
    """The four numbers of one model stage at `threshold`, from its per-prompt answers (the
    stage's probability, not the run's own `accepted`, so a report re-counts at any threshold)."""
    n = len(rows)
    got = [r for r in rows if r.get(key) is not None]
    sure = [r for r in got if r[key].get("error") is None and r[key].get("label") is not None
            and float(r[key].get("confidence") or 0.0) >= threshold]
    right = sum(r[key]["label"] == r["label"] for r in sure)
    best = sum(r[key].get("label") == r["label"] for r in got)
    errors = sum(r[key].get("error") is not None for r in got)
    return {"threshold": threshold, "n": n, "errors": errors, "confident": len(sure),
            "confident_right_n": right,
            "all_right": round(best / n, 4) if n else 0.0,
            "confident_right": round(right / n, 4) if n else 0.0,
            "coverage": round(len(sure) / n, 4) if n else 0.0,
            "confident_precision": round(right / len(sure), 4) if sure else None}


def summarise(rows: list[dict], key: str, labels: list[str]) -> dict:
    """One stage's figures from the per-prompt records."""
    conf = {t: {} for t in labels}
    correct = answered = top1 = 0
    ms = []
    for r in rows:
        a = r[key]
        if a is None:
            continue
        pred = a["label"] if a.get("accepted") else NONE
        conf[r["label"]][pred] = conf[r["label"]].get(pred, 0) + 1
        correct += pred == r["label"]
        answered += pred != NONE
        top1 += a.get("label") == r["label"]
        ms.append(a["ms"])
    n = len(rows)
    return {"n": n, "answered": answered, "correct": correct,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "accuracy_answered": round(correct / answered, 4) if answered else None,
            "top1": round(top1 / n, 4) if n else 0.0,
            "p50_ms": pct(ms, 0.5), "p95_ms": pct(ms, 0.95), "confusion": conf}


def line_of(item: dict) -> str:
    """The context line of a labelled prompt: its `context`, or a session's first message."""
    c = item.get("context") or {}
    chk = c.get("checker")
    if isinstance(chk, str) and ":" in chk:
        chk = tuple(chk.split(":", 1))
    return signals.context_line(c.get("prev"), c.get("conf"), c.get("tools") or (),
                                c.get("exts") or (), tuple(chk) if chk else None,
                                bool(c.get("compaction")))


def digests(item: dict, variant: str, clf: dict, decider_chars: int | None) -> tuple:
    """(the decider's state, Qwen's state) for one prompt under one digest variant."""
    chars = int(clf["digest_chars"])
    if variant == "today":
        d = signals.digest(item["text"], chars)
        return d, d
    line = line_of(item) if variant == "line" else None
    room = min(chars, decider_chars) if decider_chars else chars
    return signals.digest(item["text"], room, line), signals.digest(item["text"], chars, line)


def evaluate(items: list[dict], variant: str, router: dict, decider=None, qwen=None) -> list[dict]:
    """Every prompt through the stages and the router's stickiness; the per-prompt records."""
    clf = router["classifier"]
    stages = {s["kind"]: s for s in clf["stages"]}
    options = {c["id"]: c["description"] for c in router["classes"]}
    listing = "\n".join(f"- {k}: {v}" for k, v in options.items())
    prompt = clf["qwen_prompt"].replace("{classes}", listing)
    switch = float(router["stickiness"]["switch"])
    dchars = decider.state_chars() if decider is not None and hasattr(decider, "state_chars") \
        else None
    rows = []
    for it in items:
        d_state, q_state = digests(it, variant, clf, dchars)
        rec = {"text": it["text"][:200], "chars": len(it["text"]), "label": it["label"],
               "prev": (it.get("context") or {}).get("prev"), "decider": None, "qwen": None}
        if variant == "line":
            rec["line"] = line_of(it)
        if decider is not None:
            ans = decider.choice(d_state, clf["question"], options)
            rec["decider"] = dict(ans.as_dict(), accepted=ans.error is None and ans.label in options
                                  and ans.confidence >= stages["decider"]["accept"])
        if qwen is not None:
            ans = qwen.choice(q_state, prompt, options, max_chars=int(clf["digest_chars"]))
            rec["qwen"] = dict(ans.as_dict(), accepted=ans.error is None and ans.label in options
                               and ans.confidence >= stages["qwen"]["accept"])
        # the cascade in the router's order: the first stage that is sure decides
        casc, conf, ms = None, None, 0.0
        for st in clf["stages"]:
            r = rec.get(st["kind"])
            if not st.get("enabled", True) or r is None:
                continue
            ms += r["ms"]
            if r["accepted"]:
                casc, conf = r["label"], r["confidence"]
                break
        rec["cascade"] = {"label": casc, "accepted": casc is not None, "ms": round(ms, 2)}
        # the session's class after the turn, as Router.plan keeps it
        prev = rec["prev"] if rec["prev"] in options else None
        if casc is None:
            got = prev
        elif prev and casc != prev and conf < switch:
            got = prev
        else:
            got = casc
        rec["router"] = {"label": got, "accepted": got is not None, "ms": round(ms, 2)}
        rows.append(rec)
    return rows


def report(sets: dict[str, list[dict]], variants: list[str], router: dict, decider=None,
           qwen=None, progress=None) -> dict:
    """Every set under every variant: {variant: {set: {"n", "stages"}}} and the rows."""
    labels = [c["id"] for c in router["classes"]]
    out, rows_all = {}, {}
    for v in variants:
        out[v] = {}
        for name, items in sets.items():
            t0 = time.time()
            rows = evaluate(items, v, router, decider, qwen)
            figs = {}
            accept = {st["kind"]: float(st["accept"]) for st in router["classifier"]["stages"]}
            for key in ("decider", "qwen", "cascade", "router"):
                if any(r[key] is not None for r in rows) and (key in ("decider", "qwen") or
                                                              decider or qwen):
                    figs[key] = summarise(rows, key, labels)
                    if key in accept:
                        figs[key]["counting"] = counting(rows, key, accept[key])
            out[v][name] = {"n": len(rows), "stages": figs}
            rows_all[f"{v}:{name}"] = rows
            if progress:
                progress(f"  {v:6s} {name:24s} {len(rows):4d} prompts in {time.time() - t0:.0f} s")
    return {"variants": out, "rows": rows_all}


def local_deciders(names: str, directory: str, device: str, min_free_gib: float):
    """The engine's decision runtime in this process (engine/decision), registered as the
    in-process backend `self`, every model loaded before the first question. Refuses to load
    below `min_free_gib` of MemAvailable."""
    import threading
    from engine.decision import catalogue, routers
    from engine.decision.device import Device
    from engine.decision.registry import Registry
    from server.conductor import classify
    from server import memsplit
    free = memsplit.mem_available()
    free_gib = free / 2**30 if free else None
    if free_gib is not None and free_gib < min_free_gib:
        raise SystemExit(f"MemAvailable {free_gib:.1f} GiB is under {min_free_gib} GiB: not loading")
    models = catalogue([n.strip() for n in names.split(",") if n.strip()],
                       os.path.expanduser(directory))
    reg = Registry(models, gate=threading.Lock(), planner=None, device=Device(device),
                   routers=routers(models), log=lambda s: print(s, file=sys.stderr, flush=True))
    for m in models:
        reg.load(m.name, wait=True)
        if reg.state[m.name] != "loaded":
            raise SystemExit(f"{m.name}: {reg.state[m.name]} ({reg.error.get(m.name)})")
    classify.register_backend("self", lambda model, timeout_ms: classify.InProcess(
        reg, model, timeout_ms))
    return reg


def rescore(path: str, thresholds: list[float] | None = None) -> list[dict]:
    """A report's model stages re-counted: one line per variant, set, stage and threshold (the
    report's own threshold when `thresholds` is empty). Reports written before the decider was
    an interface name the stage `arbiter`; it is read as the decider."""
    rep = json.load(open(os.path.expanduser(path)))
    own = dict((rep.get("config") or {}).get("thresholds") or {})
    if "arbiter" in own:
        own.setdefault("decider", own["arbiter"])
    rows = rep["rows"]
    if isinstance(rows, list):                       # one set, one variant (before SRV-153)
        rows = {f"{rep.get('digest') or 'today'}:{rep['set']['name']}": rows}
    out = []
    for vs, recs in rows.items():
        variant, name = vs.split(":", 1)
        recs = [dict(r, decider=r.get("decider") or r.get("arbiter")) for r in recs]
        for key in ("decider", "qwen"):
            if not any(r.get(key) for r in recs):
                continue
            for th in thresholds or [own.get(key)]:
                if th is None:
                    continue
                out.append(dict(counting(recs, key, float(th)), variant=variant, set=name,
                                stage=key))
    return out


def print_counting(lines: list[dict]) -> None:
    print(f"{'variant':7} {'set':34} {'stage':8} {'at':>5} {'all right':>10} "
          f"{'confident right':>16} {'coverage':>9} {'conf. precision':>16}")
    for c in lines:
        prec = f"{c['confident_precision']:.3f}" if c["confident_precision"] is not None else "—"
        print(f"{c['variant']:7} {c['set'][-34:]:34} {c['stage']:8} {c['threshold']:5.2f} "
              f"{c['all_right']:10.3f} {c['confident_right']:10.3f} ({c['confident_right_n']:3d})"
              f" {c['coverage']:9.3f} {prec:>16}")
    for c in lines:
        if c.get("errors"):
            print(f"  NOT A COUNT: {c['errors']} of {c['n']} answers of {c['stage']} on "
                  f"{c['set'][-34:]} ({c['variant']}) are errors (a timeout counts as no answer)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", action="append", default=None,
                    help="a labelled set (JSON lines: text, label, context?); repeatable")
    ap.add_argument("--digest", default="router",
                    help=f"comma-separated variants of {', '.join(VARIANTS)}")
    ap.add_argument("--decider", default=None,
                    help="the decider: a Jev HTTP base URL, engine:<name>, or self")
    ap.add_argument("--decider-model", default="auto")
    ap.add_argument("--decider-timeout-ms", type=int, default=None,
                    help="default: the router's decider stage timeout")
    ap.add_argument("--local-deciders", default=None,
                    help="load these decision models in this process (registered as `self`)")
    ap.add_argument("--decider-dir", default="~/.qwen38-spark-engine/deciders/laya")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-free-gib", type=float, default=30.0)
    ap.add_argument("--qwen", default=None, help="engine base URL (the Qwen stage, raw model)")
    ap.add_argument("--qwen-model", default="qwen38-spark-engine")
    ap.add_argument("--config", default=None, help="the conductor directory whose router to use")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--note", default=None, help="a sentence kept in the report")
    ap.add_argument("--rescore", default=None, metavar="REPORT",
                    help="re-count a report's rows (no model asked) and print the four numbers")
    ap.add_argument("--at", default=None,
                    help="with --rescore: comma-separated thresholds (default: the report's)")
    a = ap.parse_args(argv)
    if a.rescore:
        print_counting(rescore(a.rescore, [float(x) for x in a.at.split(",")] if a.at else None))
        return 0

    router = Store(a.config).peek("router")[1] if a.config else schema.seed("router")
    clf = router["classifier"]
    stages = {s["kind"]: s for s in clf["stages"]}
    paths = a.set or [os.path.join(ROOT, "tests", "fixtures", "conductor", "task_prompts.jsonl")]
    sets = {}
    for p in paths:
        items = [json.loads(line) for line in open(p) if line.strip()]
        sets[os.path.relpath(p, ROOT) if os.path.abspath(p).startswith(ROOT) else p] = \
            items[:a.limit] if a.limit else items
    variants = []
    for v in a.digest.split(","):
        v = v.strip()
        if v not in VARIANTS:
            raise SystemExit(f"--digest: {v!r} is not one of {', '.join(VARIANTS)}")
        variants.append(("line" if clf.get("context_line") else "sized") if v == "router" else v)
    variants = list(dict.fromkeys(variants))
    if a.local_deciders:
        local_deciders(a.local_deciders, a.decider_dir, a.device, a.min_free_gib)
    timeout = a.decider_timeout_ms or stages.get("decider", {}).get("timeout_ms", 400)
    arb = decider_for(a.decider, a.decider_model, timeout) if a.decider else None
    if a.decider and arb is None:
        raise SystemExit(f"--decider {a.decider}: nothing answers to it")
    qwen = Qwen(a.qwen, a.qwen_model, stages.get("qwen", {}).get("timeout_ms", 8000)) \
        if a.qwen else None
    got = report(sets, variants, router, arb, qwen,
                 progress=lambda s: print(s, file=sys.stderr, flush=True))
    first_set, first_v = next(iter(sets)), variants[0]
    out = {"kind": "classifier",
           "created_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "set": {"name": first_set, "n": len(sets[first_set]),
                   "classes": [c["id"] for c in router["classes"]]},
           "digest": first_v,
           "config": {"router_from": a.config or "built-in", "thresholds": {
               k: v["accept"] for k, v in stages.items()},
               "switch": router["stickiness"]["switch"], "digest_chars": clf["digest_chars"],
               "decider_chars": arb.state_chars() if arb is not None else None},
           "models": {"decider": a.decider_model if arb else None,
                      "decider_device": a.device if a.local_deciders else None,
                      "qwen": a.qwen_model if qwen else None},
           "stages": got["variants"][first_v][first_set]["stages"],
           "variants": got["variants"], "rows": got["rows"]}
    if a.note:
        out["note"] = a.note
    print(f"{'variant':7} {'set':34} {'stage':8} {'accuracy':>9} {'answered':>9} {'acc|ans':>8} "
          f"{'top1':>6} {'p50':>7} {'p95':>7}")
    for v, per in got["variants"].items():
        for name, res in per.items():
            for k, s in res["stages"].items():
                aa = f"{s['accuracy_answered']:.3f}" if s["accuracy_answered"] is not None else "—"
                print(f"{v:7} {name[-34:]:34} {k:8} {s['accuracy']:9.3f} {s['answered']:5}/"
                      f"{s['n']:<3} {aa:>8} {s['top1']:6.3f} {s['p50_ms'] or 0:7.1f} "
                      f"{s['p95_ms'] or 0:7.1f}")
    print()
    print_counting([dict(s["counting"], variant=v, set=name, stage=k)
                    for v, per in got["variants"].items() for name, res in per.items()
                    for k, s in res["stages"].items() if "counting" in s])
    path = a.out or os.path.join(os.path.expanduser(a.config or "~/.qwen38-spark-engine/conductor"),
                                 "evals", f"classifier-{time.strftime('%Y%m%dT%H%M%S')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
