"""The router's class question on the labelled prompts, and what each fallback rule makes of it.

    python tools/class_fallback_eval.py ask --doc ROUTER.json --set FILE.jsonl ... --out answers.json
        [--url http://127.0.0.1:8000] [--extra 'TEXT=label' ...]
    python tools/class_fallback_eval.py rules answers.json [answers2.json ...]

`ask` puts the class question to the engine (`X-Tandem: raw`, one prompt at a time, the question
exactly as `Router.cascade` asks a first message: the digest, the document's `qwen_prompt` and its
classes, the SRV-189 code guard) under the router document in `--doc` (a store version file or a
bare document: the box's documents stay as they are), and keeps every class's probability. A run
stops when ~/BOX-HOLD appears.

`rules` replays the answers through the fallback rules and prints the accuracy per rule and per
set, and the answers each rule changes against the current one:

    chat           below the accept bar the turn takes the default class (chat), the rule before
                   0.4.0
    top            the top class always
    build_top      below the bar the top class when it is a class that builds something (its
                   policy names files-care: the classes whose turns write code or files), else chat
    build_margin   build_top, and only when the top class leads the second by `--margin`

Nothing is written to the box.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _doc(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    return d.get("doc", d)


def ask(a) -> int:
    from server.conductor import signals
    from server.conductor.classify import Qwen
    from server.conductor.router import _code_guard
    doc = _doc(a.doc)
    clf = doc["classifier"]
    chars = int(clf.get("digest_chars", 2000))
    options = {c["id"]: c["description"] for c in doc["classes"]}
    prompt = clf["qwen_prompt"].replace("{classes}", "\n".join(f"- {k}: {v}" for k, v in options.items()))
    q = Qwen(a.url, a.model, 8000)
    items = []
    for path in a.set:
        items += [dict(json.loads(x), set=os.path.basename(path)) for x in open(path) if x.strip()]
    for e in a.extra or []:
        text, _, label = e.rpartition("=")
        items.append({"text": text, "label": label, "set": "extra"})
    rows = []
    hold = os.path.expanduser("~/BOX-HOLD")
    for it in items:
        if os.path.exists(hold):
            print(f"~/BOX-HOLD appeared: stopping after {len(rows)}", flush=True)
            break
        ans = _code_guard(q.choice(signals.digest(it["text"], chars, None), prompt, options,
                                   max_chars=chars), it["text"], options)
        rows.append({"set": it["set"], "text": it["text"], "label": it.get("label"),
                     "class": ans.label, "conf": round(ans.confidence, 4), "error": ans.error,
                     "guard": bool(getattr(ans, "guard", None)),
                     "probs": {k: round(v, 4) for k, v in (ans.probs or {}).items()}})
    accept = next(float(s["accept"]) for s in clf["stages"] if s["kind"] == "qwen")
    out = {"tool": "class_fallback_eval", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "doc": os.path.basename(a.doc), "accept": accept, "default": "chat",
           # PKG3: a class builds when it lists files-care (SRV-227), as Router._below_build
           "build": sorted(c["id"] for c in doc["classes"]
                           if "files-care" in ((c.get("policy") or {}).get("packages") or [])),
           "rows": rows}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    right = sum(r["class"] == r["label"] and (r["conf"] >= accept or r["guard"]) for r in rows)
    print(f"{len(rows)} prompts, {right} right as routed now (accept {accept})")
    return 0


def route(r: dict, rule: str, accept: float, build: set, margin: float, default="chat") -> str:
    """The class one rule gives an answer (a first message, no session class)."""
    if r["error"]:
        return default
    if r["conf"] >= accept or r["guard"]:
        return r["class"]
    if rule == "chat":
        return default
    top = sorted(r["probs"].items(), key=lambda kv: -kv[1])
    if rule == "top":
        return top[0][0]
    if top[0][0] not in build:
        return default
    if rule == "build_top":
        return top[0][0]
    lead = top[0][1] - (top[1][1] if len(top) > 1 else 0.0)
    return top[0][0] if lead >= margin else default


def rules(a) -> int:
    for path in a.answers:
        rep = json.load(open(path))
        rows, accept, build = rep["rows"], rep["accept"], set(rep["build"])
        names = ["chat", "top", "build_top", "build_margin"]
        print(f"== {path} ({rep['doc']}, {len(rows)} prompts, accept {accept}, build classes {sorted(build)})")
        by_set = collections.defaultdict(collections.Counter)
        changed = collections.defaultdict(list)
        for r in rows:
            base = route(r, "chat", accept, build, a.margin)
            for n in names:
                got = route(r, n, accept, build, a.margin)
                by_set[r["set"]][n] += got == r["label"]
                by_set["all" if r["set"] != "extra" else "extra"][n] += 0
                if r["set"] != "extra":
                    by_set["labelled"][n] += got == r["label"]
                if got != base:
                    changed[n].append((r["set"], r["text"][:60], r["label"], base, got, r["conf"]))
            by_set[r["set"]]["n"] += 1
            if r["set"] != "extra":
                by_set["labelled"]["n"] += 1
        below = [r for r in rows if r["set"] != "extra" and not r["error"] and r["conf"] < accept and not r["guard"]]
        print(f"   below the bar: {len(below)} of {by_set['labelled']['n']} labelled; their labels "
              f"{dict(collections.Counter(r['label'] for r in below))}")
        for s in sorted(k for k in by_set if k != "all"):
            c = by_set[s]
            print(f"   {s:24s} n={c['n']:3d}  " + "  ".join(f"{n} {c[n]}" for n in names))
        for n in names[1:]:
            fixed = sum(1 for x in changed[n] if x[4] == x[2])
            broke = sum(1 for x in changed[n] if x[3] == x[2])
            print(f"   {n}: {len(changed[n])} answers change, {fixed} become right, {broke} become wrong")
            for x in changed[n][:a.show]:
                print(f"      {x[0]:22s} label {x[2]:10s} {x[3]:>10s} -> {x[4]:10s} conf {x[5]:.3f}  {x[1]!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ask")
    s.add_argument("--doc", required=True)
    s.add_argument("--set", action="append", required=True)
    s.add_argument("--extra", action="append")
    s.add_argument("--url", default="http://127.0.0.1:8000")
    s.add_argument("--model", default="qwen38-spark-engine")
    s.add_argument("--out", required=True)
    r = sub.add_parser("rules")
    r.add_argument("answers", nargs="+")
    r.add_argument("--margin", type=float, default=0.1)
    r.add_argument("--show", type=int, default=12)
    a = ap.parse_args()
    return ask(a) if a.cmd == "ask" else rules(a)


if __name__ == "__main__":
    sys.exit(main())
