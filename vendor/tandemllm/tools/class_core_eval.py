"""FX6: the package question before and after, on the live router's classes, one Qwen class answer per
prompt.

    python tools/class_core_eval.py --url http://127.0.0.1:8000 \\
        --set tests/fixtures/conductor/task_prompts.jsonl --set ... --out report.json

Per prompt: the class question once (the live router document's classes, `qwen_prompt`, the
accept threshold and the SRV-189 code guard, exactly as `Router.cascade` asks it; FX6 does not
change it). For a class whose packages leave more than one candidate, the package question twice:

  before  the class question's prompt with the packages as the classes, the class's whole
          candidate list (the general packages aside), the router before FX6
  after   the router's way after FX6: the candidates without the class's `core`, which goes in
          whole; the prompt is the class question's unless the overlay sets
          `classifier.package_prompt` (with `{class}`, the routed class)

Then the reading line the router would add on a session's first message (`Router._reading`).
`--overlay FILE` (JSON: {"classes": {id: {"core": [...], "reading": "..."}}, "classifier": {...}})
is laid over the live document for `after`; without it the FX6 router v15 overlay below.
Nothing is written to the box; every question goes to `--url` with `X-Tandem: raw`, one at a
time. The run stops when ~/BOX-HOLD appears.

Per class: the class accuracy (the label), how often the class's defining skill reaches the turn
before and after (general, core or chosen), the package answers before and after, and how often
the reading line fires.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import schema, signals  # noqa: E402
from server.conductor.classify import Qwen  # noqa: E402
from server.conductor.router import NONE_OPTION, Router, _about, _code_guard, _qwen_prompt  # noqa: E402

# the router document v15 FX6 puts on the box: the defining skill as core where a class has one
# that is not already general, the reading phrases of the three deliverable kinds
OVERLAY = {"classes": {
    "three_d": {"core": ["threejs-scene"], "reading": "a 3D three.js scene"},
    "design_ui": {"reading": "a flat web page"},
    "drawing": {"reading": "a 2D drawing"},
}}
# the skill that defines each class (the audit's question: does the turn get it?)
DEFINING = {"code_edit": "coding", "debug": "coding", "design_ui": "web-design",
            "three_d": "threejs-scene"}


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def overlay(router: dict, ov: dict) -> dict:
    out = copy.deepcopy(router)
    for c in out["classes"]:
        o = ov.get("classes", {}).get(c["id"]) or {}
        if "core" in o:
            c["policy"]["core"] = list(o["core"])
        if "reading" in o:
            c["reading"] = o["reading"]
    out["classifier"].update(copy.deepcopy(ov.get("classifier") or {}))
    return out


class _Plan:
    decisions = {"classify": "on"}
    class_new = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen38-spark-engine")
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--overlay")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    hold = os.path.expanduser("~/BOX-HOLD")
    live = get(a.url + "/v1/conductor/router")
    pk = get(a.url + "/v1/conductor/packages")["doc"]["packages"]
    before_doc = live["doc"]
    after_doc = overlay(before_doc, json.load(open(a.overlay)) if a.overlay else OVERLAY)
    clf = before_doc["classifier"]
    chars = int(clf.get("digest_chars", 2000))
    accept = next(float(s["accept"]) for s in clf["stages"] if s["kind"] == "qwen")
    p_accept = float(clf["package_accept"])
    classes = {c["id"]: c for c in before_doc["classes"]}
    after_classes = {c["id"]: c for c in after_doc["classes"]}
    options = {c: classes[c]["description"] for c in classes}
    q = Qwen(a.url, a.model, 8000)
    rows = []
    for path in a.set:
        items = [json.loads(x) for x in open(path) if x.strip()]
        if a.limit:
            items = items[:a.limit]
        for it in items:
            if os.path.exists(hold):
                print(f"~/BOX-HOLD appeared: stopping after {len(rows)} prompts", flush=True)
                return _write(a, rows, live, after_doc, stopped=True)
            text = it["text"]
            ans = _code_guard(q.choice(signals.digest(text, chars, None),
                                       clf["qwen_prompt"].replace("{classes}", "\n".join(
                                           f"- {k}: {v}" for k, v in options.items())),
                                       options, max_chars=chars), text, options)
            row = {"set": os.path.basename(path), "text": text, "label": it.get("label"),
                   "labels": it.get("labels"), "kind": it.get("kind"), "error": ans.error,
                   "class": ans.label, "conf": round(ans.confidence, 4),
                   "probs": {k: round(v, 4) for k, v in (ans.probs or {}).items()},
                   "ms": round(ans.ms, 1)}
            routed = ans.label if ans.error is None and (ans.confidence >= accept or
                                                         getattr(ans, "guard", None)) else None
            row["routed"] = routed
            if routed is not None:
                pol = classes[routed]["policy"]
                general = [p for p in pol.get("packages", []) if (pk.get(p) or {}).get("general")]
                core = [p for p in after_classes[routed]["policy"].get("core", [])
                        if p in pol.get("packages", [])]
                for arm, cands, prompt_of in (
                        ("before", [p for p in pol.get("packages", []) if p not in general],
                         lambda listing: clf["qwen_prompt"].replace("{classes}", listing)),
                        ("after", [p for p in pol.get("packages", [])
                                   if p not in general and p not in core],
                         lambda listing: _qwen_prompt(after_doc, clf["package_question"],
                                                      "packages", _about(routed, classes),
                                                      listing))):
                    got = {"candidates": cands, "chosen": None, "how": None}
                    if len(cands) == 1:
                        got.update(chosen=cands[0], how="only_candidate")
                    elif cands:
                        opts = {p: pk[p]["description"] for p in cands if p in pk}
                        opts[NONE_OPTION] = "None of these fits the request."
                        listing = "\n".join(f"- {k}: {v}" for k, v in opts.items())
                        pa = q.choice(signals.digest(text, chars, None), prompt_of(listing), opts,
                                      max_chars=chars)
                        got["probs"] = {k: round(v, 4) for k, v in (pa.probs or {}).items()}
                        got["answer"], got["p"] = pa.label, round(pa.confidence, 4)
                        if pa.error:
                            got["how"] = pa.error
                        elif pa.label != NONE_OPTION and pa.confidence >= p_accept:
                            got.update(chosen=pa.label, how="qwen")
                        else:
                            got["how"] = "none"
                    attached = general + (core if arm == "after" else []) + (
                        [got["chosen"]] if got["chosen"] else [])
                    got["attached"] = attached
                    row[arm] = got
                fake = {"confidence": ans.confidence, "source": "qwen", "extra": {},
                        "shadow": None, "stages": [{"label": routed, "probs": ans.probs}]}
                part = Router._reading(None, _Plan(), after_doc, routed, after_classes, fake)
                row["reading"] = fake["extra"].get("reading") if part else None
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("class", "conf", "label")}
                             | {"b": (row.get("before") or {}).get("chosen"),
                                "a": (row.get("after") or {}).get("chosen"),
                                "r": bool(row.get("reading"))}), flush=True)
    return _write(a, rows, live, after_doc)


def _write(a, rows, live, after_doc, stopped=False) -> int:
    per = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        lab = r["label"]
        c = per[lab]
        c["n"] += 1
        c["right"] += r["routed"] == lab
        d = DEFINING.get(r["routed"])
        if d:
            c["defining_n"] += 1
            c["defining_before"] += d in (r.get("before") or {}).get("attached", [])
            c["defining_after"] += d in (r.get("after") or {}).get("attached", [])
        b, af = (r.get("before") or {}).get("chosen"), (r.get("after") or {}).get("chosen")
        if r.get("before") is not None:
            c["chooser_same"] += b == af
            c["chooser_asked"] += 1
        c["reading"] += bool(r.get("reading"))
    summary = {k: dict(v) for k, v in sorted(per.items(), key=lambda kv: str(kv[0]))}
    out = {"tool": "class_core_eval", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "router_version": live.get("version"), "stopped": stopped, "n": len(rows),
           "overlay_after": after_doc["classes"], "summary": summary, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
