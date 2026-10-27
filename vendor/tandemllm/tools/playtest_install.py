#!/usr/bin/env python3
"""Install X1's texts and switches (2026-10-03): the laya-playtest package and the inline lines.

    python tools/playtest_install.py [--tree ~/x1-packages] [--base http://127.0.0.1:8000]
        [--packages] [--attach] [--inline coding[,overnight]] [--measure TOKENIZER_DIR] [--apply]

Without --apply every document is only checked (POST /v1/conductor/{kind}/check: its schema and
cross-document rules, the changes); with --apply each is PUT as a new version with a note and its
base version. Run again, it finds nothing to change and stores nothing.

  --packages  the packages document: `laya-playtest` a new inline general package (its whole
              SKILL.md after the frontmatter; `library: true`, so what goes into the prompt is
              the text before `## Reference` and the reference is read on request,
              GET /v1/library/laya-playtest); `laya-testing` and `test-web` get their new texts
              (one line pointing at it; their other fields kept)
  --attach    the router document: every class that lists laya-testing (code_edit, debug,
              design_ui, three_d, drawing on the served router) lists laya-playtest right after
              it (at most 12 a class)
  --inline    those profiles' `ui.states_channel: inline` (the state lines as titles of the
              reasoning parts, in time order in opencode); their other ui fields kept
  --measure   the texts on the engine's tokenizer (the checkpoint directory's tokenizer.json,
              the `tokenizers` package): laya-playtest's text before `## Reference` must be at
              most 300 tokens, or nothing is stored

The texts come from --tree/<name>/SKILL.md (on the Mac: ~/tandem-packages). The admin token comes
from the environment variable --token-env (default QSE_ADMIN_TOKEN; on the box
`. ~/.qwen38-spark-engine/secrets.env`) and is never printed. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

NEW = "laya-playtest"
UPDATED = ("laya-testing", "test-web")
AFTER = "laya-testing"
CORE_CAP = 300
MAX_TOKENS = 400                  # the core's cap in the prompt (the server cuts at x 3.5 chars)
MAX_PACKAGES = 12
FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
REFERENCE = re.compile(r"(?m)^## Reference\b")


def read_skill(path: str) -> tuple[dict, str]:
    """(frontmatter, the text after it), stripped."""
    raw = open(path, encoding="utf-8").read()
    meta, m = {}, FRONT.match(raw)
    if m:
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            if sep:
                meta[k.strip()] = v.strip()
        raw = raw[m.end():]
    return meta, raw.strip()


def core(body: str) -> str:
    r = REFERENCE.search(body)
    return (body[:r.start()] if r else body).strip()


def api(base, token, method, path, body=None):
    req = urllib.request.Request(base.rstrip("/") + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def packages_doc(doc: dict, texts: dict) -> dict:
    doc = json.loads(json.dumps(doc))
    pk = doc.setdefault("packages", {})
    meta, body = texts[NEW]
    old = pk.get(NEW) or {}
    pk[NEW] = {"title": "Laya plays your build", "kind": "package", "source": "inline",
               "description": meta.get("description", "")[:600], "body": body,
               "max_tokens": MAX_TOKENS, "enabled": old.get("enabled", True), "general": True,
               "library": True}
    for n in UPDATED:
        if n not in pk:
            sys.exit(f"[playtest-install] the packages document has no {n!r} (L1's install first)")
        meta, body = texts[n]
        pk[n] = dict(pk[n], description=meta.get("description", pk[n].get("description", ""))[:600],
                     body=body)
    return doc


def router_doc(doc: dict) -> tuple[dict, list[str]]:
    doc = json.loads(json.dumps(doc))
    touched = []
    for c in doc.get("classes") or []:
        pk = list((c.get("policy") or {}).get("packages") or [])
        if AFTER not in pk or NEW in pk:
            continue
        pk.insert(pk.index(AFTER) + 1, NEW)
        if len(pk) > MAX_PACKAGES:
            sys.exit(f"[playtest-install] {c['id']} would list {len(pk)} packages "
                     f"(at most {MAX_PACKAGES})")
        c["policy"]["packages"] = pk
        touched.append(c["id"])
    return doc, touched


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=os.path.expanduser("~/x1-packages"))
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--token-env", default="QSE_ADMIN_TOKEN")
    ap.add_argument("--packages", action="store_true")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--inline", default="", help="profiles, comma-separated")
    ap.add_argument("--measure", default="")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    say = lambda m: print(f"[playtest-install] {m}", flush=True)       # noqa: E731
    texts = {n: read_skill(os.path.join(a.tree, n, "SKILL.md")) for n in (NEW, *UPDATED)}
    if texts[NEW][0].get("name") != NEW:
        sys.exit(f"[playtest-install] {a.tree}/{NEW}/SKILL.md: frontmatter name is not {NEW}")
    if len(core(texts[NEW][1])) > MAX_TOKENS * 3.5:
        sys.exit(f"[playtest-install] {NEW}'s text before ## Reference would be cut at "
                 f"max_tokens {MAX_TOKENS}")
    if a.measure:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(os.path.expanduser(a.measure), "tokenizer.json"))
        count = lambda t: len(tk.encode(t, add_special_tokens=False).ids)   # noqa: E731
        for n, (_, body) in texts.items():
            k, t = count(core(body)), count(body)
            say(f"{n:14s} core {k:4d} tokens, all {t:4d}"
                + (f"  (core over {CORE_CAP})" if n == NEW and k > CORE_CAP else ""))
        if count(core(texts[NEW][1])) > CORE_CAP:
            sys.exit(f"[playtest-install] {NEW}'s core is over {CORE_CAP} tokens")
    if not (a.packages or a.attach or a.inline):
        return 0
    token = os.environ.get(a.token_env, "")
    if not token:
        sys.exit(f"[playtest-install] no admin token in ${a.token_env}")
    plans = []
    if a.packages:
        st, got = api(a.base, token, "GET", "/v1/conductor/packages")
        if st != 200:
            sys.exit(f"[playtest-install] GET packages: {st}")
        plans.append(("/v1/conductor/packages", got["doc"], packages_doc(got["doc"], texts),
                      got["version"], f"2026-10-03 X1: {NEW} (general, library: the core in the "
                      f"prompt, the reference on request); a line pointing at it in "
                      f"{' and '.join(UPDATED)}"))
    if a.attach:
        st, got = api(a.base, token, "GET", "/v1/conductor/router")
        if st != 200:
            sys.exit(f"[playtest-install] GET router: {st}")
        doc, touched = router_doc(got["doc"])
        say(f"router: {NEW} after {AFTER} on {', '.join(touched) or 'no class (already there)'}")
        plans.append(("/v1/conductor/router", got["doc"], doc, got["version"],
                      f"2026-10-03 X1: {NEW} after {AFTER} on {', '.join(touched)}"))
    for name in [p for p in a.inline.split(",") if p.strip()]:
        st, got = api(a.base, token, "GET", f"/v1/conductor/profiles/{name}")
        if st != 200:
            sys.exit(f"[playtest-install] GET profiles/{name}: {st}")
        ui = dict(got["doc"].get("ui") or {}, states_channel="inline")
        say(f"profiles/{name}: ui.states_channel {(got['doc'].get('ui') or {}).get('states_channel', 'reasoning')}"
            f" -> inline (ui.states {ui.get('states', 'off')})")
        plans.append((f"/v1/conductor/profiles/{name}", got["doc"], dict(got["doc"], ui=ui),
                      got["version"], "2026-10-03 X1: ui.states_channel inline (each state line "
                      "the title of a reasoning part of its own, in time order in opencode)"))
    bad = 0
    adding = a.packages
    for path, before, doc, v, note in plans:
        if doc == before:
            say(f"{path} v{v}: nothing to change")
            continue
        st, chk = api(a.base, token, "POST", path + "/check", {"doc": doc})
        ok = st == 200 and chk.get("valid")
        if not ok and not a.apply and adding and st == 200 and all(
                e.endswith(f"no package '{NEW}'") for e in chk.get("errors") or ()):
            ok = True
            say(f"{path}: valid once the packages version above is stored")
        say(f"{path} v{v}: {'valid' if ok else 'REFUSED'}; {len(chk.get('changes') or [])} "
            f"changes" + ("" if ok else f": {chk.get('errors') or chk}"))
        if not ok:
            bad += 1
            continue
        if a.apply:
            st, put = api(a.base, token, "PUT", path, {"doc": doc, "base_version": v, "note": note})
            say(f"{path}: PUT {st} -> v{put.get('version')}")
            bad += st != 200
    if not a.apply:
        say("checked only; --apply stores the versions")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
