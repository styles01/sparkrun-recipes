"""The first packages (OPS-39): check the folders, measure the texts, review the eval sets.

    python tools/packages_first.py                     check everything; exit 1 on a finding
    python tools/packages_first.py --tokenizer DIR     count on the engine's tokenizer (DIR/tokenizer.json,
                                                       needs the `tokenizers` package); else 3.5 chars a token
    python tools/packages_first.py --report            the table: name, kind, tokens against the cap, tools, eval

The seven packages are folders: the system set under packages/system/<name>/ (PACKAGE.md is the
text that goes in, an inline package in seed.json through `body_from`) and the four first packages
under packages/library/<name>/ (SKILL.md: the body before `## Reference` goes in, a library skill;
the reference is read through GET /v1/library/<name>). Every folder carries manifest.json,
eval/prompts.jsonl, eval/rubric.md and, where it has them, references/ and exemplars/.

What is checked, per package:
  manifest   the fields, requires_tools within signals.CAPS, classes within the shipped router's,
             every file the manifest names exists
  seed       packages/seed.json has the entry and agrees with the manifest (kind, max_tokens, general,
             library, requires_tools, fallback, description, source)
  text       the frontmatter name is the folder, a library skill has a `## Reference`, the text that
             goes in is within max_tokens (and under the server's character cut) and, on the engine's
             tokenizer, a library skill's body within the house 350 (the estimate over-counts code)
  index      a library skill's INDEX.md row names the manifest's classes
  router     a general system package is listed by every class the manifest names; a library skill
             by the classes the manifest marks `router: true` (the package question's choices)
  eval       prompts.jsonl parses, ids unique with the package's prefix, the count matches, every
             class is a router class (or `judge`), every check is one the manifest names, a prompt
             with `prompt_terms` carries `terms`; the rubric names every check
  privacy    no home path, LAN address, box host name or client name in any file of the folder, nor
             any word of TANDEM_PRIVATE_WORDS (the operator's names and hosts; the CI sets it)
  documents  the shipped packages and router documents validate (schema.check)

Run: python tools/packages_first.py; tests/test_packages_first.py calls check().
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import libsearch, schema, signals  # noqa: E402
from server.conductor.packages import CHARS_PER_TOKEN, split_frontmatter, split_reference  # noqa: E402

PKG = os.path.join(ROOT, "packages")
SYSTEM = ("tandem-core", "render-tools", "eval-judge")
FIRST = ("creative-coding", "charts", "voxel-worlds", "explainer-video")
HOUSE_BODY = 350                                      # tools/library_install.py BODY_CAP
PREFIX = {"tandem-core": "core", "render-tools": "rt", "eval-judge": "judge", "creative-coding": "cc",
          "charts": "ch", "voxel-worlds": "vw", "explainer-video": "ev"}
# text that must not be in a public package: a home path, a LAN address, a box's host name, a
# client's name (a skill names no client the request did not come from), and the words the
# operator lists in TANDEM_PRIVATE_WORDS (comma-separated: personal names, hosts, domains),
# which stay out of the repository
_WORDS = [w.strip() for w in os.environ.get("TANDEM_PRIVATE_WORDS", "").split(",") if w.strip()]
PRIVATE = re.compile("|".join([r"/Users/\w", r"/home/\w", r"192\.168\.\d", r"10\.\d+\.\d+\.\d+",
                               r"\.local\b", r"\bdgx\b", r"claude", r"anthropic", r"opencode",
                               r"hermes"] + [re.escape(w) for w in _WORDS]), re.I)
REQUIRED = ("name", "version", "title", "kind", "set", "description", "source", "max_tokens",
            "general", "library", "requires_tools", "classes", "files", "eval", "measured")


def folder(name: str) -> str:
    return os.path.join(PKG, "system" if name in SYSTEM else "library", name)


def manifest(name: str) -> dict:
    with open(os.path.join(folder(name), "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def injected(name: str, m: dict) -> tuple[str, str]:
    """(the text that goes in, the reference or '')."""
    with open(os.path.join(folder(name), m["files"]["body"]), encoding="utf-8") as f:
        raw = f.read()
    meta, body = split_frontmatter(raw)
    if m.get("library"):
        body, ref = split_reference(body)
        return body.strip(), ref.strip()
    return body.strip(), ""


def counter(tokenizer_dir: str | None):
    if tokenizer_dir:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(os.path.expanduser(tokenizer_dir), "tokenizer.json"))
        return (lambda s: len(tk.encode(s, add_special_tokens=False).ids)), "engine"
    return (lambda s: int(len(s) / CHARS_PER_TOKEN)), "estimate"


def _jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check(tokenizer_dir: str | None = None) -> tuple[list[str], list[dict]]:
    """(findings, one row per package). No finding means the seven packages are consistent."""
    out, rows = [], []
    count, how = counter(tokenizer_dir)
    seed = schema.SHIPPED_PACKAGES["packages"]
    router = schema.SHIPPED_ROUTER
    classes = {c["id"]: c for c in router["classes"]}
    with open(os.path.join(PKG, "library", "INDEX.md"), encoding="utf-8") as f:
        index = libsearch.index_classes(f.read())
    for name in SYSTEM + FIRST:
        d = folder(name)
        try:
            m = manifest(name)
        except (OSError, ValueError) as e:
            out.append(f"{name}: manifest: {e}")
            continue
        for k in REQUIRED:
            if k not in m:
                out.append(f"{name}: manifest lacks {k}")
        if m.get("name") != name:
            out.append(f"{name}: manifest name {m.get('name')!r}")
        bad = [t for t in m.get("requires_tools", []) if t not in signals.CAPS]
        if bad:
            out.append(f"{name}: requires_tools {bad} not in {signals.CAPS}")
        for c in m.get("classes", []):
            if c not in classes:
                out.append(f"{name}: class {c} is not a router class")
        for kind in ("references", "exemplars"):
            for rel in m.get("files", {}).get(kind, []):
                if not os.path.isfile(os.path.join(d, rel)):
                    out.append(f"{name}: {kind} file missing: {rel}")
        if not os.path.isfile(os.path.join(d, m.get("files", {}).get("body", "?"))):
            out.append(f"{name}: body file missing")
            continue
        # ---- the seed entry
        p = seed.get(name)
        if p is None:
            out.append(f"{name}: not in packages/seed.json")
        else:
            want = {"kind": m["kind"], "max_tokens": m["max_tokens"], "description": m["description"],
                    "requires_tools": m["requires_tools"]}
            if m.get("general"):
                want["general"] = True
            if m.get("library"):
                want["library"] = True
            if m.get("fallback"):
                want["fallback"] = m["fallback"]
            for k, v in want.items():
                if p.get(k) != v:
                    out.append(f"{name}: seed.json {k} is {p.get(k)!r}, the manifest says {v!r}")
            if p.get("general") and not m.get("general"):
                out.append(f"{name}: seed.json says general, the manifest does not")
            if m["kind"] == "skill" and p.get("source") != f"skill:{name}":
                out.append(f"{name}: seed.json source {p.get('source')!r}")
            if m.get("fallback") and m["fallback"] not in seed:
                out.append(f"{name}: fallback {m['fallback']} is not in seed.json")
        # ---- the text
        with open(os.path.join(d, m["files"]["body"]), encoding="utf-8") as f:
            raw = f.read()
        meta, _ = split_frontmatter(raw)
        if m.get("library"):
            if meta.get("name") != name:
                out.append(f"{name}: frontmatter name {meta.get('name')!r}")
            if meta.get("description") != m["description"]:
                out.append(f"{name}: frontmatter description differs from the manifest")
            if "\n## Reference" not in raw:
                out.append(f"{name}: no `## Reference` section")
        elif meta:
            out.append(f"{name}: a system package's text carries frontmatter (it would go in as text)")
        body, ref = injected(name, m)
        k_body, k_ref = count(body), (count(ref) if ref else 0)
        cap = m["max_tokens"]
        if k_body > cap:
            out.append(f"{name}: the text that goes in is {k_body} tokens ({how}), over its cap {cap}")
        if m.get("library") and how == "engine" and k_body > HOUSE_BODY:
            out.append(f"{name}: a library skill's body is {k_body} tokens ({how}), over the house {HOUSE_BODY}")
        if len(body) > cap * CHARS_PER_TOKEN:
            out.append(f"{name}: {len(body)} characters, the server would cut it at {int(cap * CHARS_PER_TOKEN)}")
        # ---- index and router
        if m.get("library"):
            if name not in index:
                out.append(f"{name}: not in packages/library/INDEX.md")
            elif index[name] != m["classes"]:
                out.append(f"{name}: INDEX.md classes {index[name]} differ from the manifest's {m['classes']}")
        for c in m.get("classes", []):
            listed = name in classes[c]["policy"]["packages"] if c in classes else False
            # PKG3: tandem-core on chat waits for a design decision (schema.CHAT_CORE, off)
            switched = c == "chat" and name in schema.CORE_PACKAGES and not schema.CHAT_CORE
            if m.get("general") and not listed and not switched:
                out.append(f"{name}: general, but class {c} does not list it")
            if m.get("router", {}).get(c) and not listed:
                out.append(f"{name}: the manifest says class {c} lists it, the router does not")
        # ---- the eval set
        ev = m.get("eval", {})
        prompts_path = os.path.join(d, ev.get("prompts", "eval/prompts.jsonl"))
        rubric_path = os.path.join(d, ev.get("rubric", "eval/rubric.md"))
        n_prompts = 0
        try:
            prompts = _jsonl(prompts_path)
        except (OSError, ValueError) as e:
            out.append(f"{name}: eval prompts: {e}")
            prompts = []
        ids = [r.get("id") for r in prompts]
        n_prompts = len(prompts)
        if n_prompts != ev.get("count"):
            out.append(f"{name}: {n_prompts} prompts, the manifest says {ev.get('count')}")
        if len(set(ids)) != len(ids):
            out.append(f"{name}: duplicate prompt ids")
        for r in prompts:
            rid = r.get("id", "?")
            if not str(rid).startswith(PREFIX[name] + "-"):
                out.append(f"{name}: prompt id {rid} lacks the prefix {PREFIX[name]}-")
            if r.get("class") not in classes and r.get("class") != "judge":
                out.append(f"{name}: prompt {rid} class {r.get('class')!r}")
            if len(r.get("prompt", "")) < 20:
                out.append(f"{name}: prompt {rid} is too short")
            unknown = [c for c in r.get("checks", []) if c not in ev.get("checks", [])]
            if unknown or not r.get("checks"):
                out.append(f"{name}: prompt {rid} checks {unknown or 'missing'}")
            if "prompt_terms" in r.get("checks", []) and not r.get("terms"):
                out.append(f"{name}: prompt {rid} has prompt_terms and no terms")
            if PRIVATE.search(json.dumps(r)):
                out.append(f"{name}: prompt {rid} carries private text")
        for s in ev.get("human_sample", []):
            if s not in ids:
                out.append(f"{name}: human sample {s} is not a prompt")
        try:
            with open(rubric_path, encoding="utf-8") as f:
                rubric = f.read()
            if not rubric.startswith(f"# Rubric: {name}"):
                out.append(f"{name}: the rubric's title is not `# Rubric: {name}`")
            for c in ev.get("checks", []):
                if f"`{c}`" not in rubric:
                    out.append(f"{name}: the rubric does not name the check {c}")
        except OSError as e:
            out.append(f"{name}: rubric: {e}")
        # ---- privacy, every file
        for dp, _, fs in os.walk(d):
            for fn in fs:
                with open(os.path.join(dp, fn), encoding="utf-8", errors="replace") as f:
                    hit = PRIVATE.search(f.read())
                if hit:
                    out.append(f"{name}: {os.path.relpath(os.path.join(dp, fn), d)} carries private text: {hit.group(0)!r}")
        rows.append({"name": name, "kind": m["kind"], "set": m.get("set"), "tokens": k_body, "cap": cap,
                     "reference": k_ref, "tokenizer": how, "requires_tools": m["requires_tools"],
                     "fallback": m.get("fallback"), "general": bool(m.get("general")),
                     "library": bool(m.get("library")), "classes": m.get("classes", []),
                     "eval": n_prompts, "measured": m.get("measured") or "not measured"})
    # ---- the shipped documents
    for kind, doc, extra in (("packages", schema.SHIPPED_PACKAGES, {}),
                             ("router", router, {"packages": schema.SHIPPED_PACKAGES})):
        errs = schema.check(kind, doc, **extra)
        if errs:
            out.append(f"shipped {kind}: {errs[:3]}")
    return out, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", help="a directory with the engine's tokenizer.json")
    ap.add_argument("--report", action="store_true", help="print the table")
    ap.add_argument("--json", action="store_true", help="the rows as JSON")
    a = ap.parse_args(argv)
    findings, rows = check(a.tokenizer)
    if a.json:
        print(json.dumps({"findings": findings, "rows": rows}, indent=1))
    elif a.report or not findings:
        print(f"{'name':18s} {'kind':8s} {'tokens':>6s} {'cap':>5s} {'ref':>5s}  {'tools':16s} {'eval':>4s}  classes")
        for r in rows:
            tools = ",".join(r["requires_tools"]) or "-"
            print(f"{r['name']:18s} {r['kind']:8s} {r['tokens']:6d} {r['cap']:5d} {r['reference']:5d}  "
                  f"{tools:16s} {r['eval']:4d}  {','.join(r['classes'])}  [{r['tokenizer']}]")
    for f in findings:
        print("FINDING", f)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
