#!/usr/bin/env python3
"""Install the package library (R1, 2026-10-03) on the box: the files, the packages, the switch.

    python tools/library_install.py [--src ~/r1lib/library] [--card ~/r1lib/tandem-library/SKILL.md]
        [--dest ~/.qwen38-spark-engine/conductor/skills] [--base http://127.0.0.1:8000]
        [--measure TOKENIZER_DIR] [--profile coding --library shadow] [--apply]

Without --apply nothing is written: the files that would be copied (new, changed, the same), the
packages document the server would store (POST /v1/conductor/packages/check: its schema and
cross-document rules, the changes), and the profile switch (its check too). With --apply:

  1  every <src>/<name>/SKILL.md and INDEX.md to <dest>/library/ (a file written beside and
     renamed over, mode 0644), the card to <dest>/tandem-library/SKILL.md;
  2  the packages document: <dest>/library in `skill_dirs` (after the others, so `skill:<name>`
     finds the library's file), every skill a package (kind skill, source `skill:<name>`, its
     frontmatter's description, `library: true`, enabled, max_tokens MAX_TOKENS), the card a
     package `tandem-library` (kind package, source `skill:tandem-library`); a package of that
     name already there keeps its `enabled`. One new version with a note and its base version;
  3  with --profile and --library: that profile's `library` switch, one new version.

--measure: every skill's text before `## Reference` and its reference on the engine's tokenizer
(the checkpoint directory's tokenizer.json; needs the `tokenizers` package), and the card's text.
Nothing is trimmed: a text over 350 tokens (the card over 200) is listed. Every text must fit its
`max_tokens` uncut (the server cuts at max_tokens x 3.5 characters), or nothing is stored.

The admin token comes from the environment variable --token-env (default QSE_ADMIN_TOKEN; on the
box `. ~/.qwen38-spark-engine/secrets.env`) and is never printed. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

MAX_TOKENS = 500
CHARS_PER_TOKEN = 3.5            # server/conductor/packages.py: the cut is at max_tokens x 3.5
BODY_CAP, CARD_CAP = 350, 200
CARD = "tandem-library"
FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
REFERENCE = re.compile(r"(?m)^## Reference\b")


def read_skill(path: str) -> tuple[dict, str, str]:
    """(frontmatter, the text before `## Reference`, the reference), stripped."""
    raw = open(path, encoding="utf-8").read()
    meta, m = {}, FRONT.match(raw)
    if m:
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            if sep:
                meta[k.strip()] = v.strip().strip('"').strip("'")
        raw = raw[m.end():]
    r = REFERENCE.search(raw)
    body, ref = (raw[:r.start()], raw[r.start():]) if r else (raw, "")
    return meta, body.strip(), ref.strip()


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


def copy_plan(pairs) -> list[tuple[str, str, str]]:
    """(src, dst, new|changed|same) for every file."""
    out = []
    for src, dst in pairs:
        data = open(src, "rb").read()
        try:
            old = open(dst, "rb").read()
        except OSError:
            old = None
        out.append((src, dst, "new" if old is None else "same" if old == data else "changed"))
    return out


def write_atomic(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".new"
    with open(src, "rb") as f, open(tmp, "wb") as g:
        g.write(f.read())
    os.chmod(tmp, 0o644)
    os.replace(tmp, dst)


def packages_doc(doc: dict, skills: dict, card: tuple | None, dest: str, lib_dir: str) -> dict:
    """The packages document with the library registered (an existing package keeps `enabled`)."""
    doc = json.loads(json.dumps(doc))
    dirs = doc.setdefault("skill_dirs", [])
    if not any(os.path.expanduser(d) == os.path.expanduser(lib_dir) for d in dirs):
        dirs.append(lib_dir)
    if not any(os.path.expanduser(d) == os.path.expanduser(dest) for d in dirs):
        dirs.append(dest)
    pk = doc.setdefault("packages", {})
    for name, (meta, _, _) in sorted(skills.items()):
        old = pk.get(name) or {}
        pk[name] = {"title": name.replace("-", " "), "kind": "skill", "source": f"skill:{name}",
                    "description": meta["description"][:600], "body": "",
                    "max_tokens": MAX_TOKENS, "enabled": old.get("enabled", True),
                    "library": True}
    if card is not None:
        old = pk.get(CARD) or {}
        pk[CARD] = {"title": "The skill library", "kind": "package",
                    "source": f"skill:{CARD}", "description": card[0]["description"][:600],
                    "body": "", "max_tokens": 300, "enabled": old.get("enabled", True)}
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/r1lib/library"))
    ap.add_argument("--card", default=os.path.expanduser("~/r1lib/tandem-library/SKILL.md"))
    ap.add_argument("--dest", default="~/.qwen38-spark-engine/conductor/skills")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--token-env", default="QSE_ADMIN_TOKEN")
    ap.add_argument("--measure", default="")
    ap.add_argument("--profile", default="")
    ap.add_argument("--library", choices=("off", "shadow", "on"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    say = lambda m: print(f"[library-install] {m}", flush=True)       # noqa: E731
    dest = a.dest.rstrip("/")
    dest_abs = os.path.expanduser(dest)
    lib_dir = dest + "/library"
    names = sorted(n for n in os.listdir(a.src) if os.path.isfile(os.path.join(a.src, n,
                                                                               "SKILL.md")))
    skills = {}
    for n in names:
        meta, body, ref = read_skill(os.path.join(a.src, n, "SKILL.md"))
        if meta.get("name") != n or len(meta.get("description", "")) < 8:
            sys.exit(f"[library-install] {n}: frontmatter name/description missing or wrong")
        skills[n] = (meta, body, ref)
    card = read_skill(a.card) if a.card and os.path.isfile(a.card) else None
    say(f"{len(skills)} skills in {a.src}; card {'found' if card else 'MISSING'} ({a.card})")
    cut = [n for n, (_, b, _) in skills.items() if len(b) > MAX_TOKENS * CHARS_PER_TOKEN]
    if card and len(card[1]) > 300 * CHARS_PER_TOKEN:
        cut.append(CARD)
    if cut:
        sys.exit(f"[library-install] the server would cut these texts at max_tokens: {cut}")
    if a.measure:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(os.path.expanduser(a.measure), "tokenizer.json"))
        count = lambda t: len(tk.encode(t, add_special_tokens=False).ids)   # noqa: E731
        rows = [(n, count(b), count(r)) for n, (_, b, r) in skills.items()]
        for n, b, r in rows:
            print(f"  {n:24s} body {b:4d}  reference {r:4d}{'  OVER 350' if b > BODY_CAP else ''}")
        over = [n for n, b, _ in rows if b > BODY_CAP]
        bs = [b for _, b, _ in rows]
        say(f"bodies: total {sum(bs)}, min {min(bs)}, median {sorted(bs)[len(bs) // 2]}, max "
            f"{max(bs)}; references total {sum(r for _, _, r in rows)}; over {BODY_CAP}: "
            f"{len(over)} {over}")
        if card:
            k = count(card[1])
            say(f"card {CARD}: {k} tokens{' OVER 200' if k > CARD_CAP else ''}")
    pairs = [(os.path.join(a.src, n, "SKILL.md"), os.path.join(dest_abs, "library", n,
                                                                  "SKILL.md")) for n in names]
    if os.path.isfile(os.path.join(a.src, "INDEX.md")):
        pairs.append((os.path.join(a.src, "INDEX.md"), os.path.join(dest_abs, "library",
                                                                    "INDEX.md")))
    if card:
        pairs.append((a.card, os.path.join(dest_abs, CARD, "SKILL.md")))
    plan = copy_plan(pairs)
    say("files: " + ", ".join(f"{s} {sum(1 for p in plan if p[2] == s)}"
                              for s in ("new", "changed", "same")))
    token = os.environ.get(a.token_env, "")
    if not token:
        sys.exit(f"[library-install] no admin token in ${a.token_env}")
    st, got = api(a.base, token, "GET", "/v1/conductor/packages")
    if st != 200:
        sys.exit(f"[library-install] GET packages: {st} {got.get('error')}")
    doc = packages_doc(got["doc"], skills, card, dest, lib_dir)
    st, chk = api(a.base, token, "POST", "/v1/conductor/packages/check", {"doc": doc})
    if st != 200 or not chk.get("valid"):
        sys.exit(f"[library-install] packages v{got['version']}: refused: {st} "
                 f"{chk.get('errors') or chk.get('error')}")
    say(f"packages v{got['version']}: valid; {len(chk['changes'])} changes; skill_dirs "
        f"{doc['skill_dirs']}")
    prof = None
    if a.profile and a.library:
        st, p = api(a.base, token, "GET", f"/v1/conductor/profiles/{a.profile}")
        if st != 200:
            sys.exit(f"[library-install] GET profile {a.profile}: {st}")
        prof = dict(p["doc"], library=a.library)
        say(f"profile {a.profile} v{p['version']}: library {p['doc'].get('library', 'off')} -> "
            f"{a.library} (stored after the packages)")
    if not a.apply:
        say("dry run: nothing written; --apply copies the files and stores the versions")
        return 0
    for src, dst, state in plan:
        if state != "same":
            write_atomic(src, dst)
    say(f"copied {sum(1 for p in plan if p[2] != 'same')} files under {dest_abs}")
    st, put = api(a.base, token, "PUT", "/v1/conductor/packages", {
        "doc": doc, "note": f"R1 package library: {len(skills)} skills (library: true) + "
                            f"{CARD} card, skill_dirs + {lib_dir}",
        "base_version": got["version"]})
    if st != 200:
        sys.exit(f"[library-install] PUT packages: {st} {put.get('error')}")
    say(f"packages v{got['version']} -> v{put['version']}")
    if prof is not None:
        st, put = api(a.base, token, "PUT", f"/v1/conductor/profiles/{a.profile}", {
            "doc": prof, "note": f"R1: library {a.library}", "base_version": p["version"]})
        if st != 200:
            sys.exit(f"[library-install] PUT profile {a.profile}: {st} {put.get('error')}")
        say(f"profile {a.profile} v{p['version']} -> v{put['version']}: library {a.library}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
