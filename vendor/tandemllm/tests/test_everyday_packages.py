"""The everyday package library (PKG2, 2026-10-07): packages/everyday/<name>/SKILL.md, for the
classes a plain chat lands on, written for small models, needing no tool.

The test matrix:

  1  every folder is a SKILL.md whose frontmatter name is the folder, with a description of
     8 to 600 characters, a body that is one text (no `## Reference`), and no `{tandem_url}`
                                                             test_every_file_is_a_package
  2  every package is in the shipped packages document as `skill:<name>` with `requires_tools`
     empty, and the shipped directories reach it; the core package is `general`; its text goes
     in uncut (the body fits max_tokens x 3.5 characters) and the router's `requires` is empty
                                                             test_registered_and_uncut
  3  every package is a candidate of at least one shipped class, every class candidate exists,
     the core is on every class that lists an everyday package, and no class lists more than
     eleven candidates (the package question's list)      test_class_wiring
  4  no private text: no hostname, address, personal name, tool or internal name in a body or a
     description; every body has the small-model shape (numbered steps, an example, a Do and a
     Don't, a check line)                                 test_plain_and_shaped
  5  every package has an eval set with 5 to 8 prompts in at least two of English, German and
     Arabic, unique ids, and a rubric of 4 to 6 lines     test_eval_sets
  6  INDEX.md names every package once with its classes   test_index

Run: python tests/test_everyday_packages.py
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import packages as packages_mod, schema  # noqa: E402

EVERYDAY = os.path.join(ROOT, "packages", "everyday")
CORE = "everyday-core"
EVERYDAY_CLASSES = ("writing", "plan", "research", "summarize", "chat")
PRIVATE = re.compile(r"(?i)(khaled|bakeer|spark\b|dgx|192\.168|\.khaled\.de|laya|tandem|opus|"
                     r"claude|fable|anthropic|open webui|opencode|hermes|\{tandem_url\}|"
                     r"\{tandem_scratch\})")
ARABIC = re.compile(r"[؀-ۿ]")


def names() -> list[str]:
    return sorted(n for n in os.listdir(EVERYDAY)
                  if os.path.isfile(os.path.join(EVERYDAY, n, "SKILL.md")))


def read(n: str) -> tuple[dict, str]:
    with open(os.path.join(EVERYDAY, n, "SKILL.md"), encoding="utf-8") as f:
        return packages_mod.split_frontmatter(f.read())


def test_every_file_is_a_package():
    ns = names()
    assert len(ns) >= 30, ns
    for n in ns:
        meta, body = read(n)
        assert meta.get("name") == n, (n, meta)
        assert 8 <= len(meta.get("description", "")) <= 600, n
        assert packages_mod.split_reference(body)[1] == "", f"{n}: a reference section"
        assert packages_mod.URL_TOKEN not in body, n
        assert body.lstrip().startswith("# "), f"{n}: no title line"


def test_registered_and_uncut():
    pk = schema.SHIPPED_PACKAGES
    assert "{repo}/packages/everyday" in pk["skill_dirs"]
    lib = packages_mod.Library()
    # the repository's own texts: a box's skill copies (~/.qwen38-spark-engine/conductor/skills,
    # earlier in skill_dirs) hold the installed copies and would answer first (PKG3)
    pk = dict(pk, skill_dirs=[d for d in pk["skill_dirs"] if d.startswith(packages_mod.REPO_TOKEN)])
    for n in names():
        p = pk["packages"].get(n)
        assert p is not None, f"{n} not in the shipped packages"
        # PKG3: the measured-neutral ones ship off (schema.EVERYDAY_OFF)
        assert p["enabled"] == (n not in schema.EVERYDAY_OFF), (n, p["enabled"])
        assert p["source"] == f"skill:{n}" and p.get("requires_tools") == [], (n, p)
        assert bool(p.get("general")) == (n == CORE), n
        assert not p.get("library"), n
        path = lib.path(pk, n)
        assert path and path.startswith(EVERYDAY), (n, path)
        # the text is checked whole even for a package that ships off (it can be switched on)
        text, err = lib.text(dict(pk, packages={**pk["packages"], n: dict(p, enabled=True)}), n)
        assert err is None and text, (n, err)
        assert packages_mod.CUT_NOTE not in text, f"{n}: cut at max_tokens {p['max_tokens']}"
        assert lib.requires(pk, n) == (), n


def test_class_wiring():
    pk = schema.SHIPPED_PACKAGES
    ns = set(names())
    listed = set()
    for c in schema.SHIPPED_ROUTER["classes"]:
        cands = c["policy"]["packages"]
        for p in cands:
            assert p in pk["packages"], (c["id"], p)
        mine = [p for p in cands if p in ns]
        listed |= set(mine)
        if c["id"] in EVERYDAY_CLASSES:
            assert CORE in mine, (c["id"], "no core")
            # the package question's list: the candidates minus the general ones
            asked = [p for p in cands if not pk["packages"][p].get("general")
                     and p not in c["policy"].get("core", [])]
            # PKG3 (measured 2026-10-08): over all 33 the question picked the prompt's package
            # 94 of 106 times, over chat's 9 38 (none 51): every prose class asks the whole set
            assert asked == schema.EVERYDAY_ALL, (c["id"], asked)
        else:
            assert CORE not in mine, c["id"]
    assert ns <= listed, ns - listed


def test_plain_and_shaped():
    pk = schema.SHIPPED_PACKAGES
    for n in names():
        meta, body = read(n)
        blob = body + "\n" + meta["description"] + "\n" + pk["packages"][n]["description"]
        hits = [m.group(0) for m in PRIVATE.finditer(blob)]
        assert not hits, (n, hits)
        if n == CORE:
            continue
        low = body.lower()
        assert re.search(r"(?m)^1\. ", body), f"{n}: no numbered steps"
        assert "example" in low and "do:" in low and "don't:" in low, f"{n}: shape"
        assert "before you answer, check" in low, f"{n}: no check line"
        # a template where the answer has a fixed shape; the creative one has length defaults
        assert "output template" in low or "defaults" in low, f"{n}: no template or defaults"


def test_eval_sets():
    ns = [n for n in names() if n != CORE]
    seen = set()
    for n in ns:
        path = os.path.join(EVERYDAY, "evals", f"{n}.json")
        assert os.path.isfile(path), path
        with open(path, encoding="utf-8") as f:
            ev = json.load(f)
        assert ev["package"] == n
        assert 4 <= len(ev["rubric"]) <= 6, n
        assert 5 <= len(ev["prompts"]) <= 8, n
        langs = set()
        for p in ev["prompts"]:
            assert p["id"] not in seen, p["id"]
            seen.add(p["id"])
            assert p["id"].startswith(n + "-") and len(p["text"]) >= 20, p
            langs.add("ar" if ARABIC.search(p["text"]) else p["lang"])
            assert not PRIVATE.search(p["text"]), p["id"]
        assert len(langs & {"en", "de", "ar"}) >= 2, (n, langs)


def test_index():
    with open(os.path.join(EVERYDAY, "INDEX.md"), encoding="utf-8") as f:
        text = f.read()
    for n in names():
        rows = re.findall(rf"(?m)^\| {re.escape(n)} \|", text)
        assert len(rows) == 1, (n, len(rows))


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print("ok", name)
    print(f"{passed} passed")
