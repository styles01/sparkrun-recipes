"""The package library's search index (R1, 2026-10-03): BM25 over the library packages' names,
descriptions and texts, the class fit from INDEX.md, rebuilt when a file or the document moves.

The test matrix:

  1  the build: every enabled library package from its SKILL.md (frontmatter off), the text
     before `## Reference` apart from the reference, the classes from INDEX.md's table (group
     rows and other lines ignored); a package without `library: true` is not in it
                                                        test_the_build
  2  the search ranks by the words of the query, name over description over text, each query
     word once; nothing that shares no word; `k`, `min_score`, `exclude`, disabled skills
                                                        test_the_search
  3  the class filter keeps the skills INDEX.md fits to the class; a skill INDEX.md does not list
     stays in                                           test_the_class_filter
  4  rebuilt when a SKILL.md changes, comes or goes, when INDEX.md changes and when the packages
     version moves; looked at once a second at most, never rebuilt when nothing moved
                                                        test_rebuilt_on_change
  5  the text that goes in: a library package's text stops before `## Reference`; any other
     package's text is the whole file as before (a `## Reference` in it stays)
                                                        test_the_injected_text
  6  time: a search over 44 skills in well under a millisecond (median), a build in a few
                                                        test_time
  7  the library itself (when it is on this machine): the bodies parse, every skill has a
     reference and classes, and plain task sentences find the skill a person would pick
                                                        test_the_real_library

Run: python tests/test_libsearch.py
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor import libsearch, schema  # noqa: E402
from server.conductor.packages import Library, with_url  # noqa: E402

SKILLS = {
    "browser-game": ("Building a 2D browser game (platformer, shooter, puzzle) - game loop with "
                     "fixed timestep, keyboard input, tile collisions, camera, HUD, lives.",
                     "Use a fixed timestep. Resolve collisions per axis. Keep lives in state.",
                     "```js\nfunction step(dt) {}\n```"),
    "canvas-2d": ("Drawing and animating on an HTML canvas with the 2D context - render loop, "
                  "device pixel ratio, sprites, transforms.",
                  "Scale by devicePixelRatio. Clear every frame. Batch draw calls.",
                  "ctx.setTransform(dpr, 0, 0, dpr, 0, 0);"),
    "pandas-cleaning": ("Loading and cleaning tabular data with pandas - reading CSV, types, "
                        "missing values, duplicates, dates.",
                        "Print row counts after every step. Parse dates explicitly.",
                        "df.drop_duplicates(subset=['id'])"),
    "pytest-practice": ("Writing and running Python tests with pytest - fixtures, parametrize, "
                        "tmp_path, monkeypatch.",
                        "One behaviour a test. Read the summary line.",
                        "pytest -q -x"),
    "debug-method": ("A method for fixing a bug when the cause is unknown - reproduce, shrink, "
                     "bisect, one cause, one change.",
                     "Reproduce before anything. Change one thing. Re-run the reproduction.",
                     "git bisect run pytest -q"),
    "git-workflow": ("Day-to-day git for a coding task - small commits, a branch per task, "
                     "resolving merge conflicts.",
                     "Read the diff before committing. Never commit secrets.",
                     "git add -p"),
}
CLASSES = {"browser-game": "code_edit, debug, drawing", "canvas-2d": "drawing, code_edit, debug",
           "pandas-cleaning": "code_edit, research", "pytest-practice": "code_edit, debug",
           "debug-method": "debug", "git-workflow": "code_edit, plan"}


def skill_md(name, desc, body, ref) -> str:
    return (f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n\n{body}\n\n"
            f"## Reference\n\n{ref}\n")


def index_md(names) -> str:
    rows = ["# Package library v1", "", "| name | description | body | ref | classes |",
            "|-|-|-|-|-|", "| **Web** | | | | |"]
    rows += [f"| {n} | {SKILLS[n][0]} | 300 | 400 | {CLASSES[n]} |" for n in names]
    rows += ["", "## Class coverage", "", "- `code_edit` (5): browser-game, ..."]
    return "\n".join(rows) + "\n"


def make_library(names=tuple(SKILLS), index=True) -> str:
    """A library directory as the installer leaves it on the box: <dir>/<name>/SKILL.md and
    INDEX.md."""
    d = tempfile.mkdtemp(prefix="lib-")
    for n in names:
        os.makedirs(os.path.join(d, n))
        with open(os.path.join(d, n, "SKILL.md"), "w") as f:
            f.write(skill_md(n, *SKILLS[n]))
    if index:
        with open(os.path.join(d, "INDEX.md"), "w") as f:
            f.write(index_md(names))
    return d


def lib_packages(d: str, names=tuple(SKILLS), base: dict | None = None) -> dict:
    """The packages document with the library registered as the installer registers it."""
    doc = base if base is not None else schema.seed("packages")
    doc["skill_dirs"] = list(doc["skill_dirs"]) + [d]
    for n in names:
        doc["packages"][n] = {"title": n.replace("-", " "), "kind": "skill",
                              "source": f"skill:{n}", "description": SKILLS[n][0], "body": "",
                              "max_tokens": 500, "enabled": True, "library": True}
    return doc


def lib_cfg(d: str) -> dict:
    return dict(schema.LIBRARY, index=os.path.join(d, "INDEX.md"))


def built(d=None, doc=None, clock=time.monotonic):
    d = d or make_library()
    doc = doc or lib_packages(d)
    idx = libsearch.SkillIndex(Library(), clock=clock)
    idx.ensure(1, doc, lib_cfg(d))
    return idx, d, doc


def names(hits) -> list[str]:
    return [e.name for e, _ in hits]


# ------------------------------------------------------------------ 1
def test_the_build():
    idx, d, doc = built()
    assert sorted(idx.by_name) == sorted(SKILLS), "every library package, nothing else"
    assert "ui-design" not in idx.by_name and "pencil-sketch" not in idx.by_name
    e = idx.by_name["browser-game"]
    assert e.description == SKILLS["browser-game"][0]
    assert e.body == "# browser-game\n\n" + SKILLS["browser-game"][1], repr(e.body)
    assert e.reference == "## Reference\n\n" + SKILLS["browser-game"][2]
    assert "name:" not in e.body, "the frontmatter is not text"
    assert e.classes == ["code_edit", "debug", "drawing"] and e.enabled
    assert idx.by_name["debug-method"].classes == ["debug"]
    assert e.card() == {"name": "browser-game", "description": e.description,
                        "classes": ["code_edit", "debug", "drawing"]}
    # INDEX.md's parser: group rows, the header, the separator and prose are not skills
    got = libsearch.index_classes(index_md(list(SKILLS)) + "| not a row\n| **Bold** | x | | | |\n")
    assert sorted(got) == sorted(SKILLS)
    assert libsearch.index_classes("| name | description | body | ref | classes |") == {}
    # an inline library package (no file) is indexed from its body
    doc["packages"]["inline-skill"] = {"title": "x", "kind": "skill", "source": "inline",
                                       "description": "An inline skill about websockets.",
                                       "body": "Reconnect with backoff.\n\n## Reference\n\nws",
                                       "max_tokens": 200, "enabled": True, "library": True}
    idx.ensure(2, doc, lib_cfg(d))
    assert idx.by_name["inline-skill"].body == "Reconnect with backoff."
    assert idx.by_name["inline-skill"].classes is None
    # a SKILL.md gone from the disk is left out
    os.remove(os.path.join(d, "git-workflow", "SKILL.md"))
    idx.ensure(3, doc, lib_cfg(d))
    assert "git-workflow" not in idx.by_name


# ------------------------------------------------------------------ 2
def test_the_search():
    idx, _, doc = built()
    assert names(idx.search("platformer game with keyboard input and lives", 3))[0] == \
        "browser-game"
    assert names(idx.search("clean a CSV with duplicates in pandas", 3))[0] == "pandas-cleaning"
    assert names(idx.search("write pytest fixtures", 3))[0] == "pytest-practice"
    assert names(idx.search("reproduce and bisect a bug whose cause is unknown", 3))[0] == "debug-method"
    assert idx.search("the of and a", 5) == [], "function words only: nothing"
    assert idx.search("zebra quantum", 5) == [], "no shared word: nothing"
    assert len(idx.search("game canvas pandas pytest git bug", 2)) == 2, "k"
    hits = idx.search("game canvas", 6)
    assert all(s > 0 for _, s in hits) and hits == sorted(hits, key=lambda x: -x[1])
    assert "browser-game" not in names(idx.search("platformer game", 5, exclude={"browser-game"}))
    top = idx.search("platformer game", 1)[0][1]
    assert idx.search("platformer game", 5, min_score=top + 1) == []
    # each query word counts once: repeating it does not change the ranking or the scores
    a = idx.search("game canvas", 5)
    b = idx.search("game game game canvas canvas", 5)
    assert [(e.name, round(s, 9)) for e, s in a] == [(e.name, round(s, 9)) for e, s in b]
    # the name outweighs the text: `debug` names debug-method, browser-game only lists it
    assert names(idx.search("debug", 6))[0] == "debug-method"
    # plural s and case
    assert names(idx.search("COLLISIONS", 1)) == names(idx.search("collision", 1))
    # a disabled skill is not found (the API and the router search the enabled ones)
    doc["packages"]["browser-game"]["enabled"] = False
    idx.ensure(2, doc, schema.LIBRARY)
    assert "browser-game" not in names(idx.search("platformer game", 6))
    assert "browser-game" in names(idx.search("platformer game", 6, enabled_only=False))


# ------------------------------------------------------------------ 3
def test_the_class_filter():
    idx, d, doc = built()
    q = "fix the bug in the canvas game and its pytest tests and the csv data and git"
    assert set(names(idx.search(q, 6, "debug"))) == {"browser-game", "canvas-2d",
                                                       "pytest-practice", "debug-method"}
    assert names(idx.search(q, 6, "research")) == ["pandas-cleaning"]
    assert idx.search(q, 6, "chat") == [], "no skill fits chat"
    assert len(idx.search(q, 6, None)) == 6
    # a skill INDEX.md does not list keeps its place in every class
    os.remove(os.path.join(d, "INDEX.md"))
    idx.ensure(2, doc, lib_cfg(d))
    assert all(e.classes is None for e in idx.entries)
    assert len(idx.search(q, 6, "chat")) == 6


# ------------------------------------------------------------------ 4
def test_rebuilt_on_change():
    now = [100.0]
    idx, d, doc = built(clock=lambda: now[0])
    cfg = lib_cfg(d)
    assert idx.builds == 1
    for _ in range(5):
        idx.ensure(1, doc, cfg)
    assert idx.builds == 1, "nothing moved: no rebuild, no stat within the second"
    now[0] += 2
    idx.ensure(1, doc, cfg)
    assert idx.builds == 1, "looked at after a second: still nothing moved"
    # a SKILL.md edited: the next look after a second rebuilds
    p = os.path.join(d, "git-workflow", "SKILL.md")
    with open(p, "w") as f:
        f.write(skill_md("git-workflow", SKILLS["git-workflow"][0],
                         "Rebase onto main before the merge request.", "git rebase -i"))
    os.utime(p, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    idx.ensure(1, doc, cfg)
    assert idx.builds == 1, "within the second the old index answers"
    now[0] += 1.5
    idx.ensure(1, doc, cfg)
    assert idx.builds == 2 and "rebase" in idx.by_name["git-workflow"].body.lower()
    assert names(idx.search("rebase", 1)) == ["git-workflow"]
    # a new skill in the document (a new version): rebuilt at once
    os.makedirs(os.path.join(d, "regex"))
    with open(os.path.join(d, "regex", "SKILL.md"), "w") as f:
        f.write(skill_md("regex", "Writing and testing regular expressions.", "Anchor it.", "re"))
    doc["packages"]["regex"] = dict(doc["packages"]["git-workflow"], source="skill:regex",
                                    description="Writing and testing regular expressions.")
    idx.ensure(2, doc, cfg)
    assert idx.builds == 3 and names(idx.search("regular expressions", 1)) == ["regex"]
    # INDEX.md changed: the classes follow
    with open(os.path.join(d, "INDEX.md"), "a") as f:
        f.write("| regex | Regular expressions. | 300 | 300 | code_edit |\n")
    os.utime(os.path.join(d, "INDEX.md"), ns=(time.time_ns() + 2 * 10**9,) * 2)
    now[0] += 1.5
    idx.ensure(2, doc, cfg)
    assert idx.builds == 4 and idx.by_name["regex"].classes == ["code_edit"]
    # a skill removed from the document
    del doc["packages"]["regex"]
    idx.ensure(3, doc, cfg)
    assert idx.builds == 5 and "regex" not in idx.by_name


# ------------------------------------------------------------------ 5
def test_the_injected_text():
    d = make_library()
    doc = lib_packages(d)
    lib = Library()
    text, err = lib.text(doc, "browser-game")
    assert err is None and text == "# browser-game\n\n" + SKILLS["browser-game"][1]
    assert "## Reference" not in text and SKILLS["browser-game"][2] not in text
    # the same file as a plain skill (no `library`): the whole text, as before R1
    plain = dict(doc["packages"]["browser-game"])
    del plain["library"]
    doc["packages"]["plain"] = plain
    full, _ = lib.text(doc, "plain")
    assert full.startswith("# browser-game") and full.endswith(SKILLS["browser-game"][2])
    assert "## Reference" in full
    # an inline package with a reference heading, not a library one: unchanged
    doc["packages"]["inl"] = {"title": "x", "kind": "package", "source": "inline",
                              "description": "an inline package", "max_tokens": 200,
                              "enabled": True, "body": "A.\n\n## Reference\n\nB."}
    assert lib.text(doc, "inl")[0] == "A.\n\n## Reference\n\nB."
    doc["packages"]["inl"]["library"] = True
    assert lib.text(doc, "inl")[0] == "A."
    # the status the Packages screen shows counts the injected text
    st = lib.status(doc)
    assert st["browser-game"]["chars"] == len(text) and st["plain"]["chars"] == len(full)
    assert schema.check("packages", doc) == [], schema.check("packages", doc)


# ------------------------------------------------------------------ 6
def test_time():
    d = tempfile.mkdtemp(prefix="lib44-")
    doc = schema.seed("packages")
    doc["skill_dirs"].append(d)
    base = list(SKILLS.items())
    for i in range(44):
        n, (desc, body, ref) = base[i % len(base)]
        name = f"{n}-{i}"
        os.makedirs(os.path.join(d, name))
        with open(os.path.join(d, name, "SKILL.md"), "w") as f:
            f.write(skill_md(name, desc, (body + " ") * 8, (ref + "\n") * 10))
        doc["packages"][name] = {"title": name, "kind": "skill", "source": f"skill:{name}",
                                 "description": desc, "body": "", "max_tokens": 500,
                                 "enabled": True, "library": True}
    idx = libsearch.SkillIndex(Library())
    t0 = time.perf_counter()
    idx.ensure(1, doc, dict(schema.LIBRARY, index=""))
    build = (time.perf_counter() - t0) * 1e3
    q = ("Build a small platformer game in the browser with a canvas, keyboard input, tile "
         "collisions and lives, then test it and fix the bugs you find. " * 6)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        idx.search(q, 4, "code_edit")
        ts.append((time.perf_counter() - t0) * 1e6)
    med = statistics.median(ts)
    print(f"    build {build:.2f} ms over 44 skills, search median {med:.1f} us "
          f"(p90 {sorted(ts)[180]:.1f} us)")
    assert len(idx.entries) == 44 and build < 200 and med < 1000, (build, med)


# ------------------------------------------------------------------ 7
REAL = [os.environ.get("TANDEM_LIBRARY_DIR") or "", os.path.expanduser("~/tandem-packages/library"),
        os.path.expanduser("~/r1lib/library")]


def test_the_real_library():
    d = next((p for p in REAL if p and os.path.isfile(os.path.join(p, "INDEX.md"))), None)
    if d is None:
        print("    (the library is not on this machine: skipped)")
        return
    lib = Library()
    names_ = sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n, "SKILL.md")))
    doc = schema.seed("packages")
    doc["skill_dirs"].append(d)
    for n in names_:
        meta, _ = libsearch.split_frontmatter(lib.file_text(os.path.join(d, n, "SKILL.md")))
        assert meta.get("name") == n and len(meta.get("description", "")) >= 8, n
        doc["packages"][n] = {"title": n, "kind": "skill", "source": f"skill:{n}",
                              "description": meta["description"][:600], "body": "",
                              "max_tokens": 500, "enabled": True, "library": True}
    assert schema.check("packages", doc) == []
    idx = libsearch.SkillIndex(lib)
    idx.ensure(1, doc, dict(schema.LIBRARY, index=os.path.join(d, "INDEX.md")))
    assert len(idx.entries) == len(names_) >= 44
    for e in idx.entries:
        assert e.reference.startswith("## Reference") and e.classes, e.name
        # the body uncut, its `{tandem_url}` and `{tandem_scratch}` filled in (the index keeps the
        # raw text; the model gets the addresses)
        assert lib.text(doc, e.name)[0] == with_url(e.body, None), "what goes in is the body, uncut"
    want = {
        "Build a 2D platformer in the browser: a hero that jumps, tile collisions, three lives":
            ("browser-game", "code_edit"),
        "Clean this CSV with pandas: drop duplicate rows, fix the date column, count nulls":
            ("pandas-cleaning", "code_edit"),
        "Add a REST endpoint to our FastAPI service with pydantic validation":
            ("fastapi-service", "code_edit"),
        "The Three.js scene drops to 20 fps with many meshes; use instancing":
            ("threejs-scene", "three_d"),
        "Write a bash script that backs up a folder and fails loudly":
            ("bash-scripting", "code_edit"),
        "Our websocket chat loses messages when the server restarts; add reconnect":
            ("websockets", "debug"),
        "Store timestamps in UTC and show them in the user's time zone":
            ("datetime-timezones", "code_edit"),
    }
    for q, (name, cls) in want.items():
        got = names(idx.search(q, 4, cls))
        print(f"    {name:20s} <- {got}")
        assert name in got[:3], (q, got)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
