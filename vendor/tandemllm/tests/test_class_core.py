"""FX6 (2026-10-06): a class's core skill always goes in, the package question knows the class, and a
low-confidence deliverable kind is stated in one line, not asked.

The live case: "Create a new folder and create there a pagoda garden" went to three_d at 0.464 (Qwen;
design_ui 0.215, code_edit 0.179, drawing 0.107), accepted at 0.45. The package question, asked of
Qwen with the CLASS question's prompt ("You sort requests ... into task classes") and without the
class, answered `none` at 0.747 (threejs-scene 0.246): the turn got the testing packages and no 3D
skill, and the model built a flat SVG page.

The test matrix:

  1  a class's `core` package goes in on every turn of the class even when the package question
     answers none; it is not offered to the question; the row names it (`package_core`)
                                                   test_core_goes_in_when_the_chooser_says_none
  2  a core name the class does not list is ignored; a core skill the client loaded is not sent
     again; a profile's routing `core` replaces the class's
                                                   test_core_edges
  3  the Qwen stage asks the package question with the class question's prompt, byte for byte as
     before (three prompts that name the class lost test-skill picks on code_edit and debug); a
     router's own `package_prompt` gets the routed class and the package question
                                                   test_the_package_question_prompt
  4  the reading line: three_d at 0.464 with design_ui at 0.215 puts "Building it as a 3D
     three.js scene. Say if you wanted a flat web page." into the note under `classify`, and the
     row says why                                  test_reading_line_on_an_unsure_new_class
  5  no reading line: confident (>= below), the other class under `other`, the other class with
     no `reading`, the class itself with no `reading`, a later turn of the same class, the
     fallback/sticky class, mode off; mode shadow journals the text and the note has none
                                                   test_no_reading_line
  6  the schema: `core`, `reading`, `package_prompt` and `classifier.reading` are taken; a core
     package that does not exist, a reading text without {reading} and a package prompt without
     {classes} are refused; a router stored before FX6 is still valid
                                                   test_schema
  7  the shipped router (the audit): every class that builds something keeps its defining skill
     out of the question's reach (general or core); three_d's core is threejs-scene; the shipped
     documents pass the checks                     test_shipped_router_audit
  8  the library never picks a package the class's note already carries
                                                   test_the_library_skips_a_core_skill
  9  with the part-level note dedup (QWEN38_NOTE_DEDUP=parts): a later user message of the same
     class carries no note again, the core skill and the reading line sit once in the served
     prompt and the first note keeps its place; `block` (the old rule) brings the core again
                                                   test_core_and_reading_are_not_injected_again

Run: python tests/test_class_core.py
"""

from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import COMPOSITE, TOOLS, setup, turn  # noqa: E402

from server.conductor import schema  # noqa: E402
from server.conductor.classify import Answer  # noqa: E402

PAGODA = "Create a new folder and create there a pagoda garden"
# the first turn's class answer, as journaled on the box (request ...8d3b155b)
PAGODA_PROBS = {"three_d": 0.4643, "design_ui": 0.2145, "code_edit": 0.1792, "drawing": 0.1066,
                "chat": 0.0242, "plan": 0.0099, "research": 0.0003}
LINE = "Building it as a 3D three.js scene. Say if you wanted a flat web page."


class Full:
    """A Qwen stage with the whole distribution: `fn(text, options, prompt)` -> (label, probs)."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def choice(self, state, prompt, options, max_chars=None):
        self.calls.append((state, prompt, dict(options)))
        label, probs = self.fn(state, options, prompt)
        return Answer("qwen", label, probs[label], dict(probs), 1.0, model="fake")


def pagoda_qwen(cls_probs=None, pkg=("none", 0.7472)):
    """The class question answers from `cls_probs`; the package question answers `pkg`."""
    cls_probs = cls_probs or PAGODA_PROBS

    def fn(state, options, prompt):
        if "none" in options:
            label, p = pkg
            rest = [o for o in options if o != label]
            probs = {label: p, **{o: (1 - p) / max(1, len(rest)) for o in rest}}
            return label, probs
        label = max(cls_probs, key=cls_probs.get)
        return label, {k: v for k, v in cls_probs.items() if k in options}
    return fn


def make(qwen_fn=None, three_d=None, reading=None, readings=True, profile=None):
    """A conductor whose Qwen stage is `Full(qwen_fn)` (the decider down), three_d listing an
    inline 3D skill as its core plus two recipes, and the reading phrases on the visual classes."""
    cond, fakes = setup()
    q = Full(qwen_fn or pagoda_qwen())
    fakes["qwen"] = q
    cond.router.factory = lambda kind, role, timeout: fakes[kind]
    cond.router._adapters.clear()
    _, p = cond.store.get("packages")
    p["packages"]["threejs-scene"] = {
        "title": "threejs scene", "kind": "skill", "source": "inline",
        "description": "Building a WebGL scene with Three.js.", "body": "# Three.js scene\n\nA "
        "renderer, a camera, lights.", "max_tokens": 500, "enabled": True}
    for n in ("photo-to-3d", "voxel-seasons"):
        p["packages"][n] = dict(p["packages"][n], source="inline", body=f"# {n} recipe")
    cond.store.put("packages", None, p, note="fx6", author="t")
    _, r = cond.store.get("router")
    r["classifier"]["stages"] = [{"kind": "decider", "enabled": False, "accept": 0.8,
                                  "timeout_ms": 400},
                                 {"kind": "qwen", "enabled": True, "accept": 0.45,
                                  "timeout_ms": 8000}]
    for c in r["classes"]:
        if c["id"] == "three_d":
            c["policy"] = three_d or {"effort": "high", "rules": [],
                                      "packages": ["threejs-scene", "photo-to-3d",
                                                   "voxel-seasons"],
                                      "core": ["threejs-scene"]}
        if readings:
            c.update({"three_d": {"reading": "a 3D three.js scene"},
                      "design_ui": {"reading": "a flat web page"},
                      "drawing": {"reading": "a 2D drawing"}}.get(c["id"], {}))
    if reading is not None:
        r["classifier"]["reading"] = reading
    cond.store.put("router", None, r, note="fx6", author="t")
    if profile:
        _, pr = cond.store.get("profiles", "coding")
        pr.update(profile)
        cond.store.put("profiles", "coding", pr, note="fx6", author="t")
    return cond, q


def ask(cond, text=PAGODA, sid="s-fx6", msgs=None):
    b = {"model": COMPOSITE, "tools": TOOLS,
         "messages": msgs or [{"role": "user", "content": text}]}
    plan, ids, _ = turn(cond, b, headers={"x-session-id": sid})
    return plan


def parts(plan) -> list[tuple[str, str]]:
    return [(x["kind"], x["name"]) for n in plan.injected for x in n["parts"]]


def note(plan) -> str:
    return "".join(n["text"] for n in plan.injected)


# ------------------------------------------------------------------ 1
def test_core_goes_in_when_the_chooser_says_none():
    cond, q = make()
    plan = ask(cond)
    assert plan.row["class_id"] == "three_d" and abs(plan.row["confidence"] - 0.4643) < 1e-6
    assert ("package", "threejs-scene") in parts(plan), parts(plan)
    assert "# Three.js scene" in note(plan)
    assert plan.row["extra"]["package_core"] == ["threejs-scene"]
    ch = plan.row["extra"]["package_choice"]
    assert ch == {"candidates": ["photo-to-3d", "voxel-seasons"], "chosen": None, "how": "none"}
    asked = [o for _, _, o in q.calls if "none" in o]
    assert len(asked) == 1 and "threejs-scene" not in asked[0], "the core skill is not asked"
    # before FX6 (no core): the same answers leave the turn without the 3D skill
    cond, _ = make(three_d={"effort": "high", "rules": [],
                            "packages": ["threejs-scene", "photo-to-3d", "voxel-seasons"]})
    plan = ask(cond)
    assert ("package", "threejs-scene") not in parts(plan)
    assert plan.row["extra"]["package_choice"]["chosen"] is None


# ------------------------------------------------------------------ 2
def test_core_edges():
    # a core name the class does not list is ignored
    cond, _ = make(three_d={"effort": "high", "rules": [], "packages": ["photo-to-3d"],
                            "core": ["threejs-scene"]})
    plan = ask(cond)
    assert ("package", "threejs-scene") not in parts(plan)
    assert "package_core" not in plan.row["extra"]
    # the client loaded the skill itself: not sent again, and not asked about
    cond, q = make()
    msgs = [{"role": "user", "content": "Load the threejs skill."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "k", "type": "function",
                 "function": {"name": "skill", "arguments": {"name": "threejs-scene"}}}]},
            {"role": "tool", "tool_call_id": "k", "content": "loaded"},
            {"role": "user", "content": PAGODA}]
    plan = ask(cond, msgs=msgs, sid="s-loaded")
    assert ("package", "threejs-scene") not in parts(plan)
    assert "already_loaded:threejs-scene" in plan.row["extra"]["notes"]
    # a profile's routing `core` replaces the class's
    cond, _ = make(profile={"routing": {"three_d": {"core": []}}})
    plan = ask(cond)
    assert ("package", "threejs-scene") not in parts(plan)
    assert "threejs-scene" in plan.row["extra"]["package_choice"]["candidates"]


# ------------------------------------------------------------------ 3
def test_the_package_question_prompt():
    cond, q = make(three_d={"effort": "high", "rules": [],
                            "packages": ["threejs-scene", "photo-to-3d", "voxel-seasons"]})
    ask(cond)
    _, r = cond.store.get("router")
    cls_prompt = [p for _, p, o in q.calls if "none" not in o]
    pkg_prompt = [p for _, p, o in q.calls if "none" in o]
    assert len(cls_prompt) == 1 and cls_prompt[0].startswith("You sort requests")
    assert len(pkg_prompt) == 1
    _, _, opts = [c for c in q.calls if "none" in c[2]][0]
    listing = "\n".join(f"- {k}: {v}" for k, v in opts.items())
    assert pkg_prompt[0] == r["classifier"]["qwen_prompt"].replace("{classes}", listing), \
        "the default is the class question's prompt with the packages as classes, as before"
    assert "- threejs-scene: Building a WebGL scene with Three.js." in pkg_prompt[0]
    assert "- none: None of these fits the request." in pkg_prompt[0]
    # the measured alternative stays available
    _, r = cond.store.get("router")
    r["classifier"]["package_prompt"] = schema.PACKAGE_PROMPT_CLASS
    cond.store.put("router", None, r, note="class prompt", author="t")
    ask(cond, sid="s-class")
    p = [p for _, p, o in q.calls if "none" in o][-1]
    assert p.startswith("A request to a coding assistant was sorted into this task class: "
                        "three_d: 3D scene, A 3D scene or model"), p
    assert "Which of these instructions fits this request best?" in p and "{" not in p
    # a router's own package prompt is used, with its placeholders filled
    cond, q = make(three_d={"effort": "high", "rules": [],
                            "packages": ["threejs-scene", "photo-to-3d"]})
    _, r = cond.store.get("router")
    r["classifier"]["package_prompt"] = "Class {class}. {question}\n{classes}"
    cond.store.put("router", None, r, note="own prompt", author="t")
    ask(cond, sid="s-own")
    p = [p for _, p, o in q.calls if "none" in o][0]
    assert p.startswith("Class three_d: 3D scene, A 3D scene") and "{" not in p, p


# ------------------------------------------------------------------ 4
def test_reading_line_on_an_unsure_new_class():
    cond, _ = make()
    plan = ask(cond)
    assert ("reading", "three_d") in parts(plan), parts(plan)
    assert LINE in note(plan), note(plan)
    assert "do not stop to ask" in note(plan)
    rd = plan.row["extra"]["reading"]
    assert rd == {"class": "three_d", "confidence": 0.4643, "other": "design_ui",
                  "other_p": 0.2145, "mode": "on"}
    # the reading line comes before the skills, after the effort sentence
    order = [k for k, _ in parts(plan)]
    assert order.index("reading") < order.index("package")
    # the test box shows it too
    out = cond.router.explain(PAGODA, "coding")
    assert LINE in out["note"] and {"kind": "reading", "name": "three_d"} in out["note_parts"]


# ------------------------------------------------------------------ 5
def test_no_reading_line():
    def none_of(cond, **kw):
        plan = ask(cond, **kw)
        assert not any(k == "reading" for k, _ in parts(plan)), parts(plan)
        assert "Say if you wanted" not in note(plan)
        return plan

    sure = dict(PAGODA_PROBS, three_d=0.85, design_ui=0.10, code_edit=0.03, drawing=0.02)
    none_of(make(pagoda_qwen(sure))[0])
    weak = {"three_d": 0.55, "design_ui": 0.10, "code_edit": 0.30, "drawing": 0.05}
    none_of(make(pagoda_qwen(weak))[0])          # the runner-up has no reading, the next < 0.15
    none_of(make(readings=False)[0])             # no class has a reading
    cond, _ = make(pagoda_qwen({"design_ui": 0.46, "code_edit": 0.40, "three_d": 0.14}))
    none_of(cond)                                # three_d at 0.14 is under `other`
    plan = none_of(make(reading={"mode": "off"})[0])
    assert "reading" not in plan.row["extra"]
    # shadow: journaled, not in the note
    cond, _ = make(reading={"mode": "shadow"})
    plan = none_of(cond)
    assert plan.row["shadow"]["reading"].startswith("Build it as a 3D three.js scene.")
    assert plan.row["extra"]["reading"]["mode"] == "shadow"
    # a later turn of the same class: the line was said when the class started
    cond, _ = make()
    first = ask(cond, sid="s-two")
    assert ("reading", "three_d") in parts(first)
    msgs = [{"role": "user", "content": PAGODA},
            {"role": "assistant", "content": "Building it as a 3D three.js scene."},
            {"role": "user", "content": "Add a koi pond next to the pagoda"}]
    later = ask(cond, msgs=msgs, sid="s-two")
    assert later.row["class_id"] == "three_d"
    assert later.block is None or "reading" not in [x["kind"] for x in later.block["parts"]]
    assert "reading" not in later.row["extra"]
    # below the accept threshold: the class is the fallback, no reading
    cond, _ = make(pagoda_qwen({"three_d": 0.40, "design_ui": 0.35, "drawing": 0.25}))
    plan = none_of(cond)
    assert plan.row["source"] == "fallback"


# ------------------------------------------------------------------ 6
def test_schema():
    pk = copy.deepcopy(schema.SHIPPED_PACKAGES)
    r = copy.deepcopy(schema.SHIPPED_ROUTER)
    assert schema.check("router", r, packages=pk) == [], schema.check("router", r, packages=pk)
    old = copy.deepcopy(r)
    for c in old["classes"]:
        c.pop("reading", None)
        c["policy"].pop("core", None)
    old["classifier"].pop("reading")
    assert schema.check("router", old, packages=pk) == [], "a router from before FX6 is valid"
    assert schema.reading(old) == schema.READING and schema.package_prompt(old) is None
    bad = copy.deepcopy(r)
    bad["classes"][4]["policy"]["core"] = ["no-such-skill"]
    bad["classifier"]["reading"]["text"] = "Build it the way the class says."
    bad["classifier"]["package_prompt"] = "Pick one of the instructions, please."
    errs = schema.check("router", bad, packages=pk)
    assert any("policy.core: no package 'no-such-skill'" in e for e in errs), errs
    assert any("reading.text: must contain {reading}" in e for e in errs), errs
    assert any("package_prompt: must contain {classes}" in e for e in errs), errs
    worse = copy.deepcopy(r)
    worse["classifier"]["reading"] = {"mode": "loud"}
    worse["classes"][0]["reading"] = ""
    errs = schema.check("router", worse, packages=pk)
    assert len(errs) >= 2, errs


# ------------------------------------------------------------------ 7
def test_shipped_router_audit():
    """Every class that lists packages: its defining skill is general or core, never only a choice
    the question may answer none to. Classes without packages (writing, plan, research, summarize,
    chat) have nothing to miss. drawing's candidates are three portrait recipes from a reference
    image; none of them is a drawing class's general skill, so `none` there is a right answer."""
    pk = schema.SHIPPED_PACKAGES["packages"]
    by = {c["id"]: c for c in schema.SHIPPED_ROUTER["classes"]}
    assert by["three_d"]["policy"]["core"] == ["threejs-scene", "coding"]
    assert {"threejs-scene", "coding"} <= set(by["three_d"]["policy"]["packages"])
    for cid in ("code_edit", "debug"):
        pol = by[cid]["policy"]
        assert "coding" in pol["packages"] and ("coding" in pol.get("core", [])
                                                 or pk["coding"].get("general")), cid
    assert {by[c].get("reading") for c in ("three_d", "design_ui", "drawing")} == \
        {"a 3D three.js scene", "a flat web page", "a 2D drawing"}
    for c in by.values():
        assert set(c["policy"].get("core", [])) <= set(c["policy"]["packages"]), c["id"]
    for kind, name, doc in __import__("server.conductor.shipped", fromlist=["x"]).documents():
        if kind == "router":
            assert schema.check(kind, doc, packages=schema.SHIPPED_PACKAGES) == []


# ------------------------------------------------------------------ 8
def test_the_library_skips_a_core_skill():
    from test_library_router import lib_cond
    from test_snapshots import S, served
    cond, fakes = lib_cond("on")
    _, r = cond.store.get("router")
    for c in r["classes"]:
        if c["id"] == "code_edit":
            c["policy"]["packages"] = ["browser-game"]
            c["policy"]["core"] = ["browser-game"]
    cond.store.put("router", None, r, note="browser-game as code_edit's core", author="t")
    plan, _ = served(cond, list(S().msgs))
    lib = plan.row["extra"]["library"]
    assert "browser-game" not in [c["name"] for c in lib["candidates"]], lib
    assert "browser-game" in (lib.get("live") or []), lib
    assert ("package", "browser-game") in parts(plan)
    assert not any(x["parts"][0]["name"] == "browser-game" for x in plan.extras
                   if x["parts"][0]["kind"] == "library")


# ------------------------------------------------------------------ 9
def test_core_and_reading_are_not_injected_again():
    from test_conductor_router import text_of
    from server.conductor import router as router_mod
    for mode in ("parts", "block"):
        old = router_mod.NOTE_DEDUP
        router_mod.NOTE_DEDUP = mode
        try:
            cond, _ = make()
            msgs = [{"role": "user", "content": PAGODA}]
            b = {"model": COMPOSITE, "tools": TOOLS, "messages": list(msgs)}
            p1, ids1, _ = turn(cond, b, headers={"x-session-id": "s-fx6-dedup"})
            msgs += [{"role": "assistant", "content": "Building it as a 3D three.js scene. Done."},
                     {"role": "user", "content": "Make the pagoda taller and add a koi pond."}]
            b = {"model": COMPOSITE, "tools": TOOLS, "messages": list(msgs)}
            p2, ids2, _ = turn(cond, b, headers={"x-session-id": "s-fx6-dedup"})
        finally:
            router_mod.NOTE_DEDUP = old
        assert ("package", "threejs-scene") in parts(p1) and ("reading", "three_d") in parts(p1)
        if mode == "parts":
            assert not p2.injected, parts(p2)
            t2 = text_of(ids2)
            assert t2.count("# Three.js scene") == 1, t2.count("# Three.js scene")
            assert t2.count(LINE) == 1
            # the first note replays where it sat: the prompt before it is the same
            first = p2.row["replayed"]["positions"][0]
            assert ids2[:first] == ids1[:first]
        else:
            assert ("package", "threejs-scene") in parts(p2), parts(p2)
            assert ("reading", "three_d") not in parts(p2)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
