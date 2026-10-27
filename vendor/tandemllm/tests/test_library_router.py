"""The router's retrieval from the package library (R1, 2026-10-03): on a new user message (and on a
low-quality or stuck outcome) a BM25 search over the message's digest, the class's fit as the
filter, the package question as the final pick among the top candidates, the skill as a note at
the prompt's tail replayed at its place; shadow first; the profile switch `library: off|shadow|on`.

The test matrix:

  1  off (no `library` in the profile, `library: off`, or no library at all): the served prompt
     ids, the journal rows and the questions asked are the same, and the index is never built
                                                        test_off_is_identical
  2  shadow: the row says what would go in (the trigger, the candidates with their scores, the
     pick, how), the card it would keep, and the prompt is the one `off` serves; the package
     question saw the candidates and `none`              test_shadow_journals_and_changes_nothing
  3  on: the first turn gets the card and the skill (its text before `## Reference`, under the
     header) after the class's block; the tool turns after it ask nothing and replay them at
     their place, so each prompt starts with the one before it up to the notes' end
                                                        test_on_injects_and_replays
  4  a later user message: a skill live in the session is not offered again; a new pick joins
                                                        test_a_new_message_and_a_live_skill
  5  a compaction gets nothing; the message after it gets the card and a skill again
                                                        test_after_a_compaction
  6  a stuck outcome (J1) searches with its reasons, once per signature, and counts as an act;
     an outcome not listed does nothing                 test_an_outcome_retrieves
  7  the question answers none: nothing goes in; `max_picks` 2: two skills, the second asked
     without the first; nothing matches: no question    test_none_two_picks_and_no_match
  8  under the engine lock (the in-context stage) Qwen is never asked through the port: the
     decider stages only, `locked` when there is none  test_locked_asks_no_qwen
  9  the profile in shadow lowers `on` to shadow; the test box shows the pick; the schema takes
     the switch and the section and refuses wrong values; the seeds stay without them
                                                        test_shadow_profile_test_box_and_schema

Run: python tests/test_library_router.py
"""

from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import QTok, setup  # noqa: E402
from test_libsearch import SKILLS, lib_packages, make_library  # noqa: E402
from test_snapshots import S, served, text_session  # noqa: E402

from server import app  # noqa: E402
from server.conductor import libsearch, outcomes, schema  # noqa: E402

CARD = ("# The skill library (TandemLLM)\n- Search it: curl -s \"$TANDEM_URL/v1/library/search"
        "?q=...\"")
NEW_TASK = "Now export the high scores as a CSV and clean the duplicates with pandas."


def decider(picks=("browser-game",)):
    """The decider: code_edit for the class; for the package question the first of `picks`
    among the options, else none."""
    def fn(text, options):
        if "code_edit" in options:
            return ("code_edit", 0.91)
        for p in picks:
            if p in options:
                return (p, 0.9)
        return ("none", 0.9) if "none" in options else "decider_unknown_label"
    return fn


def lib_cond(mode, picks=("browser-game",), card=True, section=None, profile="coding",
             install=True, outcomes_=None):
    cond, fakes = setup(decider=decider(picks))
    d = make_library()
    if install:
        _, doc = cond.store.get("packages")
        doc = lib_packages(d, base=doc)
        if card:
            doc["packages"]["tandem-library"] = {
                "title": "The skill library", "kind": "package", "source": "inline",
                "description": "How to search the TandemLLM skill library.", "body": CARD,
                "max_tokens": 300, "enabled": True}
        cond.store.put("packages", None, doc, note="the library", author="t")
        _, r = cond.store.get("router")
        r["library"] = dict(section or {}, index=os.path.join(d, "INDEX.md"))
        cond.store.put("router", None, r, note="the library section", author="t")
    _, p = cond.store.get("profiles", profile)
    if mode is not None:
        p["library"] = mode
    if outcomes_ is not None:
        p["outcomes"] = dict(outcomes_)
    if mode is not None or outcomes_ is not None:
        cond.store.put("profiles", profile, p, note=f"library {mode}", author="t")
    return cond, fakes


def kinds(p) -> list[str]:
    return [n["parts"][0]["kind"] for n in p.injected]


def lib_notes(p) -> list[str]:
    return [n["parts"][0]["name"] for n in p.injected if n["parts"][0]["kind"] == "library"]


def view(p) -> dict:
    r = p.row
    return {k: r.get(k) for k in ("class_id", "effort", "budget", "thinking", "decisions",
                                  "shadow", "injected", "replayed", "acted", "changed",
                                  "extra")} | {"extras": p.extras}


# ------------------------------------------------------------------ 1
def test_off_is_identical():
    turns = text_session() + [text_session()[-1] + [
        {"role": "assistant", "content": "The game runs."}, {"role": "user", "content": NEW_TASK}]]
    bare, fb = lib_cond(None, install=False)
    absent, fa = lib_cond(None)
    off, fo = lib_cond("off")
    for i, msgs in enumerate(turns):
        pb, ib = served(bare, msgs)
        pa, ia = served(absent, msgs)
        po, io = served(off, msgs)
        assert ib == ia == io, f"turn {i}: the same prompt ids"
        assert view(pb) == view(pa) == view(po), (i, view(pb), view(pa))
        assert "library" not in pa.row["extra"] and "library" not in pa.decisions
    for k in ("decider", "qwen"):
        assert fb[k].calls == fa[k].calls == fo[k].calls, k
    assert absent.router.index.builds == 0 and off.router.index.builds == 0, \
        "with the switch off the index is never built"


# ------------------------------------------------------------------ 2
def test_shadow_journals_and_changes_nothing():
    sh, fakes = lib_cond("shadow")
    off, _ = lib_cond("off")
    s = S()
    turns = [list(s.msgs), s.step(("bash", {"command": "ls"}, "a")),
             s.step(("bash", {"command": "ls -la"}, "b"))]
    for i, msgs in enumerate(turns):
        ps, ids_s = served(sh, msgs)
        po, ids_o = served(off, msgs)
        assert ids_s == ids_o, f"turn {i}: shadow serves the prompt off serves"
        assert ps.injected == po.injected and not any(
            x["parts"][0]["kind"].startswith("library") for x in ps.extras)
        assert ps.row["shadow"]["library_card"] == {"name": "tandem-library",
                                                     "chars": len(CARD)}
        if i == 0:
            lib = ps.row["extra"]["library"]
            assert lib["mode"] == "shadow" and lib["trigger"] == "message"
            cands = [c["name"] for c in lib["candidates"]]
            assert cands[0] == "browser-game" and set(cands) <= {
                "browser-game", "canvas-2d", "pytest-practice", "git-workflow",
                "pandas-cleaning"}, cands
            assert all(c["score"] > 0 for c in lib["candidates"]) and len(cands) <= 4
            assert lib["chosen"] == ["browser-game"] and lib["how"] == "decider"
            assert ps.row["shadow"]["library"] == {"chosen": ["browser-game"],
                                                   "trigger": "message"}
            assert lib["search_ms"] >= 0 and lib["query_chars"] > 0
            asked = [o for _, o in fakes["decider"].calls if "none" in o]
            assert len(asked) == 1 and list(asked[0]) == cands + ["none"], asked
            assert asked[0]["browser-game"] == SKILLS["browser-game"][0]
        else:
            assert "library" not in ps.row["extra"], "a tool turn asks nothing"
    assert sh.router.index.builds == 1


# ------------------------------------------------------------------ 3
def test_on_injects_and_replays():
    cond, fakes = lib_cond("on")
    s = S()
    p1, t1 = served(cond, list(s.msgs))
    assert kinds(p1)[-2:] == ["library_card", "library"], kinds(p1)
    note = p1.injected[-1]
    head = schema.LIBRARY["header"].replace("{name}", "browser-game")
    body = "# browser-game\n\n" + SKILLS["browser-game"][1]
    assert note["text"] == schema.ROUTER["injection"]["wrapper"].replace(
        "{text}", head + "\n\n" + body), note["text"]
    assert "## Reference" not in note["text"] and note["reason"] == "library browser-game (message)"
    assert p1.injected[-2]["text"].endswith(CARD + "\n</runtime-note>")
    starts = [n["at"] for n in p1.injected]
    assert starts == sorted(starts), "the class's block, then the card, then the skill"
    assert p1.row["extra"]["library"]["applied"] == ["browser-game"] and p1.row["acted"] == 1
    assert p1.decisions["library"] == "on"
    end = note["at"] + note["tokens"]
    asked = len(fakes["decider"].calls)
    for step in (("bash", {"command": "ls"}, "a"), ("bash", {"command": "ls -la"}, "b")):
        p, t = served(cond, s.step(step))
        assert lib_notes(p) == [] and "library_card" not in kinds(p)
        assert p.row["replayed"]["count"] == 3, p.row["replayed"]
        assert t[:end] == t1[:end], "the next prompt holds the notes where they were"
        assert QTok().decode(t[:end], skip_special_tokens=False).endswith(body + "\n</runtime-note>")
    assert len(fakes["decider"].calls) == asked, "the tool turns asked nothing"


# ------------------------------------------------------------------ 4
def test_a_new_message_and_a_live_skill():
    cond, fakes = lib_cond("on", picks=("browser-game", "pandas-cleaning"))
    s = S()
    served(cond, list(s.msgs))
    msgs = s.step(("bash", {"command": "ls"}, "a")) + [
        {"role": "assistant", "content": "The game runs."}, {"role": "user", "content": NEW_TASK}]
    p, _ = served(cond, msgs)
    lib = p.row["extra"]["library"]
    assert lib["live"] == ["browser-game"], lib
    assert "browser-game" not in [c["name"] for c in lib["candidates"]]
    assert lib_notes(p) == ["pandas-cleaning"] and p.row["replayed"]["count"] == 3
    # the same message again (a client's retry): the pick is the session's, asked once
    n = len(fakes["decider"].calls)
    p2, _ = served(cond, msgs)
    assert len(fakes["decider"].calls) == n
    assert lib_notes(p2) == [], "the note is live: not added twice"


# ------------------------------------------------------------------ 5
def test_after_a_compaction():
    cond, _ = lib_cond("on")
    s = S()
    served(cond, list(s.msgs))
    served(cond, s.step(("bash", {"command": "ls"}, "a")))
    summary = {"model": "TandemLLM", "messages": [{"role": "user", "content": "x" * 3000}]}
    pc = cond.plan(summary, {"x-session-id": "ses-snap"}, client_kind="opencode",
                   request_id="c", chat=True)
    assert pc.compaction and "library" not in pc.row["extra"] and pc.extras == []
    prompt, _, in_think = app.build_prompt(summary)
    pc.apply_prompt(prompt, app.STATE["tok"], in_think)      # the server serves it: notes gone
    after = [{"role": "system", "content": s.msgs[0]["content"]},
             {"role": "user", "content": "Summary: the canvas game exists. Go on with it."}]
    p, _ = served(cond, after)
    assert kinds(p)[-2:] == ["library_card", "library"] and lib_notes(p) == ["browser-game"]


# ------------------------------------------------------------------ 6
def test_an_outcome_retrieves():
    real = outcomes.classify
    cond, fakes = lib_cond("on", picks=("debug-method",),
                           outcomes_={"mode": "shadow"}, section={"class_filter": False})
    s = S()
    p1, _ = served(cond, list(s.msgs))
    assert lib_notes(p1) == [], "the first message: debug-method does not fit, none"
    try:
        outcomes.classify = lambda snap, ev, cfg: ("stuck", ["no file changed and no better "
                                                             "number: bisect the bug"])
        p, _ = served(cond, s.step(("bash", {"command": "npm test"}, "1 failed")))
        lib = p.row["extra"]["library"]
        assert lib["trigger"] == "outcome:stuck" and lib["chosen"] == ["debug-method"], lib
        assert lib_notes(p) == ["debug-method"] and len(p.acts) == 1
        asked = [t for t, o in fakes["decider"].calls if "none" in o][-1]
        assert "bisect the bug" in asked, "the outcome's reasons are in the question's text"
        n = len(fakes["decider"].calls)
        p2, _ = served(cond, s.step(("bash", {"command": "npm test"}, "1 failed")))
        assert len(fakes["decider"].calls) == n and lib_notes(p2) == [], \
            "the same signature: asked once, the note is live"
        outcomes.classify = lambda snap, ev, cfg: ("error", ["exit 1"])
        p3, _ = served(cond, s.step(("bash", {"command": "npm test"}, "boom")))
        assert "library" not in p3.row["extra"], "error is not in the section's outcomes"
    finally:
        outcomes.classify = real


# ------------------------------------------------------------------ 7
def test_none_two_picks_and_no_match():
    cond, _ = lib_cond("on", picks=())
    p, _ = served(cond, list(S().msgs))
    lib = p.row["extra"]["library"]
    assert lib["chosen"] == [] and lib["how"] == "none" and lib_notes(p) == []
    cond, fakes = lib_cond("on", picks=("canvas-2d", "browser-game"),
                           section={"max_picks": 2})
    p, _ = served(cond, list(S().msgs))
    assert lib_notes(p) == ["canvas-2d", "browser-game"], lib_notes(p)
    asked = [o for _, o in fakes["decider"].calls if "none" in o]
    assert len(asked) == 2 and "canvas-2d" not in asked[1], "the second without the first"
    cond, fakes = lib_cond("on")
    b = [{"role": "system", "content": S().msgs[0]["content"]},
         {"role": "user", "content": "Write a haiku about autumn leaves."}]
    p, _ = served(cond, b)
    assert p.row["extra"]["library"]["how"] == "no_match"
    assert not [o for _, o in fakes["decider"].calls if "none" in o], "no question"


# ------------------------------------------------------------------ 8
def test_locked_asks_no_qwen():
    cond, fakes = lib_cond("on")
    _, router = cond.store.get("router")
    v, doc = cond.store.get("packages")
    cfg = schema.library(router)
    cond.router.index.ensure(v, doc, cfg)
    hits = cond.router.index.search("canvas game in the browser", 4)

    class P:
        locked = True
    qwen_only = copy.deepcopy(router)
    for st in qwen_only["classifier"]["stages"]:
        st["enabled"] = st["kind"] == "qwen"
    _, roles = cond.store.get("roles")
    picks, how, asked = libsearch._ask(cond.router, P(), hits, "canvas game", None, cfg,
                                       qwen_only, roles)
    assert (picks, how, asked) == ([], "locked", []) and not fakes["qwen"].calls
    picks, how, _ = libsearch._ask(cond.router, P(), hits, "canvas game", None, cfg, router,
                                   roles)
    assert picks == ["browser-game"] and how == "decider" and not fakes["qwen"].calls
    P.locked = False
    fakes["decider"].fn = lambda t, o: "decider_unreachable"
    picks, how, _ = libsearch._ask(cond.router, P(), hits, "canvas game", None, cfg, router,
                                   roles)
    assert len(fakes["qwen"].calls) == 1 and (picks, how) == ([], "no_answer")


# ------------------------------------------------------------------ 9
def test_shadow_profile_test_box_and_schema():
    cond, _ = lib_cond("on")
    _, p = cond.store.get("profiles", "coding")
    p["mode"] = "shadow"
    cond.store.put("profiles", "coding", p, note="shadow", author="t")
    p1, _ = served(cond, list(S().msgs))
    assert p1.row["extra"]["library"]["mode"] == "shadow" and lib_notes(p1) == []
    cond, _ = lib_cond("on")
    out = cond.router.explain("Build a canvas platformer game for the browser.", "coding")
    assert out["extra"]["library"]["chosen"] == ["browser-game"], out["extra"]
    # the schema
    prof = schema.seed("profiles", "coding")
    assert "library" not in prof and "library" not in schema.seed("router")
    assert schema.check("profiles", dict(prof, library="shadow"), name="coding") == []
    assert schema.check("profiles", dict(prof, library="maybe"), name="coding")
    r = schema.seed("router")
    assert schema.check("router", dict(r, library={"max_picks": 2, "candidates": 5})) == []
    assert schema.check("router", dict(r, library={"max_picks": 3}))
    assert schema.check("router", dict(r, library={"outcomes": ["bored"]}))
    assert schema.library({})["candidates"] == 4 and schema.profile_library({}) == "off"


if __name__ == "__main__":
    app.STATE.setdefault("tok", QTok())
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
