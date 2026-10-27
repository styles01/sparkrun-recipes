"""The Packages and Skills screen's reads (G2, 2026-10-03): /v1/conductor/packages/catalog,
/item/{name}, /tokens and /usage, through the server's own handler.

The test matrix:

  1  the catalog: every package with its category (library, card, test_skill, task_skill,
     package), its source and path, the tokens of the text that goes in on the engine's
     tokenizer, the whole text's and the reference's, the limit (a library skill 350, the
     library card 200, every other package its max_tokens) and whether it is over (a cut text
     is), the classes of the router and of each profile that list it, INDEX.md's fit, each
     profile's library switch; against its contract file      test_the_catalog
  2  without the engine's tokenizer the counts are the server's estimate, 3.5 characters a token
                                                             test_the_estimate
  3  one package: the frontmatter of a file, its body, a library skill's reference and the text
     that goes in (the part before `## Reference`), the tokens of each, the BM25 neighbours (not
     itself, best first); an inline package; a file that is not on the box; an unknown name 404
                                                             test_one_package
  4  the tokens of texts (the editor's counter): in order, cached by the text, 400 for no list,
     an empty list, 21 texts or a text over 200,000 characters  test_the_tokens
  5  the journal's last 7 days per package: an injected class package, a library skill, the
     library card and the two cards (the tools card by the row's client), an outcome's retrieved
     package, shadow notes and cards, the library's picks in shadow and on; the last day and the
     week apart; a row older than 7 days is not counted; one package's newest rows; the
     library's retrieval rows and summary; 400 for a bad limit   test_the_usage
  6  reads: with the dashboard login off every route answers without a token through the proxy,
     the tokens POST included, while a PUT still needs one; with it on, the token opens them
                                                             test_reads_not_writes
  7  nothing moves: the documents' versions, the files of the config directory and every
     package's text the router injects are the same after every route was called
                                                             test_nothing_moves

Run: python tests/test_packages_catalog.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_auth import ADMIN, PROXY, _setup, bearer, call  # noqa: E402  (first: the CPU env)
from test_libsearch import SKILLS, lib_packages, make_library, skill_md  # noqa: E402

from server import app  # noqa: E402
from server.conductor import catalog  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from tools import contract_check  # noqa: E402

CONTRACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                        "contract", "conductor-v1")
LONG = "Keep every frame under 16 ms. " * 14          # 420 characters: over a library's 350
CARD = "Search the library with curl before unfamiliar work. " * 5     # 265: over the card's 200
PENCIL = ("---\nname: pencil-sketch\ndescription: Turn a photo into a pencil sketch.\n"
          "license: MIT\n---\n# Pencil sketch\n\nBlur, invert, dodge.\n")


def _ok(obj, name):
    errs = contract_check.validate(obj, contract_check.load(name, CONTRACT), root=CONTRACT)
    assert not errs, (name, errs[:5])


def _pkg(kind, body, **kw):
    return dict({"title": "t", "kind": kind, "source": "inline", "description": "a package here",
                 "body": body, "max_tokens": 3000, "enabled": True}, **kw)


def with_packages(login=True):
    """A conductor with the library (one skill over 350), the cards, a test skill, a general
    package, a task skill on the box, one that is not, and a text the server cuts."""
    _setup(login=login)
    cond = Conductor(tempfile.mkdtemp(prefix="cond-"), state=app.STATE, journal=False)
    d = make_library()
    with open(os.path.join(d, "canvas-2d", "SKILL.md"), "w") as f:
        f.write(skill_md("canvas-2d", SKILLS["canvas-2d"][0], LONG, "ctx.scale(2, 2);"))
    box = tempfile.mkdtemp(prefix="skills-")
    os.makedirs(os.path.join(box, "pencil-sketch"))
    with open(os.path.join(box, "pencil-sketch", "SKILL.md"), "w") as f:
        f.write(PENCIL)
    _, doc = cond.store.get("packages")
    doc = lib_packages(d, base=doc)
    doc["skill_dirs"] = [box, d]
    doc["packages"].update({
        "coding": _pkg("package", "Write small diffs and run the tests.", general=True),
        "tandem-harness": _pkg("package", "The harness card."),
        "tools-opencode": _pkg("package", "opencode's pitfalls."),
        "tandem-library": _pkg("package", CARD, max_tokens=500),
        "test-web": _pkg("skill", "Test a page through input only."),
        "long-notes": _pkg("package", "x" * 100, max_tokens=16)})
    cond.store.put("packages", None, doc, note="g2", author="t")
    _, r = cond.store.get("router")
    r["library"] = {"index": os.path.join(d, "INDEX.md")}
    for c in r["classes"]:
        if c["id"] == "code_edit":
            c["policy"]["packages"] = ["coding", "test-web"]
    cond.store.put("router", None, r, note="g2", author="t")
    _, prof = cond.store.get("profiles", "coding")
    prof["library"] = "shadow"
    prof["routing"] = {"debug": {"packages": ["coding"]}}
    cond.store.put("profiles", "coding", prof, note="g2", author="t")
    app.STATE["conductor"] = cond
    return cond


def get(path, headers=None):
    code, _, out = call("GET", path, dict(PROXY, **bearer(ADMIN)) if headers is None else headers)
    assert code == 200, (path, code, out)
    return out


# ------------------------------------------------------------------ 1
def test_the_catalog():
    with_packages()
    out = get("/v1/conductor/packages/catalog")
    _ok(out, "packages-catalog")
    p = out["packages"]
    assert out["tokenizer"] == "engine" and out["limits"] == {"library": 350, "library_card": 200}
    cats = {n: e["category"] for n, e in p.items()}
    assert cats["browser-game"] == cats["canvas-2d"] == "library"
    assert cats["tandem-harness"] == cats["tools-opencode"] == cats["tandem-library"] == "card"
    assert cats["test-web"] == "test_skill" and cats["pencil-sketch"] == "task_skill"
    assert cats["ui-design"] == "task_skill" and cats["coding"] == cats["long-notes"] == "package"
    # the engine's tokenizer (the test's: one token a character), on the text that goes in
    bg = p["browser-game"]
    body = "# browser-game\n\n" + SKILLS["browser-game"][1]
    assert bg["tokens"] == bg["body_tokens"] == bg["chars"] == len(body), bg
    assert bg["reference_tokens"] == len("## Reference\n\n" + SKILLS["browser-game"][2])
    assert (bg["limit"], bg["limit_source"], bg["over"]) == (350, "house", False)
    assert bg["fits"] == ["code_edit", "debug", "drawing"] and bg["classes"] == []
    assert bg["source"] == "skill:browser-game" and bg["path"].endswith("browser-game/SKILL.md")
    cv = p["canvas-2d"]
    assert cv["over"] and cv["body_tokens"] == len("# canvas-2d\n\n" + LONG.strip()) > 350
    lc = p["tandem-library"]
    assert (lc["limit"], lc["limit_source"], lc["over"]) == (200, "house", True), lc
    ln = p["long-notes"]
    assert ln["cut"] and ln["over"] and ln["limit"] == 16 and ln["limit_source"] == "max_tokens"
    assert ln["body_tokens"] == 100 and ln["tokens"] < 100 + 40, ln
    assert p["coding"]["classes"] == ["code_edit"] and p["coding"]["profiles"] == {
        "coding": ["debug"]}
    assert p["test-web"]["classes"] == ["code_edit"] and p["test-web"]["fits"] is None
    pen = p["pencil-sketch"]
    assert pen["available"] and pen["tokens"] == len("# Pencil sketch\n\nBlur, invert, dodge.")
    assert p["pencil-sketch"]["classes"] == ["drawing"]
    miss = p["photo-to-3d"]
    assert not miss["available"] and miss["error"] == "package_file_missing" and miss["tokens"] == 0
    assert out["library"]["card"] == "tandem-library"
    assert out["library"]["profiles"]["coding"] == "shadow" and \
        out["library"]["profiles"]["chat"] == "off"


# ------------------------------------------------------------------ 2
def test_the_estimate():
    with_packages()
    app.STATE.pop("tok")
    out = get("/v1/conductor/packages/catalog")
    assert out["tokenizer"] == "estimate"
    assert out["packages"]["coding"]["tokens"] == int(len("Write small diffs and run the tests.")
                                                      / 3.5)
    _ok(out, "packages-catalog")


# ------------------------------------------------------------------ 3
def test_one_package():
    with_packages()
    one = get("/v1/conductor/packages/item/debug-method")
    _ok(one, "packages-item")
    assert one["category"] == "library" and one["found"]
    assert one["frontmatter"] == {"name": "debug-method", "description": SKILLS["debug-method"][0]}
    assert one["body"] == "# debug-method\n\n" + SKILLS["debug-method"][1]
    assert one["reference"] == "## Reference\n\n" + SKILLS["debug-method"][2]
    assert one["injected"] == one["body"]
    assert one["tokens"] == {"injected": len(one["body"]), "body": len(one["body"]),
                             "reference": len(one["reference"])}
    names = [n["name"] for n in one["neighbours"]]
    assert "debug-method" not in names and names and len(names) <= catalog.NEIGHBOURS
    scores = [n["score"] for n in one["neighbours"]]
    assert scores == sorted(scores, reverse=True) and scores[0] > 0
    assert set(names) <= set(SKILLS), names
    pen = get("/v1/conductor/packages/item/pencil-sketch")
    assert pen["frontmatter"]["license"] == "MIT" and "license: MIT" in pen["frontmatter_raw"]
    assert pen["body"] == "# Pencil sketch\n\nBlur, invert, dodge." and pen["reference"] == ""
    assert pen["neighbours"] is None
    inl = get("/v1/conductor/packages/item/coding")
    assert inl["path"] is None and inl["frontmatter"] == {} and inl["body"].startswith("Write")
    _ok(inl, "packages-item")
    gone = get("/v1/conductor/packages/item/photo-to-3d")
    assert not gone["found"] and gone["injected"] is None and gone["error"] == \
        "package_file_missing"
    code, _, err = call("GET", "/v1/conductor/packages/item/nope", dict(PROXY, **bearer(ADMIN)))
    assert code == 404 and err["error"]["type"] == "not_found"
    _ok(err, "error")


# ------------------------------------------------------------------ 4
def test_the_tokens():
    with_packages()
    h = dict(PROXY, **bearer(ADMIN))
    code, _, out = call("POST", "/v1/conductor/packages/tokens", h, {"texts": ["abc", "", "é x"]})
    assert code == 200 and out["tokens"] == [3, 0, 3] and out["tokenizer"] == "engine", out
    _ok(out, "packages-tokens")
    before = len(catalog._COUNTS)
    call("POST", "/v1/conductor/packages/tokens", h, {"texts": ["abc"]})
    assert len(catalog._COUNTS) == before, "a text is counted once"
    for bad in ({}, {"texts": []}, {"texts": "abc"}, {"texts": ["a"] * 21},
                {"texts": ["a" * 200_001]}, {"texts": [3]}):
        code, _, err = call("POST", "/v1/conductor/packages/tokens", h, bad)
        assert code == 400 and err["error"]["type"] == "bad_request", (bad, code)
    code, _, _ = call("POST", "/v1/conductor/packages/tokens", h, {"texts": ["a" * 200_000]})
    assert code == 200


# ------------------------------------------------------------------ 5
def _row(cond, ts, **kw):
    return cond.journal.open_row(dict({"ts_ms": ts, "session_key": "c:ses_a",
                                       "class_id": "code_edit", "profile": "coding",
                                       "client_kind": "opencode", "request_id": f"r{ts}",
                                       "injected": None, "shadow": None, "extra": {}}, **kw))


def test_the_usage():
    cond = with_packages()
    now = int(time.time() * 1000)
    hour, day = 3_600_000, 86_400_000
    note = lambda *parts: [{"pos": 10, "tokens": 5, "text": "x", "reason": "r",  # noqa: E731
                            "parts": [{"kind": k, "name": n} for k, n in parts]}]
    _row(cond, now - 9 * day, injected=note(("package", "coding")))            # too old
    _row(cond, now - 3 * day, injected=note(("package", "coding"), ("effort", "high")))
    _row(cond, now - 2 * hour, injected=note(("package", "coding"), ("rules", "code_edit")))
    _row(cond, now - hour, client_kind="openai-sdk",
         injected=note(("card", "harness")) + note(("card", "tools")))
    _row(cond, now - 50 * 60_000, injected=note(("card", "tools"), ("library_card",
                                                                     "tandem-library")))
    _row(cond, now - 40 * 60_000, injected=note(("library", "debug-method")),
         extra={"library": {"mode": "on", "trigger": "message", "query_chars": 120,
                            "candidates": [{"name": "debug-method", "score": 4.2},
                                           {"name": "pytest-practice", "score": 2.0}],
                            "chosen": ["debug-method"], "how": "decider",
                            "applied": ["debug-method"]}})
    _row(cond, now - 30 * 60_000, profile="overnight",
         shadow={"notes": [{"decision": "packages", "kind": "package", "name": "test-web",
                            "chars": 30}], "cards": [{"name": "harness", "chars": 9}],
                 "library_card": {"name": "tandem-library", "chars": 265}},
         extra={"library": {"mode": "shadow", "trigger": "message", "query_chars": 80,
                            "candidates": [{"name": "browser-game", "score": 3.1}],
                            "chosen": ["browser-game"], "how": "qwen", "cached": True}})
    _row(cond, now - 20 * 60_000, injected=note(("outcome", "low_quality")),
         extra={"outcome": {"note": {"package": "test-web", "applied": True}}})
    _row(cond, now - 10 * 60_000, extra={"outcome": {"note": {"package": "test-web"}},
                                         "library": {"mode": "shadow", "trigger": "outcome:stuck",
                                                     "query_chars": 60, "candidates": [],
                                                     "chosen": [], "how": "no_match"}})
    out = get("/v1/conductor/packages/usage?name=test-web&limit=5")
    _ok(out, "packages-usage")
    p = out["packages"]
    span = lambda d, w: {"day": d, "week": w}  # noqa: E731
    assert p["coding"]["injected"] == span(1, 2), p["coding"]
    assert p["tandem-harness"]["injected"] == span(1, 1)
    assert p["tandem-harness"]["shadow"] == span(1, 1)
    assert p["tools-hermes"]["injected"] == span(1, 1), "the tools card by the row's client"
    assert p["tools-opencode"]["injected"] == span(1, 1)
    assert p["tandem-library"]["injected"] == span(1, 1) and \
        p["tandem-library"]["shadow"] == span(1, 1)
    assert p["debug-method"]["injected"] == span(1, 1) and p["debug-method"]["pick"] == span(1, 1)
    assert p["browser-game"]["pick"] == span(1, 1) and p["browser-game"]["injected"] == span(0, 0)
    assert p["test-web"]["injected"] == span(1, 1) and p["test-web"]["shadow"] == span(2, 2)
    assert "effort" not in p and "high" not in p and "code_edit" not in p
    assert [(u["how"], u["profile"]) for u in out["recent"]] == [
        ("shadow", "coding"), ("injected", "coding"), ("shadow", "overnight")], out["recent"]
    assert out["name"] == "test-web" and out["recent"][0]["session_key"] == "c:ses_a"
    lib = out["library"]
    assert lib["searched"] == span(3, 3) and lib["picked"] == span(2, 2)
    assert lib["top"] == [{"name": "browser-game", "picks": 1}, {"name": "debug-method", "picks": 1}]
    assert [r["mode"] for r in lib["rows"]] == ["shadow", "shadow", "on"]
    sh = lib["rows"][1]
    assert sh["chosen"] == ["browser-game"] and sh["cached"] and sh["applied"] is None
    assert sh["candidates"] == [{"name": "browser-game", "score": 3.1}]
    assert lib["rows"][2]["applied"] == ["debug-method"]
    few = get("/v1/conductor/packages/usage?name=test-web&limit=1")
    assert len(few["recent"]) == 1
    assert get("/v1/conductor/packages/usage")["recent"] == []
    for bad in ("0", "201", "x"):
        code, _, _ = call("GET", f"/v1/conductor/packages/usage?limit={bad}",
                          dict(PROXY, **bearer(ADMIN)))
        assert code == 400, bad


# ------------------------------------------------------------------ 6
def test_reads_not_writes():
    with_packages(login=False)
    for path in ("/v1/conductor/packages/catalog", "/v1/conductor/packages/item/coding",
                 "/v1/conductor/packages/usage"):
        assert call("GET", path, PROXY)[0] == 200, path
    assert call("POST", "/v1/conductor/packages/tokens", PROXY, {"texts": ["a"]})[0] == 200
    code, _, _ = call("PUT", "/v1/conductor/packages", PROXY, {"doc": {}, "note": "x"})
    assert code == 401, "a write still needs the token"
    with_packages(login=True)
    assert call("GET", "/v1/conductor/packages/catalog", PROXY)[0] == 401
    assert call("POST", "/v1/conductor/packages/tokens", PROXY, {"texts": ["a"]})[0] == 401
    assert call("GET", "/v1/conductor/packages/catalog", dict(PROXY, **bearer(ADMIN)))[0] == 200


# ------------------------------------------------------------------ 7
def _tree(root):
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _served(cond):
    _, doc = cond.store.get("packages")
    return {n: cond.library.text(doc, n) for n in doc["packages"]}


def test_nothing_moves():
    cond = with_packages()
    before = (_tree(cond.root), [cond.store.get(k)[0] for k in ("packages", "router")],
              _served(cond))
    get("/v1/conductor/packages/catalog")
    for n in ("debug-method", "coding", "pencil-sketch"):
        get(f"/v1/conductor/packages/item/{n}")
    call("POST", "/v1/conductor/packages/tokens", dict(PROXY, **bearer(ADMIN)), {"texts": ["x"]})
    get("/v1/conductor/packages/usage?name=coding")
    after = (_tree(cond.root), [cond.store.get(k)[0] for k in ("packages", "router")],
             _served(cond))
    assert before[0] == after[0], "no file of the config directory changed"
    assert before[1] == after[1] and before[2] == after[2]
    assert before[2]["coding"][0].startswith("Write") and before[2]["photo-to-3d"][0] is None


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
