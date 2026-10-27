"""The package library's routes for the model's own scripts (R1, 2026-10-03): GET /v1/library,
/v1/library/search?q=&k=&class=, /v1/library/{name}, through the server's own handler.

The test matrix:

  1  open as the chat API and the Laya route (L1): through the proxy, without a token, with a
     bearer (ignored) and with a wrong one, every route answers the same; the answers carry
     `Access-Control-Allow-Origin: *`                   test_open_as_the_chat_api
  2  search: [{name, description, score}] best first, `k` (default 5), `class` filters by
     INDEX.md's fit, [] when nothing matches; read: the body the router injects and the
     reference only a read gets; list: every enabled skill; each against its contract file
     (docs/contract/conductor-v1/library*.schema.json)
                                                        test_search_read_and_list
  3  limits, the OpenAI error shape: no q, q over 1,000 characters, k 0, 21 or not a number,
     an unknown class (422); an unknown, disabled or nested name and no library at all (404);
     the 241st call in a minute (429 with Retry-After)  test_limits_and_errors
  4  one journal row a call (`kind: library_api`): the query as a hash and a length, never its
     text; the names answered; the status             test_the_journal_rows
  5  a SKILL.md edited on the box is what the next read answers (the index follows the file)
                                                        test_a_file_change_is_served

Run: python tests/test_library_api.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_auth import ADMIN, PROXY, _setup, bearer, call  # noqa: E402  (first: the CPU env)
from test_libsearch import SKILLS, lib_packages, make_library, skill_md  # noqa: E402

from server import app, library_api  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from tools import contract_check  # noqa: E402

CONTRACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                        "contract", "conductor-v1")


def _ok(obj, name):
    errs = contract_check.validate(obj, contract_check.load(name, CONTRACT), root=CONTRACT)
    assert not errs, (name, errs[:5])


def with_library(install=True):
    _setup()
    library_api.RATE = library_api.Rate()
    cond = Conductor(tempfile.mkdtemp(prefix="cond-"), state=app.STATE, journal=False)
    d = make_library()
    if install:
        _, doc = cond.store.get("packages")
        cond.store.put("packages", None, lib_packages(d, base=doc), note="library", author="t")
        _, r = cond.store.get("router")
        r["library"] = {"index": os.path.join(d, "INDEX.md")}
        cond.store.put("router", None, r, note="library", author="t")
    rows = []
    cond.journal.note = lambda kind, task, **kw: rows.append(dict(kw, kind=kind, task=task))
    app.STATE["conductor"] = cond
    return cond, d, rows


# ------------------------------------------------------------------ 1
def test_open_as_the_chat_api():
    with_library()
    for path in ("/v1/library", "/v1/library/search?q=canvas+game", "/v1/library/browser-game"):
        got = [call("GET", path, h) for h in (PROXY, dict(PROXY, **bearer(ADMIN)),
                                              dict(PROXY, **bearer("x" * 40)), {})]
        assert {c for c, _, _ in got} == {200}, (path, [c for c, _, _ in got])
        assert all(b == got[0][2] for _, _, b in got), "a bearer changes nothing"
        assert "Access-Control-Allow-Origin: *" in got[0][1] and "no-store" in got[0][1]


# ------------------------------------------------------------------ 2
def test_search_read_and_list():
    with_library()
    code, _, out = call("GET", "/v1/library/search?q=platformer+game+with+lives", PROXY)
    assert code == 200 and isinstance(out, list) and out[0]["name"] == "browser-game", out
    assert set(out[0]) == {"name", "description", "score"} and out[0]["score"] > 0
    assert out[0]["description"] == SKILLS["browser-game"][0]
    _ok(out, "library-search")
    assert [o["score"] for o in out] == sorted((o["score"] for o in out), reverse=True)
    _, _, many = call("GET", "/v1/library/search?q=game+canvas+pandas+pytest+git+bug", PROXY)
    assert len(many) == 5, "k defaults to 5"
    _, _, two = call("GET", "/v1/library/search?q=game+canvas+pandas+pytest+git+bug&k=2", PROXY)
    assert [o["name"] for o in two] == [o["name"] for o in many[:2]]
    _, _, dbg = call("GET", "/v1/library/search?q=game+canvas+pandas+pytest+git+bug"
                            "&class=research", PROXY)
    assert [o["name"] for o in dbg] == ["pandas-cleaning"]
    code, _, none = call("GET", "/v1/library/search?q=zebra+quantum", PROXY)
    assert code == 200 and none == []
    code, _, one = call("GET", "/v1/library/debug-method", PROXY)
    assert code == 200 and one["name"] == "debug-method" and one["classes"] == ["debug"]
    assert one["body"] == "# debug-method\n\n" + SKILLS["debug-method"][1]
    assert one["reference"] == "## Reference\n\n" + SKILLS["debug-method"][2]
    _ok(one, "library-skill")
    code, _, lst = call("GET", "/v1/library/", PROXY)
    assert code == 200 and lst["count"] == len(SKILLS) == len(lst["skills"])
    assert lst["skills"][0].keys() == {"name", "description", "classes"}
    assert lst["search"].startswith("GET /v1/library/search") and "none" in lst["auth"]
    _ok(lst, "library")
    _ok(none, "library-search")


# ------------------------------------------------------------------ 3
def _err(path, status, param=None):
    code, head, out = call("GET", path, PROXY)
    assert code == status, (path, code, out)
    assert set(out) == {"error"} and out["error"]["message"] and out["error"]["type"], out
    if param:
        assert out["error"]["param"] == param, out
    return head, out


def test_limits_and_errors():
    cond, d, _ = with_library()
    _err("/v1/library/search", 422, "q")
    _err("/v1/library/search?q=", 422, "q")
    _err("/v1/library/search?q=%20%20", 422, "q")
    _err("/v1/library/search?q=" + "a" * 1001, 422, "q")
    assert call("GET", "/v1/library/search?q=" + "a" * 1000, PROXY)[0] == 200
    for k in ("0", "21", "x", "-1"):
        _err(f"/v1/library/search?q=game&k={k}", 422, "k")
    assert call("GET", "/v1/library/search?q=game&k=20", PROXY)[0] == 200
    _err("/v1/library/search?q=game&class=chat", 422, "class")
    _, out = _err("/v1/library/no-such-skill", 404)
    assert out["error"]["type"] == "not_found" and "search" in out["error"]["message"]
    _err("/v1/library/browser-game/extra", 404)
    _, doc = cond.store.get("packages")
    doc["packages"]["git-workflow"]["enabled"] = False
    cond.store.put("packages", None, doc, note="off", author="t")
    _err("/v1/library/git-workflow", 404)
    _, _, lst = call("GET", "/v1/library", PROXY)
    assert "git-workflow" not in [s["name"] for s in lst["skills"]]
    # the rate: RATE_PER_MIN calls a minute over every caller
    now = [1000.0]
    library_api.RATE = library_api.Rate(clock=lambda: now[0])
    for _ in range(library_api.RATE_PER_MIN):
        assert call("GET", "/v1/library/search?q=game", PROXY)[0] == 200
    head, out = _err("/v1/library/search?q=game", 429)
    assert out["error"]["type"] == "rate_limited" and "Retry-After: " in head
    now[0] += 61
    assert call("GET", "/v1/library/search?q=game", PROXY)[0] == 200
    # no library installed: 404 on every route
    with_library(install=False)
    for path in ("/v1/library", "/v1/library/search?q=game", "/v1/library/browser-game"):
        _err(path, 404)
    # no conductor (a raw server): the route is not there
    app.STATE.pop("conductor")
    assert call("GET", "/v1/library", PROXY)[0] == 404


# ------------------------------------------------------------------ 4
def test_the_journal_rows():
    _, _, rows = with_library()
    secret = "build my private game about ACME-4471"
    call("GET", "/v1/library/search?q=" + secret.replace(" ", "+") + "&k=3", PROXY)
    call("GET", "/v1/library/browser-game", PROXY)
    call("GET", "/v1/library", PROXY)
    call("GET", "/v1/library/search?k=3", PROXY)
    assert [(r["kind"], r["task"], r["decision"]) for r in rows] == [
        ("library_api", "search", "ok"), ("library_api", "read", "ok"),
        ("library_api", "list", "ok"), ("library_api", "search", "invalid_request_error")]
    s = rows[0]["detail"]
    assert s["q_chars"] == len(secret) and len(s["q"]) == 12 and s["k"] == 3
    assert s["names"][0] == "browser-game" and s["status"] == 200 and s["ms"] >= 0
    assert "ACME" not in repr(rows), "never the query's text"
    assert rows[1]["detail"]["names"] == ["browser-game"]
    assert rows[3]["detail"]["status"] == 422 and rows[3]["detail"]["param"] == "q"


# ------------------------------------------------------------------ 5
def test_a_file_change_is_served():
    cond, d, _ = with_library()
    _, _, before = call("GET", "/v1/library/regex", PROXY)
    assert before["error"]["type"] == "not_found"
    p = os.path.join(d, "git-workflow", "SKILL.md")
    with open(p, "w") as f:
        f.write(skill_md("git-workflow", SKILLS["git-workflow"][0], "Rebase before merging.",
                         "git rebase origin/main"))
    os.utime(p, ns=(time.time_ns() + 10**9,) * 2)
    cond.router.index._checked = -1e9            # the second since the last look has passed
    _, _, got = call("GET", "/v1/library/git-workflow", PROXY)
    assert got["body"].endswith("Rebase before merging.") and \
        got["reference"].endswith("git rebase origin/main")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
