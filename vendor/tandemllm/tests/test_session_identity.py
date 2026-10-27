"""Which session an out-of-band call belongs to, and a key that survives a compaction
(server/session_tokens.py).

  1  the registry: one token a key, one key a token, 12 random characters; kept across a restart in
     a 0600 file; gone 48 h after its last use
  2  the router: `{tandem_url}` in a session's notes is the request's address plus `/s/<token>`;
     two sessions planned at the same time in two threads each get their own, never the other's
  3  the handler: `/s/<token>/v1/library/...` is the library route with the session named in its
     journal row and counts, its read's `{tandem_url}` carries the token again; an unknown token is
     served and attributed to nobody; a session address of any other route is a 404
  4  a client without a session id (`h:` key): after a compaction the engine wrote, the next head
     that carries the summary is the same session; an unrelated head is its own

Run: python tests/test_session_identity.py
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_library_api import PROXY, call, with_library  # noqa: E402  (first: the CPU env)

from server import app, session_tokens as st  # noqa: E402
from server.conductor import packages as packages_mod  # noqa: E402

DOC = {"packages": {"x": {"source": "inline",
                          "body": 'curl -s "${TANDEM_URL:-{tandem_url}}/v1/laya/ask"'}}}


# ------------------------------------------------------------------ 1. the registry

def test_one_token_a_key_kept_across_a_restart_and_expiring():
    with tempfile.TemporaryDirectory() as d:
        t = [1000.0]
        p = os.path.join(d, "tok.json")
        r = st.SessionTokens(p, clock=lambda: t[0])
        a, b = r.token_for("c:A"), r.token_for("c:B")
        assert a != b and r.token_for("c:A") == a and len(a) == 12 and st.TOKEN_RE.match(a)
        assert r.key_for(a) == "c:A" and r.key_for(b) == "c:B" and r.key_for("nope") is None
        assert "c:A" not in a and stat.S_IMODE(os.stat(p).st_mode) == 0o600
        r.close()
        r2 = st.SessionTokens(p, clock=lambda: t[0])
        assert r2.key_for(a) == "c:A" and r2.token_for("c:A") == a
        t[0] += st.TTL_S + 1
        assert r2.key_for(a) is None
        r3 = st.SessionTokens(p, clock=lambda: t[0])
        assert r3.key_for(b) is None and r3.stats["expired"] == 2


def test_strip_and_with_token():
    assert st.strip("/s/Ab3xYz09_-Qq/v1/library/x?y=1") == ("/v1/library/x?y=1", "Ab3xYz09_-Qq")
    assert st.strip("/v1/library") == ("/v1/library", None)
    assert st.strip("/s/../v1/library") == ("/s/../v1/library", None)
    assert st.with_token("http://h:8000/", "T0k3nT0k3nAA") == "http://h:8000/s/T0k3nT0k3nAA"
    assert st.with_token("http://h:8000", None) == "http://h:8000"


# ------------------------------------------------------------------ 2. the router

def _plan(cond, sid, user="Write a parser for the config file.", headers=None):
    b = {"model": "TandemLLM", "messages": [{"role": "system", "content": "sys"},
                                            {"role": "user", "content": user}]}
    h = dict(headers or {"Host": "box:8000"}, **({"x-session-id": sid} if sid else {}))
    plan = cond.plan(b, h, client_kind="opencode", request_id="r", chat=True)
    return plan, cond.router.library.text(DOC, "x")[0]


def test_each_session_its_own_address_also_in_parallel():
    cond, _, _ = with_library()
    reg = cond.router.session_tokens = app.STATE["session_tokens"] = st.SessionTokens(None)
    _, ta = _plan(cond, "A")
    ka = reg.token_for("c:A")
    assert ta == f'curl -s "${{TANDEM_URL:-http://box:8000/s/{ka}}}/v1/laya/ask"', ta
    out, gate = {}, threading.Barrier(2)

    def run(sid):
        gate.wait()
        for _ in range(30):
            _, text = _plan(cond, sid)
            out.setdefault(sid, set()).add(text)
    th = [threading.Thread(target=run, args=(s,)) for s in ("P", "Q")]
    [t.start() for t in th]
    [t.join() for t in th]
    assert out["P"] == {ta.replace(ka, reg.token_for("c:P"))}, out["P"]
    assert out["Q"] == {ta.replace(ka, reg.token_for("c:Q"))}, out["Q"]


# ------------------------------------------------------------------ 3. the handler

def test_a_session_address_names_the_session_in_the_journal_and_counts():
    cond, _, rows = with_library()
    reg = cond.router.session_tokens = app.STATE["session_tokens"] = st.SessionTokens(None)
    tok = reg.token_for("c:ses-1")
    code, _, out = call("GET", f"/s/{tok}/v1/library/browser-game", PROXY)
    assert code == 200 and out["name"] == "browser-game"
    assert rows[-1]["detail"]["session_key"] == "c:ses-1"
    assert reg.calls("c:ses-1") == {"library": 1}
    code, _, out = call("GET", "/s/UnknownToken1/v1/library/search?q=game", PROXY)
    assert code == 200 and rows[-1]["detail"]["session_key"] is None
    assert reg.stats["unknown"] == 1
    code, _, _ = call("GET", "/v1/library", PROXY)
    assert code == 200 and "session_key" not in rows[-1]["detail"]
    code, _, _ = call("GET", f"/s/{tok}/v1/models", PROXY)
    assert code == 404
    code, _, _ = call("POST", f"/s/{tok}/v1/chat/completions", PROXY, {"messages": []})
    assert code == 404
    # a read's `{tandem_url}` keeps the session address
    with_url = packages_mod.with_url("{tandem_url}/v1/laya/ask", "http://x:1/s/" + tok)
    assert with_url.endswith(f"/s/{tok}/v1/laya/ask")


# ------------------------------------------------------------------ 4. a key through a compaction

def test_a_head_that_carries_the_engines_summary_is_the_same_session():
    cond, _, _ = with_library()
    p1, _ = _plan(cond, None, user="Build me a todo app in plain HTML with local storage.")
    k1 = p1.session.key
    assert k1.startswith("h:")
    summary = ("## Objective\n- Build a todo app in plain HTML that keeps its items in local "
               "storage.\n\n## Important Details\n- (none)")
    cond.router.compacted_into(k1, summary)
    p2, _ = _plan(cond, None, user="What did we do so far?\n\n" + summary + "\n\ncontinue")
    assert p2.session.key == k1
    p3, _ = _plan(cond, None, user="continue")      # the alias holds for the next turn's head too
    p3b, _ = _plan(cond, None, user="What did we do so far?\n\n" + summary + "\n\ncontinue")
    assert p3b.session.key == k1
    p4, _ = _plan(cond, None, user="An unrelated question about sorting.")
    assert p4.session.key not in (k1,) and p4.session.key.startswith("h:")
    assert p3.session.key.startswith("h:")
    # a session id is never aliased
    p5, _ = _plan(cond, "S", user=summary)
    assert p5.session.key == "c:S"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:80s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
