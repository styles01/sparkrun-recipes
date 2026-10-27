"""SRV-241 / SRV-242: Regenerate gives a different answer; every other request stays greedy.

Unit: the detection (server/regenerate.py) fed the request shapes Open WebUI 0.11.4 sends
(read off its Chat.svelte): Regenerate = the same messages again; Continue ends with the assistant's
partial answer; Edit and a suggestion change the messages; plus a client's own temperature, the
`X-Tandem-Regenerate` header, a re-rendered system prompt, a stopped answer, the fallback session
key, the bounds.

HTTP: the real server on the fake engine with the conductor (tests/test_conductor_e2e.py's
harness): the first answer is greedy, the same conversation again is a regenerate (the journal row
says so, the ledger row's `sampling` is `regenerate`, the seed is fresh each time), a follow-up and a
Continue are greedy, a client temperature is the client's, raw and the bench profile are never
looked at, and the speed sample leaves the regenerate rows out.

Run: python tests/test_regenerate.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from server import regenerate as rg  # noqa: E402

SYS = {"role": "system", "content": "You are helpful."}
U1 = {"role": "user", "content": "Give me 5 name ideas for a small bakery in Berlin."}
A1 = {"role": "assistant", "content": "1. Morgenbrot\n2. Goldkorn"}
OWUI = {"X-OpenWebUI-Chat-Id": "chat-1"}


def _r(**kw):
    return rg.Regenerates(**kw)


def test_owui_regenerate_is_detected_after_an_answer():
    r = _r()
    first = r.decide({"messages": [SYS, U1]}, OWUI)
    assert first.how is None and not first.sample          # a first answer: greedy
    r.answered(first, "req-1")
    again = r.decide({"messages": [SYS, U1]}, OWUI)
    assert again.how == "detected" and again.of == "req-1" and again.sample


def test_not_a_regenerate_before_any_answer():
    r = _r()
    d = r.decide({"messages": [U1]}, OWUI)
    # the first request failed or is still running: no answer recorded
    assert r.decide({"messages": [U1]}, OWUI).how is None and d.how is None


def test_continue_is_not_a_regenerate():
    r = _r()
    r.answered(r.decide({"messages": [SYS, U1]}, OWUI), "req-1")
    cont = r.decide({"messages": [SYS, U1, A1]}, OWUI)          # Continue: assistant last
    assert cont.how is None and cont.sig is None


def test_edit_and_suggestion_and_followup_are_not_regenerates():
    r = _r()
    r.answered(r.decide({"messages": [SYS, U1]}, OWUI), "req-1")
    edit = {"role": "user", "content": "Give me 5 name ideas for a flower shop in Hamburg."}
    assert r.decide({"messages": [SYS, edit]}, OWUI).how is None
    sugg = {"role": "user", "content": "Add details"}
    assert r.decide({"messages": [SYS, U1, A1, sugg]}, OWUI).how is None
    follow = {"role": "user", "content": "Which one is best?"}
    assert r.decide({"messages": [SYS, U1, A1, follow]}, OWUI).how is None


def test_regenerate_of_an_earlier_answer():
    r = _r()
    r.answered(r.decide({"messages": [U1]}, OWUI), "req-1")
    follow = {"role": "user", "content": "Which one is best?"}
    r.answered(r.decide({"messages": [U1, A1, follow]}, OWUI), "req-2")
    # Open WebUI regenerates any answer: the first one's conversation again
    d = r.decide({"messages": [U1]}, OWUI)
    assert d.how == "detected" and d.of == "req-1"


def test_a_regenerate_of_a_regenerate():
    r = _r()
    r.answered(r.decide({"messages": [U1]}, OWUI), "req-1")
    d2 = r.decide({"messages": [U1]}, OWUI)
    r.answered(d2, "req-2")
    d3 = r.decide({"messages": [U1]}, OWUI)
    assert d3.how == "detected" and d3.of == "req-2"


def test_client_sampling_is_kept():
    r = _r()
    r.answered(r.decide({"messages": [U1]}, OWUI), "req-1")
    for extra in ({"temperature": 0.3}, {"temperature": 0}, {"seed": 7}, {"top_p": 0.9}):
        d = r.decide({"messages": [U1], **extra}, OWUI)
        assert d.how == "detected" and d.client and not d.sample, extra
    # a null temperature is no temperature (Open WebUI with empty params sends none at all)
    assert r.decide({"messages": [U1], "temperature": None}, OWUI).sample


def test_the_header():
    r = _r()
    d = r.decide({"messages": [U1]}, {"X-Tandem-Regenerate": "1"})
    assert d.how == "header" and d.sample
    r.answered(r.decide({"messages": [U1]}, OWUI), "req-1")
    off = r.decide({"messages": [U1]}, {**OWUI, "X-Tandem-Regenerate": "0"})
    assert off.how is None and off.sig is None
    r.answered(off, "req-x")              # an opted-out request is never remembered
    assert r.report()["answered"] == 1


def test_sessions_are_separate():
    r = _r()
    r.answered(r.decide({"messages": [U1]}, OWUI), "req-1")
    assert r.decide({"messages": [U1]}, {"X-OpenWebUI-Chat-Id": "chat-2"}).how is None
    assert r.decide({"messages": [U1]}, {"x-session-id": "ses_1"}).how is None


def test_fallback_session_without_an_id():
    r = _r()
    r.answered(r.decide({"messages": [U1]}, {}), "req-1")
    assert r.decide({"messages": [U1]}, {}).how == "detected"
    other = {"role": "user", "content": "Something else"}
    assert r.decide({"messages": [other]}, {}).how is None


def test_system_prompt_and_reasoning_do_not_matter():
    r = _r()
    sys1 = {"role": "system", "content": "Now is 22:50."}
    sys2 = {"role": "system", "content": "Now is 22:51."}
    a_think = dict(A1, reasoning_content="thinking ...")
    follow = {"role": "user", "content": "Which one is best?"}
    r.answered(r.decide({"messages": [sys1, U1, a_think, follow]}, OWUI), "req-1")
    d = r.decide({"messages": [sys2, U1, A1, follow]}, OWUI)
    assert d.how == "detected", "a time variable or a dropped reasoning is the same conversation"


def test_images_and_tool_turns():
    r = _r()
    img = {"role": "user", "content": [{"type": "text", "text": "what is this"},
                                       {"type": "image_url", "image_url": {"url": "data:x"}}]}
    r.answered(r.decide({"messages": [img]}, OWUI), "req-1")
    assert r.decide({"messages": [img]}, OWUI).how == "detected"
    tool = {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    assert rg.signature([U1, A1, tool]) is None           # a tool round is never a repeat


def test_bounds_and_ttl():
    now = [1000.0]
    r = _r(per=2, sessions=2, ttl_s=60, clock=lambda: now[0])
    msgs = [{"role": "user", "content": f"q{i}"} for i in range(3)]
    for i, m in enumerate(msgs):
        r.answered(r.decide({"messages": [m]}, OWUI), f"req-{i}")
    assert r.decide({"messages": [msgs[0]]}, OWUI).how is None      # pushed out (per=2)
    assert r.decide({"messages": [msgs[2]]}, OWUI).how == "detected"
    now[0] += 61
    assert r.decide({"messages": [msgs[2]]}, OWUI).how is None      # too old
    for c in ("a", "b", "c"):
        r.answered(r.decide({"messages": [U1]}, {"X-OpenWebUI-Chat-Id": c}), c)
    assert r.report()["sessions"] == 2


def test_sampling_is_fresh_and_in_range():
    r = _r()
    a, b = r.sampling(), r.sampling()
    assert a["seed"] != b["seed"] and (a["temperature"], a["top_p"], a["top_k"]) == (0.7, 0.95, 20)
    for bad in ({"temperature": 0}, {"temperature": 3}, {"top_p": 0}, {"top_k": -1},
                {"mode": "maybe"}):
        try:
            _r(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert _r(mode="off").decide({"messages": [U1]}, OWUI) is None


def test_answered_ok():
    assert rg.answered_ok(200, "stop", 12) and rg.answered_ok(200, "length", 5)
    assert rg.answered_ok(200, "abandoned", 3)            # Stop, then Regenerate
    assert not rg.answered_ok(200, "error", 3) and not rg.answered_ok(500, None, 3)
    assert not rg.answered_ok(200, "stop", 0) and not rg.answered_ok(429, "refused", 0)


# ------------------------------------------------------------------ over HTTP, the fake engine

def _ledger_rows(path, n, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            c = sqlite3.connect(path)
            rows = c.execute("SELECT request_id, sampling, client_kind FROM requests "
                             "WHERE client_kind != 'internal' ORDER BY id").fetchall()
            c.close()
            if len(rows) >= n:
                return rows
        except sqlite3.Error:
            pass
        time.sleep(0.3)
    raise AssertionError(f"ledger has fewer than {n} rows")


def test_regenerate_over_http():
    import test_conductor_e2e as h
    port, arb_port = h._port(), h._port()
    cdir = tempfile.mkdtemp(prefix="qse-regen-")
    arb = h._decider(arb_port)
    srv, ledger = h._server(port, cdir)
    try:
        h._wait(arb_port, arb, "/readyz")
        h._wait(port, srv)
        st, _, roles = h._call(port, "GET", "/v1/conductor/roles")
        doc = roles["doc"]
        doc["decider"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        doc["classifier"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        assert h._call(port, "PUT", "/v1/conductor/roles", {"doc": doc, "note": "fake"})[0] == 200
        ow = {"X-OpenWebUI-Chat-Id": "regen-chat", "User-Agent": "Open WebUI"}
        msgs = [U1]

        def ask(m, headers=ow, **extra):
            body = {"model": "TandemLLM", "messages": m, "max_tokens": 64, **extra}
            st, hd, out = h._call(port, "POST", "/v1/chat/completions", body,
                                  headers=headers, token=None)
            assert st == 200, (st, out)
            return out

        first = ask(msgs)                                   # 1 greedy
        regen1 = ask(msgs)                                  # 2 regenerate
        regen2 = ask(msgs)                                  # 3 regenerate again
        answer = {"role": "assistant",
                  "content": regen2["choices"][0]["message"].get("content") or "x"}
        ask(msgs + [answer])                                # 4 Continue: greedy
        ask(msgs + [answer, {"role": "user", "content": "Which is best?"}])   # 5 follow-up
        ask(msgs, temperature=0.2)                          # 6 the client's own temperature
        ask(msgs, headers={**ow, "X-Tandem": "raw"})        # 7 raw: never looked at
        ask(msgs, headers={**ow, "X-Tandem-Profile": "bench"})               # 8 bench
        ask([{"role": "user", "content": "fresh"}],
            headers={"X-Tandem-Regenerate": "1"})           # 9 the header
        rows = _ledger_rows(ledger, 9)
        got = [r[1] for r in rows[:9]]
        assert got == [None, "regenerate", "regenerate", None, None, "client", None, None,
                       "regenerate"], got
        ids = [r[0] for r in rows[:9]]
        assert ids[0] == first["id"] and ids[1] == regen1["id"]
        st, _, hl = h._call(port, "GET", "/health", token=None)
        rep = hl["sampling"]["regenerate"]
        assert (rep["mode"], rep["detected"], rep["header"], rep["client_params"]) == \
            ("sample", 2, 1, 1), rep
        # the journal row names the regenerate, the answer it replaces and a fresh seed each time
        seeds = []
        for rid, of in ((ids[1], ids[0]), (ids[2], ids[1])):
            dec = h._decisions(port, request_id=rid)
            row = [d for d in dec if d.get("request_id") == rid]
            if not row:
                dec = h._decisions(port, limit=50)
                row = [d for d in dec if d.get("request_id") == rid]
            ex = (row[0].get("extra") or {}).get("regenerate")
            assert ex and ex["how"] == "detected" and ex["of"] == of and ex["temperature"] == 0.7, \
                row[0]
            seeds.append(ex["seed"])
        assert seeds[0] != seeds[1]
        # the speed sample leaves the regenerate rows out
        c = sqlite3.connect(ledger)
        from server import dashboard_api
        n_all = c.execute("SELECT COUNT(*) FROM requests WHERE status = 200 AND "
                          "client_kind != 'internal'").fetchone()[0]
        n_speed = c.execute(f"SELECT COUNT(*) FROM requests WHERE client_kind != 'internal' AND "
                            f"sampling = 'regenerate' AND {dashboard_api.DECODED}").fetchone()[0]
        c.close()
        assert n_all >= 9 and n_speed == 0
    finally:
        for p in (srv, arb):
            p.terminate()
            try:
                p.wait(10)
            except Exception:                                      # noqa: BLE001
                p.kill()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:                               # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, exc)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
