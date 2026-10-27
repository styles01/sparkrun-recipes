"""Evaluation at xhigh and retrieval (J1 steps 6 and 8, the plan of 2026-10-02, sections 1.2
and 2): the snapshot's evidence goes to Qwen at xhigh at three evaluation points, and a low-quality
or stuck outcome retrieves the package that fits the work.

    E-a  after a write and a run whose outcome is not progress: the outcome's note with the digest
         (step 3, tests/test_outcomes.py)
    E-b  a done claim the claim check does not let through: its note carries the digest
    E-c  stuck (no write and no better number) or low quality (a gate far off, failing again and
         again): xhigh and the re-plan note with the digest

The test matrix:

  1  evaluate off (absent): stuck and low quality are journaled only, no effort, no note
                                                        test_off_journals_only
  2  E-c in act: xhigh and the re-plan note once per signature; in shadow both are journaled
     as what would have happened                        test_e_c
  3  retrieval: `packages.fit` over the kind skills picks one (the router's package question,
     the decider then Qwen, no word list), its text is the low-quality note, asked once per
     signature; none fits: no note; shadow journals the choice; with evaluate on too, the
     re-plan note and the package together           test_retrieval
  4  E-b: the claim check's note with the turn's digest in act; in shadow the note is the
     policy's and the journal says the digest's size; without it the note is as before
                                                        test_e_b
  5  Qwen in place on a narrowed case (shadow): asked through the in-context stage's forward
     with the session's notes replayed, journaled beside the twin; off by default; a failure is
     the row's error                                    test_in_place_question

Run: python tests/test_evaluate.py
"""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import QTok  # noqa: E402
from test_snapshots import S, fresh, plan  # noqa: E402

from server import app  # noqa: E402
from server.conductor import schema  # noqa: E402


def bash(cmd, out):
    return ("bash", {"command": cmd}, out)


E = ("edit", {"filePath": "/work/game/game.js", "oldString": "1", "newString": "2"},
     "Edit applied successfully.")
CK = "calccheck src/c.py\nRESULT: FAIL <- (a) error {} > 1.0"


def rows_of(cond):
    got = []
    real = cond.journal.note

    def note(kind, task, **kw):
        got.append(dict(kw, kind=kind, task=task))
        return real(kind, task, **kw)
    cond.journal.note = note
    return got


def medium(cond):
    _, doc = cond.store.get("profiles", "coding")
    doc["routing"]["code_edit"] = {"effort": "medium"}
    cond.store.put("profiles", "coding", doc, note="medium", author="t")


def router_out(cond, **over):
    _, r = cond.store.get("router")
    r["outcomes"] = over
    cond.store.put("router", None, r, note="outcomes", author="t")


def stuck_session(cond, n=5):
    """A session that runs `ls` with no write: stuck at `n` tool turns (router's stuck.turns)."""
    router_out(cond, stuck={"turns": n, "minutes": 999})
    s = S()
    plan(cond, s.msgs)
    return s, [plan(cond, s.step(bash("ls", f"{i}"))) for i in range(n + 1)]


def test_off_journals_only():
    cond = fresh({"mode": "act"})
    medium(cond)
    rows = rows_of(cond)
    _, ps = stuck_session(cond)
    p = ps[4]
    assert p.row["extra"]["outcome"]["label"] == "stuck" and p.row["effort"] == "medium"
    assert p.extras == [] and p.row["extra"]["outcome"]["note"] is None
    assert [r["decision"] for r in rows if r["kind"] == "outcome"][4] == "stuck"


def test_e_c():
    cond = fresh({"mode": "act", "evaluate": "act"})
    medium(cond)
    _, ps = stuck_session(cond)
    p = ps[4]
    o = p.row["extra"]["outcome"]
    assert o["label"] == "stuck" and p.row["effort"] == "xhigh" and o["effort"]["applied"]
    note = [n for n in p.injected if n["parts"][0]["kind"] == "outcome"] if p.injected else []
    assert not note, "plan only: the note is an extra until the prompt is applied"
    ex = [x for x in p.extras if x["parts"][0]["name"] == "stuck"]
    assert len(ex) == 1 and "Re-plan in 3 lines" in ex[0]["text"] and \
        "Runtime snapshot of your last tool turn" in ex[0]["text"]
    assert ps[5].row["effort"] == "xhigh", "held one more turn"
    assert [x for x in ps[5].extras if x["parts"][0]["kind"] == "outcome"] == [], \
        "stuck is noted once until a progress"
    # low quality: a gate far off -> E-c too
    s = S()
    plan(cond, s.msgs, sid="lq")
    p = plan(cond, s.step(bash("python check.py", CK.format(3.0))), sid="lq")
    assert p.row["extra"]["outcome"]["label"] == "low_quality" and p.row["effort"] == "xhigh"
    assert "Re-plan in 3 lines" in p.extras[-1]["text"]
    # shadow: journaled as what would have been done
    cond = fresh({"mode": "act", "evaluate": "shadow"})
    medium(cond)
    rows = rows_of(cond)
    _, ps = stuck_session(cond)
    p = ps[4]
    assert p.row["effort"] == "medium" and p.row["shadow"]["outcome"]["effort"] == "xhigh"
    assert p.extras == [] and p.row["shadow"]["outcome"]["note"] == "stuck"
    r = [x for x in rows if x["kind"] == "outcome"][4]
    assert r["detail"]["note"]["mode"] == "shadow" and not r["detail"]["note"]["applied"]


def packages_on(cond):
    _, doc = cond.store.get("packages")
    for name, desc in (("test-web", "Testing a web app or a game in the browser."),
                       ("test-cli", "Testing a command-line program on real input.")):
        doc["packages"][name] = {"title": name, "kind": "skill", "source": "inline",
                                 "description": desc, "body": f"# {name}\n1. run it.",
                                 "max_tokens": 600, "enabled": True}
    cond.store.put("packages", None, doc, note="kinds", author="t")


def test_retrieval():
    for retrieve, evaluate in (("on", "off"), ("shadow", "off"), ("on", "act")):
        cond, fakes = None, None
        from test_conductor_router import setup
        cond, fakes = setup(decider=lambda t, o: ("test-cli", 0.93) if "test-cli" in o
                            else ("code_edit", 0.91))
        _, doc = cond.store.get("profiles", "coding")
        doc["outcomes"] = {"mode": "shadow", "retrieve": retrieve, "evaluate": evaluate}
        cond.store.put("profiles", "coding", doc, note="retrieve", author="t")
        packages_on(cond)
        s = S()
        plan(cond, s.msgs)
        p = plan(cond, s.step(bash("python check.py", CK.format(3.0))))
        o = p.row["extra"]["outcome"]
        assert o["label"] == "low_quality", o
        asked = [c for c in fakes["decider"].calls if "test-cli" in c[1]]
        assert len(asked) == 1 and set(asked[0][1]) == {"test-web", "test-cli", "none"}
        assert o["note"]["package"] == "test-cli", o
        if retrieve == "on":
            text = p.extras[-1]["text"]
            assert "# test-cli\n1. run it." in text
            assert ("Re-plan in 3 lines" in text) == (evaluate == "act"), text
        else:
            assert p.extras == [] and o["note"]["applied"] is False
        # once per signature: the same gate again asks nothing more
        p = plan(cond, s.step(bash("python check.py", CK.format(3.0))))
        assert len([c for c in fakes["decider"].calls if "test-cli" in c[1]]) == 1
    # none fits: no note
    from test_conductor_router import setup
    cond, fakes = setup(decider=lambda t, o: ("none", 0.95) if "test-cli" in o
                        else ("code_edit", 0.91))
    _, doc = cond.store.get("profiles", "coding")
    doc["outcomes"] = {"mode": "shadow", "retrieve": "on"}
    cond.store.put("profiles", "coding", doc, note="retrieve", author="t")
    packages_on(cond)
    s = S()
    plan(cond, s.msgs)
    p = plan(cond, s.step(bash("python check.py", CK.format(3.0))))
    assert p.extras == [] and p.row["extra"]["outcome"]["note"]["skipped"] == "no_text"


def test_e_b():
    from test_claim_check import E2, _asker, _begin, _finish, _monitor
    digest = "Runtime snapshot of your last tool turn (outcome: none).\n- calls: bash"
    for mode in (None, "shadow", "act"):
        mon = _monitor()
        sup = _begin(mon)
        if mode:
            sup.evidence = (mode, digest)
        assert _finish(sup) is not None
        act = sup.settle_claim(_asker())
        want = E2["notes"]["test_faked"] + ("\n\n" + digest if mode == "act" else "")
        assert act.text == want, (mode, act.text)
        d = mon.journal.actions()[-1]["detail"]
        assert (d.get("digest") or {}).get("chars") == (len(digest) if mode else None)
        assert d["evidence"]["label"] == "test_faked", "Q2's answer keeps its key"


def test_in_place_question():
    cond = fresh({"mode": "shadow", "evaluate": "shadow"})
    s = S()
    plan(cond, s.msgs)
    p = plan(cond, s.step(bash("node run.js", "TypeError: y")))
    assert p.ask is None, "off by default: it costs a prefill on every narrowed turn"
    router_out(cond, evaluate={"ask": {"mode": "shadow"}})
    rows = rows_of(cond)
    p = plan(cond, s.step(bash("node run.js", "TypeError: z")))
    assert p.ask == {"twin": "failure", "call": "bash"}, p.ask
    seen = []

    def runner(ids, letters, at):
        seen.append((ids, letters, at))
        return [math.log(0.8), math.log(0.15), math.log(0.05)], {"forwarded": 40, "reused": 900}
    tok = app.STATE["tok"]
    from server.conductor import incontext
    client = app.build_prompt({"model": "TandemLLM", "messages": s.msgs})[0].tolist()
    row = cond.router.ask_in_place(p, client, tok, runner)
    assert row["decision"] == "failure" and row["detail"]["agree"] is True
    assert row["detail"]["prefilled"] == 40 and p.ask is None
    ids, letters, at = seen[0]
    want, _ = incontext.prompt(p.replayed_ids(client, tok), tok,
                               schema.OUTCOMES["evaluate"]["ask"]["question"] + "\n\n"
                               + incontext.listing(schema.OUTCOMES["evaluate"]["ask"]["options"])[0])
    assert ids == want and len(letters) == 3
    assert p.row["extra"]["outcome"]["in_place"]["label"] == "failure"
    assert [r for r in rows if r["task"] == "in_place"]
    p = plan(cond, s.step(bash("node run.js", "TypeError: w")))
    row = cond.router.ask_in_place(p, client, tok, lambda *a: (_ for _ in ()).throw(
        RuntimeError("engine")))
    assert row["decision"] == "unknown" and row["detail"]["error"] == "RuntimeError"
    p = plan(cond, s.step(bash("node run.js", "fine")))
    row = cond.router.ask_in_place(
        p, client, tok, lambda ids, l, at: ([math.log(0.4), math.log(0.35), math.log(0.25)], {}))
    assert row["decision"] == "unknown" and row["detail"]["confidence"] == 0.4, "below accept"


if __name__ == "__main__":
    app.STATE.setdefault("tok", QTok())
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
