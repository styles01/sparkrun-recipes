"""SRV-475 (QA SRV-476): the loop escape. A session that repeats one call samples with the model
card's settings until it makes progress; client sampling wins; `sampling.agentic_default: card`
samples every agent turn; a sampled turn decodes on the engine's sampled verification path.

The test matrix is SRV-475's Gherkin:

  1  a repeat switches the next turn to the card's sampling         test_1_* (unit + HTTP)
  2  a near-repeat switches too                                    test_2_*
  3  progress returns the session to greedy (journaled)            test_3_* (unit + HTTP)
  4  different calls with the same result are progress            test_4_*
  5  the client's sampling wins                                    test_5_* (HTTP)
  6  a profile samples every agent turn (agentic_default card)     test_6_* (HTTP)
  7  a sampled turn uses the sampled verification path             test_7_*
  8  raw, side, no tools: never looked at                          test_8_* (HTTP)

plus the card read (generation_config.json), the fallback and the profile schema. The HTTP rows
run the real server on the fake engine with the conductor and the shipped profiles
(tests/test_conductor_e2e.py's harness), opencode's user agent, one session id.

Run: python tests/test_loop_escape.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from server.conductor import loopescape as le, schema  # noqa: E402

READ_A = {"filePath": "src/a.py"}
USER = {"role": "user", "content": "make the parser handle quoted From lines"}


def turn(*calls, start=0) -> list:
    out = []
    for i, (name, args, result) in enumerate(calls, start):
        out.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]})
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": result})
    return out


def edit(old: str) -> dict:
    return {"filePath": "src/gmail_archive/parser.py", "oldString": old,
            "newString": "    return line[1:] if line.startswith('>From ') else line"}


# ---------------------------------------------------------------------- unit
def test_1_a_repeat_is_the_signal():
    msgs = [USER] + turn(("read", READ_A, "1: import re"), ("read", READ_A, "1: import re"))
    assert le.signal(msgs) == {"why": "repeat", "name": "read", "count": 2}
    assert le.signal(msgs[:-2]) is None, "one run is no repeat"
    assert le.signal(msgs, repeats=4) is None, "C3's repeats decides the length"
    assert le.turn_of(msgs) == 3


def test_2_a_near_repeat_is_the_signal():
    err = "Error: oldString not found in content"
    msgs = [USER] + turn(("edit", edit("    if line.startswith('>From '):"), err),
                         ("edit", edit("    if line.startswith('>From'):"), err),
                         ("edit", edit("    if line.startswith(\">From \"):"), err))
    assert le.signal(msgs) == {"why": "near_repeat", "name": "edit", "count": 3}
    # two are not enough for a near-repeat
    assert le.signal(msgs[:-2]) is None
    # a result that changed breaks it
    msgs2 = msgs[:-1] + [{"role": "tool", "tool_call_id": "c2", "content": "Edit applied."}]
    assert le.signal(msgs2) is None


def test_3_progress_ends_it_and_the_record_says_so():
    tr = le.Escapes()
    sig = {"why": "repeat", "name": "read", "count": 2}
    a = tr.step("s", sig, 7)
    assert (a["state"], a["since_turn"], a["turn"], a["nth"]) == ("on", 7, 7, 0)
    b = tr.step("s", sig, 8)
    assert (b["state"], b["since_turn"], b["turn"], b["nth"]) == ("on", 7, 8, 1)
    c = tr.step("s", None, 9)
    assert c == {"state": "back", "why": "repeat", "name": "read", "since_turn": 7,
                 "back_at_turn": 9, "sampled": 2}
    assert tr.step("s", None, 10) is None
    assert tr.report() == {"sessions": 0, "started": 1, "sampled": 2, "back": 1}
    # sessions are separate and bounded
    small = le.Escapes(sessions=2)
    for k in "abc":
        small.step(k, sig, 1)
    assert small.report()["sessions"] == 2
    # the new call after the repeat: no signal on that request
    msgs = [USER] + turn(("read", READ_A, "x"), ("read", READ_A, "x"),
                         ("bash", {"command": "pytest -q"}, "1 failed"))
    assert le.signal(msgs) is None


def test_4_different_calls_with_the_same_result_are_progress():
    ok = "Wrote file successfully."
    msgs = [USER] + turn(("write", {"filePath": "a.py", "content": "import os\n" * 30}, ok),
                         ("write", {"filePath": "tests/test_b.py",
                                    "content": "def test_b():\n    assert 1\n"}, ok),
                         ("write", {"filePath": "README.md", "content": "# Parser\n\nUsage"}, ok))
    assert le.signal(msgs) is None
    # waits are no repeat either (a poll), and a user message starts over
    poll = [USER] + turn(("bash", {"command": "make status"}, "running"),
                         ("bash", {"command": "make status"}, "running"))
    assert le.signal(poll, waits={"c0": 20.0, "c1": 20.0}, wait_s=10.0) is None
    assert le.signal(poll, waits={"c0": 2.0, "c1": 2.0}, wait_s=10.0)["why"] == "repeat"
    assert le.signal(poll + [{"role": "user", "content": "check again"}]) is None


def test_the_card_is_read_from_the_checkpoint():
    d = tempfile.mkdtemp(prefix="card-")
    assert le.card_sampling(d) is None and le.card_sampling(None) is None
    with open(os.path.join(d, "generation_config.json"), "w") as f:
        json.dump({"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                   "eos_token_id": [1, 2]}, f)
    got = le.card_sampling(d)
    assert got == {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                   "source": os.path.join(d, "generation_config.json")}
    for bad in ({"do_sample": False, "temperature": 0.7}, {"top_p": 0.9}, {"temperature": 0},
                {"temperature": "hot"}):
        with open(os.path.join(d, "generation_config.json"), "w") as f:
            json.dump(bad, f)
        assert le.card_sampling(d) is None, bad
    # the escape's sampling: the card's with a fresh seed, else the regenerate's settings
    card = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "source": "/m/generation_config.json"}
    a, b = le.sampling_of(card, None), le.sampling_of(card, None)
    assert (a["temperature"], a["top_p"], a["top_k"], a["source"]) == (1.0, 0.95, 20,
                                                                        card["source"])
    assert a["seed"] != b["seed"]
    fb = le.sampling_of(None, {"temperature": 0.7, "top_p": 0.95, "top_k": 20})
    assert fb["source"] == "regenerate" and fb["temperature"] == 0.7
    assert le.sampling_of(None, None) is None


def test_the_profile_section_and_its_schema():
    assert le.profile_sampling({}) == {"agentic_default": "greedy", "loop_escape": "off"}
    assert le.profile_sampling(schema.SHIPPED_PROFILES["coding"]) == {
        "agentic_default": "greedy", "loop_escape": "on"}
    p = json.loads(json.dumps(schema.SHIPPED_PROFILES["coding"]))
    kw = dict(name="coding", router=schema.SHIPPED_ROUTER, packages=schema.SHIPPED_PACKAGES,
              profiles=schema.SHIPPED_PROFILES)
    p["sampling"] = {"agentic_default": "card"}
    assert not schema.check("profiles", p, **kw)
    for bad in ({"agentic_default": "hot"}, {"loop_escape": "maybe"}, {"other": 1}):
        p["sampling"] = bad
        assert schema.check("profiles", p, **kw), bad


# ---------------------------------------------------------------------- 7: the sampled path
def test_7_a_sampled_turn_uses_the_sampled_verification_path():
    """The escape's sampler (the card's 1.0 / 0.95 / 20, a seed) through the served loop: with no
    drafter, the chain, a sampling drafter and the tree walked by drawing (`--sampled-tree det`,
    the box's setting) the tokens are the same, and they are not the greedy decode's."""
    import torch
    import test_app_loop as tal
    from server import app
    sp = le.sampling_of({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "source": "t"}, None)
    sp["seed"] = 1234

    def run(drafter, tree=False, sampled_tree=False, greedy=False):
        tal.serve(drafter, tree=tree)
        app.STATE["sampled_tree"] = sampled_tree
        s = None if greedy else app.Sampler(temperature=sp["temperature"], top_p=sp["top_p"],
                                            top_k=sp["top_k"], seed=sp["seed"])
        assert greedy or s.on
        return list(app.generate_stream(torch.tensor([5, 6, 7, 8, 9]), 48, set(), sampler=s))

    ref = run(None)
    runs = {"chain": run(tal.FixedDrafter(97, 4, tree_mode=False)),
            "sampling drafter": run(tal.QDrafter(6)),
            "tree det": run(tal.FixedDrafter(97, 5, tree_mode=True), tree=True,
                            sampled_tree="det")}
    bad = {k: v for k, v in runs.items() if v != ref}
    assert not bad, (ref, bad)
    greedy = run(tal.FixedDrafter(97, 4, tree_mode=False), greedy=True)
    assert greedy != ref, "the escape's turn is not the greedy decode"
    assert greedy == run(None, greedy=True)


# ---------------------------------------------------------------------- HTTP
def _ledger(path, n, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            c = sqlite3.connect(path)
            rows = c.execute("SELECT request_id, sampling FROM requests WHERE client_kind != "
                             "'internal' ORDER BY rowid").fetchall()
            c.close()
            if len(rows) >= n:
                return rows
        except sqlite3.Error:
            pass
        time.sleep(0.3)
    raise AssertionError(f"ledger has fewer than {n} rows")


def _row(h, port, rid):
    for _ in range(30):
        st, _, out = h._call(port, "GET", "/v1/conductor/decisions?limit=200")
        assert st == 200, out
        got = [d for d in out["decisions"] if d.get("request_id") == rid]
        route = [d for d in got if d.get("class_id") is not None or d.get("task") == "route"]
        if route or got:
            return (route or got)[0], got
        time.sleep(0.3)
    raise AssertionError(f"no decision row for {rid}")


def test_the_escape_over_http():
    """Rows 1, 3, 5, 6 and 8 through the real server."""
    import test_conductor_e2e as h
    port, arb_port = h._port(), h._port()
    cdir = tempfile.mkdtemp(prefix="qse-esc-")
    card = tempfile.mkdtemp(prefix="qse-card-")
    with open(os.path.join(card, "generation_config.json"), "w") as f:
        json.dump({"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}, f)
    ledger = os.path.join(tempfile.mkdtemp(prefix="qse-esc-"), "ledger.sqlite3")
    arb = h._decider(arb_port)
    srv = subprocess.Popen(
        [sys.executable, "-c", h.LAUNCH, "--fake-engine", "--port", str(port),
         "--served-model", "qwen38-spark-engine", "--usage-ledger", ledger, "--fake-tps", "2000",
         "--conductor-config", cdir, "--conductor-defaults", "shipped", "--conductor", "on"],
        cwd=ROOT, env=dict(h._env(), QSE_FAKE_CARD_DIR=card), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    try:
        h._wait(arb_port, arb, "/readyz")
        h._wait(port, srv)
        st, _, roles = h._call(port, "GET", "/v1/conductor/roles")
        doc = roles["doc"]
        doc["decider"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        doc["classifier"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        assert h._call(port, "PUT", "/v1/conductor/roles", {"doc": doc, "note": "fake"})[0] == 200
        oc = {"User-Agent": "opencode/1.18.34", "x-session-id": "esc-1"}
        tools = h.TOOLS + [{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object", "properties": {
                "filePath": {"type": "string"}}}}}]

        def ask(msgs, headers=oc, with_tools=True, **extra):
            body = {"model": "TandemLLM", "messages": msgs, "max_tokens": 48,
                    "chat_template_kwargs": {"enable_thinking": False}, **extra}
            if with_tools:
                body["tools"] = tools
            st, hd, out = h._call(port, "POST", "/v1/chat/completions", body, headers=headers,
                                  token=None)
            assert st == 200, (st, out)
            return out

        rep = [USER] + turn(("read", READ_A, "1: import re"), ("read", READ_A, "1: import re"))
        r1 = ask(rep)                                                      # 1 repeat: escape
        moved = rep + turn(("bash", {"command": "pytest -q"}, "1 failed"), start=2)
        r2 = ask(moved)                                                    # 2 progress: back
        r3 = ask(moved)                                                    # 3 greedy, no record
        r4 = ask(rep, headers={**oc, "x-session-id": "esc-2"}, temperature=0.2)  # 4 client
        r5 = ask(rep, headers={**oc, "x-session-id": "esc-3"}, with_tools=False)  # 5 no tools
        r6 = ask(rep, headers={**oc, "x-session-id": "esc-4", "X-Tandem": "raw"})  # 6 raw
        # 7: agentic_default card on the coding profile
        st, _, cur = h._call(port, "GET", "/v1/conductor/profiles/coding")
        assert st == 200, cur
        pdoc = cur["doc"]
        pdoc["sampling"] = dict(pdoc.get("sampling") or {}, agentic_default="card")
        st, _, put = h._call(port, "PUT", "/v1/conductor/profiles/coding",
                             {"doc": pdoc, "note": "card", "base_version": cur["version"]})
        assert st == 200, put
        r7 = ask([USER], headers={**oc, "x-session-id": "esc-5"})
        rows = _ledger(ledger, 7)
        by = dict(rows)
        got = [by.get(r["id"]) for r in (r1, r2, r3, r4, r5, r6, r7)]
        assert got == ["loop_escape", None, None, "client", None, None, "card"], got
        # the decision row: on (since turn 3), then back at turn 4
        row1, all1 = _row(h, port, r1["id"])
        ex = row1["extra"]["loop_escape"]
        assert (ex["state"], ex["why"], ex["name"], ex["since_turn"], ex["turn"]) == \
            ("on", "repeat", "read", 3, 3), ex
        assert (ex["temperature"], ex["top_p"], ex["top_k"], ex["mode"]) == (1.0, 0.95, 20,
                                                                              "loop_escape")
        assert ex["source"].endswith("generation_config.json") and isinstance(ex["seed"], int)
        evs = [e for d in all1 for e in [d] + list(d.get("events") or [])]
        notes = [e for e in evs if e.get("task") == "C3" and "PROMPT_TAIL_NOTE" in (
            e.get("decision"), e.get("action"))]
        assert notes and "`read`" in (notes[0].get("text") or ""), [
            {k: e.get(k) for k in ("task", "decision", "action", "reason")} for e in evs]
        # the usage frame says how it sampled
        ts = r1["usage"]["tandem"]["sampling"]
        assert (ts["mode"], ts["state"], ts["since_turn"], ts["temperature"]) == \
            ("loop_escape", "on", 3, 1.0), ts
        row2, _ = _row(h, port, r2["id"])
        back = row2["extra"]["loop_escape"]
        assert (back["state"], back["since_turn"], back["back_at_turn"]) == ("back", 3, 4), back
        assert "sampling" not in (r2["usage"].get("tandem") or {}) or \
            r2["usage"]["tandem"]["sampling"].get("mode") is None
        row3, _ = _row(h, port, r3["id"])
        assert "loop_escape" not in row3["extra"]
        row4, _ = _row(h, port, r4["id"])
        assert row4["extra"]["loop_escape"]["client_params"] is True
        assert "temperature" not in row4["extra"]["loop_escape"]
        row7, _ = _row(h, port, r7["id"])
        assert row7["extra"]["sampling"]["mode"] == "card"
        st, _, hl = h._call(port, "GET", "/health", token=None)
        assert hl["sampling"]["card"]["temperature"] == 1.0
        assert hl["sampling"]["loop_escape"]["started"] >= 2, hl["sampling"]
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
