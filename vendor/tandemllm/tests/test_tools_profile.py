"""SRV-621 (QA SRV-622): the `tools` profile and the resolve rule for plain API clients that offer
tools. A request from a client kind no profile maps (an OpenAI SDK, curl, an unnamed client such as
a benchmark) that offers function tools, `tool_choice` not "none", resolves to
`settings.tools_profile` (shipped: `tools`): the prompt the client sent, token for token the raw
path's; the loop guards and the tool-call checks act; no packages, no memory, no transcript.

The test matrix (SRV-621's Gherkin):

  1  the shipped profile: derived from general, classify and effort off, packages and
     rules off, memory absent, no transcript, no state lines, no loop escape, `prompt: client`,
     D1 D1b D2 E1 C3 C4 C5 act and nothing else, valid, not locked; shipped to a box that lacks it
                                                test_the_shipped_tools_profile
  2  resolve: an unmapped kind with tools -> tools (source `tools`); without tools, with
     `tool_choice: none` or with no function tool -> the default; mapped kinds keep their
     profile; the header wins; X-Tandem: raw still lowers; `tools_profile` "" turns the rule
     off; a profile with memory or a store without the profile leaves it to the default
                                                test_resolve_*
  3  raw is the floor: the rendered prompt ids under `tools` equal the raw path's, on a plain
     turn, a turn that asks who the model is (general adds the identity note there), a tool
     result turn, an Open WebUI-shaped working-directory tool, and after a general turn in the
     same session (no replay)               test_the_tools_prompt_is_the_raw_prompt
  4  over the wire (fake engine, shipped defaults, --conductor on): the response header, the
     ledger row and the decision row say `tools` (extra.profile_source), nothing injected, no
     memory row, no session transcript; the engine input equals the raw request's; the same client
     without tools or with `tool_choice: none` stays on general, opencode on coding
                                                test_the_rule_over_http

Run: python tests/test_tools_profile.py
"""

from __future__ import annotations

import copy
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

from server.conductor import router as router_mod, schema, shipped, signals  # noqa: E402
from server.conductor.store import Store  # noqa: E402

FN = [{"type": "function", "function": {"name": "get_weather", "description": "the weather",
                                         "parameters": {"type": "object", "properties": {
                                             "city": {"type": "string"}}}}}]
UNMAPPED = ("openai-sdk", "curl", "other")
MAPPED = {"opencode": "coding", "hermes": "hermes", "open-webui": "chat", "dashboard": "chat",
          "chat-app": "chat-app"}


def _store() -> Store:
    st = Store(tempfile.mkdtemp(prefix="qse-tools-"))
    shipped.install(st)
    return st


def _resolve(st, kind, body=None, headers=None):
    return router_mod.resolve(st, "TandemLLM", "qwen38-spark-engine", headers or {}, kind,
                              offers_tools=signals.offers_tools(body or {}))


# ------------------------------------------------------------------ 1 the profile
def test_the_shipped_tools_profile():
    t, g = schema.SHIPPED_PROFILES["tools"], schema.SHIPPED_PROFILES["general"]
    assert t["mode"] == "on" and t["clients"] == [] and "locked" not in t
    assert t["decisions"] == {"classify": "off", "effort": "off", "packages": "off",
                              "rules": "off"}
    assert "memory" not in t and t["transcripts"] == "off" and t["prompt"] == "client"
    assert t["ui"]["states"] == "off"
    assert t["sampling"] == {"agentic_default": "greedy", "loop_escape": "off"}
    acting = sorted(k for k, v in t["tasks"].items() if v["mode"] == "act")
    assert acting == sorted(["D1", "D1b", "D2", "E1", "C3", "C4", "C5"]), acting
    assert all(v["mode"] in ("act", "off") for v in t["tasks"].values())
    assert t["toolgate"]["same_call"] and t["toolgate"]["malformed"] and t["toolgate"]["sequence"]
    assert not t["toolgate"]["risk"] and not t["toolgate"]["project"]
    # derived from general: every key it does not set itself is general's
    parent, own = schema.DERIVED_PROFILES["tools"]
    assert parent == "general"
    for k in set(t) | set(g):
        if k not in own:
            assert t.get(k) == g.get(k), k
    assert not schema.check("profiles", t, name="tools", router=schema.SHIPPED_ROUTER,
                            packages=schema.SHIPPED_PACKAGES, profiles=schema.SHIPPED_PROFILES)
    assert schema.SHIPPED_SETTINGS["tools_profile"] == "tools"
    assert not schema.check("settings", schema.SHIPPED_SETTINGS, profiles=schema.SHIPPED_PROFILES)
    bad = dict(schema.SHIPPED_SETTINGS, tools_profile="nope")
    assert any("tools_profile" in e for e in schema.check("settings", bad,
                                                          profiles=schema.SHIPPED_PROFILES))
    # a box that stored its documents before: the next start stores `tools` and nothing else
    st = _store()
    root = st.root
    import shutil
    shutil.rmtree(os.path.join(root, "profiles", "tools"))
    st = Store(root)
    assert shipped.install(st) == ["profiles/tools v1"]
    # a stored settings document without the key reads it as `tools`
    _, sdoc = st.get("settings")
    sdoc.pop("tools_profile", None)
    st.put("settings", None, sdoc, note="before SRV-621", author="t")
    assert schema.tools_profile(st.get("settings")[1]) == "tools"
    assert _resolve(st, "curl", {"tools": FN}).profile == "tools"


# ------------------------------------------------------------------ 2 resolve
def test_resolve_an_unmapped_client_with_tools_gets_tools():
    st = _store()
    for k in UNMAPPED:
        r = _resolve(st, k, {"tools": FN})
        assert (r.profile, r.source, r.mode) == ("tools", "tools", "on"), (k, r.profile)
        r = _resolve(st, k, {"tools": FN, "tool_choice": "auto"})
        assert r.profile == "tools"
        r = _resolve(st, k, {"tools": FN, "tool_choice": "required"})
        assert r.profile == "tools"


def test_resolve_without_tools_or_with_tool_choice_none_keeps_the_default():
    st = _store()
    for k in UNMAPPED:
        for b in ({}, {"tools": []}, {"tools": FN, "tool_choice": "none"},
                  {"tools": [{"type": "code_interpreter"}]}):
            r = _resolve(st, k, b)
            assert (r.profile, r.source) == ("general", "default"), (k, b, r.profile)
    # the function name decides, the flat form counts as the template reads it
    assert signals.offers_tools({"tools": [{"name": "f", "parameters": {}}]})
    assert not signals.offers_tools({"tools": [{"type": "function", "function": {}}]})
    assert not signals.offers_tools({"tools": "bash"})


def test_resolve_mapped_kinds_keep_their_profile():
    st = _store()
    for k, prof in MAPPED.items():
        for b in ({}, {"tools": FN}):
            r = _resolve(st, k, b)
            assert (r.profile, r.source) == (prof, "client"), (k, b, r.profile)


def test_resolve_the_header_wins_and_raw_still_lowers():
    st = _store()
    r = _resolve(st, "curl", {"tools": FN}, {"X-Tandem-Profile": "general"})
    assert (r.profile, r.source) == ("general", "header")
    r = _resolve(st, "curl", {}, {"X-Tandem-Profile": "tools"})
    assert (r.profile, r.source) == ("tools", "header")
    r = _resolve(st, "openai-sdk", {"tools": FN}, {"X-Tandem": "raw"})
    assert (r.profile, r.mode) == ("tools", "raw")
    r = _resolve(st, "openai-sdk", {"tools": FN}, {"X-Tandem": "shadow"})
    assert r.mode == "shadow"
    r = router_mod.resolve(st, "TandemLLM", "q", {}, "curl", global_mode="off", offers_tools=True)
    assert (r.profile, r.mode) == ("tools", "raw")


def test_resolve_the_settings_and_the_memory_guard():
    st = _store()
    _, sdoc = st.get("settings")
    st.put("settings", None, dict(sdoc, tools_profile=""), note="rule off", author="t")
    assert _resolve(st, "curl", {"tools": FN}).profile == "general"
    st.put("settings", None, dict(sdoc, tools_profile="chat"), note="chat", author="t")
    # chat keeps memory notes: a keyless client is never sent to it by the rule
    assert schema.profile_memory(st.get("profiles", "chat")[1]) == "on"
    assert _resolve(st, "curl", {"tools": FN}).profile == "general"
    st.put("settings", None, dict(sdoc, tools_profile="tools"), note="back", author="t")
    _, tdoc = st.get("profiles", "tools")
    st.put("profiles", "tools", dict(tdoc, memory="shadow"), note="memory", author="t")
    assert _resolve(st, "curl", {"tools": FN}).profile == "general"
    # the built-in documents (no shipped defaults): no `tools` profile, the default as before
    built = Store(tempfile.mkdtemp(prefix="qse-tools-"))
    r = _resolve(built, "curl", {"tools": FN})
    assert (r.profile, r.source) == ("chat", "default")


# ------------------------------------------------------------------ 3 raw is the floor
def test_the_tools_prompt_is_the_raw_prompt():
    from test_conductor_router import setup, turn
    from server.conductor.router import COMPOSITE
    cond, _ = setup()
    shipped.install(cond.store)
    cwd_tool = {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command.\n\nThe current working "
        "directory is: /home/user/project", "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}}}}
    cases = {
        "plain": [{"role": "user", "content": "What is the weather in Tokyo?"}],
        "identity": [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": "Who are you? Use a tool if you need one."}],
        "tool_turn": [{"role": "user", "content": "Weather in Paris?"},
                      {"role": "assistant", "content": "", "tool_calls": [{
                          "id": "c1", "type": "function", "function": {
                              "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                      {"role": "tool", "tool_call_id": "c1", "content": "18 C, rain"}],
    }

    def body(msgs, tools=FN):
        return {"model": COMPOSITE, "messages": copy.deepcopy(msgs), "tools": copy.deepcopy(tools),
                "max_tokens": 256}

    for name, msgs in list(cases.items()) + [("cwd", cases["plain"])]:
        tools = FN + [cwd_tool] if name == "cwd" else FN
        sid = {"x-session-id": f"floor-{name}"}
        raw, ids_raw, _ = turn(cond, body(msgs, tools), headers=dict(sid, **{"X-Tandem": "raw"}),
                               kind="openai-sdk")
        assert not raw.active
        plan, ids, _ = turn(cond, body(msgs, tools), headers=sid, kind="openai-sdk")
        assert plan.active and plan.res.profile == "tools" and plan.res.source == "tools", name
        assert ids == ids_raw, name
        assert plan.block is None and plan.extras == [] and not plan.changed, name
        assert plan.row["injected"] == [] and plan.row["acted"] == 0, name
        assert plan.row["extra"]["profile_source"] == "tools"
        assert plan.row["extra"]["prompt"]["mode"] == "client"
        assert "memory" not in plan.row["extra"] and plan.memory is None, name
        assert plan.budget is None and not plan.thinking_off
        assert plan.row["decisions"]["packages"] == "off"
        if name == "identity":
            # what the plan would have added is journaled by kind, never served
            assert "identity" in plan.row["extra"]["prompt"]["dropped"], plan.row["extra"]
            gen, ids_gen, _ = turn(cond, body(msgs), headers=dict(
                sid, **{"X-Tandem-Profile": "general"}), kind="openai-sdk")
            assert gen.active and ids_gen != ids_raw, "general adds the identity note"
            # the same session afterwards under tools: no note of general's is replayed
            again, ids_again, _ = turn(cond, body(msgs), headers=sid, kind="openai-sdk")
            assert again.res.profile == "tools" and ids_again == ids_raw
        if name == "cwd":
            # Open WebUI's working-directory sentence stays in the tool's description
            gen, ids_gen, _ = turn(cond, body(msgs, tools), headers={
                "x-session-id": "floor-cwd-gen", "X-Tandem-Profile": "general"},
                kind="openai-sdk")
            assert ids_gen != ids_raw, "general moves the sentence into a note"


# ------------------------------------------------------------------ 4 over the wire
def _rows(ledger, n, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            c = sqlite3.connect(ledger)
            rows = c.execute("SELECT request_id, profile FROM requests WHERE client_kind != "
                             "'internal' ORDER BY rowid").fetchall()
            c.close()
            if len(rows) >= n:
                return dict(rows)
        except sqlite3.Error:
            pass
        time.sleep(0.3)
    raise AssertionError(f"ledger has fewer than {n} rows")


def _prompts(path):
    try:
        with open(path) as f:
            return [json.loads(x) for x in f if x.strip()]
    except FileNotFoundError:
        return []


def test_the_rule_over_http():
    import test_conductor_e2e as h
    from test_loop_escape import _row
    port, arb_port = h._port(), h._port()
    cdir = tempfile.mkdtemp(prefix="qse-tools-")
    work = tempfile.mkdtemp(prefix="qse-tools-")
    ledger, plog = os.path.join(work, "ledger.sqlite3"), os.path.join(work, "prompts.jsonl")
    arb = h._decider(arb_port)
    srv = subprocess.Popen(
        [sys.executable, "-c", h.LAUNCH, "--fake-engine", "--port", str(port),
         "--served-model", "qwen38-spark-engine", "--usage-ledger", ledger, "--fake-tps", "2000",
         "--conductor-config", cdir, "--conductor-defaults", "shipped", "--conductor", "on"],
        cwd=ROOT, env=dict(h._env(), QSE_FAKE_PROMPTS=plog), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    try:
        h._wait(arb_port, arb, "/readyz")
        h._wait(port, srv)
        st, _, roles = h._call(port, "GET", "/v1/conductor/roles")
        doc = roles["doc"]
        doc["decider"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        doc["classifier"]["endpoint"] = f"http://127.0.0.1:{arb_port}"
        assert h._call(port, "PUT", "/v1/conductor/roles", {"doc": doc, "note": "fake"})[0] == 200
        sdk = {"User-Agent": "OpenAI/Python 2.24.0"}
        msgs = [{"role": "user", "content": "Who are you? Check the weather in Oslo."}]

        def ask(headers, tools=True, **extra):
            body = {"model": "TandemLLM", "messages": msgs, "max_tokens": 64, **extra}
            if tools:
                body["tools"] = FN
            before = len(_prompts(plog))
            st, hd, out = h._call(port, "POST", "/v1/chat/completions", body,
                                  headers=headers, token=None)
            assert st == 200, (st, out)
            new = _prompts(plog)[before:]
            assert new, "the engine saw no prompt"
            return out, hd, new[-1]["sha256"]       # the turn's own prompt is the last one

        r_raw, h_raw, p_raw = ask(dict(sdk, **{"X-Tandem": "raw", "x-session-id": "t-raw"}))
        r_t, h_t, p_t = ask(dict(sdk, **{"x-session-id": "t-tools"}))
        r_g, h_g, p_g = ask(dict(sdk, **{"x-session-id": "t-gen",
                                         "X-Tandem-Profile": "general"}))
        r_nt, h_nt, _ = ask(dict(sdk, **{"x-session-id": "t-notools"}), tools=False)
        r_none, h_none, _ = ask(dict(sdk, **{"x-session-id": "t-none"}), tool_choice="none")
        r_oc, h_oc, _ = ask({"User-Agent": "opencode/1.18.34", "x-session-id": "t-oc"})
        r_curl, h_curl, p_curl = ask({"User-Agent": "curl/8.5.0", "x-session-id": "t-curl"})
        assert h_t.get("X-Tandem-Profile") == "tools" and h_curl.get("X-Tandem-Profile") == "tools"
        assert h_g.get("X-Tandem-Profile") == "general"
        assert h_nt.get("X-Tandem-Profile") == "general"
        assert h_none.get("X-Tandem-Profile") == "general"
        assert h_oc.get("X-Tandem-Profile") == "coding"
        # the engine input: tools equals raw; general adds the identity note
        assert p_t == p_raw and p_curl == p_raw, "the tools prompt is the raw prompt"
        assert p_g != p_raw
        rows = _rows(ledger, 7)
        assert rows[r_t["id"]] == "tools" and rows[r_curl["id"]] == "tools", rows
        assert rows[r_g["id"]] == "general" and rows[r_nt["id"]] == "general"
        row, allrows = _row(h, port, r_t["id"])
        assert row["profile"] == "tools" and row["extra"]["profile_source"] == "tools", row
        assert row["injected"] in ([], None) and row["acted"] == 0, row
        assert row["decisions"]["packages"] == "off" and row["decisions"]["rules"] == "off"
        assert "memory" not in row["extra"], row["extra"]
        assert "identity" in row["extra"]["prompt"]["dropped"], row["extra"]
        assert not [e for d in allrows for e in [d] + list(d.get("events") or [])
                    if "memory" in str(e.get("kind") or "")], allrows
        # no session transcript for the tools requests; general's is written (the control)
        deadline, keys = time.time() + 10, set()
        while time.time() < deadline:
            st, _, lst = h._call(port, "GET", "/v1/conductor/memory/sessions")
            keys = {s["session"] for s in lst.get("sessions") or []} if st == 200 else set()
            if "c:t-gen" in keys:
                break
            time.sleep(0.3)
        assert "c:t-gen" in keys, keys
        assert "c:t-tools" not in keys and "c:t-curl" not in keys, keys
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
