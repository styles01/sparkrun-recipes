"""The harness in front of any backend (proxy P4), against fake backends, on a CPU, in seconds:
the switches and their defaults, the message-level note and its replay (with the fake's prefix
cache), harness off byte for byte, the class question through each kind's constraint field, the
effort switch per adapter, E1 and the regenerate as second requests in one client stream, the
repair, the argument guard and its note on the next turn, D1 in shadow where the backend cannot
continue and closed where it can, the side request's cap, the compaction section, the word-level
detector, and the switches in the Backends API.

The night pass (tools/proxy_e2e.sh) runs this file with the others.

Run: python tests/test_proxy_harness.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import Proxy, fake, run  # noqa: E402

from proxy import app as proxy_app  # noqa: E402
from proxy import harness  # noqa: E402
from proxy.adapters import make  # noqa: E402
from server.conductor import inject_messages as im  # noqa: E402
from server.conductor.monitor import tier0_words  # noqa: E402

UA = {"User-Agent": "opencode/1.18.34 ai-sdk/provider-utils/3"}
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"filePath": {"type": "string"},
                                         "content": {"type": "string"}}}}}]
SYSTEM = {"role": "system", "content": "You are a coding agent. Working directory: /tmp/proj"}


def _env(kind="vllm", mode="on", port=None, bid="v", extra=None):
    up = bid.upper()
    env = {"TANDEM_BACKENDS": bid, f"TANDEM_BACKEND_{up}_KIND": kind,
           f"TANDEM_BACKEND_{up}_URL": f"http://127.0.0.1:{port}",
           f"TANDEM_BACKEND_{up}_HARNESS": mode}
    if kind in ("openai", "deepseek"):
        env[f"TANDEM_BACKEND_{up}_KEY"] = "sk-fixture-ABCDEFGH1234"
    env.update(extra or {})
    return env


def _frames(raw: bytes) -> list:
    out = []
    for ev in raw.decode().split("\n\n"):
        ev = ev.strip()
        if not ev.startswith("data:"):
            continue
        d = ev[5:].strip()
        out.append(d if d == "[DONE]" else json.loads(d))
    return out


def _main_bodies(srv) -> list:
    """The fake's request bodies that were not the router's questions."""
    return [b for b in srv.fake.bodies if not (b.get("structured_outputs") or b.get("regex")
                                                or b.get("response_format")
                                                or (b.get("stream") is False and
                                                    "task classes" in str((b.get("messages")
                                                                           or [{}])[0])))]


def _journal():
    cond = proxy_app.STATE["conductor"]
    cond.journal.flush()
    return cond.journal


def _events(task=None) -> list:
    j = _journal()
    evs = list(j.events_ring)
    return [e for e in evs if task is None or e.get("task") == task]


def _stream_text(frames) -> tuple[str, str]:
    content, reasoning = "", ""
    for f in frames:
        if f == "[DONE]":
            continue
        for ch in f.get("choices") or []:
            d = ch.get("delta") or {}
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
    return content, reasoning


# --------------------------------------------------------------------------------- units
def test_switches_defaults_overrides_and_mode_cap():
    local = {"cloud": False, "harness": {"mode": "on"}}
    cloud = {"cloud": True, "harness": {"mode": "on"}}
    s = harness.switches(local, None)
    assert s["packages"] == "on" and s["e1"] == "on" and s["tools"] == "on"
    c = harness.switches(cloud, None)
    assert c["packages"] == "off" and c["e1"] == "off" and c["tools"] == "shadow"
    assert c["class"] == "on" and c["side"] == "on" and c["loops"] == "on"
    c2 = harness.switches(cloud, {"features": {"packages": "on", "e1": "on"}})
    assert c2["packages"] == "on" and c2["e1"] == "on"
    sh = harness.switches({"cloud": False, "harness": {"mode": "shadow"}}, None)
    assert all(sh[g] == "shadow" for g in harness.GROUPS)
    off = harness.switches(local, {"features": {"mode": "off"}})
    assert off["mode"] == "off" and all(off[g] == "off" for g in harness.GROUPS)
    d = harness.degradations(cloud, make({"kind": "openai", "cloud": True}).capabilities())
    assert set(d) == set(harness.GROUPS) and "never closed" in d["loops"]
    d2 = harness.degradations(local, make({"kind": "vllm"}).capabilities())
    assert "second request" in d2["loops"]


def test_inject_messages_hash_splice_and_parts():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert im.last_user(msgs) == 1
    h = im.prefix_hash(msgs, 1)
    # a field the client adds between turns does not break the hash; the text does
    assert im.prefix_hash([msgs[0], dict(msgs[1], cache_control={"type": "x"})], 1) == h
    assert im.prefix_hash([msgs[0], {"role": "user", "content": "hi!"}], 1) != h
    assert im.hashes_at(msgs, [1, 0]) == {0: im.prefix_hash(msgs, 0), 1: h}
    out = im.splice(msgs, [(1, "\n\nNOTE")])
    assert out[1]["content"] == "hi\n\nNOTE" and msgs[1]["content"] == "hi"
    parts = im.append_text({"role": "user", "content": [{"type": "text", "text": "a"}]}, "N")
    assert parts["content"][-1] == {"type": "text", "text": "N"}
    assert im.with_hidden_turn(msgs, "go")[-1] == {"role": "user", "content": "go"}


def test_adapter_switches_and_choice_bodies():
    b = {"model": "m", "messages": []}
    v = make({"kind": "vllm"})
    out, how = v.thinking(b, False, "off")
    assert out["chat_template_kwargs"] == {"enable_thinking": False} and \
        how == "enable_thinking=false"
    assert v.thinking({"chat_template_kwargs": {"enable_thinking": True}}, False)[1] == "client"
    oa = make({"kind": "openai", "cloud": True})
    assert oa.thinking({"model": "gpt-4.1"}, False, "off")[1] == "unsupported"
    assert oa.thinking({"model": "gpt-5.4-mini"}, False, "off")[0]["reasoning_effort"] == \
        "minimal"
    assert oa.thinking({"model": "o4-mini"}, True, "high")[0]["reasoning_effort"] == "high"
    ds = make({"kind": "deepseek", "cloud": True})
    assert ds.thinking({"model": "deepseek-flash"}, False)[0]["thinking"] == {"type": "disabled"}
    q, how = v.choice_body("sys", "user", ["a", "b"], "m")
    assert q["structured_outputs"] == {"choice": ["a", "b"]} and how == "choice"
    q, how = make({"kind": "sglang"}).choice_body("sys", "user", ["a.b", "c"], "m")
    assert q["regex"] == "(a\\.b|c)"
    q, how = oa.choice_body("sys", "user", ["a", "b"], "gpt-4.1")
    assert how == "json" and q["response_format"]["json_schema"]["schema"]["properties"][
        "choice"]["enum"] == ["a", "b"]
    q, how = make({"kind": "generic"}).choice_body("sys", "user", ["a", "b"], "m")
    assert how == "text" and "structured_outputs" not in q
    assert make({"kind": "tandemllm"}).choice_headers() == [("X-Tandem", "raw")]
    assert harness.parse_choice('{"choice": "b"}', ["a", "b"], "json") == "b"
    assert harness.parse_choice("`code_edit`.", ["code_edit", "debug"], "text") == "code_edit"
    assert harness.parse_choice("I think debug fits", ["code_edit", "debug"], "text") == "debug"
    assert harness.parse_choice("code_edit or debug", ["code_edit", "debug"], "text") is None
    c = v.continue_body({"model": "m", "messages": [{"role": "user", "content": "x"}]}, "<think>")
    assert c["continue_final_message"] is True and c["messages"][-1]["role"] == "assistant"
    assert oa.continue_body({"messages": []}, "x") is None
    assert harness.repair_json('{"a": 1,}') == '{"a": 1}'
    assert harness.repair_json('{"a" 1 oops') is None


def test_word_loop_detector_fires_on_a_loop_not_on_prose():
    cfg = {"calibrated": True, "loop": {"min_size": 1, "max_size": 12, "min_count": 4,
                                        "min_span": 80},
           "repeat": {"gram": 8, "window": 100, "every": 25, "min_words": 200, "act": 0.85,
                      "grey": 0.4, "in_a_row": 2},
           "novelty": {"window": 60, "gram": 5, "min_words": 120, "below": 0.05}}
    w = tier0_words.WordLoop(cfg)
    prose = " ".join(f"word{i} thing{i * 7 % 13} case{i % 5}" for i in range(400))
    for i in range(0, len(prose), 37):
        assert w.feed(prose[i:i + 37]) is None
    w2 = tier0_words.WordLoop(cfg)
    loop = "I should check the map again and then decide what to do next. " * 80
    fired = None
    for i in range(0, len(loop), 23):
        fired = fired or w2.feed(loop[i:i + 23])
    assert fired and fired.startswith(("wloop", "wrepeat", "wnovelty")), fired
    assert tier0_words.words_cfg(None)["calibrated"] is False


# --------------------------------------------------------------------------------- the relay
def test_harness_off_is_byte_exact_with_a_conductor():
    srv, port = fake(models=["m1"], cache=False)
    body = {"model": "m1", "messages": [SYSTEM, {"role": "user", "content": "hello FAKE_TOOL"}],
            "tools": TOOLS, "stream": True, "stream_options": {"include_usage": True}}
    raw = json.dumps(body).encode()
    direct = _direct_stream(port, raw)
    with Proxy(env=_env("vllm", "off", port), conductor=True) as px:
        px.sync()
        st, _, out = px.post("/v1/chat/completions", raw, headers=UA)
        assert st == 200
        assert srv.fake.last["raw"].encode() == raw            # the upstream body, byte for byte
        assert out == direct                                   # the client's stream, too
        assert len(srv.fake.bodies) == 2                       # no class question
        assert not _events()
        rows = px.rows()
        assert rows[-1]["conductor_mode"] is None and rows[-1]["profile"] is None
    srv.shutdown()


def _direct_stream(port, raw: bytes) -> bytes:
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions", raw, {"Content-Type": "application/json"})
    out = c.getresponse().read()
    c.close()
    return out


def test_note_replayed_at_the_same_index_and_the_prefix_cache_holds():
    srv, port = fake(models=["m1"])
    u1 = {"role": "user", "content": "Fix the failing test in utils.py FAKE_CLASS code_edit"}
    with Proxy(env=_env("vllm", "on", port), conductor=True) as px:
        px.sync()
        st, _, out = px.post("/v1/chat/completions", {"model": "m1", "messages": [SYSTEM, u1],
                                                      "tools": TOOLS, "stream": True},
                             headers=UA)
        assert st == 200
        b1 = _main_bodies(srv)[-1]
        note = b1["messages"][1]["content"][len(u1["content"]):]
        assert note.startswith("\n\n<runtime-note>") and b1["messages"][0] == SYSTEM
        # the class question went to the backend with vLLM's choice field
        qs = [b for b in srv.fake.bodies if b.get("structured_outputs")]
        assert qs and qs[0]["chat_template_kwargs"] == {"enable_thinking": False}
        cached = []
        msgs = [SYSTEM, u1]
        for turn in range(2):
            msgs = msgs + [{"role": "assistant", "content": f"answer {turn}"},
                           {"role": "user", "content": f"go on {turn} FAKE_CLASS code_edit"}]
            st, _, out = px.post("/v1/chat/completions", {
                "model": "m1", "messages": msgs, "tools": TOOLS, "stream": True,
                "stream_options": {"include_usage": True}}, headers=UA)
            b = _main_bodies(srv)[-1]
            assert b["messages"][1]["content"] == u1["content"] + note       # replayed
            assert "<runtime-note>" not in b["messages"][-1]["content"]      # same class: none new
            usage = [f for f in _frames(out) if f != "[DONE]" and f.get("usage")][-1]["usage"]
            cached.append(usage["prompt_tokens_details"]["cached_tokens"])
        # the backend's prefix cache covered the system, the noted message and the turns before
        assert cached[0] > len(note.split()) and cached[1] > cached[0], cached
        rows = [r for r in _journal().rows(limit=10) if r.get("kind") == "route"]
        assert any((r.get("replayed") or {}).get("count") for r in rows)
        # a changed head drops the note (and the router adds its block again)
        msgs2 = [SYSTEM, {"role": "user", "content": "Fix it differently FAKE_CLASS code_edit"}] + \
            msgs[2:]
        px.post("/v1/chat/completions", {"model": "m1", "messages": msgs2, "tools": TOOLS,
                                         "stream": True}, headers=UA)
        b = _main_bodies(srv)[-1]
        assert b["messages"][1]["content"] == msgs2[1]["content"]
        led = px.rows()
        assert any(r["client_kind"] == "harness" and r["client_id"] == "harness:class"
                   for r in led)
        main = [r for r in led if r["client_kind"] != "harness"]
        assert main[0]["profile"] and main[0]["conductor_mode"] == "on" and \
            main[0]["conductor_acted"] == 1
    srv.shutdown()


def test_e1_second_request_is_one_client_stream():
    srv, port = fake(models=["m1"])
    body = {"model": "m1", "messages": [SYSTEM, {"role": "user", "content":
                                                 "Do it FAKE_EMPTY_ANSWER FAKE_CLASS code_edit"}],
            "tools": TOOLS, "stream": True}
    with Proxy(env=_env("vllm", "on", port), conductor=True) as px:
        px.sync()
        st, _, out = px.post("/v1/chat/completions", body, headers=UA)
        fr = _frames(out)
        assert fr[-1] == "[DONE]" and fr.count("[DONE]") == 1
        ids = {f["id"] for f in fr if f != "[DONE]"}
        assert len(ids) == 1
        roles = [f for f in fr if f != "[DONE]" and any(
            "role" in (c.get("delta") or {}) for c in f.get("choices") or [])]
        assert len(roles) == 1
        content, _ = _stream_text(fr)
        assert "w0" in content                       # the second request's answer
        mains = _main_bodies(srv)
        assert len(mains) == 2 and mains[1]["messages"][-1]["role"] == "user"
        assert mains[1]["chat_template_kwargs"] == {"enable_thinking": False}
        e1 = _events("E1")
        assert e1 and e1[0]["mode"] == "act" and e1[0]["decision"] == "CONTINUE_HIDDEN"
        row = [r for r in px.rows() if r["client_kind"] != "harness"][0]
        assert row["conductor_acts"] >= 1 and row["finish_reason"] == "stop"
        # both requests' tokens are in the client's row
        assert row["completion_tokens"] > 24
    srv.shutdown()


def test_cost_comment_sums_the_harness_requests():
    """E1's second request is billed too: the cost comment chatd reads (X-Tandem-Cost: 1) is the
    client row's cost, both requests' tokens in it, once, before the one [DONE]."""
    srv, port = fake(models=["m1"])
    body = {"model": "m1", "messages": [SYSTEM, {"role": "user", "content":
                                                 "Do it FAKE_EMPTY_ANSWER FAKE_CLASS code_edit"}],
            "tools": TOOLS, "stream": True, "stream_options": {"include_usage": True}}
    with Proxy(env=_env("vllm", "on", port, extra={"TANDEM_PRICE_1": "v/m1=1,0.5,2"}),
               conductor=True) as px:
        px.sync()
        st, _, out = px.post("/v1/chat/completions", body, headers=dict(UA, **{
            "X-Tandem-Cost": "1"}))
        assert st == 200 and len(_main_bodies(srv)) == 2
        events = out.decode().split("\n\n")
        costs = [e for e in events if e.startswith(": tandem-cost ")]
        assert len(costs) == 1 and events[events.index(costs[0]) + 1] == "data: [DONE]"
        row = [r for r in px.rows() if r["client_kind"] != "harness"][0]
        said = json.loads(costs[0][len(": tandem-cost "):])
        assert said["cost_micros"] == row["cost_micros"] and row["cost_micros"] > 0
        c = row["cached_tokens"] or 0
        assert row["cost_micros"] == round((row["prompt_tokens"] - c) * 1 + c * 0.5
                                           + row["completion_tokens"] * 2)
    srv.shutdown()


def test_e1_off_on_a_cloud_backend_by_default():
    srv, port = fake(models=["deepseek-flash"])
    body = {"model": "deepseek-flash", "messages": [SYSTEM, {"role": "user", "content":
                                                             "Do it FAKE_EMPTY_ANSWER"}],
            "tools": TOOLS, "stream": True}
    with Proxy(env=_env("deepseek", "on", port, bid="ds",
                        extra={"TANDEM_BACKEND_DS_MODELS": "deepseek-flash",
                               "TANDEM_BACKEND_DS_MODELS_MODE": "list"}),
               conductor=True) as px:
        px.sync()
        st, _, out = px.post("/v1/chat/completions", body, headers=UA)
        assert st == 200
        assert len(_main_bodies(srv)) == 1 and not _events("E1")
        # packages are off on a cloud model by default: no note in the message
        rows = [r for r in _journal().rows(limit=5) if r.get("kind") == "route"]
        parts = [p["kind"] for r in rows for n in r.get("injected") or [] for p in n["parts"]]
        assert "package" not in parts and "rules" not in parts
        b = _main_bodies(srv)[-1]
        assert b.get("thinking") in (None, {"type": "disabled"})
    srv.shutdown()


def _gated_profile() -> dict:
    """`X-Tandem-Profile: gated`: the coding profile with its tool gate on (the shipped profiles
    hold no calls, as on the engine), so C4 holds and acts."""
    cond = proxy_app.STATE["conductor"]
    _, doc = cond.store.peek("profiles", "coding")
    doc = json.loads(json.dumps(doc))
    doc["toolgate"].update(risk=True, project=True, same_call=True, malformed=True)
    doc["clients"] = []
    cond.store.put("profiles", "gated", doc, note="test", author="test")
    return dict(UA, **{"X-Tandem-Profile": "gated"})


def test_regenerate_bad_call_and_repair_almost_json():
    srv, port = fake(models=["m1"])
    with Proxy(env=_env("vllm", "on", port), conductor=True) as px:
        px.sync()
        hdr = _gated_profile()
        st, _, out = px.post("/v1/chat/completions", {
            "model": "m1", "messages": [SYSTEM, {"role": "user", "content":
                                                 "Read it FAKE_BAD_CALL FAKE_CLASS code_edit"}],
            "tools": TOOLS, "stream": True}, headers=hdr)
        fr = _frames(out)
        args = "".join(((c.get("function") or {}).get("arguments") or "")
                       for f in fr if f != "[DONE]" for ch in f.get("choices") or []
                       for c in (ch.get("delta") or {}).get("tool_calls") or [])
        assert "oops" not in args                    # the client never saw the bad call
        content, _ = _stream_text(fr)
        assert "done0" in content and fr.count("[DONE]") == 1
        regen = [e for e in _events("C4") if e["decision"] == "REGENERATE_CALL"]
        assert regen and regen[0]["mode"] == "act"
        mains = _main_bodies(srv)
        assert mains[-1]["messages"][-1]["role"] == "tool"
        st, _, out = px.post("/v1/chat/completions", {
            "model": "m1", "messages": [SYSTEM, {"role": "user", "content":
                                                 "Read it FAKE_ALMOST_JSON FAKE_CLASS code_edit"}],
            "tools": TOOLS, "stream": True}, headers=hdr)
        fr = _frames(out)
        calls = [c for f in fr if f != "[DONE]" for ch in f.get("choices") or []
                 for c in (ch.get("delta") or {}).get("tool_calls") or []]
        args = "".join((c.get("function") or {}).get("arguments") or "" for c in calls)
        assert json.loads(args) == {"path": "a.txt"}
    srv.shutdown()


def test_argument_guard_stops_the_stream_and_notes_the_next_turn():
    srv, port = fake(models=["m1"])
    msgs = [SYSTEM, {"role": "user", "content": "Write the map FAKE_ARGLOOP FAKE_CLASS code_edit"}]
    with Proxy(env=_env("vllm", "on", port), conductor=True) as px:
        px.sync()
        st, _, out = px.post("/v1/chat/completions", {"model": "m1", "messages": msgs,
                                                      "tools": TOOLS, "stream": True},
                             headers=UA)
        fr = _frames(out)
        content, _ = _stream_text(fr)
        assert "TandemLLM stopped the `write` call" in content
        fins = [ch.get("finish_reason") for f in fr if f != "[DONE]"
                for ch in f.get("choices") or [] if ch.get("finish_reason")]
        assert fins == ["stop"] and fr[-1] == "[DONE]"
        a1 = _events("A1")
        assert a1 and a1[0]["decision"] == "GIVE_UP" and a1[0]["mode"] == "act"
        msgs2 = msgs + [{"role": "assistant", "content": content},
                        {"role": "user", "content": "go on FAKE_CLASS code_edit"}]
        px.post("/v1/chat/completions", {"model": "m1", "messages": msgs2, "tools": TOOLS,
                                         "stream": True}, headers=UA)
        last = _main_bodies(srv)[-1]["messages"][-1]["content"]
        assert "Runtime note: your last tool call was stopped" in last
    srv.shutdown()


def _calibrated(monkey=True):
    cfg = {"calibrated": True, "loop": {"min_size": 1, "max_size": 12, "min_count": 4,
                                        "min_span": 80},
           "repeat": {"gram": 8, "window": 100, "every": 25, "min_words": 200, "act": 0.85,
                      "grey": 0.4, "in_a_row": 2},
           "novelty": {"window": 60, "gram": 5, "min_words": 120, "below": 0.05}}
    old = tier0_words.words_cfg
    tier0_words.words_cfg = lambda doc: dict(cfg)
    return old


def test_d1_shadow_where_the_backend_cannot_continue():
    srv, port = fake(models=["gpt-5.4-mini"])
    old = _calibrated()
    try:
        with Proxy(env=_env("openai", "on", port, bid="oa",
                            extra={"TANDEM_BACKEND_OA_MODELS": "gpt-5.4-mini",
                                   "TANDEM_BACKEND_OA_MODELS_MODE": "list"}),
                   conductor=True) as px:
            px.sync()
            st, _, out = px.post("/v1/chat/completions", {
                "model": "gpt-5.4-mini", "messages": [SYSTEM, {"role": "user", "content":
                                                               "Plan it FAKE_LOOP"}],
                "tools": TOOLS, "stream": True}, headers=UA)
            assert st == 200
            d1 = _events("D1")
            assert d1 and d1[0]["mode"] == "shadow" and \
                d1[0]["detail"]["shadow_why"] == "no_continue_final_message"
            assert len(_main_bodies(srv)) == 1        # the stream untouched, no second request
            _, reasoning = _stream_text(_frames(out))
            assert reasoning.count("I should check the map again") == 120
    finally:
        tier0_words.words_cfg = old
    srv.shutdown()


def test_d1_closed_by_a_continuation_on_vllm():
    srv, port = fake(models=["m1"])
    old = _calibrated()
    try:
        with Proxy(env=_env("vllm", "on", port), conductor=True) as px:
            px.sync()
            st, _, out = px.post("/v1/chat/completions", {
                "model": "m1", "messages": [SYSTEM, {"role": "user", "content":
                                                     "Plan it FAKE_LOOP FAKE_CLASS code_edit"}],
                "tools": TOOLS, "stream": True}, headers=UA)
            fr = _frames(out)
            content, reasoning = _stream_text(fr)
            assert "closed0" in content and fr.count("[DONE]") == 1
            assert reasoning.count("I should check the map again") < 120
            d1 = _events("D1")
            assert d1 and d1[0]["mode"] == "act"
            cont = _main_bodies(srv)[-1]
            assert cont["continue_final_message"] is True and \
                cont["messages"][-1]["role"] == "assistant" and \
                "</think>" in cont["messages"][-1]["content"]
            assert len({f["id"] for f in fr if f != "[DONE]"}) == 1
    finally:
        tier0_words.words_cfg = old
    srv.shutdown()


def test_side_request_capped_and_compaction_section():
    srv, port = fake(models=["m1"])
    with Proxy(env=_env("vllm", "on", port), conductor=True) as px:
        px.sync()
        title = {"model": "m1", "max_tokens": 4000, "stream": False, "messages": [
            {"role": "system", "content": "You are a title generator. You output ONLY a thread "
                                          "title. Nothing else.\nRules..."},
            {"role": "user", "content": "Fix the failing test"}]}
        st, _, _ = px.post("/v1/chat/completions", title, headers=UA)
        b = srv.fake.last["body"]
        assert st == 200 and b["max_tokens"] == 64 and \
            b["chat_template_kwargs"] == {"enable_thinking": False}
        hist = "user: build a game\nassistant: done " * 200
        comp = {"model": "m1", "stream": False, "messages": [
            {"role": "system", "content": "You are a context summarization agent. You are given "
                                          "a conversation between a user and an agent."},
            {"role": "user", "content": hist}]}
        st, _, _ = px.post("/v1/chat/completions", comp, headers=UA)
        b = srv.fake.last["body"]
        assert "TandemLLM" in b["messages"][-1]["content"] and \
            "this session's runtime address" in b["messages"][-1]["content"]
        rows = _journal().rows(limit=5)
        assert any((r.get("extra") or {}).get("compaction", {}).get("history") ==
                   "client_history" for r in rows)
    srv.shutdown()


def test_backends_api_shows_effective_switches_and_degradations():
    srv, port = fake(models=["m1"])
    with Proxy(env=_env("vllm", "on", port), conductor=True) as px:
        px.sync()
        st, _, out = px.get("/v1/config")
        d = json.loads(out)
        b = d["backends"][0]
        m = b["model_rows"][0]
        assert m["effective"]["packages"] == "on" and m["effective"]["mode"] == "on"
        assert set(b["degradations"]) == set(harness.GROUPS)
        assert d["harness"]["conductor"] is True
    srv.shutdown()


def test_library_and_session_address_on_the_standalone_proxy():
    srv, port = fake(models=["m1"])
    with Proxy(env=_env("vllm", "on", port), conductor=True) as px:
        st, _, out = px.get("/v1/library")
        assert st in (200, 404)                       # 404: no library installed in the temp dir
        tok = proxy_app.STATE["session_tokens"].token_for("c:test-session")
        st2, _, out2 = px.get(f"/s/{tok}/v1/library")
        assert st2 == st
        st3, _, _ = px.get(f"/s/{tok}/v1/models")
        assert st3 == 404
    srv.shutdown()


def test_memory_blocks_stay_on_the_proxy_and_the_notes_go_in():
    """2026-10-05: the `memory` group with a proxy user: the model's block is taken out of the
    stream and written under the user's name; the next request carries the card and the note; a
    request without a user writes nothing."""
    srv, port = fake(models=["m1"])
    with Proxy(env=_env("vllm", "on", port), conductor=True) as px:
        px.sync()
        cond = proxy_app.STATE["conductor"]
        v, prof = cond.store.get("profiles", "coding")
        cond.store.put("profiles", "coding", dict(prof, memory="on"), note="memory",
                       author="test", base=v)
        st, _, body = px.request("POST", "/v1/config/users", {"label": "laptop"})
        key = json.loads(body)["key"]
        hdr = dict(UA, Authorization=f"Bearer {key}")
        u1 = {"role": "user", "content": "Keep answers short from now on. FAKE_MEMORY"}
        st, _, out = px.post("/v1/chat/completions", {"model": "m1", "messages": [SYSTEM, u1],
                                                      "stream": True}, headers=hdr)
        assert st == 200
        text, _ = _stream_text(_frames(out))
        assert "<tandem_memory" not in text and text.strip(), text
        ns = cond.notes.notes("laptop")
        assert [(n.title, n.scope, n.writer) for n in ns] == [("short answers", "user",
                                                               "qwen-inline")], ns
        b1 = _main_bodies(srv)[-1]
        assert "<tandem_memory op=" in b1["messages"][-1]["content"]      # the card went in
        u2 = {"role": "user", "content": "How short should my answers be, in sentences?"}
        px.post("/v1/chat/completions", {"model": "m1", "stream": True, "messages": [
            SYSTEM, {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
            u2]}, headers=hdr)
        b2 = _main_bodies(srv)[-1]
        assert "at most five sentences" in b2["messages"][-1]["content"], b2["messages"][-1]
        # no user (the loopback client without a key): nothing written
        px.post("/v1/chat/completions", {"model": "m1", "messages": [
            SYSTEM, {"role": "user", "content": "Remember the port. FAKE_MEMORY"}],
            "stream": True}, headers=UA)
        assert len(cond.notes.notes("laptop")) == 1 and cond.notes.users() == ["laptop"]
    srv.shutdown()


if __name__ == "__main__":
    run(globals())
