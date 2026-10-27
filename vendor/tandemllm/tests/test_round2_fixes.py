"""The UT2 round-2 fixes through the real handler with the fake engine in process,
and their pure parts. Needs CPU torch (the box's CPU suite). QA tickets: SRV-186, SRV-188,
SRV-190, SRV-192, VIS-59, VIS-61, OPS-68 (SRV-180/182: tests/test_turn_budget.py).

The test matrix, from the tickets' Gherkin:

  SRV-185  a memory write is visible to the person who asked
    1  an Open WebUI stream (`tags`): the line `› remembered: ...` is in the content, after the
       answer; with `reasoning_content` too; a non-streamed answer the same; X-Tandem-Memory: off
       writes nothing and shows no line          test_memory_line_*
  SRV-187  reasoning tags never leak
    2  a bare `</think>` line in the answer is not sent (stream and non-stream); the tag in a
       sentence, in code, in a tool call stays; the raw profile is untouched
                                                 test_stray_close_*
  SRV-189  prose requests are routed as chat or writing
    3  the stage's distribution holds every class; the essay at 0.57 code_edit falls to the best
       non-code class (journaled); a request with a code cue or a confident code answer keeps it
                                                 test_class_distribution, test_code_guard_*
  SRV-191  the model says what it is and where it runs
    4  a chat turn carries the identity note once; the next turn replays it at the same place;
       raw adds nothing; /v1/models names the model; QWEN38_MODEL_NAME overrides it
                                                 test_identity_*
  VIS-58   live numbers agree with the ledger
    5  cached never above prompt in the live row, hidden tokens their own figure
                                                 test_live_cached_is_clamped
  VIS-60   usage counts what the user sent
    6  the router's call is `internal`, names its parent, is left out of usage, summary,
       requests and the filters by default and listed with internal=1; schema 6 relabels the
       old rows once                             test_internal_*
  OPS-67   Open WebUI during an engine restart
    7  the responder answers /v1/models with 200 and the last list (a fixed entry without one),
       a card for a listed id, 404 for another, 503 for a chat; `set up` keeps the list
                                                 test_status_models_*
  SRV-181  the testing package texts (the migration applied, idempotent)
                                                 test_package_texts_round2

Run: python tests/test_round2_fixes.py [name-filter]
"""

from __future__ import annotations

import http.server
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_ui_states import TOOLS, _go, lines, opencode, serve  # noqa: E402  (first: the CPU environment)
from test_memory_router import setup as mem_setup  # noqa: E402

from server import app, dashboard_api as D, ledger, live as live_mod, usage  # noqa: E402
from server.conductor import schema  # noqa: E402
from server.conductor.classify import Answer, distribution  # noqa: E402
from server.conductor.router import _code_guard  # noqa: E402
from server.stream import StrayClose, drop_stray_close  # noqa: E402

OWUI_UA = "Python/3.11 aiohttp/3.12.15"
SDK_UA = "OpenAI/Python 1.109.1"


def owui(text, stream=True, fmt=None, tools=False, headers=None, history=()):
    body = {"model": "TandemLLM", "stream": stream, "max_tokens": 2000,
            "messages": [{"role": "system", "content": "You are a helpful assistant."}, *history,
                         {"role": "user", "content": text}]}
    if tools:
        body["tools"] = TOOLS
    if fmt:
        body["reasoning_format"] = fmt
    return _go(body, {"User-Agent": OWUI_UA, **(headers or {})})


# ------------------------------------------------------------------ SRV-185

def test_memory_line_in_an_open_webui_stream():
    cond, _ = mem_setup()
    out = owui("Please keep it short. FAKE_MEMORY")
    assert len(cond.notes.notes("owner")) == 1
    i = out["content"].find("› remembered: short answers")
    assert i >= 0, out["content"][-300:]
    # SRV-193: Open WebUI gets its reasoning in `reasoning_content`, the line in the answer
    assert "think>" not in out["content"] and "remembered" not in out["reasoning"]
    assert "<tandem_memory" not in out["content"]
    tags = owui("Please keep it short. FAKE_MEMORY", fmt="tags")     # the client's own format
    i = tags["content"].find("› remembered: short answers")
    assert 0 < tags["content"].rfind("</think>") < i       # after the block, in the answer


def test_memory_line_with_reasoning_content():
    cond, _ = mem_setup()
    out = owui("Please keep it short. FAKE_MEMORY", fmt="reasoning_content")
    assert "remembered: short answers" in lines(out["content"]), out["content"][-200:]
    assert "remembered" not in out["reasoning"]


def test_memory_line_in_a_non_streamed_answer():
    cond, _ = mem_setup()
    out = owui("Please keep it short. FAKE_MEMORY", stream=False)
    assert out["content"].rstrip().endswith("› remembered: short answers"), out["content"][-200:]
    assert "think>" not in out["content"] and "remembered" not in (out["reasoning"] or "")


def test_memory_line_not_shown_when_memory_is_off():
    cond, _ = mem_setup()
    out = owui("Please keep it short. FAKE_MEMORY", headers={"X-Tandem-Memory": "off"})
    assert cond.notes.notes("owner") == []
    assert "remembered" not in out["content"] + out["reasoning"]


# ------------------------------------------------------------------ SRV-187

def test_stray_close_filter():
    cases = {"Hello\n</think>\n\nWorld": ("Hello\n\nWorld", 1),
             "a </think> b": ("a </think> b", 0),
             "```\n</think>\n```\n": ("```\n</think>\n```\n", 0),
             "`</think>`\n": ("`</think>`\n", 0),
             "<tool_call>\n<function=w>\n<parameter=c>\n</think>\n</parameter>\n</function>\n"
             "</tool_call>\n": (None, 0),
             "Plan:\n<think>\nx\n</think>\nok\n": (None, 0),
             "line\n</think>": ("line\n", 1)}
    for text, (want, n) in cases.items():
        got, k = drop_stray_close(text)
        assert (got if want is not None else text) == got and k == n, (text, got, k)
        f = StrayClose()
        streamed = "".join(f.push(ch) for ch in text) + f.finish()
        assert streamed == got, (text, streamed)
    assert drop_stray_close("<think>\nhmm\n</think>\n\nA\n</think>\nB", "tags", True) == \
        ("<think>\nhmm\n</think>\n\nA\nB", 1)


def test_stray_close_in_the_stream_and_untouched_raw():
    serve("on")
    out = owui("FAKE_STRAY_CLOSE use the tool please", tools=True, fmt="tags")
    body = out["content"]
    assert body.count("</think>") == 1, body               # the block's own close only
    assert "Next I run it." in body and out["calls"], out
    # SRV-193: Open WebUI's own format (reasoning_content): no tag in the content at all
    own = owui("FAKE_STRAY_CLOSE use the tool please", tools=True)
    assert "think>" not in own["content"] and "Next I run it." in own["content"], own["content"]
    raw = owui("FAKE_STRAY_CLOSE use the tool please", tools=True,
               headers={"X-Tandem-Profile": "raw"})
    assert raw["content"].count("</think>") == 2, raw["content"]      # the release's shape
    full = owui("FAKE_STRAY_CLOSE plain", stream=False, fmt="tags")
    assert full["content"].count("</think>") == 1, full["content"]
    full = owui("FAKE_STRAY_CLOSE plain", stream=False)
    assert "think>" not in full["content"], full["content"]


# ------------------------------------------------------------------ SRV-189

def test_class_distribution():
    toks = [{"token": "code", "logprob": -0.5604,
             "top_logprobs": [{"token": "code", "logprob": -0.5604},
                              {"token": "chat", "logprob": -1.2248},
                              {"token": "writing", "logprob": -2.2662},
                              {"token": "ch", "logprob": -5.2},
                              {"token": "d", "logprob": -4.0}]},
            {"token": "_edit", "logprob": -0.0004, "top_logprobs": []}]
    ids = ["code_edit", "debug", "design_ui", "writing", "chat"]
    p = distribution(toks, ids, "code_edit", 0.5708)
    assert p["code_edit"] == 0.5708 and abs(p["chat"] - (0.2938 + 0.0055)) < 0.002, p
    assert abs(p["writing"] - 0.1037) < 0.001
    assert abs(p["debug"] - p["design_ui"]) < 1e-9 and p["debug"] > 0     # `d`, shared


def test_code_guard_sends_the_essay_away_from_code():
    opts = {c["id"]: "" for c in schema.SHIPPED_ROUTER["classes"]}
    probs = {"code_edit": 0.5708, "chat": 0.2993, "writing": 0.1037, "research": 0.0128}
    essay = "Write a detailed 700-word essay about the history of tides and how people measured them."
    g = _code_guard(Answer("qwen", "code_edit", 0.5708, probs), essay, opts)
    assert g.label == "chat" and g.guard == {"was": "code_edit", "confidence": 0.5708,
                                             "to": "chat", "why": "no_code_cue"}, g.as_dict()
    for text, conf in (("Write a Mario game very classic on with Mobil view html", 0.57),
                       ("Write an essay generator in Python", 0.6),
                       ("Write a detailed essay about tides", 0.75),
                       ("Make a spirograph drawing generator with sliders", 0.55)):
        a = _code_guard(Answer("qwen", "code_edit", conf, dict(probs, code_edit=conf)), text, opts)
        assert a.label == "code_edit" and a.guard is None, (text, a.as_dict())


# ------------------------------------------------------------------ SRV-191

def _prompt_text(seen):
    return app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)


def test_identity_note_once_and_replayed():
    cond, seen = serve("on")
    hist = []
    a = owui("Who are you?")
    p1 = list(seen[-1])
    t1 = _prompt_text(seen)
    assert "you are TandemLLM, running the Qwen3.8-27B model on the user's own machine" in t1
    assert t1.count("you are TandemLLM, running") == 1
    hist = [{"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": a["content"].split("</think>")[-1].strip()}]
    owui("And what can you do?", history=hist)
    p2 = list(seen[-1])
    t2 = _prompt_text(seen)
    assert t2.count("you are TandemLLM, running") == 1, "replayed once, not added again"
    k = t1.index("you are TandemLLM, running")
    assert t2[:k] == t1[:k]                                  # the note sits where it sat
    end = t1.index("<|im_end|>", k)
    n = len(app.STATE["tok"](t1[:end], add_special_tokens=False).input_ids)
    assert p2[:n] == p1[:n], "the prefix through the note matches the first turn's"


def test_identity_note_absent_under_raw():
    serve("on")
    seen_before = []
    _, seen = serve("on")
    owui("Who are you?", headers={"X-Tandem-Profile": "raw"})
    assert "you are TandemLLM, running" not in _prompt_text(seen)


def test_identity_model_name():
    assert schema.identity_text().startswith(
        "About you: you are TandemLLM, running the Qwen3.8-27B model on the user's own machine.")
    os.environ["QWEN38_MODEL_NAME"] = "Qwen3.8-27B-FP8"
    try:
        assert "you are TandemLLM, running the Qwen3.8-27B-FP8 model" in schema.identity_text()
        cond, _ = serve("on")
        card = cond.models("qwen38-spark-engine", 0)[0]
        assert card["description"].startswith("Qwen3.8-27B-FP8 with the conductor")
    finally:
        del os.environ["QWEN38_MODEL_NAME"]


def test_identity_note_for_every_client():
    """0.8.2: every client the conductor serves gets the note on the turn that asks: Open WebUI
    (with its tools), opencode, the Chat tab (chatd, `chat-app`) and a plain SDK client (the
    default profile)."""
    sends = {
        "open-webui": lambda: owui("Who are you?", tools=True),
        "opencode": lambda: opencode("Who are you?"),
        "chat-app": lambda: _go({"model": "TandemLLM", "stream": True, "max_tokens": 2000,
                                 "messages": [{"role": "user", "content": "Who are you?"}]},
                                {"X-QSE-Client": "chat-app", "x-session-id": "chat-id082"}),
        "openai-sdk": lambda: _go({"model": "TandemLLM", "stream": False, "max_tokens": 2000,
                                   "messages": [{"role": "user", "content": "who are you"}]},
                                  {"User-Agent": SDK_UA}),
    }
    for client, send in sends.items():
        _, seen = serve("on")
        send()
        assert "you are TandemLLM, running" in _prompt_text(seen), client


def test_identity_names_the_served_pack_and_no_other_maker():
    """0.8.2: on a Flash-Next pack the note names Qwen3.8-Flash-Next (its model card's label), not
    the 27B; it says TandemLLM on the user's own machine and that it is not made by Anthropic or
    OpenAI (Flash-Next, served raw, said "I'm Fable, an AI made by Anthropic", 2026-10-13)."""
    was = (os.environ.get("QSE_WEIGHTS"), schema.SERVED_NAME)
    try:
        for pack, label in (("flashnext-3.05", "Qwen3.8-Flash-Next"), ("exl3-4.5", "Qwen3.8-27B"),
                            ("nvfp4", "Qwen3.8-27B")):
            os.environ["QSE_WEIGHTS"] = pack
            assert app._identity_name() == label, pack
        os.environ["QSE_WEIGHTS"] = "no-such-pack"
        assert app._identity_name() is None
        schema.SERVED_NAME = "Qwen3.8-Flash-Next"
        t = schema.identity_text()
        assert t.startswith("About you: you are TandemLLM, running the Qwen3.8-Flash-Next model "
                            "on the user's own machine."), t
        assert "not made by Anthropic or OpenAI" in t and "Qwen3.8-27B" not in t
        assert "you are TandemLLM running Qwen3.8-Flash-Next on their own machine" in t
        # the env name still wins (a box that serves another checkpoint)
        os.environ["QWEN38_MODEL_NAME"] = "Qwen3.8-Other"
        try:
            assert "running the Qwen3.8-Other model" in schema.identity_text()
        finally:
            del os.environ["QWEN38_MODEL_NAME"]
        # and the note reaches an Open WebUI turn that asks, with the served name
        serve("on")
        _, seen = serve("on")
        owui("WHO are you?")
        assert "you are TandemLLM, running the Qwen3.8-Flash-Next model" in _prompt_text(seen)
    finally:
        if was[0] is None:
            os.environ.pop("QSE_WEIGHTS", None)
        else:
            os.environ["QSE_WEIGHTS"] = was[0]
        schema.SERVED_NAME = was[1]


# ------------------------------------------------------------------ VIS-58

def test_live_cached_is_clamped():
    rec = usage.RequestRecord("r1", "chat", True)
    rec.prompt_tokens, rec.cached_tokens, rec.hidden_prompt = 15957, 19456, 4625
    rec.prompt_n, rec.t_first = 1499, 1.0
    got = live_mod.LiveRegistry._cache_of(rec)
    assert got["cached_tokens"] == 15957 and got["hidden_tokens"] == 4625, got
    rec.hidden_prompt = 0
    # no notes: the cached rows are the client's, and never more than its prompt (2026-10-09:
    # the one definition, usage.client_cached_of; the engine's figure stays in timings.cache_n)
    assert live_mod.LiveRegistry._cache_of(rec)["cached_tokens"] == 15957
    rec.cached_tokens = 12000
    assert live_mod.LiveRegistry._cache_of(rec)["cached_tokens"] == 12000


# ------------------------------------------------------------------ VIS-60

def _ledger(rows):
    led = ledger.Ledger(os.path.join(tempfile.mkdtemp(prefix="qse-int-"), "l.sqlite3")).open()
    base = usage.RequestRecord("x", "chat", True).row("0.1.0", "c0de")
    for i, r in enumerate(rows):
        row = dict(base, request_id=f"r{i}", model="TandemLLM", client_id="k:a",
                   client_kind="open-webui", status=200, finish_reason="stop", prompt_tokens=100,
                   cached_tokens=0, completion_tokens=50, reasoning_tokens=0, total_ms=1000.0,
                   ts_ms=int(1_791_270_000_000 + i * 1000))
        row.update(r)
        led.submit(row)
    assert led.flush(5.0)
    return led


def test_internal_kind_and_parent():
    assert ledger.client_of({"User-Agent": "tandem-conductor/1"})[1] == "internal"
    assert ledger.client_of({"User-Agent": "Python/3.11 aiohttp/3.12.15"})[1] == "open-webui"
    assert "internal" in ledger.KINDS and "parent_request" in ledger.COLUMNS


def test_internal_rows_left_out_by_default():
    led = _ledger([{}, {}, {"client_kind": "internal", "model": "qwen38-spark-engine",
                            "client_id": "anon", "prompt_tokens": 405, "completion_tokens": 2,
                            "parent_request": "r1"}])
    api = D.DashboardAPI(led, clock=lambda: 1_791_280_000, cache_ttl=0)
    q = {"from": "2026-10-01", "to": "2026-10-08", "tz": "UTC"}
    u = api.usage(dict(q))
    assert u["totals"]["requests"] == 2, u["totals"]
    assert u["dimensions"]["models"] == ["TandemLLM"], u["dimensions"]
    assert [c["id"] for c in u["dimensions"]["clients"]] == ["k:a"]
    ui = api.usage(dict(q, internal="1"))
    assert ui["totals"]["requests"] == 3 and ui["filters"]["internal"] is True
    assert set(ui["dimensions"]["models"]) == {"TandemLLM", "qwen38-spark-engine"}
    assert len(api.requests({"limit": 10})["requests"]) == 2
    rows = api.requests({"limit": 10, "internal": "1"})["requests"]
    assert rows[0]["client"]["kind"] == "internal" and rows[0]["parent_request"] == "r1"
    assert "parent_request" not in rows[1]
    s = api.summary({"tz": "UTC"})
    assert s["windows"]["all"]["requests"] == 2


def test_internal_relabel_on_schema_6():
    path = os.path.join(tempfile.mkdtemp(prefix="qse-mig-"), "l.sqlite3")
    led = _ledger([])
    led.close()
    conn = sqlite3.connect(led.path)
    for kind, model, mx, out in (("other", "qwen38-spark-engine", 24, 2),
                                 ("other", "qwen38-spark-engine", 32000, 900),
                                 ("other", "TandemLLM", 24, 2), ("open-webui", "x", 24, 1)):
        r = dict(ts_ms=1, request_id="q", model=model, client_id="anon", client_kind=kind,
                 endpoint="chat", stream=0, status=200, max_tokens=mx, completion_tokens=out)
        conn.execute(f"INSERT INTO requests({','.join(r)}) VALUES ({','.join('?' * len(r))})",
                     list(r.values()))
    conn.execute("UPDATE meta SET value='5' WHERE key='schema_version'")
    conn.execute("DELETE FROM meta WHERE key='relabel_internal'")
    conn.commit()
    assert ledger.migrate(conn) == 5
    kinds = [r[0] for r in conn.execute("SELECT client_kind FROM requests ORDER BY id")]
    assert kinds == ["internal", "other", "other", "open-webui"], kinds
    assert conn.execute("SELECT value FROM meta WHERE key='relabel_internal'").fetchone()[0] == "1"
    assert ledger.migrate(conn) == ledger.SCHEMA_VERSION   # once
    del path


# ------------------------------------------------------------------ OPS-67

def _responder(status: dict):
    sys.path.insert(0, os.path.join(ROOT, "ops"))
    import engine_status as es
    d = tempfile.mkdtemp(prefix="qse-status-")
    p = os.path.join(d, "engine-status.json")
    with open(p, "w") as f:
        json.dump(status, f)
    es.Handler.status_file = p
    es.Handler.engine_health = "http://127.0.0.1:9/health"        # nothing answers there
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), es.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return es, srv, p


def _get(port, path, method="GET", body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_status_models_during_a_hold():
    models = [{"id": "TandemLLM", "object": "model", "owned_by": "tandemllm"}]
    # a start under way (OPS-173: a start far past the typical one reads "not answering")
    es, srv, _ = _responder({"state": "starting", "since": int(time.time()) - 10, "models": models,
                             "starts": [{"at": 1, "s": 33}]})
    try:
        port = srv.server_address[1]
        st, body = _get(port, "/v1/models")
        assert st == 200 and [m["id"] for m in body["data"]] == ["TandemLLM"], body
        assert body["tandem_engine"]["state"] == "starting"
        assert "TandemLLM is starting" in body["tandem_engine"]["message"]
        st, card = _get(port, "/v1/models/TandemLLM")
        assert st == 200 and card["id"] == "TandemLLM"
        st, nf = _get(port, "/v1/models/gpt-9")
        assert st == 404 and nf["error"]["code"] == "model_not_found"
        st, err = _get(port, "/v1/chat/completions", "POST", {"model": "TandemLLM"})
        assert st == 503 and err["error"]["code"] == "engine_starting", err   # OPS-65 unchanged
    finally:
        srv.shutdown()


def test_status_models_without_a_recorded_list():
    es, srv, _ = _responder({"state": "held", "since": 1_791_270_000, "reason": "deploy"})
    try:
        st, body = _get(srv.server_address[1], "/v1/models")
        assert st == 200 and body["data"][0]["id"] == "TandemLLM", body
    finally:
        srv.shutdown()


def test_status_set_up_keeps_the_list():
    es, srv, p = _responder({})
    srv.shutdown()
    models = [{"id": "TandemLLM", "object": "model"}]
    es.write("up", p=p, models=models)
    assert es.read(p)["models"] == models
    es.write("held", p=p, reason="deploy")
    assert es.read(p)["models"] == models                   # every later state keeps it


# ------------------------------------------------------------------ SRV-181

def test_package_texts_round2():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import packages_migrate as pm
    with open(os.path.join(ROOT, "packages", "inline", "testing.md")) as f:
        testing = f.read()
    assert "Order and budget" in testing and "First the deliverable" in testing
    assert "timeout 600 python3 -m http.server" in testing
    assert testing.count("Order and budget") == 1 and "never /tmp" in testing
    with open(os.path.join(ROOT, "packages", "inline", "laya-playtest.md")) as f:
        play = f.read()
    assert "A spot where only the bot fails is not a bug" in play
    assert 'before "done":' not in play
    for name in ("testing.md", "laya-playtest.md", "test-web.md"):
        with open(os.path.join(ROOT, "packages", "inline", name)) as f:
            text = f.read()
        assert pm.migrate_text(name, text) == (text, []), name       # idempotent
    # a box before edit 5 (edits 9-10 of 0.3.45 reword two of its lines: undone first)
    # (and edits 15-17 of 0.3.51 reword the hook's end and the final line: undone too)
    shipped = testing
    # (and edit 30 of 0.4.5 adds the progress-lines bullet after the quirk rule: undone too)
    # (and edit 34 of 0.4.7 rewords the scratch rule and the leftover rule: undone too)
    testing = testing.replace(pm.SCRATCH_RULE_V3, pm.SCRATCH_RULE).replace(pm.LEFTOVER_NEW,
                                                                           pm.LEFTOVER_OLD)
    testing = (testing.replace("\n- " + pm.PROGRESS_LINE, "").replace("\n" + pm.TOOLS_QUIRK_RULE, "")
               .replace(pm.FINISH_NEW + "\n" + pm.OVERLAY_RULE, pm.FINISH_OLD)
               .replace(pm.STATE_SAY_NEW, pm.STATE_SAY_OLD))
    r351 = ["testing: the state hook goes without a word", "testing: the final message is for the user",
            "testing: check-only overlays go before done", "testing: test-tool quirks stay out",
            "all text: plans and progress lines name no Laya",
            "testing: scratch marked, deleted only when made", "testing: a leftover is reused, never deleted"]
    v43 = (testing.replace(pm.CHECK_ONE_NEW, pm.CHECK_ONE_OLD).replace(pm.QUOTE_NEW, pm.QUOTE_OLD)
           .replace(pm.PAGE_CMD_NEW, pm.PAGE_CMD_OLD)
           .replace(pm.VERIFY_NEW, pm.VERIFY_OLD)
           .replace(pm.RUNTIME_NEW_PAGE, pm.RUNTIME_OLD_PAGE))
    old = v43.replace("\n" + pm.TMP_RULE, "").replace(pm.BUDGET, "")
    assert "Order and budget" not in old
    new, ch = pm.migrate_text("testing", old)
    assert ch == ["testing: order and budget", "testing: nothing outside the project",
                  "testing: the kit's page check, no own driver, check-fix-check",
                  "testing: a page runs in the page check",
                  "testing: one page check with every control, no test copies",
                  "testing: the page check asks Laya at {tandem_url}",
                  "testing: a page-check step in single quotes"] + r351, ch
    assert new.splitlines() == shipped.splitlines()        # the shipped text, line for line
    assert new.count("Order and budget") == 1 and pm.migrate_text("testing", new)[1] == []
    five = testing.replace("\n" + pm.TMP_RULE, "")     # a box that has edit 5 and not 8
    assert pm.migrate_text("testing", five) == (shipped, ["testing: nothing outside the project"] + r351)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and only in name:
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
