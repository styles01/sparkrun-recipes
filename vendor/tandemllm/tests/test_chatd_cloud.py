"""A chat on a cloud model through the proxy: chatd's figures come from the stream itself.

THE CASE (live, 2026-10-17, 0.11.3): the pagoda prompt on "DeepSeek V4 Flash (cloud)". The
proxy's live row of a relayed call counts no tokens (proxy/measure.py: a frame is not a token), and
chatd took its zeros over its own counts: the Context panel read 0 thinking through a minute of
thinking, "not tracked for this model" for tok/s, "nothing cached yet" beside DeepSeek's 46,208
cache hits, and no cost although the proxy priced every call in the ledger.

  1  live: tok/s, the thinking tokens and the first token's wait from chatd's own stamps of the
     stream (one token a text chunk, a call's arguments by their characters)
  2  after each call: the provider's usage (DeepSeek's prompt_cache_hit_tokens as the cached
     tokens, source `provider`), tok/s = (completion - 1) / the stream's span, and the proxy's
     price (`: tandem-cost`, X-Tandem-Cost: 1) as the message's cost_usd and the chat's usage
  3  the Context panel: the system prompt and the tool schemas as chatd's estimate (`estimated`),
     the window from the proxy's card, the answer's room as the request asks for it

CPU only: chatd -> the standalone proxy (tests/proxy_util.py) -> tests/fake_backend.py as a
DeepSeek backend. Run: python tests/test_chatd_cloud.py
"""

from __future__ import annotations

import json
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatd_util import Chat, is_done, run_all  # noqa: E402
from proxy_util import FIXTURE_KEYS, Proxy, fake  # noqa: E402

from chatd import agent, context as ctx_mod, render, runs  # noqa: E402
from test_chatd_attach import upload  # noqa: E402

PRICE = (0.30, 0.006, 1.2)      # DeepSeek V4 Flash's list price, USD per million tokens


def _env(port: int) -> dict:
    return {"TANDEM_BACKENDS": "ds", "TANDEM_BACKEND_DS_KIND": "deepseek",
            "TANDEM_BACKEND_DS_URL": f"http://127.0.0.1:{port}",
            "TANDEM_BACKEND_DS_KEY": FIXTURE_KEYS[2],
            "TANDEM_BACKEND_DS_MODELS": "deepseek-flash",
            "TANDEM_BACKEND_DS_MODELS_MODE": "list",
            "TANDEM_PRICE_1": "ds/deepseek-flash=" + ",".join(str(p) for p in PRICE)}


def test_cloud_turn_measured_from_the_stream_priced_by_the_proxy():
    srv, port = fake(models=["deepseek-flash"], tps=150)
    with Proxy(_env(port)) as px:
        px.sync()
        # the window DeepSeek's model list gives (1,048,576), as discovery would store it
        px.svc.cfg.sync_models("ds", [{"id": "deepseek-flash", "context_length": 1_048_576}],
                               ["deepseek-flash"], "discover")
        c = Chat(engine_url=f"http://127.0.0.1:{px.port}")
        try:
            cid = c.new()
            c.ok("PATCH", f"/conversations/{cid}/settings", {"model": "deepseek-flash"})
            sse = c.events(cid)
            sse.until(lambda f: f[1] == "current")
            c.send(cid, "pagoda FAKE_DEEPSEEK FAKE_CACHED 9 FAKE_REASONING FAKE_LONG 300")
            frames = sse.until(is_done, 30)
            sse.close()
            snap = c.wait_idle(cid)
            # 1  live figures: chatd's counts, never the proxy row's zeros
            live = [json.loads(f[2]) for f in frames if f[1] == "run.metrics"]
            assert live and max(m["tokens"] for m in live) > 50
            assert any(m["thinking_used"] > 0 for m in live)
            assert any(m["decode_tps"] for m in live), live[-3:]
            assert any(m.get("ttft_ms") is not None for m in live)
            # 2  the answer's figures: the provider's usage, chatd's timing, the proxy's price
            row = [r for r in px.rows() if r["backend"] == "ds"][-1]
            s = snap["messages"][-1]["stats"]
            assert s["completion_tokens"] == row["completion_tokens"] == 306
            assert s["reasoning_tokens"] == 6 and s["prompt_tokens"] == row["prompt_tokens"]
            assert s["cached_tokens"] == row["cached_tokens"] == 9
            assert s["cache_source"] == "provider"
            assert s["decode_tps"] and 20 < s["decode_tps"] < 1000 and s["ttft_ms"] > 0
            assert row["cost_micros"] > 0 and s["cost_usd"] == round(row["cost_micros"] / 1e6, 6)
            assert snap["usage"]["cost_usd"] == s["cost_usd"]
            # 3  the Context panel: estimated split, the provider's cache, the proxy's window
            ctx = snap["context"]
            assert ctx["model"]["backend"] == "cloud" and ctx["window"] == 1_048_576
            assert ctx["cache"]["source"] == "provider" and ctx["cache"]["warm"] == 9
            assert ctx["split"]["estimated"] is True and ctx["split"]["tools"] > 0
            parts = {p["kind"]: p["tokens"] for p in ctx["parts"]}
            assert parts["system"] > 0 and parts["tools"] == ctx["split"]["tools"]
            assert parts["reserved"] == 393_216        # DeepSeek's output limit (the card)
            # the request asked for the cost, and for nothing an engine reads
            assert srv.fake.last["headers"].get("X-Tandem-Cost") is None
        finally:
            c.close()
    srv.shutdown()


def test_the_cost_comment_adds_up_and_a_call_without_a_price_adds_nothing():
    r = runs.Run.__new__(runs.Run)
    r.cost_micros = None
    r._cost('{"cost_micros": null, "backend": "ds"}')
    assert r.cost_micros is None
    r._cost('{"cost_micros": 1200, "backend": "ds"}')
    r._cost('{"cost_micros": 34, "backend": "ds"}')
    r._cost("not json")
    assert r.cost_micros == 1234


def test_deepseek_cache_hits_win_over_the_openai_field():
    assert runs.cached_of({"prompt_cache_hit_tokens": 46208,
                           "prompt_tokens_details": {"cached_tokens": 0}}) == 46208
    assert runs.cached_of({"prompt_tokens_details": {"cached_tokens": 7}}) == 7
    assert runs.cached_of({}) == 0


def test_estimated_split_counts_the_skill_index_apart():
    body = {"messages": [{"role": "system", "content": "rules " * 100 + "\n- pagoda: a skill"},
                         {"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}]}
    sp = agent.estimated_split(body, 0.25, "- pagoda: a skill")
    assert sp["estimated"] is True and sp["skills"] == int(len("- pagoda: a skill") * 0.25)
    assert sp["system"] == int(len(body["messages"][0]["content"]) * 0.25) - sp["skills"]
    assert sp["tools"] > 0 and sp["images"] == 0
    r = ctx_mod.report(model={"id": "deepseek-flash", "max_len": 1_048_576, "max_output": 393_216,
                              "backend": "cloud"}, settings={}, branch_tokens=5_000,
                       files_tokens=0, stats=None, run=None, row=None, facts={}, peek=None,
                       harness=None, split=sp)
    parts = {p["kind"]: p["tokens"] for p in r["parts"]}
    assert parts["skills"] == sp["skills"] and parts["system"] == sp["system"]
    assert parts["messages"] == 5_000 - sp["system"] - sp["tools"] - sp["skills"]
    # the answer's room is the output limit while the window has more
    assert parts["reserved"] == 393_216
    small = ctx_mod.report(model={"id": "m", "max_len": 8_000, "max_output": 393_216},
                           settings={}, branch_tokens=5_000, files_tokens=0, stats=None, run=None,
                           row=None, facts={}, peek=None, harness=None)
    assert {p["kind"]: p["tokens"] for p in small["parts"]}["reserved"] == 8_000 - 5_000 - 256


def _png(w: int = 8, h: int = 8) -> bytes:
    raw = b"".join(b"\x00" + b"\x80\x40\x20" * w for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_effort_is_the_models_own_level():
    ds = ["low", "high", "max"]
    assert render.nearest_effort("max", ds) == "max"
    assert render.nearest_effort("medium", ds) == "high"    # DeepSeek maps it so itself
    assert render.nearest_effort("xhigh", ds) == "high"
    assert render.nearest_effort("low", ds) == "low"
    assert render.nearest_effort("max", ["low", "medium", "high", "xhigh"]) == "xhigh"
    assert render.nearest_effort("high", []) == "high"
    b = render.body("deepseek-flash", {"thinking": False, "effort": "max"}, [], levels=ds,
                    effort_off="none")
    assert b["reasoning_effort"] == "none" and "chat_template_kwargs" not in b
    b = render.body("TandemLLM", {"thinking": False, "effort": "high"}, [])
    assert b["chat_template_kwargs"] == {"enable_thinking": False} and "reasoning_effort" not in b


DS_CARDS = {"deepseek-flash": {"name": "DeepSeek-V4.1-Flash", "context_window": 1048576,
                               "max_model_len": None, "max_output_tokens": 393216,
                               "input_modalities": ["text", "image"],
                               "effort": {"supported_levels": ["low", "high", "max"],
                                          "default_level": "high"}},
            "deepseek-v4-pro": {"name": "DeepSeek-V4-Pro", "context_window": 1048576,
                                "max_model_len": None, "max_output_tokens": 393216,
                                "input_modalities": ["text"],
                                "effort": {"supported_levels": ["low", "high", "max"],
                                           "default_level": "high"}}}


def test_the_providers_name_levels_and_images_reach_the_chat():
    srv, port = fake(models=["deepseek-flash", "deepseek-v4-pro"])
    srv.fake.cards = DS_CARDS
    # the first discovery of a new backend: both models come enabled
    env = {k: v for k, v in _env(port).items() if "_MODELS" not in k}
    with Proxy(env) as px:
        px.sync()
        c = Chat(engine_url=f"http://127.0.0.1:{px.port}")
        try:
            caps = c.ok("GET", "/capabilities")
            m = {x["id"]: x for x in caps["models"]}
            assert m["deepseek-flash"]["name"] == "DeepSeek V4.1 Flash (cloud)"
            assert m["deepseek-v4-pro"]["name"] == "DeepSeek V4 Pro (cloud)"
            assert m["deepseek-flash"]["capabilities"]["effort_levels"] == ["low", "high", "max"]
            assert m["deepseek-flash"]["capabilities"]["effort_default"] == "high"
            assert m["deepseek-flash"]["capabilities"]["vision"] is True
            assert m["deepseek-v4-pro"]["capabilities"]["vision"] is False
            img = upload(c, "shot.png", "image/png", _png())
            for model, effort, want_effort, sees in (("deepseek-flash", "max", "max", True),
                                                     ("deepseek-v4-pro", "medium", "high", False)):
                cid = c.new()
                c.ok("PATCH", f"/conversations/{cid}/settings", {"model": model, "effort": effort})
                c.ok("POST", f"/conversations/{cid}/messages",
                     {"client_msg_id": f"i-{model}", "text": "what is in it? FAKE_DEEPSEEK",
                      "attachments": [img["id"]]})
                assert c.wait_idle(cid)["messages"][-1]["status"] == "done"
                body = [b for b in srv.fake.bodies if b.get("model") == model][-1]
                assert body["reasoning_effort"] == want_effort
                content = body["messages"][-1]["content"]
                parts = content if isinstance(content, list) else []
                has_img = any(p.get("type") == "image_url" and
                              p["image_url"]["url"].startswith("data:image/png;base64,")
                              for p in parts)
                assert has_img is sees, (model, content if isinstance(content, str) else parts)
        finally:
            c.close()
    srv.shutdown()


if __name__ == "__main__":
    run_all(globals())
