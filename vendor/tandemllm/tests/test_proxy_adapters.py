"""The adapters on frames shaped as the providers send them (tests/fixtures/proxy/frames.json):
usage with and without details, DeepSeek's cache split, reasoning deltas, <think> in content,
tool-call deltas, the Responses API's events, the usage frame asked for and dropped, the price.

The fixtures are written from each provider's documented shapes (OpenAI chat and Responses,
DeepSeek), not recorded from paid calls; a real key's check is listed in docs/proxy.md.

Run: python tests/test_proxy_adapters.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import run  # noqa: E402

from proxy import adapters as A  # noqa: E402
from proxy.adapters.base import CHAT, RESPONSES, Stamps  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "proxy",
                   "frames.json")


def _fx() -> dict:
    with open(FIX) as f:
        return json.load(f)


def _feed(adapter, path, events, injected):
    st = Stamps()
    out = []
    for ev in events:
        raw = ev.encode()
        tok, drop = adapter.read_frame(path, raw, st, injected)
        out.append((tok, drop))
    return st, out


def test_each_fixture_normalises_to_its_expected_row():
    fx = _fx()
    for name, case in fx["streams"].items():
        a = A.make({"id": "b", "kind": case["kind"], "inject_usage": True})
        st, marks = _feed(a, case.get("path", CHAT), case["events"], case["injected"])
        u = a.read_usage(st.usage, case.get("path", CHAT))
        assert u == case["usage"], (name, u)
        assert st.reasoning == case["reasoning"], (name, st.reasoning)
        assert len(st.tools) == case["tools"], (name, st.tools)
        assert sum(1 for t, _ in marks if t) == case["token_frames"], (name, marks)
        assert [i for i, (_, d) in enumerate(marks) if d] == case["dropped"], (name, marks)
        assert (st.finish_reason or "stop") == case["finish"], (name, st.finish_reason)


def test_bodies_normalise():
    fx = _fx()
    for name, case in fx["bodies"].items():
        a = A.make({"id": "b", "kind": case["kind"]})
        st = Stamps()
        a.read_body(case.get("path", CHAT), case["body"], st)
        assert a.read_usage(st.usage, case.get("path", CHAT)) == case["usage"], name
        assert st.reasoning == case["reasoning"] and len(st.tools) == case["tools"], name


def test_include_usage_injected_only_when_not_asked():
    a = A.make({"id": "b", "kind": "generic", "inject_usage": True})
    raw = b'{"model": "m", "stream": true, "messages": []}'
    body = json.loads(raw)
    out, inj = a.prepare(CHAT, raw, body, "m")
    assert inj and json.loads(out)["stream_options"] == {"include_usage": True}
    asked = b'{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[]}'
    out, inj = a.prepare(CHAT, asked, json.loads(asked), "m")
    assert not inj and out == asked, "a client that asked keeps its bytes and its frame"
    plain = b'{"model":"m","messages":[]}'
    out, inj = a.prepare(CHAT, plain, json.loads(plain), "m")
    assert out == plain and not inj, "nothing to change: the client's bytes"
    out, inj = a.prepare(CHAT, plain, json.loads(plain), "upstream-id")
    assert json.loads(out)["model"] == "upstream-id"
    off = A.make({"id": "b", "kind": "generic", "inject_usage": False})
    assert off.prepare(CHAT, raw, body, "m") == (raw, False)
    t = A.make({"id": "b", "kind": "tandemllm"})       # usage on its finish chunk, unasked
    assert t.inject_usage() is False and t.prepare(CHAT, raw, body, "m") == (raw, False)


def test_capabilities_and_routes_per_kind():
    oa, ds = A.make({"kind": "openai"}), A.make({"kind": "deepseek"})
    gen = A.make({"kind": "generic"})
    assert RESPONSES in oa.routes() and RESPONSES not in ds.routes()
    assert RESPONSES in gen.routes()
    assert oa.capabilities()["thinking_switch"] == "reasoning_effort"
    assert ds.capabilities()["thinking_switch"] == "thinking_type"
    assert ds.capabilities()["reasoning_field"] == "reasoning_content"
    assert oa.capabilities()["reasoning_field"] == "none"
    assert all(not k.capabilities()["continue_final_message"] for k in (oa, ds, gen))
    from email.message import Message
    h = Message()
    h["X-Tandem"], h["x-session-id"], h["Authorization"] = "raw", "s", "Bearer client-key"
    assert oa.forward(h) == []
    assert gen.forward(h) == [("x-tandem", "raw"), ("x-session-id", "s")], \
        "a local engine gets the conductor's headers, never the client's key"
    assert oa.headers("k")[-1] == ("Authorization", "Bearer k")


def test_price():
    p = {"input": 1.0, "cached_input": 0.1, "output": 2.0, "cache_write": None}
    u = {"prompt_tokens": 1000, "cached_tokens": 400, "completion_tokens": 100,
         "cache_write_tokens": None, "reasoning_tokens": None}
    assert A.Adapter.cost(u, p) == 600 + 40 + 200          # micro-dollars
    assert A.Adapter.cost(dict(u, cached_tokens=None), p) == 1000 + 200
    assert A.Adapter.cost(u, None) is None
    assert A.Adapter.cost(dict(u, prompt_tokens=None), p) is None
    assert A.Adapter.cost(u, dict(p, cached_input=None)) == 1000 + 200
    assert A.Adapter.cost(u, dict(p, output=None)) is None


def test_deepseek_offpeak_price_by_the_hour():
    import datetime as dt
    ds = A.make({"id": "ds", "kind": "deepseek"})
    p = {"input": 0.30, "cached_input": 0.006, "output": 1.2,
         "offpeak": {"input": 0.15, "cached_input": 0.003, "output": 0.6}}
    u = {"prompt_tokens": 1000, "cached_tokens": 0, "completion_tokens": 1000,
         "cache_write_tokens": None, "reasoning_tokens": None}
    def at(y, m, d, h):
        return dt.datetime(y, m, d, h, 30, tzinfo=dt.timezone.utc).timestamp()
    assert ds.price(u, p, at(2026, 10, 5, 2)) == 1500          # Monday 02:30 UTC: peak
    assert ds.price(u, p, at(2026, 10, 5, 5)) == 750           # Monday 05:30: off-peak
    assert ds.price(u, p, at(2026, 10, 4, 2)) == 750           # Sunday: off-peak
    assert ds.price(u, p, None) == 1500


def test_deepseek_miss_mismatch_is_a_log_line_not_a_correction():
    logs = []
    ds = A.make({"id": "ds", "kind": "deepseek"}, log=logs.append)
    u = ds.read_usage({"prompt_tokens": 100, "completion_tokens": 5,
                       "prompt_cache_hit_tokens": 64, "prompt_cache_miss_tokens": 30})
    assert u["cached_tokens"] == 64 and u["prompt_tokens"] == 100
    assert len(logs) == 1 and "kept as reported" in logs[0]


if __name__ == "__main__":
    run(globals())


def test_the_model_list_names_each_model_by_its_provider_name():
    """VIS-203: the card's label names the backend (both DeepSeek models said `DeepSeek (cloud)`);
    `name` is the provider's own name of the model (null when the adapter has none), `cloud` the
    backend's kind."""
    from types import SimpleNamespace

    from proxy.route import Table
    backends = [{"id": "deepseek", "kind": "deepseek", "label": "DeepSeek (cloud)", "enabled": True},
                {"id": "v1", "kind": "vllm", "label": "vLLM box", "enabled": True}]
    row = dict(enabled=True, present=True, public_id=None, context_length=1048576, features={},
               price=None, fallback=None)
    models = [dict(row, backend="deepseek", model="deepseek-flash"),
              dict(row, backend="deepseek", model="deepseek-v4-pro"),
              dict(row, backend="v1", model="Qwen/Qwen3.5-4B")]
    cfg = SimpleNamespace(generation=1, backends=lambda: backends, models=lambda: models,
                          rules=lambda: [])
    cards = {m["id"]: m["tandem"] for m in Table(cfg).listing(0)}
    assert cards["deepseek-flash"]["name"] == "DeepSeek V4 Flash" and cards["deepseek-flash"]["cloud"]
    assert cards["deepseek-v4-pro"]["name"] == "DeepSeek V4 Pro"
    assert cards["deepseek-flash"]["label"] == "DeepSeek (cloud)"
    assert cards["Qwen/Qwen3.5-4B"]["name"] is None and cards["Qwen/Qwen3.5-4B"]["cloud"] is False
    assert cards["deepseek-flash"]["max_output"] == cards["deepseek-v4-pro"]["max_output"] == 393216
    assert cards["Qwen/Qwen3.5-4B"]["max_output"] is None
