"""What a provider's model list says, and a model that appears in it (VIS-207, 2026-10-17).

DeepSeek's `GET /models` names each model, its window, its output limit, its input kinds and its
effort levels; the picker showed a stale "DeepSeek V4 Flash" from the adapter's table. And
`deepseek-v4.1-flash-1017` was in that list for minutes at about 01:59 UTC and gone again: a model
id the discovery never saw is stored disabled, logged and pushed once, never offered by itself.
Every answer's `system_fingerprint` goes into its ledger row, so a snapshot swapped behind
`deepseek-flash` shows. CPU only, a fake DeepSeek. Run: python tests/test_proxy_discovery.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import FIXTURE_KEYS, Proxy, fake, run  # noqa: E402

from proxy import adapters, route  # noqa: E402

FLASH = {"name": "DeepSeek-V4.1-Flash", "context_window": 1048576, "max_output_tokens": 393216,
         "input_modalities": ["text", "image"], "output_modalities": ["text"],
         "effort": {"supported_levels": ["low", "high", "max"], "default_level": "high"}}
PRO = dict(FLASH, name="DeepSeek-V4-Pro", input_modalities=["text"])
# the fake lists vLLM's max_model_len beside them; DeepSeek's list has none
SERVED = {"deepseek-flash": dict(FLASH, max_model_len=None),
          "deepseek-v4-pro": dict(PRO, max_model_len=None)}


def _env(port):
    return {"TANDEM_BACKENDS": "ds", "TANDEM_BACKEND_DS_KIND": "deepseek",
            "TANDEM_BACKEND_DS_URL": f"http://127.0.0.1:{port}",
            "TANDEM_BACKEND_DS_KEY": FIXTURE_KEYS[2], "TANDEM_BACKEND_DS_LABEL": "DeepSeek (cloud)"}


def test_deepseek_model_list_is_read_whole():
    ds = adapters.make({"kind": "deepseek", "id": "ds"})
    got = ds.parse_models({"object": "list", "data": [dict(FLASH, id="deepseek-flash"),
                                                      {"id": "bare"}]})
    assert got[0] == {"id": "deepseek-flash", "context_length": 1048576,
                      "name": "DeepSeek V4.1 Flash", "max_output": 393216,
                      "input_modalities": ["text", "image"],
                      "effort": {"levels": ["low", "high", "max"], "default": "high"}}
    assert got[1] == {"id": "bare", "context_length": None, "name": None, "max_output": None,
                      "input_modalities": None, "effort": None}
    # a generic kind keeps the provider's spelling
    assert adapters.make({"kind": "generic"}).parse_models(
        {"data": [dict(FLASH, id="x")]})[0]["name"] == "DeepSeek-V4.1-Flash"
    assert route.capabilities(adapters.KINDS["vllm"], {"input_modalities": None,
                                                       "effort": None}) is None


def test_the_card_carries_the_providers_words_and_a_new_model_comes_disabled():
    srv, port = fake(models=["deepseek-flash", "deepseek-v4-pro"])
    srv.fake.cards = SERVED
    with Proxy(_env(port)) as px:
        pushed = []
        px.svc.notifier = lambda cfg, title, msg: pushed.append((title, msg)) or "sent 200"
        px.svc.monitor_base = None
        px.sync()
        cards = {m["id"]: m["tandem"] for m in json.loads(px.get("/v1/models")[2])["data"]}
        f, p = cards["deepseek-flash"], cards["deepseek-v4-pro"]
        assert f["name"] == "DeepSeek V4.1 Flash" and f["context_length"] == 1048576
        assert f["max_output"] == 393216 and f["cloud"] is True
        assert f["capabilities"] == {"thinking": True, "effort_levels": ["low", "high", "max"],
                                     "effort_default": "high", "effort_off": "none",
                                     "tools": True, "vision": True, "structured": False}
        assert p["name"] == "DeepSeek V4 Pro" and p["capabilities"]["vision"] is False
        assert not pushed                                # the first list is the backend's own
        # a snapshot id appears for a while: stored, disabled, logged, pushed once
        srv.fake.models.append("deepseek-v4.1-flash-1017")
        px.svc.health._models_due.clear()
        px.svc.monitor_base = _policy_with_ntfy()
        px.sync()
        rows = {m["model"]: m for m in px.svc.cfg.models("ds")}
        assert rows["deepseek-v4.1-flash-1017"]["enabled"] is False
        assert "deepseek-v4.1-flash-1017" not in {m["id"] for m in json.loads(
            px.get("/v1/models")[2])["data"]}
        assert pushed == [("TandemLLM: new model", "DeepSeek (cloud): new model "
                           "deepseek-v4.1-flash-1017 in /models (disabled; turn it on under "
                           "Backends)")]
        assert any("new model deepseek-v4.1-flash-1017" in line for line in px.logs)
        # gone again, then back: new once
        srv.fake.models.remove("deepseek-v4.1-flash-1017")
        px.svc.health._models_due.clear()
        px.sync()
        srv.fake.models.append("deepseek-v4.1-flash-1017")
        px.svc.health._models_due.clear()
        px.sync()
        assert len(pushed) == 1
    srv.shutdown()


class _Task(dict):
    pass


class _Policy:
    def __init__(self, notify):
        self.tasks = {"S4": _Task(notify=notify)}


def _policy_with_ntfy():
    return _Policy({"enabled": False, "url": "https://ntfy.example/topic",
                    "token_file": "/nonexistent/token", "ca_file": "", "priority": "default"})


def test_no_push_without_a_topic():
    srv, port = fake(models=["deepseek-flash"])
    with Proxy(_env(port)) as px:
        pushed = []
        px.svc.notifier = lambda cfg, title, msg: pushed.append(title) or "sent 200"
        px.svc.monitor_base = _Policy({"enabled": False, "url": "", "token_file": ""})
        px.sync()
        srv.fake.models.append("deepseek-new")
        px.svc.health._models_due.clear()
        px.sync()
        assert not pushed and any("new model deepseek-new" in x for x in px.logs)
    srv.shutdown()


def test_the_fingerprint_goes_into_the_row():
    srv, port = fake(models=["deepseek-flash"])
    with Proxy(_env(port)) as px:
        px.sync()
        body = {"model": "deepseek-flash", "stream": True,
                "messages": [{"role": "user", "content": "hi FAKE_DEEPSEEK"}]}
        st, _, out = px.post("/v1/chat/completions", body)
        assert st == 200
        fp = [json.loads(e[6:]).get("system_fingerprint") for e in out.decode().split("\n\n")
              if e.startswith("data: {")]
        row = px.rows()[-1]
        assert fp and fp[0] == "fp_fake_e72d3c7f" and row["system_fingerprint"] == fp[0]
        req = json.loads(px.get("/v1/dashboard/requests?limit=1")[2])["requests"][0]
        assert req["system_fingerprint"] == row["system_fingerprint"]
    srv.shutdown()


def _card(px, mid):
    return {m["id"]: m["tandem"] for m in json.loads(px.get("/v1/models")[2])["data"]}[mid]


def _put(px, bid, mid, body):
    st, _, out = px.request("PUT", f"/v1/config/backends/{bid}/models/{mid}", body)
    return st, json.loads(out)


def test_an_operator_sets_a_relayed_models_output_limit():
    """VIS-227 (#21): a vLLM relay's model list has no output limit and the adapter's table knows
    no relayed name; the operator's goes on the card and a later health check keeps it."""
    srv, port = fake(models=["Qwen/Qwen3.5-9B"])
    env = {"TANDEM_BACKENDS": "v1", "TANDEM_BACKEND_V1_KIND": "vllm",
           "TANDEM_BACKEND_V1_URL": f"http://127.0.0.1:{port}/v1"}
    with Proxy(env) as px:
        px.sync()
        assert _card(px, "Qwen/Qwen3.5-9B")["max_output"] is None
        for bad in (0, -1, 2.5, "8192", True, {}):
            st, out = _put(px, "v1", "Qwen/Qwen3.5-9B", {"max_output": bad})
            assert st == 422 and out["error"]["field"] == "max_output", bad
        st, row = _put(px, "v1", "Qwen/Qwen3.5-9B", {"max_output": 8192})
        assert st == 200 and row["max_output_set"] == 8192 and row["max_output"] is None
        assert _card(px, "Qwen/Qwen3.5-9B")["max_output"] == 8192
        px.svc.health._models_due.clear()
        px.sync()
        assert _card(px, "Qwen/Qwen3.5-9B")["max_output"] == 8192
        st, row = _put(px, "v1", "Qwen/Qwen3.5-9B", {"max_output": None})
        assert st == 200 and row["max_output_set"] is None
        assert _card(px, "Qwen/Qwen3.5-9B")["max_output"] is None
    srv.shutdown()


def test_a_discovery_never_overwrites_the_operators_output_limit():
    srv, port = fake(models=["deepseek-flash"])
    srv.fake.cards = SERVED
    with Proxy(_env(port)) as px:
        px.sync()
        st, _ = _put(px, "ds", "deepseek-flash", {"max_output": 100000})
        assert st == 200
        px.svc.health._models_due.clear()
        px.sync()
        row = next(m for m in px.svc.cfg.models("ds") if m["model"] == "deepseek-flash")
        assert row["max_output"] == 393216 and row["max_output_set"] == 100000
        assert _card(px, "deepseek-flash")["max_output"] == 100000
        _put(px, "ds", "deepseek-flash", {"max_output": None})
        assert _card(px, "deepseek-flash")["max_output"] == 393216      # the provider's again
    srv.shutdown()


def test_the_row_shows_what_the_card_carries_without_the_operators_limit():
    """A DeepSeek list without max_output_tokens: the card has the adapter table's 393,216, and so
    does the Backends row's placeholder, not "none"."""
    srv, port = fake(models=["deepseek-flash"])
    with Proxy(_env(port)) as px:
        px.sync()

        def row():
            b = json.loads(px.get("/v1/config/backends/ds")[2])
            return next(m for m in b["model_rows"] if m["model"] == "deepseek-flash")
        assert row()["max_output"] is None and row()["max_output_default"] == 393216
        assert _card(px, "deepseek-flash")["max_output"] == 393216
        _put(px, "ds", "deepseek-flash", {"max_output": 100000})
        assert row()["max_output_default"] == 393216 and row()["max_output_set"] == 100000
        assert _card(px, "deepseek-flash")["max_output"] == 100000
    srv.shutdown()


if __name__ == "__main__":
    run(globals())
