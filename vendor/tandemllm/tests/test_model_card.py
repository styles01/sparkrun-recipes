"""The model card in /v1/models (SRV-347, server/model_card.py) and POST /v1/tokenize's counts
(server/tokenize_api.py). CPU only, no tokenizer: the counts take a stand-in render.

Run: python -m pytest -q tests/test_model_card.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import model_card, tokenize_api  # noqa: E402

CAPS = {"low": 1024, "medium": 8192, "high": 16384, "xhigh": 0}


def _facts(**kw):
    f = {"pack": "exl3-4.5", "max_len": 262144, "max_output": 32768, "vision": True,
         "effort_caps": CAPS, "label": "TandemLLM", "loaded": True}
    f.update(kw)
    return f


def test_the_27b_card_matches_its_card_file_and_the_loader():
    c = model_card.card(_facts())
    assert c["backend"] == "engine" and c["model"] == "qwen3.8-27b" and c["pack"] == "exl3-4.5"
    assert c["label"] == "Qwen3.8-27B" and c["context_length"] == 262144
    assert c["max_output"] == 32768 and c["loaded"] is True
    caps = c["capabilities"]
    assert caps["thinking"] and caps["tools"] and caps["vision"] and caps["structured"]
    assert caps["effort_levels"] == ["off", "low", "medium", "high", "xhigh"]
    assert c["effort_budgets"] == CAPS
    # what the process serves wins: a shorter window, no vision tower loaded
    c = model_card.card(_facts(max_len=65536, vision=False, pack="nvfp4"))
    assert c["context_length"] == 65536 and c["capabilities"]["vision"] is False
    assert c["model"] == "qwen3.8-27b"


def test_a_new_architecture_fills_its_card_with_no_server_edit():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "toy-arch"))
    with open(os.path.join(d, "toy-arch", "card.json"), "w") as fh:
        json.dump({"label": "Toy 1B", "packs": ["toy-*"], "context_limit": 8192,
                   "capabilities": {"thinking": False, "tools": False, "vision": False,
                                    "structured": True}}, fh)
    c = model_card.card(_facts(pack="toy-q4", max_len=32768, max_output=None), d)
    assert c["model"] == "toy-arch" and c["label"] == "Toy 1B" and c["context_length"] == 8192
    caps = c["capabilities"]
    assert caps["thinking"] is False and caps["effort_levels"] == [] and caps["structured"]
    assert caps["tools"] is False and c["effort_budgets"] == {}


def test_a_model_with_no_card_gets_the_loader_and_safe_defaults():
    c = model_card.card(_facts(pack="flashnext-3.05", max_len=40960), tempfile.mkdtemp())
    assert c["model"] is None and c["label"] == "TandemLLM" and c["context_length"] == 40960
    assert c["capabilities"] == {"thinking": False, "tools": True, "vision": False,
                                 "structured": False, "parallel_tool_calls": False,
                                 "effort_levels": []}


def test_the_flashnext_card_covers_its_packs():
    """rc-0.8.0 live: without a card.json the served Flash-Next pack got the safe defaults (no thinking,
    no vision), and the Chat tab sent an attached image as a file the model could only open with bash.
    Its card: thinking with the effort levels, tools, structured output, and images once the tower is
    loaded (VIS-97); the window the pack's `max_len`."""
    for pack in ("flashnext-2.05", "flashnext-3.05", "flashnext-4.05"):
        c = model_card.card(_facts(pack=pack, max_len=40960))
        assert c["model"] == "qwen3.8-flash-next" and c["label"] == "Qwen3.8-Flash-Next", c
        caps = c["capabilities"]
        assert caps["thinking"] and caps["tools"] and caps["vision"] and caps["structured"], caps
        assert caps["effort_levels"] == ["off", "low", "medium", "high", "xhigh"]
        assert c["context_length"] == 40960
    assert model_card.card(_facts(pack="flashnext-3.05", vision=False))["capabilities"]["vision"] is False
    assert model_card.card(_facts(pack="exl3-4.5"))["model"] == "qwen3.8-27b"
    return "the Flash-Next packs get their card; images only with the tower loaded"


def test_the_super_vl_card_reads_images_with_the_tower_loaded():
    """ENG-365 / VIS-205: Super VL's packs say vision once the RADIO tower is loaded (`--vision off`, or a
    checkpoint without it: no), so the Chat tab sends a chat's images and tool screenshots to it."""
    for pack in ("nemotron-super-vl-exl3-4.0", "nemotron-super-vl-exl3-2.0", "nemotron-super-vl-nvfp4"):
        c = model_card.card(_facts(pack=pack))
        assert c["model"] == "nemotron-3.5-super-vl-120b-a12b" and c["capabilities"]["vision"] is True, c
        assert model_card.card(_facts(pack=pack, vision=False))["capabilities"]["vision"] is False
    assert model_card.card(_facts(pack="nemotron-bolt-exl3-2.0"))["capabilities"]["vision"] is False
    return "the three Super VL packs: vision with the tower; Bolt: none"


def test_annotate_leaves_proxy_models_their_own_card():
    entries = [{"id": "TandemLLM", "object": "model"},
               {"id": "openai/gpt", "tandem": {"backend": "openai", "context_length": 128000}}]
    out = model_card.annotate(entries, _facts())
    assert out[0]["tandem"]["backend"] == "engine"
    assert out[1]["tandem"] == {"backend": "openai", "context_length": 128000}
    assert "tandem" not in entries[0]              # the input is not changed


def _render(body):
    """A stand-in template: 3 tokens a message plus its words, 5 per tool plus its name's
    length, 2 for the generation prompt."""
    n = 2
    for m in body["messages"]:
        n += 3 + len(str(m.get("content") or "").split())
    for t in body.get("tools") or []:
        n += 5 + len(t["function"]["name"])
    return n


def test_tokenize_parts_add_up_and_count_tools_as_rendered():
    tools = [{"type": "function", "function": {"name": n}} for n in ("bash", "read", "edit")]
    body = {"messages": [{"role": "system", "content": "be brief"},
                         {"role": "user", "content": "one two three"}], "tools": tools}
    out = tokenize_api.count(body, _render, lambda t: len(t.split()))
    assert out["tokens"] == _render(body)
    assert out["parts"]["tools"] == sum(5 + len(t["function"]["name"]) for t in tools)
    assert out["parts"]["system"] == 3 + 2
    assert sum(out["parts"].values()) == out["tokens"]
    t = tokenize_api.count({"text": "a b c d"}, _render, lambda t: len(t.split()))
    assert t == {"tokens": 4, "parts": {"system": 0, "tools": 0, "messages": 4}}
    try:
        tokenize_api.count({"messages": []}, _render, len)
        raise AssertionError("an empty list must be refused")
    except ValueError:
        pass


def test_tokenize_is_rate_limited():
    who = "test-rate"
    assert all(tokenize_api.allowed(who, now=100.0) for _ in range(tokenize_api.RATE_PER_MIN))
    assert not tokenize_api.allowed(who, now=100.5)
    assert tokenize_api.allowed(who, now=161.0)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
