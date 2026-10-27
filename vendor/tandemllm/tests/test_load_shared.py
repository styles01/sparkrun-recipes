"""SRV-341: one loader for every model (server/app.py `_load`), no GPU.

The model's builder is the one its registry entry names (`Arch.serve`; the 27B's is the server's own
`_build_qwen3_5`). Everything after it is the shared loader, run for real on the server's argument
defaults:

  * an engine whose capabilities hold no caches and no side slots (Flash-Next today) gets the request
    path's state with every cache off and is told so, `--parallel` falls back to one slot, its
    `precapture` is asked, and the builder's record (its weights) lands in STATE;
  * an engine that holds caches (the 27B) gets the memory policy and the state store from
    `_load_caches`, the tree mode and accept rule its builder chose, its side slots built, and the
    request-path state has the keys it had before the loaders were one (pinned);
  * a third architecture registered in a test, with no edit under server/, starts on CPU and answers
    /v1/chat/completions;
  * no model branch in the server: no `hasattr(eng` / `isinstance(eng`, no other model's name.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine import models  # noqa: E402
from engine.models.adapter import Capabilities  # noqa: E402
from engine.spec import Relax  # noqa: E402
from server import app  # noqa: E402
from server.fake_engine import FakeTokenizer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: the request-path state's keys as the 27B's loader wrote them before SRV-341 (server/app.py at
#: 0527fc4b, `_serving_state` and the STATE.update it replaced)
KEYS_27B = {"engine", "tok", "drafter", "k", "device", "tree", "sampled_tree", "relax", "model", "started",
            "verbose", "max_len", "default_max_tokens", "reasoning_format", "request_timeout", "max_queue",
            "queue_timeout", "draining", "cfg_eos", "think_budget", "think_stall", "reasoning_effort",
            "answer_reserve", "pen_spec", "temperature", "top_p", "top_k", "pattern_stop", "argloop",
            "state_store", "session_cache", "prefix_cache", "session_trim", "resident", "prefill_heartbeat",
            "prefix_chunk", "response_cache", "suffix_store", "suffix_scope", "usage_default"}


class _Eng:
    def __init__(self, caps):
        self.caps = caps
        self.kv = types.SimpleNamespace(pages=None, length=0, max_len=64)
        self.asked = []

    def precapture(self, drafter):
        self.asked.append(drafter)
        return None if not self.caps.caches else 3

    def cost(self):
        from engine.models.adapter import CostModel
        return CostModel(section_bytes=0, view_bytes_per_token=0, workspace_floor=3 << 30, workspace_per_token=100)

    def resident_files(self):
        return []


def _run(caps, argv, serving, registry_builder: bool):
    eng, calls, lines, seen = _Eng(caps), [], [], {}

    def build(a, ctx):
        ctx.keys(ROOT, [])
        ctx.mark("engine")
        seen["ctx"] = ctx
        return {"engine": eng, "tok": FakeTokenizer(), "drafter": None, "path": "/nonexistent", "device": "cpu",
                "cfg": "cfg", "w": "w", "serving": serving, "describe": "stub,"}

    arch = types.SimpleNamespace(serve_builder=lambda: build if registry_builder else None)
    saved_state = dict(app.STATE)
    patches = {"_arch_of": lambda m: arch, "_build_qwen3_5": build,
               "generate_stream": lambda *a, **k: iter(()), "_warm_prefill": lambda *a, **k: None,
               "_build_side_slots": lambda *a: calls.append(a)}
    saved = {k: getattr(app, k) for k in patches}
    try:
        for k, v in patches.items():
            setattr(app, k, v)
        app.print = lambda *a, **k: lines.append(" ".join(str(x) for x in a))
        app.STATE.clear()
        app._load(app.parser().parse_args(argv))
        state = dict(app.STATE)
        written = set(app.STATE)
    finally:
        for k, v in saved.items():
            setattr(app, k, v)
        del app.print
        app.STATE.clear()
        app.STATE.update(saved_state)
    return eng, state, calls, lines, written


def test_an_engine_without_caches_gets_them_off_and_one_slot():
    serving = {"tree": True, "sampled_tree": "det", "relax": Relax(1.0, 1), "weights_info": {"format": "exl3"}}
    eng, st, calls, lines, _ = _run(Capabilities(caches=False, side_slots=False), ["--parallel", "2"], serving,
                                    registry_builder=True)
    assert st["engine"] is eng and st["sampled_tree"] == "det" and st["tree"] is True and st["device"] == "cpu"
    assert st["state_store"] is None and st["resident"] is None and not st["prefix_cache"] and not st["session_cache"]
    assert st["prefix_chunk"] == 0 and st["suffix_store"] is None and st["weights_info"] == {"format": "exl3"}
    assert st["segments"] is not None and "tokenizer_sha" in st and st["weights_key"] == "none"
    assert any("not built for this model yet (ENG-157)" in x for x in lines), lines
    assert any("side slots are not built" in x for x in lines) and calls == [] and eng.asked == [None]
    assert [m["stage"] for m in st["load_ledger"].marks] == ["start", "engine", "caches", "warmup", "graphs"]
    return "caches off and said, one slot, precapture asked, the builder's record in STATE"


def test_an_engine_with_caches_gets_the_policy_the_store_and_its_old_keys():
    serving = {"tree": False, "sampled_tree": "off", "relax": Relax(1.0, 1)}
    eng, st, calls, lines, written = _run(Capabilities(caches=True, side_slots=True),
                                          ["--parallel", "2", "--drafter", "none", "--resident-gb", "0"], serving,
                                          registry_builder=False)
    assert st["prefix_cache"] and st["session_cache"] and st["state_store"] is not None
    assert st["mem"] is not None and st["resident"] is None and st["sampled_tree"] == "off"
    assert (st["mem"].ws_floor, st["mem"].ws_per_token) == (3 << 30, 100)   # the engine's own workspace
    assert st["prefix_chunk"] > 0 and "weights_info" not in written
    assert len(calls) == 1 and calls[0][1:3] == ("cfg", "w") and eng.asked == [None]
    assert any(x.startswith("[cache] session=on") for x in lines)
    assert any("verify graphs: 3 captured" in x for x in lines)
    assert KEYS_27B <= written, sorted(KEYS_27B - written)
    # SRV-475: the checkpoint's sampling (generation_config.json), read for every model
    extra = written - KEYS_27B - {"load_ledger", "tokenizer_sha", "weights_key", "segments", "mem",
                                  "card_sampling"}
    assert not extra, sorted(extra)
    return f"memory policy, state store, side slots from the builder's cfg and weights; {len(KEYS_27B)} keys as before"


def test_an_engine_that_cannot_be_lent_gets_the_side_lane_off():
    """rc-0.8.0 live (2026-10-12): Flash-Next on the side lane raised `'FlashNextCacheState' object has
    no attribute 'primed'` for every short request it took (Borrow keeps the 27B's pending commit); an
    engine whose caps say so starts with the lane off, and says so."""
    serving = {"tree": True, "sampled_tree": "det", "relax": Relax(1.0, 1), "weights_info": {"format": "exl3"}}
    eng, st, calls, lines, _ = _run(Capabilities(caches=False, side_slots=False, side_lane=False),
                                    ["--side-lane", "on"], serving, registry_builder=True)
    assert any("[side-lane] off" in x and "caps.side_lane" in x for x in lines), lines
    assert Capabilities(caches=True, side_slots=True).side_lane, "the 27B keeps its lane"
    return "the lane off and said; the default keeps it"


# ------------------------------------------------------------------ a third architecture

def toy_serve(a, ctx):
    """A test architecture's builder: the tiny random 27B-shaped engine on CPU, without caches."""
    from test_cache import fresh
    from test_end_anchor import ChatTok
    ctx.keys(a.model, [a.model])
    eng = fresh()
    eng.caps = Capabilities(caches=False, side_slots=False)
    ctx.mark("engine")
    return {"engine": eng, "tok": ChatTok(), "drafter": None, "path": a.model, "device": "cpu", "cfg": None,
            "w": None, "describe": "toy,",
            "serving": {"tree": False, "sampled_tree": a.sampled_tree, "relax": Relax(1.0, 1),
                        "weights_info": {"format": "toy"}}}


def test_a_third_architecture_starts_and_answers_with_no_server_edit():
    from test_app_loop import Req
    d = tempfile.mkdtemp(prefix="toy-arch-")
    saved_state = dict(app.STATE)
    models.register(models.Arch(name="toy", architectures=("ToyForCausalLM",), model_types=("toy",),
                                config="test_load_shared:toy_serve", engine="test_load_shared:toy_serve",
                                serve="test_load_shared:toy_serve"))
    try:
        json.dump({"architectures": ["ToyForCausalLM"], "model_type": "toy"}, open(os.path.join(d, "config.json"), "w"))
        app._load(app.parser().parse_args(["--model", d, "--max-len", "512", "--drafter", "none",
                                           "--prompt-token-cache", "0"]))
        assert app.STATE["device"] == "cpu" and app.STATE["weights_info"] == {"format": "toy"}
        assert app.weights_profile()["format"] == "toy"
        body = {"messages": [{"role": "user", "content": "".join(chr(0x100 + i) for i in range(5, 40))}],
                "max_tokens": 12, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        head, raw = Req("/v1/chat/completions", body).response()
        assert head.startswith("HTTP/1.1 200"), (head, raw[:300])
        r = json.loads(raw)
        assert r["usage"]["completion_tokens"] == 12 and r["choices"][0]["message"]["content"]
    finally:
        models.REGISTRY.pop("toy", None)
        app.STATE.clear()
        app.STATE.update(saved_state)
        shutil.rmtree(d, ignore_errors=True)
    return "registered in the test, started through _load on CPU, 12 tokens through /v1/chat/completions"


def test_no_model_branch_in_the_server():
    bad = re.compile(r"hasattr\(\s*eng\b|isinstance\(\s*eng\b|flashnext|flash-next|qwen4_exp", re.I)
    hits = []
    for p in sorted(glob.glob(os.path.join(ROOT, "server", "**", "*.py"), recursive=True)):
        for n, line in enumerate(open(p, encoding="utf-8"), 1):
            if bad.search(line):
                hits.append(f"{os.path.relpath(p, ROOT)}:{n}: {line.strip()}")
    assert not hits, hits
    return "no hasattr(eng / isinstance(eng, no other model's name under server/"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<66} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<66} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
