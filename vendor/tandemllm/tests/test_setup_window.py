"""The setup guide's window range is the model's own (ENG-363, QA ENG-364): `registry.config_max_len` reads
the longest context a checkpoint names (config.json at its top, in text_config or llm_config, the tokenizer's
model_max_length where shorter), `registry.with_config_window` takes the window and side window up to it when
the weights are on the box, and every card names its checkpoint's own maximum for a box without them.

Run: python tests/test_setup_window.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.setup import registry  # noqa: E402

# each checkpoint's own maximum (config.json max_position_embeddings, read 2026-10-16): none sets rope scaling
CHECKPOINT_MAX = {"qwen3.8-27b": 262_144, "qwen3.8-flash-next": 262_144, "lfm2.5-24b-a2b": 131_072,
                  "nemotron-3.5-bolt-74b-a9b": 262_144, "nemotron-3.5-lightning-30b-a3b": 262_144,
                  "nemotron-3.5-super-vl-120b-a12b": 1_048_576}


def _ckpt(cfg: dict, tok: dict | None = None) -> str:
    d = tempfile.mkdtemp(prefix="ckpt-")
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    if tok is not None:
        with open(os.path.join(d, "tokenizer_config.json"), "w") as fh:
            json.dump(tok, fh)
    return d


def test_the_config_names_the_longest_context():
    assert registry.config_max_len(_ckpt({"max_position_embeddings": 262_144})) == 262_144
    # Super VL's Omni wrapper keeps the language model's in llm_config; Qwen3.5's in text_config
    assert registry.config_max_len(_ckpt({"model_type": "nemotron_h_omni", "llm_config": {"max_position_embeddings": 1_048_576}},
                                         {"model_max_length": 1_048_576})) == 1_048_576
    assert registry.config_max_len(_ckpt({"text_config": {"max_position_embeddings": 262_144}})) == 262_144
    # a shorter tokenizer limit wins; LFM2.5's 1e30 is no limit
    assert registry.config_max_len(_ckpt({"max_position_embeddings": 262_144}, {"model_max_length": 131_072})) == 131_072
    assert registry.config_max_len(_ckpt({"max_position_embeddings": 131_072}, {"model_max_length": 10 ** 30})) == 131_072
    assert registry.config_max_len(_ckpt({})) is None
    assert registry.config_max_len(None) is None and registry.config_max_len("/nonexistent") is None


def test_the_window_range_follows_the_checkpoint_on_the_box():
    m = registry.get("nemotron-3.5-bolt-74b-a9b")
    assert registry.with_config_window(m, []) is m                        # no weights here: the card's
    big = registry.with_config_window(m, [None, _ckpt({"max_position_embeddings": 1_048_576})])
    for k in ("window", "side_window"):
        assert big["parallelism"][k]["max"] == 1_048_576
        assert big["parallelism"][k]["default"] == m["parallelism"][k]["default"]
    assert m["parallelism"]["window"]["max"] == 262_144                   # the card itself is untouched
    small = registry.with_config_window(m, [_ckpt({"max_position_embeddings": 100_000})])
    assert small["parallelism"]["window"]["max"] == 98_304                # on the 4,096 step
    assert small["parallelism"]["window"]["default"] == 98_304
    same = registry.with_config_window(m, [_ckpt({"max_position_embeddings": 262_144})])
    assert same is m


def test_every_card_names_its_checkpoints_maximum():
    models, _ = registry.load_all()
    assert set(CHECKPOINT_MAX) <= set(models), sorted(models)
    for mid, n in CHECKPOINT_MAX.items():
        par = models[mid]["parallelism"]
        assert par["window"]["max"] == n and par["side_window"]["max"] == n, (mid, par["window"], par["side_window"])
    sv = models["nemotron-3.5-super-vl-120b-a12b"]
    assert sv["parallelism"]["window"]["default"] == 262_144
    assert all(p["max_window"] == 1_048_576 for p in sv["profiles"])
    assert registry.window_refusal("nemotron-super-vl-exl3-4.0", 1_048_576) is None
    assert registry.window_refusal("nemotron-super-vl-exl3-4.0", 1_048_577) is not None


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, fn in fns:
        fn()
        print(f"ok  {k}")
    print(f"\n{len(fns)} passed")
