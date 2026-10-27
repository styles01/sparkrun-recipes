"""/126: the checkpoint layout in one place -- text-only configs, tied heads, canonical names.

The served checkpoint is a vision-language wrapper (`text_config`, `model.language_model.` prefix,
a vision tower that is skipped). A text-only checkpoint of the same family has the config keys at
the top level and the `model.` prefix. These tests build both kinds as tiny files on the CPU and
check that the engine reads them to the same canonical names and the same TextConfig, that a
checkpoint without `lm_head.weight` reads the embedding only when the config ties them, and that
the served checkpoint's key set maps exactly as before (the box audit of 2026-09-27: 1,250
language-model keys, 333 vision keys skipped, `mtp.*` and `lm_head.weight` unprefixed, no other
`model.` keys).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import LM_PREFIX, Layout, Weights  # noqa: E402

TEXT = {"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "vocab_size": 11, "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 4,
        "linear_value_head_dim": 4, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
        "full_attention_interval": 4, "rope_parameters": {"rope_theta": 10000.0}}


def _snap(config: dict, tensors: dict) -> str:
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f)
    save_file(tensors, os.path.join(d, "model.safetensors"))
    return d


def test_canonical_names():
    c = Layout.canonical
    assert c(LM_PREFIX + "layers.3.mlp.gate_proj.weight") == "layers.3.mlp.gate_proj.weight"
    assert c("model.layers.3.mlp.gate_proj.weight") == "layers.3.mlp.gate_proj.weight"
    assert c(LM_PREFIX + "embed_tokens.weight") == Layout.embed
    assert c("lm_head.weight") == Layout.head
    assert c("mtp.fc.weight") == "mtp.fc.weight"
    assert c("model.visual.blocks.0.attn.qkv.weight") is None
    assert c("visual.merger.weight") is None
    return "wrapper prefix, text-only prefix, top-level, mtp, vision skipped"


def test_text_only_config_equals_wrapped():
    a = load_config(_snap({"text_config": TEXT, "tie_word_embeddings": False}, {"x": torch.zeros(1)}))
    b = load_config(_snap(dict(TEXT), {"x": torch.zeros(1)}))
    fa = {k: v for k, v in vars(a).items() if k != "path"}
    fb = {k: v for k, v in vars(b).items() if k != "path"}
    assert fa == fb, {k: (fa[k], fb[k]) for k in fa if fa[k] != fb[k]}
    assert a.layer_types == ["linear_attention"] * 3 + ["full_attention"]
    return f"{len(fa)} fields equal"


def test_tied_head():
    emb = torch.randn(11, 8).to(torch.bfloat16)
    tensors = {"model.embed_tokens.weight": emb, "model.norm.weight": torch.ones(8, dtype=torch.bfloat16)}
    w = Weights(_snap(dict(TEXT, tie_word_embeddings=True), tensors), device="cpu", skip_mtp=True,
                nvfp4="", fp8_head="")
    assert w.t[Layout.head] is w.t[Layout.embed], "the tied head is not the embedding"
    assert torch.equal(w.t[Layout.embed], emb)
    try:
        Weights(_snap(dict(TEXT), tensors), device="cpu", skip_mtp=True, nvfp4="", fp8_head="")
    except RuntimeError as e:
        assert "lm_head" in str(e)
    else:
        raise AssertionError("an untied checkpoint without lm_head loaded")
    return "tied: head is the embedding (no copy); untied without a head refused"


def test_wrapped_checkpoint_names():
    t = {LM_PREFIX + "embed_tokens.weight": torch.zeros(11, 8, dtype=torch.bfloat16),
         LM_PREFIX + "norm.weight": torch.ones(8, dtype=torch.bfloat16),
         "lm_head.weight": torch.ones(11, 8, dtype=torch.bfloat16),
         "model.visual.patch_embed.proj.weight": torch.zeros(2, 2)}
    w = Weights(_snap({"text_config": TEXT}, t), device="cpu", skip_mtp=True, nvfp4="", fp8_head="")
    assert sorted(w.t) == ["embed_tokens.weight", "lm_head.weight", "norm.weight"], sorted(w.t)
    assert w.t["lm_head.weight"] is not w.t["embed_tokens.weight"]
    return "served layout: prefix stripped, vision skipped, own head kept"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<36} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<36} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
