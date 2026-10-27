"""ENG-133/221: the Flash-Next config, its layout, and the architecture registry.

The config is the 3.05bpw_h5_ng5 pack's own `config.json` (tests/fixtures/fn_pack_305_*). A field
the engine does not know, or a value that would change the arithmetic, stops the load with its
name; an unknown architecture stops it with the registered ones. Two gotchas of the architecture
are pinned: `ple_layer_ids` are one-indexed, and the grouped RMS norms weigh by `1 + w` while the
GDN gated norm weighs by plain `w`.
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models import REGISTRY, architecture  # noqa: E402
from engine.models.flashnext.config import config_from_dict  # noqa: E402
from engine.models.flashnext.pack import expected_names  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
RAW = json.load(open(os.path.join(FIX, "fn_pack_305_config.json")))
GEN = json.load(open(os.path.join(FIX, "fn_pack_305_generation_config.json")))


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert all(n in msg for n in needles), msg
        return msg
    raise AssertionError("no error")


def test_pack_config_parses():
    c = config_from_dict(RAW, "pack", GEN)
    assert (c.hidden_size, c.num_hidden_layers, c.num_experts, c.num_experts_per_tok) == (2560, 48, 512, 10)
    assert len(c.attention_layers) == 12 and c.attention_layers[0] == 3 and len(c.linear_layers) == 36
    assert (c.hc_count, c.hc_lowrank, c.ngram_heads, c.ple_head_dim) == (4, 320, 16, 160)
    assert c.rotary_dim == 64 and c.rope_theta == 1e7 and c.output_gate_type == "sigmoid"
    assert c.eos_token_id == 248044 and c.eos_token_ids == [248046, 248044]
    assert c.mtp_num_hidden_layers == 1 and c.indexer_budget == 2048 and c.indexer_compress_ratio == 4
    return "48 layers (36 GDN, 12 QSA), 512 experts top-10, hc 4 x 320, PLE 16 x 160, MTP 1"


def test_ple_ids_are_one_indexed():
    c = config_from_dict(RAW, "pack", GEN)
    assert c.ple_layer_ids == [2] and c.ple_layers == {1: 0}, c.ple_layers
    t = dict(RAW["text_config"], ple_layer_ids=[4])          # layer index 3 is an attention layer
    _raises(lambda: config_from_dict(dict(RAW, text_config=t)), "PLE layer id 4")
    return "id 2 -> in front of layer index 1; id 4 (attention) refused"


def test_unknown_field_fails_by_name():
    t = dict(RAW["text_config"], qsa_new_switch=True)
    return _raises(lambda: config_from_dict(dict(RAW, text_config=t)), "qsa_new_switch")


def test_arithmetic_field_with_other_value_fails():
    for k, v in (("attention_bias", True), ("mamba_ssm_dtype", "bfloat16"), ("indexer_kv_heads", 2)):
        t = dict(RAW["text_config"], **{k: v})
        _raises(lambda: config_from_dict(dict(RAW, text_config=t)), k)
    return "attention_bias, mamba_ssm_dtype, indexer_kv_heads"


def test_registry_selects_by_config():
    assert architecture(RAW).name == "qwen4_exp"
    assert architecture({"architectures": ["Qwen3_5ForConditionalGeneration"]}).name == "qwen3_5"
    assert architecture({"text_config": {"model_type": "qwen4_exp_text"}}).name == "qwen4_exp"
    assert REGISTRY["qwen4_exp"].package == "models/flashnext" and REGISTRY["qwen3_5"].package is None
    return "qwen4_exp by architectures and by text model_type; the 27B by its architectures"


def test_unknown_architecture_names_the_registered_ones():
    return _raises(lambda: architecture({"architectures": ["LlamaForCausalLM"]}),
                   "LlamaForCausalLM", "qwen3_5", "qwen4_exp")


def test_layout_names():
    c = config_from_dict(RAW, "pack", GEN)
    names = expected_names(c)
    assert len(names) == len(set(names))
    experts = [n for n in names if ".experts." in n]
    assert len(experts) == 49 * 512 * 3, len(experts)        # 48 layers + the MTP layer
    assert "layers.1.ple.key_proj.weight" in names and "layers.0.ple.key_proj.weight" not in names
    assert "mtp.hyper_connection_mixer.input_mix_weight_up.weight" in names
    assert "hyper_connection_mixer.block_inject_weight.weight" not in names
    assert "layers.3.self_attn.indexer.index_qk_proj.trellis" in names
    return f"{len(names)} tensors (trellis-coded ones by their .trellis)"


def test_norm_weights_one_plus_w_and_plain_w():
    """The hyper-connection norm weighs by 1 + w: a zero weight leaves the normed streams as they
    are. The GDN gated norm weighs by plain w: a zero weight zeroes the GDN output."""
    from fn_tiny import tiny
    from engine.models.flashnext.ref import FlashNextRef, FNState
    cfg, W, tables = tiny(seed=1)
    ref = FlashNextRef(cfg, W, tables, dtype=torch.float32)
    h = torch.randn(3, cfg.hc_hidden, generator=torch.Generator().manual_seed(2))
    n = ref.rms(h, torch.zeros(cfg.hc_hidden), cfg.rms_norm_eps, group=cfg.hidden_size)
    hg = h.view(3, cfg.hc_count, cfg.hidden_size)
    want = (hg * torch.rsqrt(hg.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)).flatten(1)
    assert torch.allclose(n, want, atol=1e-6)
    W2 = dict(W, **{"layers.0.linear_attn.norm.weight": torch.zeros(cfg.linear_value_head_dim)})
    ref2 = FlashNextRef(cfg, W2, tables, dtype=torch.float32)
    out = ref2.gdn(torch.randn(3, cfg.hidden_size), 0, FNState())
    assert out.abs().max().item() == 0.0
    return "hc_norm: zero weight = plain normed rows; GDN norm: zero weight = zero output"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
