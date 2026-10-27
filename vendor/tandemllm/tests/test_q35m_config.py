"""ENG-383/391: the Qwen3.5 MoE config, its layout and the architecture registry.

The config is the Ornith-1.5-122B EXL3 4.0 pack's own `config.json` (tests/fixtures/ornith_122b_*). A field
the engine does not know, or a value that would change the arithmetic, stops the load with its name; the
same code takes the MiniCPM-V 4.7 35B-A3B text shape (40 layers, hidden 2,048, 32 value heads, width 512).
"""

from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models import architecture  # noqa: E402
from engine.models.qwen3_5_moe.config import config_from_dict  # noqa: E402
from engine.models.qwen3_5_moe.pack import expected_names  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
RAW = json.load(open(os.path.join(FIX, "ornith_122b_exl3_config.json")))
GEN = json.load(open(os.path.join(FIX, "ornith_122b_generation_config.json")))


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError("no ValueError")


def test_ornith_shape():
    c = config_from_dict(RAW, "", GEN)
    assert (c.num_hidden_layers, c.hidden_size, c.num_experts, c.num_experts_per_tok) == (48, 3072, 256, 8)
    assert len(c.linear_layers) == 36 and c.attention_layers == list(range(3, 48, 4))
    assert (c.num_v_per_k, c.key_dim, c.value_dim, c.conv_dim) == (4, 2048, 8192, 12288)
    assert (c.head_dim, c.rotary_dim, c.mrope_section) == (256, 64, [11, 11, 10])
    assert c.hc_count == 1 and not c.ple_layers and not c.has_indexer and c.output_gate_type == "silu"
    assert c.eos_token_ids == [248046, 248044] and c.mtp_num_hidden_layers == 1


def test_minicpm_v47_text_shape():
    t = copy.deepcopy(RAW["text_config"])
    t.update(num_hidden_layers=40, hidden_size=2048, num_attention_heads=16, linear_num_value_heads=32,
             moe_intermediate_size=512, shared_expert_intermediate_size=512, vocab_size=248144,
             mtp_num_hidden_layers=1)
    t["layer_types"] = t["layer_types"][:40]
    c = config_from_dict({"text_config": t})
    assert (c.num_v_per_k, c.value_dim, c.num_attention_heads // c.num_key_value_heads) == (2, 4096, 8)
    assert len(expected_names(c, mtp=False)) > 0


def test_unknown_or_changed_fields_stop_the_load():
    bad = copy.deepcopy(RAW)
    bad["text_config"]["router_scaling_factor"] = 2.5
    _raises(lambda: config_from_dict(bad), "router_scaling_factor")
    for k, v in (("attn_output_gate", False), ("hidden_act", "gelu"), ("mlp_only_layers", [3]),
                 ("norm_topk_prob", False), ("tie_word_embeddings", True)):
        bad = copy.deepcopy(RAW)
        bad["text_config"][k] = v
        _raises(lambda: config_from_dict(bad), k)
    bad = copy.deepcopy(RAW)
    bad["text_config"]["rope_parameters"]["rope_type"] = "yarn"
    _raises(lambda: config_from_dict(bad), "rope_type")
    bad = copy.deepcopy(RAW)
    bad["text_config"]["mtp_num_hidden_layers"] = 2
    _raises(lambda: config_from_dict(bad), "mtp_num_hidden_layers")


def test_the_registry_picks_the_family():
    assert architecture(RAW).name == "qwen3_5_moe"
    assert architecture({"architectures": ["Qwen3_5ForConditionalGeneration"]}).name == "qwen3_5"


def test_the_layout_names_every_part():
    c = config_from_dict(RAW, "", GEN)
    names = expected_names(c, mtp=True)
    assert "layers.0.linear_attn.in_proj_qkv.trellis" in names and "layers.3.self_attn.q_proj.trellis" in names
    assert "mtp.fc.trellis" in names and "mtp.layers.0.mlp.experts.255.down_proj.trellis" in names
    assert sum(1 for n in names if ".experts." in n and not n.startswith("mtp.")) == 48 * 256 * 3


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
