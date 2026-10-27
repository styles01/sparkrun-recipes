"""The NemotronH config reader on the five published configs (Lightning 30B-A3B, Super 120B-A12B,
Puzzle 75B-A9B, Nano-Omni 30B-A3B, Bolt 74B-A9B): one layer table from each of the three spellings and from an Omni
wrapper, per-layer MoE rows, and a stop by name on a field the engine does not know."""

from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import architecture  # noqa: E402
from engine.models.nemotron_h.config import config_from_dict  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "nemotron_h")


def raw(name: str) -> dict:
    with open(os.path.join(FIX, name + ".json")) as f:
        return json.load(f)


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return str(e)
    raise AssertionError(f"no ValueError naming {needles}")


def test_lightning_layers_block_type():
    c = config_from_dict(raw("lightning-30b-a3b"))
    assert c.num_hidden_layers == 52
    assert (len(c.mamba_layers), len(c.moe_layers), len(c.attention_layers)) == (23, 23, 6)
    assert c.attention_layers == [5, 12, 19, 26, 33, 42]
    assert set(c.moe_k.values()) == {6} and set(c.moe_ffn.values()) == {1856} and set(c.moe_experts.values()) == {128}
    assert c.moe_latent_size is None and c.expert_in == 2688 and c.shared_ffn == 3712
    assert (c.intermediate, c.conv_dim, c.in_proj_dim) == (4096, 6144, 10304)
    assert c.mtp_layer_types == ["attention", "moe"] and c.mtp_moe == {1: (6, 1856, 128)}
    assert c.kv_bytes_per_token() == 6 * 2 * 2 * 128 * 2
    assert c.eos_token_ids == [2] and not c.omni


def test_super_hybrid_override_pattern():
    c = config_from_dict(raw("super-120b-a12b"))
    assert c.num_hidden_layers == 88
    assert (len(c.mamba_layers), len(c.moe_layers), len(c.attention_layers)) == (40, 40, 8)
    assert set(c.moe_k.values()) == {22} and c.moe_latent_size == 1024 and c.expert_in == 1024
    assert c.mtp_layer_types == ["attention", "moe"]


def test_puzzle_block_configs_per_layer():
    c = config_from_dict(raw("puzzle-75b-a9b"))
    assert c.num_hidden_layers == 88 and len(c.moe_layers) == 40 and len(c.attention_layers) == 8
    assert c.ssm_state_size == 96 and c.mamba_num_heads == 128
    blocks = raw("puzzle-75b-a9b")["block_configs"]
    for li in c.moe_layers:
        assert c.moe_k[li] == blocks[li]["num_experts_per_tok"]
        assert c.moe_ffn[li] == blocks[li]["moe_intermediate_size"]
        assert c.moe_experts[li] == 512
    assert min(c.moe_k.values()) == 4 and max(c.moe_k.values()) == 18
    assert sorted(set(c.moe_ffn.values())) == [1280, 1536, 1792, 2048, 2688]
    assert c.mtp_moe == {1: (22, 2688, 512)}


def test_bolt_style_per_block_expert_count():
    r = raw("puzzle-75b-a9b")
    r["block_configs"][1] = {**r["block_configs"][1], "n_routed_experts": 384}
    c = config_from_dict(r)
    assert c.moe_experts[1] == 384 and c.moe_experts[3] == 512


def test_omni_wrapper_reads_llm_config():
    r = raw("nano-omni-30b-a3b")
    c = config_from_dict(r)
    assert c.omni and c.num_hidden_layers == 52 and c.eos_token_id == 11
    assert c.layer_types == config_from_dict(raw("lightning-30b-a3b")).layer_types
    assert architecture(r).name == "nemotron_h"


def test_bolt_omni_block_configs():
    # NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 at 153d4551: the Omni wrapper over a Puzzle trunk with a
    # pruned expert count a layer, state 128 (not the Puzzle's 96), top 9 everywhere and no MTP head
    r = raw("bolt-74b-a9b")
    c = config_from_dict(r)
    assert c.omni and c.architecture == "NemotronH_Omni_Reasoning_V3"
    assert c.num_hidden_layers == 88
    assert (len(c.mamba_layers), len(c.moe_layers), len(c.attention_layers)) == (40, 40, 8)
    assert c.attention_layers == [7, 16, 25, 36, 47, 58, 69, 78]
    assert (c.ssm_state_size, c.mamba_num_heads, c.mamba_head_dim, c.n_groups) == (128, 128, 64, 8)
    assert (c.intermediate, c.conv_dim, c.in_proj_dim) == (8192, 10240, 18560)
    assert c.moe_latent_size == 1024 and c.shared_ffn == 5376 and c.routed_scaling_factor == 5.0
    assert set(c.moe_k.values()) == {9}
    assert sorted(set(c.moe_experts.values())) == [240, 304, 368, 432, 496]
    assert sorted(set(c.moe_ffn.values())) == [1536, 2048, 2304, 2560]
    blocks = r["llm_config"]["block_configs"]
    for li in c.moe_layers:
        assert (c.moe_k[li], c.moe_ffn[li], c.moe_experts[li]) == (
            blocks[li]["num_experts_per_tok"], blocks[li]["moe_intermediate_size"], blocks[li]["n_routed_experts"])
    assert c.mtp_layer_types == [] and c.mtp_moe == {}
    assert c.eos_token_ids == [11, 2] and c.max_position_embeddings == 262144 and not c.tie_word_embeddings
    assert c.kv_bytes_per_token() == 8 * 2 * 2 * 128 * 2


def test_bolt_new_fields_are_checked():
    for k, v in (("hidden_act", "swiglu"), ("scoring_func", "softmax"), ("use_qk_norm", True)):
        r = copy.deepcopy(raw("bolt-74b-a9b"))
        r["llm_config"][k] = v
        _raises(lambda: config_from_dict(r), k)
    r = copy.deepcopy(raw("bolt-74b-a9b"))
    r["llm_config"]["rms_norm_eps"] = 1e-6
    _raises(lambda: config_from_dict(r), "rms_norm_eps")
    r = copy.deepcopy(raw("bolt-74b-a9b"))
    r["llm_config"]["time_step_limit"] = [0.0, {"__float__": "0.1"}]
    _raises(lambda: config_from_dict(r), "time_step_limit")


def test_registry_selects_every_spelling():
    for n in ("lightning-30b-a3b", "super-120b-a12b", "puzzle-75b-a9b", "nano-omni-30b-a3b", "bolt-74b-a9b"):
        assert architecture(raw(n)).name == "nemotron_h", n


def test_unknown_field_stops_by_name():
    r = raw("lightning-30b-a3b")
    r["mamba_window"] = 4
    _raises(lambda: config_from_dict(r), "mamba_window")
    r = raw("puzzle-75b-a9b")
    r["block_configs"][1] = {**r["block_configs"][1], "moe_shared_expert_intermediate_size": 4096}
    _raises(lambda: config_from_dict(r), "block_configs[1]", "moe_shared_expert_intermediate_size")
    r = raw("nano-omni-30b-a3b")
    r["audio_tower_v2"] = {}
    _raises(lambda: config_from_dict(r), "audio_tower_v2")


def test_fixed_values_stop():
    for k, v in (("mlp_hidden_act", "swiglu"), ("n_group", 2), ("residual_in_fp32", True), ("sliding_window", 4096),
                 ("use_conv_bias", False)):
        r = raw("lightning-30b-a3b")
        r[k] = v
        _raises(lambda: config_from_dict(r), k)
    r = copy.deepcopy(raw("nano-omni-30b-a3b"))
    r["llm_config"]["time_step_limit"] = [0.001, 0.1]
    _raises(lambda: config_from_dict(r), "time_step_limit")


def test_unknown_block_symbol_stops():
    r = raw("super-120b-a12b")
    r["hybrid_override_pattern"] = r["hybrid_override_pattern"][:-1] + "-"
    _raises(lambda: config_from_dict(r), "'-'")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all nemotron_h config tests passed")
