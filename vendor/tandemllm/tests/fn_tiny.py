"""A tiny random Flash-Next for the CPU tests that do not need `transformers` (ENG-221).

The real layer structure (3 GDN layers and one attention layer with a QSA indexer, 8 experts top-2,
4 hyper-connection streams, PLE in front of layer index 1, an MTP layer) at sizes a laptop runs in
fp32, under the canonical names the reference reads. `tiny()` -> (cfg, W, tables).
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.config import config_from_dict  # noqa: E402
from engine.models.flashnext.ple import DenseTable, head_sizes  # noqa: E402
from engine.models.flashnext.ref import ExpertBank  # noqa: E402

TINY = dict(
    model_type="qwen4_exp_text", vocab_size=64, hidden_size=32, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, hidden_act="silu",
    max_position_embeddings=256, rms_norm_eps=1e-6,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
    linear_num_key_heads=2, linear_num_value_heads=4, output_gate_type="sigmoid",
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
    shared_expert_intermediate_size=16, intermediate_size=64,
    hc_count=4, hc_lowrank=8, ple_layer_ids=[2], ple_embed_dim=32, ple_conv_kernel_size=4,
    ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=101,
    make_ngram_vocab_size_divisible_by=16, split_ngram_parts=1,
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=8,
    indexer_compress_ratio=2, eos_token_id=3, bos_token_id=3, pad_token_id=None,
    tie_word_embeddings=False, mtp_num_hidden_layers=1,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [1, 1, 0], "mrope_interleaved": True},
)


def tiny(seed: int = 0, dtype=torch.float32, **over):
    cfg = config_from_dict(dict(TINY, **over))
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.2):
        return (torch.randn(*shape, generator=g) * scale).to(dtype)

    c = cfg
    H, R, HC = c.hidden_size, c.hc_lowrank, c.hc_hidden
    W: dict = {"embed_tokens.weight": r(c.vocab_size, H, scale=1.0), "lm_head.weight": r(c.vocab_size, H)}

    def mixers(p, inject=True):
        W[p + ".hc_norm.weight"] = r(HC, scale=0.1)
        W[p + ".input_mix_weight_down.weight"] = r(R, HC)
        W[p + ".input_mix_weight_up.weight"] = r(HC, R)
        if inject:
            W[p + ".block_inject_weight.weight"] = r(c.hc_count, HC)

    def attn(p):
        D = c.head_dim
        W[p + "self_attn.q_proj.weight"] = r(c.num_attention_heads * 2 * D, H)
        W[p + "self_attn.k_proj.weight"] = r(c.num_key_value_heads * D, H)
        W[p + "self_attn.v_proj.weight"] = r(c.num_key_value_heads * D, H)
        W[p + "self_attn.o_proj.weight"] = r(H, c.num_attention_heads * D)
        W[p + "self_attn.q_norm.weight"] = r(D, scale=0.1)
        W[p + "self_attn.k_norm.weight"] = r(D, scale=0.1)
        hd = c.indexer_head_dim
        W[p + "self_attn.indexer.index_qk_proj.weight"] = r((c.indexer_n_heads + 1) * hd, H)
        W[p + "self_attn.indexer.q_layernorm.weight"] = r(hd, scale=0.1)
        W[p + "self_attn.indexer.k_layernorm.weight"] = r(hd, scale=0.1)

    def moe(p):
        E, I = c.num_experts, c.moe_intermediate_size
        W[p + "mlp.gate.weight"] = r(E, H, scale=0.5)
        W[p + "mlp.shared_expert_gate.weight"] = r(1, H)
        for x in ("gate", "up"):
            W[p + f"mlp.shared_expert.{x}_proj.weight"] = r(c.shared_expert_intermediate_size, H)
        W[p + "mlp.shared_expert.down_proj.weight"] = r(H, c.shared_expert_intermediate_size)
        W[p + "mlp.experts.gate_proj"] = ExpertBank(dense=r(E, I, H))
        W[p + "mlp.experts.up_proj"] = ExpertBank(dense=r(E, I, H))
        W[p + "mlp.experts.down_proj"] = ExpertBank(dense=r(E, H, I))

    for li in range(c.num_hidden_layers):
        p = f"layers.{li}."
        mixers(p + "attn_hyper_connection")
        mixers(p + "mlp_hyper_connection")
        if c.is_linear(li):
            a = p + "linear_attn."
            Hv = c.linear_num_value_heads
            W[a + "in_proj_qkv.weight"] = r(c.conv_dim, H)
            W[a + "in_proj_z.weight"] = r(c.value_dim, H)
            W[a + "in_proj_b.weight"] = r(Hv, H)
            W[a + "in_proj_a.weight"] = r(Hv, H)
            W[a + "conv1d.weight"] = r(c.conv_dim, 1, c.linear_conv_kernel_dim, scale=0.5)
            W[a + "A_log"] = (torch.rand(Hv, generator=g) * 2 - 1).to(dtype)
            W[a + "dt_bias"] = torch.rand(Hv, generator=g).to(dtype)
            W[a + "norm.weight"] = r(c.linear_value_head_dim, scale=0.3)
            W[a + "out_proj.weight"] = r(H, c.value_dim)
        else:
            attn(p)
        moe(p)
    tables = {}
    for li, pi in c.ple_layers.items():
        p = f"layers.{li}.ple."
        W[p + "key_proj.weight"] = r(HC, c.ple_embed_dim)
        W[p + "value_proj.weight"] = r(H, c.ple_embed_dim)
        for n in ("norm_key", "norm_query", "norm_conv"):
            W[p + n + ".weight"] = r(HC, scale=0.1)
        W[p + "conv1d.weight"] = r(HC, 1, c.ple_conv_kernel_size, scale=0.5)
        rows = head_sizes(c.ngram_vocab_size_base, c.ngram_heads, pi)[2]
        tables[li] = DenseTable(r(rows, c.ple_head_dim, scale=1.0))
    mixers("hyper_connection_mixer", inject=False)
    if c.mtp_num_hidden_layers:
        p = "mtp.layers.0."
        mixers(p + "attn_hyper_connection")
        mixers(p + "mlp_hyper_connection")
        attn(p)
        moe(p)
        W["mtp.fc_embedding.weight"] = r(H, H)
        W["mtp.fc_hidden.weight"] = r(H, H)
        W["mtp.pre_fc_norm_embedding.weight"] = r(H, scale=0.1)
        W["mtp.pre_fc_norm_hidden.weight"] = r(HC, scale=0.1)
        mixers("mtp.hyper_connection_mixer", inject=False)
    return cfg, W, tables
