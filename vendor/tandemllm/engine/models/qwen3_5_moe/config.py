"""The Qwen3.5 MoE text configuration (`qwen3_5_moe_text`), read from the checkpoint (ENG-383).

The layer is the 27B's hybrid (Gated DeltaNet layers with a gated attention layer every fourth, the
RMS norms weighing by `1 + w`, the GDN output norm by plain `w`) with a mixture of experts in every
layer: a softmax router over `num_experts`, the top `num_experts_per_tok` renormalised (always, in the
reference: `Qwen3_5MoeTopKRouter`), and one shared expert behind a sigmoid gate. Nothing in the code is
tied to a shape: hidden size, head counts, expert count and width all come from here.

The config object is a `FlashNextConfig` (engine/models/flashnext/config.py) with that family's extras
absent (one residual stream, no n-gram table, no sparse-attention indexer), so the row and prefill
kernels the two families share read it unchanged.

A field this module does not know stops the load with its name (ENG-133); fields whose value would
change the arithmetic are checked against the one value the engine implements (`FIXED`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from engine.config import resolve_snapshot
from engine.models.flashnext.config import FlashNextConfig

FAMILY_TYPES = ("qwen3_5_moe", "qwen3_5_moe_text")

KNOWN = {
    "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
    "vocab_size", "rms_norm_eps", "rope_parameters", "rope_theta", "partial_rotary_factor",
    "max_position_embeddings", "layer_types", "full_attention_interval", "linear_conv_kernel_dim",
    "linear_key_head_dim", "linear_value_head_dim", "linear_num_key_heads", "linear_num_value_heads",
    "num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
    "eos_token_id", "mtp_num_hidden_layers", "model_type",
}
IGNORED = {
    "attention_dropout", "bos_token_id", "pad_token_id", "dtype", "torch_dtype", "initializer_range",
    "output_router_logits", "router_aux_loss_coef", "use_cache", "transformers_version",
    "intermediate_size",            # a dense layer's width: no layer is dense (`mlp_only_layers` empty)
}
FIXED = {"attention_bias": False, "mamba_ssm_dtype": "float32", "mtp_use_dedicated_embeddings": False,
         "hidden_act": "silu", "attn_output_gate": True, "tie_word_embeddings": False, "mlp_only_layers": [],
         "norm_topk_prob": True, "decoder_sparse_step": 1}
#: rope_parameters keys and the values the engine implements
ROPE_FIXED = {"rope_type": "default", "mrope_interleaved": True}
ROPE_KNOWN = {"rope_theta", "partial_rotary_factor", "mrope_section", "rope_type", "mrope_interleaved"}


def is_qwen35_moe(raw: dict) -> bool:
    t = raw.get("text_config") or raw
    return str(t.get("model_type", raw.get("model_type", ""))) in FAMILY_TYPES


@dataclass
class Qwen35MoeConfig(FlashNextConfig):
    """FlashNextConfig with one stream, no PLE and no indexer; `arch` names the family."""
    arch: str = "qwen3_5_moe"
    tie_word_embeddings: bool = False


def config_from_dict(raw: dict, path: str = "", gen: dict | None = None) -> Qwen35MoeConfig:
    t = raw.get("text_config") or raw
    rope = dict(t.get("rope_parameters") or {})
    n = t["num_hidden_layers"]
    lt = t.get("layer_types")
    if lt is None:
        step = t.get("full_attention_interval", 4)
        lt = ["linear_attention" if (i + 1) % step else "full_attention" for i in range(n)]
    eos = t.get("eos_token_id", raw.get("eos_token_id"))
    eos = eos[0] if isinstance(eos, list) else eos
    gen_eos: list[int] = []
    if gen:
        e = gen.get("eos_token_id")
        gen_eos = e if isinstance(e, list) else ([e] if e is not None else [])
    hidden = t["hidden_size"]
    cfg = Qwen35MoeConfig(
        path=path, hidden_size=hidden, num_hidden_layers=n,
        num_attention_heads=t["num_attention_heads"], num_key_value_heads=t["num_key_value_heads"],
        head_dim=t.get("head_dim") or hidden // t["num_attention_heads"], vocab_size=t["vocab_size"],
        rms_norm_eps=t.get("rms_norm_eps", 1e-6),
        rope_theta=rope.get("rope_theta", t.get("rope_theta", 1e7)),
        partial_rotary_factor=rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)),
        max_position_embeddings=t.get("max_position_embeddings", 262144), layer_types=list(lt),
        linear_conv_kernel_dim=t.get("linear_conv_kernel_dim", 4),
        linear_key_head_dim=t.get("linear_key_head_dim", 128),
        linear_value_head_dim=t.get("linear_value_head_dim", 128),
        linear_num_key_heads=t.get("linear_num_key_heads", 16),
        linear_num_value_heads=t.get("linear_num_value_heads", 32),
        output_gate_type="silu", num_experts=t["num_experts"], num_experts_per_tok=t["num_experts_per_tok"],
        moe_intermediate_size=t["moe_intermediate_size"],
        shared_expert_intermediate_size=t["shared_expert_intermediate_size"], norm_topk_prob=True,
        # the Flash-Next extras, absent: one stream, no n-gram table, no indexer
        hc_count=1, hc_lowrank=0, ple_layer_ids=[], ple_embed_dim=hidden, ple_conv_kernel_size=4, ngram_size=3,
        heads_per_ngram=0, ngram_vocab_size_base=0, make_ngram_vocab_size_divisible_by=1, seed=0,
        split_ngram_parts=1, indexer_n_heads=None, indexer_head_dim=None, indexer_budget=None,
        indexer_compress_ratio=None, eos_token_id=int(eos if eos is not None else -1),
        mtp_num_hidden_layers=int(t.get("mtp_num_hidden_layers", 0)),
        eos_token_ids=gen_eos or ([int(eos)] if eos is not None else []),
        mrope_section=list(rope.get("mrope_section", [11, 11, 10])),
    )
    validate(cfg, t, rope)
    return cfg


def validate(cfg: Qwen35MoeConfig, t: dict, rope: dict) -> None:
    unknown = sorted(set(t) - KNOWN - IGNORED - set(FIXED))
    if unknown:
        raise ValueError(f"Qwen3.5 MoE config field(s) this engine does not know: {', '.join(unknown)}")
    for k, want in FIXED.items():
        if k in t and t[k] != want:
            raise ValueError(f"Qwen3.5 MoE config {k}={t[k]!r}: this engine implements {want!r} only")
    unknown = sorted(set(rope) - ROPE_KNOWN)
    if unknown:
        raise ValueError(f"Qwen3.5 MoE rope_parameters field(s) this engine does not know: {', '.join(unknown)}")
    for k, want in ROPE_FIXED.items():
        if k in rope and rope[k] != want:
            raise ValueError(f"Qwen3.5 MoE rope_parameters {k}={rope[k]!r}: this engine implements {want!r} only")
    if sum(cfg.mrope_section) * 2 != cfg.rotary_dim:
        raise ValueError(f"mrope_section {cfg.mrope_section} does not cover the rotary dims {cfg.rotary_dim}")
    if len(cfg.layer_types) != cfg.num_hidden_layers or set(cfg.layer_types) - {"linear_attention", "full_attention"}:
        raise ValueError(f"layer_types must name {cfg.num_hidden_layers} linear_attention/full_attention layers")
    if not 0 < cfg.num_experts_per_tok <= cfg.num_experts:
        raise ValueError("num_experts_per_tok out of range")
    if cfg.linear_num_value_heads % cfg.linear_num_key_heads:
        raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
    if cfg.mtp_num_hidden_layers not in (0, 1):
        raise ValueError(f"mtp_num_hidden_layers={cfg.mtp_num_hidden_layers}: the engine drafts with one MTP layer")


def load_qwen35_moe_config(path: str | None = None) -> Qwen35MoeConfig:
    snap = resolve_snapshot(path)
    with open(os.path.join(snap, "config.json")) as f:
        raw = json.load(f)
    if not is_qwen35_moe(raw):
        raise ValueError(f"{snap}: not a Qwen3.5 MoE config")
    gen = None
    gp = os.path.join(snap, "generation_config.json")
    if os.path.isfile(gp):
        with open(gp) as f:
            gen = json.load(f)
    return config_from_dict(raw, snap, gen)
