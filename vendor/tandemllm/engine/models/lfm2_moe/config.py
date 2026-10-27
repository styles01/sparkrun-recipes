"""The LFM2.5 MoE configuration (`lfm2_moe`), read from the checkpoint (ENG-283).

LiquidAI's LFM2 MoE models (LFM2-8B-A1B, LFM2.5-24B-A2B) interleave two kinds of layer: a
double-gated short convolution (`conv`: in_proj to B, C and x, y = C * conv(B * x) over a kernel of
`conv_L_cache` taps, out_proj) and grouped-query attention with per-head q/k RMS norms
(`full_attention`). After each sits an MLP: dense SwiGLU in the first `num_dense_layers` layers, a
mixture of `num_experts` SwiGLU experts in the others, routed by sigmoid scores with a per-expert
bias for the choice (`use_expert_bias`) and the chosen scores renormalised. Every field any
lfm2_moe module reads is resolved here, with the defaults of the reference implementation
(`transformers.models.lfm2_moe.Lfm2MoeConfig`) where the checkpoint leaves a field out.

As for Flash-Next (ENG-133), a field this module does not know stops the load with its name, and
fields whose value would change the arithmetic are checked against the one value the engine
implements (`FIXED`). An EXL3 pack's `config.json` is the checkpoint's with exllamav3's
`quantization_config` added; the pack reader takes that block (engine/models/lfm2_moe/pack.py).

Two things the tests pin: the RMS norms weigh by plain `w` (not Flash-Next's `1 + w`), and the
final norm is called `embedding_norm` although it runs after the last layer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from engine.config import resolve_snapshot

FAMILY_TYPES = ("lfm2_moe",)
ARCHITECTURES = ("Lfm2MoeForCausalLM",)

# read by config_from_dict (directly or through a default)
KNOWN = {
    "hidden_size", "intermediate_size", "moe_intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "norm_eps", "rope_parameters", "rope_theta",
    "max_position_embeddings", "layer_types", "conv_L_cache", "num_dense_layers", "num_experts",
    "num_experts_per_tok", "routed_scaling_factor", "use_expert_bias", "eos_token_id", "bos_token_id",
    "pad_token_id", "model_type", "architectures",
}
# training, other runtimes, or what the tokenizer files own
IGNORED = {
    "dtype", "torch_dtype", "initializer_range", "output_router_logits", "use_cache", "transformers_version",
    "quantization_config", "full_attn_idxs", "block_dim", "block_ff_dim", "block_multiple_of",
    "block_ffn_dim_multiplier", "block_auto_adjust_ff_dim", "router_aux_loss_coef", "attention_dropout",
}
# fields whose value would change the arithmetic: the one value this engine implements
FIXED = {"conv_bias": False, "hidden_act": "silu", "norm_topk_prob": True, "tie_word_embeddings": True,
         "tie_embedding": True, "attention_bias": False}


def is_lfm2_moe(raw: dict) -> bool:
    t = raw.get("text_config") or raw
    return str(t.get("model_type", raw.get("model_type", ""))) in FAMILY_TYPES or any(
        a in ARCHITECTURES for a in raw.get("architectures") or [])


@dataclass
class Lfm2MoeConfig:
    path: str
    hidden_size: int
    intermediate_size: int                 # the dense layers' MLP
    moe_intermediate_size: int             # an expert's
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    layer_types: list[str]                 # "conv" | "full_attention"
    conv_L_cache: int                      # the short conv's taps (its state keeps conv_L_cache - 1 inputs)
    num_dense_layers: int
    num_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    use_expert_bias: bool
    eos_token_id: int
    bos_token_id: int | None = None
    eos_token_ids: list[int] = field(default_factory=list)   # generation stop ids

    # --- derived ---
    @property
    def conv_state_len(self) -> int:
        return self.conv_L_cache - 1

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def gqa(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def is_attention(self, layer: int) -> bool:
        return self.layer_types[layer] == "full_attention"

    def is_moe(self, layer: int) -> bool:
        return layer >= self.num_dense_layers

    @property
    def attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def conv_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "conv"]

    @property
    def moe_layers(self) -> list[int]:
        return list(range(self.num_dense_layers, self.num_hidden_layers))

    def kv_bytes_per_token(self, elem: int = 2) -> int:
        """K and V of every attention layer for one token (bf16: 20,480 bytes on LFM2.5-24B-A2B)."""
        return len(self.attention_layers) * 2 * self.kv_dim * elem

    def conv_state_bytes(self, elem: int = 2) -> int:
        """The short conv's state of a sequence: the last conv_L_cache - 1 inputs of every conv layer."""
        return len(self.conv_layers) * self.conv_state_len * self.hidden_size * elem


def config_from_dict(raw: dict, path: str = "", gen: dict | None = None) -> Lfm2MoeConfig:
    t = raw.get("text_config") or raw
    n = t["num_hidden_layers"]
    lt = t.get("layer_types")
    if lt is None:
        # transformers' fallback when the checkpoint lists its attention layers by index
        idx = set(t.get("full_attn_idxs") or [])
        lt = ["full_attention" if i in idx else "conv" for i in range(n)]
    rope = t.get("rope_parameters") or {}
    eos = t.get("eos_token_id", raw.get("eos_token_id"))
    eos = eos[0] if isinstance(eos, list) else eos
    if eos is None:
        raise ValueError("LFM2 MoE needs eos_token_id in its config")
    gen_eos: list[int] = []
    if gen:
        e = gen.get("eos_token_id")
        gen_eos = e if isinstance(e, list) else ([e] if e is not None else [])
    hidden = t["hidden_size"]
    cfg = Lfm2MoeConfig(
        path=path,
        hidden_size=hidden,
        intermediate_size=t["intermediate_size"],
        moe_intermediate_size=t["moe_intermediate_size"],
        num_hidden_layers=n,
        num_attention_heads=t["num_attention_heads"],
        num_key_value_heads=t.get("num_key_value_heads", t["num_attention_heads"]),
        head_dim=t.get("head_dim") or hidden // t["num_attention_heads"],
        vocab_size=t["vocab_size"],
        rms_norm_eps=t.get("norm_eps", 1e-5),
        rope_theta=float(rope.get("rope_theta", t.get("rope_theta", 1e6))),
        max_position_embeddings=t.get("max_position_embeddings", 128_000),
        layer_types=list(lt),
        conv_L_cache=t.get("conv_L_cache", 3),
        num_dense_layers=t.get("num_dense_layers", 2),
        num_experts=t.get("num_experts", 32),
        num_experts_per_tok=t.get("num_experts_per_tok", 4),
        routed_scaling_factor=float(t.get("routed_scaling_factor", 1.0)),
        use_expert_bias=bool(t.get("use_expert_bias", True)),
        eos_token_id=int(eos),
        bos_token_id=t.get("bos_token_id"),
        eos_token_ids=[int(x) for x in gen_eos] or [int(eos)],
    )
    validate(cfg, t)
    return cfg


def validate(cfg: Lfm2MoeConfig, t: dict) -> None:
    unknown = sorted(set(t) - KNOWN - IGNORED - set(FIXED))
    if unknown:
        raise ValueError(f"LFM2 MoE config field(s) this engine does not know: {', '.join(unknown)}")
    for k, want in FIXED.items():
        if k in t and t[k] != want:
            raise ValueError(f"LFM2 MoE config {k}={t[k]!r}: this engine implements {want!r} only")
    rope = t.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"LFM2 MoE rope_type {rope.get('rope_type')!r}: this engine implements 'default' only")
    if len(cfg.layer_types) != cfg.num_hidden_layers or set(cfg.layer_types) - {"conv", "full_attention"}:
        raise ValueError(f"layer_types must list conv or full_attention for each of {cfg.num_hidden_layers} layers")
    if cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
    if cfg.head_dim % 2:
        raise ValueError("the rotary needs an even head_dim")
    if not 0 < cfg.num_experts_per_tok <= cfg.num_experts:
        raise ValueError("num_experts_per_tok out of range")
    if not 0 <= cfg.num_dense_layers <= cfg.num_hidden_layers:
        raise ValueError("num_dense_layers out of range")
    if cfg.conv_L_cache < 2:
        raise ValueError("conv_L_cache must be at least 2")


def load_lfm2_moe_config(path: str | None = None) -> Lfm2MoeConfig:
    snap = resolve_snapshot(path)
    with open(os.path.join(snap, "config.json")) as f:
        raw = json.load(f)
    if not is_lfm2_moe(raw):
        raise ValueError(f"{snap}: not an LFM2 MoE config")
    gen = None
    gp = os.path.join(snap, "generation_config.json")
    if os.path.isfile(gp):
        with open(gp) as f:
            gen = json.load(f)
    return config_from_dict(raw, snap, gen)
