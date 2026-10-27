"""The Flash-Next text configuration (`qwen4_exp`), read from the checkpoint.

Qwen3.8-Flash-Next keeps the 27B's hybrid (GDN layers with an attention layer every
fourth) and changes almost everything around it: a 512-expert MoE in every layer, four residual
streams joined by hyper-connection mixers instead of RMS norms, a hashed n-gram table (PLE) in
front of one layer, and a QSA indexer on the attention layers. Every field any Flash-Next module
reads is resolved here, with the defaults of the reference implementation
(`transformers.models.qwen4_exp.Qwen4ExpTextConfig`) where the checkpoint leaves a field out.

A field this module does not know stops the load with its name (ENG-133): a new release that adds
an arithmetic switch must not be served as if it were the old one. Fields that only matter for
training or for other runtimes are listed in `IGNORED`; fields whose value would change the
arithmetic are checked against the one value the engine implements (`FIXED`).

Two gotchas of the architecture that the tests pin: `ple_layer_ids` are one-indexed (id 2 sits in
front of layer index 1), and the grouped RMS norms weigh by `1 + w` while the GDN gated norm
weighs by plain `w` (engine/models/flashnext/ref.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from engine.config import resolve_snapshot

FAMILY_TYPES = ("qwen4_exp", "qwen3_8_flash_next")

# read by config_from_dict (directly or through a default)
KNOWN = {
    "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
    "vocab_size", "rms_norm_eps", "rope_parameters", "rope_theta", "partial_rotary_factor",
    "max_position_embeddings", "layer_types", "full_attention_interval", "linear_conv_kernel_dim",
    "linear_key_head_dim", "linear_value_head_dim", "linear_num_key_heads", "linear_num_value_heads",
    "output_gate_type", "hidden_act", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
    "shared_expert_intermediate_size", "norm_topk_prob", "hc_count", "hc_lowrank", "ple_layer_ids",
    "ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram", "ngram_vocab_size_base",
    "make_ngram_vocab_size_divisible_by", "seed", "split_ngram_parts", "indexer_n_heads",
    "indexer_head_dim", "indexer_budget", "indexer_compress_ratio", "indexer_kv_heads",
    "eos_token_id", "mtp", "mtp_num_hidden_layers", "model_type",
}
# training, other runtimes, or ids the tokenizer files own
IGNORED = {
    "attention_dropout", "bos_token_id", "pad_token_id", "dtype", "torch_dtype", "initializer_range",
    "output_router_logits", "router_aux_loss_coef", "use_cache", "tie_word_embeddings",
    "transformers_version", "intermediate_size",
}
# fields whose value would change the arithmetic: the one value this engine implements
FIXED = {"attention_bias": False, "mamba_ssm_dtype": "float32", "mtp_use_dedicated_embeddings": False,
         "hidden_act": "silu", "indexer_kv_heads": 1}


def is_flashnext(raw: dict) -> bool:
    """True for a Flash-Next config (top level or its `text_config`)."""
    t = raw.get("text_config") or raw
    mt = str(t.get("model_type", raw.get("model_type", "")))
    return mt.startswith(FAMILY_TYPES) or ("hc_count" in t and "num_experts" in t)


@dataclass
class FlashNextConfig:
    path: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    layer_types: list[str]                  # "linear_attention" | "full_attention"
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    output_gate_type: str                   # GDN gated-norm activation: "sigmoid" | "silu"
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool
    hc_count: int
    hc_lowrank: int
    ple_layer_ids: list[int]                # ONE-indexed, as in the checkpoint
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    seed: int
    split_ngram_parts: int
    indexer_n_heads: int | None
    indexer_head_dim: int | None
    indexer_budget: int | None
    indexer_compress_ratio: int | None
    eos_token_id: int                       # the text config's eos: the PLE hash resets on it
    mtp_num_hidden_layers: int = 0
    eos_token_ids: list[int] = field(default_factory=list)   # generation stop ids
    # the three-axis rotary of image rows (VIS-97; interleaved, engine/vision.py `mrope_axes`)
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])

    # --- derived ---
    @property
    def key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim

    @property
    def num_v_per_k(self) -> int:
        return self.linear_num_value_heads // self.linear_num_key_heads

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def hc_hidden(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ple_head_dim(self) -> int:
        return self.ple_embed_dim // self.ngram_heads

    @property
    def ple_layers(self) -> dict[int, int]:
        """layer index (0-based) -> PLE index (position in `ple_layer_ids`)."""
        return {lid - 1: i for i, lid in enumerate(self.ple_layer_ids)}

    @property
    def has_indexer(self) -> bool:
        return self.indexer_n_heads is not None

    def is_linear(self, layer: int) -> bool:
        return self.layer_types[layer] == "linear_attention"

    @property
    def attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t != "linear_attention"]

    @property
    def linear_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]


def config_from_dict(raw: dict, path: str = "", gen: dict | None = None) -> FlashNextConfig:
    t = raw.get("text_config") or raw
    rope = t.get("rope_parameters") or {}
    n = t["num_hidden_layers"]
    lt = t.get("layer_types")
    if lt is None:
        step = t.get("full_attention_interval", 4)
        lt = ["linear_attention" if (i + 1) % step else "full_attention" for i in range(n)]
    # the reference renames "full_attention" to "qwen_sparse_attention"; the engine keeps one name
    lt = ["linear_attention" if x == "linear_attention" else "full_attention" for x in lt]
    eos = t.get("eos_token_id", raw.get("eos_token_id"))
    eos = eos[0] if isinstance(eos, list) else eos
    if eos is None:
        raise ValueError("Flash-Next needs eos_token_id in its text config (the PLE hash uses it)")
    gen_eos: list[int] = []
    if gen:
        e = gen.get("eos_token_id")
        gen_eos = e if isinstance(e, list) else ([e] if e is not None else [])
    mtp = t.get("mtp") or {}
    hidden = t["hidden_size"]
    cfg = FlashNextConfig(
        path=path,
        hidden_size=hidden,
        num_hidden_layers=n,
        num_attention_heads=t["num_attention_heads"],
        num_key_value_heads=t["num_key_value_heads"],
        head_dim=t.get("head_dim") or hidden // t["num_attention_heads"],
        vocab_size=t["vocab_size"],
        rms_norm_eps=t.get("rms_norm_eps", 1e-6),
        rope_theta=rope.get("rope_theta", t.get("rope_theta", 1e7)),
        partial_rotary_factor=rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)),
        max_position_embeddings=t.get("max_position_embeddings", 262144),
        layer_types=lt,
        linear_conv_kernel_dim=t.get("linear_conv_kernel_dim", 4),
        linear_key_head_dim=t.get("linear_key_head_dim", 128),
        linear_value_head_dim=t.get("linear_value_head_dim", 128),
        linear_num_key_heads=t.get("linear_num_key_heads", 16),
        linear_num_value_heads=t.get("linear_num_value_heads", 32),
        output_gate_type=t.get("output_gate_type") or t.get("hidden_act", "silu"),
        num_experts=t.get("num_experts", 512),
        num_experts_per_tok=t.get("num_experts_per_tok", 10),
        moe_intermediate_size=t["moe_intermediate_size"],
        shared_expert_intermediate_size=t["shared_expert_intermediate_size"],
        norm_topk_prob=bool(t.get("norm_topk_prob", True)),
        hc_count=t.get("hc_count", 4),
        hc_lowrank=t.get("hc_lowrank", 320),
        ple_layer_ids=sorted(set(t.get("ple_layer_ids") or [])),
        ple_embed_dim=t.get("ple_embed_dim") or hidden,
        ple_conv_kernel_size=t.get("ple_conv_kernel_size", 4),
        ngram_size=t.get("ngram_size", 3),
        heads_per_ngram=t.get("heads_per_ngram", 8),
        ngram_vocab_size_base=t.get("ngram_vocab_size_base", 20_000_000),
        make_ngram_vocab_size_divisible_by=t.get("make_ngram_vocab_size_divisible_by", 128),
        seed=t.get("seed", 1234),
        split_ngram_parts=t.get("split_ngram_parts", 512),
        indexer_n_heads=t.get("indexer_n_heads"),
        indexer_head_dim=t.get("indexer_head_dim"),
        indexer_budget=t.get("indexer_budget"),
        indexer_compress_ratio=t.get("indexer_compress_ratio"),
        eos_token_id=int(eos),
        mtp_num_hidden_layers=t.get("mtp_num_hidden_layers", mtp.get("num_hidden_layers", 0)),
        eos_token_ids=gen_eos or [int(eos)],
        mrope_section=list(rope.get("mrope_section", [11, 11, 10])),
    )
    validate(cfg, t)
    return cfg


def validate(cfg: FlashNextConfig, t: dict) -> None:
    """The reference's invariants, plus the ones this engine relies on."""
    unknown = sorted(set(t) - KNOWN - IGNORED - set(FIXED))
    if unknown:
        raise ValueError(f"Flash-Next config field(s) this engine does not know: {', '.join(unknown)}")
    for k, want in FIXED.items():
        if k in t and t[k] != want:
            raise ValueError(f"Flash-Next config {k}={t[k]!r}: this engine implements {want!r} only")
    if cfg.output_gate_type not in ("sigmoid", "silu"):
        raise ValueError(f"unsupported output gate {cfg.output_gate_type}")
    if cfg.hc_count <= 1:
        raise ValueError("Flash-Next needs hc_count > 1")
    if not 0 < cfg.num_experts_per_tok <= cfg.num_experts:
        raise ValueError("num_experts_per_tok out of range")
    if cfg.has_indexer:
        if cfg.indexer_budget % cfg.indexer_compress_ratio:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
        if cfg.rotary_dim > cfg.indexer_head_dim:
            raise ValueError("rotary dims must fit the indexer head")
    for lid in cfg.ple_layer_ids:
        if not 1 <= lid <= cfg.num_hidden_layers or cfg.layer_types[lid - 1] != "linear_attention":
            raise ValueError(f"PLE layer id {lid} must be a 1-indexed linear_attention layer")
    if cfg.ple_layer_ids and cfg.ple_embed_dim % cfg.ngram_heads:
        raise ValueError("ple_embed_dim must be divisible by the number of n-gram heads")


def load_flashnext_config(path: str | None = None) -> FlashNextConfig:
    snap = resolve_snapshot(path)
    with open(os.path.join(snap, "config.json")) as f:
        raw = json.load(f)
    if not is_flashnext(raw):
        raise ValueError(f"{snap}: not a Flash-Next config")
    gen = None
    gp = os.path.join(snap, "generation_config.json")
    if os.path.isfile(gp):
        with open(gp) as f:
            gen = json.load(f)
    return config_from_dict(raw, snap, gen)
