"""The NemotronH configuration (`nemotron_h`, `nemotron_h_puzzle`, the Omni wrappers), read from the
checkpoint into one layer table.

NVIDIA's Nemotron 3 / 3.5 models (Lightning 30B-A3B, Super 120B-A12B, the Puzzle 75B-A9B, Bolt
74B-A9B) stack three kinds of block, each a pre-norm residual block with one mixer:

  mamba       Mamba2: in_proj to z | xBC | dt, a causal depthwise conv of `conv_kernel` taps with bias
              and silu over xBC, the selective state update per head with B and C shared by
              `n_groups` groups, D skip, a gated group RMS norm (norm after the gate), out_proj
  attention   grouped-query attention with no positional encoding (NoPE: no rotary at all)
  moe         a sigmoid router with a correction bias for the choice, top k renormalised and
              scaled, non-gated relu^2 experts (up, down), optionally behind a latent pair
              (fc1_latent_proj / fc2_latent_proj), plus one shared relu^2 expert at full width

The layer list comes in three spellings, all read here into `layer_types` and the per-layer MoE
table (`moe_k`, `moe_ffn`, `moe_experts`):

  hybrid_override_pattern   "MEM*E..." (Super, Nano-Omni's llm_config)
  layers_block_type         ["mamba", "moe", ...] (Lightning, Puzzle)
  block_configs             one dict a block (Puzzle, Bolt): block_type and the blockwise members
                            num_experts_per_tok, moe_intermediate_size, n_routed_experts

An Omni checkpoint (`NemotronH_*_Omni_Reasoning_V3`) keeps the text trunk under `llm_config`; it is
read from there and the vision and sound towers are not. As for the other families, a field this
module does not know stops the load with its name, and a field whose other values would change the
arithmetic is checked against the one value the engine implements (`FIXED`).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from engine.config import resolve_snapshot

FAMILY_TYPES = ("nemotron_h", "nemotron_h_puzzle")
ARCHITECTURES = ("NemotronHForCausalLM", "NemotronHPuzzleForCausalLM", "NemotronH_Nano_Omni_Reasoning_V3",
                 "NemotronH_Omni_Reasoning_V3")
PATTERN = {"M": "mamba", "E": "moe", "*": "attention"}
BLOCK_TYPES = ("mamba", "moe", "attention")
#: fields a block of `block_configs` may set (the Puzzle's blockwise members and Bolt's expert count)
BLOCKWISE = {"block_type", "num_experts_per_tok", "moe_intermediate_size", "n_routed_experts"}

# read by config_from_dict (directly or through a default)
KNOWN = {
    "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "layer_norm_epsilon",
    "norm_eps", "max_position_embeddings", "layers_block_type", "hybrid_override_pattern", "block_configs",
    "num_hidden_layers", "mamba_num_heads", "mamba_head_dim", "ssm_state_size", "n_groups", "conv_kernel",
    "chunk_size", "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "moe_latent_size",
    "moe_shared_expert_intermediate_size", "routed_scaling_factor", "num_nextn_predict_layers",
    "mtp_layers_block_type", "mtp_hybrid_override_pattern", "mtp_block_configs", "tie_word_embeddings",
    "eos_token_id", "bos_token_id", "pad_token_id", "model_type", "architectures", "intermediate_size",
    "rms_norm_eps",
}
# training, initialisation, other runtimes' switches, or fields with no effect on this arithmetic
IGNORED = {
    "dtype", "torch_dtype", "initializer_range", "use_cache", "transformers_version", "auto_map",
    "attention_dropout", "hidden_dropout", "rescale_prenorm_residual", "num_logits_to_keep", "use_mamba_kernels",
    "moe_shared_expert_overlap", "time_step_floor", "time_step_min", "time_step_max", "expand",
    "blockwise_members", "quantization_config", "_megatron_bridge_hf_prefix", "aux_loss_alpha",
    "router_aux_loss_coef",
    # the unpruned expert count (Bolt: 512); each MoE block's own n_routed_experts is read
    "num_experts", "num_local_experts",
    # the MTP blocks come from mtp_layers_block_type / mtp_block_configs
    "mtp_num_hidden_layers",
    # NoPE: the rotary fields are dead config (the reference applies no positional encoding)
    "partial_rotary_factor", "rope_theta",
}
# fields whose value would change the arithmetic: the one value this engine implements
FIXED = {"mlp_hidden_act": "relu2", "mamba_hidden_act": "silu", "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
         "residual_in_fp32": False, "use_conv_bias": True, "mamba_proj_bias": False, "attention_bias": False,
         "mlp_bias": False, "use_bias": False, "sliding_window": None, "n_shared_experts": 1,
         "mamba_ssm_cache_dtype": "float32", "hidden_act": "relu2", "scoring_func": "sigmoid", "use_qk_norm": False}
# the Omni wrapper's own keys: the towers and their tokens (text is served, these are not read)
OMNI_IGNORED = {
    "llm_config", "vision_config", "sound_config", "max_sequence_length", "downsample_ratio", "force_image_size",
    "patch_size", "use_thumbnail", "ps_version", "template", "image_tag_type", "img_context_token_id",
    "video_context_token_id", "img_context_token", "video_context_token", "img_start_token", "img_end_token",
    "vit_hidden_size", "projector_hidden_size", "norm_mean", "norm_std", "video_pruning_rate",
    "sound_context_token_id", "sound_context_token", "eos_token_id", "model_type", "architectures", "auto_map",
    "torch_dtype", "dtype", "transformers_version", "bos_token_id", "pad_token_id", "tie_word_embeddings",
    # Bolt's wrapper (NemotronH_Omni_Reasoning_V3): the processor, the video patches, the tower's last norm
    "processor_class", "video_temporal_patch_size", "vision_final_layernorm",
    # an EXL3 pack's (exllamav3 writes it at the wrapper's level): engine/models/nemotron_h/pack.py reads it
    "quantization_config",
}


def is_nemotron_h(raw: dict) -> bool:
    t = raw.get("llm_config") or raw
    return str(t.get("model_type", "")) in FAMILY_TYPES or any(
        a in ARCHITECTURES for a in (raw.get("architectures") or []) + (t.get("architectures") or []))


@dataclass
class NemotronHConfig:
    path: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    max_position_embeddings: int
    layer_types: list[str]                  # "mamba" | "moe" | "attention", one a layer
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    n_groups: int
    conv_kernel: int
    chunk_size: int
    moe_k: dict                             # MoE layer -> its top k
    moe_ffn: dict                           # MoE layer -> an expert's intermediate width
    moe_experts: dict                       # MoE layer -> its routed expert count
    moe_latent_size: int | None
    shared_ffn: int                         # the shared expert's intermediate width
    routed_scaling_factor: float
    tie_word_embeddings: bool
    eos_token_id: int
    mtp_layer_types: list[str] = field(default_factory=list)   # the MTP head's blocks ("attention", "moe")
    mtp_moe: dict = field(default_factory=dict)               # MTP block -> (k, ffn, experts) of its MoE
    bos_token_id: int | None = None
    eos_token_ids: list[int] = field(default_factory=list)
    omni: bool = False                      # read from an Omni wrapper's llm_config
    architecture: str = ""

    # --- derived ---
    @property
    def num_hidden_layers(self) -> int:
        return len(self.layer_types)

    @property
    def intermediate(self) -> int:
        """Mamba2's inner width (heads x head_dim)."""
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def conv_dim(self) -> int:
        """The conv's channels: x | B | C."""
        return self.intermediate + 2 * self.n_groups * self.ssm_state_size

    @property
    def in_proj_dim(self) -> int:
        """in_proj's outputs: z | xBC | dt."""
        return self.intermediate + self.conv_dim + self.mamba_num_heads

    @property
    def conv_state_len(self) -> int:
        return self.conv_kernel - 1

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def gqa(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def expert_in(self) -> int:
        """The routed experts' input width: the latent width, else the hidden width."""
        return self.moe_latent_size or self.hidden_size

    def layers_of(self, kind: str) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == kind]

    @property
    def mamba_layers(self) -> list[int]:
        return self.layers_of("mamba")

    @property
    def attention_layers(self) -> list[int]:
        return self.layers_of("attention")

    @property
    def moe_layers(self) -> list[int]:
        return self.layers_of("moe")

    def kv_bytes_per_token(self, elem: int = 2) -> int:
        """K and V of every attention layer for one token (bf16: 6,144 bytes on Lightning's 6 layers)."""
        return len(self.attention_layers) * 2 * self.kv_dim * elem

    def ssm_state_bytes(self) -> int:
        """The fp32 SSM states of a sequence: heads x head_dim x state of every Mamba layer."""
        return len(self.mamba_layers) * self.mamba_num_heads * self.mamba_head_dim * self.ssm_state_size * 4

    def conv_state_bytes(self, elem: int = 2) -> int:
        """The conv states of a sequence: the last conv_kernel - 1 inputs of every Mamba layer."""
        return len(self.mamba_layers) * self.conv_state_len * self.conv_dim * elem

    def table(self) -> list[dict]:
        """The layer table, one row a layer (what `nh_check.py config` prints)."""
        out = []
        for li, t in enumerate(self.layer_types):
            row = {"layer": li, "type": t}
            if t == "moe":
                row.update(k=self.moe_k[li], ffn=self.moe_ffn[li], experts=self.moe_experts[li])
            out.append(row)
        return out


def _layer_list(t: dict) -> tuple[list[str], list[dict] | None]:
    """(layer types, block dicts or None) from whichever spelling the config uses."""
    if "block_configs" in t:
        blocks = list(t["block_configs"])
        for i, b in enumerate(blocks):
            extra = sorted(set(b) - BLOCKWISE)
            if extra:
                raise ValueError(f"NemotronH block_configs[{i}] field(s) this engine does not know: {', '.join(extra)}")
        types = [str(b["block_type"]) for b in blocks]
        if "layers_block_type" in t and list(t["layers_block_type"]) != types:
            raise ValueError("NemotronH block_configs and layers_block_type disagree")
        return types, blocks
    if "layers_block_type" in t:
        return [str(x) for x in t["layers_block_type"]], None
    if "hybrid_override_pattern" in t:
        pat = str(t["hybrid_override_pattern"])
        bad = sorted(set(pat) - set(PATTERN))
        if bad:
            raise ValueError(f"NemotronH hybrid_override_pattern has block symbol(s) {''.join(bad)!r}: "
                             "this engine runs M, E and * only")
        return [PATTERN[ch] for ch in pat], None
    raise ValueError("NemotronH config has no layer list (block_configs, layers_block_type or hybrid_override_pattern)")


def _mtp_list(t: dict) -> tuple[list[str], list[dict] | None]:
    if not int(t.get("num_nextn_predict_layers", 0) or 0) and "mtp_block_configs" not in t:
        return [], None
    if "mtp_block_configs" in t:
        blocks = list(t["mtp_block_configs"])
        for i, b in enumerate(blocks):
            extra = sorted(set(b) - BLOCKWISE)
            if extra:
                raise ValueError(f"NemotronH mtp_block_configs[{i}] field(s) this engine does not know: "
                                 f"{', '.join(extra)}")
        return [str(b["block_type"]) for b in blocks], blocks
    if "mtp_hybrid_override_pattern" in t:
        return [PATTERN[ch] for ch in str(t["mtp_hybrid_override_pattern"])], None
    return [str(x) for x in t.get("mtp_layers_block_type") or ["attention", "moe"]], None


def _moe_of(b: dict | None, t: dict, what: str) -> tuple[int, int, int]:
    """(k, ffn, experts) of one MoE block: the block's own value, else the global field."""
    b = b or {}

    def get(name):
        v = b.get(name, t.get(name))
        if v is None:
            raise ValueError(f"NemotronH {what}: no {name} (neither in the block nor global)")
        return int(v)
    return get("num_experts_per_tok"), get("moe_intermediate_size"), get("n_routed_experts")


def config_from_dict(raw: dict, path: str = "", gen: dict | None = None) -> NemotronHConfig:
    omni = "llm_config" in raw
    if omni:
        unknown = sorted(set(raw) - OMNI_IGNORED)
        if unknown:
            raise ValueError(f"NemotronH Omni wrapper field(s) this engine does not know: {', '.join(unknown)}")
    t = raw["llm_config"] if omni else raw
    validate_fields(t)
    types, blocks = _layer_list(t)
    if "num_hidden_layers" in t and int(t["num_hidden_layers"]) != len(types):
        raise ValueError(f"NemotronH num_hidden_layers {t['num_hidden_layers']} != {len(types)} listed layers")
    bad = sorted(set(types) - set(BLOCK_TYPES))
    if bad:
        raise ValueError(f"NemotronH block type(s) {', '.join(bad)}: this engine runs {', '.join(BLOCK_TYPES)}")
    k, ffn, ne = {}, {}, {}
    for li, ty in enumerate(types):
        b = blocks[li] if blocks is not None else None
        if ty == "moe":
            k[li], ffn[li], ne[li] = _moe_of(b, t, f"layer {li}")
        elif b is not None and set(b) - {"block_type"}:
            raise ValueError(f"NemotronH block_configs[{li}] ({ty}) sets {sorted(set(b) - {'block_type'})}")
    mtypes, mblocks = _mtp_list(t)
    mtp_moe = {}
    for j, ty in enumerate(mtypes):
        if ty not in ("attention", "moe"):
            raise ValueError(f"NemotronH MTP block {j} is {ty!r}: the head runs attention and moe blocks only")
        if ty == "moe":
            mtp_moe[j] = _moe_of(mblocks[j] if mblocks is not None else None, t, f"MTP block {j}")
    eps = float(t.get("layer_norm_epsilon", t.get("norm_eps", 1e-5)))
    for name in ("norm_eps", "rms_norm_eps"):
        if name in t and float(t[name]) != eps:
            raise ValueError(f"NemotronH {name} {t[name]} != layer_norm_epsilon {eps}")
    eos = raw.get("eos_token_id", t.get("eos_token_id")) if omni else t.get("eos_token_id")
    eos_list = eos if isinstance(eos, list) else [eos]
    if omni and t.get("eos_token_id") is not None and t["eos_token_id"] not in eos_list:
        eos_list = eos_list + [t["eos_token_id"]]
    if eos_list[0] is None:
        raise ValueError("NemotronH needs eos_token_id in its config")
    gen_eos: list[int] = []
    if gen:
        e = gen.get("eos_token_id")
        gen_eos = e if isinstance(e, list) else ([e] if e is not None else [])
    ids = [int(x) for x in eos_list]
    for x in gen_eos:
        if int(x) not in ids:
            ids.append(int(x))
    hidden = int(t["hidden_size"])
    latent = t.get("moe_latent_size")
    archs = (raw.get("architectures") or []) + (t.get("architectures") or [])
    cfg = NemotronHConfig(
        path=path,
        hidden_size=hidden,
        num_attention_heads=int(t["num_attention_heads"]),
        num_key_value_heads=int(t.get("num_key_value_heads") or t["num_attention_heads"]),
        head_dim=int(t.get("head_dim") or hidden // int(t["num_attention_heads"])),
        vocab_size=int(t["vocab_size"]),
        rms_norm_eps=eps,
        max_position_embeddings=int(t.get("max_position_embeddings", 4096)),
        layer_types=types,
        mamba_num_heads=int(t["mamba_num_heads"]),
        mamba_head_dim=int(t["mamba_head_dim"]),
        ssm_state_size=int(t["ssm_state_size"]),
        n_groups=int(t["n_groups"]),
        conv_kernel=int(t.get("conv_kernel", 4)),
        chunk_size=int(t.get("chunk_size", 128)),
        moe_k=k, moe_ffn=ffn, moe_experts=ne,
        moe_latent_size=int(latent) if latent else None,
        shared_ffn=int(t["moe_shared_expert_intermediate_size"]) if "moe" in types else 0,
        routed_scaling_factor=float(t.get("routed_scaling_factor", 1.0)),
        tie_word_embeddings=bool(t.get("tie_word_embeddings", False)),
        eos_token_id=ids[0],
        mtp_layer_types=mtypes,
        mtp_moe=mtp_moe,
        bos_token_id=t.get("bos_token_id"),
        eos_token_ids=ids,
        omni=omni,
        architecture=str(archs[0]) if archs else "",
    )
    validate(cfg)
    return cfg


def validate_fields(t: dict) -> None:
    unknown = sorted(set(t) - KNOWN - IGNORED - set(FIXED) - {"time_step_limit"})
    if unknown:
        raise ValueError(f"NemotronH config field(s) this engine does not know: {', '.join(unknown)}")
    for k, want in FIXED.items():
        if k in t and t[k] != want:
            raise ValueError(f"NemotronH config {k}={t[k]!r}: this engine implements {want!r} only")
    lim = t.get("time_step_limit")
    if lim is not None:
        # Bolt's config spells infinity as {"__float__": "Infinity"} (JSON has no inf)
        lim = [float(v["__float__"]) if isinstance(v, dict) else float(v) for v in lim]
    if lim is not None and (lim[0] != 0.0 or not (math.isinf(lim[1]) or lim[1] >= 1e30)):
        raise ValueError(f"NemotronH time_step_limit {lim!r}: this engine implements [0, inf] only (no dt clamp)")


def validate(cfg: NemotronHConfig) -> None:
    if cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
    if cfg.mamba_num_heads % cfg.n_groups:
        raise ValueError("mamba_num_heads must be a multiple of n_groups")
    if cfg.intermediate % cfg.n_groups:
        raise ValueError("the gated norm's groups must divide mamba_num_heads x mamba_head_dim")
    if cfg.conv_kernel < 2:
        raise ValueError("conv_kernel must be at least 2")
    for li in cfg.moe_layers:
        if not 0 < cfg.moe_k[li] <= cfg.moe_experts[li]:
            raise ValueError(f"layer {li}: num_experts_per_tok {cfg.moe_k[li]} out of range "
                             f"(1..{cfg.moe_experts[li]})")
    if not cfg.attention_layers:
        raise ValueError("NemotronH without attention layers: this engine needs at least one")


def load_nemotron_h_config(path: str | None = None) -> NemotronHConfig:
    snap = resolve_snapshot(path)
    with open(os.path.join(snap, "config.json")) as f:
        raw = json.load(f)
    if not is_nemotron_h(raw):
        raise ValueError(f"{snap}: not a NemotronH config")
    gen = None
    gp = os.path.join(snap, "generation_config.json")
    if os.path.isfile(gp):
        with open(gp) as f:
            gen = json.load(f)
    return config_from_dict(raw, snap, gen)
