"""The text configuration, read from the checkpoint and nothing else.

The checkpoint is a vision-language model. This is the language model's configuration: only
`text_config` is kept here, and the vision tower's own configuration is read by engine/vision.py.
Every field used anywhere in the engine is resolved here
once, so no other module parses JSON or guesses a dimension.
"""

from __future__ import annotations

import glob
import json
import os
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
from dataclasses import dataclass, field

DEFAULT_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots")


def resolve_snapshot(path: str | None = None) -> str:
    """Accept a snapshot directory, a `snapshots` directory, or nothing at all."""
    path = path or _S.get("MODEL") or DEFAULT_MODEL
    path = os.path.expanduser(path)
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    hits = sorted(glob.glob(os.path.join(path, "*", "config.json")))
    if not hits:
        raise FileNotFoundError(f"no config.json under {path}")
    return os.path.dirname(hits[0])


@dataclass
class TextConfig:
    path: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: list[int]
    max_position_embeddings: int
    layer_types: list[str]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    mtp_num_hidden_layers: int
    weight_block_size: tuple[int, int]
    eos_token_ids: list[int] = field(default_factory=list)
    bos_token_id: int | None = None
    # a checkpoint without its own `lm_head.weight` reads the embedding (tied weights)
    tie_word_embeddings: bool = False

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
    def q_dim(self) -> int:
        """`q_proj` emits the query and an output gate of the same width."""
        return self.num_attention_heads * self.head_dim * 2

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attn_out_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    def is_linear(self, layer: int) -> bool:
        return self.layer_types[layer] == "linear_attention"

    @property
    def linear_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    @property
    def attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]


def load_config(path: str | None = None) -> TextConfig:
    snap = resolve_snapshot(path)
    with open(os.path.join(snap, "config.json")) as f:
        raw = json.load(f)
    # the vision-language wrapper keeps the language model's config under `text_config`;
    # a text-only checkpoint of the same family has the same keys at the top level.
    t = raw.get("text_config") or raw
    rope = t.get("rope_parameters", {})
    q = raw.get("quantization_config") or t.get("quantization_config") or {}
    wbs = q.get("weight_block_size", [128, 128])
    gen_path = os.path.join(snap, "generation_config.json")
    eos: list[int] = []
    if os.path.isfile(gen_path):
        with open(gen_path) as f:
            g = json.load(f)
        e = g.get("eos_token_id")
        eos = e if isinstance(e, list) else ([e] if e is not None else [])
    layer_types = t.get("layer_types")
    if layer_types is None:
        step = t.get("full_attention_interval", 4)
        layer_types = ["linear_attention" if (i + 1) % step else "full_attention"
                       for i in range(t["num_hidden_layers"])]
    return TextConfig(
        path=snap,
        hidden_size=t["hidden_size"],
        intermediate_size=t["intermediate_size"],
        num_hidden_layers=t["num_hidden_layers"],
        num_attention_heads=t["num_attention_heads"],
        num_key_value_heads=t["num_key_value_heads"],
        head_dim=t["head_dim"],
        vocab_size=t["vocab_size"],
        rms_norm_eps=t["rms_norm_eps"],
        rope_theta=rope.get("rope_theta", 1e7),
        partial_rotary_factor=rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)),
        mrope_section=rope.get("mrope_section", [11, 11, 10]),
        max_position_embeddings=t["max_position_embeddings"],
        layer_types=layer_types,
        linear_conv_kernel_dim=t["linear_conv_kernel_dim"],
        linear_key_head_dim=t["linear_key_head_dim"],
        linear_value_head_dim=t["linear_value_head_dim"],
        linear_num_key_heads=t["linear_num_key_heads"],
        linear_num_value_heads=t["linear_num_value_heads"],
        mtp_num_hidden_layers=t.get("mtp_num_hidden_layers", 0),
        weight_block_size=(wbs[0], wbs[1]),
        eos_token_ids=eos,
        bos_token_id=t.get("bos_token_id"),
        tie_word_embeddings=bool(t.get("tie_word_embeddings", raw.get("tie_word_embeddings", False))),
    )
