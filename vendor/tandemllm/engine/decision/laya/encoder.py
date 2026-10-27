# Portions of this file are from the Laya SDK 0.3.4 (`laya/agent.py`: `_fix_tokenizer_config`,
# `_verify_compatibility`; `laya/common.py`: `build_model`), Copyright Convai Innovations, licensed
# under the Apache License 2.0 (engine/decision/laya/LICENSE-APACHE-2.0; NOTICE at the repository
# root). Changes: the encoder is always built from the checkpoint's own `encoder/` config (no hub
# download from the engine), and the checks raise one error type.
"""The encoder and tokenizer of one Laya checkpoint, through `transformers`.

Stage A of the decision-model design uses transformers' ModernBERT (and mmBERT, the same class)
exactly as the SDK does -- `AutoModel.from_config(..., attn_implementation="sdpa")`, the weights
from `model.safetensors`, `reference_compile` off -- because then the forward IS the SDK's forward
and the equivalence to it holds by construction (the owner, 2026-10-01 13:00). An own encoder module
comes only in stage C, behind the equivalence gate.

`reference_compile` defaults to "auto" in ModernBERT, which means torch.compile as soon as the
model is on CUDA; the SDK and arbiter turn it off, and so does this: an inductor build inside the
engine process is a compile nobody asked for, and the eager kernels are what the gate measured.
"""

from __future__ import annotations

import json
import os


class CheckpointError(ValueError):
    """A directory that is not a Laya checkpoint this code can load."""


def fix_tokenizer_config(model_dir: str) -> None:
    """The SDK's fix, the same idempotent rewrite: a tokenizer class transformers cannot resolve,
    and mmBERT's list-valued `extra_special_tokens` (transformers wants a mapping)."""
    cfg_file = os.path.join(model_dir, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return
    try:
        with open(cfg_file) as f:
            tcfg = json.load(f)
        changed = False
        if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
            tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tcfg.pop("backend", None)
            tcfg.pop("is_local", None)
            changed = True
        extra = tcfg.get("extra_special_tokens")
        if isinstance(extra, list):
            tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
            changed = True
        if changed:
            with open(cfg_file, "w") as f:
                json.dump(tcfg, f, indent=2)
    except (OSError, ValueError):
        pass


def tokenizer(model_dir: str, cfg: dict):
    from transformers import AutoTokenizer
    tok_dir = os.path.join(model_dir, "tokenizer")
    if os.path.exists(tok_dir):
        return AutoTokenizer.from_pretrained(tok_dir)
    if cfg.get("encoder"):
        return AutoTokenizer.from_pretrained(cfg["encoder"])
    raise CheckpointError(f"{model_dir}: no tokenizer/ directory and no encoder name")


def encoder(model_dir: str):
    """The bare encoder, randomly initialised from `encoder/config.json` (the weights come from
    `model.safetensors` with the head's)."""
    from transformers import AutoConfig, AutoModel
    enc_dir = os.path.join(model_dir, "encoder")
    if not os.path.exists(enc_dir):
        raise CheckpointError(f"{model_dir}: no encoder/ directory")
    enc = AutoModel.from_config(AutoConfig.from_pretrained(enc_dir), attn_implementation="sdpa")
    for module in (enc, getattr(enc, "model", None)):
        cfg = getattr(module, "config", None)
        if cfg is not None:
            cfg.reference_compile = False
    return enc


def verify(model, cfg: dict, weights: dict, where: str) -> None:
    """The weights belong to this architecture: the config keys, the four module prefixes, every
    parameter present with its shape."""
    missing_cfg = [k for k in ("encoder", "head_layers") if k not in cfg]
    if missing_cfg:
        raise CheckpointError(f"{where}: rl_agent_config.json lacks {missing_cfg}")
    for prefix in ("encoder.", "type_emb.", "scorer.", "act_head."):
        if not any(k.startswith(prefix) for k in weights):
            raise CheckpointError(f"{where}: no {prefix!r} parameters in model.safetensors")
    bad, missing = [], []
    for name, param in model.named_parameters():
        if name not in weights:
            missing.append(name)
        elif tuple(weights[name].shape) != tuple(param.shape):
            bad.append(f"{name}: expected {tuple(param.shape)}, found {tuple(weights[name].shape)}")
    if bad:
        raise CheckpointError(f"{where}: shape mismatch: " + "; ".join(bad[:5]))
    if missing:
        raise CheckpointError(f"{where}: {len(missing)} parameters missing (e.g. {missing[:3]})")
