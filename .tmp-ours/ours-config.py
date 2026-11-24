# SPDX-License-Identifier: Apache-2.0
"""Quantization config `deepseek_v41_exl3`: DeepSeek-V4.1's own FP8/MXFP8 handling everywhere,
EXL3 trellis banks for the backbone routed experts only.

Upstream anchors (vLLM main @ 6e517b15c1833cf72a7f557ee32524d98682e617):
  * register_quantization_config      model_executor/layers/quantization/__init__.py:61-109
  * custom methods are probed first    config/model.py:1397-1431 (_verify_quantization)
  * from_config(hf quantization_config) model_executor/model_loader/weight_utils.py:247-265
  * the V4.1 base class we extend      models/deepseek_v41/quant_config.py:33-231 (what
    "deepseek_v4_fp8" resolves to on CUDA, quantization/__init__.py:120-128)
  * Fp8Config.from_config keys on "fp8" in quant_method   quantization/fp8.py:158-175

Only RoutedExperts of backbone layers listed in the plan are redirected. Everything else goes to
super(): MXFP8 [32, 32] linears (attention, shared expert, compressor, indexer) via
ModelOptLinearMethod, the DSpark drafter's MXFP4 experts (layer index >= num_hidden_layers, see
models/deepseek_v41/nvidia/dspark.py:126-128) via Mxfp4MoEMethod, and any layer the plan leaves out.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

from .plan import Exl3MoEPlan

logger = init_logger(__name__)

QUANT_METHOD = "deepseek_v41_exl3"
_ROUTED_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.experts(?:\.|$)")


_ENGRAM_WKV = re.compile(r"layers\.(\d+)\.engram\.wkv$")   # V4.1 Engram layers 1 and 14 (exl3_moe.dense.engram.wkv: K)

@dataclass
class Exl3Settings:
    """The `exl3_moe` block inside config.json's quantization_config."""
    plan_file: str | None = "exl3_moe_plan.json"   # relative to the model dir
    plan: dict[str, Any] | None = None              # inline plan (wins over plan_file)
    decode_kernel: str = "coop"                     # coop (exl3_moe_coop) | fused (exl3_moe)
    coop_max_tokens: int = 32                       # rows per call on the coop path (x top_k <= 256)
    fused_rows: int = 256                           # exl3_moe rows per expert before reconstruct
    tp_layout: str = "channels"                     # channels = vLLM MoE TP split (only one in v0)
    exllamav3_version: str | None = None            # informational: encoder version
    head: str = "bf16"                              # bf16 (checkpoint) | mxfp8 (Marlin, needs patch 0005)
                                                    # | exl3 (EXL3 tensors head.*, needs 0005 + 0007)
    head_k: int | None = None                       # K of the EXL3 head
    dense: dict[str, Any] | None = None             # EXL3 dense linears (dense.py, needs patch 0006):
    #   {"k": {"attn.wq_a": 5, "attn.wq_b": 5, "attn.wo_a": 5, "attn.wo_b": 5, "ffn.shared_experts.w1": 5,
    #          "ffn.shared_experts.w2": 5}, "layers": [first, last]}
    #   keys are checkpoint module names; w1 covers the gate_up pair (w1 + w3), wq_a the fused_wqa_wkv pair (wq_a + wkv)
    #   "drafter": {"attn.wq_a": 5, "attn.wq_b": 5, "attn.wo_b": 5, "ffn.shared_experts.w1": 5,
    #               "ffn.shared_experts.w2": 5}: the DSpark drafter's dense linears from mtp.* EXL3 tensors
    #   (scripts/quantize/p4_drafter.py); modules not listed stay FP8 (wo_a always: its grouped FP8 path). main_proj
    #   reads the target's raw residual stream (|x| up to ~1.6e5): above 16 rows (exllamav3's path) its input is cast
    #   to fp16 under a power-of-two scale (dense.py), which overflowed to NaN without it (LOG Oct 10 09:18)
    #   "engram": {"wkv": 4}: V4.1's Engram projections (layers.<1|14>.engram.wkv, FP8 6144 -> 25600 column-parallel) on
    #   EXL3 at that K, from the variant's layers.<i>.engram.wkv.* tensors (scripts/quantize/p4_engram.py)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Exl3Settings":
        s = cls(raw=dict(d))
        for k in ("plan_file", "plan", "decode_kernel", "coop_max_tokens", "fused_rows", "tp_layout",
                  "exllamav3_version", "head", "head_k", "dense"):
            if k in d:
                setattr(s, k, d[k])
        # runtime overrides for A/B sweeps (must be identical on every rank)
        s.decode_kernel = os.environ.get("VLLM_EXL3_MOE_DECODE_KERNEL", s.decode_kernel)
        s.coop_max_tokens = int(os.environ.get("VLLM_EXL3_MOE_COOP_MAX_TOKENS", s.coop_max_tokens))
        s.fused_rows = int(os.environ.get("VLLM_EXL3_MOE_FUSED_ROWS", s.fused_rows))
        s.head = os.environ.get("VLLM_EXL3_HEAD", s.head)
        if s.head not in ("bf16", "mxfp8", "exl3"):
            raise ValueError(f"exl3_moe.head must be bf16, mxfp8 or exl3, got {s.head!r}")
        if s.head == "exl3" and not s.head_k:
            raise ValueError("exl3_moe.head exl3 needs head_k (the K the head was encoded at)")
        if s.dense is not None:
            from .dense import CKPT_NAME
            bad = set(s.dense.get("k", {})) - set(CKPT_NAME.values())
            if bad:
                raise ValueError(f"exl3_moe.dense.k: unknown modules {sorted(bad)}; known {sorted(set(CKPT_NAME.values()))}")
            # 1017 packs ship the drafter wo_a as EXL3 head-group slices (unlike the pre-1017 FP8 .weight),
            # so attn.wo_a is now a legal drafter entry (oracle 2026-10-18);
            # attn.wkv / shared w3 ride as shard 1 of the fused wq_a / gate_up mappings, not their own keys.
            bad = set(s.dense.get("drafter", {})) - (set(CKPT_NAME.values()) | {"main_proj"})
            if bad:
                raise ValueError(f"exl3_moe.dense.drafter: unsupported modules {sorted(bad)}")
        if s.decode_kernel not in ("coop", "fused"):
            raise ValueError(f"exl3_moe.decode_kernel must be coop or fused, got {s.decode_kernel!r}")
        if s.tp_layout != "channels":
            raise NotImplementedError("exl3_moe.tp_layout: only 'channels' (vLLM's MoE TP split) is implemented")
        return s

    def drafter_k(self, module: str) -> int | None:
        """K for a DSpark drafter module (vLLM path, or "main_proj"), or None to keep the checkpoint's FP8."""
        from .dense import CKPT_NAME
        if not self.dense:
            return None
        k = self.dense.get("drafter", {}).get(CKPT_NAME.get(module, module))
        return int(k) if k else None

    def dense_k(self, module: str, layer: int) -> int | None:
        """K for vLLM module path `module` of backbone layer `layer`, or None to keep the checkpoint's FP8."""
        from .dense import CKPT_NAME
        if not self.dense:
            return None
        lo, hi = self.dense.get("layers", [0, 1 << 30])
        if not lo <= layer <= hi:
            return None
        k = self.dense.get("k", {}).get(CKPT_NAME[module])
        return int(k) if k else None



def _resolve_plan_file(name: str) -> str:
    """plan_file relative to the model directory, or fetched from the HF repo the model came from.
    Not tested: the HF-hub branch (local snapshot dirs are the tested case)."""
    if os.path.isabs(name):
        return name
    from vllm.config import get_current_vllm_config

    mc = get_current_vllm_config().model_config
    model = str(mc.model)
    if os.path.isdir(model):
        return os.path.join(model, name)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model, filename=name, revision=getattr(mc, "revision", None))


def _num_hidden_layers() -> int | None:
    try:
        from vllm.config import get_current_vllm_config

        return int(get_current_vllm_config().model_config.hf_text_config.num_hidden_layers)
    except Exception:  # noqa: BLE001
        return None


def routed_layer_index(prefix: str) -> int | None:
    m = _ROUTED_LAYER_RE.search(prefix)
    return int(m.group(1)) if m else None


# --- checkpoint-derived per-tensor K (2026-10-18 oracle) -------------------------------
# The config map's single K per module cannot express packs whose bitrates vary per layer
# (1017 packs: shared experts 5/4 by layer with a mixed layer 29, wq_a+wkv at 6bpw on layer
# 0, head 6). The tensor itself is the truth: an EXL3 int16 trellis is
# [in/16, out/16, 16*K] (plan.py trellis_shape), so K = shape[-1] // 16. Resolve per
# module instance from the shard header at get_quant_method time; fall back to the map.

_DENSE_TENSORS = {
    "attn.fused_wqa_wkv": ("layers.{L}.attn.wq_a.trellis", "layers.{L}.attn.wkv.trellis"),
    "attn.wq_b": ("layers.{L}.attn.wq_b.trellis",),
    "attn.wo_a": ("layers.{L}.attn.wo_a.slice.0.trellis", "layers.{L}.attn.wo_a.slice.4.trellis"),
    "attn.wo_b": ("layers.{L}.attn.wo_b.trellis",),
    "ffn.shared_experts.gate_up_proj": (
        "layers.{L}.ffn.shared_experts.w1.trellis",
        "layers.{L}.ffn.shared_experts.w3.trellis",
    ),
    "ffn.shared_experts.down_proj": ("layers.{L}.ffn.shared_experts.w2.trellis",),
    "compressor.fused_wkv_wgate": (
        "layers.{L}.attn.compressor.wkv.trellis",
        "layers.{L}.attn.compressor.wgate.trellis",
    ),
    "indexer.wk": ("layers.{L}.attn.indexer.wk.trellis",),
    "indexer.wq_b": ("layers.{L}.attn.indexer.wq_b.trellis",),
    "engram.wkv": ("layers.{L}.engram.wkv.trellis",),
    "main_proj": ("main_proj.trellis",),
    "lm_head": ("head.trellis",),
}

_HDR: dict[str, dict] = {}
_IDX_CACHE: dict[str, dict] = {}


def _model_dir() -> str | None:
    try:
        from vllm.config import get_current_vllm_config

        model = str(get_current_vllm_config().model_config.model)
        return model if os.path.isdir(model) else None
    except Exception:  # noqa: BLE001
        return None


def _trellis_k_from_header(mdir: str, shard: str, tensor: str) -> int | None:
    path = os.path.join(mdir, shard)
    key = f"{path}:{tensor}"
    if key in _HDR:
        return _HDR[key]
    k: int | None = None
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        entry = hdr.get(tensor)
        if entry and len(entry["shape"]) == 3 and entry["dtype"] in ("I16", "INT16"):
            k = int(entry["shape"][2]) // 16
    except Exception:  # noqa: BLE001
        k = None
    _HDR[key] = k
    return k


def _dense_ckpt_k(module: str, layer: int, mtp: int | None = None) -> int | None:
    """Real K for a dense module instance from the checkpoint (None -> use the config map).

    `mtp` pins the DSpark drafter stage directly (the drafter's layers share the
    layers.0..2 path space with the target, so index-based inference is ambiguous) —
    the checkpoint names are then mtp.{mtp}.{module-pattern}.
    """
    mdir = _model_dir()
    if mdir is None or module == "main_proj":
        return None
    patterns = _DENSE_TENSORS.get(module)
    if patterns is None:
        return None
    idx = _IDX_CACHE.get(mdir)
    if idx is None:
        try:
            with open(os.path.join(mdir, "model.safetensors.index.json")) as f:
                idx = json.load(f).get("weight_map", {})
        except Exception:  # noqa: BLE001
            return None
        _IDX_CACHE[mdir] = idx
    if mtp is not None:
        mtp_prefix = f"mtp.{mtp}."
    else:
        mtp_prefix = ""
        try:
            if layer >= _num_hidden_layers():
                mtp_prefix = f"mtp.{layer - _num_hidden_layers()}."
        except Exception:  # noqa: BLE001
            mtp_prefix = ""
    if layer < 0:
        return None
    ks: set[int] = set()
    for p in patterns:
        name = mtp_prefix + p.replace("{L}", str(layer))
        shard = idx.get(name)
        if shard is None:
            continue
        k = _trellis_k_from_header(mdir, shard, name)
        if k:
            ks.add(k)
    if not ks:
        return None
    if len(ks) > 1:
        logger.warning("EXL3 dense %s.%s: trellis shapes disagree on K %s; using min", module, mtp_prefix or layer, sorted(ks))
    return min(ks)


class DeepseekV41Exl3Config(DeepseekV4FP8Config):
    """`quant_method: "deepseek_v41_exl3"`: the V4.1 FP8 config plus EXL3 routed-expert banks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exl3_settings = Exl3Settings()
        self._exl3_plan: Exl3MoEPlan | None = None

    @classmethod
    def get_name(cls):  # type: ignore[override]
        return QUANT_METHOD

    @classmethod
    def get_min_capability(cls) -> int:
        # mma.sync m16n8k16 + cp.async + 90 KB dynamic smem: sm_80 and up (exllamav3 arch.cuh:5-27)
        return max(80, super().get_min_capability())

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # Claim only our own checkpoints. Without this override the inherited V4.1 one would claim
        # every "fp8" DeepSeek-V4 checkpoint (quant_config.py:138-167) from the front of the probe
        # list where custom methods sit (config/model.py:1397-1404).
        if isinstance(hf_quant_cfg, dict) and hf_quant_cfg.get("quant_method") == QUANT_METHOD:
            return QUANT_METHOD
        if user_quant == QUANT_METHOD:
            return QUANT_METHOD
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepseekV41Exl3Config":
        settings = Exl3Settings.from_dict(dict(config.get("exl3_moe") or {}))
        base = {k: v for k, v in config.items() if k != "exl3_moe"}
        # Fp8Config.from_config derives is_checkpoint_fp8_serialized from "fp8" in quant_method
        # (fp8.py:159-160): hand it the original FP8 description of the non-expert tensors.
        base["quant_method"] = "fp8"
        inst = super().from_config(base)
        assert isinstance(inst, DeepseekV41Exl3Config)
        inst.exl3_settings = settings
        return inst

    @property
    def exl3_plan(self) -> Exl3MoEPlan:
        if self._exl3_plan is None:
            s = self.exl3_settings
            if s.plan is not None:
                self._exl3_plan = Exl3MoEPlan.from_dict(s.plan)
            elif s.plan_file:
                path = _resolve_plan_file(s.plan_file)
                with open(path) as f:
                    self._exl3_plan = Exl3MoEPlan.from_dict(json.load(f))
                logger.info("EXL3 MoE plan %s: %d layers, %.3f bpw, groups %s", path,
                            len(self._exl3_plan.layers), self._exl3_plan.bpw(),
                            self._exl3_plan.group_histogram())
            else:
                raise ValueError("quantization_config.exl3_moe needs plan or plan_file")
        return self._exl3_plan

    def _is_drafter_layer(self, idx: int, n_backbone: int | None) -> bool:
        """True when layer index `idx` belongs to a DSpark drafter, not the backbone.

        The drafter's decoder layers are model.layers.{0,1,2} — the same path space as the
        backbone's earliest layers. Trust first the quant config itself (only the draft's
        instance carries the drafter map with a non-empty drafter section AND is built by
        get_draft_quant_config), else fall back to index >= backbone layers.
        """
        if n_backbone is not None and idx >= n_backbone:
            return True
        try:
            return bool(getattr(self, "_draft_config", False))
        except Exception:  # noqa: BLE001
            return False

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and self.exl3_settings.dense:
            from .dense import Exl3DenseLinearMethod, Exl3DequantDenseLinearMethod, dense_module

            if prefix.endswith("compressor.fused_wkv_wgate"):
                m = re.search(r"layers\.(\d+)\.attn\.compressor", prefix)
                if m:
                    k = _dense_ckpt_k("compressor.fused_wkv_wgate", int(m.group(1)))
                    if k:
                        return Exl3DequantDenseLinearMethod("compressor.fused_wkv_wgate", k, layer_idx=int(m.group(1)), prefetch=False)

            if prefix.endswith("indexer.wk"):
                m = re.search(r"layers\.(\d+)\.attn", prefix)
                if m:
                    k = _dense_ckpt_k("indexer.wk", int(m.group(1)))
                    if k:
                        return Exl3DenseLinearMethod("indexer.wk", k, layer_idx=int(m.group(1)), prefetch=False)

            if prefix.endswith("indexer.wq_b"):
                m = re.search(r"layers\.(\d+)\.attn", prefix)
                if m:
                    k = _dense_ckpt_k("indexer.wq_b", int(m.group(1)))
                    if k:
                        return Exl3DenseLinearMethod("indexer.wq_b", k, layer_idx=int(m.group(1)), prefetch=False)

            dm = dense_module(prefix)
            n_backbone = _num_hidden_layers()
            _draft_ctx = bool(getattr(self, "_draft_config", False))
            if dm is not None and not _draft_ctx and (n_backbone is None or dm[0] < n_backbone):
                k = _dense_ckpt_k(dm[1], dm[0]) or self.exl3_settings.dense_k(dm[1], dm[0])
                logger.info("oracle-qm: BACKBONE %s L%s nb=%s k=%s id=%s", dm[1], dm[0], n_backbone, k, id(self))
                if k:
                    return Exl3DenseLinearMethod(dm[1], k, layer_idx=dm[0])
            elif dm is not None and _draft_ctx:
                # DSpark drafter layer: the checkpoint tensors live under mtp.{stage};
                # 1017 packs ship the drafter wo_a as EXL3 head-group slices (pre-1017: FP8 .weight).
                k = (_dense_ckpt_k(dm[1], dm[0], mtp=dm[0])
                     or self.exl3_settings.drafter_k(dm[1]))
                logger.info("oracle-draft-quant: %s layer=%s stage=%s -> k=%s", dm[1], dm[0], dm[0], k)
                if k:
                    return Exl3DenseLinearMethod(dm[1], k, layer_idx=dm[0], prefetch=False)
            elif prefix.endswith("main_proj"):                 # the drafter's context combiner (replicated)
                _mp = re.search(r"mtp\.(\d+)\.main_proj", prefix)
                k = (_dense_ckpt_k("main_proj", (_num_hidden_layers() or 40) + int(_mp.group(1)))
                     if _mp else None) or self.exl3_settings.drafter_k("main_proj")
                if k:
                    return Exl3DenseLinearMethod("main_proj", k, layer_idx=-1, prefetch=False)
            else:
                m = _ENGRAM_WKV.search(prefix)
                k = ((_dense_ckpt_k("engram.wkv", int(m.group(1)))
                      or int(self.exl3_settings.dense.get("engram", {}).get("wkv", 0))) if m else 0)
                if k:                                         # Engram's hash-row projection (column-parallel, gathered)
                    return Exl3DenseLinearMethod("engram.wkv", k, layer_idx=int(m.group(1)), prefetch=False)
        if isinstance(layer, ParallelLMHead) and prefix.endswith("lm_head"):
            # Only reached with patch 0005 (V4.1 builds its head without a quant config otherwise). Only the lm_head:
            # another vocab projection handed this config (e.g. a DSpark Markov head, markov_w2) has no head.* pieces.
            if self.exl3_settings.head == "mxfp8":
                from .head import Mxfp8MarlinHeadMethod

                return Mxfp8MarlinHeadMethod()
            if self.exl3_settings.head == "exl3":
                from .head import Exl3HeadMethod

                return Exl3HeadMethod(_dense_ckpt_k("lm_head", 0) or self.exl3_settings.head_k)
            return None                                  # -> UnquantizedEmbeddingMethod, as without the patch
        if isinstance(layer, RoutedExperts):
            idx = routed_layer_index(prefix)
            n_backbone = _num_hidden_layers()
            plan = self.exl3_plan
            is_backbone = idx is not None and (n_backbone is None or idx < n_backbone)
            _draft = bool(getattr(self, "_draft_config", False))
            logger.info("oracle-routed: %s idx=%s nb=%s is_backbone=%s draft=%s in_plan=%s in_drafter=%s",
                        prefix, idx, n_backbone, is_backbone, _draft,
                        idx in plan.layers if idx is not None else None,
                        idx in plan.drafter_layers if idx is not None else None)
            if (is_backbone and idx in plan.layers) or (idx is not None and not is_backbone and idx in plan.drafter_layers):
                from .moe_method import Exl3RoutedMoEMethod

                return Exl3RoutedMoEMethod(layer.moe_config, plan, idx, self.exl3_settings, prefix)
            # Layers the plan leaves native keep the checkpoint's own MXFP4 method: the DSpark drafter's experts
            # (idx >= num_hidden_layers) unless the plan's "drafter" section lists them (p4_drafter_experts.py).
        method = super().get_quant_method(layer, prefix)
        if isinstance(layer, LinearBase) and method is not None:
            from . import hcap
            if hcap.enabled(prefix):                     # VLLM_EXL3_HCAP: input Hessians of FP8 linears (hcap.py)
                return hcap.HCaptureMethod(method, prefix)
        return method


def register_config() -> None:
    """Idempotent: vLLM may load general plugins several times per process (plugins/__init__.py:77-91)."""
    register_quantization_config(QUANT_METHOD)(DeepseekV41Exl3Config)
