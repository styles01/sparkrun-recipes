"""Load NemotronH weights onto the device, in the forms the engine multiplies by.

  W[name]          the reference's canonical names (engine/models/nemotron_h/ref.py) as bf16 tensors;
                   the router's correction bias, A_log, dt_bias and D as fp32 (the reference reads
                   them in fp32)
  W[fused]         layers.L.mixer.qkv: q | k | v of an attention layer in one weight (one launch)
  experts[L]       the routed experts of MoE layer L, stacked (`ExpertsRelu2`: up [E, I, X], down
                   [E, X, I], X the latent width or the hidden width); the MTP head's MoE block J
                   under ("mtp", J)

From an EXL3 pack (ENG-E12) every projection the pack quantised is an `EXL3Block` (tools/exl3_linear.py;
the head with fp32 logits), q | k | v one EXL3 group where the three share a rate, and the routed experts
of a layer a `Relu2Bank` (tools/exl3_relu2.py); what the pack stored in fp16 (the norms, the conv taps, the
router, the embedding) is cast back to bf16, which is exact for the BF16 checkpoint's values.

From NVIDIA's ModelOpt checkpoint (Super VL's NVFP4: routed and shared experts NVFP4, the Mamba2 and attention
projections and the head FP8) an NVFP4 projection is an `NVFP4Wide` on the W4A16 kernels (tools/nvfp4_relu2.py;
a down projection with its input scaled into fp16's range), an FP8 one an `FP8Linear` (the head with fp32 logits),
and the routed experts of a layer an `NVFP4Bank`.

`load_any` takes either form of checkpoint: a GGUF (a `*.gguf` file, a directory holding one, or the file
QSE_NH_GGUF names for a directory of config and tokenizer: gguf_load.py) or the BF16 safetensors below.

Each expert is copied into its stack slot straight from the file, so the host holds one expert at a
time beside the device copy, and the shards' page cache is dropped after every layer (a 66 GB
checkpoint's cache beside its device copy does not fit a unified-memory board).
"""

from __future__ import annotations

import os
import time

import torch

from engine.models.nemotron_h.pack import PackIndex, expected

FP32 = ("gate.e_score_correction_bias", "mixer.A_log", "mixer.dt_bias", "mixer.D")


class ExpertsRelu2:
    """The routed experts of one MoE block, stacked: up [E, I, X], down [E, X, I], bf16."""

    def __init__(self, up: torch.Tensor, down: torch.Tensor):
        self.up, self.down = up, down
        self.E, self.I, self.X = up.shape

    @property
    def nbytes(self) -> int:
        return (self.up.numel() + self.down.numel()) * 2

    def expert(self, e: int):
        return self.up[e], self.down[e]


def _experts(idx: PackIndex, prefix: str, E: int, I: int, X: int, device, dtype) -> ExpertsRelu2:
    up = torch.empty(E, I, X, dtype=dtype, device=device)
    down = torch.empty(E, X, I, dtype=dtype, device=device)
    for e in range(E):
        up[e].copy_(idx.read(prefix + f"experts.{e}.up_proj.weight"))
        down[e].copy_(idx.read(prefix + f"experts.{e}.down_proj.weight"))
    return ExpertsRelu2(up, down)


def _exl3_experts(idx: PackIndex, prefix: str, E: int, device):
    from tools.exl3_relu2 import Relu2Bank
    return Relu2Bank([(idx.exl3(prefix + f"experts.{e}.up_proj", device), idx.exl3(prefix + f"experts.{e}.down_proj", device))
                      for e in range(E)])


def _modelopt_part(idx: PackIndex, base: str) -> tuple:
    """(codes, scale, tensor scale) of a ModelOpt NVFP4 projection, host tensors as stored."""
    return idx.read(base + ".weight"), idx.read(base + ".weight_scale"), idx.scale(base + ".weight_scale_2")


def _nvfp4_experts(idx: PackIndex, prefix: str, E: int, device):
    from tools.nvfp4_relu2 import NVFP4Bank
    return NVFP4Bank.stack([(_modelopt_part(idx, prefix + f"experts.{e}.up_proj"),
                             _modelopt_part(idx, prefix + f"experts.{e}.down_proj")) for e in range(E)], device)


def _modelopt_linear(idx: PackIndex, base: str, fmt: str, device):
    """A ModelOpt projection on the device: NVFP4 on the W4A16 kernels, FP8 with its tensor scale."""
    from tools.nvfp4_linear import NVFP4Block
    from tools.nvfp4_relu2 import FP8Linear, NVFP4Wide
    if fmt == "fp8":
        return FP8Linear(idx.read(base + ".weight", device), idx.scale(base + ".weight_scale"))
    w, s, s2 = _modelopt_part(idx, base)
    return NVFP4Wide(NVFP4Block(w.to(device), s.to(device), s2), wide=base.endswith("down_proj"))


def _experts_any(idx: PackIndex, prefix: str, E: int, ffn: int, X: int, device, dtype):
    fmt = idx.formats.get(prefix + "experts.0.up_proj")
    if fmt == "exl3":
        return _exl3_experts(idx, prefix, E, device)
    if fmt == "nvfp4":
        return _nvfp4_experts(idx, prefix, E, device)
    return _experts(idx, prefix, E, ffn, X, device, dtype)


def _fuse_qkv(W: dict, m: str) -> None:
    """q | k | v in one weight: a bf16 concatenation, or one EXL3 group when the three share a rate (else
    the engine launches them apart)."""
    from tools.exl3_linear import EXL3Block
    ws = [W[m + f"{x}_proj.weight"] for x in "qkv"]
    if all(torch.is_tensor(w) for w in ws):
        W[m + "qkv"] = torch.cat(ws).contiguous()
    elif all(isinstance(w, EXL3Block) for w in ws) and len({w.kb for w in ws}) == 1:
        g = EXL3Block.group(ws, ["q", "k", "v"])
        W[m + "qkv"] = g
        for i, x in enumerate("qkv"):
            W[m + f"{x}_proj.weight"] = g.member(i)


def load(path: str, device="cuda", dtype=torch.bfloat16, mtp: bool = True, log=print, idx: PackIndex | None = None):
    """(idx, W, experts, timing) of the checkpoint or EXL3 pack at `path`; `mtp`: the MTP head's tensors too."""
    t0 = time.time()
    idx = idx or PackIndex.open(path)
    idx.audit()
    if idx.is_exl3:
        idx.check_mul1()
    c = idx.cfg
    mtp = mtp and idx.has_mtp
    timing = {"index_s": round(time.time() - t0, 2), "notes": list(idx.notes)}
    W: dict = {}
    t1 = time.time()
    for name, _ in expected(c, mtp=mtp):
        if ".experts." in name:
            continue
        base = name[:-len(".weight")]
        if idx.formats.get(base) == "exl3":
            W[name] = idx.exl3(base, device, out_fp32=name == "lm_head.weight")
            continue
        if idx.formats.get(base) in ("nvfp4", "fp8"):
            W[name] = _modelopt_linear(idx, base, idx.formats[base], device)
            continue
        t = idx.read(name)
        want = torch.float32 if name.endswith(FP32) else dtype
        W[name] = t.to(device=device, dtype=want).contiguous()
    idx.drop_cache()
    if c.tie_word_embeddings:
        W["lm_head.weight"] = W["embeddings.weight"]
    for li in c.attention_layers:
        _fuse_qkv(W, f"layers.{li}.mixer.")
    timing["dense_s"] = round(time.time() - t1, 2)
    t1 = time.time()
    experts: dict = {}
    for n, li in enumerate(c.moe_layers):
        p = f"layers.{li}.mixer."
        experts[li] = _experts_any(idx, p, c.moe_experts[li], c.moe_ffn[li], c.expert_in, device, dtype)
        idx.drop_cache()
        if log and n % 8 == 7:
            log(f"[nh-load] experts through layer {li}: {time.time() - t1:.1f} s")
    if mtp:
        for j, (k, ffn, E) in c.mtp_moe.items():
            p = f"mtp.layers.{j}.mixer."
            experts[("mtp", j)] = _experts_any(idx, p, E, ffn, c.expert_in, device, dtype)
        idx.drop_cache()
        m = "mtp.layers.0.mixer."
        if m + "q_proj.weight" in W:
            _fuse_qkv(W, m)
    timing["experts_s"] = round(time.time() - t1, 2)
    timing["total_s"] = round(time.time() - t0, 2)
    timing["mtp"] = mtp
    if log:
        log(f"[nh-load] {path}: {timing}")
    return idx, W, experts, timing


def gguf_of(path: str) -> str | None:
    """The GGUF a checkpoint directory serves: QSE_NH_GGUF (a profile whose directory holds the config and
    the tokenizer, the GGUF in NVIDIA's download beside it), else a GGUF in the directory itself."""
    from engine.models.nemotron_h.gguf_load import trunk_gguf
    return os.environ.get("QSE_NH_GGUF") or trunk_gguf(path)


def load_any(path: str, device="cuda", mtp: bool = True, log=print):
    """(idx, W, experts, timing) of a GGUF (gguf_load.GGUFIndex as the index) or a safetensors checkpoint."""
    from engine.models.nemotron_h.gguf_load import GGUFIndex, load_gguf
    g = gguf_of(path)
    if g is None:
        return load(path, device, mtp=mtp, log=log)
    cfg, W, experts, timing, gf = load_gguf(g, device, mtp=mtp, log=log, config_dir=path if os.path.isdir(path) else None)
    return GGUFIndex(gf, cfg), W, experts, timing
