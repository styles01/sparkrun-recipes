"""Load a Qwen3.5 MoE EXL3 pack onto the board, in the forms the engine multiplies by (ENG-391).

  W[name]      canonical names: EXL3Blocks (members of a group are views of it) and plain tensors
  W[fused]     what the engine launches once instead of twice:
                 layers.L.linear_attn.qkvz        EXL3 group in_proj_qkv | in_proj_z
                 layers.L.linear_attn.ab          fp16 [2 Hv, H]  in_proj_a ; in_proj_b
                 layers.L.self_attn.qkv           EXL3 group q | k | v
                 layers.L.mlp.router              fp16 [E + 1, H]  gate ; shared_expert_gate
  banks[L]     the routed experts of layer L and its shared expert as expert E (tools.fn_kernels.MoEBank);
               banks[-1] the MTP layer's

The experts are read a shard range at a time into 16-byte aligned images (Flash-Next's loader,
engine/models/flashnext/load.py `load_banks`), the rest tensor by tensor.
"""

from __future__ import annotations

import time

import torch

from engine.models.flashnext.load import DROP_SHARD_CACHE, apply_tiles, drop_cache, load_banks
from engine.models.flashnext.ref import MTP
from engine.models.qwen3_5_moe.pack import Qwen35MoePackIndex


class SplitGroup:
    """Projections that read the same activation at different bit widths (a pack with per-tensor K):
    launched one by one, their outputs side by side, as one EXL3 group's would be."""

    def __init__(self, blocks: list, names: list[str]):
        self.blocks, self.names = blocks, names
        self.sizes = [b.N for b in blocks]
        self.N = sum(self.sizes)
        self.K = blocks[0].K

    def member(self, i: int):
        return self.blocks[i]

    def dequant(self):
        return torch.cat([b.dequant() for b in self.blocks], dim=0)

    def bf16_cached(self):
        return torch.cat([b.bf16_cached() for b in self.blocks], dim=0)


def group(blocks: list, names: list[str]):
    """One EXL3 group when the members share K (one launch), else a `SplitGroup`."""
    from tools.exl3_linear import EXL3Block
    if len({b.kb for b in blocks}) == 1:
        return EXL3Block.group(blocks, names)
    return SplitGroup(blocks, names)


def load(path: str, device="cuda", log=print, mtp: bool = True, layers=None):
    """(idx, W, banks, timing) for the pack at `path`; with `mtp` its MTP layer too (weights under
    `mtp.*`, experts as banks[-1]). `layers`: a subset of the trunk (a debug model)."""
    import os

    t0 = time.time()
    tiles = apply_tiles()
    idx = Qwen35MoePackIndex.open(path)
    mtp = bool(mtp and idx.cfg.mtp_num_hidden_layers and "mtp.fc.trellis" in idx.tensors)
    idx.audit(mtp=mtp)
    c = idx.cfg
    W: dict = {}
    timing = {"index_s": round(time.time() - t0, 2), "tiles": tiles, "mtp": mtp}
    lay = range(c.num_hidden_layers) if layers is None else layers
    t1 = time.time()
    for n, t in idx.tensors.items():
        if (n.startswith("mtp.") and not mtp) or t.kind == "expert" or n.endswith(".mul1"):
            continue
        li = int(n.split(".")[1]) if n.startswith("layers.") else None
        if li is not None and li not in lay:
            continue
        if t.kind == "exl3":
            if not n.endswith(".trellis"):
                continue
            base = n[: -len(".trellis")]
            if ".shared_expert." in base:
                continue                                  # in the bank
            W[base + ".weight"] = idx.exl3(base, device, out_fp32=base == "lm_head")
        else:
            W[n] = idx.read(n, device)
    timing["dense_s"] = round(time.time() - t1, 2)
    for li in list(lay) + ([MTP] if mtp else []):
        p = "mtp.layers.0." if li == MTP else f"layers.{li}."
        if li != MTP and c.is_linear(li):
            a = p + "linear_attn."
            W[a + "qkvz"] = group([W[a + "in_proj_qkv.weight"], W[a + "in_proj_z.weight"]], ["qkv", "z"])
            W[a + "ab"] = torch.cat([W[a + "in_proj_a.weight"], W[a + "in_proj_b.weight"]]).to(torch.float16).contiguous()
        else:
            a = p + "self_attn."
            W[a + "qkv"] = group([W[a + f"{x}_proj.weight"] for x in ("q", "k", "v")], ["q", "k", "v"])
        W[p + "mlp.router"] = torch.cat([W[p + "mlp.gate.weight"],
                                         W[p + "mlp.shared_expert_gate.weight"]]).to(torch.float16).contiguous()
    t1 = time.time()
    banks = {}
    for li in lay:
        banks[li] = load_banks(idx, li, device)
        if log and li % 8 == 7:
            log(f"[q35moe-load] experts through layer {li}: {time.time() - t1:.1f} s")
    if mtp:
        banks[MTP] = load_banks(idx, 0, device, prefix="mtp.")
    timing["experts_s"] = round(time.time() - t1, 2)
    if DROP_SHARD_CACHE:
        timing["shard_cache_dropped_gb"] = round(drop_cache([os.path.join(idx.path, f) for f in idx.files]) / 1e9, 1)
    timing["total_s"] = round(time.time() - t0, 2)
    if log:
        log(f"[q35moe-load] {path}: {timing}")
    return idx, W, banks, timing
