"""An EXL3 pack of a Qwen3.5 MoE checkpoint: what is in it, how to read it (ENG-391).

exllamav3 stores the checkpoint's fused expert banks (`mlp.experts.gate_up_proj` [E, 2I, H] and
`down_proj` [E, H, I]) as one EXL3 projection per expert and matrix (`experts.E.{gate,up,down}_proj`),
the layout of Flash-Next's packs, so `PackIndex` here is engine/models/flashnext/pack.py's with this
family's config and layout:

  expert   a routed expert's projection: `(mtp.)layers.L.mlp.experts.E.{gate,up,down}_proj.*`
  exl3     any other trellis-coded projection (GDN in_proj_qkv / in_proj_z / out_proj, attention q, k,
           v, o, the shared expert, `mtp.fc`, the head)
  plain    everything else the model reads, as stored (norms, router, shared-expert gate,
           `in_proj_a/b`, conv weights, A_log, dt_bias, the embedding)
  vision   the vision tower (`model.visual.*`, EXL3 projections and plain tensors), by "visual." name

K is read from each trellis' own shape: a pack whose experts and dense projections differ in bits (or
whose tensors differ one by one) is read as it is (`k_histogram`). A routed expert the converter found
all-zero (a dead expert of the checkpoint) is stored with zero output scales: its output is exactly 0.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from engine.models.flashnext.pack import (DSIZE, PATCH_FILE, PackIndex, TensorInfo, canonical, kind_of)
from engine.models.flashnext.ple import safetensors_index
from engine.models.qwen3_5_moe.config import Qwen35MoeConfig, config_from_dict


def expected_names(cfg: Qwen35MoeConfig, mtp: bool = True) -> list[str]:
    """Every canonical tensor name the engine reads for this config; a projection by its `.trellis`."""
    out = ["embed_tokens.weight", "lm_head.trellis", "norm.weight"]

    def block(p: str, linear: bool) -> list[str]:
        names = [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
        if linear:
            a = p + "linear_attn."
            names += [a + "A_log", a + "dt_bias", a + "conv1d.weight", a + "norm.weight",
                      a + "in_proj_a.weight", a + "in_proj_b.weight", a + "in_proj_qkv.trellis",
                      a + "in_proj_z.trellis", a + "out_proj.trellis"]
        else:
            a = p + "self_attn."
            names += [a + "q_norm.weight", a + "k_norm.weight"] + [a + f"{x}_proj.trellis" for x in "qkvo"]
        m = p + "mlp."
        names += [m + "gate.weight", m + "shared_expert_gate.weight"] + [
            m + f"shared_expert.{x}_proj.trellis" for x in ("gate", "up", "down")]
        names += [m + f"experts.{e}.{x}_proj.trellis" for e in range(cfg.num_experts)
                  for x in ("gate", "up", "down")]
        return names

    for li in range(cfg.num_hidden_layers):
        out += block(f"layers.{li}.", cfg.is_linear(li))
    if mtp and cfg.mtp_num_hidden_layers:
        out += block("mtp.layers.0.", False)
        out += ["mtp.fc.trellis", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
                "mtp.norm.weight"]
    return out


class Qwen35MoePackIndex(PackIndex):

    @classmethod
    def open(cls, path: str, table: bool = False) -> "Qwen35MoePackIndex":
        path = os.path.expanduser(path)
        with open(os.path.join(path, "config.json")) as f:
            raw = json.load(f)
        gen = None
        gp = os.path.join(path, "generation_config.json")
        if os.path.isfile(gp):
            with open(gp) as f:
                gen = json.load(f)
        cfg = config_from_dict(raw, path, gen)
        ip = os.path.join(path, "model.safetensors.index.json")
        if not os.path.isfile(ip):
            raise FileNotFoundError(f"{path}: no model.safetensors.index.json")
        with open(ip) as f:
            wm = json.load(f)["weight_map"]
        by_file: dict[str, list[str]] = defaultdict(list)
        for k, fn in wm.items():
            by_file[fn].append(k)
        tensors: dict[str, TensorInfo] = {}
        files: dict[str, int] = {}
        skipped: list[str] = []
        vision: dict[str, TensorInfo] = {}
        for fn in sorted(by_file):
            fp = os.path.join(path, fn)
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"{fp}: the index names it, the pack does not have it")
            start, hdr = safetensors_index(fp)
            files[fn] = os.path.getsize(fp)
            listed = set(by_file[fn])
            for k in listed:
                if k not in hdr:
                    raise ValueError(f"{fn}: the index puts {k} here, the file's header does not have it")
            end = 0
            for k, t in hdr.items():
                b0, b1 = t["data_offsets"]
                end = max(end, b1)
                if k not in listed:
                    continue
                name = canonical(k)
                if name is None:
                    skipped.append(k)
                    vname = k[len("model."):] if k.startswith("model.") else k
                    vision[vname] = TensorInfo(vname, k, fp, t["dtype"], tuple(t["shape"]), start + b0, b1 - b0)
                    continue
                if name in tensors:
                    raise ValueError(f"{name} is in {tensors[name].file} and {fn}")
                info = TensorInfo(name, k, fp, t["dtype"], tuple(t["shape"]), start + b0, b1 - b0)
                if info.nbytes != int(np.prod(info.shape)) * DSIZE[info.dtype]:
                    raise ValueError(f"{fn}: {k} has {info.nbytes} bytes for {info.shape} {info.dtype}")
                info.kind = kind_of(name)
                tensors[name] = info
            if files[fn] != start + end:
                raise ValueError(f"{fp}: {files[fn]} bytes on disk, the header describes {start + end}"
                                 " (truncated or corrupt)")
        return cls(path, cfg, tensors, files, None, None, skipped, vision=vision)

    def audit(self, mtp: bool = True) -> list[str]:
        """Every expected tensor present with its EXL3 companions, shapes that fit the config."""
        c = self.cfg
        for name in expected_names(c, mtp):
            if name not in self.tensors:
                raise ValueError(f"{name} is missing from the pack")
            if name.endswith(".trellis"):
                base = name[: -len("trellis")]
                for sfx in ("suh", "svh"):
                    if base + sfx not in self.tensors:
                        raise ValueError(f"{base + sfx} is missing (the trellis has no {sfx})")
                t = self.tensors[name].shape
                if len(t) != 3 or t[2] % 16 or self.tensors[base + "suh"].shape != (t[0] * 16,) \
                        or self.tensors[base + "svh"].shape != (t[1] * 16,):
                    raise ValueError(f"{base}: trellis {t} does not fit its suh/svh")
        e = self.tensors.get("layers.0.mlp.experts.0.gate_proj.trellis")
        if e is not None and (e.shape[0] * 16, e.shape[1] * 16) != (c.hidden_size, c.moe_intermediate_size):
            raise ValueError(f"expert gate trellis {e.shape} does not fit hidden {c.hidden_size}, "
                             f"moe {c.moe_intermediate_size}")
        notes = [f"{sum(1 for n in self.tensors if n.endswith('.mul1'))} mul1 codebook tensors"]
        self.notes = notes
        return notes

    def decode_bytes(self, mixer_bytes: int = 1, context: int = 0) -> dict:
        return super().decode_bytes(mixer_bytes=2, context=context)


__all__ = ["Qwen35MoePackIndex", "expected_names", "PATCH_FILE"]
