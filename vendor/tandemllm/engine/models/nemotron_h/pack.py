"""The NemotronH weights on disk: a BF16 safetensors checkpoint or an EXL3 pack, one index over its shards.

`PackIndex.open(dir)` reads `config.json` (engine/models/nemotron_h/config.py) and every shard's header
(nothing else), and names each tensor canonically: the stored key without its `backbone.` prefix (an
Omni checkpoint's `language_model.` first), so `backbone.layers.3.mixer.in_proj.weight` is
`layers.3.mixer.in_proj.weight`; `lm_head.weight` and the MTP head's `mtp.layers.J.*` keep their
names. Tensors outside the language model (an Omni checkpoint's vision and sound towers and their
projectors) are counted and skipped.

`audit()` checks every tensor the layer table expects, with its shape, against the index: a missing
tensor, a shape the config does not give, a truncated shard (its size against its header) stops the
load with the name. `verify_sha256(manifest)` checks the shards against a recorded manifest (the
download's `SHA256-VERIFIED.json`, the Hub's LFS hashes).

`decode_bytes()` is what one decode token reads, from the headers: every Mamba and attention
projection, the router, the k routed experts of each MoE layer at that layer's expert size, the shared
expert and latent pair, the head, one embedding row, the fp32 SSM states and conv states read and
written, and the KV rows of `context` tokens. It is the ceiling a measured tok/s is held against.

An EXL3 pack (exllamav3 with the Puzzle/Bolt patch, ENG-E12) stores a quantised projection `X` as
`X.trellis` int16 [Kin/16, N/16, 16 K], `X.suh` / `X.svh` fp16 (or the packed sign fields `X.su` / `X.sv`)
and `X.mul1`, and the rest as fp16 (norms, conv taps, the router, the embedding) or fp32; the vision tower
it copies unquantised is skipped like the BF16 checkpoint's. A projection is BF16/F16 (`X.weight`) or EXL3
per tensor (`formats`); `audit()` checks each EXL3 block's shapes against the config, its rate against the
ones the kernels decode, and the mul1 codebook.

NVIDIA's ModelOpt checkpoints (Super VL's NVFP4, `hf_quant_config.json` MIXED_PRECISION) store a projection as
NVFP4 (`X.weight` uint8 [N, K / 2], `X.weight_scale` e4m3 [N, K / 16], `X.weight_scale_2` fp32) or FP8
(`X.weight` e4m3 [N, K], `X.weight_scale` fp32); `formats` says "nvfp4" / "fp8", `audit()` checks the codes and
scales against the config's shape. The static activation scales (`input_scale`) and the KV scales (`k_scale`,
`v_scale`) are read past: the engine keeps bf16 activations and a bf16 KV cache.

A flat `NemotronHForCausalLM` config remapped from the Omni wrapper (community packs: `backbone.*` keys, no tower)
may say `tie_word_embeddings: true` while the pack stores `lm_head` (fp16, or EXL3): the stored head is the model's
(Super VL's head is untied), so a stored head is read and the tie is dropped (`stored_head`, an audit note).

A GGUF reader (Bolt's IQ2_XXS release) plugs in beside this one with the same canonical names.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from engine.models.lfm2_moe.pack import DSIZE, DT, EXL3_SUFFIX, MUL1, RATES, TensorInfo, safetensors_header
from engine.models.nemotron_h.config import NemotronHConfig, config_from_dict

DT = {**DT, "F8_E4M3": torch.float8_e4m3fn}
DSIZE = {**DSIZE, "F8_E4M3": 1}
#: a ModelOpt projection's scales beside its `weight`
MODELOPT_SUFFIX = ("weight_scale", "weight_scale_2", "input_scale", "k_scale", "v_scale")

PREFIXES = ("language_model.backbone.", "language_model.", "backbone.")
#: an Omni checkpoint's towers: not the language model's, skipped by name
SKIP_START = ("vision_model.", "vision_projector.", "mlp1.", "radio", "sound_encoder.", "sound_projection.",
              "sound_model.", "language_model.vision")


def canonical(key: str) -> str | None:
    """The canonical name of a stored key, or None for a tensor the language model never reads."""
    if key.startswith(SKIP_START):
        return None
    for p in PREFIXES:
        if key.startswith(p):
            key = key[len(p):]
            break
    if key.startswith("language_model."):
        key = key[len("language_model."):]
    return key


def _mixer_tensors(c: NemotronHConfig, p: str, kind: str, moe: tuple | None) -> list[tuple[str, tuple]]:
    H = c.hidden_size
    m = p + "mixer."
    if kind == "mamba":
        return [(m + "in_proj.weight", (c.in_proj_dim, H)), (m + "conv1d.weight", (c.conv_dim, 1, c.conv_kernel)),
                (m + "conv1d.bias", (c.conv_dim,)), (m + "dt_bias", (c.mamba_num_heads,)),
                (m + "A_log", (c.mamba_num_heads,)), (m + "D", (c.mamba_num_heads,)),
                (m + "norm.weight", (c.intermediate,)), (m + "out_proj.weight", (H, c.intermediate))]
    if kind == "attention":
        return [(m + "q_proj.weight", (c.q_dim, H)), (m + "k_proj.weight", (c.kv_dim, H)),
                (m + "v_proj.weight", (c.kv_dim, H)), (m + "o_proj.weight", (H, c.q_dim))]
    k, ffn, E = moe
    X = c.expert_in
    out = [(m + "gate.weight", (E, H)), (m + "gate.e_score_correction_bias", (E,)),
           (m + "shared_experts.up_proj.weight", (c.shared_ffn, H)),
           (m + "shared_experts.down_proj.weight", (H, c.shared_ffn))]
    if c.moe_latent_size:
        out += [(m + "fc1_latent_proj.weight", (c.moe_latent_size, H)), (m + "fc2_latent_proj.weight", (H, c.moe_latent_size))]
    for e in range(E):
        out += [(m + f"experts.{e}.up_proj.weight", (ffn, X)), (m + f"experts.{e}.down_proj.weight", (X, ffn))]
    return out


def expected(c: NemotronHConfig, mtp: bool = True) -> list[tuple[str, tuple]]:
    """Every tensor the model reads (canonical name, shape); `mtp`: the head's too."""
    H = c.hidden_size
    out = [("embeddings.weight", (c.vocab_size, H)), ("norm_f.weight", (H,))]
    if not c.tie_word_embeddings:
        out.append(("lm_head.weight", (c.vocab_size, H)))
    for li, t in enumerate(c.layer_types):
        p = f"layers.{li}."
        out.append((p + "norm.weight", (H,)))
        out += _mixer_tensors(c, p, t, (c.moe_k[li], c.moe_ffn[li], c.moe_experts[li]) if t == "moe" else None)
    if mtp and c.mtp_layer_types:
        out += [("mtp.layers.0.enorm.weight", (H,)), ("mtp.layers.0.hnorm.weight", (H,)),
                ("mtp.layers.0.eh_proj.weight", (H, 2 * H)),
                (f"mtp.layers.{len(c.mtp_layer_types) - 1}.final_layernorm.weight", (H,))]
        for j, t in enumerate(c.mtp_layer_types):
            p = f"mtp.layers.{j}."
            out.append((p + "norm.weight", (H,)))
            out += _mixer_tensors(c, p, t, c.mtp_moe.get(j))
    return out


@dataclass
class PackIndex:
    path: str
    cfg: NemotronHConfig
    tensors: dict[str, TensorInfo]
    files: dict[str, int]                       # shard file -> size on disk
    skipped: int = 0                            # tensors outside the language model
    has_mtp: bool = False
    notes: list[str] = field(default_factory=list)
    quant: dict = field(default_factory=dict)   # the pack's quantization_config (an EXL3 pack's)
    formats: dict = field(default_factory=dict)  # projection base -> "bf16" | "f16" | "exl3" (after audit)
    stored_head: bool = False                   # the config ties the head, the pack stores one: it is read

    @property
    def is_exl3(self) -> bool:
        return str(self.quant.get("quant_method", "")).lower() == "exl3" or any(
            n.endswith(".trellis") for n in self.tensors)

    @classmethod
    def open(cls, path: str) -> "PackIndex":
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
        if os.path.isfile(ip):
            with open(ip) as f:
                wm = json.load(f)["weight_map"]
            names = sorted(set(wm.values()))
        elif os.path.isfile(os.path.join(path, "model.safetensors")):
            wm, names = None, ["model.safetensors"]
        else:
            raise FileNotFoundError(f"{path}: no model.safetensors.index.json and no model.safetensors")
        tensors: dict[str, TensorInfo] = {}
        files: dict[str, int] = {}
        skipped = 0
        for fn in names:
            fp = os.path.join(path, fn)
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"{fp}: the index names it, the checkpoint does not have it")
            start, hdr = safetensors_header(fp)
            files[fn] = os.path.getsize(fp)
            listed = {k for k, v in wm.items() if v == fn} if wm is not None else set(hdr)
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
                    skipped += 1
                    continue
                if name in tensors:
                    raise ValueError(f"{name} is in {tensors[name].file} and {fn}")
                info = TensorInfo(name, k, fp, t["dtype"], tuple(t["shape"]), start + b0, b1 - b0)
                if info.nbytes != int(np.prod(info.shape)) * DSIZE[info.dtype]:
                    raise ValueError(f"{fn}: {k} has {info.nbytes} bytes for {info.shape} {info.dtype}")
                info.kind = "expert" if ".experts." in name else (
                    "exl3" if name.rsplit(".", 1)[-1] in EXL3_SUFFIX else
                    "scale" if name.rsplit(".", 1)[-1] in MODELOPT_SUFFIX else "plain")
                tensors[name] = info
            if files[fn] != start + end:
                raise ValueError(f"{fp}: {files[fn]} bytes on disk, the header describes {start + end}"
                                 " (truncated or corrupt)")
        quant = raw.get("quantization_config") or (raw.get("llm_config") or {}).get("quantization_config") or {}
        hq = os.path.join(path, "hf_quant_config.json")
        if os.path.isfile(hq):
            with open(hq) as f:
                q = json.load(f).get("quantization") or {}
            quant = {"quant_method": "modelopt", "quant_algo": q.get("quant_algo"),
                     "kv_cache_quant_algo": q.get("kv_cache_quant_algo")}
        idx = cls(path, cfg, tensors, files, skipped, quant=quant)
        idx.has_mtp = "mtp.layers.0.eh_proj.weight" in tensors or "mtp.layers.0.eh_proj.trellis" in tensors
        return idx

    def __post_init__(self):
        if self.cfg.tie_word_embeddings and ("lm_head.weight" in self.tensors or "lm_head.trellis" in self.tensors):
            # a flat config remapped from the Omni wrapper says tied, but the pack stores the head (Super VL's is
            # its own, untied in the original): the stored head is read, as the converter's own loader does
            self.cfg.tie_word_embeddings = False
            self.stored_head = True

    # --------------------------------------------------------------------------- checks
    def audit(self) -> list[str]:
        """Every tensor the layer table expects, with its shape, a projection as BF16/F16 or as a whole EXL3
        block; the MTP head when the checkpoint has one (whole, or not at all). Raises naming the first
        problem; returns notes."""
        c = self.cfg
        want = expected(c, mtp=self.has_mtp)
        formats, read = {}, set()
        for name, shape in want:
            t = self.tensors.get(name)
            base = name[:-len(".weight")] if name.endswith(".weight") else None
            if t is None and base is not None and len(shape) == 2 and base + ".trellis" in self.tensors:
                read |= self._audit_exl3(base, *shape)
                formats[base] = "exl3"
                continue
            if t is None:
                raise ValueError(f"{name} is missing from the checkpoint")
            if base is not None and len(shape) == 2 and t.dtype in ("U8", "F8_E4M3"):
                read |= self._audit_modelopt(base, *shape)
                formats[base] = "nvfp4" if t.dtype == "U8" else "fp8"
                continue
            if tuple(t.shape) != tuple(shape):
                raise ValueError(f"{name} {t.shape}: the config wants {shape}")
            if t.dtype not in ("BF16", "F32", "F16"):
                raise ValueError(f"{name}: dtype {t.dtype}; this reader takes BF16, F16 and F32")
            if base is not None and len(shape) == 2:
                formats[base] = t.dtype.lower()
            read.add(name)
        # a KV scale (k_proj.k_scale, v_proj.v_scale): the KV cache stays bf16
        read |= {n for n in self.tensors if n.endswith((".k_scale", ".v_scale"))}
        extra = sorted(set(self.tensors) - read)
        if extra and not (not self.has_mtp and all(e.startswith("mtp.") for e in extra)):
            raise ValueError(f"{len(extra)} tensor(s) the layer table does not read, first {extra[0]}")
        self.formats = formats
        dt = Counter(t.dtype for t in self.tensors.values() if t.kind != "exl3" and not t.name.endswith(".trellis"))
        self.notes = [f"tensors {len(self.tensors)} ({dict(dt)}), mtp {'yes' if self.has_mtp else 'no'}"
                      + (f", {self.skipped} tower tensors skipped" if self.skipped else "")]
        if self.stored_head:
            self.notes.append("head: the config ties it to the embedding, the pack stores lm_head; the stored head is read")
        if self.quant.get("quant_method") == "modelopt":
            self.notes.append(f"ModelOpt {self.quant.get('quant_algo')}: projections {dict(Counter(formats.values()))}"
                              f" (activations bf16, KV bf16; checkpoint KV {self.quant.get('kv_cache_quant_algo')})")
        if self.is_exl3:
            self.notes.append(f"projections {dict(Counter(formats.values()))}, EXL3 bits "
                              f"{dict(sorted(Counter(self.tensors[b + '.trellis'].k for b, f in formats.items() if f == 'exl3').items()))}")
        return self.notes

    def _audit_exl3(self, base: str, N: int, Kin: int) -> set:
        """One EXL3 block: trellis [Kin/16, N/16, 16 K] at a rate the kernels decode, the mul1 codebook, the
        channel factors (or their packed signs) at their lengths. Returns the tensors it reads."""
        t = self.tensors[base + ".trellis"]
        if base + ".mcg" in self.tensors or base + ".mul1" not in self.tensors:
            raise ValueError(f"{base}: not the mul1 codebook (mcg or 3inst); the engine decodes mul1 only")
        if len(t.shape) != 3 or (t.shape[0] * 16, t.shape[1] * 16) != (Kin, N):
            raise ValueError(f"{base}.trellis {t.shape} does not fit {(N, Kin)}")
        if t.k not in RATES:
            raise ValueError(f"{base}: {t.k} bits a weight; the kernels take 2..8, 1.5, 2.5 and 3.5")
        read = {base + ".trellis", base + ".mul1"}
        for s, alt, n in (("suh", "su", Kin), ("svh", "sv", N)):
            if base + "." + s in self.tensors:
                if self.tensors[base + "." + s].shape != (n,):
                    raise ValueError(f"{base}.{s} {self.tensors[base + '.' + s].shape} != ({n},)")
                read.add(base + "." + s)
            elif base + "." + alt in self.tensors:
                read.add(base + "." + alt)
            else:
                raise ValueError(f"{base}: no {s} and no {alt}")
        return read

    def _audit_modelopt(self, base: str, N: int, K: int) -> set:
        """One ModelOpt projection: NVFP4 codes [N, K / 2] with e4m3 scales [N, K / 16] and a tensor scale, or FP8
        codes [N, K] with a tensor scale. Returns the tensors it reads."""
        w = self.tensors[base + ".weight"]
        s = self.tensors.get(base + ".weight_scale")
        if s is None:
            raise ValueError(f"{base}: {w.dtype} codes without weight_scale")
        read = {base + ".weight", base + ".weight_scale"}
        if w.dtype == "U8":
            s2 = self.tensors.get(base + ".weight_scale_2")
            if K % 128:
                raise ValueError(f"{base}: NVFP4 needs K a multiple of 128, not {K}")
            if tuple(w.shape) != (N, K // 2) or tuple(s.shape) != (N, K // 16) or s.dtype != "F8_E4M3":
                raise ValueError(f"{base}: NVFP4 {w.shape} / scale {s.shape} {s.dtype} does not fit {(N, K)}")
            if s2 is None or int(np.prod(s2.shape)) != 1:
                raise ValueError(f"{base}: NVFP4 without one weight_scale_2")
            read.add(base + ".weight_scale_2")
        else:
            if tuple(w.shape) != (N, K) or int(np.prod(s.shape)) != 1:
                raise ValueError(f"{base}: FP8 {w.shape} / scale {s.shape} does not fit {(N, K)} with one tensor scale")
        if base + ".input_scale" in self.tensors:
            read.add(base + ".input_scale")
        return read

    def scale(self, name: str) -> float:
        """A ModelOpt tensor scale as a float."""
        return float(self.read(name).float().reshape(-1)[0])

    def check_mul1(self) -> None:
        """Every `.mul1` holds 0x83DCD12D (the codebook the kernels decode)."""
        for n in self.tensors:
            if n.endswith(".mul1"):
                v = int(self.read(n).view(-1)[0].item()) & 0xFFFFFFFF
                if v != MUL1:
                    raise ValueError(f"{n} = {v:#x}: only the mul1 codebook {MUL1:#x} is decoded")

    def verify_sha256(self, manifest: dict) -> dict:
        """Every shard against a manifest {file: {"sha256": hex}} (a missing entry is reported, a wrong
        hash raises with the file's name)."""
        out = {}
        for fn in self.files:
            want = (manifest.get(fn) or {}).get("sha256")
            if not want:
                out[fn] = "no hash recorded"
                continue
            h = hashlib.sha256()
            with open(os.path.join(self.path, fn), "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            if h.hexdigest() != want:
                raise ValueError(f"{fn}: sha256 {h.hexdigest()} != recorded {want}")
            out[fn] = "ok"
        return out

    # --------------------------------------------------------------------------- summaries
    def nbytes(self, name: str) -> int:
        """Bytes of a tensor as stored; for a projection's `X.weight` that the pack holds as EXL3, its
        trellis and channel factors."""
        t = self.tensors.get(name)
        if t is not None:
            sc = self.tensors.get(name[:-len("weight")] + "weight_scale") if name.endswith(".weight") else None
            return t.nbytes + (sc.nbytes if sc is not None else 0)
        base = name[:-len(".weight")]
        return sum(self.tensors[base + "." + s].nbytes for s in ("trellis", "suh", "svh", "su", "sv")
                   if base + "." + s in self.tensors)

    def decode_bytes(self, context: int = 0) -> dict:
        """What one decode token reads (bytes), by part; `context`: the KV rows attention reads."""
        c = self.cfg
        parts: dict[str, float] = defaultdict(float)
        for li, t in enumerate(c.layer_types):
            m = f"layers.{li}.mixer."
            if t == "mamba":
                parts["mamba_proj"] += self.nbytes(m + "in_proj.weight") + self.nbytes(m + "out_proj.weight")
                parts["mamba_small"] += sum(self.nbytes(m + s) for s in ("conv1d.weight", "conv1d.bias", "dt_bias",
                                                                          "A_log", "D", "norm.weight"))
            elif t == "attention":
                parts["attn_proj"] += sum(self.nbytes(m + f"{x}_proj.weight") for x in "qkvo")
            else:
                E = c.moe_experts[li]
                eb = sum(self.nbytes(m + f"experts.{e}.up_proj.weight") + self.nbytes(m + f"experts.{e}.down_proj.weight")
                         for e in range(E)) / E
                parts["experts"] += c.moe_k[li] * eb
                parts["shared"] += self.nbytes(m + "shared_experts.up_proj.weight") + self.nbytes(
                    m + "shared_experts.down_proj.weight")
                parts["router"] += self.nbytes(m + "gate.weight") + self.nbytes(m + "gate.e_score_correction_bias")
                if c.moe_latent_size:
                    parts["latent"] += self.nbytes(m + "fc1_latent_proj.weight") + self.nbytes(m + "fc2_latent_proj.weight")
                del E
            parts["norms"] += self.nbytes(f"layers.{li}.norm.weight")
        own = "lm_head.weight" in self.tensors or "lm_head.trellis" in self.tensors
        parts["head"] += self.nbytes("lm_head.weight" if own else "embeddings.weight")
        e = self.tensors["embeddings.weight"]
        parts["embed_row"] += e.shape[1] * DSIZE[e.dtype]
        parts["ssm_state_rw"] += 2 * c.ssm_state_bytes()
        parts["conv_state_rw"] += 2 * c.conv_state_bytes()
        if context:
            parts["kv_read"] += context * c.kv_bytes_per_token()
        parts = {k: int(v) for k, v in parts.items()}
        return {"parts": parts, "total": sum(parts.values()), "context": context}

    def resident_bytes(self, mtp: bool = True) -> int:
        """Bytes the language model's tensors hold as stored."""
        return sum(t.nbytes for n, t in self.tensors.items() if mtp or not n.startswith("mtp."))

    # --------------------------------------------------------------------------- reading
    def drop_cache(self) -> None:
        """Drop the shards' page cache (posix_fadvise DONTNEED): on a unified-memory board the cache
        of a 66 GB checkpoint beside its device copy exceeds the memory, and the GPU's allocations do
        not wait for the kernel to reclaim it (tools/drop_page_cache.py)."""
        if not hasattr(os, "posix_fadvise"):
            return
        for fn in self.files:
            try:
                fd = os.open(os.path.join(self.path, fn), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

    def read(self, name: str, device=None) -> torch.Tensor:
        """One tensor as stored (a CPU copy, or on `device`)."""
        t = self.tensors[name]
        arr = np.fromfile(t.file, dtype=np.uint8, count=t.nbytes, offset=t.offset)
        out = torch.from_numpy(arr).view(DT[t.dtype]).reshape(t.shape)
        return out.to(device) if device is not None else out


    def exl3(self, base: str, device=None, out_fp32: bool = False):
        """The EXL3 projection `base` as an EXL3Block (signs unpacked where the pack packed them)."""
        from tools.exl3_linear import EXL3Block, unpack_signs

        def part(s, alt):
            if base + "." + s in self.tensors:
                return self.read(base + "." + s, device)
            return unpack_signs(self.read(base + "." + alt)).to(device)
        return EXL3Block(self.read(base + ".trellis", device), part("suh", "su"), part("svh", "sv"), out_fp32=out_fp32)


def summary(idx: PackIndex) -> dict:
    """The audit record: counts, dtypes, the layer table, resident and decode bytes."""
    c = idx.cfg
    return {
        "path": idx.path, "tensors": len(idx.tensors), "skipped": idx.skipped, "files": len(idx.files),
        "dtypes": dict(Counter(t.dtype for t in idx.tensors.values())), "mtp": idx.has_mtp,
        "layers": dict(Counter(c.layer_types)), "resident": idx.resident_bytes(),
        "resident_no_mtp": idx.resident_bytes(mtp=False),
        "decode_bytes": idx.decode_bytes(), "decode_bytes_8k": idx.decode_bytes(context=8192),
        "formats": dict(Counter(idx.formats.values())), "quant": {k: v for k, v in idx.quant.items() if k != "tensor_storage"},
        "notes": idx.notes,
    }
