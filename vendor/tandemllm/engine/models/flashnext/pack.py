"""turboderp's EXL3 packs of Qwen3.8-Flash-Next: what is in them, what a token reads, how to read it (ENG-219).

A pack (`turboderp/Qwen3.8-Flash-Next-exl3`, branches 2.05bpw_h4_ng4 .. 6.05bpw_h6_ng6) is a set of
safetensors shards with an index, a separate n-gram table (`ngram_embedding.safetensors`, read by
`ple.EXL3NGramTable`), and in four of the five branches a small `mtp_hyper_connection_mixer_patch`
with the MTP layer's final mixer. `PackIndex.open(dir)` reads every header (nothing else) and puts
every tensor in exactly one kind:

  expert   a routed expert's projection: `(mtp.)layers.L.mlp.experts.E.{gate,up,down}_proj.*`
  exl3     any other trellis-coded projection (GDN, attention, indexer, shared expert, MTP fc_*,
           the head): `X.trellis` int16 [Kin/16, N/16, 16 K], `X.suh` fp16 [Kin], `X.svh` fp16 [N],
           `X.mul1` (the codebook's multiplier, checked to be 0x83DCD12D)
  mixer    a hyper-connection mixer's projections (`input_mix_weight_down/up`,
           `block_inject_weight`): fp16 in the pack, int8 with a per-row fp32 scale here (a project
           decision of 2026-10-11: lossy accepted; QWEN38_FN_MIX=fp16 keeps the pack's values)
  plain    everything else the model reads, as stored (norms, router, `in_proj_a/b`, PLE
           projections, conv weights, A_log, the embedding)
  table    the n-gram table (its own file)
  vision   the vision tower, by its "visual." name (engine/models/flashnext/vision.py loads it): in the
           shards, or in a file of its own outside the index (`vision_k6.safetensors` in 4.05)

K is read from each trellis' own shape, so a pack that mixes K inside a layer is described as it is
(`k_histogram`). Anything the config's layout expects and no file holds stops `audit()` with its
name; so does a file whose size is not what its header says, and a sha256 that differs from the
Hub's list (`verify`).

`decode_bytes()` is what one plain decode token reads from memory, from the headers: the 10 routed
experts of every layer at that layer's mean expert size, every dense projection, the mixers at the
stored or the served width, the plain tensors a token touches, one embedding row, 16 n-gram rows
and the GDN state read and written. It is the ceiling the measured tok/s is held against.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from engine.models.flashnext.config import FlashNextConfig, config_from_dict
from engine.models.flashnext.ple import EXL3NGramTable, safetensors_index
from engine.settings import SETTINGS as _S

PREFIX = "model.language_model."
TABLE_FILE = "ngram_embedding.safetensors"
PATCH_FILE = "mtp_hyper_connection_mixer_patch.safetensors"
MUL1 = 0x83DCD12D

EXPERT_RE = re.compile(r"^((?:mtp\.)?layers\.(\d+)\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\."
                       r"(trellis|suh|svh|mul1)$")
EXL3_SUFFIX = ("trellis", "suh", "svh", "mul1")
MIXER_RE = re.compile(r"hyper_connection(?:_mixer)?\.(input_mix_weight_down|input_mix_weight_up|"
                      r"block_inject_weight)\.weight$")
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I16": torch.int16,
      "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8, "I8": torch.int8}
DSIZE = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "I64": 8, "U8": 1, "I8": 1}


def canonical(key: str) -> str | None:
    """The engine's name of a pack tensor: the language-model prefix removed; None for vision."""
    if key.startswith("model.visual.") or key.startswith("visual."):
        return None
    if key.startswith(PREFIX):
        return key[len(PREFIX):]
    return key


@dataclass
class TensorInfo:
    name: str               # canonical
    key: str                # as stored
    file: str
    dtype: str
    shape: tuple
    offset: int             # absolute byte offset in the file
    nbytes: int
    kind: str = "plain"

    @property
    def k(self) -> int | None:
        """bits a weight of a trellis tensor."""
        return self.shape[2] // 16 if self.name.endswith(".trellis") else None


def kind_of(name: str) -> str:
    if EXPERT_RE.match(name):
        return "expert"
    if name.rsplit(".", 1)[-1] in EXL3_SUFFIX:
        return "exl3"
    if MIXER_RE.search(name):
        return "mixer"
    return "plain"


# ------------------------------------------------------------------------------ the layout
def expected_names(cfg: FlashNextConfig, mtp: bool = True) -> list[str]:
    """Every canonical tensor name the engine reads for this config (ENG-133's Layout): the trunk,
    the head, the final mixer, and the MTP layer when `mtp`. The pack's suffixes: a projection
    stored as EXL3 is listed by its `.trellis` (its suh/svh/mul1 go with it, `audit`)."""
    out = ["embed_tokens.weight", "lm_head.trellis"]
    hc = ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight")

    def mixers(p: str, inject: bool = True) -> list[str]:
        return [f"{p}.{n}" for n in hc] + ([f"{p}.block_inject_weight.weight"] if inject else [])

    def block(p: str, linear: bool) -> list[str]:
        names = mixers(p + "attn_hyper_connection") + mixers(p + "mlp_hyper_connection")
        if linear:
            a = p + "linear_attn."
            names += [a + "A_log", a + "dt_bias", a + "conv1d.weight", a + "norm.weight",
                      a + "in_proj_a.weight", a + "in_proj_b.weight", a + "in_proj_qkv.trellis",
                      a + "in_proj_z.trellis", a + "out_proj.trellis"]
        else:
            a = p + "self_attn."
            names += [a + "q_norm.weight", a + "k_norm.weight"] + [
                a + f"{x}_proj.trellis" for x in ("q", "k", "v", "o")]
            if cfg.has_indexer:
                names += [a + "indexer.index_qk_proj.trellis", a + "indexer.q_layernorm.weight",
                          a + "indexer.k_layernorm.weight"]
        m = p + "mlp."
        names += [m + "gate.weight", m + "shared_expert_gate.weight"] + [
            m + f"shared_expert.{x}_proj.trellis" for x in ("gate", "up", "down")]
        names += [m + f"experts.{e}.{x}_proj.trellis" for e in range(cfg.num_experts)
                  for x in ("gate", "up", "down")]
        return names

    for li in range(cfg.num_hidden_layers):
        out += block(f"layers.{li}.", cfg.is_linear(li))
        if li in cfg.ple_layers:
            p = f"layers.{li}.ple."
            out += [p + n for n in ("key_proj.weight", "value_proj.weight", "norm_key.weight",
                                    "norm_query.weight", "norm_conv.weight", "conv1d.weight")]
    out += mixers("hyper_connection_mixer", inject=False)
    if mtp and cfg.mtp_num_hidden_layers:
        out += block("mtp.layers.0.", False)
        out += ["mtp.fc_embedding.trellis", "mtp.fc_hidden.trellis", "mtp.pre_fc_norm_embedding.weight",
                "mtp.pre_fc_norm_hidden.weight"] + mixers("mtp.hyper_connection_mixer", inject=False)
    return out


# ------------------------------------------------------------------------------ the index
@dataclass
class PackIndex:
    path: str
    cfg: FlashNextConfig
    tensors: dict[str, TensorInfo]
    files: dict[str, int]                   # shard file -> size on disk
    table: EXL3NGramTable | None
    table_prefix: str | None
    skipped: list[str] = field(default_factory=list)      # vision keys
    notes: list[str] = field(default_factory=list)
    vision: dict[str, TensorInfo] = field(default_factory=dict)   # the vision tower's, by "visual.<name>"

    @classmethod
    def open(cls, path: str, table: bool = True) -> "PackIndex":
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
        names = sorted(by_file)
        if os.path.isfile(os.path.join(path, PATCH_FILE)):
            names.append(PATCH_FILE)
        names += sorted(n for n in os.listdir(path) if n.startswith("vision_") and n.endswith(".safetensors")
                        and n not in by_file)
        for fn in names:
            fp = os.path.join(path, fn)
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"{fp}: the index names it, the pack does not have it")
            start, hdr = safetensors_index(fp)
            files[fn] = os.path.getsize(fp)
            listed = set(by_file.get(fn, hdr.keys()))
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
        tab, prefix = None, None
        tp = os.path.join(path, TABLE_FILE)
        if table and cfg.ple_layer_ids:
            if not os.path.isfile(tp):
                raise FileNotFoundError(f"{tp}: the n-gram table is missing")
            prefixes = EXL3NGramTable.find(tp)
            if len(prefixes) != len(cfg.ple_layer_ids):
                raise ValueError(f"{tp}: {len(prefixes)} n-gram tables for {len(cfg.ple_layer_ids)} PLE layers")
            prefix = prefixes[0]
            tab = EXL3NGramTable(tp, prefix)
            files[TABLE_FILE] = os.path.getsize(tp)
        return cls(path, cfg, tensors, files, tab, prefix, skipped, vision=vision)

    # --------------------------------------------------------------------------- checks
    def audit(self, mtp: bool = True) -> list[str]:
        """Every expected tensor is present with its EXL3 companions and a 2-D/3-D shape that fits
        the config; the codebook is mul1. Returns notes; raises on the first missing tensor."""
        c = self.cfg
        for name in expected_names(c, mtp):
            if name in self.tensors:
                pass
            elif name.startswith("mtp.hyper_connection_mixer."):
                raise ValueError(f"{name} is missing: the pack has no {PATCH_FILE} and no shard "
                                 "holds the MTP layer's final mixer")
            else:
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
        notes = []
        mul = [n for n in self.tensors if n.endswith(".mul1")]
        if mul:
            notes.append(f"{len(mul)} mul1 codebook tensors")
        e = self.tensors.get("layers.0.mlp.experts.0.gate_proj.trellis")
        if e is not None and (e.shape[0] * 16, e.shape[1] * 16) != (c.hidden_size, c.moe_intermediate_size):
            raise ValueError(f"expert gate trellis {e.shape} does not fit hidden {c.hidden_size}, "
                             f"moe {c.moe_intermediate_size}")
        if self.table is not None and self.table.dim * c.ngram_heads != c.ple_embed_dim:
            raise ValueError(f"n-gram rows of {self.table.dim} x {c.ngram_heads} heads != ple_embed_dim")
        self.notes = notes
        return notes

    def check_mul1(self) -> None:
        """Every `.mul1` holds 0x83DCD12D (the codebook the kernels decode)."""
        for n, t in self.tensors.items():
            if n.endswith(".mul1"):
                v = int(self.read(n).view(-1)[0].item()) & 0xFFFFFFFF
                if v != MUL1:
                    raise ValueError(f"{n} = {v:#x}: only the mul1 codebook {MUL1:#x} is decoded")

    def verify(self, manifest: dict) -> list[str]:
        """sha256 of every file against the Hub's LFS list ({file: {"sha256": ..}}); raises naming
        the first file that differs."""
        ok = []
        for fn in sorted(self.files):
            want = (manifest.get(fn) or {}).get("sha256")
            if not want:
                continue
            h = hashlib.sha256()
            with open(os.path.join(self.path, fn), "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    h.update(b)
            if h.hexdigest() != want:
                raise ValueError(f"{fn}: sha256 {h.hexdigest()[:16]}.. differs from the Hub's {want[:16]}..")
            ok.append(fn)
        return ok

    # --------------------------------------------------------------------------- summaries
    def k_histogram(self) -> dict[str, Counter]:
        """{group: Counter(K -> tensors)} for the trellis tensors: experts by projection, dense
        by family, the head, the MTP layer."""
        out: dict[str, Counter] = defaultdict(Counter)
        for n, t in self.tensors.items():
            if not n.endswith(".trellis"):
                continue
            if n.startswith("mtp."):
                g = "mtp." + ("experts" if t.kind == "expert" else "dense")
            elif t.kind == "expert":
                g = "experts." + EXPERT_RE.match(n).group(4)
            elif n.startswith("lm_head"):
                g = "head"
            else:
                g = ".".join(p for p in n.split(".")[2:-1] if not p.isdigit())
            out[g][t.k] += 1
        return dict(out)

    def _exl3_bytes(self, base: str) -> int:
        return sum(self.tensors[base + s].nbytes for s in ("trellis", "suh", "svh") if base + s in self.tensors)

    def expert_bytes(self, layer: int) -> list[int]:
        """Bytes of each routed expert of a trunk layer (gate + up + down, trellis + suh + svh)."""
        c = self.cfg
        return [sum(self._exl3_bytes(f"layers.{layer}.mlp.experts.{e}.{x}_proj.")
                    for x in ("gate", "up", "down")) for e in range(c.num_experts)]

    def decode_bytes(self, mixer_bytes: int = 1, context: int = 0) -> dict:
        """What one plain decode token reads (bytes), by part. `mixer_bytes`: 1 for int8 mixers
        (plus a 4-byte scale a row), 2 for the pack's fp16. `context`: the KV rows attention reads
        (BF16 K and V of 12 layers, plus the indexer keys), 0 to leave them out."""
        c = self.cfg
        parts: dict[str, float] = defaultdict(float)
        per_layer_mean = []
        for li in range(c.num_hidden_layers):
            eb = self.expert_bytes(li)
            per_layer_mean.append(sum(eb) / len(eb))
            parts["experts"] += c.num_experts_per_tok * per_layer_mean[-1]
        for n, t in self.tensors.items():
            if n.startswith("mtp.") or t.kind == "expert":
                continue
            if t.kind == "exl3" and n.endswith(".mul1"):
                continue
            if t.kind == "exl3":
                if n.startswith("lm_head"):
                    parts["head"] += t.nbytes
                elif ".shared_expert." in n:
                    parts["shared_expert"] += t.nbytes
                elif ".linear_attn." in n:
                    parts["gdn_proj"] += t.nbytes
                else:
                    parts["attn_proj"] += t.nbytes
            elif t.kind == "mixer":
                rows, cols = t.shape[0], t.shape[-1]
                # int8: an fp32 scale a group of gcd(MIX_GROUP, K) weights of a row (`mixer_weight`)
                groups = cols // math.gcd(MIX_GROUP, cols)
                parts["mixers"] += (t.nbytes // 2 * mixer_bytes + (4 * rows * groups if mixer_bytes == 1 else 0))
            elif n == "embed_tokens.weight":
                parts["embed_row"] += t.shape[1] * DSIZE[t.dtype]
            elif n.endswith("mlp.gate.weight") or n.endswith("shared_expert_gate.weight"):
                parts["router"] += t.nbytes
            elif ".ple." in n:
                parts["ple_proj"] += t.nbytes
            else:
                parts["plain_small"] += t.nbytes
        if self.table is not None:
            parts["ple_rows"] += c.ngram_heads * self.table.row_bytes
        n_lin = len(c.linear_layers)
        state = n_lin * c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4
        parts["gdn_state_rw"] += 2 * state
        if context:
            kv = len(c.attention_layers) * 2 * c.num_key_value_heads * c.head_dim * 2
            ik = len(c.attention_layers) * (c.indexer_head_dim or 0) * 2
            parts["kv_read"] += context * (kv + ik)
        parts = {k: int(v) for k, v in parts.items()}
        total = sum(parts.values())
        return {"parts": parts, "total": total, "mixer_bytes": mixer_bytes, "context": context,
                "expert_mean_per_layer": [int(x) for x in per_layer_mean]}

    def resident_bytes(self, mixer_bytes: int = 1, mtp: bool = True) -> dict:
        """What the engine holds on the device (the table excluded): by kind."""
        out: dict[str, int] = defaultdict(int)
        for n, t in self.tensors.items():
            if n.startswith("mtp.") and not mtp:
                continue
            if t.kind == "mixer":
                out["mixer"] += t.nbytes // 2 * mixer_bytes + (4 * t.shape[0] if mixer_bytes == 1 else 0)
            else:
                out[t.kind] += t.nbytes
        out = dict(out)
        out["total"] = sum(out.values())
        return out

    # --------------------------------------------------------------------------- reading
    def read(self, name: str, device=None) -> torch.Tensor:
        """One tensor as stored (a CPU copy, or on `device`); a vision tensor by its "visual." name."""
        t = self.tensors[name] if name in self.tensors else self.vision[name]
        arr = np.fromfile(t.file, dtype=np.uint8, count=t.nbytes, offset=t.offset)
        out = torch.from_numpy(arr).view(DT[t.dtype]).reshape(t.shape)
        return out.to(device) if device is not None else out

    def exl3(self, base: str, device=None, out_fp32: bool = False):
        """The EXL3 projection `base` (no trailing dot) as an EXL3Block."""
        from tools.exl3_linear import EXL3Block
        return EXL3Block(self.read(base + ".trellis", device), self.read(base + ".suh", device),
                         self.read(base + ".svh", device), out_fp32=out_fp32)


# ------------------------------------------------------------------------------ int8 mixers
#: the served mixer width: int8 with an fp32 scale a group of MIX_GROUP weights (default), int8 with
#: one scale a row (`int8rows`, as served until 2026-10-12), or the pack's fp16
MIX = _S.get("FN_MIX")
#: ENG-221: a row's largest weight is 10 to 420 times its median, so one scale a row left its small
#: weights coarse: on code (8,192 tokens) KL 0.267 against the fp16 mixers, 0.0031 with a scale a
#: group of 128 (the reference noise 0.0038); the loss sat in the down|inject weights of layers 0-11
MIX_GROUP = 128


def quant_int8_rows(w: torch.Tensor, group: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """w [N, K] (fp16) -> (q int8 [N, K], scale fp32 [N] or, with `group` G, [N, K / G]), q = round(w / s),
    s = max|w| / 127 over the row (or the group of G weights). Error bound: |w - q s| <= s / 2 for every
    element, plus the fp32 rounding of the product q s (<= 2^-22 |w|); an all-zero row (group) has
    s = 0 and is exact. Deterministic (round half to even)."""
    N, K = w.shape
    G = group or K
    wf = w.float().view(N, K // G, G)
    s = wf.abs().amax(dim=2) / 127.0
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.round(wf / safe[..., None]).clamp_(-127, 127).to(torch.int8).view(N, K)
    return q, (s.view(N) if group is None else s.contiguous())


class Int8Rows:
    """A mixer projection stored as int8 with fp32 scales (engine orientation [N, K]): one scale a row
    (`s` [N]) or one a group of `group` weights (`s` [N, K / group]). `dequant()` is the weight the
    kernels multiply by: q * s in fp32."""

    __slots__ = ("q", "s")

    def __init__(self, q: torch.Tensor, s: torch.Tensor):
        self.q, self.s = q, s

    @classmethod
    def of(cls, w: torch.Tensor, group: int | None = None) -> "Int8Rows":
        return cls(*quant_int8_rows(w, group))

    @property
    def shape(self):
        return tuple(self.q.shape)

    @property
    def group(self) -> int | None:
        return None if self.s.dim() == 1 else self.q.shape[1] // self.s.shape[1]

    @property
    def nbytes(self) -> int:
        return self.q.numel() + self.s.numel() * 4

    def to(self, device) -> "Int8Rows":
        return Int8Rows(self.q.to(device), self.s.to(device))

    def dequant(self, dtype=torch.float32) -> torch.Tensor:
        if self.s.dim() == 1:
            return (self.q.float() * self.s[:, None]).to(dtype)
        N, K = self.q.shape
        return (self.q.float().view(N, self.s.shape[1], -1) * self.s[..., None]).view(N, K).to(dtype)

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., K] -> [..., N] in x's dtype: fp32 products of x and the dequantised weight."""
        return (x.float() @ self.dequant().T).to(x.dtype)


def mixer_weight(w: torch.Tensor, mode: str | None = None):
    """The served form of a pack mixer weight: int8 with a scale a group of gcd(MIX_GROUP, K) weights (the
    down|inject weights' K is 10,240, groups of 128; the up weights' 320, groups of 64; QWEN38_FN_MIX=int8,
    the default), int8 with one scale a row (`int8rows`), or the fp16 tensor itself (`fp16`)."""
    mode = mode or MIX
    if mode == "fp16":
        return w
    if mode == "int8rows":
        return Int8Rows.of(w)
    if mode != "int8":
        raise ValueError(f"QWEN38_FN_MIX={mode!r}: int8, int8rows or fp16")
    return Int8Rows.of(w, math.gcd(MIX_GROUP, w.shape[1]))


# ------------------------------------------------------------------------------ shared tables
def table_identity(path: str, manifest: dict | None = None) -> str:
    """A table's identity for sharing one copy between packs: the Hub's sha256 when the manifest
    has it, else the sha256 of its header and first and last MiB (cheap, file-content based)."""
    if manifest and (manifest.get(TABLE_FILE) or {}).get("sha256"):
        return manifest[TABLE_FILE]["sha256"]
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, size - (1 << 20)))
        h.update(f.read(1 << 20))
    h.update(str(size).encode())
    return "head-tail:" + h.hexdigest()


def summary(idx: PackIndex) -> dict:
    """The audit record (ENG-219): counts by kind, K histogram, resident and decode bytes."""
    kinds = Counter(t.kind for t in idx.tensors.values())
    return {
        "path": idx.path, "tensors": len(idx.tensors), "kinds": dict(kinds),
        "vision_skipped": len(idx.skipped),
        "files": idx.files,
        "k_histogram": {g: dict(sorted(c.items())) for g, c in sorted(idx.k_histogram().items())},
        "resident_int8_mixers": idx.resident_bytes(1),
        "resident_fp16_mixers": idx.resident_bytes(2),
        "decode_bytes_int8": idx.decode_bytes(1),
        "decode_bytes_fp16": idx.decode_bytes(2),
        "decode_bytes_int8_1k": idx.decode_bytes(1, context=1024),
        "table": None if idx.table is None else {
            "prefix": idx.table_prefix, "rows": idx.table.rows, "K": idx.table.K,
            "row_bytes": idx.table.row_bytes, "file_bytes": idx.table.file_bytes},
        "notes": idx.notes,
    }
