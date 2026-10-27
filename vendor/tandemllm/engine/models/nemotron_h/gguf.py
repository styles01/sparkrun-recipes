"""A GGUF file read as a NemotronH checkpoint: the header, the key-value metadata, the tensor table, the
data as a read-only memory map, and the per-type histogram.

    g = GGUFFile.open(path)            # header, metadata and tensor table (nothing else is read)
    g.types()                          # {"IQ2_XXS": n, "Q8_0": n, ...}
    g.raw(name)                        # the tensor's bytes as stored (a numpy view of the map)

The format (version 2 and 3): magic "GGUF", version u32, tensor count u64, metadata count u64; each
metadata entry a string key, a u32 type and its value (strings are u64 length + bytes, arrays a u32
item type, a u64 count and the items); each tensor a string name, u32 dims, the dims as u64 (ggml's
order: the fastest dimension first), a u32 ggml type and a u64 offset from the data start, which sits
at the next multiple of `general.alignment` (32 when absent) after the table.

Only the types this engine decodes are accepted (`BLOCKS`); any other stops the open with the tensor's
name and the type's. The canonical names of llama.cpp's NemotronH tensors are in `CANONICAL`, so a
GGUF reads to the same names as the safetensors checkpoint (engine/models/nemotron_h/pack.py).
"""

from __future__ import annotations

import math
import os
import re
import struct
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"GGUF"
#: ggml type id -> (name, values a block, bytes a block): the types the engine reads
BLOCKS = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 30: ("BF16", 1, 2), 8: ("Q8_0", 32, 34), 16: ("IQ2_XXS", 256, 66),
          40: ("NVFP4", 64, 36)}
#: every ggml type id this reader names in a refusal (llama.cpp's ggml.h)
TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
              19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
              27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4"}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

#: llama.cpp's NemotronH tensor names (blk.N.*) -> the canonical names of ref.py / pack.py
CANONICAL = [
    (r"^token_embd\.weight$", "embeddings.weight"),
    (r"^output_norm\.weight$", "norm_f.weight"),
    (r"^output\.weight$", "lm_head.weight"),
    (r"^blk\.(\d+)\.attn_norm\.weight$", r"layers.\1.norm.weight"),
    (r"^blk\.(\d+)\.ssm_in\.weight$", r"layers.\1.mixer.in_proj.weight"),
    (r"^blk\.(\d+)\.ssm_conv1d\.weight$", r"layers.\1.mixer.conv1d.weight"),
    (r"^blk\.(\d+)\.ssm_conv1d\.bias$", r"layers.\1.mixer.conv1d.bias"),
    (r"^blk\.(\d+)\.ssm_dt\.bias$", r"layers.\1.mixer.dt_bias"),
    (r"^blk\.(\d+)\.ssm_a$", r"layers.\1.mixer.A"),
    (r"^blk\.(\d+)\.ssm_d$", r"layers.\1.mixer.D"),
    (r"^blk\.(\d+)\.ssm_norm\.weight$", r"layers.\1.mixer.norm.weight"),
    (r"^blk\.(\d+)\.ssm_out\.weight$", r"layers.\1.mixer.out_proj.weight"),
    (r"^blk\.(\d+)\.ssm_in\.scale$", r"layers.\1.mixer.in_proj.scale"),
    (r"^blk\.(\d+)\.ssm_out\.scale$", r"layers.\1.mixer.out_proj.scale"),
    (r"^blk\.(\d+)\.attn_q\.weight$", r"layers.\1.mixer.q_proj.weight"),
    (r"^blk\.(\d+)\.attn_k\.weight$", r"layers.\1.mixer.k_proj.weight"),
    (r"^blk\.(\d+)\.attn_v\.weight$", r"layers.\1.mixer.v_proj.weight"),
    (r"^blk\.(\d+)\.attn_output\.weight$", r"layers.\1.mixer.o_proj.weight"),
    (r"^blk\.(\d+)\.ffn_gate_inp\.weight$", r"layers.\1.mixer.gate.weight"),
    (r"^blk\.(\d+)\.exp_probs_b\.bias$", r"layers.\1.mixer.gate.e_score_correction_bias"),
    (r"^blk\.(\d+)\.ffn_latent_down\.weight$", r"layers.\1.mixer.fc1_latent_proj.weight"),
    (r"^blk\.(\d+)\.ffn_latent_up\.weight$", r"layers.\1.mixer.fc2_latent_proj.weight"),
    (r"^blk\.(\d+)\.ffn_up_exps\.weight$", r"layers.\1.mixer.experts.up_proj"),
    (r"^blk\.(\d+)\.ffn_down_exps\.weight$", r"layers.\1.mixer.experts.down_proj"),
    (r"^blk\.(\d+)\.ffn_up_shexp\.weight$", r"layers.\1.mixer.shared_experts.up_proj.weight"),
    (r"^blk\.(\d+)\.ffn_down_shexp\.weight$", r"layers.\1.mixer.shared_experts.down_proj.weight"),
]
_CANON = [(re.compile(a), b) for a, b in CANONICAL]


def canonical(name: str) -> str | None:
    for rx, rep in _CANON:
        if rx.match(name):
            return rx.sub(rep, name)
    return None


@dataclass
class GGUFTensor:
    name: str               # as stored
    dims: tuple             # ggml order: the fastest dimension first
    type: int
    offset: int             # absolute byte offset in the file
    nbytes: int

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type, f"type {self.type}")

    @property
    def shape(self) -> tuple:
        """The torch order (the slowest dimension first): [N, K] for a weight of N outputs."""
        return tuple(reversed(self.dims))


@dataclass
class GGUFFile:
    path: str
    version: int
    meta: dict
    tensors: dict[str, GGUFTensor]
    data_start: int
    alignment: int
    _map: object = field(default=None, repr=False)

    @classmethod
    def open(cls, path: str, accept_all: bool = False) -> "GGUFFile":
        """The header and the tensor table; a tensor of a type outside `BLOCKS` raises with its name
        (`accept_all`: kept, for an audit of a file the engine cannot serve)."""
        with open(path, "rb") as f:
            if f.read(4) != MAGIC:
                raise ValueError(f"{path}: not a GGUF file (no GGUF magic)")
            version, n_t, n_kv = struct.unpack("<IQQ", f.read(20))
            if version not in (2, 3):
                raise ValueError(f"{path}: GGUF version {version}; this reader takes 2 and 3")
            meta = {}
            for _ in range(n_kv):
                k = _string(f)
                (vt,) = struct.unpack("<I", f.read(4))
                meta[k] = _value(f, vt)
            infos = []
            for _ in range(n_t):
                name = _string(f)
                (nd,) = struct.unpack("<I", f.read(4))
                dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
                ty, off = struct.unpack("<IQ", f.read(12))
                infos.append((name, dims, ty, off))
            pos = f.tell()
        align = int(meta.get("general.alignment", 32))
        if align <= 0 or align & (align - 1):
            raise ValueError(f"{path}: general.alignment {align} is not a power of two")
        start = (pos + align - 1) // align * align
        size = os.path.getsize(path)
        tensors = {}
        for name, dims, ty, off in infos:
            if ty not in BLOCKS and not accept_all:
                raise ValueError(f"{path}: tensor {name} is {TYPE_NAMES.get(ty, f'type {ty}')}; this engine decodes "
                                 f"{', '.join(v[0] for v in BLOCKS.values())}")
            n = math.prod(dims)                 # Python ints: np.prod wraps at 64 bits
            per, nb = BLOCKS.get(ty, (1, 0))[1:] if ty in BLOCKS else (1, 0)
            if ty in BLOCKS and n % per:
                raise ValueError(f"{name}: {n} values are not whole {TYPE_NAMES[ty]} blocks of {per}")
            nbytes = n // per * nb
            if ty in BLOCKS and start + off + nbytes > size:
                raise ValueError(f"{path}: tensor {name} ends at {start + off + nbytes}, the file has {size} bytes "
                                 "(truncated)")
            tensors[name] = GGUFTensor(name, tuple(int(d) for d in dims), ty, start + off, nbytes)
        return cls(path, version, meta, tensors, start, align)

    # --------------------------------------------------------------------------- summaries
    def types(self) -> dict:
        return dict(Counter(t.type_name for t in self.tensors.values()))

    def type_bytes(self) -> dict:
        out = Counter()
        for t in self.tensors.values():
            out[t.type_name] += t.nbytes
        return dict(out)

    def arch(self) -> str:
        return str(self.meta.get("general.architecture", ""))

    def canonical_names(self) -> dict:
        """{canonical name: stored name} of every tensor the language model reads; tower and unknown
        tensors are left out (`unmapped`)."""
        return {c: n for n, c in ((n, canonical(n)) for n in self.tensors) if c is not None}

    def unmapped(self) -> list:
        return sorted(n for n in self.tensors if canonical(n) is None)

    # --------------------------------------------------------------------------- reading
    def raw(self, name: str) -> np.ndarray:
        """The tensor's bytes as stored (uint8, a view of the read-only map)."""
        if self._map is None:
            self._map = np.memmap(self.path, dtype=np.uint8, mode="r")
        t = self.tensors[name]
        return self._map[t.offset:t.offset + t.nbytes]


def _string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _value(f, vt: int):
    if vt in _SCALAR:
        fmt = _SCALAR[vt]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vt == 8:
        return _string(f)
    if vt == 9:
        it, n = struct.unpack("<IQ", f.read(12))
        if it in _SCALAR and it != 7:
            fmt = _SCALAR[it]
            sz = struct.calcsize(fmt)
            return list(np.frombuffer(f.read(sz * n), dtype=np.dtype(fmt)).tolist())
        return [_value(f, it) for _ in range(n)]
    raise ValueError(f"GGUF metadata value type {vt} is not in the format")
