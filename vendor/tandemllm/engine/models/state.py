"""The per-model cache state interface (ENG-221): what a continuation needs, in named sections.

The caches (`engine/cache.py`, `engine/disk_cache.py`, `engine/spill.py`) were written for the 27B
and read its `eng.state.S` and `eng.state.conv` directly. A model whose state has other parts
(Flash-Next: GDN and conv as the 27B, plus the PLE context and its dilated conv, and the QSA
indexer's pooled keys) cannot be cached that way. `ModelState` is the one shape they can share:

    sections()        name -> tensor, every part of the recurrent state (not the KV rows, which
                      the caches keep by length); the disk tier writes them under these names
    views()           the position-indexed buffers, [(name, tensor, token axis)]: what the caches
                      slice by length (the KV rows, an indexer's keys); version 2, ENG-157
    nbytes()          the sum of the sections
    clone()           an independent copy (a snapshot)
    restore(snap)     copy a snapshot's values into this state in place (captured CUDA graphs
                      keep their tensors)
    signature()       names, shapes and dtypes in order: two states with the same signature can
                      restore each other; a file written under another signature is refused

`Qwen38State` is the 27B's: the two tensors it always had, under the names "gdn.S" and "gdn.conv",
and its KV cache's K and V (and the e4m3 scales with QWEN38_KV_FP8) as views. A model with no
recurrent state has no sections (an empty dict is a valid state).

The caches reach it through the engine's adapter (`eng.cache_state()`, ENG-157): every tier keeps
the sections in this order and the views' rows by length, and knows no model.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ModelState(Protocol):
    def sections(self) -> dict[str, torch.Tensor]: ...

    def views(self) -> list: ...

    def nbytes(self) -> int: ...

    def clone(self) -> "ModelState": ...

    def restore(self, snap: "ModelState") -> None: ...

    def signature(self) -> str: ...


def signature_of(sections: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, t in sections.items():
        h.update(f"{name}:{tuple(t.shape)}:{t.dtype};".encode())
    return h.hexdigest()[:16]


class SectionState:
    """A ModelState over a dict of named tensors (the shared implementation)."""

    def __init__(self, sections: dict[str, torch.Tensor]):
        self._sections = dict(sections)

    def sections(self) -> dict[str, torch.Tensor]:
        return dict(self._sections)

    def views(self) -> list:
        return []

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.sections().values())

    def clone(self) -> "SectionState":
        return SectionState({k: v.clone() for k, v in self.sections().items()})

    def restore(self, snap: ModelState) -> None:
        theirs = snap.sections()
        if signature_of(theirs) != self.signature():
            raise ValueError(f"state signature {signature_of(theirs)} cannot restore {self.signature()}")
        for k, v in self.sections().items():
            v.copy_(theirs[k])

    def signature(self) -> str:
        return signature_of(self.sections())


class Qwen38State(SectionState):
    """The 27B's recurrent state (`engine.model.GDNState`): S and conv of every GDN layer, as the
    same tensors (no copy), so a restore lands in the buffers its decode graphs captured; with its
    KV cache (`engine.model.KVCache`), K and V [layers, 1, heads, rows, dim] and, for an e4m3
    cache, their scales, as views along axis 3."""

    def __init__(self, gdn_state, kv=None):
        self.gdn, self.kv = gdn_state, kv
        super().__init__({"gdn.S": gdn_state.S, "gdn.conv": gdn_state.conv})

    def sections(self) -> dict[str, torch.Tensor]:
        # GDNState.swap() trades S with its spare between blocks: read the live one every time
        return {"gdn.S": self.gdn.S, "gdn.conv": self.gdn.conv}

    def views(self) -> list:
        kv = self.kv
        if kv is None:
            return []
        out = [("k", kv.k, 3), ("v", kv.v, 3)]
        if getattr(kv, "fp8", False):
            out += [("ks", kv.ks, 3), ("vs", kv.vs, 3)]
        return out
