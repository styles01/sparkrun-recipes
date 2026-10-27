"""The 27B's cold adapter verbs (ENG-E8): installed on `Qwen38Engine` when this module is imported, so
the shared server and caches read `eng.caps` and call `eng.cost()`, `eng.cache_state()`,
`eng.discard_pending()`, `eng.resume_at(n)`, `eng.settle()`, `eng.manifest()` and `eng.precapture()` on
either model without a branch, and `engine/model.py` (in
the 27B's arithmetic manifest) keeps its bytes. The server's loader takes the class from here; the
hot verbs stay the engine's own methods."""

from __future__ import annotations

from engine.methods.manifest import Manifest
from engine.model import Qwen38Engine
from engine.models.adapter import Capabilities, CostModel
from engine.models.state import Qwen38State

#: what the 27B's stored rows depend on (fingerprint v3, ENG-243): the forward, verify and commit
#: bodies, the KV pages and the tree they index, the vision tower (image rows are prompt rows), the
#: drafters whose state is cached, and the knob registry (the defaults of these files' knobs); the
#: `tools/` kernels they import and the tile tables their knobs name follow (engine/methods/manifest.py)
MODULES = ("engine/config.py", "engine/drafters/__init__.py", "engine/drafters/dflash2.py",
           "engine/drafters/draft_graph.py", "engine/drafters/dspark.py", "engine/drafters/mtp.py",
           "engine/gdn.py", "engine/kvpool.py", "engine/linear.py", "engine/loader.py", "engine/model.py",
           "engine/settings.py", "engine/tree.py", "engine/verify_graph.py", "engine/vision.py")


def cost(eng) -> CostModel:
    """The 27B's snapshot bytes: the GDN state and conv tails whole (the state's own count,
    `GDNState.nbytes`, as the caches budgeted before ENG-157), K and V of every attention layer a
    token (bf16); the parts as the dashboard showed them (server/app.py `cache_stats`, before
    2026-10-11)."""
    cfg = eng.cfg
    rec = eng.state.S.numel() * 4
    conv = eng.state.conv.numel() * eng.state.conv.element_size()
    return CostModel(section_bytes=eng.state.nbytes,
                     view_bytes_per_token=len(cfg.attention_layers) * cfg.num_key_value_heads * cfg.head_dim * 2 * 2,
                     parts={"recurrent": rec, "conv": conv})


def discard_pending(eng) -> None:
    """Forget an uncommitted verify and the traces of the last block (after the state was
    overwritten by a restore): a chain's walked state, a folded commit, the block's trace and tree."""
    eng._pending_walk = False
    eng._pend = None                  # a commit pending on the state just overwritten
    eng._trace = None
    eng.tree = None
    eng.trace = None


def cache_state(eng):
    """What the caches keep of a sequence (engine/models/state.py): S and conv, the KV views."""
    return Qwen38State(eng.state, eng.kv)


def resume_at(eng, n: int) -> None:
    """The engine continues at length `n` from the state the caller just wrote into it."""
    discard_pending(eng)
    eng.kv.length = n
    eng.state.primed = n > 0


def settle(eng) -> None:
    """Land a chain accepted in full whose walked state is not in yet (before a snapshot)."""
    eng._settle()


def precapture(eng, drafter) -> int | None:
    """The verify graphs for every width at the first context class, at load (None: graphs off). A
    router that sizes every block on the staircase (QWEN38_LEN_MODE calc/wide) submits any row count
    up to 32; the others stay at 2..16 and capture the rest lazily."""
    if eng._graphs_for(2, 0) is None:
        return None
    import engine.model as _M
    return eng._graphs.precapture(widths=range(2, min(_M.VERIFY_ROWS, getattr(drafter, "max_nodes", 31) + 1) + 1)
                                  if getattr(drafter, "calc", False) else range(2, 17))


def memory_parts(eng) -> dict:
    """The dashboard's memory split of the 27B (server/memsplit.py): its weights, the KV cache's
    tensors held in the allocator (pages of the shared pool are counted by the pool), the GDN state."""
    from engine import kvpool
    w, kv, st = getattr(eng, "w", None), getattr(eng, "kv", None), getattr(eng, "state", None)

    def nb(*ts):
        return sum(int(t.numel() * t.element_size()) for t in ts if t is not None)
    return {"weights": getattr(w, "bytes_fp8", 0) + getattr(w, "bytes_other", 0),
            "kv_pool": nb(*(t for t in (getattr(kv, k, None) for k in ("k", "v", "ks", "vs"))
                            if t is not None and not kvpool.paged(t))),
            "gdn_state": nb(*(getattr(st, k, None) for k in ("S", "conv", "_spare")))}


def resident_files(eng) -> list:
    """Files the engine keeps in the page cache on purpose: none (its weights are in device memory)."""
    return []


def manifest(eng=None) -> Manifest:
    return Manifest(arch="qwen3_5", modules=MODULES)


Qwen38Engine.caps = Capabilities(caches=True, side_slots=True)
Qwen38Engine.cost = cost
Qwen38Engine.discard_pending = discard_pending
Qwen38Engine.cache_state = cache_state
Qwen38Engine.resume_at = resume_at
Qwen38Engine.settle = settle
Qwen38Engine.manifest = manifest
Qwen38Engine.resident_files = resident_files
Qwen38Engine.memory_parts = memory_parts
Qwen38Engine.precapture = precapture

__all__ = ["Qwen38Engine", "cache_state", "cost", "discard_pending", "manifest", "precapture", "resume_at",
           "settle"]
