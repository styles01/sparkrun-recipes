"""Where the engine's memory goes (OPS-41): the load ledger and the resident split.

On the GB10 the GPU and the CPU share one pool, the engine holds about 80 GiB of it, and the
residency planner (server/conductor/residency.py) has to know what that is before a role may load
beside it. Two sources, both read-only:

  * the load ledger: `mark(stage)` after each load stage in `server/app.py::_load` -- the
    allocator's `memory_allocated` / `memory_reserved` and MemAvailable at that moment. The
    difference between two marks is what that stage took (weights, the engine's own buffers with
    the KV pool, vision, drafters, the warm-up's workspace, the verify graphs). A CUDA context and
    a library's own workspace are outside the allocator and show only in MemAvailable.
  * the exact parts: `split(state)` asks the objects that hold the big tensors for their
    `nbytes` -- the weights, the KV pool, the recurrent (GDN) state, the drafters' context caches,
    the vision tower -- and the caches for their reports (state store, resident prefix). What the
    allocator holds beyond the named parts is `other` (scratch, graph pools, small buffers), and
    `reserve` is what the allocator keeps from the driver without a tensor in it.

Nothing here allocates on the device, takes the engine lock, or runs during a request: the marks
are a host call between load stages, and `split` walks attributes when the dashboard asks.
"""

from __future__ import annotations

import time

GIB = float(1 << 30)


#: () -> bytes the engine keeps in the page cache on purpose (a model's warm lookup table, ENG-239):
#: MemAvailable counts them free, every floor check takes them off (SRV-335); set by the server at load
PINNED = None
_pinned_cache = [0.0, 0]


def pinned() -> int:
    """The engine's pinned page cache now (read at most once a second)."""
    if PINNED is None:
        return 0
    now = time.monotonic()
    if now - _pinned_cache[0] >= 1.0:
        try:
            _pinned_cache[1] = int(PINNED() or 0)
        except Exception:                                          # noqa: BLE001
            _pinned_cache[1] = 0
        _pinned_cache[0] = now
    return _pinned_cache[1]


def pinned_of(eng) -> int:
    """What `eng.resident_files()` keeps warm in the page cache, in bytes."""
    return sum(int(f.get("resident_bytes") or 0) for f in eng.resident_files() if f.get("policy") == "warm")


def counted_available() -> int | None:
    """MemAvailable less the engine's pinned page cache: what every floor check counts."""
    ma = mem_available()
    return None if ma is None else max(0, ma - pinned())


def counted_meminfo() -> dict | None:
    """`/proc/meminfo` in bytes (server/slots.py `read_meminfo`) with MemAvailable counted."""
    from server.slots import read_meminfo
    mi = read_meminfo()
    if mi is not None and "MemAvailable" in mi:
        mi = dict(mi, MemAvailable=max(0, mi["MemAvailable"] - pinned()))
    return mi


def mem_available() -> int | None:
    """MemAvailable in bytes, None where /proc does not exist."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class LoadLedger:
    """The allocator and MemAvailable after each load stage."""

    def __init__(self, probe=None, avail=mem_available, clock=time.time):
        self.probe = probe or _torch_probe
        self.avail, self.clock = avail, clock
        self.marks: list[dict] = []

    def mark(self, stage: str) -> dict:
        alloc, reserved = self.probe()
        m = {"stage": stage, "allocated": alloc, "reserved": reserved,
             "mem_available": self.avail(), "t": round(self.clock(), 3)}
        self.marks.append(m)
        return m

    def stages(self) -> list[dict]:
        """One row per stage after the first: what it added to the allocator (allocated and
        reserved) and what it took from MemAvailable, in bytes (None where a reading is missing)."""
        out = []
        for a, b in zip(self.marks, self.marks[1:]):
            def d(k, sign=1):
                return None if a[k] is None or b[k] is None else sign * (b[k] - a[k])
            out.append({"stage": b["stage"], "allocated": d("allocated"),
                        "reserved": d("reserved"), "mem_available_drop": d("mem_available", -1),
                        "s": round(b["t"] - a["t"], 1)})
        return out

    def report(self) -> dict:
        return {"marks": list(self.marks), "stages": self.stages()}


def _torch_probe() -> tuple[int | None, int | None]:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())
    except Exception:                                              # noqa: BLE001
        pass
    return None, None


def _nbytes(*ts) -> int:
    return sum(int(t.numel() * t.element_size()) for t in ts if t is not None)


def drafter_caches(drafter, depth: int = 4) -> list:
    """Every drafter object below `drafter` that holds a context cache (`_ck`, `_cv`): the length
    router's arms, a merged router's head, the drafter itself. Each once."""
    seen, out, todo = set(), [], [(drafter, 0)]
    while todo:
        obj, k = todo.pop()
        if obj is None or id(obj) in seen or k > depth:
            continue
        seen.add(id(obj))
        if getattr(obj, "_ck", None) is not None:
            out.append(obj)
        mod = type(obj).__module__ or ""
        if not mod.startswith("engine."):
            continue
        for v in list(vars(obj).values()) if hasattr(obj, "__dict__") else []:
            if isinstance(v, (list, tuple)):
                todo.extend((x, k + 1) for x in v)
            elif type(v).__module__.startswith("engine."):
                todo.append((v, k + 1))
    return out


def mem_total() -> int | None:
    """MemTotal in bytes, None where /proc does not exist."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _pool_bytes(where) -> tuple[int, int]:
    try:
        return tuple(int(x) for x in where.pool_bytes()) if where is not None else (0, 0)
    except Exception:                                              # noqa: BLE001
        return 0, 0


def _roles(state: dict) -> dict[str, int]:
    """The in-process small models' ALLOCATED bytes in the shared allocator (their own pools;
    what a pool keeps free shows in `reserve`): the narrator, the shadow reader (one row when they
    share a slot), the setup guide's Wren when it has its own copy, and the decision models."""
    out: dict[str, int] = {}
    seen = set()
    for name, owner in (("narrator", state.get("narrator")), ("shadow", state.get("shadow")),
                        ("wren", state.get("wren"))):
        slot = getattr(owner, "slot", None) if owner is not None else None
        if slot is None or id(slot) in seen or getattr(slot, "state", None) != "loaded":
            continue
        seen.add(id(slot))
        res, alloc = _pool_bytes(slot) if hasattr(slot, "pool_bytes") else (0, 0)
        if not alloc:
            try:
                alloc = int(slot.device_bytes())
            except Exception:                                      # noqa: BLE001
                alloc = 0
        out[name] = alloc
    reg = state.get("deciders")
    if reg is not None:
        n = 0
        try:
            for st in reg.status():
                if st.get("state") != "loaded":
                    continue
                a = _pool_bytes(getattr(reg, "_placements", {}).get(st["name"]))[1]
                n += a or int(st.get("device_bytes") or 0)
        except Exception:                                          # noqa: BLE001
            pass
        out["deciders"] = n
    return out


def split(state: dict, probe=None, avail=mem_available, total=mem_total, pool=None) -> dict:
    """The resident set, in bytes, from the objects that hold it. `state` is server/app.py's
    STATE; a part the server does not have (no vision, no store, the fake engine) is left out,
    and every part is a reading, never an estimate.

    Two kinds of memory. The allocator's (`allocated`, `reserved`): weights, recurrent state,
    vision, caches, the small models' pools. And, with the shared KV pool (engine/kvpool.py), the
    KV pages the driver maps outside the allocator: `kv_pool` (the engine's attention KV) and
    `drafter_kv` (the drafters' context KV) are then the pages COMMITTED for every slot, never a
    paged tensor's `nbytes` (that is its reserved address range, 26.6 GiB a 262,144-row slot,
    which is no memory at all: `kv_reserved_va`). `vmm_parts` names the parts outside the
    allocator. `outside` is what the box uses that neither holds -- the CUDA context, the
    process's host memory, other programs -- so that

        allocator reserved + KV pages committed + outside = MemTotal - MemAvailable (`in_use`)

    and the dashboard's bar adds up to what MemAvailable says is gone."""
    from engine import kvpool
    alloc, reserved = (probe or _torch_probe)()
    pool = pool if pool is not None else kvpool.POOL
    parts: dict[str, int] = {}

    def put(name, n):
        if n:
            parts[name] = parts.get(name, 0) + int(n)

    def static(*ts) -> int:
        return _nbytes(*(t for t in ts if t is not None and not kvpool.paged(t)))

    eng = state.get("engine")
    if eng is not None:
        # the engine's own parts, from its adapter: weights, its KV and recurrent state (the 27B's in
        # engine/models/qwen3_5/adapter.py; another model names its own, e.g. a draft layer's weights)
        for name, n in eng.memory_parts().items():
            put(name, n)
    caches = drafter_caches(state.get("drafter"))
    put("drafter_kv", sum(static(d._ck, getattr(d, "_cv", None)) for d in caches))
    put("vision", getattr(state.get("vision"), "nbytes", 0))
    put("state_store", getattr(state.get("state_store"), "bytes", 0))
    res = state.get("resident")
    put("resident_prefix", (res.report().get("bytes") or 0) if hasattr(res, "report") else 0)
    for name, n in _roles(state).items():
        put(name, n)
    in_alloc = sum(parts.values())
    vmm_parts: list[str] = []
    kinds = pool.by_kind() if pool is not None and hasattr(pool, "by_kind") else None
    vmm = 0
    if kinds is not None and kinds.get("paged"):
        for name, key in (("kv_pool", "engine_kv"), ("drafter_kv", "drafter_kv"), ("kv_pool", "other")):
            # SRV-655: a model whose KV buffers are not named `kv.*` has its rows in the pool's
            # "other" kind; its adapter counts no paged bytes itself (a 262,144-row window's
            # addresses, 6.44 GB with 0.61 committed, were its KV here)
            if kinds.get(key):
                put(name, kinds[key])
                if name not in vmm_parts:
                    vmm_parts.append(name)
        vmm = int(kinds.get("engine_kv", 0) + kinds.get("drafter_kv", 0) + kinds.get("other", 0))
    ma, mt = avail(), (total() if total is not None else None)
    in_use = (mt - ma) if ma is not None and mt is not None else None
    out = {"allocated": alloc, "reserved": reserved, "parts": parts, "vmm_parts": vmm_parts,
           "other": (alloc - in_alloc) if alloc is not None else None,
           "reserve": (reserved - alloc) if alloc is not None and reserved is not None else None,
           "kv_committed": vmm if kinds is not None else None,
           "kv_reserved_va": int(kinds["reserved_va"]) if kinds is not None else None,
           "mem_available": ma, "mem_total": mt, "in_use": in_use,
           "outside": (in_use - (reserved or 0) - vmm) if in_use is not None
           and reserved is not None else None,
           "max_len": state.get("max_len"), "drafter_caches": len(caches),
           # page cache the engine keeps on purpose (a model's warm lookup table): free to the
           # kernel, counted off by every floor check (SRV-335)
           "pinned_cache": pinned_of(eng) if eng is not None else 0}
    led = state.get("load_ledger")
    if led is not None:
        out["load"] = led.stages()
    return out
