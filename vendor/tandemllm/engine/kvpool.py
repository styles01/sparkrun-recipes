"""The shared KV pool: a sequence's KV window is reserved as ADDRESSES, and memory is committed
only for the rows a conversation actually holds.

Why: before this, every sequence slot allocated its whole window at load -- 106.5 KiB a token
(64 KiB of engine attention KV, 2 x 20 KiB of drafter context KV, 256 B of rotary table) -- so four
slots of 262,144 tokens would have taken ~106 GiB before serving anything. The box could only
afford one long window and three short ones, and even those pushed the narrator and the Laya
decider out of memory (2026-10-10, OPS-89).

How (CUDA virtual memory management, the driver's cuMem* API):

  * `PagedBuffer` reserves a virtual range the size of the full tensor (`cuMemAddressReserve`) and
    hands out ONE torch tensor over it with the exact shape, dtype and strides the static
    `torch.zeros` had. Every kernel, every view, every CUDA graph sees the same addresses as with
    a static buffer: nothing downstream changes, and results are bit-identical by construction.
  * Physical pages (`cuMemCreate`, the allocation granularity, 2 MiB on GB10) are mapped into the
    range on demand (`ensure(rows)`): for every (layer, head) region of the tensor the pages that
    hold rows [0, rows). A fresh page is zeroed on the current stream, so a slot's never-written
    rows read as the static buffer's `torch.zeros` did.
  * `trim(rows)` unmaps and releases the pages past `rows` (after a device synchronize: no queued
    kernel may still touch them). The addresses stay reserved, so captured graphs stay valid.
  * `KVPool` is the one account every paged buffer commits through: bytes committed by owner (a
    slot), a commit gate (`reclaim`, then refuse) that keeps MemAvailable over a hard floor, and the
    numbers /health and the admission read.

The rows a kernel may touch must be committed before it runs. The engine keeps a margin
(`KVPool.margin_rows`) of committed rows past every write it is about to issue, so a tile that
reads a few rows past the end (masked) never reads an unmapped address.

Without a CUDA driver (a Mac, a CPU test) `PagedBuffer` falls back to a plain `torch.zeros`
tensor and only does the bookkeeping, so the CPU tests exercise the accounting and policy.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time

GiB = 1 << 30
MiB = 1 << 20

# ------------------------------------------------------------------ the driver
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class _Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                ("location", _Loc), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", _AllocFlags)]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Loc), ("flags", ctypes.c_int)]


class CudaError(RuntimeError):
    pass


class Driver:
    """The few cuMem* calls the pool needs, through ctypes on libcuda (no extra package)."""

    _inst = None
    _inst_lock = threading.Lock()

    @classmethod
    def get(cls, device: int = 0) -> "Driver | None":
        """The driver bindings, or None where there is no CUDA driver with VMM support."""
        with cls._inst_lock:
            if cls._inst is None:
                try:
                    cls._inst = cls(device)
                except (OSError, CudaError, AttributeError):
                    cls._inst = False
            return cls._inst or None

    def __init__(self, device: int = 0):
        self.lib = ctypes.CDLL("libcuda.so.1")
        u64 = ctypes.c_uint64
        sz = ctypes.c_size_t
        L = self.lib
        L.cuInit.argtypes = [ctypes.c_uint]
        L.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        L.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        L.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        L.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        L.cuMemGetAllocationGranularity.argtypes = [ctypes.POINTER(sz), ctypes.POINTER(_Prop),
                                                    ctypes.c_int]
        L.cuMemAddressReserve.argtypes = [ctypes.POINTER(u64), sz, sz, u64, u64]
        L.cuMemAddressFree.argtypes = [u64, sz]
        L.cuMemCreate.argtypes = [ctypes.POINTER(u64), sz, ctypes.POINTER(_Prop), u64]
        L.cuMemRelease.argtypes = [u64]
        L.cuMemMap.argtypes = [u64, sz, sz, u64, u64]
        L.cuMemUnmap.argtypes = [u64, sz]
        L.cuMemSetAccess.argtypes = [u64, sz, ctypes.POINTER(_Access), sz]
        self._check(L.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self._check(L.cuDeviceGet(ctypes.byref(dev), int(device)), "cuDeviceGet")
        self.dev = dev.value
        sup = ctypes.c_int()
        self._check(L.cuDeviceGetAttribute(
            ctypes.byref(sup), CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, self.dev),
            "cuDeviceGetAttribute")
        if not sup.value:
            raise CudaError("the device has no virtual memory management")
        ctx = ctypes.c_void_p()
        self._check(L.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), self.dev), "cuDevicePrimaryCtxRetain")
        self.ctx = ctx
        self.prop = _Prop()
        self.prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        self.prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        self.prop.location.id = self.dev
        self.access = _Access()
        self.access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        self.access.location.id = self.dev
        self.access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        g = ctypes.c_size_t()
        self._bind()
        self._check(L.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(self.prop),
                                                    CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                    "cuMemGetAllocationGranularity")
        self.granularity = int(g.value)
        self._tls = threading.local()

    @staticmethod
    def _check(rc: int, what: str) -> None:
        if rc != 0:
            raise CudaError(f"{what} failed: CUresult {rc}")

    def _bind(self) -> None:
        """Make the primary context (the one torch uses) current in this thread."""
        tls = getattr(self, "_tls", None)
        if tls is not None and getattr(tls, "bound", False):
            return
        cur = ctypes.c_void_p()
        self.lib.cuCtxGetCurrent(ctypes.byref(cur))
        if not cur.value:
            self._check(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
        if tls is not None:
            tls.bound = True

    def reserve(self, size: int, align: int = 0) -> int:
        self._bind()
        p = ctypes.c_uint64()
        self._check(self.lib.cuMemAddressReserve(ctypes.byref(p), size, align, 0, 0),
                    "cuMemAddressReserve")
        return int(p.value)

    def free_va(self, ptr: int, size: int) -> None:
        self._bind()
        self._check(self.lib.cuMemAddressFree(ptr, size), "cuMemAddressFree")

    def map_new(self, ptr: int, size: int) -> int:
        """A fresh physical allocation of `size` bytes mapped read-write at `ptr`; its handle."""
        self._bind()
        h = ctypes.c_uint64()
        self._check(self.lib.cuMemCreate(ctypes.byref(h), size, ctypes.byref(self.prop), 0),
                    "cuMemCreate")
        try:
            self._check(self.lib.cuMemMap(ptr, size, 0, h.value, 0), "cuMemMap")
            try:
                self._check(self.lib.cuMemSetAccess(ptr, size, ctypes.byref(self.access), 1),
                            "cuMemSetAccess")
            except CudaError:
                self.lib.cuMemUnmap(ptr, size)
                raise
        except CudaError:
            self.lib.cuMemRelease(h.value)
            raise
        return int(h.value)

    def unmap_release(self, ptr: int, size: int, handle: int) -> None:
        self._bind()
        self._check(self.lib.cuMemUnmap(ptr, size), "cuMemUnmap")
        self._check(self.lib.cuMemRelease(handle), "cuMemRelease")


class _CAI:
    """A `__cuda_array_interface__` over a raw device range, for a zero-copy torch tensor."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (int(nbytes),), "typestr": "|u1",
                                         "data": (int(ptr), False), "version": 3,
                                         "strides": None}


# ------------------------------------------------------------------ the pool
class PoolFull(RuntimeError):
    """A commit the pool could not make: MemAvailable would fall under the hard floor even after
    every reclaim. The engine never lets this reach a kernel: the rows are not written."""


def _meminfo_avail() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


class KVPool:
    """One account for every paged KV buffer of every slot.

    `commit(owner, nbytes)` is asked before pages are mapped. It passes when MemAvailable minus the
    bytes stays at or above `hard_floor`; otherwise it calls the reclaimers (`add_reclaimer`, in
    registration order: the cheapest caches first, then idle conversations' pages) until it does,
    and raises `PoolFull` if nothing more can be given back. `hard_floor` is below the admission
    floor (server/slots.py `Admission`, QSE_PARALLEL_FLOOR_GB): admission keeps requests out while
    the box is near its floor; the hard floor is the last line that keeps a running request from
    ever taking the box under it.
    """

    def __init__(self, hard_floor: int = 8 * GiB, margin_rows: int = 256, meminfo=_meminfo_avail,
                 on_event=None):
        self.hard_floor = int(hard_floor)
        self.margin_rows = int(margin_rows)
        self.meminfo = meminfo
        self.on_event = on_event
        self.lock = threading.RLock()
        self.by_owner: dict = {}
        self.buffers: list = []
        self.reclaimers: list = []          # [(name, fn(need_bytes) -> bytes given back)]
        self.stats = {"commits": 0, "pages_mapped": 0, "pages_released": 0, "reclaims": 0,
                      "reclaimed_bytes": 0, "refused": 0, "commit_ms": 0.0, "release_ms": 0.0,
                      "peak_committed": 0, "min_avail_after": None}

    # -- accounting
    @property
    def committed(self) -> int:
        with self.lock:
            return sum(self.by_owner.values())

    def owner_bytes(self, owner) -> int:
        with self.lock:
            return int(self.by_owner.get(owner, 0))

    def _add(self, owner, nbytes: int) -> None:
        with self.lock:
            self.by_owner[owner] = self.by_owner.get(owner, 0) + int(nbytes)
            c = sum(self.by_owner.values())
            if c > self.stats["peak_committed"]:
                self.stats["peak_committed"] = c

    def add_reclaimer(self, name: str, fn) -> None:
        self.reclaimers.append((name, fn))

    def _event(self, kind: str, **kw) -> None:
        if self.on_event is not None:
            try:
                self.on_event(kind, kw)
            except Exception:                                          # noqa: BLE001
                pass

    # -- the gate
    def gate(self, owner, nbytes: int, reclaim: bool = True) -> None:
        """Pass if `nbytes` more may be committed; reclaim first if not; raise `PoolFull`."""
        if nbytes <= 0 or self.hard_floor <= 0:
            return
        avail = self.meminfo() if self.meminfo is not None else None
        if avail is None:
            return
        after = avail - nbytes
        if after >= self.hard_floor:
            self._note_after(after)
            return
        if reclaim:
            for name, fn in list(self.reclaimers):
                want = self.hard_floor - after
                try:
                    got = int(fn(want, owner) or 0)
                except Exception as exc:                               # noqa: BLE001
                    self._event("reclaim_error", level=name, error=f"{type(exc).__name__}: {exc}")
                    got = 0
                if got:
                    self.stats["reclaims"] += 1
                    self.stats["reclaimed_bytes"] += got
                    self._event("reclaim", level=name, bytes=got, owner=str(owner))
                avail = self.meminfo() or avail
                after = avail - nbytes
                if after >= self.hard_floor:
                    self._note_after(after)
                    return
        self.stats["refused"] += 1
        self._event("refused", owner=str(owner), need=nbytes, avail=avail, floor=self.hard_floor)
        raise PoolFull(f"KV pool: committing {nbytes / MiB:.0f} MiB for {owner} would leave "
                       f"{after / GiB:.1f} GiB available, under the hard floor "
                       f"{self.hard_floor / GiB:.1f} GiB")

    def _note_after(self, after: int) -> None:
        m = self.stats["min_avail_after"]
        self.stats["min_avail_after"] = after if m is None else min(m, after)

    def by_kind(self) -> dict:
        """Committed bytes by what the pages hold: `engine_kv` (the attention KV, buffers
        `kv.*`) and `drafter_kv` (the drafters' context KV, `draft.*`), and the addresses
        reserved for them (`reserved_va`: a window's whole range, NOT memory). Only buffers
        mapped through the driver count as pages; a bookkeeping-only buffer (no driver, a CPU
        test) is an ordinary allocation and is left to the allocator's figures."""
        out = {"engine_kv": 0, "drafter_kv": 0, "other": 0, "reserved_va": 0, "paged": 0}
        with self.lock:
            bufs = list(self.buffers)
        for b in bufs:
            if getattr(b, "drv", None) is None:
                continue
            out["paged"] += 1
            kind = ("drafter_kv" if str(b.name).startswith("draft") else
                    "engine_kv" if str(b.name).startswith("kv") else "other")
            out[kind] += b.committed_bytes()
            out["reserved_va"] += int(getattr(b, "va_size", b.nbytes))
        return out

    def report(self) -> dict:
        with self.lock:
            owners = {str(k): round(v / GiB, 3) for k, v in self.by_owner.items()}
            c = sum(self.by_owner.values())
        avail = self.meminfo() if self.meminfo is not None else None
        kinds = self.by_kind()
        return {"committed_gib": round(c / GiB, 3), "by_owner_gib": owners,
                "committed_by_kind_gib": {k: round(kinds[k] / GiB, 3)
                                          for k in ("engine_kv", "drafter_kv", "other")},
                "reserved_va_gib": round(sum(b.nbytes for b in self.buffers) / GiB, 2),
                "buffers": len(self.buffers), "hard_floor_gib": round(self.hard_floor / GiB, 2),
                "margin_rows": self.margin_rows,
                "mem_available_gib": None if avail is None else round(avail / GiB, 2),
                "granularity": (self.buffers[0].page if self.buffers else None),
                **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.stats.items()}}


#: the process's pool; None until the server (or a test) builds one. `PagedBuffer` without a pool
#: argument uses it, and a static allocation is used when there is none.
POOL: KVPool | None = None
#: who the next paged buffers are booked to (server/app.py sets "slot1".. before it builds a side
#: slot; slot 0's engine is built under the default)
OWNER = "slot0"
#: a window shorter than this stays a static buffer (the shadow reader's 8-row engine, tests)
PAGED_MIN_ROWS = int(os.environ.get("QSE_KV_POOL_MIN_ROWS", "16384") or 16384)


def enabled() -> bool:
    return POOL is not None


class PagedBuffer:
    """A tensor of shape `shape` whose axis `row_axis` is the sequence axis, committed by rows.

    `tensor` is the full-size torch tensor (static layout, contiguous). `ensure(rows)` maps the
    pages that hold rows [0, rows) of every region; `trim(rows)` releases the pages that hold only
    rows past `rows`. `rows` is the committed row count (a page multiple, capped at the window)."""

    def __init__(self, shape, row_axis: int, dtype, device, owner="?", name: str = "kv",
                 pool: KVPool | None = None):
        import torch
        self.pool = pool if pool is not None else POOL
        self.shape = tuple(int(s) for s in shape)
        self.row_axis = int(row_axis)
        self.dtype = dtype
        self.owner = owner
        self.name = name
        self.max_rows = self.shape[self.row_axis]
        esz = torch.empty((), dtype=dtype).element_size()
        outer = 1
        for s in self.shape[:self.row_axis]:
            outer *= s
        inner = esz
        for s in self.shape[self.row_axis + 1:]:
            inner *= s
        self.outer, self.row_bytes = outer, inner
        self.region_bytes = self.max_rows * inner
        self.nbytes = outer * self.region_bytes
        self.rows = 0
        self.handles: dict[int, int] = {}            # page index -> physical handle
        self.lock = threading.Lock()
        drv = Driver.get() if (self.pool is not None and str(device).startswith("cuda")) else None
        self.drv = drv
        if drv is None:
            # the bookkeeping only (CPU tests, a machine without the driver API)
            self.page = 2 * MiB
            self.base = None
            self._flat = None
            self.tensor = torch.zeros(self.shape, dtype=dtype, device=device)
        else:
            self.page = drv.granularity
            size = -(-self.nbytes // self.page) * self.page
            self.va_size = size
            self.base = drv.reserve(size, self.page)
            # torch's view of the range asks the driver which device the pointer is on, which an
            # unmapped address cannot answer: map the first page while the view is made
            self._map_pages([0])
            flat = torch.as_tensor(_CAI(self.base, self.nbytes), device=device)
            self._flat = flat
            self.tensor = flat.view(dtype).view(self.shape)
            self._release_pages([0], sync=False)
        if self.pool is not None:
            with self.pool.lock:
                self.pool.buffers.append(self)

    # -- pages
    def _pages_for(self, lo_rows: int, hi_rows: int) -> list[int]:
        """The pages that hold some byte of rows [lo, hi) of any region."""
        if hi_rows <= lo_rows:
            return []
        out = set()
        P = self.page
        for r in range(self.outer):
            a = r * self.region_bytes + lo_rows * self.row_bytes
            b = r * self.region_bytes + hi_rows * self.row_bytes
            out.update(range(a // P, (b - 1) // P + 1))
        return sorted(out)

    def _map_pages(self, pages) -> None:
        P = self.page
        for p in pages:
            if p in self.handles:
                continue
            self.handles[p] = self.drv.map_new(self.base + p * P, P) if self.drv is not None else 0

    def _release_pages(self, pages, sync: bool = True) -> int:
        pages = [p for p in pages if p in self.handles]
        if not pages:
            return 0
        if self.drv is not None:
            if sync:
                import torch
                torch.cuda.synchronize()
            P = self.page
            for p in pages:
                self.drv.unmap_release(self.base + p * P, P, self.handles.pop(p))
        else:
            for p in pages:
                self.handles.pop(p)
        return len(pages) * self.page

    def committed_bytes(self) -> int:
        return len(self.handles) * self.page

    def round_rows(self, rows: int) -> int:
        """`rows` rounded up to where the pages that hold it end in every region (capped)."""
        rows = max(0, min(int(rows), self.max_rows))
        if rows == 0:
            return 0
        if self.region_bytes % self.page == 0 and self.page % self.row_bytes == 0:
            per = self.page // self.row_bytes
            return min(self.max_rows, -(-rows // per) * per)
        return rows

    def need_bytes(self, rows: int) -> int:
        """What committing rows [0, rows) would add, in bytes."""
        rows = max(0, min(int(rows), self.max_rows))
        if rows <= self.rows:
            return 0
        return sum(1 for p in self._pages_for(0, self.round_rows(rows))
                   if p not in self.handles) * self.page

    def ensure(self, rows: int) -> int:
        """Commit rows [0, rows) (plus the page rounding). Returns the bytes newly committed.
        Raises `PoolFull` (nothing mapped) when the pool's gate says no."""
        rows = max(0, min(int(rows), self.max_rows))
        if rows <= self.rows:
            return 0
        with self.lock:
            if rows <= self.rows:
                return 0
            target = self.round_rows(rows)
            t0 = time.perf_counter()
            new = [p for p in self._pages_for(0, target) if p not in self.handles]
            nbytes = len(new) * self.page
            if self.pool is not None:
                self.pool.gate(self.owner, nbytes)
            self._map_pages(new)
            if self.drv is not None and new:
                P = self.page
                # zero the fresh pages: never-written rows read as the static `torch.zeros` did
                for a, b in _runs(new):
                    self._flat[a * P:min(self.nbytes, b * P)].zero_()
            self.rows = target
            if self.pool is not None:
                self.pool._add(self.owner, nbytes)
                self.pool.stats["commits"] += 1
                self.pool.stats["pages_mapped"] += len(new)
                self.pool.stats["commit_ms"] += (time.perf_counter() - t0) * 1e3
            return nbytes

    def trim(self, rows: int = 0) -> int:
        """Release the pages past rows [0, rows). Returns the bytes given back."""
        rows = max(0, min(int(rows), self.max_rows))
        with self.lock:
            target = self.round_rows(rows)
            t0 = time.perf_counter()
            keep = set(self._pages_for(0, target))
            gone = [p for p in list(self.handles) if p not in keep]
            if not gone:
                return 0
            nbytes = self._release_pages(gone)
            self.rows = min(self.rows, target)
            if self.pool is not None:
                self.pool._add(self.owner, -nbytes)
                self.pool.stats["pages_released"] += len(gone)
                self.pool.stats["release_ms"] += (time.perf_counter() - t0) * 1e3
            return nbytes

    def close(self) -> None:
        """Give everything back (the buffer must not be used after)."""
        self.trim(0)
        if self.drv is not None and self.base is not None:
            self._flat = None
            self.tensor = None
            self.drv.free_va(self.base, self.va_size)
            self.base = None
        if self.pool is not None:
            with self.pool.lock:
                if self in self.pool.buffers:
                    self.pool.buffers.remove(self)


def _runs(pages):
    """Sorted page indices as [start, end) runs."""
    out = []
    for p in sorted(pages):
        if out and out[-1][1] == p:
            out[-1][1] = p + 1
        else:
            out.append([p, p + 1])
    return out


class PagedSet:
    """The paged buffers of one sequence (its engine KV and its drafters' context KV), grown and
    trimmed together by rows. `ensure(rows)` commits every buffer up to `rows` plus the pool's
    margin; the cheap fast path is one int compare."""

    def __init__(self, owner, pool: KVPool | None = None):
        self.owner = owner
        self.pool = pool if pool is not None else POOL
        self.bufs: list[PagedBuffer] = []
        self.rows = 0               # rows every buffer holds committed (the minimum)
        self.high = 0               # the most rows ever asked of this set since its last trim

    def add(self, buf: PagedBuffer) -> PagedBuffer:
        buf.owner = self.owner
        self.bufs.append(buf)
        _BY_PTR[buf.tensor.data_ptr()] = self
        self.rows = min((b.rows for b in self.bufs), default=0)
        return buf

    def on_device(self) -> bool:
        """True when the buffers are mapped through the driver (not the bookkeeping fallback)."""
        return any(b.drv is not None for b in self.bufs)

    def margin(self) -> int:
        return self.pool.margin_rows if self.pool is not None else 0

    def set_owner(self, owner) -> None:
        """Book this set to `owner` from now on (its committed bytes move with it)."""
        if self.pool is not None:
            c = self.committed_bytes()
            self.pool._add(self.owner, -c)
            self.pool._add(owner, c)
        self.owner = owner
        for b in self.bufs:
            b.owner = owner

    def ensure(self, rows: int) -> int:
        want = int(rows) + (self.pool.margin_rows if self.pool is not None else 0)
        if want <= self.rows:
            return 0
        got = 0
        for b in self.bufs:
            got += b.ensure(want)
        self.rows = min((b.rows for b in self.bufs), default=0)
        self.high = max(self.high, int(rows))
        return got

    def need_bytes(self, rows: int) -> int:
        want = int(rows) + (self.pool.margin_rows if self.pool is not None else 0)
        return sum(b.need_bytes(want) for b in self.bufs)

    def trim(self, rows: int = 0) -> int:
        want = int(rows) + (self.pool.margin_rows if self.pool is not None and rows > 0 else 0)
        got = 0
        for b in self.bufs:
            got += b.trim(want)
        self.rows = min((b.rows for b in self.bufs), default=0)
        self.high = int(rows)
        return got

    def committed_bytes(self) -> int:
        return sum(b.committed_bytes() for b in self.bufs)

    def report(self) -> dict:
        return {"rows": self.rows, "committed_gib": round(self.committed_bytes() / GiB, 3),
                "buffers": len(self.bufs)}


#: a paged tensor's base address -> its set, for the copy-ins that only hold the tensor
_BY_PTR: dict = {}


def paged(t) -> bool:
    """Is `t` (a whole KV or draft-KV tensor) backed by driver-mapped pages of the pool? Such a
    tensor's `nbytes` is its reserved ADDRESS range, not memory: what it holds is its buffer's
    committed pages (server/memsplit.py reads those)."""
    if t is None:
        return False
    try:
        ps = _BY_PTR.get(t.data_ptr())
    except Exception:                                              # noqa: BLE001
        return False
    return ps is not None and ps.on_device()


def ensure_for(t, rows: int) -> None:
    """Commit rows [0, rows) of the paged set `t` (a whole KV or draft-KV tensor) belongs to,
    before a copy writes them (a restore, the pin, the park, the disk tier, the NVMe spill, a
    side slot's carry). Nothing for a static tensor or without a pool."""
    if POOL is None or t is None:
        return
    ps = _BY_PTR.get(t.data_ptr())
    if ps is not None and int(rows) + ps.margin() > ps.rows:
        ps.ensure(int(rows))


def env_on() -> bool:
    return (os.environ.get("QSE_KV_POOL") or "off").strip().lower() in ("1", "on", "true", "yes")
