"""Where a decision model's tensors live and run: its own CUDA stream and its own memory pool.

Qwen's output must stay byte-identical with the deciders loaded or not, and its speed must not
move. So nothing is shared: every decider allocation goes to a `torch.cuda.MemPool` of its own
(one per loaded model) through `use_mem_pool`, which routes only the current thread's allocations,
so Qwen's allocator pool and its allocation history are untouched (allocation history alone once
moved a kernel from 0.254 to 0.627 ms); and every decider kernel is issued on the decider's own
stream, which the scope synchronises before it hands the GPU back. Unloading a model drops its
tensors and then its pool, so the pool's segments go back with it.

On the CPU (the box's CPU suite, the equivalence run without the GPU) the scope does nothing.
"""

from __future__ import annotations

import contextlib
import threading
import time

import torch

# torch 2.13.0's `torch._C._cuda_memorySnapshot` takes one reference too few on every item of the
# tuple it is handed: `(pool id high, pool id low, include_traces)`. Each call of
# `MemPool.snapshot()` or `torch.cuda.memory_snapshot()` takes one reference off `True` (or
# `False`) and one off each pool-id int, which are interpreter singletons on Python 3.11 (`True`,
# and the small ints). Called for every `/metrics` collector since 0.4.0, it drained `True`
# in about three hours and the engine died with "Fatal Python error: bool_dealloc" (2026-10-05
# 13:12, the first time). Measured on the box's CPU: 1000 calls of `MemPool.snapshot()` take
# 1000 references off `True` and off each id int; a call with `None` takes none. So the pools
# are read from the one snapshot of every segment (`None`), filtered by `segment_pool_id`.
# tests/test_pool_snapshot_refcount.py holds the measurement and the fix.
SNAPSHOT_TTL_S = 0.25          # one /metrics render asks ~30 times; the pools barely move meanwhile
_SNAP_LOCK = threading.Lock()
_SNAP: dict = {"t": None, "segments": None}


def _all_segments(clock=time.monotonic) -> list:
    """Every segment of the caching allocator, from the snapshot call that is handed no tuple
    (and so drops no reference), reused for `SNAPSHOT_TTL_S`."""
    with _SNAP_LOCK:
        now = clock()
        if _SNAP["segments"] is not None and _SNAP["t"] is not None \
                and now - _SNAP["t"] < SNAPSHOT_TTL_S:
            return _SNAP["segments"]
        segs = torch._C._cuda_memorySnapshot(None)["segments"]
        _SNAP.update(t=now, segments=segs)
        return segs


def pool_segments(pool, clock=time.monotonic) -> list:
    """The segments of one `torch.cuda.MemPool`, without `MemPool.snapshot()` (see above)."""
    pid = tuple(pool.id)
    return [s for s in _all_segments(clock)
            if tuple(s.get("segment_pool_id") or ()) == pid]


def resolve(spec: str | None) -> torch.device:
    """`QWEN38_DECIDER_DEVICE`: `cuda`, `cuda:N`, `cpu`, or `auto` (cuda when there is one). A
    CUDA device always carries its index: `use_mem_pool` refuses a bare `cuda`."""
    if spec and spec != "auto":
        dev = torch.device(spec)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


class Device:
    def __init__(self, spec: str | None = "auto"):
        self.device = resolve(spec)
        self.cuda = self.device.type == "cuda"
        self._stream = None

    def __str__(self) -> str:
        return str(self.device)

    def pool(self):
        """A new memory pool for one model, None on the CPU."""
        return torch.cuda.MemPool() if self.cuda else None

    @property
    def stream(self):
        if self.cuda and self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device)
        return self._stream

    @contextlib.contextmanager
    def scope(self, pool):
        """Allocate into `pool` and issue on the decider stream; synchronised on the way out, so no
        decider kernel is in flight once the caller lets go of the GPU."""
        if not self.cuda:
            yield
            return
        stream = self.stream
        try:
            with torch.cuda.stream(stream):
                if pool is None:
                    yield
                else:
                    with torch.cuda.use_mem_pool(pool, self.device):
                        yield
        finally:
            stream.synchronize()


class Placement:
    """What a model's `load` and `predict_rows` are handed: the device, the model's own pool, the
    scope that routes into both, and `gpu()`, the scope taken in the idle window (the engine's
    lock) for the copies onto the device at load."""

    def __init__(self, device: Device, gate=None, gate_timeout_s: float = 600.0):
        self.dev = device
        self.device = device.device
        self.pool = device.pool()
        self.gate, self.gate_timeout_s = gate, gate_timeout_s

    def scope(self):
        return self.dev.scope(self.pool)

    @contextlib.contextmanager
    def gpu(self):
        if not self.dev.cuda or self.gate is None:
            with self.scope():
                yield
            return
        if not self.gate.acquire(timeout=self.gate_timeout_s):
            raise TimeoutError(f"the engine stayed busy for {self.gate_timeout_s:.0f} s; the load "
                               f"waits for an idle window")
        try:
            with self.scope():
                yield
        finally:
            self.gate.release()

    def release(self) -> None:
        """Drop the pool; called after the model let go of every tensor in it."""
        self.pool = None

    def pool_bytes(self) -> tuple[int, int]:
        """(reserved, allocated) bytes of the model's pool. A live MemPool never hands its free
        segments back (torch 2.13: `empty_cache` skips a pool still in use; measured 2026-10-04,
        556 MiB stayed reserved after every tensor in it was freed), so `reserved - allocated`
        is memory only dropping the pool returns. (0, 0) on the CPU. Read through
        `pool_segments`, never `self.pool.snapshot()`: that call drops references to `True` and
        to the pool's id ints (torch 2.13.0, see the top of this file)."""
        if self.pool is None:
            return 0, 0
        try:
            segs = pool_segments(self.pool)
        except Exception:                                          # noqa: BLE001
            return 0, 0
        return (int(sum(s.get("total_size", 0) for s in segs)),
                int(sum(s.get("allocated_size", 0) for s in segs)))
