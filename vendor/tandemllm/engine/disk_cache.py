"""The disk tier: prompt-prefix state on the NVMe, under the caches in RAM.

THE PROBLEM. opencode starts every session with the same 24k to 30k tokens (the reasoning header,
its system text, the tool block), and a cold prefill of 30k tokens takes about 44 s. The resident
prefix serves it only while those rows are the ones in the engine's buffer: a restart, a deploy, a
long unrelated request or a day of other sessions throws them away, and the next session pays the
44 s again. The owner, 2026-10-02: keep them, and the packages and memory files with them, on the
NVMe, under a budget set in the dashboard.

WHAT IS ON DISK. A restore needs, at token L: the KV rows of every position-indexed buffer for
0..L (the 16 attention layers', and the drafters' own) and the recurrent state at L (the 48 S
and their convolution tails). The two have different lifetimes, and the files follow them, as
`ResidentPrefix` does in memory:

  * a CHUNK file holds rows [b - chunk, b) of every buffer, plus the chunk's token ids, keyed by
    `(b, h(tokens[:b]))`. A row depends only on the tokens before it, so the key names everything
    the bytes depend on, and two prompts that share 30k tokens share their 30 chunk files;
  * an ANCHOR file holds S and conv at one boundary L, keyed by `(L, h(tokens[:L]))`. It is valid
    at L alone. An anchor is usable when every chunk below it is on disk.

Both are on the prefill's chunk grid, so a restore at L followed by forwards from L on the same
grid is the arithmetic of a cold prefill, and `tests/test_disk_cache.py` holds the restored
logits, S, conv and rows to bit equality with a cold prefill on the small random model. A hash is
a hint, never an answer: before any byte reaches the engine, every chunk's stored token ids are
compared with the prompt's element by element.

WHAT GOES TO DISK, decided before the prefill runs (`DiskTier.plan`):

  * a prefix prefilled `admit_after` times (default 2): every grid line a prefill forwards cold is
    counted by its hash; the longest line of this prompt counted that often gets an anchor, and
    the chunks below it that are not on disk yet. Resumed lines are not counted: the count is the
    work the tier would have saved.
  * a known head: the first long request of a session (opencode's `x-session-id`) gets anchors at
    its last `head_anchors` lines, because the next session parts from it where its first user
    message starts, and that is near the end of the first request.
  * a branch: a repeated line whose count falls at the next line -- an earlier prompt read this
    far and went on differently. A session in another folder parts from this one there, not at
    the user message: opencode's system text puts its <env> block (the working directory, the
    date) and the folder's instructions after the tools and its fixed text, and two sessions in
    two folders share the first 13,954 of 25,616 tokens (2026-10-03). The longest repeated line
    alone is not that line when this prompt shares more with an earlier one of its own folder.

The counts are traffic, not rows: they survive a new fingerprint (a deploy that changed the
engine's code sweeps every file, and the first two sessions after it were cold up to the shared
head), and only `clear` and another grid drop them. They decide what is written, never what is
read: a restore compares every token whatever the counts said.

The anchor's S and conv are copied to a staging slot on the engine thread at its line (a few ms
on the 27B, between two chunk forwards). The chunk rows stay where the prefill wrote them: a
writer thread copies them out of the engine's buffers after the prefill and writes them, while the
request decodes (decode writes rows past the prompt only). WRITE-BEHIND: the window closes when
the next prefill would write rows under the job's anchor (`fence`); what is not copied by then is
dropped and journaled, never waited for. A request is never blocked by the tier.

THE ORDER of a lookup (engine/cache.py `prefill`): the resident prefix, the pin, the park, the
state store, then this tier, and a cold prefill. A disk hit is taken only when it starts later
than every RAM source by more than its read costs (`DiskTier.find`); it reads only the chunks the
rows in place do not already hold, streams them through the staging buffer into the engine's own
buffers, and the restored prompt becomes the resident prompt with the disk anchor as its anchor
(the promotion into RAM).

CRASH SAFETY. Every file is written to `<name>.tmp`, fsynced, renamed and the directory fsynced;
the index is the files themselves (their headers), so a crash leaves at worst a `.tmp`, which the
next start deletes. Every file carries a checksum of its payload, checked on every read before a
byte is copied to the engine; a bad file is deleted with every longer chunk of its chain.

THE DISK. `budget` (the dashboard's GB) bounds the tier; `floor` (never under 50 GB) is the free
space it never takes the file system below. Eviction is least valuable first: hits plus one, times
the tokens an anchor saves per byte it holds alone, halved per day unused; past `ttl` an entry goes
whatever its value. Chunks no anchor needs go with it.

MEMORY. The staging buffer (two anchor slots and two chunk slots, 0.5 GB on the 27B) is allocated
once, at the first use, page-aligned for O_DIRECT and pinned for the device copies; it is the
tier's whole host footprint, declared by `staging_bytes` before it exists. No read or write goes
through the page cache (O_DIRECT, or fadvise DONTNEED where the file system refuses it): the
2026-09-25 lesson that page cache counted as available is not given back in time.

PRIVACY. Token ids and engine state, never text, under the engine's home on the box, 0700. The
index keeps a short preview for opencode heads and packages only (their text is the client's and
the operator's, not the user's), and the dashboard shows nothing else of an entry but its counts.
"""

from __future__ import annotations

import errno
import hashlib
import json
import mmap
import os
import queue
import re
import threading
import time
import zlib
from collections import OrderedDict

from engine import capture_lock
from engine import kvpool as _kvp

ALIGN = 4096
MAGIC = b"QSEDISK1"
FORMAT = 1
#: anchor files: format 2 names each section as the model's state does (engine/models/state.py,
#: ENG-157); format 1 (the 27B's before it) named them "S" and "conv", read as "gdn.S" and "gdn.conv"
ANCHOR_FORMAT = 2
FORMAT1_SECTIONS = {"S": "gdn.S", "conv": "gdn.conv"}
GB = 1000 ** 3
#: the free space the tier never takes its file system below, whatever the settings say
FLOOR_BYTES = 50 * GB


def align(n: int) -> int:
    return (int(n) + ALIGN - 1) // ALIGN * ALIGN


# --------------------------------------------------------------------------- the files
def _open(path: str, flags: int, mode: int = 0o600) -> tuple[int, bool]:
    """(fd, direct): O_DIRECT where the platform and the file system take it."""
    direct = getattr(os, "O_DIRECT", 0)
    if direct:
        try:
            return os.open(path, flags | direct, mode), True
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.EOPNOTSUPP):
                raise
    return os.open(path, flags, mode), False


def _drop_cache(fd: int, offset: int = 0, length: int = 0) -> None:
    fn = getattr(os, "posix_fadvise", None)
    if fn is not None:
        try:
            fn(fd, offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _write_all(fd: int, views) -> None:
    for v in views:
        mv = memoryview(v).cast("B")
        off = 0
        while off < len(mv):
            off += os.write(fd, mv[off:])


#: the longest the tier's writer waits for a session spill's save between two of its files
SPILL_WAIT_S = 30.0


def write_file(path: str, views, fsync: bool = True) -> bool:
    """Write `views` (aligned buffers of aligned lengths) to `path` crash-safe: `path.tmp`,
    fsync, rename, the directory fsynced. Returns whether O_DIRECT carried it."""
    tmp = path + ".tmp"
    fd, direct = _open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        try:
            _write_all(fd, views)
        except OSError as exc:
            if not direct or exc.errno != errno.EINVAL:
                raise
            # a buffer the device would not take directly: the same bytes through the cache
            os.close(fd)
            fd, direct = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), False
            _write_all(fd, views)
        if fsync:
            os.fsync(fd)
        if not direct:
            _drop_cache(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    if fsync:
        try:
            dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    return direct


def read_into(path: str, offset: int, view) -> int:
    """Fill `view` (aligned, of an aligned length) from `path` at `offset`."""
    mv = memoryview(view).cast("B")
    fd, direct = _open(path, os.O_RDONLY)
    try:
        off = 0
        while off < len(mv):
            try:
                if hasattr(os, "preadv"):
                    n = os.preadv(fd, [mv[off:]], offset + off)
                else:
                    data = os.pread(fd, len(mv) - off, offset + off)
                    n = len(data)
                    mv[off:off + n] = data
            except OSError as exc:
                if not direct or exc.errno != errno.EINVAL:
                    raise
                os.close(fd)
                fd, direct = os.open(path, os.O_RDONLY), False
                continue
            if n == 0:
                raise EOFError(f"{path}: {off} of {len(mv)} bytes at {offset}")
            off += n
        if not direct:
            _drop_cache(fd, offset, len(mv))
    finally:
        os.close(fd)
    return len(mv)


def checksum(view, kind: str = "crc32") -> int:
    """The payload's checksum. crc32 (zlib) by default; `sum64` is a positional sum of 64-bit
    words over 4 KiB blocks, for a box whose crc32 is slower than its NVMe."""
    if kind == "crc32":
        return zlib.crc32(memoryview(view).cast("B")) & 0xFFFFFFFF
    import numpy as np
    a = np.frombuffer(memoryview(view).cast("B"), dtype="<u8")
    blk = a[: a.size // 512 * 512].reshape(-1, 512).sum(axis=1, dtype=np.uint64)
    w = np.arange(1, 2 * blk.size, 2, dtype=np.uint64)
    tail = a[blk.size * 512:].sum(dtype=np.uint64)
    return int(((blk * w).sum(dtype=np.uint64) + tail * np.uint64(2 * blk.size + 1))
               & np.uint64(0xFFFFFFFFFFFFFFFF))


# --------------------------------------------------------------------------- the staging buffer
class _Buf:
    """A page-aligned host buffer (an anonymous mmap), pinned for the device copies when the
    engine is on a GPU. O_DIRECT needs the alignment; the copies need the pinning to be async."""

    def __init__(self, nbytes: int, cuda: bool):
        self.nbytes = align(max(ALIGN, nbytes))
        self.mm = mmap.mmap(-1, self.nbytes)
        self._t = None
        self.pinned = False
        if cuda:
            try:
                import torch
                rc = torch.cuda.cudart().cudaHostRegister(self.tensor().data_ptr(), self.nbytes, 0)
                self.pinned = int(getattr(rc, "value", rc)) == 0
            except Exception:                                      # noqa: BLE001
                self.pinned = False

    def tensor(self):
        if self._t is None:
            import torch
            self._t = torch.frombuffer(self.mm, dtype=torch.uint8)
        return self._t

    def view(self, off: int, nbytes: int):
        return memoryview(self.mm)[off:off + nbytes]

    def close(self) -> None:
        self._t = None
        try:
            self.mm.close()
        except BufferError:
            pass

    def typed(self, off: int, dtype, shape):
        import torch
        n = 1
        for s in shape:
            n *= int(s)
        nb = n * torch.empty((), dtype=dtype).element_size()
        return self.tensor()[off:off + nb].view(dtype).view(*shape)


def _sections(shapes) -> tuple[list[tuple[int, int]], int]:
    """(offset, bytes) of each tensor laid end to end on 4 KiB lines, and the total."""
    out, off = [], 0
    for nb in shapes:
        out.append((off, int(nb)))
        off += align(nb)
    return out, off


def _nb(t) -> int:
    return t.numel() * t.element_size()


def _row_shape(t, ax: int, rows: int) -> list[int]:
    s = list(t.shape)
    s[ax] = rows
    return s


def _row_bytes(t, ax: int) -> int:
    return t.numel() // t.shape[ax] * t.element_size()


def _hx(h: int) -> str:
    return f"{h & ((1 << 64) - 1):016x}"


# --------------------------------------------------------------------------- one write job
class _Job:
    """What one prefill writes: anchors at `lines` (captured on the engine thread as the prefill
    passes them) and the chunks below the longest one that are not on disk (copied out of the
    engine's buffers after the prefill, until `limit`)."""

    def __init__(self, keys, hashes, lines: dict, chunks: list[int], session, created,
                 chunk: int):
        top = max(lines)
        self.keys = tuple(int(x) for x in keys[:top])
        self.hashes = {b: hashes[b] for b in range(0, top + 1, chunk)}
        self.lines = lines                      # L -> {"kind", "label", "preview"}
        self.chunks = chunks                    # boundaries b, ascending
        self.session = session
        self.created = created
        self.limit = top                        # rows a fence has left this job
        self.ready = 0                          # rows the prefill has written so far
        self.copied = 0                         # rows below which every chunk is copied or on disk
        self.views = None
        self.event = None
        self.done = threading.Event()
        self.written = {"chunks": 0, "anchors": 0, "bytes": 0}
        self.dropped: dict[str, int] = {}

    def drop(self, why: str, n: int = 1) -> None:
        self.dropped[why] = self.dropped.get(why, 0) + n


class Hit:
    """A usable anchor: its line, its index record, the chunks below it with their headers, the
    rows the engine already holds (`have`, on the grid) and the bytes a restore reads."""
    __slots__ = ("L", "anchor", "chain", "have", "need_bytes")

    def __init__(self, L, anchor, chain, have, need_bytes):
        self.L, self.anchor, self.chain = L, anchor, chain
        self.have, self.need_bytes = have, need_bytes


#: The cold prefill rate the read cost is weighed against: 738, 699 and 629 tokens a second at
#: 32k, 64k and 128k (docs/caches.md). The higher figure, so the disk is taken only when it wins
#: against the fastest prefill.
PREFILL_TOK_S = 740.0
#: The read rate assumed until the tier has measured its own: the box's O_DIRECT read is 7.35
#: GB/s and crc32 5.6 (tools/disk_cache_bench.py, 2026-10-02); serial, about 3.2.
READ_BPS = 3.0e9
SETTINGS = {"mode": "on", "budget_gb": 16, "ttl_days": 7, "admit_after": 2, "min_free_gb": 50}
COUNT_MAX = 200_000
SESSIONS_MAX = 4096
#: the branch lines a prefill may anchor (the longest of them)
BRANCH_MAX = 4
SWEEP_S = 300.0


class DiskTier:
    """Prompt-prefix state on the NVMe (see the module's docstring).

    `root`: the directory (0700, created). `chunk`: the prefill's grid. `fingerprint`: what the
    rows on disk depend on besides the tokens (the weights, the engine's code, the arithmetic
    flags); a file with another one is swept at open. `settings`: the dashboard's (mode,
    budget_gb, ttl_days, admit_after, min_free_gb), applied live by `apply`. `on_event(kind,
    detail)`: the journal. `free_bytes(path)`: the file system's free space (a test's fake disk).
    """

    def __init__(self, root: str, *, chunk: int, fingerprint: str, settings: dict | None = None,
                 head_anchors: int = 4, min_tokens: int = 4096, on_event=None, clock=time.time,
                 free_bytes=None, preview=None, also=()):
        self.root = os.path.realpath(os.path.expanduser(root))
        self.chunk = int(chunk)
        self.fp = str(fingerprint)
        # the fingerprints whose files hold the same rows (`equivalents`): read, never written
        self.fps = frozenset((self.fp, *(str(x) for x in also)))
        self.head_anchors = max(0, int(head_anchors))
        self.min_tokens = max(self.chunk, int(min_tokens))
        self.on_event = on_event
        self.clock = clock
        self._free = free_bytes or _statvfs_free
        self.preview = preview
        self.cfg = dict(SETTINGS)
        self.apply(settings or {})
        self.lock = threading.RLock()
        self._fence = threading.Lock()
        self.chunks: dict[tuple, dict] = {}
        self.anchors: dict[tuple, dict] = {}
        self._alen: dict[int, int] = {}
        self.counts: OrderedDict = OrderedDict()
        self.sessions: OrderedDict = OrderedDict()
        self.bytes = 0
        self.read_bps = READ_BPS
        self.prefill_tok_s = PREFILL_TOK_S
        self.stats = {"lookups": 0, "hits": 0, "misses": 0, "declined_slow": 0,
                      "rejected_collisions": 0, "incomplete": 0, "tokens_restored": 0,
                      "bytes_read": 0, "read_ms": 0.0, "last_read_gbs": None,
                      "last_restore_ms": None, "jobs": 0, "writes_chunk": 0, "writes_anchor": 0,
                      "bytes_written": 0, "write_ms": 0.0, "last_write_gbs": None,
                      "evicted": {}, "dropped": {}, "corrupt": 0, "failed_restores": 0,
                      "swept_tmp": 0, "swept_fingerprint": 0, "cleared": 0, "direct": None,
                      "admitted": {}}
        self.events: OrderedDict = OrderedDict()
        self._ev_n = 0
        self._layout = None                   # (anchor_bytes, chunk_bytes) once a view is seen
        self.buf_anchor: list[_Buf] = []
        self.buf_chunk: list[_Buf] = []
        self._hdr_w = None
        self._hdr_r = None
        self._chain_w = None
        self._slots = threading.Condition()
        self._free_anchor: list[int] = []
        self._io = threading.Lock()           # the chunk slots: a restore or the writer
        # A session spill writing (engine/spill.py): the writer waits between files
        self.spill_active = 0
        self.staging_planned = 0
        self._est = (0, 0)                    # (anchor, chunk) bytes before the slots exist
        self._side = None
        self._q: queue.Queue = queue.Queue()
        self._open_jobs: list[_Job] = []
        self._dirty = False
        self._saved = 0.0
        self._thread: threading.Thread | None = None
        self._pool = None
        # the day's NVMe write budget (WriteGuard), shared with the session spill
        self.guard: WriteGuard | None = None

    # --- settings ----------------------------------------------------------------------------
    def apply(self, settings: dict) -> list[str]:
        """The dashboard's settings over the defaults; returns what changed. A lower budget or
        TTL takes effect at the writer's next sweep (queued here)."""
        new = dict(SETTINGS)
        new.update({k: v for k, v in (settings or {}).items() if k in SETTINGS and v is not None})
        new["min_free_gb"] = max(FLOOR_BYTES / GB, float(new["min_free_gb"]))
        changed = [f"{k}={new[k]}" for k in SETTINGS if new[k] != self.cfg.get(k)]
        self.cfg = new
        if changed and getattr(self, "_thread", None) is not None:
            self._q.put(("sweep",))
        return changed

    @property
    def on(self) -> bool:
        return self.cfg["mode"] == "on"

    @property
    def budget(self) -> int:
        return int(float(self.cfg["budget_gb"]) * GB)

    @property
    def floor(self) -> int:
        return int(float(self.cfg["min_free_gb"]) * GB)

    @property
    def ttl(self) -> float:
        return float(self.cfg["ttl_days"]) * 86400.0

    # --- the directory -----------------------------------------------------------------------
    def _path(self, kind: str, b: int, h: int) -> str:
        return os.path.join(self.root, "chunks" if kind == "chunk" else "anchors",
                            f"{b:07d}-{_hx(h)}.{'chk' if kind == 'chunk' else 'anc'}")

    def open(self) -> "DiskTier":
        """Create the directory, index the files on it (their headers), sweep what is stale, and
        start the writer."""
        for d in (self.root, os.path.join(self.root, "chunks"), os.path.join(self.root, "anchors")):
            os.makedirs(d, mode=0o700, exist_ok=True)
            os.chmod(d, 0o700)
        self._scan()
        self._load_index()
        self._sweep("open")
        self._thread = threading.Thread(target=self._run, name="disk-cache-writer", daemon=True)
        self._thread.start()
        return self

    def _scan(self) -> None:
        for sub, kind in (("chunks", "chunk"), ("anchors", "anchor")):
            d = os.path.join(self.root, sub)
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if name.endswith(".tmp"):
                    _unlink(p)
                    self.stats["swept_tmp"] += 1
                    continue
                try:
                    meta = self._read_header(p, kind, chain=True)
                except Exception:                                  # noqa: BLE001
                    _unlink(p)
                    self.stats["corrupt"] += 1
                    continue
                if meta.get("fp") not in self.fps or meta.get("chunk") != self.chunk:
                    _unlink(p)
                    self.stats["swept_fingerprint"] += 1
                    continue
                self._index_add(kind, meta, p)

    def _index_add(self, kind: str, meta: dict, path: str) -> None:
        nb = os.path.getsize(path)
        h = int(meta["h"], 16)
        rec = {"path": path, "bytes": nb, "created": meta.get("created", self.clock()),
               "used": meta.get("created", self.clock())}
        with self.lock:
            if kind == "chunk":
                key = (int(meta["b"]), h)
                old = self.chunks.get(key)
                self.chunks[key] = rec
            else:
                key = (int(meta["L"]), h)
                rec.update(hits=0, kind=meta.get("entry", "prefix"), label=meta.get("label"),
                           preview=meta.get("preview"), chain=list(meta["chain"]),
                           saves=int(meta["L"]))
                old = self.anchors.get(key)
                self.anchors[key] = rec
                if old is None:
                    self._alen[key[0]] = self._alen.get(key[0], 0) + 1
            if old is not None:
                self.bytes -= old["bytes"]
            self.bytes += nb

    def _index_drop(self, kind: str, key: tuple) -> int:
        with self.lock:
            d = self.chunks if kind == "chunk" else self.anchors
            rec = d.pop(key, None)
            if rec is None:
                return 0
            if kind == "anchor":
                self._alen[key[0]] -= 1
                if not self._alen[key[0]]:
                    del self._alen[key[0]]
            self.bytes -= rec["bytes"]
        _unlink(rec["path"])
        return rec["bytes"]

    def _read_header(self, path: str, kind: str, chain: bool = False) -> dict:
        """The file's metadata, `meta["_hdr"]` the header's length (where the payload starts)."""
        size = self._hdr_size(kind)
        if self._hdr_r is None or self._hdr_r.nbytes < size:
            self._hdr_r = _Buf(max(size, self._hdr_size("chunk"), self._hdr_size("anchor")),
                               cuda=False)
        mv = self._hdr_r.view(0, size)
        read_into(path, 0, mv)
        if bytes(mv[:8]) != MAGIC:
            raise ValueError(f"{path}: not a disk-cache file")
        n = int.from_bytes(mv[8:12], "little")
        full = self._hdr_len(n, kind)
        if full > size:
            # a long section list (a model of many layers): the metadata runs past the first line
            self._hdr_r.close()
            self._hdr_r = _Buf(full, cuda=False)
            mv = self._hdr_r.view(0, full)
            read_into(path, 0, mv)
            size = full
        meta = json.loads(bytes(mv[12:12 + n]).decode())
        if meta.get("kind") != kind or meta.get("format") not in ((FORMAT, ANCHOR_FORMAT) if kind == "anchor"
                                                                  else (FORMAT,)):
            raise ValueError(f"{path}: format {meta.get('format')} kind {meta.get('kind')}")
        import numpy as np
        meta["_hdr"] = size
        j = max(ALIGN, align(12 + n))
        if kind == "chunk":
            meta["tokens"] = np.frombuffer(mv[j:j + 8 * self.chunk], dtype="<i8").tolist()
        elif chain:
            sec = next(x for x in meta["sections"] if x["name"] == "chain")
            cb = _Buf(align(sec["bytes"]), cuda=False)
            try:
                read_into(path, size + sec["off"], cb.view(0, cb.nbytes))
                if checksum(cb.view(0, align(sec["bytes"]))) != sec["crc"]:
                    raise ValueError(f"{path}: the chain fails its checksum")
                meta["chain"] = [int(x) for x in np.frombuffer(cb.view(0, sec["bytes"]),
                                                               dtype="<u8")]
            finally:
                cb.close()
        return meta

    def _hdr_size(self, kind: str) -> int:
        """The shortest header: one line of metadata (and a chunk's token ids after it)."""
        return align(ALIGN + 8 * self.chunk) if kind == "chunk" else ALIGN

    def _hdr_len(self, n: int, kind: str) -> int:
        """The header of a file whose metadata is `n` bytes: the metadata on as many lines as it
        needs (one for every file before ENG-157), then a chunk's token ids."""
        j = max(ALIGN, align(12 + n))
        return j + align(8 * self.chunk) if kind == "chunk" else j

    def _chain_buf(self, chain: list[int]) -> memoryview:
        """An anchor's chain -- the hash of every chunk line up to it -- as an aligned section."""
        import numpy as np
        nb = align(8 * len(chain))
        if self._chain_w is None or self._chain_w.nbytes < nb:
            self._chain_w = _Buf(nb, cuda=False)
        mv = self._chain_w.view(0, nb)
        mv[:nb] = b"\0" * nb
        mv[:8 * len(chain)] = np.asarray(chain, dtype="<u8").tobytes()
        return mv

    def _header(self, meta: dict, kind: str, tokens=None) -> memoryview:
        raw = json.dumps(meta, separators=(",", ":")).encode()
        size = self._hdr_len(len(raw), kind)
        if self._hdr_w is None or self._hdr_w.nbytes < size:
            if self._hdr_w is not None:
                self._hdr_w.close()
            self._hdr_w = _Buf(max(size, self._hdr_size("chunk"), self._hdr_size("anchor")), cuda=False)
        mv = self._hdr_w.view(0, size)
        mv[:size] = b"\0" * size
        mv[:8] = MAGIC
        mv[8:12] = len(raw).to_bytes(4, "little")
        mv[12:12 + len(raw)] = raw
        if tokens is not None:
            import numpy as np
            j = max(ALIGN, align(12 + len(raw)))
            mv[j:j + 8 * len(tokens)] = np.asarray(tokens, dtype="<i8").tobytes()
        return mv

    # --- the persistent part of the index: counts, sessions, hits ----------------------------
    def _load_index(self) -> None:
        p = os.path.join(self.root, "index.json")
        try:
            with open(p) as f:
                doc = json.load(f)
        except (OSError, ValueError):
            return
        # the counts carry over to a new fingerprint on the same grid (a document before the grid
        # was saved in it had the served one); the rest describes files that were swept
        if doc.get("chunk", self.chunk) == self.chunk:
            for L, h, n in doc.get("counts", [])[-COUNT_MAX:]:
                self.counts[(int(L), int(h, 16))] = int(n)
            self._dirty = doc.get("fp") != self.fp
        if doc.get("fp") not in self.fps:
            return
        for s, ts in doc.get("sessions", [])[-SESSIONS_MAX:]:
            self.sessions[s] = ts
        for name, st in (doc.get("anchors") or {}).items():
            for rec in self.anchors.values():
                if os.path.basename(rec["path"]) == name:
                    rec["hits"] = int(st.get("hits", 0))
                    rec["used"] = float(st.get("used", rec["used"]))
        for name, used in (doc.get("chunks") or {}).items():
            for rec in self.chunks.values():
                if os.path.basename(rec["path"]) == name:
                    rec["used"] = float(used)
        for k in ("hits", "misses", "lookups", "tokens_restored", "bytes_read", "jobs",
                  "writes_chunk", "writes_anchor", "bytes_written"):
            self.stats[k] = int((doc.get("stats") or {}).get(k, 0))

    def _save_index(self, force: bool = False) -> None:
        now = self.clock()
        if not self._dirty or (not force and now - self._saved < 60):
            return
        with self.lock:
            doc = {"format": FORMAT, "fp": self.fp, "chunk": self.chunk, "saved": now,
                   "counts": [[L, _hx(h), n] for (L, h), n in self.counts.items()],
                   "sessions": [[s, ts] for s, ts in self.sessions.items()],
                   "anchors": {os.path.basename(r["path"]): {"hits": r["hits"], "used": r["used"]}
                               for r in self.anchors.values()},
                   "chunks": {os.path.basename(r["path"]): r["used"] for r in self.chunks.values()},
                   "stats": {k: self.stats[k] for k in ("hits", "misses", "lookups",
                                                         "tokens_restored", "bytes_read", "jobs",
                                                         "writes_chunk", "writes_anchor",
                                                         "bytes_written")}}
        p = os.path.join(self.root, "index.json")
        try:
            with open(p + ".tmp", "w") as f:
                json.dump(doc, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.chmod(p + ".tmp", 0o600)
            os.replace(p + ".tmp", p)
            self._saved, self._dirty = now, False
        except OSError:
            pass

    # --- events ------------------------------------------------------------------------------
    def _event(self, kind: str, **detail) -> None:
        ev = {"kind": kind, "ts": round(self.clock(), 3), **detail}
        with self.lock:
            self._ev_n += 1
            self.events[self._ev_n] = ev
            while len(self.events) > 50:
                self.events.popitem(last=False)
        if self.on_event is not None:
            try:
                self.on_event(kind, detail)
            except Exception:                                      # noqa: BLE001
                pass

    # --- the staging buffer ------------------------------------------------------------------
    @staticmethod
    def _anchor_layout(eng) -> tuple[list, int]:
        from engine.cache import state_now
        return _sections([_nb(t) for t in state_now(eng)])

    def _chunk_layout(self, views) -> tuple[list, int]:
        return _sections([_row_bytes(t, ax) * self.chunk for t, ax in views])

    def plan_staging(self, eng, drafter=None) -> int:
        """The staging buffer's bytes before it exists, for the residency planner: two anchor
        slots and two chunk slots at the per-token figure the state store uses."""
        from engine.cache import _drafter_bytes_per_token, _snapshottable, model_views
        per = sum(_row_bytes(t, ax) for t, ax in model_views(eng))
        if drafter is not None and (_snapshottable(drafter) or hasattr(drafter, "kv_views")):
            per += _drafter_bytes_per_token(drafter)
        self._est = (align(self._anchor_layout(eng)[1]), align(per * self.chunk + 16 * ALIGN))
        self.staging_planned = 2 * self._est[0] + 2 * self._est[1]
        return self.staging_planned

    @property
    def staging_bytes(self) -> int:
        return sum(b.nbytes for b in self.buf_anchor + self.buf_chunk)

    def staging_pending(self) -> int:
        """What the planner must still count: the planned buffer while it is not allocated (once
        it is, the board's free memory already shows it)."""
        if self.buf_anchor and self.buf_chunk:
            return 0
        return max(0, self.staging_planned - self.staging_bytes)

    def _ensure_anchor_slots(self, eng) -> None:
        need = self._anchor_layout(eng)[1]
        with self._slots:
            if self.buf_anchor and self.buf_anchor[0].nbytes >= need:
                return
            if self.buf_anchor and len(self._free_anchor) < len(self.buf_anchor):
                return                         # a slot is in use; the next capture retries
            from engine.cache import state_now
            cuda = any(t.is_cuda for t in state_now(eng))
            self.buf_anchor = [_Buf(need, cuda) for _ in range(2)]
            self._free_anchor = [0, 1]

    def _ensure_chunk_slots(self, views) -> None:
        need = self._chunk_layout(views)[1]
        if self.buf_chunk and self.buf_chunk[0].nbytes >= need:
            return
        cuda = any(t.is_cuda for t, _ in views)
        self.buf_chunk = [_Buf(need, cuda) for _ in range(2)]
        if cuda and self._side is None:
            import torch
            self._side = torch.cuda.Stream()

    def _take_anchor_slot(self, wait: float = 0.0) -> int | None:
        with self._slots:
            if not self._free_anchor and wait > 0:
                self._slots.wait_for(lambda: bool(self._free_anchor), timeout=wait)
            return self._free_anchor.pop() if self._free_anchor else None

    def _give_anchor_slot(self, i: int) -> None:
        with self._slots:
            self._free_anchor.append(i)
            self._slots.notify_all()

    # --- what goes to disk (the engine thread, before the prefill) -----------------------------
    def plan(self, keys, hashes, start: int, n: int, lim: int, kind: str | None,
             session: str | None = None, tag: dict | None = None) -> _Job | None:
        """Count the grid lines this prefill forwards cold and decide what it writes: an anchor
        at the longest line prefilled `admit_after` times, and, for the first long request of a
        session, anchors at its last `head_anchors` lines; the chunks below them that are not on
        disk. None when nothing is admitted, and for a client's side request (`tag["side"]`, S1),
        which is not counted either."""
        if not self.on or kind == "session" or self.chunk <= 0 or (tag or {}).get("side"):
            return None
        C = self.chunk
        lines = [L for L in range((start // C + 1) * C, lim, C) if L >= self.min_tokens]
        if not lines:
            return None
        if self.guard is not None and not self.guard.allow("tier", kind or "prefix", n):
            self.stats["dropped"]["write_budget"] = self.stats["dropped"].get("write_budget", 0) + 1
            return None
        now = self.clock()
        sh = hashlib.sha256(session.encode()).hexdigest()[:16] if session else None
        with self.lock:
            for L in lines:
                k = (L, hashes[L])
                self.counts[k] = self.counts.get(k, 0) + 1
                self.counts.move_to_end(k)
            while len(self.counts) > COUNT_MAX:
                self.counts.popitem(last=False)
            n_of = {L: self.counts[(L, hashes[L])] for L in lines}
            rep = [L for L in lines if n_of[L] >= int(self.cfg["admit_after"])]
            # where an earlier prompt parted from this one (the lines are consecutive), in the
            # head two sessions can share: the end of an earlier cold read of this conversation
            # past it is not a line another session reads
            fa = (tag or {}).get("first_answer")
            branch = [L for L, nx in zip(lines, lines[1:])
                      if n_of[L] >= int(self.cfg["admit_after"]) and n_of[L] > n_of[nx]
                      and (fa is None or L <= int(fa))]
            first = bool(sh) and n >= self.min_tokens and sh not in self.sessions
            if sh and n >= self.min_tokens:
                self.sessions[sh] = now
                self.sessions.move_to_end(sh)
                while len(self.sessions) > SESSIONS_MAX:
                    self.sessions.popitem(last=False)
            self._dirty = True
            want: dict[int, str] = {}
            if rep:
                want[max(rep)] = "repeated"
            for L in branch[-BRANCH_MAX:]:
                want.setdefault(L, "branch")
            if first and self.head_anchors:
                # where the next session parts from this one: where its first user message
                # starts (`head_end`, the server's), else near the end of the prompt
                he = (tag or {}).get("head_end")
                below = [L for L in lines if he is None or L <= int(he)]
                for L in below[-self.head_anchors:]:
                    want.setdefault(L, "head")
            # an anchor counts once every chunk below it is on disk too: one whose chunks were
            # never written (a writer that failed part way, a window that closed) is admitted
            # again, and the chunks it lacks are written this time
            want = {L: w for L, w in want.items()
                    if (L, hashes[L]) not in self.anchors
                    or any((b, hashes[b]) not in self.chunks for b in range(C, L + 1, C))}
            if not want:
                return None
            top = max(want)
            pending = {k for j in self._open_jobs for k in ((b, j.hashes[b]) for b in j.chunks)}
            chunks = [b for b in range(C, top + 1, C)
                      if (b, hashes[b]) not in self.chunks and (b, hashes[b]) not in pending]
        est = self._estimate(len(chunks), len(want))
        if est > self.budget // 2:
            self._drop("too_big", tokens=top, bytes=est)
            return None
        client = (tag or {}).get("client")
        spans = (tag or {}).get("spans") or []
        lines_meta = {}
        for L, why in sorted(want.items()):
            done = [s for s in spans if s[1] <= L]
            pk = sorted({nm for s in done if "packages" in s[2] for nm in s[3]})
            if pk:
                kind_ = "package"
            elif any("memory" in s[2] for s in done):
                kind_ = "memory"
            elif first and client == "opencode":
                kind_ = "opencode-head"
            elif first:
                kind_ = "head"
            else:
                kind_ = "prefix"
            prev = None
            if kind_ in ("opencode-head", "package") and self.preview is not None:
                try:
                    prev = self.preview(keys, L, kind_, done)
                except Exception:                                  # noqa: BLE001
                    prev = None
            lines_meta[L] = {"kind": kind_, "why": why, "label": ", ".join(pk) or None,
                             "preview": prev}
            self.stats["admitted"][kind_] = self.stats["admitted"].get(kind_, 0) + 1
        job = _Job(keys, hashes, lines_meta, chunks, sh, now, C)
        with self._fence:
            self._open_jobs.append(job)
        self.stats["jobs"] += 1
        return job

    def _estimate(self, n_chunks: int, n_anchors: int) -> int:
        """The bytes a job writes: the slots' sizes once they exist, the planned ones before."""
        if self.buf_chunk and self.buf_anchor:
            return n_chunks * self.buf_chunk[0].nbytes + n_anchors * self.buf_anchor[0].nbytes
        anchor, chunk = self._est
        return n_chunks * chunk + n_anchors * anchor

    def capture(self, job: _Job, eng, L: int) -> None:
        """The prefill has just written line L: the state's sections to an anchor slot, for the
        writer. A slot still being written means this anchor is not taken (journaled)."""
        if job is None or L not in job.lines:
            return
        self._ensure_anchor_slots(eng)
        i = self._take_anchor_slot()
        if i is None:
            job.drop("staging")
            return
        secs = eng.cache_state().sections()
        buf = self.buf_anchor[i]
        for (o, _), t in zip(self._anchor_layout(eng)[0], secs.values()):
            buf.typed(o, t.dtype, t.shape).copy_(t)
        self._q.put(("anchor", job, L, i, [(name, str(t.dtype), list(t.shape)) for name, t in secs.items()]))

    def seal(self, job: _Job, eng, drafter, rows: int) -> None:
        """The prefill is over (or stopped at `rows`): the chunks below that line may be copied
        out of the engine's buffers from now on, until a fence closes the window."""
        if job is None:
            return
        from engine.cache import _kv_views
        job.ready = int(rows)
        try:
            job.views = _kv_views(eng, drafter)
        except Exception:                                          # noqa: BLE001
            job.views = None
        if job.views and job.views[0][0].is_cuda:
            import torch
            job.event = torch.cuda.Event()
            job.event.record()
        self._q.put(("chunks", job))

    def fence(self, safe: int) -> None:
        """The next prefill is about to write rows at and above `safe`: an open job copies no row
        above that line any more. Waits at most for one chunk copy in flight."""
        with self._fence:
            for job in self._open_jobs:
                if job.limit > safe:
                    job.limit = int(safe)

    def guard_rows(self, n: int) -> int:
        """The side lane (server/preempt.py): a short request is about to write the
        engine's rows below `n` in the middle of another generation, and give them back after. An
        open job that has not copied its chunks below `n` yet would copy the guest's rows: it stops
        where it is (what it copied is written). Returns how many jobs stopped; waits at most for
        one chunk copy in flight."""
        stopped = 0
        with self._fence:
            for job in self._open_jobs:
                if any(job.copied < b <= job.limit and b - self.chunk < int(n)
                       for b in job.chunks):
                    job.limit = job.copied
                    stopped += 1
        return stopped

    # --- the writer --------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=SWEEP_S)
            except queue.Empty:
                self._guard(self._sweep, "timer")
                self._guard(self._save_index)
                continue
            try:
                if item is None:
                    return
                op = item[0]
                if op == "anchor":
                    self._guard(self._write_anchor, *item[1:])
                elif op == "chunks":
                    self._guard(self._write_chunks, item[1])
                elif op == "sweep":
                    self._guard(self._sweep, "settings")
                elif op == "clear":
                    self._guard(self._clear_now)
                    item[1].set()
                elif op == "flush":
                    self._guard(self._save_index, True)
                    item[1].set()
            finally:
                self._q.task_done()

    def _guard(self, fn, *a) -> None:
        try:
            fn(*a)
        except Exception as exc:                                   # noqa: BLE001
            self._event("error", error=f"{type(exc).__name__}: {exc}"[:300])

    def flush(self, timeout: float = 60.0) -> bool:
        """Wait until everything queued so far is written (tests, drain)."""
        if self._thread is None:
            return True
        ev = threading.Event()
        self._q.put(("flush", ev))
        return ev.wait(timeout)

    def close(self) -> None:
        if self._thread is not None:
            self.fence(0)
            self._q.put(None)
            self._thread.join(timeout=30)
            self._thread = None
        self._save_index(force=True)
        if self.guard is not None:
            self.guard.save()

    def _drop(self, why: str, **detail) -> None:
        self.stats["dropped"][why] = self.stats["dropped"].get(why, 0) + 1
        self._event("drop", why=why, **detail)

    def _room(self, nbytes: int, protect: _Job | None) -> str | None:
        """Make room for `nbytes` under the budget and above the floor, least valuable first.
        None when it fits; else why not."""
        while True:
            over_budget = self.bytes + nbytes > self.budget
            low = self._free(self.root) - nbytes < self.floor
            if not over_budget and not low:
                return None
            victim = self._victim(protect)
            if victim is None:
                return "budget" if over_budget else "floor"
            self._evict(victim, "budget" if over_budget else "floor", protect)

    def _write_anchor(self, job: _Job, L: int, slot: int, metas) -> None:
        try:
            key = (L, job.hashes[L])
            if key in self.anchors:
                return
            offs = self._slot_secs(metas)
            C = self.chunk
            chain = [job.hashes[b] for b in range(C, L + 1, C)]
            cb = self._chain_buf(chain)
            payload = align(offs[-1][0] + offs[-1][1]) if offs else 0
            why = self._room(self._hdr_size("anchor") + payload + len(cb), job)
            if why is not None:
                job.drop(why)
                self._event("drop", why=why, tokens=L, what="anchor")
                return
            buf = self.buf_anchor[slot]
            secs = [{"name": name, "dtype": dt, "shape": shape, "off": o, "bytes": nb, "crc": checksum(buf.view(o, nb))}
                    for (name, dt, shape), (o, nb) in zip(metas, offs)]
            secs.append({"name": "chain", "dtype": "uint64", "shape": [len(chain)],
                         "off": payload, "bytes": 8 * len(chain), "crc": checksum(cb)})
            line = job.lines[L]
            meta = {"format": ANCHOR_FORMAT, "kind": "anchor", "L": L, "chunk": C, "h": _hx(job.hashes[L]),
                    "fp": self.fp, "created": round(self.clock(), 3), "sections": secs,
                    "entry": line["kind"], "why": line["why"], "label": line["label"],
                    "preview": line["preview"]}
            hdr = self._header(meta, "anchor")
            p = self._path("anchor", L, job.hashes[L])
            t0 = time.perf_counter()
            direct = write_file(p, [hdr, buf.view(0, payload), cb])
            nbytes = len(hdr) + payload + len(cb)
            self._wrote(time.perf_counter() - t0, nbytes, direct, "anchor")
            meta["chain"] = chain
            self._index_add("anchor", meta, p)
            job.written["anchors"] += 1
            job.written["bytes"] += nbytes
        finally:
            self._give_anchor_slot(slot)

    @staticmethod
    def _slot_secs(metas):
        """(offset, bytes) of each section (name, dtype, shape) laid out in an anchor slot."""
        import torch
        return _sections([torch.empty((), dtype=getattr(torch, dt.split(".")[-1])).element_size() * _prod(shape)
                          for _, dt, shape in metas])[0]

    def _wrote(self, s: float, nbytes: int, direct: bool, kind: str) -> None:
        self.stats[f"writes_{kind}"] += 1
        self.stats["bytes_written"] += nbytes
        self.stats["write_ms"] += s * 1e3
        self.stats["direct"] = bool(direct)
        if self.guard is not None:
            self.guard.add(nbytes, "tier")
        if s > 0 and nbytes >= (1 << 24):
            self.stats["last_write_gbs"] = round(nbytes / s / GB, 3)
        self._dirty = True

    def _write_chunks(self, job: _Job) -> None:
        import torch
        C = self.chunk
        try:
            if not job.views:
                if job.chunks:
                    job.drop("no_views", len(job.chunks))
                return
            secs, payload = self._chunk_layout(job.views)
            todo = list(job.chunks)
            for j, b in enumerate(todo):
                if b > job.ready:
                    job.drop("aborted", len(todo) - j)
                    break
                key = (b, job.hashes[b])
                if key in self.chunks:
                    job.copied = b
                    continue
                why = self._room(self._hdr_size("chunk") + payload, job)
                if why is not None:
                    job.drop(why, len(todo) - j)
                    self._event("drop", why=why, tokens=b, what="chunk")
                    break
                # A session spill writing now (an eviction the request waits for)
                # has the disk to itself; this file waits, at most SPILL_WAIT_S
                t_w = time.monotonic()
                while self.spill_active and time.monotonic() - t_w < SPILL_WAIT_S:
                    time.sleep(0.005)
                with self._io:
                    # every CUDA call of this thread waits for a graph capture on the engine
                    # thread to end, and holds the next one off until it is done (2026-10-02 21:09:
                    # a copy inside the drafter's capture failed both; engine/capture_lock.py)
                    with capture_lock.LOCK:
                        self._ensure_chunk_slots(job.views)
                        buf = self.buf_chunk[0]
                        with self._fence:
                            if b > job.limit:
                                job.drop("window", len(todo) - j)
                                break
                            for (t, ax), (off, nb) in zip(job.views, secs):
                                src = t.narrow(ax, b - C, C)
                                dst = buf.typed(off, t.dtype, _row_shape(t, ax, C))
                                if t.is_cuda:
                                    with torch.cuda.stream(self._side):
                                        if job.event is not None:
                                            self._side.wait_event(job.event)
                                        dst.copy_(src, non_blocking=buf.pinned)
                                    self._side.synchronize()
                                else:
                                    dst.copy_(src)
                            job.copied = b
                    meta = {"format": FORMAT, "kind": "chunk", "b": b, "chunk": C,
                            "h": _hx(job.hashes[b]), "fp": self.fp,
                            "created": round(self.clock(), 3),
                            "sections": [{"dtype": str(t.dtype), "shape": _row_shape(t, ax, C),
                                          "ax": ax, "off": off, "bytes": nb,
                                          "crc": checksum(buf.view(off, nb))}
                                         for (t, ax), (off, nb) in zip(job.views, secs)]}
                    hdr = self._header(meta, "chunk", job.keys[b - C:b])
                    p = self._path("chunk", b, job.hashes[b])
                    t0 = time.perf_counter()
                    direct = write_file(p, [hdr, buf.view(0, payload)])
                    self._wrote(time.perf_counter() - t0, len(hdr) + payload, direct, "chunk")
                self._index_add("chunk", meta, p)
                job.written["chunks"] += 1
                job.written["bytes"] += len(hdr) + payload
        finally:
            with self._fence:
                if job in self._open_jobs:
                    self._open_jobs.remove(job)
            job.views = None
            for why, k in job.dropped.items():
                self.stats["dropped"][why] = self.stats["dropped"].get(why, 0) + k
            top = max(job.lines)
            self._event("write", tokens=top, anchors=job.written["anchors"],
                        chunks=job.written["chunks"], bytes=job.written["bytes"],
                        kinds=sorted({m["kind"] for m in job.lines.values()}),
                        dropped=dict(job.dropped) or None)
            job.done.set()
            self._sweep("job")

    # --- eviction ----------------------------------------------------------------------------
    def _refs(self) -> dict:
        C = self.chunk
        refs: dict[tuple, int] = {}
        for a in self.anchors.values():
            for j, hb in enumerate(a["chain"]):
                k = ((j + 1) * C, hb)
                refs[k] = refs.get(k, 0) + 1
        return refs

    def _value(self, key: tuple, a: dict, refs: dict, now: float) -> float:
        """Hits plus one, times the tokens the anchor saves per byte it holds alone (its file and
        its share of every chunk it reads), halved for every day unused. An anchor whose chain
        is not whole saves nothing."""
        C = self.chunk
        own = a["bytes"]
        for j, hb in enumerate(a["chain"]):
            c = self.chunks.get(((j + 1) * C, hb))
            if c is None:
                return 0.0
            own += c["bytes"] / max(1, refs.get(((j + 1) * C, hb), 1))
        age = max(0.0, now - a["used"]) / 86400.0
        return (a["hits"] + 1) * a["saves"] / max(1, own) * 0.5 ** age

    def _victim(self, protect: _Job | None):
        now = self.clock()
        with self.lock:
            refs = self._refs()
            keep = set()
            if protect is not None:
                keep = {(L, protect.hashes[L]) for L in protect.lines}
            best, victim = None, None
            for key, a in self.anchors.items():
                if key in keep:
                    continue
                v = self._value(key, a, refs, now)
                if best is None or v < best:
                    best, victim = v, key
            if victim is None:
                # no anchor to give: chunks nothing reads
                for ck in self.chunks:
                    if ck not in refs and not self._pending(ck):
                        return ("chunk", ck)
                return None
            return ("anchor", victim)

    def _pending(self, ck: tuple) -> bool:
        return any(ck[0] in j.chunks and j.hashes.get(ck[0]) == ck[1] for j in self._open_jobs)

    def _evict(self, victim: tuple, why: str, protect: _Job | None = None) -> None:
        kind, key = victim
        freed = self._index_drop(kind, key)
        n_chunks = 0
        with self.lock:
            refs = self._refs()
            orphans = [ck for ck in self.chunks if ck not in refs and not self._pending(ck)]
        for ck in orphans:
            freed += self._index_drop("chunk", ck)
            n_chunks += 1
        self.stats["evicted"][why] = self.stats["evicted"].get(why, 0) + 1
        self._dirty = True
        self._event("evict", why=why, tokens=key[0] if kind == "anchor" else None,
                    chunks=n_chunks, bytes=freed)

    def _sweep(self, why: str) -> None:
        """Past the TTL first, then over the budget or under the floor, least valuable first;
        chunks no anchor reads go with the last anchor that read them."""
        now = self.clock()
        with self.lock:
            stale = [k for k, a in self.anchors.items() if now - a["used"] > self.ttl]
        for k in stale:
            self._evict(("anchor", k), "ttl")
        with self.lock:
            refs = self._refs()
            loose = [ck for ck, c in self.chunks.items() if ck not in refs and not self._pending(ck)
                     and now - c["created"] > 60]
        for ck in loose:
            self._index_drop("chunk", ck)
            self.stats["evicted"]["orphan"] = self.stats["evicted"].get("orphan", 0) + 1
        while True:
            over = self.bytes > self.budget
            low = self.bytes > 0 and self._free(self.root) < self.floor
            if not (over or low):
                break
            victim = self._victim(None)
            if victim is None:
                break
            self._evict(victim, "budget" if over else "floor")
        self._save_index()

    def clear(self, timeout: float = 60.0) -> dict:
        """Every entry and every count gone (the dashboard's button): no live copy is read from
        here on, and the writer deletes the files between two of its writes."""
        self.fence(0)
        before = {"entries": len(self.anchors), "chunks": len(self.chunks), "bytes": self.bytes}
        if self._thread is None:
            self._clear_now()
        else:
            ev = threading.Event()
            self._q.put(("clear", ev))
            ev.wait(timeout)
        return before

    def _clear_now(self) -> None:
        with self.lock:
            anchors, chunks = list(self.anchors), list(self.chunks)
        freed = sum(self._index_drop("anchor", k) for k in anchors)
        freed += sum(self._index_drop("chunk", k) for k in chunks)
        with self.lock:
            self.counts.clear()
            self.sessions.clear()
            self.stats["cleared"] += 1
            self._dirty = True
        self._event("clear", entries=len(anchors), chunks=len(chunks), bytes=freed)
        self._save_index(force=True)

    # --- reading (the engine thread, at a request's start) -------------------------------------
    def find(self, keys, hashes, max_len: int, min_len: int, have: int = 0) -> Hit | None:
        """The longest anchor on disk for this prompt that starts later than `min_len` (what the
        caches in RAM offer) by more than its read costs, with every chunk below it on disk and
        every token of those chunks equal to the prompt's. `have`: the rows in place already
        hold the prompt's first `have` tokens, so those chunks are not read."""
        C = self.chunk
        if not self.on or max_len < self.min_tokens or max_len - min_len < C:
            return None
        self.stats["lookups"] += 1
        with self.lock:
            cands = sorted((L for L in self._alen if min_len < L <= max_len), reverse=True)
        for L in cands:
            with self.lock:
                a = self.anchors.get((L, hashes[L]))
                if a is None:
                    continue
                chain = [(b, hashes[b]) for b in range(C, L + 1, C)]
                recs = [self.chunks.get(ck) for ck in chain]
            if any(r is None for r in recs):
                self.stats["incomplete"] += 1
                continue
            have_f = min(int(have), L) // C * C
            need = a["bytes"] + sum(r["bytes"] for (b, _), r in zip(chain, recs) if b > have_f)
            if need / self.read_bps >= (L - min_len) / self.prefill_tok_s:
                self.stats["declined_slow"] += 1
                continue
            metas, ok = [], True
            try:
                for (b, _), r in zip(chain, recs):
                    meta = self._read_header(r["path"], "chunk")
                    if meta["tokens"] != list(keys[b - C:b]) or meta.get("fp") not in self.fps:
                        ok = False
                        break
                    meta["path"] = r["path"]
                    metas.append(meta)
            except Exception:                                      # noqa: BLE001
                ok = False
            if not ok:
                self.stats["rejected_collisions"] += 1
                continue
            return Hit(L, dict(a, key=(L, hashes[L])), list(zip(chain, metas)), have_f, need)
        self.stats["misses"] += 1
        return None

    def restore(self, eng, drafter, hit: Hit) -> bool:
        """Read the anchor and the chunks above `hit.have` into the engine's own buffers and
        bring the engine to `hit.L`. True: done, the same bytes a cold prefill on the grid
        leaves. False: the anchor or a chunk failed its check (deleted, journaled) or its layout
        is not this engine's; the rows below `hit.have` are as they were, and the rows restored
        before the failure hold the prompt's own bytes, so the caller resumes from the rows in
        place or starts cold."""
        import torch
        from engine.cache import _kv_views, drafter_resume
        C, L = self.chunk, hit.L
        t0 = time.perf_counter()
        drafter_resume(eng, drafter, L)        # builds a draft cache that does not exist yet
        views = _kv_views(eng, drafter)
        secs, payload = self._chunk_layout(views)
        for _, meta in hit.chain:
            got = [(s["dtype"], list(s["shape"]), s["ax"], s["off"], s["bytes"])
                   for s in meta["sections"]]
            want = [(str(t.dtype), _row_shape(t, ax, C), ax, off, nb)
                    for (t, ax), (off, nb) in zip(views, secs)]
            if got != want:
                self._fail("layout", hit, None)
                return False
        state = eng.cache_state().sections()
        cuda = any(t.is_cuda for t in state.values())
        self._ensure_anchor_slots(eng)
        slot = self._take_anchor_slot(wait=2.0)
        if slot is None:
            self._fail("staging", hit, None)
            return False
        nread = 0
        try:
            with self._io:
                self._ensure_chunk_slots(views)
                abuf = self.buf_anchor[slot]
                ap = hit.anchor["path"]
                try:
                    am = self._read_header(ap, "anchor")
                    offs = self._anchor_layout(eng)[0]
                    alias = FORMAT1_SECTIONS if am.get("format") == 1 else {}
                    sa = [s for s in am["sections"] if s["name"] != "chain"]
                    if ([(alias.get(s["name"], s["name"]), s["dtype"], s["shape"], s["off"]) for s in sa]
                            != [(name, str(t.dtype), list(t.shape), o)
                                for (name, t), (o, _) in zip(state.items(), offs)]):
                        self._fail("layout", hit, None)
                        return False
                    nread += read_into(ap, am["_hdr"],
                                       abuf.view(0, align(offs[-1][0] + offs[-1][1]) if offs else 0))
                    if any(checksum(abuf.view(o, nb)) != s["crc"] for (o, nb), s in zip(offs, sa)):
                        raise ValueError("checksum")
                except Exception:                                  # noqa: BLE001
                    self._fail("corrupt", hit, ("anchor", hit.anchor["key"]))
                    return False
                todo = [(ck, m) for ck, m in hit.chain if ck[0] > hit.have]
                pool = self._reader()
                futs = {}

                def submit(j):
                    if j < len(todo):
                        futs[j] = pool.submit(read_into, todo[j][1]["path"], todo[j][1]["_hdr"],
                                              self.buf_chunk[j % 2].view(0, payload))
                submit(0)
                submit(1)
                for j, (ck, meta) in enumerate(todo):
                    buf = self.buf_chunk[j % 2]
                    try:
                        nread += futs.pop(j).result()
                        bad = any(checksum(buf.view(s["off"], s["bytes"])) != s["crc"]
                                  for s in meta["sections"])
                    except Exception:                              # noqa: BLE001
                        bad = True
                    if bad:
                        for f in futs.values():
                            f.cancel()
                            try:
                                f.result()
                            except Exception:                      # noqa: BLE001
                                pass
                        self._fail("corrupt", hit, ("chunk", ck))
                        return False
                    b = ck[0]
                    for (t, ax), (off, nb) in zip(views, secs):
                        _kvp.ensure_for(t, b)                  # SRV-307
                        t.narrow(ax, b - C, C).copy_(buf.typed(off, t.dtype, _row_shape(t, ax, C)),
                                                     non_blocking=cuda and buf.pinned)
                    if cuda:
                        torch.cuda.current_stream().synchronize()
                    submit(j + 2)
                nb = cuda and abuf.pinned
                for (o, _), t in zip(offs, state.values()):
                    t.copy_(abuf.typed(o, t.dtype, t.shape), non_blocking=nb)
                if cuda:
                    torch.cuda.current_stream().synchronize()
        finally:
            self._give_anchor_slot(slot)
        eng.resume_at(L)
        if drafter is not None and hasattr(drafter, "state_resume"):
            drafter.state_resume(L)            # again, over the rows read back (`drafter_resume`)
        s = time.perf_counter() - t0
        now = self.clock()
        with self.lock:
            a = self.anchors.get(hit.anchor["key"])
            if a is not None:
                a["hits"] += 1
                a["used"] = now
            for ck, _ in hit.chain:
                c = self.chunks.get(ck)
                if c is not None:
                    c["used"] = now
            self.stats["hits"] += 1
            self.stats["tokens_restored"] += L - hit.have
            self.stats["bytes_read"] += nread
            self.stats["read_ms"] += s * 1e3
            self.stats["last_restore_ms"] = round(s * 1e3, 1)
            if s > 0 and nread >= (1 << 24):
                gbs = nread / s
                self.stats["last_read_gbs"] = round(gbs / GB, 3)
                self.read_bps = 0.5 * self.read_bps + 0.5 * gbs
            self._dirty = True
        self._event("hit", tokens=L, read=L - hit.have, bytes=nread, ms=round(s * 1e3, 1),
                    entry=hit.anchor.get("kind"))
        return True

    def _reader(self):
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="disk-cache-read")
        return self._pool

    def _fail(self, why: str, hit: Hit, bad) -> None:
        """A restore that did not happen: the bad file goes (a chunk takes every longer chunk of
        its chain and the anchor with it); journaled."""
        self.stats["failed_restores"] += 1
        if why == "corrupt":
            self.stats["corrupt"] += 1
        if bad is not None:
            kind, key = bad
            self._index_drop(kind, key)
            if kind == "chunk":
                for ck, _ in hit.chain:
                    if ck[0] > key[0]:
                        self._index_drop("chunk", ck)
                self._index_drop("anchor", hit.anchor["key"])
        self._event("restore_failed", why=why, tokens=hit.L)

    # --- what the dashboard and /metrics read ----------------------------------------------------
    def report(self, top: int = 20, previews: bool = False) -> dict:
        now = self.clock()
        with self.lock:
            refs = self._refs()
            C = self.chunk
            rows = []
            for key, a in self.anchors.items():
                chain = [self.chunks.get(((j + 1) * C, hb)) for j, hb in enumerate(a["chain"])]
                whole = all(c is not None for c in chain)
                rows.append({"tokens": key[0], "kind": a["kind"], "label": a.get("label"),
                             "hits": a["hits"], "used": round(a["used"], 3),
                             "created": round(a["created"], 3), "complete": whole,
                             "reads_bytes": a["bytes"] + sum(c["bytes"] for c in chain if c),
                             "value": round(self._value(key, a, refs, now), 9),
                             "preview": (a.get("preview") if previews
                                         and a["kind"] in ("opencode-head", "package") else None)})
            rows.sort(key=lambda r: (-r["hits"], -r["used"]))
            look = self.stats["lookups"]
            out = {"on": self.on, "root": self.root, "settings": dict(self.cfg),
                   "chunk": C, "fingerprint": self.fp[:16], "entries": len(self.anchors),
                   "chunks": len(self.chunks), "bytes": self.bytes, "budget": self.budget,
                   "floor": self.floor, "free_bytes": self._free(self.root),
                   "staging_bytes": self.staging_bytes, "staging_planned": self.staging_planned,
                   "head_anchors": self.head_anchors, "min_tokens": self.min_tokens,
                   "hit_rate": round(self.stats["hits"] / look, 4) if look else None,
                   "read_bps": round(self.read_bps), "counted_lines": len(self.counts),
                   "sessions_seen": len(self.sessions), "open_jobs": len(self._open_jobs),
                   "top": rows[:top],
                   "by_kind": _by_kind(rows),
                   "events": list(self.events.values())[-20:],
                   **{k: (dict(v) if isinstance(v, dict) else v) for k, v in self.stats.items()}}
        return out


def _by_kind(rows) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        k = out.setdefault(r["kind"], {"entries": 0, "hits": 0})
        k["entries"] += 1
        k["hits"] += r["hits"]
    return out


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass



# --------------------------------------------------------------------------- the write budget
#: what the engine may write to the NVMe in one local day before optional writes stop (GB)
WRITE_BUDGET_GB = 200.0
#: an eviction save of a conversation this long still goes past the budget (re-reading it costs
#: more than a minute of prefill); under the hard line only a planned stop's drain goes
GUARD_BIG_TOKENS = 32768
GUARD_HARD = 1.5


class WriteGuard:
    """The engine's NVMe writes per local day, and the line past which optional writes stop.

    WHY (2026-10-07, the owner: "take care that we don't kill the SSD"). The box's 1 TB TLC drive
    (SMART: 11.6 TB written in 1,534 hours, 1 % used) took 137 to 282 GB a day from the disk tier
    and the session spill alone, most of it on days of many restarts and test chats. Every write
    here is a cache: skipping one costs a later prefill, never an answer. So past `budget_gb` a
    day the tier writes nothing and the spill writes only a planned stop's drain and evictions of
    conversations of at least `GUARD_BIG_TOKENS`; past `GUARD_HARD` times the budget only the
    drain. Each skip is counted and logged. The day's count survives a restart (`writes.json`
    beside the tier, written at most once a minute and at close)."""

    def __init__(self, root: str, budget_gb: float = WRITE_BUDGET_GB, clock=time.time,
                 on_event=None):
        self.path = os.path.join(os.path.realpath(os.path.expanduser(root)), "writes.json")
        self.budget_gb = float(budget_gb)
        self.clock = clock
        self.on_event = on_event
        self.lock = threading.Lock()
        self.days: dict[str, dict] = {}
        self._saved = 0.0
        try:
            with open(self.path) as f:
                self.days = {str(k): dict(v) for k, v in (json.load(f).get("days") or {}).items()}
        except (OSError, ValueError, AttributeError):
            self.days = {}

    def _day(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(self.clock()))

    def _row(self) -> dict:
        d = self.days.setdefault(self._day(), {"bytes": 0, "tier": 0, "spill": 0, "skipped": {}})
        for k in [k for k in sorted(self.days)[:-14]]:
            self.days.pop(k, None)
        return d

    @property
    def budget(self) -> int:
        return int(self.budget_gb * GB)

    def today(self) -> int:
        with self.lock:
            return int(self._row()["bytes"])

    def add(self, nbytes: int, who: str) -> None:
        if nbytes <= 0:
            return
        with self.lock:
            r = self._row()
            r["bytes"] += int(nbytes)
            r[who] = r.get(who, 0) + int(nbytes)
            due = self.clock() - self._saved > 60.0
        if due:
            self.save()

    def allow(self, who: str, reason: str = "", tokens: int = 0) -> bool:
        """Whether an optional write of `who` (tier, spill) for `reason` may go now."""
        if self.budget_gb <= 0:
            return True
        with self.lock:
            used = self._row()["bytes"]
        if used < self.budget:
            return True
        hard = used >= GUARD_HARD * self.budget
        ok = (who == "spill" and (reason == "drain" or
                                  (not hard and reason in ("evict", "switch")
                                   and tokens >= GUARD_BIG_TOKENS)))
        if not ok:
            key = f"{who}:{reason or '-'}"
            with self.lock:
                sk = self._row()["skipped"]
                sk[key] = sk.get(key, 0) + 1
            if self.on_event is not None:
                try:
                    self.on_event("skip", {"who": who, "reason": reason, "tokens": tokens,
                                           "today_gb": round(used / GB, 1),
                                           "budget_gb": self.budget_gb, "hard": hard})
                except Exception:                                  # noqa: BLE001
                    pass
        return ok

    def save(self) -> None:
        with self.lock:
            doc = {"days": self.days}
            self._saved = self.clock()
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(doc, f, separators=(",", ":"))
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def report(self) -> dict:
        with self.lock:
            r = dict(self._row())
            days = {k: round(v.get("bytes", 0) / GB, 2) for k, v in sorted(self.days.items())}
        past = [v for k, v in days.items() if k != self._day()]
        return {"budget_gb": self.budget_gb, "today_gb": round(r["bytes"] / GB, 2),
                "today_tier_gb": round(r.get("tier", 0) / GB, 2),
                "today_spill_gb": round(r.get("spill", 0) / GB, 2),
                "skipped_today": dict(r.get("skipped") or {}),
                "level": ("hard" if self.budget_gb > 0 and r["bytes"] >= GUARD_HARD * self.budget
                          else "over" if self.budget_gb > 0 and r["bytes"] >= self.budget
                          else "ok"),
                "per_day_gb": days,
                "mean_gb_per_day": round(sum(past) / len(past), 1) if past else None}


def _statvfs_free(path: str) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


#: modules under `engine/` that only another model's package imports, by path under `engine/`: v2
#: leaves them out as it leaves out `engine/models/`. ENG-341: NemotronH's LiLiCorr draft head
#: (SPD-123) moved v2 of the 27B off 0.9.x's, and 0.7.2's under `SAME_ROWS` (9ac3d8a94c8a9199)
_NOT_27B_FILES = ("drafters/lilicorr.py",)


def _engine_files(root: str, skip=()) -> list[str]:
    """The engine's own modules fingerprint v2 reads: every file under `engine/` but the model
    layer's and the other packages, the other models' modules (`_NOT_27B_FILES`) and the file names
    in `skip` (at the top of `engine/`), in a fixed order. `engine/models/` (the model layer, ENG-221)
    is left out, so the 27B's fingerprint is the bytes it always was; so is `engine/methods/` (the
    methods written once for every model, ENG-E8). A model's own fingerprint is v3's, from its
    adapter's manifest (engine/methods/manifest.py, ENG-243); v2 stays as it is, frozen, to adopt the
    files it wrote."""
    out = []
    base = os.path.join(root, "engine")
    for dirpath, dirnames, files in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in ("decision", "generative", "methods",
                                                               "models", "__pycache__"))
        for name in sorted(files):
            rel = os.path.relpath(os.path.join(dirpath, name), base).replace(os.sep, "/")
            if name.endswith(".py") and not (dirpath == base and name in skip) and rel not in _NOT_27B_FILES:
                out.append(os.path.join(dirpath, name))
    return out


def fingerprint_legacy(root: str, extra: dict, skip=()) -> str:
    """The fingerprint as builds up to 0.4.6 computed it: the bytes of every engine module
    (`_engine_files`) and `extra`. Kept to adopt their files once (`equivalents`); never written."""
    h = hashlib.sha256()
    for p in _engine_files(root, skip):
        h.update(os.path.relpath(p, root).encode())
        with open(p, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()


#: the knob registry: its bytes are documentation; what it can change in a row is the defaults of
#: the knobs the arithmetic reads (`_arithmetic_knobs`)
SETTINGS_FILE = os.path.join("engine", "settings.py")

_TOOLS_IMPORT = re.compile(r"^\s*(?:from\s+tools\s+import\s+([\w\s,()]+)|from\s+tools\.(\w+)\s+"
                           r"import|import\s+tools\.(\w+))", re.M)


#: knob owners whose defaults cannot change the 27B's rows: the server's (no code under `server/`
#: is in the fingerprint), the model layer's (`engine/models/`, which `_engine_files` leaves out) and
#: Flash-Next's kernels (`tools/fn_*`, imported only from `engine/models/flashnext/`). Part of the
#: frozen v2 formula: without it v2 of this tree would not be the served tree's, and v3 could not
#: adopt its files; v3 counts only the knobs its manifest's files own
_NOT_27B_KNOBS = ("server/", "engine/models/", "tools/fn_")


def _arithmetic_knobs(path: str) -> bytes | None:
    """The knobs of `engine/settings.py` that the server does not own, as (name, default) in name
    order: the defaults the engine's and the kernels' modules read. A knob the server reads (the
    regenerate rule, a decider's queue, a package's address) cannot change a prefill's rows, as no
    code under `server/` is in the fingerprint; nor can a knob of another model's package
    (`_NOT_27B_KNOBS`). None when the file has no literal `KNOBS` table (the caller hashes the
    bytes then)."""
    import ast
    try:
        with open(path, "rb") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if isinstance(target, ast.Name) and target.id == "KNOBS" and node.value is not None:
            try:
                knobs = ast.literal_eval(node.value)
            except ValueError:
                return None
            rows = sorted((str(k), v[0]) for k, v in knobs.items()
                          if not str(v[1] if len(v) > 1 else "").startswith(_NOT_27B_KNOBS))
            return json.dumps(rows, default=str).encode()
    return None


def _tools_modules(root: str, files: list[str]) -> list[str]:
    """The `tools/` modules (kernels, quantized linears, the head) the engine's modules import,
    and the ones those import, in name order. They compute rows as much as `engine/` does."""
    seen: set[str] = set()
    todo = list(files)
    while todo:
        p = todo.pop()
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                src = f.read()
        except OSError:
            continue
        for m in _TOOLS_IMPORT.finditer(src):
            names = ([n.strip() for n in m.group(1).replace("(", " ").replace(")", " ").split(",")]
                     if m.group(1) else [m.group(2) or m.group(3)])
            for n in names:
                n = n.split(" as ")[0].strip()
                q = os.path.join(root, "tools", n + ".py")
                if n and n not in seen and os.path.isfile(q):
                    seen.add(n)
                    todo.append(q)
    return [os.path.join(root, "tools", n + ".py") for n in sorted(seen)]


def module_digest(rel: str, data: bytes, same: dict | None = None) -> bytes:
    """A module's sha256 as a fingerprint reads it: the bytes' own, or with `same` (0.8.2,
    engine/methods/manifest.py `SAME_ROWS`) the digest of the bytes they replaced."""
    d = hashlib.sha256(data).digest()
    old = same.get((rel, d.hex())) if same else None
    return bytes.fromhex(old) if old else d


def fingerprint(root: str, extra: dict, skip=(), same: dict | None = None) -> str:
    """Fingerprint v2 (frozen since ENG-243; the tier and the spill use v3, engine/methods/manifest.py,
    and compute this one to adopt its files once): what the rows on disk depend on besides the
    tokens: the engine's code (every file under `engine/` but the model layer and the other
    packages, and the file names in `skip`), the `tools/` modules
    it imports (`_tools_modules`), and `extra` (the weights, the arithmetic flags, the grid, the
    library versions). A file written under another fingerprint is swept. The tier skips nothing;
    the session spill (engine/spill.py) skips the files that keep state rather than compute it.
    `engine/settings.py` counts by its arithmetic knobs' defaults, not its bytes: 0.4.6-test
    registered seven server knobs there, and that alone swept 13 shared heads (12.4 GB) and 13
    spilled conversations (29.6 GB) at its start (FX15, 2026-10-08)."""
    h = hashlib.sha256()
    h.update(b"fp2")
    files = _engine_files(root, skip)
    for p in files + _tools_modules(root, files):
        rel = os.path.relpath(p, root)
        h.update(rel.encode())
        data = _arithmetic_knobs(p) if rel == SETTINGS_FILE else None
        if data is None:
            with open(p, "rb") as f:
                data = f.read()
        h.update(module_digest(rel, data, same))
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()


#: in the tier's directory: fingerprint -> the older ones whose files it reads (`equivalents`)
EQUIV_FILE = "fp-equal.json"


def equivalents(root_dir: str, fp: str, legacy, formula: str | None = None) -> tuple:
    """The fingerprints whose files a tier (and its session spill) under `fp` may read besides
    `fp`'s own. The first start under the current formula adopts the files of the formula before
    it computed on this same tree (`fingerprint_legacy`): same engine bytes, flags, weights and
    grid, so the same rows. The adoption is written to `root_dir/fp-equal.json` and from then on
    that record alone counts: a later build reads those files only under the same `fp`, and a
    `tools/` change (which the old formula did not see) sweeps them as it sweeps any other.

    `formula` (fingerprint v3, ENG-243, engine/methods/manifest.py): the adoption is recorded per
    formula (`formulas` in the record), so a directory that already has a record still adopts once
    more -- `legacy` is then the previous formula's fingerprint of this tree, and what that one
    read comes along. `legacy` may name several (fp4: the extra by path as v2 and as fp3, on this
    folder and the ones a tree moved from; server/diskcache.py `legacy_fingerprints`)."""
    path = os.path.join(root_dir, EQUIV_FILE)
    try:
        with open(path) as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            doc = {}
    except (OSError, ValueError):
        doc = None
    formulas = (doc or {}).get("formulas")
    formulas = formulas if isinstance(formulas, dict) else {}
    if doc is None if formula is None else formula not in formulas:
        doc = {} if doc is None else doc
        # 0.8.2: what `fp` adopted before stays (a new adoption key on an unchanged fingerprint)
        had = doc.get(fp)
        got = {str(x) for x in had} if isinstance(had, list) else set()
        for leg in ([legacy] if isinstance(legacy, str) else list(legacy or ())):
            if leg and leg != fp:
                prev = doc.get(leg) if formula is not None else None
                got |= {leg, *(str(x) for x in (prev if isinstance(prev, list) else []))}
        if got - {fp}:
            doc[fp] = sorted(got - {fp})
        if formula is not None:
            doc["formulas"] = {**formulas, formula: fp}
        try:
            os.makedirs(root_dir, mode=0o700, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(doc, f)
            os.replace(tmp, path)
        except OSError:
            pass
    got = doc.get(fp, [])
    return tuple(str(x) for x in (got if isinstance(got, list) else []) if x != fp)
