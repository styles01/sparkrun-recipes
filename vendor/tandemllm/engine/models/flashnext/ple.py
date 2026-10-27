"""PLE: the hashed n-gram embedding of Flash-Next, and the EXL3-coded table it reads (ENG-145).

For every position the model turns the current token and the one or two before it into sixteen
row indices (8 hash heads for the bigram, 8 for the trigram), reads those sixteen rows of 160
values from one table of about 320 M rows, and concatenates them into a 2,560-wide feature. The
hash is the reference's (`transformers` `Qwen4ExpTextNGramEmbedding`) operation for operation, in
int64 with the same wrap-around, so the row indices are identical (ported from branch `flashnext`).

The table of turboderp's packs (`ngram_embedding.safetensors`, exllamav3's `exl3_ngram_trellis`
format) codes each 160-value row as a tail-biting trellis ring over the mul1 codebook at K bits a
value (ng4..ng6 = K 4..6), with one fp16 scale a row and one bias vector a hash head. A packed row
is 1 + 10 K little-endian uint16 words: word 0 is the fp16 scale's bits, the rest the ring, in
which position i's 16-bit state is the window ending at ring bit (i + 1) K - 1 (mod 160 K). So

    row[i] = mul1(state_i) * scale + head_bias[head(row)]           (fp32, then the compute dtype)

exllamav3's own decoder (`ngram_codec.dequant_rows`) computes the same fp32 value; its CUDA kernel
rounds it to fp16 before use. Here it is rounded once, to the compute dtype.

The table is 32.6 GB in the 3.05 pack and is never in device memory: `EXL3NGramTable` reads the
packed rows a step needs and `decode_rows` turns them into values on any device with fixed-shape
torch ops, so the graph step can take the 16 packed rows of a token as its input. Where the rows come
from is the table's policy (ENG-239, `QWEN38_FN_TABLE`):

  warm   the file mapped and read through the page cache (page faults for a few rows, parallel
         pread for a prompt's thousands), warmed at load; MemAvailable counts that page cache as
         free, so its resident bytes are reported (`resident_bytes`, the kernel's cachestat or
         mincore) and every floor check takes them off (SRV-335)
  cold   `DirectRows`: the 4 KiB blocks holding the rows read with O_DIRECT into an aligned buffer,
         past the page cache, through a set-associative hot-row cache of `QWEN38_FN_TABLE_CACHE_GB`
         with CLOCK eviction; the file's page cache is dropped at open. Packs whose weights and
         table do not fit together (4.05 and up) need it.
"""

from __future__ import annotations

import ctypes
import json
import math
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def layer_multipliers(vocab_size: int, ngram_size: int, ple_index: int, seed: int) -> list[int]:
    multiplier_max = ((1 << 63) - 1) // max(vocab_size, 1)
    half = max(1, multiplier_max // 2)
    base = seed + _PRIME_1 * ple_index
    return [2 * (_splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram_size)]


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def head_sizes(base: int, n_heads: int, ple_index: int) -> tuple[list[int], list[int], int]:
    """The prime vocabulary size and row offset of every hash head, and the total row count."""
    sizes, offsets, total = [], [], 0
    p = base - 1
    # head h takes the (ple_index * n_heads + h + 1)-th prime after base - 1
    for _ in range(ple_index * n_heads):
        p += 1
        while not _is_prime(p):
            p += 1
    for _ in range(n_heads):
        p += 1
        while not _is_prime(p):
            p += 1
        sizes.append(p)
        offsets.append(total)
        total += p
    return sizes, offsets, total


class NGramHash:
    """Row indices for the n-gram heads of one PLE module. Stateless: the caller passes the token
    history (`context` = the last ngram_size-1 tokens before `ids`, EOS-filled at the start)."""

    def __init__(self, vocab_size: int, ngram_size: int, heads_per_ngram: int, eos: int,
                 multipliers: list[int], sizes: list[int], offsets: list[int]):
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.eos = eos
        self.mult = torch.tensor(multipliers, dtype=torch.long)
        self.sizes = torch.tensor(sizes, dtype=torch.long)
        self.offsets = torch.tensor(offsets, dtype=torch.long)

    @classmethod
    def from_config(cls, cfg, ple_index: int = 0, stored: dict | None = None) -> "NGramHash":
        """Derive the parameters from the config; `stored` (the checkpoint's own
        layer_multipliers / ngram_heads_vocab_sizes / ngram_heads_offsets) must agree."""
        mult = layer_multipliers(cfg.vocab_size, cfg.ngram_size, ple_index, cfg.seed)
        sizes, offsets, _ = head_sizes(cfg.ngram_vocab_size_base, cfg.ngram_heads, ple_index)
        if stored:
            for name, mine in (("layer_multipliers", mult), ("ngram_heads_vocab_sizes", sizes),
                               ("ngram_heads_offsets", offsets)):
                if name in stored and stored[name].tolist() != mine:
                    raise ValueError(f"PLE {name} in the checkpoint differs from the config's")
        return cls(cfg.vocab_size, cfg.ngram_size, cfg.heads_per_ngram, cfg.eos_token_id,
                   mult, sizes, offsets)

    @property
    def context_len(self) -> int:
        return self.ngram_size - 1

    def _shift(self, hist: torch.Tensor, shift: int) -> torch.Tensor:
        """The reference's `_shift_right_ignore_eos` on a [B, L] history."""
        if shift == 0:
            return hist
        B, L = hist.shape
        pos = torch.arange(L, device=hist.device, dtype=torch.long)
        eos_pos = torch.where(hist == self.eos, pos, -1)
        prev_incl = torch.cummax(eos_pos, dim=1).values
        prev = torch.cat([eos_pos.new_full((B, 1), -1), prev_incl[:, :-1]], dim=1)
        pos_in_seg = pos.unsqueeze(0) - (prev + 1)
        src = pos - shift
        shifted = hist.gather(1, src.clamp_min(0).unsqueeze(0).expand(B, -1))
        valid = (pos_in_seg >= shift) & (src.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, hist.new_full((), self.eos))

    def ids(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """tokens [B, T], context [B, ngram_size-1] -> row ids [B, T, n_heads] (int64)."""
        tokens = tokens.long()
        hist = torch.cat([context.long(), tokens], dim=-1)
        mult = self.mult.to(hist.device)
        sizes = self.sizes.to(hist.device)
        offs = self.offsets.to(hist.device)
        shifted = [self._shift(hist, s) for s in range(self.ngram_size)]
        blocks = []
        for n in range(2, self.ngram_size + 1):
            a = (n - 2) * self.heads_per_ngram
            b = a + self.heads_per_ngram
            mixed = shifted[0] * mult[0]
            for p in range(1, n):
                mixed = torch.bitwise_xor(mixed, shifted[p] * mult[p])
            ids = torch.remainder(mixed.unsqueeze(-1), sizes[a:b].view(1, 1, -1))
            blocks.append(ids + offs[a:b].view(1, 1, -1))
        return torch.cat(blocks, dim=-1)[:, -tokens.shape[1]:]

    def ids_one(self, token: int, context: list[int]) -> list[int]:
        """`ids` for one token after `context` (the last ngram_size-1 tokens), in Python integers
        with the same int64 wrap-around, without torch: the decode step's host work."""
        hist = list(context) + [int(token)]
        t = len(hist) - 1
        seg = 0                                     # the segment starts after the last EOS before t
        for i in range(t):
            if hist[i] == self.eos:
                seg = i + 1
        shifted = [hist[t - s] if t - seg >= s else self.eos for s in range(self.ngram_size)]
        mult = self.mult.tolist()
        sizes, offs = self.sizes.tolist(), self.offsets.tolist()

        def wrap(v: int) -> int:
            v &= _MASK64
            return v - (1 << 64) if v >> 63 else v
        out = []
        for n in range(2, self.ngram_size + 1):
            mixed = wrap(shifted[0] * mult[0])
            for p in range(1, n):
                mixed ^= wrap(shifted[p] * mult[p])
            a = (n - 2) * self.heads_per_ngram
            out += [mixed % sizes[h] + offs[h] for h in range(a, a + self.heads_per_ngram)]
        return out

    def next_context(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """The context after `tokens` (the last ngram_size-1 tokens of context + tokens)."""
        return torch.cat([context.long(), tokens.long()], dim=-1)[:, -self.context_len:]


# ---------------------------------------------------------------------------------- tables
ROW_DIM = 160
MUL1 = 0x83DCD12D


def words_per_row(K: int) -> int:
    return 1 + ROW_DIM * K // 16


def mul1_codebook(device=None) -> torch.Tensor:
    """All 65,536 mul1 codebook values (fp16), as exllamav3's `mul1_codebook`."""
    s = torch.arange(65536, dtype=torch.int64, device=device)
    prod = (s * MUL1) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    k_inv = torch.tensor([0x1eee], dtype=torch.int16).view(torch.float16).float().item()
    k_bias = torch.tensor([0xc931 - 0x10000], dtype=torch.int16).view(torch.float16).float().item()
    return ((1024 + bsum).float() * k_inv + k_bias).to(torch.float16)


_SRC: dict = {}


def _ring_src(K: int, device) -> torch.Tensor:
    """[160, 16] stream bit of state bit m of element i: ((i - m // K) mod 160) * K + m % K."""
    key = (K, str(device))
    if key not in _SRC:
        i = torch.arange(ROW_DIM).view(-1, 1)
        m = torch.arange(16).view(1, -1)
        _SRC[key] = (((i - m // K) % ROW_DIM) * K + m % K).to(device)
    return _SRC[key]


def decode_rows(packed: torch.Tensor, heads: torch.Tensor, bias: torch.Tensor, K: int,
                codebook: torch.Tensor) -> torch.Tensor:
    """packed int16 [N, 1 + 10 K], heads [N] (the hash head of each row), bias [n_heads, 160]
    -> fp32 [N, 160]. Fixed-shape torch ops on the rows' device (capturable in a graph)."""
    dev = packed.device
    scales = packed[:, 0].contiguous().view(torch.float16).float()
    words = packed[:, 1:].view(torch.uint16).to(torch.int32)                       # [N, 10 K]
    stream = ((words.unsqueeze(-1) >> torch.arange(16, device=dev, dtype=torch.int32)) & 1)
    stream = stream.reshape(packed.shape[0], ROW_DIM * K)
    m = torch.arange(16, device=dev, dtype=torch.int32)
    states = (stream[:, _ring_src(K, dev)] << m).sum(-1)                            # [N, 160]
    return codebook[states.long()].float() * scales[:, None] + bias[heads.long()].float()


def safetensors_index(path: str) -> tuple[int, dict]:
    """(data start offset, header) of one safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return 8 + n, hdr


class DenseTable:
    """A plain [rows, dim] table (the reference's nn.Embedding weight; the CPU tests)."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.rows, self.dim = weight.shape

    def gather(self, ids: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        return self.weight[ids.to(self.weight.device)].to(dtype)


BLOCK = 4096


class DirectRows:
    """Packed rows of a table file read with O_DIRECT, past the page cache, through a hot-row cache.

    A row of `row_bytes` at `offset + id * row_bytes` sits in one or two 4 KiB blocks; a miss reads
    those blocks into an aligned buffer (in parallel for a prompt's thousands) and copies the row out.
    The cache is set-associative (`WAYS` rows a set, the set picked by a multiplicative hash of the id)
    with a reference bit a row and CLOCK eviction within the set: a hit sets the bit, a miss takes the
    first way whose bit is clear, and clears the set's bits when none is. A filesystem that refuses
    O_DIRECT gets buffered reads and the file's page cache dropped after every miss batch (`direct`
    False, said once by the caller)."""

    WAYS = 4
    THREADS = 16

    def __init__(self, path: str, offset: int, row_bytes: int, rows: int, cache_bytes: int):
        self.path, self.offset, self.row_bytes, self.rows = path, int(offset), int(row_bytes), int(rows)
        self.words = self.row_bytes // 2
        try:
            self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
            self.direct = hasattr(os, "O_DIRECT")
        except OSError:
            self.fd = os.open(path, os.O_RDONLY)
            self.direct = False
        self._drop()
        self.sets = max(1, int(cache_bytes) // (self.row_bytes * self.WAYS))
        self.keys = np.full((self.sets, self.WAYS), -1, dtype=np.int64)
        self.vals = np.zeros((self.sets, self.WAYS, self.words), dtype=np.int16)
        self.ref = np.zeros((self.sets, self.WAYS), dtype=bool)
        self.stats = {"hits": 0, "misses": 0, "blocks_read": 0}
        self._buf = None
        self._pool = ThreadPoolExecutor(self.THREADS)

    @property
    def cache_bytes(self) -> int:
        return self.vals.nbytes + self.keys.nbytes + self.ref.nbytes

    def _drop(self) -> None:
        try:
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass

    def _set_of(self, ids: np.ndarray) -> np.ndarray:
        h = (ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(17)
        return (h % np.uint64(self.sets)).astype(np.int64)

    def rows_of(self, ids: np.ndarray) -> np.ndarray:
        """ids (unique or not, int64) -> packed rows int16 [n, words]."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        uniq, inv = np.unique(ids, return_inverse=True)
        sets = self._set_of(uniq)
        hit = self.keys[sets] == uniq[:, None]
        has = hit.any(1)
        way = hit.argmax(1)
        out = np.empty((uniq.size, self.words), dtype=np.int16)
        h = np.nonzero(has)[0]
        out[h] = self.vals[sets[h], way[h]]
        self.ref[sets[h], way[h]] = True
        m = np.nonzero(~has)[0]
        self.stats["hits"] += int(h.size)
        self.stats["misses"] += int(m.size)
        if m.size:
            got = self._read(uniq[m])
            out[m] = got
            self._insert(sets[m], uniq[m], got)
        return out[inv]

    def _insert(self, ms: np.ndarray, ids: np.ndarray, rows: np.ndarray) -> None:
        """New rows into their sets: the k-th miss of a set in this batch takes the set's k-th way
        whose bit is clear; past those the hand has gone round -- the set's bits are cleared and the
        ways are taken from the first."""
        order = np.argsort(ms, kind="stable")
        ms, ids, rows = ms[order], ids[order], rows[order]
        first = np.r_[0, np.nonzero(np.diff(ms))[0] + 1]
        rank = np.arange(ms.size) - np.repeat(first, np.diff(np.r_[first, ms.size]))
        free = ~self.ref[ms]
        nfree = free.sum(1)
        cs = np.cumsum(free, 1)
        way = np.where(rank < nfree, (cs >= (rank + 1)[:, None]).argmax(1), (rank - nfree) % self.WAYS)
        over = rank >= nfree
        if over.any():
            self.ref[ms[over]] = False              # the hand went round the set: every bit cleared
        self.keys[ms, way] = ids
        self.vals[ms, way] = rows
        self.ref[ms, way] = True

    def _read(self, ids: np.ndarray) -> np.ndarray:
        """The rows `ids` from the file: each row's block run (one or two 4 KiB blocks) read whole."""
        start = self.offset + ids * self.row_bytes
        b0 = start // BLOCK * BLOCK
        span = 2 * BLOCK                                 # a row of at most 4 KiB spans at most two blocks
        need = ids.size * span
        if self._buf is None or len(self._buf) < need:
            self._buf = mmap.mmap(-1, max(need, 64 * span))
        buf, fd = self._buf, self.fd
        size = os.fstat(fd).st_size

        def work(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                n = min(span, -(-(size - int(b0[i])) // BLOCK) * BLOCK)
                os.preadv(fd, [memoryview(buf)[i * span:i * span + n]], int(b0[i]))
        step = max(1, -(-ids.size // self.THREADS))
        if ids.size <= 4:
            work(0, ids.size)
        else:
            list(self._pool.map(lambda a: work(a, min(a + step, ids.size)), range(0, ids.size, step)))
        self.stats["blocks_read"] += int(ids.size)
        raw = np.frombuffer(buf, dtype=np.uint8, count=need).reshape(ids.size, span)
        rel = (start - b0)[:, None] + np.arange(self.row_bytes)[None, :]
        rows = np.take_along_axis(raw, rel, axis=1).copy().view(np.int16)
        if not self.direct:
            self._drop()
        return rows

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        os.close(self.fd)


class _CacheStat(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ("nr_cache", "nr_dirty", "nr_writeback", "nr_evicted",
                                                "nr_recently_evicted")]


_SYS_CACHESTAT = 451                                       # the generic syscall table (arm64, x86_64)


def page_cache_bytes(path: str, offset: int, length: int) -> int | None:
    """The bytes of [offset, offset + length) of `path` in the page cache: the kernel's cachestat
    (Linux 6.5+), else mincore over a fresh mapping; None where neither exists (a Mac)."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        rng = (ctypes.c_uint64 * 2)(offset, length)
        cs = _CacheStat()
        if hasattr(libc, "syscall") and libc.syscall(_SYS_CACHESTAT, fd, ctypes.byref(rng), ctypes.byref(cs), 0) == 0:
            return int(cs.nr_cache) * mmap.PAGESIZE
        if not hasattr(libc, "mincore"):
            return None
        a0 = offset // mmap.PAGESIZE * mmap.PAGESIZE
        n = offset + length - a0
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        addr = libc.mmap(None, n, mmap.PROT_READ, mmap.MAP_SHARED, fd, a0)
        if addr in (None, ctypes.c_void_p(-1).value):
            return None
        try:
            vec = (ctypes.c_ubyte * (-(-n // mmap.PAGESIZE)))()
            if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(n), vec) != 0:
                return None
            return int(np.frombuffer(vec, dtype=np.uint8).__and__(1).sum()) * mmap.PAGESIZE
        finally:
            libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(n))
    finally:
        os.close(fd)


class EXL3NGramTable:
    """The packed table of one PLE module, read in place from its safetensors file.

    `prefix` names the module's tensors in the file (`<prefix>trellis` or `<prefix>shard_N.trellis`,
    `<prefix>head_bias`, `<prefix>head_offsets`, `<prefix>head_vocab_sizes`,
    `<prefix>layer_multipliers`); `EXL3NGramTable.find(path)` reads it from the header."""

    # above this many unique rows a gather reads with parallel pread() instead of page faults
    # through the map, which are served one at a time (an 8k-token prompt is ~100k cold rows)
    PARALLEL_FROM = 256
    THREADS = 16

    def __init__(self, path: str, prefix: str):
        self.path = path
        self.prefix = prefix
        start, hdr = safetensors_index(path)
        self._start, self._hdr = start, hdr
        keys = []
        while f"{prefix}shard_{len(keys)}.trellis" in hdr:
            keys.append(f"{prefix}shard_{len(keys)}.trellis")
        if not keys and f"{prefix}trellis" in hdr:
            keys = [f"{prefix}trellis"]
        if not keys:
            raise ValueError(f"{path}: no n-gram trellis under {prefix!r}")
        segs = []
        for k in keys:
            t = hdr[k]
            if t["dtype"] != "I16":
                raise ValueError(f"{path}: {k} is {t['dtype']}, expected I16")
            b0, b1 = t["data_offsets"]
            segs.append((start + b0, int(t["shape"][0]), int(t["shape"][1])))
        words = {s[2] for s in segs}
        if len(words) != 1:
            raise ValueError(f"{path}: n-gram shards differ in row width {sorted(words)}")
        self.words = words.pop()
        self.K = (self.words - 1) * 16 // ROW_DIM
        if words_per_row(self.K) != self.words:
            raise ValueError(f"{path}: {self.words} words a row is no K")
        self.row_bytes = 2 * self.words
        self.rows = sum(s[1] for s in segs)
        # one contiguous run when the shards sit back to back (the converter's layout)
        contiguous = all(segs[i + 1][0] == segs[i][0] + segs[i][1] * self.row_bytes
                         for i in range(len(segs) - 1))
        if not contiguous:
            raise ValueError(f"{path}: n-gram shards are not back to back")
        self.offset = segs[0][0]
        self.dim = ROW_DIM
        self.map = np.memmap(path, dtype=np.int16, mode="r", offset=self.offset,
                             shape=(self.rows, self.words))
        self._fd = None
        self.direct: DirectRows | None = None   # the cold policy's reader (`set_policy`)
        self.bias = self._aux("head_bias").to(torch.float16)
        self.offsets = self._aux("head_offsets").long()
        self.sizes = self._aux("head_vocab_sizes").long()
        self.multipliers = self._aux("layer_multipliers").long()
        self.offsets_np = self.offsets.numpy()
        if self.bias.shape[1] != ROW_DIM or self.bias.shape[0] != self.offsets.numel():
            raise ValueError(f"{path}: head_bias {tuple(self.bias.shape)} vs {self.offsets.numel()} heads")
        self._codebook: dict = {}

    @staticmethod
    def find(path: str) -> list[str]:
        """The module prefixes a table file holds (one per PLE layer)."""
        _, hdr = safetensors_index(path)
        out = set()
        for k in hdr:
            if k.endswith(".trellis"):
                p = k[: -len("trellis")]
                if ".shard_" in p:
                    p = p[: p.rindex("shard_")]
                out.add(p)
        return sorted(out)

    _DT = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64,
           "I32": torch.int32}

    def _aux(self, name: str) -> torch.Tensor:
        k = self.prefix + name
        if k not in self._hdr:
            raise ValueError(f"{self.path}: missing {k}")
        t = self._hdr[k]
        b0, b1 = t["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self._start + b0)
            raw = bytearray(f.read(b1 - b0))
        return torch.frombuffer(raw, dtype=self._DT[t["dtype"]]).reshape(t["shape"]).clone()

    def head_of(self, ids: torch.Tensor) -> torch.Tensor:
        """The hash head each global row id belongs to (its offset range)."""
        offs = self.offsets.to(ids.device)
        return (torch.searchsorted(offs, ids.long(), right=True) - 1).clamp(0, offs.numel() - 1)

    def set_policy(self, policy: str, cache_bytes: int = 2 << 30) -> None:
        """`warm` (the page cache, the default) or `cold` (`DirectRows`, ENG-239)."""
        if policy not in ("warm", "cold"):
            raise ValueError(f"n-gram table policy {policy!r}: warm or cold")
        if self.direct is not None:
            self.direct.close()
            self.direct = None
        if policy == "cold":
            self.direct = DirectRows(self.path, self.offset, self.row_bytes, self.rows, cache_bytes)

    @property
    def policy(self) -> str:
        return "cold" if self.direct is not None else "warm"

    def rows_of(self, ids: np.ndarray) -> np.ndarray:
        """Packed rows int16 [n, words] of the int64 ids, through the table's policy (the engine's
        decode and prefill read this)."""
        if self.direct is not None:
            return self.direct.rows_of(ids)
        return np.asarray(self.map[ids])

    def resident_bytes(self) -> int | None:
        """The table's bytes in the page cache now (None: the platform cannot say)."""
        return page_cache_bytes(self.path, self.offset, self.rows * self.row_bytes)

    def report(self) -> dict:
        """What the table holds of the machine's memory, for /health and the floor checks."""
        out = {"name": "ngram_table", "path": self.path, "policy": self.policy, "file_bytes": self.file_bytes,
               "resident_bytes": self.resident_bytes()}
        if self.direct is not None:
            out.update(cache_bytes=self.direct.cache_bytes, direct=self.direct.direct, **self.direct.stats)
        return out

    def gather_packed(self, ids: torch.Tensor) -> torch.Tensor:
        """ids (any shape) -> packed rows int16 [n, words] in id order, on the CPU."""
        flat = ids.reshape(-1).cpu().numpy().astype(np.int64)
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
            raise IndexError("PLE row id out of range")
        if self.direct is not None:
            return torch.from_numpy(self.direct.rows_of(flat))
        uniq, inv = np.unique(flat, return_inverse=True)
        if uniq.size < self.PARALLEL_FROM:
            rows = np.asarray(self.map[uniq])
        else:
            rows = np.empty((uniq.size, self.words), dtype=np.int16)
            if self._fd is None:
                self._fd = os.open(self.path, os.O_RDONLY)
            fd, rb, off = self._fd, self.row_bytes, self.offset

            def work(lo: int, hi: int) -> None:
                for i in range(lo, hi):
                    rows[i] = np.frombuffer(os.pread(fd, rb, off + int(uniq[i]) * rb), dtype=np.int16)
            step = -(-uniq.size // self.THREADS)
            with ThreadPoolExecutor(self.THREADS) as ex:
                list(ex.map(lambda a: work(a, min(a + step, uniq.size)), range(0, uniq.size, step)))
        return torch.from_numpy(rows[inv.reshape(-1)])

    def warm(self) -> None:
        """Ask the kernel to read the whole table into the page cache, in the background
        (POSIX_FADV_WILLNEED): a decode step then reads its 16 rows from RAM instead of 16 random
        NVMe reads. The page cache stays reclaimable (it counts as available memory)."""
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, self.offset, self.rows * self.row_bytes, os.POSIX_FADV_WILLNEED)
        finally:
            os.close(fd)

    def codebook(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._codebook:
            self._codebook[key] = mul1_codebook(device)
        return self._codebook[key]

    def gather(self, ids: torch.Tensor, dtype=torch.bfloat16, device=None) -> torch.Tensor:
        """ids [...] -> rows [..., 160] in `dtype` (decoded where `device` says, default CPU)."""
        dev = device if device is not None else "cpu"
        packed = self.gather_packed(ids).to(dev)
        heads = self.head_of(ids.reshape(-1).to(dev))
        rows = decode_rows(packed, heads, self.bias.to(dev), self.K, self.codebook(dev))
        return rows.to(dtype).view(*ids.shape, ROW_DIM)

    @property
    def nbytes(self) -> int:
        return 0    # nothing in device memory (the page cache: `resident_bytes`)

    @property
    def file_bytes(self) -> int:
        return self.rows * self.row_bytes
