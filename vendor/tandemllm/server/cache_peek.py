"""Where a conversation's rows are before it is sent, and bringing them back while the engine is idle
(SRV-349, the Chat tab's Context panel).

    POST /v1/cache/peek      {"session", "messages", "tools"?, ...the chat body}
        -> {"tier", "cached_tokens", "reread_tokens", "reread_eta_ms", "tokens", "session",
            "source", "restore_ms"}
    POST /v1/cache/prewarm   the same body
        -> {"prewarmed": true|false, "why"?, "restore"?, ...the peek}

THE PEEK renders the body as a request would be rendered (server/app.py `build_prompt`, the host
tokenizer only), puts back what the engine puts back before its prefill -- the router's notes of the
session (`Router.replayed_for`) and the resident end anchor's hidden spans -- and asks the cache
tiers the questions `engine/cache.py prefill` asks them, without running anything:

    resident      the rows in place: an anchor of slot 0's resident prompt, its end anchor, or a side
                  slot's conversation the next request carries over (`_carry_candidate`)
    ram_park      the parked session or the pinned head (`ResidentPrefix.park_offer`, `pin_offer`)
    ram_snapshot  a prefix snapshot in the state store (`StateStore.find`)
    nvme_spill    a conversation on the NVMe that beats RAM (`SessionSpill.candidate`)
    disk_tier     the shared-prefix tier (`DiskTier.find`)
    none          nothing: the whole prompt is read

The answer is the tier the longest line comes from, ties to the cheaper tier (the order above), and
`cached_tokens` that line, in the served prompt's positions (the replayed notes count, as the
usage frame's `cache_n` counts them). `reread_eta_ms` is the re-read at the measured prefill rate of
this process (`Rates`, the median of the last prefills of at least `RATE_MIN` tokens; before the
first one, the caches' own prior), plus the NVMe or disk read for those tiers. Model-agnostic: the
tiers keep any model's state through its adapter (ENG-157); an engine without caches
(`eng.caps.caches` off) answers `none`.

The tiers' lookups count hits and misses for the dashboard; a peek is not a request, so their
counters are put back as they were (`quiet`). Under the engine lock: a peek while a request holds
the engine is refused (409) rather than read from rows a prefill is writing.

THE PREWARM: when the peek says `nvme_spill`, no request runs or waits, and the engine lock is free,
the entry is read back into the engine's buffers as the next request of that conversation would
read it (`SessionSpill.maybe_restore`: the resident conversation is written to the NVMe first when
it is worth keeping) -- so the send that follows resumes at the end anchor. It does nothing when:
the engine is busy, the tier is another one (RAM needs no prewarm; the disk tier reads its rows
during the prefill, and only where that is cheaper than the prefill), another conversation used the
resident rows in the last `IDLE_S` seconds, or the read would take longer than `MAX_MS`. A request
that arrives during a prewarm waits at most one block: the save of the conversation in place and the
restore both stop between blocks when a request waits (engine/spill.py `stop`); the entry stays on
the NVMe and the conversation in place stays in RAM.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import statistics
import threading

#: cache sources (`cache_source` of the usage frame) -> tiers. A source not here is its own name
#: (`response`, `prepared`, `continuation`: answers that read no tier)
TIER_OF = {"resident": "resident", "end": "resident", "side": "resident",
           "park": "ram_park", "pin": "ram_park", "prefix": "ram_snapshot",
           "spill": "nvme_spill", "disk": "disk_tier", "none": "none"}
#: ties go to the cheaper tier
ORDER = ("resident", "ram_park", "ram_snapshot", "nvme_spill", "disk_tier")

#: a prefill shorter than this says little about the rate of a long re-read
RATE_MIN = 1024
#: the prewarm's longest restore: a request arriving meanwhile waits at most this long
MAX_MS = 3000.0
#: another conversation that used the rows in place this recently keeps them (no prewarm)
IDLE_S = 120.0


def tier_of(source: str | None) -> str:
    return TIER_OF.get(source or "none", source or "none")


class Rates:
    """The prefill rates of this process's last requests that forwarded at least `RATE_MIN` tokens
    (tok/s), for the peek's estimate."""

    def __init__(self, n: int = 16):
        self._q: collections.deque = collections.deque(maxlen=n)
        self._lock = threading.Lock()

    def note(self, forwarded: int, prompt_ms: float | None) -> None:
        if forwarded >= RATE_MIN and prompt_ms and prompt_ms > 0:
            with self._lock:
                self._q.append(forwarded / (prompt_ms / 1e3))

    def rate(self) -> float | None:
        with self._lock:
            return statistics.median(self._q) if self._q else None


RATES = Rates()


#: the counters each tier's lookup adds to (and nothing else writes): `StateStore.find`,
#: `DiskTier.find` (its writer thread counts other keys meanwhile), `SessionSpill.candidate`
LOOKUP_KEYS = {"StateStore": ("hits", "misses", "rejected_collisions"),
               "DiskTier": ("lookups", "misses", "incomplete", "declined_slow",
                            "rejected_collisions"),
               "SessionSpill": ("refused",)}


@contextlib.contextmanager
def quiet(*tiers):
    """The tiers' own lookups asked without counting: the counters a lookup adds to as they were
    afterwards (`LOOKUP_KEYS`, and a store hit's `hits_<kind>`)."""
    saved = []
    for t in tiers:
        keys = LOOKUP_KEYS.get(type(t).__name__) if t is not None else None
        if keys:
            mine = lambda k, keys=keys: k in keys or k.startswith("hits_")   # noqa: E731
            saved.append((t, mine, {k: copy.deepcopy(v) for k, v in t.stats.items() if mine(k)},
                          getattr(t, "last_kind", None)))
    try:
        yield
    finally:
        for t, mine, s, last in saved:
            for k in [k for k in t.stats if mine(k) and k not in s]:
                del t.stats[k]
            t.stats.update(s)
            if hasattr(t, "last_kind"):
                t.last_kind = last


def _ram(res, ids: list[int], max_len: int, spill=None) -> tuple[int, str]:
    """What the rows in RAM offer `ids` and from where: (line, tier). The end anchor counts as the
    server's replay serves it (the hidden spans put back, or the copy aligned to the rows), as
    `SessionSpill.candidate` counts it."""
    from engine.cache import ResidentPrefix
    from engine.spill import _Shim
    c = res.common(ids)
    b = res.best(c, max_len)
    be = res.end_offer(c, max_len)
    end = res.end
    if end is not None and be < int(end["at"]) <= max_len:
        e = int(end["at"])
        if end.get("hidden") and ResidentPrefix.replay(_Shim(end, res.tokens, res.glue), list(ids))[1]:
            be = e
        elif getattr(spill, "offer_align", None) is not None and not end.get("mm"):
            try:
                if spill.offer_align(res.tokens[:e], end, spill._extra(res), list(ids)):
                    be = e
            except Exception:                                      # noqa: BLE001
                pass
    bp = res.pin_offer(ids, c, max_len)
    bk = res.park_offer(ids, c, max_len, align=True)
    here, tier = max(b, be), "resident"
    if max(bp, bk) > here:
        here, tier = max(bp, bk), "ram_park"
    return here, tier


def find(ids: list[int], *, res=None, spill=None, store=None, disk=None, side=None) -> dict:
    """The tier `ids` would be served from and its line, asked of the tiers without a restore.
    `side(ids)`: the line of a side slot's conversation the next request would carry over, or 0
    (server/app.py `_carry_candidate`). Under the engine lock."""
    from engine import cache
    n = len(ids)
    out = {"tier": "none", "line": 0, "restore_ms": None}
    if n < 2:
        return out
    max_len = n - 1
    lines = []                                       # (line, tier, restore ms)
    with quiet(res, spill, store, disk):
        here = 0
        if res is not None and getattr(res, "chunk", 0) > 0:
            here, tier = _ram(res, ids, max_len, spill)
            lines.append((here, tier, None))
            if side is not None:
                s = int(side(ids) or 0)
                if s:
                    lines.append((s, "resident", None))
            if spill is not None:
                got = spill.candidate(res, ids, max_len, replay=True)
                if got is not None:
                    e, line, _ = got
                    c = res.common(ids)
                    need = sum(int(b["bytes"]) for b in e.man["blocks"] if b["b"] + b["rows"] > c)
                    lines.append((line, "nvme_spill", need / spill.read_bps * 1e3))
        hashes = None
        if store is not None:
            hashes = cache.prefix_hashes(ids)
            hit = store.find(ids, hashes, max_len=max_len, min_len=here)
            if hit is not None:
                hit[1].hits -= 1                     # a peek is not a hit
                lines.append((hit[0], "ram_snapshot" if hit[1].kind == "prefix" else hit[1].kind,
                              None))
        if disk is not None:
            hashes = hashes if hashes is not None else cache.prefix_hashes(ids)
            have = here if res is None or res.taint is None else min(here, res.taint)
            dh = disk.find(ids, hashes, max_len, max([here] + [x[0] for x in lines]), have)
            if dh is not None:
                lines.append((dh.L, "disk_tier", dh.need_bytes / disk.read_bps * 1e3))
    best = None
    for line, tier, ms in lines:
        if line <= 0:
            continue
        rank = ORDER.index(tier) if tier in ORDER else len(ORDER)
        if best is None or line > best[0] or (line == best[0] and rank < best[3]):
            best = (line, tier, ms, rank)
    if best is not None:
        out.update(tier=best[1], line=best[0],
                   restore_ms=round(best[2], 1) if best[2] is not None else None)
    return out


def answer(ids: list[int], got: dict, session: str | None, rate: float | None) -> dict:
    """The peek's JSON from `find`'s result."""
    n = len(ids)
    cached = int(got["line"])
    reread = max(0, n - cached)
    eta = None
    if rate:
        eta = reread / rate * 1e3 + float(got.get("restore_ms") or 0.0)
    return {"tier": got["tier"], "cached_tokens": cached, "reread_tokens": reread,
            "reread_eta_ms": round(eta, 1) if eta is not None else None, "tokens": n,
            "restore_ms": got.get("restore_ms"), "session": session}
