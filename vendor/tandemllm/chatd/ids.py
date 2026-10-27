"""ULIDs (26 characters, Crockford base32, time first) and the times the contract uses."""

from __future__ import annotations

import datetime as _dt
import os
import threading
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_lock = threading.Lock()
_last = (0, 0)


def ulid(now_ms: int | None = None) -> str:
    """A new ULID. Within one millisecond the random part counts up, so ids made by this process
    sort in the order they were made (the message tree orders siblings by id)."""
    global _last
    ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    with _lock:
        last_ms, last_rand = _last
        if ms <= last_ms:
            ms, rand = last_ms, (last_rand + 1) & ((1 << 80) - 1)
        else:
            rand = int.from_bytes(os.urandom(10), "big")
        _last = (ms, rand)
    n = (ms << 80) | rand
    out = []
    for _ in range(26):
        out.append(_ALPHABET[n & 31])
        n >>= 5
    return "".join(reversed(out))


def now_ms() -> int:
    return int(time.time() * 1000)


def iso(ms: int | None) -> str | None:
    """Milliseconds since the epoch as the contract's ISO 8601 UTC string."""
    if ms is None:
        return None
    t = _dt.datetime.fromtimestamp(ms / 1000, _dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"
