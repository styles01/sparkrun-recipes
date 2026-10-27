"""Say so when a projection shape is missing from a tile table.

The per-shape tile tables (`nvfp4_linear._CONFIG`, `nvfp4_skinny._CONFIG`, `nvfp4_linear_v2._V2_CONFIG`)
were measured on this model's shapes. Another model's shape silently took the fallback tile, which
is correct but can be far slower, and nothing said which shape did. One line per (table, shape) on
stderr, the first time, and nothing on the hot path after that.
"""

from __future__ import annotations

import sys

_SEEN: set = set()


def missing(table: str, N: int, K: int, fallback) -> None:
    key = (table, int(N), int(K))
    if key in _SEEN:
        return
    _SEEN.add(key)
    print(f"[tiles] {table}: no measured tile for N={N} K={K}; using the fallback {fallback}",
          file=sys.stderr, flush=True)


def seen() -> set:
    return set(_SEEN)
