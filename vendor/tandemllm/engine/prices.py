"""Price tables per weight set: what a verify, a draft and a rollback cost, in ms.

The routers decide how wide a block to draft and how many tree nodes to verify by pricing each
option in milliseconds. The prices in the code are the NVFP4 weight set's (engine/router.py
`SERVED_TREE_MS`, engine/lenrouter.py `TREE_MS_B` / `DRAFT_MS_B`, the merged router's head and
rollback constants), and `QWEN38_TREE_MS` in ops/serve.env is its measured tree curve. A server on
another weight set (the plain-FP8 profile) paid another verify and was priced as if it
had not. Prices move speed, never text: every proposal is verified by the target.

A table file holds one entry per weight set:

    {"fp8-plain": {"tree_ms": {"8": 150.2, "16": 152.0, "24": 170.1, "32": 181.3},
                   "lenrouter_tree_ms": {"8": 150.2, "16": 152.0, "32": 181.3},
                   "lenrouter_draft_ms": {"8": 13.0, "16": 14.0},
                   "head_fixed_ms": 27.0, "rollback_ms": 6.4,
                   "measured": "2026-09-27 tools/verify_curve.py ..."}}

and the server takes `--price-table FILE[:NAME]` (NAME defaults to the only entry, or is required
when there are several). Precedence, highest first: `QWEN38_TREE_MS` in the environment (the
escape hatch it always was), the table, the constants in the code. No table = today's decisions.
"""

from __future__ import annotations

import json
import os

KEYS = {"tree_ms", "lenrouter_tree_ms", "lenrouter_draft_ms", "head_fixed_ms", "rollback_ms",
        "measured", "note"}
CURVES = ("tree_ms", "lenrouter_tree_ms", "lenrouter_draft_ms")


def _curve(name: str, raw) -> dict[int, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"price table: {name} must be a non-empty {{rows: ms}} object")
    out = {}
    for k, v in raw.items():
        r, ms = int(k), float(v)
        if r < 1 or not (0.0 < ms < 10_000.0):
            raise ValueError(f"price table: {name}[{k!r}] = {v!r} is not a row count and a time in ms")
        out[r] = ms
    return dict(sorted(out.items()))


def load(spec: str | None) -> tuple[str | None, dict]:
    """`FILE[:NAME]` -> (name, entry with the curves as {int: float}); None/'' -> (None, {})."""
    if not spec:
        return None, {}
    path, _, name = str(spec).partition(":")
    path = os.path.expanduser(path)
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or not doc:
        raise ValueError(f"price table {path}: expected an object of weight sets")
    if not name:
        if len(doc) != 1:
            raise ValueError(f"price table {path} has {sorted(doc)}: name one as FILE:NAME")
        name = next(iter(doc))
    if name not in doc:
        raise ValueError(f"price table {path}: no entry {name!r} (has {sorted(doc)})")
    raw = doc[name]
    extra = set(raw) - KEYS
    if extra:
        raise ValueError(f"price table {path}:{name}: unknown keys {sorted(extra)}")
    entry = {}
    for k in CURVES:
        if k in raw:
            entry[k] = _curve(k, raw[k])
    if "tree_ms" in entry and not {8, 16} <= set(entry["tree_ms"]):
        raise ValueError(f"price table {path}:{name}: tree_ms needs 8 and 16 (the two arms)")
    for k in ("head_fixed_ms", "rollback_ms"):
        if k in raw:
            v = float(raw[k])
            if not (0.0 <= v < 1000.0):
                raise ValueError(f"price table {path}:{name}: {k} = {v}")
            entry[k] = v
    return name, entry
