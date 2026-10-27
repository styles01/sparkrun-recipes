"""The documents a new box starts with (D3's shipped defaults, stored at a box's first start).

The owner, 2026-10-03: "people will not switch their profiles and set up opencode with default
settings". The conductor's built-in documents (version 0, schema.py `seed`) are the minimal ones
the engine always had, and what the tests build on; the live box's documents of 2026-10-03 are
schema.py's SHIPPED_ROUTER, SHIPPED_PACKAGES, SHIPPED_SETTINGS and SHIPPED_PROFILES (the packages'
texts under the repository's packages/). With `--conductor-defaults shipped` (ops/start.sh passes
it to the :8000 service) the server stores each of them that the conductor's directory has no
version of yet, as version 1, at its start: a new box gets all of them; a box with its own
documents gets only what it never had (the `hermes` and `general` profiles on the reference box;
the `tools` profile, SRV-621, on every box that ran before it). Nothing that has a
version is touched: a stored settings document without `tools_profile` reads it as `tools`
(schema.tools_profile), so no migration is needed for the rule.
"""

from __future__ import annotations

import copy
import os

from server.conductor import schema

NOTE = "shipped defaults (0.3.22-test): the live documents of 2026-10-03"


def documents() -> list[tuple[str, str | None, dict]]:
    # in the order their checks need: a router names packages, settings a profile
    out = [("packages", None, schema.SHIPPED_PACKAGES)]
    out += [("profiles", n, d) for n, d in schema.SHIPPED_PROFILES.items()
            if n not in schema.LOCKED]
    out += [("router", None, schema.SHIPPED_ROUTER), ("settings", None, schema.SHIPPED_SETTINGS)]
    return [(k, n, copy.deepcopy(d)) for k, n, d in out]


def install(store) -> list[str]:
    """Store every shipped document the store has no version of. Returns what was stored."""
    done = []
    for kind, name, doc in documents():
        try:
            if store._pointer(store._dir(kind, name)) is not None:
                continue
            v = store.put(kind, name, doc, note=NOTE, author="shipped")
            done.append(f"{kind}{'/' + name if name else ''} v{v}")
        except Exception as exc:                                   # noqa: BLE001
            done.append(f"{kind}{'/' + name if name else ''} refused: {exc}"[:200])
    return done


def install_at(root: str) -> list[str]:
    from server.conductor.store import Store
    os.makedirs(root, exist_ok=True)
    return install(Store(root))
