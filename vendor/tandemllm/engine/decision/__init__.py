"""Decision models inside the engine (stage A of the design, ENG-179).

A decision model answers typed questions (choice, score, noul) over a state in one forward pass,
with calibrated probabilities: Laya's three checkpoints today, probe heads and own models later.
This package is the runtime: the interface (base.py), the registry with routing and
planner-gated loading (registry.py), the cross-caller batcher in the idle window (batcher.py), the
device placement on an own stream and memory pool (device.py), the Laya family (laya/) and a fake
model for the tests (fake.py). The HTTP surface is server/decider_api.py.

Nothing here touches Qwen: no tensor, stream or allocator pool is shared, and with
`QWEN38_DECIDER_MODELS` empty the package is never imported.
"""

from __future__ import annotations

FAKE_PREFIXES = ("fake-", "other-")


def catalogue(names: list[str], root: str, *, allow_fake: bool = False) -> list:
    """The models `QWEN38_DECIDER_MODELS` names, in that order. Fake names only in a test or with
    the fake engine; an unknown name stops the start."""
    out = []
    for n in names:
        if n.startswith(FAKE_PREFIXES):
            if not allow_fake:
                raise ValueError(f"{n!r} is a test double; it is served only by a fake engine")
            from engine.decision.fake import FakeModel
            out.append(FakeModel(n))
            continue
        from engine.decision.laya.model import NAMES, LayaModel
        if n not in NAMES:
            raise ValueError(f"unknown decision model {n!r}; this engine serves {', '.join(NAMES)}")
        out.append(LayaModel(n, root))
    return out


def routers(models: list) -> dict:
    """family -> the routing function the registry calls for an `auto` name."""
    out = {}
    if any(m.family == "laya" for m in models):
        from engine.decision.laya import router as laya_router
        served = {m.name for m in models if m.family == "laya"}
        default = laya_router.ENGLISH if laya_router.ENGLISH in served else sorted(served)[0]

        def laya(state, questions, *, task=None, lang=None):
            return laya_router.route(state, questions, task=task, lang=lang, default=default)

        out["laya"] = laya
    return out
