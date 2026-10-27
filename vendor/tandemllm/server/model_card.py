"""The capability card of an engine model in `/v1/models` (SRV-347, Chat 4, ENG-E8's layering rule).

A client that wants to know what a model can do (the Chat tab: its context window, whether it
thinks and at which effort levels, whether it reads images or calls tools) reads it here, never
from a model name. The card is the `tandem` object proxy models already carry (proxy/route.py),
extended:

    {"backend": "engine", "kind": "engine", "label": "Qwen3.8-27B", "model": "qwen3.8-27b",
     "pack": "exl3-4.5", "context_length": 262144, "max_output": 32768,
     "capabilities": {"thinking": true, "effort_levels": ["off", "low", "medium", "high", "xhigh"],
                      "tools": true, "vision": true, "structured": true, "parallel_tool_calls": true},
     "effort_budgets": {"low": 1024, "medium": 8192, "high": 16384, "xhigh": 0}, "loaded": true}

WHERE THE VALUES COME FROM. What the model is able to do comes from its adapter's file,
`models/<id>/card.json` (beside its model.yaml), which names the weight packs it covers (`packs`,
shell globs against the served pack's name, QSE_WEIGHTS). What this process actually serves comes
from the loader's facts the server passes in: the pack, `max_len` (the window is the smaller of the
two), the default output limit, whether the vision tower is loaded (`vision` is false without it,
whatever the architecture can do), the effort levels' caps (the router document's, else the seed's).
A model with no card.json gets the loader's facts and conservative defaults (tools on, the rest
off). A new architecture fills its card with a card.json and no edit here (tests/test_model_card.py
serves a toy one).
"""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DEFAULT_CAPS = {"thinking": False, "tools": True, "vision": False, "structured": False,
                "parallel_tool_calls": False}

_cache: dict = {}
_lock = threading.Lock()


def cards(models_dir: str | None = None) -> list[dict]:
    """Every `models/*/card.json`, kept until a file's mtime moves; a file that does not parse is
    left out."""
    d = models_dir or MODELS_DIR
    paths = sorted(glob.glob(os.path.join(d, "*", "card.json")))
    key = (d, tuple((p, os.path.getmtime(p)) for p in paths))
    with _lock:
        if _cache.get("key") == key:
            return _cache["cards"]
    out = []
    for p in paths:
        try:
            with open(p) as fh:
                c = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(c, dict) and isinstance(c.get("packs"), list):
            c.setdefault("model", os.path.basename(os.path.dirname(p)))
            out.append(c)
    with _lock:
        _cache.update(key=key, cards=out)
    return out


def find(pack: str | None, models_dir: str | None = None) -> dict | None:
    """The card whose `packs` match the served pack's name."""
    if not pack:
        return None
    if pack.startswith("community/"):
        return _community_card(pack, models_dir)
    for c in cards(models_dir):
        if any(fnmatch.fnmatch(pack, str(g)) for g in c["packs"]):
            return c
    return None


def _community_card(pack: str, models_dir: str | None = None) -> dict | None:
    """A community pack (QSE_WEIGHTS community/<id>, docs/community-weights.md) serves under its
    base model's card with the entry's name as the label (the identity note names it) and vision
    only where the entry says images work."""
    from server.setup import registry
    for m in registry.load_all(models_dir)[0].values():
        e = registry.community(m, pack)
        if e is None:
            continue
        base = next((c for c in cards(models_dir) if c.get("model") == m["id"]), {})
        caps = dict(base.get("capabilities") or {}, vision=e["modalities"]["image"])
        return dict(base, model=m["id"], label=e["name"], packs=[pack], capabilities=caps)
    return None


def card(facts: dict, models_dir: str | None = None) -> dict:
    """The `tandem` object for the engine's model. `facts`: pack, max_len, max_output, vision
    (the tower loaded), effort_caps ({level: cap, 0 no cap}), label (the served name), loaded."""
    c = find(facts.get("pack"), models_dir) or {}
    caps = dict(DEFAULT_CAPS)
    caps.update({k: bool(v) for k, v in (c.get("capabilities") or {}).items() if k in caps})
    caps["vision"] = caps["vision"] and bool(facts.get("vision"))
    efforts = facts.get("effort_caps") or {}
    caps["effort_levels"] = (["off", *[k for k in ("low", "medium", "high", "xhigh")
                                       if k in efforts]] if caps["thinking"] else [])
    max_len = int(facts.get("max_len") or 0)
    limit = int(c.get("context_limit") or 0)
    window = min(x for x in (max_len, limit) if x) if (max_len or limit) else None
    return {
        "backend": "engine", "kind": "engine",
        "label": c.get("label") or facts.get("label") or "",
        "model": c.get("model"),
        "pack": facts.get("pack"),
        "context_length": window,
        "max_output": int(facts.get("max_output") or c.get("max_output") or 0) or None,
        "capabilities": caps,
        "effort_budgets": ({k: int(v or 0) for k, v in efforts.items()}
                           if caps["thinking"] else {}),
        "loaded": bool(facts.get("loaded", True)),
    }


def annotate(entries: list[dict], facts: dict, models_dir: str | None = None) -> list[dict]:
    """`/v1/models`' entries with the card on the engine's own (the ones that carry no `tandem`
    object yet: proxy models keep theirs)."""
    t = card(facts, models_dir)
    return [dict(e, tandem=t) if isinstance(e, dict) and "tandem" not in e else e
            for e in entries]
