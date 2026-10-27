"""The selection and its memory estimate (SRV-323): one formula, here, for the server and the fixture.

A selection is what the wizard's steps chose (dashboard/src/api/types.ts `SetupSelection`):

    {model, profile, components: {id: bool}, slots, window, side_window?, kv_pool, floor_gb,
     methods: {id: value}, advanced: {id: value}, device?}

`check(m, body)` validates one against the model's registry entry and returns it complete (every key
the body left out at the registry's default; the side window at the window); the estimate, the plan,
the apply job and Wren's tool calls all go through it. A body that names anything else (`steps`,
`cmd`, `env`, a path) is refused: nothing a client sends becomes a command or a file name.

THE FORMULA (GiB). The parts of the engine after the install, as the dashboard's memory split counts
them on the reference box:

    os          the device preset's reserved_gib (what runs outside the engine's allocator)
    weights     the profile's weights_gib            drafters   the profile's drafters_gib
    graphs, engine, allocator                         the registry's memory parts
    table       a profile's lookup table: warm, its size (the page cache); cold, its row cache (SRV-337)
    per window  the registry's per-window parts: bytes a token x the window (a prefill's score buffer)
    side_slots  (slots - 1) x side_slot_const_gib     (their verify graphs off)
    a tool      its memory_gib, for every tool that is on (worst_gb: its memory_max_gib)
    kv          KV pool on: (min(window, fill) + (slots - 1) x min(side_window, fill)) x bytes a token,
                fill = typical_fill_tokens (32,768); the pool commits pages as conversations
                write them. Pool off: every window at load, (window + (slots - 1) x side_window) x
                bytes a token. worst_gb: every slot at its full window.

    budget      = device total - floor                (what may be used with MemAvailable at the floor)
    total       = the sum of the parts                fits       = total <= budget
    room        = budget - every part but the KV      fits_worst = room >= worst KV
    context_tokens_fit = floor(room / bytes a token), over all slots (0 when room < 0), at most the
                 profile's max_window a slot; a profile with none fits no window

Rounded to 0.01 at the output only; `fits` compares the rounded figures, so a total equal to the
budget to the cent fits. The live MemAvailable (`available_gb`) is information beside it, never part
of `fits`: the estimate describes the engine after Finish, when the old engine and Wren are gone.
models/estimate-cases.json pins the formula (tests/test_setup_estimate.py); the dashboard's copy of
it passes the same file.
"""

from __future__ import annotations

import math

from server.setup import registry

GIB = float(1 << 30)
KEYS = ("model", "profile", "components", "slots", "window", "side_window", "kv_pool", "floor_gb",
        "methods", "advanced", "device")


class SelectionError(ValueError):
    """A selection the server refuses: `status` 400 (an unknown id) or 422, `code` machine-readable,
    `errors` [{field, reason}] (or {component, reason})."""

    def __init__(self, status: int, code: str, message: str, errors: list | None = None):
        super().__init__(message)
        self.status, self.code, self.errors = status, code, errors or []

    def body(self) -> dict:
        return {"error": {"type": self.code, "code": self.code, "message": str(self),
                          "errors": self.errors}}


def _int_in(name: str, v, lo, hi) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v).is_integer():
        raise SelectionError(422, "out_of_range", f"{name}: a whole number from {lo:g} to {hi:g}",
                             [{"field": name, "reason": "not a whole number"}])
    if not lo <= v <= hi:
        raise SelectionError(422, "out_of_range", f"{name}: {v:g} is outside {lo:g} to {hi:g}",
                             [{"field": name, "reason": f"from {lo:g} to {hi:g}"}])
    return int(v)


def _on_step(name: str, v: int, step: int) -> int:
    """A window is a whole number of the KV pool's pages (8,192 tokens on Qwen3.8-27B)."""
    if step and v % step:
        raise SelectionError(422, "off_step", f"{name}: a multiple of {step:,}, not {v:,}",
                             [{"field": name, "reason": f"a multiple of {step}"}])
    return v


def check(m: dict, body) -> dict:
    """The selection `body` for model `m`, complete and checked; SelectionError otherwise."""
    if not isinstance(body, dict):
        raise SelectionError(422, "invalid_selection", "the selection must be a JSON object")
    extra = sorted(k for k in body if k not in KEYS)
    if extra:
        raise SelectionError(422, "unknown_field", f"unknown field {extra[0]!r}; a selection has "
                             f"{', '.join(KEYS)}", [{"field": k, "reason": "unknown"} for k in extra])
    if body.get("model") != m["id"]:
        raise SelectionError(400, "unknown_model", f"no model {body.get('model')!r} here")
    pid = body.get("profile", registry.recommended_profile(m)["id"])
    if not isinstance(pid, str) or registry.profile(m, pid) is None:
        raise SelectionError(400, "unknown_profile", f"no weight profile {pid!r} for {m['id']}")
    comps = {c["id"]: c["default"] for c in m["components"]}
    given = body.get("components") or {}
    if not isinstance(given, dict):
        raise SelectionError(422, "invalid_selection", "components: an object of id -> true/false")
    for cid, on in given.items():
        if cid not in comps:
            raise SelectionError(400, "unknown_component", f"no tool {cid!r} for {m['id']}")
        if not isinstance(on, bool):
            raise SelectionError(422, "invalid_selection", f"components.{cid}: true or false")
        comps[cid] = on
    # a community pack whose entry says images do not work on its load path (ENG-271): the vision
    # tower stays off whatever was sent, so the overlay writes VISION=off and the estimate counts none
    if registry.profile(m, pid).get("vision") is False:
        for c in m["components"]:
            if "VISION" in c["knobs"]:
                comps[c["id"]] = False
    bad = []
    for c in m["components"]:
        if c["required"] and not comps[c["id"]]:
            bad.append({"component": c["id"], "reason": f"{c['label']} cannot be switched off"})
        if not comps[c["id"]]:
            continue
        for d in c["depends_on"]:
            if not comps[d]:
                bad.append({"component": c["id"], "reason": f"{c['label']} needs "
                            f"{registry.component(m, d)['label']} ({d}) on"})
        for d in c["conflicts"]:
            if comps[d]:
                bad.append({"component": c["id"], "reason": f"{c['label']} and "
                            f"{registry.component(m, d)['label']} ({d}) cannot both be on"})
    if bad:
        raise SelectionError(422, "invalid_combination", "; ".join(b["reason"] for b in bad), bad)
    p = m["parallelism"]
    # the card's slots.max is how far it was measured; the engine's own limit is the cap
    slots = body.get("slots", p["slots"]["default"])
    if isinstance(slots, int) and not isinstance(slots, bool) and slots > registry.ENGINE_MAX_SLOTS:
        raise SelectionError(422, "out_of_range", f"slots: {slots} is above {registry.ENGINE_MAX_SLOTS}, "
                             "the most chats the engine verifies in one pass",
                             [{"field": "slots", "reason": f"at most {registry.ENGINE_MAX_SLOTS}"}])
    slots = _int_in("slots", slots, p["slots"]["min"], registry.ENGINE_MAX_SLOTS)
    window = _on_step("window", _int_in("window", body.get("window", p["window"]["default"]),
                                        p["window"]["min"], p["window"]["max"]), p["window"]["step"])
    mw = registry.profile(m, pid).get("max_window")
    if mw and window > mw:
        raise SelectionError(422, "out_of_range", f"window: {window:,} is above {pid}'s largest window, {mw:,}",
                             [{"field": "window", "reason": f"at most {mw} with {pid}"}])
    side = body.get("side_window")
    side = window if side is None else side
    side = _on_step("side_window", _int_in("side_window", side, p["side_window"]["min"],
                                           p["side_window"]["max"]), p["side_window"]["step"])
    # a side slot never holds more than slot 0 (server/app.py: min(side_max_len, max_len))
    side = min(side, window)
    pool = body.get("kv_pool", p["kv_pool"]["default"])
    if not isinstance(pool, bool):
        raise SelectionError(422, "invalid_selection", "kv_pool: true or false",
                             [{"field": "kv_pool", "reason": "true or false"}])
    floor = body.get("floor_gb", p["floor_gb"]["default"])
    lo, hi = p["floor_gb"]["min"], p["floor_gb"]["max"]
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not lo <= floor <= hi:
        raise SelectionError(422, "out_of_range", f"floor_gb: a number from {lo:g} to {hi:g}",
                             [{"field": "floor_gb", "reason": f"from {lo:g} to {hi:g}"}])
    out = {"model": m["id"], "profile": pid, "components": comps, "slots": slots, "window": window,
           "side_window": side, "kv_pool": pool, "floor_gb": float(floor)}
    for group in ("methods", "advanced"):
        vals = {f["id"]: f["default"] for f in m[group]}
        given = body.get(group) or {}
        if not isinstance(given, dict):
            raise SelectionError(422, "invalid_selection", f"{group}: an object of id -> value")
        for fid, v in given.items():
            f = next((x for x in m[group] if x["id"] == fid), None)
            if f is None:
                raise SelectionError(422, "unknown_field", f"{group}.{fid}: no such field",
                                     [{"field": f"{group}.{fid}", "reason": "unknown"}])
            why = registry.field_error(f, v)
            if why:
                raise SelectionError(422, "invalid_value", f"{group}.{fid}: {why}",
                                     [{"field": f"{group}.{fid}", "reason": why}])
            vals[fid] = v
        out[group] = vals
    dev = body.get("device", m["devices"][0]["id"])
    if not isinstance(dev, str) or not any(d["id"] == dev for d in m["devices"]):
        raise SelectionError(400, "unknown_device", f"no device preset {dev!r} for {m['id']}")
    out["device"] = dev
    return out


def defaults(m: dict) -> dict:
    """The recommended selection: what a new box gets, and what reset level 2 goes back to."""
    return check(m, {"model": m["id"]})


def _r(x: float) -> float:
    return round(x, 2) + 0.0          # never -0.0


def estimate(m: dict, sel: dict, *, available_gb: float | None = None) -> dict:
    """The estimate of a checked selection (`check`), in the shape of `SetupEstimate`."""
    p = m["parallelism"]
    prof = registry.profile(m, sel["profile"])
    dev = next(d for d in m["devices"] if d["id"] == sel["device"])
    kvb = int(p["kv_bytes_per_token"])
    fill = int(p["typical_fill_tokens"])
    slots, window, side = sel["slots"], sel["window"], sel["side_window"]
    parts = [{"id": "os", "label": "The OS and other programs", "gb": dev["reserved_gib"],
              "kind": "fixed"},
             {"id": "weights", "label": f"Weights ({prof['label']})", "gb": prof["weights_gib"],
              "kind": "fixed"}]
    for part in m["memory"]["parts"]:
        parts.append({"id": part["id"], "label": part["label"], "gb": part["gib"],
                      "kind": part["kind"]})
    table = prof.get("table")
    if table:
        warm = table["policy"] == "warm"
        parts.append({"id": "table", "label": "n-gram table (page cache, kept warm)" if warm else
                      "n-gram row cache (the table read past the page cache)",
                      "gb": table["gib"] if warm else table["cache_gib"], "kind": "fixed"})
    for part in m["memory"].get("per_window_parts") or []:
        gb = part["bytes_per_token"] * window / GIB
        if part.get("max_gib") is not None:
            gb = min(gb, part["max_gib"])           # a part that stops growing (a prefill's workspace)
        parts.append({"id": part["id"], "label": part["label"], "gb": gb, "kind": part["kind"]})
    if slots > 1:
        parts.append({"id": "side_slots", "label": f"{slots - 1} more slot(s), beyond their KV",
                      "gb": (slots - 1) * p["side_slot_const_gib"], "kind": "fixed"})
    for c in m["components"]:
        if not sel["components"][c["id"]]:
            continue
        gb = (prof.get("components_gib") or {}).get(
            c["id"], prof["drafters_gib"] if c["id"] == "drafters" else c["memory_gib"])
        if gb <= 0 and c["memory_max_gib"] is None:
            continue
        part = {"id": c["id"], "label": c["label"], "gb": gb, "kind": "fixed"}
        if c["memory_max_gib"] is not None:
            part.update(kind="reserve", worst_gb=c["memory_max_gib"])
        parts.append(part)
    worst_rows = window + (slots - 1) * side
    typical_rows = min(window, fill) + (slots - 1) * min(side, fill)
    kv_worst = worst_rows * kvb / GIB
    kv = (typical_rows if sel["kv_pool"] else worst_rows) * kvb / GIB
    fixed = sum(x["gb"] for x in parts)
    parts.append({"id": "kv", "label": "Conversations (KV)" + (", committed as they grow"
                                                               if sel["kv_pool"] else
                                                               ", every window at load"),
                  "gb": kv, "kind": "kv", "worst_gb": kv_worst})
    total = fixed + kv
    budget = dev["total_gib"] - sel["floor_gb"]
    room = budget - fixed
    fits = _r(total) <= _r(budget)
    fits_worst = _r(room) >= _r(kv_worst)
    ctx = max(0, math.floor(room * GIB / kvb))
    mw = prof.get("max_window", 0)
    if mw is None:
        ctx = 0
    elif mw:
        ctx = min(ctx, mw * slots)
    notes = []
    if mw is None:
        notes.append(f"{prof['label']} does not fit {dev['label']} at any window: it is "
                     f"{_r(max(0.0, fixed - budget)):.2f} GiB over the budget before any context.")
    if not fits:
        big = sorted(parts, key=lambda x: -x["gb"])[:3]
        notes.append(f"Over the budget by {_r(total - budget):.2f} GiB. The largest parts: "
                     + ", ".join(f"{x['label']} {x['gb']:.2f} GiB" for x in big) + ".")
        if not sel["kv_pool"]:
            notes.append("Turn the shared KV pool on: without it every window takes its memory "
                         "at load.")
        notes.append("Fewer slots, a shorter window, a smaller weight profile or fewer tools "
                     "bring it down.")
    elif not fits_worst:
        notes.append(f"With every slot at its full window the conversations would need "
                     f"{_r(kv_worst - room):.2f} GiB more than there is. When memory runs short, "
                     f"a new request waits for room instead of the engine running out of memory.")
    if room > 0:
        per = min(window, ctx // slots) if slots else 0
        notes.append(f"About {ctx:,} tokens of conversation fit in all, {per:,} a slot when all "
                     f"{slots} are full (the window is {window:,}).")
    return {"contract_version": "1.0", "total_gb": _r(total), "available_gb": available_gb,
            "device_gb": _r(dev["total_gib"]), "floor_gb": _r(sel["floor_gb"]),
            "budget_gb": _r(budget),
            "parts": [dict(x, gb=_r(x["gb"]), **({"worst_gb": _r(x["worst_gb"])}
                                                  if "worst_gb" in x else {})) for x in parts],
            "fits": fits, "fits_worst": fits_worst, "context_tokens_fit": ctx,
            "kv_bytes_per_token": kvb, "typical_fill_tokens": fill, "kv_worst_gb": _r(kv_worst),
            "room_gb": _r(room), "notes": notes}
