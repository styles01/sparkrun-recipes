"""The engine's memory limits (OPS-47): one table, its overrides, and what each limit holds now.

Every limit is a key in `ops/serve.env`, passed to the server as a flag by `ops/start.sh`. The
dashboard's Memory plan panel may override any of them except `MAX_LEN` (the window the engine
loaded with): the override is the `engine` section of the conductor's `residency` document, versioned
like every conductor document. A key that is `live` applies from the next request on, under the
engine lock; the others are read once, when the engine loads, so a change to them says "needs a
restart" until the engine has restarted with it.

    rows = view(STATE)          # the panel: value, source, live or restart, usage against it
    refresh(STATE)              # at the start of a request: a new document version applies
    overrides_at_start(a, doc)  # before the load: the document's values over the flags

Nothing here allocates or frees memory itself: `refresh` hands the new caps to the objects that
hold the bytes (engine/cache.py), and they give back what is over a lower cap.
"""

from __future__ import annotations

GIB = 1 << 30
MIB = 1 << 20

# key (serve.env and the document), flag dest, unit, live, what it caps
LIMITS: tuple = (
    ("MEM_CEILING_GIB", "mem_ceiling_gib", "GiB", True,
     "the engine between requests, allocated plus the allocator's reserve"),
    ("MEM_PEAK_GIB", "mem_peak_gib", "GiB", True,
     "the engine during a prefill; the residency planner subtracts it"),
    ("RESERVE_SLACK_GIB", "reserve_slack_gib", "GiB", True,
     "the allocator's reserve kept between requests; over it, it goes back to the driver"),
    ("RESIDENT_GB", "resident_gb", "GiB", True, "the resident prefix's anchors"),
    ("RESIDENT_STASH_GB", "resident_stash_gb", "GiB", True,
     "the rows a short request between two turns may overwrite, copied aside"),
    ("RESIDENT_GUEST_MAX", "resident_guest_max", "tokens", True,
     "the longest request that counts as such a guest"),
    ("RESIDENT_PIN_GB", "resident_pin_gb", "GiB", True,
     "the pinned head of a prompt a long unrelated request replaced"),
    ("RESIDENT_PARK_GB", "resident_park_gb", "GiB", True,
     "the rows of one other session, swapped back on its next turn"),
    ("CACHE_GB", "cache_budget_gb", "GiB", True, "the state store"),
    ("STORE_ENTRY_GB", "store_entry_gb", "GiB", True, "one state store entry"),
    ("VISION_EMBED_MB", "image_cache_mb", "MiB", True, "encoded images"),
    ("SUFFIX_STORE_MB", "suffix_store_mb", "MiB", True,
     "the suffix store's token log (host memory and disk, not the GPU pool)"),
    ("PREFIX_CHUNK", "prefix_chunk", "tokens", False,
     "the prefill's chunk and the caches' grid"),
    ("NARROW_ARM", "narrow_arm", "on/off", False,
     "the block-8 drafter arm (SPD-75): about 8 GiB at a 262,144-token window"),
    ("MAX_LEN", "max_len", "tokens", False, "the context window"),
)
KEYS = tuple(k for k, *_ in LIMITS)
LOCKED = {"MAX_LEN": "the context window is set when the engine loads (MAX_LEN, the setup guide)"}
#: the window MEM_CEILING_GIB and MEM_PEAK_GIB are set at (ops/serve.env, every profile): a longer slot 0
#: window adds its KV above it to both (server/app.py `_window_above_limits`, ENG-363)
LIMITS_WINDOW = 262_144
_BY_KEY = {k: (dest, unit, live, what) for k, dest, unit, live, what in LIMITS}


def _scale(unit: str) -> int:
    return GIB if unit == "GiB" else MIB if unit == "MiB" else 1


def flag_value(args: dict, key: str):
    """The value the server was started with for `key`, in the key's unit."""
    dest = _BY_KEY[key][0]
    v = (args or {}).get(dest)
    if key == "NARROW_ARM":
        return 1 if v in (None, "on", True, 1) else 0
    return v


def doc_section(doc: dict | None) -> dict:
    """The document's overrides, the locked keys left out."""
    eng = (doc or {}).get("engine") or {}
    return {k: v for k, v in eng.items() if k in _BY_KEY and k not in LOCKED and v is not None}


def overrides_at_start(a, doc: dict | None, *, len_mode: str | None = None) -> list[str]:
    """Before the load: the document's values over the flags, every key. Returns one line per
    override for the start log. An override the configuration cannot take is skipped, never
    fatal: a dashboard edit must not keep the engine from starting."""
    lines = []
    for key, v in doc_section(doc).items():
        dest, unit, live, _ = _BY_KEY[key]
        if key == "NARROW_ARM":
            want = "on" if int(v) else "off"
            if want == "off" and len_mode != "wide":
                lines.append(f"{key}=0 skipped: the wide arm alone needs QWEN38_LEN_MODE=wide")
                continue
            if getattr(a, dest, "on") != want:
                lines.append(f"{key}: {getattr(a, dest, 'on')} -> {want}")
            setattr(a, dest, want)
            continue
        cur = getattr(a, dest, None)
        new = int(v) if unit == "tokens" else float(v)
        if cur != new:
            lines.append(f"{key}: {cur} -> {new}")
        setattr(a, dest, new)
    return lines


def apply(state: dict, values: dict) -> list[str]:
    """Hand the live keys in `values` to the objects that hold the bytes. Returns what changed."""
    mem, store, res = state.get("mem"), state.get("state_store"), state.get("resident")
    applied = state.setdefault("limits_applied", {})
    done = []
    for key, v in values.items():
        dest, unit, live, _ = _BY_KEY[key]
        if not live or applied.get(key) == v:
            continue
        b = int(float(v) * _scale(unit)) if unit in ("GiB", "MiB") else int(v)
        if key == "MEM_CEILING_GIB" and mem is not None:
            # the configured figure plus what the side slots hold for good (`Budget.fixed`):
            # assigning the bare figure took the side slots' 7.65 GiB out of the caches' room
            # on every start of 0.3.44 to 0.3.46, at the first request
            mem.ceiling = b + int(getattr(mem, "fixed", 0) or 0)
        elif key == "MEM_PEAK_GIB" and mem is not None:
            mem.peak = b + int(getattr(mem, "fixed", 0) or 0)
        elif key == "RESERVE_SLACK_GIB" and mem is not None:
            mem.slack = b
        elif key == "RESIDENT_GB" and res is not None:
            res.set_limits(budget=b)
        elif key == "RESIDENT_STASH_GB" and res is not None:
            res.set_limits(stash=b)
        elif key == "RESIDENT_GUEST_MAX" and res is not None:
            res.set_limits(guest_max=b)
        elif key == "RESIDENT_PIN_GB" and res is not None:
            res.set_limits(pin=b)
        elif key == "RESIDENT_PARK_GB" and res is not None:
            res.set_limits(park=b)
        elif key == "CACHE_GB" and store is not None:
            store.set_limits(budget_bytes=b)
        elif key == "STORE_ENTRY_GB" and store is not None:
            store.set_limits(max_entry_bytes=b)
        elif key == "VISION_EMBED_MB" and state.get("image_cache") is not None:
            ic = state["image_cache"]
            ic.budget = b
            while ic.bytes > ic.budget and ic._d:
                _, o = ic._d.popitem(last=False)
                ic.bytes -= o.numel() * o.element_size()
                ic.stats["evictions"] += 1
        elif key == "SUFFIX_STORE_MB" and state.get("suffix_store") is not None:
            state["suffix_store"].max_tokens = b // 4      # applies at its next rebuild
        applied[key] = v
        done.append(f"{key}={v}")
    return done


def refresh(state: dict) -> list[str]:
    """A request starts: when the residency document has a new version, its live overrides
    apply now (and a key the document no longer sets goes back to its flag)."""
    cond = state.get("conductor")
    if cond is None:
        return []
    try:
        v, doc = cond.store.peek("residency")
    except Exception:                                              # noqa: BLE001
        return []
    if v == state.get("limits_version"):
        return []
    state["limits_version"] = v
    over = doc_section(doc)
    flags = state.get("serve_flags") or state.get("args") or {}
    values = {k: over.get(k, flag_value(flags, k)) for k in KEYS if _BY_KEY[k][2]}
    done = apply(state, {k: x for k, x in values.items() if x is not None})
    if done:
        print(f"[memory] residency v{v}: " + ", ".join(done), flush=True)
    return done


def _usage(state: dict, key: str):
    """(bytes or count it holds now, its cap in the same unit) for the panel; None when the part
    is off on this server."""
    mem, store, res = state.get("mem"), state.get("state_store"), state.get("resident")
    if key in ("MEM_CEILING_GIB", "RESERVE_SLACK_GIB", "MEM_PEAK_GIB"):
        if mem is None:
            return None
        alloc, reserved = mem.memory()
        if key == "MEM_CEILING_GIB":
            return reserved, mem.ceiling
        if key == "RESERVE_SLACK_GIB":
            return reserved - alloc, mem.slack
        return mem.stats.get("prefill_peak") or 0, mem.peak
    if res is not None:
        if key == "RESIDENT_GB":
            return res.anchor_bytes, res.budget
        if key == "RESIDENT_STASH_GB":
            return res.stash_bytes, res.stash_budget
        if key == "RESIDENT_GUEST_MAX":
            return None
        if key == "RESIDENT_PIN_GB":
            return res.pin_bytes, res.pin_budget
        if key == "RESIDENT_PARK_GB":
            return res.park_bytes, res.park_budget
    if store is not None:
        if key == "CACHE_GB":
            return store.bytes, store.budget
        if key == "STORE_ENTRY_GB":
            biggest = max((e.snap.nbytes for e in list(store._d.values())), default=0)
            return biggest, store.max_entry
    if key == "VISION_EMBED_MB" and state.get("image_cache") is not None:
        ic = state["image_cache"]
        return ic.bytes, ic.budget
    if key == "SUFFIX_STORE_MB" and state.get("suffix_store") is not None:
        st = state["suffix_store"]
        return st.n_tokens * 4, st.max_tokens * 4
    return None


def engine_gib(state: dict) -> tuple:
    """(declared peak, what the engine holds now) in GiB for the residency planner; (None, None)
    when this server has no memory policy (the fake engine), and the planner then counts none.
    The peak includes the disk tier's staging buffer (C1) until it is allocated: host memory the
    engine will take at its first disk read or write, which no allocator figure shows yet."""
    mem = state.get("mem")
    if mem is None or not mem.on:
        return None, None
    disk = state.get("disk_cache")
    peak = mem.peak + (disk.staging_pending() if disk is not None else 0)
    try:
        _, res = mem.memory()
    except Exception:                                              # noqa: BLE001
        return round(peak / GIB, 2), None
    return round(peak / GIB, 2), round(res / GIB, 2)


def _decider_bytes(state: dict) -> int:
    reg = state.get("deciders")
    try:
        return int(sum(s["device_bytes"] for s in reg.status())) if reg is not None else 0
    except Exception:                                              # noqa: BLE001
        return 0


def _role_pools(state: dict) -> list:
    """(slot, pool reserved, pool allocated, claim bytes or None) of each in-process generative
    role -- the narrator, the shadow reader and the setup guide's Wren, one row for a slot they
    share."""
    out, seen = [], set()
    for owner in (state.get("narrator"), state.get("shadow"), state.get("wren")):
        slot = getattr(owner, "slot", None) if owner is not None else None
        if slot is None or id(slot) in seen:
            continue
        seen.add(id(slot))
        res = alloc = 0
        if hasattr(slot, "pool_bytes"):
            try:
                res, alloc = (int(x) for x in slot.pool_bytes())
            except Exception:                                      # noqa: BLE001
                res = alloc = 0
        claim = getattr(slot, "claim_gib", None)
        try:
            claim = int(float(claim) * GIB) if claim else None
        except (TypeError, ValueError):
            claim = None
        out.append((slot, res, alloc, claim))
    return out


def role_exclude(state: dict) -> tuple[int, int]:
    """(allocated, reserved) bytes of the in-process roles, left out of the engine's figures: the
    decision models' tensors, and each generative role's own pool up to the claim the residency
    planner admitted it with (its weights, its KV, its graph, the shadow's session states). A
    pool over its claim counts the excess as the engine's, which the gate's `roles` level can give
    back. A role without a pool or a claim (the CPU, a slot loaded without the planner) leaves out
    its model's device bytes, as before 0.3.34.

    Before it, the shadow reader's own slot was not left out at all and the
    narrator's only by its model's tensors; the shadow's first load after a restart pushed the
    engine over its ceiling and the gate shed the anchors, the end, the stash and the park
    (6.3 GB on 2026-10-05 06:18, for a 1.47 GiB model)."""
    a = r = _decider_bytes(state)
    for slot, res, alloc, claim in _role_pools(state):
        if res > 0 and claim:
            a += min(alloc, claim)
            r += min(res, claim)
        else:
            try:
                n = int(slot.device_bytes())
            except Exception:                                      # noqa: BLE001
                n = 0
            a += n
            r += n
    return a, r


def _role_bytes(state: dict) -> int:
    """The roles' reserved bytes left out of the engine's (role_exclude's second figure)."""
    return role_exclude(state)[1]


def role_pool_free(state: dict) -> int:
    """Free bytes inside the roles' own pools (reserved minus allocated) that still count in the
    engine's figures: memory `reserved` shows that the engine's allocator cannot use and only
    dropping the pool returns. A pool's part under its claim is left out by `role_exclude`, so
    only its excess counts here."""
    n = 0
    for slot, res, alloc, claim in _role_pools(state):
        if claim:
            n += max(0, (res - min(res, claim)) - (alloc - min(alloc, claim)))
        else:
            n += max(0, res - alloc)
    return n


def view(state: dict) -> dict:
    """The Memory plan panel's limits: one row per key, and the switch the panel quotes."""
    args = state.get("args") or {}                 # what the engine loaded with
    flags = state.get("serve_flags") or args       # what serve.env and the command line said
    applied = state.get("limits_applied") or {}
    cond = state.get("conductor")
    over, version = {}, None
    if cond is not None:
        try:
            version, doc = cond.store.peek("residency")
            over = doc_section(doc)
        except Exception:                                          # noqa: BLE001
            pass
    # a model whose caches keep its own grid (its prefill chunk, `caps.anchor_grid`) runs
    # that grid whatever serve.env says: the row shows it, locked, and no restart is pending
    grid = state.get("anchor_grid")
    locked = dict(LOCKED, **({"PREFIX_CHUNK": f"the model's own grid: {int(grid):,} tokens, its "
                                              f"prefill chunk"} if grid else {}))
    rows = []
    for key, dest, unit, live, what in LIMITS:
        flag = flag_value(flags, key)
        value = over.get(key, flag)
        running = applied.get(key, flag_value(args, key)) if live else flag_value(args, key)
        if key == "PREFIX_CHUNK" and grid:
            value = running = int(grid)
        u = _usage(state, key)
        rows.append({
            "key": key, "unit": unit, "what": what, "value": value, "flag": flag,
            "source": "dashboard" if key in over else "serve.env",
            "applies": "locked" if key in locked else ("next request" if live else "restart"),
            "locked": locked.get(key),
            "running": running,
            "pending_restart": (not live and key not in locked and value != running),
            # bytes for a GiB or MiB key, a count otherwise; None where the part is off
            "used": u[0] if u else None,
            "cap": u[1] if u else None,
        })
    mem = state.get("mem")
    res = state.get("resident")
    return {"version": version, "rows": rows,
            "budget": mem.report() if mem is not None else None,
            "workspace_at_max_len": mem.workspace(state.get("max_len") or 0)
            if mem is not None else None,
            "switch": (res.switches[-1] if res is not None and res.switches else None),
            "switches": res.stats.get("switches", 0) if res is not None else 0,
            "park": ({"owner": res.report()["park_owner"], "rows": res.park_rows,
                      "base": res.park_base, "bytes": res.park_bytes}
                     if res is not None else None),
            "owner": res.report()["owner"] if res is not None else None}
