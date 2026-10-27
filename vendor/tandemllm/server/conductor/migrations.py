"""Versioned changes to the conductor documents a box already stores, run once each at the engine's
start (after the shipped defaults are stored, server/conductor/shipped.py).

A shipped default reaches only a box that has no version of the document; a fix to a stored one
needs a migration. Each migration has an id, the document it reads, and a function that returns
the changed document, or None when there is nothing to change. A change is a new version of the
document (note `migration <id>`, author `migration`), so it is visible in the version list and
undone by a restore; the run is one log line and one journal row (`config`, decision
`migration`). `<conductor root>/migrations.json` records every id that ran, changed or not, so a
migration never runs twice: a switch the owner removes after it ran stays removed.

    2026-10-14-chat-app-memory   SRV-471: the chat-app profile's `memory` switch, "on" when the
                                 stored document has none (absent is off); an explicit value,
                                 "off" included, stays
    2026-10-14-chat-memory       SRV-471: the same for the chat, coding and hermes profiles
    2026-10-14-coding-memory
    2026-10-14-hermes-memory
    2026-10-14-<name>-loop-guard SRV-473, SRV-475: in coding, chat-app, hermes, overnight and chat,
                                 `toolgate.same_call` on when it is off or absent and no stored
                                 version of the profile ever had it on (a profile someone turned
                                 off after it was on stays off); chat's C3 to act by the same rule
                                 on its mode; `sampling` (greedy, loop escape on) when the
                                 document has none. The record lists what changed
    2026-10-17-coding-narrator-off  SRV-534: the narrator off by default (0.11.3):
    2026-10-17-chat-narrator-off    `ui.narrator` from "live" (0.3.28's shipped value) to "off"
                                 when no stored version of the profile set another value
                                 ("off", "on", "thoughts"; absent is the key's age, no choice)
    2026-10-17-roles-narrator-off   the same for the roles document's `narrator.mode`, "on" to
                                 "off" unless a stored version had "off" or "shadow"
    2026-10-17-residency-decider    the decider's claim 8.2 GiB (the arbiter sidecar's) to the 3.0
                                 measured with Laya in the engine (OPS-69), unless a stored
                                 version had another value
    2026-10-17-router-coding-core   the router's code_edit and debug classes get FX6's core
                                 ["coding"] when the stored class has no `core` and no stored
                                 version gave it one: under a model fit with `packages: core`
                                 (NemotronH, LFM2.5) a coding turn, decided or fallen back,
                                 got no package at all on a router stored before FX6
    2026-10-17-router-build-core    the same for design_ui (["ui-design"]) and drawing
                                 (["creative-coding"]), which had no core: under that fit a turn
                                 Laya decided as one of them got no package (a three.js prompt
                                 read as design_ui, 2026-10-17)
    2026-10-17-router-3d-coding-core  three_d's core gains `coding` (and its packages list it),
                                 unless a stored version had it in the core: three pagoda
                                 replays thought 6 to 11 minutes on their first step, drafting
                                 the files in the thinking
    2026-10-19-<name>-c5         SRV-623: C5 (one mutation per turn) in a stored profile that has
                                 neither the task nor `toolgate.sequence` (both absent: no one
                                 chose): shadow in coding, chat-app, hermes and chat, act in
                                 tools, as shipped
    2026-10-20-packages-deliverable  SRV-615: the shipped `deliverable` of write-message,
                                 formal-letter (send), event-plan, weekly-schedule,
                                 checklist-builder (schedule) and social-post (post) on a stored
                                 packages document, for each package it has without one, unless
                                 a stored version gave that package one (someone took it out)
"""

from __future__ import annotations

import datetime as _dt
import json
import os

STATE = "migrations.json"


def _memory_on(doc: dict) -> dict | None:
    if not isinstance(doc, dict) or "memory" in doc:
        return None
    return dict(doc, memory="on")


def _ever(past: list, get) -> set:
    """The values `get` reads from each earlier stored version (version 0, the built-in seed, is
    the code's default and no one's choice: not among them)."""
    out = set()
    for d in past:
        try:
            out.add(get(d))
        except (KeyError, TypeError, AttributeError):
            out.add(None)
    return out


def _loop_guard(doc: dict, past: list, c3: bool = False):
    """SRV-473/SRV-475: the same-call guard on and the sampling section, as the shipped profile
    has them. `past`: the profile's stored versions, oldest first. `toolgate.same_call` false and
    true-never-stored is the old shipped default (absent: the same), so it is turned on; a profile
    that was on in any stored version was turned off by someone, and stays as it is. With `c3`
    (chat), C3's mode goes from off to act by the same rule. Returns (document, changes) or None."""
    if not isinstance(doc, dict):
        return None
    new, changes = dict(doc), []
    gate = dict(doc.get("toolgate") or {})
    if gate.get("same_call") is not True and True not in _ever(
            past, lambda d: d["toolgate"]["same_call"]):
        gate["same_call"] = True
        new["toolgate"] = gate
        changes.append("toolgate.same_call=true")
    tasks = doc.get("tasks") or {}
    if c3 and (tasks.get("C3") or {}).get("mode") == "off" and _ever(
            past, lambda d: d["tasks"]["C3"]["mode"]) <= {"off"}:
        new["tasks"] = dict(tasks, C3=dict(tasks["C3"], mode="act"))
        changes.append("tasks.C3.mode=act")
    if "sampling" not in doc:
        new["sampling"] = {"agentic_default": "greedy", "loop_escape": "on"}
        changes.append("sampling={agentic_default: greedy, loop_escape: on}")
    return (new, changes) if changes else None


def _guard(c3: bool = False):
    def change(doc, past):
        return _loop_guard(doc, past, c3=c3)
    change.history = True                    # `run` passes the stored versions
    return change


def _unless_chosen(get, old, new_value, put, what: str):
    """A change of a value from its old shipped one to the new default, unless a stored version
    held another value (someone chose); a missing key is no choice. `put(doc, value)` returns the
    changed document."""
    def change(doc, past):
        try:
            cur = get(doc)
        except (KeyError, TypeError, AttributeError):
            return None
        if cur != old or _ever(past, get) - {old, None}:
            return None
        return put(doc, new_value), [f"{what}={new_value}"]
    change.history = True
    return change


def _ui_narrator(doc, value):
    return dict(doc, ui=dict(doc["ui"], narrator=value))


def _role_narrator(doc, value):
    return dict(doc, narrator=dict(doc["narrator"], mode=value))


def _decider_claim(doc, value):
    claim = dict(doc["claims"]["decider"], gib=value, basis="measured",
                 note="Laya's three checkpoints in the engine's process, measured beside a side "
                      "slot (OPS-69, 2026-10-06); the arbiter sidecar took 8.2")
    return dict(doc, claims=dict(doc["claims"], decider=claim))


def _class_core(cores: dict):
    """A core on router classes stored without one: `cores` maps a class to its core package,
    given when the stored class lists that package, has no `core`, and no stored version gave
    the class a core (a core someone removed stays removed)."""
    def change(doc: dict, past: list):
        if not isinstance(doc, dict) or not isinstance(doc.get("classes"), list):
            return None

        def core_of(d, cid):
            return next((c.get("policy") or {}).get("core") for c in d["classes"] if c["id"] == cid)

        classes, changes = [], []
        for c in doc["classes"]:
            pol = c.get("policy") or {}
            pkg = cores.get(c.get("id"))
            if pkg and "core" not in pol and pkg in (pol.get("packages") or []) and \
                    not (_ever(past, lambda d, i=c["id"]: core_of(d, i) is not None) - {False, None}):
                c = dict(c, policy=dict(pol, core=[pkg]))
                changes.append(f"classes.{c['id']}.core=[{pkg}]")
            classes.append(c)
        return (dict(doc, classes=classes), changes) if changes else None
    change.history = True
    return change


def _core_append(cid: str, pkg: str):
    """`pkg` added to class `cid`'s core (and its packages) of a stored router, unless the core has
    it or a stored version had it there (someone took it out)."""
    def change(doc: dict, past: list):
        if not isinstance(doc, dict) or not isinstance(doc.get("classes"), list):
            return None

        def had(d):
            return pkg in (next((c.get("policy") or {}).get("core") or []
                                for c in d["classes"] if c["id"] == cid))
        classes, changes = [], []
        for c in doc["classes"]:
            pol = c.get("policy") or {}
            core = list(pol.get("core") or [])
            if c.get("id") == cid and pkg not in core and True not in _ever(past, had):
                pkgs = list(pol.get("packages") or [])
                c = dict(c, policy=dict(pol, core=core + [pkg],
                                        packages=pkgs + ([pkg] if pkg not in pkgs else [])))
                changes.append(f"classes.{cid}.core+={pkg}")
            classes.append(c)
        return (dict(doc, classes=classes), changes) if changes else None
    change.history = True
    return change


def _c5(mode: str):
    """SRV-623: C5 and its pack, as the shipped profile has them, when the stored profile has
    neither (a profile stored before C5 existed)."""
    def change(doc: dict):
        if not isinstance(doc, dict) or not isinstance(doc.get("tasks"), dict) or \
                "C5" in doc["tasks"] or "sequence" in (doc.get("toolgate") or {}):
            return None
        task = {"mode": mode, "cap": 1, "thresholds": {"regenerations": 1}}
        return (dict(doc, tasks=dict(doc["tasks"], C5=task),
                     toolgate=dict(doc.get("toolgate") or {}, sequence=True)),
                [f"tasks.C5.mode={mode}", "toolgate.sequence=true"])
    return change


# SRV-615: the packages whose result a tool can deliver, as packages/seed.json ships them
DELIVERABLE = {"write-message": "send", "formal-letter": "send", "event-plan": "schedule",
               "weekly-schedule": "schedule", "checklist-builder": "schedule",
               "social-post": "post"}


def _deliverable(doc: dict, past: list):
    """SRV-615: `deliverable` on the stored packages that have none and never had one."""
    if not isinstance(doc, dict) or not isinstance(doc.get("packages"), dict):
        return None

    def had(name):
        return lambda d: "deliverable" in ((d.get("packages") or {}).get(name) or {})
    pkgs, changes = dict(doc["packages"]), []
    for name, want in DELIVERABLE.items():
        p = pkgs.get(name)
        if isinstance(p, dict) and "deliverable" not in p and True not in _ever(past, had(name)):
            pkgs[name] = dict(p, deliverable=want)
            changes.append(f"packages.{name}.deliverable={want}")
    return (dict(doc, packages=pkgs), changes) if changes else None


_deliverable.history = True

# SRV-534: FX6's core on the coding classes of a router stored without it
_coding_core = _class_core({"code_edit": "coding", "debug": "coding"})

_NARRATOR_OFF = _unless_chosen(lambda d: d["ui"]["narrator"], "live", "off", _ui_narrator,
                               "ui.narrator")

# (id, kind, name, change): `change(doc)`, or `change(doc, past)` when it has `history`; it
# returns the new document, (the document, [what changed]), or None
MIGRATIONS = [
    ("2026-10-14-chat-app-memory", "profiles", "chat-app", _memory_on),
    ("2026-10-14-chat-memory", "profiles", "chat", _memory_on),
    ("2026-10-14-coding-memory", "profiles", "coding", _memory_on),
    ("2026-10-14-hermes-memory", "profiles", "hermes", _memory_on),
    ("2026-10-14-coding-loop-guard", "profiles", "coding", _guard()),
    ("2026-10-14-chat-app-loop-guard", "profiles", "chat-app", _guard()),
    ("2026-10-14-hermes-loop-guard", "profiles", "hermes", _guard()),
    ("2026-10-14-overnight-loop-guard", "profiles", "overnight", _guard()),
    ("2026-10-14-chat-loop-guard", "profiles", "chat", _guard(c3=True)),
    ("2026-10-17-coding-narrator-off", "profiles", "coding", _NARRATOR_OFF),
    ("2026-10-17-chat-narrator-off", "profiles", "chat", _NARRATOR_OFF),
    ("2026-10-17-roles-narrator-off", "roles", None,
     _unless_chosen(lambda d: d["narrator"]["mode"], "on", "off", _role_narrator,
                    "narrator.mode")),
    ("2026-10-17-residency-decider", "residency", None,
     _unless_chosen(lambda d: d["claims"]["decider"]["gib"], 8.2, 3.0, _decider_claim,
                    "claims.decider.gib")),
    ("2026-10-17-router-coding-core", "router", None, _coding_core),
    ("2026-10-17-router-build-core", "router", None,
     _class_core({"design_ui": "ui-design", "drawing": "creative-coding"})),
    ("2026-10-17-router-3d-coding-core", "router", None, _core_append("three_d", "coding")),
    ("2026-10-19-coding-c5", "profiles", "coding", _c5("shadow")),
    ("2026-10-19-chat-app-c5", "profiles", "chat-app", _c5("shadow")),
    ("2026-10-19-hermes-c5", "profiles", "hermes", _c5("shadow")),
    ("2026-10-19-chat-c5", "profiles", "chat", _c5("shadow")),
    ("2026-10-19-tools-c5", "profiles", "tools", _c5("act")),
    ("2026-10-20-packages-deliverable", "packages", None, _deliverable),
]


def _past(store, kind: str, name: str, current: int) -> list:
    """The stored versions below `current`, oldest first (unreadable ones left out)."""
    out = []
    for meta in sorted(store.versions(kind, name), key=lambda m: m.get("version") or 0):
        v = meta.get("version") or 0
        if 0 < v < current:
            try:
                out.append(store.version(kind, name, v))
            except Exception:                                      # noqa: BLE001
                continue
    return out


def _state(root: str) -> dict:
    try:
        with open(os.path.join(root, STATE)) as f:
            got = json.load(f)
        return got if isinstance(got, dict) and isinstance(got.get("applied"), dict) else \
            {"applied": {}}
    except (OSError, ValueError):
        return {"applied": {}}


def _save(root: str, state: dict) -> None:
    os.makedirs(root, mode=0o700, exist_ok=True)
    path = os.path.join(root, STATE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def run(store, journal=None, migrations=None) -> list[dict]:
    """Every migration of `migrations` (default MIGRATIONS) that has not run on this store's
    root. Returns one record per migration run now: {id, kind, name, result, version}; result is
    `changed`, `unchanged` (the document already had it) or `absent` (no stored version; the
    shipped default, when stored later, has it). A read-only store runs nothing."""
    if getattr(store, "readonly", False):
        return []
    state = _state(store.root)
    done = []
    for mid, kind, name, change in (MIGRATIONS if migrations is None else migrations):
        if mid in state["applied"]:
            continue
        rec = {"id": mid, "kind": kind, "name": name, "result": "absent", "version": None}
        try:
            if store._pointer(store._dir(kind, name)) is not None:
                v, doc = store.get(kind, name)
                new = (change(doc, _past(store, kind, name, v)) if getattr(change, "history", False)
                       else change(doc))
                if isinstance(new, tuple):
                    new, rec["changes"] = new
                if new is None:
                    rec.update(result="unchanged", version=v)
                else:
                    # the change leaves the tasks as they are, so the editor's check that an act
                    # task's role is up (offline at the start) does not apply to it
                    up, store.roles_up = store.roles_up, None
                    try:
                        rec.update(result="changed", version=store.put(
                            kind, name, new, note=f"migration {mid}", author="migration", base=v))
                    finally:
                        store.roles_up = up
        except Exception as exc:                                   # noqa: BLE001
            # not recorded as run: the next start tries again
            rec.update(result="failed", error=f"{type(exc).__name__}: {exc}"[:300])
            print(f"[conductor] migration {mid} failed: {rec['error']}", flush=True)
            done.append(rec)
            continue
        state["applied"][mid] = {"at": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"), "result": rec["result"], "version": rec["version"]}
        if rec.get("changes"):
            state["applied"][mid]["changes"] = rec["changes"]
        _save(store.root, state)
        print(f"[conductor] migration {mid}: {kind}{'/' + name if name else ''} "
              f"{rec['result']}" + (f" (v{rec['version']})" if rec["version"] else "")
              + (f": {'; '.join(rec['changes'])}" if rec.get("changes") else ""), flush=True)
        if journal is not None:
            try:
                journal.note("config", kind, decision="migration",
                             detail={"id": mid, "name": name, "result": rec["result"],
                                     "version": rec["version"],
                                     **({"changes": rec["changes"]} if rec.get("changes")
                                        else {})})
            except Exception:                                      # noqa: BLE001
                pass
        done.append(rec)
    return done
