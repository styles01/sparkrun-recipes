"""SRV-471 (QA SRV-472): a shipped profile derived from another keeps the parent's top-level
switches, and a box's stored chat-app profile gets the memory switch by a versioned migration.

The test matrix (SRV-471's Gherkin, rows 4 and 5; rows 1 to 3 are in test_memory_router.py):

  1  every derived shipped profile equals its parent on every top-level key but the ones it
     declares, and sets each declared one itself (chat-app: memory "on")
                                                test_derived_profiles_keep_the_parents_switches
  2  a switch the parent gains that the derived profile misses is caught
                                                test_a_switch_the_parent_gains_is_caught
  3  `general` keeps no memory switch (scripts and SDK clients: the release's request); chat,
     coding and hermes ship memory "on" explicitly
                                                test_general_has_no_memory_switch
  4  a stored chat-app document without the key: a new version with memory "on", journaled,
     run once; an explicit "off" stays; one that has it (the live box's v2) is left alone;
     no stored version: nothing written; a read-only store: nothing run
                                                test_the_migration_*
  5  the same for stored chat, coding and hermes documents; a box whose chat, coding and
     chat-app have the switch changes only hermes, and general is never touched
                                                test_the_profile_migrations_on_a_live_box

Run: python tests/test_conductor_derived.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor import migrations, schema  # noqa: E402
from server.conductor.store import Store  # noqa: E402

MID = "2026-10-14-chat-app-memory"
ONLY_CHAT_APP = [m for m in migrations.MIGRATIONS if m[0] == MID]
MEMORY = [m for m in migrations.MIGRATIONS if m[0].endswith("-memory")]


def drift(profiles: dict, derived: dict) -> list[str]:
    """What a derived profile gets wrong: a top-level key other than its own that differs from the
    parent's, or one of its own it does not set."""
    out = []
    for name, (parent, own) in derived.items():
        d, p = profiles[name], profiles[parent]
        for k in sorted(set(d) | set(p) | set(own)):
            if k in own:
                if k not in d:
                    out.append(f"{name}.{k}: declared but not set")
            elif d.get(k, "<absent>") != p.get(k, "<absent>"):
                out.append(f"{name}.{k}: {d.get(k, '<absent>')!r} != {parent}'s "
                           f"{p.get(k, '<absent>')!r}")
    return out


# chat-app's own keys, as SRV-471 decided them: the regression guard does not read them from the
# schema, so the schema before the fix (no memory key on chat-app) fails here
# SRV-621: `tools` is general with its own decisions, tasks, tool gate, state lines, sampling,
# prompt and transcript switches; memory stays absent
DERIVED = {"chat-app": ("coding", ("description", "clients", "ui", "memory")),
           "tools": ("general", ("description", "decisions", "tasks", "toolgate", "ui",
                                 "sampling", "prompt", "transcripts", "secret_mask"))}


def test_derived_profiles_keep_the_parents_switches():
    assert drift(schema.SHIPPED_PROFILES, DERIVED) == []
    assert schema.SHIPPED_PROFILES["chat-app"]["memory"] == "on"
    assert getattr(schema, "DERIVED_PROFILES", None) == DERIVED
    assert not schema.check("profiles", schema.SHIPPED_PROFILES["chat-app"], name="chat-app",
                            router=schema.SHIPPED_ROUTER, packages=schema.SHIPPED_PACKAGES,
                            profiles=schema.SHIPPED_PROFILES)


def test_a_switch_the_parent_gains_is_caught():
    profs = copy.deepcopy(schema.SHIPPED_PROFILES)
    profs["coding"]["library"] = "on"                    # coding changes a switch later
    assert drift(profs, DERIVED) == ["chat-app.library: 'shadow' != coding's 'on'"]
    profs = copy.deepcopy(schema.SHIPPED_PROFILES)
    profs["coding"]["hints"] = dict(profs["coding"]["hints"], stair_width=True)
    assert drift(profs, DERIVED)[0].startswith("chat-app.hints:")
    profs = copy.deepcopy(schema.SHIPPED_PROFILES)
    del profs["chat-app"]["memory"]                      # the pre-fix document
    assert drift(profs, DERIVED) == ["chat-app.memory: declared but not set"]
    profs = copy.deepcopy(schema.SHIPPED_PROFILES)
    profs["general"]["memory"] = "on"                    # general gains a switch later
    assert drift(profs, DERIVED) == ["tools.memory: '<absent>' != general's 'on'"]


def test_the_agent_profiles_ship_the_loop_guard_and_the_escape():
    """SRV-473 / SRV-475: chat-app gets both from coding (no own key), hermes, overnight and chat
    set them, general keeps neither."""
    assert drift(schema.SHIPPED_PROFILES, DERIVED) == []
    for name in ("coding", "chat-app", "hermes", "overnight", "chat"):
        p = schema.SHIPPED_PROFILES[name]
        assert p["toolgate"]["same_call"] is True, name
        assert p["tasks"]["C3"]["mode"] == "act", name
        assert p["sampling"] == {"agentic_default": "greedy", "loop_escape": "on"}, name
        assert not schema.check("profiles", p, name=name, router=schema.SHIPPED_ROUTER,
                                packages=schema.SHIPPED_PACKAGES,
                                profiles=schema.SHIPPED_PROFILES), name
    g = schema.SHIPPED_PROFILES["general"]
    assert g["toolgate"]["same_call"] is False and "sampling" not in g
    # the other packs stay as they were
    for name in ("coding", "chat-app", "hermes", "overnight", "chat", "general"):
        tg = schema.SHIPPED_PROFILES[name]["toolgate"]
        assert (tg["risk"], tg["project"], tg["malformed"]) == (False, False, False), name


def test_general_has_no_memory_switch():
    assert "memory" not in schema.SHIPPED_PROFILES["general"]
    assert schema.profile_memory(schema.SHIPPED_PROFILES["general"]) == "off"
    for name in ("chat", "coding", "hermes", "chat-app"):
        assert schema.SHIPPED_PROFILES[name]["memory"] == "on", name


# ---------------------------------------------------------------------- the migration
class _Journal:
    def __init__(self):
        self.rows = []

    def note(self, kind, task, *, decision=None, detail=None, **_):
        self.rows.append((kind, task, decision, detail))


def _box(memory=...):
    """A store with the shipped documents and a chat-app version as a box stored it before the
    fix (no memory key), or with `memory` as given."""
    root = tempfile.mkdtemp(prefix="cond-mig-")
    st = Store(root)
    for kind, name, doc in (("packages", None, schema.SHIPPED_PACKAGES),
                            ("router", None, schema.SHIPPED_ROUTER),
                            ("profiles", "coding", schema.SHIPPED_PROFILES["coding"])):
        st.put(kind, name, copy.deepcopy(doc), note="shipped", author="test")
    app = copy.deepcopy(schema.SHIPPED_PROFILES["chat-app"])
    del app["memory"]
    if memory is not ...:
        app["memory"] = memory
    st.put("profiles", "chat-app", app, note="shipped defaults (0.8.0)", author="shipped")
    return st


def test_the_migration_adds_memory_on_once_and_journals_it():
    st = _box()
    j = _Journal()
    got = migrations.run(st, j, ONLY_CHAT_APP)
    assert got == [{"id": MID, "kind": "profiles", "name": "chat-app", "result": "changed",
                    "version": 2}]
    v, doc = st.get("profiles", "chat-app")
    assert v == 2 and doc["memory"] == "on"
    meta = st.versions("profiles", "chat-app")[0]
    assert meta["note"] == f"migration {MID}" and meta["author"] == "migration"
    assert j.rows == [("config", "profiles", "migration",
                       {"id": MID, "name": "chat-app", "result": "changed", "version": 2})]
    with open(os.path.join(st.root, migrations.STATE)) as f:
        assert json.load(f)["applied"][MID]["result"] == "changed"
    # run once: the owner removes the switch afterwards, the next start leaves it removed
    doc.pop("memory")
    st.put("profiles", "chat-app", doc, note="memory as before", author="owner")
    assert migrations.run(st, j, ONLY_CHAT_APP) == [] and "memory" not in st.get("profiles", "chat-app")[1]


def test_the_migration_keeps_an_explicit_off_and_the_live_boxs_on():
    for value in ("off", "on", "shadow"):
        st = _box(memory=value)
        got = migrations.run(st, migrations=ONLY_CHAT_APP)
        assert got[0]["result"] == "unchanged" and got[0]["version"] == 1, (value, got)
        v, doc = st.get("profiles", "chat-app")
        assert (v, doc["memory"]) == (1, value)


def test_the_migration_writes_nothing_without_a_stored_version():
    st = Store(tempfile.mkdtemp(prefix="cond-mig-"))
    got = migrations.run(st)
    assert [r["result"] for r in got] == ["absent"] * len(migrations.MIGRATIONS)
    assert not os.path.isdir(os.path.join(st.root, "profiles"))
    assert migrations.run(st) == []


def test_the_migration_passes_an_offline_role_and_skips_a_read_only_store():
    st = _box()
    st.roles_up = lambda: {}                             # every role offline at the start
    _, doc = st.get("profiles", "chat-app")
    one = [t for t, d in schema.TASKS.items() if d["tier"] == "1" and d["role"]][0]
    doc["tasks"][one]["mode"] = "act"
    up, st.roles_up = st.roles_up, None
    st.put("profiles", "chat-app", doc, note="act", author="test")
    st.roles_up = up
    assert migrations.run(st, migrations=ONLY_CHAT_APP)[0]["result"] == "changed"
    assert st.roles_up is up
    ro = Store(_box().root, readonly=True)
    assert migrations.run(ro) == []


def test_the_profile_migrations_on_a_live_box():
    """The shape of a box that has run the Chat tab since 0.8.0: chat, coding and chat-app were
    given the switch by hand, hermes and general were stored as shipped before it."""
    st = Store(tempfile.mkdtemp(prefix="cond-mig-"))
    for name in ("chat", "coding", "chat-app", "hermes", "general"):
        doc = copy.deepcopy(schema.SHIPPED_PROFILES[name])
        doc.pop("memory", None)
        if name in ("chat", "coding", "chat-app"):
            doc["memory"] = "on"
        st.put("profiles", name, doc, note="as stored", author="test")
    j = _Journal()
    got = {r["name"]: (r["result"], r["version"]) for r in migrations.run(st, j, MEMORY)}
    assert got == {"chat-app": ("unchanged", 1), "chat": ("unchanged", 1),
                   "coding": ("unchanged", 1), "hermes": ("changed", 2)}
    assert st.get("profiles", "hermes")[1]["memory"] == "on"
    assert st.get("profiles", "general")[0] == 1 and "memory" not in st.get("profiles",
                                                                            "general")[1]
    assert len(j.rows) == 4 and migrations.run(st, j, MEMORY) == []
    # an explicit off in a stored coding document stays
    st = Store(tempfile.mkdtemp(prefix="cond-mig-"))
    st.put("profiles", "coding", dict(copy.deepcopy(schema.SHIPPED_PROFILES["coding"]),
                                      memory="off"), note="off", author="owner")
    rec = [r for r in migrations.run(st, migrations=MEMORY) if r["name"] == "coding"][0]
    assert rec["result"] == "unchanged" and st.get("profiles", "coding")[1]["memory"] == "off"


def test_a_failed_migration_runs_again_next_start():
    st = _box()

    def broken(doc):
        raise ValueError("nope")
    got = migrations.run(st, migrations=[("x-broken", "profiles", "chat-app", broken)])
    assert got[0]["result"] == "failed" and "nope" in got[0]["error"]
    assert migrations.run(st, migrations=[("x-broken", "profiles", "chat-app",
                                           lambda d: None)])[0]["result"] == "unchanged"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:<64} ok")
            passed += 1
    print(f"{passed} passed")
