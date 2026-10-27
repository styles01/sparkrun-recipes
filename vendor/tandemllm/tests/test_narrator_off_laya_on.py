"""SRV-534 (QA SRV-535): the narrator off by default, Laya kept loaded, and the documents a box
already stores brought along by the conductor's migrations.

SRV-534's Gherkin, row by row (the router's fallback is tests/test_conductor_modelfit.py 2 and 2b,
the residency plan and the floor math tests/test_residency.py, the narrator's load route
tests/test_narrator.py `test_the_routes`):

  1. a new box: the narrator role is off, no shipped profile turns the narrator on, the residency
     plan holds no claim for it, no model card turns it on in the Setup Guide;
  2. a box that stored the old shipped values gets the new defaults once; a value someone chose
     stays (`ui.narrator` set to another value before, the role's mode, the decider's claim);
  3. Laya refused while no model is loaded: /health's roles.laya says why, with the planner's
     numbers; once a model loads, nothing.

Run: python tests/test_narrator_off_laya_on.py
"""

from __future__ import annotations

import copy
import glob
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import yaml  # noqa: E402

from server.conductor import migrations, residency, schema  # noqa: E402
from server.conductor.store import Store  # noqa: E402

NARRATOR = [m for m in migrations.MIGRATIONS if m[0].startswith("2026-10-17-")
            and m[2] != "router" and m[1] != "router"]
CORE = [m for m in migrations.MIGRATIONS if m[0] == "2026-10-17-router-coding-core"]
BUILD_CORE = [m for m in migrations.MIGRATIONS if m[0] == "2026-10-17-router-build-core"]
CODING_3D = [m for m in migrations.MIGRATIONS if m[0] == "2026-10-17-router-3d-coding-core"]


# ---------------------------------------------------------------------- 1: a new box
def test_1_a_new_box_has_no_narrator():
    roles = schema.seed("roles")
    assert roles["narrator"]["mode"] == "off"
    for name, doc in schema.SHIPPED_PROFILES.items():
        assert (doc.get("ui") or {}).get("narrator", "off") == "off", name
    for name, doc in schema.PROFILES.items():
        assert (doc.get("ui") or {}).get("narrator", "off") == "off", name
    p = residency.plan(60.0, schema.seed("residency"), roles)
    assert "narrator" not in p["order"] and p["order"][0] == "decider", p["order"]
    assert schema.seed("residency")["claims"]["decider"]["gib"] == 3.0
    # still switchable: the mode and the profile switch validate
    assert schema.check("roles", dict(roles, narrator=dict(roles["narrator"], mode="on"))) == []
    prof = copy.deepcopy(schema.SHIPPED_PROFILES["coding"])
    prof["ui"]["narrator"] = "live"
    assert schema.validate(prof, schema.SCHEMAS["profiles"]) == []


def test_1_no_model_card_turns_the_narrator_on():
    seen = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "*", "model.yaml"))):
        with open(path) as fh:
            doc = yaml.safe_load(fh)
        comps = [c for c in _walk(doc) if c.get("id") == "narrator" and "default" in c]
        for c in comps:
            assert c["default"] is False, path
            seen += 1
    assert seen >= 6, seen


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


# ---------------------------------------------------------------------- 2: the migrations
def _old_profile(name: str, narrator: str | None = "live") -> dict:
    d = copy.deepcopy(schema.SHIPPED_PROFILES[name])
    if narrator is None:
        d["ui"].pop("narrator", None)
    else:
        d["ui"]["narrator"] = narrator
    return d


def _old_roles(mode: str = "on") -> dict:
    r = schema.seed("roles")
    r["narrator"] = dict(r["narrator"], mode=mode)
    return r


def _old_residency(gib: float = 8.2) -> dict:
    d = schema.seed("residency")
    d["claims"] = dict(d["claims"], decider=dict(d["claims"]["decider"], gib=gib))
    return d


def test_2_a_box_with_the_old_shipped_values_gets_the_new_defaults_once():
    st = Store(tempfile.mkdtemp(prefix="srv534-"))
    for name in ("coding", "chat"):
        st.put("profiles", name, _old_profile(name), note="shipped", author="shipped")
    st.put("roles", None, _old_roles(), note="old", author="owner")
    st.put("residency", None, _old_residency(), note="old", author="owner")
    got = {r["id"]: r for r in migrations.run(st, migrations=NARRATOR)}
    assert {k: (v["result"], v.get("changes")) for k, v in got.items()} == {
        "2026-10-17-coding-narrator-off": ("changed", ["ui.narrator=off"]),
        "2026-10-17-chat-narrator-off": ("changed", ["ui.narrator=off"]),
        "2026-10-17-roles-narrator-off": ("changed", ["narrator.mode=off"]),
        "2026-10-17-residency-decider": ("changed", ["claims.decider.gib=3.0"])}, got
    for name in ("coding", "chat"):
        v, doc = st.get("profiles", name)
        assert doc["ui"]["narrator"] == "off" and v == 2
        rest = dict(doc, ui={k: x for k, x in doc["ui"].items() if k != "narrator"})
        old = _old_profile(name)
        assert rest == dict(old, ui={k: x for k, x in old["ui"].items() if k != "narrator"})
        assert st.versions("profiles", name)[0]["author"] == "migration"
    assert st.get("roles")[1]["narrator"]["mode"] == "off"
    claim = st.get("residency")[1]["claims"]["decider"]
    assert (claim["gib"], claim["basis"]) == (3.0, "measured") and "OPS-69" in claim["note"]
    assert migrations.run(st, migrations=NARRATOR) == [], "runs once"


def test_2_a_value_someone_chose_stays():
    st = Store(tempfile.mkdtemp(prefix="srv534-"))
    # coding: `thoughts` once, `live` now (the reference box's history): someone chose it
    st.put("profiles", "coding", _old_profile("coding", "thoughts"), note="n2", author="owner")
    st.put("profiles", "coding", _old_profile("coding", "live"), note="v3", author="owner")
    # chat: absent (stored before the key), then live: no choice, the old default
    st.put("profiles", "chat", _old_profile("chat", None), note="old", author="owner")
    st.put("profiles", "chat", _old_profile("chat", "live"), note="v3", author="owner")
    st.put("roles", None, _old_roles("shadow"), note="shadow", author="owner")
    st.put("roles", None, _old_roles("on"), note="on again", author="owner")
    st.put("residency", None, _old_residency(5.0), note="mine", author="owner")
    st.put("residency", None, _old_residency(8.2), note="back", author="owner")
    got = {r["id"]: r["result"] for r in migrations.run(st, migrations=NARRATOR)}
    assert got == {"2026-10-17-coding-narrator-off": "unchanged",
                   "2026-10-17-chat-narrator-off": "changed",
                   "2026-10-17-roles-narrator-off": "unchanged",
                   "2026-10-17-residency-decider": "unchanged"}, got
    assert st.get("profiles", "coding")[1]["ui"]["narrator"] == "live"
    assert st.get("profiles", "chat")[1]["ui"]["narrator"] == "off"
    assert st.get("roles")[1]["narrator"]["mode"] == "on"
    assert st.get("residency")[1]["claims"]["decider"]["gib"] == 8.2
    # a box with the new values, or none stored: nothing to do
    st = Store(tempfile.mkdtemp(prefix="srv534-"))
    st.put("profiles", "coding", copy.deepcopy(schema.SHIPPED_PROFILES["coding"]),
           note="shipped", author="shipped")
    got = {r["id"]: r["result"] for r in migrations.run(st, migrations=NARRATOR)}
    assert got == {"2026-10-17-coding-narrator-off": "unchanged",
                   "2026-10-17-chat-narrator-off": "absent",
                   "2026-10-17-roles-narrator-off": "absent",
                   "2026-10-17-residency-decider": "absent"}, got


def test_2_a_roles_document_from_before_the_narrator_gets_the_new_role():
    """The reference box's roles v2 (2026-10-01) has no narrator: the upgrade gives it the
    built-in one, now off, and the migration has nothing to change."""
    roles = schema.seed("roles")
    roles.pop("narrator")
    st = Store(tempfile.mkdtemp(prefix="srv534-"))
    st.put("roles", None, roles, note="before P1", author="owner")
    got = {r["id"]: r["result"] for r in migrations.run(st, migrations=NARRATOR)}
    assert got["2026-10-17-roles-narrator-off"] == "unchanged", got
    assert st.get("roles")[1]["narrator"]["mode"] == "off"


def _router_without_core(*drop) -> dict:
    r = copy.deepcopy(schema.SHIPPED_ROUTER)
    for c in r["classes"]:
        if c["id"] in drop:
            c["policy"].pop("core", None)
    return r


def _router_store() -> Store:
    st = Store(tempfile.mkdtemp(prefix="srv534-"))
    st.put("packages", None, copy.deepcopy(schema.SHIPPED_PACKAGES), note="shipped", author="shipped")
    return st


def test_2_a_router_from_before_fx6_gets_the_coding_core():
    """SRV-534: the reference box's router (v24) has no core on code_edit and debug, so under the
    fit's `packages: core` a coding turn, decided or fallen back, got no package; the migration
    gives them FX6's ["coding"] unless a stored version had a core there."""
    assert CORE and CORE[0][1] == "router"
    st = _router_store()
    st.put("router", None, _router_without_core("code_edit", "debug"), note="old", author="owner")
    got = migrations.run(st, migrations=CORE)
    assert (got[0]["result"], got[0]["changes"]) == (
        "changed", ["classes.code_edit.core=[coding]", "classes.debug.core=[coding]"]), got
    doc = st.get("router")[1]
    core = {c["id"]: c["policy"].get("core") for c in doc["classes"]}
    assert core["code_edit"] == ["coding"] and core["debug"] == ["coding"]
    assert core["three_d"] == ["threejs-scene", "coding"] and core["design_ui"] == ["ui-design"]
    assert migrations.run(st, migrations=CORE) == [], "runs once"
    # a core someone removed stays removed; the shipped router is unchanged
    st = _router_store()
    st.put("router", None, copy.deepcopy(schema.SHIPPED_ROUTER), note="shipped", author="shipped")
    st.put("router", None, _router_without_core("code_edit"), note="no core", author="owner")
    assert migrations.run(st, migrations=CORE)[0]["result"] == "unchanged"
    st = _router_store()
    st.put("router", None, copy.deepcopy(schema.SHIPPED_ROUTER), note="shipped", author="shipped")
    assert migrations.run(st, migrations=CORE)[0]["result"] == "unchanged"


def test_2_a_router_without_a_design_or_drawing_core_gets_one():
    """Live (2026-10-17, record fa3747d242314511): Laya read "make a three.js page with a red
    cube" as design_ui, a class with no core, so under Super VL's `packages: core` the turn got
    no package at all. design_ui gets ["ui-design"], drawing ["creative-coding"], as the shipped
    router has them; a class that does not list the package, or had a core before, is left."""
    assert BUILD_CORE and BUILD_CORE[0][1] == "router"
    shipped = {c["id"]: c["policy"].get("core") for c in schema.SHIPPED_ROUTER["classes"]}
    assert shipped["design_ui"] == ["ui-design"] and shipped["drawing"] == ["creative-coding"]
    st = _router_store()
    st.put("router", None, _router_without_core("design_ui", "drawing"), note="old",
           author="owner")
    got = migrations.run(st, migrations=BUILD_CORE)
    assert (got[0]["result"], got[0]["changes"]) == (
        "changed", ["classes.design_ui.core=[ui-design]",
                    "classes.drawing.core=[creative-coding]"]), got
    core = {c["id"]: c["policy"].get("core") for c in st.get("router")[1]["classes"]}
    assert core["design_ui"] == ["ui-design"] and core["drawing"] == ["creative-coding"]
    assert core["three_d"] == ["threejs-scene", "coding"] and core["code_edit"] == ["coding"]
    assert migrations.run(st, migrations=BUILD_CORE) == [], "runs once"
    # a drawing class without creative-coding among its packages gets no core of it
    r = _router_without_core("design_ui", "drawing")
    for c in r["classes"]:
        if c["id"] == "drawing":
            c["policy"]["packages"] = [p for p in c["policy"]["packages"] if p != "creative-coding"]
    st = _router_store()
    st.put("router", None, r, note="old", author="owner")
    assert migrations.run(st, migrations=BUILD_CORE)[0]["changes"] == [
        "classes.design_ui.core=[ui-design]"]
    # a core someone removed stays removed
    st = _router_store()
    st.put("router", None, copy.deepcopy(schema.SHIPPED_ROUTER), note="shipped", author="shipped")
    st.put("router", None, _router_without_core("design_ui", "drawing"), note="no core",
           author="owner")
    assert migrations.run(st, migrations=BUILD_CORE)[0]["result"] == "unchanged"


def _router_3d_old() -> dict:
    """The live box's three_d before the change: core threejs-scene, no coding in its packages."""
    r = copy.deepcopy(schema.SHIPPED_ROUTER)
    for c in r["classes"]:
        if c["id"] == "three_d":
            c["policy"]["core"] = ["threejs-scene"]
            c["policy"]["packages"] = [p for p in c["policy"]["packages"] if p != "coding"]
    return r


def test_2_three_d_gets_the_coding_core_beside_the_3d_skill():
    """Pagoda replays 1-3 (2026-10-17, Super VL, three_d): the first step thought 6 to 11 minutes
    and drafted the 15 files in the thinking (63k characters, 65 code fences in replay 1); the
    coding core's "a plan of 3 to 8 lines, code in files only" reached code_edit turns only."""
    assert CODING_3D and CODING_3D[0][1] == "router"
    st = _router_store()
    st.put("router", None, _router_3d_old(), note="old", author="owner")
    got = migrations.run(st, migrations=CODING_3D)
    assert (got[0]["result"], got[0]["changes"]) == ("changed", ["classes.three_d.core+=coding"])
    pol = {c["id"]: c["policy"] for c in st.get("router")[1]["classes"]}["three_d"]
    assert pol["core"] == ["threejs-scene", "coding"] and "coding" in pol["packages"]
    assert migrations.run(st, migrations=CODING_3D) == [], "runs once"
    # taken out by someone after a version had it: stays out
    st = _router_store()
    st.put("router", None, copy.deepcopy(schema.SHIPPED_ROUTER), note="shipped", author="shipped")
    st.put("router", None, _router_3d_old(), note="no coding", author="owner")
    assert migrations.run(st, migrations=CODING_3D)[0]["result"] == "unchanged"


# ---------------------------------------------------------------------- 3: /health
def test_3_health_says_why_laya_is_not_loaded():
    from engine.decision.registry import Registry
    from server import app
    from test_decider_runtime import FakeModel
    import threading

    refused = {"allowed": False, "reason": "refused_memory", "role": "decider",
               "mem_available_gib": 9.82, "engine_growth_gib": 0.0, "claim_gib": 3.0,
               "after_gib": 6.82, "need_gib": 21.0}
    answers = [refused]
    said = []
    reg = Registry([FakeModel("laya-english")], gate=threading.Lock(),
                   planner=lambda name, **kw: answers[-1], log=said.append,
                   policy=lambda: {"laya-english": "on_start"})
    keep = {k: app.STATE.get(k) for k in ("role_mem", "narrator", "shadow", "deciders")}
    try:
        app.STATE.update(role_mem=None, narrator=None, shadow=None, deciders=reg)
        assert app.roles_report()["roles"]["laya"] == {"models": [], "state": "not_loaded"}
        reg.autoload(service_start=True)
        for _ in range(200):
            if reg.refusal:
                break
            app.time.sleep(0.01)
        laya = app.roles_report()["roles"]["laya"]
        assert laya["state"] == "not_loaded", laya
        r = laya["refused"]
        assert (r["model"], r["reason"], r["after_gib"], r["need_gib"], r["claim_gib"]) == \
            ("laya-english", "refused_memory", 6.82, 21.0, 3.0), r
        assert sum("WARNING: no decision model is loaded" in x for x in said) == 1, said
        reg.load("laya-english")                       # refused again: no second loud line
        assert sum("WARNING" in x for x in said) == 1, said
        answers.append({"allowed": True, "reason": None, "role": "decider"})
        reg.load("laya-english", wait=True)
        laya = app.roles_report()["roles"]["laya"]
        assert laya == {"models": ["laya-english"], "state": "loaded"}, laya
    finally:
        app.STATE.update(keep)


def test_3_the_role_governor_reports_it_too():
    from server import rolemem

    class Reg:
        def __init__(self, r):
            self.r, self.state, self._batchers = r, {}, {}

        def loaded(self):
            return []

        def refused(self):
            return self.r

    role = rolemem.RegistryRole("laya", Reg({"model": "laya-english", "reason": "refused_hold"}))
    assert role.detail() == {"error": None,
                             "refused": {"model": "laya-english", "reason": "refused_hold"}}
    assert rolemem.RegistryRole("laya", types.SimpleNamespace()).detail()["refused"] is None


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
