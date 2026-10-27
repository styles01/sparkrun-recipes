"""The setup guide's selection check and memory estimate (SRV-323, server/setup/estimate.py), and the
fixture that pins them for the dashboard's copy, models/estimate-cases.json.

The test matrix:

  1  the worked example, by hand from the registry's numbers: exl3-4.5, the recommended tools, 4 slots
     of 262,144 with the pool: 67.35 GiB of a 106.63 budget, fits; every slot full would need 104.25
     GiB of KV, so fits_worst is false and a note says what happens then
                                                         test_the_worked_example
  2  the live memory split of the reference box (exl3-6.0, 4 slots, the small models, 0.56 GiB of KV
     committed, 59.34 GiB in use) is what the parts add up to, within 0.1 GiB
                                                         test_the_parts_match_the_live_split
  3  over the budget: the pool off at 4 x 262,144 does not fit, and the notes name the largest parts,
     the pool and what to change                         test_over_the_budget
  4  exactly at the budget fits; a cent more does not    test_exactly_at_the_budget
  5  the context that fits is floor(room / bytes a token) over all slots, and a note gives a slot's
     share capped by the window                          test_the_context_that_fits
  6  the selection check: unknown ids are 400s, a tool whose dependency is off, a required tool off,
     slots (past the engine's 16) or windows out of range, an unknown key, a field value that is not the field's are 422s
     with a code and the field; what the body leaves out is the registry's default, the side window
     the window                                          test_the_selection_check
  7  the fixture: at least 30 cases over every profile, every tool, slots 1 to 4, five windows, the
     pool on and off, over and at the budget, and the refusals; each equals what the code computes
     now, and its model is the registry's payload       test_the_fixture
  8  the estimate is pure and fast: under 20 ms a call (median)   test_time
  9  SRV-337, Flash-Next's packs: a warm table is a fixed part, a cold one its row cache; the prefill's
     score buffer grows with the window; the context that fits is capped at the profile's largest
     window; 6.05 fits no window and says by how much; a window above a profile's largest is a 422;
     the fixture holds a case per pack at 131,072 and 262,144     test_flash_next_packs

Run: python tests/test_setup_estimate.py            (--write regenerates the fixture)
"""

from __future__ import annotations

import copy
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.setup import estimate as E  # noqa: E402
from server.setup import registry  # noqa: E402

FIXTURE = os.path.join(registry.ROOT, "models", "estimate-cases.json")
GIB = 1 << 30


def _m() -> dict:
    return registry.get("qwen3.8-27b")


def _fn() -> dict:
    return registry.get("qwen3.8-flash-next")


def _est(**kw) -> dict:
    m = _m()
    return E.estimate(m, E.check(m, dict({"model": m["id"]}, **kw)))


# the reference box's tools: the narrator on beside Laya and the shadow reader (off by default since
# 0.11.3, SRV-534)
REF = {"narrator": True}


def test_the_worked_example():
    e = _est(components=REF)
    # by hand: os 10.0, weights 16.14, graphs 6.63, engine 1.28, allocator 2.6, 3 side slots x
    # 1.19, drafters 5.41, lookup 0.19, narrator 3.6, laya 1.6, shadow 1.52, vision 0.86, prompt
    # cache 0.43, state store 0.49; KV 4 x 32,768 x 106,752 bytes
    fixed = 10.0 + 16.14 + 6.63 + 1.28 + 2.6 + 3 * 1.19 + 5.41 + 0.19 + 3.6 + 1.6 + 1.52 + 0.86 \
        + 0.43 + 0.49
    kv = 4 * 32_768 * 106_752 / GIB
    assert abs(e["total_gb"] - (fixed + kv)) < 0.006, (e["total_gb"], fixed + kv)
    assert e["total_gb"] == 67.35 and e["budget_gb"] == 106.63 and e["device_gb"] == 121.63
    assert e["fits"] is True and e["fits_worst"] is False
    assert e["kv_worst_gb"] == 104.25
    assert any("waits for room" in n for n in e["notes"]), e["notes"]
    kinds = {p["id"]: p["kind"] for p in e["parts"]}
    assert kinds["kv"] == "kv" and kinds["allocator"] == "reserve" and kinds["weights"] == "fixed"
    assert next(p for p in e["parts"] if p["id"] == "prompt_cache")["worst_gb"] == 11.5
    assert "conductor" not in kinds and "disk_tier" not in kinds      # no memory of their own
    assert e["available_gb"] is None


def test_the_parts_match_the_live_split():
    # memory.split of /v1/dashboard/system, 2026-10-10 22:30: in_use 63,713,701,888 bytes of
    # which 603,979,776 committed KV; the estimate of the same selection with that KV
    e = _est(profile="exl3-6.0", components=REF)
    kv = next(p for p in e["parts"] if p["id"] == "kv")["gb"]
    est = e["total_gb"] - kv + 603_979_776 / GIB
    live = 63_713_701_888 / GIB
    assert abs(est - live) < 0.1, (est, live)


def test_over_the_budget():
    e = _est(kv_pool=False, profile="exl3-6.0", components={"response_cache": True})
    assert e["fits"] is False and e["total_gb"] > e["budget_gb"]
    text = " ".join(e["notes"])
    assert "Over the budget by" in text and "Conversations (KV)" in text
    assert "shared KV pool on" in text and "Fewer slots" in text
    kv = next(p for p in e["parts"] if p["id"] == "kv")
    assert kv["gb"] == kv["worst_gb"] == 104.25


def _at_budget() -> tuple[dict, float]:
    sel = {"kv_pool": False, "slots": 2, "window": 262_144, "side_window": 131_072}
    total = _est(**sel)["total_gb"]
    return sel, round(121.63 - total, 2)


def test_exactly_at_the_budget():
    sel, floor = _at_budget()
    assert 10 <= floor <= 40, floor
    e = _est(floor_gb=floor, **sel)
    assert e["total_gb"] == e["budget_gb"] and e["fits"] is True, e
    e = _est(floor_gb=round(floor + 0.01, 2), **sel)
    assert e["fits"] is False


def test_the_context_that_fits():
    e = _est(slots=1)
    fixed = sum(p["gb"] for p in e["parts"] if p["kind"] != "kv")
    room = e["budget_gb"] - fixed
    assert abs(e["context_tokens_fit"] - room * GIB / 106_752) < 400, e["context_tokens_fit"]
    assert e["room_gb"] == round(room, 2)
    note = next(n for n in e["notes"] if n.startswith("About"))
    assert f"{min(262_144, e['context_tokens_fit']):,} a slot" in note, note
    e = _est(slots=4, window=8192, side_window=8192)
    note = next(n for n in e["notes"] if n.startswith("About"))
    assert "8,192 a slot" in note, note


def _refused(body: dict, status: int, code: str, field: str | None = None) -> None:
    m = _m()
    try:
        E.check(m, body)
    except E.SelectionError as exc:
        assert (exc.status, exc.code) == (status, code), (body, exc.status, exc.code, str(exc))
        if field:
            assert field in json.dumps(exc.body()), (field, exc.body())
        return
    raise AssertionError(f"accepted {body}")


def test_the_selection_check():
    m = _m()
    base = {"model": m["id"]}
    _refused({"model": "nope"}, 400, "unknown_model")
    _refused(dict(base, profile="exl3-9.9"), 400, "unknown_profile")
    _refused(dict(base, components={"turbo": True}), 400, "unknown_component")
    _refused(dict(base, device="mac"), 400, "unknown_device")
    _refused(dict(base, components={"conductor": False}), 422, "invalid_combination", "laya")
    _refused(dict(base, components={"drafters": False}), 422, "invalid_combination", "drafters")
    _refused(dict(base, components={"disk_tier": False}), 422, "invalid_combination", "nvme_spill")
    _refused(dict(base, components={"vision": "yes"}), 422, "invalid_selection")
    _refused(dict(base, slots=17), 422, "out_of_range", "slots")
    assert E.check(m, dict(base, slots=16))["slots"] == 16     # past the card's measured 4: memory decides
    _refused(dict(base, slots=0), 422, "out_of_range", "slots")
    _refused(dict(base, slots=2.5), 422, "out_of_range", "slots")
    _refused(dict(base, window=512), 422, "out_of_range", "window")
    _refused(dict(base, side_window=10 ** 9), 422, "out_of_range", "side_window")
    _refused(dict(base, floor_gb=5), 422, "out_of_range", "floor_gb")
    _refused(dict(base, kv_pool="on"), 422, "invalid_selection", "kv_pool")
    _refused(dict(base, steps=["rm -rf /"]), 422, "unknown_field", "steps")
    _refused(dict(base, cmd="reboot"), 422, "unknown_field", "cmd")
    _refused(dict(base, methods={"reasoning_effort": "max"}), 422, "invalid_value",
             "methods.reasoning_effort")
    _refused(dict(base, advanced={"served_model": "x; reboot"}), 422, "invalid_value")
    _refused(dict(base, advanced={"secret_path": "/etc"}), 422, "unknown_field")
    _refused("not an object", 422, "invalid_selection")
    s = E.check(m, dict(base, window=65_536))
    assert s["side_window"] == 65_536 and s["profile"] == "exl3-4.5" and s["slots"] == 4
    # the side window is clamped to the window, and both keep the 8,192 step
    assert E.check(m, dict(base, window=32_768, side_window=262_144))["side_window"] == 32_768
    _refused(dict(base, window=100_000), 422, "off_step", "window")
    _refused(dict(base, side_window=20_000), 422, "off_step", "side_window")
    # SRV-534: the narrator is off by default (0.11.3)
    assert s["components"]["narrator"] is False and s["components"]["response_cache"] is False
    assert s["methods"]["sampled_tree"] == "det" and s["advanced"]["port"] == 8000
    s = E.check(m, dict(base, components={"conductor": False, "narrator": False, "laya": False,
                                          "shadow": False}))
    assert s["components"]["conductor"] is False
    assert E.defaults(m) == E.check(m, base)


def _cases() -> list[dict]:
    m = _m()
    out = []

    def case(name, **sel):
        out.append({"name": name, "selection": dict({"model": m["id"]}, **sel)})

    for p in m["profiles"]:
        case(f"profile {p['id']}, the recommended rest", profile=p["id"])
    for c in m["components"]:
        if c["required"]:
            continue
        on = not c["default"]
        comps = {c["id"]: on}
        if c["id"] == "conductor":
            comps.update(narrator=False, laya=False, shadow=False)
        if c["id"] == "disk_tier":
            comps.update(nvme_spill=False)
        case(f"{c['id']} {'on' if on else 'off'}", components=comps)
    for n in range(1, 5):
        case(f"{n} slot(s)", slots=n)
    for w in (8192, 32_768, 65_536, 131_072, 262_144):
        case(f"window {w:,} on every slot", window=w)
    case("one long window, three short ones", window=262_144, side_window=32_768)
    case("pool off, 4 x 262,144: over the budget", kv_pool=False)
    case("pool off, 2 x 65,536", kv_pool=False, slots=2, window=65_536)
    case("pool off, 1 x 262,144, exl3-2.5", kv_pool=False, slots=1, profile="exl3-2.5")
    case("floor 10", floor_gb=10)
    case("floor 40", floor_gb=40)
    case("everything on, exl3-6.0, pool off: the most it can take", profile="exl3-6.0",
         kv_pool=False, components={"response_cache": True, "proxy": True})
    case("the smallest: exl3-2.5, one slot of 8,192, no tools", profile="exl3-2.5", slots=1,
         window=8192, components={c["id"]: c["required"] for c in m["components"]})
    sel, floor = _at_budget()
    case("exactly at the budget", floor_gb=floor, **sel)
    case("a cent over the budget", floor_gb=round(floor + 0.01, 2), **sel)
    case("methods and advanced fields do not move the memory", methods={"reasoning_effort": "low"},
         advanced={"port": 8080})
    case("5 slots, past the card's measured 4: memory decides", slots=5, window=32_768)
    case("16 slots, the engine's limit, pool off: over the budget", slots=16, kv_pool=False)
    case("refused: slots 17, above the engine's limit", slots=17)
    case("refused: window 512", window=512)
    case("refused: a window off the 8,192 step", window=100_000)
    case("refused: a side window off the step", side_window=20_000)
    case("a side window longer than the window is the window", window=32_768,
         side_window=262_144)
    case("refused: the narrator without the conductor", components={"conductor": False})
    case("refused: an unknown profile", profile="exl3-9.9")
    case("refused: an unknown key", cmd="reboot")
    f = _fn()
    for p in f["profiles"]:
        for w in (131_072, 262_144):
            out.append({"name": f"Flash-Next {p['id']}, window {w:,}",
                        "selection": {"model": f["id"], "profile": p["id"], "window": w}})
    out.append({"name": "Flash-Next, two slots (measured up to one)", "selection": {"model": f["id"], "slots": 2}})
    models = {m["id"]: m, f["id"]: f}
    for c in out:
        mm = models[c["selection"].get("model", m["id"])] if c["selection"].get("model") in models else m
        try:
            c["expect"] = E.estimate(mm, E.check(mm, c["selection"]))
        except E.SelectionError as exc:
            c["error"] = {"status": exc.status, "code": exc.code}
    return out


# the parts of the model payload (GET /v1/setup/models/{id}) the formula and the check read
FORMULA_KEYS = {"id": None,
                "profiles": ("id", "label", "weights_gib", "drafters_gib", "recommended"),
                "devices": ("id", "label", "total_gib", "reserved_gib"), "memory": None,
                "components": ("id", "label", "default", "required", "depends_on", "conflicts",
                               "memory_gib", "memory_max_gib"),
                "parallelism": None}


def _formula_model(m: dict) -> dict:
    out = {}
    for k, keep in FORMULA_KEYS.items():
        v = json.loads(json.dumps(m[k]))
        out[k] = [{f: x[f] for f in keep} for x in v] if keep else v
    # SRV-337's profile keys, where a profile has them (a 27B profile has none)
    for x, src in zip(out["profiles"], m["profiles"]):
        x.update({k: src[k] for k in ("max_window", "table", "components_gib") if k in src})
    out["memory"] = {"parts": out["memory"]["parts"],
                     **({"per_window_parts": m["memory"]["per_window_parts"]}
                        if m["memory"].get("per_window_parts") else {})}
    return out


def _fixture() -> dict:
    m = _m()
    return {"contract_version": "1.0",
            "about": "server/setup/estimate.py pins its formula here; the dashboard's TypeScript copy "
                     "must give every `expect` from `model` and `selection` (each number within "
                     "0.01, the rest equal; `fits` compares the figures rounded to 0.01) and refuse "
                     "every `error` case with that status and code. `model` is the part of the "
                     "model payload (GET /v1/setup/models/{id}) the formula reads. Regenerate with "
                     "python tests/test_setup_estimate.py --write after a registry change.",
            "model": _formula_model(m), "models": {_fn()["id"]: _formula_model(_fn())}, "cases": _cases()}


def test_the_fixture():
    with open(FIXTURE) as fh:
        fx = json.load(fh)
    want = _fixture()
    assert fx["model"] == want["model"], "the registry moved: python tests/test_setup_estimate.py --write"
    assert fx["models"] == want["models"], "the registry moved: python tests/test_setup_estimate.py --write"
    assert len(fx["cases"]) >= 30, len(fx["cases"])
    for got, new in zip(fx["cases"], want["cases"]):
        assert got == new, (got["name"], got, new)
    assert len(fx["cases"]) == len(want["cases"])
    sels = [c["selection"] for c in fx["cases"] if c["selection"].get("model") == fx["model"]["id"]]
    assert {s.get("profile", "exl3-4.5") for s in sels} >= {p["id"] for p in fx["model"]["profiles"]}
    assert {s.get("slots", 4) for s in sels} >= {1, 2, 3, 4}
    assert {s.get("window") for s in sels} >= {8192, 32_768, 65_536, 131_072, 262_144}
    assert {s.get("kv_pool", True) for s in sels} == {True, False}
    toggled = {k for s in sels for k in (s.get("components") or {})}
    assert toggled >= {c["id"] for c in fx["model"]["components"] if not c["required"]}
    exp = [c["expect"] for c in fx["cases"] if "expect" in c]
    assert any(not e["fits"] for e in exp) and any(e["total_gb"] == e["budget_gb"] and e["fits"]
                                                    for e in exp)
    assert sum("error" in c for c in fx["cases"]) >= 5


def test_flash_next_packs():
    f = _fn()

    def est(**kw):
        return E.estimate(f, E.check(f, dict({"model": f["id"]}, **kw)))
    e = est(profile="flashnext-3.05", window=262_144)
    parts = {x["id"]: x for x in e["parts"]}
    # SPD-99: 3.05 reads its table cold (the row cache), and its prefill workspace was measured at 256k
    assert parts["table"]["gb"] == 2.0 and "row cache" in parts["table"]["label"]
    assert parts["prefill_workspace"]["gb"] == 10.5                    # stops growing at about 140k
    assert parts["window_buffers"]["gb"] == round(32230 * 262_144 / GIB, 2)
    assert parts["prefill_workspace"]["kind"] == "reserve"
    assert e["fits"] and e["context_tokens_fit"] == 262_144, (e["context_tokens_fit"], e["room_gb"])
    warm = {x["id"]: x for x in est(profile="flashnext-2.05", window=262_144)["parts"]}
    assert warm["table"]["gb"] == 24.44 and "kept warm" in warm["table"]["label"]
    cold = {x["id"]: x for x in est(profile="flashnext-4.05", window=131_072)["parts"]}
    assert cold["table"]["gb"] == 2.0 and "row cache" in cold["table"]["label"]
    six = est(profile="flashnext-6.05", window=8192)
    assert not six["fits"] and six["context_tokens_fit"] == 0
    assert any("at any window" in n and "GiB over the budget before any context" in n for n in six["notes"]), six["notes"]
    g = copy.deepcopy(f)
    registry.profile(g, "flashnext-3.05")["max_window"] = 131_072
    try:
        E.check(g, {"model": f["id"], "profile": "flashnext-3.05", "window": 262_144})
        raise AssertionError("a window above the profile's largest must be refused")
    except E.SelectionError as exc:
        assert (exc.status, exc.code) == (422, "out_of_range"), exc.body()
    capped = E.estimate(g, E.check(g, {"model": f["id"], "profile": "flashnext-3.05", "window": 131_072}))
    assert capped["context_tokens_fit"] == 131_072
    return "warm table and score buffer as parts; 3.05 fits 262,144; 6.05 fits none; a profile's window caps and refuses"


def test_time():
    m = _m()
    body = {"model": m["id"], "slots": 3, "window": 131_072}
    E.estimate(m, E.check(m, body))
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        E.estimate(m, E.check(m, body))
        ts.append(time.perf_counter() - t0)
    assert statistics.median(ts) < 0.02, statistics.median(ts)
    print(f"    median {statistics.median(ts) * 1e3:.3f} ms")


if __name__ == "__main__":
    if "--write" in sys.argv:
        with open(FIXTURE, "w") as fh:
            json.dump(_fixture(), fh, indent=1)
            fh.write("\n")
        print(f"wrote {FIXTURE}")
        sys.exit(0)
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
