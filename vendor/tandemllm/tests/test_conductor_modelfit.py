"""SRV-441 (QA SRV-442): the conductor's fit to the served model (server/conductor/modelfit.py), on the
router test kit of tests/test_conductor_router.py (the real Qwen template, fake classifier stages).

SRV-441's Gherkin, row by row:
  1. the served model does not answer the class question: the decider alone decides (also where the
     router leaves its stage off), a turn below its bar falls back without asking the served model;
  2. a turn nobody classified gets the profile's default effort and no effort sentence (lean
     fallback), and so does a tool turn of that session; SRV-534: it keeps the class's core and
     rules as the fit lets them (the profile's default class: coding's for the Chat tab), the tool
     turn keeps the note of its user message, and a decider that is down changes nothing of that;
  3. a classified turn keeps its level and budget, without the level's sentence;
  4. packages: the class's core alone (`core`), nothing on the chat class;
  5. memory off: no memory card or index; rules off: no class rules;
  6. the setting: `off` is the router as written, `k=v` replaces a key, a wrong key or value is refused;
  7. the LFM2.5 adapter declares a fit that resolves.

Run: python tests/test_conductor_modelfit.py
"""

from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import CODE, SYSTEM, body, by_label, setup, turn  # noqa: E402

from server.conductor import modelfit, schema  # noqa: E402

LFM = {"served_classifier": False, "fallback": "lean", "packages": "core",
       "no_package_classes": ("chat",), "effort_notes": False}


def fitted(fit: dict | None, setting: str = "model"):
    modelfit.install(fit, setting)


def _packages(cond):
    """A general coding package, a core package and a task skill on code_edit; a chat package."""
    _, p = cond.store.get("packages")
    for n, general in (("coding", True), ("core-recipe", False), ("task-skill", False),
                       ("chat-help", False)):
        p["packages"][n] = {"title": n, "kind": "package", "source": "inline",
                            "description": f"The {n} text.", "body": f"# {n} body",
                            "max_tokens": 3000, "enabled": True, "general": general}
    cond.store.put("packages", None, p, note="fit test", author="t")
    _, r = cond.store.get("router")
    for c in r["classes"]:
        if c["id"] == "code_edit":
            c["policy"]["packages"] = ["coding", "core-recipe", "task-skill"]
            c["policy"]["core"] = ["core-recipe"]
        if c["id"] == "chat":
            c["policy"]["packages"] = ["chat-help"]
    cond.store.put("router", None, r, note="fit test", author="t")


def _parts(plan) -> list[str]:
    return [p["name"] for n in plan.injected for p in n["parts"]]


def test_1_the_served_model_is_not_asked():
    fitted(LFM)
    try:
        cond, fakes = setup(decider=by_label("code_edit", 0.4), qwen=by_label("code_edit", 0.99))
        plan, _, _ = turn(cond, body([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": CODE}]))
        assert not fakes["qwen"].calls, "the served model answered the class question"
        assert len(fakes["decider"].calls) == 1
        assert [s["stage"] for s in plan.row["stages"]] == ["decider"]
        assert plan.row["source"] in ("fallback", "decider"), plan.row["source"]
        fitted(None)
        cond, fakes = setup(decider=by_label("code_edit", 0.4), qwen=by_label("code_edit", 0.99))
        plan, _, _ = turn(cond, body([{"role": "user", "content": CODE}]))
        assert fakes["qwen"].calls and plan.row["source"] == "qwen", "the router as written"
        # the shipped router leaves the decider stage off: the fit asks it anyway
        for fit, want in ((LFM, ["decider"]), (None, ["qwen"])):
            fitted(fit)
            cond, fakes = setup(decider=by_label("code_edit", 0.9), qwen=by_label("chat", 0.99))
            _, r = cond.store.get("router")
            for st in r["classifier"]["stages"]:
                st["enabled"] = st["kind"] != "decider"
            cond.store.put("router", None, r, note="decider off", author="t")
            plan, _, _ = turn(cond, body([{"role": "user", "content": CODE}]))
            assert [s["stage"] for s in plan.row["stages"]] == want, plan.row["stages"]
    finally:
        fitted(None)


def test_2_a_turn_nobody_classified_is_lean_and_so_is_its_tool_turn():
    fitted(LFM)
    try:
        cond, fakes = setup(decider=by_label("research", 0.3))
        _packages(cond)
        _, r = cond.store.get("router")
        r["classifier"]["below_accept"] = {"mode": "default"}
        cond.store.put("router", None, r, note="no below-build", author="t")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hello there"}]
        plan, ids1, _ = turn(cond, body(msgs))
        row = plan.row
        assert (row["class_id"], row["source"]) == ("code_edit", "fallback"), row["class_id"]
        assert row["extra"].get("model_fit_lean") is True
        assert row["effort"] == "medium", "the profile's default level, not the class's xhigh"
        # SRV-534: the default class's core and rules, as the fit lets them; never nothing
        assert _parts(plan) == ["core-recipe", "code_edit"], _parts(plan)
        assert sorted(row["extra"]["package_fit_dropped"]) == ["coding", "task-skill"]
        assert schema.XHIGH_NOTE not in "".join(n["text"] for n in plan.injected)
        assert row["extra"]["model_fit"]["fallback"] == "lean"
        # the tool turn of that session: still nobody decided, the same note, the prefix holds
        msgs2 = msgs + [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments":
                                                          "{\"path\": \"a\"}"}}]},
                        {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
        plan2, ids2, _ = turn(cond, body(msgs2))
        assert plan2.row["extra"].get("model_fit_lean") is True
        assert plan2.row["effort"] == "medium" and not plan2.injected, plan2.injected
        assert plan2.block["key"] == plan.block["key"], "the same note as its user message"
        assert ids2[:len(ids1)] == ids1, "the tool turn's prompt begins with the first one's"
        assert len(fakes["decider"].calls) == 1, "a tool turn asks no model"
        # the router as written: the default class's whole policy
        fitted(None)
        cond, fakes = setup(decider=by_label("research", 0.3))
        _packages(cond)
        cond.store.put("router", None, r, note="no below-build", author="t")
        plan, _, _ = turn(cond, body(msgs, ))
        assert plan.row["effort"] == "xhigh" and "coding" in _parts(plan)
    finally:
        fitted(None)


def test_2b_the_decider_down_still_routes_the_default_packages():
    """SRV-534: Laya refused or not loaded (the decider stage errs): the Chat tab (chat-app,
    coding's default class) gets code_edit's core, Open WebUI's chat class still gets nothing."""
    fitted(dict(LFM, rules=False))
    try:
        for why in ("decider_refused", "decider_unreachable", "decider_timeout"):
            cond, fakes = setup(decider=lambda t, o, w=why: w)
            _packages(cond)
            cond.store.put("profiles", "chat-app", copy.deepcopy(
                schema.SHIPPED_PROFILES["chat-app"]), note="shipped", author="t")
            plan, _, _ = turn(cond, body([{"role": "user", "content": "make a three.js page "
                                                                      "with a red cube"}]),
                              kind="chat-app")
            row = plan.row
            assert (row["profile"], row["class_id"], row["source"], row["kept"]) == \
                ("chat-app", "code_edit", "fallback", "fail_open"), row
            assert row["stages"][0]["error"] == why
            assert _parts(plan) == ["core-recipe"] and row["injected"], _parts(plan)
            assert "rules_skipped:model_fit" in row["extra"]["notes"]
        cond, _ = setup(decider=lambda t, o: "decider_refused")
        _packages(cond)
        plan, _, _ = turn(cond, body([{"role": "user", "content": "thanks, that helped"}],
                                     tools=False), kind="open-webui")
        assert plan.row["class_id"] == "chat" and "chat-help" not in _parts(plan)
    finally:
        fitted(None)


def test_3_a_classified_turn_keeps_its_level_without_the_sentence():
    fitted(LFM)
    try:
        cond, _ = setup(decider=by_label("code_edit", 0.95))
        plan, _, _ = turn(cond, body([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": CODE}]))
        row = plan.row
        assert (row["class_id"], row["effort"]) == ("code_edit", "xhigh")
        assert "effort_note_skipped:model_fit" in row["extra"].get("notes", [])
        text = "".join(n["text"] for n in plan.injected)
        assert schema.XHIGH_NOTE not in text and "Read a file before you change it" in text
        fitted(None)
        cond, _ = setup(decider=by_label("code_edit", 0.95))
        plan, _, _ = turn(cond, body([{"role": "user", "content": CODE}]))
        assert schema.XHIGH_NOTE in plan.injected[0]["text"]
    finally:
        fitted(None)


def test_4_packages_core_alone_and_none_on_chat():
    fitted(LFM)
    try:
        cond, fakes = setup(decider=lambda t, o: ("task-skill", 0.9) if "none" in o
                            else ("code_edit", 0.95))
        _packages(cond)
        plan, _, _ = turn(cond, body([{"role": "user", "content": CODE}]))
        assert [p for p in _parts(plan) if p in ("coding", "core-recipe", "task-skill")] == \
            ["core-recipe"], _parts(plan)
        assert sorted(plan.row["extra"]["package_fit_dropped"]) == ["coding", "task-skill"]
        assert "package_choice" not in plan.row["extra"], "no package question"
        assert len(fakes["decider"].calls) == 1
        cond, _ = setup(decider=by_label("chat", 0.95))
        _packages(cond)
        plan, _, _ = turn(cond, body([{"role": "user", "content": "thanks, that helped"}],
                                     tools=False), kind="open-webui")
        assert plan.row["class_id"] == "chat" and "chat-help" not in _parts(plan)
        fitted(dict(LFM, packages="all", no_package_classes=()))
        cond, _ = setup(decider=by_label("chat", 0.95))
        _packages(cond)
        plan, _, _ = turn(cond, body([{"role": "user", "content": "thanks, that helped"}],
                                     tools=False), kind="open-webui")
        assert "chat-help" in _parts(plan)
    finally:
        fitted(None)


def test_5_memory_off():
    fitted(dict(LFM, memory=False))
    try:
        cond, _ = setup(decider=by_label("chat", 0.95))
        plan, _, _ = turn(cond, body([{"role": "user", "content": "thanks"}], tools=False),
                          kind="open-webui")
        assert plan.row["extra"]["memory"] == {"mode": "off", "why": "model_fit"}
        assert not [n for n in plan.injected if "memory" in n["kinds"]]
    finally:
        fitted(None)


def test_6_the_setting():
    fit, src = modelfit.resolve(LFM, "off")
    assert fit == modelfit.DEFAULTS and src == "off"
    fit, src = modelfit.resolve(LFM, "model")
    assert fit["served_classifier"] is False and fit["no_package_classes"] == ("chat",)
    fit, src = modelfit.resolve(LFM, "fallback=class,memory=0,no_package_classes=chat+writing")
    assert fit["fallback"] == "class" and fit["memory"] is False and src.startswith("model+")
    assert fit["no_package_classes"] == ("chat", "writing") and fit["packages"] == "core"
    fit, src = modelfit.resolve(None, "")
    assert fit == modelfit.DEFAULTS and src == "default"
    for bad in ("nope=1", "fallback=wide", "memory=maybe", "served_classifier"):
        try:
            modelfit.resolve(LFM, bad)
            raise AssertionError(f"{bad!r} was taken")
        except ValueError:
            pass
    try:
        modelfit.resolve({"typo": True}, "model")
        raise AssertionError("an adapter key the router does not know was taken")
    except ValueError:
        pass
    line = modelfit.install(LFM, "model")
    assert "served_classifier=False" in line and "fallback=lean" in line
    modelfit.install(None, "model")
    assert modelfit.FIT == modelfit.DEFAULTS and modelfit.summary() == {}


def test_8_rules_off():
    fitted(dict(LFM, rules=False))
    try:
        cond, _ = setup(decider=by_label("code_edit", 0.95))
        plan, _, _ = turn(cond, body([{"role": "user", "content": CODE}]))
        assert "rules_skipped:model_fit" in plan.row["extra"].get("notes", [])
        assert "Read a file before you change it" not in "".join(n["text"] for n in plan.injected)
    finally:
        fitted(None)


def test_7_the_lfm25_adapter_declares_a_fit():
    from engine.models.lfm2_moe import serving
    fit, src = modelfit.resolve(serving.Lfm2MoeServing.caps.conductor, "model")
    assert src == "model" and fit["served_classifier"] is False and fit["fallback"] == "lean"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
