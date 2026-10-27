"""SRV-131 / SRV-132 rows 7, 8 and 10: the conductor's decider asks the engine's own runtime.

No server runs in this test at all: the conductor's router classifies a request with its decider
role at `endpoint: self` and the answer comes from the registry in this process (the fake model
"other-decider"), which is what "no HTTP request reaches /v1/systemone" means.

  7   the conductor asks in process     test_7_* (self and engine:<family>; the picks journal)
  8   the model per question            test_8_* (the deciders document maps classify, and
                                        packages, to a model; the role's model otherwise)
  10  fail open                         test_10_* (no registry: decider_no_endpoint, Qwen
                                        classifies; a deadline missed under the engine lock fails
                                        open the same way)

plus the role's health and the role test through the in-process client.

Run: python tests/test_decider_conductor.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ["BOX_LOCK"] = os.path.join(tempfile.mkdtemp(prefix="dec-lock-"), "no-box.flock")
os.environ["FAKE_MEM_AVAILABLE_GIB"] = "64"

from test_conductor_router import QTok, body, turn  # noqa: E402

from server import app, decider_api  # noqa: E402
from server.conductor import classify  # noqa: E402
from server.conductor.classify import Answer  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402

DEBUG = "It crashes on start with a stack trace in the parser. debug it please."


def _state():
    app.STATE.clear()
    app.STATE.update(tok=QTok(), device="cpu", reasoning_effort="medium", segments=None,
                     model="qwen38-spark-engine")
    return app.STATE


def _engine(models="other-decider", lock=None, factory=None):
    """A conductor and, with `models`, the engine's decision runtime as server/app.py builds it
    (decider_api.start), every model loaded."""
    for k in list(classify.BACKENDS):
        classify.BACKENDS.pop(k)
    state = _state()
    cond = state["conductor"] = Conductor(tempfile.mkdtemp(prefix="dec-cond-"), state=state,
                                          journal=False, factory=factory)
    reg = None
    if models:
        os.environ["QWEN38_DECIDER_MODELS"] = models
        try:
            reg = state["deciders"] = decider_api.start(state, lock or threading.Lock(), fake=True,
                                                        log=lambda *a, **k: None)
        finally:
            os.environ.pop("QWEN38_DECIDER_MODELS", None)
        for n in reg.order:
            reg.load(n, wait=True)
            assert reg.state[n] == "loaded", reg.error
    _, roles = cond.store.get("roles")
    roles["classifier"]["mode"] = "off"            # the decider alone answers, or nothing does
    cond.store.put("roles", None, roles, note="decider alone", author="t")
    return cond, reg


def test_7_the_conductor_asks_in_process():
    cond, reg = _engine()
    assert cond.store.get("roles")[1]["decider"]["endpoint"] == "self", "the seed: in process"
    assert app.STATE.get("port") is None, "no server: there is no port to ask over HTTP"
    plan, _, _ = turn(cond, body([{"role": "user", "content": DEBUG}]))
    st = plan.row["stages"][0]
    assert (plan.row["class_id"], plan.row["source"]) == ("debug", "decider"), plan.row
    assert st["model"] == "other-decider" and st["error"] is None and st["confidence"] >= 0.8, st
    assert reg.picks_by_source() == {"classify": {"other-decider": 1}}
    # engine:<family> names the same runtime
    _, roles = cond.store.get("roles")
    roles["decider"]["endpoint"] = "engine:fake"
    cond.store.put("roles", None, roles, note="by family", author="t")
    plan, _, _ = turn(cond, body([{"role": "user", "content": "Now plan the next release."}]),
                      headers={"x-session-id": "ses-2"})
    assert (plan.row["class_id"], plan.row["source"]) == ("plan", "decider"), plan.row


def test_7_health_and_the_role_test_go_through_the_registry():
    cond, reg = _engine()
    h = cond.roles_health(force=True)["decider"]
    assert h["up"] and h["detail"]["models"] == ["other-decider"], h
    assert h["limits"] == {"context_tokens": 256, "question_tokens": 96, "max_options": 20}
    t = cond.role_test("decider")
    assert t["ok"] and t["answer"]["model"] == "other-decider", t
    reg.unload("other-decider")
    h = cond.roles_health(force=True)["decider"]
    assert not h["up"] and h["error"] == "no decision model loaded", h


def test_7_states_are_sized_for_the_models_limits():
    """Model-agnostic: the in-process client reports the model's limits, so the digest is cut
    to (256 - 96) x 3.5 characters, as for any decider."""
    cond, reg = _engine()
    seen = []
    m = reg.models["other-decider"]
    orig = m.build_rows
    m.build_rows = lambda state, qs: (seen.append(state), orig(state, qs))[1]
    turn(cond, body([{"role": "user", "content": DEBUG + " x" * 2000}]))
    assert seen and len(seen[0]) <= int((256 - 96) * 3.5), len(seen[0])


def test_8_the_model_per_question_comes_from_the_deciders_document():
    cond, reg = _engine("other-decider,fake-two")
    _, d = cond.store.get("deciders")
    d["questions"]["classify"] = "fake-two"
    cond.store.put("deciders", None, d, note="classify by fake-two", author="t")
    plan, _, _ = turn(cond, body([{"role": "user", "content": DEBUG}]))
    assert plan.row["stages"][0]["model"] == "fake-two", plan.row["stages"]
    assert reg.picks_by_source()["classify"] == {"fake-two": 1}
    d["questions"]["classify"] = "auto"
    cond.store.put("deciders", None, d, note="back to auto", author="t")
    plan, _, _ = turn(cond, body([{"role": "user", "content": DEBUG}]),
                      headers={"x-session-id": "ses-3"})
    assert plan.row["stages"][0]["model"] == "other-decider", "auto: the family's first model"


def test_8_the_package_question_is_asked_as_packages():
    cond, reg = _engine()
    stages = cond.router.cascade("pick one", {"ui-design": "layout", "none": "nothing fits"},
                                 cond.store.get("router")[1], cond.store.get("roles")[1],
                                 question="Which fits?", accept=0.6, qid="packages")[1]
    assert stages[0]["model"] == "other-decider"
    assert "packages" in reg.picks_by_source()


class FakeQwen:
    def __init__(self):
        self.calls = 0

    def choice(self, state, prompt, options, max_chars=None):
        self.calls += 1
        return Answer("qwen", "debug", 0.97, {"debug": 0.97}, 500.0, model="qwen38-spark-engine")


def _with_qwen():
    qwen = FakeQwen()

    def factory(kind, role, timeout):
        if kind == "decider":
            return classify.decider_for(role.get("endpoint") or "", role.get("model") or "auto",
                                        timeout)
        return qwen
    return factory, qwen


def test_10_fail_open_without_the_runtime():
    factory, qwen = _with_qwen()
    cond, reg = _engine(models="", factory=factory)
    assert reg is None and "self" not in classify.BACKENDS
    _, roles = cond.store.get("roles")
    roles["classifier"]["mode"] = "on"
    cond.store.put("roles", None, roles, note="qwen on", author="t")
    plan, _, _ = turn(cond, body([{"role": "user", "content": DEBUG}]))
    st = plan.row["stages"]
    assert st[0]["error"] == "decider_no_endpoint", st
    assert (plan.row["class_id"], plan.row["source"]) == ("debug", "qwen") and qwen.calls == 1
    h = cond.roles_health(force=True)["decider"]
    assert not h["up"] and "QWEN38_DECIDER_MODELS is empty" in h["error"], h


def test_10_a_deadline_missed_under_the_engine_lock_fails_open():
    """A question asked while a completion holds the engine waits at most the stage's timeout
    (400 ms) and passes to the next stage with the runtime's own reason."""
    factory, qwen = _with_qwen()
    lock = threading.Lock()
    cond, reg = _engine(lock=lock, factory=factory)
    _, roles = cond.store.get("roles")
    roles["classifier"]["mode"] = "on"
    cond.store.put("roles", None, roles, note="qwen on", author="t")
    lock.acquire()
    try:
        plan, _, _ = turn(cond, body([{"role": "user", "content": DEBUG}]))
    finally:
        lock.release()
    st = plan.row["stages"][0]
    assert st["error"] in ("decider_deadline_missed", "decider_timeout"), st
    assert 350 <= st["ms"] < 1500, st
    assert plan.row["source"] == "qwen" and qwen.calls == 1


if __name__ == "__main__":
    passed = 0
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print(f"  {name:72s} ok")
                passed += 1
    finally:
        classify.BACKENDS.clear()
    print(f"{passed} passed")
