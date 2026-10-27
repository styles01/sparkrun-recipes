"""ENG-179 / ENG-180: the decision-model runtime (engine/decision), stage A, with the fake model.

The test matrix is ENG-179's Gherkin (ENG-180's table); row 1 (the SDK equivalence) is in
tests/test_decider_laya.py and on the box (tools/decider_equivalence.py).

  2  any decision model works         test_2_* (the fake model "other-decider", limits 256 and 96)
  3  the planner refuses a load        test_3_* (MemAvailable 23 GiB, claim 3.0: refused_memory,
                                       nothing loaded, residency.jsonl has it)
  4  no load during a hold             test_4_* (the hold's pause file; the box lock)
  5  no two loads at once              test_5_* (a load held open in its thread)
  6  questions only in the idle window test_6_* (the engine lock held: no forward; the deadline
                                       drops the question; in_decode refused)
  7  callers in one idle window        test_7_* (each in its own forward by default; collate=rows:
                                       one forward)
  8  off means not imported            test_8_* (a fresh interpreter, the flag empty)
  9  a failing model is isolated       test_9_* (3 errors in a minute: unloaded, failed)

Run: python tests/test_decider_runtime.py
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
# the box lock of the machine this runs on is not this test's: a hold there must not read as one
os.environ["BOX_LOCK"] = os.path.join(tempfile.mkdtemp(prefix="dec-lock-"), "no-box.flock")

from engine.decision import catalogue  # noqa: E402
from engine.decision.base import (DeadlineMissed, NotReadyError, OverloadedError,  # noqa: E402
                                  WindowError)
from engine.decision.fake import FakeModel  # noqa: E402
from engine.decision.registry import Registry, UnknownModel  # noqa: E402
from server import decider_api  # noqa: E402
from server.conductor import residency, schema  # noqa: E402

CHOICE = {"cls": {"type": "choice", "instructions": "Which kind of work is this?",
                  "criteria": {"debug": "fix a bug", "plan": "plan the work", "chat": "talk"}}}


def _reg(*models, planner=None, policy=None, gate=None, **kw):
    models = models or (FakeModel("other-decider"),)
    return Registry(list(models), gate=gate or threading.Lock(), planner=planner,
                    policy=policy, log=lambda s: None, **kw)


def _loaded(reg, name):
    reg.load(name, wait=True)
    assert reg.state[name] == "loaded", (name, reg.state[name], reg.error.get(name))


# ------------------------------------------------------------------ 2: any decision model works
def test_2_any_decision_model_works():
    m = FakeModel("other-decider", context_tokens=256, question_tokens=96)
    reg = _reg(m)
    _loaded(reg, "other-decider")
    assert reg.limits("other-decider") == {"context_tokens": 256, "question_tokens": 96,
                                           "max_options": 20}
    assert reg.limits("auto") == reg.limits("other-decider"), "auto: the tightest loaded"
    out = reg.ask("It crashes on start. debug please", CHOICE, model="auto").result(5)
    a = out["answers"]["cls"]
    assert out["model"] == "other-decider" and a["type"] == "choice" and a["choice"] == "debug"
    assert set(a["probabilities"]) == {"debug", "plan", "chat"} and 0 < a["confidence"] <= 1
    assert out["usage"]["output_tokens"] == 0 and out["path"] == "eager"
    assert out["routing"]["reason"] == "the fake family's first model"
    # a noul and a score in the same call, one forward
    qs = {"n": {"type": "noul", "instructions": "It is urgent."},
          "s": {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]}}
    got = reg.ask("FAKE_TRUE", qs, model="other-decider").result(5)["answers"]
    assert got["n"]["noul"] == 0.9 and got["s"]["type"] == "score"
    assert m.forwards[-1] == 2
    try:
        reg.ask("x", CHOICE, model="laya-english")
        raise AssertionError("an unknown model must be refused")
    except UnknownModel:
        pass


def test_2_fake_names_need_a_test_or_the_fake_engine():
    try:
        catalogue(["other-decider"], "/nowhere")
        raise AssertionError("a fake model outside a test")
    except ValueError as exc:
        assert "test double" in str(exc)
    try:
        catalogue(["laya-huge"], "/nowhere")
        raise AssertionError("an unknown name stops the start")
    except ValueError as exc:
        assert "unknown decision model" in str(exc)
    assert [m.name for m in catalogue(["fake-decider"], "/x", allow_fake=True)] == ["fake-decider"]


# ------------------------------------------------------------------ 3, 4, 5: the planner
def _planner_on(mem_gib, root, *, hold_file=None):
    """server/decider_api.py's planner with MemAvailable pinned, journaling into `root`."""
    state = {}

    class Cond:
        def __init__(self):
            from server.conductor.store import Store
            self.root, self.store = root, Store(root)

    state["conductor"] = Cond()
    orig = decider_api._mem_available_gib
    decider_api._mem_available_gib = lambda fake: mem_gib
    try:
        return decider_api.planner(state), lambda: setattr(decider_api, "_mem_available_gib", orig)
    except Exception:
        decider_api._mem_available_gib = orig
        raise


def test_3_the_planner_refuses_a_load_that_breaks_the_floor():
    root = tempfile.mkdtemp(prefix="dec-res-")
    plan, restore = _planner_on(23.0, root)
    try:
        m = FakeModel("other-decider")
        reg = _reg(m, planner=plan)
        said = []
        reg.log = said.append
        d = reg.load("other-decider", wait=True)
        assert not d["allowed"] and d["reason"] == "refused_memory", d
        assert d["claim_gib"] == 3.0 and d["need_gib"] == 21.0 and d["after_gib"] == 20.0, d
        # SRV-534: with no model loaded the refusal is said once, loudly, and /health has it
        loud = [x for x in said if "WARNING: no decision model is loaded" in x]
        assert len(loud) == 1 and "refused_memory" in loud[0] and "20.0 GiB" in loud[0], said
        r = reg.refused()
        assert (r["model"], r["reason"], r["after_gib"], r["need_gib"]) == \
            ("other-decider", "refused_memory", 20.0, 21.0), r
        assert m.loads == 0 and reg.state["other-decider"] == "not_loaded"
        assert reg.refusal["other-decider"]["reason"] == "refused_memory"
        last = residency.Journal(root).last_refusal()
        assert last["source"] == "engine:other-decider" and last["reason"] == "refused_memory"
        lines = len(residency.Journal(root).tail(100))
        for _ in range(3):
            try:
                reg.ask("x", CHOICE, model="other-decider")
                raise AssertionError("a question for a refused model")
            except NotReadyError as exc:
                assert exc.why == "refused" and exc.status == 503 and exc.retry
        assert len(residency.Journal(root).tail(100)) == lines, \
            "questions reuse a fresh refusal: no planner check, no journal line, per request"
        reg.load("other-decider", wait=True)               # an explicit Load asks again
        assert len(residency.Journal(root).tail(100)) == lines + 1
        assert len([x for x in said if "WARNING" in x]) == 1, "the loud line goes out once"
    finally:
        restore()
    plan, restore = _planner_on(40.0, root)
    try:
        reg = _reg(FakeModel("other-decider"), planner=plan)
        assert reg.load("other-decider", wait=True)["allowed"], "40 - 3 >= 21"
        assert reg.state["other-decider"] == "loaded"
        assert reg.refused() is None, "a loaded model: nothing to report"
    finally:
        restore()


def test_4_no_load_during_a_hold():
    repo = tempfile.mkdtemp(prefix="dec-hold-")
    assert not decider_api.held(repo=repo, lock_path=os.path.join(repo, "none"))
    open(os.path.join(repo, ".watchdog.off"), "w").close()
    assert decider_api.held(repo=repo)
    assert decider_api.held(repo=repo, service_start=True), "the pause holds at a service start"
    assert not decider_api.held(repo=repo, hold_restart=True), "the hold's own restart"
    os.remove(os.path.join(repo, ".watchdog.off"))
    lock = os.path.join(repo, "box.flock")
    open(lock, "w").close()
    assert not decider_api.held(repo=repo, lock_path=lock), "a free lock is no hold"
    fd = os.open(lock, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert decider_api.held(repo=repo, lock_path=lock), "someone holds the box lock"
        assert not decider_api.held(repo=repo, lock_path=lock, service_start=True), \
            "the watchdog's restart holds the lock itself"
    finally:
        os.close(fd)
    # and through the planner: refused_hold before any number is read
    d = residency.check("decider", 80.0, schema.seed("residency"), schema.seed("roles"), hold=True)
    assert d["reason"] == "refused_hold"

    def plan(name, *, loading, service_start, hold_restart):
        return residency.check("decider", 80.0, schema.seed("residency"), schema.seed("roles"),
                               hold=True, loading=loading)

    reg = _reg(planner=plan)
    d = reg.load("other-decider", wait=True)
    assert d["reason"] == "refused_hold" and reg.state["other-decider"] == "not_loaded"


def test_5_no_two_loads_at_once():
    gate = threading.Event()

    class Slow(FakeModel):
        def load(self, device):
            gate.wait(5)
            super().load(device)

    seen = []

    def plan(name, *, loading, service_start, hold_restart):
        seen.append(loading)
        return residency.check("decider", 80.0, schema.seed("residency"), schema.seed("roles"),
                               loading=loading)

    reg = _reg(Slow("other-decider"), FakeModel("fake-two"), planner=plan)
    assert reg.load("other-decider")["allowed"]
    d = reg.load("fake-two")
    assert d["reason"] == "refused_load_in_progress" and reg.state["fake-two"] == "not_loaded", d
    gate.set()
    for _ in range(100):
        if reg.state["other-decider"] == "loaded":
            break
        time.sleep(0.02)
    assert reg.load("fake-two", wait=True)["allowed"] and seen == [False, True, False]


# ------------------------------------------------------------------ 6: the idle window
def test_6_no_forward_while_a_request_holds_the_engine():
    lock = threading.Lock()
    m = FakeModel("other-decider")
    reg = _reg(m, gate=lock)
    _loaded(reg, "other-decider")
    n0 = len(m.forwards)
    lock.acquire()                                   # a completion holds the engine
    try:
        fut = reg.ask("debug", CHOICE, deadline_ms=2000)
        time.sleep(0.3)
        assert len(m.forwards) == n0 and not fut.done(), "a forward ran beside the request"
    finally:
        lock.release()
    assert fut.result(5)["answers"]["cls"]["choice"] == "debug"
    assert len(m.forwards) == n0 + 1


def test_6_the_deadline_drops_a_question_the_engine_never_had_time_for():
    lock = threading.Lock()
    m = FakeModel("other-decider")
    reg = _reg(m, gate=lock)
    _loaded(reg, "other-decider")
    n0 = len(m.forwards)
    lock.acquire()
    t0 = time.monotonic()
    try:
        fut = reg.ask("debug", CHOICE, deadline_ms=100)
        try:
            fut.result(5)
            raise AssertionError("ran past its deadline")
        except DeadlineMissed as exc:
            took = time.monotonic() - t0
            assert 0.09 <= took < 1.0, took
            assert exc.why == "deadline_missed"
    finally:
        lock.release()
    time.sleep(0.05)
    assert len(m.forwards) == n0, "a dropped question never runs late"
    assert reg.status()[0]["stats"]["deadline_missed"] == 1


def test_6_in_decode_questions_are_stage_c():
    reg = _reg()
    _loaded(reg, "other-decider")
    try:
        reg.ask("x", CHOICE, window="in_decode")
        raise AssertionError("in_decode must be refused in stage A")
    except WindowError as exc:
        assert exc.status == 422 and "stage C" in str(exc)


# ------------------------------------------------------------------ 7: two callers, one idle window
def _two_callers(collate):
    lock = threading.Lock()
    m = FakeModel("other-decider")
    reg = _reg(m, gate=lock, wait_ms=50, collate=collate)
    _loaded(reg, "other-decider")
    n0 = len(m.forwards)
    a = {"a1": dict(CHOICE["cls"]), "a2": {"type": "noul", "instructions": "It is a bug."}}
    b = {"b1": dict(CHOICE["cls"])}
    lock.acquire()                     # both queue while a request holds the engine
    fa = reg.ask("plan FAKE_TRUE", a, deadline_ms=3000)
    fb = reg.ask("chat", b, deadline_ms=3000)
    lock.release()
    ra, rb = fa.result(5), fb.result(5)
    assert list(ra["answers"]) == ["a1", "a2"] and ra["answers"]["a1"]["choice"] == "plan"
    assert ra["answers"]["a2"]["noul"] == 0.9
    assert list(rb["answers"]) == ["b1"] and rb["answers"]["b1"]["choice"] == "chat"
    return m.forwards[n0:], ra, rb


def test_7_callers_are_answered_in_one_idle_window_each_in_its_own_forward():
    """The default (`collate=call`): each caller's rows in a forward of their own, the SDK's
    forward for that call, so an answer never depends on who else asked (on the GB10 the
    collated forward moved a probability by 3.4e-3)."""
    forwards, ra, rb = _two_callers("call")
    assert forwards == [2, 1], f"one forward per caller, got {forwards}"
    assert (ra["batch_rows"], rb["batch_rows"]) == (2, 1)


def test_7_collate_rows_puts_the_callers_in_one_forward():
    forwards, ra, rb = _two_callers("rows")
    assert forwards == [3], f"one forward of three rows, got {forwards}"
    assert ra["batch_rows"] == rb["batch_rows"] == 3


def test_7_a_full_queue_is_overloaded():
    reg = _reg(max_queue=2)
    _loaded(reg, "other-decider")
    lock = reg.gate
    lock.acquire()
    try:
        reg.ask("x", {"q1": CHOICE["cls"], "q2": CHOICE["cls"]}, deadline_ms=2000)
        try:
            reg.ask("x", CHOICE, deadline_ms=2000)
            raise AssertionError("over the queue cap")
        except OverloadedError as exc:
            assert exc.status == 529
    finally:
        lock.release()


# ------------------------------------------------------------------ 8: off means not imported
def test_8_off_means_not_imported():
    code = ("import sys, json; sys.modules['triton'] = None; sys.modules['triton.language'] = None\n"
            "import server.app as app\n"
            "from server import decider_api\n"
            "reg = decider_api.start(app.STATE, app.LOCK)\n"
            "print(json.dumps({'reg': reg is None, 'mods': sorted(m for m in sys.modules "
            "if m.startswith('engine.decision')), 'threads': sorted(t.name for t in "
            "__import__('threading').enumerate())}))\n")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ["HOME"],
           "PYTHONPATH": ROOT, "CUDA_VISIBLE_DEVICES": "", "QWEN38_DECIDER_MODELS": ""}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["reg"] and out["mods"] == [], out
    assert not any(t.startswith("decider") for t in out["threads"]), out


# ------------------------------------------------------------------ 9: isolation
def test_9_a_failing_model_is_unloaded_and_marked_failed():
    m = FakeModel("other-decider")
    reg = _reg(m)
    _loaded(reg, "other-decider")
    m.fail = True
    for _ in range(3):
        try:
            reg.ask("x", CHOICE, deadline_ms=2000).result(5)
            raise AssertionError("the error must reach the caller")
        except RuntimeError as exc:
            assert "on purpose" in str(exc)
    for _ in range(100):
        if reg.state["other-decider"] == "failed":
            break
        time.sleep(0.02)
    assert reg.state["other-decider"] == "failed" and not m.loaded
    assert "3 failed forwards" in reg.error["other-decider"]
    try:
        reg.ask("x", CHOICE)
        raise AssertionError("a failed model answers nothing")
    except NotReadyError as exc:
        assert exc.why == "failed"
    m.fail = False
    _loaded(reg, "other-decider")                      # an operator loads it again
    assert reg.ask("debug", CHOICE).result(5)["answers"]["cls"]["choice"] == "debug"


# ------------------------------------------------------------------ the rest of the registry
def test_auto_falls_back_to_the_loaded_model_and_loads_the_wanted_one():
    routed = []

    def router(state, questions, *, task=None, lang=None):
        routed.append(state)
        return {"model": "fake-multi", "reason": "non-Latin script", "detection": None,
                "workflow": None}

    one, two = FakeModel("fake-one"), FakeModel("fake-multi")
    reg = _reg(one, two, routers={"fake": router},
               policy=lambda: {"fake-one": "on_start", "fake-multi": "on_demand"})
    _loaded(reg, "fake-one")
    out = reg.ask("Привет", CHOICE).result(5)
    assert out["model"] == "fake-one" and out["routing"]["requested"] == "fake-multi"
    assert "not loaded here, falling back to fake-one" in out["routing"]["reason"]
    for _ in range(100):                                    # the wanted one loads on demand
        if reg.state["fake-multi"] == "loaded":
            break
        time.sleep(0.02)
    assert reg.state["fake-multi"] == "loaded"
    assert reg.ask("Привет", CHOICE).result(5)["model"] == "fake-multi"


def test_load_policies_and_autoload():
    a, b, c = FakeModel("fake-a"), FakeModel("fake-b"), FakeModel("fake-c")
    reg = _reg(a, b, c, policy=lambda: {"fake-a": "on_start", "fake-b": "on_demand",
                                        "fake-c": "off"})
    assert reg.autoload() == ["fake-a"]
    for _ in range(100):
        if reg.state["fake-a"] == "loaded":
            break
        time.sleep(0.02)
    assert reg.state == {"fake-a": "loaded", "fake-b": "not_loaded", "fake-c": "not_loaded"}
    try:
        reg.ask("x", CHOICE, model="fake-c")
        raise AssertionError("off is never loaded by a question")
    except NotReadyError as exc:
        assert exc.why == "not_loaded" and exc.retry is None and reg.state["fake-c"] == "not_loaded"
    try:
        reg.ask("x", CHOICE, model="fake-b")
        raise AssertionError("on_demand answers 503 loading first")
    except NotReadyError as exc:
        assert exc.why == "loading"
    assert reg.unload("fake-a") and reg.state["fake-a"] == "not_loaded" and not a.loaded
    assert not reg.unload("fake-a"), "twice"


def test_status_reports_what_the_screen_needs():
    reg = _reg(FakeModel("other-decider"))
    _loaded(reg, "other-decider")
    reg.ask("debug", CHOICE, source="classify").result(5)
    s = reg.status()[0]
    for k in ("name", "family", "contract", "state", "load", "limits", "precision", "device",
              "device_bytes", "weight_hash", "warm", "loaded_at", "questions", "error",
              "refusal", "stats"):
        assert k in s, k
    assert s["warm"] is True and s["questions"] == 1
    assert s["stats"]["forwards"] >= 2 and s["stats"]["forward_ms_p50"] is not None
    assert reg.picks_by_source() == {"classify": {"other-decider": 1}}


def test_a_cuda_device_carries_its_index():
    """`torch.cuda.use_mem_pool` refuses a bare `cuda` (found by the first GPU run): the device
    the runtime places models on always has its index. No GPU needed: the index is asked of a
    stub."""
    import torch
    from engine.decision import device as dmod
    real = torch.cuda.current_device
    torch.cuda.current_device = lambda: 0
    try:
        assert dmod.resolve("cuda") == torch.device("cuda", 0)
        assert dmod.resolve("cuda:0").index == 0
        assert dmod.resolve("cpu") == torch.device("cpu")
    finally:
        torch.cuda.current_device = real


def test_the_seed_documents():
    d = schema.seed("deciders")
    assert [m["name"] for m in d["models"]][0] == "laya-english"
    assert {m["name"]: m["load"] for m in d["models"]}["laya-english"] == "on_start"
    assert schema.check("deciders", d) == []
    bad = dict(d, models=d["models"] + [{"name": "laya-english", "load": "off"}])
    assert "a model is listed twice" in " ".join(schema.check("deciders", bad))
    assert schema.check("deciders", dict(d, models=[{"name": "x", "load": "sometimes"}]))
    roles = schema.seed("roles")
    assert roles["decider"]["endpoint"] == "self" and roles["decider"]["contract"] == "jev"
    assert residency.claims(schema.seed("residency"), roles)["decider"]["gib"] == 3.0


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
