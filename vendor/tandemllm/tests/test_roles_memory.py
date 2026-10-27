"""The in-process roles' memory beside the engine's ceiling (NP1's open issue 3).

After a restart the shadow reader's first load landed on full caches: its own pool counted as the
engine's (only the decision models and the narrator's model tensors were left out), the next
request's memory gate found the engine over its ceiling and shed the anchors, the end, the stash and
the park (6.3 GB on 2026-10-05 06:18 for a 1.47 GiB model), and the turn's class question waited
out its 8 s behind the load. The matrix:

  1  a role's own pool up to the claim the planner admitted it with is left out of the engine's
     allocated and reserved bytes; over the claim the excess counts (and its free part is the
     stale-reserve rule's foreign free); a slot without a pool or a claim leaves out its model's
     device bytes, as before; a slot the narrator and the shadow share counts once
                                                                  test_the_roles_pools_up_to_their_claims
  2  the Budget takes the pair: the gate of a request after the shadow's load sheds nothing when the
     engine itself is under its ceiling (it shed four levels with the old single figure)
                                                                  test_the_gate_after_the_shadows_load
  3  the slot: the planner's claim is kept; the pool is the role's from the load's start (a request
     served between two copies sees it); a failed load drops it; the load's log line says
     "MemAvailable +2.87 GiB" when MemAvailable rose (it read "--2.87")
                                                                  test_the_slot_during_its_load
  4  the first load ahead of the first turn: in an idle window, once, only with a profile that has
     the shadow on, never while a request runs or another model loads, never for a slot the
     narrator's keeper owns                                       test_the_shadow_loads_ahead

Run: python tests/test_roles_memory.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.cache import GIB, Budget  # noqa: E402
from engine.generative.fake import FakeGenerative  # noqa: E402
from engine.generative.slot import Slot, _took  # noqa: E402
from server import memlimits  # noqa: E402


class _Slot:
    def __init__(self, res_gib=0.0, alloc_gib=0.0, claim_gib=None, device_gib=0.0):
        self.res, self.alloc = int(res_gib * GIB), int(alloc_gib * GIB)
        self.claim_gib, self.dev = claim_gib, int(device_gib * GIB)

    def pool_bytes(self):
        return self.res, self.alloc

    def device_bytes(self):
        return self.dev


def _owner(slot):
    return types.SimpleNamespace(slot=slot)


def test_the_roles_pools_up_to_their_claims():
    narr = _Slot(res_gib=4.0, alloc_gib=3.5, claim_gib=6.0, device_gib=3.57)
    shadow = _Slot(res_gib=8.0, alloc_gib=5.0, claim_gib=6.5, device_gib=1.47)
    st = {"narrator": _owner(narr), "shadow": _owner(shadow)}
    a, r = memlimits.role_exclude(st)
    assert (a, r) == (int(3.5 * GIB) + int(5.0 * GIB), int(4.0 * GIB) + int(6.5 * GIB)), (a, r)
    # the shadow's pool is 1.5 GiB over its claim, all of it free: that part counts as the
    # engine's reserve and is the foreign free the stale-reserve rule leaves out
    assert memlimits.role_pool_free(st) == int(8.0 * GIB) - int(6.5 * GIB)
    assert memlimits._role_bytes(st) == r
    # a pool both over the claim allocated and reserved: the excess counts, its free part too
    shadow.alloc = int(7.0 * GIB)
    a, r = memlimits.role_exclude(st)
    assert a == int(3.5 * GIB) + int(6.5 * GIB) and r == int(4.0 * GIB) + int(6.5 * GIB)
    assert memlimits.role_pool_free(st) == int(1.0 * GIB)
    # without a claim (no planner) or a pool (the CPU): the model's device bytes, as before
    st = {"narrator": _owner(_Slot(res_gib=4.0, alloc_gib=3.5, device_gib=3.0)),
          "shadow": _owner(_Slot(claim_gib=6.5, device_gib=1.5))}
    assert memlimits.role_exclude(st) == (int(4.5 * GIB), int(4.5 * GIB))
    assert memlimits.role_pool_free(st) == int(0.5 * GIB)
    # one slot the narrator and the shadow share counts once
    both = _Slot(res_gib=4.0, alloc_gib=3.0, claim_gib=6.0)
    st = {"narrator": _owner(both), "shadow": _owner(both)}
    assert memlimits.role_exclude(st) == (int(3.0 * GIB), int(4.0 * GIB))
    # the decision models' tensors, as before
    reg = types.SimpleNamespace(status=lambda: [{"device_bytes": int(2 * GIB)}])
    assert memlimits.role_exclude({"deciders": reg}) == (int(2 * GIB), int(2 * GIB))
    assert memlimits.role_exclude({}) == (0, 0) and memlimits.role_pool_free({}) == 0


class _Alloc:
    def __init__(self, alloc_gib, res_gib):
        self.alloc, self.res = int(alloc_gib * GIB), int(res_gib * GIB)

    def probe(self):
        return self.alloc, self.res

    def empty(self):
        pass

    def peak(self, reset=False):
        return self.alloc


def test_the_gate_after_the_shadows_load():
    """06:18 on the box, in round numbers: the engine holds 68.6 GiB reserved with its caches
    full, the shadow's pool adds 2 GiB (its model, its KV, its graph) inside a 6.5 GiB claim."""
    fa = _Alloc(66.0, 68.6)
    shadow = _Slot(res_gib=0.0, alloc_gib=0.0, claim_gib=6.5, device_gib=0.0)
    st = {"shadow": _owner(shadow)}
    shed = []

    def mk(exclude):
        b = Budget(int(70 * GIB), int(77 * GIB), int(2 * GIB), probe=fa.probe, empty=fa.empty,
                   peak_probe=fa.peak, exclude=exclude)
        for lv in ("store", "anchors", "end", "stash", "park"):
            b.register(lv, lambda safe, lv=lv: shed.append(lv) or int(0.1 * GIB))
        return b
    fa.alloc += int(2.0 * GIB)
    fa.res += int(2.0 * GIB)
    shadow.res = shadow.alloc = int(2.0 * GIB)
    mk(lambda: memlimits.role_exclude(st)).gate(1000)
    assert shed == [], shed
    assert mk(lambda: memlimits.role_exclude(st)).memory() == (int(66.0 * GIB), int(68.6 * GIB))
    # the old single figure, which left the shadow's own slot out: the same moment sheds
    mk(lambda: 0).gate(1000)
    assert shed[:4] == ["store", "anchors", "end", "stash"], shed


def test_the_slot_during_its_load():
    from engine.decision.device import Device
    seen = {}
    m = FakeGenerative()
    s = Slot(m, role="shadow", gate=threading.Lock(), device=Device("cpu"), log=lambda x: None,
             planner=lambda name, **kw: {"allowed": True, "reason": None, "role": "shadow",
                                         "claim_gib": 6.5})
    real = m.load

    def load(where):
        seen["where"] = s._where                  # the pool is the slot's from the start
        seen["state"] = s.state
        real(where)
    m.load = load
    s.load(wait=True)
    assert s.state == "loaded" and s.claim_gib == 6.5, (s.state, s.claim_gib)
    assert seen["where"] is not None and seen["state"] == "loading", seen
    assert s._where is seen["where"]
    s.unload()
    assert s._where is None
    # a failed load drops its placement
    m2 = FakeGenerative()

    def boom(where):
        raise RuntimeError("no weights")
    m2.load = boom
    s2 = Slot(m2, role="shadow", gate=threading.Lock(), device=Device("cpu"), log=lambda x: None)
    s2.load(wait=True)
    assert s2.state == "failed" and s2._where is None and s2.claim_gib is None
    assert _took(1.47) == ", MemAvailable -1.47 GiB"
    assert _took(-2.87) == ", MemAvailable +2.87 GiB"
    assert _took(None) == ""


class _Store:
    def __init__(self, profiles):
        self.profiles = profiles

    def names(self):
        return sorted(self.profiles)

    def peek(self, kind, name=None):
        return 1, self.profiles[name]


def test_the_shadow_loads_ahead():
    import contextlib
    import io
    from server import app, decider_api
    os.environ["BOX_LOCK"] = os.path.join(tempfile.gettempdir(), f"l2-box-lock-{os.getpid()}")
    keep = {k: app.STATE.get(k) for k in ("shadow", "narrator", "conductor", "deciders",
                                          "draining", "shadow_preload")}
    m = FakeGenerative("fake-shadow")
    s = Slot(m, role="shadow", gate=threading.Lock(), log=lambda x: None,
             planner=lambda name, **kw: {"allowed": True, "reason": None, "role": "shadow",
                                         "claim_gib": 6.5})
    profiles = {"coding": {"shadow": {"mode": "off"}}, "chat": {}}
    try:
        app.STATE.update(shadow=types.SimpleNamespace(slot=s), narrator=None, deciders=None,
                         draining=False, shadow_preload=None,
                         conductor=types.SimpleNamespace(store=_Store(profiles)))
        assert app.shadow_preload_once() is None and s.state == "not_loaded"   # no profile on
        profiles["coding"] = {"shadow": {"mode": "on"}}
        with app.QUEUE:
            app.INFLIGHT["running"] += 1
        try:
            assert app.shadow_preload_once() is None and s.state == "not_loaded"  # a request runs
        finally:
            with app.QUEUE:
                app.INFLIGHT["running"] -= 1
        other = types.SimpleNamespace(loading=lambda: True)
        app.STATE["deciders"] = other                   # a decision model is loading
        assert decider_api._other_loading(app.STATE, "shadow")
        assert app.shadow_preload_once() is None and s.state == "not_loaded"
        app.STATE["deciders"] = None
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got = app.shadow_preload_once()
        assert got == "loading", got
        assert "first load ahead of the first turn: loading" in out.getvalue(), out.getvalue()
        for _ in range(200):
            if s.state == "loaded":
                break
            app.time.sleep(0.01)
        assert s.state == "loaded" and m.loads == 1 and s.loads == 1
        s.unload()                                      # a release: the first turn reloads it
        assert s.loaded_at is None and s.state == "not_loaded"
        assert app.shadow_preload_once() is None and m.loads == 1
        # a slot the narrator owns is its keeper's
        n = Slot(FakeGenerative("fake-shared"), role="narrator", gate=threading.Lock(),
                 log=lambda x: None)
        mode = {"mode": "on"}
        app.STATE.update(shadow=types.SimpleNamespace(slot=n),
                         narrator=types.SimpleNamespace(slot=n, settings=lambda: mode))
        assert app.shadow_preload_once() is None and n.state == "not_loaded"
        # SRV-534: the narrator role off (the default): its keeper never loads, the shadow does
        mode["mode"] = "off"
        assert app.shadow_preload_once() == "loading"
        for _ in range(200):
            if n.state == "loaded":
                break
            app.time.sleep(0.01)
        assert n.state == "loaded", n.state
        n.unload()
        # a refusal is said once, with its reason
        r = Slot(FakeGenerative("fake-refused"), role="shadow", gate=threading.Lock(),
                 log=lambda x: None,
                 planner=lambda name, **kw: {"allowed": False, "reason": "refused_memory",
                                             "role": "shadow"})
        app.STATE.update(shadow=types.SimpleNamespace(slot=r), narrator=None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert app.shadow_preload_once() == "refused"
            assert app.shadow_preload_once() == "refused"
        assert out.getvalue().count("refused (refused_memory)") == 1, out.getvalue()
    finally:
        app.STATE.update(keep)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
