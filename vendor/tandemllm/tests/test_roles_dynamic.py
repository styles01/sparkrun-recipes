"""The small models under memory pressure (server/rolemem.py) and the memory split with the shared
KV pool (server/memsplit.py). The matrix:

  1  the KV pool's reclaim order is caches -> roles -> idle side conversations -> slot 0's -> the
     kept park (ENG-371): the roles level unloads the shadow reader first, then the narrator, and
     stops once enough came back (no conversation is parked when the roles were enough)
                                                       test_the_pool_reclaims_roles_before_conversations
  2  a role out for memory stays out: `Slot.ensure` says `unloaded_for_memory` and starts no
     load, `Slot.load` refuses, the narrator's keeper and the shadow's preload leave it
                                                       test_a_role_out_for_memory_stays_out
  3  Laya stays by default; with QSE_ROLE_UNLOAD_LAYA=on it goes last, only when idle, its
     questions get `unloaded_for_memory` without a load, and /readyz stays 200
                                                       test_laya_is_last_and_only_when_asked
  4  the reload: only after the headroom stayed at floor + claim + margin for hold_s with nothing
     waiting and no pressure, one role at a time, narrator before shadow, through the planner
                                                       test_the_reload_waits_and_goes_one_at_a_time
  5  no thrash: a role unloaded again within thrash_s of its reload waits twice as long; a
     refused reload waits a whole window; a busy role is skipped
                                                       test_no_thrash
  6  /health's roles report says loaded / unloaded_for_memory / reloading
                                                       test_the_report
  7  the split counts the pool's COMMITTED pages, not a paged tensor's virtual size, keeps the
     reserved range apart, and allocator + KV pages + outside = MemTotal - MemAvailable
                                                       test_the_split_counts_committed_pages

Run: python tests/test_roles_dynamic.py
"""

from __future__ import annotations

import os
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from engine.generative.fake import FakeGenerative  # noqa: E402
from engine.generative.slot import Slot  # noqa: E402
from server import rolemem  # noqa: E402

GIB = 1 << 30


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Mem:
    """MemAvailable in bytes; a model's unload gives its bytes back."""

    def __init__(self, gib):
        self.v = int(gib * GIB)

    def __call__(self):
        return self.v


def _slot(name, role, mem, gib, claim):
    m = FakeGenerative(name)
    real_load, real_unload = m.load, m.unload

    def load(where):
        mem.v -= int(gib * GIB)
        real_load(where)

    def unload():
        mem.v += int(gib * GIB)
        real_unload()
    m.load, m.unload = load, unload
    s = Slot(m, role=role, gate=threading.Lock(), log=lambda x: None,
             planner=lambda n, **kw: {"allowed": True, "reason": None, "role": role,
                                      "claim_gib": claim})
    s.load(wait=True)
    assert s.state == "loaded"
    return s


def _gov(mem, clock, roles, idle=None, **kw):
    log = []
    rm = rolemem.RoleMemory(roles, floor_gib=15, margin_gib=4, hold_s=60, thrash_s=600,
                            max_hold_s=3600, meminfo=mem, idle=idle or (lambda: True),
                            journal=log.append, log=lambda m: None, clock=clock,
                            wall=lambda: 1.7e9, **kw)
    return rm, log


def _two(mem):
    sh = _slot("qwen3.5-0.8b", "shadow", mem, 1.9, 2.5)
    na = _slot("qwen3.5-2b", "narrator", mem, 4.1, 5.5)
    return sh, na, [rolemem.SlotRole("shadow", sh), rolemem.SlotRole("narrator", na)]


def test_the_pool_reclaims_roles_before_conversations():
    from engine import kvpool
    from server import app
    mem = Mem(40)
    clock = Clock()
    sh, na, roles = _two(mem)
    rm, _ = _gov(mem, clock, roles)
    keep = {k: app.STATE.get(k) for k in ("engine", "mem", "role_mem", "pool", "resident")}
    old_pool = kvpool.POOL
    calls = []
    try:
        pool = kvpool.POOL = kvpool.KVPool(hard_floor=10 * GIB, meminfo=mem)
        ps = kvpool.PagedSet("slot0", pool=pool)
        ps.add(kvpool.PagedBuffer((1, 1, 16384, 8), 2, torch.bfloat16, "cpu", pool=pool,
                                  name="kv.k"))
        app.STATE.update(engine=types.SimpleNamespace(kv=types.SimpleNamespace(
            pages=ps, max_len=16384, length=0)), mem=None, role_mem=rm, pool=None,
            resident=None)
        app._kv_after_load(types.SimpleNamespace())
        names = [n for n, _ in pool.reclaimers]
        assert names == ["caches", "roles", "idle_side", "idle_slot0", "kept_park"], names
        # spy on the levels after the roles: they must not run when the roles were enough
        pool.reclaimers = [(n, (lambda f, n: (lambda w, o: (calls.append(n), f(w, o))[1]))(f, n))
                           for n, f in pool.reclaimers]
        # MemAvailable 34 GiB after the loads; a 25 GiB commit would leave 9 < 10: 1 GiB short
        assert abs(mem.v - 34 * GIB) < 1024
        pool.gate("slot1", 25 * GIB)
        assert sh.state == "not_loaded" and sh.memory_hold and na.state == "loaded", \
            (sh.state, na.state)
        assert calls == ["caches", "roles"], calls
        # 5 GiB short: the shadow is out already, the narrator goes next
        calls.clear()
        mem.v = int(14 * GIB)
        pool.gate("slot1", 8 * GIB)
        assert na.state == "not_loaded" and na.memory_hold, na.state
        assert calls == ["caches", "roles"], calls
        # nothing left in the roles: the conversations come next
        calls.clear()
        mem.v = int(14 * GIB)
        try:
            pool.gate("slot1", 8 * GIB)
        except kvpool.PoolFull:
            pass
        assert calls == ["caches", "roles", "idle_side", "idle_slot0", "kept_park"], calls
        st = rm.report()["roles"]
        assert st["shadow"]["state"] == st["narrator"]["state"] == "unloaded_for_memory"
        # the admission's reclaim before a wait goes through the roles too (not the caches)
        assert rm.last_pressure is not None
    finally:
        kvpool.POOL = old_pool
        app.STATE.update(keep)


def test_a_role_out_for_memory_stays_out():
    from server import app
    mem = Mem(40)
    clock = Clock()
    sh, na, roles = _two(mem)
    rm, log = _gov(mem, clock, roles)
    got = rm.reclaim(6 * GIB, "slot2")
    assert got >= 6 * GIB - 1, got / GIB          # the shadow (1.9) and the narrator (4.1)
    assert [e["kind"] for e in log] == ["role_unloaded", "role_unloaded"], log
    assert [e["role"] for e in log] == ["shadow", "narrator"]
    assert log[0]["freed_gib"] == 1.9 and log[1]["freed_gib"] == 4.1, log
    for s in (sh, na):
        assert s.ensure() == "unloaded_for_memory" and s.state == "not_loaded"
        d = s.load()
        assert not d["allowed"] and d["reason"] == "unloaded_for_memory", d
        assert s.state == "not_loaded"
        r = s.run([{"role": "user", "content": "x"}], max_new=4, budget_ms=50)
        assert r == {"why": "unloaded_for_memory"}, r          # a narrator line: none, at once
        assert s.status()["memory_hold"] is True
    keep = {k: app.STATE.get(k) for k in ("narrator", "shadow", "narrator_keep", "draining")}
    try:
        app.STATE.update(narrator=types.SimpleNamespace(
            slot=na, settings=lambda: {"mode": "on"}), narrator_keep=True, draining=False,
            shadow=types.SimpleNamespace(slot=sh))
        assert app.narrator_once() is None and na.state == "not_loaded"
        assert app.shadow_preload_once() is None and sh.state == "not_loaded"
    finally:
        app.STATE.update(keep)


def test_laya_is_last_and_only_when_asked():
    from engine.decision.base import NotReadyError
    from engine.decision.fake import FakeModel
    from engine.decision.registry import Registry
    from server import decider_api
    mem = Mem(40)
    clock = Clock()
    reg = Registry([FakeModel("laya-english")], gate=threading.Lock(),
                   log=lambda m: None)
    reg.load("laya-english", wait=True)
    assert reg.loaded() == ["laya-english"]
    # default: no Laya role at all
    st = {"narrator": None, "shadow": None, "deciders": reg}
    assert rolemem.build(st, floor_gib=15, cfg=dict(rolemem.settings(), laya=False)) is None
    rm = rolemem.build(st, floor_gib=15, meminfo=mem, log=lambda m: None,
                       cfg=dict(rolemem.settings(), laya=True))
    assert [r.name for r in rm.roles] == ["laya"]
    # a queued question keeps it
    reg._batchers["laya-english"].depth = 1
    assert rm.reclaim(GIB, "slot1") == 0 and reg.loaded() == ["laya-english"]
    assert not reg.memory_hold
    reg._batchers["laya-english"].depth = 0
    rm.reclaim(GIB, "slot1")
    assert reg.state["laya-english"] == "not_loaded" and "laya-english" in reg.memory_hold
    # a question: unloaded_for_memory at once, no load started
    try:
        reg.ask({}, {"q": {"type": "choice", "options": {"a": "A", "b": "B"}}})
        raise AssertionError("asked an unloaded model")
    except NotReadyError as exc:
        assert exc.why == "unloaded_for_memory", exc.why
    assert reg.state["laya-english"] == "not_loaded"
    assert not reg.load("laya-english")["allowed"]
    code, body = decider_api.readyz(reg, {})
    assert code == 200 and body["status"] == "unloaded_for_memory", (code, body)
    # the governor brings it back
    rm.hold_s = 0.0
    rm.st["laya"]["hold_s"] = 0.0
    rm.last_pressure = None
    assert rm.tick() == "laya"
    import time as _t
    for _ in range(100):
        if reg.state["laya-english"] == "loaded":
            break
        _t.sleep(0.01)
    rm.tick()
    assert rm.state_of("laya") == "loaded" and not reg.memory_hold, rm.report()["roles"]


def test_the_reload_waits_and_goes_one_at_a_time():
    mem = Mem(40)
    clock = Clock()
    waiting = {"n": 0}
    sh, na, roles = _two(mem)
    rm, log = _gov(mem, clock, roles, idle=lambda: waiting["n"] == 0)
    rm.reclaim(6 * GIB, "slot1")
    assert abs(mem.v - 40 * GIB) < 1024
    line_na = rm.reload_line("narrator") / GIB
    assert abs(line_na - (15 + 5.5 + 4)) < 1e-6, line_na
    # right after the pressure: nothing, however much is free
    clock.t += 30
    assert rm.tick() is None
    # past the quiet hold: the window starts, the narrator (reload order) is first
    clock.t += 31
    assert rm.tick() is None and rm.above_since == clock.t
    clock.t += 59
    assert rm.tick() is None
    # a request waits: the window starts again
    waiting["n"] = 1
    clock.t += 5
    assert rm.tick() is None and rm.above_since is None
    waiting["n"] = 0
    clock.t += 1
    rm.tick()
    # the headroom dips under the line: again
    mem.v = int(20 * GIB)
    clock.t += 30
    assert rm.tick() is None and rm.above_since is None
    mem.v = int(40 * GIB)
    rm.tick()
    clock.t += 60
    assert rm.tick() == "narrator"
    assert rm.state_of("narrator") in ("reloading", "loaded")
    # one at a time: nothing else starts while it loads, then the shadow needs its own window
    import time as _t
    for _ in range(100):
        if na.state == "loaded":
            break
        _t.sleep(0.01)
    assert rm.tick() is None and rm.state_of("narrator") == "loaded", rm.state_of("narrator")
    assert rm.st["narrator"]["reloads"] == 1
    clock.t += 59
    assert rm.tick() is None
    clock.t += 2
    assert rm.tick() == "shadow"
    kinds = [e["kind"] for e in log]
    assert kinds.count("role_reload") == 2 and "role_reloaded" in kinds, kinds


def test_no_thrash():
    mem = Mem(40)
    clock = Clock()
    sh, na, roles = _two(mem)
    rm, log = _gov(mem, clock, roles)
    rm.release("shadow", "kv_pool")
    clock.t += 200
    rm.tick()
    clock.t += 60
    assert rm.tick() == "shadow"
    rm.tick()
    assert sh.state == "loaded" and rm.state_of("shadow") == "loaded"
    clock.t += 100                                   # unloaded again 100 s after its reload
    rm.release("shadow", "kv_pool")
    assert rm.st["shadow"]["hold_s"] == 120 and rm.stats["thrash"] == 1
    # a refused reload waits a whole window, the hold back on
    sh.planner = lambda n, **kw: {"allowed": False, "reason": "refused_memory"}
    clock.t += 200
    rm.tick()
    clock.t += 120
    assert rm.tick() is None and sh.memory_hold and rm.st["shadow"]["refusal"] == "refused_memory"
    assert log[-1]["kind"] == "role_reload_refused"
    # a busy role is skipped, its hold not set
    rm2, _ = _gov(mem, clock, [rolemem.SlotRole("narrator", na)])
    na._busy.acquire()
    try:
        assert rm2.release("narrator", "kv_pool") == 0 and not na.memory_hold
        assert na.state == "loaded" and rm2.stats["skipped_busy"] == 1
    finally:
        na._busy.release()


def test_the_report():
    mem = Mem(40)
    clock = Clock()
    sh, na, roles = _two(mem)
    rm, _ = _gov(mem, clock, roles)
    rep = rm.report()
    assert rep["order"] == ["shadow", "narrator"] and rep["reload_order"] == ["narrator", "shadow"]
    assert {r["state"] for r in rep["roles"].values()} == {"loaded"}
    rm.release("narrator", "kv_pool", want=GIB)
    rep = rm.report()
    n = rep["roles"]["narrator"]
    assert n["state"] == "unloaded_for_memory" and n["freed_gib"] == 4.1 and n["unloads"] == 1
    assert n["reload_at_gib"] == 24.5 and n["models"] == ["qwen3.5-2b"], n
    assert rep["events"][0]["kind"] == "role_unloaded"


class _Drv:
    granularity = 2 << 20


def test_the_split_counts_committed_pages():
    from engine import kvpool
    from server import memsplit
    pool = kvpool.KVPool(hard_floor=0, meminfo=lambda: None)
    ps = kvpool.PagedSet("slot0", pool=pool)
    k = kvpool.PagedBuffer((2, 4, 131072, 64), 2, torch.bfloat16, "cpu", pool=pool, name="kv.k")
    d = kvpool.PagedBuffer((2, 4, 131072, 64), 2, torch.bfloat16, "cpu", pool=pool,
                           name="draft.k")
    ps.add(k)
    ps.add(d)
    ps.ensure(1000)
    # as if driver-mapped: the tensors are ranges, the committed pages are what they hold
    for b in (k, d):
        b.drv = _Drv()
        b.va_size = b.nbytes
    assert kvpool.paged(k.tensor) and kvpool.paged(d.tensor)
    kinds = pool.by_kind()
    assert kinds["engine_kv"] == k.committed_bytes() > 0 and kinds["drafter_kv"] == d.committed_bytes()
    from test_residency import _the_27b_verbs
    eng = _the_27b_verbs(types.SimpleNamespace(w=types.SimpleNamespace(bytes_fp8=10 * GIB, bytes_other=0),
                                               kv=types.SimpleNamespace(k=k.tensor, v=None),
                                               state=None))
    drafter = types.SimpleNamespace(_ck=d.tensor, _cv=None)
    state = {"engine": eng, "drafter": drafter}
    out = memsplit.split(state, probe=lambda: (12 * GIB, 13 * GIB), avail=lambda: 100 * GIB,
                         total=lambda: 120 * GIB, pool=pool)
    p = out["parts"]
    assert p["kv_pool"] == k.committed_bytes() and p["kv_pool"] < k.nbytes, p
    assert p["drafter_kv"] == d.committed_bytes() < d.nbytes, p
    assert out["vmm_parts"] == ["kv_pool", "drafter_kv"]
    assert out["kv_reserved_va"] == k.nbytes + d.nbytes
    assert out["other"] == 2 * GIB                       # allocated less the weights only
    vmm = k.committed_bytes() + d.committed_bytes()
    assert out["kv_committed"] == vmm and out["in_use"] == 20 * GIB
    assert out["outside"] == 20 * GIB - 13 * GIB - vmm
    # the bar: allocator parts + other + reserve + KV pages + outside = in use
    bar = (sum(v for n, v in p.items() if n not in out["vmm_parts"]) + out["other"]
           + out["reserve"] + vmm + out["outside"])
    assert bar == out["in_use"], (bar, out["in_use"])
    # without the pool: a static tensor is counted by its size, as before
    static = memsplit.split({"engine": _the_27b_verbs(types.SimpleNamespace(
        w=None, kv=types.SimpleNamespace(k=torch.zeros(4, 4)), state=None))},
        probe=lambda: (GIB, GIB), avail=lambda: None, total=lambda: None, pool=None)
    assert static["parts"]["kv_pool"] == 64 and static["vmm_parts"] == []
    assert static["kv_committed"] is None


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
