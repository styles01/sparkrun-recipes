"""The engine's memory limits (OPS-47, QA OPS-48) on a fake allocator: the policy, not the model.

`engine/cache.py::Budget` decides three things -- may a cache grow, must caches go back before a
prefill, may a prefill run at all -- from the allocator's two numbers (allocated, reserved). These
tests give it an allocator whose numbers the test sets, and check each decision against the limits
The owner set on 2026-10-01: a 70 GiB ceiling between requests, a 77 GiB declared peak, a 2 GiB
reserve slack, and the shed order store, body anchors, stash, park, pin.

The scenarios are OPS-47's Gherkin (rows of OPS-48): 2 evict before clone, 3 the ordered shed,
4 the 503 instead of a wedge, 5 the reserve back at idle, 7 compaction drops a session's entries,
8 the planner counts the engine's peak. The model-level rows (1, 6, 9) are tests/test_resident_park.py.

Run: python tests/test_budget.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import cache  # noqa: E402
from engine.cache import GIB, Budget, MemoryRefused, StateStore  # noqa: E402


class FakeAlloc:
    """The CUDA caching allocator's two counters, and the peak since a reset."""

    def __init__(self, allocated_gib: float, reserve_gib: float = 0.0):
        self.alloc = int(allocated_gib * GIB)
        self.res = self.alloc + int(reserve_gib * GIB)
        self.peak = self.alloc
        self.empties = 0

    def probe(self):
        return self.alloc, self.res

    def empty(self):
        self.empties += 1
        self.res = self.alloc

    def peak_probe(self, reset=False):
        if reset:
            self.peak = self.alloc
            return 0
        return self.peak

    def take(self, n):
        self.alloc += n
        self.res = max(self.res, self.alloc)
        self.peak = max(self.peak, self.alloc)

    def give(self, n):
        self.alloc -= n                       # freed blocks stay in the reserve


def budget(fa: FakeAlloc, ceiling=70, peak=77, slack=2, **kw) -> Budget:
    return Budget(int(ceiling * GIB), int(peak * GIB), int(slack * GIB), probe=fa.probe,
                  empty=fa.empty, peak_probe=fa.peak_probe, **kw)


def shedder(fa: FakeAlloc, order: list, level: str, gib: float, safe_ok: bool = True):
    """A cache holding `gib` that gives it all back when asked (and records the order)."""
    held = {"b": int(gib * GIB)}

    def fn(safe):
        if safe and not safe_ok:
            return 0
        b, held["b"] = held["b"], 0
        if b:
            order.append(level)
            fa.give(b)
        return b
    return fn


# ------------------------------------------------------------------ 3 and 4: the gate

def test_under_both_limits_nothing_is_shed():
    fa = FakeAlloc(60, 1)
    m = budget(fa)
    order = []
    for lv in Budget.LEVELS:
        m.register(lv, shedder(fa, order, lv, 1))
    g = m.gate(1000)
    assert order == [] and g["shed"] == [] and m.ws_now == cache.WORKSPACE_FLOOR


def test_over_the_ceiling_the_reserve_goes_first_then_the_caches_cheapest_first():
    fa = FakeAlloc(69.5, 3)          # reserved 72.5 over the 70 ceiling, 3 of it reserve
    m = budget(fa)
    order = []
    for lv, gib in (("store", 1), ("anchors", 1), ("stash", 1), ("park", 2), ("pin", 4)):
        m.register(lv, shedder(fa, order, lv, gib))
    m.gate(1000)
    # the reserve went back (72.5 -> 69.5): no cache had to go
    assert order == [] and fa.empties == 1 and fa.res <= 70 * GIB
    fa.take(int(1.5 * GIB))           # 71.0 allocated, no reserve: now caches go
    m.gate(1000)
    assert order == ["store"], order
    assert fa.res <= 70 * GIB


def test_a_capped_workspace_stops_growing_and_a_larger_prefill_raises_the_cap():
    """SPD-99: Flash-Next's prefill workspace grows to about 140k tokens and then stops (10.46 GiB at
    147,149, 10.47 at 261,549): the adapter gives a cap, the estimate stops there, a long prefill that
    took more raises it; without a cap (the 27B) the line goes on as before."""
    fa = FakeAlloc(50)
    m = budget(fa, peak=90, ws_floor=int(2.75 * GIB), ws_per_token=76309, ws_cap=int(10.75 * GIB))
    assert m.workspace(8192) == int(2.75 * GIB)
    assert abs(m.workspace(65_536) - 76309 * 65_536) == 0
    assert m.workspace(262_144) == int(10.75 * GIB)
    m.mark()
    fa.take(int(11.5 * GIB))                           # a prefill that took 11.5 GiB above the engine
    fa.give(int(11.5 * GIB))
    m.prefilled(200_000)
    assert m.ws_cap == int(11.5 * GIB) and any(e["kind"] == "estimate_raised" and "cap" in e for e in m.events)
    plain = budget(FakeAlloc(50))
    assert plain.ws_cap is None and plain.workspace(262_144) == cache.WORKSPACE_PER_TOKEN * 262_144
    return "the floor, the line, the cap; a larger prefill raises it; no cap for the 27B"


def test_nemotron_h_workspace_is_flat_so_a_million_token_prefill_runs():
    """ENG-363: NemotronH's prefill workspace does not grow with the prompt (3.34 GiB above the engine at
    131,054 tokens, 3.18 at 249,983 on Super VL): its adapter says so, and a 1,000,000-token prefill with
    the engine at the live numbers (80.6 GiB allocated, 82.4 reserved, peak 94.8 with slot 0's window
    and three side slots) is admitted. The 27B's line refused it on 2026-10-16 ("needs 107.3 GiB")."""
    from types import SimpleNamespace

    from engine.models.nemotron_h.serving import NemotronHServing
    srv = NemotronHServing.__new__(NemotronHServing)
    srv.st = SimpleNamespace(conv={}, ssm={})
    srv.cfg = SimpleNamespace(kv_bytes_per_token=lambda dtype_bytes: 8192)
    c = srv.cost()
    assert c.view_bytes_per_token == 8192
    fa = FakeAlloc(80.6, reserve_gib=1.8)
    m = budget(fa, ceiling=90.8, peak=94.8, ws_floor=c.workspace_floor, ws_per_token=c.workspace_per_token,
               ws_cap=c.workspace_cap)
    assert m.workspace(8192) == int(3.5 * GIB) and m.workspace(1_000_000) == int(4.0 * GIB)
    g = m.gate(1_000_000)
    assert g["workspace"] == int(4.0 * GIB)
    m.done()
    old = budget(FakeAlloc(80.6, reserve_gib=1.8), ceiling=90.8, peak=94.8)
    try:
        old.gate(1_000_000)
        raise AssertionError("the 27B's line admitted a 1,000,000-token prefill")
    except MemoryRefused:
        pass
    # what a long prefill teaches stays under the cap (3.34 GiB at 131,054 tokens)
    m.mark()
    fa.take(int(3.34 * GIB))
    fa.give(int(3.34 * GIB))
    m.prefilled(131_054)
    assert m.workspace(1_000_000) == int(4.0 * GIB)
    return "4.0 GiB at 1,000,000 tokens: admitted; the 27B's 28 GiB: refused"


def test_slot_0s_window_above_the_limits_window_raises_ceiling_and_peak():
    """ENG-363: MEM_CEILING_GIB and MEM_PEAK_GIB are set at a 262,144-token window; slot 0 at 1,048,576
    holds 786,432 more rows of KV from the load on (pool off), so both limits go up by them (6 GiB on
    Super VL), kept in `fixed` like the side slots'. At 262,144, with limits off or with no engine
    cost, nothing moves."""
    from types import SimpleNamespace

    from server import app as A
    from server import memlimits
    eng = SimpleNamespace(cost=lambda: SimpleNamespace(view_bytes_per_token=8192))
    old = A.STATE.get("mem")
    try:
        m = A.STATE["mem"] = budget(FakeAlloc(80), ceiling=76, peak=80)
        A._window_above_limits(SimpleNamespace(max_len=1_048_576), eng)
        extra = 8192 * (1_048_576 - memlimits.LIMITS_WINDOW)
        assert extra == 6 * GIB
        assert (m.ceiling, m.peak, m.fixed) == (int(76 * GIB) + extra, int(80 * GIB) + extra, extra)
        m = A.STATE["mem"] = budget(FakeAlloc(80), ceiling=76, peak=80)
        A._window_above_limits(SimpleNamespace(max_len=262_144), eng)
        assert (m.ceiling, m.fixed) == (int(76 * GIB), 0)
        m = A.STATE["mem"] = budget(FakeAlloc(80), ceiling=0, peak=0)      # limits off
        A._window_above_limits(SimpleNamespace(max_len=1_048_576), eng)
        assert m.fixed == 0
    finally:
        A.STATE["mem"] = old
    return "+6 GiB at 1,048,576; nothing at 262,144 or with the limits off"


def test_a_long_prefill_sheds_store_anchors_stash_park_pin_in_that_order_until_it_fits():
    fa = FakeAlloc(70)
    m = budget(fa)
    order = []
    for lv, gib in (("store", 2), ("anchors", 1.5), ("stash", 0.9), ("park", 3), ("pin", 4)):
        m.register(lv, shedder(fa, order, lv, gib))
    n = 262_144                       # the workspace at the full window: ~7.3 GiB
    ws = m.workspace(n)
    assert 7.0 * GIB < ws < 7.6 * GIB, ws / GIB
    g = m.gate(n)
    # 70 + 7.33 = 77.33 > 77: the store alone (2 GiB) brings it to 75.33
    assert order == ["store"] and g["shed"] == ["store"], order
    fa.take(int(4.5 * GIB))           # caches grew back and more: 72.5 + 7.33 needs three levels
    order.clear()
    g = m.gate(n)
    assert order == ["anchors", "stash", "park"], order
    assert fa.res + ws <= 77 * GIB
    assert m.stats["shed"]["park"] == 1 and [e["level"] for e in m.events if e["kind"] == "shed"]


def test_with_nothing_left_to_shed_the_prefill_is_refused_not_run():
    fa = FakeAlloc(75)                # e.g. a decision model in the same allocator, not excluded
    m = budget(fa)
    order = []
    m.register("store", shedder(fa, order, "store", 0.5))
    try:
        m.gate(262_144)
        raise AssertionError("a prefill over the peak must be refused")
    except MemoryRefused as exc:
        assert exc.peak == 77 * GIB and exc.need > exc.peak
    assert order == ["store"] and m.stats["refused"] == 1
    assert any(e["kind"] == "refused" for e in m.events)
    assert m.ws_now == 0, "a refused prefill reserves no workspace"


class StuckAlloc(FakeAlloc):
    """2026-10-04 16:41: a reserve `empty` cannot give back (a role's own pool keeps its free
    segments while it lives); `release()` drops the pool, and the next empty returns it."""

    def __init__(self, allocated_gib, stuck_gib, reserve_gib=0.0):
        super().__init__(allocated_gib, reserve_gib)
        self.stuck = int(stuck_gib * GIB)
        self.res += self.stuck

    def empty(self):
        self.empties += 1
        self.res = self.alloc + self.stuck

    def release(self, safe=False):
        b, self.stuck = self.stuck, 0
        return b


def test_a_roles_pool_full_of_free_segments_is_given_back_before_a_refusal():
    # the incident's numbers: 63 GiB in use, 17.6 GiB free inside the shadow's pool, peak 77
    fa = StuckAlloc(63.2, 17.6)
    m = budget(fa)
    order = []
    for lv in ("store", "anchors", "end", "stash", "park", "pin"):
        m.register(lv, shedder(fa, order, lv, 0))
    m.register("roles", fa.release)
    g = m.gate(587)
    assert g["shed"] == ["roles"] and fa.stuck == 0 and fa.res <= 77 * GIB, g
    assert m.stats["refused"] == 0 and m.refused_in_a_row == 0


def test_a_tiny_prefill_is_never_refused_on_a_stale_reserve_the_engine_can_use():
    fa = StuckAlloc(63.2, 17.6)       # stuck in the engine's own allocator (fragmentation)
    m = budget(fa)
    g = m.gate(587)
    assert g.get("stale") and m.stats["stale_admits"] == 1, g
    assert any(e["kind"] == "stale_admit" for e in m.events)
    # a prefill whose workspace does not fit in the free reserve is still refused
    try:
        m.gate(262_144 * 8)
        raise AssertionError("refused expected")
    except MemoryRefused:
        pass
    assert m.refused_in_a_row == 1
    m.gate(12)
    assert m.refused_in_a_row == 0


def test_the_free_reserve_inside_a_roles_pool_does_not_count_as_the_engines():
    fa = StuckAlloc(63.2, 17.6)
    m = budget(fa)
    m.foreign_free = lambda: fa.stuck   # all of it is the role's, nothing releases it
    for _ in range(3):
        try:
            m.gate(587)
            raise AssertionError("refused expected")
        except MemoryRefused:
            pass
    assert m.refused_in_a_row == 3 and m.events[-1]["in_a_row"] == 3
    assert m.events[-1]["allocated"] < m.events[-1]["reserved"]


def test_the_decision_models_are_not_the_engine():
    """Laya's checkpoints live in the same allocator but are roles with claims of their own."""
    fa = FakeAlloc(78)
    m = budget(fa, exclude=lambda: 9 * GIB)
    assert m.memory() == (69 * GIB, 69 * GIB)
    m.gate(1000)                      # 69 + 1.15 is under the peak and 69 under the ceiling


def test_limits_off_is_the_old_engine():
    fa = FakeAlloc(100)
    m = budget(fa, ceiling=0, peak=0)
    assert m.gate(262_144) == {} and m.admit("park", 10 * GIB) and m.done() == 0


# ------------------------------------------------------------------ admission

def test_a_cache_grows_only_under_the_ceiling_less_the_slack():
    fa = FakeAlloc(66)
    m = budget(fa)
    assert m.admit("park", int(1.5 * GIB))          # 67.5 <= 68
    assert not m.admit("park", int(2.5 * GIB))      # 68.5 > 68, nothing below to give back
    assert m.stats["admit_refused"] == 1


def test_a_more_valuable_cache_takes_the_room_of_the_cheaper_ones():
    fa = FakeAlloc(67)
    m = budget(fa)
    order = []
    m.register("store", shedder(fa, order, "store", 2))
    m.register("anchors", shedder(fa, order, "anchors", 0.3))
    m.register("pin", shedder(fa, order, "pin", 4))
    assert m.admit("park", int(2.5 * GIB))          # 69.5 > 68: the store and the free anchors go
    assert order == ["store", "anchors"], order
    order.clear()
    fa.alloc = int(67 * GIB)
    assert not m.admit("store", int(2 * GIB)), "the cheapest level takes nobody's room"
    assert order == []


def test_while_a_prefill_waits_its_workspace_counts():
    fa = FakeAlloc(60)
    m = budget(fa)
    m.gate(262_144)                                  # 60 + 7.33 <= 77: it runs
    assert m.ws_now > 7 * GIB
    assert not m.admit("pin", 10 * GIB)              # 70 + 7.33 > 77 before its forwards
    m.prefilled(262_144)
    assert m.ws_now == 0
    assert m.admit("pin", 4 * GIB)


# ------------------------------------------------------------------ 5: the reserve at idle

def test_the_reserve_over_its_slack_goes_back_after_the_request():
    fa = FakeAlloc(65, 7)
    m = budget(fa)
    freed = m.done()
    assert freed == 7 * GIB and fa.empties == 1 and m.stats["empties"] == 1
    fa.res += int(1.5 * GIB)                         # under the slack: kept
    assert m.done() == 0 and fa.empties == 1


def test_the_workspace_estimate_learns_upward_only():
    fa = FakeAlloc(60)
    m = budget(fa)
    was = m.ws_per_token
    m.mark()
    fa.take(int(6 * GIB))                            # 6 GiB at 100k is more than the line says
    fa.give(int(6 * GIB))
    m.prefilled(100_000)
    assert m.ws_per_token == int(6 * GIB / 100_000) > was
    assert m.stats["observed"]["workspace"] == 6 * GIB
    m.mark()
    fa.take(GIB)
    fa.give(GIB)
    m.prefilled(100_000)
    assert m.ws_per_token == int(6 * GIB / 100_000), "a cheaper prefill does not lower it"
    assert m.stats["prefill_peak"] == 66 * GIB and m.lifetime() >= 66 * GIB


def test_what_a_prefill_keeps_is_not_its_workspace():
    """The anchors and the store snapshot a prefill takes are still allocated at its end; only
    what it gave back is workspace (the 2026-10-01 gate learned 113 kB a token without this)."""
    fa = FakeAlloc(60)
    m = budget(fa)
    was = m.ws_per_token
    m.mark()
    fa.take(3 * GIB)                  # anchors and a snapshot, kept
    fa.take(2 * GIB)                  # the chunk's scores and activations
    fa.give(2 * GIB)
    m.prefilled(100_000)
    assert m.stats["observed"]["workspace"] == 2 * GIB and m.ws_per_token == was
    m.mark()
    fa.take(9 * GIB)
    fa.give(9 * GIB)
    m.prefilled(40_000)               # short: the fixed part dominates, nothing is learned
    assert m.ws_per_token == was and m.stats["observed"]["n"] == 100_000


# ------------------------------------------------------------------ 2: the store, room before clone

class Snap:
    def __init__(self, n, nbytes):
        self.length, self.nbytes = n, nbytes

    def parts(self):
        return {"kv": self.nbytes}


def test_the_store_makes_room_before_the_clone():
    st = StateStore(10, chunk=4, max_entry_bytes=6)
    toks = list(range(40))
    st.put(toks, Snap(4, 4), "a")
    st.put(toks, Snap(8, 4), "a")
    assert st.bytes == 8
    assert st.reserve(5)                             # 8 + 5 > 10: the oldest goes first
    assert st.bytes == 4 and st.stats["evictions"] == 1
    st.put(toks, Snap(12, 5), "b")
    assert st.bytes == 9 <= st.budget
    assert not st.reserve(7), "over the entry cap: refused, nothing evicted"
    assert st.bytes == 9


def test_a_compaction_drops_its_sessions_entries_only():
    st = StateStore(100, chunk=4)
    toks = list(range(40))
    st.put(toks, Snap(4, 3), "A")
    st.put(toks, Snap(8, 3), "B")
    st.put(toks, Snap(12, 3), "A")
    assert st.drop_session("A") == 6
    assert st.bytes == 3 and st.report()["entries"] == 1 and st.stats["dropped_session"] == 2
    assert st.drop_session(None) == 0 and st.drop_session("nobody") == 0


def test_the_store_sheds_and_takes_new_limits():
    st = StateStore(100, chunk=4)
    toks = list(range(40))
    for n in (4, 8, 12):
        st.put(toks, Snap(n, 10), "A")
    st.set_limits(budget_bytes=15, max_entry_bytes=9)
    assert st.bytes == 10 and st.max_entry == 9
    assert st.shed() == 10 and st.bytes == 0


# ------------------------------------------------------------------ 8: the planner counts the peak

def _residency_doc():
    from server.conductor import schema
    return schema.seed("residency")


def test_the_planner_refuses_a_12_gib_role_at_37_9_free_with_the_engine_at_70_of_77():
    from server.conductor import residency
    doc = _residency_doc()
    d = residency.check("reader", 37.9, doc, engine_peak_gib=77, engine_now_gib=70)
    assert not d["allowed"] and d["reason"] == "refused_memory"
    assert d["engine_growth_gib"] == 7 and d["after_gib"] == 18.9
    # the old rule (no peak) admitted it -- the bug
    old = residency.check("reader", 37.9, doc)
    assert old["allowed"] and old["after_gib"] == 25.9
    # the decider (3.0) still fits: 37.9 - 7 - 3 = 27.9 >= 21
    assert residency.check("decider", 37.9, doc, engine_peak_gib=77, engine_now_gib=70)["allowed"]
    # with no reading of the engine the whole peak counts
    assert residency.check("decider", 60, doc, engine_peak_gib=77)["engine_growth_gib"] == 77


def test_the_plan_takes_the_growth_once():
    from server.conductor import residency
    p = residency.plan(37.9, _residency_doc(), engine_peak_gib=77, engine_now_gib=70)
    assert p["engine_growth_gib"] == 7 and p["spendable_gib"] == 9.9
    rows = {r["role"]: r for r in p["rows"]}
    assert rows["decider"]["allowed"] and rows["decider"]["after_gib"] == 27.9
    assert not rows["reader"]["allowed"]


def test_the_command_line_takes_the_engine_numbers():
    with tempfile.TemporaryDirectory() as root:
        env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env.pop("MEM_PEAK_GIB", None)
        r = subprocess.run([sys.executable, "-m", "server.conductor.residency", "--root", root,
                            "check", "reader", "--mem-available-gib", "37.9",
                            "--engine-peak-gib", "77", "--engine-now-gib", "70"],
                           capture_output=True, text=True, env=env, cwd=env["PYTHONPATH"])
        assert r.returncode == 3, r.stdout + r.stderr
        d = json.loads(r.stdout)
        assert d["engine_growth_gib"] == 7 and d["reason"] == "refused_memory"
        env["MEM_PEAK_GIB"] = "77"                   # serve.env's, when /health does not answer
        r = subprocess.run([sys.executable, "-m", "server.conductor.residency", "--root", root,
                            "check", "decider", "--mem-available-gib", "100"],
                           capture_output=True, text=True, env=env, cwd=env["PYTHONPATH"])
        assert json.loads(r.stdout)["engine_growth_gib"] == 77


# ------------------------------------------------------------------ the limits table

class _Store:
    def __init__(self, v, doc):
        self.v, self.doc = v, doc

    def peek(self, kind):
        assert kind == "residency"
        return self.v, self.doc


class _Cond:
    def __init__(self, v, doc):
        self.store = _Store(v, doc)


def _args(**over):
    a = {"mem_ceiling_gib": 70.0, "mem_peak_gib": 77.0, "reserve_slack_gib": 2.0,
         "resident_gb": 2.5, "resident_stash_gb": 1.0, "resident_guest_max": 8192,
         "resident_pin_gb": 4.0, "resident_park_gb": 4.0, "cache_budget_gb": 2.0,
         "store_entry_gb": 1.0, "image_cache_mb": 512.0, "suffix_store_mb": 192.0,
         "prefix_chunk": 1024, "narrow_arm": "on", "max_len": 262144}
    a.update(over)
    return a


def test_the_view_has_every_key_and_max_len_is_locked():
    from server import memlimits
    st = {"args": _args(), "serve_flags": _args(), "max_len": 262144}
    v = memlimits.view(st)
    keys = [r["key"] for r in v["rows"]]
    assert keys == list(memlimits.KEYS)
    row = {r["key"]: r for r in v["rows"]}
    assert row["MAX_LEN"]["applies"] == "locked" and row["MAX_LEN"]["value"] == 262144
    assert row["PREFIX_CHUNK"]["applies"] == "restart" and not row["PREFIX_CHUNK"]["pending_restart"]
    assert row["RESIDENT_PARK_GB"]["applies"] == "next request"
    assert row["NARROW_ARM"]["value"] == 1 and row["CACHE_GB"]["source"] == "serve.env"


def test_a_models_own_grid_is_shown_locked_with_no_restart_pending():
    """rc-0.8.0 live: on Flash-Next the memory plan said PREFIX_CHUNK 1,024 "needs a restart; the
    engine runs 4,096 until it restarts", which no restart changes: the model's caches keep its own
    grid (`caps.anchor_grid`). The row shows that grid, locked, with nothing pending."""
    from server import memlimits
    st = {"args": _args(prefix_chunk=4096), "serve_flags": _args(), "max_len": 262144,
          "anchor_grid": 4096}
    row = {r["key"]: r for r in memlimits.view(st)["rows"]}["PREFIX_CHUNK"]
    assert row["value"] == row["running"] == 4096 and row["applies"] == "locked", row
    assert not row["pending_restart"] and "own grid: 4,096 tokens" in row["locked"], row
    # a dashboard override changes nothing for such a model either
    st["conductor"] = _Cond(5, {"engine": {"PREFIX_CHUNK": 512}})
    row = {r["key"]: r for r in memlimits.view(st)["rows"]}["PREFIX_CHUNK"]
    assert row["value"] == 4096 and not row["pending_restart"]


def test_a_dashboard_override_applies_live_or_says_it_needs_a_restart():
    from server import memlimits
    fa = FakeAlloc(60)
    mem = budget(fa)
    store = StateStore(2 * GIB, chunk=1024)
    doc = {"engine": {"RESIDENT_PARK_GB": 2, "CACHE_GB": 1, "MEM_CEILING_GIB": 68,
                      "PREFIX_CHUNK": 512, "MAX_LEN": 131072}}
    st = {"args": _args(), "serve_flags": _args(), "mem": mem, "state_store": store,
          "conductor": _Cond(3, doc), "max_len": 262144}
    done = memlimits.refresh(st)
    assert "MEM_CEILING_GIB=68" in done and "CACHE_GB=1" in done
    assert mem.ceiling == 68 * GIB and store.budget == GIB
    assert memlimits.refresh(st) == [], "the same version applies once"
    row = {r["key"]: r for r in memlimits.view(st)["rows"]}
    assert row["PREFIX_CHUNK"]["pending_restart"] and row["PREFIX_CHUNK"]["source"] == "dashboard"
    assert row["MAX_LEN"]["value"] == 262144, "the hard rule is not overridable"
    assert row["MEM_CEILING_GIB"]["source"] == "dashboard" and row["MEM_CEILING_GIB"]["value"] == 68
    # the key leaves the document: back to the flag
    st["conductor"] = _Cond(4, {"engine": {}})
    memlimits.refresh(st)
    assert mem.ceiling == 70 * GIB and store.budget == 2 * GIB


def test_overrides_at_start_never_stop_the_engine():
    import argparse
    from server import memlimits
    a = argparse.Namespace(**_args())
    lines = memlimits.overrides_at_start(
        a, {"engine": {"NARROW_ARM": 0, "PREFIX_CHUNK": 512, "MAX_LEN": 1024}}, len_mode="b")
    assert a.narrow_arm == "on" and any("skipped" in x for x in lines)
    assert a.prefix_chunk == 512 and a.max_len == 262144
    memlimits.overrides_at_start(a, {"engine": {"NARROW_ARM": 0}}, len_mode="wide")
    assert a.narrow_arm == "off"


def test_the_schema_takes_the_engine_section_and_checks_peak_over_ceiling():
    from server.conductor import schema
    doc = schema.seed("residency")
    assert schema.check("residency", dict(doc, engine={"RESIDENT_PARK_GB": 4})) == []
    errs = schema.check("residency", dict(doc, engine={"MEM_CEILING_GIB": 70, "MEM_PEAK_GIB": 60}))
    assert any("MEM_PEAK_GIB is under" in e for e in errs), errs
    assert schema.check("residency", dict(doc, engine={"MAX_LEN": 131072})), "MAX_LEN is refused"


# ------------------------------------------------------------------ Prometheus

def test_the_memory_metrics_render_from_the_cache_stats():
    from server import metrics as M
    rep = budget(FakeAlloc(60, 1)).report()
    rep["shed"]["park"] = 2
    rep["refused"] = 1
    stats = {"memory": rep, "resident": {"anchor_pool_bytes": 3, "stash_bytes": 0, "pin_bytes": 5,
                                         "park_bytes": 7}, "state_store": {"bytes": 11}}
    M.bind(cache_stats=lambda: stats)
    text = M.render()
    assert 'qse_memory_bytes{part="ceiling"} 75161927680' in text, text
    assert 'qse_memory_bytes{part="resident_park"} 7' in text
    assert 'qse_memory_shed_total{level="park"} 2' in text
    assert 'qse_requests_refused_total{reason="memory"} 1' in text


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
