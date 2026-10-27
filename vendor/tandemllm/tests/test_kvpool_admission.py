"""SRV-307 + SRV-305, CPU only: the parallel admission with the shared KV pool, and the wait for
memory as the dashboard sees it.

  * a request's KV part (the pages its prompt and max_tokens may still commit) is reserved; what
    its slot has committed since it was admitted is credited, never counted twice;
  * the KV part is never capped by the engine's room to its peak (the pages are outside the
    memory policy);
  * a request alone is admitted, as before; a release frees its bytes and its credit record;
  * waiters for memory keep their arrival order (`wait` gives each its place), `unwait` takes one
    out, and the report counts them (`waiting_memory`);
  * the parallel view shows a held request's lane as `waiting_memory` with the label "waiting for
    memory", counts it in `totals.waiting_memory` and `waiting`, never in `decoding`;
  * the contract schema knows the new state and the new total.

Run: python tests/test_kvpool_admission.py
"""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from server import slots  # noqa: E402

GiB = slots.GiB


def _adm(avail_gib: float, floor_gib: float = 15, committed=None, room=None):
    return slots.Admission(int(floor_gib * GiB), meminfo=lambda: {"MemAvailable": int(avail_gib * GiB)},
                           growth=lambda: 0, room=room, committed=committed)


def test_kv_part_is_reserved_and_credited_as_it_commits():
    done = {"slot1": 0, "slot2": 0}
    adm = _adm(40, committed=lambda o: done[o])
    ok, _ = adm.reserve("a", 2 * GiB + 10 * GiB, kv=10 * GiB, owner="slot1")
    assert ok                                      # alone
    ok, d = adm.reserve("b", 1 * GiB + 14 * GiB, kv=14 * GiB, owner="slot2")
    # 40 - (2 + 10 held) - (1 + 14) = 13 < 15
    assert not ok and d["kv_reserved"] == 10 * GiB and d["after"] == 13 * GiB
    # slot 1 committed 6 GiB: MemAvailable fell by those 6, the reservation by the same 6
    done["slot1"] = 6 * GiB
    adm.meminfo = lambda: {"MemAvailable": 34 * GiB}
    ok, d = adm.reserve("b", 1 * GiB + 14 * GiB, kv=14 * GiB, owner="slot2")
    assert not ok and d["after"] == 13 * GiB, "the same answer: nothing counted twice"
    adm.release("a")
    ok, d = adm.reserve("b", 15 * GiB, kv=14 * GiB, owner="slot2")
    assert ok                                      # alone again
    assert adm.report()["kv_to_commit_gib"] == 14.0


def test_credit_never_goes_negative_or_past_the_kv_part():
    c = {"slot1": 5 * GiB}
    adm = _adm(100, committed=lambda o: c[o])
    adm.reserve("a", 3 * GiB, kv=2 * GiB, owner="slot1")
    c["slot1"] = 1 * GiB                           # trimmed below its baseline
    ws, kv = adm._held()
    assert (ws, kv) == (1 * GiB, 2 * GiB)
    c["slot1"] = 50 * GiB
    ws, kv = adm._held()
    assert (ws, kv) == (1 * GiB, 0)


def test_kv_is_not_capped_by_the_engine_room():
    adm = _adm(60, room=lambda: 1 * GiB)
    adm.reserve("x", GiB, owner=None)              # alone, workspace only
    ok, d = adm.check(2 * GiB + 30 * GiB, kv=30 * GiB)
    # workspace parts capped at the 1 GiB room; the 30 GiB of pages counted in full
    assert d["after"] == 60 * GiB - 1 * GiB - 30 * GiB and ok


def test_plain_reservations_behave_as_before():
    adm = _adm(30)
    assert adm.reserve("a", 4 * GiB)[0]
    ok, d = adm.reserve("b", 12 * GiB)             # 30 - 4 - 12 = 14 < 15
    assert not ok and d["kv"] == 0
    assert adm.reserve("c", 10 * GiB)[0]           # 16 >= 15
    adm.release("a")
    adm.release("c")
    assert adm.reserved() == 0


def test_waiters_keep_their_order_and_are_counted():
    adm = _adm(10)
    assert adm.wait("w1") == 1
    assert adm.wait("w2") == 2
    assert adm.wait("w1") == 1                     # asking again keeps its place
    assert adm.report()["waiting_memory"] == 2
    adm.unwait("w1")
    assert adm.wait("w2") == 1
    adm.release("w2")
    assert adm.report()["waiting_memory"] == 0


def test_describe_names_the_kv_pages():
    adm = _adm(30)
    adm.reserve("a", 4 * GiB, kv=3 * GiB, owner="s1")
    ok, d = adm.check(20 * GiB, kv=18 * GiB)
    text = slots.Admission.describe(d)
    assert "KV pages: this request 18.0, in flight 3.0" in text and "< floor 15.0" in text


def test_parallel_lane_waiting_for_memory():
    from server import parallel

    class Rec:
        waiting_memory = True

    assert parallel.ParallelBoard._state(None, Rec(), None) == "waiting_memory"
    Rec.waiting_memory = False
    assert parallel.ParallelBoard._state(None, Rec(), 1.0) == "done"


def test_contract_knows_the_state_and_the_total():
    p = os.path.join(ROOT, "docs", "contract", "dashboard-v1", "parallel.schema.json")
    s = json.load(open(p))
    txt = json.dumps(s)
    assert '"waiting_memory"' in txt
    assert "waiting_memory" in s["properties"]["totals"]["properties"] if "totals" in s.get(
        "properties", {}) else "waiting_memory" in txt


def _app():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_forward_tree  # noqa: F401  (the CPU environment before the engine imports)
    from server import app
    return app


class _H:
    """The handler's surface `_par_admit` uses."""

    def __init__(self):
        self.sent = None

    def _client_gone(self):
        return False

    def _send_bytes(self, code, raw, extra_headers=()):
        self.sent = (code, json.loads(raw), dict(extra_headers))


def test_a_non_streamed_wait_ends_in_a_readable_503():
    import types
    import torch
    app = _app()
    from server import usage
    app.STATE.clear()
    app.STATE["queue_timeout"] = 1.5
    app.STATE["request_timeout"] = 3600.0
    adm = _adm(16)
    adm.reserve("big", 2 * GiB)
    pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
    rec = usage.RequestRecord("chatcmpl-y", "chat", False)
    said = []
    h = _H()
    t0 = time.monotonic()
    assert app.Handler._par_admit(h, pool, "small", torch.zeros(576), rec=rec,
                                  say=lambda short, place=1: said.append((short, place)),
                                  stream=False) is False
    assert 1.4 < time.monotonic() - t0 < 4.0
    code, body, hdr = h.sent
    assert code == 503 and body["error"]["code"] == "memory_wait_timeout", body
    assert "memory is almost full" in body["error"]["message"] and hdr.get("Retry-After") == "30"
    assert rec.wait_reason == "memory" and rec.waiting_memory is False
    assert said and said[0][1] == 1, said              # the line, with its place in line
    assert adm.stats["timeouts"] == 1 and adm.report()["waiting_memory"] == 0
    assert h._mem_wait is True


def test_a_streamed_wait_outlasts_the_queue_timeout():
    import threading
    import types
    import torch
    app = _app()
    from server import usage
    app.STATE.clear()
    app.STATE["queue_timeout"] = 0.5
    app.STATE["request_timeout"] = 30.0
    box = {"avail": 16 * GiB}
    adm = slots.Admission(15 * GiB, meminfo=lambda: {"MemAvailable": box["avail"]},
                          growth=lambda: 0)
    adm.reserve("big", 2 * GiB)
    pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
    rec = usage.RequestRecord("chatcmpl-z", "chat", True)
    rec.lock_acquired()

    def free_later():
        time.sleep(2.5)
        adm.release("big")
    threading.Thread(target=free_later, daemon=True).start()
    h = _H()
    assert app.Handler._par_admit(h, pool, "small", torch.zeros(576), rec=rec, stream=True) is True
    assert rec.wait_mem_ms >= 2000 and h.sent is None and rec.waiting_memory is False


def test_admission_takes_back_idle_conversations_before_it_waits():
    app = _app()
    from engine import kvpool
    calls = []
    pool = kvpool.KVPool(hard_floor=0, meminfo=lambda: 100 * GiB)
    pool.add_reclaimer("caches", lambda want, o: calls.append(("caches", want)) or GiB)
    pool.add_reclaimer("idle_side", lambda want, o: calls.append(("idle_side", want, o)) or 3 * GiB)
    pool.add_reclaimer("idle_slot0", lambda want, o: calls.append(("idle_slot0", want, o)) or 4 * GiB)
    old = kvpool.POOL
    kvpool.POOL = pool
    try:
        got = app._kv_reclaim_idle({"floor": 15 * GiB, "after": 10 * GiB}, "slot2")
        assert got == 7 * GiB, got
        assert calls == [("idle_side", 5 * GiB, "slot2"), ("idle_slot0", 2 * GiB, "slot2")], calls
        calls.clear()
        assert app._kv_reclaim_idle({"floor": 15 * GiB, "after": 13 * GiB}, "slot1") == 3 * GiB
        assert [c[0] for c in calls] == ["idle_side"], "stops once enough came back"
        assert app._kv_reclaim_idle({"floor": 15 * GiB, "after": None}, "slot1") == 0
    finally:
        kvpool.POOL = old


def test_a_release_for_a_sandbox_gives_back_idle_memory_in_the_pools_order():
    """SRV-483: chatd's sandbox has priority over what the engine holds idle."""
    app = _app()
    from engine import kvpool
    calls = []
    pool = kvpool.KVPool(hard_floor=0, meminfo=lambda: 100 * GiB)
    pool.add_reclaimer("caches", lambda want, o: calls.append(("pool caches", want)) or GiB)
    pool.add_reclaimer("roles", lambda want, o: calls.append(("roles", want, o)) or GiB)
    pool.add_reclaimer("idle_side", lambda want, o: calls.append(("idle_side", want, o)) or 2 * GiB)
    pool.add_reclaimer("idle_slot0", lambda want, o: calls.append(("idle_slot0", want, o)) or GiB)
    old = kvpool.POOL, app._kv_reclaim_caches, app._lock_when_idle
    kvpool.POOL = pool
    app._kv_reclaim_caches = lambda want, o: calls.append(("caches", want, o)) or GiB
    try:
        out = app._memory_release(3 * GiB)
        assert calls == [("caches", 3 * GiB, "release"), ("roles", 2 * GiB, "release"),
                         ("idle_side", GiB, "release")], calls
        assert out["released_bytes"] == 4 * GiB and out["asked_bytes"] == 3 * GiB
        assert out["levels"] == {"caches": GiB, "roles": GiB, "idle_side": 2 * GiB}
        assert pool.stats["reclaims"] == 2 and pool.stats["reclaimed_bytes"] == 3 * GiB
        # a request runs: the caches stay, the idle conversations still give way
        app._lock_when_idle = lambda: False
        calls.clear()
        assert app._memory_release(GiB)["levels"] == {"roles": GiB}
        assert calls == [("roles", GiB, "release")]
        calls.clear()
        assert app._memory_release(0)["released_bytes"] == 0 and calls == []
    finally:
        kvpool.POOL, app._kv_reclaim_caches, app._lock_when_idle = old


def test_the_release_route_answers_trusted_local_and_the_admin_only():
    import contextlib
    import io
    app = _app()
    from server import auth

    class Req(app.Handler):
        def __init__(self, body, headers):
            raw = json.dumps(body).encode()
            self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
            self.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json",
                            **headers}
            self.path, self.command, self.request_version = ("/v1/cache/release", "POST",
                                                             "HTTP/1.1")
            self.requestline, self.client_address = "POST /v1/cache/release", ("127.0.0.1", 0)
            self.close_connection = True

    def call(body, headers=None):
        r = Req(body, headers or {})
        with contextlib.redirect_stdout(io.StringIO()):
            r.do_POST()
        head, _, raw = r.wfile.getvalue().partition(b"\r\n\r\n")
        return int(head.split()[1]), json.loads(raw)

    asked = []
    old = app._memory_release, dict(app.STATE)
    app._memory_release = lambda want, why="sandbox": asked.append((want, why)) or {
        "asked_bytes": want, "released_bytes": 0, "levels": {}, "mem_available_bytes": None}
    try:
        app.STATE["auth"] = auth.Auth("adm-" + "5e7u9" * 10, None)
        code, out = call({"bytes": 2 * GiB, "why": "sandbox"})          # chatd on the box
        assert code == 200 and out["asked_bytes"] == 2 * GiB and asked == [(2 * GiB, "sandbox")]
        code, _ = call({"bytes": GiB}, {"X-Forwarded-For": "203.0.113.4",
                                        "X-Real-IP": "203.0.113.4"})
        assert code == 401 and len(asked) == 1, "through the reverse proxy: the admin token"
        for bad in ({"bytes": -1}, {"bytes": "x"}, {"bytes": 1 << 41}):
            assert call(bad)[0] == 400
        assert len(asked) == 1
    finally:
        app._memory_release = old[0]
        app.STATE.clear()
        app.STATE.update(old[1])


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed")
