"""`cached_tokens` counts the client's tokens (SRV-233) and the parallel admission sizes a request
by what it needs (SRV-250).

SRV-233 (UX2, 2026-10-07). An opencode turn's engine prompt held 24,706 client and 5,745 hidden
note tokens (three notes at 7,407, 13,203 and 20,711), resumed at 26,624 and forwarded 3,827:
usage said 24,706 of 24,706 cached, because the clamp of SRV-165 counted the notes inside the
cached span as client tokens. With the hidden tokens' positions the client's cached tokens are
26,624 - 5,745 = 20,879; the ledger keeps the engine's figures beside it.

SRV-250 (FX7, 2026-10-07). With `--parallel 2` every request was counted as 2.1 GiB (the
workspace floor plus a whole state-store entry) against 6.4 to 8.4 GiB of cache budgets: a
576-token request waited 63 and 83 s at the box's idle MemAvailable, the wait booked as prefill.

  1  the O03 numbers: 20,879 client tokens cached; without positions the clamp, as before; a turn
     with no hidden tokens unchanged (W05: 1.00); the ledger row's engine_cached and forwarded
  2  the positions of a served prompt: merged, the think block at the tail; a count that does not
     add up gives None (the clamp)
  3  the need is the request's own workspace; the engine's growth is capped by its room to the
     memory policy's peak; 576 tokens beside a 9,000-token request at 25 GiB available pass
  4  a held request: its wait is queue time, `wait_mem_ms` books it, the waiting line is said; a
     request alone says so in the log

Run: python tests/test_client_cached.py
"""

from __future__ import annotations

import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

from server import app, slots as slots_mod, usage  # noqa: E402
from engine import cache  # noqa: E402

GiB = slots_mod.GiB


def _rec():
    r = usage.RequestRecord("chatcmpl-x", "chat", True)
    return r


# ------------------------------------------------------------------ 1-2 SRV-233

def test_1_the_o03_turn():
    r = _rec()
    notes = [(7407, 7407 + 1915), (13203, 13203 + 1915), (20711, 20711 + 1915)]
    r.prompt_tokens, r.hidden_prompt = 24706, 5745
    r.absorb_prefill({"reused": 26624, "forwarded": 3827, "kind": "resident"})
    assert r.client_cached == 24706                    # The clamp, 1.00
    r.hidden_at = notes
    assert r.client_cached == 20879, r.client_cached
    assert r.usage()["prompt_tokens_details"]["cached_tokens"] == 20879
    row = r.row()
    assert row["cached_tokens"] == 20879 and row["engine_cached"] == 26624
    assert row["forwarded_tokens"] == 3827
    # a note above the resume point is forwarded, not cached
    r.hidden_at = notes[:2] + [(27000, 27000 + 1915)]
    assert r.client_cached == 26624 - 2 * 1915
    # no hidden tokens (an Open WebUI follow-up): every cached token is the client's
    p = _rec()
    p.prompt_tokens, p.hidden_prompt = 7627, 0
    p.absorb_prefill({"reused": 7627, "forwarded": 0, "kind": "end"})
    p.hidden_at = []
    assert p.client_cached == 7627


def test_2_the_positions():
    hp = app._hidden_positions
    assert hp([(10, 20), (15, 30), (40, 45)], 0, 100, 100, 25, None) == [(10, 30), (40, 45)]
    assert hp([(10, 20)], 3, 100, 100, 13, None) == [(10, 20), (97, 100)]
    assert hp([(10, 20)], 0, 100, 101, 10, None) is None          # the prompt changed after
    assert hp([(10, 20)], 0, 100, 100, 500, None) is None         # the count does not add up
    assert hp(None, 0, 100, 100, 10, None) is None                 # SP4 not one run


# ------------------------------------------------------------------ 3-4 SRV-250

class _Mem:
    on = True

    def __init__(self, peak, reserved, per_token=cache.WORKSPACE_PER_TOKEN):
        self.peak, self._r, self.ws_floor, self.ws_per_token = peak, reserved, \
            cache.WORKSPACE_FLOOR, per_token

    def memory(self):
        return self._r, self._r

    def workspace(self, n):
        return max(self.ws_floor, self.ws_per_token * int(n))


class _Handler:
    def __init__(self):
        self.busy = None

    def _client_gone(self):
        return False

    def _busy(self, code, msg, retry=None):
        self.busy = (code, msg)


def _pool(avail_gib, growth=0, floor_gib=15.0):
    box = {"avail": int(avail_gib * GiB)}
    adm = slots_mod.Admission(int(floor_gib * GiB), meminfo=lambda: {"MemAvailable": box["avail"]},
                              growth=lambda: growth)
    return types.SimpleNamespace(admission=adm, turn=None), box


def test_3_the_need_and_the_engines_room():
    import torch
    app.STATE.clear()
    app.STATE["mem"] = _Mem(peak=int(84.6 * GiB), reserved=int(77.4 * GiB))
    app.STATE["state_store"] = types.SimpleNamespace(budget=int(10 * GiB), bytes=int(2 * GiB),
                                                     max_entry=GiB)
    assert app._par_growth() == 8 * GiB                      # the caches' budgets left
    assert abs(app._par_room() - int(7.2 * GiB)) < GiB // 100  # the engine's room to its peak
    # FX7's live case on MemAvailable 25.0, growth 6.6, in flight 2.1, need 2.1 ->
    # 14.1 (< 15): the 576-token request waited 63 s. Now its need is its workspace and the
    # engine's room caps what all of it may take
    adm = slots_mod.Admission(int(15 * GiB), meminfo=lambda: {"MemAvailable": int(22.7 * GiB)},
                              growth=lambda: int(6.6 * GiB), room=app._par_room)
    pool = types.SimpleNamespace(admission=adm, turn=None)
    assert adm.reserve("big", app.STATE["mem"].workspace(9000))[0]
    h = _Handler()
    t0 = time.perf_counter()
    assert app.Handler._par_admit(h, pool, "small", torch.zeros(576)) is True
    assert time.perf_counter() - t0 < 0.5, "no wait"
    held = adm.held
    assert held["small"] == cache.WORKSPACE_FLOOR, held    # the workspace, no store entry
    ok, d = adm.check(cache.WORKSPACE_FLOOR, count=False)
    assert d["room"] is not None and abs(d["after"] - int((22.7 - 7.2) * GiB)) < GiB // 100, d
    assert "room to its peak" in slots_mod.Admission.describe(d)
    # the engine near its peak: little room, nothing waits that the policy would refuse anyway
    app.STATE["mem"]._r = int(84 * GiB)
    assert abs(app._par_room() - int(0.6 * GiB)) < GiB // 100
    # where MemAvailable itself is low the floor still holds: 15.7 GiB with 0.6 of room passes,
    # 15.5 GiB does not
    for avail, want in ((15.7, True), (15.5, False)):
        a2 = slots_mod.Admission(int(15 * GiB), meminfo=lambda v=avail: {"MemAvailable": int(v * GiB)},
                                 growth=lambda: int(6.6 * GiB), room=app._par_room)
        assert a2.reserve("big", GiB)[0]                  # alone
        assert a2.reserve("small", cache.WORKSPACE_FLOOR)[0] is want, avail


def test_4_a_held_request_waits_in_the_queue_and_says_so():
    import torch
    app.STATE.clear()
    app.STATE["queue_timeout"] = 30.0
    pool, box = _pool(16.0, growth=0)
    assert pool.admission.reserve("big", 2 * GiB)[0]
    rec = _rec()
    rec.lock_acquired()
    lock_at = rec.t_lock
    said = []

    def free_later():
        time.sleep(1.5)
        box["avail"] = int(30 * GiB)
    import threading
    threading.Thread(target=free_later, daemon=True).start()
    h = _Handler()
    assert app.Handler._par_admit(h, pool, "small", torch.zeros(576), rec=rec,
                                  say=said.append) is True
    assert said and said[0] > 0, said                     # "memory ..., N GiB short"
    assert rec.wait_mem_ms >= 1000, rec.wait_mem_ms
    assert rec.t_lock - lock_at >= 1.0, "the wait moved the lock time: queue_ms holds it"
    rec.t_first = time.perf_counter()
    assert rec.prompt_ms < 100, rec.prompt_ms
    # alone: admitted without the check, and the log says so
    pool2, _ = _pool(1.0)
    ok, d = pool2.admission.reserve("only", GiB)
    assert ok and "admitted alone" in slots_mod.Admission.describe(d)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:70s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
