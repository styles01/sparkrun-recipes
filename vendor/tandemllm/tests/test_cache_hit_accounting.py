"""The dashboard's cache hit and prefill rate count what the engine did (2026-10-09).

Two readings on the live EXL3 4.5 server (0.5.0), both from a phone:

  1  Perf -> Parallel: "Cache hit 117.2 % -- 112K of 95.3K prompt tokens, 10 requests". The
     parallel view summed the engine's own cached rows (`rec.cached_tokens`, served positions:
     the router's notes, the end anchor's replay and the earlier answers' hidden tokens included)
     over the client's prompt (`prompt_tokens`, SRV-165). The ledger's end-anchor row
     chatcmpl-f667a: 33,273 client tokens, 64,223 rows reused, 608 forwarded -- the lane read a
     193 % hit. `usage`, the ledger and the live view already counted the client's share
     (`client_cached`); the parallel view now uses the same definition.
  2  Perf -> Live: "Prefilling 26,624 of 29,621 (90 %), 17,892 tok/s" on a 19,381-token prefill
     that took 24.4 s (794 tok/s, ledger chatcmpl-8c5b0). An EXL3 prefill window computes 8
     chunks layer-major and then reports them within milliseconds; the hook timed each chunk as
     it was reported, so the last ~2 s of marks read 1,024 rows in ~57 ms. A window's chunks now
     carry the window's time and make one mark (`activity.pf_mark`).

  a  an engine lane's cached tokens never exceed its prompt tokens; the window's hit is <= 1
  b  a record without hidden tokens is clamped too; the hidden positions still come off
  c  a prefill window's chunk burst is one mark; the live rate is the rows computed per second
  d  a finished prefill's rate is the rows forwarded over the prefill time, the restored rows
     and the hidden ones counted in the served total, never in the rate

Run: python tests/test_cache_hit_accounting.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import activity  # noqa: E402
from server import live as live_mod  # noqa: E402
from server import parallel as par  # noqa: E402
from server.usage import RequestRecord, client_cached_of  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _board():
    perf = Clock()
    reg = live_mod.LiveRegistry(perf=perf, clock=lambda: 1.7e9 + perf.t, activity=False)
    board = par.ParallelBoard(reg, perf=perf, clock=lambda: 1.7e9 + perf.t)
    return reg, board, perf


def _engine_rec(rid, perf, prompt, hidden, cached, forwarded, hidden_at=None):
    """An engine request as the handler leaves it: the client's prompt, the hidden tokens the
    served prompt added, the engine's reused and forwarded rows (served positions)."""
    rec = RequestRecord(rid, "chat", True, t_arrival=perf.t)
    rec.model, rec.client_kind = "TandemLLM", "open-webui"
    rec.prompt_tokens, rec.hidden_prompt, rec.hidden_at = prompt, hidden, hidden_at
    rec.cached_tokens, rec.prompt_n, rec.cache_source = cached, forwarded, "end"
    rec.t_lock = perf.t + 0.01
    rec.t_first = perf.t + 0.5
    rec.t_last = perf.t + 2.0
    rec.completion_tokens = rec.n_live = 100
    rec.finish_reason = "stop"
    return rec


def _finish(reg, rec, perf):
    perf.t += 2.0
    rec.t_end = perf.t
    reg.finish(rec)


# ------------------------------------------------------------------ a, b: the cache hit
def test_engine_lane_cached_never_exceeds_its_prompt():
    reg, board, perf = _board()
    # the ledger's rows of 23:13 (end anchor, positions unknown) and 23:21 (resident, known)
    rows = [("f667a", 33273, 64831 - 33273, 64223, 608, None),
            ("5abc7", 31643, 60510 - 31643, 60059, 451, None),
            ("fa3cd", 28548, 9326, 28672, 9202, [(0, 6176), (30000, 33150)])]
    for rid, p, h, c, fw, at in rows:
        rec = _engine_rec(rid, perf, p, h, c, fw, at)
        reg.register(rec)
        _finish(reg, rec, perf)
    board._built = None
    doc = board.snapshot()
    for lane in doc["recent"]:
        assert lane["cached_tokens"] <= lane["prompt_tokens"], lane
        assert lane["cache_hit"] <= 1.0, lane
    t = doc["totals"]
    assert t["cached_tokens_window"] <= t["prompt_tokens_window"], t
    assert 0.0 <= t["cache_hit"] <= 1.0, t
    assert doc["backends"][0]["cache_hit"] <= 1.0
    # the same definition as `usage` and the ledger: the known positions come off
    fa = next(x for x in doc["recent"] if x["id"] == "fa3cd")
    assert fa["cached_tokens"] == 28672 - 6176 == 22496, fa


def test_a_prefilling_lane_counts_the_client_share_of_its_restored_rows():
    reg, board, perf = _board()
    rec = _engine_rec("live", perf, 23378, 6243, 0, None)
    rec.t_first = rec.t_last = None
    rec.completion_tokens = rec.n_live = 0
    rec.finish_reason = None
    rec.prefill_info = {"start": 26624, "chunk": 1024, "t0": perf.t + 0.02, "kind": "end"}
    rec.pf_marks = [(27648, 29621, perf.t + 1.0), (28672, 29621, perf.t + 2.2)]
    reg.register(rec)
    perf.t += 2.5
    lane = board.snapshot()["lanes"][0]
    assert lane["state"] == "prefilling", lane
    assert lane["cached_tokens"] <= lane["prompt_tokens"] == 23378, lane


def test_client_cached_of_clamps_and_takes_the_positions_off():
    class R:
        prompt_tokens, hidden_prompt, hidden_at, prompt_client = 1000, 0, None, None
    r = R()
    assert client_cached_of(r, 1200) == 1000, "no hidden tokens: still at most the prompt"
    assert client_cached_of(r, 800) == 800
    r.hidden_prompt, r.hidden_at = 300, [(100, 400)]
    assert client_cached_of(r, 1200) == 900
    r.hidden_at = None
    assert client_cached_of(r, 1200) == 1000, "positions unknown: the clamp"
    r.prompt_tokens = None
    assert client_cached_of(r, 50) == 50, "a request that never ran keeps the engine's figure"
    rec = RequestRecord("u", "chat", False)
    rec.prompt_tokens, rec.cached_tokens = 500, 700
    assert rec.usage()["prompt_tokens_details"]["cached_tokens"] == 500
    assert rec.row()["cached_tokens"] == 500 and rec.row()["engine_cached"] == 700


# ------------------------------------------------------------------ c, d: the prefill rate
def _window_prefill(hook_times, compute_s=10.3, start=10240, n=29621, chunk=1024, win=8192):
    """The engine's window loop: each window computes, then the chunk hook runs once a chunk
    within milliseconds (`hook_times` apart), as engine/cache.py does."""
    marks, info, t = [], {}, 0.0
    i = start
    while i < n:
        plan, j = [], i
        while j < n:
            k = min(chunk - (j % chunk), n - j)
            if plan and (j + k) - i > win:
                break
            plan.append((j, k))
            j += k
        t += compute_s * sum(k for _, k in plan) / win
        info["window"] = (plan[0][0], plan[-1][0] + plan[-1][1], t)
        for a, k in plan:
            t += hook_times
            i = a + k
            if i < n:
                activity.pf_mark(marks, i, n, t, info.get("window"))
    return marks


def test_a_prefill_window_is_one_mark_and_its_rate_is_the_rows_computed():
    full = _window_prefill(0.057)
    # one mark a window (the last window's last chunk is never reported, as before)
    assert [m[0] for m in full] == [18432, 26624, 28672], full
    marks = full[:2]                                 # the screenshot's moment: 26,624 of 29,621
    rec = RequestRecord("p", "chat", True)
    rec.prompt_tokens, rec.hidden_prompt = 23378, 6243
    rec.pf_marks = marks
    rec.prefill_info = {"start": 10240, "chunk": 1024, "t0": 0.0, "kind": "disk"}
    pl = activity.prefill_live(rec, 1, 2.0)
    assert pl["tps_live"] is not None and 700 <= pl["tps_live"] <= 900, pl
    assert 700 <= pl["tps_live_avg"] <= 900, pl
    rec.t_lock = 0.0
    p = activity.prefill_of(rec, marks[-1][2] + 0.1)
    assert p["done"] == 26624 and p["total"] == 29621 and p["cached"] == 10240, p
    assert 700 <= p["tps_now"] <= 900, p
    eta_s = p["eta_ms"] / 1e3
    assert 3.0 <= eta_s <= 4.5, p                    # 2,997 rows at ~795 tok/s, not "about 2 s"


def test_chunk_major_marks_are_timed_as_reported():
    marks, t = [], 0.0
    for done in (2048, 3072, 4096):
        t += 1.3
        activity.pf_mark(marks, done, 5000, t, None)
    assert marks == [(2048, 5000, 1.3), (3072, 5000, 2.6), (4096, 5000, 3.9000000000000004)]
    # a stale window that does not hold the row is ignored
    activity.pf_mark(marks, 4500, 5000, 4.0, (0, 4096, 1.0))
    assert marks[-1] == (4500, 5000, 4.0)


def test_a_finished_prefill_rate_is_the_rows_forwarded():
    rec = RequestRecord("f", "chat", True, t_arrival=0.0)
    rec.prompt_tokens, rec.hidden_prompt = 23378, 6243
    rec.cached_tokens, rec.prompt_n, rec.cache_source = 10240, 19381, "disk"
    rec.t_lock, rec.t_first = 0.013, 0.013 + 24.40134
    rec.prefill_info = {"start": 10240, "chunk": 1024, "t0": 0.02, "kind": "disk"}
    p = activity.prefill_of(rec, 30.0)
    assert p["total"] == 29621 and p["cached"] == 10240, p
    assert abs(p["tps_avg"] - 794.26) < 0.5, p      # the ledger's prefill_tps of chatcmpl-8c5b0


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            n += 1
    print(f"{n} passed")
