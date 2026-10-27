"""An untagged item finds its row through the (bid, p, c) index, not a full scan.

tick() matched every loose item by scanning every row in `done`. Both stores hold
up to ITEMS_MAX entries and the scan ran under the lock ingest() takes, so a busy
proxy could hold the request path for seconds per tick. The rows are now indexed
by (bid, prompt_tokens, completion_tokens) — the same candidates, without the
sweep. This test counts the sweeps: a dict subclass records every .items() call
on `done` while 200 loose items wait next to 5000 rows.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proxy import collect_api  # noqa: E402


class _Counting(dict):
    def __init__(self):
        super().__init__()
        self.sweeps = 0

    def items(self):
        self.sweeps += 1                     # the one from _expire's housekeeping
        return super().items()


def test_no_per_item_sweep_of_done():
    col = collect_api.Collector.__new__(collect_api.Collector)
    col._lock = __import__("threading").Lock()
    col.done = _Counting()
    col.done_by = {}
    col.items = {}
    col.loose = {}
    col.expect = {}
    col.backends = {}
    col._n = 0
    col.svc = None
    col.log = lambda *a: None
    t = [1000.0]
    col.clock = lambda: t[0]
    col._b = lambda bid: collect_api._Backend()

    # 5000 rows near the cap, 200 loose items of the same token counts (ambiguous on
    # purpose: the answer is 'no guess' either way — only the work per item is the point)
    for i in range(5000):
        dk = ("v1", f"row:{i}")
        col.done[dk] = {"request_id": i, "ts_ms": 0, "t": 1000.0, "end": 1000.0,
                        "prompt_tokens": 7, "completion_tokens": 9, "tagged": True}
        col.done_by.setdefault(("v1", 7, 9), set()).add(dk)
    for n in range(200):
        col.loose[("v1", n)] = ({"prompt_tokens": 7, "completion_tokens": 9,
                                "t_end": 1000.0}, 1000.0)

    col.tick()                              # one round of the old code = 200 sweeps
    assert len(col.loose) == 200, "the ambiguous case still declines to guess"
    assert col.done.sweeps <= 1, f"done was swept {col.done.sweeps} times in one tick"


def test_a_re_used_key_leaves_its_old_count_bucket():
    # A client that reuses an X-Request-Id overwrites done[dk]. Without eviction from
    # the old (p, c) bucket, a 100/50 item joined the 7/3 row written second; main's
    # scan, comparing tokens, returned None for that item.
    col = collect_api.Collector.__new__(collect_api.Collector)
    col._lock = __import__("threading").Lock()
    col.done = {}
    col.done_by = {}
    col.items = {}
    col.loose = {}
    col.expect = {}
    col.backends = {}
    col._n = 0
    col.svc = None
    col.log = lambda *a: None
    t = [1000.0]
    col.clock = lambda: t[0]
    col._b = lambda bid: collect_api._Backend()

    dk = ("v1", "row:r1")                   # untagged: an approximate join may use it
    col.done[dk] = {"request_id": "r1", "ts_ms": 0, "t": 1000.0, "end": 1000.0,
                    "prompt_tokens": 100, "completion_tokens": 50, "tagged": False}
    col.done_by.setdefault(("v1", 100, 50), set()).add(dk)

    assert col._approx("v1", {"prompt_tokens": 100, "completion_tokens": 50,
                              "t_end": 1000.0}) == dk        # the first row is findable
    # the overwrite as on_row does it now: pop first, re-insert under (7, 3)
    col._done_pop(dk)
    col.done[dk] = {"request_id": "r1", "ts_ms": 0, "t": 1000.0, "end": 1000.0,
                    "prompt_tokens": 7, "completion_tokens": 3, "tagged": False}
    col.done_by.setdefault(("v1", 7, 3), set()).add(dk)
    assert col._approx("v1", {"prompt_tokens": 100, "completion_tokens": 50,
                              "t_end": 1000.0}) is None      # main's answer: no match
    assert ("v1", 100, 50) not in col.done_by               # old bucket is gone


if __name__ == "__main__":
    test_no_per_item_sweep_of_done()
    test_a_re_used_key_leaves_its_old_count_bucket()
    print("2 passed")
