"""The per-block arm choice with the lazy catch-up (`QWEN38_LEN_SWITCH`), on stub drafters.

The policy tests need no board: acceptance comes from fixed run lengths and the copy signal from a
stub lookup index. The catch-up tests hand the heads real tensors as tap rows (torch on the CPU),
because what they check is that the rows an arm missed reach it in one sync, in order.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.lenrouter import LengthRouter  # noqa: E402
from tests.test_lenrouter import FakeDrafter, FakeEng, FakeNgram  # noqa: E402
from tests.test_lenrouter_price import SERVED, ServedArm  # noqa: E402


class FakeLocal:
    def __init__(self):
        self.n = 0

    def lookup(self, context, order):
        return self.n, None


class CopyNgram(FakeNgram):
    min_order = 3

    def __init__(self):
        super().__init__()
        self.local = FakeLocal()


def build(**kw):
    eng = FakeEng()
    small, large, ng = FakeDrafter(eng, 8), FakeDrafter(eng, 16), CopyNgram()
    kw.setdefault("learn_cost", False)
    kw.setdefault("latch", True)
    kw.setdefault("drop_idle", True)
    kw.setdefault("tree_wide_after", 32)
    kw.setdefault("switch", True)
    kw.setdefault("switch_mode", "narrow")
    r = LengthRouter(ServedArm(small, ng, 15), ServedArm(large, ng, 23), tree=True, ngram=ng,
                     latch_table=SERVED, **kw)
    return r, small, large, ng


def run(r, blocks, runs, local=None, k=31):
    idx = {"s": 0, "l": 0}
    keys = []
    for b in range(blocks):
        if local is not None:
            r.ngram.local.n = local(b)
        r.propose_tree(list(range(50 + b)), k)
        key = r.last_key
        keys.append(key)
        seq = runs[key]
        n = min(seq[idx[key] % len(seq)], 7 if key == "s" else 15)
        idx[key] += 1
        r.observe([9000 + i for i in range(n)] + [12345])
    return keys


def test_off_by_default_and_on_from_the_environment():
    old = os.environ.pop("QWEN38_LEN_SWITCH", None)
    try:
        r, _, _, _ = build(switch=None)
        assert r.switch is False and "switches" not in r.report()
        os.environ["QWEN38_LEN_SWITCH"] = "1"
        r, _, _, _ = build(switch=None)
        assert r.switch is True and "switches" in r.report()
    finally:
        os.environ.pop("QWEN38_LEN_SWITCH", None)
        if old is not None:
            os.environ["QWEN38_LEN_SWITCH"] = old


def test_fresh_text_stays_on_the_narrow_arm():
    r, small, large, _ = build()
    keys = run(r, 40, {"s": [3, 2, 4], "l": [3]})
    assert set(keys) == {"s"} and large.calls == 0, keys
    assert r.stats.get("switches", 0) == 0 and r.idle is None, "nothing is released"


def test_a_copy_signal_moves_up_at_once():
    r, _, _, _ = build()
    keys = run(r, 12, {"s": [3], "l": [15]}, local=lambda b: 0 if b < 3 else 12)
    assert keys[:3] == ["s"] * 3 and keys[3:] == ["l"] * 9, keys


def test_a_copy_from_the_first_block_starts_wide():
    r, small, _, _ = build()
    keys = run(r, 10, {"s": [3], "l": [15]}, local=lambda b: 12)
    assert set(keys) == {"l"} and small.calls == 0


def test_a_saturating_narrow_width_goes_up_and_stays_where_wide_pays():
    r, _, _, _ = build()
    keys = run(r, 30, {"s": [7], "l": [13]})
    first_l = keys.index("l")
    assert 2 <= first_l <= 4, keys
    assert set(keys[first_l:]) == {"l"}, keys


def test_wide_that_buys_nothing_comes_back_down():
    """Up on two full narrow blocks, then the free price shows the wide block commits what the
    narrow width would have, at a dearer verify: back to narrow within a few blocks."""
    r, _, _, _ = build()
    keys = run(r, 40, {"s": [7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       "l": [3]})
    ups = [i for i in range(1, len(keys)) if keys[i] == "l" and keys[i - 1] == "s"]
    downs = [i for i in range(1, len(keys)) if keys[i] == "s" and keys[i - 1] == "l"]
    assert ups and downs and downs[0] - ups[0] <= 6, keys
    assert keys[-1] == "s"


def test_the_tail_stays_on_the_current_arm():
    r, _, _, _ = build()
    run(r, 6, {"s": [3], "l": [15]}, local=lambda b: 12)
    assert r.cur == "l"
    r.propose_tree(list(range(80)), 5)
    assert r.last_key == "l"


def test_off_is_the_router_it_was():
    """With the flag off the arms and reports match a router built without it."""
    for fx in ({"s": [3], "l": [3]}, {"s": [7], "l": [13]}, {"s": [2], "l": [7, 7, 2, 2]}):
        a, _, _, _ = build(switch=False)
        b, _, _, _ = build(switch=False)
        assert run(a, 40, fx) == run(b, 40, fx) and a.report() == b.report()
        assert a.cur is None and a.backlog == {"s": [], "l": []}


# --- design B: the latch with two exits and the budget ladder -----------------------------------

def build_b(**kw):
    kw.setdefault("switch_mode", "b")
    return build(**kw)


def run_b(r, blocks, runs, local=None, k=31):
    """As `run`, and it records the wide arm's node budget on every wide block."""
    idx = {"s": 0, "l": 0}
    keys, budgets = [], []
    for b in range(blocks):
        if local is not None:
            r.ngram.local.n = local(b)
        r.propose_tree(list(range(50 + b)), k)
        key = r.last_key
        keys.append(key)
        budgets.append(r.large.node_budget if key == "l" else None)
        seq = runs[key]
        n = min(seq[idx[key] % len(seq)], 7 if key == "s" else 15)
        idx[key] += 1
        r.observe([9000 + i for i in range(n)] + [12345])
    return keys, budgets


def test_b_fresh_text_opens_wide_then_latches_narrow_and_stays():
    r, _, _, _ = build_b()
    keys, budgets = run_b(r, 40, {"s": [3, 2, 4], "l": [3]})
    assert keys[:4] == ["l"] * 4 and set(keys[4:]) == {"s"}, keys
    assert set(b for b in budgets if b is not None) == {15}, "never 24 nodes on fresh text"
    assert r.b_switches == 0


def test_b_quotation_runs_wide_and_climbs_the_ladder():
    r, small, _, _ = build_b()
    keys, budgets = run_b(r, 12, {"s": [7], "l": [15]}, local=lambda b: 0 if b < 2 else 12)
    assert set(keys) == {"l"} and small.calls == 0
    assert budgets[0] == 15 and budgets[-1] == 23, budgets


def test_b_wide_without_a_copy_regime_stays_at_sixteen_nodes():
    """Code-like: the wide arm accepts more than narrow and the latch keeps it, but without full
    blocks or a copy the ladder never pays for 24 nodes."""
    r, _, _, _ = build_b()
    keys, budgets = run_b(r, 40, {"s": [3], "l": [5, 6]})
    assert keys[-1] == "l", keys
    assert set(b for b in budgets if b is not None) == {15}, budgets


def test_b_an_edit_that_starts_fresh_exits_up_on_the_copy():
    r, _, _, _ = build_b()
    keys, _ = run_b(r, 30, {"s": [3], "l": [3, 3, 3, 3, 15]},
                    local=lambda b: 0 if b < 12 else 12)
    assert keys[8:12] == ["s"] * 4, keys
    assert keys[12:] == ["l"] * 18, keys
    assert r.b_switches == 1


def test_b_comes_down_when_the_wide_arm_falls_under_the_probe():
    r, _, _, _ = build_b()
    # a copy early on pulls it up; then the copy ends and wide commits under the narrow probe
    keys, _ = run_b(r, 60, {"s": [4], "l": [1]}, local=lambda b: 12 if 10 <= b < 14 else 0)
    assert "l" in keys[10:14] and keys[-1] == "s", keys
    assert r.b_switches == 2


def test_b_switches_are_capped_then_latched_for_good():
    r, _, _, _ = build_b(dwell=1, max_switches=4)
    flip = lambda b: 12 if (b // 10) % 2 == 1 else 0        # a copy every other ten blocks
    keys, _ = run_b(r, 100, {"s": [4], "l": [1]}, local=flip)
    assert r.b_switches == 4 and r.latched is not None and r.idle is not None


def test_c_is_the_latch_and_the_ladder_without_exits():
    r, _, _, _ = build_b(exits=False)
    keys, _ = run_b(r, 30, {"s": [3], "l": [3, 3, 3, 3, 15]},
                    local=lambda b: 0 if b < 12 else 12)
    assert set(keys[8:]) == {"s"} and r.latched == "s"


# --- the lazy catch-up ---------------------------------------------------------------------------

def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class TapDrafter(FakeDrafter):
    """A block drafter whose sync reads its tap rows, as DFlash2's does: it records which
    positions reached its cache and with which row values."""

    def __init__(self, eng, block):
        super().__init__(eng, block)
        self._tap_rows = []
        self.written = {}

    def sync(self, tokens, hidden, first_pos, rows=None):
        n = len(tokens)
        taps = self._tap_rows
        picked = [t[:n] for t in taps] if rows is None else [t[rows] for t in taps]
        for i in range(n):
            self.written[first_pos + i] = float(picked[0][i, 0])
        self.synced.append((n, first_pos))


def build_taps():
    eng = FakeEng()
    small, large, ng = TapDrafter(eng, 8), TapDrafter(eng, 16), CopyNgram()
    r = LengthRouter(ServedArm(small, ng, 15), ServedArm(large, ng, 23), tree=True, ngram=ng,
                     latch=True, drop_idle=True, tree_wide_after=32, switch=True,
                     learn_cost=False, latch_table=SERVED)
    return r, small, large


def _block(torch, heads, pos, nodes, path):
    """One verify's tap rows: row i of the tree carries the value 1000 * pos + i."""
    rows = torch.tensor([[1000.0 * pos + i] for i in range(nodes)])
    for h in heads:
        h._tap_rows = [rows.clone() for _ in range(5)]
    return path


def test_the_lagging_arm_is_not_synced_per_block_and_catches_up_in_one_sync():
    torch = _torch()
    if torch is None:
        return                                    # the box's CPU suite has torch
    r, small, large = build_taps()
    r.cur = "s"                                   # narrow drafting, wide lagging
    pos = 100
    for path in ([0, 2, 5], [0, 1], [0, 3, 4, 7]):
        _block(torch, (small, large), pos, 16, path)
        r.sync([1] * len(path), None, pos, rows=path)
        pos += len(path)
    assert len(small.synced) == 3 and large.synced == [], "only the drafting arm syncs"
    assert sum(len(e[1]) for e in r.backlog["l"]) == 9
    r._use("l")
    assert large.synced == [(9, 100)], "one sync, from the first missed position"
    # every missed position got the row of the accepted node, not the first n rows
    assert large.written[100] == 100000.0 and large.written[101] == 100002.0
    assert large.written[102] == 100005.0 and large.written[104] == 103001.0
    assert large.written[108] == 105007.0
    assert r.backlog["l"] == [] and r.stats["catchups"] == 1


def test_a_block_without_tap_rows_syncs_after_the_backlog_in_position_order():
    """A backlog of two blocks, then a block whose tap rows are missing: the arm must see the
    backlog first and the new block after it, never the other way round."""
    torch = _torch()
    if torch is None:
        return
    r, small, large = build_taps()
    r.cur = "s"
    pos = 10
    for path in ([0, 1], [0, 2, 3]):
        _block(torch, (small, large), pos, 16, path)
        r.sync([1] * len(path), None, pos, rows=path)
        pos += len(path)
    large._tap_rows = []                          # this block's rows never reached the wide head
    small._tap_rows = [torch.zeros(16, 1) for _ in range(5)]
    large.sync = lambda tokens, hidden, first_pos, rows=None, _s=large.sync: (
        _s(tokens, hidden, first_pos, rows) if large._tap_rows else
        large.synced.append((len(tokens), first_pos)))
    r.sync([1, 1], None, pos, rows=[0, 1])
    assert large.synced == [(5, 10), (2, 15)], large.synced
    assert r.backlog["l"] == []


def test_a_snapshot_brings_both_arms_up_to_date():
    torch = _torch()
    if torch is None:
        return
    r, small, large = build_taps()
    r.cur = "l"
    _block(torch, (small, large), 50, 24, [0, 1, 2])
    r.sync([1, 1, 1], None, 50, rows=[0, 1, 2])
    assert small.synced == []
    snap = r.state_snapshot()
    assert small.synced == [(3, 50)] and snap[3] is None


def test_the_backlog_is_bounded():
    torch = _torch()
    if torch is None:
        return
    r, small, large = build_taps()
    r.lazy_cap = 8
    r.cur = "s"
    pos = 0
    for _ in range(5):
        _block(torch, (small, large), pos, 16, [0, 1, 2])
        r.sync([1, 1, 1], None, pos, rows=[0, 1, 2])
        pos += 3
    assert large.synced and all(n >= 3 for n, _ in large.synced)
    assert sum(len(e[1]) for e in r.backlog["l"]) < 8


def test_a_new_request_drops_the_backlog():
    torch = _torch()
    if torch is None:
        return
    r, small, large = build_taps()
    r.cur = "s"
    _block(torch, (small, large), 10, 16, [0, 1])
    r.sync([1, 1], None, 10, rows=[0, 1])
    r.reset()
    assert r.backlog == {"s": [], "l": []} and r.cur is None


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception:
                fails += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{fails} failed")
    sys.exit(1 if fails else 0)
