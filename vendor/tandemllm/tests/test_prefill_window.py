"""The layer-major prefill window (engine/model.py `prefill_window`, engine/cache.py `_window_plan`)
must be the chunk-major prefill bit for bit: the logits, the KV, the recurrent and convolution state,
every resident anchor and store snapshot, and what the drafter's tap and sync receive, chunk by chunk.

The random four-layer model on the CPU, the fused norms replaced by the torch arithmetic they stand for
(as test_addnorm.py does), the grid at 16 rows and a window of 48."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_addnorm as A  # noqa: E402
import test_forward_tree as T  # noqa: E402

import torch  # noqa: E402

import engine.model as M  # noqa: E402
from engine import cache  # noqa: E402
from tools import norm_kernels as NK  # noqa: E402


class TapDrafter:
    """A drafter that keeps two tap rows a forward and records every sync."""

    def __init__(self, eng, want=(0, 2, 4)):
        self.eng = eng
        self.want = set(want)
        self.i = 0
        self.rows = {}
        self.syncs = []
        eng.tap = self._on_tap

    def tap_indices(self):
        return set(self.want)

    def _on_tap(self, h):
        if self.i == 0:
            self.rows = {}
        if self.i in self.want:
            self.rows[self.i] = h.clone()
        self.i = (self.i + 1) % (self.eng.cfg.num_hidden_layers + 1)

    def sync(self, tokens, hidden, first_pos, rows=None):
        self.syncs.append((list(tokens), hidden.clone(), first_pos,
                           {k: v.clone() for k, v in sorted(self.rows.items())}))

    def can_resume(self):
        return True

    def state_resume(self, *a, **k):
        return None

    def reset(self):
        self.i, self.rows, self.syncs = 0, {}, []


class BlindTap(TapDrafter):
    """A tap owner that cannot say what it keeps: the window must step aside."""
    tap_indices = None


def _run(window: int, ids, chunk=16, cached=True, cls=TapDrafter, start=None):
    old = (NK.rms_norm, NK.add_rms_norm, NK.rms_norm_gated, M.FUSED["norm"], M.FUSED_ADDNORM,
           cache.PREFILL_WINDOW, cache.PREFILL_WINDOW_FREE)
    NK.rms_norm, NK.add_rms_norm, NK.rms_norm_gated = A._torch_norm, A._torch_add_norm, A._torch_gated
    M.FUSED["norm"], M.FUSED_ADDNORM = True, True
    cache.PREFILL_WINDOW = cache.PREFILL_WINDOW_FREE = window
    calls = []
    try:
        cfg = T.tiny_config()
        eng = M.Qwen38Engine(cfg, T.FakeWeights(cfg, 5), max_len=256, device="cpu")
        eng.kv.k, eng.kv.v = eng.kv.k.float(), eng.kv.v.float()
        eng.state.conv = eng.state.conv.float()
        orig = eng.prefill_window

        def spy(toks, starts, bufs=None):
            calls.append(list(starts))
            return orig(toks, starts, bufs)
        eng.prefill_window = spy
        dr = cls(eng)
        res = cache.ResidentPrefix(1 << 30, chunk, tail=4) if cached else None
        store = cache.StateStore(1 << 30, chunk=chunk) if cached else None
        with torch.no_grad():
            if start:
                cache.prefill(eng, dr, ids[:start], "cpu", store=store, chunk=chunk,
                              checkpoint=cached, resident=res)
            lg = cache.prefill(eng, dr, ids, "cpu", store=store, chunk=chunk, checkpoint=cached,
                               resident=res)[0]
        n = len(ids)
        out = {"logits": lg.clone(), "k": eng.kv.k[..., :n, :].clone(),
               "v": eng.kv.v[..., :n, :].clone(), "S": eng.state.S.clone(),
               "conv": eng.state.conv.clone(), "len": eng.kv.length, "syncs": dr.syncs,
               "anchors": ({b: (s.clone(), c.clone()) for b, (s, c) in res.anchors.items()}
                           if res is not None else {}),
               "snaps": ({e.snap.length: (e.snap.state[0].clone(), e.snap.views[0].clone())
                          for e in store._d.values()} if store is not None else {})}
        return out, calls
    finally:
        (NK.rms_norm, NK.add_rms_norm, NK.rms_norm_gated, M.FUSED["norm"], M.FUSED_ADDNORM,
         cache.PREFILL_WINDOW, cache.PREFILL_WINDOW_FREE) = old


def _same(a, b, path="") -> None:
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (path, sorted(a), sorted(b))
        for k in a:
            _same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    else:
        assert a == b, (path, a, b)


def _ids(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 96, (n,), generator=g).tolist()


def test_cached_window_is_the_chunk_major_prefill():
    ids = _ids(70)
    off, c0 = _run(0, ids)
    on, c1 = _run(48, ids)
    assert c0 == [] and c1 == [[0, 16, 32], [48, 64]], c1
    _same(off, on)
    assert len(on["anchors"]) >= 3 and len(on["syncs"]) == 5
    return "logits, KV, state, 4 anchors, snapshots, 5 syncs with their taps identical"


def test_the_live_views_layer_marks_change_nothing():
    """SRV-591: with the live view's info, a window marks each layer boundary (`window_live`, by
    `WINDOW_YIELD` as the server sets it) while it runs and drops the record after it; the logits,
    the rows and the state are the same."""
    ids = _ids(70, seed=4)
    off, _ = _run(0, ids)
    seen = []
    real = cache._layer_marks

    def spy(live, device):
        mark = real(live, device)

        def m():
            mark()
            seen.append((len(live["layers"]), live["k"], live["n"]))
        return m
    info = {}
    real_prefill, old_yield = cache.prefill, M.WINDOW_YIELD
    cache._layer_marks, M.WINDOW_YIELD = spy, cache.layer_mark
    cache.prefill = lambda *a, **k: real_prefill(*a, info=info, **k)
    try:
        on, calls = _run(48, ids)
    finally:
        cache._layer_marks, cache.prefill, M.WINDOW_YIELD = real, real_prefill, old_yield
    _same(off, on)
    n = T.tiny_config().num_hidden_layers
    # on the CPU a layer has run when it returns: each mark sees every layer issued so far
    assert seen == [(k, k, n) for _ in calls for k in range(1, n)], seen
    assert info.get("window_live") is None and info.get("window") is not None, info
    cache.layer_mark()                                  # between windows: nothing to mark
    return f"{len(calls)} windows, {n - 1} layer marks each, logits and state identical"


def test_free_window_is_the_chunk_major_prefill():
    ids = _ids(70, seed=2)
    off, _ = _run(0, ids, cached=False)
    on, c1 = _run(64, ids, cached=False)
    assert c1 == [[0, 16, 32, 48], [64]] or c1 == [[0, 16, 32, 48]], c1
    _same(off, on)
    return "no caches: identical"


def test_resume_off_grid_keeps_the_grid():
    ids = _ids(75, seed=3)
    off, _ = _run(0, ids, start=21)
    on, c1 = _run(48, ids, start=21)
    # the first prefill (21 rows) windows [0, 16] + 5; the second resumes at 20 (the anchor
    # below 21 is 16, so it reforwards from there) on the grid
    assert any(len(c) > 1 for c in c1), c1
    _same(off, on)
    return f"resumed prefill identical, windows {c1}"


def test_a_tap_that_cannot_say_runs_chunk_major():
    ids = _ids(50, seed=4)
    off, _ = _run(0, ids, cls=BlindTap)
    on, c1 = _run(48, ids, cls=BlindTap)
    assert c1 == [], c1
    _same(off, on)
    return "no window without tap_indices"


def test_window_plan_grid():
    class E:
        def window_ok(self, i):
            return True

        def prefill_window(self, *a):
            raise AssertionError

    e = E()
    assert cache._window_plan(e, 0, 100, 16, 48) == [(0, 16), (16, 16), (32, 16)]
    assert cache._window_plan(e, 5, 100, 16, 48) == [(5, 11), (16, 16), (32, 16)]
    assert cache._window_plan(e, 90, 100, 16, 48) is None          # one chunk left
    assert cache._window_plan(e, 0, 100, 16, 0) is None             # off
    assert cache._window_plan(e, 0, 100, 0, 48) is None             # chunk 0: one call
    assert cache._window_plan(e, 80, 100, 16, 48) == [(80, 16), (96, 4)]
    return "grid, resume, tail, off"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
