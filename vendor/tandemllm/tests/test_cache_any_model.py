"""ENG-157: the cache tiers know no model -- they keep a sequence through the engine's adapter.

A third architecture made up here (`Toy`: two recurrent sections of different dtypes and three
position-indexed views of different shapes, dtypes and token axes) runs the tiers with no edit in
engine/cache.py, disk_cache.py or spill.py:

  * a snapshot (`capture` / `restore`) round-trips every section and every view;
  * the resident prefix anchors on the grid and resumes the next turn there, equal to a cold run;
  * the disk tier writes its anchor (format 2, the state's section names) and chunks, and after a
    restart restores them into a fresh engine, equal to a cold run;
  * the session spill saves the conversation and reads it back in place, equal to before;
and the 27B's anchor files written before the sections were named (format 1, "S" and "conv") are
still read by the new code, bit for bit.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine import cache, disk_cache, spill  # noqa: E402
from engine.models.adapter import CostModel  # noqa: E402
from engine.models.state import SectionState  # noqa: E402
from test_cache import tokens  # noqa: E402

CH = 16
BIG = 1 << 30


class _ToyState(SectionState):
    def __init__(self, eng):
        self.eng = eng

    def sections(self):
        e = self.eng
        return {"toy.sum": e.s1, "toy.count": e.s2, **{f"toy.layer.{k}": x for k, x in enumerate(e.xs)}}

    def views(self):
        e = self.eng
        return [("a", e.va, 1), ("b", e.vb, 0), ("c", e.vc, 2), *((f"x{k}", x, 0) for k, x in enumerate(e.xv))]


class Toy:
    """A recurrent model of no known family: each token folds into a fp32 sum and an int64 count;
    its rows (three views) depend on the token and the state before it, so a row and the state
    after a position depend only on the tokens up to it."""

    def __init__(self, max_len: int = 512, wide: int = 0):
        # `wide`: that many more sections and views (a model of many layers: its file headers run
        # past one line)
        self.xs = [torch.zeros(2, 3) for _ in range(wide)]
        self.xv = [torch.zeros(max_len, 2) for _ in range(wide // 2)]
        self.s1 = torch.zeros(3, 5)
        self.s2 = torch.zeros(2, dtype=torch.long)
        self.va = torch.zeros(2, max_len, 4, dtype=torch.bfloat16)
        self.vb = torch.zeros(max_len, 3)
        self.vc = torch.zeros(1, 2, max_len, dtype=torch.float16)
        self.kv = types.SimpleNamespace(length=0, max_len=max_len, pages=None)
        self.cfg = types.SimpleNamespace(vocab_size=128)
        self.hidden_post_norm = None

    # the adapter's cold verbs
    def cache_state(self):
        return _ToyState(self)

    def cost(self) -> CostModel:
        sec = sum(t.numel() * t.element_size() for t in self.cache_state().sections().values())
        row = sum(t.numel() // t.shape[ax] * t.element_size() for _, t, ax in self.cache_state().views())
        return CostModel(section_bytes=sec, view_bytes_per_token=row, parts={"recurrent": sec})

    def settle(self):
        pass

    def discard_pending(self):
        pass

    def trim_pending(self, n):
        return n == self.kv.length

    def resume_at(self, n):
        self.kv.length = n

    def reset(self):
        self.s1.zero_()
        self.s2.zero_()
        for x in self.xs:
            x.zero_()
        self.kv.length = 0

    def window_ok(self, i):
        return False

    @torch.no_grad()
    def forward(self, ids, start, last_only=True):
        assert start == self.kv.length
        ids = ids.view(-1).tolist()
        rows = []
        for j, t in enumerate(ids):
            p = start + j
            self.va[:, p] = (self.s1[:2, :4] * 0.25 + t).to(torch.bfloat16)
            self.vb[p] = self.s1[0, :3] - t
            self.vc[0, :, p] = torch.tensor([t, int(self.s2[0]) % 97], dtype=torch.float16)
            for k, x in enumerate(self.xv):
                x[p] = self.s1[1, :2] + k
            self.s1.mul_(0.5).add_(torch.arange(15.0).view(3, 5) * (t % 7))
            self.s2 += torch.tensor([1, t])
            for k, x in enumerate(self.xs):
                x.add_(self.s1[:2, :3] * (k + 1))
            rows.append(self.s1.flatten()[:8].clone())
        self.kv.length = start + len(ids)
        self.hidden_post_norm = torch.stack(rows).unsqueeze(0)
        lg = torch.zeros(1, 1, self.cfg.vocab_size)
        lg[0, 0, int(self.s2[1]) % self.cfg.vocab_size] = 1.0 + float(self.s1.sum())
        return lg


def _cold(ids, wide=0):
    eng = Toy(wide=wide)
    lg, _, _ = cache.prefill(eng, None, ids, "cpu", chunk=CH)
    return eng, lg


def _same(eng, ref, n, lg=None, ref_lg=None):
    assert eng.kv.length == ref.kv.length == n
    for a, b in zip(eng.cache_state().sections().values(), ref.cache_state().sections().values()):
        assert torch.equal(a, b)
    for (_, a, ax), (_, b, _) in zip(eng.cache_state().views(), ref.cache_state().views()):
        assert torch.equal(a.narrow(ax, 0, n), b.narrow(ax, 0, n))
    if lg is not None:
        assert torch.equal(lg, ref_lg)


def test_a_snapshot_round_trips_every_section_and_view():
    eng = Toy()
    ids = tokens(70, seed=1)
    cache.prefill(eng, None, ids, "cpu", chunk=CH)
    snap = cache.capture(eng)
    ref = Toy()
    cache.prefill(ref, None, ids, "cpu", chunk=CH)
    eng.forward(torch.tensor(tokens(9, seed=2)), start=70)
    cache.restore(eng, snap)
    _same(eng, ref, 70)
    assert snap.nbytes == eng.cost().section_bytes + 70 * eng.cost().view_bytes_per_token
    return f"2 sections, 3 views, {snap.nbytes} bytes at 70 tokens"


def test_the_resident_prefix_resumes_a_new_model_exactly():
    eng, res = Toy(), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG)
    turn1 = tokens(100, seed=3)
    cache.prefill(eng, None, turn1, "cpu", chunk=CH, resident=res)
    assert sorted(res.anchors)[-1] == 96 and len(res.anchors[96]) == 2
    turn2 = turn1 + tokens(30, seed=4)
    lg, reused, _ = cache.prefill(eng, None, turn2, "cpu", chunk=CH, resident=res)
    assert reused == 96, reused
    ref, ref_lg = _cold(turn2)
    _same(eng, ref, len(turn2), lg, ref_lg)
    return "anchors of 2 sections on the grid; turn 2 resumed at 96, equal to a cold run"


class _LandingKV:
    """A length whose read lands a pending block first, as Flash-Next's does (its chain commit): the
    rows it writes are the guest's last block."""

    def __init__(self, eng, max_len):
        self.eng, self._len, self.max_len, self.pages, self.pending = eng, 0, max_len, None, None

    @property
    def length(self):
        if self.pending is not None:
            p, n = self.pending
            self.pending = None
            self.eng.va[:, p:p + n] = 9.0                    # the block's rows, written now
            self._len = p + n
        return self._len

    @length.setter
    def length(self, v):
        self._len = int(v)


def test_a_guest_whose_last_block_lands_late_leaves_the_resident_rows_intact():
    """2026-10-12 (P0 follow-up): settle_guest put the resident's rows back and then read the engine's
    length; on an engine whose length lands a pending block when read (Flash-Next), that block's rows
    went over the restored ones, and the next turn answered differently from a cold prefill. The length
    is read before the rows go back."""
    eng, res = Toy(), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG)
    eng.kv = _LandingKV(eng, 512)
    turn1 = tokens(100, seed=3)
    cache.prefill(eng, None, turn1, "cpu", chunk=CH, resident=res)
    guest = tokens(20, seed=9)
    cache.prefill(eng, None, guest, "cpu", chunk=CH, resident=res)
    assert res._stash is not None                            # a guest: the resident's rows were stashed
    eng.kv.pending = (20, 4)                                 # its last block, not landed yet
    turn2 = turn1 + tokens(30, seed=4)
    lg, reused, _ = cache.prefill(eng, None, turn2, "cpu", chunk=CH, resident=res)
    assert reused == 96, reused
    ref, ref_lg = _cold(turn2)
    _same(eng, ref, len(turn2), lg, ref_lg)
    return "the guest's late block landed before the restore; turn 2 equal to a cold run"


def _tier(d):
    return disk_cache.DiskTier(d, chunk=CH, fingerprint="toy", settings={"admit_after": 1, "budget_gb": 1},
                               free_bytes=lambda p: 10 ** 15, head_anchors=2, min_tokens=CH).open()


def _disk_round_trip(wide):
    ids = tokens(150, seed=5)
    with tempfile.TemporaryDirectory() as d:
        t = _tier(d)
        cache.prefill(Toy(wide=wide), None, ids, "cpu", chunk=CH, disk=t)
        assert t.flush()
        t.close()
        anc = os.listdir(os.path.join(d, "anchors"))
        assert len(anc) == 1
        meta = t._read_header(os.path.join(d, "anchors", anc[0]), "anchor")
        names = [s["name"] for s in meta["sections"]]
        assert meta["format"] == 2 and names[:2] == ["toy.sum", "toy.count"] and names[-1] == "chain"
        chunk = t._read_header(os.path.join(d, "chunks", os.listdir(os.path.join(d, "chunks"))[0]), "chunk")
        t2 = _tier(d)                                  # a restart
        prompt = ids + tokens(20, seed=6)
        eng = Toy(wide=wide)
        info = {}
        lg, reused, _ = cache.prefill(eng, None, prompt, "cpu", chunk=CH, disk=t2, info=info)
        assert (info.get("kind"), reused) == ("disk", 144), (info, reused)
        ref, ref_lg = _cold(prompt, wide)
        _same(eng, ref, len(prompt), lg, ref_lg)
        t2.close()
        return meta["_hdr"], chunk["_hdr"]


def test_the_disk_tier_round_trips_a_new_model_after_a_restart():
    h = _disk_round_trip(0)
    hw = _disk_round_trip(120)
    assert h[0] == disk_cache.ALIGN and hw[0] > disk_cache.ALIGN and hw[1] > h[1], (h, hw)
    return (f"anchor format 2 with the state's names; restored after a restart, equal to a cold run; "
            f"122 sections and 63 views: headers of {hw[0]} / {hw[1]} bytes")


def _spill_round_trip(wide, codec="off"):
    d = tempfile.mkdtemp(prefix="toy-spill-")
    try:
        t = _tier(d)
        sp = spill.SessionSpill(os.path.join(d, "sessions"), chunk=CH, fingerprint="toy", tier=t,
                                settings={"budget_gb": 1.0, "min_tokens": 64, "drain_min_tokens": CH, "codec": codec},
                                free_bytes=lambda p: 10 ** 15, floor_bytes=0).open()
        eng, res = Toy(wide=wide), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True)
        ids = tokens(130, seed=7)
        cache.prefill(eng, None, ids, "cpu", chunk=CH, resident=res)
        ref = Toy(wide=wide)
        cache.prefill(ref, None, ids, "cpu", chunk=CH)
        anchors0 = {b: tuple(x.clone() for x in a) for b, a in res.anchors.items()}
        info = sp.save_resident(res, eng, None, "evict")
        assert info is not None, sp.stats
        res.clear()
        for _, v, _ in eng.cache_state().views():
            v.zero_()
        got = sp.restore(res, eng, None, sp.entries[info["eid"]])
        assert got is not None, sp.stats
        for (_, a, ax), (_, b, _) in zip(eng.cache_state().views(), ref.cache_state().views()):
            assert torch.equal(a.narrow(ax, 0, 130), b.narrow(ax, 0, 130))
        for b in sorted(anchors0)[-4:]:
            assert all(torch.equal(x, y) for x, y in zip(res.anchors[b], anchors0[b]))
        man = sp.entries[info["eid"]].man
        t.close()
        return max(int(bl.get("hdr", disk_cache.ALIGN)) for bl in man["blocks"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_session_spill_round_trips_a_new_model():
    assert _spill_round_trip(0) == disk_cache.ALIGN
    h = _spill_round_trip(300)
    assert h > disk_cache.ALIGN, h
    _spill_round_trip(300, codec="zstd")                # compressed blocks with the long header
    return f"rows and the newest anchors back in place, bit for bit; 302 sections, 153 views: block headers {h} bytes"


def test_a_drafter_resumes_over_committed_rows_and_again_over_the_restored_ones():
    """rc-0.8.0 live (2026-10-12): the spill's restore at startup told Flash-Next's MTP drafter its
    cache was valid before any row was read back; its resume pools indexer keys from those rows, which
    were pages the KV pool had released, and the GPU faulted. The rows a drafter reads are committed
    before it resumes (`cache.drafter_resume`), and it resumes again once the rows are in, so what it
    builds is the restored rows' -- for the spill, the disk tier and a side slot's carry alike."""
    calls = []
    real = cache._kvp.ensure_for

    class Drafter:
        def __init__(self, eng):
            self.eng = eng

        def kv_views(self):
            return []

        def state_resume(self, n):
            # what a resume reads: the restored rows of view `b` are there or not yet
            calls.append(("resume", int(n), bool(self.eng.vb[:n].abs().sum() > 0)))

    d = tempfile.mkdtemp(prefix="toy-resume-")
    try:
        cache._kvp.ensure_for = lambda t, rows: (calls.append(("ensure", int(rows))), real(t, rows))[1]
        t = _tier(d)
        sp = spill.SessionSpill(os.path.join(d, "sessions"), chunk=CH, fingerprint="toy", tier=t,
                                settings={"budget_gb": 1.0, "min_tokens": 64, "drain_min_tokens": CH},
                                free_bytes=lambda p: 10 ** 15, floor_bytes=0).open()
        eng, res = Toy(), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True)
        cache.prefill(eng, None, tokens(130, seed=7), "cpu", chunk=CH, resident=res)
        info = sp.save_resident(res, eng, None, "evict")
        res.clear()
        for _, v, _ in eng.cache_state().views():
            v.zero_()
        calls.clear()
        assert sp.restore(res, eng, Drafter(eng), sp.entries[info["eid"]]) is not None, sp.stats
        resumes = [c for c in calls if c[0] == "resume"]
        first = calls.index(resumes[0])
        assert ("ensure", 130) in calls[:first], calls[:first + 1]
        assert resumes[0][2] is False and resumes[-1] == ("resume", 130, True), resumes
        t.close()
    finally:
        cache._kvp.ensure_for = real
        shutil.rmtree(d, ignore_errors=True)
    return "the rows committed before the drafter resumes; it resumes again over the rows read back"


def test_a_prewarm_yields_between_blocks_and_keeps_what_it_did_not_finish():
    """SRV-349 (rc-0.8.0): a request that arrives during a prewarm waits at most one block. The
    restore stops between two blocks when `stop()` says so: the entry stays on the NVMe, the rows
    read so far are cut back, and a later restore reads it whole. The save before it (the resident
    conversation written first) stops the same way: its incomplete entry goes, the conversation in
    RAM stays as it was."""
    d = tempfile.mkdtemp(prefix="toy-yield-")
    try:
        t = _tier(d)
        sp = spill.SessionSpill(os.path.join(d, "sessions"), chunk=CH, fingerprint="toy", tier=t,
                                settings={"budget_gb": 1.0, "min_tokens": 64, "drain_min_tokens": CH},
                                free_bytes=lambda p: 10 ** 15, floor_bytes=0).open()
        eng, res = Toy(), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True)
        ids = tokens(130, seed=7)
        cache.prefill(eng, None, ids, "cpu", chunk=CH, resident=res)
        info = sp.save_resident(res, eng, None, "evict")
        e = sp.entries[info["eid"]]
        res.clear()
        for _, v, _ in eng.cache_state().views():
            v.zero_()

        def after(k):
            n = [0]

            def stop():
                n[0] += 1
                return n[0] > k
            return stop
        assert sp.restore(res, eng, None, e, stop=after(2)) is None
        assert info["eid"] in sp.entries and sp.stats["failed"].get("yielded") == 1, sp.stats["failed"]
        assert res.valid < 130, res.valid
        got = sp.restore(res, eng, None, e)
        assert got is not None and res.valid == 130
        ref = Toy()
        cache.prefill(ref, None, ids, "cpu", chunk=CH)
        for (_, a, ax), (_, b, _) in zip(eng.cache_state().views(), ref.cache_state().views()):
            assert torch.equal(a.narrow(ax, 0, 130), b.narrow(ax, 0, 130))
        # the save: a new conversation in RAM, its write stopped after one block
        res.owner = "B"
        assert sp.save_resident(res, eng, None, "switch", stop=after(1)) is None
        assert sp.stats["refused"].get("yielded") == 1 and res.valid == 130
        assert len(sp.entries) == 1, "the incomplete entry went; the first one stays"
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return "restore stopped after 2 blocks (entry kept, rows cut), read whole later; a save stopped after 1"


def test_a_27b_anchor_file_of_format_1_is_still_read():
    from test_cache import fresh
    from test_disk_cache import Pos, exact, run, tier
    ids = tokens(150, seed=8)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos())
        t.close()
        p = os.path.join(d, "anchors", os.listdir(os.path.join(d, "anchors"))[0])
        meta = t._read_header(p, "anchor")
        meta.pop("tokens", None)
        # what the 27B wrote before ENG-157: format 1, its two sections named "S" and "conv"
        meta["format"] = 1
        for s in meta["sections"]:
            s["name"] = {"gdn.S": "S", "gdn.conv": "conv"}.get(s["name"], s["name"])
        raw = json.dumps(meta, separators=(",", ":")).encode()
        with open(p, "r+b") as f:
            f.seek(8)
            f.write(len(raw).to_bytes(4, "little") + raw + b"\0" * (disk_cache.ALIGN - 12 - len(raw)))
        t2 = tier(d, settings={"admit_after": 1})
        assert t2.report()["entries"] == 1
        prompt = ids + tokens(20, seed=9)
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, prompt, t2, dr)
        assert (kind, reused) == ("disk", 144), (kind, reused)
        exact(eng, lg, prompt, dr)
        t2.close()
    return "a format-1 anchor (S, conv) restored by the new code, equal to a cold prefill"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<62} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<62} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
