"""The session spill's codec and the NVMe write budget (0.4.3-test), on the random 4-layer model.

THE CLAIMS.
  1  byte planes: a tensor of any element size and layout (a narrowed, non-contiguous view of a
     buffer, as the spill copies one) goes to the host as byte planes and back bit for bit
  2  a conversation written compressed and read back is the conversation that was in the engine,
     byte for byte, exactly as one written uncompressed: the same rows, end anchor and anchors,
     and the next turn writes the same tokens; every new block of the compressed entry carries
     its frames, and none of the uncompressed one does
  3  mixed entries: an entry written uncompressed (0.4.2) and written again compressed keeps its
     unchanged blocks as they are, compresses the new ones, and reads back bit for bit
  4  a compressed block that does not hold what was written fails the restore (a frame that does
     not decode, or its sums), the entry goes, and the prefill is a cold one
  5  the write guard: under the day's budget everything is written; past it the tier and the
     spill's small or parked conversations are not, a drain and an eviction of 32k+ tokens still
     are; past 1.5x only the drain; the count survives a restart and starts again the next day
  6  the guard in place: past the budget the tier plans nothing and the spill refuses an
     eviction save (counted), and the drain still goes
  7  the settings document: memory.session_spill and memory.disk_writes are read over the
     command line's values

Run: python tests/test_spill_codec.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import cache, disk_cache, spill  # noqa: E402
from server import app  # noqa: E402
from server import diskcache as sdc  # noqa: E402
from test_cache import tokens  # noqa: E402
from test_end_anchor import CH, TOOL, _state, turn  # noqa: E402

HUGE = 10 ** 15
GB = disk_cache.GB


def _spill(d, res, codec="zstd", guard=None):
    t = disk_cache.DiskTier(os.path.join(d, "tier"), chunk=CH, fingerprint="tier", min_tokens=CH,
                            free_bytes=lambda p: HUGE, settings={"mode": "off"}).open()
    sp = spill.SessionSpill(os.path.join(d, "tier", "sessions"), chunk=CH, fingerprint="t",
                            tier=t, settings={"budget_gb": 1.0, "min_tokens": 64,
                                              "drain_min_tokens": CH, "codec": codec},
                            free_bytes=lambda p: HUGE, floor_bytes=0).open()
    sp.guard = guard
    t.guard = guard
    res.spiller = sp
    app.STATE["spill"] = sp
    return sp, t


def _rows(eng, n):
    return [t.narrow(ax, 0, n).clone() for t, ax in cache._kv_views(eng, None)]


def _wipe(eng):
    for t, ax in cache._kv_views(eng, None):
        t.fill_(0)


# ------------------------------------------------------------------ 1. byte planes

def test_byte_planes_round_trip_every_element_size_and_layout():
    g = torch.Generator().manual_seed(5)
    cases = []
    for dt in (torch.bfloat16, torch.float16, torch.float32, torch.uint8):
        base = (torch.randn(3, 2, 4, 40, 16, generator=g) * 7).to(dt) if dt != torch.uint8 else \
            torch.randint(0, 255, (3, 2, 4, 40, 16), generator=g, dtype=torch.uint8)
        cases.append((dt, base.narrow(3, 5, 17)))         # non-contiguous: rows along axis 3
    for dt, view in cases:
        nb = view.numel() * view.element_size()
        host = torch.zeros(nb, dtype=torch.uint8)
        spill._planes_to_host(view, host, False)
        w = view.element_size()
        raw = view.contiguous().view(torch.uint8).view(-1, w)
        for i in range(w):                               # plane i = byte i of every element
            assert torch.equal(host.view(w, -1)[i], raw[:, i]), (dt, i)
        out = torch.zeros(view.shape[0], view.shape[1], view.shape[2], 60, view.shape[4], dtype=dt)
        dst = out.narrow(3, 11, 17)
        spill._planes_from_host(host, dst, False)
        assert torch.equal(dst.contiguous().view(torch.uint8), view.contiguous().view(torch.uint8))
        assert int(out.narrow(3, 0, 11).abs().sum() if dt != torch.uint8
                   else out.narrow(3, 0, 11).sum()) == 0      # nothing outside the view written
    return f"{len(cases)} dtypes, non-contiguous views, bit for bit"


# ------------------------------------------------------------------ 2. compressed = uncompressed

def _save_and_restore(codec):
    d = tempfile.mkdtemp(prefix="qse-codec-")
    try:
        res = _state()
        sp, t = _spill(d, res, codec=codec)
        prompt = tokens(230, seed=81)
        out1, _, _ = turn(prompt, 40, session="A")
        eng = app.STATE["engine"]
        e = res.end["at"]
        rows0 = _rows(eng, e)
        end0 = tuple(x.clone() for x in res._end_bufs)
        anchors0 = {b: (S.clone(), c.clone()) for b, (S, c) in res.anchors.items()}
        info = sp.save_resident(res, eng, None, "evict")
        man = sp.entries[info["eid"]].man
        zs = [bl for bl in man["blocks"] if bl.get("z")]
        res.clear()
        _wipe(eng)
        got = sp.restore(res, eng, None, sp.entries[info["eid"]])
        assert got is not None and got["blocks_read"] == len(man["blocks"]), got
        for x, y in zip(_rows(eng, e), rows0):
            assert torch.equal(x, y)
        assert torch.equal(res._end_bufs[0], end0[0]) and torch.equal(res._end_bufs[1], end0[1])
        for b in sorted(anchors0)[-4:]:
            assert torch.equal(res.anchors[b][0], anchors0[b][0])
            assert torch.equal(res.anchors[b][1], anchors0[b][1])
        out2, pre2, _ = turn(prompt + out1 + TOOL, 16, session="A")
        assert pre2["reused"] == e, pre2
        t.close()
        return out2, len(zs), len(man["blocks"]), info
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_compressed_conversation_reads_back_bit_for_bit_as_an_uncompressed_one():
    out_raw, nz_raw, nb_raw, _ = _save_and_restore("off")
    out_z, nz, nb, info = _save_and_restore("zstd")
    assert nz_raw == 0 and nz == nb == nb_raw, (nz_raw, nz, nb)
    assert out_z == out_raw
    assert info.get("codec") == spill.CODEC_NAME and info["blocks_raw_bytes"] > 0, info
    return (f"{nb} blocks compressed, rows/end/anchors equal, next turn the same 16 tokens; "
            f"{info['blocks_raw_bytes']} -> {info['blocks_disk_bytes']} bytes (random weights)")


# ------------------------------------------------------------------ 3. mixed entries

def test_an_uncompressed_entry_written_again_compressed_keeps_its_blocks_and_reads_back():
    d = tempfile.mkdtemp(prefix="qse-codec-")
    try:
        res = _state()
        sp, t = _spill(d, res, codec="off")
        prompt = tokens(200, seed=82)
        out1, _, _ = turn(prompt, 30, session="A")
        eng = app.STATE["engine"]
        first = sp.save_resident(res, eng, None, "evict")
        out2, _, _ = turn(prompt + out1 + TOOL, 40, session="A")
        sp.apply({"codec": "zstd"})                       # the deploy of 0.4.3 over 0.4.2's entry
        second = sp.save_resident(res, eng, None, "evict")
        assert second["eid"] == first["eid"] and second["blocks_kept"] > 0, second
        man = sp.entries[second["eid"]].man
        raw = [bl for bl in man["blocks"] if not bl.get("z")]
        zs = [bl for bl in man["blocks"] if bl.get("z")]
        assert len(raw) == second["blocks_kept"] and len(zs) == second["blocks_written"], second
        e = res.end["at"]
        rows0 = _rows(eng, e)
        res.clear()
        _wipe(eng)
        got = sp.restore(res, eng, None, sp.entries[second["eid"]])
        assert got is not None and got["blocks_read"] == len(man["blocks"]), got
        for x, y in zip(_rows(eng, e), rows0):
            assert torch.equal(x, y)
        t.close()
        return f"{len(raw)} blocks kept as written, {len(zs)} compressed, read back bit for bit"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 4. a bad compressed block

def test_a_compressed_block_that_does_not_hold_what_was_written_fails_cold():
    for where in ("frame", "length"):
        d = tempfile.mkdtemp(prefix="qse-codec-")
        try:
            res = _state()
            sp, t = _spill(d, res, codec="zstd")
            prompt = tokens(200, seed=83)
            out1, _, _ = turn(prompt, 20, session="A")
            info = sp.save_resident(res, app.STATE["engine"], None, "evict")
            e = sp.entries[info["eid"]]
            blk = os.path.join(e.path, "r0000032.blk")
            with open(blk, "r+b") as f:
                if where == "frame":                     # a byte inside the first frame
                    f.seek(spill.ALIGN + 100)
                    b = f.read(1)
                    f.seek(spill.ALIGN + 100)
                    f.write(bytes([b[0] ^ 0xFF]))
                else:                                    # the frames cut short
                    f.truncate(spill.ALIGN + 64)
            res.clear()
            _, pre, _ = turn(prompt + out1 + TOOL, 4, session="A")
            assert pre["kind"] is None and pre["reused"] == 0, pre
            failed = sp.stats["failed"]
            assert (failed.get("sums", 0) + failed.get("read", 0)) == 1 and not sp.entries, failed
            t.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return "a changed byte and a cut file: the restore fails, the entry goes, the prefill is cold"


# ------------------------------------------------------------------ 5. the write guard

class _Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def test_the_write_guard_policy_and_its_day():
    d = tempfile.mkdtemp(prefix="qse-guard-")
    try:
        clk = _Clock(1_790_000_000.0)
        seen = []
        g = disk_cache.WriteGuard(d, budget_gb=1.0, clock=clk,
                                  on_event=lambda k, x: seen.append((k, x)))
        g.add(int(0.5 * GB), "tier")
        assert g.allow("tier", "prefix") and g.allow("spill", "evict_park", 1000)
        g.add(int(0.6 * GB), "spill")                    # 1.1 GB: over
        assert not g.allow("tier", "prefix")
        assert not g.allow("spill", "evict_park", 100_000)
        assert not g.allow("spill", "evict", 20_000)
        assert g.allow("spill", "evict", 40_000)
        assert g.allow("spill", "drain", 5_000)
        assert g.report()["level"] == "over"
        g.add(int(0.5 * GB), "spill")                    # 1.6 GB: past 1.5x
        assert not g.allow("spill", "evict", 200_000) and g.allow("spill", "drain", 5_000)
        rep = g.report()
        assert rep["level"] == "hard" and rep["today_gb"] == 1.6, rep
        assert rep["skipped_today"] == {"tier:prefix": 1, "spill:evict_park": 1,
                                        "spill:evict": 2}, rep["skipped_today"]
        assert len(seen) == 4 and all(k == "skip" for k, _ in seen)
        g.save()
        g2 = disk_cache.WriteGuard(d, budget_gb=1.0, clock=clk)      # a restart
        assert g2.today() == g.today() and not g2.allow("tier", "prefix")
        clk.t += 86400                                                # the next day
        assert g2.today() == 0 and g2.allow("tier", "prefix")
        assert len(g2.report()["per_day_gb"]) == 2
        g3 = disk_cache.WriteGuard(d, budget_gb=0, clock=clk)        # 0 = no budget
        g3.add(int(5 * GB), "tier")
        assert g3.allow("tier", "prefix")
        return "ok/over/hard lines, skips counted, the day survives a restart and rolls over"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 6. the guard in place

def test_past_the_budget_the_tier_plans_nothing_and_the_spill_keeps_only_the_drain():
    d = tempfile.mkdtemp(prefix="qse-guard-")
    try:
        g = disk_cache.WriteGuard(os.path.join(d, "g"), budget_gb=1.0)
        os.makedirs(os.path.join(d, "g"), exist_ok=True)
        g.add(int(1.2 * GB), "tier")
        res = _state()
        sp, t = _spill(d, res, guard=g)
        prompt = tokens(200, seed=84)
        turn(prompt, 20, session="A")
        eng = app.STATE["engine"]
        assert sp.save_resident(res, eng, None, "evict") is None      # 220 tokens: small
        assert sp.stats["refused"].get("write_budget") == 1, sp.stats["refused"]
        info = sp.save_resident(res, eng, None, "drain")
        assert info is not None and info["blocks_written"] > 0
        assert g.days[g._day()]["spill"] == info["bytes_written"] > 0, (g.days, info)
        # the tier: on, and over the budget, it plans nothing
        tier = disk_cache.DiskTier(os.path.join(d, "tier2"), chunk=CH, fingerprint="x",
                                   min_tokens=CH, free_bytes=lambda p: HUGE,
                                   settings={"mode": "on", "admit_after": 1}).open()
        tier.guard = g
        keys = tokens(4 * CH, seed=85)
        hashes = cache.prefix_hashes(keys)
        job = tier.plan(keys, hashes, 0, len(keys), len(keys), "prefix", session="s1")
        assert job is None and tier.stats["dropped"].get("write_budget") == 1, tier.stats
        tier.guard = None
        job2 = tier.plan(keys, hashes, 0, len(keys), len(keys), "prefix", session="s2")
        assert job2 is not None                          # the same plan without the guard
        tier.close()
        t.close()
        return "eviction refused and counted, drain written, the tier's plan dropped"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 7. the settings document

def test_the_settings_document_overrides_the_command_line():
    class A:
        disk_write_budget_gb = 200.0
    doc = {"memory": {"session_spill": {"budget_gb": 64, "codec": "off", "bogus": 1},
                      "disk_writes": {"budget_gb": 150}}}
    assert sdc.spill_of(doc) == {"budget_gb": 64, "codec": "off"}
    assert sdc.spill_of({}) == {}
    assert sdc.writes_of(doc, A())["budget_gb"] == 150
    assert sdc.writes_of({}, A())["budget_gb"] == 200.0
    assert sdc.writes_of({}, None)["budget_gb"] is None
    assert disk_cache.SETTINGS["budget_gb"] == 16 and spill.SETTINGS["budget_gb"] == 64.0
    return "memory.session_spill and memory.disk_writes over the flags; defaults 16 / 64 GB"


def _main():
    import inspect
    fails = 0
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__], inspect.isfunction),
                           key=lambda x: x[1].__code__.co_firstlineno):
        if not name.startswith("test_"):
            continue
        try:
            msg = fn()
            print(f"PASS {name}" + (f": {msg}" if msg else ""))
        except Exception:                                              # noqa: BLE001
            import traceback
            fails += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main())
