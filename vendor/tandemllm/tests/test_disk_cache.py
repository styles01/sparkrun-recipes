"""The disk tier (engine/disk_cache.py) on the random 4-layer model.

THE CLAIM is the one tests/test_cache.py and tests/test_resident.py make for the caches in RAM: a
request that resumes from state read back off the disk computes the same logits, S, conv, KV rows
and drafter rows -- to the last bit -- as a cold prefill of its whole prompt on the same chunk
grid. Every restore below is checked against a cold prefill that way, including the ones that go
through a reopened tier (the files are the index: a restart) and the ones that fail part way.

The properties:

  * what is written: a prefix prefilled `admit_after` times, a session's first long request at
    the lines below its first user message, the line where an earlier prompt parted from this
    one inside the head two sessions share (a branch), nothing for a session snapshot, nothing
    with the tier off; the counts outlive a new fingerprint, the files do not;
  * the hash is a hint: a chunk whose stored tokens differ from the prompt is never restored;
  * crash safety: a `.tmp` is never read, a checksum that fails costs the entry and nothing else,
    a file of another fingerprint is swept;
  * write-behind: the next prefill closes a job's window, and what was not copied by then is
    dropped and journaled, never half-written; a resumed extension leaves the window open;
  * the disk's budget, its TTL, the free-space floor and least-valuable-first eviction;
  * the lookup order: the caches in RAM first, the disk only where it starts later; a disk hit
    becomes the resident prompt's anchor.

Run: python tests/test_disk_cache.py
"""

from __future__ import annotations

import os
import sys
import tempfile

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import cache, disk_cache  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402
from test_resident import same_state  # noqa: E402

CH = 16
BIG = 1 << 30
HUGE = 10 ** 15                         # a fake disk with all the room in the world


class Pos:
    """A drafter with a position-indexed cache that resumes in place, as DFlash2 does."""

    def __init__(self):
        self.buf = torch.zeros(512, 4)
        self.ctx_len = 0

    def reset(self):
        self.ctx_len = 0

    def sync(self, toks, hidden, first_pos, rows=None):
        n = len(toks)
        self.buf[first_pos:first_pos + n] = hidden[:n, :4].float()
        self.ctx_len = first_pos + n

    def state_resume(self, n):
        self.ctx_len = n

    def kv_views(self):
        return [(self.buf, 0)]


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def tier(d, events=None, **kw):
    settings = {"admit_after": 2, "budget_gb": 1, "ttl_days": 7}
    settings.update(kw.pop("settings", {}))
    kw.setdefault("free_bytes", lambda p: HUGE)
    kw.setdefault("head_anchors", 2)
    kw.setdefault("min_tokens", CH)
    on_event = (lambda k, det: events.append((k, det))) if events is not None else None
    return disk_cache.DiskTier(d, chunk=CH, fingerprint=kw.pop("fp", "test"), settings=settings,
                               on_event=on_event, **kw).open()


def run(eng, ids, t=None, drafter=None, res=None, **kw):
    info = {}
    lg, reused, fwd = cache.prefill(eng, drafter, ids, DEV, chunk=CH, disk=t, resident=res,
                                    info=info, **kw)
    if t is not None:
        assert t.flush()
    return lg, reused, info.get("kind")


def cold(ids, drafter=None):
    eng = fresh()
    lg, _, _ = cache.prefill(eng, drafter, ids, DEV, store=None, chunk=CH)
    return eng, lg


def exact(eng, lg, ids, d=None):
    """`eng` (and drafter `d`) after a prefill of `ids` holds what a cold prefill holds."""
    ref_d = Pos() if d is not None else None
    ref, ref_lg = cold(ids, ref_d)
    assert torch.equal(lg, ref_lg), (lg - ref_lg).abs().max().item()
    same_state(eng, ref, len(ids))
    if d is not None:
        assert d.ctx_len == len(ids)
        assert torch.equal(d.buf[:len(ids)], ref_d.buf[:len(ids)])


def files(d, sub):
    return sorted(f for f in os.listdir(os.path.join(d, sub)))


def resident(**kw):
    kw.setdefault("stash_bytes", BIG)
    return cache.ResidentPrefix(kw.pop("budget", BIG), CH, **kw)


# ------------------------------------------------------------------ 0. the files

def test_a_file_is_written_whole_or_not_at_all_and_reads_back():
    with tempfile.TemporaryDirectory() as d:
        a, b = disk_cache._Buf(8192, cuda=False), disk_cache._Buf(4096, cuda=False)
        a.view(0, 8192)[:] = bytes(range(256)) * 32
        b.view(0, 4096)[:] = b"\x07" * 4096
        p = os.path.join(d, "f")
        disk_cache.write_file(p, [a.view(0, 8192), b.view(0, 4096)])
        assert sorted(os.listdir(d)) == ["f"]                 # no .tmp left behind
        assert os.stat(p).st_mode & 0o777 == 0o600
        c = disk_cache._Buf(4096, cuda=False)
        disk_cache.read_into(p, 8192, c.view(0, 4096))
        assert bytes(c.view(0, 4096)) == b"\x07" * 4096
        for kind in ("crc32", "sum64"):
            x = disk_cache.checksum(a.view(0, 8192), kind)
            assert x == disk_cache.checksum(a.view(0, 8192), kind)
            a.view(0, 8192)[5000] ^= 1
            assert x != disk_cache.checksum(a.view(0, 8192), kind)
            a.view(0, 8192)[5000] ^= 1


# ------------------------------------------------------------------ 1. the claim

def test_a_disk_restore_after_a_restart_equals_a_cold_prefill_bit_for_bit():
    ids = tokens(150, seed=1)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos())
        t.close()
        assert len(files(d, "anchors")) == 1 and len(files(d, "chunks")) == 9   # line 144
        t2 = tier(d, settings={"admit_after": 1})       # a restart: the files are the index
        assert t2.report()["entries"] == 1
        prompt = ids + tokens(20, seed=2)
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, prompt, t2, dr)
        assert (kind, reused) == ("disk", 144)
        exact(eng, lg, prompt, dr)
        rep = t2.report()
        assert rep["hits"] == 1 and rep["tokens_restored"] == 144 and rep["top"][0]["hits"] == 1
        t2.close()


def test_a_disk_hit_becomes_the_resident_prompt_and_the_next_turn_resumes_in_place():
    ids = tokens(150, seed=3)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos())
        eng, dr, res = fresh(), Pos(), resident()
        turn1 = ids + tokens(10, seed=4)
        lg, reused, kind = run(eng, turn1, t, dr, res)
        assert (kind, reused) == ("disk", 144) and 144 in res.anchors
        exact(eng, lg, turn1, dr)
        turn2 = turn1 + tokens(40, seed=5)
        lg, reused, kind = run(eng, turn2, t, dr, res)
        assert kind == "resident" and reused == 144, (kind, reused)
        exact(eng, lg, turn2, dr)
        t.close()


def test_the_caches_in_ram_come_first():
    """The rows in place and their anchors cost nothing; the disk is read only past them."""
    ids = tokens(150, seed=6)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos())
        eng, dr, res = fresh(), Pos(), resident()
        run(eng, ids, None, dr, res)                          # the rows in place: the same prompt
        hits = t.stats["hits"]
        lg, reused, kind = run(eng, ids + [5, 6, 7], t, dr, res)
        assert kind == "resident" and reused == 144 and t.stats["hits"] == hits
        exact(eng, lg, ids + [5, 6, 7], dr)
        t.close()


def test_rows_in_place_are_not_read_again_and_a_short_saving_is_declined():
    ids = tokens(200, seed=7)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos())                           # anchor at 192
        eng, dr, res = fresh(), Pos(), resident(budget=1)     # anchors evicted down to the tail
        run(eng, ids[:100], None, dr, res)
        assert max(res.anchors) == 96
        read = t.stats["bytes_read"]
        lg, reused, kind = run(eng, ids + [9], t, dr, res)
        assert (kind, reused) == ("disk", 192)
        # the chunks above 96 and the anchor; not the six below it the rows in place hold
        chunk = os.path.getsize(os.path.join(d, "chunks", files(d, "chunks")[0]))
        anchor = os.path.getsize(os.path.join(d, "anchors", files(d, "anchors")[0]))
        assert t.stats["bytes_read"] - read <= 6 * chunk + anchor, (t.stats["bytes_read"] - read)
        exact(eng, lg, ids + [9], dr)
        t.read_bps = 1.0                                       # a disk that reads a byte a second
        eng2, dr2 = fresh(), Pos()
        lg, reused, kind = run(eng2, ids + [9], t, dr2)
        assert kind is None and reused == 0 and t.stats["declined_slow"] >= 1
        t.close()


# ------------------------------------------------------------------ 2. what goes to disk

def test_a_prefix_is_written_after_it_was_prefilled_twice():
    ids = tokens(120, seed=8)
    with tempfile.TemporaryDirectory() as d:
        events = []
        t = tier(d, events=events)
        run(fresh(), ids, t, Pos())
        assert files(d, "anchors") == [] and files(d, "chunks") == []
        run(fresh(), ids, t, Pos())                           # the second cold prefill
        assert len(files(d, "anchors")) == 1                  # line 112
        assert len(files(d, "chunks")) == 7
        assert any(k == "write" and det["tokens"] == 112 for k, det in events)
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, ids + [1, 2], t, dr)
        assert (kind, reused) == ("disk", 112)
        assert any(k == "hit" for k, _ in events)
        exact(eng, lg, ids + [1, 2], dr)
        # a resumed line is not a prefilled line: the hit counted nothing new
        assert t.counts[(112, cache.prefix_hashes(ids)[112])] == 2
        t.close()


def test_the_first_request_of_a_session_is_written_where_its_user_message_starts():
    head = tokens(100, seed=9)                                # the system text and the tools
    s1 = head + tokens(30, seed=10)
    s2 = head + tokens(45, seed=11)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, preview=lambda keys, L, kind, spans: f"{kind} {len(keys)}")
        tag = {"client": "opencode", "head_end": 100}
        run(fresh(), s1, t, Pos(), conv_id="ses_1", disk_tag=tag)
        rep = t.report(previews=True)
        assert sorted(r["tokens"] for r in rep["top"]) == [80, 96]     # the two lines <= 100
        assert {r["kind"] for r in rep["top"]} == {"opencode-head"}
        assert rep["top"][0]["preview"] == "opencode-head 130"
        assert t.report()["top"][0]["preview"] is None        # never without asking for them
        # the session's second request, cold again: no head of its own, but its lines were
        # prefilled twice now, and the longest is a repeated prefix (no preview: it is the user's)
        run(fresh(), s1 + [3], t, Pos(), conv_id="ses_1", disk_tag=tag)
        rows = {r["tokens"]: r for r in t.report(previews=True)["top"]}
        assert sorted(rows) == [80, 96, 128] and rows[128]["kind"] == "prefix"
        assert rows[128]["preview"] is None
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, s2, t, dr, conv_id="ses_2", disk_tag=tag)
        assert (kind, reused) == ("disk", 96)
        exact(eng, lg, s2, dr)
        t.close()


def test_the_counts_outlive_a_new_fingerprint_and_the_files_do_not():
    """A deploy that changed the engine's code sweeps every file; the counts are traffic, not
    rows, so the first cold read after it is the second one and is written."""
    x = tokens(120, seed=40)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d)
        run(fresh(), x, t, Pos())                             # counted once, nothing written
        t.close()
        t2 = tier(d, fp="the next deploy")
        assert t2.counts[(112, cache.prefix_hashes(x)[112])] == 1
        run(fresh(), x, t2, Pos())
        assert len(files(d, "anchors")) == 1
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, x + [1, 2], t2, dr)
        assert (kind, reused) == ("disk", 112)
        exact(eng, lg, x + [1, 2], dr)
        t2.close()
        t3 = tier(d, fp="and the one after")                  # the files of t2 are swept again
        assert t3.stats["swept_fingerprint"] == 8 and t3.report()["entries"] == 0
        assert t3.counts[(112, cache.prefix_hashes(x)[112])] == 2
        assert t3.report()["hits"] == 0                       # the hits were the files'
        t3.close()
        # another grid: the counts' lines name other prefixes there
        t4 = disk_cache.DiskTier(d, chunk=2 * CH, fingerprint="and the one after",
                                 settings={"admit_after": 2}, free_bytes=lambda p: HUGE,
                                 min_tokens=2 * CH).open()
        assert not t4.counts
        t4.close()


def test_a_session_in_another_folder_resumes_where_the_heads_part():
    """2026-10-03: two opencode sessions in two folders share the tools and opencode's fixed text
    (13,954 of 25,616 tokens) and part at its <env> block; the head anchors sit near the user
    message, past that line. After a deploy (the files swept, the counts kept) the first session
    is in a folder seen before: its longest repeated line is its own folder's, and the line where
    the other folders part is a branch. The next session in a new folder resumes there."""
    static = tokens(70, seed=41)                             # the tools and the fixed text
    def ses(env_seed, user_seed):
        head = static + tokens(40, seed=env_seed)             # the <env> block onwards
        ids = head + tokens(20, seed=user_seed)
        return ids, {"client": "opencode", "head_end": len(head), "first_answer": len(ids)}
    x1, tx1 = ses(42, 45)
    y1, ty1 = ses(43, 46)
    x2, tx2 = ses(42, 47)
    z1, tz1 = ses(44, 48)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d)
        run(fresh(), x1, t, Pos(), conv_id="ses_x1", disk_tag=tx1)
        run(fresh(), y1, t, Pos(), conv_id="ses_y1", disk_tag=ty1)
        t.close()
        t2 = tier(d, fp="the next deploy")
        assert t2.report()["entries"] == 0
        _, reused, _ = run(fresh(), x2, t2, Pos(), conv_id="ses_x2", disk_tag=tx2)
        assert reused == 0
        lines = sorted(r["tokens"] for r in t2.report()["top"])
        assert lines == [64, 80, 96], lines                    # the branch, the head and repeated
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, z1, t2, dr, conv_id="ses_z1", disk_tag=tz1)
        assert (kind, reused) == ("disk", 64)
        exact(eng, lg, z1, dr)
        t2.close()


def test_branch_lines_stay_in_the_head_and_are_few():
    """A conversation read cold more than once has its count fall where each read ended; past the
    first answer no other session reads those lines, and they are not anchored."""
    ids = tokens(220, seed=49)
    hs = cache.prefix_hashes(ids)
    lines = range(CH, 220, CH)                                # 16 .. 208
    def plan(counts, first_answer):
        with tempfile.TemporaryDirectory() as d:
            t = tier(d)
            for L in lines:
                t.counts[(L, hs[L])] = counts(L) - 1           # the plan counts this read
            job = t.plan(ids, hs, 0, len(ids), len(ids), None, None,
                         {"client": "opencode", "first_answer": first_answer})
            t.close()
            return {L: v["why"] for L, v in job.lines.items()}
    falls = lambda L: 4 if L <= 48 else 3 if L <= 96 else 2 if L <= 144 else 1
    assert plan(falls, 60) == {48: "branch", 144: "repeated"}
    assert plan(falls, None) == {48: "branch", 96: "branch", 144: "repeated"}
    # a staircase: one fall at every line; the longest BRANCH_MAX are anchored
    stairs = lambda L: 20 - L // CH
    assert plan(stairs, None) == {144: "branch", 160: "branch", 176: "branch", 192: "branch",
                                  208: "repeated"}
    assert disk_cache.BRANCH_MAX == 4


def test_package_entries_are_named_and_previewed_and_others_are_not():
    head = tokens(100, seed=12)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1}, preview=lambda keys, L, kind, spans: "pkg text")
        tag = {"client": "opencode", "spans": [(40, 60, ["packages"], ["voxel-seasons"])]}
        run(fresh(), head, t, Pos(), disk_tag=tag)
        rep = t.report(previews=True)
        assert rep["top"][0]["kind"] == "package" and rep["top"][0]["label"] == "voxel-seasons"
        assert rep["top"][0]["preview"] == "pkg text"
        other = tokens(100, seed=13)
        run(fresh(), other, t, Pos(), disk_tag={"client": "open-webui"})
        rows = {r["kind"]: r for r in t.report(previews=True)["top"]}
        assert rows["prefix"]["preview"] is None              # a user's own text: never shown
        t.close()


def test_a_session_snapshot_and_a_tier_that_is_off_write_nothing():
    ids = tokens(120, seed=14)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1, "mode": "off"})
        eng, dr = fresh(), Pos()
        lg, _, _ = run(eng, ids, t, dr)
        exact(eng, lg, ids, dr)
        assert files(d, "anchors") == [] and files(d, "chunks") == [] and not t.counts
        t.apply({"admit_after": 1, "mode": "on"})
        store = cache.StateStore(BIG, chunk=CH)
        eng = fresh()
        cache.prefill(eng, None, ids, DEV, store=store, chunk=CH)
        store.put(ids, cache.capture(eng), kind="session")
        turn = ids + tokens(60, seed=15)
        run(eng, turn, t, None, store=store)                  # resumed from the session snapshot
        assert files(d, "anchors") == []
        t.close()


def test_the_tier_off_and_no_tier_give_the_same_bytes():
    ids = tokens(170, seed=16)
    eng0, d0 = fresh(), Pos()
    lg0, _, _ = run(eng0, ids, None, d0)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"mode": "off"})
        eng1, d1 = fresh(), Pos()
        lg1, _, _ = run(eng1, ids, t, d1)
        assert torch.equal(lg0, lg1)
        same_state(eng0, eng1, len(ids))
        assert torch.equal(d0.buf, d1.buf)
        t.close()


def test_image_rows_key_on_their_content():
    ids = tokens(120, seed=17)
    k1 = list(ids)
    k1[20:30] = [disk_cache_key(1)] * 10
    k2 = list(ids)
    k2[20:30] = [disk_cache_key(2)] * 10
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos(), key_ids=k1)
        _, reused, kind = run(fresh(), ids + [4], t, Pos(), key_ids=k2 + [4])
        assert kind != "disk" and reused == 0
        _, reused, kind = run(fresh(), ids + [4], t, Pos(), key_ids=k1 + [4])
        assert kind == "disk"
        t.close()


def disk_cache_key(i):
    return (1 << 40) + i                                      # engine/vision.py KEY_BASE + digest


# ------------------------------------------------------------------ 3. a hint, never an answer

def test_a_chunk_whose_tokens_differ_is_never_restored():
    ids = tokens(120, seed=18)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos())
        t.close()
        # the stored ids of one chunk changed under the same key: what a 64-bit collision is
        p = os.path.join(d, "chunks", files(d, "chunks")[2])
        with open(p, "r+b") as f:
            f.seek(disk_cache.ALIGN + 8 * 3)
            f.write((999).to_bytes(8, "little"))
        t2 = tier(d, settings={"admit_after": 1})
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, ids + [1], t2, dr)
        assert kind is None and t2.stats["rejected_collisions"] == 1
        exact(eng, lg, ids + [1], dr)
        t2.close()


# ------------------------------------------------------------------ 4. crash safety

def test_a_half_written_file_is_never_read_and_a_bad_checksum_costs_only_the_entry():
    ids = tokens(150, seed=19)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos())
        t.close()
        # a crash in the middle of a write leaves a .tmp, which the next start deletes
        with open(os.path.join(d, "chunks", "0000160-00000000deadbeef.chk.tmp"), "wb") as f:
            f.write(b"QSEDISK1" + b"\0" * 100)
        # and a flipped bit in a chunk's rows
        name = files(d, "chunks")[6]                         # rows 96..112
        p = os.path.join(d, "chunks", name)
        with open(p, "r+b") as f:
            f.seek(disk_cache.align(disk_cache.ALIGN + 8 * CH) + 40)
            x = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([x[0] ^ 0x10]))
        events = []
        t2 = tier(d, events=events, settings={"admit_after": 1})
        assert t2.stats["swept_tmp"] == 1
        eng, dr, res = fresh(), Pos(), resident()
        run(eng, ids[:70], None, dr, res)                     # rows in place up to 70
        w0 = (t2.stats["writes_chunk"], t2.stats["writes_anchor"])   # counted across restarts
        lg, reused, kind = run(eng, ids + [2], t2, dr, res)
        # the restore read chunks 64..96 and failed at 112: the rows in place carry it
        assert kind == "resident" and reused == 64, (kind, reused)
        exact(eng, lg, ids + [2], dr)
        assert t2.stats["corrupt"] == 1 and t2.stats["failed_restores"] == 1
        assert any(k == "restore_failed" and det["why"] == "corrupt" for k, det in events)
        # the bad chunk went with every longer one and the anchor; the prefill that forwarded
        # those lines again wrote them again (three chunks and the anchor at 144)
        assert (t2.stats["writes_chunk"] - w0[0], t2.stats["writes_anchor"] - w0[1]) == (3, 1)
        assert len(files(d, "chunks")) == 9 and len(files(d, "anchors")) == 1
        eng2, dr2 = fresh(), Pos()
        lg, reused, kind = run(eng2, ids + [2], t2, dr2)
        assert (kind, reused) == ("disk", 144)
        exact(eng2, lg, ids + [2], dr2)
        t2.close()


def test_a_file_of_another_fingerprint_is_swept():
    ids = tokens(120, seed=20)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos())
        t.close()
        t2 = tier(d, fp="other weights", settings={"admit_after": 1})
        assert t2.stats["swept_fingerprint"] == 8 and t2.report()["entries"] == 0
        assert files(d, "chunks") == [] and files(d, "anchors") == []
        t2.close()


def test_another_drafter_layout_is_refused_before_a_row_changes():
    ids = tokens(120, seed=21)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos())

        class Wide(Pos):
            def __init__(self):
                self.buf = torch.zeros(512, 6)
                self.ctx_len = 0

            def sync(self, toks, hidden, first_pos, rows=None):
                n = len(toks)
                self.buf[first_pos:first_pos + n] = hidden[:n, :6].float()
                self.ctx_len = first_pos + n
        eng, dr = fresh(), Wide()
        lg, reused, kind = run(eng, ids + [1], t, dr)
        assert kind is None and t.stats["failed_restores"] == 1
        ref, ref_lg = cold(ids + [1], Wide())
        assert torch.equal(lg, ref_lg)
        same_state(eng, ref, len(ids) + 1)
        t.close()


# ------------------------------------------------------------------ 5. write-behind

def test_the_next_prefill_closes_the_window_and_nothing_half_written_is_offered():
    a = tokens(150, seed=22)
    with tempfile.TemporaryDirectory() as d:
        events = []
        t = tier(d, events=events, settings={"admit_after": 1})
        eng, dr = fresh(), Pos()
        t._io.acquire()                                       # the writer cannot copy a chunk yet
        try:
            cache.prefill(eng, dr, a, DEV, chunk=CH, disk=t)
            cache.prefill(eng, dr, tokens(10, seed=23), DEV, chunk=CH, disk=t)   # writes row 0 on
        finally:
            t._io.release()
        assert t.flush()
        assert t.stats["dropped"].get("window") == 9, t.stats["dropped"]
        assert files(d, "chunks") == [] and len(files(d, "anchors")) == 1
        assert any(k == "write" and det["dropped"] == {"window": 9} for k, det in events)
        eng2, dr2 = fresh(), Pos()
        lg, reused, kind = run(eng2, a + [1], t, dr2)
        assert kind is None and t.stats["incomplete"] >= 1   # an anchor without its rows
        exact(eng2, lg, a + [1], dr2)
        t.close()


def test_a_resumed_extension_leaves_the_window_open():
    a = tokens(150, seed=24)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        eng, dr, res = fresh(), Pos(), resident()
        t._io.acquire()
        try:
            cache.prefill(eng, dr, a, DEV, chunk=CH, disk=t, resident=res)
            # the next turn resumes at 144 from the rows in place: it writes from 144 on
            cache.prefill(eng, dr, a + tokens(30, seed=25), DEV, chunk=CH, disk=t, resident=res)
        finally:
            t._io.release()
        assert t.flush()
        # all nine of the first job's chunks, and the two of the second's (its anchor at 176)
        assert not t.stats["dropped"].get("window") and len(files(d, "chunks")) == 11
        assert len(files(d, "anchors")) == 2
        eng2, dr2 = fresh(), Pos()
        lg, reused, kind = run(eng2, a + [7], t, dr2)
        assert (kind, reused) == ("disk", 144)
        exact(eng2, lg, a + [7], dr2)
        t.close()


def test_a_client_that_left_part_way_leaves_what_was_prefilled():
    a = tokens(150, seed=26)

    def leave(done, total):
        if done >= 96:
            raise ConnectionResetError("client left")
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        eng, dr = fresh(), Pos()
        try:
            cache.prefill(eng, dr, a, DEV, chunk=CH, disk=t, on_chunk=leave)
        except ConnectionResetError:
            pass
        assert t.flush()
        # the anchor at 144 was never reached; the chunks up to 96 were written, then aborted
        assert files(d, "anchors") == [] and len(files(d, "chunks")) == 6
        assert t.stats["dropped"].get("aborted") == 3
        t.close()


# ------------------------------------------------------------------ 6. the disk's limits

def _sizes(d):
    c = os.path.getsize(os.path.join(d, "chunks", files(d, "chunks")[0]))
    a = os.path.getsize(os.path.join(d, "anchors", files(d, "anchors")[0]))
    return c, a


def test_the_budget_evicts_least_valuable_first():
    x, y, z = tokens(70, seed=27), tokens(70, seed=28), tokens(70, seed=29)
    with tempfile.TemporaryDirectory() as d:
        clock = Clock()
        events = []
        t = tier(d, events=events, clock=clock, settings={"admit_after": 1})
        run(fresh(), x, t, Pos())                             # line 64: 4 chunks and an anchor
        c, a = _sizes(d)
        entry = 4 * c + a
        t.apply({"admit_after": 1, "budget_gb": (2 * entry + c) / 1e9})
        clock.t += 60
        run(fresh(), y, t, Pos())
        for _ in range(3):                                     # x is used, y is not
            clock.t += 60
            run(fresh(), x + [1], t, Pos())
        clock.t += 60
        run(fresh(), z, t, Pos())                              # over the budget: y goes
        h = cache.prefix_hashes
        assert (64, h(x)[64]) in t.anchors and (64, h(z)[64]) in t.anchors
        assert (64, h(y)[64]) not in t.anchors
        assert t.bytes <= t.budget and t.stats["evicted"].get("budget") == 1
        assert len(files(d, "chunks")) == 8                    # y's chunks went with its anchor
        assert any(k == "evict" and det["why"] == "budget" for k, det in events)
        t.close()


def test_past_the_ttl_an_entry_goes():
    x = tokens(70, seed=30)
    with tempfile.TemporaryDirectory() as d:
        clock = Clock()
        t = tier(d, clock=clock, settings={"admit_after": 1, "ttl_days": 1})
        run(fresh(), x, t, Pos())
        clock.t += 2 * 86400
        t._sweep("test")
        assert t.report()["entries"] == 0 and files(d, "chunks") == []
        assert t.stats["evicted"].get("ttl") == 1
        t.close()


def test_the_free_space_floor_is_never_crossed():
    x = tokens(70, seed=31)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1, "min_free_gb": 10},
                 free_bytes=lambda p: disk_cache.FLOOR_BYTES + 1000)
        assert t.floor == disk_cache.FLOOR_BYTES                # 10 asked, 50 held
        run(fresh(), x, t, Pos())
        assert files(d, "chunks") == [] and files(d, "anchors") == []
        assert t.stats["dropped"].get("floor", 0) >= 1
        t.close()


def test_clear_empties_the_disk_and_the_counts():
    x = tokens(120, seed=32)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), x, t, Pos())
        before = t.clear()
        assert before["entries"] == 1 and before["chunks"] == 7
        assert files(d, "chunks") == [] and files(d, "anchors") == [] and not t.counts
        assert t.bytes == 0 and t.stats["cleared"] == 1
        t.close()


def test_the_staging_buffer_is_two_anchor_and_two_chunk_slots():
    x = tokens(120, seed=33)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        eng, dr = fresh(), Pos()
        planned = t.plan_staging(eng, dr)
        assert t.staging_pending() == planned and t.staging_bytes == 0
        run(eng, x, t, dr)
        run(fresh(), x + [1], t, Pos())
        anchor = disk_cache.align(eng.state.S.numel() * 4) + disk_cache.align(
            eng.state.conv.numel() * eng.state.conv.element_size())
        rows = sum(disk_cache.align(disk_cache._row_bytes(v, ax) * CH)
                   for v, ax in cache._kv_views(eng, dr))
        assert t.staging_bytes <= 2 * disk_cache.align(anchor) + 2 * rows
        assert t.staging_pending() == 0 and len(t.buf_anchor) == 2 and len(t.buf_chunk) == 2
        t.close()


def test_a_job_over_half_the_budget_is_not_admitted_and_the_estimate_is_per_kind():
    """Before the staging buffer exists the estimate is the planned per-chunk and per-anchor
    sizes, not one slot of each per file: a long prefix is not refused for the wrong reason."""
    x = tokens(200, seed=35)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        eng, dr = fresh(), Pos()
        t.plan_staging(eng, dr)
        anchor, chunk = t._est
        assert t._estimate(12, 1) == 12 * chunk + anchor
        t.apply({"admit_after": 1, "budget_gb": (12 * chunk + anchor) * 2.1 / 1e9})
        run(eng, x, t, dr)                                  # 12 chunks and an anchor: under half
        assert len(files(d, "anchors")) == 1
        est = 12 * t.buf_chunk[0].nbytes + t.buf_anchor[0].nbytes      # the slots exist now
        t.clear()
        t.apply({"admit_after": 1, "budget_gb": est * 1.5 / 1e9})
        events = []
        t.on_event = lambda k, det: events.append((k, det))
        run(fresh(), tokens(200, seed=36), t, Pos())        # over half the budget: not admitted
        assert ("drop", {"why": "too_big", "tokens": 192, "bytes": est}) in events, events
        assert files(d, "anchors") == [] and t.stats["dropped"]["too_big"] == 1
        t.close()


def test_an_index_survives_a_restart_with_its_counts_and_hits():
    x = tokens(120, seed=34)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d)
        run(fresh(), x, t, Pos())                             # counted once
        t.close()
        t2 = tier(d)
        run(fresh(), x, t2, Pos())                            # the second: written
        run(fresh(), x + [1], t2, Pos())                      # and a hit
        t2.close()
        t3 = tier(d)
        rep = t3.report()
        assert rep["entries"] == 1 and rep["top"][0]["hits"] == 1 and rep["hits"] == 1
        t3.close()


def test_the_settings_are_clamped_and_applied():
    with tempfile.TemporaryDirectory() as d:
        t = tier(d)
        changed = t.apply({"budget_gb": 20, "ttl_days": 3, "min_free_gb": 1, "bogus": 1})
        assert "budget_gb=20" in changed and t.budget == 20 * disk_cache.GB
        assert t.floor == disk_cache.FLOOR_BYTES and t.ttl == 3 * 86400
        t.close()


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


# ------------------------------------------------------------------ a side slot reads the tier

def _side_engine(max_len=320, seed=3):
    """A side slot's engine: the same weights, its own (shorter) window and buffers."""
    from engine.model import Qwen38Engine
    from test_forward_tree import FakeWeights, tiny_config
    from test_cache import DT
    cfg = tiny_config()
    eng = Qwen38Engine(cfg, FakeWeights(cfg, seed), max_len=max_len, device=DEV)
    eng.kv.k = eng.kv.k.to(DT)
    eng.kv.v = eng.kv.v.to(DT)
    eng.state.conv = eng.state.conv.to(DT)
    return eng


def test_a_side_slot_reads_the_head_into_its_own_buffers_bit_for_bit():
    """SRV-231: slot 0 wrote a client's head; a new chat on a side slot (no resident prefix, its
    own shorter window) restores it through `SideReader` and holds what its own cold prefill of
    the whole prompt holds, to the bit: logits, recurrent state, convolution tails, KV and the
    drafter's rows. The slot-0 engine's rows are not touched."""
    from server.diskcache import SideReader
    head = tokens(150, seed=21)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        slot0 = fresh()
        run(slot0, head + tokens(5, seed=22), t, Pos())
        k0 = slot0.kv.k.clone()
        side, dr = _side_engine(), Pos()
        prompt = head + tokens(30, seed=23)
        info = {}
        lg, reused, fwd = cache.prefill(side, dr, prompt, DEV, chunk=CH, disk=SideReader(t),
                                        resident=None, info=info)
        assert info.get("kind") == "disk" and reused == 144 and fwd == len(prompt) - 144
        ref, ref_d = _side_engine(), Pos()
        ref_lg, _, _ = cache.prefill(ref, ref_d, prompt, DEV, chunk=CH)
        assert torch.equal(lg, ref_lg), (lg - ref_lg).abs().max().item()
        same_state(side, ref, len(prompt))
        assert torch.equal(dr.buf[:len(prompt)], ref_d.buf[:len(prompt)])
        assert torch.equal(slot0.kv.k, k0), "slot 0's buffers are not the side slot's"
        t.close()


def test_without_the_reader_a_side_slot_prefills_cold():
    """The A side of the A/B (`QWEN38_SIDE_DISK=off`): no tier, every token read."""
    head = tokens(150, seed=21)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), head + tokens(5, seed=22), t, Pos())
        side = _side_engine()
        info = {}
        _, reused, fwd = cache.prefill(side, Pos(), head + tokens(30, seed=23), DEV, chunk=CH,
                                       disk=None, resident=None, info=info)
        assert (info.get("kind"), reused, fwd) == (None, 0, 180)
        t.close()


def test_a_side_prefill_writes_nothing_and_fences_nothing():
    """A side slot's prefill neither counts lines nor plans writes (it would write the side
    engine's rows under slot 0's names) nor fences slot 0's write-behind jobs."""
    from server.diskcache import SideReader
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        r = SideReader(t)
        cache.prefill(_side_engine(), Pos(), tokens(150, seed=31), DEV, chunk=CH, disk=r,
                      resident=None)
        assert t.flush()
        assert files(d, "anchors") == [] and files(d, "chunks") == []
        assert t.counts == {} or not any(t.counts.values()), "no line counted"
        fenced = []
        t.fence = lambda safe: fenced.append(safe)
        cache.prefill(_side_engine(), Pos(), tokens(150, seed=32), DEV, chunk=CH, disk=r,
                      resident=None)
        assert fenced == []
        assert r.plan() is None and r.guard_rows(10) == 0
        assert r.report()["lookups"] == 2 and r.report()["hits"] == 0
        t.close()


if __name__ == "__main__":
    _main()
