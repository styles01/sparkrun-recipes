"""A compaction served as the continuation of the conversation the engine holds, and the summary
prepared ahead (server/compaction.py), on the random 4-layer model.

THE MEASURED CASES (a user's ~/rrr6 session, opencode 1.18.34): a manual /compact read 63,614 tokens
cold in 189.8 s, an automatic one 129,643 tokens in 270 s, while the conversation (110k, 194k
tokens) was resident. opencode sends its compaction agent's prompt and the history as ONE text
message, which shares no token with what the engine read.

  1  `split`: opencode's request shape (system text, `<conversation>` history, instructions after
     it), the guard's first and last user lines; anything else is not split
  2  the continuation resumes at the end anchor and forwards only its own turn, and writes the
     tokens a cold prefill of the same ids writes (the session gate)
  3  after images: the end is kept with its rotary offset, offered to the continuation alone (a
     plain request resumes at the anchors in place, as before), and the continuation's tokens are
     a cold run's at that offset; the offset survives the NVMe entry
  4  `find_record`: the rows in place for their owner only, else the session's entry on the NVMe
  5  the preparer: idle, it writes the summary in scratch rows, leaves the conversation's rows,
     anchors and end anchor as they were (the next turn resumes at the end and writes the same
     tokens), and a waiting request stops it with the same guarantee
  6  `Prepared`: answered only at the same end anchor, the same tokens and the same instructions
  7  usage: a continuation's cached tokens are the client's prompt less what was forwarded

Run: python tests/test_compaction_continue.py
"""

from __future__ import annotations

import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (sets the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import cache  # noqa: E402
from server import app, compaction  # noqa: E402
from server import usage as usage_mod  # noqa: E402
from test_cache import tokens  # noqa: E402
from test_end_anchor import _state  # noqa: E402

OC = compaction.OPENCODE_SYSTEM


def _req(history, instructions=compaction.OPENCODE_INSTRUCTIONS, system=OC, tools=None):
    user = f"Here is the conversation so far:\n\n<conversation>\n{history}\n</conversation>\n\n" \
           f"{instructions}"
    b = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if tools:
        b["tools"] = tools
    return b


# ------------------------------------------------------------------ 1. the request's parts

def test_split_reads_opencodes_shape_and_nothing_else():
    hist = ("[User]: Build a cinematic   product page for AURA One.\n[Assistant]: Sure.\n"
            "[Assistant tool call]: read({})\n[Tool result]: ok\n[User]: now make the hero darker")
    p = compaction.split(_req(hist))
    assert p["system"] == OC and p["instructions"].startswith("Create a new anchored summary")
    assert p["guard"] == ["Build a cinematic product page for AURA One.",
                          "now make the hero darker"], p["guard"]
    assert compaction.split(_req(hist, tools=[{"type": "function"}])) is None
    b = _req(hist)
    b["messages"].append({"role": "user", "content": "x"})
    assert compaction.split(b) is None
    b = _req(hist)
    b["messages"][1]["content"] = "no history tags here"
    assert compaction.split(b) is None
    # a later compaction quotes the prior summary before the template: same key
    later = ("Here is the summary of the conversation before the <conversation> above:\n"
             "<prior-summary>\n## Objective\n- x\n</prior-summary>\nUpdate the anchored summary.\n"
             + compaction.OPENCODE_INSTRUCTIONS[compaction.OPENCODE_INSTRUCTIONS.find("Output"):])
    assert compaction.key(OC, later) == compaction.key(OC, compaction.OPENCODE_INSTRUCTIONS)
    assert compaction.key(OC + "x", later) != compaction.key(OC, later)


# ------------------------------------------------------------------ 2. the continuation

def _turn(ids, n, session="ses", pos_delta=0, delta=0, end_mm=False):
    out = list(app.generate_stream(torch.tensor(ids), n, set(), None, session,
                                   pos_delta=pos_delta, end_mm=end_mm))
    pre = dict(app.STATE["last_prefill"])
    app._remember(list(ids), out, session, delta=delta)
    return out, pre


def _cold(ids, n, pos_delta=0):
    """The same ids on a fresh state with no resident prefix: the reference."""
    _state(end=True)
    app.STATE["resident"] = None
    return list(app.generate_stream(torch.tensor(ids), n, set(), None, "cold",
                                    pos_delta=pos_delta))


def test_the_continuation_resumes_at_the_end_and_writes_a_cold_runs_tokens():
    res = _state(end=True)
    prompt = tokens(120, seed=61)
    out1, _ = _turn(prompt, 60)
    r = compaction.find_record(res, None, "ses")
    assert r is not None and r.where == "resident" and r.at == res.end["at"] == len(prompt) + 59
    tail = tokens(23, seed=62)
    ids = r.tokens + tail
    out2, pre2 = _turn(ids, 30)
    assert pre2["kind"] == "end" and pre2["reused"] == r.at and pre2["forwarded"] == len(tail), pre2
    assert out2 == _cold(ids, 30)


# ------------------------------------------------------------------ 3. after images

def test_after_images_the_end_is_the_continuations_alone_at_its_offset():
    d = -7
    res = _state(end=True)
    prompt = tokens(110, seed=63)
    # a conversation whose rows after the prompt take a rotary offset, as after images
    _turn(prompt, 50, pos_delta=d, delta=d)
    assert res.end["mm"] and res.end["delta"] == d, res.end
    e = res.end["at"]
    assert res.end_offer(e, 10_000) == 0 and res.end_offer(e, 10_000, mm_ok=True) == e
    r = compaction.find_record(res, None, "ses")
    assert r.delta == d and r.mm
    tail = tokens(19, seed=64)
    ids = r.tokens + tail
    out, pre = _turn(ids, 25, pos_delta=r.delta, delta=r.delta, end_mm=True)
    assert pre["kind"] == "end" and pre["reused"] == e, pre
    # the reference: the same conversation cold, its generation at the same offset, then the turn
    _state(end=True)
    app.STATE["resident"] = None
    ref = list(app.generate_stream(torch.tensor(prompt), 50, set(), None, "cold", pos_delta=d))
    assert ref[:-1] == r.tokens[len(prompt):]
    # (the continuation's own rows: a fresh run of the whole ids with that offset everywhere is
    # not the reference, because turn 1's prompt rows took offset 0; so the comparison is with a
    # resident resume of the same ids, which must not depend on how the end was reached)
    res2 = _state(end=True)
    _turn(prompt, 50, pos_delta=d, delta=d)
    out_b, pre_b = _turn(ids, 25, pos_delta=d, delta=d, end_mm=True)
    assert pre_b["kind"] == "end" and out_b == out
    # a plain request continuing the same rows resumes at the anchors in place, never at the end
    _state(end=True)
    _turn(prompt, 50, pos_delta=d, delta=d)
    _, pre_c = _turn(ids, 5)
    assert pre_c["kind"] != "end", pre_c
    assert res2.end is not None


def test_a_continuation_after_images_keeps_its_key_ids_out_of_the_lookup_store():
    """Live, 2026-10-04 13:10: the continuation's prompt carries the images' key ids (past 32
    bits); the suffix store's packing raised on them after the summary and the stream lost its
    last chunk. They are left out of the corpus now."""
    import tempfile
    from engine.cache import PersistentSuffixStore
    d = -7
    res = _state(end=True)
    prompt = tokens(90, seed=77)
    _turn(prompt, 20, pos_delta=d, delta=d)
    with tempfile.TemporaryDirectory() as root:
        store = PersistentSuffixStore(os.path.join(root, "suffix"), tokenizer_sha="t" * 64).open()
        app.STATE["suffix_store"], app.STATE["suffix_scope"] = store, "all"
        r = compaction.find_record(res, None, "ses")
        ids = list(r.tokens)
        ids[5] = (1 << 40) + 123                       # an image's key id in the record
        app._remember(ids + tokens(9, seed=78), [1, 2, 3], "ses", delta=d)
        app.STATE["suffix_store"] = None


def test_the_offset_rides_the_nvme_entry():
    import tempfile
    from engine import spill as spill_mod
    d = -5
    res = _state(end=True)
    with tempfile.TemporaryDirectory() as root:
        sp = spill_mod.SessionSpill(root, chunk=16, fingerprint="t").open()
        res.spiller = sp
        app.STATE["spill"] = sp
        prompt = tokens(100, seed=65)
        _turn(prompt, 40, pos_delta=d, delta=d)
        e = res.end["at"]
        info = sp.save_resident(res, app.STATE["engine"], None, "test")
        assert info is not None and info["end"], info
        man = next(iter(sp.entries.values())).man
        assert man["end"]["mm"] and man["end"]["delta"] == d, man["end"]
        res.clear()
        r = compaction.find_record(res, sp, "ses")
        assert r is not None and r.where == "spill" and r.at == e and r.delta == d
        # a plain prompt is not offered the entry's end; the continuation is
        ids = r.tokens + tokens(9, seed=66)
        ent = next(iter(sp.entries.values()))
        assert ent.offer(ids, 10_000) < e and ent.offer(ids, 10_000, mm_ok=True) == e
        got = sp.maybe_restore(res, app.STATE["engine"], None, ids, len(ids) - 1, mm_ok=True)
        assert got is not None and res.end is not None and res.end["delta"] == d, res.end
        sp.close()


# ------------------------------------------------------------------ 4. the record

def test_the_record_is_the_owners_alone():
    res = _state(end=True)
    _turn(tokens(80, seed=67), 20, session="A")
    assert compaction.find_record(res, None, "A") is not None
    assert compaction.find_record(res, None, "B") is None
    assert compaction.find_record(res, None, None) is None


# ------------------------------------------------------------------ 5. the preparer

class _Tok:
    """`tail_ids` reads the template as text and tokenizes it; ids inside the random vocabulary."""

    eos_token_id = None

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kw):
        return "<|im_start|>user\n" + messages[0]["content"][:20] + "<|im_end|>\n"

    def __call__(self, text, add_special_tokens=False):
        return type("R", (), {"input_ids": [1 + (ord(c) % 90) for c in text]})()

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(97 + int(t) % 26) for t in ids)


def _prep_state(idle_s=0.0, min_tokens=16):
    # the box lock another program may hold while this runs is not this test's hold
    import tempfile
    os.environ["BOX_LOCK"] = os.path.join(tempfile.gettempdir(), f"d5-box-lock-{os.getpid()}")
    res = _state(end=True)
    app.STATE.update(compaction=dict(compaction.SETTINGS, idle_s=idle_s, min_tokens=min_tokens,
                                     prepare_max_tokens=12),
                     prepared=compaction.Prepared(), last_end_mono=0.0, cfg_eos=None)
    app.STATE["tok"].eos_token_id = None
    # the box's live engine is another program on the GPU: not this test's (test_*_yields_*)
    app._gpu_others = lambda: []
    return res


def _with_eos(fn):
    """The random model never says an end token on its own: one of its greedy ids is made one."""
    real = app.eos_ids
    app.eos_ids = fn
    return real


def test_the_preparer_writes_ahead_and_leaves_the_conversation_as_it_was():
    res = _prep_state()
    prompt = tokens(90, seed=68)
    out1, _ = _turn(prompt, 30)
    before = (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint)
    prep = app.STATE["prepared"]
    assert app.prepare_once() == "client"           # no client kind seen for this session yet
    prep.seen("ses", "opencode")
    tail = compaction.tail_ids(_Tok(), *prep.instructions["opencode"])
    app.STATE["tok"] = _Tok()
    # which id ends the summary: the 5th the continuation writes
    ids = res.tokens + tail
    ref = _cold(ids, 12)
    res = _prep_state()
    _turn(prompt, 30)
    app.STATE["prepared"].seen("ses", "opencode")
    app.STATE["tok"] = _Tok()
    real = _with_eos(lambda body: {ref[4]})
    try:
        got = app.prepare_once()
    finally:
        app.eos_ids = real
    prep = app.STATE["prepared"]
    assert got == "prepared", (got, prep.stats)
    assert (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint) == before
    r = compaction.find_record(res, None, "ses")
    k = compaction.key(*prep.instructions["opencode"])
    assert prep.take("ses", r, k) == ref[:5]
    assert app.prepare_once() == "done"
    # the next turn resumes at the end and writes what it writes without the preparer
    tool = tokens(17, seed=69)
    nxt, pre = _turn(prompt + out1 + tool, 10)
    assert pre["kind"] == "end", pre
    _state(end=True)
    _turn(prompt, 30)
    nxt_ref, _ = _turn(prompt + out1 + tool, 10)
    assert nxt == nxt_ref


def test_a_waiting_request_stops_the_preparer_and_nothing_moves():
    res = _prep_state()
    prompt = tokens(90, seed=70)
    out1, _ = _turn(prompt, 30)
    app.STATE["prepared"].seen("ses", "opencode")
    app.STATE["tok"] = _Tok()
    before = (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint)
    calls = {"n": 0}
    real = app._waiting

    def waiting():
        calls["n"] += 1
        return calls["n"] > 14                      # a request arrives during the decode
    app._waiting = waiting
    try:
        got = app.prepare_once()
    finally:
        app._waiting = real
    assert got == "stopped", got
    assert (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint) == before
    assert app.STATE["prepared"].stats["stopped"] == 1
    assert not app.LOCK.locked()
    tool = tokens(17, seed=71)
    nxt, pre = _turn(prompt + out1 + tool, 10)
    assert pre["kind"] == "end", pre
    _state(end=True)
    _turn(prompt, 30)
    assert nxt == _turn(prompt + out1 + tool, 10)[0]


def test_the_preparer_takes_the_box_lock_and_a_held_one_stops_it():
    import fcntl
    res = _prep_state()
    _turn(tokens(90, seed=73), 30)
    app.STATE["prepared"].seen("ses", "opencode")
    app.STATE["tok"] = _Tok()
    fd = os.open(os.environ["BOX_LOCK"], os.O_RDONLY | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)        # another program's run beside the engine
    try:
        assert app.prepare_once() == "hold"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    seen = {}
    real = app.generate_stream

    def spy(*a, **kw):
        fd2 = os.open(os.environ["BOX_LOCK"], os.O_RDONLY)
        try:
            fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
            seen["free"] = True
            fcntl.flock(fd2, fcntl.LOCK_UN)
        except OSError:
            seen["free"] = False
        finally:
            os.close(fd2)
        return real(*a, **kw)
    app.generate_stream = spy
    try:
        app.prepare_once()
    finally:
        app.generate_stream = real
    assert seen == {"free": False}, seen                  # held for the length of the summary
    assert res.end is not None


def test_the_preparer_yields_to_a_lock_waiter_and_to_another_gpu_program():
    """A process waiting in `flock` for the box lock (a hold's `flock -w`) or another
    program on the GPU stops the summary written ahead, as a waiting request does; nothing of the
    conversation moves."""
    import shutil
    import subprocess as sp
    res = _prep_state()
    prompt = tokens(90, seed=74)
    out1 = _turn(prompt, 30)[0]
    app.STATE["prepared"].seen("ses", "opencode")
    app.STATE["tok"] = _Tok()
    end0 = res.end["at"]
    # another program on the GPU: not even started
    app._gpu_others = lambda: [424242]
    assert app.prepare_once() == "gpu"
    assert app.STATE["prepared"].stats["yielded"] == 1
    # one that starts while the summary is written
    calls = []

    def later():
        calls.append(1)
        return [] if len(calls) == 1 else [424242]
    app._gpu_others = later
    old = app.YIELD_GPU_S
    app.YIELD_GPU_S = 0.0
    try:
        assert app.prepare_once() == "yielded:gpu"
    finally:
        app.YIELD_GPU_S = old
        app._gpu_others = lambda: []
    assert res.end is not None and res.end["at"] == end0 and res.valid == end0
    # a process waiting in flock for the lock this holds
    if os.path.exists("/proc/locks") and shutil.which("flock"):
        real = app.generate_stream
        waiter = {}

        def spy(*a, **kw):
            waiter["p"] = sp.Popen(["flock", "-w", "20", os.environ["BOX_LOCK"], "true"])
            t0 = time.monotonic()
            while time.monotonic() - t0 < 5:
                fd = os.open(os.environ["BOX_LOCK"], os.O_RDONLY)
                try:
                    if app._box_waiters(fd):
                        break
                finally:
                    os.close(fd)
                time.sleep(0.02)
            return real(*a, **kw)
        app.generate_stream = spy
        try:
            got = app.prepare_once()
        finally:
            app.generate_stream = real
        assert got == "yielded:box_waiter", got
        assert waiter["p"].wait(10) == 0                   # it got the lock right after
        assert res.end is not None and res.end["at"] == end0
        assert app.STATE["prepared"].stats["yielded"] == 3
    # with nobody waiting the summary runs to its cap (the random model writes no end id here),
    # and the next turn resumes at the end
    assert app.prepare_once() == "length"
    _, pre = _turn(prompt + out1 + tokens(7, seed=75), 6)
    assert pre["kind"] == "end", pre


def test_the_preparer_waits_for_idle_length_and_no_hold():
    res = _prep_state(idle_s=3600.0)
    _turn(tokens(90, seed=72), 30)
    app.STATE["prepared"].seen("ses", "opencode")
    app.STATE["last_end_mono"] = __import__("time").monotonic()
    assert app.prepare_once() == "recent"
    app.STATE["compaction"]["idle_s"] = 0.0
    app.STATE["compaction"]["min_tokens"] = 10_000
    assert app.prepare_once() == "short"
    app.STATE["compaction"]["prepare"] = "off"
    assert app.prepare_once() == "off"
    assert res.end is not None


# ------------------------------------------------------------------ 6. what a prepared summary answers

def test_a_prepared_summary_answers_its_end_and_instructions_only():
    p = compaction.Prepared()
    r = compaction.Record(list(range(50)), 50, "resident")
    p.put("s", r, "k1", [7, 8, 9], {})
    assert p.take("s", r, "k1") == [7, 8, 9]
    assert p.take("s", r, "k2") is None
    assert p.take("s", compaction.Record(list(range(51)), 51, "resident"), "k1") is None
    assert p.take("s", compaction.Record([1] + list(range(1, 50)), 50, "resident"), "k1") is None
    assert p.take("t", r, "k1") is None
    assert p.stats["served"] == 1 and p.stats["stale"] == 3


# ------------------------------------------------------------------ 7. usage

def test_a_continuations_cached_tokens_are_the_clients_prompt_less_the_forwarded():
    rec = usage_mod.RequestRecord("c", "chat", True)
    rec.prompt_tokens = rec.prompt_client = 63_614
    rec.absorb_prefill({"reused": 110_592, "forwarded": 2_310, "kind": "end"})
    assert rec.cached_tokens == 63_614 - 2_310 and rec.cache_source == "continuation"


def test_the_guard_needs_the_first_and_last_user_lines_in_the_record():
    p = compaction.split(_req("[User]: first thing to build here\n[Assistant]: ok\n"
                              "[User]: and the last request"))
    assert compaction.guard_ok(p, "...first thing to build here ... and the last request ...")
    assert not compaction.guard_ok(p, "first thing to build here only")
    assert not compaction.guard_ok(compaction.split(_req("[Assistant]: no user")), "x")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:80s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
