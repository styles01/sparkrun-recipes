"""FX17: a router note stays live when the client changed the prompt before its message.

THE MEASUREMENT (2026-10-08 12:14:30, live 0.4.7; again on 0.4.8-test run 3): Open WebUI moves its
retrieval block from an earlier user message to its last one on every new message. The router keyed
every note on the sha of the client's prefix up to the insertion point; the block leaving the
first message broke every hash, the 4 notes (8,299 tokens) were dropped, and the whole class note
(5.8k tokens) was put into the new message again, every message.

THE RULE (server/conductor/inject.py `message_anchor` / `locate_anchor`, router.py `Plan._live`):
a note also remembers the ordinal of the user message it closes and the last 32 client tokens
before its insertion point (the user's own words). When the prefix hash breaks, the note is live
again at the end of the same ordinal message if that message still ends with those tokens; it
moves there. Any change to the user's own words there, a missing message, or a different ordinal:
dropped as before. `QWEN38_NOTE_ANCHOR=off` is the prefix hash alone.

  1  the anchor finds the message after a block left it; an edited message end does not
  2  `Plan._live`: the note moves (its hash renewed), the journal counts `notes_moved`; a copy
     whose user words changed drops it; off drops it
  3  the end alignment's moved-block pass with the notes present in the copy (what the router
     now sends) resumes at the end anchor

Run: python tests/test_note_anchor.py
"""

from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

from server import app, endalign  # noqa: E402
from server.conductor import inject, router  # noqa: E402
from test_end_align import NL, _cold, _first, _setup, tb_of  # noqa: E402
from test_end_align_moved import BLOCK, HEAD, IM_E, IM_S, MARKS, Q1, Q2, _ids, user  # noqa: E402
from test_end_anchor import turn  # noqa: E402

NOTE = _ids(40, 300)


def _prompt(block, q, tail=(NL,)):
    return HEAD + user(block, [NL], q) + list(tail)


def test_the_anchor_finds_the_message_after_the_block_left_it():
    p1 = _prompt(BLOCK, Q1)
    at = inject.insertion_point(p1, IM_E)
    a = inject.message_anchor(p1, at, MARKS)
    assert a[0] == 1 and a[1] == tuple(p1[at - 32:at])
    p2 = HEAD + user(Q1) + [NL] + [5, 6] + user(BLOCK, [NL], Q2) + [NL]
    at2 = inject.locate_anchor(p2, a, MARKS, IM_E)
    assert at2 == HEAD.__len__() + len(user(Q1)) - 1 and p2[at2] == IM_E
    edited = list(Q1)
    edited[-1] = 2 if edited[-1] != 2 else 3
    p3 = HEAD + user(edited) + [NL] + [5, 6] + user(BLOCK, [NL], Q2) + [NL]
    assert inject.locate_anchor(p3, a, MARKS, IM_E) is None
    assert inject.locate_anchor(HEAD + [NL], a, MARKS, IM_E) is None
    assert inject.message_anchor(p1, at, None) is None


def _plan(notes):
    pl = router.Plan.__new__(router.Plan)
    sess = router.Session("s")
    sess.notes = list(notes)
    sess.lock = threading.Lock()
    pl.session = sess
    pl.router_cfg = {"injection": {"replay": True, "max_replay_tokens": 16000}}
    pl.decisions = {"packages": "on"}
    pl.row = {"extra": {}}
    return pl


def test_live_moves_the_note_and_drops_an_edited_one():
    p1 = _prompt(BLOCK, Q1)
    at = inject.insertion_point(p1, IM_E)
    h = inject.hashes_at(p1, [at])[at]
    n = router.Injection(at, h, NOTE, "note", [{"kind": "package", "name": "x"}], {"packages"},
                         "k1", 1, "r1", anchor=inject.message_anchor(p1, at, MARKS))
    pl = _plan([n])
    # the next prompt: the block moved to the new last message, message 1 bare
    p2 = HEAD + user(Q1) + [NL] + [5, 6] + user(BLOCK, [NL], Q2) + [NL]
    at2 = inject.insertion_point(p2, IM_E)
    live, hashes = pl._live(p2, at2, marks=MARKS, im_end=IM_E)
    assert len(live) == 1 and live[0].pos == len(HEAD) + len(user(Q1)) - 1, [x.pos for x in live]
    assert live[0].hash == inject.hashes_at(p2, [live[0].pos])[live[0].pos] and live[0].key == "k1"
    assert pl.row["extra"]["notes_moved"] == 1
    assert inject.splice(p2, [(live[0].pos, live[0].ids)])[0][live[0].pos:live[0].pos + len(NOTE)] == NOTE
    # the user's words changed: dropped
    edited = list(Q1)
    edited[2] = 2 if edited[2] != 2 else 3
    p3 = HEAD + user(edited) + [NL] + [5, 6] + user(BLOCK, [NL], Q2) + [NL]
    pl = _plan([n])
    live, _ = pl._live(p3, inject.insertion_point(p3, IM_E), marks=MARKS, im_end=IM_E)
    assert live == [] and "notes_moved" not in pl.row["extra"]
    # off: the prefix hash alone
    was = router.NOTE_ANCHOR
    router.NOTE_ANCHOR = "off"
    try:
        pl = _plan([n])
        live, _ = pl._live(p2, at2, marks=MARKS, im_end=IM_E)
        assert live == []
    finally:
        router.NOTE_ANCHOR = was
    # the hash still matching: the note stays where it was, nothing moved
    pl = _plan([n])
    live, _ = pl._live(p1 + [7], at, marks=MARKS, im_end=IM_E)
    assert len(live) == 1 and live[0] is n and "notes_moved" not in pl.row["extra"]


def test_the_moved_block_pass_with_the_notes_in_the_copy():
    """What the engine sees now: the rows hold message 1 = block + words + note; the copy holds
    message 1 = words + note (the router kept the note live), and the block in the last message."""
    res, al = _setup()
    al.marks = MARKS
    served1 = HEAD + user(BLOCK, [NL], Q1, NOTE) + [NL]
    out, _ = _first(served1)
    e = res.end["at"]
    res.end["served_hidden"] = [(len(HEAD) + len(user(BLOCK, [NL], Q1)) - 1,
                                 len(HEAD) + len(user(BLOCK, [NL], Q1)) - 1 + len(NOTE))]
    copy = HEAD + user(Q1, NOTE) + [NL] + out + user(BLOCK, [NL], Q2) + [NL]
    ids, spans, aligned = app._end_replay(res, copy)
    assert ids[:e] == res.tokens[:e] and aligned > 0, (len(ids), e)
    a, b = al.last_moved[0]
    assert res.tokens[a:b] == BLOCK + [NL]
    got, pre, _ = turn(ids, 8)
    assert pre["kind"] == "end" and pre["reused"] == e, pre
    assert got == _cold(ids, 8)
    # and with the notes dropped (0.4.8-test's shape), the same end
    copy2 = HEAD + user(Q1) + [NL] + out + user(BLOCK, [NL], Q2, NOTE) + [NL]
    _setup()
    res2, al2 = _setup()
    al2.marks = MARKS
    out2, _ = _first(served1)
    assert out2 == out
    ns = len(HEAD) + len(user(BLOCK, [NL], Q1)) - 1
    res2.end["served_hidden"] = [(ns, ns + len(NOTE))]
    ids2, _, aligned2 = app._end_replay(res2, copy2)
    assert ids2[:e] == res2.tokens[:e] and aligned2 > 0


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:72s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
