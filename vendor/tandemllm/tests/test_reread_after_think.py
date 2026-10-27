"""Why a turn after a long thinking re-reads that thinking (rrr2, 2026-10-02), on the random model.

THE MEASUREMENT. Turn 1 of rrr2 prefilled 34,038 tokens and thought 32,773, closed by the D2
backstop with the hidden vendor phrase; turn 2's prompt of 66,964 tokens resumed at 33,792 and
re-read 33,172 (about 44 s). The hypothesis was that the client's copy of the reasoning lacks the
hidden phrase the KV holds, so the next prompt parts from the session snapshot near the end of the
thinking and falls back to the last prefill anchor.

WHAT THE LEDGER SAYS. Every turn of that session resumed at the last 1,024-token line of the
PREVIOUS PROMPT, whatever the turn generated and whether the monitor acted: 33,792 = 33 x 1,024
after the 34,038-token turn (one hidden close), 66,560 = 65 x 1,024 after the 66,964-token turn
(no act, 10,609 tokens written), and the same on all 15 turns from 17:32 to 17:50 and on the 18:08
session (34,816 after 35,189, 20,526 tokens written). A turn without any hidden token re-read its
own answer the same way, so the hidden phrase is not what decides the resume point.

THE CAUSE. Two things, both by design:
  * the resident prefix takes anchors from prefill chunks only, never from rows a verify round
    wrote (engine/cache.py, ResidentPrefix: "the generated answer is always re-read as prompt on
    the next turn"), and it holds the turn's PROMPT as its tokens, so the next turn shares at most
    that prompt with it and resumes at its last anchor;
  * the session snapshot, which would hold the state after the generation, is declined: a state
    store snapshot is about 151 MB plus 104 KB a token, so turn 1's 66,876 tokens would take about
    7 GB, over the served entry cap (`STORE_ENTRY_GB=1`, about 7,000 tokens) and the whole store
    (`CACHE_GB=2`).
Replaying the hidden phrase cannot move the resume point while neither holds the generated rows.
It matters only once a state after the generation exists: then a client copy without the phrase
parts from it before its end, and the copy with it hits.

The test pins that, on the cache's own path (`cache.prefill` with a resident prefix and a state
store, the session put as `server/app.py::_remember` does it):

  1  a capped store (the served shape): turn 2 resumes at the last anchor of turn 1's prompt,
     with the client's copy missing the hidden phrase AND with it replayed at its position
  2  a store that holds the session snapshot: the copy with the phrase resumes at the end of
     turn 1's generation, the copy without it at the last prompt anchor

Run: python tests/test_reread_after_think.py
"""

from __future__ import annotations

import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import cache  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402

CH = 16
BIG = 1 << 30
PROMPT = 100            # turn 1's prompt: anchors at 16, 32, ..., 96
THINK = 120             # the long thinking, closed by the hidden phrase
PHRASE = tokens(6, seed=41)                       # the vendor's close, in the KV only
CLOSE = tokens(2, seed=42)                        # `</think>` and the blank line
ANSWER = tokens(9, seed=43)                       # the tool call the client receives
TOOL = tokens(25, seed=44)                        # turn 2's new message


def turn1(store):
    """Turn 1 on a fresh engine: the prefill, then a generation written by a forward off the
    prefill's grid (as verify rounds write it), then the session put of `_remember`."""
    res = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG)
    eng = fresh()
    prompt = tokens(PROMPT, seed=40)
    cache.prefill(eng, None, prompt, DEV, store=store, chunk=CH, checkpoint=True, resident=res)
    think = tokens(THINK, seed=45)
    gen = think + PHRASE + CLOSE + ANSWER              # what the engine forwarded
    eng.forward(torch.tensor(gen), start=PROMPT, last_only=True)
    snap = cache.capture(eng, None, max_bytes=store.max_entry)
    if snap is None:
        store.stats["declined_big"] = store.stats.get("declined_big", 0) + 1
    else:
        store.put(prompt + gen, snap, "ses", kind="session")
    return eng, res, prompt, think


def turn2(eng, res, store, ids):
    info = {}
    _, reused, _ = cache.prefill(eng, None, ids, DEV, store=store, chunk=CH, checkpoint=True,
                                   resident=res, info=info)
    return reused, info["kind"]


def test_a_capped_store_resumes_at_the_prompts_last_anchor_with_or_without_the_phrase():
    """The served shape: the session snapshot does not fit, so the phrase changes nothing."""
    for replayed in (False, True):
        one = cache.capture(fresh(), None).nbytes            # the size of an empty snapshot
        store = cache.StateStore(BIG, chunk=CH, max_entry_bytes=one)   # nothing longer fits
        eng, res, prompt, think = turn1(store)
        assert store.stats.get("declined_big", 0) >= 1 and not any(
            e.kind == "session" for e in store._d.values()), store.stats
        copy = think + (PHRASE if replayed else []) + CLOSE + ANSWER
        reused, kind = turn2(eng, res, store, prompt + copy + TOOL)
        assert (reused, kind) == ((PROMPT // CH) * CH, "resident"), (replayed, reused, kind)


def test_with_a_session_snapshot_only_the_copy_with_the_phrase_resumes_at_the_end():
    """What the phrase would buy once a state after the generation exists."""
    store = cache.StateStore(BIG, chunk=CH)
    eng, res, prompt, think = turn1(store)
    end = PROMPT + THINK + len(PHRASE) + len(CLOSE) + len(ANSWER)
    reused, kind = turn2(eng, res, store, prompt + think + PHRASE + CLOSE + ANSWER + TOOL)
    assert (reused, kind) == (end, "session"), (reused, kind)

    store = cache.StateStore(BIG, chunk=CH)
    eng, res, prompt, think = turn1(store)
    reused, kind = turn2(eng, res, store, prompt + think + CLOSE + ANSWER + TOOL)
    assert (reused, kind) == ((PROMPT // CH) * CH, "resident"), (reused, kind)


def test_the_measured_numbers_are_the_previous_prompts_grid_line():
    """The ledger rows of rrr2 and the 18:08 session: cached = the previous prompt's last line."""
    for prev_prompt, cached in ((34038, 33792), (66964, 66560), (77593, 76800), (35189, 34816),
                                (73057, 72704)):
        assert (prev_prompt // 1024) * 1024 == cached
    # the session snapshot turn 1 would have needed, against the served caps
    need = 151e6 + 104e3 * (34038 + 32838)
    assert need > 2 * 2**30 > 1 * 2**30


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
