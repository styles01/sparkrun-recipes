"""The reasoning block's budget and the reasoning-loop detector, unit-tested on a CPU.

`ThinkBudget` counts the reasoning and, given the policy's arms, watches it for a loop; the
session monitor decides what a hit does (server/conductor/monitor/). The claims, ENG-177's
Gherkin (ENG-178's matrix):

  1  a paragraph-length loop is closed: the history arm fires on a repeated 300-token paragraph
     after 2,048 tokens, and on the three failed turns of 2026-09-30 before token 3,100
     (tests/fixtures/stall_replay.bin.xz, relabelled token ids, no text)
  2  a hex colour does not trigger the loop arm: `0x000000`, `0+0+0+0+`, `[], [], []`,
     `-4, -4, -4`; a real one-token loop still fires, after 128 tokens
  3  the 52 texts the old detector closed: the loop arm fires on none of them
  4  the 80 long healthy turns: the history arm fires on at most 4
  5  the backstop closes at min(32,768, max_tokens - 2,048) -- 17,952 for max_tokens 20,000
  6  off: no arm computes and nothing closes the block but a budget
  7  shadow: the journal records the arm's reason and nothing is closed

and what did not change: the budget, the literal tags, nothing outside the block, the novelty arm.
Also the arms' cost per block (ENG-177: under 0.05 ms).

Run: python tests/test_think.py
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine.spec import StallArms, ThinkBudget  # noqa: E402
from server.conductor.monitor import (  # noqa: E402
    BlockEvent, Journal, Monitor, RequestEvent, policy,
)

OPEN, CLOSE, TEXT = 100, 101, 102
ARMS = next(t["arms"] for t in policy.load().tasks.values() if t.rule == "reasoning_loop")
FIXTURE = os.path.join(HERE, "fixtures", "stall_replay.bin.xz")


class FakeTok:
    unk_token_id = -1

    def convert_tokens_to_ids(self, text):
        return {"<think>": OPEN, "</think>": CLOSE}.get(text, -1)

    def __call__(self, text, add_special_tokens=False):
        seq = {"<think>": [7, 8, 9], "</think>": [10, 11, 12]}.get(text, [TEXT, TEXT])
        return type("R", (), {"input_ids": seq})()


def budget(**kw) -> ThinkBudget:
    return ThinkBudget(FakeTok(), **kw)


def various(n: int, seed: int = 5):
    x = seed
    out = []
    for _ in range(n):
        x = (1103515245 * x + 12345) % (1 << 31)
        out.append(1000 + x % 100000)
    return out


def armed(arms=None) -> ThinkBudget:
    t = budget(arms=ARMS if arms is None else arms)
    t.start([OPEN])
    return t


def _qwen():
    """The served tokenizer, when this host has it (the literal runs are then real tokens)."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    snap = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots")
    for d in sorted(os.listdir(snap)) if os.path.isdir(snap) else ():
        p = os.path.join(snap, d, "tokenizer.json")
        if os.path.exists(p):
            return Tokenizer.from_file(p)
    return None


# ------------------------------------------------------------------ what did not change

def test_opens_only_when_the_prompt_ends_inside_the_block():
    t = budget()
    t.start([1, 2, OPEN])
    assert t.inside
    t.start([OPEN, CLOSE])
    assert not t.inside, "the prompt closed the block itself"
    t.start([1, 2, 3])
    assert not t.inside


def test_budget_still_fires():
    t = budget(budget=10)
    t.start([OPEN])
    t.observe(various(9))
    assert not t.hit
    t.observe(various(1))
    assert t.hit


def test_literal_tags_arm_the_budget_too():
    # The model sometimes writes the literal characters instead of the special token; a budget
    # watching only the special id never arms (e05-e10, 2026-09-19). Feed the literal sequence.
    t = budget(budget=5)
    t.start([1, 2, 3])                       # no tags in the prompt
    t.observe([7, 8, 9])                     # the literal "<think>"
    assert t.inside, "a literal open tag must arm the block"
    t.observe(various(4))
    assert not t.hit
    t.observe(various(1))
    assert t.hit


def test_literal_close_ends_the_block():
    t = budget()
    t.start([7, 8, 9])                       # the prompt ends with the literal open
    assert t.inside
    t.observe(various(3))
    assert t.closes_in([10, 11, 12]) and not t.closes_in([10, 11]), "a look, not a change"
    assert t.inside
    t.observe([10, 11, 12])                  # the literal "</think>"
    assert t.done and not t.hit


def test_nothing_fires_outside_the_block_or_after_it_closes():
    t = budget(arms=ARMS)
    t.start([1, 2, 3])                       # never inside
    t.observe([7] * 400)
    assert t.reason is None
    t.start([OPEN])
    t.observe([7] * 100)
    t.observe([CLOSE])
    t.observe([7] * 400)
    assert t.done and t.reason is None, "after the block closes there is nothing to watch"


def test_the_novelty_arm_is_unchanged():
    t = armed({"novelty": ARMS["novelty"]})
    block = various(ARMS["novelty"]["window"], seed=7)
    t.observe(block * 3)                     # windows 2 and 3 are identical
    assert t.reason is not None and t.reason.startswith("novelty("), t.reason


def test_varied_text_never_fires():
    t = armed()
    t.observe(various(6000, seed=11))
    assert t.reason is None, f"varied reasoning must not be flagged, reason={t.reason}"
    assert t.share is not None and t.share < 0.05, t.share


# ------------------------------------------------------------------ 1: the paragraph loop

def test_1_a_paragraph_length_loop_is_found_by_the_history_arm():
    """A 300-token paragraph, repeated: neighbouring 128-token windows share almost nothing
    (novelty stays high) and the whole window has been seen before. The history arm fires on
    its second check at 0.7 or more after 2,048 tokens: at 2,112."""
    t = armed()
    para = various(300, seed=3)
    t.observe(para * 12)
    assert t.reason == "repeat16(share=1.00 at=2112)", t.reason
    assert t.hit is False, "the arm only sets the reason; the monitor decides on a close"


def test_1_tonights_three_failed_turns_close_before_token_3100():
    from tools.stall_replay import fires, read_fixture
    meta, sets = read_fixture(FIXTURE)
    assert meta["until"] == "2026-09-30T20:07:36.556Z" and meta["relabelled"] is True
    got = [fires(s, ARMS) for s in sets["failed"]]
    assert [why.split("(")[0] for why, _ in got] == ["repeat16"] * 3, got
    assert [n for _, n in got] == [2112, 2880, 3008], got
    assert all(n < 3100 for _, n in got)


# ------------------------------------------------------------------ 2 and 3: literal runs

def test_2_a_hex_colour_and_other_literal_runs_do_not_fire_the_loop_arm():
    tok = _qwen()
    runs = ["fogShift is 0x000000 (black)", "sum = 0+0+0+0+0", "rows = [], [], [], []",
            "deltas -4, -4, -4, -4"]
    if tok is not None:
        literal = [tok.encode(r, add_special_tokens=False).ids for r in runs]
    else:                                    # the same shapes as bare ids
        literal = [[9, 48, 120] + [48] * 6, [9] + [48, 43] * 5, [9] + [91, 93, 44] * 4,
                   [9] + [45, 52, 44] * 4]
    for lit in literal:
        t = armed({"loop": ARMS["loop"]})
        for k in range(5):                   # five times, in different sentences
            t.observe(various(40, seed=k + 20) + lit)
        assert t.reason is None, (lit, t.reason)
    # the old arm -- a 1..16-token pattern four times, no span -- fired on a hex colour
    old = {"min_size": 1, "max_size": 16, "min_count": 4, "min_span": 1}
    a = StallArms({"loop": old})
    assert any(a.feed(x, n) for n, x in enumerate(various(10) + literal[0], 1))


def test_2_a_real_one_token_loop_still_fires_after_128_tokens():
    t = armed({"loop": ARMS["loop"]})
    t.observe(various(50) + [42] * 127)
    assert t.reason is None
    t.observe([42])
    assert t.reason == "loop(size=1 count=128 span=128)", t.reason
    t = armed({"loop": ARMS["loop"]})
    t.observe(various(50) + [7, 9] * 63)
    assert t.reason is None
    t.observe([7, 9])
    assert t.reason == "loop(size=2 count=64 span=128)", t.reason


def test_3_the_historical_literal_run_closes_do_not_fire():
    from tools.stall_replay import fires, read_fixture
    _, sets = read_fixture(FIXTURE)
    assert len(sets["closes"]) == 52
    hits = [fires(s, {"loop": ARMS["loop"]}) for s in sets["closes"]]
    assert not [h for h in hits if h[0] is not None], hits


# ------------------------------------------------------------------ 4: healthy turns

def test_4_long_healthy_reasoning_is_mostly_left_alone():
    from tools.stall_replay import fires, read_fixture
    _, sets = read_fixture(FIXTURE)
    assert len(sets["healthy"]) == 80
    fired = [n for why, n in (fires(s, {"repeat": ARMS["repeat"]}) for s in sets["healthy"])
             if why is not None]
    assert len(fired) <= 4, fired
    assert len(fired) == 4, "the design measured four; a change here moves the calibration"


# ------------------------------------------------------------------ 5: the backstop

def _sup(max_tokens: int, mode: str = "on"):
    mon = Monitor(mode, policy.load(), FakeTok(), Journal(echo=False))
    req = RequestEvent("r", max_tokens=max_tokens)
    return mon, mon.begin(mon.mode_for(mode), req, [1, 2, 3], None)


def test_5_the_backstop_leaves_room_for_the_answer():
    mon, sup = _sup(20000)
    assert sup.at_block(BlockEvent(17951, True, None, None, 17951), room=1 << 20) is None
    act = sup.at_block(BlockEvent(17952, True, None, None, 17952), room=1 << 20)
    assert act is not None and act.kind == "CLOSE_REASONING", act
    assert act.task == "D2" and act.reason == "backstop(17952)", act
    # the cap: max_tokens 65,536 closes at 32,768
    mon, sup = _sup(65536)
    assert sup.at_block(BlockEvent(32767, True, None, None, 1), room=1 << 20) is None
    assert sup.at_block(BlockEvent(32768, True, None, None, 1), room=1 << 20).reason == \
        "backstop(32768)"
    # a request too small to leave room has no backstop
    mon, sup = _sup(1500)
    assert sup.at_block(BlockEvent(1400, True, None, None, 1), room=1 << 20) is None


# ------------------------------------------------------------------ 6 and 7: off and shadow

def test_6_off_computes_nothing_and_a_budget_still_closes():
    t = budget()                              # raw: no arms
    t.start([OPEN])
    para = various(300, seed=3)
    t.observe(para * 12 + [42] * 300)
    assert t.reason is None and t.share is None and not t.hit
    t = budget(budget=100)                    # a client's max_reasoning_tokens
    t.start([OPEN])
    t.observe(para)
    assert t.hit, "the client's own budget still closes the block"


def test_7_shadow_records_the_arm_and_closes_nothing():
    mon, sup = _sup(32768, mode="shadow")
    t = budget(arms=sup.arms)
    t.start([OPEN])
    t.observe(various(300, seed=3) * 12)
    act = sup.at_block(BlockEvent(t.n, True, t.reason, t.share, t.n), room=1 << 20)
    assert act is None and not sup.acted
    ev = mon.journal.events()[-1]
    assert (ev["mode"], ev["task"], ev["decision"]) == ("shadow", "D1", "CLOSE_REASONING"), ev
    assert ev["reason"].startswith("repeat16(share=1.00"), ev


def test_the_grey_band_is_an_observation_once():
    """A share between 0.4 and 0.7 is journaled for Tier 1, once a request, and closes nothing."""
    mon, sup = _sup(32768)
    for n, share in ((2112, 0.5), (2176, 0.55)):
        assert sup.at_block(BlockEvent(n, True, None, share, n), room=1 << 20) is None
    evs = mon.journal.events()
    assert [e["decision"] for e in evs] == ["PASS"] and evs[0]["reason"].startswith(
        "repeat16_grey"), evs


# ------------------------------------------------------------------ the cost

def test_the_arms_cost_under_0_05_ms_a_block():
    """ENG-177: one hash a token and a window check every 64 tokens. A served block is about
    four tokens; measured over 20,000 tokens of varied reasoning and of a loop."""
    for seq in (various(20000, seed=9), various(300, seed=3) * 67):
        a = StallArms({k: v for k, v in ARMS.items()})
        t0 = time.perf_counter()
        for n, x in enumerate(seq, 1):
            a.feed(x, n)
        per_tok = (time.perf_counter() - t0) / len(seq)
        assert per_tok * 4 < 0.05e-3, f"{per_tok * 4e6:.1f} us a four-token block"
        print(f"  arms: {per_tok * 1e6:.2f} us a token, {per_tok * 4e6:.1f} us a 4-token block, "
              f"{per_tok * 16e6:.1f} us a 16-token block")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
