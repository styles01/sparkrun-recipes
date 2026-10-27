"""The wide drafter arm alone (SPD-75, QA SPD-76), on stub drafters.

Under `QWEN38_LEN_MODE=wide` the length router releases the narrow (block-8) arm at the first block
of every request; `--narrow-arm off` builds no narrow arm at all and gives the router a placeholder
(`engine.lenrouter.NoArm`) that starts every request released. What has to hold: the same choices,
the same node budgets and the same proposals as the router with both arms -- the policy is the
same code -- and nothing of the narrow arm in a snapshot or a resident row. Any other mode refuses
the placeholder, because it would choose an arm that is not there.

Run: python tests/test_narrow_arm.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.lenrouter import LengthRouter, NoArm  # noqa: E402
from tests.test_lenrouter import FakeDrafter, FakeEng  # noqa: E402
from tests.test_lenrouter_price import SERVED, ServedArm  # noqa: E402
from tests.test_lenrouter_switch import CopyNgram  # noqa: E402


def build(narrow: bool, **kw):
    eng = FakeEng()
    small = FakeDrafter(eng, 8) if narrow else NoArm(eng, block=8)
    large, ng = FakeDrafter(eng, 16), CopyNgram()
    kw.setdefault("switch_mode", "wide")
    r = LengthRouter(ServedArm(small, ng, 15) if narrow else small, ServedArm(large, ng, 23),
                     tree=True, ngram=ng, latch_table=SERVED, learn_cost=False, latch=True,
                     drop_idle=True, tree_wide_after=32, switch=True, **kw)
    return r, small, large


def drive(r, blocks=12, k=31):
    keys, budgets, trees = [], [], []
    for req in range(2):                     # two requests: the reset brings the narrow arm back
        r.reset()
        r.prime(list(range(40)))
        for b in range(blocks):
            t = r.propose_tree(list(range(50 + b)), k if b < blocks - 1 else 9)
            keys.append(r.last_key)
            budgets.append(r.large.node_budget)
            trees.append(None if t is None else (tuple(t.tokens), tuple(t.parents)))
            r.observe([9000 + i for i in range(5)] + [12345])
    return keys, budgets, trees


def test_the_wide_policy_proposes_the_same_with_or_without_the_narrow_arm():
    both, small, _ = build(True)
    alone, _, _ = build(False)
    a, b = drive(both), drive(alone)
    assert a[0] == b[0] and set(a[0]) == {"l"}, (a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]
    # with both arms the narrow one was built, released and never asked
    assert small.calls == 0 and small.released


def test_without_the_narrow_arm_every_request_starts_released():
    r, _, _ = build(False)
    for _ in range(2):
        r.reset()
        assert r.idle == "s" and r.stats["idle"] == "s" and r.stats["latched"] == "l"
    snap = r.state_snapshot()
    assert snap[0] == "lenrouter" and snap[1] is None and snap[3] == "s"
    assert r.small.kv_views() == [], "no narrow rows in a stash, pin or park"
    # the placeholder never keeps the router from resuming in place
    assert hasattr(r.small, "state_resume") and not hasattr(r.small, "sync")
    r.state_resume(64)                       # the narrow arm is not asked to resume


def test_the_placeholder_is_refused_outside_the_wide_mode():
    for mode in ("b", "calc", "narrow"):
        try:
            build(False, switch_mode=mode)
        except ValueError as exc:
            assert "QWEN38_LEN_MODE=wide" in str(exc)
        else:
            raise AssertionError(f"mode {mode} took a router without its narrow arm")


def test_the_flag_parses_and_defaults_on():
    from server.app import parser
    assert parser().parse_args([]).narrow_arm == "on"
    assert parser().parse_args(["--narrow-arm", "off"]).narrow_arm == "off"


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
