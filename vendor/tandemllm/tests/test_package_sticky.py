"""FX15 (UX3, 2026-10-08): the package question is asked once per class in a session, not once
per user message.

The live case (0.4.6-test, router v21: 34 everyday options on every prose class): every turn of an
Open WebUI chat asked the class question (~440 tokens, slot 1) and then the package question
(~2,360 tokens). W02's "make it shorter" re-asked and got `reply-format`, the package the first
turn's note already held (3.3 s on slot 0); W05 lost 1.3-1.5 s a turn.

The test matrix:

  1  a follow-up the class question keeps in the session's class takes the package picked for
     that class (`how: sticky`); the package question is not asked again
  2  the class question moves the session to another class: the package question is asked again
  3  after an answer of none the next message asks again (it may fit a package)
  4  a class without `policy.package_sticky` (the coding classes) asks on every new user message,
     as 0.4.6 did; `classifier.package_sticky` is the classes' default
  5  the schema takes `package_sticky` (class, message) on a class policy and on the classifier and
     refuses another value; the shipped router sets `class` on the five prose classes only

Run: python tests/test_package_sticky.py
"""

from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_class_core import PAGODA, make  # noqa: E402
from test_conductor_router import COMPOSITE, TOOLS, turn  # noqa: E402

from server.conductor import schema  # noqa: E402


class Script:
    """The class question answers `cls[0]`; the package question answers `pkg[0]`."""

    def __init__(self, cls="three_d", pkg=("photo-to-3d", 0.93)):
        self.cls, self.pkg = [cls], [pkg]

    def __call__(self, state, options, prompt):
        if "none" in options:
            label, p = self.pkg[0]
        else:
            label, p = self.cls[0], 0.95
        rest = [o for o in options if o != label]
        probs = {label: p, **{o: (1 - p) / max(1, len(rest)) for o in rest}}
        return label, probs


def _session(cond, sid, texts):
    """The turns of one session, each a new user message after an answer."""
    msgs, plans = [], []
    for t in texts:
        msgs.append({"role": "user", "content": t})
        plan, _, _ = turn(cond, {"model": COMPOSITE, "tools": TOOLS, "messages": list(msgs)},
                          headers={"x-session-id": sid})
        plans.append(plan)
        msgs.append({"role": "assistant", "content": "Done: " + t[:20]})
    return plans


def _asked(q) -> int:
    return sum(1 for _, _, o in q.calls if "none" in o)


def _set_sticky(cond, mode, where="three_d"):
    """`mode` on the class `where`'s policy, or on the classifier when `where` is None."""
    _, r = cond.store.get("router")
    r = copy.deepcopy(r)
    if where is None:
        r["classifier"]["package_sticky"] = mode
    else:
        for c in r["classes"]:
            if c["id"] == where:
                c["policy"]["package_sticky"] = mode
    cond.store.put("router", None, r, note="fx15", author="t")


def _make(s):
    cond, q = make(qwen_fn=s)
    _set_sticky(cond, "class")
    return cond, q


# ------------------------------------------------------------------ 1
def test_a_follow_up_in_the_same_class_keeps_the_pick_without_asking():
    s = Script()
    cond, q = _make(s)
    p1, = _session(cond, "s-fx15-1", [PAGODA])
    assert p1.row["extra"]["package_choice"]["chosen"] == "photo-to-3d"
    assert _asked(q) == 1
    cond2, q2 = _make(s)
    plans = _session(cond2, "s-fx15-1b", [PAGODA, "Make the pagoda taller.", "Add a koi pond."])
    for p in plans[1:]:
        ch = p.row["extra"]["package_choice"]
        assert ch["chosen"] == "photo-to-3d" and ch["how"] == "sticky", ch
        assert p.row["class_id"] == "three_d"
    assert _asked(q2) == 1, "the package question ran on a follow-up"


# ------------------------------------------------------------------ 2
def test_a_new_class_asks_again():
    s = Script()
    cond, q = _make(s)
    _session(cond, "s-fx15-2", [PAGODA])
    assert _asked(q) == 1
    s.cls[0] = "chat"
    s.pkg[0] = ("none", 0.9)
    msgs = [{"role": "user", "content": PAGODA}, {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Unrelated: why is Canberra the capital of Australia?"}]
    plan, _, _ = turn(cond, {"model": COMPOSITE, "tools": TOOLS, "messages": msgs},
                      headers={"x-session-id": "s-fx15-2"})
    assert plan.row["class_id"] == "chat"
    ch = plan.row["extra"].get("package_choice")
    if ch is not None and len(ch["candidates"]) > 1:
        assert ch["how"] != "sticky" and _asked(q) == 2, (ch, _asked(q))
    # back in three_d: the class changed again, so the question is asked again
    s.cls[0], s.pkg[0] = "three_d", ("voxel-seasons", 0.9)
    msgs += [{"role": "assistant", "content": "Because of a compromise."},
             {"role": "user", "content": "Back to the garden: make it winter."}]
    n = _asked(q)
    plan, _, _ = turn(cond, {"model": COMPOSITE, "tools": TOOLS, "messages": msgs},
                      headers={"x-session-id": "s-fx15-2"})
    ch = plan.row["extra"]["package_choice"]
    assert ch["chosen"] == "voxel-seasons" and ch["how"] != "sticky" and _asked(q) == n + 1, ch


# ------------------------------------------------------------------ 3
def test_after_none_the_next_message_asks_again():
    s = Script(pkg=("none", 0.9))
    cond, q = _make(s)
    plans = _session(cond, "s-fx15-3", [PAGODA, "Now build it with voxels."])
    assert plans[0].row["extra"]["package_choice"]["chosen"] is None
    assert _asked(q) == 2
    s.pkg[0] = ("voxel-seasons", 0.9)
    plans = _session(cond, "s-fx15-3b", [PAGODA])
    assert plans[0].row["extra"]["package_choice"]["chosen"] == "voxel-seasons"


# ------------------------------------------------------------------ 4
def test_message_mode_asks_on_every_message():
    s = Script()
    cond, q = make(qwen_fn=s)                    # no policy: `message`, as 0.4.6 and coding
    plans = _session(cond, "s-fx15-4", [PAGODA, "Make the pagoda taller.", "Add a koi pond."])
    assert _asked(q) == 3
    assert all(p.row["extra"]["package_choice"]["how"] != "sticky" for p in plans)
    # the classifier's default reaches a class without its own
    cond, q = make(qwen_fn=s)
    _set_sticky(cond, "class", where=None)
    _session(cond, "s-fx15-4b", [PAGODA, "Make the pagoda taller."])
    assert _asked(q) == 1
    # the class's own value wins over the classifier's
    cond, q = make(qwen_fn=s)
    _set_sticky(cond, "class", where=None)
    _set_sticky(cond, "message")
    _session(cond, "s-fx15-4c", [PAGODA, "Make the pagoda taller."])
    assert _asked(q) == 2


# ------------------------------------------------------------------ 5
def test_schema():
    pk = copy.deepcopy(schema.SHIPPED_PACKAGES)
    r = copy.deepcopy(schema.SHIPPED_ROUTER)
    assert "package_sticky" not in r["classifier"] and schema.package_sticky(r) == "message"
    sticky = {c["id"] for c in r["classes"]
              if schema.package_sticky(r, c["policy"]) == "class"}
    assert sticky == {"writing", "plan", "research", "summarize", "chat"}, sticky
    for mode in ("class", "message"):
        r2 = copy.deepcopy(r)
        r2["classifier"]["package_sticky"] = mode
        r2["classes"][0]["policy"]["package_sticky"] = mode
        assert schema.check("router", r2, packages=pk) == [], schema.check("router", r2,
                                                                          packages=pk)
        assert schema.package_sticky(r2) == mode
    for where in ("classifier", "policy"):
        r2 = copy.deepcopy(r)
        (r2["classifier"] if where == "classifier" else r2["classes"][0]["policy"])[
            "package_sticky"] = "turn"
        assert any("package_sticky" in e for e in schema.check("router", r2, packages=pk)), where


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
