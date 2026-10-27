"""A first message under the accept bar takes the top class when it builds something (CPU).

Router v17 grew the chat class's description, and the request "Create a new folder and
create there a pagoda garden" read three_d 0.378 instead of 0.464: under the 0.45 bar the turn fell
back to chat, got no 3D skill, wrote an ASCII pagoda into a file and the argument loop guard
stopped it twice. `classifier.below_accept` (mode `build`, `margin`) keeps such a message on its
top class when that class builds something (its policy names packages) and leads the second by the
margin; anything else falls back as before.

  1  the pagoda's own answer (v17's probabilities): three_d, kept `below_build`
  2  a top class that builds nothing (chat, research, writing ...) falls back
  3  a lead under the margin falls back
  4  a session that already has a class keeps the old rule (no first message)
  5  mode `default` and a document without the key: the old rule
  6  an errored stage or no probabilities: the old rule
  7  the shipped router carries the measured setting and validates
  8  the reading line fires for a turn kept on its top build class (its stage is under the bar),
     and still not for a below-bar stage of a turn that was not kept that way

Run: python tests/test_router_below.py
"""

from __future__ import annotations

import copy
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor import schema  # noqa: E402
from server.conductor.router import Router  # noqa: E402

CLASSES = {c["id"]: c for c in copy.deepcopy(schema.SHIPPED_ROUTER["classes"])}
PAGODA = {"three_d": 0.3784, "design_ui": 0.2561, "code_edit": 0.2321, "drawing": 0.1043,
          "plan": 0.0248, "chat": 0.0031, "research": 0.0005, "writing": 0.0002}


def router(mode="build", margin=0.1):
    r = copy.deepcopy(schema.SHIPPED_ROUTER)
    r["classifier"]["below_accept"] = {"mode": mode, "margin": margin}
    return r


def stage(probs, error=None, name="qwen"):
    return [{"stage": name, "probs": probs, "error": error, "below": True,
             "label": max(probs, key=probs.get) if probs else None}]


NEW = types.SimpleNamespace(cls=None)
OLD = types.SimpleNamespace(cls="code_edit")


def test_1_the_pagoda_stays_three_d():
    got = Router._below_build(stage(PAGODA), router(), NEW, CLASSES)
    assert got == ("three_d", 0.3784, "qwen", "below_build"), got
    ctx = Router._below_build(stage(PAGODA, name="qwen_context"), router(), NEW, CLASSES)
    assert ctx[2] == "qwen_context"


def test_2_a_class_that_builds_nothing_falls_back():
    for cls in ("chat", "research", "writing", "plan", "summarize"):
        # PKG3: the prose classes name everyday packages, but no files-care: they build nothing
        assert "files-care" not in (CLASSES[cls]["policy"].get("packages") or []), cls
        probs = {cls: 0.40, "code_edit": 0.20, "chat": 0.1}
        probs[cls] = 0.40
        assert Router._below_build(stage(probs), router(), NEW, CLASSES) is None, cls


def test_3_a_lead_under_the_margin_falls_back():
    close = {"three_d": 0.30, "design_ui": 0.25, "chat": 0.2}
    assert Router._below_build(stage(close), router(margin=0.1), NEW, CLASSES) is None
    assert Router._below_build(stage(close), router(margin=0.04), NEW, CLASSES)[0] == "three_d"


def test_4_a_session_with_a_class_keeps_the_old_rule():
    assert Router._below_build(stage(PAGODA), router(), OLD, CLASSES) is None


def test_5_mode_default_and_no_key_are_the_old_rule():
    assert Router._below_build(stage(PAGODA), router(mode="default"), NEW, CLASSES) is None
    r = router()
    del r["classifier"]["below_accept"]
    assert Router._below_build(stage(PAGODA), r, NEW, CLASSES) is None


def test_6_an_error_or_no_probabilities_is_the_old_rule():
    assert Router._below_build(stage(PAGODA, error="timeout"), router(), NEW, CLASSES) is None
    assert Router._below_build(stage({}), router(), NEW, CLASSES) is None
    assert Router._below_build([], router(), NEW, CLASSES) is None


def test_7_the_shipped_router_carries_the_setting():
    rule = schema.SHIPPED_ROUTER["classifier"].get("below_accept")
    assert rule and rule["mode"] == "build", rule
    assert not schema.doc_errors("router", schema.SHIPPED_ROUTER)
    bad = copy.deepcopy(schema.SHIPPED_ROUTER)
    bad["classifier"]["below_accept"] = {"mode": "top"}
    assert schema.doc_errors("router", bad)


class _Plan:
    decisions = {"classify": "on"}
    class_new = True


def test_8_the_reading_line_reads_the_kept_stage():
    r = router()
    st = stage(PAGODA)
    row = {"class_id": "three_d", "confidence": 0.3784, "source": "qwen", "kept": "below_build",
           "stages": st, "extra": {}}
    part = Router._reading(None, _Plan(), r, "three_d", CLASSES, row)
    assert part and part[1]["kind"] == "reading", part
    assert "3D three.js scene" in part[1]["text"] and "flat web page" in part[1]["text"], part
    assert row["extra"]["reading"]["other"] == "design_ui"
    row2 = dict(row, kept=None, extra={})
    assert Router._reading(None, _Plan(), r, "three_d", CLASSES, row2) is None


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
