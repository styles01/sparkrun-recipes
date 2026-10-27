"""Real thinking budgets per effort level (SRV-159, QA SRV-160), on a CPU.

Before this, medium, high and xhigh all had budget 0 (no cap below the backstop) and differed only
by the sentence in the tail note. Now each level's budget is a real cap: medium 8,192, high
16,384, xhigh none below the backstop (D2), low 1,024 as before, off stays thinking off. A
profile's `thinking.budgets` overrides a level for that profile; both documents are versioned by
the store and edited in the Conductor view (Routing and Profiles screens). The cap closes the
reasoning with the session monitor's hidden close (SP1, SRV-109): the vendor's phrase goes into the
KV and the model's context, the client receives `</think>` and what follows. Without a monitor
that acts, ThinkBudget's own streamed phrase closes it, as `low` did before.

The test matrix is SRV-159's Gherkin:

  1  each level's default cap                        test_the_seed_levels_*, test_each_level_*
  2  a profile overrides a level                     test_a_profile_budget_*
  3  an edit is a new version, used by the next turn test_an_edited_level_*
  4  a client's own budget still wins                test_a_client_budget_*
  5  raw is untouched                                test_raw_*
  6  the cap closes hidden                           test_the_route_budget_*, test_over_the_handler_*
  7  bad values are refused                          test_the_schema_refuses_*

Run: python tests/test_think_budgets.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import test_supervisor_frame as frame  # noqa: E402  (sets the CPU kernel switches first)
from test_conductor_router import (  # noqa: E402
    CODE, SYSTEM, body, by_label, setup, turn,
)
from test_app_loop import Req, serve  # noqa: E402
from test_window_edge import FixedDrafter  # noqa: E402

from engine.spec import ThinkBudget  # noqa: E402
from server import app  # noqa: E402
from server.conductor import schema  # noqa: E402
from server.conductor.monitor.policy import SET_THINK_BUDGET, Action  # noqa: E402
from server.conductor.router import COMPOSITE, resolve  # noqa: E402

PHRASE = "Considering the limited time by the user"


def _plan(label, conf=0.9, effort="medium", kind="opencode", headers=None, **kw):
    cond, _ = setup(decider=by_label(label, conf), effort=effort)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE}]
    plan, _, _ = turn(cond, body(msgs, **kw), headers=headers or {"x-session-id": "b-" + label},
                      kind=kind)
    return cond, plan


# ------------------------------------------------------------------ 1. the defaults

def test_the_seed_levels_have_real_caps():
    lv = schema.ROUTER["effort"]["levels"]
    assert {k: (v["thinking"], v["budget"]) for k, v in lv.items()} == {
        "off": (False, 0), "low": (True, 1024), "medium": (True, 8192),
        "high": (True, 16384), "xhigh": (True, 0)}
    assert schema.check("router", schema.seed("router")) == []


def test_each_level_sets_its_cap_on_the_turn():
    """research is medium, plan high, code_edit xhigh, chat low, summarize off."""
    want = {"research": ("medium", 8192), "plan": ("high", 16384), "code_edit": ("xhigh", None),
            "chat": ("low", 1024), "summarize": ("off", None)}
    for cls, (level, cap) in want.items():
        _, plan = _plan(cls)
        row = plan.row
        assert (row["class_id"], row["effort"], row["budget"]) == (cls, level, cap), (cls, row)
        assert plan.budget == cap, cls
        assert plan.think_budget(0) == (cap or 0), cls
        if cap:
            assert row["budget_source"] == "class" and row["acted"] == 1 and plan.changed, cls


def test_the_cap_follows_the_raised_level():
    """A failing checker raises high to xhigh for that turn: the cap goes with the level."""
    cond, _ = setup(decider=by_label("plan", 0.9))
    msgs = [{"role": "user", "content": "Plan the refactor of the lexer."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash",
                                                              "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "RESULT: FAIL  <- (a) 3 > 1"}]
    plan, _, _ = turn(cond, body(msgs))
    assert plan.row["effort"] == "xhigh" and plan.budget is None


# ------------------------------------------------------------------ 2. the profile's override

def _profile_budgets(cond, name, budgets):
    _, doc = cond.store.get("profiles", name)
    doc = copy.deepcopy(doc)
    doc["thinking"]["budgets"] = budgets
    return cond.store.put("profiles", name, doc, note="budgets", author="t")


def test_a_profile_budget_overrides_the_level():
    cond, _ = setup(decider=by_label("plan", 0.9))
    _profile_budgets(cond, "coding", {"high": 4096, "xhigh": 30000})
    msgs = [{"role": "user", "content": "Plan the refactor of the lexer."}]
    plan, _, _ = turn(cond, body(msgs), headers={"x-session-id": "p1"})
    assert (plan.row["effort"], plan.row["budget"], plan.row["budget_source"]) == \
        ("high", 4096, "profile")
    # a level the profile does not name keeps the router's cap; another profile is untouched
    cond2, _ = setup(decider=by_label("code_edit", 0.9))
    _profile_budgets(cond2, "coding", {"high": 4096, "xhigh": 30000})
    plan, _, _ = turn(cond2, body([{"role": "user", "content": CODE}]),
                      headers={"x-session-id": "p2"})
    assert (plan.row["effort"], plan.budget, plan.row["budget_source"]) == \
        ("xhigh", 30000, "profile")
    plan, _, _ = turn(cond2, body([{"role": "user", "content": CODE}]),
                      headers={"x-session-id": "p3", "X-Tandem-Profile": "overnight"})
    assert plan.res.profile == "overnight" and plan.budget is None


def test_a_profile_budget_of_zero_lifts_the_cap():
    cond, _ = setup(decider=by_label("research", 0.9))
    _profile_budgets(cond, "coding", {"medium": 0})
    plan, _, _ = turn(cond, body([{"role": "user", "content": "How do tries compare to maps?"}]),
                      headers={"x-session-id": "z1"})
    assert plan.row["effort"] == "medium" and plan.budget is None and not plan.changed


# ------------------------------------------------------------------ 3. versioned edits

def test_an_edited_level_is_a_new_version_used_by_the_next_turn():
    cond, _ = setup(decider=by_label("research", 0.9))
    v0, doc = cond.store.get("router")
    doc = copy.deepcopy(doc)
    doc["effort"]["levels"]["medium"]["budget"] = 6000
    v1 = cond.store.put("router", None, doc, note="medium 6000", author="t")
    assert (v0, v1) == (0, 1) or v1 > v0
    plan, _, _ = turn(cond, body([{"role": "user", "content": "How do tries compare to maps?"}]),
                      headers={"x-session-id": "v1"})
    assert plan.budget == 6000 and plan.row["budget_source"] == "class"
    cond.store.restore("router", None, 0, note="back", author="t")
    plan, _, _ = turn(cond, body([{"role": "user", "content": "And to hash sets?"}]),
                      headers={"x-session-id": "v2"})
    assert plan.budget == 8192, "a restore brings the seed back"


# ------------------------------------------------------------------ 4. and 5. client and raw

def test_a_client_budget_still_wins_at_every_level():
    for cls in ("research", "plan", "chat"):
        _, plan = _plan(cls, max_reasoning_tokens=777)
        assert plan.row["budget_source"] == "client" and plan.budget is None, cls
        assert plan.think_budget(777) == 777, cls


def test_raw_is_untouched():
    cond, _ = setup(decider=by_label("research", 0.9))
    for hdr in ({"X-Tandem": "raw"}, {"X-Tandem-Profile": "bench"}):
        plan, ids, clean = turn(cond, body([{"role": "user", "content": "How do tries work?"}]),
                                headers=dict(hdr, **{"x-session-id": "raw"}))
        assert not plan.active and plan.budget is None and plan.think_budget(0) == 0
        assert ids == clean and not plan.changed
    assert resolve(cond.store, COMPOSITE, "q", {"X-Tandem": "raw"}, "opencode").mode == "raw"


def test_shadow_journals_the_cap_and_sets_none():
    cond, _ = setup(decider=by_label("plan", 0.9))
    _, doc = cond.store.get("profiles", "coding")
    doc = copy.deepcopy(doc)
    doc["decisions"]["effort"] = "shadow"
    cond.store.put("profiles", "coding", doc, note="effort shadow", author="t")
    plan, _, _ = turn(cond, body([{"role": "user", "content": "Plan the refactor."}]),
                      headers={"x-session-id": "sh"})
    assert plan.budget is None and plan.think_budget(0) == 0
    assert plan.row["shadow"]["effort"]["budget"] == 16384


# ------------------------------------------------------------------ 6. the hidden close

def test_the_route_budget_closes_the_reasoning_hidden():
    mon = frame.monitor("on")
    sup = frame.begin(mon)
    assert sup.route_budget(4) and sup.think_budget == 4 and sup.watches_blocks
    for drafter, tree in ((FixedDrafter(97, 5, tree_mode=False), False),
                          (FixedDrafter(97, 5, tree_mode=True), True), (None, False)):
        sup = frame.begin(mon)
        assert sup.route_budget(4)
        out, _, ctx = frame.run(drafter, tree=tree, sup=sup)
        text = frame.MonTok().decode(out, skip_special_tokens=False)
        inner = frame.MonTok().decode(ctx, skip_special_tokens=False)
        assert "</think>" in text and PHRASE not in text, (tree, text)
        assert PHRASE in inner, "the model's context carries the phrase"
        last = [e for e in mon.journal.actions()][-1]
        assert (last["task"], last["reason"], last["decision"]) == \
            ("route", "budget(4)", "CLOSE_REASONING"), last


def test_the_route_budget_is_refused_where_it_cannot_close_hidden():
    assert not frame.begin(frame.monitor("shadow"), mode="shadow").route_budget(4)
    no_text = frame.monitor("on", texts={"vendor_close": ""})
    assert not frame.begin(no_text).route_budget(4)
    no_closer = frame.monitor("on", texts={"vendor_close": "stop thinking now"})
    assert not frame.begin(no_closer).route_budget(4)
    assert not frame.begin(frame.monitor("on")).route_budget(0)


def test_a_smaller_task_budget_stays():
    def task(ev, t, sup):
        return Action(SET_THINK_BUDGET, reason="t", budget=3, text=frame.CLOSE_TEXT)

    with frame.rule("t_cap", "SP4", task):
        mon = frame.monitor("on", {"T": {"rule": "t_cap", "mode": "act"}})
        sup = frame.begin(mon)
        sup.at_request(sup.req)
        assert sup.route_budget(8192) and sup.think_budget == 3
        sup = frame.begin(frame.monitor("on"))
        assert sup.route_budget(10) and sup.route_budget(6) and sup.route_budget(9)
        assert sup.think_budget == 6


class _Res:
    mode, profile = "on", "coding"
    doc: dict = {}                                   # the profile: no `ui` (state lines off)

    def headers(self):
        return ()


class _Plan:
    """The handler's view of a router plan with a class budget, and nothing else."""

    def __init__(self, budget):
        self.res, self.active, self.session, self.changed = _Res(), True, None, True
        self.budget, self.clean_ids = budget, None
        self.row = {"extra": {}}
        self.transcript = None                       # SRV-137: no session transcript here
        self.context = None                          # no in-context stage due
        self.ask, self.acts, self.outcome = None, [], None   # J1: no outcome, no act
        self.side = None                             # S1: not a client's side request

    def think_budget(self, budget):
        return self.budget if self.budget is not None else budget

    def apply_prompt(self, prompt, tok, in_think):
        return prompt, in_think

    def headers(self):
        return ()


class _Conductor:
    def __init__(self, budget):
        self.budget = budget

    def plan(self, body, headers, **kw):
        return _Plan(self.budget)

    def monitor_mode(self, plan, server_mode):
        return server_mode

    def monitor_policy(self, plan, base):
        return None

    def finish(self, plan, rec):
        pass


def _over_the_handler(mode, budget, n=1):
    serve(None)
    app.STATE["tok"] = frame.MonTok()
    app.STATE["conductor"] = _Conductor(budget)
    if mode is not None:
        app.STATE["monitor"] = frame.monitor(mode)
    with frame.quiet():
        _, raw = Req("/v1/chat/completions", {"messages": [
            {"role": "user", "content": "hello"}], "max_tokens": 160, "n": n,
            "reasoning_format": "tags"}).response()   # SRV-245: the default is reasoning_content
    return json.loads(raw)


def test_over_the_handler_the_class_cap_closes_hidden():
    got = _over_the_handler("on", 4)
    text = got["choices"][0]["message"]["content"]
    assert "</think>" in text and PHRASE not in text, text
    acts = app.STATE["monitor"].journal.actions()
    assert [(e["task"], e["reason"]) for e in acts] == [("route", "budget(4)")], acts


def test_over_the_handler_without_an_acting_monitor_the_phrase_streams():
    """No monitor, a monitor in shadow, or `n` > 1: ThinkBudget's own close, as `low` had."""
    for mode, n in ((None, 1), ("shadow", 1), ("on", 2)):
        got = _over_the_handler(mode, 4, n=n)
        for ch in got["choices"]:
            text = ch["message"]["content"]
            assert PHRASE in text and "</think>" in text, (mode, n, text)


def test_thinkbudget_alone_is_unchanged():
    """The engine's own cap (a client's `max_reasoning_tokens`) still streams its phrase."""
    out, _, _ = frame.run(FixedDrafter(97, 5, tree_mode=False), budget=4, max_new=160)
    assert PHRASE in frame.MonTok().decode(out, skip_special_tokens=False)
    assert ThinkBudget.PHRASE.startswith("\n\n" + PHRASE)


# ------------------------------------------------------------------ 7. validation

def test_the_schema_refuses_bad_budgets():
    doc = schema.seed("router")
    for bad in (-1, 262145, 1.5, "8k"):
        d = copy.deepcopy(doc)
        d["effort"]["levels"]["high"]["budget"] = bad
        assert schema.check("router", d), bad
    prof = schema.seed("profiles", "coding")
    for bad in ({"off": 100}, {"high": -5}, {"high": 300000}, {"turbo": 10}):
        d = copy.deepcopy(prof)
        d["thinking"]["budgets"] = bad
        assert schema.check("profiles", d), bad
    d = copy.deepcopy(prof)
    d["thinking"]["budgets"] = {"low": 512, "medium": 4096, "high": 0, "xhigh": 262144}
    assert schema.check("profiles", d) == []


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
