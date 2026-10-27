"""Tier 0: the deterministic rules. Counters and patterns decide, on the request's thread.

A rule is a function `fn(event, task, sup) -> Action | None`, registered under a name with the
safe point it runs at and the JSON Schema of the parameters a task gives it. A task in the policy
file names its rule, so a new task is a new entry in the file, and a new kind of rule is one
`@register` here (or in the module that owns it: registering at import is all it takes). The
policy's schema is built from this registry.

A rule never waits, never reads logits or engine state, and returns None or PASS for "nothing
to do". PASS with a reason is journaled as an observation (the history arm's grey band).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from server.conductor.monitor.policy import (
    PASS, CLOSE_REASONING, CONTINUE_HIDDEN, GIVE_UP, Action,
)


@dataclass(frozen=True)
class Rule:
    name: str
    safe_point: str
    fn: Callable
    params: dict                  # JSON Schema properties of the task's parameters
    required: tuple
    doc: str


RULES: dict[str, Rule] = {}


def register(name: str, safe_point: str, params: dict, required: tuple = (), doc: str = ""):
    """Add a rule to the registry; the policy schema picks it up."""
    def deco(fn):
        RULES[name] = Rule(name, safe_point, fn, params, tuple(required),
                           doc or next(iter((fn.__doc__ or "").strip().splitlines()), ""))
        return fn
    return deco


_TEXT = {"type": "string", "minLength": 1}
_CLOSE = dict(_TEXT, description="hidden close phrase; it must contain the reasoning closer "
                                 "(templates.think_close), and only what follows it is streamed")
_SHARE = {"type": "number", "minimum": 0, "maximum": 1}
_POS = {"type": "integer", "minimum": 1}

# ThinkBudget's detector arms (engine/spec.py): a task of rule `reasoning_loop` carries them.
ARMS = {
    "type": "object",
    "description": "the reasoning-loop detector's arms; an arm left out does not run",
    "properties": {
        "loop": {
            "type": "object",
            "description": "a pattern of min_size..max_size tokens repeated back to back; it counts "
                           "only when the repeats span min_span tokens or more (count = "
                           "max(min_count, ceil(min_span / size)))",
            "properties": {"min_size": _POS, "max_size": _POS, "min_count": _POS,
                           "min_span": _POS},
            "required": ["min_size", "max_size", "min_count", "min_span"],
            "additionalProperties": False,
        },
        "repeat": {
            "type": "object",
            "description": "history arm: the share of the last `window` tokens' `gram`-grams that "
                           "already occurred earlier in the block, checked every `every` tokens "
                           "after `min_tokens`; fires at `act` or above on `in_a_row` checks in a "
                           "row, reports `grey` to `act` as an observation",
            "properties": {"gram": _POS, "window": _POS, "every": _POS,
                           "min_tokens": {"type": "integer", "minimum": 0},
                           "act": _SHARE, "grey": _SHARE, "in_a_row": _POS},
            "required": ["gram", "window", "every", "min_tokens", "act", "grey", "in_a_row"],
            "additionalProperties": False,
        },
        "novelty": {
            "type": "object",
            "description": "the share of new `gram`-grams in the last `window` tokens against "
                           "the window before; fires below `below` after `min_tokens`",
            "properties": {"window": _POS, "gram": _POS,
                           "min_tokens": {"type": "integer", "minimum": 0}, "below": _SHARE},
            "required": ["window", "gram", "min_tokens", "below"],
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


# The same arms over words, for a backend behind the proxy whose tokens it never sees (proxy P4,
# server/conductor/monitor/tier0_words.py). The engine never reads this block.
_WPOS = {"type": "integer", "minimum": 0}
WORDS = {
    "type": "object",
    "description": "the word-level twin of `arms` for relayed backends (proxy P4): numbers "
                   "measured on words by tools/loop_words_fp.py, never the token ones; "
                   "`calibrated: false` keeps every verdict in shadow",
    "properties": {
        "calibrated": {"type": "boolean"},
        "loop": {"type": "object", "properties": {"min_size": _POS, "max_size": _POS,
                                                  "min_count": _POS, "min_span": _POS},
                 "required": ["min_size", "max_size", "min_count", "min_span"],
                 "additionalProperties": False},
        "repeat": {"type": "object", "properties": {"gram": _POS, "window": _POS, "every": _POS,
                                                    "min_words": _WPOS, "act": _SHARE,
                                                    "grey": _SHARE, "in_a_row": _POS},
                   "required": ["gram", "window", "every", "min_words", "act", "grey",
                                "in_a_row"],
                   "additionalProperties": False},
        "novelty": {"type": "object", "properties": {"window": _POS, "gram": _POS,
                                                     "min_words": _WPOS, "below": _SHARE},
                    "required": ["window", "gram", "min_words", "below"],
                    "additionalProperties": False},
        "measured": {"type": "object", "description": "the calibration run's date and counts"},
    },
    "required": ["calibrated"],
    "additionalProperties": False,
}


@register("reasoning_loop", "SP1", params={"close": _CLOSE, "arms": ARMS, "words": WORDS},
          required=("close", "arms"))
def reasoning_loop(ev, task, sup):
    """D1: the reasoning is going in circles (a ThinkBudget arm fired) -> close it."""
    if not ev.inside:
        return None
    if ev.stall is not None:
        return Action(CLOSE_REASONING, reason=ev.stall, text=task["close"],
                      score=ev.share, detail={"at": ev.reasoning_tokens})
    rep = (task["arms"] or {}).get("repeat")
    if (rep and ev.share is not None and rep["grey"] <= ev.share < rep["act"]
            and not sup.memo.get(task.id + ":grey")):
        sup.memo[task.id + ":grey"] = True           # once a request: the band was entered
        return Action(PASS, reason=f"repeat16_grey(share={ev.share:.2f})", score=ev.share,
                      detail={"at": ev.reasoning_tokens})
    return None


@register("reasoning_backstop", "SP1",
          params={"close": _CLOSE,
                  "cap": dict(_POS, description="the most reasoning tokens a request may write"),
                  "reserve": dict(_POS, description="tokens of max_tokens kept for the answer")},
          required=("close", "cap", "reserve"))
def reasoning_backstop(ev, task, sup):
    """D2: close the reasoning at min(cap, max_tokens - reserve), so an answer has room."""
    limit = min(task["cap"], sup.req.max_tokens - task["reserve"])
    if ev.inside and limit > 0 and ev.reasoning_tokens >= limit:
        return Action(CLOSE_REASONING, reason=f"backstop({limit})", text=task["close"],
                      detail={"at": ev.reasoning_tokens})
    return None


def _liftable(text: str, sup) -> list[str] | None:
    """The names of the calls in `text` when every block parses and names a tool the request
    offers (SRV-199's lift); None otherwise."""
    from server.toolcall import answer_items
    try:
        items = answer_items(text, sup.tool_names or None, eos=True, types=sup.tool_types)
    except Exception:                                              # noqa: BLE001
        return None
    offered = set(sup.tool_names or ())
    if not items or not offered:
        return None
    names = []
    for it in items:
        fn = it.get("function") if isinstance(it, dict) else None
        if not fn or fn.get("name") not in offered:
            return None
        names.append(fn["name"])
    return names


@register("empty_turn", "SP2",
          params={"close": dict(_CLOSE, description="case A, the end token inside the "
                                                    "reasoning: the hidden close"),
                  "nudges": {"type": "array", "items": _TEXT, "minItems": 1,
                             "description": "case B, an empty answer after the reasoning: the "
                                            "hidden user turn's text, the n-th for the n-th act "
                                            "(the last one repeats)"},
                  "tries": dict(_POS, description="acts before GIVE_UP")},
          required=("close", "nudges", "tries"))
def empty_turn(ev, task, sup):
    """E1: a request that offered tools ends with no tool call and no text -> continue hidden.

    Case C (SRV-199): the output ends with complete tool calls written inside the reasoning
    (`ev.calls`), each of a tool the request offers and each parsing. The model's own call tokens
    go out as the answer -- the reasoning closed first, hidden but for its closer, when it is still
    open -- so the client gets the calls the model wrote, with no generation in between. A call
    that does not parse or names another tool is case A or B as before."""
    req = sup.req
    if ev.kind != "eos" or not req.tools_offered or req.n != 1 or ev.answer.strip():
        return None
    done = sup.task_acts.get(task.id, 0)
    if done >= task["tries"]:
        return Action(GIVE_UP, reason=f"empty_turn(tries={done})",
                      detail={"case": "A" if ev.inside else "B"})
    names = _liftable(ev.calls, sup) if getattr(ev, "calls", "") else None
    if names:
        return Action(CONTINUE_HIDDEN, reason="empty_turn(case=C)",
                      text="\n" + sup.policy.templates["think_close"] + "\n\n" if ev.inside
                      else "",
                      detail={"case": "C", "act": done + 1, "calls": names,
                              "inside": bool(ev.inside)})
    if ev.inside:
        return Action(CONTINUE_HIDDEN, reason="empty_turn(case=A)", text=task["close"],
                      detail={"case": "A", "act": done + 1})
    nudges = task["nudges"]
    return Action(CONTINUE_HIDDEN, reason="empty_turn(case=B)",
                  text=nudges[min(done, len(nudges) - 1)], detail={"case": "B", "act": done + 1})
