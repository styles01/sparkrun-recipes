"""The loop escape (SRV-475): an agent session that repeats one call samples until it moves on.

THE TRAP. Greedy decoding is a function of the prompt. In an agent loop a tool result that does not
help leaves the conversation looking almost as it did a few steps earlier, so the model can pick
the same call again, and again: nothing pushes it onto another path. Sampling does. A third-party
OpenCode run on 0.8.3 (greedy: OpenCode sends no sampling for a Qwen model id, and the server
default is temperature 0) looped past 1,000 tool calls on two tasks; whether greedy decoding was the
cause is not established. C3 (the same-call guard, toolcheck.py) refuses the third identical call;
this module takes the decoding off the greedy fixed point as well.

THE RULE. Per request, from the client's own messages (the current user turn: a user message starts
over) and the session's tool waits:

  * repeat       the last `repeats - 1` calls (C3's parameter, 2 by default) are one call, the
                 same tool and arguments, with identical results: the state C3 fires on;
  * near_repeat  the last NEAR_CALLS calls (3) are the same tool with identical results and each
                 earlier call's arguments at least toolcheck.NEAR_RATIO (0.8) similar to the
                 newest's: an edit retried with a slightly different old string, a search with one
                 word changed, the same failing command with another flag.

A result that came back after the client took C3's `wait_s` (10 s) or more is a wait (a poll of a
running job), not a repeat. Calls with different arguments and the same result (three files
written, each answered "ok") are progress: nothing here reads a result's words.

WHAT IT DOES. While the rule holds, a request that offers tools and names no sampling parameter
decodes with the model card's sampling, read from the checkpoint's generation_config.json at the
start (Qwen3.8-27B: temperature 1.0, top_p 0.95, top_k 20), with a fresh seed recorded in the
journal; and C3's escape note goes at the end of the prompt (executor.py `_escape_note`). Once the
rule no longer holds (a different call, a new result, a user message) the session is back on the
profile's default. The journal row says why, from which turn and back at which turn; the usage
frame's `tandem.sampling` says how the request sampled.

WHAT NEVER CHANGES. A request whose client names `temperature`, `top_p`, `top_k`, `min_p` or `seed`
is served as sent. Raw and bench requests, side requests and compactions are never looked at. A
request without the rule holding decodes as before, greedy unless the profile's
`sampling.agentic_default` is `card`. The sampled path is the engine's exact one (engine/sample.py):
under speculation the target draws its token at each node, so the answer follows the target's own
sampled distribution, drafter or not.
"""

from __future__ import annotations

import collections
import json
import os
import secrets
import threading

from server.conductor.monitor import toolcheck

NEAR_CALLS = 3            # near-repeats: this many calls of one tool in a row
REPEATS = 3               # C3's `repeats` when the request has no policy
WAIT_S = 10.0             # C3's `wait_s` when the request has no policy
CARD_KEYS = ("temperature", "top_p", "top_k")


def card_sampling(model_dir: str | None) -> dict | None:
    """The checkpoint's own sampling (generation_config.json): {temperature, top_p, top_k,
    source}, or None when the file is missing, says `do_sample: false` or has no temperature."""
    if not model_dir:
        return None
    path = os.path.join(model_dir, "generation_config.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or cfg.get("do_sample") is False:
        return None
    try:
        t = float(cfg["temperature"])
        p = float(cfg.get("top_p", 1.0))
        k = int(cfg.get("top_k", 0) or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if not 0.0 < t <= 2.0 or not 0.0 < p <= 1.0 or k < 0:
        return None
    return {"temperature": t, "top_p": p, "top_k": k, "source": path}


def waits_now(session, messages, now: float) -> dict:
    """The session's tool waits (seconds by call id) with the results this request carries,
    timed from the session's last request end (`SessionState.gap`)."""
    out = dict(getattr(session, "waits", None) or {})
    gap = session.gap(now) if session is not None and hasattr(session, "gap") else None
    if gap is None:
        return out
    for m in reversed(messages or []):
        if not isinstance(m, dict) or m.get("role") != "tool":
            break
        cid = str(m.get("tool_call_id") or "")
        if cid:
            out.setdefault(cid, gap)
    return out


def signal(messages, waits: dict | None = None, repeats: int = REPEATS,
           wait_s: float = WAIT_S) -> dict | None:
    """{why, name, count} when the current user turn ends in a repeat or a near-repeat, else
    None (the module's docstring has the rule)."""
    hist = toolcheck.turn_history(messages)
    got = toolcheck.streak(hist, max(1, int(repeats) - 1), waits=waits, wait_s=wait_s)
    if got is not None:
        return {"why": "repeat", "name": got[-1]["name"], "count": len(got)}
    got = toolcheck.streak(hist, NEAR_CALLS, waits=waits, wait_s=wait_s,
                           near=toolcheck.NEAR_RATIO)
    if got is not None:
        return {"why": "near_repeat", "name": got[-1]["name"], "count": len(got)}
    return None


def turn_of(messages) -> int:
    """The request's turn in the conversation: its assistant messages plus one."""
    return 1 + sum(1 for m in messages or () if isinstance(m, dict)
                   and m.get("role") == "assistant")


class Escapes:
    """Per session, the escape in progress: the turn it started and how many requests it has
    sampled. Bounded LRU; a session that left it is forgotten."""

    def __init__(self, sessions: int = 1024):
        self.max = int(sessions)
        self._d: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()
        self.counts = {"started": 0, "sampled": 0, "back": 0}

    def step(self, key: str, sig: dict | None, turn: int) -> dict | None:
        """This request's escape record: `state` on (with `since_turn`, `turn`, `nth`, the n-th
        sampled request of the streak, from 0) while `sig` holds, `back` (with `since_turn` and
        `back_at_turn`) on the first request after it; None otherwise."""
        with self._lock:
            cur = self._d.get(key)
            if sig is None:
                if cur is None:
                    return None
                del self._d[key]
                self.counts["back"] += 1
                return {"state": "back", "why": cur["why"], "name": cur["name"],
                        "since_turn": cur["since"], "back_at_turn": turn,
                        "sampled": cur["n"]}
            if cur is None:
                cur = {"since": turn, "n": 0, "why": sig["why"], "name": sig["name"]}
                self.counts["started"] += 1
            else:
                cur["why"], cur["name"] = sig["why"], sig["name"]
            self._d[key] = cur
            self._d.move_to_end(key)
            while len(self._d) > self.max:
                self._d.popitem(last=False)
            out = {"state": "on", "why": sig["why"], "name": sig["name"],
                   "count": sig["count"], "since_turn": cur["since"], "turn": turn,
                   "nth": cur["n"]}
            cur["n"] += 1
            self.counts["sampled"] += 1
            return out

    def report(self) -> dict:
        with self._lock:
            return {"sessions": len(self._d), **self.counts}


def sampling_of(card: dict | None, fallback: dict | None) -> dict | None:
    """The escape's (and `agentic_default: card`'s) sampling with a fresh seed: the card's, else
    `fallback` (the server's regenerate sampling) marked as such; None when neither exists."""
    src = card or fallback
    if not src:
        return None
    out = {k: src[k] for k in CARD_KEYS}
    out["seed"] = secrets.randbits(63)
    out["source"] = src.get("source") if card else "regenerate"
    return out


def profile_sampling(doc: dict | None) -> dict:
    """A profile's `sampling` section with its defaults (absent: greedy, escape off)."""
    got = (doc or {}).get("sampling") or {}
    return {"agentic_default": got.get("agentic_default", "greedy"),
            "loop_escape": got.get("loop_escape", "off")}
