"""E2, the claim check (2026-10-02): a coding turn that ends by calling the work done is asked
whether its own tool evidence supports that, before the answer's end token reaches the client.

Why: in three Mario sessions on 2026-10-02 Qwen ended with "Done ... verified in-browser" after a
test that set `keys['ArrowRight'] = true` instead of pressing a key, read "lost all 3 lives in
1.5 s" as "the damage system works", and teleported Mario to the flag to test the win.

THE TRIGGER is structural (Tier 0, reads no words): the turn's class is one of `classes`, the
request offered tools, the model wrote its end token after a non-empty answer with no tool call,
and the session has a write behind it (a call of `write_tools` in the request's history, or one
the monitor recorded). Once a request; after `per_session` notes on one session the check is
`claim_check_exhausted` and the message goes out as written.

THE QUESTIONS are Qwen's own, thinking off, one letter each (the in-context stage's form,
`server/conductor/incontext.py`: every option is one letter, one token, and the softmax over the
letters of the last row is the whole constrained answer, what `structured_outputs.choice` leaves).
They cannot go through the server's port as the router's Qwen stage does: the request holds the
engine lock at SP2, so a call to itself would wait for its own lock. They are asked in place
instead, inside the conversation the KV holds, as a hidden user turn after the answer:

    Q1 `claim`     does the message the model just wrote tell the user the work is finished,
                   working or verified?  claim | no_claim; a claim at `claim_at` or above
    Q2 `evidence`  only on a claim: what do the tool calls and results since the last write say?
                   supported | no_run_after_last_write | errors_in_evidence |
                   surprise_unexplained | test_faked; taken at `admit_at` or above

The server forwards the question after the KV's rows, reads the letters and puts the engine back
(`server/app.py` `_ask_in_place`): the recurrent state copied back, the KV's length reset. Nothing
of it reaches the client, the context or the drafter. An answer other than `supported` continues
the turn hidden at SP2 with that answer's note (`notes`), as the empty-turn guard's hidden user
turn: the end token is dropped, the note and an empty closed reasoning block are forwarded, and
the model goes on in the same response. Every other outcome takes the end token as it was drawn:
no claim, `supported`, an answer below its threshold, a failure (fail-open), the caps, shadow.

The structural twin of Q2's `no_run_after_last_write`: no call after the last write matches
`run_tools`. It is journaled beside Qwen's answer, never decides.

THE OUTCOME LABEL: for every note (sent, or in shadow the one that would have been), the session
keeps the request and the time; once `label_calls` tool calls followed it, or the next claim check
of the session triggers, one PASS row (`sp` label) says what came of it: whether a run-like call
followed, whether a write (a fix) followed, and whether the claim came again.
"""

from __future__ import annotations

import re

from server.conductor import incontext
from server.conductor.monitor.policy import CONTINUE_HIDDEN, PASS, Action
from server.conductor.monitor.tier0 import register

ANSWERS = ("supported", "no_run_after_last_write", "errors_in_evidence", "surprise_unexplained",
           "test_faked")
NOTED = ANSWERS[1:]
# what the journal's row keeps of the history: the newest names after the last write
SINCE_NAMES = 12

_TEXT = {"type": "string", "minLength": 1}
_PROB = {"type": "number", "minimum": 0, "maximum": 1}


def _options(keys, what: str) -> dict:
    return {"type": "object", "description": what,
            "properties": {k: _TEXT for k in keys}, "required": list(keys),
            "additionalProperties": False}


PARAMS = {
    "classes": {"type": "array", "items": _TEXT, "minItems": 1,
                "description": "the router's classes a turn must be in (the session's class on a "
                               "tool-result turn)"},
    "write_tools": {"type": "array", "items": _TEXT, "minItems": 1,
                    "description": "the tools that write or edit a file"},
    "run_tools": dict(_TEXT, description="a pattern on a tool's name: a call that runs, loads, "
                                         "plays or tests something (the structural twin of "
                                         "no_run_after_last_write, and the outcome's `run`)"),
    "claim_question": dict(_TEXT, description="Q1, a hidden user turn after the answer"),
    "claim_options": _options(("claim", "no_claim"), "Q1's options, one letter each in this "
                                                     "order"),
    "evidence_question": dict(_TEXT, description="Q2, asked on a claim; {since} is the tools "
                                                 "called since the last write, by name"),
    "evidence_options": _options(ANSWERS, "Q2's options, one letter each in this order"),
    "notes": _options(NOTED, "the hidden user turn's text for each answer but supported"),
    "claim_at": dict(_PROB, description="Q1's yes is a claim at this probability or above"),
    "admit_at": dict(_PROB, description="Q2's answer is taken at this probability or above; "
                                        "below it the claim goes out as written"),
    "per_session": {"type": "integer", "minimum": 1,
                    "description": "notes a session gets; then claim_check_exhausted"},
    "label_calls": {"type": "integer", "minimum": 1,
                    "description": "tool calls after a note before its outcome is labelled"},
}


def _calls(messages) -> list[tuple[str, str]]:
    """Every tool call of the request's history, in order: (name, call id)."""
    out = []
    for m in messages or ():
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or ():
            fn = (c or {}).get("function") or {}
            if isinstance(fn.get("name"), str):
                out.append((fn["name"], str(c.get("id") or "")))
    return out


def evidence(messages, write_tools, run_tools: str, recorded=()) -> dict:
    """The structural facts of the history: the writes, the calls since the last one by name, and
    how many of them run something. `recorded`: tool names the session recorded (a history that a
    compaction shortened still has its writes there)."""
    calls = _calls(messages)
    writes = [i for i, (n, _) in enumerate(calls) if n in write_tools]
    since = [n for n, _ in calls[writes[-1] + 1:]] if writes else []
    rx = re.compile(run_tools)
    runs = sum(1 for n in since if rx.search(n))
    return {"writes": len(writes) + (0 if writes else
                                     sum(1 for n in recorded if n in write_tools)),
            "since": since, "runs_since": runs,
            "twin": "no_run_after_last_write" if writes and not runs else (
                "ran" if writes else "unknown")}


def since_text(since: list[str]) -> str:
    if not since:
        return "none"
    head = since[-SINCE_NAMES:]
    more = len(since) - len(head)
    return ", ".join(head) + (f" (and {more} earlier)" if more else "") + f"; {len(since)} calls"


def _has_call(ev, sup) -> bool:
    from server.toolcall import answer_items
    return bool(answer_items(ev.answer, sup.tool_names or None, eos=True, types=sup.tool_types))


def _state(sup, task) -> dict:
    return sup.session.tasks.setdefault(task.id, {"notes": 0, "open": []})


@register("claim_check", "SP2", params=PARAMS, required=tuple(PARAMS))
def claim_check(ev, task, sup):
    """E2: a coding turn ends on a done claim -> Qwen checks it against the tool evidence."""
    req = sup.req
    if (ev.kind != "eos" or ev.inside or not ev.answer.strip() or req.n != 1
            or not req.tools_offered or req.constrained or req.cls not in task["classes"]
            or sup.memo.get(task.id)):
        return None
    if _has_call(ev, sup):
        return None
    facts = evidence(req.messages, task["write_tools"], task["run_tools"],
                     [c[1] for c in sup.session.calls])
    if not facts["writes"]:
        return None
    sup.memo[task.id] = True
    st = _state(sup, task)
    for o in st["open"]:
        o["reclaimed"] = True           # the claim came again: its outcome is labelled at begin
    facts = dict(facts, since=since_text(facts["since"]), cls=req.cls)
    if st["notes"] >= task["per_session"]:
        return Action(PASS, reason="claim_check_exhausted", detail=facts)
    # the questions need the engine: the loop asks them where the KV holds the whole context
    # (`Supervisor.settle_claim`), so the action goes back to it with nothing to forward yet
    return Action(CONTINUE_HIDDEN, reason="claim_check", detail=dict(facts, ask=True))


def ask(task, facts: dict, asker, ids_of, letter_ids) -> dict:
    """Q1, and on a claim Q2, through `asker(ids, letter ids) -> the letters' logits`. `ids_of`
    gives the hidden turn's token ids for a question, `letter_ids` the token of each letter (one
    each, `incontext.letter_ids`). Returns the answers: `claim` and `evidence` as `{label,
    confidence, probs}`, `answer` (the Q2 label that stands, or None) and `why` when none
    stands. A failure raises; the supervisor fails open."""
    out: dict = {"claim": None, "evidence": None, "answer": None, "why": None}
    q1 = _one(task["claim_question"], task["claim_options"], asker, ids_of, letter_ids)
    out["claim"] = q1
    if q1["label"] != "claim" or q1["confidence"] < float(task["claim_at"]):
        out["why"] = "no_claim" if q1["label"] != "claim" else "claim_below"
        return out
    q2 = _one(task["evidence_question"].replace("{since}", facts.get("since", "none")),
              task["evidence_options"], asker, ids_of, letter_ids)
    out["evidence"] = q2
    if q2["label"] == "supported":
        out["why"] = "supported"
    elif q2["confidence"] < float(task["admit_at"]):
        out["why"] = "evidence_below"
    else:
        out["answer"] = q2["label"]
    return out


def _one(question: str, options: dict, asker, ids_of, letter_ids) -> dict:
    listing, letters = incontext.listing(options)
    logits = asker(ids_of(question.rstrip() + "\n\n" + listing), letter_ids(letters))
    label, conf, probs = incontext.answer(logits, list(options))
    return {"label": label, "confidence": round(conf, 4),
            "probs": {k: round(v, 4) for k, v in probs.items()}}


def opened(sup, task, request_id: str, at: float, mode: str, answer: str) -> None:
    """A note was sent (or in shadow would have been): its outcome is labelled later."""
    st = _state(sup, task)
    st["notes"] += 1
    st["open"].append({"request_id": request_id, "at": at, "mode": mode, "answer": answer,
                       "reclaimed": False})


def label(session, policy, journal, now: float) -> None:
    """At a request's start: the outcome of every note of this session whose window closed --
    `label_calls` calls after it, or the claim came again -- as one PASS row at `sp` label."""
    for tid, task in policy.tasks.items():
        if task.rule != "claim_check":
            continue
        st = session.tasks.get(tid)
        if not st or not st.get("open"):
            continue
        rx = re.compile(task["run_tools"])
        keep = []
        for o in st["open"]:
            names = [n for (ts, n, _) in session.calls if ts >= o["at"]]
            if len(names) < task["label_calls"] and not o["reclaimed"]:
                keep.append(o)
                continue
            run = any(rx.search(n) for n in names)
            fix = any(n in task["write_tools"] for n in names)
            outcome = "fixed" if fix else ("ran_no_fix" if run else "no_run")
            journal.write({"request_id": o["request_id"], "session": session.key, "sp": "label",
                           "task": tid, "rule": "claim_check", "tier": 1, "mode": o["mode"],
                           "decision": PASS, "reason": f"claim_outcome({outcome})",
                           "detail": {"answer": o["answer"], "calls": len(names), "run": run,
                                      "fix": fix, "reclaimed": o["reclaimed"],
                                      "minutes": round((now - o["at"]) / 60.0, 1)},
                           "policy_version": policy.version, "ts": now})
        st["open"] = keep
