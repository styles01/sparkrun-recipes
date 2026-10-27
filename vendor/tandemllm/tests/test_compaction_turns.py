"""After a client compacts a conversation, the class packages come back.

A user's opencode session in ~/rrr6 (2026-10-03): every turn before /compact carried the design
packages (conductor_hidden_tokens 6,502); the compaction request itself was routed as a design turn
(7 packages, high effort, 2,116 reasoning tokens) because opencode 1.18.34 sends it with its
`compaction` agent's prompt as a system message, which the old shape (no system text) ruled out;
and the "continue" after it was classified chat at 0.998, which has no packages, so every turn
after carried none and the model went looking for the Laya endpoint and the design skills on disk.

  1. opencode 1.18.34's compaction request (its agent's system prompt, the history as one user
     message, no tools) is the compaction: class summarize, structural, thinking off, no packages,
     nothing replayed;
  2. the first user message after it keeps the session's class (sticky, after_compaction) and the
     class block goes in again, as on a first turn, at the compacted history's insertion point;
     the tool turn after that replays it; the next new user message is classified as any other;
  3. the shape is exact: tools offered, more messages, another system text are not a compaction;
  4. the old shape (no system text, one long user message) still is.

Run: python tests/test_compaction_turns.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_conductor_router import SYSTEM, body, setup, text_of, turn  # noqa: E402

from server.conductor import signals  # noqa: E402

# opencode 1.18.34's compaction agent prompt, as the binary ships it (the `compaction` agent)
OC_COMPACTION = (
    "You are a context summarization agent. You are given a conversation between a user and an "
    "agent. Your goal is to produce a structured summary matching the format specified so another "
    "coding agent can continue the work. Always follow the exact output structure requested by "
    "the user prompt. Keep every section, preserve exact file paths and identifiers when known, "
    "and prefer terse bullets over paragraphs. Do not continue the conversation. Do not respond "
    "to any questions in the conversation. Only output the structured summary in the exact format "
    "requested by the user prompt. Respond in the same language as the conversation.\n")
H = {"x-session-id": "ses-compact"}


def classifier(answers):
    """A Qwen stage that answers by the user's text: `continue` reads as chat, as live."""
    def fn(text, options):
        for k, v in answers.items():
            if k in text:
                return v
        return ("design_ui", 0.93)
    return fn


def _session(cond):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Build a cinematic product page for AURA One."}]
    p1, ids1, _ = turn(cond, body(list(msgs)), headers=H)
    msgs += [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash",
                                                      "arguments": json.dumps({"command": "ls"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "index.html"}]
    p2, ids2, _ = turn(cond, body(list(msgs)), headers=H)
    return msgs, (p1, ids1), (p2, ids2)


def _compaction_body(history_text: str) -> dict:
    return {"model": "TandemLLM", "stream": True,
            "messages": [{"role": "system", "content": OC_COMPACTION},
                         {"role": "user", "content": history_text}]}


def test_opencode_compaction_is_a_compaction():
    cond, fakes = setup(qwen=classifier({"continue": ("chat", 0.998)}))
    msgs, (p1, _), (p2, _) = _session(cond)
    assert p1.row["class_id"] == "design_ui" and len(p1.injected) == 1
    assert p2.row["source"] == "sticky" and p2.row["replayed"]["count"] == 1
    n_calls = len(fakes["qwen"].calls)
    plan, ids, clean = turn(cond, _compaction_body(
        "Here is the conversation so far: " + "user: build it. assistant: done. " * 200 +
        "Create a new anchored summary from the conversation history above."), headers=H)
    assert plan.compaction, "opencode 1.18.34's compaction request"
    assert (plan.row["class_id"], plan.row["source"]) == ("summarize", "structural")
    assert plan.row["thinking"] == "off" and plan.thinking_off
    assert not plan.injected and not plan.row.get("replayed")
    assert "<runtime-note>" not in text_of(ids), "no class packages in a compaction"
    assert len(fakes["qwen"].calls) == n_calls, "no classifier is asked"
    return "summarize, structural, thinking off, no note"


def test_the_class_and_its_packages_come_back_after_it():
    cond, fakes = setup(qwen=classifier({"continue": ("chat", 0.998), "release plan": ("plan", 0.97)}))
    _session(cond)
    turn(cond, _compaction_body("Here is the conversation so far: " + "x " * 3000), headers=H)
    # opencode's history after /compact: its compaction part as "What did we do so far?", the
    # summary, then what the user typed
    after = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "What did we do so far?"},
             {"role": "assistant", "content": "## Objective\n- Build a cinematic page for AURA One."},
             {"role": "user", "content": "continue"}]
    n_calls = len(fakes["qwen"].calls)
    plan, ids, clean = turn(cond, body(list(after)), headers=H)
    assert (plan.row["class_id"], plan.row["source"], plan.row["kept"]) == \
        ("design_ui", "sticky", "after_compaction"), (plan.row["class_id"], plan.row["source"])
    assert len(fakes["qwen"].calls) == n_calls, "continue is not reclassified"
    assert plan.row["replayed"]["count"] == 0, "the old note is not in the compacted history"
    assert len(plan.injected) == 1 and plan.injected[0]["tokens"] > 0
    assert "# UI design" in plan.injected[0]["text"] and text_of(ids).count("# UI design") == 1
    assert "# UI design" not in text_of(clean)
    # the tool turn after it replays the note at its place
    after += [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "ok"}]
    plan2, ids2, _ = turn(cond, body(list(after)), headers=H)
    assert plan2.row["source"] == "sticky" and not plan2.injected
    assert plan2.row["replayed"]["count"] == 1 and ids2[:len(ids)] == ids
    # the next message a person writes is classified as any other
    after += [{"role": "assistant", "content": "done"},
              {"role": "user", "content": "now write the release plan for it"}]
    plan3, _, _ = turn(cond, body(list(after)), headers=H)
    assert (plan3.row["class_id"], plan3.row["source"]) == ("plan", "qwen"), plan3.row["source"]
    return "design_ui kept for 'continue', the block in again, replayed after, then classified"


def test_without_a_compaction_continue_is_classified():
    # the rule is the compaction's, not the word's: the same "continue" in an uncompacted session
    # is a new user message like any other
    cond, fakes = setup(qwen=classifier({"continue": ("chat", 0.998)}))
    msgs, _, _ = _session(cond)
    msgs += [{"role": "assistant", "content": "done"}, {"role": "user", "content": "continue"}]
    plan, _, _ = turn(cond, body(list(msgs)), headers=H)
    assert (plan.row["class_id"], plan.row["source"]) == ("chat", "qwen")
    return "chat, from qwen"


def test_the_shape_is_exact():
    sh = lambda b: signals.shape(b, ())  # noqa: E731
    ok = _compaction_body("x")
    assert signals.compaction_shape(ok)["id"] == "opencode_compaction" and sh(ok).compaction
    # rewrapped whitespace still matches
    wrapped = json.loads(json.dumps(ok))
    wrapped["messages"][0]["content"] = OC_COMPACTION.replace(". ", ".\n  ")
    assert sh(wrapped).compaction
    # tools offered, a third message, another system text, a user message first: not one
    with_tools = dict(ok, tools=[{"type": "function", "function": {"name": "read"}}])
    longer = dict(ok, messages=ok["messages"] + [{"role": "assistant", "content": "ok"}])
    other = dict(ok, messages=[{"role": "system", "content": SYSTEM}, ok["messages"][1]])
    flipped = dict(ok, messages=list(reversed(ok["messages"])))
    for b in (with_tools, longer, other, flipped):
        assert signals.compaction_shape(b) is None and not sh(b).compaction
    # the old shape: no system text, one long user message, no tools -- opencode's only
    # (SRV-175, UT1 2026-10-06: a pasted document with "summarize" from any other client has the
    # same shape and was journaled as a compaction)
    old = {"messages": [{"role": "user", "content": "Summarize. " + "x " * 1500}]}
    oc = lambda b: signals.shape(b, (), client_kind="opencode")  # noqa: E731
    assert oc(old).compaction and signals.compaction_shape(old) is None
    for kind in (None, "other", "open-webui", "hermes", "dashboard"):
        assert not signals.shape(old, (), client_kind=kind).compaction, kind
    assert not oc({"messages": [{"role": "user", "content": "hi"}]}).compaction
    # opencode 1.18.34's shape is the client's own protocol, whoever's agent string it carries
    assert signals.shape(ok, (), client_kind="other").compaction
    return "opencode 1.18.34's shape and opencode's old one; other clients' pastes and four near " \
           "misses are not"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
