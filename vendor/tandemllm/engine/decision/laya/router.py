# Portions of this file are from the Laya SDK 0.3.4 (`laya/router.py`: `Router.route`,
# `match_typed_decisions_workflow`, the aliases), Copyright Convai Innovations, licensed under the
# Apache License 2.0 (engine/decision/laya/LICENSE-APACHE-2.0; NOTICE at the repository root).
# Changes: routing only (no loading, the registry loads), the checkpoints named as the engine
# serves them (`laya-english`, `laya-multilingual`, `laya-typed-decisions`).
"""Which Laya checkpoint answers a request: explicit task, then the detected script and language.

The English checkpoint does not degrade gently off English, it collapses (MASSIVE intent at 20
options: 0.100 on Hindi, 0.103 on Korean, random is 0.050) while reporting high confidence, so
script detection is the primary signal: a non-Latin script goes to the multilingual checkpoint,
Latin text that looks like another language too. `typed-decisions` is never picked from the text:
only `task="typed_decisions"` or an explicit model name reaches it (it is fine-tuned on four
synthetic workflows and must not be a silent default).
"""

from __future__ import annotations

from engine.decision.laya.lang import analyse

ENGLISH, MULTI, TYPED = "laya-english", "laya-multilingual", "laya-typed-decisions"

# the names a client may use for each checkpoint (the SDK's aliases and arbiter's)
ALIASES = {
    ENGLISH: ("english", "en"),
    MULTI: ("multilingual", "multi", "ml"),
    TYPED: ("typed-decisions", "typed", "typed_decisions", "decisions"),
}

# the question-id sets of the four typed-decisions workflows (reported, never routed on: the
# SDK's `auto_task_detection` is off by default and stays off here)
WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {"discrepancy_severity", "disposition", "duplicate", "matches_order",
                           "urgency"},
    "security_incidents": {"credential_compromise", "disposition", "severity", "true_positive",
                           "urgency"},
}


def workflow_of(questions) -> str | None:
    ids = set(questions or {})
    for wf, sig in WORKFLOWS.items():
        if ids == sig:
            return wf
    return None


def route(state, questions=None, *, task=None, lang=None, default: str = ENGLISH) -> dict:
    """The routing decision: `model`, `reason`, `detection`, `workflow`. Nothing is loaded."""
    workflow = workflow_of(questions)
    if task is not None:
        t = str(task).strip().lower()
        key = TYPED if t.replace("-", "_") == "typed_decisions" else next(
            (n for n, al in ALIASES.items() if t == n or t in al), None)
        if key is None:
            raise ValueError(f"unknown task {task!r}; use typed_decisions or a checkpoint name")
        return {"model": key, "reason": f"explicit task={task!r}", "detection": None,
                "workflow": workflow}
    if lang is not None:
        key = ENGLISH if str(lang).lower().split("-")[0] in ("en", "eng", "english") else MULTI
        return {"model": key, "reason": f"explicit lang={lang!r}", "detection": None,
                "workflow": workflow}
    det = analyse(state)
    if det["script"] == "unknown":
        key, reason = default, f"no letters detected in state; using default ({default})"
    elif det["script"] != "latin":
        key = MULTI
        reason = ("non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it"
                  % (det["script"], 100 * float(det["non_latin_fraction"])))
    elif not det["is_english"]:
        key, reason = MULTI, f"Latin script but language looks like {det['language']!r}, not English"
    else:
        key, reason = ENGLISH, "English Latin text"
    return {"model": key, "reason": reason, "detection": det, "workflow": workflow}
