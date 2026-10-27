"""The decision-model interface, and the errors the runtime and the API share.

A decision model answers typed questions over a state in one forward pass: `choice` (one of N
options), `score` (an expected level over 2 to 10 levels) and `noul` (a probability that a
statement holds). The answer object is the Jev shape arbiter serves, so a client written against
arbiter or Jev works against the engine unchanged.

Rows are the model's own: `build_rows` makes one per question, `predict_rows` answers a batch of
them, and nothing above this file looks inside one. That is what keeps the runtime model-agnostic
(arbiter's hard rule, 2026-09-25: any decision model must work): the registry, the batcher and the
API name no checkpoint and no limit, and a fake model with limits 256 and 96 runs through all of
them in the tests.
"""

from __future__ import annotations

from typing import Any


class DecisionError(Exception):
    """A question the runtime could not answer. `why` is the journal's and the API's word for it,
    `status` the HTTP answer, `retry` the `Retry-After` seconds of a 503 (None: no retry)."""

    why = "decider_error"
    status = 500
    retry: int | None = None

    def __init__(self, message: str, **detail):
        super().__init__(message)
        self.detail = detail


class OptionBudgetError(DecisionError, ValueError):
    """A question has more options than the model's question budget can hold markers for."""
    why, status = "option_budget", 422


class OverloadedError(DecisionError):
    """More question rows in flight than the queue allows."""
    why, status = "overloaded", 529


class NotReadyError(DecisionError):
    """The model is loading, warming, refused, failed or not loaded; `why` says which."""
    status, retry = 503, 5

    def __init__(self, why: str, message: str, **detail):
        super().__init__(message, **detail)
        self.why = why


class DeadlineMissed(DecisionError):
    """The question's deadline passed before its forward could start: the engine was serving a
    request the whole time (stage A asks only in the idle window)."""
    why, status, retry = "deadline_missed", 503, 2


class WindowError(DecisionError):
    """A window this stage does not run: `in_decode` needs the low-priority stream and the
    governor (stage C of the design)."""
    why, status = "window_unsupported", 422


class DecisionModel:
    """One decision model the registry can load. Subclasses give the methods marked so.

    `name` is what clients ask for, `family` groups the models one router chooses between (Laya's
    three checkpoints), `aliases` are the other names a client may use, `contract` is
    `systemone` (typed questions over a state) or later `probe` (fixed questions over the
    reasoner's own state).
    """

    name = "decision-model"
    family = "decision-model"
    contract = "systemone"
    owned_by = "local"
    aliases: tuple = ()

    # -- the role contract it signs (SRV-133, Harmony 1) -------------------------
    # The registry checks these against the conductor's role contract before it serves the model
    # (`Registry(contract=...)`): a model without a memory claim is refused like a role without
    # one. Stage A answers in the idle window only, under a 100 ms held-call budget, and its
    # tokenizer is its own, so it shares text with Qwen and nothing else.
    kind = "decider"
    memory_claim_gib: float | None = None
    window = "idle"
    preemptible = True
    latency_budget_ms = 100
    fallback = "PASS"
    tokenizer_sha256 = "none"
    writes = "nothing"
    sharing: tuple = ("text",)

    def role(self) -> dict:
        """The contract fields as the roles document spells them."""
        return {"kind": self.kind, "memory_claim_gib": self.memory_claim_gib,
                "window": self.window, "preemptible": self.preemptible,
                "latency_budget_ms": self.latency_budget_ms, "fallback": self.fallback,
                "tokenizer_sha256": self.tokenizer_sha256, "writes": self.writes,
                "sharing": list(self.sharing)}

    # -- the model's half ------------------------------------------------------
    def load(self, device) -> None:
        """Bring the weights onto `device` (engine/decision/device.py), ready to answer."""
        raise NotImplementedError

    def unload(self) -> None:
        """Drop every tensor this model holds."""
        raise NotImplementedError

    def limits(self) -> dict:
        """`context_tokens` (the longest row it reads), `question_tokens` (the part of the row the
        question and its options may use), `max_options` (a soft limit). Known before load."""
        return {}

    def build_rows(self, state, questions: dict[str, dict]) -> list[Any]:
        """One row per question, in the order the ids were given."""
        raise NotImplementedError

    def predict_rows(self, rows: list[Any]) -> tuple[list[dict], list[int], str]:
        """One batch: the Jev answer and the real token count per row, and the path it took
        (`eager` in stage A)."""
        raise NotImplementedError

    # -- what the registry reports ---------------------------------------------
    def weight_hash(self) -> str | None:
        """A hash of the weights, once loaded: part of every later cache key (stage D)."""
        return None

    def precision(self) -> str | None:
        """How the forward runs, in words, once loaded."""
        return None

    def device_bytes(self) -> int:
        """The bytes its parameters and buffers hold on the device, once loaded."""
        return 0

    def warm_question(self) -> tuple[str, dict]:
        """A state and one question for the warm-up forward after load."""
        return ("warm up the serving path",
                {"warm": {"type": "noul", "instructions": "the text mentions a warm-up"}})
