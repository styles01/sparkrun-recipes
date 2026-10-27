"""The generative-model interface of the small-model slot, and the role contract it signs.

A generative model here writes a few dozen tokens of text for the conductor -- the narrator's one
line about a state today, a clerk's job later -- with its own tokenizer, its own weights and its
own KV cache. Nothing of Qwen is shared but text (Harmony 2: a different tokenizer shares text
only).

    m.load(where)                     the weights onto `where` (engine/decision/device.Placement)
    m.encode(messages) -> ids         its own chat template, thinking off
    m.generate(ids, max_new, deadline) -> {ids, finish, prefilled, reused}   greedy
    m.decode(ids) -> text

A model that can be advanced in slices (`jobs = True`, the Qwen3.5 family, N2) also has

    job = m.begin(ids, max_new, keep=)    a generation that has not run yet
    m.advance(job, deadline)              prefill chunks and decode steps until the deadline or the end
    m.encode_around(messages, mark, lead) -> (head ids, tail ids)   the template around a
                                      placeholder, the answer begun with `lead`
    m.text_ids(ids) -> ids                Qwen's ids this model reads as the same text

so a caller can spread one generation over several of Qwen's rounds (server/conductor/narrator.py
Thoughts).
"""

from __future__ import annotations


class Job:
    """One generation advanced in slices: its prompt, how far its prefill went, what it wrote.
    `keep`: the prompt's fixed head, whose state the model keeps for the next job that starts
    with it."""

    def __init__(self, ids: list[int], max_new: int, keep: int = 0):
        if not ids:
            raise ValueError("an empty prompt")
        self.ids, self.max_new, self.keep = list(ids), int(max_new), int(keep)
        self.pos = 0                  # prompt rows run
        self.out: list[int] = []
        self.next: int | None = None  # the token the last forward chose, not taken yet
        self.finish: str | None = None   # stop | length, once done
        self.reused = 0
        self.ms = 0.0                 # the slices' time, the slot adds it
        self.slices = 0

    @property
    def done(self) -> bool:
        return self.finish is not None


class GenerativeModel:
    name = "generative-model"
    family = "generative"
    owned_by = "local"

    # -- the role contract it signs (SRV-133, Harmony 1) -------------------------
    # A small generative model inside the engine runs between two of Qwen's rounds, on the
    # request's own thread while Qwen waits, for at most its latency budget per call; it writes
    # the state lines the client is shown in the reasoning (never content, never Qwen's prompt),
    # and on a timeout or an error its caller keeps the templated line (PASS).
    kind = "generative"
    memory_claim_gib: float | None = None
    window = "between_rounds"
    preemptible = True
    latency_budget_ms = 150
    fallback = "PASS"
    tokenizer_sha256 = "none"
    writes = "state_lines"
    sharing: tuple = ("text",)

    def role(self) -> dict:
        return {"kind": self.kind, "memory_claim_gib": self.memory_claim_gib,
                "window": self.window, "preemptible": self.preemptible,
                "latency_budget_ms": self.latency_budget_ms, "fallback": self.fallback,
                "tokenizer_sha256": self.tokenizer_sha256, "writes": self.writes,
                "sharing": list(self.sharing)}

    def load(self, where) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def encode(self, messages: list[dict]) -> list[int]:
        raise NotImplementedError

    def generate(self, ids: list[int], max_new: int, deadline: float | None) -> dict:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    # -- slices (N2): only a model with `jobs = True` --------------------------------------
    jobs = False

    def begin(self, ids: list[int], max_new: int, keep: int = 0) -> Job:
        return Job(ids, max_new, keep)

    def advance(self, job: Job, deadline: float | None) -> Job:
        raise NotImplementedError

    def job_text(self, job: Job) -> str:
        return self.decode(job.out) if job.out else ""

    def encode_around(self, messages: list[dict], mark: str,
                      lead: str = "") -> tuple[list[int], list[int]]:
        raise NotImplementedError

    def text_ids(self, ids: list[int]) -> list[int]:
        raise NotImplementedError

    def encode_text(self, text: str) -> list[int]:
        """A plain text's ids, no template (v3: a tool call's words, the answer's plan)."""
        raise NotImplementedError

    def precision(self) -> str | None:
        return None

    def device_bytes(self) -> int:
        return 0

    def weight_hash(self) -> str | None:
        return None
