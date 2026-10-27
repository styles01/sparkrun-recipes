"""A fake generative model for the tests and the fake engine: no weights, no torch.

Every word of its reply is one token, each taking `delay_s`, so the token cap and the time budget
are exact in a test. The reply is `reply(messages)`; by default it is "now " and the first five
words of the event in the narrator's prompt ("Event: ..."), visibly not a template. `fail` makes
`generate` raise; `loads` counts the loads.

Jobs (N2): `begin(ids, ...)` / `advance(job, deadline)` as the Qwen3.5 model's, the reply being
`summary(ids)` (by default "weighed <the thought's first three words>; next it answers", read from
the ids between `encode_around`'s head and tail, one character an id); `prefill_s` is the prefill's
time, spent at a job's first slice. `jobs_seen` keeps every job's prompt ids.
"""

from __future__ import annotations

import re
import time

from engine.generative.base import GenerativeModel, Job

HEAD, TAIL = "<head>Reasoning:\n", "<tail>"


def _default(messages: list[dict]) -> str:
    text = str((messages[-1] or {}).get("content") or "") if messages else ""
    m = re.search(r"^Event: (.+)$", text, re.M)
    return "now " + (" ".join(m.group(1).split()[:5]) if m else "working")


def _summary(ids: list[int]) -> str:
    text = "".join(chr(i) for i in ids if 0 <= i < 0x110000)
    thought = text.split(HEAD, 1)[-1].split(TAIL, 1)[0]
    return "weighed " + " ".join(thought.split()[:3]) + "; next it answers"


class FakeGenerative(GenerativeModel):
    family = "fake"
    owned_by = "tests"
    memory_claim_gib = 0.0
    jobs = True
    same_ids = None

    def __init__(self, name: str = "fake-narrator", *, delay_s: float = 0.0, reply=None,
                 summary=None, prefill_s: float = 0.0, clock=time.monotonic, sleep=time.sleep):
        self.name = name
        self.delay_s = float(delay_s)
        self.prefill_s = float(prefill_s)
        self.reply = reply or _default
        self.summary = summary or _summary
        self.jobs_seen: list[list[int]] = []
        self.clock, self.sleep = clock, sleep
        self.fail = False
        self.loaded = False
        self.loads = 0
        self.calls: list[dict] = []
        self._words: list[str] = []

    def load(self, where) -> None:
        self.loaded = True
        self.loads += 1

    def unload(self) -> None:
        self.loaded = False

    def encode(self, messages: list[dict]) -> list[int]:
        self._words = str(self.reply(messages)).split(" ")
        self.calls.append({"messages": messages})
        return [ord(c) for c in str(messages[-1].get("content") or "")][:512] or [0]

    def generate(self, ids: list[int], max_new: int, deadline: float | None) -> dict:
        if self.fail:
            raise RuntimeError("the fake generative model failed on purpose")
        out, finish = [], "stop"
        if deadline is not None and self.clock() >= deadline:
            return {"ids": [], "finish": "time", "prefilled": 0, "reused": 0}
        for i, _ in enumerate(self._words):
            if self.delay_s:
                self.sleep(self.delay_s)
            out.append(i)
            if len(out) >= max_new and len(out) < len(self._words):
                finish = "length"
                break
            if deadline is not None and self.clock() >= deadline and len(out) < len(self._words):
                finish = "time"
                break
        self.calls[-1]["finish"] = finish
        return {"ids": out, "finish": finish, "prefilled": len(ids), "reused": 0}

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._words[i] for i in ids)

    # ------------------------------------------------------------------ jobs (N2)
    def encode_around(self, messages: list[dict], mark: str,
                      lead: str = "") -> tuple[list[int], list[int]]:
        return [ord(c) for c in HEAD], [ord(c) for c in TAIL]

    def text_ids(self, ids: list[int], decode=None) -> list[int]:
        return [int(t) for t in ids if 0 <= t < 0x110000]

    def encode_text(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def text_of(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if 0 <= i < 0x110000)

    def begin(self, ids: list[int], max_new: int, keep: int = 0) -> Job:
        job = Job(ids, max_new, keep)
        job.words = str(self.summary(list(ids))).split(" ")
        self.jobs_seen.append(list(ids))
        return job

    def advance(self, job: Job, deadline: float | None) -> Job:
        if self.fail:
            raise RuntimeError("the fake generative model failed on purpose")
        if job.pos < len(job.ids):
            if deadline is not None and self.clock() >= deadline:
                return job
            if self.prefill_s:
                self.sleep(self.prefill_s)
            job.pos = len(job.ids)
        while not job.done:
            if len(job.out) >= len(job.words):
                job.finish = "stop"
                break
            if len(job.out) >= job.max_new:
                job.finish = "length"
                break
            if deadline is not None and self.clock() >= deadline:
                break
            if self.delay_s:
                self.sleep(self.delay_s)
            job.out.append(len(job.out))
        return job

    def job_text(self, job: Job) -> str:
        return " ".join(job.words[i] for i in job.out)

    def precision(self) -> str | None:
        return "none (a test double)" if self.loaded else None
