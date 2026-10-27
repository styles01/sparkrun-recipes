"""A fake decision model for the tests and the fake engine: no weights, no torch.

It is what proves the runtime model-agnostic: another name, another family, smaller limits (256
and 96 by default, SRV-125's check), and everything above it -- the registry, the batcher, the API,
the conductor's in-process client -- runs unchanged. It is a test double, so words in the state
steer it, which is exactly what a real decision must never do:

    FAKE_CHOICE=<option>   a choice question picks that option at 0.92
    an option id that appears in the state as a word: picked at 0.91
    otherwise              the first option at 0.2 (under every threshold)
    FAKE_TRUE              a noul question answers 0.9, else 0.1

`delay_s` makes each forward take that long (batching and deadline tests); `fail` makes it raise.
Its names are refused outside a test or the fake engine (server/app.py).
"""

from __future__ import annotations

import re
import time

from engine.decision.base import DecisionModel, OptionBudgetError

QTYPES = {"choice": 0, "score": 1, "noul": 2}


class FakeModel(DecisionModel):
    family = "fake"
    owned_by = "tests"
    memory_claim_gib = 0.0                 # no weights

    def __init__(self, name: str = "fake-decider", context_tokens: int = 256,
                 question_tokens: int = 96, max_options: int = 20, delay_s: float = 0.0):
        self.name = name
        self.aliases = ()
        self._limits = {"context_tokens": context_tokens, "question_tokens": question_tokens,
                        "max_options": max_options}
        self.delay_s, self.fail = float(delay_s), False
        self.loaded = False
        self.loads = 0
        self.forwards: list[int] = []       # rows per forward, in order

    def load(self, device) -> None:
        self.loaded = True
        self.loads += 1

    def unload(self) -> None:
        self.loaded = False

    def limits(self) -> dict:
        return dict(self._limits)

    def weight_hash(self) -> str | None:
        return "fake" if self.loaded else None

    def precision(self) -> str | None:
        return "none (a test double)" if self.loaded else None

    def build_rows(self, state, questions: dict[str, dict]) -> list:
        text = state if isinstance(state, str) else repr(state)
        rows = []
        for qid, q in questions.items():
            crit = q.get("criteria")
            opts = (list(crit) if q["type"] == "choice" else
                    [str(i) for i in range(len(crit))] if q["type"] == "score" else ["false", "true"])
            if len(opts) > 4 * self._limits["question_tokens"]:
                raise OptionBudgetError(f"question {qid!r} has more options than {self.name} "
                                        f"can hold")
            rows.append({"qid": qid, "q": q, "opts": opts, "text": text,
                         "tokens": min(len(text) // 4 + 8, self._limits["context_tokens"])})
        return rows

    def predict_rows(self, rows: list) -> tuple[list[dict], list[int], str]:
        if self.fail:
            raise RuntimeError("the fake decision model failed on purpose")
        if self.delay_s:
            time.sleep(self.delay_s)
        self.forwards.append(len(rows))
        return [self._answer(r) for r in rows], [r["tokens"] for r in rows], "eager"

    @staticmethod
    def _answer(row: dict) -> dict:
        q, opts, text = row["q"], row["opts"], row["text"]
        act = {"act_probability": 0.5}
        if q["type"] == "noul":
            p = 0.9 if "FAKE_TRUE" in text else 0.1
            return {"type": "noul", "noul": p, "confidence": max(p, 1 - p), "action": act}
        pick, conf = opts[0], 0.2
        m = re.search(r"FAKE_CHOICE=([A-Za-z0-9_-]+)", text)
        if m and m.group(1) in opts:
            pick, conf = m.group(1), 0.92
        else:
            for o in opts:
                if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(o)}(?![A-Za-z0-9_-])", text):
                    pick, conf = o, 0.91
                    break
        rest = round((1.0 - conf) / max(1, len(opts) - 1), 4)
        probs = {o: (conf if o == pick else rest) for o in opts}
        if q["type"] == "score":
            return {"type": "score", "score": float(opts.index(pick)),
                    "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                    "probabilities": probs, "confidence": conf, "action": act}
        return {"type": "choice", "choice": pick, "probabilities": probs, "confidence": conf,
                "action": act}
