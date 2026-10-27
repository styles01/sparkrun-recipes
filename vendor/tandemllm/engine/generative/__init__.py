"""Small generative models inside the engine: the slot (P1 of "One engine for all models", 2026-10-02).

The decision models (engine/decision) answer typed questions in one forward. The models here
write text: a few dozen tokens, greedy, with their own KV cache. The first is the narrator
(server/conductor/narrator.py): MiniCPM5-2B wrote the one-line summaries of TandemLLM's states
(P1); since 2026-10-03 the small-model lane is the Qwen3.5 family (0.8B, 2B), which summarises each
stretch of Qwen's thinking (N2). The pieces: the interface and the role contract (base.py), the
Llama-architecture forward in plain PyTorch (llama.py) and a Llama checkpoint directory as a model
(model.py), a Qwen3.5 checkpoint on the engine's own forward (qwen35.py), the slot that loads a
model through the residency planner and runs it inside a time budget (slot.py), and a fake for the
tests (fake.py).

Nothing here touches Qwen: no tensor, stream or allocator pool is shared (the slot uses the
decision models' Device and Placement), and with `QWEN38_NARRATOR_MODEL` empty the package is never
imported.
"""

from __future__ import annotations

import os

# the submodule `model` (a Llama checkpoint directory as a model) is loaded here, before the
# function below is defined: loaded later, the import would rebind the package's `model` to the
# submodule, and a second call of the function (a test, N2) would call a module
from engine.generative import model as _llama_dir  # noqa: E402,F401

FAKE_PREFIXES = ("fake-",)


def model(name: str, path: str | None, *, allow_fake: bool = False,
          engine_dir: str | None = None):
    """The model `QWEN38_NARRATOR_MODEL` names, from `path` (none: its directory under
    ~/.qwen38-spark-engine/models). A fake name only in a test or with the fake
    engine; an unknown name stops the start. `engine_dir`: the served checkpoint, whose tokenizer a
    Qwen3.5 model compares its own with."""
    if name.startswith(FAKE_PREFIXES):
        if not allow_fake:
            raise ValueError(f"{name!r} is a test double; it is served only by a fake engine")
        from engine.generative.fake import FakeGenerative
        return FakeGenerative(name)
    from engine.generative import qwen35
    from engine.generative.model import NAMES, LlamaModel
    path = path or os.path.join(qwen35.DEFAULT_DIR, name)
    if name in qwen35.NAMES:
        return qwen35.Qwen35Model(name, path, engine_dir=engine_dir)
    if name not in NAMES:
        raise ValueError(f"unknown generative model {name!r}; this engine serves "
                         f"{', '.join([*qwen35.NAMES, *NAMES])}")
    return LlamaModel(name, path)
