"""One Laya checkpoint as a decision model the registry serves.

The three checkpoints of the HF repo `convaiinnovations/laya` (Apache-2.0), laid out as the SDK
downloads it under `QWEN38_DECIDER_DIR`: the repo root is English (ModernBERT-large 421M,
`max_len` 512, `head_max_len` 192), `multilingual/` is mmBERT-base 322M (1,024 and 256, 100+
languages), `typed-decisions/` the English encoder fine-tuned on four workflows (1,024 and 256).

The arithmetic is the SDK's, by construction (stage A): fp32 parameters, and on CUDA the forward
under `torch.autocast` in the checkpoint's `amp_dtype` (bf16 on this board), exactly the call
`laya.Agent.system_one` makes; on the CPU plain fp32, as the SDK runs there. Arbiter measured this
mode at 0.00 against the SDK on the GB10; BF16 parameters everywhere were 2.1e-2 with one argmax
flip, which is why stage A does not store the weights in BF16 (stage B tries BF16 storage of the
matmul weights only, behind the same gate).

The rows of one forward may come from several callers; they are independent, so batching changes
nothing beyond floating-point reassociation (`tools/decider_equivalence.py` measures it).
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os

from engine.decision.base import DecisionModel, OptionBudgetError
from engine.decision.laya import encoder as enc_mod
from engine.decision.laya.router import ALIASES, ENGLISH, MULTI, TYPED
from engine.decision.laya.sequence import (QTYPES, answer, build_sequence, collate,
                                           render_options, to_internal)

SUBFOLDER = {ENGLISH: None, MULTI: "multilingual", TYPED: "typed-decisions"}
NAMES = tuple(SUBFOLDER)
# the model card's advice: past 20 options they share too little of the question budget to
# stay apart (Banking77, 77 options, falls to 0.425)
MAX_OPTIONS = 20


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class LayaModel(DecisionModel):
    family = "laya"
    contract = "systemone"
    owned_by = "convaiinnovations"
    # the MemAvailable the first checkpoint took beside the engine in stage A (2026-10-01; the
    # three together took 8.2 GiB, the residency document's claim for the decider role)
    memory_claim_gib = 4.4

    def __init__(self, name: str, root: str):
        if name not in SUBFOLDER:
            raise ValueError(f"no Laya checkpoint {name!r}; one of {', '.join(NAMES)}")
        self.name = name
        self.aliases = ALIASES[name]
        root = os.path.expanduser(root)
        self.dir = os.path.join(root, SUBFOLDER[name]) if SUBFOLDER[name] else root
        self.cfg = self._config()
        self.net = self.tok = self.where = None
        self.amp = None            # the autocast dtype, None for plain fp32
        self._hash: str | None = None

    def _config(self) -> dict:
        try:
            with open(os.path.join(self.dir, "rl_agent_config.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def limits(self) -> dict:
        return {"context_tokens": int(self.cfg.get("max_len", 512)),
                "question_tokens": int(self.cfg.get("head_max_len", 192)),
                "max_options": MAX_OPTIONS}

    # ------------------------------------------------------------------ loading
    def load(self, where) -> None:
        import torch
        from safetensors.torch import load_file
        from engine.decision.laya.head import LayaNet

        enc_mod.fix_tokenizer_config(self.dir)
        self.cfg = self._config()
        if not self.cfg:
            raise enc_mod.CheckpointError(f"{self.dir}: no rl_agent_config.json")
        wpath = os.path.join(self.dir, "model.safetensors")
        if not os.path.exists(wpath):
            raise enc_mod.CheckpointError(f"{self.dir}: no model.safetensors")
        tok = enc_mod.tokenizer(self.dir, self.cfg)
        net = LayaNet(enc_mod.encoder(self.dir), self.cfg.get("head_layers", 2),
                      len(self.cfg.get("act_costs", {})) + 1)
        weights = load_file(wpath)
        enc_mod.verify(net, self.cfg, weights, self.dir)
        net.load_state_dict(weights, strict=True)
        del weights
        net.eval()
        device = where.device if where is not None else torch.device("cpu")
        amp = None
        if device.type == "cuda":
            # the SDK: the config's amp dtype (bf16 or fp16), fp16 below compute capability 8
            amp = torch.bfloat16 if self.cfg.get("amp_dtype", "fp16") == "bf16" else torch.float16
            if torch.cuda.get_device_capability(device)[0] < 8:
                amp = torch.float16
            with where.gpu():
                net.to(device)
        self.net, self.tok, self.where, self.amp = net, tok, where, amp
        self._hash = sha256_file(wpath)
        gc.collect()

    def unload(self) -> None:
        self.net = self.tok = self.where = None
        self._hash = None
        gc.collect()

    # ------------------------------------------------------------------ what it reports
    def weight_hash(self) -> str | None:
        return self._hash

    def precision(self) -> str | None:
        if self.net is None:
            return None
        if self.amp is None:
            return "fp32"
        return f"fp32 parameters under autocast({str(self.amp).replace('torch.', '')}), eager"

    def device_bytes(self) -> int:
        if self.net is None:
            return 0
        return sum(t.numel() * t.element_size()
                   for t in list(self.net.parameters()) + list(self.net.buffers()))

    # ------------------------------------------------------------------ rows and answers
    def build_rows(self, state, questions: dict[str, dict]) -> list[dict]:
        if self.tok is None:
            raise RuntimeError(f"{self.name} is not loaded")
        max_len = int(self.cfg.get("max_len", 512))
        head_max_len = int(self.cfg.get("head_max_len", 192))
        rows = []
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise OptionBudgetError(
                    f"question {qid!r} has more options than head_max_len={head_max_len} can "
                    f"hold; split it")
            rows.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "qid": qid,
                         "q": q})
        return rows

    def predict_rows(self, rows: list[dict]) -> tuple[list[dict], list[int], str]:
        import torch
        net, where = self.net, self.where
        if net is None:
            raise RuntimeError(f"{self.name} is not loaded")
        b = collate(rows, self.tok.pad_token_id)
        device = where.device if where is not None else torch.device("cpu")
        scope = where.scope() if where is not None else contextlib.nullcontext()
        with torch.no_grad(), scope:
            ins = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=self.amp or torch.float32,
                                enabled=self.amp is not None):
                logits, act = net(ins["input_ids"], ins["attention_mask"], ins["marker_pos"],
                                  ins["marker_mask"], ins["qtype"])
            logits = logits.float().cpu().numpy()
            act = torch.softmax(act.float(), -1).cpu().numpy()
        temp = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        temp_k = self.cfg.get("temperature_by_options", {})
        out = [answer(r["q"], len(r["markers"]), r["qtype"], logits[i], act[i], temp, temp_k)
               for i, r in enumerate(rows)]
        return out, [len(r["ids"]) for r in rows], "eager"
