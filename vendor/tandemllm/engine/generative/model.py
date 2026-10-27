"""A Llama-architecture checkpoint directory as a generative model of the slot (MiniCPM5).

    <dir>/config.json  generation_config.json  *.safetensors  tokenizer.json  tokenizer_config.json
          chat_template.jinja

the layout `huggingface_hub.snapshot_download("openbmb/MiniCPM5-2B", local_dir=<dir>)` writes
(QWEN38_NARRATOR_DIR). The tokenizer and its chat template are the checkpoint's own, through
transformers' AutoTokenizer as the engine reads Qwen's; the forward is engine/generative/llama.py.
Thinking off is the template's `enable_thinking=False`: the generation prompt ends in an empty,
closed `<think>` block, so the model answers at once.

Loading copies the weights onto the device in batches of about 512 MiB, each under the
Placement's `gpu()` scope -- the engine's lock, so in the idle window, and the model's own memory
pool, so Qwen's allocator never sees them -- read from the file outside the lock. The host holds
one batch at a time, never the whole checkpoint beside the device copy.
"""

from __future__ import annotations

import contextlib
import gc
import glob
import hashlib
import json
import os
import struct

from engine.generative.base import GenerativeModel

# name -> the Hugging Face repo its directory is a snapshot of, and the memory it claims beside
# the engine (declared: the BF16 weights, the 1,024-row cache, the step's workspace and the pool's
# slack; `loaded` in the residency journal records what it took)
NAMES = {
    "minicpm5-2b": {"repo": "openbmb/MiniCPM5-2B", "claim_gib": 6.0},
    "minicpm5-1b": {"repo": "openbmb/MiniCPM5-1B", "claim_gib": 3.5},
}
MAX_CTX = 1024
BATCH_BYTES = 512 << 20


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def header_hash(paths: list[str]) -> str:
    """The safetensors headers (every tensor's name, dtype, shape and offsets) and sizes: what
    identifies the weights without reading 5 GB."""
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h.update(f.read(n))
        h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()


class LlamaModel(GenerativeModel):
    family = "llama"

    def __init__(self, name: str, path: str, *, max_ctx: int = MAX_CTX,
                 claim_gib: float | None = None):
        self.name = name
        self.dir = os.path.expanduser(path)
        meta = NAMES.get(name) or {}
        self.repo = meta.get("repo")
        self.memory_claim_gib = claim_gib if claim_gib is not None else meta.get("claim_gib")
        self.owned_by = (self.repo or "local").split("/")[0]
        self.max_ctx = int(max_ctx)
        tj = os.path.join(self.dir, "tokenizer.json")
        self.tokenizer_sha256 = sha256_file(tj) if os.path.isfile(tj) else "none"
        self.net = self.tok = self.where = None
        self.eos: set[int] = set()
        self._hash: str | None = None

    def files(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.dir, "*.safetensors")))

    # ------------------------------------------------------------------ loading
    def load(self, where) -> None:
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer
        from engine.generative.llama import Config, Llama

        cfg = Config.read(self.dir)
        files = self.files()
        if not files:
            raise FileNotFoundError(f"{self.dir}: no *.safetensors")
        tok = AutoTokenizer.from_pretrained(self.dir)
        device = where.device if where is not None else torch.device("cpu")
        gpu = where.gpu if where is not None else contextlib.nullcontext
        weights: dict = {}
        batch, size = {}, 0

        def flush():
            nonlocal batch, size
            if batch:
                with gpu():
                    for k, t in batch.items():
                        weights[k] = t.to(device, non_blocking=False)
            batch, size = {}, 0

        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    batch[k] = t
                    size += t.numel() * t.element_size()
                    if size >= BATCH_BYTES:
                        flush()
        flush()
        with gpu():
            net = Llama(cfg, weights, self.max_ctx)
        eos = set()
        for src in ("generation_config.json", "config.json"):
            p = os.path.join(self.dir, src)
            if os.path.isfile(p):
                with open(p) as fh:
                    e = json.load(fh).get("eos_token_id")
                if e is not None:
                    eos |= set(e if isinstance(e, list) else [e])
                    break
        if tok.eos_token_id is not None:
            eos.add(int(tok.eos_token_id))
        self.net, self.tok, self.where, self.eos = net, tok, where, eos
        self._hash = header_hash(files)
        gc.collect()

    def unload(self) -> None:
        self.net = self.tok = self.where = None
        self._hash = None
        gc.collect()

    # ------------------------------------------------------------------ text
    def encode(self, messages: list[dict]) -> list[int]:
        if self.tok is None:
            raise RuntimeError(f"{self.name} is not loaded")
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def generate(self, ids: list[int], max_new: int, deadline: float | None) -> dict:
        import torch
        net, where = self.net, self.where
        if net is None:
            raise RuntimeError(f"{self.name} is not loaded")
        scope = where.scope() if where is not None else contextlib.nullcontext()
        with torch.inference_mode(), scope:
            return net.generate(ids, max_new, self.eos, deadline)

    # ------------------------------------------------------------------ what it reports
    def precision(self) -> str | None:
        if self.net is None:
            return None
        return f"{str(self.net.dtype).replace('torch.', '')} parameters, eager, greedy"

    def device_bytes(self) -> int:
        return self.net.nbytes() if self.net is not None else 0

    def weight_hash(self) -> str | None:
        return self._hash
