"""A Qwen3.5 checkpoint (0.8B, 2B) as a generative model of the slot, on the engine's own forward (N2).

    <dir>/config.json  model.safetensors-*  tokenizer.json  tokenizer_config.json  chat_template.jinja

the layout `huggingface_hub.snapshot_download("Qwen/Qwen3.5-0.8B", local_dir=<dir>)` writes
(QWEN38_NARRATOR_DIR, by default ~/.qwen38-spark-engine/models/<name>). Qwen3.5 is the 27B's family
(`qwen3_5_text`: Gated DeltaNet and gated attention), so `engine.model.Qwen38Engine` runs it from its
BF16 checkpoint with no new layer code: the sizes come from `TextConfig`, the projections load as
`BF16Block`s, the tied head reads the embedding (2026-10-03: teacher-forced argmax
against transformers' Qwen3_5ForCausalLM 98.0-99.6 %, every confident position equal). The
engine's module-level switches (the fused kernels, the decode-attention kernel) are the served
process's, read once at import: the small model runs the same kernels Qwen's decode step runs.

The tokenizer is the 27B's (the same BPE vocabulary and merges; Qwen3.8 adds seven audio tokens),
so a caller hands Qwen's own token ids of a text to this model with no detokenising and no second
tokenisation (`text_ids`, after `same_ids` checked the two tokenizer.json files at load).

GENERATION IN SLICES. `begin(ids, max_new, keep)` makes a job; `advance(job, deadline)` runs prefill
chunks of PREFILL_CHUNK rows and then decode steps until the deadline (checked before every chunk
and every step) or the end, and can be called again between two of Qwen's rounds. The cache holds
one sequence; a job that finds another generation ran on it since starts its prompt over. `keep`:
the prompt's fixed head (the narrator's instructions and examples); after its last row the
recurrent and convolution state is copied (18.9 MB for 0.8B) and the next job whose prompt starts
with the same ids restores it and prefills only the rest -- the KV rows of the head are never
overwritten by such a job.

THE DECODE STEP is a CUDA graph on the GPU: captured once at load, in the idle window, from the
engine's graph-safe body (`_gv`: the position, the KV slot and the length in device buffers, the
attention through `decode_attention_dev`, as engine/verify_graph.py serves Qwen's verify), with the
argmax inside. A capture that fails leaves the eager step (status says why). The weights, the cache
and the step's workspace are in the slot's own memory pool (engine/decision/device.py); the graph
has a private pool of its own; nothing is Qwen's.
"""

from __future__ import annotations

import contextlib
import gc
import glob
import json
import os
import time
import types

from engine.generative.base import GenerativeModel, Job
from engine.generative.model import header_hash, sha256_file

# name -> the Hugging Face repo its directory is a snapshot of, and the memory it claims beside the
# engine (declared: the BF16 text weights, the 4,096-row cache, the step's workspace and graph, the
# pool's slack; `loaded` in the residency journal records what it took)
NAMES = {
    "qwen3.5-0.8b": {"repo": "Qwen/Qwen3.5-0.8B", "claim_gib": 2.5},
    "qwen3.5-2b": {"repo": "Qwen/Qwen3.5-2B", "claim_gib": 5.5},
    # SR1: the shadow reader's larger choice (32 KiB of KV a token, 50 MB of recurrent state)
    "qwen3.5-4b": {"repo": "Qwen/Qwen3.5-4B", "claim_gib": 10.5},
}
# rows of the cache: a power of two (the graph-safe attention's context class), room for the
# narrator's head (~700 tokens), a 1,500-token window of thought and the answer
MAX_CTX = 4096
PREFILL_CHUNK = 256
BATCH_BYTES = 512 << 20
DEFAULT_DIR = "~/.qwen38-spark-engine/models"


def same_ids(small: str, engine: str | None) -> str | None:
    """None when every id of the small model's tokenizer means what it means to the engine's: the
    same BPE vocabulary and merges, and each of its added tokens at the same id in the engine's.
    Else why not (no engine tokenizer to compare with, or the first difference)."""
    if not engine:
        return "no engine tokenizer to compare with"
    paths = [os.path.join(os.path.expanduser(d), "tokenizer.json") for d in (small, engine)]
    for p in paths:
        if not os.path.isfile(p):
            return f"{p} is missing"
    with open(paths[0]) as f:
        a = json.load(f)
    with open(paths[1]) as f:
        b = json.load(f)
    am, bm = a.get("model") or {}, b.get("model") or {}
    if am.get("type") != bm.get("type") or am.get("vocab") != bm.get("vocab"):
        return "the vocabulary differs"
    if am.get("merges") != bm.get("merges"):
        return "the merges differ"
    theirs = {t["id"]: t["content"] for t in b.get("added_tokens") or []}
    for t in a.get("added_tokens") or []:
        if theirs.get(t["id"]) != t["content"]:
            return f"added token {t['id']} ({t['content']!r}) is {theirs.get(t['id'])!r} there"
    return None


class Qwen35Model(GenerativeModel):
    family = "qwen3.5"
    jobs = True

    def __init__(self, name: str, path: str, *, engine_dir: str | None = None,
                 max_ctx: int = MAX_CTX, claim_gib: float | None = None, graph: bool = True,
                 clock=time.monotonic):
        self.name = name
        self.dir = os.path.expanduser(path)
        meta = NAMES.get(name) or {}
        self.repo = meta.get("repo")
        self.memory_claim_gib = claim_gib if claim_gib is not None else meta.get("claim_gib")
        self.owned_by = (self.repo or "local").split("/")[0]
        self.max_ctx = int(max_ctx)
        self.engine_dir = engine_dir
        self.use_graph = graph
        self.clock = clock
        tj = os.path.join(self.dir, "tokenizer.json")
        self.tokenizer_sha256 = sha256_file(tj) if os.path.isfile(tj) else "none"
        self.eng = self.tok = self.where = None
        self.eos: set[int] = set()
        self.added: set[int] = set()
        self.vocab = 0
        self.same_ids: str | None = "not loaded"   # None: Qwen's ids pass as they are
        self.snap = None               # (head ids, recurrent state, convolution state)
        self.owner: Job | None = None  # the job whose rows the cache holds
        self.gs = None                 # the captured decode step
        self.graph_error: str | None = None
        self._hash: str | None = None

    def files(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.dir, "*.safetensors")))

    # ------------------------------------------------------------------ loading
    def load(self, where) -> None:
        import torch
        from transformers import AutoTokenizer
        from engine.config import load_config
        from engine.loader import Weights
        from engine.model import Qwen38Engine

        cfg = load_config(self.dir)
        files = self.files()
        if not files:
            raise FileNotFoundError(f"{self.dir}: no *.safetensors")
        tok = AutoTokenizer.from_pretrained(self.dir)
        device = where.device if where is not None else torch.device("cpu")
        gpu = where.gpu if where is not None else contextlib.nullcontext
        # read on the host, outside the lock; copied in batches under it. Every weight override off
        # explicitly: the served engine's QWEN38_NVFP4 / FP8_HEAD / EXL3 name ITS checkpoints, and the
        # environment is shared (an EXL3 profile's QWEN38_EXL3 put the 27B's trellis blocks in the narrator)
        w = Weights(cfg.path, device="cpu", skip_mtp=True, nvfp4="", fp8_head="", exl3="")
        if device.type != "cpu":
            _to_device(w, device, gpu)
        with gpu():
            eng = Qwen38Engine(cfg, w, max_len=self.max_ctx, device=str(device))
        with open(os.path.join(self.dir, "tokenizer.json")) as f:
            self.added = {t["id"] for t in json.load(f).get("added_tokens") or []}
        eos = set(cfg.eos_token_ids)
        for t in ("<|im_end|>", "<|endoftext|>"):
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
                eos.add(i)
        if tok.eos_token_id is not None:
            eos.add(int(tok.eos_token_id))
        self.eng, self.tok, self.where, self.eos = eng, tok, where, eos
        self.vocab = cfg.vocab_size
        self.same_ids = same_ids(self.dir, self.engine_dir)
        self.snap, self.owner, self.gs, self.graph_error = None, None, None, None
        self._hash = header_hash(files)
        if device.type == "cuda" and self.use_graph:
            self._capture(where)
        gc.collect()

    def unload(self) -> None:
        self.eng = self.tok = self.where = self.gs = None
        self.snap = self.owner = None
        self._hash = None
        gc.collect()

    # ------------------------------------------------------------------ text
    def encode(self, messages: list[dict]) -> list[int]:
        return list(self.tok(self._render(messages), add_special_tokens=False)["input_ids"])

    def _render(self, messages: list[dict]) -> str:
        if self.tok is None:
            raise RuntimeError(f"{self.name} is not loaded")
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def encode_around(self, messages: list[dict], mark: str,
                      lead: str = "") -> tuple[list[int], list[int]]:
        """The rendered prompt's ids before and after `mark`; `lead` begins the answer."""
        text = self._render(messages) + lead
        if text.count(mark) != 1:
            raise ValueError("the placeholder must be in the prompt exactly once")
        head, tail = text.split(mark)
        enc = lambda s: list(self.tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
        return enc(head), enc(tail)

    def text_ids(self, ids: list[int], decode=None) -> list[int]:
        """Qwen's ids of a text as this model's: the same ids without the added (special) ones,
        when `same_ids` held at load; else the text `decode(ids)` gives, tokenised here."""
        if self.same_ids is None:
            return [int(t) for t in ids if 0 <= t < self.vocab and t not in self.added]
        if decode is None:
            raise RuntimeError(f"{self.name}: Qwen's ids are not this model's ({self.same_ids})")
        return list(self.tok(decode(ids), add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def encode_text(self, text: str) -> list[int]:
        if self.tok is None:
            raise RuntimeError(f"{self.name} is not loaded")
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    # ------------------------------------------------------------------ generation
    def begin(self, ids: list[int], max_new: int, keep: int = 0) -> Job:
        if len(ids) + max_new > self.max_ctx:
            raise ValueError(f"a prompt of {len(ids)} tokens and {max_new} new ones do not fit "
                             f"the cache's {self.max_ctx} rows")
        return Job(ids, max_new, keep)

    def generate(self, ids: list[int], max_new: int, deadline: float | None) -> dict:
        job = self.begin(ids, max_new)
        self.advance(job, deadline)
        return {"ids": list(job.out), "finish": job.finish or "time",
                "prefilled": job.pos - job.reused, "reused": job.reused}

    def advance(self, job: Job, deadline: float | None) -> Job:
        import torch
        if self.eng is None:
            raise RuntimeError(f"{self.name} is not loaded")
        scope = self.where.scope() if self.where is not None else contextlib.nullcontext()
        with torch.inference_mode(), scope:
            self._advance(job, deadline)
        return job

    def _advance(self, job: Job, deadline: float | None) -> None:
        import torch
        eng, clock = self.eng, self.clock
        if job.done:
            return
        if self.owner is not job:
            # the first slice, or another generation ran on the cache since: the prompt over
            job.pos, job.out, job.next, job.reused = 0, [], None, 0
            self.owner = job
        ids = job.ids
        if job.pos == 0:
            snap = self.snap
            n = len(snap[0]) if snap is not None else 0
            if snap is not None and len(ids) > n and tuple(ids[:n]) == snap[0]:
                eng.reset()
                eng.state.S.copy_(snap[1])
                eng.state.conv.copy_(snap[2])
                eng.state.primed = True
                eng.kv.length = n
                job.pos = job.reused = n
            else:
                eng.reset()
                self.snap = None                  # its rows are written over from row 0
        while job.pos < len(ids):
            if deadline is not None and clock() >= deadline:
                return
            end = min(len(ids), job.pos + PREFILL_CHUNK)
            if job.keep and job.pos < job.keep:
                end = min(end, job.keep)
            lg = eng.forward(torch.tensor(ids[job.pos:end], device=eng.device), start=job.pos,
                             last_only=True)
            job.pos = end
            if job.keep and end == job.keep and end < len(ids):
                self.snap = (tuple(ids[:end]), eng.state.S.clone(), eng.state.conv.clone())
            if end == len(ids):
                job.next = int(lg[0, -1].argmax())
        while True:
            if job.next is None:
                # the step after the last token taken is due (a slice ended before it)
                if deadline is not None and clock() >= deadline:
                    return
                job.next = self._step(job.out[-1], len(ids) + len(job.out) - 1)
            t, job.next = job.next, None
            if t in self.eos:
                job.finish = "stop"
                return
            job.out.append(t)
            if len(job.out) >= job.max_new or len(ids) + len(job.out) >= self.max_ctx:
                job.finish = "length"
                return

    def _step(self, t: int, pos: int) -> int:
        """The token after `t` at row `pos`: from the graph, or eagerly."""
        import torch
        eng, gs = self.eng, self.gs
        if gs is None:
            lg = eng.forward(torch.tensor([t], device=eng.device), start=pos, last_only=True)
            return int(lg[0, -1].argmax())
        gs.tok.fill_(t)
        gs.pos.fill_(pos)
        gs.slots.fill_(pos)
        gs.lenp[0].fill_(pos)
        gs.lenp[1].fill_(pos + 1)
        gs.graph.replay()
        eng.kv.length = pos + 1
        return int(gs.next)

    def _capture(self, where) -> None:
        """The decode step as a CUDA graph: a warm-up of the graph-safe body in the model's pool
        (the idle window), then the capture into the graph's private pool, under the engine's
        lock and the capture lock. Any failure leaves the eager step."""
        import torch
        from engine import capture_lock
        import engine.model as M
        eng = self.eng
        if M.KV_FP8:
            self.graph_error = "QWEN38_KV_FP8: the graph-safe attention reads a bf16 cache only"
            return
        try:
            with where.gpu():
                dev = eng.device
                gs = types.SimpleNamespace(
                    tok=torch.zeros(1, dtype=torch.long, device=dev),
                    pos=torch.zeros(1, dtype=torch.long, device=dev),
                    slots=torch.zeros(1, dtype=torch.long, device=dev),
                    lenp=torch.zeros(2, dtype=torch.int32, device=dev),
                    max_lc=self.max_ctx, graph=torch.cuda.CUDAGraph(), next=None)
                gv = types.SimpleNamespace(pos={1: gs.pos}, slots={1: gs.slots}, lenp=gs.lenp,
                                           max_lc=self.max_ctx)
                with torch.inference_mode():
                    eng.reset()
                    eng.forward(torch.arange(8, device=dev), start=0)
                    for x in (gs.tok, gs.pos, gs.slots):
                        x.fill_(8)
                    gs.lenp[0].fill_(8)
                    gs.lenp[1].fill_(9)
                    body = _body(eng, gv, gs.tok)
                    body()                        # every kernel compiled before the capture
            if where.gate is not None and not where.gate.acquire(timeout=where.gate_timeout_s):
                raise TimeoutError("the engine stayed busy; no idle window for the capture")
            try:
                stream = where.dev.stream
                stream.synchronize()
                with torch.inference_mode(), capture_lock.LOCK, \
                        torch.cuda.graph(gs.graph, stream=stream):
                    lg = body()
                    gs.next = lg[0, -1].argmax()
                stream.synchronize()
            finally:
                if where.gate is not None:
                    where.gate.release()
            self.gs = gs
        except Exception as exc:                                  # noqa: BLE001
            self.gs = None
            self.graph_error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            with torch.inference_mode():
                eng.reset()

    # ------------------------------------------------------------------ what it reports
    def precision(self) -> str | None:
        if self.eng is None:
            return None
        step = "decode from a CUDA graph" if self.gs is not None else (
            f"eager decode ({self.graph_error})" if self.graph_error else "eager decode")
        return f"bfloat16 parameters, the engine's Qwen3.5 forward, greedy, {step}"

    def device_bytes(self) -> int:
        if self.eng is None:
            return 0
        seen, n = set(), 0
        tensors = list(self.eng.w.t.values()) + [b.w for b in self.eng.w.q.values()]
        tensors += [self.eng.kv.k, self.eng.kv.v, self.eng.state.S, self.eng.state.conv]
        for t in tensors:
            if t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                n += t.numel() * t.element_size()
        return n

    def weight_hash(self) -> str | None:
        return self._hash


def _body(eng, gv, tok):
    """The graph-safe decode step: Qwen38Engine.forward with `_gv` set (positions, KV slots and the
    length from device buffers), one row, the last row's logits."""
    def run():
        eng._gv = gv
        try:
            return eng.forward(tok, start=0, last_only=True)
        finally:
            eng._gv = None
    return run


def _to_device(w, device, gpu) -> None:
    """The host weights onto `device` in batches of about BATCH_BYTES, each under `gpu()` (the
    engine's lock and the model's pool); a tensor under two names (the tied head) is copied once."""
    keys = [("t", k) for k in w.t] + [("q", k) for k in w.q]
    moved: dict[int, object] = {}
    batch, size = [], 0

    def flush():
        nonlocal batch, size
        if batch:
            with gpu():
                for kind, k in batch:
                    t = w.t[k] if kind == "t" else w.q[k].w
                    new = moved.get(id(t))
                    if new is None:
                        new = moved[id(t)] = t.to(device, non_blocking=False)
                    if kind == "t":
                        w.t[k] = new
                    else:
                        w.q[k].w = new
        batch, size = [], 0

    for kind, k in keys:
        t = w.t[k] if kind == "t" else w.q[k].w
        batch.append((kind, k))
        size += t.numel() * t.element_size()
        if size >= BATCH_BYTES:
            flush()
    flush()
    w.device = str(device)
