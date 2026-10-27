"""The shadow reader: a small Qwen3.5 model keeps its own running state of each session (SR1, 2026-10-04).

WHY. Both closed-form bridges from Qwen3.8-27B's state into a small model's failed: B2's soft rows
(16 soft tokens: chance) and B3's per-layer KV/GDN map (held-out R² ~0.5 against ~0.95 needed).
B3's oracle showed the one path that is exact: the small model's OWN state of the same tokens. The
Qwen3.5 small models share the 27B's text token ids (the same BPE vocabulary and merges), so the
small model can read the session as Qwen read it, with no second tokenisation, and keep reading:
each turn costs only its new tokens.

WHAT ONE SESSION HOLDS (`ShadowState`): the small model's KV rows of every attention layer (a
buffer that grows in steps of `GROW_ROWS`), the recurrent and convolution state of every
Gated-DeltaNet layer, the ids it has read (`ids`, its committed position is `len(ids)`), the ids it
should have read (`target`: the served prompt and Qwen's output, in the small model's ids), and a
few CHECKPOINTS -- the recurrent state at a committed position, so a session whose next prompt
differs from what was read (the chat template drops an earlier turn's reasoning when a new user
message comes) goes back to the newest checkpoint at or below the first differing token instead of
starting over. KV rows need no checkpoint: rows below a position stay valid, a rollback is the
length moving back.

EXACT BY CONSTRUCTION (the grid). The state advances only in whole chunks of `grid` rows at
absolute positions k*grid, the same chunks a fresh prefill of the same ids on the same grid runs.
Each chunk is a deterministic function of the rows and state before it, so the shadow after any
number of turns is bit for bit the fresh prefill of the ids it holds. The ids past the last grid
line (fewer than `grid`) are the PENDING TAIL: a question (`fork`) prefills the tail and its own
tokens in scratch rows above the committed length, reads its logits, and puts the recurrent state
back. `tools/shadow_measure.py` checks the equality on the GPU; tests/test_shadow.py on the CPU.

WHAT IS QWEN'S. Nothing: the reader's engine is a second `Qwen38Engine` over the small model's
weights (loaded once, by the slot); every tensor a session holds is allocated in the small model's
own memory pool and stream (engine/decision/device.py), inside the caller's `scope()`. The caller
decides WHEN it runs (server/conductor/shadow.py): between turns in the idle window, and inside a
turn only in the narrator's rationed slices.

ID MAPPING. Qwen's ids pass as they are (`same_ids` held at load), except the ids the small model
does not know the same way: an id at or past its vocabulary (the images' key ids), an added token
of the 27B that is not the small model's (Qwen3.8's seven audio tokens), and the image and video
placeholders (the small model gets no image features; the placeholder rows would be noise). The
mapping is a function of the id alone, so the same stream maps to the same ids every time.
"""

from __future__ import annotations

import json
import os
import time

import torch

GRID = 512
GROW_ROWS = 8192
# the session window (the 27B's): the rotary table covers it; a session past it stops advancing
WINDOW = 262_144
CHECKPOINTS = 6
# the prompt-end checkpoint sits this far below the prompt's end, so the usual divergence (the
# previous turn's re-rendered answer, a few tokens past the generation prompt) is above it
MARK_SLACK = 8
PLACEHOLDERS = ("<|image_pad|>", "<|video_pad|>")


class GrowKV:
    """A one-sequence KV buffer like `engine.model.KVCache`, which grows in steps of GROW_ROWS
    rows instead of holding the window from the start. bf16 only (the reader never runs FP8 KV)."""

    fp8 = False
    ks = vs = None

    def __init__(self, cfg, device, dtype=torch.bfloat16, rows: int = GROW_ROWS):
        n = len(cfg.attention_layers)
        self.slot = {l: i for i, l in enumerate(cfg.attention_layers)}
        self.shape = (n, 1, cfg.num_key_value_heads, cfg.head_dim)
        self.dtype, self.device = dtype, device
        self.k = torch.zeros(n, 1, cfg.num_key_value_heads, rows, cfg.head_dim, dtype=dtype,
                             device=device)
        self.v = torch.zeros_like(self.k)
        self.length = 0
        self.max_len = WINDOW

    @property
    def cap(self) -> int:
        return int(self.k.shape[3])

    def ensure(self, rows: int) -> None:
        if rows <= self.cap:
            return
        if rows > WINDOW:
            raise RuntimeError(f"the shadow's window is {WINDOW} rows; {rows} asked")
        new = min(WINDOW, max(rows, self.cap + max(GROW_ROWS, self.cap // 8)))
        new = -(-new // GROW_ROWS) * GROW_ROWS
        for name in ("k", "v"):
            old = getattr(self, name)
            t = torch.zeros(*old.shape[:3], new, old.shape[4], dtype=old.dtype, device=old.device)
            t[:, :, :, :old.shape[3]].copy_(old)
            setattr(self, name, t)

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int) -> tuple:
        i = self.slot[layer]
        t = k.shape[2]
        self.ensure(start + t)
        self.k[i, :, :, start:start + t] = k
        self.v[i, :, :, start:start + t] = v
        return self.k[i, :, :, :start + t], self.v[i, :, :, :start + t]

    def scales(self, layer: int, n: int) -> tuple:
        return None, None

    def nbytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()

    def row_bytes(self) -> int:
        n, _, h, d = self.shape
        return 2 * n * h * d * self.k.element_size()


class ShadowState:
    """One session's state in the small model."""

    def __init__(self, key: str, cfg, device, dtype):
        from engine.model import GDNState
        self.key = key
        self.kv = GrowKV(cfg, device, dtype)
        self.gdn = GDNState(cfg, device)
        if self.gdn.conv.dtype != dtype:
            self.gdn.conv = self.gdn.conv.to(dtype)
        self.ids: list[int] = []
        self.target: list[int] = []
        self.checks: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self.marks: set[int] = set()
        self.asks: list = []                 # pending questions at a position (server side)
        self.fork = None                     # an open `Fork` (its state is put back first)
        self.created = self.used = time.time()
        self.stats = {"read": 0, "chunks": 0, "ms": 0.0, "rollbacks": 0, "reread": 0,
                      "restarts": 0, "forks": 0}

    @property
    def pos(self) -> int:
        return len(self.ids)

    def behind(self) -> int:
        return max(0, len(self.target) - self.pos)

    def nbytes(self) -> int:
        n = self.kv.nbytes() + self.gdn.nbytes
        for _, s, c in self.checks:
            n += s.numel() * s.element_size() + c.numel() * c.element_size()
        return n


def lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    # a fast path over the common case (the stream only grew): compare in blocks
    step = 4096
    while i + step <= n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Reader:
    """The shadow reader's engine over one loaded Qwen3.5 model (engine/generative/qwen35.py).

    Every method that touches tensors runs inside the caller's `scope()` (the model's pool and
    stream) with the engine lock held by the caller."""

    def __init__(self, model, grid: int = GRID, checkpoints: int = CHECKPOINTS):
        from engine.model import Qwen38Engine
        if model.eng is None:
            raise RuntimeError(f"{model.name} is not loaded")
        if model.same_ids is not None:
            raise RuntimeError(f"{model.name}: Qwen's ids are not this model's ({model.same_ids})")
        src = model.eng
        self.model, self.cfg = model, src.cfg
        self.device = src.device
        self.dtype = src.kv.k.dtype
        self.grid, self.max_checks = int(grid), int(checkpoints)
        # a second engine over the same weights; its KV and state are bound per session
        eng = Qwen38Engine(src.cfg, src.w, max_len=8, device=str(src.device))
        eng.max_len = WINDOW
        eng._rope_cache = None
        self.eng = eng
        self.vocab = int(src.cfg.vocab_size)
        self.drop = self._dropped()
        self.weights = model.weight_hash()

    # ------------------------------------------------------------------ ids
    def _dropped(self) -> set[int]:
        """Ids of the 27B that do not reach the small model (see the module's ID MAPPING)."""
        drop: set[int] = set()
        tok = self.model.tok
        for name in PLACEHOLDERS:
            i = tok.convert_tokens_to_ids(name)
            if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
                drop.add(int(i))
        eng_dir = self.model.engine_dir
        p = os.path.join(os.path.expanduser(eng_dir), "tokenizer.json") if eng_dir else None
        if p and os.path.isfile(p):
            with open(p) as f:
                theirs = {int(t["id"]): t["content"] for t in json.load(f).get("added_tokens")
                          or []}
            with open(os.path.join(self.model.dir, "tokenizer.json")) as f:
                mine = {int(t["id"]): t["content"] for t in json.load(f).get("added_tokens")
                        or []}
            drop |= {i for i, c in theirs.items() if mine.get(i) != c}
        return drop

    def map_ids(self, ids) -> list[int]:
        v, drop = self.vocab, self.drop
        return [int(t) for t in ids if 0 <= int(t) < v and int(t) not in drop]

    # ------------------------------------------------------------------ sessions
    def new(self, key: str) -> ShadowState:
        return ShadowState(key, self.cfg, self.device, self.dtype)

    def _bind(self, s: ShadowState) -> None:
        e = self.eng
        e.kv, e.state = s.kv, s.gdn
        e._pend = e._pend_ctx = None
        e.kv.length = s.pos

    def sync(self, s: ShadowState, target: list[int]) -> int:
        """Point `s` at `target` (already mapped). When what it read is not a prefix of it, go
        back to the newest checkpoint at or below the first difference (or to the start). Returns
        the ids the rollback dropped (0 when none)."""
        if s.fork is not None:
            s.fork.close()
        s.target = target
        n = lcp(s.ids, target)
        if n >= s.pos:
            return 0
        dropped = s.pos - n
        best = max((c for c in s.checks if c[0] <= n), key=lambda c: c[0], default=None)
        s.checks = [c for c in s.checks if c[0] <= n]
        if best is None:
            s.gdn.S.zero_()
            s.gdn.conv.zero_()
            s.gdn.primed = False
            s.ids = []
            s.stats["restarts"] += 1
        else:
            s.gdn.S.copy_(best[1])
            s.gdn.conv.copy_(best[2])
            s.gdn.primed = best[0] > 0
            s.ids = s.ids[:best[0]]
        s.kv.length = s.pos
        s.stats["rollbacks"] += 1
        s.stats["reread"] += n - s.pos
        return dropped

    def mark(self, s: ShadowState, prompt_end: int) -> None:
        """A checkpoint is due at the grid line at or below `prompt_end - MARK_SLACK`."""
        g = max(0, prompt_end - MARK_SLACK) // self.grid * self.grid
        if g > 0:
            s.marks.add(g)

    def committable(self, s: ShadowState, stop: int | None = None) -> int:
        """The rows `advance` may still commit: whole grid chunks of the target (up to `stop`)."""
        end = len(s.target) if stop is None else min(stop, len(s.target))
        return max(0, end // self.grid * self.grid - s.pos)

    def advance(self, s: ShadowState, *, deadline: float | None = None, stop: int | None = None,
                max_chunks: int | None = None, clock=time.perf_counter) -> int:
        """Commit whole grid chunks of `s.target` (below `stop`) until the deadline (checked
        before every chunk), `max_chunks`, or nothing whole is left. Returns the rows read."""
        g, done, chunks = self.grid, 0, 0
        t0 = clock()
        if s.fork is not None:
            s.fork.close()
        self._bind(s)
        with torch.inference_mode():
            while self.committable(s, stop) >= g:
                if deadline is not None and clock() >= deadline:
                    break
                if max_chunks is not None and chunks >= max_chunks:
                    break
                a = s.pos
                if a + g > WINDOW:
                    break
                ids = s.target[a:a + g]
                self.eng.forward(torch.tensor(ids, device=self.device), start=a, last_only=True)
                s.ids.extend(ids)
                s.kv.length = s.pos
                done += g
                chunks += 1
                if s.pos in s.marks:
                    self._checkpoint(s)
        if done:
            s.stats["read"] += done
            s.stats["chunks"] += chunks
            s.stats["ms"] += (clock() - t0) * 1e3
            s.used = time.time()
        return done

    def _checkpoint(self, s: ShadowState) -> None:
        s.marks.discard(s.pos)
        s.checks = [c for c in s.checks if c[0] != s.pos]
        s.checks.append((s.pos, s.gdn.S.clone(), s.gdn.conv.clone()))
        if len(s.checks) > self.max_checks:
            # keep the oldest (the session's head) and the newest ones
            s.checks.sort(key=lambda c: c[0])
            del s.checks[1]

    # ------------------------------------------------------------------ questions
    def fork(self, s: ShadowState, upto: int, tail: list[int]) -> "Fork":
        """A question on top of `s.target[:upto]`: see `Fork`."""
        return Fork(self, s, upto, tail)

    def fork_logits(self, s: ShadowState, upto: int, tail: list[int],
                    rows: list[int] | None = None) -> torch.Tensor:
        """The logits (float32, on the host) of the last row of `s.target[:upto] + tail`, only
        `rows` of them when given; the session's state as it was after."""
        f = self.fork(s, upto, tail)
        try:
            f.prefill()
            out = f.logits
            if rows is not None:
                out = out[torch.tensor(rows, device=out.device)]
            return out.cpu()
        finally:
            f.close()

    def generate(self, s: ShadowState, upto: int, tail: list[int], max_new: int, eos: set[int],
                 deadline: float | None = None) -> tuple[list[int], str]:
        """Greedy tokens after `s.target[:upto] + tail`; (ids, stop | length | time)."""
        f = self.fork(s, upto, tail)
        try:
            if f.prefill(deadline):
                f.decode(max_new, eos, deadline)
            return list(f.out), f.finish or "time"
        finally:
            f.close()

    # ------------------------------------------------------------------ the reference
    def fresh(self, ids: list[int], grid: int | None = None) -> ShadowState:
        """A new state of `ids` read from the start in chunks of `grid` (the reader's own by
        default): what the exactness check compares a long-lived shadow with."""
        s = self.new("fresh")
        g, keep = int(grid or self.grid), self.grid
        try:
            self.grid = g
            s.target = list(ids)
            self.advance(s)
        finally:
            self.grid = keep
        return s

    # ------------------------------------------------------------------ disk
    def save(self, s: ShadowState, path: str) -> int:
        """`s` to one safetensors file (rows [0, pos), the state, the checkpoints, the ids);
        written to a temporary name and renamed. Returns the bytes written."""
        from safetensors.torch import save_file
        n = s.pos
        t = {"k": s.kv.k[:, :, :, :n].contiguous().cpu(), "v": s.kv.v[:, :, :, :n].contiguous().cpu(),
             "S": s.gdn.S.cpu(), "conv": s.gdn.conv.cpu(),
             "ids": torch.tensor(s.ids, dtype=torch.int32),
             "target": torch.tensor(s.target, dtype=torch.int32)}
        for i, (p, S, c) in enumerate(s.checks):
            t[f"check{i}.S"], t[f"check{i}.conv"] = S.cpu(), c.cpu()
        meta = {"format": "shadow-1", "key": s.key, "weights": str(self.weights),
                "grid": str(self.grid), "checks": json.dumps([c[0] for c in s.checks]),
                "marks": json.dumps(sorted(s.marks)), "primed": str(bool(s.gdn.primed))}
        tmp = path + ".tmp"
        save_file(t, tmp, metadata=meta)
        os.replace(tmp, path)
        return os.path.getsize(path)

    def load(self, path: str) -> ShadowState:
        """A state `save` wrote, onto the device; ValueError when it was written by other weights
        or on another grid."""
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if meta.get("format") != "shadow-1":
                raise ValueError("not a shadow file")
            if meta.get("weights") != str(self.weights) or meta.get("grid") != str(self.grid):
                raise ValueError("written by other weights or on another grid")
            s = self.new(meta.get("key") or "?")
            k = f.get_tensor("k")
            n = int(k.shape[3])
            s.kv.ensure(n)
            s.kv.k[:, :, :, :n].copy_(k.to(self.device))
            s.kv.v[:, :, :, :n].copy_(f.get_tensor("v").to(self.device))
            s.gdn.S.copy_(f.get_tensor("S").to(self.device))
            s.gdn.conv.copy_(f.get_tensor("conv").to(self.device))
            s.gdn.primed = meta.get("primed") == "True"
            s.ids = f.get_tensor("ids").tolist()
            s.target = f.get_tensor("target").tolist()
            for i, p in enumerate(json.loads(meta.get("checks") or "[]")):
                s.checks.append((int(p), f.get_tensor(f"check{i}.S").to(self.device),
                                 f.get_tensor(f"check{i}.conv").to(self.device)))
            s.marks = set(json.loads(meta.get("marks") or "[]"))
        if len(s.ids) != n:
            raise ValueError("the rows and the ids disagree")
        s.kv.length = n
        return s


class Fork:
    """A question asked on top of a session: the pending ids up to `upto` and `tail` prefilled in
    scratch rows above the committed length, then greedy tokens, in slices that can stop at a
    deadline and go on later (the narrator's, between two of Qwen's rounds). The recurrent and
    convolution state is copied when the fork opens and put back by `close` (a session with an
    open fork closes it before anything else reads it). Nothing below the committed length is
    written, so the session is the same after, bit for bit."""

    def __init__(self, reader: "Reader", s: ShadowState, upto: int, tail: list[int]):
        if getattr(s, "fork", None) is not None:
            s.fork.close()
        if upto < s.pos:
            raise ValueError(f"the shadow is past {upto} (at {s.pos})")
        self.reader, self.s = reader, s
        self.base = s.pos
        self.ids = s.target[s.pos:upto] + [int(t) for t in tail]
        if not self.ids:
            raise ValueError("nothing to read")
        if self.base + len(self.ids) + 1 > WINDOW:
            raise ValueError("past the window")
        self.saved = (s.gdn.S.clone(), s.gdn.conv.clone(), s.gdn.primed)
        self.done_rows = 0
        self.logits: torch.Tensor | None = None
        self.out: list[int] = []
        self.next: int | None = None
        self.finish: str | None = None
        self.closed = False
        s.fork = self

    @property
    def at(self) -> int:
        return self.base + self.done_rows + len(self.out)

    def prefill(self, deadline: float | None = None, clock=time.perf_counter) -> bool:
        """The fork's prompt, a grid chunk at a time (the deadline checked before each); True when
        all of it is read and `logits` holds its last row's."""
        r = self.reader
        r._bind(self.s)
        with torch.inference_mode():
            while self.done_rows < len(self.ids):
                if deadline is not None and clock() >= deadline:
                    return False
                a = self.done_rows
                part = self.ids[a:a + r.grid]
                lg = r.eng.forward(torch.tensor(part, device=r.device), start=self.base + a,
                                   last_only=True)
                self.done_rows += len(part)
                if self.done_rows == len(self.ids):
                    self.logits = lg[0, -1].float()
                    self.next = int(self.logits.argmax())
        return True

    def decode(self, max_new: int, eos: set[int], deadline: float | None = None,
               clock=time.perf_counter) -> bool:
        """Greedy steps until an end token, `max_new` tokens or the deadline (checked before every
        step); True when done (`finish` stop or length)."""
        r = self.reader
        if self.finish is not None:
            return True
        if self.done_rows < len(self.ids):
            return False
        r._bind(self.s)
        with torch.inference_mode():
            while True:
                if self.next is None:
                    if deadline is not None and clock() >= deadline:
                        return False
                    pos = self.base + len(self.ids) + len(self.out) - 1
                    lg = r.eng.forward(torch.tensor([self.out[-1]], device=r.device), start=pos,
                                       last_only=True)
                    self.next = int(lg[0, -1].argmax())
                t, self.next = self.next, None
                if t in eos:
                    self.finish = "stop"
                    return True
                self.out.append(t)
                if len(self.out) >= max_new or self.at + 1 >= WINDOW:
                    self.finish = "length"
                    return True

    def close(self) -> None:
        if self.closed:
            return
        s = self.s
        s.gdn.S.copy_(self.saved[0])
        s.gdn.conv.copy_(self.saved[1])
        s.gdn.primed = self.saved[2]
        s.kv.length = s.pos
        self.saved = None
        self.closed = True
        s.stats["forks"] += 1
        if getattr(s, "fork", None) is self:
            s.fork = None


def same_state(a: ShadowState, b: ShadowState) -> dict:
    """Bit equality of two states over the rows both hold: KV rows, recurrent and convolution
    state; with the largest absolute difference of each when they differ."""
    n = min(a.pos, b.pos)
    out = {"rows": n, "ids_equal": a.ids[:n] == b.ids[:n]}
    for name, x, y in (("k", a.kv.k[:, :, :, :n], b.kv.k[:, :, :, :n]),
                       ("v", a.kv.v[:, :, :, :n], b.kv.v[:, :, :, :n]),
                       ("S", a.gdn.S, b.gdn.S), ("conv", a.gdn.conv, b.gdn.conv)):
        eq = bool(torch.equal(x, y))
        out[name] = eq
        if not eq:
            out[name + "_maxdiff"] = float((x.float() - y.float()).abs().max())
    out["bit_exact"] = all(out[k] for k in ("k", "v", "S", "conv")) and out["ids_equal"]
    return out
