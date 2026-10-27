"""The shadow reader's service: which sessions it keeps, when it reads, and who asks it (SR1, 2026-10-04).

The engine side is engine/shadow.py (one session's state in the small model, read on a fixed grid,
exact by construction). This module decides:

WHAT IT READS. A profile with `shadow.mode: on` makes every turn of a session feed its tokens: the
served prompt (what Qwen read, the conductor's notes included) at the turn's start (`begin_turn`)
and Qwen's output at its end (`end_turn`), keyed by the router's session key (0.3.23's session
identity: `c:<client id>` or `h:<head hash>`). Feeding is host work only (a list of ids); nothing
runs on the GPU there.

WHEN IT READS (the bandwidth rule: during Qwen's decode only Qwen and the drafters touch the GPU).
  * In the idle window: the server's shadow loop (server/app.py `shadow_once`) takes the engine
    lock behind every waiter and the box lock without waiting, and calls `work`: grid chunks of
    the sessions that are behind, newest first, the deadline and "someone waits" checked before
    every chunk, so a request that arrives waits at most one chunk (measured: tools/shadow_measure.py).
  * Inside a turn only for the narrator (`shadow.narrator: on`): its job reads what is missing of
    the session up to the thought, then asks for the summary, in the narrator's rationed slices
    (at most 3 % of Qwen's time, server/conductor/narrator.py `Thoughts`).

WHO ASKS.
  * `ask(key, question, choices)`: a typed decision of the small model over the whole session --
    each choice a letter, one forward, the letters' softmax (the in-context stage's form,
    server/conductor/incontext.py). The route `POST /v1/shadow/ask` and the conductor's own calls.
  * The router's class question in shadow mode (`shadow.router: shadow`): for a turn whose prompt
    has at least `router_min_tokens` tokens, the question the router's in-context stage asks Qwen
    is put to the shadow at the turn's insertion point (the prompt's last `<|im_end|>`), answered
    in the idle window after the turn, and journaled beside the router's own class in
    `<conductor root>/shadow.jsonl`: the agreement, and how far behind the shadow was when the
    turn began (the freshness a live use would have had). Nothing of the turn changes.
  * The narrator (above).

WHAT IT KEEPS. Resident states up to `cap_gib` (QWEN38_SHADOW_GB); past it the least recently used
session goes to the NVMe (`disk_dir`, one safetensors file a session, read back when it is fed
again) or, without a disk, is dropped (it is read from the start next time). A planned stop writes
every resident session (`drain`); the files are kept `DISK_TTL_S` and at most `disk_gib`.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import threading
import time

from engine import shadow as eng_shadow

GRID = eng_shadow.GRID
DISK_TTL_S = 48 * 3600.0
#: the router question is asked on turns whose prompt has at least this many tokens
ROUTER_MIN_TOKENS = 8192
#: what one idle burst may hold the engine at most (the "someone waits" check comes first)
BURST_MS = 2000.0
#: inside a turn: the share of the turn's time the shadow may read in, and one slice's ceiling
#: (tools/shadow_fresh.py on the ledger's 41 opencode sessions, 1,624 turns: the median idle gap
#: between two requests is 0.31 s, so the idle window alone left the shadow fresh at 41 % of
#: turn starts; with 2 % of each turn's time it is fresh at 90 %, p90 591 tokens behind)
INLINE_SHARE = 0.02
INLINE_SLICE_MS = 150.0
JOURNAL = "shadow.jsonl"
# The model's pool keeps every segment it ever reserved (a live MemPool never gives
# its free segments back). Reading a long conversation grows the KV in steps and every chunk's
# attention temporaries with the context, each a new segment: after 195,694 tokens on 2026-10-04
# the pool held ~25 GiB, 17.6 of it free, and the engine's memory gate (which counts it) refused
# every request. Past this much free in the pool after a burst, the sessions go to the disk and
# the model is unloaded, which drops the pool; the next use loads it again (~13 s, beside).
# QWEN38_SHADOW_POOL_SLACK_GIB (engine/settings.py), read at each burst
def pool_slack_gib() -> float:
    from engine.settings import SETTINGS as _S
    return float(_S.get("SHADOW_POOL_SLACK_GIB") or 2.0)
JOURNAL_KEEP = 5000
MODES = ("off", "on")
ROUTER_MODES = ("off", "shadow")
OFF_WHY = ("this engine serves no shadow reader: QWEN38_SHADOW_MODEL is empty (set it in "
           "ops/serve.env, e.g. qwen3.5-0.8b, and restart)")
# SRV-339: a model whose adapter says its reasoning was never read by the reader (`caps.shadow`)
NOT_VALIDATED_WHY = ("the shadow reader is not started: its thought lines and router verdicts were "
                     "validated on another model's reasoning, not on the one this engine serves")

# the narrator's question on top of the session (SR1): the open thought is closed, a hidden user
# turn asks for the summary, the answer is begun with "It"
THOUGHT_QUESTION = ("Summarise the reasoning in the thinking block just above for the person "
                    "waiting on the assistant. Write about the assistant in the third person, as "
                    "\"It\". One or two short sentences, at most 30 words: what it thought about "
                    "and what it decided, then its next step if it names one (\"; next it ...\"). "
                    "Name the concrete things: files, functions, errors, options. Plain words, no "
                    "quotes, no markdown.")
THOUGHT_TAIL = ("\n</think>\n\n<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\nIt")


def settings_of(profile: dict | None) -> dict:
    """A profile's `shadow` block with its defaults (absent: everything off)."""
    sh = dict((profile or {}).get("shadow") or {})
    return {"mode": sh.get("mode", "off"), "narrator": sh.get("narrator", "off"),
            "router": sh.get("router", "off"),
            "router_min_tokens": int(sh.get("router_min_tokens", ROUTER_MIN_TOKENS))}


def _hkey(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:24]


class ShadowService:
    def __init__(self, slot, *, grid: int = GRID, cap_gib: float = 4.0,
                 disk_dir: str | None = None, disk_gib: float = 20.0, root: str | None = None,
                 log=print, clock=time.perf_counter):
        self.slot = slot
        self.grid = int(grid)
        self.cap = int(float(cap_gib) * (1 << 30))
        self.disk_dir = os.path.expanduser(disk_dir) if disk_dir else None
        self.disk_cap = int(float(disk_gib) * (1 << 30))
        self.journal_path = (os.path.join(os.path.expanduser(root), JOURNAL) if root else None)
        self.log, self.clock = log, clock
        self.reader: eng_shadow.Reader | None = None
        self._reader_of = None
        self.sessions: collections.OrderedDict[str, eng_shadow.ShadowState] = \
            collections.OrderedDict()
        self.parked: dict[str, dict] = {}
        self.feed: dict[str, dict] = {}
        self.mu = threading.Lock()
        self.stats = collections.Counter()
        self.router = collections.Counter()
        self.recent: collections.deque = collections.deque(maxlen=200)
        self.last_error: str | None = None
        if self.disk_dir:
            self._scan_disk()

    # ------------------------------------------------------------------ the model
    @property
    def model(self):
        return self.slot.model

    def ready(self) -> str | None:
        """None when the reader can run; else why not (and the slot's load starts)."""
        why = self.slot.ensure()
        if why is not None:
            return why
        if not getattr(self.model, "jobs", False) or not hasattr(self.model, "eng"):
            return "not_qwen35"
        return None

    def _scope(self):
        where = getattr(self.slot, "_where", None)
        if where is None:
            import contextlib
            return contextlib.nullcontext()
        return where.scope()

    def _get_reader(self) -> eng_shadow.Reader:
        key = (id(self.model), self.slot.loaded_at)
        if self.reader is None or self._reader_of != key:
            # a reload: the states were the old weights' (the disk files check the hash)
            self.sessions.clear()
            self.reader = eng_shadow.Reader(self.model, grid=self.grid)
            self._reader_of = key
        return self.reader

    # ------------------------------------------------------------------ feeding (host only)
    def begin_turn(self, key: str, prompt_ids: list[int], *, ask: dict | None = None) -> dict:
        """A turn of session `key` starts with this served prompt. `ask`: the router's question
        for this turn (`router_ask`). Returns what the shadow knew then (for the journal)."""
        now = time.time()
        with self.mu:
            f = self.feed.get(key)
            prev = f["raw"] if f is not None else None
            committed_raw = int(f.get("committed_raw", 0)) if f is not None else 0
            # what the shadow will have to read before it holds this prompt: the ids past both
            # the part it already read and the part this prompt shares with the last feed
            common = eng_shadow.lcp(prev, prompt_ids) if prev is not None else 0
            behind = len(prompt_ids) - min(common, committed_raw)
            known = key in self.sessions or key in self.parked
            f = self.feed[key] = {"raw": list(prompt_ids), "prompt_end": len(prompt_ids),
                                  "asks": (f or {}).get("asks", []), "t_begin": now,
                                  "committed_raw": min(common, committed_raw),
                                  "applied": False}
            if ask is not None:
                f["asks"].append(dict(ask, t_begin=now, behind_at_begin=behind, known=known))
            self.stats["turns"] += 1
        self.slot.ensure()                      # a first turn starts the load
        return {"behind": behind}

    def extend(self, key: str, out_ids: list[int]) -> None:
        """Qwen's output so far (the narrator's job reads up to it)."""
        with self.mu:
            f = self.feed.get(key)
            if f is not None:
                f["raw"] = f["raw"][:f["prompt_end"]] + [int(t) for t in out_ids]
                f["applied"] = False

    def end_turn(self, key: str, out_ids: list[int]) -> None:
        with self.mu:
            f = self.feed.get(key)
            if f is None:
                return
            f["raw"] = f["raw"][:f["prompt_end"]] + [int(t) for t in out_ids]
            f["applied"] = False
            f["t_end"] = time.time()

    def forget(self, key: str) -> None:
        """The session is gone (its key evicted): its state, file and feed."""
        with self.mu:
            self.feed.pop(key, None)
            self.sessions.pop(key, None)
            p = self.parked.pop(key, None)
        if p is not None:
            try:
                os.remove(p["path"])
            except OSError:
                pass

    def has_work(self) -> bool:
        with self.mu:
            if any(not f["applied"] or f["asks"] for f in self.feed.values()):
                return True
            return any(s.behind() >= self.grid or s.asks for s in self.sessions.values())

    # ------------------------------------------------------------------ the GPU side
    def _due(self, key: str, f: dict) -> bool:
        if not f["applied"] or f["asks"]:
            return True
        s = self.sessions.get(key)
        return s is not None and (s.asks or s.behind() >= self.grid)

    def _apply(self, key: str) -> eng_shadow.ShadowState | None:
        """The feed of `key` onto its state (created, or read back from the disk). Under the
        engine lock, in the model's scope."""
        r = self.reader
        with self.mu:
            f = self.feed.get(key)
            if f is None:
                return self.sessions.get(key)
            raw, pend, asks, applied = list(f["raw"]), f["prompt_end"], f["asks"], f["applied"]
            f["asks"] = []
        s = self.sessions.get(key)
        if s is None and key in self.parked:
            s = self._unpark(key)
        if s is None:
            s = r.new(key)
            self.stats["created"] += 1
        if key not in self.sessions or not applied or asks:
            self.sessions[key] = s
            self.sessions.move_to_end(key)
        if not applied or asks:
            mapped = r.map_ids(raw)
            mprompt = len(r.map_ids(raw[:pend]))
            dropped = r.sync(s, mapped)
            if dropped:
                self.stats["rolled_back_ids"] += dropped
            r.mark(s, mprompt)
            for a in asks:
                a["pos"] = len(r.map_ids(raw[:a["raw_at"]]))
                s.asks.append(a)
            s.asks.sort(key=lambda a: a["pos"])
            with self.mu:
                f2 = self.feed.get(key)
                if f2 is not None and f2["raw"] == raw:
                    f2["applied"] = True
        return s

    def _committed_raw(self, key: str, s: eng_shadow.ShadowState) -> None:
        """How many of the feed's raw ids the committed position covers (for `behind`)."""
        with self.mu:
            f = self.feed.get(key)
            if f is None:
                return
            # the mapping only drops ids, so the raw count is at least the mapped one
            raw, n, m = f["raw"], 0, 0
            r = self.reader
            for t in raw:
                if m >= s.pos:
                    break
                n += 1
                if 0 <= int(t) < r.vocab and int(t) not in r.drop:
                    m += 1
            f["committed_raw"] = n

    def work(self, *, deadline: float, stop=lambda: False) -> dict:
        """The idle window's reading, with the engine lock held: every feed applied, then the
        newest sessions first, their questions answered at their positions, grid chunks until the
        deadline or `stop()` (checked before every chunk); then the memory cap."""
        out = {"rows": 0, "asks": 0, "ms": 0.0, "sessions": 0, "stopped": False}
        if self.ready() is not None:
            return dict(out, why=self.ready())
        t0 = self.clock()
        try:
            with self._scope():
                self._get_reader()
                with self.mu:
                    # the sessions with something to do, the most recently fed first
                    keys = sorted((k for k, f in self.feed.items() if self._due(k, f)),
                                  key=lambda k: self.feed[k]["t_begin"], reverse=True)
                for k in keys:
                    if stop() or self.clock() >= deadline:
                        out["stopped"] = True
                        break
                    s = self._apply(k)
                    if s is None:
                        continue
                    out["sessions"] += 1
                    out["rows"] += self._read(k, s, deadline, stop, out)
                self._fit()
        except Exception as exc:                                  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            self.stats["errors"] += 1
            self.log(f"[shadow] work failed: {self.last_error}", flush=True)
        res, alloc = self.pool_bytes()
        if res - alloc > pool_slack_gib() * (1 << 30) and not stop():
            # The release parks every session on the disk under the engine lock
            # (10.4 s for two on 2026-10-04): it yields to a request that waits, between sessions,
            # and the next idle burst goes on with the rest
            out["released"] = self.release("pool_slack", stop=stop)
        out["ms"] = round((self.clock() - t0) * 1e3, 2)
        self.stats["rows"] += out["rows"]
        self.stats["work_ms"] += out["ms"]
        return out

    def _read(self, key, s, deadline, stop, out) -> int:
        r, rows = self.reader, 0
        while True:
            if stop() or self.clock() >= deadline:
                out["stopped"] = True
                break
            nxt = s.asks[0] if s.asks else None
            if nxt is not None and nxt["pos"] < s.pos:
                s.asks.pop(0)
                self._answer_missed(key, nxt)
                continue
            if nxt is not None and r.committable(s, nxt["pos"]) < r.grid:
                s.asks.pop(0)
                self._answer(key, s, nxt)
                out["asks"] += 1
                continue
            n = r.advance(s, stop=nxt["pos"] if nxt else None, max_chunks=1)
            if not n:
                break
            rows += n
        self._committed_raw(key, s)
        return rows

    def _fit(self) -> None:
        """Resident states under the cap: the least recently used to the disk (or dropped)."""
        total = sum(s.nbytes() for s in self.sessions.values())
        while total > self.cap and len(self.sessions) > 1:
            key, s = next(iter(self.sessions.items()))
            total -= s.nbytes()
            self._park(key, s)

    # ------------------------------------------------------------------ the disk
    def _path(self, key: str) -> str:
        return os.path.join(self.disk_dir, _hkey(key) + ".safetensors")

    def _scan_disk(self) -> None:
        try:
            os.makedirs(self.disk_dir, mode=0o700, exist_ok=True)
            now = time.time()
            for name in os.listdir(self.disk_dir):
                p = os.path.join(self.disk_dir, name)
                if name.endswith(".tmp") or now - os.path.getmtime(p) > DISK_TTL_S:
                    os.remove(p)
                    continue
                if name.endswith(".json"):
                    with open(p) as f:
                        m = json.load(f)
                    sp = p[:-5] + ".safetensors"
                    if os.path.isfile(sp):
                        self.parked[m["key"]] = dict(m, path=sp)
        except OSError as exc:
            self.log(f"[shadow] disk {self.disk_dir}: {exc}", flush=True)

    def _park(self, key: str, s) -> None:
        self.sessions.pop(key, None)
        if s.fork is not None:
            s.fork.close()
        if not self.disk_dir or s.pos == 0:
            self.stats["dropped"] += 1
            return
        try:
            t0 = self.clock()
            p = self._path(key)
            n = self.reader.save(s, p)
            meta = {"key": key, "pos": s.pos, "bytes": n, "at": time.time()}
            with open(p[:-len(".safetensors")] + ".json", "w") as f:
                json.dump(meta, f)
            self.parked[key] = dict(meta, path=p)
            self.stats["parked"] += 1
            self.stats["park_ms"] += round((self.clock() - t0) * 1e3, 2)
            self._disk_fit()
        except Exception as exc:                                  # noqa: BLE001
            self.stats["park_failed"] += 1
            self.log(f"[shadow] park failed: {type(exc).__name__}: {exc}", flush=True)

    def _unpark(self, key: str):
        p = self.parked.pop(key, None)
        if p is None:
            return None
        try:
            t0 = self.clock()
            s = self.reader.load(p["path"])
            s.key = key
            self.stats["unparked"] += 1
            self.stats["unpark_ms"] += round((self.clock() - t0) * 1e3, 2)
            return s
        except Exception as exc:                                  # noqa: BLE001
            self.stats["unpark_failed"] += 1
            self.log(f"[shadow] read back failed ({exc}); reading the session again", flush=True)
            return None

    def _disk_fit(self) -> None:
        items = sorted(self.parked.items(), key=lambda kv: kv[1]["at"])
        total = sum(v["bytes"] for _, v in items)
        while items and total > self.disk_cap:
            k, v = items.pop(0)
            total -= v["bytes"]
            self.parked.pop(k, None)
            for ext in (".safetensors", ".json"):
                try:
                    os.remove(v["path"][:-len(".safetensors")] + ext)
                except OSError:
                    pass

    def pool_bytes(self) -> tuple[int, int]:
        """(reserved, allocated) of the model's pool; (0, 0) when it has none."""
        fn = getattr(self.slot, "pool_bytes", None)
        try:
            return fn() if fn is not None else (0, 0)
        except Exception:                                          # noqa: BLE001
            return 0, 0

    def release(self, why: str, block: bool = False, stop=None) -> int:
        """Give the model's pool back: every resident session to the disk (or dropped without
        one), then the model unloaded, which drops the pool. Returns the pool's reserved bytes
        (what the next `empty_cache` hands back), 0 when nothing was released (no pool, or a call
        in flight). Under the engine lock: the idle window, or the memory gate. `stop()` (the idle
        window's: a request waits) is asked before each session; when it says so, the sessions
        parked so far stay on the disk, the model stays loaded, and 0 comes back."""
        res, alloc = self.pool_bytes()
        if res == 0 or self.slot.state != "loaded":
            return 0
        busy = getattr(self.slot, "_busy", None)
        if not block and busy is not None and busy.locked():
            return 0
        n = 0
        try:
            if self.reader is not None:
                with self._scope():
                    for key, s in list(self.sessions.items()):
                        if stop is not None and stop():
                            self.stats["release_yielded"] = self.stats.get("release_yielded", 0) + 1
                            self.log(f"[shadow] release ({why}) yielded to a request after {n} "
                                     f"sessions parked", flush=True)
                            return 0
                        self._park(key, s)
                        n += 1
        except Exception as exc:                                  # noqa: BLE001
            self.log(f"[shadow] release: parking failed ({type(exc).__name__}: {exc})",
                     flush=True)
        self.sessions.clear()
        self.reader, self._reader_of = None, None
        if not self.slot.unload(block=block):
            return 0
        self.stats["released"] += 1
        self.stats["released_bytes"] += res
        self.log(f"[shadow] released the model's pool ({why}): {res / 2**30:.2f} GiB reserved, "
                 f"{alloc / 2**30:.2f} GiB in use, {n} sessions parked", flush=True)
        return res

    def drain(self) -> dict:
        """A planned stop: every resident session to the disk. Under the engine lock."""
        if self.reader is None or not self.disk_dir:
            return {"parked": 0}
        n = 0
        with self._scope():
            for key, s in list(self.sessions.items()):
                self._park(key, s)
                n += 1
        return {"parked": n}

    # ------------------------------------------------------------------ questions
    def _letters(self, choices: list[str]):
        from server.conductor import incontext
        tok = self.model.tok
        letters = [incontext.LETTERS[i] for i in range(len(choices))]
        return letters, incontext.letter_ids(tok, letters)

    def _tail(self, question: str, choices: dict[str, str]) -> tuple[list[int], list[int]]:
        from server.conductor import incontext
        text, letters = incontext.listing(choices)
        q = question.replace("{classes}", text) if "{classes}" in question else \
            question + "\n\n" + text + "\n\nAnswer with the letter only."
        tail = self.model.tok(incontext.TURN.replace("{question}", q),
                              add_special_tokens=False)["input_ids"]
        return list(tail), incontext.letter_ids(self.model.tok, letters)

    def _decide(self, s, upto: int, question: str, choices: dict[str, str]) -> dict:
        from server.conductor import incontext
        t0 = self.clock()
        tail, lids = self._tail(question, choices)
        lp = self.reader.fork_logits(s, upto, tail, rows=lids)
        lp = (lp - lp.logsumexp(0)).tolist()
        label, conf, probs = incontext.answer(lp, list(choices))
        return {"label": label, "confidence": round(conf, 4),
                "probs": {k: round(v, 4) for k, v in probs.items()},
                "ms": round((self.clock() - t0) * 1e3, 2), "read": upto,
                "tail": len(tail)}

    def ask(self, key: str, question: str, choices: dict[str, str], *,
            budget_ms: float = 5000.0) -> dict:
        """A typed decision over the session's full context (all it was fed). The engine lock is
        the caller's. `{label, confidence, probs, ms, behind}` or `{why}`."""
        why = self.ready()
        if why is not None:
            return {"why": why}
        if not choices or len(choices) > 26:
            return {"why": "choices"}
        try:
            with self._scope():
                self._get_reader()
                s = self._apply(key)
                if s is None or not s.target:
                    return {"why": "unknown_session"}
                deadline = self.clock() + budget_ms / 1000.0
                self.reader.advance(s, deadline=deadline)
                behind = s.behind()
                d = self._decide(s, len(s.target), question, choices)
                self._committed_raw(key, s)
            self.stats["asks"] += 1
            return dict(d, behind=behind, session_tokens=len(s.target))
        except Exception as exc:                                  # noqa: BLE001
            self.stats["errors"] += 1
            return {"why": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}

    # ------------------------------------------------------------------ the router's question
    def router_ask(self, plan, prompt_ids: list[int], tok, settings: dict) -> dict | None:
        """The router's class question for this turn (`shadow.router: shadow`), or None when the
        turn does not get one: the prompt shorter than `router_min_tokens`, a structural class
        (the compaction), a side request, no class decided, no insertion point. Each skip is
        counted by its reason (`router_skip`)."""
        from server.conductor import inject, schema

        def skip(why):
            self.router["skip_" + why] += 1
            return None
        if settings.get("router") != "shadow" or plan is None or not plan.active:
            return None
        least = settings.get("router_min_tokens")
        if len(prompt_ids) < int(ROUTER_MIN_TOKENS if least is None else least):
            return skip("short")
        row = plan.row or {}
        if row.get("source") == "structural":
            return skip("structural")
        if getattr(plan, "side", None):
            return skip("side")
        # the router's own class: in classify shadow mode its `shadow` field holds it
        mine = row.get("shadow") if isinstance(row.get("shadow"), dict) and \
            row["shadow"].get("class_id") else {}
        cls = mine.get("class_id") or row.get("class_id")
        if not cls:
            return skip("no_class")
        router = plan.router_cfg or {}
        classes = {c["id"]: c.get("description") or c.get("label") or c["id"]
                   for c in router.get("classes") or []}
        if len(classes) < 2:
            return skip("classes")
        at = inject.insertion_point(prompt_ids, inject.special_id(tok, inject.IM_END))
        if at is None:
            return skip("no_insertion_point")
        cs = schema.context_stage(router)
        return {"raw_at": at + 1, "question": cs["question"], "choices": classes,
                "router": {"class_id": cls, "source": mine.get("source") or row.get("source"),
                           "confidence": mine.get("confidence") if mine else
                           row.get("confidence")},
                "request_id": row.get("request_id"), "row_id": row.get("id"),
                "prompt_tokens": len(prompt_ids)}

    def _answer(self, key, s, a) -> None:
        rec = {"ts": round(time.time(), 3), "kind": "router", "session": _hkey(key),
               "request_id": a.get("request_id"), "row_id": a.get("row_id"),
               "router": a.get("router"), "prompt_tokens": a.get("prompt_tokens"),
               "behind_at_begin": a.get("behind_at_begin"), "known": a.get("known")}
        try:
            d = self._decide(s, a["pos"], a["question"], a["choices"])
            agree = d["label"] == (a.get("router") or {}).get("class_id")
            rec.update(d, agree=agree, lag_s=round(time.time() - a["t_begin"], 2))
            rec.pop("read", None)
            self.router["asked"] += 1
            self.router["agree"] += int(agree)
        except Exception as exc:                                  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
            self.router["errors"] += 1
        self._journal(rec)

    def _answer_missed(self, key, a) -> None:
        self.router["missed"] += 1
        self._journal({"ts": round(time.time(), 3), "kind": "router", "session": _hkey(key),
                       "request_id": a.get("request_id"), "router": a.get("router"),
                       "error": "missed: the shadow read past the question's position first"})

    def _journal(self, rec: dict) -> None:
        self.recent.append(rec)
        if not self.journal_path:
            return
        try:
            os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ inside a turn
    def inline(self, key: str, *, share: float = INLINE_SHARE, slice_ms: float = INLINE_SLICE_MS,
               budget=None):
        """What one turn of session `key` calls between two of Qwen's rounds (`Inline.tick`).
        `budget` (v3, narrator.Share): the request's small-model time, which the voice draws on
        first; the reading takes at most what is left of it, and at most its own `share`."""
        return Inline(self, key, share=share, slice_ms=slice_ms, budget=budget)

    # ------------------------------------------------------------------ the narrator's job
    def thought_slot(self, key: str):
        """What `narrator.Thoughts` reads from instead of the narrator's own slot when the profile
        has `shadow.narrator: on` (see `ThoughtSlot`)."""
        return ThoughtSlot(self, key)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        with self.mu:
            sess = [{"session": _hkey(k), "read": s.pos, "target": len(s.target),
                     "behind": s.behind(), "bytes": s.nbytes(), "checkpoints": len(s.checks),
                     "stats": dict(s.stats, ms=round(s.stats["ms"], 1))}
                    for k, s in self.sessions.items()]
            parked = [{"session": _hkey(k), "read": v["pos"], "bytes": v["bytes"]}
                      for k, v in self.parked.items()]
        asked = self.router["asked"]
        return {"enabled": True, "model": self.model.name, "slot": self.slot.status(),
                "grid": self.grid, "cap_bytes": self.cap, "disk": self.disk_dir,
                "resident_bytes": sum(x["bytes"] for x in sess), "sessions": sess,
                "pool_bytes": dict(zip(("reserved", "allocated"), self.pool_bytes())),
                "parked": parked, "stats": dict(self.stats),
                "router": dict(self.router, agreement=(round(self.router["agree"] / asked, 4)
                                                       if asked else None)),
                "recent": list(self.recent)[-20:], "error": self.last_error}


class Inline:
    """The shadow's reading inside one turn, on the request's thread between two of Qwen's
    rounds (the engine lock is the request's): at most `share` of the time since Qwen's first
    token, in slices of at most `slice_ms` (whole grid chunks; the next chunk's cost is the
    last one's), the device synchronised first so nothing of the small model runs beside Qwen.
    It reads the prompt first, then Qwen's output so far; it answers a question it reaches (the
    router's) and never touches a session whose fork is open (the narrator's summary)."""

    def __init__(self, svc: ShadowService, key: str, *, share: float, slice_ms: float,
                 budget=None):
        self.svc, self.key = svc, key
        self.share, self.slice_ms = float(share), float(slice_ms)
        self.budget = budget
        # the share counts from Qwen's first token: the prefill is one long call the reader cannot
        # slice into, and a share earned there and spent in the decode slowed the decode by 5 %
        # (the A/B of 2026-10-04: 36.0 -> 34.1 tok/s with the share counted from the turn's start)
        self.t0: float | None = None
        self.spent = 0.0
        self.est = 50.0                      # ms a chunk, until one is measured
        self.stats = {"slices": 0, "rows": 0, "ms": 0.0, "asks": 0}
        self.broken: str | None = None

    def tick(self, out_ids: list[int]) -> None:
        if self.broken:
            return
        svc = self.svc
        now = svc.clock()
        if self.t0 is None:
            self.t0 = now
            return
        allow = self.share * (now - self.t0) * 1e3 - self.spent
        if self.budget is not None:
            self.budget.start(now)
            allow = min(allow, self.budget.allow(now))
        if allow < self.est:
            return
        try:
            if svc.ready() is not None:
                return
            cur = svc.sessions.get(self.key)
            if cur is not None and cur.fork is not None:
                return                       # the narrator's summary is being written on it
            dev = getattr(svc.slot, "device", None)
            if dev is not None and getattr(dev, "cuda", False):
                import torch
                torch.cuda.synchronize(dev.device)
            t = svc.clock()
            with svc._scope():
                r = svc._get_reader()
                st = svc._apply(self.key)
                if st is None or st.fork is not None:
                    return
                if r.committable(st) < r.grid and not st.asks:
                    svc.extend(self.key, out_ids)
                    st = svc._apply(self.key)
                chunks = max(1, int(min(self.slice_ms, allow) // max(self.est, 1.0)))
                n = [0]

                def stop():
                    n[0] += 1
                    return n[0] > chunks
                out = {"asks": 0, "stopped": False}
                rows = svc._read(self.key, st, float("inf"), stop, out)
            ms = (svc.clock() - t) * 1e3
            self.spent += ms
            if self.budget is not None:
                self.budget.spend(ms)
            self.stats["slices"] += 1
            self.stats["rows"] += rows
            self.stats["asks"] += out["asks"]
            self.stats["ms"] += ms
            if rows and not out["asks"]:
                self.est = 0.5 * self.est + 0.5 * ms * r.grid / rows
            svc.stats["inline_rows"] += rows
            svc.stats["inline_ms"] += round(ms, 2)
        except Exception as exc:                                  # noqa: BLE001
            self.broken = f"{type(exc).__name__}: {exc}"[:200]
            svc.stats["inline_errors"] += 1
            svc.log(f"[shadow] inline reading off for this turn ({self.broken})", flush=True)

    def totals(self) -> dict:
        return dict(self.stats, ms=round(self.stats["ms"], 2),
                    decode_ms=round((self.svc.clock() - self.t0) * 1e3, 1)
                    if self.t0 is not None else 0.0,
                    **({"broken": self.broken} if self.broken else {}))


class ThoughtJob:
    """The narrator's summary of a thought read from the session's shadow: what is missing of the
    session up to the thought (grid chunks), then a `Fork` with the question and the answer, in
    slices. Quacks like engine/generative/base.py `Job` for `Thoughts`."""

    def __init__(self, svc: ShadowService, key: str, max_new: int):
        self.svc, self.key, self.max_new = svc, key, int(max_new)
        self.fork = None
        self.out: list[int] = []
        self.ids: list[int] = []
        self.finish: str | None = None
        self.reused = 0
        self.ms = 0.0
        self.slices = 0
        self.read = 0

    @property
    def done(self) -> bool:
        return self.finish is not None

    def advance(self, deadline: float) -> None:
        svc = self.svc
        r = svc.reader
        if self.fork is None:
            s = svc._apply(self.key)
            if s is None:
                raise RuntimeError("the session has no shadow")
            self.read += r.advance(s, deadline=deadline)
            if r.committable(s) >= r.grid:
                return                               # more to read: the next slice goes on
            tail = svc.model.tok(THOUGHT_TAIL.format(q=THOUGHT_QUESTION),
                                 add_special_tokens=False)["input_ids"]
            self.reused = s.pos
            self.fork = r.fork(s, len(s.target), list(tail))
            self.ids = self.fork.ids
        f = self.fork
        if not f.prefill(deadline, svc.clock):
            return
        if f.decode(self.max_new, svc.model.eos, deadline, svc.clock):
            self.out = list(f.out)
            self.finish = f.finish
            f.close()
        else:
            self.out = list(f.out)

    def cancel(self) -> None:
        if self.fork is not None:
            self.fork.close()


class ThoughtSlot:
    """The narrator slot's face over the shadow (`ensure`, `begin`, `advance`, `model.job_text`,
    `calls`) for one session."""

    def __init__(self, svc: ShadowService, key: str):
        self.svc, self.key = svc, key
        self.model = svc.model
        self.calls = svc.slot.calls

    @property
    def state(self) -> str:
        return self.svc.slot.state

    def ensure(self) -> str | None:
        return self.svc.ready()

    def begin_thought(self, prompt_ids: list[int], out_ids: list[int], max_new: int) -> dict:
        self.svc.extend(self.key, out_ids)
        return {"job": ThoughtJob(self.svc, self.key, max_new)}

    def advance(self, job: ThoughtJob, *, budget_ms: float) -> dict:
        svc = self.svc
        t0 = svc.clock()
        try:
            dev = getattr(svc.slot, "device", None)
            if dev is not None and getattr(dev, "cuda", False):
                import torch
                torch.cuda.synchronize(dev.device)
            with svc._scope():
                svc._get_reader()
                job.advance(t0 + budget_ms / 1000.0)
            ms = round((svc.clock() - t0) * 1e3, 2)
        except Exception as exc:                                  # noqa: BLE001
            job.cancel()
            self.calls["error"] += 1
            return {"why": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
        job.ms += ms
        job.slices += 1
        if job.done:
            self.calls[job.finish] += 1
        return {"ms": ms, "done": job.done}


# ---------------------------------------------------------------------- the server's wiring
def configured() -> str:
    from engine.settings import SETTINGS as _S
    return (_S.get("SHADOW_MODEL") or "").strip()


def start(state: dict, gate, *, fake: bool = False, log=print):
    """The shadow reader for this server, or None when `QWEN38_SHADOW_MODEL` is empty. It shares
    the narrator's slot when both name the same model (one copy of the weights); else it has a
    slot of its own (role `shadow` in the residency planner). Nothing loads here."""
    from engine.settings import SETTINGS as _S
    name = configured()
    if not name:
        return None
    caps = getattr(state.get("engine"), "caps", None)
    if caps is not None and not getattr(caps, "shadow", True):
        state["shadow_refused"] = NOT_VALIDATED_WHY
        log(f"[shadow] {name}: {NOT_VALIDATED_WHY}")
        return None
    from engine import generative
    from engine.generative.qwen35 import NAMES
    if name not in NAMES and not name.startswith(generative.FAKE_PREFIXES):
        raise ValueError(f"QWEN38_SHADOW_MODEL={name}: the shadow reader needs a Qwen3.5 model "
                         f"({', '.join(NAMES)})")
    narr = state.get("narrator")
    slot = None
    if narr is not None and getattr(narr.slot.model, "name", None) == name:
        slot = narr.slot
    else:
        from engine.generative.slot import Slot
        from server import decider_api
        engine_dir = getattr(getattr(state.get("engine"), "cfg", None), "path", None)
        model = generative.model(name, _S.get("SHADOW_DIR") or None, allow_fake=fake,
                                 engine_dir=engine_dir if isinstance(engine_dir, str) else None)
        device = None
        if not fake:
            from engine.decision.device import Device
            device = Device(_S.get("NARRATOR_DEVICE"))
        slot = Slot(model, role="shadow", gate=gate,
                    planner=planner(state, fake=fake), device=device,
                    log=lambda s: log(s, flush=True),
                    mem_available=lambda: decider_api._mem_available_gib(fake))
    cond = state.get("conductor")
    disk = _S.get("SHADOW_DISK") or ""
    if disk == "auto":
        disk = os.path.join(os.path.expanduser("~/.qwen38-spark-engine/cache"), "shadow")
    svc = ShadowService(slot, grid=int(_S.get("SHADOW_GRID") or GRID),
                        cap_gib=float(_S.get("SHADOW_GB") or 4.0), disk_dir=disk or None,
                        disk_gib=float(_S.get("SHADOW_DISK_GB") or 20.0),
                        root=cond.root if cond is not None else None, log=log)
    log(f"[shadow] {name} ({'the narrator slot' if slot is getattr(narr, 'slot', None) else 'own slot'}"
        f"), grid {svc.grid}, cap {svc.cap / 2**30:.1f} GiB, disk {svc.disk_dir or 'off'}: loads "
        f"when a profile with shadow.mode on sends its first turn", flush=True)
    return svc


# the shadow's claim when the stored residency document has none (it predates SR1): the 0.8B's
# weights (1.6 GiB), its workspace, and QWEN38_SHADOW_GB of session states
def claim_gib() -> float:
    from engine.settings import SETTINGS as _S
    from engine.generative.qwen35 import NAMES
    meta = NAMES.get(configured()) or {}
    return round(float(meta.get("claim_gib") or 2.5) + float(_S.get("SHADOW_GB") or 4.0), 2)


def planner(state: dict, *, fake: bool = False):
    """The residency planner's check for role `shadow` (decider_api.planner's), with the built-in
    claim added to a stored document that has none."""
    from server import decider_api
    inner = decider_api.planner(state, fake=fake, role="shadow")

    def check(name, *, loading, service_start, hold_restart):
        cond = state.get("conductor")
        if cond is not None:
            doc = cond.store.peek("residency")[1]
            if "shadow" not in (doc.get("claims") or {}):
                from server.conductor import residency, schema
                from server.decider_api import _mem_available_gib, held, _other_loading
                from server import memlimits
                doc = dict(doc, claims=dict(doc.get("claims") or {}, shadow={
                    "gib": claim_gib(), "basis": "declared",
                    "note": "the shadow reader's model and its session states (SR1)"}))
                roles = cond.store.peek("roles")[1]
                peak, now = memlimits.engine_gib(state)
                d = residency.check("shadow", _mem_available_gib(fake), doc, roles,
                                    hold=held(service_start=service_start,
                                              hold_restart=hold_restart),
                                    loading=loading or _other_loading(state, "shadow"),
                                    engine_peak_gib=peak, engine_now_gib=now)
                residency.Journal(cond.root).write(dict(d, kind="check", source=f"engine:{name}"))
                return d
        return inner(name, loading=loading, service_start=service_start,
                     hold_restart=hold_restart)

    return check


def status(state: dict) -> dict:
    svc = state.get("shadow")
    if svc is None:
        return {"enabled": False, "why": state.get("shadow_refused") or OFF_WHY}
    return svc.status()
