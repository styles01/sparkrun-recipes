"""Probe snapshots of Qwen's own activations, and the probe stage that reads them (ENG-181, R2).

A probe is a linear head on rows Qwen already computes (Harmony 2, form 1; Laya study 3). Before a
head can be trained the rows must exist, so this module copies them at the router's three decision
points, without an extra forward and without changing a byte of what Qwen writes:

    prompt   the last row the turn shares with the prompt the client sent: the last token of the
             last message, right before the insertion point where the router's note goes
             (class, effort). The prompt's very last token comes after that note and the thinking
             splice, so a probe trained on it could only ever answer after the router had
             already decided; this row is the same with or without them, and the probe stage can
             read it before deciding (`ProbeStage`, the in-context path)
    think    every `QWEN38_PROBE_EVERY` (1,024) reasoning tokens, the mean of the rows the
             verified block that crossed the line committed (stuck, already decided)
    finish   the row that decided an end token, the finish candidate (done). The last one of a
             turn that ended with tool calls is not kept: a turn that hands a call to the client
             claims nothing about being done (night 2: 17 of 1,023 turns ended without a call)

Each point is one pooled row per tapped layer -- the five target layers the block drafters read
through `eng.tap` (`tapped()` of the served drafter) -- plus the `hidden_post_norm` row the drafter
sync reads, 6 x 5,120 values, fp16 on disk: 61,440 bytes a point. A drafter without taps gives the
post-norm row alone.

ON THE DEVICE the copy is a gather, a mean over the block's rows, a cast and one asynchronous copy
into a pinned host slot, queued on the request's stream behind the forward that wrote the rows,
with an event; the host never waits. The slots (128 of 6 x hidden fp16, 7.9 MB at 5,120) are
allocated once when the ring opens, never per point. A request takes at most `MAX_POINTS` points;
with no slot free a point is skipped and counted.

ON THE DISK, a ring under the engine's home (never a repository, refused at start):

    ~/.qwen38-spark-engine/probe/snapshots/<yyyy-mm-dd>/
        rows-<weights12>-<R>x<H>.f16   append-only records, R x H fp16 each
        index.jsonl                    one line a record: where it is, the request, the journal
                                       row id, the session, the class and effort the router set,
                                       the point, its position, and the outcome keys the
                                       journal's later rows join on (request id, row id)

A writer thread waits for each request's events after the request ended, appends its records and
journals one row a request (`kind = probe`: counts and bytes, never a value). A day stops taking
records at `QWEN38_PROBE_DAY_MB` (100 MB: the growth gate), journaled. Day directories older than
30 days go at start and nightly at 04:00 (the owner, 2026-10-01: box only, 30 days, tool requests
only). Activations are not text, and no text is ever written here.

SCOPE. A request through `TandemLLM` that offered tools, one choice, not a replay from the response
cache. A raw request (`X-Tandem: raw`) has no active plan and is never seen; a chat without tools
neither. `QWEN38_PROBE_SNAPSHOTS=off` (the default): nothing here is created, the loop's probe
argument is None and every line of it runs as before.

THE PROBE STAGE (`classifier.probe_stage` in the router document, off by default): a trained probe
file (tools/probe_train.py) answers the class from the `prompt` rows in well under a millisecond on
the host. `shadow`: the answer of the request's own snapshot is journaled beside the router's
(`extra.probe`), nothing changes. `on`: where the plan decides under the engine lock (the in-context
stage's follow-ups and long prompts) the probe is asked first, on a prefill of the turn's prompt up
to its insertion point; at or above `accept` it decides and the question is not asked. A probe of
other weights, other rows or another hidden width is refused.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import queue
import re
import shutil
import threading
import time

import numpy as np
import torch

from engine import capture_lock
from engine.settings import SETTINGS as _S

DEFAULT_ROOT = "~/.qwen38-spark-engine/probe"
RETENTION_DAYS = 30
POINTS = ("prompt", "think", "finish")
SLOTS = 128
MAX_ROWS = 6
MAX_POINTS = 96
FINISH_MAX = 4
QUEUE_MAX = 256
INDEX_LINE_BYTES = 420            # what an index line costs, for the growth arithmetic
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def configured() -> bool:
    return (_S.get("PROBE_SNAPSHOTS") or "off").strip().lower() == "on"


def root_dir() -> str:
    return os.path.realpath(os.path.expanduser(_S.get("PROBE_DIR") or DEFAULT_ROOT))


def weights_key(paths) -> str:
    """A fingerprint of the served weights: every weight file's name, size and modification time
    (a directory: its *.safetensors). Hashing 15 GB at every start would take minutes; a file
    that changes changes its size or its time. A probe trained on one key refuses another."""
    items = []
    for p in paths or ():
        if not p:
            continue
        for part in str(p).split(","):
            part = os.path.realpath(os.path.expanduser(part))
            files = ([os.path.join(part, f) for f in sorted(os.listdir(part))
                      if f.endswith(".safetensors")] if os.path.isdir(part) else [part])
            for f in files:
                try:
                    st = os.stat(f)
                except OSError:
                    continue
                items.append(f"{os.path.basename(f)}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(items).encode()).hexdigest()[:16] if items else "none"


def tapped(drafter):
    """(row names, tap mode, rows) of the last forward: the served drafter's tapped target
    layers, sorted, or nothing when it has no tap (an n-gram drafter, a released arm)."""
    fn = getattr(drafter, "tapped", None) if drafter is not None else None
    got = fn() if fn is not None else None
    if not got:
        return [], None, []
    layers, mode, rows = got
    return [f"tap{int(x)}" for x in layers], mode, list(rows)


def rows_of(eng, drafter) -> tuple[list[str], str | None, list]:
    """Every row source of the last forward: the tapped layers, then the post-norm rows."""
    names, mode, rows = tapped(drafter)
    post = getattr(eng, "hidden_post_norm", None)
    if post is not None:
        names, rows = names + ["post"], rows + [post[0]]
    return names, mode, rows


def wants(plan, body: dict, n_choices: int = 1) -> bool:
    """A request the snapshots cover: TandemLLM (an active plan), tools offered, one choice. Raw
    and chats without tools never; a response-cache replay runs no forward and writes nothing."""
    return (plan is not None and getattr(plan, "active", False) and plan.row is not None
            and bool(body.get("tools")) and n_choices == 1)


def night_bytes(prompts: int, think_points: int, finishes: int, rows: int = MAX_ROWS,
                hidden: int = 5120) -> int:
    """What a night writes: each point is `rows x hidden` fp16 plus its index line."""
    return (prompts + think_points + finishes) * (rows * hidden * 2 + INDEX_LINE_BYTES)


# --------------------------------------------------------------------------- one request
class Capture:
    """One request's snapshots, from the device to a ring slot. Every method is a few host
    operations unless a point is due; nothing here waits for the device."""

    def __init__(self, ring: "Ring", probe_pos: int | None, *, every: int | None = None,
                 prompt_len: int | None = None):
        self.ring = ring
        self.pos = probe_pos
        self.prompt_len = prompt_len
        self.every = int(every or ring.every)
        self.n = 0                       # reasoning tokens counted at block boundaries
        self.next = self.every
        self.points: list[dict] = []
        self.finishes: list[dict] = []
        self.skipped = 0
        self.missed = 0
        self._prev = None                # where the chunk before the next one began
        self.names: list[str] = []
        self.tap_mode = None

    # ------------------------------------------------------------ the prefill
    def chunks(self, on_prefill, info: dict):
        """`cache.prefill`'s chunk hook with the prompt point's capture in front: the rows of
        each chunk are the tap's and `hidden_post_norm`'s until the next forward."""
        def hook(done, total):
            self._chunk(info.get("start", 0), done)
            if on_prefill is not None:
                on_prefill(done, total)
        hook.info = info
        return hook

    def bind(self, eng, drafter) -> None:
        self._eng, self._drafter = eng, drafter

    def _chunk(self, start: int, done: int) -> None:
        lo = self._prev if self._prev is not None else start
        self._prev = done
        if self.pos is not None and lo <= self.pos < done and not self._has("prompt"):
            self._take("prompt", self._eng, self._drafter, slice(self.pos - lo, self.pos - lo + 1),
                       at=self.pos)

    def prefilled(self, eng, drafter, n: int, start: int) -> None:
        """After the prefill: its last chunk (or its only forward) is still in the rows."""
        self._eng, self._drafter = eng, drafter
        self._chunk(start, n)
        if self.pos is not None and not self._has("prompt"):
            self.missed += 1             # the turn resumed past the row (a resident hit)

    def _has(self, point: str) -> bool:
        return any(p["point"] == point for p in self.points)

    # ------------------------------------------------------------ the decode
    def due(self, think, n_new: int) -> bool:
        """A verified block of `n_new` tokens that started inside the reasoning: is a think
        point due? Counts the reasoning tokens either way."""
        if think is None or not think.inside or n_new <= 0:
            return False
        self.n += n_new
        if self.n < self.next:
            return False
        self.next = (self.n // self.every + 1) * self.every
        return True

    def think(self, eng, drafter, sel) -> None:
        self._take("think", eng, drafter, sel, at=self.n)

    def finish(self, eng, drafter, sel, at: int) -> None:
        if len(self.finishes) < FINISH_MAX:
            self._take("finish", eng, drafter, sel, at=at)

    def _take(self, point: str, eng, drafter, sel, at: int) -> None:
        if len(self.points) >= MAX_POINTS:
            self.skipped += 1
            return
        names, mode, rows = rows_of(eng, drafter)
        if not rows or (self.names and names[:MAX_ROWS] != self.names):
            self.missed += 1             # no rows, or an arm released mid-request changed them
            return
        slot = self.ring.acquire()
        if slot is None:
            self.skipped += 1
            return
        R = min(len(rows), MAX_ROWS)
        with torch.no_grad():
            x = torch.stack([r[sel] for r in rows[:R]])          # [R, k, H]
            v = x.float().mean(1).to(torch.float16)
            dst = self.ring.host[slot, :R]
            dst.copy_(v, non_blocking=dst.is_pinned())
            ev = None
            if v.is_cuda:
                ev = torch.cuda.Event()
                ev.record()
        self.names, self.tap_mode = names[:R], mode
        rec = {"point": point, "slot": slot, "rows": R, "k": int(x.shape[1]), "at": int(at),
               "event": ev}
        self.points.append(rec)
        if point == "finish":
            self.finishes.append(rec)

    # ------------------------------------------------------------ the end
    def vector(self, point: str = "prompt") -> torch.Tensor | None:
        """The host copy of a point's rows, fp32 [R, H] (waits for its event: called after the
        request's last token, when the copy has long finished)."""
        p = next((p for p in self.points if p["point"] == point), None)
        if p is None:
            return None
        if p["event"] is not None:
            p["event"].synchronize()
        return self.ring.host[p["slot"], :p["rows"]].float()

    def kept(self, tool_calls: int) -> list[dict]:
        """The points to write: the turn's last finish candidate goes when it ended in calls."""
        drop = self.finishes[-1] if tool_calls and self.finishes else None
        return [p for p in self.points if p is not drop]

    def release(self) -> None:
        for p in self.points:
            self.ring.free(p["slot"])
        self.points, self.finishes = [], []


# --------------------------------------------------------------------------- the ring
class Ring:
    """The pinned slots, the writer thread and the day directories on the NVMe."""

    def __init__(self, root: str | None = None, *, hidden: int, journal=None,
                 every: int | None = None, day_mb: float | None = None, weights: str = "none",
                 clock=time.time, device: str | None = None, slots: int = SLOTS):
        self.root = os.path.realpath(os.path.expanduser(root or root_dir()))
        self.dir = os.path.join(self.root, "snapshots")
        self.hidden = int(hidden)
        self.journal = journal
        self.every = int(every if every is not None else _S.get("PROBE_EVERY"))
        self.day_bytes = int(float(day_mb if day_mb is not None else _S.get("PROBE_DAY_MB"))
                             * 1e6)
        self.weights = weights
        self.clock = clock
        pin = device is not None and str(device).startswith("cuda") and torch.cuda.is_available()
        self.host = torch.empty((slots, MAX_ROWS, self.hidden), dtype=torch.float16,
                                pin_memory=pin)
        self._free = list(range(slots))
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(QUEUE_MAX)
        self._thread = None
        self._next_prune = 0.0
        self.stats = {"requests": 0, "records": 0, "bytes": 0, "skipped": 0, "missed": 0,
                      "day_cap": 0, "failed": 0, "dropped": 0, "pruned": 0}
        from server.conductor.transcripts import in_repository
        self.repo = in_repository(self.root)
        if self.repo is not None:
            raise ValueError(f"the probe snapshots may not live inside a repository "
                             f"({self.root} is in {self.repo})")

    # ------------------------------------------------------------ slots
    def acquire(self) -> int | None:
        with self._lock:
            return self._free.pop() if self._free else None

    def free(self, slot: int) -> None:
        with self._lock:
            self._free.append(slot)

    def capture(self, probe_pos: int | None, prompt_len: int | None = None) -> Capture:
        return Capture(self, probe_pos, prompt_len=prompt_len)

    # ------------------------------------------------------------ the writer
    def start(self) -> "Ring":
        self._thread = threading.Thread(target=self._run, name="probe-snapshots", daemon=True)
        self._thread.start()
        return self

    def submit(self, cap: Capture, meta: dict, tool_calls: int = 0) -> None:
        """The request ended: its points go to the writer, or back to the pool."""
        keep = cap.kept(tool_calls)
        for p in cap.points:
            if p not in keep:
                self.free(p["slot"])
        cap.points = keep
        self.stats["skipped"] += cap.skipped
        self.stats["missed"] += cap.missed
        if not keep:
            cap.points = []
            return
        job = (cap, dict(meta))
        if self._thread is None:
            self.write(*job)
            return
        try:
            self._q.put_nowait(job)
        except queue.Full:
            self.stats["dropped"] += 1
            cap.release()

    def _run(self) -> None:
        while True:
            try:
                job = self._q.get(timeout=1.0)
            except queue.Empty:
                job = None
            if job is not None:
                try:
                    self.write(*job)
                except Exception as exc:                           # noqa: BLE001
                    self._failed(job[1], exc)
                    job[0].release()
                finally:
                    self._q.task_done()
            if self.clock() >= self._next_prune:
                from server.ledger import next_local
                self._next_prune = next_local(4, self.clock())
                try:
                    self.prune()
                except Exception as exc:                           # noqa: BLE001
                    print(f"[probe] prune failed: {type(exc).__name__}: {exc}", flush=True)

    def flush(self, timeout: float = 5.0) -> None:
        end = time.time() + timeout
        while self._q.unfinished_tasks and time.time() < end:
            time.sleep(0.01)

    def day(self, ts: float | None = None) -> str:
        return _dt.datetime.fromtimestamp(self.clock() if ts is None else ts).strftime("%Y-%m-%d")

    def day_used(self, day: str) -> int:
        d = os.path.join(self.dir, day)
        try:
            return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        except OSError:
            return 0

    def write(self, cap: Capture, meta: dict) -> int:
        """Append one request's records and index lines; journal the request. Returns bytes."""
        try:
            pts = cap.points
            R = max(p["rows"] for p in pts)
            pts = [p for p in pts if p["rows"] == R]   # one layout a request (always, in practice)
            rec_bytes = R * self.hidden * 2
            day = self.day(meta.get("ts"))
            need = len(pts) * (rec_bytes + INDEX_LINE_BYTES)
            if self.day_used(day) + need > self.day_bytes:
                self.stats["day_cap"] += 1
                self._note(meta, "day_cap", {"points": len(pts), "day": day,
                                             "day_mb": self.day_bytes / 1e6})
                return 0
            d = os.path.join(self.dir, day)
            for p in (self.root, self.dir, d):
                os.makedirs(p, mode=0o700, exist_ok=True)
            name = f"rows-{self.weights[:12]}-{R}x{self.hidden}.f16"
            path = os.path.join(d, name)
            lines, data = [], []
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                off = os.fstat(fd).st_size
                for p in pts:
                    if p["event"] is not None:
                        with capture_lock.LOCK:        # not inside a graph capture
                            p["event"].synchronize()
                    data.append(self.host[p["slot"], :R].numpy().tobytes())
                    lines.append(json.dumps(dict(
                        meta, file=name, off=off + rec_bytes * len(lines), point=p["point"],
                        at=p["at"], k=p["k"], rows=cap.names, tap_mode=cap.tap_mode,
                        dim=self.hidden, weights=self.weights), ensure_ascii=False))
                os.write(fd, b"".join(data))
            finally:
                os.close(fd)
            fd = os.open(os.path.join(d, "index.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                         0o600)
            try:
                os.write(fd, "".join(line + "\n" for line in lines).encode())
            finally:
                os.close(fd)
            n = len(pts) * rec_bytes
            self.stats["requests"] += 1
            self.stats["records"] += len(pts)
            self.stats["bytes"] += n
            counts: dict[str, int] = {}
            for p in pts:
                counts[p["point"]] = counts.get(p["point"], 0) + 1
            self._note(meta, "written", {"points": counts, "bytes": n, "day": day,
                                         "skipped": cap.skipped, "missed": cap.missed})
            return n
        finally:
            cap.release()

    def _note(self, meta: dict, decision: str, detail: dict) -> None:
        if self.journal is not None:
            self.journal.note("probe", "snapshot", decision=decision, detail=detail,
                              session_key=meta.get("session"), request_id=meta.get("request_id"))

    def _failed(self, meta: dict, exc: BaseException) -> None:
        self.stats["failed"] += 1
        msg = f"{type(exc).__name__}: {exc}"[:300]
        print(f"[probe] snapshot write failed ({msg}); the request was not affected", flush=True)
        try:
            self._note(meta, "write_failed", {"error": msg})
        except Exception:                                          # noqa: BLE001
            pass

    # ------------------------------------------------------------ retention
    def prune(self, now: float | None = None) -> dict:
        """Day directories older than 30 days go. Journaled."""
        now = self.clock() if now is None else now
        cut = (_dt.datetime.fromtimestamp(now) - _dt.timedelta(days=RETENTION_DAYS)
               ).strftime("%Y-%m-%d")
        gone = []
        try:
            days = sorted(os.listdir(self.dir))
        except OSError:
            return {"deleted": 0}
        for day in days:
            if _DAY.match(day) and day < cut:
                shutil.rmtree(os.path.join(self.dir, day), ignore_errors=True)
                gone.append(day)
        self.stats["pruned"] += len(gone)
        if gone and self.journal is not None:
            self.journal.note("probe", "prune", decision="deleted",
                              detail={"days": gone, "retention_days": RETENTION_DAYS})
        return {"deleted": len(gone), "days": gone}

    def info(self) -> dict:
        with self._lock:
            free = len(self._free)
        return {"root": self.root, "every": self.every, "day_mb": self.day_bytes / 1e6,
                "retention_days": RETENTION_DAYS, "weights": self.weights, "slots_free": free,
                "queue": self._q.qsize(), **self.stats}


# --------------------------------------------------------------------------- the probe stage
class Probe:
    """A trained probe file (tools/probe_train.py): standardised rows -> class logits."""

    def __init__(self, d: dict, path: str = ""):
        self.path = path
        self.name = os.path.basename(path) or "probe"
        self.question = d["question"]
        self.classes = list(d["classes"])
        self.rows = list(d["rows"])
        self.dim = int(d["dim"])
        self.weights = d.get("weights") or "none"
        self.temperature = float(d.get("temperature") or 1.0)
        W, b = d["W"].float(), d["b"].float()        # [c, D], on standardised rows
        if W.shape != (len(self.classes), len(self.rows) * self.dim):
            raise ValueError(f"probe {self.name}: W is {tuple(W.shape)}, not "
                             f"{len(self.classes)} x {len(self.rows) * self.dim}")
        # the standardisation and the temperature folded in once: one GEMV an answer, in numpy
        # (torch.mv takes 107 us for this shape on the box's CPU, numpy's BLAS 15 us)
        W = W / d["sd"].float() / self.temperature
        self.b = (b / self.temperature - W @ d["mu"].float()).numpy()
        self.W = np.ascontiguousarray(W.numpy())
        self._idx: tuple | None = None

    @classmethod
    def load(cls, path: str) -> "Probe":
        return cls(torch.load(path, map_location="cpu", weights_only=True), path)

    def refuses(self, names: list[str], dim: int, weights: str) -> str | None:
        """Why this probe cannot read these rows, or None."""
        if weights != self.weights:
            return "weights"
        if dim != self.dim:
            return "dim"
        if any(r not in names for r in self.rows):
            return "rows"
        return None

    def answer(self, vec: torch.Tensor, names: list[str]) -> tuple[str, float, dict, float]:
        """(label, confidence, probabilities, host ms) for one point's rows [R, H]."""
        t0 = time.perf_counter()
        if self._idx is None or self._idx[0] != names:
            idx = [names.index(r) for r in self.rows]
            self._idx = (list(names), None if idx == list(range(len(names))) else idx)
        x = vec if self._idx[1] is None else vec[self._idx[1]]
        z = self.W @ x.reshape(-1).numpy() + self.b
        e = np.exp(z - z.max())
        p = (e / e.sum()).tolist()
        i = max(range(len(p)), key=p.__getitem__)
        ms = (time.perf_counter() - t0) * 1e3
        return self.classes[i], p[i], {c: round(v, 4) for c, v in zip(self.classes, p)}, ms


class ProbeStage:
    """The router's probe stage: the probe files it may load, cached by path and time."""

    def __init__(self, root: str | None = None, *, weights: str = "none", hidden: int = 0):
        self.root = os.path.realpath(os.path.expanduser(root or root_dir()))
        self.weights = weights
        self.hidden = int(hidden)
        self._cache: dict[str, tuple[float, Probe | None, str | None]] = {}

    def path(self, name: str) -> str:
        return name if os.path.isabs(name) else os.path.join(self.root, "probes", name)

    def get(self, cfg: dict) -> tuple[Probe | None, str | None]:
        """The configured probe and None, or None and why not."""
        name = (cfg or {}).get("file") or ""
        if not name:
            return None, "no_file"
        p = self.path(name)
        try:
            mt = os.path.getmtime(p)
        except OSError:
            return None, "missing"
        hit = self._cache.get(p)
        if hit is None or hit[0] != mt:
            try:
                pr, why = Probe.load(p), None
                if pr.weights != self.weights:
                    pr, why = None, "weights"
                elif self.hidden and pr.dim != self.hidden:
                    pr, why = None, "dim"
            except Exception as exc:                               # noqa: BLE001
                pr, why = None, f"load:{type(exc).__name__}"
            hit = self._cache[p] = (mt, pr, why)
        return hit[1], hit[2]

    def ready(self, cfg: dict) -> bool:
        return self.get(cfg)[0] is not None

    def stage(self, cfg: dict, vec, names: list[str], *, reason: str) -> dict:
        """The journal's stage dict for one answer (an `error` when it could not answer)."""
        probe, why = self.get(cfg)
        d = {"stage": "probe", "label": None, "confidence": 0.0, "probs": {}, "ms": 0.0,
             "error": None, "model": probe.name if probe is not None else (cfg or {}).get("file"),
             "accept": float((cfg or {}).get("accept", 0.9)), "reason": reason}
        if probe is not None and vec is None:
            why = "no_rows"
        if probe is not None and why is None:
            why = probe.refuses(names, int(vec.shape[-1]), self.weights)
        if why is not None:
            d["error"] = f"probe_{why}"
            return d
        label, conf, probs, ms = probe.answer(vec, names)
        d.update(label=label, confidence=round(conf, 4), probs=probs, ms=round(ms, 4))
        if conf < d["accept"]:
            d["below"] = True
        return d


def start(state: dict, journal, *, log=print) -> Ring | None:
    """The ring for this server, or None with `QWEN38_PROBE_SNAPSHOTS` off (nothing created)."""
    if not configured():
        log("[probe] snapshots off", flush=True)
        return None
    hidden = getattr(getattr(state.get("engine"), "cfg", None), "hidden_size", None)
    if not hidden:
        log("[probe] snapshots on, but this engine has no rows to copy (the fake engine)",
            flush=True)
        return None
    ring = Ring(root_dir(), hidden=int(hidden), journal=journal,
                weights=state.get("weights_key") or "none", device=state.get("device", "cuda"))
    log(f"[probe] snapshots on: {ring.root}, every {ring.every} reasoning tokens, "
        f"{ring.day_bytes / 1e6:g} MB a day, {RETENTION_DAYS} days, weights {ring.weights}",
        flush=True)
    return ring.start()
