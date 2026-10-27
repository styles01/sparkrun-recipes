"""Per-session transcripts on the Spark's NVMe (SRV-137, Harmony 5).

The owner asked on 2026-10-01 for the whole conversation per session as markdown on the box, so the
harness can read what a client lost after a compaction. One block per request, appended when its
stream closed:

    ~/.qwen38-spark-engine/memory/sessions/<yyyy-mm>/<session>/
        transcript.md    append-only, one block per request, 0600
        meta.json        the session's key, client, profile, project, first and last request,
                         counts, `keep`, and `bytes`: where the last complete block ends

A block holds the request's new messages (everything after the last assistant message: the tool
results of the previous turn, a new user message), each tool result clipped at `clip_chars`
(4,000) with its original size, the assistant's answer or its tool calls (the reasoning only as its
token count; a turn the D1 loop guard cut keeps the first and last `loop_chars` of it), and a
runtime section: every hidden note with the journal row it is in, the notes replayed, the acts,
the checker's RESULT line. opencode's compaction request gets one block: tokens in and out, and
the summary. The format is fixed by golden files (tests/fixtures/transcripts/).

Scope (a design decision, confirmed 2026-10-01): a request with tools is written in full; a chat
without tools changes the session's counters only. `settings.memory.transcripts.scope` says
`tools` (the default), `all` or `none`. A raw request (`X-Tandem: raw`) is never seen here.

Nothing here is on a request's path. `Conductor.finish` hands the plan and the record to
`submit()`, which queues them (a full queue drops the job and counts it); one writer thread builds
the block -- decodes the output, redacts, clips -- and appends it. Every block passes the
secret-pattern refusal (server/conductor/redact.py) before it is written; the journal gets the
counts per kind (`kind = memory`, never the value). A write that fails (disk full, permission) is
journaled and never reaches the client. The writer prunes at start and nightly at 04:00: a session
idle for longer than `retention_days` (30) goes, unless it is marked keep.

The directory is the engine's home on the box, never a repository: a memory root inside a git
work tree is refused at start, and the ledger's weekly backup never sees it (it copies the ledger
file alone).
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import os
import queue
import re
import shutil
import threading
import time

from server.conductor import redact as redact_mod
from server.conductor import schema, signals

DEFAULT_ROOT = "~/.qwen38-spark-engine/memory"
DEFAULTS = dict(schema.SETTINGS["memory"]["transcripts"])
QUEUE_MAX = 512
PER_PAGE = 200                      # blocks a page in the dashboard's viewer (VIS-48)
# what is redacted before a clip: the head of a long text, enough for a key block that starts
# before the clip and ends after it
REDACT_OVER = 8192
WORKDIR = re.compile(r"Working directory:\s*(\S+)")
_NAME = re.compile(r"[^A-Za-z0-9._-]")


def memory_root(conductor_root: str) -> str:
    """The production conductor's memory is its sibling under the engine's home; any other
    conductor directory (a test, a second server) keeps its memory inside itself."""
    from server.conductor.store import DEFAULT_ROOT as COND_DEFAULT
    cond = os.path.realpath(os.path.expanduser(conductor_root))
    if cond == os.path.realpath(os.path.expanduser(COND_DEFAULT)):
        return os.path.realpath(os.path.expanduser(DEFAULT_ROOT))
    return os.path.join(cond, "memory")


def in_repository(path: str) -> str | None:
    """The git work tree `path` is inside, if any."""
    p = os.path.realpath(path)
    while True:
        if os.path.exists(os.path.join(p, ".git")):
            return p
        up = os.path.dirname(p)
        if up == p:
            return None
        p = up


def settings_of(settings: dict | None) -> dict:
    """`settings.memory.transcripts` over the defaults (a settings version stored before SRV-137
    has none)."""
    got = ((settings or {}).get("memory") or {}).get("transcripts") or {}
    return {**DEFAULTS, **{k: v for k, v in got.items() if k in DEFAULTS}}


def dir_name(key: str) -> str:
    """A session key as a directory name: `c:ses_ab12` -> `c-ses_ab12`; a key that loses
    characters on the way gets a short hash, so two keys never share a directory."""
    head, _, rest = key.partition(":")
    name = f"{head}-{_NAME.sub('_', rest)}" if rest else _NAME.sub("_", key)
    if name != f"{head}-{rest}" or len(name) > 96:
        name = f"{name[:80]}-{hashlib.sha256(key.encode()).hexdigest()[:10]}"
    return name


def _text(content) -> tuple[str, int]:
    """A message's text and its image count."""
    return signals._text(content), signals._images(content)


# a line of a message that reads like one of the block's own headings is escaped (markdown
# shows `\###` as `###`), so a reader that cuts blocks and sections at those headings never cuts a
# message
OWN_HEADING = re.compile(r"^(?=## turn \d+ · |### (?:user|assistant|summary|runtime|states|"
                         r"tool result \(|tool call \(|reasoning \())", re.M)


def _safe(text: str) -> str:
    return OWN_HEADING.sub(lambda m: "\\", text)


def _clip(text: str, limit: int) -> tuple[str, dict[str, int], int, bool]:
    """Redact, then clip: (the text, redactions, the original size, clipped). Only the head a
    clip keeps is redacted, plus REDACT_OVER characters for a key block across the clip."""
    n = len(text)
    clipped = n > limit
    head = text[:limit + REDACT_OVER] if clipped else text
    head, counts = redact_mod.redact(head)
    if clipped:
        head = head[:limit]
    return _safe(head), counts, n, clipped


def split_reasoning(text: str, in_think: bool) -> tuple[str, str]:
    """(the answer, the reasoning) of a decoded output: with the prompt ending inside `<think>`,
    the reasoning runs to `</think>` (all of it when the turn ended inside); a think block the
    model opened itself comes off the answer the same way."""
    if not in_think:
        m = re.match(r"\s*<think>([\s\S]*?)</think>\n*", text)
        return (text[m.end():], m.group(1)) if m else (text, "")
    i = text.find("</think>")
    if i < 0:
        return "", text
    return text[i + len("</think>"):].lstrip("\n"), text[:i]


def _fmt_n(n) -> str:
    return f"{int(n):,}" if n is not None else "?"


def new_messages(msgs: list) -> list[dict]:
    """What a request adds to its session: the messages after its last assistant message (the
    tool results of the turn before, a new user message), or every message but the system ones
    on the first turn."""
    msgs = [m for m in msgs or [] if isinstance(m, dict)]
    last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
    if last is None:
        return [m for m in msgs if m.get("role") not in ("system", "developer")]
    return msgs[last + 1:]


def tool_names(msgs: list) -> dict[str, str]:
    """tool_call_id -> the function's name, from the assistant messages' calls."""
    out = {}
    for m in msgs or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                if isinstance(c, dict) and c.get("id"):
                    out[c["id"]] = ((c.get("function") or {}).get("name")) or "tool"
    return out


def block(job: dict, turn: int, cfg: dict) -> tuple[str, dict]:
    """One request's block, and what it counted: `{redactions: {kind: n}, clipped, checkers:
    [...], hidden, acts, compaction}`. `job` is the plain data `Transcripts.job()` took from the
    plan and the record; nothing here touches a request object."""
    clip, loop = int(cfg["clip_chars"]), int(cfg["loop_chars"])
    counts: dict[str, int] = {}
    stats = {"clipped": 0, "checkers": [], "hidden": 0, "acts": 0, "compaction": False}
    when = _dt.datetime.fromtimestamp(job["ts"]).strftime("%Y-%m-%d %H:%M:%S")
    out_text, out_calls = job.get("content") or "", job.get("calls") or []

    def put(text: str) -> str:
        t, c, _, _ = _clip(text, 10 ** 9)
        redact_mod.merge(counts, c)
        return t

    parts: list[str] = []
    if job.get("compaction"):
        stats["compaction"] = True
        parts.append(f"## turn {turn} · {when} · request {job['request_id']} · compaction")
        parts.append(f"compaction: {_fmt_n(job.get('prompt_tokens'))} tokens in, "
                     f"{_fmt_n(job.get('completion_tokens'))} tokens out")
        parts.append("### summary\n" + (put(out_text.strip()) or "(empty)"))
    else:
        head = [f"## turn {turn}", when, f"request {job['request_id']}"]
        if job.get("class_id"):
            head.append(f"class {job['class_id']}")
        if job.get("effort"):
            head.append(f"effort {job['effort']}")
        if job.get("reasoning_tokens"):
            head.append(f"reasoning {_fmt_n(job['reasoning_tokens'])} tokens")
        parts.append(" · ".join(head))
        names = tool_names(job.get("messages") or [])
        for m in new_messages(job.get("messages") or []):
            role = m.get("role")
            text, images = _text(m.get("content"))
            if role == "tool" or (role == "user" and signals._is_tool_response(text)):
                name = names.get(m.get("tool_call_id") or "", m.get("name") or "tool")
                body, c, n, cut = _clip(text, clip)
                redact_mod.merge(counts, c)
                stats["clipped"] += int(cut)
                size = f"{_fmt_n(n)} chars" + (f", clipped to {_fmt_n(clip)}" if cut else "")
                parts.append(f"### tool result ({name}, {size})\n{body}")
                for r in signals.checker_results(text):
                    stats["checkers"].append(r)
            elif role == "user":
                img = "\n".join(["[image]"] * images)
                parts.append("### user\n" + "\n".join(x for x in (put(text), img) if x))
            elif role in ("system", "developer"):
                continue
            else:
                parts.append(f"### {role or 'message'}\n{put(text)}")
        if out_text.strip():
            parts.append("### assistant\n" + put(out_text.strip()))
        for c in out_calls:
            fn = (c or {}).get("function") or {}
            args = fn.get("arguments")
            args = args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False)
            body, cnt, n, cut = _clip(args, clip)
            redact_mod.merge(counts, cnt)
            stats["clipped"] += int(cut)
            size = f" ({_fmt_n(n)} chars, clipped to {_fmt_n(clip)})" if cut else ""
            parts.append(f"### tool call ({fn.get('name') or 'tool'}){size}\n{body}")
        if not out_text.strip() and not out_calls:
            parts.append("### assistant\n(no text, no tool call)")
        if job.get("loop_cut") and job.get("reasoning"):
            r = job["reasoning"].strip()
            excerpt = r if len(r) <= 2 * loop else f"{r[:loop]}\n[…]\n{r[-loop:]}"
            parts.append(f"### reasoning (cut by {job['loop_cut']}: the first and last {loop} "
                         f"characters)\n{put(excerpt)}")
    if job.get("states"):
        # the state lines the client was shown (`ui.states`), a block kind of their own
        parts.append("### states\n" + "\n".join(f"› {t}" for t in job["states"]))
    run: list[str] = []
    for n in job.get("notes") or []:
        stats["hidden"] += 1
        run.append(f"hidden: {n['what']}, {_fmt_n(n.get('tokens'))} tokens"
                   + (f" at token {_fmt_n(n['pos'])}" if n.get("pos") is not None else "")
                   + (f" · journal {n['journal']}" if n.get("journal") else ""))
    rep = job.get("replayed") or {}
    if rep.get("count"):
        run.append(f"replayed: {rep['count']} note{'s' if rep['count'] != 1 else ''}, "
                   f"{_fmt_n(rep.get('tokens'))} tokens")
    acts = job.get("acts") or []
    stats["acts"] = len(acts)
    if acts:
        run.append(f"acts: {len(acts)} ({', '.join(acts)})")
    for r in stats["checkers"]:
        reasons = "; ".join(r["reasons"])
        run.append(f"checker: {r['checker']} {r['target']}".rstrip() + f" {r['verdict']}"
                   + (f" <- {reasons}" if reasons else ""))
    if job.get("snapshot_file"):
        # J1: the turn's snapshot beside the block, and its outcome when one was classified
        o = (job.get("snapshot") or {}).get("outcome") or {}
        run.append(f"snapshot: {job['snapshot_file']}"
                   + (f" · outcome {o['label']}" if o.get("label") and o.get("mode") != "off"
                      else ""))
    run.append(f"finish: {job.get('finish') or '?'} · prompt {_fmt_n(job.get('prompt_tokens'))} "
               f"tokens · completion {_fmt_n(job.get('completion_tokens'))} tokens")
    total = sum(counts.values())
    if total:
        run.append(f"redactions: {total} ("
                   + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ")")
    parts.append("### runtime\n" + "\n".join(run))
    stats["redactions"] = counts
    return "\n\n".join(parts) + "\n\n", stats


class Transcripts:
    """The writer. `settings()`: the conductor's `settings` document; `journal`: the conductor's
    journal (its `note()` takes the counts); `tok`: the tokenizer, for the output's ids."""

    def __init__(self, root: str, *, settings=lambda: {}, journal=None, tok=lambda: None,
                 clock=time.time):
        self.root = os.path.realpath(os.path.expanduser(root))
        self.sessions_dir = os.path.join(self.root, "sessions")
        self.settings, self.journal, self.tok, self.clock = settings, journal, tok, clock
        self.repo = in_repository(self.root)
        self._q: queue.Queue = queue.Queue(QUEUE_MAX)
        self._lock = threading.Lock()
        # meta.json's read-modify-write: the writer's block and the dashboard's keep toggle
        self._meta_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._dirs: dict[str, str] = {}
        self._next_prune = 0.0
        # jobs that run with the prune (at the start, then 04:00): the memory notes' nightly job
        self.nightly: list = []
        self.stats = {"blocks": 0, "counters": 0, "dropped": 0, "failed": 0, "redactions": 0,
                      "pruned": 0, "bytes": 0}
        if self.repo:
            print(f"[memory] REFUSING transcripts: {self.root} is inside the git work tree "
                  f"{self.repo}; session text never goes into a repository", flush=True)

    # ------------------------------------------------------------------ the request's side
    def cfg(self) -> dict:
        try:
            return settings_of(self.settings())
        except Exception:                                          # noqa: BLE001
            return dict(DEFAULTS)

    @staticmethod
    def scope_of(cfg: dict, tools: bool, compaction: bool) -> str | None:
        """`full` (a block), `counters` (the session's counts only) or None (nothing)."""
        if cfg["scope"] == "none":
            return None
        if cfg["scope"] == "all" or tools or compaction:
            return "full"
        return "counters"

    def submit(self, plan, rec) -> bool:
        """Queue one finished request (references only; the writer thread does the work).
        False when it was not queued: no scope, a refused root, or a full queue (counted)."""
        if self.repo or getattr(plan, "transcript", None) is None or plan.session is None:
            return False
        self.start()
        try:
            self._q.put_nowait((plan, rec))
            return True
        except queue.Full:
            self.stats["dropped"] += 1
            return False

    def start(self) -> None:
        if self.repo:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="memory-transcripts",
                                                daemon=True)
                self._thread.start()

    # ------------------------------------------------------------------ the writer thread
    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                item = None
            if item is not None:
                try:
                    self.write(self.job(*item))
                except Exception as exc:                           # noqa: BLE001
                    self._failed(None, None, exc)
                finally:
                    self._q.task_done()
            if self.clock() >= self._next_prune:
                from server.ledger import next_local
                self._next_prune = next_local(4, self.clock())
                try:
                    self.prune()
                except Exception as exc:                           # noqa: BLE001
                    print(f"[memory] prune failed: {type(exc).__name__}: {exc}", flush=True)
                for job in list(self.nightly):
                    try:
                        job()
                    except Exception as exc:                       # noqa: BLE001
                        print(f"[memory] nightly job failed: {type(exc).__name__}: {exc}",
                              flush=True)

    def flush(self, timeout: float = 5.0) -> None:
        end = time.time() + timeout
        while self._q.unfinished_tasks and time.time() < end:
            time.sleep(0.01)

    def job(self, plan, rec) -> dict:
        """The plain data of one request: what the block needs, read off the plan, the record and
        the monitor's events, with the output's ids decoded here (not on the request's thread)."""
        row = plan.row or {}
        body = plan.body or {}
        sup = getattr(rec, "sup", None)
        out = getattr(rec, "out", None)
        content, reasoning = "", ""
        if plan.transcript == "full" and out is not None:
            ids, in_think = out
            tok = self.tok()
            text = tok.decode(ids, skip_special_tokens=False) if tok is not None else ""
            content, reasoning = split_reasoning(text, bool(in_think))
            for eos in ("<|im_end|>", "<|endoftext|>"):
                content = content.replace(eos, "")
            calls = getattr(rec, "out_calls", None) or []
            if calls and "<tool_call>" in content:
                content = content[:content.index("<tool_call>")]
        elif plan.transcript == "full" and getattr(rec, "out_text", None) is not None:
            # a request the proxy relayed to another backend (proxy P4): the text it streamed
            content, reasoning = rec.out_text
            calls = getattr(rec, "out_calls", None) or []
        else:
            calls = []
        notes = []
        for n in plan.injected or []:
            parts = ", ".join(f"{p['kind']} {p['name']}" for p in n.get("parts") or [])
            notes.append({"what": f"router note ({parts})", "tokens": n.get("tokens"),
                          "pos": n.get("pos"), "journal": row.get("id")})
        acts, loop_cut = [], None
        for e in (getattr(sup, "events", None) or getattr(rec, "harness_events", None) or []):
            if e.get("mode") != "act" or e.get("decision") in (None, "PASS", "OUTCOME", "LABEL",
                                                               "COUNTS"):
                continue
            acts.append(str(e.get("task") or e.get("decision")))
            if e.get("task") in ("D1", "D1b") and loop_cut is None:
                loop_cut = e["task"]
            if e.get("text") or e.get("hidden_tokens"):
                notes.append({"what": f"{e.get('task')} {e.get('decision')}",
                              "tokens": e.get("hidden_tokens"), "pos": e.get("pos"),
                              "journal": e.get("journal_id")})
        system = next((signals._text(m.get("content")) for m in body.get("messages") or []
                       if isinstance(m, dict) and m.get("role") == "system"), "")
        wd = WORKDIR.search(system or "")
        return {"key": plan.session.key, "scope": plan.transcript, "ts": getattr(rec, "ts", None)
                or self.clock(), "request_id": getattr(rec, "request_id", "-"),
                "client": getattr(rec, "client_kind", None), "profile": plan.res.profile,
                "project": wd.group(1).rstrip("/") if wd else None,
                "compaction": bool(plan.compaction), "class_id": row.get("class_id"),
                "effort": row.get("effort"), "messages": body.get("messages") or [],
                "content": content, "calls": calls, "reasoning": reasoning,
                "loop_cut": loop_cut, "notes": notes, "replayed": row.get("replayed"),
                "acts": acts, "finish": getattr(rec, "finish_reason", None),
                "states": [e.get("shown") or e.get("summary") or e["text"]
                           for e in (getattr(getattr(rec, "states", None), "log", None) or [])
                           if not e.get("merged")],
                "prompt_tokens": getattr(rec, "prompt_tokens", None),
                "completion_tokens": getattr(rec, "completion_tokens", None),
                "reasoning_tokens": getattr(rec, "reasoning_tokens", None),
                **self._snapshot_job(plan, row, rec)}

    @staticmethod
    def _snapshot_job(plan, row, rec) -> dict:
        """J1: the turn's snapshot (the router built it at the request's start) with what the
        journal knows once the turn ended: class, effort, reasoning tokens, finish, acts."""
        snap = getattr(plan, "snapshot", None)
        if snap is None:
            return {}
        sup = getattr(rec, "sup", None)
        acts = [str(e.get("task")) for e in (getattr(sup, "events", None) or [])
                if e.get("mode") == "act" and e.get("decision") not in (None, "PASS", "OUTCOME",
                                                                       "LABEL", "COUNTS")]
        snap["turn"] = {"class": row.get("class_id"), "effort": row.get("effort"),
                        "reasoning_tokens": getattr(rec, "reasoning_tokens", None),
                        "finish": getattr(rec, "finish_reason", None),
                        "acts": acts + [f"outcome {n['parts'][0]['name']}"
                                        for n in getattr(plan, "injected", None) or []
                                        if (n.get("parts") or [{}])[0].get("kind") == "outcome"]}
        return {"snapshot": snap}

    def session_dir(self, key: str, ts: float) -> str:
        """The session's directory: the month of its first block, found again later."""
        got = self._dirs.get(key)
        if got and os.path.isdir(got):
            return got
        name = dir_name(key)
        try:
            for month in sorted(os.listdir(self.sessions_dir)):
                cand = os.path.join(self.sessions_dir, month, name)
                if os.path.isdir(cand):
                    self._dirs[key] = cand
                    return cand
        except OSError:
            pass
        month = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m")
        d = self._dirs[key] = os.path.join(self.sessions_dir, month, name)
        return d

    def write(self, job: dict) -> str | None:
        """Append the block (or count the request) and update meta.json. Returns the session's
        directory; None when the write failed (journaled)."""
        cfg = self.cfg()
        d = self.session_dir(job["key"], job["ts"])
        with self._meta_lock:
            return self._write(job, cfg, d)

    def _write(self, job: dict, cfg: dict, d: str) -> str | None:
        try:
            for p in (self.root, self.sessions_dir, os.path.dirname(d), d):
                os.makedirs(p, mode=0o700, exist_ok=True)
            meta = read_meta(d) or {"session": job["key"], "dir": os.path.basename(d),
                                    "client": job.get("client"), "profile": job.get("profile"),
                                    "project": None, "first_ms": int(job["ts"] * 1000),
                                    "keep": False, "bytes": 0,
                                    "counts": {k: 0 for k in COUNTS}, "checkers": {}}
            c = meta.setdefault("counts", {})
            for k in COUNTS:
                c.setdefault(k, 0)
            c["requests"] += 1
            meta["last_ms"] = int(job["ts"] * 1000)
            meta["profile"] = job.get("profile") or meta.get("profile")
            meta["project"] = job.get("project") or meta.get("project")
            if job["scope"] != "full":
                c["untranscribed"] += 1
                write_meta(d, meta)
                self.stats["counters"] += 1
                return d
            if job.get("snapshot") is not None:
                job["snapshot_file"] = f"snapshots/turn-{c['blocks'] + 1}.json"
            text, st = block(job, c["blocks"] + 1, cfg)
            data = text.encode()
            path = os.path.join(d, "transcript.md")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                start = os.fstat(fd).st_size
                os.write(fd, data)
            finally:
                os.close(fd)
            # where each block starts: a reader slices blocks by these, never by a heading a
            # message may also contain, and reads no further than `bytes`
            meta.setdefault("offsets", []).append(start)
            c["blocks"] += 1
            c["compactions"] += int(st["compaction"])
            c["hidden_notes"] += st["hidden"]
            c["acts"] += st["acts"]
            c["clipped"] += st["clipped"]
            c["reasoning_tokens"] += int(job.get("reasoning_tokens") or 0)
            n_red = sum(st["redactions"].values())
            c["redactions"] += n_red
            for r in st["checkers"]:
                ck = meta.setdefault("checkers", {})
                ck[r["verdict"]] = ck.get(r["verdict"], 0) + 1
                ck["last"] = f"{r['checker']} {r['target']}".strip() + f" {r['verdict']}"
            meta["bytes"] = os.path.getsize(path)
            if job.get("snapshot") is not None:
                self._snapshot(d, c["blocks"], job["snapshot"], cfg)
                c["snapshots"] = c.get("snapshots", 0) + 1
            write_meta(d, meta)
            self.stats["blocks"] += 1
            self.stats["bytes"] += len(data)
            self.stats["redactions"] += n_red
            if self.journal is not None:
                self.journal.note("memory", "transcript", decision="block",
                                  session_key=job["key"], request_id=job["request_id"],
                                  detail={"turn": c["blocks"], "bytes": len(data),
                                          "redactions": n_red,
                                          "redacted": dict(st["redactions"]),
                                          "clipped": st["clipped"],
                                          "compaction": st["compaction"]})
            return d
        except OSError as exc:
            self._failed(job["key"], job["request_id"], exc)
            return None

    def _snapshot(self, d: str, turn: int, snap: dict, cfg: dict) -> None:
        """J1: `snapshots/turn-<n>.json` (n is the block's turn; at most the router document's
        `outcomes.limits.snapshot_bytes`) and the turn's images under `shots/`, 0600; a session
        keeps its newest `shots_kept` images, the older ones keep their hash in their snapshot."""
        from server.conductor import outcomes
        lim = {**schema.OUTCOMES["limits"], **(snap.get("_limits") or {})}
        sdir, shots = os.path.join(d, "snapshots"), os.path.join(d, "shots")
        for p in (sdir, shots):
            os.makedirs(p, mode=0o700, exist_ok=True)
        for i, (sha, b64) in enumerate(snap.get("_media") or ()):
            try:
                data = base64.b64decode(b64)
            except (ValueError, TypeError):
                continue
            ext = "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "jpg" if data[:2] == b"\xff\xd8" \
                else "img"
            name = f"turn-{turn}-{i}-{sha}.{ext}"
            _write_private(os.path.join(shots, name), data)
            for s in snap["shots"]:
                if s["sha"] == sha:
                    s["file"] = f"shots/{name}"
        kept = sorted(os.listdir(shots), key=_shot_order)
        for old in kept[:max(0, len(kept) - int(lim["shots_kept"]))]:
            os.remove(os.path.join(shots, old))
        # the secret-pattern refusal, as for the block: every string of the snapshot, before the
        # JSON escapes a key block's lines
        counts: dict[str, int] = {}
        clean = outcomes.Snapshot(_redacted({k: v for k, v in snap.items()
                                             if not k.startswith("_")}, counts))
        _write_private(os.path.join(sdir, f"turn-{turn}.json"),
                       outcomes.to_json(clean, int(lim["snapshot_bytes"])))
        self.stats["redactions"] += sum(counts.values())
        snap.pop("_media", None)

    def _failed(self, key, rid, exc: BaseException) -> None:
        self.stats["failed"] += 1
        msg = f"{type(exc).__name__}: {exc}"[:300]
        print(f"[memory] transcript write failed ({msg}); the request was not affected",
              flush=True)
        if self.journal is not None:
            try:
                self.journal.note("memory", "transcript", decision="write_failed",
                                  session_key=key, request_id=rid, detail={"error": msg})
            except Exception:                                      # noqa: BLE001
                pass

    # ------------------------------------------------------------------ retention
    def prune(self, now: float | None = None) -> dict:
        """Sessions idle for longer than `retention_days` go, unless marked keep. Journaled."""
        now = self.clock() if now is None else now
        days = int(self.cfg()["retention_days"])
        cut_ms = int((now - days * 86400) * 1000)
        gone, kept = 0, 0
        try:
            months = sorted(os.listdir(self.sessions_dir))
        except OSError:
            return {"deleted": 0, "kept": 0}
        for month in months:
            mdir = os.path.join(self.sessions_dir, month)
            if not os.path.isdir(mdir):
                continue
            for name in os.listdir(mdir):
                d = os.path.join(mdir, name)
                if not os.path.isdir(d):
                    continue
                meta = read_meta(d) or {}
                last = meta.get("last_ms") or int(os.path.getmtime(d) * 1000)
                if last >= cut_ms:
                    continue
                if meta.get("keep"):
                    kept += 1
                    continue
                shutil.rmtree(d, ignore_errors=True)
                self._dirs = {k: v for k, v in self._dirs.items() if v != d}
                gone += 1
            try:
                os.rmdir(mdir)                       # only when the month is empty
            except OSError:
                pass
        self.stats["pruned"] += gone
        if gone and self.journal is not None:
            self.journal.note("memory", "prune", decision="deleted",
                              detail={"deleted": gone, "kept": kept, "retention_days": days})
        return {"deleted": gone, "kept": kept}

    def info(self) -> dict:
        return {"root": self.root, "refused": self.repo, "queue": self._q.qsize(),
                **self.stats, "settings": self.cfg()}

    # ------------------------------------------------------------------ the dashboard's reads (VIS-48)
    def find(self, key: str) -> str | None:
        """The directory of the session `key`, or None."""
        name = dir_name(key)
        try:
            months = sorted(os.listdir(self.sessions_dir), reverse=True)
        except OSError:
            return None
        for month in months:
            d = os.path.join(self.sessions_dir, month, name)
            if os.path.isdir(d):
                return d
        return None

    def sessions(self) -> list[dict]:
        """Every session on disk, newest first: its meta.json, or what the directory says when
        it has none (counts unknown)."""
        out = []
        try:
            months = sorted(os.listdir(self.sessions_dir), reverse=True)
        except OSError:
            return out
        for month in months:
            mdir = os.path.join(self.sessions_dir, month)
            if not os.path.isdir(mdir):
                continue
            for name in os.listdir(mdir):
                d = os.path.join(mdir, name)
                if not os.path.isdir(d):
                    continue
                out.append(summary(d, month))
        out.sort(key=lambda s: s.get("last_ms") or 0, reverse=True)
        return out

    def read(self, key: str, page: int | None = None, per_page: int = PER_PAGE) -> dict | None:
        """One session's blocks, `per_page` a page (default: the last page, the newest), read no
        further than the last complete block. None when the session is not on disk."""
        d = self.find(key)
        if d is None:
            return None
        s = summary(d, os.path.basename(os.path.dirname(d)))
        blocks = read_blocks(d, s.get("bytes"), s.get("offsets"))
        pages = max(1, -(-len(blocks) // per_page))
        page = pages if page is None else max(1, min(int(page), pages))
        part = blocks[(page - 1) * per_page: page * per_page]
        s.pop("offsets", None)
        return {"session": s, "blocks": part, "page": page, "pages": pages,
                "per_page": per_page, "total": len(blocks)}

    def export(self, key: str) -> tuple[dict, str] | None:
        """(the session's summary, its transcript up to the last complete block)."""
        d = self.find(key)
        if d is None:
            return None
        s = summary(d, os.path.basename(os.path.dirname(d)))
        s.pop("offsets", None)
        return s, read_text(d, s.get("bytes"))

    def set_keep(self, key: str, keep: bool) -> dict | None:
        """meta.json's `keep`, which the prune honours. None when the session is not on disk."""
        d = self.find(key)
        if d is None:
            return None
        with self._meta_lock:
            meta = read_meta(d) or {"session": key, "dir": os.path.basename(d),
                                    "counts": None}
            meta["keep"] = bool(keep)
            write_meta(d, meta)
        if self.journal is not None:
            self.journal.note("memory", "keep", decision="on" if keep else "off",
                              session_key=key)
        return {"session": key, "keep": bool(keep)}


COUNTS = ("requests", "blocks", "untranscribed", "compactions", "hidden_notes", "acts",
          "redactions", "clipped", "reasoning_tokens")
HEADER = re.compile(r"^## turn (\d+) · (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) · request (\S+)"
                    r"(?: · (compaction))?(?: · class (\S+))?(?: · effort (\S+))?"
                    r"(?: · reasoning ([\d,]+) tokens)?$")


def summary(d: str, month: str) -> dict:
    """A session's row for the list: meta.json, or the directory's name and times without it."""
    meta = read_meta(d)
    if meta is None:
        path = os.path.join(d, "transcript.md")
        st = os.stat(path) if os.path.exists(path) else os.stat(d)
        return {"session": None, "dir": os.path.basename(d), "month": month, "client": None,
                "profile": None, "project": None, "first_ms": None,
                "last_ms": int(st.st_mtime * 1000), "keep": False, "counts": None,
                "checkers": {}, "bytes": st.st_size if os.path.exists(path) else 0,
                "has_meta": False}
    return dict(meta, month=month, has_meta=True)


def read_text(d: str, upto: int | None) -> str:
    """transcript.md up to `upto` bytes (the end of the last complete block), as text."""
    try:
        with open(os.path.join(d, "transcript.md"), "rb") as f:
            data = f.read(upto) if upto is not None else f.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def read_blocks(d: str, upto: int | None, offsets: list | None) -> list[dict]:
    """The blocks of a transcript, each `{turn, time, request_id, class_id, effort,
    compaction, reasoning_tokens, text}`. Sliced by meta.json's offsets; a transcript without
    them is split at its block headers."""
    try:
        with open(os.path.join(d, "transcript.md"), "rb") as f:
            data = f.read(upto) if upto is not None else f.read()
    except OSError:
        return []
    if offsets:
        cuts = [o for o in offsets if o < len(data)] + [len(data)]
        chunks = [data[a:b].decode("utf-8", errors="replace") for a, b in zip(cuts, cuts[1:])]
    else:
        whole = data.decode("utf-8", errors="replace")
        cuts = [m.start() for m in re.finditer(r"(?:^|(?<=\n\n))## turn \d+ · ", whole)]
        cuts = (cuts or [0]) + [len(whole)]
        chunks = [whole[a:b] for a, b in zip(cuts, cuts[1:])]
    out = []
    for chunk in chunks:
        text = chunk.rstrip("\n")
        if not text:
            continue
        m = HEADER.match(text.split("\n", 1)[0])
        out.append({"turn": int(m.group(1)) if m else None, "time": m.group(2) if m else None,
                    "request_id": m.group(3) if m else None,
                    "compaction": bool(m and m.group(4)),
                    "class_id": m.group(5) if m else None, "effort": m.group(6) if m else None,
                    "reasoning_tokens": int(m.group(7).replace(",", "")) if m and m.group(7)
                    else None, "text": text})
    return out


def read_meta(d: str) -> dict | None:
    try:
        with open(os.path.join(d, "meta.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _redacted(obj, counts: dict):
    """Every string in `obj` through the secret-pattern refusal; the counts merged into `counts`."""
    if isinstance(obj, str):
        text, c = redact_mod.redact(obj)
        redact_mod.merge(counts, c)
        return text
    if isinstance(obj, dict):
        return {k: _redacted(v, counts) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redacted(v, counts) for v in obj]
    return obj


def _shot_order(name: str) -> tuple:
    """`turn-<n>-<i>-<sha>.<ext>` by its turn and index, oldest first."""
    m = re.match(r"turn-(\d+)-(\d+)-", name)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, 0)


def _write_private(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def write_meta(d: str, meta: dict) -> None:
    path = os.path.join(d, "meta.json")
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
