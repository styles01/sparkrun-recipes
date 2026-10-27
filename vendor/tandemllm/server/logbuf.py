"""The engine's own log, live: a tee on stdout/stderr into a ring buffer, and the no-content rule.

the operator's Dev tab shows what `logs/engine-*.log` shows, as it happens. Nothing
about the log file changes: `Tee.write` hands every string to the real stream first, byte for
byte, and only then reads it. Complete lines become entries `{seq, ts, level, source, msg,
request_id}` in a ring of 10,000; a traceback is one entry, however many lines it has. Subscribers
(the SSE route) get their own bounded queue: a reader that stops reading loses its OLDEST lines,
and is told how many with a `gap`, instead of growing the server's memory.

LEVELS, from what the lines already say: error = a traceback, a `[req]` line with `!!` or
`finish=error`, a failed ledger write; warning = `finish=timeout|abandoned`, `LOSSY ACCEPT RULE`,
`[cache] put declined`; debug = the `--verbose` HTTP access lines; info = everything else. The
source is the bracket tag (`req`, `server`, `cache`, `drafter`, `think`, `ledger`, `body`), or
`http` / `traceback`.

NO CONTENT. No line this server prints carries a prompt, a message, a tool argument or an answer --
except, potentially, an exception's MESSAGE: a template or a parser that raises with the text it
choked on. So an exception message is capped at 200 characters and, unless `--log-content` is set,
withheld entirely when any 12-character stretch of it occurs in the request being served
(`safe_message`, `print_exc`). The frames of a traceback are this program's source lines and stay.
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import re
import sys
import threading
import traceback

RING = 10_000
SUB_QUEUE = 2_000
MAX_SUBSCRIBERS = 4
MSG_CAP = 200
WINDOW = 12
LEVELS = {"debug": 0, "info": 1, "warning": 2, "error": 3}

_TAG = re.compile(r"^\[([a-z0-9_-]+)\]")
_RID = re.compile(r"\b((?:chatcmpl|cmpl)-[0-9a-f]{24})\b")
_HTTP = re.compile(r'^\S+ - "?(GET|POST|DELETE|PUT|HEAD|OPTIONS) ')


def _now_iso() -> str:
    t = _dt.datetime.now(_dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def classify(msg: str, stream: str = "stdout") -> tuple[str, str]:
    """`(level, source)` of one complete entry."""
    if msg.startswith("Traceback (most recent call last)") or msg.startswith("During handling") \
            or msg.startswith("The above exception"):
        return "error", "traceback"
    if _HTTP.match(msg):
        return "debug", "http"
    m = _TAG.match(msg)
    source = m.group(1) if m else ("stderr" if stream == "stderr" else "stdout")
    if " !! " in msg or "finish=error" in msg or (source == "ledger" and "failed" in msg):
        return "error", source
    if source == "memory" and " refused" in msg:
        return "error", source                  # A refusal is loud in the Dev tab
    if ("finish=timeout" in msg or "finish=abandoned" in msg or "LOSSY ACCEPT RULE" in msg
            or msg.startswith("[cache] put declined") or msg.startswith("[argloop]")
            or msg.startswith("[shadow] released")):
        return "warning", source
    return "info", source


class Subscriber:
    def __init__(self, level: int, grep: str | None, gone=None):
        self.level, self.grep = level, grep
        self.gone = gone              # () -> True once the reader has closed its end
        self.q: collections.deque = collections.deque()
        self.dropped = 0
        self.cond = threading.Condition()
        self.closed = False

    def wants(self, e: dict) -> bool:
        return LEVELS[e["level"]] >= self.level and (not self.grep or self.grep in e["msg"])

    def push(self, e: dict) -> None:
        with self.cond:
            if len(self.q) >= SUB_QUEUE:
                self.q.popleft()
                self.dropped += 1
            self.q.append(e)
            self.cond.notify()

    def take(self, timeout: float) -> tuple[list, int]:
        """What arrived (waiting up to `timeout`), and how many lines were dropped since."""
        with self.cond:
            if not self.q and not self.closed:
                self.cond.wait(timeout)
            out = list(self.q)
            self.q.clear()
            dropped, self.dropped = self.dropped, 0
        return out, dropped


class LogBuffer:
    """The ring and its subscribers. `append` is called by the tees, once per complete entry."""

    def __init__(self, size: int = RING):
        self.ring: collections.deque = collections.deque(maxlen=size)
        self.seq = 0
        self.lock = threading.Lock()
        self.subs: list[Subscriber] = []

    def append(self, msg: str, stream: str = "stdout") -> dict:
        level, source = classify(msg, stream)
        m = _RID.search(msg)
        with self.lock:
            self.seq += 1
            e = {"seq": self.seq, "ts": _now_iso(), "level": level, "source": source, "msg": msg,
                 "request_id": m.group(1) if m else None}
            self.ring.append(e)
            subs = list(self.subs)
        for s in subs:
            if s.wants(e):
                s.push(e)
        return e

    def lines(self, *, level: str = "info", after: int | None = None, grep: str | None = None,
              limit: int = 500) -> list[dict]:
        lv = LEVELS[level]
        with self.lock:
            snap = list(self.ring)
        out = [e for e in snap if (after is None or e["seq"] > after)
               and LEVELS[e["level"]] >= lv and (not grep or grep in e["msg"])]
        return out[-limit:] if limit else []

    def after_time(self, iso: str) -> int | None:
        """The last seq before an RFC 3339 time, for `since=<time>`."""
        with self.lock:
            snap = list(self.ring)
        prev = None
        for e in snap:
            if e["ts"] > iso:
                return prev if prev is not None else e["seq"] - 1
            prev = e["seq"]
        return prev

    def subscribe(self, level: str, grep: str | None, gone=None) -> Subscriber | None:
        """A place under the cap, or None. At the cap, a reader that has closed its end gives its
        place up now: its handler notices within a second, but a reopened tab asks
        sooner than that."""
        with self.lock:
            if len(self.subs) >= MAX_SUBSCRIBERS:
                for old in [x for x in self.subs if x.gone is not None and x.gone()]:
                    self.subs.remove(old)
                    with old.cond:
                        old.closed = True
                        old.cond.notify_all()
            if len(self.subs) >= MAX_SUBSCRIBERS:
                return None
            s = Subscriber(LEVELS[level], grep, gone)
            self.subs.append(s)
            return s

    def unsubscribe(self, s: Subscriber) -> None:
        with self.lock:
            if s in self.subs:
                self.subs.remove(s)
        with s.cond:
            s.closed = True
            s.cond.notify_all()

    @property
    def subscribers(self) -> int:
        with self.lock:
            return len(self.subs)


_WRITE = threading.Lock()      # one for both tees: stdout and stderr are one file (start.sh 2>&1)


class Tee:
    """A text stream that writes through unchanged and feeds complete lines to a `LogBuffer`.

    The write-through is by whole lines: text after the last newline waits for the rest of its
    line (or a `flush()`), and both tees write under one lock. `print()` under `python -u` is two
    writes -- the text, then "\\n" -- and another thread's line between them used to land inside
    the first (an access line inside a `[req]` line, 2026-09-24). The bytes are the same; only a
    line never shares its place in the file with another."""

    def __init__(self, stream, buf: LogBuffer, name: str):
        self._stream, self._buf, self._name = stream, buf, name
        self._partial = ""
        self._pending = ""
        self._tb: list[str] | None = None
        self._lock = threading.Lock()

    def write(self, s):
        with _WRITE:
            text = self._pending + s
            cut = text.rfind("\n") + 1
            if cut:
                self._stream.write(text[:cut])
            self._pending = text[cut:]
        try:
            self._feed(s)
        except Exception:                                          # noqa: BLE001
            pass                                   # the log file is what matters; never break it
        return len(s)

    def _feed(self, s: str) -> None:
        done = []
        with self._lock:
            text = self._partial + s
            *lines, self._partial = text.split("\n")
            for line in lines:
                if self._tb is not None:
                    self._tb.append(line)
                    # a traceback ends with its unindented "ExcType: message" line
                    if line and not line[0].isspace():
                        done.append("\n".join(self._tb))
                        self._tb = None
                elif line.startswith("Traceback (most recent call last)"):
                    self._tb = [line]
                elif line:
                    done.append(line)
        for entry in done:
            self._buf.append(entry, self._name)

    def flush(self):
        with _WRITE:
            if self._pending:
                self._stream.write(self._pending)
                self._pending = ""
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


BUFFER = LogBuffer()
_installed = {"on": False}
CONFIG = {"log_content": False}
_REQUEST = threading.local()


def install(buf: LogBuffer = BUFFER) -> LogBuffer:
    """Tee stdout and stderr into `buf`. Once per process."""
    if not _installed["on"]:
        sys.stdout = Tee(sys.stdout, buf, "stdout")
        sys.stderr = Tee(sys.stderr, buf, "stderr")
        _installed["on"] = True
    return buf


# ----------------------------------------------------------------- the no-content rule

def request_texts(body: dict) -> list[str]:
    """Every piece of text a client sent in this request: message contents, tool definitions,
    the prompt, stop strings. What an exception message may not quote into the log."""
    out = []

    def add(v):
        if isinstance(v, str) and v:
            out.append(v)
        elif isinstance(v, list):
            for x in v:
                add(x)
        elif isinstance(v, dict):
            for x in v.values():
                add(x)
    if isinstance(body, dict):
        for key in ("messages", "prompt", "tools", "stop", "input", "suffix"):
            add(body.get(key))
    return out


class serving:
    """`with logbuf.serving(body):` -- the request this thread is serving, for `safe_message`."""

    def __init__(self, body: dict):
        self.body = body

    def __enter__(self):
        _REQUEST.texts = None
        _REQUEST.body = self.body
        return self

    def __exit__(self, *exc):
        _REQUEST.texts = _REQUEST.body = None
        return False


def _texts() -> list[str]:
    if getattr(_REQUEST, "texts", None) is None:
        body = getattr(_REQUEST, "body", None)
        _REQUEST.texts = request_texts(body) if body is not None else []
    return _REQUEST.texts


def quotes(msg: str, texts: list[str]) -> bool:
    if not msg or not texts:
        return False
    w = min(WINDOW, len(msg))
    grams = {msg[i:i + w] for i in range(len(msg) - w + 1)}
    return any(g in t for t in texts for g in grams if g.strip())


def safe_message(exc: BaseException) -> str:
    """An exception's message for the log: capped, and withheld if it quotes the request."""
    msg = str(exc)
    if CONFIG["log_content"]:
        return msg[:MSG_CAP * 10]
    msg = msg[:MSG_CAP]
    if quotes(msg, _texts()):
        return "(message withheld: it quotes the request; --log-content shows it)"
    return msg


def print_exc(file=None) -> None:
    """`traceback.print_exc()`, with every message in the chain through `safe_message`."""
    exc = sys.exc_info()[1]
    if exc is None:
        return
    out = file or sys.stderr
    chain, seen = [], set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append(e)
        e = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
    parts = []
    for i, e in enumerate(reversed(chain)):
        if i:
            parts.append("\nDuring handling of the above exception, another exception occurred:"
                         "\n\n")
        parts.append("Traceback (most recent call last):\n")
        parts.extend(traceback.format_tb(e.__traceback__))
        mod = type(e).__module__
        name = type(e).__qualname__ if mod in ("builtins", "__main__") else f"{mod}.{type(e).__qualname__}"
        parts.append(f"{name}: {safe_message(e)}\n")
    out.write("".join(parts))
    out.flush()


REQUEST_KEYS = ("model", "stream", "stream_options", "max_tokens", "max_completion_tokens",
                "temperature", "top_p", "top_k", "seed", "presence_penalty", "frequency_penalty",
                "repetition_penalty", "no_repeat_ngram_size", "reasoning_effort",
                "reasoning_format", "max_reasoning_tokens", "thinking_budget", "n",
                "draft_temperature", "logprobs", "parallel_tool_calls", "min_p", "top_logprobs",
                # the other thinking-budget shapes (server/thinking.py)
                "reasoning", "thinking", "enable_thinking")


def request_keys_line(cid: str, body: dict) -> str:
    """`--log-request-keys`: the parameter names a client sent, and the scalar values of the ones
    that are not content -- what needs to see of Open WebUI's requests. Never a message,
    a prompt, a tool, a stop string or a user id."""
    shown = {}
    for k in REQUEST_KEYS:
        if k in body:
            v = body[k]
            if isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if isinstance(vv, (bool, int, float))}
            elif not isinstance(v, (str, bool, int, float)) and v is not None:
                continue
            shown[k] = v
    tc = body.get("tool_choice")
    if tc is not None:
        shown["tool_choice"] = tc if isinstance(tc, str) else (tc.get("type") if isinstance(tc, dict)
                                                                else "?")
    kw = body.get("chat_template_kwargs")
    if isinstance(kw, dict):
        shown["chat_template_kwargs"] = {k: v for k, v in kw.items()
                                         if isinstance(v, (bool, int, float))}
    counts = {k: len(body[k]) for k in ("messages", "tools", "stop") if isinstance(body.get(k), list)}
    return (f"[body] {cid} keys={','.join(sorted(body))} {json.dumps(shown, sort_keys=True)}"
            f" counts={json.dumps(counts, sort_keys=True)}")
