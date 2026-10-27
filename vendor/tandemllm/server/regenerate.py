"""Regenerate: the one request TandemLLM samples on its own (SRV-241, 2026-10-07).

THE FINDING. Open WebUI's Regenerate gave the word-for-word same answer (UX2, W03: five bakery
names, same wording), and on 0.3.49 the same five names in the same order. The server default is
greedy (temperature 0, the exact path) and Open WebUI sends no sampling parameter, so the same
prompt gives the same answer. To the user Regenerate looks broken.

THE RULE (the project's, 2026-10-07: follow best practice). A normal answer stays exactly as
today: greedy, lossless, the drafter on. Only a regenerate gets a varied answer.

WHAT A REGENERATE IS. Open WebUI sends no flag. What it sends was read off its own source
(0.11.4, `src/lib/components/chat/Chat.svelte` from the image's source map):

  * Regenerate: `regenerateResponse` -> `sendMessage(history, userMessage.id)`: a new, empty
    answer beside the old one, and the messages up to its parent user message (the empty answer is
    filtered out) -- exactly the messages of the request that produced the answer it replaces;
  * Continue: `continueResponse` -> `createMessagesList(history, responseMessage.id)`: the
    messages end with the partial ASSISTANT answer;
  * Regenerate with a suggestion ("Try again", "Add details"): the old answer and a new user
    message after it -- a new instruction, not a repeat;
  * Edit (a user message): a new user message with the edited text.

So a regenerate is a chat request whose conversation (every message but the system ones, which
Open WebUI may re-render with a time variable) ends with a user message and equals, byte for
byte, the conversation of an earlier request of the same session that produced an answer. Continue
ends with an assistant message and an edit or a suggestion changes the messages, so neither
matches; a tool loop appends a result every round, so it never repeats either. The session is the
conductor's (`c:<chat id>` from `X-OpenWebUI-Chat-Id`, opencode's `x-session-id`, a body id), or,
without one, a hash of the conversation's head (`h:`): there the rule reads "the same
conversation again after it was answered", which is what a regenerate is.

The match is the conversation's own bytes (a hash of the canonical JSON), never words in the
message: nothing here reads what the user wrote.

WHAT A REGENERATE GETS. The request samples (temperature 0.7, top-p 0.95, top-k 20 by default:
Qwen's own recommendation for thinking mode, with temperature 0.1 warmer, so a list of five names
comes back with other names) with a FRESH seed per attempt, recorded in the journal row, so any
regenerate can be reproduced by sending that seed. Everything else of the plan stays: the class,
the effort, the notes and the packages are the turn's as before. The sampled path is the engine's
exact one (engine/sample.py; docs/exactness.md): under speculation the target draws its own token
at each node and a drafted token survives only while the draw lands on it, so the answer follows
the target's own sampled distribution, drafter or not. On the box (`--sampled-tree det`) the
sampled request walks the greedy request's tree.

WHAT NEVER CHANGES.
  * A request that is not a regenerate: greedy, byte for byte as before.
  * A request that names any sampling parameter (`temperature`, `top_p`, `top_k`, `min_p`, `seed`):
    served exactly as the client asked, regenerate or not (the row still says it was one).
  * Raw requests (`X-Tandem: raw`, the benchmark profile) and an engine without the conductor:
    never looked at -- benchmarks and the paper's numbers keep the plain greedy engine.
  * A side request (a title, a summary) and a compaction.

OTHER CLIENTS. `X-Tandem-Regenerate: 1` asks for a regenerate outright; `X-Tandem-Regenerate: 0`
turns the detection off for that request (a client that re-sends one prompt on purpose and wants
the greedy answer each time). Off for the server: `--regenerate off` (`QWEN38_REGENERATE=off`).
"""

from __future__ import annotations

import collections
import hashlib
import json
import secrets
import threading
import time

HEADER = "X-Tandem-Regenerate"
# the sampling parameters a client may name; any of them present means the client decides
CLIENT_KEYS = ("temperature", "top_p", "top_k", "min_p", "seed")
# message fields that are presentation, not conversation: a client may or may not send an
# answer's reasoning back (Open WebUI never does), and that must not change what the turn was
_SKIP_FIELDS = ("reasoning_content", "reasoning", "thinking", "name")

DEFAULTS = {"mode": "sample", "temperature": 0.7, "top_p": 0.95, "top_k": 20}


def _header(headers, name: str) -> str | None:
    try:
        v = headers.get(name) if headers is not None else None
    except Exception:                                              # noqa: BLE001
        v = None
    return v.strip() if isinstance(v, str) and v.strip() else None


def _norm(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if k not in _SKIP_FIELDS}


def signature(messages) -> str | None:
    """The conversation's fingerprint for the match: every non-system message, canonical JSON,
    sha256. None when the request cannot be a regenerate: no messages, or the last one is not
    the user's (Continue ends with the assistant's partial answer; a tool round with its result)."""
    if not isinstance(messages, list):
        return None
    conv = [_norm(m) for m in messages if isinstance(m, dict) and m.get("role") not in
            ("system", "developer")]
    if not conv or conv[-1].get("role") != "user":
        return None
    raw = json.dumps(conv, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()


def head_key(messages) -> str:
    """The fallback session key: the conversation up to and including its first user message."""
    head = []
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict):
            continue
        head.append(_norm(m))
        if m.get("role") == "user":
            break
    raw = json.dumps(head, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "h:" + hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:24]


def session_of(body: dict, headers) -> str:
    """The conductor's session key (`router.session_key`), with the head hash as the fallback."""
    for k in ("conversation_id", "session_id"):
        v = body.get(k)
        if isinstance(v, str) and v:
            return "c:" + v[:120]
    meta = body.get("metadata")
    if isinstance(meta, dict):
        v = meta.get("conversation_id") or meta.get("session_id") or meta.get("chat_id")
        if isinstance(v, str) and v:
            return "c:" + v[:120]
    for h in ("x-session-id", "X-Conversation-Id", "X-OpenWebUI-Chat-Id"):
        v = _header(headers, h)
        if v:
            return "c:" + v[:120]
    return head_key(body.get("messages"))


def client_sampling(body: dict) -> bool:
    """The client named a sampling parameter: it decides, as before."""
    return any(body.get(k) is not None for k in CLIENT_KEYS)


class Decision:
    __slots__ = ("key", "sig", "how", "of", "client")

    def __init__(self, key, sig, how=None, of=None, client=False):
        self.key, self.sig, self.how, self.of, self.client = key, sig, how, of, client

    @property
    def sample(self) -> bool:
        """This request gets the regenerate sampling (a regenerate, and the client named none)."""
        return self.how is not None and not self.client


class Regenerates:
    """Per session, the conversations that were answered (most recent first, a few per session),
    so a regenerate of the last answer -- or of an earlier one, Open WebUI regenerates any answer
    -- is recognised. Bounded: `sessions` sessions, `per` conversations each, `ttl_s` old at most."""

    def __init__(self, mode: str = "sample", temperature: float = 0.7, top_p: float = 0.95,
                 top_k: int = 20, sessions: int = 1024, per: int = 32, ttl_s: float = 86400.0,
                 clock=time.time):
        if mode not in ("sample", "off"):
            raise ValueError(f"regenerate mode must be sample or off, got {mode!r}")
        if not 0.0 < float(temperature) <= 2.0 or not 0.0 < float(top_p) <= 1.0 or int(top_k) < 0:
            raise ValueError(f"regenerate sampling out of range: temperature {temperature} "
                             f"(0, 2], top_p {top_p} (0, 1], top_k {top_k} >= 0")
        self.mode = mode
        self.temperature, self.top_p, self.top_k = float(temperature), float(top_p), int(top_k)
        self.sessions_max, self.per, self.ttl_s, self.clock = sessions, per, ttl_s, clock
        self._s: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()
        self.counts = {"detected": 0, "header": 0, "client_params": 0, "answered": 0}

    def report(self) -> dict:
        with self._lock:
            return {"mode": self.mode, "temperature": self.temperature, "top_p": self.top_p,
                    "top_k": self.top_k, "sessions": len(self._s), **self.counts}

    def decide(self, body: dict, headers) -> Decision | None:
        """What this request is. None: the feature is off, or the request is not a chat."""
        if self.mode == "off" or not isinstance(body, dict):
            return None
        hv = _header(headers, HEADER)
        msgs = body.get("messages")
        sig = signature(msgs)
        key = session_of(body, headers)
        d = Decision(key, sig, client=client_sampling(body))
        if hv in ("0", "false", "off", "no"):
            d.sig = None                     # not detected, and not remembered either
            return d
        if hv in ("1", "true", "on", "yes"):
            d.how = "header"
        elif sig is not None:
            now = self.clock()
            with self._lock:
                sess = self._s.get(key)
                hit = sess.get(sig) if sess is not None else None
                if hit is not None and now - hit[0] <= self.ttl_s:
                    d.how, d.of = "detected", hit[1]
        if d.how is not None:
            with self._lock:
                self.counts["client_params" if d.client else d.how] += 1
        return d

    def answered(self, d: Decision | None, request_id: str) -> None:
        """The request produced an answer: its conversation counts as answered from now on."""
        if d is None or d.sig is None:
            return
        now = self.clock()
        with self._lock:
            sess = self._s.pop(d.key, None) or collections.OrderedDict()
            sess.pop(d.sig, None)
            sess[d.sig] = (now, request_id)
            while len(sess) > self.per:
                sess.popitem(last=False)
            self._s[d.key] = sess
            while len(self._s) > self.sessions_max:
                self._s.popitem(last=False)
            self.counts["answered"] += 1

    def sampling(self) -> dict:
        """The regenerate's sampling parameters, a fresh seed each call."""
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
                "seed": secrets.randbits(63)}


def answered_ok(status, finish_reason, completion_tokens) -> bool:
    """A request that wrote an answer the client saw: not an error or a refusal, at least one
    token. A stopped answer counts (Stop, then Regenerate, asks for another answer too)."""
    return ((status or 0) < 400 and finish_reason not in ("error", "refused")
            and bool(completion_tokens))
