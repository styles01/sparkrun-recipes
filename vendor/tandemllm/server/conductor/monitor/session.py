"""Sessions: which requests belong to one agent session, and what the monitor keeps about it.

The key, in order:

1. the conversation id the client sends (`conversation_id`, `session_id` or `user` in the body,
   `X-Conversation-Id`, or opencode's `x-session-id` header);
2. a compaction request (no tools and one user message: opencode's summary request) joins the
   session seen most recently within the join window, and the next request that arrives with a
   prefix nobody has seen joins it too;
3. otherwise the lineage: a hash of the first `lineage_tokens` prompt tokens, joined to the
   session with that hash when it was seen within the join window, a fresh session when not.

The state is an LRU of `max_sessions`; nothing in it is text. Tool calls are kept as a name and a
hash of the canonical arguments, tool results as a hash, so a rule can see "the same call with the
same result" without the server storing either.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict, deque


def _h(data) -> str:
    raw = data if isinstance(data, bytes) else str(data).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def args_hash(arguments) -> str:
    """The hash of a call's arguments, canonical: key order and whitespace do not count."""
    try:
        obj = json.loads(arguments) if isinstance(arguments, str) else arguments
        return _h(json.dumps(obj, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError):
        return _h(arguments)


class SessionState:
    """One session. `calls` and `results` are the newest 64 each. `tasks` is the session tasks'
    own state (S1, S2, S4: sessiontasks.py), by task id; `watch` what the monitor's watch needs
    once requests stop (S4's limit and NOTIFY settings as the last request saw them)."""

    def __init__(self, key: str, now: float):
        self.key = key
        self.created = self.last_seen = now
        self.requests = 0
        self.calls: deque = deque(maxlen=64)      # (ts, name, args hash)
        self.results: deque = deque(maxlen=64)    # (ts, tool_call_id, result hash)
        self.checks: dict = {}                    # checker gate -> results (SRV-123 fills it)
        self.last_compaction: float | None = None
        self.acts: deque = deque(maxlen=256)      # times of executed acts
        self.last_request: str | None = None      # its checker result labels this one's events
        self.tasks: dict = {}
        self.note_at: float | None = None         # the last session note (S1, S2, S4)
        self.ended_at: float | None = None        # the last request's end
        self.watch: dict | None = None
        self.gone_at: float | None = None         # the turn end a client_gone was journaled for
        # tool call id -> seconds the client took to send its result back: from the end of the
        # answer that issued the call to the request that carried the result (the tool's run and
        # the client's own time); C3 and the loop escape read a long one as a wait, not a loop
        self.waits: OrderedDict = OrderedDict()

    def idle(self, now: float) -> float | None:
        """Seconds since the last turn ended, None while a request runs or before the first
        one ended."""
        if self.ended_at is None or self.ended_at < self.last_seen:
            return None
        return now - self.ended_at

    def acts_last_hour(self, now: float) -> int:
        return sum(1 for t in self.acts if now - t < 3600.0)

    def record_act(self, now: float) -> None:
        self.acts.append(now)

    def record_call(self, name: str, arguments, now: float) -> None:
        self.calls.append((now, name, args_hash(arguments)))

    def record_results(self, messages: list, now: float) -> None:
        """The tool results a request carries after the last assistant message, as hashes."""
        tail = []
        for m in reversed(messages or []):
            if not isinstance(m, dict) or m.get("role") != "tool":
                break
            tail.append(m)
        gap = self.gap(now)
        for m in reversed(tail):
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            cid = str(m.get("tool_call_id") or "")
            self.results.append((now, cid, _h(c or "")))
            if cid and gap is not None and cid not in self.waits:
                self.waits[cid] = gap
        while len(self.waits) > 64:
            self.waits.popitem(last=False)

    def gap(self, now: float) -> float | None:
        """Seconds since the session's last request ended (the client's time on the results this
        request carries), None before any request ended."""
        return None if self.ended_at is None else max(0.0, now - self.ended_at)


class Sessions:
    """The session key for a request, and the LRU of session states."""

    def __init__(self, max_sessions: int, join_window_s: float, lineage_tokens: int):
        self.max = int(max_sessions)
        self.window = float(join_window_s)
        self.lineage = int(lineage_tokens)
        self._d: OrderedDict[str, SessionState] = OrderedDict()
        self._alias: dict[str, str] = {}
        self._after_compaction: tuple[str, float] | None = None

    def __len__(self):
        return len(self._d)

    def states(self) -> list[SessionState]:
        """A snapshot of the states, newest last (the watch reads it from its own thread)."""
        for _ in range(3):
            try:
                return list(self._d.values())
            except RuntimeError:                 # changed size under us: read it again
                continue
        return []

    def get(self, key: str) -> SessionState | None:
        return self._d.get(key)

    def key_for(self, conv_id: str | None, prompt_ids, tools_offered: bool, messages,
                now: float | None = None) -> tuple[str, str]:
        """`(key, how)` for a request; `how` is `conversation`, `compaction` or `lineage`."""
        now = time.time() if now is None else now
        if conv_id:
            return "conv:" + _h(conv_id), "conversation"
        users = [m for m in messages or () if isinstance(m, dict) and m.get("role") == "user"]
        if not tools_offered and len(users) == 1 and messages and len(messages) <= 2:
            recent = self._recent(now)
            if recent is not None:
                self._after_compaction = (recent.key, now)
                recent.last_compaction = now
                return recent.key, "compaction"
        key = "lin:" + _h(",".join(str(int(t)) for t in list(prompt_ids)[:self.lineage]))
        key = self._alias.get(key, key)
        known = self._d.get(key)
        if known is None and self._after_compaction is not None:
            joined, at = self._after_compaction
            self._after_compaction = None
            if now - at <= self.window and joined in self._d:
                self._alias[key] = joined
                return joined, "compaction"
        return key, "lineage"

    def _recent(self, now: float) -> SessionState | None:
        for s in reversed(self._d.values()):
            return s if now - s.last_seen <= self.window else None
        return None

    def touch(self, key: str, now: float | None = None) -> SessionState:
        """The state for `key`, fresh when it is new or was last seen outside the window."""
        now = time.time() if now is None else now
        s = self._d.get(key)
        if s is None or now - s.last_seen > self.window:
            s = SessionState(key, now)
        self._d[key] = s
        self._d.move_to_end(key)
        s.last_seen = now
        s.requests += 1
        while len(self._d) > self.max:
            old, _ = self._d.popitem(last=False)
            for a in [a for a, k in self._alias.items() if k == old]:
                del self._alias[a]
        return s
