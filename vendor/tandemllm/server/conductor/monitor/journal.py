"""The journal: every decision the monitor made, with the text it hid and what came of it.

The transcript a client keeps differs from what the model saw whenever the monitor acts, so each
hidden injection is recorded with its exact text, where it went, why, and the outcome. This module
keeps the newest `keep` events in memory and prints one `[conductor]` line per event -- the line
names the task, the decision and the counts, never a text. The conductor's journal (the SQLite
table `conductor_events` in the usage ledger, server/conductor/decisions.py), the Prometheus
counters (server/metrics.py) and the session export attach through `sinks` (SRV-115): a sink is
called with every event after it is recorded, and a sink that raises fails nothing.

An event is a dict. The keys the monitor writes:

    ts, request_id, session, sp, task, rule, tier, mode ("shadow" | "act"),
    decision (an action, or PASS), reason, score, detail, text, hidden_tokens, at_tokens, pos,
    cap (why an act became GIVE_UP), client, profile

and three bookkeeping events that are not decisions, which the sinks read and nobody prints:

    OUTCOME  the request ended: `outcome` (tool_call | text | empty | gave_up), acts, hidden_tokens
    LABEL    a later request carried the checker's result: `next_checker` for `request_id`'s events
    COUNTS   the checks that found nothing, `counts` = [[task, tier, mode, decision, n], ...]

`quiet` marks a decision that is journaled but not printed (an answer whose calls all passed).
"""

from __future__ import annotations

import threading
import time
from collections import deque

BOOKKEEPING = ("OUTCOME", "LABEL", "COUNTS")


class Journal:
    def __init__(self, keep: int = 4096, echo: bool = True):
        self._events: deque = deque(maxlen=int(keep))
        self._lock = threading.Lock()
        self.sinks: list = []
        self.echo = echo

    def write(self, event: dict) -> dict:
        """Record one event (a `ts` is added when it has none) and hand it to the sinks."""
        event.setdefault("ts", time.time())
        if event.get("decision") != "COUNTS":
            with self._lock:
                self._events.append(event)
        if self.echo and event.get("decision") not in ("LABEL", "COUNTS") \
                and not event.get("quiet"):
            print(_line(event), flush=True)
        for sink in list(self.sinks):
            try:
                sink(event)
            except Exception:                                      # noqa: BLE001
                pass                       # a sink must never fail a request
        return event

    def events(self, request_id: str | None = None, session: str | None = None) -> list[dict]:
        with self._lock:
            evs = list(self._events)
        return [e for e in evs
                if (request_id is None or e.get("request_id") == request_id)
                and (session is None or e.get("session") == session)]

    def actions(self, request_id: str | None = None) -> list[dict]:
        """The events that executed something: mode act and a decision other than PASS."""
        return [e for e in self.events(request_id)
                if e.get("mode") == "act" and e.get("decision") not in (None, "PASS")
                and e.get("decision") not in BOOKKEEPING]


def _line(e: dict) -> str:
    parts = [f"[conductor] {e.get('request_id', '-')} {e.get('sp', '-')}",
             f"task={e.get('task') or '-'}", f"mode={e.get('mode', '-')}",
             f"decision={e.get('decision', '-')}"]
    for k in ("reason", "cap", "hidden_tokens", "at_tokens", "outcome"):
        v = e.get(k)
        if v not in (None, ""):
            parts.append(f"{k}={v}")
    return " ".join(parts)
