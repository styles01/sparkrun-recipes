"""The subscribers of an event stream, and the SSE frames they get.

A conversation's events are written to the store and then handed to every subscriber of that
conversation, under the conversation's lock, so the order a subscriber sees is the order of `seq`.
Each subscriber has its own bounded queue (CONTRACT section 3; SRV-343): a client that stops
reading fills only its own queue, and at `MAX_QUEUE` items it is marked `overflow`, gets `event:
reset` and is dropped -- it reloads the conversation, as one that fell 2,000 events behind does.

THE SWITCH FROM REPLAY TO LIVE (SRV-345, "no gap and no duplicate"). `attach()` registers the
subscriber and reads the conversation's `last_seq` in one step under the lock that `emit` holds
while it assigns a seq and fans out. Every event up to that seq is in the store (replay reads them
from there); every event after it lands in the subscriber's queue. Nothing falls between and
nothing comes twice.
"""

from __future__ import annotations

import collections
import json
import threading

MAX_QUEUE = 10_000          # items a subscriber may have waiting before it is dropped with `reset`
RESET_BEHIND = 2_000        # a client further behind than this reloads instead of replaying
HEARTBEAT_S = 15.0


class Subscriber:
    """One open stream. Items are `(seq | None, type, data_json)`; seq None is ephemeral."""

    def __init__(self, max_queue: int = MAX_QUEUE):
        self.max_queue = max_queue
        self.items: collections.deque = collections.deque()
        self.cv = threading.Condition()
        self.overflow = False
        self.closed = False

    def put(self, item: tuple) -> None:
        with self.cv:
            if self.closed:
                return
            if len(self.items) >= self.max_queue:
                self.overflow = True
                self.items.clear()
            else:
                self.items.append(item)
            self.cv.notify()

    def take(self, timeout: float) -> list[tuple] | None:
        """Everything waiting, or [] after `timeout` with nothing; None once closed."""
        with self.cv:
            if not self.items and not self.overflow and not self.closed:
                self.cv.wait(timeout)
            if self.closed:
                return None
            out = list(self.items)
            self.items.clear()
            return out

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify()


class Fanout:
    """The subscribers of one stream (a conversation's, or the global one)."""

    def __init__(self):
        self.subs: set[Subscriber] = set()

    def add(self, s: Subscriber) -> None:
        self.subs.add(s)

    def remove(self, s: Subscriber) -> None:
        self.subs.discard(s)

    def send(self, seq: int | None, kind: str, data: str) -> None:
        for s in list(self.subs):
            s.put((seq, kind, data))
            if s.overflow:
                self.subs.discard(s)

    def __len__(self) -> int:
        return len(self.subs)


def frame(seq: int | None, kind: str, data: str) -> bytes:
    """One SSE frame: `id:` only for a persisted event."""
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}event: {kind}\ndata: {data}\n\n".encode()


def control(kind: str) -> bytes:
    """`event: current` / `event: reset`: no id, `data: {}`."""
    return f"event: {kind}\ndata: {{}}\n\n".encode()


HEARTBEAT = b": hb\n\n"


def as_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
