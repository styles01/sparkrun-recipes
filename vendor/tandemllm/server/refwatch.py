"""The interpreter's shared singletons, watched for a C extension that drops their references.

On Python 3.11 `True`, `False`, `None` and the small ints are ordinary objects with a reference
count. A C extension that gives back a reference it never took takes one off them on every call;
nothing shows until the count reaches 0, and then the process dies wherever the next reference
goes ("Fatal Python error: bool_dealloc", 2026-10-05 13:12, three hours after the start: torch
2.13.0's `_cuda_memorySnapshot`, engine/decision/device.py). The traceback names the line that let
go of the last reference, not the caller that drained it.

This thread reads `sys.getrefcount` of each of them every `EVERY_S` (a read of one field, no lock)
and logs once per `WARN_EVERY_S` while one is under `FRACTION` of the highest count it has had,
with its fall rate over the last `WINDOW_S`. A count that falls steadily while the engine idles
points at what runs while it idles (the scrapes, the keepers); the rate says how long until the
crash. On Python 3.12 and later these objects are immortal and the thread does not start.
"""

from __future__ import annotations

import collections
import sys
import threading
import time

WATCH = (("True", True), ("False", False), ("None", None), ("0", 0), ("1", 1))
EVERY_S = 30.0
WINDOW_S = 600.0
FRACTION = 0.5
WARN_EVERY_S = 600.0


def immortal() -> bool:
    """Python 3.12+ (PEP 683): the singletons cannot be freed, the counts mean nothing."""
    return sys.version_info >= (3, 12)


class RefWatch:
    def __init__(self, log=print, getref=sys.getrefcount, clock=time.monotonic,
                 watch=WATCH, fraction: float = FRACTION, window_s: float = WINDOW_S,
                 warn_every_s: float = WARN_EVERY_S):
        self.log, self.getref, self.clock = log, getref, clock
        self.watch = tuple(watch)
        self.fraction, self.window_s, self.warn_every_s = fraction, window_s, warn_every_s
        self.high: dict[str, int] = {}
        self.now: dict[str, int] = {}
        self.warned: dict[str, float] = {}
        self.warnings = 0
        self._hist = {name: collections.deque() for name, _ in self.watch}

    def rate(self, name: str) -> float:
        """References lost per second over the window (positive while it falls)."""
        h = self._hist[name]
        if len(h) < 2 or h[-1][0] <= h[0][0]:
            return 0.0
        return (h[0][1] - h[-1][1]) / (h[-1][0] - h[0][0])

    def sample(self) -> list[str]:
        """One look at every watched object; returns the lines it logged."""
        t = self.clock()
        out = []
        for name, obj in self.watch:
            n = int(self.getref(obj))
            self.now[name] = n
            self.high[name] = max(self.high.get(name, 0), n)
            h = self._hist[name]
            h.append((t, n))
            while h and t - h[0][0] > self.window_s:
                h.popleft()
            high = self.high[name]
            if n >= high * self.fraction:
                continue
            last = self.warned.get(name)
            if last is not None and t - last < self.warn_every_s:
                continue
            self.warned[name] = t
            self.warnings += 1
            r = self.rate(name)
            eta = f", about {n / r / 60:.0f} min to 0 at this rate" if r > 0 else ""
            line = (f"[refwatch] {name}: {n} references, {100 * n / max(high, 1):.0f} % of its "
                    f"high {high}; falling {r:.2f}/s over the last {self.window_s / 60:.0f} min"
                    f"{eta}. A C extension gives back references it never took; at 0 the "
                    f"process dies (Fatal Python error: *_dealloc)")
            self.log(line)
            out.append(line)
        return out

    def stats(self) -> dict:
        return {"now": dict(self.now), "high": dict(self.high), "warnings": self.warnings,
                "rate_per_s": {name: round(self.rate(name), 3) for name, _ in self.watch}}


def start(log=print, every_s: float = EVERY_S) -> RefWatch | None:
    """The watch on a daemon thread; None on an interpreter with immortal singletons."""
    if immortal():
        return None
    w = RefWatch(log=log)

    def run():
        while True:
            try:
                w.sample()
            except Exception:                                      # noqa: BLE001
                pass
            time.sleep(every_s)

    threading.Thread(target=run, name="refwatch", daemon=True).start()
    return w
