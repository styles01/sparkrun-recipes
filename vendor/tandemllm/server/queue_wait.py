"""How long a request waits for the engine, said honestly (2026-10-05).

The engine serves one generation at a time. On 2026-10-04 a second opencode session's turn waited
behind a running turn and got a 429 after `--queue-timeout` (240 s), then a second 429: minutes of
waiting, and then nothing. A streamed chat request now waits as long as one request may run
(`--request-timeout`), and the client is told that it is waiting:

  - after `OPEN_AFTER_S` the stream opens (headers, status 200), and an SSE comment
    (`: waiting, position 1, about 40 s`) goes out every `PING_EVERY_S`. A comment is not an
    event: no client reads it as a token. It keeps proxies and clients with an idle-read timeout
    from closing the connection, and the client never sees a refusal after minutes.
  - with the profile's progress lines on, the waiting line carries the estimate:
    "waiting: another request is running, position 1, about 40 s". A new line goes out when the
    reason or the position changes, and at most every `LINE_EVERY_S` otherwise.

A request without a stream cannot show it is alive; it keeps `--queue-timeout` and its 429.

The holder's live work (SRV-249, 2026-10-07). The hold-time median knew nothing about where the
holder was: on a 0.4.0 test build a one-line question waited 35.8 s behind a thinking turn that had run
14.8 s, and the line said "about 20 s". Replayed over the box's ledger (3,716 client requests since
2026-10-04, the waiter arriving at 10, 25, 50, 75 and 90 % of each hold), the hold-time median was
a median 0.52 of the real wait for waits of 10 to 60 s (78 % of them under) and 0.17 for waits over
a minute. When the handler can see the holder (`Live`: its prefill marks, the tokens it has
written, its decode rate, whether it thinks, its `max_tokens`), the remaining time is now the
prefill still to read at its live rate, plus the tokens still to write: the `TOKENS_Q` quantile of
what earlier requests of the same kind (thinking or not) wrote past the tokens the holder has
written, never more than its `max_tokens` allows, at its own rate (before its first token, the
median rate of the earlier ones). Same replay: a median 1.06 of the real wait for 10 to 60 s
(42 % under), 0.59 over a minute; waits under 10 s read about twice their length (one line at
most, rounded by `about`). Without that view (TandemLLM's own work holds the engine, too few
samples) the hold-time median stays.

The estimate. `TurnClock` keeps the wall time each request held the engine, per rank (a turn, a
side request too long for the lane). For the request that holds the engine now, having run `e`
seconds, the remaining time is the median of `d - e` over the recorded times `d` longer than `e`:
how much longer requests that ran this long went on running. A waiter ahead adds its rank's
median. Fewer than `MIN_SAMPLES` recorded times, or fewer than `MIN_LONGER` that ran longer than
the holder has, and there is no estimate: the line says only what is known. Nothing here reads the
engine or takes its lock.

TandemLLM's own work (SRV-177, 2026-10-06). The prepare and shadow jobs that take the lock between
requests yield to a waiter at their next chunk. Until 0.4.0 they counted as 0 s, and `about`
rounded every estimate up to "10 s", so the line behind them always read "about 10 s left" (9 of 9
lines in UT1's opencode runs) while the waits measured in the ledger were 0.1 to 3 s (18 of 60 turns
over 1 s, 9 over 2 s). Now the wait a request really spent behind such a job is recorded per rank
(`yielded`), the estimate is the median of those that lasted longer than the wait so far, there is
no estimate before `MIN_SAMPLES` of them, and a wait under `QUIET_S` shows no line at all.
"""

from __future__ import annotations

import statistics
import threading
from collections import deque

#: the stream opens and the first comment goes out after this long in the queue
OPEN_AFTER_S = 10.0
#: one comment this often while the request waits
PING_EVERY_S = 10.0
#: a waiting line with the same reason and position goes out again after this long
LINE_EVERY_S = 60.0
#: recorded hold times kept per rank
KEEP = 200
#: no estimate before this many recorded times
MIN_SAMPLES = 5
#: no estimate for the holder unless this many recorded times ran longer than it has
MIN_LONGER = 3
#: SRV-177: a request that has waited less than this shows no waiting line
QUIET_S = 2.0
#: the recorded waits behind a yielding job are kept under rank + this (a rank's hold times stay apart)
YIELD_KEY = 100
#: SRV-249: the holder's tokens still to write, as this quantile of what earlier requests of its
#: kind wrote past the tokens it has written (the replay in the module doc: the median read low)
TOKENS_Q = 0.75


class Live:
    """What the handler sees of the request that holds the engine (SRV-249): `pf_left` tokens
    still to prefill at `pf_rate` tok/s (0 once it decodes), `written` tokens so far at `rate`
    tok/s (None before a rate is known), `thinking`, `max_tokens` (None: no cap known)."""

    __slots__ = ("pf_left", "pf_rate", "written", "rate", "thinking", "max_tokens")

    def __init__(self, pf_left: int = 0, pf_rate: float | None = None, written: int = 0,
                 rate: float | None = None, thinking: bool = False,
                 max_tokens: int | None = None):
        self.pf_left, self.pf_rate = int(pf_left), pf_rate
        self.written, self.rate = int(written), rate
        self.thinking, self.max_tokens = bool(thinking), max_tokens


class TurnClock:
    """The wall times requests held the engine, per rank, and the wait they predict."""

    def __init__(self, keep: int = KEEP):
        self._lock = threading.Lock()
        self._keep = int(keep)
        self._held: dict[int, deque] = {}
        # SRV-249: the tokens each finished request wrote and its decode rate, by thinking
        self._wrote: dict[bool, deque] = {}
        self._rates: deque = deque(maxlen=self._keep)

    def wrote(self, thinking: bool, tokens: int, rate: float | None = None) -> None:
        """A client's request finished after writing `tokens` (thinking or not), at `rate`."""
        if tokens is None or tokens < 0:
            return
        with self._lock:
            self._wrote.setdefault(bool(thinking), deque(maxlen=self._keep)).append(int(tokens))
            if rate:
                self._rates.append(float(rate))

    def live_remaining(self, live: "Live") -> float | None:
        """SRV-249: seconds the holder still needs, from its own work (`Live`); None: no basis
        (fewer than `MIN_SAMPLES` finished requests of its kind, fewer than `MIN_LONGER` that wrote
        more than it has, or no rate)."""
        with self._lock:
            d = list(self._wrote.get(bool(live.thinking)) or ())
            rates = list(self._rates)
        if len(d) < MIN_SAMPLES:
            return None
        longer = sorted(x - live.written for x in d if x > live.written)
        if len(longer) < MIN_LONGER:
            return None
        left = longer[min(len(longer) - 1, int(TOKENS_Q * len(longer)))]
        if live.max_tokens:
            left = min(left, max(0, int(live.max_tokens) - live.written))
        rate = live.rate or (statistics.median(rates) if rates else None)
        if not rate:
            return None
        pre = 0.0
        if live.pf_left > 0:
            if not live.pf_rate:
                return None
            pre = live.pf_left / float(live.pf_rate)
        return pre + left / float(rate)

    def done(self, rank: int, held_s: float) -> None:
        """A request of `rank` held the engine `held_s` seconds."""
        if held_s is None or held_s < 0:
            return
        with self._lock:
            self._held.setdefault(int(rank), deque(maxlen=self._keep)).append(float(held_s))

    def _samples(self, rank: int) -> list[float]:
        with self._lock:
            return list(self._held.get(int(rank)) or ())

    def yielded(self, rank: int, waited_s: float) -> None:
        """SRV-177: a request waited `waited_s` seconds behind a yielding job of `rank` (the
        prepare or shadow job) before it got the engine."""
        self.done(YIELD_KEY + int(rank), waited_s)

    def remaining(self, rank: int, elapsed: float) -> float | None:
        """Seconds the holder of `rank`, `elapsed` seconds in, still holds the engine; None: no
        basis."""
        d = self._samples(rank)
        if len(d) < MIN_SAMPLES:
            return None
        longer = [x - elapsed for x in d if x > elapsed]
        if len(longer) < MIN_LONGER:
            return None
        return float(statistics.median(longer))

    def typical(self, rank: int) -> float | None:
        """The median hold time of `rank`; None: too few recorded."""
        d = self._samples(rank)
        return float(statistics.median(d)) if len(d) >= MIN_SAMPLES else None

    def estimate(self, holder: tuple[int, float] | None, ahead: list[int],
                 yields: frozenset = frozenset(), live: "Live | None" = None) -> float | None:
        """Seconds until a waiter gets the engine: the holder's remaining time (`(rank, elapsed)`,
        None when nobody holds it) plus each waiter ahead's typical time (`ahead`: their ranks).
        A rank in `yields` (the prepare and shadow jobs) counts as 0. `live`: what the handler
        sees of the holder's work (SRV-249), used when it gives a basis. None: no basis for one of
        them."""
        total = 0.0
        if holder is not None:
            r = (self.live_remaining(live)
                 if live is not None and holder[0] not in yields else None)
            # a yielding holder: what waiters behind that job really waited (SRV-177), never 0
            if r is None:
                r = self.remaining(YIELD_KEY + holder[0] if holder[0] in yields else holder[0],
                                   holder[1])
            if r is None:
                return None
            total += r
        for rank in ahead:
            if rank in yields:
                continue
            t = self.typical(rank)
            if t is None:
                return None
            total += t
        return total


def about(seconds: float) -> str:
    """`40 s`, `5 min`: the estimate in the waiting line, rounded so it does not read as exact."""
    s = max(0.0, float(seconds))
    if s < 10:
        return f"{max(1, int(round(s)))} s"   # SRV-177: no longer rounded up to "10 s"
    if s < 90:
        return f"{max(1, int(round(s / 10.0))) * 10} s"
    return f"{max(2, int(round(s / 60.0)))} min"


def comment(pos: int, eta: float | None, waited: float) -> bytes:
    """The SSE comment a waiting stream gets: position, estimate, time waited so far."""
    tail = f", about {about(eta)}" if eta is not None else ""
    return f": waiting, position {int(pos)}{tail}, {waited:.0f} s so far\n\n".encode()
