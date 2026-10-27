"""The ring and the poster of a TandemLLM engine plugin (layer L2 of docs/proxy.md).

The same file ships in both plugin packages (plugins/vllm and plugins/sglang), byte for byte, so
that neither package depends on anything but the engine it runs in; a test in the repository
checks the two copies are equal.

THE RING. The engine's own thread calls `add_request` once per finished request and `add_step`
once per scheduler step (or per logging interval). Each call is one dict append under a lock held
for a few microseconds. The ring keeps the last 10,000 requests and the last 300 step samples;
older items fall off the front and are counted as dropped. Every item gets a sequence number, so
a reader asks for what came after the last number it saw.

THE POSTER. A daemon thread posts what is new once a second to the proxy's
`POST /v1/collect/<backend>` (`TANDEM_COLLECT_URL`, the whole URL) with the shared token
(`TANDEM_COLLECT_TOKEN`) as a bearer. An idle engine still posts an empty batch every 10 s, so
the proxy can tell an idle plugin from a missing one. A failed post keeps the cursor where it was
(the ring still holds the items) and waits 1, 2, 4 ... up to 30 s; a refused token waits 60 s.
It logs at most one line a minute. Without `TANDEM_COLLECT_URL` there is no thread, and the ring
is read through the pull route alone (vLLM's endpoint plugin).

Nothing here reads a prompt, an answer or a token id: only counts and times.
"""

from __future__ import annotations

import collections
import itertools
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid

REQUESTS_MAX = 10_000
STEPS_MAX = 300
POST_EVERY_S = 1.0
HEARTBEAT_S = 10.0
TIMEOUT_S = 2.0
BATCH_MAX = 2_000
BACKOFF_MAX_S = 30.0
REFUSED_WAIT_S = 60.0
LOG_EVERY_S = 60.0


class Ring:
    """A bounded list of (sequence number, item); numbers start at 1 and never repeat."""

    def __init__(self, maxlen: int):
        self._d: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.seq = 0
        self.dropped = 0

    def add(self, item: dict) -> int:
        with self._lock:
            self.seq += 1
            if len(self._d) == self._d.maxlen:
                self.dropped += 1
            item["seq"] = self.seq
            self._d.append(item)
            return self.seq

    def after(self, seq: int, limit: int = BATCH_MAX) -> list[dict]:
        """The items numbered above `seq`, oldest first, at most `limit`."""
        with self._lock:
            if not self._d:
                return []
            first = self._d[0]["seq"]
            start = max(0, int(seq) + 1 - first)
            return list(itertools.islice(self._d, start, start + limit))

    def __len__(self) -> int:
        return len(self._d)


class Pusher:
    """One per process: the rings and, when a collect URL is set, the posting thread."""

    def __init__(self, *, plugin: str, plugin_version: str, engine: str, engine_version: str,
                 url: str | None = None, token: str | None = None, logger: str = "tandem",
                 env=None, clock=time.monotonic, post=None, start: bool = True):
        env = os.environ if env is None else env
        self.plugin, self.plugin_version = plugin, plugin_version
        self.engine, self.engine_version = engine, engine_version
        self.url = url if url is not None else (env.get("TANDEM_COLLECT_URL") or None)
        self.token = token if token is not None else (env.get("TANDEM_COLLECT_TOKEN") or None)
        self.log = logging.getLogger(logger)
        self.instance = uuid.uuid4().hex[:12]
        self.pid = os.getpid()
        self.requests = Ring(REQUESTS_MAX)
        self.steps = Ring(STEPS_MAX)
        self.sent = {"requests": 0, "steps": 0}
        self.stats = {"posts": 0, "failures": 0, "refused": 0, "errors": 0}
        self.clock = clock
        self._post = post or self._http_post
        self._last_log = -1e9
        self._last_post = -1e9
        self._wait_until = 0.0
        self._backoff = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if start and self.url:
            self.start()

    # ----------------------------------------------------------------- the engine's side
    def add_request(self, item: dict) -> None:
        self.requests.add(item)

    def add_step(self, item: dict) -> None:
        self.steps.add(item)

    # ----------------------------------------------------------------- reading
    def batch(self, after_requests: int, after_steps: int, limit: int = BATCH_MAX) -> dict:
        """The wire shape, for the poster and for the pull route alike."""
        return {"plugin": self.plugin, "plugin_version": self.plugin_version,
                "engine": self.engine, "engine_version": self.engine_version,
                "instance": self.instance, "pid": self.pid, "sent_at": time.time(),
                "dropped": {"requests": self.requests.dropped, "steps": self.steps.dropped},
                "requests": self.requests.after(after_requests, limit),
                "steps": self.steps.after(after_steps, limit)}

    # ----------------------------------------------------------------- the poster
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="tandem-collect", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(POST_EVERY_S):
            try:
                self.tick()
            except Exception as exc:                               # noqa: BLE001
                self.stats["errors"] += 1
                self._say(f"tandem: the poster failed: {type(exc).__name__}: {exc}")

    def tick(self) -> bool:
        """One round: post what is new (or a heartbeat). True when a batch went out."""
        now = self.clock()
        if not self.url or now < self._wait_until:
            return False
        b = self.batch(self.sent["requests"], self.sent["steps"])
        if not b["requests"] and not b["steps"] and now - self._last_post < HEARTBEAT_S:
            return False
        status = self._post(self.url, self.token, json.dumps(b, separators=(",", ":")).encode())
        if 200 <= status < 300:
            if b["requests"]:
                self.sent["requests"] = b["requests"][-1]["seq"]
            if b["steps"]:
                self.sent["steps"] = b["steps"][-1]["seq"]
            self.stats["posts"] += 1
            self._last_post = now
            self._backoff = 0.0
            return True
        if status in (401, 403):
            self.stats["refused"] += 1
            self._wait_until = now + REFUSED_WAIT_S
            self._say(f"tandem: the proxy refused the collect token (HTTP {status}); "
                      f"check TANDEM_COLLECT_TOKEN. Trying again in {REFUSED_WAIT_S:.0f} s")
            return False
        self.stats["failures"] += 1
        self._backoff = min(BACKOFF_MAX_S, (self._backoff * 2) or 1.0)
        self._wait_until = now + self._backoff
        self._say(f"tandem: could not post to the proxy (HTTP {status or 'no answer'}); "
                  f"the ring keeps {len(self.requests)} requests, trying again in "
                  f"{self._backoff:.0f} s")
        return False

    def _say(self, msg: str) -> None:
        now = self.clock()
        if now - self._last_log >= LOG_EVERY_S:
            self._last_log = now
            self.log.warning(msg)

    @staticmethod
    def _http_post(url: str, token: str | None, data: bytes) -> int:
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                r.read()
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except (OSError, ValueError):
            return 0


_SHARED: dict = {}
_SHARED_LOCK = threading.Lock()


def shared(**kw) -> Pusher:
    """The process's one Pusher, made at the first call. A forked child (a new pid) gets its own:
    the parent's thread does not survive a fork."""
    pid = os.getpid()
    with _SHARED_LOCK:
        p = _SHARED.get("p")
        if p is None or p.pid != pid:
            p = _SHARED["p"] = Pusher(**kw)
        return p


def num(x):
    """An int or float as it is, anything else (None, a bool, a renamed field's absence) None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return x


def ms(seconds):
    """Seconds to milliseconds, rounded to 0.01 ms; None stays None, a negative span is None."""
    s = num(seconds)
    if s is None or s < 0:
        return None
    return round(s * 1e3, 2)
