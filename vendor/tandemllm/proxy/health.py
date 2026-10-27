"""Each backend's health and model list, polled in the background.

Every 15 s for a local backend and every 60 s for a cloud API, one `GET <base>/models` with the
connect timeout and 10 s for the answer: up or down, down since when, the last error, the latency.
The model list is refreshed from the same answer every 10 minutes, and at once on "test
connection" or when a backend is added or changed. A cloud API's list costs nothing to read; the
proxy never fetches a price list (no silent network calls beyond these).

`probe(backend, key)` is the same request for a backend that is not saved yet: the GUI's "Test
connection" shows its latency and its models before the row is written.
"""

from __future__ import annotations

import json
import threading
import time

from proxy import adapters as adapters_mod
from proxy import http1

LOCAL_EVERY_S = 15.0
CLOUD_EVERY_S = 60.0
MODELS_EVERY_S = 600.0
ANSWER_TIMEOUT_S = 10.0


def probe(backend: dict, key: str | None, log=None) -> dict:
    """One `GET /models`: {ok, status, latency_ms, models, error}. Never raises."""
    adapter = adapters_mod.make(backend, log=log)
    pool = http1.Pool(backend["url"], float(backend["timeouts"]["connect"]))
    t0 = time.perf_counter()
    out = {"ok": False, "status": None, "latency_ms": None, "models": None, "error": None}
    conn = None
    try:
        conn = pool.get()
        conn.send("GET", pool.path("/v1" + adapter.models_path),
                  [h for h in adapter.headers(key) if h[0] != "Content-Type"], b"")
        status, _, uh = conn.read_head(ANSWER_TIMEOUT_S)
        data = conn.read_all(ANSWER_TIMEOUT_S, limit=16 * 1024 * 1024)
        out["latency_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        out["status"] = status
        if status == 200:
            try:
                out["models"] = adapter.parse_models(json.loads(data))
                out["ok"] = True
            except ValueError:
                out["error"] = "the /models answer is not JSON"
        elif status in (401, 403):
            out["error"] = f"HTTP {status}: the key was refused"
        else:
            out["error"] = f"HTTP {status}"
    except http1.Timeout as t:
        out["error"] = f"timeout ({t.which})"
    except (http1.UpstreamError, http1.Closed) as exc:
        out["error"] = str(exc)
    except Exception as exc:                                       # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()
        pool.close()
    return out


class Health:
    """The poller: one thread, every backend in turn."""

    def __init__(self, service, *, local_every=LOCAL_EVERY_S, cloud_every=CLOUD_EVERY_S,
                 models_every=MODELS_EVERY_S, clock=time.time):
        self.svc = service
        self.local_every, self.cloud_every = local_every, cloud_every
        self.models_every = models_every
        self.clock = clock
        self.state: dict[str, dict] = {}
        self._due: dict[str, float] = {}
        self._models_due: dict[str, float] = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> "Health":
        self._thread = threading.Thread(target=self._run, name="proxy-health", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def kick(self, bid: str | None = None) -> None:
        """Poll (and rediscover) now: a backend was added or changed."""
        with self._lock:
            for k in ([bid] if bid else list(self._due)):
                self._due[k] = 0.0
                self._models_due[k] = 0.0
        self._wake.set()

    def view(self, bid: str) -> dict:
        s = self.state.get(bid) or {}
        return {"state": s.get("state", "unknown"), "since": s.get("since"),
                "last_check": s.get("last_check"), "last_error": s.get("last_error"),
                "latency_ms": s.get("latency_ms"), "models_at": s.get("models_at")}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:                               # noqa: BLE001
                self.svc.log(f"[proxy] health poll failed: {type(exc).__name__}: {exc}")
            self._wake.wait(1.0)
            self._wake.clear()

    def tick(self) -> None:
        now = self.clock()
        for b in self.svc.cfg.backends():
            bid = b["id"]
            if not b["enabled"]:
                self.state[bid] = {"state": "disabled", "since": None}
                continue
            with self._lock:
                due = self._due.get(bid, 0.0)
            if now < due:
                continue
            with self._lock:
                self._due[bid] = now + (self.cloud_every if b["cloud"] else self.local_every)
            self.check(b)

    def check(self, b: dict) -> dict:
        bid = b["id"]
        try:
            key = self.svc.cfg.key_for(bid)
        except Exception as exc:                                   # noqa: BLE001
            r = {"ok": False, "status": None, "latency_ms": None, "models": None,
                 "error": f"key unreadable: {exc}"}
        else:
            r = probe(b, key, log=self.svc.log)
        now = self.clock()
        prev = self.state.get(bid) or {}
        st = "up" if r["ok"] else "down"
        since = prev.get("since") if prev.get("state") == st else now
        cur = {"state": st, "since": since, "last_check": now,
               "last_error": None if r["ok"] else r["error"], "latency_ms": r["latency_ms"],
               "models_at": prev.get("models_at")}
        if st != prev.get("state"):
            self.svc.log(f"[proxy] backend {bid} is {st}"
                         + (f": {r['error']}" if r["error"] else ""))
        with self._lock:
            mdue = self._models_due.get(bid, 0.0)
        if now >= mdue or b["models_mode"] == "list":
            found = r["models"] if r["ok"] else None
            new = self.svc.cfg.sync_models(bid, found, b["models"], b["models_mode"])
            if new and hasattr(self.svc, "new_models"):
                self.svc.new_models(bid, new)
            if found is not None:
                cur["models_at"] = now
                with self._lock:
                    self._models_due[bid] = now + self.models_every
        self.state[bid] = cur
        return cur
