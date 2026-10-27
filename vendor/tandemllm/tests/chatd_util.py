"""Helpers for the chatd tests: a chatd on a free port in front of tests/fake_backend.py, a client
with the forward key, an SSE reader, and the client's reducer (dashboard/src/chat/store.ts) ported
for the event types phase 1 sends, so a test can check snapshot + events = the stored message.

CPU only, standard library only; nothing here touches a real engine.
"""

from __future__ import annotations

import http.client
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import fake_backend  # noqa: E402

# The unit tests never reach a container engine. The Spark has Docker with the sandbox image (the
# :4180 chat), and there a Chat found it: `bash` was offered, the tool order changed and the
# capabilities said the sandbox was installed (the box-only failures of 2026-10-12 in test_chatd,
# test_chatd_agent and test_chat_forward). Tests that want a sandbox install `FakeBoxes`; the real
# one is tests/test_sandbox_box.py, which binds the socket before this module is imported.
if os.environ.get("QSE_SANDBOX_BOX") != "1":
    os.environ.setdefault("QSE_SANDBOX_SOCKET",
                          os.path.join(tempfile.gettempdir(), "qse-tests-no-sandbox.sock"))

# the tests' sandbox limit is 1 GiB, whatever the default (their messages name it). Both pins are
# read when chatd/sandbox.py is imported; under pytest tests/conftest.py sets them first
os.environ.setdefault("QSE_SANDBOX_MEM_GB", "1")

from chatd.http import make_server, serve_in_thread  # noqa: E402
from chatd.service import Service  # noqa: E402

KEY = "k" * 64


class Chat:
    """A chatd (service + HTTP) on 127.0.0.1:<free port>, the fake engine behind it."""

    def __init__(self, tps: float = 400.0, engine_url: str | None = None, data: str | None = None,
                 retry_s: float = 20.0, max_runs: int = 2, titles: bool = False, fake=None):
        self.owned_fake = fake is None and engine_url is None
        if fake is not None:
            self.fake_srv, self.fake_port = fake
        elif engine_url is None:
            self.fake_srv, self.fake_port = fake_backend.start(models=["fake-a"], tps=tps)
        else:
            self.fake_srv, self.fake_port = None, None
        self.engine_url = engine_url or f"http://127.0.0.1:{self.fake_port}"
        self.owned_data = data is None
        self.data = data or tempfile.mkdtemp(prefix="chatd-test-")
        self.lines: list[str] = []
        self.svc = Service(self.data, self.engine_url, max_runs=max_runs, retry_window_s=retry_s,
                           titles=titles, log=self.lines.append)
        self.srv = make_server(self.svc, "127.0.0.1", 0, KEY)
        self.port = self.srv.server_address[1]
        self.svc.start()
        serve_in_thread(self.srv)

    @property
    def fake(self):
        return self.fake_srv.fake if self.fake_srv is not None else None

    def last_body(self) -> dict:
        """The last completion body the fake engine got (it also records chatd's cache peeks,
        whose bodies name the `session`)."""
        return [b for b in self.fake.bodies if "model" in b and "session" not in b][-1]

    def close(self, keep_data: bool = False) -> None:
        self.srv.shutdown()
        self.svc.shutdown()
        self.srv.server_close()
        if self.owned_fake and self.fake_srv is not None:
            self.fake_srv.shutdown()
        if self.owned_data and not keep_data:
            shutil.rmtree(self.data, ignore_errors=True)

    # ------------------------------------------------------------------ requests
    def call(self, method: str, path: str, body=None, headers: dict | None = None,
             key: str | None = KEY, raw: bytes | None = None) -> tuple[int, object, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac "
             "OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"}
        if key:
            h["X-Tandem-Forward"] = key
        h.update(headers or {})
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        conn.request(method, "/v1/chat-app" + path, body=data, headers=h)
        r = conn.getresponse()
        text = r.read()
        conn.close()
        try:
            obj = json.loads(text) if text else None
        except ValueError:
            obj = text
        return r.status, obj, dict(r.getheaders())

    def ok(self, method: str, path: str, body=None, **kw):
        st, obj, _ = self.call(method, path, body, **kw)
        assert st in (200, 202, 204), (method, path, st, obj)
        return obj

    def new(self) -> str:
        return self.ok("POST", "/conversations", {})["id"]

    def send(self, cid: str, text: str, cmid: str | None = None) -> dict:
        return self.ok("POST", f"/conversations/{cid}/messages",
                       {"client_msg_id": cmid or f"cm-{time.monotonic_ns()}", "text": text})

    def snap(self, cid: str) -> dict:
        return self.ok("GET", f"/conversations/{cid}")

    def wait(self, pred, timeout: float = 15.0, step: float = 0.05):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            v = pred()
            if v:
                return v
            time.sleep(step)
        raise AssertionError("timed out waiting")

    def wait_idle(self, cid: str, timeout: float = 15.0) -> dict:
        def done():
            s = self.snap(cid)
            return s if s["run"] is None and s["messages"] and \
                s["messages"][-1]["status"] != "streaming" else None
        return self.wait(done, timeout)

    def events(self, cid: str, after: str | None = None, last_event_id: str | None = None):
        return SseReader(self.port, f"/v1/chat-app/conversations/{cid}/events"
                         + (f"?after={after}" if after is not None else ""), last_event_id)


class SseReader:
    """An SSE connection read on a thread: `frames` gets (id, event, data) as they arrive."""

    def __init__(self, port: int, path: str, last_event_id: str | None = None):
        self.frames: queue.Queue = queue.Queue()
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        h = {"X-Tandem-Forward": KEY, "Accept": "text/event-stream"}
        if last_event_id is not None:
            h["Last-Event-ID"] = last_event_id
        self.conn.request("GET", path, headers=h)
        self.resp = self.conn.getresponse()
        self.status = self.resp.status
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self) -> None:
        fid, ev, data = None, None, []
        try:
            while True:
                raw = self.resp.readline()
                if not raw:
                    break
                line = raw.decode().rstrip("\n")
                if line == "":
                    if ev or data:
                        self.frames.put((fid, ev, "\n".join(data), time.monotonic()))
                    fid, ev, data = None, None, []
                elif line.startswith(":"):
                    continue
                elif line.startswith("id: "):
                    fid = line[4:]
                elif line.startswith("event: "):
                    ev = line[7:]
                elif line.startswith("data: "):
                    data.append(line[6:])
        except (OSError, ValueError, http.client.HTTPException):
            pass
        self.frames.put(None)

    def until(self, pred, timeout: float = 15.0) -> list:
        """Frames up to and including the first for which `pred(frame)` is true."""
        out = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            try:
                f = self.frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if f is None:
                return out
            out.append(f)
            if pred(f):
                return out
        raise AssertionError(f"timed out; got {[x[1] for x in out][-10:]}")

    def close(self) -> None:
        try:
            self.conn.sock.shutdown(2)
        except (OSError, AttributeError):
            pass
        self.conn.close()


def is_done(f) -> bool:
    return f[1] in ("run.done", "run.error")


# ---------------------------------------------------------------------- the client's reducer
def apply(c: dict, ev: dict) -> dict:
    """dashboard/src/chat/store.ts `applyEvent`, for the events phase 1 sends."""
    c = json.loads(json.dumps(c))
    t = ev["type"]

    def msg(mid):
        return next((m for m in c["messages"] if m["id"] == mid), None)
    if t == "run.started":
        c["run"], c["status"] = ev["run"], "running"
    elif t == "run.done":
        m = msg(ev["message_id"])
        if m:
            m["status"], m["stats"] = ev["status"], ev["stats"]
            for p in m["parts"]:
                if p["type"] == "thinking":
                    p["done"] = True
        c["run"] = None
        c["head_id"] = ev["message_id"]
    elif t == "run.error":
        m = msg(ev["message_id"])
        if m:
            m["status"] = "error"
            m["parts"].append({"type": "error", "message": ev["message"],
                               "retryable": ev["retryable"]})
        c["run"] = None
    elif t == "message.created":
        if not msg(ev["message"]["id"]):
            c["messages"].append(ev["message"])
        c["head_id"] = ev["message"]["id"]
    elif t == "part.added":
        msg(ev["message_id"])["parts"].append(ev["part"])
    elif t == "part.delta":
        p = msg(ev["message_id"])["parts"][ev["index"]]
        p["text"] = p.get("text", "") + ev["delta"]
    elif t == "part.done":
        p = msg(ev["message_id"])["parts"][ev["index"]]
        if p["type"] == "thinking":
            p["done"] = True
            if ev.get("tokens") is not None:
                p["tokens"] = ev["tokens"]
    elif t == "queue.changed":
        c["queue"] = ev["queue"]
    elif t == "conversation.updated":
        c.update(ev["patch"])
    elif t == "ui.changed":
        c["ui"] = ev["ui"]
    elif t == "settings.changed":
        c["settings"] = ev["settings"]
    return c


def text_of(m: dict, kind: str = "text") -> str:
    return "".join(p.get("text", "") for p in m["parts"] if p["type"] == kind)


def run_all(globs: dict) -> None:
    passed = 0
    for name, fn in sorted(globs.items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
