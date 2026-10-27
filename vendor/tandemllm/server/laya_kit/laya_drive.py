#!/usr/bin/env python3
"""Laya plays the software under test: TandemLLM's playtest driver (laya_drive.py in the session's scratch directory).

Each step the driver reads the program's state (a small JSON object), asks Laya through TandemLLM
which action from a fixed set comes next, applies it as real input and logs both. A death, a win,
a stuck spot or a step budget ends what it says; `report` turns the log into a short text for you.

  proc     python3 laya_drive.py proc "python3 game.py --headless" --actions ACTIONS
           the program prints one JSON state per line and reads one action per line
           ("reset" starts the next episode; its exit ends one, and it is started again)
  cmd      python3 laya_drive.py cmd --state "CMD" --act "CMD {action}" [--reset "CMD"]
           a command prints the state, another applies the action (a CLI, an API with curl)
  browser  python3 laya_drive.py browser URL --keys run=ArrowRight,jump=Space,wait=
           a Chrome it starts (or --cdp http://127.0.0.1:9222) over the DevTools protocol, real
           key and mouse input; the page defines window.tandem_state() (and tandem_reset())
  page     python3 laya_drive.py page URL [--step click:#start --step wait:2000 --step read:EXPR]
           the check of a page or web app, no Laya: the same headless Chrome loads the page, runs
           the steps with real input (click:SELECTOR or click:text=Start, key:Space or key:ArrowRight*400,
           type:#name=Ada, wait:MS, read:JS, expect:JS, shot:NAME), screenshots it at 1280 and
           390 px, closes Chrome and prints one JSON report: console errors, uncaught exceptions,
           failed loads, the step results, horizontal overflow, the visible text, each
           screenshot's look in numbers (brightness, the share of its most common colour,
           saturation); exit 0 on PASS
  report   python3 laya_drive.py report [drive-log.jsonl beside the kit]
  serve-log  python3 laya_drive.py serve-log [--port 8765]: the in-page kit's log sink

ACTIONS: `run=move right,jump=jump straight up,wait=do nothing` (label=what it does) or
`run,jump,wait`. A state may carry its own choices, for a UI: "actions": ["save", "cancel"] or
"controls": [{"id": "save", "label": "Save", "x": 120, "y": 40} | {..., "selector": "#save"}].
The state's fields the driver reads: dead / over / lost (an episode lost), won / win, lives (a
drop is a death), x and y (or player.x/y, or pos) for stuck spots and the report, cause (why it
died). Everything else goes to Laya as it is: keep it under ~1,000 characters, what matters for the
next move (nearby obstacles, gaps, enemies relative to the player, velocity, lives, score).

Python: `from laya_drive import Driver; Driver(actions).run(get_state, apply, reset)`.
Stdlib only. TANDEM_URL (default http://127.0.0.1:8000) is where Laya is.
"""

from __future__ import annotations

import argparse
import base64
import collections
import ctypes
import glob
import hashlib
import http.client
import json
import re
import os
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zlib

# SRV-171: the log beside the kit (the session's scratch directory, which the packages name
# `{tandem_scratch}`), never inside the user's project; `--log` names another file
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive-log.jsonl")
QUESTION = "Which action should the player take now to stay alive and make progress?"
STATE_CHARS = 1000


# ---------------------------------------------------------------------------- Laya
class Laya:
    """One persistent connection to /v1/laya/ask; a driver session of its own."""

    def __init__(self, url: str | None = None, session: str | None = None, timeout: float = 30.0):
        self.url = urllib.parse.urlsplit((url or os.environ.get("TANDEM_URL")
                                          or "http://127.0.0.1:8000").rstrip("/"))
        self.session = session or f"drive-{os.getpid()}-{int(time.time())}"
        self.timeout = timeout
        self.conn = None

    def _post(self, path: str, body: dict) -> tuple[int, dict, dict]:
        data = json.dumps(body).encode()
        for attempt in (0, 1):
            if self.conn is None:
                cls = (http.client.HTTPSConnection if self.url.scheme == "https"
                       else http.client.HTTPConnection)
                self.conn = cls(self.url.netloc, timeout=self.timeout)
            try:
                self.conn.request("POST", self.url.path + path, data,
                                  {"Content-Type": "application/json"})
                r = self.conn.getresponse()
                raw = r.read()
                try:
                    out = json.loads(raw or b"{}")
                except ValueError:
                    out = {"error": {"type": "bad_json", "message": raw[:200].decode("utf-8", "replace")}}
                return r.status, out, dict(r.getheaders())
            except (OSError, http.client.HTTPException):
                self.conn.close()
                self.conn = None
                if attempt:
                    raise
        raise OSError("unreachable")

    def pick(self, question: str, choices: dict, text: str, deadline_ms: int = 15000) -> dict:
        """{label, p, sure, ms} or {error, retry}."""
        t0 = time.perf_counter()
        try:
            st, out, hdr = self._post("/v1/laya/ask", {
                "question": question, "choices": choices, "text": text,
                "session": self.session, "deadline_ms": deadline_ms})
        except OSError as exc:
            return {"error": f"unreachable: {exc}", "ms": round((time.perf_counter() - t0) * 1e3, 1)}
        ms = round((time.perf_counter() - t0) * 1e3, 1)
        if st != 200:
            err = (out.get("error") or {}) if isinstance(out, dict) else {}
            return {"error": err.get("type") or str(st), "message": err.get("message"),
                    "retry": hdr.get("Retry-After"), "ms": ms}
        return {"label": out["label"], "p": out["p"], "sure": out["sure"], "ms": ms,
                "laya_ms": out.get("ms")}

    def end(self, summary: dict) -> dict | None:
        try:
            st, out, _ = self._post("/v1/laya/drive/end", {"session": self.session,
                                                           "summary": summary})
            return out if st == 200 else None
        except OSError:
            return None


# ---------------------------------------------------------------------------- the loop
def parse_actions(spec) -> dict:
    """`run=move right,jump` or a list or a dict -> {label: meaning}."""
    if isinstance(spec, dict):
        return {str(k): str(v or k) for k, v in spec.items()}
    if isinstance(spec, str):
        spec = [s for s in spec.split(",") if s.strip()]
    out = {}
    for a in spec:
        k, _, v = str(a).partition("=")
        out[k.strip()] = v.strip() or k.strip()
    return out


def _num(v):
    return round(v, 1) if isinstance(v, float) else v


def compact(state, limit: int = STATE_CHARS) -> str:
    """The state as short JSON: floats to one decimal, lists cut until it fits."""
    def walk(v, cut):
        if isinstance(v, dict):
            return {k: walk(x, cut) for k, x in v.items() if x is not None}
        if isinstance(v, list):
            return [walk(x, cut) for x in v[:cut]]
        return _num(v)
    for cut in (64, 16, 8, 4, 2, 1):
        s = json.dumps(walk(state, cut), separators=(",", ":"))
        if len(s) <= limit:
            return s
    return s[:limit]


def position(state):
    """(x, y) from x/y, player.x/y or pos, else None."""
    if not isinstance(state, dict):
        return None
    for src in (state, state.get("player") if isinstance(state.get("player"), dict) else None):
        if src and isinstance(src.get("x"), (int, float)):
            return (round(src["x"], 1), round(src["y"], 1) if isinstance(src.get("y"), (int, float)) else None)
    pos = state.get("pos")
    if isinstance(pos, (list, tuple)) and pos and isinstance(pos[0], (int, float)):
        return tuple(round(p, 1) for p in pos[:2])
    return None


def lost(state) -> bool:
    return isinstance(state, dict) and any(state.get(k) is True for k in ("dead", "over", "lost", "game_over"))


def won(state) -> bool:
    return isinstance(state, dict) and any(state.get(k) is True for k in ("won", "win"))


def choices_of(state, fixed: dict) -> dict:
    """The step's choices: the state's own (`actions` / `controls`), else the fixed set."""
    if isinstance(state, dict):
        if isinstance(state.get("controls"), list) and state["controls"]:
            return {str(c.get("id")): str(c.get("label") or c.get("id"))[:200]
                    for c in state["controls"] if isinstance(c, dict) and c.get("id")}
        if isinstance(state.get("actions"), list) and state["actions"]:
            return {str(a): fixed.get(str(a), str(a)) for a in state["actions"]}
    return fixed


class Driver:
    """The decision loop. `get_state()` -> dict; `apply(action, state)`; `reset()` -> None.
    `every_ms` 0: step as fast as Laya answers (a turn-based program); > 0: one step per tick
    (a real-time one). `fallback` is applied when Laya cannot answer (busy, rate, unreachable)."""

    def __init__(self, actions, question: str = QUESTION, url: str | None = None,
                 session: str | None = None, log: str = LOG, every_ms: int = 0,
                 steps: int = 600, episodes: int = 3, fallback: str | None = None,
                 stuck_steps: int = 20, laya: Laya | None = None, label: str = ""):
        self.actions = parse_actions(actions)
        self.question, self.every = question, every_ms / 1000.0
        self.steps, self.episodes, self.stuck_steps = steps, episodes, stuck_steps
        self.fallback = fallback if fallback is not None else (
            "wait" if "wait" in self.actions else next(iter(self.actions)))
        self.laya = laya or Laya(url, session)
        # a real-time program does not wait for a busy engine: two ticks, then the fallback
        self.deadline_ms = max(250, 2 * every_ms) if every_ms else 15000
        self.log_path, self.label = log, label
        self.counts = collections.Counter()

    def _write(self, row: dict) -> None:
        self._f.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._f.flush()

    def run(self, get_state, apply, reset=None) -> dict:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._f = open(self.log_path, "a")
        t0 = time.monotonic()
        stamp = lambda: int((time.monotonic() - t0) * 1000)          # noqa: E731
        self._write({"run": self.laya.session, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "label": self.label, "actions": self.actions, "every_ms": int(self.every * 1000),
                     "steps": self.steps, "episodes": self.episodes,
                     "question": hashlib.sha256(self.question.encode()).hexdigest()[:12]})
        c = self.counts
        try:
            for ep in range(1, self.episodes + 1):
                if ep > 1 and reset is not None:
                    reset()
                c["episodes"] += 1
                state = get_state()
                lives = state.get("lives") if isinstance(state, dict) else None
                spot, same, flagged, last_pos, step = position(state), 0, set(), None, 0
                end = "death" if lost(state) else "win" if won(state) else None
                while end is None and step < self.steps:
                    step += 1
                    tick = time.monotonic()
                    ch = choices_of(state, self.actions)
                    v = self.laya.pick(self.question, ch, compact(state), self.deadline_ms)
                    if "error" in v:
                        a, why = (self.fallback if self.fallback in ch else next(iter(ch))), v["error"]
                        c["fallbacks"] += 1
                        c["busy"] += v["error"] == "busy"
                    else:
                        a, why = v["label"], None
                    c["steps"] += 1
                    c[f"a:{a}"] += 1
                    row = {"ep": ep, "step": step, "t": stamp(), "s": compact(state, 600), "a": a,
                           "p": v.get("p"), "sure": v.get("sure"), "ms": v.get("ms")}
                    if why:
                        row["why"] = why
                    self._write(row)
                    apply(a, state)
                    if self.every:
                        time.sleep(max(0.0, self.every - (time.monotonic() - tick)))
                    # where a death happens: the place of the decision before it (a game puts
                    # the player back somewhere else at once)
                    last_pos = position(state)
                    state = get_state()
                    if lost(state):
                        end = "death"
                        break
                    if won(state):
                        end = "win"
                        break
                    nl = state.get("lives") if isinstance(state, dict) else None
                    if isinstance(lives, (int, float)) and isinstance(nl, (int, float)) and nl < lives:
                        c["deaths"] += 1
                        self._write({"ep": ep, "step": step, "t": stamp(), "event": "death",
                                     "s": compact(state, 600), "pos": last_pos,
                                     "cause": state.get("cause"), "lives": nl})
                    lives = nl
                    pos = position(state)
                    if pos is not None and pos == spot:
                        same += 1
                        if same == self.stuck_steps and pos not in flagged:
                            flagged.add(pos)
                            c["stuck"] += 1
                            self._write({"ep": ep, "step": step, "t": stamp(), "event": "stuck",
                                         "pos": pos, "steps": same, "s": compact(state, 600)})
                    else:
                        spot, same = pos, 0
                end = end or "timeout"
                c[{"death": "deaths", "win": "wins", "timeout": "timeouts"}[end]] += 1
                self._write({"ep": ep, "step": step, "t": stamp(), "event": end,
                             "s": compact(state, 600),
                             "pos": last_pos if end == "death" and last_pos else position(state),
                             "cause": state.get("cause") if isinstance(state, dict) else None})
        finally:
            summary = {k: v for k, v in c.items() if ":" not in k}
            srv = self.laya.end(summary)
            self._write({"end": self.laya.session, "t": stamp(), "summary": summary,
                         "actions": {k[2:]: v for k, v in c.items() if k.startswith("a:")},
                         "laya": {k: srv.get(k) for k in ("ms_p50", "ms_p95", "unsure", "errors")}
                         if srv else None})
            self._f.close()
        return dict(self.counts)


# ---------------------------------------------------------------------------- children
def _die_with_parent() -> None:
    """In the child before exec (Linux): SIGTERM when the driver that started it dies, even by
    SIGKILL (a tool call's timeout). 2026-10-04: 42 Chrome processes of four killed drives had run
    for a day on the box, their driver gone and nothing left to close them."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG
    except Exception:                                                  # noqa: BLE001
        pass


def _spawn_kw() -> dict:
    """A child in its own process group (closed with everything it started) that dies with us."""
    kw = {"start_new_session": True}
    if sys.platform.startswith("linux"):
        kw["preexec_fn"] = _die_with_parent
    return kw


def _end_group(p, grace: float = 5.0) -> None:
    """SIGTERM to the child's whole group (its own, `_spawn_kw`), SIGKILL to what is left after
    `grace` s; the group is ended even when its first process is gone already."""
    if p is None:
        return
    pg = p.pid

    def alive() -> bool:
        p.poll()                           # reaps the first process once it exits
        try:
            os.killpg(pg, 0)
            return True
        except OSError:
            return False
    try:
        os.killpg(pg, signal.SIGTERM)
    except OSError:
        return
    end = time.monotonic() + grace
    while time.monotonic() < end:
        if not alive():
            return
        time.sleep(0.05)
    try:
        os.killpg(pg, signal.SIGKILL)
    except OSError:
        pass
    try:
        p.wait(2)
    except subprocess.TimeoutExpired:
        pass


# ---------------------------------------------------------------------------- proc and cmd
class Proc:
    """A program speaking JSON lines: a state line out, an action line in."""

    def __init__(self, cmd: str):
        self.cmd = cmd
        self.p = None

    def _start(self):
        self.p = subprocess.Popen(self.cmd, shell=True, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1, **_spawn_kw())

    def _read(self) -> dict:
        while True:
            line = self.p.stdout.readline()
            if not line:
                return {"over": True, "cause": f"the program exited ({self.p.wait()})"}
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    continue

    def state(self) -> dict:
        if self.p is None:
            self._start()
        return self._read()

    def apply(self, action: str, _state=None) -> None:
        try:
            self.p.stdin.write(action + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def reset(self) -> None:
        if self.p is not None and self.p.poll() is None:
            self.apply("reset")
        else:
            self._start()

    def close(self) -> None:
        if self.p is not None:
            _end_group(self.p, grace=2.0)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout


def cmd_state(cmd: str):
    def get():
        out = sh(cmd).strip()
        try:
            return json.loads(out[out.find("{"):]) if "{" in out else {"output": out[-STATE_CHARS:]}
        except ValueError:
            return {"output": out[-STATE_CHARS:]}
    return get


# ---------------------------------------------------------------------------- browser (CDP)
KEYS = {"ArrowLeft": 37, "ArrowUp": 38, "ArrowRight": 39, "ArrowDown": 40, "Space": 32,
        "Enter": 13, "Escape": 27, "Shift": 16, "Control": 17, "Tab": 9, "Backspace": 8}


def key_event(name: str) -> dict:
    """CDP Input.dispatchKeyEvent fields for `ArrowRight`, `Space`, `a`, `KeyA`, `1`."""
    if name == "Space" or name == " ":
        return {"key": " ", "code": "Space", "windowsVirtualKeyCode": 32, "text": " "}
    if name in KEYS:
        return {"key": name, "code": name if name not in ("Shift", "Control") else name + "Left",
                "windowsVirtualKeyCode": KEYS[name]}
    ch = name[-1] if name.startswith(("Key", "Digit")) else name
    ch = ch.lower() if len(ch) == 1 else ch
    code = ("Digit" + ch) if ch.isdigit() else ("Key" + ch.upper())
    return {"key": ch, "code": code, "windowsVirtualKeyCode": ord(ch.upper()), "text": ch}


def find_chrome() -> str | None:
    cands = [os.environ.get("CHROME"),
             "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
             "/Applications/Chromium.app/Contents/MacOS/Chromium",
             *(shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "chrome")),
             *sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome"))),
             *sorted(glob.glob(os.path.expanduser(
                 "~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/*")))]
    return next((c for c in cands if c and os.access(c, os.X_OK)), None)


class CDP:
    """The DevTools protocol over one websocket, stdlib only: evaluate, keys, mouse."""

    def __init__(self, ws_url: str):
        u = urllib.parse.urlsplit(ws_url)
        self.sock = socket.create_connection((u.hostname, u.port or 80), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                           f"Sec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise OSError("the DevTools websocket closed during the handshake")
            head += chunk
        first = head.split(b"\r\n")[0]
        if b" 101 " not in first:
            raise OSError(f"DevTools handshake: {first!r}")
        self.buf = head.split(b"\r\n\r\n", 1)[1]
        self.n = 0
        # `page` keeps the events that arrive between replies (console, exceptions, the log);
        # a playtest keeps none, so a chatty page cannot grow this for an hour
        self.keep_events = False
        self.events: list[dict] = []

    def _send(self, text: str) -> None:
        data = text.encode()
        mask = os.urandom(4)
        n = len(data)
        hdr = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else
                               bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536 else
                               bytes([0x80 | 127]) + struct.pack(">Q", n))
        self.sock.sendall(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _need(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise OSError("the DevTools websocket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _recv(self) -> str:
        parts = []
        while True:
            b0, b1 = self._need(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._need(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._need(8))[0]
            data = self._need(n)
            if b0 & 0x0F == 0x8:
                raise OSError("the DevTools websocket closed")
            if b0 & 0x0F in (0x1, 0x0):
                parts.append(data)
                if b0 & 0x80:
                    return b"".join(parts).decode("utf-8", "replace")

    def call(self, method: str, **params):
        self.n += 1
        me = self.n
        self._send(json.dumps({"id": me, "method": method, "params": params}))
        while True:
            msg = json.loads(self._recv())
            if msg.get("id") == me:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result") or {}
            if self.keep_events and "method" in msg and len(self.events) < 2000:
                self.events.append(msg)

    def evaluate(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(f"page: {r['exceptionDetails'].get('text')} "
                               f"{(r['exceptionDetails'].get('exception') or {}).get('description', '')[:300]}")
        return (r.get("result") or {}).get("value")

    def key(self, name: str, down: bool) -> None:
        ev = key_event(name)
        if not down:
            ev.pop("text", None)
        self.call("Input.dispatchKeyEvent", type=("keyDown" if down else "keyUp"), **ev)

    def click(self, x: float, y: float) -> None:
        for t in ("mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)


class Browser:
    """A page under CDP: a Chrome this starts (headless) or one at `cdp`."""

    def __init__(self, url: str, cdp: str | None = None, headless: bool = True,
                 keys: dict | None = None, width: int = 1280, height: int = 800,
                 watch: bool = False, need_state: bool = True):
        self.proc, self.tmp, self.cdp = None, None, None
        self.watch, self.need_state = watch, need_state
        if cdp is None:
            chrome = find_chrome()
            if chrome is None:
                raise SystemExit("no Chrome or Chromium found (set CHROME=...); or start one with "
                                 "--remote-debugging-port=9222 and pass --cdp http://127.0.0.1:9222")
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            self.tmp = tempfile.mkdtemp(prefix="laya-drive-chrome-")
            self.proc = subprocess.Popen(
                [chrome, f"--remote-debugging-port={port}", f"--user-data-dir={self.tmp}",
                 "--no-first-run", "--no-default-browser-check", f"--window-size={width},{height}",
                 # Linux without unprivileged user namespaces (Ubuntu 23.10+) has no sandbox for
                 # it; Playwright starts Chromium the same way there
                 *(["--no-sandbox"] if sys.platform.startswith("linux") else []),
                 # a container's /dev/shm is 64 MB: Chrome's shared memory goes to /tmp, one
                 # renderer (a chat sandbox's memory limit counts every process)
                 *(["--disable-dev-shm-usage", "--renderer-process-limit=1"]
                   if sys.platform.startswith("linux") else []),
                 # SRV-195: without these Chrome waits ~25 s for the desktop keyring at its first
                 # network request (a page on http://127.0.0.1 took 25 s to load on the box, a
                 # file:// page 0.2 s); Playwright starts Chromium with the same two
                 "--password-store=basic", "--use-mock-keychain",
                 *(["--headless=new"] if headless else []), "about:blank"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **_spawn_kw())
            cdp = f"http://127.0.0.1:{port}"
        try:
            self._attach(cdp, url, keys, chrome if self.proc is not None else None)
        except BaseException:
            self.close()                   # a page that never came up: its Chrome goes too
            raise

    def _attach(self, cdp: str, url: str, keys, chrome) -> None:
        target = None
        for _ in range(100):
            try:
                with urllib.request.urlopen(cdp + "/json", timeout=2) as r:
                    target = next((t for t in json.load(r) if t.get("type") == "page"), None)
                if target:
                    break
            except OSError:
                pass
            if self.proc is not None and self.proc.poll() is not None:
                raise SystemExit(f"Chrome exited at its start ({self.proc.returncode}): {chrome}")
            time.sleep(0.1)
        if target is None:
            raise SystemExit(f"no page at {cdp}")
        self.cdp = CDP(target["webSocketDebuggerUrl"])
        self.keys = keys or {}
        self.held: set = set()
        if self.watch:                     # console, exceptions and failed loads from the start
            self.cdp.keep_events = True
            self.cdp.call("Runtime.enable")
            self.cdp.call("Log.enable")
        self.cdp.call("Page.enable")
        self.cdp.call("Page.navigate", url=url)
        self.wait_ready()

    def wait_ready(self, timeout: float = 15.0) -> None:
        end = time.monotonic() + timeout
        ready = ("document.readyState === 'complete'" + (
            " && typeof (window.tandem_state || window.__state) === 'function'"
            if self.need_state else ""))
        while time.monotonic() < end:
            try:
                if self.cdp.evaluate(ready):
                    return
            except RuntimeError:
                pass
            time.sleep(0.1)
        if not self.need_state:
            raise SystemExit("the page did not finish loading in 15 s")
        raise SystemExit("the page has no window.tandem_state() (or __state()) after 15 s")

    def state(self) -> dict:
        return self.cdp.evaluate("(window.tandem_state || window.__state)()") or {}

    def apply(self, action: str, state=None) -> None:
        ctl = next((c for c in (state or {}).get("controls") or ()
                    if isinstance(c, dict) and str(c.get("id")) == action), None)
        if ctl is not None:
            self.release()
            if "x" in ctl and "y" in ctl:
                return self.cdp.click(ctl["x"], ctl["y"])
            r = self.cdp.evaluate(f"(() => {{ const e = document.querySelector({json.dumps(ctl.get('selector') or '#' + action)}); "
                                  f"if (!e) return null; const b = e.getBoundingClientRect(); "
                                  f"return [b.x + b.width / 2, b.y + b.height / 2]; }})()")
            if r:
                self.cdp.click(*r)
            return
        # every step presses its keys anew: a jump on two steps is two jumps
        self.release()
        for k in [k for k in self.keys.get(action, "").split("+") if k]:
            self.cdp.key(k, True)
            self.held.add(k)

    def release(self) -> None:
        for k in self.held:
            self.cdp.key(k, False)
        self.held = set()

    def reset(self) -> None:
        self.release()
        if self.cdp.evaluate("typeof window.tandem_reset === 'function'"):
            self.cdp.evaluate("window.tandem_reset()")
        else:
            self.cdp.call("Page.reload")
            self.wait_ready()

    def close(self) -> None:
        try:
            if self.cdp is not None:
                self.release()
        except (OSError, RuntimeError, AttributeError):
            pass
        if self.proc is not None:
            _end_group(self.proc)          # Chrome and every process it started
            self.proc = None
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.tmp = None


# ---------------------------------------------------------------------------- page (the check)
#
# SRV-195 (FX2's live check, 2026-10-06): opencode wrote a pomodoro page at 176 s, found no node,
# no npx and no Chrome on its PATH, only Firefox, and spent the next seven minutes on a Marionette
# driver of its own until the run's limit. The kit already starts a headless Chrome (Playwright's,
# under ~/.cache) for playtests; `page` is the same Chrome as a one-command check: load the page,
# collect console errors, uncaught exceptions and failed loads, press real keys and click with the
# real mouse, read values, screenshot at each width, close Chrome, print one JSON report.

PAGE_TIMEOUT_S = 90


class PageTimeout(Exception):
    """The check's time limit (not an OSError: the start's retry loop swallows those)."""
_FIND = """(q => {
  let e = null;
  const label = x => (x.innerText || x.value || x.getAttribute('aria-label') || x.title || '').trim().toLowerCase();
  if (!q.startsWith('text=')) { try { e = document.querySelector(q); } catch (_) { e = null; } }
  if (!e) {
    const t = (q.startsWith('text=') ? q.slice(5) : q).trim().toLowerCase();
    const c = [...document.querySelectorAll('button, a, [role=button], [role=tab], input[type=button], input[type=submit], input[type=checkbox], input[type=radio], label, summary, select, [onclick]')];
    e = c.find(x => label(x) === t) || c.find(x => label(x).includes(t)) || null;
  }
  if (!e) return null;
  e.scrollIntoView({block: 'center', inline: 'center'});
  const b = e.getBoundingClientRect();
  return {x: b.x + b.width / 2, y: b.y + b.height / 2, visible: b.width > 0 && b.height > 0,
          disabled: !!e.disabled, tag: e.tagName.toLowerCase()};
})(%s)"""
_CONTROLS = """(() => [...document.querySelectorAll('button, a[href], [role=button], [role=tab], input, select, textarea')]
  .filter(e => { const b = e.getBoundingClientRect(); return b.width > 0 && b.height > 0; })
  .slice(0, 24).map(e => ({tag: e.tagName.toLowerCase(), id: e.id || undefined,
    label: (e.innerText || e.value || e.getAttribute('aria-label') || e.placeholder || '').trim().slice(0, 40),
    disabled: e.disabled || undefined})))()"""
# the overflow is against the width the check set: a phone-sized page that overflows makes Chrome
# widen `innerWidth` to the content, so the page would measure itself as fitting
_LAYOUT = """(w => ({width: w, scroll_width: Math.max(document.documentElement.scrollWidth,
    document.body ? document.body.scrollWidth : 0),
  text: document.body ? document.body.innerText.replace(/\\s+/g, ' ').trim().slice(0, 400) : ''}))(%d)"""


def _short(v, n: int = 300):
    t = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return t if len(t) <= n else t[:n] + "..."


#: a GLSL compiler's error line (`ERROR: 0:1768: 'vReefW' : undeclared identifier`), and the
#: marked source line three.js prints after the log (`> 1768: ...`)
_GLSL_ERROR = re.compile(r"^\s*(ERROR: \d+:\d+: .*|>\s*\d+: .*)$", re.M)


def _error_line(x: str, n: int = 240) -> str:
    """One console error as the report carries it: clipped at `n`, but a shader compile or link
    error keeps its first line and every compiler error line and marked source line, the source
    dump left out. Live (2026-10-21, Ornith's reef replay) the clip ended right after the first
    `ERROR: 0:1768:`, and the model spent 14 of its 100 calls reading three.js's WebGLProgram to
    find what the compiler had already said."""
    lines = _GLSL_ERROR.findall(x)
    if not lines:
        return _short(x, n)
    head = x.strip().split("\n")[0][:n]
    keep, seen = [], set()
    for ln in (ln.strip() for ln in lines):
        if ln not in seen:
            seen.add(ln)
            keep.append(_short(ln, 200))
    more = len(keep) - 12
    return "\n".join([head] + keep[:12] + ([f"({more} more compiler lines)"] if more > 0 else []))


def _problems(events: list[dict]) -> tuple[list[str], int]:
    """(console errors, uncaught exceptions and failed loads; the number of warnings). A missing
    favicon is not the page's fault."""
    errs, warns = [], 0
    for ev in events:
        m, p = ev.get("method"), ev.get("params") or {}
        if m == "Runtime.consoleAPICalled":
            if p.get("type") in ("error", "assert"):
                errs.append("console: " + " ".join(
                    str(a.get("value", a.get("description", ""))) for a in p.get("args") or ()))
            elif p.get("type") == "warning":
                warns += 1
        elif m == "Runtime.exceptionThrown":
            d = p.get("exceptionDetails") or {}
            # where it was thrown: the frame's script and line (1-based), so a reader can open it
            top = ((d.get("stackTrace") or {}).get("callFrames") or [{}])[0]
            url = d.get("url") or top.get("url") or ""
            line = d.get("lineNumber") if d.get("url") else top.get("lineNumber")
            at = f" ({url}:{int(line) + 1})" if url and line is not None else ""
            errs.append("exception: " + ((d.get("exception") or {}).get("description")
                                         or d.get("text") or "?").split("\n")[0] + at)
        elif m == "Log.entryAdded":
            e = p.get("entry") or {}
            if "favicon.ico" in (e.get("url") or ""):
                continue
            if e.get("level") == "error":
                errs.append(f"{e.get('source', 'log')}: {e.get('text', '')} {e.get('url', '')}".strip())
            elif e.get("level") == "warning":
                warns += 1
    # the page loads once per width (and once at the start): the same error is one line
    seen: dict = {}
    for x in errs:
        seen[x] = seen.get(x, 0) + 1
    return [_error_line(x) for x, n in seen.items()], warns


class PageCheck:
    """One page in the kit's headless Chrome: steps with real input, then a screenshot and a
    layout reading at each width."""

    def __init__(self, url: str, widths: list[int], shots: str, cdp: str | None = None):
        self.url, self.widths, self.shots = url, widths, shots
        self.b = Browser(url, cdp, headless=True, width=widths[0], height=900,
                         watch=True, need_state=False)
        self.cdp = self.b.cdp

    def size(self, width: int) -> None:
        mobile = width < 600
        self.cdp.call("Emulation.setDeviceMetricsOverride", width=width,
                      height=844 if mobile else 900, deviceScaleFactor=1, mobile=mobile)
        self.cdp.call("Emulation.setTouchEmulationEnabled", enabled=mobile)

    def settle(self, ms: int = 300) -> None:
        """Let the page run, then collect what it reported meanwhile."""
        time.sleep(ms / 1000.0)
        self.cdp.evaluate("0")

    def shot(self, name: str) -> str:
        os.makedirs(self.shots, exist_ok=True)
        data = self.cdp.call("Page.captureScreenshot", format="png").get("data") or ""
        path = os.path.join(self.shots, f"{name}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))
        return os.path.relpath(path)

    def look(self, width: int) -> dict | None:
        """The screenshot in numbers, for a reader that sees no images (an eighth of its size,
        decoded here): a day scene that renders dark or a page that is one flat colour shows."""
        try:
            h = int(self.cdp.evaluate("innerHeight") or 900)
            data = self.cdp.call("Page.captureScreenshot", format="png", clip={
                "x": 0, "y": 0, "width": width, "height": h, "scale": 0.125}).get("data") or ""
            return look_of(base64.b64decode(data))
        except (RuntimeError, ValueError, OSError, zlib.error, struct.error):
            return None

    def step(self, spec: str) -> dict:
        kind, _, arg = spec.partition(":")
        kind = kind.strip().lower()
        out: dict = {"step": spec}
        if kind == "click":
            at = self.cdp.evaluate(_FIND % json.dumps(arg.strip()))
            if not at:
                return dict(out, ok=False, error=f"no element matches {arg.strip()!r}")
            if not at["visible"]:
                return dict(out, ok=False, error="the element is not visible")
            self.settle(80)                         # the scroll lands
            at = self.cdp.evaluate(_FIND % json.dumps(arg.strip())) or at
            self.cdp.click(at["x"], at["y"])
            out.update(ok=True, disabled=at["disabled"] or None)
        elif kind == "key":
            name, _, hold = arg.partition("*")
            self.cdp.key(name.strip(), True)
            time.sleep(max(0.03, float(hold or 100) / 1000.0))
            self.cdp.key(name.strip(), False)
            out["ok"] = True
        elif kind == "type":
            sel, _, text = arg.partition("=")
            at = self.cdp.evaluate(_FIND % json.dumps(sel.strip()))
            if not at:
                return dict(out, ok=False, error=f"no element matches {sel.strip()!r}")
            self.cdp.click(at["x"], at["y"])
            self.cdp.call("Input.insertText", text=text)
            out["ok"] = True
        elif kind == "wait":
            self.settle(int(float(arg or 500)))
            out["ok"] = True
        elif kind in ("read", "expect"):
            try:
                v = self.cdp.evaluate(arg)
            except RuntimeError as exc:
                return dict(out, ok=False, error=_short(str(exc), 240))
            out["value"] = v if isinstance(v, (int, float, bool)) or v is None else _short(v)
            out["ok"] = bool(v) if kind == "expect" else True
        elif kind == "shot":
            out.update(ok=True, file=self.shot(arg.strip() or "step"))
        else:
            return dict(out, ok=False, error="steps are click:, key:, type:, wait:, read:, "
                                             "expect:, shot:")
        self.settle(150)
        return out

    def run(self, steps: list[str]) -> dict:
        rep: dict = {"url": self.url, "steps": [], "widths": []}
        t0 = time.monotonic()
        self.size(self.widths[0])
        self.cdp.call("Page.reload")
        self.b.wait_ready()
        self.settle(500)
        rep["controls"] = self.cdp.evaluate(_CONTROLS)
        for s in steps:
            r = self.step(s)
            rep["steps"].append(r)
            if not r.get("ok") and s.split(":", 1)[0].strip().lower() not in ("expect",):
                break                               # a step that could not run ends the script
        try:                                        # the page's own state hook, after the steps
            st = self.cdp.evaluate("typeof (window.tandem_state || window.__state) === 'function' "
                                   "? (window.tandem_state || window.__state)() : null")
            if st is not None:
                rep["state"] = st
        except RuntimeError:
            pass
        for i, w in enumerate(self.widths):
            if i:
                self.size(w)
                self.cdp.call("Page.reload")
                self.b.wait_ready()
                self.settle(500)
            lay = self.cdp.evaluate(_LAYOUT % w)
            lay["overflow_x"] = lay["scroll_width"] > w + 1
            lay["shot"] = self.shot(f"page-{w}")
            lay["look"] = self.look(w)
            rep["widths"].append(lay)
        self.settle(100)
        rep["errors"], rep["warnings"] = _problems(self.cdp.events)
        rep["seconds"] = round(time.monotonic() - t0, 1)
        failed = [s["step"] for s in rep["steps"] if not s.get("ok")]
        rep["ok"] = not rep["errors"] and not failed and len(rep["steps"]) == len(steps)
        rep["verdict"] = ("PASS: the page loads, no console errors" + (", every step ran" if steps else "")
                          if rep["ok"] else
                          "FAIL: " + "; ".join(([f"{len(rep['errors'])} console error(s)"] if rep["errors"] else [])
                                               + [f"step failed: {f}" for f in failed]
                                               + (["steps not run after a failed one"]
                                                  if len(rep["steps"]) < len(steps) else [])))
        return rep

    def close(self) -> None:
        self.b.close()


def png_pixels(png: bytes) -> tuple[int, int, list] | None:
    """(width, height, rows of (r, g, b)) of an 8-bit RGB or RGBA PNG without interlace (what
    Chrome writes); None for any other kind."""
    pos, idat, head = 8, [], None
    while pos + 8 <= len(png):
        n, kind = struct.unpack(">I4s", png[pos:pos + 8])
        if kind == b"IHDR":
            head = struct.unpack(">IIBBBBB", png[pos + 8:pos + 21])
        elif kind == b"IDAT":
            idat.append(png[pos + 8:pos + 8 + n])
        pos += 12 + n
    if head is None or head[2] != 8 or head[3] not in (2, 6) or head[6]:
        return None
    w, h, bpp = head[0], head[1], 3 if head[3] == 2 else 4
    raw, stride = zlib.decompress(b"".join(idat)), w * bpp
    prev, rows = bytearray(stride), []
    for y in range(h):
        f, line = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b, c = prev[i], prev[i - bpp] if i >= bpp else 0
            if f == 1:
                line[i] = (line[i] + a) & 255
            elif f == 2:
                line[i] = (line[i] + b) & 255
            elif f == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append([tuple(line[x:x + 3]) for x in range(0, stride, bpp)])
        prev = line
    return w, h, rows


def look_of(png: bytes) -> dict | None:
    """A screenshot in numbers: `brightness` (mean luma, 0 to 100), `top_colour` and `top_share`
    (the most common colour, 16 levels a channel, and its share in %), `top` (the three most
    common colours over 1 % with their shares: a red cube on a dark page is a second colour of a
    few %), `colours` (the colours over 1 % of it) and `saturation` (the mean, 0 to 100)."""
    got = png_pixels(png)
    if got is None or not got[0] or not got[1]:
        return None
    px = [p for row in got[2] for p in row]
    counts = collections.Counter((r >> 4, g >> 4, b >> 4) for r, g, b in px)
    (top, n_top), = counts.most_common(1)
    n = len(px)

    def hexof(c):
        return "#" + "".join(f"{v * 17:02x}" for v in c)
    return {"brightness": round(sum(0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in px)
                                / n / 2.55),
            "top_colour": hexof(top),
            "top_share": round(100 * n_top / n),
            "top": [[hexof(c), round(100 * k / n)] for c, k in counts.most_common(3) if k * 100 >= n],
            "colours": sum(1 for v in counts.values() if v * 100 >= n),
            "saturation": round(sum(max(p) - min(p) for p in px) / n / 2.55)}


PAGE_QUESTION = "Does this page check show a failure?"


def page_verdict(rep: dict, laya_url: str | None = None) -> dict:
    """Laya's verdict on the report (the laya-testing package's rule: every run's evidence is
    judged), one call; a failed call is `unavailable` with its reason, never a retry loop."""
    text = json.dumps({"verdict": rep.get("verdict"), "errors": rep.get("errors"),
                       "steps": [{k: v for k, v in s.items() if k != "disabled"} for s in rep.get("steps") or ()],
                       "overflow": [w["width"] for w in rep.get("widths") or () if w.get("overflow_x")],
                       "text": (rep.get("widths") or [{}])[0].get("text", "")[:200]},
                      ensure_ascii=False)[:1100]
    try:
        st, out, _ = Laya(laya_url, timeout=20.0)._post("/v1/laya/ask", {
            "question": PAGE_QUESTION, "choices": ["failure", "clean"], "text": text})
    except OSError as exc:
        return {"unavailable": _short(str(exc), 120)}
    if st != 200 or "label" not in out:
        return {"unavailable": _short(str((out.get("error") or {}).get("message") or st), 120)}
    return {"label": out["label"], "p": out.get("p"), "sure": out.get("sure")}


def page_check(url: str, steps: list[str], widths: list[int], shots: str,
               cdp: str | None = None, timeout: float = PAGE_TIMEOUT_S) -> tuple[dict, int]:
    """(the report, the exit code: 0 pass, 1 a problem in the page or a step, 2 no browser here or
    the check itself failed). Chrome is closed on every path."""
    if cdp is None and find_chrome() is None:
        return ({"url": url, "ok": False, "verdict": "NO BROWSER: no Chrome or Chromium on this "
                 "machine. Do not build a browser driver: check the page by reading its code and "
                 "report the browser check as not possible."}, 2)

    def _late(_n, _f):
        raise PageTimeout(f"the check took longer than {timeout:.0f} s")
    old = None
    if hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, _late)
        signal.alarm(max(1, int(timeout)))
    pc = None
    try:
        pc = PageCheck(url, widths, shots, cdp)
        rep = pc.run(steps)
        return rep, 0 if rep["ok"] else 1
    except (PageTimeout, OSError, RuntimeError, SystemExit) as exc:
        msg = str(exc) or type(exc).__name__
        return ({"url": url, "ok": False, "verdict": f"CHECK FAILED: {_short(msg, 300)}. The page "
                 "may be fine: fix the URL or the step, or report the browser check as not "
                 "possible; do not build another driver."}, 2)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old or signal.SIG_DFL)
        if pc is not None:
            pc.close()


# ---------------------------------------------------------------------------- the report
def report(path: str = LOG, last: int = 6, run: str | None = None) -> str:
    """The log's last run (or `run`) as a short text: outcomes, deaths with where and why and the
    decisions before them, stuck spots, the actions, Laya's numbers."""
    rows = [json.loads(ln) for ln in open(path) if ln.strip().startswith("{")]
    starts = [i for i, r in enumerate(rows) if "run" in r and (run is None or r["run"] == run)]
    if not starts:
        return f"{path}: no run"
    i0 = starts[-1]
    head = rows[i0]
    body = []
    for r in rows[i0 + 1:]:
        if "run" in r:
            break
        body.append(r)
    steps = [r for r in body if "a" in r]
    events = [r for r in body if "event" in r]
    endrow = next((r for r in body if "end" in r), None)
    outc = collections.Counter(e["event"] for e in events if e["event"] in ("death", "win", "timeout"))
    ms = sorted(r["ms"] for r in steps if isinstance(r.get("ms"), (int, float)))
    pct = lambda q: ms[min(len(ms) - 1, int(q * len(ms)))] if ms else None    # noqa: E731
    eps = len({r["ep"] for r in body if "ep" in r})
    out = [f"playtest {head['run']} ({head.get('at')}{', ' + head['label'] if head.get('label') else ''}): "
           f"{eps} episodes, {len(steps)} steps; {outc['win']} won, {outc['death']} deaths, "
           f"{outc['timeout']} hit the step budget"
           + ("" if endrow else " (the log has no end: the run was cut off)")]
    if ms:
        unsure = sum(1 for r in steps if r.get("sure") is False)
        fb = sum(1 for r in steps if r.get("why"))
        out.append(f"Laya: {pct(0.5):.0f} ms p50, {pct(0.95):.0f} ms p95 a step; not sure on "
                   f"{unsure} of {len(steps)}" + (f"; {fb} fallbacks ({', '.join(sorted({r['why'] for r in steps if r.get('why')}))})" if fb else ""))
    hist = collections.Counter(r["a"] for r in steps)
    if hist:
        out.append("actions: " + ", ".join(f"{a} {n * 100 // len(steps)}%" for a, n in hist.most_common()))
    by_ep = collections.defaultdict(list)
    for r in steps:
        by_ep[r["ep"]].append(r)
    deaths = [e for e in events if e["event"] == "death"]
    for n, e in enumerate(deaths[:8], 1):
        where = f" at {fmt_pos(e.get('pos'))}" if e.get("pos") else ""
        why = f": {e['cause']}" if e.get("cause") else ""
        out.append(f"death {n} (episode {e['ep']}, step {e['step']}){where}{why}")
        before = [r for r in by_ep[e["ep"]] if r["step"] <= e["step"]][-last:]
        for r in before:
            p = f" p={r['p']:.2f}" if isinstance(r.get("p"), (int, float)) else ""
            flag = "" if r.get("sure") is not False else " (unsure)"
            out.append(f"   step {r['step']}: {r['a']}{p}{flag}{' [' + r['why'] + ']' if r.get('why') else ''}  {r['s'][:160]}")
    if len(deaths) > 8:
        out.append(f"... {len(deaths) - 8} deaths more")
    spots = collections.Counter(fmt_pos(e.get("pos")) for e in deaths if e.get("pos"))
    if spots and max(spots.values()) > 1:
        out.append("deaths by place: " + ", ".join(f"{p} x{n}" for p, n in spots.most_common(5)))
    for e in [e for e in events if e["event"] == "stuck"][:5]:
        out.append(f"stuck (episode {e['ep']}, step {e['step']}) at {fmt_pos(e.get('pos'))} for "
                   f"{e['steps']} steps: {e['s'][:160]}")
    for e in [e for e in events if e["event"] == "timeout"][:3]:
        out.append(f"step budget hit (episode {e['ep']}) at {fmt_pos(e.get('pos')) or e['s'][:120]}")
    return "\n".join(out)


def fmt_pos(p) -> str:
    if not p:
        return ""
    return "x=%s y=%s" % (p[0], p[1]) if len(p) > 1 and p[1] is not None else f"x={p[0]}"


# ---------------------------------------------------------------------------- the log sink
def serve_log(port: int, path: str = LOG) -> None:
    """POST /log (JSON lines) appends to the log; the in-page kit's sink. CORS open."""
    import http.server
    lock = threading.Lock()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    class H(http.server.BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8", "replace")
            lines = [ln for ln in data.splitlines() if ln.strip().startswith("{")]
            with lock, open(path, "a") as f:
                for ln in lines:
                    f.write(ln.strip() + "\n")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"lines": len(lines)}).encode())

        def log_message(self, *a):
            pass

    print(f"drive log sink: POST http://127.0.0.1:{port}/log -> {path}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


# ---------------------------------------------------------------------------- the command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--actions", default="", help="label=meaning,... (or labels)")
        p.add_argument("--question", default=QUESTION)
        p.add_argument("--episodes", type=int, default=3)
        p.add_argument("--steps", type=int, default=600, help="step budget per episode")
        p.add_argument("--every-ms", type=int, default=None,
                       help="one step per tick (browser: 250; else 0, as fast as Laya answers)")
        p.add_argument("--fallback", default=None, help="the action when Laya cannot answer (default wait or the first)")
        p.add_argument("--stuck", type=int, default=20, help="steps on one spot that count as stuck")
        p.add_argument("--log", default=LOG)
        p.add_argument("--url", default=None, help="TandemLLM (default $TANDEM_URL or http://127.0.0.1:8000)")
        p.add_argument("--label", default="", help="a word for the report's head (the build, the fix)")
        p.add_argument("--no-report", action="store_true")

    p = sub.add_parser("proc")
    p.add_argument("command")
    common(p)
    p = sub.add_parser("cmd")
    p.add_argument("--state", required=True)
    p.add_argument("--act", required=True, help="the command, {action} is replaced")
    p.add_argument("--reset", default=None)
    common(p)
    p = sub.add_parser("browser")
    p.add_argument("page", help="the page's URL")
    p.add_argument("--keys", default="", help="action=Key+Key,...: run=ArrowRight,jump=Space,wait=")
    p.add_argument("--cdp", default=None, help="a running Chrome's DevTools (http://127.0.0.1:9222)")
    p.add_argument("--headed", action="store_true")
    common(p)
    p = sub.add_parser("page")
    p.add_argument("page", help="the page's URL (http://127.0.0.1:8765/ or file:///.../index.html)")
    p.add_argument("--step", action="append", default=[],
                   help="click:SEL|text=Label, key:Name[*ms], type:SEL=text, wait:ms, read:JS, "
                        "expect:JS, shot:name (in order; repeat the flag)")
    p.add_argument("--widths", default="1280,390", help="viewport widths, the steps run at the first")
    p.add_argument("--shots", default=None, help="where the screenshots go (default: beside the kit)")
    p.add_argument("--cdp", default=None, help="a running Chrome's DevTools (http://127.0.0.1:9222)")
    p.add_argument("--timeout", type=float, default=PAGE_TIMEOUT_S, help="seconds for the whole check")
    p.add_argument("--url", default=None, help="TandemLLM for Laya's verdict (default $TANDEM_URL "
                                              "or http://127.0.0.1:8000)")
    p.add_argument("--no-laya", action="store_true", help="no verdict from Laya")
    p = sub.add_parser("report")
    p.add_argument("log", nargs="?", default=LOG)
    p.add_argument("--last", type=int, default=6, help="decisions shown before each death")
    p = sub.add_parser("serve-log")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--log", default=LOG)
    a = ap.parse_args(argv)
    # a SIGTERM or a hang-up (a tool call's timeout, a closed terminal) runs the `finally`s below,
    # which close the program and the Chrome this started
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, lambda n, f: sys.exit(128 + n))
        except (ValueError, OSError):
            pass

    if a.mode == "report":
        print(report(a.log, a.last))
        return 0
    if a.mode == "serve-log":
        serve_log(a.port, a.log)
        return 0
    if a.mode == "page":
        widths = [int(w) for w in a.widths.split(",") if w.strip()] or [1280]
        shots = a.shots or os.path.dirname(os.path.abspath(__file__))
        rep, code = page_check(a.page, a.step, widths, shots, a.cdp, a.timeout)
        if not a.no_laya and code in (0, 1):
            rep["laya"] = page_verdict(rep, a.url)
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        return code
    keys = {}
    if a.mode == "browser":
        for part in [x for x in a.keys.split(",") if x.strip()]:
            k, _, v = part.partition("=")
            keys[k.strip()] = v.strip()
    # a UI's state may list its own controls each step; then the fixed set is only the fallback
    actions = parse_actions(a.actions) if a.actions else ({k: k for k in keys} or {"wait": "do nothing"})
    every = a.every_ms if a.every_ms is not None else (250 if a.mode == "browser" else 0)
    drv = Driver(actions, a.question, a.url, log=a.log, every_ms=every, steps=a.steps,
                 episodes=a.episodes, fallback=a.fallback, stuck_steps=a.stuck, label=a.label)
    if a.mode == "proc":
        prog = Proc(a.command)
        try:
            drv.run(prog.state, prog.apply, prog.reset)
        finally:
            prog.close()
    elif a.mode == "cmd":
        drv.run(cmd_state(a.state), lambda act, s: sh(a.act.replace("{action}", shlex.quote(act))),
                (lambda: sh(a.reset)) if a.reset else None)
    else:
        b = Browser(a.page, a.cdp, headless=not a.headed, keys=keys)
        try:
            drv.run(b.state, b.apply, b.reset)
        finally:
            b.close()
    if not a.no_report:
        print(report(a.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
