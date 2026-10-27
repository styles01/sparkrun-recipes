"""One sandbox container per chat, and the memory guard that keeps them away from the engine
(SRV-355, Chat 6).

THE RUNTIME. chatd speaks the Docker Engine API over a unix socket (`QSE_SANDBOX_SOCKET`, default
/var/run/docker.sock). Podman serves the same API, so the plan's rootless Podman under its own user
(`ops/sandbox/install.sh`, which needs root once) is a socket path away: nothing here changes.
On the Spark today the socket is rootful Docker, which is why every container is locked down here
rather than relying on the runtime:

- `tandem-chat-<id>` from `tandem-sandbox:1` (ops/sandbox/Containerfile), as uid 1000 (`agent`),
  every capability dropped, no-new-privileges, the default seccomp profile, a read-only root, tmpfs
  /tmp (512 MB), no devices and no GPU;
- `--network none`: the container has a loopback and nothing else. Its internet goes through
  chatd's filtering proxy for that chat only (chatd/egress.py), which refuses the LAN and this box;
- mounts: /workspace (the chat's files, chatd/workspace.py), /home/agent (pip, npm, uv caches) and
  /run/tandem (the proxy's socket, read-only). Nothing of chatd's or the engine's is mounted;
- limits: 4 GiB memory with no swap (QSE_SANDBOX_MEM_GB), 4 CPUs, 512 pids (a fork bomb stops
  there). A process the kernel kills at the limit is reported to the model as "out of memory (1 GiB
  sandbox limit)" (`oom_kills`, the cgroup's memory.events).

THE MEMORY GUARD. The sandbox has priority over the engine's idle memory (ruling of 2026-10-15): a
chat's tool call gets its sandbox, and when memory is short the engine gives way. The engine keeps
itself at its own floors: its KV pages pass the pool's hard floor and its requests the admission
floor (10 GiB each on the Spark), and below them it gives back idle conversations or waits. What a
sandbox must leave is only what the engine takes without asking: a prefill's workspace, beyond
the free blocks its allocator already holds. Measured on the Spark (27B EXL3 6.0, 4 slots, KV pool,
2026-10-15, the engine's own `last_prefill_peak` and `observed.workspace`): 2.7 GiB for a
203,052-token prefill, 3.5 GiB scaled to the 262,144-token window; the allocator held 10 GiB of
free blocks at the time. So:
- admission: a sandbox starts when MemAvailable minus its limit stays at or above FLOOR_GB (4,
  QSE_SANDBOX_FLOOR_GB). When it does not, chatd asks the engine to give back the difference
  (`release`, the engine's POST /v1/cache/release: free allocator blocks, caches, the small models,
  idle conversations) and looks again;
- the running sandboxes' limits add up to at most QSE_SANDBOX_TOTAL_GB (QSE_SANDBOX_MAX_RUNNING
  times the limit), and at most QSE_SANDBOX_MAX_RUNNING (4) run at once (the least recently used
  idle one is stopped to make room);
- a call waits at most WAIT_S (60 s, QSE_SANDBOX_WAIT_S). Then it fails with the reason and the
  numbers, which the model reads as the tool's result and the chat shows on the call: it never
  hangs;
- every second: under STOP_GB (3) the engine is asked first, then the least recently used idle
  sandbox stops (a busy one when none is idle); under KILL_GB (2) every sandbox is killed. Both
  are under the admission's floor, so the guard never stops a sandbox the admission just let in.
  Stop, not pause: pausing frees nothing. The files stay; the chat says why (`stopped_reason:
  memory_guard`);
- 30 minutes with no tool call and no terminal: stop (`idle`).

chatd restarts adopt running containers by their label and do not recreate them.

ONE CHATD'S CONTAINERS. Several chatds may share one container engine (a second install, a test
instance with its own --data). Each labels its containers `tandem.chat.instance=<id>`, the id kept
in its data dir (`sandbox-instance.json`, made anew when the dir is a copy), and adopts, reaps,
stops, kills and removes only those. A container from before the label (no instance label) is this
chatd's only when the chat's workspace folder is in this data dir; one that stopped is created
again with the label at its next start (the files are mounts, nothing is lost). Live 2026-10-16: a
test chatd adopted the live chatd's containers, found their chats missing from its store and
removed them.

A CONTAINER THAT IS GONE. Before each use a running box is looked at (the engine's inspect): a
container removed or stopped outside chatd is marked stopped, a new one starts for the call, and
the call's result says once that the sandbox restarted (files kept, processes gone). An exec the
engine answers 404 or 409 (no such container, not running) does the same. Live 2026-10-16: every
call answered `exec failed (404)` and the model retried until the step limit.
"""

from __future__ import annotations

import http.client
import json
import os
import select
import shutil
import socket
import secrets
import struct
import threading
import time
import urllib.parse

from chatd.egress import Proxy

GiB = 1024 ** 3
IMAGE = os.environ.get("QSE_SANDBOX_IMAGE", "tandem-sandbox:1")
SOCKET = os.environ.get("QSE_SANDBOX_SOCKET", "/var/run/docker.sock")
# 4 GiB: headless Chromium drawing a Three.js app in software (SwiftShader) went past 1 GiB and
# the kernel killed it, so render_page and the verify phase failed on the apps they check
LIMIT_BYTES = int(float(os.environ.get("QSE_SANDBOX_MEM_GB") or 4) * GiB)
CPUS = 4
PIDS = 512
DISK_LIMIT = 10 * GiB
FLOOR_GB = float(os.environ.get("QSE_SANDBOX_FLOOR_GB") or 4)
STOP_GB, KILL_GB = 3.0, 2.0
WAIT_S = float(os.environ.get("QSE_SANDBOX_WAIT_S") or 60)
RELEASE_EVERY_S = 10.0
IDLE_S = 30 * 60
CGROUP = "/sys/fs/cgroup"
MEM_TTL_S = 5.0
LABEL = "tandem.chat"
INSTANCE_LABEL = "tandem.chat.instance"
INSTANCE_FILE = "sandbox-instance.json"
RESTARTED = ("[the sandbox was restarted: its container was gone (removed or stopped outside "
             "chatd). The files in /workspace are kept; processes that ran in it (dev servers, "
             "background jobs) are gone and must be started again]")


class Unavailable(Exception):
    """No sandbox for this call: the image is missing, the runtime is down, the run stopped while
    it waited for memory, or memory stayed short for WAIT_S. The message is the tool result the
    model reads; `data` is the call's `output.data` for the chat (the reason and the numbers)."""

    def __init__(self, message: str, data: dict | None = None):
        super().__init__(message)
        self.data = data


class Gone(Unavailable):
    """The container is gone (the engine answered 404, or 409: not running); the box is marked
    stopped and the next ensure() starts a new one."""


def instance_id(data_dir: str) -> str:
    """This chatd's id on the container engine, kept in its data dir; a dir copied elsewhere
    (another path) gets its own."""
    path = os.path.join(data_dir, INSTANCE_FILE)
    here = os.path.realpath(data_dir)
    try:
        with open(path) as fh:
            doc = json.load(fh)
        if doc.get("dir") == here and doc.get("id"):
            return str(doc["id"])
    except (OSError, ValueError, AttributeError):
        pass
    doc = {"id": secrets.token_hex(8), "dir": here}
    os.makedirs(data_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh)
    os.replace(tmp, path)
    return doc["id"]


RELAY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ops", "sandbox",
                     "relay.py")


def relay_cmd(*args: str) -> list[str]:
    """`relay.py <args>` in the sandbox, with this release's relay sent in the exec (python3 -c):
    a container keeps the image it was made from, and its /opt/tandem/relay.py with it. Without
    the source tree (an installed package), the image's own."""
    try:
        with open(RELAY) as fh:
            return ["python3", "-c", fh.read(), *args]
    except OSError:
        return ["python3", "/opt/tandem/relay.py", *args]


def limit_words() -> str:
    return f"{LIMIT_BYTES / GiB:g} GiB"


def meminfo(path: str = "/proc/meminfo") -> dict:
    out = {}
    try:
        with open(path) as fh:
            for line in fh:
                k, _, v = line.partition(":")
                out[k.strip()] = int(v.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return out


class _UnixConn(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.unix_path)
        self.sock = s


class Engine:
    """The container engine's API (Docker's, which Podman also serves)."""

    def __init__(self, path: str = SOCKET):
        self.path = path

    def call(self, method: str, path: str, body=None, timeout: float = 60.0):
        c = _UnixConn(self.path, timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            c.request(method, path, body=data,
                      headers={"Content-Type": "application/json"} if data else {})
            r = c.getresponse()
            raw = r.read()
            try:
                return r.status, json.loads(raw) if raw else None
            except ValueError:
                return r.status, raw.decode("utf-8", "replace")
        finally:
            c.close()

    def hijack(self, path: str, body: dict) -> tuple[socket.socket, bytes]:
        """POST that upgrades to the raw stream (exec start with attach)."""
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(30)
        s.connect(self.path)
        data = json.dumps(body).encode()
        s.sendall(f"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                  f"Connection: Upgrade\r\nUpgrade: tcp\r\nContent-Length: {len(data)}\r\n\r\n"
                  .encode() + data)
        head = b""
        while b"\r\n\r\n" not in head:
            d = s.recv(4096)
            if not d:
                raise OSError("the engine closed the exec stream")
            head += d
        status = head.split(b" ", 2)[1] if b" " in head else b"0"
        if status not in (b"101", b"200"):
            s.close()
            raise OSError(f"exec start answered {status.decode()}")
        s.settimeout(None)
        return s, head.split(b"\r\n\r\n", 1)[1]


def _proc_starttime(pid: int) -> int | None:
    """Field 22 of /proc/<pid>/stat (jiffies since boot): the cheap fingerprint that the pid
    still names the process we read it for (it changes on every pid recycle)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return int(fh.read().rpartition(b")")[2].split()[19])   # field 22, after comm
    except (OSError, ValueError, IndexError):
        return None


FRAME_CAP = 64 << 20


def demux(sock: socket.socket, first: bytes, on_data) -> None:
    """A non-TTY attach stream: 8-byte frames [stream, 0, 0, 0, size(4, big-endian)]."""
    buf = first
    while True:
        while len(buf) >= 8:
            stream, size = buf[0], struct.unpack(">I", buf[4:8])[0]
            # a corrupt or hostile stream can announce a ~4GB frame and then drip bytes:
            # one frame over 64 MiB is junk, drop the connection rather than buffer it
            if size > FRAME_CAP:
                raise OSError(f"an exec frame claims {size:,} bytes (over the 64 MiB cap)")
            if len(buf) < 8 + size:
                break
            on_data(stream, buf[8:8 + size])
            buf = buf[8 + size:]
        try:
            d = sock.recv(65536)
        except OSError:
            return
        if not d:
            return
        buf += d


class ClientGone(Exception):
    """The client a reader watches closed its connection."""


class ExecReader:
    """A non-TTY attach stream's stdout read as it arrives (#86): `read(wait)` gives the next
    stdout bytes, b"" once the exec's stream ended (`ended`), and raises TimeoutError when nothing
    came within `wait` seconds (None: no limit). With `watch` (the client's socket, its request
    already read) it raises ClientGone when that socket reaches its end while it waits, so a
    reader never outlives the client it serves. stderr frames are dropped; the buffer holds at
    most one frame (FRAME_CAP)."""

    def __init__(self, sock: socket.socket, first: bytes = b"", watch: socket.socket | None = None):
        self.sock, self.buf, self.ended = sock, bytearray(first), False
        self.watch = watch
        self.poll = select.poll()
        self.poll.register(sock, select.POLLIN)
        if watch is not None:
            self.poll.register(watch, select.POLLIN)

    def _frame(self) -> bytes | None:
        while len(self.buf) >= 8:
            stream, size = self.buf[0], int.from_bytes(self.buf[4:8], "big")
            if size > FRAME_CAP:
                raise OSError(f"an exec frame claims {size:,} bytes (over the 64 MiB cap)")
            if len(self.buf) < 8 + size:
                return None
            data = bytes(self.buf[8:8 + size])
            del self.buf[:8 + size]
            if stream == 1 and data:
                return data
        return None

    def _client_gone(self) -> bool:
        try:
            peek = self.watch.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True
        if peek:                         # a pipelined request: nothing more to learn here
            self.poll.unregister(self.watch)
            self.watch = None
            return False
        return True

    def read(self, wait: float | None = None) -> bytes:
        deadline = None if wait is None else time.monotonic() + wait
        while True:
            data = self._frame()
            if data is not None:
                return data
            if self.ended:
                return b""
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                raise TimeoutError("the exec sent nothing in time")
            events = self.poll.poll(None if left is None else max(1, int(left * 1000)))
            for fd, ev in events:
                if self.watch is not None and fd == self.watch.fileno():
                    if self._client_gone():
                        raise ClientGone()
                elif fd == self.sock.fileno():
                    try:
                        d = self.sock.recv(65536)
                    except OSError:
                        d = b""
                    if d:
                        self.buf += d
                    else:
                        self.ended = True
                        self.poll.unregister(self.sock)


class Box:
    """What chatd knows of one chat's sandbox."""

    def __init__(self, cid: str):
        self.cid = cid
        self.container: str | None = None
        self.state = "stopped"               # running, stopped, creating, waiting_memory, error
        self.stopped_reason: str | None = None
        self.started = 0.0
        self.last_used = 0.0
        self.busy = 0                        # execs and attached terminals
        self.error: str | None = None
        self.proxy: Proxy | None = None
        self.lock = threading.RLock()
        self.mem: tuple[float, int | None] = (0.0, None)   # (when read, bytes in use)
        self.restarted = False               # its container was gone: the next result says so


class Sandboxes:
    def __init__(self, data_dir: str, *, engine: Engine | None = None, image: str = IMAGE,
                 log=None, on_change=None, network_mode=None, meminfo_fn=meminfo, clock=time.time,
                 release=None):
        self.data = data_dir
        self.engine = engine or Engine()
        self.image = image
        self.log = log or (lambda line: None)
        self.on_change = on_change or (lambda cid, status: None)
        self.network_mode = network_mode or (lambda cid: "on")
        self.meminfo = meminfo_fn
        self.clock = clock
        self.boxes: dict[str, Box] = {}
        self.lock = threading.Lock()
        self.max_running = int(os.environ.get("QSE_SANDBOX_MAX_RUNNING") or 4)
        self.total_gb = float(os.environ.get("QSE_SANDBOX_TOTAL_GB") or
                              self.max_running * LIMIT_BYTES / GiB)
        # `release(nbytes) -> bytes given back`: the engine makes room (chatd/engine.py
        # `release_memory`); None where there is no engine to ask
        self.release = release
        self.released = 0                    # bytes the engine gave back at the last ask
        self._asked: float | None = None
        self.wait_s, self.poll_s, self.release_every_s = WAIT_S, 2.0, RELEASE_EVERY_S
        self._avail: tuple[float, bool] | None = None
        self.stopping = threading.Event()
        self._instance: str | None = None

    @property
    def instance(self) -> str:
        if self._instance is None:
            self._instance = instance_id(self.data)
        return self._instance

    def owns(self, labels: dict | None, cid: str) -> bool:
        """A container is this chatd's: its instance label is ours, or it has none (made before
        the label) and the chat's workspace folder is in this data dir."""
        labels = labels or {}
        inst = labels.get(INSTANCE_LABEL)
        if inst:
            return inst == self.instance
        return os.path.isdir(self.paths(cid)[0])

    # ------------------------------------------------------------------ availability
    def available(self) -> bool:
        """The image exists and the engine answers (cached for 60 s). A check that gets no answer
        (the Docker socket busy past 5 s, a reset) keeps the last answer: the sandbox's tools are
        the prompt's head, and a flap would make the next request read the whole chat again
        (SRV-663); a call while it is really gone fails as a call."""
        if self._avail is not None and self.clock() - self._avail[0] < 60:
            return self._avail[1]
        try:
            st, _ = self.engine.call("GET", f"/images/{urllib.parse.quote(self.image, safe='')}"
                                     f"/json", timeout=5)
            ok = st == 200
        except OSError:
            if self._avail is not None:
                self._avail = (self.clock(), self._avail[1])
                return self._avail[1]
            ok = False
        self._avail = (self.clock(), ok)
        return ok

    def box(self, cid: str) -> Box:
        with self.lock:
            b = self.boxes.get(cid)
            if b is None:
                b = self.boxes[cid] = Box(cid)
            return b

    def name(self, cid: str) -> str:
        return "tandem-chat-" + cid.lower()

    def paths(self, cid: str) -> tuple[str, str, str]:
        return (os.path.join(self.data, "workspaces", cid), os.path.join(self.data, "homes", cid),
                os.path.join(self.data, "egress", cid))

    # ------------------------------------------------------------------ adopt, status
    def adopt(self) -> int:
        """At start: the containers a previous chatd (of this data dir) left running keep
        running, and come back. Another chatd's are not looked at."""
        try:
            st, rows = self.engine.call("GET", "/containers/json?all=1&filters=" +
                                        urllib.parse.quote(json.dumps({"label": [LABEL]})))
        except OSError:
            return 0
        n = 0
        for r in rows if st == 200 and isinstance(rows, list) else []:
            cid = (r.get("Labels") or {}).get(LABEL)
            if not cid or not self.owns(r.get("Labels"), cid):
                continue
            b = self.box(cid)
            b.container = r.get("Id")
            if r.get("State") == "running":
                b.state, b.started, b.last_used = "running", self.clock(), self.clock()
                self._proxy(b)
                n += 1
        return n

    def memory_used(self, b: Box) -> int | None:
        """A running container's memory in use, as `docker stats` shows it (the cgroup's usage
        less its inactive file cache), read at most every MEM_TTL_S; None when the runtime does
        not say. It was a constant 0 before: the Context panel showed "running, 0 B"."""
        if b.state != "running" or not b.container:
            return None
        now = self.clock()
        if now - b.mem[0] < MEM_TTL_S:
            return b.mem[1]
        used = None
        try:
            st, s = self.engine.call("GET", f"/containers/{b.container}/stats?stream=false&"
                                     "one-shot=true", timeout=3)
            m = (s or {}).get("memory_stats") if st == 200 and isinstance(s, dict) else None
            if isinstance(m, dict) and m.get("usage") is not None:
                stats = m.get("stats") or {}
                cache = stats.get("inactive_file", stats.get("total_inactive_file", 0)) or 0
                used = max(0, int(m["usage"]) - int(cache))
        except (OSError, ValueError, TypeError):
            used = None
        b.mem = (now, used)
        return used

    def oom_kills(self, cid: str) -> int | None:
        """How many processes the kernel has killed at this sandbox's memory limit: `oom_kill` in
        its cgroup's memory.events (cgroup v2, Docker's systemd or cgroupfs layout); None where
        the file cannot be read (another runtime, cgroup v1)."""
        ctr = self.box(cid).container
        if not ctr:
            return None
        for path in (f"{CGROUP}/system.slice/docker-{ctr}.scope/memory.events",
                     f"{CGROUP}/docker/{ctr}/memory.events"):
            try:
                with open(path) as fh:
                    for line in fh:
                        k, _, v = line.partition(" ")
                        if k == "oom_kill":
                            return int(v)
            except (OSError, ValueError):
                continue
        return None

    def status(self, cid: str) -> dict:
        b = self.box(cid)
        ws, _, _ = self.paths(cid)
        mode = self.network_mode(cid)
        return {"state": b.state if self.available() or b.state == "running" else "error",
                "image": self.image, "cwd": "/workspace", "cpus": CPUS,
                "memory_bytes": self.memory_used(b), "memory_limit_bytes": LIMIT_BYTES,
                "disk_bytes": _du(ws), "disk_limit_bytes": DISK_LIMIT,
                "network": "on" if mode == "on" else "off", "allowlist": [],
                "uptime_s": int(self.clock() - b.started) if b.state == "running" else 0,
                "error": b.error or (None if self.available() else
                                     "the sandbox image is not installed on this box"),
                "stopped_reason": b.stopped_reason, "ports": []}

    def _changed(self, b: Box) -> None:
        try:
            self.on_change(b.cid, self.status(b.cid))
        except Exception:                                           # noqa: BLE001
            pass

    # ------------------------------------------------------------------ start
    def _running(self) -> list[Box]:
        with self.lock:
            return [b for b in self.boxes.values() if b.state == "running"]

    def _avail_gb(self) -> float:
        return self.meminfo().get("MemAvailable", 0) / GiB

    def _ask_engine(self, short_gb: float) -> int:
        """The engine gives way: it is asked for `short_gb` (at most every RELEASE_EVERY_S, since
        what it holds idle does not come back between two looks). Bytes it gave back."""
        now = time.monotonic()
        if self.release is None or (self._asked is not None and
                                    now - self._asked < self.release_every_s):
            return 0
        self._asked = now
        try:
            got = int(self.release(int(short_gb * GiB)) or 0)
        except Exception:                                           # noqa: BLE001
            got = 0
        self.released = got
        if got:
            self.log(f"[sandbox] the engine gave back {got / GiB:.1f} GiB for a sandbox")
        return got

    def short_words(self, avail: float) -> str:
        return (f"MemAvailable {avail:.1f} GiB: a {limit_words()} sandbox would leave the engine "
                f"under its {FLOOR_GB:g} GiB floor")

    def admit(self) -> str | None:
        """None when a new sandbox may start now, else the reason to wait. Short of memory, the
        engine is asked to give back the difference first."""
        avail = self._avail_gb()
        short = FLOOR_GB + LIMIT_BYTES / GiB - avail
        if avail and short > 0:
            if self._ask_engine(short):
                avail = self._avail_gb()
            if avail and avail - LIMIT_BYTES / GiB < FLOOR_GB:
                return self.short_words(avail)
        running = self._running()
        if (len(running) + 1) * LIMIT_BYTES / GiB > self.total_gb or \
                len(running) >= self.max_running:
            idle = sorted((b for b in running if not b.busy), key=lambda b: b.last_used)
            if idle:
                self.stop(idle[0].cid, "idle")
                return None if len(self._running()) < self.max_running else "making room"
            return (f"{len(running)} sandbox(es) of other chats are running commands, the most "
                    f"at once")
        return None

    def ensure(self, cid: str, cancel: threading.Event | None = None, on_wait=None) -> str:
        """The running container's id: created or started when needed, after admission."""
        if not self.available():
            raise Unavailable("the sandbox image is not installed on this box "
                              "(ops/sandbox/build.sh)")
        b = self.box(cid)
        with b.lock:
            b.last_used = self.clock()
            if b.state == "running" and b.container:
                if self._alive(b):
                    return b.container
                self.gone(cid)
            waited, t0 = False, time.monotonic()
            while True:
                why = self.admit()
                if why is None:
                    break
                spent = time.monotonic() - t0
                if not waited:
                    b.state, waited = "waiting_memory", True
                    self._changed(b)
                if spent >= self.wait_s:
                    raise self._gave_up(b, why, spent)
                if on_wait is not None:
                    on_wait(f"{why}; {spent:.0f} of {self.wait_s:.0f} s")
                if (cancel is not None and cancel.wait(self.poll_s)) or self.stopping.is_set():
                    b.state = "stopped"
                    self._changed(b)
                    raise Unavailable("stopped while waiting for memory")
                if cancel is None:
                    time.sleep(self.poll_s)
            b.state, b.error = "creating", None
            self._changed(b)
            try:
                b.container = self._start(cid)
                # a fresh workspace is a git repository, for the agent's own history
                self.run(cid, ["bash", "-lc", "[ -d .git ] || git init -q . 2>/dev/null; true"],
                         timeout=20)
            except (Unavailable, OSError) as exc:
                b.state, b.error = "error", str(exc)
                self._changed(b)
                raise Unavailable(str(exc)) from None
            b.state, b.stopped_reason = "running", None
            b.started = b.last_used = self.clock()
            self._proxy(b)
            self._changed(b)
            self.log(f"[sandbox] {cid} running")
            return b.container

    def _alive(self, b: Box) -> bool:
        """The engine still has the box's container, running. True when it cannot say."""
        try:
            st, info = self.engine.call("GET", f"/containers/{b.container}/json", timeout=5)
        except OSError:
            return True
        if st == 404:
            return False
        if st == 200 and isinstance(info, dict):
            return bool((info.get("State") or {}).get("Running", True))
        return True

    def gone(self, cid: str) -> None:
        """Its container was removed or stopped outside chatd: stopped, and a new one starts at
        the next use, whose result says so (`take_restarted`)."""
        b = self.box(cid)
        with b.lock:
            if b.state != "running":
                return
            b.state, b.stopped_reason, b.restarted = "stopped", "gone", True
            if b.proxy is not None:
                b.proxy.close()
                b.proxy = None
        self.log(f"[sandbox] {cid}: the container is gone; a new one starts at the next call")
        self._changed(b)

    def take_restarted(self, cid: str) -> str | None:
        """Once, after a box whose container was gone runs again: the words for the call's
        result."""
        b = self.box(cid)
        if b.restarted and b.state == "running":
            b.restarted = False
            return RESTARTED
        return None

    def _gave_up(self, b: Box, why: str, spent: float) -> Unavailable:
        """The wait's end: the reason and the numbers, for the model (the tool's result), the
        chat (the call's `output.data`) and the Sandbox panel (`error`)."""
        avail = self._avail_gb()
        freed = self.released / GiB
        b.state, b.error = "stopped", f"did not start after {spent:.0f} s: {why}"
        self._changed(b)
        self.log(f"[sandbox] {b.cid} not started after {spent:.0f} s (MemAvailable {avail:.1f} "
                 f"GiB, the engine gave back {freed:.1f} GiB)")
        memory = why.startswith("MemAvailable")
        msg = (f"no sandbox for this call: after waiting {spent:.0f} s it still could not start "
               f"({why}" + (f"; the engine gave back {freed:.1f} GiB" if memory else "") +
               "). The command did not run. Do not call bash or render_page again for now: answer "
               "with what you already have, or without tools, and tell the user the sandbox could "
               "not start " + ("for lack of memory." if memory else "because other chats' "
                                                                     "sandboxes are busy."))
        return Unavailable(msg, {"sandbox": {
            "reason": "no_memory" if memory else "busy", "why": why,
            "waited_s": round(spent), "mem_available_gib": round(avail, 1),
            "limit_gib": LIMIT_BYTES / GiB, "floor_gib": FLOOR_GB,
            "engine_released_gib": round(freed, 1)}})

    def _proxy(self, b: Box) -> None:
        if b.proxy is None:
            _, _, eg = self.paths(b.cid)
            b.proxy = Proxy(os.path.join(eg, "proxy.sock"), lambda: self.network_mode(b.cid),
                            self.log)
            b.proxy.start()

    def _start(self, cid: str) -> str:
        ws, home, eg = self.paths(cid)
        for d in (ws, home, eg):
            os.makedirs(d, exist_ok=True)
        name = self.name(cid)
        st, info = self.engine.call("GET", f"/containers/{name}/json")
        if st == 200:
            labels = ((info or {}).get("Config") or {}).get("Labels") or {}
            if labels.get(INSTANCE_LABEL) not in (None, self.instance):
                raise Unavailable("a sandbox container of another chatd has this chat's name "
                                  f"({name})")
            if INSTANCE_LABEL not in labels and not (info.get("State") or {}).get("Running"):
                # made before the instance label: a new one with it (the files are mounts)
                st, _ = self.engine.call("DELETE", f"/containers/{info['Id']}?force=1", timeout=30)
                st = 404 if st in (204, 404) else st
        if st == 404:
            proxy = "http://127.0.0.1:3128"
            body = {
                "Image": self.image, "User": "1000:1000", "WorkingDir": "/workspace",
                "Cmd": ["python3", "/opt/tandem/relay.py", "serve"],
                "Env": ["HOME=/home/agent", "TERM=xterm-256color", "LANG=C.UTF-8",
                        "PATH=/home/agent/.local/bin:/home/agent/.npm-global/bin:/usr/local/sbin:"
                        "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                        f"HTTP_PROXY={proxy}", f"HTTPS_PROXY={proxy}", f"http_proxy={proxy}",
                        f"https_proxy={proxy}", "NO_PROXY=localhost,127.0.0.1",
                        "no_proxy=localhost,127.0.0.1", "npm_config_prefix=/home/agent/.npm-global",
                        "PIP_DISABLE_PIP_VERSION_CHECK=1", "PYTHONUNBUFFERED=1"],
                "Labels": {LABEL: cid, INSTANCE_LABEL: self.instance},
                "HostConfig": {
                    "Binds": [f"{ws}:/workspace:rw", f"{home}:/home/agent:rw",
                              f"{eg}:/run/tandem:ro"],
                    "NetworkMode": "none", "Memory": LIMIT_BYTES, "MemorySwap": LIMIT_BYTES,
                    "NanoCpus": CPUS * 10 ** 9, "PidsLimit": PIDS, "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges"], "ReadonlyRootfs": True,
                    "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=512m"}, "Init": True,
                    "Devices": [], "Privileged": False},
            }
            st, info = self.engine.call("POST", f"/containers/create?name={name}", body)
            if st not in (200, 201):
                raise Unavailable(f"the sandbox could not be created ({st}: "
                                  f"{str((info or {}).get('message') if isinstance(info, dict) else info)[:200]})")
            container = info["Id"]
        elif st == 200:
            container = info["Id"]
            if ((info.get("HostConfig") or {}).get("Memory") or 0) != LIMIT_BYTES:
                # made under another limit (an older default, a changed QSE_SANDBOX_MEM_GB): this
                # start takes the current one
                self.engine.call("POST", f"/containers/{container}/update",
                                 {"Memory": LIMIT_BYTES, "MemorySwap": LIMIT_BYTES})
        else:
            raise Unavailable(f"the container engine answered {st}")
        st, info = self.engine.call("POST", f"/containers/{container}/start")
        if st not in (204, 304):
            raise Unavailable(f"the sandbox did not start ({st})")
        return container

    # ------------------------------------------------------------------ stop, reset
    def stop(self, cid: str, reason: str = "user") -> None:
        b = self.box(cid)
        if b.container and b.state == "running":
            try:
                self.engine.call("POST", f"/containers/{b.container}/stop?t=3", timeout=30)
            except OSError:
                pass
            self.log(f"[sandbox] {cid} stopped ({reason})")
        b.state, b.stopped_reason = "stopped", reason
        if b.proxy is not None:
            b.proxy.close()
            b.proxy = None
        self._changed(b)

    def kill_all(self, reason: str = "memory_guard") -> None:
        for b in self._running():
            try:
                self.engine.call("POST", f"/containers/{b.container}/kill", timeout=10)
            except OSError:
                pass
            b.state, b.stopped_reason = "stopped", reason
            if b.proxy is not None:
                b.proxy.close()
                b.proxy = None
            self._changed(b)
        self.log(f"[sandbox] every sandbox killed ({reason})")

    def remove(self, cid: str, files: bool = False) -> None:
        self.stop(cid, "user")
        b = self.box(cid)
        try:
            ctr = b.container
            if ctr is None:
                # not started by this chatd since it began: by name, when it is ours
                st, info = self.engine.call("GET", f"/containers/{self.name(cid)}/json",
                                            timeout=10)
                if st == 200 and isinstance(info, dict) and \
                        self.owns((info.get("Config") or {}).get("Labels"), cid):
                    ctr = info.get("Id")
            if ctr:
                self.engine.call("DELETE", f"/containers/{ctr}?force=1", timeout=30)
        except OSError:
            pass
        b.container = None
        if files:
            for d in self.paths(cid):
                shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------ exec
    def open_exec(self, cid: str, cmd: list[str], *, tty: bool = False, stdin: bool = False,
                  env: list[str] | None = None, workdir: str = "/workspace"
                  ) -> tuple[str, socket.socket, bytes]:
        container = self.box(cid).container
        st, info = self.engine.call("POST", f"/containers/{container}/exec", {
            "AttachStdin": stdin, "AttachStdout": True, "AttachStderr": True, "Tty": tty,
            "Cmd": cmd, "WorkingDir": workdir, "User": "1000:1000", "Env": env or []})
        if st in (404, 409):
            self.gone(cid)
            raise Gone("the sandbox's container was gone (removed or stopped outside chatd); "
                       "the next call starts a new one, with the files kept")
        if st != 201:
            raise Unavailable(f"exec failed ({st})")
        exec_id = info["Id"]
        sock, first = self.engine.hijack(f"/exec/{exec_id}/start", {"Detach": False, "Tty": tty})
        return exec_id, sock, first

    def exit_code(self, exec_id: str) -> tuple[bool, int | None, int | None]:
        st, info = self.engine.call("GET", f"/exec/{exec_id}/json", timeout=10)
        if st != 200 or not isinstance(info, dict):
            return False, None, None
        return bool(info.get("Running")), info.get("ExitCode"), info.get("Pid")

    def run(self, container_or_cid: str, cmd: list[str], *, timeout: float = 600,
            on_output=None, cancel: threading.Event | None = None) -> tuple[int, bool]:
        """Run `cmd` to its end; (exit code, timed out). Output goes to `on_output(text)`."""
        cid = container_or_cid
        b = self.box(cid)
        with b.lock:                     # run threads vs reader threads: keep the count exact
            b.busy += 1
        b.last_used = self.clock()
        try:
            exec_id, sock, first = self.open_exec(cid, ["setsid", "--wait", *cmd])
            done = threading.Event()

            def pump():
                try:
                    demux(sock, first, lambda s, d: on_output and on_output(
                        d.decode("utf-8", "replace")))
                finally:
                    # a demux that drops the stream (frame cap) still ends the run
                    done.set()
            threading.Thread(target=pump, daemon=True).start()
            t0, timed_out = time.monotonic(), False
            while not done.wait(0.2):
                if time.monotonic() - t0 > timeout or (cancel is not None and cancel.is_set()):
                    timed_out = time.monotonic() - t0 > timeout
                    self._kill_exec(exec_id)
                    done.wait(5)
                    break
            try:
                sock.close()
            except OSError:
                pass
            for _ in range(20):
                running, code, _ = self.exit_code(exec_id)
                if not running:
                    return (code if code is not None else -1), timed_out
                time.sleep(0.1)
            return -1, timed_out
        finally:
            with b.lock:
                b.busy -= 1
            b.last_used = self.clock()

    def _kill_exec(self, exec_id: str) -> None:
        """The exec's process group (setsid made it a leader): TERM, then KILL. Without user
        namespaces its pid is a pid of this host, owned by chatd's user. Between reading the
        exec's pid and killpg'ing it, the exec can exit and the pid be recycled into an
        unrelated host process: /proc/<pid>'s starttime is pinned while the exec still runs
        and re-checked before each signal, so a recycled pid is left alone."""
        running, _, pid = self.exit_code(exec_id)
        if not running or not pid:
            return
        born = _proc_starttime(pid)
        if born is None:
            return                       # the exec already exited; nothing to signal
        import signal
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if _proc_starttime(pid) != born:
                return                   # the pid is someone else's process now
            try:
                os.killpg(pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(2)

    def end_exec(self, exec_id: str) -> None:
        """TERM to an exec's own process (not its group) when it still runs."""
        running, _, pid = self.exit_code(exec_id)
        if running and pid:
            import signal
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    # ------------------------------------------------------------------ the guard
    def guard_once(self) -> str | None:
        """One look: what it did, for the tests and the log."""
        avail = self.meminfo().get("MemAvailable", 0) / GiB
        running = self._running()
        if not avail or not running:
            return None
        if avail < KILL_GB:
            self.kill_all("memory_guard")
            return "kill"
        if avail < STOP_GB:
            # the engine first: what it holds idle goes before a sandbox does
            if self._ask_engine(STOP_GB + LIMIT_BYTES / GiB - avail) and \
                    self._avail_gb() >= STOP_GB:
                return "engine released"
            idle = sorted((b for b in running if not b.busy), key=lambda b: b.last_used)
            victim = idle[0] if idle else sorted(running, key=lambda b: b.last_used)[0]
            self.stop(victim.cid, "memory_guard")
            return f"stop {victim.cid}"
        now = self.clock()
        for b in running:
            if not b.busy and now - b.last_used > IDLE_S:
                self.stop(b.cid, "idle")
                return f"idle {b.cid}"
        return None

    def guard(self) -> None:
        while not self.stopping.wait(1.0):
            try:
                self.guard_once()
            except Exception as exc:                                # noqa: BLE001
                self.log(f"[sandbox] guard {type(exc).__name__}")

    def touch(self, cid: str) -> None:
        self.box(cid).last_used = self.clock()


def _du(path: str) -> int:
    """The bytes of the files under `path`, a tree the sandbox writes into: os.fwalk walks by
    fds, so a folder swapped for a symlink mid-walk is not entered (os.walk would list what it
    points to, the host's / included)."""
    total = 0
    if not os.path.isdir(path):
        return 0
    for _, _, files, dirfd in os.fwalk(path, follow_symlinks=False):
        for f in files:
            try:
                total += os.stat(f, dir_fd=dirfd, follow_symlinks=False).st_size
            except OSError:
                pass
    return total
