"""chatd as an MCP client (SRV-359; plan note Chat 3). JSON-RPC 2.0, protocol revision 2025-06-18,
the standard library only.

    initialize, notifications/initialized, tools/list (with its cursor), tools/call (with a
    progress token; `notifications/progress` reach the tool card's output), notifications/
    cancelled (a call past its timeout or a Stop), notifications/tools/list_changed (the tools are
    listed again), ping both ways

TRANSPORTS.
- stdio: the server's command runs INSIDE the chat's sandbox (an exec with stdin; one JSON message
  a line), never on the host: it is a program someone else wrote, and on the host it would read
  the weights, the secrets and the LAN. A stdio server enabled for all chats starts in each chat's
  own sandbox when that chat first uses it.
- streamable HTTP: POST the message, the answer is JSON or an SSE stream; `Mcp-Session-Id` from
  initialize on every later request. From the host, so a server on the LAN the user typed in works.

SERVERS live in the `mcp_servers` table (v3 adds their arguments, scope, headers, timeout and the
last tool list). Env values and header values are write-only: the API answers every value as "",
and the store keeps them encrypted with the proxy's master key (proxy/secrets.py, AES-256-GCM,
`~/.tandemllm/secrets.key`), the server's id as associated data. They go into that server's
process (stdio) or requests (HTTP) and nowhere else: no log line, event or answer carries them.

TOOLS reach the model as `mcp__<server>__<tool>` (letters, digits and _ only), after the server's
allow-list and deny-list (a 10-tool allow-list beat 37 schemas for the x1 reply model). Their
default permission is ask; `mcp__<server>__*` rules allow a whole server. The last list is kept in
the store, so building a request never starts a server; the first call does.
"""

from __future__ import annotations

import base64
import http.client
import itertools
import json
import os
import re
import threading
import time
import urllib.parse

from chatd.ids import iso, now_ms, ulid
from chatd.sandbox import demux
from chatd.tools import Result, Tool, ToolError

PROTOCOL = "2025-06-18"
CLIENT = {"name": "tandemllm-chatd", "version": "0.1"}
DEFAULT_TIMEOUT_S = 120.0
MAX_BODY = 64 << 20       # one JSON answer over this is refused BY NAME, never truncated
                         # into a json.loads failure: a screenshot result is a few MB,
                         # the cap has to clear the work, not just the hostility
MAX_SSE_LINE = 64 << 20   # the same cap per SSE line — one line IS one answer there

class OverCap(ValueError):
    """A response over the named cap: reported as 'answer over N MiB', not as
    'cannot reach the server' — the server was reached, the answer is too big."""


def _sse_line(r, cap: int) -> bytes | None:
    """One COMPLETE SSE line, at most `cap` bytes (None at a clean EOF). readline(amt)
    returns a chunk, not a line: a 1.5 MB data line read once comes back as its first
    megabyte, which parses as no field at all while the remainder reads as a second
    half-line and drops as junk — the answer is lost and the caller waits out its
    timeout for nothing. So: keep reading to the newline, and fail by name at the cap."""
    parts: list[bytes] = []
    got = 0
    while True:
        chunk = r.readline(cap + 1 - got)
        if not chunk:
            break
        parts.append(chunk)
        got += len(chunk)
        if got > cap:
            raise OverCap(f"answer over {cap >> 20} MiB")
        if chunk.endswith(b"\n"):
            return b"".join(parts)
    if not parts:
        return None
    return b"".join(parts)

class McpError(Exception):
    pass


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", (name or "").lower()).strip("_")
    return s[:24] or "server"


# ---------------------------------------------------------------------- the protocol
class Session:
    """One connection: requests by id, the server's notifications and requests handled."""

    def __init__(self, transport, on_list_changed=None, log=None):
        self.t = transport
        self.ids = itertools.count(1)
        self.waiting: dict[int, dict] = {}
        self.lock = threading.Lock()
        self.on_list_changed = on_list_changed
        self.log = log or (lambda line: None)
        self.progress: dict[str, object] = {}
        self.alive = True
        self.info: dict = {}
        self.t.on_message = self._incoming
        self.t.on_close = self._closed

    def _closed(self) -> None:
        self.alive = False
        with self.lock:
            for w in self.waiting.values():
                w["error"] = "the server closed the connection"
                w["event"].set()

    def _incoming(self, msg: dict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg):
            with self.lock:
                w = self.waiting.get(msg["id"])
            if w is not None:
                w["msg"] = msg
                w["event"].set()
            return
        method = msg.get("method")
        if method and "id" in msg:                       # a request from the server
            result = {} if method == "ping" else {"roots": []} if method == "roots/list" else None
            if result is None:
                self.t.send({"jsonrpc": "2.0", "id": msg["id"],
                             "error": {"code": -32601, "message": "not supported"}})
            else:
                self.t.send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
            return
        if method == "notifications/tools/list_changed" and self.on_list_changed:
            threading.Thread(target=self.on_list_changed, daemon=True).start()
        elif method == "notifications/progress":
            p = msg.get("params") or {}
            cb = self.progress.get(str(p.get("progressToken")))
            if cb is not None:
                cb(p)

    def request(self, method: str, params: dict | None = None, timeout: float = 30.0,
                cancel: threading.Event | None = None) -> dict:
        rid = next(self.ids)
        w = {"event": threading.Event(), "msg": None, "error": None}
        with self.lock:
            self.waiting[rid] = w
        try:
            self.t.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            t0 = time.monotonic()
            while not w["event"].wait(0.2):
                if not self.alive:
                    break
                if (cancel is not None and cancel.is_set()) or time.monotonic() - t0 > timeout:
                    self.notify("notifications/cancelled",
                                {"requestId": rid, "reason": "cancelled" if cancel is not None
                                 and cancel.is_set() else "timeout"})
                    raise McpError("cancelled" if cancel is not None and cancel.is_set()
                                   else f"no answer within {int(timeout)} s; the call was "
                                        f"cancelled")
            if w["error"] or w["msg"] is None:
                raise McpError(w["error"] or "the server closed the connection")
            m = w["msg"]
            if "error" in m:
                e = m["error"] or {}
                raise McpError(f"{e.get('message') or 'error'} ({e.get('code')})")
            return m.get("result") or {}
        finally:
            with self.lock:
                self.waiting.pop(rid, None)

    def notify(self, method: str, params: dict | None = None) -> None:
        try:
            self.t.send({"jsonrpc": "2.0", "method": method, "params": params or {}})
        except OSError:
            pass

    def initialize(self, timeout: float = 30.0) -> dict:
        r = self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                        "clientInfo": CLIENT}, timeout=timeout)
        self.info = r
        self.notify("notifications/initialized")
        return r

    def _paged(self, method: str, key: str, need: str, timeout: float) -> list[dict]:
        out, cursor = [], None
        for _ in range(50):
            r = self.request(method, {"cursor": cursor} if cursor else {}, timeout=timeout)
            out += [t for t in r.get(key) or [] if isinstance(t, dict) and t.get(need)]
            cursor = r.get("nextCursor")
            if not cursor:
                break
        return out

    def list_tools(self, timeout: float = 30.0) -> list[dict]:
        return self._paged("tools/list", "tools", "name", timeout)

    def offers(self, what: str) -> bool:
        """Whether the server said it has prompts or resources (its initialize answer)."""
        return what in ((self.info or {}).get("capabilities") or {})

    def list_prompts(self, timeout: float = 30.0) -> list[dict]:
        return self._paged("prompts/list", "prompts", "name", timeout) if \
            self.offers("prompts") else []

    def list_resources(self, timeout: float = 30.0) -> list[dict]:
        return self._paged("resources/list", "resources", "uri", timeout) if \
            self.offers("resources") else []

    def get_prompt(self, name: str, args: dict, timeout: float = 30.0) -> str:
        """The prompt's messages as one text (MCP prompts become slash commands, Chat 3)."""
        r = self.request("prompts/get", {"name": name, "arguments": args}, timeout=timeout)
        out = []
        for m in r.get("messages") or []:
            c = m.get("content") or {}
            for part in c if isinstance(c, list) else [c]:
                if part.get("type") == "text":
                    out.append(part.get("text") or "")
                elif part.get("type") == "resource":
                    out.append((part.get("resource") or {}).get("text") or "")
        return "\n\n".join(x for x in out if x)

    def read_resource(self, uri: str, timeout: float = 30.0) -> list[dict]:
        return self.request("resources/read", {"uri": uri}, timeout=timeout).get("contents") or []

    def call(self, name: str, args: dict, timeout: float, cancel=None, on_progress=None) -> dict:
        token = ulid()
        if on_progress is not None:
            self.progress[token] = on_progress
        try:
            return self.request("tools/call", {"name": name, "arguments": args,
                                               "_meta": {"progressToken": token}},
                                timeout=timeout, cancel=cancel)
        finally:
            self.progress.pop(token, None)

    def close(self) -> None:
        self.alive = False
        self.t.close()


# ---------------------------------------------------------------------- transports
class StdioTransport:
    """Newline-delimited JSON over a process's stdin and stdout: `open_pipe()` gives
    (write(bytes), a reader that calls on_line per stdout line, close())."""

    def __init__(self, open_pipe):
        self.on_message = None
        self.on_close = None
        self.write, self.start_reader, self.closer = open_pipe(self._line)
        self.wlock = threading.Lock()
        self.start_reader()

    def _line(self, line: bytes | None) -> None:
        if line is None:
            if self.on_close:
                self.on_close()
            return
        try:
            msg = json.loads(line)
        except ValueError:
            return
        if isinstance(msg, dict) and self.on_message:
            self.on_message(msg)

    def send(self, msg: dict) -> None:
        with self.wlock:
            self.write(json.dumps(msg, separators=(",", ":")).encode() + b"\n")

    def close(self) -> None:
        self.closer()


def exec_pipe(sandboxes, cid: str, command: str, args: list[str], env: dict):
    """open_pipe for a stdio server inside the chat's sandbox."""
    def open_pipe(on_line):
        sandboxes.ensure(cid)
        envs = [f"{k}={v}" for k, v in env.items()]
        exec_id, sock, first = sandboxes.open_exec(cid, [command, *args], stdin=True, env=envs)
        buf = {"b": b""}

        def data(stream, chunk):
            if stream != 1:
                return
            buf["b"] += chunk
            while b"\n" in buf["b"]:
                ln, buf["b"] = buf["b"].split(b"\n", 1)
                if ln.strip():
                    on_line(ln)

        def reader():
            # box() may raise (a chat gone mid-connect): even then on_line(None) must run, or
            # the pending request hangs until timeout and the socket leaks
            try:
                box = sandboxes.box(cid)
                with box.lock:               # run threads vs reader threads: keep the count exact
                    box.busy += 1
                try:
                    demux(sock, first, data)
                finally:
                    with box.lock:
                        box.busy -= 1
            finally:
                on_line(None)
        return (sock.sendall, lambda: threading.Thread(target=reader, daemon=True).start(),
                lambda: sock.close())
    return open_pipe


def process_pipe(argv: list[str], env: dict | None = None):
    """open_pipe for a local process (the tests' fake stdio server)."""
    import subprocess

    def open_pipe(on_line):
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, env={**os.environ, **(env or {})})

        def reader():
            for ln in p.stdout:
                if ln.strip():
                    on_line(ln)
            on_line(None)

        def write(b):
            p.stdin.write(b)
            p.stdin.flush()
        return write, lambda: threading.Thread(target=reader, daemon=True).start(), p.kill
    return open_pipe


class HttpTransport:
    """Streamable HTTP: each message is a POST; the answer is JSON or an SSE stream whose events
    are messages (the response, and any notification before it)."""

    def __init__(self, url: str, headers: dict, timeout: float = 120.0):
        self.u = urllib.parse.urlsplit(url)
        if self.u.scheme not in ("http", "https") or not self.u.hostname:
            raise McpError("the URL must start with http:// or https://")
        self.headers = dict(headers)
        self.timeout = timeout
        self.session_id = None
        self.on_message = None
        self.on_close = None

    def _conn(self):
        cls = http.client.HTTPSConnection if self.u.scheme == "https" else \
            http.client.HTTPConnection
        return cls(self.u.hostname, self.u.port, timeout=self.timeout)

    def send(self, msg: dict) -> None:
        threading.Thread(target=self._post, args=(msg,), daemon=True).start()

    def _post(self, msg: dict) -> None:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
             "MCP-Protocol-Version": PROTOCOL, **self.headers}
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        conn = self._conn()
        try:
            path = (self.u.path or "/") + (f"?{self.u.query}" if self.u.query else "")
            conn.request("POST", path, body=json.dumps(msg).encode(), headers=h)
            r = conn.getresponse()
            sid = r.getheader("Mcp-Session-Id")
            if sid:
                self.session_id = sid
            ctype = (r.getheader("Content-Type") or "").lower()
            if r.status in (202, 204):
                r.read(MAX_BODY)
                return
            if r.status >= 400:
                body = r.read(2000).decode("utf-8", "replace")
                if "id" in msg and self.on_message:
                    self.on_message({"jsonrpc": "2.0", "id": msg["id"], "error": {
                        "code": r.status, "message": f"HTTP {r.status}: {body[:200]}"}})
                return
            if "text/event-stream" in ctype:
                data: list[str] = []
                while True:
                    raw = _sse_line(r, MAX_SSE_LINE)
                    if raw is None:
                        break
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if line == "":
                        if data and self.on_message:
                            try:
                                self.on_message(json.loads("\n".join(data)))
                            except ValueError:
                                pass
                        data = []
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
            else:
                raw = r.read(MAX_BODY + 1)
                # r.read(amt) truncates silently: a json.loads failure was the silence
                if len(raw) > MAX_BODY:
                    raise OverCap(f"answer over {MAX_BODY >> 20} MiB")
                if raw.strip() and self.on_message:
                    obj = json.loads(raw)
                    for m in obj if isinstance(obj, list) else [obj]:
                        self.on_message(m)
        except OverCap as exc:
            if "id" in msg and self.on_message:
                self.on_message({"jsonrpc": "2.0", "id": msg["id"],
                                 "error": {"code": -32000, "message": str(exc)}})
        except (OSError, http.client.HTTPException, ValueError) as exc:
            if "id" in msg and self.on_message:
                self.on_message({"jsonrpc": "2.0", "id": msg["id"],
                                 "error": {"code": -32000,
                                           "message": f"cannot reach the server "
                                                      f"({type(exc).__name__})"}})
        finally:
            conn.close()

    def close(self) -> None:
        self.on_message = None


# ---------------------------------------------------------------------- the servers
class Mcp:
    def __init__(self, svc, key_dir: str):
        self.svc = svc
        self.store = svc.store
        self.key_dir = key_dir
        self.sessions: dict[tuple, Session] = {}
        self.lock = threading.Lock()
        self.stdio_pipe = None                   # tests: (server row, cid) -> open_pipe
        self.tried: set = set()                  # (server, chat) whose first listing was tried
        self._vault = None

    def start(self) -> None:
        pass

    # ---- secrets
    def vault(self):
        if self._vault is None:
            try:
                from proxy.secrets import Vault
                self._vault = Vault.load(self.key_dir)
            except Exception as exc:                                # noqa: BLE001
                raise McpError(f"secrets cannot be stored here ({type(exc).__name__}); install "
                               f"the `cryptography` package") from None
        return self._vault

    def _seal(self, sid: str, values: dict) -> bytes | None:
        if not values:
            return None
        ct, nonce, kid = self.vault().encrypt(sid, json.dumps(values))
        return json.dumps({"ct": base64.b64encode(ct).decode(),
                           "nonce": base64.b64encode(nonce).decode(), "kid": kid}).encode()

    def _open(self, sid: str, blob) -> dict:
        if not blob:
            return {}
        d = json.loads(blob)
        return json.loads(self.vault().decrypt(sid, base64.b64decode(d["ct"]),
                                               base64.b64decode(d["nonce"]), d["kid"]))

    # ---- the API's shape
    def _row(self, sid: str):
        r = self.store.one("SELECT * FROM mcp_servers WHERE id = ?", (sid,))
        if r is None:
            from chatd.service import ApiError
            raise ApiError(404, "not_found", "no such MCP server")
        return r

    def public(self, r) -> dict:
        tools = json.loads(r["tools_json"]) if r["tools_json"] else []
        allow = json.loads(r["tool_allow"]) if r["tool_allow"] else None
        deny = json.loads(r["tool_deny"]) if r["tool_deny"] else []
        shown = self._filter(tools, allow, deny)
        return {"id": r["id"], "name": r["name"], "transport": r["transport"],
                "command": r["command"], "args": json.loads(r["args"]) if r["args"] else [],
                "url": r["url"],
                "env": {k: "" for k in json.loads(r["env_keys"] or "[]")},
                "headers": {k: "" for k in json.loads(r["header_keys"] or "[]")},
                "status": r["status"] or "stopped", "error": r["error"],
                "tools": [{"name": t["name"], "description": (t.get("description") or "")[:300]}
                          for t in tools],
                "tool_allow": allow, "tool_deny": deny,
                "prompts": [{"name": p["name"], "description": (p.get("description") or "")[:300],
                             "arguments": [{"name": a.get("name"), "required": bool(a.get("required"))}
                                           for a in p.get("arguments") or [] if a.get("name")]}
                            for p in json.loads(r["prompts_json"] or "[]")],
                "resources": [{"uri": x["uri"], "name": x.get("name") or x["uri"],
                               "mime": x.get("mimeType")}
                              for x in json.loads(r["resources_json"] or "[]")][:200],
                "schema_tokens": sum(len(json.dumps(t)) for t in shown) // 4,
                "enabled_global": bool(r["enabled_global"]),
                "timeout_s": r["timeout_s"] or DEFAULT_TIMEOUT_S,
                "updated_at": iso(r["updated"])}

    def list(self) -> list[dict]:
        return [self.public(r) for r in self.store.all("SELECT * FROM mcp_servers ORDER BY name")]

    def _check(self, d: dict) -> dict:
        from chatd.service import ApiError
        t = d.get("transport")
        if t not in ("stdio", "http"):
            raise ApiError(400, "bad_request", "transport is stdio or http")
        if t == "stdio" and not str(d.get("command") or "").strip():
            raise ApiError(400, "bad_request", "a stdio server needs a command")
        if t == "http" and not re.match(r"^https?://", str(d.get("url") or "")):
            raise ApiError(400, "bad_request", "the URL must start with http:// or https://")
        if not str(d.get("name") or "").strip():
            raise ApiError(400, "bad_request", "a name is needed")
        return d

    def add(self, d: dict) -> dict:
        self._check(d)
        sid = ulid()
        env = {k: str(v) for k, v in (d.get("env") or {}).items() if v}
        hdr = {k: str(v) for k, v in (d.get("headers") or {}).items() if v}
        self.store.run(
            "INSERT INTO mcp_servers (id, name, transport, command, url, env_enc, enabled_global, "
            "tool_allow, tool_deny, args, headers_enc, env_keys, header_keys, timeout_s, updated, "
            "status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'starting')",
            (sid, str(d["name"]).strip()[:60], d["transport"], d.get("command"), d.get("url"),
             self._seal(sid, env), 1 if d.get("enabled_global", True) else 0,
             json.dumps(d["tool_allow"]) if d.get("tool_allow") is not None else None,
             json.dumps(d.get("tool_deny") or []), json.dumps(d.get("args") or []),
             self._seal(sid, hdr), json.dumps(sorted(d.get("env") or {})),
             json.dumps(sorted(d.get("headers") or {})),
             float(d.get("timeout_s") or DEFAULT_TIMEOUT_S), now_ms()))
        self.svc.log(f"[mcp] server {sid} added ({d['transport']})")
        if d["transport"] == "http":
            threading.Thread(target=self.refresh, args=(sid, None), daemon=True).start()
        else:
            self.store.run("UPDATE mcp_servers SET status = 'stopped' WHERE id = ?", (sid,))
        return self.public(self._row(sid))

    def patch(self, sid: str, d: dict) -> dict:
        r = self._row(sid)
        sets, args = [], []
        for k in ("name", "command", "url"):
            if k in d:
                sets.append(f"{k} = ?")
                args.append(d[k])
        if "args" in d:
            sets.append("args = ?")
            args.append(json.dumps(d["args"] or []))
        if "enabled_global" in d:
            sets.append("enabled_global = ?")
            args.append(1 if d["enabled_global"] else 0)
        if "tool_allow" in d:
            sets.append("tool_allow = ?")
            args.append(json.dumps(d["tool_allow"]) if d["tool_allow"] is not None else None)
        if "tool_deny" in d:
            sets.append("tool_deny = ?")
            args.append(json.dumps(d["tool_deny"] or []))
        for field, keys, col in (("env", "env_keys", "env_enc"),
                                 ("headers", "header_keys", "headers_enc")):
            if field in d:
                old = self._open(sid, r[col])
                new = {k: (str(v) if v else old.get(k, "")) for k, v in (d[field] or {}).items()}
                new = {k: v for k, v in new.items() if v}
                sets += [f"{col} = ?", f"{keys} = ?"]
                args += [self._seal(sid, new), json.dumps(sorted(d[field] or {}))]
        if sets:
            sets.append("updated = ?")
            args += [now_ms(), sid]
            self.store.run(f"UPDATE mcp_servers SET {', '.join(sets)} WHERE id = ?", args)
            self.drop(sid)
        return self.public(self._row(sid))

    def remove(self, sid: str) -> None:
        self._row(sid)
        self.drop(sid)
        self.store.run("DELETE FROM mcp_servers WHERE id = ?", (sid,))

    def drop(self, sid: str) -> None:
        with self.lock:
            for k in [k for k in self.sessions if k[0] == sid]:
                self.sessions.pop(k).close()

    # ---- connections
    def _transport(self, r, cid: str | None):
        if r["transport"] == "http":
            return HttpTransport(r["url"], self._open(r["id"], r["headers_enc"]),
                                 r["timeout_s"] or DEFAULT_TIMEOUT_S)
        if cid is None:
            raise McpError("a stdio server runs in a chat's sandbox; use it in a chat")
        args = json.loads(r["args"]) if r["args"] else []
        env = self._open(r["id"], r["env_enc"])
        pipe = (self.stdio_pipe(r, cid) if self.stdio_pipe is not None else
                exec_pipe(self.svc.sandboxes, cid, r["command"], args, env))
        return StdioTransport(pipe)

    def session(self, sid: str, cid: str | None) -> Session:
        r = self._row(sid)
        key = (sid, cid if r["transport"] == "stdio" else None)
        with self.lock:
            s = self.sessions.get(key)
            if s is not None and s.alive:
                return s
        s = Session(self._transport(r, cid), on_list_changed=lambda: self.refresh(sid, cid),
                    log=self.svc.log)
        try:
            s.initialize(timeout=min(60.0, r["timeout_s"] or DEFAULT_TIMEOUT_S))
        except McpError as exc:
            s.close()
            self._status(sid, "failed", str(exc))
            raise
        with self.lock:
            self.sessions[key] = s
        return s

    def _status(self, sid: str, status: str, error: str | None = None, tools=None) -> None:
        if tools is not None:
            self.store.run("UPDATE mcp_servers SET status = ?, error = ?, tools_json = ?, "
                           "updated = ? WHERE id = ?",
                           (status, error, json.dumps(tools), now_ms(), sid))
        else:
            self.store.run("UPDATE mcp_servers SET status = ?, error = ?, updated = ? WHERE id = ?",
                           (status, error, now_ms(), sid))

    def refresh(self, sid: str, cid: str | None) -> dict:
        try:
            s = self.session(sid, cid)
            tools = s.list_tools()
            self.store.run("UPDATE mcp_servers SET prompts_json = ?, resources_json = ? WHERE id = ?",
                           (json.dumps(s.list_prompts()), json.dumps(s.list_resources()), sid))
            self._status(sid, "connected", None, tools)
        except McpError as exc:
            self._status(sid, "failed", str(exc)[:300])
        return self.public(self._row(sid))

    def prompt_text(self, sid: str, name: str, args: dict, cid: str | None) -> str:
        from chatd.service import ApiError
        try:
            return self.session(sid, cid).get_prompt(name, {k: str(v) for k, v in args.items()})
        except McpError as exc:
            raise ApiError(502, "mcp", f"the MCP server answered: {exc}") from None

    def resource(self, sid: str, uri: str, cid: str | None) -> dict:
        """A resource as an attachment of the chat (text, or bytes from a blob)."""
        from chatd.service import ApiError
        try:
            contents = self.session(sid, cid).read_resource(uri)
        except McpError as exc:
            raise ApiError(502, "mcp", f"the MCP server answered: {exc}") from None
        if not contents:
            raise ApiError(404, "not_found", "the server sent nothing for that resource")
        c = contents[0]
        name = (c.get("uri") or uri).rstrip("/").rsplit("/", 1)[-1][:120] or "resource"
        if c.get("blob") is not None:
            data, mime = base64.b64decode(c["blob"]), c.get("mimeType") or "application/octet-stream"
        else:
            data, mime = str(c.get("text") or "").encode(), c.get("mimeType") or "text/plain"
        return self.svc.store_blob(None, name, mime, data)

    def restart(self, sid: str) -> dict:
        self.drop(sid)
        r = self._row(sid)
        if r["transport"] == "http":
            return self.refresh(sid, None)
        self._status(sid, "stopped")
        return self.public(self._row(sid))

    def test(self, d: dict, cid: str | None = None) -> dict:
        """Connect with a draft, list its tools, say how it went; nothing is stored."""
        self._check(d)
        fake = {"id": "test-" + ulid(), "transport": d["transport"], "url": d.get("url"),
                "command": d.get("command"), "args": json.dumps(d.get("args") or []),
                "timeout_s": float(d.get("timeout_s") or 30), "env_enc": None,
                "headers_enc": None}
        t = None
        s = None
        try:
            if d["transport"] == "http":
                t = HttpTransport(d["url"], {k: v for k, v in (d.get("headers") or {}).items()
                                             if v}, 30)
            else:
                if cid is None:
                    return {"ok": False, "tools": [], "message": "a stdio server is tested in a "
                                                                 "chat's sandbox: add it, then "
                                                                 "use it in a chat"}
                pipe = (self.stdio_pipe(fake, cid) if self.stdio_pipe is not None else
                        exec_pipe(self.svc.sandboxes, cid, d["command"], d.get("args") or [],
                                  {k: v for k, v in (d.get("env") or {}).items() if v}))
                t = StdioTransport(pipe)
            s = Session(t)
            info = s.initialize(timeout=30)
            tools = s.list_tools()
            s.close()
            return {"ok": True, "tools": [{"name": x["name"], "description":
                                           (x.get("description") or "")[:300]} for x in tools],
                    "message": f"handshake ok: protocol {info.get('protocolVersion')}, "
                               f"{len(tools)} tool{'s' if len(tools) != 1 else ''}"}
        except (McpError, OSError) as exc:
            # a stdio draft spawned a server into the chat's sandbox: a failed test that
            # never closes the transport leaks that server, its socket and the reader
            # thread's busy-pin on the sandbox forever (the pin holds off its idle-stop).
            # session() closes on the same failure (SRV-359); test() must too.
            if s is not None:
                s.close()
            elif t is not None:
                t.close()
            return {"ok": False, "tools": [], "message": str(exc)[:300]}

    # ---- the tools for the model
    @staticmethod
    def _filter(tools: list[dict], allow, deny) -> list[dict]:
        out = [t for t in tools if allow is None or t["name"] in allow]
        return [t for t in out if t["name"] not in (deny or [])]

    def _tools_of(self, r) -> list[Tool]:
        tools = json.loads(r["tools_json"]) if r["tools_json"] else []
        allow = json.loads(r["tool_allow"]) if r["tool_allow"] else None
        deny = json.loads(r["tool_deny"]) if r["tool_deny"] else []
        out = []
        for t in self._filter(tools, allow, deny):
            name = f"mcp__{slug(r['name'])}__{re.sub(r'[^A-Za-z0-9_-]', '_', t['name'])}"[:64]
            schema = t.get("inputSchema") if isinstance(t.get("inputSchema"), dict) else \
                {"type": "object", "properties": {}}
            # the server's own annotation (MCP 2025-03-26): a tool marked readOnlyHint does not
            # change anything, so auto mode runs it; every other tool counts as a write
            ann = t.get("annotations") if isinstance(t.get("annotations"), dict) else {}
            out.append(Tool(name, (t.get("description") or "")[:1024], schema, risk="network",
                            default="ask", source={"kind": "mcp", "id": r["id"]},
                            executor=self._executor(r["id"], t["name"]),
                            side_effects=ann.get("readOnlyHint") is not True))
        return out

    def all_tools(self) -> list[Tool]:
        return [t for r in self.store.all("SELECT * FROM mcp_servers ORDER BY name")
                for t in self._tools_of(r)]

    def tools_for(self, settings: dict, cid: str | None = None) -> list[Tool]:
        """The chat's MCP tools. A stdio server whose tools were never listed is started in this
        chat's sandbox and listed once (it needs the sandbox for its first call anyway)."""
        chosen = set(settings.get("mcp") or [])
        rows = [r for r in self.store.all("SELECT * FROM mcp_servers ORDER BY name")
                if r["enabled_global"] or r["id"] in chosen]
        for r in rows:
            if cid and r["transport"] == "stdio" and not r["tools_json"] and \
                    (r["id"], cid) not in self.tried:
                self.tried.add((r["id"], cid))
                self.refresh(r["id"], cid)
        rows = [self._row(r["id"]) for r in rows]
        return [t for r in rows for t in self._tools_of(r)]

    def tool(self, name: str):
        return next((t for t in self.all_tools() if t.name == name), None)

    def _executor(self, sid: str, tool_name: str):
        def run(ctx, args):
            r = self._row(sid)
            try:
                s = self.session(sid, ctx.cid)
                res = s.call(tool_name, args, timeout=r["timeout_s"] or DEFAULT_TIMEOUT_S,
                             cancel=ctx.run.cancel,
                             on_progress=lambda p: ctx.output(
                                 f"progress {p.get('progress')}"
                                 f"{'/' + str(p.get('total')) if p.get('total') else ''} "
                                 f"{p.get('message') or ''}\n"))
            except McpError as exc:
                raise ToolError(f"the MCP server answered: {exc}") from None
            texts, images = [], []
            for c in res.get("content") or []:
                if c.get("type") == "text":
                    texts.append(c.get("text") or "")
                elif c.get("type") == "image" and c.get("data"):
                    images.append((c.get("mimeType") or "image/png", base64.b64decode(c["data"])))
                elif c.get("type") == "resource":
                    texts.append((c.get("resource") or {}).get("text") or "")
            if res.get("structuredContent") is not None and not texts:
                texts.append(json.dumps(res["structuredContent"]))
            return Result("\n".join(texts) or "(no content)", is_error=bool(res.get("isError")),
                          images=images)
        return run
