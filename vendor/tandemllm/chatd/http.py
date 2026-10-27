"""chatd's routes (CONTRACT section 2), SSE (section 3) and the forward key.

Every route answers under `/v1/chat-app/...` (the path the engine server forwards unchanged) and
needs `X-Tandem-Forward: <key>`; anything without it gets 403. `GET /health` is the one exception
(the watchdog's): it says `ok` and the version, nothing else.

The forward adds what only the engine server knows: `X-Tandem-Password-Set` and
`X-Tandem-Signed-In` (the capabilities' `session`), and passes the browser's User-Agent, from which
the snapshot names the device a run was started on.

Phase 2 routes: tools, packages, MCP, prompts, presets, permissions, compaction, the sandbox, its
files and terminals, artifacts, user skills, Open WebUI tools. On a box with no sandbox runtime,
starting a sandbox answers 503 `sandbox_unavailable` and bash, render_page and the terminals are
not offered; the file routes work on the chat's workspace folder.
"""

from __future__ import annotations

import hmac
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from chatd import __version__
from chatd.events import HEARTBEAT, HEARTBEAT_S, control, frame
from chatd.mcp import McpError
from chatd.service import ApiError, Service, device_label

PREFIX = "/v1/chat-app"
MAX_JSON = 1 << 20
MAX_UPLOAD = 51 * 1024 * 1024
FILE_CHUNK = 256 * 1024
_ID = r"([A-Za-z0-9_.:-]{1,80})"

# (method, pattern, handler name)
ROUTES = [
    ("GET", r"/capabilities", "capabilities"),
    ("GET", r"/session", "session"),
    ("GET", r"/conversations", "list"),
    ("POST", r"/conversations", "create"),
    ("GET", rf"/conversations/{_ID}", "get"),
    ("PATCH", rf"/conversations/{_ID}", "patch"),
    ("DELETE", rf"/conversations/{_ID}", "delete"),
    ("POST", rf"/conversations/{_ID}/restore", "restore"),
    ("POST", r"/folders", "folder"),
    ("GET", rf"/conversations/{_ID}/events", "events"),
    ("POST", rf"/conversations/{_ID}/messages", "send"),
    ("POST", rf"/conversations/{_ID}/queue", "queue"),
    ("DELETE", rf"/conversations/{_ID}/queue/{_ID}", "unqueue"),
    ("POST", rf"/conversations/{_ID}/queue/{_ID}/now", "send_now"),
    ("POST", rf"/conversations/{_ID}/stop", "stop"),
    ("POST", rf"/conversations/{_ID}/continue", "continue_"),
    ("POST", rf"/conversations/{_ID}/messages/{_ID}/regenerate", "regenerate"),
    ("POST", rf"/conversations/{_ID}/messages/{_ID}/edit", "edit"),
    ("POST", rf"/conversations/{_ID}/branch", "branch"),
    ("PATCH", rf"/conversations/{_ID}/settings", "settings"),
    # plan mode, the todo list, rewind (SRV-363)
    ("POST", rf"/conversations/{_ID}/plan/approve", "approve_plan"),
    ("GET", rf"/conversations/{_ID}/todos", "todos"),
    ("PUT", rf"/conversations/{_ID}/todos", "put_todos"),
    ("POST", rf"/conversations/{_ID}/rewind", "rewind"),
    # workers (VIS-E15)
    ("GET", rf"/conversations/{_ID}/tasks", "tasks"),
    ("GET", rf"/conversations/{_ID}/tasks/{_ID}", "task"),
    ("GET", r"/events", "global_events"),
    ("POST", r"/attachments", "upload"),
    ("GET", rf"/attachments/{_ID}", "attachment"),
    # Open WebUI tools (SRV-361)
    ("POST", r"/tools/import/owui", "import_owui"),
    ("GET", r"/tools/user", "user_tools"),
    ("PATCH", rf"/tools/user/{_ID}", "patch_user_tools"),
    ("DELETE", rf"/tools/user/{_ID}", "delete_user_tools"),
    # tools and permissions (SRV-353)
    ("GET", r"/tools", "tools"),
    ("PATCH", rf"/tools/{_ID}", "set_tool"),
    ("GET", r"/permissions/rules", "rules"),
    ("GET", r"/web-search", "web_search"),
    ("PUT", r"/web-search", "put_web_search"),
    ("POST", r"/web-search/test", "test_web_search"),
    ("PATCH", r"/defaults", "patch_defaults"),
    ("GET", r"/permissions/mode", "permission_modes"),
    ("PUT", r"/permissions/mode", "set_permission_default"),
    ("GET", rf"/conversations/{_ID}/permission-mode", "chat_permission_mode"),
    ("PUT", rf"/conversations/{_ID}/permission-mode", "set_chat_permission_mode"),
    ("DELETE", rf"/permissions/rules/{_ID}", "delete_rule"),
    ("POST", rf"/conversations/{_ID}/permissions/{_ID}", "permission"),
    ("POST", rf"/conversations/{_ID}/compact", "compact"),
    # the sandbox and the panes (SRV-355, SRV-357)
    ("GET", rf"/conversations/{_ID}/sandbox", "sandbox"),
    ("POST", rf"/conversations/{_ID}/sandbox/(start|stop|reset)", "sandbox_action"),
    ("GET", rf"/conversations/{_ID}/sandbox/files", "files"),
    ("GET", rf"/conversations/{_ID}/sandbox/file", "file"),
    ("PUT", rf"/conversations/{_ID}/sandbox/file", "put_file"),
    ("DELETE", rf"/conversations/{_ID}/sandbox/file", "delete_file"),
    ("POST", rf"/conversations/{_ID}/sandbox/upload", "upload_files"),
    ("GET", rf"/conversations/{_ID}/sandbox/diff", "diff"),
    ("POST", rf"/conversations/{_ID}/sandbox/undo", "undo"),
    ("GET", rf"/conversations/{_ID}/sandbox/view/(.+)", "view"),
    ("GET", rf"/conversations/{_ID}/sandbox/ports", "sandbox_ports"),
    ("*", rf"/conversations/{_ID}/sandbox/ports/(\d+)(/.*)?", "port"),
    ("POST", rf"/conversations/{_ID}/sandbox/serve", "serve"),
    ("POST", rf"/conversations/{_ID}/preview/report", "preview_report"),
    ("GET", rf"/conversations/{_ID}/terminals", "terminals"),
    ("POST", rf"/conversations/{_ID}/terminals", "new_terminal"),
    ("GET", rf"/conversations/{_ID}/terminals/([^/]+)/ws", "terminal_ws"),
    ("GET", rf"/conversations/{_ID}/artifacts/([^/]+)", "artifact"),
    # MCP servers (SRV-359)
    ("GET", r"/mcp/servers", "mcp_list"),
    ("POST", r"/mcp/servers/test", "mcp_test"),
    ("POST", r"/mcp/servers", "mcp_add"),
    ("POST", rf"/mcp/servers/{_ID}/restart", "mcp_restart"),
    ("PATCH", rf"/mcp/servers/{_ID}", "mcp_patch"),
    ("DELETE", rf"/mcp/servers/{_ID}", "mcp_remove"),
    ("POST", rf"/mcp/servers/{_ID}/prompts/get", "mcp_prompt"),
    ("POST", rf"/mcp/servers/{_ID}/resources/read", "mcp_resource"),
    # skills, prompts, presets (SRV-361)
    ("GET", r"/(packages|skills)", "packages"),
    ("PATCH", rf"/(?:packages|skills)/{_ID}", "set_package"),
    ("GET", r"/prompts", "prompts"),
    ("POST", r"/prompts", "save_prompt"),
    ("POST", rf"/prompts/{_ID}", "save_prompt"),
    ("DELETE", rf"/prompts/{_ID}", "delete_prompt"),
    ("GET", r"/presets", "presets"),
    ("POST", r"/presets", "save_preset"),
    ("POST", rf"/presets/{_ID}", "save_preset"),
    ("DELETE", rf"/presets/{_ID}", "delete_preset"),
    ("POST", r"/(?:packages/upload|skills/user)", "upload_skill"),
    ("DELETE", rf"/(?:packages|skills)/{_ID}", "delete_skill"),
]
_COMPILED = [(m, re.compile(p + "$"), n) for m, p, n in ROUTES]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "chatd"
    disable_nagle_algorithm = True            # an SSE frame goes out at once, not after 40 ms
    svc: Service = None                       # set by make_server
    key: str = ""

    def log_message(self, fmt, *args):        # ids and statuses only; never bodies
        pass

    # ------------------------------------------------------------------ plumbing
    def _send(self, code: int, body: bytes, ctype: str = "application/json",
              headers: tuple = ()) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _send_file(self, fh, meta: dict, headers: tuple = ()) -> None:
        """A file streamed in FILE_CHUNK pieces (VIS-159): the whole of it, or one byte range
        (206 with Content-Range; 416 for a range past the end). `If-Range` with another validator
        gets the whole file; `If-None-Match` with this one a 304. A file that shrank while it was
        read ends the connection, so the client sees a short body, not a hang."""
        from email.utils import formatdate
        from chatd.panes import UNSATISFIABLE, byte_range
        size, tag = meta["bytes"], meta["etag"]
        modified = formatdate(meta["mtime"], usegmt=True)
        common = [("Accept-Ranges", "bytes"), ("ETag", tag), ("Last-Modified", modified),
                  ("Cache-Control", "no-cache"), *headers]
        if tag in [t.strip() for t in (self.headers.get("If-None-Match") or "").split(",")]:
            return self._head(304, None, common, 0)
        want = self.headers.get("Range")
        cond = (self.headers.get("If-Range") or "").strip()
        if want and cond and cond not in (tag, modified):
            want = None
        rng = byte_range(want, size) if want else None
        if rng == UNSATISFIABLE:
            return self._head(416, meta["mime"], common + [("Content-Range", f"bytes */{size}")],
                              0)
        first, last = rng if rng else (0, size - 1)
        n = max(0, last - first + 1)
        extra = common + ([("Content-Range", f"bytes {first}-{last}/{size}")] if rng else [])
        if not self._head(206 if rng else 200, meta["mime"], extra, n):
            return
        try:
            fh.seek(first)
            while n > 0:
                chunk = fh.read(min(FILE_CHUNK, n))
                if not chunk:
                    self.close_connection = True
                    return
                self.wfile.write(chunk)
                n -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _head(self, code: int, ctype: str | None, headers, length: int) -> bool:
        try:
            self.send_response(code)
            if ctype:
                self.send_header("Content-Type", ctype)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(length))
            self.end_headers()
            return True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            return False

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    def _empty(self, code: int = 204) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_JSON:
            raise ApiError(413, "too_large", "the body is too large")
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
        except ValueError:
            raise ApiError(400, "bad_request", "the body is not JSON") from None
        if not isinstance(obj, dict):
            raise ApiError(400, "bad_request", "the body must be a JSON object")
        return obj

    @property
    def viewer(self) -> str | None:
        return device_label(self.headers.get("User-Agent"))

    def _authorized(self) -> bool:
        got = self.headers.get("X-Tandem-Forward") or ""
        return bool(self.key) and hmac.compare_digest(got.encode(), self.key.encode())

    def _dispatch(self, method: str) -> None:
        split = urlsplit(self.path)
        path = split.path
        if path in ("/health", "/healthz") and method == "GET":
            return self._json(200, {"status": "ok", "service": "chatd", "version": __version__})
        if not self._authorized():
            self.close_connection = True
            return self._json(403, {"error": {"type": "forbidden",
                                              "message": "chatd answers the engine server's "
                                                         "forward only"}})
        if path.startswith(PREFIX):
            path = path[len(PREFIX):]
        path = path.rstrip("/") or "/"
        self.query = {k: v[-1] for k, v in parse_qs(split.query).items()}
        for m, rx, name in _COMPILED:
            if m not in (method, "*"):
                continue
            hit = rx.match(path)
            if hit:
                try:
                    return getattr(self, "r_" + name)(*[unquote(g) if g else g
                                                         for g in hit.groups()])
                except ApiError as exc:
                    return self._json(exc.status, exc.body())
                except McpError as exc:
                    return self._json(400, {"error": {"type": "mcp", "message": str(exc)}})
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
                    return
                except Exception as exc:                           # noqa: BLE001
                    self.svc.log(f"[http] {method} {name} failed: {type(exc).__name__}")
                    return self._json(500, {"error": {"type": "internal",
                                                      "message": type(exc).__name__}})
        return self._json(404, {"error": {"type": "not_found", "message": f"no route {path}"}})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_PUT(self):
        self._dispatch("PUT")

    # ------------------------------------------------------------------ routes
    def r_capabilities(self):
        pw = (self.headers.get("X-Tandem-Password-Set") or "0") == "1"
        signed = (self.headers.get("X-Tandem-Signed-In") or "1") == "1"
        self._json(200, self.svc.capabilities(password_set=pw, signed_in=signed))

    def r_session(self):
        pw = (self.headers.get("X-Tandem-Password-Set") or "0") == "1"
        signed = (self.headers.get("X-Tandem-Signed-In") or "1") == "1"
        self._json(200, {"password_set": pw, "signed_in": signed, "sandbox": "not_installed"})

    def r_list(self):
        q = self.query
        self._json(200, self.svc.list(q.get("q", ""), q.get("archived", "0") in ("1", "true"),
                                      q.get("folder") or None))

    def r_create(self):
        self._json(200, self.svc.create(self._body(), self.viewer))

    def r_get(self, cid):
        self._json(200, self.svc.snapshot(cid, self.viewer))

    def r_patch(self, cid):
        self._json(200, self.svc.patch(cid, self._body()))

    def r_delete(self, cid):
        # ?forever=1: the purge now (the container and the chat's folders), not the trash
        if self.query.get("forever") in ("1", "true"):
            self.svc.purge(cid)
        else:
            self.svc.delete(cid)
        self._empty(204)

    def r_restore(self, cid):
        self._json(200, self.svc.restore(cid))

    def r_folder(self):
        self._json(200, self.svc.create_folder(str(self._body().get("name") or "")))

    def r_send(self, cid):
        self._json(202, self.svc.send(cid, self._body(), self.viewer))

    def r_queue(self, cid):
        self._json(202, self.svc.queue(cid, self._body(), self.viewer))

    def r_unqueue(self, cid, qid):
        self.svc.unqueue(cid, qid)
        self._empty(204)

    def r_send_now(self, cid, qid):
        self.svc.send_now(cid, qid, self.viewer)
        self._json(202, {})

    def r_stop(self, cid):
        self.svc.stop(cid)
        self._json(202, {})

    def r_continue_(self, cid):
        self._json(202, self.svc.continue_run(cid, self.viewer))

    def r_regenerate(self, cid, mid):
        self._body()
        self._json(202, self.svc.regenerate(cid, mid, self.viewer))

    def r_edit(self, cid, mid):
        self._json(202, self.svc.edit(cid, mid, self._body(), self.viewer))

    def r_branch(self, cid):
        self._json(200, self.svc.switch_branch(cid, self._body(), self.viewer))

    def r_settings(self, cid):
        # VIS-219: `reread_tokens`, what the switch of Agents mode makes the engine read once more
        self._json(200, {**self.svc.update_settings(cid, self._body()),
                         "reread_tokens": self.svc.reread_tokens(cid)})

    def r_mcp_prompt(self, sid):
        b = self._body()
        self._json(200, {"text": self.svc.mcp.prompt_text(
            sid, str(b.get("name") or ""), b.get("arguments") or {}, b.get("conversation_id"))})

    def r_mcp_resource(self, sid):
        b = self._body()
        self._json(200, self.svc.mcp.resource(sid, str(b.get("uri") or ""),
                                              b.get("conversation_id")))

    def r_approve_plan(self, cid):
        self._json(202, self.svc.approve_plan(cid, self._body(), self.viewer))

    def r_todos(self, cid):
        self.svc.conv(cid)
        self._json(200, self.svc.todos(cid))

    def r_put_todos(self, cid):
        self._json(200, self.svc.set_todos(cid, self._body().get("items")))

    def r_rewind(self, cid):
        self._json(200, self.svc.rewind(cid, self._body(), self.viewer))

    def r_tasks(self, cid):
        self._json(200, self.svc.list_tasks(cid))

    def r_task(self, cid, task_id):
        self._json(200, self.svc.get_task(cid, task_id))

    def r_upload(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ApiError(413, "too_large", "an attachment may be at most 50 MB")
        raw = self.rfile.read(n) if n else b""
        self._json(200, self.svc.upload(self.headers.get("Content-Type") or "", raw))

    def r_attachment(self, att_id):
        mime, data = self.svc.attachment(att_id)
        self._send(200, data, mime, (("Content-Disposition", "inline"),
                                     ("X-Content-Type-Options", "nosniff"),
                                     ("Content-Security-Policy", "default-src 'none'; sandbox")))

    def r_empty_list(self, *_):
        self._json(200, [])

    def _raw(self, limit: int) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ApiError(413, "too_large", f"at most {limit // (1024 * 1024)} MB")
        return self.rfile.read(n) if n else b""

    def r_upload_skill(self):
        raw = self._raw(11 * 1024 * 1024)
        self._json(200, self.svc.upload_skill(self.headers.get("Content-Type") or "", raw))

    def r_delete_skill(self, name):
        self.svc.delete_skill(name)
        self._send(204, b"")

    def r_import_owui(self):
        raw = self._raw(2 * 1024 * 1024)
        self._json(200, self.svc.import_owui(self.headers.get("Content-Type") or "", raw))

    def r_user_tools(self):
        self._json(200, self.svc.owui.list())

    def r_patch_user_tools(self, tid):
        self._json(200, self.svc.owui.patch(tid, self._body()))

    def r_delete_user_tools(self, tid):
        self.svc.owui.remove(tid)
        self._send(204, b"")

    # ---- skills, prompts, presets
    def r_packages(self, *_):
        self._json(200, self.svc.packages())

    def r_set_package(self, pid):
        self._json(200, self.svc.set_package(pid, self._body()))

    def r_prompts(self):
        self._json(200, self.svc.prompts())

    def r_save_prompt(self, pid=None):
        self._json(200, self.svc.save_prompt(pid, self._body()))

    def r_delete_prompt(self, pid):
        self.svc.delete_prompt(pid)
        self._empty(204)

    def r_presets(self):
        self._json(200, self.svc.presets())

    def r_save_preset(self, pid=None):
        self._json(200, self.svc.save_preset(pid, self._body()))

    def r_delete_preset(self, pid):
        self.svc.delete_preset(pid)
        self._empty(204)

    # ---- MCP servers
    def r_mcp_list(self):
        self._json(200, self.svc.mcp.list())

    def r_mcp_test(self):
        body = self._body()
        self._json(200, self.svc.mcp.test(body, body.get("conversation_id")))

    def r_mcp_add(self):
        self._json(200, self.svc.mcp.add(self._body()))

    def r_mcp_restart(self, sid):
        self._json(200, self.svc.mcp.restart(sid))

    def r_mcp_patch(self, sid):
        self._json(200, self.svc.mcp.patch(sid, self._body()))

    def r_mcp_remove(self, sid):
        self.svc.mcp.remove(sid)
        self._empty(204)

    # ---- tools and permissions
    def r_tools(self):
        self._json(200, self.svc.tool_infos())

    def r_set_tool(self, name):
        self._json(200, self.svc.set_tool(name, self._body()))

    def r_web_search(self):
        self._json(200, self.svc.web_search_view())

    def r_put_web_search(self):
        self._json(200, self.svc.set_web_search(self._body()))

    def r_test_web_search(self):
        self._json(200, self.svc.test_web_search(self._body()))

    def r_rules(self):
        self._json(200, self.svc.rules())

    def r_patch_defaults(self):
        self._json(200, self.svc.patch_defaults(self._body()))

    def r_permission_modes(self):
        self._json(200, self.svc.permission_modes())

    def r_set_permission_default(self):
        self._json(200, self.svc.set_permission_default(self._body()))

    def r_chat_permission_mode(self, cid):
        self._json(200, self.svc.chat_permission_mode(cid))

    def r_set_chat_permission_mode(self, cid):
        self._json(200, self.svc.set_chat_permission_mode(cid, self._body()))

    def r_delete_rule(self, rid):
        self.svc.delete_rule(rid)
        self._empty(204)

    def r_permission(self, cid, call_id):
        self.svc.answer_permission(cid, call_id, self._body())
        self._empty(204)

    def r_compact(self, cid):
        self.svc.compact_now(cid)
        self._json(202, {})

    # ---- the sandbox and the panes
    def r_sandbox(self, cid):
        self._json(200, self.svc.sandbox_status(cid))

    def r_sandbox_action(self, cid, action):
        self._json(200, self.svc.sandbox_action(cid, action))

    def r_files(self, cid):
        self._json(200, self.svc.files(cid, self.query.get("path") or "/workspace"))

    def r_file(self, cid):
        path = self.query.get("path") or ""
        if self.query.get("raw") in ("1", "true"):
            fh, meta = self.svc.open_file(cid, path)
            with fh:
                return self._send_file(fh, meta, (("X-Content-Type-Options", "nosniff"),
                                                  ("Content-Disposition", "attachment")))
        meta, _ = self.svc.file(cid, path)
        if "content" not in meta:
            meta["content"] = None
        self._json(200, meta)

    def r_put_file(self, cid):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ApiError(413, "too_large", "a file may be at most 50 MB")
        data = self.rfile.read(n) if n else b""
        self.svc.put_file(cid, self.query.get("path") or "", data,
                          self.headers.get("If-Match") or None)
        self._empty(204)

    def r_delete_file(self, cid):
        self.svc.delete_file(cid, self.query.get("path") or "")
        self._empty(204)

    def r_upload_files(self, cid):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD * 4:
            raise ApiError(413, "too_large", "the upload is too large")
        raw = self.rfile.read(n) if n else b""
        self._json(200, self.svc.upload_files(cid, self.query.get("dir") or "/workspace",
                                              self.headers.get("Content-Type") or "", raw))

    def r_diff(self, cid):
        self._json(200, self.svc.diff(cid, self.query.get("run_id")))

    def r_undo(self, cid):
        self._json(202, self.svc.undo(cid, str(self._body().get("run_id") or "")))

    def r_view(self, cid, path):
        from chatd.panes import TEXT_LIMIT, view_csp
        from chatd.preview import cors
        fh, meta = self.svc.open_file(cid, path)
        headers = (("Content-Security-Policy", view_csp(meta["mime"], path)),
                   ("X-Content-Type-Options", "nosniff")) + cors(self.headers.get("Origin"))
        with fh:
            if meta["mime"] == "text/html" and meta["bytes"] <= TEXT_LIMIT:
                # a page: its absolute URLs to its own directory, and the preview's shim
                body = self.svc.view_page(cid, path, fh.read(TEXT_LIMIT))
                return self._send(200, body, meta["mime"], headers)
            self._send_file(fh, meta, headers)

    def r_artifact(self, cid, ref):
        self._json(200, self.svc.artifact(cid, ref))

    def r_port(self, cid, port, _rest):
        self.svc.proxy_port(self, cid, int(port))   # the path the app gets: port_target(self.path)

    def r_sandbox_ports(self, cid):
        self._json(200, self.svc.sandbox_ports(cid))

    def r_serve(self, cid):
        self._json(202, self.svc.serve(cid, self._body()))

    def r_preview_report(self, cid):
        body = self._body()
        body["viewer"] = self.viewer
        self.svc.preview_report(cid, body)
        self._empty(204)

    def r_terminals(self, cid):
        self._json(200, self.svc.terminals_list(cid))

    def r_new_terminal(self, cid):
        self._json(200, self.svc.terminal_open(cid, str(self._body().get("session") or "")))

    def r_terminal_ws(self, cid, session):
        self.svc.terminal_ws(self, cid, session)

    # ------------------------------------------------------------------ SSE
    def _after(self) -> int | None:
        raw = self.headers.get("Last-Event-ID") or self.query.get("after")
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            return -1          # an id from somewhere else: reset

    def _stream(self, attach, detach) -> None:
        sub, replay = attach(self._after())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        w = self.wfile
        try:
            if sub is None:
                w.write(control("reset"))
                w.flush()
                return
            for seq, kind, data in replay:
                w.write(frame(seq, kind, data))
            w.write(control("current"))
            w.flush()
            while True:
                items = sub.take(HEARTBEAT_S)
                if items is None:
                    return
                if sub.overflow:
                    w.write(control("reset"))
                    w.flush()
                    return
                if not items:
                    w.write(HEARTBEAT)
                else:
                    w.write(b"".join(frame(*it) for it in items))
                w.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            if sub is not None:
                detach(sub)
                sub.close()

    def r_events(self, cid):
        self.svc.conv(cid)
        self._stream(lambda after: self.svc.attach(cid, after),
                     lambda sub: self.svc.detach(cid, sub))

    def r_global_events(self):
        self._stream(self.svc.attach_global, lambda sub: self.svc.gfan.remove(sub))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(svc: Service, host: str, port: int, key: str) -> Server:
    handler = type("ChatdHandler", (Handler,), {"svc": svc, "key": key})
    return Server((host, port), handler)


def serve_in_thread(srv: Server) -> threading.Thread:
    t = threading.Thread(target=srv.serve_forever, name="chatd-http", daemon=True)
    t.start()
    return t
