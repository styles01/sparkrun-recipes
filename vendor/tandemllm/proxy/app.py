"""The standalone proxy: the dashboard and the OpenAI routes in front of any backend, no torch.

    python -m proxy --config-db ~/.tandemllm/config.sqlite3 --port 8080

serves on one port:

    POST /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/responses
                                   relayed to the backend the model routes to (proxy/route.py)
    GET  /v1/models, /v1/models/{id}
                                   every backend's real models, `owned_by` its id
    GET  /dashboard/               the dashboard (dashboard/dist, served from disk)
    GET  /v1/dashboard/*           summary, usage, requests, system, live, logs, metrics, session
    *    /v1/config/*              the Backends and Setup screens (proxy/config_api.py)
    GET  /metrics                  the proxy's families in the Prometheus text format
    GET  /health                   {"status": "ok"}; the full body for the admin or the machine
    *    /v1/conductor/*           the conductor's documents, journal and sessions (P4, with
                                   --conductor-config; the engine-only screens answer 404)
    GET  /v1/library[/...]         the package library for the model's scripts (P4)
    POST /v1/laya/ask              Laya, when the `decider` role names a Jev endpoint (P4)
    *    /s/<token>/v1/library..., /s/<token>/v1/laya...
                                   the same, attributed to the session the token names

The engine's own server mounts the same service with `--backends` (server/app.py); this entry
point is for everyone without a Spark. The dashboard's engine-only panels (caches, the
conductor, speculation, memory) answer 404 here and the views say so.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from proxy import config as proxy_config
from proxy import config_api
from proxy import hostcollect
from proxy import route as route_mod
from proxy.service import PROXY_PATHS, Service
from server import auth as auth_mod
from server import chat_forward
from server import dashboard_api
from server import live as live_mod
from server import parallel as parallel_mod
from server import logbuf
from server import metrics as metrics_mod
from server import static

STATE: dict = {}
# SRV-605: the request phase has a timeout, the answer has none. A client that connects and sends
# nothing, stops halfway through its request, or keeps a connection idle between requests gives
# the thread back after this many seconds (`--idle-timeout`, TANDEM_IDLE_TIMEOUT_S; 0 for none).
# Handler.handle_one_request arms it on the client's socket before each request line and
# send_response lifts it as the answer starts, so a stream (a relayed answer, the live view, the
# logs, a chatd SSE, the chat WebSocket; chat_forward lifts it before chatd's head) is never cut
# by it, however long its upstream is quiet or the client is slow to read.
IDLE_TIMEOUT_S = 60.0


def version() -> str:
    """VERSION in a checkout, the distribution's version when installed (proxy/_version.py)."""
    from proxy import _version
    return _version.release()


def _live_engine() -> dict:
    """The live stream's engine block for a server with no engine: the proxy's in-flight
    count."""
    svc: Service = STATE["svc"]
    return {"model": None, "version": STATE.get("version", ""),
            "draining": bool(STATE.get("draining")), "kv": {"length": 0, "max_len": 0},
            "memory": {"mem_available_gib": None, "rss_gib": None}, "store": None}


def _live_status() -> dict:
    svc: Service = STATE["svc"]
    n = svc.relay.inflight.total()
    return {"status": "draining" if STATE.get("draining") else ("busy" if n else "ok"),
            "running": n, "waiting": 0, "lane": 0,
            "uptime_s": round(time.time() - svc.started, 1),
            "last_request_at": None, "backends": svc.backends_view()}


def _system() -> dict:
    svc: Service = STATE["svc"]
    led = STATE.get("ledger")
    li = led.info() if led is not None else None
    args = STATE.get("args") or {}
    hosts = hostcollect.hosts_for(
        STATE, svc.cfg.backends(),
        hostcollect.engine_entry(args.get("port"), "this proxy", backend="proxy"),
        paths=[os.path.dirname(svc.cfg.path)])
    mem = dashboard_api.meminfo()
    if hosts and mem["unified_total_bytes"] is None:          # no /proc here (macOS)
        mem["unified_total_bytes"] = hosts[0]["memory"]["total_bytes"]
        mem["unified_available_bytes"] = hosts[0]["memory"]["available_bytes"]
    gpu = hostcollect.gpu_block(STATE) if hosts else None
    return {
        "engine": {"version": STATE.get("version", ""), "git_sha": "", "code_sha256": "",
                   "started_at": dashboard_api.iso_utc_s(svc.started),
                   "uptime_s": round(time.time() - svc.started, 1), "pid": os.getpid(),
                   "model": None, "base_model": None, "max_len": None, "drafter": None,
                   "tree": False, "reasoning_format": None, "reasoning_effort": None,
                   "status": _live_status()["status"]},
        "flags": {"args": dashboard_api.redact_args(STATE.get("args") or {}),
                  "env": dashboard_api.redact_env(dict(os.environ))},
        "memory": dict(mem, gpu_allocated_bytes=None,
                       gpu_reserved_bytes=None, gpu_max_allocated_bytes=None),
        "gpu": gpu or STATE.setdefault("gpu", dashboard_api.GpuSampler()).sample(),
        "caches": None,
        "queue": {"running": svc.relay.inflight.total(), "waiting": 0, "max_queue": None,
                  "queue_timeout_s": None, "request_timeout_s": None},
        "inflight": {"served": 0, "refused": 0, "errors": 0, "timeouts": 0, "abandoned": 0},
        "ledger": {"enabled": li is not None,
                   "rows": (li or {}).get("rows") or 0, "bytes": (li or {}).get("bytes") or 0,
                   "oldest": dashboard_api.iso_utc((li or {}).get("oldest_ms")),
                   "queue": (li or {}).get("queue") or 0,
                   "dropped": (li or {}).get("dropped") or 0},
        "disk": {"state_dir_free_bytes": dashboard_api.disk_free(os.path.dirname(
            STATE["svc"].cfg.path))},
        "proxy": {"mode": "standalone", "backends": svc.backends_view()},
        **({"hosts": hosts} if hosts is not None else {}),
    }


def dashboard() -> "dashboard_api.DashboardAPI":
    api = STATE.get("dashboard_api")
    if api is None:
        api = STATE["dashboard_api"] = dashboard_api.DashboardAPI(
            STATE.get("ledger"), live=_live_status, system=_system,
            backends=lambda: STATE["svc"].cfg.backends())
    return api


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TandemLLM-proxy"

    def log_message(self, fmt, *args):
        if STATE.get("verbose"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def handle_one_request(self):
        self.connection.settimeout(self.timeout)          # the request phase (IDLE_TIMEOUT_S)
        super().handle_one_request()

    def send_response(self, code, message=None):
        self.connection.settimeout(None)                  # the answer: no timeout
        super().send_response(code, message)

    # ----------------------------------------------------------------- helpers
    def _json(self, code: int, payload, extra_headers: tuple = ()) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in extra_headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _send_empty(self, code: int, headers: tuple = ()) -> None:
        try:
            self.send_response(code)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _read_raw(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _peer(self) -> str:
        return (self.client_address or ("",))[0]

    def _allowed(self, route: str) -> bool:
        verdict = STATE["auth"].decide(route, self._peer(), self.headers)
        if verdict == "ok":
            return True
        path = self.path.split("?")[0]
        if verdict == "absent":
            self._json(404, {"error": {"type": "not_found", "message": f"no route {path}"}})
        else:
            self._json(401, {"error": {"type": "unauthorized",
                                       "message": "a bearer token or the dashboard session is "
                                                  "needed"}},
                       extra_headers=(("WWW-Authenticate", 'Bearer realm="TandemLLM"'),))
        return False

    def _reader_gone(self) -> bool:
        from proxy.relay import client_gone
        return client_gone(self)

    # ----------------------------------------------------------------- routes
    def do_GET(self):
        if self._strip_session():
            return
        if chat_forward.route(self, "GET", STATE["auth"]):     # the Chat tab's chatd (SRV-343)
            return
        raw = self.path.split("?")[0]
        if raw == "/dashboard" or raw.startswith("/dashboard/"):
            return static.serve(self, self.path, STATE.get("dashboard_dir"))
        path = raw.rstrip("/") or "/"
        if path == "/":
            return self._send_empty(302, (("Location", "/dashboard/"),))
        if path.startswith("/v1/dashboard/"):
            return self._dashboard_get(path)
        if path == "/v1/config" or path.startswith("/v1/config/"):
            return config_api.handle(self, STATE["svc"], "GET", path, None)
        if path in ("/health", "/healthz", "/v1/health"):
            code = 503 if STATE.get("draining") else 200
            out = {"status": "draining" if STATE.get("draining") else "ok"}
            if STATE["auth"].decide("health_full", self._peer(), self.headers) == "ok":
                out["backends"] = STATE["svc"].backends_view()
                out["version"] = STATE.get("version")
            return self._json(code, out)
        if path == "/metrics":
            if not self._allowed("metrics"):
                return
            return self._metrics()
        if path == "/v1/models":
            ok, _ = STATE["svc"].authorize(self)
            if not ok:
                return
            return self._json(200, {"object": "list",
                                    "data": STATE["svc"].models(int(STATE["svc"].started))})
        if path.startswith("/v1/models/"):
            # the list above asks for a key; a single card was answering without one,
            # so a configured deployment leaked model ids and card metadata to strangers
            ok, _ = STATE["svc"].authorize(self)
            if not ok:
                return
            name = unquote(path[len("/v1/models/"):])
            card = STATE["svc"].table.card(name, int(STATE["svc"].started))
            if card is None:
                return self._json(404, {"error": {"message": f"no model {name!r}",
                                                  "type": "invalid_request_error",
                                                  "param": "model", "code": "model_not_found"}})
            return self._json(200, card)
        if path.startswith("/v1/conductor/"):
            return self._conductor("GET")
        if (path == "/v1/library" or path.startswith("/v1/library/")) and \
                STATE.get("conductor") is not None:
            from server import library_api
            return library_api.handle_get(self, STATE["conductor"])
        if path in ("/v1/laya", "/v1/laya/") and STATE.get("laya") is not None:
            from server import laya_api
            return laya_api.handle_get(self, STATE["laya"], STATE)
        return self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})

    def _conductor(self, method: str, body=None):
        cond = STATE.get("conductor")
        if cond is None:
            return self._json(404, {"error": {"type": "not_found",
                                              "message": "the conductor is off on this proxy "
                                                         "(--conductor-config off)"}})
        from server.conductor import routes as conductor_routes
        return conductor_routes.handle(self, cond, method, body, STATE["auth"])

    def _strip_session(self) -> bool:
        """`/s/<token>/v1/library...` and `/s/<token>/v1/laya...` (server/session_tokens.py):
        the prefix off, the session on the handler; True when answered here (a 404)."""
        from server import session_tokens as st_mod
        self._sess = self._stok = None
        path, tok = st_mod.strip(self.path)
        if tok is None:
            return False
        p = path.split("?")[0].rstrip("/")
        if not (p in ("/v1/library", "/v1/laya") or p.startswith("/v1/library/")
                or p.startswith("/v1/laya/")):
            self._json(404, {"error": {"message": f"no route {p} under a session address",
                                       "type": "not_found"}})
            return True
        self.path, self._stok = path, tok
        reg = STATE.get("session_tokens")
        self._sess = reg.key_for(tok) if reg is not None else None
        return False

    def do_POST(self):
        if self._strip_session():
            return
        if chat_forward.route(self, "POST", STATE["auth"]):
            return
        path = self.path.split("?")[0].rstrip("/")
        t_arrival = time.perf_counter()
        raw = self._read_raw()
        if path.startswith("/v1/collect/"):          # P6: a plugin's batch, its own token
            from proxy import collect_api
            return collect_api.handle(self, STATE["svc"], path, raw)
        try:
            body = json.loads(raw or b"{}")
        except ValueError as exc:
            return self._json(400, {"error": {"message": f"bad json: {exc}",
                                              "type": "invalid_request_error"}})
        if path == "/v1/dashboard/session":
            return self._session("POST", body)
        if path == "/v1/config" or path.startswith("/v1/config/"):
            return config_api.handle(self, STATE["svc"], "POST", path, body)
        if path in PROXY_PATHS:
            return self._complete(path, raw, body, t_arrival)
        if path.startswith("/v1/conductor/"):
            return self._conductor("POST", body)
        if path == "/v1/laya/ask" and STATE.get("laya") is not None:
            from server import laya_api
            return laya_api.handle_post(self, STATE["laya"], STATE, body)
        return self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})

    def _complete(self, path, raw, body, t_arrival):
        svc: Service = STATE["svc"]
        ok, user = svc.authorize(self)
        if not ok:
            svc.account_refusal(self, path, body, 401, finish="refused",
                                error_type="invalid_api_key")
            return
        if not isinstance(body, dict):
            return self._json(400, {"error": {"message": "the body must be a JSON object",
                                              "type": "invalid_request_error"}})
        try:
            route = svc.table.resolve(body.get("model") or "")
        except route_mod.NotFound as exc:
            svc.account_refusal(self, path, body, 404, user=user, error_type="model_not_found")
            return self._json(404, exc.body())
        svc.serve(self, path, raw, body, route, user, t_arrival=t_arrival)

    def _write(self, method: str):
        if chat_forward.route(self, method, STATE["auth"]):
            return
        path = self.path.split("?")[0].rstrip("/")
        raw = self._read_raw()
        try:
            body = json.loads(raw) if raw else None
        except ValueError as exc:
            return self._json(400, {"error": {"message": f"bad json: {exc}",
                                              "type": "invalid_request_error"}})
        if method == "DELETE" and path == "/v1/dashboard/session":
            return self._session("DELETE")
        if path == "/v1/config" or path.startswith("/v1/config/"):
            return config_api.handle(self, STATE["svc"], method, path, body)
        if path.startswith("/v1/conductor/"):
            return self._conductor(method, body)
        return self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})

    def do_PUT(self):
        return self._write("PUT")

    def do_PATCH(self):
        return self._write("PATCH")

    def do_DELETE(self):
        return self._write("DELETE")

    def _metrics(self) -> None:
        raw = STATE["registry"].render().encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", metrics_mod.CONTENT_TYPE)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    # ----------------------------------------------------------------- the dashboard's API
    def _dashboard_get(self, path: str) -> None:
        if path == "/v1/dashboard/session":
            return self._session("GET")
        if not self._allowed("dashboard"):
            return
        if path == "/v1/dashboard/metrics":
            return self._metrics()
        q = {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}
        try:
            if path == "/v1/dashboard/live":
                return self._live(q)
            if path == "/v1/dashboard/parallel":
                return self._parallel(q)
            if path == "/v1/dashboard/logs":
                return self._logs(q)
            api = dashboard()
            fn = {"/v1/dashboard/summary": api.summary, "/v1/dashboard/usage": api.usage,
                  "/v1/dashboard/requests": api.requests,
                  "/v1/dashboard/system": lambda _q: api.system()}.get(path)
            if fn is None:
                return self._json(404, {"error": {"type": "not_found",
                                                  "message": f"no route {path}"}})
            return self._json(200, fn(q), extra_headers=(("Cache-Control", "no-store"),))
        except dashboard_api.ApiError as exc:
            return self._json(exc.status, exc.body())

    def _session(self, method: str, body=None) -> None:
        a = STATE["auth"]
        # AU1: no admin password set, or the login off -- nothing to sign in to. GET says so
        # (`login: false`, `password_set`), POST and DELETE do not exist.
        if not a.login or not a.password_set:
            if method == "GET":
                return self._json(200, {"contract_version": dashboard_api.SESSION_CONTRACT,
                                        "authenticated": True, "login": False,
                                        "password_set": a.password_set, "expires_at": None},
                                  extra_headers=(("Cache-Control", "no-store"),))
            return self._json(404, {"error": {"type": "not_found",
                                              "message": "the dashboard login is off" if
                                              a.password_set else "no admin password is set"}})
        secure = "; Secure" if (self.headers.get("X-Forwarded-Proto") or "") == "https" else ""
        if method == "GET":
            exp = a.session(self.headers)
            if exp is None and not a.admin_bearer(self.headers):
                return self._allowed("dashboard")
            exp = exp or int(time.time()) + auth_mod.SESSION_S
            return self._json(200, {"contract_version": dashboard_api.SESSION_CONTRACT,
                                    "authenticated": True, "login": True, "password_set": True,
                                    "expires_at": dashboard_api.iso_utc_s(exp)})
        if method == "DELETE":
            a.revoke(self.headers)
            return self._send_empty(204, (("Set-Cookie", f"{auth_mod.COOKIE}=; HttpOnly; "
                                                         f"SameSite=Strict; Path=/; Max-Age=0"
                                                         f"{secure}"),))
        who = self.headers.get("X-Real-IP") or self._peer()
        if a.login_blocked(who):
            return self._json(429, {"error": {"type": "too_many",
                                              "message": "too many failed logins; wait a minute"}},
                              extra_headers=(("Retry-After", "60"),))
        token = (body or {}).get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not auth_mod._eq(token, a.admin):
            a.login_failed(who)
            return self._json(401, {"error": {"type": "unauthorized", "message": "wrong token"}},
                              extra_headers=(("WWW-Authenticate", 'Bearer realm="TandemLLM"'),))
        value, _ = a.make_cookie()
        return self._send_empty(204, (("Set-Cookie", f"{auth_mod.COOKIE}={value}; HttpOnly; "
                                                     f"SameSite=Strict; Path=/; "
                                                     f"Max-Age={auth_mod.SESSION_S}{secure}"),))

    def _sse_head(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _parallel(self, q: dict) -> None:
        """`GET /v1/dashboard/parallel[?since=<t>]`: every request in flight on every backend
        (server/parallel.py)."""
        board = STATE.get("parallel")
        if board is None:
            raise dashboard_api.ApiError(404, "not_found", "the parallel view is off")
        try:
            since = float(q["since"]) if q.get("since") else None
        except ValueError:
            raise dashboard_api.ApiError(400, "bad_request", "since: a number (unix seconds)")
        data = parallel_mod.encode(board.snapshot(since))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _live(self, q: dict) -> None:
        reg = STATE["live"]
        if q.get("follow", "1") in ("0", "false", "no"):
            return self._json(200, reg.snapshot(), extra_headers=(("Cache-Control", "no-store"),))
        if not reg.subscribe():
            raise dashboard_api.ApiError(429, "too_many",
                                         f"at most {live_mod.MAX_STREAMS} live streams at once")
        try:
            self._sse_head()
            sent, data = reg.first_event()
            self.wfile.write(data)
            self.wfile.flush()
            pinged = time.monotonic()
            while not STATE.get("draining"):
                reg.wait_event(sent, 0.25)
                ev = reg.event()
                if ev is not None and ev[0] > sent:
                    sent = ev[0]
                    self.wfile.write(ev[1])
                    self.wfile.flush()
                    pinged = time.monotonic()
                elif self._reader_gone():
                    break
                elif time.monotonic() - pinged > 15.0:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    pinged = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True
        finally:
            reg.unsubscribe()

    def _logs(self, q: dict) -> None:
        buf = logbuf.BUFFER
        level = q.get("level") or "info"
        if level not in logbuf.LEVELS:
            raise dashboard_api.ApiError(400, "bad_request",
                                         f"level must be one of {', '.join(logbuf.LEVELS)}")
        grep = q.get("grep") or None
        try:
            backlog = int(q.get("backlog") or 500)
            lei = self.headers.get("Last-Event-ID")
            since = q.get("since")
            after = int(lei) if lei else (int(since) if since and since.isdigit() else None)
        except ValueError:
            raise dashboard_api.ApiError(400, "bad_request", "backlog and Last-Event-ID are "
                                                             "integers")
        if q.get("follow", "1") in ("0", "false", "no"):
            return self._json(200, {"contract_version": dashboard_api.LOGS_CONTRACT,
                                    "lines": buf.lines(level=level, after=after, grep=grep,
                                                       limit=backlog), "last_seq": buf.seq},
                              extra_headers=(("Cache-Control", "no-store"),))
        sub = buf.subscribe(level, grep, self._reader_gone)
        if sub is None:
            raise dashboard_api.ApiError(429, "too_many",
                                         f"at most {logbuf.MAX_SUBSCRIBERS} log streams at once")
        try:
            self._sse_head()
            last = after or 0

            def send(entries):
                nonlocal last
                for e in entries:
                    if e["seq"] > last:
                        self.wfile.write(f"event: log\nid: {e['seq']}\ndata: "
                                         f"{json.dumps(e, ensure_ascii=False)}\n\n".encode())
                        last = e["seq"]

            send(buf.lines(level=level, after=after, grep=grep, limit=backlog))
            self.wfile.flush()
            quiet = time.monotonic()
            while not STATE.get("draining"):
                items, dropped = sub.take(1.0)
                if sub.closed or self._reader_gone():
                    break
                if dropped:
                    self.wfile.write(f"event: gap\ndata: {json.dumps({'dropped': dropped})}\n\n"
                                     .encode())
                if items:
                    send(items)
                elif not dropped:
                    if time.monotonic() - quiet < 15.0:
                        continue
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
                quiet = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True
        finally:
            buf.unsubscribe(sub)


class Server(ThreadingHTTPServer):
    """The standalone server: a thread per request; a peer that resets a keep-alive connection
    between requests is not an error worth a traceback."""

    daemon_threads = True
    # clients that connect in the same moment all get in (the night pass's bench: 64 at once);
    # the default backlog of 5 reset some of them on Linux
    request_queue_size = 128

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def build(args, env=None) -> Service:
    """The service of the standalone server from its arguments (tests call this too)."""
    from proxy import config as config_mod
    from server import ledger as ledger_mod
    env = os.environ if env is None else env
    Handler.timeout = getattr(args, "idle_timeout", IDLE_TIMEOUT_S) or None
    cfg = config_mod.ConfigDB(args.config_db, env=env,
                              use_keychain=not getattr(args, "no_keychain", False)).open()
    cfg.apply_env()
    led = None
    if args.usage_ledger and args.usage_ledger != "off":
        ledger_mod.check_config(args.usage_ledger, args.usage_retention_days,
                                test=env.get("QSE_TEST") == "1")
        led = ledger_mod.Ledger(args.usage_ledger, retention_days=args.usage_retention_days).open()
    login = (args.dashboard_login == "on")
    admin = env.get("TANDEM_ADMIN_TOKEN") or env.get("QSE_ADMIN_TOKEN") or None
    a = auth_mod.Auth(admin, env.get("QSE_METRICS_TOKEN") or None, login=login,
                      trust_loopback=args.trust_loopback == "on")
    registry = metrics_mod.Registry()
    live = live_mod.LiveRegistry(blocks=lambda: None, last_prefill=lambda: None,
                                 engine=_live_engine, hz=live_mod.DEFAULT_HZ, activity=False,
                                 queue_timeout=lambda: 0.0)
    STATE.update(version=version(), ledger=led, auth=a, registry=registry, live=live,
                 dashboard_dir=args.dashboard_dir, verbose=args.verbose,
                 args={k: v for k, v in vars(args).items()})
    cond, base = conductor_of(args, led, env)
    svc = Service(cfg, ledger=led, live=live, auth=a, registry=registry,
                  engine_version=STATE["version"], log=lambda m: print(m, flush=True),
                  conductor=cond, monitor_base=base)
    STATE["svc"] = svc
    STATE["parallel"] = parallel_mod.ParallelBoard(live, cost=svc.live_cost,
                                                   peers=svc.peer_backends)
    # the sampler: it sleeps while nothing is in flight (the live stream and the parallel view's
    # once-a-second sample; before 0.3.32 the standalone proxy never started it)
    live.start()
    STATE.pop("dashboard_api", None)
    return svc


def conductor_of(args, led, env) -> tuple:
    """The standalone proxy's conductor (P4): its documents under `--conductor-config`, its
    journal in the usage ledger's file (as the engine's), the session tokens beside the
    documents, the monitor's policy file (`ops/supervisor.yaml`, when PyYAML is there), and
    Laya when the `decider` role names a Jev endpoint. (None, None) with `off`."""
    root = getattr(args, "conductor_config", "off") or "off"
    STATE.update(conductor=None, session_tokens=None, laya=None)
    if root == "off":
        return None, None
    from server import session_tokens as st_mod
    from server.conductor import shipped
    from server.conductor.decisions import Journal
    from server.conductor.hook import Conductor
    root = os.path.realpath(os.path.expanduser(root))
    os.makedirs(root, mode=0o700, exist_ok=True)
    if getattr(args, "conductor_defaults", "shipped") == "shipped":
        for line in shipped.install_at(root):
            print(f"[conductor] shipped default stored: {line}", flush=True)
    journal = Journal(led.path if led is not None else None)
    cond = Conductor(root, state={"port": getattr(args, "port", None), "model": "TandemLLM"},
                     journal=journal)
    tokens = st_mod.SessionTokens(os.path.join(root, "session-tokens.json"))
    cond.router.session_tokens = tokens
    from server.conductor import migrations
    migrations.run(cond.store, cond.journal)
    base = None
    try:
        from server.conductor.monitor import policy as policy_mod
        base = policy_mod.load_default()
    except Exception as exc:                                       # noqa: BLE001
        print(f"[conductor] the monitor's policy file did not load ({type(exc).__name__}): "
              f"E1, the tool checks and the loop guards stay off", flush=True)
    STATE.update(conductor=cond, session_tokens=tokens, laya=laya_of(cond))
    return cond, base


def laya_of(cond):
    """Laya for `/v1/laya/ask` on the standalone proxy: the `decider` role's Jev endpoint over
    HTTP (another engine's decision models, arbiter), as the engine's own registry answers."""
    from concurrent.futures import Future
    from server.conductor.classify import JevHttp
    _, roles = cond.store.peek("roles")
    r = roles.get("decider") or {}
    ep = str(r.get("endpoint") or "")
    if r.get("mode", "off") == "off" or not ep.startswith(("http://", "https://")):
        return None
    d = JevHttp(ep, r.get("model") or "auto", int(r.get("timeout_ms") or 400))

    class Registry:
        device = None

        def ask(self, state, questions, model="auto", window="idle", deadline_ms=400,
                source=None):
            f: Future = Future()
            try:
                d.timeout_ms = max(d.timeout_ms, int(deadline_ms))
                f.set_result(d.ask(state, questions))
            except Exception as exc:                               # noqa: BLE001
                f.set_exception(exc)
            return f

        def limits(self, model="auto"):
            return d.limits()

        def ready(self):
            return bool(d.health().get("up"))

        def loaded(self):
            return [d.model]

    return Registry()


def serve(args) -> None:
    logbuf.install()
    svc = build(args)
    svc.start()
    srv = Server((args.host, args.port), Handler)
    led = STATE.get("ledger")
    print(f"[proxy] {version()} on {args.host}:{args.port}, config {svc.cfg.path} "
          f"(schema {svc.cfg.schema_was or 'new'} -> {proxy_config.SCHEMA_VERSION}), {len(svc.cfg.backends())} backends, "
          f"ledger {'off' if led is None else led.path}, {STATE['auth'].describe()}",
          flush=True)
    if svc.cfg.empty():
        print(f"[proxy] no backend yet: open http://{args.host}:{args.port}/dashboard/ for the "
              f"Setup screen, or set TANDEM_BACKENDS", flush=True)
    import signal

    def drain(signum, frame):
        STATE["draining"] = True
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, drain)
    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
        if led is not None:
            led.close()
        svc.cfg.close()
