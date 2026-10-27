"""Who may read what: the OpenAI API stays open; the rest is open too until an admin password is set.

THE ADMIN PASSWORD (AU1, 2026-10-06, a design decision: no password
prompt anywhere in the UI until a password is set). The one switch is the admin
token, `QSE_ADMIN_TOKEN` (`TANDEM_ADMIN_TOKEN` for the standalone proxy), from secrets.env:

    not set (the default)   every route below answers anyone: the dashboard's reads, the conductor's
                            writes, the memory (notes, transcripts, the disk tier), the deciders,
                            the cache routes and the full /health. The dashboard asks for nothing.
                            Only /metrics keeps its own token when QSE_METRICS_TOKEN is set (the
                            scrapers'). The server is then meant for a trusted network only.
    set                     the table below, unchanged: the admin token (bearer) or a dashboard
                            session, or trusted-local.

`Auth.password_set` is that switch; nothing else turns the admin rule on.

your-host.example proxies every path to this server, and the reverse proxy reaches it through an ssh tunnel,
so a request from any device on the LAN arrives from 127.0.0.1 -- the source address cannot tell
the box's own tools from anyone else. So the mechanism is a bearer token, with one narrow
exemption, and a session cookie so the dashboard's EventSource needs no token in its URL:

    route                                   who
    POST /v1/chat/completions, /v1/completions, GET /v1/models      anyone (per-key API auth is)
    GET /health                             anyone gets {"status"}; the full body needs a token or trusted-local
    GET /v1/library, /v1/library/*          anyone, as the chat API (R1: the package library for the
                                            model's own scripts, rate-limited; server/library_api.py)
    GET /metrics                            the metrics token, the admin token/session, or trusted-local
    GET /v1/cache/stats                     the admin token/session, or trusted-local
    POST /v1/cache/peek, /v1/cache/prewarm  the same (`cache_read`; SRV-349: where a chat's rows
                                            are, and the NVMe read back while the engine is idle)
    POST /v1/cache/clear                    the admin BEARER token (never the cookie), or trusted-local
    POST /v1/cache/release                  the same (SRV-483: chatd asks for room for a sandbox)
    PUT/POST /v1/conductor/* (writes)       the admin token/session, or trusted-local; reads follow
                                            the dashboard rule below
    POST /v1/systemone, /v1/predict         the admin token/session, or trusted-local (decision
                                            models, SRV-131: LAN only, behind the bearer token)
    /v1/setup/* (GET and POST)              the admin token/session, or trusted-local (the setup
                                            guide: its reads too, since apply and reset follow
                                            them; server/setup_api.py)
    POST /v1/laya/ask, GET /v1/laya         anyone, as the chat API (L1: Laya for the model's own
                                            scripts, typed answers on a short text, rate-limited;
                                            server/laya_api.py); X1: POST /v1/laya/drive/end and
                                            GET /v1/laya/kit/<file> (the playtest kit) the same
    GET /dashboard, /dashboard/*            anyone: the static shell, no data in it
    /v1/dashboard/*                         login off (the default): anyone; login on: the admin token or
                                            session only (POST /v1/dashboard/session is the login)

THE DASHBOARD LOGIN (`--dashboard-login on|off`, from `QSE_DASHBOARD_LOGIN`, off by default). The
dashboard only reads -- counts, timings, the log (no prompt or answer text unless --log-content),
the settings with their secrets redacted -- and it is meant for a local network, so by default its
read API answers without a token, even with no secrets.env, and the page opens without a sign-in
screen. `GET /v1/dashboard/session` then says `login: false`, so the page knows there is nothing
to sign in to. With the login on, everything below applies unchanged. Either way the login decides
only `/v1/dashboard/*` (which includes `/v1/dashboard/metrics`, the /metrics page for the
Performance view): /metrics itself, the cache routes and the full /health keep their own rules.

TRUSTED-LOCAL (`--trust-loopback`, on by default): the TCP peer is loopback AND the request has
neither `X-Forwarded-For` nor `X-Real-IP`. The reverse proxy (nginx) sets both on everything it proxies, so
tunnelled traffic never qualifies, and the watchdog, row3, gate.sh and soak on the box keep working
with no token.

NO PASSWORD, OPEN (AU1): with no admin token configured the admin routes answer everyone (until
2026-10-06 they answered 404, fail closed, and the dashboard asked for the token). Tokens come from `~/.qwen38-spark-engine/secrets.env` (made by
`ops/make-secrets.sh`, mode 600, sourced by `ops/start.sh`), are at least 32 characters, are
compared with `hmac.compare_digest`, and are never printed, logged or returned.

THE SESSION: `POST /v1/dashboard/session {"token"}` sets `qse_dash=<expiry>.<nonce>.<HMAC-SHA256>` (a
key derived from the admin token, so rotating the token ends every session; the nonce makes each
login its own session, so signing out ends that one only), HttpOnly, SameSite=Strict,
Path=/, 12 hours, Secure behind https. Five failed logins a minute from one address (X-Real-IP, or
the peer) and the sixth gets a 429. The dashboard API only reads, and `/v1/cache/clear` does not
take the cookie. SameSite=Strict is NOT the whole CSRF story for `/v1/setup/*`, which writes
(apply, reset) and takes the cookie: an open box, or loopback through a tunnel, needs no cookie at
all, and a text/plain POST from any page needs no preflight. So every /v1/setup POST must also be
declared JSON, carry no `Sec-Fetch-Site: cross-site`, and name this server's host in its `Origin`
when it sends one (server/setup_api.py same_origin_json).
"""

from __future__ import annotations

import collections
import hashlib
import hmac
import os
import secrets
import threading
import time

COOKIE = "qse_dash"
SESSION_S = int(os.environ.get("QSE_SESSION_S", 400 * 86400))  # a LAN-only dashboard: one login lasts 400 days (the cookie cap browsers keep)
MIN_TOKEN = 32
LOGIN_ENV = "QSE_DASHBOARD_LOGIN"
LOGIN_DEFAULT = "off"
LOGIN_FAILS = 5
LOGIN_WINDOW_S = 60.0
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")


def _eq(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and hmac.compare_digest(a.encode(), b.encode())


class Auth:
    def __init__(self, admin_token: str | None = None, metrics_token: str | None = None, *,
                 trust_loopback: bool = True, login: bool = True, clock=time.time):
        for name, tok in (("QSE_ADMIN_TOKEN", admin_token), ("QSE_METRICS_TOKEN", metrics_token)):
            if tok is not None and len(tok) < MIN_TOKEN:
                raise SystemExit(f"[auth] {name} is shorter than {MIN_TOKEN} characters; make a "
                                 f"new one with ops/make-secrets.sh --rotate")
        self.admin = admin_token or None
        self.metrics = metrics_token or None
        self.trust_loopback = bool(trust_loopback)
        # the object on its own is the strict policy; the server's default (off) comes from
        # `--dashboard-login`, and from_env reads the same variable with the same default
        self.login = bool(login)
        self.clock = clock
        self._key = (hmac.new(self.admin.encode(), b"qse-dash-session-v1", hashlib.sha256).digest()
                     if self.admin else None)
        self._fails: dict = collections.defaultdict(collections.deque)
        self._revoked: dict[str, int] = {}          # logged-out cookie values, until they expire
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, env=None, **kw) -> "Auth":
        env = os.environ if env is None else env
        kw.setdefault("login", env.get(LOGIN_ENV, LOGIN_DEFAULT) == "on")
        return cls(env.get("QSE_ADMIN_TOKEN") or None, env.get("QSE_METRICS_TOKEN") or None, **kw)

    @property
    def password_set(self) -> bool:
        """THE switch (AU1): an admin password (the admin token) is configured. False: every
        admin route is open and the dashboard asks for nothing."""
        return bool(self.admin)

    def describe(self) -> str:
        """The startup line: whether auth is on, never a token."""
        if not self.password_set:
            return (f"admin password: not set (every admin route and the dashboard are open; keep "
                    f"this server on a trusted network), "
                    f"metrics token: {'set' if self.metrics else 'not set'}")
        if not self.login:
            dash = "off (its read API needs no token)"
        else:
            dash = "on"
        return (f"admin password: set, dashboard login: {dash}, "
                f"metrics token: {'set' if self.metrics else 'not set'}, "
                f"trusted loopback: {'on' if self.trust_loopback else 'off'}")

    # ----------------------------------------------------------------- who is asking
    @staticmethod
    def _header(headers, name: str) -> str:
        try:
            return headers.get(name) or ""
        except Exception:                                          # noqa: BLE001
            return ""

    def trusted_local(self, peer: str, headers) -> bool:
        if not self.trust_loopback or peer not in LOOPBACK:
            return False
        return not (self._header(headers, "X-Forwarded-For") or self._header(headers, "X-Real-IP"))

    def bearer(self, headers) -> str | None:
        auth = self._header(headers, "Authorization").strip()
        if auth[:7].lower() != "bearer ":
            return None
        return auth[7:].strip() or None

    def admin_bearer(self, headers) -> bool:
        return _eq(self.bearer(headers), self.admin)

    def metrics_bearer(self, headers) -> bool:
        return _eq(self.bearer(headers), self.metrics)

    # ----------------------------------------------------------------- the session cookie
    def make_cookie(self, now: float | None = None) -> tuple[str, int]:
        """`<expiry>.<nonce>.<mac>`. The nonce makes each login its own session: the value
        was the expiry second and its mac alone, so two logins in one second shared a cookie and a
        sign-out revoked both, or a login right after a sign-out was born revoked."""
        exp = int((self.clock() if now is None else now) + SESSION_S)
        nonce = secrets.token_hex(8)
        mac = hmac.new(self._key, f"{exp}.{nonce}".encode(), hashlib.sha256).hexdigest()
        return f"{exp}.{nonce}.{mac}", exp

    def cookie(self, headers) -> str | None:
        val = None
        for part in self._header(headers, "Cookie").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE:
                val = v
        return val or None

    def revoke(self, headers) -> None:
        """Logout: the presented session stops working here too, not only in the browser."""
        exp = self.session(headers)
        if exp is not None:
            now = self.clock()
            with self._lock:
                self._revoked = {v: e for v, e in self._revoked.items() if e > now}
                self._revoked[self.cookie(headers)] = exp

    def session(self, headers, now: float | None = None) -> int | None:
        """The session's expiry if the request carries a valid, unexpired one, else None."""
        if self._key is None:
            return None
        val = self.cookie(headers)
        if not val or val in self._revoked:
            return None
        parts = val.split(".")
        if len(parts) != 3 or not parts[0].isdigit():
            return None
        exp_s, nonce, mac = parts
        want = hmac.new(self._key, f"{exp_s}.{nonce}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, mac):
            return None
        exp = int(exp_s)
        return exp if exp > (self.clock() if now is None else now) else None

    def admin_any(self, headers) -> bool:
        return self.admin_bearer(headers) or self.session(headers) is not None

    # ----------------------------------------------------------------- login rate limit
    def login_blocked(self, who: str) -> bool:
        now = self.clock()
        with self._lock:
            q = self._fails[who]
            while q and now - q[0] > LOGIN_WINDOW_S:
                q.popleft()
            return len(q) >= LOGIN_FAILS

    def login_failed(self, who: str) -> None:
        with self._lock:
            self._fails[who].append(self.clock())
            if len(self._fails) > 10_000:
                self._fails.clear()

    # ----------------------------------------------------------------- the policy
    def decide(self, route: str, peer: str, headers) -> str:
        """`ok` or `unauthorized` (401) for one guarded route kind (`absent`, the 404 of the old
        fail-closed rule, is no longer returned; the callers still handle it):
        `metrics`, `cache_read`, `cache_clear`, `conductor_write`, `decider`, `dashboard`,
        `health_full`, `setup`. With no admin password set (AU1) every kind is `ok`, but /metrics with a
        metrics token set still wants that token (or trusted-local)."""
        local = self.trusted_local(peer, headers)
        if not self.password_set:
            if route == "metrics" and self.metrics and not local:
                return "ok" if self.metrics_bearer(headers) else "unauthorized"
            return "ok"
        if route == "dashboard":
            if not self.login:
                return "ok"
            return "ok" if self.admin_any(headers) else "unauthorized"
        if local:
            return "ok"
        if route == "metrics":
            return ("ok" if self.metrics_bearer(headers) or self.admin_any(headers)
                    else "unauthorized")
        if route == "health_full":
            return ("ok" if self.metrics_bearer(headers) or self.admin_any(headers)
                    else "unauthorized")
        if route in ("cache_read", "conductor_write", "decider", "setup"):
            return "ok" if self.admin_any(headers) else "unauthorized"
        if route == "cache_clear":
            return "ok" if self.admin_bearer(headers) else "unauthorized"
        return "unauthorized"
