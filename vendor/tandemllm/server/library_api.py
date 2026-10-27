"""The package library for the model's own scripts (R1, 2026-10-03).

The owner, 2026-10-03: the model must be able to search the library and use it later. The router
picks a skill for a turn by itself (server/conductor/libsearch.py); these routes let the model
look one up from its own shell, before unfamiliar work, with curl or a few lines of Python (the
`tandem-library` package tells it how):

    GET /v1/library                  what the library holds: every enabled skill's name,
                                     description and classes, and the two routes below
    GET /v1/library/search?q=&k=&class=
                                     [{name, description, score}], best first: BM25 over the
                                     skills' names, descriptions and texts (`k` 1 to 20, default
                                     5; `class` keeps the skills INDEX.md fits to that router
                                     class); [] when nothing matches
    GET /v1/library/{name}           {name, description, classes, body, reference}: the text the
                                     router injects and the reference (snippets, commands) that
                                     only an explicit read gets

Errors are the OpenAI API's shape, `{"error": {"message", "type", "param"?}}`, as the chat API's
and the Laya route's (server/laya_api.py, L1): 422 `invalid_request_error` (no `q`, a `q` over
1,000 characters, `k` out of range, an unknown class), 404 `not_found` (an unknown or disabled
skill, or no library installed), 429 `rate_limited` (RATE_PER_MIN calls a minute over every
caller) with `Retry-After`.

Auth is the chat API's and the Laya route's: none at the engine (server/auth.py: the OpenAI
routes stay open; the service is LAN-only behind the reverse proxy), so a script needs no key and
a bearer header it sends anyway is ignored. The library holds no secret: the library's skills. The
answers carry `Access-Control-Allow-Origin: *`, so a page the model builds can `fetch` them (a GET
without custom headers needs no preflight).

Every call is one journal row (`kind: library_api`, task `list`, `search` or `read`): the query
as a hash and a length, never a text, `k`, the class, the names answered, the status and the time.
"""

from __future__ import annotations

import collections
import hashlib
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit

from server.conductor import packages as packages_mod

MAX_QUERY = 1000
MAX_K, DEFAULT_K = 20, 5
RATE_PER_MIN = 240
PREFIX = "/v1/library"
CORS = (("Access-Control-Allow-Origin", "*"), ("Cache-Control", "no-store"))


class Refused(ValueError):
    """A request the route answers with an error."""

    def __init__(self, status: int, kind: str, message: str, param: str | None = None,
                 retry: int | None = None):
        super().__init__(message)
        self.status, self.kind, self.param, self.retry = status, kind, param, retry

    def body(self) -> dict:
        err = {"message": str(self), "type": self.kind}
        if self.param:
            err["param"] = self.param
        return {"error": err}


class Rate:
    """Calls a minute over every caller: a sliding window of the last minute's stamps."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._t: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def take(self, per_min: int) -> int | None:
        """None when one more call fits (it is counted), else the seconds to wait."""
        now = self.clock()
        with self._lock:
            while self._t and now - self._t[0] >= 60.0:
                self._t.popleft()
            if len(self._t) >= per_min:
                return max(1, int(60.0 - (now - self._t[0])) + 1)
            self._t.append(now)
            return None


RATE = Rate()


def _index(cond):
    """The router's index, rebuilt if the documents or the files moved."""
    from server.conductor import schema
    v, doc = cond.store.peek("packages")
    _, router = cond.store.peek("router")
    idx = cond.router.index
    idx.ensure(v, doc, schema.library(router))
    if not any(e.enabled for e in idx.entries):
        raise Refused(404, "not_found", "no library skill is installed on this engine")
    return idx


def answer(cond, path: str, query: str) -> tuple[str, object, dict]:
    """(the task, the answer, the journal detail) for one GET, or `Refused`."""
    wait = RATE.take(RATE_PER_MIN)
    if wait is not None:
        raise Refused(429, "rate_limited", f"at most {RATE_PER_MIN} calls a minute; wait {wait} s",
                      retry=wait)
    idx = _index(cond)
    q = {k: v[-1] for k, v in parse_qs(query, keep_blank_values=True).items()}
    if path == PREFIX:
        skills = [e.card() for e in idx.entries if e.enabled]
        return "list", {"object": "library", "count": len(skills), "skills": skills,
                        "search": f"GET {PREFIX}/search?q=<words>&k=5",
                        "read": f"GET {PREFIX}/<name>",
                        "auth": "none at the engine, as the chat API"}, {"count": len(skills)}
    if path == PREFIX + "/search":
        text = q.get("q", "")
        if not text.strip():
            raise Refused(422, "invalid_request_error", "q: the words to search for", "q")
        if len(text) > MAX_QUERY:
            raise Refused(422, "invalid_request_error", f"q: at most {MAX_QUERY} characters", "q")
        try:
            k = int(q.get("k") or DEFAULT_K)
        except ValueError:
            raise Refused(422, "invalid_request_error", "k: a whole number", "k") from None
        if not 1 <= k <= MAX_K:
            raise Refused(422, "invalid_request_error", f"k: 1 to {MAX_K}", "k")
        cls = q.get("class") or None
        if cls is not None and not any(cls in (e.classes or ()) for e in idx.entries):
            raise Refused(422, "invalid_request_error", f"class: no skill fits {cls!r}", "class")
        hits = idx.search(text, k, cls)
        out = [{"name": e.name, "description": e.description, "score": round(s, 3)}
               for e, s in hits]
        return "search", out, {"q": hashlib.sha256(text.encode()).hexdigest()[:12],
                               "q_chars": len(text), "k": k, "class": cls,
                               "names": [h["name"] for h in out]}
    name = unquote(path[len(PREFIX) + 1:])
    e = idx.by_name.get(name)
    if e is None or not e.enabled or "/" in name:
        raise Refused(404, "not_found", f"no library skill {name!r}; GET {PREFIX}/search?q=...")
    return "read", dict(e.card(), body=e.body, reference=e.reference), {"names": [e.name]}


def handle_get(handler, cond) -> None:
    """`GET /v1/library[/...]`: the answer and its journal row."""
    t0 = time.perf_counter()
    split = urlsplit(handler.path)
    path = split.path.rstrip("/")
    task, out, detail, err = "read", None, {}, None
    try:
        task, out, detail = answer(cond, path, split.query)
    except Refused as exc:
        err = exc
        task = {PREFIX: "list", PREFIX + "/search": "search"}.get(path, "read")
    detail.update(status=err.status if err else 200,
                  ms=round((time.perf_counter() - t0) * 1e3, 3))
    if err is not None:
        detail.update(error=err.kind, param=err.param)
    # The session the address's token names (server/session_tokens.py)
    sess = getattr(handler, "_sess", None)
    if getattr(handler, "_stok", None) is not None:
        detail["session_key"] = sess
        reg = getattr(cond, "router", None) and cond.router.session_tokens
        if reg is not None:
            reg.note(sess, "library")
    cond.journal.note("library_api", task, decision=err.kind if err else "ok", detail=detail,
                      **({"session_key": sess} if sess is not None else {}))
    if err is not None:
        return handler._json(err.status, err.body(), extra_headers=CORS + (
            (("Retry-After", str(err.retry)),) if err.retry else ()))
    if task == "read":
        # a shipped text names the provider `{tandem_url}`: this request's address
        url = packages_mod.request_url(handler.headers)
        if getattr(handler, "_stok", None) is not None:
            from server.session_tokens import with_token
            url = with_token(url or packages_mod.URL_DEFAULT, handler._stok)
        out = dict(out, body=packages_mod.with_url(out["body"], url),
                   reference=packages_mod.with_url(out["reference"], url))
    handler._json(200, out, extra_headers=CORS)
