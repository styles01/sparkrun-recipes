"""The model's own memory from its scripts (2026-10-05; the design: Memo "Memory for TandemLLM:
design advice", section 2, the second path beside the hidden block).

Only under the session's address (`{tandem_url}` in the memory card and skill is
`http://host:8000/s/<token>`, server/session_tokens.py): the token names the session, and the
session names the user and the project the router saw for it. A call without a session address,
or with a token the engine does not know, is refused (403): memories belong to one user, and an
open route would hand them to anyone on the LAN.

    GET  /s/<token>/v1/memory                  the index this session sees: the notes' lines
    GET  /s/<token>/v1/memory/search?q=&k=     [{id, kind, scope, status, title, score}], best
                                               first (k 1 to 10, default 5)
    GET  /s/<token>/v1/memory/<id>             {id, title, kind, scope, status, body, ...}
    POST /s/<token>/v1/memory                  {op, scope, kind, title, tags, body, id, reason}:
                                               the block's fields; 200 {op, id, title, ...} or
                                               422 {error: {type: <why>, message}}

Errors are the OpenAI API's shape. 429 past RATE_PER_MIN calls a minute. Every call is one
journal row (`kind: memory_notes`, task `api_<what>`): ids, counts, the status. Never a text.
"""

from __future__ import annotations

import hashlib
import time
from urllib.parse import parse_qs, unquote, urlsplit

from server.library_api import Rate, Refused
from server.conductor import notes as notes_mod

PREFIX = "/v1/memory"
RATE_PER_MIN = 120
RATE = Rate()
MAX_QUERY = 1000


def _ctx(handler, cond) -> dict:
    if getattr(handler, "_stok", None) is None:
        raise Refused(403, "forbidden", "the memory routes answer under this conversation's "
                                        "address only: {tandem_url}/v1/memory from your notes")
    key = getattr(handler, "_sess", None)
    sess = cond.memory.session(key) if key else None
    if sess is None:
        raise Refused(403, "forbidden", "this session address is not known (it may be from "
                                        "before a restart); memory is off for it")
    return {"session": key, "user": sess["user"], "project": sess["project"]}


def answer(cond, handler, method: str, body: dict | None) -> tuple[str, int, object, dict]:
    wait = RATE.take(RATE_PER_MIN)
    if wait is not None:
        raise Refused(429, "rate_limited", f"at most {RATE_PER_MIN} calls a minute; wait {wait} s",
                      retry=wait)
    ctx = _ctx(handler, cond)
    store = cond.notes
    split = urlsplit(handler.path)
    path = split.path.rstrip("/")
    q = {k: v[-1] for k, v in parse_qs(split.query, keep_blank_values=True).items()}
    if method == "POST":
        if path != PREFIX:
            raise Refused(404, "not_found", f"POST {PREFIX} only")
        if not isinstance(body, dict):
            raise Refused(422, "invalid_request_error", "send a JSON object")
        try:
            res = store.write(ctx["user"], {k: body.get(k) for k in (
                "op", "scope", "kind", "title", "tags", "body", "id", "reason", "verify_by")},
                writer="api", project=ctx["project"], session=ctx["session"], from_user=False)
        except notes_mod.Refused as exc:
            raise Refused(422, exc.why, str(exc)) from None
        return "api_write", 200, {k: res.get(k) for k in ("op", "id", "title", "scope", "kind",
                                                           "revised") if k in res}, \
            {"op": res.get("op"), "id": res.get("id"), "revised": bool(res.get("revised"))}
    if path == PREFIX:
        groups = store.index_lines(ctx["user"], ctx["project"], ctx["session"])
        return "api_index", 200, {"object": "memory", "scopes": [
            {"scope": s, "lines": lines} for s, lines in groups],
            "search": f"GET {PREFIX}/search?q=<words>&k=5", "read": f"GET {PREFIX}/<id>"}, \
            {"lines": sum(len(x[1]) for x in groups)}
    if path == PREFIX + "/search":
        text = q.get("q", "")
        if not text.strip() or len(text) > MAX_QUERY:
            raise Refused(422, "invalid_request_error", f"q: 1 to {MAX_QUERY} characters", "q")
        try:
            k = int(q.get("k") or 5)
        except ValueError:
            raise Refused(422, "invalid_request_error", "k: a whole number", "k") from None
        if not 1 <= k <= 10:
            raise Refused(422, "invalid_request_error", "k: 1 to 10", "k")
        hits = store.search(ctx["user"], text, project=ctx["project"], session=ctx["session"],
                            k=k)
        out = [{"id": n.id, "kind": n.kind, "scope": n.scope, "status": n.status,
                "title": n.title, "score": round(s, 3)} for n, s in hits]
        return "api_search", 200, out, {"q": hashlib.sha256(text.encode()).hexdigest()[:12],
                                        "ids": [h["id"] for h in out]}
    nid = unquote(path[len(PREFIX) + 1:])
    n = store.get(ctx["user"], nid)
    if n is None or not store.visible(n, ctx["project"], ctx["session"]):
        raise Refused(404, "not_found", f"no note {nid!r} in this conversation's memory; "
                                        f"GET {PREFIX}/search?q=...")
    out = {k: v for k, v in n.full().items() if k not in ("sources",)}
    store.mark_used(ctx["user"], [n.id], "read")
    return "api_read", 200, out, {"ids": [n.id]}


def handle(handler, cond, method: str, body: dict | None) -> None:
    t0 = time.perf_counter()
    task, err, detail = "api_" + method.lower(), None, {}
    try:
        task, status, out, detail = answer(cond, handler, method, body)
    except Refused as exc:
        err, status, out = exc, exc.status, exc.body()
    detail.update(status=status, ms=round((time.perf_counter() - t0) * 1e3, 3))
    if err is not None:
        detail.update(error=err.kind)
    sess = getattr(handler, "_sess", None)
    if sess is not None:
        reg = cond.router.session_tokens
        if reg is not None:
            reg.note(sess, "memory")
    cond.journal.note("memory_notes", task, decision=err.kind if err else "ok", detail=detail,
                      **({"session_key": sess} if sess is not None else {}))
    extra = (("Cache-Control", "no-store"),)
    if err is not None and err.retry:
        extra += (("Retry-After", str(err.retry)),)
    handler._json(status, out, extra_headers=extra)
