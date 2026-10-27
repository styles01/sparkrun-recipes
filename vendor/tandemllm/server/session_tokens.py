"""Which session an out-of-band call belongs to (the design: Memo "Session
identity", 2026-10-03).

THE GAP. The model fetches a library skill (`GET /v1/library/<name>`) or asks Laya
(`POST /v1/laya/ask`) with curl or a script from the user's machine. Those calls carry no session
id, so the engine could not tell which conversation asked: with two sessions served in turn (or a
Hermes chat beside an opencode session) the journal rows of the calls were nobody's.

THE TOKEN. Every session the router plans (its key: `c:<client id>` or `h:<head hash>`) gets a
short random token, and the address the runtime notes give the model (`{tandem_url}` in a package's
text, and the `{tandem_url}` a library skill's reference names) is the request's own address plus
`/s/<token>`: `http://host:8000/s/Ab3x.../v1/library/<name>`, `${TANDEM_URL}/v1/laya/ask`. A path
prefix rather than a query string, so every form the shipped texts and kits use (a shell variable,
`URL + "/v1/laya/ask"`, the kits' `TANDEM_URL` with a path) carries it without an edit. The server
strips `/s/<token>` before routing; a known token names the session in the call's journal row and
in the session's counts, an unknown one (a token from before a restart that was not kept, a
typo) is served all the same, attributed to nobody. A token is 12 characters of `secrets`
(72 bits), reveals nothing of the key, lives 48 h after its last use, and is kept in a 0600 file
next to the conductor's documents, so a note already in a conversation keeps working after a
restart. One token a key, one key a token: two sessions served at once never share one.

A KEY THAT SURVIVES A COMPACTION (`Aliases`). A client without a session id is keyed by its head
(`h:`): the messages up to its first user message. A client-side compaction replaces that history
with the summary, so the next request has another head and lost its session (class, notes,
token). The engine wrote the summary: after a compaction request its first `SNIPPET` characters
are kept with the session key, and a request whose `h:` key is new and whose first messages carry
that text is the same session from then on.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from collections import OrderedDict

TTL_S = 48 * 3600.0
MAX_SESSIONS = 4096
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,32}$")
PREFIX = "/s/"
SNIPPET = 200
#: messages of a new `h:` request read for a summary the engine wrote (the summary opens the
#: compacted history: opencode puts it in the first user message, others after a system text)
ALIAS_MESSAGES = 4


def _fold(s: str) -> str:
    return " ".join(s.split())


class SessionTokens:
    def __init__(self, path: str | None = None, clock=time.time):
        self.path, self.clock = path, clock
        self.lock = threading.Lock()
        self.by_key: OrderedDict[str, dict] = OrderedDict()
        self.by_token: dict[str, str] = {}
        self.stats = {"issued": 0, "calls": 0, "unknown": 0, "expired": 0}
        self._dirty_at = 0.0
        if path:
            try:
                with open(path) as f:
                    for key, e in json.load(f).get("sessions", {}).items():
                        if TOKEN_RE.match(e.get("token", "")):
                            self.by_key[key] = {"token": e["token"], "used": float(e["used"]),
                                                "calls": dict(e.get("calls") or {})}
                            self.by_token[e["token"]] = key
            except (OSError, ValueError, AttributeError):
                pass
            self._sweep()

    # -------------------------------------------------------------- the registry
    def _sweep(self) -> None:
        now = self.clock()
        for key in [k for k, e in self.by_key.items() if now - e["used"] > TTL_S]:
            self.by_token.pop(self.by_key.pop(key)["token"], None)
            self.stats["expired"] += 1
        while len(self.by_key) > MAX_SESSIONS:
            key, e = self.by_key.popitem(last=False)
            self.by_token.pop(e["token"], None)

    def _save(self, force: bool = False) -> None:
        if not self.path:
            return
        now = self.clock()
        if not force and now - self._dirty_at < 30.0:
            return
        self._dirty_at = now
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump({"sessions": dict(self.by_key)}, f)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def token_for(self, key: str) -> str:
        """The session's token, issued at its first request."""
        with self.lock:
            e = self.by_key.get(key)
            new = e is None
            if new:
                tok = secrets.token_urlsafe(9)
                while tok in self.by_token:
                    tok = secrets.token_urlsafe(9)
                e = self.by_key[key] = {"token": tok, "used": self.clock(), "calls": {}}
                self.by_token[tok] = key
                self.stats["issued"] += 1
                self._sweep()
            e["used"] = self.clock()
            self.by_key.move_to_end(key)
            self._save(force=new)
            return e["token"]

    def key_for(self, token: str | None) -> str | None:
        with self.lock:
            key = self.by_token.get(token or "")
            if key is not None and self.clock() - self.by_key[key]["used"] > TTL_S:
                return None
            return key

    def note(self, key: str | None, kind: str) -> None:
        """One out-of-band call of the session (`library`, `laya`)."""
        with self.lock:
            if key is None or key not in self.by_key:
                self.stats["unknown"] += 1
                return
            e = self.by_key[key]
            e["calls"][kind] = e["calls"].get(kind, 0) + 1
            e["used"] = self.clock()
            self.stats["calls"] += 1
            self._save()

    def calls(self, key: str) -> dict:
        with self.lock:
            e = self.by_key.get(key)
            return dict(e["calls"]) if e else {}

    def report(self) -> dict:
        with self.lock:
            return {"sessions": len(self.by_key), **self.stats}

    def close(self) -> None:
        with self.lock:
            self._save(force=True)


def strip(path: str) -> tuple[str, str | None]:
    """`/s/<token>/rest` -> (`/rest`, token); any other path as it is, None."""
    if not path.startswith(PREFIX):
        return path, None
    rest = path[len(PREFIX):]
    tok, sep, tail = rest.partition("/")
    if not TOKEN_RE.match(tok):
        return path, None
    return "/" + tail if sep else "/", tok


def with_token(url: str, token: str | None) -> str:
    return f"{url.rstrip('/')}{PREFIX}{token}" if token else url


class Aliases:
    """`h:` keys that continue a session after a compaction the engine wrote (see the module)."""

    def __init__(self, keep: int = 256):
        self.keep = keep
        self.lock = threading.Lock()
        self.snippets: OrderedDict[str, str] = OrderedDict()   # summary snippet -> session key
        self.alias: OrderedDict[str, str] = OrderedDict()      # h: key -> session key

    def compacted(self, key: str, summary: str) -> None:
        s = _fold(summary)[:SNIPPET]
        if len(s) < 40:
            return
        with self.lock:
            self.snippets[s] = key
            while len(self.snippets) > self.keep:
                self.snippets.popitem(last=False)

    def resolve(self, key: str, messages: list, known=lambda k: False) -> str:
        """The session key a request belongs to: an `h:` key aliased before, or one whose first
        messages carry a summary the engine wrote; any other key as it is."""
        if not key.startswith("h:"):
            return key
        with self.lock:
            if key in self.alias:
                return self.alias[key]
            if not self.snippets or known(key):
                return key
            texts = []
            for m in (messages or [])[:ALIAS_MESSAGES]:
                c = m.get("content") if isinstance(m, dict) else None
                if isinstance(c, list):
                    c = "".join(p.get("text", "") for p in c if isinstance(p, dict)
                                and isinstance(p.get("text"), str))
                if isinstance(c, str):
                    texts.append(_fold(c))
            blob = "\n".join(texts)
            for snip, owner in reversed(self.snippets.items()):
                if snip in blob:
                    self.alias[key] = owner
                    while len(self.alias) > self.keep:
                        self.alias.popitem(last=False)
                    return owner
        return key
