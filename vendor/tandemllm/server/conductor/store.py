"""Versioned configuration files for the conductor, outside every git tree (SRV-119).

    ~/.qwen38-spark-engine/conductor/
        settings/0001.json  settings/current
        router/0003.json    router/current          one document each
        roles/...           packages/...            monitor/... (the session monitor's policy)
        residency/...       (OPS-41: the residency planner's limits and claims)
        deciders/...        (SRV-131: the decision models' load policy, the model per question)
        profiles/coding/0004.json  profiles/coding/current

A version file is `{"version", "note", "author", "created_at", "restored_from", "doc"}`, JSON so
the server stays standard-library only (YAML was in the ticket; JSON is valid YAML and the API
speaks it anyway). `current` holds the version number and nothing else. A missing `current` means
version 0: the built-in document from `schema.seed` -- for `monitor`, the server's policy file
(`--conductor-policy`, else ops/supervisor.yaml), given to the store as `monitor_seed`.

Writes are serialised by one lock and never change a document a request already holds: `get()`
returns the parsed document, and a request keeps the one it was given (profiles are resolved once
per request and frozen). A write goes to a temp file first and is renamed into place, the version
file before the pointer, so a reader sees the old version or the new one and never half of one.

Hot reload: `get()` stats the `current` file (a few microseconds) and re-reads the document when
its mtime or the number in it moved, so a restore done by hand on the box, or by another process,
is picked up by the next request. A file that does not parse or does not validate is refused: the
last good version stays loaded, and `status()` reports the error, the file and the time, which
`GET /v1/conductor/config` shows.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import re
import threading

from server.conductor import schema

DEFAULT_ROOT = "~/.qwen38-spark-engine/conductor"
KINDS = ("settings", "router", "roles", "packages", "monitor", "profiles", "residency",
         "deciders")
NOTE_MAX = 200


class ConfigError(Exception):
    """A write or restore the store refuses: `status` is the HTTP answer (404, 409, 422)."""

    def __init__(self, status: int, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.status = status
        self.errors = errors or []

    def body(self) -> dict:
        kind = {404: "not_found", 409: "conflict", 422: "invalid_config"}.get(self.status,
                                                                                "bad_request")
        out = {"error": {"type": kind, "message": str(self)}}
        if self.errors:
            out["error"]["errors"] = self.errors[:50]
        return out


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Loaded:
    __slots__ = ("version", "doc", "meta", "ptr", "loaded_at", "error")

    def __init__(self, version, doc, meta, ptr):
        # `ptr`: the (version, mtime) of the `current` file this was loaded for, None for none
        self.version, self.doc, self.meta, self.ptr = version, doc, meta, ptr
        self.loaded_at = _now_iso()
        self.error: dict | None = None


class Store:
    """All conductor documents under one root. Thread-safe; cheap to read."""

    def __init__(self, root: str = DEFAULT_ROOT, *, readonly: bool = False, roles_up=None,
                 monitor_seed: dict | None = None, engine_tokenizer=None, on_defaults=None):
        self.root = os.path.realpath(os.path.expanduser(root))
        self.readonly = readonly
        # version 0 of the monitor's policy: the file the server was started with
        self.monitor_seed = monitor_seed
        # role -> bool, asked on a profile write (SRV-119: act on an offline role is refused)
        self.roles_up = roles_up
        # SRV-133: the engine's tokenizer hash, asked on a roles write (a role that shares token
        # ids must have it), and who hears which defaults a stored version was loaded with
        self.engine_tokenizer = engine_tokenizer
        self.on_defaults = on_defaults
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str], _Loaded] = {}
        self.writes = 0

    # ------------------------------------------------------------------ paths
    def _dir(self, kind: str, name: str | None) -> str:
        if kind not in KINDS:
            raise ConfigError(404, f"no document kind {kind!r}")
        if kind == "profiles":
            if not name or not re.match(schema.ID_RE, name):
                raise ConfigError(404, f"no profile {name!r}")
            return os.path.join(self.root, "profiles", name)
        return os.path.join(self.root, kind)

    @staticmethod
    def _vfile(d: str, v: int) -> str:
        return os.path.join(d, f"{v:04d}.json")

    def _pointer(self, d: str) -> tuple[int, float] | None:
        p = os.path.join(d, "current")
        try:
            st = os.stat(p)
            with open(p) as f:
                return int(f.read().strip() or "0"), st.st_mtime
        except (OSError, ValueError):
            return None

    def _seed(self, kind: str, name: str | None = None) -> dict | None:
        if kind == "monitor" and self.monitor_seed is not None:
            return copy.deepcopy(self.monitor_seed)
        return schema.seed(kind, name)

    # ------------------------------------------------------------------ reads
    def names(self) -> list[str]:
        """Every profile: the built-ins and whatever has a directory on disk."""
        out = set(schema.PROFILES)
        d = os.path.join(self.root, "profiles")
        try:
            for n in os.listdir(d):
                if re.match(schema.ID_RE, n) and os.path.isdir(os.path.join(d, n)):
                    out.add(n)
        except OSError:
            pass
        return sorted(out)

    def exists(self, kind: str, name: str | None = None) -> bool:
        if kind != "profiles":
            return kind in KINDS
        return name in schema.PROFILES or self._pointer(self._dir(kind, name)) is not None

    def get(self, kind: str, name: str | None = None) -> tuple[int, dict]:
        """(version, a copy of the document) of the current version; version 0 is the built-in
        seed. A copy, so an editor can change it and a refused write leaves nothing behind."""
        loaded = self._load(kind, name)
        return loaded.version, copy.deepcopy(loaded.doc)

    def peek(self, kind: str, name: str | None = None) -> tuple[int, dict]:
        """`get()` without the copy, for the request path: the caller only reads it. A write
        replaces the cached object, so a request keeps the version it started with."""
        loaded = self._load(kind, name)
        return loaded.version, loaded.doc

    def _load(self, kind: str, name: str | None) -> _Loaded:
        d = self._dir(kind, name)
        key = (kind, name or "")
        ptr = self._pointer(d)
        cached = self._cache.get(key)
        if cached is not None and cached.ptr == ptr:
            return cached
        with self._lock:
            if ptr is None or ptr[0] == 0:
                doc = self._seed(kind, name)
                if doc is None:
                    raise ConfigError(404, f"no {kind[:-1] if kind == 'profiles' else kind} "
                                           f"{name!r}")
                got = _Loaded(0, doc, {"version": 0, "note": "built-in defaults",
                                       "author": "server", "created_at": None,
                                       "restored_from": None}, ptr)
                self._cache[key] = got
                return got
            v = ptr[0]
            path = self._vfile(d, v)
            try:
                with open(path) as f:
                    rec = json.load(f)
                doc, applied = schema.upgrade(kind, rec["doc"])
                errs = schema.doc_errors(kind, doc)
                if errs:
                    raise ValueError("; ".join(errs[:3]))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # keep the last good version; say what is wrong and where
                keep = cached or self._fallback(kind, name, d, v)
                keep.error = {"file": path, "message": f"{type(exc).__name__}: {exc}"[:400],
                              "at": _now_iso(), "pointer": v}
                keep.ptr = ptr                   # do not re-read the broken file every request
                self._cache[key] = keep
                print(f"[conductor] {kind}{'/' + name if name else ''}: version {v} is unusable "
                      f"({keep.error['message']}); keeping version {keep.version}", flush=True)
                return keep
            meta = {k: rec.get(k) for k in ("version", "note", "author", "created_at",
                                            "restored_from")}
            got = _Loaded(v, doc, meta, ptr)
            self._cache[key] = got
            if applied:
                self._defaults(kind, name, v, applied)
                got.meta["defaults"] = applied
            return got

    def _defaults(self, kind, name, v: int, applied: list[str]) -> None:
        """A stored version loaded with defaults for fields it predates: one log line, and the
        journal row the conductor writes (`on_defaults`)."""
        print(f"[conductor] {kind}{'/' + name if name else ''} v{v}: loaded with defaults for "
              f"{len(applied)} field(s): {'; '.join(applied)}", flush=True)
        if self.on_defaults is not None:
            try:
                self.on_defaults(kind, name, v, applied)
            except Exception as exc:                               # noqa: BLE001
                print(f"[conductor] the defaults row failed: {type(exc).__name__}: {exc}",
                      flush=True)

    def _fallback(self, kind, name, d, bad: int) -> _Loaded:
        """The newest version below `bad` that parses and validates, else the seed."""
        for v in range(bad - 1, 0, -1):
            try:
                with open(self._vfile(d, v)) as f:
                    rec = json.load(f)
                doc, applied = schema.upgrade(kind, rec["doc"])
                if not schema.doc_errors(kind, doc):
                    got = _Loaded(v, doc, {k: rec.get(k) for k in (
                        "version", "note", "author", "created_at", "restored_from")}, None)
                    if applied:
                        self._defaults(kind, name, v, applied)
                        got.meta["defaults"] = applied
                    return got
            except (OSError, ValueError, KeyError, TypeError):
                continue
        doc = self._seed(kind, name) or {}
        return _Loaded(0, doc, {"version": 0, "note": "built-in defaults", "author": "server",
                                "created_at": None, "restored_from": None}, None)

    def meta(self, kind: str, name: str | None = None) -> dict:
        loaded = self._load(kind, name)
        return dict(loaded.meta, loaded_at=loaded.loaded_at, error=loaded.error)

    def versions(self, kind: str, name: str | None = None) -> list[dict]:
        """Every version on disk, newest first, plus the built-in version 0."""
        d = self._dir(kind, name)
        out = []
        try:
            files = sorted((f for f in os.listdir(d) if re.match(r"^\d{4,}\.json$", f)),
                           reverse=True)
        except OSError:
            files = []
        cur = (self._pointer(d) or (0, 0))[0]
        for f in files:
            try:
                with open(os.path.join(d, f)) as fh:
                    rec = json.load(fh)
                out.append({k: rec.get(k) for k in ("version", "note", "author", "created_at",
                                                    "restored_from")} | {"current":
                                                                         rec.get("version") == cur})
            except (OSError, ValueError):
                out.append({"version": int(f[:-5]), "note": "(unreadable)", "author": None,
                            "created_at": None, "restored_from": None, "current": False})
        if self._seed(kind, name) is not None:
            out.append({"version": 0, "note": "built-in defaults", "author": "server",
                        "created_at": None, "restored_from": None, "current": cur == 0})
        return out

    def version(self, kind: str, name: str | None, v: int) -> dict:
        """One version's document, for a diff or a restore; a version stored before a field
        existed comes with the field's default (a restore writes it whole)."""
        if v == 0:
            doc = self._seed(kind, name)
            if doc is None:
                raise ConfigError(404, "no version 0")
            return doc
        try:
            with open(self._vfile(self._dir(kind, name), v)) as f:
                return schema.upgrade(kind, json.load(f)["doc"])[0]
        except (OSError, ValueError, KeyError):
            raise ConfigError(404, f"no version {v}")

    # ------------------------------------------------------------------ writes
    def put(self, kind: str, name: str | None, doc, *, note: str, author: str,
            base: int | None = None, restored_from: int | None = None) -> int:
        """Validate, write a new version, move the pointer. Returns the new version number.
        `base`: the version the editor started from; a newer current version is a 409, so two
        tabs cannot overwrite each other silently."""
        if self.readonly:
            raise ConfigError(409, "this server's conductor config is read-only")
        if kind == "profiles" and name in schema.LOCKED:
            raise ConfigError(409, f"profile {name!r} is locked: {schema.LOCKED[name]}")
        note = (note or "").strip()
        if not note:
            raise ConfigError(422, "a change note is required", ["$note: missing"])
        if len(note) > NOTE_MAX:
            raise ConfigError(422, f"the note is longer than {NOTE_MAX} characters")
        errs = self.check(kind, name, doc)
        if errs:
            raise ConfigError(422, errs[0], errs)
        d = self._dir(kind, name)
        with self._lock:
            cur = (self._pointer(d) or (0, 0))[0]
            if base is not None and base != cur:
                raise ConfigError(409, f"the {kind if kind != 'profiles' else 'profile'} changed "
                                       f"to version {cur} after you opened version {base}")
            os.makedirs(d, mode=0o700, exist_ok=True)
            nums = [int(f[:-5]) for f in os.listdir(d) if re.match(r"^\d{4,}\.json$", f)]
            v = max(nums + [cur]) + 1
            rec = {"version": v, "note": note, "author": author[:80] if author else "unknown",
                   "created_at": _now_iso(), "restored_from": restored_from, "doc": doc}
            tmp = self._vfile(d, v) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rec, f, indent=1, ensure_ascii=False)
            os.replace(tmp, self._vfile(d, v))
            ptmp = os.path.join(d, "current.tmp")
            with open(ptmp, "w") as f:
                f.write(f"{v}\n")
            os.replace(ptmp, os.path.join(d, "current"))
            self._cache.pop((kind, name or ""), None)
            self.writes += 1
        return v

    def restore(self, kind: str, name: str | None, v: int, *, author: str,
                note: str | None = None) -> int:
        """A new version equal to version `v`, with the note "restore of v<v>"."""
        doc = self.version(kind, name, v)
        return self.put(kind, name, doc, note=note or f"restore of v{v}", author=author,
                        restored_from=v)

    def check(self, kind: str, name: str | None, doc) -> list[str]:
        """The validation a write would run, against the other current documents."""
        _, router = self.peek("router")
        _, packages = self.peek("packages")
        profiles = {n: self.peek("profiles", n)[1] for n in self.names()}
        roles_up = self.roles_up() if (self.roles_up is not None and kind == "profiles") else None
        if kind == "router":
            router = doc
        if kind == "packages":
            packages = doc
        tok = self.engine_tokenizer() if (self.engine_tokenizer is not None and
                                          kind == "roles") else None
        return schema.check(kind, doc, name=name, router=router, packages=packages,
                            profiles=profiles, roles_up=roles_up, engine_tokenizer=tok)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        """What `GET /v1/conductor/config` reports: every document's loaded version and time,
        and any file that failed to load."""
        docs = {}
        for kind in KINDS:
            if kind == "profiles":
                for n in self.names():
                    docs[f"profiles/{n}"] = self._status1(kind, n)
            else:
                docs[kind] = self._status1(kind, None)
        return {"root": self.root, "readonly": self.readonly, "writes": self.writes,
                "documents": docs}

    def _status1(self, kind, name) -> dict:
        try:
            ld = self._load(kind, name)
        except ConfigError as exc:
            return {"version": None, "loaded_at": None, "error": {"message": str(exc)}}
        return {"version": ld.version, "loaded_at": ld.loaded_at, "error": ld.error,
                "note": ld.meta.get("note")}


def diff(a, b, path: str = "$") -> list[dict]:
    """Leaf differences between two documents: `{path, before, after}`; lists of objects with
    an `id` compare by id, other lists by position."""
    out: list[dict] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                out.append({"path": f"{path}.{k}", "before": None, "after": b[k]})
            elif k not in b:
                out.append({"path": f"{path}.{k}", "before": a[k], "after": None})
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if all(isinstance(x, dict) and "id" in x for x in a + b):
            ia, ib = {x["id"]: x for x in a}, {x["id"]: x for x in b}
            for k in list(ia) + [x for x in ib if x not in ia]:
                if k not in ia:
                    out.append({"path": f"{path}[{k}]", "before": None, "after": ib[k]})
                elif k not in ib:
                    out.append({"path": f"{path}[{k}]", "before": ia[k], "after": None})
                else:
                    out += diff(ia[k], ib[k], f"{path}[{k}]")
            if [x["id"] for x in a] != [x["id"] for x in b] and set(ia) == set(ib):
                out.append({"path": f"{path}(order)", "before": [x["id"] for x in a],
                            "after": [x["id"] for x in b]})
            return out
        if a != b:
            out.append({"path": path, "before": a, "after": b})
        return out
    if a != b:
        out.append({"path": path, "before": a, "after": b})
    return out

