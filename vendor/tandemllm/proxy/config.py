"""The proxy's configuration: one SQLite file for the GUI, an env overlay that wins field by field.

The store is `~/.tandemllm/config.sqlite3` (`--config-db`), WAL, mode 0600, separate from the usage
ledger. Its tables:

    backends         one row per backend: id, kind, label, url, switches, timeouts, harness
    backend_secrets  the backend's API key as AES-256-GCM ciphertext (proxy/secrets.py)
    models           (backend, model id): public id, present, source, context length, prices,
                     a fallback list of its own
    model_features   (backend, model, group): the harness switches P4 acts on
    routing_rules    ordered glob patterns: a model id that matches goes to the rule's backend
    users            the proxy's own API keys: a label, the key's sha256, a 4-character hint
    settings         small JSON values (the dashboard's choices)
    audit            who changed what and when, field names only, never a value of a key
    meta             schema_version, created_at

Migrations are forward only and keyed on meta.schema_version, as the ledger's are.

THE ENVIRONMENT WINS. At every start `apply_env` reads the `TANDEM_*` variables and writes them
over the rows (a backend is created when missing). Each field it sets is locked for the run: the
GUI shows the variable's name on the field and a save that changes it answers 422 naming the
variable. A key from the environment is held in memory for the run and never written to the file.
When a variable goes away the field unlocks and keeps the last value it had. Nothing else is
written by the environment, so a docker deploy with an env file and a laptop with the GUI are the
same code path.

    TANDEM_BACKENDS=deepseek,vllm1                  the backends the environment declares
    TANDEM_BACKEND_<ID>_KIND=deepseek               generic | tandemllm | vllm | sglang |
                                                    openai | deepseek
    TANDEM_BACKEND_<ID>_URL=https://api.deepseek.com/v1
    TANDEM_BACKEND_<ID>_KEY=sk-...                  held in memory, never stored
    TANDEM_BACKEND_<ID>_LABEL=DeepSeek
    TANDEM_BACKEND_<ID>_MODELS=deepseek-flash,deepseek-v4-pro
    TANDEM_BACKEND_<ID>_MODELS_MODE=discover        discover (GET /models, plus MODELS) | list
    TANDEM_BACKEND_<ID>_FALLBACK=openrouter         backends tried after this one, in order
    TANDEM_BACKEND_<ID>_MAX_INFLIGHT=none           requests at once, past it a fallback or a 503;
                                                    none (the default): no limit from the
                                                    proxy, the backend takes what it accepts
    TANDEM_BACKEND_<ID>_ENABLED=1
    TANDEM_BACKEND_<ID>_INJECT_USAGE=1              ask a stream for its usage frame
    TANDEM_BACKEND_<ID>_HARNESS=off                 off | shadow | on (the harness, P4)
    TANDEM_BACKEND_<ID>_HARNESS_PROFILE=coding      the conductor profile of its requests
    TANDEM_BACKEND_<ID>_WRITER=auto                 the background memory writer on this backend
                                                    (proxy/memwriter.py): auto (on, off for a
                                                    paid cloud model) | off | shadow | on
    TANDEM_BACKEND_<ID>_WRITER_USD_DAY=0.25         its spend cap a day on a paid model
    TANDEM_BACKEND_<ID>_WRITER_RUNS_DAY=48          its runs a day
    TANDEM_BACKEND_<ID>_WRITER_PROMPT_TOKENS=32000  its prompt at most (older messages dropped)
    TANDEM_BACKEND_<ID>_TIMEOUT_CONNECT=10          seconds; also _TIMEOUT_FIRST_BYTE, _TIMEOUT_IDLE
    TANDEM_BACKEND_<ID>_HOST=auto                   where the System view reads the backend's
                                                    machine: auto (this machine when the URL is),
                                                    same, collector, exporter, off
    TANDEM_BACKEND_<ID>_HOST_URL=http://gpu1:9108   the remote collector (python -m
                                                    proxy.hostcollect), or a node_exporter page
    TANDEM_BACKEND_<ID>_HOST_DCGM_URL=http://gpu1:9400/metrics
                                                    a DCGM exporter page (mode exporter)
    TANDEM_COLLECTOR_TOKEN=...                      the remote collectors' bearer token, in
                                                    memory only (proxy/hostcollect.py)
    TANDEM_ROUTE_<n>="gpt-*=openai"                 routing rules, ordered by n
    TANDEM_PRICE_<n>="deepseek/deepseek-flash=0.30,0.006,1.2"
                                                    USD per million: input, cached input, output,
                                                    and optionally cache write
    TANDEM_PRICE_<n>_OFFPEAK="0.15,0.003,0.6"       the same model outside the provider's peak
                                                    hours (DeepSeek)
    TANDEM_API_KEYS="laptop:tdm-...,ci:tdm-..."     the proxy's own keys, in memory only
    TANDEM_SECRETS_KEY=<base64 of 32 bytes>         the master key (proxy/secrets.py)

`<ID>` is the backend id in capitals with `-` written as `_`. An unknown field, an unknown backend
or a value the schema refuses stops the start with a message that names the variable.
"""

from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
import os
import re
import secrets as _rand
import sqlite3
import threading
import time

from proxy import schema
from proxy import secrets as vault_mod

SCHEMA_VERSION = 2
DEFAULT_PATH = "~/.tandemllm/config.sqlite3"

# Per kind: is it a cloud API, its default base URL, and whether a stream is asked for its usage
# frame by default (our own engine puts usage on the finish chunk unasked)
KINDS = {
    "generic": {"cloud": False, "url": None, "inject_usage": True},
    "tandemllm": {"cloud": False, "url": None, "inject_usage": False},
    "vllm": {"cloud": False, "url": None, "inject_usage": True},
    "sglang": {"cloud": False, "url": None, "inject_usage": True},
    "openai": {"cloud": True, "url": "https://api.openai.com/v1", "inject_usage": True},
    "deepseek": {"cloud": True, "url": "https://api.deepseek.com/v1", "inject_usage": True},
}
TIMEOUTS = {False: {"connect": 5.0, "first_byte": 600.0, "idle": 120.0},
            True: {"connect": 10.0, "first_byte": 120.0, "idle": 60.0}}
# A backend's requests at once: None, the default, is no limit from the proxy (a cloud API, vLLM or
# SGLang takes as many as it accepts and answers its own 429/503); a number 1..4096 is a cap the
# user set. Stored as 0 for none (the column predates the choice). Until schema 2 the default was
# 64; a stored 64 becomes none in the migration (PL1, 2026-10-06).
MAX_INFLIGHT = None
OLD_DEFAULT_INFLIGHT = 64
FEATURE_GROUPS = ("mode", "class", "effort", "packages", "side", "loops", "e1", "tools",
                  "compaction", "states", "memory")
KEY_PREFIX = "tdm-"

DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS backends (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT, url TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1, models_mode TEXT NOT NULL DEFAULT 'discover',
  models TEXT NOT NULL DEFAULT '[]', fallback TEXT NOT NULL DEFAULT '[]',
  max_inflight INTEGER NOT NULL DEFAULT 0, timeouts TEXT NOT NULL DEFAULT '{}',
  inject_usage INTEGER, harness TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, updated_by TEXT,
  host TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS backend_secrets (
  backend_id TEXT PRIMARY KEY REFERENCES backends(id) ON DELETE CASCADE,
  ciphertext BLOB NOT NULL, nonce BLOB NOT NULL, key_id TEXT NOT NULL,
  last4 TEXT, sha256_6 TEXT, updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS models (
  backend_id TEXT NOT NULL REFERENCES backends(id) ON DELETE CASCADE, model_id TEXT NOT NULL,
  public_id TEXT, enabled INTEGER NOT NULL DEFAULT 1, present INTEGER NOT NULL DEFAULT 1,
  source TEXT NOT NULL, discovered_at INTEGER, context_length INTEGER,
  price_input REAL, price_cached_input REAL, price_cache_write REAL, price_output REAL,
  currency TEXT, price_updated_at INTEGER, price_offpeak TEXT, fallback TEXT,
  PRIMARY KEY (backend_id, model_id));
CREATE TABLE IF NOT EXISTS model_features (
  backend_id TEXT NOT NULL, model_id TEXT NOT NULL, feature TEXT NOT NULL, value TEXT NOT NULL,
  PRIMARY KEY (backend_id, model_id, feature));
CREATE TABLE IF NOT EXISTS routing_rules (
  id INTEGER PRIMARY KEY, position INTEGER NOT NULL, pattern TEXT NOT NULL,
  backend_id TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, source TEXT NOT NULL,
  env_var TEXT);
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, label TEXT NOT NULL UNIQUE, key_hash TEXT NOT NULL UNIQUE,
  key_hint TEXT, role TEXT NOT NULL DEFAULT 'user', enabled INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL, last_seen_at INTEGER);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,
  updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, who TEXT, action TEXT NOT NULL,
  object TEXT NOT NULL, object_id TEXT, fields TEXT);
"""


# Columns of schema v1 added in place to a file that lacks them (forward only, additive): a later
# release adds its own here instead of bumping the version for a nullable column.
ADDED = (("models", "price_offpeak", "TEXT"),
         ("backends", "host", "TEXT NOT NULL DEFAULT '{}'"),       # P1.5
         # 0.11.3: what a provider's model list says of a model (DeepSeek's GET /models): its
         # own name, output limit, input kinds (JSON list) and effort levels (JSON object)
         ("models", "name", "TEXT"), ("models", "max_output", "INTEGER"),
         ("models", "input_modalities", "TEXT"), ("models", "effort", "TEXT"),
         # VIS-227: the operator's output limit, apart from the provider's so a discovery never
         # overwrites it
         ("models", "max_output_set", "INTEGER"))


def _inflight_col(v) -> int:
    """The column's value: 0 for no limit."""
    return 0 if v is None else int(v)


class ConfigError(Exception):
    """A value the schema or a rule refuses. `field` is the dotted field; `var` the variable."""

    def __init__(self, message: str, field: str | None = None, var: str | None = None,
                 status: int = 422):
        super().__init__(message)
        self.field, self.var, self.status, self.message = field, var, status, message

    def body(self) -> dict:
        err = {"type": "invalid_config" if self.status == 422 else "not_found",
               "message": self.message}
        if self.field:
            err["field"] = self.field
        if self.var:
            err["variable"] = self.var
        return {"error": err}


def env_name(backend_id: str) -> str:
    return backend_id.upper().replace("-", "_")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _bool(text: str, var: str) -> bool:
    t = text.strip().lower()
    if t in ("1", "true", "yes", "on"):
        return True
    if t in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{var}: '{text}' is not a switch (1/0, true/false, on/off)", var=var)


def _list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def norm_url(url: str) -> str:
    """A base URL with no trailing slash; a bare host gets `/v1` (the OpenAI SDK's base)."""
    u = url.strip().rstrip("/")
    m = re.match(r"^(https?://[^/]+)(/.*)?$", u)
    if m and not m.group(2):
        u += "/v1"
    return u


# The env fields of a backend: variable suffix -> (field, parser)
def _env_fields():
    return {
        "KIND": ("kind", lambda v, var: v.strip()),
        "URL": ("url", lambda v, var: v.strip()),
        "LABEL": ("label", lambda v, var: v.strip()),
        "MODELS": ("models", lambda v, var: _list(v)),
        "MODELS_MODE": ("models_mode", lambda v, var: v.strip()),
        "FALLBACK": ("fallback", lambda v, var: _list(v)),
        "MAX_INFLIGHT": ("max_inflight", lambda v, var: _inflight(v, var)),
        "ENABLED": ("enabled", _bool),
        "INJECT_USAGE": ("inject_usage", _bool),
        "HARNESS": ("harness.mode", lambda v, var: v.strip()),
        "HARNESS_PROFILE": ("harness.profile", lambda v, var: v.strip() or None),
        "WRITER": ("harness.writer", lambda v, var: v.strip()),
        "WRITER_USD_DAY": ("harness.writer_usd_day", lambda v, var: _num(v, var)),
        "WRITER_RUNS_DAY": ("harness.writer_runs_day", lambda v, var: _int(v, var)),
        "WRITER_PROMPT_TOKENS": ("harness.writer_prompt_tokens", lambda v, var: _int(v, var)),
        "TIMEOUT_CONNECT": ("timeouts.connect", lambda v, var: _num(v, var)),
        "TIMEOUT_FIRST_BYTE": ("timeouts.first_byte", lambda v, var: _num(v, var)),
        "TIMEOUT_IDLE": ("timeouts.idle", lambda v, var: _num(v, var)),
        "HOST": ("host.mode", lambda v, var: v.strip()),
        "HOST_URL": ("host.url", lambda v, var: v.strip()),
        "HOST_DCGM_URL": ("host.dcgm_url", lambda v, var: v.strip()),
    }


def _int(v: str, var: str) -> int:
    try:
        return int(v.strip())
    except ValueError:
        raise ConfigError(f"{var}: '{v}' is not an integer", var=var)


def _inflight(v: str, var: str) -> int | None:
    """A backend's cap from the environment: none, off or empty is no limit, else an integer."""
    if v.strip().lower() in ("", "none", "off", "unlimited"):
        return None
    return _int(v, var)


def _num(v: str, var: str) -> float:
    try:
        return float(v.strip())
    except ValueError:
        raise ConfigError(f"{var}: '{v}' is not a number", var=var)


def parse_env(env) -> dict:
    """The overlay the environment declares, validated; ConfigError names the variable."""
    out = {"backends": {}, "keys": {}, "rules": [], "prices": [], "users": [], "vars": {}}
    ids = _list(env.get("TANDEM_BACKENDS", ""))
    by_name = {}
    for bid in ids:
        if schema.validate(bid, "backend_id"):
            raise ConfigError(f"TANDEM_BACKENDS: '{bid}' is not a backend id (lower case letters, "
                              f"digits, - and _, at most 32)", var="TANDEM_BACKENDS")
        name = env_name(bid)
        if name in by_name:
            raise ConfigError(f"TANDEM_BACKENDS: '{bid}' and '{by_name[name]}' share the "
                              f"variable prefix TANDEM_BACKEND_{name}_", var="TANDEM_BACKENDS")
        by_name[name] = bid
        out["backends"][bid] = {}
    fields = _env_fields()
    for var in sorted(env):
        if not var.startswith("TANDEM_BACKEND_"):
            continue
        rest = var[len("TANDEM_BACKEND_"):]
        if any(rest == name + "_KEY" for name in by_name):
            continue                                       # read below, held in memory only
        hit = None
        for name, bid in by_name.items():
            if rest.startswith(name + "_") and rest[len(name) + 1:] in fields:
                if hit is None or len(name) > len(hit[0]):
                    hit = (name, bid)
        if hit is None:
            raise ConfigError(f"{var}: no backend in TANDEM_BACKENDS with this prefix, or an "
                              f"unknown field (known: {', '.join(sorted(fields))})", var=var)
        name, bid = hit
        suffix = rest[len(name) + 1:]
        value = env[var]
        if suffix not in fields:
            raise ConfigError(f"{var}: unknown field", var=var)
        field, parse = fields[suffix]
        out["backends"][bid][field] = parse(value, var)
        out["vars"][("backend", bid, field)] = var
    for bid in ids:
        kvar = f"TANDEM_BACKEND_{env_name(bid)}_KEY"
        if env.get(kvar):
            out["keys"][bid] = env[kvar]
            out["vars"][("key", bid)] = kvar
    offpeak = {}
    for var in env:
        m = re.fullmatch(r"TANDEM_PRICE_(\d+)_OFFPEAK", var)
        if m:
            vals = [x.strip() for x in env[var].split(",")]
            if len(vals) not in (3, 4):
                raise ConfigError(f"{var}: write it as <input>,<cached input>,<output>[,<cache "
                                  f"write>]", var=var)
            p = [_num(x, var) for x in vals]
            offpeak[m.group(1)] = {"input": p[0], "cached_input": p[1], "output": p[2],
                                   "cache_write": p[3] if len(p) > 3 else None}
    for var in sorted(env, key=lambda v: (len(v), v)):
        m = re.fullmatch(r"TANDEM_ROUTE_(\d+)", var)
        if m:
            pat, sep, bid = env[var].rpartition("=")
            if not sep or not pat.strip() or not bid.strip():
                raise ConfigError(f"{var}: write it as <model pattern>=<backend id>", var=var)
            out["rules"].append({"n": int(m.group(1)), "pattern": pat.strip(),
                                 "backend": bid.strip(), "var": var})
        m = re.fullmatch(r"TANDEM_PRICE_(\d+)", var)
        if m:
            ref, sep, nums = env[var].partition("=")
            bid, slash, mid = ref.strip().partition("/")
            vals = [x.strip() for x in nums.split(",")]
            if not sep or not slash or len(vals) not in (3, 4):
                raise ConfigError(f"{var}: write it as <backend>/<model>=<input>,<cached input>,"
                                  f"<output>[,<cache write>] in USD per million tokens", var=var)
            p = [_num(x, var) for x in vals]
            out["prices"].append({"backend": bid, "model": mid, "var": var,
                                  "price": {"input": p[0], "cached_input": p[1], "output": p[2],
                                            "cache_write": p[3] if len(p) > 3 else None,
                                            "currency": "USD"}})
            if m.group(1) in offpeak:
                out["prices"][-1]["price"]["offpeak"] = offpeak.pop(m.group(1))
    for n in offpeak:
        raise ConfigError(f"TANDEM_PRICE_{n}_OFFPEAK has no TANDEM_PRICE_{n}",
                          var=f"TANDEM_PRICE_{n}_OFFPEAK")
    out["rules"].sort(key=lambda r: r["n"])
    if env.get("TANDEM_API_KEYS"):
        for item in _list(env["TANDEM_API_KEYS"]):
            label, sep, key = item.partition(":")
            if not sep or not label.strip() or len(key.strip()) < 16:
                raise ConfigError("TANDEM_API_KEYS: write it as label:key,label:key with keys of "
                                  "at least 16 characters", var="TANDEM_API_KEYS")
            out["users"].append({"label": label.strip(), "key": key.strip()})
    # each backend the environment declares must validate once its defaults are filled in
    for bid, fields_set in out["backends"].items():
        if "kind" not in fields_set:
            raise ConfigError(f"TANDEM_BACKEND_{env_name(bid)}_KIND is required for a backend "
                              f"in TANDEM_BACKENDS", var=f"TANDEM_BACKEND_{env_name(bid)}_KIND")
        doc = _nest(fields_set)
        doc["id"] = bid
        kind = doc.get("kind")
        if kind not in KINDS:
            var = f"TANDEM_BACKEND_{env_name(bid)}_KIND"
            raise ConfigError(f"{var}: '{kind}' is not a kind ({', '.join(KINDS)})", var=var,
                              field="kind")
        if "url" not in doc and KINDS[kind]["url"]:
            doc["url"] = KINDS[kind]["url"]
        if "url" not in doc:
            raise ConfigError(f"TANDEM_BACKEND_{env_name(bid)}_URL is required for kind '{kind}'",
                              var=f"TANDEM_BACKEND_{env_name(bid)}_URL")
        for path, msg in schema.validate(doc, "backend"):
            var = out["vars"].get(("backend", bid, path)) or \
                f"TANDEM_BACKEND_{env_name(bid)}_{path.upper().replace('.', '_')}"
            raise ConfigError(f"{var}: {msg}", var=var, field=path)
    for r in out["rules"]:
        for path, msg in schema.validate({"pattern": r["pattern"], "backend": r["backend"]},
                                         "rule"):
            raise ConfigError(f"{r['var']}: {path} {msg}", var=r["var"])
    return out


def _nest(flat: dict) -> dict:
    """{'timeouts.connect': 5, 'kind': 'x'} -> {'timeouts': {'connect': 5}, 'kind': 'x'}"""
    out: dict = {}
    for k, v in flat.items():
        if "." in k:
            a, b = k.split(".", 1)
            out.setdefault(a, {})[b] = v
        else:
            out[k] = v
    return out


class ConfigDB:
    """The store and its in-memory overlay. Thread safe: one connection under one lock; reads are
    short and writes are rare (a GUI click, a discovery every 10 minutes)."""

    def __init__(self, path: str = DEFAULT_PATH, *, env=None, vault=None, keychain_cmd=None,
                 use_keychain: bool = True, clock=time.time):
        self.path = os.path.realpath(os.path.expanduser(path))
        self.env = os.environ if env is None else env
        self._vault = vault
        self._keychain_cmd = keychain_cmd
        self._use_keychain = use_keychain
        self.clock = clock
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.generation = 0
        self.locks: dict = {}           # ("backend", id, field) | ("key", id) | ... -> variable
        self.env_keys: dict[str, str] = {}
        self.env_users: dict[str, str] = {}    # key hash -> label
        self.schema_was: int | None = None
        self._seen: dict[int, int] = {}        # user id -> last seen ms, flushed once a minute
        self._seen_flushed = 0.0

    # ----------------------------------------------------------------- lifecycle
    def open(self) -> "ConfigDB":
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._conn = conn
        self.schema_was = self.migrate()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return self

    def migrate(self) -> int:
        with self._lock:
            c = self._conn
            c.executescript(DDL)
            # columns added to v1 after its first files existed: in place, idempotent, no row touched
            for table, col, kind in ADDED:
                if col not in {r[1] for r in c.execute(f"PRAGMA table_info({table})")}:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {kind}")
            got = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            was = int(got[0]) if got else 0
            if 0 < was < 2:
                # schema 2: no in-flight cap unless the user set one. 64 was the default every
                # backend got without asking; it becomes none (a variable that sets it again is
                # applied over the row at this start, as always)
                c.execute("UPDATE backends SET max_inflight = 0 WHERE max_inflight = ?",
                          (OLD_DEFAULT_INFLIGHT,))
            if was < SCHEMA_VERSION:
                rows = [("schema_version", str(SCHEMA_VERSION))]
                if not was:
                    rows.append(("created_at", str(_now_ms())))
                c.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", rows)
            c.commit()
            return was

    def close(self) -> None:
        with self._lock:
            self.flush_seen(force=True)
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def vault(self) -> "vault_mod.Vault":
        if self._vault is None:
            self._vault = vault_mod.Vault.load(os.path.dirname(self.path), env=self.env,
                                               keychain_cmd=self._keychain_cmd,
                                               use_keychain=self._use_keychain)
        return self._vault

    def key_source(self) -> str | None:
        """Where the master key comes from, without creating a key file: env, keychain, file, or
        None when no key has been stored yet and nothing would answer but a new file."""
        if self._vault is not None:
            return self._vault.source
        if self.env.get(vault_mod.ENV):
            return "env"
        if os.path.exists(os.path.join(os.path.dirname(self.path), vault_mod.KEY_FILE)):
            return "file"
        return None

    def _bump(self) -> None:
        self.generation += 1

    def _audit(self, who: str | None, action: str, obj: str, oid: str | None,
               fields=None) -> None:
        self._conn.execute("INSERT INTO audit(ts_ms, who, action, object, object_id, fields) "
                           "VALUES (?, ?, ?, ?, ?, ?)",
                           (_now_ms(), who, action, obj, oid,
                            json.dumps(sorted(fields)) if fields else None))

    # ----------------------------------------------------------------- the env overlay
    def apply_env(self) -> dict:
        """Write the environment's fields over the rows and lock them for this run."""
        ov = parse_env(self.env)
        with self._lock:
            c = self._conn
            self.locks = dict(ov["vars"])
            for bid, flat in ov["backends"].items():
                fields = _nest(flat)
                row = c.execute("SELECT id FROM backends WHERE id = ?", (bid,)).fetchone()
                if row is None:
                    doc = dict(fields)
                    doc["id"] = bid
                    if "url" not in doc:
                        doc["url"] = KINDS[doc["kind"]]["url"]
                    self._insert_backend(doc, "env")
                else:
                    self._update_backend_fields(bid, fields, "env")
            self.env_keys = dict(ov["keys"])
            # rules: upsert by variable; an env rule whose variable went away unlocks and stays
            have = {r[0]: r[1] for r in c.execute(
                "SELECT env_var, id FROM routing_rules WHERE source = 'env'")}
            cur = set()
            for r in ov["rules"]:
                cur.add(r["var"])
                self.locks[("rule", r["var"])] = r["var"]
                if r["var"] in have:
                    c.execute("UPDATE routing_rules SET pattern = ?, backend_id = ?, enabled = 1 "
                              "WHERE id = ?", (r["pattern"], r["backend"], have[r["var"]]))
                else:
                    c.execute("INSERT INTO routing_rules(position, pattern, backend_id, enabled, "
                              "source, env_var) VALUES (?, ?, ?, 1, 'env', ?)",
                              (r["n"], r["pattern"], r["backend"], r["var"]))
            for var, rid in have.items():
                if var not in cur:
                    c.execute("UPDATE routing_rules SET source = 'gui', env_var = NULL "
                              "WHERE id = ?", (rid,))
            for p in ov["prices"]:
                if c.execute("SELECT 1 FROM backends WHERE id = ?", (p["backend"],)).fetchone() \
                        is None:
                    raise ConfigError(f"{p['var']}: no backend '{p['backend']}'", var=p["var"])
                self._ensure_model(p["backend"], p["model"], "env")
                self._set_price(p["backend"], p["model"], p["price"])
                self.locks[("price", p["backend"], p["model"])] = p["var"]
            self.env_users = {key_hash(u["key"]): u["label"] for u in ov["users"]}
            c.commit()
            self._bump()
        return ov

    def locked(self, *key) -> str | None:
        return self.locks.get(tuple(key))

    def backend_locks(self, bid: str) -> dict:
        out = {}
        for k, var in self.locks.items():
            if k[0] == "backend" and k[1] == bid:
                out[k[2]] = var
            elif k == ("key", bid):
                out["key"] = var
        return out

    # ----------------------------------------------------------------- backends
    def _insert_backend(self, doc: dict, who: str) -> None:
        now = _now_ms()
        kind = doc["kind"]
        self._conn.execute(
            "INSERT INTO backends(id, kind, label, url, enabled, models_mode, models, fallback, "
            "max_inflight, timeouts, inject_usage, harness, created_at, updated_at, updated_by, "
            "host) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc["id"], kind, doc.get("label"), norm_url(doc["url"]),
             int(doc.get("enabled", True)), doc.get("models_mode", "discover"),
             json.dumps(doc.get("models", [])), json.dumps(doc.get("fallback", [])),
             _inflight_col(doc.get("max_inflight", MAX_INFLIGHT)),
             json.dumps(doc.get("timeouts", {})),
             None if doc.get("inject_usage") is None else int(doc["inject_usage"]),
             json.dumps(doc.get("harness", {})), now, now, who, json.dumps(doc.get("host", {}))))
        self._audit(who, "create", "backend", doc["id"], [k for k in doc if k != "key"])

    def _update_backend_fields(self, bid: str, fields: dict, who: str) -> None:
        cols, args = [], []
        for k, v in fields.items():
            if k == "key":
                continue
            if k in ("timeouts", "harness", "host"):
                old = json.loads(self._conn.execute(f"SELECT {k} FROM backends WHERE id = ?",
                                                    (bid,)).fetchone()[0] or "{}")
                old.update(v or {})
                v = json.dumps(old)
            elif k in ("models", "fallback"):
                v = json.dumps(v)
            elif k in ("enabled", "inject_usage"):
                v = None if v is None else int(v)
            elif k == "url":
                v = norm_url(v)
            elif k == "max_inflight":
                v = _inflight_col(v)
            cols.append(f"{k} = ?")
            args.append(v)
        if not cols:
            return
        cols += ["updated_at = ?", "updated_by = ?"]
        args += [_now_ms(), who, bid]
        self._conn.execute(f"UPDATE backends SET {', '.join(cols)} WHERE id = ?", args)
        self._audit(who, "update", "backend", bid, [k for k in fields if k != "key"])

    def _row_backend(self, r) -> dict:
        (bid, kind, label, url, enabled, mmode, models, fallback, maxin, timeouts, inject,
         harness, created, updated, by, host) = r
        cloud = KINDS.get(kind, KINDS["generic"])["cloud"]
        to = dict(TIMEOUTS[cloud])
        to.update(json.loads(timeouts or "{}"))
        h = {"mode": "off", "profile": None}
        h.update(json.loads(harness or "{}"))
        return {"id": bid, "kind": kind, "label": label or bid, "url": url,
                "enabled": bool(enabled), "models_mode": mmode,
                "models": json.loads(models or "[]"), "fallback": json.loads(fallback or "[]"),
                "max_inflight": int(maxin) if maxin else None, "timeouts": to,
                "inject_usage": (KINDS.get(kind, KINDS["generic"])["inject_usage"]
                                 if inject is None else bool(inject)),
                "harness": h, "cloud": cloud,
                "host": dict({"mode": "auto", "url": None, "dcgm_url": None},
                             **json.loads(host or "{}")),
                "created_at": created, "updated_at": updated, "updated_by": by}

    _BCOLS = ("id, kind, label, url, enabled, models_mode, models, fallback, max_inflight, "
              "timeouts, inject_usage, harness, created_at, updated_at, updated_by, host")

    def backends(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(f"SELECT {self._BCOLS} FROM backends ORDER BY created_at, "
                                      f"id").fetchall()
        return [self._row_backend(r) for r in rows]

    def backend(self, bid: str) -> dict | None:
        with self._lock:
            r = self._conn.execute(f"SELECT {self._BCOLS} FROM backends WHERE id = ?",
                                   (bid,)).fetchone()
        return self._row_backend(r) if r else None

    def create_backend(self, doc: dict, who: str = "gui") -> dict:
        doc = dict(doc)
        kind = doc.get("kind")
        if not doc.get("url") and kind in KINDS and KINDS[kind]["url"]:
            doc["url"] = KINDS[kind]["url"]
        errs = schema.validate(doc, "backend")
        if errs:
            raise ConfigError(f"{errs[0][0]} {errs[0][1]}", field=errs[0][0])
        with self._lock:
            if self._conn.execute("SELECT 1 FROM backends WHERE id = ?",
                                  (doc["id"],)).fetchone():
                raise ConfigError(f"a backend '{doc['id']}' exists already", field="id",
                                  status=409)
            self._check_fallback(doc["id"], doc.get("fallback") or [])
            self._insert_backend(doc, who)
            if doc.get("key"):
                self._store_key(doc["id"], doc["key"])
            self._conn.commit()
            self._bump()
        return self.backend(doc["id"])

    def update_backend(self, bid: str, patch: dict, who: str = "gui") -> dict:
        errs = schema.validate(patch, "backend_patch")
        if errs:
            raise ConfigError(f"{errs[0][0]} {errs[0][1]}", field=errs[0][0])
        cur = self.backend(bid)
        if cur is None:
            raise ConfigError(f"no backend '{bid}'", status=404)
        locks = self.backend_locks(bid)
        for k, v in patch.items():
            subs = ([f"{k}.{s}" for s in v] if isinstance(v, dict) and k in
                    ("timeouts", "harness", "host") else [k])
            for f in subs:
                var = locks.get(f)
                if var is None:
                    continue
                if f == "key":
                    raise ConfigError(f"the key is set by {var}", field="key", var=var)
                a, _, b = f.partition(".")
                old = cur.get(a) if not b else (cur.get(a) or {}).get(b)
                new = v if not b else v.get(b)
                if k == "url":
                    old, new = norm_url(old or ""), norm_url(new or "")
                if old != new:
                    raise ConfigError(f"{f} is set by {var}; change the variable instead",
                                      field=f, var=var)
        with self._lock:
            if "fallback" in patch:
                self._check_fallback(bid, patch["fallback"])
            self._update_backend_fields(bid, {k: v for k, v in patch.items()
                                              if not locks.get(k)}, who)
            if "key" in patch:
                if patch["key"]:
                    self._store_key(bid, patch["key"])
                else:
                    self._conn.execute("DELETE FROM backend_secrets WHERE backend_id = ?", (bid,))
                self._audit(who, "update", "backend_key", bid, ["key"])
            self._conn.commit()
            self._bump()
        return self.backend(bid)

    def delete_backend(self, bid: str, who: str = "gui") -> None:
        if any(k[0] == "backend" and k[1] == bid for k in self.locks):
            raise ConfigError(f"backend '{bid}' is declared by TANDEM_BACKENDS; remove it there",
                              var="TANDEM_BACKENDS", status=422)
        with self._lock:
            if not self._conn.execute("SELECT 1 FROM backends WHERE id = ?", (bid,)).fetchone():
                raise ConfigError(f"no backend '{bid}'", status=404)
            for t, col in (("backend_secrets", "backend_id"), ("models", "backend_id"),
                           ("model_features", "backend_id"), ("routing_rules", "backend_id"),
                           ("backends", "id")):
                self._conn.execute(f"DELETE FROM {t} WHERE {col} = ?", (bid,))
            self._audit(who, "delete", "backend", bid)
            self._conn.commit()
            self._bump()

    def _check_fallback(self, bid: str, fb: list[str]) -> None:
        for i, ref in enumerate(fb):
            target = ref.split("/", 1)[0]
            if target == bid:
                raise ConfigError(f"fallback[{i}] names the backend itself",
                                  field=f"fallback[{i}]")

    # ----------------------------------------------------------------- keys
    def _store_key(self, bid: str, key: str) -> None:
        ct, nonce, kid = self.vault().encrypt(bid, key)
        h = vault_mod.hint(key)
        self._conn.execute("INSERT OR REPLACE INTO backend_secrets(backend_id, ciphertext, nonce, "
                           "key_id, last4, sha256_6, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (bid, ct, nonce, kid, h["last4"], h["sha256_6"], _now_ms()))

    def key_for(self, bid: str) -> str | None:
        """The backend's key in plain text, for the relay's header and nothing else. The env's
        first; None when there is none; SecretError when the stored one cannot be read."""
        if bid in self.env_keys:
            return self.env_keys[bid]
        with self._lock:
            r = self._conn.execute("SELECT ciphertext, nonce, key_id FROM backend_secrets WHERE "
                                   "backend_id = ?", (bid,)).fetchone()
        if r is None:
            return None
        return self.vault().decrypt(bid, bytes(r[0]), bytes(r[1]), r[2])

    def key_hint(self, bid: str) -> dict:
        if bid in self.env_keys:
            h = vault_mod.hint(self.env_keys[bid])
            h["source"] = "env"
            h["readable"] = True
            return h
        with self._lock:
            r = self._conn.execute("SELECT last4, sha256_6, key_id FROM backend_secrets WHERE "
                                   "backend_id = ?", (bid,)).fetchone()
        if r is None:
            return {"key_set": False, "last4": None, "sha256_6": None, "source": None,
                    "readable": None}
        readable = None
        src = self.key_source()
        if src is not None:
            try:
                readable = r[2] == self.vault().key_id
            except vault_mod.SecretError:
                readable = False
        return {"key_set": True, "last4": r[0], "sha256_6": r[1], "source": "database",
                "readable": readable}

    # ----------------------------------------------------------------- models
    def _ensure_model(self, bid: str, mid: str, source: str) -> None:
        self._conn.execute("INSERT OR IGNORE INTO models(backend_id, model_id, source, present) "
                           "VALUES (?, ?, ?, 1)", (bid, mid, source))

    def _set_price(self, bid: str, mid: str, price: dict | None) -> None:
        p = price or {}
        self._conn.execute("UPDATE models SET price_input = ?, price_cached_input = ?, "
                           "price_cache_write = ?, price_output = ?, currency = ?, "
                           "price_updated_at = ?, price_offpeak = ? WHERE backend_id = ? AND "
                           "model_id = ?",
                           (p.get("input"), p.get("cached_input"), p.get("cache_write"),
                            p.get("output"), p.get("currency", "USD") if price else None,
                            _now_ms() if price else None,
                            json.dumps(p["offpeak"]) if p.get("offpeak") else None, bid, mid))

    def sync_models(self, bid: str, found: list[dict] | None, listed: list[str],
                    mode: str) -> list[str]:
        """After a discovery (or a list change): the rows of this backend's models. `found` is
        None when discovery failed: the last list stays. Returns the ids the discovery found that
        the backend never had a row for. After the backend's first discovery, such a new id (DeepSeek's
        `deepseek-v4.1-flash-1017`, in its list for minutes on 2026-10-17) is stored disabled:
        nothing new reaches the picker until it is turned on under Backends."""
        now = _now_ms()
        new: list[str] = []
        with self._lock:
            c = self._conn
            want = set(listed)
            if mode == "discover" and found is not None:
                rows = c.execute("SELECT model_id, discovered_at FROM models WHERE "
                                 "backend_id = ?", (bid,)).fetchall()
                have = {r[0] for r in rows}
                seen = any(r[1] is not None for r in rows)    # a discovery came before this one
                for m in found:
                    want.add(m["id"])
                    if seen and m["id"] not in have and m["id"] not in listed:
                        new.append(m["id"])
                        c.execute("INSERT INTO models(backend_id, model_id, source, present, "
                                  "enabled) VALUES (?, ?, 'discovered', 1, 0)", (bid, m["id"]))
                    mods, eff = m.get("input_modalities"), m.get("effort")
                    c.execute("INSERT INTO models(backend_id, model_id, source, present, "
                              "discovered_at, context_length, name, max_output, input_modalities, "
                              "effort) VALUES (?, ?, 'discovered', 1, ?, ?, ?, ?, ?, ?) "
                              "ON CONFLICT(backend_id, model_id) DO UPDATE SET present = 1, "
                              "discovered_at = excluded.discovered_at, context_length = "
                              "COALESCE(excluded.context_length, models.context_length), name = "
                              "COALESCE(excluded.name, models.name), max_output = "
                              "COALESCE(excluded.max_output, models.max_output), input_modalities "
                              "= COALESCE(excluded.input_modalities, models.input_modalities), "
                              "effort = COALESCE(excluded.effort, models.effort)",
                              (bid, m["id"], now, m.get("context_length"), m.get("name"),
                               m.get("max_output"), json.dumps(mods) if mods is not None else None,
                               json.dumps(eff) if eff is not None else None))
            for mid in listed:
                c.execute("INSERT INTO models(backend_id, model_id, source, present) VALUES "
                          "(?, ?, 'list', 1) ON CONFLICT(backend_id, model_id) DO UPDATE SET "
                          "present = 1", (bid, mid))
            if mode == "list" or found is not None:
                rows = c.execute("SELECT model_id, source FROM models WHERE backend_id = ?",
                                 (bid,)).fetchall()
                for mid, src in rows:
                    if mid not in want and src in ("discovered", "list"):
                        c.execute("UPDATE models SET present = 0 WHERE backend_id = ? AND "
                                  "model_id = ?", (bid, mid))
            c.commit()
            self._bump()
        return new

    def models(self, bid: str | None = None) -> list[dict]:
        sql = ("SELECT backend_id, model_id, public_id, enabled, present, source, discovered_at, "
               "context_length, price_input, price_cached_input, price_cache_write, "
               "price_output, currency, price_updated_at, fallback, price_offpeak, name, "
               "max_output, input_modalities, effort, max_output_set FROM models")
        args: tuple = ()
        if bid is not None:
            sql += " WHERE backend_id = ?"
            args = (bid,)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY backend_id, model_id", args).fetchall()
            feats: dict = {}
            for b, m, f, v in self._conn.execute(
                    "SELECT backend_id, model_id, feature, value FROM model_features"):
                feats.setdefault((b, m), {})[f] = v
        out = []
        for (b, m, pub, en, present, src, disc, ctx, pi, pci, pcw, po, cur, pat, fb,
             off, name, mout, mods, eff, mset) in rows:
            price = None
            if pat is not None:
                price = {"input": pi, "cached_input": pci, "cache_write": pcw, "output": po,
                         "currency": cur or "USD", "updated_at": pat}
                if off:
                    price["offpeak"] = json.loads(off)
            f = {g: "inherit" for g in FEATURE_GROUPS}
            f.update(feats.get((b, m), {}))
            out.append({"backend": b, "model": m, "public_id": pub, "enabled": bool(en),
                        "present": bool(present), "source": src, "discovered_at": disc,
                        "context_length": ctx, "price": price,
                        # 0.11.3: the provider's own words, null until a discovery brought them
                        "name": name, "max_output": mout,
                        # VIS-227: the operator's output limit, the card's before the provider's
                        "max_output_set": mset,
                        "input_modalities": json.loads(mods) if mods else None,
                        "effort": json.loads(eff) if eff else None,
                        "fallback": json.loads(fb) if fb else None, "features": f,
                        "locks": {k: v for k, v in (("price", self.locks.get(("price", b, m))),)
                                  if v}})
        return out

    def update_model(self, bid: str, mid: str, patch: dict, who: str = "gui",
                     create: bool = False) -> dict:
        errs = schema.validate(patch, "model_patch")
        if errs:
            raise ConfigError(f"{errs[0][0]} {errs[0][1]}", field=errs[0][0])
        var = self.locks.get(("price", bid, mid))
        if var and "price" in patch:
            raise ConfigError(f"price is set by {var}; change the variable instead",
                              field="price", var=var)
        with self._lock:
            c = self._conn
            if not c.execute("SELECT 1 FROM backends WHERE id = ?", (bid,)).fetchone():
                raise ConfigError(f"no backend '{bid}'", status=404)
            exists = c.execute("SELECT 1 FROM models WHERE backend_id = ? AND model_id = ?",
                               (bid, mid)).fetchone()
            if not exists:
                if not create:
                    raise ConfigError(f"no model '{mid}' on backend '{bid}'", status=404)
                self._ensure_model(bid, mid, "manual")
            if "public_id" in patch:
                c.execute("UPDATE models SET public_id = ? WHERE backend_id = ? AND model_id = ?",
                          (patch["public_id"], bid, mid))
            if "enabled" in patch:
                c.execute("UPDATE models SET enabled = ? WHERE backend_id = ? AND model_id = ?",
                          (int(patch["enabled"]), bid, mid))
            if "context_length" in patch:
                c.execute("UPDATE models SET context_length = ? WHERE backend_id = ? AND "
                          "model_id = ?", (patch["context_length"], bid, mid))
            if "max_output" in patch:
                c.execute("UPDATE models SET max_output_set = ? WHERE backend_id = ? AND "
                          "model_id = ?", (patch["max_output"], bid, mid))
            if "price" in patch:
                self._set_price(bid, mid, patch["price"])
            if "fallback" in patch:
                self._check_fallback(bid, patch["fallback"] or [])
                c.execute("UPDATE models SET fallback = ? WHERE backend_id = ? AND model_id = ?",
                          (json.dumps(patch["fallback"]) if patch["fallback"] is not None
                           else None, bid, mid))
            for g, v in (patch.get("features") or {}).items():
                if v == "inherit":
                    c.execute("DELETE FROM model_features WHERE backend_id = ? AND model_id = ? "
                              "AND feature = ?", (bid, mid, g))
                else:
                    c.execute("INSERT OR REPLACE INTO model_features VALUES (?, ?, ?, ?)",
                              (bid, mid, g, v))
            self._audit(who, "update" if exists else "create", "model", f"{bid}/{mid}",
                        list(patch))
            c.commit()
            self._bump()
        return next(m for m in self.models(bid) if m["model"] == mid)

    def delete_model(self, bid: str, mid: str, who: str = "gui") -> None:
        if self.locks.get(("price", bid, mid)):
            raise ConfigError(f"{self.locks[('price', bid, mid)]} names this model",
                              var=self.locks[("price", bid, mid)])
        with self._lock:
            n = self._conn.execute("DELETE FROM models WHERE backend_id = ? AND model_id = ?",
                                   (bid, mid)).rowcount
            self._conn.execute("DELETE FROM model_features WHERE backend_id = ? AND model_id = ?",
                               (bid, mid))
            if not n:
                raise ConfigError(f"no model '{mid}' on backend '{bid}'", status=404)
            self._audit(who, "delete", "model", f"{bid}/{mid}")
            self._conn.commit()
            self._bump()

    # ----------------------------------------------------------------- routing rules
    def rules(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT id, position, pattern, backend_id, enabled, source, "
                                      "env_var FROM routing_rules ORDER BY position, id").fetchall()
        return [{"id": i, "position": p, "pattern": pat, "backend": b, "enabled": bool(e),
                 "source": s, "variable": v if s == "env" else None}
                for i, p, pat, b, e, s, v in rows]

    def set_rules(self, rules: list[dict], who: str = "gui") -> list[dict]:
        """Replace the ordered list. A rule the environment sets must come back unchanged."""
        clean = [{k: v for k, v in r.items() if k in ("pattern", "backend", "enabled")}
                 for r in rules]
        errs = schema.validate(clean, "rules")
        if errs:
            raise ConfigError(f"{errs[0][0]} {errs[0][1]}", field=errs[0][0])
        cur = {r["id"]: r for r in self.rules()}
        envs = {i: r for i, r in cur.items() if r["source"] == "env"}
        sent_env = set()
        for i, r in enumerate(rules):
            rid = r.get("id")
            if rid in envs:
                e = envs[rid]
                if (r["pattern"], r["backend"]) != (e["pattern"], e["backend"]) or \
                        r.get("enabled", True) is False:
                    raise ConfigError(f"rules[{i}] is set by {e['variable']}",
                                      field=f"rules[{i}]", var=e["variable"])
                sent_env.add(rid)
        for rid, e in envs.items():
            if rid not in sent_env:
                raise ConfigError(f"the rule set by {e['variable']} cannot be removed here",
                                  var=e["variable"])
        known = {b["id"] for b in self.backends()}
        for i, r in enumerate(rules):
            if r["backend"] not in known:
                raise ConfigError(f"rules[{i}].backend: no backend '{r['backend']}'",
                                  field=f"rules[{i}].backend")
        with self._lock:
            c = self._conn
            c.execute("DELETE FROM routing_rules WHERE source != 'env'")
            for pos, r in enumerate(rules):
                if r.get("id") in envs:
                    c.execute("UPDATE routing_rules SET position = ? WHERE id = ?",
                              (pos, r["id"]))
                else:
                    c.execute("INSERT INTO routing_rules(position, pattern, backend_id, enabled, "
                              "source) VALUES (?, ?, ?, ?, 'gui')",
                              (pos, r["pattern"], r["backend"], int(r.get("enabled", True))))
            self._audit(who, "replace", "routing_rules", None, ["rules"])
            c.commit()
            self._bump()
        return self.rules()

    # ----------------------------------------------------------------- users and their keys
    def users(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT id, label, key_hint, role, enabled, created_at, "
                                      "last_seen_at FROM users ORDER BY id").fetchall()
        out = [{"id": i, "label": lab, "key_hint": h, "role": role, "enabled": bool(en),
                "created_at": cr, "last_seen_at": self._seen.get(i, seen), "source": "database"}
               for i, lab, h, role, en, cr, seen in rows]
        out += [{"id": None, "label": lab, "key_hint": None, "role": "user", "enabled": True,
                 "created_at": None, "last_seen_at": None, "source": "env"}
                for lab in sorted(set(self.env_users.values()))]
        return out

    def users_active(self) -> bool:
        """True once any key exists: then the OpenAI routes need one."""
        if self.env_users:
            return True
        with self._lock:
            return bool(self._conn.execute("SELECT 1 FROM users WHERE enabled = 1 "
                                           "LIMIT 1").fetchone())

    def create_user(self, doc: dict, who: str = "gui") -> tuple[dict, str]:
        """A new user and its key, which is returned this once and never again."""
        errs = schema.validate(doc, "user_new")
        if errs:
            raise ConfigError(f"{errs[0][0]} {errs[0][1]}", field=errs[0][0])
        key = KEY_PREFIX + _rand.token_urlsafe(30)
        with self._lock:
            if self._conn.execute("SELECT 1 FROM users WHERE label = ?",
                                  (doc["label"],)).fetchone() or \
                    doc["label"] in self.env_users.values():
                raise ConfigError(f"a user '{doc['label']}' exists already", field="label",
                                  status=409)
            cur = self._conn.execute(
                "INSERT INTO users(label, key_hash, key_hint, role, enabled, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (doc["label"], key_hash(key), key[-4:], doc.get("role", "user"), _now_ms()))
            uid = cur.lastrowid
            self._audit(who, "create", "user", str(uid), ["label"])
            self._conn.commit()
            self._bump()
        return next(u for u in self.users() if u["id"] == uid), key

    def update_user(self, uid: int, patch: dict, who: str = "gui") -> dict:
        errs = schema.validate(patch, "user_patch")
        if errs:
            raise ConfigError(f"{errs[0][0]} {errs[0][1]}", field=errs[0][0])
        with self._lock:
            n = self._conn.execute("UPDATE users SET enabled = ? WHERE id = ?",
                                   (int(patch.get("enabled", True)), uid)).rowcount
            if not n:
                raise ConfigError(f"no user {uid}", status=404)
            self._audit(who, "update", "user", str(uid), list(patch))
            self._conn.commit()
            self._bump()
        return next(u for u in self.users() if u["id"] == uid)

    def delete_user(self, uid: int, who: str = "gui") -> None:
        with self._lock:
            if not self._conn.execute("DELETE FROM users WHERE id = ?", (uid,)).rowcount:
                raise ConfigError(f"no user {uid}", status=404)
            self._audit(who, "delete", "user", str(uid))
            self._conn.commit()
            self._bump()

    def user_for_key(self, key: str | None) -> str | None:
        """The label of the user whose key this is, or None. Compared by hash, in constant time
        per candidate."""
        if not key:
            return None
        h = key_hash(key)
        for eh, label in self.env_users.items():
            if hmac.compare_digest(eh, h):
                return label
        with self._lock:
            r = self._conn.execute("SELECT id, label FROM users WHERE key_hash = ? AND "
                                   "enabled = 1", (h,)).fetchone()
        if r is None:
            return None
        self._seen[r[0]] = _now_ms()
        self.flush_seen()
        return r[1]

    def flush_seen(self, force: bool = False) -> None:
        now = time.monotonic()
        if not self._seen or (not force and now - self._seen_flushed < 60.0):
            return
        with self._lock:
            if self._conn is None:
                return
            seen, self._seen = self._seen, {}
            self._conn.executemany("UPDATE users SET last_seen_at = ? WHERE id = ?",
                                   [(t, i) for i, t in seen.items()])
            self._conn.commit()
            self._seen_flushed = now

    # ----------------------------------------------------------------- settings
    def setting(self, key: str, default=None):
        with self._lock:
            r = self._conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_setting(self, key: str, value, who: str = "gui") -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO settings(key, value, updated_at) "
                               "VALUES (?, ?, ?)", (key, json.dumps(value), _now_ms()))
            self._audit(who, "set", "setting", key)
            self._conn.commit()
            self._bump()

    def audit(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT ts_ms, who, action, object, object_id, fields FROM "
                                      "audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts_ms": t, "who": w, "action": a, "object": o, "object_id": i,
                 "fields": json.loads(f) if f else None} for t, w, a, o, i, f in rows]

    def empty(self) -> bool:
        """No backend at all: the dashboard opens on the Setup screen."""
        with self._lock:
            return not self._conn.execute("SELECT 1 FROM backends LIMIT 1").fetchone()


def glob(pattern: str, model: str) -> bool:
    """A routing rule's match: a shell glob over the whole model id, case sensitive."""
    return fnmatch.fnmatchcase(model, pattern)
