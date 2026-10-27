"""The `tandem-dash` commands other than `serve` (proxy/__main__.py): check, secrets, backends.

Each returns the process's exit code. `check` and `backends list` read a copy of the config
database (SQLite's backup API), so they never write to the real one; `secrets init` is the one
command that writes, and only a missing key file.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import shutil
import sqlite3
import sys
import tempfile

from proxy import config as config_mod
from proxy import secrets as vault_mod


def _say(*parts) -> None:
    print(*parts, flush=True)


@contextlib.contextmanager
def _copy_of(path: str, env):
    """A ConfigDB opened on a copy of `path` (or on a fresh file when there is none), with the
    environment applied. The copy and its directory are removed afterwards."""
    real = os.path.realpath(os.path.expanduser(path))
    d = tempfile.mkdtemp(prefix="tandem-dash-")
    try:
        tmp = os.path.join(d, "config.sqlite3")
        if os.path.exists(real):
            src = sqlite3.connect(f"file:{real}?mode=ro", uri=True)
            dst = sqlite3.connect(tmp)
            src.backup(dst)
            src.close()
            dst.close()
        key = os.path.join(os.path.dirname(real), vault_mod.KEY_FILE)
        if os.path.exists(key):
            shutil.copy2(key, os.path.join(d, vault_mod.KEY_FILE))
        cfg = config_mod.ConfigDB(tmp, env=env, use_keychain=False).open()
        try:
            yield cfg, os.path.exists(real)
        finally:
            cfg.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _tilde(p: str) -> str:
    home = os.path.expanduser("~")
    return "~" + p[len(home):] if home != "/" and p.startswith(home) else p


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:                                                # noqa: BLE001
        return False


def _backend_lines(cfg) -> list[str]:
    out = []
    for b in cfg.backends():
        hint = cfg.key_hint(b["id"])
        key = (f"key …{hint['last4'] or '????'} ({hint['source']})" if hint["key_set"]
               else "no key")
        locks = cfg.backend_locks(b["id"])
        host = b.get("host") or {}
        out.append(f"  {b['id']:<16} {b['kind']:<10} {b['url']:<40} "
                   f"{'on ' if b['enabled'] else 'off'} {key}; host {host.get('mode', 'auto')}"
                   f"{' ' + host['url'] if host.get('url') else ''}"
                   f"{'; set by the environment: ' + ', '.join(sorted(locks)) if locks else ''}")
    return out


def check(args) -> int:
    env = os.environ
    from proxy import _version
    _say(f"tandemllm {_version.release() or 'unknown'}, python {sys.version.split()[0]}")
    try:
        config_mod.parse_env(env)
    except config_mod.ConfigError as exc:
        _say(f"FAIL the environment: {exc.message}")
        return 1
    try:
        with _copy_of(args.config_db, env) as (cfg, existed):
            _say(f"config  {_tilde(os.path.realpath(os.path.expanduser(args.config_db)))}"
                 f"{'' if existed else ' (does not exist yet: serve creates it)'}")
            ov = cfg.apply_env()
            real = config_mod.ConfigDB(args.config_db, env=env)
            src = real.key_source()
            _say(f"keys    master key from {src or 'a new key file at the first stored key'}")
            for b in cfg.backends():
                hint = cfg.key_hint(b["id"])
                if hint["key_set"] and hint["source"] == "database" and src is None:
                    _say(f"WARN    {b['id']}: a key is stored but no master key answers: "
                         f"enter it again or restore secrets.key")
            bl = _backend_lines(cfg)
            _say(f"backends {len(bl)}" + ("" if bl else ": none yet, the page opens on Setup"))
            for line in bl:
                _say(line)
            rules = cfg.rules()
            for r in rules:
                _say(f"  rule {r['pattern']} -> {r['backend']}"
                     f"{' (' + r['variable'] + ')' if r.get('variable') else ''}")
            users = cfg.users()
            _say(f"users   {len(users)} (keys from the environment: {len(ov['users'])})")
    except config_mod.ConfigError as exc:
        _say(f"FAIL the configuration: {exc.message}")
        return 1
    from server import static
    dist = args.dashboard_dir or static.default_dir()
    ok_dist = os.path.exists(os.path.join(dist, "index.html"))
    _say(f"page    {_tilde(dist)}{'' if ok_dist else ' MISSING: the page shows a placeholder'}")
    led = args.usage_ledger
    _say(f"ledger  {_tilde(os.path.expanduser(led)) if led != 'off' else 'off'}")
    _say(f"extras  cryptography {'yes' if _has('cryptography') else 'NO (keys cannot be stored)'}"
         f", psutil {'yes' if _has('psutil') else 'no'}, nvml {'yes' if _has('pynvml') else 'no'}"
         f" (the host view falls back to /proc and nvidia-smi)")
    try:
        loop = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loop = args.host == "localhost"
    admin = bool(env.get("TANDEM_ADMIN_TOKEN") or env.get("QSE_ADMIN_TOKEN"))
    _say(f"listen  {args.host}:{args.port}, dashboard login {args.dashboard_login}, admin "
         f"password {'set' if admin else 'not set (every route is open, the dashboard asks for nothing)'}")
    if not loop and not admin:
        _say("WARN    bound beyond loopback with no admin password: anyone on the network reads "
             "and changes the configuration. Keep it on a trusted network, never on the internet")
    if not _has("cryptography"):
        return 1
    _say("OK")
    return 0


def secrets(args) -> int:
    real = os.path.realpath(os.path.expanduser(args.config_db))
    d = os.path.dirname(real)
    cfg = config_mod.ConfigDB(args.config_db, env=os.environ)
    src = cfg.key_source()
    if args.action == "status":
        _say(f"master key: {src or 'none yet'} ({_tilde(d)})")
        return 0
    path = os.path.join(d, vault_mod.KEY_FILE)
    if src is not None:
        _say(f"a master key answers already: {src}"
             f"{' at ' + _tilde(path) if src == 'file' else ''}; nothing written")
        return 0
    try:
        v = vault_mod.Vault.load(d, env=os.environ, use_keychain=not args.no_keychain)
    except vault_mod.SecretError as exc:
        _say(f"FAIL {exc}")
        return 1
    _say(f"master key: {v.source}" + (f", written to {_tilde(path)} (0600)" if v.source == "file"
                                      else ""))
    return 0


def backends(args) -> int:
    try:
        with _copy_of(args.config_db, os.environ) as (cfg, existed):
            cfg.apply_env()
            lines = _backend_lines(cfg)
    except config_mod.ConfigError as exc:
        _say(f"FAIL {exc.message}")
        return 1
    if not lines:
        _say("no backends")
    for line in lines:
        _say(line.strip())
    return 0
