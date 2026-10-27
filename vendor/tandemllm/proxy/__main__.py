"""`tandem-dash` / `python -m proxy`: the standalone proxy and dashboard (docs/proxy.md).

    tandem-dash [serve]                              ~/.tandemllm/config.sqlite3, 127.0.0.1:8080
    tandem-dash serve --env-file proxy.env --host 0.0.0.0 --port 8080
    tandem-dash serve --config-db /data/config.sqlite3 --usage-ledger /data/ledger.sqlite3
    tandem-dash check [--env-file F]                 the schema over the env and the database,
                                                     and what would run; writes nothing
    tandem-dash secrets init | status                the master key for keys at rest
    tandem-dash backends list                        the backends, their keys' hints and locks
    tandem-dash hostcollect [--host 0.0.0.0 ...]     the remote host collector (P1.5)
    tandem-dash launch vllm|sglang --model M --port P [-- engine args]
                                                     the engine with TandemLLM's plugin, registered
                                                     as a backend (P6, proxy/launch.py)
    tandem-dash version

`python -m proxy` takes the same commands; with none it serves, so `python -m proxy --port 8080`
works as before. Needs Python 3.11 or newer, the standard library and `cryptography` (for keys
at rest); `pip install 'tandemllm[host]'` adds psutil and NVML for the System view's hosts. No
torch, no GPU. A fresh `--config-db` path is created at schema version 1.
"""

from __future__ import annotations

import argparse
import os
import sys

if sys.version_info < (3, 11):                                    # pragma: no cover
    raise SystemExit("the proxy needs Python 3.11 or newer")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy import config as config_mod  # noqa: E402
from server import ledger as ledger_mod  # noqa: E402


def load_env_file(path: str, env=None) -> list[str]:
    """`KEY=VALUE` lines into the environment; a variable already set keeps its value (the
    process environment wins over the file). Returns the names it set."""
    env = os.environ if env is None else env
    got = []
    with open(os.path.expanduser(path)) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            k, sep, v = line.partition("=")
            k = k.strip()
            if not sep or not k.replace("_", "").isalnum():
                raise SystemExit(f"[proxy] {path}:{n}: not a KEY=VALUE line")
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            if k not in env:
                env[k] = v
                got.append(k)
    return got


def _at_least_0(var: str, unit: str):
    """argparse's `type` for a number >= 0 that may come from the variable `var`."""
    def conv(raw: str) -> float:
        try:
            v = float(raw)
        except ValueError:
            v = -1.0
        if not v >= 0 or v == float("inf"):
            raise argparse.ArgumentTypeError(f"{raw!r} is not a number of {unit} >= 0 (the "
                                             f"option or {var})")
        return v
    return conv


def parser() -> argparse.ArgumentParser:
    env = os.environ
    ap = argparse.ArgumentParser(prog="python -m proxy", description=__doc__.split("\n")[0])
    ap.add_argument("--config-db", default=env.get("TANDEM_CONFIG_DB", config_mod.DEFAULT_PATH),
                    help="the config database (created when missing); TANDEM_CONFIG_DB")
    ap.add_argument("--env-file", default=None,
                    help="KEY=VALUE lines read before the start; the process environment wins")
    ap.add_argument("--host", default=env.get("TANDEM_HOST", "127.0.0.1"),
                    help="the address to bind (default 127.0.0.1; the page has no sign-in "
                         "screen: keep it on loopback, or behind a reverse proxy with its own "
                         "sign-in and TLS)")
    ap.add_argument("--port", type=int, default=int(env.get("TANDEM_PORT", "8080")))
    ap.add_argument("--usage-ledger", default=env.get("TANDEM_USAGE_LEDGER",
                                                      "~/.tandemllm/ledger.sqlite3"),
                    help="the usage ledger (one row per request, no text); 'off' for none")
    ap.add_argument("--usage-retention-days", type=int, default=ledger_mod.MIN_RETENTION_DAYS)
    ap.add_argument("--dashboard-login",
                    default=env.get("TANDEM_DASHBOARD_LOGIN", env.get("QSE_DASHBOARD_LOGIN",
                                                                      "off")),
                    choices=("on", "off"),
                    help="on, with TANDEM_ADMIN_TOKEN (the admin password) set: the dashboard "
                         "API needs it or a session too; with no password set every route is open")
    ap.add_argument("--dashboard-dir", default=None, help="default: dashboard/dist")
    ap.add_argument("--trust-loopback", default="on", choices=("on", "off"),
                    help="a request from this machine without proxy headers counts as the admin")
    ap.add_argument("--no-keychain", action="store_true",
                    help="do not ask the OS keychain for the master key")
    ap.add_argument("--conductor-config",
                    default=env.get("TANDEM_CONDUCTOR", "~/.tandemllm/conductor"),
                    help="the conductor's documents (profiles, router, packages) for the harness "
                         "in front of the backends (P4); 'off' relays every backend as P1 does; "
                         "TANDEM_CONDUCTOR")
    # a string default goes through `type` too: a bad variable is a usage error, not a traceback
    ap.add_argument("--idle-timeout", type=_at_least_0("TANDEM_IDLE_TIMEOUT_S", "seconds"),
                    default=env.get("TANDEM_IDLE_TIMEOUT_S", "60"),
                    help="seconds a client may take to send its request, or stay idle between "
                         "requests, before the connection is closed (an answer has no timeout); "
                         "0 for none; TANDEM_IDLE_TIMEOUT_S")
    ap.add_argument("--conductor-defaults", default="shipped", choices=("shipped", "builtin"),
                    help="what a new conductor directory starts with")
    ap.add_argument("--verbose", action="store_true")
    return ap


COMMANDS = ("serve", "check", "secrets", "backends", "hostcollect", "launch", "version")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = "serve"
    if argv and argv[0] in COMMANDS:
        cmd, argv = argv[0], argv[1:]
    if cmd == "hostcollect":
        from proxy import hostcollect
        return hostcollect.main(argv)
    if cmd == "launch":
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--env-file")
        known, rest = pre.parse_known_args(argv)
        if known.env_file:
            load_env_file(known.env_file)
        from proxy import launch
        raise SystemExit(launch.main(rest))
    if cmd == "version":
        from proxy import _version
        print(f"tandemllm {_version.release() or 'unknown'} (python {sys.version.split()[0]})")
        return
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file")
    known, _ = pre.parse_known_args(argv)
    if known.env_file:
        load_env_file(known.env_file)
    if cmd == "serve":
        args = parser().parse_args(argv)
        from proxy import app
        try:
            app.serve(args)
        except config_mod.ConfigError as exc:
            raise SystemExit(f"[proxy] configuration error: {exc.message}")
        return
    ap = parser()
    ap.prog = f"tandem-dash {cmd}"
    if cmd == "secrets":
        ap.add_argument("action", choices=("init", "status"))
    elif cmd == "backends":
        ap.add_argument("action", choices=("list",))
    args = ap.parse_args(argv)
    from proxy import cli
    raise SystemExit(getattr(cli, cmd)(args))


if __name__ == "__main__":
    main()
