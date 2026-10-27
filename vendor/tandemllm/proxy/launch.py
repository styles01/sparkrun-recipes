"""`tandem-dash launch vllm|sglang`: start an engine with TandemLLM's plugin, registered as a backend (L3).

    tandem-dash launch vllm --model Qwen/Qwen3.5-4B --port 8100 [--id vllm1] [-- vllm args]
    tandem-dash launch sglang --model Qwen/Qwen3.5-4B --port 30000 [--id sglang1] [-- sglang args]

One command from nothing to a backend with L2 data:

1. The backend is created in the config database (or updated when the id exists): kind `vllm` or
   `sglang`, URL `http://127.0.0.1:<port>/v1`, host `same`, so the proxy's health poller, its
   /metrics scrape (L1) and the host collector find it at once.
2. The engine gets the plugin's two variables: `TANDEM_COLLECT_URL` (this proxy's
   `/v1/collect/<id>`; `--collect-url` names another proxy's base URL) and `TANDEM_COLLECT_TOKEN`
   (the proxy's collect token, made and stored at the first need). When `VLLM_PLUGINS` or
   `SGLANG_PLUGINS` is already set, the plugin's name is added to it; unset, the engine loads
   every installed plugin of the group, ours included.
3. The flags each engine needs for the numbers the dashboard shows are added when the arguments
   lack them: vLLM `--enable-prompt-tokens-details` (cached tokens per request; prefix caching is
   on by default in vLLM's V1 engine), SGLang `--enable-metrics` (the page and the plugin's hooks)
   and `--enable-cache-report`.
4. The launcher checks that the plugin is installed in the engine's Python (its entry point is
   listed); when it is not, it says how to install it and starts the engine anyway, at L1.
5. Then it replaces itself with the engine (`exec`): the pid it records in the settings
   (`launch:<id>`) is the engine's own, and a signal to it is a signal to the engine.

`--dry-run` prints the command and the environment (the token shown as its first 4 characters)
and starts nothing. `--engine-cmd` replaces the engine's executable (a test stand-in, or a
`python -m ...` of a particular environment).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

from proxy import collect_api
from proxy import config as config_mod

ENGINES = {
    "vllm": {"cmd": ["vllm", "serve"], "model_flag": None, "port_flag": "--port",
             "plugins_var": "VLLM_PLUGINS", "plugin_name": "tandem_stat_logger",
             "group": "vllm.stat_logger_plugins", "default_port": 8000,
             "need": [("--enable-prompt-tokens-details", None)],
             "package": "plugins/vllm (pip install tandemllm-vllm-plugin)"},
    "sglang": {"cmd": ["sglang", "serve"], "model_flag": "--model-path", "port_flag": "--port",
               "plugins_var": "SGLANG_PLUGINS", "plugin_name": "tandem",
               "group": "sglang.srt.plugins", "default_port": 30000,
               "need": [("--enable-metrics", None), ("--enable-cache-report", None)],
               "package": "plugins/sglang (pip install tandemllm-sglang-plugin)"},
}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tandem-dash launch", description=__doc__.split("\n")[0])
    ap.add_argument("engine", choices=sorted(ENGINES))
    ap.add_argument("--model", required=True, help="the model id or path the engine serves")
    ap.add_argument("--port", type=int, default=None, help="the engine's port")
    ap.add_argument("--host", default="127.0.0.1",
                    help="the engine's bind address (default 127.0.0.1)")
    ap.add_argument("--id", default=None, help="the backend id (default: the engine's name and "
                                                "port, e.g. vllm8100)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--config-db", default=os.environ.get("TANDEM_CONFIG_DB",
                                                          config_mod.DEFAULT_PATH))
    ap.add_argument("--collect-url", default=os.environ.get("TANDEM_COLLECT_BASE"),
                    help="the proxy's base URL the plugin posts to (default "
                         "http://127.0.0.1:$TANDEM_PORT, 8080)")
    ap.add_argument("--engine-cmd", default=None,
                    help="the engine's command instead of `vllm serve` / `sglang serve`")
    ap.add_argument("--python", default=None,
                    help="the Python of the engine's environment, for the plugin check "
                         "(default: the one behind the engine's command)")
    ap.add_argument("--no-register", action="store_true",
                    help="do not create or update the backend")
    ap.add_argument("--dry-run", action="store_true", help="print, start nothing")
    ap.epilog = "Everything after -- goes to the engine as it is."
    return ap


def _has_flag(args: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in args)


def command(engine: str, a, rest: list[str]) -> list[str]:
    spec = ENGINES[engine]
    cmd = shlex.split(a.engine_cmd) if a.engine_cmd else list(spec["cmd"])
    if spec["model_flag"]:
        cmd += [spec["model_flag"], a.model]
    else:
        cmd += [a.model]
    port = a.port or spec["default_port"]
    if not _has_flag(rest, spec["port_flag"]):
        cmd += [spec["port_flag"], str(port)]
    if not _has_flag(rest, "--host"):
        cmd += ["--host", a.host]
    for flag, value in spec["need"]:
        if not _has_flag(rest, flag):
            cmd += [flag] + ([value] if value is not None else [])
    return cmd + rest


def environment(engine: str, base_env: dict, url: str, token: str) -> dict:
    spec = ENGINES[engine]
    env = dict(base_env)
    env["TANDEM_COLLECT_URL"] = url
    env["TANDEM_COLLECT_TOKEN"] = token
    cur = env.get(spec["plugins_var"])
    if cur is not None:
        names = [x.strip() for x in cur.split(",") if x.strip()]
        if spec["plugin_name"] not in names:
            names.append(spec["plugin_name"])
        env[spec["plugins_var"]] = ",".join(names)
    return env


def engine_python(cmd: list[str], given: str | None) -> str | None:
    """The interpreter of the engine's environment: --python, else the shebang of the engine's
    executable, else a `python ...` command's own first word."""
    if given:
        return given
    first = cmd[0]
    if os.path.basename(first).startswith("python"):
        return shutil.which(first) or first
    path = shutil.which(first)
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            line = f.readline(512).decode(errors="replace").strip()
    except OSError:
        return None
    if line.startswith("#!"):
        parts = line[2:].split()
        if parts and os.path.basename(parts[0]) == "env" and len(parts) > 1:
            return shutil.which(parts[1])
        return parts[0] if parts else None
    return None


def plugin_installed(python: str | None, group: str, name: str) -> bool | None:
    """True or False from the engine's Python; None when it cannot be asked."""
    if not python:
        return None
    code = ("import importlib.metadata as m, sys; "
            f"sys.exit(0 if any(e.name == {name!r} for e in m.entry_points(group={group!r})) "
            "else 3)")
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return True if r.returncode == 0 else (False if r.returncode == 3 else None)


def register(cfg, engine: str, bid: str, port: int, label: str | None) -> dict:
    doc = {"kind": engine, "url": f"http://127.0.0.1:{port}/v1", "label": label or bid,
           "host": {"mode": "same"}}
    if cfg.backend(bid) is None:
        return cfg.create_backend(dict(doc, id=bid), who="launch")
    return cfg.update_backend(bid, doc, who="launch")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    a = parser().parse_args(argv)
    spec = ENGINES[a.engine]
    port = a.port or spec["default_port"]
    bid = a.id or f"{a.engine}{port}"
    env = os.environ
    cfg = config_mod.ConfigDB(a.config_db, env=env, use_keychain=False).open()
    try:
        cfg.apply_env()
        token = collect_api.token_of(cfg, env)
        if not a.no_register and not a.dry_run:
            register(cfg, a.engine, bid, port, a.label)
        if not a.dry_run:
            cfg.set_setting(f"launch:{bid}", {"pid": os.getpid(), "engine": a.engine,
                                              "port": port, "model": a.model,
                                              "started": int(time.time())}, who="launch")
    except config_mod.ConfigError as exc:
        print(f"[launch] configuration error: {exc.message}", file=sys.stderr, flush=True)
        return 2
    finally:
        cfg.close()
    base = (a.collect_url or f"http://127.0.0.1:{env.get('TANDEM_PORT', '8080')}").rstrip("/")
    url = f"{base}/v1/collect/{bid}"
    cmd = command(a.engine, a, rest)
    run_env = environment(a.engine, env, url, token)
    py = engine_python(cmd, a.python)
    ok = plugin_installed(py, spec["group"], spec["plugin_name"])
    if ok is False:
        print(f"[launch] the plugin is not installed in {py}: install {spec['package']} into the "
              f"engine's environment. Starting {a.engine} without it (L1: its /metrics only).",
              file=sys.stderr, flush=True)
    elif ok is None:
        print(f"[launch] no Python found behind {cmd[0]} to ask whether the plugin is "
              "installed (--python names it); starting anyway", file=sys.stderr, flush=True)
    shown = {k: (v[:4] + "…" if k == "TANDEM_COLLECT_TOKEN" else v) for k, v in run_env.items()
             if k.startswith("TANDEM_COLLECT") or k == spec["plugins_var"]}
    print(f"[launch] backend {bid} ({a.engine}) at http://127.0.0.1:{port}/v1"
          f"{' (not registered)' if a.no_register else ''}; plugin "
          f"{'installed' if ok else 'missing' if ok is False else 'unchecked'}", flush=True)
    print(f"[launch] env {json.dumps(shown, ensure_ascii=False)}", flush=True)
    print(f"[launch] exec {shlex.join(cmd)}", flush=True)
    if a.dry_run:
        return 0
    try:
        os.execvpe(cmd[0], cmd, run_env)
    except OSError as exc:
        print(f"[launch] could not start {cmd[0]}: {exc}", file=sys.stderr, flush=True)
        return 127
    return 0                                                       # pragma: no cover
