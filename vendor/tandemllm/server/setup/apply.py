"""The apply job (SRV-327): a checked selection becomes the engine's configuration, through a fixed
list of steps in this file and the repository's own scripts. Nothing a client sends is run.

    POST /v1/setup/plan {selection}             the review step's facts; nothing changes
    POST /v1/setup/apply {selection, dry_run?}  dry_run: the plan; else 202 and a job
    GET  /v1/setup/jobs/{id}                    SSE: the log on disk, the job now, then its events
                                                (`id:` = sequence, Last-Event-ID resumes);
                                                ?format=json: the job

THE STEPS (STEPS below; the argv of every command is built here from the registry's ids and the
selection's checked numbers, `shell=False`, a minimal environment, a timeout each):

    check             tandem doctor first (OPS-171: modes, run/ops copies, stale pid files, the
                      marker repaired; its report on the job as `doctor`), then the selection again
                      (the runner checks it itself), the estimate, the disk, the assets, the HF
                      token, the overlay hook in serve.env, the box lock
    unload_assistant  Wren unloaded (the server does it before it starts the runner)
    fetch             bash ops/use-profile.sh <profile> --fetch-only    (skipped when the weights and
                      the assets are on disk; OPS-95: it fetches the missing assets too; VIS-153:
                      and the models of the tools the selection turns on, QSE_FETCH_ASSETS)
    preflight         OPS-169: before anything stops, what the restart needs (`preflight` below):
                      the scripts parse, the new configuration sources under set -u, the weights
                      are complete (the hub manifest's sizes), memory, the port and the locks, the
                      interpreters; a failure ends the job here and the running setup keeps running
    (waiting)         write_config waits until no request runs or waits on the engine (its /health,
                      IDLE_WAIT_S at most; then nothing has changed and the job fails)
    write_config      run/setup.env from the registry's keys and the selection, the old one kept
                      as run/setup.env.bak-<utc>, written whole and renamed into place; with a
                      port change the engine on the old port is stopped first (bash ops/stop.sh);
                      with `admin_password: generate`, bash ops/make-secrets.sh --admin
    switch            bash ops/use-profile.sh <profile>: the box lock, stop, start, its smoke,
                      its own rollback of the profile (in setup mode the marker run/setup-pending
                      goes first, so start.sh starts the configured engine)
    smoke             /health names the profile; one short chat answers; then (VIS-153) the
                      tools with a model of their own: one that failed to load is a warning on
                      the job (job.warnings), one waiting for memory a note (job.notes)
    rollback          on a failed switch or smoke: the old run/setup.env back, then
                      bash ops/use-profile.sh <previous profile> (setup mode: the marker back and
                      bash ops/start.sh --setup)

ops/serve.env is never edited: it reads run/setup.env after the weight profile. The runner is a
process of its own (`python -m server.setup.apply run <job folder>`, its own session), because
the switch replaces the server that started it; it writes run/setup-jobs/<id>/state.json and events.jsonl,
which the restarted server (and ops/engine_status.py's responder meanwhile) reads. Log lines pass
the conductor's secret filter, and the HF and admin tokens are cut out of them by value.
"""

from __future__ import annotations

import glob
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from server.setup import device as device_mod
from server.setup import estimate as estimate_mod
from server.setup import registry

# the checkout this server runs from; serve.env's REPO must be the same folder (the plan checks),
# since serve.env reads ${REPO}/run/setup.env and start.sh ${REPO}/run/setup-pending
ROOT = registry.ROOT


def run_dir() -> str:
    return os.path.join(ROOT, "run")


def overlay_path() -> str:
    return os.path.join(run_dir(), "setup.env")


def pending_path() -> str:
    return os.path.join(run_dir(), "setup-pending")


def jobs_dir() -> str:
    return os.path.join(run_dir(), "setup-jobs")


DISK_MARGIN_GB = 60          # ops/use-profile.sh's: the disk tier and the spill grow into it
SCRIPTS = ("use-profile.sh", "start.sh", "stop.sh", "make-secrets.sh")
TIMEOUTS = {"fetch": 7200, "switch": 1800, "stop": 300, "secrets": 60, "start": 900}
# before anything stops, the engine's requests finish: at most this long, asked this often
IDLE_WAIT_S = 600
IDLE_POLL_S = 2.0
# VIS-153: the tools whose model is a role of the engine (`/health` roles; Laya: the decision
# models of /v1/models), and how long the smoke step waits for them to load after the start
ROLE_OF = {"narrator": "narrator", "shadow": "shadow", "laya": None}
ROLE_POLL_S = 3.0
ROLE_SETTLE_POLLS = 30
JOB_ID = re.compile(r"^[a-f0-9]{16}$")
PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
STEPS = (("check", "Check the selection, the disk and the box"),
         ("unload_assistant", "Unload Wren"),
         ("fetch", "Download the weights"),
         ("preflight", "Check the new setup before the restart"),
         ("write_config", "Write the configuration (run/setup.env)"),
         ("switch", "Restart the engine with the new setup"),
         ("smoke", "Check that the new engine answers"),
         ("rollback", "Roll back"))
LABELS = dict(STEPS)


def profile_ok(pid: str, ops: str | None = None) -> bool:
    """`pid` is a profile ops/use-profile.sh takes: ops/profiles/<pid>.env, or community/<id> of an
    entry in the index in use (use-profile.sh writes its profile from the entry)."""
    if isinstance(pid, str) and pid.startswith(registry.COMMUNITY_PREFIX):
        return any(registry.community(m, pid) for m in registry.load_all()[0].values())
    return bool(PROFILE_ID.match(pid or "")) and os.path.isfile(
        os.path.join(device_mod.repo_ops(ops), "profiles", pid + ".env"))


class PlanError(ValueError):
    """apply refused before anything changed: `status` 409 or 412, `code`, the plan."""

    def __init__(self, status: int, code: str, message: str, plan: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.plan = status, code, plan

    def body(self) -> dict:
        return {"error": {"type": self.code, "code": self.code, "message": str(self)},
                **({"plan": self.plan} if self.plan is not None else {})}


# ---------------------------------------------------------------------- the overlay
def _value_ok(v: str) -> bool:
    """A value that goes into run/setup.env: one line, no quote, backslash, backtick or command
    substitution (`${HOME}` from the registry is the one expansion it keeps)."""
    return isinstance(v, str) and not re.search(r"[\n\r\"'`\\]|\$\(", v) and \
        all(x == "${HOME}" for x in re.findall(r"\$\{?[A-Za-z_]*\}?", v))


def overlay_values(m: dict, sel: dict) -> dict:
    """serve.env key -> value for a checked selection, in the order they are written; a tool's
    `null` state writes nothing (the profile's value stands)."""
    out = {}
    p = m["parallelism"]
    k = p["knobs"]
    out[k["slots"]] = str(sel["slots"])
    out[k["window"]] = str(sel["window"])
    out[k["side_window"]] = str(sel["side_window"])
    out[k["kv_pool"]] = "on" if sel["kv_pool"] else "off"
    out[k["floor_gb"]] = f"{sel['floor_gb']:g}"
    out.update(p.get("fixed") or {})
    for c in m["components"]:
        on = sel["components"][c["id"]]
        for key, v in c["knobs"].items():
            val = v["on" if on else "off"]
            if val is not None:
                out[key] = val
    for group in ("methods", "advanced"):
        for f in m[group]:
            if f["knob"]:
                v = sel[group][f["id"]]
                out[f["knob"]] = ("true" if v else "false") if isinstance(v, bool) else \
                    (f"{v:g}" if isinstance(v, float) else str(v))
    bad = [key for key, v in out.items() if not re.match(registry.ENV_KEY, key) or not _value_ok(v)]
    if bad:
        raise ValueError(f"values that cannot go into run/setup.env: {', '.join(bad)}")
    return out


def render_overlay(m: dict, sel: dict, when: str | None = None) -> str:
    when = when or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = [f"# Written by the setup guide (server/setup/apply.py) on {when}: {m['id']}, profile "
             f"{sel['profile']}.",
             "# Generated: the next apply replaces it and reset level 2 removes it. ops/serve.env",
             "# reads it after the weight profile, so it never sets the weights."]
    # always quoted: single quotes, or double quotes around the one expansion it keeps (${HOME});
    # _value_ok refuses quotes, backslashes, backticks and every other $ in a value
    for key, v in _without_unset_defaults(overlay_values(m, sel), sel["profile"]).items():
        lines.append(f'{key}="{v}"' if "${HOME}" in v else f"{key}='{v}'")
    return "\n".join(lines) + "\n"


def _without_unset_defaults(values: dict, profile: str | None = None) -> dict:
    """`values` less every key ops/serve.env alone leaves unset whose value is what the engine
    runs with then anyway (`_unset_default`): the engine's environment stays as it was. The disk
    tier's and the spill's fingerprint counts every QWEN38_* variable that is set, so writing the
    lookup store's default folder into the overlay of a box that never set it swept that profile's
    saved heads and parked chats at the switch."""
    base = resolve_env(list(values), profile=profile, overlay=os.devnull)
    return {k: v for k, v in values.items()
            if k in base or _unset_default(k) is None or _expand(v) != _unset_default(k)}


def hook_present(ops: str | None = None) -> bool:
    """ops/serve.env reads the overlay (a box whose serve.env predates the setup guide does not)."""
    try:
        with open(os.path.join(ops or device_mod.ops_dir(), "serve.env")) as fh:
            return "run/setup.env" in fh.read()
    except OSError:
        return False


def resolve_env(keys: list[str], *, profile: str | None = None, overlay: str | None = None,
                ops: str | None = None, timeout: float = 10.0) -> dict:
    """key -> value as ops/serve.env resolves them (QSE_PROFILE_OVERRIDE `profile`, QSE_SETUP_ENV
    `overlay`), in a bash with nothing of this process's environment but PATH and HOME."""
    env_file = os.path.join(ops or device_mod.ops_dir(), "serve.env")
    if not os.path.isfile(env_file):
        return {}
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.path.expanduser("~"),
           "ENVF": env_file}
    if profile:
        env["QSE_PROFILE_OVERRIDE"] = profile
    if overlay:
        env["QSE_SETUP_ENV"] = overlay
    script = ('set -a; . "$ENVF" >/dev/null 2>&1; for k in "$@"; do '
              'if [ -n "${!k+x}" ]; then printf "%s\\t%s\\n" "$k" "${!k}"; fi; done')
    try:
        r = subprocess.run(["bash", "-c", script, "-", *keys], capture_output=True, text=True,
                           timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition("\t")
        if k in keys:
            out[k] = v
    return out


# ---------------------------------------------------------------------- the box now
def _expand(v: str | None) -> str:
    return (v or "").replace("${HOME}", os.path.expanduser("~"))


def _unset_default(key: str) -> str | None:
    """What the engine runs with when serve.env leaves `key` unset: a QWEN38_* knob's default in
    engine/settings.py, else the `${KEY:-default}` the ops/start.sh that started this server
    passes on. None: neither names one."""
    if key.startswith("QWEN38_"):
        from engine.settings import KNOBS
        d = KNOBS.get(key[len("QWEN38_"):], (None,))[0]
        return None if d is None else os.path.expanduser(str(d))
    try:
        with open(os.path.join(device_mod.ops_dir(), "start.sh")) as fh:
            found = set(re.findall(r"\$\{" + re.escape(key) + r":-([^}]*)\}", fh.read()))
    except OSError:
        return None
    return found.pop() if len(found) == 1 else None


def current_selection(m: dict) -> dict:
    """What the box runs now as a selection (GET /v1/setup/models/{id}/current): the active
    profile and every registry knob as ops/serve.env resolves it (with run/setup.env), so the
    wizard starts from this box and not from the recommended setup, and the plan only lists what
    the user changes. A value the registry cannot express (a HOST that is neither option, a tool
    whose knobs disagree) stays at the recommended one and is listed in `unmatched`."""
    keys = sorted(set(registry.env_keys(m)) | set(m["parallelism"]["knobs"].values()))
    env = resolve_env(keys)
    unmatched = []
    body: dict = {"model": m["id"]}
    active = device_mod.active_profile()
    if active and registry.profile(m, active) is not None:
        body["profile"] = active
    comps = {}
    for c in m["components"]:
        if c["required"] or not c["knobs"]:
            continue
        votes = set()
        for key, v in c["knobs"].items():
            on, off = _expand(v["on"]), _expand(v["off"])
            now = env.get(key)
            if now is None:
                # unset is what the engine then runs with when that is one of the tool's two
                # values (the lookup store's default folder: on; start.sh's ${VISION:-on}), else
                # empty, as a role's unset model is
                d = _unset_default(key)
                now = d if d is not None and d in ((off,) if v["on"] is None else (on, off)) else ""
            if v["on"] is None:
                votes.add(now != off)
            elif now == on:
                votes.add(True)
            elif now == off:
                votes.add(False)
            else:
                votes.add(None)
        if votes in ({True}, {False}):
            comps[c["id"]] = votes == {True}
        else:
            unmatched.append({"field": f"components.{c['id']}",
                              "value": {k: env.get(k) for k in c["knobs"]}})
    body["components"] = comps
    par = m["parallelism"]
    for name, key in par["knobs"].items():
        v = env.get(key)
        if v is None:
            continue
        if name == "kv_pool":
            if v in ("on", "off"):
                body[name] = v == "on"
            continue
        try:
            body[name] = float(v) if name == "floor_gb" else int(v)
        except ValueError:
            unmatched.append({"field": f"parallelism.{name}", "value": v})
    for group in ("methods", "advanced"):
        vals = {}
        for f in m[group]:
            if not f["knob"] or f["knob"] not in env:
                continue
            raw = env[f["knob"]]
            try:
                v = {"int": int, "number": float}.get(f["type"], str)(raw)
            except ValueError:
                v = raw
            if f["type"] == "bool":
                v = raw.lower() in ("1", "true", "on", "yes")
            if registry.field_error(f, v):
                unmatched.append({"field": f"{group}.{f['id']}", "value": raw})
            else:
                vals[f["id"]] = v
        body[group] = vals
    # what does not check (a window off its step, tools that need each other) falls back to the
    # recommended values, the fewest parts first
    parts = {"components": ["components"], "parallelism": list(par["knobs"]),
             "methods": ["methods", "advanced"]}
    tries = [()] + [(k,) for k in parts] + [("components", "parallelism"),
                                             ("components", "methods"),
                                             ("parallelism", "methods"), tuple(parts)]
    first_error = None
    for drop in tries:
        b = {k: v for k, v in body.items() if not any(k in parts[d] for d in drop)}
        try:
            sel = estimate_mod.check(m, b)
        except estimate_mod.SelectionError as exc:
            first_error = first_error or str(exc)
            continue
        if drop:
            unmatched.append({"field": ", ".join(drop), "value": first_error})
        break
    return {"contract_version": "1.0", "model": m["id"], "selection": sel, "unmatched": unmatched}


# ---------------------------------------------------------------------- the plan
# the keys a change of which every client notices, and what to do about it
CLIENT_FACING = {
    "HOST": "the engine listens on another address; clients and the reverse proxy may not reach "
            "it any more.",
    "PORT": "clients, the reverse proxy and the watchdog must use the new port.",
    "SERVED_MODEL": "clients that ask for the old model name get an error until they use the new "
                    "one.",
    "QSE_BACKENDS": "the model names routed to other servers or cloud APIs change with it.",
}


def secrets_file() -> str:
    """The secrets file the engine reads, in ops/start.sh's order: QSE_SECRETS (an install's
    DIR/state/secrets.env), $QSE_STATE_DIR/secrets.env, ~/.qwen38-spark-engine/secrets.env."""
    if os.environ.get("QSE_SECRETS"):
        return os.path.expanduser(os.environ["QSE_SECRETS"])
    if os.environ.get("QSE_STATE_DIR"):
        return os.path.join(os.path.expanduser(os.environ["QSE_STATE_DIR"]), "secrets.env")
    return os.path.expanduser("~/.qwen38-spark-engine/secrets.env")


def plan(m: dict, sel: dict, *, setup_mode: bool, password_set: bool,
         available_gb: float | None = None, mem_total_gb: float | None = None) -> dict:
    """The review step's facts (SetupPlan): what is wrong, what changes, what will run."""
    est = estimate_mod.estimate(m, sel, available_gb=available_gb)
    prof = registry.profile(m, sel["profile"])
    disk = device_mod.on_disk(m, fresh=True).get(prof["id"], {})
    errors, codes, warnings = [], [], []

    def error(code: str, text: str) -> None:
        errors.append(text)
        codes.append(code)

    if not est["fits"]:
        error("over_budget", " ".join(n for n in est["notes"] if not n.startswith("About")))
    missing, fetched = _assets_missing(m, disk)
    if missing:
        error("missing_assets", "This profile needs files that are not on this box and that "
              "nothing here can download: " + "; ".join(
                  f"{a['label']} ({registry.expand(a['path'], disk.get('repo'))})"
                  for a in missing) + ". See docs/setup.md.")
    # VIS-153: the models of the tools the selection turns on, as the profile's own assets
    seen = {a["id"] for a in missing + fetched}
    tools_missing, tools_fetched = _tool_assets_missing(m, sel, disk.get("repo"), seen)
    for c, a in tools_missing:
        error("missing_assets", f"{c['label']} is on, but {a['label']} is not on this box "
              f"({registry.expand(a['path'], disk.get('repo'))}) and nothing here can download "
              f"it. Turn {c['label']} off in the Tools step, or put its files there.")
    fetched = fetched + [a for _, a in tools_fetched]
    weights_here = bool(disk.get("weights"))
    # OPS-95: the fetch step downloads the weights and the missing assets with a download list
    on_disk = weights_here and not fetched
    download_gb = round((0.0 if weights_here else prof["disk_gb"]) + _fetch_gb(fetched), 2)
    downloads = ([] if weights_here else [{"id": prof["id"], "label": prof["hf_repo"],
                                           "gb": prof["disk_gb"], "for": prof["label"]}]) + [
        {"id": a["id"], "label": a["label"], "gb": round(_fetch_gb([a]), 2),
         "for": next((c["label"] for c, x in tools_fetched if x is a), prof["label"])}
        for a in fetched]
    if not weights_here and prof["hf_private"] and not device_mod.hf_token_present():
        error("missing_token", f"{prof['hf_repo']} is private and no Hugging Face token is set. "
              f"Add one with `hf auth login` on the box, or set HF_TOKEN, then try again.")
    if not on_disk:
        try:
            free = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
        except OSError:
            free = None
        if free is not None and free < download_gb + DISK_MARGIN_GB:
            # SRV-560: and what could go to make room
            from server.setup import remove as remove_mod
            try:
                more = remove_mod.removable_text(remove_mod.installed(setup_mode=setup_mode),
                                                 download_gb + DISK_MARGIN_GB - free)
            except Exception:                                     # noqa: BLE001
                more = ""
            error("disk_full", f"The download needs {download_gb:g} GB and {DISK_MARGIN_GB} "
                  f"GB left after it; the disk has {free:.0f} GB free, {download_gb + DISK_MARGIN_GB - free:.0f} "
                  f"GB too few." + (f" {more}" if more else ""))
    if not hook_present():
        error("no_overlay_hook", "ops/serve.env on this box does not read run/setup.env. Add the "
              "setup overlay lines from the repository's ops/serve.env (the end of the file) "
              "first.")
    if sel["advanced"].get("admin_password") == "generate" and os.path.exists(secrets_file()):
        error("secrets_exist", "secrets.env already exists. ops/make-secrets.sh --rotate --admin "
              "would replace the metrics token too, so run it on the box yourself.")
    keys = list(overlay_values(m, sel))
    # VIS-155: the keys the run/setup.env on the box sets and the new one does not (the previous
    # model's own knobs) are compared too: the rewrite drops them
    gone = [k for k in overlay_keys(overlay_path()) if k not in keys]
    current = resolve_env(keys + gone + ["REPO"])
    repo = current.pop("REPO", None)
    if repo and os.path.realpath(repo) != os.path.realpath(ROOT):
        error("repo_mismatch", f"ops/serve.env's REPO is {repo}, but this server runs from "
                               f"{ROOT}; the overlay would not be read.")
    with_overlay = _preview(m, sel, keys + gone)
    changes = []
    for k in keys + gone:
        cur, nxt = current.get(k), with_overlay.get(k)
        # unset is what the engine runs with then (the overlay leaves such a key out)
        if (cur if cur is not None else _unset_default(k)) != \
                (nxt if nxt is not None else _unset_default(k)):
            c = {"key": k, "current": cur, "next": nxt,
                 "from": _source(m, k) or ("run/setup.env" if k in gone else ""),
                 "removed": k in gone}
            if SECRET_KEY.search(k):
                c.update(current=cur and "[redacted]", next=nxt and "[redacted]")
            changes.append(c)
    for c in changes:
        why = CLIENT_FACING.get(c["key"])
        if why:
            warnings.append(f"{c['key']} changes from {c['current'] or 'unset'} to "
                            f"{c['next'] or 'unset'}: {why}")
    old_port = int(current.get("PORT") or 8000) if (current.get("PORT") or "").isdigit() else 8000
    new_port = int(sel["advanced"].get("port") or old_port)
    if new_port != old_port and _port_taken(new_port):
        error("port_in_use", f"Port {new_port} is in use by another program on this box.")
    if not setup_mode and os.path.exists(pending_path()):
        warnings.append("run/setup-pending is on this box while the engine serves a model: its "
                        "next plain start would be setup mode. The install removes it.")
    if not password_set:
        warnings.append("No admin password is set: anyone who can reach this port can run the "
                        "setup, apply it and reset the box. Generate one in the advanced step.")
    if est["fits"] and not est["fits_worst"]:
        warnings.append(est["notes"][0])
    if mem_total_gb is not None and device_mod.preset(m, mem_total_gb, None) is None:
        warnings.append("These numbers were measured on a DGX Spark, not on this machine.")
    active = device_mod.active_profile()
    commands = []
    if not on_disk:
        what = ([prof["hf_repo"]] if not weights_here else []) + [a["label"] for a in fetched]
        commands.append(f"bash ops/use-profile.sh {prof['id']} --fetch-only   "
                        f"(about {download_gb:g} GB: {', '.join(what)})")
    if new_port != old_port:
        commands.append(f"bash ops/stop.sh 120   (the engine on port {old_port})")
    written = len(_without_unset_defaults(overlay_values(m, sel), sel["profile"]))
    commands.append(f"write run/setup.env ({written} keys; the old one kept as a .bak)")
    if sel["advanced"].get("admin_password") == "generate":
        commands.append("bash ops/make-secrets.sh --admin")
    commands.append(f"bash ops/use-profile.sh {prof['id']}   (stop, start, smoke, roll back on "
                    f"failure)")
    commands.append(f"smoke: GET http://127.0.0.1:{new_port}/health and one short chat")
    return {"contract_version": "1.0", "ok": not errors, "errors": errors, "error_codes": codes,
            "warnings": warnings,
            "profile": {"current": None if setup_mode and not active else active,
                        "next": prof["id"], "on_disk": on_disk, "download_gb": download_gb},
            "downloads": downloads, "fetch_assets": [a["id"] for _, a in tools_fetched],
            "changes": changes, "restart": True, "commands": commands, "estimate": est,
            "ports": {"current": old_port, "next": new_port}}


# a key whose value the plan never shows (a hand-edited run/setup.env can hold one)
SECRET_KEY = re.compile(r"TOKEN|PASSWORD|SECRET|PASSWD|_KEY$")


def overlay_keys(path: str) -> list[str]:
    """The keys a run/setup.env sets, in its order (none when there is no file)."""
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return []
    return list(dict.fromkeys(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)=", text, re.M)))


def _assets_missing(m: dict, disk: dict) -> tuple[list[dict], list[dict]]:
    """The profile's missing assets: (those nothing downloads, those use-profile.sh --fetch-only
    downloads: the ones with a fetch list, OPS-95)."""
    missing = [registry.asset(m, a) for a in disk.get("assets_missing", [])]
    return [a for a in missing if not a.get("fetch")], [a for a in missing if a.get("fetch")]


def _tool_assets_missing(m: dict, sel: dict, repo: str | None, seen: set) -> tuple[list, list]:
    """VIS-153: the assets of the tools the selection turns on that are not on this box, each once
    (the narrator and Wren share Qwen3.5-2B), as (tool, asset) pairs: (those nothing downloads,
    those the fetch step downloads)."""
    out_missing, out_fetched = [], []
    for c in m["components"]:
        if not sel["components"].get(c["id"]):
            continue
        for aid in c["assets"]:
            a = registry.asset(m, aid)
            if a is None or aid in seen or registry.asset_present(a, repo, m):
                continue
            seen.add(aid)
            (out_fetched if a.get("fetch") else out_missing).append((c, a))
    return out_missing, out_fetched


def _fetch_gb(assets: list[dict]) -> float:
    return sum(e.get("gb", 0.0) for a in assets for e in a["fetch"])


def _preview(m: dict, sel: dict, keys: list[str]) -> dict:
    import tempfile
    fd, path = tempfile.mkstemp(prefix="setup-preview-", suffix=".env")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(render_overlay(m, sel))
        return resolve_env(keys, profile=sel["profile"], overlay=path)
    finally:
        os.unlink(path)


def _source(m: dict, key: str) -> str:
    return registry.env_keys(m).get(key, "")


def _port_taken(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def reset_plan(m: dict, *, setup_mode: bool) -> dict:
    """Reset level 2's plan (the shape of `plan`): run/setup.env and its backups go, the engine
    starts on the recommended profile with what ops/serve.env alone says."""
    sel = estimate_mod.defaults(m)
    prof = registry.recommended_profile(m)
    disk = device_mod.on_disk(m, fresh=True).get(prof["id"], {})
    errors, codes = [], []
    missing, fetched = _assets_missing(m, disk)
    if missing:
        errors.append("The recommended profile needs files that are not on this box: " + "; ".join(
            f"{a['label']} ({registry.expand(a['path'], disk.get('repo'))})" for a in missing) + ".")
        codes.append("missing_assets")
    on_disk = bool(disk.get("weights")) and not fetched
    download_gb = round((0.0 if disk.get("weights") else prof["disk_gb"]) + _fetch_gb(fetched), 2)
    if not disk.get("weights") and prof["hf_private"] and not device_mod.hf_token_present():
        errors.append(f"{prof['hf_repo']} is private and no Hugging Face token is set.")
        codes.append("missing_token")
    keys = list(overlay_values(m, sel))
    keys += [k for k in overlay_keys(overlay_path()) if k not in keys]      # VIS-155
    extra = ["PORT", "SERVED_MODEL", "QSE_CONDUCTOR_CONFIG"]
    now = resolve_env(keys + extra)
    base = resolve_env(keys + extra, profile=prof["id"], overlay=os.path.join(run_dir(), ".none"))
    changes = [{"key": k, "current": now.get(k), "next": base.get(k), "from": "reset level 2",
                "removed": base.get(k) is None}
               for k in keys if now.get(k) != base.get(k) and not SECRET_KEY.search(k)]
    port = lambda d: int(d["PORT"]) if (d.get("PORT") or "").isdigit() else 8000  # noqa: E731
    cond = base.get("QSE_CONDUCTOR_CONFIG") or "off"
    smoke = "TandemLLM" if cond not in ("", "off") else (base.get("SERVED_MODEL") or
                                                         "qwen38-spark-engine")
    active = device_mod.active_profile()
    commands = ([f"bash ops/use-profile.sh {prof['id']} --fetch-only"] if not on_disk else []) + \
        ([f"bash ops/stop.sh 120   (the engine on port {port(now)})"]
         if port(now) != port(base) else []) + \
        ["remove run/setup.env and its backups", f"bash ops/use-profile.sh {prof['id']}",
         f"smoke: GET http://127.0.0.1:{port(base)}/health and one short chat"]
    return {"contract_version": "1.0", "ok": not errors, "errors": errors, "error_codes": codes,
            "warnings": [], "profile": {"current": None if setup_mode and not active else active,
                                        "next": prof["id"], "on_disk": on_disk,
                                        "download_gb": download_gb},
            "changes": changes, "restart": True, "commands": commands,
            "estimate": estimate_mod.estimate(m, sel), "ports": {"current": port(now),
                                                                 "next": port(base)},
            "smoke_model": smoke, "selection": sel}


# ---------------------------------------------------------------------- jobs
def job_dir(jid: str) -> str:
    if not JOB_ID.match(jid or ""):
        raise ValueError("not a job id")
    return os.path.join(jobs_dir(), jid)


def _write_json(path: str, data: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def read_job(jid: str) -> dict | None:
    """The job's state (SetupJob), with a runner that died marked failed (interrupted)."""
    try:
        with open(os.path.join(job_dir(jid), "state.json")) as fh:
            job = json.load(fh)
    except (OSError, ValueError):
        return None
    starting = job["state"] == "queued" and time.time() - job.get("created_at", 0) < 30
    if job["state"] in ("queued", "running") and not starting and not _alive(
            job.get("pid"), job.get("pid_start"), job.get("boot_id")):
        job.update(state="failed", message="interrupted: the job's process is gone",
                   ended_at=job.get("ended_at") or time.time())
    return job


def _proc_identity(pid: int) -> dict:
    """What tells this process apart from a later one with the same pid: its start time (clock
    ticks since boot, /proc/<pid>/stat field 22) and the boot's id. Empty off Linux."""
    out = {}
    try:
        with open(f"/proc/{pid}/stat") as fh:
            out["pid_start"] = int(fh.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/sys/kernel/random/boot_id") as fh:
            out["boot_id"] = fh.read().strip()
    except (OSError, IndexError, ValueError):
        return {}
    return out


def _alive(pid, pid_start=None, boot_id=None) -> bool:
    """The runner `pid` still runs: the pid exists, is no zombie, and (when the job recorded
    them) it is the same process: the same boot and the same start time; a reboot or a reused pid
    is not the runner."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if pid_start is not None or boot_id is not None:
        now = _proc_identity(pid)
        if now and (now.get("boot_id") != boot_id or now.get("pid_start") != pid_start):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                                  # a zombie child of this server
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def running_job() -> dict | None:
    try:
        ids = sorted(os.listdir(jobs_dir()))
    except OSError:
        return None
    for jid in ids:
        if JOB_ID.match(jid):
            j = read_job(jid)
            if j is not None and j["state"] in ("queued", "running"):
                return j
    return None


def events(jid: str, after: int = 0) -> list[dict]:
    """The job's events after sequence `after`, each {seq, event}."""
    out = []
    try:
        with open(os.path.join(job_dir(jid), "events.jsonl")) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("seq", 0) > after:
                    out.append(rec)
    except OSError:
        pass
    return out


def events_from(jid: str, offset: int = 0, after: int = 0) -> tuple[list[dict], int]:
    """The job's events from byte `offset` of events.jsonl on, those after sequence `after`, and
    the offset after the last whole line: a follower reads each line once (a line the runner is
    still writing waits for the next call)."""
    out = []
    try:
        with open(os.path.join(job_dir(jid), "events.jsonl"), "rb") as fh:
            fh.seek(offset)
            data = fh.read()
    except OSError:
        return out, offset
    end = data.rfind(b"\n") + 1
    for line in data[:end].splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("seq", 0) > after:
            out.append(rec)
    return out, offset + end


def job_summary(m: dict, sel: dict) -> dict:
    """VIS-211: what the install sets up, in the words of the page's finished screen: the model,
    the weights, the tools that are on, the parallel chats and their windows."""
    prof = registry.profile(m, sel["profile"]) or {}
    return {"model": m.get("name") or m["id"], "profile": sel["profile"],
            "profile_label": prof.get("label") or sel["profile"],
            "tools": [c["label"] for c in m["components"]
                      if sel["components"].get(c["id"]) and not c.get("required")],
            "slots": sel.get("slots"), "window": sel.get("window"),
            "side_window": sel.get("side_window")}


def new_job(m: dict, sel: dict, p: dict, *, setup_mode: bool, kind: str = "apply",
            retry_of: str | None = None, remove_after: list[str] | None = None) -> dict:
    """The job's folder and its job.json (what the runner reads), state queued. `kind`: apply, or
    reset (reset level 2: the overlay goes and the recommended profile starts)."""
    jid = secrets.token_hex(8)
    d = job_dir(jid)
    os.makedirs(d, mode=0o700)
    now = time.time()
    steps = [{"id": sid, "label": label, "state": "pending", "started_at": None, "ended_at": None,
              "detail": None} for sid, label in STEPS]
    job = {"id": jid, "kind": kind, "state": "queued", "created_at": now, "ended_at": None,
           "profile": sel["profile"], "steps": steps, "message": None, "changes": p["changes"],
           "rolled_back": False, "failed_step": None, "pid": None,
           "url": f"http://<this box>:{p['ports']['next']}/dashboard/",
           "model": m["id"], "summary": job_summary(m, sel), "retry_of": retry_of}
    _write_json(os.path.join(d, "job.json"),
                {"kind": kind, "model": m["id"], "selection": sel, "setup_mode": setup_mode,
                 "previous": p["profile"]["current"], "ports": p["ports"],
                 "on_disk": p["profile"]["on_disk"], "smoke_model": p.get("smoke_model"),
                 "remove_after": remove_after or []})
    _write_json(os.path.join(d, "state.json"), job)
    return job


def job_spec(jid: str) -> dict | None:
    """The job.json the server wrote (the selection, the kind): a retry starts from it."""
    try:
        with open(os.path.join(job_dir(jid), "job.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def diagnostics(jid: str, n_lines: int = 200) -> str | None:
    """VIS-213: a failed job as text for a bug report: versions, the job, its steps, tandem doctor
    (the job's report and a fresh check), the selection and the last log lines. Nothing secret:
    the log was filtered when it was written, and the whole text passes the filter again."""
    import platform
    from server.conductor.redact import redact
    from server.setup import doctor
    job = read_job(jid)
    if job is None:
        return None
    spec = job_spec(jid) or {}
    ver = ""
    try:
        with open(os.path.join(ROOT, "VERSION")) as fh:
            ver = fh.read().strip()
    except OSError:
        pass
    commit = ""
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    out = ["TandemLLM Setup Guide diagnostics",
           f"TandemLLM {ver or '?'}{' (' + commit + ')' if commit else ''}; Python {platform.python_version()}; "
           f"{platform.platform()}{'; GPU ' + gpu if gpu else ''}",
           f"job {job['id']} ({job.get('kind')}), {job.get('state')}"
           + (f", a retry of {job['retry_of']}" if job.get("retry_of") else "")
           + f"; profile {job.get('profile')}; model {job.get('model') or spec.get('model')}",
           f"message: {job.get('message') or '-'}", "", "steps:"]
    for st in job.get("steps") or []:
        took = f" {st['ended_at'] - st['started_at']:.1f} s" if st.get("started_at") and st.get("ended_at") else ""
        out.append(f"  {st['id']:<16} {st['state']:<8}{took}  {st.get('detail') or ''}".rstrip())
    d = job.get("doctor") or {}
    out += ["", "tandem doctor before the job: "
            + (f"repaired {len(d.get('repaired') or [])}, to look at {len(d.get('problems') or [])}" if d else "not run")]
    out += [f"  repaired: {t}" for t in d.get("repaired") or []] + [f"  to look at: {t}" for t in d.get("problems") or []]
    try:
        now = doctor.run(ROOT, repair=False, log=False)
        out += ["tandem doctor now: " + ("nothing to look at" if now["ok"] else f"{len(now['problems'])} to look at")]
        out += [f"  to look at: {t}" for t in now["problems"]]
    except Exception as exc:                                      # noqa: BLE001
        out.append(f"tandem doctor now: did not run ({type(exc).__name__})")
    sel = spec.get("selection") or {}
    out += ["", "selection: " + json.dumps({k: sel.get(k) for k in ("profile", "slots", "window", "side_window",
                                                                    "kv_pool", "components")}, sort_keys=True)]
    logs = [e["event"].get("line", "") for e in events(jid) if e["event"].get("type") == "log"]
    out += ["", f"log (the last {min(n_lines, len(logs))} of {len(logs)} lines):"] + logs[-n_lines:]
    text, _ = redact("\n".join(out) + "\n")
    for k in ("HF_TOKEN", "QSE_ADMIN_TOKEN"):
        v = os.environ.get(k)
        if v:
            text = text.replace(v, "[redacted]")
    return text


def spawn(jid: str) -> int:
    """The runner, detached: its own session, so the engine's restart does not take it along."""
    d = job_dir(jid)
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH", "HF_TOKEN", "HF_HOME",
                    "HF_HUB_CACHE", "HF_TOKEN_PATH", "QSE_OPS_DIR", "BOX_LOCK", "TZ",
                    "CUDA_HOME", "MEM_NEED_GB")}
    with open(os.path.join(d, "runner.out"), "ab") as out:
        p = subprocess.Popen([sys.executable, "-m", "server.setup.apply", "run", d], cwd=ROOT,
                             env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                             start_new_session=True, close_fds=True)
    return p.pid


# ---------------------------------------------------------------------- the runner
def _run_cmd(argv: list[str], *, timeout: float, env: dict, cwd: str, line) -> int:
    """One allowlisted command, its output line by line to `line`; the exit status (124 after
    the timeout)."""
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    t_end = time.monotonic() + timeout
    import selectors
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ)
    while True:
        if time.monotonic() > t_end:
            # its own session: the whole group goes (use-profile.sh's start.sh, curl, a fetch)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                p.kill()
            p.wait()
            line(f"[setup] {os.path.basename(argv[1])}: no end after {timeout:.0f} s, stopped")
            return 124
        if sel.select(timeout=1.0):
            ln = p.stdout.readline()
            if ln:
                line(ln.rstrip("\n"))
                continue
        if p.poll() is not None:
            for ln in p.stdout:
                line(ln.rstrip("\n"))
            return p.returncode


class Runner:
    """Runs one job's steps. `cmd(argv, timeout=, env=, cwd=, line=) -> rc` and
    `http(url, body=None, timeout=) -> dict` are injectable (tests/test_setup_apply.py records
    every argv)."""

    def __init__(self, jid: str, *, cmd=_run_cmd, http=None, ops: str | None = None,
                 clock=time.time, sleep=time.sleep):
        self.jid, self.dir = jid, job_dir(jid)
        self.cmd, self.http = cmd, http or _http
        self.ops = ops or device_mod.ops_dir()
        self.clock, self.sleep = clock, sleep
        with open(os.path.join(self.dir, "job.json")) as fh:
            self.spec = json.load(fh)
        with open(os.path.join(self.dir, "state.json")) as fh:
            self.job = json.load(fh)
        self.seq = len(events(jid))
        self.secrets = [v for v in (os.environ.get("HF_TOKEN"), os.environ.get("QSE_ADMIN_TOKEN"))
                        if v]

    # -- bookkeeping
    def _emit(self, ev: dict) -> None:
        self.seq += 1
        with open(os.path.join(self.dir, "events.jsonl"), "a") as fh:
            fh.write(json.dumps({"seq": self.seq, "event": ev}) + "\n")

    def _save(self) -> None:
        _write_json(os.path.join(self.dir, "state.json"), self.job)

    def log(self, text: str) -> None:
        from server.conductor.redact import redact
        for s in self.secrets:
            text = text.replace(s, "[redacted]")
        text, _ = redact(text)
        self._emit({"type": "log", "line": text[:2000], "ts": round(self.clock(), 3)})

    def step(self, sid: str, state: str, detail: str | None = None) -> None:
        st = next(s for s in self.job["steps"] if s["id"] == sid)
        now = self.clock()
        if state == "running":
            st["started_at"] = now
        elif state in ("done", "failed", "skipped"):
            st["ended_at"] = now
            st["started_at"] = st["started_at"] or now
        st["state"], st["detail"] = state, detail
        self._save()
        self._emit({"type": "step", "step": dict(st)})

    def argv(self, script: str, *args: str) -> list[str]:
        """An allowlisted script of ops/ and its arguments (registry ids and numbers only)."""
        if script not in SCRIPTS:
            raise ValueError(f"{script} is not one of the setup job's scripts")
        for a in args:
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", a):
                raise ValueError(f"argument {a!r} is not an id or a number")
        return ["bash", os.path.join(self.ops, script), *args]

    def env(self, **extra) -> dict:
        keep = ("PATH", "HOME", "LANG", "LC_ALL", "HF_TOKEN", "HF_HOME", "HF_HUB_CACHE",
                "HF_TOKEN_PATH", "BOX_LOCK", "TZ", "CUDA_HOME", "MEM_NEED_GB")
        e = {k: v for k, v in os.environ.items() if k in keep}
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def run(self, argv: list[str], timeout: float, **env) -> int:
        self.log(f"$ {' '.join(os.path.relpath(a, ROOT) if a.startswith(ROOT) else a for a in argv)}")
        self.last_lines = []

        def line(text: str) -> None:
            self.last_lines = (self.last_lines + [text])[-40:]
            self.log(text)
        return self.cmd(argv, timeout=timeout, env=self.env(**env), cwd=ROOT, line=line)

    def explain(self, what: str, rc: int, timeout: float) -> str:
        """VIS-213: a failed command in plain words, from its exit status and its last lines; when
        the cause is a file mode, tandem doctor repairs it now and the sentence says so."""
        lines = [ln for ln in getattr(self, "last_lines", []) if ln.strip()]
        text = "\n".join(lines)
        last = next((ln for ln in reversed(lines) if not ln.startswith("$ ")), "")
        if rc == 124:
            return f"{what}: it did not finish within {timeout / 60:.0f} minutes and was stopped. Press Retry."
        m = re.search(r"([\w./-]+\.sh): Permission denied", text)
        if m:
            from server.setup import doctor
            try:
                fixed = doctor.run(ROOT, repair=True, log=False)["repaired"]
            except Exception:                                     # noqa: BLE001
                fixed = []
            self.job.setdefault("doctor", {}).setdefault("repaired", [])
            self.job["doctor"]["repaired"] += fixed
            return (f"{what}: the script {os.path.basename(m.group(1))} could not be run (Permission "
                    f"denied: it lost its execute bit). " + ("tandem doctor has repaired its file mode; "
                                                             "press Retry." if fixed else
                                                             "Run tandem doctor on the box, then Retry."))
        rules = ((r"a hold owns the board", "a hold owns the board (a measurement or maintenance runs); "
                                            "try again when it ends."),
                 (r"an engine is already alive|REFUSING: an engine", "another engine is running on this "
                                                                     "box; stop it, then press Retry."),
                 (r"MemAvailable [\d.]+ GB < ", "there was not enough free memory to start it; stop other "
                                                 "programs that use memory, then press Retry."),
                 (r"FAILED to become healthy|did not answer /health",
                  "the new engine did not answer in time" + (f" (its log: {last})" if last else "") +
                  "."),
                 (r"smoke: /health does not name", "the engine that answered is not the new one."),
                 (r"smoke: the turn through", "the new engine started but did not answer a short chat."),
                 (r"download/verify FAILED|did not fetch|REFUSING the download",
                  f"the download stopped ({last or 'no reason given'}); press Retry to go on, the "
                  f"finished files are kept."),
                 (r"Traceback|Error:", f"it failed: {last}."))
        for pat, words in rules:
            if re.search(pat, text):
                return f"{what}: {words} (exit {rc})"
        return f"{what} (exit {rc})" + (f": {last}" if last else "") + "."

    # -- the job
    def main(self) -> int:
        self.job.update(state="running", pid=os.getpid(), **_proc_identity(os.getpid()))
        self._save()
        self._emit({"type": "job", "job": dict(self.job)})
        self.step("unload_assistant", "done", "Wren was unloaded before the job started")
        ok = False
        try:
            ok = self._steps()
        except Exception as exc:                                  # noqa: BLE001
            self.log(f"[setup] the job stopped: {type(exc).__name__}: {exc}")
            self.job["message"] = f"{type(exc).__name__}: {exc}"[:400]
            ok = False
        if not ok and self.spec.get("setup_mode") and not os.path.exists(pending_path()) \
                and self._restore_marker():
            # a box with no engine must still come up in setup mode on its next start
            self.log("[setup] run/setup-pending put back: the next start is setup mode again")
        end = self.job["state"] if self.job["state"] in ("done", "rolled_back") else (
            "done" if ok else "failed")
        self.job.update(state=end, ended_at=self.clock())
        for s in self.job["steps"]:
            if s["state"] == "pending":
                s["state"] = "skipped"
        self._save()
        # the job as it ended, the steps that did not run skipped, for a page that follows it live
        # (it showed them "not started" under a failed job, VIS-213)
        self._emit({"type": "job", "job": dict(self.job)})
        # VIS-153: the tools that did not load reach a page that follows the job live too
        self._emit({"type": "done", "state": end, "message": self.job["message"],
                    "warnings": self.job.get("warnings") or [], "notes": self.job.get("notes") or []})
        return 0 if end == "done" else 1

    def _fail(self, sid: str, why: str) -> bool:
        self.step(sid, "failed", why)
        self.job.update(failed_step=sid, message=why)
        return False

    def _doctor(self) -> None:
        """OPS-171: tandem doctor before anything else: what it repaired (file modes, the run/ops
        copies, stale pid files, the setup marker) and what it found go into the job and its log; a
        finding does not stop the job (the plan and the preflight decide)."""
        from server.setup import doctor
        try:
            rep = doctor.run(ROOT, repair=True, log=False)
        except Exception as exc:                                  # noqa: BLE001
            self.log(f"[doctor] did not run: {type(exc).__name__}: {exc}")
            return
        self.job["doctor"] = {"repaired": rep["repaired"], "problems": rep["problems"]}
        self._save()
        for t in rep["repaired"]:
            self.log(f"[doctor] repaired: {t}")
        for t in rep["problems"]:
            self.log(f"[doctor] to look at: {t}")

    def _steps(self) -> bool:
        spec = self.spec
        self.step("check", "running")
        self._doctor()
        m = device_mod.windowed(registry.get(spec["model"]))
        if m is None:
            return self._fail("check", f"no model {spec['model']!r} in models/")
        if spec.get("kind") == "reset":
            return self._reset_steps(m)
        try:
            sel = estimate_mod.check(m, spec["selection"])
        except estimate_mod.SelectionError as exc:
            return self._fail("check", f"the selection does not check: {exc}")
        if not profile_ok(sel["profile"], self.ops):
            return self._fail("check", f"no profile {sel['profile']} on this box (ops/profiles/, "
                                       f"or a community pack in the index)")
        prev = self._previous()
        p = plan(m, sel, setup_mode=spec["setup_mode"], password_set=True)
        if p["errors"]:
            return self._fail("check", " ".join(p["errors"]))
        if _box_lock_held():
            return self._fail("check", "the box lock (~/.qwen38-box.flock) is held: a hold or "
                                       "another job owns the board; try again later")
        self.step("check", "done", f"{len(p['changes'])} settings change")
        # fetch
        if p["profile"]["on_disk"]:
            self.step("fetch", "skipped", "the weights are on this box")
        else:
            self.step("fetch", "running", f"about {p['profile']['download_gb']:g} GB")
            extra = p.get("fetch_assets") or []
            if any(not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", x) for x in extra):
                return self._fail("fetch", "an asset id that is not an id")
            rc = self.run(self.argv("use-profile.sh", sel["profile"], "--fetch-only"),
                          TIMEOUTS["fetch"], **({"QSE_FETCH_ASSETS": " ".join(extra)}
                                                if extra else {}))
            if rc != 0:
                return self._fail("fetch", self.explain("The download did not finish", rc,
                                                        TIMEOUTS["fetch"]) + " Nothing else changed.")
            self.step("fetch", "done")
        old_port, new_port = spec["ports"]["current"], spec["ports"]["next"]
        if not self._preflight_step(m, sel["profile"], render_overlay(m, sel), old_port, new_port,
                                    secrets=sel["advanced"].get("admin_password") == "generate"):
            return False
        # write_config: nothing has changed until the engine is idle
        if not self._wait_idle("write_config", old_port):
            return False
        self.step("write_config", "running")
        stopped = new_port != old_port
        if stopped:
            self.run(self.argv("stop.sh", "120"), TIMEOUTS["stop"])
        backup = None
        try:
            backup = write_overlay(m, sel, self.clock)
            self.job["backup"] = backup
            if sel["advanced"].get("admin_password") == "generate":
                rc = self.run(self.argv("make-secrets.sh", "--admin"), TIMEOUTS["secrets"])
                if rc != 0:
                    raise RuntimeError(f"ops/make-secrets.sh failed (exit {rc})")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            if not stopped:
                restore_overlay(backup)
                return self._fail("write_config", f"{exc}; the old configuration is back")
            # the old engine is stopped already: the rollback starts it again
            self._fail("write_config", f"{exc}")
            return self._rollback(m, spec, backup, prev, old_port, new_port)
        self.step("write_config", "done", f"run/setup.env, {len(_without_unset_defaults(overlay_values(m, sel), sel['profile']))} keys"
                                          + (f"; the old one is {os.path.basename(backup)}"
                                             if backup else ""))
        # switch
        self.step("switch", "running", "about 4 minutes; the dashboard is away meanwhile")
        self._take_marker()
        smoke_model = "TandemLLM" if sel["components"].get("conductor") else \
            sel["advanced"].get("served_model", "qwen38-spark-engine")
        # use-profile.sh does not roll back itself (QSE_NO_ROLLBACK): it would restart the old
        # profile under the new overlay, and this runner's rollback would do it all again
        rc = self.run(self.argv("use-profile.sh", sel["profile"]), TIMEOUTS["switch"],
                      SMOKE_URL=f"http://127.0.0.1:{new_port}/v1", SMOKE_MODEL=smoke_model,
                      QSE_NO_ROLLBACK="1")
        if rc != 0:
            self._fail("switch", self.explain("The new setup did not start", rc, TIMEOUTS["switch"]))
            return self._rollback(m, spec, backup, prev, old_port, new_port)
        self.step("switch", "done")
        # smoke
        self.step("smoke", "running")
        why = self._smoke(sel["profile"], new_port, smoke_model)
        if why:
            self._fail("smoke", why)
            return self._rollback(m, spec, backup, prev, old_port, new_port)
        failed, notes = self._roles(m, sel, new_port)
        self.job.update(warnings=failed, notes=notes)
        self.step("smoke", "done", f"{sel['profile']} answers on port {new_port}" + (
            f"; {len(failed)} tool{'s' if len(failed) != 1 else ''} did not load" if failed else ""))
        self.job["message"] = " ".join([f"{sel['profile']} is live on port {new_port}.", *failed])
        self._drop_marker_copy()
        self._remove_after(sel["profile"])
        return True

    def _remove_after(self, live: str) -> None:
        """SRV-560: the weights the user chose to remove with this switch, now that the new setup
        answers: never the live profile's; a refusal is a note, the install is done either way."""
        want = [p for p in self.spec.get("remove_after") or [] if p != live]
        if not want:
            return
        from server.setup import remove as remove_mod
        try:
            r = remove_mod.remove(want, setup_mode=False, root=ROOT)
        except remove_mod.RemoveError as exc:
            self.job.setdefault("notes", []).append(f"The old weights were not removed: {exc}")
            self.log(f"[remove] not removed: {exc}")
            return
        self.job["removed"] = r
        for g in r["removed"]:
            self.log(f"[remove] {g['profile']}: {g['dir']} removed ({g['gb']:g} GB); the line that "
                     f"downloads it again is in {r['log']}")

    def _roles(self, m: dict, sel: dict, port: int) -> tuple[list[str], list[str]]:
        """VIS-153: the tools the selection turns on that run a model of their own, as the new
        engine reports them (`/health` roles, `/v1/models` decision_models), polled until none is
        still loading or ROLE_SETTLE_POLLS have passed: (those that failed, each with its reason;
        notes: waiting for memory, still loading). Nothing when the engine reports no roles."""
        want = {c["id"]: c["label"] for c in m["components"]
                if sel["components"].get(c["id"]) and c["id"] in ROLE_OF}
        st: dict = {}
        for i in range(ROLE_SETTLE_POLLS):
            st = self._role_states(port, want)
            if not any(v[0] == "pending" for v in st.values()):
                break
            if i + 1 < ROLE_SETTLE_POLLS:
                self.sleep(ROLE_POLL_S)
        failed = [f"{want[k]} did not load ({why or 'no reason given'})."
                  for k, (state, why) in st.items() if state == "failed"]
        # VIS-211: a tool that waits is normal, and the page says so in plain words; the log keeps
        # the planner's reason
        notes = [f"{want[k]} loads later, when memory frees up; nothing to do."
                 for k, (state, _) in st.items() if state == "refused"]
        notes += [f"{want[k]} is still loading; nothing to do." for k, (state, _) in st.items()
                  if state == "pending"]
        for line in failed:
            self.log(f"[smoke] {line}")
        for k, (state, why) in st.items():
            if state in ("refused", "pending"):
                self.log(f"[smoke] {want[k]}: {state}" + (f" ({why})" if why else ""))
        return failed, notes

    def _role_states(self, port: int, want: dict) -> dict:
        """tool id -> (loaded | failed | refused | pending, why) for the tools the engine reports."""
        base = f"http://127.0.0.1:{port}"
        out = {}
        try:
            roles = ((self.http(base + "/health", timeout=10) or {}).get("roles") or {}).get(
                "roles") or {}
        except Exception:                                         # noqa: BLE001
            roles = {}
        for k in want:
            if ROLE_OF[k] is None or ROLE_OF[k] not in roles:
                continue
            r = roles[ROLE_OF[k]]
            state = str(r.get("state"))
            if state == "loaded":
                out[k] = ("loaded", None)
            elif state == "failed":
                out[k] = ("failed", r.get("error"))
            elif state == "unloaded_for_memory" or r.get("refused") or r.get("refusal"):
                out[k] = ("refused", r.get("refused") or r.get("refusal") or r.get("why") or
                          "memory")
            else:
                out[k] = ("pending", None)
        if "laya" in want:
            try:
                dm = (self.http(base + "/v1/models", timeout=10) or {}).get("decision_models")
            except Exception:                                     # noqa: BLE001
                dm = None
            if isinstance(dm, list) and dm:
                sts = [str(d.get("state")) for d in dm]
                out["laya"] = (("failed", "a decision model failed to load") if "failed" in sts
                               else ("loaded", None) if "loaded" in sts else ("pending", None))
        return out

    def _previous(self) -> str | None:
        prev = self.spec.get("previous")
        if prev is not None and not profile_ok(prev, self.ops):
            return None
        return prev

    def _reset_steps(self, m: dict) -> bool:
        """Reset level 2: the overlay and its backups go (the overlay kept in the job's folder
        for the rollback), then the recommended profile starts as serve.env alone says."""
        spec = self.spec
        p = reset_plan(m, setup_mode=spec["setup_mode"])
        prof = p["profile"]["next"]
        if p["errors"]:
            return self._fail("check", " ".join(p["errors"]))
        if _box_lock_held():
            return self._fail("check", "the box lock (~/.qwen38-box.flock) is held; try again "
                                       "later")
        prev = self._previous()
        self.step("check", "done", f"{len(p['changes'])} settings go back")
        if p["profile"]["on_disk"]:
            self.step("fetch", "skipped", "the weights are on this box")
        else:
            self.step("fetch", "running", f"about {p['profile']['download_gb']:g} GB")
            rc = self.run(self.argv("use-profile.sh", prof, "--fetch-only"), TIMEOUTS["fetch"])
            if rc != 0:
                return self._fail("fetch", f"the download did not finish (exit {rc}); nothing "
                                           f"else changed")
            self.step("fetch", "done")
        old_port, new_port = p["ports"]["current"], p["ports"]["next"]
        if not self._preflight_step(m, prof, None, old_port, new_port):
            return False
        if not self._wait_idle("write_config", old_port):
            return False
        self.step("write_config", "running")
        keep = os.path.join(self.dir, "setup.env.before")
        overlay = overlay_path()
        if os.path.exists(overlay):
            shutil.copy2(overlay, keep)
        backup = keep if os.path.exists(keep) else None
        if new_port != old_port:
            # stopped under the overlay that names the old port
            self.run(self.argv("stop.sh", "120"), TIMEOUTS["stop"])
        gone = 0
        for f in glob.glob(overlay + "*"):
            if f == overlay or ".bak-" in f:
                os.unlink(f)
                gone += 1
        self.step("write_config", "done", f"run/setup.env and {max(0, gone - 1)} backup(s) removed")
        self.step("switch", "running", "about 4 minutes; the dashboard is away meanwhile")
        self._take_marker()
        rc = self.run(self.argv("use-profile.sh", prof), TIMEOUTS["switch"],
                      SMOKE_URL=f"http://127.0.0.1:{new_port}/v1", SMOKE_MODEL=p["smoke_model"],
                      QSE_NO_ROLLBACK="1")
        if rc != 0:
            self._fail("switch", self.explain("The engine did not start", rc, TIMEOUTS["switch"]))
            return self._rollback(m, spec, backup, prev, old_port, new_port)
        self.step("switch", "done")
        self.step("smoke", "running")
        why = self._smoke(prof, new_port, p["smoke_model"])
        if why:
            self._fail("smoke", why)
            return self._rollback(m, spec, backup, prev, old_port, new_port)
        self.step("smoke", "done", f"{prof} answers on port {new_port}")
        self.job["message"] = f"The shipped configuration runs, on {prof}."
        self._drop_marker_copy()
        return True

    # -- the preflight (OPS-169)
    def _preflight_step(self, m: dict, profile: str, overlay: str | None, old_port: int,
                        new_port: int, secrets: bool = False) -> bool:
        self.step("preflight", "running")
        try:
            problems = preflight(m, profile, overlay, ops=self.ops, old_port=old_port,
                                 new_port=new_port, setup_mode=bool(self.spec.get("setup_mode")),
                                 secrets=secrets, http=self.http)
        except Exception as exc:                                  # noqa: BLE001
            problems = [f"the check itself failed ({type(exc).__name__}: {exc})"]
        for t in problems:
            self.log(f"[preflight] {t}")
        if problems:
            return self._fail("preflight", " ".join(problems) + " Nothing was stopped or changed.")
        self.step("preflight", "done", "the scripts, the new configuration, the weights, memory and "
                                       "the port are ready")
        return True

    # -- idle, the setup-mode marker
    def _wait_idle(self, sid: str, port: int) -> bool:
        """Before anything stops: no request runs or waits on the engine (its /health, which a
        tool on the box reads in full), at most IDLE_WAIT_S; the step shows `waiting`
        meanwhile. Setup mode has no engine to wait for."""
        if self.spec.get("setup_mode"):
            return True
        t_end = time.monotonic() + IDLE_WAIT_S
        shown = None
        while True:
            try:
                h = self.http(f"http://127.0.0.1:{port}/health", timeout=5)
                inf = (h or {}).get("inflight") or {}
                n = int(inf.get("running", 0)) + int(inf.get("waiting", 0)) + \
                    int(inf.get("lane", 0))
            except Exception:                                    # noqa: BLE001
                return True                 # no engine answers: there is nothing to drain
            if n == 0:
                return True
            if n != shown:
                shown = n
                self.step(sid, "waiting", f"waiting for {n} request{'s' if n != 1 else ''} to "
                                          f"finish before the restart")
            if time.monotonic() > t_end:
                return self._fail(sid, f"{n} request{'s' if n != 1 else ''} still ran after "
                                       f"{IDLE_WAIT_S // 60} minutes; nothing changed. Try again "
                                       f"when the engine is idle.")
            self.sleep(IDLE_POLL_S)

    def _take_marker(self) -> None:
        """run/setup-pending goes before the switch (start.sh would start setup mode again), a
        copy kept in the job's folder: a rollback or a crash puts it back unchanged (its HOST)."""
        pend = pending_path()
        if os.path.exists(pend):
            shutil.copy2(pend, os.path.join(self.dir, "setup-pending.before"))
            os.unlink(pend)

    def _restore_marker(self) -> bool:
        saved = os.path.join(self.dir, "setup-pending.before")
        if not os.path.exists(saved):
            return False
        os.makedirs(run_dir(), exist_ok=True)
        shutil.copy2(saved, pending_path())
        return True

    def _drop_marker_copy(self) -> None:
        try:
            os.unlink(os.path.join(self.dir, "setup-pending.before"))
        except OSError:
            pass

    def _smoke(self, profile: str, port: int, model: str) -> str | None:
        base = f"http://127.0.0.1:{port}"
        try:
            h = self.http(base + "/health", timeout=10)
        except Exception as exc:                                  # noqa: BLE001
            return f"/health did not answer ({type(exc).__name__})"
        if ((h or {}).get("weights") or {}).get("name") != profile:
            return f"/health does not name {profile}"
        # use-profile.sh's own smoke: a thinking-capable template may spend tokens before the word
        body = {"model": model, "max_tokens": 512, "temperature": 0,
                "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = self.http(base + "/v1/chat/completions", body=body, timeout=300)
            text = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        except Exception as exc:                                  # noqa: BLE001
            return f"the chat did not answer ({type(exc).__name__})"
        if not text.strip():
            return "the chat answered with no text"
        self.log(f"[smoke] {text.strip()[:60]}")
        return None

    def _rollback(self, m, spec, backup, prev, old_port, new_port=None) -> bool:
        self.step("rollback", "running")
        if new_port is not None and new_port != old_port:
            # whatever started on the new port goes first, while the overlay still names it
            self.run(self.argv("stop.sh", "120"), TIMEOUTS["stop"])
        restore_overlay(backup)
        if spec["setup_mode"]:
            if not self._restore_marker():
                os.makedirs(run_dir(), exist_ok=True)
                with open(pending_path(), "w") as fh:
                    fh.write("setup mode: the apply job rolled back\n")
            self.run(self.argv("stop.sh", "120"), TIMEOUTS["stop"])
            rc = self.run(self.argv("start.sh", "--setup"), TIMEOUTS["start"])
        elif prev is not None:
            got = resolve_env(["SERVED_MODEL", "QSE_CONDUCTOR_CONFIG"])
            cond = got.get("QSE_CONDUCTOR_CONFIG") or "off"
            smoke = "TandemLLM" if cond not in ("", "off") else (got.get("SERVED_MODEL") or
                                                                 "qwen38-spark-engine")
            rc = self.run(self.argv("use-profile.sh", prev), TIMEOUTS["switch"],
                          SMOKE_URL=f"http://127.0.0.1:{old_port}/v1", SMOKE_MODEL=smoke,
                          QSE_NO_ROLLBACK="1")
        else:
            rc = self.run(self.argv("start.sh"), TIMEOUTS["start"])
        if rc == 0:
            self.step("rollback", "done", "the previous setup runs again")
            self.job.update(state="rolled_back", rolled_back=True)
            self._drop_marker_copy()
        else:
            self.step("rollback", "failed", f"the previous setup did not start either (exit {rc}); "
                                            f"see logs/engine.log on the box")
            self.job["rolled_back"] = False
        return False


# ---------------------------------------------------------------------- the preflight (OPS-169)
#: the scripts the switch runs (a missing roles.sh only warns in start.sh; make-secrets.sh only with
#: a generated password)
PREFLIGHT_SCRIPTS = ("use-profile.sh", "start.sh", "stop.sh", "engines.sh")
PREFLIGHT_PY = ("engine_status.py", "fetch_weights.py")


def _sources(ops: str, profile: str, overlay: str | None, keys: list[str]) -> tuple[dict, str | None]:
    """serve.env with `profile` and the overlay text the job will write (None: no overlay), under
    set -u as start.sh sources it, in a subshell: the keys' values, and the shell's own error."""
    import tempfile
    envf = os.path.join(ops, "serve.env")
    fd, tmp = tempfile.mkstemp(prefix="setup-preflight-", suffix=".env")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(overlay or "")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.path.expanduser("~"),
               "ENVF": envf, "QSE_PROFILE_OVERRIDE": profile, "QSE_SETUP_ENV": tmp}
        script = ('set -u; set -a; . "$ENVF" || exit 3; set +a; for k in "$@"; do '
                  'if [ -n "${!k+x}" ]; then printf "%s\\t%s\\n" "$k" "${!k}"; fi; done')
        r = subprocess.run(["bash", "-c", script, "-", *keys], capture_output=True, text=True,
                           timeout=20, env=env)
    finally:
        os.unlink(tmp)
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition("\t")
        out[k] = v
    if r.returncode != 0:
        return out, (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1].replace(envf, "serve.env")
    return out, None


def _runs(py: str) -> str | None:
    if not py:
        return "is not set"
    if os.sep in py and not os.path.exists(py):
        return f"({py}) does not exist"
    try:
        r = subprocess.run([py, "-c", "import json, os, sys"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"({py}) does not run: {exc}"
    return None if r.returncode == 0 else f"({py}) does not run: " + \
        (r.stderr.strip().splitlines() or ["?"])[-1]


def _weights_sizes(wdir: str) -> list[str]:
    """The files of the folder's hub manifest (ops/fetch_weights.py) whose size is not the hub's."""
    try:
        with open(os.path.join(wdir, "_hub_manifest.json")) as fh:
            files = json.load(fh).get("files") or {}
    except (OSError, ValueError):
        return []
    bad = []
    for name, f in sorted(files.items()):
        p = os.path.join(wdir, name)
        if isinstance(f, dict) and f.get("size") and os.path.isfile(p) and os.path.getsize(p) != f["size"]:
            bad.append(f"{name} has {os.path.getsize(p):,} bytes, the hub lists {f['size']:,}")
    return bad


def preflight(m: dict, profile: str, overlay: str | None, *, ops: str, old_port: int, new_port: int,
              setup_mode: bool, secrets: bool = False, http=None) -> list[str]:
    """What the restart needs, checked while the running setup still runs (OPS-169): each problem as a
    sentence; none: go on. Nothing here changes anything."""
    out = []
    # 1. the scripts the switch runs are there and parse
    for name in PREFLIGHT_SCRIPTS + (("make-secrets.sh",) if secrets else ()) + ("serve.env",):
        p = os.path.join(ops, name)
        if not os.path.isfile(p):
            out.append(f"ops/{name} is missing on this box (run tandem doctor, or the installer again).")
            continue
        r = subprocess.run(["bash", "-n", p], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            out.append(f"ops/{name} has an error: {(r.stderr.strip().splitlines() or ['?'])[-1]}")
    for name in PREFLIGHT_PY:
        p = os.path.join(ops, name)
        try:
            with open(p) as fh:
                compile(fh.read(), p, "exec")
        except OSError:
            out.append(f"ops/{name} is missing on this box.")
        except SyntaxError as exc:
            out.append(f"ops/{name} has an error on line {exc.lineno}: {exc.msg}.")
    # 2. the new configuration sources as start.sh sources it
    env, err = _sources(ops, profile, overlay, ["QSE_WEIGHTS", "QSE_WEIGHTS_DIR", "PY", "CHATD_PY",
                                                 "QSE_CHATD", "QSE_MEM_NEED_GIB"])
    if err:
        out.append(f"serve.env does not load with the new configuration: {err}.")
    elif env.get("QSE_WEIGHTS") and not profile.startswith(registry.COMMUNITY_PREFIX) and \
            env["QSE_WEIGHTS"] != profile:
        out.append(f"serve.env names the weights {env['QSE_WEIGHTS']} with the new configuration, not "
                   f"{profile}.")
    # 3. the weights are complete
    disk = device_mod.on_disk(m, fresh=True).get(profile, {})
    if not disk.get("weights"):
        out.append(f"The weights of {profile} are not complete on this box"
                   f"{' (' + disk['dir'] + ')' if disk.get('dir') else ''}: run the install again to "
                   f"finish the download.")
    else:
        out += [f"A weight file is cut short: {b}." for b in _weights_sizes(disk.get("dir") or
                                                                            env.get("QSE_WEIGHTS_DIR", ""))]
    missing = [registry.asset(m, a)["label"] for a in disk.get("assets_missing", []) if registry.asset(m, a)]
    if missing:
        out.append("Files the profile reads are not on this box: " + ", ".join(missing) + ".")
    # 4. memory: what is free now and what the running engine gives back at its stop
    mem = device_mod.meminfo()
    # what ops/use-profile.sh waits for after the stop: MEM_NEED_GB, else the profile's, else 80
    need = float(os.environ.get("MEM_NEED_GB") or env.get("QSE_MEM_NEED_GIB") or 80)
    if "MemAvailable" in mem:
        freed = 0.0
        if not setup_mode and http is not None:
            try:
                mm = (http(f"http://127.0.0.1:{old_port}/health", timeout=5) or {}).get("memory") or {}
                freed = max([float(mm[k]) for k in ("engine_reserved_gb", "reserved_gb", "rss_gb")
                             if isinstance(mm.get(k), (int, float))] or [0.0])
            except Exception:                                     # noqa: BLE001
                freed = None
        if freed is not None and mem["MemAvailable"] + freed + 2 < need:
            out.append(f"Not enough memory for {profile}: it needs {need:g} GiB free to start; "
                       f"{mem['MemAvailable']:.1f} GiB are free now and the running engine gives back "
                       f"about {freed:.1f}. Stop other programs that use memory, then try again.")
    # 5. the port, the box lock, a hold
    if new_port != old_port and _port_taken(new_port):
        out.append(f"Port {new_port} is in use by another program on this box.")
    if _box_lock_held():
        out.append("The box lock (~/.qwen38-box.flock) is held: a hold or another job owns the board.")
    pause = os.path.join(ROOT, ".watchdog.off")
    if os.path.exists(pause) and time.time() - os.stat(pause).st_mtime < 2400:
        out.append("A hold owns the board (.watchdog.off): the start would refuse.")
    # 6. the interpreters
    why = _runs(env.get("PY", ""))
    if why:
        out.append(f"The engine's Python {why}.")
    if (env.get("QSE_CHATD") or "on") != "off":
        why = _runs(env.get("CHATD_PY") or env.get("PY", ""))
        if why:
            out.append(f"chatd's Python {why}.")
    return out


def write_overlay(m: dict, sel: dict, clock=time.time) -> str | None:
    """run/setup.env, written whole and renamed into place after `bash -n`; the one it replaces
    kept as run/setup.env.bak-<utc>. Returns that backup's path (None: there was none)."""
    os.makedirs(run_dir(), exist_ok=True)
    text = render_overlay(m, sel)
    overlay = overlay_path()
    tmp = overlay + f".tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        os.unlink(tmp)
        raise ValueError(f"the generated overlay does not parse: {r.stderr.strip()[:200]}")
    backup = None
    if os.path.exists(overlay):
        backup = overlay + time.strftime(".bak-%Y%m%dT%H%M%SZ", time.gmtime(clock()))
        shutil.copy2(overlay, backup)
    os.replace(tmp, overlay)
    return backup


def restore_overlay(backup: str | None) -> None:
    overlay = overlay_path()
    if backup and os.path.exists(backup):
        shutil.copy2(backup, overlay)
    elif os.path.exists(overlay):
        os.unlink(overlay)


def _box_lock_held() -> bool:
    from server.decider_api import held
    return held(service_start=False, hold_restart=False)


def _http(url: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "X-Tandem-Memory": "off"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def main(argv: list[str] | None = None) -> int:
    """`run <checkout>/run/setup-jobs/<id>`: the job the server wrote, in the checkout it serves."""
    global ROOT
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 2 and a[0] == "run":
        d = os.path.realpath(a[1])
        jobs = os.path.dirname(d)
        if JOB_ID.match(os.path.basename(d)) and os.path.basename(jobs) == "setup-jobs" and \
                os.path.basename(os.path.dirname(jobs)) == "run":
            ROOT = os.path.dirname(os.path.dirname(jobs))
            return Runner(os.path.basename(d)).main()
    print("usage: python -m server.setup.apply run <checkout>/run/setup-jobs/<id>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
