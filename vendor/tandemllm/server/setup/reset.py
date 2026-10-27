"""The reset levels (SRV-329, decision D9): what each one removes, how big it is, and the removal.

    1  the setup page's progress and Wren's chat     the browser's own (localStorage); 400 here
    2  the applied configuration                     run/setup.env and its backups go, the profile goes
                                                     back to the registry's recommended one: a job of
                                                     the apply runner (server/setup/apply.py), so the
                                                     restart, the smoke and the rollback are the same
    3  caches and stored chats                       the disk tier of every profile and its spilled
                                                     sessions, the RAM caches, the response cache,
                                                     the lookup store, the shadow reader's disk
    4  downloaded weights but the active profile's   each other profile's weight folder and its
                                                     Hugging Face cache copy
    5  usage history and conductor data              the usage ledger file (and its -wal, -shm and
                                                     -journal sidecars), the conductor's config and
                                                     journal, the memory notes and transcripts

GET /v1/setup/reset lists every level with its items and sizes; POST /v1/setup/reset {level,
confirm, dry_run?} removes one. Level 2 needs `confirm: true`; 3 to 5 the level's word typed
exactly (WORDS). Every reset is refused while requests are in flight or a setup job runs.

WHAT IS NEVER REMOVED by levels 2 to 4: the active profile's weights, the protected assets (the K3
drafters, the K4 draft head, the FP8 base checkpoint: nothing downloads them again), Wren's and the
small roles' models, the Hugging Face token, the secrets file the engine reads and its folder
(QSE_SECRETS, else $QSE_STATE_DIR/secrets.env, else ~/.qwen38-spark-engine/secrets.env: an install's
is DIR/state/secrets.env), the ledger (QSE_USAGE_LEDGER), the conductor's data and this checkout. Only paths from the table here, resolved from serve.env and the registry, can go, each
checked to be under the home folder and none of the above; a symbolic link is removed as a link and
never followed. A cache the engine runs is cleared through its own clear function under the engine's
lock (wiping a disk tier behind a running engine's back broke it, SRV-289); the shadow reader's disk,
which has none, is left alone while the shadow reader runs. Every call writes one line to
run/setup-audit.log: the level, dry run or not, the GiB freed and the result, no paths of content.

LEVEL 4 never removes what an engine reads: besides the active profile's folder, the weight paths
in the argument vector and the environment of every engine process on any port (ops/engines.sh
engine_pids: --nvfp4, --fp8-head, QWEN38_EXL3, QWEN38_DRAFT_*, QWEN38_NVFP4, QSE_WEIGHTS_DIR) and
this server's own, and ~/nvfp4-bf16 while the ~/tandem-staircut rollback tree is on the box. With
no active profile known (no QSE_WEIGHTS, no ops/active-profile) it removes nothing.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time

from server.setup import apply, device as device_mod, registry

WORDS = {2: None, 3: "delete caches", 4: "delete weights", 5: "delete history"}
LABELS = {2: ("Applied configuration", "The setup guide's configuration goes, and the engine "
                                       "restarts on the recommended weight profile."),
          3: ("Caches and stored chats", "Every cache on the SSD and in memory, and the "
                                         "conversations parked there. Answers stay right; the next "
                                         "long prompts are slower once."),
          4: ("Downloaded weights", "The weights of every profile but the active one. They "
                                    "download again when you pick that profile."),
          5: ("Usage history and conductor data", "The usage history, the conductor's settings and "
                                                  "journal, and the memory notes. This cannot be "
                                                  "undone.")}
RESTART = {2: True, 3: False, 4: False, 5: True}
SIZE_S = 2.0


def state_dir() -> str:
    """The caches, the conductor, the memory notes, the models and the deciders: an install keeps
    them here too (install.sh moves only the secrets and the ledger, to DIR/state)."""
    return os.path.expanduser("~/.qwen38-spark-engine")


def secrets_files() -> list[str]:
    """The secrets file the engine reads (apply.secrets_file()) and the one serve.env names, when
    that is another (OPS-163)."""
    out = [apply.secrets_file()]
    got = _env(["QSE_SECRETS", "QSE_STATE_DIR"])
    if got.get("QSE_SECRETS"):
        out.append(os.path.expanduser(got["QSE_SECRETS"]))
    elif got.get("QSE_STATE_DIR"):
        out.append(os.path.join(os.path.expanduser(got["QSE_STATE_DIR"]), "secrets.env"))
    return list(dict.fromkeys(out))


def state_dirs() -> list[str]:
    """Real paths of the state folders: ~/.qwen38-spark-engine and each secrets file's folder
    (an install's DIR/state). A guard, and an item inside one may go."""
    return list(dict.fromkeys(os.path.realpath(p) for p in
                              [state_dir(), *(os.path.dirname(f) for f in secrets_files())]))


def _label(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path == home or path.startswith(home + "/") else path


def size(path: str, limit_s: float = SIZE_S) -> tuple[int, int, bool]:
    """(bytes, files, at_least): a walk that does not follow links and stops after `limit_s`."""
    try:
        st = os.lstat(path)
    except OSError:
        return 0, 0, False
    if not os.path.isdir(path) or os.path.islink(path):
        return st.st_size, 1, False
    total, n, t_end = 0, 0, time.monotonic() + limit_s
    for d, dirs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += os.lstat(os.path.join(d, f)).st_size
                n += 1
            except OSError:
                pass
        if time.monotonic() > t_end:
            return total, n, True
    return total, n, False


# the arguments and environment variables that name weights an engine reads
WEIGHT_ARGS = ("--nvfp4", "--fp8-head", "--exl3", "--draft-exl3", "--draft-head")
WEIGHT_ENV = ("QWEN38_EXL3", "QWEN38_NVFP4", "QWEN38_FP8_HEAD", "QSE_WEIGHTS_DIR")
WEIGHT_ENV_PREFIX = ("QWEN38_DRAFT_",)
ACTIVE_UNKNOWN = "the active weight profile is not known on this box, so no weights are removed"


def _weight_paths(argv: list[str], env: dict) -> list[str]:
    """The paths in one engine's argument vector and environment that name weights it reads."""
    vals = []
    for i, a in enumerate(argv):
        for flag in WEIGHT_ARGS:
            if a == flag and i + 1 < len(argv):
                vals.append(argv[i + 1])
            elif a.startswith(flag + "="):
                vals.append(a[len(flag) + 1:])
    for k, v in env.items():
        if k in WEIGHT_ENV or k.startswith(WEIGHT_ENV_PREFIX):
            vals.append(v)
    out = []
    for v in vals:
        for part in str(v).split(","):
            part = os.path.expanduser(part.strip())
            if part.startswith("/"):
                out.append(part)
    return out


def engine_pids(ops: str | None = None) -> list[int]:
    """Every engine process on any port, as ops/engines.sh decides (ENGINE_PROC for tests)."""
    lib = os.path.join(ops or device_mod.ops_dir(), "engines.sh")
    if not os.path.isfile(lib):
        lib = os.path.join(registry.ROOT, "ops", "engines.sh")
    try:
        r = subprocess.run(["bash", "-c", '. "$1"; engine_pids', "-", lib], capture_output=True,
                           text=True, timeout=10,
                           env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                                **({"ENGINE_PROC": os.environ["ENGINE_PROC"]}
                                   if os.environ.get("ENGINE_PROC") else {})})
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in r.stdout.split() if x.isdigit()]


def in_use() -> list[str]:
    """The weight paths the engines on this box read now: every engine process's (its
    /proc/<pid>/cmdline and environ) and this server's own."""
    out = _weight_paths(sys.argv, dict(os.environ))
    root = os.environ.get("ENGINE_PROC") or "/proc"
    for pid in engine_pids():
        try:
            with open(os.path.join(root, str(pid), "cmdline"), "rb") as fh:
                argv = [a.decode(errors="replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            argv = []
        env = {}
        try:
            with open(os.path.join(root, str(pid), "environ"), "rb") as fh:
                for kv in fh.read().split(b"\0"):
                    k, _, v = kv.decode(errors="replace").partition("=")
                    if k:
                        env[k] = v
        except OSError:
            pass
        out += _weight_paths(argv, env)
    if os.path.isdir(os.path.expanduser("~/tandem-staircut")):
        # the rollback tree (0.4.8) serves NVFP4 from there: it must still start
        out.append(os.path.expanduser("~/nvfp4-bf16"))
    return out


def protected(m: dict | None, repo: str | None = None, level: int = 2) -> list[str]:
    """Real paths a level never removes (a path under one of them, or one of them under it): the
    checkout, the home and state folders, secrets.env, the roles' models, the HF token and the
    assets; below level 5 also the ledger, the conductor's data and the memory notes."""
    keep = [apply.ROOT, os.path.expanduser("~"), *state_dirs(), *secrets_files(),
            os.path.join(state_dir(), "secrets.env"), os.path.join(state_dir(), "models"),
            os.path.join(state_dir(), "deciders"),
            os.path.expanduser(os.environ.get("HF_TOKEN_PATH") or "~/.cache/huggingface/token")]
    if level != 5:
        keep += [os.path.join(state_dir(), x) for x in ("usage", "conductor", "memory")]
        # where serve.env puts them: an install's ledger is DIR/state/usage/ledger.sqlite3
        got = _env(["QSE_USAGE_LEDGER", "QSE_CONDUCTOR_CONFIG"])
        if got.get("QSE_USAGE_LEDGER") not in (None, "", "off"):
            keep.append(os.path.dirname(os.path.expanduser(got["QSE_USAGE_LEDGER"])))
        if got.get("QSE_CONDUCTOR_CONFIG") not in (None, "", "off"):
            keep.append(os.path.expanduser(got["QSE_CONDUCTOR_CONFIG"]))
    if level == 4:
        keep += in_use()
    out = [os.path.realpath(p) for p in keep]
    # a secrets file that is a symbolic link: the link itself, as _allowed names an item that is one
    out += [os.path.join(os.path.realpath(os.path.dirname(f)), os.path.basename(f))
            for f in secrets_files()]
    for a in (m or {}).get("assets") or []:
        for p in [a["path"], *a.get("also", [])]:
            for hit in glob.glob(registry.expand(p, repo)) or [registry.expand(p, repo)]:
                out.append(os.path.realpath(hit if a["protected"] else os.path.dirname(hit)))
    return out


def _allowed(path: str, guard: list[str], level: int) -> str | None:
    """None when `path` may go at `level`; else why not."""
    home = os.path.realpath(os.path.expanduser("~"))
    real = os.path.realpath(path) if not os.path.islink(path) else \
        os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))
    if not real.startswith(home + os.sep):
        return "outside the home folder"
    roots = [home, *state_dirs()]
    for g in guard:
        if real == g or g.startswith(real + os.sep):
            return "it holds something that is never removed"
        if real.startswith(g + os.sep) and g not in roots:
            return "it is inside something that is never removed"
    return None


def _item(path: str, label: str, *, kind: str = "path", action: str = "remove",
          note: str | None = None, restart: bool = False, clear: str | None = None) -> dict:
    """`action`: remove (the files go), engine (the running engine clears it with its own
    function, `clear` names it for clear_live; a memory item's path is that name), skip (in use
    and nothing can clear it now)."""
    b, n, at_least = size(path) if kind == "path" else (0, 0, False)
    return {"path": _label(path) if kind == "path" else path, "label": label, "bytes": b,
            "files": n, "at_least": at_least, "exists": kind != "path" or os.path.lexists(path),
            "real": path if kind == "path" else None, "kind": kind, "action": action,
            "note": note, "restart": restart, "clear": clear or (path if kind == "memory" else None)}


def _env(keys: list[str]) -> dict:
    return apply.resolve_env(keys)


def items(level: int, app=None) -> list[dict]:
    """The items of `level` (each {path, label, bytes, files, at_least, exists, kind, note}); a
    `memory` item is a cache in RAM, `clear` one the running engine clears itself."""
    st = getattr(app, "STATE", {}) if app is not None else {}
    models, _ = registry.load_all()
    m = next(iter(models.values()), None)
    out = []
    if level == 2:
        for p in sorted(glob.glob(apply.overlay_path() + "*")):
            if p == apply.overlay_path() or ".bak-" in p:
                out.append(_item(p, "The setup guide's configuration" if p == apply.overlay_path()
                                 else "An earlier one, kept as a backup"))
        return out
    if level == 3:
        tier = st.get("disk_cache")
        live = os.path.realpath(getattr(tier, "root", "") or "") if tier is not None else None
        for p in sorted(glob.glob(os.path.join(state_dir(), "cache", "disk*"))):
            running = live is not None and os.path.realpath(p) == live
            out.append(_item(p, "Disk cache and parked chats" + (" (the running profile's)"
                                                                 if running else ""),
                             action="engine" if running else "remove",
                             note="the engine clears it itself" if running else None,
                             clear="disk" if running else None))
        shadow = os.path.join(state_dir(), "cache", "shadow")
        if os.path.lexists(shadow):
            on = st.get("shadow") is not None
            out.append(_item(shadow, "The shadow reader's saved sessions",
                             action="skip" if on else "remove",
                             note="kept while the shadow reader runs" if on else None))
        suffix = _env(["QWEN38_SUFFIX_STORE"]).get("QWEN38_SUFFIX_STORE") or \
            os.path.join(state_dir(), "suffix")
        if suffix and suffix != "off" and os.path.lexists(os.path.expanduser(suffix)):
            # a running store appends to its log after every request: removing the folder under
            # it made every later request fail, so the engine empties it under its own lock
            on = st.get("suffix_store") is not None
            out.append(_item(os.path.expanduser(suffix), "The lookup store",
                             action="engine" if on else "remove",
                             note="the engine empties it itself" if on else None,
                             clear="suffix_store" if on else None))
        for name, label in (("state_store", "The state store (memory)"),
                            ("response_cache", "The response cache (memory)"),
                            ("resident", "The conversation cache (memory)")):
            if st.get(name) is not None:
                out.append(_item(name, label, kind="memory", action="engine"))
        return out
    if level == 4 and m is not None:
        active = device_mod.active_profile()
        dirs = device_mod.profile_dirs([p["id"] for p in m["profiles"]])
        hub = registry.hf_hub()
        for p in m["profiles"]:
            if p["id"] == active:
                continue
            d = (dirs.get(p["id"]) or {}).get("dir")
            if d and os.path.lexists(d):
                out.append(_item(d, f"Weights of {p['id']}"))
            c = os.path.join(hub, "models--" + p["hf_repo"].replace("/", "--"))
            if os.path.lexists(c):
                out.append(_item(c, f"Hugging Face cache copy of {p['id']}"))
        return out
    if level == 5:
        got = _env(["QSE_USAGE_LEDGER", "QSE_CONDUCTOR_CONFIG"])
        ledger = got.get("QSE_USAGE_LEDGER") or os.path.join(state_dir(), "usage", "ledger.sqlite3")
        cond = got.get("QSE_CONDUCTOR_CONFIG") or os.path.join(state_dir(), "conductor")
        # the ledger and its SQLite sidecars only: its folder may hold other things
        led = [] if ledger in ("", "off") else [os.path.expanduser(ledger) + x for x in
                                                ("", "-wal", "-shm", "-journal")]
        for p, label in ([(x, "The usage history") for x in led] +
                         [(os.path.expanduser(cond), "The conductor's settings and journal"),
                          (os.path.join(state_dir(), "memory"), "Memory notes and transcripts")]):
            if p and p != "off" and os.path.lexists(p):
                out.append(_item(p, label))
        return out
    return out


def level(n: int, app=None) -> dict:
    """One level as SetupResetLevel."""
    its = items(n, app)
    models, _ = registry.load_all()
    m = next(iter(models.values()), None)
    guard = protected(m, None, n)
    shown = []
    unknown = n == 4 and device_mod.active_profile() is None
    for it in its:
        why = ACTIVE_UNKNOWN if unknown else (
            _allowed(it["real"], guard, n) if it["kind"] == "path" else None)
        shown.append({k: v for k, v in dict(it, refused=why).items() if k not in ("real", "clear")})
    label, what = LABELS[n]
    if n == 2 and m is not None:
        rec = registry.recommended_profile(m)["id"]
        what += f" The recommended profile is {rec}."
    return {"level": n, "label": label, "what": what, "items": shown,
            "bytes": sum(i["bytes"] for i in shown if not i.get("refused")),
            "restart": RESTART[n] or any(i.get("restart") for i in shown),
            "confirm_word": WORDS[n]}


def levels(app=None, busy: bool = False) -> dict:
    return {"contract_version": "1.0", "levels": [level(n, app) for n in (2, 3, 4, 5)],
            "busy": bool(busy)}


def audit(rec: dict) -> None:
    os.makedirs(apply.run_dir(), exist_ok=True)
    with open(os.path.join(apply.run_dir(), "setup-audit.log"), "a") as fh:
        fh.write(json.dumps(dict(rec, ts=round(time.time(), 3))) + "\n")


def _remove(path: str) -> None:
    if os.path.islink(path) or not os.path.isdir(path):
        os.unlink(path)
    else:
        shutil.rmtree(path)


def execute(n: int, app=None, clear_live=None) -> dict:
    """Remove level `n` (3 to 5; level 2 is a job). `clear_live(name) -> bytes` clears a cache the
    running engine holds (the disk tier, a RAM cache) through the engine's own function."""
    lv = level(n, app)
    removed, failed = [], []
    real = {(_label(i["real"]) if i["kind"] == "path" else i["path"]): i
            for i in items(n, app)}
    for it in lv["items"]:
        src = real.get(it["path"])
        if it.get("refused"):
            failed.append({"path": it["path"], "reason": it["refused"]})
            continue
        if it["action"] == "engine":
            if clear_live is None:
                failed.append({"path": it["path"], "reason": "no running engine to clear it"})
                continue
            try:
                freed = int(clear_live(src["clear"]) or 0)
                removed.append({"path": it["path"], "bytes": freed or it["bytes"]})
            except Exception as exc:                              # noqa: BLE001
                failed.append({"path": it["path"], "reason": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        if it["action"] == "skip":
            failed.append({"path": it["path"], "reason": "the shadow reader runs; reset again in "
                                                         "setup mode or after a restart"})
            continue
        try:
            _remove(src["real"])
            removed.append({"path": it["path"], "bytes": it["bytes"]})
        except OSError as exc:
            failed.append({"path": it["path"], "reason": f"{type(exc).__name__}: {exc}"[:200]})
    freed = sum(r["bytes"] for r in removed)
    msg = (f"Removed {len(removed)} of {len(lv['items'])} items, {freed / 2**30:.2f} GiB."
           + (f" {len(failed)} could not be removed: see `failed`." if failed else "")
           + (" Restart the engine to finish." if lv["restart"] else ""))
    return {"contract_version": "1.0", "level": n, "removed": removed, "failed": failed,
            "freed_gb": round(freed / 2**30, 2), "restart_needed": lv["restart"],
            "message": msg, "ok": not failed}
