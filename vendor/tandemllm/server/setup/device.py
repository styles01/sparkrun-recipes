"""The box, read only (SRV-323): GET /v1/setup/device and the facts the plan and Wren read.

Everything here reads what is already there -- /proc/meminfo, the dashboard's GPU sampler (one
`nvidia-smi` call every 5 s at most), the disk, the profile files, the environment -- and never
starts the engine, loads a model or allocates GPU memory. The answer is kept for a second, and the
profiles' weight folders (one `bash` that sources serve.env per profile) for 30 s, so the memory bar
can poll it every few seconds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

from server.setup import registry

GIB = float(1 << 30)
TTL_S = 1.0
DISK_TTL_S = 30.0

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "out": None, "disk_at": 0.0, "disk": {}}


def ops_dir() -> str:
    """The ops folder of the serve.env this server was started with (ops/start.sh exports it; an
    install.sh box has its own beside the generated serve.env), else the repository's."""
    return os.environ.get("QSE_OPS_DIR") or os.path.join(registry.ROOT, "ops")


_repo_cache: dict = {}


def repo_ops(ops: str | None = None, timeout: float = 10.0) -> str:
    """The ops folder that holds profiles/ and active-profile: serve.env's REPO/ops, as
    ops/use-profile.sh reads them. On an install.sh box the serve.env (and QSE_OPS_DIR) is in
    run/ops, beside the scripts, while the profiles are under the checkout serve.env names."""
    ops = ops or ops_dir()
    hit = _repo_cache.get(ops)
    if hit is not None and time.monotonic() - hit[0] < DISK_TTL_S:
        return hit[1]
    out = ops
    env_file = os.path.join(ops, "serve.env")
    if os.path.isfile(env_file):
        try:
            r = subprocess.run(["bash", "-c", 'set -a; . "$ENVF" >/dev/null 2>&1; '
                                              'printf "%s" "${REPO:-}"'],
                               capture_output=True, text=True, timeout=timeout,
                               env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                                    "HOME": os.path.expanduser("~"), "ENVF": env_file})
            repo = r.stdout.strip()
            if repo and os.path.isdir(os.path.join(repo, "ops", "profiles")):
                out = os.path.join(repo, "ops")
        except (OSError, subprocess.SubprocessError):
            pass
    _repo_cache[ops] = (time.monotonic(), out)
    return out


def meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    out[k] = int(v.split()[0]) * 1024 / GIB
    except (OSError, ValueError, IndexError):
        pass
    return out


def hf_token_present() -> bool:
    """A Hugging Face token is configured (the variable, or the file `hf auth login` writes). The
    token itself is never read into an answer."""
    if os.environ.get("HF_TOKEN"):
        return True
    path = os.environ.get("HF_TOKEN_PATH") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "token")
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def profile_dirs(names: list[str], ops: str | None = None, timeout: float = 10.0) -> dict:
    """profile -> {dir, files, repo}: QSE_WEIGHTS_DIR, QSE_WEIGHTS_FILES and REPO as serve.env
    resolves them for that profile (QSE_PROFILE_OVERRIDE, as ops/use-profile.sh --list reads
    them), one bash for all."""
    ops = ops or ops_dir()
    env_file = os.path.join(ops, "serve.env")
    if not names or not os.path.isfile(env_file):
        return {}
    script = ('for n in "$@"; do ( export QSE_PROFILE_OVERRIDE="$n"; set -a; . "$ENVF" >/dev/null '
              '2>&1; printf "%s\\t%s\\t%s\\t%s\\n" "$n" "${QSE_WEIGHTS_DIR:-}" "${QSE_WEIGHTS_FILES:-}" '
              '"${REPO:-}" ); '
              'done')
    try:
        r = subprocess.run(["bash", "-c", script, "-", *names], capture_output=True, text=True,
                           timeout=timeout, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                                                 "HOME": os.path.expanduser("~"), "ENVF": env_file})
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 4 and parts[0] in names:
            out[parts[0]] = {"dir": parts[1], "files": parts[2].split(), "repo": parts[3]}
    return out


def weights_present(d: dict | None) -> bool:
    """The profile's weights are complete in its folder, as ops/use-profile.sh `weights_ok` reads
    them (the listed files, else config.json and the index's shards, or model.safetensors alone)."""
    if not d or not d["dir"] or not os.path.isdir(d["dir"]):
        return False
    if d["files"]:
        paths = [os.path.join(d["dir"], f) for f in d["files"]]
        return all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in paths)
    idx = os.path.join(d["dir"], "model.safetensors.index.json")
    if not os.path.isfile(os.path.join(d["dir"], "config.json")):
        return False
    if not os.path.isfile(idx):
        # a pack in one file has no index (LFM2.5's 2.5 pack: model.safetensors alone)
        one = os.path.join(d["dir"], "model.safetensors")
        return os.path.isfile(one) and os.path.getsize(one) > 0
    import json
    try:
        with open(idx) as fh:
            shards = set(json.load(fh)["weight_map"].values())
    except (OSError, ValueError, KeyError):
        return False
    return all(os.path.isfile(os.path.join(d["dir"], s)) for s in shards)


def on_disk(m: dict, *, fresh: bool = False) -> dict:
    """profile -> {weights, dir, assets_missing}; kept DISK_TTL_S unless `fresh`."""
    now = time.monotonic()
    with _lock:
        if not fresh and now - _cache["disk_at"] < DISK_TTL_S and m["id"] in _cache["disk"]:
            return _cache["disk"][m["id"]]
    dirs = profile_dirs([p["id"] for p in m["profiles"]])
    out = {}
    for p in m["profiles"]:
        d = dirs.get(p["id"])
        repo = (d or {}).get("repo") or None          # ${REPO} of an asset: serve.env's
        missing = [a for a in p["assets_required"]
                   if not registry.asset_present(registry.asset(m, a), repo, m)]
        out[p["id"]] = {"weights": weights_present(d), "dir": (d or {}).get("dir") or None,
                        "assets_missing": missing, "repo": repo}
    # a community pack's profile is written at its first switch, so its folder is the one the
    # profile will name (community.pack_dir), not serve.env's; ${REPO} is the box's
    from server.setup import community as community_mod
    repo = next((v["repo"] for v in out.values() if v["repo"]), None)
    for c in registry.community_packs(m):
        d = {"dir": community_mod.pack_dir(c["id"]), "files": []}
        missing = [a for a in registry.community_assets(c, m)
                   if not registry.asset_present(registry.asset(m, a), repo, m)]
        out[c["profile"]] = {"weights": weights_present(d), "dir": d["dir"],
                             "assets_missing": missing, "repo": repo}
    with _lock:
        _cache["disk_at"] = now
        _cache["disk"][m["id"]] = out
    return out


def windowed(m: dict | None) -> dict | None:
    """ENG-363: the model with its window range up to what the checkpoint on this box holds
    (registry.with_config_window over the folders of its profiles with weights present); the card's
    range when none is here or the disk cannot be read."""
    if m is None:
        return None
    try:
        disk = on_disk(m)
    except Exception:                                              # noqa: BLE001
        return m
    return registry.with_config_window(m, [v["dir"] for v in disk.values() if v.get("weights") and v.get("dir")])


def active_profile() -> str | None:
    """The profile the running engine serves (QSE_WEIGHTS), else active-profile under serve.env's
    REPO (repo_ops: an install.sh box keeps it there, not beside its run/ops/serve.env)."""
    if os.environ.get("QSE_WEIGHTS"):
        return os.environ["QSE_WEIGHTS"]
    for d in dict.fromkeys((repo_ops(), ops_dir())):
        try:
            with open(os.path.join(d, "active-profile")) as fh:
                return fh.read().strip() or None
        except OSError:
            continue
    return None


def preset(m: dict, total_gib: float | None, gpu_name: str | None) -> str | None:
    """The model's device preset this box is (MemTotal within 2 GiB), else None: the numbers were
    not measured on this machine."""
    if total_gib is None:
        return None
    for d in m["devices"]:
        if abs(d["total_gib"] - total_gib) <= 2.0 and (gpu_name is None or "GB10" in gpu_name
                                                         or d["id"] != "dgx-spark"):
            return d["id"]
    return None


def read(state: dict, *, inflight: dict, busy: bool, assistant: dict, gpu=None,
         mem=None) -> dict:
    """The SetupDevice answer, kept TTL_S. `gpu`: the dashboard's sampler's sample; `mem`: a
    meminfo dict (tests)."""
    now = time.monotonic()
    with _lock:
        if _cache["out"] is not None and now - _cache["at"] < TTL_S:
            return dict(_cache["out"], assistant=assistant)
    mem = meminfo() if mem is None else mem
    g = gpu or {}
    gpu_block = None
    if g.get("name"):
        util = g.get("utilization")
        gpu_block = {"name": g.get("name"), "util_pct": round(util * 100, 1) if util is not None
                     else None, "temp_c": g.get("temperature_c"), "power_w": g.get("power_w")}
    try:
        du = shutil.disk_usage(os.path.expanduser("~"))
        disk = {"path": "~", "free_gb": round(du.free / 1e9, 1), "total_gb": round(du.total / 1e9, 1)}
    except OSError:
        disk = None
    models, _ = registry.load_all()
    present = {}
    pre = None
    for m in models.values():
        for pid, v in on_disk(m).items():
            present[pid] = v["weights"]
        pre = pre or preset(m, mem.get("MemTotal"), (gpu_block or {}).get("name"))
    auth = state.get("auth")
    pw = bool(getattr(auth, "password_set", False))
    started = state.get("started", time.time())
    out = {"contract_version": "1.0",
           "mode": "setup" if state.get("setup_mode") else "serving",
           "preset": pre,
           "mem_total_gb": round(mem["MemTotal"], 2) if "MemTotal" in mem else None,
           "mem_available_gb": round(mem["MemAvailable"], 2) if "MemAvailable" in mem else None,
           "gpu": gpu_block, "disk": disk,
           "engine": {"version": state.get("version", ""),
                      "profile": None if state.get("setup_mode") else active_profile(),
                      "uptime_s": round(time.time() - started, 1),
                      "running": int(inflight.get("running", 0)),
                      "waiting": int(inflight.get("waiting", 0)), "busy": bool(busy),
                      "mode": "setup" if state.get("setup_mode") else "serving"},
           "active_profile": active_profile(),
           "profiles_on_disk": present,
           "hf_token_present": hf_token_present(),
           "password_set": pw, "admin_password_set": pw}
    with _lock:
        _cache["out"], _cache["at"] = out, now
    return dict(out, assistant=assistant)


def clear() -> None:
    with _lock:
        _cache.update(at=0.0, out=None, disk_at=0.0, disk={})
        _repo_cache.clear()
