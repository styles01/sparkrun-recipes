"""Removing a model's weights safely (SRV-560, QA SRV-561): the installed weights with their sizes, the
removal of the chosen ones, never the running profile's, and a line that downloads them again.

    GET  /v1/setup/weights                         installed(): every profile's weights on the box
    POST /v1/setup/remove {profiles, confirm, dry_run?}   remove(): the chosen folders, with a log
    POST /v1/setup/apply {selection, remove_after}  the old weights go after a switch that succeeded

A weight folder is removed only when it is the profile's own QSE_WEIGHTS_DIR (as serve.env resolves it
for that profile), a real folder (not a link) under HOME or the Hugging Face cache, not HOME itself, not
the checkout or one of its parents, not shared with a profile that stays, and holding no folder another
profile or asset reads. In the Hugging Face cache the repository's whole `models--*` folder goes (a
snapshot's files are links into its blobs), and only when nothing else reads from it. Assets (the
drafters, the draft head, the base checkpoint) are shared and listed apart; this never removes them.
Each removal appends to run/weights-removed.txt: the time, the profile, its repository at its revision,
the folder, the GB and the command that downloads it again.
"""

from __future__ import annotations

import os
import shutil
import time

from server.setup import device as device_mod
from server.setup import registry

LOG_NAME = "weights-removed.txt"


class RemoveError(ValueError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def size_gb(path: str) -> float:
    """The bytes under `path`, each inode once, links not followed, in GB."""
    seen, total = set(), 0
    for d, dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(d, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            total += st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
    return round(total / 1e9, 3)


def _hf_repo_dir(path: str) -> str | None:
    """The `models--*` folder of the Hugging Face cache that `path` is in, else None."""
    hub = os.path.realpath(registry.hf_hub())
    p = os.path.realpath(path)
    if not p.startswith(hub + os.sep):
        return None
    first = p[len(hub) + 1:].split(os.sep)[0]
    return os.path.join(hub, first) if first.startswith("models--") else None


def _profiles(models: dict) -> list[dict]:
    """Every profile with weights on the box: model, profile, folder, the folder that goes, size."""
    out = []
    for m in models.values():
        disk = device_mod.on_disk(m, fresh=True)
        for pid, d in disk.items():
            if not d.get("weights") or not d.get("dir"):
                continue
            prof = registry.profile(m, pid) or {"id": pid}
            hf = _hf_repo_dir(d["dir"])
            out.append({"model": m["id"], "model_name": m.get("name") or m["id"], "profile": pid,
                        "label": prof.get("label") or pid, "dir": d["dir"], "removes": hf or d["dir"],
                        "hf_repo": prof.get("hf_repo"), "hf_revision": prof.get("hf_revision"),
                        "repo": d.get("repo")})
    return out


def _assets(models: dict) -> list[dict]:
    out = []
    for m in models.values():
        repo = next((v.get("repo") for v in device_mod.on_disk(m).values() if v.get("repo")), None)
        for a in m.get("assets") or []:
            if not registry.asset_present(a, repo, m):
                continue
            p = registry.expand(a["path"], repo)
            used = [p_["id"] for p_ in m["profiles"] if a["id"] in p_.get("assets_required", [])]
            out.append({"id": a["id"], "label": a["label"], "model": m["id"], "path": p,
                        "gb": size_gb(p) if os.path.isdir(p) else round(os.path.getsize(p) / 1e9, 2)
                        if os.path.isfile(p) else 0.0, "used_by": used})
    return out


def installed(models: dict | None = None, *, setup_mode: bool = False) -> dict:
    """GET /v1/setup/weights: the weights on the box, the running profile marked, each folder's
    size and who shares it; the assets apart (never removed here); the free disk."""
    models = models if models is not None else registry.load_all()[0]
    active = None if setup_mode else device_mod.active_profile()
    profs = _profiles(models)
    by_folder: dict = {}
    for p in profs:
        by_folder.setdefault(os.path.realpath(p["removes"]), []).append(p["profile"])
    sizes: dict = {}
    for p in profs:
        key = os.path.realpath(p["removes"])
        if key not in sizes:
            sizes[key] = size_gb(p["removes"])
        p["gb"] = sizes[key]
        p["active"] = p["profile"] == active
        p["shared_with"] = [x for x in by_folder[key] if x != p["profile"]]
        p["removable"] = not p["active"] and not any(x == active for x in p["shared_with"])
    try:
        free = round(shutil.disk_usage(os.path.expanduser("~")).free / 1e9, 1)
    except OSError:
        free = None
    return {"weights": profs, "assets": _assets(models), "active": active, "free_gb": free}


def _safe(path: str, root: str, others: list[str]) -> str | None:
    """Why `path` may not be removed, else None."""
    home = os.path.realpath(os.path.expanduser("~"))
    hub = os.path.realpath(registry.hf_hub())
    if os.path.islink(path.rstrip(os.sep)):
        return f"{path} is a link; remove it by hand"
    p = os.path.realpath(path)
    if not os.path.isdir(p):
        return f"{path} is not a folder"
    if p in ("/", home) or os.path.realpath(root).startswith(p + os.sep) or p == os.path.realpath(root):
        return f"{path} holds this installation"
    if not (p.startswith(home + os.sep) or p.startswith(hub + os.sep)):
        return f"{path} is outside the home folder and the Hugging Face cache"
    for o in others:
        q = os.path.realpath(o)
        if q != p and (q.startswith(p + os.sep)):
            return f"{path} holds {o}, which something else reads"
    return None


def remove(profiles: list[str], *, setup_mode: bool, root: str, dry_run: bool = False,
           models: dict | None = None, clock=time.time) -> dict:
    """POST /v1/setup/remove: the chosen profiles' weight folders, or RemoveError (nothing removed)."""
    if not isinstance(profiles, list) or not profiles or not all(isinstance(p, str) for p in profiles):
        raise RemoveError(422, "invalid_request", "profiles: a list of profile ids")
    state = installed(models, setup_mode=setup_mode)
    by_id = {p["profile"]: p for p in state["weights"]}
    for pid in profiles:
        if pid not in by_id:
            raise RemoveError(404, "not_installed", f"{pid} has no weights on this box")
        if by_id[pid]["active"]:
            raise RemoveError(409, "active_profile", f"{pid} is the running setup: switch to another model "
                                                     f"first, or remove it after the switch")
        kept = [x for x in by_id[pid]["shared_with"] if x not in profiles]
        if kept:
            raise RemoveError(409, "shared", f"{pid}'s folder is also {', '.join(kept)}'s; remove them together")
    # what else on the box reads files (the kept profiles' folders, every asset)
    others = [p["dir"] for p in state["weights"] if p["profile"] not in profiles] + \
        [a["path"] for a in state["assets"]]
    plan = []
    for pid in profiles:
        p = by_id[pid]
        if any(x["removes"] == p["removes"] for x in plan):
            continue
        why = _safe(p["removes"], root, others)
        if why is None and p["removes"] != p["dir"]:
            # the Hugging Face cache: the repository's folder, only when nothing else reads from it
            inside = [o for o in others if os.path.realpath(o).startswith(os.path.realpath(p["removes"]) + os.sep)]
            if inside:
                why = f"{p['removes']} also holds {inside[0]}, which something else reads"
        if why:
            raise RemoveError(409, "unsafe", f"{pid} is not removed: {why}")
        plan.append(p)
    gone = [{"profile": p["profile"], "dir": p["removes"], "gb": p["gb"]} for p in plan]
    freed = round(sum(p["gb"] for p in plan), 3)
    if dry_run:
        return {"removed": gone, "freed_gb": freed, "dry_run": True}
    os.makedirs(os.path.join(root, "run"), exist_ok=True)
    when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clock()))
    with open(os.path.join(root, "run", LOG_NAME), "a") as fh:
        for p in plan:
            repo = f"{p['hf_repo']}@{p['hf_revision'] or 'main'}" if p.get("hf_repo") else "(no repository)"
            fh.write(f"{when}\t{p['profile']}\t{repo}\t{p['removes']}\t{p['gb']:g} GB\t"
                     f"bash ops/use-profile.sh {p['profile']} --fetch-only\n")
            shutil.rmtree(p["removes"])
    device_mod.clear()
    return {"removed": gone, "freed_gb": freed, "log": os.path.join("run", LOG_NAME)}


def removable_text(state: dict, need_gb: float) -> str:
    """For the plan's disk refusal: the installed weights that could go, largest first."""
    can = sorted((p for p in state["weights"] if p["removable"]), key=lambda p: -p["gb"])
    seen, parts = set(), []
    for p in can:
        if p["removes"] in seen:
            continue
        seen.add(p["removes"])
        parts.append(f"{p['profile']} ({p['gb']:g} GB)")
    return ("Weights on this box that could be removed in the Setup Guide: " + ", ".join(parts) + ".") \
        if parts else ""
