"""Fetch a weight profile's target weights and its assets from Hugging Face and verify them (ops/use-profile.sh).

    python ops/fetch_weights.py [--revision R] <repo> <dir> [file ...]   # files: only these (plus checksum lists)
    python ops/fetch_weights.py --plan [--revision R] <repo> <dir> [file ...]   # every URL and size; no download
    python ops/fetch_weights.py --verify <dir>                            # checksums of what is there
    python ops/fetch_weights.py --assets <profile> --repo <dir> [--also "ID ..."] [--dry-run] [--extra-gb G] [--margin-gb G]

Downloads into <dir>.partial with the token in ~/.cache/huggingface/token (HF_TOKEN wins; none is needed,
every repository a profile reads is public), checks every file a checksum list names (crc32.txt: EXL3
repos; quality/sha256.txt: the NVFP4 repo) and every LFS file against the hub's own sha256 (written to
_hub_manifest.json before the download, the repo pinned to that commit), then renames the directory into
place, so <dir> exists only complete. An interrupted download resumes in the .partial folder. Exit 0 =
verified. --revision: a branch of the repo (turboderp's Flash-Next packs, one branch a size, OPS-97); a
file a sibling pack beside <dir> already holds with the same sha256 is hard-linked instead of downloaded
(the n-gram table of the 4.05, 5.05 and 6.05 packs is one file of 36 GiB).

--assets (OPS-95): the files the profile's registry entry (models/*/model.yaml, `assets_required`) needs
besides its weights, each asset's `fetch` list, downloaded when the asset is not on the box: into its
folder through a .partial-<asset> folder, or into the Hugging Face cache (`${HF_HUB}`, the base
checkpoint), where ~/.qwen38-spark-engine/fetch/<asset>.json says whether the last download finished.
The disk is checked first: what is still to download, --extra-gb (the profile's weights, fetched after)
and --margin-gb left over, per filesystem. --also (VIS-153): more assets of the same registry entry,
the models of the tools a selection turns on (the shadow reader's, the narrator's, Laya's), fetched the
same way; one that two tools share is fetched once. Exit 3: an asset has no download (or no such id);
4: not enough disk; 5: a download or a checksum failed.
"""
from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import os
import shutil
import sys
import zlib


def _token() -> str | None:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        return open(os.path.expanduser("~/.cache/huggingface/token")).read().strip() or None
    except OSError:
        return None


def _crc32(path: str) -> str:
    c = 0
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            c = zlib.crc32(b, c)
    return f"{c & 0xFFFFFFFF:08x}"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


MANIFEST = "_hub_manifest.json"


def _matches(name: str, patterns: list[str] | None) -> bool:
    return not patterns or any(fnmatch.fnmatch(name, p) for p in patterns)


def verify(d: str, only: list[str] | None = None, hub: dict | None = None) -> list[str]:
    """Every (file, checksum) the lists in `d` name and `d` holds (or, with `only`, names or globs as a
    fetch entry's `files`, must hold): the mismatches and the missing, empty when all match. `hub`: the
    manifest's files, for a folder that has no _hub_manifest.json of its own (a snapshot in the Hugging
    Face cache)."""
    bad, checked = [], 0
    mp = os.path.join(d, MANIFEST)
    if hub is None and os.path.isfile(mp):
        hub = json.load(open(mp))["files"]
    if hub:
        for name, m in sorted(hub.items()):
            if not m.get("sha256") or not _matches(name, only):
                continue
            f = os.path.join(d, name)
            if not os.path.isfile(f):
                bad.append(f"missing {name}")
                continue
            got = _sha256(f)
            checked += 1
            if got != m["sha256"]:
                bad.append(f"{name}: {got} != {m['sha256']}")
    for lst, fn in (("crc32.txt", _crc32), ("quality/sha256.txt", _sha256)):
        p = os.path.join(d, lst)
        if not os.path.isfile(p):
            continue
        for line in open(p):
            parts = line.split()
            if len(parts) < 2:
                continue
            want, name = parts[0].lower(), parts[-1].lstrip("*")
            if not _matches(name, only):
                continue
            f = os.path.join(d, name)
            if not os.path.isfile(f):
                if only or name.endswith(".safetensors"):
                    bad.append(f"missing {name}")
                continue
            got = fn(f)
            checked += 1
            if got != want:
                if name.endswith(".safetensors"):
                    bad.append(f"{name}: {got} != {want}")
                else:
                    # the EXL3 repos' crc32.txt predates their template/config edits (the served 4.5 has
                    # the same three "mismatches"): a weight file must match, a text file is reported
                    print(f"[fetch] note: {name} differs from {lst} ({got} != {want}); not a weight file")
    if checked == 0:
        bad.append("no checksum list matched any file")
    print(f"[fetch] {checked} files checked in {d}: {'OK' if not bad else 'FAILED'}", flush=True)
    return bad


def hub_files(repo: str, revision: str | None = None) -> dict:
    """The hub's sizes and LFS sha256 of `revision` (None: the default branch): {repo, revision,
    commit, files}."""
    from huggingface_hub import HfApi
    info = HfApi(token=_token()).repo_info(repo, revision=revision, files_metadata=True)
    files = {s.rfilename: {"size": s.size, "sha256": s.lfs.sha256 if s.lfs else None} for s in info.siblings}
    return {"repo": repo, "revision": revision, "commit": info.sha, "files": files}


def manifest(repo: str, revision: str | None, part: str) -> str:
    """hub_files of `revision`, written to `part`/_hub_manifest.json; its commit."""
    m = hub_files(repo, revision)
    os.makedirs(part, exist_ok=True)
    with open(os.path.join(part, MANIFEST), "w") as fh:
        json.dump(m, fh, indent=1)
    return m["commit"]


def _merge(src: str, dst: str) -> None:
    """Every file of `src` moved into `dst`, its folders kept (a file of the same name is replaced,
    nothing else in `dst` is touched), then `src` removed."""
    for root, _dirs, names in os.walk(src):
        to = os.path.normpath(os.path.join(dst, os.path.relpath(root, src)))
        os.makedirs(to, exist_ok=True)
        for n in names:
            os.replace(os.path.join(root, n), os.path.join(to, n))
    shutil.rmtree(src)


def share(part: str, d: str) -> list[str]:
    """The files of `part`'s manifest that a sibling of `d` holds with the same sha256 and size (its own
    manifest says so), hard-linked into `part`; their names. A link that fails is downloaded."""
    mine = json.load(open(os.path.join(part, MANIFEST)))["files"]
    skip = {os.path.realpath(d), os.path.realpath(part)}
    out = []
    for sib in sorted(glob.glob(os.path.join(os.path.dirname(d.rstrip("/")), "*", MANIFEST))):
        sd = os.path.dirname(sib)
        if os.path.realpath(sd) in skip:
            continue
        theirs = json.load(open(sib))["files"]
        for name, m in mine.items():
            t, f, dst = theirs.get(name) or {}, os.path.join(sd, name), os.path.join(part, name)
            if name in out or not m.get("sha256") or t.get("sha256") != m["sha256"]:
                continue
            if not os.path.isfile(f) or os.path.getsize(f) != m["size"]:
                continue
            try:
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.link(f, dst)
            except OSError as exc:
                print(f"[fetch] {name}: no link from {sd} ({exc}); downloading it", flush=True)
                continue
            print(f"[fetch] {name}: the same file as {sd}'s, linked ({m['size'] / 1e9:.1f} GB not downloaded)",
                  flush=True)
            out.append(name)
    return out


def fetch(repo: str, d: str, files: list[str], revision: str | None = None, part: str | None = None) -> int:
    from huggingface_hub import snapshot_download
    part = part or d.rstrip("/") + ".partial"
    allow = (files + ["crc32.txt", "quality/sha256.txt", "README.md", "LICENSE"]) if files else None
    commit = manifest(repo, revision, part)
    linked = share(part, d) if revision else []
    print(f"[fetch] {repo}@{revision or 'main'} ({commit}) -> {part}"
          + (f" ({' '.join(files)})" if files else ""), flush=True)
    snapshot_download(repo_id=repo, revision=commit, local_dir=part, token=_token(), allow_patterns=allow,
                      ignore_patterns=linked or None)
    bad = verify(part, files or None)
    if bad:
        print("[fetch] checksum failures, the download stays in " + part + ":\n  " + "\n  ".join(bad))
        return 1
    shutil.rmtree(os.path.join(part, ".cache"), ignore_errors=True)
    if os.path.exists(d):
        # a partial earlier set (NVFP4 profile with some files present), or a folder other assets
        # share (~/.qwen38-spark-engine/weights): merge, never delete
        _merge(part, d)
    else:
        os.makedirs(os.path.dirname(d.rstrip("/")) or ".", exist_ok=True)
        os.rename(part, d)
    print(f"[fetch] {d} complete", flush=True)
    return 0


def plan(repo: str, d: str, files: list[str], revision: str | None = None, quiet: bool = False,
         m: dict | None = None) -> tuple[dict, int]:
    """What a fetch of `files` (names or globs; none: the whole repo) into `d` would download, each
    file's URL, size and sha256 printed; (hub_files, bytes still to download). A file already in `d`,
    in its .partial folder or (d = the HF cache) in the snapshot of that commit with the hub's size
    counts as there."""
    m = m or hub_files(repo, revision)
    cache = os.path.join(d, "models--" + repo.replace("/", "--"), "snapshots", m["commit"])
    where = [cache] if d == hf_hub() else [d, d.rstrip("/") + ".partial"] + sorted(glob.glob(d.rstrip("/") + ".partial-*"))
    todo = 0
    for name, f in sorted(m["files"].items()):
        if not _matches(name, files):
            continue
        have = any(os.path.isfile(os.path.join(w, name)) and os.path.getsize(os.path.join(w, name)) == f["size"]
                   for w in where)
        todo += 0 if have else f["size"]
        if not quiet:
            print(f"[fetch]   https://huggingface.co/{repo}/resolve/{m['commit']}/{name}  {f['size'] / 1e9:.3f} GB"
                  + (f"  sha256 {f['sha256'][:16]}" if f["sha256"] else "") + ("  (here)" if have else ""))
    return m, todo


def hf_hub() -> str:
    return os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")


def state_path(aid: str) -> str:
    return os.path.expanduser(f"~/.qwen38-spark-engine/fetch/{aid}.json")


def _write_state(aid: str, data: dict) -> None:
    p = state_path(aid)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(p + ".tmp", p)


def fetch_cache(aid: str, repo: str, files: list[str], m: dict) -> int:
    """`files` (globs) of `repo` at m's commit into the Hugging Face cache, where the engine reads the
    base snapshot, and checked against the hub's sha256. state_path(aid) says `complete: false` until
    they are, so an interrupted download is resumed, not taken for the asset."""
    from huggingface_hub import snapshot_download
    want = {n: f for n, f in m["files"].items() if _matches(n, files)}
    _write_state(aid, {"repo": repo, "commit": m["commit"], "files": files, "complete": False})
    print(f"[fetch] {repo}@{m['commit']} -> the Hugging Face cache ({' '.join(files)})", flush=True)
    snap = snapshot_download(repo_id=repo, revision=m["commit"], allow_patterns=files, token=_token())
    bad = verify(snap, hub=want)
    if bad:
        print("[fetch] checksum failures in " + snap + ":\n  " + "\n  ".join(bad))
        return 1
    _write_state(aid, {"repo": repo, "commit": m["commit"], "files": files, "complete": True, "snapshot": snap})
    print(f"[fetch] {snap} complete", flush=True)
    return 0


# ---------------------------------------------------------------- OPS-133: the tensors a load reads
PLAIN_MANIFEST = "_plain_manifest.json"
RANGE_TRIES = 6                 # a range request's attempts, each resumed where the last stopped


def is_plain(e: dict) -> bool:
    """A fetch entry that takes only some tensors of its safetensors files (`skip_dtypes`,
    `skip_tensors`; the EXL3 profiles' base: the FP8 projections an EXL3 pack replaces left out)."""
    return bool(e.get("skip_dtypes") or e.get("skip_tensors"))


def hub_url(repo: str, commit: str, name: str) -> str:
    return f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}/{repo}/resolve/{commit}/{name}"


def open_range(url: str, start: int, end: int):
    """bytes [start, end) of `url` (a public repository's file; the redirect to the hub's storage
    keeps the Range header): the response, refused unless it is a 206 of exactly that range."""
    import urllib.request
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}",
                                               "User-Agent": "tandemllm-fetch"})
    r = urllib.request.urlopen(req, timeout=60)
    if r.status != 206 or not (r.headers.get("Content-Range") or "").startswith(
            f"bytes {start}-{end - 1}/"):
        r.close()
        raise OSError(f"{url}: no range answer (HTTP {r.status}, {r.headers.get('Content-Range')})")
    return r


def copy_range(url: str, start: int, end: int, fh) -> None:
    """bytes [start, end) of `url` appended to `fh`, resumed where a dropped connection stopped."""
    import time
    pos = start
    tries = RANGE_TRIES
    for attempt in range(tries):
        try:
            with open_range(url, pos, end) as r:
                while pos < end:
                    b = r.read(min(1 << 23, end - pos))
                    if not b:
                        raise OSError(f"{url}: the answer ended at byte {pos} of {end}")
                    fh.write(b)
                    pos += len(b)
            return
        except OSError as exc:
            if attempt == tries - 1:
                raise
            print(f"[fetch] {url}: {exc}; again from byte {pos}", flush=True)
            time.sleep(min(30, 2 ** attempt))


def read_header(url: str) -> tuple[int, dict]:
    """A safetensors file's header over two range requests: (its length, the header)."""
    import io
    import struct
    b = io.BytesIO()
    copy_range(url, 0, 8, b)
    n = struct.unpack("<Q", b.getvalue())[0]
    if not 0 < n < 1 << 28:
        raise OSError(f"{url}: a header length of {n} bytes is not a safetensors file")
    b = io.BytesIO()
    copy_range(url, 8, 8 + n, b)
    return n, json.loads(b.getvalue())


def kept_tensors(header: dict, skip_dtypes, skip_tensors) -> dict:
    """name -> info of the tensors a plain fetch keeps: not of `skip_dtypes`, not named in
    `skip_tensors`, and not the `<name>_scale_inv` of a tensor left out (an FP8 block's scales)."""
    t = {k: v for k, v in header.items() if k != "__metadata__"}
    out = {k: v for k, v in t.items() if v["dtype"] not in set(skip_dtypes or ()) and
           k not in set(skip_tensors or ())}
    return {k: v for k, v in out.items()
            if not (k.endswith("_scale_inv") and k[: -len("_scale_inv")] in t and
                    k[: -len("_scale_inv")] not in out)}


def runs_of(kept: dict) -> list[list[int]]:
    """The kept tensors' byte ranges of the data section, merged where they touch."""
    out: list[list[int]] = []
    for a, b in sorted(tuple(v["data_offsets"]) for v in kept.values()):
        if out and out[-1][1] == a:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def write_plain(url: str, n: int, header: dict, kept: dict, path: str) -> int:
    """`path`: a safetensors file of the `kept` tensors of `url`, their bytes fetched run by run;
    its size."""
    runs = runs_of(kept)
    base, where = 0, {}
    for a, b in runs:
        where[(a, b)] = base
        base += b - a
    new = {}
    for k, v in sorted(kept.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        a, b = v["data_offsets"]
        (ra, rb), off = next(((r[0], r[1]), where[tuple(r)]) for r in runs if r[0] <= a and b <= r[1])
        new[k] = {"dtype": v["dtype"], "shape": v["shape"],
                  "data_offsets": [off + a - ra, off + b - ra]}
    if "__metadata__" in header:
        new["__metadata__"] = header["__metadata__"]
    raw = json.dumps(new, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    import struct
    with open(path + ".tmp", "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw)
        for a, b in runs:
            copy_range(url, 8 + n + a, 8 + n + b, fh)
    if os.path.getsize(path + ".tmp") != 8 + len(raw) + base:
        raise OSError(f"{path}: {os.path.getsize(path + '.tmp')} bytes, not {8 + len(raw) + base}")
    os.replace(path + ".tmp", path)
    return 8 + len(raw) + base


def fetch_plain(aid: str, repo: str, to: str, e: dict, m: dict) -> int:
    """OPS-133: `e`'s files of `repo` at m's commit into `to`, the safetensors files with only the
    tensors `kept_tensors` keeps (range requests on each file's header and the kept bytes), the
    others whole; the index's weight map cut to the tensors kept. Each file's header must add up to
    the hub's size for it. Built in <to>.partial-<aid> (a stopped fetch resumes file by file), then
    moved into place, so `to` exists only complete."""
    from huggingface_hub import snapshot_download
    part = to.rstrip("/") + ".partial-" + aid
    os.makedirs(part, exist_ok=True)
    mp = os.path.join(part, PLAIN_MANIFEST)
    man = json.load(open(mp)) if os.path.isfile(mp) else {}
    if man.get("commit") != m["commit"]:
        man = {"repo": repo, "commit": m["commit"], "skip_dtypes": e.get("skip_dtypes") or [],
               "skip_tensors": e.get("skip_tensors") or [], "files": {}}
    names = sorted(n for n in m["files"] if _matches(n, e["files"]))
    whole = [n for n in names if not n.endswith(".safetensors")]
    print(f"[fetch] {repo}@{m['commit']} -> {part}: {len(whole)} files whole, "
          f"{len(names) - len(whole)} safetensors files without "
          f"{', '.join((e.get('skip_dtypes') or []) + (e.get('skip_tensors') or []))}", flush=True)
    if whole:
        snapshot_download(repo_id=repo, revision=m["commit"], local_dir=part, allow_patterns=whole,
                          token=_token())
        short = [n for n in whole if not os.path.isfile(os.path.join(part, n)) or
                 os.path.getsize(os.path.join(part, n)) != m["files"][n]["size"]]
        if short:
            print(f"[fetch] not the hub's size: {' '.join(short)}")
            return 1
    kept_names: dict[str, str] = {}
    for name in names:
        if name in whole:
            continue
        url = hub_url(repo, m["commit"], name)
        out = os.path.join(part, name)
        rec = man["files"].get(name)
        try:
            n, header = read_header(url)
            data = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"),
                       default=0)
            if 8 + n + data != m["files"][name]["size"]:
                print(f"[fetch] {name}: its header says {8 + n + data} bytes, the hub "
                      f"{m['files'][name]['size']}")
                return 1
            kept = kept_tensors(header, e.get("skip_dtypes"), e.get("skip_tensors"))
            if not (rec and os.path.isfile(out) and os.path.getsize(out) == rec["size"]):
                size = write_plain(url, n, header, kept, out)
                rec = {"hub_size": m["files"][name]["size"], "hub_sha256": m["files"][name]["sha256"],
                       "size": size, "sha256": _sha256(out), "tensors": len(kept)}
                man["files"][name] = rec
                _write_json(mp, man)
        except OSError as exc:
            print(f"[fetch] {name}: {exc}")
            return 1
        kept_names.update({k: name for k in kept})
    idx = os.path.join(part, "model.safetensors.index.json")
    if os.path.isfile(idx):
        doc = json.load(open(idx))
        doc["weight_map"] = {k: v for k, v in doc.get("weight_map", {}).items() if k in kept_names}
        doc.setdefault("metadata", {})["total_size"] = sum(
            r["size"] for r in man["files"].values())
        _write_json(idx, doc)
    shutil.rmtree(os.path.join(part, ".cache"), ignore_errors=True)
    total = sum(r["size"] for r in man["files"].values())
    if os.path.exists(to):
        _merge(part, to)
    else:
        os.makedirs(os.path.dirname(to.rstrip("/")) or ".", exist_ok=True)
        os.rename(part, to)
    print(f"[fetch] {to} complete: {len(kept_names)} tensors, {total / 1e9:.2f} GB", flush=True)
    return 0


def _write_json(path: str, data: dict) -> None:
    with open(path + ".tmp", "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(path + ".tmp", path)


def _free_at(path: str) -> tuple[int, int]:
    """(device, free bytes) of the filesystem `path` is or would be on."""
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return os.stat(path).st_dev, shutil.disk_usage(path).free


def assets(profile: str, repo: str, dry_run: bool = False, extra_gb: float = 0.0, margin_gb: float = 60.0,
           also: list[str] | tuple = ()) -> int:
    """OPS-95: the profile's `assets_required` and the `also` ids that are not on this box, fetched (see
    the module doc)."""
    sys.path.insert(0, repo)
    try:
        from server.setup import registry
        models = registry.load_all(os.path.join(repo, "models"))[0]
    except ImportError as exc:
        print(f"[fetch] no model registry under {repo} ({exc}): the profile's files are checked as they are")
        return 0
    m = next((x for x in models.values() if registry.profile(x, profile)), None)
    if m is None:
        print(f"[fetch] no model registry entry lists {profile}: its files are checked as they are")
        return 0
    expand = lambda p: registry.expand(p, repo)  # noqa: E731  (${HF_HUB} is hf_hub() there too)
    ids = list(dict.fromkeys([*registry.profile(m, profile)["assets_required"], *also]))
    unknown = [aid for aid in ids if registry.asset(m, aid) is None]
    if unknown:
        print(f"[fetch] {m['id']} has no asset {' '.join(unknown)}")
        return 3
    def state(aid: str) -> dict:
        return json.load(open(state_path(aid))) if os.path.isfile(state_path(aid)) else {}

    def here(aid: str) -> bool:
        """on the box and its last download finished (the HF cache's base), or (OPS-133) an asset
        it names in `satisfied_by` is"""
        a = registry.asset(m, aid)
        if registry.asset_present(a, repo) and state(aid).get("complete", True):
            return True
        return any(registry.asset(m, b) is not None and here(b) for b in a.get("satisfied_by") or [])

    todo = []                       # (asset id, entry, its `to` expanded)
    for aid in ids:
        a = registry.asset(m, aid)
        st = state(aid)
        if here(aid):
            print(f"[fetch] {aid}: on this box")
            continue
        if not a.get("fetch"):
            print(f"[fetch] {aid}: {a['label']} is not on this box ({expand(a['path'])}) and has no download")
            return 3
        for e in a["fetch"]:
            to = expand(e["to"])
            done = (st.get("complete") and registry.asset_present(a, repo)) if to == hf_hub() else \
                all(glob.glob(os.path.join(to, f)) for f in e["files"])
            if not done:
                todo.append((aid, e, to))
    if not todo:
        return 0
    need: dict[int, int] = {}
    free: dict[int, int] = {}
    plans = []
    for aid, e, to in todo:
        print(f"[fetch] {aid}: {e['repo']} ({' '.join(e['files'])}) -> {to}")
        hm, left = plan(e["repo"], to, e["files"], revision=e.get("revision"))
        if is_plain(e):
            # OPS-133: only the tensors kept; the plan's sizes are the whole files
            left = int(e["gb"] * 1e9)
            print(f"[fetch] {aid}: the safetensors files without "
                  f"{', '.join((e.get('skip_dtypes') or []) + (e.get('skip_tensors') or []))}, "
                  f"about {e['gb']:g} GB")
        plans.append(hm)
        dev, fr = _free_at(to)
        need[dev] = need.get(dev, 0) + left
        free[dev] = fr
    dev, fr = _free_at(os.path.expanduser("~"))
    mine = dict(need)
    need[dev] = need.get(dev, 0) + int(extra_gb * 1e9)
    free.setdefault(dev, fr)
    short = [d for d in need if free[d] < need[d] + margin_gb * 1e9]
    for d in need:
        print(f"[fetch] disk: {mine.get(d, 0) / 1e9:.1f} GB of assets to download"
              + (f", then {extra_gb:g} GB of weights" if d == dev and extra_gb else "")
              + f"; {free[d] / 1e9:.1f} GB free, need {(need[d] + margin_gb * 1e9) / 1e9:.1f} GB (margin {margin_gb:g})")
    if short:
        print(f"[fetch] {'dry run: would refuse' if dry_run else 'REFUSING'} the download: not enough disk")
        if not dry_run:
            return 4
    if dry_run:
        print("[fetch] dry run: nothing downloaded")
        return 0
    for (aid, e, to), hm in zip(todo, plans):
        if is_plain(e):
            rc = fetch_plain(aid, e["repo"], to, e, hm)
        elif to == hf_hub():
            rc = fetch_cache(aid, e["repo"], e["files"], hm)
        else:
            rc = fetch(e["repo"], to, list(e["files"]), revision=e.get("revision"),
                       part=to.rstrip("/") + ".partial-" + aid)
        if rc != 0:
            return 5
    missing = [aid for aid in ids if not here(aid)]
    if missing:
        print(f"[fetch] still not on this box after the download: {' '.join(missing)}")
        return 5
    print(f"[fetch] {profile}: every asset on this box")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--verify"] and len(a) == 2:
        sys.exit(1 if verify(a[1]) else 0)
    if a[:1] == ["--assets"]:
        import argparse
        ap = argparse.ArgumentParser(prog="fetch_weights.py --assets")
        ap.add_argument("--assets", required=True, metavar="PROFILE")
        ap.add_argument("--repo", required=True)
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--extra-gb", type=float, default=0.0)
        ap.add_argument("--margin-gb", type=float, default=60.0)
        ap.add_argument("--also", default="", help="more asset ids of the entry, space or comma separated")
        o = ap.parse_args(a)
        sys.exit(assets(o.assets, o.repo, o.dry_run, o.extra_gb, o.margin_gb,
                        [x for x in o.also.replace(",", " ").split() if x]))
    dry = a[:1] == ["--plan"]
    a = a[1:] if dry else a
    rev = None
    if a[:1] == ["--revision"] and len(a) >= 2:
        rev, a = a[1], a[2:]
    if len(a) < 2:
        sys.exit(__doc__)
    if dry:
        _m, left = plan(a[0], os.path.expanduser(a[1]), a[2:], rev)
        print(f"[fetch] {a[0]}@{rev or 'main'} ({_m['commit']}): {left / 1e9:.1f} GB to download into {a[1]}")
        sys.exit(0)
    sys.exit(fetch(a[0], os.path.expanduser(a[1]), a[2:], rev))
