"""Community weight packs (docs/community-weights.md): the index, its sync from GitHub, the profile.

    models/<model>/community/<id>.json   one pack a file; the public repository's main is the source
    models/community-index.json          the bundle: every entry of those folders (`--index`), shipped
                                         with the release
    models/community-builtin.json        the entries one of the model's own profiles serves (`profile`,
                                         VIS-175: turboderp's packs of one model), written by `--index`
                                         beside it; read from the release only, never fetched, since
                                         the profile ships with the release (and a box before 0.11.1
                                         would drop such an entry from the fetched index)
    ~/.tandemllm/community/index.json    the index last fetched from QSE_COMMUNITY_URL, and beside it
                                         index.meta.json (its URL, ETag and when it came)
    ~/.tandemllm/community/<id>/         a pack's download (ops/use-profile.sh community/<id>)
    REPO/run/community/<id>.env          the weight profile ops/serve.env reads for community/<id>
                                         (`env`, written by use-profile.sh before every switch)

THE SYNC. Nothing fetches on a timer or at the start. The setup guide's community section asks for a
sync when it opens and on its Refresh button (POST /v1/setup/community/sync): one GET of
QSE_COMMUNITY_URL (default: the index on the public repository's main) with the cached ETag, so an
unchanged index is a 304, a 10 s timeout and a 2 MiB cap. Calls that arrive while one runs wait for
its answer instead of fetching again. `QSE_COMMUNITY_URL=off` never fetches. The entries come from,
in this order: the index fetched (`github`, from the cache file once it is written), the cache file
from an earlier fetch when this one failed (`cache`), the index shipped in models/ (`bundled`).

The fetched index is untrusted: an index whose outer shape is wrong is not written over the cache,
and every entry passes the registry's checks (registry.community_errors: the strict schema, the
pinned revision, https links); one that does not is dropped and listed with the reason.

    python -m server.setup.community --index [--check]       write (check) models/community-index.json
    python -m server.setup.community env community/<id> [--out FILE]   the profile, from the registry
    python -m server.setup.community sync                    fetch now; the sync state as JSON
    python -m server.setup.community check-pack community/<id> [--dir DIR]
                                                             the pack's files against its base (check_pack)
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from server.conductor.schema import validate
from server.setup import registry

DEFAULT_URL = "https://raw.githubusercontent.com/0xBakeer/TandemLLM/main/models/community-index.json"
REPO = "0xBakeer/TandemLLM"                 # where "Add yours" opens its issue form
GENERATOR = "server.setup.community 1"
INDEX_SCHEMA_VERSION = 1
TIMEOUT_S = 10.0
MAX_BYTES = 2 * 1024 * 1024
INDEX_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["schema", "generator", "entries"],
    "properties": {
        "schema": {"const": INDEX_SCHEMA_VERSION},
        "generator": {"type": "string", "maxLength": 80},
        "entries": {"type": "array", "maxItems": 2000, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["model", "id", "path", "sha256", "entry"],
            "properties": {"model": {"type": "string", "pattern": registry.MODEL_ID},
                           "id": {"type": "string", "maxLength": 80, "pattern": registry.COMMUNITY_SLUG},
                           "path": {"type": "string", "maxLength": 200,
                                    "pattern": r"^models/[a-z0-9.-]+/community/[a-z0-9.-]+\.json$"},
                           "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
                           "entry": {"type": "object"}}}}}}


def home() -> str:
    return os.environ.get("QSE_COMMUNITY_DIR") or os.path.expanduser("~/.tandemllm/community")


def cache_path() -> str:
    return os.path.join(home(), "index.json")


def meta_path() -> str:
    return os.path.join(home(), "index.meta.json")


def bundled_path(models_dir: str | None = None) -> str:
    return os.path.join(models_dir or registry.MODELS_DIR, "community-index.json")


def builtin_path(models_dir: str | None = None) -> str:
    return os.path.join(models_dir or registry.MODELS_DIR, "community-builtin.json")


def url() -> str | None:
    """QSE_COMMUNITY_URL, else the public repository's; None for `off`."""
    u = (os.environ.get("QSE_COMMUNITY_URL") or DEFAULT_URL).strip()
    return None if u == "off" else u


def pack_dir(cid: str) -> str:
    """Where community/<cid>'s files go (the profile's QSE_WEIGHTS_DIR)."""
    return os.path.join(home(), cid)


# ---------------------------------------------------------------------- the index
def build_index(models_dir: str | None = None, builtin: bool = False) -> dict:
    """The bundle from the folders: every models/<model>/community/<id>.json as written, with its
    sha256, sorted by model and id; the entries with a `profile` of the model's own when `builtin`
    (community-builtin.json), the others when not (community-index.json). A file that is not JSON
    stops it (the check names it)."""
    d = models_dir or registry.MODELS_DIR
    entries = []
    for p in sorted(glob.glob(os.path.join(d, "*", "community", "*.json"))):
        with open(p, "rb") as fh:
            raw = fh.read()
        model = os.path.basename(os.path.dirname(os.path.dirname(p)))
        data = json.loads(raw)
        if bool(isinstance(data, dict) and data.get("profile")) != builtin:
            continue
        entries.append({"model": model, "id": os.path.basename(p)[:-len(".json")],
                        "path": f"models/{model}/community/{os.path.basename(p)}",
                        "sha256": hashlib.sha256(raw).hexdigest(), "entry": data})
    return {"schema": INDEX_SCHEMA_VERSION, "generator": GENERATOR, "entries": entries}


def render_index(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def _read_index(path: str) -> dict | None:
    """A valid index document at `path`, else None."""
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if not validate(doc, INDEX_SCHEMA) else None


def entries(doc: dict, models: dict[str, dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """model id -> its entries as the API gives them (newest first), and the dropped ones:
    [{model, id, path, reason}]."""
    out: dict[str, list[dict]] = {}
    dropped = []
    seen = set()
    for r in doc.get("entries") or []:
        errs = registry.community_errors(r["entry"], r["id"], r["model"], models)
        if not errs and (r["model"], r["id"]) in seen:
            errs = ["$.id: listed twice in the index"]
        if errs:
            dropped.append({"model": r["model"], "id": r["id"], "path": r["path"],
                            "reason": "; ".join(errs[:8])})
            continue
        seen.add((r["model"], r["id"]))
        out.setdefault(r["model"], []).append(registry.community_entry(r["entry"], models[r["model"]]))
    for lst in out.values():
        lst.sort(key=lambda c: c["id"])
        lst.sort(key=lambda c: c["added"], reverse=True)
    return out, dropped


# ---------------------------------------------------------------------- the sync
_STATE: dict = {"fetched": False, "checked_at": None, "error": None, "note": None}
_GUARD = threading.Lock()
_INFLIGHT: dict | None = None


def builtin(models_dir: str | None = None) -> tuple[dict, tuple]:
    """The release's community-builtin.json and a key that moves when it changes."""
    bp = builtin_path(models_dir)
    doc = _read_index(bp) or {"schema": INDEX_SCHEMA_VERSION, "generator": "", "entries": []}
    return doc, (bp, os.path.getmtime(bp) if os.path.isfile(bp) else 0.0)


def current(models_dir: str | None = None) -> tuple[dict, str, tuple]:
    """The index in use, where it came from (github | cache | bundled) and a key that moves when
    either changes (registry.load_all's cache)."""
    cp = cache_path()
    doc = _read_index(cp)
    if doc is not None:
        src = "github" if _STATE["fetched"] else "cache"
        return doc, src, (cp, os.path.getmtime(cp), src)
    bp = bundled_path(models_dir)
    doc = _read_index(bp) or {"schema": INDEX_SCHEMA_VERSION, "generator": "", "entries": []}
    return doc, "bundled", (bp, os.path.getmtime(bp) if os.path.isfile(bp) else 0.0, "bundled")


def _meta() -> dict:
    try:
        with open(meta_path()) as fh:
            m = json.load(fh)
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(path + ".tmp", path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fail(u: str, short: str, detail: str | None = None) -> None:
    """A fetch that did not work: the page gets `short` (no URL, one line), the server's log the
    URL and the detail."""
    print(f"[community] {u}: {detail or short}", flush=True)
    _STATE.update(fetched=False, error=short, note=None)


def _fetch(timeout: float) -> None:
    """One conditional GET of url() into the cache; the outcome in _STATE. A 404 is not an error:
    the repository has published no index yet (its main before the first release that carries
    one), so the page shows the cache or the bundle with a quiet note."""
    u = url()
    _STATE["checked_at"] = _now()
    if u is None:
        _STATE.update(fetched=False, error=None, note="fetching is off (QSE_COMMUNITY_URL=off)")
        return
    meta = _meta()
    headers = {"Accept": "application/json", "User-Agent": "TandemLLM-setup"}
    if meta.get("etag") and meta.get("url") == u and _read_index(cache_path()) is not None:
        headers["If-None-Match"] = meta["etag"]
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers=headers), timeout=timeout) as r:
            body = r.read(MAX_BYTES + 1)
            etag = r.headers.get("ETag")
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            _STATE.update(fetched=True, error=None, note=None)
            return
        if exc.code == 404:
            print(f"[community] {u}: HTTP 404, no published index yet", flush=True)
            _STATE.update(fetched=False, error=None, note="no published list yet")
            return
        _fail(u, f"GitHub answered HTTP {exc.code}")
        return
    except TimeoutError:
        _fail(u, f"GitHub did not answer within {timeout:g} s")
        return
    except (OSError, ValueError) as exc:          # URLError, a refused connection, no DNS
        reason = getattr(exc, "reason", None) or exc
        if isinstance(reason, TimeoutError) or "timed out" in str(reason):
            _fail(u, f"GitHub did not answer within {timeout:g} s", str(reason))
        else:
            _fail(u, "GitHub could not be reached", str(reason))
        return
    if len(body) > MAX_BYTES:
        _fail(u, f"the published list is larger than {MAX_BYTES // 1024} KiB")
        return
    try:
        doc = json.loads(body.decode("utf-8"))
    except ValueError as exc:
        _fail(u, "the published list is not valid JSON", f"not JSON ({exc})")
        return
    bad = validate(doc, INDEX_SCHEMA)
    if bad:
        _fail(u, "the published list has the wrong shape", "not a community index: " + "; ".join(bad[:3]))
        return
    _write(cache_path(), render_index(doc))
    _write(meta_path(), json.dumps({"url": u, "etag": etag, "fetched_at": _now()}, indent=1) + "\n")
    _STATE.update(fetched=True, error=None, note=None)


def sync(timeout: float = TIMEOUT_S) -> dict:
    """Fetch now (the page opened, or Refresh), then the state. A call that arrives while another
    fetches waits for that one."""
    global _INFLIGHT
    with _GUARD:
        job, lead = (_INFLIGHT, False) if _INFLIGHT is not None else (None, True)
        if lead:
            job = _INFLIGHT = {"done": threading.Event()}
    if not lead:
        job["done"].wait(timeout + 5)
        return state()
    try:
        _fetch(timeout)
    finally:
        with _GUARD:
            _INFLIGHT = None
        job["done"].set()
    return state()


def add_url(model_id: str | None = None) -> str:
    """The "Add yours" link: the repository's issue form, its base model filled in (GitHub takes a
    form field's value from the query parameter of its id)."""
    q = {"template": "community-quant.yml", "title": "community: <name of the pack>"}
    if model_id:
        q["base_model"] = model_id
    return f"https://github.com/{REPO}/issues/new?{urllib.parse.urlencode(q)}"


def state(model_id: str | None = None) -> dict:
    """CommunitySync: where the entries come from, when the index was fetched and checked, the last
    error (short, for the page; the server's log has the URL), a quiet note (no published list
    yet, fetching off), the dropped entries (of `model_id` only, when given)."""
    _doc, src, _key = current()
    dropped = registry.community_dropped()
    if model_id is not None:
        dropped = [d for d in dropped if d["model"] == model_id]
    return {"source": src, "url": url(), "fetched_at": _meta().get("fetched_at") if src != "bundled" else None,
            "checked_at": _STATE["checked_at"], "error": _STATE["error"], "note": _STATE["note"],
            "dropped": dropped,
            "add_url": add_url(model_id)}


# ---------------------------------------------------------------------- the profile
def env(m: dict, c: dict) -> str:
    """The weight profile of community entry `c` (registry.community_entry) of model `m`, as
    ops/serve.env sources it: the pack in its own folder, standalone (the pack is the model,
    QWEN38_MODEL, as --model; #1's loader) or an EXL3 overlay over the base (QWEN38_EXL3); the
    drafters; the vision tower only where the entry says images work; its own disk-tier folder.
    Every value comes from a checked field (the registry's patterns: no quote, space or $)."""
    d = f"${{HOME}}/.tandemllm/community/{c['id']}"
    if registry.community_engine(c) == "nemotron_h":
        return _env_nemotron(m, c, d)
    lines = [f"# Weight profile {c['profile']}: {c['name']} by @{c['author']['github']}, "
             f"{c['format']} {c['bpw']:g} bpw, {c['load']},",
             f"# {c['hf']['repo']}@{c['hf']['revision']}. Generated by server/setup/community.py from",
             f"# models/{m['id']}/community/{c['id']}.json; ops/use-profile.sh {c['profile']} writes it again "
             f"before every switch.",
             f"QSE_WEIGHTS='{c['profile']}'",
             f"QSE_WEIGHTS_HF='{c['hf']['repo']}'",
             f"QSE_WEIGHTS_REVISION='{c['hf']['revision']}'",
             f'QSE_WEIGHTS_DIR="{d}"',
             f"QSE_WEIGHTS_GB='{c['size_gb']:g}'",
             "QSE_WEIGHTS_FILES=",
             "unset QWEN38_EXL3 QWEN38_DRAFT_EXL3 QWEN38_DRAFT_HEAD_EXL3 QSE_ENGINE_FLAGS QSE_ASSETS"]
    if c["load"] == "standalone":
        lines.append('QWEN38_MODEL="${QSE_WEIGHTS_DIR}"')
    else:
        lines.append('QWEN38_EXL3="${QSE_WEIGHTS_DIR}"')
    if c["drafters"] == "tandem-exl3":
        lines += ['QWEN38_DRAFT_EXL3="${HOME}/.qwen38-spark-engine/weights/exl3-drafters-k3"',
                  'QWEN38_DRAFT_HEAD_EXL3="${HOME}/.qwen38-spark-engine/weights/exl3-draft-head-k4"']
    if c["drafters"] in ("tandem-exl3", "dflash2"):
        lines += ['CKPT8="${REPO}/train/ft-b8-v2"', 'CKPT16="${REPO}/train/ft-b16"']
    else:
        # no drafter: start.sh's model flags in place of the 27B's drafter flags
        lines.append('QSE_ENGINE_FLAGS="' + ("--model ${QSE_WEIGHTS_DIR} " if c["load"] == "standalone"
                                             else "") + '--drafter none"')
    lines += ["QWEN38_SAMPLE_NUCLEUS=1",
              f"VISION={'on' if c['modalities']['image'] else 'off'}",
              f'QSE_DISK_CACHE="${{HOME}}/.qwen38-spark-engine/cache/disk-community-{c["id"]}"']
    return "\n".join(lines) + "\n"


def _env_nemotron(m: dict, c: dict, d: str) -> str:
    """A NemotronH base's community profile (ENG-377): the nearest own profile (`like`: its limits, the sampler,
    LiLiCorr and the lookup, their assets) with the pack in its place, the model flags again for the pack's
    folder, the tower only where the entry says images work, a disk-tier folder of its own."""
    like = c["memory"]["like"]
    lines = [f"# Weight profile {c['profile']}: {c['name']} by @{c['author']['github']}, "
             f"{c['format']} {c['bpw']:g} bpw, {c['load']},",
             f"# {c['hf']['repo']}@{c['hf']['revision']}. Generated by server/setup/community.py from",
             f"# models/{m['id']}/community/{c['id']}.json; ops/use-profile.sh {c['profile']} writes it again "
             f"before every switch.",
             f"# The model's own profile {like} gives the limits and the drafters; the pack takes its place.",
             f'. "${{REPO}}/ops/profiles/{like}.env"',
             f"QSE_NH_VARIANT='community-{c['id']}'",
             f"QSE_WEIGHTS='{c['profile']}'",
             f"QSE_WEIGHTS_HF='{c['hf']['repo']}'",
             f"QSE_WEIGHTS_REVISION='{c['hf']['revision']}'",
             f'QSE_WEIGHTS_DIR="{d}"',
             f"QSE_WEIGHTS_GB='{c['size_gb']:g}'",
             "QSE_WEIGHTS_FILES=",
             'QSE_NH_DIR="${QSE_WEIGHTS_DIR}"',
             'QSE_ENGINE_FLAGS="--model ${QSE_WEIGHTS_DIR} --drafter mtp ${QSE_NH_HEAD:+--draft-head $QSE_NH_HEAD --depth 3}"',
             'QSE_PAGE_CACHE_DROP="${QSE_WEIGHTS_DIR}"',
             f"VISION={'on' if c['modalities']['image'] else 'off'}",
             f'QSE_DISK_CACHE="${{HOME}}/.qwen38-spark-engine/cache/disk-community-{c["id"]}"']
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- the pack's files
#: the tower tensors an image-reading NemotronH pack holds (an Omni checkpoint's, engine/models/nemotron_h/vision.py)
NH_TOWER = ("vision_model.radio_model.", ("vision_projector.", "mlp1."))


def check_pack(m: dict, c: dict, folder: str) -> list[str]:
    """Why the files in `folder` are not community entry `c`'s pack on its base `m`, one line a problem, each
    naming the file or the key; [] when they are. Headers and JSON only, no weights read. On a NemotronH base
    (ENG-377): config.json is NemotronH (flat, or an Omni wrapper's llm_config) with the base's layer count, the
    quantization is EXL3, every shard the index names is there, and where the entry lists images in the pack the
    tower is: its tensors, preprocessor_config.json and the config's vision_config. The loader's own audit
    (engine/models/nemotron_h/pack.py) checks every tensor's shape and rate at load."""
    rules = registry.COMMUNITY_BASE_RULES.get(c["base_model"]) or {}
    if rules.get("engine") != "nemotron_h":
        return []
    errs = []
    try:
        with open(os.path.join(folder, "config.json")) as fh:
            raw = json.load(fh)
    except (OSError, ValueError) as exc:
        return [f"config.json: cannot read ({exc})"]
    t = raw.get("llm_config") if isinstance(raw.get("llm_config"), dict) else raw
    if t.get("model_type") not in ("nemotron_h", "nemotron_h_puzzle"):
        errs.append(f"config.json: model_type {t.get('model_type')!r}, not nemotron_h")
    layers = t.get("layers_block_type") or t.get("hybrid_override_pattern") or t.get("block_configs") or []
    if len(layers) != rules["layers"]:
        errs.append(f"config.json: {rules['layers']} layers expected, {len(layers)} found")
    q = raw.get("quantization_config") or t.get("quantization_config") or {}
    if c["format"] == "exl3" and str(q.get("quant_method", "")).lower() != "exl3":
        errs.append(f"config.json: quantization_config.quant_method {q.get('quant_method')!r}, not exl3")
    try:
        with open(os.path.join(folder, "model.safetensors.index.json")) as fh:
            wm = json.load(fh)["weight_map"]
    except (OSError, ValueError, KeyError) as exc:
        return errs + [f"model.safetensors.index.json: cannot read ({exc})"]
    for fn in sorted(set(wm.values())):
        if not os.path.isfile(os.path.join(folder, fn)):
            errs.append(f"{fn}: the index names it, the folder does not have it")
    if "image" in (c.get("modalities_in_pack") or []):
        tower, proj = NH_TOWER
        if not any(k.startswith(tower) for k in wm):
            errs.append(f"{tower}* not found in model.safetensors.index.json")
        if not any(k.startswith(proj) for k in wm):
            errs.append(f"{' / '.join(p + '*' for p in proj)} not found in model.safetensors.index.json")
        if not os.path.isfile(os.path.join(folder, "preprocessor_config.json")):
            errs.append("preprocessor_config.json: not in the folder")
        if not isinstance(raw.get("vision_config"), dict):
            errs.append("config.json: no vision_config (the tower's config)")
    return errs


def find(pid: str) -> tuple[dict, dict] | None:
    """(model, entry) of profile community/<id>, from the index in use."""
    for m in registry.load_all()[0].values():
        c = registry.community(m, pid)
        if c is not None:
            return m, c
    return None


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m server.setup.community")
    ap.add_argument("--index", action="store_true", help="write models/community-index.json")
    ap.add_argument("--check", action="store_true", help="with --index: exit 1 when it differs")
    ap.add_argument("cmd", nargs="?", choices=("env", "sync", "check-pack"))
    ap.add_argument("profile", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--dir", help="check-pack: the pack's folder (default: its download folder)")
    a = ap.parse_args(argv)
    if a.index:
        bad = 0
        for text, path in ((render_index(build_index()), bundled_path()),
                           (render_index(build_index(builtin=True)), builtin_path())):
            if a.check:
                try:
                    same = open(path, encoding="utf-8").read() == text
                except OSError:
                    same = False
                if not same:
                    print(f"FAIL {os.path.relpath(path, registry.ROOT)} is not the folders' index: "
                          f"python -m server.setup.community --index")
                    bad = 1
                else:
                    print(f"ok   {os.path.relpath(path, registry.ROOT)}")
                continue
            _write(path, text)
            print(f"wrote {os.path.relpath(path, registry.ROOT)}")
        return bad
    if a.cmd == "sync":
        print(json.dumps(sync(), indent=1))
        return 0
    if a.cmd == "check-pack":
        hit = find(a.profile or "") if (a.profile or "").startswith(registry.COMMUNITY_PREFIX) else None
        if hit is None:
            print(f"no community entry {a.profile!r}", file=sys.stderr)
            return 2
        folder = a.dir or pack_dir(hit[1]["id"])
        errs = check_pack(*hit, folder)
        for e in errs:
            print(f"FAIL {folder}: {e}")
        if not errs:
            print(f"ok   {folder}: {hit[1]['profile']}")
        return 1 if errs else 0
    if a.cmd == "env":
        hit = find(a.profile or "") if (a.profile or "").startswith(registry.COMMUNITY_PREFIX) else None
        if hit is None:
            cid = (a.profile or "")[len(registry.COMMUNITY_PREFIX):]
            own = next((c["profile"] for m in registry.load_all()[0].values() for c in m.get("community") or []
                        if c["id"] == cid), None)
            if own:   # VIS-175: a pack one of the model's own profiles serves
                print(f"{a.profile} is served by the profile {own}: ops/use-profile.sh {own}", file=sys.stderr)
            else:
                print(f"no community entry {a.profile!r} (models/<model>/community/<id>.json; "
                      f"python -m server.setup.registry --check)", file=sys.stderr)
            return 2
        text = env(*hit)
        if a.out:
            _write(a.out, text)
        else:
            sys.stdout.write(text)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
