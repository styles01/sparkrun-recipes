"""The model registry: one folder per base model under models/, its model.yaml (schema 1).

    models/<id>/model.yaml           the tile, the weight profiles, the measured table, the tools that
                                     run beside the model, parallelism, methods, the device presets
    models/<id>/README.md            optional
    models/<id>/measurements/*       optional raw data (the measured table's own files)
    models/schema/model.schema.json  SCHEMA below as a JSON Schema file, for editors (`--schema`)

`load_all()` reads every model.yaml with the safe loader, checks it against SCHEMA and the cross rules
below, and returns the models as the JSON the dashboard gets (GET /v1/setup/models/{id}): the file as
written, with each measured cell's list `[tok/s, min, ms a round, tokens a round, n(, ttft s)]` as an
object. A file that does not load is reported with its path and the rest still list; a folder with no
model.yaml is skipped. The result is kept until a file's mtime moves.

The cross rules: a schema version this code does not read is refused whole (a schema 2 must not be
half-read); ids unique per list; exactly one recommended profile; every measured row, asset,
dependency and default names something that exists; the tools that are on by default satisfy their
own dependencies and conflicts, and no dependency goes round in a circle; a field's default and
recommended value pass the field's own check; every `docs` link is a file in the repository. What the
repository adds (tests/test_setup_registry.py): every serve.env key a model names is one the engine
or the scripts read (`unregistered_keys`), and every ops/profiles/<name>.env has an entry
(`unlisted_profiles`).

    python -m server.setup.registry --check [model.yaml ...]   exit 1 and the problems when one fails
    python -m server.setup.registry --schema                   print models/schema/model.schema.json
    python -m server.setup.registry --community-schema         print models/schema/community.schema.json

Community packs (models/<id>/community/<pack>.json, docs/community-weights.md) are checked by
community_errors: COMMUNITY_SCHEMA, the file's name is the id, the folder is the base model, the
revision is a commit, the format has a load path, and the modalities are ones that load path serves
(standalone: text only, ENG-271). `--check` reads them from the folders and checks that
models/community-index.json is their index (models/community-builtin.json for the entries one of the
model's own profiles serves, VIS-175); the API reads the index in use (server/setup/community.py) and
gives each model its `community` list, with the profiles community/<pack> (`profile`), or the own
profile that serves the pack.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import re
import sys
import threading

from server.conductor.schema import validate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
SCHEMA_VERSION = 1

STEPS = (("models", "Models"), ("overview", "Model"), ("weights", "Weights"),
         ("components", "Tools"), ("parallelism", "Parallelism and memory"),
         ("methods", "Methods and advanced"), ("review", "Review"), ("install", "Install"))
MODES = ("greedy_512", "greedy_128", "sampled_512", "sampled_128", "long_greedy", "long_sampled")
TASKS = ("prose", "chat", "code", "long32k", "long64k")
GROUPS = ("speed", "roles", "caches", "disk", "features")
FIELD_TYPES = ("enum", "int", "number", "bool", "string")
ENV_KEY = r"^[A-Z][A-Z0-9_]*$"
HF_REPO_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
GITHUB_LOGIN = r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$"
MAKER_KINDS = ("ours", "vendor", "community")
MODEL_ID = r"^[a-z0-9][a-z0-9.-]{0,63}$"
# a string field without a pattern of its own: what a value written into serve.env may hold
SAFE_VALUE = r"^[A-Za-z0-9._:-]{1,64}$"
#: VIS-199: the most parallel chats (sequence slots) the guide and apply take, whatever a card
#: measured. A joint verify pass holds at most 32 rows (server/slots.py `Meeting`,
#: `--parallel-max-rows`) and a decoding chat keeps at least 2 of them (`share`), so 16 is the most
#: that verify in one pass. A card's `slots.max` is how far it was measured; above it the guide says
#: untested, and memory decides the rest (the estimate's `fits`).
ENGINE_MAX_SLOTS = 16

_NUM = {"type": "number"}
_NUM0 = {"type": "number", "minimum": 0}
_STR = {"type": "string", "maxLength": 2000}
_ID = {"type": "string", "pattern": r"^[a-z0-9][a-z0-9._-]{0,63}$"}
_IDS = {"type": "array", "items": _ID}
_NOTE = {"type": ["string", "null"], "maxLength": 2000}


def _obj(props: dict, required=None) -> dict:
    return {"type": "object", "properties": props,
            "required": list(props) if required is None else list(required),
            "additionalProperties": False}


_CELL = {"type": "array", "items": _NUM0}
_QUALITY = _obj({"kl_vs_bf16": {"type": ["number", "null"]}, "top1_pct": {"type": ["number", "null"]},
                 "ppl": {"type": ["number", "null"]}, "date": _STR, "source": _STR})
# OPS-95: `fetch`, optional, what ops/fetch_weights.py --assets downloads when the asset is missing:
# each entry's files (names or globs of the repo) into `to` (${HOME}, ${REPO} or ${HF_HUB}: the HF cache)
# SPD-121: `revision`, optional: the commit the files are fetched at (default: the repo's default branch)
# OPS-133: `skip_dtypes` / `skip_tensors`, optional: the safetensors files are fetched without the
# tensors of those dtypes (and their `<name>_scale_inv`) and those names, by range requests
_FETCH = _obj({"repo": _STR, "files": {"type": "array", "items": _STR}, "to": _STR, "gb": _NUM0,
               "revision": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
               "skip_dtypes": {"type": "array", "items": _STR},
               "skip_tensors": {"type": "array", "items": _STR}},
              required=("repo", "files", "to", "gb"))
# OPS-133: `satisfied_by`, optional: other assets of the entry that serve for this one when present
# (the full FP8 base for the EXL3 profiles' plain base)
_ASSET = _obj({"id": _ID, "label": _STR, "path": _STR, "also": {"type": "array", "items": _STR},
               "fetchable": {"type": "boolean"}, "protected": {"type": "boolean"},
               "hf_repo": _NOTE, "fetch": {"type": "array", "items": _FETCH}, "source": _STR,
               "satisfied_by": _IDS},
              required=("id", "label", "path", "also", "fetchable", "protected", "hf_repo", "source"))
_PROFILE_REQUIRED = ("id", "label", "format", "bpw", "disk_gb", "weights_gib", "drafters_gib", "hf_repo",
                     "hf_private", "recommended", "badges", "quality", "assets_required", "note", "made_by")
# VIS-175: who made the weights: our own pack (ours), the vendor's release (vendor) or someone else's
# (community, with the author's GitHub login); `repo` names the Hugging Face repository of the weights
# when it is not hf_repo (Bolt's IQ2: hf_repo holds its text files, NVIDIA's GGUF repository the weights)
_MADE_BY = _obj({"kind": {"enum": list(MAKER_KINDS)},
                 "name": {"type": "string", "minLength": 1, "maxLength": 80, "pattern": r"^[^\x00-\x1f<>]+$"},
                 "github": {"type": ["string", "null"], "pattern": GITHUB_LOGIN},
                 "repo": {"type": ["string", "null"], "pattern": HF_REPO_ID}},
                required=("kind", "name"))
# SRV-337, all optional (a 27B profile has none of them): the repository branch, the largest window
# that fits the device preset (null: none fits), a lookup table beside the weights (warm: in the page
# cache; cold: read past it through a row cache), a tool's memory under this profile
_PROFILE = _obj({"id": _ID, "label": _STR, "format": {"enum": ["exl3", "nvfp4", "bf16", "gguf"]},
                 "bpw": {"type": ["number", "null"]}, "disk_gb": _NUM0, "weights_gib": _NUM0,
                 "drafters_gib": _NUM0, "hf_repo": _STR, "hf_private": {"type": "boolean"},
                 "recommended": {"type": "boolean"}, "badges": {"type": "array", "items": _STR},
                 "quality": _QUALITY, "assets_required": _IDS, "note": _NOTE,
                 "hf_revision": _NOTE,
                 "max_window": {"type": ["integer", "null"], "minimum": 0},
                 "table": {"type": ["object", "null"],
                           "properties": {"gib": _NUM0, "policy": {"enum": ["warm", "cold"]}, "cache_gib": _NUM0},
                           "required": ["gib", "policy", "cache_gib"], "additionalProperties": False},
                 "components_gib": {"type": "object", "additionalProperties": _NUM0},
                 "made_by": _MADE_BY},
                required=_PROFILE_REQUIRED)
_LOSSLESS = _obj({"identical": {"type": "integer", "minimum": 0},
                  "near_tie": {"type": "integer", "minimum": 0},
                  "other": {"type": "integer", "minimum": 0},
                  "max_near_tie_ulp": {"type": ["number", "null"]}})
_ROW = _obj({"profile": _ID, "build": _STR, "date": _STR, "comparable": {"type": "boolean"},
             "note": _NOTE,
             "cells": {"type": "object", "additionalProperties": False,
                       "properties": {m: {"type": "object", "additionalProperties": False,
                                          "properties": {t: _CELL for t in TASKS}}
                                      for m in MODES}},
             "ttft_s": {"type": "object"},
             "lossless": {"type": ["object", "null"]}},
            required=("profile", "build", "date", "comparable", "cells", "ttft_s", "lossless"))
_COMPONENT = _obj({"id": _ID, "label": _STR, "group": {"enum": list(GROUPS)}, "what": _STR,
                   "default": {"type": "boolean"}, "required": {"type": "boolean"},
                   "memory_gib": _NUM0, "memory_max_gib": {"type": ["number", "null"]},
                   "elastic": {"type": "boolean"}, "disk_gb": _NUM0,
                   "depends_on": _IDS, "conflicts": _IDS,
                   # serve.env key -> {"on": value, "off": value}; checked in _cross (the keys are
                   # upper case, which the conductor's validator does not take as map keys)
                   "knobs": {"type": "object"},
                   "assets": _IDS, "docs": _STR, "note": _NOTE})
_FIELD = _obj({"id": _ID, "label": _STR, "what": _STR, "type": {"enum": list(FIELD_TYPES)},
               "options": {"type": "array", "items": _obj({"value": _STR, "label": _STR})},
               "min": _NUM, "max": _NUM, "pattern": _STR,
               "default": {"type": ["string", "number", "boolean"]},
               "recommended": {"type": ["string", "number", "boolean"]},
               "knob": {"type": ["string", "null"]}, "docs": _NOTE},
              required=("id", "label", "what", "type", "default", "recommended", "knob", "docs"))
_RANGE = _obj({"min": _NUM0, "max": _NUM0, "default": _NUM0})
_SRANGE = _obj({"min": _NUM0, "max": _NUM0, "default": _NUM0, "step": _NUM0})
SCHEMA = _obj({
    "schema": {"const": SCHEMA_VERSION},
    "id": {"type": "string", "pattern": MODEL_ID},
    "name": _STR, "family": _STR, "params": _STR, "pitch": _STR, "architecture": _STR,
    "license": _STR, "icon": _NOTE, "recommended": {"type": "boolean"},
    # the tile's place on the models step (lower first; a file without one is 0), then the folder's name
    "order": {"type": "integer", "minimum": 0},
    "links": {"type": "array", "items": _obj({"label": _STR, "url": _STR})},
    "devices": {"type": "array", "items": _obj({"id": _ID, "label": _STR, "total_gib": _NUM0,
                                                "reserved_gib": _NUM0, "note": _NOTE})},
    "assets": {"type": "array", "items": _ASSET},
    "quality_source": _STR,
    "profiles": {"type": "array", "items": _PROFILE},
    "measurements": _obj({"build": _STR, "date": _STR, "method": _STR, "source": _STR,
                          "rows": {"type": "array", "items": _ROW}}),
    "memory": _obj({"parts": {"type": "array",
                              "items": _obj({"id": _ID, "label": _STR, "gib": _NUM0,
                                             "kind": {"enum": ["fixed", "reserve"]},
                                             "source": _STR})},
                    "sources": {"type": "array", "items": _STR},
                    # SRV-337: parts that grow with the window (slot 0's), e.g. a prefill's score buffer
                    "per_window_parts": {"type": "array",
                                         "items": _obj({"id": _ID, "label": _STR,
                                                        "bytes_per_token": {"type": "integer", "minimum": 0},
                                                        "max_gib": _NUM0,
                                                        "kind": {"enum": ["fixed", "reserve"]},
                                                        "source": _STR},
                                                       required=("id", "label", "bytes_per_token", "kind",
                                                                 "source"))}},
                   required=("parts", "sources")),
    "components": {"type": "array", "items": _COMPONENT},
    "parallelism": _obj({"slots": _RANGE, "window": _SRANGE, "side_window": _SRANGE,
                         "kv_pool": _obj({"default": {"type": "boolean"}}),
                         "floor_gb": _RANGE, "kv_hard_floor_gb": _NUM0,
                         "kv_bytes_per_token": {"type": "integer", "minimum": 1},
                         "typical_fill_tokens": {"type": "integer", "minimum": 1},
                         "side_slot_const_gib": _NUM0,
                         "knobs": _obj({k: {"type": "string", "pattern": ENV_KEY}
                                        for k in ("slots", "window", "side_window", "kv_pool",
                                                  "floor_gb")}),
                         "fixed": {"type": "object"}}),
    "methods": {"type": "array", "items": _FIELD},
    "advanced": {"type": "array", "items": _FIELD},
}, required=("schema", "id", "name", "family", "params", "pitch", "architecture", "license",
             "links", "devices", "assets", "profiles", "measurements", "memory", "components",
             "parallelism", "methods", "advanced"))

# A community weight pack (docs/community-weights.md): models/<model>/community/<id>.json, one file a
# pack, schema 1. Fetched entries (server/setup/community.py) pass the same checks: the data is
# untrusted, so every string has a pattern or a length, every link is https and the avatar is
# GitHub's own. What the schema cannot say is in community_errors.
COMMUNITY_SCHEMA_VERSION = 1
COMMUNITY_SLUG = r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$"
COMMUNITY_FORMATS = ("exl3", "nvfp4", "fp8", "gguf")
COMMUNITY_SUPPORTED = ("exl3",)            # a format with a load path here; the others are refused
# what the engine reads on each load path today: the standalone path is text only (ENG-271); the
# overlay keeps the base checkpoint's vision tower; no path takes audio or video
COMMUNITY_LOADS = {"standalone": ("text",), "overlay": ("text", "image")}
# the base models a community pack loads on, each with its rules (another model's packs load only
# through that model's own profiles, `profile`):
#   engine    the loader of a standalone pack: "qwen38" the 27B family's (engine/loader.py) with the EXL3
#             overlay; "nemotron_h" NemotronH's (engine/models/nemotron_h, ENG-377), whose profile is the
#             nearest own profile's (`like`: its limits, its drafters LiLiCorr and the lookup, its assets)
#             with the pack in its place
#   loads     the load paths a pack may take and what each serves
#   drafters  the `drafters` values an entry may name; on NemotronH the pack brings none of its own
#             ("none") and drafts with the model's own
#   layers    the layer count a pack's config must have (server/setup/community.py check_pack)
COMMUNITY_BASE_RULES = {
    "qwen3.8-27b": {"engine": "qwen38", "loads": COMMUNITY_LOADS, "drafters": ("tandem-exl3", "dflash2", "none")},
    "nemotron-3.5-super-vl-120b-a12b": {"engine": "nemotron_h", "loads": {"standalone": ("text",)},
                                        "drafters": ("none",), "layers": 88},
}
COMMUNITY_BASES = tuple(COMMUNITY_BASE_RULES)
MODALITIES = ("text", "image", "audio", "video")
# the files under ~/.tandemllm/community/ that are not a pack's folder (community.py)
_RESERVED_SUFFIXES = (".json", ".partial", ".env", ".tmp")
_HTTPS = {"type": "string", "pattern": r"^https://[^\s\"'<>`]{1,400}$"}
_HF_ID = {"type": "string", "pattern": HF_REPO_ID}
_DATE = {"type": "string", "pattern": r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]$"}
_LINE = {"type": "string", "minLength": 1, "maxLength": 120, "pattern": r"^[^\x00-\x1f<>]+$"}
_SOCIAL = {"type": "object", "additionalProperties": False, "required": [], "properties": {
    "x": {"type": "string", "pattern": r"^[A-Za-z0-9_]{1,15}$"},
    "huggingface": {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"},
    "bluesky": {"type": "string", "maxLength": 253,
                "pattern": r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{1,62}$"},
    "mastodon": {"type": "string", "pattern": r"^https://[a-z0-9.-]{1,253}/@[A-Za-z0-9_]{1,30}$"},
    "website": _HTTPS}}
_TASK_NUMS = {k: _NUM0 for k in ("prose", "code", "chat", "plain")}
COMMUNITY_SCHEMA = _obj({
    "schema": {"const": COMMUNITY_SCHEMA_VERSION},
    "id": {"type": "string", "maxLength": 80, "pattern": COMMUNITY_SLUG},
    "name": _LINE,
    "base_model": {"type": "string", "pattern": MODEL_ID},
    "finetune_of": {"type": ["string", "null"], "pattern": _HF_ID["pattern"]},
    "author": _obj({"github": {"type": "string", "pattern": GITHUB_LOGIN},
                    "name": {"type": ["string", "null"], "maxLength": 80, "pattern": r"^[^\x00-\x1f<>]*$"},
                    "social": _SOCIAL}, required=("github",)),
    "hf": _obj({"repo": _HF_ID, "revision": {"type": "string", "pattern": r"^[0-9a-f]{40}$"}}),
    "format": {"enum": list(COMMUNITY_FORMATS)},
    "bpw": {"type": "number", "minimum": 1, "maximum": 16},
    "size_gb": {"type": "number", "minimum": 0.1, "maximum": 2000},
    "load": {"enum": list(COMMUNITY_LOADS)},
    "modalities": _obj({k: {"type": "boolean"} for k in MODALITIES}),
    "modalities_in_pack": {"type": "array", "maxItems": 4, "items": {"enum": list(MODALITIES)}},
    "drafters": {"enum": ["tandem-exl3", "dflash2", "in-pack", "none"]},
    "license": _LINE,
    "license_url": {"type": ["string", "null"], "pattern": _HTTPS["pattern"]},
    "description": {"type": "string", "maxLength": 500, "pattern": r"^[^\x00-\x08\x0b-\x1f<>]*$"},
    "tags": {"type": "array", "maxItems": 12,
             "items": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9-]{0,31}$"}},
    "added": _DATE,
    "status": {"enum": ["submitted", "verified", "broken"]},
    "verified": {"type": ["object", "null"], "additionalProperties": False,
                 "required": ["build", "device", "date"],
                 "properties": {"build": {"type": "string", "pattern": r"^[0-9][0-9A-Za-z.+-]{0,31}$"},
                                "device": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9-]{0,39}$"},
                                "date": _DATE,
                                "tok_s": _obj(_TASK_NUMS, required=()),
                                "tokens_per_round": _obj(_TASK_NUMS, required=()),
                                "notes": {"type": ["string", "null"], "maxLength": 300,
                                          "pattern": r"^[^\x00-\x1f<>]*$"}}},
    "links": {"type": "object", "additionalProperties": _HTTPS},
    # VIS-175: the model's own profile that serves this pack (turboderp's EXL3 branches): Use
    # selects that profile, and the pack never becomes a community/<id> profile of its own
    "profile": _ID,
}, required=("schema", "id", "name", "base_model", "author", "hf", "format", "bpw", "size_gb", "load",
             "modalities", "drafters", "license", "added", "status"))


class RegistryError(ValueError):
    def __init__(self, path: str, problems: list[str]):
        super().__init__(f"{path}: " + "; ".join(problems[:8]))
        self.path, self.problems = path, problems


def field_error(f: dict, value) -> str | None:
    """Why `value` is not a valid value of field `f` (a method, an advanced setting or a step
    field), or None. Strings that came from a client are written into serve.env, so a string is
    only ever an option's value or one that matches the field's pattern."""
    t = f["type"]
    if t == "bool":
        return None if isinstance(value, bool) else "expected true or false"
    if t == "enum":
        opts = [o["value"] if isinstance(o, dict) else o for o in f.get("options") or []]
        return None if isinstance(value, str) and value in opts else f"expected one of {', '.join(opts)}"
    if t in ("int", "number"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                (t == "int" and not float(value).is_integer()):
            return "expected an integer" if t == "int" else "expected a number"
        if value != value or value in (float("inf"), float("-inf")):
            return "expected a number"
        if f.get("min") is not None and value < f["min"]:
            return f"at least {f['min']:g}"
        if f.get("max") is not None and value > f["max"]:
            return f"at most {f['max']:g}"
        return None
    pattern = f.get("pattern") or SAFE_VALUE
    if not isinstance(value, str) or "\n" in value or not re.fullmatch(pattern, value):
        return f"letters, digits and . _ : - only ({pattern})"
    return None


def _docs_ok(link: str, root: str) -> bool:
    path = link.split("#", 1)[0]
    return bool(path) and not os.path.isabs(path) and ".." not in path.split("/") and \
        os.path.isfile(os.path.join(root, path))


def _cycle(comps: list[dict]) -> str | None:
    """A component that depends on itself through others, or None."""
    deps = {c["id"]: c["depends_on"] for c in comps}
    for start in deps:
        seen, todo = set(), list(deps[start])
        while todo:
            d = todo.pop()
            if d == start:
                return start
            if d not in seen and d in deps:
                seen.add(d)
                todo += deps[d]
    return None


def _cross(m: dict, root: str) -> list[str]:
    errs = []
    for key in ("profiles", "components", "methods", "advanced", "devices", "assets"):
        ids = [x["id"] for x in m[key]]
        if len(set(ids)) != len(ids):
            errs.append(f"$.{key}: two entries share an id")
    fields = [f["id"] for f in m["methods"] + m["advanced"]]
    if len(set(fields)) != len(fields):
        errs.append("$.methods/$.advanced: a field id is used twice")
    if not m["devices"]:
        errs.append("$.devices: at least one device preset")
    rec = [p["id"] for p in m["profiles"] if p["recommended"]]
    if len(rec) != 1:
        errs.append(f"$.profiles: exactly one recommended profile, not {len(rec)}")
    for i, p in enumerate(m["profiles"]):
        if p["made_by"]["kind"] == "community" and not p["made_by"].get("github"):
            errs.append(f"$.profiles[{i}].made_by.github: a community pack names its author's GitHub login")
    assets = {a["id"] for a in m["assets"]}
    for i, a in enumerate(m["assets"]):
        if a.get("fetch") and not a["fetchable"]:
            errs.append(f"$.assets[{i}]: an asset with a fetch list is fetchable")
        for b in a.get("satisfied_by") or []:
            if b not in assets or b == a["id"]:
                errs.append(f"$.assets[{i}].satisfied_by: no other asset {b!r}")
        for j, f in enumerate(a.get("fetch") or []):
            if not f["files"] or not re.match(r"^\$\{(HOME|REPO|HF_HUB)\}", f["to"]):
                errs.append(f"$.assets[{i}].fetch[{j}]: files, and a `to` under ${{HOME}}, "
                            f"${{REPO}} or ${{HF_HUB}}")
    for i, p in enumerate(m["profiles"]):
        for a in p["assets_required"]:
            if a not in assets:
                errs.append(f"$.profiles[{i}].assets_required: no asset {a!r}")
        if not p["quality"]["date"] or not p["quality"]["source"]:
            errs.append(f"$.profiles[{i}].quality: a measured value needs its date and source")
    profiles = {p["id"] for p in m["profiles"]}
    rows = m["measurements"]["rows"]
    for i, r in enumerate(rows):
        if r["profile"] not in profiles:
            errs.append(f"$.measurements.rows[{i}]: no profile {r['profile']!r}")
        if not r["comparable"] and not r.get("note"):
            errs.append(f"$.measurements.rows[{i}]: a row that is not comparable says why (note)")
        if r["comparable"] and r["build"] != m["measurements"]["build"]:
            errs.append(f"$.measurements.rows[{i}]: build {r['build']} is not the table's "
                        f"{m['measurements']['build']}; mark it comparable: false")
        for mode, cells in r["cells"].items():
            for task, c in cells.items():
                if len(c) not in (5, 6):
                    errs.append(f"$.measurements.rows[{i}].cells.{mode}.{task}: 5 or 6 numbers")
    comps = {c["id"]: c for c in m["components"]}
    for i, c in enumerate(m["components"]):
        for d in c["depends_on"] + c["conflicts"]:
            if d not in comps or d == c["id"]:
                errs.append(f"$.components[{i}]: no other component {d!r}")
        for a in c["assets"]:
            if a not in assets:
                errs.append(f"$.components[{i}].assets: no asset {a!r}")
        if c["required"] and not c["default"]:
            errs.append(f"$.components[{i}]: a required component is on by default")
        if c["default"]:
            for d in c["depends_on"]:
                if d in comps and not comps[d]["default"]:
                    errs.append(f"$.components[{i}]: on by default, but {d!r} it depends on is not")
            for d in c["conflicts"]:
                if d in comps and comps[d]["default"]:
                    errs.append(f"$.components[{i}]: on by default together with {d!r}, a conflict")
        if c["memory_max_gib"] is not None and c["memory_max_gib"] < c["memory_gib"]:
            errs.append(f"$.components[{i}].memory_max_gib: under memory_gib")
        for k, v in c["knobs"].items():
            if not re.match(ENV_KEY, k):
                errs.append(f"$.components[{i}].knobs: {k!r} is not a serve.env key")
            if not isinstance(v, dict) or set(v) != {"on", "off"} or \
                    not all(x is None or isinstance(x, str) for x in v.values()):
                errs.append(f"$.components[{i}].knobs.{k}: {{\"on\": string|null, \"off\": "
                            f"string|null}} (quote on and off: YAML reads them as booleans)")
            elif any(isinstance(x, str) and "\n" in x for x in v.values()):
                errs.append(f"$.components[{i}].knobs.{k}: one line only")
        if not _docs_ok(c["docs"], root):
            errs.append(f"$.components[{i}].docs: no file {c['docs']!r} in the repository")
    loop = _cycle(m["components"])
    if loop:
        errs.append(f"$.components: {loop!r} depends on itself")
    p = m["parallelism"]
    for k in ("slots", "window", "side_window", "floor_gb"):
        r = p[k]
        if not r["min"] <= r["default"] <= r["max"]:
            errs.append(f"$.parallelism.{k}: the default is outside min..max")
    for k, v in (p.get("fixed") or {}).items():
        if not re.match(ENV_KEY, k) or not isinstance(v, str) or not re.fullmatch(SAFE_VALUE, v):
            errs.append(f"$.parallelism.fixed.{k}: a serve.env key and a plain string value")
    for group in ("methods", "advanced"):
        for i, f in enumerate(m[group]):
            if f["type"] == "enum" and not f.get("options"):
                errs.append(f"$.{group}[{i}]: an enum needs options")
            if f["knob"] is not None and not re.match(ENV_KEY, f["knob"]):
                errs.append(f"$.{group}[{i}].knob: {f['knob']!r} is not a serve.env key")
            for which in ("default", "recommended"):
                why = field_error(f, f[which])
                if why:
                    errs.append(f"$.{group}[{i}].{which}: {why}")
            if f.get("docs") and not _docs_ok(f["docs"], root):
                errs.append(f"$.{group}[{i}].docs: no file {f['docs']!r} in the repository")
    return errs


def _cell(c: list) -> dict:
    out = {"tok_s": c[0], "min": c[1], "ms_round": c[2], "tok_round": c[3], "n": int(c[4])}
    if len(c) > 5:
        out["ttft_s"] = c[5]
    return out


def parse(data, path: str = "model.yaml", root: str | None = None) -> dict:
    """A model.yaml's data, checked, as the API's JSON; RegistryError with every problem."""
    if isinstance(data, dict) and data.get("schema") != SCHEMA_VERSION:
        raise RegistryError(path, [f"$.schema: {data.get('schema')!r}; this server reads schema "
                                   f"{SCHEMA_VERSION} only (a newer file needs a newer TandemLLM)"])
    errs = validate(data, SCHEMA)
    if not errs:
        errs = _cross(data, root or ROOT)
    if errs:
        raise RegistryError(path, errs)
    m = dict(data)
    m["measurements"] = dict(data["measurements"], rows=[
        dict(r, note=r.get("note"),
             cells={mode: {t: _cell(c) for t, c in cells.items()} for mode, cells in r["cells"].items()})
        for r in data["measurements"]["rows"]])
    m.setdefault("recommended", False)
    m["parallelism"] = dict(data["parallelism"], fixed=dict(data["parallelism"].get("fixed") or {}))
    m.pop("quality_source", None)
    return m


def load(path: str, root: str | None = None) -> dict:
    import yaml
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryError(path, [f"cannot read: {exc}"]) from None
    m = parse(data, path, root)
    if os.path.basename(os.path.dirname(path)) != m["id"]:
        raise RegistryError(path, [f"the folder is not named after the id {m['id']!r}"])
    return m


_CACHE: dict = {"key": None, "models": {}, "errors": [], "dropped": []}
_LOCK = threading.Lock()


def load_all(models_dir: str | None = None) -> tuple[dict[str, dict], list[dict]]:
    """id -> model, and [{path, message}] for the files that did not load. Each model carries its
    community packs (`community`, from the index in use: server/setup/community.py); the entries
    that did not check are community_dropped()."""
    from server.setup import community as community_mod
    d = models_dir or MODELS_DIR
    paths = sorted(glob.glob(os.path.join(d, "*", "model.yaml")))
    doc, _src, ckey = community_mod.current(d)
    bdoc, bkey = community_mod.builtin(d)
    key = (d, tuple((p, os.path.getmtime(p)) for p in paths), ckey, bkey)
    with _LOCK:
        if _CACHE["key"] == key:
            return _CACHE["models"], _CACHE["errors"]
        models, errors = {}, []
        root = os.path.dirname(os.path.abspath(d))
        for p in paths:
            try:
                m = load(p, root)
            except RegistryError as exc:
                errors.append({"path": os.path.relpath(p, root), "message": str(exc)})
                continue
            models[m["id"]] = m
        models = dict(sorted(models.items(), key=lambda kv: int(kv[1].get("order") or 0)))
        # the release's profile-backed entries (VIS-175) first; a fetched entry of the same id is dropped
        own, dropped = community_mod.entries(bdoc, models)
        packs, more = community_mod.entries(doc, models)
        dropped += more
        for m in models.values():
            ids = {c["id"] for c in own.get(m["id"], [])}
            m["community"] = own.get(m["id"], []) + [c for c in packs.get(m["id"], []) if c["id"] not in ids]
        _CACHE.update(key=key, models=models, errors=errors, dropped=dropped)
        return models, errors


def community_dropped(models_dir: str | None = None) -> list[dict]:
    """The community entries of the index in use that did not check: [{model, id, path, reason}]."""
    load_all(models_dir)
    return _CACHE["dropped"]


def community_file_errors(models_dir: str | None = None) -> list[tuple[str, list[str]]]:
    """(path, problems) of every models/<model>/community/<id>.json that does not check, read
    from the folders themselves (the --check of a pull request)."""
    d = models_dir or MODELS_DIR
    models = load_all(d)[0]
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*", "community", "*"))):
        folder = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if not p.endswith(".json"):
            out.append((p, ["only <id>.json files belong in a community folder"]))
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            out.append((p, [f"cannot read: {exc}"]))
            continue
        errs = community_errors(data, os.path.basename(p)[:-len(".json")], folder, models)
        if errs:
            out.append((p, errs))
    return out


def get(model_id: str, models_dir: str | None = None) -> dict | None:
    return load_all(models_dir)[0].get(model_id)


def recommended_profile(m: dict) -> dict:
    return next(p for p in m["profiles"] if p["recommended"])


def profile(m: dict, pid: str) -> dict | None:
    """The profile `pid`: one of the model's own, or community/<id> (community_profile)."""
    if isinstance(pid, str) and pid.startswith(COMMUNITY_PREFIX):
        c = community(m, pid)
        return community_profile(m, c) if c is not None else None
    return next((p for p in m["profiles"] if p["id"] == pid), None)


def config_max_len(folder: str | None) -> int | None:
    """The longest context a checkpoint's own config names: `max_position_embeddings` at the top of its
    config.json or in `text_config` / `llm_config` (a multimodal wrapper: Super VL's Omni), the tokenizer's
    `model_max_length` where that is shorter; None without a readable config."""
    if not folder:
        return None
    try:
        with open(os.path.join(folder, "config.json")) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return None
    n = next((c.get("max_position_embeddings") for c in (cfg, cfg.get("text_config") or {},
                                                          cfg.get("llm_config") or {})
              if isinstance(c, dict) and isinstance(c.get("max_position_embeddings"), int)), None)
    if not n or n <= 0:
        return None
    try:
        with open(os.path.join(folder, "tokenizer_config.json")) as fh:
            t = json.load(fh).get("model_max_length")
        if isinstance(t, int) and 0 < t < n:          # LFM2.5's reads 1e30: no limit
            n = t
    except (OSError, ValueError):
        pass
    return int(n)


def with_config_window(m: dict, folders: list) -> dict:
    """ENG-363: the model with its window range up to what its checkpoint holds. The first folder of
    `folders` (the weight folders of its profiles present on this box) with a config sets the largest
    window and side window (`config_max_len`); a box without the weights keeps the card's, which name
    the same numbers. The defaults stay the card's (lowered only to the new largest)."""
    n = next((x for x in map(config_max_len, folders) if x), None)
    par = m["parallelism"]
    if not n or all(par[k]["max"] == n for k in ("window", "side_window")):
        return m
    m = copy.deepcopy(m)
    for k in ("window", "side_window"):
        r = m["parallelism"][k]
        r["max"] = n - n % r["step"]
        r["default"] = min(r["default"], r["max"])
    return m


def window_refusal(profile_id: str, max_len: int, models: dict[str, dict] | None = None) -> str | None:
    """SRV-335: why a start of `profile_id` with `max_len` must not load, or None. A profile whose entry
    names its largest window on the device (`max_window`) refuses a longer one, and null means it fits
    no window there; a profile without the key (the 27B's) or in no entry is never refused here."""
    for m in (models if models is not None else load_all()[0]).values():
        p = profile(m, profile_id)
        if p is None or "max_window" not in p:
            continue
        dev, mw = m["devices"][0]["label"], p["max_window"]
        if mw is None:
            return f"{profile_id} fits no window on the {dev} (models/{m['id']}/model.yaml: max_window null)"
        if max_len > mw:
            return (f"--max-len {max_len:,} is above {profile_id}'s largest window, {mw:,} on the {dev} "
                    f"(models/{m['id']}/model.yaml)")
        return None
    return None


def component(m: dict, cid: str) -> dict | None:
    return next((c for c in m["components"] if c["id"] == cid), None)


def asset(m: dict, aid: str) -> dict | None:
    return next((a for a in m["assets"] if a["id"] == aid), None)


def fields(m: dict) -> dict[str, tuple[str, dict]]:
    """field id -> (methods | advanced, the field)."""
    return {f["id"]: (g, f) for g in ("methods", "advanced") for f in m[g]}


def tile(m: dict) -> dict:
    """The models step's tile: the fastest greedy-512 cell of the recommended profile."""
    rec = recommended_profile(m)
    best = None
    for r in m["measurements"]["rows"]:
        if r["profile"] != rec["id"] or not r["comparable"]:
            continue
        for task, c in (r["cells"].get("greedy_512") or {}).items():
            if best is None or c["tok_s"] > best["tok_s"]:
                best = {"tok_s": c["tok_s"], "task": task, "profile": rec["id"]}
    return {"id": m["id"], "name": m["name"], "family": m["family"], "params": m["params"],
            "pitch": m["pitch"], "recommended": bool(m.get("recommended")), "icon": m.get("icon"),
            "best_speed": best}


def _field(f: dict) -> dict:
    out = {"field": f["id"], "label": f["label"], "type": f["type"]}
    if f["type"] == "enum":
        out["options"] = [o["value"] for o in f["options"]]
    for k in ("min", "max", "pattern"):
        if f.get(k) is not None:
            out[k] = f[k]
    return out


def wizard_fields(m: dict) -> list[dict]:
    """The steps and their fields: what the step forms show and what Wren's tool calls are checked
    against (`set_field(step, field, value)`, `toggle(component, on)`, `select_profile(id)`)."""
    p = m["parallelism"]
    per_step = {
        "weights": [{"field": "profile", "label": "Weight profile", "type": "enum",
                     "options": [x["id"] for x in profiles(m)]}],
        "components": [{"field": c["id"], "label": c["label"], "type": "bool"}
                       for c in m["components"]],
        "parallelism": [
            {"field": "slots", "label": "Parallel requests", "type": "int",
             "min": p["slots"]["min"], "max": ENGINE_MAX_SLOTS},
            {"field": "window", "label": "Context window, first slot", "type": "int",
             "min": p["window"]["min"], "max": p["window"]["max"]},
            {"field": "side_window", "label": "Context window, other slots", "type": "int",
             "min": p["side_window"]["min"], "max": p["side_window"]["max"]},
            {"field": "kv_pool", "label": "Shared KV pool", "type": "bool"},
            {"field": "floor_gb", "label": "Memory kept free (GiB)", "type": "number",
             "min": p["floor_gb"]["min"], "max": p["floor_gb"]["max"]}],
        "methods": [_field(f) for f in m["methods"] + m["advanced"]],
    }
    return [{"index": i, "id": sid, "label": label, "fields": per_step.get(sid, [])}
            for i, (sid, label) in enumerate(STEPS)]


def step_field(m: dict, step, name: str) -> dict | None:
    """The field `name` of step `step` (its id or index), or None."""
    for s in wizard_fields(m):
        if step in (s["id"], s["index"]):
            return next((f for f in s["fields"] if f["field"] == name), None)
    return None


def env_keys(m: dict) -> dict[str, str]:
    """Every serve.env key the model names -> where (for the knob test and the plan)."""
    out = {}
    for c in m["components"]:
        for k in c["knobs"]:
            out[k] = f"components.{c['id']}"
    for f in m["methods"] + m["advanced"]:
        if f["knob"]:
            out[f["knob"]] = f"{'methods' if f in m['methods'] else 'advanced'}.{f['id']}"
    for k, v in m["parallelism"]["knobs"].items():
        out[v] = f"parallelism.{k}"
    for k in m["parallelism"].get("fixed") or {}:
        out[k] = "parallelism.fixed"
    return out


_ASSIGN = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=", re.M)


def serve_env_keys(root: str | None = None) -> set[str]:
    """The keys ops/serve.env and the profiles assign, commented-out template lines included
    (`#QSE_KV_POOL=on` documents a key the scripts read), plus the keys ops/start.sh reads."""
    r = root or ROOT
    keys = set()
    for p in [os.path.join(r, "ops", "serve.env")] + glob.glob(os.path.join(r, "ops", "profiles", "*.env")):
        with open(p) as fh:
            keys |= set(_ASSIGN.findall(fh.read()))
    with open(os.path.join(r, "ops", "start.sh")) as fh:
        keys |= set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", fh.read()))
    return keys


def server_env_reads(root: str | None = None) -> set[str]:
    """The QSE_* variables server/ reads from its environment (`os.environ.get("QSE_...")`)."""
    r = root or ROOT
    out = set()
    for p in glob.glob(os.path.join(r, "server", "**", "*.py"), recursive=True):
        with open(p) as fh:
            out |= set(re.findall(r"""environ(?:\.get\(|\[)\s*["'](QSE_[A-Z0-9_]+)["']""", fh.read()))
    return out


def unregistered_keys(m: dict, root: str | None = None) -> list[str]:
    """Keys the model names that nothing reads: a QWEN38_* key outside engine/settings.py's
    registry; any other key that ops/serve.env, a profile or ops/start.sh never names and, for a
    QSE_* key, that server/ never reads from its environment either."""
    from engine.settings import KNOBS
    known = serve_env_keys(root)
    reads = server_env_reads(root)
    bad = []
    for k, where in sorted(env_keys(m).items()):
        if k.startswith("QWEN38_"):
            if k[len("QWEN38_"):] not in KNOBS:
                bad.append(f"{k} ({where}): not in engine/settings.py")
        elif k not in known and k not in reads:
            bad.append(f"{k} ({where}): not in ops/serve.env, a profile or ops/start.sh")
    return bad


def profile_names(root: str | None = None) -> list[str]:
    r = root or ROOT
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(r, "ops", "profiles", "*.env"))
                  if not os.path.basename(p).startswith("_"))


def unlisted_profiles(models: dict[str, dict], root: str | None = None) -> list[str]:
    listed = {p["id"] for m in models.values() for p in m["profiles"]}
    return [n for n in profile_names(root) if n not in listed]


def hf_hub() -> str:
    return os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")


def expand(path: str, repo: str | None = None) -> str:
    """`${HOME}`, `${REPO}` and `${HF_HUB}` in a registry path (the only three it uses)."""
    return (path.replace("${HOME}", os.path.expanduser("~")).replace("${REPO}", repo or ROOT)
            .replace("${HF_HUB}", hf_hub()))


def asset_present(a: dict, repo: str | None = None, m: dict | None = None) -> bool:
    """Every path of the asset exists (a glob matches), or (with the model `m`) an asset it names
    in `satisfied_by` is present."""
    if all(glob.glob(expand(p, repo)) for p in [a["path"], *a.get("also", [])]):
        return True
    return m is not None and any(asset_present(asset(m, b), repo) for b in
                                 a.get("satisfied_by") or [] if asset(m, b) is not None)


# ---------------------------------------------------------------------- community packs
COMMUNITY_PREFIX = "community/"
# the base model's assets a community pack reads besides its own folder, by its drafters and its
# load path (the K3 files are named after the BF16 drafter folders, which give the rest of each arm)
_COMMUNITY_ASSETS = {"tandem-exl3": ["exl3-drafters-k3", "exl3-draft-head-k4", "nvfp4-drafters"],
                     "dflash2": ["nvfp4-drafters"], "in-pack": [], "none": []}
_SOCIAL_URL = (("x", "X", "https://x.com/{}"), ("huggingface", "Hugging Face", "https://huggingface.co/{}"),
               ("bluesky", "Bluesky", "https://bsky.app/profile/{}"), ("mastodon", "Mastodon", "{}"),
               ("website", "Website", "{}"))


def _date_ok(s: str) -> bool:
    import datetime
    try:
        datetime.date.fromisoformat(s)
        return True
    except ValueError:
        return False


def _like(m: dict, fmt: str, bpw: float) -> dict | None:
    """The model's own profile of `fmt` nearest to `bpw` (the lower one on a tie): the memory a
    community pack of that size is estimated with."""
    same = [p for p in m["profiles"] if p["format"] == fmt and p["bpw"] is not None]
    return min(same, key=lambda p: (abs(p["bpw"] - bpw), p["bpw"])) if same else None


def community_engine(e: dict) -> str | None:
    """The loader of entry `e`'s base (COMMUNITY_BASE_RULES), None outside the community bases."""
    return (COMMUNITY_BASE_RULES.get(e["base_model"]) or {}).get("engine")


def community_assets(e: dict, m: dict | None = None) -> list[str]:
    """The base model's assets a community pack reads besides its own folder: by its drafters and its load
    path on the 27B; on NemotronH (`m` the base) its nearest profile's, which it drafts with."""
    if community_engine(e) == "nemotron_h":
        like = _like(m, e["format"], e["bpw"]) if m is not None else None
        return list(like["assets_required"]) if like else []
    return _COMMUNITY_ASSETS[e["drafters"]] + (["base-fp8"] if e["load"] == "overlay" else [])


def community_errors(data, slug: str | None, folder: str | None, models: dict[str, dict]) -> list[str]:
    """Why a community entry does not load, one line a problem with its field: the schema, then
    what the schema cannot say. `slug`: the file's name without .json; `folder`: the model folder
    it sits in (an index entry's `model`); `models`: id -> the models that loaded."""
    if isinstance(data, dict) and data.get("schema") != COMMUNITY_SCHEMA_VERSION:
        return [f"$.schema: {data.get('schema')!r}; this server reads community schema "
                f"{COMMUNITY_SCHEMA_VERSION} only (a newer file needs a newer TandemLLM)"]
    errs = validate(data, COMMUNITY_SCHEMA)
    if errs:
        return errs
    e = data
    if slug is not None and e["id"] != slug:
        errs.append(f"$.id: {e['id']!r} is not the file's name {slug!r}")
    if e["id"].endswith(_RESERVED_SUFFIXES):
        errs.append(f"$.id: may not end in {', '.join(_RESERVED_SUFFIXES)}")
    if folder is not None and e["base_model"] != folder:
        errs.append(f"$.base_model: {e['base_model']!r} is not the folder's model {folder!r}")
    m = models.get(e["base_model"])
    own = e.get("profile")
    if m is None:
        errs.append(f"$.base_model: no model {e['base_model']!r} in models/")
    elif own is not None:
        # VIS-175: a pack the model's own profile serves is that profile's repository, format and size
        p = next((x for x in m["profiles"] if x["id"] == own), None)
        if p is None:
            errs.append(f"$.profile: {e['base_model']} has no profile {own!r}")
        else:
            if p["hf_repo"] != e["hf"]["repo"]:
                errs.append(f"$.hf.repo: {e['hf']['repo']!r} is not profile {own}'s {p['hf_repo']!r}")
            if p["format"] != e["format"] or p["bpw"] != e["bpw"]:
                errs.append(f"$.format: {e['format']} {e['bpw']:g} bpw is not profile {own}'s "
                            f"{p['format']} {p['bpw']} bpw")
    elif e["base_model"] not in COMMUNITY_BASES:
        errs.append(f"$.base_model: community packs load on {', '.join(COMMUNITY_BASES)} only")
    rules = COMMUNITY_BASE_RULES.get(e["base_model"]) if own is None else None
    if rules is not None and e["load"] not in rules["loads"]:
        errs.append(f"$.load: {e['base_model']} packs load {', '.join(rules['loads'])} only")
    if rules is not None and e["drafters"] not in rules["drafters"]:
        errs.append(f"$.drafters: a {e['base_model']} pack takes {', '.join(rules['drafters'])}"
                    + (" (it drafts with the model's own drafters, from its nearest profile)"
                       if rules["engine"] == "nemotron_h" else ""))
    if own is None and e["format"] not in COMMUNITY_SUPPORTED:
        errs.append(f"$.format: TandemLLM has no load path for {e['format']} packs yet "
                    f"({', '.join(COMMUNITY_SUPPORTED)} only)")
    mods = e["modalities"]
    if not mods["text"]:
        errs.append("$.modalities.text: a pack serves text")
    # a profile-backed pack serves what its model serves: images where the model has a Vision tool
    served = (("text", "image") if m and any("VISION" in c["knobs"] for c in m["components"]) else ("text",)) \
        if own is not None else (rules["loads"] if rules is not None else COMMUNITY_LOADS).get(e["load"], ())
    for k in MODALITIES:
        if mods[k] and k not in served:
            errs.append(f"$.modalities.{k}: the {e['load']} load path does not serve {k} yet"
                        + (" (ENG-271)" if own is None and e["load"] == "standalone" and k == "image" else ""))
    pack = e.get("modalities_in_pack")
    if pack is not None:
        if len(set(pack)) != len(pack):
            errs.append("$.modalities_in_pack: a modality is listed twice")
        for k in MODALITIES:
            if mods[k] and k not in pack:
                errs.append(f"$.modalities_in_pack: {k} works but the pack does not list it")
    if e["drafters"] == "in-pack" and own is None:
        errs.append("$.drafters: a draft layer in the pack runs on one of the model's own profiles only "
                    "(`profile`)")
    if len(set(e.get("tags") or [])) != len(e.get("tags") or []):
        errs.append("$.tags: a tag is listed twice")
    if not _date_ok(e["added"]):
        errs.append(f"$.added: {e['added']!r} is not a date")
    v = e.get("verified")
    if e["status"] == "verified" and not v:
        errs.append("$.verified: a verified entry says what was measured (build, device, date)")
    if v and not _date_ok(v["date"]):
        errs.append(f"$.verified.date: {v['date']!r} is not a date")
    if m is not None and own is None and e["base_model"] in COMMUNITY_BASES:
        have = {a["id"] for a in m["assets"]}
        for a in community_assets(e, m):
            if a not in have:
                errs.append(f"$.drafters: {e['base_model']} has no asset {a!r} for "
                            f"{e['drafters']} on the {e['load']} path")
        like = _like(m, e["format"], e["bpw"]) if e["format"] in COMMUNITY_SUPPORTED else None
        if e["format"] in COMMUNITY_SUPPORTED and like is None:
            errs.append(f"$.format: {e['base_model']} has no {e['format']} profile to estimate "
                        f"the memory from")
        if like is not None and community_engine(e) == "nemotron_h" and \
                not os.path.isfile(os.path.join(ROOT, "ops", "profiles", like["id"] + ".env")):
            errs.append(f"$.format: no ops/profiles/{like['id']}.env, the profile a {e['base_model']} pack "
                        f"is served with")
    return errs


def community_entry(e: dict, m: dict) -> dict:
    """A checked entry as the API gives it: every key filled, and the computed ones (the
    profile's name, the avatar, the links, the memory the estimate counts). A pack the model's own
    profile serves (`profile`, VIS-175) keeps that profile's name and figures."""
    a = e["author"]
    social = dict(a.get("social") or {})
    like = (next(p for p in m["profiles"] if p["id"] == e["profile"]) if e.get("profile")
            else _like(m, e["format"], e["bpw"]))
    mods = e["modalities"]
    return dict(
        e, finetune_of=e.get("finetune_of"),
        author={"github": a["github"], "name": a.get("name"), "social": social},
        modalities_in_pack=list(e.get("modalities_in_pack") or [k for k in MODALITIES if mods[k]]),
        license_url=e.get("license_url"), description=e.get("description") or "",
        tags=list(e.get("tags") or []),
        verified=None if not e.get("verified") else {
            "build": e["verified"]["build"], "device": e["verified"]["device"],
            "date": e["verified"]["date"], "tok_s": dict(e["verified"].get("tok_s") or {}),
            "tokens_per_round": dict(e["verified"].get("tokens_per_round") or {}),
            "notes": e["verified"].get("notes")},
        links=dict(e.get("links") or {}),
        profile=e.get("profile") or COMMUNITY_PREFIX + e["id"],
        avatar_url=f"https://github.com/{a['github']}.png?size=96",
        author_url=f"https://github.com/{a['github']}",
        social_links=[{"network": k, "label": label, "url": fmt.format(social[k])}
                      for k, label, fmt in _SOCIAL_URL if social.get(k)],
        hf_url=f"https://huggingface.co/{e['hf']['repo']}/tree/{e['hf']['revision']}",
        memory={"weights_gib": like["weights_gib"],
                "drafters_gib": like["drafters_gib"] if e["drafters"] != "none" or
                community_engine(e) == "nemotron_h" else 0.0,
                "like": like["id"]})


def community_profile(m: dict, c: dict) -> dict:
    """An API entry (community_entry) as a weight profile: what the selection, the estimate, the
    plan and the apply job read of `profile`. Its memory is its `like` profile's; its disk the
    entry's download."""
    like = profile(m, c["memory"]["like"])
    v = c["verified"]
    return {"id": c["profile"], "label": c["name"], "format": c["format"], "bpw": c["bpw"],
            "disk_gb": c["size_gb"], "weights_gib": c["memory"]["weights_gib"],
            "drafters_gib": c["memory"]["drafters_gib"], "hf_repo": c["hf"]["repo"],
            "hf_private": False, "hf_revision": c["hf"]["revision"], "recommended": False,
            "badges": ["community"],
            "quality": {"kl_vs_bf16": None, "top1_pct": None, "ppl": None,
                        "date": v["date"] if v else c["added"],
                        "source": f"community entry {c['profile']} ({c['status']})"},
            "assets_required": community_assets(c, m), "note": c["description"] or None,
            "components_gib": dict(like.get("components_gib") or {}),
            "made_by": {"kind": "community", "name": c["author"]["github"], "github": c["author"]["github"],
                        "repo": None},
            "community": c["id"], "vision": c["modalities"]["image"]}


def community_packs(m: dict) -> list[dict]:
    """The model's community entries that are profiles of their own (community/<id>); not the ones
    one of the model's own profiles serves (VIS-175)."""
    return [c for c in m.get("community") or [] if c["profile"].startswith(COMMUNITY_PREFIX)]


def community(m: dict, cid: str) -> dict | None:
    """The model's community entry `cid` (its id, or its profile name community/<id>), when it is a
    profile of its own (community_packs)."""
    cid = cid[len(COMMUNITY_PREFIX):] if cid.startswith(COMMUNITY_PREFIX) else cid
    return next((c for c in community_packs(m) if c["id"] == cid), None)


def profiles(m: dict) -> list[dict]:
    """The model's own profiles, then its community packs as profiles."""
    return m["profiles"] + [community_profile(m, c) for c in community_packs(m)]


def json_schema() -> dict:
    """SCHEMA as models/schema/model.schema.json."""
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/0xBakeer/TandemLLM/models/schema/model.schema.json",
            "title": "TandemLLM model registry entry, schema 1 (models/<id>/model.yaml)", **SCHEMA}


def community_json_schema() -> dict:
    """COMMUNITY_SCHEMA as models/schema/community.schema.json."""
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/0xBakeer/TandemLLM/models/schema/community.schema.json",
            "title": "TandemLLM community weight pack, schema 1 (models/<model>/community/<id>.json)",
            **COMMUNITY_SCHEMA}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m server.setup.registry")
    ap.add_argument("--check", action="store_true", help="check model.yaml files (default: all)")
    ap.add_argument("--schema", action="store_true", help="print the JSON Schema")
    ap.add_argument("--community-schema", action="store_true",
                    help="print the community pack's JSON Schema")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args(argv)
    if a.schema:
        print(json.dumps(json_schema(), indent=2, sort_keys=True))
        return 0
    if a.community_schema:
        print(json.dumps(community_json_schema(), indent=2, sort_keys=True))
        return 0
    paths = a.paths or sorted(glob.glob(os.path.join(MODELS_DIR, "*", "model.yaml")))
    bad = 0
    for p in paths:
        try:
            m = load(p)
        except RegistryError as exc:
            bad += 1
            print(f"FAIL {p}")
            for line in exc.problems:
                print(f"  {line}")
            continue
        extra = unregistered_keys(m)
        if extra:
            bad += 1
            print(f"FAIL {p}")
            for line in extra:
                print(f"  {line}")
            continue
        print(f"ok   {p}: {m['id']}, {len(m['profiles'])} profiles, {len(m['components'])} tools")
    if not a.paths:
        models, _ = load_all()
        for n in unlisted_profiles(models):
            bad += 1
            print(f"FAIL ops/profiles/{n}.env: no model lists this profile")
        # the community packs, from their files (docs/community-weights.md), and the bundled index
        for p, problems in community_file_errors():
            bad += 1
            print(f"FAIL {os.path.relpath(p, os.path.dirname(MODELS_DIR))}")
            for line in problems:
                print(f"  {line}")
        n = len(glob.glob(os.path.join(MODELS_DIR, "*", "community", "*.json")))
        from server.setup import community as community_mod
        stale = []
        for path, built in ((community_mod.bundled_path(), False), (community_mod.builtin_path(), True)):
            try:
                same = community_mod.render_index(community_mod.build_index(builtin=built)) == \
                    open(path, encoding="utf-8").read()
            except (OSError, ValueError):
                same = False
            if not same:
                stale.append(os.path.basename(path))
        for name in stale:
            bad += 1
            print(f"FAIL models/{name} is not the folders' index: python -m server.setup.community --index")
        if not stale:
            print(f"ok   models/*/community: {n} packs, models/community-index.json matches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
