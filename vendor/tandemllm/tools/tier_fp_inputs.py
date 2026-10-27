"""Which inputs make two trees' disk-tier fingerprints differ? (CPU only, no engine, no GPU)

The tier's fingerprint is the engine code's hash and `fingerprint_extra` (server/diskcache.py): the
weight and drafter files, the QWEN38_* environment, the grid, the drafter and the library versions.
`--dump` rebuilds the extra a tree's server computes, from that tree's environment as ops/start.sh
turns it into arguments, and writes the inputs and the fingerprints. `--diff` then sets B's inputs
into A's one at a time, and one after another, and says which ones close the gap: the A/B for a
tier that a deploy swept.

    bash -c 'set -a; QSE_PROFILE_OVERRIDE=exl3-6.0; . ~/tandem-live/ops/serve.env; env -0' > live.env
    python tools/tier_fp_inputs.py --dump live.json --repo ~/tandem-live --env live.env
    python tools/tier_fp_inputs.py --dump rc.json --repo ~/tandem-rc080 --env rc.env
    python tools/tier_fp_inputs.py --diff live.json rc.json

`--dump` imports the tree's own server/diskcache.py and engine/disk_cache.py (run it once per tree,
each in its own process). `--diff` recomputes with A's tree and checks that A's own inputs give A's
fingerprint again before it trusts a substitution.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
import types


def read_env(path: str) -> dict:
    """`env -0` output (NUL-separated), or one KEY=VALUE a line."""
    with open(path, "rb") as f:
        raw = f.read()
    rows = raw.split(b"\0") if b"\0" in raw else raw.split(b"\n")
    out = {}
    for r in rows:
        k, sep, v = r.decode("utf-8", "replace").partition("=")
        if sep and k and not k.startswith("BASH_FUNC"):
            out[k] = v
    return out


def args_of(env: dict) -> types.SimpleNamespace:
    """The arguments fingerprint_extra reads, as ops/start.sh builds them from serve.env: a profile
    with QSE_ENGINE_FLAGS (Flash-Next) names its model and drafter, else the 27B's drafter, its two
    checkpoints and the NVFP4 and head files; the grid and the narrow arm at the server's defaults
    unless serve.env sets them."""
    flags = shlex.split(env.get("QSE_ENGINE_FLAGS") or "")
    a = types.SimpleNamespace(model=None, nvfp4=None, fp8_head=None, head=None, dflash2_ckpt=None,
                              dflash2_ckpt16=None, drafter="lenrouter", prefix_chunk=1024,
                              narrow_arm="off" if env.get("NARROW_ARM") == "0" else "on")
    if flags:
        for i, f in enumerate(flags[:-1]):
            if f == "--model":
                a.model = flags[i + 1]
            elif f == "--drafter":
                a.drafter = flags[i + 1]
    else:
        a.dflash2_ckpt, a.dflash2_ckpt16 = env.get("CKPT8"), env.get("CKPT16")
        a.nvfp4, a.fp8_head = env.get("NV"), env.get("HEAD")
    if env.get("PREFIX_CHUNK"):
        a.prefix_chunk = int(env["PREFIX_CHUNK"])
    return a


#: a model whose caches keep their own grid has the server set it before the tier starts
#: (server/app.py: `a.prefix_chunk = eng.caps.anchor_grid`); `--chunk` names it
GRIDS = {"mtp": 4096}


class _Environ:
    """os.environ as `env` while computing (fingerprint_extra reads the process environment)."""

    def __init__(self, env: dict):
        self.env = env

    def __enter__(self):
        self.saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(self.env)

    def __exit__(self, *exc):
        os.environ.clear()
        os.environ.update(self.saved)


def _modules(repo: str):
    sys.path.insert(0, repo)
    from engine import disk_cache as dc
    from engine import spill as sp
    from server import diskcache as sdc
    return dc, sp, sdc


def v2_of(dc, sp, repo: str, extra: dict) -> str:
    """0.7.2's tier fingerprint (server/diskcache.py up to 0.7.2), which the RC still computes to
    adopt those files."""
    return dc.fingerprint(repo, extra, skip=sp.NOT_ARITHMETIC)


def dump(out: str, repo: str, env_path: str, chunk: int = 0) -> dict:
    repo = os.path.abspath(os.path.expanduser(repo))
    dc, sp, sdc = _modules(repo)
    env = read_env(env_path)
    if "FUSE_PROJ" in env:                 # ops/start.sh exports it under the engine's name
        env["QWEN38_FUSE_PROJ"] = env["FUSE_PROJ"]
    a = args_of(env)
    a.prefix_chunk = chunk or GRIDS.get(a.drafter, a.prefix_chunk)
    by_path = getattr(sdc, "fingerprint_extra_paths", sdc.fingerprint_extra)
    with _Environ(env):
        extra = by_path(a)
        doc = {"repo": repo, "args": vars(a), "extra_paths": extra,
               "v2": v2_of(dc, sp, repo, extra),
               "code": dc.fingerprint(repo, {}, skip=sp.NOT_ARITHMETIC)[:16]}
        if by_path is not sdc.fingerprint_extra:
            content = sdc.fingerprint_extra(a)
            doc["extra"] = content
            doc["v2_content"] = v2_of(dc, sp, repo, content)
    with open(out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(f"{repo}: code {doc['code']}, tier v2 {doc['v2'][:16]}"
          + (f", by content {doc['v2_content'][:16]}" if "v2_content" in doc else ""))
    return doc


def _keys(extra: dict) -> list:
    keys = [("files", i) for i in range(len(extra.get("files") or []))]
    keys += [("env", k) for k in sorted(extra.get("env") or {})]
    keys += [(k, None) for k in sorted(extra) if k not in ("files", "env")]
    return keys


def _get(extra, key):
    k, sub = key
    if k == "files":
        f = extra.get("files") or []
        return f[sub] if sub < len(f) else None
    if k == "env":
        return (extra.get("env") or {}).get(sub)
    return extra.get(k)


def _set(extra, key, value):
    k, sub = key
    if k == "files":
        f = extra.setdefault("files", [])
        while len(f) <= sub:
            f.append(None)
        f[sub] = value
    elif k == "env":
        if value is None:
            extra.setdefault("env", {}).pop(sub, None)
        else:
            extra.setdefault("env", {})[sub] = value
    else:
        extra[k] = value


def diff(path_a: str, path_b: str, field: str = "extra_paths", fp: str = "v2") -> list:
    """The inputs where A and B differ; for each, A's fingerprint with B's value there alone, and
    with B's values set one after another up to it. Returns the rows."""
    with open(path_a) as f:
        A = json.load(f)
    with open(path_b) as f:
        B = json.load(f)
    dc, sp, _ = _modules(A["repo"])
    ea, eb = A[field], B[field]
    again = v2_of(dc, sp, A["repo"], ea)
    if again != A[fp]:
        raise SystemExit(f"A's own inputs give {again[:16]}, not its {A[fp][:16]}: not comparable")
    keys = sorted(set(_keys(ea)) | set(_keys(eb)), key=lambda k: (k[0], str(k[1])))
    differ = [k for k in keys if _get(ea, k) != _get(eb, k)]
    rows, mixed = [], copy.deepcopy(ea)
    print(f"A {A['repo']} {A[fp][:16]}  B {B['repo']} {B[fp][:16]}; code A {A['code']} B {B['code']}")
    for k in differ:
        one = copy.deepcopy(ea)
        _set(one, k, _get(eb, k))
        _set(mixed, k, _get(eb, k))
        alone, upto = v2_of(dc, sp, A["repo"], one), v2_of(dc, sp, A["repo"], mixed)
        row = {"input": f"{k[0]}[{k[1]}]" if k[1] is not None else k[0], "a": _get(ea, k),
               "b": _get(eb, k), "alone_equals_b": alone == B[fp], "so_far_equals_b": upto == B[fp]}
        rows.append(row)
        print(f"  {row['input']:<40} A {json.dumps(row['a'])[:70]}\n  {'':<40} B {json.dumps(row['b'])[:70]}"
              f"   alone -> B: {row['alone_equals_b']}, with the ones above -> B: {row['so_far_equals_b']}")
    if not differ:
        print("  no input differs")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--env", default="")
    ap.add_argument("--diff", nargs=2, default=None)
    ap.add_argument("--chunk", type=int, default=0, help="the grid, when serve.env does not set it")
    ap.add_argument("--field", default="extra_paths", help="extra_paths (the inputs 0.7.2 hashed) or extra")
    a = ap.parse_args()
    if a.diff:
        diff(*a.diff, field=a.field, fp="v2" if a.field == "extra_paths" else "v2_content")
    elif a.dump and a.repo and a.env:
        dump(a.dump, a.repo, a.env, a.chunk)
    else:
        ap.error("--dump OUT --repo TREE --env ENVFILE, or --diff A.json B.json")


if __name__ == "__main__":
    main()
