"""Fingerprint v3 (ENG-243): what a model's stored rows depend on, as the model's adapter lists it.

The disk tier's fingerprint v2 (`engine/disk_cache.fingerprint`) hashes every file under `engine/`
but a few, so moving method code -- StairCut, the router, sampling, the caches, the server's loop --
sweeps every saved conversation, and it misses the tile tables that set the kernels' summation order
(a knob names the file; v2 hashed the name, never the table). v3 hashes the adapter's `manifest()`:

  modules   the files whose code computes stored rows (forward, verify and commit bodies, the
            drafters whose state is cached), and the `tools/` modules they import, transitively
  knobs     the defaults of the knobs those files own (engine/settings.py `KNOBS`); a value set in
            the environment is in the caller's `extra`, as under v2
  data      the tables those files read that set the arithmetic (tile tables), by content: the
            manifest's own, every .json a knob of theirs names, a module's default table
  extra     the weights, the environment, the grid, the libraries (server/diskcache.py)

Method code is not in it: it chooses which rows are computed, never how.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
from dataclasses import dataclass

from engine import disk_cache as dc

#: fp4 (rc-0.8.0): fp3's hash over the server's extra by content (server/diskcache.py
#: `fingerprint_extra`), where fp3's named files and tables by path; fp3 is still computed, by path,
#: to adopt the files the builds before wrote (`server/diskcache.py legacy_fingerprints`)
FORMULA = "fp4"

#: 0.8.2: module bytes that changed without changing a stored row on any profile the bytes they
#: replace served: (repo-relative path, sha256 of the bytes now) -> sha256 of the bytes replaced.
#: The fingerprints read the bytes as they are; the legacy ones (server/diskcache.py
#: `legacy_fingerprints`) also read them with the replaced bytes, and the first start under
#: `adoption()` adopts the files written under those, once (engine/disk_cache.py `equivalents`).
#: #1 (0xWhiteMage, a6f2142c): engine/loader.py loads a standalone EXL3 pack as --model (its
#: config.json says exl3 and no QWEN38_EXL3 overlay is set); a 27B profile sets QWEN38_EXL3 or serves
#: NVFP4 and takes the path it took before, reading the same tensors (the shards now opened at once)
#: ENG-285 (LFM2.5): the shared EXL3 kernels decode half-integer rates (1.5, 2.5, 3.5 bits) beside the whole
#: ones; a whole rate's code is the same (its tile width 8 K words, no new branch taken), and on the Spark
#: every dense, skinny-GEMM, slab, MoE and prefill-MoE output at rates 2..8 equalled the tree before's bit
#: for bit (the same random blocks and inputs through both trees)
#: SPD-125: tools/fn_kernels.py adds `dense_rows` (a verify block's EXL3 projections, opt-in) beside the
#: kernels it had; their code is the same, and on the Spark `dense` at 1..32 rows at Bolt's, the 27B's and
#: Flash-Next's shapes equalled 0.9.0-0.9.2's byte for byte (tools/fn_dense_rows_check.py same)
#: ENG-381 (ENG-363, 0.11.2): engine/models/lfm2_moe/engine.py keeps at most 32 joint decode graphs (which
#: graphs are kept, never what one computes) and kernels.py's `attn_rows` sizes its split partials for a
#: class's largest row count and reads the first rows' share (the same kernel and grid); on the Spark
#: `attn_rows` at 1..32 rows, two classes, lone and with shared buffers, equalled 0.11.1's bit for bit
#: (tools/lfm_attn_rows_check.py)
#: OPS-215 (ENG-383): tools/fn_kernels.py's router takes up to 512 experts, with a narrower instantiation for
#: up to 256; before it refused any count but 512, and at 512 it launches the same instantiation with the
#: same arguments, so no row a released profile stored changes (by the code; not run on the GPU)
#: Entries stay when their bytes are replaced again: a key may be another entry's replaced bytes, and
#: `same_variants` follows the chain (ENG-341)
SAME_ROWS = {("engine/loader.py", "823f9dd4c230ec45d133b52e0a5edc0343098184526d9dfcad5ee4271f933d97"):
             "fa9277e004043d172d29b2dc2b8b7958b4b7b2834bc26a73e6130bb0f0d05441",
             ("tools/exl3_linear.py", "a75b6a3f1fd36ff9ecafe79c09cb797106ce48f53edc73b10d4ef19a0fe3c40f"):
             "5078b33c3d03eac42706d3b05650b0604f2ae96d267db2c996f59409929a3a48",
             ("tools/fn_kernels.py", "152025a29380f1bc9c0a1d93b2382531260b443cbe542549f332fa245ac23b7f"):
             "26139deda6a564355375158270ddedd8ba7acbbf16c97aaf2b5646c6c8678e93",
             ("tools/fn_kernels.py", "26139deda6a564355375158270ddedd8ba7acbbf16c97aaf2b5646c6c8678e93"):
             "630236e58c8230010d683b22604bfb97083ad932bfa36847ab048ef0ec5af55a",
             ("tools/fn_kernels.py", "85e31c84a0709bfaec8f40c195354f29c225ed72defe4df85a5ad0a175a985e8"):
             "152025a29380f1bc9c0a1d93b2382531260b443cbe542549f332fa245ac23b7f",
             ("engine/models/lfm2_moe/engine.py", "76bfb598fee3b1d731a51bd4f4ceb4afc8bb30ee8388ed4ca9551de82d7d388b"):
             "2360b32729713510020bd47ed86e6c185a40f54b22c622fc33c4fa7e6646783e",
             ("engine/models/lfm2_moe/kernels.py", "8f2e42e559a134a9992729497524b6fa5d57f6cb08378e56ed3d0b4fa8f02dea"):
             "dbfbfd3aa2cc532ecca1d2ce0b5f23d6d48d68c7bfea6463298113fe8a49d754",
             ("tools/fn_prefill_kernels.py", "653b447087545778f29eb856e12681d0d1229847ada9ec7291477aeda8c54106"):
             "13fd5c1579579b11f94af618835798548216881173821cbb6447b3e6b21ec987",
             # SRV-619: the answer constraint's union and the call markers; a mask on the sampled
             # token, never a stored row
             ("engine/grammar.py", "ddcb9cf828c5e89c80a29c4ff25b20fac4a5d6230bfa4248a70893a94a97778b"):
             "c7fbee9ad0f1526b7c30df798e8c178c3a2185244ab7a2b29e44eed5ffe50749",
             # SRV-647: the masks keyed by a serial of the grammar, not its id(); no row changes
             ("engine/grammar.py", "9c6617a799bd64b1fdbc3cfc23a2eba5b2ee471b64d1d21b4da7dc23adf59c65"):
             "c7fbee9ad0f1526b7c30df798e8c178c3a2185244ab7a2b29e44eed5ffe50749"}


def adoption() -> str:
    """The key `equivalents` records an adoption under: the formula, and `SAME_ROWS`' entries, so
    a tree that adds one adopts once more on a directory that already adopted under the formula."""
    if not SAME_ROWS:
        return FORMULA
    rows = sorted([k[0], k[1], v] for k, v in SAME_ROWS.items())
    return f"{FORMULA}+same-{hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:12]}"


@dataclass(frozen=True)
class Manifest:
    arch: str                   # the registry name (engine/models: qwen3_5, qwen4_exp)
    modules: tuple              # repo-relative files that compute stored rows
    data: tuple = ()            # tables they read that set the arithmetic (repo-relative or absolute)


def files(root: str, m: Manifest) -> list[str]:
    """The manifest's modules and the `tools/` modules they import (absolute paths, in order)."""
    mods = [os.path.join(root, p) for p in m.modules]
    return mods + dc._tools_modules(root, mods)


def same_variants(root: str, m: Manifest) -> list[dict]:
    """`same` tables, one a combination of the bytes `m`'s files had (ENG-341): each file whose bytes
    `SAME_ROWS` names as its own bytes or as any bytes they replaced, down the chain, the others as
    they are. `SAME_ROWS` itself reads every named file one step back, which is no release's tree once
    a release replaced one file's bytes and kept another's (0.9.x: fn_kernels.py's 0.9 bytes beside
    exl3_linear.py's, which replaced 0.8.2's); one of these tables is that tree's. Without the one
    that changes nothing."""
    choices = []
    for p in files(root, m):
        rel = os.path.relpath(p, root)
        if rel == dc.SETTINGS_FILE:
            continue
        with open(p, "rb") as f:
            now = hashlib.sha256(f.read()).hexdigest()
        olds, cur = [], now
        while (rel, cur) in SAME_ROWS and SAME_ROWS[(rel, cur)] not in olds:
            cur = SAME_ROWS[(rel, cur)]
            olds.append(cur)
        if olds:
            choices.append([None] + [((rel, now), old) for old in olds])
    return [dict(c for c in combo if c) for combo in itertools.product(*choices)][1:]


#: tables a module reads when its knob names none (the knob's value, when set, is read instead)
DEFAULT_TABLES = {"tools/exl3_linear.py": ("EXL3_TILES", os.path.join("ops", "exl3-tiles-fused.json"))}


def _knob_table(settings_path: str) -> dict:
    with open(settings_path, "rb") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if isinstance(target, ast.Name) and target.id == "KNOBS" and node.value is not None:
            return ast.literal_eval(node.value)
    raise ValueError(f"{settings_path} has no literal KNOBS table")


def knobs_of(settings_path: str, owners: set) -> bytes:
    """(name, default) of every knob in `settings_path` whose owner is one of `owners`, in name order."""
    knobs = _knob_table(settings_path)
    return json.dumps(sorted((str(k), v[0]) for k, v in knobs.items()
                             if len(v) > 1 and str(v[1]) in owners), default=str).encode()


def tables(root: str, owners: set, env=None) -> list[str]:
    """The tables the owners read: a knob of theirs whose value (the environment's, else its default)
    names a .json file, and a module's own default table when its knob is unset (`DEFAULT_TABLES`)."""
    env = os.environ if env is None else env
    knobs = _knob_table(os.path.join(root, dc.SETTINGS_FILE))
    out = set()
    for k, v in knobs.items():
        if len(v) > 1 and str(v[1]) in owners:
            val = env.get("QWEN38_" + k, v[0])
            if isinstance(val, str) and val.endswith(".json"):
                out.add(val)
    for mod, (k, path) in DEFAULT_TABLES.items():
        if mod in owners and not env.get("QWEN38_" + k, knobs.get(k, (None,))[0]):
            out.add(path)
    return sorted(out)


def fingerprint(root: str, m: Manifest, extra: dict, formula: str = FORMULA,
                same: dict | None = None) -> str:
    """`same` (`SAME_ROWS`): a module whose bytes it names counts as the bytes it replaced."""
    h = hashlib.sha256()
    h.update(formula.encode())
    h.update(m.arch.encode())
    paths = files(root, m)
    owners = {os.path.relpath(p, root) for p in paths}
    for p in paths:
        rel = os.path.relpath(p, root)
        h.update(rel.encode())
        if rel == dc.SETTINGS_FILE:
            data = knobs_of(p, owners)
        else:
            with open(p, "rb") as f:
                data = f.read()
        h.update(dc.module_digest(rel, data, same))
    for d in sorted(set(m.data) | set(tables(root, owners))):
        full = os.path.expanduser(d) if os.path.isabs(os.path.expanduser(d)) else os.path.join(root, d)
        rel = os.path.relpath(full, root)
        h.update((full if rel.startswith("..") else rel).encode())
        try:
            with open(full, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        except OSError:
            h.update(b"missing")
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()
