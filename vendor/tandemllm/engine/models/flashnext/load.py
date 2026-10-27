"""Load a Flash-Next EXL3 pack onto the board, in the forms the engine multiplies by (ENG-219/221).

  W[name]      the reference's canonical names: EXL3Blocks (members of a group are views of it),
               plain tensors, mixer weights as `pack.Int8Rows` (or fp16 with QWEN38_FN_MIX=fp16)
  W[fused]     what the engine launches once instead of twice:
                 layers.L.linear_attn.qkvz        EXL3 group in_proj_qkv | in_proj_z
                 layers.L.linear_attn.ab          fp16 [2 Hv, H]  in_proj_a ; in_proj_b
                 layers.L.self_attn.qkv           EXL3 group q | k | v
                 <mixer>.down_inject              the down and inject rows of one mixer, one weight
                 layers.L.mlp.router              fp16 [E + 1, H]  gate ; shared_expert_gate
  banks[L]     the routed experts of layer L and its shared expert as expert E (tools.fn_kernels.MoEBank)
  tables[L]    the PLE n-gram table (memory-mapped, nothing resident)

The routed experts are 98 % of the bytes. A layer's expert tensors sit together in a shard (the
names sort by layer), so a layer is one read of its file range into host memory; its trellis, suh
and svh are copied from there into a 16-byte-aligned image (the kernels' cp.async needs it; the
4-byte mul1 tensors between them break the file's alignment), which goes to the device in one copy.
The MTP layer is not loaded here (ENG-153).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
import torch

from engine.models.flashnext.pack import TABLE_FILE, PackIndex, mixer_weight
from engine.settings import SETTINGS as _S

ALIGN = 16
#: drop the page cache of the shards once their tensors are on the device (posix_fadvise DONTNEED,
#: no root): the GPU's allocations do not wait for the kernel to reclaim cache, and ~50 GB of it
#: beside the weights is the 2026-09-25 NVRM out-of-memory (tools/drop_page_cache.py). The n-gram
#: table, which decode reads, is kept (QWEN38_FN_WARM_TABLE).
DROP_SHARD_CACHE = _S.get("FN_DROP_SHARD_CACHE") == "1"
#: read the n-gram table into the page cache at load (asynchronously; QWEN38_FN_WARM_TABLE=0: rows
#: come from the NVMe on first use)
WARM_TABLE = _S.get("FN_WARM_TABLE") == "1"
#: where the n-gram table's rows come from (ENG-239): warm = the page cache (warmed at load when
#: QWEN38_FN_WARM_TABLE=1, counted by the floor checks), cold = O_DIRECT reads through a hot-row cache of
#: QWEN38_FN_TABLE_CACHE_GB, never the page cache (packs whose weights and table do not fit together)
TABLE = (_S.get("FN_TABLE") or "warm").strip().lower()
TABLE_CACHE_GB = float(_S.get("FN_TABLE_CACHE_GB") or 2)


def _region_image(idx: PackIndex, names: list[str]) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """One file range read covering `names` (all in one shard), repacked at 16-byte offsets:
    (image uint8, {name: (byte offset in image, nbytes)})."""
    infos = [idx.tensors[n] for n in names]
    files = {t.file for t in infos}
    assert len(files) == 1, files
    lo = min(t.offset for t in infos)
    hi = max(t.offset + t.nbytes for t in infos)
    raw = np.fromfile(infos[0].file, dtype=np.uint8, count=hi - lo, offset=lo)
    total = sum(-(-t.nbytes // ALIGN) * ALIGN for t in infos)
    img = np.empty(total, dtype=np.uint8)
    where = {}
    o = 0
    for t in infos:
        img[o:o + t.nbytes] = raw[t.offset - lo:t.offset - lo + t.nbytes]
        where[t.name] = (o, t.nbytes)
        o += -(-t.nbytes // ALIGN) * ALIGN
    return img, where


def drop_cache(paths: list[str]) -> int:
    """posix_fadvise(DONTNEED) over each file; the bytes asked for (the cache dropped is at most that)."""
    n = 0
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += os.fstat(fd).st_size
        finally:
            os.close(fd)
    return n


def load_banks(idx: PackIndex, layer: int, device, prefix: str = "") -> "object":
    """The MoEBank of one layer: its E routed experts (one image per shard they live in) and the
    shared expert as expert E."""
    from tools.fn_kernels import MoEBank
    c = idx.cfg
    p = f"{prefix}layers.{layer}.mlp."
    by_file: dict[str, list[str]] = defaultdict(list)
    for e in range(c.num_experts):
        for x in ("gate", "up", "down"):
            for s in ("trellis", "suh", "svh"):
                n = f"{p}experts.{e}.{x}_proj.{s}"
                by_file[idx.tensors[n].file].append(n)
    dev_t: dict[str, torch.Tensor] = {}
    for _, names in sorted(by_file.items()):
        img, where = _region_image(idx, names)
        d = torch.from_numpy(img).to(device)
        for n, (o, nb) in where.items():
            t = idx.tensors[n]
            dt = torch.int16 if n.endswith(".trellis") else torch.float16
            dev_t[n] = d[o:o + nb].view(dt).view(t.shape)
    experts = []
    for e in range(c.num_experts):
        experts.append(tuple((dev_t[f"{p}experts.{e}.{x}_proj.trellis"], dev_t[f"{p}experts.{e}.{x}_proj.suh"],
                              dev_t[f"{p}experts.{e}.{x}_proj.svh"]) for x in ("gate", "up", "down")))
    sh = []
    for x in ("gate", "up", "down"):
        b = f"{p}shared_expert.{x}_proj."
        sh.append((idx.read(b + "trellis", device).contiguous(), idx.read(b + "suh", device),
                   idx.read(b + "svh", device)))
    experts.append(tuple(sh))
    return MoEBank(experts, device)


TILES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     "ops", "fn-exl3-tiles.json")


DENSE_SPLITS = os.path.join(os.path.dirname(TILES), "fn-dense-splits.json")


def apply_tiles(path: str = TILES, dense_path: str = DENSE_SPLITS) -> int:
    """Flash-Next's own EXL3 splits for its dense shapes: the library kernel's (tools/fn_tiles.py,
    prompt chunks) and tools/fn_kernels.dense's (tools/fn_tiles.py --dense, decode and verify). The
    27B's shapes are in neither file, so its table is untouched. Returns the entries applied."""
    import json
    from tools import fn_kernels as K
    from tools.exl3_linear import set_config
    n = 0
    if os.path.isfile(path):
        for k, v in json.load(open(path)).items():
            if k.startswith("_"):
                continue
            N, Kin, kb = (int(x) for x in k.split("x"))
            set_config(N, Kin, kb, {"S": int(v["S"]), "wk": int(v["wk"]), "cfg": int(v["cfg"]), "cfg2": int(v["cfg2"])})
            n += 1
    if os.path.isfile(dense_path):
        for k, v in json.load(open(dense_path)).items():
            if k.startswith("_"):
                continue
            N, Kin, kb = (int(x) for x in k.split("x"))
            K.DENSE_SPLIT[(N, Kin, kb)] = (int(v["S"]), int(v["warps"]))
            n += 1
    return n


def load(path: str, device="cuda", mix: str | None = None, log=print, layers=None, first: int | None = None,
         mtp: bool = False):
    """(idx, W, banks, tables, timing) for the trunk of the pack at `path`, and with `mtp` the MTP
    layer (its weights under `mtp.*`, its experts as banks[-1]; ENG-153). `first`: only the first N
    layers, with `idx.cfg` cut to them (a debug model for checks beside :8000)."""
    import dataclasses
    if first is not None:
        layers = range(first)
    from tools.exl3_linear import EXL3Block
    t0 = time.time()
    tiles = apply_tiles()
    idx = PackIndex.open(path)
    idx.audit(mtp=mtp)
    c = idx.cfg
    W: dict = {}
    timing = {"index_s": round(time.time() - t0, 2), "tiles": tiles}
    t1 = time.time()
    lay = range(c.num_hidden_layers) if layers is None else layers
    # every non-expert tensor of the trunk (and of the MTP layer)
    for n, t in idx.tensors.items():
        if (n.startswith("mtp.") and not mtp) or t.kind == "expert" or n.endswith(".mul1"):
            continue
        li = int(n.split(".")[1]) if n.startswith("layers.") else None
        if li is not None and li not in lay:
            continue
        if t.kind == "exl3":
            if not n.endswith(".trellis"):
                continue
            base = n[: -len(".trellis")]
            if ".shared_expert." in base:
                continue                                  # in the bank
            W[base + ".weight"] = idx.exl3(base, device, out_fp32=base == "lm_head")
        elif t.kind == "mixer":
            continue                                      # below, with its site
        else:
            W[n] = idx.read(n, device)
    timing["dense_s"] = round(time.time() - t1, 2)
    t1 = time.time()
    # mixers: down + inject as one weight; up alone
    sites = sorted({n.rsplit(".", 2)[0] for n, t in idx.tensors.items()
                    if t.kind == "mixer" and (mtp or not n.startswith("mtp."))
                    and (not n.startswith("layers.") or int(n.split(".")[1]) in lay)})
    for site in sites:
        dn = idx.read(site + ".input_mix_weight_down.weight", device)
        up = idx.read(site + ".input_mix_weight_up.weight", device)
        ij = site + ".block_inject_weight.weight"
        di = torch.cat([dn, idx.read(ij, device)]) if ij in idx.tensors else dn
        W[site + ".down_inject"] = mixer_weight(di.contiguous(), mix)
        W[site + ".input_mix_weight_up.weight"] = mixer_weight(up.contiguous(), mix)
        W[site + ".rank"] = dn.shape[0]
    timing["mixers_s"] = round(time.time() - t1, 2)
    # fused launches
    from engine.models.flashnext.ref import MTP
    for li in list(lay) + ([MTP] if mtp else []):
        p = "mtp.layers.0." if li == MTP else f"layers.{li}."
        if li != MTP and c.is_linear(li):
            a = p + "linear_attn."
            g = EXL3Block.group([W[a + "in_proj_qkv.weight"], W[a + "in_proj_z.weight"]], ["qkv", "z"])
            W[a + "qkvz"] = g
            W[a + "in_proj_qkv.weight"], W[a + "in_proj_z.weight"] = g.member(0), g.member(1)
            W[a + "ab"] = torch.cat([W[a + "in_proj_a.weight"], W[a + "in_proj_b.weight"]]).contiguous()
        else:
            a = p + "self_attn."
            g = EXL3Block.group([W[a + f"{x}_proj.weight"] for x in ("q", "k", "v")], ["q", "k", "v"])
            W[a + "qkv"] = g
            for i, x in enumerate(("q", "k", "v")):
                W[a + f"{x}_proj.weight"] = g.member(i)
        W[p + "mlp.router"] = torch.cat([W[p + "mlp.gate.weight"], W[p + "mlp.shared_expert_gate.weight"]]).contiguous()
    t1 = time.time()
    banks = {}
    for li in lay:
        banks[li] = load_banks(idx, li, device)
        if log and li % 8 == 7:
            log(f"[fn-load] experts through layer {li}: {time.time() - t1:.1f} s")
    if mtp:
        banks[MTP] = load_banks(idx, 0, device, prefix="mtp.")
    timing["experts_s"] = round(time.time() - t1, 2)
    tables = {li: idx.table for li in c.ple_layers} if idx.table is not None else {}
    if DROP_SHARD_CACHE:
        timing["shard_cache_dropped_gb"] = round(drop_cache([os.path.join(idx.path, f) for f in idx.files
                                                             if f != TABLE_FILE]) / 1e9, 1)
    if idx.table is not None:
        idx.table.set_policy(TABLE, int(TABLE_CACHE_GB * (1 << 30)))
        if TABLE == "cold" and not idx.table.direct.direct and log:
            log(f"[fn-load] {idx.table.path}: no O_DIRECT on this filesystem: buffered reads, the page cache dropped after each")
    if idx.table is not None and WARM_TABLE and TABLE == "warm":
        idx.table.warm()
    timing["table_warm"] = bool(idx.table is not None and WARM_TABLE and TABLE == "warm")
    timing["table_policy"] = TABLE
    if first is not None:
        idx.cfg = dataclasses.replace(c, num_hidden_layers=first, layer_types=c.layer_types[:first])
    timing["total_s"] = round(time.time() - t0, 2)
    if log:
        log(f"[fn-load] {path}: {timing}")
    return idx, W, banks, tables, timing
