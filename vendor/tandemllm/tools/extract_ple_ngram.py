"""Lift the hashed n-gram embedding table out of a Flash-Next checkpoint and put it on disk.

The Flash-Next checkpoint carries a per-layer-embedding module whose input is not the token
embedding but a *hashed n-gram* table: every position is turned into sixteen row indices, the
sixteen rows are concatenated, and the result is a 2,560-wide feature describing the last two or
three tokens. The table is the largest single object in that checkpoint and it is split across a
hundred and twenty-eight shards, which makes it awkward to use for anything other than running
that model.

This tool writes it out once, in one flat row-major file per component, so that any process can
memory-map it and gather rows by index:

    table.u8.bin      [rows, 80]  uint8   two NVFP4 codes per byte, low nibble first
    scale.e4m3.bin    [rows, 10]  uint8   one e4m3 scale per group of sixteen codes
    meta.json                             hash parameters, shapes, provenance, dequant recipe

The hash parameters travel with the table because the table is useless without them: the row
index of a position is a function of the token ids, three 64-bit multipliers and sixteen primes,
and all of those live in the checkpoint beside the shards.

Nothing here is dequantised. Sixteen rows of NVFP4 are 1,280 bytes and sixteen rows of bf16 are
5,120; at 320 million rows that is the difference between 28.8 GB and 102.4 GB, and the gather is
sixteen random rows per position either way. `--dequant-sample` writes a small bf16 slice so the
recipe in `meta.json` can be checked against something.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")

# NVFP4 element codes, index = 4-bit pattern. e2m1: sign, two exponent bits, one mantissa bit.
FP4_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def find_table(index_path: str) -> tuple[str, dict[int, str], dict[str, str]]:
    """Return (module prefix, {shard index: file}, {aux name: file}) for the n-gram table."""
    with open(index_path) as handle:
        weight_map = json.load(handle)["weight_map"]
    shards: dict[int, str] = {}
    prefix = None
    for key, file in weight_map.items():
        match = SHARD_RE.search(key)
        if match is None:
            continue
        this_prefix = key[: match.start()] + ".ngram_embedding"
        if prefix is None:
            prefix = this_prefix
        elif prefix != this_prefix:
            raise SystemExit(f"two n-gram tables in one checkpoint: {prefix} and {this_prefix}")
        shards[int(match.group(1))] = file
    if prefix is None:
        return "", {}, {}
    parent = prefix.rsplit(".", 1)[0]
    aux = {}
    for name in (
        f"{prefix}.weight_scale_2",
        f"{parent}.ngram_heads_offsets",
        f"{parent}.ngram_heads_vocab_sizes",
        f"{parent}.layer_multipliers",
    ):
        if name in weight_map:
            aux[name] = weight_map[name]
    return prefix, shards, aux


def read_tensor(root: str, file: str, key: str) -> torch.Tensor:
    with safe_open(os.path.join(root, file), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="snapshot directory of the checkpoint")
    parser.add_argument("--out", required=True, help="directory to write the table into")
    parser.add_argument("--dequant-sample", type=int, default=0,
                        help="write this many leading rows as bf16, to check the recipe")
    parser.add_argument("--probe-only", action="store_true", help="report what is there, write nothing")
    args = parser.parse_args()

    index_path = os.path.join(args.checkpoint, "model.safetensors.index.json")
    prefix, shards, aux = find_table(index_path)
    if not shards:
        print(f"no hashed n-gram table in {args.checkpoint}")
        return 1
    print(f"table  {prefix}")
    print(f"shards {len(shards)}")

    config = json.load(open(os.path.join(args.checkpoint, "config.json")))
    text = config.get("text_config", config)
    parent = prefix.rsplit(".", 1)[0]
    offsets = read_tensor(args.checkpoint, aux[f"{parent}.ngram_heads_offsets"],
                          f"{parent}.ngram_heads_offsets").tolist()
    sizes = read_tensor(args.checkpoint, aux[f"{parent}.ngram_heads_vocab_sizes"],
                        f"{parent}.ngram_heads_vocab_sizes").tolist()
    multipliers = read_tensor(args.checkpoint, aux[f"{parent}.layer_multipliers"],
                              f"{parent}.layer_multipliers").tolist()
    global_scale = float(read_tensor(args.checkpoint, aux[f"{prefix}.weight_scale_2"],
                                     f"{prefix}.weight_scale_2").reshape(-1)[0])

    # Shape of one shard, read from the header of the first one.
    with safe_open(os.path.join(args.checkpoint, shards[0]), framework="pt", device="cpu") as handle:
        first = handle.get_slice(f"{prefix}.shard_0.weight")
        rows_per_shard, words = first.get_shape()
        scale_slice = handle.get_slice(f"{prefix}.shard_0.weight_scale")
        _, scale_groups = scale_slice.get_shape()

    num_heads = len(offsets)
    head_dim = words * 2
    total_rows = rows_per_shard * len(shards)
    used_rows = offsets[-1] + sizes[-1]
    group_size = head_dim // scale_groups

    print(f"rows/shard {rows_per_shard}  words {words} (= {head_dim} codes)  groups {scale_groups}"
          f" (= {group_size} codes/group)")
    print(f"heads {num_heads}  head_dim {head_dim}  ple_embed_dim {num_heads * head_dim}")
    print(f"rows total {total_rows}  addressable {used_rows}  pad {total_rows - used_rows}")
    print(f"ngram_size {text.get('ngram_size')}  heads_per_ngram {text.get('heads_per_ngram')}")
    print(f"multipliers {multipliers}")
    print(f"vocab sizes {sizes}")
    print(f"global scale {global_scale!r}")
    print(f"packed bytes {total_rows * words / 1e9:.2f} GB + scales {total_rows * scale_groups / 1e9:.2f} GB")
    if args.probe_only:
        return 0

    os.makedirs(args.out, exist_ok=True)
    table_path = os.path.join(args.out, "table.u8.bin")
    scale_path = os.path.join(args.out, "scale.e4m3.bin")
    table = np.lib.format.open_memmap(table_path + ".npy", mode="w+", dtype=np.uint8,
                                      shape=(total_rows, words))
    scales = np.lib.format.open_memmap(scale_path + ".npy", mode="w+", dtype=np.uint8,
                                       shape=(total_rows, scale_groups))
    started = time.time()
    for shard in range(len(shards)):
        file = shards[shard]
        with safe_open(os.path.join(args.checkpoint, file), framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(f"{prefix}.shard_{shard}.weight")
            scale = handle.get_tensor(f"{prefix}.shard_{shard}.weight_scale")
        lo = shard * rows_per_shard
        table[lo:lo + weight.shape[0]] = weight.numpy()
        scales[lo:lo + scale.shape[0]] = scale.view(torch.uint8).numpy()
        if shard % 16 == 0 or shard == len(shards) - 1:
            done = (shard + 1) / len(shards)
            print(f"  shard {shard:3d}  {done * 100:5.1f}%  {time.time() - started:6.1f}s", flush=True)
    table.flush()
    scales.flush()
    del table, scales

    meta = {
        "source": {
            "checkpoint": os.path.basename(os.path.realpath(args.checkpoint)),
            "model_type": config.get("model_type"),
            "module": prefix,
            "shards": len(shards),
        },
        "shape": {
            "rows_total": total_rows,
            "rows_addressable": used_rows,
            "words_per_row": words,
            "codes_per_row": head_dim,
            "scale_groups_per_row": scale_groups,
            "group_size": group_size,
        },
        "hash": {
            "ngram_size": text.get("ngram_size"),
            "heads_per_ngram": text.get("heads_per_ngram"),
            "num_heads": num_heads,
            "head_dim": head_dim,
            "ple_embed_dim": num_heads * head_dim,
            "layer_multipliers": multipliers,
            "head_offsets": offsets,
            "head_vocab_sizes": sizes,
            "eos_token_id": text.get("eos_token_id"),
            "vocab_size": text.get("vocab_size"),
        },
        "dequant": {
            "format": "nvfp4",
            "codes": "two 4-bit e2m1 codes per byte, low nibble is the even code",
            "code_values": FP4_VALUES.tolist()[:8] + FP4_VALUES.tolist()[8:],
            "value": "code_value * e4m3(scale[row, i // group_size]) * global_scale",
            "global_scale": global_scale,
        },
        "files": {
            "table": "table.u8.bin.npy",
            "scale": "scale.e4m3.bin.npy",
        },
    }
    with open(os.path.join(args.out, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)

    if args.dequant_sample:
        rows = dequant(args.out, np.arange(args.dequant_sample, dtype=np.int64))
        np.save(os.path.join(args.out, "sample-bf16.npy"), rows.astype(np.float32))
        print(f"sample {rows.shape} absmax {np.abs(rows).max():.4f} nonzero "
              f"{(rows != 0).mean() * 100:.1f}%")
    print(f"done in {time.time() - started:.1f}s")
    return 0


def dequant(out_dir: str, row_ids: np.ndarray) -> np.ndarray:
    """Gather rows and turn them into float32. The recipe meta.json describes, executed."""
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    table = np.load(os.path.join(out_dir, meta["files"]["table"]), mmap_mode="r")
    scales = np.load(os.path.join(out_dir, meta["files"]["scale"]), mmap_mode="r")
    group = meta["shape"]["group_size"]
    packed = np.asarray(table[row_ids])
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    values = FP4_VALUES[codes]
    raw = np.asarray(scales[row_ids])
    scale = torch.from_numpy(raw).view(torch.float8_e4m3fn).float().numpy()
    return values * np.repeat(scale, group, axis=1) * meta["dequant"]["global_scale"]


if __name__ == "__main__":
    sys.exit(main())
