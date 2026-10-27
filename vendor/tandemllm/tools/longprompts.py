"""Long held-out prompts for the prefill gates: real text, three domains, exact token lengths.

A prefill gate needs prompts that are long, that are not in anything this engine was built from,
and that have the STRUCTURE the model's own activations have. The last of those is not a formality:
phase 11 approved a matrix inversion on synthetic tensors of independent random keys, which are very
nearly orthogonal, and it produced NaN on the model within one layer. A synthetic tensor is a
hypothesis about the real one.

So every prompt here is real text, sliced to an exact token count, and taken from a place nothing in
this repository has read:

  prose    English Wikipedia, `/tmp/wt2.parquet` from row 3000 -- the drafter's suffix store took
           rows 0..2554 of that same file (`corpus/meta.json`), so the offset is the split
  german   German Wikipedia, `/tmp/de0.parquet` from row 1000 -- the store took rows 0..539
  code     C and C++ from a checkout of llama.cpp. Every line of code in the store and in
           `bench/calib.txt` is Python, and the quantiser was calibrated on Python; C++ is the
           domain none of that has seen

Documents are concatenated only until a prompt is full, so a prompt is one or two articles rather
than a hundred fragments, and no token is used by two prompts or by two lengths.

    python tools/longprompts.py --out bench/longprompts --lens 2048,8192,16384 --per-domain 7

Writes `ids-<len>.npy` of shape [N, len] int32 and `manifest.json` with the provenance, the domain
of every row and the SHA-256 of the token bytes, which is what a later run quotes to say it used
this set and not another.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402


def parquet_docs(path: str, column: str, skip: int):
    """Yield the text column of `path` from row `skip` on, one document at a time."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[column])
    col = table.column(column)
    for idx in range(skip, table.num_rows):
        text = col[idx].as_py()
        if text and len(text) > 200:
            yield text


def file_docs(patterns: list[str], skip: int = 0):
    """Yield whole source files, largest first, which is where the long ones are."""
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.expanduser(pat)))
    paths = sorted(set(paths), key=lambda p: -os.path.getsize(p))
    for path in paths[skip:]:
        try:
            with open(path, encoding="utf-8", errors="strict") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        if len(text) > 2000:
            yield text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="bench/longprompts")
    ap.add_argument("--lens", default="2048,8192,16384")
    ap.add_argument("--per-domain", type=int, default=7,
                    help="prompts per domain per length; three domains, so 7 gives 21")
    ap.add_argument("--prose", default="/tmp/wt2.parquet:text:3000",
                    help="PATH:COLUMN:SKIP_ROWS")
    ap.add_argument("--german", default="/tmp/de0.parquet:text:1000")
    ap.add_argument("--code", default="~/llama.cpp/src/*.cpp,~/llama.cpp/ggml/src/*.c,"
                                      "~/llama.cpp/common/*.cpp")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    cfg = load_config(a.model)
    tok = AutoTokenizer.from_pretrained(cfg.path)

    lens = [int(x) for x in a.lens.split(",")]
    sources = {}
    for name, spec in (("prose", a.prose), ("german", a.german)):
        path, column, skip = spec.split(":")
        sources[name] = (parquet_docs(path, column, int(skip)), f"{path} column {column} "
                                                                f"from row {skip}")
    sources["code"] = (file_docs(a.code.split(",")), a.code)

    os.makedirs(a.out, exist_ok=True)
    manifest = {"lens": lens, "per_domain": a.per_domain, "model": cfg.path,
                "sources": {k: v[1] for k, v in sources.items()}, "prompts": {}}

    # A per-domain buffer of tokens that survives across lengths, so the longest prompt of one
    # length never overlaps the shortest of the next.
    buf: dict[str, list[int]] = {k: [] for k in sources}
    for length in lens:
        rows, rowmeta = [], []
        for domain, (docs, _) in sources.items():
            for i in range(a.per_domain):
                while len(buf[domain]) < length:
                    try:
                        text = next(docs)
                    except StopIteration:
                        raise SystemExit(f"{domain} ran out of documents at {length} tokens, "
                                         f"prompt {i}: widen the source or lower --per-domain")
                    buf[domain].extend(tok(text, return_tensors=None)["input_ids"])
                ids = buf[domain][:length]
                buf[domain] = buf[domain][length:]
                rows.append(ids)
                digest = hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()[:16]
                rowmeta.append({"domain": domain, "i": i, "sha256_16": digest})
        arr = np.asarray(rows, dtype=np.int32)
        np.save(os.path.join(a.out, f"ids-{length}.npy"), arr)
        manifest["prompts"][str(length)] = rowmeta
        print(f"[longprompts] {length:>6} tokens  {arr.shape[0]} prompts  "
              f"{', '.join(sorted({r['domain'] for r in rowmeta}))}")

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[longprompts] wrote {a.out}/manifest.json")


if __name__ == "__main__":
    main()
