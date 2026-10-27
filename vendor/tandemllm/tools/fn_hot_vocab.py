"""The hot vocabulary of Flash-Next's MTP draft head (ENG-153): token ids by corpus frequency.

    python tools/fn_hot_vocab.py --corpus ~/tandem-live/corpus/tokens.npy --pack ~/fn2/packs/3.05 \
        --out ops/fn-hot-vocab.npy

A draft only needs the target's likely next token, and the full head (248,320 rows, 0.4 GB at 5 bits)
is most of a draft step's bytes. `FlashNextEngine.set_draft_head(ids[:K])` drafts over the first K
ids of this list instead (int8 rows). The list: every added/special token of the pack's tokenizer
(the chat template's markers), then the corpus's token ids by descending count (ties by id), up to
`--size`. The corpus is the lookup drafter's (English and German wiki text, Python sources; the
prompt texts of bench/ are not in it). Only the drafts depend on it: the verify reads the served
head, so the output is the same whatever the list.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--size", type=int, default=65536)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.pack, "config.json")))
    vocab = (cfg.get("text_config") or cfg)["vocab_size"]
    tc = json.load(open(os.path.join(a.pack, "tokenizer_config.json")))
    special = sorted(int(k) for k in (tc.get("added_tokens_decoder") or {}) if int(k) < vocab)
    t = np.load(a.corpus, mmap_mode="r")
    t = np.asarray(t, dtype=np.int64)
    t = t[(t >= 0) & (t < vocab)]
    counts = np.bincount(t, minlength=vocab)
    order = np.lexsort((np.arange(vocab), -counts))
    seen = set(special)
    ids = list(special) + [int(i) for i in order if counts[i] > 0 and int(i) not in seen]
    ids = np.asarray(ids[:a.size], dtype=np.int32)
    np.save(a.out, ids)
    cover = {k: round(float(counts[ids[:k]].sum() / counts.sum()), 4) for k in (16384, 32768, 49152, 65536)
             if k <= len(ids)}
    about = {"_about": "hot vocabulary of the Flash-Next MTP draft head (tools/fn_hot_vocab.py)",
             "corpus": a.corpus, "corpus_tokens": int(counts.sum()), "special": len(special), "size": int(len(ids)),
             "corpus_coverage": cover}
    json.dump(about, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps(about))


if __name__ == "__main__":
    main()
