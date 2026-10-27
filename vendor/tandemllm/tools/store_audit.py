"""What the persistent suffix store holds of a benchmark's own prompts, and a copy without them.

the store (`~/.qwen38-spark-engine/suffix`, every server on the board appends to it) keeps
each request as one document -- the templated prompt's ids followed by the answer's -- and the atlas
row is fifty fixed prompts decoded greedily. This finds every document that contains one of a
dataset's prompts, says how many copies of each prompt and of each answer the store holds, and can
write a copy of the store with those documents removed: the store as real traffic built it, minus
the benchmark.

    # the audit, for the prompts a row actually sent (ids from its raw records)
    python tools/store_audit.py --dataset ~/inf-atlas/datasets/prompts-mixed-v1/prompts.jsonl \
        --row-records results/row3/base-0923

    # and a filtered copy that drops EVERY prompt of the dataset, row or not
    python tools/store_audit.py --dataset ... --write-filtered ~/qwen38-suffix-norow

A prompt is matched by the ids of its user text alone (the template around it is the same for every
request), taken from the middle so that a merge with the template's first or last token cannot hide
it. CPU only; the store is 45 MB.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

DOC_SEP = 1 << 30
SIG = 12                   # tokens of a prompt's middle that identify it


def split_docs(tokens: np.ndarray) -> list[np.ndarray]:
    """The store's documents, in order, separators dropped."""
    cuts = np.nonzero(tokens >= DOC_SEP)[0]
    out, start = [], 0
    for c in cuts.tolist():
        if c > start:
            out.append(tokens[start:c])
        start = c + 1
    if start < tokens.shape[0]:
        out.append(tokens[start:])
    return out


def signature(ids: list[int], n: int = SIG) -> tuple[int, ...]:
    """The middle `n` ids of a prompt's text (all but its first and last id when shorter)."""
    inner = ids[1:-1] if len(ids) > 4 else ids
    if len(inner) <= n:
        return tuple(inner)
    mid = (len(inner) - n) // 2
    return tuple(inner[mid:mid + n])


def find(doc: np.ndarray, sig: tuple[int, ...]) -> int:
    """Index of the first occurrence of `sig` in `doc`, or -1."""
    k = len(sig)
    if k == 0 or doc.shape[0] < k:
        return -1
    first = np.nonzero(doc[:doc.shape[0] - k + 1] == sig[0])[0]
    for i in first.tolist():
        if tuple(doc[i:i + k].tolist()) == sig:
            return i
    return -1


def _hashes(a: np.ndarray, k: int) -> np.ndarray:
    """A 64-bit rolling hash of every k-window of `a` (wrapping arithmetic; verified on a hit)."""
    if a.shape[0] < k:
        return np.zeros(0, dtype=np.uint64)
    x = a.astype(np.uint64)
    h = np.zeros(a.shape[0] - k + 1, dtype=np.uint64)
    base = np.uint64(1000003)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * base + x[j:j + h.shape[0]]
    return h


def match_docs(docs: list[np.ndarray], sigs: dict[str, tuple[int, ...]],
               k: int = 8) -> dict[int, str]:
    """Document index -> the prompt id whose signature it contains (the first one found)."""
    by_hash = defaultdict(list)
    for pid, sig in sigs.items():
        if len(sig) >= k:
            by_hash[int(_hashes(np.array(sig[:k], dtype=np.int64), k)[0])].append(pid)
    keys = np.array(sorted(by_hash), dtype=np.uint64)
    out = {}
    for di, doc in enumerate(docs):
        h = _hashes(doc, k)
        for i in np.nonzero(np.isin(h, keys))[0].tolist():
            for pid in by_hash[int(h[i])]:
                sig = sigs[pid]
                if tuple(doc[i:i + len(sig)].tolist()) == sig:
                    out[di] = pid
                    break
            if di in out:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="~/.qwen38-spark-engine/suffix")
    ap.add_argument("--dataset", required=True, help="an atlas prompts.jsonl")
    ap.add_argument("--row-records", default="",
                    help="a row3 label directory; its raw records name the prompts the row sent")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--write-filtered", default="",
                    help="write a copy of the store without any document holding a dataset prompt")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [json.loads(line) for line in open(os.path.expanduser(a.dataset))]
    text = {r["id"]: " ".join(m["content"] for m in r["messages"] if m["role"] == "user")
            for r in rows}
    sigs = {pid: signature(tok(t, add_special_tokens=False)["input_ids"]) for pid, t in text.items()}

    row_ids: list[str] = []
    if a.row_records:
        for f in glob.glob(os.path.join(os.path.expanduser(a.row_records), "raw-*", "**", "*.json"),
                           recursive=True):
            d = json.load(open(f))
            row_ids += [r["prompt_id"] for r in d["raw"]["payload"]["requests"]]
        row_ids = sorted(set(row_ids))

    path = os.path.join(os.path.expanduser(a.store), "tokens.bin")
    tokens = np.fromfile(path, dtype="<i4").astype(np.int64)
    docs = split_docs(tokens)
    hit = match_docs(docs, sigs)
    per = defaultdict(list)
    for di, pid in hit.items():
        per[pid].append(di)

    n_tok = sum(int(docs[di].shape[0]) for di in hit)
    print(f"[store] {path}: {tokens.shape[0]:,} tokens, {len(docs):,} documents")
    print(f"[store] {len(hit):,} documents hold one of the dataset's {len(sigs):,} prompts "
          f"({n_tok:,} tokens, {100.0 * n_tok / max(tokens.shape[0], 1):.1f} % of the store); "
          f"{len(per):,} distinct prompts")
    out = {"store_tokens": int(tokens.shape[0]), "documents": len(docs),
           "dataset_docs": len(hit), "dataset_tokens": n_tok, "distinct_prompts": len(per)}
    if row_ids:
        copies = [len(per.get(pid, [])) for pid in row_ids]
        # identical answers: the tail after the prompt's signature, compared across copies
        same = 0
        for pid in row_ids:
            tails = set()
            for di in per.get(pid, []):
                doc = docs[di]
                i = find(doc, sigs[pid])
                tails.add(tuple(doc[i + len(sigs[pid]):].tolist()))
            same += int(len(per.get(pid, [])) > 1 and len(tails) == 1)
        print(f"[row] {len(row_ids)} prompts the row sent: copies in the store min {min(copies)} "
              f"median {int(np.median(copies))} max {max(copies)}; prompts with zero copies "
              f"{sum(1 for c in copies if c == 0)}; prompts whose every stored copy has the same "
              f"answer {same} of {sum(1 for c in copies if c > 1)} with more than one copy")
        out["row"] = {"prompts": len(row_ids), "copies": dict(zip(row_ids, copies)),
                      "identical_answers": same}

    if a.write_filtered:
        dst = os.path.expanduser(a.write_filtered)
        os.makedirs(dst, exist_ok=True)
        keep = [d for di, d in enumerate(docs) if di not in hit]
        buf = np.concatenate([np.append(d, DOC_SEP) for d in keep]) if keep else np.zeros(0)
        buf.astype("<i4").tofile(os.path.join(dst, "tokens.bin"))
        meta = json.load(open(os.path.join(os.path.expanduser(a.store), "meta.json")))
        meta["filtered"] = {"from": path, "dropped_documents": len(hit),
                            "dropped_tokens": n_tok, "dataset": os.path.expanduser(a.dataset)}
        json.dump(meta, open(os.path.join(dst, "meta.json"), "w"))
        os.chmod(dst, 0o700)
        print(f"[filtered] {dst}: {int(buf.shape[0]):,} tokens, {len(keep):,} documents kept")
        out["filtered"] = {"path": dst, "tokens": int(buf.shape[0]), "documents": len(keep)}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
