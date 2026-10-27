#!/usr/bin/env python3
"""Pack held-out text files into fixed token rows for exllamav3's eval/ppl.py (-t): a safetensors
file with "ids" [rows, length] int64 and "ranges" [rows, 2] (the whole row is scored). Each file
is tokenised on its own and cut into full rows; the remainder of a file is dropped, so a row never
straddles two texts and the token counts in the manifest are exact.

Optionally checks that the rows are disjoint from a calibration trace (the packed rows of
sc_trace.py): the share of 16-token windows of the held-out rows that also occur in the trace.
The texts are human-written and the trace is model-sampled, so the expected share is 0; a value
above 0.5 % means a text leaked into the calibration and the row set must not be used.
"""
import argparse, hashlib, json, os, sys
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="model directory with tokenizer.json")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="JSON with per-file token and row counts (default: <out>.json)")
    ap.add_argument("--disjoint-from", default=None, help="cal_trace.safetensors to check 16-token windows against")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("texts", nargs="+")
    a = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
    rows, manifest = [], {"length": a.length, "files": []}
    for path in a.texts:
        text = open(path, encoding="utf-8").read()
        ids = tok.encode(text, add_special_tokens=False).ids
        n_rows = len(ids) // a.length
        for r in range(n_rows):
            rows.append(torch.tensor(ids[r * a.length:(r + 1) * a.length], dtype=torch.long))
        manifest["files"].append({"file": os.path.basename(path), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                  "chars": len(text), "tokens": len(ids), "rows": n_rows,
                                  "dropped_tokens": len(ids) - n_rows * a.length})
        print(f"{os.path.basename(path)}: {len(ids)} tokens -> {n_rows} rows of {a.length}")
    if not rows:
        sys.exit("no full rows; texts too short for --length")
    ids = torch.stack(rows)
    ranges = torch.tensor([[0, a.length]] * ids.shape[0], dtype=torch.long)
    manifest["rows"] = int(ids.shape[0])
    manifest["tokens_scored"] = int(ids.shape[0] * (a.length - 1))

    if a.disjoint_from:
        # sc_trace.py writes "input_ids" [rows, cols] padded, plus "lengths" [rows]
        t = load_file(a.disjoint_from)
        trace = [r[:n] for r, n in zip(t["input_ids"].tolist(), t["lengths"].tolist())]
        seen = set()
        w = a.window
        for row in trace:
            for i in range(0, len(row) - w + 1):
                seen.add(hash(tuple(row[i:i + w])))
        hits = total = 0
        for row in ids.tolist():
            for i in range(0, len(row) - w + 1, w):
                total += 1
                hits += hash(tuple(row[i:i + w])) in seen
        share = hits / max(total, 1)
        manifest["overlap"] = {"trace": os.path.basename(a.disjoint_from), "window": w, "windows": total,
                               "hits": hits, "share": share}
        print(f"overlap with {os.path.basename(a.disjoint_from)}: {hits}/{total} windows of {w} ({share:.4%})")
        if share > 0.005:
            sys.exit("held-out rows overlap the calibration trace; refusing to write them")

    save_file({"ids": ids, "ranges": ranges}, a.out)
    json.dump(manifest, open(a.manifest or a.out + ".json", "w"), indent=1)
    print(f"wrote {a.out}: {tuple(ids.shape)}")


if __name__ == "__main__":
    main()
