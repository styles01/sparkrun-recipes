"""A hashed n-gram successor table for the 27 B, built in one pass and scored where it matters.

The engram this track is pricing is a hashed n-gram memory with a learned head. Before spending
board hours recording labels for that head, there is a cheaper object that answers the question the
head would answer: a *count* table over the same hash. For every order `n` in 2..5 the last `n`
tokens are hashed into one of `R` buckets and the bucket keeps the few successors seen most often
after that hash. Prediction backs off from the longest order that has a bucket with enough support.

A count table is not the final design, but it is the right first measurement, for two reasons. It
is the best a predictor can do from n-gram counts alone up to smoothing, so if it cannot clear the
bar a learned head over the same features will not either. And it costs one CPU pass over a token
stream instead of sixteen board-hours of recording.

WHAT IS SCORED, AND WHY IT IS NOT ACCURACY
------------------------------------------
A speculative block ends at the first position the drafter gets wrong, and the positions that end
blocks are the ones where the target itself is unsure. Reporting a predictor's top-1 accuracy over
all positions therefore says almost nothing: most positions are easy, every predictor gets them,
and the block still ends where it always ended. So everything here is stratified by the target's own
top-1 probability, and the column that decides the design is the one for positions below `p1 = 0.5`.

Two token streams can train it, and telling them apart is the point of the exercise:

    --corpus       raw public text. Free, no board, already on the NVMe. Trains the predictor to
                   say what the TEXT does next.
    --labelled     recorded sequences carrying the target's own argmax per position. Expensive.
                   Trains the predictor to say what the TARGET does next.

The difference between the two, measured on the same held-out set, is what a labelled token is
worth.
"""

from __future__ import annotations

import argparse
import glob
import json
import hashlib
import os
import sys
import time

import numpy as np
import torch

# Odd 64-bit multipliers, one per n-gram slot, applied to the token at that distance and XORed.
# Same construction as the published table's, with its three constants first so an order-3 bucket
# of this table and one of that table are the same function of the tokens up to the modulus.
MULTIPLIERS = np.array(
    [23703573157769, 20109073645365, 8052911324071, 15971315351597, 11400714819323],
    dtype=np.int64,
)
ORDERS = (2, 3, 4, 5)

# `tools/build_corpus.py` writes this id between documents so that no n-gram spans two of them.
DOC_SEP = 1073741824


def next_prime(value: int) -> int:
    def is_prime(v: int) -> bool:
        if v < 2:
            return False
        if v % 2 == 0:
            return v == 2
        d = 3
        while d * d <= v:
            if v % d == 0:
                return False
            d += 2
        return True
    while not is_prime(value):
        value += 1
    return value


def in_document(tokens: np.ndarray, order: int) -> np.ndarray:
    """True where the `order`-gram ending at the position lies inside one document.

    The corpus is a concatenation with a separator id between documents. An n-gram that spans one
    is a fact about nothing, and a count table that stores those facts answers with them.
    """
    ok = tokens != DOC_SEP
    out = ok.copy()
    for slot in range(1, order):
        shifted = np.zeros_like(ok)
        shifted[slot:] = ok[:-slot]
        out &= shifted
    return out


def mix(tokens: np.ndarray, order: int) -> np.ndarray:
    """The 64-bit hash of the `order`-gram ending at each position."""
    mixed = tokens.astype(np.int64) * MULTIPLIERS[0]
    for slot in range(1, order):
        shifted = np.empty_like(mixed)
        shifted[slot:] = tokens[:-slot].astype(np.int64) * MULTIPLIERS[slot]
        shifted[:slot] = 0
        mixed = np.bitwise_xor(mixed, shifted)
    return mixed


def buckets(tokens: np.ndarray, order: int, modulus: int) -> np.ndarray:
    """Bucket of the `order`-gram ENDING at each position. Positions before `order - 1` are -1."""
    out = np.remainder(mix(tokens, order), modulus)
    out[: order - 1] = -1
    return out


def fingerprints(tokens: np.ndarray, order: int) -> np.ndarray:
    """A 16-bit tag of the same n-gram, from bits the bucket index does not use.

    Without it a count table answers for contexts it has never seen: at 4 million buckets and 5
    million training n-grams, most buckets are occupied, so an unseen context lands on somebody
    else's counts and the table emits them with full confidence. The tag rejects 65,535 of every
    65,536 such collisions for two bytes a bucket.
    """
    return ((mix(tokens, order) >> np.int64(40)) & np.int64(0xFFFF)).astype(np.uint16)


class HashGram:
    """Per order: a [rows, slots] table of successor ids and one of counts, most frequent first."""

    def __init__(self, rows: int, slots: int, vocab: int):
        self.modulus = next_prime(rows)
        self.slots = slots
        self.vocab = vocab
        self.base = vocab
        self.succ: dict[int, np.ndarray] = {}
        self.count: dict[int, np.ndarray] = {}
        self.tag: dict[int, np.ndarray] = {}
        self.seen: dict[int, int] = {}

    def fit(self, streams: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Streams: list of (context tokens, label per position). Label -1 means "skip"."""
        # The (bucket, successor) pair is packed into one int64 so that one sort counts them all.
        # The base has to exceed every id actually present, not the config's vocabulary size: a
        # tokenizer that emits an id above `vocab_size` would silently carry a count into the next
        # bucket, which is a corruption no later assertion would catch.
        self.base = self.vocab
        for order in ORDERS:
            keys = []
            tags = []
            for tokens, labels in streams:
                bucket = buckets(tokens, order, self.modulus)
                valid = (bucket >= 0) & (labels >= 0) & (labels < self.base) & in_document(tokens, order)
                keys.append(bucket[valid] * self.base + labels[valid])
                tags.append(bucket[valid] * 65536 + fingerprints(tokens, order)[valid].astype(np.int64))
            # The tag a bucket answers for is the one most n-grams in it carry; the rest collide
            # and are rejected at lookup, which is the point.
            tag_key = np.concatenate(tags)
            tag_key.sort()
            tag_u, tag_c = np.unique(tag_key, return_counts=True)
            tag_bucket = tag_u // 65536
            by_count = np.lexsort((-tag_c, tag_bucket))
            tag_bucket, tag_value = tag_bucket[by_count], (tag_u % 65536)[by_count]
            first = np.r_[True, tag_bucket[1:] != tag_bucket[:-1]]
            tag_table = np.zeros(self.modulus, dtype=np.uint16)
            owner = np.zeros(self.modulus, dtype=bool)
            tag_table[tag_bucket[first]] = tag_value[first].astype(np.uint16)
            owner[tag_bucket[first]] = True
            self.tag[order] = tag_table
            del tag_key, tag_u, tag_c, tag_bucket, tag_value, by_count, first
            key = np.concatenate(keys)
            key.sort()
            uniq, counts = np.unique(key, return_counts=True)
            bucket = uniq // self.base
            token = (uniq % self.base).astype(np.int32)
            # Within a bucket, keep the `slots` most frequent successors.
            order_by = np.lexsort((-counts, bucket))
            bucket, token, counts = bucket[order_by], token[order_by], counts[order_by]
            rank = np.arange(len(bucket)) - np.repeat(
                np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]]),
                np.diff(np.r_[np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]]), len(bucket)]),
            )
            keep = rank < self.slots
            succ = np.zeros((self.modulus, self.slots), dtype=np.int32)
            cnt = np.zeros((self.modulus, self.slots), dtype=np.int32)
            succ[bucket[keep], rank[keep]] = token[keep]
            cnt[bucket[keep], rank[keep]] = counts[keep]
            # Counts that belong to a context whose tag lost the bucket are not this bucket's.
            self.succ[order] = succ
            self.count[order] = cnt
            self.count[order][~owner] = 0
            self.seen[order] = int(len(uniq))

    def predict(self, tokens: np.ndarray, top: int, min_count: int) -> tuple[np.ndarray, np.ndarray]:
        """Back off from order 5 to 2. Returns (top predictions [n, top], order used [n])."""
        n = len(tokens)
        out = np.full((n, top), -1, dtype=np.int32)
        used = np.zeros(n, dtype=np.int8)
        remaining = np.ones(n, dtype=bool)
        for order in sorted(ORDERS, reverse=True):
            if not remaining.any():
                break
            bucket = buckets(tokens, order, self.modulus)
            ok = remaining & (bucket >= 0)
            if not ok.any():
                continue
            rows = bucket[ok]
            cnt = self.count[order][rows]
            tag = fingerprints(tokens, order)[ok]
            hit = ((cnt[:, 0] >= min_count) & in_document(tokens, order)[ok]
                   & (self.tag[order][rows] == tag))
            idx = np.flatnonzero(ok)[hit]
            out[idx] = self.succ[order][rows[hit], :top]
            used[idx] = order
            remaining[idx] = False
        return out, used

    def nbytes(self) -> int:
        return sum(self.succ[o].nbytes + self.count[o].nbytes + self.tag[o].nbytes
                   for o in ORDERS)


def read_corpus(path: str, limit: int) -> np.ndarray:
    tokens = np.load(os.path.join(path, "tokens.npy"), mmap_mode="r")
    if limit and limit < len(tokens):
        tokens = tokens[:limit]
    return np.asarray(tokens).astype(np.int32)


def read_labelled(paths: list[str], as_text: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """The recorded sequences, either with the target's own argmax as the label or with the text's
    own next token.

    The two differ, and how much is the whole question. `--labelled-as-text` is the control the
    label-value measurement needs: the same tokens, the same count, the same domain, and the only
    change is what the predictor is asked to say.
    """
    out = []
    for path in paths:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        ids = raw["ids"].numpy().astype(np.int32)
        if as_text:
            label = np.empty_like(ids)
            label[:-1] = ids[1:]
            label[-1] = -1
        else:
            label = raw["label"].numpy().astype(np.int32)
        out.append((ids, label))
    return out


def read_eval(paths: list[str]) -> list[dict]:
    out = []
    for path in paths:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        out.append({
            "ids": raw["ids"].numpy().astype(np.int32),
            "label": raw["label"].numpy().astype(np.int32),
            "p1": np.exp(raw["top_lp"][:, 0].float().numpy().astype(np.float64)),
            "gen_start": int(raw.get("gen_start", 0)),
            "name": raw.get("name", os.path.basename(path)),
        })
    return out


BANDS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]


def consistency(streams: list[tuple[np.ndarray, np.ndarray]], vocab: int) -> None:
    """How often two occurrences of the same n-gram are followed by the same target token.

    This is the question underneath every table size and every corpus in this track, and it does not
    need a model. If the same `n` tokens are followed by a different argmax the second time, no
    predictor whose input is those `n` tokens can get both right, and the gap is not data, table or
    architecture. Reported for contexts that occur at least twice, weighted by occurrence, so a
    context seen a hundred times counts for what it is worth.
    """
    print(f"\n{'order':>6}{'positions':>11}{'repeated ctx':>14}{'share repeated':>16}"
          f"{'same target':>13}{'ceiling':>9}")
    for order in (1, 2, 3, 4, 5):
        keys = []
        for tokens, labels in streams:
            mixed = mix(tokens, order)
            valid = (labels >= 0) & (labels < vocab) & in_document(tokens, order)
            valid[: order - 1] = False
            # Pack the 64-bit context hash with the label. Collisions between distinct contexts are
            # possible and rare; they can only make the measured consistency LOOK worse.
            ctx = (mixed[valid] >> np.int64(8)).astype(np.int64)
            keys.append(np.stack([ctx, labels[valid].astype(np.int64)]))
        arr = np.concatenate(keys, axis=1)
        ctx, lab = arr[0], arr[1]
        order_by = np.argsort(ctx, kind="stable")
        ctx, lab = ctx[order_by], lab[order_by]
        starts = np.flatnonzero(np.r_[True, ctx[1:] != ctx[:-1]])
        sizes = np.diff(np.r_[starts, len(ctx)])
        repeated = sizes > 1
        # For each repeated context, the share of its occurrences that carry its most common label.
        best = 0
        total = 0
        for start, size in zip(starts[repeated], sizes[repeated]):
            labels_here = lab[start:start + size]
            _, counts = np.unique(labels_here, return_counts=True)
            best += int(counts.max())
            total += int(size)
        share = total / len(ctx)
        same = best / total if total else float("nan")
        # A predictor that is perfect on unseen contexts is still wrong on 1 - same of the repeated
        # ones, so this is an upper bound on any n-gram predictor's accuracy on this traffic.
        print(f"{order:>6}{len(ctx):>11,}{int(repeated.sum()):>14,}{share * 100:>15.1f}%"
              f"{same * 100:>12.1f}%{same * 100:>8.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="", help="directory with tokens.npy")
    parser.add_argument("--corpus-tokens", type=int, default=0, help="use at most this many")
    parser.add_argument("--labelled", default="", help="glob of recorded .pt sequences to TRAIN on")
    parser.add_argument("--eval", required=True, help="glob of recorded .pt sequences to SCORE on")
    parser.add_argument("--eval-holdout", type=int, default=0,
                        help="score on the last N eval files only, train on the rest if --labelled is 'eval'")
    parser.add_argument("--rows", type=int, default=1 << 22)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--top", type=int, default=1, help="candidates the drafter would emit")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--tag", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--consistency", action="store_true",
                        help="report how often the same n-gram is followed by the same target token, "
                             "and stop. No model, no table, no corpus")
    parser.add_argument("--labelled-as-text", action="store_true",
                        help="use the text's own next token instead of the target's argmax. The "
                             "control for what a label is worth")
    parser.add_argument("--holdout-topics", default="",
                        help="comma-separated topics to score on; every other topic trains. A split "
                             "by sequence still shares the prompt templates, so this is the honest one")
    args = parser.parse_args()

    eval_files = sorted(glob.glob(args.eval))
    if not args.keep_duplicates:
        unique, seen = [], set()
        for path in eval_files:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            digest = hashlib.sha1(raw["ids"].numpy().tobytes()).digest()
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(path)
        if len(unique) != len(eval_files):
            print(f"[hashgram] {len(eval_files)} sequences, {len(unique)} distinct; "
                  f"dropped {len(eval_files) - len(unique)} repeats before splitting")
        eval_files = unique
    if args.holdout_topics:
        wanted = set(args.holdout_topics.split(","))
        train_files, held = [], []
        for path in eval_files:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            (held if raw.get("topic") in wanted else train_files).append(path)
        eval_files = held
        print(f"[hashgram] topic split: {len(train_files)} train, {len(eval_files)} held-out "
              f"({args.holdout_topics})")
    elif args.eval_holdout:
        train_files, eval_files = eval_files[:-args.eval_holdout], eval_files[-args.eval_holdout:]
    else:
        train_files = []
    evals = read_eval(eval_files)

    streams: list[tuple[np.ndarray, np.ndarray]] = []
    corpus_tokens = 0
    if args.corpus:
        tokens = read_corpus(args.corpus, args.corpus_tokens)
        corpus_tokens = len(tokens)
        labels = np.empty_like(tokens)
        labels[:-1] = tokens[1:]
        labels[-1] = -1
        streams.append((tokens, labels))
    labelled_tokens = 0
    sources = []
    if args.labelled == "eval":
        sources = train_files
    elif args.labelled:
        sources = sorted(glob.glob(args.labelled))
    if sources:
        for ids, label in read_labelled(sources, args.labelled_as_text):
            streams.append((ids, label))
            labelled_tokens += len(ids)
    if not streams:
        print("nothing to train on")
        return 1

    if args.consistency:
        consistency(streams, args.vocab)
        return 0

    model = HashGram(args.rows, args.slots, args.vocab)
    started = time.time()
    model.fit(streams)
    fit_seconds = time.time() - started
    total = corpus_tokens + labelled_tokens
    print(f"tag {args.tag or '-'}")
    print(f"train  corpus {corpus_tokens:,}  labelled {labelled_tokens:,}  "
          f"rows {model.modulus:,}  slots {args.slots}  table {model.nbytes() / 1e6:.0f} MB  "
          f"fit {fit_seconds:.1f}s ({total / max(fit_seconds, 1e-9) / 1e6:.1f} M tok/s)")
    print("distinct (bucket, successor) pairs per order: "
          + "  ".join(f"n={o}:{model.seen[o]:,}" for o in ORDERS))

    ids, labels, p1s, orders = [], [], [], []
    for seq in evals:
        lo = seq["gen_start"]
        # The n-gram context may reach back into the prompt, so predict over the whole sequence and
        # score only the generated part.
        pred, used = model.predict(seq["ids"], args.top, args.min_count)
        ids.append(pred[lo:])
        labels.append(seq["label"][lo:])
        p1s.append(seq["p1"][lo:])
        orders.append(used[lo:])
    pred = np.concatenate(ids)
    label = np.concatenate(labels)
    p1 = np.concatenate(p1s)
    used = np.concatenate(orders)
    fired = used > 0
    hit1 = pred[:, 0] == label
    hitk = (pred == label[:, None]).any(axis=1)

    print(f"\nscored on {len(eval_files)} held-out sequences, {len(label):,} generated positions")
    print(f"{'target p1 band':<16}{'share':>8}{'fire':>8}{'top-1':>9}{'top-' + str(args.top):>9}"
          f"{'top-1|fire':>12}{'mean p1':>9}")
    rows = {}
    for lo, hi in BANDS:
        band = (p1 >= lo) & (p1 < hi)
        if not band.any():
            continue
        f = band & fired
        cond = hit1[f].mean() if f.any() else float("nan")
        print(f"{f'{lo:.2f} - {hi:.2f}':<16}{band.mean() * 100:>7.1f}%{fired[band].mean() * 100:>7.1f}%"
              f"{hit1[band].mean() * 100:>8.1f}%{hitk[band].mean() * 100:>8.1f}%{cond * 100:>11.1f}%"
              f"{p1[band].mean():>9.3f}")
        rows[f"{lo}-{hi}"] = {"share": float(band.mean()), "fire": float(fired[band].mean()),
                              "top1": float(hit1[band].mean()), "topk": float(hitk[band].mean())}
    print(f"{'all':<16}{100.0:>7.1f}%{fired.mean() * 100:>7.1f}%{hit1.mean() * 100:>8.1f}%"
          f"{hitk.mean() * 100:>8.1f}%{hit1[fired].mean() * 100:>11.1f}%{p1.mean():>9.3f}")
    print("order used when it fired: " + "  ".join(
        f"n={o}:{(used == o).mean() * 100:.1f}%" for o in ORDERS))
    unsure = p1 < 0.5
    print(f"\nTHE COLUMN THAT DECIDES IT -- positions where the target is unsure (p1 < 0.5): "
          f"{unsure.mean() * 100:.1f}% of positions, engram top-1 {hit1[unsure].mean() * 100:.1f}%, "
          f"top-{args.top} {hitk[unsure].mean() * 100:.1f}%")

    report = {"tag": args.tag, "corpus_tokens": corpus_tokens, "labelled_tokens": labelled_tokens,
              "rows": model.modulus, "slots": args.slots, "top": args.top,
              "table_bytes": model.nbytes(), "fit_seconds": fit_seconds,
              "positions": int(len(label)), "top1": float(hit1.mean()), "topk": float(hitk.mean()),
              "top1_unsure": float(hit1[unsure].mean()), "topk_unsure": float(hitk[unsure].mean()),
              "fire": float(fired.mean()), "top1_given_fire": float(hit1[fired].mean()),
              "bands": rows}
    # What it would be worth as a second branch beside the block drafter. Under the model of the
    # previous ledger entry the block drafter errs at a position with probability 1 - p1, so the
    # branch earns a token exactly where it is right and the drafter is not. `a` is the per-token
    # agreement the block drafter is measured at; the branch moves it to `a + gain`.
    gain1 = float(((1.0 - p1) * hit1).mean())
    gaink = float(((1.0 - p1) * hitk).mean())
    print(f"\nAS A SECOND BRANCH  P(engram right AND drafter wrong) = {gain1 * 100:.2f}% top-1, "
          f"{gaink * 100:.2f}% top-{args.top}")
    for base in (0.797, 0.860):
        for m, cost in ((8, 139.1), (16, 154.2)):
            before = sum(base ** j for j in range(1, m + 1))
            after = sum((base + gain1) ** j for j in range(1, m + 1))
            print(f"  a {base:.3f} -> {base + gain1:.3f} at block {m:2d}: "
                  f"{(before + 1) / cost * 1000:5.1f} -> {(after + 1) / cost * 1000:5.1f} tok/s")
    report["branch_gain_top1"] = gain1
    report["branch_gain_topk"] = gaink
    if args.json_out:
        with open(args.json_out, "a") as handle:
            handle.write(json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
