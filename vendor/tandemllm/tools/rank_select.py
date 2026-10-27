"""Pick the training and holdout sequences for the drafter ranking fine-tune (RND-37). CPU only.

Input: the generation shards of the private draft-training dataset (one JSON object a line with
`name, klass, topic, think, effort, prompt_ids, gen_ids`; the prompts are already chat-templated,
thinking on or off), and the lattice traces the gate replays (`*.json` with `prompt_ids`).

Output (`--out`):
  train.jsonl        the sequences to teacher-force through the target for taps and labels
  think-src/*.json   source traces for the fresh thinking-on holdout (prompt only; the target writes
                     the continuation with `tools/regen_traces.py --max-new N`)
  select.json        what was picked, the counts, and the hashes it was checked against

Two rules carry the whole holdout argument:
  - a prompt is identified by its user content (the templated ids minus the template's tail, which
    is the only part thinking on/off changes), and nothing that shares that key with a gate trace or
    with the thinking-on holdout is ever trained on;
  - the stratification is by output class and thinking mode, seeded, so a re-run picks the same set.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random

THINK_OPEN, THINK_CLOSE = 248068, 248069


def key(prompt_ids: list[int]) -> str:
    """The user content's identity: everything up to the assistant turn's opening.

    The template ends `<|im_start|>assistant\\n<think>\\n` (thinking on) or
    `... <think>\\n\\n</think>\\n\\n` (thinking off); cutting at the last `<think>` and the two
    tokens before it removes exactly the part the thinking switch changes."""
    ids = list(prompt_ids)
    if THINK_OPEN in ids:
        ids = ids[:len(ids) - 1 - ids[::-1].index(THINK_OPEN)]
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:20]


def load_rows(gen_dir: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(os.path.join(gen_dir, "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="dir of gen/shard-*.jsonl")
    ap.add_argument("--gate-traces", required=True, action="append",
                    help="dir(s) of *.json traces whose prompts must never be trained on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--mix", default="prose:0.4,code:0.3,de:0.3")
    ap.add_argument("--think-share", type=float, default=0.30)
    ap.add_argument("--think-holdout", default="prose:8,code:6,de:6",
                    help="fresh thinking-on prompts kept out of training, per class")
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = load_rows(a.gen)
    gate_keys = set()
    for d in a.gate_traces:
        for f in glob.glob(os.path.join(os.path.expanduser(d), "*.json")):
            with open(f) as fh:
                tr = json.load(fh)
            gate_keys.add(key(tr["prompt_ids"]))

    seen: set[str] = set()
    pool = collections.defaultdict(list)            # (klass, think) -> rows
    dropped = collections.Counter()
    for r in rows:
        k = key(r["prompt_ids"])
        if k in gate_keys:
            dropped["shares a gate trace's prompt"] += 1
            continue
        if k in seen:
            dropped["duplicate prompt"] += 1
            continue
        if not r.get("gen_ids"):
            dropped["no generation"] += 1
            continue
        seen.add(k)
        r["_key"] = k
        pool[(r["klass"], bool(r["think"]))].append(r)
    for v in pool.values():
        rng.shuffle(v)

    # the thinking-on holdout first, so the training pick can never take one of them
    held: list[dict] = []
    for part in a.think_holdout.split(","):
        kl, n = part.split(":")
        take = pool[(kl, True)][:int(n)]
        pool[(kl, True)] = pool[(kl, True)][int(n):]
        held.extend(take)

    train: list[dict] = []
    for part in a.mix.split(","):
        kl, share = part.split(":")
        n = round(a.n * float(share))
        n_think = round(n * a.think_share)
        for think, want in ((True, n_think), (False, n - n_think)):
            got = pool[(kl, think)][:want]
            if len(got) < want:
                raise SystemExit(f"only {len(got)} {kl}/think={think} rows, wanted {want}")
            train.extend(got)
    rng.shuffle(train)
    hk = {r["_key"] for r in held}
    assert not (hk & {r["_key"] for r in train}), "holdout prompt in the training pick"
    assert not (gate_keys & {r["_key"] for r in train}), "gate prompt in the training pick"

    os.makedirs(os.path.join(a.out, "think-src"), exist_ok=True)
    with open(os.path.join(a.out, "train.jsonl"), "w") as f:
        for r in train:
            f.write(json.dumps({"name": r["name"], "topic": r["klass"], "klass": r["klass"],
                                "think": bool(r["think"]), "effort": r.get("effort"),
                                "split": "train", "prompt_ids": r["prompt_ids"],
                                "gen_ids": r["gen_ids"]}) + "\n")
    for i, r in enumerate(held):
        name = f"think-{r['klass']}-{i:02d}"
        with open(os.path.join(a.out, "think-src", f"{name}.json"), "w") as f:
            json.dump({"name": name, "klass": r["klass"], "thinking": True,
                       "source_name": r["name"], "prompt_ids": r["prompt_ids"],
                       "output_ids": []}, f)
    pos = sum(len(r["prompt_ids"]) + len(r["gen_ids"]) for r in train)
    gen_pos = sum(len(r["gen_ids"]) for r in train)
    think_gen = sum(len(r["gen_ids"]) for r in train if r["think"])
    summary = {
        "seed": a.seed, "n_train": len(train), "positions": pos, "gen_positions": gen_pos,
        "think_gen_positions": think_gen,
        "by_class": dict(collections.Counter((r["klass"], bool(r["think"])) for r in train)
                         .most_common()).__repr__(),
        "think_holdout": [r["name"] for r in held], "gate_prompt_keys": len(gate_keys),
        "dropped": dict(dropped),
    }
    with open(os.path.join(a.out, "select.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
