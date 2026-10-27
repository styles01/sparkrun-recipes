"""Replay the reasoning-loop detector's arms over real reasoning, offline.

The arms are `engine/spec.py`'s `StallArms`, with the parameters of the policy's reasoning-loop
task (ops/supervisor.yaml), so what this prints is what the served monitor would have seen.
Three sets, as the calibration of 2026-09-30 defined them:

    failed   turns that ended with no tool call and no text (an empty stop, or `length`)
    closes   each forced close's reasoning before the close phrase: what the old stall
             detector fired on (the text of that request's reasoning part up to the phrase)
    healthy  turns that ended in a tool call after 2,048 or more reasoning tokens (the turn's
             reasoning up to its first close phrase)

From a COPY of opencode's database (opened read-only), tokenised with the served tokenizer:

    python tools/stall_replay.py --db ~/oc-copy.db --tokenizer .../tokenizer.json \\
        --until 2026-09-30T20:07:36.556Z [--write-fixture tests/fixtures/stall_replay.bin.xz]

or from the fixture the tests read:

    python tools/stall_replay.py --fixture tests/fixtures/stall_replay.bin.xz

The fixture holds no text: each sequence is relabelled, every token replaced by the rank of its
first occurrence in that sequence. Every arm only compares tokens for equality, so the replay of
a relabelled sequence is exactly the replay of the original, and the ids decode to nothing. The
failed turns are cut at `--failed-keep` tokens (the claim is a close before 3,100). The file is
xz: one JSON line (the metadata and every sequence's length), then the labels as uint16.
"""

from __future__ import annotations

import argparse
import array
import collections
import datetime as _dt
import json
import lzma
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHRASE = "Considering the limited time"
MODEL = "qwen38-spark-engine"


def relabel(ids) -> list[int]:
    """Each token as the rank of its first occurrence in `ids`."""
    rank: dict[int, int] = {}
    return [rank.setdefault(int(t), len(rank)) for t in ids]


def write_fixture(path: str, meta: dict, sets: dict) -> None:
    head = {"meta": meta, "lengths": {k: [len(s) for s in v] for k, v in sets.items()}}
    body = b"".join(array.array("H", s).tobytes() for v in sets.values() for s in v)
    with lzma.open(path, "wb", preset=9) as f:
        f.write(json.dumps(head, separators=(",", ":")).encode() + b"\n" + body)


def read_fixture(path: str) -> tuple[dict, dict]:
    """`(meta, sets)` of a fixture `write_fixture` wrote."""
    with lzma.open(path, "rb") as f:
        raw = f.read()
    nl = raw.index(b"\n")
    head = json.loads(raw[:nl])
    labels = array.array("H")
    labels.frombytes(raw[nl + 1:])
    sets, i = {}, 0
    for name, lengths in head["lengths"].items():
        sets[name] = []
        for n in lengths:
            sets[name].append(labels[i:i + n].tolist())
            i += n
    return head["meta"], sets


def fires(ids, arms: dict) -> tuple[str | None, int | None]:
    """`(reason, token count)` of the first fire of `arms` over `ids`, or `(None, None)`."""
    from engine.spec import StallArms
    a = StallArms(arms)
    for n, t in enumerate(ids, 1):
        why = a.feed(int(t), n)
        if why is not None:
            return why, n
    return None, None


def replay(sets: dict, arms: dict) -> dict:
    """Per set: how many sequences each arm alone, and all arms together, fire on."""
    out = {}
    for name, seqs in sets.items():
        row = {"sequences": len(seqs), "tokens": sum(len(s) for s in seqs)}
        for arm in ("loop", "repeat", "novelty"):
            if arm in arms:
                hit = [fires(s, {arm: arms[arm]}) for s in seqs]
                row[arm] = sum(1 for why, _ in hit if why is not None)
                row[arm + "_at"] = [n for _, n in hit]
        hit = [fires(s, arms) for s in seqs]
        row["any"] = sum(1 for why, _ in hit if why is not None)
        row["reasons"] = [why for why, _ in hit]
        out[name] = row
    return out


def from_db(db_path: str, tokenizer: str, until: str | None, failed_keep: int) -> dict:
    """The three sets from a copy of opencode's database, tokenised, relabelled."""
    import sqlite3
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer)
    cut = (_dt.datetime.fromisoformat(until.replace("Z", "+00:00")).timestamp() * 1000
           if until else None)
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    msgs = {}
    for mid, tc, data in db.execute("select id, time_created, data from message"):
        d = json.loads(data)
        if d.get("role") != "assistant" or d.get("modelID") != MODEL:
            continue
        if cut is not None and tc > cut:
            continue
        msgs[mid] = (tc, d.get("finish"), (d.get("tokens") or {}).get("reasoning") or 0)
    parts = collections.defaultdict(list)
    for mid, data in db.execute("select message_id, data from part order by id"):
        if mid in msgs:
            parts[mid].append(json.loads(data))

    def ids(text):
        return tok.encode(text, add_special_tokens=False).ids

    sets = {"failed": [], "closes": [], "healthy": []}
    for mid, (tc, finish, rt) in sorted(msgs.items(), key=lambda kv: kv[1][0]):
        ps = parts[mid]
        steps = [p.get("text", "") for p in ps if p.get("type") == "reasoning"]
        reasoning = "".join(steps)
        has_tool = any(p.get("type") == "tool" for p in ps)
        text = "".join(p.get("text", "") for p in ps if p.get("type") == "text").strip()
        if not has_tool and not text and finish in ("stop", "length"):
            sets["failed"].append(relabel(ids(reasoning)[:failed_keep]))
        for step in steps:
            pieces = step.split(PHRASE)
            for k in range(1, len(pieces)):
                sets["closes"].append(relabel(ids(PHRASE.join(pieces[:k]))))
        if has_tool and rt >= 2048 and reasoning:
            sets["healthy"].append(relabel(ids(reasoning.split(PHRASE)[0])))
    return sets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", help="a COPY of opencode.db (opened read-only)")
    ap.add_argument("--tokenizer", help="the served tokenizer's tokenizer.json")
    ap.add_argument("--until", default=None, help="only turns created up to this time (ISO, UTC)")
    ap.add_argument("--failed-keep", type=int, default=4096)
    ap.add_argument("--fixture", help="replay this fixture instead of a database")
    ap.add_argument("--write-fixture", help="write the relabelled sets here (.bin.xz)")
    ap.add_argument("--policy", default=None, help="the policy file (default ops/supervisor.yaml)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args()
    from server.conductor.monitor import policy
    pol = policy.load(a.policy)
    arms = next(t["arms"] for t in pol.tasks.values() if t.rule == "reasoning_loop")
    if a.fixture:
        meta, sets = read_fixture(a.fixture)
    elif a.db and a.tokenizer:
        sets = from_db(a.db, a.tokenizer, a.until, a.failed_keep)
        meta = {"until": a.until, "failed_keep": a.failed_keep, "relabelled": True,
                "phrase": PHRASE}
    else:
        ap.error("--fixture, or --db and --tokenizer")
    if a.write_fixture:
        write_fixture(a.write_fixture, meta, sets)
    res = replay(sets, arms)
    if a.json:
        print(json.dumps({"meta": meta, "policy_version": pol.version, "replay": res}))
        return 0
    print(f"policy version {pol.version}; " + ", ".join(f"{k}={v}" for k, v in meta.items()))
    for name, row in res.items():
        arms_s = "  ".join(f"{arm}={row[arm]}" for arm in ("loop", "repeat", "novelty")
                           if arm in row)
        print(f"{name:8s} {row['sequences']:4d} sequences {row['tokens']:8d} tokens  fired: "
              f"{arms_s}  any={row['any']}")
    print("failed, first fire (reason at token):",
          [(why, n) for why, n in (fires(s, arms) for s in sets["failed"])])
    return 0


if __name__ == "__main__":
    sys.exit(main())
