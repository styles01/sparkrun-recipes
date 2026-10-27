"""The word-level reasoning-loop detector's calibration (proxy P4), the twin of argloop_fp.py.

    python tools/loop_words_fp.py --db ~/.local/share/opencode/opencode.db [--json out.json]
        [--cfg words.json]

The sets, from the reasoning parts of at least 200 words in the opencode database:

    legit    a finished turn (`stop` or `tool-calls`) whose reasoning compresses under 5 times
             (zlib), the shape of healthy reasoning
    loops    a turn the token limit cut while the reasoning was the last thing written, or any
             reasoning that compresses 8 times or more
    unclear  the rest (reported, not counted)

Each text is streamed through `server.conductor.monitor.tier0_words.WordLoop` in pieces of a few
words, as the proxy feeds it from a stream. The legit set is not clean by construction: the
engine's own D1 acted on that traffic, so a reasoning part that looped and was closed by it reads
as finished. A fire on the legit set is therefore a false alarm or a loop the engine closed, and
the text alone cannot say which; the shipped block stays `calibrated: false` until the Compare
view's labels say.

Prints counts and numbers only, never a text: per set, how many fire under the configuration (the
shipped DEFAULTS, or `--cfg`), and for the legit set the largest values each arm saw (the longest
repeated run, the highest repeat share past `min_words`, the lowest novelty past `min_words`), so a
threshold can be set beyond them with a margin. For the loops, the word at which the detector
fires.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor.monitor import tier0_words  # noqa: E402


def sets(db) -> tuple[list[str], list[str]]:
    legit, loops = [], []
    q = ("select json_extract(m.data,'$.finish'), json_extract(p.data,'$.text'), "
         "(select count(*) from part p2 where p2.message_id = m.id and p2.id > p.id and "
         "json_extract(p2.data,'$.type') in ('text','tool')) "
         "from part p join message m on p.message_id = m.id "
         "where json_extract(p.data,'$.type')='reasoning'")
    for finish, text, after in db.execute(q):
        if not isinstance(text, str) or len(text.split()) < 200:
            continue
        raw = text.encode("utf-8", "surrogatepass")
        z = len(raw) / max(1, len(zlib.compress(raw, 6)))
        if (finish == "length" and not after) or z >= 8:
            loops.append(text)
        elif finish in ("stop", "tool-calls") and z < 5:
            legit.append(text)
    return legit, loops


def probe(text: str, cfg: dict, piece: int = 3) -> dict:
    """Stream `text` in pieces of `piece` words; the verdict and the arms' extremes."""
    w = tier0_words.WordLoop(cfg)
    words = text.split(" ")
    best_share, low_nov, run = 0.0, 1.0, 0
    fired_at = None
    arms = w.arms
    for i in range(0, len(words), piece):
        chunk = " ".join(words[i:i + piece]) + " "
        if w.feed(chunk) is not None and fired_at is None:
            fired_at = w.words
        if arms.share is not None:
            best_share = max(best_share, arms.share)
        if arms._sizes:
            run = max(run, max(arms._runs[s] + s for s in arms._sizes))
        if arms.nov and w.words >= arms.nov["min_words"] and len(arms._recent) >= 2 * \
                arms.nov["window"]:
            low_nov = min(low_nov, arms._novelty())
        if fired_at is not None:
            break
    w.flush()
    return {"fired": w.reason, "at": fired_at, "words": w.words, "share": best_share,
            "novelty": low_nov, "run": run}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/opencode/opencode.db"))
    ap.add_argument("--cfg", default=None, help="a JSON words block to test instead of DEFAULTS")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    cfg = dict(tier0_words.DEFAULTS)
    if a.cfg:
        with open(a.cfg) as f:
            cfg.update(json.load(f))
    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    legit, loops = sets(db)
    out: dict = {"cfg": {k: cfg[k] for k in ("loop", "repeat", "novelty")}}
    for name, texts in (("legit", legit), ("loops", loops)):
        res = [probe(t, cfg) for t in texts]
        fired = [r for r in res if r["fired"]]
        kinds: dict = {}
        for r in fired:
            k = r["fired"].split("(")[0]
            kinds[k] = kinds.get(k, 0) + 1
        out[name] = {
            "n": len(res), "fired": len(fired), "by_arm": kinds,
            "words_max": max((r["words"] for r in res), default=0),
            "words_p50": sorted(r["words"] for r in res)[len(res) // 2] if res else 0,
            "max_share": round(max((r["share"] for r in res if not r["fired"]), default=0), 3),
            "min_novelty": round(min((r["novelty"] for r in res if not r["fired"]), default=1),
                                 3),
            "max_run": max((r["run"] for r in res if not r["fired"]), default=0),
            "fired_at": sorted(r["at"] for r in fired)[:50],
        }
        print(f"{name}: {len(res)} texts, {len(fired)} fired {kinds}; longest "
              f"{out[name]['words_max']} words (p50 {out[name]['words_p50']}); unfired extremes: "
              f"share {out[name]['max_share']}, novelty {out[name]['min_novelty']}, run "
              f"{out[name]['max_run']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
