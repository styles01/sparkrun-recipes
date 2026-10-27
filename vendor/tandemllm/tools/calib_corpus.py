"""Build the calibration texts for the quantiser, and a disjoint text for the sensitivity pass.

The quantiser weights every input channel by how much a calibration text uses it, so the text
decides which channels get the fine grid. `bench/calib.txt` alone is two thirds Python and one
third package documentation; the engine serves prose, German and code. This tool adds:

  * code: modules of the Python standard library of the interpreter that runs it (PSF licence;
    tests, site-packages and anything the held-out files come from are skipped);
  * prose: Wikipedia articles, English and German, from `wikimedia/wikipedia` (read only for
    statistics, never redistributed).

Documents are taken in a fixed order from fixed shards, so the same interpreter and the same shards
give the same bytes. Two splits, disjoint by document: `calib-{code,en,de}.txt` for the
quantiser's statistics and `sens-{code,prose}.txt` for `tools/quant_sensitivity.py`. Neither may
contain the gate's held-out texts: any document sharing a run of 12 words with `bench/heldout_*.txt` is dropped, and so is any document
that mentions a string passed with `--ban`.

    python tools/calib_corpus.py --out ~/calib --code-chars 240000 --prose-chars 260000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import sysconfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELDOUT = [os.path.join(HERE, "bench", f) for f in ("heldout_prose.txt", "heldout_code.txt")]
SHINGLE = 12


def shingles(text: str, n: int = SHINGLE) -> set[int]:
    words = re.findall(r"\S+", text)
    return {hash(" ".join(words[i:i + n])) for i in range(max(0, len(words) - n + 1))}


class Guard:
    """Rejects a document that overlaps the held-out texts or names a banned string."""

    def __init__(self, heldout: list[str], ban: list[str]):
        self.sh = set()
        for p in heldout:
            self.sh |= shingles(open(p).read())
        self.ban = [b.lower() for b in ban if b]
        self.dropped = 0

    def ok(self, text: str) -> bool:
        low = text.lower()
        if any(b in low for b in self.ban) or (shingles(text) & self.sh):
            self.dropped += 1
            return False
        return True


def stdlib_modules() -> list[str]:
    root = sysconfig.get_paths()["stdlib"]
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if not (x.startswith("test") or x in (
            "site-packages", "dist-packages", "__pycache__", "idlelib", "lib2to3", "tkinter",
            "turtledemo", "ensurepip", "encodings", "config-3") or x.startswith("config-")))
        for f in sorted(files):
            if f.endswith(".py") and not f.startswith("test"):
                out.append(os.path.join(d, f))
    return out


def take(docs, guard: Guard, chars: int) -> tuple[list[str], list[str]]:
    """Documents in order until `chars` are collected; returns (texts, ids)."""
    texts, ids, n = [], [], 0
    for doc_id, text in docs:
        if n >= chars:
            break
        if len(text) < 400 or not guard.ok(text):
            continue
        texts.append(text)
        ids.append(doc_id)
        n += len(text)
    return texts, ids


def code_docs(skip_names: list[str], max_bytes: int = 120_000):
    mods = stdlib_modules()
    # a fixed interleave rather than the alphabet, so a split does not end up all `email.*`
    mods = sorted(mods, key=lambda p: hashlib.sha1(os.path.relpath(p, sysconfig.get_paths()["stdlib"])
                                                    .encode()).hexdigest())
    for p in mods:
        if any(s in p for s in skip_names) or os.path.getsize(p) > max_bytes:
            continue
        try:
            text = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        # the first ~8 kB of a module, cut at a blank line: many modules rather than a few long ones
        if len(text) > 8000:
            cut = text.rfind("\n\n", 4000, 8000)
            text = text[: cut if cut > 0 else 8000]
        yield os.path.relpath(p, sysconfig.get_paths()["stdlib"]), text


def wiki_docs(lang: str, shard: int, cache: str | None):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    total = {"en": 41, "de": 20}[lang]
    path = hf_hub_download("wikimedia/wikipedia", f"20231101.{lang}/train-{shard:05d}-of-{total:05d}.parquet",
                           repo_type="dataset", cache_dir=cache)
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["id", "title", "text"]).to_pydict()
        for i, title, text in zip(t["id"], t["title"], t["text"]):
            # long articles only, and only their first ~6 kB: the lead and first sections are prose;
            # the tails are reference lists, tables and categories
            if len(text) < 3000:
                continue
            cut = text[:6000]
            cut = cut[: cut.rfind("\n") if cut.rfind("\n") > 3000 else len(cut)]
            yield f"{lang}:{i}:{title}", f"{title}\n\n{cut.strip()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-chars", type=int, default=240_000, help="stdlib code for calibration")
    ap.add_argument("--prose-chars", type=int, default=260_000, help="English prose for calibration")
    ap.add_argument("--de-chars", type=int, default=60_000, help="German prose for calibration")
    ap.add_argument("--sens-chars", type=int, default=20_000, help="each sensitivity split")
    ap.add_argument("--eval-chars", type=int, default=0,
                    help="also write eval-{code,prose}.txt, held out from everything else")
    ap.add_argument("--ban", action="append", default=["pagoda garden"],
                    help="drop any document containing this string (repeatable)")
    ap.add_argument("--skip-code", action="append", default=["triton"],
                    help="skip stdlib paths containing this (the held-out code is Triton's)")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    guard = Guard(HELDOUT, args.ban)
    calib_base = open(os.path.join(HERE, "bench", "calib.txt")).read()
    assert guard.ok(calib_base), "bench/calib.txt overlaps a held-out file"
    manifest: dict = {"python": sys.version.split()[0], "stdlib": sysconfig.get_paths()["stdlib"],
                      "ban": args.ban, "shingle_words": SHINGLE, "files": {}}

    code = code_docs(args.skip_code)
    sens_code, sens_code_ids = take(code, guard, args.sens_chars)
    calib_code, calib_code_ids = take(code, guard, args.code_chars)
    en = wiki_docs("en", 0, args.cache)
    sens_en, sens_en_ids = take(en, guard, args.sens_chars * 3 // 4)
    calib_en, calib_en_ids = take(en, guard, args.prose_chars)
    de = wiki_docs("de", 0, args.cache)
    sens_de, sens_de_ids = take(de, guard, args.sens_chars // 4)
    calib_de, calib_de_ids = take(de, guard, args.de_chars)
    # a third split, after the other two in the same order: held-out text for a wider gate. Taking
    # it last leaves the calibration and sensitivity splits byte-identical with or without it.
    eval_code, eval_code_ids = take(code, guard, args.eval_chars)
    eval_en, eval_en_ids = take(en, guard, args.eval_chars * 3 // 4)
    eval_de, eval_de_ids = take(de, guard, args.eval_chars // 4)
    assert not (set(sens_code_ids) & set(calib_code_ids)) and not (set(sens_en_ids) & set(calib_en_ids))

    def write(name: str, texts: list[str], ids: list[str]) -> None:
        body = "\n\n".join(texts)
        p = os.path.join(args.out, name)
        with open(p, "w") as f:
            f.write(body)
        manifest["files"][name] = {"chars": len(body), "docs": len(ids),
                                   "sha256": hashlib.sha256(body.encode()).hexdigest(),
                                   "ids": ids}
        print(f"  {name:22s} {len(ids):4d} docs {len(body):8d} chars")

    write("calib-code.txt", [calib_base] + calib_code, ["bench/calib.txt"] + calib_code_ids)
    write("calib-en.txt", calib_en, calib_en_ids)
    write("calib-de.txt", calib_de, calib_de_ids)
    write("sens-code.txt", sens_code, sens_code_ids)
    write("sens-prose.txt", sens_en + sens_de, sens_en_ids + sens_de_ids)
    if args.eval_chars:
        write("eval-code.txt", eval_code, eval_code_ids)
        write("eval-prose.txt", eval_en + eval_de, eval_en_ids + eval_de_ids)
    manifest["dropped_by_guard"] = guard.dropped
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[corpus] {guard.dropped} documents dropped by the held-out/ban guard -> {args.out}")


if __name__ == "__main__":
    main()
