"""Build the global suffix store the lookup drafter reads.

The measured limit on the first suffix memory was coverage: it could only propose where an n-gram
had already occurred *inside the current request*, which on 128 tokens of fresh output is about 8 %
of steps. A store over a corpus changes the question from "has this occurred
since the request started" to "has this occurred at all", and this board has roughly 90 GB of unused
unified memory to answer it with.

What goes in is text the engine is likely to be asked to continue: source code in the languages it
serves, technical prose, and the model's own past outputs. What comes out is two `.npy` files --
the token stream and a suffix array sorted by the first `--max-order` tokens -- which the drafter
memory-maps and binary-searches in microseconds.

The suffix array is sorted by a *bounded* prefix on purpose. The drafter never matches more than
eight tokens, so ordering beyond eight is work nobody reads, and bounding it turns log n rounds of
prefix doubling into three.

Runs on the CPU. It needs no GPU and therefore no box lock.

Privacy: the store holds token ids and never text, the build runs on the machine that serves the
model, and `corpus/` is gitignored and excluded from every sync back. `--private` is the hook for
local repositories and notes, which are the data that would move the prose and German rows; the tool
walks that directory only when the command line names it, and nothing it reads reaches the
repository.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402

# Marks a document boundary in the token stream. Above every real id, so it sorts last and a
# continuation that would run past the end of a document stops there instead.
DOC_SEP = 1 << 30

TEXT_GLOBS = ("*.py", "*.go", "*.ts", "*.tsx", "*.js", "*.rs", "*.c", "*.h", "*.cpp", "*.cu",
              "*.md", "*.txt", "*.rst", "*.toml", "*.yaml", "*.yml", "*.json", "*.sh")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".mypy_cache",
             ".pytest_cache", "site-packages/torch/include", "test_data", "testdata"}


def build_suffix_array(tokens, max_order: int = 8):
    """Suffix array of `tokens`, ordered by the first `max_order` tokens.

    Prefix doubling: ranks by one token, then two, four, eight. Suffixes shorter than the window
    pad with a value below every real rank, so they sort before any longer suffix sharing their
    prefix -- the same convention `CorpusSuffixStore._cmp_at` uses when it compares a pattern.
    """
    toks = np.asarray(tokens)
    n = int(toks.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    _, rank = np.unique(toks, return_inverse=True)
    rank = rank.astype(np.int64).reshape(-1)
    k = 1
    while k < max_order:
        second = np.full(n, -1, dtype=np.int64)
        if n > k:
            second[:n - k] = rank[k:]
        sa = np.lexsort((second, rank))
        r0, r1 = rank[sa], second[sa]
        changed = np.empty(n, dtype=bool)
        changed[0] = True
        if n > 1:
            changed[1:] = (r0[1:] != r0[:-1]) | (r1[1:] != r1[:-1])
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[sa] = np.cumsum(changed) - 1
        rank = new_rank
        k *= 2
    sa = np.lexsort((np.arange(n), rank))
    return sa.astype(np.int32)


from engine.tokfp import fingerprint  # noqa: E402


def write_store(out_dir: str, tokens, sa, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "tokens.npy"), np.asarray(tokens, dtype=np.int32))
    np.save(os.path.join(out_dir, "sa.npy"), np.asarray(sa, dtype=np.int32))
    meta = dict(meta)
    meta.setdefault("doc_sep", DOC_SEP)
    meta.setdefault("n_tokens", int(len(tokens)))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def iter_files(root: str, globs: tuple[str, ...], max_bytes: int):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            if not any(fnmatch.fnmatch(name, g) for g in globs):
                continue
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size == 0 or size > max_bytes:
                continue
            yield path


def read_text(path: str) -> str | None:
    try:
        with open(path, "r", errors="strict") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", default=[],
                    help="PATH[:MAXTOKENS] -- directory to walk, with its own token budget; "
                         "repeatable")
    ap.add_argument("--parquet", action="append", default=[],
                    help="PATH[:COLUMN[:ROWS[:MAXTOKENS]]] -- a text column from a parquet file, "
                         "with its own budget; repeatable")
    ap.add_argument("--traces", default=None,
                    help="directory of recorded traces; their outputs are the model's own style")
    ap.add_argument("--private", default=None,
                    help="an extra directory, never synced and never committed")
    ap.add_argument("--out", default=None, help="store directory (default corpus/)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=40_000_000)
    ap.add_argument("--max-file-bytes", type=int, default=400_000)
    ap.add_argument("--globs", default=",".join(TEXT_GLOBS))
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = a.out or os.path.join(root, "corpus")
    globs = tuple(g.strip() for g in a.globs.split(",") if g.strip())

    cfg = load_config(a.model)
    # the fast tokeniser file alone is enough here, which keeps this runnable anywhere the
    # checkpoint directory can be read -- no torch, no transformers
    tok_file = os.path.join(cfg.path, "tokenizer.json")
    if os.path.exists(tok_file):
        from tokenizers import Tokenizer
        _t = Tokenizer.from_file(tok_file)

        def encode(text: str) -> list[int]:
            return _t.encode(text, add_special_tokens=False).ids
    else:
        from transformers import AutoTokenizer
        _t = AutoTokenizer.from_pretrained(cfg.path)

        def encode(text: str) -> list[int]:
            return _t(text, add_special_tokens=False).input_ids

    chunks: list[np.ndarray] = []
    total = 0
    sources: list[dict] = []
    t0 = time.perf_counter()

    def add(text: str) -> int:
        nonlocal total
        ids = encode(text)
        if not ids:
            return 0
        chunks.append(np.array(ids + [DOC_SEP], dtype=np.int64))
        total += len(ids) + 1
        return len(ids)

    for spec in a.parquet:
        parts = spec.split(":")
        path = os.path.expanduser(parts[0])
        column = parts[1] if len(parts) > 1 and parts[1] else "text"
        rows = int(parts[2]) if len(parts) > 2 and parts[2] else None
        budget = int(parts[3]) if len(parts) > 3 and parts[3] else None
        if not os.path.exists(path):
            print(f"skip (no such file): {path}")
            continue
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=[column])
        col = table.column(column)
        n_docs, n_tokens = 0, 0
        # batch the rows so a page of prose is one document rather than one line
        buf: list[str] = []
        limit = rows if rows is not None else table.num_rows
        for idx in range(min(limit, table.num_rows)):
            if total >= a.max_tokens or (budget is not None and n_tokens >= budget):
                break
            text = col[idx].as_py()
            if not text or len(text.strip()) < 40:
                continue
            buf.append(text.strip())
            if sum(len(x) for x in buf) > 20_000:
                got = add("\n\n".join(buf))
                buf = []
                if got:
                    n_docs += 1
                    n_tokens += got
        if buf:
            got = add("\n\n".join(buf))
            if got:
                n_docs += 1
                n_tokens += got
        sources.append({"path": f"{path}#{column}", "files": n_docs, "tokens": n_tokens})
        print(f"{path}#{column}: {n_docs} documents, {n_tokens:,} tokens")

    for spec in list(a.src) + ([a.private] if a.private else []):
        parts = spec.rsplit(":", 1)
        src = os.path.expanduser(parts[0])
        budget = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        if not os.path.isdir(src):
            src, budget = os.path.expanduser(spec), None
        if not os.path.isdir(src):
            print(f"skip (not a directory): {spec}")
            continue
        n_files, n_tokens = 0, 0
        for path in iter_files(src, globs, a.max_file_bytes):
            if total >= a.max_tokens or (budget is not None and n_tokens >= budget):
                break
            text = read_text(path)
            if not text:
                continue
            got = add(text)
            if got:
                n_files += 1
                n_tokens += got
        sources.append({"path": src, "files": n_files, "tokens": n_tokens})
        print(f"{src}: {n_files} files, {n_tokens:,} tokens")

    if a.traces and os.path.isdir(a.traces):
        n_tokens = 0
        for name in sorted(os.listdir(a.traces)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(a.traces, name)) as f:
                tr = json.load(f)
            ids = tr.get("output_ids") or []
            if ids:
                chunks.append(np.array(list(ids) + [DOC_SEP], dtype=np.int64))
                total += len(ids) + 1
                n_tokens += len(ids)
        sources.append({"path": a.traces, "files": "traces", "tokens": n_tokens,
                        "kind": "traces"})
        print(f"{a.traces}: {n_tokens:,} tokens of the model's own output")
        print("  WARNING: these are the streams tools/sim_draft.py replays against. A store that\n"
              "  contains them lets the drafter look up the answer, and the simulator will report\n"
              "  a fire rate near 95 % and a draft acceptance near 98 %. Build the store without\n"
              "  --traces to measure anything.")

    if not chunks:
        raise SystemExit("nothing to build from; pass --src")

    tokens = np.concatenate(chunks)[:a.max_tokens].astype(np.int32)
    print(f"\ntokenised {len(tokens):,} tokens in {time.perf_counter() - t0:.1f} s")
    t1 = time.perf_counter()
    sa = build_suffix_array(tokens, a.max_order)
    print(f"suffix array in {time.perf_counter() - t1:.1f} s")
    write_store(out_dir, tokens, sa, {
        "max_order": a.max_order,
        "sources": sources,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": os.path.basename(cfg.path),
        # the ids' tokenizer, so another model's engine refuses this store
        "tokenizer_sha256": fingerprint(cfg.path),
    })
    mb = (tokens.nbytes + sa.nbytes) / 1e6
    print(f"store at {out_dir}: {len(tokens):,} tokens, {mb:.0f} MB on disk")


if __name__ == "__main__":
    main()
