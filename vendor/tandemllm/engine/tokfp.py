"""A tokenizer fingerprint, so a store of token ids is never read by another tokenizer.

The suffix stores (`tools/build_corpus.py`'s corpus, `engine.cache.PersistentSuffixStore`) hold token
ids and no text. Ids from another tokenizer are valid integers with another meaning: a lookup would
match nothing useful or, worse, propose plausible-looking wrong continuations that the verify then
rejects one by one. The fingerprint is the SHA-256 of the checkpoint's `tokenizer.json` (the file
that defines the ids); a store records it in `meta.json` and a store with another one is refused.
A store written before the fingerprint existed (the served ones) is read with one warning.
"""

from __future__ import annotations

import hashlib
import os
import sys

KEY = "tokenizer_sha256"
_WARNED: set = set()


def fingerprint(snapshot: str | None) -> str | None:
    """Sha256 of `<snapshot>/tokenizer.json`, or None when there is no such file."""
    if not snapshot:
        return None
    path = os.path.join(os.path.expanduser(snapshot), "tokenizer.json")
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_store(meta: dict, fp: str | None, where: str) -> None:
    """Refuse a store whose recorded tokenizer is not this one; warn once for an unrecorded one."""
    have = (meta or {}).get(KEY)
    if not fp:
        return
    if have is None:
        if where not in _WARNED:
            _WARNED.add(where)
            print(f"[store] {where}: no {KEY} in meta.json (written by an older build); read as this "
                  f"tokenizer's", file=sys.stderr, flush=True)
        return
    if have != fp:
        raise ValueError(f"{where}: its token ids come from another tokenizer "
                         f"({KEY} {have[:12]}..., this model's {fp[:12]}...)")
