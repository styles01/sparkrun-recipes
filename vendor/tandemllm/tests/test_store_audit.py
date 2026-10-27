"""CPU tests for tools/store_audit.py: finding a benchmark's prompts in the suffix store.

The filtered store is the instrument the honest row is measured on, so what it drops has to be
exactly the documents that hold a dataset prompt -- no more (it would hide real traffic) and no less
(the row would still read itself).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.store_audit import DOC_SEP, match_docs, signature, split_docs  # noqa: E402


def test_documents_split_on_the_separator():
    t = np.array([1, 2, DOC_SEP, 3, DOC_SEP, DOC_SEP, 4, 5], dtype=np.int64)
    assert [d.tolist() for d in split_docs(t)] == [[1, 2], [3], [4, 5]]


def test_signature_is_the_middle_and_skips_the_edges():
    ids = list(range(100, 130))
    sig = signature(ids, 12)
    assert len(sig) == 12 and sig[0] > 101 and sig[-1] < 128


def test_a_document_is_matched_only_by_a_whole_signature():
    sigs = {"a": tuple(range(500, 512)), "b": tuple(range(700, 712))}
    docs = [np.array([1, 2] + list(range(500, 512)) + [9, 9], dtype=np.int64),   # holds a
            np.array(list(range(500, 511)) + [0] + list(range(700, 712)), dtype=np.int64),  # a cut, b whole
            np.array(list(range(700, 711)), dtype=np.int64)]                        # b cut short
    assert match_docs(docs, sigs) == {0: "a", 1: "b"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
