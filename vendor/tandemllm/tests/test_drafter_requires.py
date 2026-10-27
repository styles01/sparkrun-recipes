"""/130: drafters declare what they read from the target; suffix stores record their tokenizer.

A drafter built for another target (other hidden width, taps past the last layer, a head or an
embedding that is not loaded) must be refused at load with the dependency named, not discovered as
an acceptance rate of zero. A suffix store holds token ids only; one recorded for another tokenizer
must be refused, and the stores written before this (the served ones, no fingerprint) are read with
one warning.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.cache import PersistentSuffixStore  # noqa: E402
from engine.drafters import Drafter, check_target  # noqa: E402
from engine.drafters.ngram import CorpusSuffixStore  # noqa: E402
from engine.tokfp import KEY, fingerprint  # noqa: E402


def _eng(hidden=64, layers=8, vocab=100, tensors=("embed_tokens.weight", "lm_head.weight")):
    cfg = SimpleNamespace(hidden_size=hidden, num_hidden_layers=layers, vocab_size=vocab)
    return SimpleNamespace(cfg=cfg, w=SimpleNamespace(t={k: object() for k in tensors}))


class _D(Drafter):
    name = "probe"

    def __init__(self, req):
        self._req = req

    def requires(self):
        return self._req


def _refused(req, eng, word):
    try:
        check_target(_D(req), eng)
    except ValueError as e:
        assert word in str(e), (word, str(e))
        return
    raise AssertionError(f"{req} accepted")


def test_requirements():
    eng = _eng()
    check_target(_D({"hidden_size": 64, "tap_layers": [1, 7], "tensors": ("lm_head.weight",),
                     "vocab_size": 100}), eng)
    check_target(Drafter(), eng)                       # declares nothing, needs nothing
    _refused({"hidden_size": 32}, eng, "hidden_size")
    _refused({"tap_layers": [8]}, eng, "tap layer 8")
    _refused({"tensors": ("mtp.fc.weight",)}, eng, "mtp.fc.weight")
    _refused({"vocab_size": 101}, eng, "vocab")
    return "met: accepted; four mismatches refused by name"


def test_dflash2_and_mtp_declare():
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.drafters.mtp import MTPDrafter
    fake = SimpleNamespace(cfg=SimpleNamespace(hidden_size=64, target_layer_ids=[1, 3]))
    r = DFlash2Drafter.requires(fake)
    assert r["hidden_size"] == 64 and r["tap_layers"] == [1, 3] and "lm_head.weight" in r["tensors"]
    r = MTPDrafter.requires(SimpleNamespace(cfg=SimpleNamespace(hidden_size=64)))
    assert "mtp.fc.weight" in r["tensors"]
    return "dflash2: taps + embed + head; mtp: its layer + embed + head"


def _tokdir(content: bytes) -> str:
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "tokenizer.json"), "wb") as f:
        f.write(content)
    return d


def test_fingerprint():
    a, b = fingerprint(_tokdir(b'{"v": 1}')), fingerprint(_tokdir(b'{"v": 2}'))
    assert a and b and a != b and len(a) == 64
    assert fingerprint(tempfile.mkdtemp()) is None and fingerprint(None) is None
    return a[:12]


def _corpus(meta):
    d = tempfile.mkdtemp()
    np.save(os.path.join(d, "tokens.npy"), np.arange(10, dtype=np.int32))
    np.save(os.path.join(d, "sa.npy"), np.arange(10, dtype=np.int64))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return d


def test_corpus_store_checked():
    fp = "a" * 64
    assert CorpusSuffixStore.load(_corpus({KEY: fp}), tokenizer_sha=fp) is not None
    try:
        CorpusSuffixStore.load(_corpus({KEY: "b" * 64}), tokenizer_sha=fp)
    except ValueError as e:
        assert "another tokenizer" in str(e)
    else:
        raise AssertionError("a store from another tokenizer was read")
    buf = io.StringIO()
    with redirect_stderr(buf):
        assert CorpusSuffixStore.load(_corpus({}), tokenizer_sha=fp) is not None
    assert "no tokenizer_sha256" in buf.getvalue()
    assert CorpusSuffixStore.load(_corpus({KEY: "b" * 64})) is not None   # no fingerprint asked: as before
    return "same: read; other: refused; legacy: read with a warning"


def test_persistent_store_records_and_checks():
    d = os.path.join(tempfile.mkdtemp(), "suffix")
    fp = "c" * 64
    PersistentSuffixStore(d, tokenizer_sha=fp).open()
    with open(os.path.join(d, "meta.json")) as f:
        assert json.load(f)[KEY] == fp
    PersistentSuffixStore(d, tokenizer_sha=fp).open()
    try:
        PersistentSuffixStore(d, tokenizer_sha="d" * 64).open()
    except ValueError:
        pass
    else:
        raise AssertionError("reopened under another tokenizer")
    return "recorded on create, refused under another tokenizer"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<40} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<40} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
