"""ENG-221: the per-model state interface (engine/models/state.py) round-trips for both models.

Given an engine after a prefill, its ModelState is cloned, the engine runs on (which changes every
section), the clone is restored and the KV length set back: the engine then continues exactly as if
it had never moved. The 27B is the random 4-layer engine of tests/test_forward_tree.py; Flash-Next
is the tiny random one of tests/fn_tiny.py on the reference forward.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.state import ModelState, Qwen38State, SectionState  # noqa: E402


def test_flashnext_state_round_trip():
    from fn_tiny import tiny
    from engine.models.flashnext.ref import FlashNextRef, FNModelState, FNState
    cfg, W, tables = tiny(seed=3)
    ref = FlashNextRef(cfg, W, tables, dtype=torch.float32)
    ids = torch.randint(0, cfg.vocab_size, (16,), generator=torch.Generator().manual_seed(4))
    st = FNState(cap=64)
    ref.forward(ids[:8], st)
    ms = FNModelState(st)
    assert isinstance(ms, ModelState)
    names = sorted(ms.sections())
    assert names == ["gdn.S.0", "gdn.S.1", "gdn.S.2", "gdn.conv.0", "gdn.conv.1", "gdn.conv.2",
                     "ple.conv.1", "ple.ctx.1"], names
    snap = ms.clone()
    want = ref.forward(ids[8:16], st)                       # moves every section
    assert any(not torch.equal(a, b) for a, b in zip(ms.sections().values(), snap.sections().values()))
    ms.restore(snap)
    st.length = 8
    got = ref.forward(ids[8:16], st)
    assert torch.equal(got, want)
    return f"{len(names)} sections, {ms.nbytes()} bytes; continuation bit-identical after restore"


def test_qwen38_state_round_trip():
    try:
        import test_forward_tree as T
        eng, n = T.build(seed=5)
    except ModuleNotFoundError as e:                     # (the 27B engine imports triton)
        return f"skipped: {e}"
    ms = Qwen38State(eng.state)
    assert sorted(ms.sections()) == ["gdn.S", "gdn.conv"]
    snap = ms.clone()
    kv = (eng.kv.k.clone(), eng.kv.v.clone())
    ids = torch.randint(1, eng.cfg.vocab_size, (6,), generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        want = eng.forward(ids, start=n)
    ms.restore(snap)
    eng.kv.k.copy_(kv[0]); eng.kv.v.copy_(kv[1])
    with torch.no_grad():
        got = eng.forward(ids, start=n)
    assert torch.equal(got, want)
    return f"S and conv, {ms.nbytes()} bytes; continuation bit-identical after restore"


def test_signature_mismatch_is_refused():
    a = SectionState({"x": torch.zeros(2, 3)})
    b = SectionState({"x": torch.zeros(3, 2)})
    try:
        a.restore(b)
    except ValueError as e:
        return str(e)
    raise AssertionError("restored across shapes")


def test_empty_state_is_valid():
    s = SectionState({})
    s.restore(s.clone())
    assert s.nbytes() == 0 and s.sections() == {}
    return "a model with no recurrent state has no sections"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<36} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<36} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
