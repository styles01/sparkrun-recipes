"""The NemotronH refactor gate: the exact bytes of a tiny model's run, recorded once, reproduced forever.

As tests/test_refactor_gate.py for the 27B: every logit and every byte of state after a fixed sequence
of prefill, decode, a tree verify and a moved-path commit, on the board engine (engine/models/nemotron_h,
the row kernels under the CPU interpreter or on the GPU) and on the reference (fp32, CPU). The digests
depend on the torch build, the device and the kernels, so the fixture is keyed by host, device and torch
version; a host without a recorded key prints its digests and skips the comparison:

    QSE_GATE_RECORD=1 python tests/test_nemotron_h_gate.py      # (re)record this host's key

Re-recording is allowed only for a commit that is supposed to change the arithmetic, and says so.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine.models.nemotron_h.config import config_from_dict  # noqa: E402
from engine.models.nemotron_h.ref import NemotronHRef, chain  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "nemotron_h", "gate.json")
PROMPT = torch.tensor([5, 9, 3, 17, 22, 3, 40, 41, 12, 7, 63, 2, 30, 88])
TREE = ([7, 11, 13, 17, 19], [-1, 0, 0, 2, 3])


def _digest(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16
                          else t.view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def run_ref() -> dict:
    from tests.test_nemotron_h_ref import PUZZLE, random_weights
    cfg = config_from_dict(PUZZLE, "gate")
    ref = NemotronHRef(cfg, random_weights(cfg, 3))
    st = ref.new_state(64)
    out = {"prefill": ref.prefill(PROMPT, st)}
    out["decode"] = ref.decode(31, st)
    out["tree"] = ref.rows(torch.tensor(TREE[0]), TREE[1], st)
    ref.commit(st, [0, 2, 3])
    out["after_commit"] = ref.decode(5, st)
    out["chain"] = ref.rows(torch.tensor([9, 10, 11]), chain(3), st)
    for li in st.ssm:
        out[f"S.{li}"] = st.ssm[li]
        out[f"conv.{li}"] = st.conv[li]
    for li in st.k:
        out[f"k.{li}"] = st.k[li][:, :st.length]
    return {k: _digest(v) for k, v in out.items()}


def run_engine() -> dict | None:
    from engine.models.nemotron_h import kernels as K
    if not (K.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))):
        return None
    from tests.test_nemotron_h_engine import tiny
    cfg, eng, _ = tiny("puzzle")
    st = eng.new_state(256)
    out = {"prefill": eng.prefill(PROMPT, st).clone()}
    eng.stage(31, st)
    out["decode"] = eng.step(st).clone()
    st.length += 1
    eng.verify(TREE[0], TREE[1], st)
    out["tree"] = eng._rb[len(TREE[0])]["logits"].clone()
    eng.commit(st, [0, 2, 3])
    eng.stage(5, st)
    out["after_commit"] = eng.step(st).clone()
    st.length += 1
    for li in st.ssm:
        out[f"S.{li}"] = st.ssm[li]
        out[f"conv.{li}"] = st.conv[li]
    for li in st.k:
        out[f"k.{li}"] = st.k[li][:, :st.length]
    return {k: _digest(v) for k, v in out.items()}


def host_key(kind: str) -> str:
    dev = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") and kind == "engine" else "cpu"
    if dev == "cpu" and kind == "engine":
        dev = "interp"
    return f"{platform.machine()}-{sys.platform}-{dev}-torch{torch.__version__.split('+')[0]}"


def _check(kind: str):
    fn = run_ref if kind == "ref" else run_engine
    got = fn()
    if got is None:
        return "skipped: Triton on a GPU, or TRITON_INTERPRET=1"
    again = fn()
    assert got == again, f"{kind}: two runs in one process differ: {[k for k in got if got[k] != again[k]]}"
    fx = json.load(open(FIXTURE)) if os.path.isfile(FIXTURE) else {}
    key = host_key(kind)
    if os.environ.get("QSE_GATE_RECORD") == "1":
        fx.setdefault(key, {})[kind] = got
        with open(FIXTURE, "w") as f:
            json.dump(fx, f, indent=1, sort_keys=True)
            f.write("\n")
        return f"recorded {len(got)} digests for {key}"
    want = fx.get(key, {}).get(kind)
    if want is None:
        return f"SKIP compare: no digests recorded for {key} (known: {sorted(fx) or 'none'})"
    bad = [k for k in want if want[k] != got.get(k)]
    assert not bad, f"{kind}: bytes moved in {bad} (host {key})"
    return f"{len(want)} digests reproduce ({key})"


def test_reference_bytes():
    return _check("ref")


def test_engine_bytes():
    return _check("engine")


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<28} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<28} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
