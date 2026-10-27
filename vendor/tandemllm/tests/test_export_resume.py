"""CPU tests for tools/export_resume.py: a drafter out of a trainer's resume state.

What can go wrong here is quiet. A state that lacks a trained tensor exports the RELEASED weights
for that tensor under a fine-tune's name, and the A/B then measures a hybrid nobody trained. So the
merge must take every tensor the state carries, list what it took from the base, and refuse a key
or a shape it cannot place.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import export_resume  # noqa: E402


def _raises(fn, *args):
    try:
        fn(*args)
    except SystemExit:
        return True
    return False


def _run_main(argv):
    old = sys.argv
    sys.argv = ["export_resume.py"] + argv
    try:
        export_resume.main()
    finally:
        sys.argv = old


def _base():
    return {"fc.weight": torch.zeros(4, 3, dtype=torch.bfloat16),
            "layers.0.w": torch.zeros(2, 2, dtype=torch.bfloat16),
            "candidate_selector.codebook": torch.full((5,), 7.0, dtype=torch.bfloat16)}


def test_merge_takes_every_state_tensor_and_lists_the_rest():
    trained = {"fc.weight": torch.ones(4, 3, dtype=torch.bfloat16),
               "layers.0.w": torch.full((2, 2), 2.0, dtype=torch.bfloat16)}
    w, from_base = export_resume.merge(_base(), trained)
    assert torch.equal(w["fc.weight"], trained["fc.weight"])
    assert torch.equal(w["layers.0.w"], trained["layers.0.w"])
    assert torch.equal(w["candidate_selector.codebook"], _base()["candidate_selector.codebook"])
    assert from_base == ["candidate_selector.codebook"]


def test_merge_refuses_an_unknown_key():
    assert _raises(export_resume.merge, _base(), {"layers.9.w": torch.zeros(2, 2)})


def test_merge_refuses_a_shape_mismatch():
    assert _raises(export_resume.merge, _base(), {"fc.weight": torch.zeros(3, 4)})


def _write_snapshot(d, block=16):
    os.makedirs(d)
    save_file(_base(), os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"dflash_config": {"block_size": block}}, f)


def _write_state(path, weights, block=8):
    torch.save({"version": 1, "tag": "lr3e5", "block": block, "lr": 3e-5, "train": "all",
                "step": 8000, "steps_total": 8000, "best": 4.248, "fingerprint": "f",
                "weights": weights, "opt": {"state": {}, "param_groups": []}}, path)


def test_end_to_end_writes_weights_config_block_and_manifest():
    tmp_path = Path(tempfile.mkdtemp())
    snap, out, state = tmp_path / "base", tmp_path / "out", tmp_path / "resume-lr3e5.pt"
    _write_snapshot(str(snap))
    trained = {"fc.weight": torch.ones(4, 3, dtype=torch.bfloat16),
               "layers.0.w": torch.full((2, 2), 2.0, dtype=torch.bfloat16)}
    _write_state(str(state), trained)
    _run_main(["--state", str(state), "--base", str(snap), "--out", str(out)])
    got = load_file(str(out / "model.safetensors"))
    assert torch.equal(got["fc.weight"], trained["fc.weight"])
    assert torch.equal(got["candidate_selector.codebook"], _base()["candidate_selector.codebook"])
    cfg = json.load(open(out / "config.json"))
    assert cfg["dflash_config"]["block_size"] == 8          # the block it TRAINED at, not the base's
    man = json.load(open(out / "MANIFEST.json"))
    assert man["step"] == 8000 and man["tag"] == "lr3e5" and len(man["sha256"]) == 64
    assert man["from_base"] == ["candidate_selector.codebook"]


def test_end_to_end_refuses_a_state_missing_a_trained_tensor():
    tmp_path = Path(tempfile.mkdtemp())
    snap, out, state = tmp_path / "base", tmp_path / "out", tmp_path / "resume-lr3e5.pt"
    _write_snapshot(str(snap))
    _write_state(str(state), {"fc.weight": torch.ones(4, 3, dtype=torch.bfloat16)})
    assert _raises(_run_main, ["--state", str(state), "--base", str(snap), "--out", str(out)])
    assert not (out / "model.safetensors").exists()


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
