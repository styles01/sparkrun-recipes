"""A servable drafter out of a trainer's RESUME state, for a run that never wrote a checkpoint.

The 2026-09-17 learning-rate sweep was cancelled on instruction before its export step, so
the one configuration that lifted every class on the held-out gate -- ft-b8 at `lr 3e-5`, 8,000
steps, 4.248 accepted a block against the released drafter's 4.100 -- exists only as
`train/lrprobe/state/resume-lr3e5.pt` in the bucket: weights, both Adam moments, the step and the
data cursor, 11 GB. The Spark needs the weights and nothing else.

    python tools/export_resume.py --state stage/resume-lr3e5.pt \
        --base ~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/<rev> \
        --out stage/ft-b8-lr3e5

The base is the drafter the run STARTED from. Every tensor the resume state carries replaces the
base's; a tensor it does not carry is taken from the base, and those are listed, because the
trainer holds the selector codebooks frozen and a state that silently lacked a TRAINED tensor would
export the released weights under a fine-tune's name. `--require-all-trained` is the guard for
that: every key the state carries must exist in the base with the same shape, and nothing outside
`candidate_selector.` may come from the base.

The config is the base's with `block_size` set to the block the run trained at, which is what
`tools/train_dflash2.py::export` does; a MANIFEST.json records where the weights came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.dflash2 import load_weights  # noqa: E402
from tools.train_dflash2 import export  # noqa: E402

FROZEN = ("candidate_selector.",)


def merge(base: dict[str, torch.Tensor], trained: dict[str, torch.Tensor]) -> tuple[dict, list]:
    """The base with every tensor of `trained` laid over it. Returns the merged dict and the keys
    that came from the base. Raises on a key the base does not have or a shape that differs."""
    out = dict(base)
    for k, v in trained.items():
        if k not in base:
            raise SystemExit(f"resume state carries {k}, which the base drafter does not have")
        if tuple(v.shape) != tuple(base[k].shape):
            raise SystemExit(f"{k}: resume state {tuple(v.shape)} against base {tuple(base[k].shape)}")
        out[k] = v
    from_base = sorted(k for k in base if k not in trained)
    return out, from_base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="a resume-<tag>.pt written by train_dflash2.py")
    ap.add_argument("--base", required=True, help="the drafter snapshot the run started from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-all-trained", action="store_true", default=True)
    a = ap.parse_args()

    t0 = time.time()
    # mmap: the state is 11 GB and two thirds of it is the optimiser, which is never touched here
    blob = torch.load(a.state, map_location="cpu", weights_only=False, mmap=True)
    tag, block, step = blob.get("tag"), int(blob.get("block", 0)), blob.get("step")
    print(f"[export] {a.state}: tag {tag}  block {block}  step {step}/{blob.get('steps_total')}  "
          f"lr {blob.get('lr')}  train {blob.get('train')}  best {blob.get('best')}", flush=True)
    base = load_weights(a.base, device="cpu")
    w, from_base = merge(base, blob["weights"])
    stray = [k for k in from_base if not k.startswith(FROZEN)]
    print(f"[export] {len(blob['weights'])} tensors from the state, {len(from_base)} from the base "
          f"({', '.join(sorted({k.split('.')[0] for k in from_base})) or 'none'})", flush=True)
    if a.require_all_trained and stray:
        raise SystemExit(f"tensors outside {FROZEN} missing from the state: {stray[:8]}")
    export(w, a.base, a.out, block=block)

    h = hashlib.sha256()
    with open(os.path.join(a.out, "model.safetensors"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    man = {"source_state": os.path.abspath(a.state), "base": os.path.abspath(a.base),
           "tag": tag, "block": block, "step": step, "steps_total": blob.get("steps_total"),
           "lr": blob.get("lr"), "train": blob.get("train"), "best": blob.get("best"),
           "fingerprint": blob.get("fingerprint"), "from_base": from_base,
           "sha256": h.hexdigest(), "exported": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"[export] wrote {a.out}  sha256 {h.hexdigest()[:16]}  in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
