#!/usr/bin/env python3
"""run_convert_omni.py — exllamav3 convert_model driver for the NVIDIA nemotron_h_omni
BF16 checkpoint (InternVL-style wrapper on GB10 / dsv41-serve:v4 build).

- NemotronHConfig.get_tensor_name_fixes: dynamically renames every
  "language_model.*" tensor to its stripped name (exact, from file headers), so
  the text backbones line up with exllamav3's NemotronH module map.
- Vision tensors (vision_model.*) are NOT renamed/mapped: no vision component in
  NemotronHConfig → tower stays out of the pack (text-only lane).
- Then runs convert_model.prepare() + main() with the same CLI flags as the script.
- Resume-safe: --resume reuses work_dir/args.json as before.
"""
import glob, json, os, struct, sys

from exllamav3.architecture.nemotronh import NemotronHConfig


def _header_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return [k for k in h if k != "__metadata__"]


def get_tensor_name_fixes(self):
    fixes = {}
    for p in glob.glob(os.path.join(self.directory, "*.safetensors")):
        for k in _header_keys(p):
            if k.startswith("language_model."):
                fixes[k] = k[len("language_model."):]
    print(f" -- omni tensor fixes: {len(fixes)} rename rules")
    return fixes


NemotronHConfig.get_tensor_name_fixes = get_tensor_name_fixes

# hand the rest to the stock pipeline
from exllamav3.conversion.convert_model import parser, prepare, main

args = parser.parse_args()
in_args, job_state, ok, err = prepare(args)
if not ok:
    print(f" !! Error: {err}")
    sys.exit(1)
main(in_args, job_state)