#!/bin/bash
# qwenfn-vision-variant.sh — build the vision-tower variant of the qwen-fn pack on spark1.
# 1. download shard 00001 (1.04GB, holds all model.visual.* weights)
# 2. extract model.visual.* tensors -> model-00008-of-00008.safetensors (fp16 in, tower passthrough)
# 3. patch the pack index (weight_map + total_size) in a NEW variant dir (serving pack untouched)
exec >> /tmp/qwenfn-visvar.log 2>&1
echo "=== vision-variant build start $(date -u) ==="
set -u
SRC=/home/jaita/models/hf/turboderp/Qwen3.8-Flash-Next-exl3
DST=/home/jaita/models/hf/turboderp/Qwen3.8-Flash-Next-exl3-vis
PY=/home/jaita/venvs/cb/bin/python3
[ -x "$PY" ] || PY=python3

if [ ! -f "$DST/model-00008-of-00008.safetensors" ]; then
  # 1. mirror the whole pack (hard-link first to save space; fallback copy) then add the tower
  mkdir -p "$DST"
  cp -al "$SRC"/. "$DST"/ 2>/dev/null || cp -a "$SRC"/. "$DST"/
  [ -f "$DST/model.safetensors.index.json" ] || { echo "FATAL: pack mirror missing index"; exit 1; }

  # fetch the source shard into tmp
  [ -s /tmp/fn-shard1.safetensors ] || hf download Qwen/Qwen3.8-Flash-Next --include "model-00001-of-00131.safetensors" --local-dir /tmp/fn-base >/dev/null 2>&1
  SZ=$(du -m /tmp/fn-base/model-00001-of-00131.safetensors 2>/dev/null | cut -f1)
  echo "source shard: ${SZ}M"
fi

$PY - <<PYEOF
import json, os, struct
import torch
from safetensors.torch import load_file, save_file

SRC = "/tmp/fn-base/model-00001-of-00131.safetensors"
DST = "/home/jaita/models/hf/turboderp/Qwen3.8-Flash-Next-exl3-vis"
sd_all = load_file(SRC)
vis = {k: v for k, v in sd_all.items() if k.startswith("model.visual.")}
assert len(vis) >= 250, f"vision tensors missing: {len(vis)}"
print("tower tensors:", len(vis), "| dtypes:", {str(t.dtype) for t in vis.values()})

ix = json.load(open(os.path.join(DST, "model.safetensors.index.json")))
for k in vis:
    if k in ix["weight_map"]:
        del ix["weight_map"][k]          # drop pre-existing vision refs (none expected)
N = ix["weight_map"].get("__max_shard__")
# name the new shard: model-00008-of-00008 if pack goes to 00007
tf_name = "model-00008-of-00008.safetensors"
for k, t in vis.items():
    ix["weight_map"][k] = tf_name
ix["metadata"]["total_size"] = ix["metadata"].get("total_size", 0) + sum(v.numel() * v.element_size() for v in vis.values())
save_file({k: v.contiguous() for k, v in vis.items()}, os.path.join(DST, tf_name),
          metadata={"format": "pt-fp16-tower"})
json.dump(ix, open(os.path.join(DST, "model.safetensors.index.json"), "w"))
print("variant built:", tf_name)
PYEOF
echo "vision-variant rc=$? $(date -u)"
du -sh "$DST"
echo "=== vision-variant build end $(date -u) ==="