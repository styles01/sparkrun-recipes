#!/bin/bash
# nemotron-exl3-overnight.sh — v2 (rewritten by oracle 2026-10-16, ~07:11 UTC)
# Runs ON spark2. Phase 1: BF16 base download (~242GB). Phase 2: exllamav3 NATIVE
# flat-K conversion (convert_model.py, resumable + checkpointed, NemotronH natively
# supported in dsv41-serve:v4 — replaces the hand-rolled hook encoder, which cannot
# work: exllamav3 Model is not an nn.Module). Phase 3: verify pack, then trash base.
set -uo pipefail
exec >> /tmp/nemotron-exl3-overnight.log 2>&1
echo "=== overnight chain v2 start $(date -u) ==="
export PATH="$HOME/.local/bin:/home/jaita/venvs/cb/bin:$PATH"

DEST=/home/jaita/MODELS/nemotron-super-bf16
WORK=/home/jaita/MODELS/nem-super-exl3-work
OUT=/home/jaita/MODELS/nemotron-super-vl-exl3

# ── Phase 1: BF16 base (resumable hf download) ──
# SKIP when already complete — hf download re-syncs the dir and would overwrite
# the remapped config.json with the upstream omni wrapper (happened 11:53-11:58)
PRE=$(ls "$DEST" 2>/dev/null | grep -c safetensors)
if [ "${PRE:-0}" -ge 60 ]; then
  echo "phase 1: base already present ($PRE safetensors) — skipping hf download, config untouched"
else
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
  --local-dir "$DEST" --max-workers 2 2>&1 | tail -4
echo "download rc=$? $(date -u)"
fi
du -sh "$DEST"
N=$(ls "$DEST" 2>/dev/null | grep -c safetensors)
echo "base-safetensors: $N"
[ "$N" -lt 20 ] && { echo "BASE INCOMPLETE - chain halted for review"; exit 1; }

# ── Phase 1.5: idempotent config remap (omni wrapper -> naked NemotronH text arch) ──
# hf download always reverts config.json to the upstream omni wrapper; remap it here
# so convert_model sees NemotronHForCausalLM no matter what phase 1 did.
echo "=== phase 1.5: config remap $(date -u) ==="
python3 - <<'PY'
import json, os, shutil
D = "/home/jaita/MODELS/nemotron-super-bf16"
P = os.path.join(D, "config.json")
c = json.load(open(P))
arcs = c.get("architectures", [])
if "NemotronHForCausalLM" in arcs:
    print("config already remapped")
else:
    L = c["llm_config"]
    orig = os.path.join(D, "config.json.orig-omni")
    if not os.path.exists(orig):
        shutil.copy2(P, orig)
    tmap = {"mamba": "M", "attention": "*", "moe": "E"}
    tsl = [float(d.get("__float__")) if isinstance(d, dict) else float(d) for d in L["time_step_limit"]]
    out = dict(L)
    out["hybrid_override_pattern"] = "".join(tmap[v] for v in L["layers_block_type"])
    out["num_hidden_layers"] = len(L["layers_block_type"])
    out["mtp_hybrid_override_pattern"] = "".join(tmap[v] for v in L.get("mtp_layers_block_type", []))
    out["time_step_limit"] = tsl
    out["architectures"] = ["NemotronHForCausalLM"]
    out["model_type"] = "nemotron_h"
    out["dtype"] = "bfloat16"
    out.pop("auto_map", None)
    json.dump(out, open(P, "w"), indent=2)
    print("remapped:", len(L["layers_block_type"]), "layers")
PY
python3 -c "import json; c=json.load(open('/home/jaita/MODELS/nemotron-super-bf16/config.json')); a=c['architectures'][0]; assert a=='NemotronHForCausalLM', a; print('config verified:', a)"

# ── Phase 2: native exllamav3 conversion ──
# -b 3.5 global bpw, --hq bumps MoE select layers, head/embeddings unquantized (16),
# bundled standard corpus calibration, checkpointed + --resume
echo "=== phase 2: convert_model start $(date -u) ==="
# --resume only if a prior job state exists; fresh encode must NOT pass it
RES=""
[ -f "$WORK/args.json" ] && RES="--resume"
# driver = stock convert_model + omni tensor-name fixes (language_model.* -> stripped)
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -e HF_HUB_OFFLINE=1 \
  -v /home/jaita/MODELS:/MODELS \
  -v /home/jaita/nem-exl3:/work \
  --entrypoint python3 \
  dsv41-serve:v4 /work/run_convert_omni.py \
  -i /MODELS/nemotron-super-bf16 \
  -w /MODELS/nem-super-exl3-work \
  -o /MODELS/nemotron-super-vl-exl3 \
  -b 3.5 --hq -hb 16 -eb 16 --devices 0 $RES
echo "convert rc=$? $(date -u)"
du -sh "$OUT" "$WORK"

# ── Phase 3: verify pack, then trash the base ──
# PACK COMPLETION (corrected 2026-10-17 ~05:2x UTC): the native convert_model
# output = ~8 x ~7GB SHARDS + index + config — NOT per-tensor files. Verify by
# shard count + index + config, NOT a file-count from the BF16 layout.
ENCODED=$(ls "$OUT" 2>/dev/null | grep -c 'model-\(.*-\)*of-\(.*-\)*safetensors\|model-.*safetensors')
echo "pack-shard-files: $ENCODED"
ls -la "$OUT" 2>/dev/null | head -5
HAS_CONFIG=$([ -f "$OUT/config.json" ] && echo yes || echo no)
HAS_QCONF=$([ -f "$OUT/quantization_config.json" ] && echo yes || echo no)
COMPLETE=$([ "${ENCODED:-0}" -ge 6 ] && [ "$HAS_CONFIG" = yes ] && echo complete || echo incomplete)
echo "pack-complete=$COMPLETE (config=$HAS_CONFIG quantcfg=$HAS_QCONF)"
python3 - <<'PY' 2>&1 | tail -5
import json
try:
    c = json.load(open("/home/jaita/MODELS/nemotron-super-vl-exl3/config.json"))
    q = c.get("quantization_config", {})
    print("out config OK; quant keys:", list(q.keys())[:6])
except Exception as e:
    print("out config CHECK FAILED:", e)
PY

# ── MANDATORY CLEANUP (James 2026-10-15): trash the 240GB BF16 base when done ──
if [ "$COMPLETE" = complete ]; then
  mkdir -p /home/jaita/trash
  mv "$DEST" "/home/jaita/trash/nemotron-super-bf16-$(date -u +%m%d-%H%M)" 2>/dev/null \
    && echo "BF16 base moved to ~/trash (pack present: $ENCODED files) $(date -u)"
  echo "trash-empty reminder: James confirms pack first, then empty ~/trash"
else
  echo "KEEPING BF16 base (pack incomplete: $ENCODED files) - manual review required"
fi
echo "=== overnight chain v2 end $(date -u) ==="