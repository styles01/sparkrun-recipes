#!/bin/bash
# nem-exl3-recover-upload.sh — post-mortem recovery, 2026-10-17 ~05:3x UTC.
# The native convert finished (rc=0, "All done", 8 shards + index, config verified)
# but the chain mis-graded it (file-count threshold calibrated for the BF16 layout).
# This script: verify pack PROPERLY -> trash base -> upload to HF -> UPLOAD-OK marker
# (appended to /tmp/nem-exl3-finalize.log so the drama3 stager proceeds).
exec >> /tmp/nem-exl3-finalize.log 2>&1
echo "=== RECOVERY-UPLOAD start $(date -u) ==="
PACK=/home/jaita/MODELS/nemotron-super-vl-exl3
BASE=/home/jaita/MODELS/nemotron-super-bf16

N=$(ls "$PACK" 2>/dev/null | grep -c '^model-.*of-.*safetensors\|^model-.*safetensors' )
CFG=$([ -f "$PACK/config.json" ] && [ -f "$PACK/quantization_config.json" ] && [ -f "$PACK/model.safetensors.index.json" ] && echo yes || echo no)
SZ=$(du -s --block-size=1G "$PACK" 2>/dev/null | cut -f1)
echo "verify: shards=$N config=$CFG size=${SZ}G"
if [ "${N:-0}" -lt 6 ] || [ "$CFG" != yes ] || [ "${SZ:-0}" -lt 40 ]; then
  echo "RECOVERY-ABORT: pack really is incomplete (shards=$N cfg=$CFG ${SZ}G) — encode needs --resume review"
  exit 1
fi

# trash the base (James's standing instruction, machine-enforced, trash-only)
if [ -d "$BASE" ]; then
  mkdir -p /home/jaita/trash
  mv "$BASE" "/home/jaita/trash/nemotron-super-bf16-$(date -u +%m%d-%H%M)" \
    && echo "BF16 base moved to ~/trash (pack verified: $N shards, ${SZ}G) $(date -u)"
fi

REPO=aitamedia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-EXL3-3.5bpw
PY=/home/jaita/venvs/cb/bin/python3
[ -x "$PY" ] || PY=python3
$PY - "$REPO" "$PACK" <<'PYEOF'
import sys, time
from huggingface_hub import HfApi
repo, pack = sys.argv[1], sys.argv[2]
api = HfApi()
ok = False
for attempt in range(1, 4):
    try:
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            folder_path=pack, repo_id=repo, repo_type="model",
            commit_message=("EXL3 3.5bpw flat-K quant of NVIDIA-Nemotron-3.5-Super-VL-120B-A12B "
                            "(exllamav3 convert_model, mul1; experts 3.5 / attn+dense 5 / head+embed 16; "
                            "standard-corpus cal 250x2048; quantized on DGX Spark GB10)"),
        )
        print(f"UPLOAD-OK attempt {attempt} -> https://huggingface.co/{repo}", flush=True)
        ok = True
        break
    except Exception as e:
        print(f"upload attempt {attempt} failed: {repr(e)[:500]}", flush=True)
        time.sleep(180)
if not ok:
    print("UPLOAD-FAILED after 3 attempts — pack remains local at", pack)
PYEOF
echo "=== RECOVERY-UPLOAD end $(date -u) ==="