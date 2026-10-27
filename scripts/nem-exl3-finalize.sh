#!/bin/bash
# nem-exl3-finalize.sh — stages AFTER the encode: waits for convert driver to exit,
# verifies the pack, then uploads it to HuggingFace (aitamedia account, token file).
# Base-trash is the chain's own job (phase 3, mv to ~/trash) — not duplicated here.
exec >> /tmp/nem-exl3-finalize.log 2>&1
echo "=== finalize watcher start $(date -u) ==="

PY=/home/jaita/venvs/cb/bin/python3
[ -x "$PY" ] || PY=python3

# wait for the convert driver to exit (bracketed pattern, checks every 5 min)
while pgrep -f "run_convert_omn[i]" >/dev/null; do sleep 300; done
sleep 60

PACK=/home/jaita/MODELS/nemotron-super-vl-exl3
N=$(ls "$PACK" 2>/dev/null | grep -c safetensors)
echo "pack safetensors: $N"
if [ "$N" -lt 60 ]; then
  echo "PACK-INCOMPLETE ($N) — NOT UPLOADING — chain probably failed mid-encode; check /tmp/nemotron-exl3-overnight.log"
  exit 1
fi

REPO=aitamedia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-EXL3-3.5bpw
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
            folder_path=pack,
            repo_id=repo,
            repo_type="model",
            commit_message=(f"EXL3 3.5bpw flat-K quant of NVIDIA-Nemotron-3.5-Super-VL-120B-A12B (exllamav3 "
                            f"convert_model, mul1 codebook; experts 3.5 / attn+dense 5 / head+embed 16; "
                            f"cal=standard corpus 250x2048; quantized on DGX Spark GB10)"),
        )
        print(f"UPLOAD-OK attempt {attempt} -> https://huggingface.co/{repo}")
        ok = True
        break
    except Exception as e:
        print(f"upload attempt {attempt} failed: {repr(e)[:500]}", flush=True)
        time.sleep(180)
if not ok:
    print("UPLOAD-FAILED after 3 attempts — pack remains local at", pack)
PYEOF
echo "=== finalize done rc-persisted $(date -u) ==="