#!/bin/bash
# nemf-drama3.sh — drama3 checkpoint stager (spark2).
# WAITS for the nemotron-pack HF upload to complete (UPLOAD-OK in the finalize
# log) AND the encode driver gone; checks disk gate (>=50G free); THEN pulls
# fishaudio/drama-3 (39GB) to /home/jaita/MODELS/MEDIA/Drama3.
# Never runs beside the encode (heavy-IO law); pull starts only after upload.
exec >> /tmp/nemf-drama3.log 2>&1
echo "=== drama3 stager start $(date -u) ==="

# 1. wait for encode driver to exit
while pgrep -f "run_convert_omn[i]" >/dev/null; do sleep 300; done
echo "driver gone $(date -u)"

# 2. wait for the pack upload to complete (max ~6h patience)
for i in $(seq 1 72); do
  grep -q "UPLOAD-OK" /tmp/nem-exl3-finalize.log 2>/dev/null && { UPLOAD=1; break; }
  grep -q "UPLOAD-FAILED" /tmp/nem-exl3-finalize.log 2>/dev/null && { echo "upload FAILED — staging anyway (pack safe locally)"; UPLOAD=0; break; }
  # chain died without pack -> abort staging
  N=$(ls /home/jaita/MODELS/nemotron-super-vl-exl3 2>/dev/null | grep -c safetensors)
  [ "${N:-0}" -ge 60 ] || sleep 300
  [ "${N:-0}" -ge 60 ] && break
done
echo "proceed to gate: UPLOAD=${UPLOAD:-?} $(date -u)"

# 3. disk gate
FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
echo "free GB: $FREE"
if [ "${FREE:-0}" -lt 50 ]; then
  echo "GATE-BLOCKED: only ${FREE}G free — NOT pulling 39GB; manual review required"
  exit 1
fi

# 4. pull the checkpoint (resumable hf download to the weights-law path)
HF=/home/jaita/venvs/cb/bin/hf
[ -x "$HF" ] || HF=hf
DEST=/home/jaita/MODELS/MEDIA/Drama3
"$HF" download fishaudio/drama-3 --local-dir "$DEST" --max-workers 2 2>&1 | tail -3
echo "drama3 download rc=$? $(date -u)"
N=$(ls "$DEST" 2>/dev/null | grep -c safetensors)
[ -f "$DEST/model.safetensors.index.json" ] && echo "index present"
echo "checkpoint safetensors: $N"
[ "$N" -ge 1 ] && echo "=== DRAMA3-STAGED-CLEAN ===" || echo "=== DRAMA3-STAGE-INCOMPLETE ==="
du -sh "$DEST"