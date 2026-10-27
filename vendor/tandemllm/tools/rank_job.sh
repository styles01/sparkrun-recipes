#!/bin/bash
# RND-37: the drafter ranking fine-tune, run INSIDE one Hugging Face Jobs container (h200).
#
# Mount (set by the launcher): /work = the private bucket (code under $W/code, recorded taps under
# $W/data/train). The drafters and the target's embed/head are downloaded below.
# Secrets: HF_TOKEN (for the bucket sync).
#
# What it does: copy the taps to local disk, then per arm (b8, b16, in parallel) a --probe over the
# ranking weights at $STEPS steps, then the best weight (in-loop tree eval on a validation split of
# the training pool, never the gate's traces) again at $FINAL_STEPS. Every configuration's final and
# best weights go to $OUT, which a background loop mirrors to the bucket every few minutes, so a
# timeout loses at most the configuration that was running.
#
# Env: STEPS (4000) FINAL_STEPS (8000) LAMBDAS (0.5,1,2) LR (3e-5) SEL_LR (1e-4) SMOKE (0)
set -uo pipefail
STEPS=${STEPS:-4000}; FINAL_STEPS=${FINAL_STEPS:-8000}; LAMBDAS=${LAMBDAS:-0.5,1,2}
LR=${LR:-3e-5}; SEL_LR=${SEL_LR:-1e-4}; SMOKE=${SMOKE:-0}
BUCKET=${BUCKET:-hf://buckets/0xBakeer/qwen38-engine-work}
RUN=${RUN:-rank/run1}
W=/work/rank
OUT=/tmp/out; DATA=/tmp/data; CODE=/tmp/code
mkdir -p $OUT $DATA $CODE
log() { echo "[rank_job $(date -u +%H:%M:%S)] $*"; }

log "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
pip install -q --no-input safetensors numpy 'huggingface_hub>=1.31' >/dev/null 2>&1 || \
  pip install -q safetensors numpy huggingface_hub
cp -r $W/code/. $CODE/
cd $CODE
export PYTHONPATH=$CODE PY=python3 QWEN38_MODEL=/tmp/models/target

# The model mounts read at a few MB/s; a download of exactly the files the trainer reads is fast.
# Target: config + `outside.safetensors` (embed_tokens + lm_head, the only target tensors training
# touches). Drafters: the served bf16 checkpoints, checked against the box's sha256.
log "models: download"
t0=$(date +%s)
hf download Qwen/Qwen3.8-27B-FP8 --include "*.json" --include "outside.safetensors" --local-dir /tmp/models/target >/dev/null 2>&1 || exit 1
hf download 0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b8 --local-dir /tmp/models/b8 >/dev/null 2>&1 || exit 1
hf download 0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16 --local-dir /tmp/models/b16 >/dev/null 2>&1 || exit 1
echo "32ed444f1b2e1d443f9330a5d1c6e3dc9dcfee648eba9f282a1a351dab582f02  /tmp/models/b8/model.safetensors
843d7698e043118863847a560b0375e93db253f00352282d20ff0d6b6816356a  /tmp/models/b16/model.safetensors" \
  | sha256sum -c - || { log "drafter sha mismatch"; exit 1; }
ls /tmp/models/target | tr "\n" " "; echo
log "models: $(du -sh /tmp/models | cut -f1) in $(( $(date +%s) - t0 )) s"

log "taps: bucket -> local disk"
t0=$(date +%s)
if [ "$SMOKE" = 1 ]; then
  hf buckets sync "$BUCKET/rank/data/smoke" $DATA >/dev/null 2>&1 || cp -r $W/data/smoke/. $DATA/
else
  hf buckets sync "$BUCKET/rank/data/train" $DATA >/dev/null 2>&1 || cp -r $W/data/train/. $DATA/
fi
log "taps: $(du -sh $DATA | cut -f1), $(ls $DATA | wc -l) files in $(( $(date +%s) - t0 )) s"
VAL=$(cat $W/val.txt 2>/dev/null | tr -d '\n')
[ "$SMOKE" = 1 ] && VAL=$(python3 -c "import json;print(json.load(open('$DATA/manifest.json'))['sequences'][0]['name'])")

( while true; do sleep 300; hf buckets sync $OUT "$BUCKET/$RUN" >/dev/null 2>&1; done ) &
SYNC=$!

arm() {   # arm <b8|b16> <block> <ckpt>   (STEPS_B16 / FINAL_B16 override the b16 arm's budget)
  local A=$1 B=$2 CK=$3 specs="" l
  local STEPS=$STEPS FINAL_STEPS=$FINAL_STEPS
  [ $A = b16 ] && { STEPS=${STEPS_B16:-$STEPS}; FINAL_STEPS=${FINAL_B16:-$FINAL_STEPS}; }
  for l in ${LAMBDAS//,/ }; do specs="$specs,$A-l$l=rank@$l:$LR:$B"; done
  specs=${specs#,}
  # SEL_LRS: sweep the selector's learning rate at the first ranking weight instead
  if [ -n "${SEL_LRS:-}" ]; then
    specs=""; l=${LAMBDAS%%,*}
    for q in ${SEL_LRS//,/ }; do
      md=${q%%/*}; q2=${q#*/}; [ "$md" = "$q" ] && { md=rank; q2=$q; }
      specs="$specs,$A-$md-l$l-s$q2=$md@$l@$q2:$LR:$B"
    done
    specs=${specs#,}
  fi
  log "$A probe: $specs, $STEPS steps"
  python3 -u tools/train_dflash2.py --data $DATA --ckpt $CK --model /tmp/models/target \
    --probe "$specs" --probe-out $OUT/$A --export-final --heldout "$VAL" --steps $STEPS \
    --selector-lr $SEL_LR --eval-tree auto --eval-every $(( STEPS / 2 )) --gen-weight 1.0 \
    2>&1 | tee $OUT/$A-probe.log | grep --line-buffered -E "eval @|probe |saved|NOTE|Traceback|Error|step +(1|25|1000|2000|3000|4000|6000|8000) " | sed -u "s/^/[$A] /"
  local best
  best=$(grep -E "^  probe $A-(l|rank)" $OUT/$A-probe.log | sed -E 's/^  probe ([^ ]+) -> .*ALL=([0-9.]+).*/\2 \1/' \
         | sort -rn | head -1 | awk '{print $2}')
  log "$A best: $best ($(grep -E "^  probe $A-(l|rank)" $OUT/$A-probe.log | sed -E 's/.*probe ([^ ]+) -> .*ALL=([0-9.]+).*/\1=\2/' | tr '\n' ' '))"
  [ -z "$best" ] && return 1
  [ "$FINAL_STEPS" = 0 ] && return 0
  l=${best#$A-l}
  log "$A final: lambda $l, $FINAL_STEPS steps"
  python3 -u tools/train_dflash2.py --data $DATA --ckpt $CK --model /tmp/models/target \
    --probe "$A-l$l-s$FINAL_STEPS=rank@$l:$LR:$B" --probe-out $OUT/$A --export-final \
    --heldout "$VAL" --steps $FINAL_STEPS --selector-lr $SEL_LR --eval-tree auto \
    --eval-every $(( FINAL_STEPS / 2 )) --gen-weight 1.0 2>&1 | tee $OUT/$A-final.log | grep --line-buffered -E "eval @|probe |saved|NOTE|Traceback|Error|step +(1|25|1000|2000|3000|4000|6000|8000) " | sed -u "s/^/[$A] /"
  log "$A final: $(grep -E '^  probe ' $OUT/$A-final.log | tail -1)"
}

held_eval() {   # held_eval <block> <base ckpt> <label> [ckpt,...]: the gate's 62 + 20 traces, tree eval, bf16
  local B=$1 CK=$2 L=$3 E=${4:-base}
  [ -d /tmp/held ] || { mkdir -p /tmp/held; hf buckets sync "$BUCKET/rank/data/held" /tmp/held >/dev/null 2>&1 \
    || cp -r $W/data/held/. /tmp/held/; }
  python3 -u tools/train_dflash2.py --data /tmp/held --ckpt $CK --model /tmp/models/target --block $B \
    --eval-only --eval-ckpts "$E" --eval-tree auto --eval-blocks 1000 2>&1 | tee $OUT/held-$L.log \
    | grep --line-buffered -E "accept/block|^base|^/tmp|Traceback|Error"
  log "held $L: $(grep -vE '^\s*\[|^data|^drafter|^--train|training seq' $OUT/held-$L.log | tail -n +1 | tr '\n' ' ' | cut -c1-900)"
}

if [ "$SMOKE" = 1 ]; then
  held_eval 8 /tmp/models/b8 base-b8
  held_eval 16 /tmp/models/b16 base-b16
fi

arm b8 8 /tmp/models/b8 & P8=$!
arm b16 16 /tmp/models/b16 & P16=$!
wait $P8; R8=$?; wait $P16; R16=$?
if [ "$SMOKE" != 1 ]; then
  for A in b8 b16; do
    B=8; [ $A = b16 ] && B=16
    E=base; for d in $(ls -d $OUT/$A/*/ 2>/dev/null); do E="$E,${d%/}"; done
    held_eval $B /tmp/models/$A held-$A "$E"
  done
fi
kill $SYNC 2>/dev/null
log "arms done (b8 rc $R8, b16 rc $R16); final sync"
hf buckets sync $OUT "$BUCKET/$RUN" 2>&1 | tail -2
log done
