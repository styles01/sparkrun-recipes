#!/usr/bin/env bash
# CONVERT <bpw>: one EXL3 quant of Qwen3.8-27B from the BF16 release, then its quality checks,
# then the upload to the private repo <REPO_PREFIX>-<bpw>. One h200 job per target, all six in
# parallel after MEASURE. Expected ~95 min (conversion ~75, checks ~15, transfer ~5); estimate.
#
#   2.5 ... 4.5  convert.py -rcp recipe_<bpw>.yaml -cd cal_trace.safetensors  (optimized recipe,
#                in-domain calibration; head bits from the recipe)
#   6.0          convert.py -b 6 -hb 8 -cd cal_trace.safetensors                (uniform reference)
#   all          -mb 4 (MTP head quantized without calibration; the engine ignores it, exllamav3 on
#                the Spark may use it as its draft model) -vb 16 (vision tower copied unquantized,
#                0.92 GB bf16, zero decode bytes; the engine reads the tower in bf16 anyway)
#                -eb 16 (embedding unquantized; read per token row, no decode bandwidth)
#
# Env: RESUME=1 resumes a timed-out conversion from the bucket mirror of the work dir (loses at
# most SYNC_EVERY=600 s); EVAL_ONLY=1 skips the conversion and re-runs the checks on the quant in
# the private repo; UPLOAD=0 keeps everything in the bucket.
source "$(dirname "$0")/common.sh"
BPW=${1:?usage: convert.sh <bpw>}
case " $TARGETS " in *" $BPW "*) ;; *) echo "unknown target $BPW (one of: $TARGETS)"; exit 2 ;; esac
RESUME=${RESUME:-0}; EVAL_ONLY=${EVAL_ONLY:-0}; UPLOAD=${UPLOAD:-1}
REPO="$REPO_PREFIX-$BPW"
OUT=/tmp/out-$BPW
WORK=/tmp/work-$BPW
Q=/tmp/quality-$BPW
log_to $W/logs/convert-$BPW.log
el "CONVERT $BPW start (RESUME=$RESUME EVAL_ONLY=$EVAL_ONLY UPLOAD=$UPLOAD)"
gpu_banner
setup_env
fetch_bf16
bdown cal $CAL
[ -f $CAL/cal_trace.safetensors ] || { el "no calibration trace in the bucket; run PREP first"; exit 2; }
mkdir -p $Q

if [ "$EVAL_ONLY" = 1 ]; then
  el "download the quant from $REPO"
  python3 - <<PY
from huggingface_hub import snapshot_download, HfApi
assert HfApi().repo_info("$REPO").private
snapshot_download("$REPO", local_dir="$OUT", max_workers=16)
PY
else
  # ---- conversion
  if [ "$BPW" = "6.0" ]; then
    ARGS="-b 6 -hb 8"
  else
    RCP=$CAL/recipes/recipe_$BPW.yaml
    [ -f $RCP ] || { el "no recipe for $BPW in the bucket; run MEASURE first"; exit 2; }
    ARGS="-rcp $RCP"
  fi
  if [ "$RESUME" = 1 ]; then
    el "resume: pull the work dir mirror"
    bdown work/$BPW $WORK
    ls -la $WORK | head
    start_work_sync $WORK work/$BPW
    python3 $EXL3_SRC/convert.py -w $WORK -r 2>&1 | tee -a $Q/convert.txt
  else
    rm -rf $WORK $OUT; mkdir -p $WORK
    start_work_sync $WORK work/$BPW
    el "convert.py $ARGS -cd trace -mb 4 -vb 16 -eb 16"
    python3 $EXL3_SRC/convert.py -i $BF16 -o $OUT -w $WORK $ARGS -cd $CAL/cal_trace.safetensors \
        -mb 4 -vb 16 -eb 16 -cpi 120 -ss 8192 2>&1 | tee $Q/convert.txt
  fi
  stop_work_sync
  [ -f $OUT/config.json ] || { el "conversion produced no output; the work dir mirror stays in the bucket for RESUME=1"; bup $WORK work/$BPW; exit 3; }
  el "conversion done"
  du -sh $OUT
  python3 - $OUT $BPW <<'PY' | tee $Q/size.json
import sys, os, json
out, bpw = sys.argv[1], sys.argv[2]
files = {f: os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".safetensors")}
total = sum(files.values())
print(json.dumps({"bpw": bpw, "safetensors_bytes": total, "safetensors_gb": round(total / 1e9, 3), "files": files}, indent=1))
PY
  (cd $OUT && sha256sum *.safetensors *.json) > $Q/sha256.txt

  # ---- upload the weights first (the checks may fail without losing the quant)
  if [ "$UPLOAD" = 1 ]; then
    mkdir -p $OUT/calibration $OUT/quality
    [ "$BPW" = "6.0" ] || cp $CAL/recipes/recipe_$BPW.yaml $OUT/calibration/ || true                       
    cp $CAL/versions.json $Q/size.json $Q/sha256.txt $Q/convert.txt $OUT/quality/
    upload_private "$REPO" $OUT "EXL3 $BPW bpw of $BF16_REPO@${BF16_REV:0:7} (exllamav3 $EXL3_VERSION, self-calibrated)"
    hf buckets rm -r "$BROOT/work/$BPW" >/dev/null 2>&1 || true
  fi
fi

# ---- quality checks, each non-fatal. The BF16 reference numbers come from PREP (bf16_ref.json).
el "checks"
bdown cal $CAL   # again: a convert launched while PREP still ran gets the held-out files PREP wrote later
python3 $EXL3_SRC/eval/ppl.py -m $OUT -t $CAL/heldout_rows.safetensors 2>&1 | tee $Q/ppl_rows.txt || el "WARN ppl rows failed"
python3 $EXL3_SRC/eval/ppl.py -m $OUT -t $CAL/qbench_prompts_gen.json 2>&1 | tee $Q/ppl_trace.txt || el "WARN ppl trace failed"
# wikitext-2 (public, standard, disjoint from the trace): KLD both ways and top-K agreement vs BF16
python3 $EXL3_SRC/eval/model_diff.py -ma $OUT -mb $BF16 -r 50 -l 2048 -tkm 5 -d 0 2>&1 | tee $Q/model_diff.txt || el "WARN model_diff failed"
# in-domain held-out trace: KLD vs BF16 over the sampled response tokens, with the noise floor
mkdir -p /tmp/qb-$BPW
python3 $W/code/qbench_project.py --quant $OUT --ref $BF16 --trace $CAL/qbench_prompts_gen.json \
    --label "$BPW bpw" --out /tmp/qb-$BPW/project.yaml --results $Q/qbench_results.json
(cd $EXL3_SRC/eval && python3 qbench.py /tmp/qb-$BPW/project.yaml -d 0 2>&1 | tee $Q/qbench.txt) || el "WARN qbench failed"
python3 $W/code/summarize.py --label "$BPW" --ppl-rows $Q/ppl_rows.txt --ppl-trace $Q/ppl_trace.txt \
    --model-diff $Q/model_diff.txt --qbench $Q/qbench_results.json --size $Q/size.json \
    --ref $CAL/bf16_ref.json --out $Q/summary.json
cat $Q/summary.json
bup $Q quality/$BPW
if [ "$UPLOAD" = 1 ]; then
  mkdir -p /tmp/upq-$BPW/quality && cp $Q/* /tmp/upq-$BPW/quality/
  upload_private "$REPO" /tmp/upq-$BPW "quality checks for $BPW bpw: held-out rows, held-out trace, wikitext-2 KLD and top-K agreement"
fi
el "CONVERT $BPW done"
