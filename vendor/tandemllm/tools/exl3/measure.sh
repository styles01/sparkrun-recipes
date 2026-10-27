#!/usr/bin/env bash
# MEASURE: per-tensor sensitivity of the BF16 model on the calibration trace (sc_measure.py,
# shaped noise; ~35-45 min on a fast GPU for a 27B) and one optimized recipe per target
# (sc_optimize.py). The measurement is a property of the model, so it is done once and serves
# every target. The 6.0 reference takes no recipe (uniform `-b 6`).
# Needs PREP's outputs in the bucket. Resumable: sc_measure writes incrementally and resumes from
# a partial file, which this job mirrors to the bucket every 5 minutes.
source "$(dirname "$0")/common.sh"
log_to $W/logs/measure.log
el "MEASURE start"
gpu_banner
setup_env
fetch_bf16
bdown cal $CAL
[ -f $CAL/cal_trace.safetensors ] || { el "no calibration trace in the bucket; run PREP first"; exit 2; }

# ---- 1. sensitivity, LDLQ-shaped noise, global anchors (no 2.0 bpw rfn probe: see the plan note)
if [ ! -f $CAL/noise_attrib.done ]; then
  SYNC_EVERY=300 start_work_sync $CAL cal
  python3 $EXL3_SRC/sc_measure.py -m $BF16 -d 0 --shaped -tr $CAL/cal_trace.safetensors \
      -o $CAL/noise_attrib.json 2>&1 | tee $CAL/sc_measure.txt
  stop_work_sync
  touch $CAL/noise_attrib.done
  bput cal/noise_attrib.json $CAL/noise_attrib.json
  bput cal/noise_attrib.done $CAL/noise_attrib.done
  bput cal/sc_measure.txt $CAL/sc_measure.txt
fi
el "measurement ready"

# ---- 2. one recipe per target. alpha 2.0 (the doc's recommendation over the fitted ~1.7-1.9),
#         no tensor below 2 bits, integer rates (the kernel phase supports fewer K values that way),
#         default tie groups (k+v, gate+up, in_proj_qkv+in_proj_z: the fused GEMMs of the fast paths).
mkdir -p $CAL/recipes
for bpw in $TARGETS; do
  [ "$bpw" = "6.0" ] && continue
  hb=$(head_bits $bpw)
  el "optimize $bpw bpw, head $hb bits"
  python3 $EXL3_SRC/sc_optimize.py -m $CAL/noise_attrib.json -b $bpw -hb $hb -al 2.0 -mink 2 -ks 1.0 \
      -o $CAL/recipes/recipe_$bpw.yaml 2>&1 | tee $CAL/recipes/optimize_$bpw.txt
  # no -t above: upstream builds the table with range() over the float --min_k and crashes
done
python3 - $CAL/recipes <<'PY'
import sys, os, yaml, json, collections
d = sys.argv[1]
summary = {}
for f in sorted(os.listdir(d)):
    if not f.endswith(".yaml"): continue
    r = yaml.safe_load(open(os.path.join(d, f)))
    hist = collections.Counter(r["tensors"].values())
    summary[f] = {"target_bpw": r.get("target_bpw"), "achieved_bpw": r.get("achieved_bpw"), "head_bits": r.get("head_bits"),
                  "tensors": len(r["tensors"]), "bits_histogram": {str(k): v for k, v in sorted(hist.items())}}
    print(f, summary[f])
json.dump(summary, open(os.path.join(d, "recipes_summary.json"), "w"), indent=1)
PY
bup $CAL/recipes cal/recipes
el "recipes ready"

mkdir -p /tmp/up/calibration/recipes
cp $CAL/noise_attrib.json $CAL/sc_measure.txt /tmp/up/calibration/
cp $CAL/recipes/* /tmp/up/calibration/recipes/
(cd /tmp/up && sha_dir .) | tee /tmp/up/calibration/sha256-measure.txt
upload_private "$REPO_PREFIX-calibration" /tmp/up "EXL3 sensitivity measurement and recipes for 2.5/3.0/3.5/4.0/4.5 bpw"
el "MEASURE done"
