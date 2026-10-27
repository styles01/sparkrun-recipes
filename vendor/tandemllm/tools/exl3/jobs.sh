#!/usr/bin/env bash
# EXL3 quants of Qwen3.8-27B on HF Jobs: the launcher that runs on the Mac. One `hf jobs run`
# per stage, every job detached with a --timeout, the HF token only through --secrets, the
# private bucket at /work, all outputs in PRIVATE repos under 0xBakeer. Launch order:
#
#   jobs.sh setup          private repos + bucket check (idempotent, free)
#   jobs.sh push-code      tools/exl3 and the held-out texts into the bucket (free)
#   jobs.sh smoke          cpu-basic dry run: image, wheel URL, scripts parse (about $0.01)
#   jobs.sh prep           h200, ~1 h, timeout 2 h            -> wait
#   jobs.sh measure        h200, ~55 min, timeout 1 h 45      -> wait
#   jobs.sh convert-all    6 x h200 in parallel, ~95 min each, timeout from the cap
#   jobs.sh resume <bpw>   only if a convert hit its timeout (resumes from the bucket mirror)
#   jobs.sh eval <bpw>     only to redo the checks of an uploaded quant
#   jobs.sh spend          running total against the cap; jobs.sh status; jobs.sh wait <ids>
#
# The cap: CAP_USD (70) is enforced before every launch as finished spend + running jobs at their
# timeouts + the new job at its timeout. convert-all sizes the six timeouts so the whole program
# fits under the cap and refuses below MIN_CONVERT (1 h 40) unless FORCE=1.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OWNER=0xBakeer
BUCKET=$OWNER/qwen38-engine-work
REPO_PREFIX=$OWNER/TandemLLM-Qwen3.8-27B-EXL3
TARGETS="2.5 3.0 3.5 4.0 4.5 6.0"
FLAVOR=${FLAVOR:-h200}                                  # the project's default for every HF step
IMAGE=${IMAGE:-pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel}   # the image the NVFP4 build ran on
PRICE_H=${PRICE_H:-5.00}                                # h200, `hf jobs hardware` 2026-10-08
CAP_USD=${CAP_USD:-70}
MIN_CONVERT=${MIN_CONVERT:-6000}                        # seconds; below this a convert is refused
MAX_CONVERT=${MAX_CONVERT:-7800}                        # 2 h 10: expected ~95 min plus margin
TIMEOUT_PREP=${TIMEOUT_PREP:-2h}
TIMEOUT_MEASURE=${TIMEOUT_MEASURE:-105m}
TIMEOUT_EVAL=${TIMEOUT_EVAL:-1h}
VOL="-v hf://buckets/$BUCKET:/work"
cmd=${1:-help}; shift || true

need() { command -v hf >/dev/null || { echo "hf CLI missing"; exit 1; }; }
usd() { python3 -c "print(f'{$1 / 3600 * $PRICE_H:.2f}')"; }
to_secs() { python3 - "$1" <<'PY'
import re, sys
s, t = sys.argv[1], 0
for n, u in re.findall(r"(\d+)([hms])", s): t += int(n) * {"h": 3600, "m": 60, "s": 1}[u]
print(t or int(s))
PY
}
committed() { python3 "$HERE/spend.py" --json 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['committed_usd'])" || echo 0; }

# refuse a launch that could take the program over the cap
guard() {   # $1 = timeout seconds of the new job(s) in total
  local now worst
  now=$(committed); worst=$(usd "$1")
  python3 - "$now" "$worst" "$CAP_USD" <<'PY'
import sys
now, worst, cap = map(float, sys.argv[1:4])
print(f"committed so far ${now:.2f} + this launch at its timeout ${worst:.2f} = ${now + worst:.2f} (cap ${cap:.2f})")
sys.exit(1 if now + worst > cap and not __import__("os").environ.get("FORCE") else 0)
PY
}

run() {   # $1 = name, $2 = timeout, rest = command; detached, labelled, token via --secrets
  local name=${1//./p} timeout=$2; shift 2   # job names and labels allow [a-zA-Z0-9_-] only: 6.0 -> 6p0
  echo "+ hf jobs run --flavor $FLAVOR --timeout $timeout --name $name  $*"
  hf jobs run --detach --flavor "$FLAVOR" --timeout "$timeout" --name "$name" \
      -l program=exl3 -l stage="${name#tandem-exl3-}" --secrets HF_TOKEN $VOL ${ENVS[@]+"${ENVS[@]}"} \
      "$IMAGE" bash -c "$*"
}
ENVS=()

case "$cmd" in
  setup)
    need
    hf auth whoami
    hf buckets info "$BUCKET" >/dev/null
    for r in calibration $TARGETS; do
      hf repo create "$REPO_PREFIX-$r" --private --type model 2>&1 | tail -1 || true   # no-op when it exists
    done
    python3 - <<PY
from huggingface_hub import HfApi
api = HfApi()
for r in "calibration $TARGETS".split():
    info = api.repo_info("$REPO_PREFIX-" + r)
    assert info.private, "NOT PRIVATE: " + info.id
    print("private", info.id)
PY
    ;;
  push-code)
    need
    chmod +x "$HERE"/*.sh "$HERE"/*.py
    hf buckets sync "$HERE" "hf://buckets/$BUCKET/exl3/code"
    mkdir -p /tmp/exl3-texts && cp "$ROOT/bench/heldout_prose.txt" "$ROOT/bench/heldout_code.txt" /tmp/exl3-texts/
    hf buckets sync /tmp/exl3-texts "hf://buckets/$BUCKET/exl3/texts"
    hf buckets ls -R "$BUCKET/exl3/code"
    ;;
  smoke)
    need
    # no GPU, no model: the image boots, the release wheel URL resolves for this image's
    # torch/python, the stage scripts parse, the bucket mount and the code copy are readable
    FLAVOR=cpu-basic run tandem-exl3-smoke 20m 'bash /work/exl3/code/smoke.sh'
    ;;
  prep)
    need; guard "$(to_secs $TIMEOUT_PREP)"
    run tandem-exl3-prep "$TIMEOUT_PREP" 'bash /work/exl3/code/prep.sh'
    ;;
  measure)
    need; guard "$(to_secs $TIMEOUT_MEASURE)"
    run tandem-exl3-measure "$TIMEOUT_MEASURE" 'bash /work/exl3/code/measure.sh'
    ;;
  convert)
    need; bpw=${1:?bpw}; t=${TIMEOUT_CONVERT:-$MAX_CONVERT}
    guard "$t"
    ENVS=(-e RESUME=0 -e EVAL_ONLY=0)
    run "tandem-exl3-convert-$bpw" "${t}s" "bash /work/exl3/code/convert.sh $bpw"
    ;;
  convert-all)
    need
    now=$(committed)
    per=$(python3 -c "import math; left = ($CAP_USD - $now) / 6 / $PRICE_H * 3600; print(int(min($MAX_CONVERT, math.floor(left / 60) * 60)))")
    echo "committed so far \$$now; six converts get $per s each (\$$(usd $per) each, \$$(usd $((per * 6))) in all; cap \$$CAP_USD)"
    if [ "$per" -lt "$MIN_CONVERT" ] && [ -z "${FORCE:-}" ]; then
      echo "below MIN_CONVERT=$MIN_CONVERT s: stop, look at the spend, launch fewer targets with 'convert <bpw>' or FORCE=1"; exit 1
    fi
    ENVS=(-e RESUME=0 -e EVAL_ONLY=0)
    for bpw in $TARGETS; do run "tandem-exl3-convert-$bpw" "${per}s" "bash /work/exl3/code/convert.sh $bpw"; done
    ;;
  resume)
    need; bpw=${1:?bpw}; t=${TIMEOUT_CONVERT:-5400}
    guard "$t"
    ENVS=(-e RESUME=1 -e EVAL_ONLY=0)
    run "tandem-exl3-resume-$bpw" "${t}s" "bash /work/exl3/code/convert.sh $bpw"
    ;;
  eval)
    need; bpw=${1:?bpw}
    guard "$(to_secs $TIMEOUT_EVAL)"
    ENVS=(-e RESUME=0 -e EVAL_ONLY=1)
    run "tandem-exl3-eval-$bpw" "$TIMEOUT_EVAL" "bash /work/exl3/code/convert.sh $bpw"
    ;;
  spend)   python3 "$HERE/spend.py" --cap "$CAP_USD" "$@" ;;
  status)  hf jobs ls 2>/dev/null | grep -E "tandem-exl3|^name" || hf jobs ls ;;
  wait)    hf jobs wait "$@" ;;
  logs)    hf jobs logs "${1:?job id}" ;;
  *)
    sed -n '2,20p' "$0"
    ;;
esac
