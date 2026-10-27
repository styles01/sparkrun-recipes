#!/usr/bin/env bash
# Spark-side check of the EXL3 quants (the SPD ticket of the plan): real decode tok/s per size
# with exllamav3's aarch64 port on the box (~/exl3x, 2026-09-02) and the loss on the same held-out
# rows, next to the served NVFP4 profile measured in the same hour. NOT part of the HF jobs and
# NOT to be run while TandemLLM serves :8000 for its users: it refuses unless TANDEM_EXL3_BOX_OK=1
# is set by whoever holds the box. Weights come from the private repos; nothing is written to the
# engine's directories.
#
#   TANDEM_EXL3_BOX_OK=1 spark_check.sh fetch 3.0      # ~/exl3-quants/3.0 from the private repo
#   TANDEM_EXL3_BOX_OK=1 spark_check.sh speed 3.0      # eval/perf.py decode rows, 5 runs, M=1
#   TANDEM_EXL3_BOX_OK=1 spark_check.sh ppl 3.0        # eval/ppl.py -t heldout_rows.safetensors
#   TANDEM_EXL3_BOX_OK=1 spark_check.sh nvfp4          # the served set on the same rows, same hour
set -euo pipefail
[ "${TANDEM_EXL3_BOX_OK:-0}" = 1 ] || { echo "refusing: set TANDEM_EXL3_BOX_OK=1 only when the box is handed over (no :8000 traffic)"; exit 2; }
OWNER=0xBakeer
REPO_PREFIX=$OWNER/TandemLLM-Qwen3.8-27B-EXL3
EXL3X=${EXL3X:-$HOME/exl3x}                 # the aarch64 port (x86 sources excluded, arm64 stubs, yield pause)
Q=${EXL3_QUANTS:-$HOME/exl3-quants}
cmd=${1:-help}; bpw=${2:-}
case "$cmd" in
  fetch)
    [ -n "$bpw" ] || exit 2
    python3 - <<PY
from huggingface_hub import snapshot_download, HfApi
for repo, d in (("$REPO_PREFIX-$bpw", "$Q/$bpw"), ("$REPO_PREFIX-calibration", "$Q/calibration")):
    assert HfApi().repo_info(repo).private, repo
    snapshot_download(repo, local_dir=d, max_workers=8)
    print("fetched", repo, "->", d)
PY
    ;;
  speed)
    # decode at M=1, the shape the engine's verify sees least; the block shapes M=2..32 are the
    # kernel phase's own bench once the engine has the format. Five runs, report the median.
    [ -n "$bpw" ] || exit 2
    cd "$EXL3X" && for i in 1 2 3 4 5; do python3 eval/perf.py -m "$Q/$bpw" 2>&1 | tail -3; done
    ;;
  ppl)
    [ -n "$bpw" ] || exit 2
    cd "$EXL3X" && python3 eval/ppl.py -m "$Q/$bpw" -t "$Q/calibration/calibration/heldout_rows.safetensors"
    ;;
  nvfp4)
    # the served profile on the same rows through the engine's own gate; the gate takes texts, so
    # the rows' source texts (bench/heldout_*.txt and the sens-*.txt split) are passed as the plan
    # of tools/quality_gate.py; speed from the dashboard's NOW rate on a fresh prose prompt.
    echo "run tools/quality_gate.py with the four source texts against ops/serve.env; record tok/s from the dashboard in the same hour"
    ;;
  *) sed -n '2,13p' "$0" ;;
esac
