#!/usr/bin/env bash
# Qualification-only launcher. It has no default action and never edits Loca/Hermes.
set -euo pipefail

readonly SOURCE_REPO="https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2.git"
readonly SOURCE_COMMIT="bac473f81a9eff0d3c1ba7dbc56a0bfc311f36ce"
readonly TARGET_REPO="RadixArk/Qwen3.8-27B-NVFP4"
readonly TARGET_REV="554ebba9b5f1b79dc11246341960360e6ef05ef4"
readonly DRAFT_REPO="maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal"
readonly DRAFT_REV="bd7a934213c47a9e7ef69eef36bb3325f47fd1f1"
readonly IMAGE="qwen38-27b-sglang-dflash2-sm121:candidate-bac473f"
readonly NAME="qwen38-dflash2-experimental"
readonly UNIT="qwen38-dflash2-experimental.service"
readonly PORT="${PORT:-8003}"
readonly HF_ROOT="${HF_ROOT:-$HOME/models/hf}"
readonly SOURCE_DIR="$HF_ROOT/sources/pangoleen-qwen3.8-27b-dgx-spark-dflash2"
readonly TARGET_DIR="$HF_ROOT/checkpoints/RadixArk--Qwen3.8-27B-NVFP4/$TARGET_REV"
readonly DRAFT_DIR="$HF_ROOT/checkpoints/maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/$DRAFT_REV"
readonly LOCK="$HF_ROOT/.qwen38-dflash2-experimental.lock"

usage() {
  cat <<'EOF'
Usage: switch-to-qwen38-sglang-dflash2-experimental.sh --check|--stage|--download|--start

No action is implicit. This qualification-only lane stores source and model
artifacts below ~/models/hf (or $HF_ROOT), does not alter Loca/Hermes, does not
stop another service, and uses Docker restart=no.
EOF
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_model_root() {
  case "$HF_ROOT" in "$HOME/models/hf"|"$HOME/models/hf"/*) ;; *) fail "HF_ROOT must be ~/models/hf or a child: $HF_ROOT" ;; esac
}
need() { command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"; }

check() {
  require_model_root
  need docker; need git; need python3; need flock
  docker info >/dev/null || fail "Docker daemon is unavailable"
  [[ "$(uname -m)" == "aarch64" ]] || fail "this candidate is DGX Spark/aarch64 only"
  [[ -d "$HF_ROOT" || ! -e "$HF_ROOT" ]] || fail "HF_ROOT is not a directory"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
    fail "exclusive lane occupied: container $NAME exists; inspect/remove it manually"
  fi
  if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$PORT" | grep -q LISTEN; then
    fail "exclusive lane occupied: port $PORT is listening"
  fi
  if [[ -d "$SOURCE_DIR/.git" ]]; then
    [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || fail "staged source is not the required commit"
  fi
  printf 'check passed: inert candidate prerequisites only\n'
}

stage() {
  require_model_root
  need docker; need git
  mkdir -p "$HF_ROOT/sources"
  if [[ -e "$SOURCE_DIR" && ! -d "$SOURCE_DIR/.git" ]]; then fail "refusing non-git source path: $SOURCE_DIR"; fi
  if [[ ! -d "$SOURCE_DIR/.git" ]]; then git clone "$SOURCE_REPO" "$SOURCE_DIR"; fi
  git -C "$SOURCE_DIR" fetch --quiet origin "$SOURCE_COMMIT"
  git -C "$SOURCE_DIR" checkout --detach --force "$SOURCE_COMMIT"
  [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || fail "source commit verification failed"
  docker build --tag "$IMAGE" --file "$SOURCE_DIR/image/Dockerfile" "$SOURCE_DIR/image"
  docker image inspect "$IMAGE" >/dev/null
  printf 'staged source %s at %s and built %s\n' "$SOURCE_REPO" "$SOURCE_COMMIT" "$IMAGE"
}

download() {
  require_model_root
  need hf
  mkdir -p "$TARGET_DIR" "$DRAFT_DIR"
  hf download "$TARGET_REPO" --revision "$TARGET_REV" --local-dir "$TARGET_DIR"
  hf download "$DRAFT_REPO" --revision "$DRAFT_REV" --local-dir "$DRAFT_DIR"
  validate_nvfp4
  printf 'downloaded verified revisions below %s\n' "$HF_ROOT"
}

validate_nvfp4() {
  [[ -f "$TARGET_DIR/config.json" ]] || fail "missing target config.json; run --download"
  [[ -n "$(find "$TARGET_DIR" -type f -name '*.safetensors' -print -quit)" ]] || fail "no target safetensors found"
  python3 - "$TARGET_DIR/config.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8'))
quant = cfg.get('quantization_config')
if not isinstance(quant, dict):
    raise SystemExit('target has no structured quantization_config; refusing name-only NVFP4 claim')
text = json.dumps(quant, sort_keys=True).lower()
markers = ('nvfp4', 'modelopt_fp4', 'modelopt_mixed', 'fp4')
if not any(marker in text for marker in markers):
    raise SystemExit('target quantization_config lacks an NVFP4/FP4 format marker')
print('NVFP4 config gate passed:', quant.get('quant_algo', quant.get('quant_method', 'structured marker present')))
PY
}

start() {
  check
  require_model_root
  need systemd-run
  docker image inspect "$IMAGE" >/dev/null || fail "missing staged image; run --stage"
  [[ -d "$TARGET_DIR" && -d "$DRAFT_DIR" ]] || fail "missing pinned checkpoints; run --download"
  validate_nvfp4
  mkdir -p "$HF_ROOT"
  exec 9>"$LOCK"
  flock -n 9 || fail "exclusive lane lock is held: $LOCK"
  # No tactic-cache mount/import: it is a separate cold/warm remeasurement experiment.
  systemd-run --user --unit="$UNIT" --collect --property=Delegate=yes --property=Restart=no \
    docker run --name "$NAME" --restart=no --cgroup-parent="$UNIT" \
      --gpus all --ipc=host --shm-size=32g -p "127.0.0.1:$PORT:$PORT" \
      -v "$TARGET_DIR:/models/target:ro" -v "$DRAFT_DIR:/models/draft:ro" \
      -e HF_HUB_OFFLINE=1 "$IMAGE" \
      sglang serve --model-path /models/target --served-model-name qwen3.8-27b-dflash2-experimental \
      --host 0.0.0.0 --port "$PORT" --trust-remote-code \
      --context-length 65536 --mem-fraction-static 0.65 --attention-backend flashinfer \
      --kv-cache-dtype bfloat16 --chunked-prefill-size 2048 --mamba-ssm-dtype bfloat16 \
      --mamba-radix-cache-strategy extra_buffer --page-size 1 \
      --speculative-algorithm DFLASH --speculative-draft-model-path /models/draft \
      --speculative-draft-model-quantization modelopt_fp4 --speculative-num-draft-tokens 16 \
      --reasoning-parser qwen3 --tool-call-parser qwen3_coder --max-running-requests 1 --enable-metrics
  printf 'started candidate in exclusive user-systemd cgroup %s; qualify before any use\n' "$UNIT"
}

[[ $# -eq 1 ]] || { usage >&2; exit 64; }
case "$1" in
  --check) check ;;
  --stage) stage ;;
  --download) download ;;
  --start) start ;;
  -h|--help) usage ;;
  *) usage >&2; exit 64 ;;
esac
