#!/usr/bin/env bash
# Inert, exclusive Qwen3.8-27B EXL3 source-pinned experimental launcher.
# Pinned runtime: MiaAI-Lab/exllamav3@63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed
# Pinned model: Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw@19441ac874c4018295da848e250f23511361cda4
set -euo pipefail

RUNTIME_REPO="https://github.com/MiaAI-Lab/exllamav3.git"
RUNTIME_REV="63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed"
MODEL_REPO="Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw"
MODEL_REV="19441ac874c4018295da848e250f23511361cda4"
RUNTIME_DIR="${QWEN38_EXL3_RUNTIME_DIR:-$HOME/src/exllamav3-mia-qwen38}"
VENV="${QWEN38_EXL3_VENV:-$HOME/venvs/qwen38-exl3-mia}"
MODEL_DIR="${QWEN38_EXL3_MODEL_DIR:-$HOME/models/hf/Mia-AiLab--Qwen3.8-27B-EXL3-3.5bpw/$MODEL_REV}"
STATE_DIR="${QWEN38_EXL3_STATE_DIR:-$HOME/state/qwen38-exl3-native-mtp}"
PORT="${QWEN38_EXL3_PORT:-8888}"
MEMORY_MAX="${QWEN38_EXL3_MEMORY_MAX:-110G}"
MTP_N=0
DO_CHECK=0
DO_STAGE=0
DO_DOWNLOAD=0
DO_START=0

usage() {
  cat <<'USAGE'
Usage: switch-to-qwen38-exl3-native-mtp-experimental.sh [--check] [--stage] [--download] [--start] [--mtp-n 0|2]

No action is performed without an explicit action flag.
  --check       Validate aarch64/SM121 prerequisites and any staged identities.
  --stage       After prerequisites, clone/detach exactly the pinned Mia source and build it.
  --download    Download only the exact pinned model revision into its revision-named directory.
  --start       Refuse any H3/vLLM/DS4/llama co-residence, then start the 131072 batch-1 MTP-off baseline.
  --mtp-n 2     Never launches automatically: after baseline gates, write a fail-closed verified-command template.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) DO_CHECK=1 ;;
    --stage) DO_STAGE=1 ;;
    --download) DO_DOWNLOAD=1 ;;
    --start) DO_START=1 ;;
    --mtp-n) MTP_N="${2:?--mtp-n requires 0 or 2}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if (( ! DO_CHECK && ! DO_STAGE && ! DO_DOWNLOAD && ! DO_START )); then
  usage >&2
  exit 2
fi
[[ "$MTP_N" == 0 || "$MTP_N" == 2 ]] || { echo "Only baseline MTP=0 or gated MTP=2 are accepted." >&2; exit 2; }

require_prerequisites() {
  [[ "$(uname -m)" == "aarch64" ]] || { echo "Refusing: require aarch64, found $(uname -m)." >&2; return 1; }
  command -v nvidia-smi >/dev/null || { echo "Refusing: nvidia-smi is required to verify SM121." >&2; return 1; }
  local caps cap
  caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '[:space:]')"
  [[ -n "$caps" ]] || { echo "Refusing: cannot read GPU compute capability." >&2; return 1; }
  while IFS= read -r cap; do
    [[ "$cap" == "12.1" ]] || { echo "Refusing: require every GPU to be SM121/12.1; found $caps." >&2; return 1; }
  done <<< "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)"
  command -v nvcc >/dev/null || { echo "Refusing: nvcc is required for the source build." >&2; return 1; }
  command -v python3 >/dev/null || { echo "Refusing: python3 is required." >&2; return 1; }
  command -v systemd-run >/dev/null || { echo "Refusing: systemd-run is required for the MemoryMax scope." >&2; return 1; }
  systemd-run --user --version >/dev/null || { echo "Refusing: systemd user manager unavailable." >&2; return 1; }
  echo "[qwen38-exl3] prerequisites OK: aarch64, SM121 ($caps), nvcc=$(nvcc --version | tail -1)"
}

verify_runtime() {
  [[ -d "$RUNTIME_DIR/.git" ]] || { echo "Missing pinned source; run --stage." >&2; return 1; }
  [[ "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" == "$RUNTIME_REV" ]] || { echo "Runtime revision mismatch; refusing." >&2; return 1; }
  [[ -x "$VENV/bin/python" ]] || { echo "Missing staged venv; run --stage." >&2; return 1; }
}

verify_model() {
  [[ -f "$MODEL_DIR/.sparkrun-model-revision" ]] || { echo "Missing model revision marker; run --download." >&2; return 1; }
  [[ "$(<"$MODEL_DIR/.sparkrun-model-revision")" == "$MODEL_REV" ]] || { echo "Model revision marker mismatch; refusing." >&2; return 1; }
  [[ -f "$MODEL_DIR/config.json" ]] || { echo "Missing model config.json; refusing." >&2; return 1; }
}

check() {
  require_prerequisites
  if [[ -d "$RUNTIME_DIR/.git" ]]; then
    echo "[qwen38-exl3] runtime HEAD: $(git -C "$RUNTIME_DIR" rev-parse HEAD)"
  fi
  if [[ -f "$MODEL_DIR/.sparkrun-model-revision" ]]; then
    echo "[qwen38-exl3] model marker: $(<"$MODEL_DIR/.sparkrun-model-revision")"
  fi
}

stage() {
  require_prerequisites
  mkdir -p "$(dirname "$RUNTIME_DIR")"
  if [[ -d "$RUNTIME_DIR/.git" ]]; then
    git -C "$RUNTIME_DIR" fetch --depth 1 origin "$RUNTIME_REV"
  elif [[ -e "$RUNTIME_DIR" ]]; then
    echo "Refusing: runtime path exists but is not a git checkout: $RUNTIME_DIR" >&2; return 1
  else
    git clone "$RUNTIME_REPO" "$RUNTIME_DIR"
  fi
  git -C "$RUNTIME_DIR" checkout --detach "$RUNTIME_REV"
  [[ "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" == "$RUNTIME_REV" ]] || { echo "Pinned checkout failed." >&2; return 1; }
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r "$RUNTIME_DIR/requirements.txt"
  "$VENV/bin/python" -m pip install "$RUNTIME_DIR"
  "$VENV/bin/python" -c 'import exllamav3; print("exllamav3 import OK")'
  verify_runtime
}

download() {
  verify_runtime
  if [[ -e "$MODEL_DIR" && ! -f "$MODEL_DIR/.sparkrun-model-revision" ]]; then
    echo "Refusing to mix an unmarked model directory: $MODEL_DIR" >&2; return 1
  fi
  if [[ -f "$MODEL_DIR/.sparkrun-model-revision" ]] && [[ "$(<"$MODEL_DIR/.sparkrun-model-revision")" != "$MODEL_REV" ]]; then
    echo "Refusing model revision mismatch in $MODEL_DIR." >&2; return 1
  fi
  mkdir -p "$MODEL_DIR"
  MODEL_REPO="$MODEL_REPO" MODEL_REV="$MODEL_REV" MODEL_DIR="$MODEL_DIR" "$VENV/bin/python" - <<'PY'
from huggingface_hub import model_info, snapshot_download
import os
info = model_info(os.environ["MODEL_REPO"], revision=os.environ["MODEL_REV"])
if info.sha != os.environ["MODEL_REV"]:
    raise SystemExit(f"resolved {info.sha}, expected {os.environ['MODEL_REV']}")
snapshot_download(repo_id=os.environ["MODEL_REPO"], revision=os.environ["MODEL_REV"], local_dir=os.environ["MODEL_DIR"])
PY
  printf '%s\n' "$MODEL_REV" > "$MODEL_DIR/.sparkrun-model-revision"
  verify_model
}

occupancy() {
  local found=0 line
  echo "[qwen38-exl3] occupancy scan (H3/vLLM/DS4/llama):"
  while IFS= read -r line; do echo "  process: $line"; found=1; done < <(ps -axo pid=,ppid=,command= | grep -Ei '[h]3|[v]llm|[d]s4|[l]lama' || true)
  if command -v docker >/dev/null 2>&1; then
    while IFS= read -r line; do echo "  container: $line"; found=1; done < <(docker ps --format '{{.ID}} {{.Names}} {{.Image}} {{.Command}}' | grep -Ei '[h]3|[v]llm|[d]s4|[l]lama' || true)
  fi
  while IFS= read -r line; do echo "  user-unit: $line"; found=1; done < <(systemctl --user list-units --all --no-legend 2>/dev/null | grep -Ei '[h]3|[v]llm|[d]s4|[l]lama' || true)
  (( found == 0 )) || { echo "Refusing start: exclusive candidate must not co-reside; nothing was stopped." >&2; return 1; }
  echo "  none discovered"
}

write_mtp_template() {
  mkdir -p "$STATE_DIR"
  local template="$STATE_DIR/verified-mtp-launch-template.sh"
  cat > "$template" <<EOF
#!/usr/bin/env bash
# FAIL-CLOSED TEMPLATE. Do not use until a reviewer records a command verified
# against MiaAI-Lab/exllamav3@$RUNTIME_REV that exposes native MTP n=2.
# The pinned tools/serve_openai.py accepts --draft_model mtp but does not expose
# --num_draft_tokens; no MTP n=2 command is invented here.
# Required evidence: baseline gates passed, exact source/model revisions, and
# a reviewed explicit command with MTP n=2.
set -euo pipefail
echo "No verified native-MTP n=2 command is present. Fill this template only after review." >&2
exit 2
# Reviewed replacement shape (insert a verified command after --):
# exec systemd-run --user --collect --no-block -p MemoryMax=$MEMORY_MAX -p MemorySwapMax=0 -p Restart=no -- VERIFIED_COMMAND
EOF
  chmod 700 "$template"
  echo "Refusing MTP n=2: wrote explicit-command template at $template" >&2
  return 1
}

start() {
  require_prerequisites
  verify_runtime
  verify_model
  occupancy
  if [[ "$MTP_N" == 2 ]]; then write_mtp_template; fi
  echo "[qwen38-exl3] starting baseline: context=131072 request=1 batch=1 native-MTP=off"
  # A transient service gives this launch its own cgroup and an explicit no-restart policy.
  systemd-run --user --collect --no-block --unit=qwen38-exl3-native-mtp-baseline.service \
    -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 -p Restart=no \
    "$VENV/bin/python" "$RUNTIME_DIR/tools/serve_openai.py" \
      --model "$MODEL_DIR" --draft_model none --cache_size 131072 --port "$PORT" --host 127.0.0.1
}

(( DO_STAGE )) && stage
(( DO_CHECK )) && check
(( DO_DOWNLOAD )) && download
(( DO_START )) && start
