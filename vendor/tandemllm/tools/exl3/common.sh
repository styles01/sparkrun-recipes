#!/usr/bin/env bash
# Shared by every EXL3 stage that runs inside an HF job (prep.sh, measure.sh, convert.sh).
# Sourced, never executed. Everything big lives under /tmp (the job's local disk); the bucket
# mount at /work is slow (20-30 MB/s, measured 2026-09-28) and carries only logs and small
# files. Big files move through `hf buckets cp/sync` (xet) and `snapshot_download`.
set -euo pipefail

OWNER=0xBakeer
BUCKET=$OWNER/qwen38-engine-work                 # private storage bucket, outlives jobs
BROOT=hf://buckets/$BUCKET/exl3                  # every EXL3 artefact under this prefix
W=/work/exl3                                     # the same prefix through the mount
REPO_PREFIX=$OWNER/TandemLLM-Qwen3.8-27B-EXL3    # private repos: -calibration, -2.5 ... -6.0
BF16_REPO=Qwen/Qwen3.8-27B
BF16_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0   # the revision the NVFP4 set was built from
EXL3_VERSION=1.6.0
EXL3_COMMIT=151539c77abc7ab7425d30da7a4e8e3c5c154e7b # turboderp-org/exllamav3 v1.6.0 (2026-10-07)
EXL3_SRC=/tmp/exl3src
BF16=/tmp/bf16
CAL=/tmp/cal
TARGETS="2.5 3.0 3.5 4.0 4.5 6.0"

export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
       HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1

t0=$(date +%s)
el() { echo "[$(( $(date +%s) - t0 )) s] $*"; }

# Log to the bucket mount as well as stdout. Any later `wait` must name explicit pids: a bare
# `wait` also waits for this tee process substitution and never returns (bash 5.2, seen 2026-09-28).
log_to() { mkdir -p "$(dirname "$1")"; exec > >(tee -a "$1") 2>&1; }

# Head bits per target: the lm_head is 1.27 B weights (5 % of the model). Low targets take a
# 4-bit head (0.63 GB), the middle a 5-6-bit one, the reference keeps 8 bits (1.27 GB, the same
# bytes as the e4m3 head the engine serves today).
head_bits() {
  case "$1" in
    2.5|3.0) echo 4 ;;
    3.5) echo 5 ;;
    4.0|4.5) echo 6 ;;
    6.0) echo 8 ;;
    *) echo 6 ;;
  esac
}

gpu_banner() {
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  df -h /tmp | tail -1
  python3 -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cap', torch.cuda.get_device_capability())"
}

# exllamav3: the release wheel that matches the image's torch minor and python, else a source
# build (nvcc is in the devel image; ~10-15 min for sm_90 only). The source checkout is needed
# in any case for convert.py, sc_*.py and eval/.
setup_env() {
  el "pip deps"
  pip install -q --no-input 'huggingface_hub[hf_xet]>=1.3' safetensors tokenizers pyyaml rich \
      marisa_trie pydantic 'llguidance>=1.7.0' pillow ninja typing_extensions \
      transformers datasets matplotlib pandas tabulate seaborn >/dev/null 2>&1 \
    || pip install -q --no-input 'huggingface_hub[hf_xet]>=1.3' safetensors tokenizers pyyaml rich \
      marisa_trie pydantic 'llguidance>=1.7.0' pillow ninja typing_extensions \
      transformers datasets matplotlib pandas tabulate seaborn
  if [ ! -f $EXL3_SRC/sc_trace.py ]; then
    # the image has no git: fetch the pinned commit as a tarball
    el "exllamav3 source at $EXL3_COMMIT"
    mkdir -p $EXL3_SRC
    python3 -c "import urllib.request; urllib.request.urlretrieve('https://codeload.github.com/turboderp-org/exllamav3/tar.gz/$EXL3_COMMIT', '/tmp/exl3src.tgz')"
    tar -xzf /tmp/exl3src.tgz -C $EXL3_SRC --strip-components=1
    echo $EXL3_COMMIT > $EXL3_SRC/COMMIT
  fi
  local pytag torchmm whl url
  pytag=$(python3 -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
  torchmm=$(python3 -c 'import torch; v = torch.__version__.split("+")[0].split("."); print(f"{v[0]}.{v[1]}.0")')
  whl="exllamav3-${EXL3_VERSION}+cu128.torch${torchmm}-${pytag}-${pytag}-linux_x86_64.whl"
  url="https://github.com/turboderp-org/exllamav3/releases/download/v${EXL3_VERSION}/${whl}"
  if ! python3 -c "import exllamav3, exllamav3.ext" >/dev/null 2>&1; then
    el "exllamav3 wheel $whl"
    if ! pip install -q --no-input --no-deps "$url" || ! python3 -c "import exllamav3, exllamav3.ext"; then
      el "wheel did not import; building from source for sm_90 (this takes 10-15 min)"
      pip uninstall -q -y exllamav3 || true
      (cd $EXL3_SRC && TORCH_CUDA_ARCH_LIST="9.0" pip install -q --no-input --no-deps --no-build-isolation .)
    fi
  fi
  python3 - <<'PY'
import exllamav3, torch, sys, json, os
from importlib.metadata import version
v = {"exllamav3": version("exllamav3"), "torch": torch.__version__, "python": sys.version.split()[0],
     "cuda_cap": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
     "exllamav3_src": open("/tmp/exl3src/COMMIT").read().strip()}
print("versions", json.dumps(v))
os.makedirs("/tmp/cal", exist_ok=True)
json.dump(v, open("/tmp/cal/versions.json", "w"), indent=1)
PY
}

# Qwen's BF16 release, pinned revision. 55.6 GB in 18 shards; 600-800 MB/s through xet (90 s).
fetch_bf16() {
  if [ -f $BF16/config.json ] && [ -f $BF16/model.safetensors.index.json ]; then return; fi
  el "download $BF16_REPO@$BF16_REV"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$BF16_REPO", revision="$BF16_REV", local_dir="$BF16", max_workers=16)
PY
  du -sh $BF16
}

bget()  { hf buckets cp "$BROOT/$1" "$2" >/dev/null; }                  # one file down
bput()  { hf buckets cp "$2" "$BROOT/$1" >/dev/null; }                  # one file up
bdown() {   # prefix -> dir; retried, the CAS CDN answers 429 when several jobs pull at once
  mkdir -p "$2"
  local i
  for i in 1 2 3 4 5 6; do
    hf buckets sync "$BROOT/$1" "$2" >/dev/null && return 0
    el "bdown $1: attempt $i failed, retry in $((i * 30)) s"; sleep $((i * 30))
  done
  return 1
}
bup()   { hf buckets sync "$1" "$BROOT/$2" >/dev/null; }                # dir -> prefix
bhas()  { hf buckets ls "$BUCKET/exl3/$1" 2>/dev/null | grep -qF "exl3/$1"; }   # ls exits 0 and prints (empty) when missing

# Checkpoint safety net for convert.py: its work dir (checkpoint every 120 s) is mirrored to the
# bucket every SYNC_EVERY seconds, so a job that hits its --timeout is resumed by `convert.sh`
# with RESUME=1 and loses at most one interval. Explicit pid, never a bare `wait`.
SYNC_PID=""
start_work_sync() {   # $1 = local dir, $2 = bucket prefix
  local every=${SYNC_EVERY:-600}
  ( while sleep "$every"; do hf buckets sync "$1" "$BROOT/$2" >/dev/null 2>&1 || true; done ) &
  SYNC_PID=$!
  el "work sync every $every s to $BROOT/$2 (pid $SYNC_PID)"
}
stop_work_sync() {
  [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null || true
  SYNC_PID=""
}

# Upload a folder into one of OUR private repos. Refuses a public repo, creates a missing one
# as private. Nothing of this program is ever public.
upload_private() {   # $1 = repo id, $2 = folder, $3 = commit message
  python3 - "$1" "$2" "$3" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder, msg = sys.argv[1:4]
api = HfApi()
if not api.repo_exists(repo):
    api.create_repo(repo, private=True, exist_ok=True)
    print("created private repo", repo)
info = api.repo_info(repo)
assert info.private, f"refusing to upload into a public repo: {repo}"
api.upload_folder(repo_id=repo, folder_path=folder, commit_message=msg)
print("uploaded", folder, "->", repo)
PY
}

sha_dir() { (cd "$1" && find . -type f -name '*.safetensors' -o -type f -name '*.json' -o -type f -name '*.yaml' | sort | xargs sha256sum); }
