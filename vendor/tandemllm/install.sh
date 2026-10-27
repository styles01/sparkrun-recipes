#!/usr/bin/env bash
# TandemLLM installer: Qwen3.8-27B served with StairCut on the published NVFP4 weights.
#
#   curl -fsSL https://raw.githubusercontent.com/0xBakeer/TandemLLM/main/install.sh | bash
#   bash install.sh [--dir DIR] [--port PORT] [--no-start] [--yes] ...
#
# What it does, in order (each step is skipped when it is already done, so a re-run updates):
#   1. checks: NVIDIA GPU and driver, CUDA compiler, disk, memory, Python, Hugging Face access
#   2. the engine: clones (or updates) the repository into DIR/src at a pinned release
#   3. a Python venv in DIR/venv with the pinned dependencies (torch for CUDA 13.0, Triton, ...)
#   4. the weights from Hugging Face, into the normal Hugging Face cache
#   5. a lookup corpus built from public text (tools/build_corpus.py); --no-corpus skips it
#   6. the config: DIR/run/ops/serve.env (the served profile with this machine's paths) and the
#      control script DIR/bin/tandem (start | stop | status | smoke | logs)
#   7. start, wait for /health, a smoke test; --no-start stops after step 6
#
# With --setup (docs/setup.md) it installs the code and the venv, downloads only Wren's model
# (Qwen3.5-2B, the setup guide's assistant), writes the config with no weights and the marker
# src/run/setup-pending, and starts SETUP MODE: the dashboard's #/setup page and Wren, no model of
# Qwen's. The page picks the weights, the tools and the parallelism and installs them; until then a
# start (and a reboot) comes back in setup mode. It binds 127.0.0.1 unless --host says otherwise.
#
# The primary target is one NVIDIA DGX Spark (GB10, aarch64, sm_121). Other CUDA GPUs are best
# effort: the served profile's skinny NVFP4 kernel is built for sm_121a only and is switched off
# there, and StairCut's cost tables were measured on the Spark.
#
# Nothing here needs root. The repository's own ops/serve.env is never modified.
set -euo pipefail

# ---------------------------------------------------------------------------------------------
# Pinned versions
# ---------------------------------------------------------------------------------------------

# The engine release. v0.13.0 adds Ornith-1.5-122B from our EXL3 4.0 pack, drafting with its MTP layer;
# v0.12.2 fits a Chat-tab run's tool results to the window, streams the port preview and
# lets LFM2.5 adopt its 0.9.x to 0.11.1 conversations again;
# v0.12.1 serves the first community pack on Nemotron 3.5 Super VL and takes a request with
# images on a side slot;
# v0.12.0 gives API clients that offer tools the tools profile, structured outputs with tools,
# one mutation per turn and the secret echo mask;
# v0.11.4 adds Agents mode to the Chat tab and keeps a worker's and its parent's cache rows;
# v0.11.3 hardens the Setup Guide (tandem doctor, a check before the restart, scripts through bash, Retry);
# v0.11.2 starts chatd with the engine, scopes the sandbox containers to their chatd and fixes the install's scripts and secrets;
# v0.11.1 fixes symlinks in the Chat tab sandbox's folders and prices Bolt's EXL3 packs;
# v0.11.0 adds Nemotron 3.5 Super VL 120B-A12B (the nemotron-super-vl-* profiles);
# v0.10.0 adds Nemotron 3.5 Bolt 74B-A9B (the nemotron-bolt-* profiles);
# v0.9.2 fixes the glob checksum check and the EXL3 base after install.sh --setup;
# v0.9.1 adds web search and media playback to the Chat tab and the loop guard;
# v0.9.0 adds LFM2.5-24B-A2B (the lfm25-* profiles); v0.8.3 is 0.8.2's code with the repository cleaned for the public release;
# v0.8.2 adds community packs and permission modes; v0.8.1 fetches every file a weight profile reads (OPS-95), so --setup and then
# ops/use-profile.sh exl3-<size> or flashnext-<size> work on a new box; v0.8.0 added Qwen3.8-Flash-Next
# and the Chat tab; v0.4.8 two chats beside each other and the moved-block alignment; v0.4.7 the
# conductor, memory notes, packages, the long-session caches, the proxy and the dashboard's views of other
# engines. The raw path (X-Tandem: raw) decodes as in v0.3.0; v0.2.0-staircut is the StairCut paper's
# version (merge c111fa7).
# Any tag, branch or commit of the repository works here (TANDEM_REF=main for the newest code).
TANDEM_REF="${TANDEM_REF:-v0.13.0}"
TANDEM_REPO="${TANDEM_REPO:-https://github.com/0xBakeer/TandemLLM.git}"

# The weights. The base checkpoint gives every tensor the NVFP4 overlays do not replace
# (embeddings, norms, the recurrent layers' small tensors) and the tokenizer.
BASE_REPO="Qwen/Qwen3.8-27B-FP8"
NVFP4_REPO="0xBakeer/TandemLLM-Qwen3.8-27B-NVFP4"
B8_REPO="0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b8"
B16_REPO="0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16"
# The public text the lookup corpus is built from (Wikipedia, CC BY-SA 4.0), one shard each.
WIKI_REPO="wikimedia/wikipedia"
WIKI_EN="20231101.en/train-00000-of-00041.parquet"
WIKI_DE="20231101.de/train-00000-of-00020.parquet"

TORCH_INDEX="https://download.pytorch.org/whl/cu130"
UV_VERSION="0.12.12"     # only fetched when neither uv nor python3.11 is on the machine
PYPI_INDEX="https://pypi.org/simple"
# The Python packages the engine imports, and everything they pull in, pinned to the versions the
# served engine runs (resolved for Linux aarch64 and x86_64, Python 3.11 and 3.12). torch,
# torchvision and the nvidia-* wheels come from the CUDA 13.0 index. pyarrow is for the corpus
# build only. pillow decodes a request's images (server/images.py) and torchvision is the image
# processor's pinned backend (`--image-backend`, server/images.py): without it transformers
# picks the PIL backend, whose patches differ slightly, and the engine refuses to start.
LOCK="$(cat <<'EOF'
annotated-doc==0.0.5
anyio==4.15.1
certifi==2026.7.22
click==8.5.0
cuda-bindings==13.3.1
cuda-pathfinder==1.8.1
cuda-toolkit==13.0.3.0
filelock==3.32.6
fsspec==2026.6.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.30.0
idna==3.19
jinja2==3.1.6
markdown-it-py==4.2.0
markupsafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
networkx==3.6.1
ninja==1.13.2
numpy==2.3.5
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.3.33
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
packaging==26.3
pillow==11.3.0
pyarrow==25.0.1
pygments==2.21.0
pyyaml==6.0.1
regex==2026.9.3
rich==15.0.0
safetensors==0.8.0
setuptools==84.0.0
shellingham==1.5.4
sympy==1.14.0
tokenizers==0.22.2
torch==2.13.0+cu130
torchvision==0.28.0+cu130
tqdm==4.70.0
transformers==5.12.1
triton==3.7.1
typer==0.27.2
typing-extensions==4.16.0
EOF
)"

# Sizes in GiB, for the disk check: what each repository's download adds to the HF cache (the
# base checkpoint without its MTP file, the four NVFP4 files, one drafter each, two Wikipedia
# shards), and the rest of an install.
BASE_GIB=28; NVFP4_GIB=14; DRAFTER_GIB=4; WIKI_GIB=1
LOCAL_GIB=12         # the venv (~8 GiB with the CUDA wheels), the corpus, the logs

# ---------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------

DIR="${TANDEM_DIR:-$HOME/TandemLLM}"
PORT="${TANDEM_PORT:-8000}"
HOST="${TANDEM_HOST:-127.0.0.1}"
SERVED_MODEL="${TANDEM_MODEL_NAME:-qwen3.8-27b-tandem}"
START=1
SETUP=0
FORCE_SETUP=0
HOST_SET=0
CORPUS_MODE=build
YES=0
DRY=0
SYSTEMD=0
UNINSTALL=0
REPO_SET=0

usage() {
    cat <<EOF
TandemLLM installer: Qwen3.8-27B with StairCut on one NVIDIA DGX Spark.

Usage: bash install.sh [options]
       curl -fsSL https://raw.githubusercontent.com/0xBakeer/TandemLLM/main/install.sh | bash -s -- [options]

  --dir DIR         install directory (default ~/TandemLLM)
  --port PORT       the API port (default 8000)
  --host ADDR       the address to bind (default 127.0.0.1; 0.0.0.0 serves the LAN, with no auth
                    on the OpenAI routes)
  --ref REF         the engine release: tag, branch or commit (default $TANDEM_REF)
  --repo URL        the git repository to clone (default $TANDEM_REPO)
  --hf-token TOKEN  a Hugging Face token (or set HF_TOKEN; a stored 'hf auth login' works too)
  --no-corpus       skip the lookup corpus (the lookup drafter then copies from the conversation only)
  --no-start        install and configure, but do not start the engine
  --setup           set up in the browser instead: install the code and Wren's model only, then
                    start setup mode (the dashboard's #/setup page; docs/setup.md); 127.0.0.1
                    unless --host says otherwise
  --force-setup     with --setup, on a directory that already has a configured engine
  --systemd         also install a systemd user unit, tandemllm.service (off by default)
  --yes, -y         non-interactive: answer yes to every question
  --dry-run         print every step and command, change nothing
  --uninstall       stop the engine, remove the unit and DIR (the weights stay in the HF cache)
  -h, --help        this text

Environment: TANDEM_REF, TANDEM_REPO, TANDEM_DIR, TANDEM_PORT, TANDEM_HOST, HF_TOKEN, HF_HOME.
EOF
}

die() { printf '\n[tandem] ERROR: %s\n' "$*" >&2; exit 1; }
need_arg() { [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) need_arg "$@"; DIR="$2"; shift 2 ;;
        --dir=*) DIR="${1#*=}"; shift ;;
        --port) need_arg "$@"; PORT="$2"; shift 2 ;;
        --port=*) PORT="${1#*=}"; shift ;;
        --host) need_arg "$@"; HOST="$2"; HOST_SET=1; shift 2 ;;
        --host=*) HOST="${1#*=}"; HOST_SET=1; shift ;;
        --ref) need_arg "$@"; TANDEM_REF="$2"; shift 2 ;;
        --ref=*) TANDEM_REF="${1#*=}"; shift ;;
        --repo) need_arg "$@"; TANDEM_REPO="$2"; REPO_SET=1; shift 2 ;;
        --repo=*) TANDEM_REPO="${1#*=}"; REPO_SET=1; shift ;;
        --hf-token) need_arg "$@"; export HF_TOKEN="$2"; shift 2 ;;
        --hf-token=*) export HF_TOKEN="${1#*=}"; shift ;;
        --no-corpus) CORPUS_MODE=none; shift ;;
        --no-start) START=0; shift ;;
        --setup) SETUP=1; shift ;;
        --force-setup) FORCE_SETUP=1; shift ;;
        --systemd) SYSTEMD=1; shift ;;
        --yes|-y) YES=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
done

case "$PORT" in ''|*[!0-9]*) die "--port must be a number, got '$PORT'" ;; esac
[ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || die "--port out of range: $PORT"
# shellcheck disable=SC2088  # a literal "~" that the caller's shell did not expand
case "$DIR" in "~"|"~/"*) DIR="$HOME${DIR#"~"}" ;; esac
case "$DIR" in /*) ;; *) DIR="$PWD/$DIR" ;; esac
DIR="${DIR%/}"
case "$DIR" in *[[:space:]]*) die "the install directory may not contain spaces: '$DIR'" ;; esac
[ "$DIR" != "$HOME" ] && [ -n "$DIR" ] || die "the install directory cannot be \$HOME itself"

SRC="$DIR/src"
VENV="$DIR/venv"
PY="$VENV/bin/python"
RUN="$DIR/run"
CORPUS_DIR="$DIR/corpus"
MARK="$DIR/.tandemllm-install"
UNIT="$HOME/.config/systemd/user/tandemllm.service"
HF_HUB_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
WREN_REPO="Qwen/Qwen3.5-2B"
WREN_DIR="$HOME/.qwen38-spark-engine/models/qwen3.5-2b"
if [ "$SETUP" = 1 ]; then
    CORPUS_MODE=none        # the corpus needs the base checkpoint, which the setup guide fetches later
    if [ -f "$RUN/ops/serve.env" ] && [ ! -f "$SRC/run/setup-pending" ] && [ "$FORCE_SETUP" != 1 ]; then
        die "$DIR already has a configured engine; --setup would stop it. Add --force-setup to set it up again in the browser"
    fi
    if [ "$HOST_SET" = 0 ] && [ "$DRY" != 1 ] && [ "$YES" != 1 ] && { : </dev/tty; } 2>/dev/null; then
        printf '    Open the setup page to your network (anyone who reaches port %s can run the setup)? [y/N] ' "$PORT"
        read -r ans </dev/tty || ans=""
        case "$ans" in y|Y|yes|YES) HOST=0.0.0.0 ;; *) HOST=127.0.0.1 ;; esac
    fi
fi
# fast parallel Xet downloads (the defaults leave most of a fast line unused)
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"

# ---------------------------------------------------------------------------------------------
# Logging and helpers
# ---------------------------------------------------------------------------------------------

STEP=0
STEPS=7
if [ -t 1 ]; then B=$'\033[1m'; Y=$'\033[33m'; G=$'\033[32m'; N=$'\033[0m'; else B=""; Y=""; G=""; N=""; fi
step() { STEP=$((STEP + 1)); printf '\n%s==> [%d/%d] %s%s\n' "$B" "$STEP" "$STEPS" "$*" "$N"; }
info() { printf '    %s\n' "$*"; }
ok() { printf '    %sok%s %s\n' "$G" "$N" "$*"; }
warn() { printf '    %sWARNING:%s %s\n' "$Y" "$N" "$*" >&2; }
# A failed check stops the install; under --dry-run it is reported and the plan goes on.
fail() { if [ "$DRY" = 1 ]; then warn "(dry-run, would stop here) $*"; else die "$*"; fi; }
# run CMD...: execute, or under --dry-run print it
run() {
    if [ "$DRY" = 1 ]; then printf '    [dry-run] %s\n' "$(printf '%q ' "$@")"; return 0; fi
    "$@"
}
# note TEXT: a step that --dry-run describes instead of printing a command
note() { if [ "$DRY" = 1 ]; then printf '    [dry-run] %s\n' "$*"; fi; }

# confirm QUESTION: yes under --yes; asks on the terminal when there is one (stdin is the script
# itself under curl | bash); with no terminal and no --yes the answer is the default, yes, since
# running the installer was the request. --uninstall asks for --yes explicitly instead.
confirm() {
    [ "$YES" = 1 ] && return 0
    [ "$DRY" = 1 ] && return 0
    local ans=""
    if { : </dev/tty; } 2>/dev/null; then
        printf '    %s [Y/n] ' "$1"
        read -r ans </dev/tty || ans=""
        case "$ans" in ""|y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
    fi
    return 0
}

gib_avail_at() {   # free GiB on the filesystem holding PATH (or its nearest existing parent)
    local p="$1"
    while [ ! -e "$p" ]; do p="$(dirname "$p")"; done
    df -Pk "$p" | awk 'NR == 2 { printf "%d", $4 / 1048576 }'
}
fs_of() {
    local p="$1"
    while [ ! -e "$p" ]; do p="$(dirname "$p")"; done
    df -Pk "$p" | awk 'NR == 2 { print $1 }'
}
meminfo_gib() {
    awk -v k="$1:" '$1 == k { printf "%d", $2 / 1048576; f = 1 } END { if (!f) print 0 }' /proc/meminfo 2>/dev/null || echo 0
}
hf_repo_dir() { printf '%s/models--%s' "$HF_HUB_DIR" "${1//\//--}"; }
sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -c1-16; else shasum -a 256 | cut -c1-16; fi
}

# ---------------------------------------------------------------------------------------------
# --uninstall
# ---------------------------------------------------------------------------------------------

if [ "$UNINSTALL" = 1 ]; then
    printf '%sTandemLLM uninstall: %s%s\n' "$B" "$DIR" "$N"
    [ -d "$DIR" ] || die "nothing installed at $DIR"
    [ -f "$MARK" ] || die "$DIR was not made by this installer (no $MARK); remove it by hand"
    if [ "$YES" != 1 ] && [ "$DRY" != 1 ]; then
        if { : </dev/tty; } 2>/dev/null; then
            printf '    Stop the engine and delete %s? [y/N] ' "$DIR"
            read -r ans </dev/tty || ans=""
            case "$ans" in y|Y|yes|YES) ;; *) die "cancelled" ;; esac
        else
            die "no terminal to confirm on: re-run with --uninstall --yes"
        fi
    fi
    if [ -f "$DIR/bin/tandem" ]; then run bash "$DIR/bin/tandem" stop 120 || warn "the stop reported an error"; fi
    # chatd too, should this tandem predate its stop (OPS-155)
    if [ -f "$SRC/ops/chatd.sh" ]; then run bash "$SRC/ops/chatd.sh" stop || true; fi
    # the status responder of this install (OPS-175), which outlives an engine stop on purpose
    SPORT="$(sed -n "s/^QSE_STATUS_PORT=//p" "$RUN/ops/serve.env" 2>/dev/null | tail -1 | tr -dc 0-9)"
    if [ -n "$SPORT" ] && [ -f "$RUN/ops/engine_status.py" ]; then
        run env QSE_STATUS_TREE="$SRC" python3 "$RUN/ops/engine_status.py" stop --port "$SPORT" || true
    fi
    if [ -f "$UNIT" ] && grep -q "$DIR/bin/tandem" "$UNIT" 2>/dev/null; then
        run systemctl --user disable --now tandemllm.service || true
        run rm -f "$UNIT"
        run systemctl --user daemon-reload || true
    fi
    run rm -rf "$DIR"
    printf '\n    %s %s. The weights stay in the Hugging Face cache (%s):\n' \
        "$([ "$DRY" = 1 ] && echo 'Would remove' || echo 'Removed')" "$DIR" "$HF_HUB_DIR"
    for r in "$BASE_REPO" "$NVFP4_REPO" "$B8_REPO" "$B16_REPO"; do printf '      %s\n' "$(hf_repo_dir "$r")"; done
    printf '      %s/datasets--wikimedia--wikipedia\n' "$HF_HUB_DIR"
    printf '    Delete those directories to free the space (other tools may share the base checkpoint).\n'
    exit 0
fi

printf '%sTandemLLM installer%s  ref %s, into %s, port %s%s\n' "$B" "$N" "$TANDEM_REF" "$DIR" "$PORT" \
    "$([ "$DRY" = 1 ] && echo ', DRY RUN: nothing is changed')"

# ---------------------------------------------------------------------------------------------
# 1. Checks
# ---------------------------------------------------------------------------------------------

step "Checking this machine"
SPARK=0
GPU_NAME=""
GPU_MEM_GIB=0
[ "$(uname -s)" = Linux ] || fail "TandemLLM runs on Linux with an NVIDIA GPU (this is $(uname -s))"
ARCH="$(uname -m)"
case "$ARCH" in aarch64|x86_64) ok "Linux $ARCH" ;; *) fail "unsupported CPU architecture: $ARCH" ;; esac
for tool in git curl awk df; do command -v "$tool" >/dev/null 2>&1 || fail "'$tool' is not installed"; done

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    IFS=',' read -r GPU_NAME DRIVER CC GPU_MEM <<<"$(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
        --format=csv,noheader,nounits | head -1 | sed 's/ *, */,/g')"
    ok "GPU $GPU_NAME, compute capability $CC, driver $DRIVER"
    [ "$(nvidia-smi -L | wc -l)" -gt 1 ] && info "more than one GPU: the engine uses GPU 0 (set CUDA_VISIBLE_DEVICES to choose)"
    [ "${DRIVER%%.*}" -ge 580 ] 2>/dev/null \
        || fail "driver $DRIVER is too old: the CUDA 13.0 wheels need driver 580 or newer"
    if [ "$CC" = "12.1" ]; then
        SPARK=1
        GPU_MEM_GIB="$(meminfo_gib MemTotal)"   # unified memory: the GPU's is the board's
        ok "DGX Spark class board (sm_121), ${GPU_MEM_GIB} GiB unified memory"
    else
        case "$GPU_MEM" in ''|*[!0-9]*) GPU_MEM_GIB="$(meminfo_gib MemTotal)" ;; *) GPU_MEM_GIB=$((GPU_MEM / 1024)) ;; esac
        warn "this is not a DGX Spark (GB10, sm_121). Other GPUs are best effort and untested:"
        warn "the skinny NVFP4 kernel is sm_121a-only and is switched off, StairCut's cost tables"
        warn "were measured on the Spark, and speeds will differ from the published numbers."
        [ "$GPU_MEM_GIB" -ge 40 ] || fail "$GPU_NAME has ${GPU_MEM_GIB} GiB; the engine needs at least 40 GiB of GPU memory"
        confirm "Continue on $GPU_NAME anyway?" || die "stopped at the GPU check"
    fi
else
    fail "no NVIDIA GPU found (nvidia-smi is missing or cannot talk to the driver)"
fi

# The CUDA compiler: the skinny NVFP4 kernel is compiled on first use (torch cpp_extension).
CUDA_HOME_FOUND=""
for c in "${CUDA_HOME:-}" /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda; do
    if [ -n "$c" ] && [ -x "$c/bin/nvcc" ]; then CUDA_HOME_FOUND="$c"; break; fi
done
if [ -z "$CUDA_HOME_FOUND" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME_FOUND="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [ -n "$CUDA_HOME_FOUND" ]; then
    NVCC_VER="$("$CUDA_HOME_FOUND/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    ok "nvcc $NVCC_VER at $CUDA_HOME_FOUND"
    case "$NVCC_VER" in 13.*) ;; *) warn "nvcc $NVCC_VER: the engine is built and tested with CUDA 13.0" ;; esac
elif [ "$SPARK" = 1 ]; then
    fail "no CUDA compiler (nvcc) found: install the CUDA 13.0 toolkit, or set CUDA_HOME"
else
    warn "no CUDA compiler (nvcc) found; only needed for the sm_121a kernel, which is off here"
fi
if ! command -v g++ >/dev/null 2>&1 && [ "$SPARK" = 1 ]; then
    fail "no C++ compiler (g++): the NVFP4 kernel build needs it (apt install g++)"
fi

# Disk: the weights go to the Hugging Face cache, the rest to DIR.
# still_to_get DIR GIB: GiB of a repository not in the cache yet (0 once it is about complete)
still_to_get() {
    local have=0
    [ -d "$1" ] && have="$(du -sk "$1" 2>/dev/null | awk '{printf "%d", $1 / 1048576 + 0.5}')"
    [ "$have" -ge "$2" ] && echo 0 || echo $(( $2 - have ))
}
TO_GET=$(( $(still_to_get "$(hf_repo_dir "$BASE_REPO")" "$BASE_GIB") + $(still_to_get "$(hf_repo_dir "$NVFP4_REPO")" "$NVFP4_GIB")
         + $(still_to_get "$(hf_repo_dir "$B8_REPO")" "$DRAFTER_GIB") + $(still_to_get "$(hf_repo_dir "$B16_REPO")" "$DRAFTER_GIB") ))
[ "$SETUP" = 1 ] && TO_GET=5           # Wren's model alone; the setup guide checks the weights' room
[ "$CORPUS_MODE" = build ] && [ ! -f "$CORPUS_DIR/meta.json" ] \
    && TO_GET=$((TO_GET + $(still_to_get "$HF_HUB_DIR/datasets--wikimedia--wikipedia" "$WIKI_GIB")))
NEED_HF=$((TO_GET + 1))
NEED_DIR=$LOCAL_GIB
[ -d "$VENV" ] && NEED_DIR=2
FREE_HF="$(gib_avail_at "$HF_HUB_DIR")"
FREE_DIR="$(gib_avail_at "$DIR")"
if [ "$(fs_of "$HF_HUB_DIR")" = "$(fs_of "$DIR")" ]; then
    NEED=$((NEED_HF + NEED_DIR))
    if [ "$FREE_DIR" -ge "$NEED" ]; then
        ok "disk: ${FREE_DIR} GiB free, about ${NEED} GiB needed (${TO_GET} GiB to download)"
    else
        fail "disk: ${FREE_DIR} GiB free, about ${NEED} GiB needed (${TO_GET} GiB to download)"
    fi
elif [ "$FREE_HF" -lt "$NEED_HF" ]; then
    fail "disk: ${FREE_HF} GiB free for the HF cache ($HF_HUB_DIR), about ${NEED_HF} GiB needed"
elif [ "$FREE_DIR" -lt "$NEED_DIR" ]; then
    fail "disk: ${FREE_DIR} GiB free at $DIR, about ${NEED_DIR} GiB needed"
else
    ok "disk: ${FREE_HF} GiB free for the HF cache, ${FREE_DIR} GiB at $DIR"
fi

# Memory: the served stack takes about 64 GiB of the board at load (docs/operations.md).
MEM_AVAIL="$(meminfo_gib MemAvailable)"
MIN_FREE=64; [ "$SPARK" = 1 ] || MIN_FREE=16      # host RAM on a discrete-GPU box
[ "$SETUP" = 1 ] && MIN_FREE=8                     # setup mode holds Wren alone
if [ "$MEM_AVAIL" -ge "$MIN_FREE" ]; then
    ok "memory: ${MEM_AVAIL} GiB available"
elif [ "$START" = 1 ]; then
    fail "memory: ${MEM_AVAIL} GiB available, the engine needs about ${MIN_FREE} GiB to load. Stop other GPU work, or install with --no-start"
else
    warn "memory: ${MEM_AVAIL} GiB available; about ${MIN_FREE} GiB must be free when you start the engine"
fi
if pgrep -f 'server/app\.py' >/dev/null 2>&1; then
    if [ "$START" = 1 ]; then
        fail "a TandemLLM engine is already running (pgrep 'server/app.py'). One engine per board: stop it, or install with --no-start"
    fi
    warn "an engine is already running; this install will not start a second one (--no-start)"
fi
if [ "$START" = 1 ] && command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT\$"; then
    fail "port $PORT is in use; choose another with --port"
fi

# Python 3.11, the served engine's: tests/test_grammar.py passes under 3.11 and fails under 3.12
# (the structured-output grammar's empty-mask check). uv when it is on PATH, else a system
# python3.11 with venv, else uv (pinned) is fetched into DIR/uv and brings its own 3.11 into
# DIR/python -- user-local, no root.
PY_MODE=""
UV=""
if command -v uv >/dev/null 2>&1; then
    PY_MODE=uv; UV="$(command -v uv)"; ok "uv $(uv --version | awk '{print $2}') (the venv gets Python 3.11)"
elif command -v python3.11 >/dev/null 2>&1 && python3.11 -c 'import venv, ensurepip' 2>/dev/null; then
    PY_MODE=venv; ok "$(python3.11 -V) with venv"
else
    PY_MODE=fetch-uv; UV="$DIR/uv/uv"
    info "no uv and no python3.11: uv $UV_VERSION goes into $DIR/uv and fetches Python 3.11 (user-local)"
fi

# Hugging Face access: every repository the installer and the weight profiles read is public; a token
# is used when there is one, and needed only for a repository someone made private or gated.
HF_TOKEN_FILE="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
HF_AUTH=""
if [ -n "${HF_TOKEN:-}" ]; then HF_AUTH="$HF_TOKEN"; ok "HF token from HF_TOKEN/--hf-token"
elif [ -r "$HF_TOKEN_FILE" ]; then HF_AUTH="$(cat "$HF_TOKEN_FILE")"; ok "HF token from $HF_TOKEN_FILE"
else info "no Hugging Face token: fine, the weights are public"
fi
if [ "$SETUP" = 1 ]; then
    code="$(curl -s -o /dev/null -w '%{http_code}' -m 20 "https://huggingface.co/api/models/$WREN_REPO" || true)"
    [ "$code" = 200 ] && ok "Hugging Face: $WREN_REPO is readable" \
        || warn "cannot read $WREN_REPO (HTTP ${code:-none}): setup mode starts without Wren's chat"
fi
for r in "$BASE_REPO" "$NVFP4_REPO" "$B8_REPO" "$B16_REPO"; do
    [ "$SETUP" = 1 ] && break             # the setup guide checks the weights it is asked for
    # the token goes to curl on stdin, never on its command line
    code="$( { [ -n "$HF_AUTH" ] && printf 'header = "Authorization: Bearer %s"\n' "$HF_AUTH"; true; } \
        | curl -s -K - -o /dev/null -w '%{http_code}' -m 20 "https://huggingface.co/api/models/$r" || true)"
    case "$code" in
        200) ;;
        401|403|404) fail "cannot read $r on Hugging Face (HTTP $code). The repositories are public, so this is a network or proxy problem, or the repository moved; a token (--hf-token, HF_TOKEN or 'hf auth login') helps only if it was made private" ;;
        *) fail "cannot reach huggingface.co for $r (HTTP ${code:-none}); check the network" ;;
    esac
done
[ "$SETUP" = 1 ] || ok "Hugging Face: the four model repositories are readable"
unset HF_AUTH

cat <<EOF

    The plan: engine $TANDEM_REF into $SRC, a venv, about $TO_GET GiB of
    downloads into $HF_HUB_DIR, $([ "$CORPUS_MODE" = build ] && echo "a public lookup corpus," || echo "no corpus,") the config$([ "$START" = 1 ] && echo ", then start on port $PORT" || echo " (no start)").
EOF
confirm "Go ahead?" || die "cancelled"
# a chatd running now keeps the code it started with: the last message says how to restart it
CHATD_WAS_UP=0
if [ "$DRY" != 1 ] && [ -f "$SRC/ops/chatd.sh" ] && bash "$SRC/ops/chatd.sh" status >/dev/null 2>&1; then
    CHATD_WAS_UP=1
fi
chat_note() {
    if [ "$CHATD_WAS_UP" = 1 ] && bash "$SRC/ops/chatd.sh" status >/dev/null 2>&1; then
        printf '\n  chatd was running before this update and keeps the old code until: %s/bin/tandem chat restart\n' "$DIR"
    fi
}
if [ "$DRY" != 1 ]; then mkdir -p "$DIR" "$RUN/ops" "$DIR/bin" "$DIR/state"; touch "$MARK"; fi

# ---------------------------------------------------------------------------------------------
# 2. The engine
# ---------------------------------------------------------------------------------------------

step "Engine source: $TANDEM_REF"
export GIT_TERMINAL_PROMPT=0     # a private repository fails here with a message, not a prompt
if [ -d "$SRC/.git" ]; then
    [ "$REPO_SET" = 1 ] && run git -C "$SRC" remote set-url origin "$TANDEM_REPO"
    if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
        fail "$SRC has local changes; commit or stash them (git -C $SRC stash), then re-run"
    fi
    run git -C "$SRC" fetch --quiet --tags --force --prune origin || die "git fetch failed (origin: $(git -C "$SRC" remote get-url origin))"
else
    [ -e "$SRC" ] && die "$SRC exists and is not a git checkout; move it away"
    run git clone --quiet "$TANDEM_REPO" "$SRC" \
        || die "git clone $TANDEM_REPO failed. Check the network, or pass another --repo"
fi
if [ "$DRY" = 1 ] && [ ! -d "$SRC/.git" ]; then
    note "git checkout --detach $TANDEM_REF"
else
    TARGET=""
    for cand in "refs/tags/$TANDEM_REF" "refs/remotes/origin/$TANDEM_REF" "$TANDEM_REF"; do
        if TARGET="$(git -C "$SRC" rev-parse -q --verify "$cand^{commit}" 2>/dev/null)"; then break; fi
        TARGET=""
    done
    [ -n "$TARGET" ] || die "no tag, branch or commit '$TANDEM_REF' in $TANDEM_REPO"
    run git -C "$SRC" -c advice.detachedHead=false checkout --quiet --detach "$TARGET"
    ok "$SRC at $TANDEM_REF (${TARGET:0:7}), version $(cat "$SRC/VERSION" 2>/dev/null || echo '?')"
fi

# ---------------------------------------------------------------------------------------------
# 3. Python
# ---------------------------------------------------------------------------------------------

step "Python environment: $VENV"
LOCK_SHA="$(printf '%s\n' "$LOCK" | sha_of)"
if [ -x "$PY" ] && ! "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' 2>/dev/null; then
    info "$VENV is not Python 3.11 ($("$PY" -V 2>&1)); rebuilding it"
    run rm -rf "$VENV"
fi
if [ -x "$PY" ] && [ "$(cat "$VENV/.tandem-lock" 2>/dev/null)" = "$LOCK_SHA" ]; then
    ok "up to date (lock $LOCK_SHA)"
else
    if [ "$PY_MODE" = fetch-uv ]; then
        # uv's own Python downloads stay inside DIR too
        export UV_PYTHON_INSTALL_DIR="$DIR/python"
        if [ ! -x "$UV" ]; then
            note "curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | UV_INSTALL_DIR=$DIR/uv UV_NO_MODIFY_PATH=1 sh"
            if [ "$DRY" != 1 ]; then
                curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" \
                    | env UV_INSTALL_DIR="$DIR/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null \
                    || die "could not install uv; install python3.11 with venv, or uv, and re-run"
                [ -x "$UV" ] || UV="$DIR/uv/bin/uv"
                [ -x "$UV" ] || die "uv was installed but not found under $DIR/uv"
            fi
        fi
    fi
    if [ ! -x "$PY" ]; then
        if [ "$PY_MODE" = venv ]; then run python3.11 -m venv "$VENV"
        else run "$UV" venv --quiet --python 3.11 "$VENV"
        fi
    fi
    if [ "$DRY" != 1 ]; then printf '%s\n' "$LOCK" > "$RUN/requirements.lock"; fi
    info "installing $(printf '%s\n' "$LOCK" | wc -l | tr -d ' ') pinned packages (torch 2.13.0+cu130, torchvision 0.28.0+cu130, triton 3.7.1, transformers 5.12.1, pillow 11.3.0, ...)"
    if [ "$PY_MODE" = venv ]; then
        run "$PY" -m pip install --disable-pip-version-check -r "$RUN/requirements.lock" \
            --index-url "$PYPI_INDEX" --extra-index-url "$TORCH_INDEX"
    else
        run "$UV" pip install --python "$PY" -r "$RUN/requirements.lock" \
            --index-url "$PYPI_INDEX" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
    fi
    if [ "$DRY" != 1 ]; then printf '%s\n' "$LOCK_SHA" > "$VENV/.tandem-lock"; fi
fi
if [ "$DRY" != 1 ]; then
    "$PY" - <<'EOF' || die "the Python environment does not see the GPU (see above)"
import torch, triton, transformers, PIL, torchvision
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
cc = ".".join(map(str, torch.cuda.get_device_capability(0)))
print(f"    ok torch {torch.__version__} (CUDA {torch.version.cuda}), triton {triton.__version__}, "
      f"transformers {transformers.__version__}; {torch.cuda.get_device_name(0)} sm_{cc.replace('.', '')}")
# the image path: what server/images.py decodes with and the processor class the engine's
# pinned backend (torchvision) resolves to for Qwen3.8 (model type qwen3_5)
from transformers.models.auto.image_processing_auto import _load_class_with_fallback, IMAGE_PROCESSOR_MAPPING_NAMES
cls = _load_class_with_fallback(IMAGE_PROCESSOR_MAPPING_NAMES["qwen3_5"], "torchvision")
name = getattr(cls, "__name__", str(cls))
assert not name.endswith("Pil"), f"the image processor resolved to {name} (PIL backend): torchvision is missing or too old"
print(f"    ok pillow {PIL.__version__}, torchvision {torchvision.__version__}; image processor {name} (torchvision backend)")
EOF
fi

# ---------------------------------------------------------------------------------------------
# 4. Weights
# ---------------------------------------------------------------------------------------------

if [ "$SETUP" = 1 ]; then
step "Wren's model (the setup guide's assistant): $WREN_REPO"
# no weights yet; the base snapshot's folder is where ops/use-profile.sh fetches it for a 27B profile
# (OPS-95), this install's HF cache (HF_HOME), which the engine's default would not follow
SNAP_BASE="$(hf_repo_dir "$BASE_REPO")/snapshots"; SNAP_NV=""; SNAP_B8=""; SNAP_B16=""
if [ -f "$WREN_DIR/config.json" ] && [ -f "$WREN_DIR/tokenizer.json" ]; then
    ok "already here: $WREN_DIR"
elif [ "$DRY" = 1 ]; then
    note "snapshot_download $WREN_REPO into $WREN_DIR (about 4.3 GB)"
else
    "$PY" - "$WREN_REPO" "$WREN_DIR" <<'EOF' && ok "$WREN_DIR" \
        || warn "Wren's model did not download: setup mode starts without its chat (the page works without it)"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=8)
EOF
fi
else
step "Weights (Hugging Face cache: $HF_HUB_DIR)"
# hf_get REPO PATTERN...: download (or verify) the files matching the patterns; prints the
# snapshot directory on stdout, progress on stderr. Resumes an interrupted download.
hf_get() {
    "$PY" - "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError, GatedRepoError
repo, patterns = sys.argv[1], sys.argv[2:]
try:
    path = snapshot_download(repo, allow_patterns=patterns, max_workers=16)
except (RepositoryNotFoundError, GatedRepoError) as e:
    sys.exit(f"cannot read {repo}: {type(e).__name__}. It is public: check the network (a token, --hf-token or HF_TOKEN, helps only if it was made private).")
except HfHubHTTPError as e:
    sys.exit(f"download of {repo} failed: {e}")
print(path)
EOF
}
# The board shares one memory between CPU and GPU, and page cache is not handed back fast enough
# for the GPU's allocations (docs/operations.md): drop the cache of what is downloaded as it lands.
drop_cache_loop() {
    while :; do "$PY" "$SRC/tools/drop_page_cache.py" "$@" >/dev/null 2>&1 || true; sleep 5; done
}
if [ "$DRY" = 1 ]; then
    note "snapshot_download $BASE_REPO  (layers-*.safetensors, outside.safetensors, tokenizer and config; not mtp.safetensors)"
    note "snapshot_download $NVFP4_REPO  (mlp, gdn, attn, head-fp8 .safetensors)"
    note "snapshot_download $B8_REPO, $B16_REPO  (config.json, model.safetensors)"
    SNAP_BASE="$(hf_repo_dir "$BASE_REPO")/snapshots/<rev>"; SNAP_NV="$(hf_repo_dir "$NVFP4_REPO")/snapshots/<rev>"
    SNAP_B8="$(hf_repo_dir "$B8_REPO")/snapshots/<rev>"; SNAP_B16="$(hf_repo_dir "$B16_REPO")/snapshots/<rev>"
else
    drop_cache_loop "$(hf_repo_dir "$BASE_REPO")" "$(hf_repo_dir "$NVFP4_REPO")" \
        "$(hf_repo_dir "$B8_REPO")" "$(hf_repo_dir "$B16_REPO")" &
    DROPPER=$!
    trap 'kill "$DROPPER" 2>/dev/null || true' EXIT
    info "$BASE_REPO"
    SNAP_BASE="$(hf_get "$BASE_REPO" '*.json' '*.jinja' 'merges.txt' 'LICENSE' 'layers-*.safetensors' 'outside.safetensors' | tail -n 1)" \
        || die "base checkpoint download failed"
    info "$NVFP4_REPO"
    SNAP_NV="$(hf_get "$NVFP4_REPO" 'mlp.safetensors' 'gdn.safetensors' 'attn.safetensors' 'head-fp8.safetensors' \
        'README.md' 'LICENSE' 'quality/sha256.txt' | tail -n 1)" || die "NVFP4 download failed"
    info "$B8_REPO"
    SNAP_B8="$(hf_get "$B8_REPO" 'config.json' 'model.safetensors' 'README.md' 'LICENSE' | tail -n 1)" || die "b8 drafter download failed"
    info "$B16_REPO"
    SNAP_B16="$(hf_get "$B16_REPO" 'config.json' 'model.safetensors' 'README.md' 'LICENSE' | tail -n 1)" || die "b16 drafter download failed"
    kill "$DROPPER" 2>/dev/null || true
    trap - EXIT
    "$PY" "$SRC/tools/drop_page_cache.py" "$SNAP_BASE" "$SNAP_NV" "$SNAP_B8" "$SNAP_B16" >/dev/null 2>&1 || true
    for f in "$SNAP_BASE/config.json" "$SNAP_BASE/tokenizer.json" "$SNAP_BASE/outside.safetensors" \
             "$SNAP_NV/mlp.safetensors" "$SNAP_NV/gdn.safetensors" "$SNAP_NV/attn.safetensors" \
             "$SNAP_NV/head-fp8.safetensors" "$SNAP_B8/model.safetensors" "$SNAP_B16/model.safetensors"; do
        [ -s "$f" ] || die "missing after download: $f"
    done
    ok "base      $SNAP_BASE"
    ok "NVFP4     $SNAP_NV"
    ok "drafters  $SNAP_B8"
    ok "          $SNAP_B16"
fi

fi

# ---------------------------------------------------------------------------------------------
# 5. Lookup corpus
# ---------------------------------------------------------------------------------------------

# The served engine's corpus (38 M tokens) is not published. This builds a smaller one of the same
# kinds of public text with the repository's own tool: English and German Wikipedia (one shard
# each) and the Python sources of the installed transformers, torch and triton packages, about
# 31 M tokens. The paper's numbers were measured with the unpublished corpus; with this one, or
# none, the lookup drafter proposes different continuations and the speed differs (not measured).
step "Lookup corpus"
if [ "$CORPUS_MODE" = none ]; then
    info "skipped (--no-corpus): the lookup drafter copies from the conversation only"
    CORPUS_VALUE=""
elif [ -f "$CORPUS_DIR/meta.json" ]; then
    ok "already built: $CORPUS_DIR ($(sed -n 's/.*"n_tokens": \([0-9]*\).*/\1/p' "$CORPUS_DIR/meta.json") tokens)"
    CORPUS_VALUE="$CORPUS_DIR"
else
    CORPUS_VALUE="$CORPUS_DIR"
    if [ "$DRY" = 1 ]; then
        note "hf_hub_download $WIKI_REPO $WIKI_EN and $WIKI_DE (public, CC BY-SA 4.0, 1.2 GB)"
        note "python tools/build_corpus.py --model <base> --out $CORPUS_DIR --parquet <en>:text::12000000 --parquet <de>:text::7000000 --src <site-packages>/{transformers,torch,triton} --src engine --src tools"
    else
        info "fetching one English and one German Wikipedia shard ($WIKI_REPO, CC BY-SA 4.0)"
        WIKI_PATHS="$("$PY" - "$WIKI_REPO" "$WIKI_EN" "$WIKI_DE" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
repo = sys.argv[1]
for f in sys.argv[2:]:
    print(hf_hub_download(repo, f, repo_type="dataset"))
EOF
)" || die "Wikipedia download failed (or re-run with --no-corpus)"
        EN_PQ="$(sed -n 1p <<<"$WIKI_PATHS")"; DE_PQ="$(sed -n 2p <<<"$WIKI_PATHS")"
        SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
        info "tokenising and indexing (CPU only, a few minutes)"
        rm -rf "$CORPUS_DIR.partial"
        (cd "$SRC" && nice -n 10 "$PY" tools/build_corpus.py --model "$SNAP_BASE" --out "$CORPUS_DIR.partial" \
            --parquet "$EN_PQ:text::12000000" --parquet "$DE_PQ:text::7000000" \
            --src "$SITE/transformers:6000000" --src "$SITE/torch:5000000" --src "$SITE/triton:1500000" \
            --src "$SRC/engine" --src "$SRC/tools" --max-tokens 40000000) 2>&1 | sed 's/^/    | /' \
            || die "the corpus build failed (re-run with --no-corpus to skip it)"
        [ -f "$CORPUS_DIR.partial/meta.json" ] || die "the corpus build wrote no meta.json"
        mv "$CORPUS_DIR.partial" "$CORPUS_DIR"
        ok "built: $CORPUS_DIR"
    fi
fi

# ---------------------------------------------------------------------------------------------
# 6. Config and control script
# ---------------------------------------------------------------------------------------------

step "Configuration: $RUN/ops/serve.env"
TEMPLATE="$SRC/ops/serve.env"
if [ ! -f "$TEMPLATE" ] && [ "$DRY" = 1 ]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd || true)"
    [ -n "$here" ] && [ -f "$here/ops/serve.env" ] && TEMPLATE="$here/ops/serve.env"
fi
q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }    # single-quoted for the env file
# The keys this install sets. Every other line of the release's ops/serve.env -- StairCut, the
# tree, the kernels, every QWEN38_* switch -- is copied as it is.
ENV_KEYS=()
ENV_ASSIGN=()
setv() { ENV_KEYS+=("$1"); ENV_ASSIGN+=("TV_$1=$2"); }
setv REPO "$(q "$SRC")"
setv PY "$(q "$PY")"
setv PYTHONPATH ""
setv HOST "$(q "$HOST")"
setv PORT "$PORT"
setv SERVED_MODEL "$(q "$SERVED_MODEL")"
if [ "$SETUP" = 1 ]; then
    # no weights yet: the weight profile the setup guide installs sets them
    setv NV "''"; setv HEAD "''"; setv CKPT8 "''"; setv CKPT16 "''"
else
    setv NV "$(q "$SNAP_NV/mlp.safetensors,$SNAP_NV/gdn.safetensors,$SNAP_NV/attn.safetensors")"
    setv HEAD "$(q "$SNAP_NV/head-fp8.safetensors")"
    setv CKPT8 "$(q "$SNAP_B8")"
    setv CKPT16 "$(q "$SNAP_B16")"
fi
setv CORPUS "$(q "$CORPUS_VALUE")"
setv QSE_USAGE_LEDGER "$(q "$DIR/state/usage/ledger.sqlite3")"
setv DROP_PAGE_CACHE 0              # bin/tandem drops the weights' page cache with this install's paths
setv QWEN38_MODEL "$(q "$SNAP_BASE")"
# a cache of its own (HF_HOME or HF_HUB_CACHE here): the engine, the setup guide and ops/use-profile.sh
# look for the base and fetch into the same one (OPS-143)
[ "$HF_HUB_DIR" = "$HOME/.cache/huggingface/hub" ] || setv HF_HUB_CACHE "$(q "$HF_HUB_DIR")"
setv QSE_SECRETS "$(q "$DIR/state/secrets.env")"
# the status responder (ops/engine_status.py): it answers for the engine while a Setup Guide install
# restarts it, so the page shows the restart and the job instead of errors (OPS-175)
[ "$PORT" -le 65526 ] && setv QSE_STATUS_PORT "$((PORT + 9))"
if [ -n "$CUDA_HOME_FOUND" ]; then setv CUDA_HOME "$(q "$CUDA_HOME_FOUND")"; fi
if [ "$SPARK" != 1 ]; then
    # best effort off the Spark: the skinny kernel is sm_121a code; a smaller context below 100 GiB
    setv QWEN38_NVFP4_SKINNY 0
    setv QWEN38_SKINNY_SRUN 0
    if [ "$GPU_MEM_GIB" -lt 100 ]; then setv MAX_LEN 65536; fi
fi
render_env() {
    printf '# Written by install.sh on %s for %s. Regenerated on every run of the installer:\n' "$(date -Iseconds 2>/dev/null || date)" "$DIR"
    printf '# put your own settings in %s/local.env, which is read last.\n' "$RUN"
    printf '# The template is the release'"'"'s ops/serve.env (%s); these keys are this machine'"'"'s:\n' "$TANDEM_REF"
    printf '#   %s\n\n' "${ENV_KEYS[*]}"
    # The keys the template lacks go in before its weight profile (OPS-143), which may replace them: an
    # EXL3 profile's QWEN38_MODEL is base-plain on a box without the FP8 snapshot, a standalone
    # community pack's is the pack. Appended after it, install.sh's values replaced the profile's.
    # shellcheck disable=SC2016  # the $0/$1 are awk's
    env "${ENV_ASSIGN[@]}" awk -v keys="${ENV_KEYS[*]}" '
        BEGIN { n = split(keys, k, " "); for (i = 1; i <= n; i++) want[k[i]] = 1 }
        function mine() {
            if (done++) return
            printf "# set by install.sh (not in the template)\n"
            for (i = 1; i <= n; i++) if (!(k[i] in tmpl)) print k[i] "=" ENVIRON["TV_" k[i]]
            printf "\n"
        }
        FNR == NR { if (match($0, /^[A-Za-z_][A-Za-z0-9_]*=/)) tmpl[substr($0, 1, RLENGTH - 1)] = 1; next }
        /^# Weight profile \(ops\/use-profile\.sh\)/ { mine() }
        match($0, /^[A-Za-z_][A-Za-z0-9_]*=/) {
            key = substr($0, 1, RLENGTH - 1)
            if (key in want) { if (!(key in seen)) print key "=" ENVIRON["TV_" key]; seen[key] = 1; next }
        }
        { print }
        END { if (!done) { printf "\n"; mine() } }' "$TEMPLATE" "$TEMPLATE"
    printf '\nif [ -f %s ]; then . %s; fi\n' "$(q "$RUN/local.env")" "$(q "$RUN/local.env")"
}
if [ ! -f "$TEMPLATE" ]; then
    note "render $SRC/ops/serve.env with this machine's paths into $RUN/ops/serve.env"
else
    if [ "$DRY" = 1 ]; then
        note "would write $RUN/ops/serve.env; the lines this install changes:"
        render_env | grep -E "^($(IFS='|'; echo "${ENV_KEYS[*]}"))=" | sed 's/^/        /'
    else
        render_env > "$RUN/ops/serve.env.tmp"
        # it is sourced by bash: prove it parses before it replaces the old one
        bash -n "$RUN/ops/serve.env.tmp" || die "the generated serve.env does not parse"
        mv "$RUN/ops/serve.env.tmp" "$RUN/ops/serve.env"
        # the release's own service scripts, next to this env file (they source ./serve.env)
        # and what the setup guide's apply job runs beside them (server/setup/apply.py); roles.sh is
        # what start.sh runs after /health, watchdog.sh the cron line of docs/operations.md, hold.sh
        # a box-alone command (OPS-161). Not copied: chatd.sh runs from the checkout (it finds the
        # chatd package there and this file through DIR), gate.sh is the release gate of a
        # development checkout. tests/test_install_ops_copy.py keeps this list and their "$HERE/"
        # calls in step.
        for s in start.sh stop.sh engines.sh use-profile.sh make-secrets.sh engine_status.py \
                 fetch_weights.py roles.sh watchdog.sh hold.sh; do install -m 755 "$SRC/ops/$s" "$RUN/ops/$s"; done
        if [ "$SETUP" = 1 ]; then
            # setup mode until the setup guide's apply job removes it (a reboot comes back to it)
            mkdir -p "$SRC/run"
            printf 'HOST=%s\n' "$HOST" > "$SRC/run/setup-pending"
        elif [ -f "$SRC/run/setup-pending" ]; then
            # a plain install configures the engine itself: an old --setup's marker would make
            # every start setup mode again
            rm -f "$SRC/run/setup-pending"
            info "removed an earlier --setup's run/setup-pending: the engine starts configured"
        fi
        mkdir -p "$DIR/state/usage"
        ok "wrote $RUN/ops/serve.env (StairCut on: QWEN38_LEN_SWITCH=$(sed -n 's/^QWEN38_LEN_SWITCH=//p' "$RUN/ops/serve.env"), QWEN38_LEN_MODE=$(sed -n 's/^QWEN38_LEN_MODE=//p' "$RUN/ops/serve.env"))"
    fi
fi

# DIR/bin/tandem: start | stop | restart | status | smoke | logs | env | doctor | chat
write_control() {
    printf '#!/usr/bin/env bash\n# TandemLLM control script, written by install.sh.\nTANDEM_DIR=%q\n' "$DIR"
    cat <<'CTL'
set -euo pipefail
RUN="$TANDEM_DIR/run"
ENV_FILE="$RUN/ops/serve.env"
# tandem doctor (OPS-171): checks this install and repairs what is safe (file modes, the run/ops
# copies, stale pid files, the setup marker); before the config loads, so a config that does not
# load is one of its findings. --check changes nothing.
doctor() {
    local py="$TANDEM_DIR/venv/bin/python"
    [ -x "$py" ] || py=python3
    (cd "$TANDEM_DIR/src" && PYTHONPATH="$TANDEM_DIR/src" "$py" -m server.setup.doctor "$@")
}
if [ "${1:-}" = doctor ]; then shift; doctor "$@"; exit $?; fi
[ -f "$ENV_FILE" ] || { echo "[tandem] no $ENV_FILE; re-run install.sh" >&2; exit 1; }
# a command that fails in the config (or a file it sources) names its file and line instead of
# ending this script silently under set -e (OPS-159)
trap 'echo "[tandem] ${BASH_SOURCE[0]:-$ENV_FILE}: line $LINENO failed (status $?); the config did not load" >&2; exit 1' ERR
set -a
# shellcheck source=/dev/null
. "$ENV_FILE"
set +a
trap - ERR
URL="http://127.0.0.1:$PORT"
healthy() { curl -sf -m 3 "$URL/health" >/dev/null 2>&1; }
engine_alive() { [ -f "$REPO/logs/engine.pid" ] && kill -0 "$(cat "$REPO/logs/engine.pid")" 2>/dev/null; }
# chatd, the Chat tab's service (OPS-155): its own process, from the checkout's ops/chatd.sh. start
# starts it once the engine answers, stop stops it first (open runs end `interrupted`, Continue
# resumes them), restart leaves it running; QSE_CHATD=off (local.env) keeps start from starting it.
CHATD="$REPO/ops/chatd.sh"
chat_up() { bash "$CHATD" status >/dev/null 2>&1; }
chat_auto() {
    [ "${QSE_CHATD:-on}" = off ] && return 0
    [ -f "$REPO/run/setup-pending" ] && return 0       # setup mode: the Chat tab waits for the setup
    chat_up && return 0
    bash "$CHATD" start || echo "[tandem] WARNING: chatd did not start (above); the engine serves without the Chat tab. Log: $REPO/logs/chatd.log" >&2
}
chat_line() {
    if [ "${QSE_CHATD:-on}" = off ] && ! chat_up; then echo "[tandem] chat: off (QSE_CHATD=off)"; return 0; fi
    bash "$CHATD" status 2>&1 | head -1 || true
}

memcheck() {
    local avail need
    [ "${TANDEM_SKIP_MEMCHECK:-0}" = 1 ] && return 0
    if nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q '^12\.1'; then
        avail=$(awk '$1 == "MemAvailable:" { printf "%d", $2 / 1048576 }' /proc/meminfo)
        need="${TANDEM_MIN_FREE_GIB:-64}"
    else
        avail=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1 / 1024}')
        need="${TANDEM_MIN_FREE_GIB:-36}"
    fi
    if [ "$avail" -lt "$need" ]; then
        echo "[tandem] ${avail} GiB free, the engine needs about ${need} GiB to load; stop other GPU work first" >&2
        echo "         (TANDEM_SKIP_MEMCHECK=1 skips this check)" >&2
        exit 1
    fi
}

start() {
    if healthy; then echo "[tandem] already healthy on :$PORT"; chat_auto; return 0; fi
    # what it repairs or finds is said; a finding does not stop the start (start.sh decides)
    doctor --quiet || echo "[tandem] doctor found something to look at (above); starting anyway" >&2
    memcheck
    local rc=0 t0 limit
    bash "$RUN/ops/start.sh" || rc=$?
    if [ "$rc" != 0 ]; then
        # start.sh waits 600 s; a first load also compiles kernels, so keep waiting while it lives
        engine_alive || { echo "[tandem] the engine did not start; log: $REPO/logs/engine.log" >&2; return 1; }
        limit="${TANDEM_HEALTH_TIMEOUT:-1800}"; t0=$(date +%s)
        echo "[tandem] still loading; waiting up to ${limit}s for /health"
        until healthy; do
            engine_alive || { echo "[tandem] the engine exited; tail of the log:" >&2; tail -30 "$REPO/logs/engine.log" >&2; return 1; }
            [ $(( $(date +%s) - t0 )) -lt "$limit" ] || { echo "[tandem] not healthy after ${limit}s" >&2; return 1; }
            sleep 10
        done
    fi
    # the weights are on the device now: drop their page cache (docs/operations.md). Only the paths
    # serve.env names: an empty one (setup mode has no weights) was ".", the whole working directory
    local -a nv drop=()
    local f
    IFS=, read -r -a nv <<<"$NV"
    for f in "$QWEN38_MODEL" ${nv[@]+"${nv[@]}"} "$HEAD" "$CKPT8" "$CKPT16"; do [ -n "$f" ] && drop+=("$f"); done
    [ "${#drop[@]}" = 0 ] || "$PY" "$REPO/tools/drop_page_cache.py" "${drop[@]}" 2>&1 | tail -1 || true
    echo "[tandem] healthy: $URL/v1"
    chat_auto
}

smoke() {
    "$PY" - "$URL" "$SERVED_MODEL" <<'PY'
import json, sys, time, urllib.request
url, model = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
    models = [m["id"] for m in json.load(r)["data"]]
print(f"[smoke] /v1/models: {', '.join(models)}")
body = {"model": model, "stream": True, "stream_options": {"include_usage": True},
        "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content":
                      "Write a Python function that returns the n-th Fibonacci number iteratively, "
                      "with a docstring and two doctests."}]}
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter(); first = None; n = 0; usage = None; text = []
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[5:])
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                if first is None:
                    first = time.perf_counter()
                n += 1
                text.append(piece)
t1 = time.perf_counter()
tokens = (usage or {}).get("completion_tokens") or n
if first is None or tokens < 2:
    sys.exit("[smoke] FAILED: no content came back")
print("[smoke] reply: " + "".join(text)[:160].replace("\n", " ") + " ...")
print(f"[smoke] {tokens} tokens, first token after {1000 * (first - t0):.0f} ms, "
      f"{(tokens - 1) / (t1 - first):.1f} tok/s decode (one request, a cold engine's first)")
PY
}

case "${1:-}" in
    start) start ;;
    stop) bash "$CHATD" stop || true; bash "$RUN/ops/stop.sh" "${2:-60}" ;;
    restart) bash "$RUN/ops/stop.sh" "${2:-60}"; start ;;
    status)
        rc=0
        if healthy; then echo "[tandem] healthy on :$PORT"; curl -s -m 5 "$URL/v1/models"; echo
        elif engine_alive; then echo "[tandem] loading (pid $(cat "$REPO/logs/engine.pid"))"
        else echo "[tandem] not running"; rc=1; fi
        chat_line
        exit "$rc" ;;
    smoke) smoke ;;
    logs) tail -n "${2:-100}" -f "$REPO/logs/engine.log" ;;
    env) cat "$ENV_FILE" ;;
    chat)
        case "${2:-status}" in
            start|stop|restart|status) bash "$CHATD" "${2:-status}" ;;
            logs) tail -n "${3:-100}" -f "$REPO/logs/chatd.log" ;;
            *) echo "usage: $0 chat start | stop | restart | status | logs [n]" >&2; exit 2 ;;
        esac ;;
    *) echo "usage: $0 start | stop [grace-s] | restart | status | smoke | logs [n] | env | doctor [--check] | chat start|stop|restart|status|logs" >&2; exit 2 ;;
esac
CTL
}
if [ "$DRY" = 1 ]; then
    note "would write $DIR/bin/tandem (start | stop | restart | status | smoke | logs | env | chat)"
else
    write_control > "$DIR/bin/tandem.tmp" && chmod 755 "$DIR/bin/tandem.tmp" && mv "$DIR/bin/tandem.tmp" "$DIR/bin/tandem"
    ok "wrote $DIR/bin/tandem"
fi

if [ "$SYSTEMD" = 1 ]; then
    info "systemd user unit: $UNIT"
    info "one engine per board: enable it only if nothing else (cron, another unit) starts one here;"
    info "bin/tandem start refuses while any engine is alive, but two supervisors can still race at boot"
    if [ "$DRY" = 1 ]; then
        note "would write $UNIT (ExecStart=/bin/bash $DIR/bin/tandem start) and enable it"
    else
        mkdir -p "$(dirname "$UNIT")"
        cat > "$UNIT" <<EOF
[Unit]
Description=TandemLLM: Qwen3.8-27B with StairCut, OpenAI-compatible API on :$PORT
After=network-online.target

[Service]
# bin/tandem start runs the release's ops/start.sh (through bash, as every call of a script here: a
# copy that lost its execute bit still runs, OPS-167), which starts the server in the background
# and returns once /health answers, then chatd (the Chat tab's service); both stay in this unit's
# cgroup. bin/tandem stop stops chatd first, then the engine, so systemctl restart restarts chatd
# too; tandem restart leaves chatd and its runs alone.
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash $DIR/bin/tandem start
ExecStop=/bin/bash $DIR/bin/tandem stop 120
TimeoutStartSec=2400
TimeoutStopSec=180

[Install]
WantedBy=default.target
EOF
        if systemctl --user daemon-reload && systemctl --user enable tandemllm.service >/dev/null; then
            ok "enabled tandemllm.service (user). To start it at boot without a login: sudo loginctl enable-linger $USER"
        else
            warn "systemctl --user failed (no user session bus?); the unit file is at $UNIT"
        fi
    fi
fi

# ---------------------------------------------------------------------------------------------
# 7. Start, health, smoke test
# ---------------------------------------------------------------------------------------------

step "Start"
if [ "$SETUP" = 1 ]; then
    if [ "$START" != 1 ]; then
        info "skipped (--no-start). Start setup mode with: bash $RUN/ops/start.sh --setup"
    elif [ "$DRY" = 1 ]; then
        note "bash $RUN/ops/start.sh --setup --host $HOST   (the dashboard and Wren; waits for /health)"
    else
        bash "$RUN/ops/start.sh" --setup --host "$HOST" || die "setup mode did not come up; log: $SRC/logs/engine.log"
    fi
    # no model is loaded yet: the user picks one in the Setup Guide (OPS-149)
    printf '\n%sSetup mode%s: no model is loaded yet. Open the Setup Guide in a browser and pick a model there.\n' "$B" "$N"
    printf 'The Chat tab starts with the engine once the setup is done (%s/bin/tandem chat status).\n\n' "$DIR"
    case "$HOST" in 127.0.0.1|0.0.0.0) printf '  On this machine:        http://127.0.0.1:%s/dashboard/#/setup\n' "$PORT" ;; esac
    if [ "$HOST" = 127.0.0.1 ]; then
        printf '  From another computer:  ssh -L %s:127.0.0.1:%s %s@%s\n' "$PORT" "$PORT" \
            "$(id -un 2>/dev/null || echo '<user>')" "$(hostname 2>/dev/null || echo '<this box>')"
        printf '                          then open http://127.0.0.1:%s/dashboard/#/setup on that computer\n' "$PORT"
        printf '  (To open it to your network instead, install with --host 0.0.0.0.)\n'
    else
        SETUP_ADDR="$HOST"; [ "$HOST" = 0.0.0.0 ] && SETUP_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
        printf '  From your network:      http://%s:%s/dashboard/#/setup\n' "${SETUP_ADDR:-<this box>}" "$PORT"
        printf '\n%sWARNING:%s no admin password is set, so anyone who reaches port %s can run the setup.\n' "$Y" "$N" "$PORT"
        printf '  Set one with: bash %s/ops/make-secrets.sh --admin   (then restart setup mode)\n' "$RUN"
    fi
    chat_note
    exit 0
fi
if [ "$START" != 1 ]; then
    info "skipped (--no-start). Start it with: $DIR/bin/tandem start"
elif [ "$DRY" = 1 ]; then
    note "$DIR/bin/tandem start   (ops/start.sh with $RUN/ops/serve.env; waits for /health)"
    note "$DIR/bin/tandem smoke   (/v1/models and one short chat completion, prints tok/s)"
else
    info "loading the model (a few minutes; the first start also compiles kernels)"
    if [ "$SYSTEMD" = 1 ] && systemctl --user is-enabled tandemllm.service >/dev/null 2>&1; then
        systemctl --user start tandemllm.service || die "tandemllm.service did not start: journalctl --user -u tandemllm; $SRC/logs/engine.log"
    else
        bash "$DIR/bin/tandem" start || die "the engine did not come up; log: $SRC/logs/engine.log"
    fi
    bash "$DIR/bin/tandem" smoke || die "the smoke test failed; log: $SRC/logs/engine.log"
fi

API_HOST="$HOST"; [ "$HOST" = 0.0.0.0 ] && API_HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"; API_HOST="${API_HOST:-127.0.0.1}"
cat <<EOF

${B}Done.${N}  TandemLLM $TANDEM_REF in $DIR

  Control:   $DIR/bin/tandem start | stop | restart | status | smoke | logs | doctor | chat
  Config:    $RUN/ops/serve.env   (your overrides: $RUN/local.env)
  Watchdog:  bash $RUN/ops/watchdog.sh   (optional, from cron every minute: docs/operations.md)
  Update:    re-run the installer (same --dir); downloads resume, finished steps are skipped
  Remove:    bash install.sh --dir $DIR --uninstall   (the weights stay in the HF cache)
  Dashboard: http://$API_HOST:$PORT/dashboard (no password: anyone who reaches the port can read
             and change it, the memory included; keep the port on a trusted network)
  Chat:      http://$API_HOST:$PORT/dashboard/#/chat  chatd starts and stops with the engine
             ($DIR/bin/tandem chat status). The sandbox is optional: without a container runtime
             the chat answers and uses the file tools; bash $SRC/ops/sandbox/install.sh adds the rest

  OpenAI-compatible API at http://$API_HOST:$PORT/v1, model "$SERVED_MODEL", no API key needed:

    export OPENAI_BASE_URL=http://$API_HOST:$PORT/v1 OPENAI_API_KEY=none
    curl -s \$OPENAI_BASE_URL/chat/completions -H 'Content-Type: application/json' \\
      -d '{"model": "$SERVED_MODEL", "messages": [{"role": "user", "content": "Say hello."}]}'

  opencode (~/.config/opencode/opencode.json):

    {
      "\$schema": "https://opencode.ai/config.json",
      "provider": {
        "tandem": {
          "npm": "@ai-sdk/openai-compatible",
          "name": "TandemLLM",
          "options": { "baseURL": "http://$API_HOST:$PORT/v1" },
          "models": { "$SERVED_MODEL": { "name": "Qwen3.8-27B (TandemLLM)" } }
        }
      }
    }
EOF
chat_note
