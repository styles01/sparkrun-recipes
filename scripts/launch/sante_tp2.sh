#!/usr/bin/env bash
# Two-node TP=2 serve of the DeepSeek-V4.1-Flash recipe on 2x DGX Spark (GB10), over RoCE v2 on the ConnectX link.
# Modeled on sparkctl's launch: one container per node, vLLM's MP backend across nodes (--nnodes 2,
# the worker headless), NCCL pinned to the RoCE HCA that carries the fabric IPv4 with the GID index derived from that
# IPv4 (auto-detect hands NCCL the port without an IP and dies), --device /dev/infiniband instead of --privileged,
# --restart no (docker restarting one rank of a TP pair yields a head that can never rejoin).
#
#   MODEL_DIR=/home/spark/models/<ours> ./dsv41_tp2.sh up        start both ranks, wait for /v1/models
#   LOAD_FORMAT=dummy MODEL_DIR=... ./dsv41_tp2.sh up          G0 fit / speed probe without weights
#   ./dsv41_tp2.sh status | logs [head|worker] | down
# Knobs (env): IMAGE SERVED PORT MAX_LEN MAX_SEQS MAX_BATCH UTIL KV_BYTES SPEC (JSON or empty = off) GRAPH KERNEL_CONFIG TEXT_ONLY EXTRA HEAD WORKER
#   DOCKER_ENV="VLLM_EXL3_HEAD=mxfp8 ..." (extra container env, same on both ranks)
set -euo pipefail
HEAD=${HEAD:-spark}; WORKER=${WORKER:-spark2}
FAB_IF=${FAB_IF:-enp1s0f0np0}; FAB_HCA=${FAB_HCA:-rocep1s0f0}
IMAGE=${IMAGE:-dsv41-serve:v4}   # the release image: patches 0001-0008 and 0010-0021 + the plugin (README quickstart)
SERVED=${SERVED:-dsv41}; PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-262144}; MAX_SEQS=${MAX_SEQS:-4}; MAX_BATCH=${MAX_BATCH:-2048}; UTIL=${UTIL:-0.85}
# KV pool pinned per rank instead of sized from UTIL: 2.5 GiB, as Mia-AiLab's kit ships (KV_CACHE_MEMORY_BYTES, her README:
# ~774K tokens of fp8_ds_mla per rank), so both serves get the same pool; KV_BYTES= (empty) sizes it from UTIL again
KV_BYTES=${KV_BYTES-4000000000}
SPEC=${SPEC-'{"method":"mtp","num_speculative_tokens":3}'}
GRAPH=${GRAPH:-'{"cudagraph_mode":"FULL_DECODE_ONLY"}'}
# 256 reader threads: with 32, Engram disk reads stalled a 512-token prefill step ~74 ms (dummy-boot profile, Oct 8);
# scripts/bench/engram_reader_bench.py, cold rows: 4,096 rows 80.2 ms at 32 threads, 5.8 ms at 256 (16,384: 318 -> 20.4 ms).
ENGRAM=${ENGRAM-''}
LOAD_FORMAT=${LOAD_FORMAT:-auto}
# FlashInfer autotune off: on GB10 every new kernel shape grows driver-side memory; with autotune on, a dummy-weight
# boot went from 4-5 to 0.8 GiB MemAvailable in about 5 minutes of mixed requests
KERNEL_CONFIG=${KERNEL_CONFIG:-'{"enable_flashinfer_autotune":false}'}
# Text-only by default: an image limit of 0 makes vLLM stub out the V4.1 vision tower and skip its weights (~0.9 GiB
# per rank, which the 262K KV cache needs more). TEXT_ONLY=0 serves images.
TEXT_ONLY=${TEXT_ONLY:-1}
EXTRA=${EXTRA:---quantization mxfp4}
# The measured-best serve settings (BENCHMARKS.md, RECIPE.md): our EXL3 MoE knobs, DSpark per-round depth from the
# confidence head with the measured round-cost table (patch 0010; ms by depth 0-5, drafter included, measured with the
# ring and the prefetch on), Markov bias rows cached (patch 0011; the token list is copied to each node's cache below),
# the paced L2 prefetch and the RDMA all-reduce ring (plugin), and the levers picked since: the local draft argmax
# (0012), smoe run flags (evict-first is compiled in), mHC overlap (0014), the Engram spin reader (0015), short-context
# graph twins (0019), our router with its gate in L2, exl3_mrow's grouped launch, and since v1.1 TensorFold's grouped
# EXL3 prompt kernels for the routed experts (Apache-2.0, plugin csrc/tf_experts/: prefill +18-20%, decode unchanged)
# with the JIT builds kept in the mounted cache (no recompile on every boot). The round-cost table is the one
# measured on the shipping weights (drafter experts on EXL3 K3; scripts/bench/depth_sweep.sh at fixed depths, Oct 12);
# re-measure it after changing weights, image or knobs. RECIPE_ENV= (empty) runs without them; DOCKER_ENV comes after
# it, so a knob there overrides the same knob here.
RECIPE_ENV=${RECIPE_ENV-""}
DOCKER_ENV=${DOCKER_ENV:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
NAME=sante-serve
cmd=${1:-status}

sh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }
fab_ip() { sh_ "$1" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1"; }
gid_of() {   # RoCE v2 GID index whose IPv4-mapped GID is this node's fabric address
  sh_ "$1" "want=\$(ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1); P=/sys/class/infiniband/$FAB_HCA/ports/1
    hex=\$(printf '%02x%02x%02x%02x' \$(echo \$want | tr '.' ' '))
    for i in \$(seq 0 15); do t=\$(cat \$P/gid_attrs/types/\$i 2>/dev/null); g=\$(cat \$P/gids/\$i 2>/dev/null)
      case \"\$t\" in *v2*) ;; *) continue;; esac
      case \"\$g\" in *ffff:\${hex:0:4}:\${hex:4:4}*) echo \$i; break;; esac; done"
}

launch() {   # $1 node  $2 rank  $3 master ip  $4 self ip  $5 gid
  local n=$1 rank=$2 master=$3 self=$4 gid=$5 tmp envs="" kv
  for kv in $RECIPE_ENV $DOCKER_ENV; do envs="$envs -e $kv"; done
  tmp=$(mktemp)
  {
    echo '#!/usr/bin/env bash'
    echo 'set -e'
    echo "exec vllm serve '$MODEL_DIR' --served-model-name $SERVED --host 0.0.0.0 --port $PORT \\"
    echo "  --tensor-parallel-size 2 --nnodes 2 --node-rank $rank --master-addr $master --master-port 25000 \\"
    echo "  --gpu-memory-utilization $UTIL --max-model-len $MAX_LEN --max-num-seqs $MAX_SEQS \\"
    echo "  --max-num-batched-tokens $MAX_BATCH --load-format $LOAD_FORMAT \\"
    echo "--compilation-config '$GRAPH' --kernel-config '$KERNEL_CONFIG' \\"
    [ -n "$ENGRAM" ] &&     echo "  --engram-config '$ENGRAM' \\"
    [ -n "$SPEC" ] && echo "  --speculative-config '$SPEC' \\"
    [ -n "$KV_BYTES" ] && echo "  --kv-cache-memory-bytes $KV_BYTES \\"
    [ "$TEXT_ONLY" = 1 ] && echo "  --limit-mm-per-prompt '{\"image\":0}' \\"
    [ "$rank" != 0 ] && echo "  --headless \\"
    echo "  $EXTRA"
  } > "$tmp"
  scp -q -o BatchMode=yes "$tmp" "$n:/tmp/$NAME.sh"
  rm -f "$tmp"
  sh_ "$n" "chmod +x /tmp/$NAME.sh; docker rm -f $NAME >/dev/null 2>&1; mkdir -p \$HOME/.cache/sante-serve
    docker run -d --name $NAME --network host --ipc host --shm-size 64g --gpus all --device /dev/infiniband \
      --ulimit memlock=-1 --ulimit stack=67108864 --restart no --entrypoint /bin/bash \
      -v '$MODEL_DIR':'$MODEL_DIR':ro -v /tmp/$NAME.sh:/serve.sh:ro -v \$HOME/.cache/sante-serve:/root/.cache \
      -e NCCL_IB_HCA=$FAB_HCA -e NCCL_IB_GID_INDEX=$gid -e NCCL_IB_DISABLE=0 -e NCCL_NET=IB \
      -e NCCL_SOCKET_IFNAME=$FAB_IF -e GLOO_SOCKET_IFNAME=$FAB_IF -e TP_SOCKET_IFNAME=$FAB_IF \
      -e VLLM_HOST_IP=$self -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
      -e VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-FlashInferCutlassMxfp8LinearKernel} \
      -e HF_HUB_OFFLINE=1 $envs $IMAGE /serve.sh >/dev/null && echo '$n rank $rank started'"
}

case "$cmd" in
  up)
    # operator hold: a chain stage waits here while any ~/.dsv41_serve_hold* exists (lever A/Bs slot in between stages
    # without editing the running chain); the A/B scripts pass SKIP_SERVE_HOLD=1. TERM is ignored while holding: chain
    # stages wrap this in `timeout 1500`, a hold outlives that, and a killed boot would send the chain on to its next
    # `down`, which stops whatever lever serve is up. Released after the timer fired, that boot still
    # returns timeout's 124 (a pipefail chain skips what follows it), and kill -9 is what stops a held instance.
    if compgen -G "$HOME/.dsv41_serve_hold*" >/dev/null && [ -z "${SKIP_SERVE_HOLD:-}" ]; then
      trap '' TERM
      while compgen -G "$HOME/.dsv41_serve_hold*" >/dev/null; do sleep 30; done
      trap - TERM
    fi
    : "${MODEL_DIR:?set MODEL_DIR (same path on both nodes)}"
    hip=$(fab_ip "$HEAD"); wip=$(fab_ip "$WORKER")
    hg=$(gid_of "$HEAD"); wg=$(gid_of "$WORKER")
    [ -n "$hip" ] && [ -n "$wip" ] && [ -n "$hg" ] && [ -n "$wg" ] || { echo "fabric not ready: $hip/$wip gid $hg/$wg"; exit 1; }
    echo "head $HEAD $hip gid $hg | worker $WORKER $wip gid $wg | image $IMAGE | model $MODEL_DIR | load $LOAD_FORMAT"
    for n in "$HEAD" "$WORKER"; do sh_ "$n" "test -d '$MODEL_DIR'" || { echo "$MODEL_DIR missing on $n"; exit 1; }; done
    for n in "$HEAD" "$WORKER"; do sh_ "$n" "docker image inspect $IMAGE >/dev/null 2>&1" \
      || { echo "$IMAGE missing on $n: build it (README quickstart, step 2) or docker save | ssh $n docker load"; exit 1; }; done
    # patch 0011's token list (most frequent first) into each node's cache, which the containers mount at /root/.cache
    case "$RECIPE_ENV $DOCKER_ENV" in *VLLM_DSPARK_MARKOV_CACHE=*)
      for n in "$HEAD" "$WORKER"; do
        sh_ "$n" "mkdir -p \$HOME/.cache/sante-serve" && scp -q "$REPO/data/dspark/markov_top_tokens.json" "$n:.cache/sante-serve/"
      done ;;
    esac
    # GB10: CUDA allocations do not reclaim page cache; vLLM would size its pools to "free", not "available"
    for n in "$HEAD" "$WORKER"; do sh_ "$n" "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || echo 'drop_caches needs sudo -n on $n'"; done
    launch "$WORKER" 1 "$hip" "$wip" "$wg"
    launch "$HEAD" 0 "$hip" "$hip" "$hg"
    t0=$(date +%s)
    until sh_ "$HEAD" "curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1"; do
      for n in "$HEAD" "$WORKER"; do
        st=$(sh_ "$n" "docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null" || echo false)
        [ "$st" = true ] || { echo "$n: container stopped"; sh_ "$n" "docker logs --tail 40 $NAME 2>&1"; exit 1; }
      done
      sleep 15
    done
    echo "ready after $(( $(date +%s) - t0 )) s"
    for n in "$HEAD" "$WORKER"; do echo "$n: $(sh_ "$n" "free -g | awk '/Mem:/{print \"used\", \$3, \"GiB, available\", \$7, \"GiB\"}'")"; done ;;
  down)
    for n in "$HEAD" "$WORKER"; do sh_ "$n" "docker rm -f $NAME >/dev/null 2>&1; echo '$n down'"; done ;;
  logs)
    n=$([ "${2:-head}" = worker ] && echo "$WORKER" || echo "$HEAD"); sh_ "$n" "docker logs --tail ${3:-60} $NAME 2>&1" ;;
  status)
    for n in "$HEAD" "$WORKER"; do echo "$n: $(sh_ "$n" "docker ps --filter name=$NAME --format '{{.Status}}'; free -g | awk '/Mem:/{print \"avail\", \$7, \"GiB\"}'" | tr '\n' ' ')"; done
    sh_ "$HEAD" "curl -s http://localhost:$PORT/v1/models | head -c 300"; echo ;;
  *) echo "usage: $0 up|down|logs [head|worker]|status"; exit 2 ;;
esac
