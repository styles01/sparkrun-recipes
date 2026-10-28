#!/usr/bin/env bash
# Sante FP4 hybrid pack — TP1 single-node serve (spark1). Fork of sante_tp2.sh shape:
# same container contract (host net, IPC, shm 64g, infiniband device, memlock), TP=1,
# no nnodes/master; pack declares its quantization in config.json (no --quantization flag).
#   ./sante_fp4_tp1.sh up|down|status|logs
set -euo pipefail
HEAD=${HEAD:-10.200.0.1}
MODEL_DIR=${MODEL_DIR:-/home/jaita/models/ling30-sante-fp4-pack}
IMAGE=${IMAGE:-dsv41-serve:v032}
SERVED=${SERVED:-"sante Ling-3.0-flash-Sante"}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-262144}; MAX_SEQS=${MAX_SEQS:-3}; MAX_BATCH=${MAX_BATCH:-2048}; UTIL=${UTIL:-0.85}
KV_BYTES=${KV_BYTES:-6500000000}
SPEC=${SPEC:-'{"method":"mtp","num_speculative_tokens":3}'}
GRAPH=${GRAPH:-'{"cudagraph_mode":"FULL_DECODE_ONLY"}'}
KERNEL_CONFIG=${KERNEL_CONFIG:-'{"enable_flashinfer_autotune":false}'}
EXTRA=${EXTRA:---trust-remote-code --enable-auto-tool-choice --tool-call-parser ling3 --reasoning-parser ling3}
NAME=sante-fp4-tp1
CACHEDIR=$HOME/.cache/sante-serve
cmd=${1:-status}

run_node() {
  ssh -o BatchMode=yes "$HEAD" "chmod +x /tmp/$NAME.sh; docker rm -f $NAME >/dev/null 2>&1; mkdir -p $CACHEDIR
    docker run -d --name $NAME --network host --ipc host --shm-size 64g --gpus all --device /dev/infiniband \
      --ulimit memlock=-1 --ulimit stack=67108864 --restart no --entrypoint /bin/bash \
      -v $MODEL_DIR:$MODEL_DIR:ro -v /tmp/$NAME.sh:/serve.sh:ro -v $CACHEDIR:/root/.cache \
      -e HF_HUB_OFFLINE=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
      $IMAGE /serve.sh >/dev/null && echo 'rank started'"
}

case "$cmd" in
  up)
    [ -d "$MODEL_DIR" ] || { echo "$MODEL_DIR missing on head"; exit 1; }
    SHARDS=$(ssh -o BatchMode=yes "$HEAD" "find $MODEL_DIR -name 'model-*.safetensors' | wc -l; du -sb $MODEL_DIR | cut -f1")
    echo "pack check: $SHARDS"
    tmp=$(mktemp)
    {
      echo '#!/usr/bin/env bash'; echo 'set -e'
      echo "exec vllm serve '$MODEL_DIR' --served-model-name $SERVED --host 0.0.0.0 --port $PORT \\"
      echo "  --tensor-parallel-size 1 --gpu-memory-utilization $UTIL --max-model-len $MAX_LEN --max-num-seqs $MAX_SEQS \\"
      echo "  --max-num-batched-tokens $MAX_BATCH --load-format auto \\"
      echo "--compilation-config '$GRAPH' --kernel-config '$KERNEL_CONFIG' \\"
      echo "  --speculative-config '$SPEC' \\"
      echo "  --kv-cache-memory-bytes $KV_BYTES \\"
      echo "  $EXTRA"
    } > "$tmp"
    scp -q "$tmp" "$HEAD:/tmp/$NAME.sh"; rm -f "$tmp"
    run_node
    t0=$(date +%s)
    until ssh -o BatchMode=yes "$HEAD" "curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1"; do
      st=$(ssh -o BatchMode=yes "$HEAD" "docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null" || echo false)
      [ "$st" = true ] || { echo "container stopped"; ssh "$HEAD" "docker logs --tail 40 $NAME 2>&1"; exit 1; }
      sleep 15
    done
    echo "ready after $(( $(date +%s) - t0 )) s"
    ssh -o BatchMode=yes "$HEAD" "free -g | awk '/Mem:/{print \"avail\", \$7, \"GiB\"}'" ;;
  down)
    ssh -o BatchMode=yes "$HEAD" "docker rm -f $NAME >/dev/null 2>&1; echo down" ;;
  logs)
    ssh -o BatchMode=yes "$HEAD" "docker logs --tail \${3:-60} $NAME 2>&1" ;;
  status)
    ssh -o BatchMode=yes "$HEAD" "docker ps --filter name=$NAME --format '{{.Status}}'; free -g | awk '/Mem:/{print \"avail\", \$7, \"GiB\"}'; curl -s http://localhost:$PORT/v1/models | head -c 300; echo" ;;
  *) echo "usage: $0 up|down|logs|status"; exit 2 ;;
esac