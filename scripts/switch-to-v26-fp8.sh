#!/bin/bash
# switch-to-v26-fp8.sh — Switch Qwen 122B to vLLM v26 with fp8 KV cache
# Usage: bash switch-to-v26-fp8.sh
#
# What this does:
#   1. Kills any existing Qwen container
#   2. Starts vLLM v26 (built from main) with fp8 KV + DFlash n=7
#   3. Result: 1.43M KV tokens, 5.46× concurrency at 256K
#
# Prerequisites:
#   - vllm-v26-patched:latest image must exist on the Spark
#   - Models cached in ~/.cache/huggingface
#
# Rollback: bash switch-to-v26-fp8.sh --rollback

SPARK="user@<spark-host>"
IMAGE="vllm-v26-patched:latest"
MODEL="bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid"
DRAFTER="z-lab/Qwen3.5-122B-A10B-DFlash"

if [ "$1" = "--rollback" ]; then
    echo "Rolling back to aeon 0.23 (bf16 KV)..."
    ssh $SPARK 'docker rm -f qwen-spark 2>/dev/null; cd ~/qwen3.5-122B-A10B-on-spark && CTX=262144 GPU_MEM=0.85 MAX_NUM_SEQS=3 MAX_BATCHED_TOKENS=8192 SERVED_NAME=qwen bash install.sh --start --profile dense --nspec 7 --no-smoke'
    echo "Rolled back to aeon 0.23. Waiting for startup..."
    sleep 60
    ssh $SPARK 'curl -s http://127.0.0.1:8000/v1/models | python3 -c "import sys,json; print(\"Serving:\", json.load(sys.stdin)[\"data\"][0][\"id\"])" 2>&1'
    exit 0
fi

echo "Switching to vLLM v26 + fp8 KV + DFlash n=7..."
echo "  Image: $IMAGE"
echo "  Model: $MODEL"
echo "  KV cache: fp8"
echo "  DFlash: n=7"
echo "  Expected: 1.43M KV tokens, 5.46x concurrency"
echo ""

# Kill existing container
echo "Stopping existing Qwen container..."
ssh $SPARK 'docker rm -f qwen-spark vllm-v26-test 2>/dev/null && echo "killed" || echo "none running"'

# Start v26. A real CUDA allocation catches the GB10 cold-boot Docker policy
# before we attempt a large model load; nvidia-smi alone is not sufficient.
echo "Checking CUDA context inside the v26 image..."
ssh $SPARK "docker run --rm --privileged --gpus all --entrypoint python3 $IMAGE -c 'import torch; assert torch.cuda.is_available(); torch.zeros(1, device=\"cuda\")'"

echo "Starting vLLM v26..."
ssh $SPARK "docker run -d \
  --name qwen-spark --privileged --gpus all -p 8000:8000 --user root \
  -v \$HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $IMAGE \
  $MODEL \
    --served-model-name qwen-122b --host 0.0.0.0 --port 8000 \
    --trust-remote-code --max-model-len 262144 --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85 --max-num-seqs 3 --max-num-batched-tokens 8192 \
    --enable-prefix-caching --enable-chunked-prefill \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --attention-backend FLASHINFER \
    --speculative-config '{\"method\":\"dflash\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":7}'"

echo ""
echo "Container started. Model load takes ~7-10 min."
echo "Monitor: ssh $SPARK 'docker logs -f qwen-spark'"
echo ""
echo "Wait for startup, then check:"
echo "  ssh $SPARK 'curl -s http://127.0.0.1:8000/v1/models'"
echo "  ssh $SPARK 'docker logs qwen-spark 2>&1 | grep -E \"KV cache size|concurrency|startup\"'"
echo ""
echo "Rollback: bash switch-to-v26-fp8.sh --rollback"