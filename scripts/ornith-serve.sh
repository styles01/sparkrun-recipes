#!/bin/bash
# ornith-serve.sh — STANDING lane launcher for Ornith-1.5-122B (registry-canonical).
# Source of truth: ~/sparkrun-recipes/scripts/ornith-serve.sh (Mac) -> spark1:/home/jaita/serve-ornith.sh
# Serves: full HF id + 'ornith' short alias. No MTP (k-sweep measured net-negative 10/21).
# Drain-gate: refuses if anything holds :8000. TRASH-ONLY. Logs /tmp/ornith-serve.log
set -u
NAME="ornith-serve"
IMG="vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
NAMES="Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw ornith"
LOG="/tmp/ornith-serve.log"

# drain gate: refuse if :8000 occupied
if curl -s --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "DRAIN-GATE: :8000 occupied — refusing. Aborting."; exit 2
fi

SETUP='export CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include:/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/include:$CPATH && pip install --no-build-isolation "vllm-exl3 @ https://github.com/Zeuss5/cuda-exl3/archive/refs/tags/v1.0.0.tar.gz" && python3 -c "import vllm_exl3, torch; assert torch.cuda.get_device_capability()[0] >= 12; print(\"vllm_exl3 import ok:\", vllm_exl3.__file__)"'

REPL="vllm serve Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw --served-model-name ${NAMES} --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --quantization exl3 --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.9"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --gpus all --network host --shm-size 16gb --user root \
  -v /home/jaita/.cache/sparkrun/runtime-cache/vllm/Mia-AiLab__Ornith-1.5-122B-EXL3-EAQv1-4-bpw-596d85ab:/cache/runtime \
  -v /home/jaita/.cache/huggingface:/cache/huggingface \
  -e TORCH_CUDA_ARCH_LIST=12.1 \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_CACHE=/cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e XDG_CACHE_HOME=/cache/runtime \
  -e CUDA_EXL3_TUNE_CACHE=/cache/runtime/cuda_exl3_tune \
  -e CUDA_EXL3_DEBUG_NAMES=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  --entrypoint bash \
  "$IMG" -c "$SETUP && echo ===SETUP-DONE=== && $REPL" > "$LOG" 2>&1

echo "ornith-serve container launched; blind window = plugin build (~86s) + 62GB load + JIT (5-10 min)"