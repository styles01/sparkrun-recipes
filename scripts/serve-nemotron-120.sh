#!/bin/bash
# serve-nemotron-120.sh — Nemotron 3.5 Super VL 120B-A12B NVFP4 test lane, spark1 :8000, name Nemotron-3.5-120
# Same official config as the arena recipe; alias name for loca's test.
IMAGE=vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989   # b12x-complete nightly (FP6 warning persists on 3f1bcfd → Marlin fallback)
R=/home/jaita/.cache/sparkrun/runtime-cache/vllm/nvidia__NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-N-9183bfa1
set -e
docker rm -f nemotron-120 2>/dev/null || true
# ── DRAIN GATE (L8, James 2026-10-16: loca+viktor were mid-work during a restart) ──
# Refuse to take the lane down while ANY request is in flight or queued.
INFLIGHT=$(curl -s -m 4 http://127.0.0.1:8000/metrics 2>/dev/null | grep -E "^vllm:num_requests_(running|waiting)" | awk '{s+=$2} END {print int(s)}')
if [ "${FORCE_RESTART:-0}" != "1" ] && [ "${INFLIGHT:-0}" -gt 0 ]; then
  echo "!! LANE BUSY: ${INFLIGHT} request(s) in flight — refusing restart (set FORCE_RESTART=1 to override, and ONLY after announcing to consumers)"
  exit 1
fi
echo "drain-gate clear (inflight=${INFLIGHT:-0})"
docker run -d --name nemotron-120 --network=host --ipc=host --shm-size 16g --gpus all \
  -v /home/jaita/.cache/huggingface:/cache/huggingface \
  -v /home/jaita/vllm-kernel-cache:/root/.cache \
  -v $R:/cache/runtime \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/cache/huggingface -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_DISABLED_KERNELS="FlashInferCutlassMxfp8LinearKernel,FlashInferFP8ScaledMMLinearKernel" \
  --entrypoint /bin/bash \
  $IMAGE -c "
echo '== installing B12X backend (native NVFP4 kernels; without it Marlin fallback runs) =='
pip install -q b12x && pip install -q 'vllm[b12x]' && python3 -c 'import b12x; print(\"B12X-IMPORT-OK\", getattr(b12x, \"__version__\", \"?\"))'
curl -fsSL https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/resolve/main/ultra_v3_reasoning_parser.py -o /cache/huggingface/ultra_v3_reasoning_parser.py && echo PARSER-CACHED
vllm serve /cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/2e5cc3953bb1ea66b2fb7c831fee3aa317427e69 \
  --served-model-name Nemotron-3.5-120 \
  --trust-remote-code --load-format auto --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.86 --max-model-len 275000 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --kv-cache-dtype fp8 \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr \
  --speculative_config.num_speculative_tokens 7 \
  --speculative_config.method dflash \
  --block-size 64 --prefix-match-unit 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin /cache/huggingface/ultra_v3_reasoning_parser.py \
  --reasoning-parser ultra_v3" > /tmp/nemotron-120-docker.log 2>&1
echo launched