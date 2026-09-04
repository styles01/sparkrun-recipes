#!/usr/bin/env bash
# Shared GB10 Docker CUDA preflight.
#
# A cold Spark boot can leave ordinary Docker GPU containers unable to create a
# CUDA context even though `nvidia-smi` works. On the affected boot path,
# cuInit returns CUDA_ERROR_UNKNOWN (999) unless the dedicated inference
# container receives the host's required privileged GPU access.
#
# Usage: require_spark_docker_cuda <image>
# Call before starting a direct Docker GPU inference container and launch that
# container with `--privileged --gpus all`.

require_spark_docker_cuda() {
  local image="$1"
  local probe='import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0; torch.zeros(1, device="cuda")'

  echo "[gpu-preflight] Verifying a real CUDA context in ${image}..."
  if ! docker run --rm --privileged --gpus all --entrypoint python3 "$image" \
      -c "$probe" >/tmp/spark-docker-cuda-preflight.log 2>&1; then
    echo "[gpu-preflight] ERROR: CUDA context is unavailable inside Docker."
    cat /tmp/spark-docker-cuda-preflight.log
    return 1
  fi
  echo "[gpu-preflight] CUDA context ready."
}
