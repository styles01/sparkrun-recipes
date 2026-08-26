# Qwen3.8-Flash-Next Q4_K_XL llama-server image (qwen4exp fork) for DGX Spark GB10.
# Public serving image for the styles01 spark-arena first-post llama-cpp benchmark.
# Bundles the working build + the 0xBakeer "180B-fits-in-119GB" trick:
#   -ot per_layer_token_embd=CPU  pin 51B n-gram tensor to CPU
#   -lm mmap                      serve it from NVMe via mmap
#
# Base: nvcr.io/nvidia/vllm:26.06-py3 (aarch64, CUDA 13.0 — has libcublas.so.13 /
# libcudart.so.13 required by libggml-cuda built against CUDA 13).
FROM nvcr.io/nvidia/vllm:26.06-py3

# The llama.cpp qwen4exp fork build (PR #27742, commit 035e227) — build/bin/*.
# We overlay the whole build/bin so llama-server finds its sibling .so libs via
# LD_LIBRARY_PATH set by the launch script.
COPY build/ /opt/qwen4exp/bin/
COPY launch_qwen38_flash_next.sh /opt/qwen4exp/launch.sh
RUN chmod +x /opt/qwen4exp/launch.sh

# Python bits for auto-download (huggingface_hub) and bash.
RUN apt-get update && apt-get install -y --no-install-recommends bash curl ca-certificates \
    && pip install -q -U huggingface_hub \
    && rm -rf /var/lib/apt/lists/*

# Model is bind-mounted by sparkrun at /models; if absent the entrypoint downloads.
VOLUME ["/models"]

# sparkrun expects an "absent" entrypoint probe to pass and then execs the server;
# mirror the proven vLLM-patch image pattern (CMD sleep infinity + exec).
ENTRYPOINT ["/opt/qwen4exp/launch_qwen38_flash_next.sh"]
CMD []
