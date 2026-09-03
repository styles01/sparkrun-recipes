# Qwen3.8-Flash-Next Q4_K_XL llama-server image (qwen4exp fork) for DGX Spark GB10.
# Public serving image for the styles01 spark-arena first-post llama-cpp benchmark.
# Bundles the working build + the 0xBakeer "180B-fits-in-119GB" trick:
#   -ot per_layer_token_embd=CPU  pin 51B n-gram tensor to CPU
#   -lm mmap                      serve it from NVMe via mmap
# Plus the 0xBakeer NVMe trick: -ot per_layer_token_embd=CPU + -lm mmap.
# Spec decode: stock ngram-mod (default n_max=3).
#
# Base: nvcr.io/nvidia/vllm:26.06-py3 (aarch64, CUDA 13.0 — has libcublas.so.13 /
# libcudart.so.13 required by libggml-cuda built against CUDA 13).
FROM nvcr.io/nvidia/vllm:26.06-py3

# Build input must come from PR #27941 merge 36b10154383b60eb15baac2c7a40d2a5f784faa7
# or an auditable descendant. Verify the source checkout before copying build/:
# scripts/verify-qwen38-flash-next-build.sh --source-dir <llama.cpp-checkout>
# We overlay build/bin so llama-server finds sibling .so libs via LD_LIBRARY_PATH.
COPY build/ /opt/qwen4exp/bin/
COPY launch.sh /opt/qwen4exp/launch.sh
RUN chmod +x /opt/qwen4exp/launch.sh \
    && ln -sf /opt/qwen4exp/bin/llama-server /usr/local/bin/llama-server

# Python bits for auto-download (huggingface_hub) and bash.
RUN apt-get update && apt-get install -y --no-install-recommends bash curl ca-certificates \
    && pip install -q -U huggingface_hub \
    && rm -rf /var/lib/apt/lists/*

# Link this package to the public styles01/sparkrun-recipes repo so the GHCR
# package inherits the repository's PUBLIC visibility (REST has no visibility
# mutation endpoint; repo-linking via this label is the supported mechanism).
LABEL org.opencontainers.image.source="https://github.com/styles01/sparkrun-recipes"
LABEL org.opencontainers.image.title="qwen38-flash-next"
LABEL org.opencontainers.image.description="Qwen3.8-Flash-Next Q4_K_XL llama-server (qwen4exp fork) on DGX Spark GB10"

# Model is bind-mounted by sparkrun at /models; if absent the entrypoint downloads.
VOLUME ["/models"]

ENTRYPOINT ["/opt/qwen4exp/launch.sh"]
CMD []
