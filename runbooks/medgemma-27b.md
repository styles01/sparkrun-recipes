# Recipe: MedGemma 27B FP8

**Status:** ✅ PROD — AI-Scribe corpus formatting
**Served name:** `medgemma`
**Stack:** Docker — `vllm-node:latest`
**Model path:** `~/models/medgemma-27b-fp8` (SaitBurak/medgemma-27b-text-it-FP8-dynamic, 27GB)
**Source:** AI-Scribe corpus formatter (docs/CORPUS.md, ADRs 0004/0005/0008)
**Tested:** 4,961/4,963 MTSamples notes completed on <spark-host>

> **Recipe contract:** [`recipes/medgemma-27b.yaml`](../recipes/medgemma-27b.yaml)

## Container

| Key | Value |
|---|---|
| Image | `vllm-node:latest` |
| Name | `medgemma-spark` |
| Port | `8000` |
| GPU | all |
| Restart | `unless-stopped` |
| Volume | `$HOME/models/medgemma-27b-fp8` → `/model:ro` |
| Volume | `$HOME/.cache/huggingface` → `/root/.cache/huggingface` |

### Environment

```bash
HF_HOME=/root/.cache/huggingface
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## vLLM Serve Command (optimized — 256 seqs)

```bash
vllm serve /model \
  --served-model-name medgemma \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.88 \
  --max-num-seqs 256 \
  --attention-backend flashinfer \
  --enable-prefix-caching
```

## Flag Rationale

| Flag | Value | Why |
|---|---|---|
| `--max-num-seqs` | **256** | Biggest throughput lever — lets vLLM batch more decode streams |
| `--attention-backend` | **flashinfer** | Required — Triton degrades at batch ≥12 (ADR-0008) |
| `--kv-cache-dtype` | **fp8** | Halves KV memory → more concurrent seqs |
| `--gpu-memory-utilization` | **0.88** | Leaves room for sparkDash co-location (~3GB free) |
| `--max-model-len` | **8192** | Small KV per seq = more concurrency. Don't raise. |
| `--enable-prefix-caching` | ON | 88% hit rate — fixed system prompt shared across all requests |
| FP8 quant | Baked into checkpoint | No `--quantization` flag needed (compressed-tensors) |

## Performance

| Metric | Value |
|---|---|
| Model load | 26.79 GiB, ~177s |
| KV cache | 76.93 GiB → 618,265 tokens |
| Max concurrency | 75.47× at 8K |
| Aggregate tok/s | ~336 (32 seqs), scales with batch |
| Per-request tok/s | ~7 (memory-bandwidth-bound at 27GB FP8) |
| GPU util | 96% at 37W — bandwidth-bound, not compute-bound |
| Prefix cache hit | 88% |

## OOM Fallbacks

If OOM at boot:
1. Drop `--gpu-memory-utilization` to `0.85`
2. Drop `--max-num-seqs` to `128`
3. Drop `--max-model-len` to `4096`

## SparkDash Co-location

MedGemma at 0.88 GMU leaves ~3GB free for sparkDash. Launch sparkDash after medgemma:

```bash
docker run -d --name sparkDash --network host --privileged --restart unless-stopped \
  -v ~/sparkDash/server:/app/server \
  -v ~/sparkDash/dist:/app/dist:ro \
  -v /proc:/host/proc:ro \
  -v /sys:/host/sys:ro \
  -v /:/host/root:ro \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  -v /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro \
  -v ~/sparkDash/config:/app/config \
  -e PORT=5555 -e LLM_PORT=8000 -e NODE_ENV=production \
  -e HOST_PROC_PATH=/host/proc -e HOST_SYS_PATH=/host/sys -e HOST_ROOT_PATH=/host/root \
  sparkdash-sparkdash node --watch server/index.js
```

**Critical:** sparkDash must run with `--network host --privileged` for:
- LLM probe to reach `localhost:8000` (not bridge network)
- nsenter to access host nvidia-smi for GPU metrics
- VRAM fallback: GB10 nvidia-smi returns `[N/A]` for memory — SystemCollector.js falls back to `MemTotal - MemAvailable` from `/proc/meminfo`

**sparks.json** must have `llmPort: 8000` (singular, not `llmPorts` array).

## Verify

```bash
curl -s http://<spark-host>:8000/v1/models | jq '.data[].id'
# expect: medgemma

curl -s http://<spark-host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"medgemma","messages":[{"role":"user","content":"List three differential diagnoses for fatigue."}],"max_tokens":200}' \
  | jq '.choices[0].message.content'
```

## Sources

- AI-Scribe repo: `docs/CORPUS.md` § Operations
- ADR-0004: vLLM + MedGemma over Ollama
- ADR-0005: Grammar-constrained JSON via response_format
- ADR-0008: FlashInfer attention backend (Triton cliff at batch ≥12)
- Our fork: https://github.com/styles01/sparkDash