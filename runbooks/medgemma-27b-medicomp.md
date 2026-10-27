# MedGemma 27B FP8 — Medicomp Spark (VPN) vLLM Recipe

**Target:** `<spark-host>` on Medicomp VPN (`<spark-ip>`)
**Model:** `SaitBurak/medgemma-27b-text-it-FP8-dynamic` (27GB, FP8 compressed-tensors)
**Served name:** `medgemma-27b`
**No SparkDash — inference only**

> **Recipe contract:** [`recipes/medgemma-27b-medicomp.yaml`](../recipes/medgemma-27b-medicomp.yaml)

## Config

| Parameter | Value |
|---|---|
| Context | 3,800 |
| Max seqs | 75 |
| GMU | 0.83 |
| KV cache dtype | fp8 |
| Attention backend | flashinfer |
| Prefix caching | ON |

## SSH Commands

### 1. Pull model (if not already present)

```bash
ssh user@<spark-ip> 'huggingface-cli download SaitBurak/medgemma-27b-text-it-FP8-dynamic --local-dir ~/models/medgemma-27b-fp8'
```

### 2. Kill any existing inference

```bash
ssh user@<spark-ip> 'docker rm -f medgemma-spark 2>/dev/null; ps aux | grep "vllm serve" | grep -v grep | awk "{print \$2}" | xargs kill -9 2>/dev/null; echo "cleared"'
```

### 3. Launch vLLM

```bash
ssh user@<spark-ip> 'docker run -d \
  --name medgemma-spark \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/models/medgemma-27b-fp8:/model:ro \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm/vllm-openai:latest \
  vllm serve /model \
    --served-model-name medgemma-27b \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 3800 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.83 \
    --max-num-seqs 75 \
    --attention-backend flashinfer \
    --enable-prefix-caching'
```

### 4. Wait for health check

```bash
ssh user@<spark-ip> 'for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "READY"
    docker logs medgemma-spark 2>&1 | grep -E "Model loading took|Available KV cache|GPU KV cache size|Maximum concurrency"
    curl -s http://127.0.0.1:8000/v1/models | python3 -c "import sys,json; print(\"Model:\", json.load(sys.stdin)[\"data\"][0][\"id\"])"
    exit 0
  fi
  if ! docker ps --format "{{.Names}}" | grep -q "^medgemma-spark$"; then
    echo "CRASHED"
    docker logs medgemma-spark 2>&1 | tail -20
    exit 1
  fi
  sleep 5
done
echo "TIMEOUT"'
```

### 5. Verify

```bash
ssh user@<spark-ip> 'curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"medgemma-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"List three differential diagnoses for fatigue.\"}],\"max_tokens\":200}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)[\"choices\"][0][\"message\"][\"content\"])"'
```

### 6. Stop

```bash
ssh user@<spark-ip> 'docker rm -f medgemma-spark'
```

## Expected Performance

| Metric | Value |
|---|---|
| Model load | ~27 GiB, ~3 min |
| KV cache | ~71 GB → ~575K tokens |
| Max concurrency | 151× at 3.8K (only need 75) |
| Free RAM | ~21 GB |
| Aggregate tok/s | ~336+ (scales with batch) |
| Per-request tok/s | ~7 (memory-bandwidth-bound) |
| GPU util | 96% at 37W — bandwidth-bound |

## Notes

- **FlashInfer is required** — default Triton backend degrades at batch ≥12 (vLLM issue #48076)
- **FP8 is baked into the checkpoint** — no `--quantization` flag needed
- **Prefix caching** — 88% hit rate with shared system prompt
- **No SparkDash** — inference only on this box
- If OOM: drop GMU to 0.75, then drop max-num-seqs to 48
- The `vllm/vllm-openai:latest` image works; if it OOMs during graph capture, try `nvcr.io/nvidia/vllm:26.06-py3` (auto-selects CUTLASS backend, smaller CUDA graph footprint)