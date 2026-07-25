# Step 3.7 Flash — DGX Spark Deployment Handoff

## Goal
Run StepFun Step 3.7 Flash (198B MoE, 11B active) on DGX Spark via llama.cpp with 165K context.

## What's Done
1. **llama.cpp fork built** — `~/llama.cpp-step37/` on Spark, branch `step3.7`, compiled with CUDA 13.0 (`CUDACXX=/usr/local/cuda-13.0/bin/nvcc`). Binary: `~/llama.cpp-step37/build-cuda/bin/llama-server`
2. **IQ4_XS GGUF downloaded** — `~/models/step3.7/IQ4_XS/` (3 shards, ~99GB total)
3. **MTP draft model downloaded** — `~/models/step3.7/Step3.7-flash-mtp-Q8_0.gguf` (3.5GB)
4. **Vision projector NOT downloaded** — `mmproj-step3.7-flash-f16.gguf` if needed later

## Current Blocker
Server hangs at "fitting params to device memory" when loading with `-ngl 999`. The auto-fitting algorithm struggles with 99GB model on 121GB unified memory. Last attempt used `-fit off` but SSH timed out before seeing if it loaded.

## What to Try Next
1. **Check if the `-fit off` attempt succeeded** — `ssh jaita@larryspark.local 'tail -30 /tmp/step37_nofit.log && curl -s http://localhost:8080/health'`
2. **If still stuck, try fewer GPU layers** — `-ngl 250` instead of 999 (model has 288 layers, some may not fit)
3. **Try smaller context first** — `-c 32768` to verify it loads, then scale up to 165K
4. **MTP draft model** — The first attempt with `--model-draft` failed with `missing tensor 'blk.0.attn_norm.weight'` even though the tensor exists in the main model. The draft model may need different handling. Try loading WITHOUT draft model first, then add it.
5. **Q4_K_S alternative** — If IQ4_XS won't load, Q4_K_S (112GB) is the README's default quant. Tighter fit but the examples all use it.

## Working Launch Command (from README)
```bash
cd ~/llama.cpp-step37
./build-cuda/bin/llama-server \
  -m ~/models/step3.7/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf \
  -c 32768 -ngl 99 -fa on \
  --host 0.0.0.0 --port 8080
```
Note: README uses `-ngl 99` not 999, and `-fa on` (flash attention).

## Research File
`/Users/clawdio/step37_research_report.md` — full research with benchmarks, issue links, quant fit analysis.

## Spark Access
- Host: `larryspark.local` (192.168.2.185)
- User: `jaita`, password in `~/.hermes/.env`
- CUDA: 13.0 at `/usr/local/cuda-13.0/bin/nvcc`
- 121GB unified memory, ~101GB used when model loads