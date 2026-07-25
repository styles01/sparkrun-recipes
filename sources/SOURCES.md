# Sources & Provenance

## Community Recipes & Benchmarks

### Spark Arena
- **Qwen 35B NVFP4 Recipe A (Luis Poveda):** https://spark-arena.com/benchmark/13321ed7-516e-412a-ba13-bf00c4d805c3
  - 104 data points, 109 tok/s single, 292 tok/s peak at c10
  - GMU 0.65, 4 seqs, MTP k=3, tool parser qwen3_xml

### GitHub Repos
- **MiaAI-Lab/Qwen3.6-35B-A3B-NVFP4-vLLM** — Production recipe with custom chat template (vision + thinking + tools). Mia's agentic benchmark: 91.0.
- **Entrpi/qwen3.5-122B-A10B-on-spark** — DFlash 122B deployment. Docker-based, aeon-vllm-ultimate image. The serve.sh includes tool calling by default (qwen3_xml).
- **Sapid-Labs/vLLM-Moet** (spark-gb10 branch) — DS4 deployment. Venv-based (not Docker). 2-bit MoE experts, prepacked planes. Tool calling NOT included in serve script (our fix).
- **eugr/spark-vllm-docker** — Community Docker recipes, env var optimizations (VLLM_MARLIN_USE_ATOMIC_ADD, VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS).

### Researchers / Twitter
- **fank** — DGX Spark community. Cracked Puzzle-75B MTP after 11 attempts. Key insight: no --enforce-eager, let FlashInfer autotune cache, 0.85 GMU, 131K ctx. ~30 tok/s.
- **DocAI (k3dani)** — Blog posts on Qwen3.6 MTP on GB10, vLLM tuning. docai.hu/en/blog/
- **MiaAI Lab** — Agentic workflow benchmarks, custom chat templates for Qwen 3.6.

### Internal (Mais/Lara/Oracle)
- `~/.hermes/workspace/dgx-spark/SPARK-STATE.md` — Hardware, software, model locations (Mais)
- `~/.hermes/workspace/dgx-spark/BENCHMARKS.md` — Full benchmark history (Mais)
- `~/.hermes/workspace/dgx-spark/QWEN-35B-NVFP4-RECIPE.md` — Three 35B recipes A/B/C (Mais)
- `~/.hermes/profiles/lara/memories/MEMORY.md` — Lara's locked 122B config
- `~/.hermes/profiles/oracle/docs/ds4-spark-deployment.md` — Oracle's DS4 deployment guide
- Skill: `ds4-dgx-spark-deploy` — Oracle's DS4 deployment skill

## vLLM Tool Parser Registry (vLLM-Moet 0.24.0 on Spark)

Verified available parsers on `~/venvs/vllm-moet/`:

### Tool Parsers (for `--tool-call-parser`)
- `deepseek_v4` → `deepseekv4_tool_parser` (extends DeepSeekV32, DSML format)
- `qwen3_xml` → `qwen3_engine_tool_parser` (XML function format)
- `qwen3_coder` → `qwen3_engine_tool_parser` (coder format)
- `hermes` → `hermes_tool_parser` (JSON format)
- `kimi_k2` → `kimi_k2_tool_parser`
- `glm47_moe` → `glm47_moe_tool_parser`
- + 30+ others (llama, mistral, granite, phi4mini, etc.)

### Reasoning Parsers (for `--reasoning-parser`)
- `deepseek_v4` → `deepseek_v3_reasoning_parser`
- `qwen3` → `qwen3_engine_reasoning_parser`
- + others

## Docker Images on Spark

| Image | Size | Used For |
|---|---|---|
| `vllm/vllm-openai:v0.24.0` | 21.3 GB | Qwen 35B NVFP4 |
| `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix` | 40.7 GB | Qwen 122B DFlash |
| `eugr/spark-vllm:latest` | 19.2 GB | (community alt) |
| (venv, no Docker) | — | DS4 (vLLM-Moet) |