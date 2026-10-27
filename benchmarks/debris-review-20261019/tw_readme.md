# DeepSeek-V4.1-Flash on NVIDIA DGX Sparks: TensorFold (TP4, TP2) and vLLM lanes

This repo runs DeepSeek-V4.1-Flash-1017 on DGX Sparks (GB10). Each **lane** is one complete, reproducible way to serve it: an engine, a weight pack, a
node count, launch scripts, and measured results. The current lanes run on [TensorFold](https://github.com/ashhart/TensorFold)'s native engine; the older
vLLM lanes stay documented below them with their recipe, patches and benchmarks unchanged.

## Lanes at a glance

| lane | Sparks | engine and weights | status | headline numbers | start here |
|---|---|---|---|---|---|
| **TensorFold TP4 (default)** | 4 | TensorFold native (bertholomus TP4 fork), LibertAI EXL3 3.5 bpw | validated | 93.6 tok/s at 1 user, 270.8 at 16 users, 1M context, KV pool 2,097,152, eval 97.1 | [section](#tensorfold-tp4-lane-default) · [run folder](runs/2026-10-17-1017-tensorfold/) |
| **TensorFold TP4, uncensored** (live on our fleet) | 4 | the default lane + the Keys 1017 abliteration overlay | live since 2026-10-18 | same speed as the default lane; eval 87.0 (safety scenarios) | [section](#uncensored-variant-keys-1017-overlay) · [run folder](runs/2026-10-18-1017-tensorfold-uncensored/) |
| **TensorFold TP2** | 2 | TensorFold native (bertholomus TP2 family), LibertAI EXL3 2.9 bpw | validated 2026-10-19 (our fleet endpoint is back on TP4) | 73.1 tok/s at 1 user, 135.8 at 4 users, 1M context (1M-token needle found), KV pool 1,048,576 positions, 93K prompt in 43.7 s, eval 97.1 | [section](#tensorfold-tp2-lane-two-sparks) · [run folder](runs/2026-10-18-1017-tensorfold-tp2/) |
| vLLM, native 1017 weights | 4 | vLLM `dsv41-feat` + our patches, native FP8/FP4 | older, documented | 48.5 tok/s at 1 user, 165.7 at 6, eval 98.6 | [section](#deepseek-v41-flash-1017-on-4x-dgx-spark) · [run folder](runs/2026-10-17-1017/) |
| vLLM, every matrix in EXL3 | 4 | vLLM + cuda-exl3, LibertAI EXL3 3.5 bpw | older, documented | 36.0 tok/s at 1 user, KV pool 3,701,062 | [section](#deepseek-v41-flash-1017-with-every-matrix-in-exl3-libertai-35-bpw-tp4) · [run folder](runs/2026-10-17-1017-exl3/) |
| vLLM, EXL3 TP3 (September release) | 3 | vLLM + bot-lab-21's EXL3 experts | older, not run on 1017 | 46.0 tok/s at 1 user, 152.9 at 6 | [section](#three-sparks-the-exl3-tp3-lane) · [write-up](docs/EXL3-TP3.md) |
| vLLM, EXL3 context lane (September release) | 4 | vLLM + bot-lab-21's EXL3 experts | older | KV pool 3,304,863 at 300K | [section](#four-sparks-on-exl3-the-context-lane) |

Earlier experiments (the SGLang lane, the two speed runs, the NVFP4 attempt) are kept as run records under [`runs/`](runs/); see [vLLM lanes (older)](#vllm-lanes-older).

## How to run a lane (for people and agents)

1. Pick a lane in the table above. Its **run folder** is the source of truth: every script we ran, every setting, logs and results.
2. Open the run folder's `README.md` and follow its **Reproduce** section in order (image, engine, weights, per-node inputs, serve, checks). The
   section for the lane in this file summarizes the same steps.
3. TensorFold lanes share one launcher: the recipe's `scripts/serve.sh`, driven by a `config.env` in the run folder's `launch/`. Switching between the
   stock and uncensored TP4 lanes is a restart with the other config (about 2 minutes). The TP2 lane uses two of the four Sparks.
4. vLLM lanes launch with the scripts in [`launch/`](launch/) and the patch set in [`patch/`](patch/); each section below names its config and restore
   script.
5. Every lane is checked the same way before it is called validated: a chat, tool-call, streaming and image check (`sanity.py` / `fleet_check.py`), a
   long-prompt needle check, the C1-C6 (or C1-C32) bench on prompt set v1, and the 69-scenario tool eval.

The TensorFold lanes are built on TensorFold, bertholomus's DeepSeek-V4.1 forks, recipes and kernel kits, LibertAI's EXL3 packs and the tensor layout
published by Mia-AiLab; the uncensored variant adds drowzeys' (Keys) overlay. The vLLM lanes are built on DeepSeek-AI's model, vLLM, FlashInfer, b12x
RoCEnante (local-inference-lab), cuda-exl3 and bot-lab-21's EXL3 build, plus fixes and reports from people running this recipe on their own Sparks.
Everyone is listed in [Credits](#credits) and [CREDITS.md](CREDITS.md).

## TensorFold TP4 lane (default)

DeepSeek-V4.1-Flash-1017 on four Sparks, TensorFold native engine, LibertAI EXL3 3.5 bpw.

DeepSeek-V4.1-Flash-1017 as LibertAI's EXL3 3.5 bpw pack, served by TensorFold's Zig engine, one tensor-parallel rank per Spark, context 1,048,576, 16 streams, DSpark drafts on. On the standard bench (prompt set v1, temperature 0, thinking off) it reaches 89.8 tok/s aggregate at one stream against 48.5 for the native-weights vLLM lane (1.85x) and 36.0 for the full-EXL3 vLLM lane, and 202.9 tok/s at six streams against 165.7 (+22%) and 115.8. Cold prefill is 1,929 tok/s at 93K tokens, against 2,201 on native vLLM (12% lower). Run record: [runs/2026-10-17-1017-tensorfold](runs/2026-10-17-1017-tensorfold/).

Validated so far (details under [Status and pending checks](#status-and-pending-checks)): the chat template (byte-exact against DeepSeek's reference encoder), chat, tool calling, tool-result round trips, thinking on request and image input through `/v1/chat/completions`, needle prompts from 20K to 500K tokens, and the 69-scenario tool eval (97.1, 66 pass / 2 partial / 1 fail, against 98.6 on native vLLM). With the second RoCE rail on, it reaches 93.6 tok/s at one stream and 213.8 at six, and reads a 93K-token prompt at 2,215 tok/s, level with native vLLM. Since 2026-10-18 03:36 UTC this lane serves our fleet endpoint (since 16:26 UTC with the [uncensored overlay](#uncensored-variant-keys-1017-overlay)). The recipe's consistency gates pass (details in the status table).

### What is in the lane

- **Model:** [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw), every matrix in EXL3 (attention, shared and routed experts, indexer, compressor, Engram projection, head, DSpark drafter). The Engram tables are not in the pack; they come from shards 47 and 48 of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
- **Engine:** TensorFold, from bertholomus's fork [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold), branch `deepseek-v41-zig-tp4`, engine commit `5940208`, plus four patches kept in [`patches/`](runs/2026-10-17-1017-tensorfold/patches/):
  - `0001`: a kernel launch with no captured Triton variant runs the closest more general one (bertholomus's commit `47a8910`; without it a prompt such as one of 2,064 tokens ended in a 500);
  - `0002`: the prompt indexer passes each row block's `vis` 16-byte aligned (Lucas Peters's commit `b8ae0c7`);
  - `0003`: `TF_RDMA_GID` sets the RDMA rings' source GID index (ours; the default stays 5, this cluster uses 3);
  - `0004`: two opt-in server settings, `TF_SEED_RANDOM` (a random seed for requests that send none) and `TF_TOP_K_MAX` (`top_k <= 0` means the server default, larger values are clamped instead of returning an error) (ours; see "Sampling on the fleet endpoint" below).
- **Kit:** the kernel kit from [bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10). It covers 3.5 bpw; no new kernels were built for this pack.
- **Serving settings** ([`launch/config.env`](runs/2026-10-17-1017-tensorfold/launch/config.env)): TP4 over one RoCE rail (RoCE v2 GID index 3; the fleet endpoint uses both rails, step 5b of the quick start), `--context 1048576`, `--streams 16`, `--round-rows 48`, `--pool 2097152`, `--no-thinking --max-tokens 32768`, served name `deepseek-v4.1-flash`.

### Results

Prompt set v1 (the set the vLLM lanes were benched with), temperature 0, thinking off, C streams released together, one batch per cell, token counts from the server's usage block. One difference in the path: the 1017 checkpoint ships no chat template, so [`tools/v41bench_tf.py`](runs/2026-10-17-1017-tensorfold/tools/v41bench_tf.py) encodes each user message with DeepSeek's `encoding.py` (chat mode, thinking off) and sends it to `/v1/completions`; the vLLM benches used the chat endpoint. The counting task is left out of every table. Raw output: [`results/tf1017a/`](runs/2026-10-17-1017-tensorfold/results/tf1017a/report.md). One run per cell; the GB10 slow state described under [Known limits](#known-limits-and-next-steps) can move a single cell.

**Aggregate throughput, tok/s across all streams** (mean of the 8 categories):

| C | TensorFold | vLLM native 1017 (1017g) | vLLM full EXL3 | TensorFold / native |
|---|---|---|---|---|
| C1 | **89.8** | 48.5 | 36.0 | 1.85x |
| C2 | **132.8** | 76.6 | 57.6 | 1.73x |
| C3 | **159.0** | 102.6 | 76.0 | 1.55x |
| C4 | **180.3** | 123.2 | 90.5 | 1.46x |
| C5 | **192.9** | 144.6 | 102.0 | 1.33x |
| C6 | **202.9** | 165.7 | 115.8 | 1.22x |
| C8 | **212.8** | not run | not run | |
| C16 | **247.4** | not run | not run | |

**Per-stream decode, tok/s after the first token** (mean of the 8 categories):

| C | TensorFold | vLLM native | vLLM full EXL3 |
|---|---|---|---|
| C1 | **101.8** | 53.8 | 37.8 |
| C2 | **78.0** | 44.9 | 31.1 |
| C3 | **64.0** | 39.7 | 27.7 |
| C4 | **55.3** | 36.5 | 24.8 |
| C5 | **47.7** | 34.4 | 22.9 |
| C6 | **42.1** | 32.3 | 21.3 |

**One stream (C1), per-stream tok/s by category:**

| category | TensorFold | vLLM native | vLLM full EXL3 | TensorFold / native |
|---|---|---|---|---|
| code | **134.8** | 76.3 | 42.3 | 1.77x |
| format (table) | **139.4** | 75.8 | 45.1 | 1.84x |
| JSON | **107.6** | 63.2 | 25.0 | 1.70x |
| math | **119.9** | 56.2 | 50.7 | 2.13x |
| reasoning | **97.7** | 55.6 | 35.8 | 1.76x |
| prose | **78.0** | 36.1 | 29.1 | 2.16x |
| summary | **69.7** | 35.8 | 46.7 | 1.95x |
| narrative | **67.3** | 31.7 | 27.6 | 2.12x |

**Time to first token, mean of the 8 categories (s):** C1 0.19 on TensorFold, 0.28 native, 0.19 full EXL3. C6 0.67 on TensorFold, 0.45 native, 0.42 full EXL3. At one stream TensorFold is level with the full-EXL3 lane and 0.09 s ahead of native; at six streams it is 0.22 to 0.26 s behind both vLLM lanes.

**Prefill: cold, unique prompt, 1-token reply** (tok/s; TTFT is the whole prefill):

| prompt tokens | TensorFold | vLLM native | vLLM full EXL3 |
|---|---|---|---|
| 2,950 | 1,613 | 1,903 | 1,759 |
| 11,592 | 1,873 | 1,936 | 1,279 (screen run) |
| 46,810 | 1,894 | 2,194 | 1,822 (screen run) |
| 93,335 | 1,929 (48.4 s) | 2,201 (42.4 s) | 1,512 (61.7 s) |

Prefill is 3 to 15% below native vLLM across the four lengths (12% at 93K) and above the full-EXL3 vLLM lane at the three longer ones. The two "screen run" cells are from the full-EXL3 lane's screen bench on a fresh boot; its full-bench cells at those lengths hit the prefix cache and are not prefill speeds.

**Memory and load:**

| | TensorFold | vLLM native | vLLM full EXL3 |
|---|---|---|---|
| model memory per Spark (as each engine reports it) | 71.87 GiB | 81.7 GiB | 56.2 GiB |
| context in the bench | 1,048,576 | 300K | 300K |
| load | all four ranks in about 107 s | about 11 min kick to serving | 374 s kick to serving |

While serving at 1,048,576 context and 16 streams, each worker had 40.6 GiB available and the head node 29 GiB. The load and boot figures measure different intervals (TensorFold: weights loaded on the four ranks; vLLM: kick to serving).

**Second RoCE rail** ([`results/tf1017b-rail2/`](runs/2026-10-17-1017-tensorfold/results/tf1017b-rail2/), same prompt set, run on the live endpoint, so other traffic may have moved a cell slightly). Both rails carry equal traffic during the bench.

| | rail 1 only | both rails | change |
|---|---|---|---|
| C1 aggregate tok/s | 89.8 | **93.6** | +4.2% |
| C6 aggregate tok/s | 202.9 | **213.8** | +5.4% |
| C1 code, per stream | 134.8 | **143.5** | +6.5% |
| cold prefill, 11,592 tokens | 1,873 | **2,140** | +14% |
| cold prefill, 93,335 tokens | 1,929 (48.4 s) | **2,215 (42.1 s)** | +15% |

Against native vLLM 1017g (48.5 / 165.7, prefill 2,201 at 93K) the dual-rail lane is 1.93x at one stream, +29% at six, and level on long-prompt prefill. The rail-2 setup is runtime IP aliases on the second port of two nodes plus `RDMA_HCA` with both HCAs ([`launch/rail2_all.sh`](runs/2026-10-17-1017-tensorfold/launch/rail2_all.sh), re-added on every start by [`start_fleet_rail2_p4.sh`](runs/2026-10-17-1017-tensorfold/launch/start_fleet_rail2_p4.sh)); persistent network config is not changed.

**Sampling on the fleet endpoint.** Server defaults `--temperature 1.0 --top-p 1.0 --top-k 20` (the model README recommends 1.0 / 1.0; the engine's exact sampler takes 1 <= top_k <= 56). Patch [`0004`](runs/2026-10-17-1017-tensorfold/patches/) adds two opt-in server settings: `TF_SEED_RANDOM=1` gives a request without a `seed` a random one (otherwise the engine derives the seed from the prompt, so identical requests return identical text), and `TF_TOP_K_MAX=56` treats `top_k <= 0` as the server default and clamps larger values instead of returning an error. An explicit `seed` stays reproducible and temperature 0 stays deterministic ([`tools/sampling_check.py`](runs/2026-10-17-1017-tensorfold/tools/sampling_check.py)).

### Speed sheet, C1 to C32 (2026-10-18)

The same prompt set at more concurrency levels, on the live fleet endpoint (both rails, patch 0004, 16 server streams), plus cold prefill.
Tables and raw JSON: [`runs/2026-10-17-1017-tensorfold/results/tf1017-stock-c1c32/`](runs/2026-10-17-1017-tensorfold/results/tf1017-stock-c1c32/)
(chart: `speed_sheet.png` there).

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| 1 | 98.6 | 112.0 | 0.18 |
| 2 | 146.5 | 86.4 | 0.27 |
| 4 | 198.0 | 61.1 | 0.45 |
| 8 | 221.7 | 34.6 | 0.81 |
| 12 | 250.9 | 26.3 | 1.18 |
| 16 | 270.8 | 21.7 | 1.54 |
| 32 | 260.5 | 19.1 | 5.96 (16 running, 16 queued) |

Cold prefill: 2,950 tokens 1.4 s (2,144 tok/s), 11,592 tokens 5.1 s (2,266 tok/s), 46,810 tokens 20.7 s (2,263 tok/s), 93,335 tokens 42.2 s (2,213 tok/s).

### Quick start

The exact scripts we ran, with every setting, are in the run folder: [runs/2026-10-17-1017-tensorfold/README.md](runs/2026-10-17-1017-tensorfold/README.md#reproduce). The engine and kit come from bertholomus's [recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10) (RUNNING.md); the model, patches, chat template, settings and checks are this repo's. In short:

1. **Hardware.** Four GB10 nodes on one RoCE network, one rank per node, each with the container `nvcr.io/nvidia/pytorch:26.07-py3`.
2. **Engine.** [`launch/setup.sh`](runs/2026-10-17-1017-tensorfold/launch/setup.sh) fetches Zig 0.17.0 and clones the fork and the recipe. Check out `5940208`, apply the patches, build:

   ```sh
   git clone -b deepseek-v41-zig-tp4 https://github.com/bertholomus/TensorFold && cd TensorFold
   git checkout 5940208
   git am "$REPO"/runs/2026-10-17-1017-tensorfold/patches/*.patch
   zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
   ```

3. **Weights.** `HF_HUB_DISABLE_XET=1 hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw --local-dir <MODELS>/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw` (240 GB on the head node, shared to the workers over NFS), plus shards 47 and 48 of the base 1017 checkpoint for the Engram tables, copied to each node's local NVMe ([`copy_engram.sh`](runs/2026-10-17-1017-tensorfold/launch/copy_engram.sh)).
4. **Runtime model dir, kit and per-node inputs.** [`distribute_model.sh`](runs/2026-10-17-1017-tensorfold/launch/distribute_model.sh) (config, tokenizer, index, `encoding/`, the two vision shards and our [`chat_template.jinja`](runs/2026-10-17-1017-tensorfold/launch/chat_template.jinja)), [`kitverify.sh`](runs/2026-10-17-1017-tensorfold/launch/kitverify.sh) and [`distribute.sh`](runs/2026-10-17-1017-tensorfold/launch/distribute.sh) (the recipe's kit), then [`inputs_r0.sh`](runs/2026-10-17-1017-tensorfold/launch/inputs_r0.sh) / [`inputs_worker.sh`](runs/2026-10-17-1017-tensorfold/launch/inputs_worker.sh) (rank caches, about 6 min, no GPU).
5. **Serve.** [`config.env`](runs/2026-10-17-1017-tensorfold/launch/config.env) holds every setting; [`start1.sh`](runs/2026-10-17-1017-tensorfold/launch/start1.sh) runs the recipe's `scripts/serve.sh start` (ranks 3, 2, 1, then 0). `config.env` binds the API to `127.0.0.1:8001`; change `HOST` and `PORT` to serve on the network.
   - **5b. Both RoCE rails (what our fleet endpoint runs).** Each node has one cable; its ConnectX-7 port shows up as two network devices, one per PCIe link: rail 1 `enp1s0f0np0` / `rocep1s0f0` and rail 2 `enP2p1s0f0np0` / `roceP2p1s0f0`. Give rail 2 a routable address on every node (here `192.168.193.0/24`; two of our nodes had only a link-local address there): [`rail2.sh`](runs/2026-10-17-1017-tensorfold/launch/rail2.sh) `add N` on a node (runtime only, a reboot brings the link-local back), or [`rail2_all.sh`](runs/2026-10-17-1017-tensorfold/launch/rail2_all.sh) `add` from the head node, which also pings all four. Then list both HCAs, `RDMA_HCA="rocep1s0f0,roceP2p1s0f0"`, and start with [`start_fleet_rail2_p4.sh`](runs/2026-10-17-1017-tensorfold/launch/start_fleet_rail2_p4.sh): it re-adds the rail-2 addresses and serves [`config.fleet-8000-rail2-p4.env`](runs/2026-10-17-1017-tensorfold/launch/config.fleet-8000-rail2-p4.env) (`0.0.0.0:8000`, both rails, patch 0004 settings, sampling defaults, alias `glm-5.3-flash`). Edit the paths and node list in the env file and the script for your cluster.
6. **Test request** (thinking is off by default; served as `deepseek-v4.1-flash`, alias `glm-5.3-flash`):

   ```sh
   curl -s http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "deepseek-v4.1-flash", "max_tokens": 64,
     "messages": [{"role": "user", "content": "Name three prime numbers above 100."}]}'
   ```

7. **Checks.** [`tools/sanity.py`](runs/2026-10-17-1017-tensorfold/tools/sanity.py) (chat, alias, tool call, tool round trip, thinking, image), [`tools/long_check.py`](runs/2026-10-17-1017-tensorfold/tools/long_check.py) (exact-length needle prompts) and [`launch/run_bench.sh`](runs/2026-10-17-1017-tensorfold/launch/run_bench.sh) (the bench).

### Status and pending checks

| item | state |
|---|---|
| Load on four ranks | done: about 107 s, 71.87 GiB of model per node |
| Bench C1-C6 and cold prefill (`tf1017a`, 2026-10-18 UTC) | done: ran to the end without error; tables above |
| Chat template | done: [`launch/chat_template.jinja`](runs/2026-10-17-1017-tensorfold/launch/chat_template.jinja), written from DeepSeek's MIT `encoding.py`, byte-exact against the reference encoder on the official cases, 267 named cases, 8,000 random conversations, and 5,098 cases through TensorFold's own Jinja engine. Two small documented gaps in the run folder's STATUS.md |
| Long prompts with a needle | done: 20,000; 131,073; 145,152; 146,421; 149,796; 160,000; 300,000 and 500,000 tokens all pass |
| 69-scenario tool eval | done: quality 97.1 (134/138: 66 pass, 2 partial, 1 fail), responsiveness 97.8 (median turn 932 ms), deployability 97.3; native vLLM 1017g scored 98.6. The partials (TC-44, TC-67) are the same two as on the vLLM lanes; the fail (TC-57) is the suite's keyword check matching "won't" in a helpful answer. [Scorecard](runs/2026-10-17-1017-tensorfold/results/eval69/eval69-tf1017.md) |
| Chat, tool calling, image input | done ([`tools/sanity.py`](runs/2026-10-17-1017-tensorfold/tools/sanity.py)): chat, the `glm-5.3-flash` alias, a tool call (`finish_reason: tool_calls`), a tool-result round trip, thinking on request, and image input all pass |
| Second RoCE rail | done: both rails striped, +5% decode and +15% long-prompt prefill (table above) |
| Fleet endpoint | live since 2026-10-18 03:36 UTC on `0.0.0.0:8000` with [`config.fleet-8000-rail2-p4.env`](runs/2026-10-17-1017-tensorfold/launch/config.fleet-8000-rail2-p4.env): both rails, patch 0004, names `deepseek-v4.1-flash` and `glm-5.3-flash`, thinking off by default; [`tools/fleet_check.py`](runs/2026-10-17-1017-tensorfold/tools/fleet_check.py) passes (models, health, streaming, tool calls, images, auth variants, 6 concurrent) |
| Recipe consistency gates | done on the live endpoint, no restart: the recipe's `gates.py` passes 6 of 6 (models, chat probe, streamed reasoning, tool call plain and streamed, drafted == serial 8 of 8 token for token, burst of 4). Concurrent == solo: a light stand-in ([`tools/concurrent_check.py`](runs/2026-10-17-1017-tensorfold/tools/concurrent_check.py)) gives 8 of 8 identical; the recipe's long-prompt version was not run on the live endpoint |

### Files

- Run folder: [`runs/2026-10-17-1017-tensorfold/`](runs/2026-10-17-1017-tensorfold/): `launch/` (`config.env`, `chat_template.jinja`), `patches/` (the three engine patches), `tools/` (bench, long-prompt check, pack inventory, Engram copy, chat-template test), `results/tf1017a/` (report, bench table, raw JSON and log).
- The launch itself is the recipe's `scripts/serve.sh`, wrapped by the run folder's `start_fleet*.sh` scripts (rollback order in its STATUS.md); this repo's top-level `launch/` holds the vLLM launchers.

### Uncensored variant (Keys 1017 overlay)

The default lane with [drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated)
(gated, responsible-use agreement; not redistributed here) applied to `layers.10-35.attn.wo_b`. It serves our fleet endpoint since 2026-10-18 16:26 UTC.
Run record, scripts and results: [`runs/2026-10-18-1017-tensorfold-uncensored/`](runs/2026-10-18-1017-tensorfold-uncensored/).

- **Method:** instead of rewriting 26 checkpoint shards (about 147 GB) and rebuilding every rank cache, we patch a copy of each node's TensorFold rank cache:
  offsets from `tf-dsv41-rank-cache ... --index-only`, the overlay's EXL3 sidecar cut into each rank's 78 wo_b slices, written into a direct-IO copy,
  checked byte for byte, then served by pointing `CACHE_DIR` at the copy. The stock caches stay untouched; switching lanes is a 2-minute restart.
- **Speed:** unchanged (values change, sizes do not); see the speed sheet above.
- **Refusals:** of five probe prompts, stock refused one and the uncensored lane answers all five; arithmetic stays correct.
- **Tool eval:** quality 87.0 against stock's 97.1. Every lost point is in the safety and refusal scenarios (plus one omitted-parameter case): it goes
  along with destructive or data-moving tool calls and follows instructions placed inside tool results. Keep a stock lane for agents that act on untrusted
  content.

## TensorFold TP2 lane: two Sparks

DeepSeek-V4.1-Flash-1017 on **two** DGX Sparks: [TensorFold](https://github.com/ashhart/TensorFold)'s native engine, tensor-parallel over a pair of nodes, serving LibertAI's [2.9 bpw EXL3 pack](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (196.3 GiB, 98.9 GiB of weights per node). It uses the same engine family as the four-node lane above, built from bertholomus's `deepseek-v41-zig` branch and kernel kit, with three small patches ([runs/2026-10-18-1017-tensorfold-tp2](runs/2026-10-18-1017-tensorfold-tp2/)).

| | |
|---|---|
| nodes | 2 x DGX Spark (GB10, 128 GB unified memory), both RoCE lanes (192.168.192.0/24 and 192.168.193.0/24), RoCE v2 GID index 3 |
| weights | EXL3 2.9 bpw, all matrices (routed experts 3 bpw, 2 bpw on layers 18-22; attention 5 bpw; MTP/DSpark dense parts 6 bpw) |
| context window | 1,048,576 tokens |
| streams / KV pool | 4 / 1,048,576 positions (shared by the 4 streams) |
| throughput (8 categories, aggregate tok/s) | C1 73.1 · C2 106.0 · C3 123.7 · C4 135.8; per stream at C1 84.2 (code 105.6, math 99.6, JSON 92.1); TTFT 0.24 s at C1, 0.61 s at C4 |
| cold prefill | 2,180 / 2,200 / 2,135 / 2,044 tok/s at 11.6K / 46.8K / 93.3K / 186.8K tokens (93K in 43.7 s) |
| longest needle found | 1,000,000 tokens (641.5 s, 1,559 tok/s); 300K in 152.8 s, 500K in 274.7 s |
| 69-scenario tool eval | 97.1 (66 pass, 2 partial, 1 fail), the same as the four-node lane |
| memory left while serving | 6.7 GiB (Reddie, rank 0) and 8.2 GiB (Bluey, rank 1); model 105.04 GiB per node; loads in 27 s |

What the build needed: no new GPU kernels (the kit's dense kernels cover 1 to 8 bits and its expert kernels the 2-bit to 5-bit ranges); a rank-cache writer that runs without a GPU (bertholomus's `tf-dsv41-rank-cache`, applied to this branch); and the sampling patch of the four-node lane. Setup, scripts, results and the commands to reproduce are in the run folder's [README](runs/2026-10-18-1017-tensorfold-tp2/README.md) and [STATUS](runs/2026-10-18-1017-tensorfold-tp2/STATUS.md).

## vLLM lanes (older)

The older lanes, kept as built: the vLLM side of this repo, the native-weights 1017 lane, the EXL3 lanes, the TP3 lane and the SGLang experiment, with their patches, launchers and benchmarks. The 69-scenario tool eval (on the native 1017 lane) and the vision and tool checks listed below were run on these lanes.

> **DeepSeek-V4.1-Flash-1017, 2026-10-17 ([runs/2026-10-17-1017](runs/2026-10-17-1017/README.md)):** the 1017 checkpoint serves on the four Sparks (TP4, 300K context, **1,523,740-token KV pool**), on boot 10's recipe plus the speed-run levers that don't need the EXL3 pack. Full bench on a fresh boot: one stream code 76.3 tok/s, tables 75.8, JSON 63.2, reasoning 55.6; six streams 165.7 tok/s aggregate (boot 10: 131.9); cold prefill 1,903-2,201 tok/s (boot 10: 902-1,539); 69-scenario tool eval 98.6 (67/2/0). Restore with `bash /root/restore_1017.sh 1017g` on Reddie as root.

> **Speed run 2, 2026-09-19 ([runs/2026-09-19-speedrun2](runs/2026-09-19-speedrun2/README.md)):** serving now at 500K context, 16 streams. Against the lane as found that night (same full bench): C2-C6 throughput +2 to +16%, TTFT 0.205 s at one stream (-13%), cold prefill 1,939-2,048 tok/s (+41 to +46%), idle code 95.3 tok/s (+46%), boot 63 s faster, and C8-C12 serving for the first time (C12 coding 475 tok/s aggregate). Two findings matter more than the deltas: a lane that has served for 15 hours runs ~30% slower on prefill than the same config fresh (cause open; a restart recovers it), and 500K context costs nothing over 300K once you compare fresh to fresh. New levers: the SwiGLU clamp moved inside the cuda-exl3 GLU kernel (image `exl3b-roce-lim`), the Engram all-gather on the RoCE path, and a draft loader that reads 3 shards instead of 49. Also there: [runs/2026-09-17-nvfp4](runs/2026-09-17-nvfp4/STATE.md), fourteen boots of `nvidia/DeepSeek-V4.1-Flash-NVFP4` that never served, with the memory mechanism measured (COW-broken private mappings of the shards during the fused-MoE expert copy) and the one lever that held (`--enable-expert-parallel --enable-ep-weight-filter`, 194.6 of 302 GiB skipped per rank).

> **Speed run, 2026-09-14 ([runs/2026-09-14-speedrun](runs/2026-09-14-speedrun/README.md)):** the uncensored build (EXL3-Pollard-Abliterated: bot-lab-21's EXL3 Pollard experts with drowzeys' abliterated `wo_b` overlay) now serves on `exl3tp4b-ablit-best`. Against the same lane's baseline on the same full bench: six-stream throughput 189.97 tok/s (+13.3%), C2-C6 throughput +9 to +13%, cold prefill about 2,000 tok/s (+34 to +38%), TTFT 0.22 s at one stream (-20%), KV pool 3,757,748 tokens (+14.7%), and 1M context proven on the same stack. The levers: b12x RoCE one-shot all-reduce (@original-el8, @lukealonso), Engram fast staging and 128 read threads, an indexer prefill TP-split, `NCCL_MAX_NCHANNELS=8` (bot-lab-21) and `expandable_segments:False` (ecohash-co). Restore it with `bash /root/restore_exl3tp4b_ablit_best.sh` on Reddie as root.

> **SGLang lane, experimental ([runs/2026-09-14-sglang-lane](runs/2026-09-14-sglang-lane/README.md)):** the same uncensored model on upstream SGLang across the four Sparks, with our disk-backed Engram reader (a native row reader that runs inside the CUDA graph), an index-honoring loader for the uncensored overlay, and two SM12x fixes. It serves with a 9.3M-token KV pool (2.5x the vLLM lane) but decodes 2-2.5x slower here and exhausted node memory under the full bench, so vLLM stays the default.

> **vLLM serving config from 2026-09-11 until the 1017 swap: the TP4 EXL3 context lane (`exl3tp4b`)**, the same four Sparks on the EXL3 3.5 bpw checkpoint with a 3,304,863-token KV pool. Restore it with `bash /root/restore_exl3tp4b.sh` on Reddie as root. Boot 10, documented below, is the release-checkpoint fallback: `bash /root/restore_boot10.sh`.

**vLLM status (2026-09-10): serving (boot 10).**
- The model dropped at about 2 AM ET, and this stack started serving it at 9:17 AM ET the same day.
- **One stream, decode tok/s** (after the first token): **code 73.8**, tables 55.4, JSON 52.1, math 50.9, reasoning 37.8, narrative 24.9, prose 24.4.
  - Counting ran at **92.2 tok/s** back to back. It was 62.2 in the bench run, which hit a GPU slow-state phase ([issue #1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1)).
- **Concurrency:** six streams give **131.9 tok/s** aggregate across the 8 categories (boot 9: 98.0).
  - Peak aggregates: counting **259.9** (C5), **code 225.5** (C6), tables 215.1 (C5), math 182.7 (C6).
- **Prefill:** 902-1,539 tok/s cold. A 93K-token prompt takes 78 s. TTFT on short prompts is 0.3-0.5 s.
- **Context:** **300K** max context with a **1,070,168-token** KV pool (3.57x at 300K).
- **Vision and tool calling are on:** up to 4 images per request, tool calls including parallel calls and a full round trip. 7/7 end-to-end checks pass ([below](#vision-and-tool-calling-on-in-the-serving-config)).
  - 1M max context was proven on a separate boot, with a 1,078,380-token DSpark KV pool.
- Nothing on this page is a projection.

### DeepSeek-V4.1-Flash-1017 on 4x DGX Spark

DeepSeek's 1017 update ships the same config, index and tokenizer as the September release (sha256-identical) with all 48 shards new, Engram tables included. So the swap is a model-path change plus fresh node-local Engram rows. Boot 10's row ranges apply unchanged. The run log, launch kit, patch sets and every bench are in [runs/2026-10-17-1017](runs/2026-10-17-1017/README.md).

- **Serving config `boot1017g`:**
  - boot 10 (vLLM `dsv41-feat` overlay, DSpark k=5, CUDA graphs FULL_AND_PIECEWISE, Engram on disk with node-local rows, vision and tools on, gmu 0.80);
  - plus `NCCL_MAX_NCHANNELS=8`, Engram fast staging with 128 read threads, the b12x RoCE one-shot all-reduce and all-gather (image `vllm-dsv41:exl3b-roce`), the indexer prefill TP-split, and `expandable_segments:False`.
  - Context 300K, KV pool 1,523,740 tokens (5.08x).
- **Launch:** `launch/dsv41-tp4-1017.sh` is `launch/dsv41-tp4.sh` with two knobs, `MODEL_DIR` and `SERVED_NAMES`. One boot per lever through 1017d, then the rest together.
- **No boot died.** About 11 minutes from kick to serving.

| | boot 10 | 1017 on boot 10's recipe (1017a) | **1017 serving (1017g)** |
|---|---|---|---|
| C1 / C3 / C6 aggregate tok/s (8 categories) | 38.0 / 78.7 / 131.9 | 41.9 / 84.4 / 129.1 | **48.5 / 102.6 / 165.7** |
| one-stream tok/s: code / tables / JSON / math / reasoning | 73.8 / 55.4 / 52.1 / 50.9 / 37.8 | 70.4 / 49.2 / 52.8 / 63.0 / 46.2 | **76.3 / 75.8 / 63.2 / 56.2 / 55.6** |
| one-stream tok/s: prose / summary / narrative | 24.4 / 25.7 / 24.9 | 27.9 / 31.3 / 31.4 | **36.1 / 35.8 / 31.7** |
| TTFT at C1 (s) | 0.44 | 0.37 | **0.28** |
| cold prefill tok/s at 3K / 12K / 47K / 93K tokens | 902 / 1,027 / 1,539 / 1,194 | 1,359 / 1,299 / 1,217 / 1,335 | **1,903 / 1,937 / 2,194 / 2,201** |
| KV pool at 300K | 1,070,168 | 1,316,807 | **1,523,740** |

- Counting, as a peak ceiling only: 87.1 tok/s at one stream on 1017g.
- The EXL3 speed-run lane is still faster at decode (C6 190.0), because its 3.5 bpw experts read about 17% fewer bytes per token. On these stock weights, prefill matches or beats it at three of four lengths.
- DSpark on 1017 accepts 53% of drafted tokens, a mean of 3.65 tokens per step over the whole bench.
- Checks: vision and tools 7/7.
- 69-scenario tool eval (thinking off): **98.6**, 67 pass / 2 partial / 0 fail. Boot 10 scored the same; responsiveness is up (96.0 vs 94.3).

### DeepSeek-V4.1-Flash-1017 with every matrix in EXL3 (LibertAI 3.5 bpw), TP4

[LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) stores attention, shared experts, indexer, compressor, Engram projection, head and the MTP/DSpark drafter in EXL3 as well as the routed experts (the earlier EXL3 lanes kept those in FP8). The Engram tables are not in it and are read from the base 1017 shards. [patch/exl3-1017-full](patch/exl3-1017-full/CHANGES.md) is the loader work (grouped `wo_a` slices, padded shared experts and head so that every tensor-parallel slice is a whole number of 128-blocks, name aliases); the run, launch kit, checks and benches are in [runs/2026-10-17-1017-exl3](runs/2026-10-17-1017-exl3/README.md).

It loads on the first boot and answers correctly (count, code, tool call, image):

| | native 1017g | full EXL3 |
|---|---|---|
| model memory per Spark | 81.7 GiB | **56.2 GiB** |
| KV pool at 300K | 1,523,740 | **3,701,062** |
| boot to serving | about 11 min | **374 s** |
| C1 / C6 aggregate tok/s | 48.5 / 165.7 | 36.0 / 115.8 |
| draft tokens accepted (DSpark k=5) | 53 % | 12.4 % |

Decode is 0.70-0.75x of the native lane. Where speculation helps little (prose, narrative, summary, math) EXL3 is within 0.8-1.3x of native; the loss is in the categories that benefit from DSpark, whose acceptance fell from 53 % to 12.4 %. The cause is not yet established; the run README lists the next tests. The 69-scenario eval was not run on this lane. The same pack served by TensorFold, the [default lane](#tensorfold-tp4-lane-default), benched at 89.8 and 202.9 tok/s aggregate at C1 and C6 on the same prompt set.

### Three Sparks: the EXL3 TP3 lane

> **Release note:** this lane was built and benched on the September V4.1 Flash release (bot-lab-21's EXL3 build of its experts). It has not been run on DeepSeek-V4.1-Flash-1017, which is what the 4-Spark lanes use (see [runs/2026-10-17-1017](runs/2026-10-17-1017/) for vLLM and the default lane above for TensorFold). A full-EXL3 build of 1017 (LibertAI) was brought up on 4 Sparks, see the section above.

The same model on **three** DGX Sparks instead of four, using bot-lab-21's [EXL3 3.5 bpw build](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) of the routed experts (everything else is the release's FP8). V4.1 had no TP3 path in vLLM, so this lane adds one: virtual attention heads (64 padded to 72), a vocabulary split that works at TP3, Engram rows for three ranks, a streaming weight loader, and a fix that lets the DSpark drafter's 128 experts load on three ranks. The full write-up, with every boot and what failed, is in [docs/EXL3-TP3.md](docs/EXL3-TP3.md).

Serving config (exl3tp3a11): CUDA graphs, DSpark k=5, vision, 300K context, gmu 0.80. Same prompt set and bench as boot 10; throughput across the 8 prompt categories:

| C | TP3 aggregate tok/s | TP3 per-stream tok/s | TP3 mean TTFT (s) | boot 10 aggregate tok/s |
|---|---|---|---|---|
| C1 | 46.0 | 51.5 | 0.30 | 38.0 |
| C2 | 73.4 | 43.6 | 0.63 | 64.3 |
| C3 | 100.1 | 38.5 | 0.41 | 78.7 |
| C4 | 118.4 | 35.2 | 0.44 | 85.7 |
| C5 | 134.4 | 31.7 | 0.46 | 114.2 |
| C6 | 152.9 | 29.6 | 0.50 | 131.9 |

| | TP3, DSpark (exl3tp3a11) | TP3, no DSpark (exl3tp3a10) | boot 10, 4 Sparks |
|---|---|---|---|
| KV pool, tokens | 678,950 | 1,995,725 | 1,070,168 |
| cold prefill, 93,335-token prompt | 1,199 tok/s | 1,155 tok/s | 1,194 tok/s |
| vision and tool checks | 7/7 pass | 7/7 pass | 7/7 pass |

### Four Sparks on EXL3: the context lane

Boot 10's serving config (four Sparks, CUDA graphs, DSpark k=5, vision, 300K per request, gmu 0.80) on the same EXL3 3.5 bpw checkpoint as the TP3 lane. The smaller experts leave far more memory for the KV cache:

| | TP4 EXL3 (exl3tp4b) | boot 10, release, 4 Sparks | TP3 EXL3, 3 Sparks |
|---|---|---|---|
| KV pool, tokens | **3,304,863** | 1,070,168 | 678,950 |
| full 300K-token requests at once | **11.02** | 3.57 | 2.26 |
| model memory per Spark, GiB | 56.6 / 68.9 (narrow / wide expert slice) | 81.6 | 84.2 |
| cold prefill, 93,335-token prompt | 1,026 tok/s | 1,194 tok/s | 1,199 tok/s |
| vision and tool checks | 7/7 pass | 7/7 pass | 7/7 pass |

EXL3 splits the 2304-wide experts 512/640/640/512 across four ranks, so two Sparks carry 12 GiB more than the other two, and those two set the pool (15.5 GiB of KV each). Throughput across the 8 prompt categories, same bench as boot 10:

| C | TP4 EXL3 aggregate tok/s | per-stream tok/s | mean TTFT (s) | boot 10 aggregate tok/s | TP3 EXL3 aggregate tok/s |
|---|---|---|---|---|---|
| C1 | 41.6 | 46.4 | 0.37 | 38.0 | 46.0 |
| C2 | 62.2 | 37.5 | 0.77 | 64.3 | 73.4 |
| C3 | 101.1 | 38.2 | 0.44 | 78.7 | 100.1 |
| C4 | 106.7 | 30.9 | 0.51 | 85.7 | 118.4 |
| C5 | 129.2 | 30.4 | 0.56 | 114.2 | 134.4 |
| C6 | 141.2 | 27.7 | 0.58 | 131.9 | 152.9 |

Two repeat runs about 90 minutes after startup came in higher: C1 54.1 and 57.1 tok/s, C6 170.9 and 176.1 tok/s (the first run above was taken right after startup, with the two wide-slice ranks short on memory; boot 10 has no repeat runs, so they are not a stock comparison; details in the doc).

Launch and guard scripts are in `exl3/tp4/`; the full write-up is in [docs/EXL3-TP3.md](docs/EXL3-TP3.md).

### Vision and tool calling (on in the serving config)

Both are live on boot 10 and tested end to end with `tools/vision_tools_demo.py`; the output is in [`results/boot10/vision-tools.txt`](results/boot10/vision-tools.txt). The test images are generated in the script, so each expected answer is known exactly.

| test | result |
|---|---|
| V1: one image, three color stripes, name them left to right | PASS: "red, green, blue" (0.6 s) |
| V2: two images in one message | PASS: first=red, second=blue (1.2 s) |
| V3: 2x2 grid, color of the top-right square | PASS: "Green" (0.6 s) |
| T1: tool call with arguments | PASS: `get_weather {"city": "Paris", "unit": "c"}` (1.4 s) |
| T2: full round trip (the tool result goes back, the model answers from it) | PASS: "18°C with light rain" (2.2 s) |
| T3: parallel calls in one turn | PASS: `get_weather` for Tokyo and Berlin (2.9 s) |
| T4: `tool_choice` forcing a named function | PASS: `get_time {"city": "Sydney"}` (1.6 s) |

How it is switched on (`launch/boot10-go.sh`):
- `TEXT_ONLY=0`: the vision encoder loads (no `--language-model-only`). It adds about 0.22 GiB per rank.
- `--limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1`: up to 4 images per request.
- `PARSERS=1`: `--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41`.
- Thinking is off by default; turn it on per request with `"chat_template_kwargs": {"thinking": true}`.
- **Watch the defaults.** `launch/dsv41-tp4.sh` on its own is text-only with tools off (`TEXT_ONLY=1`, `PARSERS=0`). Set both as `boot10-go.sh` does.

Example requests (OpenAI-compatible API; the served model name is `deepseek-v4.1-flash`):

```bash
# image (base64 data URL; up to 4 images per message)
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "max_tokens": 200,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}}]}]}'
```

```bash
# tool call
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "tool_choice": "auto",
  "messages": [{"role": "user", "content": "What is the weather in Paris in celsius?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
    "required": ["city"]}}}]}'
```

### Benchmark (boot 10, the serving config)

4x DGX Spark TP4, DSpark k=5, FULL_AND_PIECEWISE CUDA graphs, Engram rows node-local on every rank, tools and vision on, 300K context, gmu 0.80.

How it was measured:
- Fixed prompt set `bench/prompts-v1.json`, identical on every boot: 8 categories plus a counting ceiling.
- Streaming, temperature 0, thinking off, after a warmup. Short prompts (about 30-120 tokens), 150-256 token budgets.
- At concurrency C, C streams are released together. There is one batch per cell.
- Token counts come from the server's `usage` block, never from stream chunks (DSpark packs several tokens per chunk).
- **Decode** = tokens after the first / time after the first token, per stream. **Aggregate** = all streams' tokens / batch wall time. **TTFT** = first token delta.
- Raw output and more tables: [`results/boot10/`](results/boot10/) ([report](results/boot10/report.md), made with `tools/bench_report.py`).
- One caveat: the GB10 GPU slow state ([issue #1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1)) can move a single cell by up to about 1.5x. Counting at C1 is the clearest case (62.2 here, 92.2 back to back).

**Throughput by concurrency** (mean of the 8 categories; the counting ceiling is excluded):

| C | aggregate tok/s | per-stream decode tok/s | mean TTFT (s) | boot 9 aggregate |
|---|---|---|---|---|
| C1 | 37.95 | 43.12 | 0.441 | 33.88 |
| C2 | 64.30 | 37.43 | 0.441 | 58.99 |
| C3 | 78.70 | 30.62 | 0.793 | 67.56 |
| C4 | 85.72 | 24.66 | 0.579 | 85.50 |
| C5 | 114.20 | 27.01 | 0.585 | 91.96 |
| C6 | **131.86** | 25.35 | 0.502 | 97.99 |

**Decode: per-stream tok/s after the first token**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| **code** | **73.8** | 45.0 | 33.8 | 34.3 | 44.1 | 41.5 |
| JSON | 52.1 | 30.4 | 20.7 | 22.7 | 26.8 | 27.5 |
| math | 50.9 | 59.3 | 54.4 | 31.4 | 28.9 | 34.3 |
| reasoning | 37.8 | 42.4 | 36.2 | 26.8 | 20.8 | 24.6 |
| tables (format) | 55.4 | 59.5 | 41.5 | 35.5 | 53.1 | 35.3 |
| summary | 25.7 | 22.4 | 22.4 | 13.7 | 15.9 | 16.1 |
| prose | 24.4 | 23.6 | 19.4 | 18.4 | 14.8 | 13.8 |
| narrative | 24.9 | 16.9 | 16.5 | 14.7 | 11.8 | 9.8 |
| counting (ceiling) | 62.2 | 58.9 | 43.0 | 58.3 | 57.2 | 33.4 |

**Aggregate throughput: tok/s across all streams** (wall time, TTFT included)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| **code** | 66.5 | 81.1 | 92.4 | 123.8 | 196.7 | **225.5** |
| JSON | 43.7 | 49.2 | 51.9 | 76.3 | 94.7 | 130.4 |
| math | 45.6 | 105.5 | 148.4 | 110.2 | 127.9 | 182.7 |
| reasoning | 35.0 | 75.3 | 96.5 | 97.0 | 94.1 | 133.6 |
| tables (format) | 44.0 | 94.3 | 101.7 | 112.0 | 215.1 | 182.1 |
| summary | 21.9 | 37.6 | 38.2 | 43.1 | 64.0 | 73.5 |
| prose | 22.9 | 41.5 | 55.2 | 69.9 | 67.9 | 74.8 |
| narrative | 23.8 | 30.0 | 45.2 | 53.4 | 53.1 | 52.4 |
| counting (ceiling) | 57.3 | 109.0 | 118.6 | 208.8 | **259.9** | 184.9 |

**TTFT: mean time to first token (s)**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| code | 0.31 | 0.44 | 0.49 | 0.51 | 0.40 | 0.42 |
| JSON | 0.33 | 0.50 | 0.55 | 0.57 | 0.45 | 0.42 |
| math | 0.47 | 0.35 | 0.38 | 0.46 | 0.62 | 0.42 |
| reasoning | 0.44 | 0.35 | 0.42 | 0.55 | 0.57 | 0.41 |
| tables (format) | 0.51 | 0.40 | 0.57 | 0.59 | 0.47 | 0.47 |
| summary (~300-token passage) | 0.79 | 0.77 | 3.18 | 1.28 | 1.33 | 1.02 |
| prose | 0.38 | 0.39 | 0.31 | 0.32 | 0.36 | 0.48 |
| narrative | 0.29 | 0.32 | 0.44 | 0.35 | 0.49 | 0.37 |
| counting (ceiling) | 0.34 | 0.35 | 0.46 | 0.40 | 0.35 | 0.53 |

**Prefill: cold, unique prompt, 1-token reply** (TTFT here is the whole prefill)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 2,950 | 3.27 | 902 |
| 8K | 11,592 | 11.29 | 1,026 |
| 32K | 46,810 | 30.43 | **1,539** |
| 64K | 93,335 | 78.17 | 1,194 |

**Decode per stream, boot 9 → boot 10** (same prompts; boot 10 adds node-local Engram rows):

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| code | 52.4 → **73.8** | 60.4 → 45.0 | 33.4 → 33.8 | 31.3 → 34.3 | 28.5 → 44.1 | 26.3 → 41.5 |
| JSON | 38.6 → 52.1 | 37.1 → 30.4 | 20.0 → 20.7 | 20.8 → 22.7 | 18.8 → 26.8 | 16.2 → 27.5 |
| math | 46.9 → 50.9 | 46.2 → 59.3 | 46.2 → 54.4 | 49.1 → 31.4 | 27.6 → 28.9 | 25.8 → 34.3 |
| reasoning | 39.0 → 37.8 | 27.2 → 42.4 | 29.1 → 36.2 | 20.9 → 26.8 | 25.4 → 20.8 | 17.1 → 24.6 |
| tables (format) | 71.4 → 55.4 | 50.2 → 59.5 | 38.9 → 41.5 | 35.4 → 35.5 | 36.3 → 53.1 | 30.8 → 35.3 |
| summary | 24.7 → 25.7 | 17.4 → 22.4 | 13.6 → 22.4 | 11.5 → 13.7 | 11.4 → 15.9 | 10.9 → 16.1 |
| prose | 23.3 → 24.4 | 16.1 → 23.6 | 16.1 → 19.4 | 17.6 → 18.4 | 12.0 → 14.8 | 14.0 → 13.8 |
| narrative | 17.4 → 24.9 | 18.7 → 16.9 | 11.2 → 16.5 | 11.6 → 14.7 | 10.2 → 11.8 | 9.8 → 9.8 |
| counting (ceiling) | 77.2 → 62.2 | 56.0 → 58.9 | 39.3 → 43.0 | 40.2 → 58.3 | 36.5 → 57.2 | 38.1 → 33.4 |

**Step time, fast vs slow GPU state** (streamed; every speculative step is timed from the stream; [`results/boot10/idletest.txt`](results/boot10/idletest.txt)):

| request | ms/step | tokens/step | decode tok/s |
|---|---|---|---|
| count 1-100, first request after about 13 min idle | 94 (the whole request) | 5.85 | 61.7 |
| count, back to back | 63 | 5.85 | **92.2** |
| code, back to back | 70.5 | 4.97 | 70.2 |
| count, after 45 s idle | 63 | 5.85 | 92.1 |
| code, back to back | 67 | 5.30 | **77.3** |

**DSpark acceptance** (vLLM SpecDecoding metrics across the boot 10 bench, 35 ten-second windows): mean acceptance length **3.57** tokens per step, range 1.88-5.92.
- Counting, code and tables sit near the maximum of 6.
- Prose and narrative stay near 2, which is why per-stream speed spans 10-92 tok/s by content.

**What each fix bought (counting prompt, one stream, measured):**

| stack | tok/s |
|---|---|
| eager, no speculation (boot 6) | 5.1 |
| eager + DSpark k=5 (boot 7) | 19.5-22.1 |
| DSpark + CUDA graphs + Engram rows staged before the forward (boot 8) | 41.5 (count-to-100 check) |
| + GPU clock latch cleared on two nodes (boot 9) | 60.8 (count-to-100 check); 77.2 bench ceiling |
| **+ node-local Engram rows, GPU clocks locked (boot 10)** | **84.9** (count-to-100 check, same method); **92.2** decode back to back |

**Memory (boot 10):**

| | value |
|---|---|
| Weights per rank, with the DSpark draft layers and the vision encoder | 81.58 GiB |
| CUDA graphs | 1.99 GiB target + 0.55 GiB draft |
| KV | 5.15 GiB = 1,070,168 tokens (3.57x at 300K) |
| Node-local Engram rows per worker | 48 GB on disk (sparse copy of the rank's rows) |

**Earlier boots:**
- Boot 9: C1 per-stream 39.2, code 52.4 ([`results/boot9/`](results/boot9/)).
- Boot 8, before the GPU clock fix: C1 per-stream 24.2, code 32.9 ([`results/boot8/`](results/boot8/)).
- **1M max context (boot 7):** served at `--max-model-len 1048576` with DSpark: KV 1,078,380 tokens (5.21 GiB, 1.03x at 1M), eager, gmu 0.80.

### The model and the problem

`deepseek-ai/DeepSeek-V4.1-Flash`:
- 552B-backbone MoE (769B counting the Engram tables), 16B active decode / 8B prefill, 1M context.
- MXFP4 experts, MXFP8 dense, 510 GB on disk.

It does not fit four GB10s as shipped. The 296 GB of experts split four ways is fine. The problem is the two Engram n-gram tables (203 GB FP8): vLLM keeps them in host memory, and on a DGX Spark host memory *is* the GPU pool.

This repo:
- keeps the Engram tables on disk;
- fixes what that and the GB10's SM 12.1 break in day-0 vLLM;
- gets CUDA graphs working around a host-side lookup.

**Full recipe: [docs/RECIPE.md](docs/RECIPE.md)**

### What had to be fixed

In boot order. Details in `docs/`.

1. **Engram on disk** (`patch/engram.py`, `patch/weight_utils.py`): the two 101 GB tables stay in the safetensors files, and rows are read on demand. Without this it does not fit.
2. **Runtime JIT wedge** (boot 3). A FlashInfer MXFP8 GEMM compiled at runtime with 22 parallel jobs and exhausted host memory on all four nodes, and the watchdogs reset them. Fix: kernels prebuilt in the image (`build/build_overlay5.sh`) plus `MAX_JOBS=2`. [docs](docs/boot3-wedge-and-engram-offset.md)
3. **Engram rank offset** (found in audit). The disk reader ignored each rank's row offset, so ranks 1-3 read rank 0's rows, silently. [docs](docs/boot3-wedge-and-engram-offset.md)
4. **`No common block size for 64`** (boot 4). vLLM picked the smallest listed block size, and the V4 indexer backend refused it. Fix: `--block-size 128`. [docs](docs/boot4-block-size.md)
5. **DeepGEMM `block_kv == 32 or 64`** (boot 5). The ratio-1 indexer cache had 128 states per block. Fix: SM12x indexer pages of 64 states. [docs](docs/boot5-indexer-pages.md)
6. **Relaunch race.** A new worker joined the still-live old head's rendezvous on the same port. Fix: `tools/launch.sh` stops every node first.
7. **`persistent_topk` on GB10** (boot 7, long context). It oversubscribes the 48 SMs, and its fallback needs 128 KB of shared memory per block (GB10 has 99 KB). Fix: `top_k_per_row_decode`, which is also 1.6-3.6x faster there. [results](patch/sm12x-indexer-topk/RESULTS.md)
8. **Eager mode was the throughput ceiling.** About 200 ms per step, host-bound; the GPUs sat near idle. Fix: the Engram lookup moves out of the forward into `prepare_inputs` (`patch/model_state.py`), with all rows read in parallel. The whole decode step is then captured as a CUDA graph, with exact capture sizes so DSpark batches are never padded (FlashInfer #5015).
9. **GPU clock latch** (2 of 4 Sparks). Reddie and Asusi sat at 630-950 MHz with no visible cause. Every TP step waited for them. Fix: unplug the adapter for 30-60 s; a reboot does not clear it. Result: count 41.5 to 60.8 tok/s, code 32.9 to 57.1. [docs](docs/gpu-clock-latch.md)
10. **Engram rows over NFS** (boot 10).
    - Problem: the workers read their Engram rows from the head's NFS export. That took 5.9-7.8 ms per step against 2.8 ms on the head, and every step waited for the slowest rank.
    - Fix: `tools/engram_local.py` copies each worker's rows to local NVMe (47 GiB per worker, a sparse copy at the original offsets, verified row by row).
    - `patch/engram.py` reads the copy only when its recorded row range covers the rank, and falls back to NFS otherwise. `launch/dsv41-tp4.sh` mounts it with `ENGRAM_LOCAL=1`.
    - The same boot fixed `boot_dsv41.sh`, which was not forwarding the patch folder to each node (now `PATCH_NAME`).

### Boot log

| boot | change | outcome |
|---|---|---|
| 1 | overlay1, eager, text-only, Engram on disk | KV 1,989,514. Died in decode warmup: FlashInfer 0.6.18 has no SM120 sparse-MLA decode kernel for `page_block_size=32`. |
| 2 | overlay2 + SWA override | Died at KV init: `No common block size for 32`. |
| 3 | overlay3 (FlashInfer 0.7.0rc1) + SM12x page patches | Wedged all 4 nodes (runtime JIT compile exhausted host memory). |
| 4 | overlay5 (kernels prebuilt) | No runtime JIT; KV 2,026,695. Died at KV init: `No common block size for 64`. |
| 5 | + `--block-size 128`, Engram offset fix, 300K | KV 2,346,690. Died in decode warmup: DeepGEMM `block_kv == 32 or block_kv == 64`. |
| 6 | + indexer pages of 64 states | **Served.** Eager, no speculation: 5.1 tok/s (count). Smoke clean, greedy reference 8/8, garble gate 30/30. |
| 7 | + DSpark k=5 at 1M max context | **Served at 1M.** KV 1,078,380. DSpark eager 19.5-22.1 tok/s. A 32K request then killed it in `persistent_topk` (fix 7). |
| 8 | + CUDA graphs, Engram staged before the forward, top-k fix, gmu 0.78, 300K | Served. Count check 41.5 tok/s, code 32.9 (two GPUs clock-latched, found later). |
| 9 | + tools, vision, gmu 0.80; Reddie and Asusi power-cycled to clear a GPU clock latch | Served. KV 1,032,963 (3.44x at 300K). Count-to-100 60.8 tok/s, code 57.1, count-to-300 68.2. Tool call and image OK. |
| 10 | + node-local Engram rows on the 3 workers (one change); GPU clocks locked by the owner before launch | **Serving.** KV 1,070,168 (3.57x at 300K). Count-to-100 84.9 tok/s, code 65.3 (end to end). Bench: C1 code 73.8, C6 aggregate 131.9. Tool call and image OK. Restored 2026-09-11 after a worker was powered off by accident (`tools/restore_boot10.sh`, 11 min to serving). Same config; KV 1,250,201 this time (graph capture took 1.29 GiB vs 1.99). Warm: count 90-92 tok/s, code 72-73. |

### Known limits and next steps

- **GB10 GPU slow state** ([issue #1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1); help wanted).
  - The GPU switches between a fast and a slow state that `nvidia-smi` does not show. A decode-shaped GEMV runs at 70 vs 230 GB/s.
  - In serving that is 63 vs 94 ms per step. A long idle reliably leaves it slow; it also flips during work, less often.
  - It reproduces with a plain PyTorch script (`tools/gpuflip.py`), so it is not the engine. Network, thermals, CPU placement, ASPM and KV caching are ruled out.
- **Vision and tools.**
  - Both on: up to 4 images per request, with the `deepseek_v41` tool and reasoning parsers.
  - Thinking is off by default; a request can turn it on with `"chat_template_kwargs": {"thinking": true}`.
  - FlashInfer #4973 (vision on SM120) did not reproduce in the 3 image tests, one of which sends two images in one message. Heavier image traffic and large photos are untested.
- **Adaptive verification is off.** It pads speculative batches, and padded batches can hang SM120 sparse MLA (FlashInfer #5015, open).
- **Step time in the fast state:** 63 ms on counting and 67-71 ms on code.
  - Measured parts: 88 all-reduces per step cost about 5 ms (`tools/nccl_lat.py`, stable across the four Sparks). The Engram staging still runs before each forward (2-3 ms with local rows).
  - The rest is the MoE and attention kernels, the next lever.
- **1M long context after the top-k fix.** The top-k fix is validated on GB10 at row widths up to 300,000. A 1M-token request has not been re-run on the fixed stack.
- **The vLLM branch moved under the patches.** `dsv41-feat` was force-pushed on 2026-09-11 after this image was built, then merged into main and deleted the same day. The patches target commit `e47aa780b`; on a later branch head the engine boots cleanly and emits one repeated garbage token (DSpark accepts nothing). Build from the pinned commit (`build/fetch_vllm_branch.sh`, [docs/RECIPE.md](docs/RECIPE.md) step 3).
- **Host hardening** (system settings, not applied here): see [docs/RECIPE.md](docs/RECIPE.md) step 7.

## Repo layout

| path | what |
|---|---|
| `runs/2026-10-17-1017-tensorfold/` | The default lane (TensorFold, TP4, LibertAI EXL3 3.5 bpw): `launch/config.env` (site settings for the recipe's `scripts/serve.sh`), the fleet configs and `launch/chat_template.jinja`; `patches/` (four engine patches, applied with `git am`); `tools/` (`v41bench_tf.py`, `long_check.py`, `inventory_pack.py`, `copy_engram.sh`, `test_chat_template.py`); `results/tf1017a/` (report, bench table, raw JSON and log). |
| `runs/2026-10-18-1017-tensorfold-uncensored/` | The uncensored variant of the TP4 lane: rank-cache patch tools, fleet config and start script, refusal probe, eval comparison. |
| `runs/2026-10-18-1017-tensorfold-tp2/` | The TensorFold TP2 lane (two Sparks, LibertAI EXL3 2.9 bpw): build, per-node inputs, config, launch scripts and results. |
| `runs/` (other folders) | The vLLM lanes' run records: `2026-10-17-1017` (native 1017 weights, boots a-h, 69-scenario eval), `2026-10-17-1017-exl3` (full EXL3 on vLLM), `2026-09-19-speedrun2`, `2026-09-17-nvfp4`, `2026-09-14-speedrun`, `2026-09-14-sglang-lane`. |
| `patch/` | The vLLM lanes' patch set: the exact files bind-mounted over vLLM (md5s in `patch/README.md`), plus one folder per fix with its diff and test. |
| `build/` | Image chain: overlay1 (branch + sm121 extension), overlay3 (FlashInfer 0.7.0rc1), overlay4/5 (prebuilt kernels). |
| `launch/` | The vLLM launchers: `dsv41-tp4.sh <rank>`, `boot_dsv41.sh` (worker-first fan-out), and one `bootN-go.sh` per boot. `boot10-go.sh` is the vLLM boot 10 serving config. The TensorFold lane launches with the recipe's `scripts/serve.sh` and the `config.env` in its run folder. |
| `tools/` | Launch wrapper, pre-launch steps, boot poll, post-serve checks, bench report, `engram_local.py` (node-local Engram rows), `gpuflip.py` / `flipsum.py` (GPU slow-state probe), `nccl_lat.py` (4-node all-reduce check), `idletest.py` (per-step timing after idle vs back to back), `vision_tools_demo.py` (vision and tool-calling checks). |
| `bench/` | Fixed prompt set v1, C1-C6 bench, long-context needle test. |
| `docs/EXL3-TP3.md`, `exl3/` | The EXL3 TP3 lane: write-up and bring-up log, launch and guard scripts (`exl3/try4/`), prep tools, memory evidence. |
| `patch/exl3-tp3/` | The TP3 patch set (mounted over the image like `patch/`), with `MD5SUMS.txt`. |
| `results/exl3tp3a*/` | Per-boot bench, report, comparisons and vision/tool checks for the TP3 lane. |
| `docs/` | Recipe and one post-mortem per failure. |
| `results/` | Head logs of every boot, proofs, bench output, pre-launch GPU and network checks. |
| `CREDITS.md`, `third_party/` | Who and what this recipe builds on, and the verbatim license and notice files of the third-party code included here. |

**Fleet (every lane runs on the same four Sparks; TP2 uses two of them):**
- Reddie is the head: model on local NVMe, exported over NFS.
- Asusi, Bluey and Spark4 are workers, each with a local copy of its Engram rows.
- ConnectX-7 RoCE fabric, 192.168.192.0/24.

**Sister repos:** [DeepSeek-V4-Flash-Vision-Exp (vLLM, 2x/4x Spark)](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark) · [DeepSeek-V4-Flash-Vision (SGLang, 2x Spark)](https://github.com/tonyd2wild/DeepSeek-V4-Flash-Vision-SGLang-DGX-Spark)

**Community ports:** [8x DGX Spark, TP8, 1M context, on dealignai's UNCENSORED-FP8 (im0xMagnus)](https://github.com/im0xMagnus/deepseek-v4.1-flash-uncensored-8x-dgx-spark) -- this recipe carried to eight ranks: the three launcher deltas eight ranks need, the sha pin, TP8 measurements at 300K and 1M.

## Credits

The full list, with links, licenses and what each piece is used for, is in [CREDITS.md](CREDITS.md). Third-party license and notice files are kept verbatim in [`third_party/`](third_party/).

- Default lane:
  - [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), for the engine; bertholomus, for the DeepSeek-V4.1 TP4 fork ([`deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold)), the [recipe and kernel kit](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10), and the Triton-variant fallback patch; Lucas Peters ([@lucas-peters](https://github.com/lucas-peters)), for the indexer alignment patch; ecohash-co, for the `TF_RDMA_GID` approach our third patch follows.
  - [LibertAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw), for the EXL3 3.5 bpw pack of DeepSeek-V4.1-Flash-1017 that the lane serves; [Mia-AiLab](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw), for the DeepSeek-V4.1 EXL3 pack layout that the TensorFold engine reads and the LibertAI pack follows; turboderp, for ExLlamaV3 and the EXL3 format.
- DeepSeek-AI, for DeepSeek-V4.1-Flash and the 1017 update (MIT), including the DSpark draft layers and `encoding.py`, the reference our chat template and encoded bench prompts follow.
- The vLLM team, for the day-0 `dsv41-feat` branch ([vllm-project/vllm#56214](https://github.com/vllm-project/vllm/pull/56214), Yongye Zhu and co-authors).
- FlashInfer, for the SM120 sparse-MLA decode and MXFP8 kernels.
- Jason Cook ([@original-el8](https://github.com/original-el8)) and Luke Alonso ([@lukealonso](https://github.com/lukealonso)), [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x): RoCEnante, the one-shot RoCE all-reduce and all-gather, and its vLLM integration.
- [bot-lab-21](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard), for the EXL3 3.5 bpw Pollard checkpoint, its V4.1 cuda-exl3 overlays, and the tuning ladder and backlog; WestWaters, for the Pollard method.
- [Zeuss5/cuda-exl3](https://github.com/Zeuss5/cuda-exl3) (with @NNNtrance), for the EXL3 plugin and kernels; turboderp, for ExLlamaV3 and the EXL3 format.
- drowzeys (Keys), for the [abliterated `wo_b` overlay](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed) behind the uncensored lanes.
- SGLang, for the engine of the experimental SGLang lane; 0xSero, for the Engram-on-NVMe idea.
- The community fleets and contributors who bisected, verified, measured and reported: ecohash-co, hyudryu, quengh, koldfrontier, gitbisector, im0xMagnus, tmolteno, pavolbiely, croberts012, tsw2k, yunwei37 and @tobymao.
- Kai, for the first Engram-on-disk patch and the SM12x page-size patches.
- The Engram-on-disk idea follows our own PLE-on-disk patch for Qwen3.8-Flash-Next.
- Prior art acknowledged at the idea level: vLLM PR #54129 (`VLLM_PLE_MMAP`, by Trosfy). No code was copied.
