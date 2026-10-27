# Documentation

Each page covers one part of TandemLLM and names the files that implement it. Start with the architecture page, a one-page tour that links to all the others. To install, start with [setup.md](setup.md). The manual, [`tandemllm-manual.pdf`](../manual/tandemllm-manual.pdf), puts these pages and the reference tables in one book.

| page | what it covers |
|-|-|
| [architecture.md](architecture.md) | the 27B and Flash-Next, the byte budget, the decode loop, the layers every model plugs into, a map of the code |
| [quantisation.md](quantisation.md) | the checkpoint's FP8, NVFP4, the clip search, GPTQ from the BF16 release, the quality gate, the three profiles, Flash-Next's EXL3 packs |
| [kernels.md](kernels.md) | row invariance, the 4-bit and 8-bit matrix products, the recurrent-layer kernels, graphs |
| [speculative-decoding.md](speculative-decoding.md) | the block drafter, the lookup drafter, StairCut, the older length router, tree verify through recurrent layers, Flash-Next's MTP drafter |
| [exactness.md](exactness.md) | the promise about the text, the near-tie exception, and the checks behind it |
| [caches.md](caches.md) | the resident prefix, the state store, the persistent suffix store |
| [server.md](server.md) | the OpenAI API, tool calls, structured outputs, streaming, admission |
| [parallel-requests.md](parallel-requests.md) | serving several requests at once (`--parallel N`): switching it on, memory, speed, the text, Flash-Next |
| [dashboard.md](dashboard.md) | the Live activity view, the other views, the API contract |
| [packages.md](packages.md) | the packages the conductor attaches by class: their shape, the first packages, how each is measured |
| [operations.md](operations.md) | running a profile, memory rules, holds, deploys, troubleshooting |
| [measurement.md](measurement.md) | the benchmark row, the teacher-forced bench, noise, the release gate |
| [benchmarks.md](benchmarks.md) | tool-eval-bench against TandemLLM: the raw and conductor paths, the settings a score must name, results on a DGX Spark |
| [tool-eval-bench.md](tool-eval-bench.md) | tool-eval-bench results of 2026-10-19: the raw path at five thinking settings, four conductor profiles, hard mode |
| [adding-a-model.md](adding-a-model.md) | the adapter path a new model takes, the seams, and the tests it must pass |
| [lfm2.5-24b-a2b.md](lfm2.5-24b-a2b.md) | LiquidAI's LFM2.5-24B-A2B: the model, its profiles, its tool-call format, how the engine runs it |
| [nemotron-h.md](nemotron-h.md) | NVIDIA's NemotronH (Nemotron 3.5 Bolt, Super VL, Lightning): the models, the EXL3, GGUF and NVFP4 profiles, LiLiCorr, how the engine runs them |
| [setup.md](setup.md) | a new box from the command line, the dashboard's setup guide, the model registry behind it, and Wren |
| [chat.md](chat.md) | the Chat tab (in beta) and chatd, the service behind it: the sandbox, the permission modes, the known limitations |
| [community-weights.md](community-weights.md) | packs other people published: running one, adding yours with one JSON file, how a maintainer verifies it |
| [memory.md](memory.md) | memory notes: what the model keeps about the user and the work |
| [shadow-reader.md](shadow-reader.md) | the small model that follows each session and answers questions about it |
| [proxy.md](proxy.md) | the dashboard as a proxy in front of other engines and cloud APIs |
| [threat-model.md](threat-model.md) | where untrusted input enters, the parts that matter most for security |
| [roadmap.md](roadmap.md) | what Flash-Next still lacks, parallel requests, other GPUs |

Two more references live next to the code. [server/METRICS.md](../server/METRICS.md) lists every Prometheus metric, and [contract/dashboard-v1/](contract/dashboard-v1/README.md) holds the dashboard API's schemas.

Every number on these pages was measured on one DGX Spark, unless the page calls it arithmetic or an estimate. [measurement.md](measurement.md) says how.

This documentation is licensed under CC BY 4.0 ([LICENSE](LICENSE)). The code is licensed separately (see the [README](../README.md)).
