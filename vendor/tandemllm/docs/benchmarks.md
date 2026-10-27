# Benchmarks with tool-eval-bench

[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) scores tool calling over an OpenAI-compatible API: tool selection, arguments, multi-step chains, restraint, error recovery, injection resistance and structured output. It detects TandemLLM by its model owner or `Server` header. This page says how to run it against TandemLLM and which settings a run must name, so that another box gets the same score (OPS-189).

## Install

```bash
uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git
```

The numbers below are from bench 2.8.1.

## Two paths

A benchmark of the model runs the plain engine. The `X-Tandem: raw` header serves it: no turn class, no notes, no effort change, the session monitor off ([server.md](server.md#tandemllm-and-the-conductor)). Run the default path, without the header, only when the conductor itself is under test. The two runs then differ in the conductor alone, and the decision journal shows what it added to each turn.

Without a header the bench, a client kind no profile maps that offers tools, gets the `tools` profile (SRV-621): the raw path's prompt ids, with the loop guards and the tool-call checks acting (D1, D1b, D2, E1, C3, C4). An answer that differs from the raw run's comes from a journaled monitor act or the secret echo mask (SRV-625: a key from a tool result repeated in the answer, `extra.secret_mask`). `--header X-Tandem-Profile=tools` names the profile; `--header X-Tandem-Profile=general` runs the profile the default path served before.

```bash
# the model alone
tool-eval-bench run --short --base-url http://<host>:8000/v1 --model TandemLLM \
  --header X-Tandem=raw --seed 42 --timeout 600
# the conductor in front of it
tool-eval-bench run --short --base-url http://<host>:8000/v1 --model TandemLLM \
  --seed 42 --timeout 600
```

`--short` runs the 15 core scenarios. Leave it out for the full suite, 69 scenarios. `--hardmode` adds category P on top; a score with it is not comparable to one without. The bench's default temperature is 0, so the engine decodes greedily. Keep `--parallel` at 1, or at most the server's `--parallel` slots (`/health`, `parallel.max_running`): a scenario that queues behind another counts the wait as its latency. The bench's default request timeout, 120 s, cuts long reasoning and scores it as a failure, so give `--timeout 600` with thinking on.

`/v1/models` lists the relay backends beside `TandemLLM`, so pass `--model TandemLLM`. `/metrics` needs a bearer token and has none of the counters the bench's speculative-decoding report reads.

## Settings that change the score

| setting | where | what it changes |
|-|-|-|
| weight pack | `/health` `weights.name`, the setup guide's model profile, `ops/use-profile.sh` | the logits: answers on a near tie can differ between packs |
| thinking | the bench's `--no-think` | on by default; off was faster and scored lower in the runs below |
| default effort | `--reasoning-effort` (`REASONING_EFFORT` in `ops/serve.env`, `medium`; the setup guide's thinking effort) | the raw path's effort sentence and how long the model thinks; a client's own word wins, `--backend-kwargs '{"reasoning_effort": "high"}'` |
| profile | `X-Tandem-Profile`, else the profile mapped to the client kind, else `tools_profile` for an unmapped client that offers tools (the bench: `tools`), else `default_profile` | the conductor run only: classes, packages, effort, memory; `tools` adds none of them to the prompt |
| memory | the profile's `memory` block | the conductor run only: a keyless client on a single-user box gets the owner's notes; `general` and `tools` have memory off |
| reasoning format | `reasoning_format` in the request, `--reasoning-format` | where the thinking goes; see below |
| response cache | `--response-cache` | a rerun of an identical request within the hour replays its greedy answer: timings change, answers do not |

Name the pack, the thinking setting, the effort, the profile and the path with every score. `/health` gives the version and the pack in `version`, for example `0.11.4+exl3-6.0`, and the bench records it.

A plain API client gets the thinking in `reasoning_content` and the answer alone in `content` (SRV-245). Up to 0.11.4 the raw path left it inline as `<think>...</think>` in `content`. Scenarios that check the exact answer then failed: valid JSON only (TC-22), a terse answer (TC-24). Since SRV-617 the raw path follows the same rule for an SDK, curl or an unnamed client, as the bench is. Against an older server, send the format yourself:

```bash
--backend-kwargs '{"reasoning_format": "reasoning_content"}'
```

## A batch

`tools/teb_batch.sh` runs rows one at a time against one server and keeps what each ran under; `tools/teb_summary.py` prints the table (OPS-209).

```bash
bash tools/teb_batch.sh --base-url http://<host>:8000/v1 --out results/teb-1020 \
  --health "ssh <host> curl -s http://127.0.0.1:8000/health" \
  --restart "ssh <host> tandem restart" standard
python3 tools/teb_summary.py results/teb-1020 --md --misses
```

- Every row: `--seed 42 --temperature 0 --timeout 600 --no-live --parallel 1`. The batch stops when `/health` reports no slot.
- `standard` is four rows: raw at `xhigh` (`--header X-Tandem=raw --backend-kwargs '{"reasoning_effort":"xhigh"}'`), the conductor with no header, and both again with `--hardmode-only`. `pack` adds `--variant-seed 7` on both paths and, with `--pack DIR`, a held-out scenario pack on both. A row of your own is `label:flags`.
- The conductor rows pin no profile. The server resolves each request by what it offers: `tools` when it offers tools, the default profile when it offers none, as any client meets it. A hard-mode scenario without tools then runs under the default profile. Pinned to `tools` on 2026-10-20, hard mode scored 82 twice and failed TC-88 both times; unpinned it scored 86 and passed TC-88 (n = 1 each; one scenario, the cause not tested). Pin a profile in a row of your own with `--header X-Tandem-Profile=<name>`.
- `batch.knobs` holds the bench's version, the server's version and pack, its slots, `max_len`, the default `max_tokens`, sampling, the prefix and response cache settings, and the output of `--knobs CMD` (the effort default and the slots from `serve.env`, for example).
- Each row keeps its result JSON, log, flags, start and end times and `/health` before and after. A row during which the response cache's hits grew replayed answers of an earlier request; the summary marks it, and its times are not the engine's. `--restart` before the batch empties the cache.

## Results on a DGX Spark

[tool-eval-bench.md](tool-eval-bench.md) has the latest results: every thinking setting on the raw path, four conductor profiles and hard mode.

Qwen3.8-27B in the EXL3 6.0 pack, TandemLLM 0.11.4, default effort `xhigh`, seed 42, temperature 0, timeout 600 s, one scenario at a time. The bench ran on another machine on the LAN. Each row is one run (n = 1).

Core 15:

| path | score | deployability | responsiveness | time | not passed |
|-|-|-|-|-|-|
| raw, thinking on | 100 | 89 | 63 | 102 s | none |
| raw, thinking on, effort `medium` | 97 | 82 | 48 | 117 s | TC-14 partial |
| raw, thinking off | 90 | 81 | 60 | 88 s | TC-03, TC-14 partial |
| conductor, `chat` profile with memory | 93 | 80 | 49 | 187 s | TC-03 |
| conductor, `general` profile | 93 | 79 | 47 | 161 s | TC-03 |

Full suite, 69 scenarios:

| path | score | deployability | responsiveness | time |
|-|-|-|-|-|
| raw, thinking on | 93 | 80 | 51 | 14 min |
| raw, thinking off | 88 | 80 | 60 | 10 min |
| conductor, `chat` profile | 89 | 68 | 20 | 30 min |

The first core row ran on main after SRV-617. The other rows ran on 0.11.4: the full raw row with thinking on sent `reasoning_format: reasoning_content` itself, and the `medium` row got the thinking inline in `content`.

TC-03 asks the model to tell Sarah the meeting moved. The conductor classes the turn as message writing, adds the write-message package, and the model drafts the email instead of looking up the contact and sending it (SRV-615). On the raw path TC-03 sits near a tie: one run of the core 15 missed it, the next passed. On every path the structured-output scenarios TC-65 to TC-69 end partial or failed, and TC-68, schema violation resistance, fails.

The first request of each scenario pays the prefill of the tool prompt, about 4,400 tokens; on the Spark the first token arrives after about 7 s, which the responsiveness score counts. Later turns reuse the prefix.

### 2026-10-19: structured outputs with tools, the `tools` profile, C5 and the secret mask

The settings above hold. The conductor rows ran with no header, so the bench, an unnamed client that offers tools, got the `tools` profile (SRV-621). Both arms ran on the same build, and no answer in the full-suite rows was a response-cache replay.

Full suite, 69 scenarios:

| path | score | deployability | responsiveness | time | not passed |
|-|-|-|-|-|-|
| raw, thinking on | 95 | 83 | 54 | 13.3 min | TC-23, TC-49, TC-58, TC-62, TC-67 partial; TC-68 |
| conductor, `tools` profile | 96 | 83 | 53 | 12.2 min | TC-23, TC-49, TC-62, TC-67 partial; TC-68 |

With `--variant-seed 7` and a held-out pack of 12 scenarios of our own, kept outside the repository, raw scored 96 and the conductor 97. The pack passed 12 of 12 on both paths. Those two runs replayed the answers of unchanged requests from the cache, so their times are not given.

- TC-65, TC-66 and TC-69 pass on both paths: the schema now reaches the engine alongside the tools, and the model's first answer token picks a call or the document (SRV-619).
- TC-58 passes on the conductor only: the server masks the key the file held in the answer (SRV-625).
- TC-51 passed on both paths in this run; C5 (SRV-623) journaled no act in it.
- The graders of TC-23 and TC-49 read "doesn't provide historical data" and "no email will be sent" as failures (OPS-193). TC-67 adds one calculator call the grader counts as unrelated. TC-68 asks for a value the model declines to invent.
