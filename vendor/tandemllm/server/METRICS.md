# Metrics

The server answers `GET /metrics` in the Prometheus text format. Nothing has to be installed for
it: `prometheus_client` is not on the board and the interpreter there has no pip, so
`server/metrics.py` writes the format out by hand.

```bash
curl -s localhost:8000/metrics | head -40
```

Names carry a `qse_` prefix and follow vLLM's names where the quantity is the same one, so a
dashboard written against vLLM needs a prefix change and not a rewrite. Units are in the name, and
they are seconds and bytes. The contract version is a label of `qse_engine_info`; it is `0.1.0` and
it moves when a name, a label or a unit changes.

The first sections describe contract 0.1.0. Later sections add to it and say what they add; nothing
below them renames or removes a 0.1.0 metric.

## Scraping it

```yaml
# prometheus.yml
scrape_configs:
  - job_name: tandemllm
    scrape_interval: 15s
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

A scrape costs about 180 lines and under a millisecond. Every gauge is read when the scrape asks,
from the server's own objects, so a scrape never holds the engine lock and never waits for a
generation. A gauge whose source raises is left out of the page instead of failing it.

## What each metric is

### Requests

| metric | type | labels | what it counts |
|-|-|-|-|
| `qse_requests_total` | counter | `finish_reason` | generations that ended, by how they ended: `stop`, `length`, `timeout`, `abandoned`, `error` |
| `qse_request_success_total` | counter | | the `stop` and `length` rows of the line above |
| `qse_requests_refused_total` | counter | `reason` | requests turned away with a `Retry-After` before the engine saw them: `queue` (the queue was full or the wait timed out), `memory` (OPS-47: the prefill would have taken the engine over `MEM_PEAK_GIB` with every cache given back) |
| `qse_requests_running` | gauge | | generations the engine is working on. This engine holds one sequence, so it is 0 or 1 |
| `qse_requests_waiting` | gauge | | callers holding a socket and waiting for the engine lock |
| `qse_prompt_tokens_total` | counter | | prompt tokens accepted, tokens served out of a cache included |
| `qse_generation_tokens_total` | counter | | tokens the engine wrote |
| `qse_errors_total` | counter | `type` | exceptions that ended a generation, by exception class |
| `qse_images_left_out_total` | counter | | a long conversation's earlier images left out of a request over `--max-images` or `--max-image-rows` (0.3.21-test, docs/server.md, Images) |

`finish_reason` is the server's own set and it is worth reading carefully. `timeout` means the
wall clock in `--request-timeout` ran out and `length` means the caller's token budget did;
`abandoned` means the client hung up, which is not an error and is not counted as one.

### Latency

| metric | type | what it measures |
|-|-|-|
| `qse_time_to_first_token_seconds` | histogram | request arrival to its first token, queue wait included |
| `qse_time_per_output_token_seconds` | histogram | the gap between two tokens of one generation |
| `qse_e2e_request_latency_seconds` | histogram | request arrival to the end of its stream |

Read the inter-token histogram knowing what produces it. A verified block writes several tokens
into the socket at once, so about half of these gaps are microseconds and the other half are a
whole block, and the median is not a rate. The atlas row reports
`(completion_tokens - 1) / (e2e - ttft)` per request for that reason, and
`rate(qse_generation_tokens_total[1m])` is the closest thing here to the same number.

Buckets are chosen from what this engine does rather than copied: 761 ms is the median time to
first token on a 256-token prompt, 526 ms is a warm one on a 1,724-token prompt out of the prefix
cache, and 2.0 ms is a response-cache hit.

### Speculation

| metric | type | labels | what it counts |
|-|-|-|-|
| `qse_spec_decode_num_drafts_total` | counter | | blocks the drafter proposed |
| `qse_spec_decode_num_draft_tokens_total` | counter | | tokens proposed, the anchor excluded |
| `qse_spec_decode_num_accepted_tokens_total` | counter | | drafted tokens the target agreed with |
| `qse_spec_accept_per_block` | histogram | | tokens one block committed, the target's own token included |
| `qse_draft_width_chosen_total` | counter | `width` | blocks proposed at each width, the anchor included |
| `qse_verify_seconds` | histogram | | one verify pass, as the decode loop paid for it |
| `qse_draft_seconds` | histogram | | one call to the drafter, lookup and block drafter together |

The draft acceptance rate is `accepted / draft_tokens`. Tokens per block is
`qse_spec_accept_per_block_sum / _count`, and it is the number the whole engine is tuned on: the
row moves when it moves. On the five-workload bench at width 16 it reads 2.60 on fresh prose and
14.86 on a quotation, which is the spread a single average hides.

`width` is 8 or 16 on this build, counting the anchor: the length router holds two drafters, one
fine-tuned at each block length, and picks one. A run that shows `width="8"` on most blocks is the
latch deciding the narrow arm wins, which happens on fresh prose and nowhere else.

`qse_verify_seconds` is the same measurement `engine/lenrouter.py` prices its own decisions on. It
comes from the loop that pays for it rather than from a profiler, because
`tools/profile_block.py` once read a rollback at 23.4 ms that the serving loop reads at 6.4.

### Caches

| metric | type | labels | what it counts |
|-|-|-|-|
| `qse_cache_hits_total` | counter | `cache` | prefix lookups answered from a cache |
| `qse_cache_misses_total` | counter | `cache` | lookups that found nothing |
| `qse_cache_evictions_total` | counter | `cache` | entries dropped to stay inside the byte budget |
| `qse_cache_collisions_total` | counter | `cache` | stored prefixes whose 64-bit hash matched and whose tokens did not |
| `qse_cache_bytes` | gauge | `cache` | bytes held |
| `qse_cache_entries` | gauge | `cache` | entries held |
| `qse_prefill_tokens_reused_total` | counter | | prompt tokens restored from a snapshot |
| `qse_prefill_tokens_forwarded_total` | counter | | prompt tokens run through the 64 layers |
| `qse_memory_bytes` | gauge | `part` | OPS-47: the engine against its memory limits. `allocated` and `reserve` (the allocator's free blocks), the caches the policy caps (`resident_anchors`, `resident_stash`, `resident_pin`, `resident_park`, `state_store`), the limits themselves (`ceiling`, `peak`), and `prefill_peak`, the largest allocator peak a prefill has reached. The decision models are left out: they are roles with claims of their own |
| `qse_memory_shed_total` | counter | `level` | caches given back to keep the engine under its ceiling or its declared peak, by level, cheapest first: `store`, `anchors`, `stash`, `park`, `pin` |
| `qse_disk_cache_bytes` | gauge | `part` | C1, the disk tier: `entries` (what its files hold), `budget`, `free` (the file system's free space), `floor` (the free space it never takes it under, at least 50 GB), `staging` (the host buffer its reads and writes go through, 0 until first used) |
| `qse_disk_cache_entries` | gauge | `kind` | `anchors` (one recurrent state each) and `chunks` (1,024 tokens of rows each) |
| `qse_disk_cache_lookups_total` | counter | | prefills the tier could have shortened by a chunk or more past what the caches in RAM offered |
| `qse_disk_cache_hits_total` | counter | | prefills that started from the disk |
| `qse_disk_cache_misses_total` | counter | | lookups it did not answer (no entry, a chain not whole, a read dearer than the prefill it saves, a token that differs) |
| `qse_disk_cache_tokens_restored_total` | counter | | prompt tokens read back instead of prefilled |
| `qse_disk_cache_read_bytes_total`, `qse_disk_cache_written_bytes_total` | counter | | the tier's reads and writes on the NVMe |
| `qse_disk_cache_writes_total` | counter | `kind` | files written, `chunk` or `anchor` |
| `qse_disk_cache_evictions_total` | counter | `reason` | entries given up: `budget`, `floor`, `ttl`, `orphan` (chunks no anchor reads) |
| `qse_disk_cache_dropped_total` | counter | `reason` | writes not done: `window` (the next prefill came first), `budget`, `floor`, `staging`, `aborted` (the client left), `too_big` |
| `qse_disk_cache_failed_restores_total` | counter | | restores that failed a check (a checksum, another layout) and took the cold path |
| `qse_disk_cache_last_restore_seconds` | gauge | | the last restore's wall time |

`cache` is `state`, `response` or `suffix`, and one thing about that label needs saying. The
session cache and the prefix cache are **one object**: `engine/cache.py::StateStore` holds a
snapshot per token boundary and a session boundary is a boundary like any other, so the store
cannot tell you which of the two a hit belonged to and neither can this page. The pair of numbers
that answers the question the label was meant to answer is
`qse_prefill_tokens_reused_total` against `qse_prefill_tokens_forwarded_total`: on a shared system
prompt of 1,724 tokens the warm request reuses 1,536 and forwards 188.

The suffix store reports its size and nothing else. Its lookups happen inside the drafter, tens of
thousands of times per generation, and counting them would cost more than they cost.

A cache that is switched off reports no rows at all rather than zeroes, so a panel that goes blank
is telling you the flag is off.

### The conductor

SRV-115: every decision of the TandemLLM conductor, the router's and the session monitor's, passes included. With the conductor raw (the default) nothing is counted and the families stay at 0. A monitor decision is counted when it is journaled; the checks that found nothing are counted at the end of their request.

| name | type | labels | what |
|-|-|-|-|
| `qse_conductor_decisions_total` | counter | `task`, `tier`, `mode`, `decision` | every decision: the monitor's tasks (`D1`, `E1`, `C3`, ...; `SP3` is an answer whose calls all passed) with `mode` act or shadow and `decision` the action in lower case or `pass`; the router's class with `task="route"`, `mode` the profile's (on, shadow) and `decision` the class id |
| `qse_conductor_package_kept_out_total` | counter | `why`, `mode` | SRV-615: a package with a `deliverable` the router kept out of a turn because an offered tool delivers its result (`why` `tool_delivers`); `mode` act, or shadow under `tool_fit.deliverable: shadow` |
| `qse_conductor_actions_total` | counter | `task`, `action` | decisions executed: `CLOSE_REASONING`, `CONTINUE_HIDDEN`, `REGENERATE_CALL`, `REPAIR_CALL`, `GIVE_UP`, `PROMPT_TAIL_NOTE` (a router note, `task="route"`), `SET_THINK_BUDGET` |
| `qse_conductor_hidden_tokens_total` | counter | | tokens the model saw that the client did not send or receive: hidden closes, hidden turns, reminders, router notes new and replayed |
| `qse_conductor_gave_up_total` | counter | | turns ended with the GIVE_UP sentence |
| `qse_conductor_tier1_seconds` | histogram | | one tier 1 question (a classifier stage: the decider, Qwen), answered or not; a cached answer is not a question |
| `qse_conductor_tier1_failures_total` | counter | `reason` | tier 1 questions without a usable answer: `decider_timeout`, `decider_unreachable`, `qwen_unreachable`, ... |
| `qse_conductor_reasoning_tokens_cut_total` | counter | | reasoning tokens written before a hidden close cut the block: the size of the blocks the conductor closed |

`rate(qse_conductor_actions_total[1h])` by task is the Overview's ribbon; `qse_conductor_gave_up_total` going up overnight is the one to alert on. The rows behind every count are in the journal (`GET /v1/conductor/events`).

### The board

| metric | type | what it reads |
|-|-|-|
| `qse_gpu_memory_used_bytes` | gauge | `torch.cuda.memory_allocated()`, what the engine holds |
| `qse_gpu_memory_reserved_bytes` | gauge | what the allocator took from the driver and has not given back |
| `qse_unified_memory_free_bytes` | gauge | free bytes in the board's one pool |
| `qse_engine_uptime_seconds` | gauge | seconds since the server started |
| `qse_engine_info` | gauge | always 1; the labels are the configuration |

GB10 shares one memory pool between the GPU and the host and `nvidia-smi` reports N/A for used
memory, so these three are the only honest numbers about memory on this board. A leak shows as
`used` climbing across requests; fragmentation shows as `reserved` climbing while `used` does not.

`qse_engine_info` carries `version`, `model`, `drafter`, `width`, `tree`, `nvfp4`, `fp8_head`,
`max_len` and `caches`. Join on it when you compare two runs, because a tok/s number without the
weight set it was produced by is not comparable to anything.

## An example scrape

From a run of the atlas row, with the histogram buckets cut down to the interesting ones:

```
# HELP qse_requests_total generations that finished, by the reason they finished
# TYPE qse_requests_total counter
qse_requests_total{finish_reason="abandoned"} 1
qse_requests_total{finish_reason="length"} 50
# TYPE qse_request_success_total counter
qse_request_success_total 50
# TYPE qse_generation_tokens_total counter
qse_generation_tokens_total 12800
# TYPE qse_time_to_first_token_seconds histogram
qse_time_to_first_token_seconds_bucket{le="0.75"} 12
qse_time_to_first_token_seconds_bucket{le="1"} 50
qse_time_to_first_token_seconds_bucket{le="+Inf"} 50
qse_time_to_first_token_seconds_sum 38.07
qse_time_to_first_token_seconds_count 50
# TYPE qse_spec_decode_num_accepted_tokens_total counter
qse_spec_decode_num_accepted_tokens_total 6338
# TYPE qse_spec_decode_num_draft_tokens_total counter
qse_spec_decode_num_draft_tokens_total 24000
# TYPE qse_spec_decode_num_drafts_total counter
qse_spec_decode_num_drafts_total 1600
# TYPE qse_spec_accept_per_block histogram
qse_spec_accept_per_block_bucket{le="1"} 174
qse_spec_accept_per_block_bucket{le="4"} 1022
qse_spec_accept_per_block_bucket{le="16"} 1600
qse_spec_accept_per_block_bucket{le="+Inf"} 1600
qse_spec_accept_per_block_sum 7938
qse_spec_accept_per_block_count 1600
# TYPE qse_draft_width_chosen_total counter
qse_draft_width_chosen_total{width="16"} 1600
# TYPE qse_verify_seconds histogram
qse_verify_seconds_bucket{le="0.1"} 898
qse_verify_seconds_bucket{le="0.11"} 1600
qse_verify_seconds_sum 159.43
qse_verify_seconds_count 1600
# TYPE qse_requests_running gauge
qse_requests_running 1
# TYPE qse_requests_waiting gauge
qse_requests_waiting 0
# TYPE qse_cache_bytes gauge
qse_cache_bytes{cache="state"} 4541000000
qse_cache_bytes{cache="suffix"} 943052
# TYPE qse_prefill_tokens_reused_total counter
qse_prefill_tokens_reused_total 1536
# TYPE qse_prefill_tokens_forwarded_total counter
qse_prefill_tokens_forwarded_total 188
# TYPE qse_engine_info gauge
qse_engine_info{version="0.1.0",model="qwen38-spark-engine",drafter="LengthRouter",width="15",tree="1",nvfp4="mlp-clip,gdn-clip,attn-clip",fp8_head="1",max_len="32768",caches="session,prefix,suffix"} 1
```

Reading it: 1,600 blocks proposed 24,000 tokens and the target kept 6,338 of them, a draft
acceptance of 26.4 %, and 7,938 tokens came out of 1,600 blocks, so a block committed 4.96. The
verify sum over its count is 99.6 ms, which is the width-16 verify the length router is priced
against.

## Panels for a dashboard

Twelve panels, in the order they answer a question. Every query is PromQL against the names above.

#### 1. Tokens a second, served

The headline.
```promql
rate(qse_generation_tokens_total[1m])
```

#### 2. Tokens per block

The one number the whole engine is tuned on.
```promql
rate(qse_spec_accept_per_block_sum[5m]) / rate(qse_spec_accept_per_block_count[5m])
```

#### 3. Draft acceptance

Falls before tokens per block does, and it is the leading indicator of a
drafter that has lost the workload.
```promql
rate(qse_spec_decode_num_accepted_tokens_total[5m])
  / rate(qse_spec_decode_num_draft_tokens_total[5m])
```

#### 4. The block, split

Two series on one axis, in milliseconds. Their sum is the block, and the
gap between that sum and the reciprocal of panel 1 is the commit, the sync and the idle.
```promql
1000 * rate(qse_verify_seconds_sum[5m]) / rate(qse_verify_seconds_count[5m])
1000 * rate(qse_draft_seconds_sum[5m]) / rate(qse_draft_seconds_count[5m])
```

#### 5. Time to first token, p50 and p99

```promql
histogram_quantile(0.5,  sum by (le) (rate(qse_time_to_first_token_seconds_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(qse_time_to_first_token_seconds_bucket[5m])))
```

#### 6. Queue depth

`qse_requests_waiting`, as a graph rather than a number. A server that is
never above 0 has no queueing problem whatever its rate looks like.

#### 7. Refusals a minute

`rate(qse_requests_refused_total[5m])`. Anything above zero means
callers are being sent away, and the queue bound is `--max-queue`, 8 by default.

#### 8. Finish reasons

`sum by (finish_reason) (rate(qse_requests_total[5m]))`, stacked.
`length` climbing means callers are hitting a token cap; `timeout` climbing means generations are
running past `--request-timeout`.

#### 9. Errors

`sum by (type) (rate(qse_errors_total[5m]))`. Alert on any of it.

#### 10. Prompt tokens reused

The caches, as the one ratio that says what they bought.
```promql
rate(qse_prefill_tokens_reused_total[10m])
  / (rate(qse_prefill_tokens_reused_total[10m]) + rate(qse_prefill_tokens_forwarded_total[10m]))
```

#### 11. Cache bytes against the budget

`qse_cache_bytes` by `cache`, with
`--cache-budget-gb` drawn as a threshold. Watch `qse_cache_evictions_total` beside it: eviction
starting is the moment the budget became the binding constraint.

#### 12. Memory

`qse_gpu_memory_used_bytes` and `qse_gpu_memory_reserved_bytes` on one axis,
`qse_unified_memory_free_bytes` on another. This is the soak-test panel: used climbing across
requests is a leak, reserved climbing while used does not is fragmentation.

Two alerts are worth having before any dashboard: `rate(qse_errors_total[5m]) > 0`, and
`qse_requests_waiting > 6` for five minutes, which says the queue is about to start refusing.

## What is deliberately not here

- **Batch size, KV utilisation, preemptions, swaps.** vLLM reports them because it has a
  scheduler. This engine holds one sequence and has none of those states.
  [docs/roadmap.md](../docs/roadmap.md) says what would have to be built first, and it adds the
  metrics it would need.
- **Per-model and per-user labels.** One process serves one model, and a user label on a counter
  is personal data on a monitoring endpoint. The label set here has a fixed, small cardinality:
  five finish reasons, two widths, three caches, and one exception class per distinct failure.
- **Rollbacks.** The loop rolls back on a partial accept and the count follows from
  `drafts - (blocks that accepted everything)`, which no counter here can separate without a hook
  inside the accept loop. It is in `engine/spec.py::DecodeStats` for the bench, where it belongs.
- **A pull of the drafter's own report.** `engine/lenrouter.py::report()` prints the latch, the
  ceiling and the calibration factor once a generation, and those are per-request beliefs rather
  than server state. They are in the log line, not on this page.

## Where the hooks are, and why they are not in the loop

`server/metrics.py` wraps four things and `server/app.py` carries an import, a route and one
`install()` call:

| wrapped | gives |
|-|-|
| `generate_stream` | time to first token, the gaps between tokens |
| `_log_request` | finish reasons, token counts, end-to-end latency, errors |
| `Handler._complete` | the arrival time, so the queue wait is inside time to first token |
| the drafter's `propose` / `propose_tree` / `observe` / `on_verify` | every speculation counter |

The alternative was a counter next to each of those calls inside the generation loop. Phase 9 was
editing that loop for three serving bugs at the same time this was written, and a hook scattered
through it would have collided with every one of those edits. The wrappers cost two
`perf_counter()` calls and a few dictionary increments per block, against a block of 133 ms.

`tests/test_metrics.py` is 26 tests, no torch and no board. Half of them are the exposition format,
parsed the way a scraper parses it; the other half are the semantics, and the one worth knowing
about asserts that an `observe` that follows no proposal is not counted as a block. The loop calls
`observe` after a prefill, after a step the drafter declined, and for the tokens a reasoning budget
forces, and counting those would put the accepted-tokens counter above the drafted one.

## Per-response usage and timings

Not a change to this page's names: the same numbers now travel with every response, so a client
sees them without a scrape. Open WebUI prints the `usage` of a stream merged with its llama.cpp
`timings` in its (i) tooltip, and adds the token counts of every chunk that carries `usage`,
so exactly one chunk of a stream carries them (the finish chunk by default, the separate
`choices: []` chunk when the client sent `include_usage: true`, none on `include_usage: false` or
`--usage-default off`). `server/usage.py::RequestRecord` is the one source; the example below has
consistent numbers.

```json
"usage":   {"prompt_tokens": 1959, "completion_tokens": 412, "total_tokens": 2371,
            "prompt_tokens_details": {"cached_tokens": 1536},
            "completion_tokens_details": {"reasoning_tokens": 230}},
"timings": {"cache_n": 1536, "prompt_n": 423, "prompt_ms": 2239.7, "prompt_per_token_ms": 5.29,
            "prompt_per_second": 188.86, "predicted_n": 412, "predicted_ms": 6021.0,
            "predicted_per_token_ms": 14.65, "predicted_per_second": 68.26, "draft_n": 1350,
            "draft_n_accepted": 322, "ttft_ms": 2240.1, "queue_ms": 0.4, "total_ms": 8261.1,
            "blocks": 90, "tokens_per_block": 4.57, "reasoning_n": 230, "cache_source": "prefix"},
"metrics": {"time_to_first_token_ms": 2240.1, "generation_time_ms": 6021.0, "queue_time_ms": 0.4,
            "mean_itl_ms": 14.65, "tokens_per_second": 49.87,
            "speculative_decoding": {"mean_acceptance_length": 4.57, "draft_acceptance_rate": 0.2385}}
```

| field | definition |
|-|-|
| `prompt_tokens` | the templated prompt, cached part included |
| `cached_tokens`, `cache_n` | prompt tokens restored from the state store; all of them on a response-cache hit |
| `prompt_n` | prompt tokens forwarded through the 64 layers (0 on a response-cache hit) |
| `completion_tokens`, `predicted_n` | ids the engine committed, the EOS included; a drafted-and-rejected token is never one |
| `reasoning_tokens`, `reasoning_n` | committed ids through the one that closes the reasoning block (the special id or the literal text); all of them when it never closed; 0 with thinking off |
| `queue_ms` | arrival to the engine lock |
| `prompt_ms` | the lock to the first token: template, tokenise, prefill. `ttft_ms = queue_ms + prompt_ms` |
| `predicted_ms` | first token to last token |
| `predicted_per_second` | `(completion_tokens - 1) / predicted_ms`; llama.cpp divides by `predicted_n`; the first token is the prefill's |
| `total_ms` | arrival to the `[req]` line, which is logged just before the finish chunk is written |
| `blocks`, `tokens_per_block` | forwards the decode loop paid (`BlockStats`), and `(completion_tokens - 1) / blocks` |
| `draft_n`, `draft_n_accepted` | sums over the request's first-miss histogram: tokens proposed and kept |
| `cache_source` | `response`, `session`, `prefix` or `none` |

Floats are rounded to two decimals, the acceptance rate to four. A failed stream's finish chunk
carries the partial counts beside its `error` object; an abandoned one is only in the log.

## Contract 0.2.0

`qse_engine_info{version="0.2.0"}`. Every 0.1.0 name is still here with its meaning; what is new
comes from the request's `RequestRecord` once the request is over (`metrics.on_record`, called from
`server/app.py::_account`), from `Handler.send_response`, or is read at scrape time. No new hook
is in the decode loop, and no label carries a user, a client, a model path or a request: per-client
numbers are the usage ledger's.

| metric | type | labels | what it is |
|-|-|-|-|
| `qse_http_requests_total` | counter | `route`, `code` | responses, by a fixed route set: `chat`, `completions`, `models`, `health`, `metrics`, `cache`, `dashboard`, `static`, `other`, never the raw path |
| `qse_requests_total` | counter | `finish_reason` | gains `refused` (503/429 before the engine) beside `tool_calls`, which it already counted |
| `qse_prompt_tokens_cached_total` | counter | | prompt tokens restored from the state store (all of them on a response-cache replay) |
| `qse_reasoning_tokens_total` | counter | | committed tokens inside the reasoning block |
| `qse_cache_hits_total` | counter | `cache`, `kind` | the state store's hits now split by where the snapshot came from: `kind="session"` (a turn's end) or `kind="prefix"` (a prefill chunk boundary). The response cache's row keeps `cache` only. `sum by (cache)` reads as before |
| `qse_suffix_store_drafts_total` | counter | `outcome` | proposed blocks for which the persistent suffix store matched the context (`hit`) or not (`miss`), counted once a block: the store bumps a counter on a match and the drafter wrapper reads it around each proposal. METRICS.md's objection to per-lookup counting stands; this is not that |
| `qse_usage_ledger_rows_total` | counter | | rows the usage ledger wrote |
| `qse_usage_ledger_dropped_total` | counter | | rows it lost: a full queue or a failed write |
| `qse_request_queue_seconds` | histogram | | arrival to the engine lock |
| `qse_request_prefill_seconds` | histogram | | the lock to the first token (template, tokenise, prefill) |
| `qse_request_decode_tokens_per_second` | histogram | | per decoded request, `(completion - 1) / (last token - first token)`, a response's `predicted_per_second`. Not observed for errors, refusals or response-cache replays |
| `qse_request_prefill_tokens_per_second` | histogram | | forwarded prompt tokens over the prefill seconds |
| `qse_request_prompt_tokens`, `qse_request_completion_tokens` | histogram | | sizes per request |
| `qse_suffix_store_tokens` | gauge | | token ids in the persistent suffix store |
| `qse_state_store_budget_bytes` | gauge | | the state cache's budget, to draw `qse_cache_bytes` against |
| `qse_process_resident_memory_bytes` | gauge | | the server's RSS |
| `qse_engine_start_time_seconds` | gauge | | unix time the server started (a restart is a step in it) |
| `qse_log_subscribers` | gauge | | open `/v1/dashboard/logs` streams |
| `qse_build_info` | gauge | `version`, `git_sha`, `code_sha256`, `flags_sha256` | always 1. `code_sha256` is `tools/row3.py::code_hash` of the served directory, the hash row reports record; `flags_sha256` is over the effective flags and every `QWEN38_*` variable. The values themselves are in `/v1/dashboard/system` |

Buckets are this engine's: the queue times out at 240 s; a prefill is ~0.4 s at 256 tokens and
minutes at 256k; decode is 30-70 tok/s on new text and 100+ on a quotation.

### Three more panels

#### 13. Decode tok/s per request, p50 and p90

The per-request view of panel 1: panel 1 is tokens a second
SERVED (idle time included), this is how fast a request decoded once it was decoding.
```promql
histogram_quantile(0.5, sum by (le) (rate(qse_request_decode_tokens_per_second_bucket[15m])))
histogram_quantile(0.9, sum by (le) (rate(qse_request_decode_tokens_per_second_bucket[15m])))
```

#### 14. Reasoning share

Of the tokens written, how many were thinking.
```promql
rate(qse_reasoning_tokens_total[1h]) / rate(qse_generation_tokens_total[1h])
```

#### 15. Cached share of the prompt

What the caches saved, as a fraction of every prompt token accepted.
```promql
rate(qse_prompt_tokens_cached_total[1h]) / rate(qse_prompt_tokens_total[1h])
```

Two alerts join the two above (`ops/monitoring/` has the rules): `increase(qse_usage_ledger_dropped_total[15m])
> 0`, and the scrape itself (`up == 0` outside a benchmark hold).

Access: through a reverse proxy, `/metrics` needs `QSE_METRICS_TOKEN` (or the admin
token or the dashboard session); a tool on the box itself, on loopback with no proxy header, needs
nothing. `GET /metrics/up` is public and carries one series, `qse_up` (1, or 0 while draining), so a
scrape can tell an engine that is down from a token that is wrong (the two scrape jobs in
`ops/monitoring/qse-engine-scrape.yaml`).
