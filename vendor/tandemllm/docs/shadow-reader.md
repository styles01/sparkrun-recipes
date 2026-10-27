# The shadow reader

A small Qwen3.5 model keeps its own state of each session: its KV rows and its Gated-DeltaNet state for every token the session holds, the served prompt and Qwen's output. It reads each turn's new tokens and nothing else, so a small model can answer a question about a 100,000-token session in one forward of the question.

Code: `engine/shadow.py` (one session's state, the grid, forks, the disk), `server/conductor/shadow.py` (the service: feeding, scheduling, the cap, the questions), the wiring in `server/app.py`. Tests: `tests/test_shadow.py`. Measurements: `tools/shadow_measure.py`, `tools/shadow_fresh.py`, `tools/shadow_ab.py`.

## Why a shadow and not a bridge

Two cheaper ways to give a small model Qwen's context failed on 2026-10-04. Sixteen soft tokens made from Qwen3.8-27B's hidden rows (B2) answered at chance, 0.13 against 0.13. A per-layer map from the 27B's KV and GDN state into Qwen3.5-4B's (B3) reached a held-out R² of about 0.5, and its own oracle needs about 0.95 for 90 % agreement. The oracle also showed what does work: the small model's own state of the same tokens, which is exact.

The Qwen3.5 small models (0.8B, 2B, 4B) use the same BPE vocabulary and merges as Qwen3.8-27B. Qwen's token ids pass to the small model as they are, with no detokenising. The engine's own Qwen3.5 forward runs them (`engine/generative/qwen35.py`); the reader is a second engine object over the same weights.

## One session's state

| part | 0.8B | 2B | 4B |
|-|-|-|-|
| KV a token (6 or 8 attention layers) | 12 KiB | 12 KiB | 32 KiB |
| recurrent and convolution state | 19.5 MB | 19.5 MB | about 50 MB |
| a 41,472-token session (measured) | 0.60 GiB | 0.60 GiB | |
| a 57,856-token session (measured) | 0.88 GiB | | |

The KV buffer grows in steps of 8,192 rows. A session also keeps up to six checkpoints of its recurrent state (19.5 MB each on the 0.8B), one at the end of each turn's prompt.

## Exact on a grid

The state advances only in whole chunks of 512 rows at absolute positions k × 512. A fresh prefill of the same ids on the same grid runs the same chunks on the same inputs, so the two states are equal bit for bit, however many turns it took. The ids past the last grid line (fewer than 512) wait as the pending tail. A question prefills the tail and its own tokens above the committed rows and puts the recurrent state back afterwards (a fork); nothing below the committed length changes.

Measured on the GPU (2026-10-04, 0.8B, two real conversations of 41,802 and 58,278 tokens): the replayed state equals a fresh prefill on the same grid in every KV row, the recurrent state and the convolution state. Against a fresh prefill on a 2,048-row grid it is not bit-equal (largest difference 0.44 in a K row, 0.018 in the recurrent state, the rounding of another chunking), and 8 of 8 question probes give the same top token, KL at most 4.1e-5.

When a new prompt differs from what was read (the chat template drops an earlier turn's reasoning once a new user message comes), the reader goes back to the newest checkpoint at or below the first differing token and reads from there. Without one it starts over. KV rows need no checkpoint: rows below the cut stay valid.

## When it reads

The bandwidth rule: while Qwen decodes, only Qwen and its drafters use the GPU. The reader has two windows.

- The idle window. `shadow_once` in `server/app.py` looks every 0.2 s. When no request runs or waits and no hold or box lock is on, it takes the engine lock behind every waiter (rank 4, after the summary written ahead) and the box lock without waiting, and reads for at most 2 s. Before every chunk it checks whether a request waits, so a request that arrives waits at most one chunk: 36 to 50 ms on the 0.8B at grid 512.
- Inside a turn, rationed. Between two of Qwen's rounds, on the request's thread, the reader may use 2 % of the time since Qwen's first token, in slices of at most 150 ms (whole chunks, the next one's cost estimated from the last). The device is synchronised first, so nothing of the small model runs beside Qwen. It reads the prompt first, then Qwen's output so far. The share counts from the first token on purpose: the first build counted it from the turn's start, so a 17 s prefill earned about 340 ms that the reader then spent inside a 9.6 s decode, and the decode lost 5.3 %.

The idle window alone is not enough for an agent. On the usage ledger's 41 opencode sessions (1,624 turns, 2026-10-02 to 10-04) the median gap between the end of one request and the start of the next is 0.31 s: opencode runs the tool and sends the next request at once. Replaying those timelines with the measured read rate (87 ms per 1,000 tokens), the idle window alone leaves the shadow fresh (less than one grid chunk behind) at 41 % of turn starts, p90 49,000 tokens behind. With 2 % of each turn's decode time on top it is fresh at 84 % of turn starts, p50 307 and p90 4,347 tokens behind. A session's first long prompt is the exception: 40,000 tokens take about 3.5 s to read.

## What it costs Qwen

`tools/shadow_ab.py` in a hold (2026-10-04, a candidate server on :8011, the served engine flags, one 16,350-token prompt, 346 tokens out, a new session each run so the shadow reads the whole prompt each time, 1.5 s between requests, the arms alternating). Qwen's decode rate, median of 6 runs an arm: shadow off 36.02 tok/s, shadow on 35.28 tok/s, -2.05 %; paired run by run the deltas are -2.2, -2.0, -1.9, -1.8, -2.2 and +0.5 %. That is the 2 % share, as designed. TTFT did not move (17.46 s off, 17.40 s on): the idle bursts stop before the next chunk once a request waits. The gate was 3 %.

## Read rate

`tools/shadow_measure.py`, the served process's arithmetic flags, cold weights excluded, ms per grid chunk (median of 3):

| model | grid | depth 0 | depth 16k | depth 41k |
|-|-|-|-|-|
| 0.8B | 256 | 22.9 | 24.4 | 30.1 |
| 0.8B | 512 | 36.4 | 41.1 | 49.9 |
| 0.8B | 1,024 | 63.1 | 74.5 | 94.0 |
| 0.8B | 2,048 | 126.8 | 149.5 | 186.9 |
| 2B | 512 | 50.5 | 56.1 | 69.1 (33k) |

At grid 512 that is 71 to 97 ms per 1,000 tokens on the 0.8B and 99 to 135 ms on the 2B. A whole 58k conversation read in one go took 5.1 s. A larger grid reads faster per token but makes a waiting request wait longer; 512 is the default (`QWEN38_SHADOW_GRID`).

## Who asks

- `ask(session, question, choices)`, also `POST /v1/conductor/shadow/ask`: a typed decision of the small model over the whole session. Each choice is a letter; one forward of the question gives the letters' softmax, as the router's in-context stage asks Qwen (`server/conductor/incontext.py`). At the end of a 41k session the router's 9-class question (290 tokens plus a 330-token pending tail) took 68 ms on the 0.8B and 96 ms on the 2B.
- The router's class question, in shadow mode (`shadow.router: shadow`). For a turn whose prompt has at least `router_min_tokens` tokens (8,192 by default), the question the router's in-context stage asks Qwen goes to the shadow at the turn's insertion point (the prompt's last `<|im_end|>`). It is answered when the reader reaches that point and journaled in `<conductor root>/shadow.jsonl` beside the router's own class: the label, its probability, agreement, how far behind the shadow was when the turn began, and the lag. Nothing of the turn changes.
- The narrator's thought summary (`shadow.narrator: on` with `ui.narrator: thoughts`). The narrator's job reads what the shadow misses up to the end of the thought, then asks for the summary on top of the whole session instead of the last 1,500 ids of the thought. It runs in the narrator's slices (at most 3 % of Qwen's time). The fork decodes eagerly: 12 to 17 ms a token on the 0.8B (the narrator's own path has a CUDA graph at about 9.5 ms).

## Memory and the disk

Resident states are capped at `QWEN38_SHADOW_GB` (4 GiB). Past the cap, the least recently used session goes to the NVMe (`QWEN38_SHADOW_DISK`, one safetensors file a session, 48 h, at most `QWEN38_SHADOW_DISK_GB`) and comes back exact when the session is fed again; without a disk it is dropped and read again from the start. A planned stop writes every resident session after the work in flight. A file records the weights' hash and the grid; another model or grid does not read it.

The model loads through the residency planner (role `shadow`: the model's claim plus `QWEN38_SHADOW_GB`): its first load in an idle window soon after the start, when some profile has `shadow.mode: on` and no other model is loading (0.3.34-test; before, the first turn of such a profile started it, waited behind it and landed it on full caches), and again with the next turn after a release. Its own pool, up to the claim the planner admitted, is left out of the engine's memory ceiling, as the narrator's (`server/memlimits.role_exclude`). A load that failed (its weights not on the box yet, a load error) is tried again in an idle window once `SHADOW_RETRY_S` (300 s) has passed, and `/health` `roles.shadow.error` says why it failed. When `QWEN38_SHADOW_MODEL` names the narrator's model, both share one slot and one copy of the weights.

## Switches

- serve.env: `QWEN38_SHADOW_MODEL` (empty: off, nothing imported), `QWEN38_SHADOW_DIR`, `QWEN38_SHADOW_GB`, `QWEN38_SHADOW_GRID`, `QWEN38_SHADOW_DISK` (`auto`: `~/.qwen38-spark-engine/cache/shadow`), `QWEN38_SHADOW_DISK_GB`.
- A profile's `shadow` block (absent: off): `mode` (`off`, `on`), `narrator` (`off`, `on`), `router` (`off`, `shadow`), `router_min_tokens`.
- `GET /v1/conductor/shadow/status`: the slot, every session's read position, backlog, bytes and checkpoints, the router's agreement, the last answers.

## Live

0.3.25-test runs it for the coding profile since 2026-10-04 16:05: `shadow: {mode: on, router: shadow, router_min_tokens: 8192, narrator: off}`, Qwen3.5-0.8B sharing the narrator's slot. A 5-turn test session of opencode's shape (prompts of 12,676 to 21,491 tokens): the shadow followed every turn, went back to a prompt-end checkpoint 3 times when an answer came back re-rendered (1,176 ids read again), and ended 83 ids behind. The router's question, asked of the 0.8B on all 5 turns, agreed with the router 0 times: it picked `debug` 4 times and `design_ui` once at 0.15 to 0.50, where the router said `chat` or `code_edit` at 0.72 to 0.997. As a router the 0.8B is not usable with this question; the journal will show whether that holds on real sessions, and a 2B or 4B shadow, or a narrower question, is the next thing to try.

## The router question with the 2B and the 4B (2026-10-05)

`tools/shadow_router_eval.py` asked the router's question again, offline, at the places the live shadow asked it: 58 journaled turns of 6 opencode sessions (2026-10-04 16:30 to 22:00), rebuilt from each session's parked file. The live model is the control: its offline answers gave the journal's probabilities again within 0.02 on 46 of the 58 turns and the same label on 57. Two questions: `live` (the classes with their descriptions, as the shadow asks) and `labels` (the classes' short labels only).

| model | live question | labels question | read, ms per 1,000 tokens (offline, no graph) |
|-|-|-|-|
| qwen3.5-0.8b | 8 of 58 (5 of 46 exact, 1 of 9 fresh) | 2 of 58 | 146 |
| qwen3.5-2b | 25 of 58 (18 of 46, 3 of 9) | 8 of 58 | 175 |
| qwen3.5-4b | 19 of 58 (16 of 46, 2 of 9) | 44 of 58 (34 of 46, 4 of 9) | 447 |

"Fresh" are the 9 turns whose class Qwen decided on that turn; the others carry the session's class forward. The router said `code_edit` on 42 of the 58 turns, so a constant `code_edit` scores 42 of 58 and 4 of 9 fresh. The 4B with the short labels is at that baseline: it says `code_edit` 47 times. None of the three reads the class better than the constant does, so the shadow keeps the 0.8B (shared slot, 1.5 GiB of weights, the fastest read) and the question stays in shadow. A 2B shadow would share the voice's weights, but it buys nothing here. A useful next step is a question the small models can answer: two or three classes (code, chat, other), asked only where the router decides fresh.

## Open

- The read rate. The 0.8B reads only 1.3 times faster than the 2B, so its chunks are bound by kernel launches and small GEMMs, not by its weights. A graph per chunk shape or a fused GDN prefill for the small sizes would help.
- The fork decode has no CUDA graph: the session's KV buffer moves when it grows, and a graph holds its address.
- The first long prompt of a session is read in about 3.5 s per 40,000 tokens; until then a question about that session waits or is late.
