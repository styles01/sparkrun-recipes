# Exactness

Under greedy decoding, TandemLLM writes the text that its target model writes one token at a time, with one exception: at a near tie between the two best tokens, a batched verify can round the other way. Drafters, the lookup store and the caches change how fast the text arrives, not what it says; the shape of the verified tree can move a near tie. This page states the promise precisely, names the exception, and lists the checks that hold the engine to it.

## The promise

Our reference is plain greedy decoding on the same weights: one forward pass per token, take the highest-scoring token, repeat. For every greedy request, the engine's output tokens equal the reference's tokens up to the first near tie where the two paths choose differently.

"The same weights" matters. A profile chooses the weights. The NVFP4 profile is held to NVFP4 plain decoding, not to the BF16 release or the FP8 checkpoint. The difference between weight sets is a quality question, and [quantisation.md](quantisation.md) measures it with a separate gate. Speculation never adds to that difference.

The promise, with the near-tie exception, covers:

- any drafter, including a drafter that is wrong on purpose on every token;
- chains and trees of any shape the router builds, and both block widths;
- a request resumed from the prefix cache or the resident prefix;
- request options that change the scores (penalties, `logit_bias`, structured-output masks), because the engine applies them to every verified row exactly as plain decoding applies them to its one row.

## The one exception: near ties

Two tokens can score the same up to one rounding step of the arithmetic. Then which one wins an argmax depends on the order of a floating-point sum. A batched verify and a one-token step do not always sum in the same order, so at such a position the two can pick different tokens. After that position the texts differ, and both are correct greedy continuations of their own prefixes.

TandemLLM keeps this rare by design. The head returns fp32 logits, and every matrix-product kernel fixes its reduction order by the weight's shape and never by the row count ([kernels.md](kernels.md)). Two differences remain between a batched verify and a one-token step: the output head uses one kernel for a single row and another for several rows, and the verify path of the recurrent and attention layers does the one-token arithmetic in a different order. A small fp32 difference that falls on a BF16 rounding boundary changes one value, and the recurrent state and the KV cache carry it forward, so over a long output the verify's logits drift from the one-token decode's. Where the two best tokens are closer than that drift, the two paths can choose differently.

The agreement gate, `tools/verify_spec.py`, prints the gap between the top two logits in BF16 units (ulps) whenever outputs differ. A difference passes only when the gap at that position is within one ulp. Anything larger is a bug. The gate decodes a fixed number of tokens and does not stop at the model's end-of-turn token. A difference after that token is not part of any answer served with the default stop tokens, so it is judged separately: in the StairCut release's checks, the two differences above one ulp (1.46 and 1.73) both fall after the end of the answer.

The paper measures how often this happens with StairCut in its served configuration on the published NVFP4 set. Of 30 prompts (held-out prose, code and chat, 256 new tokens), 17 outputs are identical to plain greedy decoding up to the end of the answer: 13 over all 256 tokens, and 4 more that differ only past the end of the answer (largest gap 1.46 ulp). The other 13 follow plain decoding up to a near tie inside the answer, with a gap of 0.04 to 0.76 ulp, take the other token and continue from there; in one of them the token taken is the end-of-answer token, so that answer ends early. No output differs inside the answer at a larger gap. The one-ulp rule names the condition under which the two decoders can differ; it does not show that the difference is harmless.

## Parallel requests

With `--parallel N` ([server.md](server.md#parallel-requests-0338-test---parallel-n-off-by-default)) two or more decoding requests verify their trees in one pass. The promise is the same: each request writes its own greedy text up to a near tie. Two things can move a near tie that a request alone would not meet. A joint round verifies the request's tree cut to its share (16 nodes at two requests, 8 at four), another tree shape than the one it verifies alone. And the joint pass runs eagerly, where a lone request's verify runs from CUDA graphs. The batch itself is not one of them: every projection kernel keeps a row's result independent of the other rows up to 32 rows, and a joint pass stays at or under 32. On the box, `tools/parallel_verify_check.py` gives each slot's logits from the joint pass bit for bit what its own verify gives (with the lone verify eager too, as the joint one is), and on the CPU `tests/test_parallel_verify.py` holds two and three requests in lockstep to the tokens each writes alone.

What a request writes beside others is measured against what it writes alone (`tools/parallel_bench.py`, agreement): where a text differs, the tool prints the first differing token and the gap between the two best tokens there in the alone run. On 30 prompts (10 held-out prose continuations, 10 held-out code continuations, 10 chat questions; 256 tokens, greedy, store off; 2026-10-05, single runs), 17 of 30 texts were identical to their alone text with two requests at once and 15 of 30 with four; every first difference sat at a top-2 logprob gap between 0.0019 and 0.118 nats, under one BF16 step of a logit of magnitude 16 to 32 (0.125; the logits' magnitude there was not recorded). The packed draft proposes bit for bit what each slot's own draft proposes (`tools/packed_draft_check.py`, two slots on either drafter arm, four on the narrow one). One request at a time takes the path it takes without `--parallel`.

## Stronger than the promise: release identity

Every release candidate is also held to the release before it. Before a change ships, 13 plain requests (the five benchmark workloads, streamed and not, one with thinking, one with tools, one under penalties, one seeded sampled request) must come back byte-identical to the current release (`tools/api_text_check.py`). The served configuration must also produce bit-identical logits with the new flags off (`tools/flagoff_identity.py`). A speed change that alters one token of text does not ship. StairCut (2026-09-29) is the one documented exception. Its verify shapes differ from rc10's, so at near ties its text can differ from rc10's. It is held to plain greedy decoding instead, under the near-tie rule above.

## Sampling

A sampled request (temperature above 0) is exact in distribution. At each verified node the engine draws the target's own token, and a drafted token survives exactly while the draw lands on it. When the draw lands elsewhere, that draw is the token. For a drafter that proposes one fixed token this is the standard rejection sampler, `accept with min(1, p/q)`, with no second draw. The output therefore follows the target's own sampled distribution. A drafter that samples its own proposal carries its proposal distribution, and the accept uses it.

All sampling filters (`temperature`, `top_p`, `top_k`, `min_p`) run after the penalties, through one function in `engine/sample.py`. Drafted rows and plain rows see the same distribution.

A request with a `seed` draws with noise keyed by its position in the sequence: the token at index t is the argmax of `log p_t + G_t`, where `G_t` is Gumbel noise from a generator seeded by `(seed, t)`. That is an exact sample from `p_t`, and it does not depend on how many draws the drafters made before. The same seed therefore gives the same text whatever the drafters proposed.

`tests/test_sample.py` checks the samplers statistically on the CPU, and `tools/sampled_dist.py` checks the served engine's sampled answers against the target sampled alone.

## Caches

A cache restore gives back bytes, not an approximation. The prefix cache, the resident prefix and the disk tier take their snapshots on the prefill chunk grid, and a resumed request forwards the rest of its prompt on the same grid. It therefore computes the same arithmetic as a cold prefill on that grid. `tests/test_cache.py`, `tests/test_resident.py` and `tests/test_disk_cache.py` check, on a small random model, that a resumed request's logits and state equal a cold prefill's bit for bit. The disk tier's files carry a fingerprint of the weights, the engine's code and the arithmetic flags; a file of another fingerprint is never read. Whether Triton's autotuning after a restart can pick another prefill configuration, so that rows written before it differ from a cold prefill after it, the tests cannot say; `tools/disk_restore_check.py` checks it on the served engine in a hold, in two processes. On 2026-10-02 (0.3.14-test's code, a 10,701-token opencode-shaped prompt) the second process's cold prefill was the first's bit for bit (logits, recurrent states, convolution tails, every row, the state at each 1,024 line), and the first process's disk files restored to it: the state at the line, then the resumed prefill's logits, states and all six position-indexed buffers, and 48 greedy tokens.

The paper's long-context check sent a long prose prompt twice, cold and then resumed from the anchors. The two 256-token outputs were identical at 32k and 128k tokens and different at 64k, and that difference is not explained.

Session resumes are one step weaker. A session resume continues a conversation from the state its last turn ended in, which verify rounds wrote rather than prefill chunks. That state produced the previous answer, but it is not the state a cold re-read would compute. It is held to the engine's greedy gate (same tokens) and not to bit equality. [caches.md](caches.md) has the details.

The resident prefix's end anchor (0.3.14-test, `--end-anchor`, on) is a session resume of the same kind: the recurrent state at the end of the last generation, which verify rounds wrote, with the rows in place, and the next turn forwards only its new messages from there. It is held to the greedy gate, as the session cache is, and so is every anchor a prefill that resumed there takes (the rows from the resident prefix's `taint` up). Two things keep the rest of the caches exact: the disk tier writes nothing of such a prompt and reads over those rows, and the state store takes no prefix snapshot of it; a grid anchor at or below `taint`, or a cold prefill, resumes bit for bit as before. The tokens the session monitor hid are put back into the client's copy where the rows hold them, so the prompt the engine continues is the context it really had, the hidden phrase included, as a router note is replayed; whitespace the re-rendered copy holds differently right next to a hidden span is the rows' own there too (0.3.15-test), and any other difference serves the copy as sent. `tests/test_end_anchor.py` checks, on the random model, that the turn after a generation resumed at its end writes the same greedy tokens as the same turn resumed at the prompt's last anchor (no drafter, wrong and right chains and trees), that a pending commit cut back is the shorter commit's state bit for bit, and that `--end-anchor off` leaves every prefill bit for bit as before. [caches.md](caches.md#after-a-long-generation) has the details.

An opt-in response cache (`--response-cache`) answers an identical greedy request from memory. It is off by default, so a benchmark can never be pointed at a dictionary.

## What changes output on purpose

Some features change the text because the client asked for it. They are not exceptions to the promise. Plain decoding with the same setting writes the same text.

- Penalties (`repetition_penalty`, `presence_penalty`, `frequency_penalty`) and `no_repeat_ngram_size`.
- Structured outputs and forced tool calls, which mask the tokens the constraint does not allow.
- A reasoning budget or the stall close, which end the thinking block early ([speculative-decoding.md](speculative-decoding.md)).

- **A regenerate under TandemLLM** (0.4.4-test, SRV-241; the project's rule of 2026-10-07). This was the first place where TandemLLM samples without the client asking for it; the loop escape below is the second. Open WebUI's Regenerate sends the same conversation again and no sampling parameter, so under greedy it got the same answer word for word. `server/regenerate.py` recognises a regenerate: a chat request under an active TandemLLM plan whose conversation (every non-system message, ending with a user message) equals an earlier request of the same session that produced an answer. `X-Tandem-Regenerate: 1` also counts as a regenerate. That one request samples with temperature 0.7, top-p 0.95, top-k 20 and a fresh seed (`--regenerate-temperature`, `--regenerate-top-p`, `--regenerate-top-k`). The seed is in the journal row (`extra.regenerate`), so sending it reproduces the answer. The sampled path is the exact one above: the answer follows the target's own sampled distribution, drafter and tree included. Every other request is untouched and greedy, byte for byte as before. That covers the first answer, a follow-up, Continue, an edited message and a tool round. Also untouched: a request that names any sampling parameter (it gets the client's own), a raw or `bench` request, a side request and a compaction. `--regenerate off` (`QWEN38_REGENERATE=off`) turns detection off, and `X-Tandem-Regenerate: 0` turns it off for one request. The ledger column `sampling` is `regenerate` on these rows, and the dashboard's speed sample leaves them out.

- **The loop escape under TandemLLM** (SRV-475, 2026-10-14). An agent turn that ends in a repeat or near-repeat of one tool call samples with the model card's settings (the checkpoint's `generation_config.json`) and a fresh seed, until the session makes progress; then it is greedy again ([server.md](server.md#sampling-and-penalties)). A profile with `sampling.agentic_default: card` samples every agent turn; the shipped profiles keep `greedy`. The seed is in the journal row (`extra.loop_escape`). The same requests as for a regenerate are untouched, the client's own sampling included. The ledger column `sampling` is `loop_escape` or `card`, and the dashboard's speed sample leaves those rows out.

Two relaxed accept rules also exist (`--relax-tau`, `--relax-rank`) that accept a drafted token when it is close to the target's choice. They break the promise, exist only for measurement, and are off in every profile. When either is set, the server prints a warning line.

## The checks

| check | what it holds |
|-|-|
| `tools/verify_spec.py` | greedy output with no drafter, an adversarial drafter, a half-right drafter and the served drafters must match, up to the one-ulp near-tie rule |
| `tools/flagoff_identity.py` | the served configuration with new flags off is bit-identical to the build it replaces |
| `tools/api_text_check.py` | 13 plain requests byte-identical to the current release |
| `tools/cache_gate.py` | warm and cold runs agree (bit-identical on the chunk grid, same tokens for a session) |
| `tests/test_forward_tree.py`, `tests/test_tree.py` | tree verify equals chain verify on a tiny model, on the CPU |
| `tests/test_cache.py`, `tests/test_resident.py`, `tests/test_disk_cache.py` | restores are bit-exact, the disk tier's through a reopened tier and after a failed restore |
| `tests/test_end_anchor.py` | a resume at the end of the last generation writes the same greedy tokens as one at the prompt's line; off, every prefill as before |
| `tests/test_sample.py`, `tools/sampled_dist.py` | sampled output follows the target's distribution |
