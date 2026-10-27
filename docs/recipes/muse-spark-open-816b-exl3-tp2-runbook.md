# Runbook — Muse-Spark-Open-816B-A42B EXL3 1.8bpw, TP2 (2× DGX Spark) TensorFold

Arena recipe: `@styles01/muse-spark-open-816b-exl3-tp2` — TP2 across sparks over
TensorFold's own RoCE mailbox, DFlash2 drafter (EXL3 4.0bpw), fp16 KV, prompt cache.

## Staging (dark-window, 2026-10-26/27)

- Pack `ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw` @ `38ad2925…` staged on
  BOTH nodes (194.44 GB / 27 shards; md5 spots + byte-verified — see
  `~/muse/STAGING-RECEIPTS.md`), hardlink trees at
  `/home/jaita/models/muse-spark/Muse-Spark-Open-exl3-1.8s`. The author-verified
  revision `efcdf2ec…` resolves from the same blobs — zero extra bytes.
- Drafter `RadixArk/Muse-Spark-Open-DFlash2` bf16 → **EXL3 4.0bpw** via the
  author's `scripts/quantise_drafter.sh` (46 EXL3 matrices, 0 missing, 2.5 GB;
  bf16 drafter cannot fit arena 100K depths — author estimator max 61K ctx).
- TensorFold Zig build: main `40c6406b` + patches tree `71a9f224` (built `5d6b7dfb`),
  sm_121 kernels + TF TP mailbox kernel. Built on spark2 ONLY (serving-only law);
  `zig-out` mirrored to spark1, SHA256SUMS-verified, tf-tp-test runs on both.
- Engine image: author-pinned public base
  `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591…4529`
  resolving on BOTH nodes; binaries volume-mounted read-only (`/zig`).
- sparkrun TensorFold plugin extended for multi-host cluster mode (ranked
  containers, worker-exec-before-head, head API wait on `/v1/models`); TF
  rendezvous = recipe-pinned `TF_TP_MASTER`/`TF_TP_PORT` (no init_port math).

## Serve shape (community fidelity, muspark-2dgx v1.1)

- rank0 = head on spark2: `tensorfold-native serve /model --host 0.0.0.0
  --port 8888 --drafter /drafter --parallel 4 --max-tokens 32768 --context <fit>`
- rank1 = spark1: `/zig/bin/tf-muse follow /model /drafter`
- Container mounts: model→`/model`, drafter→`/drafter`, zig-out→`/zig` (read-only).

## Operation

```bash
# fire (detached, on spark1):
nohup env PATH="$HOME/.local/bin:$PATH" python3 -m sparkrun arena benchmark run \
  @styles01/muse-spark-open-816b-exl3-tp2 --cluster spark \
  --hosts 10.200.0.2,10.200.0.1 > /tmp/muse-arena-bench.log 2>&1 &
# watch (separate ssh):
ls ~/.cache/sparkrun/benchmarks/bench_*/runs/*.json | wc -l
# stop:
python3 -m sparkrun stop sparkrun_9b730fff75bd100c_<clusterhash> --hosts 10.200.0.2,10.200.0.1
```

## Author reference numbers (their box, muse v1.1)

- c=1 decode: 30.1 greedy / 23.2 at Meta sampling
- 4-stream total: 43.1 tok/s — prefill ~200-209 tok/s
- Owner's daily-driver bar: c=1 ≥ 80 tok/s — this drop is a submission +
  long-context evaluation, not a crown candidate.