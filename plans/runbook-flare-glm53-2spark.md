# RUNBOOK — GLM-5.3-Flash-FP4-Local on 2x DGX Spark (Flare primary, eugr-b12x vLLM fallback)

Bench window: **2026-10-23 05:00 UTC (1am ET)** — cron `dc4a106da9d8` (flare-glm53-arena-bench-1amET).
Staged by cron `fab873d43398` 2026-10-22 (day-side). TRASH-ONLY deletions. The dsv41 canon lane
relights at the end via `/home/jaita/relaunch-dsv41-canonical.sh` (drain-gate before teardown, full verify after).

Serving contract for both runways: `served_model_name = nvidia/GLM-5.3-Flash-FP4-Local` (full HF id —
llama-benchy resolves the recipe `model:` id; a display name fails warmup 404, Task 1 instant death).
Checkpoint: rev `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` (== upstream HEAD), 178.18 GB stored
(42 shards + index; recipe `device_bytes` 177.78 GB + MTP draft 4.30 GB; vision tower NOT loaded).
GLM recipe claims: context 1,048,576; page 256 tok; MTP draft_tokens 5; stream_overlap ON; ranks {1,2,4}.
TP2 batched-spec defect FIXED at `22887c1` (validated Spark TP2+TP4; present in our checkout 5f37bd1 —
`git merge-base --is-ancestor 22887c1 HEAD` = yes). **TP2: keep concurrency ≤ 8.** Open defect:
TP4-at-c16 INVALID_SPECULATION ~1/16 — does not affect TP2.

---

## 0. Staged state (verified 10/22; re-check each line at 05:00Z)

| Artifact | State | Re-check |
| --- | --- | --- |
| Checkpoint hub cache (spark1) | `~/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local` — 52-file tree @76348d, ALL SIZES VERIFIED vs HF api (`/tmp/flare53-fetch.log` = `ALL SIZES VERIFIED`), zero `.incomplete`/`.part`; snapshot at `snapshots/76348d…`, `refs/main` = 76348d | `ls snapshots/76348ddb…` count = 51; find -name "*.part" = none |
| Checkpoint mirror (spark2) | same hub-cache tree rsynced from spark1 (`mirror-verified` marker in `/tmp/flare53-mirror.log`) | count + du match spark1 |
| Image b12x | `eugr/spark-vllm-b12x:latest` on BOTH nodes (spark1 pull + spark2 pull) | `docker images eugr/spark-vllm-b12x` each node |
| Image flare | built CPU-only on spark1 (`flare:local`), pushed as **`ghcr.io/styles01/flare-serve:0.1.0`**, `docker save \| ssh 10.200.0.2 docker load` to spark2; binary also at `/home/jaita/opt/flare-0.1.0/flare` (both hosts) | `docker run --rm flare:local --version` both nodes |
| flare agents | `flare agent install --accept-first` on BOTH; systemd `--user` flare-agent up; controller pairing in `~/.config/flare/orbit.toml`; cluster `[cluster.sparks] nodes = [...]` (rank order: spark1 first) | `flare orbit status --cluster sparks` |
| prep token | `/home/jaita/.config/flare/agent.token` — IDENTICAL bytes both hosts, 600 | md5 compare |
| ignition env | `.venv` in `~/src/flare-local-inference` on BOTH (uv sync `--extra prepare`); CUDA toolkit path in config | `.venv/bin/ignition export --help` works both |
| sparkrun plugin | `~/sparkrun-recipes/runtime/sparkrun-flare/src` + entry in `~/.config/sparkrun/config.yaml` plugins.paths | `sparkrun recipe show @styles01/glm53-flare` resolves runtime flare |
| recipes | `@styles01/glm53-flare` + `@styles01/glm53-b12x-fallback` registered cache-copy + GH push; BOTH pass `~/verify_arena_recipe.py` exit 0 ("flare" added to local gate VALID_RUNTIMES 10/22 — arena-side whitelist for flare is UNVERIFIED until submission; fallback covers refusal) | rerun gate now |
| runbook | this file, Mac copy `/Users/clawdio/sparkrun-recipes/plans/`, md5-synced to spark1 | md5 match |

Disk budget on spark1 at bench: ~482G free − 178G fetch = ≈304G pre-export. ignite export writes ≈190 GB
of package payloads → ≈115G free. **Do NOT stage anything else on spark1's root fs in-window.** spark2
stays >200G free (mirror 178G + peer payloads from export cache). The 120G floor is a FETCH guard, not a
serving guard — 115G post-export is expected and accepted.

Estate network (for all commands below):
- spark1 = rank zero/head. ConnectX twins: `enp1s0f0np0` @ **10.200.0.1** and `enP2p1s0f0np0` @ 10.200.1.1.
- spark2 = peer. `10.200.0.2` (and `10.200.1.2` on the second twin).
- RoCE HCA names (card's `IB_IF`): `rocep1s0f0,roceP2p1s0f0` — "use both twins".

---

## 1. In-window prepare — ignite TP2 multi-host over RoCE ring

GPU must be FREE (dsv41 canon down; verify with `nvidia-smi`/`/metrics` first — drain-gate law).
Runs ONE package: output name and paths are fixed because the recipe defaults + flare-pub staging below reference them.

**Rank zero (spark1)** run from `~/src/flare-local-inference`:

```sh
mkdir -p /home/jaita/flare-runtime /home/jaita/flare-pub /home/jaita/flare-trust
.venv/bin/ignition export glm-5.3-flash \
  --gpus 0 --ranks 2 --rendezvous 10.200.0.1:29630 \
  --max-concurrency 8 \
  --prefill-chunk-size 4096 \
  --roce-ring --cluster sparks \
  --output /home/jaita/flare-runtime/glm53-spark-tp2-mtp5-c8.safetensors \
  --verbose
```

Notes:
- checkpoint = the recipe's declared alias `glm-5.3-flash` → `nvidia/GLM-5.3-Flash-FP4-Local` @76348d from
  the hub cache (primed, offline-resolvable). If a `[defaults] model` config exists it is overridden by the alias arg.
- `--roce-ring --cluster sparks` = physical RoCE ring; requires the flare agents listening AND paired (staged);
  admission checks each rank's GPU UUID against its paired agent.
- `--max-concurrency 8` = the doc-validated TP2 envelope (batched-spec defect doc: packages TP2 `--ranks 2
  --max-concurrency 8 --gpus 0 --prefill-chunk-size 4096 --no-context-parallel` were the validated shape; we keep
  stream_overlap + context-parallel defaults from the 10/21 recipe head).
- Package storage: output on /home/jaita (≈190 GB payloads + primary); hardlink dedup with
  `[paths] package_cache` = `~/.cache/flare/packages` (same fs). Budget listed in §0.
- 8 TCP streams/peer default for checkpoint transfer (`--transfer-streams N` to change).

**Peer rank (spark2)**, from `~/src/flare-local-inference`:

```sh
.venv/bin/ignition join 10.200.0.1:29630 --rank 1 --ranks 2
```

Start the join process FIRST or within seconds of the export — rank zero waits at admission; either order
works (export waits for peers), but do not stray >a minute. Peers request only the checkpoint ranges they
need from rank zero over the ring; peer payload blobs are retained locally on spark2's package cache.

Watch for: `source-identity mismatch` (both checkouts must be the same rev — md5 the venv + checkout),
`preparation authentication failed` (token mismatch md5), fabric errors (agents not listening; run
`journalctl --user -u flare-agent`).

**After export, stage the publisher profile for docker serving (rank zero's dev key):**

```sh
cp ~/.config/flare/development-publisher.json /home/jaita/flare-pub/ 2>/dev/null
chmod 644 /home/jaita/flare-pub/development-publisher.json
# fingerprint: read it from the package (reads no GPU, takes seconds)
/home/jaita/opt/flare-0.1.0/flare inspect /home/jaita/flare-runtime/glm53-spark-tp2-mtp5-c8.safetensors --plain | grep -i -m1 -E "finger|publisher"
```

Copy the printed **fingerprint** — the serve commands need
`--dangerously-approve-key <FINGERPRINT> --dangerously-accept-publisher local-development` (unattended first
use; dev packages look up `development-publisher.json` when `--publisher-keys` is omitted — inside the
container pass the mounted profile explicitly). Note: `flare inspect` reads no GPU and takes seconds.

---

## 2. Serve — PRIMARY runway: Flare

Canonical per multi-host.md **Start ranks manually** (docker, per docs/container.md — the recipe executor):

- **Rank zero (spark1)** container (image `flare:local`, or `ghcr.io/styles01/flare-serve:0.1.0`):

```sh
docker run -d --name flare-r0 \
  --gpus '"device=0,capabilities=compute"' \
  --ulimit memlock=-1:-1 --stop-timeout 60 \
  -p 8000:8000 \
  --mount type=bind,src=/home/jaita/flare-runtime,dst=/models \
  --mount type=bind,src=/home/jaita/flare-pub,dst=/flare-pub,readonly \
  --mount type=bind,src=/home/jaita/flare-trust,dst=/flare-trust \
  flare:local run /models/glm53-spark-tp2-mtp5-c8.safetensors \
  --publisher-keys /flare-pub/development-publisher.json \
  --trust-store /flare-trust/trust.toml \
  --dangerously-approve-key <FINGERPRINT> --dangerously-accept-publisher local-development \
  --rank 0 --rendezvous 0.0.0.0:29800 --gpus 0 \
  --host 0.0.0.0 --port 8000 \
  --served-model-name nvidia/GLM-5.3-Flash-FP4-Local
```

- **Rank one (spark2)** — package data first (operator-owned placement for manual ranks):
  rank-1 payload blobs live in spark2's ignite package cache from the export; `flare run --rank 1` reads the
  primary + its rank's payloads. Copy the primary file + rank-1 payloads into a per-rank dir:

```sh
# on spark1: identify rank-1 payload files (inspect lists them; no GPU, seconds)
/home/jaita/opt/flare-0.1.0/flare inspect /home/jaita/flare-runtime/glm53-spark-tp2-mtp5-c8.safetensors --plain --rank 1 | head -40
# assemble a self-contained rank-1 copy on spark2 (primary + its payload blobs), via rsync over RoCE
rsync -a /home/jaita/flare-runtime/glm53-spark-tp2-mtp5-c8.safetensors 10.200.0.2:/home/jaita/flare-runtime/
rsync -a <rank1-payload-blob-dir>/ 10.200.0.2:/home/jaita/flare-runtime/payloads/
```

```sh
docker run -d --name flare-r1 \
  --gpus '"device=0,capabilities=compute"' \
  --ulimit memlock=-1:-1 --stop-timeout 60 \
  --mount type=bind,src=/home/jaita/flare-runtime,dst=/models \
  --mount type=bind,src=/home/jaita/flare-pub,dst=/flare-pub,readonly \
  --mount type=bind,src=/home/jaita/flare-trust,dst=/flare-trust \
  flare:local run /models/glm53-spark-tp2-mtp5-c8.safetensors \
  --publisher-keys /flare-pub/development-publisher.json \
  --trust-store /flare-trust/trust.toml \
  --dangerously-approve-key <FINGERPRINT> --dangerously-accept-publisher local-development \
  --rank 1 --rendezvous 10.200.0.1:29800 --gpus 0 --headless
```

Worker-before-head does not matter for flare (ranks rendezvous; rank zero waits for the peer at admission) —
fire the spark2 container first anyway for load overlap.

**Alternative serve (no docker — orbit, if the agent path is preferred at the window):**
`flare orbit run spark1:/home/jaita/flare-runtime/glm53-spark-tp2-mtp5-c8.safetensors --cluster sparks
--name glm53 --gpus 0 --host 0.0.0.0 --port 8000 --detach` from the controller (spark1); distribution is
automatic (peers fetch their rank's data through the agents). Use this if the docker manual-rank path
stalls on package placement. Stop with `flare orbit stop glm53 --cluster sparks`.

**Loading behavior**: no compile/capture at load (all prepared in the package). Progress = ten-column
stages; expect minutes, not hours. First-acceptance disclosures are suppressed by the approve-key flags.

### Smoke checks (before ANY bench client)
1. `curl --fail http://localhost:8000/health` (rank zero).
2. `curl -s localhost:8000/v1/models` → exactly `nvidia/GLM-5.3-Flash-FP4-Local`.
3. Fixed probe (dsv41-lane-watcher pattern):
   `curl -s -X POST localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"nvidia/GLM-5.3-Flash-FP4-Local","messages":[{"role":"user","content":"Name three European capitals."}],"max_tokens":60,"temperature":0}'` → factual, non-empty.
4. Defect sanity at TP2 c8: two CONCURRENT probes (different world-knowledge questions, e.g. capital of
   the Netherlands vs the deepest ocean trench); both must be factual and distinct (the 22887c1-defect
   symptom = fluent wrong-record answers / empty replies above 16 rows — should not reproduce ≤8).

---

## 3. FALLBACK runway — eugr/spark-vllm-b12x (vLLM, card-validated 2x DGX Spark)

Use when the flare path cannot run: package export fails, `flare` not whitelisted by the arena, or serve
smoke failed. Config = **verbatim from the model card** (nvidia/GLM-5.3-Flash-FP4-Local → "2 x NVIDIA DGX
Spark"). Cache is primed on BOTH nodes (§0) — containers run `HF_HUB_OFFLINE=1`-style offline resolution
through the hub-cache mount.

Container run env (both nodes) — card verbatim with our values substituted (HEAD_IP=10.200.0.1,
ETH_IF=enp1s0f0np0, IB_IF=rocep1s0f0,roceP2p1s0f0, MASTER_PORT=29501):

```sh
export HEAD_IP="10.200.0.1"
export ETH_IF="enp1s0f0np0"
export IB_IF="rocep1s0f0,roceP2p1s0f0"
export MASTER_PORT=29501
export IMAGE="eugr/spark-vllm-b12x:latest"
export HF_CACHE_DIR="$HOME/.cache/huggingface"
mkdir -p "$HF_CACHE_DIR" "$HOME/.cache/vllm" "$HOME/.cache/flashinfer" "$HOME/.triton" "$HOME/.cache/b12x"
docker run --rm -it \
  --privileged \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size=16g \
  --ulimit nofile=1048576:1048576 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN=$HF_TOKEN \
  -e MN_IF_NAME="$ETH_IF" \
  -e UCX_NET_DEVICES="$ETH_IF" \
  -e NCCL_SOCKET_IFNAME="$ETH_IF" \
  -e NCCL_IB_HCA="$IB_IF" \
  -e NCCL_IB_DISABLE=0 \
  -e OMPI_MCA_btl_tcp_if_include="$ETH_IF" \
  -e GLOO_SOCKET_IFNAME="$ETH_IF" \
  -e TP_SOCKET_IFNAME="$ETH_IF" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e HEAD_IP="$HEAD_IP" \
  -e MASTER_PORT="$MASTER_PORT" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOME/.triton:/root/.triton" \
  -v $HOME/.cache/b12x:/root/.cache/b12x \
  --entrypoint /bin/bash \
  "$IMAGE"
```

Shared engine env (both nodes) — card verbatim:

```sh
export CUTE_DSL_ARCH="sm_121a"
export SAFETENSORS_FAST_GPU="1"
export VLLM_ENABLE_ROCE_ALLREDUCE="1"
export VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_CTAS=8
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_SSM_CONV_STATE_LAYOUT="DS"
export VLLM_USE_AOT_COMPILE="1"
export VLLM_USE_MEGA_AOT_ARTIFACT="1"
export VLLM_USE_V2_MODEL_RUNNER="1"
export VLLM_ENABLE_PCIE_ALLREDUCE="0"
export B12X_POLICY_MODE="auto"
export VLLM_B12X_MOE_FP4_FORCE_A16=1
export B12X_W4A16_A4_PREFILL=1
```

Serve — WORKER FIRST (`--headless`, node-rank 1 = spark2), then HEAD (node-rank 0 = spark1):

```sh
vllm serve nvidia/GLM-5.3-Flash-FP4-Local \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --attention-backend B12X \
  --block-size 256 \
  --moe-backend b12x \
  --linear-backend b12x \
  --no-enable-flashinfer-autotune \
  --load-format b12x \
  --max-model-len 500000 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"moe_backend":"humming","attention_backend":"B12X"}' \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.87 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "10.200.0.1" \
  --master-port 29501 \
  --headless
```

HEAD: identical with `--node-rank 0` and NO `--headless`. Card law: same CX7 port on every node, both port
subinterfaces, cache on every node, workers BEFORE head, docker privileged.

Recipe `@styles01/glm53-b12x-fallback` encodes this exactly (runtime `vllm-distributed`, sparkrun injects
per-node `--nnodes/--node-rank/--master-addr/--master-port` + worker `--headless`). Smoke: /health →
/v1/models = the HF id → one completion probe → the TP2-c8 two-probe defect sanity.

---

## 4. The bench fire

```sh
export PATH=$HOME/.local/bin:$PATH
ssh jaita@192.168.2.185 'nohup env PATH="$PATH" python3 -m sparkrun arena benchmark run @styles01/glm53-flare --cluster sparkduo --trust > /tmp/glm53-bench.log 2>&1 &'
```

- Hosts by REGISTERED identity: sparkduo cluster = `10.200.0.1,10.200.0.2` (never 192.168.2.x for TP2).
- `--trust` needed for the recipe's executor_config/hooks (sage-night lesson: sparkrun strips
  `executor_config.entrypoint` on untrusted recipes — the flare image's CMD-carrying entrypoint must be
  under trust).
- Gate first (exit 0), pre-fire box check (only our container on GPU), then fire detached; progress via
  `~/.cache/sparkrun/benchmarks/bench_*/runs/*.json` count — never watch the SSH itself.
- If the ARENA whitelist refuses runtime `flare` → immediately run the SAME command with
  `@styles01/glm53-b12x-fallback` after the runbook's fallback serve steps; note why in the report.
- The b12x config in the recipe must NOT be edited post-run (post-run recipe edits are forbidden).

## 5. Teardown + relight (mandatory, in-window law)
- `--no-stop` variant: if serve was launched manually and the client runs with `--skip-run`/`--no-stop`,
  stop the flare containers/serve AFTER upload: `docker rm -f flare-r0 flare-r1` (containers run as root in
  container; host-side kill is ineffective — `docker rm -f`).
- dsv41 canon relight: `/home/jaita/relaunch-dsv41-canonical.sh` (the registry's v52rc ablit lane line).
  Drain-gate BEFORE any restart: `/metrics` num_requests_running+waiting must be 0. After relight: re-arm
  the keepalive pinger `~/sparkrun-recipes/scripts/keepalive-8000.sh`, verify /v1/models (dsv41 @262144) +
  one live reply + DSpark drafts on /metrics; md5-sync LANE-REGISTRY + registry staging note (below).
- If the flare package is kept for further A/B: the hub cache stays; flare-runtime dir + payload caches
  stay on disk (regulate at clearing; James's call) — do not trash anything tonight.

## 6. Registry staging note (append to LANE-REGISTRY glm-flare53 entry)

```yaml
    status: "STAGED 2026-10-22 (fab873d43398): checkpoint 52-file tree @76348d VERIFIED spark1 hubcache (~/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local) + spark2 mirror; flare repo ~/src/flare-local-inference @5f37bd1 (22887c1 in-tree, unshallowed); flare:local built CPU-only + ghcr.io/styles01/flare-serve:0.1.0 both nodes; eugr/spark-vllm-b12x:latest pulled both; flare agents installed+paired (--accept-first) cluster sparks; ignite .venv both hosts; sparkrun-flare runtime plugin registered plugins.paths; recipes @styles01/glm53-flare (runtime flare, NEW local-gate runtime) + @styles01/glm53-b12x-fallback BOTH gate-exit-0; runbook plans/runbook-flare-glm53-2spark.md. In-window: ignite export glm-5.3-flash --ranks 2 --rendezvous 10.200.0.1:29630 --max-concurrency 8 --roce-ring --cluster sparks (package ~190GB -> ~115G free); serve via flare:local manual ranks (rank1 @10.200.0.1:29800) or flare orbit run; served name MUST be full HF id (--served-model-name nvidia/GLM-5.3-Flash-FP4-Local); dev-publisher approval flags --dangerously-approve-key <fp-from-inspect> --dangerously-accept-publisher local-development; fallback = card-verbatim b12x TP2 (MASTER_PORT 29501, workers BEFORE head)."
```

## 7. Open risks (read before firing)
1. **Arena runtime whitelist for `flare` is unverified** — the local gate accepts it (10/22 add); the
   arena may refuse at submission. Fallback recipe + serve steps are staged and complete.
2. **sparkrun-flare plugin is NEW code** (0.1.0, written 10/22) — its native-cluster per-node commands
   (`flare run --rank N --rendezvous`) have never run end-to-end; the bench agent should
   `sparkrun benchmark performance @styles01/glm53-flare --cluster sparkduo -n` (dry-run) to inspect the
   generated commands BEFORE trusting the arena run. If the plugin misbehaves: serve manually per §2 and
   submit through the fallback recipe (or run the bench client with --skip-run).
3. **~115G post-export on spark1** — tight; do not stage anything else; if export fails mid-write, .part
   cleanup is TRASH-ONLY and re-runnable.
4. **Peer payloads for the docker manual-rank path** require the rank-1 copy step (§2); if that stalls,
   switch to the orbit serve (distribution is automatic there).
5. **TP2 c8 envelope is the safety rail**: never raise max-concurrency beyond 8 for flare TP2 tonight;
   the open TP4 defect does not affect TP2≤8 but stay in the validated envelope.
6. **James context ceiling (275K/lane)** applies to daily-driver configs — the bench keeps the card's
   500,000 max-model-len (submission-grade card config; not a daily-driver config).
7. If both runways die before ~02:00 ET, re-slot minimax-31 forward per the bench-window law (never
   measurement theater) and reschedule flare — do NOT burn the window.
8. Mac loses spark2 entirely if spark1 wedges (private link 10.200.0.x hops through spark1). If spark1
   TP2-wedges: power-cycle is the ~20-min-max recovery; never wait 2h for a wedged boot.