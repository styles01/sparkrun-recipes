# TP2 vLLM Serving on 2x DGX Spark (GB10) — Full Setup Handoff

**Audience:** someone who has never stood up a two-node vLLM serve, manually, no automation.
**Worked example:** Ling-3.0-flash-Sante (BailingMoeV3, hybrid attention) served FP8-online, TP2, 3 slots @ 262,144 ctx, with native MTP speculation. Everything below mirrors what we run in production; substitute your model where noted.
**Machines:** `spark1` = rank 0 (HTTP head + one share of the tensor ring); `spark2` = rank 1 (headless worker, same vLLM image, same weights). Spark1 and spark2 have a private 10.200.0.1 / 10.200.0.2 overlay link plus your LAN. GB10 = 128GB unified memory per box, no separate VRAM.

---

## 0. Mental model (read once)

- **TP2** splits one model across two boxes. vLLM's multiprocessing backend does it across nodes: `--nnodes 2 --tensor-parallel-size 2`. Rank 0 hosts the OpenAI-compatible HTTP listener; rank 1 runs the same image with `--headless` (no HTTP, it just joins the ring).
- The two ranks rendezvous via a small TCP ring (**master addr/port :25000**) and then move all real tensor traffic over **the RoCE fabric NIC** (ConnectX, RDMA). NCCL must be TOLD which NIC to use — this is the #1 place people lose a week: on GB10, NCCL's auto-detect picks a NIC that has no fabric IPv4 and the worker hangs or dies in init. This is all explained in §2.4 and it is already coded into the launcher in §3.
- **Same image, same weights, same path, both boxes.** The model directory path must be identical on both nodes (vLLM ships the path in a distributed handle).
- The boxes are **unified memory**: the GPU and the OS share one pool. Nothing is "VRAM" vs "RAM" — the full weight+KV budget counts against one pool (~121GB usable per box after OS).

---

## 1. One-time setup — BOTH nodes

Do this on each node unless marked HEAD-only/WORKER-only.

### 1.1 SSH (the launcher talks ssh between nodes)

The launcher script drives the other node through `ssh -o BatchMode=yes`, so ALL node-to-node ssh must be key-based with NO password prompts:

```bash
# on spark1, generate a key if none:
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
# copy it to spark2 (use whatever auth you have the FIRST time, e.g. password):
ssh-copy-id 10.200.0.2
# verify passwordless, and also self-ssh (the launcher ssh's its own node too):
ssh -o BatchMode=yes 10.200.0.2 'echo ok'
ssh -o BatchMode=yes 10.200.0.1 'echo ok'
```

Also make the LAN/overlay names resolvable (`/etc/hosts` on both or your DNS):

```
10.200.0.1   spark1
10.200.0.2   spark2
```

If you run the launcher FROM a workstation (we run it from spark1), that box needs key-ssh to both nodes as well.

### 1.2 Sudo note

Our boxes are password-`sudo` (no NOPASSWD). The launcher's page-cache flush step runs `echo 3 | sudo -n tee /proc/sys/vm/drop_caches`. On a pure password box this prints a harmless warning and boots without the flush — acceptable, but the better setup is to grant that single command:

```bash
/etc/sudoers.d/drop-caches:   <user> ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
```

### 1.3 Docker

Docker on both nodes, and the user you drive this with needs docker group rights.

### 1.4 The NVIDIA Container Toolkit (if the engine runs in containers)

`nvidia-ctk runtime configure --runtime=docker` + docker daemon restart; verify with `docker run --rm --gpus all <image> nvidia-smi`.

---

## 2. Weights — fetch, VERIFY, tree-build (both nodes)

### 2.1 Fetch at a pinned revision

```bash
python3 - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("inclusionAI/Ling-3.0-flash-Sante",
                      revision="f74bd27bd26ffc71a50dd847805f8f7666ff0b43",
                      cache_dir="/home/<you>/models/hf")
print(p)
EOF
```

Pinned revs matter: a card's floating `main` moves under you; record what you fetched.

### 2.2 VERIFY REAL WEIGHTS (this is the stub trap)

`hf download` exits 0 even when the repo is an empty stub (a README + a tiny placeholder). Never trust exit codes — verify **bytes and count** on disk:

```bash
du -sh /path/to/snapshot        # Sante = ~255GB across 27 model-*.safetensors
find /path -name "*.safetensors" | wc -l     # 27
find /path -name "*.safetensors" -size -1G  # must print NOTHING (all shards are multi-GB)
```

A "27 files" repo of a few hundred MB total is a stub. If your colleague had "vLLM in trouble before" — odds are decent it was a stub, a mismatched quant scheme, or a fabric hang; all three are covered here.

### 2.3 Build the serving tree (hardlinks, not symlinks)

vLLM's container only mounts the model dir; symlinked ancillary files (tokenizer, remote-code .py) dangle inside the container and the boot dies on tokenizer instantiation. Same filesystem → hardlink tree, instant, zero extra disk:

```bash
mkdir -p /home/<you>/models/ling30-sante-tree
cp -alL <snapshot-dir>/* /home/<you>/models/ling30-sante-tree/
find /home/<you>/models/ling30-sante-tree -type l | wc -l   # MUST be 0
```

(`cp -aL` instead would DATA-COPY the whole 255GB — we once filled two 4TB disks doing that.)

Do BOTH nodes — the worker reads weights directly.

### 2.4 The generation config (do not skip)

The Ling family collapses with neutral sampling (multilingual garbage mid-sentence after ~1.5K thinking tokens — looks like "the model can't stay in English"; it's actually the sampler). If your tree lacks `generation_config.json`, the engine serves temperature 1.0 with nothing else — write the family's canonical sampler into the tree root:

```json
{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05, "do_sample": true}
```

The engine reads it from the model dir as wire defaults for clients that omit sampler fields — no client side needed. This one file is the difference between "clean English" and drift.

---

## 3. The engine image + launcher

### 3.1 Image on BOTH nodes

TP2 needs the engine image present on each node. If it exists only on spark1:

```bash
docker save dsv41-serve:v52rc | ssh -o BatchMode=yes 10.200.0.2 docker load
# verify SAME image id both sides:
ssh 10.200.0.1 docker images --format "{{.ID}} dsv41-serve:v52rc"
ssh 10.200.0.2 docker images --format "{{.ID}} dsv41-serve:v52rc"
```

### 3.2 The launcher (what ours actually does)

Ours: `/home/jaita/dsv41_recipe_ablit/scripts/launch/sante_tp2.sh` (142 lines). Contract:

```
./sante_tp2.sh up|down|status|logs [head|worker]
```

It:
1. discovers each node's fabric IPv4 (`ip -4 addr show enp1s0f0np0`) and the RoCE **GID index** that carries that IPv4 (`/sys/class/infiniband/rocep1s0f0/...` walk) — NCCL gets the pair as `NCCL_IB_HCA + NCCL_IB_GID_INDEX`. Refuses to boot if either is missing (fabric not ready);
2. verifies `MODEL_DIR` exists on both nodes + the image exists on both nodes;
3. flushes page cache (`drop_caches`) on both;
4. generates a per-rank `vllm serve` script, `docker run`s one container per node with the RoCE devices/env, launches **rank 1 (worker, --headless) FIRST, then rank 0 (head)**;
5. polls `:8000/v1/models` every 15s until answers, printing `ready after N s`.

### 3.3 The complete env contract (copy verbatim, this IS the recipe)

Our canonical relaunch, `/home/<you>/relaunch-sante-canonical.sh`:

```bash
cd /home/<you>/<recipe-repo> || exit 1
HEAD=10.200.0.1 WORKER=10.200.0.2 \
MODEL_DIR=/home/<you>/models/ling30-sante-tree \
TEXT_ONLY=0 \
SPEC='{"method":"mtp","num_speculative_tokens":3}' \
UTIL=0.85 MAX_SEQS=3 KV_BYTES=6500000000 \
EXTRA='--quantization fp8 --trust-remote-code --enable-auto-tool-choice --tool-call-parser ling3 --reasoning-parser ling3' \
IMAGE=dsv41-serve:v52rc \
SERVED='sante Ling-3.0-flash-Sante' \
bash scripts/launch/sante_tp2.sh up
```

Knob-by-knob (what to mimic and what to swap for your model):

| Knob | Meaning | For your model |
|---|---|---|
| `HEAD`/`WORKER` | rank-0 / rank-1 host **IPs** (fabric overlay, not LAN names) | your two nodes |
| `MODEL_DIR` | same absolute path both nodes | your tree |
| `SPEC` | speculative decoding config, JSON; `mtp` + 3 draft tokens | Ling-family ships the NEXTN/MTP head inside the checkpoint — keep; other families may need it off (`SPEC=` empty) or a separate drafter |
| `UTIL` | fraction of the box used by model+KV | 0.85 safe default |
| `MAX_SEQS` | concurrent streams | 3 |
| `KV_BYTES` | KV pool pins in **PLAIN BYTES** (6.5e9 = 3x250K tokens at ~8.06KB/token for this arch's attention) | the one number to actually calculate: (bytes/token) x tokens-per-stream x MAX_SEQS; wrong = boot refuses or slots shrink |
| `EXTRA` | `--quantization fp8` = ONLINE quant (weights stay bf16 on disk, ~64GB/rank); `--trust-remote-code` mandatory for Ling-family (repo .py files); tool/reasoning parsers = agent-use | fp8 is the safest online scheme on CUDA/vLLM 0.31; online `mxfp4` has NO CUDA path — don't chase it |
| `IMAGE` | engine image tag | your built image tag |
| `SERVED` | space-separated served names — provide ALL aliases your clients know | at least the family canon name |
| `TEXT_ONLY=0` | serves images if the model has a tower; 1 stubs vision out `--limit-mm-per-prompt {"image":0}` | your call |

Inside the containers (set by the launcher, replicate if you roll your own):

```
NCCL_IB_HCA=rocep1s0f0            # the RoCE HCA that matches FAB_IF below
NCCL_IB_GID_INDEX=<derived>       # GID whose IPv4-mapped value == the node fabric IP
NCCL_IB_DISABLE=0  NCCL_NET=IB
NCCL_SOCKET_IFNAME=/GLOO/TP_SOCKET_IFNAME=enp1s0f0np0   # ALL THREE, same fabric NIC
VLLM_HOST_IP=<fabric self IP>
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800    # a 255GB load over the ring takes a while; default timeout kills mid-load
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
HF_HUB_OFFLINE=1
```

Container shape (verbatim from the launcher, reproduce the parts that matter):

```
--network host --ipc host --shm-size 64g --gpus all --device /dev/infiniband
--ulimit memlock=-1 --ulimit stack=67108864
--restart no                 # deliberate: docker auto-restarting ONE rank of a TP pair
                             #   produces a head that can never rejoin — restart BOTH by hand
-v <MODEL_DIR>:<MODEL_DIR>:ro
-v /home/<you>/.cache/sante-serve:/root/.cache
```

The server line rank 0 gets:

```
vllm serve <MODEL_DIR> --served-model-name <SERVED> --host 0.0.0.0 --port 8000
  --tensor-parallel-size 2 --nnodes 2 --node-rank 0 --master-addr <head fabric IP> --master-port 25000
  --gpu-memory-utilization 0.85 --max-model-len 262144 --max-num-seqs 3
  --max-num-batched-tokens 2048 <extra flags>
```

Rank 1 = identical + `--headless`, `--node-rank 1`, worker IP as VLLM_HOST_IP.

### 3.4 RoCE fabric pre-checks (before EVER launching)

```bash
ip -4 -o addr show enp1s0f0np0                     # each node: a private fabric IPv4 must exist
ls /sys/class/infiniband/                          # rocep1s0f0 present
ping 10.200.0.2                                    # overlay reachable both ways
ss -tln | grep 8000                                # port free on the HEAD
```

If the fabric IP is missing: someone removed the NIC config or it came up without the bond — fix THAT first; the launcher's refusal ("fabric not ready") is the guard rail.

---

## 4. Launch + verify (every single time)

### 4.1 Boot

Fire the §3.3 script. It BLOCKS until ready — run it inside `nohup`/a terminal multiplexer, NOT a plain ssh call:

```bash
nohup bash /home/<you>/relaunch-sante-canonical.sh > /tmp/relaunch.log 2>&1 &
tail -f /tmp/relaunch.log
```

Expect: `head ... gid ... | worker ...`, both `rank started` lines, then `ready after ~700-800 s` for a 255GB-model first load. A 13-minute dark load is NORMAL — don't diagnose inside it.

### 4.2 Verify by EFFECT (metadata endpoints can lie)

1. **Names**: `curl -s http://<head>:8000/v1/models` — every name from `SERVED` present; check `max_model_len` = what you set.
2. **TP2 actually formed** (the dead-ring trap: with a dead rank the endpoints STILL answer 200 while every completion hangs forever):
   - `ss -tn state established '( sport = :25000 )'` on the head — you want multiple PEER connections, not a loopback pair;
   - on the worker: `free -g` shows the RAM resident (~half the weights at rank 1). Idle-worker RAM ~4GB = rank never joined;
   - a REAL completion POST returns text — never call a lane "up" on a `/v1/models` GET.
3. **One live generation**:
   ```bash
   curl -s http://<head>:8000/v1/chat/completions -H "Content-Type: application/json" \
     -d '{"model":"sante","max_tokens":60,"messages":[{"role":"user","content":"Name three European capital cities."}]}'
   ```
   (Our healthy output: "Paris, Berlin, Madrid.")
4. **Clock cap (hardware safety law here)** — GB10 heat cap is volatile, re-required after every boot:
   ```
   sudo nvidia-smi -lgc 0,2200        # on BOTH nodes
   ```
5. **Reasoning/thinking families**: budget max_tokens past the thinking block — a 4200-token request with 15K chars of thinking returns EMPTY content at `finish_reason: length`. Not a bug.

---

## 5. Teardown + relaunch (swaps)

```bash
cd /home/<you>/<recipe-repo> && HEAD=<head-ip> WORKER=<worker-ip> \
  bash scripts/launch/sante_tp2.sh down
```

Then verify teardown by EFFECT: `ss -tln | grep 8000` empty, `docker ps` shows no lane container, `free -g` available actually rose. The `down` returning exit 0 proves nothing — the launcher prints `down` per node either way.

Before relaunching anyone else's lane: check no client requests are in flight (`curl /metrics | grep num_requests`) — restarting with a queue in flight kills them.

---

## 6. Troubleshooting — the classics, in order

| Symptom | Actual cause | Fix |
|---|---|---|
| boot dies in first minutes: "MODEL_DIR missing on <node>" | tree built on one node only | build the hardlink tree on BOTH |
| "IMAGE missing on <n>" | image only on head | docker save \| ssh <n> docker load (§3.1) |
| "fabric not ready: / gid" | fabric NIC has no IP, or HCA name differs | §3.4 pre-checks; find the HCA with `ls /sys/class/infiniband/` |
| containers up forever, no READY, head logs repeat waiting for worker | NCCL init wedged (wrong HCA/GID), or master :25000 blocked by firewall | grep head container logs for NCCL lines; confirm the env triple §3.3; ensure :25000 open on the head |
| `/v1/models` answers but completions hang | the dead-ring lie: rank 1 never joined | §4.2 triple check; `docker logs` the WORKER too (`scripts ... logs worker`) |
| boot OOMs/aborts near the end | page-cache ate the "available" pool (GB10 does NOT reclaim it for CUDA) | `drop_caches` both nodes, re-try; check `KV_BYTES` math |
| boot fine one day, OOM next morning after a reboot | volatile GPU reservations/clocks + leftover processes | after any reboot: heatcap re-apply + `pgrep -a vllm` clean + reboot-clears-residue (~3 min) |
| container refuses to die | engine runs as root inside | `docker rm -f <name>` (works) |
| workers crash at startup on env parse | an env var forwarded as EMPTY STRING bypassed defaults | set real values or unset the var entirely |
| outputs degraded vs last week, gibberish/mixed-language mid-sentence | tree lost its generation_config / sampler defaults | §2.4; write the file, relaunch |
| tool-calling client gets HTTP 400 "auto tool choice requires --enable-auto-tool-choice..." | flags omitted from the SERVE flags | put them in EXTRA, relaunch, re-verify with a REAL tool-call request — a config glance is not a proof; curl one real request every time |
| completions 200 but the client still fails: "provider said HTTP 400" — and you KNOW flags are set | you checked the wrong engine (older boot/other node) | `docker inspect <name> --format {{.State.StartedAt}}` + compare boot time vs the failing request time; turns during a relaunch hit the OLD engine |
| everything starves, ping works, ssh banner dead | unified-memory pressure: page-cache churn + another job during a load | power cycle only after ~20 min of no progress; heavy I/O never beside a served lane |

**Proof discipline (the meta-rule):** every "it works" claim = one real client request through the real consumer door, after the boot boundary. Metadata GETs and on-node curls have burned this lesson into us more than once.

---

## 7. FP8 specifics (the online-quant lane you're mimicking)

- FP8 online = per-tensor dynamic weights quantized AT LOAD from the bf16 tree. No pack to download, no build; disk holds the full 255GB tree; memory takes ~64GB/rank + KV.
- `--quantization fp8` requires nothing else in-tree; the arch's quant config (`quantization_config` in config.json) governs. Verify post-boot with one completion.
- If you'd rather serve the model as fp4-class: that is a DIFFERENT loader path (needs engine ≥ v0.32 for this family's hybrid format) — do not swap the scheme in EXTRA and expect the same boot.
- Measure and record your decode speed on a fixed workload (1-token prompt, max_tokens 100, temperature 0, ignore_eos) — it's your health baseline forever after.

## 8. Our measured numbers on this exact recipe (GB10 x2)

- Boot to ready: ~759s
- Decode: ~59–68 tok/s single stream (MTP k=3, draft acceptance ~57% drafted→accepted 2.8 tok/step)
- Prefill: ~3.2K tok/s on a 33.7K-token prompt
- Pool: KV_BYTES 6.5e9 = 3 slots @ 262,144; per-seq cap 262,144
- Box reserve at steady state: ~38-40GB available per node

*Handoff prepared from: `<repo>/scripts/launch/sante_tp2.sh` + the production relaunch script + heat-cap/liveness laws from our register. Questions to the estate register, not to memory.*