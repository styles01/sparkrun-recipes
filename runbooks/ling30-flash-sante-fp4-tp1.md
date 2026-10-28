# Runbook: Ling-3.0-flash-Sante FP4 hybrid — TP1 on DGX Spark (vLLM 0.32.0)

**Lane**: Sante FP4 · **Status**: LIVE 2026-10-27 (owner-ordered boot) ·
**Recipe**: [`recipes/ling30-flash-sante-fp4-tp1.yaml`](../recipes/ling30-flash-sante-fp4-tp1.yaml) ·
**Sibling**: [FP8-TP2 lane](../recipes/ling30-flash-sante-fp8-tp2.yaml)

Own FP4 pack (70.4GB, 36 shards) produced from the Sante bf16 parent in the
exact layout of `inclusionAI/Ling-3.0-flash-fp4` — first Sante quant servable
on a single Spark. Tradeoff vs the FP8-TP2 lane: ~33 tok/s single-stream
decode (Marlin-emulated class on GB10) in exchange for TP1 (one box free).

## 1. The pack

- Source: bf16 parent `ling30-sante-tree` (255GB, 27 shards) → `produce_fp4_pack.py`
  (vectorized; ~59 min CPU-only beside serving; resumable; container path
  `docker run -v <sante tree>:/src:ro -v <official config>:/off -v out:/out`).
- Layout (Ling-fp4 hybrid EXACT): routed experts → e2m1 nibbles I8
  low-nibble-first + per-group-32 e8m0 CEIL pow2 scales (expo =
  ceil(log2(amax/6.0))); dense 2D → fp8-block-128 e4m3 + e8m0 pow2 CEIL;
  exempt BF16: norms, conv1d, b_proj, mlp.gate, MLA kv-proj/g_proj, embeddings,
  lm_head; F32: A_log, dt_bias. Verified: dtype histogram matches the official
  pack's signature (I8 62,976 = exact expert-tensor count); expert tensor
  round-trip decode ±0.1875 (e2m1 boundary = format-inherent).
- **Tree MUST carry** (or long thinking traces drift into multilingual salad):
  `generation_config.json` = `{"temperature":0.6,"top_p":0.95,"top_k":20,
  "repetition_penalty":1.05,"do_sample":true}` — vLLM ≥0.32 reads it as wire
  defaults. `config.json` with the official `quantization_config` (NO
  `--quantization` flag), tokenizer, remote code, chat template.

## 2. Engine (vLLM 0.32.0 — hard requirement)

The Ling-fp4 hybrid loader (`Mxfp4MoEMethod` Ling route, `float8_e8m0`)
exists only in the **v0.32.0 release**; 0.31.x cannot load this pack. Image
`dsv41-serve:v032` = v52rc + official cu130 aarch64 wheel layered with
`pip install --no-deps` (torch 2.13.0+cu130 unchanged — wheel METADATA pins
torch==2.13.0 = match). Gotchas:
- Fetch the wheel from the **commit-hash URL prefix**, not the pretty index
  URL (that returns a 301-byte error page); verify it is a real zip after download.
- COPY it under its **canonical filename** (`vllm-0.32.0-cp38-abi3-
  manylinux_2_28_aarch64.whl`) — pip's wheel-name parser rejects custom names.
- `pip install --no-deps` ONLY — full dep resolution would try to move torch.

## 3. Launch / stop (spark1, canonical)

```bash
cd /home/jaita/dsv41_recipe_ablit
HEAD=10.200.0.1 bash scripts/launch/sante_fp4_tp1.sh up      # ready in ~5 min cold / ~2 min warm
HEAD=10.200.0.1 bash scripts/launch/sante_fp4_tp1.sh down
scripts/launch/sante_fp4_tp1.sh status|logs
```

Serve contract (from the launcher): `vllm serve <pack> --served-model-name
"sante Ling-3.0-flash-Sante" --port 8000 --tensor-parallel-size 1
--gpu-memory-utilization 0.85 --max-model-len 262144 --max-num-seqs 3
--max-num-batched-tokens 2048 --kv-cache-memory-bytes 6500000000
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
--kernel-config '{"enable_flashinfer_autotune":false}'
--trust-remote-code --enable-auto-tool-choice --tool-call-parser ling3
--reasoning-parser ling3`

**Do NOT forget the tool-call flags** — Loca/medhi/gordon profiles call tools
with `tool_choice=auto`; without the two parser flags every tool turn 400s.

**MANDATORY post-boot** (or after ANY reboot — volatile lock):
```bash
sudo nvidia-smi -lgc 0,2200        # heat cap, BOTH nodes
```

## 4. Verify by effect (the boot is not real until these pass)

```bash
curl -s :8000/v1/models | jq '.data[].id'     # expect sante + Ling-3.0-flash-Sante, max_model_len 262144
# real gen + tool call + trace hygiene + speed — one script:
python3 fp4_proof_bar.py                       # gen / tool / think-trace non-ascii / tok-s sample
```
Measured 2026-10-27: tool call REAL (`get_weather {"city":"Lisbon"}`); trace
14,620 think chars / 11 non-ascii (English fix holds at engine defaults);
~33 tok/s decode (100-tok fixed workload); 37GB headroom after load.

## 5. Troubleshooting (this lane's real traps)

- **Worker staging silently truncates** — a 2026-10-27 rsync to a 100%-full
  disk "succeeded" (exit 23, to-chk=0/48) but landed only 11/47 files. After
  ANY pack transfer: `du -sb <pack> | cut -f1` must equal 70,428,755,599 and
  `find -name 'model-*.safetensors' | wc -l` = 36. Trust byte counts, never
  rsync exit codes.
- **Ready fast but trace is salad** → generation_config.json missing from the
  tree (see §1). Fix the tree, relaunch.
- **HTTP 400 'auto tool choice' on tool turns** → parser flags missing (§3).
- **Load OOM / import errors after image changes** → you rebuilt past the
  torch pin; v0.32 + torch 2.13.0+cu130 only.
- **Multicast/RDMA fabric down (TP2 variant)** → RoCE pre-checks in
  `private/HANDOFF-TP2-VLLM.md` (per-spark env contract, 3 IFNAMEs, GID).

## 6. Numbers log

| date | boot | ready | decode (c=1) | notes |
|---|---|---|---|---|
| 2026-10-27 | #1 v032 TP1 | 315s cold | ~33 tok/s | proof bar pass (tools+trace+262K) |
| 2026-10-27 | #2 v032 TP1 | 126s warm | ~33 tok/s | parser flags added; proof bar pass |