# Project Plan — Spark LLM Optimization

**Created:** July 11, 2026
**Owner:** Oracle

## Phase 1: Foundation (CURRENT)

**Goal:** Lock down the three existing flavors with tool calling, deploy switch scripts, update Loca config.

| Task | Status | Notes |
|---|---|---|
| Project structure created | ✅ Done | All dirs, README, INDEX, STATE |
| Recipes documented (all 3) | ✅ Done | From Mais/Lara/Oracle history |
| Switch scripts written | ✅ Done | Not yet deployed to Spark |
| Sources & provenance documented | ✅ Done | Spark Arena, GitHub, Twitter researchers |
| ADRs written | 🔄 In progress | 5 ADRs to write |
| Boot logic (AGENTS.md, memory) | 🔄 In progress | Update after ADRs |
| Patch DS4 serve script | ⏳ Pending approval | Add tool calling flags |
| Deploy switch scripts to Spark | ⏳ Pending approval | Copy to `~/` on larryspark |
| Update Loca config (3 models) | ⏳ Pending approval | profiles/loca/config.yaml |

## Phase 2: Retest & Benchmark

**Goal:** Run all three flavors through Loca, capture standardized benchmarks, verify tool calls work.

| Task | Status | Notes |
|---|---|---|
| Retest DS4 with tool calling patch | ⏳ Pending | ~10 min (5 min load + tests) |
| Retest 122B | ⏳ Pending | ~15 min (8-12 min load + tests) |
| First test of 35B | ⏳ Pending | ~20 min (FlashInfer autotune first boot) |
| Clone MiaAI repo on Spark | ⏳ Pending | For 35B chat template (Recipe B/C) |
| Standardized benchmarks | ⏳ Pending | Use BENCHMARK-TEMPLATE.md |
| Fill STATE.md last-known-good | ⏳ Pending | After each successful test |
| Write benchmark results | ⏳ Pending | benchmarks/ directory |

## Phase 3: Lock & Optimize

**Goal:** Determine optimal concurrency-intelligence balance per use case. Lock recipes. Make switching fully deterministic.

| Task | Status | Notes |
|---|---|---|
| Compare all 3 flavors side-by-side | ⏳ | tok/s, TTFT, concurrency, tool call success, Loca agent loop |
| Test 35B headless maxout (Recipe C) | ⏳ | GMU 0.85, 32 seqs — may be better than Recipe A |
| Test 35B with 8+ concurrent Loca sessions | ⏳ | Real-world agent concurrency |
| Economic analysis: Ollama replacement | ⏳ | Can 35b replace $150/mo Ollama? |
| Lock final recipes | ⏳ | Update STATE.md, recipes/, scripts/ |
| Create `hermes` skill for switching | ⏳ | So any profile can call switch-to-35b etc. |

## Phase 4: New Recipe Intake (ONGOING)

**Goal:** When James brings new research from Twitter/GitHub/Spark Arena, Oracle has a process to test it.

| Task | Status | Notes |
|---|---|---|
| Recipe intake process documented | ⏳ | See ADR-001 |
| Test → benchmark → lock workflow | ⏳ | New recipe → Loca test → benchmark → STATE update |
| Community source monitoring | ⏳ | Track Mia, spark-arena, new GitHub repos |

## Phase 5: Larry Server Preparation (FUTURE)

**Goal:** Spark becomes Larry's server. Stable, deterministic, no experimentation in production.

| Task | Status | Notes |
|---|---|---|
| Determine Larry's workload profile | ⏳ | ECE instructions, ~12K context, many users |
| Pick winning flavor for Larry | ⏳ | Likely 35b for concurrency |
| Production hardening | ⏳ | Auto-restart on crash, health checks, monitoring |
| Lock production config | ⏳ | No more switching without explicit approval |