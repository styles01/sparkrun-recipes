# Spark LLM Optimization — Oracle's Project

**Created:** July 11, 2026
**Owner:** Oracle (chief LLM-optimizer)
**Test pilot:** Loca 🏠 ("crazy local model babe" — wired to Spark exclusively)
**Hardware:** NVIDIA DGX Spark (GB10, 121GB unified memory, SM121, aarch64)
**Host:** `larryspark.local` (user `jaita`)

---

## Mission

James constantly finds new research on Twitter (Mia, spark-benchmarkers, other researchers) — recipes, benchmarks, GitHub repos, Spark Arena results. Oracle's job is to take that research, test it on the Spark via Loca, benchmark it, debug it, prevent crashes, and lock in the best working configurations. When James comes with a new recipe or needs to swap models, Oracle knows exactly what to do — from memory, not from scratch.

The Spark is a testbed for **concurrency-intelligence balancing**. Eventually it will become the Larry server. Until then, we find the right balance.

## The Three Flavors

| Flavor | Model | Concurrency | Speed | Intelligence | Tool Parser | Use Case |
|---|---|---|---|---|---|---|
| **35b** | Qwen3.6-35B-A3B NVFP4 | 4-32 | 109-292 tok/s | Good (91.0 agentic) | `qwen3_xml` | Daily driver, max concurrency, replace Ollama |
| **122b** | Qwen3.5-122B-A10B DFlash | 3 | 15-83 tok/s* | Frontier | `qwen3_xml` | When agents need to be smarter |

*\*122B is workload-dependent: 80 tok/s on code/tool calls, ~15 tok/s on prose/conversation. DFlash n=12 excels on predictable traffic, struggles on open-ended conversation.*
| **ds4** | DeepSeek-V4-Flash | 1 | ~21 tok/s | Frontier+ | `deepseek_v4` | Hard reasoning, coding, single-user |

## Roles

- **James** — Research scout. Finds recipes on Twitter/GitHub/Spark Arena, brings them to Oracle. Says "switch to 35b" and it works. Approves all changes. Ultimate decision-maker.
- **Oracle** — Chief LLM-optimizer. Takes research → tests on Spark → benchmarks → debugs → locks recipes → maintains memory. Engineer + memory keeper.
- **Loca** — Test pilot. Wired to Spark exclusively. Every flavor runs through her first. If Loca can't complete an agent loop, the recipe isn't ready.
- **Mais** — Workspace maintainer. Owns `~/.hermes/workspace/dgx-spark/` with historical benchmarks and state docs. Oracle references but doesn't own these.

## Flavor Selection Guidance (Updated July 11, 2026 — with Loca qualitative testing)

| Use case | Best flavor | Why |
|---|---|---|
| Daily driving all agents, max concurrency | **35B** | 112 tok/s, 4 concurrent, but "teenager that doesn't pay attention" |
| Need one agent to be smarter | **122B** | 80 tok/s on code/tools, frontier smart, "adult" |
| Hard reasoning / coding / philosophical debate | **DS4** | 21 tok/s, smartest, no speculative overhead on prose |
| Don't want to switch | Ollama cloud ($20 tier) | |

### James's Qualitative Assessment (July 11, 2026)

**35B vs 122B:** "It's like a teenager that doesn't pay attention vs an adult. Faster but noticeably dumber and more hallucinatey."

**122B (philosophical debate):** "I'm genuinely impressed with Qwen 122's discussion — it feels thoughtful and relevant and intellectual."

**DS4:** Felt fastest end-to-end (~35s per response) — but that was partly because it was the only config where `model.default` matched the served model (no 404s).

### The Real Tradeoff

| | 35B | 122B | DS4 |
|---|---|---|---|
| Speed | ⭐⭐⭐⭐⭐ 112 tok/s | ⭐⭐⭐ 80 tok/s (code) / 15 tok/s (prose) | ⭐⭐ 21 tok/s |
| Intelligence | ⭐⭐⭐ "teenager" | ⭐⭐⭐⭐⭐ "thoughtful adult" | ⭐⭐⭐⭐⭐ smartest |
| Concurrency | ⭐⭐⭐⭐ 4+ | ⭐⭐⭐ 3 | ⭐ 1 |
| Tool calls | ✅ (with patch) | ✅ | ✅ |
| Reliability | ⚠️ hallucinatey | ✅ solid | ✅ solid |

**The $150→$20 Ollama replacement only works if 35B is smart enough for daily agent tasks. James's initial impression is that it's not.** This may change with the PR #48375 patch (was the intelligence drop partly the Mamba corruption bug?).

## The Critical Engineering Challenge

**MEMORY IS THE SYSTEM.**

Oracle's context will reset. Sessions will be lost. The difference between this project working and failing is whether Oracle can boot cold, read her files, and know:
1. What's currently running on the Spark
2. What the last-known-good config was for each flavor
3. How to switch flavors deterministically
4. What went wrong last time and how to avoid it

Everything must be written down. Every recipe locked. Every benchmark logged. Every failure documented. The boot logic (AGENTS.md → INDEX.md → STATE.md) must be sufficient for a fresh Oracle session to pick up where the last one left off.

## Directory Structure

```
~/.hermes/profiles/oracle/workspace/spark-llm-optimization/
├── README.md              ← This file (project brief & vision)
├── INDEX.md               ← Boot file — start here on every session
├── PROJECT-PLAN.md        ← Phased roadmap
├── STATE.md               ← Last-known-good tracking (what's running NOW)
├── adr/                   ← Architecture Decision Records
│   ├── ADR-001-recipe-locking.md
│   ├── ADR-002-tool-calling-requirement.md
│   ├── ADR-003-switch-script-architecture.md
│   ├── ADR-004-memory-boot-logic.md
│   └── ADR-005-loca-test-pilot-protocol.md
├── recipes/               ← Locked recipes (one per flavor)
│   ├── qwen-35b.md
│   ├── qwen-122b.md
│   └── deepseek-v4-flash.md
├── scripts/               ← Executable switch scripts (deploy to Spark)
│   ├── switch-to-35b.sh
│   ├── switch-to-122b.sh
│   └── switch-to-ds4.sh
├── sources/               ← Community recipes, references, provenance
│   └── SOURCES.md
└── benchmarks/            ← Our own benchmark results + template
    ├── BENCHMARK-TEMPLATE.md
    └── (results populate after each test run)
```

## Key Constraints

- **One model at a time** on the Spark (121GB unified memory)
- **Tool calling must be ON** for all flavors (Hermes agent loop requires it — non-negotiable)
- **DS4 loads in ~5 min** — plan for the wait
- **Qwen 122B loads in ~8-12 min** (Docker pull + model load + compile)
- **Qwen 35B loads in ~5-15 min** (FlashInfer autotune on first boot, cached after)
- Use `larryspark.local` (bonjour), never the raw IP `192.168.2.185`
- **NEVER wire Oracle or global config to the Spark** — only Loca
- All changes require James's explicit approval before execution