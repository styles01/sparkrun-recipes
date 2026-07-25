# INDEX — Spark LLM Optimization Boot File

**Read this first on every new Oracle session.**

## Quick Boot Sequence

1. Read this file (INDEX.md)
2. Read `STATE.md` — what's running on the Spark RIGHT NOW
3. Read `SUCCESS-CRITERIA.md` — minimum bars for a recipe to be "working"
4. Check `recipes/` for the flavor you need
5. If James says "switch to X" → use `scripts/switch-to-X.sh`
6. If testing a new recipe from Twitter/GitHub → see `adr/ADR-001-recipe-locking.md`
7. Update `STATE.md` after every switch, test, or benchmark

## Current State Summary

| Flavor | Status | Last Tested | Tool Calls |
|---|---|---|---|
| 35b | Recipe locked, not yet tested by us | — | ✅ in recipe |
| 122b | ✅ Working (Lara verified, July 10) | July 10, 2026 | ✅ built into serve.sh |
| ds4 | ❌ Tool calls missing from serve script | July 5, 2026 | ❌ needs patch |

**Currently running on Spark:** DeepSeek-V4-Flash (256K, 1 concurrent, NO tool calling)

## File Index

| File | Purpose | Read when |
|---|---|---|
| `STATE.md` | What's running NOW, last-known-good per flavor | EVERY BOOT |
| `PROJECT-PLAN.md` | Roadmap, phases, current phase | Planning |
| `recipes/*.md` | Locked recipe per flavor | Before switching or testing |
| `scripts/*.sh` | Executable switch scripts | When James says "switch to X" |
| `sources/SOURCES.md` | Community sources, parser registry, provenance | Researching new recipes |
| `adr/*.md` | Architecture Decision Records | Understanding WHY decisions were made |
| `benchmarks/*.md` | Benchmark results per test run | Comparing flavors |
| `benchmarks/BENCHMARK-TEMPLATE.md` | Template for new benchmark runs | Before running benchmarks |
| `MODEL-SWITCH-PLAYBOOK.md` | Loca config update procedure (3 fields!) | BEFORE switching models |

## Cross-Profile Resources

| Resource | Location | Owner |
|---|---|---|
| Spark hardware state | `~/.hermes/workspace/dgx-spark/SPARK-STATE.md` | Mais |
| Historical benchmarks | `~/.hermes/workspace/dgx-spark/BENCHMARKS.md` | Mais |
| 35B NVFP4 recipes (A/B/C) | `~/.hermes/workspace/dgx-spark/QWEN-35B-NVFP4-RECIPE.md` | Mais |
| Lara's locked 122B config | `~/.hermes/profiles/lara/memories/MEMORY.md` | Lara |
| DS4 deployment skill | skill: `ds4-dgx-spark-deploy` | Oracle |
| Loca's config | `~/.hermes/profiles/loca/config.yaml` | Oracle manages, Loca owns |

5. **Key Decisions (see adr/ for full context)**

1. **All flavors must have tool calling ON** (ADR-002)
2. **Switch scripts are the only way to change models** (ADR-003)
3. **Loca tests every flavor before it's declared good** (ADR-005)
4. **STATE.md is updated after every state change on the Spark** (ADR-004)
5. **NEVER launch a model without the pre-flight checklist** (ADR-006) — crashes can require physical power-cycle