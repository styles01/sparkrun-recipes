#!/usr/bin/env python3
"""
Spark LLM Weighted Score Calculator
Calculates a composite score for each model based on:
- Intelligence benchmarks (60%)
- Infrastructure metrics (40%)
"""

import json

# ============================================================
# BENCHMARK DATA (from research/intelligence-benchmarks.md)
# N/A = not reported by publisher
# ============================================================

models = {
    "Step 3.7 Flash": {
        "total_params": 198, "active_params": 11,
        "claweval": 67.1, "swe_bench": 56.3, "toolathlon": 49.5, "terminal_bench": 59.6,
        "mmlu_pro": None, "gpqa": None, "hle": None, "aime25": None,
        "hmmt_no_tools": None, "hmmt_with_tools": None,
        "scicode": None, "humaneval": None,
        # Infrastructure (our tested configs)
        "tok_s": 28.1, "lanes": 2, "ctx_per_lane": 200, "free_ram_gb": 0.5,
        "disk_gb": 105, "runtime": "llama.cpp",
    },
    "DS4-Flash": {
        "total_params": 159, "active_params": 11,
        "claweval": 57.8, "swe_bench": 52.3, "toolathlon": 43.5, "terminal_bench": 56.6,
        "mmlu_pro": 86.4, "gpqa": 87.4, "hle": 29.4, "aime25": None,
        "hmmt_no_tools": 91.9, "hmmt_with_tools": None,
        "scicode": None, "humaneval": 69.5,
        "tok_s": 21, "lanes": 2, "ctx_per_lane": 128, "free_ram_gb": 22,
        "disk_gb": 227, "runtime": "vLLM",
    },
    "Qwen 122B": {
        "total_params": 122, "active_params": 10,
        "claweval": None, "swe_bench": None, "toolathlon": None, "terminal_bench": 49.4,
        "mmlu_pro": 86.7, "gpqa": 86.6, "hle": 25.3, "aime25": None,
        "hmmt_no_tools": 91.4, "hmmt_with_tools": None,
        "scicode": 42.0, "humaneval": None,
        "tok_s": 29.6, "lanes": 3, "ctx_per_lane": 150, "free_ram_gb": 16,
        "disk_gb": 67, "runtime": "vLLM DFlash n=4",
    },
    "Nemotron Super 120B": {
        "total_params": 120, "active_params": 12,
        "claweval": None, "swe_bench": None, "toolathlon": None, "terminal_bench": 31.0,
        "mmlu_pro": 83.73, "gpqa": 79.23, "hle": 18.26, "aime25": 90.21,
        "hmmt_no_tools": 93.67, "hmmt_with_tools": 94.73,
        "scicode": 42.05, "humaneval": None,
        "tok_s": None, "lanes": None, "ctx_per_lane": None, "free_ram_gb": None,
        "disk_gb": 67, "runtime": "vLLM (untested)",
    },
    "Puzzle-75B": {
        "total_params": 75, "active_params": 9.3,
        "claweval": None, "swe_bench": None, "toolathlon": None, "terminal_bench": 24.0,
        "mmlu_pro": 82.4, "gpqa": 78.6, "hle": 16.5, "aime25": 89.7,
        "hmmt_no_tools": 93.4, "hmmt_with_tools": 93.9,
        "scicode": 40.6, "humaneval": None,
        "tok_s": 30.5, "lanes": 1, "ctx_per_lane": 131, "free_ram_gb": None,
        "disk_gb": 40, "runtime": "vLLM MTP k=3",
    },
    "Qwen 35B": {
        "total_params": 35, "active_params": 3,
        "claweval": None, "swe_bench": None, "toolathlon": None, "terminal_bench": 40.5,
        "mmlu_pro": 85.3, "gpqa": 84.2, "hle": 22.4, "aime25": None,
        "hmmt_no_tools": 89.0, "hmmt_with_tools": None,
        "scicode": None, "humaneval": None,
        "tok_s": 102.8, "lanes": 6, "ctx_per_lane": 256, "free_ram_gb": 45,
        "disk_gb": 22, "runtime": "vLLM MTP k=3",
    },
}

# ============================================================
# WEIGHTS
# ============================================================
WEIGHTS = {
    # Intelligence (80% total)
    "agent": 0.30,      # ClawEval, SWE-Bench, Toolathlon, Terminal-Bench
    "reasoning": 0.25,  # MMLU-Pro, GPQA, HLE, AIME25
    "code": 0.15,       # SciCode, HumanEval
    "math": 0.10,       # HMMT
    # Infrastructure (20% total)
    "speed": 0.05,      # tok/s
    "lanes": 0.05,      # concurrent agents
    "context": 0.05,    # context per lane (K)
    "overhead": 0.05,   # free RAM (GB)
}

# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize(value, min_val, max_val):
    """Normalize to 0-100 scale"""
    if value is None:
        return None
    if max_val == min_val:
        return 100
    return (value - min_val) / (max_val - min_val) * 100

def normalize_abs(value, floor, ceiling):
    """Normalize to 0-100 scale using absolute bounds (not relative min/max).
    This prevents a good score from getting 0 just because it's the lowest in the group."""
    if value is None:
        return None
    if ceiling == floor:
        return 100
    score = (value - floor) / (ceiling - floor) * 100
    return max(0, min(100, score))

def avg_scores(*scores):
    """Average non-None scores, return None if all None"""
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)

# ============================================================
# CALCULATE SCORES
# ============================================================

# Find max/min for normalization
all_claweval = [m["claweval"] for m in models.values() if m["claweval"] is not None]
all_swe = [m["swe_bench"] for m in models.values() if m["swe_bench"] is not None]
all_tool = [m["toolathlon"] for m in models.values() if m["toolathlon"] is not None]
all_term = [m["terminal_bench"] for m in models.values() if m["terminal_bench"] is not None]
all_mmlu = [m["mmlu_pro"] for m in models.values() if m["mmlu_pro"] is not None]
all_gpqa = [m["gpqa"] for m in models.values() if m["gpqa"] is not None]
all_hle = [m["hle"] for m in models.values() if m["hle"] is not None]
all_aime = [m["aime25"] for m in models.values() if m["aime25"] is not None]
all_hmmt = [m["hmmt_no_tools"] for m in models.values() if m["hmmt_no_tools"] is not None]
all_scicode = [m["scicode"] for m in models.values() if m["scicode"] is not None]
all_humaneval = [m["humaneval"] for m in models.values() if m["humaneval"] is not None]
all_tok = [m["tok_s"] for m in models.values() if m["tok_s"] is not None]
all_lanes = [m["lanes"] for m in models.values() if m["lanes"] is not None]
all_ctx = [m["ctx_per_lane"] for m in models.values() if m["ctx_per_lane"] is not None]
all_ram = [m["free_ram_gb"] for m in models.values() if m["free_ram_gb"] is not None]

results = []

for name, m in models.items():
    # Agent score — use absolute normalization (0-100 benchmarks, floor=0, ceiling=100)
    agent_raw = avg_scores(
        m["claweval"],
        m["swe_bench"],
        m["toolathlon"],
        m["terminal_bench"],
    )
    
    # Reasoning score — absolute normalization
    reasoning_raw = avg_scores(
        m["mmlu_pro"],
        m["gpqa"],
        m["hle"] * 3 if m["hle"] is not None else None,  # HLE is harder, weight up
        m["aime25"],
    )
    
    # Code score — absolute normalization
    code_raw = avg_scores(
        m["scicode"],
        m["humaneval"],
    )
    
    # Math score — use absolute scale (0-100 benchmark scores, floor=50, ceiling=100)
    # HMMT is already a percentage, normalize against 50-100 range
    math_raw = normalize_abs(m["hmmt_no_tools"], 50, 100) if m["hmmt_no_tools"] is not None else None
    
    # Infrastructure scores
    speed_raw = normalize(m["tok_s"], min(all_tok), max(all_tok)) if m["tok_s"] else None
    lanes_raw = normalize(m["lanes"], min(all_lanes), max(all_lanes)) if m["lanes"] else None
    ctx_raw = normalize(m["ctx_per_lane"], min(all_ctx), max(all_ctx)) if m["ctx_per_lane"] else None
    overhead_raw = normalize(m["free_ram_gb"], min(all_ram), max(all_ram)) if m["free_ram_gb"] is not None else None
    
    # Handle missing: use 50 as neutral score for missing intelligence benchmarks
    # (not 0 — that would over-penalize models that simply didn't publish)
    MISSING_DEFAULT = 50
    
    agent_score = agent_raw if agent_raw is not None else MISSING_DEFAULT
    reasoning_score = reasoning_raw if reasoning_raw is not None else MISSING_DEFAULT
    code_score = code_raw if code_raw is not None else MISSING_DEFAULT
    math_score = math_raw if math_raw is not None else MISSING_DEFAULT
    
    # Overall Intelligence — blend all 4 intelligence categories with their weights
    intel_total_weight = WEIGHTS["agent"] + WEIGHTS["reasoning"] + WEIGHTS["code"] + WEIGHTS["math"]
    overall_intel = (
        agent_score * WEIGHTS["agent"] +
        reasoning_score * WEIGHTS["reasoning"] +
        code_score * WEIGHTS["code"] +
        math_score * WEIGHTS["math"]
    ) / intel_total_weight * 100 / 100  # normalize to 0-100
    speed_score = speed_raw if speed_raw is not None else 0
    lanes_score = lanes_raw if lanes_raw is not None else 0
    ctx_score = ctx_raw if ctx_raw is not None else 0
    overhead_score = overhead_raw if overhead_raw is not None else 0
    
    # Weighted total
    total = (
        agent_score * WEIGHTS["agent"] +
        reasoning_score * WEIGHTS["reasoning"] +
        code_score * WEIGHTS["code"] +
        math_score * WEIGHTS["math"] +
        speed_score * WEIGHTS["speed"] +
        lanes_score * WEIGHTS["lanes"] +
        ctx_score * WEIGHTS["context"] +
        overhead_score * WEIGHTS["overhead"]
    )
    
    results.append({
        "model": name,
        "total_score": round(total, 1),
        "overall_intel": round(overall_intel, 1),
        "agent": round(agent_score, 1),
        "reasoning": round(reasoning_score, 1),
        "code": round(code_score, 1),
        "math": round(math_score, 1),
        "speed": round(speed_score, 1),
        "lanes": round(lanes_score, 1),
        "context": round(ctx_score, 1),
        "overhead": round(overhead_score, 1),
        "params": m["total_params"],
        "active": m["active_params"],
        "disk_gb": m["disk_gb"],
        "runtime": m["runtime"],
    })

# Sort by total score
results.sort(key=lambda x: x["total_score"], reverse=True)

# Print results
print("=" * 120)
print("SPARK LLM WEIGHTED SCORE — RANKED")
print("=" * 120)
print(f"\nWeights: Agent 25% | Reasoning 20% | Code 10% | Math 5% | Speed 10% | Lanes 10% | Context 10% | Overhead 10%")
print(f"Missing intelligence benchmarks scored as 50 (neutral). Missing infra scored as 0.\n")

print(f"{'Rank':<5} {'Model':<22} {'Score':<7} {'Intel':<7} {'Agent':<7} {'Reason':<7} {'Code':<7} {'Math':<7} {'Speed':<7} {'Lanes':<7} {'Ctx':<7} {'OH':<7} {'Params':<8} {'Disk':<6}")
print("-" * 130)

for i, r in enumerate(results, 1):
    print(f"{i:<5} {r['model']:<22} {r['total_score']:<7} {r['overall_intel']:<7} {r['agent']:<7} {r['reasoning']:<7} {r['code']:<7} {r['math']:<7} {r['speed']:<7} {r['lanes']:<7} {r['context']:<7} {r['overhead']:<7} {r['params']:<5}B  {r['disk_gb']:<4}GB")

print("-" * 120)
print("\nNote: Nemotron Super scores are estimates based on benchmark data (untested on Spark).")
print("Note: Step 3.7 has no published reasoning/math/code benchmarks — scored as neutral (50).")