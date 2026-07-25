#!/usr/bin/env python3
"""
Spark LLM Weighted Score Calculator v4 — Blended AA + HF Intelligence Data
Blends Artificial Analysis Intelligence Index with HuggingFace model card benchmarks.
"""

# ============================================================
# BLENDED INTELLIGENCE DATA (AA + HF)
# Where both sources have a score, we average them.
# Where only one has data, we use that.
# Where neither has data, we use neutral 50.
# ============================================================

models = {
    "Step 3.7 Flash": {
        "total_params": 198, "active_params": 11,
        # AA Intelligence Index (composite 0-100)
        "aa_index": 30,
        # HF benchmarks (from model cards/blog)
        "hf_claweval": 67.1, "hf_swe_bench": 56.3, "hf_toolathlon": 49.5, "hf_terminal_bench": 59.6,
        "hf_mmlu_pro": None, "hf_gpqa": None, "hf_hle": None, "hf_aime25": None,
        "hf_hmmt": None, "hf_scicode": None, "hf_humaneval": None,
        # AA individual evals (percentages)
        "aa_gdpval": 18, "aa_tau_banking": 11, "aa_terminal_bench": 26, "aa_scicode": 36,
        "aa_hle": 18, "aa_gpqa_diamond": 78, "aa_critpt": 2, "aa_omniscience": 26, "aa_lcr": 64,
        # HF model card scores (from benchmark image + text)
        "hf_gdpval": 45.8, "hf_hle_with_tool": 47.2, "hf_simplevqa": 79.2, "hf_v_star": 95.3,
        # Infrastructure (our tested configs)
        "tok_s": 28.1, "lanes": 2, "ctx_per_lane": 200, "free_ram_gb": 0.5,
        "disk_gb": 105, "runtime": "llama.cpp IQ4_XS",
    },
    "DS4-Flash": {
        "total_params": 159, "active_params": 11,
        "aa_index": 40,
        "hf_claweval": 57.8, "hf_swe_bench": 52.3, "hf_toolathlon": 43.5, "hf_terminal_bench": 56.6,
        "hf_mmlu_pro": 86.4, "hf_gpqa": 87.4, "hf_hle": 29.4, "hf_aime25": None,
        "hf_hmmt": 91.9, "hf_scicode": None, "hf_humaneval": 69.5,
        "aa_gdpval": 34, "aa_tau_banking": 23, "aa_terminal_bench": 62, "aa_scicode": 45,
        "aa_hle": 32, "aa_gpqa_diamond": 89, "aa_critpt": 7, "aa_omniscience": 37, "aa_lcr": 63,
        "tok_s": 21, "lanes": 2, "ctx_per_lane": 128, "free_ram_gb": 22,
        "disk_gb": 227, "runtime": "vLLM MTP k=2",
    },
    "Qwen 122B": {
        "total_params": 122, "active_params": 10,
        "aa_index": 32,
        # Card publishes SWE-bench Verified (72.0), not SWE-Bench PRO — using Verified here
        "hf_claweval": None, "hf_swe_bench": 72.0, "hf_toolathlon": None, "hf_terminal_bench": 49.4,
        "hf_mmlu_pro": 86.7, "hf_gpqa": 86.6, "hf_hle": 25.3, "hf_aime25": None,
        "hf_hmmt": 91.4, "hf_scicode": 42.0, "hf_humaneval": None,
        "aa_gdpval": 24, "aa_tau_banking": 14, "aa_terminal_bench": 48, "aa_scicode": 42,
        "aa_hle": 23, "aa_gpqa_diamond": 86, "aa_critpt": 1, "aa_omniscience": 25, "aa_lcr": 67,
        "tok_s": 29.6, "lanes": 3, "ctx_per_lane": 150, "free_ram_gb": 16,
        "disk_gb": 67, "runtime": "vLLM DFlash n=4",
    },
    "Nemotron Super 120B": {
        "total_params": 120, "active_params": 12,
        "aa_index": 25,
        "hf_claweval": None, "hf_swe_bench": None, "hf_toolathlon": None, "hf_terminal_bench": 31.0,
        "hf_mmlu_pro": 83.73, "hf_gpqa": 79.23, "hf_hle": 18.26, "hf_aime25": 90.21,
        "hf_hmmt": 93.67, "hf_scicode": 42.05, "hf_humaneval": None,
        "aa_gdpval": 8, "aa_tau_banking": 10, "aa_terminal_bench": 39, "aa_scicode": 38,
        "aa_hle": 19, "aa_gpqa_diamond": 80, "aa_critpt": 3, "aa_omniscience": 24, "aa_lcr": 60,
        "tok_s": None, "lanes": None, "ctx_per_lane": None, "free_ram_gb": None,
        "disk_gb": 67, "runtime": "vLLM (untested)",
    },
    "Puzzle-75B": {
        "total_params": 75, "active_params": 9.3,
        "aa_index": None,  # Not on AA
        "hf_claweval": None, "hf_swe_bench": None, "hf_toolathlon": None, "hf_terminal_bench": 24.0,
        "hf_mmlu_pro": 82.4, "hf_gpqa": 78.6, "hf_hle": 16.5, "hf_aime25": 89.7,
        "hf_hmmt": 93.4, "hf_scicode": 40.6, "hf_humaneval": None,
        "aa_gdpval": None, "aa_tau_banking": None, "aa_terminal_bench": None, "aa_scicode": None,
        "aa_hle": None, "aa_gpqa_diamond": None, "aa_critpt": None, "aa_omniscience": None, "aa_lcr": None,
        # Infrastructure (OUR TESTED CONFIG — Jul 20, NGC 26.06-py3, GMU 0.65, 300K, 3 seqs)
        "tok_s": 39.7, "lanes": 3, "ctx_per_lane": 300, "free_ram_gb": 37,
        "disk_gb": 50, "runtime": "vLLM NGC 26.06-py3 MTP k=3",
    },
    "Qwen 35B": {
        "total_params": 35, "active_params": 3,
        "aa_index": 29,
        "hf_claweval": None, "hf_swe_bench": None, "hf_toolathlon": None, "hf_terminal_bench": 40.5,
        "hf_mmlu_pro": 85.3, "hf_gpqa": 84.2, "hf_hle": 22.4, "hf_aime25": None,
        "hf_hmmt": 89.0, "hf_scicode": None, "hf_humaneval": None,
        "aa_gdpval": 15, "aa_tau_banking": 5, "aa_terminal_bench": 41, "aa_scicode": 38,
        "aa_hle": 20, "aa_gpqa_diamond": 85, "aa_critpt": 1, "aa_omniscience": 10, "aa_lcr": 63,
        "tok_s": 102.8, "lanes": 6, "ctx_per_lane": 256, "free_ram_gb": 45,
        "disk_gb": 22, "runtime": "vLLM MTP k=3",
    },
}

# ============================================================
# WEIGHTS — Intelligence 85% | Infrastructure 15%
# Lanes 11% — concurrency is the dominant infra metric for multi-agent Spark use
# ============================================================
WEIGHTS = {
    "agent": 0.3188,    # Agent/tool use      (85% intel total)
    "reasoning": 0.2656,# Reasoning
    "code": 0.1594,     # Code
    "math": 0.1062,     # Math
    "speed": 0.02,      # tok/s
    "lanes": 0.11,      # concurrent agents — dominant infra metric
    "context": 0.01,    # context per lane
    "overhead": 0.01,   # free RAM
}

MISSING_DEFAULT = 50

def blend(hf_val, aa_val):
    """Blend HF and AA scores. Average if both exist, use whichever exists."""
    vals = [v for v in [hf_val, aa_val] if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

def avg_scores(*scores):
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)

# ============================================================
# CALCULATE
# ============================================================

all_tok = [m["tok_s"] for m in models.values() if m["tok_s"] is not None]
all_lanes = [m["lanes"] for m in models.values() if m["lanes"] is not None]
all_ctx = [m["ctx_per_lane"] for m in models.values() if m["ctx_per_lane"] is not None]
all_ram = [m["free_ram_gb"] for m in models.values() if m["free_ram_gb"] is not None]

import math

def norm_log(value, mn, mx, floor=0):
    """Logarithmic normalization — diminishing returns for higher values.
    floor sets minimum score. Default floor=0 means min value scores 0.
    Use floor=1 for ~20% minimum, floor=2 for ~35% minimum."""
    if value is None: return None
    if mx == mn: return 100
    if value <= 0:
        return 0
    log_val = math.log(value + 1)
    log_mn = math.log(mn + 1) if mn > 0 else 0
    log_mx = math.log(mx + 1)
    if log_mx == log_mn:
        return 100
    score = (log_val - log_mn) / (log_mx - log_mn) * 100
    # Apply floor — minimum value gets floor% instead of 0
    if floor > 0:
        score = floor + score * (100 - floor) / 100
    return max(0, min(100, score))

results = []

for name, m in models.items():
    # === AGENT === (blend HF + AA agent scores)
    agent_scores = [
        m["hf_claweval"],
        blend(m["hf_terminal_bench"], m["aa_terminal_bench"]),
        blend(m.get("hf_gdpval"), m["aa_gdpval"]),
        m["aa_tau_banking"],
        m["hf_swe_bench"],
        m["hf_toolathlon"],
    ]
    agent_raw = avg_scores(*agent_scores)
    agent_score = agent_raw if agent_raw is not None else MISSING_DEFAULT

    # === REASONING === (blend HF + AA)
    reasoning_scores = [
        blend(m["hf_gpqa"], m["aa_gpqa_diamond"]),
        blend(m["hf_hle"], m["aa_hle"]),
        m["hf_aime25"],
        m["aa_omniscience"],
        m["aa_lcr"],  # long context reasoning
    ]
    reasoning_raw = avg_scores(*reasoning_scores)
    reasoning_score = reasoning_raw if reasoning_raw is not None else MISSING_DEFAULT

    # === CODE === (blend HF + AA)
    code_scores = [
        blend(m["hf_scicode"], m["aa_scicode"]),
        m["hf_humaneval"],
    ]
    code_raw = avg_scores(*code_scores)
    code_score = code_raw if code_raw is not None else MISSING_DEFAULT

    # === MATH === (HF only — AA doesn't test HMMT)
    math_score = m["hf_hmmt"] if m["hf_hmmt"] is not None else MISSING_DEFAULT

    # === OVERALL INTELLIGENCE === (weighted blend of 4 categories)
    intel_weight = WEIGHTS["agent"] + WEIGHTS["reasoning"] + WEIGHTS["code"] + WEIGHTS["math"]
    overall_intel = (
        agent_score * WEIGHTS["agent"] +
        reasoning_score * WEIGHTS["reasoning"] +
        code_score * WEIGHTS["code"] +
        math_score * WEIGHTS["math"]
    ) / intel_weight

    # Also blend in AA composite index if available (gives it a 25% influence on overall intel)
    if m["aa_index"] is not None:
        overall_intel = (overall_intel * 0.75) + (m["aa_index"] * 0.25)
    else:
        overall_intel = overall_intel * 0.75 + MISSING_DEFAULT * 0.25

    # === INFRASTRUCTURE ===
    speed_score = norm_log(m["tok_s"], min(all_tok), max(all_tok), floor=15) if m["tok_s"] else 0
    lanes_score = norm_log(m["lanes"], min(all_lanes), max(all_lanes), floor=15) if m["lanes"] else 0
    ctx_score = norm_log(m["ctx_per_lane"], min(all_ctx), max(all_ctx), floor=15) if m["ctx_per_lane"] else 0
    overhead_score = norm_log(m["free_ram_gb"], min(all_ram), max(all_ram), floor=15) if m["free_ram_gb"] is not None else 0

    # === TOTAL ===
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
        "aa_index": m["aa_index"],
    })

results.sort(key=lambda x: x["total_score"], reverse=True)

print("=" * 140)
print("SPARK LLM WEIGHTED SCORE v4 — BLENDED AA + HF DATA")
print("=" * 140)
print(f"\nWeights: Agent 31.88% | Reasoning 26.56% | Code 15.94% | Math 10.62% | Speed 2.0% | Lanes 11.0% | Context 1.0% | Overhead 1.0%  (Intel 85% / Infra 15%)")
print(f"Intelligence scores blended from Artificial Analysis + HuggingFace model cards. Missing = neutral 50.\n")

print(f"{'Rank':<5} {'Model':<22} {'Score':<7} {'Intel':<7} {'AA Idx':<7} {'Agent':<7} {'Reason':<7} {'Code':<7} {'Math':<7} {'Speed':<7} {'Lanes':<7} {'Ctx':<7} {'OH':<7} {'Params':<8} {'Disk':<6}")
print("-" * 140)

for i, r in enumerate(results, 1):
    aa = str(r['aa_index']) if r['aa_index'] else "N/A"
    print(f"{i:<5} {r['model']:<22} {r['total_score']:<7} {r['overall_intel']:<7} {aa:<7} {r['agent']:<7} {r['reasoning']:<7} {r['code']:<7} {r['math']:<7} {r['speed']:<7} {r['lanes']:<7} {r['context']:<7} {r['overhead']:<7} {r['params']:<5}B  {r['disk_gb']:<4}GB")

print("-" * 140)