#!/usr/bin/env python3
"""One JSON per quant from the texts the exllamav3 eval tools print, plus the loss deltas against
the BF16 reference numbers PREP recorded. Used twice: with --label bf16 in PREP (reference only)
and per quant in CONVERT. Missing inputs leave their fields null; the job never fails here.

  ppl.py         " -- Evaluation perplexity: 6.1234"    -> nll = ln(ppl)
  model_diff.py  " -- A perplexity: ...", " -- B perplexity: ...", "KL divergence" lines,
                 " -- Top-K agreement, A vs B:" then "      K = 1: 0.9876"
  qbench         results JSON: kld mean/median/p90 of the EXL3 row and the noise floor row
"""
import argparse, json, math, os, re


def read(path):
    return open(path, encoding="utf-8", errors="replace").read() if path and os.path.exists(path) else ""


def ppl_of(text):
    m = re.findall(r"perplexity:\s*([0-9.]+)", text, re.I)   # ppl.py prints " -- Perplexity: 2.70"
    return float(m[-1]) if m else None


def model_diff(text):
    out = {}
    a = re.search(r"-- A perplexity:\s*([0-9.]+)", text)
    b = re.search(r"-- B perplexity:\s*([0-9.]+)", text)
    if a: out["ppl_a"] = float(a.group(1))
    if b: out["ppl_b"] = float(b.group(1))
    for line in text.splitlines():
        if "KL" in line and ":" in line:
            m = re.search(r"([0-9]*\.[0-9]+(?:e-?[0-9]+)?)\s*$", line.strip())
            if m:
                key = re.sub(r"[^a-z0-9]+", "_", line.split(":")[0].strip().lower()).strip("_")
                out[key] = float(m.group(1))
    sec = text.split("Top-K agreement, A vs B:")
    if len(sec) > 1:
        agree = {}
        for line in sec[1].splitlines()[1:8]:
            m = re.match(r"\s*K = (\d+):\s*([0-9.]+)", line)
            if m: agree[f"top{m.group(1)}"] = float(m.group(2))
            elif line.strip() == "" or not line.startswith(" "): break
        out["topk_agreement"] = agree
    return out


def qbench(path):
    if not path or not os.path.exists(path): return None
    rows = json.load(open(path))
    rows = rows.get("results", rows) if isinstance(rows, dict) else rows
    out = {}
    for r in rows:
        g = r.get("group")
        key = {"EXL3": "exl3", "noise_floor": "noise_floor", "reference": "reference"}.get(g)
        if not key: continue
        out[key] = {k: r.get(k) for k in ("label", "ppl", "kld", "kld_median", "kld_p90") if k in r}
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ppl-rows"); ap.add_argument("--ppl-trace"); ap.add_argument("--model-diff")
    ap.add_argument("--qbench"); ap.add_argument("--size"); ap.add_argument("--ref")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = {"label": a.label, "ppl_heldout_rows": ppl_of(read(a.ppl_rows)), "ppl_heldout_trace": ppl_of(read(a.ppl_trace))}
    for k in ("ppl_heldout_rows", "ppl_heldout_trace"):
        s[k.replace("ppl", "nll")] = math.log(s[k]) if s[k] else None
    if a.model_diff: s["wikitext2"] = model_diff(read(a.model_diff))
    if a.qbench: s["heldout_trace_kld"] = qbench(a.qbench)
    if a.size and os.path.exists(a.size): s["size"] = json.load(open(a.size))
    if a.ref and os.path.exists(a.ref):
        ref = json.load(open(a.ref))
        s["reference"] = {k: ref.get(k) for k in ("nll_heldout_rows", "nll_heldout_trace")}
        for k in ("nll_heldout_rows", "nll_heldout_trace"):
            if s.get(k) is not None and ref.get(k) is not None:
                s["delta_" + k] = s[k] - ref[k]
    json.dump(s, open(a.out, "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
