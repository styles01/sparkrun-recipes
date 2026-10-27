#!/usr/bin/env python3
"""Write a qbench project (exllamav3 eval/qbench.py) that scores one EXL3 quant against the BF16
checkpoint on the held-out in-domain trace from eval/qbench_prompts.py. The reference runs through
exllamav3's own module-streamed pass over the unquantized checkpoint (the same loader sc_measure
uses), so no transformers class for Qwen3_5 is needed; pass --ref-engine transformers to switch.
Paths in the YAML are absolute. The logit cache lives next to the project file."""
import argparse, os, yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True)
    ap.add_argument("--ref", required=True, help="BF16 model directory")
    ap.add_argument("--ref-engine", default="exllamav3", choices=["exllamav3", "transformers"])
    ap.add_argument("--trace", required=True, help="qbench_prompts_gen.json")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, help="project YAML")
    ap.add_argument("--results", required=True, help="results JSON the run writes")
    a = ap.parse_args()
    ref = {"label": "BF16", "group": "reference", "engine": a.ref_engine, "source": os.path.abspath(a.ref)}
    if a.ref_engine == "transformers":
        ref["options"] = {"streaming": True}
    project = {
        "title": f"Qwen3.8-27B EXL3 {a.label} vs BF16 (held-out trace)",
        "test_trace": os.path.abspath(a.trace),
        "logit_cache": {"dir": os.path.join(os.path.dirname(os.path.abspath(a.out)), "_logit_cache"), "max_size_gb": 200},
        "models": [ref, {"label": a.label, "group": "EXL3", "engine": "exllamav3", "source": os.path.abspath(a.quant)}],
        "output": {"results": os.path.abspath(a.results)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    yaml.safe_dump(project, open(a.out, "w"), sort_keys=False)
    print(open(a.out).read())


if __name__ == "__main__":
    main()
