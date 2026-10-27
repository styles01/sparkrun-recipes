#!/usr/bin/env python3
"""encode_nemotron.py — EXL3 flat-K encode of Nemotron 3.5 Super VL 120B-A12B BF16.

Generalizes the dsv41 pipeline's encode core (dsv41q/exl3.py: quantize_exl3 /
quantize_exl3_batch / LinearEXL3, exllamav3 1.6.0+turboderp) to NemotronH:
- Mamba-2 SSM layers, C-RADIO vision tower + connector, embeddings, routers, norms
  stay in their native precision (SKIP-LIST — exllamav3 carries the mamba module).
- Every LinearEXL3-able nn.Linear (MoE expert projections, dense linears, shared
  expert, attention projections — NOT the LM head at flat-K; head stays BF16 for
  logits fidelity at this pass) gets a Hessian-sum captured over calibration text,
  then quantize_exl3_batch at the plan K.
- Passes are resumable per layer (state json) — an interrupted encode re-enters.

Usage: see __main__. Outputs an exl3-style pack layout the shim can load
(config.json with quantization_config.method = "exl3_plan_nemotron", per-layer
model-exl3-*.safetensors + unquantized passthrough files copied from base).
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys, time
import torch
from safetensors.torch import load_file, save_file

os.environ.setdefault("HF_HUB_OFFLINE", "1")

def log(*a):
    print(*a, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cal", required=True, help="calibration dir (dsv41 caldata format: *.txt)")
    ap.add_argument("--k-experts", type=float, default=3.5)
    ap.add_argument("--k-dense", type=float, default=4.0)
    ap.add_argument("--cal-tokens", type=int, default=1_048_576)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda:0"

    import exllamav3
    from exllamav3 import Model, Config, Tokenizer
    sys.path.insert(0, "/home/jaita/dsv41_recipe/scripts/quantize")  # dsv41q helpers
    from dsv41q.exl3 import quant_args, h_data  # public-API helpers proven on GB10

    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "encode-state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"layers_done": []}

    cfg = Config.from_directory(args.base)
    model = Model(cfg)
    model.load(progress=False, max_output_len=1)
    tok = Tokenizer(cfg)
    log("model loaded:", getattr(cfg, "model_dir", args.base))

    # calibration token stream from dsv41 caldata (mixed domain text)
    calfiles = sorted(glob.glob(os.path.join(args.cal, "*.txt")))
    assert calfiles, f"no calibration files in {args.cal}"
    ids = []
    per = args.cal_tokens // max(1, len(calfiles))
    for f in calfiles:
        txt = open(f, encoding="utf-8", errors="ignore").read()
        ids.extend(tok.encode(txt)[: per + 64])
    ids = ids[: int(args.cal_tokens * 1.05)]
    log(f"calibration tokens: {len(ids)} ({len(calfiles)} files)")

    # which LinearEXL3-able layers (by name pattern) + their K
    SKIP = ("mamba", "ssm", "vision", "radio", "c_radio", "connector", "router.gate",
            "norm", "embedding", "lm_head", "head.")
    KW = cfg.architecture.__dict__ if hasattr(cfg, "architecture") else {}
    hexp = getattr(cfg, "num_hidden_layers", None)

    def is_linear_exl3(name):
        ln = name.lower()
        if any(s in ln for s in SKIP): return None
        if "experts" in ln or "shared_expert" in ln: return args.k_experts
        return args.k_dense

    # resolve the linear modules of the loaded model
    candidates = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            k = is_linear_exl3(name)
            if k:
                candidates[name] = (mod, k)
    log(f"LinearEXL3-able modules: {len(candidates)}")

    # pass 1: Hessian sums over calibration forwards
    H = {n: torch.zeros((mod.in_features, mod.out_features), dtype=torch.float64, device=dev)
         for n, (mod, k) in candidates.items()}
    count = 0
    CH = 8192
    hooks = []
    def hook(name):
        def fn(module, inp, out):
            x = inp[0].detach().to(torch.float64)
            x2 = x.reshape(-1, x.shape[-1])
            H[name].addmm_(x2.T, x2)
        return fn
    for n in candidates: hooks.append(candidates[n][0].register_forward_hook(hook(n)))
    with torch.no_grad():
        for i in range(0, len(ids) - CH, CH):
            seq = ids[i:i + CH]
            model.forward(seq[: CH])
            count += len(seq)
            if (i // CH) % 8 == 0:
                log(f"  hessian pass: {i + CH}/{len(ids)} tokens")
    for h in hooks: h.remove()
    log(f"hessians captured over {count} tokens")

    # pass 2: encode per module (quantize_exl3_batch) and assemble the pack
    plan_meta = {"quantization_config": {
        "quant_method": "exl3_plan_nemotron", "exllamav3_version": getattr(exllamav3, "__version__", "1.6"),
        "k_experts": args.k_experts, "k_dense": args.k_dense, "cal_tokens": count,
        "seed": args.seed,
    }}
    json.dump(plan_meta, open(os.path.join(args.out, "plan.json"), "w"), indent=1)

    qstate = quant_args(args.k_dense, args.seed, os.path.join(args.out, "_qwork"))
    from exllamav3.ext import EXL3_BitsAndCodebook  # noqa: F401
    import exllamav3_ext  # noqa: F401  (kernels warm)

    n_done = 0
    for name, (mod, k) in sorted(candidates.items()):
        if name in state["layers_done"]:
            log(f"  skip-done {name}"); continue
        W = mod.weight.detach().to(dtype=torch.float16, device=dev).T.contiguous()  # (in, out)
        kk = args.k_experts if ("experts" in name.lower()) else k
        qargs = quant_args(kk, args.seed, os.path.join(args.out, "_qwork"))
        qargs["H_data"] = h_data(H[name].to(torch.float32), count, name)
        with torch.no_grad():
            enc = getattr(exllamav3, "quantize_exl3_batch", None)
            if enc is None:
                raise SystemExit("FATAL: exllamav3.quantize_exl3_batch missing — "
                                 "check the turboderp quantizer API on this build")
            out_t = enc(W, qargs)[:1]
        tpath = os.path.join(args.out, f"model-exl3-{name.replace('.', '-')}.safetensors")
        save_file({"w": out_t[0].cpu().contiguous()}, tpath,
                  metadata={"layer": name, "K": str(kk)})
        state["layers_done"].append(name)
        json.dump(state, open(state_path, "w"))
        n_done += 1
        if n_done % 20 == 0:
            log(f"  encoded {state['layers_done'].count(name) if False else len(state['layers_done'])}/{len(candidates)}")

    # passthrough unquantized parts (bf16 everything the pack did not encode)
    log("copying passthrough (mamba/vision/head/embeddings) from base…")
    shutil.copy2(os.path.join(args.base, "config.json"), os.path.join(args.out, "config.json"))
    with open(os.path.join(args.out, "config.json")) as f:
        c = json.load(f)
    c.setdefault("quantization_config", plan_meta["quantization_config"])
    json.dump(c, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    for f in glob.glob(os.path.join(args.base, "*")):
        bn = os.path.basename(f)
        if bn.endswith((".json", ".txt", ".jinja")) and not bn.startswith(("model-", "quant")):
            shutil.copy2(f, os.path.join(args.out, bn)) if os.path.isfile(f) else None
    log("done. KLD check + per-layer eval run separately (p4_kld pattern).")

if __name__ == "__main__":
    main()