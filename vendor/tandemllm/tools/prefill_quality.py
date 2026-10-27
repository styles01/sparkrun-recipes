"""The prefill QUALITY gate: what a prefill path costs the model, rather than how many ulps it moves.

Phase 11 gated the fused prefill on "the logits equal the shipped path's within bf16 rounding" and
the gate turned out to be unmeetable by anything, including the shipped path itself: run against its
OWN other exact inverse -- a `solve_triangular` where the other does forward substitution, one
environment variable apart, both in `engine/gdn.py` -- the shipped path reads 8.2 bf16 ulp at 8k and
writes different text from the eleventh greedy token. Forty-eight recurrent layers amplify any
reordering of the same fp32 that far. A tolerance whose floor is above it cannot decide anything.

So this file asks the question the ulp gate was standing in for: **does the model get worse?** Three
measurements, all of them on the same tokens in one process, and a path passes only if all three
hold.

  1. **Held-out teacher-forced NLL**, on long real prompts of three domains, per length. A prefill
     path that damages the model shows up here first, and the threshold is a delta against the
     shipped path on exactly the same tokens, never an absolute.
  2. **Argmax agreement** over those positions, reported over all of them and restricted to the
     positions where the shipped path's top-2 logit gap is at least 1.0. On a flat distribution over
     248,320 rows the argmax is decided by rounding, so an unrestricted agreement number is mostly a
     statement about the flat positions.
  3. **Greedy-continuation divergence**, PAIRED against the control. For every prompt: how many
     tokens the fused path writes before it differs from the shipped path, and how many the shipped
     path's own alternative inverse writes before it differs from the shipped path. The second is
     the engine's existing internal variance, it is the only honest yardstick there is at a prefill,
     and the comparison is a sign test over the prompts rather than two averages.

    python tools/prefill_quality.py --data bench/longprompts --lens 2048,8192,16384 \
        --nvfp4 $NV --fp8-head $HEAD --greedy 64 --out results/prefill_quality.json

No number in this file is a time. It compares two paths' OUTPUTS, so it may be run beside another
engine on the board; nothing it prints is a claim that needs the board to itself.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402

ARMS = ("ref", "fused", "subst", "dattn", "kvfp8", "v2pre")
ARM_WHAT = {
    "ref": "the shipped path",
    "fused": "the fused Triton prefill pair",
    "subst": "the shipped path with its other exact inverse (the control)",
    # the decode-attention kernel over the bf16 cache (a reordering of the same
    # arithmetic, so the control for the next arm), and the same kernel over an e4m3 cache. Neither
    # touches a prefill chunk of 64 rows or more; `--tail` puts the last tokens through verify-
    # shaped blocks so the NLL and argmax columns read the kernel over the whole context.
    "dattn": "attention by tools/attn_kernels.py below 64 rows, bf16 cache",
    "kvfp8": "the same kernel over an e4m3 KV cache with a scale per (head, token)",
    # the v2 W4A16 kernel with the prefill tile for every projection between the decode
    # band and QWEN38_NVFP4_PREFILL_V2_UNTIL rows (the prefill chunk has to fall inside it).
    "v2pre": "prefill projections on the v2 kernel's prefill tile instead of v1 / unpack+GEMM",
}


def sign_test(worse: int, better: int) -> float:
    """Two-sided exact sign test over the prompts where the two paths disagree.

    `worse` is the number of prompts on which the path under test diverged EARLIER than the control.
    Ties carry no information about direction and are excluded, which is what a sign test is.
    """
    n = worse + better
    if n == 0:
        return 1.0
    k = min(worse, better)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def segments(T: int, chunk: int, tail: int, tail_block: int) -> list[tuple[int, int]]:
    """Prefill chunks over the head of the prompt, then verify-shaped blocks over its last `tail`."""
    head = max(0, T - tail)
    out = [(c0, min(c0 + chunk, head)) for c0 in range(0, head, chunk)]
    out += [(c0, min(c0 + tail_block, T)) for c0 in range(head, T, tail_block)]
    return out


def run_arm(eng, M, G, ids: torch.Tensor, arm: str, chunk: int, greedy: int,
            caches: dict | None = None, tail: int = 0, tail_block: int = 16) -> dict:
    """One prompt through one path: NLL, argmax, the top-2 gap, the last logits, a continuation."""
    M.FUSED["gdnprefill"] = (arm == "fused")
    keep, G.UT_INVERSE = G.UT_INVERSE, (arm != "subst")
    keep_da, M.DECODE_ATTN = M.DECODE_ATTN, arm in ("dattn", "kvfp8")
    keep_kv = eng.kv
    from tools import nvfp4_linear as NL
    keep_pv, NL.PREFILL_V2 = NL.PREFILL_V2, arm == "v2pre"
    if arm == "kvfp8":
        eng.kv = caches["fp8"]
    try:
        eng.reset()
        nll, n = 0.0, 0
        argmax, gap = [], []
        last = None
        start = 0
        T = int(ids.numel())
        with torch.no_grad():
            for c0, c1 in segments(T, chunk, tail, tail_block):
                piece = ids[c0:c1]
                logits = eng.forward(piece, start=start)[0]
                start += int(piece.numel())
                last = logits[-1].float().clone()
                tgt = ids[c0 + 1:c0 + 1 + int(piece.numel())]
                m = int(tgt.numel())
                if m:
                    for r0 in range(0, m, 512):     # fp32 log-softmax over 248,320 rows is 1 GB a
                        sl = logits[r0:min(r0 + 512, m)].float()   # thousand rows; do it in bands
                        lp = F.log_softmax(sl, dim=-1)
                        t = tgt[r0:r0 + sl.shape[0]]
                        nll += float(-lp.gather(1, t[:, None]).sum())
                        top2 = sl.topk(2, dim=-1)
                        argmax.append(top2.indices[:, 0].cpu())
                        gap.append((top2.values[:, 0] - top2.values[:, 1]).cpu())
                        del sl, lp, top2
                    n += m
                del logits
            state = eng.state.S.clone()
            # The functional test: continue greedily from the state this prefill left behind. A
            # logit difference that never changes a token is a difference the engine cannot express.
            cont = []
            tokcur = int(last.argmax())
            pos = T
            for _ in range(greedy):
                cont.append(tokcur)
                lg = eng.forward(torch.tensor([tokcur], device=ids.device), start=pos,
                                 last_only=True)
                tokcur = int(lg.float().reshape(-1).argmax())
                pos += 1
    finally:
        G.UT_INVERSE = keep
        M.FUSED["gdnprefill"] = False
        M.DECODE_ATTN = keep_da
        eng.kv = keep_kv
        NL.PREFILL_V2 = keep_pv
    return {"nll": nll, "n": n, "argmax": torch.cat(argmax) if argmax else torch.empty(0),
            "gap": torch.cat(gap) if gap else torch.empty(0), "last": last, "state": state,
            "cont": cont}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default="bench/longprompts")
    ap.add_argument("--lens", default="2048,8192,16384")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts of each length, 0 = all")
    ap.add_argument("--per-domain", type=int, default=0,
                    help="first N prompts of each DOMAIN at each length, 0 = all")
    ap.add_argument("--arms", default="ref,fused,subst")
    ap.add_argument("--chunk", type=int, default=2048,
                    help="prefill chunk; the same for every arm, so the comparison is one shape")
    ap.add_argument("--greedy", type=int, default=64)
    ap.add_argument("--conf", type=float, default=1.0, help="top-2 gap that counts as confident")
    ap.add_argument("--nll-tol", type=float, default=0.005, help="nats the gate allows")
    ap.add_argument("--gate-arm", default="fused", help="the arm the NLL gate is applied to")
    ap.add_argument("--control", default="subst",
                    help="the arm every other non-ref arm's divergence is paired against")
    ap.add_argument("--tail", type=int, default=0,
                    help="the last N tokens of each prompt go through blocks of --tail-block rows "
                         "instead of prefill chunks, so they are read by the decode-side path")
    ap.add_argument("--tail-block", type=int, default=16)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.nvfp4:
        os.environ["QWEN38_NVFP4"] = a.nvfp4
    if a.fp8_head:
        os.environ["QWEN38_FP8_HEAD"] = a.fp8_head
    os.environ.setdefault("QWEN38_NVFP4_DEQUANT_FROM", "512")

    import engine.model as M
    from engine import gdn as G
    from engine.loader import Weights

    arms = [x for x in a.arms.split(",") if x]
    lens = [int(x) for x in a.lens.split(",")]
    with open(os.path.join(a.data, "manifest.json")) as f:
        manifest = json.load(f)

    cfg = load_config(a.model)
    t0 = time.time()
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    print(f"[load] {time.time() - t0:.1f}s  {w.report()}", flush=True)
    eng = M.Qwen38Engine(cfg, w, max_len=max(lens) + a.greedy + 8, device=a.device)
    caches = {}
    if "kvfp8" in arms:
        caches["fp8"] = M.KVCache(cfg, max(lens) + a.greedy + 8, a.device, fp8=True)

    out: dict = {"data": os.path.abspath(a.data), "sources": manifest["sources"], "arms": arms,
                 "chunk": a.chunk, "greedy": a.greedy, "conf": a.conf, "tail": a.tail,
                 "tail_block": a.tail_block, "gate_arm": a.gate_arm, "control": a.control,
                 "lens": {}}

    for length in lens:
        ids_all = np.load(os.path.join(a.data, f"ids-{length}.npy"))
        meta = manifest["prompts"][str(length)]
        if a.limit:
            ids_all, meta = ids_all[:a.limit], meta[:a.limit]
        if a.per_domain:
            # the set is written domain by domain, so a plain --limit would take one domain only
            seen: dict = {}
            keep = [i for i, m in enumerate(meta)
                    if seen.setdefault(m["domain"], []).append(i) or
                    len(seen[m["domain"]]) <= a.per_domain]
            ids_all, meta = ids_all[keep], [meta[i] for i in keep]
        rows = []
        print(f"\n=== {length} tokens, {len(ids_all)} prompts, chunk {a.chunk} ===", flush=True)
        for i, (row, rm) in enumerate(zip(ids_all, meta)):
            ids = torch.from_numpy(row.astype(np.int64)).to(a.device)
            res = {}
            for arm in arms:
                res[arm] = run_arm(eng, M, G, ids, arm, a.chunk, a.greedy, caches,
                                   a.tail, a.tail_block)
            ref = res["ref"]
            rec = {"domain": rm["domain"], "sha256_16": rm["sha256_16"],
                   "n": ref["n"], "nll": {arm: res[arm]["nll"] for arm in arms}}
            conf = ref["gap"] >= a.conf
            for arm in arms:
                if arm == "ref":
                    continue
                r = res[arm]
                same = (r["argmax"] == ref["argmax"])
                rec[arm] = {
                    "nll_delta": (r["nll"] - ref["nll"]) / max(ref["n"], 1),
                    "agree": float(same.float().mean()),
                    "agree_conf": float(same[conf].float().mean()) if int(conf.sum()) else 1.0,
                    "n_conf": int(conf.sum()),
                    "ulp": float((r["last"] - ref["last"]).abs().max()
                                 / (ref["last"].abs().max() * 2 ** -8 + 1e-30)),
                    "state_rel": float((r["state"] - ref["state"]).abs().max()
                                       / (ref["state"].abs().max() + 1e-30)),
                    "diverge": next((j for j, (x, y) in enumerate(zip(ref["cont"], r["cont"]))
                                     if x != y), a.greedy),
                }
                bad = [k for k in ("nll_delta", "ulp", "state_rel") if rec[arm][k] != rec[arm][k]]
                if bad:
                    print(f"    [!] {arm} prompt {i}: NaN in {bad} -- a gate that reads a NaN as a "
                          f"pass is worse than no gate", flush=True)
            rows.append(rec)
            print(f"  [{i + 1:>2}/{len(ids_all)}] {rm['domain']:<7} nll/tok "
                  + "  ".join(f"{arm} {res[arm]['nll'] / max(ref['n'], 1):.4f}" for arm in arms)
                  + "   diverge "
                  + "  ".join(f"{arm} {rec[arm]['diverge']}" for arm in arms if arm != "ref"),
                  flush=True)
            del res
            torch.cuda.empty_cache()
        out["lens"][str(length)] = rows

    # ---- the tables -------------------------------------------------------------------------
    print("\n\n### held-out NLL per token, and the delta against the shipped path\n")
    print(f"{'tokens':>7}{'domain':>9}{'prompts':>9}{'ref nll':>10}"
          + "".join(f"{arm + ' nll':>11}{arm + ' d':>10}" for arm in arms if arm != "ref"))
    gate_nll = True
    for length in lens:
        rows = out["lens"][str(length)]
        for domain in ("prose", "code", "german", "ALL"):
            sel = rows if domain == "ALL" else [r for r in rows if r["domain"] == domain]
            if not sel:
                continue
            n = sum(r["n"] for r in sel)
            ref = sum(r["nll"]["ref"] for r in sel) / n
            line = f"{length:>7}{domain:>9}{len(sel):>9}{ref:>10.4f}"
            for arm in arms:
                if arm == "ref":
                    continue
                got = sum(r["nll"][arm] for r in sel) / n
                line += f"{got:>11.4f}{got - ref:>+10.4f}"
                if arm == a.gate_arm and not (got - ref <= a.nll_tol):
                    gate_nll = False
            print(line)

    print("\n### argmax agreement with the shipped path, and the last position's distance\n")
    print(f"{'tokens':>7}{'arm':>7}{'agree':>9}{'agree conf':>12}{'conf pos':>10}"
          f"{'ulp p50':>9}{'ulp max':>9}{'state rel p50':>15}")
    for length in lens:
        rows = out["lens"][str(length)]
        for arm in arms:
            if arm == "ref":
                continue
            ag = [r[arm]["agree"] for r in rows]
            agc = [r[arm]["agree_conf"] for r in rows]
            ulp = [r[arm]["ulp"] for r in rows]
            st = [r[arm]["state_rel"] for r in rows]
            print(f"{length:>7}{arm:>7}{statistics.mean(ag):>9.4f}{statistics.mean(agc):>12.4f}"
                  f"{sum(r[arm]['n_conf'] for r in rows):>10}{statistics.median(ulp):>9.2f}"
                  f"{max(ulp):>9.2f}{statistics.median(st):>15.2e}")

    print("\n### greedy divergence: tokens written before the path differs from the shipped one\n")
    print(f"{'tokens':>7}{'arm':>7}{'min':>6}{'p50':>6}{'mean':>7}{'max':>6}{'all agree':>11}"
          f"   paired against the control")
    gate_div = True
    for length in lens:
        rows = out["lens"][str(length)]
        div = {arm: [r[arm]["diverge"] for r in rows] for arm in arms if arm != "ref"}
        for arm, d in div.items():
            note = ""
            if arm != a.control and a.control in div:
                c = div[a.control]
                worse = sum(1 for x, y in zip(d, c) if x < y)
                better = sum(1 for x, y in zip(d, c) if x > y)
                p = sign_test(worse, better)
                ok = not (worse > better and p < 0.05)
                gate_div = gate_div and ok
                note = (f"earlier on {worse}, later on {better}, same on "
                        f"{len(d) - worse - better}; sign test p = {p:.3f}  "
                        f"{'not worse' if ok else 'WORSE'}")
            print(f"{length:>7}{arm:>7}{min(d):>6}{int(statistics.median(d)):>6}"
                  f"{statistics.mean(d):>7.1f}{max(d):>6}"
                  f"{sum(1 for x in d if x >= a.greedy):>4}/{len(d):<6}   {note}")

    out["gate"] = {"nll": gate_nll, "divergence": gate_div, "nll_tol": a.nll_tol}
    print(f"\nGATE  held-out NLL delta of {a.gate_arm} at most {a.nll_tol:g} nats: "
          f"{'PASS' if gate_nll else 'FAIL'}")
    print(f"GATE  greedy divergence no worse than the control ({a.control}): "
          f"{'PASS' if gate_div else 'FAIL'}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[out] {a.out}")
    sys.exit(0 if (gate_nll and gate_div) else 1)


if __name__ == "__main__":
    main()
