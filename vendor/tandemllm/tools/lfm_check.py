"""ENG-287 / SPD-107: LFM2.5-24B-A2B's correctness ladder and its speed numbers, one model a process.

    PY=<engine python>; PYTHONPATH=<repo>:$PYTHONPATH
    $PY tools/lfm_check.py engine --model <dir> --out run.pt      greedy decode of the prompts below on
                                                                   the engine (prefill, then the captured
                                                                   decode step), every step's logits kept
    $PY tools/lfm_check.py hf --model <bf16 dir> --inp run.pt     rung b: transformers' Lfm2MoeForCausalLM
                                                                   on the same tokens, teacher-forced
    <exllamav3 python> tools/lfm_check.py exl3 --model <pack> --inp run.pt
                                                                   rung c: exllamav3's own forward
    <exllamav3 python> tools/lfm_check.py xcheck --pack <pack>    our trellis decode against LinearEXL3's
    $PY tools/lfm_check.py bench --model <dir>                    single-stream decode tok/s, graph on/off
    $PY tools/lfm_check.py staircase --model <dir>                verify ms for 1..16 rows at 1k and 8k
                                                                   context, and the experts a block reads

A comparison reports, over every decode position: the argmax agreement, the positions where the two
disagree and whether each is a near tie (the reference's margin between the two tokens under 8 bf16
steps of its largest logit), and the logit difference (max and mean |diff|, relative to the largest
logit). The reference runs in its own process because only one large model is on the board at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    "Explain in three short paragraphs why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Give me a packing list for a weekend hiking trip in the Alps in October.",
    "Translate into German: The meeting is moved to Thursday at ten, please bring the report.",
    "A train leaves at 14:35 and arrives at 17:10. How long is the trip? Show the steps.",
    "Summarise the plot of Romeo and Juliet in five sentences.",
    "List five differences between TCP and UDP as a table.",
]


def _ids(tok, prompt: str) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def _engine(model: str, max_len: int):
    from engine.models.lfm2_moe.engine import Lfm2MoeEngine
    from engine.models.lfm2_moe.load import load
    t0 = time.time()
    idx, W, experts, timing = load(model, "cuda", log=None)
    eng = Lfm2MoeEngine(idx.cfg, W, experts, device="cuda", max_len=max_len)
    torch.cuda.synchronize()
    return idx, eng, {"load_s": round(time.time() - t0, 1), "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                      **timing}


def cmd_engine(a) -> None:
    from transformers import AutoTokenizer

    from engine.models.lfm2_moe.engine import DecodeGraph
    tok = AutoTokenizer.from_pretrained(a.model)
    idx, eng, info = _engine(a.model, a.max_len)
    print(json.dumps({"load": info, "notes": idx.notes}), flush=True)
    out = {"model": a.model, "tokens": a.tokens, "runs": [], "info": info}
    forced = torch.load(a.force, weights_only=False)["runs"] if a.force else None
    for j, p in enumerate(PROMPTS[:a.prompts]):
        ids = forced[j]["ids"] if forced else _ids(tok, p)
        st = eng.new_state(a.max_len)
        lg = eng.prefill(torch.tensor(ids), st)
        rows = [lg[0].to(torch.bfloat16).cpu()]
        gen = [forced[j]["gen"][0] if forced else int(lg[0].argmax())]
        dg = DecodeGraph(eng, st) if a.graph == "on" else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.tokens - 1):
            if dg is not None:
                lg = dg(gen[-1])
            else:
                eng.stage(gen[-1], st)
                lg = eng.step(st)
                st.length += 1
            rows.append(lg[0].to(torch.bfloat16).cpu())
            gen.append(int(lg[0].argmax()))
            if forced:
                gen[-1] = forced[j]["gen"][len(gen) - 1]           # teacher-forced: the other run's tokens
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        r = {"prompt": p, "ids": ids, "gen": gen, "decode_tok_s": round((a.tokens - 1) / dt, 2),
             "text": tok.decode(gen)}
        out["runs"].append({**r, "logits": torch.stack(rows)})
        print(json.dumps({k: r[k] for k in ("prompt", "decode_tok_s")} | {"text": r["text"][:160]}), flush=True)
    torch.save(out, a.out)


def _compare(run: dict, ref_logits: list) -> dict:
    agree = tot = near = other = 0
    maxrel, mean = 0.0, []
    bad = []
    for r, ref in zip(run["runs"], ref_logits):
        eng = r["logits"].float()
        ref = ref.float()
        scale = ref.abs().max(-1).values
        d = (eng - ref).abs()
        maxrel = max(maxrel, float((d.max(-1).values / scale).max()))
        mean.append(float((d.mean(-1) / scale).mean()))
        ea, ra = eng.argmax(-1), ref.argmax(-1)
        for i in range(eng.shape[0]):
            tot += 1
            if ea[i] == ra[i]:
                agree += 1
                continue
            margin = float(ref[i, ra[i]] - ref[i, ea[i]])
            tie = margin < 8 * 2 ** -7 * float(scale[i])
            near += tie
            other += not tie
            if len(bad) < 12:
                bad.append({"prompt": r["prompt"][:40], "pos": i, "margin": round(margin, 4), "near_tie": bool(tie)})
    return {"positions": tot, "argmax_agree": agree, "agree_pct": round(100 * agree / max(tot, 1), 2),
            "near_tie": near, "other": other, "max_rel_diff": round(maxrel, 4),
            "mean_rel_diff": round(sum(mean) / max(len(mean), 1), 5), "disagreements": bad}


def cmd_hf(a) -> None:
    from transformers import AutoModelForCausalLM
    run = torch.load(a.inp, weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    refs = []
    with torch.no_grad():
        for r in run["runs"]:
            seq = torch.tensor([r["ids"] + r["gen"][:-1]], device="cuda")
            lg = model(seq).logits[0, len(r["ids"]) - 1:].float().cpu()
            refs.append(lg)
    if a.save:
        torch.save([r.to(torch.bfloat16) for r in refs], a.save)
    res = {"rung": "b", "reference": "transformers Lfm2MoeForCausalLM bf16 sdpa", **_compare(run, refs)}
    print(json.dumps(res, indent=1), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_exl3(a) -> None:
    from exllamav3 import Config, Model
    run = torch.load(a.inp, weights_only=False)
    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    model.load(max_output_size=2048, max_output_factor=5)
    refs = []
    with torch.inference_mode():
        for r in run["runs"]:
            seq = torch.tensor([r["ids"] + r["gen"][:-1]])
            lg = model.forward(seq, {"attn_mode": "flash_attn_nc"})
            lg = lg["logits"] if isinstance(lg, dict) else lg
            refs.append(lg[0, len(r["ids"]) - 1:, :config.vocab_size].float().cpu())
    res = {"rung": "c", "reference": "exllamav3 1.6.0 forward (flash_attn_nc)", **_compare(run, refs)}
    print(json.dumps(res, indent=1), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_cmp(a) -> None:
    """Two runs on the same tokens: `--a` an engine run, `--b` an engine run or a reference's saved logits
    (hf/exl3 --save): the comparison of `_compare` with b as the reference (a quality figure when b is
    BF16 and a is a pack)."""
    run = torch.load(a.a, weights_only=False)
    b = torch.load(a.b, weights_only=False)
    refs = [r["logits"] for r in b["runs"]] if isinstance(b, dict) else b
    res = _compare(run, refs)
    kls = []
    for r, ref in zip(run["runs"], refs):
        p = torch.log_softmax(ref.float(), -1)
        q = torch.log_softmax(r["logits"].float(), -1)
        kls.append((p.exp() * (p - q)).sum(-1))
    k = torch.cat(kls)
    res["kl_ref_to_a"] = {"mean": round(float(k.mean()), 5), "median": round(float(k.median()), 5),
                          "p90": round(float(k.quantile(0.9)), 4)}
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_xcheck(a) -> None:
    from exllamav3.modules.quant.exl3 import LinearEXL3

    from engine.models.lfm2_moe.pack import PackIndex
    from tools import exl3_linear as X
    idx = PackIndex.open(a.pack)
    idx.audit()
    c = idx.cfg
    la, lc = c.attention_layers[0], c.conv_layers[0]
    bases = [f"layers.{li}.feed_forward.experts.{e}.{w}" for li, e in ((2, 0), (20, 31), (39, 63)) for w in ("w1", "w3", "w2")]
    bases += [f"layers.{lc}.conv.in_proj", f"layers.{lc}.conv.out_proj", f"layers.{la}.self_attn.q_proj",
              f"layers.{la}.self_attn.k_proj", f"layers.{la}.self_attn.out_proj", "layers.0.feed_forward.w1",
              "layers.0.feed_forward.w2"]
    if "lm_head.trellis" in idx.tensors:
        bases.append("lm_head")
    rows = []
    for b in bases:
        if idx.formats.get(b) != "exl3":
            continue
        blk = idx.exl3(b, "cuda")
        if b == "lm_head":                       # a 16,384-column slice
            from tools.exl3_linear import EXL3Block
            blk = EXL3Block(blk.t[:, :1024].contiguous(), blk.suh, blk.svh[:16384])
        lin = LinearEXL3(None, blk.K, blk.N, suh=blk.suh[0].contiguous(), svh=blk.svh.contiguous(),
                         trellis=blk.t.contiguous(), mul1=idx.read(b + ".mul1", "cuda"))
        inner_eq = bool(torch.equal(blk.inner(), lin.get_inner_weight_tensor().to(torch.float16)))
        w, tw = blk.dequant().float(), lin.get_weight_tensor().float().T
        r = {"tensor": b, "K": X.rate(blk.kb), "shape": [blk.N, blk.K], "inner_bit_equal": inner_eq,
             "weight_rel": ((w - tw).abs().max() / tw.abs().max()).item()}
        rows.append(r)
        print(json.dumps(r), flush=True)
        del lin, w, tw
        torch.cuda.empty_cache()
    print(json.dumps({"all_inner_bit_equal": all(r["inner_bit_equal"] for r in rows),
                      "max_weight_rel": max(r["weight_rel"] for r in rows),
                      "k_histogram": {g: dict(v) for g, v in idx.k_histogram().items()}}), flush=True)


def cmd_bench(a) -> None:
    """Single-stream decode: `tokens` greedy tokens after a short prompt, graph on and off (one fresh
    state each), ms a token from the host's wall clock with a sync every token (the server's loop)."""
    from transformers import AutoTokenizer

    from engine.models.lfm2_moe.engine import DecodeGraph
    tok = AutoTokenizer.from_pretrained(a.model)
    idx, eng, info = _engine(a.model, a.max_len)
    ids = _ids(tok, PROMPTS[0])
    res = {"model": a.model, "load": info, "decode_bytes": idx.decode_bytes(context=len(ids) + a.tokens)["total"]}
    for mode in ("graph", "eager", "graph"):
        st = eng.new_state(a.max_len)
        lg = eng.prefill(torch.tensor(ids), st)
        t = int(lg[0].argmax())
        dg = DecodeGraph(eng, st) if mode == "graph" else None
        if dg is not None:
            t = dg.greedy(t)                     # the capture
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.tokens):
            if dg is not None:
                t = dg.greedy(t)
            else:
                eng.stage(t, st)
                lg = eng.step(st)
                st.length += 1
                t = int(lg[0].argmax())
        dt = time.perf_counter() - t0
        res[mode] = {"tok_s": round(a.tokens / dt, 2), "ms_token": round(1e3 * dt / a.tokens, 3)}
        print(json.dumps({mode: res[mode]}), flush=True)
    res["GBps_graph"] = round(res["decode_bytes"] * res["graph"]["tok_s"] / 1e9, 1)
    if a.profile:
        # where a decode step's GPU time goes: 16 eager steps under the profiler, kernels by total time
        from torch.profiler import ProfilerActivity, profile
        st = eng.new_state(a.max_len)
        eng.prefill(torch.tensor(ids), st)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(16):
                eng.stage(t, st)
                eng.step(st)
                st.length += 1
            torch.cuda.synchronize()
        rows = []
        for ev in prof.key_averages():
            ms = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
            if ms:
                rows.append((ms / 1e3 / 16, ev.count // 16, ev.key[:90]))
        rows.sort(reverse=True)
        res["profile_ms_per_step"] = [{"ms": round(m, 4), "calls": c, "kernel": k} for m, c, k in rows[:25]]
        res["profile_total_ms"] = round(sum(r[0] for r in rows), 3)
        for r in res["profile_ms_per_step"]:
            print(json.dumps(r), flush=True)
    res["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(json.dumps(res), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_staircase(a) -> None:
    """Verify blocks of 1..16 rows (chains) from captured graphs at two context lengths: ms a block
    (CUDA events, the median of `reps`), and the distinct experts the block's rows route to, summed
    over the MoE layers (from an eager run of the same block)."""
    from engine.models.lfm2_moe.engine import VerifyGraph
    from engine.models.lfm2_moe import kernels as K
    idx, eng, info = _engine(a.model, a.max_len)
    g = torch.Generator().manual_seed(0)
    res = {"model": a.model, "load": info, "rows": {}}
    res["prefill_tok_s"] = {}
    for ctx in a.contexts:
        st = eng.new_state(a.max_len)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.prefill(torch.randint(0, 120_000, (ctx,), generator=g), st)
        torch.cuda.synchronize()
        res["prefill_tok_s"][ctx] = round(ctx / (time.perf_counter() - t0), 1)
        print(json.dumps({"ctx": ctx, "prefill_tok_s": res["prefill_tok_s"][ctx]}), flush=True)
        toks = torch.randint(0, 120_000, (16,), generator=g).tolist()
        out = {}
        for M in range(1, 17):
            ns = K.class_of(st.length + M, st.cap)
            eng.route_trace = []
            eng.stage_rows(toks[:M], list(range(-1, M - 1)), st)
            eng.rows_step(st, M, ns=ns)
            experts = sum(len(set(ids.view(-1).tolist())) for _, ids in eng.route_trace)
            eng.route_trace = None
            vg = VerifyGraph(eng, st, M, ns)
            vg.capture()
            eng.stage_rows(toks[:M], list(range(-1, M - 1)), st)
            times = []
            for _ in range(a.reps):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                vg.replay()
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1))
            times.sort()
            out[M] = {"ms": round(times[len(times) // 2], 3), "experts_read": experts}
            print(json.dumps({"ctx": ctx, "rows": M, **out[M]}), flush=True)
            del vg
        res["rows"][ctx] = out
    print(json.dumps(res), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_audit(a) -> None:
    """The pack's audit record from its headers alone (CPU, nothing loaded): kinds, formats, the K
    histogram, resident and decode bytes; a tensor the kernels cannot decode is named."""
    from engine.models.lfm2_moe.pack import PackIndex, summary
    idx = PackIndex.open(a.pack)
    try:
        idx.audit()
        ok = None
    except ValueError as e:
        ok = str(e)
    s = summary(idx)
    s.pop("files")
    s["refused"] = ok
    s["safetensors_gb"] = round(sum(os.path.getsize(os.path.join(idx.path, f)) for f in os.listdir(idx.path)
                                    if f.endswith(".safetensors")) / 1e9, 2)
    print(json.dumps(s, indent=1))


def cmd_prices(a) -> None:
    """StairCut's price file from a staircase run and a bench run: per attention class (the adapter's
    price_class of the staircase's contexts) the chain verify ms for 1..16 rows, the plain step's ms."""
    from engine.models.lfm2_moe import kernels as K
    stair = json.load(open(a.stair))
    bench = json.load(open(a.bench))
    plain = bench["graph"]["ms_token"]
    out = {}
    for ctx, rows in stair["rows"].items():
        cls = K.CHUNK * K.class_of(int(ctx) + 16, int(a.max_len))
        out[str(cls)] = {"chain": {str(r): v["ms"] for r, v in rows.items()}, "plain": {"1": plain}}
    lo = min(out, key=int)
    out = {("0" if k == lo else k): v for k, v in out.items()}
    out["step_ms"] = round(max(0.1, (stair["rows"][max(stair["rows"], key=int)]["16"]["ms"]
                                     - stair["rows"][max(stair["rows"], key=int)]["15"]["ms"])), 2)
    out["measured"] = (f"tools/lfm_check.py staircase and bench on {os.path.basename(a.stair)}, "
                       f"{os.path.basename(a.bench)}: chain = a VerifyGraph replay of 1..16 rows (CUDA events, the "
                       "median), at the staircase's contexts (the lowest as class 0); plain = the captured decode "
                       "step's ms a token, host wall clock with a sync a token. No spec or prime prior: the cut "
                       "drafts and learns them online.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("engine")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--tokens", type=int, default=128)
    e.add_argument("--prompts", type=int, default=8)
    e.add_argument("--graph", default="on", choices=("on", "off"))
    e.add_argument("--max-len", type=int, default=8192)
    e.add_argument("--force", help="a run whose prompts and tokens are fed instead of the greedy ones (teacher-forced)")
    for name in ("hf", "exl3"):
        r = sub.add_parser(name)
        r.add_argument("--model", required=True)
        r.add_argument("--inp", required=True)
        r.add_argument("--json")
        r.add_argument("--save", help="the reference's logits, for a closer look")
    cp = sub.add_parser("cmp")
    cp.add_argument("--a", required=True)
    cp.add_argument("--b", required=True)
    cp.add_argument("--json")
    x = sub.add_parser("xcheck")
    x.add_argument("--pack", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--model", required=True)
    b.add_argument("--tokens", type=int, default=256)
    b.add_argument("--max-len", type=int, default=8192)
    b.add_argument("--json")
    b.add_argument("--profile", action="store_true", help="the decode step's kernels by GPU time")
    s = sub.add_parser("staircase")
    s.add_argument("--model", required=True)
    s.add_argument("--contexts", type=int, nargs="+", default=[1024, 8192])
    s.add_argument("--reps", type=int, default=20)
    s.add_argument("--max-len", type=int, default=16384)
    s.add_argument("--json")
    au = sub.add_parser("audit")
    au.add_argument("--pack", required=True)
    pr = sub.add_parser("prices")
    pr.add_argument("--stair", required=True)
    pr.add_argument("--bench", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--max-len", type=int, default=131072)
    a = ap.parse_args()
    {"cmp": cmd_cmp, "audit": cmd_audit, "prices": cmd_prices, "engine": cmd_engine, "hf": cmd_hf, "exl3": cmd_exl3, "xcheck": cmd_xcheck, "bench": cmd_bench,
     "staircase": cmd_staircase}[a.cmd](a)


if __name__ == "__main__":
    main()
