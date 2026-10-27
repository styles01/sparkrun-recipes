"""NemotronH's correctness ladder and its speed numbers, one model a process.

    PY=<engine python>; PYTHONPATH=<repo>:$PYTHONPATH
    $PY tools/nh_check.py config --model <dir>                    the layer table the config reads to
    $PY tools/nh_check.py audit --pack <dir>                      the checkpoint's audit from its headers
    $PY tools/nh_check.py engine --model <dir> --out run.pt       greedy decode of the prompts below on
                                                                   the engine (prefill, then the captured
                                                                   decode step), every step's logits kept
    $PY tools/nh_check.py hf --model <bf16 dir> --inp run.pt      transformers' NemotronHForCausalLM on the
                                                                   same tokens, teacher-forced
    $PY tools/nh_check.py bench --model <dir>                     single-stream decode tok/s, graph on/off
    $PY tools/nh_check.py staircase --model <dir>                 verify ms for 1..16 rows at 1k and 8k
                                                                   context, and the experts a block reads
    $PY tools/nh_check.py mtp --model <dir> --depth 3                the checkpoint's MTP head against the
                                                                   trunk's greedy text: agreement a depth
    $PY tools/nh_check.py lilicorr --model <dir> --draft <dir>        LiLiCorr (SPD-123) against the trunk's
                                                                   greedy text: a draft at every position,
                                                                   agreement a depth, the accepted length
                                                                   of a greedy speculative run, draft ms
    $PY tools/nh_check.py prices --stair s.json --bench b.json --out <price file> [--lilicorr l.json]
    $PY tools/nh_check.py llamacpp --model <dir> --url <llama-server> --out ref.pt
                                                                   llama.cpp's greedy text on the same
                                                                   token ids, its top 10 log-probs a step
    $PY tools/nh_check.py engine --model <dir> --force ref.pt --out forced.pt
    $PY tools/nh_check.py llamacmp --ref ref.pt --a forced.pt [--free run.pt]
                                                                   the engine teacher-forced on llama.cpp's
                                                                   tokens against llama.cpp's choices

A comparison reports, over every decode position: the argmax agreement, the positions where the two
disagree and whether each is a near tie (the reference's margin between the two tokens under 8 bf16
steps of its largest logit), and the logit difference (max and mean |diff|, relative to the largest
logit). The reference runs in its own process because only one large model is on the board at a time.

llama.cpp (a GGUF's same-weights reference) answers through llama-server's /completion with the prompt
as token ids, greedy, `ignore_eos`, `n_probs` 10: its greedy tokens and each step's top 10 log-probs.
Teacher-forced on those tokens the engine's argmax is compared with llama.cpp's; a disagreement is a near
tie when llama.cpp's own log-prob margin between the two tokens (its logit margin) is under 8 bf16 steps
of the engine's largest logit at that step, and the log-probs of the 10 tokens are compared.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _engine(model: str, max_len: int, taps=None):
    """The engine of a checkpoint; `taps`: a LiLiCorr config whose taps it keeps."""
    from engine.models.nemotron_h.engine import NemotronHEngine
    from engine.models.nemotron_h.load import load_any
    t0 = time.time()
    idx, W, experts, timing = load_any(model, "cuda", mtp=False, log=None)
    eng = NemotronHEngine(idx.cfg, W, experts, device="cuda", max_len=max_len)
    if taps is not None:
        eng.set_taps(taps.target_layer_ids, taps.window)
    torch.cuda.synchronize()
    return idx, eng, {"load_s": round(time.time() - t0, 1), "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                      **timing}


def cmd_engine(a) -> None:
    from transformers import AutoTokenizer

    from engine.models.nemotron_h.engine import DecodeGraph
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


def _hf_from_engine_weights(model_dir: str):
    """transformers' NemotronHForCausalLM built on the meta device and given the engine loader's device
    tensors (no second copy, no host staging): its arithmetic on the same bytes. transformers' own loader
    stages the stacked experts on the host and does not fit beside the device copy on a 121 GiB board."""
    from transformers import AutoConfig
    from transformers.models.nemotron_h import modeling_nemotron_h as hm

    from engine.models.nemotron_h.load import load
    idx, W, experts, _ = load(model_dir, "cuda", mtp=False, log=None)
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg.use_mamba_kernels = False
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = hm.NemotronHForCausalLM(cfg)
    sd = {}
    for k, v in W.items():
        if k.endswith("qkv"):
            continue
        sd[k if k.startswith("lm_head") else "model." + k] = v
    for li, e in experts.items():
        sd[f"model.layers.{li}.mixer.experts.up_proj"] = e.up
        sd[f"model.layers.{li}.mixer.experts.down_proj"] = e.down
    for k in list(sd):
        if k.endswith(("A_log", "dt_bias", ".D")):
            sd[k] = sd[k].to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if not m.endswith("inv_freq")]
    if missing or unexpected:
        raise ValueError(f"missing {missing[:4]}, unexpected {unexpected[:4]}")
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            raise ValueError(f"buffer {name} left on meta")
    return model.eval()


def cmd_hf(a) -> None:
    from transformers import AutoModelForCausalLM
    run = torch.load(a.inp, weights_only=False)
    if a.shared:
        model = _hf_from_engine_weights(a.model)
        longest = max(len(r["ids"]) + len(r["gen"]) for r in run["runs"])
        for m in model.modules():
            if hasattr(m, "chunk_size") and hasattr(m, "A_log"):
                m.chunk_size = max(m.chunk_size, 1 << (longest - 1).bit_length())
        return _hf_run(a, run, model)
    # the torch Mamba path (mamba_ssm is not on the box), one SSD chunk per sequence: transformers'
    # multi-chunk torch path disagrees with its own one-chunk result (tests/test_nemotron_h_ref.py)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", use_mamba_kernels=False).eval()
    longest = max(len(r["ids"]) + len(r["gen"]) for r in run["runs"])
    for m in model.modules():
        if hasattr(m, "chunk_size") and hasattr(m, "A_log"):
            m.chunk_size = max(m.chunk_size, 1 << (longest - 1).bit_length())
    return _hf_run(a, run, model)


def _hf_run(a, run, model) -> None:
    refs = []
    with torch.no_grad():
        for r in run["runs"]:
            seq = torch.tensor([r["ids"] + r["gen"][:-1]], device="cuda")
            lg = model(seq).logits[0, len(r["ids"]) - 1:].float().cpu()
            refs.append(lg)
    if a.save:
        torch.save([r.to(torch.bfloat16) for r in refs], a.save)
    res = {"rung": "b", "reference": "transformers NemotronHForCausalLM bf16 sdpa", **_compare(run, refs)}
    print(json.dumps(res, indent=1), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_mtp(a) -> None:
    """The checkpoint's MTP head against the trunk's own greedy text: per prompt the greedy decode,
    then at every decode position the head fed the trunk's row and the next token, its `depth` drafts
    against the tokens the trunk wrote: the agreement at each depth and the mean accepted run."""
    from transformers import AutoTokenizer

    from engine.models.nemotron_h.engine import DecodeGraph, NemotronHEngine
    from engine.models.nemotron_h.load import load
    from engine.models.nemotron_h.mtp import NemotronHMTPHead
    tok = AutoTokenizer.from_pretrained(a.model)
    idx, W, experts, timing = load(a.model, "cuda", mtp=True, log=None)
    eng = NemotronHEngine(idx.cfg, W, experts, device="cuda", max_len=a.max_len)
    agree = [0] * a.depth
    runs, tot, accepted = [], 0, 0
    for p in PROMPTS[:a.prompts]:
        ids = _ids(tok, p)
        st = eng.new_state(a.max_len)
        lg = eng.prefill(torch.tensor(ids), st)
        rows = [eng.prompt_hidden.clone()]
        gen = [int(lg[0].argmax())]
        dg = DecodeGraph(eng, st)
        for _ in range(a.tokens):
            gen.append(dg.greedy(gen[-1]))
            rows.append(dg.rb["hidden"].clone())
        H = torch.cat(rows)                         # the row of every position 0 .. P + tokens - 1
        seq = ids + gen
        head = NemotronHMTPHead(eng, a.depth, a.max_len)
        P = len(ids)
        head.prime(H[:P - 1], seq[1:P])
        n_ok = []
        for t in range(P - 1, P - 1 + a.tokens - a.depth):
            head.launch(H[t:t + 1], [seq[t + 1]])
            drafts, _ = head.read()
            run = 0
            for d in range(a.depth):
                ok = drafts[d][0] == seq[t + 2 + d]
                agree[d] += ok
                if ok and run == d:
                    run += 1
            n_ok.append(run)
            tot += 1
        accepted += sum(n_ok)
        runs.append({"prompt": p[:40], "mean_run": round(sum(n_ok) / max(len(n_ok), 1), 3)})
        print(json.dumps(runs[-1]), flush=True)
    res = {"model": a.model, "depth": a.depth, "positions": tot,
           "agree_at_depth": [round(x / max(tot, 1), 4) for x in agree],
           "mean_accepted_run": round(accepted / max(tot, 1), 3), "runs": runs}
    print(json.dumps(res), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_lilicorr(a) -> None:
    """LiLiCorr against the trunk's own greedy text: per prompt the prefill and `tokens` greedy decode
    steps from the captured graph (the taps on), and before each step a draft anchored at that
    position (the context the committed rows give it). Reported: the agreement of the draft with the
    trunk's next tokens at each depth, the mean run of agreeing tokens, the accepted length a greedy
    speculative run of these drafts would have (blocks of 1 + the run, from the first token on: the
    figure llama.cpp and vLLM report), and the draft's ms (CUDA events, eager)."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from engine.drafters.lilicorr import build_head, load_config
    from engine.models.nemotron_h.engine import DecodeGraph
    tok = AutoTokenizer.from_pretrained(a.model)
    idx, eng, info = _engine(a.model, a.max_len, taps=load_config(a.draft))
    srv = SimpleNamespace(fn=eng, st=None, kv=SimpleNamespace(length=0))
    head = build_head(srv, a.draft, log=lambda m: print(m, flush=True))
    D = head.depth
    prompts = [p for p in PROMPTS[:a.prompts]] + [open(f).read() for f in (a.prompt_file or [])]
    agree, runs_all, al_all, ms_all = [0] * D, [], [], []
    out = {"model": a.model, "draft": a.draft, "tokens": a.tokens, "thinking": a.thinking, "load": info, "prompts": []}
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       **({} if a.thinking else {"enable_thinking": False}))
        ids = tok(text, add_special_tokens=False)["input_ids"]
        st = eng.new_state(a.max_len)
        srv.st = st
        lg = eng.prefill(torch.tensor(ids), st)
        seq = list(ids) + [int(lg[0].argmax())]
        dg = DecodeGraph(eng, st)
        drafts, ms = [], []
        for _ in range(a.tokens):
            n = st.length                                   # the anchor seq[n] at position n
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            d = head.draft_at(seq[n], n)
            e1.record()
            drafts.append(d[:, 0].long().tolist())
            ms.append(e0.elapsed_time(e1))
            seq.append(dg.greedy(seq[n]))
        P = len(ids)
        runs = []
        for j, d in enumerate(drafts[:a.tokens - D]):
            t = P + j
            r = 0
            while r < D and d[r] == seq[t + 1 + r]:
                r += 1
            runs.append(r)
            for k in range(D):
                agree[k] += d[k] == seq[t + 1 + k]
        # a greedy speculative run of these drafts: a block keeps its run and the trunk's next token
        t, blocks, toks = 0, 0, 0
        while t < len(runs):
            blocks += 1
            toks += runs[t] + 1
            t += runs[t] + 1
        rec = {"prompt": p[:48], "mean_run": round(sum(runs) / max(len(runs), 1), 3),
               "accepted_length": round(toks / max(blocks, 1), 3), "draft_ms": round(sorted(ms)[len(ms) // 2], 2),
               "text": tok.decode(seq[P:])[:120]}
        runs_all += runs
        al_all.append(rec["accepted_length"])
        ms_all += ms
        out["prompts"].append(rec)
        print(json.dumps(rec), flush=True)
    n = max(len(runs_all), 1)
    out.update({"positions": len(runs_all), "agree_at_depth": [round(x / n, 4) for x in agree],
                "mean_run": round(sum(runs_all) / n, 3),
                "accepted_length_mean": round(sum(al_all) / max(len(al_all), 1), 3),
                "accepted_length_range": [min(al_all), max(al_all)] if al_all else None,
                "draft_ms_median": round(sorted(ms_all)[len(ms_all) // 2], 2) if ms_all else None,
                "head_stats": head.stats})
    print(json.dumps({k: v for k, v in out.items() if k != "prompts"}), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


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


def cmd_llamacpp(a) -> None:
    """llama.cpp's greedy run on the prompts (token ids from this checkpoint's chat template)."""
    import urllib.request
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    out = {"model": a.model, "url": a.url, "tokens": a.tokens, "runs": []}
    for p in PROMPTS[:a.prompts]:
        ids = _ids(tok, p)
        body = {"prompt": ids, "n_predict": a.tokens, "temperature": 0.0, "top_k": 1, "n_probs": 10, "ignore_eos": True,
                "cache_prompt": False, "samplers": ["top_k"]}
        req = urllib.request.Request(a.url.rstrip("/") + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=1800) as f:
            d = json.load(f)
        dt = time.perf_counter() - t0
        probs = d["completion_probabilities"]
        gen = [int(x["id"]) for x in probs]
        top = [[(int(t["id"]), float(t["logprob"])) for t in x["top_logprobs"]] for x in probs]
        timings = d.get("timings", {})
        r = {"prompt": p, "ids": ids, "gen": gen, "top": top, "text": tok.decode(gen), "wall_s": round(dt, 2),
             "decode_tok_s": timings.get("predicted_per_second"), "prompt_tok_s": timings.get("prompt_per_second")}
        out["runs"].append(r)
        print(json.dumps({k: r[k] for k in ("prompt", "decode_tok_s")} | {"text": r["text"][:160]}), flush=True)
    torch.save(out, a.out)


def cmd_llamacmp(a) -> None:
    """The engine teacher-forced on llama.cpp's tokens (`engine --force ref.pt`) against llama.cpp's
    choices; `--free`: the engine's own greedy run, its first token that differs from llama.cpp's."""
    ref = torch.load(a.ref, weights_only=False)["runs"]
    run = torch.load(a.a, weights_only=False)["runs"]
    tot = agree = near = 0
    bad, lpd, p1 = [], [], []
    for r, e in zip(ref, run):
        assert r["ids"] == e["ids"] and r["gen"] == e["gen"], "the engine run is not forced on this reference"
        lg = e["logits"].float()
        for i, top in enumerate(r["top"]):
            tot += 1
            row = lg[i]
            lp = torch.log_softmax(row, -1)
            ours, theirs = int(row.argmax()), top[0][0]
            ids = torch.tensor([t for t, _ in top])
            lpd.append((lp[ids] - torch.tensor([v for _, v in top])).abs().mean().item())
            p1.append(abs(lp[theirs].exp().item() - math.exp(top[0][1])))
            if ours == theirs:
                agree += 1
                continue
            have = dict(top)
            margin = top[0][1] - have.get(ours, top[-1][1])           # at least this when ours is not in the top 10
            tie = ours in have and margin < 8 * 2 ** -7 * float(row.abs().max())
            near += tie
            if len(bad) < 16:
                bad.append({"prompt": r["prompt"][:40], "pos": i, "llama_margin": round(margin, 4), "near_tie": bool(tie),
                            "engine_margin": round(float(row[ours] - row[theirs]), 4)})
    res = {"positions": tot, "argmax_agree": agree, "agree_pct": round(100 * agree / max(tot, 1), 2), "near_tie": near,
           "other": tot - agree - near, "top10_logprob_mean_abs_diff": round(sum(lpd) / max(len(lpd), 1), 5),
           "top1_prob_mean_abs_diff": round(sum(p1) / max(len(p1), 1), 5), "disagreements": bad}
    if a.free:
        free = torch.load(a.free, weights_only=False)["runs"]
        res["free_greedy_first_diff"] = [next((i for i, (x, y) in enumerate(zip(f["gen"], r["gen"])) if x != y), None)
                                         for f, r in zip(free, ref)]
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_bench(a) -> None:
    """Single-stream decode: `tokens` greedy tokens after a short prompt, graph on and off (one fresh
    state each), ms a token from the host's wall clock with a sync every token (the server's loop)."""
    from transformers import AutoTokenizer

    from engine.models.nemotron_h.engine import DecodeGraph
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
    from engine.models.nemotron_h.engine import VerifyGraph
    from engine.models.nemotron_h import kernels as K
    idx, eng, info = _engine(a.model, a.max_len)
    g = torch.Generator().manual_seed(0)
    res = {"model": a.model, "load": info, "rows": {}}
    res["prefill_tok_s"] = {}
    for ctx in a.contexts:
        st = eng.new_state(a.max_len)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.prefill(torch.randint(0, 131_000, (ctx,), generator=g), st)
        torch.cuda.synchronize()
        res["prefill_tok_s"][ctx] = round(ctx / (time.perf_counter() - t0), 1)
        print(json.dumps({"ctx": ctx, "prefill_tok_s": res["prefill_tok_s"][ctx]}), flush=True)
        toks = torch.randint(0, 131_000, (16,), generator=g).tolist()
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


def cmd_config(a) -> None:
    """The config read into the layer table (CPU, nothing loaded): one row a layer, the MTP head's blocks."""
    from engine.models.nemotron_h.config import load_nemotron_h_config
    c = load_nemotron_h_config(a.model)
    print(json.dumps({"architecture": c.architecture, "omni": c.omni, "layers": len(c.layer_types),
                      "hidden": c.hidden_size, "mamba": [c.mamba_num_heads, c.mamba_head_dim, c.ssm_state_size,
                                                         c.n_groups], "latent": c.moe_latent_size,
                      "shared_ffn": c.shared_ffn, "eos": c.eos_token_ids, "mtp": c.mtp_layer_types,
                      "mtp_moe": c.mtp_moe}))
    for row in c.table():
        print(json.dumps(row))
    if a.against:
        # the release-day checklist: every field and layer-table row that differs from a known config
        from engine.models.nemotron_h.config import config_from_dict
        b = config_from_dict(json.load(open(a.against)), a.against)
        fields = ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "rms_norm_eps",
                  "max_position_embeddings", "mamba_num_heads", "mamba_head_dim", "ssm_state_size", "n_groups",
                  "conv_kernel", "chunk_size", "moe_latent_size", "shared_ffn", "routed_scaling_factor",
                  "tie_word_embeddings", "eos_token_ids", "mtp_layer_types", "mtp_moe", "omni")
        diff = {f: [getattr(c, f), getattr(b, f)] for f in fields if getattr(c, f) != getattr(b, f)}
        rows = [(x, y) for x, y in zip(c.table(), b.table()) if x != y]
        print(json.dumps({"against": a.against, "fields_differ": diff, "layers": [len(c.layer_types), len(b.layer_types)],
                          "rows_differ": len(rows) + abs(len(c.layer_types) - len(b.layer_types)), "first_rows": rows[:6]}))


def cmd_audit(a) -> None:
    """The checkpoint's audit record from its headers alone (CPU, nothing loaded): dtypes, the layer
    counts, resident and decode bytes; a missing or misshapen tensor is named."""
    from engine.models.nemotron_h.pack import PackIndex, summary
    idx = PackIndex.open(a.pack)
    try:
        idx.audit()
        ok = None
    except ValueError as e:
        ok = str(e)
    s = summary(idx)
    s["refused"] = ok
    s["safetensors_gb"] = round(sum(os.path.getsize(os.path.join(idx.path, f)) for f in os.listdir(idx.path)
                                    if f.endswith(".safetensors")) / 1e9, 2)
    print(json.dumps(s, indent=1))


def spec_prior(stair: dict, lilicorr: dict) -> float:
    """StairCut's `spec` prior, a drafted round's ms a token: the 8-row chain at the staircase's lowest
    context plus the draft, over 0.75 of the accepted length `lilicorr` measured offline. It anchors the
    EMA the first rounds set (their captures and first kernel calls ran 2-3 times the steady cost and,
    with no prior, priced every drafted round above the plain step for good)."""
    rows = stair["rows"][min(stair["rows"], key=int)]
    return round((rows["8"]["ms"] + lilicorr["draft_ms_median"]) / (0.75 * lilicorr["accepted_length_mean"]), 1)


def cmd_prices(a) -> None:
    """StairCut's price file from a staircase run and a bench run: per attention class (the adapter's
    price_class of the staircase's contexts) the chain verify ms for 1..16 rows, the plain step's ms."""
    from engine.models.nemotron_h import kernels as K
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
    measured = (f"tools/nh_check.py staircase and bench on {os.path.basename(a.stair)}, "
                f"{os.path.basename(a.bench)}: chain = a VerifyGraph replay of 1..16 rows (CUDA events, the "
                "median), at the staircase's contexts (the lowest as class 0); plain = the captured decode "
                "step's ms a token, host wall clock with a sync a token.")
    if a.lilicorr:
        spec = spec_prior(stair, json.load(open(a.lilicorr)))
        for k in out:
            if k.isdigit():
                out[k]["spec"] = {"1": spec}
        out["measured"] = (f"{measured} No prime prior (learned online). Spec prior {spec} ms a token: (8-row "
                           f"chain + draft) / (0.75 x LiLiCorr's offline accepted length), {os.path.basename(a.lilicorr)}.")
    else:
        out["measured"] = f"{measured} No spec or prime prior: the cut drafts and learns them online."
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
    for name in ("hf",):
        r = sub.add_parser(name)
        r.add_argument("--model", required=True)
        r.add_argument("--inp", required=True)
        r.add_argument("--json")
        r.add_argument("--save", help="the reference's logits, for a closer look")
        r.add_argument("--shared", action="store_true",
                       help="transformers' modules on the engine loader's device tensors (no host staging)")
    cp = sub.add_parser("cmp")
    cp.add_argument("--a", required=True)
    cp.add_argument("--b", required=True)
    cp.add_argument("--json")
    mt = sub.add_parser("mtp")
    mt.add_argument("--model", required=True)
    mt.add_argument("--depth", type=int, default=3)
    mt.add_argument("--tokens", type=int, default=128)
    mt.add_argument("--prompts", type=int, default=8)
    mt.add_argument("--max-len", type=int, default=8192)
    mt.add_argument("--json")
    li = sub.add_parser("lilicorr")
    li.add_argument("--model", required=True)
    li.add_argument("--draft", required=True, help="the LiLiCorr checkpoint directory")
    li.add_argument("--tokens", type=int, default=256)
    li.add_argument("--prompts", type=int, default=8)
    li.add_argument("--prompt-file", nargs="*", help="more prompts, a file each")
    li.add_argument("--thinking", type=int, default=1, help="the chat template's thinking (1 on, 0 off)")
    li.add_argument("--max-len", type=int, default=8192)
    li.add_argument("--json")
    cf = sub.add_parser("config")
    cf.add_argument("--model", required=True)
    cf.add_argument("--against", help="a known config.json (Puzzle 75B-A9B's): the fields and rows that differ")
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
    pr.add_argument("--lilicorr", help="a `lilicorr --json` run: the file gets its spec prior")
    lc = sub.add_parser("llamacpp")
    lc.add_argument("--model", required=True, help="the checkpoint directory (its tokenizer and chat template)")
    lc.add_argument("--url", required=True, help="llama-server on the same GGUF")
    lc.add_argument("--out", required=True)
    lc.add_argument("--tokens", type=int, default=128)
    lc.add_argument("--prompts", type=int, default=8)
    lm = sub.add_parser("llamacmp")
    lm.add_argument("--ref", required=True)
    lm.add_argument("--a", required=True, help="engine --force <ref>")
    lm.add_argument("--free", help="the engine's own greedy run")
    lm.add_argument("--json")
    a = ap.parse_args()
    {"llamacpp": cmd_llamacpp, "llamacmp": cmd_llamacmp, "cmp": cmd_cmp, "audit": cmd_audit, "prices": cmd_prices, "engine": cmd_engine, "hf": cmd_hf, "config": cmd_config, "mtp": cmd_mtp,
     "lilicorr": cmd_lilicorr,
     "bench": cmd_bench, "staircase": cmd_staircase}[a.cmd](a)


if __name__ == "__main__":
    main()
