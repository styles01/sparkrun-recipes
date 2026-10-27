"""What a SAMPLED round costs over a greedy one (SPD-87 follow-up): the served tree router
(`tools/verify_spec.py::tree_router`, wide arm) on the fast run's prompts, greedy and sampled
(temperature / top-p as a client sends them), ms a round and tokens a round, then one profiled
stretch of the sampled loop: CPU time by op (where the host waits), CUDA time by kernel.

Usage: sample_budget.py --prompts prose,chat --new 256 --temperature 0.7 --top-p 0.95
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.sample import Sampler  # noqa: E402
from engine.spec import generate_spec_tree  # noqa: E402
from tools.verify_spec import tree_router  # noqa: E402


def prompt_ids(tok, text: str) -> list[int]:
    msgs = [{"role": "user", "content": text}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok.encode(s, add_special_tokens=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prose,chat")
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--profile-new", type=int, default=96)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--modes", default="greedy,sampled,fast",
                    help="greedy; sampled (the sort, one draw a node); fast (SAMPLE_TOPP_K "
                         "--fast-k and SAMPLE_BATCH on)")
    ap.add_argument("--fast-k", type=int, default=1024)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    from tools.exl3_fastrun import PROSE, CHAT, CODE
    texts = {"prose": PROSE, "chat": CHAT, "code": CODE}
    cfg = load_config(None)
    w = Weights(cfg.path, device="cuda", skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device="cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    ns = types.SimpleNamespace(dflash2_ckpt="train/ft-b8-v2", dflash2_path="greedy",
                               max_len=a.max_len, lenrouter="train/ft-b16", corpus="",
                               drop_idle=True, deep=int(os.environ.get("QWEN38_DEEP", "0")),
                               deep_after=int(os.environ.get("QWEN38_DEEP_AFTER", "2")), len_fixed=0)
    lr = tree_router(eng, ns, narrow=False)
    out = {"temperature": a.temperature, "top_p": a.top_p, "rows": {}}
    hist = collections.Counter()
    _ft = eng.forward_tree

    def counted(tokens, parents, start):
        hist[int(tokens.numel())] += 1
        return _ft(tokens, parents, start)
    eng.forward_tree = counted
    for name in a.prompts.split(","):
        ids = torch.tensor(prompt_ids(tok, texts[name]), device="cuda")
        from engine import sample as SM
        for mode in a.modes.split(",") * a.reps:
            SM.TOPP_K, SM.BATCH = (a.fast_k, True) if mode == "fast" else (0, False)
            smp = Sampler(a.temperature, a.top_p) if mode != "greedy" else None
            if smp is not None:
                lr.set_sampling(smp) if hasattr(lr, "set_sampling") else None
            else:
                lr.set_sampling(None) if hasattr(lr, "set_sampling") else None
            hist.clear()
            got, st = generate_spec_tree(eng, ids, a.new, lr, 15, sampler=smp)
            r = {"tokens": st.tokens, "blocks": st.blocks,
                 "tok_round": st.tokens / max(1, st.blocks),
                 "ms_round": st.decode_s * 1e3 / max(1, st.blocks),
                 "draft_ms": st.draft_s * 1e3 / max(1, st.blocks),
                 "tok_s": st.tokens / max(1e-9, st.decode_s),
                 "verify_rows": dict(sorted(hist.items()))}
            out["rows"].setdefault(f"{name}/{mode}", []).append(r)
            print(f"[sample] {name} {mode}: " + json.dumps(
                {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}),
                  flush=True)
    # one profiled sampled stretch
    from torch.profiler import ProfilerActivity, profile
    ids = torch.tensor(prompt_ids(tok, texts[a.prompts.split(",")[0]]), device="cuda")
    smp = Sampler(a.temperature, a.top_p)
    if hasattr(lr, "set_sampling"):
        lr.set_sampling(smp)
    generate_spec_tree(eng, ids, 32, lr, 15, sampler=smp)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _, st = generate_spec_tree(eng, ids, a.profile_new, lr, 15, sampler=smp)
    blocks = max(1, st.blocks)
    cpu = collections.Counter()
    gpu = collections.Counter()
    for e in prof.key_averages():
        cpu[e.key[:80]] += e.self_cpu_time_total / 1e3 / blocks
        dt = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        gpu[e.key[:80]] += dt / 1e3 / blocks
    out["profile"] = {"blocks": blocks, "wall_ms_round": st.decode_s * 1e3 / blocks,
                      "cpu_ms_round": [(k, round(v, 3)) for k, v in cpu.most_common(30)],
                      "gpu_ms_round": [(k, round(v, 3)) for k, v in gpu.most_common(30)]}
    print(json.dumps(out["profile"], indent=1), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
