"""Flash-Next on the board: load a pack, greedy sanity through the chat template, plain decode tok/s.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_generate.py --pack ~/fn2/packs/3.05 \
        --classes prose,code,chat --ctx 1024 --new 256 --json out.json

Single stream, greedy, thinking off, no speculation. Per prompt class (tools/exl3_fastrun.py's
prose, code and chat requests) the prompt is that request after held-out text of its kind
(bench/heldout_prose.txt, bench/heldout_code.txt; past their end the other held-out texts and
bench/calib.txt, repeated), cut so the templated prompt is `--ctx` tokens;
the prompt is prefilled (the reference's batched forward on the kernels), then `--new` tokens are
decoded one at a time through the captured decode step (`DecodeGraph`). decode tok/s =
(new - 1) / the time from the first decoded token to the last (each token waits for its argmax on
the host). Also recorded: load time by part, MemAvailable after the load and at the end, the graph
step against the eager step (bit for bit) and against the reference's one-token forward (argmax
and max |diff|), and every knob that sets the arithmetic or the shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from engine.models.flashnext.pack import MIX  # noqa: E402
from tools.exl3_fastrun import CHAT, CODE, PROSE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILL = {"prose": "heldout_prose.txt", "code": "heldout_code.txt", "chat": "heldout_prose.txt"}
ASK = {"prose": PROSE, "code": CODE, "chat": CHAT}


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / (1 << 20), 1)
    return -1.0


def chat_ids(tok, text: str) -> list[int]:
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False)["input_ids"]


def class_prompt(tok, cls: str, ctx: int) -> list[int]:
    return chat_ids(tok, class_text(tok, cls, ctx))


def class_text(tok, cls: str, ctx: int) -> str:
    """`class_prompt`'s user message: the request after held-out text of its kind, the templated
    prompt `ctx` tokens."""
    ask = ASK[cls]
    if ctx <= 0:
        return ask
    # the class's own held-out text first; past its end (8k, 32k) the other held-out texts and
    # bench/calib.txt, repeated (tools/exl3_fastrun.py's prefill_text)
    names = [FILL[cls]] + [n for n in ("heldout_prose.txt", "heldout_code.txt", "calib.txt") if n != FILL[cls]]
    fill = "\n\n".join(open(os.path.join(ROOT, "bench", n)).read() for n in names)
    fid = tok(fill, add_special_tokens=False)["input_ids"]
    while len(fid) < ctx:
        fid = fid + fid
    lo, hi = 0, len(fid)
    while lo < hi:                                  # the longest fill that keeps the prompt <= ctx
        mid = (lo + hi + 1) // 2
        if len(chat_ids(tok, tok.decode(fid[:mid]) + "\n\n" + ask)) <= ctx:
            lo = mid
        else:
            hi = mid - 1
    return tok.decode(fid[:lo]) + "\n\n" + ask


def greedy(eng: FlashNextEngine, ids: list[int], new: int, eos: set, graph_lc: int = 2048):
    st = eng.new_state(eng.max_len)
    t0 = time.time()
    logits = eng.prefill(torch.tensor(ids), st)
    torch.cuda.synchronize()
    t_prefill = time.time() - t0
    g = DecodeGraph(eng, st, max_lc=graph_lc)
    out = [int(logits[0].argmax())]
    g.capture()
    torch.cuda.synchronize()
    t1 = time.time()
    per = []
    while len(out) < new and out[-1] not in eos:
        ts = time.perf_counter()
        out.append(g.greedy(out[-1]))
        per.append(time.perf_counter() - ts)
    torch.cuda.synchronize()
    t_dec = time.time() - t1
    n = len(out) - 1
    srt = sorted(per) or [0.0]
    return out, {"prompt_tokens": len(ids), "new_tokens": len(out), "prefill_s": round(t_prefill, 3),
                 "prefill_tok_s": round(len(ids) / t_prefill, 1), "decode_s": round(t_dec, 3),
                 "decode_tok_s": round(n / t_dec, 2) if n > 0 else None,
                 "ms_per_token": round(1e3 * t_dec / n, 2) if n > 0 else None,
                 "ms_p50": round(1e3 * srt[len(srt) // 2], 2), "ms_p90": round(1e3 * srt[int(len(srt) * 0.9)], 2),
                 "ms_max": round(1e3 * srt[-1], 2)}


@torch.no_grad()
def step_checks(eng: FlashNextEngine, ids: list[int], k: int = 8) -> dict:
    """The graph step equals the eager step bit for bit; both against the reference's one-token
    forward (the prefill path) by argmax and max |diff| of the logits."""
    a, b, r = eng.new_state(eng.max_len), eng.new_state(eng.max_len), eng.new_state(eng.max_len)
    la = eng.prefill(torch.tensor(ids), a)
    eng.prefill(torch.tensor(ids), b)
    eng.prefill(torch.tensor(ids), r)
    g = DecodeGraph(eng, a)
    g.capture()
    tok = int(la[0].argmax())
    equal, agree, worst = 0, 0, 0.0
    for _ in range(k):
        x = g(tok).clone()
        y = eng.decode_eager(tok, b).clone()
        z = eng.forward(torch.tensor([tok], device=eng.device), r)[-1:].float()
        equal += int(torch.equal(x, y))
        agree += int(int(x.argmax()) == int(z.argmax()))
        worst = max(worst, (x - z).abs().max().item())
        tok = int(x.argmax())
    # the fused trunk against the reference-structured layers on the same kernels (torch glue)
    s1, s2 = eng.new_state(eng.max_len), eng.new_state(eng.max_len)
    f = eng.forward(torch.tensor(ids), s1)[-1]
    rl = eng.forward(torch.tensor(ids), s2, layers=list(range(eng.cfg.num_hidden_layers)))[-1]
    return {"steps": k, "graph_eq_eager": equal, "argmax_eq_forward": agree, "max_abs_vs_forward": round(worst, 4),
            "fused_vs_ref_layers_argmax_eq": int(int(f.argmax()) == int(rl.argmax())),
            "fused_vs_ref_layers_max_abs": round((f - rl).abs().max().item(), 4)}


def injected_self_check(eng: FlashNextEngine, ids: list[int]) -> dict:
    """The self-check on a build whose multi-row expert call differs from the one-row call in one
    bit (the lowest bit of one value of row 5's first pair): it must fail and name that row."""
    from tools import fn_kernels as K
    orig = K.moe_experts

    def flipped(x, ids_, bank, **kw):
        yd = orig(x, ids_, bank, **kw)
        if x.shape[0] > 5:
            r = 5 * ids_.shape[1]
            yd[r, 7] = (yd[r, 7].view(torch.int32) ^ 1).view(torch.float32)
        return yd
    K.moe_experts = flipped
    before = eng.spec_ok
    try:
        r = eng.self_check(ids)
    finally:
        K.moe_experts = orig
        eng.spec_ok = before
    assert not r["ok"] and r["first_diff"]["row"] == 5 and r["first_diff"]["call"] == "chain8", r
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--classes", default="prose,code,chat")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--json")
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--first", type=int, default=None, help="only the first N layers (a debug model)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    res = {"pack": a.pack, "mem_before_load_gib": mem_available_gib()}
    t0 = time.time()
    idx, W, banks, tables, timing = load(a.pack, "cuda", first=a.first)
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=a.max_len)
    torch.cuda.synchronize()
    res["load"] = dict(timing, wall_s=round(time.time() - t0, 1))
    res["mem_after_load_gib"] = mem_available_gib()
    res["cuda_allocated_gib"] = round(torch.cuda.memory_allocated() / (1 << 30), 2)
    print(f"[fn] loaded {res['load']}, MemAvailable {res['mem_after_load_gib']} GiB", flush=True)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    from engine.settings import SETTINGS
    res["knobs"] = {"QWEN38_FN_MIX": MIX, "QWEN38_FN_PDL": SETTINGS.get("FN_PDL"),
                    "QWEN38_FN_WARM_TABLE": SETTINGS.get("FN_WARM_TABLE"), "max_len": a.max_len, "graph_max_lc": 2048, "ctx": a.ctx, "new": a.new,
                    "greedy": True, "thinking": False, "speculation": "off", "batch": 1, "kv_dtype": "bf16",
                    "prefill_chunk": 2048, "env": {k: v for k, v in os.environ.items()
                                                   if k.startswith(("QWEN38_", "QSE_"))}}
    # greedy sanity through the chat template
    sids = chat_ids(tok, "The capital of Italy is")
    out, _ = greedy(eng, sids, 24, eos)
    text = tok.decode(out, skip_special_tokens=True)
    res["sanity"] = {"prompt": "The capital of Italy is", "text": text, "rome": "Rome" in text}
    print(f"[fn] sanity: {text!r}", flush=True)
    if not a.skip_checks:
        res["step_checks"] = step_checks(eng, chat_ids(tok, "Name three rivers in Europe."))
        print(f"[fn] step checks: {res['step_checks']}", flush=True)
        res["self_check"] = eng.self_check(chat_ids(tok, "Count from one to twenty in words."))
        res["self_check_injected"] = injected_self_check(eng, chat_ids(tok, "Count from one to twenty in words."))
    res["classes"] = {}
    for cls in [c for c in a.classes.split(",") if c]:
        ids = class_prompt(tok, cls, a.ctx)
        out, m = greedy(eng, ids, a.new, eos)
        m["text_head"] = tok.decode(out[:48], skip_special_tokens=True)
        res["classes"][cls] = m
        print(f"[fn] {cls}: {json.dumps(m)}", flush=True)
    res["mem_end_gib"] = mem_available_gib()
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
