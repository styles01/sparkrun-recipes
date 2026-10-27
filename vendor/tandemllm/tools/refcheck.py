"""The M1 gate: do the engine's logits agree with the published implementation?

Two phases, run as two processes, because only one model fits on this board at a time:

    python tools/refcheck.py dump   --out /tmp/ref.pt   # published implementation, bf16
    python tools/refcheck.py engine --ref /tmp/ref.pt   # this engine, fp8 as stored

The reference is loaded by dequantising the checkpoint's fp8 blocks to bf16 and handing the result
to the published module, so the comparison isolates *this engine's* arithmetic rather than the
quantiser's: both sides see the same numbers, one through a stock module in bf16, one through the
engine's kernels against the codes.

What the gate reports: argmax agreement over the teacher-forced positions, the maximum absolute
logit difference, and the mean absolute difference. Argmax agreement is the number that decides the
gate; the logit differences say how much headroom there is before it starts to matter.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402

REF_TEXT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "bench", "refprompt.txt")


def tokens_for(cfg, n: int) -> torch.Tensor:
    """Tokens the model is confident about, which is what makes an agreement number mean anything.

    An earlier version of this fed the model one instruction repeated six times with no answer
    after it. The reference implementation scores that at 9.5 nats -- it is far outside anything an
    instruction-tuned model has seen -- and on a distribution that flat the top-1 and top-2 logits
    are within a rounding error of each other almost everywhere, so two correct implementations
    disagree on a fifth of all positions. The comparison measured the prompt, not the engine.
    Ordinary prose, which this model scores at 1.4 nats, is the right input for a numerical gate.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    with open(REF_TEXT) as f:
        text = f.read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    if ids.numel() < n:
        raise ValueError(f"{REF_TEXT} is {ids.numel()} tokens, need {n}")
    return ids[:n].contiguous()


def report(a: torch.Tensor, b: torch.Tensor, label: str) -> dict:
    """Compare two sets of teacher-forced logits, `a` under test against `b` as the reference.

    A plain argmax agreement is close to meaningless on this model: its top-1/top-2 logit gap has a
    median near 1.3 and a tenth percentile near 0.25, so a fifth of all positions are a coin flip
    that any bf16 rounding decides. What separates a correct implementation from a broken one is
    agreement where the reference is actually decided, the overlap of the top of the distribution,
    and the KL divergence -- all three are reported, against the same three numbers for the
    reference compared with itself under a different attention kernel.
    """
    aa, bb = a.argmax(-1), b.argmax(-1)
    agree = (aa == bb).float().mean().item()
    top2 = b.topk(2, dim=-1).values
    gap = top2[:, 0] - top2[:, 1]
    out = {"label": label, "n": a.shape[0], "argmax": agree}
    for thr in (0.5, 1.0, 2.0):
        sel = gap >= thr
        out["gap%s" % thr] = ((aa == bb)[sel].float().mean().item(), int(sel.sum()))
    ta = a.topk(5, dim=-1).indices
    tb = b.topk(5, dim=-1).indices
    overlap = torch.tensor([len(set(x.tolist()) & set(y.tolist())) for x, y in zip(ta, tb)],
                           dtype=torch.float32)
    out["top5"] = (overlap.mean() / 5).item()
    pa = torch.log_softmax(a.float(), -1)
    pb = torch.log_softmax(b.float(), -1)
    out["kl"] = (pb.exp() * (pb - pa)).sum(-1).mean().item()
    d = (a - b).abs()
    out["maxd"] = d.max().item()
    out["meand"] = d.mean().item()
    return out


def print_report(r: dict) -> None:
    print("--- %s  (%d positions)" % (r["label"], r["n"]))
    print("    argmax agreement            %7.3f %%" % (r["argmax"] * 100))
    for thr in (0.5, 1.0, 2.0):
        v, n = r["gap%s" % thr]
        print("    argmax where gap >= %-4s    %7.3f %%   (%d positions)" % (thr, v * 100, n))
    print("    top-5 overlap               %7.3f %%" % (r["top5"] * 100))
    print("    KL(reference || test)       %9.5f nats" % r["kl"])
    print("    max |logit diff|            %9.4f" % r["maxd"])
    print("    mean |logit diff|           %9.5f" % r["meand"])


def do_dump(args) -> None:
    """Teacher-forced logits from the published implementation, written to disk.

    The stock module is given bf16 weights obtained by dequantising the checkpoint's fp8 blocks
    exactly as the format defines (`code.to(f32) * scale[n // 128, k // 128]`). The engine reads the
    same codes and applies the same scales inside its GEMM, so what the comparison measures is this
    engine's arithmetic and not a quantiser's -- both sides start from identical numbers.

    (The checkpoint's own quantiser path in transformers 5.12.1 leaves the projections as plain
    `nn.Linear` holding fp8 weights and fails in the first MLP, so it is not usable as a reference.)
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5ForCausalLM,
                                                              Qwen3_5TextRotaryEmbedding)
    from engine.loader import Weights

    cfg = load_config(args.model)
    ids = tokens_for(cfg, args.n)
    print(f"[dump] {args.n} tokens, snapshot {cfg.path}", flush=True)

    hf = AutoConfig.from_pretrained(cfg.path).text_config
    hf._attn_implementation = args.attn
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(hf)

    t0 = time.time()
    w = Weights(cfg.path, device=args.device, skip_mtp=True)
    print(f"[dump] codes loaded {time.time() - t0:.1f}s  {w.report()}", flush=True)

    sd: dict[str, torch.Tensor] = {}
    for name, t in list(w.t.items()):
        if name == "lm_head.weight":
            sd[name] = t
        elif name.startswith("layers.") or name in ("embed_tokens.weight", "norm.weight"):
            sd["model." + name] = t
    for base in sorted(w.q):
        if not base.startswith("layers."):
            continue
        blk = w.q.pop(base)
        sd["model." + base + ".weight"] = blk.dequant()
        blk.w = None  # release the codes as each weight is expanded
    torch.cuda.empty_cache()
    print(f"[dump] dequantised, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated", flush=True)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    if missing:
        raise RuntimeError(f"reference missing weights: {missing[:8]} ({len(missing)} total)")
    # the rotary buffers were built on the meta device with the skeleton
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(hf, device=args.device)
    model.eval()
    del sd, w
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids[None].to(args.device), output_hidden_states=True)
    logits = out.logits[0].float().cpu()
    hidden = [h[0].float().cpu() for h in out.hidden_states]
    print(f"[dump] forward {time.time() - t0:.1f}s  logits {tuple(logits.shape)}  "
          f"{len(hidden)} hidden states", flush=True)
    torch.save({"ids": ids, "logits": logits, "hidden": hidden}, args.out)
    print(f"[dump] wrote {args.out}", flush=True)


def do_engine(args) -> None:
    from engine.loader import Weights
    from engine.model import Qwen38Engine

    ref = torch.load(args.ref, map_location="cpu")
    ids, ref_logits = ref["ids"], ref["logits"]
    cfg = load_config(args.model)
    t0 = time.time()
    w = Weights(cfg.path, device=args.device, skip_mtp=True)
    print(f"[engine] load {time.time() - t0:.1f}s  {w.report()}", flush=True)
    print(f"[engine] step bytes {w.decode_step_bytes(cfg.num_hidden_layers)}", flush=True)
    eng = Qwen38Engine(cfg, w, max_len=max(1024, ids.numel() + 64), device=args.device)
    taps: list[torch.Tensor] = []
    if args.layers:
        eng.tap = taps.append
    t0 = time.time()
    with torch.no_grad():
        logits = eng.forward(ids.to(args.device)).float()[0].cpu()
    print(f"[engine] forward {time.time() - t0:.1f}s", flush=True)
    if args.layers and "hidden" in ref:
        print("layer   max|diff|    mean|diff|   ref rms   note")
        for i, (mine, theirs) in enumerate(zip(taps, ref["hidden"])):
            d = (mine.float().cpu() - theirs).abs()
            kind = "embed" if i == 0 else ("linear" if cfg.is_linear(i - 1) else "ATTN")
            print(f"{i:5d}   {d.max().item():9.4f}   {d.mean().item():9.5f}   "
                  f"{theirs.float().pow(2).mean().sqrt().item():8.4f}   {kind}")

    print_report(report(logits, ref_logits, "this engine vs the published implementation"))
    if args.floor and os.path.isfile(args.floor):
        other = torch.load(args.floor, map_location="cpu")["logits"]
        print_report(report(other, ref_logits, "the reference against itself, other attention kernel"))
    conf = report(logits, ref_logits, "x")["gap1.0"][0]
    print()
    print("GATE  argmax on confident positions (gap >= 1.0) >= 99 %%   %s  (%.3f %%)"
          % ("PASS" if conf >= 0.99 else "FAIL", conf * 100))


def do_pair(args) -> None:
    """Two reference dumps against each other: the model's own sensitivity, with no engine in it."""
    a = torch.load(args.ref, map_location="cpu")
    b = torch.load(args.other, map_location="cpu")
    assert torch.equal(a["ids"], b["ids"])
    la, lb = a["logits"], b["logits"]
    print_report(report(la, lb, os.path.basename(args.ref) + " vs " + os.path.basename(args.other)))
    g = lb.topk(2, dim=-1).values
    gap = g[:, 0] - g[:, 1]
    print("    top1-top2 gap               median %.3f, p10 %.3f"
          % (gap.median().item(), gap.kthvalue(max(1, gap.numel() // 10)).values.item()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["dump", "engine", "pair"])
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                   help="attention implementation for the reference dump")
    p.add_argument("--out", default="/tmp/qwen38-ref.pt")
    p.add_argument("--ref", default="/tmp/qwen38-ref.pt")
    p.add_argument("--other", default="/tmp/qwen38-ref-eager.pt")
    p.add_argument('--floor', default='/tmp/qwen38-ref-eager.pt',
                   help='a second reference dump, for the model own noise floor')
    p.add_argument("--layers", action="store_true",
                   help="compare the per-layer hidden states, to locate a divergence")
    a = p.parse_args()
    {"dump": do_dump, "engine": do_engine, "pair": do_pair}[a.mode](a)
