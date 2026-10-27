"""ENG-225 on the board: the prefill kernels against the reference forward.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_prefill_check.py --pack ~/fn2/packs/3.05 \
        --ctx 1024,8192,32768 --classes prose,code --new 64 --json out.json

Per context and class (tools/fn_generate.py's prompts filled to `--ctx` tokens), two states read the
same prompt in the same chunks, one through the reference forward (`forward`), one through the prefill
kernels (`forward_pre`):

  teacher-forced   after each chunk, the next-token logits of every row of both (the same final mixer
                   and head on each path's own streams, in blocks of 256 rows): argmax agreement over
                   every position binned by the yardstick's top-2 margin, max |logit diff| per row, and
                   the disagreements as (position, both top-1s, both margins, max |logit diff|),
                   those where both paths are sure (both margins) first. With `--rows`, a
                   third state reads the prompt through the decode path's own kernels (verify blocks of
                   32 chain rows, each committed: the row-by-row arithmetic, bit-equal to decode steps),
                   and every pair of the three paths is compared the same way.
  continuation     greedy decode of `--new` tokens from each state (the decode graph): the two texts,
                   and at the first difference the reference state's own top-2 margin (a near tie or not).
  MTP prime        with `--mtp`: the MTP layer fed the prompt's streams and next tokens on each path
                   (`mtp_feed` / `mtp_pre`), its draft argmax per position compared the same way; and
                   the MTP layer alone (`mtp_same_streams`): both fed the reference's streams.

Also the wall time of each path's prompt (the kernel path's tok/s is ENG-225's number when the reference
does not run beside it: tools/fn_prefill_profile.py).
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
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402


@torch.no_grad()
def head_logits(eng: FlashNextEngine, stack: torch.Tensor) -> torch.Tensor:
    """fp32 next-token logits of rows from their pre-final-mixer streams (the final mixer and head)."""
    from tools import fn_kernels as K
    c = eng.cfg
    n = K.hc_norm(stack.contiguous(), eng._site_norm(c.num_hidden_layers), c.hidden_size, c.rms_norm_eps)
    return eng.final(n)


@torch.no_grad()
def pair_rows(eng: FlashNextEngine, sx: torch.Tensor, sy: torch.Tensor, block: int = 256) -> dict:
    """Per row of two paths' streams: each one's top-1 and top-1 - top-2 margin, and max |logit diff|."""
    out = {"ix": [], "mx": [], "iy": [], "my": [], "dmax": []}
    for r0 in range(0, sx.shape[0], block):
        lx, ly = head_logits(eng, sx[r0:r0 + block]), head_logits(eng, sy[r0:r0 + block])
        for (i_, m_), lg in ((("ix", "mx")), lx), ((("iy", "my")), ly):
            v, i = torch.topk(lg, 2, dim=-1)
            out[i_].append(i[:, 0])
            out[m_].append(v[:, 0] - v[:, 1])
        out["dmax"].append((lx - ly).abs().amax(-1))
        del lx, ly
    return {k: torch.cat(v).cpu() for k, v in out.items()}


BINS = (0.0, 0.5, 1.0, 2.0, 5.0, float("inf"))


def compare(r: dict) -> dict:
    """Agreement of two paths (x = the yardstick), binned by x's margin; max |logit diff| percentiles;
    the disagreements in order of the smaller of the two margins, largest first, as (position, x top-1,
    y top-1, x margin, y margin, max |logit diff|); how many have both margins above 5."""
    eq = r["ix"] == r["iy"]
    mx, my, d = r["mx"], r["my"], r["dmax"]
    bins = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (mx >= lo) & (mx < hi)
        n = int(sel.sum())
        bins[f"[{lo:g},{hi:g})"] = {"n": n, "agree": round(float(eq[sel].float().mean()), 4) if n else None}
    bad = (~eq).nonzero().view(-1)
    key = torch.minimum(mx[bad], my[bad])
    order = bad[torch.argsort(key, descending=True)]
    rows = [(int(p), int(r["ix"][p]), int(r["iy"][p]), round(float(mx[p]), 3), round(float(my[p]), 3),
             round(float(d[p]), 3)) for p in order[:12]]
    ds = torch.sort(d).values
    return {"positions": int(eq.numel()), "agree": int(eq.sum()), "agree_frac": round(float(eq.float().mean()), 5),
            "by_margin": bins, "both_margins_over_5": int(((~eq) & (mx > 5) & (my > 5)).sum()),
            "max_logit_diff": {"p50": round(float(ds[len(ds) // 2]), 3), "p99": round(float(ds[int(len(ds) * 0.99)]), 3),
                               "max": round(float(ds[-1]), 3)},
            "outliers": rows}


@torch.no_grad()
def mtp_top2(eng: FlashNextEngine, h: torch.Tensor, block: int = 256):
    """The same for the MTP layer's own streams (its final mixer, the shared head)."""
    ids, margin = [], []
    for r0 in range(0, h.shape[0], block):
        x = eng.mixer(h[r0:r0 + block], "mtp.hyper_connection_mixer", combine=False)
        v, i = torch.topk(eng.lin(x, eng.W["lm_head.weight"]).float(), 2, dim=-1)
        ids.append(i[:, 0])
        margin.append(v[:, 0] - v[:, 1])
    return torch.cat(ids), torch.cat(margin)


@torch.no_grad()
def rows_path(eng: FlashNextEngine, ids: list[int], st, block: int = 32) -> torch.Tensor:
    """The prompt through verify blocks of `block` chain rows, each committed: every row's streams."""
    out = []
    for i in range(0, len(ids), block):
        toks = ids[i:i + block]
        eng.verify(toks, list(range(-1, len(toks) - 1)), st)
        out.append(eng.rec["stack"][:len(toks)].clone())
        eng.commit(st, list(range(len(toks))))
    eng.flush()
    return torch.cat(out)


@torch.no_grad()
def check(eng: FlashNextEngine, ids: list[int], chunk: int, new: int, eos: set, mtp: bool,
          rows: bool = False) -> dict:
    dev = eng.device
    A, B = eng.new_state(eng.max_len), eng.new_state(eng.max_len)
    sa, sb = [], []
    ta = tb = 0.0
    for i in range(0, len(ids), chunk):
        t = torch.tensor(ids[i:i + chunk], device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.forward(t, A, head="none")
        sa.append(eng.stack)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        eng.forward_pre(t, B, head="none")
        sb.append(eng.stack)
        torch.cuda.synchronize()
        tb += time.perf_counter() - t1
        ta += t1 - t0
        if A.length > 4096:
            torch.cuda.empty_cache()
    eng.build_pool(A)
    sa, sb = torch.cat(sa), torch.cat(sb)
    ab = pair_rows(eng, sa, sb)
    res = {"prompt_tokens": len(ids), "ref_s": round(ta, 2), "kernels_s": round(tb, 2), "teacher_forced": compare(ab)}
    ma = ab["mx"]
    states = [(A, int(ab["ix"][-1])), (B, int(ab["iy"][-1]))]
    if rows:
        C = eng.new_state(eng.max_len)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sc = rows_path(eng, ids, C)
        res["rows_s"] = round(time.perf_counter() - t0, 2)
        ac, cb = pair_rows(eng, sa, sc), pair_rows(eng, sc, sb)
        res["rows_vs_ref"] = compare(ac)
        res["kernels_vs_rows"] = compare(cb)
        mc = cb["mx"]
        states.append((C, int(cb["ix"][-1])))
    # continuation: greedy from each state (the first token is each path's own last-row argmax)
    outs = []
    for st, first in states:
        g = DecodeGraph(eng, st)
        toks, margins = [first], []
        while len(toks) < new and toks[-1] not in eos:
            lg = g(toks[-1])[0]
            v, i = torch.topk(lg, 2)
            toks.append(int(i[0]))
            margins.append(float(v[0] - v[1]))
        outs.append((toks, margins))
        del g

    def cont(x, y, m0):
        (tx, mg), (ty, _) = x, y
        d = next((j for j in range(min(len(tx), len(ty))) if tx[j] != ty[j]), None)
        return {"tokens": len(tx), "equal": tx == ty, "first_diff": d,
                "margin_at_diff": None if d is None else round(float(m0) if d == 0 else mg[d - 1], 4)}
    res["continuation"] = cont(outs[0], outs[1], ma[-1])
    if rows:
        res["continuation_rows_vs_ref"] = cont(outs[0], outs[2], ma[-1])
        res["continuation_kernels_vs_rows"] = cont(outs[2], outs[1], mc[-1])
    if mtp:
        L = len(ids)
        ma_st, mb_st, mc_st = (eng.new_mtp_state(eng.max_len) for _ in range(3))
        xa, xm, xb, xc = [], [], [], []
        for i in range(0, L - 1, chunk):
            j = min(i + chunk, L - 1)
            nt = torch.tensor(ids[i + 1:j + 1])
            _, ha = eng.mtp(sa[i:j], nt.to(dev), ma_st, head="x")
            hb = eng.mtp_pre(sb[i:j], nt, mb_st)
            hc = eng.mtp_pre(sa[i:j], nt, mc_st)
            a1, a2 = mtp_top2(eng, ha)
            b1, b2 = mtp_top2(eng, hb)
            c1, c2 = mtp_top2(eng, hc)
            xa.append(a1), xm.append(a2), xb.append((b1, b2)), xc.append((c1, c2))
        xa, xm = torch.cat(xa).cpu(), torch.cat(xm).cpu()
        for name, got in (("mtp_prime", xb), ("mtp_same_streams", xc)):
            iy, my = torch.cat([g[0] for g in got]).cpu(), torch.cat([g[1] for g in got]).cpu()
            res[name] = compare({"ix": xa, "mx": xm, "iy": iy, "my": my, "dmax": torch.zeros_like(xm)})
    del A, B, states
    torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", default="1024,8192")
    ap.add_argument("--classes", default="prose,code")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--rows", action="store_true", help="also the decode path's own kernels (verify blocks)")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    ctxs = [int(x) for x in a.ctx.split(",")]
    idx, W, banks, tables, timing = load(a.pack, "cuda", mtp=a.mtp)
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max(ctxs) + a.new + 64)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    res = {"pack": a.pack, "load": timing, "mem_after_load_gib": mem_available_gib(),
           "knobs": {"ctx": ctxs, "chunk": a.chunk, "new": a.new, "greedy": True,
                     "kv_dtype": "bf16", "mtp": a.mtp, "rows_path": a.rows}, "rows": []}
    for n in ctxs:
        for c in a.classes.split(","):
            r = {"ctx": n, "class": c, **check(eng, class_prompt(tok, c, n), a.chunk, a.new, eos, a.mtp, a.rows)}
            r["mem_gib"] = mem_available_gib()
            res["rows"].append(r)
            print(f"[prefill-check] {json.dumps(r)}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
