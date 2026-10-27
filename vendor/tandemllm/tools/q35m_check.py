"""ENG-391: a Qwen3.5 MoE EXL3 pack on the board, checked against itself (the GPU suite, on the Spark).

  1. the prompt through the prefill kernels and through the reference forward on the same weights
     (top-1 of the last row, max |diff|)
  2. greedy decode steps one at a time (eager, then from the captured decode graph): equal tokens
  3. the same tokens as verify blocks of 2..32 rows (chains, and one tree): every row's logits are the
     decode step's bit for bit, its pick the next greedy token
  4. the expert kernels' self-check (verify widths against the decode rows on real routing)
  5. plain decode tok/s from the graph, and the text

    python tools/q35m_check.py --pack ~/models/ornith-1.5-122b-exl3-4.0 [--steps 32] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from engine.models.flashnext.engine import DecodeGraph
    from engine.models.qwen3_5_moe.engine import Qwen35MoeEngine
    from engine.models.qwen3_5_moe.load import load
    out: dict = {"pack": a.pack}
    t0 = time.time()
    idx, W, banks, timing = load(a.pack, "cuda", mtp=True)
    eng = Qwen35MoeEngine(idx.cfg, W, banks, device="cuda", max_len=a.max_len)
    out["load_s"] = round(time.time() - t0, 1)
    out["allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    tok = AutoTokenizer.from_pretrained(a.pack)
    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
    # 1. prefill kernels vs the reference forward
    st = eng.new_state(a.max_len)
    lk = eng.prefill(ids, st)
    st_r = eng.new_state(a.max_len)
    lr = eng.prefill(ids, st_r, path="ref")
    out["prefill_vs_ref"] = {"top1_equal": int(lk.argmax()) == int(lr.argmax()),
                             "max_abs_diff": round(float((lk - lr).abs().max()), 4),
                             "logit_scale": round(float(lr.abs().max()), 2)}
    print("[q35m-check] prefill vs ref:", out["prefill_vs_ref"], flush=True)
    del st_r
    # 2. greedy decode, eager
    first = int(lk.argmax())
    toks, rows = [first], []
    for _ in range(a.steps):
        lg = eng.decode_eager(toks[-1], st)
        rows.append(lg[0].clone())
        toks.append(int(lg[0].argmax()))
    out["text"] = tok.decode(toks)
    print("[q35m-check] text:", repr(out["text"]), flush=True)
    # ... and from the decode graph
    st_g = eng.new_state(a.max_len)
    eng.prefill(ids, st_g)
    dg = DecodeGraph(eng, st_g)
    gt = [first]
    for _ in range(a.steps):
        gt.append(dg.greedy(gt[-1]))
    out["graph_equals_eager"] = gt == toks
    # 3. verify blocks against the decode rows
    bad = []
    for M in (2, 4, 8, 16, 32):
        if M > a.steps:
            break
        st_v = eng.new_state(a.max_len)
        eng.prefill(ids, st_v)
        picks = eng.verify(toks[:M], list(range(-1, M - 1)), st_v)
        lg = eng._rb[M]["logits"][:M]
        for i in range(M):
            if not torch.equal(lg[i], rows[i]) or picks[i] != toks[i + 1]:
                bad.append({"M": M, "row": i})
                break
        del st_v
    # a tree: row 0, two children (the greedy one and another token), the greedy child's child
    st_v = eng.new_state(a.max_len)
    eng.prefill(ids, st_v)
    other = (toks[1] + 1) % idx.cfg.vocab_size
    picks = eng.verify([toks[0], other, toks[1], toks[2]], [-1, 0, 0, 2], st_v)
    lg = eng._rb[4]["logits"][:4]
    tree_ok = torch.equal(lg[0], rows[0]) and torch.equal(lg[2], rows[1]) and torch.equal(lg[3], rows[2])
    out["verify_rows_equal_decode"] = {"chains_2_to_32": not bad, "first_bad": bad[:1], "tree": tree_ok}
    print("[q35m-check] verify rows:", out["verify_rows_equal_decode"], flush=True)
    del st_v
    # 4. the expert kernels' self-check
    out["self_check"] = eng.self_check(ids.tolist(), steps=16, log=None)
    print("[q35m-check] self-check:", out["self_check"], flush=True)
    # 5. plain decode speed from the graph (thinking off, the prompt's continuation)
    st_s = eng.new_state(a.max_len)
    eng.prefill(ids, st_s)
    dg = DecodeGraph(eng, st_s)
    t = first
    for _ in range(4):
        t = dg.greedy(t)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    n = 128
    for _ in range(n):
        t = dg.greedy(t)
    dt = time.perf_counter() - t1
    out["decode_tok_s"] = round(n / dt, 2)
    out["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(f"[q35m-check] decode {out['decode_tok_s']} tok/s; load {out['load_s']} s, "
          f"{out['allocated_gib']} GiB after load", flush=True)
    ok = (out["prefill_vs_ref"]["top1_equal"] and out["graph_equals_eager"] and not bad and tree_ok
          and out["self_check"]["ok"])
    out["ok"] = bool(ok)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    print("[q35m-check]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
