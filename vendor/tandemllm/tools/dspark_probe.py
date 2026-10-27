"""Does the DSpark drafter load, run, and propose tokens the target agrees with?

Three questions in order, because a drafter that is wired wrong still returns tokens:

  1. build     every expected tensor present, the module constructed, the byte budget printed
  2. propose   one block against a real prompt, with the target's own continuation beside it --
               a drafter reading its positions or its norms wrong does not get slot 0 right
  3. accept    accepted tokens a block over a short generation, beside DFlash2 on the same prompt

Losslessness is not asked here and cannot be: a drafter cannot change greedy output. That gate is
`tools/verify_spec.py`, and it is run separately against this drafter like any other.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_spec  # noqa: E402

PROMPT = ("Write four paragraphs about the way a harbour town wakes up in winter.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--dspark", default=None)
    ap.add_argument("--ckpt8", default=None)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--no-markov", action="store_true")
    ap.add_argument("--taps", default="entry,output",
                    help="which hidden state target_layer_ids names")
    a = ap.parse_args()

    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.drafters.dspark import DSparkDrafter
    from transformers import AutoTokenizer

    cfg = load_config(a.model)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    ids = tok(tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False),
              return_tensors="pt").input_ids[0].cuda()
    eos = cfg.eos_token_ids

    print("=== 1. build ===", flush=True)
    tapmodes = a.taps.split(",")
    d = DSparkDrafter(eng, a.dspark, markov=not a.no_markov, max_len=a.max_len,
                      tap=tapmodes[0])
    d._build()
    c = d.cfg
    print(f"block_size {c.block_size}  layers {c.num_hidden_layers}  heads {c.num_attention_heads}"
          f"/{c.num_key_value_heads}x{c.head_dim}  inter {c.intermediate_size}  "
          f"rope {c.rope_type} factor {c.rope_factor}  taps {c.target_layer_ids}  "
          f"mask {c.mask_token_id}  markov {c.markov_rank}  conf {c.enable_confidence_head}")
    b = d.draft_bytes()
    print(f"bytes/draft: backbone {b['backbone_B'] / 1e9:.3f} GB  markov "
          f"{b['markov_B'] / 1e9:.3f} GB  head {b['head_B'] / 1e9:.3f} GB  "
          f"total {b['total_B'] / 1e9:.3f} GB for {b['proposals']} proposals")

    print("\n=== 2. propose, against the target's own continuation ===", flush=True)
    # `target_layer_ids` names either the residual ENTERING that layer or its OUTPUT, one index
    # apart, and no amount of reading settles it -- the reference's `hidden_states[layer_id + 1]`
    # is HF's indexing, which this engine's tap numbers differently. So both are run and the
    # accepted prefix decides, which is the same way the DFlash2 header says to settle it.
    best = None
    for mode in tapmodes:
        dd = (d if mode == tapmodes[0] else
              DSparkDrafter(eng, a.dspark, markov=not a.no_markov, max_len=a.max_len, tap=mode))
        dd._build()
        dd.attach()
        dd.reset()
        eng.reset()
        with torch.no_grad():
            logits = eng.forward(ids, start=0, last_only=True)
            dd.sync(ids.tolist(), eng.hidden_post_norm[0], 0)
        anchor = int(logits[0, -1].argmax())
        ctx = ids.tolist() + [anchor]
        draft = dd.propose(ctx, c.block_size - 1)
        with torch.no_grad():
            blk = torch.tensor([anchor] + draft, device=ids.device)
            lg = eng.forward_block(blk, start=ids.numel())
            picks = lg.argmax(-1).tolist()
            eng.rollback_to(1)
        agree = 0
        for x, y in zip(draft, picks):
            if x != y:
                break
            agree += 1
        print(f"tap={mode:<7} draft  {draft} {tok.decode(draft)!r}")
        print(f"{'':12} target {picks[:-1]} {tok.decode(picks[:-1])!r}")
        if dd.last_conf:
            print(f"{'':12} conf   {[round(x, 3) for x in dd.last_conf]}")
        print(f"{'':12} accepted prefix {agree} of {len(draft)}")
        if best is None or agree > best[0]:
            best = (agree, mode, dd)
        dd.detach()
    d = best[2]
    print(f"--> using tap={best[1]}")

    print("\n=== 3. accepted tokens a block, DSpark vs DFlash2 ===", flush=True)
    rows = []
    for name, dr, width in (("dspark", d, c.block_size - 1),
                            ("dflash2-b8", None, 7)):
        if dr is None:
            if not a.ckpt8:
                continue
            dr = DFlash2Drafter(eng, a.ckpt8, blocks=1, path="greedy", max_len=a.max_len, block=8)
            dr._build()
            width = dr.cfg.block_size - 1
        dr.attach()
        for _ in range(2):                      # warm, then measure
            out, st = generate_spec(eng, ids, a.new, dr, width, eos)
        dr.detach()
        rows.append((name, st, out))
        print(f"{name:<12} {st.tok_s:7.2f} tok/s  {st.accept_len:5.2f} accepted/block  "
              f"{st.accept_rate * 100:5.1f} % of draft  {st.blocks:4d} blocks  "
              f"draft {st.draft_s / max(st.blocks, 1) * 1e3:5.1f} ms", flush=True)
    if len(rows) == 2:
        same = rows[0][2] == rows[1][2]
        print(f"\nsame greedy output from both drafters: {same}")


if __name__ == "__main__":
    main()
