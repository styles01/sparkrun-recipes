"""ENG-219/145: our reading of a Flash-Next pack against exllamav3's own, tensor by tensor.

    PYTHONPATH=<repo> ~/exl3k/venv/bin/python tools/fn_exl3_xcheck.py --pack ~/fn2/packs/3.05 --json out.json

For routed experts (several layers, gate/up/down), dense projections of every K the pack uses
(GDN, attention, the indexer at K3), a shared expert and a slice of the head: our trellis decode
(`EXL3Block.inner`, fp16 [Kin, N]) must equal exllamav3's `LinearEXL3.get_inner_weight_tensor()` bit
for bit (the lossy part of the format), and the dequantised weight (`EXL3Block.dequant`, the fp32
rotations rounded once to bf16) is held against exllamav3's `get_weight_tensor()` (its own rounding
order): max |diff| / max |w|. For n-gram rows: our `EXL3NGramTable.gather` in fp32 must equal
exllamav3's `ngram_codec.dequant_rows` on the same packed rows, bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.pack import PackIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from exllamav3.modules.quant.exl3_lib import ngram_codec as NC
    idx = PackIndex.open(a.pack)
    dev = "cuda"
    bases = [f"layers.{li}.mlp.experts.{e}.{x}_proj" for li, e in ((0, 0), (11, 300), (47, 511))
             for x in ("gate", "up", "down")]
    bases += ["layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.out_proj", "layers.3.self_attn.q_proj",
              "layers.3.self_attn.o_proj", "layers.3.self_attn.indexer.index_qk_proj",
              "layers.5.mlp.shared_expert.down_proj", "lm_head"]
    rows = []
    for b in bases:
        blk = idx.exl3(b, dev)
        if b == "lm_head":                       # a 16,384-column slice (the whole head is 2.5 GB in fp32)
            from tools.exl3_linear import EXL3Block
            blk = EXL3Block(blk.t[:, :1024].contiguous(), blk.suh, blk.svh[:16384])
        lin = LinearEXL3(None, blk.K, blk.N, suh=blk.suh[0].contiguous(), svh=blk.svh.contiguous(),
                         trellis=blk.t.contiguous(), mul1=idx.read(b + ".mul1", dev))
        ours_i = blk.inner()
        theirs_i = lin.get_inner_weight_tensor()
        inner_eq = bool(torch.equal(ours_i, theirs_i.to(ours_i.dtype)))
        w = blk.dequant().float()
        tw = lin.get_weight_tensor().float().T
        rel = ((w - tw).abs().max() / tw.abs().max()).item()
        r = {"tensor": b, "K": blk.kb, "shape": [blk.N, blk.K], "inner_bit_equal": inner_eq, "weight_rel": rel}
        rows.append(r)
        print(json.dumps(r), flush=True)
        del lin, ours_i, theirs_i, w, tw
        torch.cuda.empty_cache()
    tab = idx.table
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, tab.rows, (4096,), generator=g)
    ours = tab.gather(ids, dtype=torch.float32)
    packed = tab.gather_packed(ids)
    heads = tab.head_of(ids)
    theirs = NC.dequant_rows(packed, tab.K, NC.mul1_codebook("cpu"), tab.bias.float()[heads])
    ng = {"rows": int(ids.numel()), "K": tab.K, "bit_equal": bool(torch.equal(ours, theirs.float()))}
    print(json.dumps({"ngram": ng}), flush=True)
    res = {"pack": a.pack, "tensors": rows, "ngram": ng,
           "all_inner_bit_equal": all(r["inner_bit_equal"] for r in rows),
           "max_weight_rel": max(r["weight_rel"] for r in rows)}
    print(json.dumps({k: res[k] for k in ("all_inner_bit_equal", "max_weight_rel")}), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
