"""ENG-135/221: the Flash-Next engine against the official implementation, on a real pack, on the board.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_refcheck.py --pack ~/fn2/packs/3.05 --out r.json

The engine is loaded once (the trunk, ~48 GiB). Then, layer by layer, the official
`Qwen4ExpTextDecoderLayer` (transformers 5.17 qwen4_exp, eager attention, bf16) is built on the
dequantised weights the engine serves -- EXL3 projections through `EXL3Block.dequant()`, the mixers
as their int8 rows dequantised, the n-gram rows as the engine decodes them -- and fed the same input
streams as the engine's layer. Only the routed experts the official router picks are materialised
(a compact bank, as tools/fn_refcheck.py on branch flashnext did). The next layer's input is the
engine's output, so each row measures one layer. With every layer the final mixer and the head run
too, and both greedy next tokens are printed. Rows: max |diff|, |diff| / |want| (Frobenius), the
output's scale, MemAvailable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402

PROMPT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    return 1e9


def dense(v, dtype):
    if hasattr(v, "bf16_cached"):                  # EXL3Block
        return v.dequant().to(dtype)
    if hasattr(v, "q"):                            # Int8Rows
        return v.dequant().to(dtype)
    return v


def official_layer(tcfg, li: int, eng: FlashNextEngine, dtype, dev):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    W, bank, c = eng.W, eng.banks[li], eng.cfg
    with torch.device("meta"):
        L = m.Qwen4ExpTextDecoderLayer(tcfg, li)
    tiny = lambda: torch.nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)  # noqa: E731
    L.mlp.experts.gate_up_proj, L.mlp.experts.down_proj = tiny(), tiny()
    if L.ple is not None:
        L.ple.ple_embedding.ngram_embedding.weight = tiny()
    L = L.to_empty(device=dev).to(dtype)
    pre = f"layers.{li}."
    E = c.num_experts
    for name, p in list(L.named_parameters()) + list(L.named_buffers()):
        if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj") or "ngram_embedding.weight" in name:
            continue
        key = pre + name
        if name.startswith("mlp.shared_expert."):
            j = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}[name.split(".")[2]]
            src = bank.block(E, j).dequant()
        elif "hyper_connection.input_mix_weight_down" in name or "hyper_connection.block_inject_weight" in name:
            site = key.rsplit(".", 2)[0]
            di = W[site + ".down_inject"]
            full = dense(di, torch.float32)
            R = W[site + ".rank"]
            src = full[:R] if "input_mix_weight_down" in name else full[R:]
        elif name.startswith("ple.ple_embedding."):
            tab = eng.tables[li]
            src = {"layer_multipliers": tab.multipliers, "ngram_heads_vocab_sizes": tab.sizes,
                   "ngram_heads_offsets": tab.offsets}[name.rsplit(".", 1)[1]]
        else:
            src = dense(W[key], dtype)
        with torch.no_grad():
            p.data = src.to(dev).to(p.dtype if p.is_floating_point() else src.dtype).reshape(p.shape).clone()
    ex = L.mlp.experts
    fwd = type(ex).forward

    def experts_forward(hidden_states, top_k_index, top_k_weights):
        uniq = torch.unique(top_k_index)
        remap = torch.full((int(uniq.max()) + 1,), -1, dtype=torch.long, device=dev)
        remap[uniq] = torch.arange(uniq.numel(), device=dev)
        gu = [torch.cat([bank.block(int(e), 0).dequant(), bank.block(int(e), 1).dequant()]) for e in uniq]
        ex.gate_up_proj = torch.nn.Parameter(torch.stack(gu).to(dtype), requires_grad=False)
        ex.down_proj = torch.nn.Parameter(torch.stack([bank.block(int(e), 2).dequant() for e in uniq]).to(dtype),
                                          requires_grad=False)
        ex.num_experts = uniq.numel()
        return fwd(ex, hidden_states, remap[top_k_index], top_k_weights)
    ex.forward = experts_forward
    if L.ple is not None:
        L.ple.ple_embedding.ngram_embedding.forward = lambda ids: eng.tables[li].gather(ids, dtype=dtype, device=dev)
    return L.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--out")
    ap.add_argument("--first", type=int, default=None, help="only the first N layers (a debug model)")
    a = ap.parse_args()
    dev = "cuda"
    dtype = torch.bfloat16
    from transformers import AutoTokenizer
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    t0 = time.time()
    idx, W, banks, tables, timing = load(a.pack, dev, first=a.first)
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device=dev, max_len=4096)
    c = idx.cfg
    print(f"[refcheck] loaded in {time.time() - t0:.1f} s, MemAvailable {mem_available_gib():.1f} GiB", flush=True)
    raw = json.load(open(os.path.join(a.pack, "config.json")))["text_config"]
    tcfg = Qwen4ExpTextConfig(**{k: v for k, v in raw.items() if k not in ("model_type", "mtp")})
    tcfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(a.pack)
    ids = torch.tensor(tok(a.prompt)["input_ids"])
    T = ids.numel()
    layers = list(range(c.num_hidden_layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    rot = m.Qwen4ExpTextRotaryEmbedding(tcfg).to(dev)
    pos = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rot(torch.zeros(1, T, dtype=dtype, device=dev), pos)
    tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev)).view(1, 1, T, T)
    mask4 = torch.where(tril, torch.zeros((), dtype=dtype, device=dev),
                        torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=dev))
    st = eng.new_state(4096)
    positions = torch.arange(T, device=dev)
    h = eng.embed(ids).repeat(1, c.hc_count)
    rows = []
    t1 = time.time()
    for li in layers:
        L = official_layer(tcfg, li, eng, dtype, dev)
        with torch.no_grad():
            want = L(h[None], position_embeddings=(cos, sin), attention_mask=mask4, conv_mask=None,
                     past_key_values=None, ple_input_ids=ids[None].to(dev))
            want = (want[0] if isinstance(want, tuple) else want)[0]
            got = eng.layer(h, li, ids.to(dev), st, positions)
        d = got.float() - want.float()
        row = {"layer": li, "type": c.layer_types[li], "max_abs": round(d.abs().max().item(), 5),
               "rel": round((d.norm() / want.float().norm()).item(), 6),
               "scale": round(want.float().abs().max().item(), 3),
               "mem_avail_gib": round(mem_available_gib(), 1), "s": round(time.time() - t1, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        h = got
        del L
        torch.cuda.empty_cache()
    res = {"pack": a.pack, "prompt": a.prompt, "tokens": T, "layers": rows, "load": timing}
    if layers == list(range(c.num_hidden_layers)):
        x = eng.mixer(h, "hyper_connection_mixer", combine=False)
        ours = eng.lin(x[-1:], eng.W["lm_head.weight"]).float()[0]
        theirs = torch.nn.functional.linear(x[-1:], eng.W["lm_head.weight"].dequant()).float()[0]
        res["next_top5_engine"] = [(tok.decode([int(i)]), round(float(v), 3))
                                   for v, i in zip(*torch.topk(ours, 5))]
        res["next_top5_dequant_head"] = [(tok.decode([int(i)]), round(float(v), 3))
                                         for v, i in zip(*torch.topk(theirs, 5))]
        print("next token, engine head:", res["next_top5_engine"], flush=True)
        print("next token, dequantised head:", res["next_top5_dequant_head"], flush=True)
    res["max_rel"] = max(r["rel"] for r in rows)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
