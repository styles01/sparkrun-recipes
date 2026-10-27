"""The quantiser reads the BF16 release: sources, the clip search, GPTQ on the NVFP4 grid, the tap,
`build`, and the FP8/NVFP4 mix. CPU only, on tiny checkpoints written to disk the way the releases
lay them out.

  * `Source` finds every projection by the checkpoint's index in both layouts, skips the vision
    tower and the MTP layer, and returns the FP8 release's weight exactly as the old reader did.
  * GPTQ with an identity second moment is the clip search, byte for byte (no correlation, nothing
    to push onto the next column); with a correlated one it leaves less output error than clip.
  * the tap names every projection's input, and projections that share an input see the same one.
  * `build` writes files the loader overlays on a BF16 checkpoint, and the pass split (the
    second-moment budget) does not change a byte.
  * `mix` keeps exactly the units the knapsack chose in FP8 and writes the rest.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from engine.loader import LM_PREFIX, Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from test_refactor_gate import TinyWeights, config, quantize_fp8  # noqa: E402
from tools import quant_nvfp4 as qn  # noqa: E402
from tools import quant_sensitivity as qs  # noqa: E402
from tools.nvfp4_linear import NVFP4Block  # noqa: E402


def _text_config(cfg) -> dict:
    return {"hidden_size": cfg.hidden_size, "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers, "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
            "vocab_size": cfg.vocab_size, "rms_norm_eps": cfg.rms_norm_eps,
            "max_position_embeddings": cfg.max_position_embeddings, "layer_types": cfg.layer_types,
            "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
            "linear_key_head_dim": cfg.linear_key_head_dim,
            "linear_value_head_dim": cfg.linear_value_head_dim,
            "linear_num_key_heads": cfg.linear_num_key_heads,
            "linear_num_value_heads": cfg.linear_num_value_heads,
            "rope_parameters": {"rope_theta": cfg.rope_theta,
                                "partial_rotary_factor": cfg.partial_rotary_factor,
                                "mrope_section": cfg.mrope_section}}


def _key(name: str) -> str:
    return name if name == "lm_head.weight" else LM_PREFIX + name


def bf16_checkpoint(cfg, seed=3):
    """The BF16 release's layout: two shards with an index, layers split across them, the MTP layer
    and a vision tensor inside the last shard."""
    tw = TinyWeights(cfg, "bf16", seed=seed)
    plain = {}
    for name, t in tw.t.items():
        key = _key(name)
        if not name.endswith(".weight") and t.dim() == 2:
            key += ".weight"
        plain[key] = t.to(torch.bfloat16).contiguous()
    plain["mtp.layers.0.mlp.gate_proj.weight"] = torch.zeros(cfg.intermediate_size, cfg.hidden_size,
                                                             dtype=torch.bfloat16)
    plain["mtp.fc.weight"] = torch.zeros(cfg.hidden_size, 2 * cfg.hidden_size, dtype=torch.bfloat16)
    plain["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"text_config": _text_config(cfg), "tie_word_embeddings": False}, f)
    keys = sorted(plain)
    shards = {"model-00001-of-00002.safetensors": keys[: len(keys) // 2],
              "model-00002-of-00002.safetensors": keys[len(keys) // 2:]}
    wm = {}
    for fname, ks in shards.items():
        save_file({k: plain[k] for k in ks}, os.path.join(d, fname))
        wm.update({k: fname for k in ks})
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": wm}, f)
    return d, plain


def fp8_checkpoint(cfg, seed=3):
    """The FP8 release's layout: one file per layer, every projection an e4m3 pair."""
    tw = TinyWeights(cfg, "fp8", seed=seed)
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"text_config": _text_config(cfg), "tie_word_embeddings": False}, f)
    files: dict[str, dict] = {}
    for name, t in tw.t.items():
        fname = f"layers-{name.split('.')[1]}.safetensors" if name.startswith("layers.") \
            else "outside.safetensors"
        key = _key(name)
        if hasattr(t, "w") and hasattr(t, "s"):
            files.setdefault(fname, {})[key + ".weight"] = t.w
            files[fname][key + ".weight_scale_inv"] = t.s
        else:
            files.setdefault(fname, {})[key] = t
    wm = {}
    for fname, tensors in files.items():
        save_file(tensors, os.path.join(d, fname))
        wm.update({k: fname for k in tensors})
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": wm}, f)
    return d, tw


def test_source_bf16_layout():
    cfg = config("fp8")
    d, plain = bf16_checkpoint(cfg)
    src = qn.Source(d)
    n_proj = sum(1 for k in plain if k.startswith(LM_PREFIX + "layers.")
                 and k[len(LM_PREFIX):-len(".weight")].split(".", 2)[2] in qn.TARGETS["all"])
    assert src.kind == "bf16" and len(src.where) == n_proj, (src.kind, len(src.where), n_proj)
    assert not any("mtp" in b or "visual" in b for b in src.where)
    base = "layers.0.mlp.down_proj"
    assert torch.equal(src.read(base), plain[LM_PREFIX + base + ".weight"].float())
    assert src.read("layers.0.self_attn.q_proj") is None           # a GDN layer has no attention
    N, K = plain[LM_PREFIX + base + ".weight"].shape
    assert src.stored_bytes(base) == N * K + (N // 128) * (K // 128) * 2
    return f"{len(src.where)} projections found across two shards; MTP and vision skipped"


def test_source_fp8_matches_old_reader():
    cfg = config("fp8")
    d, tw = fp8_checkpoint(cfg)
    src = qn.Source(d)
    assert src.kind == "fp8"
    for base in ("layers.0.mlp.gate_proj", "layers.3.self_attn.o_proj", "layers.1.linear_attn.in_proj_z"):
        blk = tw.t[base]
        old = qn.fp8_dequant(blk.w, blk.s)                          # the reader the served files came from
        assert torch.equal(src.read(base), old), base
    return "FP8 pairs read exactly as the served files' reader read them"


def test_loader_skips_mtp_in_shards():
    cfg = config("fp8")
    d, _ = bf16_checkpoint(cfg)
    w = Weights(d, device="cpu", skip_mtp=True, nvfp4="", fp8_head="")
    assert not any(k.startswith("mtp.") for k in list(w.q) + list(w.t))
    w2 = Weights(d, device="cpu", skip_mtp=False, nvfp4="", fp8_head="")
    assert any(k.startswith("mtp.") for k in list(w2.q) + list(w2.t))
    return "MTP tensors inside a shard are skipped with skip_mtp"


def test_gptq_identity_is_clip():
    g = torch.Generator().manual_seed(0)
    W = (torch.randn(96, 256, generator=g) * 0.05).to(torch.bfloat16).float()
    H = torch.eye(256)
    a = qn.gptq_nvfp4(W, H, damp=0.0)
    b = qn.quantize_clipped(W.to(torch.bfloat16), torch.ones(256))
    assert a.s2 == b.s2
    assert torch.equal(a.s.view(torch.uint8), b.s.view(torch.uint8)), "scales differ"
    assert torch.equal(a.w, b.w), "codes differ"
    return "identity second moment: GPTQ codes and scales == clip search, byte for byte"


def test_gptq_beats_clip_on_correlated_inputs():
    g = torch.Generator().manual_seed(1)
    K, N, T = 256, 128, 4096
    mix = torch.randn(K, K, generator=g) * 0.3 + torch.eye(K)
    X = torch.randn(T, K, generator=g) @ mix                     # correlated input channels
    X[:, :8] *= 6.0                                              # a few channels the model leans on
    H = X.t() @ X
    W = (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16).float()
    act = (X.pow(2).mean(0))
    clip = qn.quantize_clipped(W.to(torch.bfloat16), act)
    gptq = qn.gptq_nvfp4(W, H)
    e_clip, e_gptq = qn.output_error(W, clip, H), qn.output_error(W, gptq, H)
    assert e_gptq < 0.8 * e_clip, (e_gptq, e_clip)
    # the format is the engine's: codes decode to values on the grid times the group scales
    deq = gptq.dequant().float()
    assert deq.shape == W.shape and torch.isfinite(deq).all()
    return f"relative output error: clip {e_clip:.5f}, gptq {e_gptq:.5f} ({e_gptq / e_clip:.2f}x)"


def _tiny_build(d, tmp, budget_gb, methods="clip,gptq", corpus=None):
    corpus = corpus or os.path.join(tmp, "ids.pt")
    if not os.path.exists(corpus):
        torch.save(torch.randint(1, 211, (600,), generator=torch.Generator().manual_seed(5)), corpus)
    out = os.path.join(tmp, f"out-{budget_gb:g}")
    args = types.SimpleNamespace(model=d, corpus=corpus, tokens=600, chunk=128, methods=methods,
                                 targets="all", ratios="1,0.95,0.9,0.85,0.8", damp=0.01,
                                 h_budget_gb=budget_gb, layers=0, device="cpu", out_dir=out)
    qn.cmd_build(args)
    return out


def test_tap_names_every_input_and_shared_inputs_agree():
    cfg = config("fp8")
    d, _ = bf16_checkpoint(cfg)
    w = Weights(d, device="cpu", skip_mtp=True, nvfp4="", fp8_head="")
    eng = Qwen38Engine(cfg, w, max_len=80, device="cpu")
    sums = {}

    def fn(name, x):
        sums[name] = sums.get(name, 0) + x.float().pow(2).sum(0)

    ids = torch.randint(1, 211, (64,), generator=torch.Generator().manual_seed(2))
    with qn.ProjTap(w, fn) as tap:
        eng.reset()
        with torch.no_grad():
            eng.forward(ids, start=0, last_only=True)
    want = {b for b in w.q if b.startswith("layers.")}
    assert tap.seen == want, sorted(want - tap.seen)
    for b in want:
        lead = qn.input_key(b)
        assert torch.equal(sums[b], sums[lead]), (b, lead)
    return f"{len(want)} projections tapped; members of a shared input see the same activation"


def test_build_loads_and_passes_do_not_change_bytes():
    cfg = config("fp8")
    d, _ = bf16_checkpoint(cfg)
    tmp = tempfile.mkdtemp()
    one = _tiny_build(d, tmp, 10.0)
    many = _tiny_build(d, tmp, 1e-7)                      # one layer a pass
    rep = json.load(open(os.path.join(many, "report.json")))
    assert rep["passes"] == cfg.num_hidden_layers, rep["passes"]
    for m in ("clip", "gptq"):
        for t in ("mlp", "gdn", "attn"):
            a, b = os.path.join(one, m, f"{t}.safetensors"), os.path.join(many, m, f"{t}.safetensors")
            with safe_open(a, "pt") as fa, safe_open(b, "pt") as fb:
                assert set(fa.keys()) == set(fb.keys())
                for k in fa.keys():
                    assert torch.equal(fa.get_tensor(k).view(torch.uint8) if fa.get_tensor(k).dtype
                                       == torch.float8_e4m3fn else fa.get_tensor(k),
                                       fb.get_tensor(k).view(torch.uint8) if fb.get_tensor(k).dtype
                                       == torch.float8_e4m3fn else fb.get_tensor(k)), (m, t, k)
                assert fa.metadata()["source_kind"] == "bf16" and fa.metadata()["mode"] == m
    # the loader overlays the files on the BF16 checkpoint, and the model still runs
    files = ",".join(os.path.join(one, "gptq", f"{t}.safetensors") for t in ("mlp", "gdn", "attn"))
    w = Weights(d, device="cpu", skip_mtp=True, nvfp4=files, fp8_head="")
    n4 = sum(1 for v in w.q.values() if isinstance(v, NVFP4Block))
    assert n4 == len(w.q), (n4, len(w.q))
    eng = Qwen38Engine(cfg, w, max_len=64, device="cpu")
    with torch.no_grad():
        out = eng.forward(torch.arange(1, 17), start=0, last_only=True)
    assert torch.isfinite(out.float()).all()
    e = [(r["clip"], r["gptq"]) for r in rep["per_projection"].values()]
    better = sum(1 for c, g_ in e if g_ <= c)
    return (f"{rep['passes']} passes == 1 pass byte for byte; {n4} NVFP4 projections overlay the "
            f"BF16 checkpoint; gptq <= clip on {better}/{len(e)} projections")


def test_mix_keeps_the_chosen_units():
    cfg = config("fp8")
    d, _ = bf16_checkpoint(cfg)
    tmp = tempfile.mkdtemp()
    out = _tiny_build(d, tmp, 10.0, methods="clip")
    files = ",".join(os.path.join(out, "clip", f"{t}.safetensors") for t in ("mlp", "gdn", "attn"))
    blocks = qs.load_blocks(files, "cpu")
    units = qs.units_of(blocks)
    rows = []
    for i, (u, members) in enumerate(sorted(units.items())):
        b8 = sum(qs.fp8_bytes(*blocks[m].shape) for m in members)
        b4 = sum(qn.nvfp4_bytes(*blocks[m].shape) for m in members)
        rows.append({"unit": u, "layer": int(u.split(".")[1]), "kind": u.split(".", 2)[2],
                     "kl": 0.001 * (i + 1) if "down" in u else 1e-6, "dnll": 0.0,
                     "bytes_fp8": b8, "bytes_nvfp4": b4})
    sens = os.path.join(tmp, "sens.json")
    json.dump({"units": rows}, open(sens, "w"))
    # budget = exactly the down projections' extra bytes: the knapsack must keep all four
    budget = sum(r["bytes_fp8"] - r["bytes_nvfp4"] for r in rows if "down" in r["unit"])
    mixf = os.path.join(tmp, "mix.safetensors")
    qs.cmd_mix(types.SimpleNamespace(sens=sens, nvfp4=files, keep_gb=budget / 1e9, out=mixf))
    with safe_open(mixf, "pt") as f:
        bases = {k.rsplit(".", 1)[0] for k in f.keys()}
        kept = f.metadata()["kept_fp8"].split(",")
    assert sorted(kept) == sorted(r["unit"] for r in rows if "down" in r["unit"]), kept
    assert not any(b.endswith("down_proj") for b in bases)
    assert len(bases) == len(blocks) - cfg.num_hidden_layers
    w = Weights(d, device="cpu", skip_mtp=True, nvfp4=mixf, fp8_head="")
    assert not isinstance(w.q["layers.0.mlp.down_proj"], NVFP4Block)
    assert isinstance(w.q["layers.0.mlp.gate_proj"], NVFP4Block)
    return f"{len(kept)} units kept in FP8 by the knapsack, {len(bases)} NVFP4 projections written"


def test_decode_step_bytes_with_e4m3_head():
    cfg = config("fp8")
    d, _ = bf16_checkpoint(cfg)
    w = Weights(d, device="cpu", skip_mtp=True, nvfp4="", fp8_head="build")
    b = w.decode_step_bytes(cfg.num_hidden_layers)
    head = w.t["lm_head.weight"]
    assert abs(b["lm_head_GB"] * 1e9 - head.nbytes) < 1 and b["total_GB"] > b["lm_head_GB"]
    return f"e4m3 head counted as {head.nbytes} bytes"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
