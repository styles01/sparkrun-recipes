"""The NemotronH checkpoint reader: a tiny BF16 checkpoint written on the CPU reads back to the
reference's canonical names (and loads to the same forward); Lightning 30B-A3B's real headers (6,513
tensors, 270 of them the MTP head's) audit clean; a missing tensor, a wrong shape, a truncated shard and
a wrong sha256 stop with the name."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.lfm2_moe.pack import TensorInfo  # noqa: E402
from engine.models.nemotron_h.config import config_from_dict  # noqa: E402
from engine.models.nemotron_h.load import load  # noqa: E402
from engine.models.nemotron_h.pack import PackIndex, canonical, expected  # noqa: E402
from engine.models.nemotron_h.ref import NemotronHRef  # noqa: E402
from tests.test_nemotron_h_ref import PROMPT, PUZZLE, TINY, random_weights  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "nemotron_h")


def _raises(fn, *needles):
    try:
        fn()
    except (ValueError, FileNotFoundError) as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"no error naming {needles}")


def _write(d: str, raw: dict, W: dict, shards: int = 2) -> None:
    from safetensors.torch import save_file
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(raw, f)
    keys = sorted(W)
    wm = {}
    for s in range(shards):
        part = {("backbone." + k if not k.startswith(("lm_head", "mtp.")) else k): W[k].to(
            torch.float32 if k.endswith("e_score_correction_bias") else torch.bfloat16).contiguous()
            for k in keys[s::shards]}
        fn = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        save_file(part, os.path.join(d, fn))
        wm.update({k: fn for k in part})
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)


def test_canonical_names():
    assert canonical("backbone.layers.3.mixer.in_proj.weight") == "layers.3.mixer.in_proj.weight"
    assert canonical("language_model.backbone.norm_f.weight") == "norm_f.weight"
    assert canonical("language_model.lm_head.weight") == "lm_head.weight"
    assert canonical("mtp.layers.0.eh_proj.weight") == "mtp.layers.0.eh_proj.weight"
    assert canonical("vision_model.radio_model.blocks.0.attn.qkv.weight") is None


def test_tiny_checkpoint_round_trip():
    for raw in (TINY, PUZZLE):
        cfg = config_from_dict(raw, "tiny")
        W = random_weights(cfg)
        with tempfile.TemporaryDirectory() as d:
            _write(d, raw, W)
            idx, W2, experts, timing = load(d, "cpu", dtype=torch.float32, log=None)
            assert not idx.has_mtp and len(idx.tensors) == len(W)
            a = NemotronHRef(cfg, {k: v.to(torch.bfloat16).float() if not k.endswith("e_score_correction_bias")
                                   else v for k, v in W.items()})
            b = NemotronHRef(idx.cfg, W2, experts=experts)
            la = a.prefill(PROMPT, a.new_state(32))
            lb = b.prefill(PROMPT, b.new_state(32))
            assert torch.equal(la, lb), (la - lb).abs().max()
            db = idx.decode_bytes(context=1000)
            assert db["parts"]["kv_read"] == 1000 * cfg.kv_bytes_per_token()


def test_omni_checkpoint_reads_its_text_trunk():
    """An Omni checkpoint: the trunk under `language_model.` with the config in `llm_config`, a vision
    tower beside it; the reader takes the trunk's tensors and skips the tower's by name."""
    from safetensors.torch import save_file
    cfg = config_from_dict(TINY, "tiny")
    W = random_weights(cfg)
    raw = {"architectures": ["NemotronH_Omni_Reasoning_V3"], "model_type": "NemotronH_Omni_Reasoning_V3",
           "eos_token_id": 11, "llm_config": dict(TINY, eos_token_id=11), "vision_config": {"patch_size": 16},
           "img_context_token_id": 18}
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(raw, f)
        part = {("language_model.backbone." + k if not k.startswith("lm_head") else "language_model." + k):
                v.to(torch.float32 if k.endswith("e_score_correction_bias") else torch.bfloat16).contiguous()
                for k, v in W.items()}
        part["vision_model.radio_model.blocks.0.attn.qkv.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
        part["mlp1.0.weight"] = torch.zeros(4, dtype=torch.bfloat16)
        save_file(part, os.path.join(d, "model.safetensors"))
        idx = PackIndex.open(d)
        idx.audit()
        assert idx.cfg.omni and idx.cfg.eos_token_id == 11 and idx.skipped == 2 and len(idx.tensors) == len(W)


def test_audit_stops_by_name():
    cfg = config_from_dict(TINY, "tiny")
    W = random_weights(cfg)
    with tempfile.TemporaryDirectory() as d:
        W2 = dict(W)
        del W2["layers.3.mixer.dt_bias"]
        _write(d, TINY, W2)
        _raises(lambda: PackIndex.open(d).audit(), "layers.3.mixer.dt_bias")
    with tempfile.TemporaryDirectory() as d:
        W2 = dict(W)
        W2["layers.2.mixer.k_proj.weight"] = torch.zeros(3, cfg.hidden_size)
        _write(d, TINY, W2)
        _raises(lambda: PackIndex.open(d).audit(), "layers.2.mixer.k_proj.weight")
    with tempfile.TemporaryDirectory() as d:
        _write(d, TINY, W)
        fn = os.path.join(d, "model-00002-of-00002.safetensors")
        with open(fn, "r+b") as f:
            f.truncate(os.path.getsize(fn) - 16)
        _raises(lambda: PackIndex.open(d), "model-00002-of-00002.safetensors", "truncated")
    with tempfile.TemporaryDirectory() as d:
        _write(d, TINY, W)
        idx = PackIndex.open(d)
        man = {}
        for fn in idx.files:
            man[fn] = {"sha256": hashlib.sha256(open(os.path.join(d, fn), "rb").read()).hexdigest()}
        assert set(idx.verify_sha256(man).values()) == {"ok"}
        man["model-00001-of-00002.safetensors"]["sha256"] = "0" * 64
        _raises(lambda: idx.verify_sha256(man), "model-00001-of-00002.safetensors")


def test_lightning_real_headers_audit():
    with open(os.path.join(FIX, "lightning-30b-a3b.json")) as f:
        cfg = config_from_dict(json.load(f), "lightning")
    with gzip.open(os.path.join(FIX, "lightning-30b-a3b-headers.json.gz"), "rt") as f:
        hdr = json.load(f)
    assert len(hdr) == 6513 and sum(k.startswith("mtp.") for k in hdr) == 270
    DS = {"BF16": 2, "F32": 4}
    tensors = {}
    for k, (dt, shape) in hdr.items():
        n = canonical(k)
        nb = DS[dt]
        for s in shape:
            nb *= s
        tensors[n] = TensorInfo(n, k, "", dt, tuple(shape), 0, nb)
    idx = PackIndex("lightning", cfg, tensors, {})
    idx.has_mtp = True
    idx.audit()
    assert {n for n, _ in expected(cfg)} == set(tensors)
    db = idx.decode_bytes()
    # 23 MoE layers x 6 experts x (1856 x 2688 x 2 x 2 bytes) of routed experts a token
    assert db["parts"]["experts"] == 23 * 6 * 1856 * 2688 * 4
    assert 6.0e9 < db["total"] < 7.5e9, db["total"]


def test_community_flat_pack_headers_audit():
    """ENG-377: a community EXL3 pack of Super VL (PR #63) with a flat config remapped from the Omni wrapper:
    `backbone.*` keys, no tower, `tie_word_embeddings` true beside a stored fp16 `lm_head.weight` (the original
    head is untied), experts at 3.5 bits, the dense projections at 5, the MTP head at 4 / 5 / 6. Its real headers
    (one expert a layer stored here, the other 511 alike) audit clean, the stored head is the one read."""
    with open(os.path.join(FIX, "super-vl-exl3-3.5-community.json")) as f:
        raw = json.load(f)
    cfg = config_from_dict(raw, "community")
    assert cfg.tie_word_embeddings and not cfg.omni and len(cfg.layer_types) == 88
    with gzip.open(os.path.join(FIX, "super-vl-exl3-3.5-community-headers.json.gz"), "rt") as f:
        hdr = json.load(f)
    DS = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4}
    tensors = {}
    for k, (dt, shape) in hdr.items():
        keys = [k] if ".experts.0." not in k else [k.replace(".experts.0.", f".experts.{e}.") for e in range(512)]
        for key in keys:
            n = canonical(key)
            assert n is not None and not n.startswith("backbone."), key
            nb = DS[dt]
            for s in shape:
                nb *= s
            tensors[n] = TensorInfo(n, key, "", dt, tuple(shape), 0, nb)
    assert len(tensors) == 169478 and tensors["lm_head.weight"].dtype == "F16"
    idx = PackIndex("community", cfg, tensors, {}, quant=raw["quantization_config"])
    idx.has_mtp = True
    assert idx.stored_head and not idx.cfg.tie_word_embeddings
    notes = idx.audit()
    assert any("stored head is read" in n for n in notes), notes
    assert idx.is_exl3 and idx.formats["lm_head"] == "f16" and idx.formats["layers.1.mixer.gate"] == "f16"
    rates = {}
    for base, f in idx.formats.items():
        if f == "exl3":
            kind = "mtp-" if base.startswith("mtp.") else ""
            kind += "expert" if ".experts." in base else base.rsplit(".", 1)[-1]
            rates.setdefault(kind, set()).add(tensors[base + ".trellis"].k)
    assert rates["expert"] == {3.5} and rates["mtp-expert"] == {4.0}, rates
    assert all(v == {5.0} for k, v in rates.items() if not k.startswith("mtp-") and k != "expert"), rates
    assert rates["mtp-eh_proj"] == {5.0} and rates["mtp-q_proj"] == {6.0}, rates
    assert {n for n, _ in expected(idx.cfg)} | {n for n in tensors if n.rsplit(".", 1)[-1] in
                                                 ("trellis", "suh", "svh", "mul1")} >= set(tensors)
    db = idx.decode_bytes()
    assert db["parts"]["head"] == 131072 * 4096 * 2          # the stored fp16 head, not the embedding
    # without the stored head the tie stands and the embedding is the head
    no_head = {n: t for n, t in tensors.items() if n != "lm_head.weight"}
    idx2 = PackIndex("community", config_from_dict(raw, "community"), no_head, {}, quant=raw["quantization_config"])
    idx2.has_mtp = True
    idx2.audit()
    assert not idx2.stored_head and idx2.cfg.tie_word_embeddings and "lm_head" not in idx2.formats


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all nemotron_h pack tests passed")
