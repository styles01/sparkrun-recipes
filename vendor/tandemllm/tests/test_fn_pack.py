"""ENG-219/220: the pack reader on a synthetic mini-pack (no download, no GPU).

The mini-pack has the real pack's layout at the tiny config's sizes (tests/fn_tiny.py): seven-ish
shards with an index, routed experts and dense projections as EXL3 (trellis/suh/svh/mul1, experts
at K 3 with one expert at K 4, dense at K 5), mixers fp16, the n-gram table in exllamav3's
`exl3_ngram_trellis` format in its own file, and the MTP final mixer in the patch file. Checked:
every tensor is accounted for with its kind and K, the byte totals, the decode byte budget's parts,
the n-gram rows against exllamav3's encoder, the int8 mixers' error bound and the fp16 flag, the
shared-table identity, a vision tower in a file outside the index (4.05's), and that a bad hash, a
missing tensor, a truncated shard and a missing patch each fail naming the cause.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fn_tiny import TINY  # noqa: E402
from engine.models.flashnext import pack as P  # noqa: E402
from engine.models.flashnext.config import config_from_dict  # noqa: E402
from engine.models.flashnext.ple import (ROW_DIM, EXL3NGramTable, decode_rows, head_sizes,  # noqa: E402
                                         mul1_codebook, words_per_row)

# trellis tiles are 16 x 16: the indexer projection ((heads + 1) x head dim) must be a multiple of 16
MINI = dict(TINY, indexer_head_dim=16, ple_embed_dim=16 * 160)   # (exllamav3 rows are 160 wide)
CFG = config_from_dict(dict(MINI))
G = torch.Generator().manual_seed(0)


# --------------------------------------------------------------------- exllamav3's n-gram encoder
def pack_rows(states: torch.Tensor, scales: torch.Tensor, K: int) -> torch.Tensor:
    """exllamav3 `ngram_codec.pack_rows` (vcruz305 fork 329e051), the encoder of the format."""
    N = states.shape[0]
    new_bits = states.to(torch.int64) & ((1 << K) - 1)
    bits = (new_bits.unsqueeze(-1) >> torch.arange(K)) & 1
    bits = bits.reshape(N, ROW_DIM * K // 16, 16)
    words = (bits << torch.arange(16)).sum(dim=-1)
    words = (words & 0xFFFF).to(torch.int32).to(torch.int16)
    scale_words = scales.to(torch.float16).view(torch.int16).unsqueeze(1)
    return torch.cat((scale_words, words), dim=1).contiguous()


def ring_states(N: int, K: int) -> torch.Tensor:
    """Valid tail-biting ring states: each state's low K bits are fresh, its high bits the
    previous states' (state_i = bits of positions i-15/K .. i)."""
    fresh = torch.randint(0, 1 << K, (N, ROW_DIM), generator=G)
    states = torch.zeros(N, ROW_DIM, dtype=torch.int64)
    for i in range(ROW_DIM):
        s = torch.zeros(N, dtype=torch.int64)
        for m in range(16):
            j, b = (i - m // K) % ROW_DIM, m % K
            s |= ((fresh[:, j] >> b) & 1) << m
        states[:, i] = s
    return states


# --------------------------------------------------------------------- the mini-pack
def trellis(Kin: int, N: int, k: int) -> dict:
    return {"trellis": torch.randint(-32768, 32767, (Kin // 16, N // 16, 16 * k), generator=G,
                                     dtype=torch.int32).to(torch.int16),
            "suh": torch.randn(Kin, generator=G).half(), "svh": torch.randn(N, generator=G).half(),
            "mul1": torch.tensor(P.MUL1 - (1 << 32), dtype=torch.int32)}


def build(d: str, patch: bool = True, ngram_k: int = 5) -> dict:
    c = CFG
    H, I, E = c.hidden_size, c.moe_intermediate_size, c.num_experts
    t: dict[str, torch.Tensor] = {}
    pre = P.PREFIX

    def ex(name, Kin, N, k):
        for s, v in trellis(Kin, N, k).items():
            t[f"{name}.{s}"] = v

    def plain(name, *shape, dt=torch.bfloat16):
        t[name] = torch.randn(*shape, generator=G).to(dt)

    def mixers(p, inject=True):
        plain(p + ".hc_norm.weight", c.hc_hidden, dt=torch.float16)
        plain(p + ".input_mix_weight_down.weight", c.hc_lowrank, c.hc_hidden, dt=torch.float16)
        plain(p + ".input_mix_weight_up.weight", c.hc_hidden, c.hc_lowrank, dt=torch.float16)
        if inject:
            plain(p + ".block_inject_weight.weight", c.hc_count, c.hc_hidden, dt=torch.float16)

    def block(p, linear):
        mixers(p + "attn_hyper_connection")
        mixers(p + "mlp_hyper_connection")
        if linear:
            a = p + "linear_attn."
            ex(a + "in_proj_qkv", H, c.conv_dim, 5)
            ex(a + "in_proj_z", H, c.value_dim, 5)
            ex(a + "out_proj", c.value_dim, H, 5)
            for n, shp in (("A_log", (c.linear_num_value_heads,)), ("dt_bias", (c.linear_num_value_heads,)),
                           ("conv1d.weight", (c.conv_dim, 1, 4)), ("norm.weight", (c.linear_value_head_dim,))):
                plain(a + n, *shp)
            plain(a + "in_proj_a.weight", c.linear_num_value_heads, H, dt=torch.float16)
            plain(a + "in_proj_b.weight", c.linear_num_value_heads, H, dt=torch.float16)
        else:
            a = p + "self_attn."
            D = c.head_dim
            ex(a + "q_proj", H, c.num_attention_heads * 2 * D, 5)
            ex(a + "k_proj", H, c.num_key_value_heads * D, 5)
            ex(a + "v_proj", H, c.num_key_value_heads * D, 5)
            ex(a + "o_proj", c.num_attention_heads * D, H, 5)
            ex(a + "indexer.index_qk_proj", H, (c.indexer_n_heads + 1) * c.indexer_head_dim, 3)
            for n, dim in (("q_norm", D), ("k_norm", D), ("indexer.q_layernorm", c.indexer_head_dim),
                           ("indexer.k_layernorm", c.indexer_head_dim)):
                plain(a + n + ".weight", dim)
        m = p + "mlp."
        plain(m + "gate.weight", E, H, dt=torch.float16)
        plain(m + "shared_expert_gate.weight", 1, H, dt=torch.float16)
        ex(m + "shared_expert.gate_proj", H, c.shared_expert_intermediate_size, 5)
        ex(m + "shared_expert.up_proj", H, c.shared_expert_intermediate_size, 5)
        ex(m + "shared_expert.down_proj", c.shared_expert_intermediate_size, H, 5)
        for e in range(E):
            k = 4 if e == 5 else 3                       # one expert at another K (a mixed-K layer)
            ex(m + f"experts.{e}.gate_proj", H, I, k)
            ex(m + f"experts.{e}.up_proj", H, I, k)
            ex(m + f"experts.{e}.down_proj", I, H, k)

    plain(pre + "embed_tokens.weight", c.vocab_size, H)
    ex("lm_head", H, c.vocab_size if c.vocab_size % 16 == 0 else 64, 6)
    for li in range(c.num_hidden_layers):
        block(pre + f"layers.{li}.", c.is_linear(li))
    p = pre + "layers.1.ple."
    plain(p + "key_proj.weight", c.hc_hidden, c.ple_embed_dim, dt=torch.float16)
    plain(p + "value_proj.weight", H, c.ple_embed_dim, dt=torch.float16)
    for n in ("norm_key", "norm_query", "norm_conv"):
        plain(p + n + ".weight", c.hc_hidden)
    plain(p + "conv1d.weight", c.hc_hidden, 1, 4, dt=torch.float16)
    mixers(pre + "hyper_connection_mixer", inject=False)
    block("mtp.layers.0.", False)
    ex("mtp.fc_embedding", H, H, 3)
    ex("mtp.fc_hidden", H, H, 3)
    plain("mtp.pre_fc_norm_embedding.weight", H)
    plain("mtp.pre_fc_norm_hidden.weight", c.hc_hidden)
    t["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(16, 16, dtype=torch.bfloat16)
    # the MTP layer's final mixer lives in the patch file (or nowhere: the 4.05 question)
    mixers("mtp.hyper_connection_mixer", inject=False)
    patch_t = {k: t.pop(k) for k in list(t) if k.startswith("mtp.hyper_connection_mixer.")}
    # shards of ~equal tensor count, an index, the patch
    keys = sorted(t)
    n_sh = 3
    wm = {}
    for i in range(n_sh):
        part = {k: t[k] for k in keys[i::n_sh]}
        fn = f"model-{i + 1:05d}-of-{n_sh:05d}.safetensors"
        save_file(part, os.path.join(d, fn))
        wm.update({k: fn for k in part})
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)
    if patch:
        save_file(patch_t, os.path.join(d, P.PATCH_FILE))
    # the n-gram table: exllamav3's format, two shards back to back
    rows = head_sizes(c.ngram_vocab_size_base, c.ngram_heads, 0)[2]
    states = ring_states(rows, ngram_k)
    scales = (torch.rand(rows, generator=G) + 0.5).half()
    packed = pack_rows(states, scales, ngram_k)
    sizes, offs, _ = head_sizes(c.ngram_vocab_size_base, c.ngram_heads, 0)
    from engine.models.flashnext.ple import layer_multipliers
    tp = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    half = rows // 2
    tab = {tp + "shard_0.trellis": packed[:half].clone(), tp + "shard_1.trellis": packed[half:].clone(),
           tp + "head_bias": torch.randn(c.ngram_heads, ROW_DIM, generator=G).half(),
           tp + "head_offsets": torch.tensor(offs), tp + "head_vocab_sizes": torch.tensor(sizes),
           tp + "layer_multipliers": torch.tensor(layer_multipliers(c.vocab_size, c.ngram_size, 0, c.seed))}
    save_file(tab, os.path.join(d, P.TABLE_FILE))
    raw = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
           "text_config": dict(MINI),
           "quantization_config": {"quant_method": "exl3", "bits": 3.05, "head_bits": 6}}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(raw, f)
    return {"states": states, "scales": scales, "packed": packed, "bias": tab[tp + "head_bias"],
            "offsets": offs}


def mini(**kw):
    d = tempfile.mkdtemp(prefix="fnpack-")
    info = build(d, **kw)
    return d, info


def manifest(d: str) -> dict:
    out = {}
    for fn in os.listdir(d):
        with open(os.path.join(d, fn), "rb") as f:
            out[fn] = {"sha256": hashlib.sha256(f.read()).hexdigest(), "size": os.path.getsize(os.path.join(d, fn))}
    return out


def _raises(fn, *needles):
    try:
        fn()
    except (ValueError, FileNotFoundError) as e:
        assert all(n in str(e) for n in needles), str(e)
        return str(e)
    raise AssertionError("no error")


# --------------------------------------------------------------------- tests
def test_every_tensor_accounted_for():
    d, _ = mini()
    try:
        idx = P.PackIndex.open(d)
        idx.audit()
        idx.check_mul1()
        kinds = {}
        for t in idx.tensors.values():
            kinds[t.kind] = kinds.get(t.kind, 0) + 1
        assert set(kinds) == {"expert", "exl3", "mixer", "plain"}, kinds
        assert len(idx.skipped) == 1 and idx.skipped[0].startswith("model.visual.")
        n_exp = 5 * CFG.num_experts * 3 * 4                      # 4 trunk + MTP, 3 proj, 4 suffixes
        assert kinds["expert"] == n_exp, kinds
        shard_bytes = sum(v for k, v in idx.files.items() if k != P.TABLE_FILE)
        headers = sum(P.safetensors_index(os.path.join(d, k))[0] for k in idx.files if k != P.TABLE_FILE)
        tensor_bytes = sum(t.nbytes for t in idx.tensors.values()) + 16 * 16 * 2      # + the vision one
        assert shard_bytes == headers + tensor_bytes, (shard_bytes, headers, tensor_bytes)
        assert "mtp.hyper_connection_mixer.input_mix_weight_up.weight" in idx.tensors
        assert idx.tensors["mtp.hyper_connection_mixer.input_mix_weight_up.weight"].file.endswith(P.PATCH_FILE)
        return f"{len(idx.tensors)} tensors: {kinds}; shards = headers + tensors, to the byte"
    finally:
        shutil.rmtree(d)


def test_a_vision_file_outside_the_index():
    d, _ = mini()
    try:
        tower = {"model.visual.blocks.0.attn.q_proj.trellis": torch.zeros(4, 4, 48, dtype=torch.int16),
                 "model.visual.pos_embed.weight": torch.ones(16, 8, dtype=torch.bfloat16)}
        save_file(tower, os.path.join(d, "vision_k6.safetensors"))
        idx = P.PackIndex.open(d)
        t = idx.vision["visual.pos_embed.weight"]
        assert t.file.endswith("vision_k6.safetensors") and "visual.blocks.0.attn.q_proj.trellis" in idx.vision
        assert torch.equal(idx.read("visual.pos_embed.weight"), tower["model.visual.pos_embed.weight"])
        assert not any(k.startswith("visual") for k in idx.tensors)
        return "4.05's vision_k6.safetensors: its tensors under their visual. names, read back as stored"
    finally:
        shutil.rmtree(d)

def test_k_histogram_and_budget():
    d, _ = mini()
    try:
        idx = P.PackIndex.open(d)
        h = idx.k_histogram()
        assert h["experts.gate_proj"] == {3: 4 * 8 - 4, 4: 4}, h["experts.gate_proj"]
        assert h["head"] == {6: 1} and h["mtp.experts"] == {3: 21, 4: 3}, h
        b1, b2 = idx.decode_bytes(1), idx.decode_bytes(2)
        mix_fp16 = sum(t.nbytes for n, t in idx.tensors.items() if t.kind == "mixer" and not n.startswith("mtp."))
        assert b2["parts"]["mixers"] == mix_fp16
        rows = sum(t.shape[0] for n, t in idx.tensors.items() if t.kind == "mixer" and not n.startswith("mtp."))
        assert b1["parts"]["mixers"] == mix_fp16 // 2 + 4 * rows
        eb = idx.expert_bytes(0)
        assert eb[5] > eb[0]
        want = sum(CFG.num_experts_per_tok * sum(idx.expert_bytes(l)) / CFG.num_experts for l in range(4))
        assert abs(b1["parts"]["experts"] - want) <= 4, (b1["parts"]["experts"], want)
        assert b1["parts"]["ple_rows"] == CFG.ngram_heads * words_per_row(5) * 2
        assert b1["parts"]["head"] == idx._exl3_bytes("lm_head.")
        assert b1["total"] == sum(b1["parts"].values())
        s = P.summary(idx)
        json.dumps(s)
        return f"decode {b1['total']} B int8 mixers, {b2['total']} B fp16; parts {sorted(b1['parts'])}"
    finally:
        shutil.rmtree(d)


def test_ngram_rows_equal_the_encoder():
    d, info = mini()
    try:
        idx = P.PackIndex.open(d)
        tab = idx.table
        assert (tab.K, tab.words, tab.rows) == (5, 51, info["packed"].shape[0])
        ids = torch.tensor([[0, 7, tab.rows - 1, 12345 % tab.rows], [3, 3, 101, 103]])
        got = tab.gather(ids, dtype=torch.float32)
        cb = mul1_codebook()
        flat = ids.reshape(-1)
        heads = torch.searchsorted(torch.tensor(info["offsets"]), flat, right=True) - 1
        want = cb[info["states"][flat]].float() * info["scales"][flat].float()[:, None] + \
            info["bias"][heads].float()
        assert torch.equal(got.reshape(-1, ROW_DIM), want)
        # the parallel pread path gives the same rows as the page-fault path
        many = torch.randint(0, tab.rows, (600,), generator=G)
        assert torch.equal(tab.gather_packed(many), info["packed"][many])
        assert torch.equal(decode_rows(info["packed"][:4], heads[:4] * 0, info["bias"], 5, cb),
                           cb[info["states"][:4]].float() * info["scales"][:4].float()[:, None]
                           + info["bias"][0].float())
        return f"K 5, {tab.words} words a row, {tab.rows} rows; values = mul1(state) * scale + bias"
    finally:
        shutil.rmtree(d)


def test_hash_parameters_from_the_table():
    d, _ = mini()
    try:
        from engine.models.flashnext.ple import NGramHash
        idx = P.PackIndex.open(d)
        t = idx.table
        h = NGramHash.from_config(CFG, 0, {"layer_multipliers": t.multipliers,
                                           "ngram_heads_vocab_sizes": t.sizes, "ngram_heads_offsets": t.offsets})
        bad = t.multipliers.clone()
        bad[0] += 2
        _raises(lambda: NGramHash.from_config(CFG, 0, {"layer_multipliers": bad}), "layer_multipliers")
        return f"{h.mult.numel()} multipliers, {h.sizes.numel()} heads agree with the config"
    finally:
        shutil.rmtree(d)


def test_int8_mixers_bound_and_fp16_flag():
    w = (torch.randn(320, 1024, generator=G) * 0.05).half()
    w[3] = 0
    w[7, 11] = 3.0                                            # an outlier row
    q = P.mixer_weight(w, "int8rows")
    deq = q.dequant()
    err = (deq - w.float()).abs()
    bound = w.float().abs().amax(1, keepdim=True) / 254 + w.float().abs() * 2 ** -22
    assert (err <= bound).all()
    assert deq[3].abs().max() == 0 and q.nbytes == 320 * 1024 + 320 * 4 and q.group is None
    rel = (err.max(1).values / w.float().abs().amax(1).clamp_min(1e-30)).max().item()
    assert P.mixer_weight(w, "fp16") is w
    x = torch.randn(2, 1024, generator=G).bfloat16()
    assert torch.equal(q.matmul(x), (x.float() @ deq.T).bfloat16())
    _raises(lambda: P.mixer_weight(w, "int4"), "int8, int8rows or fp16")
    # ENG-221, the default: a scale a group of 128 weights; the outlier row's other weights keep their
    # resolution (their groups' scales, not the outlier's)
    g = P.mixer_weight(w, "int8")
    dg = g.dequant()
    assert g.group == 128 and g.s.shape == (320, 8) and g.nbytes == 320 * 1024 + 320 * 8 * 4
    eg = (dg - w.float()).abs().view(320, 8, 128)
    gbound = w.float().abs().view(320, 8, 128).amax(2, keepdim=True) / 254 + w.float().abs().view(320, 8, 128) * 2 ** -22
    assert (eg <= gbound).all() and dg[3].abs().max() == 0
    row7 = (dg[7, 128:] - w.float()[7, 128:]).abs().max().item()
    assert row7 < (deq[7, 128:] - w.float()[7, 128:]).abs().max().item() / 10
    assert torch.equal(g.matmul(x), (x.float() @ dg.T).bfloat16())
    assert P.mixer_weight(w[:, :320].contiguous(), "int8").group == 64        # the up weights' K: gcd(128, 320)
    return (f"rows: |w - q s| <= max|row| / 254 (worst relative {rel:.4f}); groups of 128: bound a group, the "
            f"outlier row's other weights 10x finer; fp16 flag returns the pack's tensor")


def test_bad_hash_names_the_file():
    d, _ = mini()
    try:
        m = manifest(d)
        idx = P.PackIndex.open(d)
        assert len(idx.verify(m)) == len(idx.files)
        fn = "model-00002-of-00003.safetensors"
        m[fn]["sha256"] = "0" * 64
        return _raises(lambda: idx.verify(m), fn)
    finally:
        shutil.rmtree(d)


def test_missing_tensor_names_it():
    d, _ = mini()
    try:
        ip = os.path.join(d, "model.safetensors.index.json")
        doc = json.load(open(ip))
        doc["weight_map"]["model.language_model.layers.2.mlp.experts.4.down_proj.trellis"] = "model-09.safetensors"
        json.dump(doc, open(ip, "w"))
        _raises(lambda: P.PackIndex.open(d), "model-09.safetensors")
        # a tensor no file has at all: drop it from the index and its shard
        doc["weight_map"].pop("model.language_model.layers.2.mlp.experts.4.down_proj.trellis")
        json.dump(doc, open(ip, "w"))
        idx = P.PackIndex.open(d)
        return _raises(idx.audit, "layers.2.mlp.experts.4.down_proj.trellis")
    finally:
        shutil.rmtree(d)


def test_truncated_shard_is_refused():
    d, _ = mini()
    try:
        fp = os.path.join(d, "model-00003-of-00003.safetensors")
        os.truncate(fp, os.path.getsize(fp) - 10)
        return _raises(lambda: P.PackIndex.open(d), "model-00003-of-00003.safetensors", "truncated")
    finally:
        shutil.rmtree(d)


def test_missing_patch_is_reported():
    d, _ = mini(patch=False)
    try:
        idx = P.PackIndex.open(d)
        assert "mtp.hyper_connection_mixer.hc_norm.weight" not in idx.tensors
        msg = _raises(idx.audit, P.PATCH_FILE, "mtp.hyper_connection_mixer")
        idx.audit(mtp=False)                                    # the trunk alone is complete
        return msg
    finally:
        shutil.rmtree(d)


def test_shared_table_identity():
    d1, _ = mini()
    d2 = tempfile.mkdtemp(prefix="fnpack-")
    try:
        shutil.copy(os.path.join(d1, P.TABLE_FILE), os.path.join(d2, P.TABLE_FILE))
        a = P.table_identity(os.path.join(d1, P.TABLE_FILE))
        b = P.table_identity(os.path.join(d2, P.TABLE_FILE))
        m = manifest(d1)
        c = P.table_identity(os.path.join(d1, P.TABLE_FILE), m)
        assert a == b and c == m[P.TABLE_FILE]["sha256"]
        return f"same file -> same identity ({a[:22]}..); the Hub sha256 when known"
    finally:
        shutil.rmtree(d1)
        shutil.rmtree(d2)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<36} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<36} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
