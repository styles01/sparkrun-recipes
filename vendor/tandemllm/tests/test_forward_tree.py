"""The tree verify, checked against the chain verify it has to agree with.

Two claims are worth a test and neither needs a GPU or the 27 B checkpoint:

  1. a tree that happens to be a chain must compute exactly what `forward_block` computes -- logits
     and, after committing the whole thing, state;
  2. on a branching tree, the path the target accepts must be worth exactly what verifying that
     path alone as a chain would have been worth. That is what "lossless" means for a tree: the
     branches that were not taken must not have leaked into the answer through the attention mask,
     the convolution window, the UT factorisation or the committed state.

The model here is a random 4-layer one with the real layer structure -- three gated-delta-net
layers and one full-attention layer, partial rotary, the depthwise convolution, the gated output
norm -- at a hidden size a laptop can run in fp32. The arithmetic the tree changes is all in that
structure; none of it is in the size.
"""

from __future__ import annotations

import os
import sys

# The fused kernels are Triton and this file runs on a CPU. They are a different arithmetic order
# for the same quantities and each has its own check against the function it replaces; what is
# being tested here is the tree, against the reference forward.
for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
# and the chain delegation off, or `test_chain_tree_matches_block_verify` would be comparing
# `forward_block` with itself
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import TextConfig  # noqa: E402
from engine.models.qwen3_5.adapter import Qwen38Engine  # noqa: E402  (the served class: its cold verbs)
from engine.tree import DraftTree  # noqa: E402

DEV = "cpu"
DT = torch.float32


def tiny_config() -> TextConfig:
    return TextConfig(
        path="<random>", hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=97,
        rms_norm_eps=1e-6, rope_theta=10000.0, partial_rotary_factor=0.5,
        mrope_section=[2, 1, 1], max_position_embeddings=512,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, mtp_num_hidden_layers=0,
        weight_block_size=(128, 128), eos_token_ids=[], bos_token_id=0)


class FakeWeights:
    """Plain tensors under the checkpoint's own names; `linear()` falls through to `F.linear`."""

    def __init__(self, cfg: TextConfig, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.t: dict[str, torch.Tensor] = {}

        def r(*shape, scale=0.2):
            return torch.randn(*shape, generator=g, dtype=DT, device=DEV) * scale

        c = cfg
        self.t["embed_tokens.weight"] = r(c.vocab_size, c.hidden_size, scale=1.0)
        self.t["norm.weight"] = r(c.hidden_size, scale=0.1)
        self.t["lm_head.weight"] = r(c.vocab_size, c.hidden_size, scale=0.5)
        for l in range(c.num_hidden_layers):
            p = f"layers.{l}"
            self.t[f"{p}.input_layernorm.weight"] = r(c.hidden_size, scale=0.1)
            self.t[f"{p}.post_attention_layernorm.weight"] = r(c.hidden_size, scale=0.1)
            self.t[f"{p}.mlp.gate_proj"] = r(c.intermediate_size, c.hidden_size)
            self.t[f"{p}.mlp.up_proj"] = r(c.intermediate_size, c.hidden_size)
            self.t[f"{p}.mlp.down_proj"] = r(c.hidden_size, c.intermediate_size)
            if c.is_linear(l):
                self.t[f"{p}.linear_attn.in_proj_qkv"] = r(c.conv_dim, c.hidden_size)
                self.t[f"{p}.linear_attn.conv1d.weight"] = r(c.conv_dim, 1,
                                                             c.linear_conv_kernel_dim, scale=0.5)
                self.t[f"{p}.linear_attn.in_proj_z"] = r(c.value_dim, c.hidden_size)
                self.t[f"{p}.linear_attn.in_proj_b.weight"] = r(c.linear_num_value_heads,
                                                                c.hidden_size)
                self.t[f"{p}.linear_attn.in_proj_a.weight"] = r(c.linear_num_value_heads,
                                                                c.hidden_size)
                self.t[f"{p}.linear_attn.A_log"] = r(c.linear_num_value_heads, scale=0.5)
                self.t[f"{p}.linear_attn.dt_bias"] = r(c.linear_num_value_heads, scale=0.5)
                self.t[f"{p}.linear_attn.norm.weight"] = r(c.linear_value_head_dim, scale=0.1)
                self.t[f"{p}.linear_attn.out_proj"] = r(c.hidden_size, c.value_dim)
            else:
                self.t[f"{p}.self_attn.q_proj"] = r(c.q_dim, c.hidden_size)
                self.t[f"{p}.self_attn.k_proj"] = r(c.kv_dim, c.hidden_size)
                self.t[f"{p}.self_attn.v_proj"] = r(c.kv_dim, c.hidden_size)
                self.t[f"{p}.self_attn.o_proj"] = r(c.hidden_size, c.attn_out_dim)
                self.t[f"{p}.self_attn.q_norm.weight"] = r(c.head_dim, scale=0.1)
                self.t[f"{p}.self_attn.k_norm.weight"] = r(c.head_dim, scale=0.1)

    def norm(self, name): return self.t[name]
    def proj(self, name): return self.t[name]

    def group(self, name):
        """No fused projection groups here: these weights are plain bf16 tensors and the fused
        path is an NVFP4 layout. `Weights.group` returning None is the supported way to say so,
        and it is how a layer the quality gate left in fp8 keeps its separate launches."""
        return None


def build(seed: int = 0, prompt_len: int = 12):
    cfg = tiny_config()
    eng = Qwen38Engine(cfg, FakeWeights(cfg, seed), max_len=256, device=DEV)
    eng.kv.k = eng.kv.k.to(DT)
    eng.kv.v = eng.kv.v.to(DT)
    eng.state.conv = eng.state.conv.to(DT)
    torch.manual_seed(seed)
    ids = torch.randint(1, cfg.vocab_size, (prompt_len,))
    with torch.no_grad():
        eng.forward(ids, start=0, last_only=True)
    return eng, prompt_len


def snapshot(eng):
    return (eng.state.S.clone(), eng.state.conv.clone(), eng.kv.k.clone(), eng.kv.v.clone(),
            eng.kv.length)


def restore(eng, snap):
    eng.state.S.copy_(snap[0]); eng.state.conv.copy_(snap[1])
    eng.kv.k.copy_(snap[2]); eng.kv.v.copy_(snap[3]); eng.kv.length = snap[4]


def diff(a, b):
    return (a - b).abs().max().item()


# ------------------------------------------------------------------ 1. a chain tree is a chain

def test_chain_tree_matches_block_verify():
    eng, pos = build(seed=1)
    toks = [7, 11, 23, 5, 31]
    snap = snapshot(eng)
    with torch.no_grad():
        lg_chain = eng.forward_block(torch.tensor(toks), start=pos)
    # the state the chunked kernel wrote itself, before any reconstruction touches it -- so this
    # test checks the SpecLA gather against the kernel, not only against the prefix form of itself
    natural = eng.state.S.clone()
    with torch.no_grad():
        eng.rollback_to(len(toks))
    after_chain = snapshot(eng)

    restore(eng, snap)
    tree = DraftTree.chain(toks[0], toks[1:])
    with torch.no_grad():
        lg_tree = eng.forward_tree(torch.tensor(tree.tokens), tree.parents, start=pos)
        eng.commit_tree(list(range(len(toks))))
    after_tree = snapshot(eng)

    d_lg = diff(lg_chain, lg_tree)
    d_S = diff(after_chain[0], after_tree[0])
    d_conv = diff(after_chain[1], after_tree[1])
    d_kv = diff(after_chain[2][..., :pos + len(toks), :], after_tree[2][..., :pos + len(toks), :])
    d_nat = diff(natural, after_tree[0])
    assert d_lg < 1e-4, f"logits differ by {d_lg}"
    assert d_S < 1e-4, f"recurrent state differs by {d_S}"
    assert d_nat < 1e-4, f"tree commit differs from the kernel's own final state by {d_nat}"
    assert d_conv < 1e-5, f"conv state differs by {d_conv}"
    assert d_kv < 1e-5, f"kv differs by {d_kv}"
    assert after_tree[4] == after_chain[4]
    return f"logits {d_lg:.2e}  S {d_S:.2e} (vs kernel {d_nat:.2e})  conv {d_conv:.2e}  kv {d_kv:.2e}"


# ------------------------------------------------- 2. a branch is worth what that branch alone is

BRANCHY = DraftTree(
    #        0    1   2   3    4   5   6    7   8
    tokens=[41,  13, 62, 29,  55, 17, 88,  3, 71],
    parents=[-1,  0,  1,  2,   1,  4,  0,   6, 6])


def test_branching_tree_path_equals_chain_verify():
    """Every root-to-leaf path of a branching tree, verified alone as a chain, must give the same
    logits at its nodes and the same committed state as taking that path out of the tree."""
    tree = BRANCHY
    tree.check()
    out = []
    for leaf in tree.leaves():
        path = tree.path(leaf)
        chain_tokens = [tree.tokens[i] for i in path]

        eng, pos = build(seed=2)
        snap = snapshot(eng)
        with torch.no_grad():
            lg_chain = eng.forward_block(torch.tensor(chain_tokens), start=pos)
            eng.rollback_to(len(chain_tokens))
        after_chain = snapshot(eng)

        restore(eng, snap)
        with torch.no_grad():
            lg_tree = eng.forward_tree(torch.tensor(tree.tokens), tree.parents, start=pos)
            eng.commit_tree(path)
        after_tree = snapshot(eng)

        d_lg = diff(lg_chain, lg_tree[path])
        d_S = diff(after_chain[0], after_tree[0])
        d_conv = diff(after_chain[1], after_tree[1])
        n = len(path)
        d_kv = diff(after_chain[2][..., :pos + n, :], after_tree[2][..., :pos + n, :])
        assert d_lg < 1e-4, f"leaf {leaf}: logits differ by {d_lg}"
        assert d_S < 1e-4, f"leaf {leaf}: state differs by {d_S}"
        assert d_conv < 1e-5, f"leaf {leaf}: conv differs by {d_conv}"
        assert d_kv < 1e-5, f"leaf {leaf}: kv differs by {d_kv}"
        assert after_tree[4] == pos + n
        out.append(f"leaf {leaf} (depth {n - 1}): logits {d_lg:.1e} S {d_S:.1e}")
    return "; ".join(out)


def test_siblings_do_not_see_each_other():
    """The negative control: change a token in one branch and nothing on another branch may move.

    Every mechanism the tree touches -- the attention mask, the convolution's gather, the ancestor
    masking of the UT factorisation -- fails this test if it leaks, and a numeric agreement test
    alone would not catch a leak that happens to be small.
    """
    eng, pos = build(seed=3)
    snap = snapshot(eng)
    t = BRANCHY
    with torch.no_grad():
        a = eng.forward_tree(torch.tensor(t.tokens), t.parents, start=pos)
    restore(eng, snap)
    toks = list(t.tokens)
    toks[7] = (toks[7] + 37) % 97            # a node under 6, which is under the root only
    with torch.no_grad():
        b = eng.forward_tree(torch.tensor(toks), t.parents, start=pos)
    untouched = [i for i in range(len(toks)) if i != 7 and not t._is_descendant(i, 7)]
    d = max(diff(a[i], b[i]) for i in untouched)
    moved = diff(a[7], b[7])
    assert d < 1e-6, f"changing node 7 moved unrelated rows by {d}"
    assert moved > 1e-3, f"changing node 7 did not move its own row ({moved})"
    return f"unrelated rows moved {d:.1e}, node 7's own row moved {moved:.3f}"


def test_accept_tree_walk():
    """The accept walk follows the target's argmax and stops where the tree has no matching child."""
    eng, _ = build(seed=4)
    t = BRANCHY
    picks = [0] * len(t.tokens)
    picks[0] = t.tokens[1]      # take node 1
    picks[1] = t.tokens[4]      # then node 4
    picks[4] = 999              # no child carries this
    path, new = eng.accept_tree(t, picks)
    assert path == [0, 1, 4], path
    assert new == [t.tokens[1], t.tokens[4], 999], new
    picks[0] = 999
    path, new = eng.accept_tree(t, picks)
    assert path == [0] and new == [999]
    return "walk stops where the tree runs out, and always yields one token"


def test_conv_windows_on_a_chain_are_a_sliding_window():
    from engine.tree import DraftTree as DT2
    chain = DT2.chain(1, [2, 3, 4, 5])
    w = chain.conv_windows(4)
    assert w[0] == [0, 1, 2, 3], w[0]          # conv state cols 0..2, then node 0
    assert w[1] == [1, 2, 3, 4], w[1]
    assert w[3] == [3, 4, 5, 6], w[3]
    br = BRANCHY.conv_windows(4)
    assert br[4] == [2, 3, 4, 7], br[4]        # node 4's ancestors are 1 and 0, then the state
    assert br[7] == [2, 3, 9, 10], br[7]       # node 7 under 6 under 0: cols for 0, 6, 7
    return "chain windows slide, tree windows follow ancestors"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            msg = fn()
            print(f"  {name:48s} ok   {msg or ''}")
            passed += 1
    print(f"{passed} passed")
