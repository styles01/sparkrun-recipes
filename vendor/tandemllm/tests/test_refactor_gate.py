"""The refactor gate: the exact bytes of a tiny model's run, recorded once, reproduced forever.

The extraction work (settings, layout, one Linear interface, drafter
dependencies, the package move) moves code around without changing what it computes. The
other CPU tests check properties (a tree equals its chain, a kernel equals its reference) within a
tolerance; this one checks that NOTHING moved: every logit and every byte of state after a fixed
sequence of prefill, decode, block verify, rollback, tree verify and commit, on two tiny models:

  * `bf16`: plain tensors, the same random 4-layer structure as tests/test_forward_tree.py
    (fp32 on the CPU, `linear()` falls through to `F.linear`);
  * `fp8`: the same structure at dims that are multiples of 128, every projection an `FP8Block`
    (e4m3 codes + a bf16 128x128 scale table, as the checkpoint stores them), run in bf16 -- the
    plain-FP8 weight path, `fp8_matmul`'s CPU branch.

The digests depend on the torch build and the CPU's GEMM kernels, so the fixture is keyed by
platform and torch version. The suite on the box (ops/gate.sh step 1) is the one that counts; its
key is committed. A host without a recorded key prints its digests and skips the comparison:

    QSE_GATE_RECORD=1 python tests/test_refactor_gate.py     # (re)record this host's key

Re-recording is allowed only for a commit that is SUPPOSED to change the arithmetic, and the
ledger entry has to say so.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine.config import TextConfig  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from tools.fp8_linear import BLOCK, FP8Block  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "refactor_gate.json")
DEV = "cpu"

PROJ_FP8 = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")


def config(kind: str) -> TextConfig:
    if kind == "bf16":
        return TextConfig(
            path="<random>", hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=97,
            rms_norm_eps=1e-6, rope_theta=10000.0, partial_rotary_factor=0.5,
            mrope_section=[2, 1, 1], max_position_embeddings=512,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4, mtp_num_hidden_layers=0,
            weight_block_size=(128, 128), eos_token_ids=[], bos_token_id=0)
    # every projection's N and K a multiple of 128: hidden 128, intermediate 256, conv_dim 256,
    # value_dim 128, q 512 (with the output gate), kv 128, attention out 256
    return TextConfig(
        path="<random>", hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=4, head_dim=32, vocab_size=211,
        rms_norm_eps=1e-6, rope_theta=10000.0, partial_rotary_factor=0.25,
        mrope_section=[2, 1, 1], max_position_embeddings=512,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, mtp_num_hidden_layers=0,
        weight_block_size=(128, 128), eos_token_ids=[], bos_token_id=0)


def quantize_fp8(w: torch.Tensor) -> FP8Block:
    """A 128x128 block-scaled e4m3 weight, the way the checkpoint stores one."""
    N, K = w.shape
    blocks = w.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    scale = (amax / 448.0).to(torch.bfloat16)
    s = scale.float()[:, None, :, None]
    codes = (blocks / s).clamp(-448, 448).to(torch.float8_e4m3fn).view(N, K)
    return FP8Block(codes, scale)


class TinyWeights:
    """The checkpoint's names over random tensors; projections as FP8Blocks for the fp8 model."""

    def __init__(self, cfg: TextConfig, kind: str, seed: int = 0, perturb: str | None = None):
        g = torch.Generator().manual_seed(seed)
        dt = torch.float32 if kind == "bf16" else torch.bfloat16
        self.t: dict = {}

        def r(*shape, scale=0.2):
            return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dt)

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
                self.t[f"{p}.linear_attn.conv1d.weight"] = r(c.conv_dim, 1, c.linear_conv_kernel_dim, scale=0.5)
                self.t[f"{p}.linear_attn.in_proj_z"] = r(c.value_dim, c.hidden_size)
                self.t[f"{p}.linear_attn.in_proj_b.weight"] = r(c.linear_num_value_heads, c.hidden_size)
                self.t[f"{p}.linear_attn.in_proj_a.weight"] = r(c.linear_num_value_heads, c.hidden_size)
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
        if kind == "fp8":
            for name in list(self.t):
                if name.split(".", 2)[-1] in PROJ_FP8:
                    self.t[name] = quantize_fp8(self.t[name])
        if perturb:
            # one step of the stored format on one weight: a bf16 ulp on a plain weight, one e4m3
            # code step on a quantised one. The gate must see either.
            t = self.t[perturb]
            if isinstance(t, FP8Block):
                codes = t.w.view(torch.uint8).view(-1)
                codes[0] = codes[0] + 1 if int(codes[0]) & 0x7F < 0x7E else codes[0] - 1
            else:
                flat = t.view(-1)
                flat[0] = (flat[0].float() * (1 + 2 ** -7)).to(t.dtype)

    def norm(self, name): return self.t[name]
    def proj(self, name): return self.t[name]
    def group(self, name): return None


BRANCHY = DraftTree(tokens=[41, 13, 62, 29, 55, 17, 88, 3, 71],
                    parents=[-1, 0, 1, 2, 1, 4, 0, 6, 6])


def _digest(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool
                          else t.numpy().tobytes()).hexdigest()[:24]


def run(kind: str, perturb: str | None = None) -> dict[str, str]:
    """One fixed script through the engine; the digest of everything it produced."""
    cfg = config(kind)
    eng = Qwen38Engine(cfg, TinyWeights(cfg, kind, seed=7, perturb=perturb), max_len=256, device=DEV)
    if kind == "bf16":
        eng.kv.k = eng.kv.k.to(torch.float32)
        eng.kv.v = eng.kv.v.to(torch.float32)
        eng.state.conv = eng.state.conv.to(torch.float32)
    torch.manual_seed(7)
    prompt = torch.randint(1, cfg.vocab_size, (12,))
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        out["prefill"] = eng.forward(prompt, start=0, last_only=True)
        pos = 12
        for i, tok in enumerate((5, 9, 44)):
            out[f"decode{i}"] = eng.forward(torch.tensor([tok]), start=pos, last_only=True)
            pos += 1
        blk = [7, 11, 23, 5, 31]
        out["block"] = eng.forward_block(torch.tensor(blk), start=pos)
        eng.rollback_to(3)
        pos += 3
        out["tree"] = eng.forward_tree(torch.tensor(BRANCHY.tokens), BRANCHY.parents, start=pos)
        eng.commit_tree(BRANCHY.path(5))
        pos += len(BRANCHY.path(5))
        out["after_commit"] = eng.forward(torch.tensor([19]), start=pos, last_only=True)
    out["S"] = eng.state.S
    out["conv"] = eng.state.conv
    out["kv_k"] = eng.kv.k[..., :pos + 1, :]
    out["kv_v"] = eng.kv.v[..., :pos + 1, :]
    return {k: _digest(v) for k, v in out.items()}


def host_key() -> str:
    return f"{platform.machine()}-{sys.platform}-torch{torch.__version__.split('+')[0]}"


def _load() -> dict:
    if os.path.isfile(FIXTURE):
        with open(FIXTURE) as f:
            return json.load(f)
    return {}


def _check(kind: str):
    got = run(kind)
    again = run(kind)
    assert got == again, f"{kind}: two runs in one process differ, the gate cannot work: " \
                         f"{[k for k in got if got[k] != again[k]]}"
    fx = _load()
    key = host_key()
    if os.environ.get("QSE_GATE_RECORD") == "1":
        fx.setdefault(key, {})[kind] = got
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w") as f:
            json.dump(fx, f, indent=1, sort_keys=True)
            f.write("\n")
        return f"recorded {len(got)} digests for {key}"
    want = fx.get(key, {}).get(kind)
    if want is None:
        return f"SKIP compare: no digests recorded for {key} (known: {sorted(fx) or 'none'})"
    bad = [k for k in want if want[k] != got.get(k)]
    assert not bad, f"{kind}: bytes moved in {bad} (host {key})"
    return f"{len(want)} digests reproduce ({key})"


def test_bf16_model_bytes():
    return _check("bf16")


def test_fp8_model_bytes():
    return _check("fp8")


def test_one_ulp_is_caught():
    """The gate is only worth something if it sees a one-ulp change in one projection."""
    msgs = []
    for kind, name in (("bf16", "layers.0.mlp.gate_proj"), ("fp8", "layers.3.self_attn.o_proj")):
        a, b = run(kind), run(kind, perturb=name)
        moved = [k for k in a if a[k] != b[k]]
        assert moved, f"{kind}: perturbing {name} moved nothing"
        msgs.append(f"{kind} {name}: {len(moved)}/{len(a)} digests moved")
    return "; ".join(msgs)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            msg = fn()
            print(f"  {name:<48} ok   {msg or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<48} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
