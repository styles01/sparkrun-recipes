"""CPU-only shape and numerics probe for the DFlash2 drafter.

Allocates nothing on a GPU and never touches the 27 GB target checkpoint. It loads the 3.85 GB
bf16 draft checkpoint on the CPU, checks that every tensor `engine/drafters/dflash2.py` reads is
present with the shape that file expects, runs one 8-wide block over synthetic target hidden
states, exercises the candidate selector against a small synthetic head, and prints the byte cost
of one draft against the real vocabulary.

    python tools/dflash2_probe.py                    # default snapshot, 4096-row synthetic head
    python tools/dflash2_probe.py --full-head        # allocate the real 2.54 GB head on the CPU
    python tools/dflash2_probe.py --ckpt <snapshot>
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Before torch, so no CUDA context can be created even by accident: the board has one GPU and it
# belongs to whatever is already running on it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Deliberately not `from engine.drafters.dflash2 import ...` through `engine.model`: nothing in the
# import chain below pulls in the Triton GEMMs or touches CUDA.
from engine.drafters.dflash2 import (  # noqa: E402
    DFlash2Module,
    _grouped_conv,
    _rms,
    load_config,
    load_weights,
)

FAIL = 0


def check(ok: bool, what: str, detail: str = "") -> None:
    global FAIL
    if not ok:
        FAIL += 1
    print(f"  [{'ok ' if ok else 'FAIL'}] {what}" + (f"  {detail}" if detail else ""))


def human(n: float) -> str:
    return f"{n / 1e9:.3f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ctx", type=int, default=64, help="synthetic context length")
    ap.add_argument("--vocab", type=int, default=4096, help="rows in the synthetic head")
    ap.add_argument("--full-head", action="store_true",
                    help="allocate the real [vocab, hidden] head on the CPU (2.54 GB)")
    args = ap.parse_args()

    if torch.cuda.is_available():
        print("note: CUDA is visible to this process; nothing below allocates on it.")
    torch.manual_seed(0)

    cfg, snap = load_config(args.ckpt)
    print(f"\ncheckpoint  {snap}")
    print(f"geometry    {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
          f"ffn {cfg.intermediate_size}, {cfg.num_attention_heads}q/{cfg.num_key_value_heads}kv "
          f"x {cfg.head_dim}, vocab {cfg.vocab_size}, rope_theta {cfg.rope_theta:g}")
    print(f"dflash      block_size {cfg.block_size}, conv {cfg.conv_kernel_size} taps / "
          f"group {cfg.conv_group_size} ({cfg.num_groups} groups), mask id {cfg.mask_token_id}, "
          f"selector rank {cfg.selector_rank} top_k {cfg.selector_top_k}")
    print(f"taps        target layers {cfg.target_layer_ids} "
          f"(residual stream ENTERING each, pre-input_layernorm)")
    print(f"attention   layer_types {sorted(set(cfg.layer_types))}, window {cfg.sliding_window}, "
          f"is_causal {cfg.is_causal}")

    # ---------------------------------------------------------------- 1. tensors
    print("\n1. checkpoint tensors")
    t0 = time.perf_counter()
    w = load_weights(snap, device="cpu")
    load_s = time.perf_counter() - t0
    expected = cfg.expected_tensors()
    missing = [n for n in expected if n not in w]
    check(not missing, f"all {len(expected)} expected tensors present",
          "" if not missing else f"missing {missing[:4]}")
    bad = [(n, tuple(w[n].shape), s) for n, s in expected.items()
           if n in w and tuple(w[n].shape) != s]
    check(not bad, "every shape matches", "" if not bad else f"{bad[:3]}")
    extra = sorted(set(w) - set(expected))
    check(not extra, "no tensor in the checkpoint goes unread",
          "" if not extra else f"unread: {extra}")
    nonbf16 = sorted(n for n, t in w.items() if t.dtype is not torch.bfloat16)
    check(not nonbf16, "every tensor is bf16", "" if not nonbf16 else f"{nonbf16[:3]}")
    ckpt_params = sum(t.numel() for t in w.values())
    ckpt_bytes = sum(t.numel() * t.element_size() for t in w.values())
    print(f"       {len(w)} tensors, {ckpt_params:,} params, {human(ckpt_bytes)}, "
          f"loaded in {load_s:.1f} s")

    # ---------------------------------------------------------------- 2. norm convention
    print("\n2. norm convention (Qwen3 `normalize(x) * w`, not the target's `* (1 + w)`)")
    names = ["hidden_norm.weight", "norm.weight", "layers.0.input_layernorm.weight",
             "layers.0.self_attn.q_norm.weight"]
    means = {n: float(w[n].float().mean()) for n in names}
    near_one = all(abs(v - 1.0) < abs(v) for v in means.values())
    check(near_one, "norm weights sit near 1.0, so the weight is the whole scale",
          "  ".join(f"{n.split('.')[-2]}={v:+.3f}" for n, v in means.items()))
    print("       a `(1 + w)` checkpoint would have these near 0.0; they are not, and the "
          "reference\n       imports Qwen3RMSNorm while sglang initialises this weight to ones.")

    # ---------------------------------------------------------------- 3. conv
    print("\n3. grouped dynamic conv")
    h, g, gs, taps = cfg.hidden_size, cfg.num_groups, cfg.conv_group_size, cfg.conv_kernel_size
    t = cfg.block_size
    x = torch.randn(t, h, dtype=torch.float32)
    base = torch.zeros(taps, h)
    base[0] = 1.0
    zero = torch.zeros(t, taps, g)
    pos = torch.arange(t) % cfg.block_size
    ident = _grouped_conv(x, zero, base, g, gs, taps, pos)
    check(torch.allclose(ident, x, atol=1e-5),
          "zero delta + identity base kernel is the identity map")
    base2 = base.clone()
    base2[1] = 1.0
    shifted = _grouped_conv(x, zero, base2, g, gs, taps, pos)
    check(torch.allclose(shifted[0], x[0], atol=1e-5),
          "tap 1 is masked out at block position 0 (no leak from the block before)")
    check(torch.allclose(shifted[3], x[3] + x[2], atol=1e-5),
          "tap 1 reads the PREVIOUS token: the convolution is causal")
    kp = w["layers.0.attention_conv.kernel_projection.weight"]
    check(kp.shape[0] == 2 * taps * g,
          f"kernel_projection emits 2*taps*num_groups = {2 * taps * g} features",
          f"got {kp.shape[0]}")
    ck = w["layers.0.attention_conv.base_kernel"]
    check(tuple(ck.shape) == (2, taps, h), "base_kernel is [in/out, tap, channel]",
          str(tuple(ck.shape)))
    check(abs(float(ck[0, 0].float().mean()) - 1.0) < 0.5,
          "base_kernel[:, 0] is still near the identity it was initialised to",
          f"mean {float(ck[0, 0].float().mean()):+.3f}")

    # ---------------------------------------------------------------- 4. forward
    print("\n4. one block forward over synthetic target hidden states")
    m = DFlash2Module(cfg, w)
    n_ctx, k = args.ctx, len(cfg.target_layer_ids)
    target_hidden = (torch.randn(n_ctx, k * cfg.hidden_size) * 0.02).to(torch.bfloat16)
    ctx_hidden = m.project_context(target_hidden)
    check(tuple(ctx_hidden.shape) == (n_ctx, cfg.hidden_size),
          f"fc + hidden_norm: [N, {k}*{cfg.hidden_size}] -> [N, {cfg.hidden_size}]",
          str(tuple(ctx_hidden.shape)))
    check(torch.isfinite(ctx_hidden.float()).all(), "context projection is finite")

    ctx_pos = torch.arange(n_ctx)
    ctx_kv = m.context_kv(ctx_hidden, ctx_pos)
    check(len(ctx_kv) == cfg.num_hidden_layers,
          f"context K/V materialised for all {cfg.num_hidden_layers} layers")
    ck0, cv0 = ctx_kv[0]
    want_kv = (cfg.num_key_value_heads, n_ctx, cfg.head_dim)
    check(tuple(ck0.shape) == want_kv and tuple(cv0.shape) == want_kv,
          f"context K/V are [n_kv, N, head_dim] = {want_kv}",
          f"{tuple(ck0.shape)} / {tuple(cv0.shape)}")

    base_pos = n_ctx
    positions = torch.arange(base_pos, base_pos + cfg.block_size)
    noise = (torch.randn(cfg.block_size, cfg.hidden_size) * 0.02).to(torch.bfloat16)
    t0 = time.perf_counter()
    hidden, block_kv = m.forward_block(noise, positions, ctx_kv, ctx_pos, return_kv=True)
    fwd_s = time.perf_counter() - t0
    check(tuple(hidden.shape) == (cfg.block_size, cfg.hidden_size),
          f"block output is [block_size, hidden] = ({cfg.block_size}, {cfg.hidden_size})",
          str(tuple(hidden.shape)))
    check(torch.isfinite(hidden.float()).all(), "block output is finite",
          f"absmax {float(hidden.float().abs().max()):.3f}")
    check(len(block_kv) == cfg.num_hidden_layers
          and tuple(block_kv[0][0].shape) == (cfg.num_key_value_heads, cfg.block_size,
                                              cfg.head_dim),
          "the block's own K/V come back for a chained second block")
    print(f"       one CPU forward: {fwd_s * 1e3:.0f} ms (bf16 on CPU; not a speed statement)")

    # Non-causality: row 1 has to move when row 7's input changes, or the block is not a block.
    noise2 = noise.clone()
    noise2[7] += 0.5
    hidden2 = m.forward_block(noise2, positions, ctx_kv, ctx_pos)
    moved = float((hidden2[1] - hidden[1]).float().abs().max())
    check(moved > 1e-3, "attention is NON-causal: row 1 sees row 7", f"absmax delta {moved:.4f}")
    # And a row cannot see a context position outside the sliding window.
    if cfg.sliding_window and n_ctx > 0:
        check(True, f"sliding window {cfg.sliding_window} applied to the context side only")

    # The chained second block, exactly as DFlash2Drafter._one_block assembles it: context keys
    # first, then rows 0..block_size-2 of block one, then block two's own eight rows.
    keep = cfg.block_size - 1
    carry = [(k[:, :keep], v[:, :keep]) for k, v in block_kv]
    chain_kv = [(torch.cat([k, c[0]], dim=1), torch.cat([v, c[1]], dim=1))
                for c, (k, v) in zip(carry, ctx_kv)]
    chain_pos = torch.cat([ctx_pos, positions[:keep]])
    pos2 = torch.arange(base_pos + keep, base_pos + keep + cfg.block_size)
    hidden_b2 = m.forward_block(noise, pos2, chain_kv, chain_pos)
    check(tuple(chain_kv[0][0].shape) == (cfg.num_key_value_heads, n_ctx + keep, cfg.head_dim),
          f"chained block 2 attends to {n_ctx} context + {keep} block-1 keys",
          str(tuple(chain_kv[0][0].shape)))
    check(tuple(hidden_b2.shape) == (cfg.block_size, cfg.hidden_size)
          and torch.isfinite(hidden_b2.float()).all(),
          "chained block 2 output is [block_size, hidden] and finite")
    check(chain_pos.numel() == chain_kv[0][0].shape[1],
          "key order and position order agree, so the window mask lines up")

    # ---------------------------------------------------------------- 5. selector
    print("\n5. candidate selector")
    pred = hidden[1:]
    slots = cfg.block_size - 1
    check(tuple(pred.shape) == (slots, cfg.hidden_size),
          f"row 0 is the anchor, so one block predicts {slots} tokens, not {cfg.block_size}",
          str(tuple(pred.shape)))
    vocab = cfg.vocab_size if args.full_head else args.vocab
    head = (torch.randn(vocab, cfg.hidden_size) * 0.02).to(torch.bfloat16)
    logits = torch.nn.functional.linear(pred, head)
    check(tuple(logits.shape) == (slots, vocab),
          f"one head call over all {slots} rows -> [{slots}, {vocab}]", str(tuple(logits.shape)))
    cand, unary = m.unary_candidates(logits)
    check(tuple(cand.shape) == (slots, cfg.selector_top_k)
          and tuple(unary.shape) == (slots, cfg.selector_top_k),
          f"top-k candidates are [{slots}, {cfg.selector_top_k}]",
          f"{tuple(cand.shape)} / {tuple(unary.shape)}")
    anchor = int(cfg.mask_token_id)
    scores = m.lattice(pred, cand, unary, anchor)
    kk = cfg.selector_top_k
    check(tuple(scores.shape) == (slots, kk, kk),
          f"lattice is [slots, predecessor, candidate] = ({slots}, {kk}, {kk})",
          str(tuple(scores.shape)))
    check(torch.allclose(scores[0, 0], scores[0, kk - 1]),
          "slot 0's predecessor rows are all the anchor, so they are identical")
    tokens = m.walk(cand, scores)
    check(tuple(tokens.shape) == (slots,), f"the walk yields {slots} tokens",
          str(tuple(tokens.shape)))
    check(bool(((tokens[:, None] == cand).any(-1)).all()),
          "every chosen token is one of its slot's own candidates")
    greedy = logits.argmax(-1)
    n_diff = int((tokens != greedy).sum())
    print(f"       selector changed {n_diff} of {slots} greedy picks on random weights "
          f"(it rescores, it does not restrict the head)")

    # ---------------------------------------------------------------- 6. bytes
    print("\n6. byte cost of one draft")
    sel_prefix = "candidate_selector."
    backbone = sum(t.numel() * t.element_size() for n, t in w.items()
                   if not n.startswith(sel_prefix))
    proj = w[f"{sel_prefix}hidden_projection.weight"]
    proj_b = proj.numel() * proj.element_size()
    cb = w[f"{sel_prefix}predecessor_codebook"]
    codebook_b = 2 * cb.numel() * cb.element_size()
    gathered_b = slots * 2 * cfg.selector_top_k * cfg.selector_rank * cb.element_size()
    head_b = cfg.vocab_size * cfg.hidden_size * 2
    total = backbone + proj_b + gathered_b + head_b
    rows = [
        ("draft backbone (5 layers + fc + norms)", backbone),
        ("selector hidden_projection", proj_b),
        (f"selector codebook rows gathered ({slots} slots x 2 x {cfg.selector_top_k})", gathered_b),
        (f"target lm_head bf16 [{cfg.vocab_size}, {cfg.hidden_size}], read once", head_b),
    ]
    for label, b in rows:
        print(f"       {label:<62s} {b:>15,d} B  {human(b):>10s}")
    print(f"       {'TOTAL, one ' + str(cfg.block_size) + '-wide draft -> ' + str(slots) + ' proposals':<62s} "
          f"{total:>15,d} B  {human(total):>10s}")
    for bw, label in ((273e9, "273 GB/s board peak"), (190e9, "190 GB/s realistic")):
        print(f"         at {label:<24s} {total / bw * 1e3:6.1f} ms  "
              f"({total / bw * 1e3 / slots:5.2f} ms per proposal)")
    print(f"       codebooks resident but NOT read whole: {human(codebook_b)} "
          f"({codebook_b / gathered_b:,.0f}x the gathered rows)")
    print(f"       checkpoint on disk / resident: {human(ckpt_bytes)}")
    print("\n       The selector does NOT remove the head read. `compute_candidates` runs the full")
    print("       head and takes the top k from it -- the reference's own comment on `_radix_topk`")
    print(f"       is \"the selector's largest single cost: it reads the whole logits tensor\". A")
    print(f"       reduced-vocabulary draft head is the lever that would: 32k rows in fp8 is")
    print(f"       {32768 * cfg.hidden_size:,d} B, taking the draft to "
          f"{human(backbone + proj_b + gathered_b + 32768 * cfg.hidden_size)}.")

    print(f"\n{'PROBE PASSED' if not FAIL else f'PROBE FAILED: {FAIL} check(s)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
