"""Fine-tune the block drafter on the target's own distribution, offline.

WHY THIS EXISTS
---------------
The engine's speed is acceptance times step rate, and the step is at the weight format's floor. The
row this program is measured against is fresh prose and chat, where the released drafter accepts
3.28 tokens a block against 6.35 on an edit. That is not a kernel problem: it is a distribution
problem. The drafter was trained on a generic English instruction and code mixture, and it is asked
here to predict a particular 27 B model on a particular workload.

Nothing about output can change. A drafter proposes; the target verifies every token against its
own argmax and keeps the matching prefix. Training it wrong makes the engine slower and never wrong,
which is why this file has no quality gate and the losslessness gate in `tools/verify_spec.py` is
the only correctness statement needed.

THE OBJECTIVE
-------------
The drafter's forward pass is non-causal over a block of eight rows: row 0 carries the last token
the target committed, rows 1..7 are mask tokens, and row j predicts the token at anchor + j. So the
loss is over those seven rows, per block:

    L = CE(draft_logits[j], target_argmax[anchor + j])
        + w * KL(target_top64[anchor + j] || draft[j])

The hard term is the one that matches the gate exactly -- greedy verification accepts row j if and
only if its argmax equals the target's -- and the soft term is what stops the hard term from
over-fitting a single token when the target itself was nearly indifferent.

WHAT MAKES IT CHEAP
-------------------
The target never runs. `tools/train_data.py` has already written its five tap tensors and its top-64
for every position, so a step reads tensors from disk and touches 1.9 B parameters, not 27 B. The
drafter's context is materialised the same way serving materialises it -- `project_context` then
`context_kv` -- and many blocks of one sequence are packed into a single pass behind a mask that
keeps them from seeing each other.

THE GATE
--------
`--eval-only`, or the evaluation that runs at every checkpoint, replays the serving loop against
held-out self-generated sequences: draft seven tokens from the anchor, count the matching prefix,
commit that many plus the target's bonus token, move the anchor, repeat. The number it reports is
the number the ledger calls accepted tokens per block, and it is exact rather than estimated,
because the target's greedy continuation of those sequences is known.

Held-out sequences are named in the data manifest and are never sampled for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.dflash2 import (  # noqa: E402
    DFlash2Module, load_config, load_weights, resolve_checkpoint,
)


# ----------------------------------------------------------------------------- target tensors

def target_tensors(snapshot: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """`embed_tokens.weight` and `lm_head.weight`, and nothing else of the 27 B target.

    The drafter embeds its noise block with the target's embedding table and turns its output into
    tokens with the target's head. Those two tensors are 5.1 GB together; the other 25 GB of the
    checkpoint has no role in training.
    """
    from safetensors import safe_open
    import glob
    want = {"embed_tokens.weight": None, "lm_head.weight": None}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for name in list(want):
                for cand in (name, f"model.{name}", f"model.language_model.{name}"):
                    if cand in keys and want[name] is None:
                        want[name] = f.get_tensor(cand).to(torch.bfloat16).to(device)
                        break
    if want["embed_tokens.weight"] is None:
        raise RuntimeError(f"no embed_tokens.weight under {snapshot}")
    if want["lm_head.weight"] is None:                       # tied embeddings, if it ever happens
        want["lm_head.weight"] = want["embed_tokens.weight"]
    return want["embed_tokens.weight"], want["lm_head.weight"]


# ----------------------------------------------------------------------------- data

def klass(topic: str) -> str:
    """The three regimes the gate is written in terms of, from the prompt's topic.

    The row this program is measured against is chat-shaped prompts across eleven topics; what
    separates the acceptance regimes is not the topic but whether the output is code, German, or
    English prose, which is how the ledger has reported acceptance since 10:17."""
    if topic in ("code",):
        return "code"
    if topic in ("multilingual", "de"):
        return "de"
    if topic == "en":
        return "prose"
    return "prose"


class Sample:
    __slots__ = ("name", "kind", "topic", "klass", "split", "ids", "fused", "label", "top_ids",
                 "top_lp", "gen_start")


    def __init__(self, meta: dict, blob: dict, device: str, keep_on: str):
        self.name = meta["name"]
        self.kind = meta["kind"]
        self.topic = meta["topic"]
        self.klass = klass(meta["topic"])
        self.split = meta["split"]
        self.gen_start = int(meta.get("gen_start", 0))
        dev = device if keep_on == "cuda" else "cpu"
        self.ids = blob["ids"].to(dev).long()
        self.fused = blob["fused"].to(dev)
        self.label = blob["label"].to(dev).long()
        self.top_ids = blob["top_ids"].to(dev).long()
        self.top_lp = blob["top_lp"].to(dev)

    def __len__(self) -> int:
        return int(self.ids.numel())


def load_data(path: str, device: str, keep_on: str, limit: int = 0) -> list[Sample]:
    """Every recorded sequence, from either layout: one file each, or a few large shards.

    THE TWO LAYOUTS, AND WHY THE SECOND ONE EXISTS

    The recorder writes one `.pt` per sequence because a recorder has to be resumable by
    name. That layout costs a Python open, unpickle and convert per sequence, and on the full set --
    5,863 files, 109 GB -- it did not finish in fifty-four minutes off a bucket mount and was still
    running at ten minutes off local NVMe. The pipeline calls this function four to seven times in
    one job, so the cost is paid four to seven times with the GPU idle.

    The sharder repacks the same bytes into ~40 files of ~2.7 GB and writes a manifest
    that says where each sequence sits. This function then opens forty files instead of 5,863 and
    memory-maps them, so `fused` -- 99 % of the bytes, and the only field the trainer reads more
    than one sequence of -- is never copied into host RAM at all. The page cache decides what stays
    resident; a step touches ~20 MB of one sequence.

    `--keep-on cuda` is the one case that cannot be lazy: the tensors go to the GPU on the way in,
    which is a copy whatever the layout, and 109 GB does not fit. It stays supported for the small
    recordings it was written for.
    """
    with open(os.path.join(path, "manifest.json")) as f:
        manifest = json.load(f)
    metas = manifest["sequences"]
    if limit:
        metas = metas[:limit]
    if manifest.get("sharded"):
        return _load_sharded(path, metas, device, keep_on)
    out = []
    for meta in metas:
        f = os.path.join(path, f"{meta['name']}.pt")
        if not os.path.exists(f):
            continue
        out.append(Sample(meta, torch.load(f, map_location="cpu"), device, keep_on))
    return out


# the per-sequence fields a shard holds, in the sharder's order
SHARD_FIELDS = ("fused", "ids", "label", "top_ids", "top_lp")


def _load_sharded(path: str, metas: list[dict], device: str, keep_on: str) -> list[Sample]:
    """One `torch.load(..., mmap=True)` per shard, then a contiguous view per sequence.

    The shards are opened in the order the sequences ask for them and held for as long as any
    sample references them -- a slice of a memory-mapped tensor shares its storage, so letting the
    bundle go would unmap rows the trainer has not read yet.
    """
    bundles: dict[int, dict] = {}
    out: list[Sample] = []
    t0 = time.perf_counter()
    for meta in metas:
        i = meta.get("shard")
        if i is None:
            continue
        if i not in bundles:
            f = os.path.join(path, f"shard-{i:04d}.pt")
            if not os.path.exists(f):
                continue
            bundles[i] = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
            print(f"  [shard {i:4d}] {len(bundles)} open, {len(out):6d} sequences, "
                  f"{time.perf_counter()-t0:6.1f} s", flush=True)
        sh = bundles[i]
        o, n = int(meta["offset"]), int(meta["n"])
        out.append(Sample(meta, {f: sh[f][o:o + n] for f in SHARD_FIELDS}, device, keep_on))
    print(f"  [shards] {len(out)} sequences from {len(bundles)} shards in "
          f"{time.perf_counter()-t0:.1f} s", flush=True)
    return out


# ----------------------------------------------------------------------------- masks

def block_masks(anchor_pos: torch.Tensor, ctx_len: int, block: int, window: int | None,
                device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The (full, sliding) attention masks for `B` packed blocks over a shared context.

    Query rows are `B * block` of them, keys are `ctx_len` context rows followed by the same query
    rows. A query in block b may read a context position strictly before its anchor -- that is the
    serving rule, `ctx_len == anchor position` -- and may read every row of its OWN block and no row
    of any other. Inside a block the pass is deliberately non-causal.
    """
    b = anchor_pos.numel()
    t = b * block
    qblk = torch.arange(t, device=device) // block
    qpos = anchor_pos[qblk] + (torch.arange(t, device=device) % block)
    ctx_pos = torch.arange(ctx_len, device=device)
    kpos = torch.cat([ctx_pos, qpos])
    kblk = torch.cat([torch.full((ctx_len,), -1, device=device, dtype=torch.long), qblk])
    is_ctx = torch.arange(ctx_len + t, device=device) < ctx_len

    same_block = kblk[None, :] == qblk[:, None]
    before_anchor = kpos[None, :] < anchor_pos[qblk][:, None]
    full = torch.where(is_ctx[None, :], before_anchor, same_block)
    if window is None:
        return full, full
    in_window = (qpos[:, None] - kpos[None, :]) < window
    slide = torch.where(is_ctx[None, :], before_anchor & in_window, same_block)
    return full, slide


# ----------------------------------------------------------------------------- one training step

def run_blocks(m: DFlash2Module, embed: torch.Tensor, s: Sample, anchors: torch.Tensor,
               device: str, block: int = 0) -> torch.Tensor:
    """Draft hidden states for `len(anchors)` blocks of one sequence. [B, block-1, H]."""
    cfg = m.cfg
    bs = block or cfg.block_size
    b = anchors.numel()
    ctx_len = int(anchors.max().item())               # every block reads strictly before its anchor
    fused = s.fused[:ctx_len].to(device, non_blocking=True)
    ctx_hidden = m.project_context(fused.to(m.dtype))
    ctx_pos = torch.arange(ctx_len, device=device)
    ctx_kv = m.context_kv(ctx_hidden, ctx_pos)

    ids = s.ids.to(device, non_blocking=True)
    block_ids = torch.full((b, bs), cfg.mask_token_id, dtype=torch.long, device=device)
    block_ids[:, 0] = ids[anchors]
    noise = F.embedding(block_ids.reshape(-1), embed).to(m.dtype)
    positions = (anchors[:, None] + torch.arange(bs, device=device)[None, :]).reshape(-1)
    masks = block_masks(anchors, ctx_len, bs, cfg.sliding_window, device)
    out = m.forward_block(noise, positions, ctx_kv, ctx_pos, block_size=bs, masks=masks)
    return out.view(b, bs, -1)[:, 1:, :]


def loss_of(pred: torch.Tensor, head: torch.Tensor, s: Sample, anchors: torch.Tensor,
            kl_weight: float, device: str,
            logits: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy to the target's argmax plus KL to its top-64. Returns (loss, hits).

    `logits` lets a caller that also needs the head's output (the ranking term) compute it once;
    passing it changes nothing about the result."""
    b, l, h = pred.shape
    if logits is None:
        logits = F.linear(pred.reshape(-1, h), head).float()              # [b*l, V]
    # Row j of a block (j counted from 1, because row 0 is the anchor and dead) predicts the token
    # at anchor + j, and `label[i]` is the token that follows position i -- so the label wanted is
    # `label[anchor + j - 1]`. Getting this one index wrong costs nothing visible: the loss still
    # falls, on a task one position to the left of the one the verify pass grades.
    slots = anchors[:, None] + torch.arange(1, l + 1, device=device)[None, :]
    idx = (slots - 1).reshape(-1)
    label = s.label.to(device, non_blocking=True)[idx]
    ce = F.cross_entropy(logits, label)
    hits = (logits.argmax(-1) == label).float().sum()
    if kl_weight <= 0:
        return ce, hits
    top_ids = s.top_ids.to(device, non_blocking=True)[idx]
    top_lp = s.top_lp.to(device, non_blocking=True)[idx].float()
    p = torch.softmax(top_lp, dim=-1)                       # renormalised over the stored top-k
    q = torch.log_softmax(logits, dim=-1).gather(-1, top_ids)
    kl = (p * (torch.log(p.clamp_min(1e-9)) - q)).sum(-1).mean()
    return ce + kl_weight * kl, hits


# ----------------------------------------------------------------------------- the ranking term

PRED_CB = "candidate_selector.predecessor_codebook"
SUCC_CB = "candidate_selector.successor_codebook"
HPROJ = "candidate_selector.hidden_projection.weight"


def rank_loss_of(m: DFlash2Module, pred: torch.Tensor, logits: torch.Tensor, s: Sample,
                 anchors: torch.Tensor, device: str) -> tuple[torch.Tensor, int, int]:
    """The engine's own ranking objective on the served candidate set (RND-37).

    The tree builder orders a slot's 16 candidates by the selector's lattice score
    (`DFlash2Module.lattice`):

        Score[j, p, c] = unary[j, c] + < A[pred[j, p]] * P h[j], B[cand[j, c]] >

    with `unary` the head's top-16 logits (`unary_candidates`). On the TRUE path, the row that
    matters at slot j is the one whose predecessor is the true token before it (the anchor at slot
    0). This is the cross-entropy over the 16 candidates on that row, against the index of the true
    token at slot j -- counted only where the true path is still inside the lattice (the true tokens
    of slots 0..j are all among their slot's candidates), because no tree can reach slot j otherwise.

    The codebooks are read through `F.embedding(..., sparse=True)`: a step touches at most
    (1 + 16) rows a slot of a 248,320-row table, and a sparse gradient is what lets the optimiser
    leave every other row bit for bit where it was.

    Returns (mean CE over the counted slots, counted slots, slots where the true token already
    ranks first)."""
    b, l, h = pred.shape
    w = m.w
    k = m.cfg.selector_top_k
    cand, unary = m.unary_candidates(logits)                         # [b*l, k] (unary has grad)
    cand = cand.view(b, l, k)
    unary = unary.view(b, l, k)
    idx = anchors[:, None] + torch.arange(l, device=device)[None, :]  # label index of slot j
    lab = s.label.to(device, non_blocking=True)[idx]                  # true token at slot j
    hit = cand == lab[..., None]
    inl = hit.any(-1)
    tgt = hit.long().argmax(-1)
    reach = torch.cumprod(inl.long(), dim=1).bool()                   # slots 0..j all in-lattice
    anchor_ids = s.ids.to(device, non_blocking=True)[anchors]
    prev = torch.cat([anchor_ids[:, None], lab[:, :-1]], dim=1)       # true predecessor of slot j
    hidden = F.linear(pred, w[HPROJ])                                 # [b, l, r]
    a_rows = F.embedding(prev, w[PRED_CB], sparse=True)               # [b, l, r]
    b_rows = F.embedding(cand, w[SUCC_CB], sparse=True)               # [b, l, k, r]
    pair = a_rows.float() * hidden.float()
    scores = unary + torch.einsum("blr,blkr->blk", pair, b_rows.float())
    n = int(reach.sum())
    if n == 0:
        return scores.sum() * 0.0, 0, 0
    sc, tg = scores[reach], tgt[reach]
    top1 = int((sc.argmax(-1) == tg).sum())
    return F.cross_entropy(sc, tg), n, top1


@torch.no_grad()
def tree_acceptance(m: DFlash2Module, embed: torch.Tensor, head: torch.Tensor,
                    samples: list[Sample], device: str, budget: int, block: int = 0,
                    max_blocks: int = 64, temp: float = 1.0) -> dict:
    """Committed tokens a block of the SERVED tree on held-out sequences, and the true token's rank.

    Each block: the drafter's lattice at the anchor (`unary_candidates` + `lattice`, exactly as the
    engine builds it), `engine.tree.lattice_tree` at `budget` nodes (nodes mode, temperature
    `temp`, as `tools/lattice_oracle.py` replays it), the accepted run against the target's own
    labels, plus the target's token; the anchor moves by that. Also, following the true path at
    slots 0..2, the share of positions where the true token is the row's first choice.

    Classes are the sequences' topics (`think-*` sequences reported as `think`)."""
    import numpy as np
    from engine.tree import lattice_tree
    cfg = m.cfg
    bs = block or cfg.block_size
    acc: dict[str, list[float]] = {}
    r1: dict[str, list[int]] = {}
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
    for s in samples:
        kl = "think" if s.name.startswith("think-") else s.topic
        n = len(s)
        fused = s.fused.to(device, non_blocking=True)
        ids = s.ids.to(device, non_blocking=True)
        label = s.label.tolist()
        with ac:
            ctx_hidden = m.project_context(fused.to(m.dtype))
            ctx_pos_all = torch.arange(n, device=device)
            kv_all = [(kk.to(torch.bfloat16), vv.to(torch.bfloat16))
                      for kk, vv in m.context_kv(ctx_hidden, ctx_pos_all)]
        p = max(1, s.gen_start - 1)
        got: list[float] = []
        firsts = [0, 0, 0, 0, 0, 0]                      # (rank-1, reached) for slots 0..2
        while p <= n - bs and len(got) < max_blocks:
            lo = 0 if cfg.sliding_window is None else max(0, p - cfg.sliding_window + 1)
            ctx_kv = [(kk[:, lo:p], vv[:, lo:p]) for kk, vv in kv_all]
            block_ids = torch.full((bs,), cfg.mask_token_id, dtype=torch.long, device=device)
            block_ids[0] = ids[p]
            noise = F.embedding(block_ids, embed).to(m.dtype)
            positions = torch.arange(p, p + bs, device=device)
            with ac:
                pred = m.forward_block(noise, positions, ctx_kv, ctx_pos_all[lo:p],
                                       block_size=bs)[1:]
                logits = F.linear(pred.to(head.dtype), head).float()
                cand, unary = m.unary_candidates(logits)
                scores = m.lattice(pred, cand, unary, int(ids[p]))
            cand_l = cand.tolist()
            sc = scores.float().cpu().numpy()
            y = sc / temp
            y = y - y.max(axis=-1, keepdims=True)
            logp = y - np.log(np.exp(y).sum(axis=-1, keepdims=True))
            greedy = [int(sc[0, 0].argmax())]
            for e in range(1, sc.shape[0]):
                greedy.append(int(sc[e, greedy[-1]].argmax()))
            t = lattice_tree(int(ids[p]), cand_l, logp.tolist(), greedy, budget - 1)
            kids: dict[int, dict[int, int]] = {}
            for i, par in enumerate(t.parents[1:], start=1):
                kids.setdefault(par, {})[t.tokens[i]] = i
            node, a_n = 0, 0
            while p + a_n < n - 1 and a_n < bs - 1:
                nxt = kids.get(node, {}).get(label[p + a_n])
                if nxt is None:
                    break
                node, a_n = nxt, a_n + 1
            row = 0
            for e in range(3):
                tt = label[p + e]
                if tt not in cand_l[e]:
                    break
                c = cand_l[e].index(tt)
                firsts[2 * e + 1] += 1
                firsts[2 * e] += int(int(sc[e, row].argmax()) == c)
                row = c
            got.append(a_n + 1.0)
            p += a_n + 1
        acc.setdefault(kl, []).extend(got)
        r1.setdefault(kl, [0] * 6)
        r1[kl] = [x + y_ for x, y_ in zip(r1[kl], firsts)]
    out = {k_: sum(v) / len(v) for k_, v in acc.items() if v}
    allv = [x for k_, v in acc.items() if k_ != "think" for x in v]
    out["ALL"] = sum(allv) / max(1, len(allv))
    out["blocks"] = float(len(allv))
    for k_, f in r1.items():
        for e in range(3):
            if f[2 * e + 1]:
                out[f"r1s{e}_{k_}"] = f[2 * e] / f[2 * e + 1]
    return out


# ----------------------------------------------------------------------------- the gate

@torch.no_grad()
def acceptance(m: DFlash2Module, embed: torch.Tensor, head: torch.Tensor, samples: list[Sample],
               device: str, use_selector: bool = True, max_blocks: int = 64,
               block: int = 0) -> dict:
    """Replay the serving loop on held-out self-generated sequences. Exact, not estimated.

    A block anchored at `p` drafts seven tokens; the verify pass accepts the matching prefix and
    adds one bonus token of its own, so `m + 1` positions are committed and the next anchor is
    `p + m + 1`. That is the number the ledger calls accepted tokens per block.
    """
    cfg = m.cfg
    bs = block or cfg.block_size
    per_topic: dict[str, list[float]] = {}
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
    for s in samples:
        n = len(s)
        fused = s.fused.to(device, non_blocking=True)
        ids = s.ids.to(device, non_blocking=True)
        label = s.label.to(device, non_blocking=True)
        with ac:
            ctx_hidden = m.project_context(fused.to(m.dtype))
            ctx_pos_all = torch.arange(n, device=device)
            kv_all = [(k.to(torch.bfloat16), v.to(torch.bfloat16))
                      for k, v in m.context_kv(ctx_hidden, ctx_pos_all)]
        p = max(1, s.gen_start - 1)
        got: list[float] = []
        while p <= n - bs and len(got) < max_blocks:
            lo = 0 if cfg.sliding_window is None else max(0, p - cfg.sliding_window + 1)
            ctx_kv = [(k[:, lo:p], v[:, lo:p]) for k, v in kv_all]
            ctx_pos = ctx_pos_all[lo:p]
            block_ids = torch.full((bs,), cfg.mask_token_id, dtype=torch.long, device=device)
            block_ids[0] = ids[p]
            noise = F.embedding(block_ids, embed).to(m.dtype)
            positions = torch.arange(p, p + bs, device=device)
            with ac:
                pred = m.forward_block(noise, positions, ctx_kv, ctx_pos, block_size=bs)[1:]
                logits = F.linear(pred.to(head.dtype), head).float()
            if use_selector and cfg.selector_rank:
                cand, unary = m.unary_candidates(logits)
                scores = m.lattice(pred, cand, unary, int(ids[p]))
                draft = m.walk(cand, scores)
            else:
                draft = logits.argmax(-1)
            want = label[p:p + bs - 1]                       # label[p+j] is the token at p+j+1
            match = int((draft != want).float().argmax()) if (draft != want).any() else bs - 1
            got.append(match + 1.0)
            p += match + 1
        per_topic.setdefault(s.klass, []).extend(got)
    out = {k: sum(v) / len(v) for k, v in per_topic.items() if v}
    allv = [x for v in per_topic.values() for x in v]
    out["ALL"] = sum(allv) / max(1, len(allv))
    out["blocks"] = float(len(allv))
    return out


# ----------------------------------------------------------------------------- export

def export(w: dict[str, torch.Tensor], snapshot: str, out_dir: str, block: int = 0) -> None:
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
        src = os.path.join(snapshot, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, name))
    if block:
        # The checkpoint has to say what it was trained at, or a drafter loaded from it a week from
        # now proposes seven tokens from a module that learned fifteen.
        path = os.path.join(out_dir, "config.json")
        with open(path) as f:
            raw = json.load(f)
        raw["dflash_config"]["block_size"] = int(block)
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)
    flat = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in w.items()}
    save_file(flat, os.path.join(out_dir, "model.safetensors"))


# ----------------------------------------------------------------------------- main

def select_params(w: dict, cfg, mode: str) -> list[str]:
    """Which tensors move. The selector codebooks never do: they are 254 MB of rows gathered
    sixteen at a time, whose gradient at this batch size is almost all zeros, and what they score
    is a re-ranking of the head's own top-16 rather than the prediction itself."""
    if mode == "all":
        return [k for k in w if not k.startswith("candidate_selector.")]
    base = ["fc.weight", "hidden_norm.weight", "norm.weight"]
    if mode == "ranksel":
        # the selector alone: the head's marginal (and so every unary score) stays exactly as served,
        # only the bilinear order term learns
        return [HPROJ, PRED_CB, SUCC_CB]
    if mode.startswith("rank"):
        # RND-37: the selector (its projection dense, its two codebooks sparse) plus the top of the
        # drafter: fc, the norms and the last N layers (`rank` = last 2, `rank3` = last 3).
        n = int(mode[4:] or 2)
        keep = set(range(cfg.num_hidden_layers - n, cfg.num_hidden_layers))
        return ([HPROJ, PRED_CB, SUCC_CB] + base
                + [k for k in w if k.startswith("layers.") and int(k.split(".")[1]) in keep])
    if mode == "fc":
        return base
    if mode.startswith("last"):
        n = int(mode[4:])
        keep = set(range(cfg.num_hidden_layers - n, cfg.num_hidden_layers))
        return base + [k for k in w if k.startswith("layers.")
                       and int(k.split(".")[1]) in keep]
    raise SystemExit(f"unknown --train {mode}")


# ----------------------------------------------------------------------------- resume

RESUME_VERSION = 1


def data_fingerprint(train_gen, train_corp, held) -> str:
    """What the resume state was trained against, in one line.

    A resume that silently accepts a different recording would restore an optimiser and a random
    state that index into a pool of another length -- the run would continue, the loss would look
    normal, and the data cursor would be meaningless. So the names and the split are hashed and the
    resume refuses on a mismatch rather than carrying on.
    """
    h = hashlib.sha256()
    for pool, kind in ((train_gen, "gen"), (train_corp, "corp"), (held, "held")):
        h.update(kind.encode())
        for s_ in pool:
            h.update(s_.name.encode())
            h.update(str(len(s_)).encode())
    return h.hexdigest()[:16]


def state_path(d: str, tag: str) -> str:
    return os.path.join(d, f"resume-{tag}.pt")


def done_path(d: str, tag: str) -> str:
    return os.path.join(d, f"resume-{tag}.done")


def save_state(path: str, *, w, opt, step, best, rng, tag, block, lr, mode, steps_total,
               fingerprint, elapsed, dtype=torch.bfloat16) -> float:
    """Everything a continuation needs: weights, optimiser, schedule position, data cursor.

    There is no scheduler OBJECT to save -- the learning rate is a closed form of `step`, so the
    step is the schedule. The data cursor is the same: the training loop draws every sequence and
    every anchor from one `random.Random`, so its state plus the step is exactly where the run had
    got to in the data.

    Written to a `.part` and renamed, because a resume state that exists has to be one that can be
    read. A job killed mid-write would otherwise leave a file that fails to unpickle, which is a
    worse outcome than having no state at all.

    Stored at `dtype` (bf16 by default). The weights are exported at bf16 anyway, and the two Adam
    moments are 15.2 GB at fp32 against 7.6 at bf16 -- and bf16 has fp32's exponent range, so the
    small second moments do not underflow, they lose mantissa. Returns the bytes written.
    """
    blob = {
        "version": RESUME_VERSION, "tag": tag, "block": block, "lr": lr, "train": mode,
        "step": step, "best": best, "steps_total": steps_total, "elapsed": elapsed,
        "fingerprint": fingerprint,
        "rng_py": rng.getstate(), "rng_torch": torch.get_rng_state(),
        "weights": {k: v.detach().to("cpu", dtype) for k, v in w.items()
                    if v.is_floating_point()},
        "opt": _opt_to(opt.state_dict(), dtype),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".part"
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return float(os.path.getsize(path))


def _opt_to(sd: dict, dtype) -> dict:
    out = {"param_groups": sd["param_groups"], "state": {}}
    for k, v in sd["state"].items():
        out["state"][k] = {kk: (vv.to("cpu", dtype) if torch.is_tensor(vv) and
                                vv.is_floating_point() and vv.numel() > 1
                                else (vv.cpu() if torch.is_tensor(vv) else vv))
                           for kk, vv in v.items()}
    return out


def load_state(path: str, *, w, opt, rng, fingerprint: str, tag: str, block: int) -> dict:
    """Restore in place. Raises rather than continuing if the state is for another run."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("version") != RESUME_VERSION:
        raise SystemExit(f"{path}: resume version {blob.get('version')}, this trainer writes "
                         f"{RESUME_VERSION}")
    if blob.get("fingerprint") != fingerprint:
        raise SystemExit(f"{path}: was trained on data fingerprint {blob.get('fingerprint')}, "
                         f"this run has {fingerprint}. Refusing to resume onto other data.")
    if blob.get("tag") != tag or int(blob.get("block", 0)) != int(block):
        raise SystemExit(f"{path}: state is {blob.get('tag')}/block {blob.get('block')}, "
                         f"this configuration is {tag}/block {block}")
    for k, v in blob["weights"].items():
        if k in w:
            w[k].detach().copy_(v.to(w[k].device).to(w[k].dtype))
    if opt is not None:
        opt.load_state_dict(_opt_to(blob["opt"], torch.float32))
    rng.setstate(blob["rng_py"])
    torch.set_rng_state(blob["rng_torch"].to(torch.uint8))
    return blob


def parse_probe(spec: str, a, cfg) -> tuple[str, str, float, int]:
    """`tag=train:lr:block` -> (tag, train, lr, block). Every field but the first is optional.

    The block length being per configuration is the whole point: the data load is what costs
    minutes on the full recording, and `b8=all:1.5e-4:8,b16=all:1.5e-4:16` pays it once for both
    block lengths instead of once each. An empty field falls back to the flag of the same name, so
    `all` on its own still means what it meant when this only compared learning rates.
    """
    spec = spec.strip()
    tag, sep, rest = spec.partition("=")
    if not sep:
        rest, tag = tag, ""
    parts = (rest.split(":") + ["", "", ""])[:3]
    mode = parts[0].strip() or a.train
    lr = float(parts[1]) if parts[1].strip() else a.lr
    blk = int(parts[2]) if parts[2].strip() else (a.block or cfg.block_size)
    if blk < 2:
        raise SystemExit(f"--probe {spec!r}: a block of {blk} drafts nothing")
    tag = tag.strip() or f"{mode}-lr{lr:g}-b{blk}"
    if os.sep in tag or tag in (".", ".."):
        raise SystemExit(f"--probe {spec!r}: {tag!r} is not usable as a directory name")
    return tag, mode, lr, blk


def evaluate(m, embed, head, held, dev, a, blk) -> dict:
    """The in-loop gate: the chain walk (`acceptance`), or the served tree with --eval-tree."""
    spec = str(getattr(a, "eval_tree", "0") or "0")
    if spec == "0":
        return acceptance(m, embed, head, held, dev, max_blocks=a.eval_blocks, block=blk)
    bs = blk or m.cfg.block_size
    budget = (16 if bs <= 8 else 32) if spec == "auto" else int(spec)
    return tree_acceptance(m, embed, head, held, dev, budget, block=bs, max_blocks=a.eval_blocks)


def arm(w: dict, cfg, mode: str) -> list[torch.Tensor]:
    out = []
    for k in select_params(w, cfg, mode):
        if w[k].is_floating_point():
            w[k].requires_grad_(True)
            out.append(w[k])
    return out


class Optim:
    """AdamW over the dense tensors (backbone at `lr`, the selector's projection at `sel_lr`) and,
    when the codebooks train, SparseAdam over their gathered rows at `sel_lr`.

    One object so the training loop has one `step`, one `zero_grad` and one learning-rate schedule:
    every group keeps its own base rate and the schedule scales them all by the same factor.
    SparseAdam is the lazy Adam: a row's moments and value change only on a step that gathered it,
    so a row no batch ever touched leaves training bit for bit as it came in."""

    def __init__(self, w: dict, params: list[torch.Tensor], lr: float, sel_lr: float, wd: float):
        cb = {id(w[k]) for k in (PRED_CB, SUCC_CB) if k in w}
        sel = {id(w[HPROJ])} if HPROJ in w else set()
        dense = [p for p in params if id(p) not in cb and id(p) not in sel]
        sel_p = [p for p in params if id(p) in sel]
        self.sparse_params = [p for p in params if id(p) in cb]
        groups = [{"params": dense, "lr": lr, "base_lr": lr}] if dense else []
        if sel_p:
            groups.append({"params": sel_p, "lr": sel_lr, "base_lr": sel_lr})
        self.dense = torch.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(0.9, 0.95), eps=1e-8)
        self.sparse = (torch.optim.SparseAdam(self.sparse_params, lr=sel_lr, betas=(0.9, 0.95),
                                              eps=1e-8) if self.sparse_params else None)
        if self.sparse is not None:
            for g in self.sparse.param_groups:
                g["base_lr"] = sel_lr
        self.dense_params = dense + sel_p
        self.param_groups = self.dense.param_groups + (self.sparse.param_groups
                                                       if self.sparse is not None else [])

    def set_scale(self, f: float) -> None:
        for g in self.param_groups:
            g["lr"] = g["base_lr"] * f

    def clip(self, max_norm: float) -> float:
        """Clip the dense and the sparse gradients to `max_norm` each; returns the dense norm."""
        gn = torch.nn.utils.clip_grad_norm_(self.dense_params, max_norm)
        for p in self.sparse_params:
            if p.grad is None:
                continue
            p.grad = p.grad.coalesce()
            n = p.grad.values().norm()
            if float(n) > max_norm:
                p.grad = torch.sparse_coo_tensor(p.grad.indices(), p.grad.values() * (max_norm / n),
                                                 p.grad.shape)
        return float(gn)

    def step(self) -> None:
        self.dense.step()
        if self.sparse is not None:
            self.sparse.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.dense.zero_grad(set_to_none=set_to_none)
        if self.sparse is not None:
            self.sparse.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        if self.sparse is not None:
            raise SystemExit("--state-dir is not supported with a sparse (codebook) optimiser")
        return self.dense.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        if self.sparse is not None:
            raise SystemExit("--resume is not supported with a sparse (codebook) optimiser")
        self.dense.load_state_dict(sd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.expanduser("~/qwen38-spark-engine/train/data"))
    ap.add_argument("--ckpt", default=None, help="the drafter to start from")
    ap.add_argument("--model", default=None, help="the target snapshot, for embed and head")
    ap.add_argument("--out", default=os.path.expanduser("~/qwen38-spark-engine/train/ft"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--block", type=int, default=0,
                    help="the drafter's BLOCK LENGTH, 0 = the checkpoint's own (8). A block of 16 "
                         "is sixteen rows, fifteen of them masked, at positions p..p+15; nothing "
                         "in the module is tied to eight, only its training is. The exported "
                         "config.json records the length it was trained at")
    ap.add_argument("--blocks", type=int, default=32, help="blocks packed into one pass. The "
                    "optimiser costs the same whatever the batch is -- 1.8 B parameters read and "
                    "written five times -- so the batch is what amortises it")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--kl", type=float, default=0.5, help="weight of the soft term")
    ap.add_argument("--train", default="all", help="all | fc | lastN (e.g. last2)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-blocks", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-ckpts", default=None,
                    help="with --eval-only, score several drafter checkpoints in one process. "
                         "`base` is the released one; the rest are directories this trainer wrote")
    ap.add_argument("--heldout", default=None,
                    help="override the manifest's split with this comma-separated list of sequence "
                         "names. The reason it exists is in the ledger at 15:20: a prompt template "
                         "with no variable part produces IDENTICAL prompts, so its four instances "
                         "are four copies of one greedy generation, and a split by template put "
                         "copies of the same sequence on both sides of it")
    ap.add_argument("--keep-on", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gen-weight", type=float, default=0.5,
                    help="share of steps drawn from the self-generated sequences. They are the "
                         "serving distribution -- the drafter conditions on hidden states of text "
                         "the target itself wrote -- and the corpus half is there to keep the "
                         "drafter from narrowing onto a few hundred generations")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget-min", type=float, default=0.0)
    ap.add_argument("--log", default=None, help="jsonl of every logged step")
    ap.add_argument("--probe", default=None,
                    help="several configurations in ONE process, comma-separated, each from the "
                         "released weights again, each for --steps. One model load, ONE DATA LOAD, "
                         "one allocation history -- which is what makes the acceptance numbers "
                         "comparable to each other, and what makes the 109 GB affordable. A "
                         "configuration is `tag=train:lr:block`, and everything but the first "
                         "field may be left out: `all`, `all:3e-5`, `b16=all:1.5e-4:16`. The "
                         "block length is per configuration, so `b8=all:1.5e-4:8,b16=all:1.5e-4:16` "
                         "trains both block lengths off one load of the data")
    ap.add_argument("--state-dir", default=None,
                    help="where the RESUME state goes: weights, the two Adam moments, the step "
                         "(which is the whole learning-rate schedule, because the rate is a closed "
                         "form of it) and the random state (which is the whole data cursor, "
                         "because every sequence and every anchor is drawn from one Random). Point "
                         "it at storage that outlives the job -- a Storage Bucket -- or it buys "
                         "nothing. A run of 35,500 steps is three hours; losing it to a cancel is "
                         "$8.35 and an evening")
    ap.add_argument("--state-every", type=int, default=0,
                    help="write the resume state every N steps. 0 = never. This is a FLOOR: the "
                         "cadence backs off on its own if the write turns out to cost more than "
                         "--state-max-overhead of the training time, which is the only honest way "
                         "to set it when nobody has measured what the storage writes at")
    ap.add_argument("--state-max-overhead", type=float, default=0.05,
                    help="the share of wall clock the resume state is allowed to cost. The bucket "
                         "mount wrote 109 GB of small files at 53 MB/s, and at that rate an 11.5 GB "
                         "state is nearly four minutes -- more than the thousand steps it is "
                         "protecting. So the cadence is measured rather than chosen")
    ap.add_argument("--state-dtype", default="bf16", choices=("bf16", "fp32"),
                    help="bf16 halves the two Adam moments from 15.2 GB to 7.6 and keeps fp32's "
                         "exponent range, so the small second moments lose mantissa rather than "
                         "underflowing. On a mount that writes at 53 MB/s that is four minutes a "
                         "checkpoint instead of eight")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the state in --state-dir. A configuration whose .done "
                         "marker is there is skipped, so a job that died during b16 does not "
                         "retrain b8")
    ap.add_argument("--probe-out", default=None,
                    help="with --probe, export each configuration to <dir>/<tag>. Without it a "
                         "probe measures and throws the weights away, which is what it was for "
                         "when it only compared learning rates")
    ap.add_argument("--rank-loss", type=float, default=0.0,
                    help="RND-37: weight of the lattice ranking term (`rank_loss_of`), added to the "
                         "CE + KL. 0 = the objective as before, bit for bit")
    ap.add_argument("--selector-lr", type=float, default=1e-4,
                    help="learning rate of the selector's projection and codebooks (a `rank*` "
                         "--train mode); the rest trains at --lr")
    ap.add_argument("--eval-tree", default="0",
                    help="in-loop eval on the served TREE instead of the chain: a node budget, or "
                         "`auto` = 16 at block 8 and 32 at block 16 (the served cuts). `best` and "
                         "the checkpoint choice follow its ALL (thinking-on sequences excluded)")
    ap.add_argument("--export-final", action="store_true",
                    help="with --probe-out, also write every configuration's FINAL weights to "
                         "<dir>/<tag>-final, whether or not the in-loop eval improved: the gate "
                         "that decides is the Spark's replay, not this eval")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = a.device

    cfg, snap = load_config(a.ckpt)
    base = load_weights(snap, device="cpu")
    w = {k: v.detach().clone().to(dev).float() if v.is_floating_point() else v.to(dev)
         for k, v in base.items()}
    m = DFlash2Module(cfg, w)

    from engine.config import load_config as load_target
    tcfg = load_target(a.model)
    embed, head = target_tensors(tcfg.path, dev)

    data = load_data(a.data, dev, a.keep_on, a.limit)
    if a.heldout:
        want = set(a.heldout.split(","))
        for s_ in data:
            s_.split = "heldout" if s_.name in want else "train"
    train = [s for s in data if s.split == "train"]
    held = [s for s in data if s.split == "heldout"]
    if not held:
        raise SystemExit("no held-out sequences in the manifest")
    train_gen = [s for s in train if s.kind == "gen"]
    train_corp = [s for s in train if s.kind != "gen"]
    fingerprint = data_fingerprint(train_gen, train_corp, held)
    print(f"data fingerprint {fingerprint}", flush=True)
    print(f"drafter {snap}\n{len(train)} training sequences "
          f"({sum(len(s) for s in train)} positions, {len(train_gen)} self-generated / "
          f"{len(train_corp)} corpus), {len(held)} held out", flush=True)

    params = arm(w, cfg, a.train)
    print(f"--train {a.train}: {len(params)} tensors, "
          f"{sum(p.numel() for p in params)/1e9:.3f} B parameters", flush=True)

    if a.probe and not a.eval_only:
        ev = {}                       # every probe configuration measures its own base
    else:
        ev = evaluate(m, embed, head, held, dev, a, a.block)
        print("accept/block before: " + " ".join(f"{k}={v:.3f}" for k, v in sorted(ev.items())),
              flush=True)
    if a.eval_only:
        if a.eval_ckpts:
            for spec in a.eval_ckpts.split(","):
                if spec not in ("base", ""):
                    fresh = load_weights(resolve_checkpoint(os.path.expanduser(spec)),
                                         device="cpu")
                    for k, v in fresh.items():
                        w[k].detach().copy_(v.to(w[k].device).to(w[k].dtype))
                ev = evaluate(m, embed, head, held, dev, a, a.block)
                print(f"{spec:44s} " + " ".join(f"{k}={v:.3f}"
                                                for k, v in sorted(ev.items())), flush=True)
        return

    if a.probe:
        base_cpu = {k: v.detach().to("cpu", copy=True) for k, v in w.items()}
        rank0, sel0 = a.rank_loss, a.selector_lr
        for spec in a.probe.split(","):
            tag, mode, lr, blk = parse_probe(spec, a, cfg)
            # `rank@0.5` in the train field: this configuration's --rank-loss (RND-37 sweeps the
            # ranking weight inside one data load)
            a.rank_loss = rank0
            a.selector_lr = sel0
            if "@" in mode:
                parts = mode.split("@")
                mode, a.rank_loss = parts[0], float(parts[1])
                if len(parts) > 2 and parts[2]:
                    a.selector_lr = float(parts[2])     # `rank@1@1e-3`: this config's --selector-lr
            if a.resume and a.state_dir and os.path.exists(done_path(a.state_dir, tag)):
                print(f"\n--- probe {tag}: already finished (its .done marker is in "
                      f"{a.state_dir}), skipping ---", flush=True)
                continue
            # One log per configuration, named after it and sitting beside its checkpoint, because
            # that is where the export step looks for the curve it packs into MANIFEST.json.
            # A single shared log would give both checkpoints the same, last-written numbers.
            logp = os.path.join(a.probe_out, f"{tag}.jsonl") if a.probe_out else a.log
            if logp:
                os.makedirs(os.path.dirname(os.path.abspath(logp)) or ".", exist_ok=True)
            logf = open(logp, "a") if logp else None
            for k, v in base_cpu.items():
                w[k].detach().copy_(v.to(w[k].device))
                w[k].requires_grad_(False)
            pp = arm(w, cfg, mode)
            out = os.path.join(a.probe_out, tag) if a.probe_out else None
            start, opt = 0, None
            sp = state_path(a.state_dir, tag) if a.state_dir else None
            if a.resume and sp and os.path.exists(sp):
                if mode.startswith("rank") or a.rank_loss > 0:
                    raise SystemExit("--resume is not supported for the ranking runs")
                opt = torch.optim.AdamW(pp, lr=lr, weight_decay=a.wd, betas=(0.9, 0.95), eps=1e-8)
                blob = load_state(sp, w=w, opt=opt, rng=rng, fingerprint=fingerprint,
                                  tag=tag, block=blk)
                start = int(blob["step"])
                print(f"  [resume] {sp}: step {start}/{blob['steps_total']}, "
                      f"best {blob['best']:.3f}, {blob['elapsed']/60:.0f} min already paid",
                      flush=True)
            print(f"\n--- probe {tag}: --train {mode} lr {lr:.1e} block {blk}, "
                  f"{sum(p.numel() for p in pp)/1e9:.3f} B parameters, {a.steps} steps "
                  f"{'-> ' + out if out else '(not exported)'} ---", flush=True)
            # The base acceptance has to be re-measured at THIS configuration's block length. A
            # released drafter accepts a different number at 8 and at 16, and `best` is what
            # decides whether a checkpoint is written at all.
            ev0 = evaluate(m, embed, head, held, dev, a, blk)
            print("  accept/block " + ("at resume:   " if start else "before: ")
                  + " ".join(f"{k}={v:.3f}" for k, v in sorted(ev0.items())), flush=True)
            if logf:
                logf.write(json.dumps({"probe": tag, "block": blk, "before": ev0}) + "\n")
                logf.flush()
            best0 = float(blob["best"]) if start else ev0["ALL"]
            best = train_loop(m, w, pp, embed, head, train_gen, train_corp, held, a, dev, lr, rng,
                              cfg=cfg, block=blk, snap=snap, out=out, best=best0, logf=logf,
                              tag=tag, start_step=start, state_dir=a.state_dir,
                              fingerprint=fingerprint, mode=mode, opt=opt)
            ev = evaluate(m, embed, head, held, dev, a, blk)
            print(f"  probe {tag} -> "
                  + " ".join(f"{k}={v:.3f}" for k, v in sorted(ev.items())), flush=True)
            if logf:
                logf.write(json.dumps({"probe": tag, "block": blk, "final": ev}) + "\n")
                logf.flush()
                logf.close()
                logf = None
            # The checkpoint worth keeping is the best one the gate saw, and the last step is not
            # always it -- so the final weights are written only if they beat every eval before.
            if out and a.export_final:
                export(w, snap, out + "-final", block=blk)
                print(f"  [saved] {out}-final at {ev['ALL']:.3f} (final weights, --export-final)",
                      flush=True)
            if out and ev["ALL"] > best:
                export(w, snap, out, block=blk)
                print(f"  [saved] {out} at {ev['ALL']:.3f} accepted/block (final)", flush=True)
            elif out and best <= best0:
                print(f"  NOTE: {tag} never beat {best0:.3f}; nothing was written to {out}",
                      flush=True)
            if a.state_dir:
                # The marker is what makes a resumed job skip a configuration it has finished.
                # It is written after the checkpoint, so a crash between the two costs a re-run
                # of this configuration and never a missing drafter.
                os.makedirs(a.state_dir, exist_ok=True)
                with open(done_path(a.state_dir, tag), "w") as f:
                    f.write(json.dumps({"tag": tag, "block": blk, "final": ev,
                                        "fingerprint": fingerprint}) + "\n")
        return

    logf = open(a.log, "a") if a.log else None
    tag = os.path.basename(a.out.rstrip("/")) or "run"
    start, opt, best0 = 0, None, ev["ALL"]
    sp = state_path(a.state_dir, tag) if a.state_dir else None
    if a.resume and sp and os.path.exists(sp):
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95), eps=1e-8)
        blob = load_state(sp, w=w, opt=opt, rng=rng, fingerprint=fingerprint, tag=tag,
                          block=a.block or cfg.block_size)
        start, best0 = int(blob["step"]), float(blob["best"])
        print(f"[resume] {sp}: step {start}/{blob['steps_total']}, best {best0:.3f}", flush=True)
    train_loop(m, w, params, embed, head, train_gen, train_corp, held, a, dev, a.lr, rng,
               snap=snap, out=a.out, best=best0, logf=logf, cfg=cfg, tag=tag,
               start_step=start, state_dir=a.state_dir, fingerprint=fingerprint, mode=a.train,
               opt=opt, block=a.block or cfg.block_size)
    ev = evaluate(m, embed, head, held, dev, a, a.block)
    print("accept/block after:  " + " ".join(f"{k}={v:.3f}" for k, v in sorted(ev.items())),
          flush=True)
    export(w, snap, a.out, block=a.block)
    print(f"[saved] {a.out} at {ev['ALL']:.3f} accepted/block")
    if logf:
        logf.write(json.dumps({"final": ev}) + "\n")
        logf.close()


def train_loop(m, w, params, embed, head, train_gen, train_corp, held, a, dev, lr, rng,
               *, snap=None, out=None, best=0.0, logf=None, cfg=None, block=None, tag=None,
               start_step=0, state_dir=None, fingerprint=None, mode=None, opt=None):
    """One run of the objective. Separated out so `--probe` can do several in one process.

    `block` overrides `--block` for this run and nothing else, which is what lets one process --
    and therefore one load of the recorded tensors -- train both block lengths.

    Returns the best held-out acceptance it saw, so the caller knows whether the weights it is
    holding are better than the checkpoint already on disk.
    """
    cfg = cfg or m.cfg
    blk = a.block if block is None else block
    rank_run = (mode or a.train).startswith("rank") or getattr(a, "rank_loss", 0.0) > 0
    if opt is None:
        if rank_run:
            opt = Optim(w, params, lr, a.selector_lr, a.wd)
        else:
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=a.wd, betas=(0.9, 0.95), eps=1e-8)
    rank_n = rank_t = 0
    t0 = time.perf_counter()
    t_eval = 0.0
    t_state = 0.0
    state_every = a.state_every
    bs = blk or cfg.block_size
    hist = []
    step = start_step
    for step in range(start_step + 1, a.steps + 1):
        f_lr = min(1.0, step / max(1, a.warmup)) * \
            (0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))) * 0.9 + 0.1)
        if isinstance(opt, Optim):
            opt.set_scale(f_lr)
        else:
            for g in opt.param_groups:
                g["lr"] = lr * f_lr
        pool = train_gen if (train_gen and train_corp and rng.random() < a.gen_weight) \
            else (train_corp or train_gen)
        s = pool[rng.randrange(len(pool))]
        n = len(s)
        lo = max(1, s.gen_start - 1) if s.kind == "gen" else 1
        hi = n - bs
        if hi <= lo:
            continue
        anchors = torch.tensor(sorted(rng.sample(range(lo, hi + 1),
                                                 min(a.blocks, hi - lo + 1))),
                               dtype=torch.long, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            pred = run_blocks(m, embed, s, anchors, dev, block=blk)
            if a.rank_loss > 0:
                logits = F.linear(pred.reshape(-1, pred.shape[-1]), head).float()
                loss, hits = loss_of(pred, head, s, anchors, a.kl, dev, logits=logits)
                rl, rn, rt = rank_loss_of(m, pred, logits, s, anchors, dev)
                loss = loss + a.rank_loss * rl
                rank_n += rn
                rank_t += rt
            else:
                loss, hits = loss_of(pred, head, s, anchors, a.kl, dev)
        loss.backward()
        gn = opt.clip(a.clip) if isinstance(opt, Optim) else \
            torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        acc = float(hits) / (anchors.numel() * (bs - 1))
        hist.append((float(loss.detach()), acc))
        if step % 25 == 0 or step == 1:
            k = hist[-25:]
            msg = ((f"[{tag}] " if tag else "")
                   + f"step {step:5d}  loss {sum(x[0] for x in k)/len(k):.4f}  "
                   f"row-hit {sum(x[1] for x in k)/len(k):.3f}  gn {float(gn):.2f}  "
                   f"lr {opt.param_groups[0]['lr']:.2e}  "
                   + (f"rank-top1 {rank_t / max(rank_n, 1):.3f} ({rank_n})  " if rank_n else "")
                   + f"{(time.perf_counter()-t0-t_eval-t_state)/max(1, step-start_step)*1000:.0f}"
                   " ms/step")
            print(msg, flush=True)
            if logf:
                logf.write(json.dumps({"tag": tag, "step": step,
                                       "loss": sum(x[0] for x in k)/len(k),
                                       "row_hit": sum(x[1] for x in k)/len(k),
                                       "rank_top1": rank_t / max(rank_n, 1)}) + "\n")
                logf.flush()
            rank_n = rank_t = 0
        if a.eval_every and step % a.eval_every == 0:
            t_ev = time.perf_counter()
            ev = evaluate(m, embed, head, held, dev, a, blk)
            t_eval += time.perf_counter() - t_ev
            print(f"  {'[' + tag + '] ' if tag else ''}[eval @{step}] "
                  + " ".join(f"{k}={v:.3f}" for k, v in sorted(ev.items())), flush=True)
            if logf:
                logf.write(json.dumps({"tag": tag, "step": step, "eval": ev}) + "\n")
                logf.flush()
            if ev["ALL"] > best:
                best = ev["ALL"]
                if out:
                    export(w, snap, out, block=blk)
                    print(f"  [saved] {out} at {best:.3f} accepted/block", flush=True)
        if state_dir and state_every and step % state_every == 0:
            t_st = time.perf_counter()
            n = save_state(state_path(state_dir, tag or "run"), w=w, opt=opt, step=step,
                           best=best, rng=rng, tag=tag or "run", block=blk, lr=lr,
                           mode=mode or a.train, steps_total=a.steps, fingerprint=fingerprint,
                           elapsed=time.perf_counter() - t0,
                           dtype=torch.bfloat16 if a.state_dtype == "bf16" else torch.float32)
            dt = time.perf_counter() - t_st
            t_state += dt
            # What a checkpoint costs is a property of the storage and nobody has measured it here,
            # so measure it and back off. Below the floor the cadence never goes; above it, the
            # write is allowed to be at most `--state-max-overhead` of the steps it protects.
            per_step = (time.perf_counter() - t0 - t_eval - t_state) / max(1, step - start_step)
            want = math.ceil(dt / max(per_step * max(a.state_max_overhead, 1e-3), 1e-9))
            new_every = max(a.state_every, int(round(want / a.state_every)) * a.state_every)
            note = "" if new_every == state_every else f", cadence -> every {new_every} steps"
            state_every = new_every
            print(f"  [state @{step}] {n/2**30:.1f} GB in {dt:.0f} s "
                  f"({n/2**20/max(dt,1e-9):.0f} MB/s) -> {state_path(state_dir, tag or 'run')}"
                  f"{note}", flush=True)

        if a.budget_min and (time.perf_counter() - t0) / 60 > a.budget_min:
            print(f"[budget] stopping at step {step}", flush=True)
            break
    if state_dir and a.state_every and step > start_step and step % max(state_every, 1) != 0:
        # The last steps since the previous checkpoint are the ones a resume would otherwise repeat.
        save_state(state_path(state_dir, tag or "run"), w=w, opt=opt, step=step, best=best,
                   rng=rng, tag=tag or "run", block=blk, lr=lr, mode=mode or a.train,
                   steps_total=a.steps, fingerprint=fingerprint,
                   elapsed=time.perf_counter() - t0,
                   dtype=torch.bfloat16 if a.state_dtype == "bf16" else torch.float32)
        print(f"  [state @{step}] final", flush=True)
    for p_ in params:
        p_.grad = None
    del opt
    torch.cuda.empty_cache()
    return best


if __name__ == "__main__":
    main()
