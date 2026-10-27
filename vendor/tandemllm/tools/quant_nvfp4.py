"""Quantise the projections of every layer to NVFP4, from the BF16 release or the FP8 one.

69.7 % of the bytes a decode step reads are `gate_proj`, `up_proj` and `down_proj`. At FP8 they are
267.4 MB per layer; at NVFP4 they are 150.4 MB, and the step's byte floor falls from 98.6 ms to
71.2 ms. `--targets` adds the linear-attention and attention projections; `lm_head` has its own
tool (`tools/quant_head.py`).

The source is whatever checkpoint `--model` names. The BF16 release is the one to quantise from:
its weights are rounded once, to NVFP4. The FP8 release has already been rounded to e4m3 on a
128x128 block grid, so an NVFP4 copy of it is a second rounding of a first. Both are read through
`Source`, which finds each projection by the checkpoint's own index and returns the weight the
checkpoint defines: the bf16 matrix as stored, or `code * scale[n // 128, k // 128]`.

Three ways to choose the codes, because the choice is what the quality gate measures:

  * `rtn` -- per group of 16, the scale is `amax / 6`, rounded to e4m3 and rounded up where rounding
    down would clip. No data, no calibration, one pass over the weights.
  * `clip` -- the same, then a short search over scale multipliers per group, minimising the error
    the *model* sees rather than the error the weight sees: each input channel is weighted by its
    mean square activation over a calibration corpus, so a channel the model never excites is
    allowed to be approximated worse than one it leans on. The statistics come from `stats`.
  * `gptq` -- the clip search chooses each group's scale, and the codes are chosen one input
    column at a time with the rounding error of each column pushed onto the columns not yet
    rounded, weighted by the full second moment of the projection's input (`H = X^T X`, not only
    its diagonal). Same format, same bytes, same kernel; only which codes are stored changes.

The two levels of scale are not interchangeable with one wider level. A 128x128 fp8 block covers
16,384 weights with a single scale; an NVFP4 group covers 16. That is the entire reason four bits
are usable here, and it is why the format carries the per-tensor `weight_scale_2` as well: the e4m3
group scales themselves need a scale to sit inside e4m3's range.

Usage:

    python tools/quant_nvfp4.py stats --out ~/nvfp4/stats.pt --corpus bench/calib.txt
    python tools/quant_nvfp4.py quant --mode rtn  --out ~/nvfp4/mlp-rtn.safetensors
    python tools/quant_nvfp4.py quant --mode clip --stats ~/nvfp4/stats.pt \
                                      --out ~/nvfp4/mlp-clip.safetensors
    # clip and gptq for every target in one process, statistics included (GPU, BF16 source):
    python tools/quant_nvfp4.py build --model <bf16 snapshot> --corpus code.txt:65536,prose.txt:65536 \
                                      --methods clip,gptq --out-dir ~/nvfp4-bf16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import resolve_snapshot  # noqa: E402
from tools.fp8_linear import BLOCK as FP8_BLOCK  # noqa: E402
from tools.nvfp4_linear import (E2M1_MAX, E4M3_MAX, FP4_GRID, GROUP, NVFP4Block,  # noqa: E402
                                _round_e2m1, quantize_to_nvfp4)

MLP_PROJ = ("gate_proj", "up_proj", "down_proj")
LM_PREFIX = "model.language_model."
CLIP_RATIOS = (1.0, 0.95, 0.90, 0.85, 0.80)

# Which projections each target quantises. The exclusions are as load-bearing as the inclusions:
#   * `conv1d`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias` and every norm stay bf16. The two small
#     `a`/`b` projections are 48x[48, 5120] -- 0.17 % of the bytes -- and every published recipe
#     that quantises this family keeps them in full precision by name.
#   * the recurrent STATE is fp32 and is not a weight. It is not touched here and must not be:
#     a bf16 state costs 5.8 AIME points and an fp8 one 56.
#   * the drafter's own weights stay bf16. It proposes; the verify pass decides.
TARGETS: dict[str, tuple[str, ...]] = {
    "mlp": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    "gdn": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"),
    "attn": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
}
TARGETS["all"] = TARGETS["mlp"] + TARGETS["gdn"] + TARGETS["attn"]

# The smallest pieces a mix of NVFP4 and FP8 can choose between. Each is a whole fusion group of
# `engine.loader.PROJ_GROUPS` (or a projection outside every group), so a mix never splits a group
# into two formats and never costs the fused launch.
UNITS: dict[str, tuple[str, ...]] = {
    "mlp.gate_up": ("mlp.gate_proj", "mlp.up_proj"),
    "mlp.down": ("mlp.down_proj",),
    "gdn.qkvz": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
    "gdn.out": ("linear_attn.out_proj",),
    "attn.qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "attn.o": ("self_attn.o_proj",),
}

# Projections that read the same activation share one input statistic: the first of each tuple.
SHARED_INPUT = {m: members[0] for members in (UNITS["mlp.gate_up"], UNITS["gdn.qkvz"],
                                              UNITS["attn.qkv"]) for m in members}


def target_of(proj: str) -> str:
    """`mlp.down_proj` -> `mlp`, the output file a projection belongs to."""
    for t in ("mlp", "gdn", "attn"):
        if proj in TARGETS[t]:
            return t
    raise KeyError(proj)


def input_key(base: str) -> str:
    """`layers.5.mlp.up_proj` -> `layers.5.mlp.gate_proj`: the projection whose input statistic it shares."""
    layer, proj = base.split(".", 2)[1], base.split(".", 2)[2]
    return f"layers.{layer}.{SHARED_INPUT.get(proj, proj)}"


# ------------------------------------------------------------------ reading a checkpoint
def fp8_dequant(codes: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """The FP8 checkpoint's own definition: `code * scale[n // 128, k // 128]`."""
    s = scale_inv.float()
    s = s.repeat_interleave(FP8_BLOCK, 0)[: codes.shape[0]]
    s = s.repeat_interleave(FP8_BLOCK, 1)[:, : codes.shape[1]]
    return codes.to(torch.float32) * s


def _canonical(key: str) -> str | None:
    from engine.loader import Layout
    return Layout.canonical(key)


class Source:
    """Where every quantisable projection of a checkpoint is stored, and its weight as fp32.

    The FP8 release keeps one layer per file and every projection as an e4m3 pair; the BF16 release
    keeps several layers per shard and every projection as one bf16 matrix. Both are found through
    `model.safetensors.index.json` when there is one, else by listing every file. Keys outside the
    language model -- the vision tower, the MTP layer -- are not projections of the 64 layers and
    are not found.
    """

    def __init__(self, snapshot: str):
        self.snapshot = snapshot
        idx = os.path.join(snapshot, "model.safetensors.index.json")
        if os.path.isfile(idx):
            with open(idx) as f:
                wm = json.load(f)["weight_map"]
        else:
            wm = {}
            for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
                with safe_open(path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        wm[k] = os.path.basename(path)
        self.where: dict[str, tuple[str, str, str | None]] = {}
        for key, fname in wm.items():
            name = _canonical(key)
            if name is None or not name.startswith("layers.") or not name.endswith(".weight"):
                continue
            base = name[: -len(".weight")]
            if base.split(".", 2)[2] not in TARGETS["all"]:
                continue
            skey = key[: -len(".weight")] + ".weight_scale_inv"
            self.where[base] = (os.path.join(snapshot, fname), key, skey if skey in wm else None)
        kinds = {s is not None for _, _, s in self.where.values()}
        self.kind = "fp8" if kinds == {True} else "bf16" if kinds == {False} else "mixed"

    def layers(self) -> list[int]:
        return sorted({int(b.split(".")[1]) for b in self.where})

    def read(self, base: str, device: str = "cpu") -> torch.Tensor | None:
        """The fp32 weight of `layers.N.<proj>`, or None where the layer has no such projection
        (48 of the 64 layers are Gated DeltaNet and have no `self_attn`, the other 16 no
        `linear_attn`)."""
        hit = self.where.get(base)
        if hit is None:
            return None
        path, key, skey = hit
        with safe_open(path, framework="pt", device=device) as f:
            w = f.get_tensor(key)
            if skey is not None:
                return fp8_dequant(w, f.get_tensor(skey))
        if w.dtype == torch.float8_e4m3fn:
            raise RuntimeError(f"{key}: e4m3 codes without a {skey or 'scale'} tensor")
        return w.float()

    def stored_bytes(self, base: str) -> int:
        """The bytes this projection costs as the FP8 release stores it (codes + bf16 block scales),
        whatever this checkpoint is: it is what a mix keeps in place of NVFP4."""
        path, key, _ = self.where[base]
        with safe_open(path, framework="pt", device="cpu") as f:
            N, K = f.get_slice(key).get_shape()
        return N * K + -(-N // FP8_BLOCK) * -(-K // FP8_BLOCK) * 2


def nvfp4_bytes(N: int, K: int) -> int:
    return N * K // 2 + N * K // GROUP


# ------------------------------------------------------------------ the clip search
def quantize_clipped(ref: torch.Tensor, act_ms: torch.Tensor | None,
                     ratios=CLIP_RATIOS, rows: int = 1024) -> NVFP4Block:
    """Per group of 16, pick the scale multiplier that minimises the activation-weighted error.

    `act_ms[k]` is the mean square of input channel `k` over the calibration corpus. The quantity
    minimised is `sum_k act_ms[k] * (w_k - dequant(q(w_k)))^2`, which is the contribution of this
    group to the squared error of the projection's output, to first order and under the assumption
    that the input channels are uncorrelated. Weighting by activation is the whole point: a channel
    the model never excites can be approximated badly for free.

    A smaller multiplier clips the group's outlier and represents the rest of the group more
    finely. Which side wins is data-dependent, so it is searched rather than argued.
    """
    N, K = ref.shape
    dev = ref.device
    amax_t = ref.abs().max().float()
    scale_2 = float((amax_t / (E2M1_MAX * E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny))
    if act_ms is None:
        act_ms = torch.ones(K, device=dev, dtype=torch.float32)
    w_k = act_ms.to(dev).float().reshape(1, K // GROUP, GROUP)
    codes = torch.empty(N, K // 2, dtype=torch.uint8, device=dev)
    scales = torch.empty(N, K // GROUP, dtype=torch.float8_e4m3fn, device=dev)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        x = ref[r0:r1].float().reshape(r1 - r0, K // GROUP, GROUP)
        gmax = x.abs().amax(dim=2, keepdim=True)
        best_err = None
        best_s8 = None
        for ratio in ratios:
            s = (gmax * ratio / E2M1_MAX) / scale_2
            s8 = s.clamp(0.0, E4M3_MAX).to(torch.float8_e4m3fn)
            eff = (s8.float() * scale_2).clamp_min(torch.finfo(torch.float32).tiny)
            q = _round_e2m1((x / eff).abs().clamp(0.0, E2M1_MAX)).to(torch.int64)
            grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
            deq = grid[q] * torch.sign(x) * eff
            err = (w_k * (x - deq).pow(2)).sum(dim=2, keepdim=True)
            if best_err is None:
                best_err, best_s8 = err, s8.float()
            else:
                take = err < best_err
                best_err = torch.where(take, err, best_err)
                best_s8 = torch.where(take, s8.float(), best_s8)
            del s, s8, eff, q, deq, err
        s8 = best_s8.to(torch.float8_e4m3fn)
        eff = (s8.float() * scale_2).clamp_min(torch.finfo(torch.float32).tiny)
        v = x / eff
        idx = _round_e2m1(v.abs().clamp(0.0, E2M1_MAX))
        nib = (idx | ((v < 0).to(torch.uint8) * 8)).reshape(r1 - r0, K)
        codes[r0:r1] = nib[:, 0::2] | (nib[:, 1::2] << 4)
        scales[r0:r1] = s8.reshape(r1 - r0, K // GROUP)
        del x, gmax, best_err, best_s8, s8, eff, v, idx, nib
    return NVFP4Block(codes, scales, scale_2)


def _group_scale(x: torch.Tensor, w_k: torch.Tensor, scale_2: float, ratios) -> torch.Tensor:
    """The clip search for ONE group of 16 columns across all rows: x [N, 16], w_k [16] ->
    the chosen e4m3 scale as float [N, 1]. The same search and the same error as
    `quantize_clipped`, on the weights as they stand when GPTQ reaches the group."""
    grid = FP4_GRID.to(x.device)
    gmax = x.abs().amax(dim=1, keepdim=True)
    best_err, best_s8 = None, None
    for ratio in ratios:
        s8 = ((gmax * ratio / E2M1_MAX) / scale_2).clamp(0.0, E4M3_MAX).to(torch.float8_e4m3fn).float()
        eff = (s8 * scale_2).clamp_min(torch.finfo(torch.float32).tiny)
        q = _round_e2m1((x / eff).abs().clamp(0.0, E2M1_MAX)).long()
        err = (w_k[None, :] * (x - grid[q] * torch.sign(x) * eff).pow(2)).sum(dim=1, keepdim=True)
        if best_err is None:
            best_err, best_s8 = err, s8
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_s8 = torch.where(take, s8, best_s8)
    return best_s8


def gptq_nvfp4(W: torch.Tensor, H: torch.Tensor, *, ratios=CLIP_RATIOS, damp: float = 0.01,
               block: int = 128) -> NVFP4Block:
    """GPTQ onto the NVFP4 grid: W [N, K] (any float), H [K, K] = sum of x x^T over calibration.

    Columns are rounded in their stored order (a group of 16 is 16 consecutive columns, so the
    order cannot be permuted without breaking the format). At the first column of each group the
    group's e4m3 scale is chosen by the clip search, weighted by diag(H), on the weights as updated
    so far; then each column is rounded on that scale and its error, scaled by the inverse Hessian,
    is subtracted from the columns after it (Frantar et al., 2022). `weight_scale_2` comes from the
    original tensor's largest weight, as in `quantize_clipped`.
    """
    N, K = W.shape
    assert K % GROUP == 0 and block % GROUP == 0, (K, block)
    dev = W.device
    W = W.float().clone()
    H = H.float().clone()
    diag = torch.diagonal(H)
    dead = diag <= 0
    if dead.any():
        # a channel the corpus never excited: no information about it, so no update through it
        H[dead, dead] = 1.0
    act_w = diag.clamp_min(0).clone()
    amax_t = W.abs().max()
    scale_2 = float((amax_t / (E2M1_MAX * E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny))
    ar = torch.arange(K, device=dev)
    d = damp * float(diag.mean())
    L = None
    for attempt in range(6):
        H[ar, ar] += d * (10 ** attempt if attempt else 1)
        L, info = torch.linalg.cholesky_ex(H)
        if int(info) == 0:
            break
    else:
        raise RuntimeError("H is not positive definite even with 1e5 x damping")
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    del L, H
    grid = FP4_GRID.to(dev)
    nib = torch.empty(N, K, dtype=torch.uint8, device=dev)
    s8_all = torch.empty(N, K // GROUP, dtype=torch.float32, device=dev)
    tiny = torch.finfo(torch.float32).tiny
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        eff = None
        for j in range(i2 - i1):
            col = i1 + j
            if col % GROUP == 0:
                s8 = _group_scale(W1[:, j:j + GROUP], act_w[col:col + GROUP], scale_2, ratios)
                s8_all[:, col // GROUP] = s8[:, 0]
                eff = (s8[:, 0] * scale_2).clamp_min(tiny)
            w = W1[:, j]
            v = w / eff
            q = _round_e2m1(v.abs().clamp(0.0, E2M1_MAX))
            nib[:, col] = q | ((v < 0).to(torch.uint8) * 8)
            deq = grid[q.long()] * torch.sign(v) * eff
            err = (w - deq) / Hinv1[j, j]
            W1[:, j:] -= err[:, None] * Hinv1[j, j:][None, :]
            Err1[:, j] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        del W1, Err1
    codes = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return NVFP4Block(codes.contiguous(), s8_all.to(torch.float8_e4m3fn), scale_2)


def output_error(W: torch.Tensor, blk: NVFP4Block, H: torch.Tensor) -> float:
    """tr(E H E^T) / tr(W H W^T), E = W - dequant(blk): the relative squared error of the
    projection's output over the calibration inputs. The number both methods minimise."""
    E = W.float() - blk.dequant().float()
    num = float(((E @ H) * E).sum())
    den = float(((W.float() @ H) * W.float()).sum())
    return num / max(den, 1e-30)


# ------------------------------------------------------------------ the activation tap
class ProjTap:
    """Calls `fn(name, x)` with the input of every quantisable projection of the 64 layers.

    The tap is on `engine.model.linear`, keyed by the identity of the weight object, so it names
    the activation exactly whatever order the forward fetches weights in. Fused projection groups
    are one launch for several names and cannot be attributed; the tap refuses to run over them.
    """

    def __init__(self, w, fn):
        self.fn = fn
        self.ids = {id(b): n for n, b in w.q.items()
                    if n.startswith("layers.") and n.split(".", 2)[2] in TARGETS["all"]}
        if getattr(w, "g", None):
            raise RuntimeError("projection groups are fused (QWEN38_FUSE_PROJ=1); the tap cannot "
                               "name their inputs. Run with QWEN38_FUSE_PROJ=0.")
        self.seen: set[str] = set()

    def __enter__(self):
        import engine.model as model_mod
        self._mod = model_mod
        self._orig = model_mod.linear
        orig, ids, fn, seen = self._orig, self.ids, self.fn, self.seen

        def tap_linear(x, weight):
            name = ids.get(id(weight))
            if name is not None:
                seen.add(name)
                fn(name, x.reshape(-1, x.shape[-1]))
            return orig(x, weight)

        model_mod.linear = tap_linear
        return self

    def __exit__(self, *exc):
        self._mod.linear = self._orig
        return False


def read_corpus(spec: str, tok, default_tokens: int) -> list[tuple[str, torch.Tensor]]:
    """`a.txt:65536,b.txt` -> [(path, ids)], each file cut to its own token budget.

    `tok` is a tokenizer, or a function returning one (called only if a text file is named). A
    `.pt` file is a 1-D tensor of token ids, already tokenized."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        path, _, n = part.partition(":")
        path = os.path.expanduser(path)
        if path.endswith(".pt"):
            ids = torch.load(path).long().view(-1)
        else:
            if callable(tok) and not hasattr(tok, "encode"):
                tok = tok()
            ids = tok(open(path).read(), return_tensors="pt").input_ids[0]
        out.append((path, ids[: int(n) if n else default_tokens]))
    return out


def run_corpus(eng, corpus, chunk: int, device: str = "cuda", log: bool = True) -> int:
    """Every chunk of every file through the engine as its own sequence; returns tokens seen."""
    n = 0
    for path, ids in corpus:
        for start in range(0, ids.numel(), chunk):
            piece = ids[start:start + chunk]
            if piece.numel() < 32:
                break
            eng.reset()
            with torch.no_grad():
                eng.forward(piece.to(device), start=0, last_only=True)
            n += piece.numel()
        if log:
            print(f"  {os.path.basename(path)}: {ids.numel()} tokens", flush=True)
    return n


# ------------------------------------------------------------------ commands
def cmd_quant(args) -> None:
    snapshot = resolve_snapshot(args.model)
    src = Source(snapshot)
    stats = torch.load(args.stats, map_location="cpu") if args.stats else None
    if args.mode == "clip" and stats is None:
        raise SystemExit("--mode clip needs --stats; run `quant_nvfp4.py stats` first")
    out: dict[str, torch.Tensor] = {}
    src_bytes = 0
    nvfp4_total = 0
    missing_stats: list[str] = []
    t0 = time.time()
    layers = src.layers()[: args.layers] if args.layers else src.layers()
    projs = TARGETS[args.targets]
    for layer in layers:
        for proj in projs:
            base = f"layers.{layer}.{proj}"
            ref = src.read(base, args.device)
            if ref is None:
                continue
            src_bytes += src.stored_bytes(base)
            if args.mode == "rtn":
                blk = quantize_to_nvfp4(ref.to(torch.bfloat16))
            else:
                act = stats.get(base, None) if stats else None
                if act is None:
                    missing_stats.append(base)
                blk = quantize_clipped(ref.to(torch.bfloat16), act)
            out[f"{base}.weight"] = blk.w.cpu()
            out[f"{base}.weight_scale"] = blk.s.cpu()
            out[f"{base}.weight_scale_2"] = torch.tensor(blk.s2, dtype=torch.float32)
            nvfp4_total += blk.nbytes
            del ref, blk
        if layer % 8 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  layer {layer:2d}  {time.time() - t0:6.1f} s", flush=True)
    if missing_stats:
        # Not fatal, and not silent: an unweighted clip search is a different quantiser from the
        # one the MLP gate passed on, and the gate has to know which it is looking at.
        print(f"[warn] {len(missing_stats)} projections had no activation statistics and were "
              f"clipped on unweighted error, e.g. {missing_stats[0]}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out, metadata={"format": "nvfp4", "group": str(GROUP),
                                       "mode": args.mode, "targets": args.targets,
                                       "source": snapshot, "source_kind": src.kind})
    print(f"[{args.mode}/{args.targets}] {len(layers)} layers, {len(out) // 3} projections  "
          f"fp8 {src_bytes / 1e9:.2f} GB -> nvfp4 {nvfp4_total / 1e9:.2f} GB  "
          f"({nvfp4_total / max(1, src_bytes):.3f}x)  -> {args.out}")


def _engine(model: str | None, chunk: int, device: str = "cuda"):
    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    cfg = load_config(model)
    w = Weights(cfg.path, device=device, skip_mtp=True, nvfp4="", fp8_head="")
    eng = Qwen38Engine(cfg, w, max_len=chunk + 16, device=device)

    def tok():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(cfg.path)
    return cfg, w, eng, tok


def cmd_stats(args) -> None:
    """Per-input-channel mean square activation for every quantisable projection, over a corpus.

    `--corpus` is one file or several (`a.txt:65536,b.txt:32768`, a token budget per file; a file
    without one takes `--tokens`). Every chunk is its own sequence.
    """
    cfg, w, eng, tok = _engine(args.model, args.chunk, args.device)
    acc: dict[str, torch.Tensor] = {}

    def fn(name, x):
        v = x.float().pow(2).sum(0)
        acc[name] = v if name not in acc else acc[name] + v

    corpus = read_corpus(args.corpus, tok, args.tokens)
    with ProjTap(w, fn):
        n = run_corpus(eng, corpus, args.chunk, args.device)
    out = {k: (v / max(1, n)).cpu() for k, v in acc.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"[stats] {len(out)} projections over {n} tokens -> {args.out}")


def _plan_passes(w, layers: list[int], budget: float) -> list[list[int]]:
    """Layer ranges whose input second moments (fp32, one per distinct input) fit in `budget` bytes."""
    passes, cur, used = [], [], 0
    for layer in layers:
        need = 0
        for base in {input_key(b) for b in w.q if b.startswith(f"layers.{layer}.")
                     and b.split(".", 2)[2] in TARGETS["all"]}:
            K = w.q[base].shape[1]
            need += K * K * 4
        if cur and used + need > budget:
            passes.append(cur)
            cur, used = [], 0
        cur.append(layer)
        used += need
    if cur:
        passes.append(cur)
    return passes


def _weight_fp32(blk) -> torch.Tensor:
    return blk.dequant().float() if hasattr(blk, "dequant") else blk.float()


def cmd_build(args) -> None:
    """Statistics, second moments and every method's weights, in one process from one load.

    The model is loaded once (bf16, ~52 GB on the device). The first pass over the corpus records
    every projection's mean square input (the `stats` file, which `clip` weights by) and the full
    second moments of the first layers' inputs; later passes record the second moments of the
    rest, as many layers per pass as `--h-budget-gb` holds, and each pass's projections are
    quantised by GPTQ before the next pass starts. Writes, under `--out-dir`:
    `stats.pt`, `<method>/{mlp,gdn,attn}.safetensors` and `report.json` (per projection: the
    relative output error each method leaves on the calibration inputs).
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ("clip", "gptq", "rtn"):
            raise SystemExit(f"unknown method {m}")
    ratios = tuple(float(r) for r in args.ratios.split(","))
    cfg, w, eng, tok = _engine(args.model, args.chunk, args.device)
    print(f"[load] {w.report()}", flush=True)
    corpus = read_corpus(args.corpus, tok, args.tokens)
    bases = sorted((b for b in w.q if b.startswith("layers.") and b.split(".", 2)[2]
                    in TARGETS[args.targets]), key=lambda b: (int(b.split(".")[1]), b))
    layers = sorted({int(b.split(".")[1]) for b in bases})
    if args.layers:
        layers = layers[: args.layers]
        bases = [b for b in bases if int(b.split(".")[1]) in layers]
    passes = _plan_passes(w, layers, args.h_budget_gb * 1e9) if "gptq" in methods else [layers]
    print(f"[plan] {len(bases)} projections, {len(layers)} layers, {len(passes)} passes "
          f"(second-moment budget {args.h_budget_gb} GB)", flush=True)
    out: dict[str, dict[str, dict[str, torch.Tensor]]] = {m: {} for m in methods}
    report: dict[str, dict] = {}
    sq: dict[str, torch.Tensor] = {}
    n_tok = 0
    t_all = time.time()

    def store(method: str, base: str, blk: NVFP4Block) -> None:
        tgt = target_of(base.split(".", 2)[2])
        d = out[method].setdefault(tgt, {})
        d[f"{base}.weight"] = blk.w.cpu()
        d[f"{base}.weight_scale"] = blk.s.cpu()
        d[f"{base}.weight_scale_2"] = torch.tensor(blk.s2, dtype=torch.float32)

    for pi, pass_layers in enumerate(passes):
        keys = sorted({input_key(b) for b in bases if int(b.split(".")[1]) in pass_layers}) \
            if "gptq" in methods else []
        H = {k: torch.zeros(w.q[k].shape[1], w.q[k].shape[1], dtype=torch.float32,
                            device=w.q[k].w.device) for k in keys}
        first = pi == 0

        def fn(name, x, H=H, first=first):
            if first:
                v = x.float().pow(2).sum(0)
                sq[name] = v if name not in sq else sq[name] + v
            h = H.get(name)
            if h is not None:
                xf = x.float()
                h.addmm_(xf.t(), xf)

        t0 = time.time()
        with ProjTap(w, fn) as tap:
            n = run_corpus(eng, corpus, args.chunk, args.device, log=first)
        missing = [b for b in bases if b not in tap.seen]
        if missing:
            raise RuntimeError(f"{len(missing)} projections never reached the tap, e.g. {missing[0]}")
        if first:
            n_tok = n
            stats = {k: (v / max(1, n)).cpu() for k, v in sq.items()}
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(stats, os.path.join(args.out_dir, "stats.pt"))
            print(f"[stats] {len(stats)} projections over {n} tokens", flush=True)
        print(f"[pass {pi + 1}/{len(passes)}] layers {pass_layers[0]}-{pass_layers[-1]}, "
              f"{len(keys)} second moments, {time.time() - t0:.1f} s", flush=True)
        if first and ("clip" in methods or "rtn" in methods):
            t1 = time.time()
            for base in bases:
                Wt = _weight_fp32(w.q[base])
                if "clip" in methods:
                    store("clip", base, quantize_clipped(Wt.to(torch.bfloat16), stats[base], ratios=ratios))
                if "rtn" in methods:
                    store("rtn", base, quantize_to_nvfp4(Wt.to(torch.bfloat16)))
                del Wt
            print(f"[clip] {len(bases)} projections, {time.time() - t1:.1f} s", flush=True)
        if "gptq" in methods:
            t1 = time.time()
            for base in bases:
                if int(base.split(".")[1]) not in pass_layers:
                    continue
                Wt = _weight_fp32(w.q[base])
                h = H[input_key(base)]
                blk = gptq_nvfp4(Wt, h, ratios=ratios, damp=args.damp)
                store("gptq", base, blk)
                rep = report.setdefault(base, {"shape": list(Wt.shape)})
                rep["gptq"] = output_error(Wt, blk, h)
                if "clip" in methods:
                    t = out["clip"][target_of(base.split(".", 2)[2])]
                    cblk = NVFP4Block(t[f"{base}.weight"].to(Wt.device),
                                      t[f"{base}.weight_scale"].to(Wt.device),
                                      float(t[f"{base}.weight_scale_2"]))
                    rep["clip"] = output_error(Wt, cblk, h)
                    del cblk
                del Wt, blk
            print(f"[gptq] layers {pass_layers[0]}-{pass_layers[-1]}, {time.time() - t1:.1f} s",
                  flush=True)
        del H
        torch.cuda.empty_cache()

    snapshot = cfg.path
    meta_common = {"format": "nvfp4", "group": str(GROUP), "source": snapshot,
                   "source_kind": Source(snapshot).kind if os.path.isdir(snapshot) else "?",
                   "calibration": args.corpus, "calibration_tokens": str(n_tok),
                   "ratios": args.ratios}
    for m in methods:
        d = os.path.join(args.out_dir, m)
        os.makedirs(d, exist_ok=True)
        for tgt, tensors in sorted(out[m].items()):
            meta = dict(meta_common, mode=m, targets=tgt)
            if m == "gptq":
                meta["damp"] = str(args.damp)
            path = os.path.join(d, f"{tgt}.safetensors")
            save_file(tensors, path, metadata=meta)
            print(f"[out] {m}/{tgt}: {len(tensors) // 3} projections -> {path}", flush=True)
    summary = {"tokens": n_tok, "passes": len(passes), "seconds": time.time() - t_all,
               "per_projection": report}
    if report:
        for m in ("clip", "gptq"):
            vals = [r[m] for r in report.values() if m in r]
            if vals:
                summary[f"mean_rel_output_error_{m}"] = sum(vals) / len(vals)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[build] done in {time.time() - t_all:.0f} s; "
          + "  ".join(f"{k} {v:.5f}" for k, v in summary.items() if k.startswith("mean_")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quant")
    q.add_argument("--model", default=None)
    q.add_argument("--mode", choices=("rtn", "clip"), default="rtn")
    q.add_argument("--stats", default=None)
    q.add_argument("--out", required=True)
    q.add_argument("--device", default="cuda")
    q.add_argument("--layers", type=int, default=0, help="quantise only the first N layers")
    q.add_argument("--targets", choices=tuple(TARGETS), default="mlp",
                   help="which projections to quantise; see TARGETS for what each excludes")
    q.set_defaults(fn=cmd_quant)

    s = sub.add_parser("stats")
    s.add_argument("--model", default=None)
    s.add_argument("--corpus", required=True, help="file[:tokens][,file[:tokens]...]")
    s.add_argument("--out", required=True)
    s.add_argument("--tokens", type=int, default=8192, help="token budget of a file without one")
    s.add_argument("--chunk", type=int, default=1024)
    s.add_argument("--device", default="cuda")
    s.set_defaults(fn=cmd_stats)

    b = sub.add_parser("build", help="stats + clip and/or gptq for every target, one process")
    b.add_argument("--model", default=None)
    b.add_argument("--corpus", required=True, help="file[:tokens][,file[:tokens]...]")
    b.add_argument("--tokens", type=int, default=65536)
    b.add_argument("--chunk", type=int, default=2048)
    b.add_argument("--methods", default="clip,gptq")
    b.add_argument("--targets", choices=tuple(TARGETS), default="all")
    b.add_argument("--ratios", default=",".join(f"{r:g}" for r in CLIP_RATIOS))
    b.add_argument("--damp", type=float, default=0.01)
    b.add_argument("--h-budget-gb", type=float, default=22.0)
    b.add_argument("--layers", type=int, default=0, help="only the first N layers (smoke runs)")
    b.add_argument("--device", default="cuda")
    b.add_argument("--out-dir", required=True)
    b.set_defaults(fn=cmd_build)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
