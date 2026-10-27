"""E1 -- the deciding experiment for the pruning branch: what does dropping layers cost?

The pruning study ranked MLP-width thinning third and gates it behind one
free measurement, because the whole surgery is only worth about +4 tok/s and it carries all of the
quality risk in the program. This tool is that measurement, done with **no training at all**: drop
L whole blocks from the 64-layer hybrid, and read off what the model loses.

Four things are measured, for `L` in {0, 4, 8, 12, 16} and for two ways of choosing the layers:

  (a) **mean p1 on the model's own greedy trajectory**, teacher-forced. `p1` is the target's own
      top-1 probability -- the ceiling on any drafter's per-token agreement,
      so it is the direct proxy for `tau`, and by the note's section 0.2 `tau` decides everything.
      Reported next to `p1(teacher)`, the probability the pruned model puts on the token the
      UNPRUNED model actually wrote, and next to argmax agreement with that token.
  (b) **free-generation suffix repetition** over 256 new tokens, ShortOPD's collapse metric, as the
      fraction of token 8-grams that repeat one seen earlier in the same continuation. Teacher-forced
      loss cannot see a model that fails to stay on its own trajectory (ARCHITECTURE.md 1.9), and
      this is the only number here that can.
  (c) **real ms/step and verify(8)/verify(16) ms** with the layers skipped in the engine, to check
      the byte model in the note's section 0.1 against the board.
  (d) the **implied cold-prose tok/s** from (a) and (c) under the ledger's cost model.

Which layers to drop is decided by a block-influence score in the ShortGPT / Gromov sense: the
angular distance between a block's input and output residual, averaged over the model's own prose,
computed on TOPICS THE EVALUATION NEVER SEES. Layer 0 and the last layer are never candidates. An
attention layer is admitted to the mixed set only if its score is clearly below the cheapest GDN
layer it would displace -- "close" is not a reason to delete a full-attention stage out of a chain
of 64 serial ones (note section 1.4).

A skipped block is the identity: the residual passes through untouched, and the block's state slots
are left exactly as they were, because nothing reads them again.

    python tools/e1_layer_drop.py --stage score --out results/e1
    python tools/e1_layer_drop.py --stage eval,time --out results/e1
    python tools/e1_layer_drop.py --stage gen --configs L0,L4-gdn,L4-mix,L8-gdn --out results/e1

Stages write JSON into `--out` and are separable so that no single run holds the box for long.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402

# --------------------------------------------------------------------- the skip

# The engine is patched from the outside rather than edited, for two reasons: other tracks are
# working in the same tree and `engine/model.py` must not grow an experiment's flag, and a block
# that returns zeros into its own residual add IS the identity, so nothing about the forward pass
# needs to know the layer is gone. `*a, **kw` forwards whatever signature the day's engine has.

_ATTENTION = Qwen38Engine.attention
_LINEAR = Qwen38Engine.linear_attention
_MLP = Qwen38Engine.mlp

Qwen38Engine.skip_layers = frozenset()


def _attention(self, h, p, layer, *a, **kw):
    if layer in self.skip_layers:
        return torch.zeros_like(h)
    return _ATTENTION(self, h, p, layer, *a, **kw)


def _linear_attention(self, h, p, layer, *a, **kw):
    if layer in self.skip_layers:
        return torch.zeros_like(h)
    return _LINEAR(self, h, p, layer, *a, **kw)


def _mlp(self, h, p, *a, **kw):
    if int(p.split(".")[1]) in self.skip_layers:
        return torch.zeros_like(h)
    return _MLP(self, h, p, *a, **kw)


Qwen38Engine.attention = _attention
Qwen38Engine.linear_attention = _linear_attention
Qwen38Engine.mlp = _mlp

# ------------------------------------------------------------------ cost model

# NVFP4 projections + fp8 head. `s` is the 09:55 fit of the verified-token
# slope; `d` is the block drafter plus the trimmed fp8 draft head (RESEARCH-0917 section 0.3).
BYTES_GB = 15.00
BANDWIDTH_GB_S = 163.0
GDN_LAYER_GB = 0.2165
ATTN_LAYER_GB = 0.2093
SLOPE_MS = 1.896
DRAFT_MS = 17.0
NODES = 16

# ------------------------------------------------------------------- the data

PROSE_TOPICS = ("business", "creative", "everyday", "history", "law", "math", "medicine",
                "reasoning", "science")


def load_sequences(data: str) -> list[dict]:
    """Every recorded own-output continuation, duplicates by token id dropped, sorted by name."""
    seen: set[bytes] = set()
    out = []
    for path in sorted(glob.glob(os.path.join(data, "gen*.pt"))):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        digest = hashlib.sha1(raw["ids"].numpy().tobytes()).digest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append({
            "name": os.path.basename(path)[:-3],
            "topic": raw["topic"],
            "gen_start": int(raw["gen_start"]),
            "ids": raw["ids"].long(),
            "label": raw["label"].long(),
        })
    return out


def split_by_topic(sequences: list[dict], score_topics: tuple[str, ...]) -> tuple[list, dict]:
    """The scoring set and the evaluation set share no topic, so no layer is chosen on its own test."""
    score = [s for s in sequences if s["topic"] in score_topics]
    groups: dict[str, list] = {}
    for s in sequences:
        if s["topic"] in score_topics:
            continue
        key = "prose" if s["topic"] in PROSE_TOPICS else s["topic"]
        groups.setdefault(key, []).append(s)
    return score, groups


# ----------------------------------------------------------- block influence


def block_influence(engine: Qwen38Engine, sequences: list[dict], want: int) -> dict:
    """Angular distance between each block's input and output residual, on the model's own prose.

    ShortGPT's Block Influence and Gromov's angular distance are the same statistic read two ways:
    `1 - cos` and `arccos(cos) / pi`. Both are reported because the literature quotes both, and the
    ordering they induce is what actually picks the layers.
    """
    n_layers = engine.cfg.num_hidden_layers
    cos_sum = np.zeros(n_layers)
    ang_sum = np.zeros(n_layers)
    count = 0
    used = []
    for seq in sequences:
        if count >= want:
            break
        taps: list[torch.Tensor] = []
        engine.reset()
        engine.tap = taps.append
        try:
            engine.forward(seq["ids"].to(engine.device), start=0)
        finally:
            engine.tap = None
        lo = seq["gen_start"]
        for layer in range(n_layers):
            a = taps[layer][lo:].float()
            b = taps[layer + 1][lo:].float()
            c = F.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0)
            cos_sum[layer] += float(c.sum())
            ang_sum[layer] += float((torch.arccos(c) / math.pi).sum())
        count += taps[0][lo:].shape[0]
        used.append(seq["name"])
        del taps
    return {
        "positions": count,
        "sequences": used,
        "cos": (cos_sum / count).tolist(),
        "bi": (1.0 - cos_sum / count).tolist(),
        "angular": (ang_sum / count).tolist(),
    }


def choose(cfg, angular: list[float], drops: int, mode: str, margin: float = 0.10) -> list[int]:
    """The layers to delete, lowest angular distance first, under the note's exclusions.

    `gdn` restricts the candidates to linear-attention blocks. `mixed` lets a full-attention block
    in only when its score is at least `margin` BELOW the cheapest GDN block it would displace: a
    near-tie is not evidence, and on a sequential hybrid the sixteen attention layers are the only
    blocks that can move information across a distance the recurrent state has forgotten.
    """
    if drops == 0:
        return []
    last = cfg.num_hidden_layers - 1
    eligible = [i for i in range(1, last) if mode != "gdn" or cfg.is_linear(i)]
    order = sorted(eligible, key=lambda i: angular[i])
    if mode in ("gdn", "raw"):
        return sorted(order[:drops])
    chosen: list[int] = []
    gdn_pool = [i for i in order if cfg.is_linear(i)]
    for layer in order:
        if len(chosen) == drops:
            break
        if cfg.is_linear(layer):
            chosen.append(layer)
            continue
        rival = next((i for i in gdn_pool if i not in chosen), None)
        if rival is not None and angular[layer] > (1.0 - margin) * angular[rival]:
            continue  # too close to call: keep the attention stage, take the GDN block instead
        chosen.append(layer)
    for layer in gdn_pool:  # top up if attention candidates were refused
        if len(chosen) == drops:
            break
        if layer not in chosen:
            chosen.append(layer)
    return sorted(chosen)


def build_configs(cfg, angular: list[float], levels: tuple[int, ...]) -> list[dict]:
    """Three ways of reading the same ranking, with the duplicates collapsed.

    `gdn` deletes linear-attention blocks only. `mix` takes the lowest-scoring blocks of either
    kind under the near-tie guard. `raw` takes the lowest-scoring blocks with the guard switched
    off, which is what a naive ShortGPT reading of the score would do. When the guard changes
    nothing the three collapse to fewer configurations and only the distinct ones are run, so the
    depth-vs-kind question gets an answer either way and no box time is spent twice.
    """
    out = [{"name": "L0", "drops": 0, "mode": "none", "layers": []}]
    for L in levels:
        if L == 0:
            continue
        seen: list[list[int]] = []
        for mode, suffix in (("gdn", "gdn"), ("mixed", "mix"), ("raw", "raw")):
            layers = choose(cfg, angular, L, mode)
            if layers in seen:
                continue
            seen.append(layers)
            out.append({"name": f"L{L}-{suffix}", "drops": L, "mode": mode, "layers": layers})
    return out


def config_bytes(cfg, layers: list[int]) -> float:
    gone = sum(GDN_LAYER_GB if cfg.is_linear(i) else ATTN_LAYER_GB for i in layers)
    return BYTES_GB - gone


# ------------------------------------------------------------- teacher forcing


def teacher_forced(engine: Qwen38Engine, seq: dict, chunk: int = 128) -> dict:
    """P1, p1 against the unpruned model's own token, and argmax agreement, over the continuation."""
    ids = seq["ids"].to(engine.device)
    label = seq["label"].to(engine.device)
    lo = seq["gen_start"]
    hi = ids.numel() - 1                     # position `hi` has no recorded successor
    engine.reset()
    logits = engine.forward(ids, start=0)[0]
    p1, p_teacher, agree = [], [], []
    for c0 in range(lo, hi, chunk):
        c1 = min(c0 + chunk, hi)
        probs = logits[c0:c1].float().softmax(dim=-1)
        top = probs.max(dim=-1)
        p1.append(top.values.cpu())
        agree.append((top.indices == label[c0:c1]).float().cpu())
        p_teacher.append(probs.gather(1, label[c0:c1, None])[:, 0].cpu())
        del probs, top
    del logits
    return {
        "p1": torch.cat(p1).numpy(),
        "p_teacher": torch.cat(p_teacher).numpy(),
        "agree": torch.cat(agree).numpy(),
    }


def run_length(p: np.ndarray, block: int) -> float:
    """Expected accepted tokens per block for per-position agreement probabilities `p`.

    Copied in spirit from `tools/entropy_ceiling.py`: a block accepts its j-th token only if the
    j - 1 before it were accepted, so the expectation is a product-sum, averaged over every start
    that has a full block after it.
    """
    n = len(p)
    if n <= block:
        return float("nan")
    cum = np.cumprod(np.lib.stride_tricks.sliding_window_view(p, block), axis=1)
    return float(cum.sum(axis=1)[: n - block].mean())


# ------------------------------------------------------------ free generation


def generate(engine: Qwen38Engine, prompt: torch.Tensor, new: int, eos: set[int]) -> list[int]:
    engine.reset()
    out: list[int] = []
    logits = engine.forward(prompt.to(engine.device), start=0, last_only=True)[0, -1]
    pos = prompt.numel()
    for _ in range(new):
        nxt = int(logits.argmax())
        out.append(nxt)
        if nxt in eos:
            break
        logits = engine.forward(torch.tensor([nxt], device=engine.device), start=pos,
                                last_only=True)[0, -1]
        pos += 1
    return out


def repetition(tokens: list[int], n: int = 8) -> dict:
    """Fraction of token n-grams that repeat one already seen in the same continuation."""
    if len(tokens) <= n:
        return {"tokens": len(tokens), "rep8": 0.0, "uniq8": 1.0, "longest_run": 0}
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    uniq = len(set(grams)) / len(grams)
    longest = run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        longest = max(longest, run)
    return {"tokens": len(tokens), "rep8": 1.0 - uniq, "uniq8": uniq, "longest_run": longest}


# -------------------------------------------------------------------- timing


def timings(engine: Qwen38Engine, ids: torch.Tensor, blocks: tuple[int, ...], reps: int) -> dict:
    """Wall-clock of a verify pass of each block length, after a real 256-token prefill."""
    out = {}
    for m in blocks:
        samples = []
        for _ in range(reps + 1):
            engine.reset()
            engine.forward(ids[:256].to(engine.device), start=0, last_only=True)
            piece = ids[256:256 + m].to(engine.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.forward_block(piece, start=256)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        out[m] = float(np.median(samples[1:]))     # the first pass pays for lazy allocations
    return out


# ---------------------------------------------------------------------- main


def build_engine(args) -> tuple:
    cfg = load_config(args.model)
    w = Weights(cfg.path, skip_mtp=True)
    engine = Qwen38Engine(cfg, w, max_len=args.max_len)
    print(f"[engine] {w.report()}", flush=True)
    return cfg, w, engine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default="train/data")
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--stage", default="score", help="comma list of score,eval,gen,time")
    ap.add_argument("--score-topics", default="science")
    ap.add_argument("--score-tokens", type=int, default=2048)
    ap.add_argument("--levels", default="4,8,12,16")
    ap.add_argument("--configs", default="", help="restrict to these config names")
    ap.add_argument("--eval-groups", default="prose,multilingual,code")
    ap.add_argument("--code-limit", type=int, default=8, help="code sequences, as a control row")
    ap.add_argument("--gen-new", type=int, default=256)
    ap.add_argument("--gen-prompts", type=int, default=8)
    ap.add_argument("--time-reps", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)
    score_topics = tuple(t.strip() for t in args.score_topics.split(","))
    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())

    sequences = load_sequences(args.data)
    score_set, groups = split_by_topic(sequences, score_topics)
    if "code" in groups:
        groups["code"] = groups["code"][: args.code_limit]
    print(f"[data] {len(sequences)} distinct recorded continuations; "
          f"scoring on {score_topics} ({len(score_set)} seq); "
          + ", ".join(f"{k} {len(v)} seq" for k, v in sorted(groups.items())), flush=True)

    cfg, w, engine = build_engine(args)
    eos = set(cfg.eos_token_ids)
    print(f"[layers] {cfg.num_hidden_layers} total, {len(cfg.linear_layers)} linear, "
          f"{len(cfg.attention_layers)} attention at {cfg.attention_layers}", flush=True)

    score_path = os.path.join(args.out, "block_influence.json")
    if "score" in stages:
        t0 = time.time()
        engine.skip_layers = frozenset()
        bi = block_influence(engine, score_set, args.score_tokens)
        bi["topics"] = list(score_topics)
        bi["layer_types"] = cfg.layer_types
        json.dump(bi, open(score_path, "w"), indent=1)
        print(f"[score] {bi['positions']} own-prose positions from {len(bi['sequences'])} "
              f"sequences in {time.time() - t0:.1f} s -> {score_path}", flush=True)
        order = sorted(range(cfg.num_hidden_layers), key=lambda i: bi["angular"][i])
        print("        layer  kind  angular     BI")
        for i in order:
            mark = "linear" if cfg.is_linear(i) else " ATTN "
            print(f"        {i:5d}  {mark} {bi['angular'][i]:8.4f} {bi['bi'][i]:8.4f}")

    if not os.path.exists(score_path):
        print(f"[fatal] no {score_path}; run --stage score first")
        return 1
    bi = json.load(open(score_path))
    configs = build_configs(cfg, bi["angular"], levels)
    if args.configs:
        want = {c.strip() for c in args.configs.split(",")}
        configs = [c for c in configs if c["name"] in want]
    print("\n[configs]")
    for c in configs:
        kinds = "".join("A" if not cfg.is_linear(i) else "g" for i in c["layers"])
        print(f"  {c['name']:<9} L={c['drops']:<3} {config_bytes(cfg, c['layers']):.3f} GB  "
              f"{c['layers']} {kinds}")
    json.dump(configs, open(os.path.join(args.out, "configs.json"), "w"), indent=1)

    for cname, run in (("eval", "eval"), ("time", "time"), ("gen", "gen")):
        if cname not in stages:
            continue
        path = os.path.join(args.out, f"{run}.json")
        acc = json.load(open(path)) if os.path.exists(path) else {}
        for c in configs:
            engine.skip_layers = frozenset(c["layers"])
            entry = acc.setdefault(c["name"], {"layers": c["layers"], "drops": c["drops"],
                                               "mode": c["mode"],
                                               "bytes_gb": config_bytes(cfg, c["layers"])})
            t0 = time.time()
            if run == "eval":
                for gname, seqs in sorted(groups.items()):
                    if gname not in args.eval_groups.split(","):
                        continue
                    p1, pt, ag, per_seq8, per_seq16 = [], [], [], [], []
                    for seq in seqs:
                        r = teacher_forced(engine, seq)
                        p1.append(r["p1"])
                        pt.append(r["p_teacher"])
                        ag.append(r["agree"])
                        per_seq8.append(run_length(r["p1"], 8))
                        per_seq16.append(run_length(r["p1"], 16))
                    p1 = np.concatenate(p1)
                    entry[gname] = {
                        "sequences": len(seqs), "positions": int(p1.size),
                        "mean_p1": float(p1.mean()), "median_p1": float(np.median(p1)),
                        "frac_p1_below_half": float((p1 < 0.5).mean()),
                        "mean_p_teacher": float(np.concatenate(pt).mean()),
                        "argmax_agreement": float(np.concatenate(ag).mean()),
                        "acc8": float(np.nanmean(per_seq8)),
                        "acc16": float(np.nanmean(per_seq16)),
                    }
                    e = entry[gname]
                    print(f"  {c['name']:<9} {gname:<12} p1 {e['mean_p1']:.4f}  "
                          f"p1(teacher) {e['mean_p_teacher']:.4f}  "
                          f"argmax {e['argmax_agreement']:.4f}  "
                          f"acc8 {e['acc8']:.2f}  acc16 {e['acc16']:.2f}  "
                          f"({e['positions']} pos, {time.time() - t0:.0f} s)", flush=True)
            elif run == "time":
                ids = groups["prose"][0]["ids"]
                if ids.numel() < 256 + 16:
                    ids = torch.cat([ids, ids])
                entry["ms"] = timings(engine, ids, (1, 8, 16), args.time_reps)
                m = entry["ms"]
                print(f"  {c['name']:<9} step {m[1]:7.2f} ms   verify(8) {m[8]:7.2f} ms   "
                      f"verify(16) {m[16]:7.2f} ms   "
                      f"model says {config_bytes(cfg, c['layers']) / BANDWIDTH_GB_S * 1000:.2f} ms",
                      flush=True)
            elif run == "gen":
                prompts = []
                seen_topic: set[str] = set()
                for seq in groups["prose"]:
                    if seq["topic"] in seen_topic:
                        continue
                    seen_topic.add(seq["topic"])
                    prompts.append(seq)
                    if len(prompts) >= args.gen_prompts:
                        break
                entry["samples"] = []
                for seq in prompts:
                    tokens = generate(engine, seq["ids"][: seq["gen_start"]], args.gen_new, eos)
                    st = repetition(tokens)
                    st["prompt"] = seq["name"]
                    st["topic"] = seq["topic"]
                    st["ids"] = tokens
                    entry["samples"].append(st)
                    print(f"  {c['name']:<9} gen/{seq['topic']:<10} {st['tokens']:4d} tok  "
                          f"rep8 {st['rep8'] * 100:5.1f}%  longest run {st['longest_run']}  "
                          f"({time.time() - t0:.0f} s)", flush=True)
                entry["rep8"] = float(np.mean([s["rep8"] for s in entry["samples"]]))
                print(f"  {c['name']:<9} mean rep8 {entry['rep8'] * 100:.2f}%", flush=True)
            json.dump(acc, open(path, "w"), indent=1)
        print(f"[{run}] -> {path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
