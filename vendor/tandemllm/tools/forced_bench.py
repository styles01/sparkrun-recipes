"""Router configurations on the same text: teacher-forced decoding against one reference.

Changing which drafter proposes, or how many rows a verify carries, moves the bf16 ties of the
batched verify, so two configurations of a lossless engine can write different texts from the first
tie on (every configuration on the box did, fixed widths included). A comparison between different
texts compares their difficulty as much as the configurations. This tool removes that: it writes the
reference once (today's router, greedy), then runs every configuration against it. Each block is
drafted and verified by the real kernels at the real context length, as the serving loop does; the
accepted path is the longest path of the tree that the REFERENCE follows, and the round commits that
path plus the reference's next token. The ms are the real rounds; the tokens are the reference's,
the same for every configuration.

Where the target's own argmax would differ from the reference (a tie), the forced token is taken
anyway; that is at most a tie a request, and it is the same text for all configurations.

    python tools/forced_bench.py --ckpt8 train/ft-b8-v2 --ckpt16 train/ft-b16 --corpus corpus \\
        --nvfp4 A,B,C --fp8-head H --configs base,wide,b8,b16 --repeat 3 --json-out out.json

A config name ending in `_nn` is that config with the narrow arm's place held by
`engine.lenrouter.NoArm` (SPD-75, the server's `--narrow-arm off`): `widetrs_nn` against `widetrs`
is the served policy with and without the block-8 arm, on the same text.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.ngram import NgramDrafter  # noqa: E402
from engine.lenrouter import LengthRouter  # noqa: E402
from engine.router import MergedRouter, served_tree_table, tree_nodes  # noqa: E402
from engine.settings import SETTINGS as _S  # noqa: E402


def workloads(tok, sets: set, long_tokens: int, row_n: int):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from bench_decode import PROMPTS, SNIPPET
    W = []
    if "five" in sets:
        W += [(n, t, 256) for n, t in PROMPTS.items()]
    if "more" in sets:
        W.append(("short", "In one sentence: why does a language model that writes one token at a "
                  "time run faster on a machine with more memory bandwidth?", 48))
        W.append(("edit_heavy", "Here is a Python module:\n\n```python" + SNIPPET + "```\n\n"
                  "Refactor it: put the functions into one class with methods, give every name a "
                  "clearer one, add input checks with helpful error messages, and rewrite the "
                  "docstrings. Output the complete new module.", 768))
        W.append(("edit_light", "Here is a Python module:\n\n```python" + SNIPPET + "```\n\n"
                  "Fix the spelling in the comments and docstrings only. Change nothing else. "
                  "Output the complete module.", 768))
        W.append(("mixed", "Here is a Python module:\n\n```python" + SNIPPET + "```\n\n"
                  "First explain in two short paragraphs what it does and what could go wrong. "
                  "Then output the complete module again with type hints added and nothing else "
                  "changed.", 768))
    if "row" in sets:
        sys.path.insert(0, os.path.expanduser("~/inf-atlas/bench"))
        from atlas_bench.data import filter_prompt_rows, load_prompt_rows, sample_prompts
        from atlas_bench.registry import Registry
        reg = Registry(os.path.expanduser("~/inf-atlas"))
        wl = json.load(open(os.path.expanduser(
            "~/inf-atlas/workloads/serve-single-i256-o256-v1.json")))
        rows = filter_prompt_rows(load_prompt_rows(reg, wl["dataset_id"]), ["s", "m"])
        for i, p in enumerate(sample_prompts(rows, row_n, seed=42, target_tokens=256)):
            W.append((f"row{i:02d}", p.messages, 256))
    if "long" in sets:
        import numpy as np
        ids = np.load(os.path.expanduser("~/pub-bench/longprompts/ids-32768.npy"))
        text = tok.decode(ids[0][:long_tokens].tolist())
        code = tok.decode(ids[2][:long_tokens].tolist())
        short_text = tok.decode(ids[0][:1024].tolist())
        W.append(("copy_1k", short_text + "\n\nRepeat the text above word for word.", 768))
        W.append(("long_fresh", text + "\n\nSummarise the text above in five paragraphs.", 256))
        W.append(("long_copy", text + "\n\nRepeat the first three paragraphs of the text above "
                  "word for word.", 256))
        W.append(("long_code_edit", code + "\n\nRewrite the first function in the code above with "
                  "clearer variable names. Output only that function.", 512))
    return W


RHO: dict = {}
RHO_OLD: dict = {}


def _load_rho(path: str) -> dict:
    out = {}
    for k, v in json.load(open(path)).items():
        if k.startswith("_"):
            continue
        s_, f_ = v
        src, m, rb = k.split("|")
        if s_ + f_ >= 20:
            w = min(50.0, s_ + f_)
            out[(src, int(m), int(rb))] = [w * s_ / (s_ + f_), w * f_ / (s_ + f_)]
    return out


def build(eng, small, large, corpus, config: str, tables=None):
    alone = config.endswith("_nn")                # SPD-75: the wide arm alone
    config = config[:-3] if alone else config
    table = served_tree_table()
    ng = NgramDrafter(corpus_path=corpus, min_order=3, max_depth=16,
                      node_budget=large.cfg.block_size - 1, branch_top_k=3, min_expected=0.2,
                      alpha=0.6, corpus_weight=0.5, min_corpus_order=8,
                      verify_base_ms=table[8], verify_per_node_ms=(table[16] - table[8]) / 8)
    arms = [MergedRouter(ng, head, mtp_depth=head.cfg.block_size - 1,
                         node_budget=tree_nodes(head.cfg.block_size) - 1, mtp_ms_per_token=0.0,
                         head_fixed_ms=27.0, adaptive_depth=False, rollback_ms=6.4,
                         verify_ms_table=dict(table), tree_ms_table=dict(table))
            for head in (small, large)]
    if alone:
        from engine.lenrouter import NoArm
        arms[0] = NoArm(eng, block=small.cfg.block_size)
    kw = dict(tree=True, ngram=ng, latch=True, drop_idle=True,
              deep=int(_S.get("DEEP") or 0), deep_after=int(_S.get("DEEP_AFTER") or 2),
              latch_table=dict(table), latch_price=False, switch=False)
    if config == "b8":
        kw["fixed"] = 8
    elif config == "b16":
        kw["fixed"] = 16
    elif config.startswith("wide"):
        kw.update(switch=True, switch_mode="wide")
        if config in ("widet", "widetr", "widetrs", "widetrs_old"):
            kw["class_tables"] = tables          # per-context-class chain/tree prices
        if config in ("widetr", "widetrs"):
            kw["rho_prior"] = RHO                # the copy estimator, counted online
        if config == "widetrs_old":
            kw["rho_prior"] = RHO_OLD            # the estimator as measured before the fix:
            kw["rho_online"] = False             # the old prior alone, nothing counted
        if config in ("widetrs", "widetrs_old"):
            kw["stair_skip"] = True              # the head-skip rule on the staircase
        if config == "widel":
            kw["learn_block"] = True             # the learned round cost
    r = LengthRouter(arms[0], arms[1], **kw)
    if config == "wide0":
        # the cut on the staircase alone, without the learned cost of a round
        for arm in arms:
            arm.stair_factor = None
    r.detach()
    return r


def _walk(tree, ref: list[int], a: int) -> list[int]:
    """The longest path of `tree` the reference follows after position `a`."""
    kids: dict[int, dict[int, int]] = {}
    for i, p in enumerate(tree.parents[1:], start=1):
        kids.setdefault(p, {})[tree.tokens[i]] = i
    node, path = 0, [0]
    while a + len(path) < len(ref):
        nxt = kids.get(node, {}).get(ref[a + len(path)])
        if nxt is None:
            break
        path.append(nxt)
        node = nxt
    return path


def run(eng, router, ids, n_max, eos, ref=None, k_loop=31, chunk=4096):
    """One request. With `ref` the decode is forced onto it; without, it is plain greedy."""
    from engine import cache
    dev = ids.device
    ctx = ids.tolist()
    eng.reset()
    router.reset()
    router.attach()
    for arm in (router.small, router.large):
        st = getattr(arm, "stats", None)
        if isinstance(st, dict):
            for x in ("mtp", "chain", "ngram", "merged", "head_skipped", "declined"):
                if x in st:
                    st[x] = 0
    router.prime(ctx)
    with torch.no_grad():
        logits, _, _ = cache.prefill(eng, router, ctx, dev, store=None, chunk=chunk)
    torch.cuda.synchronize()
    tok = ref[0] if ref is not None else int(logits[0, -1].argmax())
    out = [tok]
    ctx.append(tok)
    router.observe([tok])
    pos = ids.numel()
    blocks, rows, t0 = 0, 0, time.perf_counter()
    rounds = []                      # (rows, chain-shaped, ms) per round, host time end to end
    t_r = t0
    limit = len(ref) if ref is not None else n_max
    with torch.no_grad():
        while len(out) < limit and (ref is not None or tok not in eos):
            tree = router.propose_tree(ctx, min(k_loop, limit - len(out)))
            if tree is not None:
                tree = tree.truncate(eng.max_len - pos)
            if tree is None or tree.n_draft == 0:
                prev = tok
                lg = eng.forward(torch.tensor([tok], device=dev), start=pos, last_only=True)
                router.sync([prev], eng.hidden_post_norm[0], pos)
                pos += 1
                tok = ref[len(out)] if ref is not None else int(lg[0, -1].argmax())
                out.append(tok)
                ctx.append(tok)
                router.observe([tok])
                blocks += 1
                rows += 1
                continue
            block = torch.tensor(tree.tokens, device=dev)
            lg = eng.forward_tree(block, tree.parents, start=pos)
            picks = lg.argmax(-1).tolist()
            if ref is not None:
                path = _walk(tree, ref, len(out) - 1)
                new = [tree.tokens[i] for i in path[1:]]
                if len(out) + len(new) < len(ref):
                    new.append(ref[len(out) + len(new)])
            else:
                path, new = eng.accept_tree(tree, picks)
            eng.commit_tree(path)
            sel = torch.tensor(path, device=dev)
            router.sync([int(tree.tokens[i]) for i in path], eng.hidden_post_norm[0, sel], pos,
                        rows=path)
            pos += len(path)
            new = new[:limit - len(out)]
            for t in new:
                out.append(t)
                ctx.append(t)
            router.observe(new)
            tok = out[-1]
            blocks += 1
            rows += len(tree.tokens)
            now = time.perf_counter()
            chain = all(p == i - 1 for i, p in enumerate(tree.parents[1:], start=1))
            rounds.append((len(tree.tokens), chain, round((now - t_r) * 1e3, 2)))
            t_r = now
            if ref is None and any(t in eos for t in new):
                break
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    n = len(out) - 1
    arms = {k: {x: v for x, v in getattr(arm, "stats", {}).items()
                if x in ("mtp", "chain", "ngram", "merged", "head_skipped", "declined")}
            for k, arm in (("s", router.small), ("l", router.large))}
    return out, {"tokens": n, "ms": ms, "blocks": blocks, "tok_s": n / ms * 1e3 if ms else 0.0,
                 "tpb": n / blocks if blocks else 0.0, "mspb": ms / blocks if blocks else 0.0,
                 "rows": rows / blocks if blocks else 0.0, "report": router.report(),
                 "arms": arms, "rounds": rounds,
                 "median_round_ms": statistics.median(r[2] for r in rounds) if rounds else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt8", required=True)
    ap.add_argument("--ckpt16", required=True)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--max-len", type=int, default=40960)
    ap.add_argument("--configs", default="base,wide,b8,b16")
    ap.add_argument("--sets", default="five,more,row,long")
    ap.add_argument("--row-n", type=int, default=12)
    ap.add_argument("--long-tokens", type=int, default=32000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--stair-tables", default="", help="per-class verify tables (json) for widet")
    ap.add_argument("--stair-rho", default="", help="lookup continuation counts (json) for widetr")
    ap.add_argument("--stair-rho-old", default="", help="the prior measured before the fix (widetrs_old)")
    a = ap.parse_args()
    from engine.config import load_config
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    cfg = load_config(None)
    w = Weights(cfg.path, device="cuda", skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device="cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eos = set(cfg.eos_token_ids)
    small = DFlash2Drafter(eng, a.ckpt8, blocks=1, max_len=a.max_len, block=8)
    small._build()
    large = DFlash2Drafter(eng, a.ckpt16, blocks=1, max_len=a.max_len, block=16)
    large._build()
    base_temp = float(_S.get("DF2_TEMP"))
    names = a.configs.split(",")
    tables = json.load(open(a.stair_tables)) if a.stair_tables else None
    if a.stair_rho:
        RHO.update(_load_rho(a.stair_rho))
    if a.stair_rho_old:
        RHO_OLD.update(_load_rho(a.stair_rho_old))
    routers = {c: build(eng, small, large, a.corpus, c, tables) for c in names + ["base"]}

    def use(c):
        r = routers[c]
        for h in (small, large):
            h.tree_temp = 1.4 if r.calc else base_temp
        return r

    W = workloads(tok, set(a.sets.split(",")), a.long_tokens, a.row_n)
    enc = []
    for name, msg, n in W:
        msgs = msg if isinstance(msg, list) else [{"role": "user", "content": msg}]
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False), return_tensors="pt"
                  ).input_ids[0].cuda()
        enc.append((name, ids, n))
    # warm every configuration on a short and a long request (Triton, graphs at both classes)
    for c in names:
        for name, ids, n in [e for e in enc if e[0] in ("prose", "long_fresh")]:
            run(eng, use(c), ids, 64, eos)
    refs = {}
    for name, ids, n in enc:
        out, st = run(eng, use("base"), ids, n, eos)
        refs[name] = out
        print(f"[ref] {name} prompt {ids.numel()} tokens {len(out)}", flush=True)
    rows = []
    for rep in range(a.repeat):
        order = names[rep % len(names):] + names[:rep % len(names)]
        for name, ids, n in enc:
            for c in order:
                _, st = run(eng, use(c), ids, n, eos, ref=refs[name])
                st.update(config=c, workload=name, repeat=rep, prompt=int(ids.numel()))
                rows.append(st)
                print(f"[forced] r{rep} {name:14s} {c:6s} {st['tok_s']:7.2f} tok/s "
                      f"{st['tpb']:5.2f} t {st['mspb']:6.1f} ms {st['rows']:5.1f} rows", flush=True)
        json.dump({"rows": rows, "configs": names}, open(a.json_out, "w"), indent=1)
    print("\n" + f"{'workload':15s} " + " ".join(f"{c:>24s}" for c in names))
    for name, _, _ in enc:
        cells = []
        for c in names:
            xs = [r for r in rows if r["workload"] == name and r["config"] == c]
            t = [x["tok_s"] for x in xs]
            sp = 100 * (max(t) - min(t)) / statistics.mean(t) if len(t) > 1 else 0.0
            cells.append(f"{statistics.mean(t):7.2f} ±{sp:3.1f}% "
                         f"{statistics.mean(x['tpb'] for x in xs):5.2f}t "
                         f"{statistics.mean(x['mspb'] for x in xs):5.1f}")
        print(f"{name:15s} " + " ".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()
