"""Does a flag that is OFF leave the engine exactly as it was, and what does it change when ON?

Every kernel of the 2026-09-23 kernel work sits behind a flag that defaults off. This
tool runs one fixed scenario on the real model -- a prefill, a sixteen-row chain verify with a
partial accept, an eight-row chain verify, a branching tree verify with a path commit, and a
single-token decode step -- and dumps every logit and the state after each step.

  --dump OUT       run the scenario with every kernel flag off and save it. Run from the base
                   commit's checkout as well as from this branch: the two files must be
                   bit-identical, which is "the flag-off path is byte-identical".
  --flags          then, in the same process, run it again with each flag on alone and with all
                   on, and compare each against the flags-off run: bit-identity where the change is
                   an arithmetic no-op, and otherwise the distance of the logits, the argmax
                   agreement, and the state's relative distance.
  --compare A B    compare two dumps bit for bit and exit.

The scenario uses random token ids, fixed by a seed: it measures the arithmetic, not the text.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# the kernel flags of this branch, as module:attribute; the base commit has none of them
FLAGS = ["tools.nvfp4_skinny:SKINNY", "engine.model:FUSED_COMMIT",
         "engine.model:FUSED_GDNVERIFY", "engine.model:FUSED_ADDNORM",
         "engine.model:TREE_HOST_DEPTH", "engine.model:VERIFY_GRAPH",
         "engine.model:COMMIT_IN_VERIFY", "engine.model:FUSED_ATTN_PREP"]
# the pending commit is the eager verify's; a graphed verify applies it before replaying,
# so the graph sections compare graphed and eager with it off on both sides
FOLD = "engine.model:COMMIT_IN_VERIFY"
# with --gdn-ab the graph sections also run with the fixed-order gate projections on
AB_FLAG = "engine.model:GDN_AB"
# flags whose ON path must equal the OFF path bit for bit (the arithmetic is not touched); the
# verify graphs need the commit and the mixer, so alone they change nothing
IDENTICAL_ON = {"engine.model:FUSED_ADDNORM", "engine.model:TREE_HOST_DEPTH",
                "engine.model:VERIFY_GRAPH", "engine.model:FUSED_ATTN_PREP"}

TREE_PARENTS = [-1, 0, 1, 2, 1, 4, 0, 6, 6, 8, 9, 10, 11, 12, 13, 14]


def _S(eng) -> torch.Tensor:
    """The state as the engine means it; with a commit pending that is not `state.S`."""
    f = getattr(eng, "committed_state", None)
    return (f() if f is not None else eng.state.S).float().cpu()


def scenario(eng, seed: int = 7, shift: int = 0) -> dict:
    """The fixed scenario. Returns every logit and the state after each step, on the CPU.
    `shift` lengthens the prompt, so the same blocks run at other positions."""
    from engine.tree import DraftTree
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 100000, (200 + shift,), generator=g).cuda()
    eng.reset()
    out = {}
    with torch.no_grad():
        out["prefill"] = eng.forward(ids, start=0, last_only=True).float().cpu()
        pos = 200 + shift
        blk = torch.randint(1000, 100000, (16,), generator=g).cuda()
        out["chain16"] = eng.forward_block(blk, start=pos).float().cpu()
        eng.rollback_to(6)
        pos += 6
        out["S_after_rollback"] = _S(eng)
        out["conv_after_rollback"] = eng.state.conv.float().cpu()
        blk = torch.randint(1000, 100000, (8,), generator=g).cuda()
        out["chain8"] = eng.forward_block(blk, start=pos).float().cpu()
        pos += 8                                              # a full accept: no rollback call
        blk = torch.randint(1000, 100000, (4,), generator=g).cuda()
        out["chain4"] = eng.forward_block(blk, start=pos).float().cpu()
        pos += 4                                              # and a second one in a row
        out["S_after_full"] = _S(eng)
        tree = DraftTree(tokens=torch.randint(1000, 100000, (len(TREE_PARENTS),),
                                              generator=g).tolist(), parents=TREE_PARENTS)
        tree.check()
        lg = eng.forward_tree(torch.tensor(tree.tokens).cuda(), tree.parents, start=pos)
        out["tree"] = lg.float().cpu()
        path = tree.path(15)                                  # the deepest leaf, 12 nodes
        eng.commit_tree(path)
        pos += len(path)
        out["S_after_tree"] = _S(eng)
        out["conv_after_tree"] = eng.state.conv.float().cpu()
        n = eng.kv.length
        out["kv_k"] = eng.kv.k[..., :n, :].float().cpu()
        out["decode"] = eng.forward(torch.tensor([int(out["tree"][path[-1]].argmax())]).cuda(),
                                    start=pos, last_only=True).float().cpu()
    return out


def graph_sweep(eng, seed: int = 13) -> dict:
    """Every chain length 2..16 and a dozen random tree shapes, each verified at two positions and
    committed -- the row counts and shapes the served loop reaches and `scenario` does not. Returns
    each verify's logits and the state after each commit."""
    import random
    from engine.tree import DraftTree
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    trees = []
    for n in list(range(3, 17)) * 1:
        parents, path = [-1], [0]
        for i in range(1, n):
            path = path[:rng.randint(1, len(path))]
            parents.append(path[-1])
            path.append(i)
        trees.append(parents)
    out = {}
    with torch.no_grad():
        for shift in (0, 53):
            ids = torch.randint(1000, 100000, (200 + shift,), generator=g).cuda()
            eng.reset()
            eng.forward(ids, start=0, last_only=True)
            pos = 200 + shift
            for T in range(2, 17):
                blk = torch.randint(1000, 100000, (T,), generator=g).cuda()
                out[f"chain{T}+{shift}"] = eng.forward_block(blk, start=pos).float().cpu()
                keep = 1 + (3 * T + 5) % T
                eng.rollback_to(keep)
                pos += keep
            for j, parents in enumerate(trees):
                toks = torch.randint(1000, 100000, (len(parents),), generator=g).tolist()
                tr = DraftTree(tokens=toks, parents=parents)
                if list(parents) == [-1] + list(range(len(parents) - 1)):
                    continue
                lg = eng.forward_tree(torch.tensor(toks).cuda(), parents, start=pos)
                out[f"tree{j}+{shift}"] = lg.float().cpu()
                path = tr.path(len(parents) - 1)
                eng.commit_tree(path)
                pos += len(path)
            out[f"S+{shift}"] = _S(eng)
    return out


def compare(a: dict, b: dict) -> tuple[bool, list[str]]:
    lines, same = [], True
    for k in a:
        x, y = a[k], b[k]
        eq = torch.equal(x, y)
        same &= eq
        if eq:
            lines.append(f"  {k:<20} bit-identical")
            continue
        d = (x - y).abs().max().item()
        rel = d / max(y.abs().max().item(), 1e-30)
        extra = ""
        if x.dim() >= 2 and x.shape[-1] > 1000:              # logits: argmax agreement
            agree = (x.argmax(-1) == y.argmax(-1)).float().mean().item()
            extra = f"  argmax agree {agree * 100:.1f} %"
        lines.append(f"  {k:<20} max|d| {d:.3e}  rel {rel:.2e}{extra}")
    return same, lines


def set_flags(on: set[str]) -> None:
    for spec in FLAGS:
        mod, attr = spec.split(":")
        try:
            m = importlib.import_module(mod)
        except ImportError:                    # an older checkout has none of the new modules
            continue
        setattr(m, attr, spec in on)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="")
    ap.add_argument("--flags", action="store_true")
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--gdn-ab", action="store_true",
                    help="run the graph identity sections with QWEN38_GDN_AB on in both arms")
    ap.add_argument("--from-env", action="store_true",
                    help="dump with every flag as the environment set it (the served "
                         "configuration) instead of forcing the kernel flags off. Phase 1's "
                         "identity (ops/gate.sh): the served engine with a new flag off is the "
                         "served engine it replaces, bit for bit")
    ap.add_argument("--graph-only", action="store_true",
                    help="with --flags: skip the per-flag section, run the graph identity only")
    ap.add_argument("--nvfp4", default=os.environ.get("QWEN38_NVFP4"))
    ap.add_argument("--fp8-head", default=os.environ.get("QWEN38_FP8_HEAD"))
    a = ap.parse_args()
    if a.compare:
        same, lines = compare(torch.load(a.compare[0]), torch.load(a.compare[1]))
        print("\n".join(lines))
        print(f"IDENTITY {'PASS' if same else 'FAIL'}: {a.compare[0]} vs {a.compare[1]}")
        sys.exit(0 if same else 1)

    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=1024)
    def present(spec: str) -> bool:
        mod, attr = spec.split(":")
        try:
            return hasattr(importlib.import_module(mod), attr)
        except ImportError:                    # the base commit has none of the new modules
            return False

    have = [f for f in FLAGS if present(f)]
    if have and not a.from_env:
        set_flags(set())
    scenario(eng)                                            # warm every kernel once
    base = scenario(eng)
    again = scenario(eng)
    same, _ = compare(base, again)
    print(f"determinism, {'flags as the environment set them' if a.from_env else 'flags off'}, "
          f"run twice: {'bit-identical' if same else 'DIFFERENT'}")
    if a.dump:
        torch.save(base, a.dump)
        print(f"dumped {a.dump}")
    if not a.flags:
        return
    fails = 0
    for spec in ([] if a.graph_only else FLAGS + ["all"]):
        on = set(FLAGS) if spec == "all" else {spec}
        set_flags(on)
        scenario(eng)
        got = scenario(eng)
        set_flags(set())
        same, lines = compare(got, base)
        want_same = spec in IDENTICAL_ON
        verdict = ("PASS" if same else "FAIL") if want_same else "measured"
        if want_same and not same:
            fails += 1
        print(f"--- {spec} on vs off: {'bit-identical' if same else 'differs'} "
              f"({'must be identical' if want_same else 'arithmetic changes'}): {verdict}")
        print("\n".join(lines))
    if not a.graph_only:
        print(f"FLAG-ON IDENTITY {'PASS' if fails == 0 else 'FAIL'} ({fails} failures)")

    # the verify graphs: with every other flag on, a graphed verify must be the eager
    # verify bit for bit -- at the position it was captured at and at another one
    graph = "engine.model:VERIFY_GRAPH"
    rest = set(FLAGS) - {graph, FOLD}
    if a.gdn_ab:
        import engine.model as M
        M.GDN_AB = True
        print("graph identity sections with QWEN38_GDN_AB on")
    set_flags(rest)
    eager = [scenario(eng, shift=0), scenario(eng, shift=37)]
    set_flags(rest | {graph})
    scenario(eng, shift=0)                                   # captures
    graphed = [scenario(eng, shift=0), scenario(eng, shift=37)]
    set_flags(set())
    gfail = 0
    for sh, g_, e_ in zip((0, 37), graphed, eager):
        same, lines = compare(g_, e_)
        gfail += not same
        print(f"--- graphed vs eager verify, prompt +{sh}: {'bit-identical' if same else 'DIFFERS'}")
        if not same:
            print("\n".join(lines))
    # and every row count and a dozen tree shapes, which the scenario does not reach
    set_flags(rest)
    e_sw = graph_sweep(eng)
    set_flags(rest | {graph})
    graph_sweep(eng)                                         # captures what is new
    g_sw = graph_sweep(eng)
    set_flags(set())
    same, lines = compare(g_sw, e_sw)
    gfail += not same
    diff = [ln for ln in lines if "bit-identical" not in ln]
    print(f"--- graphed vs eager, chains of 2..16 rows and 14 tree shapes at two positions: "
          f"{'bit-identical' if same else f'{len(diff)} of {len(lines)} differ'}")
    for ln in diff[:12]:
        print(ln)
    # the commit folded into the verify, graphed against eager with it on in both: the
    # graphs of both parities, and a pending commit carried from an eager block into a graphed
    # one and back
    import engine.model as M
    if hasattr(M, "COMMIT_IN_VERIFY"):
        set_flags(rest | {FOLD})
        e_f = [scenario(eng, shift=0), scenario(eng, shift=37), graph_sweep(eng)]
        set_flags(rest | {FOLD, graph})
        scenario(eng, shift=0)
        graph_sweep(eng)                                     # captures both parities
        g_f = [scenario(eng, shift=0), scenario(eng, shift=37), graph_sweep(eng)]
        set_flags(set())
        for name, g_, e_ in zip(("prompt +0", "prompt +37", "the sweep"), g_f, e_f):
            same, lines = compare(g_, e_)
            gfail += not same
            diff = [ln for ln in lines if "bit-identical" not in ln]
            print(f"--- folded commit, graphed vs eager, {name}: "
                  f"{'bit-identical' if same else f'{len(diff)} of {len(lines)} differ'}")
            for ln in diff[:8]:
                print(ln)
    gr = getattr(eng, "_graphs", None)
    print(f"verify graphs: {gr.stats if gr is not None else 'none'}")
    print(f"GRAPH IDENTITY {'PASS' if gfail == 0 else 'FAIL'}")
    sys.exit(1 if fails or gfail else 0)


if __name__ == "__main__":
    main()
