"""Flash-Next's joint verify against each state's lone verify, in one process (ENG-241 step 3).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_joint_check.py --pack ~/flashnext-packs/3.05 \
        --tokens 3000,7000,11000 --rounds 12 --json out.json

One engine (the weights once) and two states a prompt: one runs every round alone (`forward_tree`,
with its graphs), the other runs the same rounds together with the other prompts' states
(`forward_tree_multi`, one eager pass). Each round every state gets the same block: its own greedy
continuation as a chain of `--block` rows on even rounds, a tree with a second branch on odd rounds,
and commits the same path (the walk of its lone verify). Per round: whether every state's logits in
the joint pass are its lone logits bit for bit; at the end, each state's recurrent sections and KV
rows against its lone twin.

`--timing R`: then R more rounds of chain blocks, timed with a device sync around each: every state's
lone verify and commit from its graphs (what the slots do alone, summed), the same eagerly (no graphs),
and the joint pass with its commits; and the joint pass's GPU time by part (`FlashNextEngine.
joint_timing`: staging, PLE/norms/mixers, projections, GDN recurrence, attention, experts, head).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.methods import walk  # noqa: E402
from engine.models.flashnext.serving import FlashNextServing, build  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from tools.fn_generate import class_prompt  # noqa: E402


def block_of(cont: list[int], at: int, n: int, tree: bool) -> tuple[list[int], list[int]]:
    """Row 0 the next token, then the continuation as a chain; a tree adds a second child of row 1
    (a token that is not the continuation) before the chain goes on."""
    toks = cont[at:at + n]
    parents = list(range(-1, len(toks) - 1))
    if tree and len(toks) >= 3:
        toks = toks[:2] + [(toks[2] + 17) % 200000] + toks[2:]
        parents = [-1, 0, 1, 1] + list(range(3, len(toks) - 1))
    return toks, parents


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", default="3000,7000,11000")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--timing", type=int, default=0)
    ap.add_argument("--moe-group", action="store_true", help="the grouped expert kernels (tools/fn_kernels.py MOE_GROUP)")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.moe_group:
        from tools import fn_kernels
        fn_kernels.MOE_GROUP = True
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    lens = [int(x) for x in a.tokens.split(",")]
    L = max(lens) + (a.rounds + a.timing) * (a.block + 1) + 1024
    eng, _, _ = build(a.pack, L, tok, spec=False, log=lambda s: print(s, flush=True))
    fn = eng.fn
    prompts = [torch.tensor(class_prompt(tok, kind, n), device="cuda") for kind, n in zip(("prose", "code", "chat"), lens)]
    lone = [eng] + [FlashNextServing(fn, L) for _ in prompts[1:]]
    joint = [FlashNextServing(fn, L) for _ in prompts]
    eager = [FlashNextServing(fn, L) for _ in prompts] if a.timing else []
    conts, at = [], []
    for s_l, s_j, ids in zip(lone, joint, prompts):
        lg = s_l.forward(ids, 0)
        s_j.forward(ids, 0)
        if eager:
            eager[len(conts)].forward(ids, 0)
        # the continuation the blocks are made of: greedy from a third state
        tmp = FlashNextServing(fn, L)
        x = tmp.forward(ids, 0)
        cont = [int(x.view(-1).argmax())]
        for i in range((a.rounds + a.timing) * (a.block + 1)):
            x = tmp.forward(torch.tensor([cont[-1]], device="cuda"), ids.numel() + i)
            cont.append(int(x.view(-1).argmax()))
        del tmp
        conts.append(cont)
        at.append(0)
        assert int(lg.view(-1).argmax()) == cont[0]
    rounds = []
    for r in range(a.rounds):
        blocks = [block_of(conts[j], at[j], a.block, r % 2 == 1) for j in range(len(prompts))]
        lone_out, paths = [], []
        for j, s in enumerate(lone):
            toks, parents = blocks[j]
            lg = s.forward_tree(torch.tensor(toks, device="cuda"), parents, s.st.length).clone()
            path, new = walk.accept_tree(DraftTree(tokens=toks, parents=parents), lg.argmax(-1).tolist())
            s.commit_tree(path)
            lone_out.append(lg)
            paths.append(path)
        outs = FlashNextServing.forward_tree_multi(
            [(s, torch.tensor(blocks[j][0], device="cuda"), blocks[j][1], s.st.length) for j, s in enumerate(joint)])
        same = [bool(torch.equal(lone_out[j], outs[j])) for j in range(len(joint))]
        diff = [float((lone_out[j] - outs[j]).abs().max()) for j in range(len(joint))]
        for j, s in enumerate(joint):
            s.commit_tree(paths[j])
            at[j] += len(paths[j])
        rows = sum(len(b[0]) for b in blocks)
        rounds.append({"round": r, "rows": rows, "tree": r % 2 == 1, "equal": same, "max_diff": diff})
        print(f"[joint] round {r}: {rows} rows, equal {same}, max diff {diff}", flush=True)
    timing = None
    if a.timing:
        tl = te = tj = 0.0
        fn.joint_timing = {}
        for r in range(a.timing):
            blocks = [block_of(conts[j], at[j], a.block, False) for j in range(len(prompts))]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for j, s in enumerate(lone):
                s.forward_tree(torch.tensor(blocks[j][0], device="cuda"), blocks[j][1], s.st.length)
                s.commit_tree(list(range(len(blocks[j][0]))))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for j, s in enumerate(eager):                     # the engine's verify and commit, no graphs
                fn.verify(blocks[j][0], blocks[j][1], s.st)
                fn.commit(s.st, list(range(len(blocks[j][0]))))
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            FlashNextServing.forward_tree_multi(
                [(s, torch.tensor(blocks[j][0], device="cuda"), blocks[j][1], s.st.length) for j, s in enumerate(joint)])
            for j, s in enumerate(joint):
                s.commit_tree(list(range(len(blocks[j][0]))))
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            if r:                                               # the first round warms
                tl, te, tj = tl + t1 - t0, te + t2 - t1, tj + t3 - t2
            for j in range(len(prompts)):
                at[j] += len(blocks[j][0])
        n = max(1, a.timing - 1)
        jt = {k: round(v / fn.joint_timing["passes"], 2) for k, v in fn.joint_timing.items() if k != "passes"}
        timing = {"states": len(prompts), "rows": len(prompts) * a.block, "lone_graphs_ms": round(tl / n * 1e3, 2),
                  "lone_eager_ms": round(te / n * 1e3, 2), "joint_ms": round(tj / n * 1e3, 2), "joint_parts_ms": jt}
        fn.joint_timing = None
        print(f"[joint] timing {timing}", flush=True)
    end = []
    for s_l, s_j in zip(lone, joint):
        s_l._settle_chain()
        s_j._settle_chain()
        fn.flush()
        secs_l, secs_j = s_l.state.sections(), s_j.state.sections()
        bad = [k for k in secs_l if not torch.equal(secs_l[k], secs_j[k])]
        views = [n for (n, t, ax), (_, u, _) in zip(s_l.state.views(), s_j.state.views())
                 if not torch.equal(t.narrow(ax, 0, s_l.st.length), u.narrow(ax, 0, s_j.st.length))]
        end.append({"length": [s_l.st.length, s_j.st.length], "sections_differ": bad, "views_differ": views})
        print(f"[joint] end: {end[-1]}", flush=True)
    if a.json:
        json.dump({"rounds": rounds, "end": end, "timing": timing}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
