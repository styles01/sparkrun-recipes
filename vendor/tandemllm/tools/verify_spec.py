"""The M2 correctness gate: speculation must not change greedy output.

Runs the same prompts three ways -- no drafter, a drafter that is wrong on purpose, and the engram
drafter -- and requires the three token sequences to be identical. The adversarial drafter is the
one that matters: it drives the rollback path on nearly every block, at every possible accept
length, which a drafter that is usually right would almost never do.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.dflash2 import DFlash2Drafter  # noqa: E402
from engine.drafters.engram import EngramDrafter  # noqa: E402
from engine.drafters.fixed import AdversarialDrafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_greedy, generate_spec, generate_spec_tree  # noqa: E402

PROMPTS = {
    "prose": "Write three paragraphs about why the memory system, and not the arithmetic units, "
             "sets the speed of a language model that generates one token at a time.",
    "chat": "I have a machine with 121 GB of unified memory and about 273 GB/s of bandwidth. "
            "Explain in plain language what that means for running a 27-billion-parameter model, "
            "and what I should expect.",
    "code": "Write a Python function that reads a safetensors file header and prints every tensor "
            "name, dtype and shape, sorted by the number of bytes it occupies. Include a short "
            "docstring and a main guard.",
}


def bf16_ulp(x: float) -> float:
    """The distance between adjacent bf16 numbers at magnitude `x`.

    bf16 keeps 7 stored mantissa bits, so the spacing at magnitude 2^e is 2^(e-7): at a logit of 20
    it is 0.125. That number is not a detail. The engine's logits ARE bf16, so two tokens whose
    logits differ by one spacing are as close as this model can represent them being, and which of
    the two comes out of an argmax depends on the rounding of whatever matmul produced them.
    """
    import math
    if x == 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def compare(base: list[int], got: list[int], gaps: list[float], tok,
            tops: list[float] | None = None) -> tuple[bool, str]:
    """Does a speculative run reproduce the greedy run?

    With one exception, which the chat prompt of 2026-09-17 10:02 forced into the open. Greedy
    decoding is only defined where the argmax is: at a position where the top two logits are the
    SAME NUMBER there is no greedy answer, only whichever of the two a reduction happened to return
    first, and a block verify reduces over [T, V] where the one-token path reduces over [1, V].
    A run that follows the greedy run to an exact tie and then takes the other token has not broken
    losslessness. It has run out of greedy to be lossless about.

    So a mismatch is a failure unless the gap at that position is exactly zero, and the verdict says
    which of the two it was. Anything above zero is a real divergence, however small.
    """
    for i, (x, y) in enumerate(zip(base, got)):
        if x == y:
            continue
        g = gaps[i] if i < len(gaps) else float("nan")
        ulp = bf16_ulp(tops[i]) if tops and i < len(tops) else 0.0
        where = (f"at {i}: greedy {x} ({tok.decode([x])!r}) vs spec {y} ({tok.decode([y])!r}), "
                 f"top1-top2 gap {g:.4f} = {g / ulp:.2f} bf16 ulp" if ulp else
                 f"at {i}: greedy {x} vs spec {y}, top1-top2 gap {g:.4f}")
        if g == 0.0:
            return True, f"follows greedy to an exact logit tie, {where}"
        if g <= ulp:
            return True, f"follows greedy to a one-ulp logit tie, {where}"
        return False, f"DIVERGES {where}"
    return True, "identical"


def _refuse_if_service_is_up() -> None:
    """The one-engine rule, mechanically.

    A second model process beside the :8000 service has wedged sshd twice (2026-09-18), at ~70 GB
    resident and with no large arena involved. The service is the box's job; a tool that needs an
    engine must run while it is stopped. Refuse loudly instead of starting a second one.
    """
    import urllib.request
    for port in (8000,):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as fh:
                if fh.status == 200:
                    raise SystemExit(
                        f"[guard] REFUSING to start an engine: the service answers on :{port}. "
                        "Stop it first (ops/stop.sh) and arm .watchdog.off, or the box runs two "
                        "engines and wedges sshd.")
        except SystemExit:
            raise
        except Exception:
            pass


def build(args):
    _refuse_if_service_is_up()
    cfg = load_config(args.model)
    w = Weights(cfg.path, device=args.device, skip_mtp=False)
    eng = Qwen38Engine(cfg, w, max_len=args.max_len, device=args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    return cfg, w, eng, tok


def encode(tok, text: str, chat: bool, device: str, think: bool = True) -> torch.Tensor:
    if chat:
        text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True,
                                       **({} if think else {"enable_thinking": False}))
    return tok(text, return_tensors="pt").input_ids[0].to(device)


def tree_router(eng, a, narrow: bool = True):
    """The server's `--tree` length router (server/app.py), rebuilt here for the gate; with
    `narrow=False` the wide arm alone (`--narrow-arm off`, SPD-75)."""
    from engine.drafters.ngram import NgramDrafter
    from engine.lenrouter import LengthRouter
    from engine.router import MergedRouter, served_tree_table, tree_nodes
    small = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=1, path=a.dflash2_path,
                           max_len=a.max_len, block=8)
    large = DFlash2Drafter(eng, os.path.expanduser(a.lenrouter), blocks=1,
                           path=a.dflash2_path, max_len=a.max_len, block=16)
    small._build()
    large._build()
    tree_table = served_tree_table()
    ng = NgramDrafter(corpus_path=a.corpus, min_order=3, max_depth=16,
                      node_budget=large.cfg.block_size - 1, branch_top_k=3,
                      min_expected=0.2, alpha=0.6, corpus_weight=0.5, min_corpus_order=8,
                      verify_base_ms=tree_table[8],
                      verify_per_node_ms=(tree_table[16] - tree_table[8]) / 8)
    arms = [MergedRouter(ng, head, mtp_depth=head.cfg.block_size - 1,
                         node_budget=tree_nodes(head.cfg.block_size) - 1, mtp_ms_per_token=0.0,
                         head_fixed_ms=27.0, adaptive_depth=False, rollback_ms=6.4,
                         verify_ms_table=dict(tree_table), tree_ms_table=dict(tree_table))
            for head in (small, large)]
    if not narrow:
        from engine.lenrouter import NoArm
        arms[0] = NoArm(eng, block=small.cfg.block_size)
    return LengthRouter(arms[0], arms[1], tree=True, ngram=ng, latch=True,
                        drop_idle=a.drop_idle, deep=a.deep, deep_after=a.deep_after,
                        latch_table=dict(tree_table), fixed=a.len_fixed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="run the whole gate under a repetition penalty: the rule is "
                         "applied to the target logits in BOTH the plain and the speculative "
                         "loops, so the losslessness gate holds under the rule")
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--logit-bias", default="",
                    help="run the whole gate under a logit bias, KEY:VALUE pairs separated "
                         "by commas; a KEY is a token id or a piece of text whose first token is "
                         "biased (e.g. ' the:-6,' and:4'). Applied like the penalties, to the "
                         "target rows of both loops")
    ap.add_argument("--constraint", default="",
                    help="run the whole gate under a structured-output constraint, "
                         "`json_object` or a regex, applied like the penalties to the target rows "
                         "of both loops. The --chat prompts are rendered with thinking off, so the "
                         "answer, where the constraint acts, starts at once")
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--dflash2", default=None,
                    help="also gate the block drafter, at this many chained blocks")
    ap.add_argument("--dflash2-path", default="greedy", choices=["greedy", "viterbi"])
    ap.add_argument("--dflash2-head", default=None)
    ap.add_argument("--dflash2-ckpt", default=None,
                    help="gate a fine-tuned drafter. Losslessness is by construction -- a drafter "
                         "only proposes -- but a drafter trained by this repository is exactly the "
                         "kind of thing that should be made to prove it anyway")
    ap.add_argument("--dflash2-block", type=int, default=0)
    ap.add_argument("--lenrouter", default=None,
                    help="the sixteen-wide checkpoint. With --dflash2-ckpt as the eight-wide one, "
                         "this also gates engine/lenrouter.py: a router that picks the block "
                         "length per step must still write what the unspeculated loop writes")
    ap.add_argument("--tree-router", action="store_true",
                    help="with --lenrouter: also gate the router the server builds for --tree "
                         "(lookup drafter + both arms merged, the latch), on the tree loop")
    ap.add_argument("--deep-after", type=int,
                    default=int(os.environ.get("QWEN38_DEEP_AFTER", "2")),
                    help="with --deep: full blocks in a row before a deep chain")
    ap.add_argument("--deep", type=int, default=int(os.environ.get("QWEN38_DEEP", "0")),
                    help="with --tree-router: the deep chain, up to this many rows (0 = off)")
    ap.add_argument("--corpus", default="", help="the lookup drafter's corpus, for --tree-router")
    ap.add_argument("--len-fixed", type=int, default=0,
                    help="with --tree-router: pin the router to one width (8 or 16), as the "
                         "server's --len-fixed does")
    ap.add_argument("--extra-prompts", action="store_true",
                    help="add the bench's quote and edit prompts, the two that fill a wide block")
    ap.add_argument("--drop-idle", action="store_true",
                    help="gate the phase-10 released arm: --lenrouter with the latch on and the "
                         "arm that loses it released. It changes which drafter proposes, including "
                         "for the last block of a generation, and a change to which drafter "
                         "proposes must still write what the unspeculated loop writes")
    ap.add_argument("--narrow-arm", default="on", choices=("on", "off", "both"),
                    help="with --tree-router: the served router with the block-8 arm (on), the "
                         "wide arm alone (off, SPD-75; needs QWEN38_LEN_MODE=wide), or both runs")
    a = ap.parse_args()

    cfg, w, eng, tok = build(a)
    ok = True
    from engine.penalty import PenaltySpec, PenaltyState
    bias = {}
    for pair in filter(None, a.logit_bias.split(",")):
        key, _, val = pair.rpartition(":")
        tid = int(key) if key.strip().lstrip("-").isdigit() else \
            tok(key, add_special_tokens=False).input_ids[0]
        bias[tid] = float(val)
    spec = PenaltySpec(a.rep_penalty, a.presence_penalty, a.frequency_penalty,
                       a.no_repeat_ngram, bias=bias)
    pen = (PenaltyState(spec, cfg.vocab_size, a.device) if spec.on else None)
    if bias:
        print(f"[gate] logit bias {spec.bias}", flush=True)
    if a.constraint:
        import json as _json
        from engine import grammar as G
        pattern = G.json_object_regex(3) if a.constraint == "json_object" else a.constraint
        eos = {tok.eos_token_id} if isinstance(tok.eos_token_id, int) else set()
        gen = os.path.join(cfg.path, "generation_config.json")
        if os.path.isfile(gen):
            e = _json.load(open(gen)).get("eos_token_id")
            eos |= set(e) if isinstance(e, list) else ({e} if isinstance(e, int) else set())
        gram = G.Grammar(pattern, G.Vocab.from_tokenizer(tok, cfg.vocab_size))
        cons = G.Constraint(gram, eos, a.device)
        pen = cons if pen is None else G.LogitChain([pen, cons])
        print(f"[gate] constraint {a.constraint[:60]!r}: {gram.dfa.states} states, eos {sorted(eos)}",
              flush=True)

    prompts = dict(PROMPTS)
    if a.extra_prompts:
        from tools.bench_decode import PROMPTS as BENCH
        prompts.update(quote=BENCH["quote"], edit=BENCH["edit"])
    for name, text in prompts.items():
        if a.only and name != a.only:
            continue
        ids = encode(tok, text, a.chat, a.device, think=not a.constraint)
        base, sb = generate_greedy(eng, ids, a.new, record_gaps=True, pen=pen)
        ties = sum(1 for g, t in zip(sb.gaps, sb.tops) if g <= bf16_ulp(t))
        print(sb.line(f"{name}/no drafter"))
        print(f"    greedy top1-top2 logit gap: median {sorted(sb.gaps)[len(sb.gaps) // 2]:.3f}, "
              f"minimum {min(sb.gaps):.4f}, positions within one bf16 ulp "
              f"{ties} of {len(sb.gaps)}")

        adv = AdversarialDrafter(cfg.vocab_size, seed=1)
        got, sa = generate_spec(eng, ids, a.new, adv, a.k, pen=pen)
        print(sa.line(f"{name}/adversarial"))
        same, why = compare(base, got, sb.gaps, tok, sb.tops)
        ok &= same
        print(f"    adversarial: {why}")

        # a drafter that is right some of the time, to exercise partial accepts of every length
        half = AdversarialDrafter(cfg.vocab_size, seed=2, truth=base, accept_prefix=a.k // 2)
        got2, sh = generate_spec(eng, ids, a.new, half, a.k, pen=pen)
        print(sh.line(f"{name}/half-right"))
        same2, why2 = compare(base, got2, sb.gaps, tok, sb.tops)
        ok &= same2
        print(f"    half-right:  {why2}")

        if a.dflash2:
            nb = int(a.dflash2)
            dd = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=nb, path=a.dflash2_path,
                                draft_head=a.dflash2_head, max_len=a.max_len,
                                block=a.dflash2_block or None)
            width = (dd.cfg.block_size - 1) * nb
            got4, sd = generate_spec(eng, ids, a.new, dd, width, pen=pen)
            dd.detach()
            print(sd.line(f"{name}/dflash2 b={nb}"))
            same4, why4 = compare(base, got4, sb.gaps, tok, sb.tops)
            ok &= same4
            print(f"    dflash2:     {why4}")

        if a.lenrouter:
            from engine.lenrouter import LengthRouter
            small = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=1, path=a.dflash2_path,
                                   max_len=a.max_len, block=8)
            large = DFlash2Drafter(eng, os.path.expanduser(a.lenrouter), blocks=1,
                                   path=a.dflash2_path, max_len=a.max_len, block=16)
            # `explore_period=1` forces a wide probe on every other block, so the run exercises
            # both widths and both rollback lengths rather than settling into whichever one this
            # prompt happens to pay for. A gate wants the paths, not the policy.
            #
            # `--drop-idle` is the exception, because there the POLICY is the path under test: the
            # release only happens when the latch closes, so the gate has to run the latch.
            lr = (LengthRouter(small, large, latch=True, drop_idle=True) if a.drop_idle
                  else LengthRouter(small, large, explore_period=1))
            got5, sl = generate_spec(eng, ids, a.new, lr, large.cfg.block_size - 1, pen=pen)
            lr.detach()
            print(sl.line(f"{name}/lenrouter" + (" drop-idle" if a.drop_idle else "")))
            print(f"      {lr.report()}")
            same5, why5 = compare(base, got5, sb.gaps, tok, sb.tops)
            ok &= same5
            print(f"    lenrouter:   {why5}")

        for narrow in ((True, False) if a.narrow_arm == "both" else (a.narrow_arm == "on",)):
            if not (a.lenrouter and a.tree_router):
                break
            # The router as the server builds it for `--tree`: both arms a MergedRouter over one
            # lookup drafter, the latch, and optionally the deep chain (`--deep`). The chain gate
            # above does not reach this path at all -- it has no lookup drafter and no tree.
            lr6 = tree_router(eng, a, narrow=narrow)
            got6, s6 = generate_spec_tree(eng, ids, a.new, lr6, max(15, a.deep - 1), pen=pen)
            lr6.detach()
            tag = "" if narrow else " no-narrow-arm"
            print(s6.line(f"{name}/tree-router" + (f" deep={a.deep}" if a.deep else "") + tag))
            print(f"      {lr6.report()}")
            same6, why6 = compare(base, got6, sb.gaps, tok, sb.tops)
            ok &= same6
            print(f"    tree-router{tag}: {why6}")

        eg = EngramDrafter()
        got3, se = generate_spec(eng, ids, a.new, eg, a.k, pen=pen)
        print(se.line(f"{name}/engram"))
        same3, why3 = compare(base, got3, sb.gaps, tok, sb.tops)
        ok &= same3
        print(f"    engram:      {why3}")
        print(f"    engram hit rate {eg.stats['hits']}/{eg.stats['calls']}, "
              f"orders {eg.stats['order_hist']}")
        print()

    from engine.model import ROLLBACK_DIFF
    if ROLLBACK_DIFF:
        worst = max(ROLLBACK_DIFF, key=lambda r: r[2])
        rel = [d / max(m, 1e-9) for _, _, d, m in ROLLBACK_DIFF]
        print(f"rank-k vs replay rollback over {len(ROLLBACK_DIFF)} layer-rollbacks: "
              f"worst absolute {worst[2]:.3e} (layer {worst[0]}, keep {worst[1]}, "
              f"state absmax {worst[3]:.3f}), worst relative {max(rel):.3e}, "
              f"mean relative {sum(rel) / len(rel):.3e}")
    print("=" * 78)
    print("GATE  speculation changes greedy output only where the top two logits are within one")
    print(f"      bf16 ulp of each other, which is as close as this engine can represent them: "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
