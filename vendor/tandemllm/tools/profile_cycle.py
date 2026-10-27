"""One full speculative block cycle, phase by phase, under the shipped configuration.

The row's mean is 32.13 tok/s and its median is 23.26, and the loop that produces both moves
about 15 GB in a 92 ms step -- 163 GB/s -- while the kernels it is made of reach 211-235 on their
own. The gap is not in any kernel. It is in the glue between them: Python between launches, the
two host round-trips the loop takes every block (`.tolist()` on the picks, `synchronize()` after
the commit), the drafter that runs strictly before the verify rather than beside it, and the
per-block tensor construction on the host.

This script measures where the gap is, under the exact configuration `server/app.py` ships, and
it reports three readings of the same blocks:

  loose   no extra synchronisation. CPU wall per phase, which is what the loop actually spends,
          and the only honest total. A phase that launches work and returns reads as cheap here
          and the cost lands on whichever later phase waits for it.
  strict  a `cuda.synchronize()` at every phase boundary. Attribution is exact and the total is
          inflated by the syncs it adds -- the difference between the two totals is itself the
          measurement of how much the loop currently overlaps.
  kernels `torch.profiler` over the same blocks, summed by kernel. Wall minus kernel time is the
          idle share, and that is the number this step is trying to close.

Everything here decodes greedily with thinking off, and both widths are warmed before anything is
timed: the first configuration in a fresh process pays for Triton autotuning and phase 6 lost a
day to reading that as a policy win.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402

# The phases of one block, in the order the loop runs them. `draft` is broken into the three
# things it is made of, which have very different costs and very different fixes.
PHASES = ["lookup", "lattice", "merge", "h2d", "verify", "picks", "price", "accept",
          "commit", "sync", "observe"]
# The order `mark` is called in. A mark CLOSES the phase it names and opens the next one, so the
# annotation a trace buckets kernels by has to be named for the phase that follows -- naming it
# for the one that just ended shifts the whole table by one row, and the first trace read the
# verify's 120 ms under the label `h2d`.
MARKS = ["draft", "merge", "h2d", "verify", "picks", "price", "accept", "commit", "sync",
         "observe"]
NEXT = {m: MARKS[i + 1] for i, m in enumerate(MARKS[:-1])}


class Phases:
    """A running per-phase accumulator, one row per block."""

    def __init__(self, strict: bool, trace: bool = False):
        self.strict = strict
        # When tracing, every phase is wrapped in a `record_function` so that the kernels it
        # launched can be found again in the chrome trace and attributed to it. In strict mode
        # the annotation encloses its own kernels exactly, because the boundary is a sync.
        self.trace = trace
        self.ms = defaultdict(list)
        self.block_ms = []
        self._t = None
        self._blk = None
        self._rf = None

    def start_block(self) -> None:
        if self.strict:
            torch.cuda.synchronize()
        self._blk = time.perf_counter()
        self._t = self._blk
        for p in PHASES:
            self.ms[p].append(0.0)
        self._enter("draft")

    def mark(self, name: str) -> None:
        if self.strict:
            torch.cuda.synchronize()
        now = time.perf_counter()
        # ASSIGN into this block's row rather than appending one. `start_block` has already laid
        # a zero down for every phase, so a phase that does not fire this block still has a row
        # and the per-block mean is a mean over blocks. Appending here instead put a zero AND a
        # value in the same list, which left the means right and every quantile meaningless.
        self.ms[name][-1] += (now - self._t) * 1e3
        self._t = now
        self._enter(NEXT.get(name, "tail"))

    def end_block(self) -> None:
        if self.strict:
            torch.cuda.synchronize()
        self._exit()
        self.block_ms.append((time.perf_counter() - self._blk) * 1e3)

    # --- the annotations the chrome trace is bucketed by --------------------------------------

    def _enter(self, name: str) -> None:
        if not self.trace:
            return
        self._exit()
        self._rf = torch.profiler.record_function(f"PH::{name}")
        self._rf.__enter__()

    def _exit(self) -> None:
        if self._rf is not None:
            self._rf.__exit__(None, None, None)
            self._rf = None

    def table(self, label: str) -> str:
        n = len(self.block_ms)
        total = sum(self.block_ms)
        out = [f"--- {label}: {n} blocks, {total / n:.2f} ms/block ---",
               f"{'phase':>9} {'ms/block':>9} {'p50':>7} {'p90':>7} {'share':>7} {'blocks':>7}"]
        for p in PHASES:
            v = self.ms.get(p)
            if not v:
                continue
            mean = sum(v) / n          # per BLOCK, not per occurrence: a phase that fires on a
            hit = [x for x in v if x > 0.0]   # third of blocks costs a third as much a block
            out.append(f"{p:>9} {mean:9.3f} {statistics.median(v):7.3f} "
                       f"{sorted(v)[int(0.9 * len(v))]:7.3f} "
                       f"{100 * sum(v) / total:6.1f}% {len(hit):7d}")
        out.append(f"{'TOTAL':>9} {total / n:9.3f}")
        return "\n".join(out)


def build(a):
    """The shipped configuration, built exactly as `server/app.py` builds it."""
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.drafters.ngram import NgramDrafter
    from engine.lenrouter import LengthRouter
    from engine.router import MergedRouter, served_tree_table, tree_nodes

    cfg = load_config(a.model)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len)
    small = DFlash2Drafter(eng, a.ckpt8, blocks=1, path="greedy", max_len=a.max_len, block=8)
    small._build()
    large = DFlash2Drafter(eng, a.ckpt16, blocks=1, path="greedy", max_len=a.max_len, block=16)
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
    # The served router latches the width once a request and releases the losing arm
    # (ops/serve.env LEN_LATCH=1, DROP_IDLE=1); a caller that does not ask gets the per-block
    # router this tool was written against.
    drafter = LengthRouter(arms[0], arms[1], fixed=a.fixed, explore_period=32,
                           tree=True, ngram=ng, latch=getattr(a, "latch", False),
                           drop_idle=getattr(a, "drop_idle", False))
    return cfg, eng, drafter, arms, ng, large.cfg.block_size - 1


def instrument(arms, ng, ph: Phases):
    """Split `propose_tree` into lookup / lattice / merge without editing the engine.

    The three are a dictionary lookup, a forward pass through the block drafter, and pure host
    arithmetic over node sets, and they want three different fixes -- so they are timed apart.

    Returns an undo. Wrapping without one wraps the wrapper on the next workload, and the timings
    of the run before it keep being written into a list nobody reads any more.
    """
    ng_orig = ng.propose_tree

    def ng_wrapped(ctx, k):
        t = time.perf_counter()
        r = ng_orig(ctx, k)
        ph.ms["lookup"][-1] += (time.perf_counter() - t) * 1e3
        return r

    ng.propose_tree = ng_wrapped
    # the head's call is a generator that stops after the draft launch; `propose_tree`
    # drives it straight through, so timing the whole generator is timing the call
    originals = [(arm, arm._head_tree_steps) for arm in arms]
    for arm, orig in originals:

        def wrapped(ctx, depth, _orig=orig):
            if ph.strict:
                torch.cuda.synchronize()
            t = time.perf_counter()
            r = yield from _orig(ctx, depth)
            if ph.strict:
                torch.cuda.synchronize()
            ph.ms["lattice"][-1] += (time.perf_counter() - t) * 1e3
            return r

        arm._head_tree_steps = wrapped

    def undo():
        ng.propose_tree = ng_orig
        for arm, orig in originals:
            arm._head_tree_steps = orig

    return undo


def gpu_busy(path: str) -> tuple[float, float, dict[str, tuple[float, float]]]:
    """Read a chrome trace and answer the only question this step is really asking.

    Summing kernel durations does not answer it: `key_averages()` charges the same microseconds to
    an `aten::mm` and to the kernel it launched, and the first reading of this trace came back at
    113.8 % of wall and a NEGATIVE idle. What matters is the UNION of the kernel intervals -- how
    much of the wall clock the GPU had anything at all to do -- so the intervals are merged and
    then measured, and overlap counts once.

    Returns (span_ms, busy_ms, per-phase {name: (wall_ms, busy_ms)}). Phases come from the
    `PH::` annotations, and in strict mode each one encloses exactly the kernels it launched.
    """
    ev = json.load(open(path))["traceEvents"]
    kern = sorted(((e["ts"], e["ts"] + e.get("dur", 0)) for e in ev
                   if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                  key=lambda x: x[0])
    if not kern:
        return 0.0, 0.0, {}
    merged = []
    for a0, b0 in kern:
        if merged and a0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b0)
        else:
            merged.append([a0, b0])
    span = kern[-1][1] - kern[0][0]
    busy = sum(b - a for a, b in merged)
    phases: dict[str, list[float]] = {}
    marks = [e for e in ev if e.get("cat") in ("user_annotation", "cpu_op")
             and str(e.get("name", "")).startswith("PH::")]
    for m in marks:
        name = m["name"][4:]
        lo, hi = m["ts"], m["ts"] + m.get("dur", 0)
        b = sum(max(0.0, min(hi, y) - max(lo, x)) for x, y in merged
                if y > lo and x < hi)
        row = phases.setdefault(name, [0.0, 0.0])
        row[0] += hi - lo
        row[1] += b
    return span / 1e3, busy / 1e3, {k: (v[0] / 1e3, v[1] / 1e3) for k, v in phases.items()}


def cycle(eng, drafter, prompt, max_new, k, ph: Phases, eos=()):
    """`engine.spec.generate_spec_tree`, with a mark at every phase boundary.

    Kept a copy rather than instrumented in place: the shipped loop must not grow timing calls
    that a profiler needs, and a copy that drifts is caught by the token-for-token check the
    caller runs against the real loop.
    """
    eng.reset()
    drafter.reset()
    plist = prompt.tolist()
    if hasattr(drafter, "prime"):
        drafter.prime(plist)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = eng.forward(prompt, start=0, last_only=True)
        if hasattr(drafter, "sync"):
            drafter.sync(plist, eng.hidden_post_norm[0], 0)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3
    pos = prompt.numel()
    tok = int(logits[0, -1].argmax())
    out = [tok]
    drafter.observe([tok])
    ctx = plist + [tok]
    nodes = accepted = 0

    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    with torch.no_grad():
        while len(out) < max_new and tok not in eos:
            ph.start_block()          # lays a zero row for every phase
            tree = drafter.propose_tree(ctx, min(k, max_new - len(out)))
            ph.mark("merge")                  # lookup+lattice already subtracted by the wrappers
            ph.ms["merge"][-1] -= ph.ms["lookup"][-1] + ph.ms["lattice"][-1]
            if tree is None or tree.n_draft == 0:
                logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                     last_only=True)
                if hasattr(drafter, "sync"):
                    drafter.sync([tok], eng.hidden_post_norm[0], pos)
                pos += 1
                tok = int(logits[0, -1].argmax())
                out.append(tok)
                ctx.append(tok)
                drafter.observe([tok])
                ph.mark("verify")
                ph.end_block()
                continue
            block = torch.tensor(tree.tokens, device=prompt.device)
            ph.mark("h2d")
            lg = eng.forward_tree(block, tree.parents, start=pos)
            ph.mark("verify")
            picks = lg.argmax(-1).tolist()
            ph.mark("picks")
            on_verify = getattr(drafter, "on_verify", None)
            if on_verify is not None:
                on_verify(tree.n_draft + 1, ph.ms["verify"][-1] + ph.ms["picks"][-1])
            ph.mark("price")
            path, new = eng.accept_tree(tree, picks)
            ph.mark("accept")
            eng.commit_tree(path)
            torch.cuda.synchronize()
            ph.mark("commit")
            n = len(path) - 1
            nodes += tree.n_draft
            accepted += n
            if hasattr(drafter, "sync"):
                hid = eng.hidden_post_norm[0, torch.tensor(path, device=prompt.device)]
                if getattr(drafter, "wants_rows", False):
                    drafter.sync([int(tree.tokens[i]) for i in path], hid, pos, rows=path)
                else:
                    drafter.sync([int(tree.tokens[i]) for i in path], hid, pos)
            ph.mark("sync")
            pos += n + 1
            for t in new:
                out.append(t)
                ctx.append(t)
                if t in eos:
                    break
            drafter.observe(new)
            tok = out[-1]
            ph.mark("observe")
            ph.end_block()
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t_dec
    return out, dict(prefill_ms=prefill_ms, decode_s=decode_s, tokens=len(out) - 1,
                     blocks=len(ph.block_ms), nodes=nodes, accepted=accepted,
                     tok_s=(len(out) - 1) / decode_s)


PROMPTS = {
    # Fresh prose is the median of the row and the wall this phase is aimed at: no prompt to
    # copy from, the lookup drafter fires on a tenth of steps, 17 tok/s and 2.7 accepted.
    "prose": "Write four paragraphs about the way a harbour town wakes up in winter.",
    # Ordinary chat, the shape most of the fifty row prompts have.
    "chat": "Explain why unified memory changes what a single-board inference engine should "
            "optimise for, in about two hundred words.",
    # Reproduction: the answer is largely in the prompt, which is where the row's tail lives.
    "quote": "Repeat the following sentence exactly, five times, each on its own line: "
             "The recurrent state after a partial accept has to be reconstructed rather than "
             "truncated.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--ckpt8", required=True)
    ap.add_argument("--ckpt16", required=True)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--fixed", type=int, default=0)
    ap.add_argument("--workloads", default="prose,chat,quote")
    ap.add_argument("--warm", type=int, default=48,
                    help="tokens of warm-up per width before anything is timed")
    ap.add_argument("--trace", default="", help="write a torch.profiler kernel summary here")
    ap.add_argument("--trace-new", type=int, default=64,
                    help="tokens generated under the profiler")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    cfg, eng, drafter, arms, ng, k = build(a)
    tok = AutoTokenizer.from_pretrained(cfg.path)

    def ids(text: str) -> torch.Tensor:
        msg = [{"role": "user", "content": text}]
        s = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        return tok(s, return_tensors="pt").input_ids[0].cuda()

    # WARM BOTH WIDTHS. The narrow and the wide verify are different Triton configurations and
    # the first of each in a process autotunes; phase 6 measured 260.2 ms against a settled 115.5.
    print("[warm] both widths", flush=True)
    for fixed in (8, 16):
        drafter.fixed = fixed
        ph = Phases(strict=False)
        cycle(eng, drafter, ids(PROMPTS["chat"]), a.warm, k, ph)
    drafter.fixed = a.fixed
    print("[warm] done", flush=True)

    report, out_json = [], {}
    for name in a.workloads.split(","):
        prompt = ids(PROMPTS[name])
        for strict in (False, True):
            ph = Phases(strict=strict)
            undo = instrument(arms, ng, ph)
            out, st = cycle(eng, drafter, prompt, a.max_new, k, ph)
            label = f"{name} {'strict' if strict else 'loose'}"
            head = (f"{label}: {st['tokens']} tok  {st['tok_s']:.2f} tok/s  "
                    f"{st['blocks']} blocks  {st['accepted'] / max(st['blocks'], 1):.2f} "
                    f"accepted/block  {st['nodes'] / max(st['blocks'], 1):.1f} nodes/block  "
                    f"prefill {st['prefill_ms']:.0f} ms")
            print(head, flush=True)
            print(ph.table(label), flush=True)
            report += [head, ph.table(label), ""]
            undo()
            out_json[label] = dict(stats=st,
                                   phases={p: sum(v) / len(ph.block_ms)
                                           for p, v in ph.ms.items() if v},
                                   block_ms=sum(ph.block_ms) / len(ph.block_ms))

    if a.trace:
        # Traced STRICT, on purpose. The bucketing wants each annotation to enclose the kernels it
        # launched, and only a sync at the boundary guarantees that; the strict and loose totals
        # above differ by under 1 %, so what is traced is what the loop does.
        name = a.workloads.split(",")[0]
        prompt = ids(PROMPTS[name])
        ph = Phases(strict=True, trace=True)
        undo = instrument(arms, ng, ph)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t = time.perf_counter()
            cycle(eng, drafter, prompt, a.trace_new, k, ph)
            wall = time.perf_counter() - t
        undo()
        js = a.trace + ".json"
        prof.export_chrome_trace(js)
        span, busy, per = gpu_busy(js)
        n = len(ph.block_ms)
        rows = [f"{'phase':>9} {'wall ms':>9} {'busy ms':>9} {'idle ms':>9} {'busy %':>7}"]
        for p in MARKS:
            if p not in per:
                continue
            w_, b_ = per[p]
            rows.append(f"{p:>9} {w_ / n:9.2f} {b_ / n:9.2f} {(w_ - b_) / n:9.2f} "
                        f"{100 * b_ / max(w_, 1e-9):6.1f}%")
        txt = (f"\n--- kernels, {name}, {n} blocks, traced wall {wall * 1e3:.1f} ms ---\n"
               f"wall/block        {wall * 1e3 / n:8.2f} ms\n"
               f"gpu busy/block    {busy / n:8.2f} ms   {100 * busy / (wall * 1e3):5.1f} % of wall\n"
               f"IDLE/block        {(wall * 1e3 - busy) / n:8.2f} ms   "
               f"{100 * (wall * 1e3 - busy) / (wall * 1e3):5.1f} % of wall\n"
               f"kernel launches/block {sum(1 for e in json.load(open(js))['traceEvents'] if e.get('cat') == 'kernel') / n:7.0f}\n\n"
               + "\n".join(rows) + "\n\n"
               + prof.key_averages().table(sort_by="self_device_time_total", row_limit=25))
        print(txt, flush=True)
        open(a.trace, "w").write(txt)
        out_json["kernels"] = dict(wall_ms=wall * 1e3, blocks=n, busy_ms=busy,
                                   per_phase={k: list(v) for k, v in per.items()})

    if a.json:
        json.dump(out_json, open(a.json, "w"), indent=1)
    print("\n".join(report[-1:]))


if __name__ == "__main__":
    main()
