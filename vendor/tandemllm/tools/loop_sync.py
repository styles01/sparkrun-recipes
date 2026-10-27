"""The served loop's host synchronisations per round and its tokens under the flags.

`tools/bubbles.py` and `tools/block_budget.py` drive `tools/profile_cycle.cycle`, a copy of the bench
loop; changed `server/app.py::generate_stream`, the loop the server runs. This drives that
loop in one process on the real model -- the served drafter built as the server builds it
(`profile_cycle.build` plus the deep chain from the environment), `server.app.STATE` holding what
the loop reads -- over the five workloads (tools/bench_decode.py, thinking off), under each flag
combination:

    off     the rc4 loop           (QWEN38_LAUNCH_FIRST=0, QWEN38_HOST_ASYNC=0)
    async   QWEN38_HOST_ASYNC=1    pinned copies, one draft read-back, the verify graph's argmax
    first   QWEN38_LAUNCH_FIRST=1  the next draft launched before the block is streamed
    both

Pass 1 (timing, nothing instrumented): the tokens of every combination against `off`'s, which must
be identical, and ms a round (the loop's own BlockStats). Pass 2 (counting): every synchronising
PyTorch call the host makes after the prefill -- `torch.cuda.set_sync_debug_mode("warn")` reports
`.tolist()`, `.item()`, pageable copies and the rest -- plus explicit stream / event / device
synchronisations, per round and by call site. the acceptance: at most two a round.

    python tools/loop_sync.py --ckpt8 $CKPT8 --ckpt16 $CKPT16 --corpus $CORPUS --new 128 \\
        --out results/p5/loop_sync.json          (served environment, under the box lock)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in __import__("sys").path:
    # run as a script from tools/: the repo root, appended (lowest priority), for engine.settings
    __import__("sys").path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMBOS = {"off": (False, False), "async": (False, True), "first": (True, False),
          "both": (True, True)}


def _site(frame) -> str:
    return f"{os.path.relpath(frame.f_code.co_filename, os.getcwd())}:{frame.f_lineno}"


class SyncCounter:
    """Synchronisations the host makes while armed, by call site."""

    def __init__(self):
        import torch
        self.torch = torch
        self.sites: collections.Counter = collections.Counter()
        self.armed = False
        self._orig = []

    def install(self):
        torch = self.torch
        me = self

        def wrap(owner, name):
            orig = getattr(owner, name)

            def f(*a, **kw):
                if me.armed:
                    me.sites[f"{name}@{_site(sys._getframe(1))}"] += 1
                return orig(*a, **kw)
            setattr(owner, name, f)
            self._orig.append((owner, name, orig))

        wrap(torch.cuda.Stream, "synchronize")
        wrap(torch.cuda.Event, "synchronize")
        wrap(torch.cuda, "synchronize")
        torch.cuda.set_sync_debug_mode("warn")

    def uninstall(self):
        self.torch.cuda.set_sync_debug_mode("default")
        for owner, name, orig in reversed(self._orig):
            setattr(owner, name, orig)
        self._orig.clear()

    def take(self, caught) -> None:
        for w in caught:
            # an explicit synchronize is counted by its wrapper; the debug mode reports it again
            # from inside torch/cuda (streams.py), which is the same wait
            if "synchroniz" in str(w.message) and os.sep + "torch" + os.sep + "cuda" not in w.filename:
                self.sites[f"implicit@{os.path.relpath(w.filename, os.getcwd())}:{w.lineno}"] += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--nvfp4", default=_S.get("NVFP4"))
    ap.add_argument("--fp8-head", default=_S.get("FP8_HEAD"))
    ap.add_argument("--ckpt8", required=True)
    ap.add_argument("--ckpt16", required=True)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--combos", default="off,async,first,both")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.fixed, a.latch, a.drop_idle = 0, True, True

    import torch
    from transformers import AutoTokenizer

    import engine.model as M
    from engine.spec import Relax
    from server import app
    from tools import profile_cycle as pc
    from tools.bench_decode import PROMPTS

    cfg, eng, drafter, arms, ng, k = pc.build(a)
    # the served router's deep chain (ops/serve.env QWEN38_DEEP / _AFTER), which build() leaves off
    drafter.deep = int(_S.get("DEEP") or 0)
    drafter.deep_after = int(_S.get("DEEP_AFTER") or 2)
    k = max(k, drafter.deep - 1)
    tk = AutoTokenizer.from_pretrained(cfg.path)
    eos = {tk.convert_tokens_to_ids("<|im_end|>"), tk.convert_tokens_to_ids("<|endoftext|>")}
    app.STATE.clear()
    app.STATE.update(engine=eng, drafter=drafter, k=k, tree=True, sampled_tree=False,
                     relax=Relax(1.0, 1), verbose=False, state_store=None, prefix_cache=False,
                     prefix_chunk=0)
    prompts = {}
    for name, text in PROMPTS.items():
        s = tk.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
        prompts[name] = tk(s, return_tensors="pt").input_ids[0].cuda()

    def once(ids, combo, counter=None):
        app.LAUNCH_FIRST, M.HOST_ASYNC = COMBOS[combo]
        eng.reset()
        gen = app.generate_stream(ids, a.new, eos)
        out = [next(gen)]                          # the prefill's token
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if counter is not None:
                counter.armed = True
            for t in gen:
                out.append(t)
            if counter is not None:
                counter.armed = False
            torch.cuda.synchronize()
        if counter is not None:
            counter.take(caught)
        bs = app.STATE["blocks"]
        return out, bs.blocks, (bs.t_last - bs.t_first) * 1e3

    combos = a.combos.split(",")
    report = {"args": {**{k_: v for k_, v in vars(a).items()}, "k": k,
                       "served_env": sorted(f"{k_}={v}" for k_, v in os.environ.items()
                                            if k_.startswith("QWEN38_"))},
              "timing": {}, "syncs": {}, "identical": {}}
    ref = {}
    with torch.no_grad():
        # pass 1: warm every combination (graph captures), then the timed runs
        for combo in combos:
            for name, ids in prompts.items():
                once(ids, combo)
        for combo in combos:
            for name, ids in prompts.items():
                out, blocks, ms = once(ids, combo)
                ref.setdefault(name, out)
                same = out == ref[name]
                report["identical"].setdefault(combo, {})[name] = same
                report["timing"].setdefault(combo, {})[name] = {
                    "tokens": len(out), "blocks": blocks, "ms": round(ms, 2),
                    "ms_blk": round(ms / max(blocks, 1), 3),
                    "tok_blk": round((len(out) - 1) / max(blocks, 1), 3)}
                print(f"[loop_sync] {combo:<6} {name:<6} tokens {len(out):4d} blocks {blocks:3d} "
                      f"{ms / max(blocks, 1):7.2f} ms/blk  identical={same}", flush=True)
        # pass 2: the synchronisations
        for combo in combos:
            per = {}
            for name, ids in prompts.items():
                c = SyncCounter()
                c.install()
                try:
                    _, blocks, _ = once(ids, combo, c)
                finally:
                    c.uninstall()
                total = sum(c.sites.values())
                per[name] = {"rounds": blocks, "syncs": total,
                             "per_round": round(total / max(blocks, 1), 2),
                             "sites": dict(c.sites.most_common())}
                print(f"[loop_sync] {combo:<6} {name:<6} {total} syncs / {blocks} rounds = "
                      f"{total / max(blocks, 1):.2f} a round", flush=True)
                for site, n in c.sites.most_common(12):
                    print(f"             {n:5d}  {site}", flush=True)
            report["syncs"][combo] = per
    app.LAUNCH_FIRST, M.HOST_ASYNC = False, False
    ok = all(all(v.values()) for v in report["identical"].values())
    report["verdict"] = "IDENTICAL" if ok else "DIFFERENT"
    print(f"[loop_sync] tokens across {combos}: {report['verdict']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
