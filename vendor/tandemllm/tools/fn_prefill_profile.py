"""Where a Flash-Next prefill spends its time, by part (ENG-225's profile).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_prefill_profile.py --pack ~/fn2/packs/3.05 \
        --ctx 2048,8192,32768 --chunk 2048 --json out.json

Per context (tools/fn_generate.py's prose prompt filled to `--ctx` tokens): one prefill with every part
timed (a device sync before and after each call, time EXCLUSIVE of the parts it calls), then one
prefill untimed for the honest wall time and tok/s. Parts: the PLE branch, the hyper-connection norms,
mixers (down GEMV, mix_up) and injects, EXL3 projections, other projections, the GDN conv and gates, the
delta rule, attention (the core, the QSA indexer), the router, the experts, the final mixer and head.
`--path kernel` profiles `FlashNextEngine.prefill_kernels` (ENG-225); `ref` the reference forward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402


class PartTimer:
    """Exclusive wall time of wrapped calls, each bracketed by device syncs."""

    def __init__(self):
        self.t: dict = defaultdict(float)
        self.n: dict = defaultdict(int)
        self.stack: list = []
        self.undo: list = []

    def wrap(self, obj, name: str, label):
        f = getattr(obj, name)

        def g(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.stack.append(0.0)
            try:
                return f(*a, **kw)
            finally:
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                child = self.stack.pop()
                lab = label(*a) if callable(label) else label
                self.t[lab] += dt - child
                self.n[lab] += 1
                if self.stack:
                    self.stack[-1] += dt
        self.undo.append((obj, name, obj.__dict__.get(name) if hasattr(obj, "__dict__") else None, f))
        setattr(obj, name, g)

    def restore(self) -> None:
        for obj, name, own, f in reversed(self.undo):
            if own is None and not isinstance(obj, type(sys)):
                delattr(obj, name)                       # back to the class's method
            else:
                setattr(obj, name, f)
        self.undo = []


def instrument(eng: FlashNextEngine) -> PartTimer:
    from tools import fn_kernels as K
    from tools import fn_seq_kernels as SK
    from tools.exl3_linear import EXL3Block
    T = PartTimer()

    def lin_label(x, w):
        return "exl3 projections" if isinstance(w, EXL3Block) else "other projections"
    T.wrap(eng, "lin", lin_label)
    for name, lab in (("ple", "ple"), ("gdn", "gdn other"), ("attention", "attention other"),
                      ("delta_rule", "gdn delta rule"), ("qsa_mask", "attention qsa"), ("final", "final mixer + head"),
                      ("moe", "moe (reference)"), ("moe_parts", "moe other"), ("build_pool", "qsa pool"),
                      ("_ple_pre", "ple"), ("_gdn_pre", "gdn other"), ("_attn_pre", "attention other"),
                      ("_moe_pre", "moe other"), ("_mix_pre", "mixers (prefill)")):
        if hasattr(eng, name):
            T.wrap(eng, name, lab)
    for name, lab in (("gemv", "mixer down / router gemv"), ("mix_up", "mixer up"), ("hc_norm", "norms + injects"),
                      ("inject_norm", "norms + injects"), ("combine_inject_norm", "norms + injects"),
                      ("router", "router"), ("moe_experts", "moe experts"), ("gemv_bf16", "other projections")):
        T.wrap(K, name, lab)
    for name, lab in (("attn_rows", "attention core"), ("qsa_scores", "attention qsa"), ("qsa_select", "attention qsa"),
                      ("qsa_pool", "qsa pool"), ("gdn_pre", "gdn conv + gates")):
        T.wrap(SK, name, lab)
    from tools import fn_prefill_kernels as PK
    for name, lab in (("moe_grouped", "moe experts"), ("mm_rows", "library GEMM rows (mixers, router, a/b, ple)"),
                      ("mix_tail", "mixer up")):
        T.wrap(PK, name, lab)
    return T


def run(eng: FlashNextEngine, ids: list[int], path: str, chunk: int) -> float:
    st = eng.new_state(eng.max_len)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    eng.prefill(torch.tensor(ids), st, chunk=chunk or None, path="kernels" if path == "kernel" else "ref")
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del st
    torch.cuda.empty_cache()
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", default="2048,8192")
    ap.add_argument("--chunk", type=int, default=0, help="rows a chunk (0: the path's default)")
    ap.add_argument("--path", default="ref", choices=("ref", "kernel"))
    ap.add_argument("--timed", default="1", help="0: wall time only (no per-part profile)")
    ap.add_argument("--wall-max", type=int, default=1 << 30, help="no untimed run past this context")
    ap.add_argument("--kernels", type=int, default=0, help="also the top N kernels of one untimed run (torch.profiler)")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    ctxs = [int(x) for x in a.ctx.split(",")]
    idx, W, banks, tables, timing = load(a.pack, "cuda")
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max(ctxs) + 64)
    tok = AutoTokenizer.from_pretrained(a.pack)
    res = {"pack": a.pack, "path": a.path, "chunk": a.chunk, "load": timing, "mem_after_load_gib": mem_available_gib(),
           "knobs": {"kv_dtype": "bf16", "prefix_cache": "off", "head": "last row of the last chunk"}, "rows": []}
    run(eng, class_prompt(tok, "prose", 256), a.path, a.chunk)          # build the kernels
    for n in ctxs:
        ids = class_prompt(tok, "prose", n)
        row = {"ctx": len(ids)}
        if a.timed == "1":
            T = instrument(eng)
            dt = run(eng, ids, a.path, a.chunk)
            T.restore()
            parts = sorted(T.t.items(), key=lambda kv: -kv[1])
            row["timed_s"] = round(dt, 2)
            row["parts_s"] = {k: round(v, 3) for k, v in parts}
            row["parts_share"] = {k: round(v / dt, 3) for k, v in parts}
            row["calls"] = dict(T.n)
        if len(ids) <= a.wall_max:
            dt = run(eng, ids, a.path, a.chunk)
            row["wall_s"] = round(dt, 2)
            row["tok_s"] = round(len(ids) / dt, 1)
        if a.kernels:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                run(eng, ids, a.path, a.chunk)
            per = {}
            for e in prof.key_averages():
                us = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
                if us:
                    per[e.key] = (per.get(e.key, (0.0, 0))[0] + us / 1e6, e.count)
            row["gpu_s"] = round(sum(v[0] for v in per.values()), 2)
            row["kernels"] = [(k[:80], round(v[0], 3), v[1]) for k, v in sorted(per.items(), key=lambda kv: -kv[1][0])[:a.kernels]]
        row["mem_gib"] = mem_available_gib()
        res["rows"].append(row)
        print(f"[prefill-profile] {json.dumps(row)}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
