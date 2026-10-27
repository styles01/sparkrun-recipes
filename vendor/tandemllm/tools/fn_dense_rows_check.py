"""The multi-row EXL3 dense path (SPD-125, tools/fn_kernels.dense_rows) against the one-row calls: bytes and time.

  exact   random blocks at Bolt's, the 27B's and Flash-Next's shapes, M = 1..32: every output byte of
          dense_rows(x[:M]) equals the one-row dense calls on those rows; plus groups, an fp32 head,
          forced splits (S > 1, 8 warps)
  same    the one-row path against another tree's tools/fn_kernels.py (--ref, e.g. `git show
          <rev>:tools/fn_kernels.py > ref.py`): dense at M = 1..32 byte for byte, so the 27B, Flash-Next
          and LFM2.5 rows are the tree before's
  real    every EXL3 projection of a NemotronH pack (--model), 16 rows of activations at the scale the
          layer sees (--seeds), the same comparison
  engine  a NemotronH pack's engine on the rows path: the load-time self-check (24 and 8 rows), a tree of 16
          rows against each branch's sequential decode, a captured verify graph against the eager rows, and
          32 decode steps with the rows path on and off
  bank    the relu^2 expert bank of a verify block (pairs of one expert grouped) against each row's lone call,
          and its time at 1 / 8 / 16 rows with few and many shared experts
  bench   CUDA-graph time a call: dense at M rows (one CTA a row) vs dense_rows, at the shapes, M = 1..16

    python tools/fn_dense_rows_check.py exact --seeds 8
    python tools/fn_dense_rows_check.py same --ref /tmp/fn_kernels_ref.py
    python tools/fn_dense_rows_check.py real --model ~/models/nemotron-3.5-bolt-74b-a9b-exl3-2.0
    python tools/fn_dense_rows_check.py bench --json out.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as FK  # noqa: E402

# (name, N, Kin, bits): Bolt 2.0's dense projections (4 bits, the head 6) and the 27B's / Flash-Next's
BOLT = [("mamba.in_proj", 18560, 4096, 4), ("mamba.out_proj", 4096, 8192, 4), ("attn.qkv", 4608, 4096, 4),
        ("attn.o_proj", 4096, 4096, 4), ("latent.fc1", 1024, 4096, 4), ("latent.fc2", 4096, 1024, 4),
        ("shared.up", 5376, 4096, 4), ("shared.down", 4096, 5376, 4), ("head", 131072, 4096, 6)]
Q27 = [("27b.qkv", 34816, 5120, 5), ("27b.down", 5120, 17408, 5), ("27b.gu", 16384, 5120, 5),
       ("27b.attn", 14336, 5120, 5), ("27b.o", 5120, 6144, 5)]
FN = [("fn.16384x2560", 16384, 2560, 5), ("fn.2560x6144", 2560, 6144, 5), ("fn.640x2560", 640, 2560, 3)]
ROWS = (1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 24, 32)


def _x(M: int, K: int, seed: int, scale: float) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(M, K, generator=g) * scale).to(torch.bfloat16).cuda()


def _one_rows(x: torch.Tensor, w, mod=FK) -> torch.Tensor:
    return torch.cat([mod.dense(x[m:m + 1], w) for m in range(x.shape[0])])


def _diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Rows whose bytes differ (0 = equal byte for byte)."""
    return int((a.view(torch.int16 if a.dtype != torch.float32 else torch.int32)
                != b.view(torch.int16 if b.dtype != torch.float32 else torch.int32)).any(-1).sum())


def _block(N, K, kb, seed, fp32=False):
    w = X.random_block(N, K, kb, seed=seed)
    w.out_fp32 = fp32
    return w


def cmd_exact(a) -> dict:
    out, bad = [], 0
    shapes = BOLT + Q27 + FN
    for name, N, K, kb in shapes:
        if name == "head":
            N = 16384                                  # the head's arithmetic at a smaller N (every N is the same code)
        for s in range(a.seeds):
            w = _block(N, K, kb, 1000 * s + kb, fp32=name == "head")
            x = _x(32, K, 7 + s, 0.5 if s % 2 else 4.0)
            one = _one_rows(x, w)
            for M in ROWS:
                d = _diff(FK.dense_rows(x[:M], w), one[:M])
                bad += d
                if d:
                    out.append(f"DIFF {name} seed {s} M {M}: {d} rows")
        out.append(f"{name} {N}x{K}x{kb} S{FK.dense_split(N, K, kb)[0]}: {a.seeds} seeds x M {ROWS[0]}..{ROWS[-1]} equal")
    # forced splits: clusters and 8 warps at a few shapes
    for (N, K, kb), split in (((4096, 8192, 4), (8, 4)), ((4096, 8192, 4), (2, 8)), ((1024, 4096, 4), (4, 8)),
                              ((18560, 4096, 4), (1, 8)), ((4096, 1024, 4), (1, 4)), ((2560, 6144, 5), (2, 8))):
        FK.DENSE_SPLIT[(N, K, kb)] = split
        try:
            for s in range(a.seeds):
                w = _block(N, K, kb, 77 + s)
                x = _x(32, K, 70 + s, 1.0)
                one = _one_rows(x, w)
                for M in ROWS:
                    d = _diff(FK.dense_rows(x[:M], w), one[:M])
                    bad += d
                    if d:
                        out.append(f"DIFF split {split} {N}x{K}x{kb} seed {s} M {M}: {d}")
        finally:
            FK.DENSE_SPLIT.pop((N, K, kb), None)
        out.append(f"split S{split[0]} w{split[1]} {N}x{K}x{kb}: equal")
    # every rate the kernels decode
    for kb in (2, 3, 5, 8, 15, 25, 35):
        w = _block(4096, 4096, kb, 300 + kb)
        x = _x(16, 4096, 31, 1.0)
        one = _one_rows(x, w)
        for M in (2, 8, 16):
            d = _diff(FK.dense_rows(x[:M], w), one[:M])
            bad += d
            if d:
                out.append(f"DIFF rate {kb} M {M}: {d}")
    out.append("rates 2, 3, 5, 8, 1.5, 2.5, 3.5: equal")
    # a group (members side by side, each its own suh) and a strided input row view
    blocks = [X.random_block(n, 4096, 4, seed=60 + i) for i, n in enumerate((8192, 4096, 1024))]
    grp = X.EXL3Block.group(blocks, ["a", "b", "c"])
    x = _x(16, 4096, 52, 1.0)
    one = _one_rows(x, grp)
    for M in (2, 9, 16):
        d = _diff(FK.dense_rows(x[:M], grp), one[:M])
        bad += d
    xs = _x(16, 4096 + 128, 53, 1.0)[:, 64:64 + 4096]
    w = _block(4096, 4096, 4, 9)
    d = _diff(FK.dense_rows(xs, w), torch.cat([FK.dense(xs[m:m + 1], w) for m in range(16)]))
    bad += d
    out.append(f"group of 3 members and a strided x: {'equal' if not bad else 'DIFF'}")
    return {"ok": bad == 0, "lines": out}


def _load_ref(path: str):
    spec = importlib.util.spec_from_file_location("fn_kernels_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_same(a) -> dict:
    ref = _load_ref(a.ref)
    out, bad = [], 0
    for name, N, K, kb in BOLT + Q27 + FN:
        if name == "head":
            N = 16384
        for s in range(a.seeds):
            w = _block(N, K, kb, 2000 + s, fp32=name == "head")
            x = _x(32, K, 9 + s, 1.0)
            for M in (1, 2, 8, 16, 32):
                d = _diff(FK.dense(x[:M], w), ref.dense(x[:M], w))
                bad += d
                if d:
                    out.append(f"DIFF {name} seed {s} M {M}: {d}")
        out.append(f"{name}: dense M 1..32 = the reference tree's")
    return {"ok": bad == 0, "lines": out}


def cmd_real(a) -> dict:
    from engine.models.nemotron_h.load import load_any
    idx, W, experts, _ = load_any(a.model, "cuda", mtp=False, log=None)
    out, bad, n = [], 0, 0
    seen = {}
    for key, w in W.items():
        if not isinstance(w, X.EXL3Block):
            continue
        n += 1
        for s in range(a.seeds):
            # activations at a normed row's scale and at a residual's (a few large channels)
            x = _x(16, w.K, 11 + s + n, 1.0 if s % 2 == 0 else 3.0)
            one = _one_rows(x, w)
            for M in (2, 4, 8, 16):
                d = _diff(FK.dense_rows(x[:M], w), one[:M])
                bad += d
                if d:
                    out.append(f"DIFF {key} seed {s} M {M}: {d}")
        shape = f"{w.N}x{w.K}x{w.kb}"
        seen[shape] = seen.get(shape, 0) + 1
    out.append(f"{n} EXL3 projections ({', '.join(f'{k} x{v}' for k, v in sorted(seen.items()))}), "
               f"{a.seeds} seeds x M 2/4/8/16: {'equal' if not bad else 'DIFF'}")
    return {"ok": bad == 0, "lines": out}


def _bank(E: int, I: int, X_: int, kbs, seed: int):
    from tools.exl3_relu2 import Relu2Bank
    return Relu2Bank([(X.random_block(I, X_, kbs[e % len(kbs)], seed=seed + e),
                       X.random_block(X_, I, kbs[(e + 1) % len(kbs)], seed=seed + 1000 + e)) for e in range(E)])


def _ids(M: int, k: int, E: int, pool: int, seed: int) -> torch.Tensor:
    """M rows of k distinct experts each, drawn from the first `pool` (a small pool: rows share experts)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack([torch.randperm(pool, generator=g)[:k] * (E // pool) for _ in range(M)]).to(torch.int32).cuda()


def cmd_bank(a) -> dict:
    """The relu^2 expert bank of a verify block (tools/exl3_relu2.py, pairs of one expert grouped) against
    each row's lone call, byte for byte; pools from 9 experts (every row the same 9) to 496 (few shared)."""
    from tools import exl3_relu2 as RB
    out, bad = [], 0
    for I, kbs in ((1536, (2,)), (2560, (2,)), (2048, (2, 3, 25, 4))):
        bank = _bank(48, I, 1024, kbs, 10 * I)
        for s in range(a.seeds):
            x = _x(32, 1024, 400 + s, 2.0 if s % 2 else 0.5)
            for pool in (9, 12, 24, 48):
                ids = _ids(32, 9, 48, pool, 50 * s + pool)
                one = torch.cat([bank.moe_relu2(x[m:m + 1], ids[m:m + 1]) for m in range(32)])
                for M in (2, 3, 8, 9, 16, 17, 32):
                    d = int((bank.moe_relu2(x[:M], ids[:M]).view(torch.int32) != one[:M * 9].view(torch.int32)).any(-1).sum())
                    bad += d
                    if d:
                        out.append(f"DIFF I {I} kbs {kbs} seed {s} pool {pool} M {M}: {d} pairs")
        out.append(f"bank I {I} bits {kbs}: {a.seeds} seeds x pools 9..48 x M 2..32 = the lone rows ({'equal' if not bad else 'DIFF'})")
    # time: Bolt's widest layer, 496 experts; ids with the sharing of a real block or none
    bank = _bank(496, 2560, 1024, (2,), 7)
    x = _x(16, 1024, 3, 1.0)
    for M in (1, 8, 16):
        for pool, what in ((496, "few shared"), (64, "many shared")):
            ids = _ids(M, 9, 496, pool, 9)
            ung = {}
            for flag in (False, True):
                RB.GROUP = flag
                ung[flag] = round(_time(lambda: bank.moe_relu2(x[:M], ids), 20), 1)
            RB.GROUP = True
            uniq = int(torch.unique(ids).numel())
            out.append(f"bank 496 x 2560 @2 bits, M {M}, {uniq} distinct of {M * 9} ({what}): a CTA a pair {ung[False]} us, "
                       f"grouped {ung[True]} us")
    return {"ok": bad == 0, "lines": out}


PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists.\"\"\"\n",
           "Explain why the sky is blue to a ten year old, in three sentences."]


def cmd_engine(a) -> dict:
    """The whole model on the rows path: the load-time self-check (a chain of 24 rows = two groups of 16,
    every row and the committed states against the decode steps), a branching tree of 16 rows against each
    branch's sequential decode, and a captured verify graph's replay against the eager rows."""
    from transformers import AutoTokenizer

    from engine.models.nemotron_h import engine as E
    from tools.nh_check import _engine
    calls = {"rows": 0}
    real = FK.dense_rows

    def counted(*args, **kw):
        calls["rows"] += 1
        return real(*args, **kw)
    FK.dense_rows = counted
    tok = AutoTokenizer.from_pretrained(a.model)
    idx, eng, info = _engine(a.model, 8192)
    out, ok = [f"load {info}", f"DENSE_ROWS {E.DENSE_ROWS}"], True
    for p in PROMPTS:
        ids = tok(p)["input_ids"]
        for steps in (24, 8):
            res = eng.self_check(ids, steps=steps, log=None)
            ok &= bool(res["ok"])
            out.append(f"self-check {steps} rows, prompt of {len(ids)}: {res}")
        # a tree: a greedy chain of 8, and branches with the chain's runner-up tokens
        st = eng.new_state(1024)
        lg = eng.prefill(torch.tensor(ids), st)
        L = st.length
        snap = [t.clone() for t in st.conv.values()] + [t.clone() for t in st.ssm.values()]
        chain_t, want = [int(lg[0].argmax())], []
        alt = []
        for _ in range(8):
            eng.stage(chain_t[-1], st)
            r = eng.step(st)[0].clone()
            st.length += 1
            want.append(r)
            chain_t.append(int(r.argmax()))
            alt.append(int(r.topk(2).indices[1]))
        # DFS pre-order: off row 6, its child; off row 4, its child; off row 2, two deep; off row 0
        toks = chain_t[:8] + [alt[6], alt[1], alt[4], alt[3], alt[2], alt[5], alt[0], alt[0]]
        parents = list(range(-1, 7)) + [6, 8, 4, 10, 2, 12, 13, 0]

        def restore():
            st.length = L
            for t, s in zip(list(st.conv.values()) + list(st.ssm.values()), snap):
                t.copy_(s)
        restore()
        eng.stage_rows(toks, parents, st)
        got = eng.rows_step(st, len(toks)).clone()
        bad = [i for i in range(8) if not torch.equal(got[i], want[i])]
        # every branch row against the sequential decode of its path from the prefill's state
        for i in range(8, len(toks)):
            path, j = [], i
            while j >= 0:
                path.append(toks[j])
                j = parents[j]
            restore()
            for t in reversed(path):
                eng.stage(t, st)
                r = eng.step(st)[0].clone()
                st.length += 1
            if not torch.equal(got[i], r):
                bad.append(i)
        restore()
        ns = E.K.class_of(st.length + len(toks), st.cap)
        vg = E.VerifyGraph(eng, st, len(toks), ns)
        vg.capture()
        eng.stage_rows(toks, parents, st)
        vg.replay()
        torch.cuda.synchronize()
        gbad = not torch.equal(eng._rows_buffers(len(toks))["logits"], got)
        ok &= not bad and not gbad
        out.append(f"tree of 16 (chain 8 + 8 branch rows) vs sequential decode: "
                   f"{'equal' if not bad else 'DIFF rows ' + str(bad)}; graph replay {'= eager' if not gbad else 'DIFF'}")
    # the decode step with DENSE_ROWS on and off: 32 greedy steps, every logit byte the same
    ids = tok(PROMPTS[0])["input_ids"]
    runs = {}
    for flag in (True, False):
        E.DENSE_ROWS = flag
        st = eng.new_state(1024)
        lg = eng.prefill(torch.tensor(ids), st)
        t, rows = int(lg[0].argmax()), []
        for _ in range(32):
            eng.stage(t, st)
            r = eng.step(st)[0].clone()
            st.length += 1
            rows.append(r)
            t = int(r.argmax())
        runs[flag] = torch.stack(rows)
    E.DENSE_ROWS = True
    same = torch.equal(runs[True], runs[False])
    ok &= same
    out.append(f"32 decode steps, DENSE_ROWS on vs off: {'every logit byte equal' if same else 'DIFF'}")
    out.append(f"dense_rows calls: {calls['rows']}")
    ok &= calls["rows"] > 0
    return {"ok": ok, "lines": out}


def _time(fn, reps: int) -> float:
    """us a call: `reps` calls in one CUDA graph, the median of 7 replays."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(7):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / reps)
    return sorted(ts)[3]


def cmd_bench(a) -> dict:
    rows = []
    out = []
    shapes = BOLT + (Q27 if a.q27 else [])
    for name, N, K, kb in shapes:
        w = _block(N, K, kb, 5, fp32=name == "head")
        x = _x(16, K, 3, 1.0)
        y = torch.empty(16, N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device="cuda")
        r = {"name": name, "shape": f"{N}x{K}x{kb}", "S": FK.dense_split(N, K, kb)[0], "one": {}, "rows": {}}
        for M in (1, 2, 4, 8, 16):
            xm, ym = x[:M], y[:M]
            r["one"][M] = round(_time(lambda: FK.dense(xm, w, out=ym), a.reps), 1)
            r["rows"][M] = round(_time(lambda: FK.dense_rows(xm, w, out=ym), a.reps), 1)
        rows.append(r)
        out.append(f"{name:14s} {r['shape']:16s} S{r['S']}  dense " + " ".join(f"{m}:{v}" for m, v in r["one"].items())
                   + "  rows " + " ".join(f"{m}:{v}" for m, v in r["rows"].items()) + " us")
    return {"ok": True, "lines": out, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("exact", "same", "real", "engine", "bank", "bench"):
        s = sub.add_parser(c)
        s.add_argument("--seeds", type=int, default=4)
        s.add_argument("--json")
        if c == "same":
            s.add_argument("--ref", required=True)
        if c in ("real", "engine"):
            s.add_argument("--model", required=True)
        if c == "bench":
            s.add_argument("--reps", type=int, default=20)
            s.add_argument("--q27", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    r = {"exact": cmd_exact, "same": cmd_same, "real": cmd_real, "engine": cmd_engine, "bank": cmd_bank,
         "bench": cmd_bench}[a.cmd](a)
    r["seconds"] = round(time.time() - t0, 1)
    r["device"] = torch.cuda.get_device_name()
    for line in r["lines"]:
        print(line, flush=True)
    print(json.dumps({"ok": r["ok"], "seconds": r["seconds"]}))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
