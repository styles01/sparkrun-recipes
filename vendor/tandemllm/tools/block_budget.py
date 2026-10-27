"""The block, itemised: every millisecond of one speculative block, what it reads, and how fast.

`tools/profile_cycle.py` splits a block into its phases. This splits the phases into their parts --
the MLP's three projections, the attention and linear-attention projections, the recurrence, the
attention core, the norms, the head over the verify rows, the drafter's backbone and head, the
commit, the drafter's context sync -- and prices each one against the board:

    bytes      what the part reads and writes, from the tensors it was handed (weights dominate)
    GB/s       bytes over the GPU time of the kernels it launched
    @235       the time the same bytes take at 235 GB/s, the practical ceiling of this board
    recover    GPU time minus @235 for a part that moves bytes; the whole GPU time for glue whose
               bytes are negligible (a fusion would delete it, not speed it up)

and it puts the host beside the device: a phase's wall clock minus the kernel time inside it is
time the GPU had nothing to do, and that is recoverable too, by removing a sync or overlapping.

HOW A KERNEL IS ATTRIBUTED. Each part runs inside a `record_function` whose name is its full path
(`verify/mlp/gate`, `draft/head`, ...). In the chrome trace a kernel carries a `correlation` id,
and so does the host-side launch call that issued it; the launch's timestamp falls inside exactly
the annotations that were open when it was issued, and the innermost one is the part. Launch time
rather than execution time, because the device runs behind the host and a kernel's own interval can
sit under a later annotation. The phases are traced STRICT (a sync at every boundary), so a kernel
launched in a phase also runs in it.

The wrappers are installed by this tool only and removed afterwards; the engine carries no
instrumentation. They cost host time under the profiler, which is why the wall-clock split comes
from an UNTRACED run of the same blocks and the traced run supplies only the device side.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import profile_cycle as pc  # noqa: E402

CEIL_GBPS = 235.0

# New text only. `prose` and `chat` are profile_cycle's; `code` is a request whose answer is not in
# its prompt. The copy workloads are deliberately absent: this budget is about the block the row's
# median pays, and on copy text the width, not the block time, is the limit.
PROMPTS = {
    "prose": pc.PROMPTS["prose"],
    "chat": pc.PROMPTS["chat"],
    "code": "Write a Python function that parses an ISO-8601 duration such as P3DT4H5M6S into a "
            "number of seconds. Validate the input, raise ValueError with a useful message, and "
            "include three doctest examples.",
}

# Weight shape -> the projection's name, per mixer. (5120, 6144) is the GDN output projection in a
# linear-attention layer and `o_proj` in an attention layer; the enclosing part says which.
SHAPES = {
    (17408, 5120): "gate|up", (5120, 17408): "down",
    (10240, 5120): "qkv", (6144, 5120): "z", (5120, 6144): "out",
    (12288, 5120): "q", (1024, 5120): "k|v", (48, 5120): "a|b",
    (34816, 5120): "gate+up", (14336, 5120): "q+k+v", (16384, 5120): "qkv+z",
    (10240 + 6144, 5120): "qkv+z",
}


def _wbytes(w) -> int:
    """Bytes a projection reads for its weight, whatever it is stored as."""
    if isinstance(w, torch.Tensor):
        return w.numel() * w.element_size()
    if hasattr(w, "nbytes") and not callable(w.nbytes):
        return int(w.nbytes)
    n = 0
    for name in ("w", "s", "codes", "scale", "s2v"):
        t = getattr(w, name, None)
        if isinstance(t, torch.Tensor):
            n += t.numel() * t.element_size()
    return n


def _nb(*ts) -> int:
    return sum(t.numel() * t.element_size() for t in ts if isinstance(t, torch.Tensor))


class Parts:
    """The label stack, the per-part byte counts, and the wrappers that feed both."""

    def __init__(self):
        self.stack: list[str] = []
        self.bytes = defaultdict(float)      # "phase|label" -> bytes over the traced run
        self.calls = defaultdict(int)
        self.on = False                      # only count during the traced blocks
        self.phase = "?"                     # set from the Phases annotations as they open

    @contextlib.contextmanager
    def scope(self, name: str, nbytes: int = 0):
        self.stack.append(name)
        label = "/".join(self.stack)
        if self.on:
            self.bytes[f"{self.phase}|{label}"] += nbytes
            self.calls[f"{self.phase}|{label}"] += 1
        try:
            with torch.profiler.record_function("C::" + label):
                yield
        finally:
            self.stack.pop()

    def install(self):
        """Wrap the engine's parts. Returns an undo."""
        import engine.gdn as G
        import engine.model as M
        from engine.drafters import dflash2 as D
        from tools import attn_kernels as AK
        from tools import gdn_kernels as GK
        from tools import gdn_tree_kernels as GT

        undo = []

        def patch(obj, name, make):
            orig = getattr(obj, name)
            setattr(obj, name, make(orig))
            undo.append((obj, name, orig))

        parts = self

        def proj_name(w) -> str:
            shp = tuple(w.shape) if hasattr(w, "shape") else (getattr(w, "N", 0), getattr(w, "K", 0))
            return SHAPES.get(tuple(int(x) for x in shp), f"{shp[0]}x{shp[1]}")

        def linear(orig):
            def f(x, w):
                if isinstance(w, M.FP8Head):
                    return orig(x, w)            # head_logits labels it
                m = x.numel() // x.shape[-1]
                n = w.shape[0] if isinstance(w, torch.Tensor) else w.N
                with parts.scope(proj_name(w), _wbytes(w) + x.numel() * 2 + m * n * 2):
                    return orig(x, w)
            return f

        def group(orig):
            def f(x, g):
                with parts.scope(proj_name(g), _wbytes(g) + x.numel() * 2 + x.shape[0] * g.N * 2):
                    return orig(x, g)
            return f

        def norm(orig):
            def f(x, *a, **k):
                extra = a[0] if a and isinstance(a[0], torch.Tensor) and a[0].shape == x.shape else None
                with parts.scope("norm", 2 * _nb(x) + _nb(extra)):
                    return orig(x, *a, **k)
            return f

        def head(orig):
            def f(h, weight):
                m = h.numel() // h.shape[-1]
                n = weight.N if hasattr(weight, "N") else weight.shape[0]
                with parts.scope("head", _wbytes(weight) + m * n * 4):
                    return orig(h, weight)
            return f

        def method(name, nbytes=None):
            def make(orig):
                def f(*a, **k):
                    with parts.scope(name, nbytes(*a, **k) if nbytes else 0):
                        return orig(*a, **k)
                return f
            return make

        def rec_bytes(q, k, v, g, beta, state, *a, **kw):
            # the state tile is read once and written once; the per-token vectors are small
            return 2 * _nb(state) + _nb(q, k, v, g, beta) + _nb(v)

        def tree_bytes(q, k, v, g, beta, depths, state, *a, **kw):
            return _nb(state) + _nb(q, k, v, g, beta) + 2 * _nb(v)

        def attn_bytes(q, k, v, start, bm, *a, **kw):
            T = q.shape[2]
            L = start + T
            per = k.shape[1] * k.shape[3] * k.element_size()
            return 2 * per * L + 2 * _nb(q)

        def clone_bytes(self, tokens, start, *a, **kw):
            return 2 * _nb(self.state.S, self.state.conv)

        patch(M, "linear", linear)
        # every fused group's one launch, whatever its format (NVFP4Group, FP8Group, an EXL3 group)
        patch(M, "matmul_group", group)
        # an EXL3 projection's input rotation (tools/exl3_linear.py), its own part under the projection
        from tools import exl3_linear as XL
        patch(XL, "rotate_in", method("rot", lambda x, w: _nb(x) + x.shape[0] * w.K * 2 * w.suh.shape[0]))
        patch(M, "rms_norm", norm)
        patch(M, "rms_norm_gated", norm)
        patch(M, "head_logits", head)
        patch(G, "conv_tree", method("conv"))
        patch(G, "conv_update", method("conv"))
        patch(GK, "fused_block_step", method("recurrence", rec_bytes))
        patch(GT, "fused_tree_step", method("recurrence", tree_bytes))
        patch(AK, "decode_attention", method("core", attn_bytes))
        patch(M.Qwen38Engine, "mlp", method("mlp"))
        patch(M.Qwen38Engine, "attention", method("attn"))
        patch(M.Qwen38Engine, "linear_attention", method("gdn"))
        # `chain` encloses a chain verify, and what it launches itself is the entry snapshot of the
        # recurrent and convolution states; `forward` encloses the glue of the pass (embedding,
        # residual adds, the rotary gather)
        patch(M.Qwen38Engine, "forward_block", method("chain", clone_bytes))
        patch(M.Qwen38Engine, "forward", method("forward"))
        patch(M.Qwen38Engine, "rollback_to", method("rollback"))
        patch(M.Qwen38Engine, "commit_tree", method("commit"))
        patch(D, "_lin", linear)
        patch(D.DFlash2Module, "forward_block", method("backbone"))
        patch(D.DFlash2Module, "project_context", method("fc"))
        patch(D.DFlash2Module, "context_kv", method("ctx_kv"))
        patch(D.DFlash2Module, "unary_candidates", method("topk"))
        patch(D.DFlash2Module, "lattice", method("lattice"))

        def restore():
            for obj, name, orig in reversed(undo):
                setattr(obj, name, orig)
        return restore


PHASE_OF = {"lookup": "draft", "lattice": "draft", "draft": "draft", "merge": "draft",
            "h2d": "verify", "verify": "verify", "picks": "accept", "price": "accept",
            "accept": "accept", "commit": "commit", "sync": "sync", "observe": "observe",
            "tail": "observe"}


def attribute(trace_path: str) -> tuple[dict, dict, float]:
    """Kernel time per part and per phase from a chrome trace.

    Returns ({(phase, part): [gpu_us, launches]}, {phase: busy_us}, span_us).
    """
    ev = json.load(open(trace_path))["traceEvents"]
    launch_ts: dict[int, float] = {}
    kernels = []
    ann = []
    for e in ev:
        cat = e.get("cat", "")
        args = e.get("args") or {}
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append((e["ts"], e.get("dur", 0.0), args.get("correlation")))
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in args:
            launch_ts[args["correlation"]] = e["ts"]
        elif cat in ("user_annotation", "cpu_op") and str(e.get("name", "")).startswith(("C::",
                                                                                          "PH::")):
            ann.append((e["ts"], e["ts"] + e.get("dur", 0.0), e["name"]))
    # One sweep. The annotations are properly nested (one thread, scopes and phases never
    # straddle each other), so a stack of the open ones, popped as they close, has the innermost
    # open annotation on top at any host timestamp.
    ann.sort(key=lambda x: (x[0], -x[1]))
    launches = sorted((launch_ts[c], i) for i, (_, _, c) in enumerate(kernels) if c in launch_ts)
    where: dict[int, tuple[str, str]] = {}
    stack: list = []
    ai = 0
    for ts, i in launches:
        while ai < len(ann) and ann[ai][0] <= ts:
            while stack and stack[-1][1] < ann[ai][0]:
                stack.pop()
            stack.append(ann[ai])
            ai += 1
        while stack and stack[-1][1] < ts:
            stack.pop()
        phase = part = None
        for s0, t0, name in reversed(stack):
            if part is None and name.startswith("C::"):
                part = name[3:]
            if phase is None and name.startswith("PH::"):
                phase = PHASE_OF.get(name[4:], name[4:])
            if part is not None and phase is not None:
                break
        where[i] = (phase or "?", part or "")

    out: dict = defaultdict(lambda: [0.0, 0])
    busy: dict = defaultdict(float)
    unmatched = 0
    for i, (ts, dur, corr) in enumerate(kernels):
        if i not in where:
            unmatched += 1
            continue
        phase, part = where[i]
        key = (phase, part or "(glue)")
        out[key][0] += dur
        out[key][1] += 1
        busy[phase] += dur
    span = (max(k[0] + k[1] for k in kernels) - min(k[0] for k in kernels)) if kernels else 0.0
    if unmatched:
        print(f"[budget] {unmatched} of {len(kernels)} kernels had no launch record", flush=True)
    return dict(out), dict(busy), span


def gaps(trace_path: str, blocks: int) -> dict:
    """The host's share of a block, from a trace of the loop as it really runs (no extra syncs).

    Every kernel and copy interval is merged; what lies between two merged intervals is time the
    GPU had nothing queued. A gap of a few microseconds is the device's own step from one kernel to
    the next with the queue full; a longer one means the host had not issued the next launch yet --
    Python between kernels, or a device-to-host read the host was waiting on and then the Python
    that acts on it. Each gap is charged to the phase that launched the kernel ending it.
    """
    ev = json.load(open(trace_path))["traceEvents"]
    launch_ts: dict[int, float] = {}
    iv, ann = [], []
    n_kernel = n_dtoh = n_htod = n_sync = 0
    for e in ev:
        cat = e.get("cat", "")
        args = e.get("args") or {}
        name = str(e.get("name", ""))
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            iv.append((e["ts"], e["ts"] + e.get("dur", 0.0), args.get("correlation")))
            if cat == "kernel":
                n_kernel += 1
            elif "DtoH" in name:
                n_dtoh += 1
            elif "HtoD" in name:
                n_htod += 1
        elif cat in ("cuda_runtime", "cuda_driver"):
            if "correlation" in args:
                launch_ts[args["correlation"]] = e["ts"]
            if "Synchronize" in name:
                n_sync += 1
        elif cat == "user_annotation" and name.startswith("PH::"):
            ann.append((e["ts"], e["ts"] + e.get("dur", 0.0), PHASE_OF.get(name[4:], name[4:])))
    iv.sort()
    ann.sort()
    import bisect
    starts = [x[0] for x in ann]

    def phase_at(ts):
        i = bisect.bisect_right(starts, ts) - 1
        return ann[i][2] if i >= 0 and ann[i][1] >= ts else "?"

    first = ann[0][0] if ann else (iv[0][0] if iv else 0.0)
    last = ann[-1][1] if ann else 0.0
    buckets = {"<=5us": [0, 0.0], "5-50us": [0, 0.0], ">50us": [0, 0.0]}
    by_phase: dict = defaultdict(float)
    end = None
    for a0, b0, corr in iv:
        if a0 < first or a0 > last:
            continue
        if end is not None and a0 > end:
            g = a0 - end
            key = "<=5us" if g <= 5 else ("5-50us" if g <= 50 else ">50us")
            buckets[key][0] += 1
            buckets[key][1] += g
            by_phase[phase_at(launch_ts.get(corr, a0))] += g
        end = b0 if end is None else max(end, b0)
    nb = max(blocks, 1)
    return dict(kernels=n_kernel / nb, dtoh=n_dtoh / nb, htod=n_htod / nb, sync_calls=n_sync / nb,
                gap_count={k: v[0] / nb for k, v in buckets.items()},
                gap_ms={k: v[1] / 1e3 / nb for k, v in buckets.items()},
                gap_ms_by_phase={k: v / 1e3 / nb for k, v in by_phase.items()})


def run(a, eng, drafter, arms, ng, k, tok, name: str, fixed: int, parts: Parts) -> dict:
    prompt = tok(PROMPTS[name])
    drafter.fixed = fixed
    label = f"{name} {'latch' if fixed == 0 else f'fixed{fixed}'}"

    # 1. the wall clock, loose, untraced: what the loop really spends a block
    ph = pc.Phases(strict=False)
    undo = pc.instrument(arms, ng, ph)
    out_loose, st = pc.cycle(eng, drafter, prompt, a.max_new, k, ph)
    undo()
    n = len(ph.block_ms)
    wall = {p: sum(v) / n for p, v in ph.ms.items() if v}
    block_ms = sum(ph.block_ms) / n
    acc = st["accepted"] / max(st["blocks"], 1)

    # 2. the device side, strict + traced, over the first `trace_new` tokens of the same request
    ph2 = pc.Phases(strict=True, trace=True)
    undo = pc.instrument(arms, ng, ph2)
    restore = parts.install()
    enter = ph2._enter

    def entered(pname: str) -> None:
        parts.phase = PHASE_OF.get(pname, pname)
        enter(pname)

    ph2._enter = entered
    parts.phase = "?"
    parts.bytes.clear()
    parts.calls.clear()
    parts.on = True
    from torch.profiler import ProfilerActivity, profile
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            out_tr, st2 = pc.cycle(eng, drafter, prompt, a.trace_new, k, ph2)
    finally:
        parts.on = False
        restore()
        undo()
    js = os.path.join(a.out, f"trace-{name}-{fixed}.json")
    prof.export_chrome_trace(js)
    per, busy, span = attribute(js)
    if not a.keep_traces:
        os.remove(js)
    nb = len(ph2.block_ms)

    # 3. the host's share, traced LOOSE: the loop as it runs, annotations only, no added syncs
    ph3 = pc.Phases(strict=False, trace=True)
    undo = pc.instrument(arms, ng, ph3)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof3:
        pc.cycle(eng, drafter, prompt, a.trace_new, k, ph3)
    undo()
    js3 = os.path.join(a.out, f"trace-loose-{name}-{fixed}.json")
    prof3.export_chrome_trace(js3)
    host = gaps(js3, len(ph3.block_ms))
    host["traced_loose_block_ms"] = sum(ph3.block_ms) / max(len(ph3.block_ms), 1)
    if not a.keep_traces:
        os.remove(js3)
    if out_tr != out_loose[:len(out_tr)]:
        print(f"[budget] WARNING {label}: traced run diverged from the loose run", flush=True)
    return dict(label=label, workload=name, fixed=fixed, out=list(out_loose), blocks=n, tokens=st["tokens"],
                tok_s=st["tok_s"], accepted=acc + 1.0, nodes=st["nodes"] / max(st["blocks"], 1),
                block_ms=block_ms, wall=wall, traced_blocks=nb,
                traced_block_ms=sum(ph2.block_ms) / nb,
                strict_wall={p: sum(v) / nb for p, v in ph2.ms.items() if v},
                parts={f"{p}|{q}": [v[0] / 1e3 / nb, v[1] / nb] for (p, q), v in per.items()},
                bytes={q: b / nb for q, b in parts.bytes.items()},
                calls={q: c / nb for q, c in parts.calls.items()},
                busy={p: b / 1e3 / nb for p, b in busy.items()}, host=host)


def table(r: dict) -> str:
    """One run's block, itemised."""
    lines = [f"=== {r['label']}: {r['blocks']} blocks, {r['tokens']} tok, {r['tok_s']:.2f} tok/s, "
             f"{r['accepted']:.2f} committed/block (incl. the target's own), "
             f"{r['nodes']:.1f} drafted nodes/block, block {r['block_ms']:.2f} ms loose "
             f"(traced strict {r['traced_block_ms']:.2f}) ===",
             f"{'part':<34} {'calls':>6} {'GPU ms':>8} {'GB':>7} {'GB/s':>7} {'@235':>7} "
             f"{'recover':>8}"]
    # A chain verify runs inside `chain`, a tree verify does not; the parts are the same parts, so
    # the prefix is folded away, and `chain` itself -- what it launched outside the forward -- is
    # the entry snapshot of the recurrent and convolution states.
    merged: dict = defaultdict(lambda: [0.0, 0.0, 0.0])
    for key, (ms, launches) in r["parts"].items():
        if key.split("|", 1)[0] == "?":
            continue                         # the prefill, before the first block
        b = r["bytes"].get(key, 0.0)
        phase, part = key.split("|", 1)
        part = "snapshot" if part == "chain" else part.removeprefix("chain/")
        row = merged[f"{phase}|{part}"]
        row[0] += ms
        row[1] += launches
        row[2] += b
    rows = []
    for key, (ms, launches, b) in merged.items():
        phase, part = key.split("|", 1)
        gb = b / 1e9
        at = gb / CEIL_GBPS * 1e3
        bw = gb / (ms / 1e3) if ms > 0 and b > 0 else 0.0
        big = b >= 1e6 * max(ms, 1e-9)          # >= 1 GB/s worth of bytes: a bandwidth part
        rec = ms - at if big else ms
        rows.append((phase, part, launches, ms, gb, bw, at, rec, big))
    rows.sort(key=lambda x: (x[0], -x[3]))
    tot_gpu = tot_rec = 0.0
    for phase, part, launches, ms, gb, bw, at, rec, big in rows:
        tot_gpu += ms
        tot_rec += max(rec, 0.0)
        lines.append(f"{phase + ':' + part:<34} {launches:6.0f} {ms:8.3f} "
                     f"{gb if big else float('nan'):7.3f} {bw if big else float('nan'):7.1f} "
                     f"{at if big else float('nan'):7.3f} {rec:8.3f}")
    lines.append(f"{'device total':<34} {'':>6} {tot_gpu:8.3f} {'':>7} {'':>7} {'':>7} "
                 f"{tot_rec:8.3f}")
    lines.append("host side (loose wall minus the kernels of the same phase, per block):")
    tot_w = tot_b = 0.0
    for phase in ("draft", "verify", "accept", "commit", "sync", "observe"):
        w = sum(v for p, v in r["wall"].items() if PHASE_OF.get(p, p) == phase)
        b = r["busy"].get(phase, 0.0)
        tot_w += w
        tot_b += b
        lines.append(f"   {phase:<8} wall {w:8.3f}  kernels {b:8.3f}  host/idle {w - b:8.3f}")
    lines.append(f"   {'block':<8} wall {tot_w:8.3f}  kernels {tot_b:8.3f}  host/idle {tot_w - tot_b:8.3f}")
    h = r.get("host")
    if h:
        gm, gc = h["gap_ms"], h["gap_count"]
        lines.append(f"loop as it runs (traced loose, {h['traced_loose_block_ms']:.1f} ms/block under the "
                     f"profiler): {h['kernels']:.0f} kernels, {h['dtoh']:.1f} DtoH + {h['htod']:.1f} HtoD "
                     f"copies, {h['sync_calls']:.1f} synchronise calls a block")
        lines.append("   GPU idle gaps a block: " + ", ".join(
            f"{k} {gc[k]:.0f} = {gm[k]:.2f} ms" for k in ("<=5us", "5-50us", ">50us")))
        lines.append("   idle by the phase that ended it: " + ", ".join(
            f"{p} {v:.2f}" for p, v in sorted(h["gap_ms_by_phase"].items(), key=lambda x: -x[1])))
    return "\n".join(lines)


def ab_states(attrs: list[str], also: str = "") -> list[tuple]:
    """--ab's configurations: all off, each on alone, all on (with more than one), then every
    `--also` combination ('+'-joined attribute names), each once, in that order."""
    n = len(attrs)
    states = [tuple([False] * n)] + [tuple(i == j for j in range(n)) for i in range(n)]
    if n > 1:
        states.append(tuple([True] * n))
    for combo in filter(None, also.split(",")):
        on = set(combo.split("+"))
        unknown = on - set(attrs)
        if unknown:
            raise SystemExit(f"--also names {sorted(unknown)}, which --ab does not")
        st = tuple(attr in on for attr in attrs)
        if st not in states:
            states.append(st)
    return states


def ab_assign(mod, attr: str):
    """An --ab flag written `attr=value[;attr=value...]`: (the values it sets when on, the module's
    own it restores when off), each value cast to the type the module holds. None for a bare
    attribute, which is set to True / False."""
    if "=" not in attr:
        return None
    on = {}
    for kv in attr.split(";"):
        name, value = kv.split("=", 1)
        on[name] = type(getattr(mod, name))(value)
    return on, {name: getattr(mod, name) for name in on}


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """The index of the first token two runs disagree on (a length difference counts), or None."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--ckpt8", required=True)
    ap.add_argument("--ckpt16", required=True)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--trace-new", type=int, default=96)
    ap.add_argument("--warm", type=int, default=48)
    ap.add_argument("--workloads", default="prose,chat,code")
    ap.add_argument("--widths", default="0,8,16", help="0 = the served latch, else fixed width")
    ap.add_argument("--out", default="results/kernels")
    ap.add_argument("--keep-traces", action="store_true")
    ap.add_argument("--latch", action="store_true", default=True)
    ap.add_argument("--drop-idle", action="store_true", default=True)
    ap.add_argument("--fixed", type=int, default=0)
    ap.add_argument("--ab", default="",
                    help="module:attribute[,module:attribute...] -- every configuration with all "
                         "of them off, each on alone, and all on, in one process. A flag may set "
                         "values instead of True: module:attr=value[;attr=value...] (e.g. "
                         "the WY thresholds and slices as one flag); off restores what the module "
                         "had")
    ap.add_argument("--precapture", type=int, default=0,
                    help="capture the verify graphs of every row count 2..N (chain and tree) in "
                         "every --ab state before measuring (a 24-node tree's graph is "
                         "otherwise captured inside the measured run, in each state; 0 = as before)")
    ap.add_argument("--greedy-check", action="store_true",
                    help="with --ab: decode each workload once one token at a time (the greedy run "
                         "every speculative run must reproduce, state-independent) and judge every "
                         "state's tokens against it as tools/verify_spec.py does: identical, a "
                         "one-ulp logit tie, or a divergence (a state whose tokens differ "
                         "from the base's)")
    ap.add_argument("--also", default="",
                    help="with --ab: more states, comma-separated, each a '+'-joined set of the "
                         "--ab attributes that are on (e.g. VERIFY_GRAPH+GDN_AB)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer
    cfg, eng, drafter, arms, ng, k = pc.build(a)
    tk = AutoTokenizer.from_pretrained(cfg.path)

    def ids(text: str) -> torch.Tensor:
        msg = [{"role": "user", "content": text}]
        s = tk.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
        return tk(s, return_tensors="pt").input_ids[0].cuda()

    # --ab: every flag off, each flag on alone, and (with more than one) all of them on
    flags = []
    if a.ab:
        import importlib
        for spec in a.ab.split(","):
            mod_name, attr = spec.split(":", 1)
            flags.append((importlib.import_module(mod_name), attr))
    states: list = ab_states([attr for _, attr in flags], a.also) if flags else [None]
    # a flag written attr=value[;attr=value...] sets those values when on and restores the module's
    # own when off; a bare attribute is True / False as before
    assigns = [ab_assign(mod, attr) for mod, attr in flags]

    def apply(st) -> str:
        if st is None:
            return ""
        # the flags that are off restore first, then the ones that are on set theirs, so two flags
        # may share an attribute (both set the same slice width, each its own threshold)
        for want in (False, True):
            for (mod, attr), sets, on in zip(flags, assigns, st):
                if on != want:
                    continue
                if sets is None:
                    setattr(mod, attr, on)
                else:
                    for k_, v_ in sets[0 if on else 1].items():
                        setattr(mod, k_, v_)
        return " " + ("+".join(attr for (mod, attr), on in zip(flags, st) if on) or "base")

    # warm both widths in every state: the first call of a kernel configuration in a process pays
    # its compilation, and the state measured first would otherwise read slowest
    print("[warm] both widths", flush=True)
    for st in states:
        apply(st)
        for fixed in (8, 16):
            if fixed == 8 and eng._graphs_for(2, 0) is not None:
                pc.cycle(eng, drafter, ids(PROMPTS["chat"]), 8, k, pc.Phases(strict=False))
                with torch.no_grad():
                    n_g = eng._graphs.precapture()
                print(f"[warm] verify graphs captured: {n_g}", flush=True)
            drafter.fixed = fixed
            pc.cycle(eng, drafter, ids(PROMPTS["chat"]), a.warm, k, pc.Phases(strict=False))
        if a.precapture and eng._graphs_for(2, 0) is not None:
            with torch.no_grad():
                n_g = eng._graphs.precapture(widths=range(2, a.precapture + 1))
            print(f"[warm] verify graphs 2..{a.precapture} captured: {n_g}", flush=True)
    print("[warm] done", flush=True)

    parts = Parts()
    results = []
    greedy: dict = {}
    for name in a.workloads.split(","):
        if a.greedy_check and name not in greedy:
            from engine.spec import generate_greedy
            with torch.no_grad():
                g_out, g_st = generate_greedy(eng, ids(PROMPTS[name]), a.max_new, record_gaps=True)
            greedy[name] = (g_out, g_st.gaps, g_st.tops)
        for fixed in [int(x) for x in a.widths.split(",")]:
            base_out = None
            for st in states:
                tag = apply(st)
                t = time.perf_counter()
                r = run(a, eng, drafter, arms, ng, k, ids, name, fixed, parts)
                out = r.pop("out")
                if name in greedy:
                    from tools.verify_spec import compare
                    g_out, gaps, tops = greedy[name]
                    ok, why = compare(g_out, out, gaps, tk, tops)
                    r["greedy"] = why if ok else "FAIL " + why
                    print(f"[budget] {r['label']}{tag} against greedy: {r['greedy']}",
                          flush=True)
                if st is not None:
                    r["label"] += tag
                    r["ab"] = list(st)
                    # the first state is every flag off: each other state's tokens against it
                    if base_out is None:
                        base_out = out
                    else:
                        d = first_divergence(base_out, out)
                        r["tokens_vs_base"] = "identical" if d is None else d
                        print(f"[budget] {r['label']} tokens against base: "
                              + ("identical" if d is None else f"DIFFER from token {d}"), flush=True)
                results.append(r)
                print(table(r), flush=True)
                print(f"[budget] {r['label']} took {time.perf_counter() - t:.0f} s\n", flush=True)
    path = os.path.join(a.out, "block-budget.json")
    json.dump(results, open(path, "w"), indent=1)
    print(f"[budget] wrote {path}")


if __name__ == "__main__":
    main()
