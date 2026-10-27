"""The block's kernels with counters, inside the served loop.

`tools/block_budget.py` prices each part by its time and its bytes at 235 GB/s; what the time over
the bytes is made of -- stall reasons, achieved occupancy, sector waste, fill and drain tails -- it
cannot say. Nsight Compute can, per kernel, and since the 2026-09-25 07:52 reboot the counters are
open to a user (`RmProfilingAdminOnly` 0). This tool is the driver and the reader:

    ncu --profile-from-start off --graph-profiling node --clock-control none --cache-control none \\
        --replay-mode application --csv --page raw --log-file OUT.csv --metrics ... \\
        python tools/ncu_budget.py run --workloads prose,chat,code --skip 12 --blocks 2 ...
    python tools/ncu_budget.py report OUT.csv --meta OUT.json

`run` builds the served engine (tools/profile_cycle.py build), freezes the router's learned costs
(so every replay of the application takes the same trees), warms both widths, captures the verify
graphs 2..32, and then, per workload, brackets blocks skip+1 .. skip+blocks of one generation with
cudaProfilerStart/Stop. A marker kernel (`torch.cuda._sleep`) opens every profiled block, so the
reader can split the kernel list into workloads and blocks without trusting anything else.

Why these ncu options: `--replay-mode application` because a kernel replay saves every allocation
the kernel could touch before its first pass, and on this board device memory IS host memory
(~52 GB allocated by the engine: a save of that size is how a box wedges); `--cache-control none`
because the question is the warm, in-loop L2, not a flushed one; `--clock-control none` because
the served loop runs at whatever clock the board gives it (2.43 GHz measured in a row, hold 12).
Counters only: ncu serialises kernels, so its durations are not the budget's; the time of every
part comes from block_budget, and this tool's durations are used only as ratios within a kernel
(the tail estimate, the stall shares).

`report` classes every kernel (the skinny NVFP4 projection by its template and grid, the head, the
GDN tree walk, the WY chain/tree, the verify convolution, attention, norms, the drafter's kernels,
torch glue), and per class prints: launches a block, bytes read against the weight bytes it should
read (the bytes ratio; > 1.05 is sector waste), GB/s at ncu's own duration, achieved occupancy,
registers, the dominant stall and its share, local-memory (spill) sectors, and the tail estimate
1 - mean(active cycles) / max(active cycles) over the SMs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The metric sets. BASIC is one pass on GB10 (checked by the hold's probe before the run); STALLS
# are the warp-state ratios of Nsight's WarpStateStats section, a pass of their own.
BASIC = ["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
         "sm__warps_active.avg.pct_of_peak_sustained_active", "sm__cycles_active.avg",
         "sm__cycles_active.max", "lts__t_sector_hit_rate.pct",
         "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
         "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
         "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "launch__registers_per_thread",
         "launch__grid_size", "launch__block_size"]
STALL_REASONS = ["long_scoreboard", "short_scoreboard", "wait", "barrier", "not_selected",
                 "no_instruction", "math_pipe_throttle", "lg_throttle", "mio_throttle", "drain",
                 "membar", "branch_resolving", "dispatch_stall", "imc_miss", "selected",
                 "sleeping", "tex_throttle", "misc"]
STALLS = [f"smsp__average_warps_issue_stalled_{r}_per_issue_active.ratio" for r in STALL_REASONS]

# Weight shapes of the target and the drafter (N, K) -> name; NVFP4 bytes are N*K/2 values plus an
# fp8 scale per 16 values; the e4m3 head is N*K bytes plus a scale a row.
SHAPES = {(17408, 5120): "gate|up", (5120, 17408): "down", (10240, 5120): "gdn qkv",
          (6144, 5120): "gdn z", (5120, 6144): "out|o_proj", (12288, 5120): "attn q",
          (1024, 5120): "attn k|v", (5120, 1024 * 5): "drafter fc"}


def nvfp4_bytes(n: int, k: int) -> int:
    return n * k // 2 + n * k // 16


def shape_of(bytes_read: float, n_est: float | None = None,
             tol: float = 0.06) -> tuple[str, int] | None:
    """The projection whose NVFP4 weight a kernel reading this many bytes streams (within tol).
    gate|up and down stream the same bytes (so do GDN z and out): `n_est`, the output width the
    launch covers (grid x 8 x NT for the skinny kernel), picks between them."""
    cands = []
    for i, ((n, k), name) in enumerate(SHAPES.items()):
        b = nvfp4_bytes(n, k)
        err = abs(bytes_read - b) / b
        if err <= tol:
            cands.append((abs(n - n_est) if n_est else 0, err, i, name, b))
    if not cands:
        return None
    *_, name, b = min(cands)
    return name, b


# ---------------------------------------------------------------- run (under ncu)


def run(a) -> None:
    import torch
    from transformers import AutoTokenizer

    from tools import profile_cycle as pc
    from tools.block_budget import PROMPTS

    cfg, eng, drafter, arms, ng, k = pc.build(a)
    drafter.learn_cost = False
    tk = AutoTokenizer.from_pretrained(cfg.path)

    def ids(text: str):
        s = tk.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
        return tk(s, return_tensors="pt").input_ids[0].cuda()

    for fixed in (8, 16):
        drafter.fixed = fixed
        pc.cycle(eng, drafter, ids(PROMPTS["chat"]), a.warm, k, pc.Phases(strict=False))
    drafter.fixed = a.fixed
    if a.precapture and eng._graphs_for(2, 0) is not None:
        with torch.no_grad():
            n_g = eng._graphs.precapture(widths=range(2, a.precapture + 1))
        print(f"[ncu-budget] verify graphs 2..{a.precapture}: {n_g}", flush=True)

    class Window(pc.Phases):
        """Profiles blocks skip+1 .. skip+blocks; a marker kernel opens each one."""

        def __init__(self):
            super().__init__(strict=False)
            self.n = 0
            self.on = False
            self.shapes = []

        def start_block(self):
            if self.n == a.skip:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                self.on = True
            if self.on and self.n == a.skip + a.blocks:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                self.on = False
            if self.on:
                torch.cuda._sleep(1)
            self.n += 1
            super().start_block()

        def stop(self):
            if self.on:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                self.on = False

    meta = {"workloads": [], "skip": a.skip, "blocks": a.blocks,
            "env": {k_: v for k_, v in sorted(os.environ.items()) if k_.startswith("QWEN38_")}}
    for name in a.workloads.split(","):
        ph = Window()
        undo = pc.instrument(arms, ng, ph)
        # the tree each block verified: its node count is the row count of that block's verify
        rows = []
        orig = eng.forward_tree

        def forward_tree(block, parents, start, *x, **kw):
            if ph.on:
                rows.append(int(block.numel()))
            return orig(block, parents, start, *x, **kw)

        eng.forward_tree = forward_tree
        try:
            out, st = pc.cycle(eng, drafter, ids(PROMPTS[name]), a.max_new, k, ph)
        finally:
            ph.stop()
            eng.forward_tree = orig
            undo()
        profiled = min(a.blocks, max(ph.n - a.skip, 0))
        meta["workloads"].append({"name": name, "profiled_blocks": profiled, "verify_rows": rows,
                                  "blocks": len(ph.block_ms), "tokens": st["tokens"],
                                  "block_ms": sum(ph.block_ms) / max(len(ph.block_ms), 1)})
        print(f"[ncu-budget] {name}: {len(ph.block_ms)} blocks, profiled {profiled} "
              f"(verify rows {rows})", flush=True)
    torch.cuda.synchronize()
    if a.meta:
        json.dump(meta, open(a.meta, "w"), indent=1)


# ---------------------------------------------------------------- report


def read_csv(path: str) -> list[dict]:
    """Ncu's `--csv --page raw` log: a header row, a units row, one row a kernel. Lines ncu itself
    prints (==PROF==, warnings) are skipped."""
    lines = [ln for ln in open(path, errors="replace") if ln.startswith('"')]
    rows = list(csv.reader(lines))
    if not rows:
        return []
    head = rows[0]
    out = []
    for r in rows[1:]:
        if len(r) != len(head) or r[0] == "" or not r[0].isdigit():
            continue                      # the units row, a truncated line
        out.append(dict(zip(head, r)))
    return out


def num(x) -> float:
    if x is None or x == "" or x == "n/a":
        return float("nan")
    return float(str(x).replace(",", ""))


def kclass(name: str) -> str:
    """A kernel's class from its (demangled) name."""
    n = name
    m = re.search(r"skinny_kernel<(\d+), (\d+), (\d+), (\d+), (\d+)(?:, (\d+))?(?:, (\d+))?>", n)
    if m:
        nt, mt, wk, pf = m.group(1), m.group(2), m.group(3), m.group(4)
        kr = m.group(7) or "0"
        return f"skinny nt{nt} mt{mt} wk{wk} pf{pf}" + (" kr1" if kr != "0" else "")
    if "skinny" in n:
        return "skinny (other)"
    for key, cls in (("_head_gemm_fp8", "head gemm fp8"), ("_head_gemv_fp8", "head gemv fp8"),
                     ("_head_gemv", "head gemv"), ("_gdn_tree_step", "gdn tree walk"),
                     ("_tree_step", "gdn tree step"), ("_block_step", "gdn block step"),
                     ("_wy_prep", "gdn wy prep"), ("_wy_apply", "gdn wy apply"),
                     ("_verify_conv", "gdn verify conv"), ("_verify_gate", "gdn verify gate"),
                     ("_gdn_commit", "gdn commit"), ("_pending", "gdn pending"),
                     ("_attn_split", "attn split"), ("_attn_combine", "attn combine"),
                     ("_add_rms_norm", "add+rmsnorm"), ("_rms_norm_gated", "rmsnorm gated"),
                     ("_rms_norm", "rmsnorm"), ("spin_kernel", "marker"),
                     ("nvjet", "cublas"), ("gemm", "cublas"), ("gemv", "cublas gemv"),
                     ("elementwise", "torch elementwise"), ("reduce", "torch reduce"),
                     ("index", "torch index"), ("copy", "torch copy"), ("cat", "torch cat")):
        if key in n:
            return cls
    return "other: " + n.split("(")[0][:48]


def split_blocks(rows: list[dict], meta: dict) -> list[tuple[str, int, list[dict]]]:
    """(workload, block, kernels) from the marker kernels, in profile order."""
    out, cur, i = [], None, -1
    order = [(w["name"], b) for w in meta["workloads"] for b in range(w["profiled_blocks"])]
    for r in sorted(rows, key=lambda r: int(r["ID"])):
        if kclass(r.get("Kernel Name", "")) == "marker":
            i += 1
            cur = (order[i] if i < len(order) else ("?", i)) + ([],)
            out.append(cur)
            continue
        if cur is not None:
            cur[2].append(r)
    return out


def summarise(blocks, stalls: bool = True) -> dict:
    """Per (workload arm, class, shape): launches a block and the counters' medians."""
    agg = defaultdict(list)
    nblk = defaultdict(int)
    for w, b, ks in blocks:
        nblk[w] += 1
        for r in ks:
            cls = kclass(r.get("Kernel Name", ""))
            rd = num(r.get("dram__bytes_read.sum"))
            m = re.match(r"skinny nt(\d+)", cls)
            n_est = num(r.get("launch__grid_size")) * 8 * int(m.group(1)) if m else None
            shp = shape_of(rd, n_est if n_est == n_est else None) if m else None
            key = (w, cls, shp[0] if shp else "")
            agg[key].append((r, shp[1] if shp else float("nan")))
    table = {}
    for (w, cls, shp), items in agg.items():
        rs = [r for r, _ in items]

        def med(metric):
            xs = [num(r.get(metric)) for r in rs]
            xs = [x for x in xs if x == x]
            return statistics.median(xs) if xs else float("nan")

        dur_ns = med("gpu__time_duration.sum")
        rd = med("dram__bytes_read.sum")
        theo = items[0][1]
        act_avg, act_max = med("sm__cycles_active.avg"), med("sm__cycles_active.max")
        st = {}
        if stalls:
            for reason, m in zip(STALL_REASONS, STALLS):
                v = med(m)
                if v == v:
                    st[reason] = v
        tot = sum(v for k, v in st.items() if k != "selected")
        top = sorted(((v / tot if tot else 0.0, k) for k, v in st.items() if k != "selected"),
                     reverse=True)[:3]
        table[(w, cls, shp)] = {
            "per_block": len(rs) / max(nblk[w], 1), "dur_us": dur_ns / 1e3, "bytes_read": rd,
            "bytes_ratio": rd / theo if theo == theo and theo else float("nan"),
            "gbps": rd / dur_ns if dur_ns else float("nan"),
            "occupancy": med("sm__warps_active.avg.pct_of_peak_sustained_active"),
            "regs": med("launch__registers_per_thread"), "grid": med("launch__grid_size"),
            "block": med("launch__block_size"), "l2_hit": med("lts__t_sector_hit_rate.pct"),
            "sect_per_req": (med("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
                             / med("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")),
            "local_sectors": med("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"),
            "tail": 1.0 - act_avg / act_max if act_max else float("nan"),
            "stall_top": top}
    return table


def fmt(table: dict) -> str:
    lines = [f"{'arm':<6} {'class':<28} {'shape':<11} {'n/blk':>5} {'us':>7} {'MB':>7} {'ratio':>6} "
             f"{'GB/s':>6} {'occ%':>5} {'regs':>4} {'L2hit':>5} {'sec/rq':>6} {'spill':>7} "
             f"{'tail':>5}  top stalls"]
    for (w, cls, shp), v in sorted(table.items(), key=lambda kv: (kv[0][0], -kv[1]["per_block"]
                                                                   * kv[1]["dur_us"])):
        tops = ", ".join(f"{k} {s:.0%}" for s, k in v["stall_top"])
        lines.append(f"{w:<6} {cls:<28} {shp:<11} {v['per_block']:5.0f} {v['dur_us']:7.1f} "
                     f"{v['bytes_read'] / 1e6:7.2f} {v['bytes_ratio']:6.3f} {v['gbps']:6.1f} "
                     f"{v['occupancy']:5.1f} {v['regs']:4.0f} {v['l2_hit']:5.1f} "
                     f"{v['sect_per_req']:6.2f} {v['local_sectors']:7.0f} {v['tail']:5.2f}  {tops}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default=None)
    r.add_argument("--nvfp4", default=os.environ.get("QWEN38_NVFP4"))
    r.add_argument("--fp8-head", default=os.environ.get("QWEN38_FP8_HEAD"))
    r.add_argument("--ckpt8", required=True)
    r.add_argument("--ckpt16", required=True)
    r.add_argument("--corpus", default="")
    r.add_argument("--max-len", type=int, default=4096)
    r.add_argument("--max-new", type=int, default=96)
    r.add_argument("--warm", type=int, default=32)
    r.add_argument("--workloads", default="prose,chat,code")
    r.add_argument("--skip", type=int, default=12, help="blocks before the profiled ones (the "
                   "wide arm's 24-node tree starts after 32 committed tokens)")
    r.add_argument("--blocks", type=int, default=2)
    r.add_argument("--precapture", type=int, default=32)
    r.add_argument("--fixed", type=int, default=0)
    r.add_argument("--meta", default="")
    p = sub.add_parser("report")
    p.add_argument("csv")
    p.add_argument("--meta", required=True)
    p.add_argument("--json", default="")
    a = ap.parse_args()
    if a.cmd == "run":
        a.latch = a.drop_idle = True
        run(a)
        return
    meta = json.load(open(a.meta))
    rows = read_csv(a.csv)
    blocks = split_blocks(rows, meta)
    print(f"[ncu-budget] {len(rows)} kernels, {len(blocks)} profiled blocks "
          f"({', '.join(f'{w} {b}' for w, b, _ in blocks)})")
    t = summarise(blocks)
    print(fmt(t))
    if a.json:
        json.dump({"|".join(k): v for k, v in t.items()}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
