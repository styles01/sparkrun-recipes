#!/usr/bin/env python3
"""The NVMe under the session spill and the disk tier, measured without the engine.

Why (2026-10-04 12:59): one eviction wrote 7.0 GB in 4.5 s right after the disk tier
had written 17.6 GB, where the same path wrote 5.2 GB in 1.03 s otherwise. This writes files the
way both writers do (engine/disk_cache.py `write_file`, O_DIRECT, the spill's 1,024-row blocks
without an fsync each and the files fsynced at the end; the tier's chunk files with one fsync
each) and times every file, so a cliff after some number of GB shows up as a row.

Phases (each prints GB, seconds, GB/s, and the slowest files):
  spill   --spill-gb of spill-shaped files
  tier    --tier-gb of tier-shaped files, then at once a spill of --spill-gb
  rest    --rest-s of nothing, then a spill of --spill-gb again
  gpu-paced  the same with the tier's files paced to --pace-mbs on average (after a first burst
          that does not happen here: the pacing is from the first file), first save and the save
          after the tier only
  gpu     the real save (engine/spill.py `SessionSpill._save`) of a --gpu-rows conversation from
          buffers on the GPU, three times in one process: the first save (staging allocated and
          pinned), a second one, and one right after --tier-gb of tier-shaped files; each with its
          parts (`info["parts"]`, ms). Needs the GPU: run it beside the idle engine under the box lock.

Run on the box (no GPU, no engine; the files go under --dir and are deleted at the end):
  python tools/spill_io_probe.py --dir ~/d6-io --json ~/d6-io.json
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import disk_cache as dc  # noqa: E402

MB = 1 << 20
GB = 1e9


def _buf(size: int):
    m = mmap.mmap(-1, size)
    # every page touched once: the first write must not pay for faults
    step = mmap.PAGESIZE
    for off in range(0, size, step):
        m[off] = (off // step) & 0xFF
    return m


def write_set(d: str, tag: str, total: float, size: int, fsync_each: bool, buf,
              pace_mbs: float = 0.0) -> dict:
    os.makedirs(d, exist_ok=True)
    n = max(1, int(total * GB // size))
    times, paths = [], []
    t0 = time.perf_counter()
    for i in range(n):
        p = os.path.join(d, f"{tag}-{i:05d}.blk")
        if pace_mbs > 0:                         # at most `pace_mbs` on average
            ahead = i * size / (pace_mbs * 1e6) - (time.perf_counter() - t0)
            if ahead > 0:
                time.sleep(ahead)
        a = time.perf_counter()
        direct = dc.write_file(p, [memoryview(buf)[:size]], fsync=fsync_each)
        times.append(time.perf_counter() - a)
        paths.append(p)
    t_write = time.perf_counter() - t0
    if not fsync_each:
        for p in paths:                          # engine/spill.py `_fsync_dir_files`
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    s = time.perf_counter() - t0
    nbytes = n * size
    rates = [size / t / GB for t in times]
    # the file at which the rate first falls under half the first tenth's median
    head = sorted(rates[:max(1, n // 10)])[len(rates[:max(1, n // 10)]) // 2]
    cliff = next((i for i, r in enumerate(rates) if r < 0.5 * head), None)
    return {"tag": tag, "files": n, "file_mb": round(size / MB, 1), "gb": round(nbytes / GB, 2),
            "s": round(s, 3), "write_s": round(t_write, 3), "fsync_s": round(s - t_write, 3),
            "gbs": round(nbytes / s / GB, 2), "direct": bool(direct),
            "file_gbs_first10pct_median": round(head, 2),
            "file_gbs_min": round(min(rates), 2), "file_gbs_max": round(max(rates), 2),
            "cliff_at_gb": None if cliff is None else round(cliff * size / GB, 2),
            "rates": [round(r, 2) for r in rates]}


def gpu_saves(d: str, a, buf, tsz: int, paced: bool = False) -> list[dict]:
    import types
    import torch
    from engine import spill
    rows, C = int(a.gpu_rows), 1024
    # 122 KB a row, as the live engine's target and draft KV (two buffers, axis 3)
    shape = (16, 1, 4, rows, 480)
    k = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    for t in (k, v):
        t.view(torch.int16).random_(-30000, 30000)
    S = torch.ones(48, 48, 128, 128, dtype=torch.float32, device="cuda")
    conv = torch.randn(48, 8192, 3, dtype=torch.bfloat16, device="cuda")
    eng = types.SimpleNamespace(kv=types.SimpleNamespace(k=k, v=v, fp8=False),
                                state=types.SimpleNamespace(S=S, conv=conv))
    torch.cuda.synchronize()
    sp = spill.SessionSpill(d, chunk=C, fingerprint="probe", tier=None,
                            settings={"budget_gb": 100.0}, floor_bytes=0).open()
    anchors = {b: (S, conv) for b in range(C, rows + 1, C)}
    out = []

    def one(tag, owner):
        info = sp._save(eng, None, tokens=list(range(rows)), n=rows, rows_from=None,
                        anchors=anchors, end=None, taint=None, owner=owner, reason="probe",
                        extra=None, resident=True)
        out.append({"tag": tag, "gb": round(info["bytes_written"] / GB, 2), "ms": info["ms"],
                    "gbs": round(info["bytes_written"] / info["ms"] / 1e6, 2),
                    "parts": info["parts"], "writer_ms": info["writer_ms"]})

    try:
        one("gpu-first-save", "p1")
        if not paced:
            one("gpu-second-save", "p2")
        sp.clear()
        t = write_set(os.path.join(d, "t"), "tier-paced" if paced else "tier", a.tier_gb, tsz,
                      True, buf, pace_mbs=a.pace_mbs if paced else 0.0)
        out.append({k_: v_ for k_, v_ in t.items() if k_ != "rates"} | {"ms": t["s"] * 1e3,
                                                                        "parts": {}, "writer_ms": 0})
        one("gpu-save-after-paced-tier" if paced else "gpu-save-after-tier", "p3")
        shutil.rmtree(os.path.join(d, "t"), ignore_errors=True)
    finally:
        sp.close()
        del k, v, S, conv
        torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--spill-gb", type=float, default=7.0)
    ap.add_argument("--tier-gb", type=float, default=17.6)
    ap.add_argument("--spill-mb", type=float, default=122.0, help="a spill block file (1,024 rows)")
    ap.add_argument("--tier-mb", type=float, default=112.0, help="a tier chunk file")
    ap.add_argument("--rest-s", type=float, default=60.0)
    ap.add_argument("--phases", default="spill,tier,rest")
    ap.add_argument("--gpu-rows", type=int, default=58278, help="the gpu phase's conversation")
    ap.add_argument("--pace-mbs", type=float, default=92.0,
                    help="gpu-paced: the tier's average rate (92 MB/s: 17.5 GB in 190 s)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    d = os.path.realpath(os.path.expanduser(a.dir))
    if os.path.exists(d):
        raise SystemExit(f"{d} exists; give a new directory")
    ssz = int(a.spill_mb * MB) // 4096 * 4096
    tsz = int(a.tier_mb * MB) // 4096 * 4096
    buf = _buf(max(ssz, tsz))
    out = {"dir": d, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "phases": []}

    def row(r):
        out["phases"].append(r)
        print(f"[io] {r['tag']}: {r['gb']} GB in {r['s']} s = {r['gbs']} GB/s (write {r['write_s']} s, "
              f"fsync {r['fsync_s']} s; files {r['file_gbs_min']}-{r['file_gbs_max']} GB/s, "
              f"first tenth {r['file_gbs_first10pct_median']}, cliff at {r['cliff_at_gb']} GB)",
              flush=True)

    try:
        for ph in [p.strip() for p in a.phases.split(",") if p.strip()]:
            if ph == "spill":
                row(write_set(os.path.join(d, "s0"), "spill", a.spill_gb, ssz, False, buf))
                shutil.rmtree(os.path.join(d, "s0"))
            elif ph == "tier":
                row(write_set(os.path.join(d, "t"), "tier", a.tier_gb, tsz, True, buf))
                row(write_set(os.path.join(d, "s1"), "spill-after-tier", a.spill_gb, ssz, False, buf))
                shutil.rmtree(os.path.join(d, "t"))
                shutil.rmtree(os.path.join(d, "s1"))
            elif ph == "rest":
                time.sleep(a.rest_s)
                row(write_set(os.path.join(d, "s2"), f"spill-after-{a.rest_s:g}s", a.spill_gb, ssz,
                              False, buf))
                shutil.rmtree(os.path.join(d, "s2"))
            elif ph in ("gpu", "gpu-paced"):
                for r in gpu_saves(os.path.join(d, "g"), a, buf, tsz, paced=ph == "gpu-paced"):
                    out["phases"].append(r)
                    print(f"[io] {r['tag']}: {r['gb']} GB in {r['ms']:.0f} ms = {r['gbs']} GB/s; "
                          f"parts {r['parts']}, writer {r['writer_ms']} ms", flush=True)
            elif ph.startswith("sleep"):
                time.sleep(float(ph[5:] or 0))
            else:
                raise SystemExit(f"unknown phase {ph}")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if a.json:
        with open(os.path.expanduser(a.json), "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
