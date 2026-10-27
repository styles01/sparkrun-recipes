"""The host collector against the machine's own tools, and what it costs (P1.5's gate).

    python tools/hostcollect_check.py [--port 8000] [--samples 60] [--out FILE]

Reads the collector once and, at the same moment, `/proc/meminfo` and `nvidia-smi`, and checks:
memory total and available within 1 % of `/proc/meminfo`, the GPU's temperature, power, SM clock
and use against `nvidia-smi` (those move between two reads, so within 5 % or 3 units), the GPU
memory of each process `nvidia-smi --query-compute-apps` lists within 1 %, and, with `--port`,
that the process behind the port is found with its tree. Then it times `--samples` forced
samples with a process scan and reports the CPU per sample and the share of one core at the
real cadence (one sample in 5 s, one scan in 15 s). Read-only: no GPU work, no writes but
`--out`. Exit 0 when every check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy import hostcollect as hc  # noqa: E402


def _meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            out[k] = int(v.split()[0]) * 1024
    return out


def _smi(args):
    try:
        r = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return [[x.strip() for x in line.split(",")] for line in r.stdout.splitlines() if line.strip()]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None, help="a backend's port to find")
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    c = hc.HostCollector()
    watch = [{"backend": "probe", "port": a.port}] if a.port else []
    checks, ok = [], True

    def check(name, good, detail):
        nonlocal ok
        ok = ok and bool(good)
        checks.append({"check": name, "pass": bool(good), "detail": detail})
        print(f"{'PASS' if good else 'FAIL'} {name}: {detail}", flush=True)

    c.sample(watch, force=True)
    time.sleep(1.0)
    s = c.sample(watch, force=True)
    mi = _meminfo() if os.path.exists("/proc/meminfo") else {}
    if mi:
        for key, field in (("MemTotal", "total_bytes"), ("MemAvailable", "available_bytes")):
            got, ref = s["memory"][field], mi[key]
            check(f"memory {key}", got is not None and abs(got - ref) <= 0.01 * ref,
                  f"collector {got}, /proc/meminfo {ref}")
    rows = _smi(["--query-gpu=index,name,temperature.gpu,power.draw,clocks.sm,utilization.gpu",
                 "--format=csv,noheader,nounits"])
    if rows:
        for r in rows:
            g = next((x for x in s["gpus"] if x["index"] == int(r[0])), None)
            check(f"gpu {r[0]} present", g is not None, f"{r[1]} via {s['gpu_source']}")
            if g is None:
                continue
            for i, field, scale in ((2, "temperature_c", 1), (3, "power_w", 1),
                                    (4, "sm_clock_mhz", 1), (5, "utilization", 0.01)):
                ref = _num(r[i])
                got = g[field]
                if ref is None:
                    check(f"gpu {r[0]} {field}", True,
                          f"nvidia-smi N/A, collector {got}")
                    continue
                ref *= scale
                tol = max(0.05 * abs(ref), 3 * scale)
                check(f"gpu {r[0]} {field}", got is not None and abs(got - ref) <= tol,
                      f"collector {got}, nvidia-smi {ref} (a live value, read a moment apart)")
            check(f"gpu {r[0]} memory", True,
                  "unified (no device figure)" if g["memory_unified"]
                  else f"{g['memory_used_bytes']} of {g['memory_total_bytes']}")
        apps = _smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        if apps and a.port:
            p = s["processes"][0]
            tree = {p["pid"]: p["gpu_memory_bytes"]}
            tree.update({k["pid"]: k["gpu_memory_bytes"] for k in p["children"]})
            for pid_s, mib in apps:
                if not pid_s.isdigit() or int(pid_s) not in tree or _num(mib) is None:
                    continue
                ref = _num(mib) * 2 ** 20
                got = tree[int(pid_s)]
                check(f"gpu memory of pid {pid_s}",
                      got is not None and abs(got - ref) <= 0.01 * ref,
                      f"collector {got}, nvidia-smi {int(ref)}")
    if a.port:
        p = s["processes"][0]
        check(f"process behind :{a.port}", p["found"],
              f"pid {p['pid']} via {p['via']}, {p['name']}, {len(p['children'])} children, "
              f"RSS {p['total_rss_bytes']}, GPU {p['gpu_memory']} {p['total_gpu_memory_bytes']}")
    # the cost: forced samples, each with a process scan
    t0 = time.process_time()
    w0 = time.perf_counter()
    for _ in range(a.samples):
        c.sample(watch, force=True)
    cpu_ms = (time.process_time() - t0) / a.samples * 1000
    wall_ms = (time.perf_counter() - w0) / a.samples * 1000
    share = cpu_ms / 5000                      # at most one sample with a scan every 5 s
    real = (cpu_ms - (c.cost["last_scan_ms"] or 0)) / 5000 + (c.cost["last_scan_ms"] or 0) / 15000
    check("cost under 0.5 % of one CPU", share < 0.005,
          f"{cpu_ms:.2f} ms CPU and {wall_ms:.2f} ms wall per sample with a scan; at most "
          f"{share * 100:.3f} % of a core if every sample scanned, {real * 100:.3f} % at the "
          f"real cadence (scan {c.cost['last_scan_ms']} ms)")
    out = {"checks": checks, "pass": ok, "psutil": c.psutil is not None,
           "nvml": c._nvml is not None, "gpu_source": s["gpu_source"],
           "cpu_ms_per_sample": round(cpu_ms, 3), "wall_ms_per_sample": round(wall_ms, 3),
           "share_of_core_max": round(share, 6), "share_of_core_real": round(real, 6)}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
