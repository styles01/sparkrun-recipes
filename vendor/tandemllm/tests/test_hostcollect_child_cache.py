"""The shared host collector's CPU baselines (parent + children) survive a scan from another
view: the per-view `live`-set prune used to wipe them on every alternating rescan, so a
Processes view and a GPU view reset each other's children forever. The prune is by TTL now.

    python tests/test_hostcollect_child_cache.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proxy import hostcollect as hc  # noqa: E402

TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
MEMINFO = "MemTotal: 128452068 kB\nMemFree: 2203344 kB\nMemAvailable: 40112640 kB\n"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _pid(d: str, pid: int, ppid: int, comm: str, utime_ticks: int) -> None:
    """One fake /proc/<pid> entry; rewrite it between samples to advance its cpu seconds."""
    pd = os.path.join(d, str(pid))
    os.makedirs(os.path.join(pd, "fd"), exist_ok=True)
    # rest[1]/rest[11]/rest[12]/rest[19]: ppid, utime, stime, starttime after the comm
    fields = (["S", str(ppid)] + ["0"] * 9 + [str(utime_ticks), str(utime_ticks // 3)]
              + ["0"] * 6 + [str(50 * TICK)] + ["0"] * 20)
    with open(os.path.join(pd, "stat"), "w") as f:
        f.write(f"{pid} ({comm}) " + " ".join(fields) + "\n")
    with open(os.path.join(pd, "status"), "w") as f:
        f.write(f"Name:\t{comm}\nVmRSS:\t10000 kB\nThreads:\t2\n")
    with open(os.path.join(pd, "cmdline"), "w") as f:
        f.write(f"{comm}\0--x\0")


def _proc_tree(d: str) -> None:
    with open(os.path.join(d, "meminfo"), "w") as f:
        f.write(MEMINFO)
    with open(os.path.join(d, "stat"), "w") as f:
        f.write("cpu  1000 0 500 8000 500 0 0 0 0 0\nbtime 1700000000\n")
    # view A watches 7001 (child 7002); view B watches 7003 — different scan keys
    for pid, ppid, comm in ((7001, 1, "vllm"), (7002, 7001, "EngineCore"), (7003, 1, "sshd")):
        _pid(d, pid, ppid, comm, 100 * TICK)


def _bump(d: str, pid: int, utime_ticks: int) -> None:
    _pid(d, pid, {7002: 7001}.get(pid, 1), {7001: "vllm", 7002: "EngineCore",
                                            7003: "sshd"}[pid], utime_ticks)


def _collector(d: str, clock: Clock) -> hc.HostCollector:
    return hc.HostCollector(proc=d, psutil_mod=None, nvml_mod=None, smi="/nonexistent/nvidia-smi",
                            clock=clock, proc_ttl=60.0)


def _entry(doc: dict, backend: str) -> dict:
    return next(p for p in doc["processes"] if p["backend"] == backend)


def test_children_survive_the_other_view():
    """A: watch a parent with a child, then alternate views: the child keeps its baseline."""
    d = tempfile.mkdtemp(prefix="hc-children-")
    try:
        _proc_tree(d)
        clock = Clock()
        c = _collector(d, clock)
        a = [{"backend": "a", "pid": 7001}]
        b = [{"backend": "b", "pid": 7003}]
        doc = c.sample(a)                                     # first scan: no baselines yet
        assert _entry(doc, "a")["cpu"] is None
        assert _entry(doc, "a")["children"][0]["pid"] == 7002
        for _ in range(2):                                    # interleave the two views
            _bump(d, 7001, 120 * TICK)
            _bump(d, 7002, 120 * TICK)
            clock.t += 5.0
            c.sample(b)                                       # other view's key: full rescan,
                                                              # used to drop A's cached children
            _bump(d, 7001, 150 * TICK)
            _bump(d, 7002, 150 * TICK)
            clock.t += 5.0
            doc = c.sample(a)
        e = _entry(doc, "a")
        assert e["cpu"] is not None, "parent cpu% came back after the other view scanned"
        assert e["children"][0]["cpu"] is not None, "the child kept its cached baseline"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_stale_baselines_are_dropped_by_ttl():
    """The cache is still bounded: a scan after the TTL forgets the pids it no longer sees."""
    d = tempfile.mkdtemp(prefix="hc-children-")
    try:
        _proc_tree(d)
        clock = Clock()
        c = _collector(d, clock)
        a = [{"backend": "a", "pid": 7001}]
        b = [{"backend": "b", "pid": 7003}]
        c.sample(a)
        clock.t += 5.0
        c.sample(b)
        assert {p for p, _ in c._prev_proc} == {7001, 7002, 7003}, "both views kept, no per-view prune"
        clock.t += 120.0                                      # past the 60 s TTL
        _bump(d, 7001, 150 * TICK)
        _bump(d, 7002, 150 * TICK)
        doc = c.sample(a, force=True)
        assert 7003 not in {p for p, _ in c._prev_proc}, "the stale baseline of the other view was dropped"
        assert _entry(doc, "a")["cpu"] is not None, "the old baseline served as the wide window"
        assert {7001, 7002} <= {p for p, _ in c._prev_proc}, "the freshly sampled ones stay"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_no_baseline_leak_without_any_view():
    """A pid watched by no view at all ages out too: nothing is kept just because it was seen."""
    d = tempfile.mkdtemp(prefix="hc-children-")
    try:
        _proc_tree(d)
        clock = Clock()
        c = _collector(d, clock)
        c.sample([{"backend": "a", "pid": 7001}])
        clock.t += 120.0
        c.sample([{"backend": "b", "pid": 7003}])             # different key: rescan prunes
        assert {p for p, _ in c._prev_proc} <= {7003}, "gone past the TTL, not because of the live set"
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"{name:48s} ok")
            passed += 1
        except Exception:                                     # noqa: BLE001
            print(f"{name:48s} FAIL")
            traceback.print_exc()
    print(f"{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
