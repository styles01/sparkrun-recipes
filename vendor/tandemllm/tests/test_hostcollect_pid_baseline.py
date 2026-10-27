"""A recycled pid must not get a cpu% computed against the dead process's counters (audit P2 #8).

The cpu% baseline in HostCollector._proc_info used to be keyed by the bare pid; the key now
carries the process's starttime (/proc/<pid>/stat field 22), so a pid reuse starts a fresh
baseline instead of dividing against the old process's cpu seconds. CPU only, a fake /proc.

Run: python tests/test_hostcollect_pid_baseline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proxy import hostcollect as hc  # noqa: E402

BTIME = 1700000000


def collector(d):
    return hc.HostCollector(proc=os.path.join(d, "proc"), sys_root=os.path.join(d, "sys"),
                            psutil_mod=None, nvml_mod=None, smi="/nonexistent/nvidia-smi")


def write_stat(d, pid, utime, stime, start_ticks):
    pdir = os.path.join(d, "proc", str(pid))
    os.makedirs(pdir, exist_ok=True)
    # the 19 fields after ')': state ppid pgrp session tty tpgid flags minflt cminflt majflt
    # cmajflt utime stime cutime cstime priority nice num_threads itrealvalue starttime
    after = (["S", "1", "1", "1", "0", "0", "0", "0", "0", "0", "0", str(utime),
              "0", "0", "0", "0", "2", "0", "0", str(start_ticks)])
    with open(os.path.join(pdir, "stat"), "w") as f:
        f.write(f"{pid} (worker) " + " ".join(after) + "\n")


def test_baseline_and_recycled_pid():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "proc"), exist_ok=True)
        with open(os.path.join(d, "proc", "stat"), "w") as f:
            f.write(f"cpu  0 0 0 0 0 0 0 0 0 0\nbtime {BTIME}\n")
        c = collector(d)
        tick = c._tick

        # the same process over two samples: a rate, as before
        write_stat(d, 4242, 0, 0, 500000)
        assert c._proc_info(4242, 100.0)["cpu"] is None, "no rate before a second sample"
        write_stat(d, 4242, 5 * tick, 0, 500000)
        i = c._proc_info(4242, 110.0)
        assert i["cpu"] == 0.5, f"utime+stime rose by 5 s in 10 s: {i['cpu']}"

        # pid 4242 dies and is reused: new starttime, counters start again from zero.
        # The old code keyed by the bare pid and computed a spike of hundreds of percent.
        write_stat(d, 4242, 1000 * tick, 0, 900000)
        i = c._proc_info(4242, 110.1)
        assert i["cpu"] is None, f"a recycled pid must have no rate on its first sample: {i}"

        # ... and its own second sample is a rate of the new process alone
        write_stat(d, 4242, 1001 * tick, 0, 900000)
        i = c._proc_info(4242, 111.1)
        assert i["cpu"] == 1.0, f"the new process rates against its own baseline: {i['cpu']}"

        # the dead process's baseline sits under its own old key, never the new one's
        assert (4242, 500000.0) in c._prev_proc and len(c._prev_proc) == 2
        return "pid 4242 recycled: no rate, then 1.0 of its own; key (pid, starttime)"


def test_psutil_path_keys_on_create_time():
    class FakeProc:
        def __init__(self, pid, create, cpu):
            self.pid, self._c, self._cpu = pid, create, cpu

        def oneshot(self):
            class _Ctx:
                def __enter__(_):
                    return None

                def __exit__(_, *a):
                    return False
            return _Ctx()

        def name(self):
            return "worker"

        def cmdline(self):
            return ["worker"]

        def memory_info(self):
            return type("M", (), {"rss": 1000})()

        def num_threads(self):
            return 1

        def num_fds(self):
            return 3

        def create_time(self):
            return self._c

        def cpu_times(self):
            return type("T", (), {"user": self._cpu, "system": 0.0})()

    class FakePsutil:
        def __init__(self):
            self.create = 1000.0
            self.cpu = 0.0

        def Process(self, pid):
            return FakeProc(pid, self.create, self.cpu)

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ps = FakePsutil()
        c = hc.HostCollector(proc=d, psutil_mod=ps, nvml_mod=None)
        assert c._proc_info(7, 2000.0)["cpu"] is None
        ps.cpu = 5.0
        assert c._proc_info(7, 2005.0)["cpu"] == 1.0
        # pid 7 reused by a process created later, counters from zero
        ps.create, ps.cpu = 3000.0, 6.0
        i = c._proc_info(7, 3005.0)
        assert i["cpu"] is None, f"a recycled pid gets no rate from the old create_time: {i}"
        return "the psutil path keys on create_time the same way"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:45s} ok   {fn() or ''}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"  {name:45s} FAIL {exc!r}")
    print(f"{2 - fails} passed" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
