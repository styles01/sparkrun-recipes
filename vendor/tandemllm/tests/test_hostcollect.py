"""The host collector (P1.5): recorded /proc files, a fake NVML, a fake nvidia-smi, the port
match on a socket this test opens, the remote collector and the exporter pages, placement, the
config field, the contract, and the collector's own cost.

    python tests/test_hostcollect.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import stat
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proxy import config as config_mod  # noqa: E402
from proxy import hostcollect as hc  # noqa: E402
from proxy import scrape  # noqa: E402
from tools import contract_check  # noqa: E402

HOSTS_SCHEMA = contract_check.load("system")["properties"]["hosts"]


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _check(hosts):
    errs = contract_check.validate(hosts, HOSTS_SCHEMA)
    assert not errs, errs[:5]


# ---------------------------------------------------------------------- a recorded /proc

MEMINFO = """MemTotal:       128452068 kB
MemFree:         2203344 kB
MemAvailable:   40112640 kB
SwapTotal:      16777212 kB
SwapFree:       16000000 kB
"""
STAT1 = "cpu  1000 0 500 8000 500 0 0 0 0 0\ncpu0 1 1 1 1 1 0 0 0 0 0\nbtime 1700000000\n"
STAT2 = "cpu  1600 0 700 8400 500 0 0 0 0 0\ncpu0 1 1 1 1 1 0 0 0 0 0\nbtime 1700000000\n"
DISK1 = ("   8       0 sda 100 0 2048 0 50 0 4096 0 0 1000 0 0 0 0 0\n"
         "   8       1 sda1 100 0 2048 0 50 0 4096 0 0 1000 0 0 0 0 0\n"
         " 259       0 nvme0n1 10 0 1000000 0 10 0 2000000 0 0 2000 0 0 0 0 0\n"
         "   7       0 loop0 1 0 8 0 0 0 0 0 0 0 0 0 0 0 0\n")
DISK2 = ("   8       0 sda 100 0 2048 0 50 0 4096 0 0 1000 0 0 0 0 0\n"
         "   8       1 sda1 100 0 2048 0 50 0 4096 0 0 1000 0 0 0 0 0\n"
         " 259       0 nvme0n1 10 0 11000000 0 10 0 2000000 0 0 4500 0 0 0 0 0\n"
         "   7       0 loop0 1 0 8 0 0 0 0 0 0 0 0 0 0 0 0\n")
NET_HEAD = ("Inter-|   Receive                                                |  Transmit\n"
            " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets "
            "errs drop fifo colls carrier compressed\n")
NET1 = NET_HEAD + ("    lo: 500 1 0 0 0 0 0 0 500 1 0 0 0 0 0 0\n"
                   "  eth0: 1000 1 0 0 0 0 0 0 2000 1 0 0 0 0 0 0\n"
                   "veth12: 9 1 0 0 0 0 0 0 9 1 0 0 0 0 0 0\n")
NET2 = NET_HEAD + ("    lo: 900 1 0 0 0 0 0 0 900 1 0 0 0 0 0 0\n"
                   "  eth0: 6000 1 0 0 0 0 0 0 4500 1 0 0 0 0 0 0\n"
                   "veth12: 99 1 0 0 0 0 0 0 99 1 0 0 0 0 0 0\n")
# 8100 = 0x1FA4, LISTEN = 0A; a second line for an established connection on the same port
TCP = ("  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  "
       "timeout inode\n"
       "   0: 00000000:1FA4 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000 "
       "       0 5555 1 0000000000000000 100 0 0 10 0\n"
       "   1: 0100007F:1FA4 0100007F:D431 01 00000000:00000000 00:00000000 00000000  1000 "
       "       0 7777 1 0000000000000000 20 4 30 10 -1\n")


def _proc_tree(d: str, stat_text: str = STAT1, disk: str = DISK1, net: str = NET1) -> None:
    os.makedirs(os.path.join(d, "net"), exist_ok=True)
    files = {"meminfo": MEMINFO, "stat": stat_text, "diskstats": disk, "net/dev": net,
             "net/tcp": TCP, "net/tcp6": TCP.splitlines()[0] + "\n"}
    for name, text in files.items():
        with open(os.path.join(d, name), "w") as f:
            f.write(text)
    tick = os.sysconf("SC_CLK_TCK")
    # 4242: the vLLM API server, listening; 4243 its engine core; 4244 a grandchild; 99 other
    for pid, ppid, comm, rss_kb, threads, cmd, fds in (
            (4242, 1, "vllm", 2_000_000, 12,
             ["/home/u/venv/bin/python", "-m", "vllm.entrypoints.openai.api_server",
              "--api-key", "sk-secret-123", "--model", "/home/u/models/qwen"], {"3": 5555}),
            (4243, 4242, "VLLM::EngineCore", 6_000_000, 40, ["VLLM::EngineCore"], {}),
            (4244, 4243, "python3", 100_000, 2, ["python3", "-c", "x"], {}),
            (99, 1, "sshd", 5_000, 1, ["sshd"], {"4": 7777})):
        pd = os.path.join(d, str(pid))
        os.makedirs(os.path.join(pd, "fd"), exist_ok=True)
        utime, stime, start = 300 * tick // 100, 100 * tick // 100, 50 * tick
        fields = ["S", str(ppid)] + ["0"] * 9 + [str(utime), str(stime)] + ["0"] * 6 + \
            [str(start)] + ["0"] * 20
        with open(os.path.join(pd, "stat"), "w") as f:
            f.write(f"{pid} ({comm}) " + " ".join(fields) + "\n")
        with open(os.path.join(pd, "status"), "w") as f:
            f.write(f"Name:\t{comm}\nVmRSS:\t{rss_kb} kB\nThreads:\t{threads}\n")
        with open(os.path.join(pd, "cmdline"), "w") as f:
            f.write("\0".join(cmd) + "\0")
        for fd, inode in fds.items():
            os.symlink(f"socket:[{inode}]", os.path.join(pd, "fd", fd))
        os.symlink("/dev/null", os.path.join(pd, "fd", "0"))


class FakeNVML:
    """pynvml's surface as the collector calls it. `gb10`: memory info not supported."""

    NVML_TEMPERATURE_GPU, NVML_CLOCK_SM, NVML_CLOCK_MEM = 0, 1, 2

    class NotSupported(Exception):
        pass

    def __init__(self, gb10: bool, procs=None, graphics=None):
        self.gb10 = gb10
        self.procs = procs or []
        self.graphics = graphics or []
        self.inits = 0
        self.calls = 0

    def nvmlInit(self):
        self.inits += 1

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, i):
        return ("h", i)

    def nvmlDeviceGetName(self, h):
        return "NVIDIA GB10" if self.gb10 else b"NVIDIA RTX PRO 6000"

    def nvmlDeviceGetUUID(self, h):
        return "GPU-0000"

    def nvmlDeviceGetUtilizationRates(self, h):
        self.calls += 1
        return type("U", (), {"gpu": 87, "memory": 10})()

    def nvmlDeviceGetMemoryInfo(self, h):
        if self.gb10:
            raise self.NotSupported("Not Supported")
        return type("M", (), {"used": 40 * 2 ** 30, "total": 96 * 2 ** 30, "free": 56 * 2 ** 30})()

    def nvmlDeviceGetPowerUsage(self, h):
        return 37_799

    def nvmlDeviceGetEnforcedPowerLimit(self, h):
        if self.gb10:
            raise self.NotSupported("Not Supported")
        return 600_000

    def nvmlDeviceGetTemperature(self, h, which):
        return 57

    def nvmlDeviceGetClockInfo(self, h, which):
        if which == self.NVML_CLOCK_MEM and self.gb10:
            raise self.NotSupported("Not Supported")
        return 2398 if which == self.NVML_CLOCK_SM else 9001

    def _p(self, pid, mem):
        return type("P", (), {"pid": pid, "usedGpuMemory": mem})()

    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return [self._p(*x) for x in self.procs]

    def nvmlDeviceGetGraphicsRunningProcesses(self, h):
        return [self._p(*x) for x in self.graphics]


def _collector(d, clock, **kw):
    kw.setdefault("psutil_mod", None)
    kw.setdefault("nvml_mod", None)
    kw.setdefault("smi", os.path.join(d, "no-such-nvidia-smi"))
    kw.setdefault("lsof", None)
    kw.setdefault("ss", None)
    return hc.HostCollector(proc=os.path.join(d, "proc"), sys_root=os.path.join(d, "sys"),
                            clock=clock, home="/home/u", label="box", **kw)


def _thermal(d, temps):
    for i, t in enumerate(temps):
        z = os.path.join(d, "sys", "class", "thermal", f"thermal_zone{i}")
        os.makedirs(z, exist_ok=True)
        with open(os.path.join(z, "temp"), "w") as f:
            f.write(f"{t}\n")


def test_proc_fallback_memory_cpu_disks_net_and_the_process_tree():
    d = tempfile.mkdtemp(prefix="hc-")
    try:
        _proc_tree(os.path.join(d, "proc"))
        _thermal(d, [41000, 63500, 200000])               # 200 °C is a broken sensor, left out
        clk = Clock()
        nv = FakeNVML(gb10=True, procs=[(4243, 70 * 2 ** 30)], graphics=[(99, 19_324_928)])
        c = _collector(d, clk, nvml_mod=nv, scan_period=10)
        watch = [{"backend": "vllm1", "port": 8100, "label": "vLLM"}]
        s1 = c.sample(watch)
        assert s1["cpu"]["utilization"] is None, "no rate before a second sample"
        # the second sample: 10 s later, new counters
        _proc_tree_update(os.path.join(d, "proc"))
        clk.t += 10
        s2 = c.sample(watch)
        m = s2["memory"]
        assert m["total_bytes"] == 128452068 * 1024 and m["available_bytes"] == 40112640 * 1024
        assert m["swap_used_bytes"] == (16777212 - 16000000) * 1024
        assert m["unified"] is True, "GB10: one pool"
        # 1,200 jiffies passed, 400 of them idle (iowait +0)
        assert s2["cpu"]["utilization"] == 0.6667, s2["cpu"]
        assert s2["cpu"]["temperature_c"] == 63.5
        disks = {x["name"]: x for x in s2["disks"]}
        assert set(disks) == {"sda", "nvme0n1"}, "partitions and loops left out"
        assert disks["nvme0n1"]["nvme"] and disks["nvme0n1"]["read_bps"] == 10_000_000 * 512 / 10
        assert disks["nvme0n1"]["busy"] == 0.25 and disks["sda"]["read_bps"] == 0.0
        net = {x["name"]: x for x in s2["net"]}
        assert set(net) == {"eth0"} and net["eth0"]["rx_bps"] == 500.0
        g = s2["gpus"][0]
        assert s2["gpu_source"] == "nvml" and g["memory_unified"] and g["memory_used_bytes"] is None
        assert g["utilization"] == 0.87 and g["power_w"] == 37.8 and g["power_limit_w"] is None
        assert g["sm_clock_mhz"] == 2398 and g["mem_clock_mhz"] is None
        p = s2["processes"][0]
        assert p["found"] and p["pid"] == 4242 and p["via"] == "proc", p
        assert p["name"] == "vllm" and p["rss_bytes"] == 2_000_000 * 1024 and p["threads"] == 12
        assert [k["pid"] for k in p["children"]] == [4243, 4244]
        assert p["total_rss_bytes"] == (2_000_000 + 6_000_000 + 100_000) * 1024
        assert p["gpu_memory"] == "measured" and p["total_gpu_memory_bytes"] == 70 * 2 ** 30
        assert p["children"][0]["gpu_memory_bytes"] == 70 * 2 ** 30
        assert "unified pool" in p["note"]
        assert p["cmdline"].startswith("~/venv/bin/python -m vllm"), p["cmdline"]
        assert "sk-secret" not in json.dumps(s2) and "--api-key ***" in p["cmdline"]
        assert "~/models/qwen" in p["cmdline"]
        assert p["cpu"] == 0.5, "utime+stime rose by 5 s in 10 s"
        assert p["open_files"] == 2 and p["uptime_s"] is not None
        _check([dict(s2, id="local", label="box", source="local", url=None,
                     backends=["vllm1"], state="ok", error=None)])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _proc_tree_update(proc):
    for name, text in (("stat", STAT2), ("diskstats", DISK2), ("net/dev", NET2)):
        with open(os.path.join(proc, name), "w") as f:
            f.write(text)
    tick = os.sysconf("SC_CLK_TCK")
    path = os.path.join(proc, "4242", "stat")
    with open(path) as f:
        st = f.read()
    head, rest = st[:st.rindex(")") + 2], st[st.rindex(")") + 2:].split()
    rest[11] = str(600 * tick // 100)
    rest[12] = str(300 * tick // 100)
    with open(path, "w") as f:
        f.write(head + " ".join(rest) + "\n")


def test_cadence_fast_part_every_5s_scan_every_15s():
    d = tempfile.mkdtemp(prefix="hc-")
    try:
        _proc_tree(os.path.join(d, "proc"))
        clk = Clock()
        nv = FakeNVML(gb10=False)
        c = _collector(d, clk, nvml_mod=nv)
        w = [{"backend": "v", "port": 8100}]
        c.sample(w)
        for _ in range(4):
            clk.t += 1
            c.sample(w)
        assert c.cost["samples"] == 1 and c.cost["scans"] == 1 and nv.calls == 1
        clk.t += 1                                           # 5 s
        c.sample(w)
        assert c.cost["samples"] == 2 and c.cost["scans"] == 1
        clk.t += 10                                          # 15 s
        c.sample(w)
        assert c.cost["scans"] == 2
        c.sample(w + [{"backend": "x", "port": 9}])          # a new watch list scans at once
        assert c.cost["scans"] == 3
        assert nv.inits == 1, "NVML initialised once"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_discrete_gpu_per_process_memory_and_gpu_block():
    d = tempfile.mkdtemp(prefix="hc-")
    try:
        _proc_tree(os.path.join(d, "proc"))
        nv = FakeNVML(gb10=False, procs=[(4242, 1 * 2 ** 30), (4243, 30 * 2 ** 30)],
                      graphics=[(4243, 2 ** 20)])
        c = _collector(d, Clock(), nvml_mod=nv)
        s = c.sample([{"backend": "v", "port": 8100}])
        g = s["gpus"][0]
        assert g["name"] == "NVIDIA RTX PRO 6000" and not g["memory_unified"]
        assert g["memory_total_bytes"] == 96 * 2 ** 30 and g["power_limit_w"] == 600.0
        assert s["memory"]["unified"] is False
        p = s["processes"][0]
        assert p["gpu_memory"] == "measured" and p["gpu_memory_bytes"] == 2 ** 30
        assert p["total_gpu_memory_bytes"] == 31 * 2 ** 30 + 2 ** 20, "compute + graphics, tree"
        assert p["note"] is None
        b = c.gpu_block()
        assert b["source"] == "nvml" and b["utilization"] == 0.87 and b["temperature_c"] == 57
        errs = contract_check.validate(b, contract_check.load("system")["properties"]["gpu"])
        assert not errs, errs
        # a process the driver lists without a figure, on a discrete GPU: not reported
        nv.procs = [(4242, None)]
        nv.graphics = []
        s = c.sample([{"backend": "v", "port": 8100}], force=True)
        assert s["processes"][0]["gpu_memory"] == "not_reported"
        nv.procs = []
        s = c.sample([{"backend": "v", "port": 8100}], force=True)
        assert s["processes"][0]["gpu_memory"] == "none"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_nvidia_smi_fallback_and_a_missing_smi_is_remembered():
    d = tempfile.mkdtemp(prefix="hc-")
    try:
        _proc_tree(os.path.join(d, "proc"))
        smi = os.path.join(d, "nvidia-smi")
        log = os.path.join(d, "calls")
        with open(smi, "w") as f:
            f.write(f"""#!/bin/sh
echo "$@" >> {log}
case "$1" in
  --query-gpu=*) echo "0, NVIDIA GB10, 19, [N/A], [N/A], 56.02, [N/A], 61, 2470, [N/A]";;
  --query-compute-apps=*) echo "4242, 73352"; echo "4244, [N/A]";;
esac
""")
        os.chmod(smi, os.stat(smi).st_mode | stat.S_IEXEC)
        c = _collector(d, Clock(), smi=smi)
        s = c.sample([{"backend": "v", "port": 8100}])
        g = s["gpus"][0]
        assert s["gpu_source"] == "nvidia-smi" and g["source"] == "nvidia-smi"
        assert g["memory_unified"] and g["memory_used_bytes"] is None and g["power_w"] == 56.02
        assert g["utilization"] == 0.19 and g["temperature_c"] == 61.0 and g["mem_clock_mhz"] is None
        p = s["processes"][0]
        assert p["gpu_memory_bytes"] == 73352 * 2 ** 20 and p["gpu_memory"] == "measured"
        assert len(open(log).read().splitlines()) == 2, "one query for the GPU, one for apps"
        # without nvidia-smi: no GPU, and the binary is not looked for again
        c2 = _collector(d, Clock(), smi=os.path.join(d, "absent"))
        c2.sample([], force=True)
        c2.sample([], force=True)
        assert c2._smi_ok is False and c2.sample([], force=True)["gpus"] == []
        assert c2.gpu_block() is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_port_to_pid_on_a_real_socket():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        c = hc.HostCollector(nvml_mod=None, smi="/nonexistent/nvidia-smi")
        if c.psutil is None and not os.path.isdir("/proc/1") and not shutil.which("lsof"):
            print("    (no psutil, /proc or lsof here: skipped)")
            return
        got = c.sample([{"backend": "me", "port": port}])["processes"][0]
        assert got["pid"] == os.getpid() and got["found"], got
        assert got["via"] in ("psutil", "proc", "lsof", "ss")
        if c.psutil is not None or os.path.isdir("/proc/1"):
            assert got["rss_bytes"] and got["threads"], got
        none = c.sample([{"backend": "x", "port": _free_port()}], force=True)["processes"][0]
        assert none["pid"] is None and not none["found"] and none["note"]
    finally:
        s.close()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_redaction_of_command_lines():
    r = hc.redact_cmdline(["/home/u/v/bin/python", "server.py", "--api-key", "sk-1",
                           "--token=abc", "--hf-token", "hf_x", "HF_TOKEN=hf_y", "--port",
                           "8000", "--password", "p"], home="/home/u")
    assert r == ("~/v/bin/python server.py --api-key *** --token=*** --hf-token *** "
                 "HF_TOKEN=*** --port 8000 --password ***"), r
    assert len(hc.redact_cmdline(["x" * 1000])) == hc.CMDLINE_MAX


def test_placement():
    loc = hc._Locality(resolve=lambda h: {"192.0.2.77"})
    b = lambda url, **kw: dict({"id": "b", "url": url, "cloud": False, "enabled": True,  # noqa
                                "host": {"mode": "auto"}}, **kw)
    assert hc.placement(b("http://127.0.0.1:8100/v1"), loc)[0] == "local"
    assert hc.placement(b("http://localhost:8100/v1"), loc)[0] == "local"
    assert hc.placement(b("http://[::1]:8100/v1"), loc)[0] == "local"
    assert hc.placement(b("http://192.0.2.1:8100/v1"), loc) is None, "TEST-NET is never local"
    assert hc.placement(b("https://api.deepseek.com/v1", cloud=True), loc) is None
    assert hc.placement(b("http://127.0.0.1:1/v1", enabled=False), loc) is None
    assert hc.placement(b("http://192.0.2.1:1/v1", host={"mode": "same"}), loc)[0] == "local"
    assert hc.placement(b("http://127.0.0.1:1/v1", host={"mode": "off"}), loc) is None
    assert hc.placement(b("http://x:1/v1", host={"mode": "collector", "url": "http://x:9108"}),
                        loc)[0] == "collector"
    assert hc.placement(b("http://x:1/v1", host={"mode": "collector"}), loc) is None
    assert hc.placement(b("http://x:1/v1", host={"mode": "exporter",
                                                   "dcgm_url": "http://x:9400/metrics"}),
                        loc)[0] == "exporter"
    # a name resolves on a thread; until it answers the backend is not placed
    assert hc.placement(b("http://gpu-box.lan:8100/v1"), loc) is None
    for _ in range(50):
        if "gpu-box.lan" in loc._cache and not loc._cache["gpu-box.lan"][2]:
            break
        time.sleep(0.01)
    assert hc.placement(b("http://gpu-box.lan:8100/v1"), loc) is None, "192.0.2.77 is not here"
    loc2 = hc._Locality(resolve=lambda h: {"127.0.0.1"})
    loc2.is_local("http://mybox:1")
    for _ in range(50):
        if "mybox" in loc2._cache and not loc2._cache["mybox"][2]:
            break
        time.sleep(0.01)
    assert loc2.is_local("http://mybox:1")


NODE1 = """# HELP node_memory_MemTotal_bytes x
# TYPE node_memory_MemTotal_bytes gauge
node_memory_MemTotal_bytes 1.37e+11
node_memory_MemAvailable_bytes 4.0e+10
node_memory_SwapTotal_bytes 0
node_memory_SwapFree_bytes 0
node_cpu_seconds_total{cpu="0",mode="idle"} 900
node_cpu_seconds_total{cpu="0",mode="user"} 100
node_cpu_seconds_total{cpu="1",mode="idle"} 900
node_cpu_seconds_total{cpu="1",mode="user"} 100
node_load1 2.5
node_load5 2
node_load15 1.5
node_disk_read_bytes_total{device="nvme0n1"} 1e9
node_disk_written_bytes_total{device="nvme0n1"} 0
node_disk_io_time_seconds_total{device="nvme0n1"} 10
node_disk_read_bytes_total{device="nvme0n1p1"} 1e9
node_network_receive_bytes_total{device="enP7s7"} 1000
node_network_transmit_bytes_total{device="enP7s7"} 1000
node_network_receive_bytes_total{device="lo"} 5
node_hwmon_temp_celsius{chip="x",sensor="temp1"} 55.5
node_thermal_zone_temp{type="acpitz",zone="0"} NaN
weird_metric{a="say \\"hi\\" {there}",b="x"} 3 1700000000000
"""
NODE2 = NODE1.replace('mode="idle"} 900', 'mode="idle"} 950').replace(
    'mode="user"} 100', 'mode="user"} 150').replace(
    'node_disk_read_bytes_total{device="nvme0n1"} 1e9',
    'node_disk_read_bytes_total{device="nvme0n1"} 3e9').replace(
    'node_disk_io_time_seconds_total{device="nvme0n1"} 10',
    'node_disk_io_time_seconds_total{device="nvme0n1"} 15').replace(
    'node_network_receive_bytes_total{device="enP7s7"} 1000',
    'node_network_receive_bytes_total{device="enP7s7"} 11000')
DCGM_PAGE = """# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (in %).
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-a",device="nvidia0",modelName="NVIDIA H100 80GB HBM3",Hostname="h"} 42
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-a",modelName="NVIDIA H100 80GB HBM3"} 40960
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-a",modelName="NVIDIA H100 80GB HBM3"} 40960
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-a",modelName="NVIDIA H100 80GB HBM3"} 312.5
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a",modelName="NVIDIA H100 80GB HBM3"} 61
DCGM_FI_DEV_SM_CLOCK{gpu="0",UUID="GPU-a",modelName="NVIDIA H100 80GB HBM3"} 1980
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-b",modelName="NVIDIA H100 80GB HBM3"} 0
"""


def test_scrape_parser_node_and_dcgm():
    ss = scrape.parse(NODE1)
    w = [x for x in ss if x[0] == "weird_metric"][0]
    assert w[1] == {"a": 'say "hi" {there}', "b": "x"} and w[2] == 3.0
    c1, c2 = scrape.node_counters(ss), scrape.node_counters(scrape.parse(NODE2))
    h = scrape.node_host(c2, c1, 10.0)
    assert h["memory"]["total_bytes"] == 137_000_000_000
    assert h["memory"]["used_bytes"] == 97_000_000_000
    assert h["cpu"]["count"] == 2 and h["cpu"]["utilization"] == 0.5
    assert h["cpu"]["load"] == [2.5, 2.0, 1.5] and h["cpu"]["temperature_c"] == 55.5
    assert [d["name"] for d in h["disks"]] == ["nvme0n1"]
    assert h["disks"][0]["read_bps"] == 2e8 and h["disks"][0]["busy"] == 0.5
    assert h["net"] == [{"name": "enP7s7", "rx_bps": 1000.0, "tx_bps": 0.0}]
    gp = scrape.dcgm_gpus(scrape.parse(DCGM_PAGE))
    assert len(gp) == 2 and gp[0]["memory_used_bytes"] == 40960 * 2 ** 20
    assert gp[0]["memory_total_bytes"] == 81920 * 2 ** 20 and gp[0]["utilization"] == 0.42
    assert gp[0]["power_w"] == 312.5 and gp[1]["memory_used_bytes"] is None
    assert not gp[0]["memory_unified"]


def test_remote_collector_and_exporter_through_hosts():
    d = tempfile.mkdtemp(prefix="hc-")
    try:
        _proc_tree(os.path.join(d, "proc"))
        col = _collector(d, time.time, nvml_mod=FakeNVML(gb10=True, procs=[(4243, 2 ** 30)]))
        try:
            hc.make_server("0.0.0.0", 0, None)
            raise AssertionError("bound beyond loopback without a token")
        except SystemExit as exc:
            assert "without a token" in str(exc)
        srv = hc.make_server("127.0.0.1", 0, "tok-123456", [], col)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                         daemon=True).start()
        base = f"http://127.0.0.1:{port}"
        try:
            urllib.request.urlopen(base + "/host", timeout=5)
            raise AssertionError("no token, no answer")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        assert urllib.request.urlopen(base + "/health", timeout=5).status == 200
        text = hc._fetch(base + "/metrics?watch=v:8100", "tok-123456", 5).decode()
        names = {x[0] for x in scrape.parse(text)}
        assert {"tandem_host_memory_available_bytes", "tandem_host_gpu_utilization_ratio",
                "tandem_host_process_rss_bytes"} <= names, names
        # the exporter pages, served by a tiny server of their own
        pages = {"/node": NODE1, "/dcgm": DCGM_PAGE}
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class P(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = pages[self.path].encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        psrv = ThreadingHTTPServer(("127.0.0.1", 0), P)
        threading.Thread(target=psrv.serve_forever, daemon=True).start()
        pbase = f"http://127.0.0.1:{psrv.server_address[1]}"
        hosts = hc.Hosts(_collector(d, time.time), env={hc.TOKEN_ENV: "tok-123456"})
        backends = [
            {"id": "vllm1", "label": "vLLM", "url": "http://192.0.2.10:8100/v1", "cloud": False,
             "enabled": True, "host": {"mode": "collector", "url": base}},
            {"id": "sg", "url": "http://192.0.2.11:30000/v1", "cloud": False, "enabled": True,
             "host": {"mode": "exporter", "url": pbase + "/node", "dcgm_url": pbase + "/dcgm"}},
            {"id": "ds", "url": "https://api.deepseek.com/v1", "cloud": True, "enabled": True,
             "host": {"mode": "auto"}}]
        v = hosts.view(backends, hc.engine_entry(8080, "proxy", backend="proxy"), wait=5)
        _check(v)
        assert [h["source"] for h in v] == ["local", "collector", "exporter"]
        assert v[0]["backends"] == ["proxy"], "the cloud backend has no host"
        r = v[1]
        assert r["state"] == "ok" and r["backends"] == ["vllm1"] and r["label"] == "box"
        p = r["processes"][0]
        assert p["backend"] == "vllm1" and p["label"] == "vLLM" and p["pid"] == 4242
        assert p["gpu_memory"] == "measured" and r["memory"]["unified"] is True
        e = v[2]
        assert e["processes"] is None and e["gpus"][0]["name"] == "NVIDIA H100 80GB HBM3"
        assert e["memory"]["total_bytes"] == 137_000_000_000 and e["memory"]["unified"] is False
        assert e["cpu"]["utilization"] is None, "a rate needs a second page"
        # a wrong token: the host is an error with the last answer kept
        hosts.env = {hc.TOKEN_ENV: "wrong"}
        hosts.remotes.clear()
        v = hosts.view(backends[:1], None, wait=5)
        assert v[1]["state"] == "error" and v[1]["error"] == "HTTP 401", v[1]
        _check(v)
        # an unreachable collector never holds the view
        dead = dict(backends[0], host={"mode": "collector", "url": "http://192.0.2.1:9"})
        t0 = time.time()
        v = hosts.view([dead], None)
        assert time.time() - t0 < 0.5 and v[1]["state"] == "pending"
        srv.shutdown()
        psrv.shutdown()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_config_host_field_env_lock_and_migration():
    d = tempfile.mkdtemp(prefix="hc-cfg-")
    try:
        path = os.path.join(d, "config.sqlite3")
        # a v1 database from before the column: the column is added in place
        con = sqlite3.connect(path)
        con.executescript(config_mod.DDL.replace(",\n  host TEXT NOT NULL DEFAULT '{}'", ""))
        con.execute("INSERT INTO meta VALUES ('schema_version', '1')")
        con.execute("INSERT INTO backends(id, kind, url, created_at, updated_at) VALUES "
                    "('old', 'generic', 'http://127.0.0.1:1/v1', 1, 1)")
        con.commit()
        assert "host" not in {r[1] for r in con.execute("PRAGMA table_info(backends)")}
        con.close()
        env = {"TANDEM_BACKENDS": "gpu1", "TANDEM_BACKEND_GPU1_KIND": "generic",
               "TANDEM_BACKEND_GPU1_URL": "http://192.0.2.10:8100",
               "TANDEM_BACKEND_GPU1_HOST": "collector",
               "TANDEM_BACKEND_GPU1_HOST_URL": "http://192.0.2.10:9108",
               "TANDEM_SECRETS_KEY": "A" * 43 + "="}
        cfg = config_mod.ConfigDB(path, env=env, use_keychain=False).open()
        cfg.apply_env()
        assert cfg.backend("old")["host"] == {"mode": "auto", "url": None, "dcgm_url": None}
        b = cfg.backend("gpu1")
        assert b["host"]["mode"] == "collector" and b["host"]["url"] == "http://192.0.2.10:9108"
        assert cfg.backend_locks("gpu1")["host.mode"] == "TANDEM_BACKEND_GPU1_HOST"
        try:
            cfg.update_backend("gpu1", {"host": {"mode": "off"}})
            raise AssertionError("a locked field changed")
        except config_mod.ConfigError as exc:
            assert exc.var == "TANDEM_BACKEND_GPU1_HOST", exc.message
        cfg.update_backend("gpu1", {"host": {"dcgm_url": "http://192.0.2.10:9400/metrics"}})
        b = cfg.backend("gpu1")
        assert b["host"]["dcgm_url"].endswith(":9400/metrics") and b["host"]["mode"] == \
            "collector", "a patch merges into the object"
        cfg.update_backend("old", {"host": {"mode": "same"}})
        assert cfg.backend("old")["host"]["mode"] == "same"
        try:
            cfg.update_backend("old", {"host": {"mode": "nearby"}})
            raise AssertionError("bad mode accepted")
        except config_mod.ConfigError as exc:
            assert exc.field == "host.mode"
        cfg.close()
        bad = dict(env, TANDEM_BACKEND_GPU1_HOST="nearby")
        try:
            config_mod.parse_env(bad)
            raise AssertionError("bad env mode accepted")
        except config_mod.ConfigError as exc:
            assert exc.var == "TANDEM_BACKEND_GPU1_HOST", exc.message
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_standalone_proxy_system_has_the_hosts():
    from proxy_util import Proxy, fake
    fsrv, fport = fake(models=["fake-a"])
    try:
        env = {"TANDEM_BACKENDS": "f1", "TANDEM_BACKEND_F1_KIND": "generic",
               "TANDEM_BACKEND_F1_URL": f"http://127.0.0.1:{fport}/v1"}
        with Proxy(env=env) as px:
            st, _, body = px.get("/v1/dashboard/system")
            assert st == 200
            doc = json.loads(body)
            errs = contract_check.validate(doc, contract_check.load("system"))
            assert not errs, errs[:5]
            local = doc["hosts"][0]
            assert local["source"] == "local" and local["backends"] == ["proxy", "f1"]
            f1 = [p for p in local["processes"] if p["backend"] == "f1"][0]
            c = local["cost"]
            assert c["fast_period_s"] == 5.0 and c["scan_period_s"] == 15.0
            if f1["via"] is not None:              # psutil, /proc or lsof answered here
                assert f1["pid"] == os.getpid(), f1
            assert doc["memory"]["unified_total_bytes"], "from the collector where no /proc"
    finally:
        fsrv.shutdown()


def test_collector_cost_on_this_machine():
    """The real collector here, forced every call: what one fast sample and one scan cost."""
    c = hc.HostCollector(smi="/nonexistent/nvidia-smi")
    w = [{"backend": "engine", "pid": os.getpid(), "port": None}]
    c.sample(w, force=True)
    t0 = time.process_time()
    n = 20
    for _ in range(n):
        c.sample(w, force=True)
    per = (time.process_time() - t0) / n * 1000
    # at the real cadence (one sample in 5 s, one scan in 15 s) the share of one CPU:
    share = per / 5000
    print(f"    one sample with a scan: {per:.2f} ms CPU here, <= {share * 100:.3f} % of a CPU "
          f"at the 5 s cadence")
    assert share < 0.005, "under 0.5 % of one CPU"


if __name__ == "__main__":
    from proxy_util import run
    run(globals())
