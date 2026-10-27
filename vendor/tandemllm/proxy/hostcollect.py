"""The host collector: the machine behind a backend, for the System view (P1.5).

    hosts = Hosts()                                   one per process
    hosts.view(backends, engine=...)                  -> system.hosts[] (contract 1.3)

    python -m proxy.hostcollect --port 9108           the remote collector, on a backend's host
            [--host 0.0.0.0 --token-file F] [--watch vllm1:8100,sglang:30000]

What one host reports, each field null when its source cannot answer on that tick:

    memory     total, available, used, swap; `unified` on a GB10 (one pool for CPU and GPU:
               MemAvailable is the truth there, the GPU has no memory figure of its own)
    cpu        count, utilization over the last interval, load 1/5/15, the hottest thermal zone
    gpus       per device: use, memory used and total (or `memory_unified`), power and its limit,
               temperature, SM and memory clocks, from NVML (`nvidia-ml-py`) or `nvidia-smi`
    disks      whole block devices: read and write bytes a second, busy share (NVMe marked)
    fs         free and total bytes of the paths the caller names (the state directory)
    net        bytes a second in and out per interface (loopback and container bridges left out)
    processes  per backend on this host: the process listening on the backend's port and its
               children (vLLM's engine core, SGLang's scheduler), each with RSS, threads, open
               files, uptime, CPU in cores, GPU memory (NVML or `nvidia-smi --query-compute-apps`)
               and the command line with the home written as `~` and secret values masked

Sources, first that answers: `psutil` and NVML when installed (`pip install tandemllm[host]`),
else `/proc` and `nvidia-smi`, else (macOS without psutil) `os` and `lsof`. A source that fails
once is not asked again on every tick: a missing `nvidia-smi` is remembered.

Cost: nothing runs between requests. A view samples on demand, at most every `fast_period`
(5 s) for memory, CPU, GPU, disks and network, and at most every `scan_period` (15 s) for the
process scan, whose cost is the port-to-pid match and the process tree. Rates are the difference
to the previous sample. NVML calls are microseconds; `nvidia-smi` is a process start, which is
why it is the fallback. `cost` in each host says what the last sample took.

Remote hosts: a backend whose `host.mode` is `collector` is read from this module running on its
host (`GET /host?watch=id:port,...`, a bearer token from TANDEM_COLLECTOR_TOKEN); `exporter`
reads a node_exporter page (`host.url`) and a DCGM exporter page (`host.dcgm_url`) with
proxy/scrape.py, which gives no per-process figures. Remote reads run on a thread of their own,
so the System view never waits for a slow host: it shows the last answer and its age.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

if __package__ in (None, ""):                                    # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy import scrape  # noqa: E402

_AUTO = object()
TOKEN_ENV = "TANDEM_COLLECTOR_TOKEN"
DEFAULT_PORT = 9108
SECRET_FLAG = re.compile(r"(key|token|secret|passw|auth|credential)", re.I)
CMDLINE_MAX = 400


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _opt(name: str):
    try:
        return __import__(name)
    except Exception:                                                # noqa: BLE001
        return None


def redact_cmdline(args: list[str], home: str | None = None) -> str:
    """The command line as one string: the home as `~`, the value after a flag whose name says
    key, token, secret, password or auth masked, capped at 400 characters."""
    home = home if home is not None else os.path.expanduser("~")
    out = []
    mask_next = False
    for a in args:
        if mask_next:
            out.append("***")
            mask_next = False
            continue
        if a.startswith("-") and SECRET_FLAG.search(a.split("=", 1)[0]):
            if "=" in a:
                out.append(a.split("=", 1)[0] + "=***")
            else:
                out.append(a)
                mask_next = True
            continue
        if re.match(r"^[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*=", a):
            out.append(a.split("=", 1)[0] + "=***")
            continue
        if home and home != "/" and home in a:
            a = a.replace(home, "~")
        out.append(a)
    s = " ".join(out)
    return s if len(s) <= CMDLINE_MAX else s[:CMDLINE_MAX - 1] + "…"


# ====================================================================== the local collector

class HostCollector:
    """One machine, sampled on demand. Thread safe (one lock); every read is in a try block."""

    def __init__(self, *, proc: str = "/proc", sys_root: str = "/sys", psutil_mod=_AUTO,
                 nvml_mod=_AUTO, smi: str = "nvidia-smi", lsof: str = "lsof", ss: str = "ss",
                 clock=time.time, fast_period: float = 5.0, scan_period: float = 15.0,
                 proc_ttl: float = 600.0,
                 paths: list[str] | None = None, home: str | None = None,
                 label: str | None = None):
        self.proc, self.sys_root = proc, sys_root
        self.psutil = _opt("psutil") if psutil_mod is _AUTO else psutil_mod
        self._nvml_mod = _opt("pynvml") if nvml_mod is _AUTO else nvml_mod
        self.smi, self.lsof, self.ss = smi, lsof, ss
        self.clock = clock
        self.fast_period, self.scan_period = fast_period, scan_period
        self.proc_ttl = proc_ttl
        self.paths = list(paths or [])
        self.home = home
        self.label = label or socket.gethostname()
        self._lock = threading.Lock()
        self._nvml = None              # the module after a good nvmlInit
        self._nvml_tried = False
        self._smi_ok: bool | None = None
        self._fast: dict | None = None
        self._fast_at = 0.0
        self._scan: dict | None = None
        self._scan_at = 0.0
        self._scan_key: tuple | None = None
        self._prev: dict = {}          # counters of the previous fast sample
        self._prev_proc: dict = {}     # (pid, starttime) -> (t, cpu seconds), kept across views, TTL-pruned
        self._port_pid: dict = {}      # port -> pid, checked first on the next scan
        self.cost = {"samples": 0, "scans": 0, "cpu_ms_total": 0.0, "last_ms": None,
                     "last_scan_ms": None}
        self._tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

    # ------------------------------------------------------------------ public
    def sample(self, watch: list[dict] | None = None, *, force: bool = False) -> dict:
        """The host now. `watch`: [{"backend": id, "port": n} | {"backend": id, "pid": n}]."""
        watch = watch or []
        with self._lock:
            now = self.clock()
            c0 = time.process_time()
            if force or self._fast is None or now - self._fast_at >= self.fast_period:
                self._fast = self._read_fast(now)
                self._fast_at = now
                self.cost["samples"] += 1
            key = tuple(sorted((w.get("backend") or "", w.get("port") or 0, w.get("pid") or 0)
                               for w in watch))
            scanned = False
            if force or self._scan is None or key != self._scan_key or \
                    now - self._scan_at >= self.scan_period:
                s0 = time.process_time()
                self._scan = self._read_procs(watch, now)
                self._scan_at, self._scan_key = now, key
                self.cost["scans"] += 1
                self.cost["last_scan_ms"] = round((time.process_time() - s0) * 1000, 3)
                scanned = True
            spent = (time.process_time() - c0) * 1000
            if spent > 0 or scanned:
                self.cost["cpu_ms_total"] = round(self.cost["cpu_ms_total"] + spent, 3)
                self.cost["last_ms"] = round(spent, 3)
            out = dict(self._fast)
            out["processes"] = self._scan["processes"]
            out["sampled_at"] = _iso(self._fast_at)
            out["age_s"] = round(max(0.0, now - self._fast_at), 1)
            out["cost"] = dict(self.cost, fast_period_s=self.fast_period,
                               scan_period_s=self.scan_period)
            return out

    def gpu_block(self) -> dict | None:
        """The 1.0 `system.gpu` block from the newest sample's first GPU, or None without one;
        lets the server drop its own `nvidia-smi` call when this collector has a GPU."""
        f = self._fast
        if not f or not f.get("gpus"):
            return None
        g = f["gpus"][0]
        return {"name": g["name"], "temperature_c": g["temperature_c"], "power_w": g["power_w"],
                "sm_clock_mhz": g["sm_clock_mhz"], "utilization": g["utilization"],
                "source": f.get("gpu_source"), "sampled_at": _iso(self._fast_at)}

    # ------------------------------------------------------------------ fast part
    def _read_fast(self, now: float) -> dict:
        prev, cur = self._prev, {"t": now}
        out = {"memory": self._memory(), "cpu": self._cpu(cur, prev, now),
               "gpus": [], "gpu_source": None, "disks": self._disks(cur, prev, now),
               "fs": self._fs(), "net": self._net(cur, prev, now)}
        gpus, src = self._gpus()
        out["gpus"], out["gpu_source"] = gpus, src
        if gpus and all(g.get("memory_unified") for g in gpus):
            out["memory"]["unified"] = True
        elif gpus:
            out["memory"]["unified"] = False
        self._prev = cur
        return out

    def _read(self, rel: str) -> str | None:
        try:
            with open(os.path.join(self.proc, rel)) as f:
                return f.read()
        except OSError:
            return None

    def _memory(self) -> dict:
        m = {"total_bytes": None, "available_bytes": None, "used_bytes": None,
             "swap_total_bytes": None, "swap_used_bytes": None, "unified": None}
        ps = self.psutil
        try:
            if ps is not None:
                vm, sw = ps.virtual_memory(), ps.swap_memory()
                m.update(total_bytes=int(vm.total), available_bytes=int(vm.available),
                         swap_total_bytes=int(sw.total), swap_used_bytes=int(sw.used))
            else:
                text = self._read("meminfo")
                if text is not None:
                    kv = {}
                    for line in text.splitlines():
                        k, _, v = line.partition(":")
                        p = v.split()
                        if p and p[0].isdigit():
                            kv[k.strip()] = int(p[0]) * 1024
                    m.update(total_bytes=kv.get("MemTotal"),
                             available_bytes=kv.get("MemAvailable"),
                             swap_total_bytes=kv.get("SwapTotal"))
                    if kv.get("SwapTotal") is not None and kv.get("SwapFree") is not None:
                        m["swap_used_bytes"] = kv["SwapTotal"] - kv["SwapFree"]
                elif hasattr(os, "sysconf"):
                    m["total_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except Exception:                                            # noqa: BLE001
            pass
        if m["total_bytes"] is not None and m["available_bytes"] is not None:
            m["used_bytes"] = m["total_bytes"] - m["available_bytes"]
        return m

    def _cpu(self, cur: dict, prev: dict, now: float) -> dict:
        c = {"count": os.cpu_count(), "utilization": None, "load": [None, None, None],
             "temperature_c": None}
        try:
            c["load"] = [round(x, 2) for x in os.getloadavg()]
        except (OSError, AttributeError):
            pass
        busy_idle = None
        try:
            if self.psutil is not None:
                t = self.psutil.cpu_times()
                d = t._asdict()
                idle = d.get("idle", 0.0) + d.get("iowait", 0.0)
                total = sum(v for k, v in d.items() if k not in ("guest", "guest_nice"))
                busy_idle = (total, idle)
            else:
                text = self._read("stat")
                if text:
                    p = text.splitlines()[0].split()
                    vals = [int(x) for x in p[1:9]]
                    busy_idle = (float(sum(vals)), float(vals[3] + vals[4]))
        except Exception:                                            # noqa: BLE001
            busy_idle = None
        if busy_idle is not None:
            cur["cpu"] = busy_idle
            if prev.get("cpu"):
                dt = busy_idle[0] - prev["cpu"][0]
                di = busy_idle[1] - prev["cpu"][1]
                if dt > 0:
                    c["utilization"] = round(max(0.0, min(1.0, 1 - di / dt)), 4)
        c["temperature_c"] = self._thermal()
        return c

    def _thermal(self) -> float | None:
        base = os.path.join(self.sys_root, "class", "thermal")
        best = None
        try:
            names = os.listdir(base)
        except OSError:
            return None
        for n in names:
            if not n.startswith("thermal_zone"):
                continue
            try:
                with open(os.path.join(base, n, "temp")) as f:
                    v = int(f.read().strip()) / 1000.0
            except (OSError, ValueError):
                continue
            if -40 < v < 150 and (best is None or v > best):
                best = v
        return None if best is None else round(best, 1)

    def _disks(self, cur: dict, prev: dict, now: float) -> list[dict]:
        counters = {}       # dev -> (read bytes, written bytes, busy ms)
        try:
            text = self._read("diskstats")
            if text is not None:
                for line in text.splitlines():
                    p = line.split()
                    if len(p) < 14 or not scrape.DISK_RE.match(p[2]):
                        continue
                    counters[p[2]] = (int(p[5]) * 512, int(p[9]) * 512, int(p[12]))
            elif self.psutil is not None:
                for dev, io in (self.psutil.disk_io_counters(perdisk=True) or {}).items():
                    if scrape.DISK_RE.match(dev):
                        counters[dev] = (io.read_bytes, io.write_bytes,
                                         getattr(io, "busy_time", None))
        except Exception:                                            # noqa: BLE001
            counters = {}
        cur["disks"] = counters
        return [self._rate_row(dev, v, (prev.get("disks") or {}).get(dev), now - prev.get("t", 0),
                               disk=True) for dev, v in sorted(counters.items())]

    def _net(self, cur: dict, prev: dict, now: float) -> list[dict]:
        counters = {}       # dev -> (rx bytes, tx bytes)
        try:
            text = self._read("net/dev")
            if text is not None:
                for line in text.splitlines()[2:]:
                    name, _, rest = line.partition(":")
                    name, p = name.strip(), rest.split()
                    if len(p) >= 9 and not scrape.NET_SKIP.match(name):
                        counters[name] = (int(p[0]), int(p[8]))
            elif self.psutil is not None:
                for name, io in (self.psutil.net_io_counters(pernic=True) or {}).items():
                    if not scrape.NET_SKIP.match(name) and not name.startswith(("utun", "awdl",
                                                                                 "llw", "anpi",
                                                                                 "gif", "stf",
                                                                                 "bridge", "ap")):
                        counters[name] = (io.bytes_recv, io.bytes_sent)
        except Exception:                                            # noqa: BLE001
            counters = {}
        counters = {k: v for k, v in counters.items() if v[0] or v[1]}
        cur["net"] = counters
        dt = now - prev.get("t", 0)
        out = []
        for name, v in sorted(counters.items()):
            b = (prev.get("net") or {}).get(name)
            ok = b is not None and prev.get("t") and dt > 0
            out.append({"name": name,
                        "rx_bps": round((v[0] - b[0]) / dt, 1) if ok and v[0] >= b[0] else None,
                        "tx_bps": round((v[1] - b[1]) / dt, 1) if ok and v[1] >= b[1] else None})
        return out

    @staticmethod
    def _rate_row(dev, v, b, dt, disk=False) -> dict:
        ok = b is not None and dt > 0
        row = {"name": dev, "nvme": dev.startswith("nvme"),
               "read_bps": round((v[0] - b[0]) / dt, 1) if ok and v[0] >= b[0] else None,
               "write_bps": round((v[1] - b[1]) / dt, 1) if ok and v[1] >= b[1] else None,
               "busy": None}
        if ok and v[2] is not None and b[2] is not None and v[2] >= b[2]:
            row["busy"] = round(min(1.0, (v[2] - b[2]) / 1000.0 / dt), 4)
        return row

    def _fs(self) -> list[dict]:
        out, seen = [], set()
        for p in ["/"] + self.paths:
            q = os.path.expanduser(p)
            while q and not os.path.exists(q):
                q = os.path.dirname(q)
            try:
                st = os.stat(q or "/")
                if st.st_dev in seen:
                    continue
                seen.add(st.st_dev)
                du = shutil.disk_usage(q or "/")
            except OSError:
                continue
            home = self.home if self.home is not None else os.path.expanduser("~")
            shown = p if not home or home == "/" else p.replace(home, "~")
            out.append({"path": shown, "total_bytes": int(du.total), "free_bytes": int(du.free)})
        return out

    # ------------------------------------------------------------------ GPUs
    def _nvml_handle(self):
        if self._nvml is not None or self._nvml_tried:
            return self._nvml
        self._nvml_tried = True
        n = self._nvml_mod
        if n is None:
            return None
        try:
            n.nvmlInit()
            self._nvml = n
        except Exception:                                            # noqa: BLE001
            self._nvml = None
        return self._nvml

    def _gpus(self) -> tuple[list[dict], str | None]:
        n = self._nvml_handle()
        if n is not None:
            try:
                return self._gpus_nvml(n), "nvml"
            except Exception:                                        # noqa: BLE001
                pass
        rows = self._smi(["--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                          "power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem",
                          "--format=csv,noheader,nounits"])
        if rows is None:
            return [], None
        out = []
        for p in rows:
            if len(p) < 10:
                continue
            used, total = _smi_num(p[3]), _smi_num(p[4])
            unified = total is None or "GB10" in p[1]
            out.append({"index": int(_smi_num(p[0]) or 0), "name": p[1] or None, "uuid": None,
                        "utilization": _scale(_smi_num(p[2]), 0.01),
                        "memory_used_bytes": None if unified or used is None
                        else int(used * 1024 * 1024),
                        "memory_total_bytes": None if unified or total is None
                        else int(total * 1024 * 1024),
                        "memory_unified": unified,
                        "power_w": _smi_num(p[5]), "power_limit_w": _smi_num(p[6]),
                        "temperature_c": _smi_num(p[7]), "sm_clock_mhz": _smi_num(p[8]),
                        "mem_clock_mhz": _smi_num(p[9]), "source": "nvidia-smi"})
        return out, "nvidia-smi"

    def _gpus_nvml(self, n) -> list[dict]:
        out = []
        for i in range(n.nvmlDeviceGetCount()):
            h = n.nvmlDeviceGetHandleByIndex(i)

            def get(f, *a):
                try:
                    return f(h, *a)
                except Exception:                                    # noqa: BLE001
                    return None
            name = get(n.nvmlDeviceGetName)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            uuid = get(n.nvmlDeviceGetUUID)
            if isinstance(uuid, bytes):
                uuid = uuid.decode(errors="replace")
            util = get(n.nvmlDeviceGetUtilizationRates)
            mem = get(n.nvmlDeviceGetMemoryInfo)
            power = get(n.nvmlDeviceGetPowerUsage)
            limit = get(n.nvmlDeviceGetEnforcedPowerLimit)
            unified = mem is None or bool(name and "GB10" in name)
            out.append({
                "index": i, "name": name, "uuid": uuid,
                "utilization": None if util is None else round(util.gpu / 100.0, 4),
                "memory_used_bytes": None if unified else int(mem.used),
                "memory_total_bytes": None if unified else int(mem.total),
                "memory_unified": unified,
                "power_w": None if power is None else round(power / 1000.0, 2),
                "power_limit_w": None if limit is None else round(limit / 1000.0, 1),
                "temperature_c": get(n.nvmlDeviceGetTemperature, getattr(
                    n, "NVML_TEMPERATURE_GPU", 0)),
                "sm_clock_mhz": get(n.nvmlDeviceGetClockInfo, getattr(n, "NVML_CLOCK_SM", 1)),
                "mem_clock_mhz": get(n.nvmlDeviceGetClockInfo, getattr(n, "NVML_CLOCK_MEM", 2)),
                "source": "nvml"})
        return out

    def _smi(self, args: list[str]) -> list[list[str]] | None:
        if self._smi_ok is False:
            return None
        try:
            r = subprocess.run([self.smi] + args, capture_output=True, text=True, timeout=2.0)
        except FileNotFoundError:
            self._smi_ok = False
            return None
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode != 0:
            return None
        self._smi_ok = True
        return [[x.strip() for x in line.split(",")] for line in r.stdout.splitlines()
                if line.strip()]

    def _gpu_procs(self) -> dict | None:
        """pid -> GPU memory in bytes (None when the driver does not say), across devices; None
        when no source answers."""
        n = self._nvml_handle()
        if n is not None:
            try:
                got: dict = {}
                for i in range(n.nvmlDeviceGetCount()):
                    h = n.nvmlDeviceGetHandleByIndex(i)
                    for fn in ("nvmlDeviceGetComputeRunningProcesses",
                               "nvmlDeviceGetGraphicsRunningProcesses"):
                        try:
                            procs = getattr(n, fn)(h)
                        except Exception:                            # noqa: BLE001
                            continue
                        for p in procs:
                            v = getattr(p, "usedGpuMemory", None)
                            v = None if v is None or v < 0 or v >= 2 ** 63 else int(v)
                            old = got.get(p.pid)
                            got[p.pid] = v if old is None else old + (v or 0)
                return got
            except Exception:                                        # noqa: BLE001
                pass
        if not self._fast or self._fast.get("gpu_source") != "nvidia-smi":
            return None
        rows = self._smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        if rows is None:
            return None
        got = {}
        for p in rows:
            if len(p) >= 2 and p[0].isdigit():
                v = _smi_num(p[1])
                got[int(p[0])] = None if v is None else int(v * 1024 * 1024) + \
                    (got.get(int(p[0])) or 0)
        return got

    # ------------------------------------------------------------------ processes
    def _read_procs(self, watch: list[dict], now: float) -> dict:
        if not watch:
            return {"processes": []}
        gpu = self._gpu_procs()
        unified = bool(self._fast and self._fast["memory"].get("unified"))
        ppid_map = None
        out = []
        for w in watch:
            pid, via = w.get("pid"), "self" if w.get("pid") else None
            port = w.get("port")
            if pid is None and port:
                pid, via = self._pid_of_port(int(port))
            entry = {"backend": w.get("backend"), "label": w.get("label"), "port": port,
                     "pid": pid, "found": False, "via": via, "name": None, "cmdline": None,
                     "rss_bytes": None, "threads": None, "open_files": None, "uptime_s": None,
                     "cpu": None, "gpu_memory_bytes": None, "gpu_memory": "not_reported",
                     "children": [], "total_rss_bytes": None, "total_gpu_memory_bytes": None,
                     "note": None}
            if pid is None:
                entry["note"] = ("no process listens on this port, or it belongs to another user"
                                 if port else "no port to look up")
                out.append(entry)
                continue
            info = self._proc_info(pid, now)
            if info is None:
                entry["note"] = ("process figures need psutil on this system (pip install "
                                 "'tandemllm[host]')" if self.psutil is None and not
                                 os.path.isdir(self.proc) else
                                 "the process is gone or not readable")
                out.append(entry)
                continue
            entry.update(info)
            entry["found"] = True
            if ppid_map is None and self.psutil is None:
                ppid_map = self._ppid_map()
            kids = self._children(pid, ppid_map)
            for k in kids[:32]:
                ki = self._proc_info(k, now)
                if ki is None:
                    continue
                entry["children"].append({"pid": k, "name": ki["name"],
                                          "rss_bytes": ki["rss_bytes"], "threads": ki["threads"],
                                          "cpu": ki["cpu"],
                                          "gpu_memory_bytes": None if gpu is None
                                          else gpu.get(k)})
            entry["gpu_memory_bytes"] = None if gpu is None else gpu.get(pid)
            tree = [pid] + [c["pid"] for c in entry["children"]]
            if gpu is None:
                entry["gpu_memory"] = "not_reported"
            elif any(p in gpu for p in tree):
                vals = [gpu[p] for p in tree if p in gpu and gpu[p] is not None]
                if vals:
                    entry["gpu_memory"] = "measured"
                    entry["total_gpu_memory_bytes"] = int(sum(vals))
                else:
                    entry["gpu_memory"] = "unified" if unified else "not_reported"
            else:
                entry["gpu_memory"] = "none"
            rss = [x for x in [entry["rss_bytes"]] + [c["rss_bytes"] for c in entry["children"]]
                   if x is not None]
            entry["total_rss_bytes"] = int(sum(rss)) if rss else None
            if unified and entry["gpu_memory"] == "measured":
                entry["note"] = "GPU memory comes out of the same unified pool as the RSS"
            out.append(entry)
        # the CPU baselines are shared across views: a scan drops only the entries older than
        # the TTL. Pruning them to this view's pids instead (the old `live` set) made two views
        # with different watch keys — Processes vs GPU — wipe each other's cached children on
        # every alternating rescan, so each one's cpu% stayed None forever.
        self._prev_proc = {k: v for k, v in self._prev_proc.items()
                           if now - v[0] < self.proc_ttl}
        return {"processes": out}

    def _pid_of_port(self, port: int) -> tuple[int | None, str | None]:
        old = self._port_pid.get(port)
        if old is not None and self._proc_exists(old):
            inodes = self._listen_inodes(port)
            if inodes and self._pid_has_socket(old, inodes):
                return old, "proc"
        ps = self.psutil
        if ps is not None:
            try:
                for c in ps.net_connections(kind="inet"):
                    if c.status == "LISTEN" and c.laddr and c.laddr.port == port and c.pid:
                        self._port_pid[port] = c.pid
                        return c.pid, "psutil"
            except Exception:                                        # noqa: BLE001
                pass
        inodes = self._listen_inodes(port)
        if inodes:
            pid = self._scan_fds(inodes)
            if pid is not None:
                self._port_pid[port] = pid
                return pid, "proc"
        pid = self._lsof(port)
        if pid is not None:
            self._port_pid[port] = pid
            return pid, "lsof"
        pid = self._ss(port)
        if pid is not None:
            self._port_pid[port] = pid
            return pid, "ss"
        return None, None

    def _listen_inodes(self, port: int) -> set[str]:
        got = set()
        for name in ("net/tcp", "net/tcp6"):
            text = self._read(name)
            if not text:
                continue
            for line in text.splitlines()[1:]:
                p = line.split()
                if len(p) < 10 or p[3] != "0A":
                    continue
                try:
                    if int(p[1].rsplit(":", 1)[1], 16) == port:
                        got.add(p[9])
                except (ValueError, IndexError):
                    continue
        return got

    def _pid_has_socket(self, pid: int, inodes: set[str]) -> bool:
        d = os.path.join(self.proc, str(pid), "fd")
        try:
            for fd in os.listdir(d):
                try:
                    t = os.readlink(os.path.join(d, fd))
                except OSError:
                    continue
                if t.startswith("socket:[") and t[8:-1] in inodes:
                    return True
        except OSError:
            pass
        return False

    def _scan_fds(self, inodes: set[str]) -> int | None:
        try:
            pids = [int(x) for x in os.listdir(self.proc) if x.isdigit()]
        except OSError:
            return None
        for pid in sorted(pids, reverse=True):
            if self._pid_has_socket(pid, inodes):
                return pid
        return None

    def _lsof(self, port: int) -> int | None:
        if os.path.isdir(os.path.join(self.proc, "1")) or self.lsof is None:
            return None                               # Linux: /proc answered or nothing will
        try:
            r = subprocess.run([self.lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                               capture_output=True, text=True, timeout=2.0)
            first = r.stdout.split()
            return int(first[0]) if first else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def _ss(self, port: int) -> int | None:
        if self.ss is None or not os.path.isdir(os.path.join(self.proc, "1")):
            return None
        try:
            r = subprocess.run([self.ss, "-ltnpH", f"sport = :{port}"], capture_output=True,
                               text=True, timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            self.ss = None
            return None
        m = re.search(r"pid=(\d+)", r.stdout)
        return int(m.group(1)) if m else None

    def _proc_exists(self, pid: int) -> bool:
        if self.psutil is not None:
            try:
                return self.psutil.pid_exists(pid)
            except Exception:                                        # noqa: BLE001
                return False
        return os.path.isdir(os.path.join(self.proc, str(pid)))

    def _ppid_map(self) -> dict:
        m: dict = {}
        try:
            pids = [x for x in os.listdir(self.proc) if x.isdigit()]
        except OSError:
            return m
        for p in pids:
            st = self._read(f"{p}/stat")
            if not st:
                continue
            try:
                ppid = int(st[st.rindex(")") + 2:].split()[1])
            except (ValueError, IndexError):
                continue
            m.setdefault(ppid, []).append(int(p))
        return m

    def _children(self, pid: int, ppid_map: dict | None) -> list[int]:
        if self.psutil is not None:
            try:
                return [c.pid for c in self.psutil.Process(pid).children(recursive=True)]
            except Exception:                                        # noqa: BLE001
                return []
        out, todo = [], list((ppid_map or {}).get(pid, []))
        while todo and len(out) < 256:
            c = todo.pop(0)
            out.append(c)
            todo.extend((ppid_map or {}).get(c, []))
        return out

    def _proc_info(self, pid: int, now: float) -> dict | None:
        ps = self.psutil
        info = {"name": None, "cmdline": None, "rss_bytes": None, "threads": None,
                "open_files": None, "uptime_s": None, "cpu": None}
        cpu_s = None
        start = None            # the starttime: a pid alone does not identify a process
        if ps is not None:
            try:
                p = ps.Process(pid)
                with p.oneshot():
                    info["name"] = p.name()
                    try:
                        info["cmdline"] = redact_cmdline(p.cmdline(), self.home)
                    except Exception:                                # noqa: BLE001
                        pass
                    info["rss_bytes"] = int(p.memory_info().rss)
                    info["threads"] = p.num_threads()
                    try:
                        info["open_files"] = p.num_fds()
                    except Exception:                                # noqa: BLE001
                        pass
                    start = p.create_time()
                    info["uptime_s"] = round(max(0.0, now - start), 1)
                    t = p.cpu_times()
                    cpu_s = t.user + t.system
            except Exception:                                        # noqa: BLE001
                return None
        else:
            st = self._read(f"{pid}/stat")
            if st is None:
                return None
            try:
                info["name"] = st[st.index("(") + 1:st.rindex(")")]
                rest = st[st.rindex(")") + 2:].split()
                cpu_s = (int(rest[11]) + int(rest[12])) / self._tick
                start_ticks = int(rest[19]) / self._tick
                start = float(rest[19])   # raw ticks since boot: a recycled pid gets a new one
                btime = None
                for line in (self._read("stat") or "").splitlines():
                    if line.startswith("btime "):
                        btime = int(line.split()[1])
                if btime is not None:
                    info["uptime_s"] = round(max(0.0, now - (btime + start_ticks)), 1)
            except (ValueError, IndexError):
                pass
            status = self._read(f"{pid}/status") or ""
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    info["rss_bytes"] = int(line.split()[1]) * 1024
                elif line.startswith("Threads:"):
                    info["threads"] = int(line.split()[1])
            raw = self._read(f"{pid}/cmdline")
            if raw:
                info["cmdline"] = redact_cmdline([a for a in raw.split("\0") if a], self.home)
            try:
                info["open_files"] = len(os.listdir(os.path.join(self.proc, str(pid), "fd")))
            except OSError:
                pass
        if cpu_s is not None:
            # (P2#8): a recycled pid is a different process -- the starttime is in the key, so
            # the new process finds no baseline and never divides against the dead one's counters
            prev = self._prev_proc.get((pid, start))
            if prev is not None and now > prev[0] and cpu_s >= prev[1]:
                info["cpu"] = round((cpu_s - prev[1]) / (now - prev[0]), 3)
            self._prev_proc[(pid, start)] = (now, cpu_s)
        return info


def _smi_num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None                                       # "[N/A]" on GB10 for some fields


def _scale(v, k):
    return None if v is None else round(v * k, 4)


# ====================================================================== remote hosts

def _fetch(url: str, token: str | None, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/json, text/plain"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:            # noqa: S310
        return r.read(8 * 1024 * 1024)


class _Remote:
    """One remote source, read on a thread of its own at most every `period` seconds."""

    def __init__(self, kind: str, url: str, dcgm_url: str | None, token: str | None,
                 clock=time.time, period: float = 5.0, timeout: float = 2.0, fetch=_fetch):
        self.kind, self.url, self.dcgm_url, self.token = kind, url, dcgm_url, token
        self.clock, self.period, self.timeout, self.fetch = clock, period, timeout, fetch
        self.watch: list[dict] = []
        self.data: dict | None = None
        self.at: float | None = None
        self.error: str | None = None
        self.busy = False
        self._lock = threading.Lock()
        self._prev: tuple | None = None     # (t, node counters) for the exporter's rates

    def poke(self, watch: list[dict], wait: float = 0.0) -> None:
        with self._lock:
            self.watch = watch
            now = self.clock()
            if self.busy or (self.at is not None and now - self.at < self.period
                             and self.error is None) or \
                    (self.error is not None and self.at is not None and now - self.at < 15):
                return
            self.busy = True
        t = threading.Thread(target=self._run, name="hostcollect-remote", daemon=True)
        t.start()
        if wait:
            t.join(wait)

    def _run(self) -> None:
        try:
            data = self._read()
            err = None
        except Exception as exc:                                     # noqa: BLE001
            data, err = None, _short_error(exc)
        with self._lock:
            self.at = self.clock()
            self.error = err
            if data is not None:
                self.data = data
            self.busy = False

    def _read(self) -> dict:
        if self.kind == "collector":
            q = ",".join(f"{w['backend']}:{w['port']}" for w in self.watch if w.get("port"))
            url = self.url.rstrip("/") + "/host" + (f"?watch={urllib.parse.quote(q)}" if q
                                                    else "")
            body = json.loads(self.fetch(url, self.token, self.timeout))
            if not isinstance(body, dict) or "memory" not in body:
                raise ValueError("not a host collector's answer")
            return body
        # exporter: node_exporter for the machine, DCGM for the GPUs
        now = self.clock()
        out = {"memory": None, "cpu": None, "disks": [], "net": [], "fs": [], "gpus": [],
               "gpu_source": None, "processes": None}
        if self.url:
            cur = scrape.node_counters(scrape.parse(
                self.fetch(self.url, None, self.timeout).decode(errors="replace")))
            prev = self._prev
            out.update(scrape.node_host(cur, prev[1] if prev else None,
                                        now - prev[0] if prev else None))
            self._prev = (now, cur)
        if self.dcgm_url:
            out["gpus"] = scrape.dcgm_gpus(scrape.parse(
                self.fetch(self.dcgm_url, None, self.timeout).decode(errors="replace")))
            out["gpu_source"] = "dcgm" if out["gpus"] else None
            if out["gpus"] and out["memory"] is not None:
                out["memory"]["unified"] = all(g["memory_unified"] for g in out["gpus"])
        return out


def _short_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return f"unreachable: {exc.reason}"[:160]
    return f"{type(exc).__name__}: {exc}"[:160]


# ====================================================================== which host a backend is on

class _Locality:
    """Is a URL's host this machine? Loopback, this host's name, or an address this machine can
    bind. A name is resolved on a thread and remembered for 10 minutes, so a slow DNS never
    holds the System view (the answer is `False` until it comes)."""

    def __init__(self, clock=time.time, resolve=None):
        self.clock = clock
        self.resolve = resolve or (lambda h: {ai[4][0] for ai in socket.getaddrinfo(h, None)})
        self._cache: dict = {}
        self._lock = threading.Lock()

    def is_local(self, url: str) -> bool:
        try:
            host = urllib.parse.urlsplit(url).hostname or ""
        except ValueError:
            return False
        host = host.strip("[]").lower()
        if not host:
            return False
        if host == "localhost" or host.endswith(".localhost"):
            return True
        if host in (socket.gethostname().lower(), socket.gethostname().split(".")[0].lower()):
            return True
        try:
            return _ip_local(ipaddress.ip_address(host))
        except ValueError:
            pass
        with self._lock:
            got = self._cache.get(host)
            fresh = got is not None and self.clock() - got[0] < 600
            if not fresh and not (got and got[2]):
                self._cache[host] = (got[0] if got else 0, got[1] if got else False, True)
                threading.Thread(target=self._resolve, args=(host,), daemon=True).start()
        return bool(got and got[1])

    def _resolve(self, host: str) -> None:
        try:
            ips = self.resolve(host)
            local = any(_ip_local(ipaddress.ip_address(i.split("%")[0])) for i in ips)
        except Exception:                                            # noqa: BLE001
            local = False
        with self._lock:
            self._cache[host] = (self.clock(), local, False)


def _ip_local(ip) -> bool:
    if ip.is_loopback or ip.is_unspecified:
        return True
    fam = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
    try:
        with socket.socket(fam, socket.SOCK_DGRAM) as s:
            s.bind((str(ip), 0))
        return True
    except OSError:
        return False


def _port_of(url: str) -> int | None:
    try:
        u = urllib.parse.urlsplit(url)
        return u.port or (443 if u.scheme == "https" else 80)
    except ValueError:
        return None


def placement(backend: dict, locality: _Locality) -> tuple[str, dict] | None:
    """('local' | 'collector' | 'exporter', the backend's host settings) or None."""
    h = dict(backend.get("host") or {})
    mode = h.get("mode") or "auto"
    if mode == "off" or not backend.get("enabled", True):
        return None
    if mode == "same":
        return "local", h
    if mode == "collector" and h.get("url"):
        return "collector", h
    if mode == "exporter" and (h.get("url") or h.get("dcgm_url")):
        return "exporter", h
    if mode == "auto" and not backend.get("cloud") and locality.is_local(backend.get("url", "")):
        return "local", h
    return None


class Hosts:
    """`system.hosts[]`: this machine first (always, with the engine and the backends on it),
    then one entry per remote collector or exporter pair the backends name."""

    def __init__(self, collector: HostCollector | None = None, *, env=None, clock=time.time,
                 fetch=_fetch, locality: _Locality | None = None):
        self.collector = collector or HostCollector()
        self.env = os.environ if env is None else env
        self.clock = clock
        self.fetch = fetch
        self.locality = locality or _Locality(clock=clock)
        self.remotes: dict = {}
        # view() runs per dashboard request (ThreadingHTTPServer); the get-or-create of a
        # _Remote is two steps, and two requests missing the same key used to each build
        # one — the replaced object's fetch thread then ran against a discarded entry
        self._remotes_guard = threading.Lock()

    def view(self, backends: list[dict] | None, engine: dict | None = None,
             wait: float = 0.0) -> list[dict]:
        local_watch = [dict(engine, backend=engine.get("backend") or "engine")] if engine else []
        groups: dict = {}
        for b in backends or []:
            where = placement(b, self.locality)
            if where is None:
                continue
            kind, h = where
            port = _port_of(b.get("url", ""))
            w = {"backend": b["id"], "label": b.get("label"), "port": port}
            if kind == "local":
                if engine and engine.get("port") == port:
                    continue                    # our engine registered as a backend of itself
                local_watch.append(w)
            else:
                key = (kind, (h.get("url") or "").rstrip("/"), h.get("dcgm_url") or "")
                groups.setdefault(key, []).append(w)
        loc = self.collector.sample(local_watch)
        out = [dict(loc, id="local", label=self.collector.label, source="local", url=None,
                    backends=[w["backend"] for w in local_watch], state="ok", error=None)]
        token = self.env.get(TOKEN_ENV) or None
        with self._remotes_guard:
            for key in list(self.remotes):
                if key not in groups:
                    del self.remotes[key]
            for key, _watch in groups.items():
                if key not in self.remotes:
                    kind, url, dcgm = key
                    self.remotes[key] = _Remote(kind, url, dcgm or None,
                                                token if kind == "collector" else None,
                                                clock=self.clock, fetch=self.fetch)
        for i, (key, watch) in enumerate(sorted(groups.items())):
            kind, url, dcgm = key
            r = self.remotes.get(key)
            if r is None:                              # pruned by a concurrent view — rebuild
                r = self.remotes[key] = _Remote(kind, url, dcgm or None,
                                                token if kind == "collector" else None,
                                                clock=self.clock, fetch=self.fetch)
            r.poke(watch, wait)
            out.append(self._remote_view(r, kind, url or dcgm, watch, i))
        return out

    def _remote_view(self, r: _Remote, kind: str, url: str, watch: list[dict], i: int) -> dict:
        d = r.data or {}
        procs = d.get("processes") if kind == "collector" else None
        if procs is not None:
            by_port = {w["port"]: w for w in watch}
            for p in procs:
                w = by_port.get(p.get("port"))
                if w:
                    p["backend"], p["label"] = w["backend"], w.get("label")
        host = urllib.parse.urlsplit(url).hostname or url
        age = None if r.at is None else round(max(0.0, self.clock() - r.at), 1)
        state = "pending" if r.data is None and r.error is None else \
            ("error" if r.error else ("stale" if age is not None and age > 60 else "ok"))
        return {"id": f"{kind}-{i + 1}", "label": d.get("label") or host, "source": kind,
                "url": url, "backends": [w["backend"] for w in watch], "state": state,
                "error": r.error,
                "memory": d.get("memory"), "cpu": d.get("cpu"), "gpus": d.get("gpus") or [],
                "gpu_source": d.get("gpu_source"), "disks": d.get("disks") or [],
                "fs": d.get("fs") or [], "net": d.get("net") or [], "processes": procs,
                "sampled_at": d.get("sampled_at") or (_iso(r.at) if r.at and r.data else None),
                "age_s": age, "cost": d.get("cost") if kind == "collector" else None}


def engine_entry(port: int | None, label: str | None = None, backend: str = "engine") -> dict:
    """The watch entry for the server's own process (our engine, or the proxy itself)."""
    return {"backend": backend, "label": label, "pid": os.getpid(), "port": port}


def hosts_for(state: dict, backends, engine: dict | None, paths=()) -> list | None:
    """`system.hosts` for a server: one `Hosts` kept in its STATE, None (the key left out) when
    the collector fails, with one log line the first time."""
    try:
        h = state.get("hosts")
        if h is None:
            h = state["hosts"] = Hosts(HostCollector(paths=[p for p in paths if p]))
        return h.view(backends, engine)
    except Exception as exc:                                         # noqa: BLE001
        if not state.get("hosts_error"):
            state["hosts_error"] = True
            print(f"[hostcollect] off for this run: {type(exc).__name__}: {exc}", flush=True)
        return None


def gpu_block(state: dict) -> dict | None:
    h = state.get("hosts")
    return h.collector.gpu_block() if h is not None else None


# ====================================================================== the remote collector server

def _esc(x) -> str:
    return str(x).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def prometheus(host: dict) -> str:
    """The host as Prometheus text (`tandem_host_*`), for a scraper that wants it that way."""
    lines = []

    def g(name, help_, rows):
        rows = [(lab, v) for lab, v in rows if v is not None]
        if not rows:
            return
        lines.append(f"# HELP {name} {help_}")
        lines.append(f"# TYPE {name} gauge")
        for lab, v in rows:
            ls = ",".join(f'{k}="{_esc(x)}"' for k, x in lab.items())
            lines.append(f"{name}{{{ls}}} {v}" if ls else f"{name} {v}")
    m, c = host.get("memory") or {}, host.get("cpu") or {}
    g("tandem_host_memory_total_bytes", "memory total", [({}, m.get("total_bytes"))])
    g("tandem_host_memory_available_bytes", "memory available (MemAvailable)",
      [({}, m.get("available_bytes"))])
    g("tandem_host_cpu_utilization_ratio", "CPU use over the last interval, 0-1",
      [({}, c.get("utilization"))])
    g("tandem_host_load1", "load average, 1 minute", [({}, (c.get("load") or [None])[0])])
    gp = host.get("gpus") or []
    g("tandem_host_gpu_utilization_ratio", "GPU use 0-1",
      [({"gpu": x["index"]}, x["utilization"]) for x in gp])
    g("tandem_host_gpu_power_watts", "GPU power draw", [({"gpu": x["index"]}, x["power_w"])
                                                         for x in gp])
    g("tandem_host_gpu_temperature_celsius", "GPU temperature",
      [({"gpu": x["index"]}, x["temperature_c"]) for x in gp])
    g("tandem_host_gpu_sm_clock_mhz", "GPU SM clock", [({"gpu": x["index"]}, x["sm_clock_mhz"])
                                                        for x in gp])
    g("tandem_host_gpu_memory_used_bytes", "GPU memory used (absent on unified memory)",
      [({"gpu": x["index"]}, x["memory_used_bytes"]) for x in gp])
    g("tandem_host_disk_read_bytes_per_second", "disk reads",
      [({"device": d["name"]}, d["read_bps"]) for d in host.get("disks") or []])
    g("tandem_host_disk_write_bytes_per_second", "disk writes",
      [({"device": d["name"]}, d["write_bps"]) for d in host.get("disks") or []])
    g("tandem_host_network_receive_bytes_per_second", "network in",
      [({"device": d["name"]}, d["rx_bps"]) for d in host.get("net") or []])
    g("tandem_host_network_transmit_bytes_per_second", "network out",
      [({"device": d["name"]}, d["tx_bps"]) for d in host.get("net") or []])
    procs = host.get("processes") or []
    g("tandem_host_process_rss_bytes", "RSS of a watched process and its children",
      [({"port": p["port"], "backend": p["backend"] or ""}, p["total_rss_bytes"]) for p in procs])
    g("tandem_host_process_gpu_memory_bytes", "GPU memory of a watched process tree",
      [({"port": p["port"], "backend": p["backend"] or ""}, p["total_gpu_memory_bytes"])
       for p in procs])
    return "\n".join(lines) + "\n"


def parse_watch(text: str | None) -> list[dict]:
    out = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        bid, sep, port = item.rpartition(":")
        if not sep:
            bid, port = None, item
        try:
            p = int(port)
        except ValueError:
            continue
        if 0 < p < 65536:
            out.append({"backend": (bid or None) and bid[:64], "port": p})
    return out[:32]


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT, token: str | None = None,
                watch: list[dict] | None = None, collector: HostCollector | None = None):
    """The remote collector's HTTP server (not started). `GET /host` answers the host as JSON,
    `GET /metrics` as Prometheus text, both behind the bearer token when one is set;
    `GET /health` is open. A request's `?watch=id:port,...` adds ports to the fixed ones."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback and not token:
        raise SystemExit(f"[hostcollect] bound to {host} without a token: set {TOKEN_ENV} or "
                         f"--token-file (the answer names processes and their command lines)")
    col = collector or HostCollector()
    fixed = list(watch or [])

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "TandemLLM-hostcollect"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def do_GET(self):
            u = urllib.parse.urlsplit(self.path)
            if u.path == "/health":
                return self._send(200, b'{"status":"ok"}')
            if token:
                got = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                if not hmac.compare_digest(got.encode(), token.encode()):
                    return self._send(401, b'{"error":{"type":"unauthorized"}}')
            if u.path not in ("/host", "/metrics"):
                return self._send(404, b'{"error":{"type":"not_found"}}')
            ports = {f["port"] for f in fixed}
            extra = parse_watch(urllib.parse.parse_qs(u.query).get("watch", [""])[0])
            host = col.sample(fixed + [w for w in extra if w["port"] not in ports])
            host["label"] = col.label
            if u.path == "/host":
                return self._send(200, json.dumps(host).encode())
            return self._send(200, prometheus(host).encode(),
                              "text/plain; version=0.0.4; charset=utf-8")

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    srv.collector = col
    return srv


def main(argv=None) -> None:                                          # pragma: no cover
    ap = argparse.ArgumentParser(prog="python -m proxy.hostcollect",
                                 description="the host collector for backends on this machine")
    ap.add_argument("--host", default="127.0.0.1",
                    help="the address to bind; anything but loopback needs a token")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--token-file", default=None,
                    help=f"a file whose first line is the bearer token; else ${TOKEN_ENV}")
    ap.add_argument("--watch", default="",
                    help="ports to report always, as id:port,id:port (a request may add more)")
    ap.add_argument("--path", action="append", default=[],
                    help="a path whose filesystem to report (repeatable)")
    a = ap.parse_args(argv)
    token = os.environ.get(TOKEN_ENV) or None
    if a.token_file:
        with open(os.path.expanduser(a.token_file)) as f:
            token = f.readline().strip() or None
    srv = make_server(a.host, a.port, token, parse_watch(a.watch), HostCollector(paths=a.path))
    col = srv.collector
    print(f"[hostcollect] on {a.host}:{a.port}, token {'set' if token else 'off (loopback)'}, "
          f"psutil {'yes' if col.psutil else 'no'}, nvml "
          f"{'yes' if col._nvml_handle() else 'no'}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":                                            # pragma: no cover
    main()
