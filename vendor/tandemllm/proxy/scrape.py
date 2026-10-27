"""Prometheus text pages read in Python: the parser, and the name tables per kind of page.

    samples = parse(text)        [(name, {label: value}, value), ...]

P1.5 reads two kinds of page a user may already run on a backend's host: node_exporter (memory,
CPU, load, disks, network, temperatures) and NVIDIA's DCGM exporter (GPU use, framebuffer, power,
temperature, clocks). `node_host` and `dcgm_gpus` turn them into the host collector's shape
(proxy/hostcollect.py); counters become rates against the previous page. P2 and P3 add the vLLM
and SGLang tables here.

The parser takes the text exposition format 0.0.4 (and the OpenMetrics text that exporters send
with it): comments and `# TYPE` lines are skipped, label values may hold escaped quotes, a sample
may carry a timestamp, and `NaN`, `+Inf` and `-Inf` are floats. A line it cannot read is skipped,
never an error.
"""

from __future__ import annotations

import math
import re

_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*,?')


def _unescape(v: str) -> str:
    return v.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")


def _float(text: str) -> float | None:
    t = text.strip()
    try:
        return float(t)
    except ValueError:
        low = t.lower()
        if low in ("nan",):
            return math.nan
        if low in ("+inf", "inf"):
            return math.inf
        if low == "-inf":
            return -math.inf
        return None


def parse(text: str) -> list[tuple[str, dict, float]]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _NAME.match(line)
        if not m:
            continue
        name = m.group(0)
        rest = line[m.end():]
        labels: dict = {}
        if rest.startswith("{"):
            end = _label_end(rest)
            if end < 0:
                continue
            body = rest[1:end]
            pos = 0
            while pos < len(body):
                lm = _LABEL.match(body, pos)
                if not lm:
                    break
                labels[lm.group(1)] = _unescape(lm.group(2))
                pos = lm.end()
            rest = rest[end + 1:]
        parts = rest.split()
        if not parts:
            continue
        v = _float(parts[0])
        if v is None:
            continue
        out.append((name, labels, v))
    return out


def _label_end(s: str) -> int:
    """The index of the `}` that closes the label set, skipping quoted values."""
    q = False
    i = 1
    while i < len(s):
        c = s[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                q = False
        elif c == '"':
            q = True
        elif c == "}":
            return i
        i += 1
    return -1


def _num(v: float | None) -> float | None:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return v


# ---------------------------------------------------------------------- node_exporter

# whole block devices only: a partition or a device-mapper volume counts its I/O twice
DISK_RE = re.compile(r"^(nvme\d+n\d+|sd[a-z]+|vd[a-z]+|xvd[a-z]+|hd[a-z]+|mmcblk\d+|disk\d+)$")
NET_SKIP = re.compile(r"^(lo|veth.*|docker.*|br-.*|virbr.*|cni.*|flannel.*|cali.*|tunl.*)$")


def node_counters(samples) -> dict:
    """The counters and gauges of a node_exporter page that the host shape needs."""
    c = {"mem_total": None, "mem_available": None, "swap_total": None, "swap_free": None,
         "cpu": {}, "load": [None, None, None], "disks": {}, "net": {}, "temps": [],
         "cpus": set(), "boot": None}
    for name, lab, v in samples:
        if name == "node_memory_MemTotal_bytes":
            c["mem_total"] = v
        elif name == "node_memory_MemAvailable_bytes":
            c["mem_available"] = v
        elif name == "node_memory_SwapTotal_bytes":
            c["swap_total"] = v
        elif name == "node_memory_SwapFree_bytes":
            c["swap_free"] = v
        elif name == "node_cpu_seconds_total":
            mode = lab.get("mode", "")
            c["cpu"][mode] = c["cpu"].get(mode, 0.0) + v
            c["cpus"].add(lab.get("cpu"))
        elif name in ("node_load1", "node_load5", "node_load15"):
            c["load"][("node_load1", "node_load5", "node_load15").index(name)] = v
        elif name in ("node_disk_read_bytes_total", "node_disk_written_bytes_total",
                      "node_disk_io_time_seconds_total"):
            dev = lab.get("device", "")
            if DISK_RE.match(dev):
                c["disks"].setdefault(dev, {})[name] = v
        elif name in ("node_network_receive_bytes_total", "node_network_transmit_bytes_total"):
            dev = lab.get("device", "")
            if not NET_SKIP.match(dev):
                c["net"].setdefault(dev, {})[name] = v
        elif name in ("node_hwmon_temp_celsius", "node_thermal_zone_temp"):
            if _num(v) is not None:
                c["temps"].append(v)
        elif name == "node_boot_time_seconds":
            c["boot"] = v
    return c


def node_host(cur: dict, prev: dict | None, dt: float | None) -> dict:
    """The host shape's memory, CPU, disks and network from two node_exporter readings."""
    cpu_util = None
    if prev and dt and cur["cpu"] and prev["cpu"]:
        tot = sum(cur["cpu"].values()) - sum(prev["cpu"].values())
        idle = (cur["cpu"].get("idle", 0) + cur["cpu"].get("iowait", 0)
                - prev["cpu"].get("idle", 0) - prev["cpu"].get("iowait", 0))
        if tot > 0:
            cpu_util = round(max(0.0, min(1.0, 1 - idle / tot)), 4)

    def rate(a, b, key):
        if not prev or not dt or b is None or key not in a or key not in b:
            return None
        d = a[key] - b[key]
        return None if d < 0 else round(d / dt, 1)
    disks = []
    for dev, a in sorted(cur["disks"].items()):
        b = (prev or {}).get("disks", {}).get(dev)
        busy = rate(a, b, "node_disk_io_time_seconds_total")
        disks.append({"name": dev, "nvme": dev.startswith("nvme"),
                      "read_bps": rate(a, b, "node_disk_read_bytes_total"),
                      "write_bps": rate(a, b, "node_disk_written_bytes_total"),
                      "busy": None if busy is None else round(min(1.0, busy), 4)})
    net = []
    for dev, a in sorted(cur["net"].items()):
        if not a.get("node_network_receive_bytes_total") and \
                not a.get("node_network_transmit_bytes_total"):
            continue
        b = (prev or {}).get("net", {}).get(dev)
        net.append({"name": dev, "rx_bps": rate(a, b, "node_network_receive_bytes_total"),
                    "tx_bps": rate(a, b, "node_network_transmit_bytes_total")})
    total, avail = cur["mem_total"], cur["mem_available"]
    st, sf = cur["swap_total"], cur["swap_free"]
    return {
        "memory": {"total_bytes": _int(total), "available_bytes": _int(avail),
                   "used_bytes": _int(total - avail) if total is not None and avail is not None
                   else None,
                   "swap_total_bytes": _int(st),
                   "swap_used_bytes": _int(st - sf) if st is not None and sf is not None
                   else None,
                   "unified": None},
        "cpu": {"count": len([x for x in cur["cpus"] if x is not None]) or None,
                "utilization": cpu_util, "load": cur["load"],
                "temperature_c": round(max(cur["temps"]), 1) if cur["temps"] else None},
        "disks": disks, "net": net,
    }


def _int(v) -> int | None:
    v = _num(v)
    return None if v is None else int(v)


# ---------------------------------------------------------------------- DCGM exporter

DCGM = {
    "DCGM_FI_DEV_GPU_UTIL": ("utilization", 0.01),
    "DCGM_FI_DEV_FB_USED": ("memory_used_bytes", 1024 * 1024),
    "DCGM_FI_DEV_FB_FREE": ("_fb_free", 1024 * 1024),
    "DCGM_FI_DEV_POWER_USAGE": ("power_w", 1.0),
    "DCGM_FI_DEV_GPU_TEMP": ("temperature_c", 1.0),
    "DCGM_FI_DEV_SM_CLOCK": ("sm_clock_mhz", 1.0),
    "DCGM_FI_DEV_MEM_CLOCK": ("mem_clock_mhz", 1.0),
}


def dcgm_gpus(samples) -> list[dict]:
    """One GPU entry per `gpu` label of a DCGM exporter page, in the host collector's shape."""
    gpus: dict = {}
    for name, lab, v in samples:
        if name not in DCGM:
            continue
        idx = lab.get("gpu", "0")
        g = gpus.setdefault(idx, {"index": int(idx) if str(idx).isdigit() else None,
                                  "name": lab.get("modelName"), "uuid": lab.get("UUID"),
                                  "utilization": None, "memory_used_bytes": None,
                                  "memory_total_bytes": None, "memory_unified": False,
                                  "power_w": None, "power_limit_w": None,
                                  "temperature_c": None, "sm_clock_mhz": None,
                                  "mem_clock_mhz": None, "source": "dcgm"})
        field, scale = DCGM[name]
        val = _num(v)
        g[field] = None if val is None else round(val * scale, 4)
    out = []
    for g in gpus.values():
        free = g.pop("_fb_free", None)
        if g["memory_used_bytes"] is not None:
            g["memory_used_bytes"] = int(g["memory_used_bytes"])
            if free is not None:
                g["memory_total_bytes"] = int(g["memory_used_bytes"] + free)
        if g["name"] and "GB10" in g["name"]:
            g["memory_unified"] = True
        out.append(g)
    return sorted(out, key=lambda g: (g["index"] is None, g["index"] or 0))
