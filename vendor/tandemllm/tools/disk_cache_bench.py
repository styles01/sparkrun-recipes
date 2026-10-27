"""The disk tier's numbers on this box, CPU only: what the NVMe writes and reads, and a checksum.

    python tools/disk_cache_bench.py [--dir DIR] [--gb 4] [--chunk-mb 104]

Writes `--gb` of chunk-sized files into a fresh temporary directory under `--dir` (default: the
system's temporary directory), the way `engine/disk_cache.py` writes them -- O_DIRECT from an
aligned buffer, fsync, rename -- then reads them back with O_DIRECT, and prints the rates. The
directory is removed at the end. It never touches the GPU, the engine's home or a cache that is in
use; run it with `nice -n 19` beside a serving engine, and never during a hold.

The checksum rows say what the read path pays per byte on top of the read: the tier verifies every
chunk before a byte of it reaches the engine's buffers.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import disk_cache  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=tempfile.gettempdir(),
                    help="where the temporary directory goes (the file system to measure)")
    ap.add_argument("--gb", type=float, default=4.0, help="GB written and read back")
    ap.add_argument("--chunk-mb", type=float, default=104.0,
                    help="one file's payload: a 1,024-token chunk of the 27B is 104 MiB "
                         "(65,536 B of target KV and 40,960 B of drafter KV a token)")
    a = ap.parse_args(argv)
    size = disk_cache.align(int(a.chunk_mb * (1 << 20)))
    n = max(1, int(a.gb * 1e9 // size))
    d = tempfile.mkdtemp(prefix="disk-cache-bench-", dir=a.dir)
    buf = mmap.mmap(-1, size)
    view = memoryview(buf)
    for i in range(0, size, 1 << 20):
        view[i:i + 8] = i.to_bytes(8, "little")
    out = {"dir": d, "files": n, "file_bytes": size, "direct": None}
    try:
        t0 = time.perf_counter()
        direct = True
        for i in range(n):
            direct &= disk_cache.write_file(os.path.join(d, f"c{i}"), [view], fsync=True)
        w = time.perf_counter() - t0
        out["direct"] = direct
        out["write_gbs"] = round(n * size / w / 1e9, 3)
        t0 = time.perf_counter()
        for i in range(n):
            disk_cache.read_into(os.path.join(d, f"c{i}"), 0, view)
        r = time.perf_counter() - t0
        out["read_gbs"] = round(n * size / r / 1e9, 3)
        for name in ("crc32", "sum64"):
            t0 = time.perf_counter()
            for _ in range(3):
                disk_cache.checksum(view, name)
            out[f"{name}_gbs"] = round(3 * size / (time.perf_counter() - t0) / 1e9, 3)
    finally:
        view.release()
        buf.close()
        shutil.rmtree(d, ignore_errors=True)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
