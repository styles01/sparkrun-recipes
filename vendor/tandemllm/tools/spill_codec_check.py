"""The spill codec on a real entry (0.4.3-test): every block through the engine's own byte planes
and zstd frames and back, compared byte for byte and against the sums the manifest holds.

Run it on a COPY of an entry (never on the live sessions/ directory), on the CPU:

    nice -n 15 python tools/spill_codec_check.py ~/cache1-work/entries/<eid> [--threads 4]

Prints the ratio per part (target KV, drafter KV), the size of the entry compressed, and the
compress and decompress rates; exits 1 if any block does not come back bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine import disk_cache as dc  # noqa: E402
from engine import spill  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.entry, "manifest.json")))
    layout = man["layout"]
    sp = spill.SessionSpill(os.path.join(a.entry, ".check"), chunk=man["chunk"], fingerprint="x",
                            settings={"codec_threads": a.threads})
    pool = sp._zp()
    raw_b = disk_b = 0
    part = {"target": [0, 0], "drafter": [0, 0]}
    tc = td = 0.0
    bad = 0
    buf = None
    for bl in man["blocks"]:
        if bl.get("z"):
            print(f"block {bl['b']}: already compressed, skipped")
            continue
        p = os.path.join(a.entry, f"r{int(bl['b']):07d}.blk")
        size = int(bl["bytes"])
        if buf is None or buf.nbytes < size:
            buf = dc._Buf(size, False)
        dc.read_into(p, 0, buf.view(0, size))
        meta = spill.SessionSpill._get_header(buf)
        r = int(meta["rows"])
        views = []
        for (dt, shape, ax), (off, nb) in zip(layout, meta["sections"]):
            shp = list(shape)
            shp[ax] = r
            t = buf.typed(spill.ALIGN + off, getattr(torch, dt.split(".")[-1]), shp).clone()
            views.append((t, ax))
        sums = [spill.block_sums(t, ax, 0, r) for t, ax in views]
        if sums != bl["sums"]:
            print(f"block {bl['b']}: the file's rows do not match the manifest's sums (copy?)")
            bad += 1
            continue
        for i, (t, ax) in enumerate(views):
            nb = t.numel() * t.element_size()
            host = torch.empty(nb, dtype=torch.uint8)
            spill._planes_to_host(t, host, False)
            mv = memoryview(host.numpy())
            frames = [(o, min(spill.CODEC_FRAME, nb - o)) for o in range(0, nb, spill.CODEC_FRAME)]
            q = time.perf_counter()
            zs = list(pool.map(spill._zc, [mv[o:o + n] for o, n in frames]))
            tc += time.perf_counter() - q
            back = torch.zeros(nb, dtype=torch.uint8)
            bm = memoryview(back.numpy())
            q = time.perf_counter()
            got = list(pool.map(spill._zd, [(z, bm[o:o + n]) for z, (o, n) in zip(zs, frames)]))
            td += time.perf_counter() - q
            assert got == [n for _, n in frames]
            out = torch.zeros_like(t)
            spill._planes_from_host(back, out, False)
            if not torch.equal(out.view(torch.uint8), t.view(torch.uint8)):
                bad += 1
                print(f"block {bl['b']} section {i}: NOT bit for bit")
            k = "target" if i < 2 else "drafter"
            zb = sum(len(z) for z in zs)
            part[k][0] += nb
            part[k][1] += zb
            raw_b += nb
            disk_b += zb
        if sp.codec is None:
            print("zstandard missing")
            return 1
    anchors = len(man.get("anchor_files") or [])
    anc = anchors * (os.path.getsize(os.path.join(a.entry, man["anchor_files"][0]["file"]))
                     if anchors else 0)
    print(f"entry {man['eid']}: {man['valid']} tokens, {len(man['blocks'])} blocks, "
          f"{anchors} anchors ({anc / 1e9:.2f} GB, kept as they are)")
    for k, (r_, z_) in part.items():
        if z_:
            print(f"  {k:8s} {r_ / 1e9:6.2f} GB -> {z_ / 1e9:6.2f} GB  {r_ / z_:.3f}x")
    if disk_b:
        print(f"  rows     {raw_b / 1e9:6.2f} GB -> {disk_b / 1e9:6.2f} GB  {raw_b / disk_b:.3f}x; "
              f"entry {(raw_b + anc) / 1e9:.2f} -> {(disk_b + anc) / 1e9:.2f} GB "
              f"({(raw_b + anc) / max(1, disk_b + anc):.3f}x)")
        print(f"  zstd on {a.threads} threads: compress {raw_b / tc / 1e9:.2f} GB/s, "
              f"decompress {raw_b / td / 1e9:.2f} GB/s")
    print("BIT FOR BIT" if not bad else f"{bad} BLOCKS DIFFER")
    sp.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
