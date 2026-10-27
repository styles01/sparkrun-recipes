"""`tools/bubbles.py`: the idle time between kernels, from an Nsight Systems export.

Overlapping kernels (a second stream) count once, gaps land in their size bucket, and the kernel
table is read from the export's own schema."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gaps_merge_overlaps_and_bucket_the_idle():
    from tools.bubbles import gaps
    # [0,10) [5,20) overlap -> busy 20; gap 3 us; [23,30); gap 20 us; [50,60); gap 100 us; [160,170)
    us = 1_000
    iv = [(160 * us, 170 * us), (0, 10 * us), (5 * us, 20 * us), (23 * us, 30 * us),
          (50 * us, 60 * us)]
    g = gaps(iv)
    assert g["busy_ns"] == (20 + 7 + 10 + 10) * us
    assert g["span_ns"] == 170 * us
    assert g["idle_ns"] == {"<=5us": 3 * us, "5-50us": 20 * us, ">50us": 100 * us}
    assert g["idle_count"] == {"<=5us": 1, "5-50us": 1, ">50us": 1}
    assert g["busy_ns"] + sum(g["idle_ns"].values()) == g["span_ns"]
    assert g["kernels"] == 5


def test_no_kernels_is_zero():
    from tools.bubbles import gaps
    assert gaps([])["busy_ns"] == 0


def test_the_report_reads_the_export_and_divides_by_blocks():
    from tools.bubbles import gaps, read_kernels, report
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, name INTEGER)")
    con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0)",
                    [(0, 4_000_000), (4_002_000, 8_000_000)])
    con.commit()
    con.close()
    try:
        text = report(gaps(read_kernels(path)), blocks=2)
    finally:
        os.remove(path)
    assert "span 4.00 ms a block = kernels 4.00 + idle 0.00" in text
    assert "<=5us 0 = 0.00 ms" in text


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
