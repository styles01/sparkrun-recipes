"""`tools/longctx_probe.py`'s memory curve: what it reads, when it stops, which prompt.

The 131k probe of 2026-09-23 wedged the board; the probe is how the long lengths come back, so
the parts that decide whether it goes on to a longer length are tested here without a board:

  * /proc/meminfo is read in GB, MemAvailable and MemFree;
  * the sampler keeps the minimum over the window, and polls once more at the end;
  * the kernel log's NVRM lines are what journalctl gives since the probe started, None when it
    cannot be read;
  * an NVRM out-of-memory line, or a minimum below the floor, stops the probe; neither does not;
  * a length the prompt set lacks is the head of the same domain's next longer prompt.

Run: python tests/test_longctx_probe.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.longctx_probe import (MemSampler, load_prompt, meminfo_gb, nvrm_lines,  # noqa: E402
                                 repeat_report, stop_reason)

MEMINFO = """MemTotal:       127535716 kB
MemFree:         9437184 kB
MemAvailable:   66060288 kB
Buffers:           12345 kB
"""


def test_meminfo_is_read_in_gb():
    m = meminfo_gb(MEMINFO)
    assert abs(m["MemAvailable"] - 66060288 * 1024 / 1e9) < 1e-9, m
    assert abs(m["MemFree"] - 9437184 * 1024 / 1e9) < 1e-9, m


def test_the_sampler_keeps_the_minimum_and_polls_at_the_end():
    seq = iter([{"MemAvailable": 60.0, "MemFree": 9.0}, {"MemAvailable": 41.5, "MemFree": 3.0}]
               + [{"MemAvailable": 55.0, "MemFree": 8.0}] * 1000)
    with MemSampler(period=0.01, read=lambda: next(seq)) as ms:
        time.sleep(0.05)
    r = ms.report()
    assert r["min_available_gb"] == 41.5 and r["min_free_gb"] == 3.0, r
    assert r["samples"] >= 3, r
    # the final poll: a drop that lands after the last tick is still seen
    ms2 = MemSampler(period=10, read=iter([{"MemAvailable": 50.0, "MemFree": 5.0},
                                           {"MemAvailable": 12.0, "MemFree": 1.0}]).__next__)
    with ms2:
        pass
    assert ms2.report()["min_available_gb"] == 12.0


def test_nvrm_lines_come_from_the_kernel_log_since_the_start():
    seen = {}

    def run(cmd, **kw):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="Sep 25 kernel: usb 1-1: new device\n"
                               "Sep 25 kernel: NVRM: _memdescAllocInternal: Out of memory "
                               "[NV_ERR_NO_MEMORY]\n")
    got = nvrm_lines(1758760000.7, run=run)
    assert got == ["Sep 25 kernel: NVRM: _memdescAllocInternal: Out of memory [NV_ERR_NO_MEMORY]"]
    assert seen["cmd"][:2] == ["journalctl", "-k"] and "@1758760000" in seen["cmd"], seen
    assert nvrm_lines(0, run=lambda *a, **k: SimpleNamespace(returncode=1, stdout="")) is None

    def boom(*a, **k):
        raise FileNotFoundError("journalctl")
    assert nvrm_lines(0, run=boom) is None
    assert nvrm_lines(0, run=lambda *a, **k: (_ for _ in ()).throw(
        subprocess.TimeoutExpired("journalctl", 30))) is None


def test_the_probe_stops_on_an_oom_line_or_below_the_floor():
    ok = {"min_available_gb": 35.0}
    assert stop_reason(ok, [], 20.0) is None
    assert stop_reason(ok, None, 20.0) is None
    assert stop_reason(ok, ["NVRM: GPU at PCI:0000:01:00: GPU-x Xid 13"], 20.0) is None
    why = stop_reason(ok, ["NVRM: ... Out of memory [NV_ERR_NO_MEMORY]"], 20.0)
    assert why and "NVRM" in why, why
    why = stop_reason({"min_available_gb": 19.9}, [], 20.0)
    assert why and "19.9" in why, why


def test_a_missing_length_is_the_head_of_the_next_longer_prompt():
    d = tempfile.mkdtemp(prefix="longctx-")
    man = {"prompts": {"8192": [{"domain": "prose"}, {"domain": "code"}],
                       "131072": [{"domain": "german"}, {"domain": "prose"}]}}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    np.save(os.path.join(d, "ids-8192.npy"), np.arange(2 * 8192, dtype=np.int32).reshape(2, 8192))
    big = np.arange(2 * 131072, dtype=np.int32).reshape(2, 131072) + 10**6
    np.save(os.path.join(d, "ids-131072.npy"), big)
    ids, src = load_prompt(d, 8192, "prose", man)
    assert ids == list(range(8192)) and src == "ids-8192.npy[0]", src
    ids, src = load_prompt(d, 65536, "prose", man)
    assert ids == big[1, :65536].tolist() and src == "ids-131072.npy[1][:65536]", src
    ids, src = load_prompt(d, 16384, "prose", man)
    assert len(ids) == 16384 and src == "ids-131072.npy[1][:16384]", src
    try:
        load_prompt(d, 200000, "prose", man)
    except SystemExit as e:
        assert "200000" in str(e)
    else:
        raise AssertionError("a length longer than every prompt must be refused")


def test_the_last_request_splits_into_tokens_and_milliseconds_a_block():
    from tools.longctx_probe import last_request
    log = ("[req] a stream prompt=40 completion=16 finish=length 900 ms 20.00 tok/s blocks=4 "
           "committed=15 decode_ms=300.0 accept=16:3x4\n"
           "[drafter] something\n"
           "[req] b stream prompt=32795 completion=256 finish=length 20000 ms 12.80 tok/s "
           "blocks=100 committed=255 decode_ms=12500.0 accept=16:1x100\n")
    r = last_request(log)
    assert r["cid"] == "b" and r["prompt"] == 32795 and r["blocks"] == 100, r
    assert abs(r["tok_blk"] - 2.55) < 1e-9 and abs(r["ms_blk"] - 125.0) < 1e-9, r
    assert "accept" not in r
    assert last_request("no requests here") is None


def test_a_longer_length_is_refused_on_its_projected_transient():
    from tools.longctx_probe import projected_stop
    # hold 2's caches-off 16k: peak 67.1 over a loaded 50.6, 44.6 GB free -> 64k projects ~-5 GB
    why = projected_stop({"min_free_gb": 44.6}, 67.1, 50.6, 16384, 65536, 20.0)
    assert why and "65536" in why, why
    # and to 32k: 44.6 - 16.5 = 28.1 GB, allowed
    assert projected_stop({"min_free_gb": 44.6}, 67.1, 50.6, 16384, 32768, 20.0) is None
    # the served path: a small transient clears 128k from 64k
    assert projected_stop({"min_free_gb": 48.0}, 62.0, 59.3, 65536, 131072, 20.0) is None



def test_a_repeat_says_whether_and_where_its_text_parts_from_the_first():
    runs = {"8192": {"text_sha256": "a", "text": "the cat sat on the mat"},
            "8192-r2": {"text_sha256": "b", "text": "the cat sat in the hat"},
            "8192-r3": {"text_sha256": "a", "text": "the cat sat on the mat"},
            "16384": {"text_sha256": "c"},
            "16384-r2": {"text_sha256": "d"},                  # hashes only: no position
            "32768-r2": {"text_sha256": "e"}}                  # its first request never ran
    rep = repeat_report(runs)
    assert rep == {"8192-r2": {"same": False, "first_diff_char": 12},
                   "8192-r3": {"same": True, "first_diff_char": None},
                   "16384-r2": {"same": False, "first_diff_char": None}}, rep
    prefix = repeat_report({"1": {"text_sha256": "a", "text": "abc"},
                            "1-r2": {"text_sha256": "b", "text": "abcd"}})
    assert prefix["1-r2"]["first_diff_char"] == 3


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
