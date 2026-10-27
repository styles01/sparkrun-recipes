"""SRV-335: page cache the engine keeps on purpose counts against every floor check, no GPU.

Flash-Next's warm n-gram table sits in the page cache, which MemAvailable reports as free; a role or a
request admitted on that number would spend the table. `server/memsplit.py` counts the engine's
`resident_files()` (warm ones) as pinned, and the floor checks read MemAvailable less it:
  * `counted_available` / `counted_meminfo` subtract it (and read it at most once a second);
  * the residency planner's `_mem_gib`, the side slots' `Admission` and the dashboard's split use it;
  * an engine with no such file (the 27B) changes nothing.
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import memsplit, slots  # noqa: E402
from server.conductor import residency  # noqa: E402

GIB = 1 << 30


class _Eng:
    def __init__(self, files):
        self.files = files

    def resident_files(self):
        return self.files

    def memory_parts(self):
        return {"weights": 48 * GIB}


def _with(avail_gib, eng, fn):
    saved = (memsplit.mem_available, slots.read_meminfo, memsplit.PINNED)
    memsplit.mem_available = lambda: int(avail_gib * GIB)
    slots.read_meminfo = lambda path="/proc/meminfo": {"MemAvailable": int(avail_gib * GIB), "MemTotal": 121 * GIB}
    memsplit.PINNED = (lambda: memsplit.pinned_of(eng)) if eng is not None else None
    memsplit._pinned_cache[:] = [0.0, 0]
    try:
        return fn()
    finally:
        memsplit.mem_available, slots.read_meminfo, memsplit.PINNED = saved
        memsplit._pinned_cache[:] = [0.0, 0]


def test_the_warm_table_is_counted_off():
    table = {"name": "ngram_table", "policy": "warm", "resident_bytes": 30 * GIB}
    cold = {"name": "ngram_table", "policy": "cold", "resident_bytes": 0, "cache_bytes": 2 * GIB}
    assert _with(68, _Eng([table]), memsplit.counted_available) == 38 * GIB
    assert _with(68, _Eng([table]), lambda: residency._mem_gib()) == 38.0
    assert _with(68, _Eng([table]), memsplit.counted_meminfo)["MemAvailable"] == 38 * GIB
    assert _with(68, _Eng([cold]), memsplit.counted_available) == 68 * GIB          # cold: nothing pinned
    assert _with(68, _Eng([]), memsplit.counted_available) == 68 * GIB              # the 27B
    assert _with(68, None, memsplit.counted_available) == 68 * GIB                  # no engine yet
    return "68 GiB free with 30 GiB of warm table counts as 38; cold and none count nothing"


def test_admission_and_the_planner_refuse_what_would_spend_the_table():
    table = {"name": "ngram_table", "policy": "warm", "resident_bytes": 30 * GIB}

    def check():
        adm = slots.Admission(15 * GIB, meminfo=memsplit.counted_meminfo, growth=lambda: 0)
        ok, d = adm.check(10 * GIB, count=False)
        return ok
    assert _with(52, _Eng([]), check) is True                    # 52 - 10 >= 15
    assert _with(52, _Eng([table]), check) is False              # 52 - 30 - 10 < 15
    return "a 10 GiB request on 52 GiB free: admitted without the table, refused with 30 GiB of it"


def test_the_split_reports_the_pinned_cache():
    table = {"name": "ngram_table", "policy": "warm", "resident_bytes": 30 * GIB}

    def split():
        return memsplit.split({"engine": _Eng([table])}, probe=lambda: (50 * GIB, 51 * GIB), avail=lambda: 68 * GIB,
                              total=lambda: 121 * GIB, pool=None)
    out = _with(68, _Eng([table]), split)
    assert out["pinned_cache"] == 30 * GIB and out["parts"]["weights"] == 48 * GIB
    return "pinned_cache 30 GiB beside the engine's parts"


def test_a_max_len_above_the_profiles_window_is_refused():
    """SRV-335 (4): the start refuses a --max-len above the profile's max_window, naming both numbers;
    6.05 (null) fits none; the 27B's profiles and unknown names are never refused."""
    from server.setup import registry
    models, errors = registry.load_all()
    assert errors == [], errors
    assert registry.window_refusal("flashnext-3.05", 262_144, models) is None
    why = registry.window_refusal("flashnext-3.05", 262_145, models)
    assert "262,145" in why and "262,144" in why and "flashnext-3.05" in why, why
    assert "fits no window" in registry.window_refusal("flashnext-6.05", 8_192, models)
    assert registry.window_refusal("exl3-4.5", 1 << 20, models) is None
    assert registry.window_refusal("no-such-profile", 1 << 20, models) is None
    return why


def test_pinned_is_read_once_a_second():
    calls = []
    eng = _Eng([{"policy": "warm", "resident_bytes": GIB}])
    orig = eng.resident_files
    eng.resident_files = lambda: (calls.append(1), orig())[1]

    def twice():
        memsplit.counted_available()
        return memsplit.counted_available()
    _with(10, eng, twice)
    assert len(calls) == 1
    return "two floor checks in a second read the residency once"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<66} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<66} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
