"""`tools/ncu_budget.py report` on a hand-made ncu raw CSV: the classes, the shapes by the
bytes a kernel read, the split into workloads and blocks by the marker kernels, and the per-class
numbers (bytes ratio, GB/s, tail, the dominant stall)."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_kernel_classes_from_their_names():
    from tools.ncu_budget import kclass
    assert kclass("void skinny_kernel<2, 1, 16, 2, 1, 0, 0>(const __nv_bfloat16 *, ...)") == \
        "skinny nt2 mt1 wk16 pf2"
    assert kclass("void skinny_kernel<4, 2, 16, 0, 1, 0, 1>(const __nv_bfloat16 *, ...)") == \
        "skinny nt4 mt2 wk16 pf0 kr1"
    assert kclass("_gdn_tree_step") == "gdn tree walk"
    assert kclass("_head_gemm_fp8") == "head gemm fp8"
    assert kclass("_add_rms_norm") == "add+rmsnorm"
    assert kclass("_rms_norm") == "rmsnorm"
    assert kclass("void at::cuda::(anonymous namespace)::spin_kernel(long)") == "marker"
    assert kclass("something_new(int)").startswith("other: something_new")
    return "skinny by template (kr1 named), triton kernels, the marker, the rest named"


def test_a_projection_is_known_by_the_bytes_it_streams():
    from tools.ncu_budget import nvfp4_bytes, shape_of
    b = nvfp4_bytes(17408, 5120)
    assert shape_of(b * 1.02) == ("gate|up", b)
    assert shape_of(nvfp4_bytes(10240, 5120) * 0.98)[0] == "gdn qkv"
    assert shape_of(1e6) is None
    # down streams gate|up's bytes; the launch's output width tells them apart
    assert shape_of(b, n_est=5120)[0] == "down" and shape_of(b, n_est=17408)[0] == "gate|up"
    zb = nvfp4_bytes(6144, 5120)
    assert shape_of(zb, n_est=6144)[0] == "gdn z" and shape_of(zb, n_est=5120)[0] == "out|o_proj"
    return f"gate|up streams {b / 1e6:.1f} MB; +-6 % finds it; the grid splits gate|up/down, z/out"


HEAD = ['"ID"', '"Kernel Name"', '"gpu__time_duration.sum"', '"dram__bytes_read.sum"',
        '"sm__cycles_active.avg"', '"sm__cycles_active.max"',
        '"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"',
        '"smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"',
        '"smsp__average_warps_issue_stalled_selected_per_issue_active.ratio"']


def _csv(rows) -> str:
    from tools.ncu_budget import nvfp4_bytes  # noqa: F401
    lines = ["==PROF== Connected to process 1", ",".join(HEAD),
             ",".join(['""', '""', '"ns"', '"byte"', '"cycle"', '"cycle"', '""', '""', '""'])]
    for i, (name, dur, rd, act, amax, ls, wt) in enumerate(rows):
        lines.append(",".join(f'"{x}"' for x in (i, name, dur, rd, act, amax, ls, wt, 1.0)))
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n==PROF== Disconnected\n")
    return path


def test_the_report_splits_by_marker_and_prices_each_class():
    from tools.ncu_budget import nvfp4_bytes, read_csv, split_blocks, summarise
    gu = nvfp4_bytes(17408, 5120)
    sk = "void skinny_kernel<2, 1, 16, 2, 1, 0, 0>(...)"
    wide = "void skinny_kernel<4, 2, 16, 0, 1, 0, 0>(...)"
    mark = "void at::cuda::(anonymous namespace)::spin_kernel(long)"
    rows = [(mark, 1000, 0, 1, 1, 0, 0),
            (sk, 220_000, gu, 900, 1000, 6.0, 2.0), (sk, 230_000, gu * 1.01, 950, 1000, 6.0, 2.0),
            ("_gdn_tree_step", 60_000, 3e6, 500, 1000, 1.0, 9.0),
            (mark, 1000, 0, 1, 1, 0, 0),
            (wide, 250_000, gu * 1.04, 800, 1000, 3.0, 1.0)]
    path = _csv(rows)
    try:
        got = read_csv(path)
    finally:
        os.remove(path)
    assert len(got) == 6, got
    meta = {"workloads": [{"name": "prose", "profiled_blocks": 1},
                          {"name": "chat", "profiled_blocks": 1}]}
    blocks = split_blocks(got, meta)
    assert [(w, b, len(k)) for w, b, k in blocks] == [("prose", 0, 3), ("chat", 0, 1)]
    t = summarise(blocks)
    p = t[("prose", "skinny nt2 mt1 wk16 pf2", "gate|up")]
    assert p["per_block"] == 2
    assert abs(p["bytes_ratio"] - 1.005) < 1e-9
    assert abs(p["gbps"] - (gu * 1.005) / 225_000) < 1e-9          # bytes / ns = GB/s
    assert abs(p["tail"] - (1 - 925 / 1000)) < 1e-9
    assert p["stall_top"][0][1] == "long_scoreboard" and abs(p["stall_top"][0][0] - 0.75) < 1e-9
    g = t[("prose", "gdn tree walk", "")]
    assert g["stall_top"][0][1] == "wait" and g["tail"] == 0.5
    c = t[("chat", "skinny nt4 mt2 wk16 pf0", "gate|up")]
    assert abs(c["bytes_ratio"] - 1.04) < 1e-9
    return "2 blocks by marker; ratio, GB/s, tail, stall shares ('selected' left out of the shares)"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:60s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
