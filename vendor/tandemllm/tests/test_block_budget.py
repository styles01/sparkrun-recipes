"""`tools/block_budget.py` attributes each kernel to the part that LAUNCHED it.

The device runs behind the host, so a kernel's own interval routinely sits under an annotation that
opened after the one it belongs to. The tool therefore reads the launch call's timestamp through the
correlation id and takes the innermost annotation open at that moment. These tests build a chrome
trace by hand in which the execution-time answer and the launch-time answer differ, and check the
tool gives the launch-time one; and that the table folds a chain verify's `chain/` prefix onto the
same parts a tree verify reports, with the snapshot the chain takes as its own row.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ann(name, ts, dur):
    return {"ph": "X", "cat": "user_annotation", "name": name, "ts": ts, "dur": dur}


def _launch(corr, ts, cat="cuda_runtime"):
    return {"ph": "X", "cat": cat, "name": "cudaLaunchKernel", "ts": ts, "dur": 1,
            "args": {"correlation": corr}}


def _kernel(corr, ts, dur):
    return {"ph": "X", "cat": "kernel", "name": f"k{corr}", "ts": ts, "dur": dur,
            "args": {"correlation": corr}}


def _trace(events) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"traceEvents": events}, f)
    return path


def test_a_kernel_belongs_to_the_part_that_launched_it():
    from tools.block_budget import attribute
    ev = [
        _ann("PH::verify", 0, 1000),
        _ann("C::mlp", 10, 100),
        _ann("C::mlp/gate|up", 20, 10), _launch(1, 25),
        _ann("C::mlp/down", 40, 10), _launch(2, 45),
        _launch(3, 60),                                   # inside mlp, outside both projections
        _ann("C::norm", 200, 10), _launch(4, 205),
        # the kernels run later than they were launched: 1 and 2 execute while `norm` is open
        _kernel(1, 201, 300), _kernel(2, 502, 250), _kernel(3, 753, 5), _kernel(4, 760, 7),
        _ann("PH::sync", 1000, 100), _launch(5, 1010, cat="cuda_driver"),
        _ann("C::fc", 1020, 10), _launch(6, 1025),
        _kernel(5, 1011, 4), _kernel(6, 1030, 6),
    ]
    path = _trace(ev)
    try:
        per, busy, span = attribute(path)
    finally:
        os.remove(path)
    assert per[("verify", "mlp/gate|up")] == [300, 1], per
    assert per[("verify", "mlp/down")] == [250, 1], per
    assert per[("verify", "mlp")] == [5, 1], per
    assert per[("verify", "norm")] == [7, 1], per
    assert per[("sync", "(glue)")] == [4, 1], per          # a driver launch counts as a launch
    assert per[("sync", "fc")] == [6, 1], per
    assert busy == {"verify": 562, "sync": 10}, busy
    return "launch-time attribution, nested parts, runtime and driver launches"


def test_a_kernel_without_a_launch_record_is_counted_not_guessed():
    from tools.block_budget import attribute
    ev = [_ann("PH::verify", 0, 100), _ann("C::norm", 10, 10), _launch(1, 12),
          _kernel(1, 20, 3), _kernel(2, 30, 9)]
    path = _trace(ev)
    try:
        per, busy, _ = attribute(path)
    finally:
        os.remove(path)
    assert per == {("verify", "norm"): [3, 1]}, per
    assert busy == {"verify": 3}, busy
    return "an unmatched kernel is left out of every row"


def test_the_table_folds_a_chain_onto_the_tree_parts():
    from tools.block_budget import table
    r = dict(label="prose fixed16", blocks=10, tokens=30, tok_s=25.0, accepted=3.0, nodes=15.0,
             block_ms=120.0, traced_block_ms=125.0,
             parts={"verify|chain/mlp/gate|up": [30.0, 128],
                    "verify|mlp/gate|up": [10.0, 128],
                    "verify|chain": [1.5, 2],
                    "?|mlp/gate|up": [99.0, 128]},
             bytes={"verify|chain/mlp/gate|up": 6.4e9, "verify|mlp/gate|up": 0.0,
                    "verify|chain": 0.3e9},
             wall={"verify": 100.0, "lattice": 14.0}, busy={"verify": 41.5, "draft": 12.0})
    out = table(r)
    rows = {line.split()[0]: line.split() for line in out.splitlines()
            if line.startswith("verify:")}
    assert set(rows) == {"verify:mlp/gate|up", "verify:snapshot"}, rows
    gate = rows["verify:mlp/gate|up"]
    assert float(gate[2]) == 40.0 and float(gate[3]) == 6.4, gate     # both paths, the bytes of one
    assert abs(float(gate[4]) - 160.0) < 0.1, gate                     # 6.4 GB in 40 ms
    assert "99.000" not in out, "the prefill's kernels leaked into the block"
    assert "host/idle   58.500" in out, out                            # 100 - 41.5
    return "chain/ folded, snapshot named, prefill excluded, host gap = wall - kernels"


def test_gaps_split_the_idle_by_size_and_by_the_phase_that_ended_it():
    from tools.block_budget import gaps
    ev = [
        _ann("PH::verify", 0, 500),
        _launch(1, 1), _kernel(1, 10, 100),                  # busy 10..110
        _launch(2, 2), _kernel(2, 113, 50),                  # gap 3 us: the device's own step
        _launch(3, 200), _kernel(3, 203, 40),                # gap 40 us: the host was behind
        _ann("PH::picks", 500, 100),
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH (Device -> Pageable)", "ts": 243,
         "dur": 2, "args": {"correlation": 4}}, _launch(4, 243),
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": 246, "dur": 3,
         "args": {}},
        _ann("PH::commit", 600, 300),
        _launch(5, 610), _kernel(5, 620, 10),                # gap 375 us, launched in commit
    ]
    path = _trace(ev)
    try:
        g = gaps(path, blocks=1)
    finally:
        os.remove(path)
    assert g["kernels"] == 4 and g["dtoh"] == 1 and g["htod"] == 0 and g["sync_calls"] == 1, g
    assert g["gap_count"] == {"<=5us": 1, "5-50us": 1, ">50us": 1}, g
    assert abs(g["gap_ms"][">50us"] - 0.375) < 1e-9, g
    assert abs(g["gap_ms_by_phase"]["commit"] - 0.375) < 1e-9, g
    assert abs(g["gap_ms_by_phase"]["verify"] - 0.043) < 1e-9, g
    return "gap sizes bucketed, charged to the launching phase, copies and syncs counted"



def test_ab_states_take_extra_combinations_once_and_refuse_unknown_names():
    from tools.block_budget import ab_states
    st = ab_states(["G", "A", "W"], "G+A,G+A+W,A")
    assert st[:5] == [(False, False, False), (True, False, False), (False, True, False),
                      (False, False, True), (True, True, True)]
    assert st[5:] == [(True, True, False)], "all-on and A alone are already there"
    try:
        ab_states(["G"], "G+X")
    except SystemExit as e:
        assert "X" in str(e)
    else:
        raise AssertionError("an unknown name must be refused")

def test_ab_assign_sets_typed_values_and_restores_the_module_s_own():
    """One --ab flag can move several module values (the WY thresholds and slices) together."""
    import types
    from tools.block_budget import ab_assign
    mod = types.SimpleNamespace(WY_KC=0, WY_MAXT=16, WY=True)
    assert ab_assign(mod, "WY") is None
    on, off = ab_assign(mod, "WY_KC=32;WY_MAXT=32")
    assert on == {"WY_KC": 32, "WY_MAXT": 32} and off == {"WY_KC": 0, "WY_MAXT": 16}
    assert all(type(v) is int for v in on.values())
    try:
        ab_assign(mod, "NOPE=1")
    except AttributeError:
        pass
    else:
        raise AssertionError("an attribute the module does not have must be refused")
    return "typed values on, the module's own off, unknown names refused"


def test_first_divergence_names_the_first_token_and_a_short_run():
    from tools.block_budget import first_divergence
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 5, 3]) == 1
    assert first_divergence([1, 2, 3], [1, 2]) == 2
    return "identical -> None; a changed token -> its index; a shorter run -> where it stopped"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
