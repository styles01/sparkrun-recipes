"""The decisive row's arithmetic, without a board.

`tools/row3.py` exists because a single row decided nothing in phase 9, and the thing it adds on
top of the row is arithmetic: one rate formula per request, a median across runs, a spread beside
the median, and a verdict that refuses to call a difference smaller than its own noise a result.
All four are wrong in ways that would be invisible on the board -- a warm-up counted as a request
moves the mean by a few per cent and looks like a change, and a verdict that reads `>=` where it
means `>` calls a tie a win. So they are tested here, on records made by hand.
"""

from __future__ import annotations

import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.row3 import MemGuard, across, mem_available_gb, row_stats, server_cmd  # noqa: E402


def _req(n, ttft_ms, decode_ms, warmup=False, status="ok"):
    return {"warmup": warmup, "status": status, "completion_tokens": n,
            "ttft_ms": ttft_ms, "e2e_ms": ttft_ms + decode_ms}


def _record(reqs, duration_s=100.0):
    return {"raw": {"payload": {"requests": reqs, "selected_iteration": 0,
                                "iterations": [{"duration_s": duration_s}]}}}


def test_formula_subtracts_the_prefill_token():
    """257 tokens in 2.56 s of decode is 100 tok/s only if the first token is not decode's."""
    st = row_stats(_record([_req(257, 800.0, 2560.0)]))
    assert abs(st["mean"] - 100.0) < 1e-9, st["mean"]
    assert st["ttft_p50_ms"] == 800.0


def test_warmups_and_failures_are_not_the_row():
    """Three warm-ups and a failure are in every atlas record; none of them is a measurement."""
    reqs = [_req(101, 900.0, 10_000.0, warmup=True),                # 10 tok/s
            _req(101, 900.0, 10_000.0, warmup=True),
            _req(101, 900.0, 10_000.0, status="error"),
            _req(101, 500.0, 1000.0),                               # 100 tok/s
            _req(101, 500.0, 2000.0)]                               # 50 tok/s
    st = row_stats(_record(reqs))
    assert st["n"] == 2, st
    assert abs(st["mean"] - 75.0) < 1e-9
    assert abs(st["p50"] - 75.0) < 1e-9
    assert abs(st["max"] - 100.0) < 1e-9


def test_a_request_with_one_token_is_not_a_rate():
    """`(n - 1) / decode` is zero over zero for a single-token answer; it is dropped, not counted."""
    st = row_stats(_record([_req(1, 700.0, 0.0), _req(11, 700.0, 1000.0)]))
    assert st["n"] == 1 and abs(st["mean"] - 10.0) < 1e-9


def test_percentiles_are_taken_on_the_sorted_rates():
    rates = [_req(n + 1, 600.0, 1000.0) for n in range(1, 11)]       # 1..10 tok/s
    st = row_stats(_record(rates))
    assert abs(st["p50"] - 5.5) < 1e-9
    assert abs(st["p90"] - 9.0) < 1e-9, st["p90"]
    assert abs(st["max"] - 10.0) < 1e-9


def test_median_and_spread_across_runs():
    """Phase 9's own three RC rows: the median is the middle one and the spread is what the first
    row would have been believed to have proved."""
    runs = [{"mean": 53.74}, {"mean": 59.06}, {"mean": 59.30}]
    got = across(runs, "mean")
    assert got["median"] == 59.06
    assert got["min"] == 53.74 and got["max"] == 59.30
    assert abs(got["spread_pct"] - 100.0 * (59.30 - 53.74) / 59.06) < 1e-9
    assert got["spread_pct"] > 9.0                                  # the 10 % that killed the A/B


def test_spread_of_one_run_is_zero_and_says_nothing():
    """One run has no spread, which is exactly the phase-9 mistake stated as arithmetic: the
    verdict function must not be allowed to read 0 % noise off a single row."""
    got = across([{"mean": 59.06}], "mean")
    assert got["spread_pct"] == 0.0 and got["median"] == 59.06


def test_a_missing_statistic_is_absent_rather_than_zero():
    assert across([{"mean": None}], "mean") == {}
    assert across([], "mean") == {}


def _verdict(b, o):
    """The verdict as `compare` computes it, over two lists of per-run values."""
    B, O = across([{"x": v} for v in b], "x"), across([{"x": v} for v in o], "x")
    delta = 100.0 * (O["median"] - B["median"]) / B["median"]
    noise = max(B["spread_pct"], O["spread_pct"])
    disjoint = O["min"] > B["max"] or O["max"] < B["min"]
    return "RESOLVED" if abs(delta) > noise and disjoint else "not resolved"


def test_the_verdict_is_strict():
    """The 0.2 % tree-mode difference against a 10 % spread must never read as a result, and
    neither may a difference the exact size of the noise."""
    assert _verdict([53.74, 59.06, 59.30], [53.80, 59.20, 59.40]) == "not resolved"
    assert _verdict([100.0, 100.0, 100.0], [110.0, 110.0, 110.0]) == "RESOLVED"
    # big delta, overlapping runs: one configuration did NOT beat the other every time
    assert _verdict([100.0, 100.0, 160.0], [110.0, 150.0, 150.0]) == "not resolved"
    # disjoint but only by less than the noise the runs themselves carry
    assert _verdict([100.0, 130.0, 140.0], [141.0, 150.0, 160.0]) == "not resolved"


def test_wall_comes_from_the_selected_iteration():
    rec = _record([_req(11, 600.0, 1000.0)], duration_s=261.9)
    rec["raw"]["payload"]["iterations"] = [{"duration_s": 999.0}, {"duration_s": 261.9}]
    rec["raw"]["payload"]["selected_iteration"] = 1
    assert row_stats(rec)["wall_s"] == 261.9


def _ns(**kw):
    from pathlib import Path
    from types import SimpleNamespace
    base = dict(python=Path("py"), port=8001, max_len=4096, len_fixed=0, budget=16,
                repo=Path("/r"), nvfp4="nv", head="hd", len_latch=True, server_arg=[])
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_row_never_reads_the_persistent_suffix_store():
    """Every server on the board opens the same store, it keeps prompts and answers, and the row's
    prompts are fixed: left on, the row reads its own previous answers (2026-09-23 10:37)."""
    cmd = server_cmd(_ns())
    assert "--suffix-store=" in cmd
    assert "--no-session-cache" in cmd and "--no-prefix-cache" in cmd


def test_a_signalled_tool_stops_its_server_through_finally():
    """2026-09-25 03:44: hold.sh signalled a probe's process group; the probe died without its
    `finally: stop_server`, and its :8011 engine (its own session) outlived the hold. After
    `exit_on_term()` SIGTERM and SIGHUP run the `finally` blocks."""
    import subprocess
    import tempfile
    import time as _t
    for sig in ("SIGTERM", "SIGHUP"):
        mark = tempfile.mktemp(prefix="row3-term-")
        code = ("import sys, time; sys.path.insert(0, %r)\n"
                "from tools.row3 import exit_on_term\n"
                "exit_on_term()\n"
                "try:\n    print('up', flush=True); time.sleep(60)\n"
                "finally:\n    open(%r, 'w').write('stopped')\n") % (ROOT, mark)
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        assert p.stdout.readline().strip() == "up"
        import signal as _s
        p.send_signal(getattr(_s, sig))
        rc = p.wait(timeout=30)
        assert os.path.exists(mark) and open(mark).read() == "stopped", sig
        assert rc == 128 + getattr(_s, sig), (sig, rc)


def test_a_probe_can_run_the_caches_the_service_runs():
    """Wants the memory curve in the configuration :8000 serves -- prefix cache on, its
    1,024-row prefill chunks, the state cache's budget -- beside the row's caches-off one. The
    row itself never gets it: it is an attribute only the long-context probe sets."""
    cmd = server_cmd(_ns(served_caches=True, cache_gb=8.0))
    assert "--no-session-cache" not in cmd and "--no-prefix-cache" not in cmd, cmd
    assert cmd[cmd.index("--cache-budget-gb") + 1] == "8.0", cmd
    assert "--suffix-store=" in cmd, "the store stays off: the probe measures the engine"
    cmd = server_cmd(_ns())
    assert cmd[cmd.index("--cache-budget-gb") + 1] == "0", cmd


def test_the_store_can_be_put_back_to_reproduce_an_old_row():
    assert "--suffix-store=" not in server_cmd(_ns(with_suffix_store=True))
    # and a --server-arg comes after it, so an explicit store path still wins
    cmd = server_cmd(_ns(server_arg=["--suffix-store=/tmp/s"]))
    assert cmd.index("--suffix-store=/tmp/s") > cmd.index("--suffix-store=")


def test_the_three_store_views():
    assert "--suffix-store-readonly" in server_cmd(_ns(store="live"))
    assert "--suffix-store=" not in server_cmd(_ns(store="live"))
    cmd = server_cmd(_ns(store="clean", clean_store="/c"))
    assert "--suffix-store=/c" in cmd and "--suffix-store-readonly" in cmd
    assert "--suffix-store=" in server_cmd(_ns(store="off"))


def test_mem_available_is_read_in_gb():
    info = "MemTotal:  127000000 kB\nMemFree: 1000 kB\nMemAvailable:   20000000 kB\n"
    assert abs(mem_available_gb(info) - 20.48) < 0.01


class _Proc:
    pid = 999999

    def __init__(self):
        self.alive = True

    def poll(self):
        return None if self.alive else 0


def test_the_memory_guard_trips_below_the_floor_and_not_above():
    """The server is killed at the floor, before the board wedges; above it, never."""
    killed = []
    import tools.row3 as R
    keep = (R.os.killpg, R.os.getpgid)
    R.os.killpg = lambda pg, sig: killed.append((pg, sig))
    R.os.getpgid = lambda pid: pid
    try:
        g = MemGuard(_Proc(), floor_gb=10.0, read=lambda: 40.0)
        assert g.check() is False and not killed
        g.read = lambda: 9.5
        assert g.check() is True and killed == [(999999, R.signal.SIGKILL)] and g.tripped == 9.5
        dead = _Proc()
        dead.alive = False
        assert MemGuard(dead, read=lambda: 1.0).check() is False     # nothing to kill
    finally:
        R.os.killpg, R.os.getpgid = keep


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


def test_the_report_records_every_knob_of_the_run():
    """Rule 5: a report must carry every limit knob, defaults included. The policy flags were
    row3's own arguments and never reached the report, so a fixed-policy row read as a latch row."""
    import argparse
    from pathlib import Path
    from tools.row3 import recorded_args, server_cmd
    a = argparse.Namespace(compare=None, label="x", runs=3, port=8001, env=[],
                           server_arg=["--drop-idle"], restart_each=False, no_server=False, repo=Path("/r"), python=Path("/py"),
                           pythonpath=Path("/pp"), atlas=Path("/a"), spec="s.json", login="l",
                           tokenizer="t", max_len=262144, budget=16, len_fixed=16, len_latch=False,
                           nvfp4="n", head="h", start_timeout=600, out_dir=Path("/o"))
    args = recorded_args(a)
    for k in ("max_len", "budget", "len_fixed", "len_latch", "server_arg", "port", "spec"):
        assert k in args, k
    assert args["len_fixed"] == 16 and args["len_latch"] is False and "compare" not in args
    cmd = server_cmd(a)
    assert cmd[cmd.index("--len-fixed") + 1] == "16" and "--no-len-latch" in cmd
    assert cmd[cmd.index("--max-len") + 1] == "262144" and cmd[-1] == "--drop-idle"


# ---------------------------------------------------------------- both factors of the speed

LOG_2REQ = (
    "[drafter] lenrouter ... commits - cap arm 0 depth 0\n"
    "[req] w1 json prompt=256 completion=256 finish=length 9000 ms 28.4 tok/s blocks=100 "
    "committed=255 decode_ms=8500.0 accept=15:0x60,2x40\n"
    "[req] r1 json prompt=256 completion=256 finish=length 7000 ms 36.6 tok/s blocks=80 "
    "committed=255 decode_ms=8000.0 accept=15:0x40,1x20,15x20\n"
    "[req] r2 json prompt=256 completion=101 finish=stop 3100 ms 33.3 tok/s blocks=25 "
    "committed=100 decode_ms=2500.0 accept=7:3x20|15:0x4\n")


def test_the_req_line_parser_reads_the_block_count_and_leaves_old_lines_empty():
    from tools.rowlog import parse_requests
    text = LOG_2REQ + ("[req] old json prompt=5 completion=9 finish=stop 90 ms 88.9 tok/s\n"
                       "[req] pf json prompt=5 completion=1 finish=stop 90 ms 0.00 tok/s blocks=0 "
                       "committed=0 decode_ms=0.0 accept=-\n"
                       "[req] e json prompt=5 completion=3 finish=error 90 ms 22.2 tok/s blocks=2 "
                       "committed=2 decode_ms=40.0 accept=- !! RuntimeError: boom\n")
    reqs = parse_requests(text)
    assert [r["cid"] for r in reqs] == ["w1", "r1", "r2", "old", "pf", "e"]
    assert reqs[1]["blocks"] == 80 and reqs[1]["committed"] == 255
    assert reqs[1]["accept"] == {15: {0: 40, 1: 20, 15: 20}}
    assert reqs[3]["blocks"] is None, "an old server's line is empty, not zero"
    assert reqs[4]["blocks"] is None, "a request that ended in its prefill has no block"
    assert reqs[5]["blocks"] == 2 and reqs[5]["accept"] == {}, "the exception tail is not a field"


def test_the_run_takes_its_factors_from_its_own_requests_only():
    """The warm-up's line is in the log and in the record; it is not the row's."""
    from tools.row3 import block_factors
    record = _record([_req(256, 500.0, 9000.0, warmup=True), _req(256, 500.0, 7000.0),
                      _req(101, 500.0, 3000.0)])
    f = block_factors(record, LOG_2REQ)
    assert f["requests"] == 2 and f["blocks"] == 105
    assert abs(f["tok_blk"] - 355 / 105) < 1e-12
    assert abs(f["ms_blk"] - 10500.0 / 105) < 1e-12
    assert abs(f["tok_blk_p50"] - 0.5 * (255 / 80 + 100 / 25)) < 1e-12
    assert f["accept"] == {15: {0: 44, 1: 20, 15: 20}, 7: {3: 20}}
    assert f["matched"] == "3 lines / 3 requests"


def test_a_log_that_does_not_match_the_record_drops_the_warm_ups_from_the_front():
    from tools.row3 import block_factors
    record = _record([_req(256, 500.0, 9000.0, warmup=True), _req(256, 500.0, 7000.0)])
    f = block_factors(record, LOG_2REQ)                  # 3 lines, 2 requests
    assert f["requests"] == 2 and f["matched"] == "3 lines / 2 requests"


def test_a_server_without_the_count_gives_no_factors():
    from tools.row3 import block_factors
    record = _record([_req(256, 500.0, 7000.0)])
    assert block_factors(record, "[req] a json prompt=1 completion=256 finish=length 9 ms 1 tok/s\n") == {}


def _report(label, stats):
    summ = {}
    for k, (med, lo, hi) in stats.items():
        summ[k] = {"median": med, "min": lo, "max": hi, "spread_pct": 100.0 * (hi - lo) / med}
    return {"label": label, "runs": [{}, {}, {}], "summary": summ}


def test_compare_resolves_tokens_a_block_and_ms_a_block_with_their_direction():
    from tools.row3 import verdicts
    base = _report("b", {"mean": (34.0, 33.8, 34.2), "tok_blk": (3.30, 3.28, 3.32),
                         "ms_blk": (100.0, 99.5, 100.5)})
    other = _report("o", {"mean": (34.1, 33.8, 34.3), "tok_blk": (3.00, 2.98, 3.02),
                          "ms_blk": (90.0, 89.5, 90.5)})
    v = {x["stat"]: x for x in verdicts(base, other)}
    assert v["tok_blk"]["verdict"] == "RESOLVED WORSE" and v["tok_blk"]["worse"]
    assert v["ms_blk"]["verdict"] == "RESOLVED better", "a shorter block is better"
    assert v["mean"]["verdict"] == "not resolved"


def test_old_reports_still_compare_and_the_new_columns_read_na():
    from tools.row3 import verdicts
    base = _report("old", {"mean": (34.0, 33.8, 34.2)})
    other = _report("new", {"mean": (38.0, 37.8, 38.2), "tok_blk": (3.3, 3.3, 3.3)})
    v = {x["stat"]: x for x in verdicts(base, other)}
    assert v["mean"]["verdict"] == "RESOLVED better"
    assert v["tok_blk"]["verdict"] == "n/a" and v["p50"]["verdict"] == "n/a"


def test_the_gate_rule_fails_a_resolved_worse_factor_and_adopt_needs_the_mean():
    from tools.gatecheck import check
    base = _report("b", {"mean": (34.0, 33.8, 34.2), "tok_blk": (3.3, 3.28, 3.32)})
    worse = _report("w", {"mean": (38.0, 37.8, 38.2), "tok_blk": (3.0, 2.98, 3.02)})
    tie = _report("t", {"mean": (34.1, 33.9, 34.3), "tok_blk": (3.3, 3.28, 3.32)})
    better = _report("x", {"mean": (38.0, 37.8, 38.2), "tok_blk": (3.3, 3.28, 3.32)})
    assert check(base, worse, "noworse")[0] == 1
    assert check(base, tie, "noworse")[0] == 0 and check(base, tie, "adopt")[0] == 2
    assert check(base, better, "adopt")[0] == 0


def test_the_item_rule_reads_the_block_or_the_tokens_and_still_refuses_anything_worse():
    # the rule ratified 2026-09-24 for one item stacked on a phase: its own factor resolved better,
    # nothing resolved worse; the mean is left to the phase's set gate
    from tools.gatecheck import check
    base = _report("b", {"mean": (36.7, 36.0, 37.0), "p90": (49.8, 49.7, 49.9),
                         "ms_blk": (92.7, 92.6, 92.8), "tok_blk": (3.32, 3.26, 3.34)})
    faster = _report("f", {"mean": (37.0, 36.2, 37.3), "p90": (49.8, 49.6, 50.0),
                           "ms_blk": (91.5, 91.4, 91.6), "tok_blk": (3.32, 3.28, 3.35)})
    faster_p90 = _report("g", {"mean": (37.0, 36.2, 37.3), "p90": (45.5, 45.3, 45.7),
                               "ms_blk": (91.5, 91.4, 91.6), "tok_blk": (3.32, 3.28, 3.35)})
    wider = _report("w", {"mean": (37.5, 36.9, 38.0), "p90": (49.8, 49.6, 50.0),
                          "ms_blk": (92.8, 92.7, 93.0), "tok_blk": (3.60, 3.58, 3.62)})
    assert check(base, faster, "block")[0] == 0
    assert check(base, faster, "adopt")[0] == 2, "the mean alone does not resolve"
    assert check(base, faster_p90, "block")[0] == 1, "p90 resolved worse still fails"
    assert check(base, wider, "block")[0] == 2, "a slower block is not a block item"
    assert check(base, wider, "tokens")[0] == 0
    assert check(base, faster, "tokens")[0] == 2
    # an acceptance item pays for its tokens with a longer block: that is its price under `tokens`,
    # a regression under every other mode; a slower mean is a regression under all of them
    dearer = _report("d", {"mean": (37.5, 36.9, 38.0), "p90": (49.8, 49.6, 50.0),
                           "ms_blk": (98.0, 97.9, 98.1), "tok_blk": (3.60, 3.58, 3.62)})
    assert check(base, dearer, "tokens")[0] == 0
    assert check(base, dearer, "block")[0] == 1 and check(base, dearer, "noworse")[0] == 1
    slower = _report("s", {"mean": (33.0, 32.8, 33.2), "p90": (49.8, 49.6, 50.0),
                           "ms_blk": (98.0, 97.9, 98.1), "tok_blk": (3.60, 3.58, 3.62)})
    assert check(base, slower, "tokens")[0] == 1
    rc, lines = check(base, faster, "block")
    assert lines[-1].endswith("ms_blk resolved better"), lines[-1]


# ---------------------------------------------------------------- the code a report measured

def test_the_code_hash_names_the_tree_and_moves_with_any_file():
    import tempfile
    from pathlib import Path
    from tools.row3 import code_hash, git_head
    def tree(root):
        for d in ("engine", "server", "tools", "ops"):
            (root / d).mkdir(parents=True)
            (root / d / "a.py").write_text(f"# {d}\n")
        (root / "tools" / "__pycache__").mkdir()
        (root / "tools" / "__pycache__" / "a.cpython.pyc").write_bytes(b"x")
        (root / "notes.md").write_text("not code")
    a, b = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    tree(a); tree(b)
    assert code_hash(a) == code_hash(b), "the same files in two directories are the same code"
    (b / "tools" / "__pycache__" / "a.cpython.pyc").write_bytes(b"y")
    (b / "notes.md").write_text("still not code")
    assert code_hash(a) == code_hash(b), "caches and notes are not code"
    (b / "ops" / "gate.sh").write_text("#!/bin/bash\n")
    assert code_hash(a) != code_hash(b), "a new script is different code"
    assert git_head(a) is None, "an rsync copy has no git head"


if __name__ == "__main__":
    sys.exit(_main())
