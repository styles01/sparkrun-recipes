"""`ops/gate.sh`, the standing protocol in one command, run for real against fake tools.

On 2026-09-23 three agents re-typed the protocol and it drifted three ways: a row that exited in
15 s, rows cut by a pattern kill, and a gate loop that read PASS off the wrong line. The script is
the protocol, so what it must get right is tested here, with nothing real behind it: the script and
the real `tools/row3.py`, `tools/rowlog.py`, `tools/gatecheck.py` are copied into a temporary tree
beside a `serve.env` whose python is a fake that answers each tool the way the tool answers, logs
every call, and writes row reports whose numbers the test chooses.

  * the steps run in the protocol's order: suite, GPU batteries, identity, lossless gate, rows,
    compare -- and a failing step stops the script with a non-zero exit before the next one;
  * the exit code is the ship rule: `adopt` needs the mean resolved better against every base and
    nothing resolved worse; a resolved-worse statistic fails either mode;
  * the report under results/gate/<label>/ is dated and holds every command; nothing is written
    outside results/;
  * a missing base report is refused before anything runs;
  * nothing on the gate's own command line matches the engine search of stop.sh / hold.sh.

Run: python tests/test_gate_script.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE_PY = r'''#!{python}
"""A python that is every tool the gate calls."""
import json, os, subprocess, sys
args = sys.argv[1:]
# the suite runs under `env -i`, so the call log sits beside this file, not in the environment
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calls.log"), "a") as fh:
    fh.write(" ".join(args) + "\n")
if args and args[0] == "-u":
    args = args[1:]
if args[:1] == ["-c"]:
    print("0" * 64); sys.exit(0)
if args[:1] == ["-"] or (args and args[0].endswith("gatecheck.py")):
    sys.exit(subprocess.call([{python!r}] + args))
tool = os.path.basename(args[0])
if tool == "script_suite.py":
    # SRV-661: the suite's runner, each file a call of its own, the count line at the end
    import glob
    files = sorted(glob.glob("tests/test_*.py"))
    bad = [f for f in files if "FAIL" in open(f).read()]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calls.log"), "a") as fh:
        fh.writelines(f + "\n" for f in files)
    for f in files:
        print(f, "1 tests", "FAIL (exit 1)" if f in bad else "ok")
    print(f"files={{len(files)}} ok={{len(files) - len(bad)}} fail={{len(bad)}} tests={{len(files)}}"
          + (" -- " + " ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)
if tool.startswith("test_"):
    sys.exit(1 if "FAIL" in open(args[0]).read() else 0)
if tool.startswith("flagoff_identity"):
    if "--compare" in args:
        print("IDENTITY PASS"); sys.exit(0)
    print("dumped"); sys.exit(0)
if tool == "verify_spec.py":
    print("GATE speculation changes greedy output only at one-ulp ties:")
    print("PASS" if os.environ.get("GATE_LOSSLESS", "PASS") == "PASS" else "FAIL"); sys.exit(0)
if tool == "row3.py":
    label = args[args.index("--label") + 1]
    mean = float(os.environ.get("ROW_MEAN", "38.0"))
    tb = float(os.environ.get("ROW_TOKBLK", "3.3"))
    def st(m, spread=1.0):
        return {{"median": m, "min": m * (1 - spread / 200), "max": m * (1 + spread / 200),
                "spread_pct": spread}}
    summ = {{"mean": st(mean), "p50": st(mean * 0.92), "p90": st(mean * 1.3), "max": st(78.0),
            "ttft_p50_ms": st(423.0), "wall_s": st(300.0), "tok_blk": st(tb), "ms_blk": st(96.0)}}
    os.makedirs("results/row3", exist_ok=True)
    json.dump({{"label": label, "runs": [{{}}, {{}}, {{}}], "summary": summ,
               "code_sha256": "0" * 64, "env": {{}}}}, open(f"results/row3/{{label}}.json", "w"))
    print(f"[row3] results/row3/{{label}}.json"); sys.exit(0)
print("fake python: unknown call", args, file=sys.stderr); sys.exit(3)
'''

SERVE_ENV = """REPO={repo}
QWEN38_ADOPTED=1
PY=/bin/false
NV=nv
HEAD=head
CKPT8=ck8
CKPT16=ck16
CORPUS=corpus
FUSE_PROJ=0
QWEN38_NVFP4_SKINNY=1
QWEN38_SKINNY_TILES=${{REPO}}/ops/skinny-tiles.json
"""

def _base_report(label: str, mean: float = 34.14) -> dict:
    def st(m, spread=1.5):
        return {"median": m, "min": m * (1 - spread / 200), "max": m * (1 + spread / 200),
                "spread_pct": spread}
    return {"label": label, "runs": [{}, {}, {}],
            "summary": {"mean": st(mean), "p50": st(31.85), "p90": st(44.2), "max": st(78.5),
                        "ttft_p50_ms": st(424.0), "wall_s": st(302.0)}}


def make_tree() -> str:
    d = tempfile.mkdtemp(prefix="gate-")
    repo = os.path.join(d, "p1")
    for sub in ("ops", "tools", "tests/gpu", "results/row3", "engine"):
        os.makedirs(os.path.join(repo, sub))
    shutil.copy(os.path.join(ROOT, "ops/gate.sh"), os.path.join(repo, "ops/gate.sh"))
    shutil.copy(os.path.join(ROOT, "ops/engines.sh"), os.path.join(repo, "ops/engines.sh"))
    os.makedirs(os.path.join(d, "proc"))                   # the process table the gate sees
    for t in ("row3.py", "rowlog.py", "gatecheck.py", "flagoff_identity.py"):
        shutil.copy(os.path.join(ROOT, "tools", t), os.path.join(repo, "tools", t))
    open(os.path.join(repo, "tools/__init__.py"), "w").close()
    for t in ("tests/test_a.py", "tests/test_b.py", "tests/gpu/test_g.py"):
        open(os.path.join(repo, t), "w").write("pass\n")
    open(os.path.join(repo, "ops/serve.env"), "w").write(SERVE_ENV.format(repo=repo))
    json.dump(_base_report("rc4k-nostore"), open(os.path.join(repo, "results/row3/rc4k-nostore.json"), "w"))
    json.dump(_base_report("rc4k-clean", 34.17), open(os.path.join(repo, "results/row3/rc4k-clean.json"), "w"))
    base = os.path.join(d, "base")
    os.makedirs(os.path.join(base, "engine"))
    os.makedirs(os.path.join(base, "tools"))
    os.makedirs(os.path.join(base, "ops"))
    # the base knows the served flag but not the phase's adopted one
    open(os.path.join(base, "ops/serve.env"), "w").write("QWEN38_NVFP4_SKINNY=1\n")
    fake = os.path.join(d, "fakepy")
    open(fake, "w").write(FAKE_PY.format(python=sys.executable))
    os.chmod(fake, 0o755)
    fbin = os.path.join(d, "bin")
    os.makedirs(fbin)
    # no engine is running: the gate's own check must pass
    open(os.path.join(fbin, "pgrep"), "w").write("#!/bin/bash\nexit 1\n")
    os.chmod(os.path.join(fbin, "pgrep"), 0o755)
    if shutil.which("timeout") is None:                        # macOS
        open(os.path.join(fbin, "timeout"), "w").write('#!/bin/bash\nshift\nexec "$@"\n')
        os.chmod(os.path.join(fbin, "timeout"), 0o755)
    return d


def gate(d: str, *args: str, **env) -> tuple[int, str, list[str]]:
    repo = os.path.join(d, "p1")
    calls = os.path.join(d, "calls.log")
    open(calls, "w").close()
    e = dict(os.environ, GATE_PY=os.path.join(d, "fakepy"),
             GATE_BASE_DIR=os.path.join(d, "base"), ENGINE_PROC=os.path.join(d, "proc"),
             PATH=os.path.join(d, "bin") + os.pathsep + os.environ["PATH"])
    e.update(env)
    r = subprocess.run(["bash", os.path.join(repo, "ops/gate.sh"), *args], cwd=d, env=e,
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr, open(calls).read().splitlines()


def _index(calls: list[str], needle: str) -> int:
    return next(i for i, c in enumerate(calls) if needle in c)


def test_the_full_protocol_runs_in_order_and_passes():
    d = make_tree()
    rc, out, calls = gate(d, "cand", "--flags", "QWEN38_X=1")
    assert rc == 0, out
    order = ["tests/test_a.py", "tests/test_b.py", "tests/gpu/test_g.py",
             "flagoff_identity_gate.py --from-env --dump", "flagoff_identity.py --from-env --dump",
             "flagoff_identity.py --compare", "verify_spec.py", "row3.py --label cand-nostore ",
             "gatecheck.py --mode adopt", "row3.py --label cand-nostore-r2",
             "row3.py --label cand-clean"]
    idx = [_index(calls, n) for n in order]
    assert idx == sorted(idx), list(zip(order, idx))
    row = next(c for c in calls if "--label cand-nostore " in c)
    # the candidate flag and the served flags ride on the row as --env, the store never written
    assert "--env QWEN38_X=1" in row and "--env QWEN38_NVFP4_SKINNY=1" in row, row
    assert "--store off" in row and "--port 8011" in row and "--max-len 262144" in row, row
    assert "--server-arg=--drop-idle" in row, row
    # the identity ran with the base's flag set: the phase's adopted flag unset, the served one kept
    assert "--env QWEN38_ADOPTED=1" in row
    stub = open(os.path.join(d, "p1/results/gate/cand/report.md")).read()
    assert "env -u QWEN38_ADOPTED" in stub and "-u QWEN38_NVFP4_SKINNY" not in stub, stub
    assert "--store clean" in next(c for c in calls if "--label cand-clean" in c)
    assert "PASS" in out.splitlines()[-2], out


def test_a_failing_step_stops_the_script():
    d = make_tree()
    open(os.path.join(d, "p1/tests/test_b.py"), "w").write("raise SystemExit('FAIL')\n")
    rc, out, calls = gate(d, "cand", "--flags", "QWEN38_X=1")
    assert rc == 1, out
    assert not any("verify_spec" in c or "row3.py" in c or "flagoff" in c for c in calls), calls
    assert "ABORT at 1 suite" in out, out
    d = make_tree()
    rc, out, calls = gate(d, "cand", "--flags", "QWEN38_X=1", GATE_LOSSLESS="FAIL")
    assert rc == 1, out
    assert not any("row3.py" in c for c in calls), calls


def test_the_exit_code_is_the_ship_rule():
    d = make_tree()
    # a resolved-worse mean fails
    rc, out, _ = gate(d, "worse", "--flags", "QWEN38_X=1", "--skip-suite", "--skip-gpu",
                      "--skip-identity", "--skip-lossless", ROW_MEAN="30.0")
    assert rc == 1 and "RESOLVED-WORSE" in out, out
    # adopt: a mean the row cannot resolve is not an adoption
    rc, out, _ = gate(d, "tie", "--flags", "QWEN38_X=1", "--skip-suite", "--skip-gpu",
                      "--skip-identity", "--skip-lossless", ROW_MEAN="34.2")
    assert rc == 1 and "not resolved better" in out, out
    # a phase baseline is held to "nothing worse" while the base reports keep "adopt"
    import shutil as _sh
    _sh.copy(os.path.join(d, "p1/results/row3/rc4k-nostore.json"), os.path.join(d, "p1/results/row3/ph-ns.json"))
    _sh.copy(os.path.join(d, "p1/results/row3/rc4k-clean.json"), os.path.join(d, "p1/results/row3/ph-cl.json"))
    rc, out, _ = gate(d, "ph", "--flags", "QWEN38_X=1", "--skip-suite", "--skip-gpu",
                      "--skip-identity", "--skip-lossless", "--base-nostore", "results/row3/rc4k-nostore.json",
                      "--phase-nostore", "results/row3/ph-ns.json", "--phase-clean", "results/row3/ph-cl.json",
                      ROW_MEAN="38.0")
    assert rc == 0 and "compare ph-nostore vs ph-ns (noworse)" in out, out
    assert "vs rc4k-nostore (adopt): PASS" in out and "(noworse): PASS" in out, out
    # noworse (a baseline run, no flags): the same tie passes
    rc, out, _ = gate(d, "base", "--skip-suite", "--skip-gpu", "--skip-identity",
                      "--skip-lossless", ROW_MEAN="34.2")
    assert rc == 0, out


def test_a_second_clean_row_is_a_clean_row():
    """The deploy gate of 2026-09-25 wants the clean row twice (the noise rule: run one of two
    configurations twice before deciding). `--rows "... clean-r2"` must write the store's clean copy
    and compare against the clean base, not run store off against the store-off base."""
    d = make_tree()
    rc, out, calls = gate(d, "cand", "--skip-suite", "--skip-gpu", "--skip-identity", "--skip-lossless",
                          "--rows", "nostore clean clean-r2")
    assert rc == 0, out
    row = next(c for c in calls if "--label cand-clean-r2" in c)
    assert "--store clean" in row, row
    assert "compare cand-clean-r2 vs rc4k-clean" in out, out


def test_server_args_ride_on_every_row_and_in_the_stub():
    """The candidate is a server option (`--sampled-tree=det`), not an environment variable:
    `--server-args` must reach every row as --server-arg=..., beside the served flags, and the stub
    must say so. Without it the rows run exactly as before."""
    d = make_tree()
    rc, out, calls = gate(d, "eng109", "--skip-suite", "--skip-gpu", "--skip-identity", "--skip-lossless",
                          "--mode", "noworse", "--server-args", "--sampled-tree=det")
    assert rc == 0, out
    rows = [c for c in calls if "row3.py --label" in c]
    assert len(rows) == 3 and all("--server-arg=--sampled-tree=det" in r for r in rows), rows
    assert all("--server-arg=--drop-idle" in r and "--env QWEN38_NVFP4_SKINNY=1" in r for r in rows), rows
    stub = open(os.path.join(d, "p1/results/gate/eng109/report.md")).read()
    assert "server args: `--sampled-tree=det`" in stub, stub
    d2 = make_tree()
    rc, out, calls = gate(d2, "plain", "--skip-suite", "--skip-gpu", "--skip-identity", "--skip-lossless",
                          "--mode", "noworse")
    assert rc == 0 and not any("sampled-tree" in c for c in calls), calls
    assert "server args" not in open(os.path.join(d2, "p1/results/gate/plain/report.md")).read()


def test_the_report_holds_every_step_and_nothing_leaves_results():
    d = make_tree()
    rc, out, _ = gate(d, "cand", "--flags", "QWEN38_X=1")
    assert rc == 0, out
    stub = open(os.path.join(d, "p1/results/gate/cand/report.md")).read()
    assert re.search(r"^## \d{4}-\d\d-\d\d \d\d:\d\d -- gate cand \(ops/gate.sh\)$", stub, re.M), stub
    assert "$ " in stub and "verify_spec.py" in stub and "row3.py" in stub, stub
    assert stub.rstrip().endswith("GATE cand: every step PASS."), stub[-200:]
    assert not os.path.exists(os.path.join(d, "p1/notes")), "the gate writes under results/ only"


def test_a_missing_base_report_is_refused_before_anything_runs():
    d = make_tree()
    os.remove(os.path.join(d, "p1/results/row3/rc4k-clean.json"))
    rc, out, calls = gate(d, "cand")
    assert rc == 1 and "REFUSING" in out and "rc4k-clean.json" in out, out
    assert calls == [], calls


def test_the_wide_tile_table_is_the_candidates_not_the_serving_dirs():
    """Serve.env names its tables under ${REPO}, the serving dir; the gate measures ITS directory's
    code, so both tables -- the 16-row one and the 17..32-row one -- must be its own. A missing
    wide table would silently serve every 17..32-row verify on the base tile."""
    d = make_tree()
    repo = os.path.join(d, "p1")
    with open(os.path.join(repo, "ops/serve.env"), "a") as fh:
        fh.write("QWEN38_SKINNY_TILES_WIDE=/nowhere/ops/skinny-tiles-wide.json\n")
    rc, out, calls = gate(d, "cand", "--flags", "QWEN38_X=1")
    assert rc == 0, out
    row = next(c for c in calls if "--label cand-nostore " in c)
    assert f"--env QWEN38_SKINNY_TILES_WIDE={repo}/ops/skinny-tiles-wide.json" in row, row
    assert f"--env QWEN38_SKINNY_TILES={repo}/ops/skinny-tiles.json" in row, row
    # and a serve.env without one gets none
    d2 = make_tree()
    rc, out, calls = gate(d2, "cand", "--flags", "QWEN38_X=1")
    assert "QWEN38_SKINNY_TILES_WIDE" not in next(c for c in calls if "--label cand-nostore " in c)


def test_the_gate_command_line_is_not_an_engine():
    """Stop.sh and hold.sh found engines by `server/app.py` in a command line, and a lock holder
    whose command line matched was killed on 2026-09-23. They check the executable and
    the argument vector now, but the served copy is older than that: the documented invocation,
    with a candidate flag, must still not mention the path."""
    head = open(os.path.join(ROOT, "ops/gate.sh")).read().split("\nset -u", 1)[0]
    inv = " ".join(ln.lstrip("# ").rstrip("\\ ") for ln in head.splitlines()
                   if "flock -o" in ln or "gate.sh spd" in ln)
    assert "gate.sh" in inv and "flock -o" in inv, inv
    assert not re.search(r"server/app.py", inv), inv


def _proc(d: str, pid: int, argv: list[str], exe: str) -> None:
    p = os.path.join(d, "proc", str(pid))
    os.makedirs(p)
    open(os.path.join(p, "cmdline"), "wb").write(b"".join(a.encode() + b"\0" for a in argv))
    os.symlink(exe, os.path.join(p, "exe"))


def test_the_gate_refuses_beside_an_engine_and_not_beside_a_look_alike():
    """In the gate's own check: an engine alive refuses the run, naming it; a lock holder
    that only mentions server/app.py in its command line does not."""
    d = make_tree()
    _proc(d, 4242, ["flock", "-o", "L", "bash", "-c", "python -u server/app.py --port 8000"],
          "/usr/bin/flock")
    rc, out, calls = gate(d, "cand", "--flags", "QWEN38_X=1")
    assert rc == 0 and "REFUSING" not in out, out
    _proc(d, 5151, ["/v/bin/python", "-u", "server/app.py", "--port", "8011"], "/u/bin/python3.11")
    rc, out, calls = gate(d, "cand2", "--flags", "QWEN38_X=1")
    assert rc == 1 and "REFUSING: an engine is running (5151)" in out, out


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
