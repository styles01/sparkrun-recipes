#!/usr/bin/env python3
"""Acceptance checker for minicalc. Prints `calccheck minicalc` and a RESULT line."""
import math, subprocess, sys, importlib, os, unittest, io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print("calccheck minicalc")
fails = {}
def bad(group, why):
    fails.setdefault(group, []).append(why)

try:
    mc = importlib.import_module("minicalc")
    evaluate, CalcError = mc.evaluate, mc.CalcError
except Exception as exc:  # noqa: BLE001
    print(f"import failed: {type(exc).__name__}: {exc}")
    print("RESULT: FAIL <- (a) import 1 > 0")
    sys.exit(1)

VALUES = [
    ("arith", "1 + 2 * 3", 7), ("arith", "(1 + 2) * 3", 9), ("arith", "10 / 4", 2.5),
    ("arith", "10 % 4", 2), ("arith", "-3 + 5", 2), ("arith", "--3", 3), ("arith", ".5 + 1", 1.5),
    ("arith", "7 - 2 - 1", 4), ("arith", "2 * 3 % 4", 2),
    ("power", "2 ^ 3 ^ 2", 512), ("power", "-2 ^ 2", -4), ("power", "(-2) ^ 2", 4), ("power", "2 ^ -1", 0.5),
    ("vars", "x = 3; x * x", 9), ("vars", "a = 2; b = a + 1; a * b", 6), ("vars", "y = 4", 4),
    ("funcs", "sqrt(16)", 4), ("funcs", "abs(-2.5)", 2.5), ("funcs", "max(1, 7, 3)", 7),
    ("funcs", "min(4)", 4), ("funcs", "round(2.567, 2)", 2.57), ("funcs", "round(2.5)", 2),
    ("funcs", "pi", math.pi), ("funcs", "e ^ 1", math.e), ("funcs", "max(2, sqrt(81)) + 1", 10),
]
for group, src, want in VALUES:
    try:
        got = evaluate(src)
        if isinstance(want, int) and not isinstance(want, bool) and group != "funcs":
            ok = got == want
        else:
            ok = abs(float(got) - float(want)) < 1e-9
        if not ok:
            bad(group, f"{src!r} -> {got!r}, want {want!r}")
    except Exception as exc:  # noqa: BLE001
        bad(group, f"{src!r} raised {type(exc).__name__}: {exc}")

if evaluate("6 / 3") != 2:
    bad("arith", "6 / 3 should equal 2")
if type(evaluate("2 * 3")) is not int:
    bad("arith", "2 * 3 should stay an int")

ERRORS = [("1 / 0", "zero"), ("5 % 0", "zero"), ("q + 1", "q"), ("foo(1)", "foo"), ("sqrt(1, 2)", "arg"),
          ("2 $ 3", "3"), ("(1 + 2", "paren"), ("1 + 2)", ""), ("1 2", ""), ("round()", "arg")]
for src, needle in ERRORS:
    try:
        r = evaluate(src)
        bad("errors", f"{src!r} gave {r!r}, want CalcError")
    except CalcError as exc:
        if needle and needle not in str(exc).lower():
            bad("errors", f"{src!r}: message {str(exc)!r} does not mention {needle!r}")
    except Exception as exc:  # noqa: BLE001
        bad("errors", f"{src!r} raised {type(exc).__name__}, not CalcError")

env = {}
try:
    evaluate("z = 5", env)
    if evaluate("z + 1", env) != 6:
        bad("vars", "env not kept between calls")
except Exception as exc:  # noqa: BLE001
    bad("vars", f"env: {type(exc).__name__}: {exc}")

def run(args, stdin=None):
    return subprocess.run([sys.executable, "-m", "minicalc", *args], input=stdin, capture_output=True,
                          text=True, timeout=20, cwd=os.path.dirname(os.path.abspath(__file__)))
r = run(["2 + 2"])
if r.returncode != 0 or r.stdout.strip() != "4":
    bad("cli", f"one-shot '2 + 2' -> rc {r.returncode} out {r.stdout.strip()!r}")
r = run(["1 / 0"])
if r.returncode != 1 or not r.stdout.strip().startswith("error:") and not r.stderr.strip().startswith("error:"):
    bad("cli", f"one-shot error -> rc {r.returncode} out {r.stdout.strip()!r} err {r.stderr.strip()[:80]!r}")
r = run([], "a = 2\n\na * 21\n")
if r.stdout.split() != ["2", "42"]:
    bad("cli", f"repl -> {r.stdout.split()!r}, want ['2', '42']")

tests = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
if os.path.isdir(tests):
    suite = unittest.defaultTestLoader.discover(tests)
    n = suite.countTestCases()
    res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    if n < 15:
        bad("own", f"{n} own tests < 15")
    if res.failures or res.errors:
        bad("own", f"own tests: {len(res.failures)} failures, {len(res.errors)} errors")
else:
    bad("own", "no tests/ directory")

for g, whys in fails.items():
    for w in whys[:4]:
        print(f"  [{g}] {w}")
if fails:
    labels = "abcdefg"
    parts = [f"({labels[i]}) {g} failed {len(w)} > 0" for i, (g, w) in enumerate(fails.items())]
    print("RESULT: FAIL <- " + "; ".join(parts))
    sys.exit(1)
print("RESULT: PASS")
