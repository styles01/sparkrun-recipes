#!/usr/bin/env python3
"""Acceptance checker for csvq. Prints `taskcheck csvq` and a RESULT line."""
import importlib, io, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
print("taskcheck csvq")
fails = {}


def bad(group, why):
    fails.setdefault(group, []).append(why)


try:
    cq = importlib.import_module("csvq")
    query, QueryError = cq.query, cq.QueryError
except Exception as exc:  # noqa: BLE001
    print(f"import failed: {type(exc).__name__}: {exc}")
    print("RESULT: FAIL <- (a) import 1 > 0")
    sys.exit(1)

DATA = """city,country,price,rooms
Berlin,DE,120,3
Hamburg,DE,95,2
Paris,FR,180,3
Lyon,FR,80,1
Rome,IT,150,2
O'Hare,US,99.5,4
"""

CASES = [
    ("select", "SELECT city FROM t LIMIT 2", [{"city": "Berlin"}, {"city": "Hamburg"}]),
    ("select", "select * from t where city = 'Lyon'",
     [{"city": "Lyon", "country": "FR", "price": 80, "rooms": 1}]),
    ("select", "SELECT city, price FROM t WHERE price > 150", [{"city": "Paris", "price": 180}]),
    ("where", "SELECT city FROM t WHERE country = 'DE' AND rooms >= 3", [{"city": "Berlin"}]),
    ("where", "SELECT city FROM t WHERE country = 'IT' OR country = 'US' AND rooms = 4",
     [{"city": "Rome"}, {"city": "O'Hare"}]),
    ("where", "SELECT city FROM t WHERE (country = 'IT' OR country = 'US') AND rooms = 4",
     [{"city": "O'Hare"}]),
    ("where", "SELECT city FROM t WHERE city = 'O''Hare'", [{"city": "O'Hare"}]),
    ("where", "SELECT city FROM t WHERE city > 5", []),
    ("where", "SELECT city FROM t WHERE price != 120 AND price < 99.6",
     [{"city": "Hamburg"}, {"city": "Lyon"}, {"city": "O'Hare"}]),
    ("order", "SELECT city FROM t ORDER BY price DESC LIMIT 3",
     [{"city": "Paris"}, {"city": "Rome"}, {"city": "Berlin"}]),
    ("order", "SELECT city, rooms FROM t ORDER BY rooms, city DESC LIMIT 3",
     [{"city": "Lyon", "rooms": 1}, {"city": "Rome", "rooms": 2}, {"city": "Hamburg", "rooms": 2}]),
    ("agg", "SELECT COUNT(*) FROM t", [{"COUNT(*)": 6}]),
    ("agg", "SELECT SUM(rooms), MAX(price) FROM t WHERE country = 'FR'",
     [{"SUM(rooms)": 4, "MAX(price)": 180}]),
    ("agg", "SELECT AVG(rooms) FROM t WHERE country = 'DE'", [{"AVG(rooms)": 2.5}]),
    ("group", "SELECT country, COUNT(*) FROM t GROUP BY country ORDER BY COUNT(*) DESC, country",
     [{"country": "DE", "COUNT(*)": 2}, {"country": "FR", "COUNT(*)": 2},
      {"country": "IT", "COUNT(*)": 1}, {"country": "US", "COUNT(*)": 1}]),
    ("group", "SELECT country, MIN(price) FROM t GROUP BY country ORDER BY MIN(price) LIMIT 2",
     [{"country": "FR", "MIN(price)": 80}, {"country": "DE", "MIN(price)": 95}]),
]
for group, sql, want in CASES:
    try:
        got = query(DATA, sql)
        if got != want:
            bad(group, f"{sql!r} -> {got!r}, want {want!r}")
    except Exception as exc:  # noqa: BLE001
        bad(group, f"{sql!r} raised {type(exc).__name__}: {exc}")

try:
    if type(query(DATA, "SELECT SUM(rooms) FROM t")[0]["SUM(rooms)"]) is not int:
        bad("agg", "SUM of ints should stay an int")
except Exception as exc:  # noqa: BLE001
    bad("agg", f"SUM type: {type(exc).__name__}")

ERRORS = [("SELECT nope FROM t", "nope"), ("SELECT city FROM t WHERE", "position"),
          ("SELECT city, COUNT(*) FROM t", "group"), ("SELECT SUM(zz) FROM t", "zz"),
          ("SELEC city FROM t", "position"), ("SELECT city FROM t ORDER BY", "position"),
          ("SELECT rooms, COUNT(*) FROM t GROUP BY country", "group")]
for sql, needle in ERRORS:
    try:
        r = query(DATA, sql)
        bad("errors", f"{sql!r} gave {r!r}, want QueryError")
    except QueryError as exc:
        if needle not in str(exc).lower():
            bad("errors", f"{sql!r}: message {str(exc)!r} does not mention {needle!r}")
    except Exception as exc:  # noqa: BLE001
        bad("errors", f"{sql!r} raised {type(exc).__name__}, not QueryError")

with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
    f.write(DATA)
    path = f.name


def run(args):
    return subprocess.run([sys.executable, "-m", "csvq", *args], capture_output=True, text=True,
                          timeout=20, cwd=HERE)


r = run([path, "SELECT city, price FROM t WHERE country = 'FR' ORDER BY price"])
if r.returncode != 0 or r.stdout != "city,price\nLyon,80\nParis,180\n":
    bad("cli", f"query -> rc {r.returncode} out {r.stdout!r}")
r = run([path, "SELECT nope FROM t"])
if r.returncode != 2 or not r.stderr.strip().startswith("error:"):
    bad("cli", f"error -> rc {r.returncode} err {r.stderr.strip()[:80]!r}")
os.unlink(path)

tests = os.path.join(HERE, "tests")
if os.path.isdir(tests):
    suite = unittest.defaultTestLoader.discover(tests)
    n = suite.countTestCases()
    res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    if n < 12:
        bad("own", f"{n} own tests < 12")
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
