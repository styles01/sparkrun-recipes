"""Check the ledger's migration to schema version 2 on a COPY of a real ledger (SRV-116 row 6).

    python tools/ledger_migrate_check.py ~/.qwen38-spark-engine/usage/ledger.sqlite3

The source is opened read-only and copied with SQLite's online backup API (consistent while the
server writes it); the copy is migrated with the server's own `ledger.migrate`, twice, and the
report says the versions, the row counts and sums before and after, and the new tables. The live
file is never opened for writing. Exit 0 when every row and every sum is kept.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import ledger  # noqa: E402

SUMS = "COUNT(*), SUM(COALESCE(prompt_tokens, 0)), SUM(COALESCE(completion_tokens, 0)), " \
       "MIN(ts_ms), MAX(ts_ms)"


def _schema(conn) -> int:
    try:
        got = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.Error:
        return 0
    return int(got[0]) if got else 0


def check(src: str) -> dict:
    src = os.path.realpath(os.path.expanduser(src))
    d = tempfile.mkdtemp(prefix="ledger-migrate-")
    copy = os.path.join(d, "ledger.sqlite3")
    try:
        ro = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        out = sqlite3.connect(copy)
        try:
            ro.backup(out)
        finally:
            ro.close()
        before = out.execute(f"SELECT {SUMS} FROM requests").fetchone()
        was = _schema(out)
        ledger.migrate(out)
        again = ledger.migrate(out)                       # idempotent: the second run is a no-op
        after = out.execute(f"SELECT {SUMS} FROM requests").fetchone()
        tables = sorted(r[0] for r in out.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"))
        cols = [r[1] for r in out.execute("PRAGMA table_info(requests)")]
        rep = {"source": src, "copy": copy, "schema_before": was, "schema_after": _schema(out),
               "second_run_saw": again, "rows_before": before[0], "rows_after": after[0],
               "sums_before": list(before[1:]), "sums_after": list(after[1:]),
               "tables": tables, "conductor_columns": [c for c in cols
                                                      if c.startswith("conductor_")]}
        out.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    rep["ok"] = (rep["rows_before"] == rep["rows_after"] and rep["sums_before"] ==
                 rep["sums_after"] and rep["schema_after"] == ledger.SCHEMA_VERSION and
                 rep["second_run_saw"] == ledger.SCHEMA_VERSION and
                 {"conductor_events", "conductor_texts"} <= set(rep["tables"]))
    return rep


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    rep = check(argv[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
