"""CPU tests for chatd/store.py _Tx: a BEGIN IMMEDIATE that fails must not leak the store lock.

A failed BEGIN never reaches __exit__ (with-statements only call it when __enter__ returned), so
without the release in __enter__ every later store.tx() in the process blocks forever on a lock
whose holder is gone (issue #8). Reproduced with a real second connection holding the write lock.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatd.store import Store  # noqa: E402


def test_failed_begin_releases_the_lock_and_the_store_recovers():
    path = os.path.join(tempfile.mkdtemp(), "chat.db")
    store = Store(path)

    # A second process pins the database: BEGIN IMMEDIATE from here holds the write lock.
    other = sqlite3.connect(path)
    other.execute("BEGIN IMMEDIATE")
    other.execute("CREATE TABLE pinned (x)")

    # The store's BEGIN now fails on busy_timeout (5000 ms) instead of taking the write lock.
    try:
        with store.tx() as db:
            db.execute("SELECT 1")
        raise AssertionError("BEGIN IMMEDIATE should have failed while the other conn pinned it")
    except sqlite3.OperationalError:
        pass

    # The fix: the process-wide lock came back. Without it this assert is reached only never.
    # _thread.RLock is reentrant and has no .locked() here: probe from a foreign thread.
    probe = []
    def _taker():
        probe.append(store.lock.acquire(timeout=0.3))
        if probe[0]:
            store.lock.release()
    t = threading.Thread(target=_taker)
    t.start()
    t.join()
    assert probe and probe[0], "store lock leaked by a failed BEGIN — chatd is deadlocked"

    other.rollback()
    other.close()

    # And the store works again for a real transaction.
    with store.tx() as db:
        db.execute("CREATE TABLE later (x TEXT)")
        db.execute("INSERT INTO later VALUES ('ok')")
    assert store.db.execute("SELECT x FROM later").fetchone()[0] == "ok"


def test_successful_txn_still_commits():
    path = os.path.join(tempfile.mkdtemp(), "chat.db")
    store = Store(path)
    with store.tx() as db:
        db.execute("CREATE TABLE t (x)")
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    row = store.db.execute(
        "SELECT name FROM sqlite_master WHERE name = 't'").fetchone()
    assert row and row[0] == "t"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
