"""A poisoned journal batch costs one batch, not the writer.

The writer thread caught only sqlite3.Error around a batch, so one bug in the
shaping of an item (a TypeError, like the ones this repo's tests exist to catch)
ended the thread: every note after it queued, filled, and dropped in silence —
and nothing exposed that the writer was gone. The batch is now rolled back,
counted and skipped, and info() says whether the writer is alive.
"""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from server.conductor import decisions  # noqa: E402


def test_writer_survives_a_poisoned_batch():
    path = os.path.join(tempfile.mkdtemp(prefix="journal-live-"), "ledger.sqlite3")
    jr = decisions.Journal(path)
    try:
        orig = jr._apply
        seen = []

        def poison(conn, it):
            seen.append(it)
            if len(seen) == 1:
                raise TypeError("shaping bug")
            return orig(conn, it)

        jr._apply = poison
        jr.note("config", "loaded")                 # the batch that will be poisoned
        jr.flush(timeout=2.0)
        assert jr.errors == 1, f"the poison batch: errors={jr.errors}"
        assert jr.info()["writer_alive"], "the writer survived its poison"

        jr.note("memory", "prune", decision="ok")   # the batch that used to never arrive
        jr.flush(timeout=2.0)
        rows = [r for r in jr.rows(kind="memory") if r.get("task") == "prune"]
        assert rows, "the next batch still writes after a poisoned one"
        assert jr.errors == 1, "a clean batch adds no error"
    finally:
        jr.close()


if __name__ == "__main__":
    test_writer_survives_a_poisoned_batch()
    print("1 passed")
