"""A finish_run that raises does not wedge the conversation: the guard forces it terminal.

`Run._main`'s finally called `finish_run` with no try. finish_run is a dozen store
transactions, title work, emits, a context re-send and the queue drain; any raise escaping
it left the run row non-terminal, `run.active` true and `conv.run` set -- every later
message queued behind a run that no longer existed, the queue never firing, the only trace
a threading.excepthook print. Recovery was a chatd restart. The finally now calls
service.finish_guard on failure: forced terminal state, the conversation released, the
queue drained, everything after the first failure logged.

Run: python tests/test_chatd_finish_guard.py
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from chatd_util import Chat  # noqa: E402

TERMINAL = ("completed", "cancelled", "failed", "interrupted")


def test_a_failing_finish_run_leaves_the_conversation_usable():
    c = Chat()
    try:
        svc = c.svc
        real = svc.finish_run
        seen, first = {"run": None}, {"n": 0}

        def finish_once_then_boom(run, outcome):
            first["n"] += 1
            seen["run"] = run
            if first["n"] == 1:
                raise RuntimeError("the store exploded")     # the issue's failure
            return real(run, outcome)

        svc.finish_run = finish_once_then_boom
        try:
            cid = c.new()
            c.send(cid, "hello")

            # the run thread dies out of finish_run without the guard: conv.run would never
            # clear and every later message would queue behind this ghost forever
            t0 = time.monotonic()
            while time.monotonic() - t0 < 15.0 and svc.conv(cid).run is not None:
                time.sleep(0.02)

            run = seen["run"]
            assert run is not None, "the run never reached finish_run"
            assert svc.conv(cid).run is None, "the conversation is wedged behind a ghost run"
            assert not run.active, f"run.state is {run.state!r}: never forced terminal"

            row = svc.store.one("SELECT state, ended FROM runs WHERE id = ?", (run.id,))
            assert row and row["state"] in TERMINAL, f"runs row says {row['state']!r}"
            assert row["ended"] is not None
            msg = svc.store.one("SELECT status FROM messages WHERE id = ?", (run.message_id,))
            assert msg and msg["status"] != "streaming", f"the assistant message: {msg['status']!r}"
            assert any("finish_run raised RuntimeError" in ln for ln in c.lines), c.lines[-8:]

            # the user's symptom was the NEXT message: it must run, not queue
            got = c.send(cid, "second")
            assert "queued" not in got, f"the ghost run still swallows new messages: {got}"
            c.wait_idle(cid)
        finally:
            svc.finish_run = real
    finally:
        c.close()


def test_the_guard_never_raises_even_when_the_store_is_gone():
    # finish_guard is the last line: a second failure must degrade to a log, not re-wedge

    from chatd.runs import Run
    c = Chat()
    try:
        svc = c.svc
        cid = c.new()
        c.send(cid, "hello")
        c.wait_idle(cid)

        conv = svc.conv(cid)
        run = Run(svc, conv, user_message_id="x", message_id="y", max_steps=1,
                  started_by=None)
        run.state = "running"
        conv.run = run
        real_run_fn = svc.store.run
        svc.store.run = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("the database file is gone"))
        try:
            svc.finish_guard(run, "failed", RuntimeError("boom"))
        finally:
            svc.store.run = real_run_fn
        assert svc.conv(cid).run is None, "memory must be released even when sql fails"
        assert not run.active
        assert any("failed too" in ln for ln in c.lines), c.lines[-8:]
    finally:
        c.close()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
