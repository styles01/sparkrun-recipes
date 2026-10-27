"""The destructive reset (levels 3-5) is serialized against apply jobs by the job lock.

The reset route checked `apply.running_job()` and `busy(app)` and then deleted synchronously
-- without the `_JOB_LOCK` that apply's own admission holds. Two admin requests, a
`POST /v1/setup/reset {level: 5}` and a `POST /v1/setup/apply`, therefore ran a download job
into directories being removed: ENOENT mid-install, half-deleted model folders, a freed_gb
read over files the job had just written. apply-vs-apply was already serialized by the lock;
only this path walked around it. The route now holds `_JOB_LOCK` across a re-check of
`running_job()` and the whole deleting pass.

Run: python tests/test_setup_reset_job_lock.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_setup_apply import Box, FakeWren, _api  # noqa: E402

from server import app, auth  # noqa: E402
from server import setup_api  # noqa: E402
from server.setup import apply, reset  # noqa: E402


def _fake_execute(state):
    """Stands in for the deleting pass; records everything the lock must cover."""

    def execute(n, app_, clear_live=None):
        state["ran"] = n
        state["lock_held"] = not setup_api._JOB_LOCK.acquire(blocking=False)
        try:                                            # a delete is a second of work, here
            time.sleep(0.05)
        finally:
            if not state["lock_held"]:
                setup_api._JOB_LOCK.release()           # pre-fix: we grabbed it ourselves
        return {"contract_version": "1.0", "level": n, "removed": [], "failed": [],
                "freed_gb": 0.0, "restart_needed": True, "message": "fake", "ok": True}
    return execute


def _enter():
    app.STATE.clear()
    app.STATE.update(auth=auth.Auth(None, None), started=time.time(), wren=FakeWren())
    app.INFLIGHT.update(running=0, waiting=0)


def test_a_deleting_reset_holds_the_job_lock_and_rechecks_inside_it():
    with Box():
        _enter()
        state = {"ran": None, "lock_held": None, "jobs": []}
        real_exec, real_jobs = reset.execute, apply.running_job
        try:
            reset.execute = _fake_execute(state)
            code, b = _api("POST", "/v1/setup/reset", {"level": 5, "confirm": "delete history"})
            assert code == 200, b
            assert state["ran"] == 5
            # (a) the deleting ran with _JOB_LOCK held: a concurrent apply could not have
            #     passed its gate (295, the same lock) -- pre-fix, this grab SUCCEEDS:
            assert state["lock_held"] is True, "the deleting ran without the job lock"
        finally:
            reset.execute, apply.running_job = real_exec, real_jobs


def test_a_job_appearing_after_the_fast_check_is_still_a_409():
    # the racy pair, pinned at the check level: running_job() answers None to the route's
    # fast check (line ~446) and a job to the one inside the lock -- the apply that slipped
    # through. The fix must re-check and refuse; pre-fix it deletes anyway.
    with Box():
        _enter()
        ran = []
        calls = iter([None, {"id": "01JOB", "state": "queued"}])
        real_exec, real_jobs = reset.execute, apply.running_job
        try:
            reset.execute = lambda *a, **k: ran.append(a) or {"level": 5, "removed": [],
                                                              "failed": [], "freed_gb": 0.0,
                                                              "ok": True, "restart_needed": True}
            apply.running_job = lambda: next(calls)
            code, b = _api("POST", "/v1/setup/reset", {"level": 5, "confirm": "delete history"})
            assert code == 409 and b["error"]["code"] == "job_running", (code, b)
            assert ran == [], "the reset deleted while a job had appeared"
        finally:
            reset.execute, apply.running_job = real_exec, real_jobs


def test_a_level_3_reset_still_completes_and_reports():
    # the lock must serialize, not break: the happy path answers 200 with its summary
    with Box():
        _enter()
        state = {"ran": None, "lock_held": None}
        real_exec = reset.execute
        try:
            reset.execute = _fake_execute(state)
            code, b = _api("POST", "/v1/setup/reset", {"level": 3, "confirm": "delete caches"})
            assert code == 200 and b["ok"] is True, b
            assert state["ran"] == 3 and state["lock_held"] is True
        finally:
            reset.execute = real_exec


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
