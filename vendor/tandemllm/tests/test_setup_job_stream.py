"""The install page's job stream (VIS-149, VIS-151; GET /v1/setup/jobs/{id} in server/setup_api.py),
on a CPU against a checkout in a temporary folder (tests/test_setup_apply.py's Box).

The test matrix, the Gherkin of VIS-149 and VIS-151 (the page's half is
dashboard/tests/setup-install.test.ts):

  1  reload after the job finished: one job frame, final, no step running or not started, then the
     done event; every log line of the job comes first, in order (236 lines)
                                                     test_reload_after_the_job_finished
  2  reload during the download: the log up to event 150, then the job with the fetch step
     running and the steps before it done, with no id of a log line (VIS-211); the stream stays open
                                                     test_reload_during_the_download
  3  a final state never regresses: no job frame after a final one, also on a resume
     (Last-Event-ID) at every point of the job        test_a_final_state_never_regresses
  4  a job folder that is gone is 404, which the page reads as "no longer on the box"
                                                     test_job_folder_gone

Run: python tests/test_setup_job_stream.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_setup_apply as T  # noqa: E402
from server import app, auth, setup_api  # noqa: E402
from server.setup import apply  # noqa: E402

FINAL = ("done", "failed", "rolled_back")


def _frames(raw: str) -> list[tuple[int | None, dict]]:
    out = []
    for f in raw.split("\n\n"):
        if not f.strip() or f.startswith(":"):
            continue
        fid = None
        data = None
        for line in f.split("\n"):
            if line.startswith("id: "):
                fid = int(line[4:])
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        out.append((fid, data))
    return out


def _stream(jid: str, last_id: int | None = None) -> list[tuple[int | None, dict]]:
    headers = {"Accept": "text/event-stream"}
    if last_id is not None:
        headers["Last-Event-ID"] = str(last_id)
    code, _, raw = T.Req("GET", f"/v1/setup/jobs/{jid}", None, headers).go()
    assert code == 200, code
    return _frames(raw)


def _server():
    app.STATE.clear()
    app.STATE.update(auth=auth.Auth(None, None), started=0.0, wren=T.FakeWren())


def _finished_job(box, n_lines: int) -> str:
    """A job run to done whose events hold `n_lines` log lines."""
    jid = T._job(T._sel())
    # the runner logs each command line and its output: the switch is the one command
    out = [f"line {i}" for i in range(n_lines - 2)]
    apply.Runner(jid, cmd=T.Recorder(out=out), http=T._smoke_ok(), ops=box.ops).main()
    logs = [e for e in apply.events(jid) if e["event"]["type"] == "log"]
    assert len(logs) == n_lines, len(logs)
    return jid


def test_reload_after_the_job_finished():
    with T.Box() as box:
        _server()
        jid = _finished_job(box, 236)
        fr = _stream(jid)
        kinds = [d["type"] for _, d in fr]
        assert kinds == ["log"] * 236 + ["job", "done"], kinds[-4:]
        job = fr[-2][1]["job"]
        assert job["state"] == "done"
        assert not [s for s in job["steps"] if s["state"] in ("running", "pending", "waiting")], \
            job["steps"]
        want = [e["event"]["line"] for e in apply.events(jid) if e["event"]["type"] == "log"]
        assert [d["line"] for _, d in fr[:-2]] == want
        ids = [i for i, _ in fr if i is not None]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
        assert fr[-1][0] == len(apply.events(jid))           # the done event's own sequence


def test_reload_during_the_download():
    with T.Box() as box:
        _server()
        jid = T._job(T._sel())
        r = apply.Runner(jid, cmd=T.Recorder(), http=T._smoke_ok(), ops=box.ops)
        # the runner's first events, by hand: it runs (this process stands for it), Wren, the
        # check and the fetch, and the download's log lines up to event 150
        r.job.update(state="running", pid=os.getpid(), **apply._proc_identity(os.getpid()))
        r._save()
        r._emit({"type": "job", "job": dict(r.job)})
        r.step("unload_assistant", "done", "Wren was unloaded before the job started")
        r.step("check", "running")
        r.step("check", "done", "16 settings change")
        r.step("fetch", "running", "about 63.34 GB")
        while r.seq < 150:
            r.log(f"[fetch] line {r.seq}")
        h = T.Req("GET", f"/v1/setup/jobs/{jid}", None, {"Accept": "text/event-stream"})
        h.wfile = io.BytesIO()
        after, offset, ended = setup_api._catch_up(h, jid, 0)
        assert not ended and after == 150, (after, ended)
        assert offset == os.path.getsize(os.path.join(apply.job_dir(jid), "events.jsonl"))
        fr = _frames(h.wfile.getvalue().decode())
        assert [d["type"] for _, d in fr] == ["log"] * 145 + ["job"], [d["type"] for _, d in fr]
        # VIS-211: the log line before it carries 150, the snapshot no id of its own (a page that
        # counted 150 dropped a snapshot with that id and spun over a finished install)
        assert fr[-1][0] is None and fr[-2][0] == 150, (fr[-2][0], fr[-1][0])
        steps = {s["id"]: s["state"] for s in fr[-1][1]["job"]["steps"]}
        assert steps["fetch"] == "running" and steps["check"] == "done", steps
        assert steps["unload_assistant"] == "done" and steps["switch"] == "pending", steps
        assert fr[-2][1]["line"] == "[fetch] line 149"
        # the live part of the stream follows from there: the next event is 151
        r.step("fetch", "done")
        recs, _ = apply.events_from(jid, offset, after)
        assert [x["seq"] for x in recs] == [151]


def test_a_final_state_never_regresses():
    with T.Box() as box:
        _server()
        jid = _finished_job(box, 40)
        n = len(apply.events(jid))
        for last in [None] + list(range(0, n + 1)):
            fr = _stream(jid, last)
            jobs = [d["job"] for _, d in fr if d["type"] == "job"]
            assert len(jobs) == 1, (last, len(jobs))
            assert jobs[0]["state"] in FINAL, (last, jobs[0]["state"])
            seen_final = False
            for _, d in fr:
                if d["type"] == "job":
                    seen_final = True
                elif d["type"] == "step":
                    assert not seen_final, (last, d)
            assert fr[-1][1]["type"] == "done", (last, fr[-1])
            # a resume never repeats what came before it
            assert all(i > (last or 0) for i, _ in fr if i is not None), last


def test_job_folder_gone():
    with T.Box() as box:
        _server()
        jid = _finished_job(box, 10)
        shutil.rmtree(apply.job_dir(jid))
        code, body = T._api("GET", f"/v1/setup/jobs/{jid}", headers={"Accept": "text/event-stream"})
        assert code == 404 and body["error"]["code"] == "not_found", (code, body)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
