"""A failed Setup Guide step in plain words, Retry from the failed step and Copy diagnostics (VIS-213,
QA VIS-214): server/setup/apply.py (`Runner.explain`, `diagnostics`) and the routes POST
/v1/setup/jobs/{id}/retry and GET /v1/setup/jobs/{id}/diagnostics, on a CPU against
tests/test_setup_apply.py's Box. The page's half is dashboard/tests/setup-install.test.ts.

On 2026-10-17 the failed install said "the new setup did not start (ops/use-profile.sh exit 1)" and
offered only "Back to Review".

The test matrix, the Gherkin of VIS-213:

  1  the start script lost its mode: the step says the script could not be run, tandem doctor
     repaired its mode, press Retry; the other failures read as what broke
                                                     test_failures_in_plain_words
  2  Retry after a failed switch: a new job (retry_of), the check runs again, the download is not
     repeated, the switch runs again; a running job, a done job and an unknown id are refused
                                                     test_retry_from_the_failed_step
  3  Copy diagnostics: the steps, the log's tail, the doctor report and the versions; no secret
                                                     test_the_diagnostics

Run: python tests/test_setup_retry.py
"""
from __future__ import annotations

import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_setup_apply as T  # noqa: E402
from server import app, auth  # noqa: E402
from server.setup import apply, device as device_mod  # noqa: E402

INCIDENT = ["[use-profile 07:05:06] switching setup mode -> exl3-6.0 (MemAvailable 115 GB)",
            "[stop] stopped",
            "/home/u/TandemLLM/src/ops/use-profile.sh: line 172: /home/u/TandemLLM/src/ops/start.sh: Permission denied",
            "[use-profile 07:05:06] FAILED on exl3-6.0; the caller rolls back"]


class Scripted(T.Recorder):
    """A recorder whose switch prints `lines` and exits `rc`; `before` runs first (the box changes
    under the job, as a deploy did)."""

    def __init__(self, lines, rc, before=None):
        super().__init__()
        self.lines, self.code, self.before = lines, rc, before

    def __call__(self, argv, *, timeout, env, cwd, line):
        key = " ".join(os.path.basename(a) for a in argv[1:])
        if key == "use-profile.sh exl3-4.5":
            self.calls.append({"argv": list(argv), "env": dict(env), "timeout": timeout, "cwd": cwd})
            if self.before:
                self.before()
            for ln in self.lines:
                line(ln)
            return self.code
        return super().__call__(argv, timeout=timeout, env=env, cwd=cwd, line=line)


def _failed(box, lines, rc=1, before=None):
    jid = T._job(T._sel())
    apply.Runner(jid, cmd=Scripted(lines, rc, before), http=T._smoke_ok(), ops=box.ops).main()
    return apply.read_job(jid)


def test_failures_in_plain_words():
    with T.Box() as box:
        with open(os.path.join(box.ops, "executables.txt"), "w") as fh:
            fh.write("ops/start.sh\n")
        os.chmod(os.path.join(box.ops, "start.sh"), 0o755)
        j = _failed(box, INCIDENT, before=lambda: os.chmod(os.path.join(box.ops, "start.sh"), 0o644))
        sw = next(s for s in j["steps"] if s["id"] == "switch")
        assert sw["detail"] == ("The new setup did not start: the script start.sh could not be run (Permission "
                                "denied: it lost its execute bit). tandem doctor has repaired its file mode; "
                                "press Retry."), sw["detail"]
        assert "ops/start.sh: execute bit restored (was 644)" in j["doctor"]["repaired"], j["doctor"]
        assert os.stat(os.path.join(box.ops, "start.sh")).st_mode & 0o100
        assert "exit 1" not in sw["detail"] or "Permission" in sw["detail"]
    with T.Box() as box:
        for lines, rc, want in (
                (["[use-profile 07:05] REFUSING: a hold owns the board (.watchdog.off, 12s old)"], 6,
                 "The new setup did not start: a hold owns the board (a measurement or maintenance runs); "
                 "try again when it ends. (exit 6)"),
                (["[use-profile] MemAvailable 40 GB < 80 GB after the stop; the caller rolls back"], 7,
                 "The new setup did not start: there was not enough free memory to start it; stop other "
                 "programs that use memory, then press Retry. (exit 7)"),
                (["[start] FAILED to become healthy in 600s; tail of x:", "CUDA error: out of memory"], 1,
                 "The new setup did not start: the new engine did not answer in time (its log: CUDA error: "
                 "out of memory). (exit 1)"),
                ([], 124, "The new setup did not start: it did not finish within 30 minutes and was "
                          "stopped. Press Retry."),
                (["something odd"], 3, "The new setup did not start (exit 3): something odd.")):
            j = _failed(box, lines, rc)
            sw = next(s for s in j["steps"] if s["id"] == "switch")
            assert sw["detail"] == want, (sw["detail"], want)


def test_retry_from_the_failed_step():
    with T.Box(on_disk=()) as box:
        app.STATE.clear()
        app.STATE.update(auth=auth.Auth(None, None), started=time.time(), wren=T.FakeWren())
        app.INFLIGHT.update(running=0, waiting=0)
        spawned = []
        real = apply.spawn
        apply.spawn = lambda jid: spawned.append(jid) or 4242
        try:
            # the first install: the download runs, then the switch fails and rolls back
            first = _failed(box, ["[start] something broke"], 1)
            assert first["state"] == "rolled_back" and first["failed_step"] == "switch", first
            code, b = T._api("POST", f"/v1/setup/jobs/{first['id']}/retry", {})
            assert code == 202 and b["job"]["retry_of"] == first["id"] and spawned == [b["job"]["id"]], b
            # the retry: the check again, no download (the weights are here now), the switch again
            rec = T.Recorder()
            apply.Runner(b["job"]["id"], cmd=rec, http=T._smoke_ok(), ops=box.ops).main()
            j = apply.read_job(b["job"]["id"])
            st = {s["id"]: s["state"] for s in j["steps"]}
            seq = [" ".join(os.path.basename(a) for a in c["argv"][1:]) for c in rec.calls]
            assert j["state"] == "done" and st["check"] == "done" and st["fetch"] == "skipped", st
            assert seq == ["use-profile.sh exl3-4.5"], seq
            # refused: a done job, a job while another runs, an unknown one
            code, b = T._api("POST", f"/v1/setup/jobs/{j['id']}/retry", {})
            assert code == 409 and b["error"]["code"] == "not_retryable", b
            other = T._job(T._sel())
            state = apply.read_job(other)
            state.update(state="running", pid=os.getpid(), **apply._proc_identity(os.getpid()))
            apply._write_json(os.path.join(apply.job_dir(other), "state.json"), state)
            code, b = T._api("POST", f"/v1/setup/jobs/{first['id']}/retry", {})
            assert code == 409 and b["error"]["code"] == "job_running", b
            code, b = T._api("POST", "/v1/setup/jobs/0123456789abcdef/retry", {})
            assert code == 404, b
        finally:
            apply.spawn = real


def test_the_diagnostics():
    with T.Box() as box:
        app.STATE.clear()
        app.STATE.update(auth=auth.Auth(None, None), started=time.time(), wren=T.FakeWren())
        os.environ["HF_TOKEN"] = T.HF
        try:
            j = _failed(box, INCIDENT + [f"token {T.HF} in a line"], 1)
            code, head, raw = T.Req("GET", f"/v1/setup/jobs/{j['id']}/diagnostics").go()
        finally:
            os.environ.pop("HF_TOKEN", None)
        assert code == 200 and "text/plain" in head, head
        text = raw if isinstance(raw, str) else raw.decode()
        assert text.startswith("TandemLLM Setup Guide diagnostics\nTandemLLM "), text[:200]
        assert f"job {j['id']} (apply), rolled_back" in text and "Python " in text
        assert "  switch" in text and "failed" in text and "tandem doctor before the job:" in text
        assert "tandem doctor now:" in text and "Permission denied" in text
        assert "selection: {" in text and '"profile": "exl3-4.5"' in text
        assert T.HF not in text and "[redacted" in text, text[-500:]
        code, _ = T._api("GET", "/v1/setup/jobs/0123456789abcdef/diagnostics")
        assert code == 404


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"  {name:45s} ok ({time.time() - t0:.1f} s)")
            passed += 1
    print(f"{passed} passed")
