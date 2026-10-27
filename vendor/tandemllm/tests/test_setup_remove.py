"""Removing a model's weights (SRV-560, QA SRV-561): server/setup/remove.py and GET /v1/setup/weights,
POST /v1/setup/remove, `remove_after` of an apply and the plan's disk refusal, on a CPU against
tests/test_setup_apply.py's Box, never the box's own weights.

The test matrix, the Gherkin of SRV-560:

  1  list installed weights: both profiles with their sizes, the running one active and not
     removable, the assets apart                              test_list_installed_weights
  2  remove a profile that does not run: its folder goes, run/weights-removed.txt has its
     repository, revision and the fetch command, the GB freed are named; a dry run removes nothing
                                                              test_remove_a_profile_that_does_not_run
  3  the running profile is refused (409 active_profile); no confirm (422); an unknown profile (404);
     while a job runs (409)                                    test_what_is_refused
     and a job that starts while remove waits for the job lock (409, nothing removed)
                                test_a_job_that_starts_while_remove_waits_for_the_lock_is_refused
  4  never HOME, a folder outside HOME and the HF cache, a link, the checkout, a folder another
     profile or asset reads                                    test_the_paths_it_never_removes
  5  remove after switching away: the old weights go in the same job, the job says so
                                                              test_remove_after_the_switch
  6  a download that does not fit: the plan names the GB too few and what could be removed
                                                              test_a_download_that_does_not_fit

Run: python tests/test_setup_remove.py
"""
from __future__ import annotations

import collections
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_setup_apply as T  # noqa: E402
from server import app, auth  # noqa: E402
from server.setup import apply, remove  # noqa: E402


def _app(setup_mode=False):
    app.STATE.clear()
    app.STATE.update(auth=auth.Auth(None, None), started=time.time(), wren=T.FakeWren(), setup_mode=setup_mode)
    app.INFLIGHT.update(running=0, waiting=0)


def _big(box, bpw: str, mb: int = 3) -> None:
    with open(os.path.join(box.home, "exl3-quants", bpw, "m-1.safetensors"), "wb") as fh:
        fh.write(b"w" * (mb * 1_000_000))


def test_list_installed_weights():
    with T.Box(on_disk=("exl3-4.5", "exl3-6.0")) as box:
        _app()
        _big(box, "4.5")
        code, b = T._api("GET", "/v1/setup/weights")
        assert code == 200, b
        w = {x["profile"]: x for x in b["weights"] if x["model"] == "qwen3.8-27b"}
        assert set(w) >= {"exl3-4.5", "exl3-6.0"}, sorted(w)
        assert w["exl3-6.0"]["active"] and not w["exl3-6.0"]["removable"]
        assert not w["exl3-4.5"]["active"] and w["exl3-4.5"]["removable"] and w["exl3-4.5"]["gb"] >= 0.003
        assert w["exl3-4.5"]["dir"] == os.path.join(box.home, "exl3-quants", "4.5")
        assert w["exl3-4.5"]["hf_repo"].endswith("EXL3-4.5")
        assert b["assets"] and all("used_by" in a for a in b["assets"]) and b["free_gb"] is not None


def test_remove_a_profile_that_does_not_run():
    with T.Box(on_disk=("exl3-4.5", "exl3-6.0")) as box:
        _app()
        _big(box, "4.5")
        d = os.path.join(box.home, "exl3-quants", "4.5")
        code, b = T._api("POST", "/v1/setup/remove", {"profiles": ["exl3-4.5"], "dry_run": True})
        assert code == 200 and b["dry_run"] and b["removed"][0]["dir"] == d and os.path.isdir(d), b
        code, b = T._api("POST", "/v1/setup/remove", {"profiles": ["exl3-4.5"], "confirm": True})
        assert code == 200 and not os.path.exists(d) and b["freed_gb"] >= 0.003, b
        line = open(os.path.join(box.root, "run", "weights-removed.txt")).read().strip().split("\t")
        assert line[1] == "exl3-4.5" and line[2].startswith("0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-4.5@"), line
        assert line[3] == d and line[5] == "bash ops/use-profile.sh exl3-4.5 --fetch-only", line
        code, b = T._api("GET", "/v1/setup/weights")
        assert "exl3-4.5" not in {x["profile"] for x in b["weights"]}
        assert os.path.isdir(os.path.join(box.home, "exl3-quants", "6.0"))       # the running one stays


def test_what_is_refused():
    with T.Box(on_disk=("exl3-4.5", "exl3-6.0")) as box:
        _app()
        code, b = T._api("POST", "/v1/setup/remove", {"profiles": ["exl3-6.0"], "confirm": True})
        assert code == 409 and b["error"]["code"] == "active_profile", b
        assert os.path.isdir(os.path.join(box.home, "exl3-quants", "6.0"))
        code, b = T._api("POST", "/v1/setup/remove", {"profiles": ["exl3-4.5"]})
        assert code == 422 and b["error"]["code"] == "confirm_required", b
        code, b = T._api("POST", "/v1/setup/remove", {"profiles": ["exl3-2.5"], "confirm": True})
        assert code == 404 and b["error"]["code"] == "not_installed", b
        code, b = T._api("POST", "/v1/setup/remove", {"profiles": "exl3-4.5", "confirm": True})
        assert code == 422, b
        jid = T._job(T._sel())
        st = apply.read_job(jid)
        st.update(state="running", pid=os.getpid(), **apply._proc_identity(os.getpid()))
        apply._write_json(os.path.join(apply.job_dir(jid), "state.json"), st)
        code, b = T._api("POST", "/v1/setup/remove", {"profiles": ["exl3-4.5"], "confirm": True})
        assert code == 409 and b["error"]["code"] == "job_running", b
        # setup mode: nothing serves, so the profile active-profile names may go
        shutil.rmtree(apply.job_dir(jid))
        _app(setup_mode=True)
        code, b = T._api("POST", "/v1/setup/remove", {"profiles": ["exl3-6.0"], "confirm": True})
        assert code == 200 and not os.path.exists(os.path.join(box.home, "exl3-quants", "6.0")), b


def test_a_job_that_starts_while_remove_waits_for_the_lock_is_refused():
    with T.Box(on_disk=("exl3-4.5", "exl3-6.0")) as box:
        _app()
        from server import setup_api
        d = os.path.join(box.home, "exl3-quants", "4.5")
        real = setup_api._JOB_LOCK

        class Racing:                     # an apply takes the lock first and spawns its job
            def __enter__(self):
                jid = T._job(T._sel(profile="exl3-4.5"))
                st = apply.read_job(jid)
                st.update(state="running", pid=os.getpid(), **apply._proc_identity(os.getpid()))
                apply._write_json(os.path.join(apply.job_dir(jid), "state.json"), st)
                return real.__enter__()

            def __exit__(self, *exc):
                return real.__exit__(*exc)
        setup_api._JOB_LOCK = Racing()
        try:
            code, b = T._api("POST", "/v1/setup/remove", {"profiles": ["exl3-4.5"], "confirm": True})
        finally:
            setup_api._JOB_LOCK = real
        assert code == 409 and b["error"]["code"] == "job_running", b
        assert os.path.isdir(d)


def test_the_paths_it_never_removes():
    with T.Box() as box:
        home = box.home
        q = os.path.join(home, "exl3-quants", "4.5")
        assert remove._safe(q, box.root, []) is None
        assert "holds this installation" in remove._safe(home, box.root, [])
        assert "holds this installation" in remove._safe(box.tmp, box.root, [])   # the checkout's parent
        outside = os.path.realpath(os.path.join(box.tmp, "elsewhere"))
        os.makedirs(outside)
        assert "outside the home folder" in remove._safe(outside, box.root, [])
        link = os.path.join(home, "linked")
        os.symlink(q, link)
        assert "is a link" in remove._safe(link, box.root, [])
        assert "holds" in remove._safe(os.path.join(home, "exl3-quants"), box.root, [q])


def test_remove_after_the_switch():
    with T.Box(on_disk=("exl3-4.5", "exl3-6.0")) as box:
        class Switch(T.Recorder):
            def __call__(self, argv, *, timeout, env, cwd, line):
                rc = super().__call__(argv, timeout=timeout, env=env, cwd=cwd, line=line)
                if " ".join(os.path.basename(a) for a in argv[1:]) == "use-profile.sh exl3-4.5":
                    with open(os.path.join(box.ops, "active-profile"), "w") as fh:
                        fh.write("exl3-4.5\n")
                return rc
        sel = T._sel()
        p = T._plan(sel)
        p["profile"]["current"] = "exl3-6.0"
        jid = apply.new_job(T.M, sel, p, setup_mode=False, remove_after=["exl3-6.0"])["id"]
        assert apply.Runner(jid, cmd=Switch(), http=T._smoke_ok(), ops=box.ops).main() == 0
        j = apply.read_job(jid)
        assert j["state"] == "done" and j["removed"]["removed"][0]["profile"] == "exl3-6.0", j.get("removed")
        assert not os.path.exists(os.path.join(box.home, "exl3-quants", "6.0"))
        assert os.path.isdir(os.path.join(box.home, "exl3-quants", "4.5"))
        # through the route: remove_after is checked, the live profile never in it
        _app()
        code, b = T._api("POST", "/v1/setup/apply", {"selection": {"model": T.M["id"]}, "remove_after": ["../x"]})
        assert code == 422, b


def test_a_download_that_does_not_fit():
    with T.Box(on_disk=("exl3-4.5", "exl3-6.0")) as box:
        apply.DISK_MARGIN_GB = 60
        real = apply.shutil.disk_usage
        apply.shutil.disk_usage = lambda p: collections.namedtuple("u", "total used free")(1e12, 9e11, 30e9)
        try:
            p = T._plan(T._sel(profile="exl3-3.0"))
        finally:
            apply.shutil.disk_usage = real
        msg = " ".join(p["errors"])
        assert "disk_full" in p["error_codes"], p["errors"]
        assert "the disk has 30 GB free" in msg and "GB too few." in msg, msg
        assert "Weights on this box that could be removed in the Setup Guide: exl3-4.5" in msg, msg
        assert "exl3-6.0 (" not in msg, msg                       # the running one is not offered


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"  {name:45s} ok ({time.time() - t0:.1f} s)")
            passed += 1
    print(f"{passed} passed")
