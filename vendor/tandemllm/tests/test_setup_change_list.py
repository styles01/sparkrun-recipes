"""The review's change list (VIS-155, server/setup/apply.py `plan`), on a CPU against a checkout in a
temporary folder (tests/test_setup_apply.py's Box) with the repository's ops/start.sh.

The test matrix, the Gherkin of VIS-155 (the page's half is dashboard/tests/setup-install.test.ts):

  1  switching from LFM2.5 to Qwen3.8: the run/setup.env the LFM2.5 setup wrote holds TEMPERATURE
     0.1, TOP_K 50 and REP_PENALTY 1.05; the plan lists the three as removed, with their old
     values and what the engine runs with then; the list's length is the number of effective
     differences between that file and the one the job writes      test_switching_from_lfm25
  2  keys equal to the engine's unset value: SESSION_SPILL on (start.sh's ${SESSION_SPILL:-on}) and
     the draft temperature are not listed                          test_keys_equal_to_the_unset_value
  3  same model, new profile: no key is listed as removed; a secret in a hand-edited run/setup.env
     never shows its value                                         test_same_model_new_profile
  4  the job carries the list: its check step says the count, and job.changes is the list
                                                                   test_the_job_page_lists_the_changes

Run: python tests/test_setup_change_list.py
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_setup_apply as T  # noqa: E402
from server.setup import apply, estimate  # noqa: E402

M = T.M
# what the setup guide wrote for LFM2.5-24B-A2B lfm25-exl3-2.5 (3 slots of 131,072, its sampling)
LFM25_OVERLAY = """# Written by the setup guide (server/setup/apply.py) on 2026-10-14T13:08:20Z: lfm2.5-24b-a2b, profile lfm25-exl3-2.5.
# Generated: the next apply replaces it and reset level 2 removes it. ops/serve.env
# reads it after the weight profile, so it never sets the weights.
QSE_PARALLEL='3'
MAX_LEN='131072'
QSE_SIDE_MAX_LEN='131072'
QSE_KV_POOL='off'
QSE_PARALLEL_FLOOR_GB='15'
CONDUCTOR='on'
QSE_CONDUCTOR_CONFIG="${HOME}/.qwen38-spark-engine/conductor"
QWEN38_NARRATOR_MODEL='qwen3.5-2b'
QWEN38_NARRATOR_DIR="${HOME}/.qwen38-spark-engine/models/qwen3.5-2b"
QWEN38_DECIDER_MODELS='laya-english,laya-multilingual,laya-typed-decisions'
RESPONSE_CACHE='on'
QSE_BACKENDS="${HOME}/.qwen38-spark-engine/backends/config.sqlite3"
RESIDENT_GB='2.5'
CACHE_GB='2'
TEMPERATURE='0.1'
TOP_K='50'
REP_PENALTY='1.05'
DEFAULT_MAX_TOKENS='32768'
REQUEST_TIMEOUT='3600'
MAX_QUEUE='8'
QUEUE_TIMEOUT='240'
PORT='8000'
HOST='0.0.0.0'
SERVED_MODEL='qwen38-spark-engine'
QSE_DASHBOARD_LOGIN='off'
"""


def _qwen_sel():
    return estimate.check(M, {"model": M["id"], "profile": "exl3-6.0", "slots": 4,
                              "window": 262_144, "side_window": 98_304})


def _box_after_lfm25(box):
    shutil.copy(os.path.join(T.REPO, "ops", "start.sh"), box.ops)
    with open(os.path.join(box.ops, "active-profile"), "w") as fh:
        fh.write("lfm25-exl3-2.5\n")
    with open(apply.overlay_path(), "w") as fh:
        fh.write(LFM25_OVERLAY)


def _effective(keys, **kw):
    got = apply.resolve_env(keys, **kw)
    return {k: (got[k] if k in got else apply._unset_default(k)) for k in keys}


def test_switching_from_lfm25():
    with T.Box() as box:
        _box_after_lfm25(box)
        sel = _qwen_sel()
        p = T._plan(sel)
        by = {c["key"]: c for c in p["changes"]}
        for k, old in (("TEMPERATURE", "0.1"), ("TOP_K", "50"), ("REP_PENALTY", "1.05")):
            assert k in by and by[k]["removed"] is True, (k, by.get(k))
            assert by[k]["current"] == old and by[k]["next"] is None, by[k]
            assert by[k]["from"] == "run/setup.env", by[k]
        assert by["QSE_PARALLEL"]["removed"] is False and by["QSE_PARALLEL"]["next"] == "4"
        # the list is the effective difference between the old file and the one the job writes
        keys = sorted(set(apply.overlay_keys(apply.overlay_path())) |
                      set(apply.overlay_values(M, sel)))
        before = _effective(keys)
        apply.write_overlay(M, sel)
        with open(os.path.join(box.ops, "active-profile"), "w") as fh:
            fh.write("exl3-6.0\n")
        after = _effective(keys)
        diff = {k for k in keys if before[k] != after[k]}
        assert diff == set(by), (sorted(diff ^ set(by)))
        # the keys by name, not a count: since SRV-534 the narrator is off by default, so leaving
        # LFM2.5's setup (narrator on) also turns it off, which the list had not named when it said 16
        assert by["QWEN38_NARRATOR_MODEL"]["current"] == "qwen3.5-2b" and \
            by["QWEN38_NARRATOR_MODEL"]["next"] == "", by["QWEN38_NARRATOR_MODEL"]
        assert set(by) == {"QSE_PARALLEL", "MAX_LEN", "QSE_SIDE_MAX_LEN", "QSE_KV_POOL", "QSE_SIDE_GRAPHS",
                           "QSE_SIDE_SLOT_CONST_GB", "QSE_KV_HARD_FLOOR_GB", "QSE_BARRIER_IDLE_OWNER",
                           "QWEN38_SUFFIX_STORE", "QWEN38_NARRATOR_MODEL", "QWEN38_NARRATOR_DIR",
                           "QWEN38_SHADOW_MODEL", "VISION", "RESPONSE_CACHE", "QSE_BACKENDS",
                           "TEMPERATURE", "TOP_K", "REP_PENALTY"}, sorted(by)
        assert len(p["changes"]) == len(diff), len(diff)


def test_keys_equal_to_the_unset_value():
    with T.Box() as box:
        _box_after_lfm25(box)
        p = T._plan(_qwen_sel())
        listed = {c["key"] for c in p["changes"]}
        assert "SESSION_SPILL" in apply.overlay_values(M, _qwen_sel())
        assert not listed & {"SESSION_SPILL", "QWEN38_DRAFT_TEMP"}, sorted(listed)


def test_same_model_new_profile():
    with T.Box() as box:
        shutil.copy(os.path.join(T.REPO, "ops", "start.sh"), box.ops)
        apply.write_overlay(M, T._sel(profile="exl3-6.0"))
        with open(apply.overlay_path(), "a") as fh:
            fh.write("QSE_ADMIN_TOKEN='s3cret-value'\n")
        p = T._plan(T._sel(profile="exl3-4.5"))
        removed = [c for c in p["changes"] if c["removed"]]
        assert [c["key"] for c in removed] == ["QSE_ADMIN_TOKEN"], removed
        assert "s3cret-value" not in repr(p), removed
        assert removed[0]["current"] == "[redacted]"


def test_the_job_page_lists_the_changes():
    with T.Box() as box:
        _box_after_lfm25(box)
        sel = _qwen_sel()
        jid = T._job(sel, previous="lfm25-exl3-2.5")
        job = apply.read_job(jid)
        # the plan's list, the job's, and the count the check step says are one list
        want = T._plan(sel)["changes"]
        assert job["changes"] == want and len(want) == 18 and any(c["removed"] for c in job["changes"])
        apply.Runner(jid, cmd=T.Recorder(), http=T._smoke_ok("exl3-6.0"), ops=box.ops).main()
        check = next(s for s in apply.read_job(jid)["steps"] if s["id"] == "check")
        assert check["detail"] == f"{len(want)} settings change", check


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
