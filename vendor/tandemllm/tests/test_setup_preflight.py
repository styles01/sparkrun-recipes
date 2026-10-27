"""The preflight of a Setup Guide install (OPS-169, QA OPS-170): everything the restart needs is checked
before anything stops (server/setup/apply.py `preflight`, the step between the download and the
configuration), on a CPU against tests/test_setup_apply.py's Box, never the box's own tree.

On 2026-10-17 an install stopped the running setup before it found out that start.sh could not run.

The test matrix, the Gherkin of OPS-169:

  1  a script that does not parse: the preflight step fails naming it, nothing was stopped (no
     stop.sh, no switch) and no configuration was written     test_a_script_that_does_not_parse
  2  incomplete weights (a shard the index names is missing; a file shorter than the hub manifest
     says) are named                                          test_incomplete_weights
  3  not enough memory: MemAvailable plus what the engine frees under the profile's need, both
     numbers named; setup mode counts nothing to free         test_not_enough_memory
  4  the new configuration does not source under set -u: the line's error named; the engine's
     interpreter missing: named                               test_the_configuration_and_the_interpreter
  5  all checks pass: the step is done and the job goes on     test_all_checks_pass

Run: python tests/test_setup_preflight.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_setup_apply as T  # noqa: E402
from server.setup import apply, device as device_mod  # noqa: E402


def _run(box, sel=None, http=None, setup_mode=False):
    rec = T.Recorder()
    jid = T._job(sel or T._sel(), setup_mode=setup_mode)
    rc = apply.Runner(jid, cmd=rec, http=http or T._smoke_ok(), ops=box.ops).main()
    j = apply.read_job(jid)
    seq = [" ".join(os.path.basename(a) for a in c["argv"][1:]) for c in rec.calls]
    return rc, j, seq


def _stopped_nothing(box, j, seq):
    st = {s["id"]: s["state"] for s in j["steps"]}
    assert j["state"] == "failed" and j["failed_step"] == "preflight", j
    assert st["write_config"] == "skipped" and st["switch"] == "skipped", st
    assert not any(s.startswith(("stop.sh", "start.sh")) or s == "use-profile.sh exl3-4.5" for s in seq), seq
    assert not os.path.exists(os.path.join(box.root, "run", "setup.env"))
    assert "Nothing was stopped or changed." in j["message"], j["message"]


def test_a_script_that_does_not_parse():
    with T.Box() as box:
        with open(os.path.join(box.ops, "start.sh"), "w") as fh:
            fh.write("#!/bin/bash\nif then fi (\n")
        rc, j, seq = _run(box)
        assert rc == 1
        _stopped_nothing(box, j, seq)
        assert "ops/start.sh has an error:" in j["message"], j["message"]
        os.unlink(os.path.join(box.ops, "stop.sh"))
        rc, j, seq = _run(box)
        assert "ops/stop.sh is missing on this box" in j["message"], j["message"]


def test_incomplete_weights():
    with T.Box() as box:
        d = os.path.join(box.home, "exl3-quants", "4.5")
        with open(os.path.join(d, "_hub_manifest.json"), "w") as fh:
            json.dump({"files": {"m-1.safetensors": {"size": 999, "sha256": "x"}}}, fh)
        device_mod.clear()
        rc, j, seq = _run(box)
        _stopped_nothing(box, j, seq)
        assert "A weight file is cut short: m-1.safetensors has 1 bytes, the hub lists 999." in j["message"], \
            j["message"]
    with T.Box() as box:
        # the plan saw the shard, then it went (a download that another program cut, a disk): named
        real = apply.plan

        def plan(*a, **k):
            p = real(*a, **k)
            os.unlink(os.path.join(box.home, "exl3-quants", "4.5", "m-1.safetensors"))
            device_mod.clear()
            apply.plan = real
            return p
        apply.plan = plan
        try:
            rc, j, seq = _run(box)
        finally:
            apply.plan = real
        _stopped_nothing(box, j, seq)
        assert "The weights of exl3-4.5 are not complete on this box" in j["message"], j["message"]


def test_not_enough_memory():
    with T.Box() as box:
        device_mod.meminfo = lambda: {"MemTotal": 121.6, "MemAvailable": 30.0}

        def http(url, body=None, timeout=10):
            if url.endswith("/health"):
                return {"status": "ok", "weights": {"name": "exl3-4.5"}, "memory": {"engine_reserved_gb": 20.0}}
            return {"choices": [{"message": {"content": "ready"}}]}
        rc, j, seq = _run(box, http=http)
        _stopped_nothing(box, j, seq)
        assert "Not enough memory for exl3-4.5: it needs 80 GiB free to start; 30.0 GiB are free now and " \
               "the running engine gives back about 20.0." in j["message"], j["message"]
        # with room: through
        device_mod.meminfo = lambda: {"MemTotal": 121.6, "MemAvailable": 62.0}
        rc, j, seq = _run(box, http=http)
        assert rc == 0 and j["state"] == "done", j["message"]


def test_the_configuration_and_the_interpreter():
    with T.Box() as box:
        with open(os.path.join(box.ops, "serve.env"), "a") as fh:
            fh.write("BROKEN=$NOT_SET_ANYWHERE\n")
        rc, j, seq = _run(box)
        _stopped_nothing(box, j, seq)
        assert "serve.env does not load with the new configuration:" in j["message"] and \
            "NOT_SET_ANYWHERE" in j["message"], j["message"]
    with T.Box() as box:
        text = open(os.path.join(box.ops, "serve.env")).read().replace(f"PY={sys.executable}", "PY=/nowhere/python")
        with open(os.path.join(box.ops, "serve.env"), "w") as fh:
            fh.write(text)
        rc, j, seq = _run(box)
        _stopped_nothing(box, j, seq)
        assert "The engine's Python (/nowhere/python) does not exist." in j["message"], j["message"]


def test_all_checks_pass():
    with T.Box() as box:
        rc, j, seq = _run(box)
        st = {s["id"]: s["state"] for s in j["steps"]}
        assert rc == 0 and st["preflight"] == "done" and st["switch"] == "done", (st, j["message"])
        pf = next(s for s in j["steps"] if s["id"] == "preflight")
        assert pf["detail"].startswith("the scripts, the new configuration, the weights") and \
            seq == ["use-profile.sh exl3-4.5"], (pf, seq)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"  {name:50s} ok ({time.time() - t0:.1f} s)")
            passed += 1
    print(f"{passed} passed")
