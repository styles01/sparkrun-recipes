"""The setup guide's apply job (SRV-327, server/setup/apply.py and its routes), on a CPU against a
checkout in a temporary folder: never the box's own serve.env, run/ or weights.

The test matrix:

  1  the overlay: the registry's keys with the selection's values, ${HOME} kept for bash, every
     value one plain line; it parses (bash -n) and serve.env reads it after the profile; a value
     with a command substitution in it is refused         test_the_overlay
  2  the plan: what changes against serve.env now, the commands it will run, the warnings (no
     password, the 4 x 262,144 worst case); slots past the card's measured 4 go through while they
     fit (VIS-199); and each refusal with its code: over_budget,
     missing_assets (only an asset nothing downloads; OPS-95: a missing asset with a fetch list is
     part of the fetch step and its download_gb), missing_token (a private profile), disk_full,
     no_overlay_hook, secrets_exist, port_in_use, repo_mismatch
                                                         test_the_plan
  3  the allowlist, with a recording runner over a corpus of selections: every command is bash
     and one of ops/use-profile.sh, start.sh, stop.sh, make-secrets.sh, every argument a
     registry id, a number or one of the scripts' own flags, the environment a short fixed list,
     nothing written outside the checkout's run/        test_nothing_outside_the_allowlist
  4  the steps: on disk the fetch is skipped; a download runs --fetch-only first; a port change
     stops the old engine before the overlay; generate runs make-secrets.sh --admin
                                                         test_the_steps
  5  rollback: a failed switch and a failed smoke put the old run/setup.env back and start the
     previous profile (setup mode: the marker back and start.sh --setup); the job says rolled_back
                                                         test_rollback
  6  the routes: a body that names a step, a command, an env or a path is 422 unknown_field; a
     hostile profile or name is refused; dry_run is the plan and no job; busy and job_running are
     409, a missing asset 412; slots 17 (past the engine's 16) is 422 out_of_range; a good apply
     is 202, unloads Wren and starts the runner
                                                         test_the_routes
  7  the job route: ?format=json the job; SSE sends the log on disk, the job as it is now and its
     done event, with their ids, and Last-Event-ID resumes after one; an unknown or malformed id is 404; the stream reads the log from a byte
     offset, each line once, a half-written line on the next read   test_the_job_route
  8  no secret in the log: the HF token, an hf_ token and a bearer token are cut out of every
     line the runner writes                              test_secrets_are_redacted
  9  the real runner, detached (its own session) with stub scripts and a local engine for the
     smoke: the job runs to done on its own and the status responder serves its state
                                                         test_the_detached_runner
 10  the runner's hard cases: it waits for the engine to be idle before anything stops (the step
     says `waiting`; too long and nothing changed); a port change that fails is stopped on the new
     port before the old overlay comes back, and a failure after the old engine stopped rolls back;
     use-profile.sh never rolls back itself (QSE_NO_ROLLBACK) and the rollback's smoke names the
     old served model; the setup-mode marker comes back unchanged (its HOST) on a rollback and after
     a crash; the smoke asks for 512 tokens          test_the_runners_hard_cases
 11  a timed-out command's whole process group goes; a pid that is not the runner's any more (another
     start time, another boot) is not alive; every overlay value is quoted, so a value with a ; in
     it stays one word                                   test_timeouts_liveness_and_quoting
 12  the wizard's seed is the box as it runs (GET .../current): the active profile, the overlay's
     slots, window, tools, port and served name; a HOST neither option names is listed as unmatched;
     the plan warns when HOST, PORT, SERVED_MODEL or QSE_BACKENDS change
                                                         test_the_current_selection
 13  an install.sh box (serve.env and the scripts in run/ops, the profiles and active-profile under
     serve.env's REPO): the active profile and the profile files are found there, and a job runs
                                                         test_an_install_sh_box
 14  a benchmark's --env-file (tools/exl3_fastrun.py) reads serve.env without the overlay
                                                         test_benchmarks_ignore_the_overlay

Run: python tests/test_setup_apply.py
"""

from __future__ import annotations

import contextlib
import http.server
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app, auth  # noqa: E402
from server.setup import apply, device as device_mod, estimate, registry  # noqa: E402

REPO = registry.ROOT
M = registry.get("qwen3.8-27b")
HF = "hf_" + "Zq9" * 12


class Box:
    """A checkout in a temporary folder: ops/ with serve.env (REPO pointing here), the profiles and
    stub scripts; run/; a HOME with the assets and the weights of the profiles named on disk."""

    def __init__(self, on_disk=("exl3-4.5", "exl3-6.0"), assets=True, hook=True, port=8000):
        self.tmp = tempfile.mkdtemp(prefix="setup-box-")
        self.root = os.path.join(self.tmp, "repo")
        self.home = os.path.join(self.tmp, "home")
        ops = self.ops = os.path.join(self.root, "ops")
        shutil.copytree(os.path.join(REPO, "ops", "profiles"), os.path.join(ops, "profiles"))
        with open(os.path.join(REPO, "ops", "serve.env")) as fh:
            env = fh.read()
        env = env.replace("REPO=${HOME}/qwen38-spark-engine", f"REPO={self.root}")
        env = env.replace("PORT=8000", f"PORT={port}")
        # OPS-169: the preflight runs the engine's interpreter; this one stands for it
        env = env.replace("PY=${HOME}/recipes/ling3-flash-dgx-spark/.venv/bin/python", f"PY={sys.executable}")
        if not hook:
            env = env[:env.index("# The setup guide's overlay")]
        with open(os.path.join(ops, "serve.env"), "w") as fh:
            fh.write(env)
        for s in apply.SCRIPTS:
            with open(os.path.join(ops, s), "w") as fh:
                fh.write(f'#!/bin/bash\necho "[{s}] $*"\nexit ${{STUB_RC:-0}}\n')
        # what the preflight (OPS-169) checks beside them: they parse
        for f in ("engines.sh",) + apply.PREFLIGHT_PY:
            shutil.copy(os.path.join(REPO, "ops", f), os.path.join(ops, f))
        # tandem doctor (OPS-171) reads the release's executable files from here: none in this box
        with open(os.path.join(ops, "executables.txt"), "w") as fh:
            fh.write("# none\n")
        os.makedirs(os.path.join(self.root, "run"))
        os.makedirs(self.home)
        if assets:
            for a in M["assets"]:
                for p in [a["path"], *a["also"]]:
                    p = p.replace("${HOME}", self.home).replace("${REPO}", self.root).replace(
                        "${HF_HUB}", os.path.join(self.home, ".cache", "huggingface", "hub")
                    ).replace("*", "snap")
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "w") as fh:
                        fh.write("x")
        for pid in on_disk:
            d = os.path.join(self.home, "exl3-quants", pid.split("-")[1])
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "config.json"), "w") as fh:
                fh.write("{}")
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as fh:
                json.dump({"weight_map": {"a": "m-1.safetensors"}}, fh)
            with open(os.path.join(d, "m-1.safetensors"), "w") as fh:
                fh.write("w")
        self.old = {k: os.environ.get(k) for k in ("HOME", "QSE_OPS_DIR", "HF_TOKEN",
                                                    "HF_HUB_CACHE", "QSE_WEIGHTS", "QSE_SECRETS")}
        os.environ.update(HOME=self.home, QSE_OPS_DIR=ops, QSE_SECRETS=os.path.join(
            self.home, ".qwen38-spark-engine", "secrets.env"))
        for k in ("HF_TOKEN", "HF_HUB_CACHE", "QSE_WEIGHTS"):
            os.environ.pop(k, None)
        with open(os.path.join(ops, "active-profile"), "w") as fh:
            fh.write("exl3-6.0\n")
        apply.ROOT = self.root
        apply.DISK_MARGIN_GB = 0              # this machine's free disk is not the test's
        # nor its free memory (OPS-169's preflight): a board with room for any profile
        self.meminfo = device_mod.meminfo
        device_mod.meminfo = lambda: {"MemTotal": 121.6, "MemAvailable": 115.0}
        device_mod.clear()

    def close(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        apply.ROOT = REPO
        apply.DISK_MARGIN_GB = 60
        device_mod.meminfo = self.meminfo
        device_mod.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def __enter__(self):
        # the box's own ports (8000, 8124) are free here, whatever this machine runs on them (a
        # live engine holds :8000 on the reference box); any other port is asked for real
        self._port_taken = apply._port_taken
        apply._port_taken = lambda port: port not in (8000, 8124) and self._port_taken(port)
        return self

    def __exit__(self, *exc):
        apply._port_taken = self._port_taken
        self.close()

    def files(self) -> set:
        out = set()
        for d, _, fs in os.walk(self.tmp):
            for f in fs:
                out.add(os.path.relpath(os.path.join(d, f), self.tmp))
        return out


def _sel(**kw):
    """The selection the tests plan: the reference box's tools, the narrator on beside them (off by
    default since 0.11.3, SRV-534), and what `kw` sets."""
    kw["components"] = dict({"narrator": True}, **(kw.get("components") or {}))
    return estimate.check(M, dict({"model": M["id"]}, **kw))


def test_the_overlay():
    with Box() as box:
        sel = _sel(slots=2, window=65_536, components={"vision": False},
                   advanced={"served_model": "my-model"})
        text = apply.render_overlay(M, sel)
        assert "QSE_PARALLEL='2'\n" in text and "MAX_LEN='65536'\n" in text
        assert "QSE_SIDE_MAX_LEN='65536'\n" in text and "QSE_KV_POOL='on'\n" in text
        assert 'QSE_CONDUCTOR_CONFIG="${HOME}/.qwen38-spark-engine/conductor"' in text
        assert "VISION='off'\n" in text and "SERVED_MODEL='my-model'\n" in text
        assert "QSE_DISK_CACHE" not in text                  # on: the profile's folder stands
        apply.write_overlay(M, sel)
        got = apply.resolve_env(["QSE_PARALLEL", "VISION", "QSE_DISK_CACHE", "SERVED_MODEL",
                                 "QWEN38_EXL3", "QSE_CONDUCTOR_CONFIG"])
        assert got["QSE_PARALLEL"] == "2" and got["VISION"] == "off", got
        assert got["SERVED_MODEL"] == "my-model"
        assert got["QSE_DISK_CACHE"].endswith("/cache/disk-exl3-6.0"), got   # the profile's
        assert got["QWEN38_EXL3"] == os.path.join(box.home, "exl3-quants", "6.0")
        assert got["QSE_CONDUCTOR_CONFIG"] == os.path.join(box.home, ".qwen38-spark-engine",
                                                           "conductor")
        bad = json.loads(json.dumps(M))
        bad["components"][6]["knobs"]["VISION"]["off"] = "$(reboot)"
        try:
            apply.overlay_values(bad, sel)
        except ValueError as exc:
            assert "VISION" in str(exc)
        else:
            raise AssertionError("a command substitution went into the overlay")
        for v in ("a b", "x'y", 'x"y', "`id`", "a\nb", "${PATH}", "$HOME/x"):
            assert not apply._value_ok(v) or v == "a b", v


def _private(m: dict, pid: str) -> dict:
    """`m` with profile `pid` private (the token check; every published profile is public)."""
    m = json.loads(json.dumps(m))
    registry.profile(m, pid)["hf_private"] = True
    return m


def _no_download(m: dict, aid: str) -> dict:
    """`m` with asset `aid` as one nothing downloads (before OPS-95, every 27B asset)."""
    m = json.loads(json.dumps(m))
    a = registry.asset(m, aid)
    a["fetchable"] = False
    a.pop("fetch")
    return m


def _plan(sel=None, **kw):
    return apply.plan(M, sel or _sel(), setup_mode=kw.pop("setup_mode", False),
                      password_set=kw.pop("password_set", False), **kw)


def test_the_plan():
    with Box():
        p = _plan()
        assert p["ok"], p["errors"]
        keys = {c["key"]: c for c in p["changes"]}
        assert keys["QSE_PARALLEL"]["current"] == "1" and keys["QSE_PARALLEL"]["next"] == "4"
        assert keys["QSE_PARALLEL"]["from"] == "parallelism.slots"
        assert keys["QWEN38_NARRATOR_MODEL"]["next"] == "qwen3.5-2b"
        assert "PORT" not in keys                              # unchanged keys are not listed
        assert p["profile"] == {"current": "exl3-6.0", "next": "exl3-4.5", "on_disk": True,
                                "download_gb": 0.0}, p["profile"]
        assert any("use-profile.sh exl3-4.5" in c for c in p["commands"])
        assert not any("--fetch-only" in c for c in p["commands"])
        assert any("No admin password" in w for w in p["warnings"])
        assert any("full window" in w for w in p["warnings"])
        assert p["estimate"]["fits"] is True and p["restart"] is True
        p = _plan(_sel(kv_pool=False))
        assert not p["ok"] and p["error_codes"][0] == "over_budget", p
        # VIS-199: past the card's measured 4 the memory decides: 6 short chats fit and go into
        # the overlay; 12 long ones do not and the plan says so (over_budget), never clamped
        p = _plan(_sel(slots=6, window=32_768))
        assert p["ok"], p["errors"]
        assert {c["key"]: c for c in p["changes"]}["QSE_PARALLEL"]["next"] == "6"
        assert "QSE_PARALLEL='6'" in apply.render_overlay(M, _sel(slots=6, window=32_768))
        p = _plan(_sel(slots=12, kv_pool=False, window=65_536))
        assert not p["ok"] and p["error_codes"][0] == "over_budget", p
        # OPS-95: every published profile is public; a private one needs a token
        p = _plan(_sel(profile="exl3-3.0"))
        assert p["ok"] and p["profile"]["download_gb"] == 13.5, p
        assert any("--fetch-only" in c for c in p["commands"])
        p = apply.plan(_private(M, "exl3-3.0"), _sel(profile="exl3-3.0"), setup_mode=False,
                       password_set=False)
        assert "missing_token" in p["error_codes"] and p["profile"]["download_gb"] == 13.5
        os.environ["HF_TOKEN"] = HF
        p = apply.plan(_private(M, "exl3-3.0"), _sel(profile="exl3-3.0"), setup_mode=False,
                       password_set=False)
        assert p["ok"], p["errors"]
        assert HF not in json.dumps(p)
        apply.DISK_MARGIN_GB = 10 ** 6
        p = _plan(_sel(profile="exl3-3.0"))
        assert p["error_codes"] == ["disk_full"] and "GB free" in p["errors"][0], p["errors"]
        apply.DISK_MARGIN_GB = 0
        os.environ.pop("HF_TOKEN")
        p = _plan(_sel(advanced={"admin_password": "generate"}))
        assert p["ok"] and any("make-secrets.sh --admin" in c for c in p["commands"])
        os.makedirs(os.path.dirname(os.environ["QSE_SECRETS"]), exist_ok=True)
        with open(os.environ["QSE_SECRETS"], "w") as fh:
            fh.write("QSE_METRICS_TOKEN=x\n")
        assert "secrets_exist" in _plan(_sel(advanced={"admin_password": "generate"}))[
            "error_codes"]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            p = _plan(_sel(advanced={"port": port}))
            assert "port_in_use" in p["error_codes"], p["errors"]
    with Box(assets=False):
        # OPS-95: the 27B's assets have a download: the fetch step gets them, with the weights
        p = _plan()
        assert p["ok"], p["errors"]
        tools = {"qwen3.5-2b", "laya", "qwen3.5-0.8b"}        # the default selection's tools
        gb = sum(e["gb"] for a in M["assets"] if a["id"] in set(registry.profile(M, "exl3-4.5")[
            "assets_required"]) | tools for e in a["fetch"])
        assert p["profile"]["on_disk"] is False and p["profile"]["download_gb"] == round(gb, 2), p
        fetch = [c for c in p["commands"] if "--fetch-only" in c]
        assert len(fetch) == 1 and "Draft models (EXL3 K3)" in fetch[0] and "Base tensors for EXL3" in fetch[0]
        assert "0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-4.5" not in fetch[0]   # the weights are here
        # VIS-153: the tools' models are downloads too, Qwen3.5-2B once for the narrator
        assert "Qwen3.5-2B (the narrator, and Wren)" in fetch[0], fetch
        assert p["fetch_assets"] == ["qwen3.5-2b", "laya", "qwen3.5-0.8b"], p["fetch_assets"]
        p = _plan(_sel(profile="exl3-3.0"))
        assert p["profile"]["download_gb"] == round(13.5 + gb, 2), p["profile"]
        # an asset nothing downloads is still refused before anything changes
        p = apply.plan(_no_download(M, "base-plain"), _sel(), setup_mode=False, password_set=False)
        assert p["error_codes"][0] == "missing_assets" and "Base tensors for EXL3" in p["errors"][0], p
        assert "Draft models (EXL3 K3)" not in p["errors"][0]
    with Box(hook=False):
        assert "no_overlay_hook" in _plan()["error_codes"]
    with Box() as box:
        apply.ROOT = os.path.join(box.tmp, "elsewhere")
        assert "repo_mismatch" in _plan()["error_codes"]


class Recorder:
    def __init__(self, rc=None, out=()):
        self.calls, self.rc, self.out = [], rc or {}, out

    def __call__(self, argv, *, timeout, env, cwd, line):
        self.calls.append({"argv": list(argv), "env": dict(env), "timeout": timeout, "cwd": cwd})
        for ln in self.out:
            line(ln)
        key = " ".join(os.path.basename(a) for a in argv[1:])
        rc = next((rc for k, rc in self.rc.items() if key.startswith(k)), 0)
        if rc == 0 and key.endswith(" --fetch-only"):
            fetched(argv[2], os.path.dirname(argv[1]))
        return rc


#: the files fetched() wrote (the allowlist test tells them from the runner's own)
FETCHED: set = set()


def fetched(profile: str, ops: str) -> None:
    """What a download leaves: the profile's weights complete in its folder (the files its profile
    lists, else config.json and one model.safetensors), so the preflight (OPS-169) finds them."""
    d = device_mod.profile_dirs([profile], ops).get(profile)
    if not d or not d["dir"]:
        return
    os.makedirs(d["dir"], exist_ok=True)
    for f in d["files"] or ["config.json", "model.safetensors"]:
        os.makedirs(os.path.dirname(os.path.join(d["dir"], f)), exist_ok=True)
        FETCHED.add(os.path.realpath(os.path.join(d["dir"], f)))
        with open(os.path.join(d["dir"], f), "w") as fh:
            fh.write("{}" if f.endswith(".json") else "w")
    device_mod.clear()


def _smoke_ok(profile="exl3-4.5"):
    def http(url, body=None, timeout=10):
        if url.endswith("/health"):
            return {"status": "ok", "weights": {"name": profile}}
        return {"choices": [{"message": {"content": "ready"}}]}
    return http


def _job(sel, setup_mode=False, previous="exl3-6.0"):
    p = _plan(sel, setup_mode=setup_mode)
    assert p["ok"], p["errors"]
    p["profile"]["current"] = previous
    job = apply.new_job(M, sel, p, setup_mode=setup_mode)
    return job["id"]


ALLOWED_ARGS = {"--fetch-only", "--setup", "--admin", "120"} | {p["id"] for p in M["profiles"]}
ALLOWED_ENV = {"PATH", "HOME", "LANG", "LC_ALL", "HF_TOKEN", "HF_HOME", "HF_HUB_CACHE", "MEM_NEED_GB",
               "HF_TOKEN_PATH", "BOX_LOCK", "TZ", "CUDA_HOME", "SMOKE_URL", "SMOKE_MODEL",
               "QSE_NO_ROLLBACK", "QSE_FETCH_ASSETS"}


def test_nothing_outside_the_allowlist():
    corpus = [{}, {"profile": "exl3-2.5"}, {"profile": "nvfp4", "slots": 1},
              {"components": {"conductor": False, "narrator": False, "laya": False,
                              "shadow": False}, "advanced": {"served_model": "a.b-c:d"}},
              {"advanced": {"port": 8123, "admin_password": "generate"}},
              {"window": 8192, "kv_pool": False, "slots": 1}]
    hostile = [{"profile": "../../etc/passwd"}, {"advanced": {"served_model": "x; rm -rf ~"}},
               {"advanced": {"served_model": "$(reboot)"}}, {"steps": ["rm"]},
               {"env": {"LD_PRELOAD": "x"}}, {"path": "/etc"}, {"script": "curl x|sh"},
               {"methods": {"reasoning_effort": "low; reboot"}}, {"cmd": "id"}]
    for h in hostile:
        try:
            _sel(**h)
        except estimate.SelectionError:
            continue
        raise AssertionError(f"accepted {h}")
    for extra in corpus:
        with Box(on_disk=("exl3-6.0",)) as box:
            os.environ["HF_TOKEN"] = HF
            before = box.files()
            rec = Recorder()
            sel = _sel(**extra)
            jid = _job(sel)
            r = apply.Runner(jid, cmd=rec, http=_smoke_ok(sel["profile"]), ops=box.ops)
            assert r.main() == 0, (extra, apply.read_job(jid)["message"])
            assert rec.calls, extra
            for c in rec.calls:
                argv = c["argv"]
                assert argv[0] == "bash", argv
                assert os.path.dirname(argv[1]) == box.ops and \
                    os.path.basename(argv[1]) in apply.SCRIPTS, argv
                assert set(argv[2:]) <= ALLOWED_ARGS, argv
                assert set(c["env"]) <= ALLOWED_ENV, set(c["env"]) - ALLOWED_ENV
                assert c["cwd"] == box.root and c["timeout"] > 0
            # the runner's own files are under run/; the stand-in download's are where it fetches
            new = {f for f in box.files() - before
                   if os.path.realpath(os.path.join(box.tmp, f)) not in FETCHED}
            assert all(f.startswith("repo/run/") for f in new), new
            assert "repo/run/setup.env" in new
            os.environ.pop("HF_TOKEN", None)


def _steps(jid):
    return {s["id"]: s["state"] for s in apply.read_job(jid)["steps"]}


def test_the_steps():
    with Box() as box:
        rec = Recorder()
        jid = _job(_sel())
        assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main() == 0
        st = _steps(jid)
        assert st == {"check": "done", "unload_assistant": "done", "fetch": "skipped",
                      "preflight": "done", "write_config": "done", "switch": "done",
                      "smoke": "done", "rollback": "skipped"}, st
        assert [os.path.basename(a) for a in rec.calls[0]["argv"][1:]] == ["use-profile.sh",
                                                                           "exl3-4.5"]
        assert rec.calls[0]["env"]["SMOKE_URL"] == "http://127.0.0.1:8000/v1"
        assert rec.calls[0]["env"]["SMOKE_MODEL"] == "TandemLLM"
        job = apply.read_job(jid)
        assert job["state"] == "done" and "exl3-4.5 is live" in job["message"]
        assert os.path.exists(os.path.join(box.root, "run", "setup.env"))
        # again: the overlay before is kept as a backup
        jid = _job(_sel(slots=2))
        apply.Runner(jid, cmd=Recorder(), http=_smoke_ok(), ops=box.ops).main()
        assert [f for f in os.listdir(os.path.join(box.root, "run")) if ".bak-" in f]
    with Box(on_disk=()) as box:
        os.environ["HF_TOKEN"] = HF
        rec = Recorder()
        jid = _job(_sel(advanced={"port": 8124, "admin_password": "generate"}))
        assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main() == 0
        seq = [" ".join(os.path.basename(a) for a in c["argv"][1:]) for c in rec.calls]
        assert seq == ["use-profile.sh exl3-4.5 --fetch-only", "stop.sh 120",
                       "make-secrets.sh --admin", "use-profile.sh exl3-4.5"], seq
        assert rec.calls[-1]["env"]["SMOKE_URL"] == "http://127.0.0.1:8124/v1"
        # the stand-in download of the job before left the weights: gone again for a failed one
        shutil.rmtree(os.path.join(box.home, "exl3-quants", "4.5"))
        device_mod.clear()
        rec = Recorder(rc={"use-profile.sh exl3-4.5 --fetch-only": 5})
        jid = _job(_sel())
        assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main() == 1
        job = apply.read_job(jid)
        assert job["failed_step"] == "fetch" and len(rec.calls) == 1, job
        assert _steps(jid)["write_config"] == "skipped"
        os.environ.pop("HF_TOKEN", None)


def test_rollback():
    with Box() as box:
        apply.write_overlay(M, _sel(slots=3))
        with open(os.path.join(box.root, "run", "setup.env")) as fh:
            old = fh.read()
        rec = Recorder(rc={"use-profile.sh exl3-4.5": 1})
        jid = _job(_sel())
        assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main() == 1
        job = apply.read_job(jid)
        assert job["state"] == "rolled_back" and job["rolled_back"] is True, job
        assert job["failed_step"] == "switch" and _steps(jid)["rollback"] == "done"
        with open(os.path.join(box.root, "run", "setup.env")) as fh:
            assert fh.read() == old
        seq = [" ".join(os.path.basename(a) for a in c["argv"][1:]) for c in rec.calls]
        assert seq == ["use-profile.sh exl3-4.5", "use-profile.sh exl3-6.0"], seq
        # a smoke that fails rolls back too; with no overlay before, the new one goes
        os.unlink(os.path.join(box.root, "run", "setup.env"))
        jid = _job(_sel())
        bad = _smoke_ok("exl3-6.0")
        assert apply.Runner(jid, cmd=Recorder(), http=bad, ops=box.ops).main() == 1
        assert apply.read_job(jid)["failed_step"] == "smoke"
        assert not os.path.exists(os.path.join(box.root, "run", "setup.env"))
        # setup mode: the marker goes before the switch and comes back with start.sh --setup
        pending = os.path.join(box.root, "run", "setup-pending")
        with open(pending, "w") as fh:
            fh.write("x")
        seen = []
        rec = Recorder(rc={"use-profile.sh exl3-4.5": 1})
        orig = rec.__call__

        def look(argv, **kw):
            seen.append((os.path.basename(argv[1]), os.path.exists(pending)))
            return orig(argv, **kw)

        jid = _job(_sel(), setup_mode=True, previous=None)
        assert apply.Runner(jid, cmd=look, http=_smoke_ok(), ops=box.ops).main() == 1
        assert seen == [("use-profile.sh", False), ("stop.sh", True), ("start.sh", True)], seen
        assert rec.calls[-1]["argv"][2:] == ["--setup"]
        assert apply.read_job(jid)["state"] == "rolled_back"


def test_the_runners_hard_cases():
    # idle: two requests run, then none; the step waits and says so
    with Box() as box:
        seen = iter([2, 2, 0])

        def busy_http(url, body=None, timeout=10):
            if url.endswith("/health") and ":8000/" in url:
                n = next(seen, 0)
                return {"status": "ok", "weights": {"name": "exl3-4.5"},
                        "inflight": {"running": n, "waiting": 0, "lane": 0}}
            return _smoke_ok()(url, body, timeout)

        rec = Recorder()
        jid = _job(_sel())
        r = apply.Runner(jid, cmd=rec, http=busy_http, ops=box.ops, sleep=lambda s: None)
        assert r.main() == 0
        evs = [e["event"] for e in apply.events(jid)]
        waits = [e["step"] for e in evs if e["type"] == "step" and e["step"]["state"] == "waiting"]
        assert waits and waits[0]["id"] == "write_config" and "2 requests" in waits[0]["detail"]
        # too long: nothing stops, nothing is written
        real = apply.IDLE_WAIT_S
        apply.IDLE_WAIT_S = 0
        try:
            def always_busy(url, body=None, timeout=10):
                return {"status": "ok", "inflight": {"running": 1}}
            rec = Recorder()
            before = open(apply.overlay_path()).read()
            jid = _job(_sel(slots=2))
            assert apply.Runner(jid, cmd=rec, http=always_busy, ops=box.ops,
                                sleep=lambda s: None).main() == 1
            job = apply.read_job(jid)
            assert job["failed_step"] == "write_config" and "nothing changed" in job["message"]
            assert rec.calls == [] and open(apply.overlay_path()).read() == before
        finally:
            apply.IDLE_WAIT_S = real
    # a port change that fails: the new port is stopped under the new overlay, then the old one
    # comes back; use-profile.sh never rolls back itself; the rollback's smoke names the old model
    with Box() as box:
        apply.write_overlay(M, _sel(slots=3, advanced={"served_model": "old-name"},
                                    components={"conductor": False, "narrator": False,
                                                "laya": False, "shadow": False}))
        old = open(apply.overlay_path()).read()
        seen = []
        rec = Recorder(rc={"use-profile.sh exl3-4.5": 1})

        def look(argv, **kw):
            with open(apply.overlay_path()) as fh:
                seen.append((" ".join(os.path.basename(a) for a in argv[1:]),
                             "PORT='8124'" in fh.read()))
            return rec(argv, **kw)

        jid = _job(_sel(advanced={"port": 8124}))
        assert apply.Runner(jid, cmd=look, http=_smoke_ok(), ops=box.ops).main() == 1
        assert seen == [("stop.sh 120", False), ("use-profile.sh exl3-4.5", True),
                        ("stop.sh 120", True), ("use-profile.sh exl3-6.0", False)], seen
        assert open(apply.overlay_path()).read() == old
        assert all(c["env"].get("QSE_NO_ROLLBACK") == "1" for c in rec.calls
                   if os.path.basename(c["argv"][1]) == "use-profile.sh")
        assert rec.calls[-1]["env"]["SMOKE_MODEL"] == "old-name", rec.calls[-1]["env"]
        assert rec.calls[-1]["env"]["SMOKE_URL"] == "http://127.0.0.1:8000/v1"
        # a failure after the old engine stopped (make-secrets) goes through the rollback
        rec = Recorder(rc={"make-secrets.sh": 1})
        jid = _job(_sel(advanced={"port": 8124, "admin_password": "generate"}))
        assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main() == 1
        seq = [" ".join(os.path.basename(a) for a in c["argv"][1:]) for c in rec.calls]
        assert seq == ["stop.sh 120", "make-secrets.sh --admin", "stop.sh 120",
                       "use-profile.sh exl3-6.0"], seq
        job = apply.read_job(jid)
        assert job["failed_step"] == "write_config" and job["state"] == "rolled_back", job
        assert open(apply.overlay_path()).read() == old
    # setup mode: the marker (with its HOST) comes back unchanged, on a rollback and after a crash
    with Box() as box:
        pending = os.path.join(box.root, "run", "setup-pending")
        with open(pending, "w") as fh:
            fh.write("HOST=0.0.0.0\n")
        # a stale marker beside a serving engine: the plan says so
        assert any("run/setup-pending" in w for w in _plan()["warnings"])
        assert not any("run/setup-pending" in w for w in _plan(setup_mode=True)["warnings"])
        jid = _job(_sel(), setup_mode=True, previous=None)
        assert apply.Runner(jid, cmd=Recorder(rc={"use-profile.sh exl3-4.5": 1}),
                            http=_smoke_ok(), ops=box.ops).main() == 1
        assert open(pending).read() == "HOST=0.0.0.0\n"

        def crash(argv, **kw):
            if os.path.basename(argv[1]) == "use-profile.sh":
                raise RuntimeError("the runner died here")
            return 0

        jid = _job(_sel(), setup_mode=True, previous=None)
        assert apply.Runner(jid, cmd=crash, http=_smoke_ok(), ops=box.ops).main() == 1
        assert open(pending).read() == "HOST=0.0.0.0\n"
        # a good install drops it for good
        bodies = []

        def ok(url, body=None, timeout=10):
            if body is not None:
                bodies.append(body)
            return _smoke_ok()(url, body, timeout)

        jid = _job(_sel(), setup_mode=True, previous=None)
        assert apply.Runner(jid, cmd=Recorder(), http=ok, ops=box.ops).main() == 0
        assert not os.path.exists(pending)
        assert bodies and bodies[-1]["max_tokens"] == 512


def test_timeouts_liveness_and_quoting():
    with Box() as box:
        pidfile = os.path.join(box.tmp, "child.pid")
        script = os.path.join(box.tmp, "slow.sh")
        with open(script, "w") as fh:
            fh.write(f'#!/bin/bash\nsleep 60 &\necho $! > "{pidfile}"\nwait\n')
        lines = []
        t0 = time.monotonic()
        rc = apply._run_cmd(["bash", script], timeout=1.5, env={"PATH": os.environ["PATH"]},
                            cwd=box.tmp, line=lines.append)
        assert rc == 124 and time.monotonic() - t0 < 10
        child = int(open(pidfile).read())
        time.sleep(0.3)
        try:
            os.kill(child, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        assert not alive, "the script's child outlived the timeout"
        # liveness: the same pid with another start time or another boot is not the runner
        real = apply._proc_identity
        apply._proc_identity = lambda pid: {"pid_start": 111, "boot_id": "b1"}
        try:
            assert apply._alive(os.getpid(), 111, "b1")
            assert not apply._alive(os.getpid(), 222, "b1")
            assert not apply._alive(os.getpid(), 111, "b2")
            assert apply._alive(os.getpid())                  # an older job without them
        finally:
            apply._proc_identity = real
        # quoting: a value with ; and a space stays one value when serve.env sources it
        sel = _sel()
        m = json.loads(json.dumps(M))
        m["components"][6]["knobs"]["VISION"]["on"] = "a b;touch " + os.path.join(box.tmp, "pwn")
        apply.write_overlay(m, sel)
        got = apply.resolve_env(["VISION"])
        assert got["VISION"].startswith("a b;touch"), got
        assert not os.path.exists(os.path.join(box.tmp, "pwn"))


def test_the_current_selection():
    with Box() as box:
        apply.write_overlay(M, _sel(slots=2, window=65_536, components={"vision": False},
                                    advanced={"port": 8124, "served_model": "mine"}))
        app.STATE.clear()
        app.STATE.update(auth=auth.Auth(None, None), started=time.time(), wren=FakeWren())
        code, b = _api("GET", f"/v1/setup/models/{M['id']}/current")
        assert code == 200, b
        sel = b["selection"]
        assert sel["profile"] == "exl3-6.0" and sel["slots"] == 2 and sel["window"] == 65_536
        assert sel["components"]["vision"] is False and sel["advanced"]["port"] == 8124
        assert sel["advanced"]["served_model"] == "mine" and b["unmatched"] == [], b
        # the plan from that seed changes nothing; a port and a name change are warned about
        p = _plan(estimate.check(M, sel))
        assert p["changes"] == [], p["changes"]
        p = _plan(estimate.check(M, dict(sel, advanced=dict(sel["advanced"], port=8125,
                                                                served_model="other"))))
        assert any(w.startswith("PORT changes from 8124 to 8125") for w in p["warnings"]), p
        assert any(w.startswith("SERVED_MODEL changes from mine to other") for w in p["warnings"])
        # a HOST the registry has no option for stays recommended and is listed
        with open(apply.overlay_path(), "a") as fh:
            fh.write("HOST='192.168.1.143'\n")
        b = apply.current_selection(M)
        assert b["selection"]["advanced"]["host"] == "0.0.0.0"
        assert {"field": "advanced.host", "value": "192.168.1.143"} in b["unmatched"], b


def test_the_current_selection_of_unset_keys():
    # a box whose serve.env predates the setup guide's keys (the reference box's): an unset key is
    # what the engine then runs with, the lookup store's default folder or start.sh's default
    with Box() as box:
        shutil.copy(os.path.join(REPO, "ops", "start.sh"), box.ops)
        env_file = os.path.join(box.ops, "serve.env")
        with open(env_file) as fh:
            lines = fh.read().splitlines()
        gone = ("VISION=", "RESPONSE_CACHE=", "SESSION_SPILL=", "QSE_BACKENDS=")
        with open(env_file, "w") as fh:
            fh.write("\n".join(x for x in lines if not x.startswith(gone)) + "\n")
        b = apply.current_selection(M)
        comps = b["selection"]["components"]
        assert comps["lookup"] is True and comps["vision"] is True, b
        assert comps["response_cache"] is False and comps["nvme_spill"] is True, b
        assert comps["proxy"] is False, comps         # start.sh's ${QSE_BACKENDS:-off}
        assert b["unmatched"] == [], b["unmatched"]
        # Review from there does not list them: the overlay writes what the engine runs with anyway
        p = _plan(estimate.check(M, b["selection"]))
        listed = {c["key"] for c in p["changes"]}
        assert not listed & {"QWEN38_SUFFIX_STORE", "VISION", "RESPONSE_CACHE", "SESSION_SPILL"}, \
            p["changes"]
        # the overlay of that selection leaves the engine's environment as it was: every key it
        # resolves to the same value, and no QWEN38_* variable is set that was not (the disk tier's
        # fingerprint counts them, and a new one swept the profile's saved chats at the switch)
        sel = estimate.check(M, b["selection"])
        keys = list(apply.overlay_values(M, sel))
        before = apply.resolve_env(keys)
        apply.write_overlay(M, sel)
        text = open(apply.overlay_path()).read()
        for k in ("QWEN38_SUFFIX_STORE", "VISION", "RESPONSE_CACHE", "SESSION_SPILL"):
            assert f"\n{k}=" not in text, text
        written = sum(1 for x in text.splitlines() if x and not x.startswith("#"))
        assert f"write run/setup.env ({written} keys;" in p["commands"][0], (written, p["commands"])
        after = apply.resolve_env(keys)
        assert {k: v for k, v in after.items() if k.startswith("QWEN38_")} == \
            {k: v for k, v in before.items() if k.startswith("QWEN38_")}
        assert apply.current_selection(M)["selection"] == b["selection"]
        # a change the user asks for is written: the store off
        apply.write_overlay(M, estimate.check(M, dict(b["selection"], components=dict(
            comps, lookup=False))))
        assert "QWEN38_SUFFIX_STORE=''" in open(apply.overlay_path()).read()
        assert apply.current_selection(M)["selection"]["components"]["lookup"] is False
        os.unlink(apply.overlay_path())
        # an explicit empty folder in serve.env is the store off, as before
        with open(env_file, "a") as fh:
            fh.write("QWEN38_SUFFIX_STORE=\n")
        assert apply.current_selection(M)["selection"]["components"]["lookup"] is False


def test_an_install_sh_box():
    with Box() as box:
        run_ops = os.path.join(box.tmp, "install", "run", "ops")
        os.makedirs(run_ops)
        shutil.copy(os.path.join(box.ops, "serve.env"), run_ops)
        for sc in apply.SCRIPTS + ("engines.sh",) + apply.PREFLIGHT_PY:
            shutil.copy(os.path.join(box.ops, sc), run_ops)
        os.environ["QSE_OPS_DIR"] = run_ops
        device_mod.clear()
        try:
            assert device_mod.repo_ops() == box.ops
            assert device_mod.active_profile() == "exl3-6.0"
            rec = Recorder()
            jid = _job(_sel())
            assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=run_ops).main() == 0, \
                apply.read_job(jid)
            assert rec.calls[0]["argv"][1] == os.path.join(run_ops, "use-profile.sh")
        finally:
            os.environ["QSE_OPS_DIR"] = box.ops
            device_mod.clear()


def test_benchmarks_ignore_the_overlay():
    """tools/exl3_fastrun.py --env-file reads serve.env's QWEN38_* knobs without run/setup.env."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import exl3_fastrun
    with Box() as box:
        env_file = os.path.join(box.ops, "serve.env")
        before = exl3_fastrun.env_from_file(env_file)
        apply.write_overlay(M, _sel(methods={"draft_temperature": 0.9}))
        assert apply.resolve_env(["QWEN38_DRAFT_TEMP"])["QWEN38_DRAFT_TEMP"] == "0.9"
        after = exl3_fastrun.env_from_file(env_file)
        assert after == before and after.get("QWEN38_DRAFT_TEMP") != "0.9", after


class Req(app.Handler):
    def __init__(self, method, path, body=None, headers=None):
        raw = json.dumps(body if body is not None else {}).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = dict({"Content-Length": str(len(raw)), "Content-Type": "application/json"},
                            **(headers or {}))
        self.path, self.command, self.request_version = path, method, "HTTP/1.1"
        self.requestline, self.client_address = f"{method} {path}", ("127.0.0.1", 0)
        self.close_connection = True

    def go(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            getattr(self, f"do_{self.command}")()
        head, _, raw = self.wfile.getvalue().partition(b"\r\n\r\n")
        return int(head.decode().split(" ", 2)[1]), head.decode(), raw.decode()


def _api(method, path, body=None, headers=None):
    code, head, raw = Req(method, path, body, headers).go()
    try:
        return code, json.loads(raw)
    except ValueError:
        return code, raw


class FakeWren:
    def __init__(self):
        self.unloads = []

    def unload(self, why="asked"):
        self.unloads.append(why)

    def status(self):
        return {"name": "Wren", "state": "not_loaded"}


def test_the_routes():
    with Box():
        app.STATE.clear()
        w = FakeWren()
        app.STATE.update(auth=auth.Auth(None, None), started=time.time(), wren=w)
        app.INFLIGHT.update(running=0, waiting=0)
        spawned = []
        real = apply.spawn
        apply.spawn = lambda jid: spawned.append(jid) or 4242
        try:
            sel = {"model": M["id"]}
            for extra in ({"steps": []}, {"cmd": "id"}, {"script": "x"}, {"env": {}},
                          {"path": "/tmp"}):
                code, b = _api("POST", "/v1/setup/apply", dict({"selection": sel}, **extra))
                assert code == 422 and b["error"]["code"] == "unknown_field", (extra, b)
            for bad in ({"profile": "../../etc"}, {"advanced": {"served_model": "a;b"}}):
                code, b = _api("POST", "/v1/setup/apply", {"selection": dict(sel, **bad)})
                assert code in (400, 422), (bad, code, b)
            code, b = _api("POST", "/v1/setup/apply", {"selection": sel, "dry_run": True})
            assert code == 200 and b["dry_run"] is True and b["ok"], b
            assert not os.path.exists(apply.jobs_dir()) or not os.listdir(apply.jobs_dir())
            code, b = _api("POST", "/v1/setup/plan", sel)
            assert code == 200 and b["dry_run"] is False and b["changes"]
            app.INFLIGHT["running"] = 1
            code, b = _api("POST", "/v1/setup/apply", {"selection": sel})
            assert code == 409 and b["error"]["code"] == "busy" and b["error"]["running"] == 1
            app.INFLIGHT["running"] = 0
            code, b = _api("POST", "/v1/setup/apply", {"selection": dict(sel, kv_pool=False)})
            assert code == 409 and b["error"]["code"] == "over_budget" and "plan" in b
            # VIS-199: the engine's limit, 16 parallel chats, whatever the card measured
            code, b = _api("POST", "/v1/setup/apply", {"selection": dict(sel, slots=17)})
            assert code == 422 and b["error"]["code"] == "out_of_range", b
            assert "above 16" in b["error"]["message"], b
            code, b = _api("POST", "/v1/setup/apply", {"selection": sel})
            assert code == 202 and b["job"]["state"] == "queued", b
            assert spawned == [b["job"]["id"]] and w.unloads == ["apply"]
            code, b2 = _api("POST", "/v1/setup/apply", {"selection": sel})
            assert code == 409 and b2["error"]["code"] == "job_running", b2
        finally:
            apply.spawn = real
    with Box(assets=False):
        app.STATE.update(wren=FakeWren())
        real_get = registry.get
        registry.get = lambda mid, d=None: _no_download(M, "base-plain") if mid == M["id"] else real_get(mid, d)
        try:
            code, b = _api("POST", "/v1/setup/apply", {"selection": {"model": M["id"]}})
        finally:
            registry.get = real_get
        assert code == 412 and b["error"]["code"] == "missing_assets", b


def test_the_job_route():
    with Box() as box:
        app.STATE.clear()
        app.STATE.update(auth=auth.Auth(None, None), started=time.time(), wren=FakeWren())
        jid = _job(_sel())
        apply.Runner(jid, cmd=Recorder(out=["a line"]), http=_smoke_ok(), ops=box.ops).main()
        code, j = _api("GET", f"/v1/setup/jobs/{jid}?format=json")
        assert code == 200 and j["state"] == "done" and j["id"] == jid
        code, head, raw = Req("GET", f"/v1/setup/jobs/{jid}", None,
                              {"Accept": "text/event-stream"}).go()
        assert code == 200 and "text/event-stream" in head
        frames = [f for f in raw.split("\n\n") if f.strip()]
        evs = [json.loads(f.split("data: ", 1)[1]) for f in frames]
        ids = [int(f.split("\n")[0][4:]) for f in frames if f.startswith("id: ")]
        # VIS-149: the log on disk, the job as it is now, its done event
        assert [e["type"] for e in evs[-2:]] == ["job", "done"], evs[-2:]
        assert evs[-2]["job"]["state"] == "done" and {e["type"] for e in evs[:-2]} == {"log"}
        assert ids == sorted(ids) and ids[-1] == len(apply.events(jid))
        assert any(e["type"] == "log" and e["line"] == "a line" for e in evs)
        code, _, raw = Req("GET", f"/v1/setup/jobs/{jid}", None,
                           {"Accept": "text/event-stream", "Last-Event-ID": str(ids[-2])}).go()
        again = [f for f in raw.split("\n\n") if f.strip()]
        assert len(again) == 2 and again[1].startswith(f"id: {ids[-1]}"), again
        assert json.loads(again[0].split("data: ", 1)[1])["job"]["state"] == "done"
        # VIS-211: a job whose newest event is a log line: the snapshot carries no id, so a page
        # that counted the line's id does not drop it as a repeat (it held no job and showed
        # "Starting the install" over a finished one)
        jid2 = _job(_sel())
        r2 = apply.Runner(jid2, cmd=Recorder(), http=_smoke_ok(), ops=box.ops)
        r2.step("check", "running")
        r2.log("the newest line")
        frames2 = _catch_up_frames(jid2)
        snap = [f for f in frames2 if '"type": "job"' in f]
        assert len(snap) == 1 and not snap[0].startswith("id: "), frames2
        assert frames2[-2].startswith(f"id: {len(apply.events(jid2))}") and "the newest line" in frames2[-2]
        r2.step("check", "done")
        frames3 = _catch_up_frames(jid2)
        assert frames3[-1].startswith(f"id: {len(apply.events(jid2))}") and '"type": "job"' in frames3[-1]
        for bad in ("0123456789abcdef", "../../etc", "nope"):
            code, _ = _api("GET", f"/v1/setup/jobs/{bad}?format=json")
            assert code == 404, bad
        # a follower reads each line once from its offset; a half-written line waits
        recs, off = apply.events_from(jid)
        assert [r["seq"] for r in recs] == list(range(1, ids[-1] + 1)) and off == os.path.getsize(
            os.path.join(apply.job_dir(jid), "events.jsonl"))
        with open(os.path.join(apply.job_dir(jid), "events.jsonl"), "a") as fh:
            fh.write('{"seq": 999, "event": {"type": "lo')
        more, off2 = apply.events_from(jid, off)
        assert more == [] and off2 == off
        with open(os.path.join(apply.job_dir(jid), "events.jsonl"), "a") as fh:
            fh.write('g", "line": "x", "ts": 1}}\n')
        more, off3 = apply.events_from(jid, off2)
        assert [r["seq"] for r in more] == [999] and off3 > off2


def _catch_up_frames(jid: str) -> list[str]:
    """The frames setup_api's catch-up writes for a running job (the stream itself would follow)."""
    from server import setup_api

    class H:
        wfile = io.BytesIO()
    setup_api._catch_up(H, jid, 0)
    return [f for f in H.wfile.getvalue().decode().split("\n\n") if f.strip()]


def test_secrets_are_redacted():
    with Box() as box:
        os.environ["HF_TOKEN"] = HF
        jid = _job(_sel())
        rec = Recorder(out=[f"token {HF} here", "Authorization: Bearer abcdefabcdefabcdefabcdef12",
                            "hf_" + "A1b2C3" * 6])
        apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main()
        os.environ.pop("HF_TOKEN")
        blob = open(os.path.join(apply.job_dir(jid), "events.jsonl")).read() + \
            open(os.path.join(apply.job_dir(jid), "state.json")).read()
        for s in (HF, "abcdefabcdefabcdefabcdef12", "A1b2C3A1b2C3"):
            assert s not in blob, s
        assert "[redacted" in blob


class Engine(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self._send({"status": "ok", "weights": {"name": "exl3-4.5"}})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send({"choices": [{"message": {"content": "ready"}}]})

    def _send(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


def test_the_detached_runner():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old_pp = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = REPO
    try:
        with Box(port=port) as box:
            jid = _job(_sel(advanced={"port": port}))
            # the runner is a process of its own: this machine's memory is not the box's (the
            # preflight and use-profile.sh read MEM_NEED_GB)
            os.environ["MEM_NEED_GB"] = "0"
            try:
                pid = apply.spawn(jid)
            finally:
                os.environ.pop("MEM_NEED_GB", None)
            t0 = time.time()
            while time.time() - t0 < 60:
                j = apply.read_job(jid)
                if j["state"] not in ("queued", "running"):
                    break
                time.sleep(0.2)
            out = open(os.path.join(apply.job_dir(jid), "runner.out")).read()
            assert j["state"] == "done", (j, out)
            assert j["pid"] == pid
            assert os.getsid(0) != _sid(pid, os.getsid(0))
            evs = [json.loads(x)["event"] for x in open(os.path.join(
                apply.job_dir(jid), "events.jsonl"))]
            assert any(e.get("line", "").startswith("[use-profile.sh] exl3-4.5") for e in evs), evs
            # the status responder answers for the job while the engine is away
            r = subprocess.run([sys.executable, "-c", (
                "import sys, json; sys.path.insert(0, sys.argv[1]); import importlib.util as u; "
                "s = u.spec_from_file_location('es', sys.argv[1] + '/ops/engine_status.py'); "
                "m = u.module_from_spec(s); s.loader.exec_module(m); m.REPO = sys.argv[2]; "
                "import io; h = m.Handler.__new__(m.Handler); h.command = 'GET'; "
                "h.path = '/v1/setup/jobs/' + sys.argv[3]; h.wfile = io.BytesIO(); "
                "h.request_version = 'HTTP/1.1'; h.requestline = 'GET'; h.client_address = ('x', 0); "
                "h.headers = {}; h._send = lambda c, b, t, r, **k: print(c, b.decode()); h._answer()"),
                REPO, box.root, jid], capture_output=True, text=True, timeout=30)
            code, _, body = r.stdout.partition(" ")
            assert code == "200" and json.loads(body)["id"] == jid, r.stdout + r.stderr
    finally:
        srv.shutdown()
        if old_pp is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pp


def _sid(pid, default):
    try:
        return os.getsid(pid)
    except ProcessLookupError:
        return default + 1          # gone already: it ran in a session of its own


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
