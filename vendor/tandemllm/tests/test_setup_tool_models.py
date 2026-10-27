"""The models of the tools a selection turns on (VIS-153): the plan, the fetch step, the smoke step's
role check and the shadow reader's retry after a failed load. On a CPU against a checkout in a
temporary folder (tests/test_setup_apply.py's Box); no network, no GPU.

The test matrix, the Gherkin of VIS-153 (the page's half is dashboard/tests/setup-install.test.ts):

  1  missing tool weights that can be fetched: the shadow reader on, Qwen3.5-0.8B not on the box;
     the plan lists it as a download with its size, the fetch step passes it to use-profile.sh
     (QSE_FETCH_ASSETS); Qwen3.5-2B, which the narrator and Wren share, is listed once
                                                    test_missing_tool_weights_that_can_be_fetched
  2  missing tool weights that cannot be fetched: an error that names the asset and the Tools
     step, code missing_assets (the review's fix link goes to the Tools step)
                                                    test_missing_tool_weights_that_cannot_be_fetched
  3  a tool fails after the restart: the job ends installed with a warning naming the tool and
     its reason; one the planner refused is a note (waiting for memory), not a failure
                                                    test_a_tool_fails_after_the_restart
  4  ops/fetch_weights.py --assets --also: an unknown id is exit 3; assets on the box are not
     fetched again; ops/use-profile.sh hands QSE_FETCH_ASSETS to it, also for a profile with its
     own QSE_ASSETS (LFM2.5)                        test_fetch_weights_also,
                                                    test_use_profile_passes_the_tools_models
  5  the shadow reader's failed load is tried again in an idle window once SHADOW_RETRY_S passed,
     never before; /health roles carry the load error
                                                    test_a_failed_shadow_load_is_tried_again

Run: python tests/test_setup_tool_models.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_setup_apply as T  # noqa: E402
from server.setup import apply, registry  # noqa: E402

M = T.M


def _remove(box, aid):
    a = registry.asset(M, aid)
    for p in [a["path"], *a["also"]]:
        os.unlink(p.replace("${HOME}", box.home))


def test_missing_tool_weights_that_can_be_fetched():
    with T.Box() as box:
        _remove(box, "qwen3.5-0.8b")
        p = T._plan()
        assert p["ok"], p["errors"]
        assert p["fetch_assets"] == ["qwen3.5-0.8b"], p["fetch_assets"]
        d = [x for x in p["downloads"] if x["id"] == "qwen3.5-0.8b"]
        assert d == [{"id": "qwen3.5-0.8b", "label": "Qwen3.5-0.8B (the shadow reader)",
                      "gb": 1.77, "for": "Shadow reader"}], p["downloads"]
        assert p["profile"]["on_disk"] is False and p["profile"]["download_gb"] == 1.77
        assert any("--fetch-only" in c and "Qwen3.5-0.8B" in c for c in p["commands"])
        # the fetch step hands the id to use-profile.sh, then the job goes on
        rec = T.Recorder()
        jid = T._job(T._sel())
        assert apply.Runner(jid, cmd=rec, http=T._smoke_ok(), ops=box.ops).main() == 0
        fetch = [c for c in rec.calls if "--fetch-only" in c["argv"]]
        assert len(fetch) == 1 and fetch[0]["env"]["QSE_FETCH_ASSETS"] == "qwen3.5-0.8b", fetch
        # the shadow reader off: nothing to fetch
        p = T._plan(T._sel(components={"shadow": False}))
        assert p["fetch_assets"] == [] and p["profile"]["on_disk"] is True
    with T.Box() as box:
        # the narrator's model is Wren's too: one download, named once
        _remove(box, "qwen3.5-2b")
        _remove(box, "laya")
        p = T._plan()
        assert p["fetch_assets"] == ["qwen3.5-2b", "laya"], p["fetch_assets"]
        assert [x["id"] for x in p["downloads"]] == ["qwen3.5-2b", "laya"], p["downloads"]
        assert p["profile"]["download_gb"] == round(4.57 + 2.35, 2), p["profile"]


def test_missing_tool_weights_that_cannot_be_fetched():
    with T.Box() as box:
        _remove(box, "qwen3.5-0.8b")
        p = apply.plan(T._no_download(M, "qwen3.5-0.8b"), T._sel(), setup_mode=False,
                       password_set=False)
        assert not p["ok"] and p["error_codes"] == ["missing_assets"], p["errors"]
        e = p["errors"][0]
        assert "Shadow reader is on" in e and "Qwen3.5-0.8B (the shadow reader)" in e, e
        assert "Turn Shadow reader off in the Tools step" in e, e
        p = apply.plan(T._no_download(M, "qwen3.5-0.8b"), T._sel(components={"shadow": False}),
                       setup_mode=False, password_set=False)
        assert p["ok"], p["errors"]


def _health(roles, models=None):
    def http(url, body=None, timeout=10):
        if url.endswith("/health"):
            return {"status": "ok", "weights": {"name": "exl3-4.5"},
                    "roles": {"dynamic": True, "roles": roles}}
        if url.endswith("/v1/models"):
            return {"data": [], **({"decision_models": models} if models is not None else {})}
        return {"choices": [{"message": {"content": "ready"}}]}
    return http


def test_a_tool_fails_after_the_restart():
    err = ("FileNotFoundError: no config.json under "
           "/x/.qwen38-spark-engine/models/qwen3.5-0.8b")
    with T.Box() as box:
        jid = T._job(T._sel())
        http = _health({"shadow": {"models": ["qwen3.5-0.8b"], "state": "failed", "error": err},
                        "narrator": {"models": ["qwen3.5-2b"], "state": "loaded"}},
                       [{"id": "laya-english", "state": "loaded"}])
        r = apply.Runner(jid, cmd=T.Recorder(), http=http, ops=box.ops, sleep=lambda s: None)
        assert r.main() == 0
        j = apply.read_job(jid)
        assert j["state"] == "done", j
        assert j["warnings"] == [f"Shadow reader did not load ({err})."], j["warnings"]
        assert j["message"] == f"exl3-4.5 is live on port 8000. Shadow reader did not load ({err}).", \
            j["message"]
        smoke = next(s for s in j["steps"] if s["id"] == "smoke")
        assert smoke["detail"] == "exl3-4.5 answers on port 8000; 1 tool did not load", smoke
        # a page that follows the job live gets them with the done event
        done = apply.events(jid)[-1]["event"]
        assert done["type"] == "done" and done["warnings"] == j["warnings"], done
    with T.Box() as box:
        # refused for memory: a note, not a failure; a role still loading is polled, then named
        jid = T._job(T._sel())
        polls = []

        def http(url, body=None, timeout=10):
            if url.endswith("/health"):
                polls.append(url)
            return _health({"shadow": {"models": ["qwen3.5-0.8b"], "state": "not_loaded",
                                       "refused": "refused_hold"},
                            "narrator": {"models": ["qwen3.5-2b"],
                                         "state": "loaded" if len(polls) > 3 else "loading"}},
                           [{"id": "laya-english", "state": "failed"}])(url, body, timeout)
        r = apply.Runner(jid, cmd=T.Recorder(), http=http, ops=box.ops, sleep=lambda s: None)
        assert r.main() == 0
        j = apply.read_job(jid)
        # VIS-211: plain words for the page, the planner's reason in the log
        assert j["state"] == "done" and j["notes"] == [
            "Shadow reader loads later, when memory frees up; nothing to do."], j
        logs = [e["event"]["line"] for e in apply.events(jid) if e["event"]["type"] == "log"]
        assert "[smoke] Shadow reader: refused (refused_hold)" in logs, logs
        assert j["warnings"] == ["Laya decision models did not load (a decision model failed to "
                                 "load)."], j["warnings"]
        assert 4 <= len(polls) <= apply.ROLE_SETTLE_POLLS + 1, len(polls)
    with T.Box() as box:
        # an engine that reports no roles: as before, nothing added
        jid = T._job(T._sel())
        assert apply.Runner(jid, cmd=T.Recorder(), http=T._smoke_ok(), ops=box.ops,
                            sleep=lambda s: None).main() == 0
        j = apply.read_job(jid)
        assert j["warnings"] == [] and j["notes"] == [] and j["message"] == \
            "exl3-4.5 is live on port 8000.", j


def _fetch_weights():
    spec = importlib.util.spec_from_file_location("fw", os.path.join(T.REPO, "ops",
                                                                     "fetch_weights.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_weights_also():
    fw = _fetch_weights()

    def offline(*a, **kw):
        raise AssertionError("the test asked the hub")
    fw.fetch = fw.fetch_cache = fw.plan = fw.hub_files = offline
    present = registry.asset_present
    with T.Box() as box:
        # the registry of this checkout; ${REPO} paths (the BF16 drafters) in the box's copy
        registry.asset_present = lambda a, repo=None, m=None: present(a, box.root, m)
        os.environ["HF_HUB_CACHE"] = os.path.join(box.home, ".cache", "huggingface", "hub")
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert fw.assets("exl3-4.5", T.REPO, also=["qwen3.5-0.8b", "qwen3.5-2b"]) == 0
                assert fw.assets("exl3-4.5", T.REPO, also=["no-such-model"]) == 3
            text = out.getvalue()
            assert "[fetch] qwen3.5-0.8b: on this box" in text, text
            assert text.count("[fetch] qwen3.5-2b: on this box") == 1, text
            assert "has no asset no-such-model" in text, text
        finally:
            registry.asset_present = present
            os.environ.pop("HF_HUB_CACHE", None)


def test_use_profile_passes_the_tools_models():
    """ops/use-profile.sh --fetch-only hands QSE_FETCH_ASSETS to fetch_weights.py --also, for a
    27B profile and for one that names its own files (QSE_ASSETS: LFM2.5, Flash-Next)."""
    import shutil
    import subprocess
    with tempfile.TemporaryDirectory() as home:
        repo = os.path.join(home, "repo")
        os.makedirs(os.path.join(repo, "ops"))
        shutil.copy(os.path.join(T.REPO, "ops/use-profile.sh"), os.path.join(repo, "ops"))
        shutil.copytree(os.path.join(T.REPO, "ops/profiles"), os.path.join(repo, "ops/profiles"))
        tail = open(os.path.join(T.REPO, "ops/serve.env")).read().split(
            "# Weight profile (ops/use-profile.sh)")[1]
        calls = os.path.join(home, "calls")
        py = os.path.join(home, "py")
        with open(py, "w") as fh:                   # notes every call; every check passes
            fh.write(f'#!/bin/bash\necho "$*" >> {calls}\nexit 0\n')
        os.chmod(py, 0o755)
        with open(os.path.join(repo, "ops/serve.env"), "w") as fh:
            fh.write(f"REPO={repo}\nPY={py}\nPORT=18997\n# Weight profile{tail}")
        for d in ("exl3-quants/4.5", "models/lfm2.5-24b-a2b-exl3-3.5"):
            os.makedirs(os.path.join(home, d))
            for n in ("config.json", "model.safetensors.index.json", "model.safetensors"):
                open(os.path.join(home, d, n), "w").write("{}")
        env = {"HOME": home, "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "QSE_FETCH_ASSETS": "qwen3.5-2b laya"}
        for prof in ("exl3-4.5", "lfm25-exl3-3.5"):
            open(calls, "w").close()
            subprocess.run(["bash", os.path.join(repo, "ops/use-profile.sh"), prof, "--fetch-only"],
                           env=env, capture_output=True, text=True, timeout=60)
            got = [x for x in open(calls).read().splitlines() if "fetch_weights.py --assets" in x]
            assert len(got) == 1 and got[0].endswith("--also qwen3.5-2b laya"), (prof, got)
            assert f"--assets {prof} " in got[0], got
        # without QSE_FETCH_ASSETS the LFM2.5 profile asks fetch_weights.py nothing, as before
        env.pop("QSE_FETCH_ASSETS")
        open(calls, "w").close()
        subprocess.run(["bash", os.path.join(repo, "ops/use-profile.sh"), "lfm25-exl3-3.5",
                        "--fetch-only"], env=env, capture_output=True, text=True, timeout=60)
        assert "fetch_weights.py --assets" not in open(calls).read()


def test_a_failed_shadow_load_is_tried_again():
    from engine.generative.fake import FakeGenerative
    from engine.generative.slot import Slot
    from server import app, rolemem

    class Missing(FakeGenerative):
        missing = True

        def load(self, where):
            if self.missing:
                raise FileNotFoundError("no config.json under /x/qwen3.5-0.8b")
            super().load(where)

    class Store:
        def names(self):
            return ["coding"]

        def peek(self, kind, name=None):
            return 1, {"shadow": {"mode": "on"}}

    os.environ["BOX_LOCK"] = os.path.join(tempfile.gettempdir(), f"vis153-box-lock-{os.getpid()}")
    keep = {k: app.STATE.get(k) for k in ("shadow", "narrator", "conductor", "deciders",
                                          "draining", "shadow_preload", "role_mem")}
    now = [1000.0]
    m = Missing("qwen3.5-0.8b")
    s = Slot(m, role="shadow", gate=threading.Lock(), log=lambda x: None, clock=lambda: now[0])
    try:
        app.STATE.update(shadow=types.SimpleNamespace(slot=s), narrator=None, deciders=None,
                         draining=False, shadow_preload=None, role_mem=None,
                         conductor=types.SimpleNamespace(store=Store()))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert app.shadow_preload_once() == "loading"
            for _ in range(200):
                if s.state == "failed":
                    break
                app.time.sleep(0.01)
            assert s.state == "failed" and s.failed_at == 1000.0, s.state
            # /health roles say why
            rep = app.roles_report()["roles"]["shadow"]
            assert rep["state"] == "failed" and "FileNotFoundError" in rep["error"], rep
            assert rolemem.slot_detail(s)["error"] == rep["error"]
            # not before the retry interval
            now[0] += app.SHADOW_RETRY_S - 1
            assert app.shadow_preload_once() is None and s.state == "failed"
            # the files arrive; after the interval the next idle look loads it
            m.missing = False
            now[0] += 1
            assert app.shadow_preload_once() == "loading"
            for _ in range(200):
                if s.state == "loaded":
                    break
                app.time.sleep(0.01)
        assert s.state == "loaded" and m.loads == 1, (s.state, out.getvalue())
        assert "trying again" in out.getvalue(), out.getvalue()
        assert app.roles_report()["roles"]["shadow"]["error"] is None
    finally:
        app.STATE.update(keep)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
