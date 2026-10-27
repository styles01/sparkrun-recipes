"""A fresh box (OPS-143): fetch_weights.py checks a fetch entry's globs, and install.sh --setup into an
empty HOME leaves a config the EXL3 profiles accept. No GPU and no network: the downloads run against
tests/test_weight_profiles.py's fake hub, and install.sh runs with stand-ins for nvidia-smi, nvcc, uv
and the venv's Python (the 3.11 check and the GPU check pass, everything else is this Python).

The test matrix, the Gherkin of OPS-143:

  1  a fetch entry's names and globs (`*.safetensors`, `encoder/*`) select the files verify() checks
     against the hub's sha256: a matching file is checked, a changed one fails, a missing one fails, a
     file no pattern matches is left alone                       test_verify_reads_globs
  2  a tool model whose entry is all globs (Qwen3.5-0.8B, the shadow reader's) fetches and verifies;
     a changed file under a glob (Laya's encoder/*) fails the fetch and stays out of the folder
                                                                 test_a_tool_model_of_globs_fetches
  3  install.sh --setup on an empty HOME writes QWEN38_MODEL before the weight profile: an EXL3
     profile reads base-plain while the HF cache has no FP8 snapshot and the snapshot once it has one,
     NVFP4 reads the HF cache, a standalone community pack reads its own folder
                                                                 test_a_setup_install_resolves_the_base
  4  on that install, ops/use-profile.sh exl3-4.0 --fetch-only with the pack and its assets on disk
     (base-plain, no FP8 snapshot) passes                         test_a_setup_install_takes_an_exl3_profile
  5  install.sh with HF_HOME: serve.env names that cache, so a later shell without HF_HOME finds the
     FP8 snapshot there and use-profile.sh's fetch_weights.py looks in it
                                                                 test_a_setup_install_with_hf_home_keeps_one_cache

OPS-146 (release 0.10.0): the same install takes Nemotron 3.5 Bolt's default profile

  6  ops/use-profile.sh nemotron-bolt-exl3-2.0 resolves to the public 2.0 pack's folder with LiLiCorr as
     its draft head, and --fetch-only with the pack and the LiLiCorr asset on disk passes
                                                                 test_a_setup_install_takes_the_bolt_profile

ENG-359 (release 0.11.0): the same install takes Nemotron 3.5 Super VL's default profile

  7  ops/use-profile.sh nemotron-super-vl-exl3-4.0 resolves to the public 4.0 pack's folder at the
     revision the registry pins, with LiLiCorr as its draft head; --fetch-only with the pack and the
     LiLiCorr asset on disk passes; no Super VL profile note says "not public" or asks for a token
                                                                 test_a_setup_install_takes_the_super_vl_profile

OPS-149: the setup message

  8  install.sh --setup ends with the Setup Guide's address on the box, the ssh tunnel from another
     computer and --host 0.0.0.0, and no profile to switch to; the Chat tab starts once the setup
     is done (OPS-155)                                                 test_the_setup_message

Run: python tests/test_fresh_install.py   (3 and 4 need Linux: install.sh stops anywhere else)
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_weight_profiles as W  # noqa: E402

BASE_SNAP = ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots"
PLAIN = ".qwen38-spark-engine/weights/base-plain"


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def test_verify_reads_globs():
    sys.path.insert(0, os.path.join(ROOT, "ops"))
    import fetch_weights as fw
    with tempfile.TemporaryDirectory() as d:
        body = {"model-00001-of-00002.safetensors": b"a" * 64, "model-00002-of-00002.safetensors": b"b" * 64,
                "encoder/model.safetensors": b"e" * 64, "tokenizer.json": b"{}", "mtp.safetensors": b"m" * 64}
        for n, b in body.items():
            _write(os.path.join(d, n), b)
        files = {n: {"size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in body.items()}
        files["config.json"] = {"size": 2, "sha256": None}
        json.dump({"repo": "r", "revision": None, "commit": "c", "files": files},
                  open(os.path.join(d, fw.MANIFEST), "w"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert fw.verify(d, ["*.json", "model-*.safetensors", "encoder/*"]) == []
            assert fw.verify(d, ["tokenizer.json"]) == []          # a name, as before
        assert "[fetch] 4 files checked" in out.getvalue() and "[fetch] 1 files checked" in out.getvalue(), \
            out.getvalue()
        # a file outside the patterns is not checked, one under a glob is
        open(os.path.join(d, "mtp.safetensors"), "ab").write(b"x")
        open(os.path.join(d, "encoder/model.safetensors"), "ab").write(b"x")
        with contextlib.redirect_stdout(io.StringIO()):
            bad = fw.verify(d, ["model-*.safetensors", "encoder/*"])
        assert [b.split(":")[0] for b in bad] == ["encoder/model.safetensors"], bad
        os.remove(os.path.join(d, "model-00002-of-00002.safetensors"))
        with contextlib.redirect_stdout(io.StringIO()):
            bad = fw.verify(d, ["model-*.safetensors"])
        assert bad == ["missing model-00002-of-00002.safetensors"], bad


def test_a_tool_model_of_globs_fetches():
    fw, tmp, home, repo, old = W._fetch_box()
    hub = W._Hub(fw, os.path.join(home, ".cache", "huggingface", "hub"))
    for r, names in {"Qwen/Qwen3.5-0.8B": ["config.json", "tokenizer.json", "tokenizer_config.json",
                                           "chat_template.jinja", "merges.txt", "LICENSE",
                                           "model.safetensors-00001-of-00001.safetensors", "README.md"],
                     "convaiinnovations/laya": ["config.json", "rl_agent_config.json", "model.safetensors",
                                                "encoder/model.safetensors", "encoder/config.json",
                                                "tokenizer/tokenizer.json", "multilingual/model.safetensors",
                                                "typed-decisions/model.safetensors", "laya.py"]}.items():
        for n in names:
            hub.body[(r, n)] = f"{r}/{n}".encode() * 3
            hub.sha[(r, n)] = hashlib.sha256(hub.body[(r, n)]).hexdigest()
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = fw.assets("exl3-4.5", repo, margin_gb=0, also=["qwen3.5-0.8b"])
        assert rc == 0, out.getvalue()
        small = os.path.join(home, ".qwen38-spark-engine/models/qwen3.5-0.8b")
        assert set(os.listdir(small)) == {"LICENSE", "README.md", fw.MANIFEST, "chat_template.jinja", "config.json",
                                          "merges.txt", "model.safetensors-00001-of-00001.safetensors",
                                          "tokenizer.json", "tokenizer_config.json"}, os.listdir(small)
        assert f"[fetch] 2 files checked in {small}.partial-qwen3.5-0.8b: OK" in out.getvalue(), out.getvalue()
        # Laya: a changed file under encoder/* fails the fetch, and the folder the engine reads stays absent
        hub.bad = {"encoder/model.safetensors"}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert fw.assets("exl3-4.5", repo, margin_gb=0, also=["laya"]) == 5, out.getvalue()
        laya = os.path.join(home, ".qwen38-spark-engine/deciders/laya")
        assert "encoder/model.safetensors: " in out.getvalue() and not os.path.exists(laya), out.getvalue()
        hub.bad = set()
        with contextlib.redirect_stdout(io.StringIO()):
            assert fw.assets("exl3-4.5", repo, margin_gb=0, also=["laya"]) == 0
        assert os.path.isfile(os.path.join(laya, "encoder/model.safetensors"))
        assert not os.path.exists(os.path.join(laya, "laya.py"))
    finally:
        W._fetch_box_close(tmp, old, hub)


# ---------------------------------------------------------------- install.sh --setup on an empty HOME
_STUBS = {
    # a DGX Spark's answers (GB10, sm_121, driver 580)
    "bin/nvidia-smi": """#!/bin/bash
case "$*" in
  -L) echo "GPU 0: NVIDIA GB10 (UUID: GPU-00000000-0000-0000-0000-000000000000)" ;;
  *query-gpu=name,driver_version,compute_cap,memory.total*) echo "NVIDIA GB10, 580.95.05, 12.1, [N/A]" ;;
  *) exit 0 ;;
esac
""",
    "cuda/bin/nvcc": '#!/bin/bash\necho "Cuda compilation tools, release 13.0, V13.0.88"\n',
    "bin/uv": '#!/bin/bash\n[ "$1" = --version ] && echo "uv 0.12.12"\nexit 0\n',
    "bin/g++": "#!/bin/bash\nexit 0\n",
}
# the venv's Python: the 3.11 check and the GPU check pass, the rest runs on this Python
_PY = """#!/bin/bash
if [ "$1" = -c ] && [[ "$2" == *version_info* ]]; then exit 0; fi
if [ "$1" = - ]; then
    s="$(cat)"
    case "$s" in *"import torch"*) echo "    ok torch (stand-in)"; exit 0 ;; esac
    shift; exec {py} -c "$s" "$@"
fi
exec {py} "$@"
"""
_BOX: dict = {}


def _box(name: str = "plain", hf_home: str | None = None) -> dict | None:
    """install.sh --setup --no-start into a HOME of its own (`hf_home`: HF_HOME under it), once per
    name; the release is this checkout's files as one commit of a repository of their own. None off
    Linux."""
    if platform.system() != "Linux":
        return None
    if name in _BOX:
        return _BOX[name]
    if not _BOX:
        tmp = tempfile.mkdtemp(prefix="fresh-install-")
        rel, stubs = os.path.join(tmp, "release"), os.path.join(tmp, "stubs")
        skip = shutil.ignore_patterns("node_modules", "__pycache__", "*.pdf")
        shutil.copytree(ROOT, rel, symlinks=True, ignore=lambda d, names: set(skip(d, names)) | (
            {".git", "run", "logs", "train"} & set(names) if os.path.samefile(d, ROOT) else set()))
        git = ["git", "-C", rel, "-c", "user.name=t", "-c", "user.email=t@example.com"]
        for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "release"]):
            subprocess.run(git + cmd, check=True, capture_output=True)
        for p, text in _STUBS.items():
            _write(os.path.join(stubs, p), text.encode())
            os.chmod(os.path.join(stubs, p), 0o755)
        _BOX["_"] = {"tmp": tmp, "rel": rel, "stubs": stubs, "ref": subprocess.run(
            git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()}
    rel, stubs, ref = (_BOX["_"][k] for k in ("rel", "stubs", "ref"))
    home = os.path.join(_BOX["_"]["tmp"], name)
    d = os.path.join(home, "TandemLLM")
    _write(os.path.join(d, "venv/bin/python"), _PY.format(py=sys.executable).encode())
    os.chmod(os.path.join(d, "venv/bin/python"), 0o755)
    for f in ("config.json", "tokenizer.json"):          # Wren's model, so nothing is downloaded
        _write(os.path.join(home, ".qwen38-spark-engine/models/qwen3.5-2b", f), b"{}")
    env = {"HOME": home, "PATH": os.path.join(stubs, "bin") + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
           "CUDA_HOME": os.path.join(stubs, "cuda"), "LANG": "C.UTF-8"}
    if hf_home:
        env["HF_HOME"] = os.path.join(home, hf_home)
    r = subprocess.run(["bash", os.path.join(ROOT, "install.sh"), "--setup", "--no-start", "--yes", "--dir", d,
                        "--repo", rel, "--ref", ref, "--port", "18996", "--host", "127.0.0.1"],
                       env=env, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    env.pop("HF_HOME", None)            # what follows runs as a later shell would, without it
    _BOX[name] = {"home": home, "dir": d, "env": env, "out": r.stdout}
    return _BOX[name]


def _resolve(box: dict, profile: str, keys=("QWEN38_MODEL",), extra: dict | None = None) -> dict:
    script = "set -a; . \"$1\"; " + "; ".join(f'echo "{k}=${{{k}-<unset>}}"' for k in keys)
    env = dict(box["env"], QSE_PROFILE_OVERRIDE=profile, **(extra or {}))
    out = subprocess.run(["bash", "-c", script, "-", os.path.join(box["dir"], "run/ops/serve.env")], env=env,
                         capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines())


def test_a_setup_install_resolves_the_base():
    box = _box()
    if box is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return
    home = box["home"]
    text = open(os.path.join(box["dir"], "run/ops/serve.env")).read()
    model = text.index("\nQWEN38_MODEL=")
    assert model < text.index("\nQSE_PROFILE_NAME="), "QWEN38_MODEL is written after the weight profile"
    assert text.count("\nQWEN38_MODEL=") == 1
    snaps = os.path.join(home, BASE_SNAP)
    assert _resolve(box, "")["QWEN38_MODEL"] == snaps
    assert _resolve(box, "nvfp4")["QWEN38_MODEL"] == snaps
    for p in ("exl3-2.5", "exl3-4.0", "exl3-6.0"):
        assert _resolve(box, p)["QWEN38_MODEL"] == os.path.join(home, PLAIN), p
    # a standalone community pack reads its own folder (server/setup/community.py)
    from server.setup import community, registry
    m, c = next((m, c) for m in registry.load_all()[0].values()
                for c in registry.community_packs(m) if c["load"] == "standalone")
    with tempfile.TemporaryDirectory() as cdir:
        _write(os.path.join(cdir, c["profile"] + ".env"), community.env(m, c).encode())
        got = _resolve(box, c["profile"], extra={"QSE_COMMUNITY_ENV_DIR": cdir})["QWEN38_MODEL"]
    assert got == os.path.join(home, ".tandemllm/community", c["id"]), got
    # with the full FP8 snapshot in this install's HF cache, the EXL3 profiles read it
    snap = os.path.join(snaps, "c0ffee")
    _write(os.path.join(snap, "config.json"), b"{}")
    try:
        assert _resolve(box, "exl3-4.0")["QWEN38_MODEL"] == snaps
    finally:
        shutil.rmtree(os.path.join(home, ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"))


def test_the_setup_message():
    """OPS-149: install.sh --setup ends by saying where the Setup Guide is: on the box, through an ssh
    tunnel from another computer, or on the network with --host 0.0.0.0; no profile to switch to."""
    box = _box()
    if box is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return
    out = re.sub(r"\x1b\[[0-9;]*m", "", box["out"])
    tail = out[out.index("Setup mode"):]
    assert "no model is loaded yet. Open the Setup Guide in a browser and pick a model there." in tail, tail
    assert "On this machine:        http://127.0.0.1:18996/dashboard/#/setup" in tail, tail
    assert re.search(r"From another computer:  ssh -L 18996:127\.0\.0\.1:18996 \S+@\S+", tail), tail
    assert "then open http://127.0.0.1:18996/dashboard/#/setup on that computer" in tail, tail
    assert "--host 0.0.0.0" in tail and "use-profile.sh" not in tail, tail
    # OPS-155: the Chat tab waits for the setup, and says how to see chatd
    assert "The Chat tab starts with the engine once the setup is done (" in tail and \
        "/bin/tandem chat status)." in tail, tail


def test_a_setup_install_takes_an_exl3_profile():
    box = _box()
    if box is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return
    from server.setup import registry
    home, src = box["home"], os.path.join(box["dir"], "src")
    m = registry.load_all(os.path.join(src, "models"))[0]["qwen3.8-27b"]
    for aid in registry.profile(m, "exl3-4.0")["assets_required"]:
        a = registry.asset(m, aid)
        for p in [a["path"], *a.get("also", [])]:
            p = p.replace("${HOME}", home).replace("${REPO}", src)
            if p.endswith((".json", ".safetensors")):
                _write(p, b"{}")
            else:
                os.makedirs(p, exist_ok=True)
    for f in ("config.json", "model.safetensors"):
        _write(os.path.join(home, "exl3-quants/4.0", f), b"{}")
    assert not os.path.exists(os.path.join(home, BASE_SNAP))
    r = subprocess.run(["bash", os.path.join(box["dir"], "run/ops/use-profile.sh"), "exl3-4.0", "--fetch-only"],
                       env=box["env"], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "fetch-only: done" in out and "REFUSING" not in out, out



def test_a_setup_install_takes_the_bolt_profile():
    box = _box()
    if box is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return
    from server.setup import registry
    home, src = box["home"], os.path.join(box["dir"], "src")
    m = registry.load_all(os.path.join(src, "models"))[0]["nemotron-3.5-bolt-74b-a9b"]
    p = registry.profile(m, "nemotron-bolt-exl3-2.0")
    assert registry.recommended_profile(m)["id"] == "nemotron-bolt-exl3-2.0"
    assert p["hf_repo"] == "0xBakeer/TandemLLM-Nemotron-3.5-Bolt-74B-A9B-EXL3-2.0" and not p["hf_private"]
    pack = os.path.join(home, "models/nemotron-3.5-bolt-74b-a9b-exl3-2.0")
    got = _resolve(box, "nemotron-bolt-exl3-2.0", keys=("QSE_WEIGHTS_DIR", "QSE_WEIGHTS_HF", "QSE_ENGINE_FLAGS"))
    assert got["QSE_WEIGHTS_DIR"] == pack, got
    assert got["QSE_WEIGHTS_HF"] == p["hf_repo"], got
    assert f"--model {pack} " in got["QSE_ENGINE_FLAGS"] and "--draft-head lilicorr" in got["QSE_ENGINE_FLAGS"], got
    for aid in p["assets_required"]:
        a = registry.asset(m, aid)
        for f in [a["path"], *a.get("also", [])]:
            f = f.replace("${HOME}", home).replace("${REPO}", src)
            if f.endswith((".json", ".safetensors")):
                _write(f, b"{}")
            else:
                os.makedirs(f, exist_ok=True)
    for f in ("config.json", "model.safetensors"):
        _write(os.path.join(pack, f), b"{}")
    try:
        r = subprocess.run(["bash", os.path.join(box["dir"], "run/ops/use-profile.sh"), "nemotron-bolt-exl3-2.0",
                            "--fetch-only"], env=box["env"], capture_output=True, text=True, timeout=120,
                           stdin=subprocess.DEVNULL)
        out = r.stdout + r.stderr
        assert r.returncode == 0 and "fetch-only: done" in out and "REFUSING" not in out, out
    finally:
        shutil.rmtree(pack)


def test_a_setup_install_takes_the_super_vl_profile():
    box = _box()
    if box is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return
    from server.setup import registry
    home, src = box["home"], os.path.join(box["dir"], "src")
    m = registry.load_all(os.path.join(src, "models"))[0]["nemotron-3.5-super-vl-120b-a12b"]
    p = registry.profile(m, "nemotron-super-vl-exl3-4.0")
    assert registry.recommended_profile(m)["id"] == "nemotron-super-vl-exl3-4.0"
    for q in m["profiles"]:
        note = q.get("note") or ""
        assert "not public" not in note and not re.search(r"\bneeds\b(?! no token)[^.]*\btoken\b", note), q["id"]
        if q["hf_repo"].startswith("0xBakeer/"):
            assert not q["hf_private"], q["id"]
    assert p["hf_repo"] == "0xBakeer/TandemLLM-Nemotron-3.5-Super-VL-120B-A12B-EXL3-4.0"
    pack = os.path.join(home, "models/nemotron-3.5-super-vl-120b-a12b-exl3-4.0")
    got = _resolve(box, "nemotron-super-vl-exl3-4.0",
                   keys=("QSE_WEIGHTS_DIR", "QSE_WEIGHTS_HF", "QSE_WEIGHTS_REVISION", "QSE_ENGINE_FLAGS"))
    assert got["QSE_WEIGHTS_DIR"] == pack, got
    assert got["QSE_WEIGHTS_HF"] == p["hf_repo"] and got["QSE_WEIGHTS_REVISION"] == p["hf_revision"], got
    assert f"--model {pack} " in got["QSE_ENGINE_FLAGS"] and "--draft-head lilicorr" in got["QSE_ENGINE_FLAGS"], got
    for aid in p["assets_required"]:
        a = registry.asset(m, aid)
        for f in [a["path"], *a.get("also", [])]:
            f = f.replace("${HOME}", home).replace("${REPO}", src)
            if f.endswith((".json", ".safetensors")):
                _write(f, b"{}")
            else:
                os.makedirs(f, exist_ok=True)
    for f in ("config.json", "model.safetensors"):
        _write(os.path.join(pack, f), b"{}")
    try:
        r = subprocess.run(["bash", os.path.join(box["dir"], "run/ops/use-profile.sh"), "nemotron-super-vl-exl3-4.0",
                            "--fetch-only"], env=box["env"], capture_output=True, text=True, timeout=120,
                           stdin=subprocess.DEVNULL)
        out = r.stdout + r.stderr
        assert r.returncode == 0 and "fetch-only: done" in out and "REFUSING" not in out, out
    finally:
        shutil.rmtree(pack)

def test_a_setup_install_with_hf_home_keeps_one_cache():
    """install.sh run with HF_HOME: serve.env names that cache (HF_HUB_CACHE) before the profile, so
    a later shell without HF_HOME finds the FP8 snapshot there, and ops/use-profile.sh hands the cache
    to fetch_weights.py: base-plain needs no download, the snapshot stands for it."""
    box = _box("hf-home", hf_home="hf")
    if box is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return
    home, src = box["home"], os.path.join(box["dir"], "src")
    hub = os.path.join(home, "hf", "hub")
    text = open(os.path.join(box["dir"], "run/ops/serve.env")).read()
    assert f"\nHF_HUB_CACHE='{hub}'\n" in text and text.index("\nHF_HUB_CACHE=") < text.index("\nQSE_PROFILE_NAME=")
    snaps = os.path.join(hub, "models--Qwen--Qwen3.8-27B-FP8", "snapshots")
    assert _resolve(box, "exl3-4.0")["QWEN38_MODEL"] == os.path.join(home, PLAIN)
    _write(os.path.join(snaps, "c0ffee", "config.json"), b"{}")
    got = _resolve(box, "exl3-4.0", keys=("QWEN38_MODEL", "HF_HUB_CACHE"))
    assert got == {"QWEN38_MODEL": snaps, "HF_HUB_CACHE": hub}, got
    from server.setup import registry
    m = registry.load_all(os.path.join(src, "models"))[0]["qwen3.8-27b"]
    for aid in ("exl3-drafters-k3", "exl3-draft-head-k4", "nvfp4-drafters"):
        a = registry.asset(m, aid)
        for p in [a["path"], *a.get("also", [])]:
            p = p.replace("${HOME}", home).replace("${REPO}", src)
            if p.endswith(".safetensors"):
                _write(p, b"{}")
            else:
                os.makedirs(p, exist_ok=True)
    for f in ("config.json", "model.safetensors"):
        _write(os.path.join(home, "exl3-quants/4.0", f), b"{}")
    r = subprocess.run(["bash", os.path.join(box["dir"], "run/ops/use-profile.sh"), "exl3-4.0", "--fetch-only"],
                       env=box["env"], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "[fetch] base-plain: on this box" in out and "fetch-only: done" in out, out
    assert not os.path.exists(os.path.join(home, PLAIN))


if __name__ == "__main__":
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn(); print(f"  {name:60s} ok")
    finally:
        if _BOX:
            shutil.rmtree(_BOX["_"]["tmp"], ignore_errors=True)
