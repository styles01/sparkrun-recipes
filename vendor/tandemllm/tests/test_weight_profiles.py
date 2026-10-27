"""ops/profiles: serve.env's overlay sets only the weight-specific values, and every profile resolves."""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import struct
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("QSE_WEIGHTS", "QWEN38_EXL3", "QWEN38_DRAFT_EXL3", "QWEN38_DRAFT_HEAD_EXL3", "QWEN38_SAMPLE_NUCLEUS",
        "QSE_DISK_CACHE", "NV", "HEAD", "QSE_WEIGHTS_HF", "PORT", "SERVED_MODEL", "QSE_CONDUCTOR_CONFIG",
        "QSE_WEIGHTS_DIR", "QSE_WEIGHTS_REVISION", "QSE_ENGINE_FLAGS", "QSE_ASSETS", "QSE_MEM_NEED_GIB",
        "QWEN38_FN_TABLE", "QSE_PAGE_CACHE_DROP", "QSE_PAGE_CACHE_KEEP", "VISION", "MAX_LEN", "QWEN38_FN_LOOKUP",
        "RESIDENT_STASH_GB", "QSE_PARALLEL", "QSE_KV_POOL", "QSE_SIDE_MAX_LEN", "QWEN38_DECIDER_MODELS",
        "QWEN38_DECIDER_DIR")


def _env(profile: str | None, extra: dict | None = None) -> dict:
    with tempfile.TemporaryDirectory() as home:
        # serve.env's REPO (${HOME}/<dir>: the repository's, or a deploy tree's own name)
        m = re.search(r"^REPO=\$\{HOME\}/(\S+)$", open(os.path.join(ROOT, "ops/serve.env")).read(), re.M)
        os.symlink(ROOT, os.path.join(home, m.group(1) if m else "qwen38-spark-engine"))
        script = (f"set -a; . {ROOT}/ops/serve.env; "
                  + "; ".join(f'echo "{k}=${{{k}-<unset>}}"' for k in KEYS))
        env = {"HOME": home, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        env.update(extra or {})
        if profile is not None:
            env["QSE_PROFILE_OVERRIDE"] = profile
        out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True).stdout
        return {k.replace(home, "~"): v.replace(home, "~") for k, v in (l.split("=", 1) for l in out.splitlines())}


def test_no_profile_keeps_serve_env():
    e = _env(None)
    assert e["QSE_WEIGHTS"] == "<unset>" and e["QWEN38_EXL3"] == "<unset>", e


def test_exl3_profile_sets_only_the_weights():
    base, e = _env(None), _env("exl3-4.5")
    assert e["QSE_WEIGHTS"] == "exl3-4.5" and e["QWEN38_EXL3"] == "~/exl3-quants/4.5", e
    assert e["QWEN38_DRAFT_HEAD_EXL3"].endswith("exl3-draft-head-k4") and e["QWEN38_SAMPLE_NUCLEUS"] == "1"
    assert e["QSE_WEIGHTS_HF"] == "0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-4.5"
    assert e["QSE_DISK_CACHE"].endswith("disk-exl3-4.5")
    for k in ("PORT", "SERVED_MODEL", "QSE_CONDUCTOR_CONFIG", "NV", "HEAD"):
        assert e[k] == base[k], k


def test_nvfp4_profile_unsets_exl3():
    e = _env("nvfp4", {"QWEN38_EXL3": "/leak", "QWEN38_DRAFT_EXL3": "/leak"})
    assert e["QSE_WEIGHTS"] == "nvfp4" and e["QWEN38_EXL3"] == "<unset>" and e["QWEN38_SAMPLE_NUCLEUS"] == "0", e


def test_flashnext_profiles_name_their_pack_and_replace_the_27b_flags():
    """OPS-97: a Flash-Next profile is a branch of turboderp's repo, its engine flags replace the 27B's
    (start.sh), it needs no 27B asset, keeps a warm table out of the page-cache drop, and leaves the
    shared values alone."""
    base = _env(None)
    for bpw, table in (("2.05", "warm"), ("3.05", "cold"), ("4.05", "cold"), ("5.05", "cold"), ("6.05", "cold")):
        e = _env(f"flashnext-{bpw}", {"QWEN38_EXL3": "/leak", "QWEN38_DRAFT_EXL3": "/leak"})
        d = f"~/flashnext-packs/{bpw}"
        assert e["QSE_WEIGHTS"] == f"flashnext-{bpw}" and e["QSE_WEIGHTS_DIR"] == d, e
        assert e["QSE_WEIGHTS_HF"] == "turboderp/Qwen3.8-Flash-Next-exl3", e
        assert e["QSE_WEIGHTS_REVISION"].startswith(f"{bpw}bpw_"), e
        assert e["QSE_ENGINE_FLAGS"] == f"--model {d} --drafter mtp", e
        assert e["QWEN38_EXL3"] == e["QWEN38_DRAFT_EXL3"] == e["QWEN38_DRAFT_HEAD_EXL3"] == "<unset>", e
        assert e["QSE_ASSETS"] == "" and int(e["QSE_MEM_NEED_GIB"]) > 0, e
        assert e["QWEN38_FN_TABLE"] == table and e["QSE_PAGE_CACHE_DROP"] == d, e
        assert e["QSE_PAGE_CACHE_KEEP"] == ("ngram_embedding.safetensors" if table == "warm" else ""), e
        assert e["QSE_DISK_CACHE"].endswith(f"disk-flashnext-{bpw}") and e["VISION"] == "on", e
        assert e["QWEN38_FN_LOOKUP"] == "1", e
        # ENG-241: three slots on the KV pool; the side window serve.env's
        assert e["QSE_PARALLEL"] == "3" and e["QSE_KV_POOL"] == "on", e
        # 0.8.2: Laya beside it, as on the 27B box; its checkpoints where serve.env says
        assert e["QWEN38_DECIDER_MODELS"] == "laya-english,laya-multilingual,laya-typed-decisions", e
        assert e["QWEN38_DECIDER_DIR"] == base["QWEN38_DECIDER_DIR"], e
        # the guest stash as serve.env sets it (ENG-157 fixed: it was off on Flash-Next)
        for k in ("PORT", "SERVED_MODEL", "QSE_CONDUCTOR_CONFIG", "MAX_LEN", "RESIDENT_STASH_GB", "QSE_SIDE_MAX_LEN"):
            assert e[k] == base[k], k
    # a 27B profile sets none of them: start.sh's 27B flags and use-profile.sh's 27B assets stay
    e = _env("exl3-4.5")
    assert all(e[k] == "<unset>" for k in ("QSE_ENGINE_FLAGS", "QSE_ASSETS", "QSE_MEM_NEED_GIB",
                                           "QSE_WEIGHTS_REVISION", "QSE_PAGE_CACHE_KEEP")), e
    assert e["QWEN38_DECIDER_MODELS"] == base["QWEN38_DECIDER_MODELS"] == "", e


def _use_profile(*args: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as home:
        text = open(os.path.join(ROOT, "ops/serve.env")).read()
        m = re.search(r"^REPO=\$\{HOME\}/(\S+)$", text, re.M)
        os.symlink(ROOT, os.path.join(home, m.group(1) if m else "qwen38-spark-engine"))
        py = re.search(r"^PY=\$\{HOME\}/(\S+)$", text, re.M)     # serve.env's python: this one (a
        if py:                                                     # link would lose its venv)
            os.makedirs(os.path.dirname(os.path.join(home, py.group(1))))
            with open(os.path.join(home, py.group(1)), "w") as fh:
                fh.write(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
            os.chmod(os.path.join(home, py.group(1)), 0o755)
        env = {"HOME": home, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        return subprocess.run(["bash", os.path.join(ROOT, "ops/use-profile.sh"), *args], env=env,
                              capture_output=True, text=True, timeout=180)


def _hub_reachable() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8", timeout=10)
        return True
    except OSError:
        return False


def test_use_profile_fetches_a_flashnext_branch_and_needs_no_27b_asset():
    """OPS-97: a Flash-Next switch on a box with none of the 27B's drafters, draft head or base snapshot
    is not refused; it would fetch the pack's branch, and the memory it waits for is the profile's.
    OPS-95: a 27B profile on the same box would fetch its assets first (the K3 drafters, the K4 head,
    the BF16 drafters into train/, the base into the HF cache), then its weights, every file's URL
    and size resolved against the hub; with no token."""
    if sys.platform == "darwin":
        print("    (skipped: GNU df -- run it on the box)")
        return
    if not _hub_reachable():
        print("    (skipped: huggingface.co does not answer)")
        return
    r = _use_profile("flashnext-3.05", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "turboderp/Qwen3.8-Flash-Next-exl3@3.05bpw_h5_ng5 (~85.1 GB)" in r.stdout, r.stdout
    assert "dry run: would fetch" in r.stdout and "MemAvailable >= 70 GB" in r.stdout, r.stdout
    assert "/resolve/" in r.stdout and "ngram_embedding.safetensors" in r.stdout, r.stdout
    r = _use_profile("exl3-4.5", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    for want in ("exl3-drafters-k3/ft-b8-v2.safetensors", "exl3-draft-head-k4/model.safetensors",
                 "TandemLLM-Qwen3.8-27B-DFlash2-b8/resolve/", "TandemLLM-Qwen3.8-27B-DFlash2-b16/resolve/",
                 "Qwen/Qwen3.8-27B-FP8/resolve/", "outside.safetensors", "dry run: nothing downloaded",
                 "TandemLLM-Qwen3.8-27B-EXL3-4.5/resolve/", "model-00001-of-00003.safetensors",
                 "then 18.4 GB of weights", "dry run: would fetch"):
        assert want in r.stdout, (want, r.stdout)
    assert "mtp.safetensors" not in r.stdout and "REFUSING" not in r.stdout, r.stdout
    r = _use_profile("--list")
    assert r.returncode == 0 and "flashnext-6.05" in r.stdout, r.stdout


def test_a_branch_without_a_checksum_list_is_verified_against_the_hubs():
    """fetch_weights.py: a Flash-Next branch has no crc32.txt; the hub's LFS sha256 (its
    _hub_manifest.json) is the list -- a changed or missing weight file fails, a non-LFS file is not
    checked."""
    import hashlib
    import json
    sys.path.insert(0, os.path.join(ROOT, "ops"))
    import fetch_weights
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "model-00001-of-00001.safetensors"), "wb").write(b"w" * 4096)
        open(os.path.join(d, "ngram_embedding.safetensors"), "wb").write(b"t" * 1024)
        open(os.path.join(d, "config.json"), "w").write("{}")
        files = {n: {"size": os.path.getsize(os.path.join(d, n)),
                     "sha256": hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest()}
                 for n in ("model-00001-of-00001.safetensors", "ngram_embedding.safetensors")}
        files["config.json"] = {"size": 2, "sha256": None}
        json.dump({"repo": "r", "revision": "b", "commit": "c", "files": files},
                  open(os.path.join(d, fetch_weights.MANIFEST), "w"))
        assert fetch_weights.verify(d) == []
        open(os.path.join(d, "ngram_embedding.safetensors"), "ab").write(b"x")
        assert [b.split(":")[0] for b in fetch_weights.verify(d)] == ["ngram_embedding.safetensors"]
        os.remove(os.path.join(d, "ngram_embedding.safetensors"))
        assert fetch_weights.verify(d) == ["missing ngram_embedding.safetensors"]


def test_a_pack_links_the_table_a_sibling_pack_already_holds():
    """OPS-97: 4.05, 5.05 and 6.05 share one n-gram table (same sha256): a fetch of one next to another
    hard-links it instead of downloading 39 GB; a file with another sha256 is downloaded."""
    import json
    sys.path.insert(0, os.path.join(ROOT, "ops"))
    import fetch_weights
    with tempfile.TemporaryDirectory() as parent:
        have, part = os.path.join(parent, "4.05"), os.path.join(parent, "5.05.partial")
        os.makedirs(have)
        os.makedirs(part)
        open(os.path.join(have, "ngram_embedding.safetensors"), "wb").write(b"t" * 64)
        open(os.path.join(have, "model-00001-of-00001.safetensors"), "wb").write(b"a" * 64)
        files = {"ngram_embedding.safetensors": {"size": 64, "sha256": "t" * 64},
                 "model-00001-of-00001.safetensors": {"size": 64, "sha256": "a" * 64}}
        json.dump({"files": files}, open(os.path.join(have, fetch_weights.MANIFEST), "w"))
        mine = dict(files, **{"model-00001-of-00001.safetensors": {"size": 64, "sha256": "b" * 64}})
        json.dump({"files": mine}, open(os.path.join(part, fetch_weights.MANIFEST), "w"))
        assert fetch_weights.share(part, os.path.join(parent, "5.05")) == ["ngram_embedding.safetensors"]
        a, b = (os.stat(os.path.join(x, "ngram_embedding.safetensors")) for x in (have, part))
        assert a.st_ino == b.st_ino and not os.path.exists(os.path.join(part, "model-00001-of-00001.safetensors"))


def _fake_box(home: str, fails: str) -> tuple[str, str, dict]:
    """A checkout under `home` with the real use-profile.sh, profiles and serve.env tail, fake stop.sh and
    start.sh that note what they do (start.sh: setup mode while run/setup-pending is there, as the real
    one; the profile `fails` does not come up), a fake curl whose /health names the active profile, and a
    complete Flash-Next 3.05 pack. (repo, marks file, env)."""
    repo = os.path.join(home, "repo")
    os.makedirs(os.path.join(repo, "ops"))
    shutil.copy(os.path.join(ROOT, "ops/use-profile.sh"), os.path.join(repo, "ops"))
    shutil.copytree(os.path.join(ROOT, "ops/profiles"), os.path.join(repo, "ops/profiles"))
    tail = open(os.path.join(ROOT, "ops/serve.env")).read().split("# Weight profile (ops/use-profile.sh)")[1]
    with open(os.path.join(repo, "ops/serve.env"), "w") as fh:
        fh.write(f"REPO={repo}\nPY={sys.executable}\nPORT=18996\n# Weight profile{tail}")
    marks = os.path.join(home, "marks")
    fakes = {"stop.sh": f'echo "stop" >> {marks}\n',
             "start.sh": f'if [ -f {repo}/run/setup-pending ]; then echo "start setup" >> {marks}; exit 0; fi\n'
                         f'echo "start $(cat {repo}/ops/active-profile)" >> {marks}\n'
                         f'[ "$(cat {repo}/ops/active-profile)" != {fails} ]\n'}
    for name, body in fakes.items():
        with open(os.path.join(repo, "ops", name), "w") as fh:
            fh.write("#!/bin/bash\n" + body)
        os.chmod(os.path.join(repo, "ops", name), 0o755)
    binary = os.path.join(home, "bin")
    os.makedirs(binary)
    with open(os.path.join(binary, "curl"), "w") as fh:         # /health names the active profile
        fh.write(f'#!/bin/bash\ncase "$*" in *health*) echo "{{\\"weights\\": {{\\"name\\": '
                 f'\\"$(cat {repo}/ops/active-profile)\\"}}}}";; '
                 '*) echo \'{"choices": [{"message": {"content": "ready"}}]}\';; esac\n')
    os.chmod(os.path.join(binary, "curl"), 0o755)
    pack = os.path.join(home, "flashnext-packs/3.05")            # a complete pack on disk
    os.makedirs(pack)
    for n, body in (("config.json", "{}"), ("model.safetensors.index.json", '{"weight_map": {}}'),
                    ("ngram_embedding.safetensors", "t")):
        open(os.path.join(pack, n), "w").write(body)
    env = {"HOME": home, "PATH": binary + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
           "BOX_LOCK": os.path.join(home, "box.flock"), "MEM_NEED_GB": "1"}
    return repo, marks, env


def test_a_failed_switch_to_flashnext_rolls_back_to_the_27b():
    """OPS-99: a switch from a 27B profile to Flash-Next whose start fails stops, starts Flash-Next,
    stops again and starts the 27B profile -- one engine at a time, in that order -- and says so. The
    scripts it calls are fakes; only use-profile.sh and the profiles are real."""
    if not os.path.isfile("/proc/meminfo") or not shutil.which("flock"):
        print("    (skipped: needs /proc/meminfo and flock -- run it on the box)")
        return
    with tempfile.TemporaryDirectory() as home:
        repo, marks, env = _fake_box(home, fails="flashnext-3.05")
        open(os.path.join(repo, "ops/active-profile"), "w").write("exl3-4.5\n")
        r = subprocess.run(["bash", os.path.join(repo, "ops/use-profile.sh"), "flashnext-3.05"], env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 1 and "rolled back: exl3-4.5 live" in r.stdout, r.stdout + r.stderr
        assert open(marks).read().split("\n")[:-1] == ["stop", "start flashnext-3.05", "stop", "start exl3-4.5"]
        assert open(os.path.join(repo, "ops/active-profile")).read().strip() == "exl3-4.5"


def test_a_pack_in_one_file_is_complete():
    """A pack whose weights are one model.safetensors and no index (LFM2.5's 2.5 pack) is on the box: the
    switch takes it as it is, with no download, and the Setup Guide counts it as there
    (server/setup/device.py `weights_present`); an empty file is not a pack."""
    from server.setup import device
    if not os.path.isfile("/proc/meminfo") or not shutil.which("flock"):
        print("    (skipped: needs /proc/meminfo and flock -- run it on the box)")
        return
    with tempfile.TemporaryDirectory() as home:
        repo, marks, env = _fake_box(home, fails="none")
        open(os.path.join(repo, "ops/active-profile"), "w").write("lfm25-exl3-3.5\n")
        pack = os.path.join(home, "models/lfm2.5-24b-a2b-exl3-2.5")
        os.makedirs(pack)
        open(os.path.join(pack, "config.json"), "w").write("{}")
        open(os.path.join(pack, "model.safetensors"), "w").write("t")
        assert device.weights_present({"dir": pack, "files": [], "repo": ""})
        r = subprocess.run(["bash", os.path.join(repo, "ops/use-profile.sh"), "lfm25-exl3-2.5"], env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and "weights" in r.stdout and "present" in r.stdout, r.stdout + r.stderr
        assert "fetch" not in r.stdout, r.stdout
        assert open(marks).read().split("\n")[:-1] == ["stop", "start lfm25-exl3-2.5"]
        open(os.path.join(pack, "model.safetensors"), "w").close()
        assert not device.weights_present({"dir": pack, "files": [], "repo": ""})


def test_a_switch_from_setup_mode_configures_the_box_or_goes_back_to_setup_mode():
    """OPS-95: on a box install.sh --setup left in setup mode (run/setup-pending, no active profile),
    ops/use-profile.sh flashnext-3.05 removes the marker before the start, so start.sh starts the
    profile, not setup mode again; a failed start puts the marker back as it was, removes the active
    profile and starts setup mode."""
    if not os.path.isfile("/proc/meminfo") or not shutil.which("flock"):
        print("    (skipped: needs /proc/meminfo and flock -- run it on the box)")
        return
    for fails in ("none", "flashnext-3.05"):
        with tempfile.TemporaryDirectory() as home:
            repo, marks, env = _fake_box(home, fails=fails)
            os.makedirs(os.path.join(repo, "run"))
            open(os.path.join(repo, "run/setup-pending"), "w").write("HOST=0.0.0.0\n")
            r = subprocess.run(["bash", os.path.join(repo, "ops/use-profile.sh"), "flashnext-3.05"], env=env,
                               capture_output=True, text=True, timeout=120)
            out = r.stdout + r.stderr
            pending, active = os.path.join(repo, "run/setup-pending"), os.path.join(repo, "ops/active-profile")
            if fails == "none":
                assert r.returncode == 0 and "OK: flashnext-3.05 live" in out, out
                assert open(marks).read().split("\n")[:-1] == ["stop", "start flashnext-3.05"], out
                assert not os.path.exists(pending) and open(active).read().strip() == "flashnext-3.05"
            else:
                assert r.returncode == 1 and "back in setup mode" in out, out
                assert open(marks).read().split("\n")[:-1] == ["stop", "start flashnext-3.05", "stop",
                                                              "start setup"], out
                assert open(pending).read() == "HOST=0.0.0.0\n" and not os.path.exists(active), out


def test_exl3_profiles_read_the_bf16_drafters_under_train():
    """OPS-95: an EXL3 profile's drafter arms load their K3 file by the checkpoint folder's name
    (engine/drafters/dflash2.py _exl3_file), so CKPT8/CKPT16 are the repository's train/ft-b8-v2 and
    train/ft-b16 whatever serve.env says (an install.sh box points them into the HF cache, or leaves
    them empty with --setup), the folders the nvfp4-drafters asset is fetched into."""
    for n in ("exl3-2.5", "exl3-4.5", "exl3-6.0"):
        e = _env(n, {"CKPT8": "/hf/snapshots/abc", "CKPT16": ""})
        out = subprocess.run(["bash", "-c", f"set -a; . {ROOT}/ops/serve.env; echo $CKPT8 $CKPT16"],
                             env={"HOME": "/h", "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                                  "QSE_PROFILE_OVERRIDE": n}, capture_output=True, text=True).stdout.split()
        assert out == ["/h/qwen38-spark-engine/train/ft-b8-v2", "/h/qwen38-spark-engine/train/ft-b16"], out
        assert os.path.basename(e["QWEN38_DRAFT_EXL3"]) == "exl3-drafters-k3", e


class _Hub:
    """fetch_weights' view of the hub, on the disk: hub_files from a table, snapshot_download writes
    the table's bytes (into local_dir, or a snapshot folder of the cache), and every call is noted."""

    def __init__(self, fw, cache: str, bad: tuple = ()):
        import hashlib
        import huggingface_hub
        self.fw, self.cache, self.calls, self.bad, self.revisions = fw, cache, [], set(bad), []
        self.body = {}
        for repo, names in {
                "0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-drafters": [
                    "exl3-drafters-k3/ft-b8-v2.safetensors", "exl3-drafters-k3/ft-b16.safetensors",
                    "exl3-draft-head-k4/model.safetensors", "exl3-draft-head-k4/source-config.json", "README.md"],
                "0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b8": ["config.json", "model.safetensors", "README.md"],
                "0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16": ["config.json", "model.safetensors", "README.md"],
                "Qwen/Qwen3.8-27B-FP8": ["config.json", "tokenizer.json", "layers-0.safetensors",
                                         "outside.safetensors", "mtp.safetensors", "README.md"],
                "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF": [
                    "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf", "README.md"]}.items():
            for n in names:
                self.body[(repo, n)] = f"{repo}/{n}".encode() * 3
        # the base's safetensors files are real ones (OPS-133: the EXL3 profiles' plain base reads
        # their headers and the kept tensors by range requests)
        for n, ts in {"layers-0.safetensors": [("model.language_model.layers.0.input_layernorm.weight", "BF16", 8),
                                               ("model.language_model.layers.0.mlp.up_proj.weight_scale_inv", "BF16", 4),
                                               ("model.language_model.layers.0.mlp.up_proj.weight", "F8_E4M3", 64)],
                      "outside.safetensors": [("lm_head.weight", "BF16", 32),
                                              ("model.language_model.embed_tokens.weight", "BF16", 32)],
                      "mtp.safetensors": [("mtp.fc.weight", "BF16", 8)]}.items():
            hdr, off = {}, 0
            for name, dtype, nb in ts:
                hdr[name] = {"dtype": dtype, "shape": [nb if dtype == "F8_E4M3" else nb // 2],
                             "data_offsets": [off, off + nb]}
                off += nb
            h = json.dumps(hdr).encode()
            h += b" " * (-len(h) % 8)
            self.body[("Qwen/Qwen3.8-27B-FP8", n)] = struct.pack("<Q", len(h)) + h + os.urandom(off)
        self.sha = {k: hashlib.sha256(v).hexdigest() for k, v in self.body.items()}
        self.saved = (fw.hub_files, huggingface_hub.snapshot_download, fw.open_range)
        fw.hub_files = self.hub_files
        huggingface_hub.snapshot_download = self.snapshot_download
        fw.open_range = self.open_range

    def close(self):
        import huggingface_hub
        self.fw.hub_files, huggingface_hub.snapshot_download, self.fw.open_range = self.saved

    def open_range(self, url, start, end):
        import io
        repo, rest = url.split("://", 1)[1].split("/", 1)[1].split("/resolve/", 1)
        name = rest.split("/", 1)[1]
        self.calls.append((repo, "range", (name,)))
        r = io.BytesIO(self.body[(repo, name)][start:end])
        r.status = 206
        return r

    def hub_files(self, repo, revision=None):
        files = {n: {"size": len(b), "sha256": self.sha[(r, n)]
                     if n.endswith((".safetensors", ".json", ".gguf")) and not n.endswith("config.json") else None}
                 for (r, n), b in self.body.items() if r == repo}
        # the GGUF repository's branch has moved past its pin: an unpinned read gets another commit
        head = "moved" if repo.startswith("nvidia/") and revision is None else "c0ffee"
        return {"repo": repo, "revision": revision, "commit": revision or head, "files": files}

    def snapshot_download(self, repo_id, revision=None, local_dir=None, allow_patterns=None,
                          ignore_patterns=None, token=None, **_kw):
        import fnmatch
        self.calls.append((repo_id, local_dir, tuple(allow_patterns or ())))
        self.revisions.append((repo_id, revision))
        out = local_dir or os.path.join(self.cache, "models--" + repo_id.replace("/", "--"), "snapshots", revision)
        for (r, n), b in self.body.items():
            if r == repo_id and (not allow_patterns or any(fnmatch.fnmatch(n, p) for p in allow_patterns)):
                os.makedirs(os.path.dirname(os.path.join(out, n)), exist_ok=True)
                with open(os.path.join(out, n), "wb") as fh:
                    fh.write(b + (b"!" if n in self.bad else b""))
        return out


def _fetch_box():
    """A HOME, a checkout (the repository's files, train/ empty) and fetch_weights on a fake hub."""
    sys.path.insert(0, os.path.join(ROOT, "ops"))
    import fetch_weights
    tmp = tempfile.mkdtemp(prefix="fetch-box-")
    home, repo = os.path.join(tmp, "home"), os.path.join(tmp, "repo")
    os.makedirs(home)
    os.makedirs(repo)
    for d in os.listdir(ROOT):                 # a checkout of its own: train/ and run/ empty
        if d not in ("train", "run", ".git"):
            os.symlink(os.path.join(ROOT, d), os.path.join(repo, d))
    old = {k: os.environ.get(k) for k in ("HOME", "HF_HOME", "HF_HUB_CACHE", "HF_TOKEN")}
    os.environ.update(HOME=home)
    for k in ("HF_HOME", "HF_HUB_CACHE", "HF_TOKEN"):
        os.environ.pop(k, None)
    return fetch_weights, tmp, home, repo, old


def _fetch_box_close(tmp, old, hub=None):
    if hub:
        hub.close()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    shutil.rmtree(tmp, ignore_errors=True)


def test_assets_are_fetched_where_the_profile_reads_them():
    """OPS-95: fetch_weights.py --assets on a new box (no token): the K3 drafters and the K4 head into
    ~/.qwen38-spark-engine/weights (the repo's folders kept), the BF16 drafters into the checkout's
    train/ft-b8-v2 and train/ft-b16, the plain base (OPS-133) into ~/.qwen38-spark-engine/weights/base-plain
    without mtp.safetensors; every LFS file checked against the hub's sha256. Run again, nothing is downloaded; a Flash-Next profile needs no
    asset; an asset that is gone (the K4 head) is the only one fetched again."""
    fw, tmp, home, repo, old = _fetch_box()
    hub = _Hub(fw, os.path.join(home, ".cache", "huggingface", "hub"))
    try:
        assert fw.assets("exl3-4.5", repo, margin_gb=0) == 0
        w = os.path.join(home, ".qwen38-spark-engine", "weights")
        for f in ("exl3-drafters-k3/ft-b8-v2.safetensors", "exl3-drafters-k3/ft-b16.safetensors",
                  "exl3-draft-head-k4/model.safetensors"):
            assert open(os.path.join(w, f), "rb").read() == hub.body[("0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-drafters", f)]
        for arm, r in (("ft-b8-v2", "b8"), ("ft-b16", "b16")):
            d = os.path.join(repo, "train", arm)
            assert open(os.path.join(d, "model.safetensors"), "rb").read() == \
                hub.body[(f"0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-{r}", "model.safetensors")]
        # OPS-133: the EXL3 profile's base is the plain one, without mtp.safetensors and lm_head
        plain = os.path.join(w, "base-plain")
        assert os.path.isfile(os.path.join(plain, "outside.safetensors")) and not os.path.exists(
            os.path.join(plain, "mtp.safetensors"))
        assert not os.path.exists(os.path.join(home, ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"))
        assert not glob.glob(os.path.join(home, ".qwen38-spark-engine", "weights.partial*")), "a .partial left"
        assert not glob.glob(os.path.join(w, "*.partial*")), "a .partial left"
        assert len([c for c in hub.calls if c[1] != "range"]) == 5, hub.calls
        hub.calls.clear()
        assert fw.assets("exl3-4.5", repo, margin_gb=0) == 0 and hub.calls == []
        assert fw.assets("flashnext-3.05", repo, margin_gb=0) == 0 and hub.calls == []
        shutil.rmtree(os.path.join(w, "exl3-draft-head-k4"))
        assert fw.assets("exl3-2.5", repo, margin_gb=0) == 0
        assert [c[0] for c in hub.calls] == ["0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-drafters"], hub.calls
        assert os.path.isfile(os.path.join(w, "exl3-drafters-k3/ft-b16.safetensors"))   # the merge kept it
    finally:
        _fetch_box_close(tmp, old, hub)


def test_an_interrupted_or_bad_download_is_not_taken_for_the_asset():
    """OPS-95: a base download that stopped half-way leaves its state `complete: false`, so the next
    run fetches it again even though config.json is there; a file whose sha256 is not the hub's fails
    the fetch (exit 5) and stays out of the folder the profile reads."""
    fw, tmp, home, repo, old = _fetch_box()
    hub = _Hub(fw, os.path.join(home, ".cache", "huggingface", "hub"), bad=("model.safetensors",))
    try:
        snap = os.path.join(home, ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/c0ffee")
        os.makedirs(snap)
        open(os.path.join(snap, "config.json"), "w").write("{}")
        fw._write_state("base-fp8", {"complete": False})
        assert fw.assets("nvfp4", repo, margin_gb=0) == 5
        assert not os.path.exists(os.path.join(repo, "train", "ft-b8-v2")), "a bad file reached train/"
        assert os.path.isfile(os.path.join(repo, "train", "ft-b8-v2.partial-nvfp4-drafters", "model.safetensors"))
        hub.bad.clear()
        hub.calls.clear()
        assert fw.assets("nvfp4", repo, margin_gb=0) == 0
        assert [c[0] for c in hub.calls] == ["0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b8",
                                             "0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16",
                                             "Qwen/Qwen3.8-27B-FP8"], hub.calls
        assert json.load(open(fw.state_path("base-fp8")))["complete"] is True
    finally:
        _fetch_box_close(tmp, old, hub)


def test_the_disk_is_checked_before_any_download_and_a_dry_run_downloads_nothing():
    """OPS-95: the assets still to download, the weights after them (--extra-gb) and the margin must
    fit, or nothing is downloaded (exit 4); a dry run prints every file's URL and size and downloads
    nothing; an asset with no download is refused (exit 3)."""
    import contextlib
    import io
    fw, tmp, home, repo, old = _fetch_box()
    hub = _Hub(fw, os.path.join(home, ".cache", "huggingface", "hub"))
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert fw.assets("exl3-4.5", repo, margin_gb=10 ** 9) == 4
        assert hub.calls == [] and "REFUSING the download" in out.getvalue(), out.getvalue()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert fw.assets("exl3-4.5", repo, dry_run=True, extra_gb=18.4, margin_gb=0) == 0
        text = out.getvalue()
        assert hub.calls == [] and "dry run: nothing downloaded" in text, text
        assert "https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/c0ffee/outside.safetensors" in text, text
        assert "mtp.safetensors" not in text and "then 18.4 GB of weights" in text, text
        # a model.yaml whose asset has no download: refused, nothing fetched
        os.remove(os.path.join(repo, "models"))
        shutil.copytree(os.path.join(ROOT, "models"), os.path.join(repo, "models"))
        y = os.path.join(repo, "models", "qwen3.8-27b", "model.yaml")
        text = open(y).read()
        i = text.index("  - id: base-plain")
        j = text.index("    source:", i)
        k = text.index("    fetch:", i)
        open(y, "w").write(text[:i] + text[i:k].replace("fetchable: true", "fetchable: false") + text[j:])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert fw.assets("exl3-4.5", repo, margin_gb=0) == 3
        assert "base-plain" in out.getvalue() and "has no download" in out.getvalue(), out.getvalue()
        assert [c[0] for c in hub.calls] == [], hub.calls
    finally:
        _fetch_box_close(tmp, old, hub)


def test_an_asset_with_a_pinned_revision_is_downloaded_at_that_revision():
    """SPD-121: the Bolt GGUF asset names a revision in model.yaml; its download reads that commit, not
    the repository's branch, which may have moved on."""
    import contextlib
    import io
    fw, tmp, home, repo, old = _fetch_box()
    hub = _Hub(fw, os.path.join(home, ".cache", "huggingface", "hub"))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert fw.assets("nemotron-bolt-iq2", repo, margin_gb=0) == 0
        pin = "724e3fee41becc4048998e2204748147876ed56b"
        assert hub.revisions == [("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF", pin)], hub.revisions
        assert os.path.isfile(os.path.join(home, "models", "nemotron-3.5-bolt-74b-a9b-iq2-gguf",
                                           "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"))
    finally:
        _fetch_box_close(tmp, old, hub)


def test_every_profile_resolves():
    names = [f[:-4] for f in os.listdir(os.path.join(ROOT, "ops/profiles")) if f.endswith(".env") and not f.startswith("_")]
    assert {"nvfp4", "exl3-2.5", "exl3-3.0", "exl3-3.5", "exl3-4.0", "exl3-4.5", "exl3-6.0",
            "flashnext-2.05", "flashnext-3.05", "flashnext-4.05", "flashnext-5.05", "flashnext-6.05"} <= set(names), names
    for n in names:
        assert _env(n)["QSE_WEIGHTS"] == n, n


def test_super_vl_profiles_read_images_and_the_other_nemotrons_do_not():
    """ENG-365: the Super VL profiles load the checkpoint's RADIO tower (VISION=on, after _nemotron.env's
    off); Bolt and Lightning stay text only."""
    names = [f[:-4] for f in os.listdir(os.path.join(ROOT, "ops/profiles")) if f.startswith("nemotron-")]
    vl = [n for n in names if n.startswith("nemotron-super-vl-")]
    assert len(vl) == 3, vl
    for n in names:
        assert _env(n)["VISION"] == ("on" if n in vl else "off"), n


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name:60s} ok")
