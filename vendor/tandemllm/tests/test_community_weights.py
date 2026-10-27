"""Community weight packs (docs/community-weights.md): models/<model>/community/<id>.json, their checks
(server/setup/registry.py), the index and its sync from GitHub, the profile (server/setup/community.py),
the setup API and ops/use-profile.sh community/<id>.

The test matrix:

  1  the ThinkingCap entry loads with every computed field: the profile name, the avatar, the social
     links, the HF link at its commit, the memory of exl3-4.0; the weights step and Wren list it
                                                            test_the_first_entry
  2  models/schema/community.schema.json is the schema the code checks, models/community-index.json
     the folders' index                                     test_the_schema_and_index_files
  3  each rule refuses a bad entry and names the field       test_each_rule_refuses_a_bad_entry
  4  --check names the file and the field of a bad entry, and a stale index
                                                            test_the_cli_names_the_file_and_field
  5  the selection, the estimate, the overlay and the disk see the pack: vision off where images do
     not work, the apply job takes community/<id>           test_the_setup_path_takes_a_pack
  6  the profile, resolved through ops/serve.env: standalone (the pack is QWEN38_MODEL), overlay
     (QWEN38_EXL3 over the base), each drafter choice      test_the_profile_through_serve_env
  7  the served card names the pack, so the identity note does   test_the_card_names_the_pack
  8  the sync against a local server: 200 into the cache, 304 with the ETag, a timeout falls back to
     the cache, an invalid entry is dropped with its reason, a bad index keeps the cache, off fetches
     nothing, ten page loads at once make one request, no cache falls back to the bundle
                                                            test_the_sync
  9  the API: tiles count the packs, the model carries them and its sync state, GET/POST community
                                                            test_the_api
 10  the fetch plan of ThinkingCap from the real hub, with no token: every URL at the pinned commit,
     16.38 GB                                               test_the_fetch_plan_of_the_first_entry
 11  ops/use-profile.sh community/<id> --dry-run writes its profile to a folder of its own, plans
     the pack and our drafters, and leaves run/ alone (on the box: GNU df)
                                                            test_use_profile_dry_run

VIS-175: a pack one of the model's own profiles serves (turboderp's Flash-Next branches)

 12  each of turboderp's five entries checks, names its own profile, its repository at a commit and
     turboderp; it is not a community/<id> profile, switch, served card or download folder; a
     mismatch with its profile (repository, bits a weight, an unknown profile) and a draft layer in
     a pack without a profile are refused; the fetched index cannot replace it, and the index a
     box before 0.11.1 fetches holds none of them          test_a_pack_an_own_profile_serves

ENG-377: a pack on the Nemotron 3.5 Super VL base (PR #63)

 13  the entry checks and loads on NemotronH: memory and drafters (LiLiCorr) of the nearest own EXL3
     profile, text only; another NemotronH base, an overlay, a 27B drafter choice, images without a
     tower are refused with their field; the profile through serve.env is the 4.0 profile with the
     pack in its place; the served card is Super VL's, vision off
                                                            test_a_nemotron_pack
 14  the file check: the pack's real config and index (the tower absent) pass; a wrong model type,
     layer count or quantization, a missing shard, images listed without the tower each fail naming
     the file or key                                        test_the_nemotron_file_check
 15  the Super VL pack's card is its runs: verified build, date, tok/s and tokens a round equal
     measurements/0.12.1-summary.json, every drafted text the plain run's
                                                            test_the_nemotron_card_is_the_raw_data

Run: python tests/test_community_weights.py
"""

from __future__ import annotations

import copy
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server.setup import community, registry  # noqa: E402

ROOT = registry.ROOT
SLUG = "thinkingcap-27b-kearuga-exl3-4.00"
PID = "community/" + SLUG
ENTRY = os.path.join(ROOT, "models", "qwen3.8-27b", "community", SLUG + ".json")
REV = "11447639033dd80a4c04536b5b37ed7d642802ee"
FN_BPW = ("2.05", "3.05", "4.05", "5.05", "6.05")
FN_ENTRY = os.path.join(ROOT, "models", "qwen3.8-flash-next", "community", "turboderp-flashnext-3.05.json")
NH = "nemotron-3.5-super-vl-120b-a12b"
NH_SLUG = "nemotron-super-vl-exl3-3.5bpw"
NH_PID = "community/" + NH_SLUG
NH_ENTRY = os.path.join(ROOT, "models", NH, "community", NH_SLUG + ".json")


def _entry() -> dict:
    with open(ENTRY) as fh:
        return json.load(fh)


def _models() -> dict:
    return registry.load_all()[0]


class _Env:
    """os.environ with these keys set (None: removed) for the block, and the module states reset."""

    def __init__(self, **kv):
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        community._STATE.update(fetched=False, checked_at=None, error=None, note=None)
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        community._STATE.update(fetched=False, checked_at=None, error=None, note=None)


def test_the_first_entry():
    with _Env(QSE_COMMUNITY_DIR=tempfile.mkdtemp(prefix="cw-"), QSE_COMMUNITY_URL="off"):
        m = _models()["qwen3.8-27b"]
        assert [c["id"] for c in m["community"]] == [SLUG], m["community"]
        c = m["community"][0]
        assert c["profile"] == PID and c["name"] == "ThinkingCap Qwen3.8-27B Kearuga"
        assert c["avatar_url"] == "https://github.com/0xWhiteMage.png?size=96"
        assert c["author_url"] == "https://github.com/0xWhiteMage"
        assert c["author"] == {"github": "0xWhiteMage", "name": "thewhitemage.",
                               "social": {"x": "0xWhiteMage", "huggingface": "0xWhiteMage",
                                          "website": "https://0xWhiteMage.github.io"}}
        assert [(s["network"], s["url"]) for s in c["social_links"]] == [
            ("x", "https://x.com/0xWhiteMage"), ("huggingface", "https://huggingface.co/0xWhiteMage"),
            ("website", "https://0xWhiteMage.github.io")], c["social_links"]
        assert c["hf"] == {"repo": "0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw", "revision": REV}
        assert c["hf_url"].endswith("/tree/" + REV)
        assert c["finetune_of"] == "bottlecapai/ThinkingCap-Qwen3.8-27B"
        assert (c["format"], c["bpw"], c["size_gb"], c["load"], c["drafters"]) == \
            ("exl3", 4.0, 16.38, "standalone", "tandem-exl3")
        assert c["modalities"] == {"text": True, "image": False, "audio": False, "video": False}
        assert c["modalities_in_pack"] == ["text", "image"]
        assert c["license"].startswith("PolyForm-Small-Business-1.0.0") and "Apache-2.0" in c["license"]
        assert c["status"] == "verified" and c["verified"]["tok_s"] == \
            {"prose": 39.6, "code": 95.9, "chat": 51.3, "plain": 15.2}
        assert c["verified"]["tokens_per_round"] == {"prose": 3.09, "code": 7.69, "chat": 4.11}
        assert (c["verified"]["build"], c["verified"]["device"], c["verified"]["date"]) == \
            ("0.8.2", "dgx-spark", "2026-10-13")
        assert c["links"] == {"pr": "https://github.com/0xBakeer/TandemLLM-early-access/pull/1"}
        assert c["memory"] == {"weights_gib": 14.76, "drafters_gib": 5.41, "like": "exl3-4.0"}
        p = registry.profile(m, PID)
        assert p["disk_gb"] == 16.38 and p["vision"] is False and p["hf_revision"] == REV
        assert p["assets_required"] == ["exl3-drafters-k3", "exl3-draft-head-k4", "nvfp4-drafters"]
        assert registry.profile(m, "community/nope") is None
        opts = registry.step_field(m, "weights", "profile")["options"]
        assert opts[-1] == PID and opts[:-1] == [x["id"] for x in m["profiles"]], opts
        from server.setup import tools
        sel = next(t for t in tools.definitions(m) if t["function"]["name"] == "select_profile")
        assert PID in sel["function"]["parameters"]["properties"]["id"]["enum"]
        # Flash-Next's are turboderp's packs its own profiles serve (VIS-175), and the payload holds
        # nothing of this machine
        assert [c["profile"] for c in _models()["qwen3.8-flash-next"]["community"]] == \
            [f"flashnext-{b}" for b in FN_BPW]
        assert os.path.expanduser("~") not in json.dumps(m["community"])


def test_the_schema_and_index_files():
    with open(os.path.join(ROOT, "models", "schema", "community.schema.json")) as fh:
        assert json.load(fh) == json.loads(json.dumps(registry.community_json_schema())), \
            "regenerate: python -m server.setup.registry --community-schema > models/schema/community.schema.json"
    with open(community.bundled_path(), encoding="utf-8") as fh:
        assert fh.read() == community.render_index(community.build_index()), \
            "regenerate: python -m server.setup.community --index"
    idx = community.build_index()
    assert [(e["model"], e["id"]) for e in idx["entries"]] == [(NH, NH_SLUG), ("qwen3.8-27b", SLUG)]
    import hashlib
    assert idx["entries"][1]["sha256"] == hashlib.sha256(open(ENTRY, "rb").read()).hexdigest()
    # the entries one of the model's own profiles serves are in community-builtin.json alone (VIS-175)
    with open(community.builtin_path(), encoding="utf-8") as fh:
        assert fh.read() == community.render_index(community.build_index(builtin=True)), \
            "regenerate: python -m server.setup.community --index"
    assert [(e["model"], e["id"]) for e in community.build_index(builtin=True)["entries"]] == \
        [("qwen3.8-flash-next", f"turboderp-flashnext-{b}") for b in FN_BPW]


BAD = [  # (change, the field the message names, a part of the message)
    (lambda e: e.update(schema=2), "$.schema", "schema 1 only"),
    (lambda e: e.update(id="other-name"), "$.id", "file's name"),
    (lambda e: e.update(base_model="qwen3.8-flash-next"), "$.base_model", "folder's model"),
    (lambda e: e["hf"].update(revision="main"), "$.hf.revision", "does not match"),
    (lambda e: e["hf"].update(revision=REV[:39]), "$.hf.revision", "does not match"),
    (lambda e: e["hf"].update(repo="no-owner"), "$.hf.repo", "does not match"),
    (lambda e: e.update(format="nvfp4"), "$.format", "no load path for nvfp4"),
    (lambda e: e.update(format="safetensors"), "$.format", "is not one of"),
    (lambda e: e["modalities"].update(image=True), "$.modalities.image", "ENG-271"),
    (lambda e: e.update(load="overlay") or e["modalities"].update(audio=True), "$.modalities.audio",
     "overlay load path does not serve audio"),
    (lambda e: e["modalities"].update(text=False), "$.modalities.text", "serves text"),
    (lambda e: e.update(modalities_in_pack=["image"]), "$.modalities_in_pack", "text works"),
    (lambda e: e.update(status="verified", verified=None), "$.verified", "what was measured"),
    (lambda e: e.update(added="2026-02-30"), "$.added", "not a date"),
    (lambda e: e.update(extra=1), "$", "unknown key 'extra'"),
    (lambda e: e["author"]["social"].update(x="bad handle!"), "$.author.social.x", "does not match"),
    (lambda e: e["author"]["social"].update(myspace="me"), "$.author.social", "unknown key 'myspace'"),
    (lambda e: e["author"]["social"].update(mastodon="http://m.social/@me"), "$.author.social.mastodon",
     "does not match"),
    (lambda e: e["author"]["social"].update(website="javascript:alert(1)"), "$.author.social.website",
     "does not match"),
    (lambda e: e["author"].update(github="-bad-"), "$.author.github", "does not match"),
    (lambda e: e["links"].update(pr="http://example.com/pr"), "$.links.pr", "does not match"),
    (lambda e: e.update(license_url="ftp://x/LICENSE"), "$.license_url", "does not match"),
    (lambda e: e.update(description="<script>alert(1)</script>"), "$.description", "does not match"),
    (lambda e: e.update(name="x" * 121), "$.name", "longer than 120"),
    (lambda e: e.update(tags=["a", "a"]), "$.tags", "twice"),
    (lambda e: e.update(id="foo.json"), "$.id", "may not end in"),
    (lambda e: e.update(id="Bad_Slug"), "$.id", "does not match"),
    (lambda e: e.update(bpw=40), "$.bpw", "above 16"),
    (lambda e: e.update(size_gb="16"), "$.size_gb", "expected number"),
]


def test_each_rule_refuses_a_bad_entry():
    models = _models()
    assert registry.community_errors(_entry(), SLUG, "qwen3.8-27b", models) == []
    for change, field, part in BAD:
        e = copy.deepcopy(_entry())
        change(e)
        slug = e["id"] if field == "$.id" and part != "file's name" else SLUG
        errs = registry.community_errors(e, slug, "qwen3.8-27b", models)
        hit = [x for x in errs if x.startswith(field + ":") and part in x] or \
              [x for x in errs if x.startswith(field) and part in x]
        assert hit, (field, part, errs)
    # a base model with no community load path, and one that is not there
    e = dict(_entry(), base_model="qwen3.8-flash-next")
    assert any("qwen3.8-27b, nemotron-3.5-super-vl-120b-a12b only" in x
               for x in registry.community_errors(e, SLUG, "qwen3.8-flash-next", models))
    e = dict(_entry(), base_model="no-such-model")
    assert any("no model 'no-such-model'" in x for x in registry.community_errors(e, SLUG, None, models))
    # an overlay may claim images (the base checkpoint's tower); a drafter choice needs the base's assets
    e = copy.deepcopy(_entry())
    e.update(load="overlay")
    e["modalities"]["image"] = True
    assert registry.community_errors(e, SLUG, "qwen3.8-27b", models) == []
    m = copy.deepcopy(models["qwen3.8-27b"])
    m["assets"] = [a for a in m["assets"] if a["id"] != "base-fp8"]
    errs = registry.community_errors(e, SLUG, "qwen3.8-27b", dict(models, **{"qwen3.8-27b": m}))
    assert any(x.startswith("$.drafters:") and "base-fp8" in x for x in errs), errs


def test_the_cli_names_the_file_and_field():
    tmp = tempfile.mkdtemp(prefix="cw-cli-")
    try:
        for name in os.listdir(ROOT):           # a checkout of its own, models/ copied
            if name not in ("models", ".git"):
                os.symlink(os.path.join(ROOT, name), os.path.join(tmp, name))
        md = os.path.join(tmp, "models")
        shutil.copytree(os.path.join(ROOT, "models"), md)
        bad = os.path.join(md, "qwen3.8-27b", "community", "bad-pack.json")
        json.dump(dict(_entry(), id="bad-pack", hf={"repo": _entry()["hf"]["repo"], "revision": "main"}),
                  open(bad, "w"))
        saved = registry.MODELS_DIR
        registry.MODELS_DIR = md
        try:
            with _Env(QSE_COMMUNITY_DIR=os.path.join(tmp, "c"), QSE_COMMUNITY_URL="off"):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = registry.main(["--check"])
                text = out.getvalue()
                assert rc == 1, text
                assert "FAIL models/qwen3.8-27b/community/bad-pack.json" in text, text
                assert "$.hf.revision: 'main' does not match" in text, text
                # the bundled index is now stale: it does not list bad-pack.json
                assert "models/community-index.json is not the folders' index" in text, text
                os.remove(bad)
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = registry.main(["--check"])
                assert rc == 0 and "7 packs, models/community-index.json matches" in out.getvalue(), out.getvalue()
        finally:
            registry.MODELS_DIR = saved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    r = subprocess.run([sys.executable, "-m", "server.setup.community", "--index", "--check"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_the_setup_path_takes_a_pack():
    from server.setup import apply, device, estimate
    tmp = tempfile.mkdtemp(prefix="cw-setup-")
    try:
        with _Env(QSE_COMMUNITY_DIR=tmp, QSE_COMMUNITY_URL="off"):
            m = _models()["qwen3.8-27b"]
            sel = estimate.check(m, {"model": m["id"], "profile": PID, "components": {"vision": True}})
            assert sel["profile"] == PID and sel["components"]["vision"] is False, sel
            assert apply.overlay_values(m, sel)["VISION"] == "off"
            est = estimate.estimate(m, sel)
            w = next(p for p in est["parts"] if p["id"] == "weights")
            assert w["gb"] == 14.76 and "ThinkingCap" in w["label"], w
            assert not any(p["id"] == "vision" for p in est["parts"]), est["parts"]
            # our own profile keeps the tool as sent
            own = estimate.check(m, {"model": m["id"], "profile": "exl3-4.0", "components": {"vision": True}})
            assert own["components"]["vision"] is True
            try:
                estimate.check(m, {"model": m["id"], "profile": "community/nope"})
                raise AssertionError("an unknown pack checked")
            except estimate.SelectionError as exc:
                assert exc.code == "unknown_profile"
            assert apply.profile_ok(PID) and not apply.profile_ok("community/nope")
            assert not apply.profile_ok("community/../etc")
            # the disk: not there, then a complete pack in its folder
            device.clear()
            assert device.on_disk(m, fresh=True)[PID]["weights"] is False
            d = os.path.join(tmp, SLUG)
            os.makedirs(d)
            json.dump({"weight_map": {"a": "model-00001-of-00001.safetensors"}},
                      open(os.path.join(d, "model.safetensors.index.json"), "w"))
            open(os.path.join(d, "config.json"), "w").write("{}")
            open(os.path.join(d, "model-00001-of-00001.safetensors"), "wb").write(b"w")
            got = device.on_disk(m, fresh=True)[PID]
            assert got["weights"] is True and got["dir"] == d, got
            device.clear()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _resolve(profile_env: str, keys: tuple) -> dict:
    """The keys as ops/serve.env resolves them for community/<id> with `profile_env` written where
    a dry run writes it (QSE_COMMUNITY_ENV_DIR); HOME a temporary folder, shown as ~."""
    with tempfile.TemporaryDirectory() as home:
        m = re.search(r"^REPO=\$\{HOME\}/(\S+)$", open(os.path.join(ROOT, "ops/serve.env")).read(), re.M)
        os.symlink(ROOT, os.path.join(home, m.group(1) if m else "qwen38-spark-engine"))
        cdir = os.path.join(home, "cenv")
        os.makedirs(os.path.join(cdir, "community"))
        open(os.path.join(cdir, "community", "x.env"), "w").write(profile_env)
        script = f"set -a; . {ROOT}/ops/serve.env; " + "; ".join(f'echo "{k}=${{{k}-<unset>}}"' for k in keys)
        env = {"HOME": home, "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "QSE_PROFILE_OVERRIDE": "community/x",
               "QSE_COMMUNITY_ENV_DIR": cdir, "QWEN38_EXL3": "/leak", "QWEN38_DRAFT_EXL3": "/leak"}
        out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True).stdout
        return {k: v.replace(home, "~") for k, v in (l.split("=", 1) for l in out.splitlines())}


KEYS = ("QSE_WEIGHTS", "QSE_WEIGHTS_HF", "QSE_WEIGHTS_REVISION", "QSE_WEIGHTS_DIR", "QSE_WEIGHTS_GB",
        "QWEN38_MODEL", "QWEN38_EXL3", "QWEN38_DRAFT_EXL3", "QWEN38_DRAFT_HEAD_EXL3", "CKPT8", "CKPT16",
        "QSE_ENGINE_FLAGS", "QSE_ASSETS", "VISION", "QSE_DISK_CACHE", "QWEN38_SAMPLE_NUCLEUS", "PORT", "NV")


def test_the_profile_through_serve_env():
    with _Env(QSE_COMMUNITY_DIR=tempfile.mkdtemp(prefix="cw-"), QSE_COMMUNITY_URL="off"):
        models = _models()
        m = models["qwen3.8-27b"]
        c = m["community"][0]
        e = _resolve(community.env(m, c), KEYS)
        d = f"~/.tandemllm/community/{SLUG}"
        assert e["QSE_WEIGHTS"] == PID and e["QSE_WEIGHTS_DIR"] == d and e["QSE_WEIGHTS_GB"] == "16.38", e
        assert e["QSE_WEIGHTS_HF"] == c["hf"]["repo"] and e["QSE_WEIGHTS_REVISION"] == REV, e
        # standalone: the pack is the model (#1's loader), no overlay; text only
        assert e["QWEN38_MODEL"] == d and e["QWEN38_EXL3"] == "<unset>" and e["VISION"] == "off", e
        assert e["QWEN38_DRAFT_EXL3"] == "~/.qwen38-spark-engine/weights/exl3-drafters-k3", e
        assert e["QWEN38_DRAFT_HEAD_EXL3"] == "~/.qwen38-spark-engine/weights/exl3-draft-head-k4", e
        assert e["CKPT8"].endswith("/train/ft-b8-v2") and e["CKPT16"].endswith("/train/ft-b16"), e
        assert e["QSE_ENGINE_FLAGS"] == "<unset>" and e["QSE_ASSETS"] == "<unset>", e
        assert e["QSE_DISK_CACHE"] == f"~/.qwen38-spark-engine/cache/disk-community-{SLUG}", e
        assert e["QWEN38_SAMPLE_NUCLEUS"] == "1" and e["PORT"] == _resolve("", ("PORT",))["PORT"], e
        # overlay: QWEN38_EXL3 over the base (serve.env's QWEN38_MODEL stays), the tower where images work
        oc = copy.deepcopy(_entry())
        oc.update(load="overlay")
        oc["modalities"]["image"] = True
        o = _resolve(community.env(m, registry.community_entry(oc, m)), KEYS)
        assert o["QWEN38_EXL3"] == d and o["QWEN38_MODEL"] == e_base("QWEN38_MODEL") and o["VISION"] == "on", o
        # dflash2: the BF16 drafters, no K3 files; none: no drafter, the model flag only on standalone
        oc.update(drafters="dflash2")
        o = _resolve(community.env(m, registry.community_entry(oc, m)), KEYS)
        assert o["QWEN38_DRAFT_EXL3"] == "<unset>" and o["CKPT8"].endswith("/train/ft-b8-v2"), o
        assert registry.community_assets(oc) == ["nvfp4-drafters", "base-fp8"]
        oc.update(drafters="none")
        o = _resolve(community.env(m, registry.community_entry(oc, m)), KEYS)
        assert o["QSE_ENGINE_FLAGS"] == "--drafter none", o
        sc = dict(_entry(), drafters="none")
        o = _resolve(community.env(m, registry.community_entry(sc, m)), KEYS)
        assert o["QSE_ENGINE_FLAGS"] == f"--model {d} --drafter none", o
        assert registry.community_entry(sc, m)["memory"]["drafters_gib"] == 0.0
        # the CLI: an unknown pack is refused, a known one printed
        r = subprocess.run([sys.executable, "-m", "server.setup.community", "env", "community/nope"], cwd=ROOT,
                           capture_output=True, text=True)
        assert r.returncode == 2 and "no community entry" in r.stderr, r.stderr
        r = subprocess.run([sys.executable, "-m", "server.setup.community", "env", PID], cwd=ROOT,
                           capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout == community.env(m, c), r.stdout + r.stderr


def e_base(key: str) -> str:
    """`key` as serve.env alone resolves it (no profile)."""
    with tempfile.TemporaryDirectory() as home:
        out = subprocess.run(["bash", "-c", f'set -a; . {ROOT}/ops/serve.env; echo "${{{key}-<unset>}}"'],
                             env={"HOME": home, "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                             capture_output=True, text=True).stdout.strip()
        return out.replace(home, "~")


def test_the_card_names_the_pack():
    from server import model_card
    with _Env(QSE_COMMUNITY_DIR=tempfile.mkdtemp(prefix="cw-"), QSE_COMMUNITY_URL="off"):
        c = model_card.find(PID)
        assert c["label"] == "ThinkingCap Qwen3.8-27B Kearuga" and c["model"] == "qwen3.8-27b", c
        assert c["capabilities"]["vision"] is False and c["capabilities"]["thinking"] is True, c
        card = model_card.card({"pack": PID, "max_len": 262144, "vision": True, "label": "x"})
        assert card["label"] == "ThinkingCap Qwen3.8-27B Kearuga" and card["capabilities"]["vision"] is False
        assert model_card.find("community/nope") is None
        assert model_card.find("exl3-4.0")["label"] == "Qwen3.8-27B"


class _Hub:
    """A local index server: `body` and `etag`, `delay` seconds before it answers, every request
    noted with its If-None-Match."""

    def __init__(self):
        hub = self
        self.body, self.etag, self.delay, self.status, self.seen = b"", '"v1"', 0.0, 200, []

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                hub.seen.append(self.headers.get("If-None-Match"))
                time.sleep(hub.delay)
                if hub.status != 200:
                    self.send_response(hub.status)
                    self.end_headers()
                    return
                if self.headers.get("If-None-Match") == hub.etag:
                    self.send_response(304)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("ETag", hub.etag)
                self.send_header("Content-Length", str(len(hub.body)))
                self.end_headers()
                try:
                    self.wfile.write(hub.body)
                except (BrokenPipeError, ConnectionResetError):
                    pass                        # the timeout case: the client gave up

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/models/community-index.json"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _index(*entries: dict) -> bytes:
    doc = community.build_index()
    for e in entries:
        doc["entries"].append({"model": "qwen3.8-27b", "id": e["id"],
                               "path": f"models/qwen3.8-27b/community/{e['id']}.json", "sha256": "0" * 64,
                               "entry": e})
    return json.dumps(doc).encode()


def test_the_sync():
    hub = _Hub()
    tmp = tempfile.mkdtemp(prefix="cw-sync-")
    second = dict(copy.deepcopy(_entry()), id="second-pack", added="2026-10-14", name="Second pack")
    broken = dict(copy.deepcopy(_entry()), id="broken-pack")
    broken["hf"]["revision"] = "main"
    try:
        with _Env(QSE_COMMUNITY_DIR=tmp, QSE_COMMUNITY_URL=hub.url):
            # no fetch yet: the bundle; nothing is fetched by reading
            st = community.state()
            assert st["source"] == "bundled" and st["fetched_at"] is None and hub.seen == [], st
            assert [c["id"] for c in _models()["qwen3.8-27b"]["community"]] == [SLUG]
            # 200: into the cache; the invalid entry dropped with its reason; newest first
            hub.body = _index(second, broken)
            st = community.sync()
            assert st["source"] == "github" and st["error"] is None and st["fetched_at"], st
            assert hub.seen == [None], hub.seen
            assert os.path.isfile(os.path.join(tmp, "index.json"))
            assert json.load(open(os.path.join(tmp, "index.meta.json")))["etag"] == '"v1"'
            assert [c["id"] for c in _models()["qwen3.8-27b"]["community"]] == ["second-pack", SLUG]
            assert [(d["id"], d["path"]) for d in st["dropped"]] == \
                [("broken-pack", "models/qwen3.8-27b/community/broken-pack.json")], st["dropped"]
            assert "$.hf.revision" in st["dropped"][0]["reason"], st["dropped"]
            first_fetch = st["fetched_at"]
            # 304: the ETag goes along, the cache stays, still github
            st = community.sync()
            assert hub.seen[-1] == '"v1"' and st["source"] == "github" and st["fetched_at"] == first_fetch, st
            # a timeout: the cache, with the reason
            hub.delay = 2.0
            st = community.sync(timeout=0.3)
            assert st["source"] == "cache" and st["error"] == "GitHub did not answer within 0.3 s", st
            assert "second-pack" in json.dumps(community_ids())
            hub.delay = 0.0
            # an index of the wrong shape is not written over the cache
            hub.etag, hub.body = '"v2"', b'{"entries": "nope"}'
            st = community.sync()
            assert st["source"] == "cache" and st["error"] == "the published list has the wrong shape", st
            assert "second-pack" in community_ids()
            # an HTTP error: the cache, a short error without the URL (the server's log has it)
            hub.status = 503
            out = io.StringIO()
            with redirect_stdout(out):
                st = community.sync()
            assert st["source"] == "cache" and st["error"] == "GitHub answered HTTP 503", st
            assert hub.url not in json.dumps(st["error"]) and hub.url in out.getvalue(), out.getvalue()
            # a 404: no published list yet, which is no error (public main before tonight's release)
            hub.status = 404
            st = community.sync()
            assert st["source"] == "cache" and st["error"] is None and st["note"] == "no published list yet", st
            hub.status = 200
            # ten page loads at once: one request
            hub.etag, hub.body, hub.delay = '"v3"', _index(second), 0.5
            n = len(hub.seen)
            outs = []
            ts = [threading.Thread(target=lambda: outs.append(community.sync())) for _ in range(10)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            assert len(hub.seen) == n + 1 and len(outs) == 10, hub.seen
            assert all(o["source"] == "github" and o["dropped"] == [] for o in outs), outs
            hub.delay = 0.0
        # off: no request, the cache still serves
        with _Env(QSE_COMMUNITY_DIR=tmp, QSE_COMMUNITY_URL="off"):
            n = len(hub.seen)
            st = community.sync()
            assert len(hub.seen) == n and st["source"] == "cache" and st["error"] is None, st
            assert "off" in st["note"] and st["url"] is None, st
        # no cache and nothing answers: the bundle
        with _Env(QSE_COMMUNITY_DIR=os.path.join(tmp, "empty"),
                  QSE_COMMUNITY_URL="http://127.0.0.1:9/models/community-index.json"):
            st = community.sync(timeout=2)
            assert st["source"] == "bundled" and st["error"] == "GitHub could not be reached", st
            assert community_ids() == [SLUG]
    finally:
        hub.close()
        shutil.rmtree(tmp, ignore_errors=True)


def community_ids() -> list[str]:
    return [c["id"] for c in _models()["qwen3.8-27b"]["community"]]


class _H:
    """The request handler the setup routes write to."""

    def __init__(self):
        self.out, self.headers = None, {"Content-Type": "application/json", "Content-Length": "2"}

    def _allowed(self, kind):
        return True

    def _json(self, status, body, extra_headers=()):
        self.out = (status, json.loads(json.dumps(body)))


def test_a_pack_an_own_profile_serves():
    from server import model_card
    from server.setup import apply, device
    models = _models()
    m = models["qwen3.8-flash-next"]
    for b in FN_BPW:
        c = next(x for x in m["community"] if x["id"] == f"turboderp-flashnext-{b}")
        own = registry.profile(m, f"flashnext-{b}")
        assert c["profile"] == own["id"] and c["hf"]["repo"] == own["hf_repo"] == "turboderp/Qwen3.8-Flash-Next-exl3"
        assert re.fullmatch(r"[0-9a-f]{40}", c["hf"]["revision"]) and c["author"]["github"] == "turboderp"
        assert own["made_by"] == {"kind": "community", "name": "turboderp", "github": "turboderp"}
        assert c["memory"]["like"] == own["id"] and c["drafters"] == "in-pack"
        assert apply.profile_ok("community/" + c["id"]) is False
        assert model_card.find("community/" + c["id"]) is None
    assert [p["id"] for p in registry.profiles(m)] == [p["id"] for p in m["profiles"]]
    assert registry.community(m, "community/turboderp-flashnext-3.05") is None
    assert registry.step_field(m, "weights", "profile")["options"] == [p["id"] for p in m["profiles"]]
    statuses = {c["id"].rsplit("-", 1)[1]: c["status"] for c in m["community"]}
    assert statuses == {"2.05": "verified", "3.05": "verified", "4.05": "verified", "5.05": "broken",
                        "6.05": "submitted"}
    # the on-disk map keeps the profiles' own folders and adds none
    disk = device.on_disk(m, fresh=True)
    assert sorted(disk) == sorted(p["id"] for p in m["profiles"])
    # use-profile.sh community/<id> points at the profile
    r = subprocess.run([sys.executable, "-m", "server.setup.community", "env", "community/turboderp-flashnext-3.05"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "ops/use-profile.sh flashnext-3.05" in r.stderr, r.stderr
    # what the check refuses
    base = json.load(open(FN_ENTRY))
    assert registry.community_errors(base, base["id"], "qwen3.8-flash-next", models) == []
    for change, field, part in [
            (lambda e: e["hf"].update(repo="someone/Qwen3.8-Flash-Next-exl3"), "$.hf.repo", "is not profile"),
            (lambda e: e.update(bpw=4.05), "$.format", "is not profile flashnext-3.05's"),
            (lambda e: e.update(profile="flashnext-9.05"), "$.profile", "has no profile"),
            (lambda e: e["modalities"].update(audio=True) or e.update(modalities_in_pack=["text", "image", "audio"]),
             "$.modalities.audio", "does not serve audio"),
            (lambda e: e["hf"].update(revision="3.05bpw_h5_ng5"), "$.hf.revision", "does not match")]:
        e = copy.deepcopy(base)
        change(e)
        errs = registry.community_errors(e, e["id"], "qwen3.8-flash-next", models)
        assert any(x.startswith(field) and part in x for x in errs), (field, errs)
    e = dict(_entry(), drafters="in-pack")
    assert any(x.startswith("$.drafters:") and "own profiles only" in x
               for x in registry.community_errors(e, SLUG, "qwen3.8-27b", models))
    # a fetched index with an entry of the same id does not replace the release's
    tmp = tempfile.mkdtemp(prefix="cw-own-")
    try:
        fake = dict(base, profile="flashnext-2.05", bpw=2.05, description="replaced")
        doc = {"schema": 1, "generator": "t", "entries": [
            {"model": "qwen3.8-flash-next", "id": base["id"], "path": "models/qwen3.8-flash-next/community/"
             + base["id"] + ".json", "sha256": "0" * 64, "entry": fake}]}
        with _Env(QSE_COMMUNITY_DIR=tmp, QSE_COMMUNITY_URL="off"):
            with open(community.cache_path(), "w") as fh:
                json.dump(doc, fh)
            got = [c for c in registry.load_all()[0]["qwen3.8-flash-next"]["community"] if c["id"] == base["id"]]
            assert len(got) == 1 and got[0]["profile"] == "flashnext-3.05", got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # the index every box fetches holds no such entry: a box before 0.11.1 would drop it
    with open(community.bundled_path()) as fh:
        assert all("profile" not in r["entry"] for r in json.load(fh)["entries"])


def _nh_entry() -> dict:
    with open(NH_ENTRY) as fh:
        return json.load(fh)


def test_a_nemotron_pack():
    from server import model_card
    from server.setup import estimate
    with _Env(QSE_COMMUNITY_DIR=tempfile.mkdtemp(prefix="cw-nh-"), QSE_COMMUNITY_URL="off"):
        models = _models()
        m = models[NH]
        assert registry.community_errors(_nh_entry(), NH_SLUG, NH, models) == []
        c = registry.community(m, NH_PID)
        assert c is not None and c["author"]["github"] == "styles01" and c["load"] == "standalone"
        assert c["modalities"]["image"] is False and c["modalities_in_pack"] == ["text"]
        assert "TensorFold" not in c["description"] and "EXL3 (exllamav3)" in c["description"]
        # the nearest own EXL3 profile gives the memory and the drafters, whatever `drafters` says
        assert c["memory"] == {"weights_gib": 58.35, "drafters_gib": 1.32, "like": "nemotron-super-vl-exl3-4.0"}, c
        p = registry.profile(m, NH_PID)
        assert p["assets_required"] == ["nemotron-super-vl-lilicorr"] and p["vision"] is False, p
        assert registry.community_engine(c) == "nemotron_h" and registry.community_engine(_entry()) == "qwen38"
        sel = estimate.check(m, {"model": NH, "profile": NH_PID, "components": {"vision": True}})
        assert sel["components"].get("vision") is False, sel
        # what the check refuses on this base
        for change, field, part in [
                (lambda e: e.update(load="overlay"), "$.load", "standalone only"),
                (lambda e: e.update(drafters="tandem-exl3"), "$.drafters", "takes none"),
                (lambda e: e["modalities"].update(image=True) or e.update(modalities_in_pack=["text", "image"]),
                 "$.modalities.image", "does not serve image"),
                (lambda e: e.update(modalities_in_pack=["image"]), "$.modalities_in_pack", "text works"),
                (lambda e: e.update(format="nvfp4"), "$.format", "no load path for nvfp4")]:
            e = copy.deepcopy(_nh_entry())
            change(e)
            errs = registry.community_errors(e, NH_SLUG, NH, models)
            assert any(x.startswith(field) and part in x for x in errs), (field, errs)
        bolt = dict(_nh_entry(), base_model="nemotron-3.5-bolt-74b-a9b")
        errs = registry.community_errors(bolt, NH_SLUG, "nemotron-3.5-bolt-74b-a9b", models)
        assert any(x.startswith("$.base_model") and "qwen3.8-27b, nemotron-3.5-super-vl-120b-a12b only" in x
                   for x in errs), errs
        # the profile through serve.env: the 4.0 profile's limits and drafters, the pack in its place
        keys = ("QSE_WEIGHTS", "QSE_WEIGHTS_HF", "QSE_WEIGHTS_REVISION", "QSE_WEIGHTS_DIR", "QSE_WEIGHTS_GB",
                "QSE_NH_MODEL", "QSE_NH_DIR", "QSE_NH_HEAD", "QSE_NH_LILICORR", "QSE_ENGINE_FLAGS", "QSE_ASSETS",
                "VISION", "QSE_DISK_CACHE", "QSE_PAGE_CACHE_DROP", "QSE_MEM_NEED_GIB", "MAX_LEN", "QSE_PARALLEL",
                "QWEN38_SUFFIX_STORE", "QWEN38_EXL3", "QWEN38_MODEL")
        e = _resolve(community.env(m, c), keys)
        d = f"~/.tandemllm/community/{NH_SLUG}"
        assert e["QSE_WEIGHTS"] == NH_PID and e["QSE_WEIGHTS_DIR"] == d == e["QSE_NH_DIR"], e
        assert e["QSE_WEIGHTS_HF"] == c["hf"]["repo"] and e["QSE_WEIGHTS_REVISION"] == c["hf"]["revision"], e
        assert e["QSE_WEIGHTS_GB"] == "57.8" and e["QSE_NH_MODEL"] == NH and e["QSE_NH_HEAD"] == "lilicorr", e
        assert e["QSE_ENGINE_FLAGS"] == f"--model {d} --drafter mtp --draft-head lilicorr --depth 3", e
        assert e["QSE_ASSETS"] == "~/models/nemotron-3.5-super-vl-lilicorr/model.safetensors", e
        assert e["VISION"] == "off" and e["QSE_PAGE_CACHE_DROP"] == d and e["QWEN38_EXL3"] == "<unset>", e
        assert e["QSE_DISK_CACHE"] == f"~/.qwen38-spark-engine/cache/disk-community-{NH_SLUG}", e
        assert (e["QSE_MEM_NEED_GIB"], e["MAX_LEN"], e["QSE_PARALLEL"]) == ("92", "262144", "3"), e
        assert e["QWEN38_SUFFIX_STORE"] == "~/.qwen38-spark-engine/suffix-nemotron", e
        # the served card is Super VL's, named after the pack, vision off
        card = model_card.find(NH_PID)
        assert card["model"] == NH and card["label"] == c["name"] and card["capabilities"]["vision"] is False, card


def test_the_nemotron_card_is_the_raw_data():
    with open(os.path.join(ROOT, "models", NH, "measurements", "0.12.1-summary.json")) as fh:
        meas = json.load(fh)
    run = meas["profiles"][NH_PID]
    plain, drafted = run["plain_runs"][0]["classes"], run["runs"][0]["classes"]
    with open(NH_ENTRY) as fh:
        v = json.load(fh)["verified"]
    assert (v["build"], v["device"], v["date"]) == (meas["build"], meas["device"], run["date"]), v
    assert v["tok_s"] == {**{k: round(drafted[k]["tok_s"], 1) for k in ("prose", "code", "chat")},
                          "plain": round(plain["prose"]["tok_s"], 1)}, v
    assert v["tokens_per_round"] == {k: drafted[k]["tok_round"] for k in ("prose", "code", "chat")}, v
    assert all(run["same_text"].values()) and all(c["same_text_as_plain"] for c in drafted.values()), run
    assert run["agreement"]["top1_agree"] == 510 and run["agreement"]["positions"] == 512, run["agreement"]
    assert run["load"]["weights_gib"] == 53.58 and "53.58 GiB" in v["notes"], run["load"]


def _nh_pack(d: str, raw: dict | None = None, keys: tuple = (), shards: int = 2) -> None:
    """A pack folder with PR #63's config and an index of its names (the shards empty files)."""
    with open(os.path.join(ROOT, "tests", "fixtures", "nemotron_h", "super-vl-exl3-3.5-community.json")) as fh:
        cfg = json.load(fh)
    json.dump(raw if raw is not None else cfg, open(os.path.join(d, "config.json"), "w"))
    names = ["backbone.embeddings.weight", "lm_head.weight", *keys]
    wm = {k: f"model-{i % shards + 1:05d}-of-{shards:05d}.safetensors" for i, k in enumerate(names)}
    for fn in set(wm.values()):
        open(os.path.join(d, fn), "wb").close()
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    return cfg


def test_the_nemotron_file_check():
    with _Env(QSE_COMMUNITY_DIR=tempfile.mkdtemp(prefix="cw-nhf-"), QSE_COMMUNITY_URL="off"):
        m = _models()[NH]
        c = registry.community(m, NH_PID)
        with tempfile.TemporaryDirectory() as d:
            cfg = _nh_pack(d)
            assert community.check_pack(m, c, d) == []
            # images listed: the tower is not in this pack
            ci = dict(c, modalities_in_pack=["text", "image"])
            errs = community.check_pack(m, ci, d)
            assert "vision_model.radio_model.* not found in model.safetensors.index.json" in errs, errs
            assert any("preprocessor_config.json" in x for x in errs) and any("vision_config" in x for x in errs)
            # a pack with the tower passes the image check
            _nh_pack(d, dict(cfg, vision_config={"model_type": "radio"}),
                     keys=("vision_model.radio_model.model.blocks.0.attn.qkv.weight", "mlp1.0.weight"))
            open(os.path.join(d, "preprocessor_config.json"), "w").write("{}")
            assert community.check_pack(m, ci, d) == []
            for bad, want in ((dict(cfg, model_type="qwen3_5"), "config.json: model_type 'qwen3_5', not nemotron_h"),
                              (dict(cfg, layers_block_type=cfg["layers_block_type"][:80],
                                    hybrid_override_pattern=cfg["hybrid_override_pattern"][:80]),
                               "config.json: 88 layers expected, 80 found"),
                              (dict(cfg, quantization_config={"quant_method": "fp8"}),
                               "config.json: quantization_config.quant_method 'fp8', not exl3")):
                _nh_pack(d, bad)
                assert want in community.check_pack(m, c, d), (want, community.check_pack(m, c, d))
            _nh_pack(d, keys=("backbone.norm_f.weight",), shards=3)
            os.remove(os.path.join(d, "model-00003-of-00003.safetensors"))
            errs = community.check_pack(m, c, d)
            assert errs == ["model-00003-of-00003.safetensors: the index names it, the folder does not have it"], errs
            # the CLI: the folder given, ok and exit 0 on a good pack
            _nh_pack(d)
            r = subprocess.run([sys.executable, "-m", "server.setup.community", "check-pack", NH_PID, "--dir", d],
                               cwd=ROOT, capture_output=True, text=True,
                               env=dict(os.environ, QSE_COMMUNITY_URL="off"))
            assert r.returncode == 0 and r.stdout.startswith("ok"), r.stdout + r.stderr
        # a 27B pack has no file check of this kind
        assert community.check_pack(_models()["qwen3.8-27b"], _models()["qwen3.8-27b"]["community"][0], "/nope") == []


def test_the_api():
    from server import setup_api
    hub = _Hub()
    tmp = tempfile.mkdtemp(prefix="cw-api-")
    try:
        with _Env(QSE_COMMUNITY_DIR=tmp, QSE_COMMUNITY_URL=hub.url):
            hub.status = 404                   # the public main has no index yet: the bundle
            h = _H()
            setup_api._get(h, None, ["models"])
            tiles = {t["id"]: t for t in h.out[1]["models"]}
            assert tiles["qwen3.8-27b"]["community_count"] == 1 and tiles["qwen3.8-flash-next"]["community_count"] == len(FN_BPW)
            setup_api._get(h, None, ["models", "qwen3.8-27b"])
            body = h.out[1]
            assert [c["profile"] for c in body["community"]] == [PID], body["community"]
            cs = body["community_sync"]
            assert cs["source"] == "bundled" and cs["dropped"] == [] and "base_model=qwen3.8-27b" in cs["add_url"], cs
            assert cs["add_url"].startswith("https://github.com/0xBakeer/TandemLLM/issues/new?template=community-quant.yml")
            assert hub.seen == [], "a GET fetched"
            setup_api._get(h, None, ["community"])
            assert h.out[0] == 200 and list(h.out[1]["models"]["qwen3.8-27b"][0]) == list(body["community"][0])
            setup_api._post(h, None, ["community", "sync"], {})
            sync = h.out[1]["sync"]
            assert len(hub.seen) == 1 and sync["source"] == "bundled", sync
            assert sync["error"] is None and sync["note"] == "no published list yet", sync
            hub.status, hub.body = 200, _index()
            setup_api._post(h, None, ["community", "sync"], {})
            assert h.out[1]["sync"]["source"] == "github" and h.out[1]["sync"]["fetched_at"], h.out
            assert [c["id"] for c in h.out[1]["models"]["qwen3.8-27b"]] == [SLUG]
            # the estimate route takes the pack and turns vision off
            setup_api._post(h, None, ["estimate"], {"selection": {"model": "qwen3.8-27b", "profile": PID}})
            assert h.out[0] == 200, h.out
            assert next(p for p in h.out[1]["parts"] if p["id"] == "weights")["gb"] == 14.76, h.out
            assert not any(p["id"] == "vision" for p in h.out[1]["parts"]), h.out
    finally:
        hub.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _hub_reachable() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8", timeout=10)
        return True
    except OSError:
        return False


def test_the_fetch_plan_of_the_first_entry():
    if not _hub_reachable():
        print("    (skipped: huggingface.co does not answer)")
        return
    sys.path.insert(0, os.path.join(ROOT, "ops"))
    import fetch_weights
    saved = {k: os.environ.pop(k, None) for k in ("HF_TOKEN",)}
    saved_token = fetch_weights._token
    fetch_weights._token = lambda: None         # no token: the pack is public
    try:
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            with redirect_stdout(out):
                m, left = fetch_weights.plan(_entry()["hf"]["repo"], os.path.join(d, SLUG), [], REV)
            text = out.getvalue()
        assert m["commit"] == REV, m["commit"]
        base = f"https://huggingface.co/{_entry()['hf']['repo']}/resolve/{REV}/"
        for f in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                  "model.safetensors.index.json", "config.json", "tokenizer.json", "LICENSE"):
            assert base + f in text, (f, text)
        assert round(left / 1e9, 2) == _entry()["size_gb"] == 16.38, left
        shas = {n: f["sha256"] for n, f in m["files"].items() if n.endswith(".safetensors")}
        assert len(shas) == 2 and all(re.fullmatch(r"[0-9a-f]{64}", s) for s in shas.values()), shas
    finally:
        fetch_weights._token = saved_token
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_use_profile_dry_run():
    if sys.platform == "darwin":
        print("    (skipped: GNU df -- run it on the box)")
        return
    if not _hub_reachable():
        print("    (skipped: huggingface.co does not answer)")
        return
    import test_weight_profiles as twp
    run = os.path.join(ROOT, "run", "community")
    before = sorted(os.listdir(run)) if os.path.isdir(run) else None
    r = twp._use_profile(PID, "--dry-run")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert f"{PID}: profile /tmp" in out or f"{PID}: profile " in out, out
    assert f"0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw@{REV} (~16.38 GB)" in out, out
    assert f"/resolve/{REV}/model-00001-of-00002.safetensors" in out, out
    for want in ("exl3-drafters-k3/ft-b8-v2.safetensors", "exl3-draft-head-k4/model.safetensors",
                 "TandemLLM-Qwen3.8-27B-DFlash2-b8/resolve/", "dry run: nothing downloaded",
                 "then 16.38 GB of weights", "dry run: would fetch", f"echo {PID} >"):
        assert want in out, (want, out)
    assert "Qwen/Qwen3.8-27B-FP8" not in out and "REFUSING" not in out, out      # standalone: no base
    after = sorted(os.listdir(run)) if os.path.isdir(run) else None
    assert after == before, (before, after)
    # an id that is not in the index, and a path out of community/
    for bad in ("community/nope", "community/../x", "community/a/b"):
        r = twp._use_profile(bad, "--dry-run")
        assert r.returncode == 2, (bad, r.stdout + r.stderr)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
