"""The setup guide's model registry (OPS-91): models/<id>/model.yaml, schema 1, its loader and its
validator (server/setup/registry.py).

The test matrix:

  1  the seed loads: 7 profiles, one recommended (exl3-4.5), the 4.0 badge, the NVFP4 row marked
     not comparable with its reason, cells as objects, a payload json.dumps takes and that holds no
     path of this machine                                   test_the_seed
  2  the measured table is the run's own numbers: every cell, TTFT, lossless count and the largest
     near tie equal measurements/0.6.2-summary.json         test_the_table_is_the_raw_data
  3  every serve.env key the file names is read by something; a planted QWEN38_/QSE_/plain key that
     nothing reads is caught                                test_every_knob_is_registered
  4  every ops/profiles/<name>.env has an entry and every entry a file, with the file's HF repo and
     size; a planted extra profile is caught                test_profiles_match_ops
  5  each validator rule refuses a bad file, naming the YAML path: a newer schema, two recommended
     profiles, an unknown dependency, a cycle, a default whose dependency is off, a conflict on by
     default, YAML's on/off booleans as knob keys, an unknown asset, a docs file that is not there,
     a default out of range, a row of another build marked comparable, a non-comparable row without
     its reason, a quality value without date, an unknown key   test_the_validator_rules
  6  a second folder lists with no code change; a broken file is reported while the others still
     list; a folder without model.yaml is skipped; a folder named after another id is refused
                                                            test_more_models
  7  models/schema/model.schema.json is the schema the code checks   test_the_schema_file
  8  the CLI: exit 0 on the seed, 1 and the YAML path on a bad file  test_the_cli
  9  the step schema: the eight steps in order, a field per tool, the profile and enum options, slots
     up to the engine's 16 (a joint pass of 32 rows, 2 a chat at least)   test_wizard_fields, test_engine_max_slots
                                                            test_wizard_fields
 10  a string a client sends is a plain value or nothing: newlines, $(...), quotes and spaces are
     refused, and numbers keep their range                  test_field_values
 11  the KV bytes a token are the server's (server/slots.py) and the units add up: 104.25 KiB,
     26.06 GiB = 27.98 GB at 262,144 tokens                 test_kv_units
 12  the dashboard mock's copies (dashboard/mock/setup/<id>.json: the 27B, Flash-Next, LFM2.5, Bolt,
     Super VL) are what GET /v1/setup/models/<id> answers, key for key        test_the_mock_copy

 13  LFM2.5's card is its runs' own numbers: per profile the disk size and commit of its Hugging Face
     repository, the weights at load, the chat KL and top-1 (exllamav3's model_diff, the engine's own beside
     it), every measured cell, and its profile file's
     size and revision equal measurements/0.8.3-summary.json   test_the_lfm25_card_is_the_raw_data
 14  Bolt's and Super VL's tables are their runs (Bolt 0.11.1, Super VL 0.11.0): per measured profile one
     comparable row, each greedy-512 cell the mean, minimum, ms and tokens a round of its 3 runs, the same
     text in every run, no request from a cache; where plain servers ran beside them (Bolt, SPD-127), every
     class's drafted mean is at least its plain mean less the plain runs' spread; the tile's speed is the
     recommended row's best cell
                                                            test_the_nemotron_tables_are_the_raw_data
 15  every weight profile says who made it (VIS-175): ours (0xBakeer), the vendor (NVIDIA, LiquidAI) or a
     community author with a GitHub login (turboderp); Bolt's IQ2 names NVIDIA's GGUF repository; a
     profile without `made_by`, a community maker without a login, an unknown kind and a repo that is
     not owner/name are refused                             test_every_pack_names_its_maker
 16  dashboard/mock/setup/community-qwen3.8-flash-next.json is the API's list of turboderp's packs
                                                            test_the_builtin_community_mock

Run: python tests/test_setup_registry.py
"""

from __future__ import annotations

import ast
import copy
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

# the community packs from the bundled index, never a box's fetched copy (tests/test_community_weights.py)
os.environ["QSE_COMMUNITY_DIR"] = tempfile.mkdtemp(prefix="registry-community-")
os.environ["QSE_COMMUNITY_URL"] = "off"

from server.setup import registry  # noqa: E402

ROOT = registry.ROOT
SEED = os.path.join(ROOT, "models", "qwen3.8-27b", "model.yaml")


def _seed() -> dict:
    with open(SEED) as fh:
        return yaml.safe_load(fh)


def test_the_seed():
    m = registry.load(SEED)
    assert m["schema"] == 1 and m["id"] == "qwen3.8-27b"
    ids = [p["id"] for p in m["profiles"]]
    assert sorted(ids) == ["exl3-2.5", "exl3-3.0", "exl3-3.5", "exl3-4.0", "exl3-4.5", "exl3-6.0",
                           "nvfp4"], ids
    assert registry.recommended_profile(m)["id"] == "exl3-4.5"
    assert "best speed for quality" in registry.profile(m, "exl3-4.0")["badges"]
    nv = next(r for r in m["measurements"]["rows"] if r["profile"] == "nvfp4")
    assert nv["comparable"] is False and "0.4.8" in nv["note"] and "not comparable" in nv["note"]
    cell = next(r for r in m["measurements"]["rows"]
                if r["profile"] == "exl3-4.5")["cells"]["greedy_512"]["prose"]
    assert cell == {"tok_s": 41.23, "min": 40.7, "ms_round": 83.9, "tok_round": 3.46, "n": 6}, cell
    text = json.dumps(m)
    assert "quality_source" not in m
    # the registry names paths by ${HOME}, ${REPO} and ${HF_HUB}, never this machine's
    for bad in ("/home/", "/Users/", os.path.expanduser("~")):
        assert bad not in text, bad
    t = registry.tile(m)
    assert t["best_speed"] == {"tok_s": 84.44, "task": "code", "profile": "exl3-4.5"}, t
    assert t["recommended"] is True
    # every tool's memory and default is there, and the defaults are the reference box's but the
    # narrator, off by default since 0.11.3 (SRV-534)
    on = {c["id"] for c in m["components"] if c["default"]}
    assert on == {"drafters", "lookup", "conductor", "laya", "shadow", "vision",
                  "prompt_cache", "state_store", "disk_tier", "nvme_spill"}, on


def test_the_table_is_the_raw_data():
    m = registry.load(SEED)
    with open(os.path.join(ROOT, "models", "qwen3.8-27b", "measurements", "0.6.2-summary.json")) as fh:
        raw = json.load(fh)
    modes = {"g512": "greedy_512", "g128": "greedy_128", "s512": "sampled_512",
             "s128": "sampled_128", "long-g": "long_greedy", "long-s": "long_sampled"}
    n = 0
    for size, d in raw.items():
        row = next(r for r in m["measurements"]["rows"] if r["profile"] == f"exl3-{size}")
        assert row["build"] == "0.6.2" and row["comparable"]
        for k, mode in modes.items():
            for task, c in d[k].items():
                if task == "files":
                    continue
                got = row["cells"][mode][task]
                want = {"tok_s": c["mean"], "min": c["min"], "ms_round": c["ms_round"],
                        "tok_round": c["tok_round"], "n": c["n"]}
                if k.startswith("long"):
                    want["ttft_s"] = c["ttft"]
                assert got == want, (size, mode, task, got, want)
                n += 1
        assert {k: float(v) for k, v in row["ttft_s"].items()} == \
            {k: float(v) for k, v in d["ttft"].items()}
        counts = ast.literal_eval(d["lossless"]["counts"])      # the summary keeps a dict's repr
        ulps = [json.loads(r[2]).get("ulp") for r in d["lossless_rows"]]
        assert row["lossless"] == {"identical": counts["identical"], "near_tie": counts["near-tie"],
                                   "other": counts["other"],
                                   "max_near_tie_ulp": max(u for u in ulps if u is not None)}
        # the load stages behind weights_gib and drafters_gib
        stages = dict(re.findall(r"(\w+) ([\d.]+)/", d["load_stages"]))
        p = registry.profile(m, f"exl3-{size}")
        assert p["weights_gib"] == float(stages["weights"]), (size, p["weights_gib"])
        assert p["drafters_gib"] == float(stages["drafters"]), (size, p["drafters_gib"])
    assert n == 6 * (4 * 3 + 2 * 2), n


def test_the_lfm25_card_is_the_raw_data():
    m = registry.load(os.path.join(ROOT, "models", "lfm2.5-24b-a2b", "model.yaml"))
    with open(os.path.join(ROOT, "models", "lfm2.5-24b-a2b", "measurements", "0.8.3-summary.json")) as fh:
        raw = json.load(fh)
    n = 0
    for size in ("bf16", "2.5", "3.0", "3.5", "4.0", "4.5", "6.0"):
        d = raw[size]
        pid = "lfm25-bf16" if size == "bf16" else f"lfm25-exl3-{size}"
        p = registry.profile(m, pid)
        assert p["weights_gib"] == d["decode_step"]["allocated_gib"] or size == "bf16", (pid, p["weights_gib"])
        row = next(r for r in m["measurements"]["rows"] if r["profile"] == pid)
        assert row["build"] == "0.8.3" and row["comparable"] and all(d["rows"]["same_text"].values()), pid
        for task in ("prose", "chat", "code"):
            c, got = d["rows"]["lookup"][task], row["cells"]["greedy_512"][task]
            assert got == {"tok_s": c["tok_s"], "min": c["tok_s"], "ms_round": round(1000 * c["per_block"] / c["tok_s"], 2),
                           "tok_round": c["per_block"], "n": 1}, (pid, task, got)
            n += 1
        if size == "bf16":
            continue
        assert d["hub"]["local_match"] and p["hf_repo"] == d["hub"]["repo"], pid   # the measured pack is the repo's
        assert p["hf_revision"] == d["hub"]["revision"] and p["disk_gb"] == round(d["hub"]["bytes"] / 1e9, 2), pid
        q = d["model_diff"]["oasst1"]                       # exllamav3's model_diff on chat; the engine's own beside it
        assert p["quality"]["kl_vs_bf16"] == round(q["kl"], 4) and p["quality"]["top1_pct"] == q["top1_pct"], pid
        assert f"top-1 {d['quality']['oasst1']['top1_pct']} %" in p["quality"]["source"], pid
        env = open(os.path.join(ROOT, "ops", "profiles", f"{pid}.env")).read()
        assert f"QSE_WEIGHTS_REVISION={d['hub']['revision']}\n" in env, pid
        assert f"QSE_WEIGHTS_GB={round(d['hub']['bytes'] / 1e9, 2)}\n" in env, pid
    assert n == 7 * 3, n


NEMOTRON_TABLES = ("nemotron-3.5-bolt-74b-a9b", "nemotron-3.5-super-vl-120b-a12b")
# the build each table was measured on
NEMOTRON_BUILDS = {"nemotron-3.5-bolt-74b-a9b": "0.11.1", "nemotron-3.5-super-vl-120b-a12b": "0.11.0"}


def test_the_nemotron_tables_are_the_raw_data():
    for mid in NEMOTRON_TABLES:
        build = NEMOTRON_BUILDS[mid]
        m = registry.load(os.path.join(ROOT, "models", mid, "model.yaml"))
        with open(os.path.join(ROOT, "models", mid, "measurements", f"{build}-summary.json")) as fh:
            raw = json.load(fh)
        meas = m["measurements"]
        assert meas["build"] == raw["build"] == build, mid
        assert meas["source"] == f"models/{mid}/measurements/{build}-summary.json", mid
        rows = {r["profile"]: r for r in meas["rows"]}
        assert sorted(rows) == sorted(raw["profiles"]), (mid, sorted(rows))
        for pid, d in raw["profiles"].items():
            row = rows[pid]
            assert row["build"] == build and row["comparable"] and row["date"] == d["date"], pid
            assert list(row["cells"]) == ["greedy_512"] and row["ttft_s"] == {} and row["lossless"] is None, pid
            assert len(d["runs"]) == 3, pid
            for task in ("prose", "chat", "code"):
                xs = [r["classes"][task] for r in d["runs"]]
                # a greedy text is the same in every run (and, where a plain server ran beside it, the plain one's)
                assert d.get("same_text", {}).get(task, True), (pid, task)
                assert all(x.get("same_text_as_plain", True) and x.get("cache_source", "none") == "none"
                           for x in xs), (pid, task)
                mean = lambda k: sum(x[k] for x in xs) / len(xs)
                want = {"tok_s": round(mean("tok_s"), 2), "min": min(x["tok_s"] for x in xs),
                        "ms_round": round(mean("ms_round"), 1), "tok_round": round(mean("tok_round"), 2), "n": 3}
                assert row["cells"]["greedy_512"][task] == want, (pid, task, row["cells"]["greedy_512"][task], want)
                if "plain_runs" in d:
                    # SPD-127: a drafted profile never writes a class slower than its plain step, beyond the
                    # plain runs' own spread (IQ2's lookup on prose: 27.72 against 27.56 to 27.91 plain)
                    ps = [r["classes"][task]["tok_s"] for r in d["plain_runs"]]
                    assert len(ps) == 3 and want["tok_s"] >= sum(ps) / 3 - (max(ps) - min(ps)), (pid, task, want["tok_s"], ps)
        # the tile's speed is the recommended profile's fastest greedy-512 cell
        rec = registry.recommended_profile(m)["id"]
        best = max(((c["tok_s"], t) for t, c in rows[rec]["cells"]["greedy_512"].items()))
        assert registry.tile(m)["best_speed"] == {"tok_s": best[0], "task": best[1], "profile": rec}, mid



def test_the_ornith_card_is_the_raw_data():
    """OPS-215: Ornith-1.5-122B's card, profile and pin against measurements/0.13.0-summary.json and the raw runs."""
    mid, pid = "ornith-1.5-122b", "ornith-1.5-122b-exl3-4.0"
    m = registry.load(os.path.join(ROOT, "models", mid, "model.yaml"))
    meas_dir = os.path.join(ROOT, "models", mid, "measurements")
    with open(os.path.join(meas_dir, "0.13.0-summary.json")) as fh:
        raw = json.load(fh)
    with open(os.path.join(meas_dir, "0.12.2-drafters.json")) as fh:
        runs = json.load(fh)
    assert m["measurements"]["build"] == raw["build"] == "0.13.0"
    assert m["measurements"]["source"] == f"models/{mid}/measurements/0.13.0-summary.json"
    p = registry.profile(m, pid)
    # the public repository at the card's commit; the weights are the measured commit's
    assert p["hf_revision"] == raw["pack"]["revision"] and len(p["hf_revision"]) == 40 and not p["hf_private"]
    assert p["disk_gb"] == raw["pack"]["disk_gb"]
    env = open(os.path.join(ROOT, "ops", "profiles", f"{pid}.env")).read()
    assert f"QSE_WEIGHTS_REVISION={p['hf_revision']}\n" in env and f"QSE_WEIGHTS_GB={p['disk_gb']}\n" in env
    q = raw["quality_vs_bf16"]
    assert p["quality"]["kl_vs_bf16"] == round(q["kl"], 4) and p["quality"]["top1_pct"] == q["top1_pct"]
    assert p["quality"]["ppl"] == round(q["engine_ppl"], 3)
    # exllamav3's forward on the same pack: not exact, and the summary says so
    x = raw["vs_exllamav3_forward"]
    assert x["exact_agreement"] is False and x["top1_agree"] < x["positions"]
    (row,) = m["measurements"]["rows"]
    assert row["profile"] == pid and row["build"] == "0.13.0" and row["comparable"]
    for task, c in row["cells"]["greedy_512"].items():
        r = raw["rows"]["profile"][task]
        assert r == runs["rows"]["e"][task], task              # the summary copies the raw run
        assert c == {"tok_s": r["tok_s"], "min": r["tok_s"], "ms_round": round(1000 * r["per_block"] / r["tok_s"], 2),
                     "tok_round": r["per_block"], "n": 1}, (task, c)
        assert raw["same_text_as_plain"][task], task
        assert c["tok_s"] > raw["rows"]["plain"][task]["tok_s"], task
    assert raw["rows"]["plain"]["long_code_2048"] == runs["long"]["plain"]
    assert raw["rows"]["profile"]["long_code_2048"] == runs["long"]["e"]
    assert raw["rows"]["dflash2"]["long_code_2048"] == runs["long"]["c"]

def test_every_knob_is_registered():
    m = registry.load(SEED)
    assert registry.unregistered_keys(m) == [], registry.unregistered_keys(m)
    keys = registry.env_keys(m)
    for k in ("QSE_PARALLEL", "MAX_LEN", "QSE_KV_POOL", "QWEN38_NARRATOR_MODEL", "VISION",
              "RESPONSE_CACHE", "SAMPLED_TREE", "QSE_SIDE_GRAPHS"):
        assert k in keys, k
    bad = copy.deepcopy(m)
    bad["components"][1]["knobs"] = {"QWEN38_NOT_A_KNOB": {"on": "1", "off": ""},
                                     "QSE_NOT_READ": {"on": "1", "off": ""},
                                     "NOT_IN_SERVE_ENV": {"on": "1", "off": ""}}
    got = registry.unregistered_keys(bad)
    assert len(got) == 3 and all(k in " ".join(got) for k in
                                 ("QWEN38_NOT_A_KNOB", "QSE_NOT_READ", "NOT_IN_SERVE_ENV")), got


def _profile_env(name: str) -> dict:
    """What ops/profiles/<name>.env sets for the registry's fields, through bash (the _exl3 and
    _flashnext templates included)."""
    out = subprocess.run(["bash", "-c", 'REPO="$1"; HOME=/h; set -a; . "$1/ops/profiles/$2.env"; '
                          'echo "$QSE_WEIGHTS_HF|$QSE_WEIGHTS_GB|$QSE_WEIGHTS_REVISION|$QWEN38_FN_TABLE"',
                          "-", ROOT, name], capture_output=True, text=True, check=True).stdout.strip()
    hf, gb, rev, table = out.split("|")
    return {"hf": hf, "gb": float(gb), "revision": rev or None, "table": table or None}


def test_profiles_match_ops():
    models, errors = registry.load_all()
    assert errors == [], errors
    assert registry.unlisted_profiles(models) == []
    for m in models.values():
        for p in m["profiles"]:
            f = os.path.join(ROOT, "ops", "profiles", p["id"] + ".env")
            assert os.path.isfile(f), f
            env = _profile_env(p["id"])
            assert env["hf"] == p["hf_repo"], (p["id"], env)
            assert env["gb"] == p["disk_gb"], (p["id"], env)
            # OPS-97: a pack on a branch of its repo, its n-gram table's policy (none for the 27B)
            assert env["revision"] == p.get("hf_revision"), (p["id"], env)
            assert env["table"] == (p.get("table") or {}).get("policy"), (p["id"], env)
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(os.path.join(ROOT, "ops", "profiles"), os.path.join(d, "ops", "profiles"))
        with open(os.path.join(d, "ops", "profiles", "exl3-9.9.env"), "w") as fh:
            fh.write("QSE_EXL3_BPW=9.9\n")
        assert registry.unlisted_profiles(models, d) == ["exl3-9.9"]


def _refused(data: dict, want: str) -> None:
    try:
        registry.parse(data, "x.yaml")
    except registry.RegistryError as exc:
        text = "\n".join(exc.problems)
        assert want in text, (want, text)
        return
    raise AssertionError(f"accepted a file that should fail with {want!r}")


def test_every_pack_names_its_maker():
    """VIS-175: our packs are 0xBakeer's, NVIDIA's and LiquidAI's releases are the vendor's, and
    turboderp's Flash-Next packs are a community author's; the weights' repository is hf_repo except
    Bolt's IQ2, whose weights are NVIDIA's GGUF repository"""
    kinds = {"0xBakeer": ("ours", "0xBakeer"), "nvidia": ("vendor", "NVIDIA"), "LiquidAI": ("vendor", "LiquidAI"),
             "turboderp": ("community", "turboderp")}
    seen = set()
    for m in registry.load_all()[0].values():
        for p in m["profiles"]:
            mb = p["made_by"]
            assert (mb["kind"], mb["name"]) == kinds[p["hf_repo"].split("/")[0]], (p["id"], mb)
            assert mb.get("github") == ({"ours": "0xBakeer", "community": mb["name"]}.get(mb["kind"])), (p["id"], mb)
            seen.add(mb["kind"])
            if p["id"] == "nemotron-bolt-iq2":
                assert mb["repo"] == "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF"
            else:
                assert mb.get("repo") is None, p["id"]
    assert seen == {"ours", "vendor", "community"}


def test_the_validator_rules():
    d = _seed()
    registry.parse(copy.deepcopy(d), "seed")                       # the seed itself passes
    x = copy.deepcopy(d)
    x["schema"] = 2
    _refused(x, "reads schema 1 only")
    x = copy.deepcopy(d)
    x["profiles"][0]["recommended"] = True
    _refused(x, "exactly one recommended profile, not 2")
    x = copy.deepcopy(d)
    x["components"][3]["depends_on"] = ["nope"]
    _refused(x, "$.components[3]: no other component 'nope'")
    x = copy.deepcopy(d)
    x["components"][2]["depends_on"] = ["narrator"]                # conductor <-> narrator
    _refused(x, "depends on itself")
    x = copy.deepcopy(d)
    x["components"][2]["default"] = False                          # conductor off, narrator on
    _refused(x, "on by default, but 'conductor' it depends on is not")
    x = copy.deepcopy(d)
    x["components"][1]["conflicts"] = ["vision"]
    _refused(x, "a conflict")
    x = copy.deepcopy(d)
    x["components"][6]["knobs"] = {"VISION": {True: "on", False: "off"}}
    _refused(x, "quote on and off")
    x = copy.deepcopy(d)
    x["components"][3]["assets"] = ["nope"]
    _refused(x, "no asset 'nope'")
    x = copy.deepcopy(d)
    x["profiles"][0]["assets_required"] = ["nope"]
    _refused(x, "$.profiles[0].assets_required: no asset 'nope'")
    x = copy.deepcopy(d)
    x["components"][0]["docs"] = "docs/not-there.md"
    _refused(x, "no file 'docs/not-there.md'")
    x = copy.deepcopy(d)
    x["components"][0]["docs"] = "../../etc/passwd"
    _refused(x, "no file '../../etc/passwd'")
    x = copy.deepcopy(d)
    x["methods"][2]["default"] = 10
    _refused(x, "$.methods[2].default: at least 1024")
    x = copy.deepcopy(d)
    x["measurements"]["rows"][-1]["comparable"] = True               # the 0.4.8 NVFP4 row
    _refused(x, "build 0.4.8 is not the table's 0.6.2")
    x = copy.deepcopy(d)
    del x["measurements"]["rows"][-1]["note"]
    _refused(x, "a row that is not comparable says why")
    x = copy.deepcopy(d)
    x["profiles"][0]["quality"]["date"] = ""
    _refused(x, "needs its date and source")
    x = copy.deepcopy(d)
    x["profiles"][0]["speed"] = 1
    _refused(x, "unknown key 'speed'")
    # VIS-175: every profile says who made it; a community pack names its author's GitHub login
    x = copy.deepcopy(d)
    del x["profiles"][0]["made_by"]
    _refused(x, "made_by")
    x = copy.deepcopy(d)
    x["profiles"][0]["made_by"] = {"kind": "community", "name": "someone"}
    _refused(x, "$.profiles[0].made_by.github: a community pack names its author's GitHub login")
    x = copy.deepcopy(d)
    x["profiles"][0]["made_by"]["kind"] = "friend"
    _refused(x, "made_by.kind")
    x = copy.deepcopy(d)
    x["profiles"][0]["made_by"]["repo"] = "not a repo"
    _refused(x, "made_by.repo")
    x = copy.deepcopy(d)
    x["parallelism"]["slots"]["default"] = 9
    _refused(x, "$.parallelism.slots: the default is outside min..max")
    x = copy.deepcopy(d)
    x["parallelism"]["fixed"]["QSE_X"] = "a b"
    _refused(x, "$.parallelism.fixed.QSE_X")


def test_more_models():
    with tempfile.TemporaryDirectory() as root:
        md = os.path.join(root, "models")
        shutil.copytree(os.path.join(ROOT, "docs"), os.path.join(root, "docs"))
        os.makedirs(os.path.join(md, "qwen3.8-27b"))
        shutil.copy(SEED, os.path.join(md, "qwen3.8-27b", "model.yaml"))
        d = _seed()
        d["id"], d["name"] = "other-9b", "Other-9B"
        os.makedirs(os.path.join(md, "other-9b"))
        with open(os.path.join(md, "other-9b", "model.yaml"), "w") as fh:
            yaml.safe_dump(d, fh)
        os.makedirs(os.path.join(md, "only-readme"))
        with open(os.path.join(md, "only-readme", "README.md"), "w") as fh:
            fh.write("no model.yaml here\n")
        models, errors = registry.load_all(md)
        assert sorted(models) == ["other-9b", "qwen3.8-27b"] and errors == [], (models, errors)
        d["profiles"][0]["recommended"] = True
        os.makedirs(os.path.join(md, "broken"))
        d["id"] = "broken"
        with open(os.path.join(md, "broken", "model.yaml"), "w") as fh:
            yaml.safe_dump(d, fh)
        os.makedirs(os.path.join(md, "misnamed"))
        d2 = _seed()
        with open(os.path.join(md, "misnamed", "model.yaml"), "w") as fh:
            yaml.safe_dump(d2, fh)
        models, errors = registry.load_all(md)
        assert sorted(models) == ["other-9b", "qwen3.8-27b"], sorted(models)
        paths = sorted(e["path"] for e in errors)
        assert paths == ["models/broken/model.yaml", "models/misnamed/model.yaml"], errors
        assert "exactly one recommended" in errors[0]["message"] or \
            "exactly one recommended" in errors[1]["message"]
        assert any("not named after the id" in e["message"] for e in errors)


def test_the_schema_file():
    with open(os.path.join(ROOT, "models", "schema", "model.schema.json")) as fh:
        assert json.load(fh) == json.loads(json.dumps(registry.json_schema())), \
            "regenerate: python -m server.setup.registry --schema > models/schema/model.schema.json"


def test_the_cli():
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert registry.main(["--check"]) == 0
    assert "ok" in buf.getvalue() and "7 profiles" in buf.getvalue(), buf.getvalue()
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "qwen3.8-27b"))
        bad = _seed()
        bad["profiles"][1]["recommended"] = True
        path = os.path.join(d, "qwen3.8-27b", "model.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(bad, fh)
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert registry.main(["--check", path]) == 1
        assert path in buf.getvalue() and "$.profiles" in buf.getvalue(), buf.getvalue()


def test_wizard_fields():
    m = registry.load(SEED)
    steps = registry.wizard_fields(m)
    assert [s["id"] for s in steps] == ["models", "overview", "weights", "components",
                                        "parallelism", "methods", "review", "install"]
    assert [s["index"] for s in steps] == list(range(8))
    assert steps[2]["fields"][0]["options"] == [p["id"] for p in m["profiles"]]
    assert {f["field"] for f in steps[3]["fields"]} == {c["id"] for c in m["components"]}
    assert registry.step_field(m, "parallelism", "slots") == \
        {"field": "slots", "label": "Parallel requests", "type": "int", "min": 1, "max": 16}
    assert registry.step_field(m, 5, "reasoning_effort")["options"] == ["low", "medium", "xhigh"]
    assert registry.step_field(m, "methods", "nope") is None
    assert registry.step_field(m, "review", "slots") is None


def test_engine_max_slots():
    """The guide's cap on parallel chats is the engine's: a joint pass of 32 rows, 2 a chat at least."""
    from server import slots
    meet = slots.Meeting()
    assert registry.ENGINE_MAX_SLOTS == meet.max_rows // 2 == 16
    n = registry.ENGINE_MAX_SLOTS
    assert slots.share(n, meet.max_rows) * n <= meet.max_rows
    assert slots.share(n + 1, meet.max_rows) * (n + 1) > meet.max_rows


def test_field_values():
    m = registry.load(SEED)
    _, name = registry.fields(m)["served_model"]
    for bad in ("a b", "x\ny", "$(reboot)", "a'b", 'a"b', "`id`", "", "a" * 65, "x;y", None, 3):
        assert registry.field_error(name, bad), bad
    assert registry.field_error(name, "my-model.v2:latest") is None
    _, port = registry.fields(m)["port"]
    assert registry.field_error(port, 8000) is None
    assert registry.field_error(port, 80) and registry.field_error(port, 8000.5)
    assert registry.field_error(port, True) and registry.field_error(port, "8000")
    assert registry.field_error(port, float("nan"))
    _, eff = registry.fields(m)["reasoning_effort"]
    assert registry.field_error(eff, "medium") is None and registry.field_error(eff, "max")
    plain = {"id": "x", "type": "string"}
    assert registry.field_error(plain, "ok-1.2") is None and registry.field_error(plain, "/etc")


def test_kv_units():
    from server.slots import slot_bytes
    m = registry.load(SEED)
    per = m["parallelism"]["kv_bytes_per_token"]
    assert per == slot_bytes(1, constant=0) == 64 * 1024 + 2 * 20 * 1024 + 256
    assert per / 1024 == 104.25
    w = 262_144
    assert round(per * w / 2**30, 2) == 26.06 and round(per * w / 1e9, 2) == 27.98



def test_the_mock_copy():
    """The page's e2e and unit tests run on the mock's copy of the registry entry; a model.yaml
    change that does not reach it would test the page against a model the server does not serve.
    Write it again with: python tests/test_setup_registry.py --write-mock"""
    for mid in MOCKED:
        with open(_mock_copy(mid)) as fh:
            assert json.load(fh) == _served(registry.get(mid)), (
                f"{_mock_copy(mid)} is not the registry's payload; run python tests/test_setup_registry.py --write-mock")


# the 27B and, for the models step's second tile and a pack's largest window, Flash-Next (OPS-97); the
# NemotronH tables for the measured table's unit tests (not served by the mock)
MOCKED = ("qwen3.8-27b", "qwen3.8-flash-next", "lfm2.5-24b-a2b") + NEMOTRON_TABLES


def _mock_copy(mid: str) -> str:
    return os.path.join(ROOT, "dashboard", "mock", "setup", mid + ".json")


# VIS-175: Flash-Next's community list is the release's own (models/community-builtin.json), so the mock
# keeps the API's list as it is
BUILTIN_MOCKED = ("qwen3.8-flash-next",)


def _builtin_copy(mid: str) -> str:
    return os.path.join(ROOT, "dashboard", "mock", "setup", f"community-{mid}.json")


def _builtin_served(mid: str) -> dict:
    return json.loads(json.dumps({"community": registry.get(mid)["community"]}))


def test_the_builtin_community_mock():
    """dashboard/mock/setup/community-qwen3.8-flash-next.json is the API's community list of Flash-Next
    (turboderp's packs). Write it again with: python tests/test_setup_registry.py --write-mock"""
    for mid in BUILTIN_MOCKED:
        with open(_builtin_copy(mid)) as fh:
            assert json.load(fh) == _builtin_served(mid), (
                f"{_builtin_copy(mid)} is not the registry's list; run python tests/test_setup_registry.py --write-mock")


def _served(m: dict) -> dict:
    """GET /v1/setup/models/{id} (server/setup_api.py), as JSON, without the community packs: the
    dashboard's mock keeps those in mock/setup/community-<id>.json (tests/test_community_weights.py
    checks the API's own list)."""
    m = {k: v for k, v in dict(m, community=[]).items() if k != "community"}
    return json.loads(json.dumps(dict(m, contract_version="1.0",
                                      wizard_fields=registry.wizard_fields(m))))


if __name__ == "__main__":
    if "--write-mock" in sys.argv:
        for mid in MOCKED:
            with open(_mock_copy(mid), "w") as fh:
                json.dump(_served(registry.get(mid)), fh, indent=1, ensure_ascii=False)
                fh.write("\n")
            print(f"wrote {_mock_copy(mid)}")
        for mid in BUILTIN_MOCKED:
            with open(_builtin_copy(mid), "w") as fh:
                json.dump(_builtin_served(mid), fh, indent=1, ensure_ascii=False)
                fh.write("\n")
            print(f"wrote {_builtin_copy(mid)}")
        sys.exit(0)
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
