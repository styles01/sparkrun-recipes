"""SPD-127: NemotronH's StairCut price files. `nh_check.py prices --lilicorr` writes the `spec` prior from an
offline LiLiCorr run, and every NemotronH profile that drafts with LiLiCorr ships a price file with the
one-row chain and the plain step: without one the one-row block is priced at 1 ms for good (the anchor's
rounds price `spec`, never the chain), so the cut takes the anchor alone after a paid draft and a drafted
profile runs below its plain step (Bolt EXL3 3.0 to 4.0 on 0.11.0: prose 31 to 33 tok/s against 40.5 plain).
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.methods.staircut import PriceModel, StairCut  # noqa: E402
from tools import nh_check  # noqa: E402

STAIR = {"rows": {"1024": {str(r): {"ms": 20.0 + 1.5 * r} for r in range(1, 17)},
                  "8192": {str(r): {"ms": 21.0 + 1.5 * r} for r in range(1, 17)}}}
BENCH = {"graph": {"ms_token": 24.5}}
LILICORR = {"draft_ms_median": 10.0, "accepted_length_mean": 4.0}


def _profile(path: str) -> dict:
    env = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line and not line.startswith("."):
            k, v = line.split("=", 1)
            env[k] = v
    return env


def test_spec_prior_is_the_8_row_chain_and_the_draft_over_three_quarters_of_the_accepted_length():
    # (8-row chain at the lowest context 32.0 + draft 10.0) / (0.75 x 4.0) = 14.0
    assert nh_check.spec_prior(STAIR, LILICORR) == 14.0


def test_prices_writes_the_spec_prior_into_every_class():
    with tempfile.TemporaryDirectory() as d:
        files = {}
        for name, doc in (("stair", STAIR), ("bench", BENCH), ("lilicorr", LILICORR)):
            files[name] = os.path.join(d, f"{name}.json")
            json.dump(doc, open(files[name], "w"))
        out = os.path.join(d, "prices", "p.dgx-spark.json")
        a = SimpleNamespace(stair=files["stair"], bench=files["bench"], out=out, max_len=262144,
                            lilicorr=files["lilicorr"])
        nh_check.cmd_prices(a)
        doc = json.load(open(out))
        classes = [k for k in doc if k.isdigit()]
        assert len(classes) == 2 and "0" in classes, classes
        for c in classes:
            assert doc[c]["spec"] == {"1": 14.0} and doc[c]["plain"] == {"1": 24.5}, doc[c]
        assert "Spec prior 14.0 ms a token" in doc["measured"], doc["measured"]
        # the file loads, and the cut reads its priors
        cut = StairCut(PriceModel.load(out, log=lambda m: None))
        assert cut.spec_ms(0) == 14.0 and cut.plain_ms(0) == 24.5
        a.lilicorr = None
        nh_check.cmd_prices(a)
        doc = json.load(open(out))
        assert all("spec" not in doc[c] for c in classes) and "No spec or prime prior" in doc["measured"]


def test_a_missing_file_prices_the_one_row_block_at_1_ms():
    # the failure the files prevent: learned 2..8-row prices and no one-row price make the anchor alone the
    # best block wherever the head is less than sure (prose and chat); with the file's one-row price it drafts
    pm = PriceModel()
    for r in range(2, 9):
        pm.observe(r, 30.0 + r)
    cut = StairCut(pm)
    cut.draft[0] = 10.0
    tok = [(1, 2)] * 7
    prob = [(0.6, 0.05)] * 7
    toks, _ = cut.cut(0, tok, prob)
    assert toks == [0], toks
    priced = StairCut(PriceModel({0: {"chain": {str(r): 22.0 + 1.5 * r for r in range(1, 17)}}}))
    priced.draft[0] = 10.0
    toks, _ = priced.cut(0, tok, prob)
    assert len(toks) > 1, toks


def test_every_lilicorr_profile_ships_its_price_file():
    seen = []
    for path in sorted(glob.glob(os.path.join(ROOT, "ops", "profiles", "nemotron-*.env"))):
        env = _profile(path)
        if env.get("QSE_NH_HEAD") != "lilicorr":
            continue
        name = os.path.basename(path)[:-4]
        pf = os.path.join(ROOT, "ops", "prices", env["QSE_NH_MODEL"], f"{name}.dgx-spark.json")
        assert os.path.isfile(pf), f"{name}: no price file {pf}"
        doc = json.load(open(pf))
        for c in (k for k in doc if k.lstrip("-").isdigit()):
            assert set(doc[c]["chain"]) >= {str(r) for r in range(1, 17)}, f"{name}[{c}] chain"
            assert float(doc[c]["plain"]["1"]) > 0.0, f"{name}[{c}] plain"
        seen.append(name)
    assert len(seen) >= 10, seen
    return f"{len(seen)} profiles"


def test_every_nemotron_community_pack_ships_its_price_file():
    # ENG-377: a community pack on a NemotronH base drafts with LiLiCorr like its nearest profile; its price file
    # is ops/prices/<base>/community/<id>.dgx-spark.json (serving.py: the profile's name is community/<id>)
    from server.setup import registry
    seen = []
    for p in sorted(glob.glob(os.path.join(ROOT, "models", "*", "community", "*.json"))):
        e = json.load(open(p))
        if registry.community_engine(e) != "nemotron_h":
            continue
        pf = os.path.join(ROOT, "ops", "prices", e["base_model"], "community", f"{e['id']}.dgx-spark.json")
        assert os.path.isfile(pf), f"{e['id']}: no price file {pf}"
        doc = json.load(open(pf))
        for c in (k for k in doc if k.lstrip("-").isdigit()):
            assert set(doc[c]["chain"]) >= {str(r) for r in range(1, 17)}, f"{e['id']}[{c}] chain"
            assert float(doc[c]["plain"]["1"]) > 0.0 and "spec" in doc[c], f"{e['id']}[{c}]"
        seen.append(e["id"])
    assert seen, "no NemotronH community pack"
    return f"{len(seen)} packs"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<40} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<40} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
