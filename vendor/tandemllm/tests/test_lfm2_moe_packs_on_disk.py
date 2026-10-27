"""ENG-286 (QA of ENG-285): the real LFM2.5-24B-A2B weights on this machine, read from their headers
alone (CPU, nothing loaded on a device). Every profile of models/lfm2.5-24b-a2b/model.yaml whose folder
is where ops/profiles/_lfm25.env reads it (`$LFM25_MODELS`, else `~/models`, then
lfm2.5-24b-a2b-<variant>) is checked; a machine without them skips and says so.

  1  the BF16 checkpoint is read: the audit finds every tensor the layout expects, the formats are
     bf16 (the head tied), and the byte budget of a decode token is printed
                                                        test_the_bf16_checkpoint_is_read
  2  each EXL3 pack is audited: every trellis is the mul1 codebook, the K histogram lists the
     experts' three projections, the attention and conv projections, the dense MLPs and the head,
     and each tensor's K is a rate the kernels decode (1.5..8)   test_each_exl3_pack_is_audited
  3  the card is the files: each profile's disk_gb is its safetensors bytes, and its weights_gib
     (measured at load) is within 5 % of what the index says the engine holds
                                                        test_the_card_is_the_files

The refusal of a tensor the kernels cannot decode is tests/test_lfm2_moe_pack.py (synthetic packs).
Run: python tests/test_lfm2_moe_packs_on_disk.py
"""

from __future__ import annotations

import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.models.lfm2_moe.pack import PackIndex, summary  # noqa: E402

CARD = yaml.safe_load(open(os.path.join(ROOT, "models", "lfm2.5-24b-a2b", "model.yaml")))
BASE = os.environ.get("LFM25_MODELS") or os.path.expanduser("~/models")
RATES = {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0}
_IDX: dict = {}


def _dir(pid: str) -> str:
    return os.path.join(BASE, "lfm2.5-24b-a2b-" + pid.removeprefix("lfm25-"))


def _present() -> list[dict]:
    return [p for p in CARD["profiles"] if os.path.isfile(os.path.join(_dir(p["id"]), "config.json"))]


def _idx(pid: str) -> PackIndex:
    if pid not in _IDX:
        _IDX[pid] = PackIndex.open(_dir(pid))
    return _IDX[pid]


def _gb(d: str) -> float:
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors")) / 1e9


def test_the_bf16_checkpoint_is_read():
    if not any(p["id"] == "lfm25-bf16" for p in _present()):
        return f"skipped: no BF16 checkpoint under {BASE}"
    idx = _idx("lfm25-bf16")
    idx.audit()
    assert not idx.is_exl3 and set(idx.formats.values()) <= {"bf16", "tied"}, idx.formats
    db = idx.decode_bytes(context=8192)
    assert db["total"] == sum(db["parts"].values()) and db["parts"]["experts"] > 0
    assert 4.0e9 < idx.decode_bytes()["total"] < 6.0e9, idx.decode_bytes()
    return (f"{len(idx.tensors)} tensors; a decode token reads {idx.decode_bytes()['total'] / 1e9:.2f} GB "
            f"({', '.join(f'{k} {v / 1e9:.2f}' for k, v in sorted(db['parts'].items(), key=lambda x: -x[1])[:5])})")


def test_each_exl3_pack_is_audited():
    packs = [p for p in _present() if p["format"] == "exl3"]
    if not packs:
        return f"skipped: no EXL3 pack under {BASE}"
    seen = []
    for p in packs:
        idx = _idx(p["id"])
        idx.audit()
        idx.check_mul1()
        assert idx.is_exl3, p["id"]
        h = idx.k_histogram()
        groups = set(h)
        assert {"experts.w1", "experts.w2", "experts.w3", "head"} <= groups, (p["id"], groups)
        assert any("self_attn" in g for g in groups) and any("conv" in g for g in groups), (p["id"], groups)
        assert any("feed_forward" in g and not g.startswith("experts") for g in groups), (p["id"], groups)
        ks = {k for c in h.values() for k in c}
        assert ks <= RATES, (p["id"], ks)
        s = summary(idx)
        assert s["decode_bytes"]["total"] > 0 and s["resident"]["total"] > 0
        seen.append(f"{p['id'].removeprefix('lfm25-exl3-')}: K {sorted(ks)}, "
                    f"{s['decode_bytes']['total'] / 1e6:,.0f} MB a token")
    return "; ".join(seen)


def test_the_card_is_the_files():
    present = _present()
    if not present:
        return f"skipped: no LFM2.5 weights under {BASE}"
    rows = []
    for p in present:
        d = _dir(p["id"])
        gb = _gb(d)
        assert abs(gb - p["disk_gb"]) <= 0.05, (p["id"], round(gb, 2), p["disk_gb"])
        held = _idx(p["id"]).resident_bytes()["total"] / 2**30
        assert abs(held - p["weights_gib"]) <= 0.05 * p["weights_gib"], (p["id"], round(held, 2), p["weights_gib"])
        rows.append(f"{p['id'].removeprefix('lfm25-')} {gb:.2f} GB / {p['weights_gib']} GiB (index {held:.2f})")
    return "; ".join(rows)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except (AssertionError, ValueError) as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
