"""The EXL3 profiles' plain base (OPS-133, models/qwen3.8-27b/model.yaml `base-plain`,
ops/fetch_weights.py `fetch_plain`): the Qwen3.8-27B-FP8 tensors an EXL3 load reads, fetched by range
requests, on a CPU with no network (a local range server stands for the hub) and no model load.

The test matrix, the Gherkin of OPS-133:

  1  the plain base holds every tensor the EXL3 load reads: the real FP8 checkpoint's 1,584 tensor
     names (tests/fixtures/qwen38-27b-fp8-headers.json.gz) as tiny tensors, the plain base fetched
     from them, and engine/loader.py `Weights` with the six packs' quantised set reads the same 450
     tensors byte for byte from both; the vision tower (engine/vision.py) is all there. Every
     layers-N file holds some (its norms), so no whole file can be left out; what goes is the
     F8_E4M3 codes with their scales and lm_head.weight: 26.87 of the 30.39 GB
                                                       test_every_tensor_the_exl3_load_reads_is_kept
  2  fresh EXL3 install: the plan downloads 3.54 GB of base, not 30.4 (exl3-6.0: 36.48 GB with its
     drafters, was 63.34); no F8_E4M3 projection file is fetched whole
                                                       test_fresh_exl3_install
  3  a box that has the full FP8 base: nothing of the base is downloaded, and the EXL3 profile
     reads that snapshot (QWEN38_MODEL unset); without it, the plain base
                                                       test_box_that_has_the_full_fp8_base
  4  the NVFP4 profile still gets the full base: a box with the plain base only lists the FP8
     checkpoint (30.4 GB) as a download; a full base whose download did not finish does not serve
     for the plain one                                 test_nvfp4_profile_still_gets_the_full_base,
                                                       test_an_unfinished_full_base_does_not_serve
  5  the fetch itself: valid safetensors files of the kept tensors, byte for byte; the index's
     weight map cut to them; a stopped fetch resumes without fetching the done files again; a header
     that does not add up to the hub's size, and a server that ignores Range, fail the fetch
                                                       test_the_plain_fetch

Run: python tests/test_base_plain.py
"""

from __future__ import annotations

import gzip
import http.server
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_setup_apply as T  # noqa: E402
from server.setup import apply, registry  # noqa: E402

M = T.M
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                       "qwen38-27b-fp8-headers.json.gz")


def _fw():
    spec = importlib.util.spec_from_file_location("fw_plain", os.path.join(T.REPO, "ops",
                                                                           "fetch_weights.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry():
    return registry.asset(M, "base-plain")["fetch"][0]


def _tiny(dtype: str, shape: list) -> bytes:
    """the bytes of a tensor of `shape`'s rank with every dimension 1 or 2"""
    n = 1
    for d in shape:
        n *= min(2, d)
    return os.urandom(n * (1 if dtype == "F8_E4M3" else 2))


def test_every_tensor_the_exl3_load_reads_is_kept():
    import contextlib
    import io
    import huggingface_hub
    import torch
    from engine import loader
    from engine.vision import VISUAL_PREFIXES
    fw = _fw()
    doc = json.loads(gzip.decompress(open(FIXTURE, "rb").read()))
    e = _entry()
    # the sizes, from the real headers
    kept_b = total_b = f8_b = 0
    for name, f in doc["files"].items():
        header = {k: {"dtype": d, "shape": shape, "data_offsets": [a, b]}
                  for k, (d, shape, a, b) in f["tensors"].items()}
        kept = fw.kept_tensors(header, e["skip_dtypes"], e["skip_tensors"])
        assert kept, f"{name}: nothing kept, the file could be left out"
        kept_b += sum(b - a for _d, _s, a, b in (f["tensors"][k] for k in kept))
        total_b += f["size"]
        f8_b += sum(b - a for d, _s, a, b in f["tensors"].values() if d == "F8_E4M3")
    assert len(doc["files"]) == 65 and abs(total_b / 1e9 - 30.390) < 0.001, total_b
    assert abs(kept_b / 1e9 - 3.517) < 0.001 and abs(f8_b / 1e9 - 24.327) < 0.001, (kept_b, f8_b)
    assert abs((total_b - kept_b) / 1e9 - 26.873) < 0.001, total_b - kept_b
    # the registry's size: the kept tensors and the files fetched whole (0.023 GB), rounded up
    assert e["gb"] * 1e9 >= kept_b + 23_113_424 and e["gb"] - (kept_b + 23_113_424) / 1e9 < 0.01
    # the same names with tiny tensors: the full checkpoint, served by a range server; the plain
    # base fetched from it; an EXL3 pack whose index names the six packs' quantised set
    files = {n: _st([(k, d, _tiny(d, shape)) for k, (d, shape, _a, _b) in
                     sorted(f["tensors"].items(), key=lambda kv: kv[1][2])])
             for n, f in doc["files"].items()}
    files["config.json"] = json.dumps({"model_type": "qwen3_5", "tie_word_embeddings": False}).encode()
    files["model.safetensors.index.json"] = json.dumps({"metadata": {"total_size": total_b}, "weight_map": {
        **{k: n for n, f in doc["files"].items() for k in f["tensors"]},
        "mtp.fc.weight": "mtp.safetensors"}}).encode()
    m = {"repo": doc["repo"], "commit": doc["commit"],
         "files": {n: {"size": len(b), "sha256": None} for n, b in files.items()}}
    tmp = tempfile.mkdtemp(prefix="plain-load-")
    full = os.path.join(tmp, "full")
    os.makedirs(full)
    for n, b in files.items():
        open(os.path.join(full, n), "wb").write(b)
    pack = os.path.join(tmp, "pack")
    os.makedirs(pack)
    json.dump({"weight_map": {k: "model.safetensors" for k in doc["exl3_trellis"]}},
              open(os.path.join(pack, "model.safetensors.index.json"), "w"))
    srv = _RangeServer(files)
    fw.hub_url = lambda repo, commit, name: f"{srv.url}/{repo}/resolve/{commit}/{name}"

    def snap(repo_id, revision=None, local_dir=None, allow_patterns=None, **_kw):
        for n in allow_patterns:
            open(os.path.join(local_dir, n), "wb").write(files[n])
        return local_dir
    saved = (huggingface_hub.snapshot_download, loader.Weights.load_exl3)
    huggingface_hub.snapshot_download = snap
    # the pack's own blocks are not this test's: what the base gives is
    loader.Weights.load_exl3 = lambda self, path: None
    try:
        plain = os.path.join(tmp, "base-plain")
        with contextlib.redirect_stdout(io.StringIO()):
            assert fw.fetch_plain("base-plain", m["repo"], plain, e, m) == 0
            a = loader.Weights(full, device="cpu", exl3=pack, nvfp4="", fp8_head="")
            b = loader.Weights(plain, device="cpu", exl3=pack, nvfp4="", fp8_head="")
        assert len(a._skip) == 401 and "lm_head" in a._skip, len(a._skip)
        # engine/loader.py reads the same tensors from both, byte for byte, and no projection
        assert sorted(a.t) == sorted(b.t) and not a.q and not b.q, (len(a.t), len(b.t))
        assert len(a.t) == 450, len(a.t)
        for k in a.t:                   # random bytes: compared as bytes (a NaN is not equal)
            x, y = (w.t[k].contiguous().view(-1).view(torch.uint8) for w in (a, b))
            assert a.t[k].dtype == b.t[k].dtype and torch.equal(x, y), k
        # the vision tower (engine/vision.py reads it by the index): every tensor kept
        idx = json.load(open(os.path.join(plain, "model.safetensors.index.json")))
        vis = {k for f in doc["files"].values() for k in f["tensors"] if k.startswith(VISUAL_PREFIXES)}
        assert len(vis) == 333 and vis <= set(idx["weight_map"]), len(vis)
        assert set(idx["weight_map"]) == {k for f in doc["files"].values() for k, v in
                                          fw.kept_tensors({x: {"dtype": d, "data_offsets": [p, q]}
                                                           for x, (d, _s, p, q) in f["tensors"].items()},
                                                          e["skip_dtypes"], e["skip_tensors"]).items()}
        with open(os.path.join(plain, "outside.safetensors"), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            assert vis <= set(json.loads(fh.read(n))), "a vision tensor is missing"
    finally:
        srv.close()
        huggingface_hub.snapshot_download, loader.Weights.load_exl3 = saved
        shutil.rmtree(tmp, ignore_errors=True)


def _drop_assets(box, ids):
    for aid in ids:
        a = registry.asset(M, aid)
        for p in [a["path"], *a["also"]]:
            p = p.replace("${HOME}", box.home).replace("${REPO}", box.root).replace(
                "${HF_HUB}", os.path.join(box.home, ".cache", "huggingface", "hub")).replace(
                "*", "snap")
            if os.path.isfile(p):
                os.unlink(p)


def test_fresh_exl3_install():
    with T.Box(on_disk=(), assets=False):
        p = T._plan(T._sel(profile="exl3-6.0", components={"narrator": False, "laya": False,
                                                            "shadow": False}))
        ids = [d["id"] for d in p["downloads"]]
        assert "base-plain" in ids and "base-fp8" not in ids, ids
        plain = next(d for d in p["downloads"] if d["id"] == "base-plain")
        assert plain["gb"] == 3.54, plain
        assert p["profile"]["download_gb"] == round(23.3 + 1.3 + 0.64 + 3.85 + 3.85 + 3.54, 2), \
            p["profile"]
        assert p["profile"]["download_gb"] == 36.48
        # every EXL3 profile reads the plain base, NVFP4 the full one
        for prof in M["profiles"]:
            want = "base-plain" if prof["format"] == "exl3" else "base-fp8"
            assert want in prof["assets_required"], (prof["id"], prof["assets_required"])
    e = _entry()
    assert e["skip_dtypes"] == ["F8_E4M3"] and "lm_head.weight" in e["skip_tensors"]


def test_box_that_has_the_full_fp8_base():
    with T.Box() as box:
        _drop_assets(box, ["base-plain"])
        p = T._plan(T._sel(profile="exl3-6.0"))
        assert p["profile"]["on_disk"] is True and p["downloads"] == [], p["downloads"]
        assert registry.asset_present(registry.asset(M, "base-plain"), box.root, M)
        assert not registry.asset_present(registry.asset(M, "base-plain"), box.root)
        got = apply.resolve_env(["QWEN38_MODEL"], profile="exl3-6.0")
        assert "QWEN38_MODEL" not in got, got                   # engine/config.py: the snapshot
        _drop_assets(box, ["base-fp8"])
        got = apply.resolve_env(["QWEN38_MODEL"], profile="exl3-6.0")
        assert got["QWEN38_MODEL"] == os.path.join(box.home, ".qwen38-spark-engine", "weights",
                                                   "base-plain"), got
        assert "QWEN38_MODEL" not in apply.resolve_env(["QWEN38_MODEL"], profile="nvfp4")


def test_an_unfinished_full_base_does_not_serve():
    """fetch_weights.py: the full base serves for the plain one only when its last download
    finished (~/.qwen38-spark-engine/fetch/base-fp8.json `complete`)."""
    import contextlib
    import io
    fw = _fw()
    asked = []
    fw.plan = lambda repo, d, files, *a, **kw: (asked.append(d) or
                                                ({"commit": "c0ffee", "files": {}}, 0))
    with T.Box() as box:
        _drop_assets(box, ["base-plain"])
        os.environ["HF_HUB_CACHE"] = os.path.join(box.home, ".cache", "huggingface", "hub")
        present = registry.asset_present
        registry.asset_present = lambda a, repo=None, m=None: present(a, box.root, m)
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert fw.assets("exl3-6.0", T.REPO, dry_run=True) == 0
            assert "[fetch] base-plain: on this box" in out.getvalue() and not asked, out.getvalue()
            fw._write_state("base-fp8", {"complete": False})
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert fw.assets("exl3-6.0", T.REPO, dry_run=True) == 0
            assert asked == [os.path.join(box.home, ".qwen38-spark-engine", "weights",
                                          "base-plain")], (asked, out.getvalue())
        finally:
            registry.asset_present = present
            os.environ.pop("HF_HUB_CACHE", None)


def test_nvfp4_profile_still_gets_the_full_base():
    with T.Box(on_disk=("exl3-6.0",)) as box:
        _drop_assets(box, ["base-fp8"])
        p = T._plan(T._sel(profile="exl3-6.0"))
        assert p["profile"]["on_disk"] is True, p["downloads"]       # the plain base serves
        p = T._plan(T._sel(profile="nvfp4", slots=1))
        d = {x["id"]: x["gb"] for x in p["downloads"]}
        assert d.get("base-fp8") == 30.4, p["downloads"]


# ---------------------------------------------------------------------- the fetch, on a range server
def _st(tensors: list[tuple[str, str, bytes]], meta=None) -> bytes:
    """A safetensors file of (name, dtype, raw bytes), in that order."""
    hdr, off = {}, 0
    for name, dtype, raw in tensors:
        hdr[name] = {"dtype": dtype, "shape": [len(raw) // (1 if dtype == "F8_E4M3" else 2)],
                     "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    if meta:
        hdr["__metadata__"] = meta
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h + b"".join(r for _n, _d, r in tensors)


class _RangeServer:
    def __init__(self, files: dict, ranges: bool = True):
        self.files, self.calls = files, []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                name = self.path.split("/resolve/", 1)[1].split("/", 1)[1]
                body = outer.files[name]
                rng = self.headers.get("Range")
                outer.calls.append((name, rng))
                if not ranges or not rng:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                a, b = (int(x) for x in rng.split("=", 1)[1].split("-"))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(body)}")
                self.send_header("Content-Length", str(b - a + 1))
                self.end_headers()
                self.wfile.write(body[a:b + 1])

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()


def test_the_plain_fetch():
    import huggingface_hub
    import torch
    fw = _fw()
    fw.RANGE_TRIES = 1
    rnd = os.urandom
    layer = [("model.language_model.layers.0.input_layernorm.weight", "BF16", rnd(64)),
             ("model.language_model.layers.0.linear_attn.A_log", "BF16", rnd(16)),
             ("model.language_model.layers.0.mlp.down_proj.weight_scale_inv", "BF16", rnd(32)),
             ("model.language_model.layers.0.post_attention_layernorm.weight", "BF16", rnd(64)),
             ("model.language_model.layers.0.mlp.down_proj.weight", "F8_E4M3", rnd(4096))]
    outside = [("lm_head.weight", "BF16", rnd(2048)),
               ("model.language_model.embed_tokens.weight", "BF16", rnd(2048)),
               ("model.language_model.norm.weight", "BF16", rnd(64)),
               ("model.visual.blocks.0.attn.proj.weight", "BF16", rnd(256))]
    files = {"layers-0.safetensors": _st(layer, {"format": "pt"}),
             "outside.safetensors": _st(outside),
             "config.json": b'{"model_type": "qwen3_8"}',
             "model.safetensors.index.json": json.dumps({"metadata": {"total_size": 1}, "weight_map": {
                 **{n: "layers-0.safetensors" for n, _d, _r in layer},
                 **{n: "outside.safetensors" for n, _d, _r in outside},
                 "mtp.fc.weight": "mtp.safetensors"}}).encode(),
             "mtp.safetensors": _st([("mtp.fc.weight", "BF16", rnd(64))])}
    m = {"repo": "Qwen/Qwen3.8-27B-FP8", "commit": "c0ffee",
         "files": {n: {"size": len(b), "sha256": None} for n, b in files.items()}}

    def snap(repo_id, revision=None, local_dir=None, allow_patterns=None, **_kw):
        for n in allow_patterns:
            with open(os.path.join(local_dir, n), "wb") as fh:
                fh.write(files[n])
        return local_dir
    saved = huggingface_hub.snapshot_download
    huggingface_hub.snapshot_download = snap
    srv = _RangeServer(files)
    fw.hub_url = lambda repo, commit, name: f"{srv.url}/{repo}/resolve/{commit}/{name}"
    tmp = tempfile.mkdtemp(prefix="plain-")
    try:
        from safetensors import safe_open
        e = dict(_entry())
        to = os.path.join(tmp, "base-plain")
        assert fw.fetch_plain("base-plain", m["repo"], to, e, m) == 0
        assert sorted(os.listdir(to)) == [fw.PLAIN_MANIFEST, "config.json",
                                          "layers-0.safetensors", "model.safetensors.index.json",
                                          "outside.safetensors"], os.listdir(to)
        assert not os.path.exists(to + ".partial-base-plain")
        want = {"layers-0.safetensors": [x for x in layer if x[1] != "F8_E4M3" and
                                         "scale_inv" not in x[0]],
                "outside.safetensors": outside[1:]}
        for name, ts in want.items():
            with safe_open(os.path.join(to, name), framework="pt") as f:
                assert sorted(f.keys()) == sorted(n for n, _d, _r in ts), (name, list(f.keys()))
                for n, _d, raw in ts:
                    t = f.get_tensor(n)
                    assert t.contiguous().view(-1).view(torch.uint8).numpy().tobytes() == raw, n
                if name == "layers-0.safetensors":
                    assert f.metadata() == {"format": "pt"}
        idx = json.load(open(os.path.join(to, "model.safetensors.index.json")))
        assert set(idx["weight_map"]) == {n for ts in want.values() for n, _d, _r in ts}, idx
        # a stopped fetch: the done file is not fetched again (its header only)
        part = to + ".partial-base-plain"
        shutil.copytree(to, part)
        shutil.rmtree(to)
        os.unlink(os.path.join(part, "outside.safetensors"))
        srv.calls.clear()
        assert fw.fetch_plain("base-plain", m["repo"], to, e, m) == 0
        data_calls = [c for c in srv.calls if c[0] == "layers-0.safetensors"]
        assert len(data_calls) == 2, data_calls          # the length and the header
        assert os.path.isfile(os.path.join(to, "outside.safetensors"))
        # a header that does not add up to the hub's size fails
        shutil.rmtree(to)
        bad = json.loads(json.dumps(m))
        bad["files"]["outside.safetensors"]["size"] += 1
        assert fw.fetch_plain("base-plain", m["repo"], to, e, bad) == 1
        assert not os.path.exists(to)
        # a server that answers a range with the whole file fails
        shutil.rmtree(to + ".partial-base-plain")
        srv.close()
        srv = _RangeServer(files, ranges=False)
        assert fw.fetch_plain("base-plain", m["repo"], to, e, m) == 1
    finally:
        srv.close()
        huggingface_hub.snapshot_download = saved
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
