"""`tools/drop_page_cache.py`: which files it advises, and that it survives the odd one.

Run: python tests/test_drop_page_cache.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.drop_page_cache import cached_gb, default_paths, drop  # noqa: E402


def _tree() -> str:
    d = tempfile.mkdtemp(prefix="dropcache-")
    os.makedirs(os.path.join(d, "hub/blobs"))
    os.makedirs(os.path.join(d, "hub/snapshots/abc"))
    open(os.path.join(d, "hub/blobs/1111"), "wb").write(b"x" * 1000)
    # an HF snapshot is symlinks into blobs: the blob is advised once, by its real path
    os.symlink("../../blobs/1111", os.path.join(d, "hub/snapshots/abc/model.safetensors"))
    open(os.path.join(d, "one.safetensors"), "wb").write(b"y" * 500)
    return d


def test_every_file_once_through_the_symlinks():
    d = _tree()
    seen = []
    r = drop([os.path.join(d, "hub"), os.path.join(d, "one.safetensors"), os.path.join(d, "missing")],
             advise=lambda fd: seen.append(os.readlink(f"/proc/self/fd/{fd}") if os.path.exists("/proc/self/fd") else fd))
    assert r["files"] == 2 and r["failed"] == 0, r
    assert abs(r["gb"] - 1500 / 1e9) < 1e-12, r
    assert len(seen) == 2, seen


def test_a_failing_advice_is_counted_not_raised():
    d = _tree()

    def bad(fd):
        raise OSError("EINVAL")
    r = drop([d], advise=bad)
    assert r["files"] == 0 and r["failed"] == 2, r


def test_the_defaults_leave_the_suffix_corpus_alone():
    names = [str(p) for p in default_paths()]
    assert any("Qwen3.8-27B-FP8" in n for n in names) and any(n.endswith("nvfp4") for n in names)
    assert not any("corpus" in n or "suffix" in n for n in names), names


def test_the_exl3_weights_join_the_defaults_when_set():
    keys = ("QWEN38_EXL3", "QWEN38_DRAFT_EXL3", "QWEN38_DRAFT_HEAD_EXL3")
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        base = [str(p) for p in default_paths()]
        assert len(base) == 4, base
        os.environ.update({"QWEN38_EXL3": "/w/exl3-quants/4.5", "QWEN38_DRAFT_EXL3": "/w/k3",
                           "QWEN38_DRAFT_HEAD_EXL3": "/w/head-k4"})
        names = [str(p) for p in default_paths()]
        assert names[:len(base)] == base and names[len(base):] == ["/w/exl3-quants/4.5", "/w/k3", "/w/head-k4"]
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_a_profile_adds_its_pack_and_keeps_its_warm_table():
    """OPS-99: QSE_PAGE_CACHE_DROP joins the defaults; a file QSE_PAGE_CACHE_KEEP names (a Flash-Next
    pack's warm n-gram table) is never advised."""
    keys = ("QSE_PAGE_CACHE_DROP", "QSE_PAGE_CACHE_KEEP")
    old = {k: os.environ.get(k) for k in keys}
    d = tempfile.mkdtemp(prefix="dropcache-")
    for n in ("model-00001-of-00002.safetensors", "ngram_embedding.safetensors", "config.json"):
        open(os.path.join(d, n), "wb").write(b"z" * 100)
    try:
        for k in keys:
            os.environ.pop(k, None)
        base = [str(p) for p in default_paths()]
        os.environ["QSE_PAGE_CACHE_DROP"] = d
        assert [str(p) for p in default_paths()] == base + [d]
        assert drop([d], advise=lambda fd: None)["files"] == 3
        os.environ["QSE_PAGE_CACHE_KEEP"] = "ngram_embedding.safetensors"
        seen = []
        r = drop([d], advise=lambda fd: seen.append(fd))
        assert r["files"] == 2 and len(seen) == 2, r
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_cached_is_read_in_gb():
    assert abs(cached_gb("MemTotal: 1 kB\nCached:  1048576 kB\n") - 1048576 * 1024 / 1e9) < 1e-9


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
