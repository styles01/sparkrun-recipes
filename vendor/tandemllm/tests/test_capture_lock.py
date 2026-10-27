"""The lock between CUDA graph captures and the disk tier's writer, and a head whose chunks failed.

THE INCIDENT (2026-10-02 21:09, the first request after the restart; engine log and
results/disk/live-20261002210930.json). The disk tier's writer thread copied the session head's
rows out of the engine's buffers while the drafter captured its CUDA graph on the engine thread.
The copy failed (`cudaErrorStreamCaptureUnsupported`) after 1 of the head's 15 chunks, the
capture failed with it (`info.status != cudaStreamCaptureStatusInvalidated` in the caching
allocator) and the request ended in an error. The tier held 4 anchors and 1 chunk: no anchor's
chunk chain was whole, so the live check's next session found nothing on disk and prefilled its
17,074 tokens cold (22.5 s). And `plan` counted those anchors as written, so no later request was
ever admitted to write the missing chunks.

  1  every CUDA graph capture in engine/ and server/ holds `engine.capture_lock.LOCK` (the source,
     every `with torch.cuda.graph(...)`), and the drafter's capture does when it runs
  2  the writer copies nothing while a capture holds the lock, and finishes the job after it
  3  a head whose chunks failed part way is admitted again by the next session's first request,
     the missing chunks are written, and the session after it starts from the disk, bit for bit
     a cold prefill

The GPU side is tools/capture_race_check.py (a hold step).

Run: python tests/test_capture_lock.py
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import types

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from engine import cache, capture_lock, disk_cache  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402
from test_disk_cache import CH, Pos, exact, run, tier  # noqa: E402

TAG = {"client": "opencode", "head_end": 100}


# ------------------------------------------------------------------ 1. every capture holds it

def _is_graph_call(node) -> bool:
    f = node.func if isinstance(node, ast.Call) else None
    return (isinstance(f, ast.Attribute) and f.attr == "graph"
            and isinstance(f.value, ast.Attribute) and f.value.attr == "cuda")


def _is_lock(node) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "LOCK"
            and isinstance(node.value, ast.Name) and node.value.id == "capture_lock")


def test_every_capture_in_the_engine_and_the_server_holds_the_lock():
    found = []
    for top in ("engine", "server"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, top)):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                tree = ast.parse(open(path).read(), path)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.With):
                        continue
                    exprs = [it.context_expr for it in node.items]
                    if any(_is_graph_call(e) for e in exprs):
                        rel = os.path.relpath(path, ROOT)
                        found.append(rel)
                        assert any(_is_lock(e) for e in exprs), (
                            f"{rel}:{node.lineno}: a CUDA graph capture without "
                            f"capture_lock.LOCK in the same `with`")
    # N2: the narrator's decode step (engine/generative/qwen35.py) is the third
    # draft_graph.py twice: the lone draft graph and the packed draft of parallel requests
    # ENG-221: Flash-Next's decode step (engine/models/flashnext/engine.py DecodeGraph); ENG-227/153:
    # its verify block (VerifyGraph), its commit (CommitGraph) and its MTP chain draft (MTPChainGraph)
    # ENG-287: LFM2's decode step, joint decode step, verify block and commit (engine/models/lfm2_moe/engine.py)
    # ENG-319: NemotronH's decode step, joint decode step, verify block and commit; SPD-123: the LiLiCorr draft
    assert sorted(found) == ["engine/drafters/draft_graph.py", "engine/drafters/draft_graph.py",
                             "engine/drafters/lilicorr.py", "engine/generative/qwen35.py"] + [
                             "engine/models/flashnext/engine.py"] * 4 + ["engine/models/lfm2_moe/engine.py"] * 4 + [
                             "engine/models/nemotron_h/engine.py"] * 4 + ["engine/verify_graph.py"], found
    return f"{len(found)} capture sites, each under the lock"


def test_the_drafters_capture_runs_with_the_lock_held():
    from engine.drafters import draft_graph as dg
    seen = []

    class Graph:
        def replay(self):
            seen.append(("replay", capture_lock.LOCK._is_owned()))

    class Capture:
        def __init__(self, g, stream=None, pool=None):
            pass

        def __enter__(self):
            seen.append(("capture", capture_lock.LOCK._is_owned()))

        def __exit__(self, *exc):
            return False

    class Stream:
        def wait_stream(self, other):
            pass

    class On:
        def __init__(self, stream):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    saved = {k: getattr(torch.cuda, k) for k in ("graph", "CUDAGraph", "stream", "current_stream")}
    torch.cuda.graph, torch.cuda.CUDAGraph = Capture, Graph
    torch.cuda.stream, torch.cuda.current_stream = On, Stream
    try:
        g = dg.DraftGraph.__new__(dg.DraftGraph)
        g.d = types.SimpleNamespace(ctx_len=10, tree_temp=1.0)
        g.host, g.scal = torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)
        g.win, g.graphs, g.stream = 2048, {}, Stream()
        g.stats = {"captured": 0, "replayed": 0}
        g._body = lambda span, temp: ("cand", "scores")
        g.run(5, 9)
        g.run(6, 10)                                  # the same class: a replay only
    finally:
        for k, v in saved.items():
            setattr(torch.cuda, k, v)
    assert seen == [("capture", True), ("replay", False), ("replay", False)], seen
    assert not capture_lock.LOCK._is_owned()


# ------------------------------------------------------------------ 2. the writer waits

def test_the_writer_copies_nothing_while_a_capture_holds_the_lock():
    head = tokens(100, seed=31)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d)
        with capture_lock.LOCK:                       # the engine thread captures a graph
            cache.prefill(fresh(), Pos(), head, DEV, chunk=CH, disk=t, conv_id="ses_c",
                          disk_tag=TAG)
            assert not t.flush(timeout=0.5), "the writer went on inside the capture"
            assert not t.chunks
        assert t.flush()
        rows = t.report()["top"]
        assert rows and all(r["complete"] for r in rows) and len(t.chunks) == 6, rows
        t.close()


# ------------------------------------------------------------------ 3. a head that failed part way

def test_a_head_whose_chunks_failed_is_admitted_again_and_then_read_bit_for_bit():
    head = tokens(100, seed=21)
    s1, s2, s3 = (head + tokens(30, seed=k) for k in (22, 23, 24))
    with tempfile.TemporaryDirectory() as d:
        t = tier(d)
        real, n = disk_cache.write_file, {"chunks": 0}

        def failing(path, parts):
            if path.endswith(".chk"):
                n["chunks"] += 1
                if n["chunks"] > 1:
                    raise RuntimeError("CUDA error: operation not permitted when stream is "
                                       "capturing")
            return real(path, parts)

        disk_cache.write_file = failing
        try:
            run(fresh(), s1, t, Pos(), conv_id="ses_1", disk_tag=TAG)
        finally:
            disk_cache.write_file = real
        rows = t.report()["top"]
        assert rows and not any(r["complete"] for r in rows) and len(t.chunks) == 1, rows
        assert any(e["kind"] == "error" for e in t.report()["events"])
        # the next session's first request: nothing whole to read, so it prefills cold, and its
        # head is admitted again although its anchors are on disk; the chunks they lack go too
        _, reused, kind = run(fresh(), s2, t, Pos(), conv_id="ses_2", disk_tag=TAG)
        assert kind is None and reused == 0, kind
        rows = t.report()["top"]
        assert all(r["complete"] for r in rows) and len(t.chunks) == 6, rows
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, s3, t, dr, conv_id="ses_3", disk_tag=TAG)
        assert (kind, reused) == ("disk", 96), (kind, reused)
        exact(eng, lg, s3, dr)
        t.close()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:75s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
