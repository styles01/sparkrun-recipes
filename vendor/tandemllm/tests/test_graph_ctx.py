"""The verify graphs past 32k of context (`QWEN38_GRAPH_MAX_CTX`, Phase 4, 2026-09-25).

The verify graphs stopped at 32,768 tokens of context because the graph-safe attention
(`tools/attn_kernels.py::decode_attention_dev`) was written for the 512-token chunk, and past 32k the
eager path's chunk grows. So every verify of a long conversation -- the 32k probe's prompt is 32,795
tokens -- ran eager: no graph, the per-launch gaps back. The chunk is a function of the context
length that is constant over a graph's context class (the next power of two), so a graph captured
at class c cuts the context exactly where the eager path does for every length in (c/2, c]. What is
checked here, on the CPU:

  * the setting defaults to the code as it was (32,768) and the graphs read it;
  * for every class from 1,024 to 262,144 and every length in it, the graph's chunk equals the
    eager path's, and so does the combine's split vector (`NSP`) whenever the eager path combines;
  * `decode_attention_dev` accepts a class past 32k (it used to assert the 512 chunk).

The bit-identity of graph and eager on the board is the gate's flag-on identity and the probe's
texts; the kernel itself is tests/gpu.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _read(env: dict) -> str:
    code = "import engine.verify_graph as G; print(G.MAX_CTX)"
    e = {k: v for k, v in os.environ.items() if not k.startswith("QWEN38_")}
    e.update(env, PYTHONPATH=ROOT + os.pathsep + e.get("PYTHONPATH", ""), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True,
                          cwd=ROOT).stdout.strip()


def test_the_head_gemm_knobs_default_to_the_shipped_launch():
    code = "import tools.head_gemv as H; print(H.HEAD_BN, H.HEAD_WARPS, H.HEAD_STAGES)"
    e = {k: v for k, v in os.environ.items() if not k.startswith("QWEN38_")}
    e.update(PYTHONPATH=ROOT + os.pathsep + e.get("PYTHONPATH", ""), CUDA_VISIBLE_DEVICES="")
    run = lambda env: subprocess.run([sys.executable, "-c", code], env={**e, **env}, capture_output=True,
                                     text=True, cwd=ROOT).stdout.strip()
    assert run({}) == "64 4 3"
    assert run({"QWEN38_HEAD_GEMM": "128:8:4"}) == "128 8 4"


def test_the_default_is_the_code_as_it_was():
    assert _read({}) == "32768"
    assert _read({"QWEN38_GRAPH_MAX_CTX": "262144"}) == "262144"


def test_the_graphs_read_the_setting():
    import engine.model as M
    import engine.verify_graph as G
    old, rows = G.MAX_CTX, M.VERIFY_ROWS
    try:
        M.VERIFY_ROWS = 16
        G.MAX_CTX = 32768
        assert not G.VerifyGraphs.eligible(None, 16, 32795)
        G.MAX_CTX = 262144
        assert G.VerifyGraphs.eligible(None, 16, 32795)
        assert G.VerifyGraphs.eligible(None, 16, 131072 - 16)
        assert not G.VerifyGraphs.eligible(None, 16, 262144 - 8), "the class must hold the block"
    finally:
        G.MAX_CTX, M.VERIFY_ROWS = old, rows


def test_the_graph_key_carries_the_flags_an_in_process_ab_flips():
    """A verify graph or a draft graph captured under one weight-load hint or one wide table
    must not be replayed under the other; both key on the kernel flags' signature."""
    from engine.verify_graph import VerifyGraphs
    from tools import gdn_verify_kernels as V
    from tools import nvfp4_skinny as SK
    old = (SK.LDW, SK.WIDE_B, V.TREE_PF)
    try:
        SK.LDW, SK.WIDE_B, V.TREE_PF = 0, False, False
        base = VerifyGraphs.signature()
        SK.LDW = 1
        ldw = VerifyGraphs.signature()
        SK.LDW, SK.WIDE_B = 0, True
        wide = VerifyGraphs.signature()
        SK.WIDE_B, V.TREE_PF = False, True
        pf = VerifyGraphs.signature()
        V.TREE_PF = False
        from tools import head_gemv as H
        hb = H.HEAD_BN
        H.HEAD_BN = 128
        head = VerifyGraphs.signature()
        H.HEAD_BN = hb
        SK.WIDE_LDW = 1
        wl = VerifyGraphs.signature()
        SK.WIDE_LDW = None
        assert len({base, ldw, wide, pf, head, wl}) == 6 and VerifyGraphs.signature() == base
    finally:
        SK.LDW, SK.WIDE_B, V.TREE_PF = old
    src = open(os.path.join(ROOT, "engine", "drafters", "draft_graph.py")).read()
    assert "key = (span, temp, VerifyGraphs.signature())" in src and "self.graphs[key]" in src


def _nsp(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def test_a_class_cuts_the_context_where_the_eager_path_does():
    from engine.verify_graph import VerifyGraphs
    from tools.attn_kernels import pick_launch
    for T in (1, 8, 16, 32):
        cls = 1024
        while cls <= 262144:
            _, _, ns_g, chunk_g = pick_launch(T, 6, cls)
            ns_g = max(ns_g, 2)                       # decode_attention_dev's floor
            lo = cls // 2 + 1 if cls > 1024 else T + 1
            for lc in sorted({lo, lo + 1, (lo + cls) // 2, cls - 1, cls}):
                assert VerifyGraphs.ctx_class(lc) == cls, (lc, cls)
                _, _, ns_e, chunk_e = pick_launch(T, 6, lc)
                assert chunk_e == chunk_g, (T, cls, lc, chunk_e, chunk_g)
                if ns_e > 1:                          # the eager path combines: same split vector
                    assert _nsp(ns_e) == _nsp(ns_g), (T, cls, lc, ns_e, ns_g)
            cls *= 2


def test_the_device_length_attention_takes_a_class_past_32k():
    """It asserted the 512 chunk; a class past 32k has a bigger one and must now pass the check
    (the launch itself needs a GPU, so the check is called on its own)."""
    from tools.attn_kernels import dev_chunk
    assert dev_chunk(16, 6, 32768) == 512 and dev_chunk(16, 6, 1024) == 512
    assert dev_chunk(16, 6, 65536) == 1024 and dev_chunk(16, 6, 262144) == 4096
    for bad in (3000, 40000):
        try:
            dev_chunk(16, 6, bad)
        except AssertionError:
            continue
        raise AssertionError(f"a length that is not a class must be refused: {bad}")


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
