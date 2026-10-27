"""How many rows a verify takes the fast path at, as one setting.

`QWEN38_VERIFY_ROWS` (default 16, the code as it was) is read by the three places that used to say
sixteen: the fused GDN verify mixer's routing for a chain, the fold's static factor buffers and the
verify graphs' eligibility. On a CPU none of the three runs (they are Triton and CUDA graphs), so
what is checked here is that the default is sixteen and that the graphs read the setting at call
time; the kernels at 17..32 rows are tests/gpu/test_gdn_gpu.py, the engine is the gate.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _read(env: dict) -> str:
    code = "import engine.model as M; print(M.VERIFY_ROWS)"
    e = {k: v for k, v in os.environ.items() if not k.startswith("QWEN38_")}
    e.update(env, PYTHONPATH=ROOT + os.pathsep + e.get("PYTHONPATH", ""), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True,
                          cwd=ROOT).stdout.strip()


def test_the_default_is_the_code_as_it_was():
    assert _read({}) == "16"
    assert _read({"QWEN38_VERIFY_ROWS": "32"}) == "32"


def test_the_graphs_take_what_the_setting_allows():
    import engine.model as M
    from engine.verify_graph import MAX_CTX, VerifyGraphs
    old = M.VERIFY_ROWS
    try:
        M.VERIFY_ROWS = 16
        assert VerifyGraphs.eligible(None, 16, 100) and not VerifyGraphs.eligible(None, 17, 100)
        M.VERIFY_ROWS = 32
        assert VerifyGraphs.eligible(None, 17, 100) and VerifyGraphs.eligible(None, 32, 100)
        assert not VerifyGraphs.eligible(None, 33, 100) and not VerifyGraphs.eligible(None, 1, 100)
        assert not VerifyGraphs.eligible(None, 32, MAX_CTX - 16), "the context limit still holds"
    finally:
        M.VERIFY_ROWS = old


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
