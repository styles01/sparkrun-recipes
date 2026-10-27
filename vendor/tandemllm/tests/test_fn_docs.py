"""OPS-101: Flash-Next is documented like the 27B, no GPU.

  * every QWEN38_FN_* switch in engine/settings.py is in the manual's generated switch table
    (manual/generated/engine-knobs.tex, `manual/tools/gen_reference.py`) with its default;
  * Wren's knowledge index (server/setup/knowledge.py) reads the Flash-Next registry entry: a question
    about the 6.05 pack's context finds that it fits no window, one about 3.05 its largest window and
    its warm n-gram table;
  * the docs name Flash-Next where the 27B's chapters describe the same thing (setup, operations,
    caches, speculation, server).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _tex(s: str) -> str:
    return s.replace("_", "\\_\\allowbreak{}")


def test_every_flashnext_switch_is_in_the_manual():
    from engine.settings import KNOBS
    table = open(os.path.join(ROOT, "manual", "generated", "engine-knobs.tex")).read()
    fn = sorted(k for k in KNOBS if k.startswith("FN_"))
    assert len(fn) >= 10, fn
    missing = [k for k in fn if f"\\texttt{{QWEN38\\_\\allowbreak{{}}{_tex(k)}}} & \\texttt{{{_tex(KNOBS[k][0])}}}"
               not in table]
    assert not missing, f"not in manual/generated/engine-knobs.tex (run manual/tools/gen_reference.py): {missing}"
    return f"{len(fn)} QWEN38_FN_* switches with their defaults"


def test_wren_finds_each_packs_window_and_table():
    from server.setup.knowledge import Knowledge
    k = Knowledge(ROOT, packages=())
    hits = k.search("flashnext-6.05 EXL3 6.05 bpw context window fit", k=6)
    six = [h for h in hits if h["path"] == "models/qwen3.8-flash-next/model.yaml" and "6.05" in h["title"]]
    assert six and "does not fit at any window" in six[0]["text"], [(h["path"], h["title"]) for h in hits]
    hits = k.search("flashnext-3.05 EXL3 3.05 bpw longest conversation n-gram table", k=6)
    three = [h for h in hits if h["path"] == "models/qwen3.8-flash-next/model.yaml" and "3.05" in h["title"]]
    assert three and "longest conversation it holds" in three[0]["text"] and "row cache" in three[0]["text"], hits
    return "6.05: fits no window; 3.05: its largest window and its cold table"


def test_the_docs_name_flashnext():
    names = ("setup.md", "operations.md", "caches.md", "speculative-decoding.md", "server.md")
    missing = [n for n in names if "Flash-Next" not in open(os.path.join(ROOT, "docs", n)).read()]
    assert not missing, missing
    return ", ".join(names)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<50} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<50} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
