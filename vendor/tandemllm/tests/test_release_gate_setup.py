"""The Setup Guide's release gate itself (OPS-177, QA OPS-178): tools/release_gate_setup.sh keeps to its
rules, and the release checklist names it. The gate's own run on the box is the end-to-end part; its
report is attached to OPS-177.

  1  it parses; its ports are spare ones (never 8000, 8009 or 8016); its scratch install runs no chatd,
     the fake engine and its own process table; the interpreter copy is not named python*; it stops
     what it started on every exit                           test_the_gate_keeps_its_rules
  2  both runs are there: the browser at 390 px with a failed try (plain words, Copy diagnostics,
     Retry), the restart card, the finished screen and the Remove button, and the stripped execute
     bits                                                    test_both_runs
  3  docs/operations.md's release checklist lists it as mandatory, and the main Playwright suite
     leaves its spec to it                                   test_the_checklist

Run: python tests/test_release_gate_setup.py
"""
from __future__ import annotations

import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(ROOT, "tools", "release_gate_setup.sh")


def _text() -> str:
    return open(GATE).read()


def test_the_gate_keeps_its_rules():
    assert subprocess.run(["bash", "-n", GATE], capture_output=True).returncode == 0
    t = _text()
    assert "case \"$p\" in 8000|8009|8016) continue ;; esac" in t
    assert re.search(r"shuf -i 18100-18900", t)
    for line in ("QSE_CHATD=off", "ENGINE_PROC=$PROC", "QSE_ENGINE_FLAGS=\"--fake-engine --fake-load-s $LOAD_S\""):
        assert line in t, line
    assert 'cp "$GATE/pyenv/bin/$(basename "$REALPY")" "$GATE/pyenv/bin/tandem-gate-py"' in t
    assert "trap cleanup EXIT" in t and 'engine_status.py" stop --port' in t and 'kill "$KEEPER"' in t
    assert 'HOME="$H"' in t and "env -i" in t                       # the scratch's own HOME, nothing inherited
    assert '"advanced": {"port": %s, "host": "127.0.0.1"}' in t     # never the live port


def test_both_runs():
    t = _text()
    assert "Run A:" in t and "Run B:" in t
    assert 'chmod -x "$DIR/src/ops/start.sh" "$DIR/run/ops/start.sh" "$DIR/run/ops/stop.sh" "$DIR/run/ops/use-profile.sh"' in t
    assert "mcr.microsoft.com/playwright:v1.63.0-noble" in t and "playwright.gate.config.ts" in t
    assert '/v1/setup/remove \'{"profiles": ["exl3-3.0"], "confirm": true}\'' in t
    assert "printf '\\nif then fi (\\n' >> \"$DIR/run/ops/stop.sh\"" in t and 'echo fixed > "$OUT/shots/fixed.txt"' in t
    spec = open(os.path.join(ROOT, "dashboard", "e2e", "setup-gate.spec.ts")).read()
    assert "[data-restart-card]" in spec and "[data-start-chatting]" in spec and "timeout: 10_000" in spec
    for sel in ("[data-job-failed-end]", "[data-copy-diagnostics]", "[data-diagnostics-text]", "[data-job-retry]",
                "[data-remove-confirm]", "[data-remove-yes]", "retry_of"):
        assert sel in spec, sel
    cfg = open(os.path.join(ROOT, "dashboard", "playwright.gate.config.ts")).read()
    assert "width: 390" in cfg


def test_the_checklist():
    ops = open(os.path.join(ROOT, "docs", "operations.md")).read()
    sec = ops[ops.index("## Release checklist"):ops.index("## Deploying a new version")]
    assert "`bash tools/release_gate_setup.sh` (OPS-177), mandatory" in sec, sec
    main = open(os.path.join(ROOT, "dashboard", "playwright.config.ts")).read()
    assert main.count("testIgnore: /(shots|setup-gate)\\.spec\\.ts/") == 3


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:40s} ok")
            passed += 1
    print(f"{passed} passed")
