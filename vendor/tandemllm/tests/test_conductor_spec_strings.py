"""A hardcoded e2e assertion and the shipped string it reads are one pair (GitHub issue
#6, deep audit 2026-10-16).

`conductor.spec.ts` asserted the notify-rule paragraph says "the owner on <date>"
while the shipped sentence is "A design decision of <date>: …" — a deterministic
drift, a mock run could only ever fail on it. The spec now asks for the shipped
wording; this test keeps the pair true without a browser, for the one shape of
drift a static grep can catch (both sides hardcoded).

Run: python tests/test_conductor_spec_strings.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "dashboard", "e2e", "conductor.spec.ts")
VIEW = os.path.join(ROOT, "dashboard", "src", "views", "conductor", "profiles.ts")


def _pair(anchor: str) -> tuple[str, str]:
    """(the spec's expected substring, the view's rendered text of that slot)."""
    with open(SPEC, encoding="utf-8") as fh:
        spec = fh.read()
    want = re.search(rf"\[data-{anchor}\]'\)\)\s*\.toContainText\('([^']*)'\)", spec)
    assert want, f"the {anchor} assertion moved out of shape in the spec"
    with open(VIEW, encoding="utf-8") as fh:
        view = fh.read()
    rendered = re.search(rf'data-{anchor}[^>]*>([^<]*)<', view)
    assert rendered, f"the {anchor} paragraph moved out of shape in the view"
    return want.group(1), rendered.group(1)


def test_the_notify_rule_pair_holds():
    want, sentence = _pair("notify-rule")
    assert want and want in sentence, f"the spec expects {want!r}, the view says {sentence!r}"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
