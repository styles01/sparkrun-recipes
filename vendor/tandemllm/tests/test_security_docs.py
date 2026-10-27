"""OPS-202: SECURITY.md and docs/threat-model.md hold what OPS-201 says, and every path they name exists.

The test matrix (the Gherkin of OPS-201):

  1  SECURITY.md names GitHub's private reporting on 0xBakeer/TandemLLM as the only channel, says a
     public issue would show the report, and names no e-mail address      test_one_private_channel
  2  it asks for the version, the part, the trigger and the gain; a script or a failing test
                                                                     test_what_a_report_says
  3  an admin route without a password is expected, and the link to the admin rule resolves; more
     than the route, or a way out of the sandbox or its egress, is in scope  test_the_admin_rule
  4  out of scope: a shell on the box, model quality, GPU kernel numbers, third-party weights,
     bench/ and tools/                                               test_out_of_scope
  5  no promises: no reply time, advisory or credit, test before a release, supported versions or
     severity table                                                  test_no_promises
  6  the threat model has its four parts, every repo path and test script it names exists, its
     links resolve, and docs/README.md links it                      test_the_threat_model

Whether private vulnerability reporting is on is a repository setting: `gh api
repos/0xBakeer/TandemLLM/private-vulnerability-reporting` before a release, not here.

Run: python tests/test_security_docs.py
"""

from __future__ import annotations

import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORM = "https://github.com/0xBakeer/TandemLLM/security/advisories/new"
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _prose(md: str) -> str:
    """The text without link targets, on one line, lower case."""
    return " ".join(re.sub(r"\]\([^)]*\)", "]", md).split()).lower()


def _anchors(md: str) -> set[str]:
    return {re.sub(r"[^a-z0-9 -]", "", h.strip().lower()).replace(" ", "-")
            for h in re.findall(r"^#+ (.+)$", md, re.M)}


def _links_resolve(md: str, base: str) -> list[str]:
    bad = []
    for target in LINK.findall(md):
        if target.startswith(("http://", "https://")):
            continue
        path, _, anchor = target.partition("#")
        full = os.path.normpath(os.path.join(ROOT, base, path))
        if not os.path.exists(full):
            bad.append(target)
        elif anchor and anchor not in _anchors(open(full, encoding="utf-8").read()):
            bad.append(target)
    return bad


SEC = _read("SECURITY.md")
TM = _read("docs", "threat-model.md")


def test_one_private_channel():
    assert FORM in LINK.findall(SEC), "SECURITY.md links GitHub's private report form"
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", SEC), "SECURITY.md names no e-mail address"
    others = [t for t in LINK.findall(SEC) if t.startswith("http") and t != FORM]
    assert not others, others
    p = _prose(SEC)
    assert "public issue or pull request would show it" in p


def test_what_a_report_says():
    p = _prose(SEC)
    for need in ("`version`", "get /v1/models", "the part of tandemllm", "the request or file that sets it off",
                 "what an attacker gets", "a script or a failing test helps most"):
        assert need in p, need


def test_the_admin_rule():
    assert "docs/server.md#the-admin-rule" in LINK.findall(SEC)
    assert not _links_resolve(SEC, "."), _links_resolve(SEC, ".")
    p = _prose(SEC)
    assert "an admin route that answers without a password is therefore expected, not a bug" in p
    assert "gets more than the route is for" in p and "is in scope" in p
    assert "out of a chat's sandbox or past its egress rules is in scope" in p


def test_out_of_scope():
    p = _prose(SEC)
    out = p[p.index("out of scope:"):]
    for item in ("a shell on the box", "model output quality", "gpu kernels", "third-party weights",
                 "`bench/`", "`tools/`"):
        assert item in out, item


def test_no_promises():
    p = _prose(SEC)
    for word in (r"\bwithin\b", r"\bdays?\b", r"\bhours?\b", r"\brespond", r"\backnowledg", r"\badvisor(y|ies)\b",
                 r"\bcredit", r"\bcve\b", r"before (a|the|each|every) release", r"supported versions?",
                 r"\bseverity\b", r"\bbounty\b"):
        assert not re.search(word, p), word
    assert not re.search(r"^\s*\|", SEC, re.M), "SECURITY.md has no table"


def test_the_threat_model():
    for head in ("## What runs and where input enters", "## Parts that matter most", "## Where to test",
                 "## Not in scope"):
        assert head in TM, head
    assert not _links_resolve(TM, "docs"), _links_resolve(TM, "docs")
    missing = []
    for code in re.findall(r"`([^`\s]+)`", TM):
        if code.startswith(("/", "-")) or "<" in code or "/" not in code and not code.startswith("test_"):
            continue
        if code.startswith("test_"):
            code = os.path.join("tests", code)
        if not glob.glob(os.path.join(ROOT, code.rstrip("/"))):
            missing.append(code)
    assert not missing, missing
    assert "(threat-model.md)" in _read("docs", "README.md")
    return f"{len(re.findall(r'`test_[a-z0-9_]+[.]py`', TM))} test scripts named"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            note = fn()
            print(f"  {name:40s} ok {note or ''}")
            passed += 1
    print(f"{passed} passed")
