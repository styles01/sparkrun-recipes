"""OPS-149: the install text for a simple user, and no model count in the docs or the dashboard, no GPU.

The test matrix (the Gherkin of OPS-149):

  1  the README's Install section, docs/setup.md's Setup mode and the manual's section "A new box" each
     open with exactly two commands (the installer's curl line and `bash install.sh --setup --yes`),
     with no weight profile in them                          test_two_commands
  2  each says how to open the Setup Guide: on the box at http://127.0.0.1:8000/dashboard/#/setup, from
     another computer through `ssh -L 8000:127.0.0.1:8000`, or with `--host 0.0.0.0`
                                                             test_how_to_open_the_guide
  3  each has an "Advanced: from the command line" part whose first command is use-profile.sh --list
                                                             test_the_command_line_is_advanced
  4  install.sh's setup message names the Setup Guide's address on the box, the ssh tunnel and
     --host 0.0.0.0 (tests/test_fresh_install.py runs it on Linux)   test_the_installer_says_where
  5  no README, docs, manual or dashboard text counts the models ("five models", "three models")
                                                             test_no_model_count

Run: python tests/test_install_docs.py
"""

from __future__ import annotations

import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "http://127.0.0.1:8000/dashboard/#/setup"
CURL = re.compile(r"^curl -fsSLO https://raw\.githubusercontent\.com/0xBakeer/TandemLLM/v[0-9.]+/install\.sh$")
PROFILE = re.compile(r"\b(?:flashnext|exl3|lfm25|nemotron|nvfp4)-[0-9a-z.]")


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _section(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.find(end, i + len(start))
    return text[i:j if j >= 0 else len(text)]


def _blocks(md: str) -> list[list[str]]:
    """The fenced code blocks of a Markdown text, as their non-empty lines without comments."""
    out = []
    for body in re.findall(r"```(?:bash|sh)?\n(.*?)```", md, re.S):
        out.append([ln.split("  #")[0].strip() for ln in body.splitlines() if ln.strip()])
    return out


def _shells(tex: str) -> list[list[str]]:
    return [[ln.strip() for ln in body.splitlines() if ln.strip()]
            for body in re.findall(r"\\begin\{shell\}\n(.*?)\\end\{shell\}", tex, re.S)]


def _docs() -> dict[str, tuple[str, list[list[str]], str]]:
    """name -> (the simple-install text, its command blocks, the advanced part)"""
    readme = _section(_read("README.md"), "## Install\n", "\n## Quick start")
    setup = _read("docs", "setup.md")
    simple = _section(setup, "## Setup mode\n", "\n## The model registry")
    tex = _section(_read("manual", "chapters", "install-engine.tex"), r"\section{A new box", r"\section{Installing from")
    return {
        "README.md": (readme, _blocks(readme), _section(readme, "### Advanced: from the command line", "\n| flag |")),
        "docs/setup.md": (simple, _blocks(simple), _section(simple, "## Advanced: from the command line", "\n| profile |")),
        "manual install-engine.tex": (tex.replace(r"\#", "#"), _shells(tex),
                                      _section(tex, r"\subsection{Advanced: from the command line}", r"\begin{table}")),
    }


def test_two_commands():
    for name, (_text, blocks, _adv) in _docs().items():
        first = blocks[0]
        assert len(first) == 2 and CURL.match(first[0]) and first[1] == "bash install.sh --setup --yes", (name, first)
        assert not PROFILE.search(" ".join(first)), (name, first)


def test_how_to_open_the_guide():
    for name, (text, _cmds, _adv) in _docs().items():
        before = text.split("Advanced: from the command line")[0]
        for want in (URL, "ssh -L 8000:127.0.0.1:8000", "--host 0.0.0.0"):
            assert want in before, (name, want)
        assert "flashnext-3.05" not in before, name


def test_the_command_line_is_advanced():
    for name, (_text, _cmds, adv) in _docs().items():
        cmds = [ln for b in (_blocks(adv) or _shells(adv)) for ln in b]
        assert cmds and cmds[0] == "bash ~/TandemLLM/run/ops/use-profile.sh --list", (name, cmds)
        assert all(not PROFILE.search(c) for c in cmds), (name, cmds)


def test_the_installer_says_where():
    src = _read("install.sh")
    msg = src[src.index("# no model is loaded yet"):src.index("    exit 0\nfi\nif [ \"$START\" != 1 ]")]
    for want in ("Open the Setup Guide in a browser and pick a model", "http://127.0.0.1:%s/dashboard/#/setup",
                 "ssh -L %s:127.0.0.1:%s", "--host 0.0.0.0", "From your network:"):
        assert want in msg, want


# a count of the models: "five models", "3 models", "all five share"; not "the Qwen3.8 models" or a
# step list's "0 models"
COUNT = re.compile(r"(?<![\w.])(?:two|three|four|five|six|seven|eight|nine|ten|[2-9]|1[0-9])\s+"
                   r"(?:big |base |large |supported |different )?models\b|\ball (?:three|four|five|six|seven) share\b", re.I)
# the intro's "two models working in tandem" is the drafter and the model, not the set
ALLOWED = ("two models working in tandem",)


def test_no_model_count():
    files = (["README.md"] + glob.glob(os.path.join(ROOT, "docs", "**", "*.md"), recursive=True)
             + glob.glob(os.path.join(ROOT, "manual", "chapters", "*.tex"))
             + glob.glob(os.path.join(ROOT, "dashboard", "src", "**", "*.ts"), recursive=True))
    hits = []
    for f in files:
        text = _read(f)
        for m in COUNT.finditer(text):
            line = text[text.rfind("\n", 0, m.start()) + 1:text.find("\n", m.end())]
            if not any(a in line for a in ALLOWED):
                hits.append(f"{os.path.relpath(os.path.join(ROOT, f), ROOT)}: {m.group(0)!r}")
    assert not hits, hits


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
