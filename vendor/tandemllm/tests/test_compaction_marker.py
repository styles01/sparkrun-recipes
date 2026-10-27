"""SRV-351: the explicit compaction marker. `tandem: {"compaction": true}` in a body makes it a
compaction without the shape guess (signals.shape), and a marked body is never rewritten as
opencode's text-rendered history (compaction.split); without the field nothing changes.

Run: python -m pytest -q tests/test_compaction_marker.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import compaction  # noqa: E402
from server.conductor import signals  # noqa: E402

OPENCODE = {"messages": [
    {"role": "system", "content": "You are a context summarization agent. You are given a "
                                  "conversation between a user and an agent."},
    {"role": "user", "content": "Here is the conversation so far: <conversation>\n[User]: fix the "
                                "flaky test in the worker please\n[Assistant]: done\n[User]: now "
                                "run it two hundred times\n</conversation>\nSummarize it."}]}
CHAT = {"messages": [{"role": "user", "content": "a long paste " * 2000},
                     {"role": "assistant", "content": "ok"},
                     {"role": "user", "content": "Summarize the conversation so far."}]}


def test_the_marker_reads_only_booleans():
    assert compaction.marker({"tandem": {"compaction": True}}) is True
    assert compaction.marker({"tandem": {"compaction": False}}) is False
    assert compaction.marker({"tandem": {"compaction": "yes"}}) is None
    assert compaction.marker({"tandem": None}) is None and compaction.marker({}) is None


def test_without_the_field_nothing_changes():
    assert compaction.split(OPENCODE) is not None
    assert signals.shape(OPENCODE, client_kind="opencode").compaction is True
    assert signals.shape(CHAT, client_kind="dashboard").compaction is False


def test_the_marker_decides_and_the_shape_is_not_guessed():
    marked = dict(CHAT, tandem={"compaction": True})
    assert signals.shape(marked, client_kind="dashboard").compaction is True
    assert compaction.split(marked) is None                 # never rewritten
    not_one = dict(OPENCODE, tandem={"compaction": False})
    assert signals.shape(not_one, client_kind="opencode").compaction is False
    assert compaction.split(not_one) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
