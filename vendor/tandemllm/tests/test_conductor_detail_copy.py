"""_detail() redacts a copy — the caller's own event keeps its shape.

The function shallow-copied the dict but rewrote the nested findings' 'detail'
in place: the caller (monitor_sink handing an event to several sinks) saw its
own findings' details stripped of 'arguments'/'calls'/'t_end' as a side effect
of someone else asking for the journalled view.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor.decisions import _detail  # noqa: E402


def test_caller_findings_keep_their_fields():
    ev = {"id": "e1", "findings": [{"file": "a.py",
                                    "detail": {"arguments": "secret args",
                                     "calls": [{"function": {"name": "read"}}],
                                     "t_end": 12, "kept": "visible"}}]}
    out = _detail(ev)
    # the copy is redacted...
    assert out["findings"][0]["detail"] == {"kept": "visible", "calls": ["read"]}
    # ...and the caller's event is untouched
    assert ev["findings"][0]["detail"]["arguments"] == "secret args" and \
               ev["findings"][0]["detail"]["t_end"] == 12


def test_non_dict_findings_pass_through():
    ev = {"findings": ["plain", {"detail": "string"}]}
    out = _detail(ev)
    assert out["findings"] == ["plain", {"detail": "string"}]
    assert ev["findings"] == ["plain", {"detail": "string"}]


if __name__ == "__main__":
    test_caller_findings_keep_their_fields()
    test_non_dict_findings_pass_through()
    print("2 passed")
