"""SRV-361: a client's pins and blocks in the body (`tandem.packages`), read by the router
(server/conductor/packages.py `body_packages`, used in router._block). No change for a body without
the field.

Run: python -m pytest -q tests/test_package_pins.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor.packages import body_packages  # noqa: E402

KNOWN = {"coding": {}, "web-design": {}, "fact-check": {}}


def test_no_field_no_change():
    assert body_packages({"messages": []}, KNOWN) == ([], set(), [])
    assert body_packages({"tandem": {"compaction": False}}, KNOWN) == ([], set(), [])
    assert body_packages({"tandem": {"packages": "coding"}}, KNOWN) == ([], set(), [])


def test_pins_in_order_blocks_win_unknown_reported():
    pin, block, unknown = body_packages({"tandem": {"packages": {
        "pin": ["web-design", "coding", "web-design", "nope", "fact-check", 3],
        "block": ["fact-check"]}}}, KNOWN)
    assert pin == ["web-design", "coding"] and block == {"fact-check"} and unknown == ["nope"]


if __name__ == "__main__":
    test_no_field_no_change()
    test_pins_in_order_blocks_win_unknown_reported()
    print("ok")
