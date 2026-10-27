"""CPU test for tools/ttft_probe.py: the per-prompt paired difference is what it reports, so a
configuration that adds 2 ms to every prompt reads +2 ms whatever the prompts' own spread."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.ttft_probe import parse_config, summarize  # noqa: E402


def test_the_paired_difference_and_the_config_spec():
    assert parse_config("eng107:QWEN38_A=1,QWEN38_B=x") == ("eng107", {"QWEN38_A": "1",
                                                                     "QWEN38_B": "x"})
    assert parse_config("base:") == ("base", {})
    base = {1: {f"p{i}": [0.400 + 0.01 * i, 0.401 + 0.01 * i] for i in range(20)}}
    slow = {1: {f"p{i}": [0.402 + 0.01 * i, 0.403 + 0.01 * i] for i in range(20)}}
    lines = summarize({"base": base, "slow": slow})
    row = next(line for line in lines if line.startswith("slow"))
    assert row.rstrip().endswith("+2.00"), row


if __name__ == "__main__":
    test_the_paired_difference_and_the_config_spec()
    print("  ok  test_the_paired_difference_and_the_config_spec\n\n1/1 passed")
