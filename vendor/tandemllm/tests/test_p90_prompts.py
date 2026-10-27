"""CPU test for tools/p90_prompts.py: the rates are row3's, warm-ups are left out, and the table
names the prompt that sits at the 45th place."""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.p90_prompts import per_prompt, table  # noqa: E402


def _report(d, name, rates):
    runs = []
    for i, rs in enumerate(rates):
        reqs = [{"prompt_id": "warm", "warmup": True, "completion_tokens": 256, "e2e_ms": 1000,
                 "ttft_ms": 500}]
        for pid, r in rs.items():
            # (256 - 1) / decode_s = r
            reqs.append({"prompt_id": pid, "warmup": False, "completion_tokens": 256,
                         "ttft_ms": 400.0, "e2e_ms": 400.0 + 1000.0 * 255 / r})
        rec = os.path.join(d, f"{name}-{i}.json")
        json.dump({"raw": {"payload": {"requests": reqs}}}, open(rec, "w"))
        runs.append({"record": rec})
    p = os.path.join(d, f"{name}.json")
    json.dump({"runs": runs}, open(p, "w"))
    return p


def test_the_rate_is_row3s_and_the_prompt_at_p90_is_named():
    d = tempfile.mkdtemp()
    base = {f"p{i:02d}": 20.0 + i for i in range(50)}
    other = dict(base, p44=40.0)                      # the 45th prompt's answer decodes slower
    a = _report(d, "a", [base, base])
    b = _report(d, "b", [other])
    runs = per_prompt(a)
    assert len(runs) == 2 and "warm" not in runs[0]
    assert abs(runs[0]["p44"] - 64.0) < 1e-9
    out = table({"a": per_prompt(a), "b": per_prompt(b)})
    assert any(line.startswith("p44") and "64.0" in line and "40.0" in line for line in out), out


if __name__ == "__main__":
    test_the_rate_is_row3s_and_the_prompt_at_p90_is_named()
    print("  ok  test_the_rate_is_row3s_and_the_prompt_at_p90_is_named\n\n1/1 passed")
