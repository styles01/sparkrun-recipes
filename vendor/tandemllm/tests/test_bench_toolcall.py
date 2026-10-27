"""The harness's scoring, on a CPU: what counts as parsed, named and faithful.

Run: python tests/test_bench_toolcall.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.bench_toolcall import SCENARIOS, TOOLS, score, summarise  # noqa: E402


def test_an_exact_answer_scores_one_everywhere():
    want = [("write_file", {"path": "/a", "content": "x"}), ("read_file", {"path": "/b"})]
    s = score(list(reversed(want)), want, "I'll do both.")
    assert s["parse"] and s["name"] and s["args_exact"] == s["args_total"] == 3 and not s["misses"]


def test_a_near_miss_is_written_down():
    want = [("write_file", {"path": "/a", "content": "hello world"})]
    s = score([("write_file", {"path": "/a", "content": "hello world\n"})], want, "")
    assert s["parse"] and s["name"] and (s["args_exact"], s["args_total"]) == (1, 2)
    assert s["misses"] == [{"function": "write_file", "param": "content", "want": "hello world",
                            "got": "hello world\n"}]


def test_leftover_block_text_is_a_parse_failure():
    want = [("read_file", {"path": "/b"})]
    s = score([("read_file", {"path": "/b"})], want,
              "<think>\n<tool_call>x</tool_call>\n</think>\n\nok <tool_call><function=read")
    assert not s["parse"], "an unread block in the answer is a miss"
    s = score([("read_file", {"path": "/b"})], want, "<think>\n<tool_call>x</tool_call>\n</think>\n\n")
    assert s["parse"], "a block inside the reasoning is not the answer's"


def test_wrong_count_and_wrong_name():
    want = [("read_file", {"path": "/a"}), ("read_file", {"path": "/b"})]
    s = score([("read_file", {"path": "/a"})], want, "")
    assert not s["parse"] and not s["name"] and s["args_exact"] == 1
    s = score([("list_dir", {"path": "/a"}), ("read_file", {"path": "/b"})], want, "")
    assert s["parse"] and not s["name"] and s["args_exact"] == 1


def test_the_summary_pools_runs_and_rates_compliance():
    rows = [dict(score([("read_file", {"path": "/x"})], [("read_file", {"path": "/x"})], ""),
                 calls=[("read_file", {"path": "/x"})], stream=st) for st in (False, True)]
    choice = {"auto": [{"needs_tool": True, "calls": ["read_file"]},
                       {"needs_tool": False, "calls": []}],
              "none": [{"complied": True}, {"complied": False}]}
    s = summarise({"scenarios": {"single/read": rows}, "choice": choice, "malformed": {},
                   "round_trip": [{"completed": True}, {"completed": False}]})
    assert s["overall"] == {"n": 2, "parse": 1.0, "name": 1.0, "args": 1.0}
    assert s["scenarios"]["single/read"]["stream_agrees"]
    assert s["choice"]["auto"] == {"call_rate_when_needed": 1.0, "call_rate_when_not": 0.0}
    assert s["choice"]["none"]["compliance"] == 0.5 and s["round_trip"] == 0.5


def test_the_matrix_is_fixed():
    assert [t["function"]["name"] for t in TOOLS] == ["write_file", "read_file", "list_dir", "run"]
    # nine string scenarios since, two typed ones since
    assert len(SCENARIOS) == 11 and SCENARIOS["json/answer"][1] == []
    assert [k for k in SCENARIOS if k.startswith("typed/")] == ["typed/read_lines",
                                                               "typed/tag_items"]


def test_a_typed_value_is_exact_only_with_its_type():
    # "40" for an integer is the miss opencode refused; True is not 1
    want = [("read_lines", {"path": "/a", "offset": 40, "follow": False})]
    s = score([("read_lines", {"path": "/a", "offset": "40", "follow": 0})], want, "")
    assert (s["args_exact"], s["args_total"]) == (1, 3)
    assert s["by_type"] == {"str": [1, 1], "int": [0, 1], "bool": [0, 1]}
    s = score([("read_lines", {"path": "/a", "offset": 40, "follow": False})], want, "")
    assert s["args_exact"] == 3 and not s["misses"]
    from tools.bench_toolcall import TYPED_TOOLS
    assert SCENARIOS["typed/read_lines"][2]["tools"] is TYPED_TOOLS
    assert "tools" not in SCENARIOS["single/read"][2], "string scenarios keep their prompt"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:56s} ok")
            passed += 1
    print(f"{passed} passed")
