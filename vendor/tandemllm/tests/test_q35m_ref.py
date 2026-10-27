"""ENG-383: the Qwen3.5 MoE reference forward against transformers' Qwen3_5MoeForCausalLM on tiny random
configs of both shapes (Ornith's and MiniCPM-V 4.7's head and expert ratios), fp32 on the CPU: one pass,
and a prefill followed by single-token steps. Skipped without transformers' qwen3_5_moe."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(shape: str):
    try:
        import transformers  # noqa: F401
        from transformers import Qwen3_5MoeForCausalLM  # noqa: F401
    except Exception:                                              # noqa: BLE001
        return "skipped: transformers without qwen3_5_moe"
    from tools import q35m_refcheck
    sys.argv = ["q35m_refcheck", "--shape", shape, "--tokens", "24"]
    assert q35m_refcheck.main() == 0


def test_ornith_shape():
    _run("ornith")


def test_minicpm_shape():
    _run("minicpm")


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
