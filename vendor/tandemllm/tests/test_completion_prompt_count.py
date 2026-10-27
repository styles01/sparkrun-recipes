"""SRV-443 (QA SRV-444): a /v1/completions text prompt through `build_prompt` answers every `count`
form a chat prompt does: the host ids `_par_side_need` reads before it picks a slot, the length,
and (the length, not inside <think>). Before, the text branch returned the tensor for each, and
`_shared_head` raised on it: every /v1/completions request with parallel slots was a 500.

Run: python tests/test_completion_prompt_count.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import QTok, setup  # noqa: E402

from server import app  # noqa: E402


def test_a_text_prompt_answers_every_count():
    setup()
    app.STATE["tok"] = QTok()
    body = {"prompt": "Once upon a time"}
    ids, kind, in_think = app.build_prompt(body)
    want = QTok().encode("Once upon a time")
    assert kind == "text" and not in_think and ids.tolist() == want
    got = app.build_prompt(body, count="ids")
    assert isinstance(got, list) and got == want and all(isinstance(t, int) for t in got)
    assert app.build_prompt(body, count=True) == len(want)
    assert app.build_prompt(body, count="think") == (len(want), False)
    assert app._shared_head(got, want + [7]) == len(want)
    assert app.build_prompt({"prompt": ["first", "second"]}, count="ids") == QTok().encode("first")


if __name__ == "__main__":
    test_a_text_prompt_answers_every_count()
    print("  test_a_text_prompt_answers_every_count ok\n1 passed")
