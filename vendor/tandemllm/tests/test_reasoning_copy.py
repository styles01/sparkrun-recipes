"""SRV-239 (UX2, 2026-10-07): where an Open WebUI follow-up parts from the rows of an answer that
thought, on the served template and tokenizer (server/endalign.py, THE REASONING).

W02, W03, W04, W15, W16, W17 on two 0.4.0 test builds: the first follow-up of a chat whose first
answer thought resumed at the head's 6,144 (cached 0.71 to 0.89, TTFT 2.5 to 5.0 s); W05 and W06,
whose first turn did not think, resumed at the end (1.00). Open WebUI sends an answer back without
its reasoning, and the Qwen3.8 template (`preserve_thinking` on unless a request turns it off)
renders that message as `<think>\\n\\n</think>\\n\\n` + the answer.

  1  the copy parts from the rows at the prompt's last token (`<think>\\n`: the copy's `\\n\\n` is
     one token), whatever the reasoning was; a turn that did not think does not part at all
  2  the end alignment's second pass serves the rows up to the end anchor and the copy after it:
     the resume is the end anchor, the client's tokens all come from the rows
  3  a copy that carries the reasoning (opencode sends `reasoning_content` back) holds the rows
     token for token up to the reasoning's end, where the template's trim leaves the first pass
     its whitespace work, as before

The tokenizer is the served model's tokenizer.json when this machine has it (the box and the Mac
do); without it the file says so and passes. Run: python tests/test_reasoning_copy.py
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import endalign  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "fixtures", "qwen38_chat_template.jinja")
SYSTEM = "You are a helpful assistant. Answer in the user's language."
USER1 = "Explain how a heat pump works"
REASONING = ("The user wants an explanation of a heat pump. I should cover the refrigerant cycle: "
             "evaporator, compressor, condenser, expansion valve, and why it moves more heat than "
             "it uses. Keep it clear.")
ANSWER = ("A heat pump moves heat instead of making it.\n\n1. **Evaporator**: cold refrigerant takes "
          "heat from the outside air.\n2. **Compressor**: squeezing the gas makes it hot.\n3. "
          "**Condenser**: the hot gas gives its heat to the house.\n4. **Expansion valve**: the "
          "pressure drops and the cycle starts again.")
USER2 = "make it shorter"


def _tok():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    hub = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    found = sorted(glob.glob(os.path.join(hub, "models--Qwen--Qwen3.8-27B/snapshots/*/"
                                               "tokenizer.json")))
    return Tokenizer.from_file(found[0]) if found else None


def _render(messages, gen=True, **kw) -> str:
    import jinja2
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(TEMPLATE)))

    def raise_exception(msg):
        raise ValueError(msg)
    t = env.get_template(os.path.basename(TEMPLATE))
    return t.render(messages=messages, add_generation_prompt=gen,
                    raise_exception=raise_exception, **kw)


class _Res:
    """What EndAlign reads of the resident prefix: its rows, its end anchor, its own replay."""

    def __init__(self, rows, e):
        self.tokens, self.end = list(rows), {"at": e, "hidden": []}

    def replay(self, ids):
        return ids, []


class _TokBytes:
    """`convert_ids_to_tokens` over a `tokenizers.Tokenizer`, for endalign.token_bytes."""

    def __init__(self, tk):
        self.tk = tk

    def convert_ids_to_tokens(self, i):
        return self.tk.id_to_token(int(i))

    def decode(self, ids, skip_special_tokens=False):
        return self.tk.decode(ids, skip_special_tokens=skip_special_tokens)


def _common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _turn(tk, think: bool):
    """(rows of turn 1 up to its end anchor, the first prompt's length, the follow-up's copy)."""
    sys_user = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER1}]
    kw = {} if think else {"enable_thinking": False}
    prompt = tk.encode(_render(sys_user, **kw), add_special_tokens=False).ids
    gen = (REASONING + "\n</think>\n\n" + ANSWER) if think else ANSWER
    out = tk.encode(gen, add_special_tokens=False).ids + [tk.token_to_id("<|im_end|>")]
    rows = prompt + out
    e = len(rows) - 1                          # the last token is decided, never forwarded
    # Open WebUI's follow-up: the answer back without its reasoning, then the new message
    copy = tk.encode(_render(sys_user + [{"role": "assistant", "content": ANSWER},
                                         {"role": "user", "content": USER2}], **kw),
                     add_special_tokens=False).ids
    return rows, e, len(prompt), copy


def test_the_copy_parts_at_the_think_newline_and_the_second_pass_resumes_at_the_end():
    tk = _tok()
    if tk is None:
        print("      (no Qwen3.8 tokenizer.json here -- skipped)")
        return
    rows, e, n1, copy = _turn(tk, think=True)
    c = _common(copy, rows)
    assert c == n1 - 1, (c, n1)                # `<think>` the last shared token; `\n` vs `\n\n`
    assert tk.decode([rows[n1 - 1]], skip_special_tokens=False) == "\n"
    assert tk.decode([copy[n1 - 1]], skip_special_tokens=False) == "\n\n"
    al = endalign.EndAlign(endalign.token_bytes(_TokBytes(tk)), reasoning=False)
    res = _Res(rows, e)
    al.kept(res, n1, "kept")
    assert al.regions(res)[0][2] > n1          # the region knows where its reasoning ends
    out, spans, aligned = al.replay(res, copy)
    assert out == copy and aligned == 0        # Served as sent, resumes below c
    al.reasoning = True
    out, spans, aligned = al.replay(res, copy)
    assert out[:e] == rows[:e] and aligned > 0, (aligned, len(out))
    tail = out[e:]
    assert copy[len(copy) - len(tail):] == tail     # after the end anchor: the copy's own
    grid = (c // 1024) * 1024
    return (f"parts at {c} of {len(copy)}; before: resume <= {c} (grid {grid}), "
            f"after: end anchor {e}, {len(tail)} new tokens, reasoning "
            f"{al.stats['reasoning_tokens']} rows served")


def test_a_turn_that_did_not_think_does_not_part():
    tk = _tok()
    if tk is None:
        print("      (no Qwen3.8 tokenizer.json here -- skipped)")
        return
    rows, e, n1, copy = _turn(tk, think=False)
    assert copy[:e] == rows[:e], _common(copy, rows)      # W05: the end, 1.00


def test_a_copy_with_the_reasoning_holds_the_rows_to_its_end():
    tk = _tok()
    if tk is None:
        print("      (no Qwen3.8 tokenizer.json here -- skipped)")
        return
    rows, e, n1, _ = _turn(tk, think=True)
    sys_user = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER1}]
    copy = tk.encode(_render(sys_user + [
        {"role": "assistant", "content": ANSWER, "reasoning_content": REASONING},
        {"role": "user", "content": USER2}]), add_special_tokens=False).ids
    assert copy[:e] == rows[:e], _common(copy, rows)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:80s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
