"""The end anchor for a copy whose whitespace is not the rows' (server/endalign.py).

THE MEASUREMENT (the box's suffix store, 2026-10-03, the last 4,000 documents, token ids only).
Of the follow-ups whose copy parted inside the previous generation, 220 parted in a tool call's
parameter and 34 at `</think>`; 221 of those 254 differ from what the engine wrote in whitespace
alone: server/toolcall.py trims a parameter value (an `oldString`'s indentation, the newline the
model wrote before `</parameter>`), and the template writes `reasoning|trim + "\\n</think>"`.
Such a turn resumed at the previous prompt's last 1,024 line and re-read the whole generation.

  1  a newline more in the copy (the template's before `</think>`), a whitespace token less (the
     parser's trim) and a token the parser trimmed into another one (`"   x"` -> `"x"`): the copy
     resumes at the end anchor, served with the rows' tokens, and writes the same greedy tokens
     as a cold prefill of the served prompt (the end anchor's gate)
  2  anything but whitespace, a difference inside the prompt (not a generation), a copy that
     stops at the end anchor: served as the client sent it, the prompt's line as before
  3  off: `ResidentPrefix.replay` alone, as before
  4  the regions travel: the turn after an aligned turn is aligned again, its own generation too
  5  a hidden span and a whitespace difference in one generation
  6  the bytes of a byte-level BPE tokenizer's tokens
  7  through the handler: `usage.prompt_tokens` is the client's count, the journal's resume `end`
  8  SRV-239: a copy that leaves an answer's reasoning out (Open WebUI sends none back; the
     template renders `<think>\n\n</think>`) resumes at the end anchor, served with the rows'
     reasoning; two turns of it in a row; anything else in the answer changed is served as sent;
     `reasoning` off (QWEN38_END_ALIGN_REASONING=off) refuses it as before

Run: python tests/test_end_align.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from server import app, endalign  # noqa: E402
from test_cache import tokens  # noqa: E402
from test_end_anchor import (ChatTok, PHRASE, TOOL, TOOL2, _chat, _hidden_turn,  # noqa: E402
                             _state, turn)

NL, SP = 95, 96                                   # whitespace-only tokens in these tests


def tb_of(over=None):
    """Every id is its own non-whitespace bytes, but NL and SP, and `over`'s."""
    over = dict(over or {})

    def tb(i):
        i = int(i)
        if i in over:
            return over[i]
        return b"\n" if i == NL else b"  " if i == SP else b"<%d>" % i
    return tb


def _setup(tb=None, on=True):
    res = _state()
    al = app.STATE["end_align"] = endalign.EndAlign(tb or tb_of(), on=on)
    return res, al


def _first(prompt, n=60):
    out, pre, ctx = turn(prompt, n)
    return out, ctx


def _cold(ids, n):
    """The same served prompt from a cold start, no anchor: what the gate compares with."""
    _state(end=False)
    out, pre, _ = turn(ids, n)
    assert pre["reused"] == 0 or pre["kind"] != "end", pre
    return out


# ------------------------------------------------------------------ 1. whitespace is aligned

def test_a_newline_more_in_the_copy_resumes_at_the_end():
    res, al = _setup()
    prompt = tokens(100, seed=70)
    out, ctx = _first(prompt)
    e = res.end["at"]
    k = 30
    copy = prompt + out[:k] + [NL] + out[k:] + TOOL          # the template's newline
    ids, spans, aligned = app._end_replay(res, copy)
    assert ids == prompt + out + TOOL and ids[:e] == res.tokens[:e], (len(ids), e)
    assert aligned == e - len(prompt) and spans == [], aligned
    got, pre, _ = turn(ids, 12)
    assert pre["kind"] == "end" and pre["reused"] == e, pre
    assert got == _cold(ids, 12), "the end anchor's gate: the same greedy tokens as cold"


def test_a_whitespace_token_the_copy_dropped_is_aligned():
    prompt = tokens(100, seed=71)
    res, al = _setup()
    out, ctx = _first(prompt)
    k = 20
    # the rows' token at k is whitespace (an indentation the parser trimmed)
    al.tbytes = tb_of({out[k]: b"    "})
    copy = prompt + out[:k] + out[k + 1:] + TOOL
    ids, spans, aligned = app._end_replay(res, copy)
    e = res.end["at"]
    assert ids[:e] == res.tokens[:e] and ids[e:] == copy[e - 1:], (len(ids), e)
    _, pre, _ = turn(ids, 8)
    assert pre["kind"] == "end" and pre["reused"] == e, pre


def test_a_token_the_parser_trimmed_into_another_is_aligned():
    """`"\\n      x"` as the model wrote it, `"\\nx"` in the copy: another token, the same text
    once the whitespace is out."""
    prompt = tokens(100, seed=72)
    res, al = _setup()
    out, ctx = _first(prompt)
    k = 25
    x = out[k]
    al.tbytes = tb_of({x: b"      <%d>" % x, SP: b"<%d>" % x})
    copy = prompt + out[:k] + [SP] + out[k + 1:] + TOOL
    ids, _, aligned = app._end_replay(res, copy)
    e = res.end["at"]
    assert ids[:e] == res.tokens[:e] and aligned > 0
    _, pre, _ = turn(ids, 8)
    assert pre["kind"] == "end" and pre["reused"] == e, pre


# ------------------------------------------------------------------ 2. anything else as sent

def test_anything_but_whitespace_is_served_as_sent():
    prompt = tokens(100, seed=73)
    res, al = _setup()
    out, ctx = _first(prompt)
    k = next(i for i in range(30, len(out) - 2) if out[i] not in (NL, SP))
    other = next(t for t in range(1, 95) if t != out[k])
    for copy in (prompt + out[:k] + [other] + out[k + 1:] + TOOL,      # a character changed
                 prompt + out[:k] + [other] + out[k:] + TOOL,          # one more
                 prompt + out[:k] + out[k + 1:] + TOOL):               # one less
        assert app._end_replay(res, copy) == (copy, [], 0), k
    # the copy must go past the end anchor
    short = prompt + out[:k] + [NL] + out[k:-1]
    assert app._end_replay(res, short)[2] == 0
    _, pre, _ = turn(prompt + out[:k] + [other] + out[k + 1:] + TOOL, 4)
    assert pre["kind"] == "resident", pre


def test_whitespace_in_the_prompt_is_the_clients():
    """Only a generation the engine wrote is aligned: a newline more in the prompt's part is the
    client's text."""
    prompt = tokens(100, seed=74)
    res, al = _setup()
    out, ctx = _first(prompt)
    copy = prompt[:50] + [NL] + prompt[50:] + out + TOOL
    assert app._end_replay(res, copy) == (copy, [], 0)


# ------------------------------------------------------------------ 3. off

def test_off_is_the_hidden_spans_replay_alone():
    prompt = tokens(100, seed=75)
    res, al = _setup(on=False)
    out, ctx = _first(prompt)
    copy = prompt + out[:30] + [NL] + out[30:] + TOOL
    assert app._end_replay(res, copy) == (copy, [], 0)
    assert res.replay(copy) == (copy, [])
    assert al.regions(res) == []
    _, pre, _ = turn(copy, 4)
    assert pre["kind"] == "resident", pre
    # without an aligner at all: the server as before
    app.STATE.pop("end_align")
    assert app._end_replay(res, copy) == (copy, [], 0)


# ------------------------------------------------------------------ 4. the regions travel

def test_the_turn_after_an_aligned_turn_is_aligned_again():
    prompt = tokens(100, seed=76)
    res, al = _setup()
    out1, ctx1 = _first(prompt, 40)
    e1 = res.end["at"]
    copy2 = prompt + out1[:10] + [NL] + out1[10:] + TOOL
    ids2, _, a2 = app._end_replay(res, copy2)
    assert a2 and ids2[:e1] == res.tokens[:e1]
    out2, pre2, ctx2 = turn(ids2, 30)
    assert pre2["kind"] == "end"
    e2 = res.end["at"]
    assert [r[:2] for r in al.regions(res)] == [(len(prompt), e1), (len(ids2), e2)], \
        al.regions(res)
    # turn 3: the client re-renders turn 1 the same way, and turn 2 with a newline of its own
    copy3 = copy2 + out2[:5] + [NL] + out2[5:] + TOOL2
    ids3, _, a3 = app._end_replay(res, copy3)
    assert a3 and ids3[:e2] == res.tokens[:e2], (a3, len(ids3), e2)
    _, pre3, _ = turn(ids3, 4)
    assert pre3["kind"] == "end" and pre3["reused"] == e2, pre3


def test_a_later_generation_the_copy_rewrote_resumes_at_the_highest_aligned_anchor():
    """SRV-657 (2026-10-21 02:26, Ornith): every turn aligned an early answer's whitespace, until
    one generation's tool call came back with a character changed (not whitespace): the whole
    alignment was refused and the turn resumed at the grid line below the FIRST difference,
    132,635 of 144,923 tokens read again. Now the rows are served up to the highest resident
    anchor inside the part the copy holds token for token past the aligned regions, and the turn
    resumes there; the regions below it travel on."""
    prompt = tokens(100, seed=79)
    res, al = _setup()
    out1, _ = _first(prompt, 40)
    e1 = res.end["at"]
    copy2 = prompt + out1[:10] + [NL] + out1[10:] + TOOL
    ids2, _, a2 = app._end_replay(res, copy2)
    assert a2 and ids2[:e1] == res.tokens[:e1]
    out2, pre2, _ = turn(ids2, 30)
    assert pre2["kind"] == "end"
    line = max(b for b in res.anchors if b <= len(ids2))
    assert line > e1, (sorted(res.anchors), e1)       # turn 2's prefill crossed a grid line
    # turn 3: turn 1 re-rendered the same way, turn 2's answer with a character changed
    k = next(i for i in range(5, len(out2) - 2) if out2[i] not in (NL, SP))
    other = next(t for t in range(1, 95) if t != out2[k])
    copy3 = copy2 + out2[:k] + [other] + out2[k + 1:] + TOOL2
    ids3, spans, a3 = app._end_replay(res, copy3)
    assert al.last_partial == line, (al.last_partial, line)
    assert ids3[:line] == res.tokens[:line] and ids3[line:] == copy3[line + 1:], (len(ids3), line)
    assert a3 == line - (len(prompt) + 10) and al.stats["partial"] == 1, (a3, al.stats)
    got, pre3, _ = turn(ids3, 6)
    assert pre3["kind"] == "resident" and pre3["reused"] == line, pre3
    assert got == _cold(ids3, 6), "the anchor's gate: the same greedy tokens as cold"
    return f"resumed at {line} of {len(ids3)} (the whole alignment refused; before: at 96)"


# ------------------------------------------------------------------ 5. with a hidden span

def test_a_hidden_span_and_whitespace_in_one_generation():
    prompt = tokens(100, seed=77)
    res, al = _setup()
    out, ctx = _hidden_turn(prompt)                   # the phrase at len(prompt) + 31
    q = len(prompt) + 31
    assert res.end["hidden"] == [(q, PHRASE)]
    # the replay alone handles the copy without the phrase
    ids, spans, a = app._end_replay(res, prompt + out + TOOL)
    assert spans == [(q, PHRASE)] and a == 0
    # a newline more after the phrase's place as well: the replay alone parts, the alignment not
    k = 40
    copy = prompt + out[:k] + [NL] + out[k:] + TOOL
    assert res.replay(copy) == (copy, [])
    ids, spans, a = app._end_replay(res, copy)
    e = res.end["at"]
    assert a and spans == [(q, PHRASE)] and ids[:e] == res.tokens[:e], (a, spans)
    _, pre, _ = turn(ids, 6, hidden=spans)
    assert pre["kind"] == "end" and pre["reused"] == e, pre


# ------------------------------------------------------------------ 6. the bytes

def test_the_bytes_of_a_byte_level_tokenizer():
    class Tok:
        def convert_ids_to_tokens(self, i):
            return {1: "Ġfoo", 2: "Ċ", 3: "<|im_end|>", 4: "ĠĠĠ", 5: "é"}.get(i)

        def decode(self, ids, skip_special_tokens=False):
            return "?"
    tb = endalign.token_bytes(Tok())
    assert tb(1) == b" foo" and tb(2) == b"\n" and tb(3) == b"<|im_end|>" and tb(4) == b"   "
    assert tb(5) == "é".encode("latin-1") and tb(9) == b"?"
    assert endalign._bare(b" a\n\tb\r ") == b"ab"


# ------------------------------------------------------------------ 7. through the handler

def test_through_the_handler_usage_is_the_clients_and_the_resume_the_end():
    res = _state()
    tok = app.STATE["tok"] = ChatTok()
    # ChatTok's ids are characters offset by 0x100: id NL is a character of its own, whitespace
    # here by the aligner's table
    app.STATE["end_align"] = endalign.EndAlign(tb_of())
    user = tok.decode(tokens(90, seed=81))
    _chat([{"role": "user", "content": user}])
    n1 = 93
    out = list(app.STATE["last_ctx"][n1:])
    e = res.end["at"]
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": tok.decode(out[:10] + [NL] + out[10:])},
            {"role": "user", "content": tok.decode(TOOL)}]
    sent = len(tok.apply_chat_template(msgs)["input_ids"][0])
    r2 = _chat(msgs)
    last = app.STATE["last_prefill"]
    assert last["kind"] == "end" and last["reused"] == e, last
    assert r2["usage"]["prompt_tokens"] == sent, (r2["usage"], sent)
    assert app.cache_stats()["end_align"]["aligned"] == 1


# ------------------------------------------------------------------ 8. the reasoning left out

def _thinking(prompt, n=60):
    """A first turn whose generation holds a `</think>`: a token of the greedy output no earlier
    token repeats is taken as the tag (the random model writes no real one). Returns (res, al,
    out, k) with the tag at out[k]."""
    res, al = _setup()
    out, _ = _first(prompt, n)
    k = next(i for i in range(12, 30) if out[i] not in out[:i] and out[i] not in prompt
             and out[i] not in (NL, SP))
    res, al = _setup(tb_of({out[k]: endalign.THINK_END}))
    again, _ = _first(prompt, n)
    assert again == out
    return res, al, out, k


def test_srv239_a_copy_without_the_reasoning_resumes_at_the_end():
    """W02 (UX2): 'Explain how a heat pump works' thought, the follow-up resumed at
    the head's 6,144 instead of the end. The prompt ends `<think>\n` (its last token whitespace),
    the copy writes `<think>\n\n</think>\n\n` + the answer: one whitespace token where the rows
    have the newline and the reasoning, then the tag and the answer as the rows hold them."""
    prompt = tokens(99, seed=90) + [NL]
    res, al, out, k = _thinking(prompt)
    e = res.end["at"]
    assert al.regions(res) == [(len(prompt), e, len(prompt) + k)], al.regions(res)
    copy = prompt[:-1] + [SP] + out[k:] + TOOL
    ids, spans, aligned = app._end_replay(res, copy)
    # the rows' tokens up to the end anchor (the reasoning included), then the copy's own: the
    # generation's last token (decided, never forwarded) and the new message
    assert ids == res.tokens[:e] + out[e - len(prompt):] + TOOL, (len(ids), e)
    assert spans == [] and aligned > 0
    assert al.stats["aligned_reasoning"] == 1 and al.stats["reasoning_tokens"] == k
    got, pre, _ = turn(ids, 12)
    assert pre["kind"] == "end" and pre["reused"] == e, pre
    assert got == _cold(ids, 12), "the end anchor's gate: the same greedy tokens as cold"


def test_srv239_off_and_a_changed_answer_are_served_as_sent():
    prompt = tokens(99, seed=91) + [NL]
    res, al, out, k = _thinking(prompt)
    copy = prompt[:-1] + [SP] + out[k:] + TOOL
    al.reasoning = False                       # QWEN38_END_ALIGN_REASONING=off: a 0.4.0 test build
    assert app._end_replay(res, copy) == (copy, [], 0)
    al.reasoning = True
    j = next(i for i in range(k + 2, len(out) - 2) if out[i] not in (NL, SP))
    other = next(t for t in range(1, 95) if t not in (out[j], out[k]))
    changed = prompt[:-1] + [SP] + out[k:j] + [other] + out[j + 1:] + TOOL
    assert app._end_replay(res, changed) == (changed, [], 0)
    # a copy WITH the reasoning (opencode sends it back) aligns as before, on the first pass
    assert app._end_replay(res, prompt + out + TOOL)[0] == prompt + out + TOOL
    assert al.stats["aligned_reasoning"] == 0


def test_srv239_two_thinking_turns_in_a_row():
    """The second follow-up's copy leaves out both answers' reasoning: the first answer's region
    travels with the end anchor, its reasoning end with it."""
    prompt = tokens(99, seed=92) + [NL]
    res, al, out1, k1 = _thinking(prompt, 50)
    e1 = res.end["at"]
    copy2 = prompt[:-1] + [SP] + out1[k1:] + TOOL + [NL]
    ids2, _, a2 = app._end_replay(res, copy2)
    assert a2 and ids2 == res.tokens[:e1] + out1[e1 - len(prompt):] + TOOL + [NL]
    out2, pre2, _ = turn(ids2, 40)
    assert pre2["kind"] == "end"
    # the second tag: a token neither the first answer's reasoning nor this one's start holds,
    # and the first tag not before it (each region's first `</think>` is its own)
    k2 = next(i for i in range(4, 35) if out2[i] not in out2[:i] and out2[i] not in out1[:k1 + 1]
              and out2[i] not in (NL, SP) and out1[k1] not in out2[:i])
    over = {out1[k1]: endalign.THINK_END, out2[k2]: endalign.THINK_END}
    al.tbytes = tb_of(over)
    e2 = res.end["at"]
    al._regions = [(a, b, al._think_end(res.tokens, a, b)) for a, b, _ in al._regions]
    assert [r[2] for r in al.regions(res)] == [len(prompt) + k1, len(ids2) + k2], al.regions(res)
    copy3 = copy2[:-1] + [SP] + out2[k2:] + TOOL2
    ids3, _, a3 = app._end_replay(res, copy3)
    assert a3 and ids3 == res.tokens[:e2] + out2[e2 - len(ids2):] + TOOL2, (len(ids3), e2)
    _, pre3, _ = turn(ids3, 4)
    assert pre3["kind"] == "end" and pre3["reused"] == e2, pre3


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with torch.no_grad():
                print(f"  {name:80s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
