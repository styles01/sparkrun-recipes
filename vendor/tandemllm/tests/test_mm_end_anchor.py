"""SRV-546: a request with images resumes at the end anchor, as a text request does.

THE MEASUREMENT (the Chat tab's pagoda run, 2026-10-17, v0.11.2 + Super VL vision, slot 0, no
other request on slot 0). Steps 2 to 22 resumed at the end anchor ("end", 49,026 to 103,899
tokens cached). Step 22 called render_page; step 23 carried its screenshot and was served as
rendered: no replay, no alignment. Its copy parted from the rows at 88,640, in step 13's edit_file
`old` parameter, where the model wrote four spaces of indentation and the server's parser handed
the client the value trimmed (the suffix store's generated ids against chatd's re-rendered copy).
The end anchor (112,642) needs the copy to hold every row up to it, so the prefill resumed at the
largest grid anchor below 88,640: 29,696, the end of the run's first prompt, because an agent's
rows are written by verify blocks and its prefills are a tool result each, so few grid lines
were ever crossed. 84,640 tokens read again in 149 s (568 tok/s). Every later turn kept the
screenshot, and an end kept after images was offered only to a compaction, so each turn resumed
at a grid anchor below its new tokens; at 10:18:36, after the memory policy had shed the anchors,
at 32,768 (90,186 tokens again, 161 s).

THE FIX. `_mm_end_replay` runs the end anchor's replay and alignment on the request's expanded
key ids (the rows' own coordinates) and rebuilds the served prompt from what they return; the
prefill offers an end kept after images to a request whose rows hold every token up to it, the
images' keys included (`end_mm`); `_remember` keeps such an end with its hidden spans and its
regions.

  1  text end, then a request with an image whose copy lacks the end's hidden spans: replayed in
     key space, resumed at the end, the same greedy tokens as the same prompt cold
  2  the same with a copy that differs in whitespace inside the generation (the trimmed
     parameter): aligned in key space, resumed at the end, the same greedy tokens
  3  an end kept after images, then the next request with the same image and more text: resumed
     at the end, the same greedy tokens; the hidden spans and regions carry over
  4  another image in that place: never the end (the keys part at the image)
  5  MM_END off: the second request with an image goes back to the grid, as in 0.11.3; the
     router's in-context stage of a new user turn with images resumes at the end too
  6  the expansion's inverse and positions for both towers' shapes (no marks; `<img>` `</img>`)
  7  an image key id's bytes for the alignment; the words of a re-read

Run: python tests/test_mm_end_anchor.py     (needs transformers with qwen3_5 for 1-5)
"""

from __future__ import annotations

import contextlib
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import test_vision as TV  # noqa: E402  (sets the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import cache  # noqa: E402
from engine import vision as V  # noqa: E402
from engine.penalty import PenaltySpec  # noqa: E402
from engine.spec import Relax  # noqa: E402
from server import app, endalign  # noqa: E402
from test_app_loop import FakeTok  # noqa: E402

CH = 8
BIG = 1 << 30
IMG, VS, VE = TV.IMG, TV.VS, TV.VE
WS = {"id": 7}                             # the token the alignment's bytes call whitespace


def text(n: int, seed: int) -> list[int]:
    """Text ids of the tiny vocabulary: never an image mark, never the whitespace token."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 89, (n,), generator=g).tolist()


def tbytes(i: int) -> bytes:
    if int(i) >= endalign.KEY_BASE:
        return endalign.token_bytes(None)(i)
    return b" " if int(i) == WS["id"] else b"t%d" % int(i)


def serve(mm_end: str = "on", end: bool = True):
    """The server's state around `generate_stream` on the tiny vision-language model, with the
    resident prefix, its end anchor and the whitespace alignment."""
    _, d = TV.reference()
    eng, tower = TV.engine_for(d, max_len=256)
    res = cache.ResidentPrefix(BIG, CH, end=end, stash_bytes=BIG)
    al = endalign.EndAlign(tbytes, on=True, reasoning=False, moved=False)
    app.STATE.clear()
    app.STATE.update(engine=eng, drafter=None, k=4, tree=False, sampled_tree=False,
                     relax=Relax(1.0, 1), verbose=False, state_store=None, prefix_cache=True,
                     prefix_chunk=CH, session_cache=False, tok=FakeTok(), device="cpu",
                     max_len=256, default_max_tokens=32, pen_spec=PenaltySpec(), model="t",
                     cfg_eos=None, think_budget=0, think_stall=False, reasoning_format="tags",
                     request_timeout=0.0, max_queue=8, queue_timeout=5.0, pattern_stop=None,
                     response_cache=None, suffix_store=None, resident=res, disk_cache=None,
                     mem=None, vision=tower, image_cache=None, end_align=al)
    app.MM_END = mm_end
    return res


def cold(full_ids: list[int], images: list, n: int) -> list[int]:
    """The same served prompt on a fresh engine with no cache: the greedy reference."""
    _, d = TV.reference()
    eng, tower = TV.engine_for(d, max_len=256)
    saved = dict(app.STATE)
    app.STATE.update(engine=eng, resident=None, vision=tower, end_align=None)
    try:
        if images:
            full, mm = app._vision_context(full_ids, images, eng)
            out = list(app.generate_stream(torch.tensor(full), n, set(), None, "cold", mm=mm))
        else:
            out = list(app.generate_stream(torch.tensor(full_ids), n, set(), None, "cold"))
    finally:
        app.STATE.clear()
        app.STATE.update(saved)
    return out


def first_turn(n_prompt: int, n: int, seed: int, mm_end: str = "on"):
    """A fresh server and a first text turn whose answer holds no image mark (the random model
    writes any id; a real one never writes an image placeholder)."""
    for k in range(50):
        res = serve(mm_end=mm_end)
        p1 = text(n_prompt, seed + 1000 * k)
        out1, pre1 = text_turn(p1, n)
        if not {IMG, VS, VE} & set(out1):
            return res, p1, out1
    raise AssertionError("no clean first turn")


def text_turn(ids: list[int], n: int, hidden=()):
    out = list(app.generate_stream(torch.tensor(ids), n, set(), None, "ses"))
    pre = dict(app.STATE["last_prefill"])
    app._remember(list(ids), out, "ses", hidden=hidden)
    return out, pre


def image_turn(ids: list[int], images: list, n: int):
    """One request with images as the handler serves it: expanded, the replay in key space
    (`_mm_end_replay`), the prefill with `end_mm`, then `_remember`."""
    res = app.STATE["resident"]
    t = torch.tensor(ids)
    prompt, mm = app._mm_prompt(t, images)
    spans, aligned = [], 0
    if app.MM_END == "on" and res.end is not None:
        prompt, mm, spans, aligned, _ = app._mm_end_replay(res, t, prompt, mm, images)
    kw = {"end_mm": True} if app.MM_END == "on" else {}
    served = list(mm.key_ids)              # the served prompt, image rows as their keys
    out = list(app.generate_stream(prompt, n, set(), None, "ses", mm=mm, **kw))
    pre = dict(app.STATE["last_prefill"])
    app._remember(list(ids), out, "ses", mm=mm, hidden=spans)
    return out, pre, served, mm, aligned


def with_image(ids: list[int]) -> list[int]:
    return list(ids) + [VS, IMG, VE]


def need():
    return TV.need_hf()


# ------------------------------------------------------------------ 1. hidden spans, key space

def test_a_first_screenshot_resumes_at_the_text_end_through_the_replay():
    if need():
        return need()
    x = TV.image(51, (1, 4, 6))
    res, p1, out1 = first_turn(40, 20, 1)
    e1 = res.end["at"]
    # as if the monitor had hidden 3 of the answer's tokens: the client's copy lacks them
    q = len(p1) + 5
    res.end["hidden"] = [(q, out1[5:8])]
    copy = p1 + out1[:5] + out1[8:] + with_image(text(6, 2)) + text(4, 3)
    out2, pre2, served, mm, _ = image_turn(copy, [x], 12)
    assert pre2["kind"] == "end" and pre2["reused"] == e1, (pre2, e1)
    ref = cold(app._mm_collapse(list(mm.key_ids), mm), [x], 12)
    assert out2 == ref, (out2, ref)
    # 0.11.3 served the copy as rendered: it parts at the hidden span and resumes below it
    res0 = serve(mm_end="off")
    assert text_turn(p1, 20)[0] == out1
    res0.end["hidden"] = [(q, out1[5:8])]
    _, pre0, _, _, _ = image_turn(copy, [x], 12)
    assert pre0["reused"] <= q, (pre0, q)
    app.MM_END = "on"
    return (f"resumed at the end ({e1} of {len(served)}); 0.11.3 resumed at {pre0['reused']} "
            f"(below the span at {q}); 12 greedy tokens equal")


# ------------------------------------------------------------------ 2. whitespace, key space

def test_a_copy_trimmed_inside_the_generation_is_aligned_in_key_space():
    if need():
        return need()
    x = TV.image(52, (1, 4, 6))
    res, p1, out1 = first_turn(40, 20, 4)
    e1 = res.end["at"]
    # the rows hold a token the client's copy lacks and the alignment reads as whitespace (step
    # 13's indentation, which the parser trimmed from the edit_file parameter it handed back)
    j = 6
    WS["id"] = out1[j]
    copy = p1 + out1[:j] + out1[j + 1:] + with_image(text(5, 5))
    out2, pre2, served, mm, aligned = image_turn(copy, [x], 10)
    WS["id"] = 7
    assert aligned > 0 and pre2["kind"] == "end" and pre2["reused"] == e1, (pre2, aligned)
    assert served[:e1] == (p1 + out1)[:e1]
    ref = cold(app._mm_collapse(served, mm), [x], 10)
    assert out2 == ref, (out2, ref)
    return f"aligned {aligned} tokens, resumed at the end ({e1}); 10 greedy tokens equal"


# ------------------------------------------------------------------ 3. an end kept after images

def clean(out) -> bool:
    return not {IMG, VS, VE} & set(out)


def _three_turns(mm_end: str, seed: int = 6):
    x = TV.image(53, (1, 4, 6))
    for k in range(20):
        res, p1, out1 = first_turn(30, 16, seed + 7 * k, mm_end)
        c2 = p1 + out1 + with_image(text(6, 7)) + text(3, 8)
        out2, pre2, served2, mm2, _ = image_turn(c2, [x], 16)
        if clean(out2):
            break
    e2 = res.end["at"] if res.end is not None else None
    c3 = c2 + out2 + text(9, 9)
    out3, pre3, served3, mm3, _ = image_turn(c3, [x], 10)
    return res, x, (out2, pre2), (out3, pre3, served3, mm3), e2, k


def test_an_end_kept_after_images_serves_the_next_request_with_them():
    if need():
        return need()
    res, x, (out2, pre2), (out3, pre3, served3, mm3), e2, k = _three_turns("on")
    assert pre2["kind"] == "end", pre2
    assert e2 is not None and pre3["kind"] == "end" and pre3["reused"] == e2, (pre3, e2)
    assert res.end.get("mm") and res.stats["end_hits"] == 2, (res.end, res.stats)
    ref = cold(app._mm_collapse(served3, mm3), [x], 10)
    assert out3 == ref, (out3, ref)
    # 0.11.3: the end after images is a compaction's only; the turn resumes at a grid line
    _, _, _, (out3o, pre3o, _, _), e2o, ko = _three_turns("off")
    assert ko == k
    assert pre3o["kind"] != "end" and pre3o["reused"] < e2o, (pre3o, e2o)
    assert out3o == out3
    app.MM_END = "on"
    return (f"resumed at the end ({e2} of {len(served3)}); 0.11.3: {pre3o['kind']} at "
            f"{pre3o['reused']}; 10 greedy tokens equal")


def test_the_router_stage_of_a_new_user_turn_resumes_at_the_end_after_images():
    """Live (the after-fix run, turn D): a new user turn with a screenshot ran the router's
    in-context stage first; its prefill (`_context_prefill`) still served images as rendered, read
    from a grid line and took the rows, and the turn after it re-read 6,448 tokens."""
    if need():
        return need()
    x = TV.image(57, (1, 4, 6))
    for k in range(20):
        res, p1, out1 = first_turn(30, 16, 20 + 7 * k)
        c2 = p1 + out1 + with_image(text(6, 7)) + text(3, 8)
        out2, pre2, served2, mm2, _ = image_turn(c2, [x], 16)
        if clean(out2):
            break
    e2 = res.end["at"]
    stage = c2 + out2 + text(7, 21)                # the new user message, up to its insertion
    _, reused, fwd = app._context_prefill(list(stage), len(stage), "ses", None, [x])
    assert reused == e2 and res.end is not None and res.end["at"] == e2, (reused, e2)
    out3, pre3, served3, mm3, _ = image_turn(stage + text(4, 22), [x], 8)
    assert pre3["reused"] >= e2, (pre3, e2)
    assert out3 == cold(app._mm_collapse(served3, mm3), [x], 8)
    return f"the stage resumed at the end ({e2}), the turn after it at {pre3['reused']}"


# ------------------------------------------------------------------ 4. another image

def test_another_image_in_its_place_never_takes_the_end():
    if need():
        return need()
    x, y = TV.image(54, (1, 4, 6)), TV.image(55, (1, 4, 6))
    for k in range(20):
        res, p1, out1 = first_turn(30, 12, 10 + 7 * k)
        c2 = p1 + out1 + with_image(text(4, 11))
        out2, pre2, served2, mm2, _ = image_turn(c2, [x], 12)
        if clean(out2):
            break
    e2 = res.end["at"]
    img_at = mm2.spans[0].start
    c3 = c2 + out2 + text(5, 12)
    out3, pre3, served3, mm3, _ = image_turn(c3, [y], 8)
    assert pre3["kind"] != "end" and pre3["reused"] <= img_at, (pre3, img_at, e2)
    ref = cold(app._mm_collapse(served3, mm3), [y], 8)
    assert out3 == ref
    return f"other image: resumed at {pre3['reused']} (image at {img_at}, end at {e2})"


# ------------------------------------------------------------------ 6. the expansion's inverse

def _ctx(full, spans, n_text, ph):
    return SimpleNamespace(n=len(full), spans=spans, n_text=n_text, placeholder=ph,
                           key_ids=list(full))


def test_the_expansion_comes_back_to_the_template_text_for_both_towers():
    img = TV.image(56, (1, 4, 6))
    kid = V.key_id(img.digest)
    ids = [5, 6, VS, IMG, VE, 7, 8, VS, IMG, VE, 9]
    # the Qwen towers: the placeholder's rows, no marks
    full, spans = V.expand(ids, [img, img], IMG, 2)
    keys = list(full)
    for s in spans:
        keys[s.start:s.start + s.n] = [kid] * s.n
    ctx = _ctx(full, spans, len(ids), IMG)
    assert app._mm_collapse(keys, ctx) == ids
    assert [app._mm_text_pos(ctx, p) for p in (0, 3, 4, 6, 10)] == \
        [0, 3, 3 + spans[0].n, 5 + spans[0].n, 8 + spans[0].n + spans[1].n]
    assert app._mm_runs(keys) == [kid, kid]
    # Super VL: `<img>` rows `</img>` for each placeholder
    from engine.models.nemotron_h import vision as NV
    cfg = SimpleNamespace(image_token_id=IMG, image_start_id=94, image_end_id=95, downsample=2)
    full2, spans2 = NV.expand(ids, [img, img], cfg)
    keys2 = list(full2)
    for s in spans2:
        keys2[s.start:s.start + s.n] = [kid] * s.n
    ctx2 = _ctx(full2, spans2, len(ids), IMG)
    assert app._mm_collapse(keys2, ctx2) == ids
    assert app._mm_text_pos(ctx2, 10) == 10 + 2 * (spans2[0].n + 1)
    return f"Qwen {len(full)} rows, Super VL {len(full2)} rows -> {len(ids)} text ids"


# ------------------------------------------------------------------ 7. bytes and words

def test_an_image_key_is_content_for_the_alignment():
    tb = endalign.token_bytes(SimpleNamespace(convert_ids_to_tokens=lambda i: None,
                                              decode=lambda ids, **k: ""))
    a, b = V.KEY_BASE + 5, V.KEY_BASE + 6
    assert tb(a) and endalign._bare(tb(a)) == tb(a) and tb(a) != tb(b) and tb(a) == tb(a)
    return "an image key's bytes: no whitespace, its own"


# ------------------------------------------------------------------ 8. a side slot's conversation

#: ENG-371's image tests: a resume's logits against a cold prefill's (the rows below the end were
#: written by the decode's forwards: a few 1e-6 here; a resume on wrong rows is 1e-2 and more off)
TOL = 1e-4


@contextlib.contextmanager
def prefill_logits():
    """The logits of every `cache.prefill` in the block (the request's own, as
    `generate_stream` runs it)."""
    got, real = [], cache.prefill

    def spy(*a, **k):
        out = real(*a, **k)
        got.append(out[0].detach().float().clone())
        return out
    cache.prefill = spy
    try:
        yield got
    finally:
        cache.prefill = real


def need_or_skip():
    """`need()`, a real skip under pytest (a returned string counts as a pass there)."""
    why = need()
    if why and "pytest" in sys.modules:
        import pytest
        pytest.skip(why)
    return why


def assert_cold_mm(lg, mm, images, n, out):
    """The image request's resume is the cold prefill of its served prompt: logits within TOL,
    the same greedy tokens."""
    with prefill_logits() as got:
        ref = cold(app._mm_collapse(list(mm.key_ids), mm), images, n)
    d = float((lg - got[-1]).abs().max())
    assert d <= TOL, f"logits {d:.3g} off the cold prefill's"
    assert out == ref, (out, ref)
    return d


def _side_turn(p1, n):
    """`serve` with a side slot beside slot 0 (its own engine on the same weights) and a text
    turn on it; the carry record its end leaves."""
    from server import slots as S
    res = serve()
    _, d = TV.reference()
    seng, _ = TV.engine_for(d, max_len=256)
    vals = {k: None for k in S.OVERLAY}
    vals.update(engine=seng, drafter=None, max_len=256)
    side = S.Slot(1, 256, values=vals)
    pool = S.Pool([S.Slot(0, 256), side])
    app.STATE["pool"] = pool
    s = pool.take_side(len(p1) + n, owner="ses")
    S.bind(s)
    try:
        out1, _ = text_turn(p1, n)
    finally:
        S.bind(None)
        pool.release_side(s)
    return res, side, out1


def _carried_image_turn(ids, images, n, carry=True):
    """The handler's order for a request with images on slot 0 (ENG-371): expanded, the side
    slot's conversation carried in on the key ids, the replay in key space, the prefill."""
    res = app.STATE["resident"]
    t = torch.tensor(ids)
    prompt, mm = app._mm_prompt(t, images)
    if carry:
        with torch.no_grad():
            app._carry_from_side(list(mm.key_ids), "ses")
    if res.end is not None:
        prompt, mm, _, _, _ = app._mm_end_replay(res, t, prompt, mm, images)
    served = list(mm.key_ids)
    with prefill_logits() as got:
        out = list(app.generate_stream(prompt, n, set(), None, "ses", mm=mm, end_mm=True))
    pre = dict(app.STATE["last_prefill"])
    pre["logits"] = got[-1]
    return out, pre, served, mm


def test_a_side_slots_conversation_with_its_first_screenshot_is_carried_into_slot_0():
    """ENG-371: a conversation that ran on a side slot (text only) gets its first screenshot and
    must run on slot 0: the side slot's rows are carried in on the request's key ids, as a text
    turn's are, and it resumes at the side slot's end. Before, a request with images skipped the
    carry and read its prompt again (live 2026-10-17 23:41: a parent's turn with a screenshot
    re-read 14,606 of 19,726 tokens)."""
    if need_or_skip():
        return need()
    x = TV.image(53, (1, 4, 6))
    for k in range(50):
        p1 = text(40, 7 + 1000 * k)
        res, side, out1 = _side_turn(p1, 20)
        if not {IMG, VS, VE} & set(out1):
            break
    rec = side.carry
    assert rec is not None and rec["at"] == len(p1) + len(out1) - 1, rec
    copy = p1 + out1 + with_image(text(6, 4)) + text(4, 5)
    out2, pre2, served, mm = _carried_image_turn(copy, [x], 12)
    assert pre2["reused"] == rec["at"] and pre2["forwarded"] == len(served) - rec["at"], pre2
    assert res.stats["side_carries"] == 1 and side.carry is None, res.stats
    d = assert_cold_mm(pre2.pop("logits"), mm, [x], 12, out2)
    # without the carry (0.11.4): slot 0 holds nothing of it, the prompt is read again
    _side_turn(p1, 20)
    _, pre0, _, _ = _carried_image_turn(copy, [x], 12, carry=False)
    assert pre0["reused"] == 0, pre0["reused"]
    return (f"resumed at the side slot's end ({rec['at']} of {len(served)}), logits {d:.1g} off "
            f"cold; without the carry {pre0['forwarded']} forwarded")


def _turn_as(session, ids, n, images=None):
    """A text turn, or one with images as `image_turn` serves it, of `session`."""
    res = app.STATE["resident"]
    if not images:
        out = list(app.generate_stream(torch.tensor(ids), n, set(), None, session))
        pre = dict(app.STATE["last_prefill"])
        app._remember(list(ids), out, session)
        return out, pre, None
    t = torch.tensor(ids)
    prompt, mm = app._mm_prompt(t, images)
    app._spill_before_replay(list(mm.key_ids), mm_ok=True)
    if res.end is not None:
        prompt, mm, _, _, _ = app._mm_end_replay(res, t, prompt, mm, images)
    with prefill_logits() as got:
        out = list(app.generate_stream(prompt, n, set(), None, session, mm=mm, end_mm=True))
    pre = dict(app.STATE["last_prefill"])
    pre["logits"] = got[-1]
    app._remember(list(ids), out, session, mm=mm)
    return out, pre, mm


def _two_with_images(old_park: bool):
    """A conversation with a screenshot, another session on slot 0 in between (A is parked with
    its end kept after the image), then A's next request with the same image."""
    x = TV.image(58, (1, 4, 6))
    for k in range(30):
        res = serve()
        res.park_budget = BIG
        if old_park:                       # the park as before ENG-371: an end after images unoffered
            real = res.park_offer
            res.park_offer = lambda ids, c, m, align=True, mm_ok=False: real(ids, c, m, align)
        a1 = text(30, 30 + 1000 * k)
        oa1, _, _ = _turn_as("A", a1, 12)
        a2 = a1 + oa1 + with_image(text(5, 31)) + text(3, 32)
        oa2, _, mm2 = _turn_as("A", a2, 12, [x])
        if not ({IMG, VS, VE} & set(oa1 + oa2)):
            break
    e2 = res.end["at"]
    assert res.end.get("mm"), res.end
    _turn_as("B", text(40, 33), 8)                     # B takes slot 0: A goes to the park
    assert res.park_owner == "A" and res.park_end is not None and res.park_end.get("mm")
    # a worker's rows carried in from a side slot, or written by verify blocks, hold no grid
    # anchor the park could offer instead (the live case)
    for b in list(res.park_anchors):
        res._free.append(res.park_anchors.pop(b))
    a3 = a2 + oa2 + text(6, 34)
    oa3, pre3, mm3 = _turn_as("A", a3, 10, [x])
    return res, e2, oa3, pre3, mm3, x


def test_a_parked_end_after_images_serves_its_next_request_with_them():
    """ENG-371: two workers with screenshots take turns on slot 0 (side slots read no images), and
    each parks the other. A parked end kept after images is offered to a request whose keys hold
    every row up to it, as the end in place is (live 2026-10-18 00:58: a worker's call read
    16,161 tokens again from 0 although its rows were in the park)."""
    if need_or_skip():
        return need()
    res, e2, out3, pre3, mm3, x = _two_with_images(old_park=False)
    assert pre3["reused"] == e2 and res.stats["park_end_hits"] >= 1, (pre3["reused"], e2)
    d = assert_cold_mm(pre3.pop("logits"), mm3, [x], 10, out3)
    _, e2o, out3o, pre3o, _, _ = _two_with_images(old_park=True)
    assert pre3o["reused"] < e2o and out3o == out3, (pre3o["reused"], e2o)
    return (f"resumed at the parked end ({e2} of {len(mm3.key_ids)}), logits {d:.1g} off cold; "
            f"the park as before: {pre3o['reused']}")


def test_a_long_re_read_says_why():
    res = cache.ResidentPrefix(BIG, 1024, end=True)
    res.valid, res.tokens = 100_000, [1] * 100_000
    res.end = {"at": 100_000, "hidden": [], "mark": None}
    w = cache.reread_words(res, 88_640, 29_696, 114_336, "resident", 100_000)
    assert "88,640 of 100,000" in w, w
    res.end, res.end_lost = None, "memory"
    assert "free memory" in cache.reread_words(res, 99_000, 29_696, 114_336, "resident", 100_000)
    assert "nothing of this chat" in cache.reread_words(res, 0, 0, 50_000, None, 0)
    return w


def test_a_long_prompt_with_images_the_end_replay_did_not_serve_says_why():
    """SRV-657 (live 2026-10-21 00:05:59, Ornith, chunk 4,096): a request with images re-read
    43,856 tokens from 4,096, its copy parted from the rows at 7,175, and the log said nothing:
    `_say_reread` stayed silent under two chunks (8,192), its guard against another
    conversation's head. A request the end replay did not serve says so from 2,048 on, with the
    replay's reason; a text request keeps the two-chunk guard."""
    import io
    from contextlib import redirect_stdout

    class Res:
        chunk, end = 4096, {"at": 20_000}
        tokens = list(range(20_000))
    copy = list(range(7_175)) + [-1] * 13_000
    out = io.StringIO()
    with redirect_stdout(out):
        app._say_reread("chatcmpl-x", Res(), copy, None, why_mm="the images differ from the "
                        "rows' (ones or order)")
        app._say_reread("chatcmpl-y", Res(), copy, None)
    log = out.getvalue()
    assert "chatcmpl-x re-read ahead: the copy (20175 tokens) parts from the end anchor's rows " \
           "at token 7175 of 20000" in log and "with images not served: the images differ" in log, log
    assert "chatcmpl-y" not in log, log
    return "with images: said from 2,048 shared tokens with the reason; text: the 8,192 guard"


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:80s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
