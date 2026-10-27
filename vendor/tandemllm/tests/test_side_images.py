"""SRV-629: a request with images on a side slot.

THE CASE (live 2026-10-19, two Chat-tab agent runs). Both called render_page, and every later
request carried 3 to 8 screenshots. `_par_side_need` sent every request with images to slot 0,
so after the first carry into slot 0 all 31 requests of both chats ran there: 56.6 min of wall
time against 18.9 min of decoding, slot 1 idle, NVMe swaps of 1.7 to 3.0 s each.

What makes a side slot able to serve images: each side slot is its own engine over the same
weights, and `Engine.mm`, `pos_delta` and the rotary rows are the engine's own; the vision tower
runs inside a prefill chunk. Checked here, on the tiny vision-language model of
tests/test_vision.py:

  1  an image request on a side slot's engine decodes the tokens it decodes on slot 0's (and the
     reference's), while slot 0's engine holds a text conversation that it never touches
  2  the same through `generate_stream` with the slot bound: the same greedy tokens, and slot 0's
     engine never sees the request's images

Run: python tests/test_side_images.py     (needs transformers with qwen3_5)
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import test_vision as TV  # noqa: E402  (sets the CPU environment before the engine imports)

import torch  # noqa: E402

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **k: None

import test_mm_end_anchor as MA  # noqa: E402
from engine import vision as V  # noqa: E402
from server import app  # noqa: E402
from server import slots as S  # noqa: E402

IMG, VS, VE = TV.IMG, TV.VS, TV.VE


def need_or_skip():
    return MA.need_or_skip()


def _pair(max_len: int = 256):
    """Slot 0's engine and a side slot's over ONE copy of the weights, as `_build_side_slots`
    builds them, and the tower they share."""
    from engine.config import load_config
    from engine.loader import Weights
    from engine.models.qwen3_5.adapter import Qwen38Engine
    _, d = TV.reference()
    cfg = load_config(d)
    w = Weights(d, device="cpu", skip_mtp=True, nvfp4="", fp8_head="")
    engs = []
    for _ in range(2):
        e = Qwen38Engine(cfg, w, max_len=max_len, device="cpu")
        e.kv.k = e.kv.k.to(TV.DT)
        e.kv.v = e.kv.v.to(TV.DT)
        e.state.conv = e.state.conv.to(TV.DT)
        engs.append(e)
    tower = V.VisionTower.from_checkpoint(d, device="cpu", dtype=TV.DT)
    return engs[0], engs[1], tower


def _snap(eng):
    n = eng.kv.length
    return (eng.state.S.clone(), eng.state.conv.clone(), eng.kv.k[..., :n, :].clone(),
            eng.kv.v[..., :n, :].clone(), n)


def _same_snap(a, b) -> bool:
    return a[4] == b[4] and all(torch.equal(x, y) for x, y in zip(a[:4], b[:4]))


def test_an_image_request_on_a_side_slots_engine_decodes_what_slot_0s_decodes():
    if need_or_skip():
        return need_or_skip()
    model, _ = TV.reference()
    ims = [TV.image(7, (1, 4, 6)), TV.image(8, (1, 8, 4))]
    ids1 = TV.prompt(ims)
    n = 12
    # slot 0 alone: the image request on slot 0's engine
    e0, _, tower = _pair()
    full, mm = TV.mm_context(e0, tower, ids1, ims)
    e0.mm, e0.pos_delta = mm, mm.delta
    alone = []
    with torch.no_grad():
        lg = e0.forward(torch.tensor(full), start=0, last_only=True)[0, -1]
        for i in range(n):
            t = int(lg.argmax())
            alone.append(t)
            lg = e0.forward(torch.tensor([t]), start=len(full) + i)[0, -1]
    x, mmt, pv, grid = TV.hf_inputs(full, ims)
    with torch.no_grad():
        ref = model.generate(input_ids=x, pixel_values=pv, image_grid_thw=grid,
                             mm_token_type_ids=mmt, attention_mask=torch.ones_like(x),
                             max_new_tokens=n, do_sample=False, eos_token_id=None,
                             pad_token_id=0)[0, len(full):].tolist()
    assert alone == ref, (alone, ref)
    # the side slot: slot 0 decodes a text conversation, a step of each in turn
    s0, s1, tower = _pair()
    text = MA.text(30, 3)
    with torch.no_grad():
        lg0 = s0.forward(torch.tensor(text), start=0, last_only=True)[0, -1]
    t0_alone = []
    r0, _, _ = _pair()
    with torch.no_grad():
        lr = r0.forward(torch.tensor(text), start=0, last_only=True)[0, -1]
        for i in range(n):
            t = int(lr.argmax())
            t0_alone.append(t)
            lr = r0.forward(torch.tensor([t]), start=len(text) + i)[0, -1]
    full1, mm1 = TV.mm_context(s1, tower, ids1, ims)
    s1.mm, s1.pos_delta = mm1, mm1.delta
    side, t0 = [], []
    with torch.no_grad():
        lg1 = s1.forward(torch.tensor(full1), start=0, last_only=True)[0, -1]
        for i in range(n):
            t = int(lg1.argmax())
            side.append(t)
            lg1 = s1.forward(torch.tensor([t]), start=len(full1) + i)[0, -1]
            u = int(lg0.argmax())
            t0.append(u)
            lg0 = s0.forward(torch.tensor([u]), start=len(text) + i)[0, -1]
    assert side == alone, (side, alone)
    assert t0 == t0_alone, (t0, t0_alone)
    assert s0.mm is None and s0.pos_delta == 0, (s0.mm, s0.pos_delta)
    assert _same_snap(_snap(s0), _snap(r0)), "slot 0's rows or state moved"
    return f"{n} greedy tokens identical on both slots and to the reference (delta {mm.delta})"


def _side_pool():
    """`MA.serve()`'s slot 0 and a side slot over the same weights, bound through the pool."""
    res = MA.serve()
    eng0 = app.STATE["engine"]
    from engine.models.qwen3_5.adapter import Qwen38Engine
    seng = Qwen38Engine(eng0.cfg, eng0.w, max_len=256, device="cpu")   # one copy of the weights
    seng.kv.k = seng.kv.k.to(TV.DT)
    seng.kv.v = seng.kv.v.to(TV.DT)
    seng.state.conv = seng.state.conv.to(TV.DT)
    vals = {k: None for k in S.OVERLAY}
    vals.update(engine=seng, drafter=None, max_len=256)
    side = S.Slot(1, 256, values=vals)
    pool = S.Pool([S.Slot(0, 256), side])
    app.STATE["pool"] = pool
    return res, pool, side, eng0, seng


def _image_request(ids, images, n, session="ses"):
    prompt, mm = app._mm_prompt(torch.tensor(ids), images)
    out = list(app.generate_stream(prompt, n, set(), None, session, mm=mm))
    return out, mm


def test_generate_stream_on_a_side_slot_with_images_is_slot_0s_answer():
    if need_or_skip():
        return need_or_skip()
    ims = [TV.image(61, (1, 4, 6)), TV.image(62, (1, 8, 4))]
    for k in range(30):
        ids = MA.with_image(MA.text(20, 5 + 1000 * k)) + MA.with_image(MA.text(6, 6)) + \
            MA.text(5, 7)
        MA.serve()
        on0, mm0 = _image_request(ids, ims, 12)
        if not {IMG, VS, VE} & set(on0):
            break
    res, pool, side, eng0, seng = _side_pool()
    s = pool.take_side(len(mm0.key_ids) + 12, owner="ses")
    assert s is side
    S.bind(s)
    try:
        on1, mm1 = _image_request(ids, ims, 12)
        pre1 = dict(app.STATE["last_prefill"])
        assert seng.mm is mm1 and seng.pos_delta == mm1.delta
    finally:
        S.bind(None)
        pool.release_side(s)
    assert on1 == on0, (on1, on0)
    assert list(mm1.key_ids) == list(mm0.key_ids)
    assert pre1["forwarded"] == len(mm1.key_ids), pre1
    assert eng0.mm is None and eng0.pos_delta == 0 and eng0.kv.length == 0, "slot 0 was touched"
    assert res.valid == 0, "slot 0's resident rows were touched"
    return f"12 greedy tokens identical, {len(mm1.key_ids)} rows read on slot 1 (delta {mm1.delta})"



def test_an_end_kept_after_images_on_a_side_slot_is_carried_into_slot_0_with_its_offset():
    """A side slot with a resident prefix keeps its image request's end; the conversation's next
    turn takes slot 0 and the end is carried in (FX10). The carry record kept no `delta` or `mm`,
    so slot 0 took it as a text end: offered to any request, and a compaction would continue
    it at the text rotary (server/compaction.py)."""
    if need_or_skip():
        return need_or_skip()
    from engine import cache
    x = TV.image(71, (1, 4, 6))
    for k in range(30):
        res0, pool, side, eng0, seng = _side_pool()
        side.values["resident"] = cache.ResidentPrefix(MA.BIG, MA.CH, end=True,
                                                       stash_bytes=MA.BIG)
        p1 = MA.with_image(MA.text(30, 11 + 1000 * k)) + MA.text(5, 12)
        s = pool.take_side(256, owner="ses")
        S.bind(s)
        try:
            prompt, mm1 = app._mm_prompt(torch.tensor(p1), [x])
            out1 = list(app.generate_stream(prompt, 16, set(), None, "ses", mm=mm1, end_mm=True))
            app._remember(list(p1), out1, "ses", mm=mm1)
        finally:
            S.bind(None)
            pool.release_side(s)
        if not {IMG, VS, VE} & set(out1):
            break
    rec = side.carry
    assert rec is not None and rec["at"] == len(mm1.key_ids) + len(out1) - 1, rec
    assert rec["mm"] is True and rec["delta"] == mm1.delta != 0, rec
    copy = p1 + out1 + MA.text(6, 13)
    out2, pre2, served, mm2 = MA._carried_image_turn(copy, [x], 12)
    assert res0.stats["side_carries"] == 1 and side.carry is None, res0.stats
    assert res0.end["mm"] is True and res0.end["delta"] == mm1.delta, res0.end
    assert pre2["reused"] == rec["at"], pre2
    d = MA.assert_cold_mm(pre2.pop("logits"), mm2, [x], 12, out2)
    return (f"carried {rec['at']} rows with delta {rec['delta']}, resumed there, logits {d:.1g} "
            f"off cold")


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:80s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
