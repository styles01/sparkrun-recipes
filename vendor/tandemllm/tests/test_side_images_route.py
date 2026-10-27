"""SRV-629: the slot choice for a request with images. CPU only, no engine: `_par_side_need`,
`_side_home`, the view keep and the admission with tests/test_par_continue.py's stand-ins.

THE CASE (live 2026-10-19). Two Chat-tab agent runs called render_page; every later request
carried 3 to 8 screenshots and `_par_side_need` refused each one a side slot, so after the first
carry into slot 0 all 31 requests of both chats ran there (56.6 min of wall time against 18.9
min of decoding) while slot 1 stayed idle. tests/test_side_images.py proves a side slot's
engine decodes an image request as slot 0's does; this file checks the choice:

  1  the rows: the window, the output cut and the admission count each image's expanded rows,
     not its one placeholder (8 screenshots: about 7,700 rows); `_mm_keys` is the context's
     `key_ids`, on the host
  2  `SIDE_IMAGES` off: slot 0 only, as 0.12.0; on: a side slot when it fits
  3  every routing test on the key ids, as the caches compare them: slot 0's rows and view, a
     side slot's home, the park and the NVMe with an end kept after images, the carry's choice;
     a request with images is never a guest of slot 0
  4  the Lanes view's queued line and the `[parallel]` line for both flag values

Run: python tests/test_side_images_route.py
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine import vision as V  # noqa: E402
from server import app as A  # noqa: E402
from server import slots as S  # noqa: E402
from test_par_continue import CHUNK, HEAD, FakeResident, _restore, _session, _setup  # noqa: E402

IMG = 900_001                     # the template's image placeholder in these stand-in ids
MERGE = 2


def shot(seed: int, h: int = 54, w: int = 84) -> V.Image:
    """A preprocessed screenshot stand-in: `h x w` patches, `h * w / 4` rows (54 x 84: 1,134)."""
    g = torch.Generator().manual_seed(seed)
    pv = torch.randn(h * w, 4, generator=g)
    grid = (1, h, w)
    return V.Image(pv, grid, V.digest_of(pv, grid), "data")


def tower(own=None):
    """The vision tower as the router reads it: its placeholder and its merge (the Qwen towers),
    or a tower with its own `context` (Super VL)."""
    t = types.SimpleNamespace(cfg=types.SimpleNamespace(image_token_id=IMG,
                                                        spatial_merge_size=MERGE))
    if own is not None:
        t.context = own
    return t


def with_shots(ids: list[int], n: int) -> list[int]:
    return list(ids) + [IMG] * n


def need_img(ids, images, max_new=32_000, plan_active=True):
    """`_par_side_need` for a request with `images`, the renderer answering `ids`."""
    real = A.build_prompt

    class Plan:
        active = plan_active

    def fake(body, client=None, *, count=False, left_out=0):
        assert count == "ids"
        return list(ids)
    A.build_prompt = fake
    try:
        why: dict = {}
        need = A._par_side_need({"messages": []}, None, 0, max_new, Plan(), images, None, why)
        return need, why
    finally:
        A.build_prompt = real


class flag:
    """`SIDE_IMAGES` and the tower for the block."""

    def __init__(self, value: str, t=None):
        self.value, self.t = value, t if t is not None else tower()

    def __enter__(self):
        self.was, self.had = A.SIDE_IMAGES, A.STATE.get("vision")
        A.SIDE_IMAGES = self.value
        A.STATE["vision"] = self.t
        return self

    def __exit__(self, *exc):
        A.SIDE_IMAGES = self.was
        if self.had is None:
            A.STATE.pop("vision", None)
        else:
            A.STATE["vision"] = self.had


# ------------------------------------------------------------------ 1. the rows
def test_mm_keys_are_the_contexts_key_ids():
    ims = [shot(1), shot(2, 40, 64), shot(1)]                   # the same screenshot twice
    ids = with_shots([5, 6, 7], 1) + [8, 9] + with_shots([], 1) + [10] + with_shots([], 1) + [11]
    with flag("on"):
        keys = A._mm_keys(ids, ims)
        full, spans = V.expand(ids, ims, IMG, MERGE)
        eng_cfg = types.SimpleNamespace(rotary_dim=16, mrope_section=[2, 1, 1])
        mm = V.MMContext(full, spans, ims, A.STATE["vision"], eng_cfg)
        assert keys == list(mm.key_ids), "the host keys are the caches' keys"
        assert len(keys) == len(ids) - 3 + 1_134 + 640 + 1_134
        assert A._mm_rows(len(ids), ims) == len(keys)
        assert keys[3] == keys[3 + 1_134 + 2 + 640 + 1] == V.key_id(ims[0].digest)
        assert keys[3 + 1_134 + 2] != keys[3]
        assert A._mm_keys([5, IMG], ims) is None, "a placeholder short: the handler's 400"
    # Super VL builds its own context: its images stay on slot 0; its rows are still counted
    own = tower(own=lambda ids, images, cache=None, on_encode=None: (
        [t for i in ids for t in ([IMG - 1] + [IMG] * 9 + [IMG + 1] if i == IMG else [i])], None))
    with flag("on", own):
        assert A._mm_keys(ids, ims) is None
        assert A._mm_rows(len(ids), ims) == len(ids) + 3 * 10
    return f"{len(keys)} rows for {len(ids)} rendered tokens"


def test_the_window_counts_the_expanded_rows():
    """Eight screenshots (9,072 rows) in a 27,000-token conversation on a 65,536-token side slot:
    counted as eight placeholders the output limit would be cut to 30,269 and the generation
    would run past the window; the rows leave it 21,205."""
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]
    ims = [shot(10 + i) for i in range(8)]
    ids = with_shots(_session(4, HEAD, 20_000), 8) + [7, 8, 9]
    pool, saved = _setup(FakeResident(r1), view=r1)
    try:
        with flag("on"):
            need, why = need_img(ids, ims)
            margin = A.SIDE_MARGIN + A.PLAN_MARGIN
            rows = len(ids) - 8 + 8 * 1_134
            assert why["n_rows"] == rows and why["n_images"] == 8, why
            assert need == rows + margin + A.SIDE_MIN_ROOM and why["clip"], (need, why)
            room = A._side_room(pool.side()[0], why["n_rows"], margin, 32_000)
            assert room == 65_536 - rows - margin == 21_205, room
    finally:
        _restore(saved)
    return f"{rows} rows, need {need}, max_tokens 32000 -> {room}"


def test_the_admission_reserves_the_expanded_rows():
    seen = []

    class Mem:
        def workspace(self, n):
            seen.append(n)
            return n

    class H:
        def _client_gone(self):
            return False
    adm = S.Admission(0, meminfo=lambda: {"MemAvailable": 1 << 40})
    pool = types.SimpleNamespace(admission=adm, turn=None)
    saved = A.STATE.get("mem")
    A.STATE["mem"] = Mem()
    try:
        assert A.Handler._par_admit(H(), pool, "a", torch.zeros(600)) is True
        assert A.Handler._par_admit(H(), pool, "b", torch.zeros(600), rows=9_672) is True
    finally:
        if saved is None:
            A.STATE.pop("mem", None)
        else:
            A.STATE["mem"] = saved
    assert seen == [600, 9_672] and adm.held["b"] == 9_672, (seen, adm.held)


# ------------------------------------------------------------------ 2. the flag
def test_side_images_is_on_by_default():
    """Live 2026-10-19 on the DGX Spark (the 27B, EXL3 6.0, `--parallel 2`): an image request on
    slot 1 beside a text run on slot 0 wrote slot 0's 300 tokens, and two image requests decoded
    in joint rounds; QWEN38_SIDE_IMAGES=off restores 0.12.0."""
    from engine.settings import KNOBS
    assert KNOBS["SIDE_IMAGES"][0] == "on"
    if "QWEN38_SIDE_IMAGES" not in os.environ:
        assert A.SIDE_IMAGES == "on"


def test_flag_off_keeps_images_on_slot_0():
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]
    ids = with_shots(_session(4, HEAD, 900), 2) + [7, 8, 9]
    pool, saved = _setup(FakeResident(r1), view=r1)
    try:
        with flag("off"):
            need, why = need_img(ids, [shot(1), shot(2)], max_new=4_000)
            assert need is None and why == {"images": True}, why
    finally:
        _restore(saved)


def test_flag_on_gives_a_new_chat_with_images_a_side_slot():
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]
    ids = with_shots(_session(4, HEAD, 900), 2) + [7, 8, 9]
    pool, saved = _setup(FakeResident(r1), view=r1)
    try:
        with flag("on"):
            need, why = need_img(ids, [shot(1), shot(2)], max_new=4_000)
            margin = A.SIDE_MARGIN + A.PLAN_MARGIN
            assert need == len(ids) - 2 + 2 * 1_134 + 4_000 + margin, (need, why)
            assert "images" not in why and not why.get("wait"), why
    finally:
        _restore(saved)
    return f"need {need}"



# ------------------------------------------------------------------ 3. key space
GEN = [888_000 + i for i in range(900)]


def keys_of(ids, ims):
    with flag("on"):
        return A._mm_keys(ids, ims)


def _image_chat():
    """An agent's turn with two screenshots, the rows its slot holds after it (the served prompt
    as the caches key it, then the generation), and its next turn with a third screenshot."""
    ims = [shot(1), shot(2)]
    p1 = with_shots(_session(1, HEAD, 1_500), 1) + [50, 51] + with_shots([], 1) + [7, 8, 9]
    k1 = keys_of(p1, ims)
    rows = k1 + GEN
    p2 = p1[:-1] + [6] + GEN[:300] + with_shots([60], 1) + [7, 8, 9]
    return ims, p1, k1, rows, p2, ims + [shot(3)]


def test_a_turn_with_screenshots_continues_slot_0s_rows_in_key_space():
    """The rows hold each image as its key id; the rendered ids part from them at the first
    placeholder (8,500 here), so a text test calls the next turn a new conversation and sends it
    to the side slot, which reads all of it again. In key space it continues slot 0's."""
    ims, p1, k1, rows, p2, ims2 = _image_chat()
    first = p1.index(IMG)
    assert FakeResident(rows).common(p2) == first, "the rendered ids part at the image"
    for view in (k1, None):
        pool, saved = _setup(FakeResident(rows), view=view)
        try:
            with flag("on"):
                need, why = need_img(p2, ims2, max_new=4_000)
            assert need is not None and why.get("slot0_first"), why
            if view is not None:
                assert why["wait"] == "view" and why["view_common"] == len(k1) - 1, why
            else:
                assert why["wait"] == "resident" and why["res_common"] == len(k1) - 1, why
        finally:
            _restore(saved)
    return f"shares {len(k1) - 1} keys (the rendered ids: {first})"


def test_another_screenshot_in_the_same_place_is_another_conversation():
    ims, p1, k1, rows, p2, ims2 = _image_chat()
    other = [shot(9), ims[1], shot(3)]
    pool, saved = _setup(FakeResident(rows), view=k1)
    try:
        with flag("on"):
            need, why = need_img(p2, other, max_new=4_000)
        assert need is not None and not why.get("wait") and not why.get("slot0_first"), why
        assert why["view_common"] == p1.index(IMG), why       # the keys part at that image
    finally:
        _restore(saved)


class SideRes(FakeResident):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.end = {"at": self.valid}


def test_a_side_slots_image_conversation_is_its_home():
    ims, p1, k1, rows, p2, ims2 = _image_chat()
    r0 = _session(2, HEAD, 1_500) + [7, 8, 9]
    pool, saved = _setup(FakeResident(r0), view=r0)
    try:
        vals = {k: None for k in S.OVERLAY}
        vals.update(max_len=65_536, resident=SideRes(rows), slot_view=list(k1))
        pool.slots[1].values = vals
        with flag("on"):
            need, why = need_img(p2, ims2, max_new=4_000)
            assert need is not None and why.get("home") == 1, why
            # its rows alone, no view kept: the slot's own rows in key space
            vals["slot_view"] = None
            need, why = need_img(p2, ims2, max_new=4_000)
            assert why.get("home") == 1, why
            # the same text with another first screenshot: not its home
            need, why = need_img(p2, [shot(9), ims[1], shot(3)], max_new=4_000)
            assert "home" not in why, why
    finally:
        pool.slots[1].values = None
        _restore(saved)


def test_the_park_and_the_nvme_are_asked_for_an_end_after_images():
    """A request with images may resume at an end kept after images (`mm_ok`), as the handler's
    own restore asks (`_spill_before_replay(..., mm_ok=True)`); a text request may not."""
    ims, p1, k1, rows, p2, ims2 = _image_chat()
    asked = []

    class Res(FakeResident):
        park_rows = 1

        def park_offer(self, ids, c, max_len, align=True, mm_ok=False):
            asked.append(("park", mm_ok, list(ids[:len(k1) - 1]) == k1[:-1]))
            return len(k1) if mm_ok else 0

    class Spill:
        entries = {"e": 1}
        stats: dict = {}

        def candidate(self, res, keys, max_len, replay=False, mm_ok=False, miss=None):
            asked.append(("spill", mm_ok, list(keys[:len(k1) - 1]) == k1[:-1]))
            return None
    r0 = _session(2, HEAD, 1_500) + [7, 8, 9]
    pool, saved = _setup(Res(r0), view=r0)
    A.STATE["spill"] = Spill()
    try:
        with flag("on"):
            need, why = need_img(p2, ims2, max_new=4_000)
        assert asked == [("spill", True, True), ("park", True, True)], asked
        assert why["wait"] == "park" and why["park"] == len(k1) and why.get("slot0_first"), why
    finally:
        A.STATE.pop("spill", None)
        _restore(saved)


def test_a_request_with_images_is_never_a_guest_of_slot_0():
    """A short unrelated request is served on slot 0 over the resident rows with the stash when
    no side slot is free (FX5); no request with images ever was."""
    r0 = _session(2, HEAD, 1_500) + [7, 8, 9]

    class Res(FakeResident):
        guest_max = 8_192
    pool, saved = _setup(Res(r0), view=r0)
    try:
        short = [400_000 + i for i in range(30)]
        with flag("on"):
            _, why = need_img(short, [])
            assert why.get("guest"), why
            _, why = need_img(with_shots(short, 1), [shot(1)])
            assert not why.get("guest"), why
    finally:
        _restore(saved)


def test_the_view_is_kept_in_key_space():
    """At a request's end the rows it left are compared with its served prompt: for one with
    images, the image rows as their keys (`_keyed`), else the placeholders never match the rows
    and no view is kept."""
    ims, p1, k1, rows, p2, ims2 = _image_chat()
    full, spans = V.expand(p1, ims, IMG, MERGE)
    mm = types.SimpleNamespace(n=len(full), key_ids=k1)
    ctx = full + GEN
    pool, saved = _setup(FakeResident(rows))
    try:
        assert not A._par_view_keep(ctx[:len(full)], k1)
        assert A._par_view_keep(A._keyed(ctx[:len(full)], mm), k1)
        assert A.STATE["slot0_view"] == k1
        side = pool.slots[1]
        side.values = {k: None for k in S.OVERLAY}
        side.values.update(resident=SideRes(rows))
        assert A._side_view_keep(side, A._keyed(ctx[:len(full)], mm), k1)
        assert side.values["slot_view"] == k1
    finally:
        pool.slots[1].values = None
        A.STATE.pop("slot0_seen", None)
        _restore(saved)
    assert A._keyed(full[:5], mm) == full[:5] and A._keyed(full, None) == full



def test_a_side_slots_shorter_end_is_not_carried_over_slot_0s_end_after_images():
    """`_carry_candidate`: slot 0's own end kept after images counts for a prompt with images
    (its keys), as its prefill offers it; a side slot's shorter record of the conversation is
    not carried over it."""
    ims, p1, k1, rows, p2, ims2 = _image_chat()
    k2 = keys_of(p2, ims2)

    class Res(FakeResident):
        def best(self, c, m):
            return 0

        def end_offer(self, c, m, mm_ok=False):
            return len(k1) - 1 if mm_ok else 0

        def pin_offer(self, ids, c, m):
            return 0

        def park_offer(self, ids, c, m, align=True, mm_ok=False):
            return 0
    pool, saved = _setup(Res(rows))
    try:
        side = pool.slots[1]
        side.carry = {"gen": side.gen, "at": 4_000, "tokens": k2[:4_000], "n_prompt": 4_000,
                      "hidden": []}
        assert A._carry_candidate(pool, Res(rows), k2, 262_144) is None
        # a text prompt is not offered that end: the record beats what slot 0 offers it
        text = [t if t < V.KEY_BASE else 5 for t in k2]
        side.carry["tokens"] = text[:4_000]
        assert A._carry_candidate(pool, Res(rows), text, 262_144) == (side, side.carry)
    finally:
        pool.slots[1].carry = None
        _restore(saved)



# ------------------------------------------------------------------ 4. what the Lanes view and the log say
def test_the_queued_line_and_the_parallel_line_for_both_flag_values():
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]
    ids = with_shots(_session(4, HEAD, 900), 2) + [7, 8, 9]
    me = {"rank": 0, "t": 0.0, "side": None, "own": False, "rec": None}
    pool, saved = _setup(FakeResident(r1), view=r1)
    try:
        with flag("off"):
            need, why = need_img(ids, [shot(1), shot(2)], max_new=4_000)
            assert A._queue_why(pool, why, need, 0.0, me) == \
                "it has images, which only slot 0 reads; waits for slot 0"
            assert "images=" not in A._par_nums(why)
        with flag("on"):
            need, why = need_img(ids, [shot(1), shot(2)], max_new=4_000)
            line = A._queue_why(pool, why, need, 0.0, me)
            assert "images" not in line, line
            nums = A._par_nums(why)
            assert nums.startswith(f"images=2 rows={len(ids) - 2 + 2 * 1_134} "), nums
            assert f"need={need}" in nums, nums
        own = tower(own=lambda ids, images, cache=None, on_encode=None: (list(ids), None))
        with flag("on", own):
            need, why = need_img(ids, [shot(1), shot(2)], max_new=4_000)
            assert need is None and A._queue_why(pool, why, need, 0.0, me) == \
                "this model's images are read on slot 0 only; waits for slot 0"
    finally:
        _restore(saved)
    return nums


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            out = fn()
            print(f"ok {name}" + (f"  {out}" if out else ""), flush=True)
            n += 1
    print(f"{n} passed")
