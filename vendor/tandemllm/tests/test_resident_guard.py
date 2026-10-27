"""A short request between two turns of a long conversation leaves slot 0's conversation in place
(FX5, 2026-10-06). CPU only.

THE LIVE CASE (`--parallel 2`): an Open WebUI agent chat held 41,474 tokens on slot 0.
A new user message made the router ask Qwen its class and its package first, two short requests
(410 and 169 tokens) over the engine's own port; each ran on slot 0 as a guest of the
conversation. A guest's rows are copied aside first (the stash) and put back after it, but the
memory policy refused the copy -- the side slot's 7.65 GiB had been taken out of the caches' room
at the first request (`memlimits.apply` set the ceiling to the bare 70 GiB) -- and a refused stash
cuts the resident rows at the guest's start: the user's turn read all 41,749 tokens again (56 s).
Replayed on the live engine (one build, n = 1 each): ceiling 70 -> cached share 0.000 and 0.000 on
two new user messages (67.5 s, 70.7 s); ceiling 78 -> 0.999 and 0.993 (6.4 s, 9.4 s).

  1. the side slots' bytes stay in the limits when a residency document sets them again
  2. a guest's stash holds its reach, not the whole stash budget; the regression itself: a long
     resident, a short request under a tight memory policy, the next turn resumes bit for bit
  3. with `--parallel N`, a guest of slot 0's conversation takes an idle side slot first
  4. Open WebUI's live working directory leaves the tool block for a note

Run: python tests/test_resident_guard.py
"""

from __future__ import annotations

import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import cache  # noqa: E402
from engine.models.qwen3_5.adapter import Qwen38Engine  # noqa: E402  (the served class: its cold verbs)
from server import memlimits  # noqa: E402
from test_cache import DEV, DT, tokens  # noqa: E402
from test_forward_tree import FakeWeights, tiny_config  # noqa: E402

GIB = 1 << 30
CH = 16


def engine(max_len: int = 4096, seed: int = 3):
    cfg = tiny_config()
    cfg.max_position_embeddings = max_len
    eng = Qwen38Engine(cfg, FakeWeights(cfg, seed), max_len=max_len, device=DEV)
    eng.kv.k = eng.kv.k.to(DT)
    eng.kv.v = eng.kv.v.to(DT)
    eng.state.conv = eng.state.conv.to(DT)
    return eng


def cold(ids, max_len: int = 4096):
    eng = engine(max_len)
    lg, _, _ = cache.prefill(eng, None, ids, DEV, store=None, chunk=CH)
    return eng, lg


def per_row(eng) -> int:
    return sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in cache._kv_views(eng, None))


# ------------------------------------------------------------------ 1. the limits

def test_the_side_slots_stay_in_the_limits_when_a_document_sets_them_again():
    mem = cache.Budget(70 * GIB, 77 * GIB, 2 * GIB, probe=lambda: (0, 0))
    side = int(7.65 * GIB)
    mem.fixed += side                      # what server/app.py `_build_side_slots` does
    mem.ceiling += side
    mem.peak += side
    state = {"mem": mem}
    memlimits.apply(state, {"MEM_CEILING_GIB": 70.0, "MEM_PEAK_GIB": 77.0})
    # 0.3.44 to 0.3.46: 70 and 77 GiB here, the side slot's bytes gone from the caches' room
    assert mem.ceiling == 70 * GIB + side, mem.ceiling / GIB
    assert mem.peak == 77 * GIB + side
    memlimits.apply(state, {"MEM_CEILING_GIB": 72.0})
    assert mem.ceiling == 72 * GIB + side
    assert mem.report()["fixed"] == side


def test_without_side_slots_the_limits_are_the_configured_ones():
    mem = cache.Budget(70 * GIB, 77 * GIB, 2 * GIB, probe=lambda: (0, 0))
    memlimits.apply({"mem": mem}, {"MEM_CEILING_GIB": 64.0, "MEM_PEAK_GIB": 70.0})
    assert (mem.ceiling, mem.peak, mem.fixed) == (64 * GIB, 70 * GIB, 0)


# ------------------------------------------------------------------ 2. the stash

def test_a_guest_stash_holds_its_reach():
    eng = engine()
    res = cache.ResidentPrefix(GIB, CH, stash_bytes=GIB)
    turn1 = tokens(3000, seed=1)
    cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
    guest = tokens(30, seed=2)
    cache.prefill(eng, None, guest, DEV, chunk=CH, resident=res, reach=30 + 8)
    st = res._stash
    assert st is not None and st["start"] == 0
    assert st["rows"] == 38 + 2 * cache.GUEST_SLACK_ROWS, st["rows"]
    assert res._stash_rows == 1072, res._stash_rows     # 1,062 rounded up to the 16-row grid
    # without a reach (any other caller): every row the budget holds, as before
    res2, eng2 = cache.ResidentPrefix(GIB, CH, stash_bytes=GIB), engine()
    cache.prefill(eng2, None, turn1, DEV, chunk=CH, resident=res2)
    cache.prefill(eng2, None, guest, DEV, chunk=CH, resident=res2)
    assert res2._stash["rows"] == 3000


def _tight(eng, rows_room: int):
    """A memory policy with room for `rows_room` stash rows above what is allocated, and plenty for
    the anchors (a few KiB each on the random model)."""
    room = rows_room * per_row(eng)
    alloc = 60 * GIB
    return cache.Budget(alloc + room + 2 * GIB, 100 * GIB, 2 * GIB, probe=lambda: (alloc, alloc))


def _long_then_guest_then_turn(reach):
    eng = engine()
    mem = _tight(eng, 2000)                              # the whole 3,000-row stash does not fit
    res = cache.ResidentPrefix(GIB, CH, stash_bytes=GIB, mem=mem)
    turn1 = tokens(3000, seed=11)
    cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
    q = tokens(40, seed=12)                              # the router's question, say
    cache.prefill(eng, None, q, DEV, chunk=CH, resident=res, reach=reach)
    eng.forward(torch.tensor(tokens(3, seed=13)), start=40, last_only=True)   # its answer
    turn2 = turn1 + tokens(25, seed=14)                  # the new user message
    info: dict = {}
    warm, reused, fwd = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res, info=info)
    return eng, res, turn2, warm, reused, fwd


def test_a_short_request_under_a_tight_memory_policy_no_longer_costs_the_conversation():
    """The regression: long resident + internal short request + new user turn."""
    # before: the stash asks for every row it could hold, is refused, and the rows are cut at 0
    _, res, turn2, _, reused, fwd = _long_then_guest_then_turn(None)
    assert res.stats["refused_stash"] == 1 and reused == 0 and fwd == len(turn2)
    # now: the stash asks for the question's reach, is admitted, and the turn resumes in place
    eng, res, turn2, warm, reused, fwd = _long_then_guest_then_turn(40 + 8)
    assert res.stats["refused_stash"] == 0 and res.stats["stash_restored"] == 1
    assert reused == 2992 and fwd == len(turn2) - 2992, (reused, fwd)
    assert reused / len(turn2) >= 0.95
    ref, lg = cold(turn2)
    assert torch.equal(warm, lg), (warm - lg).abs().max().item()
    assert torch.equal(eng.state.S, ref.state.S) and torch.equal(eng.state.conv, ref.state.conv)
    n = len(turn2)
    assert torch.equal(eng.kv.k[:, :, :, :n], ref.kv.k[:, :, :, :n])
    assert torch.equal(eng.kv.v[:, :, :, :n], ref.kv.v[:, :, :, :n])


def test_a_guest_that_writes_past_its_reach_still_cuts_safely():
    eng = engine()
    res = cache.ResidentPrefix(GIB, CH, stash_bytes=GIB)
    turn1 = tokens(3000, seed=21)
    cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
    cache.prefill(eng, None, tokens(30, seed=22), DEV, chunk=CH, resident=res, reach=38)
    rows = res._stash["rows"]
    eng.forward(torch.tensor(tokens(rows, seed=23)), start=30, last_only=True)   # far past it
    turn2 = turn1 + tokens(9, seed=24)
    warm, reused, _ = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res)
    assert res.stats["stash_short"] == 1 and reused <= rows
    ref, lg = cold(turn2)
    assert torch.equal(warm, lg)


# ------------------------------------------------------------------ 3. the side slot first

def test_a_guest_of_slot_0_takes_an_idle_side_slot_first():
    from server import app as A
    from server import slots as S

    class Res:
        def __init__(self, toks):
            self.tokens, self.valid, self.chunk = list(toks), len(toks), 1024
            self.guest_max, self.guest_frac = 8192, 0.25

        def common(self, ids):
            return A._shared_head(ids, self.tokens)

    conv = [10_000 + i for i in range(41_474)]
    pool = S.Pool([S.Slot(0, 262_144), S.Slot(1, 65_536)])
    saved = {k: A.STATE.get(k) for k in ("pool", "resident", "slot0_view", "prefix_chunk")}
    A.STATE.update(pool=pool, resident=Res(conv), prefix_chunk=1024, slot0_view=conv)
    real = A.build_prompt

    def need_for(ids, max_new=64, lane=None):
        A.build_prompt = lambda body, client=None, *, count=False, left_out=0: list(ids)
        try:
            why: dict = {}
            return A._par_side_need({"messages": []}, None, 0, max_new, None, [], lane, why), why
        finally:
            A.build_prompt = real
    try:
        question = [5, 6, 7] + [900_000 + i for i in range(407)]       # 410 tokens, unrelated
        need, why = need_for(question)
        assert need is not None and why["guest"] == 41_474, why
        # the router's question is short enough for the side lane: still a guest (live on the first
        # 0.3.47 build it was not, and ran on slot 0)
        need, why = need_for(question, lane="short")
        assert need is not None and why["guest"] == 41_474, why
        # FX15: the package question (2,354 tokens) over a fresh Open WebUI chat of 8,112 rows is a
        # guest too: more than a quarter of the chat, which bounds only a guest ON slot 0's stash
        chat = conv[:8112]
        A.STATE["resident"], A.STATE["slot0_view"] = Res(chat), chat
        package_q = [5, 6, 7] + [700_000 + i for i in range(2351)]
        need, why = need_for(package_q, lane="short")
        assert need is not None and why.get("guest") == 8112, {k: v for k, v in why.items()
                                                                 if k != "ids"}
        # a lane job that is no guest stays the lane's (slot 0's safe points), as before
        A.STATE["resident"], A.STATE["slot0_view"] = Res(conv[:1500]), conv[:1500]
        need, why = need_for(question, lane="short")
        assert need is None and "guest" not in why
        A.STATE["resident"], A.STATE["slot0_view"] = Res(conv), conv
        # the conversation's own next turn waits for slot 0, as SRV-197 made it
        need, why = need_for(conv + [1, 2, 3], max_new=32_000)
        assert (need is None or why.get("slot0_first")) and why["wait"] == "view" and "guest" not in why
        # a request that shares the head (a sub-agent of the same session) is not sent away:
        # slot 0's anchors serve its head
        need, why = need_for(conv[:2048] + [3, 4], max_new=64)
        assert "guest" not in why, why
        # too long to be a guest: a new long conversation (it takes slot 0 or a side slot as before)
        need, why = need_for([800_000 + i for i in range(12_000)], max_new=64)
        assert "guest" not in why
        # a short resident is nothing to guard
        A.STATE["resident"], A.STATE["slot0_view"] = Res(conv[:1500]), conv[:1500]
        need, why = need_for(question)
        assert "guest" not in why
        # off: the old behaviour (slot 0 with the stash)
        A.STATE["resident"], A.STATE["slot0_view"] = Res(conv), conv
        was, A.SIDE_GUESTS = A.SIDE_GUESTS, "off"
        try:
            need, why = need_for(question)
            assert "guest" not in why
        finally:
            A.SIDE_GUESTS = was
    finally:
        for k, v in saved.items():
            if v is None:
                A.STATE.pop(k, None)
            else:
                A.STATE[k] = v


def test_the_routers_question_on_a_side_slot_is_no_conversation_to_carry():
    """FX15 (0.4.7-test b1, live W01/W02): with the router's questions on the side slot, the package
    question (2,364 tokens) was longer than two chunks, so its end was noted as a conversation;
    the next package question shares its options, was carried into slot 0 over the chat (parked),
    and the note kept the next guests off the side slot for CARRY_KEEP_S. A slot taken by the
    router's own question or a side-lane job notes no end; a client's turn still does."""
    from server import app as A
    from server import slots as S
    slot = S.Slot(1, 65_536)
    slot.carry = {"gen": 0, "t": 0.0, "at": 1}
    slot.no_carry = True
    was = A.SIDE_CARRY
    A.SIDE_CARRY = "on"
    S.bind(slot)
    try:
        assert A._side_keep_end(list(range(3000)), [], [], "c:x") == "lane"
        assert slot.carry is None
    finally:
        S.bind(None)
        A.SIDE_CARRY = was
    src = open(A.__file__).read()
    # 0.4.8's park and side-slot work (798b413, b375177) added three more side paths with the rule
    assert src.count('side.no_carry = bool(me["own"]) or lane_kind is not None') == 6


# ------------------------------------------------------------------ 4. the working directory

def _owui_tools(cwd: str) -> list:
    desc = "Run a shell command in the terminal and return its output."
    return [{"type": "function", "function": {"name": "run_command", "description":
             desc + f"\n\nThe current working directory is: {cwd}",
             "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write a file.",
             "parameters": {"type": "object", "properties": {}}}}]


class _Plan:
    def __init__(self, mode="on"):
        self.res = type("R", (), {"mode": mode})()
        self.decisions: dict = {}
        self.extras: list = []


def _turn(router_mod, sess, cwd, mode="on", compaction=False):
    body = {"tools": _owui_tools(cwd)}
    plan, row = _Plan(mode), {"extra": {}}
    sh = type("Sh", (), {"compaction": compaction})()
    router_mod.Router._client_cwd(plan, body, sh, sess, row)
    return body, plan, row


def test_the_working_directory_leaves_the_tool_block_for_a_note():
    from server.conductor import router as R
    sess = R.Session("h:test")
    body, plan, row = _turn(R, sess, "/home/user")
    d = body["tools"][0]["function"]["description"]
    assert d == "Run a shell command in the terminal and return its output.", repr(d)
    assert body["tools"][1]["function"]["description"] == "Write a file."
    assert plan.decisions["client"] == "on" and len(plan.extras) == 1
    x = plan.extras[0]
    assert x["text"].endswith("is: /home/user") and x["kinds"] == {"client"}
    k1 = x["key"]
    # a later turn in the same directory: the same key, so the live note is not added twice
    sess.turns += 3
    body2, plan2, _ = _turn(R, sess, "/home/user")
    assert plan2.extras[0]["key"] == k1
    assert body2["tools"][0]["function"]["description"] == d       # the tool block is the same
    # the agent did `cd pagoda-garden`: a new note, the tool block still the same
    sess.turns += 1
    body3, plan3, row3 = _turn(R, sess, "/home/user/pagoda-garden")
    assert plan3.extras[0]["key"] != k1 and row3["extra"]["client_cwd"]["changed"]
    assert plan3.extras[0]["text"].endswith("/home/user/pagoda-garden")
    assert body3["tools"][0]["function"]["description"] == d
    # and back to the first directory: a new note again (the newest note must say where it is)
    sess.turns += 1
    _, plan4, _ = _turn(R, sess, "/home/user")
    assert plan4.extras[0]["key"] not in (k1, plan3.extras[0]["key"])


def test_the_working_directory_is_left_alone_unless_the_profile_is_on():
    from server.conductor import router as R
    for mode in ("shadow", "off"):
        body, plan, _ = _turn(R, R.Session("h:x"), "/home/user", mode=mode)
        assert body["tools"][0]["function"]["description"].endswith("is: /home/user")
        assert not plan.extras
    body, plan, _ = _turn(R, R.Session("h:y"), "/home/user", compaction=True)
    assert body["tools"][0]["function"]["description"].endswith("is: /home/user")
    was, R.CLIENT_CWD = R.CLIENT_CWD, "off"
    try:
        body, plan, _ = _turn(R, R.Session("h:z"), "/home/user")
        assert body["tools"][0]["function"]["description"].endswith("is: /home/user")
    finally:
        R.CLIENT_CWD = was
    # tools without the sentence: nothing changes, no note
    body = {"tools": [{"type": "function", "function": {"name": "read", "description": "Read."}}]}
    plan = _Plan()
    R.Router._client_cwd(plan, body, type("Sh", (), {"compaction": False})(), R.Session("h:w"),
                         {"extra": {}})
    assert body["tools"][0]["function"]["description"] == "Read." and not plan.extras


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
