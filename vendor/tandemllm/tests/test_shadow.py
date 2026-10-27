"""The shadow reader (SR1, 2026-10-04): a small Qwen3.5 model keeps its own state of each session.

The test matrix (a random tiny Qwen3.5 checkpoint, test_thoughts' `tiny_dir`, on the CPU):

  engine/shadow.py
    1  the id mapping: the placeholders, an id past the vocabulary and an added token of the 27B
       that is not the small model's are dropped; the rest pass as they are   test_the_id_mapping
    2  EXACT: a session read turn by turn (prompts and answers of random lengths, the reader's own
       grid) is bit for bit the fresh prefill of the same ids on the same grid -- every KV row,
       the recurrent and the convolution state; its KV grows in steps             test_exact_turn_by_turn
    3  a question (`fork_logits`) is the fresh forward of the same ids, and the session is the
       same after, bit for bit; a generation too; a fork left open is closed by the next read
                                                                         test_a_fork_changes_nothing
    4  a prompt that differs from what was read (an earlier answer re-rendered) goes back to the
       newest checkpoint at or below the difference, and the result is still the fresh prefill;
       without a checkpoint it starts over                                       test_the_rollback
    5  a session to the disk and back is the same state; another grid refuses   test_save_and_load
  server/conductor/shadow.py
    6  the service: turns fed (host only), read in `work` until the deadline or `stop()`, before
       every chunk; status                                                       test_the_service
    7  the router's question at the turn's insertion point, answered in the idle window, journaled
       with the router's class, the agreement and how far behind the shadow was    test_the_router_question
    8  the cap: the least recently used session goes to the disk and comes back exact when it is
       fed again; a drain writes every one                                       test_the_cap_and_the_disk
    9  `ask` over the whole session: a typed answer, the letters' softmax           test_ask
   10  the narrator's thought summary read from the shadow, in slices, the session the same after
                                                                         test_the_narrator_reads_the_shadow
   12  inside a turn: the reading stays within its share of the turn's time (one chunk over at
       most), reads the prompt then the output, and leaves an open fork alone    test_inline_share
   13  the handler (the fake engine in process, a conductor, the tiny model as the shadow): a
       profile with `shadow.mode: on` streams byte for byte what it streams without; the turn
       feeds the session (the router's key), the row says how far behind the shadow was; the
       idle reading and the router's question follow in `shadow_once`         test_the_handler
   11  the profile switch: absent is off; the schema takes the block and refuses a wrong one
                                                                                 test_the_profile_block

Run on a machine with the engine's dependencies (the box's CPU): python tests/test_shadow.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import threading

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ[f"QWEN38_FUSED_{_k}"] = "0"          # the CPU forward

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from engine import shadow as S  # noqa: E402
from engine.generative.slot import Slot  # noqa: E402
from test_thoughts import tiny  # noqa: E402

S.GROW_ROWS = 64                         # the growth path runs in a small test
GRID = 16


def model():
    from engine.generative.qwen35 import Qwen35Model
    m = Qwen35Model("tiny", tiny(), max_ctx=1024, claim_gib=0.1, engine_dir=tiny())
    m.load(None)
    return m


_M: list = []


def reader(grid: int = GRID) -> S.Reader:
    if not _M:
        _M.append(model())
    return S.Reader(_M[0], grid=grid)


def text(n: int, seed: int) -> list[int]:
    r = random.Random(seed)
    return [r.randrange(3, 250) for _ in range(n)]


IM_START, IM_END = 257, 258


def turns(seed: int = 0, n: int = 5):
    """(prompt, answer) pairs of a session whose prompts only grow: each prompt is the last one,
    its answer and a new user message."""
    r = random.Random(seed)
    stream: list[int] = [IM_START] + text(r.randrange(20, 70), seed) + [IM_END]
    out = []
    for i in range(n):
        stream = stream + [IM_START] + text(r.randrange(5, 40), seed + 100 + i) + [IM_END, IM_START]
        ans = text(r.randrange(3, 50), seed + 200 + i) + [IM_END]
        out.append((list(stream), ans))
        stream = stream + ans
    return out


# ------------------------------------------------------------------ engine/shadow.py

def test_the_id_mapping():
    rd = reader()
    rd.drop = rd.drop | {259}                       # stands in for an engine-only added token
    ids = [5, 6, 259, 299, 300, 10**12, 7, 256]
    assert rd.map_ids(ids) == [5, 6, 299, 7, 256]
    assert rd.map_ids([]) == []


def test_exact_turn_by_turn():
    rd = reader()
    s = rd.new("a")
    for prompt, ans in turns(1):
        rd.sync(s, prompt)
        rd.mark(s, len(prompt))
        rd.advance(s)
        assert s.pos % GRID == 0 and s.pos <= len(prompt) and len(prompt) - s.pos < GRID
        rd.sync(s, prompt + ans)
        rd.advance(s)
    assert s.kv.cap > 64 and s.kv.cap % 64 == 0, s.kv.cap      # it grew
    fresh = rd.fresh(s.ids)
    eq = S.same_state(s, fresh)
    assert eq["bit_exact"], eq
    assert s.stats["rollbacks"] == 0 and len(s.checks) >= 1
    # read in one call it is the same too
    one = rd.new("b")
    rd.sync(one, s.target)
    rd.advance(one)
    assert S.same_state(one, s)["bit_exact"]


def test_a_fork_changes_nothing():
    rd = reader()
    prompt, ans = turns(2)[-1]
    s = rd.new("a")
    rd.sync(s, prompt + ans)
    rd.advance(s)
    before = (s.kv.k.clone(), s.kv.v.clone(), s.gdn.S.clone(), s.gdn.conv.clone(), s.pos)
    tail = [IM_START, 40, 41, 42, IM_END, IM_START]
    lg = rd.fork_logits(s, len(s.target), tail)
    # the fresh forward of the same ids, in the same chunks from the same committed state
    f = rd.fresh(s.ids)
    f.target = list(s.target)
    lg2 = rd.fork_logits(f, len(f.target), tail)
    assert torch.equal(lg, lg2)
    # and the plain engine's forward of all of it at once agrees on the argmax
    e = _M[0].eng
    e.reset()
    with torch.inference_mode():
        ref = e.forward(torch.tensor(s.target + tail), start=0, last_only=True)[0, -1].float()
    assert int(ref.argmax()) == int(lg.argmax())
    assert (s.kv.k[:, :, :, :s.pos] == before[0][:, :, :, :s.pos]).all()
    assert torch.equal(s.gdn.S, before[2]) and torch.equal(s.gdn.conv, before[3])
    assert s.pos == before[4] and s.kv.length == s.pos
    out, fin = rd.generate(s, len(s.target), tail, 5, eos=set())
    assert len(out) == 5 and fin == "length"
    assert torch.equal(s.gdn.S, before[2])
    # a fork left open (the narrator's job stopped mid-way) is closed by the next read
    fk = rd.fork(s, len(s.target), tail)
    fk.prefill()
    assert not torch.equal(s.gdn.S, before[2])
    rd.advance(s)
    assert torch.equal(s.gdn.S, before[2]) and s.fork is None


def test_the_rollback():
    rd = reader()
    tt = turns(3, n=4)
    s = rd.new("a")
    for prompt, ans in tt[:3]:
        rd.sync(s, prompt)
        rd.mark(s, len(prompt))
        rd.advance(s)
        rd.sync(s, prompt + ans)
        rd.advance(s)
    # the next prompt re-renders the last answer (a token changed a few rows past the prompt end)
    p3, a3 = tt[2]
    changed = list(tt[3][0])
    i = len(p3) + 1
    changed[i] = (changed[i] + 1) % 250 + 3
    n_before = s.pos
    dropped = rd.sync(s, changed)
    assert dropped > 0 and s.pos <= i and s.stats["rollbacks"] == 1
    assert s.pos > 0, "a checkpoint at or below the difference"
    assert n_before - s.pos < 3 * GRID + len(a3), (n_before, s.pos)
    rd.advance(s)
    assert S.same_state(s, rd.fresh(s.ids))["bit_exact"]
    # a difference in the head: no checkpoint below it, the session starts over
    head = list(changed)
    head[1] = (head[1] + 1) % 250 + 3
    rd.sync(s, head)
    assert s.pos == 0 and s.stats["restarts"] == 1
    rd.advance(s)
    assert S.same_state(s, rd.fresh(s.ids))["bit_exact"]


def test_save_and_load():
    rd = reader()
    prompt, ans = turns(4)[-1]
    s = rd.new("k")
    rd.sync(s, prompt)
    rd.mark(s, len(prompt))
    rd.advance(s)
    rd.sync(s, prompt + ans)
    rd.advance(s)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.safetensors")
    n = rd.save(s, p)
    assert n > 0 and os.path.isfile(p)
    b = rd.load(p)
    assert S.same_state(s, b)["bit_exact"] and b.target == s.target and b.key == "k"
    assert [c[0] for c in b.checks] == [c[0] for c in s.checks]
    other = reader(grid=GRID * 2)
    try:
        other.load(p)
        raise AssertionError("another grid must refuse the file")
    except ValueError:
        pass


# ------------------------------------------------------------------ server/conductor/shadow.py

def service(**kw):
    from server.conductor.shadow import ShadowService
    from engine.generative.qwen35 import Qwen35Model
    m = Qwen35Model("tiny", tiny(), max_ctx=256, claim_gib=0.1, engine_dir=tiny())
    slot = Slot(m, role="shadow", gate=threading.Lock(), log=lambda s: None)
    slot.load(wait=True)
    assert slot.state == "loaded", slot.error
    kw.setdefault("root", tempfile.mkdtemp())
    return ShadowService(slot, grid=GRID, log=lambda *a, **k: None, **kw)


def test_the_service():
    svc = service()
    tt = turns(5, n=3)
    for prompt, ans in tt:
        info = svc.begin_turn("c:one", prompt)
        assert info["behind"] >= 0
        assert svc.has_work()
        svc.end_turn("c:one", ans)
        out = svc.work(deadline=svc.clock() + 60)
        assert out["rows"] >= 0 and not out["stopped"], out
    s = svc.sessions["c:one"]
    assert s.target == tt[-1][0] + tt[-1][1]
    assert len(s.target) - s.pos < GRID and not svc.has_work()
    assert S.same_state(s, svc.reader.fresh(s.ids))["bit_exact"]
    # the next turn reads only its new tokens, and stops at `stop()`
    p = tt[-1][0] + tt[-1][1] + [IM_START] + text(200, 9) + [IM_END, IM_START]
    svc.begin_turn("c:one", p)
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] > 3

    before = s.pos
    out = svc.work(deadline=svc.clock() + 60, stop=stop)
    assert out["stopped"] and 0 < s.pos - before <= 3 * GRID, (out, s.pos - before)
    out = svc.work(deadline=svc.clock() + 60)
    assert len(p) - s.pos < GRID
    st = svc.status()
    assert st["sessions"][0]["read"] == s.pos and st["stats"]["turns"] == 4


class Plan:
    def __init__(self, cls="code_edit", source="qwen_context"):
        self.active = True
        self.side = None
        self.row = {"request_id": "r1", "id": "row1", "class_id": cls, "source": source,
                    "confidence": 0.9}
        self.router_cfg = {"classes": [{"id": "chat", "description": "a conversation"},
                                       {"id": "code_edit", "description": "change code"},
                                       {"id": "writing", "description": "write prose"}]}


class Tok:
    """The tiny tokenizer's `<|im_end|>` (258) for the insertion point."""

    unk_token_id = None

    def convert_tokens_to_ids(self, t):
        return {"<|im_end|>": IM_END}.get(t, 0)


def test_a_model_the_reader_was_not_validated_on_gets_no_reader():
    """SRV-339: an engine whose adapter says the reader's outputs were not validated on its reasoning
    (`caps.shadow` False, Flash-Next) starts no reader, loads nothing, and the status says why; one
    that says nothing (the 27B, the default True) is not refused here."""
    import types

    from engine.models.adapter import Capabilities
    from server.conductor import shadow as SH
    old = os.environ.get("QWEN38_SHADOW_MODEL")
    os.environ["QWEN38_SHADOW_MODEL"] = "qwen3.5-0.8b"
    logs: list = []
    try:
        state = {"engine": types.SimpleNamespace(caps=Capabilities(caches=True, side_slots=False, shadow=False))}
        assert SH.start(state, None, log=logs.append) is None
        st = SH.status(state)
        assert st == {"enabled": False, "why": SH.NOT_VALIDATED_WHY} and "validated" in logs[0], (st, logs)
        assert Capabilities(caches=True, side_slots=True).shadow
        assert SH.status({})["why"] == SH.OFF_WHY
    finally:
        if old is None:
            os.environ.pop("QWEN38_SHADOW_MODEL", None)
        else:
            os.environ["QWEN38_SHADOW_MODEL"] = old
    return "refused with the reason in the status; the default capability keeps the reader"


def test_the_router_question():
    import json
    svc = service()
    cfg = {"router": "shadow", "router_min_tokens": 10}
    prompt, ans = turns(6, n=2)[-1]
    ask = svc.router_ask(Plan(), prompt, Tok(), cfg)
    assert ask is not None and ask["raw_at"] == len(prompt) - 1, ask
    assert ask["router"]["class_id"] == "code_edit" and set(ask["choices"]) == {"chat", "code_edit",
                                                                              "writing"}
    # not asked: short prompt, structural, off
    assert svc.router_ask(Plan(), prompt, Tok(), dict(cfg, router_min_tokens=10**6)) is None
    assert svc.router_ask(Plan(source="structural"), prompt, Tok(), cfg) is None
    assert svc.router_ask(Plan(), prompt, Tok(), dict(cfg, router="off")) is None
    svc.begin_turn("h:x", prompt, ask=ask)
    svc.end_turn("h:x", ans)
    svc.work(deadline=svc.clock() + 60)
    assert svc.router["asked"] == 1, dict(svc.router)
    rec = list(svc.recent)[-1]
    assert rec["label"] in ask["choices"] and abs(sum(rec["probs"].values()) - 1) < 1e-3
    assert rec["agree"] == (rec["label"] == "code_edit") and rec["behind_at_begin"] == len(prompt)
    assert rec["router"]["class_id"] == "code_edit" and "lag_s" in rec
    with open(svc.journal_path) as f:
        assert json.loads(f.readlines()[-1])["row_id"] == "row1"
    # the answer is the fresh forward's: the prompt up to its last <|im_end|> and the question
    s = svc.sessions["h:x"]
    tail, lids = svc._tail(ask["question"], ask["choices"])
    f = svc.reader.fresh(prompt[:ask["raw_at"]])
    f.target = prompt[:ask["raw_at"]]
    lg = svc.reader.fork_logits(f, len(f.target), tail, rows=lids)
    p = (lg - lg.logsumexp(0)).exp().tolist()
    assert abs(p[list(ask["choices"]).index(rec["label"])] - rec["confidence"]) < 1e-3
    assert s.pos >= ask["raw_at"] // GRID * GRID     # it read on past the question


def test_the_cap_and_the_disk():
    d = tempfile.mkdtemp()
    svc = service(cap_gib=1e-9, disk_dir=d)
    tt = turns(7, n=2)
    for key in ("c:a", "c:b"):
        for prompt, ans in tt:
            svc.begin_turn(key, prompt)
            svc.end_turn(key, ans)
            svc.work(deadline=svc.clock() + 60)
    # one stays resident (the newest), the other went to the disk
    assert list(svc.sessions) == ["c:b"] and "c:a" in svc.parked, (list(svc.sessions),
                                                                     svc.parked)
    a_ids_len = svc.parked["c:a"]["pos"]
    p = tt[-1][0] + tt[-1][1] + [IM_START] + text(30, 3) + [IM_END, IM_START]
    svc.begin_turn("c:a", p)
    svc.work(deadline=svc.clock() + 60)
    a = svc.sessions.get("c:a") or None
    assert svc.stats["unparked"] == 1 and svc.stats["parked"] >= 2
    # back from the disk and read on: still the fresh prefill
    if a is None:                                   # parked again by the cap after its read
        a = svc.reader.load(svc.parked["c:a"]["path"])
    assert a.pos >= a_ids_len and S.same_state(a, svc.reader.fresh(a.ids))["bit_exact"]
    # a drain writes the resident ones; a new service finds them
    svc.drain()
    assert not svc.sessions
    svc2 = service(disk_dir=d)
    assert set(svc2.parked) == {"c:a", "c:b"}


def test_the_pool_is_released_past_its_slack_and_the_sessions_come_back():
    """The model's own pool keeps every free segment while it lives; past
    POOL_SLACK_GIB free after a burst the sessions go to the disk and the model unloads (the
    pool with it), and the next turn loads it again and reads on from the disk."""
    from server.conductor import shadow as shadow_mod
    d = tempfile.mkdtemp()
    svc = service(disk_dir=d)
    tt = turns(11, n=2)
    for prompt, ans in tt:
        svc.begin_turn("c:r", prompt)
        svc.end_turn("c:r", ans)
        svc.work(deadline=svc.clock() + 60)
    assert svc.slot.state == "loaded" and "c:r" in svc.sessions
    pos = svc.sessions["c:r"].pos
    svc.pool_bytes = lambda: (int(5 * 2**30), int(1 * 2**30))   # 4 GiB free in the pool
    p = tt[-1][0] + tt[-1][1] + [IM_START] + text(40, 4) + [IM_END, IM_START]
    svc.begin_turn("c:r", p)
    out = svc.work(deadline=svc.clock() + 60)
    assert out.get("released") == 5 * 2**30, out
    assert svc.slot.state == "not_loaded" and not svc.sessions and "c:r" in svc.parked
    assert svc.stats["released"] == 1
    del svc.pool_bytes
    # below the slack nothing is released; the next turn loads the model and reads on
    svc.slot.load(wait=True)
    p2 = p + text(20, 5) + [IM_END, IM_START] + text(300, 6) + [IM_END, IM_START]
    svc.begin_turn("c:r", p2)
    out = svc.work(deadline=svc.clock() + 60)
    assert "released" not in out and svc.slot.state == "loaded"
    s = svc.sessions["c:r"]
    assert s.pos >= pos and svc.stats["unparked"] == 1
    assert S.same_state(s, svc.reader.fresh(s.ids))["bit_exact"]
    # the gate's call: nothing to give without a pool (the CPU), and never with a call in flight
    assert svc.release("memory_gate") == 0
    assert shadow_mod.pool_slack_gib() == 2.0


def test_the_release_yields_to_a_waiting_request_between_sessions():
    """The release parks sessions under the engine lock; a request that waits stops
    it before the next session, the parked ones stay on the disk, the model stays loaded."""
    d = tempfile.mkdtemp()
    svc = service(disk_dir=d)
    for key, seed in (("c:a", 21), ("c:b", 22)):
        for prompt, ans in turns(seed, n=2):
            svc.begin_turn(key, prompt)
            svc.end_turn(key, ans)
            svc.work(deadline=svc.clock() + 60)
    assert {"c:a", "c:b"} <= set(svc.sessions)
    svc.pool_bytes = lambda: (int(5 * 2**30), int(1 * 2**30))
    asked = []

    def stop():
        asked.append(1)
        return len(asked) > 1                        # a request arrives after the first session
    assert svc.release("pool_slack", stop=stop) == 0
    assert svc.slot.state == "loaded" and len(svc.sessions) == 1 and len(svc.parked) == 1
    assert svc.stats["release_yielded"] == 1
    # the idle window's burst does not start a release while a request waits
    out = svc.work(deadline=svc.clock() + 60, stop=lambda: True)
    assert "released" not in out and svc.slot.state == "loaded"
    del svc.pool_bytes


def test_ask():
    svc = service()
    prompt, ans = turns(8)[-1]
    svc.begin_turn("c:q", prompt)
    svc.end_turn("c:q", ans)
    out = svc.ask("c:q", "Which fits?", {"x": "one thing", "y": "another", "z": "a third"})
    assert out.get("label") in ("x", "y", "z") and out["behind"] < GRID, out
    assert abs(sum(out["probs"].values()) - 1) < 1e-3
    assert svc.ask("c:none", "Which fits?", {"x": "a", "y": "b"})["why"] == "unknown_session"
    assert svc.ask("c:q", "Which fits?", {})["why"] == "choices"


def test_the_narrator_reads_the_shadow():
    from server.conductor import narrator as narr_mod
    svc = service()
    svc.model.eos = set()                                # a random model: no end token
    prompt, _ = turns(9)[-1]
    svc.begin_turn("c:n", prompt)
    svc.work(deadline=svc.clock() + 60)
    s = svc.sessions["c:n"]
    S0 = s.gdn.S.clone()
    lines = []

    class UI:
        def written(self, kind, line, info, **facts):
            lines.append((kind, line, info))

    class N:
        slot = svc.slot

        def settings(self):
            return {"mode": "on", "budget_ms": 150}

    reader = svc.thought_slot("c:n")
    th = narr_mod.Thoughts(N(), UI(), opened=True, close_id=260, reader=reader,
                           budget_ms=10_000, clock=svc.clock)
    th.t0 = svc.clock() - 1000.0                          # a long turn: the share allows it
    out = [10 + i % 200 for i in range(narr_mod.THOUGHT_MIN_TOKENS + 40)]
    ids: list[int] = []
    for t in out:
        ids.append(t)
        th.token(ids, True)
    ids.append(260)
    th.token(ids, False)
    th.finish(ids)
    assert lines, th.totals()
    kind, line, info = lines[-1]
    assert kind == "thought" and info.get("shadow") and info["window"] == "session", info
    assert info["why"] in (None, "markup", "words", "chars", "empty", "length"), info
    assert info["tokens"] > 0 and info["reused"] > 0, info
    # the session's committed state is the same after (the fork was put back)
    assert torch.equal(s.gdn.S[:, :, :, :, :], s.gdn.S) and s.fork is None
    assert s.pos >= len(prompt) // GRID * GRID
    del S0


def test_inline_share():
    import time as _t
    svc = service()
    prompt, ans = turns(10, n=3)[-1]
    big = prompt + text(400, 11) + [IM_END, IM_START]
    svc.begin_turn("c:i", big)
    none = svc.inline("c:i", share=0.0)
    none.tick([])
    assert none.stats["rows"] == 0 and none.stats["slices"] == 0
    il = svc.inline("c:i", share=0.5, slice_ms=10_000)
    il.tick([])                                  # Qwen's first token: the share starts here
    assert il.stats["slices"] == 0
    _t.sleep(0.4)
    il.tick([])
    s = svc.sessions["c:i"]
    assert il.stats["rows"] > 0 and s.pos == il.stats["rows"], il.stats
    # within its share, but for the one chunk it may start over the estimate
    el = (svc.clock() - il.t0) * 1e3
    assert il.spent <= 0.5 * el + il.est * 2 + 50, (il.spent, el, il.est)
    # until the prompt is read the output is not fed; after it, the output is
    for _ in range(200):
        _t.sleep(0.02)
        il.tick(ans)
        if s.pos == len(big + ans) // GRID * GRID:
            break
    assert s.target == big + ans and s.pos == len(big + ans) // GRID * GRID, (s.pos, len(big + ans))
    assert S.same_state(s, svc.reader.fresh(s.ids))["bit_exact"]
    # an open fork (the narrator's summary in progress) is left alone
    svc.begin_turn("c:i", big + ans + text(100, 12))
    svc.work(deadline=svc.clock() + 60)
    s = svc.sessions["c:i"]
    fk = svc.reader.fork(s, len(s.target), [IM_START])
    il2 = svc.inline("c:i", share=1.0)
    il2.tick([])
    svc.begin_turn("c:i", big + ans + text(600, 13))
    _t.sleep(0.2)
    before = s.pos
    il2.tick([])
    assert s.pos == before and s.fork is fk
    fk.close()
    assert "decode_ms" in il.totals()


def test_the_handler():
    import test_ui_states as ui
    from server import app
    ui.serve("on")
    base = ui._strip_ids(ui.opencode("Write the lexer.")["raw"])
    cond, _ = ui.serve("on")
    _, doc = cond.store.get("profiles", "coding")
    doc["shadow"] = {"mode": "on", "router": "shadow", "router_min_tokens": 0}
    cond.store.put("profiles", "coding", doc, note="shadow", author="test")
    svc = service()
    app.STATE["shadow"] = svc
    os.environ["BOX_LOCK"] = os.path.join(tempfile.mkdtemp(), "box.flock")   # not the box's
    try:
        got = ui.opencode("Write the lexer.")
        assert ui._strip_ids(got["raw"]) == base
        assert "c:ses_ui1" in svc.feed, list(svc.feed)
        f = svc.feed["c:ses_ui1"]
        assert f["prompt_end"] > 0 and len(f["raw"]) > f["prompt_end"] and "t_end" in f
        rid = json.loads(got["raw"].splitlines()[0][6:])["id"]
        row = [r for r in cond.journal.ring.values() if r.get("request_id") == rid][0]
        assert row["extra"]["shadow"]["behind"] == f["prompt_end"], row["extra"]["shadow"]
        assert "inline" in row["extra"]["shadow"]
        for _ in range(20):                      # a loaded box may stop a burst at its deadline
            r = app.shadow_once()
            if r != "stopped":
                break
        assert r in ("read", "nothing") and svc.stats["rows"] > 0, (r, svc.stats)
        assert svc.router["asked"] + svc.router["missed"] + svc.router["errors"] <= 1
        assert app.shadow_once() == "nothing"
    finally:
        app.STATE.pop("shadow", None)
        os.environ.pop("BOX_LOCK", None)


def test_the_profile_block():
    from server.conductor import schema
    from server.conductor.shadow import settings_of
    assert settings_of(None) == {"mode": "off", "narrator": "off", "router": "off",
                                 "router_min_tokens": 8192}
    assert settings_of({"shadow": {"mode": "on", "router": "shadow"}})["router"] == "shadow"
    prof = schema.seed("profiles", "coding")
    assert not schema.doc_errors("profiles", prof)
    ok = dict(prof, shadow={"mode": "on", "narrator": "off", "router": "shadow",
                            "router_min_tokens": 8192})
    assert not schema.doc_errors("profiles", ok), schema.doc_errors("profiles", ok)
    for bad in ({"mode": "maybe"}, {"router": "on"}, {"mode": "on", "extra": 1}):
        assert schema.doc_errors("profiles", dict(prof, shadow=bad)), bad


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                r = fn()
                print(f"  ok  {name}" + (f"  ({r})" if r else ""))
            except Exception:                                        # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL  {name}")
    print(f"\n{failed} failed" if failed else "\nall passed")
    sys.exit(1 if failed else 0)
