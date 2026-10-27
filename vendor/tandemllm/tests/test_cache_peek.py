"""SRV-349: the cache peek, the prewarm and the usage frame's split (server/cache_peek.py), on the
CPU, for both models.

  1  the 27B (the random 4-layer engine) through the server's own loop: the peek names the tier and
     the line the next request is served from -- resident (its end anchor), nvme_spill after another
     session wrote it to the disk, none for a prompt nothing holds -- and the request that follows
     reports that source and reuses that many tokens; the tiers' counters do not move for a peek
  2  the prewarm reads a spilled conversation back while the engine is idle: the next request
     resumes at its end anchor (`end`, not `spill`) and writes the same tokens as without it; the
     other session went to the disk in its place, and the peek says so
  3  the prewarm does nothing while a request runs or waits, while the engine lock is held, when
     another conversation used the rows a moment ago, or for a chat RAM already holds; the peek
     answers None (409) while the lock is held
  4  the parked session: `ram_park`, and the next request comes back from the park
  5  Flash-Next (the tiny random model of tests/fn_tiny.py on its reference forward, behind the
     caches' adapter interface): resident, nvme_spill after a save, a prewarm, and the next prefill
     resumes from the rows read back, equal to a cold prefill -- no model name in the peek
  6  through the handler: `usage.tandem.parts` adds up (system + tools + history + images =
     prompt_tokens, plus `injected` = the prompt the engine read = cached + reread), the peek
     before a send matches the `tandem.cache` of that send, the peek and the prewarm routes answer,
     and a busy engine gets a 409
  7  the usage record's `tandem` object and the tier names, without the engine

Run: python tests/test_cache_peek.py
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import cache  # noqa: E402
from engine.models.adapter import CostModel  # noqa: E402
from server import app, cache_peek  # noqa: E402
from server.usage import RequestRecord  # noqa: E402
from test_cache import tokens  # noqa: E402
from test_end_anchor import BIG, CH, TOOL, ChatTok, Req, _state, turn  # noqa: E402
from test_session_spill import _spill  # noqa: E402


def _peek(ids, session=None):
    return app._cache_peek(list(ids), session)


def _counters(*tiers):
    return [copy.deepcopy(t.stats) for t in tiers]


# ------------------------------------------------------------------ 1. the 27B, the tiers

def _ab(d, prewarm: bool):
    """Session A (220 + 40 tokens) answered, then session B's request (not a guest) writes A to
    the NVMe; with `prewarm`, A is read back before its next request. Returns what the peeks
    said, A's next request and its prefill."""
    res = _state(park_bytes=0, guest_max=40)
    sp, t = _spill(d, res)
    prompt = tokens(220, seed=81)
    out1, _, _ = turn(prompt, 40, session="A")
    e = res.end["at"]
    nxt = prompt + out1 + TOOL
    seen = {"resident": _peek(nxt, "A"), "none": _peek(tokens(150, seed=99), "C")}
    other = tokens(200, seed=82)
    turn(other, 8, session="B")
    before = _counters(sp, res)
    seen["spill"] = _peek(nxt, "A")
    assert _counters(sp, res) == before, "a peek is not a lookup the dashboard counts"
    if prewarm:
        seen["prewarm"] = app._cache_prewarm(list(nxt), "A")
        seen["after"] = _peek(nxt, "A")
        seen["B"] = _peek(other + tokens(9, seed=83), "B")
    out2, pre2, _ = turn(nxt, 16, session="A")
    t.close()
    return seen, e, len(nxt), out2, pre2, res, sp


def test_the_27b_peek_names_the_tier_and_line_of_the_next_request():
    d = tempfile.mkdtemp(prefix="qse-peek-")
    try:
        seen, e, n, _, pre2, _, _ = _ab(d, prewarm=False)
        r = seen["resident"]
        assert (r["tier"], r["cached_tokens"], r["reread_tokens"], r["tokens"]) == \
            ("resident", e, n - e, n), r
        assert seen["none"]["tier"] == "none" and seen["none"]["cached_tokens"] == 0
        s = seen["spill"]
        assert s["tier"] == "nvme_spill" and s["cached_tokens"] == e and s["restore_ms"] is not None
        # the request that follows: served from the tier the peek named, from the same line
        assert pre2["kind"] == "spill" and pre2["reused"] == s["cached_tokens"], (pre2, s)
        assert s["reread_eta_ms"] is not None and s["reread_eta_ms"] > 0
        return f"resident at {e}, none, nvme_spill at {e} = the next request's spill restore"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 2. the prewarm

def test_the_prewarm_reads_a_spilled_chat_back_and_its_next_turn_resumes_at_the_end():
    d1, d2 = tempfile.mkdtemp(prefix="qse-peek-"), tempfile.mkdtemp(prefix="qse-peek-")
    try:
        _, e, _, ref, pre_ref, _, _ = _ab(d1, prewarm=False)
        seen, e2, _, out2, pre2, res, sp = _ab(d2, prewarm=True)
        assert e2 == e
        w = seen["prewarm"]
        assert w["prewarmed"] is True and w["tier"] == "nvme_spill" and w["restore"]["tokens"] >= e, w
        assert seen["after"]["tier"] == "resident" and seen["after"]["cached_tokens"] == e
        # B, the conversation in place before, went to the disk (it was long enough)
        assert seen["B"]["tier"] == "nvme_spill", seen["B"]
        assert pre2["kind"] == "end" and pre2["reused"] == e, pre2
        assert pre_ref["kind"] == "spill"
        assert out2 == ref, "the same tokens as the turn that read the disk itself"
        return f"read back {w['restore']['tokens']} tokens in {w['restore']['ms']} ms; the turn resumed at the end ({e})"
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


# ------------------------------------------------------------------ 3. when the prewarm does nothing

def test_the_prewarm_yields_to_requests_and_to_another_conversation_in_use():
    d = tempfile.mkdtemp(prefix="qse-peek-")
    try:
        res = _state(park_bytes=0, guest_max=40)
        sp, t = _spill(d, res)
        prompt = tokens(220, seed=84)
        out1, _, _ = turn(prompt, 40, session="A")
        nxt = prompt + out1 + TOOL
        # a chat RAM holds needs nothing
        assert app._cache_prewarm(list(nxt), "A")["why"] == "not_on_nvme"
        turn(tokens(200, seed=85), 8, session="B")
        # a request running or waiting: not even a lookup
        for key in ("running", "waiting"):
            app.INFLIGHT[key] += 1
            try:
                assert app._cache_prewarm(list(nxt), "A") is None
            finally:
                app.INFLIGHT[key] -= 1
        # the engine lock held and a request waiting for it: peek and prewarm answer None at once,
        # the route's 409
        held, done = threading.Event(), threading.Event()

        def hold(s=5):
            with app.LOCK:
                held.set()
                done.wait(s)
        th = threading.Thread(target=hold)
        th.start()
        held.wait(5)
        app.INFLIGHT["waiting"] += 1
        t0 = time.monotonic()
        try:
            assert _peek(nxt, "A") is None
            assert time.monotonic() - t0 < 0.5, "no wait while a request waits"
        finally:
            app.INFLIGHT["waiting"] -= 1
            done.set()
            th.join()
        # held by a request's tail (its bookkeeping after the answer went out) or the engine's
        # idle work after it (live: right after a turn, 1,339 ms): they wait for it, and answer
        # None past PEEK_IDLE_WAIT_S
        held.clear(); done.clear()
        app.INFLIGHT["running"] += 1
        th = threading.Thread(target=hold, args=(0.3,))
        th.start()
        held.wait(5)
        try:
            p = _peek(nxt, "A")
        finally:
            th.join()
            app.INFLIGHT["running"] -= 1
        assert p is not None and p["tier"] in cache_peek.ORDER + ("none",), p
        held.clear(); done.clear()
        real_wait = app.PEEK_IDLE_WAIT_S
        app.PEEK_IDLE_WAIT_S = 0.2
        th = threading.Thread(target=hold)
        th.start()
        held.wait(5)
        try:
            assert _peek(nxt, "A") is None and app._cache_prewarm(list(nxt), "A") is None
        finally:
            app.PEEK_IDLE_WAIT_S = real_wait
            done.set()
            th.join()
        # B used the rows a moment ago: they stay
        app.STATE["last_request_end"] = time.time()
        w = app._cache_prewarm(list(nxt), "A")
        assert w["prewarmed"] is False and w["why"] == "resident_in_use", w
        assert res.owner == "B" and sp.stats["restores"] == 0
        # a read longer than MAX_MS is not started
        app.STATE["last_request_end"] = 0
        real = cache_peek.MAX_MS
        cache_peek.MAX_MS = -1.0
        try:
            assert app._cache_prewarm(list(nxt), "A")["why"] == "too_long"
        finally:
            cache_peek.MAX_MS = real
        assert app._cache_prewarm(list(nxt), "A")["prewarmed"] is True
        t.close()
        return "busy, locked, in use, too long: nothing read; idle work waited for; idle: read back"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 4. the park

def test_a_parked_session_peeks_as_ram_park_and_comes_back_from_the_park():
    res = _state(park_bytes=BIG, guest_max=40)
    prompt = tokens(160, seed=86)
    out1, _, _ = turn(prompt, 24, session="A")
    turn(tokens(150, seed=87), 8, session="B")
    assert res.stats["parks"] >= 1
    nxt = prompt + out1 + TOOL
    p = _peek(nxt, "A")
    _, pre, _ = turn(nxt, 8, session="A")
    assert p["tier"] == "ram_park" and pre["kind"] in ("park", "end", "resident"), (p, pre)
    assert pre["reused"] == p["cached_tokens"], (p, pre)
    # the usage frame names the tier the rows came back from, as the peek does (live: the peek said
    # ram_park and the send said resident, the anchor kind after the swap)
    assert pre.get("via") == "park", pre
    r = RequestRecord("r", "chat", False)
    r.prompt_tokens, r.parts = len(nxt), {"system": 0, "tools": 0, "images": 0}
    r.absorb_prefill(pre)
    assert r.usage()["tandem"]["cache"]["tier"] == "ram_park", r.usage()["tandem"]["cache"]
    return f"ram_park at {p['cached_tokens']}; the next request reused {pre['reused']} ({pre['kind']})"


def test_a_parked_end_brought_back_before_the_replay_reports_the_park():
    """Live (rc-0.8.0): the parked chat's end comes back in place before the handler's replay
    (`_spill_before_replay`: `unpark_end`), so the prefill resumed at the end anchor and the usage
    frame said resident where the peek had said ram_park. The return is recorded (`park_returned`)
    and the usage frame's tier is the park's."""
    d = tempfile.mkdtemp(prefix="qse-peek-")
    try:
        res = _state(park_bytes=BIG, guest_max=40)
        sp, t = _spill(d, res)
        prompt = tokens(160, seed=88)
        out1, _, _ = turn(prompt, 24, session="A")
        turn(tokens(150, seed=89), 8, session="B")
        nxt = prompt + out1 + TOOL
        p = _peek(nxt, "A")
        app._spill_before_replay(list(nxt))
        assert app.STATE.get("park_returned") is True, res.stats
        _, pre, _ = turn(nxt, 8, session="A")
        assert p["tier"] == "ram_park" and pre.get("via") == "park" and pre["reused"] == p["cached_tokens"], (p, pre)
        t.close()
        return f"ram_park at {p['cached_tokens']}; the send came back via the park"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 5. Flash-Next

class _FNCacheState:
    """What the caches keep of the tiny Flash-Next, as FlashNextCacheState names it: the GDN
    states and conv tails, the PLE conv state and its token context (a fixed buffer), K, V and the
    indexer keys of the attention layer."""

    def __init__(self, e):
        self.e = e

    def sections(self) -> dict:
        e, st = self.e, self.e.st
        for li, buf in e.ctx.items():
            c = st.ple_ctx.get(li)
            buf.copy_(c.view(1, -1) if c is not None else torch.full_like(buf, e.cfg.eos_token_id))
        out = {f"gdn.S.{li}": st.S[li] for li in sorted(st.S)}
        out.update({f"gdn.conv.{li}": st.conv[li] for li in sorted(st.conv)})
        out.update({f"ple.conv.{li}": st.ple_conv[li] for li in sorted(st.ple_conv)})
        out.update({f"ple.ctx.{li}": e.ctx[li] for li in sorted(e.ctx)})
        return out

    def views(self) -> list:
        st = self.e.st
        out = []
        for li in sorted(st.k):
            out += [(f"k.{li}", st.k[li], 1), (f"v.{li}", st.v[li], 1)]
            if li in st.ik:
                out.append((f"ik.{li}", st.ik[li], 0))
        return out


class _FNKV:
    def __init__(self, st, max_len: int):
        self.st, self.max_len, self.pages = st, max_len, None

    @property
    def length(self) -> int:
        return self.st.length

    @length.setter
    def length(self, v) -> None:
        self.st.length = int(v)


class FNTiny:
    """The tiny random Flash-Next on its reference forward, in the engine interface the caches use
    (the adapter's cold verbs): `cache_state`, `forward`, `resume_at`, `reset`, `kv.length`."""

    def __init__(self, seed: int = 3, max_len: int = 256):
        from engine.models.flashnext.ref import FlashNextRef, FNState
        from fn_tiny import tiny
        self.cfg, W, tables = tiny(seed=seed)
        self.ref = FlashNextRef(self.cfg, W, tables, dtype=torch.float32)
        self.st = FNState(cap=max_len)
        self.ref.forward(torch.tensor([5]), self.st, head="none")   # the buffers, once
        self.ctx = {li: torch.full((1, self.ref.hash[li].context_len), self.cfg.eos_token_id,
                                   dtype=torch.long) for li in self.cfg.ple_layers}
        self.reset()
        self.kv = _FNKV(self.st, max_len)
        self.hidden_post_norm = None

    def cache_state(self):
        return _FNCacheState(self)

    def cost(self) -> CostModel:
        s = self.cache_state()
        sec = sum(t.numel() * t.element_size() for t in s.sections().values())
        row = sum(t.numel() // t.shape[ax] * t.element_size() for _, t, ax in s.views())
        return CostModel(section_bytes=sec, view_bytes_per_token=row, parts={"recurrent": sec})

    def settle(self):
        pass

    def discard_pending(self):
        pass

    def trim_pending(self, n):
        return n == self.st.length

    def window_ok(self, i):
        return False

    def resume_at(self, n):
        for li, buf in self.ctx.items():
            self.st.ple_ctx[li] = buf.view(1, -1).clone()
        self.st.length = int(n)
        self.st.clamp_pool()

    def reset(self):
        self.st.reset_inplace()

    @torch.no_grad()
    def forward(self, ids, start, last_only=True):
        assert start == self.st.length, (start, self.st.length)
        lg = self.ref.forward(ids.view(-1), self.st, head="last" if last_only else "all")
        return lg.view(1, -1, lg.shape[-1])


def _fn_cold(ids):
    eng = FNTiny()
    lg, _, _ = cache.prefill(eng, None, ids, "cpu", chunk=CH)
    return lg


def test_flashnext_peek_prewarm_and_resume_through_the_adapter():
    d = tempfile.mkdtemp(prefix="qse-peek-fn-")
    try:
        g = torch.Generator().manual_seed(7)
        ids = torch.randint(4, 60, (100,), generator=g).tolist()
        nxt = ids + torch.randint(4, 60, (20,), generator=g).tolist()
        eng = FNTiny()
        res = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True)
        app.STATE.clear()
        app.STATE.update(engine=eng, drafter=None, resident=res, state_store=None, disk_cache=None,
                         max_len=256)
        sp, t = _spill(d, res)
        cache.prefill(eng, None, ids, "cpu", chunk=CH, resident=res, conv_id="A")
        p = _peek(nxt, "A")
        assert p["tier"] == "resident" and p["cached_tokens"] == 96, p
        assert sp.save_resident(res, eng, None, "evict") is not None, sp.stats
        res.clear()
        for _, v, _ in eng.cache_state().views():
            v.zero_()
        eng.reset()
        p = _peek(nxt, "A")
        assert p["tier"] == "nvme_spill" and p["cached_tokens"] == 96, p
        w = app._cache_prewarm(list(nxt), "A")
        assert w["prewarmed"] is True, w
        info = {}
        lg, reused, _ = cache.prefill(eng, None, nxt, "cpu", chunk=CH, resident=res, conv_id="A",
                                      info=info)
        assert info["kind"] == "resident" and reused == 96, (info, reused)
        ref = _fn_cold(nxt)
        assert torch.allclose(lg, ref, atol=1e-5, rtol=0), (lg - ref).abs().max()
        t.close()
        return "resident 96, nvme_spill 96, prewarmed; the next prefill resumed at 96, equal to a cold one"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 6. through the handler

class ToolTok(ChatTok):
    """ChatTok with a system turn (marker 3) and the tool schemas' names in a block of their own
    (marker 5) after the system turns, as the Qwen template writes them into its system block."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tools=None, **kw):
        ids = []
        rest = [m for m in messages if m["role"] not in ("system", "developer")]
        for m in messages:
            if m["role"] in ("system", "developer"):
                ids += [3] + self.enc(m["content"]) + [4]
        if tools:
            ids += [5] + [6 + ord(c) % 50 for t in tools for c in t["function"]["name"]] + [4]
        for m in rest:
            ids += [1 if m["role"] == "user" else 2] + self.enc(m["content"]) + [4]
        ids += [2] if add_generation_prompt else []
        if kw.get("tokenize") is False:
            return self.decode(ids)
        return {"input_ids": torch.tensor([ids])}


TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
         for n in ("read_file", "write_file", "bash")]


def _post(path, body, session="chat-1"):
    r = Req(path, body)
    r.headers["x-session-id"] = session
    head, raw = r.response()
    return int(head.split()[1]), json.loads(raw)


def test_through_the_handler_the_parts_add_up_and_the_peek_matches_the_send():
    app.cache_peek.RATES._q.clear()
    res = _state()
    tok = app.STATE["tok"] = ToolTok()
    system = tok.decode(tokens(30, seed=90))
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": tok.decode(tokens(60, seed=91))}]
    body = {"messages": msgs, "tools": TOOLS, "max_tokens": 24, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    st, r1 = _post("/v1/chat/completions", body)
    assert st == 200, r1
    u = r1["usage"]
    tp = u["tandem"]["parts"]
    names = len("read_filewrite_filebash")
    assert tp["system"] == 30 + 2 and tp["tools"] == names + 2 and tp["images"] == 0, tp
    assert tp["system"] + tp["tools"] + tp["history"] + tp["images"] == u["prompt_tokens"]
    c = u["tandem"]["cache"]
    assert c["tier"] == "none" and c["cached"] + c["reread"] == u["prompt_tokens"] + tp["injected"]
    assert u["tandem"]["harness"] is None              # no conductor: no plan
    # the next turn: peek first, then send; the peek names what the send then reports
    out = list(app.STATE["last_ctx"][-24:])
    msgs2 = msgs + [{"role": "assistant", "content": tok.decode(out)},
                    {"role": "user", "content": tok.decode(TOOL)}]
    body2 = dict(body, messages=msgs2)
    st, p = _post("/v1/cache/peek", dict(body2, session="chat-1"))
    assert st == 200 and p["session"] == "chat-1", p
    st, w = _post("/v1/cache/prewarm", dict(body2, session="chat-1"))
    assert st == 200 and w["prewarmed"] is False and w["why"] == "not_on_nvme", w
    st, r2 = _post("/v1/chat/completions", body2)
    c2 = r2["usage"]["tandem"]["cache"]
    assert p["tier"] == c2["tier"] == "resident" and p["cached_tokens"] == c2["cached"], (p, c2)
    assert p["reread_tokens"] == c2["reread"] and p["tokens"] == r2["usage"]["prompt_tokens"]
    assert r2["timings"]["cache_n"] == c2["cached"]
    # bad bodies, and a busy engine
    assert _post("/v1/cache/peek", {"messages": []})[0] == 400
    held, done = threading.Event(), threading.Event()

    def hold():
        with app.LOCK:
            held.set()
            done.wait(5)
    th = threading.Thread(target=hold)
    th.start()
    held.wait(5)
    app.INFLIGHT["waiting"] += 1
    try:
        st, busy = _post("/v1/cache/peek", dict(body2, session="chat-1"))
        assert st == 409 and busy["error"]["type"] == "busy", busy
    finally:
        app.INFLIGHT["waiting"] -= 1
        done.set()
        th.join()
    return (f"parts {tp} add up to {u['prompt_tokens']}; peek {p['tier']} {p['cached_tokens']} = "
            f"the send's {c2['tier']} {c2['cached']}")


# ------------------------------------------------------------------ 6b. the router's notes

def test_the_peek_replays_the_sessions_notes_and_the_harness_names_the_packages():
    """The router's note of turn 1 sits in the rows; turn 2's client copy lacks it. The peek renders
    the copy with the session's notes replayed as the request will (`Router.replayed_for` = the
    plan's own `replayed_ids`), without creating a session; the usage frame's harness names the
    class, the effort and the packages in the prompt with their tokens."""
    from test_snapshots import S, body, fresh, served
    cond = fresh(None)
    tok = app.STATE["tok"]
    s = S()
    p1, ids1 = served(cond, s.msgs, sid="chat-x")
    assert p1.changed and p1.spans, p1.spans
    h = app._harness_of(p1)
    names = [n for sp in p1.spans for n in sp[3]]
    assert h["class"] == "code_edit" and h["effort"] == p1.row["effort"], h
    assert [x["name"] for x in h["packages"]] == list(dict.fromkeys(names))
    # a package in a note: its name from the note's span, its tokens from its text
    note = types.SimpleNamespace(parts=[{"kind": "package", "name": "coding", "text": "abcdef"},
                                        {"kind": "rules", "name": "r", "text": "xyz"}])
    fake = types.SimpleNamespace(active=True, row={"class_id": "code_edit", "effort": "xhigh"},
                                 session=types.SimpleNamespace(notes=[note]), injected=[],
                                 spans=[(10, 30, ["packages", "rules"], ["coding"])])
    assert app._harness_of(fake) == {"class": "code_edit", "effort": "xhigh",
                                     "packages": [{"name": "coding", "tokens": 6}],
                                     "memory": {"mode": "off", "why": "unset", "index_notes": 0,
                                                "index_tokens": 0, "bodies": 0,
                                                "body_tokens": 0, "card_tokens": 0}}
    msgs2 = s.step(("bash", {"command": "ls"}, "a.txt"))
    client2 = app.build_prompt(body(msgs2), count="ids")
    n_sessions = len(cond.router._sessions)
    got = cond.router.replayed_for("c:chat-x", client2, tok)
    p2, ids2 = served(cond, msgs2, sid="chat-x")
    assert got == p2.replayed_ids(client2, tok) and len(got) > len(client2)
    assert ids2[:len(ids1) - 1] == got[:len(ids1) - 1], "the rows of turn 1, notes and all"
    assert cond.router.replayed_for("c:nobody", client2, tok) == client2
    assert len(cond.router._sessions) == n_sessions, "a peek creates no session"
    assert app._harness_of(None) is None
    return f"{len(got) - len(client2)} note tokens replayed; harness {h['class']} {h['effort']}"


def test_the_harness_counts_the_packages_of_notes_that_keep_no_text():
    """2026-10-14 (a phone's Context panel: eight harness packages, each "— tokens"): a note keeps
    its parts as {kind, name} with one text hash each, so `_harness_of` found no text and counted
    every package null. The count comes from the turn's block, from an earlier count of the same
    name and hash when the note is replayed, and from the library when its text has that hash."""
    from test_snapshots import S, fresh, served
    cond = fresh(None)
    _, pk = cond.store.get("packages")
    pk["packages"]["coding"] = {"title": "Coding", "kind": "package", "source": "inline",
                                "description": "How programming work must happen.",
                                "body": "# Coding core\n\nRun the tests after every step.",
                                "max_tokens": 3000, "enabled": True, "general": True}
    cond.store.put("packages", None, pk, note="a general package", author="t")
    _, r = cond.store.get("router")
    for c in r["classes"]:
        if c["id"] == "code_edit":
            c["policy"]["packages"] = ["coding"]
    cond.store.put("router", None, r, note="coding on code_edit", author="t")
    s = S()
    p1, _ = served(cond, s.msgs, sid="chat-tokens")
    assert p1.block and all("text" not in p for p in p1.block["parts"]), "the case: no texts"
    h1 = app._harness_of(p1)
    got1 = {x["name"]: x["tokens"] for x in h1["packages"]}
    assert got1 and all(isinstance(n, int) and n > 0 for n in got1.values()), h1
    for it in p1.block["items"]:
        if it["part"]["kind"] == "package":
            assert got1[it["part"]["name"]] == len(app.STATE["tok"](
                it["text"], add_special_tokens=False).input_ids)
    p2, _ = served(cond, s.step(("bash", {"command": "ls"}, "a.txt")), sid="chat-tokens")
    replayed = {x["name"]: x["tokens"] for x in app._harness_of(p2)["packages"]}
    assert {n: replayed[n] for n in got1 if n in replayed} == \
        {n: got1[n] for n in got1 if n in replayed} and replayed, (got1, replayed)
    # after a restart nothing is remembered: the library's text, when its hash is the note's
    app._PKG_TOKENS.clear()
    held = app.STATE.get("conductor")
    app.STATE["conductor"] = cond
    try:
        again = {x["name"]: x["tokens"] for x in app._harness_of(p2)["packages"]}
    finally:
        app.STATE["conductor"] = held
    assert again == replayed, (again, replayed)
    app._PKG_TOKENS.clear()
    assert app._package_tokens("nope", None, "0" * 12) is None
    # the resident report's owner tag: chatd finds its chat in the slot without the key shown
    import hashlib
    assert cache.owner_tag("chat-01J9Z3K7Q2W5E8R4T6Y1U3I5O7") == hashlib.sha256(
        b"chat-01J9Z3K7Q2W5E8R4T6Y1U3I5O7").hexdigest()[:16] and cache.owner_tag(None) is None
    return f"{len(got1)} packages counted: {sorted(got1.items())}"


# ------------------------------------------------------------------ 7. the record alone

def test_the_usage_records_tandem_object():
    r = RequestRecord("r", "chat", False)
    assert "tandem" not in r.usage(), "no rendered chat prompt, no tandem key"
    r.prompt_tokens, r.hidden_prompt = 84469 - 4413, 4413
    r.parts = {"system": 1834, "tools": 2210, "images": 0}
    r.harness = {"class": "code_edit", "effort": "xhigh",
                 "packages": [{"name": "coding", "tokens": 4413}]}
    r.absorb_prefill({"reused": 74200, "forwarded": 10269, "kind": "spill"})
    t = r.usage()["tandem"]
    assert t["parts"] == {"system": 1834, "tools": 2210, "history": 76012, "injected": 4413,
                          "images": 0}, t["parts"]
    assert sum(t["parts"].values()) == t["cache"]["cached"] + t["cache"]["reread"] == 84469
    assert t["cache"] == {"tier": "nvme_spill", "source": "spill", "cached": 74200,
                          "reread": 10269}
    assert [cache_peek.tier_of(s) for s in ("end", "side", "pin", "park", "prefix", "disk",
                                            "none", "response")] == \
        ["resident", "resident", "ram_park", "ram_park", "ram_snapshot", "disk_tier", "none",
         "response"]
    rates = cache_peek.Rates()
    rates.note(512, 100.0)                             # too short to say anything
    assert rates.rate() is None
    for n, ms in ((2048, 1000.0), (4096, 1000.0), (8192, 2000.0)):
        rates.note(n, ms)
    assert rates.rate() == 4096.0
    got = cache_peek.answer(list(range(100)), {"tier": "nvme_spill", "line": 90,
                                               "restore_ms": 5.0}, "s", 1000.0)
    assert got["reread_eta_ms"] == 15.0 and got["reread_tokens"] == 10
    return "the ticket's example frame, and the tier names"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<80} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<80} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
