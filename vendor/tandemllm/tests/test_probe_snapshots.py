"""Probe snapshots of Qwen's own rows at the router's decision points (ENG-181).

The test matrix:

  the ring (server/probe.py, a temp directory, a hidden width of 8):
    1  a request's points become fp16 records and index lines (request, journal row, session,
       class, effort, point, rows, weights), mode 0600, one journal row a request
                                                      test_the_ring_writes_*
    2  day directories older than 30 days go, younger stay, journaled   test_the_prune_*
    3  a day stops at its byte cap (the growth gate), journaled          test_a_full_day_*
    4  never inside a repository                                         test_never_in_a_repo
    5  the last finish candidate of a turn that ended in calls is not kept; slots come back
                                                      test_a_turn_that_ended_in_calls_*
  the scope:
    6  TandemLLM with tools, one choice; raw, no tools, n > 1 never
                                                      test_the_scope
    7  flag off: nothing is created                   test_flag_off_creates_nothing
  the loop (server/app.py generate_stream, the random 4-layer model, a drafter holding the tap):
    8  the tokens, the context, the KV and the recurrent state with a capture are the ones without
       it, on the chain, the tree and the single-token step   test_identity_*
    9  the prompt point is the row of the last message's last token, the one a cold forward
       computes, also when it sits in an earlier prefill chunk   test_the_prompt_row_*
   10  a think point every `every` reasoning tokens, a finish point at the end token
                                                      test_think_and_finish_points
   11  a point pools the rows the block committed (the mean, per source)   test_a_point_pools_*
   14  over the handler, a client budget beside: the capture reaches the loop, the answer is the
       one without it, the points reach the ring     test_over_the_handler_*
  the position:
   12  the served prompt's row of the last message's last token, with an earlier note replayed
       and this turn's note added (the router on the checkpoint's template)
                                                      test_the_probe_row_*
  the arithmetic:
   13  night 2's traffic writes under 100 MB          test_night_two_is_under_the_gate

Run: python tests/test_probe_snapshots.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import tempfile
import time

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine.spec import ThinkBudget  # noqa: E402
from server import app  # noqa: E402
from server import probe as probe_mod  # noqa: E402
from server.conductor import inject  # noqa: E402
from test_app_loop import OPEN, FakeTok, serve  # noqa: E402
from test_window_edge import FixedDrafter  # noqa: E402
from tools import probe_train  # noqa: E402

H = 8


class Journal:
    def __init__(self):
        self.rows = []

    def note(self, kind, task, *, decision=None, detail=None, session_key=None, request_id=None):
        self.rows.append({"kind": kind, "task": task, "decision": decision, "detail": detail,
                          "session_key": session_key, "request_id": request_id})


class Src:
    """An engine and a drafter as `rows_of` sees them: tapped rows and post-norm rows."""

    def __init__(self, T=4, seed=0, layers=(5, 19)):
        g = torch.Generator().manual_seed(seed)
        self.rows = [torch.randn(T, H, generator=g) for _ in layers]
        self.hidden_post_norm = torch.randn(1, T, H, generator=g)
        self.layers = list(layers)

    def tapped(self):
        return self.layers, "output", self.rows


def _ring(**kw) -> tuple[probe_mod.Ring, Journal, str]:
    d = tempfile.mkdtemp(prefix="probe-")
    j = Journal()
    return probe_mod.Ring(d, hidden=H, journal=j, every=8, weights="w1", slots=8, **kw), j, d


def _meta(rid="chatcmpl-1", **kw):
    return dict({"ts": time.time(), "request_id": rid, "row_id": "row" + rid[-1],
                 "session": "c:ses_a", "class_id": "code_edit", "effort": "xhigh",
                 "source": "qwen_context"}, **kw)


# ------------------------------------------------------------------ the ring

def test_the_ring_writes_records_and_index_lines():
    ring, j, d = _ring()
    s = Src()
    cap = ring.capture(2)
    cap._take("prompt", s, s, slice(2, 3), at=2)
    cap._take("think", s, s, slice(0, 4), at=8)
    ring.submit(cap, _meta())
    recs = probe_train.read_index(d)
    assert [r["point"] for r in recs] == ["prompt", "think"]
    r = recs[0]
    assert (r["request_id"], r["row_id"], r["session"], r["class_id"], r["effort"]) == \
        ("chatcmpl-1", "row1", "c:ses_a", "code_edit", "xhigh")
    assert r["rows"] == ["tap5", "tap19", "post"] and r["dim"] == H and r["weights"] == "w1"
    v = probe_train.vectors(recs, r["rows"]).reshape(2, 3, H)
    want0 = torch.stack([s.rows[0][2], s.rows[1][2], s.hidden_post_norm[0, 2]]).half().float()
    want1 = torch.stack([s.rows[0].mean(0), s.rows[1].mean(0),
                         s.hidden_post_norm[0].mean(0)]).half().float()
    assert torch.equal(torch.from_numpy(v[0]), want0)
    assert torch.equal(torch.from_numpy(v[1]), want1)
    day = os.path.join(d, "snapshots", recs[0]["day"])
    for f in os.listdir(day):
        assert os.stat(os.path.join(day, f)).st_mode & 0o777 == 0o600, f
    assert os.path.getsize(r["path"]) == 2 * 3 * H * 2
    assert [x["decision"] for x in j.rows] == ["written"]
    assert j.rows[0]["detail"]["points"] == {"prompt": 1, "think": 1}
    assert ring.info()["slots_free"] == 8, "every slot came back"
    # nothing that is text: only the keys the join needs
    assert not ({"messages", "content", "text", "digest"} & set(r))


def test_the_prune_keeps_thirty_days():
    ring, j, d = _ring()
    now = time.time()
    snap = os.path.join(d, "snapshots")
    days = {k: (_dt.date.fromtimestamp(now) - _dt.timedelta(days=k)).isoformat()
            for k in (0, 29, 30, 31, 45)}
    for day in days.values():
        os.makedirs(os.path.join(snap, day))
    got = ring.prune(now)
    left = sorted(os.listdir(snap))
    assert left == sorted([days[0], days[29], days[30]]), left
    assert sorted(got["days"]) == sorted([days[31], days[45]])
    assert j.rows[-1]["kind"] == "probe" and j.rows[-1]["decision"] == "deleted"


def test_a_full_day_takes_no_more_records():
    ring, j, d = _ring(day_mb=(3 * H * 2 + probe_mod.INDEX_LINE_BYTES) * 1.5 / 1e6)
    s = Src()
    for i in range(2):
        cap = ring.capture(0)
        cap._take("prompt", s, s, slice(0, 1), at=0)
        ring.submit(cap, _meta(f"chatcmpl-{i}"))
    assert len(probe_train.read_index(d)) == 1
    assert [x["decision"] for x in j.rows] == ["written", "day_cap"]
    assert ring.stats["day_cap"] == 1 and ring.info()["slots_free"] == 8


def test_never_in_a_repo():
    d = tempfile.mkdtemp(prefix="probe-repo-")
    os.makedirs(os.path.join(d, ".git"))
    try:
        probe_mod.Ring(os.path.join(d, "probe"), hidden=H)
    except ValueError as exc:
        assert "repository" in str(exc)
    else:
        raise AssertionError("a ring inside a work tree must be refused")


def test_a_turn_that_ended_in_calls_keeps_no_last_finish():
    ring, j, d = _ring()
    s = Src()
    cap = ring.capture(0)
    cap._take("prompt", s, s, slice(0, 1), at=0)
    cap.finish(s, s, slice(1, 2), at=10)      # an empty stop the monitor continued
    cap.finish(s, s, slice(3, 4), at=40)      # the end token after the call
    ring.submit(cap, _meta(), tool_calls=1)
    recs = probe_train.read_index(d)
    assert [(r["point"], r["at"]) for r in recs] == [("prompt", 0), ("finish", 10)]
    cap = ring.capture(0)
    cap.finish(s, s, slice(3, 4), at=40)      # an answer without calls: a done claim
    ring.submit(cap, _meta("chatcmpl-2"), tool_calls=0)
    assert [r["point"] for r in probe_train.read_index(d)][-1] == "finish"
    assert ring.info()["slots_free"] == 8


def test_the_scope():
    class P:
        active, row = True, {"id": "r"}
    assert probe_mod.wants(P(), {"tools": [{}]})
    assert not probe_mod.wants(P(), {})                          # a chat without tools
    assert not probe_mod.wants(P(), {"tools": [{}]}, n_choices=2)
    raw = P()
    raw.active = False                                           # X-Tandem: raw
    assert not probe_mod.wants(raw, {"tools": [{}]})
    assert not probe_mod.wants(None, {"tools": [{}]})


def test_flag_off_creates_nothing():
    os.environ.pop("QWEN38_PROBE_SNAPSHOTS", None)
    lines = []
    assert probe_mod.start({"engine": None}, None, log=lambda *a, **k: lines.append(a)) is None
    assert "off" in lines[0][0]


# ------------------------------------------------------------------ the loop

class Tapped:
    """A drafter that holds the engine's tap as the served block drafters do (the residual
    stream after layers 1 and 3), around one that proposes."""

    def __init__(self, inner, eng, layers=(1, 3)):
        self.inner, self.eng, self.layers = inner, eng, list(layers)
        self.want = {lid + 1: j for j, lid in enumerate(self.layers)}
        self.n_taps = eng.cfg.num_hidden_layers + 1
        self.i, self.rows = 0, []
        eng.tap = self._on

    def _on(self, h):
        if self.i == 0:
            self.rows = [None] * len(self.want)
        j = self.want.get(self.i)
        if j is not None:
            self.rows[j] = h
        self.i = (self.i + 1) % self.n_taps

    def tapped(self):
        if not self.rows or any(r is None for r in self.rows):
            return None
        return self.layers, "output", self.rows

    def __getattr__(self, k):
        return getattr(self.inner, k)


PROMPT = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, OPEN]


def _run(probe_on: bool, *, tree=False, width=5, chunk=0, max_new=40, eos=(), every=8, pos=9):
    eng = serve(None, tree=tree, k=7)
    inner = FixedDrafter(eng.cfg.vocab_size, width, tree_mode=tree) if width else None
    dr = Tapped(inner, eng) if inner is not None else Tapped(type("N", (), {
        "reset": lambda s: None, "observe": lambda s, t: None,
        "propose": lambda s, c, k: []})(), eng)
    app.STATE["drafter"] = dr
    app.STATE["prefix_chunk"] = chunk
    hidden = eng.cfg.hidden_size
    cap = None
    if probe_on:
        ring = probe_mod.Ring(tempfile.mkdtemp(prefix="probe-loop-"), hidden=hidden, every=every,
                              slots=64)
        cap = ring.capture(pos)
    think = ThinkBudget(FakeTok(), 0)
    out = list(app.generate_stream(torch.tensor(PROMPT), max_new, set(eos), think, probe=cap))
    n = eng.kv.length
    return {"out": out, "ctx": list(app.STATE["last_ctx"]), "n": n,
            "k": eng.kv.k[..., :n, :].clone(), "v": eng.kv.v[..., :n, :].clone(),
            "S": eng.state.S.clone(), "conv": eng.state.conv.clone(), "cap": cap, "eng": eng}


def _same(a, b, label):
    assert a["out"] == b["out"], (label, a["out"], b["out"])
    assert a["ctx"] == b["ctx"] and a["n"] == b["n"], label
    for key in ("k", "v", "S", "conv"):
        assert torch.equal(a[key], b[key]), (label, key)


def test_identity_on_every_path():
    for label, kw in (("chain", {"tree": False, "width": 5}), ("tree", {"tree": True, "width": 5}),
                      ("single", {"width": 0}), ("chunked", {"chunk": 4})):
        off, on = _run(False, **kw), _run(True, **kw)
        _same(off, on, label)
        assert on["cap"].points, label
    return "chain, tree, single-token step and a chunked prefill: identical with the capture"


def _reference_rows(pos: int):
    """The rows a cold forward of the whole prompt computes at `pos`: layers 1 and 3, post."""
    eng = serve(None)
    dr = Tapped(FixedDrafter(eng.cfg.vocab_size, 1, tree_mode=False), eng)
    with torch.no_grad():
        eng.forward(torch.tensor(PROMPT), start=0)
    return torch.stack([dr.rows[0][pos], dr.rows[1][pos], eng.hidden_post_norm[0, pos]])


def test_the_prompt_row_is_the_cold_forwards_row():
    want = _reference_rows(9).half().float()
    got = _run(True)["cap"].vector("prompt")
    assert torch.equal(got, want), (got - want).abs().max()
    # a chunked prefill (chunks of 4 over 12 tokens): 9 sits in the last chunk, taken after the
    # prefill; 5 in the second, taken by the chunk hook before the third forward
    for pos in (5, 9):
        r = _run(True, chunk=4, pos=pos)["cap"]
        assert r.names == ["tap1", "tap3", "post"]
        assert torch.allclose(r.vector("prompt"), _reference_rows(pos), atol=2e-3), pos


def test_think_and_finish_points():
    base = _run(False)
    eos_tok = base["out"][29]
    first = base["out"].index(eos_tok)
    r = _run(True, eos=(eos_tok,))
    cap = r["cap"]
    thinks = [p["at"] for p in cap.points if p["point"] == "think"]
    assert thinks and all(b > a for a, b in zip(thinks, thinks[1:]))
    # one point each time the count crossed a multiple of 8, never two for one crossing
    assert len(thinks) == (thinks[-1] // 8), thinks
    fins = [p for p in cap.points if p["point"] == "finish"]
    assert len(fins) == 1 and fins[0]["at"] == first, (fins, first)
    assert r["out"][-1] == eos_tok


def test_a_point_pools_the_rows_the_block_committed():
    ring, _, _ = _ring()
    s = Src(T=6)
    cap = ring.capture(None)
    cap._take("think", s, s, torch.tensor([0, 2, 3]), at=8)
    got = cap.vector("think")
    want = torch.stack([s.rows[0][[0, 2, 3]].mean(0), s.rows[1][[0, 2, 3]].mean(0),
                        s.hidden_post_norm[0, [0, 2, 3]].mean(0)]).half().float()
    assert torch.equal(got, want)
    assert cap.points[0]["k"] == 3


def test_over_the_handler_with_a_client_budget():
    """The handler with the snapshots on and a client reasoning budget (whose cap the handler
    also calls `cap`: the first suite run of 0.3.12 handed the loop that int as the capture): the
    loop gets the capture, the answer is the one without the snapshots, and the request's points
    reach the ring with the row's keys when it ends."""
    import contextlib
    import io

    import test_supervisor_frame as frame
    from test_app_loop import Req
    from test_client_budgets import _Conductor, _Plan

    class Plan(_Plan):
        router_cfg = None

        def probe_row(self, prompt, tok):
            return int(prompt.numel()) - 3

    class Cond(_Conductor):
        def plan(self, body, headers, **kw):
            p = Plan(self.budget, self.decides)
            p.row = {"extra": {}, "id": "row-h", "class_id": "debug", "effort": "high"}
            self.plans.append(p)
            return p

    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 40,
            "max_reasoning_tokens": 6, "reasoning_format": "reasoning_content",
            "tools": [{"type": "function", "function": {"name": "read"}}]}
    msgs, d = {}, None
    for on in (False, True):
        eng = serve(None)
        app.STATE.update(tok=frame.MonTok(), think_budget=0, conductor=Cond(None),
                         monitor=frame.monitor("on"))
        if on:
            d = tempfile.mkdtemp(prefix="probe-handler-")
            app.STATE["probe"] = probe_mod.Ring(d, hidden=eng.cfg.hidden_size, every=2, slots=16)
        with contextlib.redirect_stdout(io.StringIO()):
            _, raw = Req("/v1/chat/completions", json.loads(json.dumps(body))).response()
        got = json.loads(raw)["choices"][0]
        msgs[on] = (got["message"].get("content"), got["message"].get("reasoning_content"),
                    got["finish_reason"])
    assert msgs[True] == msgs[False], msgs
    recs = probe_train.read_index(d)
    assert recs and recs[0]["point"] == "prompt" and recs[0]["at"] == recs[0]["pos"], recs
    r = recs[0]
    assert (r["row_id"], r["class_id"], r["effort"], r["profile"]) == \
        ("row-h", "debug", "high", "coding")
    assert app.STATE["probe"].info()["slots_free"] == 16


# ------------------------------------------------------------------ the position

def test_the_probe_row_is_the_last_messages_last_token():
    from test_conductor_router import SYSTEM, body, setup, turn
    cond, fakes = setup(decider=lambda t, o: ("design_ui", 0.95))
    m1 = [{"role": "system", "content": SYSTEM},
          {"role": "user", "content": "Design a settings page for the dashboard."}]
    p1, ids1, clean1 = turn(cond, body(m1))
    assert p1.injected, "the ui-design note in turn 1"
    pos1 = p1.probe_row(torch.tensor(ids1), app.STATE["tok"])
    at1 = inject.insertion_point(clean1, inject.special_id(app.STATE["tok"], inject.IM_END))
    assert ids1[pos1] == clean1[at1 - 1] == ord("."), "the last token of the user's message"
    assert ids1[pos1 + 1] != clean1[at1], "this turn's note follows it"
    fakes["decider"].fn = lambda t, o: ("code_edit", 0.95)
    m2 = m1 + [{"role": "assistant", "content": "Here is the page."},
               {"role": "user", "content": "Now wire the save button to the API?"}]
    p2, ids2, clean2 = turn(cond, body(m2))
    assert p2.row["replayed"]["count"] == 1, "turn 1's note is replayed before the row"
    pos2 = p2.probe_row(torch.tensor(ids2), app.STATE["tok"])
    at2 = inject.insertion_point(clean2, inject.special_id(app.STATE["tok"], inject.IM_END))
    first = p2.session.notes[0]
    assert first.pos < at2 and pos2 == at2 - 1 + len(first.ids)
    assert ids2[pos2] == clean2[at2 - 1] == ord("?")
    assert p2.injected and ids2[pos2 + 1:pos2 + 1 + len(p2.session.notes[1].ids)] == \
        p2.session.notes[1].ids, "this turn's note right after the row"
    # a plan that changed nothing: the prompt's own tail
    from server.conductor.router import Plan

    class Bare:
        _probe_at = None
    assert Plan.probe_row(Bare(), torch.tensor(clean2), app.STATE["tok"]) == at2 - 1


# ------------------------------------------------------------------ the arithmetic

def test_night_two_is_under_the_gate():
    # the usage ledger, 2026-10-01 21:00 to 2026-10-02 09:00: 1,023 TandemLLM requests with tools
    # (964 raw ones are never written), 400,229 reasoning tokens = 271 points at 1,024, 17 turns
    # that ended without a call
    b = probe_mod.night_bytes(1023, 271, 17)
    assert b == (1023 + 271 + 17) * (6 * 5120 * 2 + probe_mod.INDEX_LINE_BYTES)
    assert b < 100e6, b
    # every request at the gate would not fit: the day cap is what holds it
    assert probe_mod.night_bytes(1987, 862, 40) > 100e6
    return f"night 2: {b / 1e6:.1f} MB"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            got = fn()
            print(f"  {name:62s} ok" + (f"  ({got})" if isinstance(got, str) else ""))
            passed += 1
    print(f"{passed} passed")
