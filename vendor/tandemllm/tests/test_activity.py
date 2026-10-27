"""The live activity of one request and the prefill progress fields, with no
torch: hand-built records, a stub reasoning budget and the real tool-call buffer.

Scenario map (the Gherkin of):

  a thinking request that ends in a tool call walks through its states
      test_a_thinking_request_that_ends_in_a_tool_call_walks_its_states
  the reasoning block is closed by the engine          test_the_engine_closing_the_reasoning_block
  thinking off goes straight to writing                test_thinking_off_goes_straight_to_writing
  a response-cache replay                              test_a_response_cache_replay
  the client leaves during a prefill                   test_the_client_leaving_mid_prefill_*
  refusals and rejections                              test_refusals_and_rejections
  a failed read never breaks the sampler               test_a_failed_read_is_null_for_one_tick
  no new per-token work                                test_stamps_grow_with_transitions_not_tokens
                                                       (+ tests/test_activity_app.py for the loop)

Run: python tests/test_activity.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import activity, live, usage  # noqa: E402
from server.toolcall import ToolCallBuffer  # noqa: E402


class Think:
    """The fields of engine/spec.py's ThinkBudget the sampler reads (that module imports torch)."""

    def __init__(self, budget=0):
        self.n, self.inside, self.done, self.reason = 0, True, False, None
        self.budget, self.t_closed, self.t_forced = budget, None, None

    @property
    def hit(self):
        # a stall only sets `reason`; the conductor's monitor decides on a close (ENG-177)
        if self.done or not self.inside:
            return False
        return bool(self.budget) and self.n >= self.budget

    def close(self):
        self.inside, self.done = False, True
        self.t_closed = time.perf_counter()


class Blocks:
    def __init__(self):
        self.blocks, self.t_first, self.t_last, self.accept = 0, None, None, {}


def _rec(rid="chatcmpl-a", thinking=True, prompt=1000):
    r = usage.RequestRecord(rid, "chat", True)
    r.model, r.client_kind = "m", "opencode"
    r.prompt_tokens, r.thinking = prompt, thinking
    return r


def _reg(**kw):
    return live.LiveRegistry(blocks=kw.pop("blocks", lambda: None),
                             last_prefill=kw.pop("last_prefill", lambda: None), **kw)


def _tokens(rec, n):
    src = rec.track(iter(range(n)))
    for _ in range(n):
        next(src)


# ------------------------------------------------------------------ the states
def test_a_thinking_request_that_ends_in_a_tool_call_walks_its_states():
    rec = _rec()
    think, tbuf = Think(), ToolCallBuffer(names={"write_file"})
    rec.live_refs = (think, tbuf)
    assert activity.state_of(rec) == "queued"
    rec.lock_acquired()
    assert activity.state_of(rec) == "prefilling"
    _tokens(rec, 1)
    assert activity.state_of(rec) == "thinking"
    think.n = 40
    think.close()
    assert activity.state_of(rec) == "writing"
    tbuf.feed("I will write the file.\n")
    assert activity.state_of(rec) == "writing"
    tbuf.feed("<tool_call>\n<function=wri")
    assert activity.state_of(rec) == "tool_call"
    assert activity.tool_of(rec)["name"] is None                 # not decoded yet
    assert activity.label({"state": "tool_call", "tool": activity.tool_of(rec)}) == \
        "Calling a tool"
    tbuf.feed("te_file>\n<parameter=content>\nabc")
    t1 = activity.tool_of(rec)
    assert t1["name"] == "write_file" and t1["index"] == 0 and t1["calls_done"] == 0, t1
    tbuf.feed("def" * 50)
    t2 = activity.tool_of(rec)
    assert t2["arg_bytes"] > t1["arg_bytes"], (t1, t2)
    tbuf.feed("\n</parameter>\n</function>\n</tool_call>")
    assert activity.state_of(rec) == "writing"
    rec.end_state = activity.state_of(rec, ignore_step=True)
    rec.step, rec.t_finishing = "flush", time.perf_counter()
    assert activity.state_of(rec) == "finishing"
    rec.tool_names = [c["function"]["name"] for c in tbuf.calls]
    rec.completion_tokens, rec.finish_reason, rec.tool_calls = rec.n_live, "tool_calls", 1
    rec.end()
    stop = activity.stop_of(rec, rec.t_end)
    assert stop["reason"] == "tool_calls" and stop["state"] == "tool_call", stop
    assert stop["sentence"].startswith("tool call: write_file after "), stop
    path = activity.path_of(rec)
    assert path == ["queued", "prefilling", "thinking", "writing", "tool_call", "writing",
                    "finishing"], path
    tl = activity.timeline_of(rec, rec.t_end, "tool_calls")
    assert [e["state"] for e in tl][-2:] == ["finishing", "done"]
    assert any(e.get("detail") == "write_file" for e in tl if e["state"] == "tool_call"), tl


def test_the_engine_closing_the_reasoning_block():
    rec = _rec()
    think = Think()
    rec.live_refs = (think, None)
    rec.lock_acquired()
    _tokens(rec, 5)
    think.reason = "novelty(0.05)"                               # the stall arm fired
    assert activity.state_of(rec) == "thinking" and activity.closed_by(think) is None
    think.t_forced = time.perf_counter()                         # and the monitor closed it
    think.close()
    assert activity.state_of(rec) == "writing" and activity.closed_by(think) == "stall"
    s = Think()                                                  # in shadow: nothing is forced
    s.reason = "repeat16(share=0.91 at=2112)"
    s.close()
    assert activity.closed_by(s) == "model"
    b = Think(budget=10)
    b.n = 10
    assert b.hit and activity.closed_by(b) == "budget"
    m = Think()
    m.close()
    assert activity.closed_by(m) == "model"
    assert activity.closed_by(Think()) is None                   # still thinking


def test_thinking_off_goes_straight_to_writing():
    rec = _rec(thinking=False)
    think = Think()
    think.inside = False                                         # the template closed it
    rec.live_refs = (think, None)
    rec.lock_acquired()
    _tokens(rec, 3)
    assert activity.state_of(rec) == "writing"
    d = activity.decode_of(rec, time.perf_counter(), None, None)
    assert d["thinking_tokens"] == 0 and d["content_tokens"] == 3, d
    assert activity.path_of(rec) == ["queued", "prefilling", "writing"]


def test_a_response_cache_replay():
    rec = _rec()
    rec.live_refs = (Think(), None)
    rec.lock_acquired()
    rec.absorb_response_cache()
    assert activity.state_of(rec) == "replaying"
    assert activity.prefill_of(rec, time.perf_counter()) is None
    _tokens(rec, 4)
    assert activity.state_of(rec) == "replaying"
    assert activity.path_of(rec) == ["queued", "replaying"]


# ------------------------------------------------------------------ the prefill
def test_prefill_progress_from_the_chunk_hook():
    rec = _rec(prompt=48210)
    rec.lock_acquired()
    t0 = rec.t_lock
    rec.prefill_info = {"start": 12288, "chunk": 1024, "t0": t0, "kind": "resident"}
    p = activity.prefill_of(rec, t0)
    assert p["done"] == 12288 and p["pct"] is None and p["progress"] == "chunked", p
    rec.pf_marks.append((35840, 48210, t0 + 10.0))
    rec.pf_marks.append((36864, 48210, t0 + 10.5))
    p = activity.prefill_of(rec, t0 + 10.6)
    assert p["done"] == 36864 and p["total"] == 48210 and p["cached"] == 12288
    assert p["pct"] == 76.5 and p["tps_now"] == 2048.0, p
    assert p["tps_avg"] == round((36864 - 12288) / 10.5, 2)
    # the ETA runs on the rate after the first chunk (VIS-40), here the one chunk after it
    assert p["tps_live_avg"] == 2048.0 and p["warmup"] == {"chunks": 2, "reached": True}
    assert abs(p["eta_ms"] - (48210 - 36864) / p["tps_live_avg"] * 1e3) < 1.0
    a = {"state": "prefilling", "prefill": p}
    assert activity.label(a) == "Prefilling 36,864 of 48,210 (76 %)", activity.label(a)


def test_a_prefill_has_a_rate_from_its_first_mark():
    """ENG-373: a short prefill or re-read reports one mark (a chunk, or a layer-major window),
    and its rate is that mark's rows over the time since the loop started, the unit of
    `tps_avg`; a second mark makes it the last chunk's rate again."""
    rec = _rec(prompt=8647)
    rec.lock_acquired()
    t0 = rec.t_lock
    rec.prefill_info = {"start": 2048, "chunk": 1024, "t0": t0, "kind": "resident"}
    assert activity.prefill_of(rec, t0)["tps_now"] is None          # nothing measured yet
    rec.pf_marks.append((7168, 8647, t0 + 6.4))
    p = activity.prefill_of(rec, t0 + 6.5)
    assert p["tps_now"] == p["tps_avg"] == 800.0, p
    rec.pf_marks.append((8192, 8647, t0 + 7.4))
    p = activity.prefill_of(rec, t0 + 7.5)
    assert p["tps_now"] == 1024.0 and p["tps_avg"] == round(6144 / 7.4, 2), p


def test_a_window_running_has_a_rate_before_its_mark():
    """SRV-591: a layer-major window is one mark, written when the whole window has run; while it
    runs, `tps_now` is its rows times the share of its layers seen run (`k` at `t_k`, from the
    prefill's thread) over the time to then."""
    rec = _rec(prompt=7105)
    rec.lock_acquired()
    t0 = rec.t_lock
    rec.prefill_info = {"start": 2048, "chunk": 1024, "t0": t0, "kind": "resident"}
    live = {"rows": (2048, 7105), "t0": t0, "n": 64, "layers": []}
    rec.prefill_info["window_live"] = live
    assert activity.prefill_of(rec, t0 + 1.0)["tps_now"] is None     # no layer seen run
    live.update(k=32, t_k=t0 + 4.0)
    p = activity.prefill_of(rec, t0 + 4.5)
    assert p["tps_now"] == round(5057 * 32 / 64 / 4.0, 2), p              # 632.12
    assert abs(p["eta_ms"] - (7105 - (2048 + 5057 / 2)) / (5057 / 8.0) * 1e3) < 1.0, p
    assert p["done"] == 2048 and p["tps_avg"] is None, p                   # no mark yet
    # past a mark, the window running now is the rate
    rec.pf_marks.append((10240, 20000, t0 + 10.0))
    live.update(rows=(10240, 18432), t0=t0 + 10.0, k=16, t_k=t0 + 12.0)
    p = activity.prefill_of(rec, t0 + 12.5)
    assert p["tps_now"] == round(8192 * 16 / 64 / 2.0, 2) and p["done"] == 10240, p


def test_a_single_call_prefill_shows_total_and_elapsed_only():
    rec = _rec(prompt=4000)
    rec.lock_acquired()
    rec.prefill_info = {"start": 0, "chunk": 0, "t0": rec.t_lock, "kind": None}
    p = activity.prefill_of(rec, rec.t_lock + 1.0)
    assert p["progress"] == "single_call" and p["pct"] is None and p["tps_avg"] is None, p
    assert activity.label({"state": "prefilling", "prefill": p}) == "Prefilling 4,000 tokens"


def test_a_finished_prefill_is_the_response_numbers():
    rec = _rec(prompt=1000)
    rec.lock_acquired()
    rec.t_lock -= 0.5
    _tokens(rec, 1)
    rec.absorb_prefill({"reused": 200, "forwarded": 800, "kind": "session"})
    p = activity.prefill_of(rec, time.perf_counter())
    assert p["pct"] == 100.0 and p["cached"] == 200 and p["done"] == 1000
    assert abs(p["tps_avg"] - round(rec.prefill_tps, 2)) < 0.02, (p, rec.prefill_tps)


# ------------------------------------------------------------------ the client and the stop
class Sock:
    closed = False


def test_the_client_leaving_mid_prefill_is_seen_within_one_tick():
    rec = _rec(prompt=60000)
    rec.sock = Sock()
    reg = _reg(sock_closed=lambda s: s.closed)
    reg.register(rec)
    rec.lock_acquired()
    reg.subscribe()
    reg.tick()
    row = reg.snapshot()["requests"][0]
    assert row["activity"]["client"]["connected"] is True
    rec.sock.closed = True
    reg.tick()
    row = reg.snapshot()["requests"][0]
    assert row["activity"]["client"]["connected"] is False
    assert row["activity"]["state"] == "prefilling"
    assert rec.client_gone_at is not None
    # the handler notices at its next chunk hook: ClientGone, settled `abandoned`
    rec.finish_reason = "abandoned"
    rec.end()
    reg.finish(rec)
    stop = rec.stop
    assert stop["reason"] == "abandoned" and stop["state"] == "prefilling", stop
    assert stop["tokens_sent"] == 0 and stop["detail"] == "left_during_prefill"
    assert stop["silent_ms"] is not None and stop["client_gone_ms"] is not None
    assert "of silent prefill, 0 tokens sent" in stop["sentence"], stop["sentence"]
    reg.unsubscribe()


def test_refusals_and_rejections():
    full = _rec("r1")
    full.status, full.finish_reason, full.stop_detail = 503, "refused", "queue_full"
    full.end()
    s = activity.stop_of(full, full.t_end)
    assert (s["reason"], s["detail"], s["state"]) == ("refused", "queue_full", "queued")
    assert s["sentence"] == "refused: queue full"
    late = _rec("r2")
    late.status, late.finish_reason, late.stop_detail = 429, "refused", "queue_timeout"
    late.end()
    assert activity.stop_of(late, late.t_end)["sentence"].startswith(
        "refused: timed out in the queue after")
    big = _rec("r3")
    big.lock_acquired()
    big.status, big.stop_detail = 400, "prompt_too_long"
    big.end()
    s = activity.stop_of(big, big.t_end)
    assert (s["reason"], s["detail"]) == ("rejected", "prompt_too_long")
    bad = _rec("r4")
    bad.status = 400
    bad.end()
    assert activity.stop_of(bad, bad.t_end)["detail"] == "bad_request"
    down = _rec("r5")
    down.status, down.finish_reason, down.stop_detail = 503, "refused", "shutting_down"
    down.end()
    s = activity.stop_of(down, down.t_end)
    assert s["reason"] == "cancelled" and s["sentence"] == "cancelled: the server was shutting down"


def test_every_stop_reason_has_a_sentence():
    base = time.perf_counter()
    cases = {
        ("stop", "eos"): "finished at the end of the answer after",
        ("stop", "stop_string"): "finished at a stop string after",
        ("stop", "pattern_guard"): "finished by the repetition guard after",
        ("length", None): "stopped at the length limit (32,768 tokens)",
        ("timeout", None): "timed out after",
        ("error", "RuntimeError"): "error: RuntimeError after 1,204 tokens",
        ("abandoned", None): "abandoned by the client",
        ("tool_calls", None): "tool call: write_file, bash (2 calls) after",
    }
    for (reason, detail), want in cases.items():
        rec = _rec()
        rec.t_arrival = base
        rec.lock_acquired()
        _tokens(rec, 1204)
        rec.max_tokens = 32768
        rec.finish_reason = reason
        rec.error_type = detail if reason == "error" else None
        rec.stop_detail = detail if reason == "stop" else None
        rec.tool_names = ["write_file", "bash"]
        rec.end()
        s = activity.stop_of(rec, rec.t_end)
        assert s["reason"] == reason, (reason, s)
        assert want in s["sentence"], (reason, s["sentence"])
    rec = _rec()
    rec.t_arrival = base - 41.23
    rec.end()
    rec.t_end = base
    rec.finish_reason, rec.tool_names = "tool_calls", ["bash"]
    assert activity.stop_of(rec, base)["sentence"] == "tool call: bash after 41.2 s"
    assert activity.STOP_REASONS == ("stop", "length", "tool_calls", "timeout", "abandoned",
                                     "error", "refused", "rejected", "cancelled")


def test_a_failed_read_is_null_for_one_tick():
    class Flaky(ToolCallBuffer):
        fail = True

        @property
        def _streamed(self):
            if Flaky.fail:
                raise IndexError("the list was reset under the reader")
            return self.__dict__.get("_s", [])

        @_streamed.setter
        def _streamed(self, v):
            self.__dict__["_s"] = v

    rec = _rec()
    tbuf = Flaky(names={"bash"})
    rec.live_refs = (Think(), tbuf)
    rec.lock_acquired()
    _tokens(rec, 2)
    rec.live_refs[0].close()
    Flaky.fail = False
    tbuf.feed("<tool_call>\n<function=bash>\n<parameter=command>\nls")
    Flaky.fail = True
    assert activity.tool_of(rec) is None                         # this tick: null
    assert activity.state_of(rec) == "tool_call"                 # the rest still reads
    Flaky.fail = False
    assert activity.tool_of(rec)["name"] == "bash"               # the next tick: normal
    reg = _reg()
    reg.register(rec)
    Flaky.fail = True
    snap = reg.snapshot()                                        # never raises
    assert snap["requests"][0]["activity"]["tool"] is None


def test_the_queue_place_is_the_arrival_order():
    reg = _reg(queue_timeout=lambda: 120.0)
    run = _rec("run")
    reg.register(run)
    run.lock_acquired()
    a, b = _rec("a"), _rec("b")
    reg.register(a)
    reg.register(b)
    rows = {r["request_id"]: r for r in reg.snapshot()["requests"]}
    assert rows["a"]["activity"]["queue"]["place"] == 1
    assert rows["b"]["activity"]["queue"]["place"] == 2
    assert rows["b"]["activity"]["label"] == "Queued, about 2nd in line"
    assert rows["b"]["activity"]["queue"]["timeout_s"] == 120.0


# ------------------------------------------------------------------ between two requests
def _finish(reg, rec, reason, tools=()):
    rec.completion_tokens, rec.finish_reason = rec.n_live, reason
    rec.tool_names = list(tools)
    rec.end()
    reg.finish(rec)


def test_waiting_for_client_and_the_next_request_continues_it():
    reg = _reg()
    first = _rec("first")
    first.conv = "ses_1"
    reg.register(first)
    first.lock_acquired()
    _tokens(first, 3)
    _finish(reg, first, "tool_calls", ["bash"])
    eng = reg.snapshot()["engine"]
    assert eng["state"] == "waiting_for_client", eng
    assert eng["label"] == "Waiting for client: running tool bash"
    assert eng["waiting_for_client"]["request_id"] == "first"
    nxt = _rec("next")
    nxt.conv = "ses_1"
    reg.register(nxt)
    snap = reg.snapshot()
    assert snap["engine"]["state"] == "busy" and snap["engine"]["waiting_for_client"] is None
    c = snap["requests"][0]["activity"]["continues"]
    assert c["request_id"] == "first" and c["inferred"] is False and c["tool_names"] == ["bash"]
    # no conversation id: linked only once the prefill resumed the state the first one saved
    _finish(reg, nxt, "tool_calls", ["read"])
    third = _rec("third")
    reg.register(third)
    third.lock_acquired()
    assert reg.snapshot()["requests"][0]["activity"]["continues"] is None
    third.prefill_info = {"kind": "session", "start": 900, "chunk": 1024, "t0": 0.0}
    c = reg.snapshot()["requests"][0]["activity"]["continues"]
    assert c["request_id"] == "next" and c["inferred"] is True, c
    # a finish that is not a tool call leaves no wait
    _finish(reg, third, "stop")
    assert reg.snapshot()["engine"]["state"] == "idle"


# ------------------------------------------------------------------ the hot path
def test_stamps_grow_with_transitions_not_tokens():
    tbuf = ToolCallBuffer(names={"write_file"})
    tbuf.feed("<tool_call>\n<function=write_file>\n<parameter=content>\n")
    for _ in range(2000):
        tbuf.feed("x")
    tbuf.feed("\n</parameter>\n</function>\n</tool_call>")
    assert [e[1] for e in tbuf.events] == ["open", "name", "close"], tbuf.events
    for _ in range(40):
        tbuf.feed("<tool_call>\n<function=write_file>\n<parameter=a>\n1\n</parameter>\n"
                  "</function>\n</tool_call>")
    assert len(tbuf.events) == 16                                # capped
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "server", "toolcall.py")).read()
    feed = src.split("    def feed(self, piece: str)")[1].split("    def _credit_streamed")[0]
    assert feed.count("self._event(") == 2, "an open and a close, both outside the piece loop"


def test_the_token_loops_gained_no_statement():
    """The hot-path rule, read off the source: the decode generator, the reasoning splitter
    and the budget's per-token path carry nothing of the live view but the one stamp in the
    `</think>` branch (the record's `track` is held by tests/test_live.py)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "server", "app.py")).read()
    gen = app.split("def generate_stream(")[1].split("\ndef _force_close(")[0]
    loop = gen.split("while n_out < max_new:")[1]
    code = "\n".join(ln.split("#")[0] for ln in loop.splitlines())
    for word in ("rec.", "activity", "live", ".pf", "info", "events"):
        assert word not in code, word
    stream = open(os.path.join(root, "server", "stream.py")).read()
    assert "activity" not in stream and "perf_counter" not in stream
    spec = open(os.path.join(root, "engine", "spec.py")).read()
    observe = spec.split("    def observe(self, ids) -> None:")[1].split("    def _novelty")[0]
    assert observe.count("perf_counter") == 1 and "self.t_closed = time.perf_counter()" in observe

if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
