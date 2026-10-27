"""The Context panel's figures (SRV-445, SRV-449, SRV-451, SRV-453). Seen on a phone on 2026-10-14
after a 48-step run that wrote an HTML file: "tool calls 0" after about 60 calls, "files 0" after a
write_file of 7.8 KB and seven edit_file calls, "KV for this chat —", "resident prefix —" and
"sandbox running, 0 B".

  1  the usage counts every tool call of the branch
  2  the context's `files` part counts what the file tools carry in the next request (a write's
     content or its stub, an edit's old and new text, a read's result), from the first message
     a compaction kept
  3  the memory says where the engine holds the chat: its rows in the slot or the park, by the
     owner's tag (or its short key from an engine without tags), else the tier the next turn reads
  4  a running sandbox's memory is the runtime's figure (usage less the inactive file cache),
     read at most every 5 s, and null when the runtime does not say
  5  the harness's picks and the prompt's split outlive a chatd restart (SRV-455)
  6  the prompt's images are a part of their own, the engine's count (VIS-205)

CPU only. Run: python -m pytest -q tests/test_chatd_panel.py
"""

from __future__ import annotations

import hashlib
import json
import shutil

from chatd_util import Chat
from test_chatd_agent import calls

from chatd import context as ctx_mod, render
from chatd import sandbox as sb_mod


def test_the_usage_counts_every_tool_call_and_the_files_part_what_the_file_tools_carried():
    c = Chat()
    try:
        cid = c.new()
        page = "<!doctype html><h1>" + "x" * 4000 + "</h1>"
        c.send(cid, calls(("write_file", {"path": "site/a.html", "content": page}),
                          ("edit_file", {"path": "site/a.html", "old": "<h1>", "new": "<h2>"})))
        s = c.wait_idle(cid, 20)
        made = [p for m in s["messages"] if m["role"] == "assistant" for p in m["parts"]
                if p["type"] == "tool_call"]
        assert len(made) == 2 and s["usage"]["tool_calls"] == 2, s["usage"]
        files = next(p for p in s["context"]["parts"] if p["kind"] == "files")["tokens"]
        # the write's content stays whole in the next request (chatd/render.py `folded_calls`):
        # the page's tokens and the edit's
        assert ctx_mod.estimate_tokens(page) <= files <= ctx_mod.estimate_tokens(page) + 10, \
            (files, s["context"]["parts"])
        c.send(cid, calls(("read_file", {"path": "site/a.html"})))
        s = c.wait_idle(cid, 20)
        assert s["usage"]["tool_calls"] == 3
        files2 = next(p for p in s["context"]["parts"] if p["kind"] == "files")["tokens"]
        assert files2 > files + len(page) // 4 - 10, (files, files2)
        msgs = next(p for p in s["context"]["parts"] if p["kind"] == "messages")["tokens"]
        assert msgs >= 0
    finally:
        c.close()


def test_file_tokens_start_at_the_compaction_boundary():
    w = {"type": "tool_call", "name": "write_file", "step": 1,
         "input": json.dumps({"path": "a", "content": "y" * 400})}
    late = dict(w, step=3)
    branch = [{"role": "user", "parts": []}, {"role": "assistant", "parts": [w]},
              {"role": "user", "parts": []}, {"role": "assistant", "parts": [w, late]}]
    assert ctx_mod.file_tokens(branch) == 300
    assert ctx_mod.file_tokens(branch, 2) == 200
    assert ctx_mod.file_tokens(branch, 2, (3, 2)) == 100
    bad = {"type": "tool_call", "name": "write_file", "input": "{\"path\": "}
    failed_read = {"type": "tool_call", "name": "read_file",
                   "output": {"content": "no such file", "is_error": True}}
    assert ctx_mod.file_tokens([{"role": "assistant", "parts": [bad, failed_read]}]) == 0


def _res(**kw):
    base = {"valid": 37614, "bytes": 3_000_000, "park_bytes": 1_000_000, "pin_bytes": 500_000,
            "park_rows": 900, "owner": None, "park_owner": None}
    return {"caches": {"resident": {**base, **kw}}, "engine": {"base_model": "x"}}


def _tag(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def test_the_memory_says_where_the_engine_holds_the_chat():
    me, other = "chat-01J9Z3K7Q2W5E8R4T6Y1U3I5O7", "chat-01J9Z3K0A1S2D3F4G5H6J7K8L9"
    warm = {"warm": 1000, "source": "resident"}
    h = ctx_mod.holding(_res(owner="chat-01J9Z3K…", owner_tag=_tag(me)), me, warm)
    assert h == {"kv_tokens": 37614, "kv_where": "slot", "resident_bytes": 1_500_000,
                 "resident_owner": "this"}, h
    # the tag decides: the same short key, another chat
    h = ctx_mod.holding(_res(owner="chat-01J9Z3K…", owner_tag=_tag(other)), me, warm)
    assert h["kv_where"] == "other" and h["resident_owner"] == "other" and h["kv_tokens"] is None
    # an engine without tags: the short key
    assert ctx_mod.holding(_res(owner="chat-01J9Z3K…"), me, warm)["kv_where"] == "slot"
    h = ctx_mod.holding(_res(owner="chat-01J9Z3X…", park_owner="chat-01J9Z3K…",
                             park_owner_tag=_tag(me), owner_tag=_tag(other)), me, warm)
    assert h == {"kv_tokens": 900, "kv_where": "park", "resident_bytes": 1_000_000,
                 "resident_owner": "this"}, h
    spilled = {"warm": 30000, "source": "nvme_spill"}
    assert ctx_mod.holding(_res(owner="x", owner_tag=_tag(other)), me, spilled)["kv_where"] == \
        "nvme_spill"
    assert ctx_mod.holding(_res(), me, {"warm": 0, "source": "none"}) == {
        "kv_tokens": None, "kv_where": "none", "resident_bytes": None, "resident_owner": "none"}
    assert ctx_mod.holding({}, me, warm)["kv_where"] is None


def test_the_report_carries_the_holding_for_our_engine_only():
    model = {"id": "m", "max_len": 1000, "max_output": 100}
    common = dict(model=model, settings={}, branch_tokens=10, files_tokens=0, stats=None,
                  run=None, row=None, peek=None, harness=None)
    me = "chat-abc"
    r = ctx_mod.report(**common, facts=_res(owner=me, owner_tag=_tag(me)), session=me)
    assert r["memory"]["kv_where"] == "slot" and r["memory"]["kv_tokens"] == 37614
    r = ctx_mod.report(**common, facts={}, session=me)
    assert r["memory"]["kv_where"] is None and r["memory"]["resident_owner"] is None


class _Docker:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def call(self, method, path, body=None, timeout=60.0):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        assert path.startswith("/containers/box-1/stats?stream=false"), path
        return self.answer


def test_a_running_sandbox_reports_its_memory_in_use(tmp_path):
    t = [100.0]
    eng = _Docker((200, {"memory_stats": {"usage": 64_405_504,
                                          "stats": {"inactive_file": 32_956_416}}}))
    boxes = sb_mod.Sandboxes(str(tmp_path), engine=eng, clock=lambda: t[0])
    b = boxes.box("c1")
    assert boxes.memory_used(b) is None, "a stopped box has no figure"
    b.state, b.container = "running", "box-1"
    assert boxes.memory_used(b) == 31_449_088
    t[0] += 1
    assert boxes.memory_used(b) == 31_449_088 and eng.calls == 1, "read at most every 5 s"
    t[0] += 10
    eng.answer = OSError("socket gone")
    assert boxes.memory_used(b) is None
    t[0] += 10
    eng.answer = (200, {"memory_stats": {}})
    assert boxes.memory_used(b) is None, "a runtime that does not say"


def test_the_harness_and_the_split_outlive_a_chatd_restart():
    """SRV-455: the usage frame's `tandem` (the harness's picks, the system and tool tokens) lived
    in chatd's memory only; after a restart the panel lost the skills and showed the system prompt
    and the tool schemas as 0 until the next turn. The answer's stats keep them now."""
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_TANDEM hi there")
        s = c.wait_idle(cid)
        want = s["context"]["harness"]
        assert want and want["packages"][0]["name"] == "coding", want
        assert s["messages"][-1]["stats"]["tandem"]["harness"] == want
        c.owned_fake = False                       # the same fake engine serves the restart
        c.close(keep_data=True)
        again = Chat(data=c.data, fake=(c.fake_srv, c.fake_port))
        try:
            ctx = again.snap(cid)["context"]
            parts = {p["kind"]: p["tokens"] for p in ctx["parts"]}
            assert ctx["harness"] == want and parts["system"] == 2 and parts["tools"] == 3, ctx
        finally:
            again.close()
    finally:
        c.fake_srv.shutdown()
        shutil.rmtree(c.data, ignore_errors=True)


# ---------------------------------------------------------------------- SRV-509, live 2026-10-16
# An agent turn on Nemotron 3.5 Super VL (13,270 tokens, 2,193 thinking, a tool call a step): Now
# said "not tracked for this model" for the speed and the acceptance while it decoded, showed an
# earlier call's 923 tokens and 797 thinking; after the turn "thinking, this turn" was 0 and the
# conversation counted every earlier call's answer twice.

def _run_after_one_call():
    """A run whose first call (r1) finished with its usage frame; its second call is in flight."""
    import types

    from chatd.runs import Run
    run = Run(types.SimpleNamespace(), types.SimpleNamespace(id="c1"), user_message_id="u",
              message_id="m", max_steps=10, started_by=None)
    run.request_ids, run.tokens, run.thinking_tokens = {"r1"}, 923, 797
    run.add_spec({"predicted_n": 923, "predicted_ms": 25_000.0, "blocks": 330, "draft_n": 600,
                  "draft_n_accepted": 330})
    run.call_base, run.request_id, run.usage = (923, 797), None, None
    return run


def _rows(*rows):
    from chatd.engine import LiveTail
    tail = LiveTail(None, lambda: False)
    tail._take({"requests": list(rows)})
    return tail


LIVE_R2 = {"request_id": "r2", "conversation_id": "chat-c1", "phase": "decode",
           "tokens_per_block": None, "ended_ms_ago": None,
           "activity": {"state": "thinking", "decode": {
               "tokens": 120, "thinking_tokens": 100, "tps_now": 36.0,
               "tokens_per_round_now": None}}}
DONE_R1 = {"request_id": "r1", "conversation_id": "chat-c1", "phase": "done",
           "tokens_per_block": 2.8, "ended_ms_ago": 5000.0,
           "activity": {"state": "done", "decode": {"tokens": 923, "thinking_tokens": 797,
                                                    "tps_now": None}}}


def test_the_live_tail_takes_the_call_in_flight_over_the_finished_ones():
    """The snapshot lists the live rows first, then the finished ones of the last 30 s: the last
    row of a conversation won, an earlier call's finished one."""
    older = dict(DONE_R1, request_id="r0", ended_ms_ago=20_000.0)
    assert _rows(LIVE_R2, DONE_R1, older).row("chat-c1", None)["request_id"] == "r2"
    assert _rows(older, DONE_R1).row("chat-c1", None)["request_id"] == "r1"   # the newest end


def _metrics(run, tail):
    import time
    import types

    from chatd.service import Service
    events = []
    svc = types.SimpleNamespace(live=tail, emit=lambda conv, ev, **kw: events.append(ev),
                                session_of=lambda conv: "chat-" + conv.id)
    conv = types.SimpleNamespace(id="c1", context_sent_at=time.monotonic(), parent_id=None)
    Service._metrics(svc, conv, run)
    return events[-1]


def test_the_now_figures_are_the_call_in_flights_and_the_turns():
    run = _run_after_one_call()
    ev = _metrics(run, _rows(LIVE_R2, DONE_R1))
    assert ev["decode_tps"] == 36.0
    assert ev["tokens"] == 923 + 120 and ev["thinking_used"] == 797 + 100, ev
    assert ev["thinking_call"] == 100                       # what the call's budget caps
    assert ev["acceptance"] == 0.55 and ev["tokens_per_round"] == 2.79, ev   # the first call's
    d = run.as_dict()
    assert d["thinking"] == {"used": 897, "budget": None, "call": 100}
    # between two calls only the finished one is there: no speed, the counts stay the run's
    run.tokens, run.thinking_tokens = 1043, 897
    ev = _metrics(run, _rows(DONE_R1))
    assert ev["decode_tps"] is None and ev["tokens"] == 1043 and ev["acceptance"] == 0.55
    # another run's finished call is not this run's
    run2 = _run_after_one_call()
    run2.request_ids = {"r9"}
    ev = _metrics(run2, _rows(DONE_R1))
    assert ev["tokens"] == 923 and ev["tokens_per_round"] == 2.79


def test_the_turns_stats_weigh_every_call():
    run = _run_after_one_call()
    run.add_spec({"predicted_n": 707, "predicted_ms": 19_400.0, "blocks": 200, "draft_n": 400,
                  "draft_n_accepted": 300})
    run.usage = {"usage": {"prompt_tokens": 32289, "completion_tokens": 707},
                 "timings": {"predicted_per_second": 36.41, "tokens_per_block": 3.53},
                 "metrics": {"speculative_decoding": {"draft_acceptance_rate": 0.75}}}
    run.step_usage = [{"step": 1, "ttft_ms": 900.0}, {"step": 2, "ttft_ms": 6082.7}]
    run.started_ms -= 38_000
    s = run.stats("stop")
    assert s["ttft_ms"] == 900.0 and 38_000 <= s["total_ms"] < 60_000, s   # the turn's, not a call's
    assert s["acceptance"] == round(630 / 1000, 4) and s["tokens_per_round"] == round(
        (922 + 706) / 530, 2), s
    assert s["decode_tps"] == round((922 + 706) * 1000 / 44_400, 2)


def test_thinking_this_turn_live_and_after_the_turn():
    model = {"id": "m", "max_len": 262144, "max_output": 32768}
    common = dict(model=model, settings={}, files_tokens=0, facts={}, peek=None, harness=None)
    run = {"thinking": {"used": 897, "call": 100, "budget": None}}
    row = {"prompt_tokens": 30318, "cached_tokens": 30286}
    r = ctx_mod.report(**common, branch_tokens=20_000, stats=None, run=run, row=row)
    parts = {p["kind"]: p for p in r["parts"]}
    # the first call's 797 are in the second call's prompt: counted once, as thinking
    assert parts["thinking"]["tokens"] == 897 and parts["messages"]["tokens"] == 30318 - 797
    assert parts["thinking"]["label"] == "thinking, this turn"
    stats = {"prompt_tokens": 32289, "completion_tokens": 13270, "reasoning_tokens": 2193}
    r = ctx_mod.report(**common, branch_tokens=30_803, stats=stats, run=None, row=None)
    parts = {p["kind"]: p for p in r["parts"]}
    assert parts["thinking"]["tokens"] == 2193 and parts["thinking"]["label"] == \
        "thinking, last turn"
    assert r["used"] == 30_803 + 2193
    r = ctx_mod.report(**common, branch_tokens=10, stats=None, run=None, row=None)
    assert {p["kind"]: p for p in r["parts"]}["thinking"]["label"] == "thinking, this turn"


def test_the_images_part_is_the_engines_count_out_of_the_conversations_share():
    """VIS-205: the prompt's image rows (the usage frame's `tandem.parts.images`: a 1,280 x 800 screenshot is
    1,000 rows on Super VL) are a part of their own, taken out of the conversation's share."""
    model = {"id": "m", "max_len": 262144, "max_output": 32768}
    common = dict(model=model, settings={}, files_tokens=0, facts={}, peek=None, harness=None, stats=None,
                  run=None)
    row = {"prompt_tokens": 9_000, "cached_tokens": 0}
    split = {"system": 1_500, "tools": 2_000, "history": 4_500, "injected": 0, "images": 1_000}
    r = ctx_mod.report(**common, branch_tokens=9_000, row=row, split=split)
    parts = {p["kind"]: p for p in r["parts"]}
    assert parts["images"]["tokens"] == 1_000 and parts["images"]["label"] == "images"
    assert parts["messages"]["tokens"] == 9_000 - 1_500 - 2_000 - 1_000
    assert r["used"] == sum(p["tokens"] for p in r["parts"] if p["kind"] != "reserved") == 9_000
    r = ctx_mod.report(**common, branch_tokens=9_000, row=row, split=None)    # an engine before the split
    assert {p["kind"]: p for p in r["parts"]}["images"]["tokens"] == 0


def test_the_next_request_counts_the_last_calls_prompt_and_answer_once():
    from chatd.service import _branch_tokens
    stats = {"prompt_tokens": 32289, "completion_tokens": 13270, "reasoning_tokens": 2193,
             "steps": 9, "step_usage": [{"step": 8, "prompt_tokens": 31445,
                                         "completion_tokens": 816},
                                        {"step": 9, "prompt_tokens": 32289,
                                         "completion_tokens": 707}]}
    branch = [{"role": "user", "parts": [{"type": "text", "text": "x" * 40}]},
              {"role": "assistant", "parts": [], "stats": stats}]
    assert _branch_tokens(branch) == 32289 + 707 - 2193
    # a one-call turn saved before the calls' own counts: as before
    one = {"prompt_tokens": 100, "completion_tokens": 50, "reasoning_tokens": 20, "steps": 1}
    assert _branch_tokens([{"role": "assistant", "parts": [], "stats": one}]) == 130
    # a turn of several calls saved before them: the prompt and the last step's text
    old = {"prompt_tokens": 100, "completion_tokens": 900, "reasoning_tokens": 20, "steps": 2}
    parts = [{"type": "text", "text": "a" * 400, "step": 1},
             {"type": "text", "text": "b" * 40, "step": 2}]
    assert _branch_tokens([{"role": "assistant", "parts": parts, "stats": old}]) == 110


def test_a_running_sandboxs_figures_go_out_with_the_context():
    """The sandbox's memory and disk went out only when its state changed (live: 20 MB and 24 KB
    of disk through an npm install of 51 MB); now with the context, at most every SANDBOX_S."""
    import types

    from chatd import service as svc_mod
    events, disk = [], [24_000]
    state = types.SimpleNamespace(state="running")
    boxes = types.SimpleNamespace(box=lambda cid: state,
                                  status=lambda cid: {"state": state.state, "disk_bytes": disk[0]})
    svc = types.SimpleNamespace(sandboxes=boxes, emit=lambda conv, ev, **kw: events.append(ev))
    conv = types.SimpleNamespace(id="c1", context={"used": 1}, fan=[1], context_sent_at=0.0,
                                 sandbox_sent=(0.0, None), parent_id=None)
    svc_mod.Service._send_context(svc, conv)
    assert [e["type"] for e in events] == ["context.changed", "sandbox.changed"]
    disk[0] = 51_098_428
    svc_mod.Service._send_context(svc, conv)                 # within SANDBOX_S: the context only
    assert [e["type"] for e in events][2:] == ["context.changed"]
    conv.sandbox_sent = (conv.sandbox_sent[0] - svc_mod.SANDBOX_S, conv.sandbox_sent[1])
    svc_mod.Service._send_context(svc, conv)
    assert events[-1] == {"type": "sandbox.changed",
                          "sandbox": {"state": "running", "disk_bytes": 51_098_428}}
    state.state = "stopped"
    conv.sandbox_sent = (0.0, None)
    svc_mod.Service._send_context(svc, conv)
    assert events[-1]["type"] == "context.changed"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
