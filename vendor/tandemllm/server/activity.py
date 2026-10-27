"""What one request is doing right now, read off the loop's own state.

The live registry (server/live.py) knew four phases: queued, prefill, decode, done. This
module says what happens inside them -- prefilling with its progress, thinking, writing, calling
tool `write_file`, finishing -- and why a request stopped, in one sentence.

THE HOT PATH, the rule this was built under. The decode loop gets no new statement per token.
Every fact here comes from one of two places:

  1. state the loop already keeps, read by the sampler thread: `ThinkBudget.inside/.n/.done/
     .reason` (engine/spec.py), the `ToolCallBuffer`'s open block and streamed arguments
     (server/toolcall.py), the running `BlockStats`, the `RequestRecord`'s own stamps;
  2. stamps written in branches that already run once per transition: the `</think>` branch of
     `ThinkBudget.observe` (`t_closed`), `_force_close` (`t_forced`), the block open / function
     name / block close of the `ToolCallBuffer` (`events`), the prefill's chunk hook
     (`on_chunk`), and the three steps after the token loop.

Every read below sits inside `_safe`: under the GIL one attribute read is atomic, two reads may
see two moments (fine for a display), and a read that raises -- a list emptied between `len` and
`[-1]` -- gives null for that field on that tick. Nothing here imports torch, calls CUDA or takes
a lock.

Definitions (also in docs/contract/dashboard-v1, live.schema.json):

    state            queued, prefilling, replaying, thinking, closing_reasoning, writing,
                     tool_call, finishing, done
    engine state     idle, busy, waiting_for_client, draining, starting
    stop.reason      stop, length, tool_calls, timeout, abandoned, error, refused, rejected,
                     cancelled; `detail` refines it (eos, stop_string, pattern_guard; the error
                     type; queue_full, queue_timeout; left_while_queued, left_during_prefill,
                     write_failed; shutting_down; prompt_too_long, bad_request)
    silent_ms        end (or now) minus the last token, or minus the lock with no token
    client_gone_ms   end minus the moment the server first saw the socket closed (the sampler
                     at its tick, or the handler at a prefill chunk, in the queue or on a write)
    conductor        (contract 1.2, SRV-115) the session monitor on this request: its mode, acts,
                     hidden tokens, the newest decision and action, and `supervising` while a
                     hidden forward runs (held for a second after it, so a 1 Hz reader sees it
                     too); null for a request the monitor does not watch. The state keeps its
                     enum; the label says "Supervising: ..." meanwhile. VIS-44 adds `shadow` (the
                     shadow decisions that were not a pass) and the decisions' `score`, `tier`
                     and `at_tokens` (the reasoning tokens at the decision), for the banner
"""

from __future__ import annotations

import select
import socket
import time

#: answered from memory, nothing decoded (server/usage.py REPLAY; kept here, this module imports
#: nothing of the server's)
REPLAY = ("response", "prepared")

STATES = ("queued", "prefilling", "replaying", "thinking", "closing_reasoning", "writing",
          "tool_call", "finishing", "done")
ENGINE_STATES = ("idle", "busy", "waiting_for_client", "draining", "starting")
STOP_REASONS = ("stop", "length", "tool_calls", "timeout", "abandoned", "error", "refused",
                "rejected", "cancelled")
FINISH_STEPS = ("flush", "saving_state", "final_chunk")
TIMELINE_N = 16

SUPERVISING_HOLD_S = 1.0          # `conductor.supervising` stays on this long after a forward

# What a read of another thread's object may raise while that thread changes it.
READ_ERRORS = (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError)


def _safe(fn, default=None):
    try:
        return fn()
    except READ_ERRORS:
        return default


def _r(x, nd: int = 2):
    return round(float(x), nd) if x is not None else None


def _ms(a, b):
    return (b - a) * 1e3 if a is not None and b is not None else None


# ------------------------------------------------------------------ the client's socket
def socket_closed(sock) -> bool:
    """The peer closed its end: the socket reads as end-of-file.

    The request body was read before generation started, so a client sends nothing more and a
    readable socket is the close (`Handler._reader_gone`'s test, done from the sampler thread).
    `MSG_PEEK` consumes nothing, so the handler's own checks see the same bytes. A socket the
    handler already closed raises and counts as closed."""
    try:
        ready, _, _ = select.select([sock], [], [], 0)
        return bool(ready) and not sock.recv(1, socket.MSG_PEEK)
    except (OSError, ValueError):
        return True


# ------------------------------------------------------------------ the state
def _refs(rec):
    refs = getattr(rec, "live_refs", None)
    return refs if refs else (None, None)


def _tool_open(tbuf) -> bool:
    if tbuf is None:
        return False
    return bool(tbuf._open) or (tbuf._jmode == "hold" and bool(tbuf._jkey))


def state_of(rec, *, ignore_step: bool = False) -> str:
    """The state of a request in flight (never `done`: the registry knows when a record ended)."""
    if rec.t_lock is None:
        return "queued"
    if rec.cache_source in REPLAY:
        return "replaying"
    if rec.t_first is None:
        return "prefilling"
    if not ignore_step and getattr(rec, "step", None) is not None:
        return "finishing"
    think, tbuf = _refs(rec)
    if _safe(lambda: _tool_open(tbuf), False):
        return "tool_call"
    if think is not None and _safe(lambda: think.inside and not think.done, False):
        return "closing_reasoning" if _safe(lambda: think.hit, False) else "thinking"
    return "writing"


_ACTION_WORDS = {"CLOSE_REASONING": "closing the reasoning", "CONTINUE_HIDDEN":
                 "continuing an empty turn", "REGENERATE_CALL": "having the tool calls written "
                 "again", "PROMPT_TAIL_NOTE": "adding a note", "SET_THINK_BUDGET":
                 "setting the thinking budget", "GIVE_UP": "giving up"}


def conductor_of(sup, now_p: float) -> dict | None:
    """The monitor's block of a request's activity (contract 1.2), read off its supervisor:
    attributes the request's own thread sets once per decision, never per token."""
    if sup is None:
        return None

    def read():
        fwd, end = sup.forwarding, sup.forwarded_at
        on = fwd is not None or (end is not None and now_p - end < SUPERVISING_HOLD_S)
        act = ({"task": fwd.task, "action": fwd.kind, "reason": fwd.reason} if fwd is not None
               else ({"task": sup.last_action["task"], "action": sup.last_action["decision"],
                      "reason": sup.last_action["reason"]} if on and sup.last_action else None))
        wall = time.time()

        def ago(d):
            # score, tier and at_tokens (VIS-44: the banner's "at 2,880 tokens (D1, 0.93)")
            return None if d is None else dict(
                {k: d[k] for k in ("task", "decision", "mode", "reason")},
                score=d.get("score"), tier=d.get("tier"), at_tokens=d.get("at_tokens"),
                ago_ms=_r(max(0.0, wall - d["at"]) * 1e3, 1))
        return {"mode": sup.mode, "acts": int(sup.acts), "hidden_tokens": int(sup.hidden_tokens),
                "shadow": int(getattr(sup, "shadows", 0)),
                "gave_up": bool(sup.gave_up), "supervising": bool(on), "action": act,
                "last": ago(sup.last), "last_action": ago(sup.last_action)}
    return _safe(read)


def closed_by(think) -> str | None:
    """Who ended the reasoning block: the model, the budget or the stall check."""
    if think is None:
        return None

    def read():
        forced = getattr(think, "t_forced", None) is not None or think.hit
        if not think.done and not forced:
            return None
        if think.reason is not None and forced:
            return "stall"
        if forced or (think.budget and think.n >= think.budget):
            return "budget"
        return "model"
    return _safe(read)


def tool_of(rec) -> dict | None:
    """The call being written: its index, name (null until `<function=NAME>` is complete),
    argument characters so far, and the calls already closed."""
    _, tbuf = _refs(rec)
    if tbuf is None:
        return None

    def read():
        name = tbuf._stream_name
        streamed = tbuf._streamed
        last = streamed[-1] if streamed else None
        if name is None and last is not None and tbuf._open:
            name = last["name"]
        if name is None:
            ev = [e for e in list(tbuf.events) if e[1] == "name"]
            name = ev[-1][2] if ev and tbuf._open else None
        idx = tbuf._stream_index
        if idx is None:
            idx = tbuf._next_index
        return {"index": int(idx), "name": name,
                "arg_bytes": len(last["args"]) if last is not None and tbuf._open else None,
                "calls_done": len(tbuf.calls)}
    return _safe(read)


def pf_mark(marks: list, done: int, total: int, now: float, window=None) -> None:
    """One prefill progress mark `(done, total, t)` for the live view (VIS-40).

    A layer-major prefill window (`engine/cache.py`, `prefill_window`) computes several chunks
    in one pass and then reports them chunk by chunk within milliseconds: timed as they are
    reported, two marks of one window read 1,024 rows in ~57 ms, an 18k tok/s "live rate" for
    an 800 tok/s prefill (2026-10-09). A chunk inside the running window (`window` = the
    engine's `(first row, end row, t)`) carries the window's own time and replaces the window's
    earlier mark, so every mark is rows computed at the time they were: one mark a window, one
    a chunk-major chunk. The list's length only grows (readers index it)."""
    if window is not None and window[0] < done <= window[1]:
        t = float(window[2])
        if marks and marks[-1][2] == t:
            marks[-1] = (done, total, t)
            return
        marks.append((done, total, t))
        return
    marks.append((done, total, now))


# ------------------------------------------------------------------ the blocks of one activity
def prefill_live(rec, warm: int = 1, window_s: float = 2.0, end_p: float | None = None) -> dict:
    """The prefill's rate with its warm-up left out, off the chunk hook's marks (VIS-40).

    `rec.pf_marks` holds one `(done, total, t)` per chunk, appended by the handler's chunk hook
    (`done` counts from position 0, so it includes the tokens restored). The anchor is the mark
    at the end of chunk `warm`; with `warm = 0` it is the loop's start (`prefill_info` start,
    t0). Everything before the anchor -- the queue, the restore, the first forward's allocation
    and warm-up, the host's lead over the GPU -- is out of both rates:

        tps_live      new tokens between the oldest and the newest mark of the last `window_s`
                      after the anchor (at least two marks; when chunks take longer than the
                      window, the last two); null before two marks exist past the anchor
        tps_live_avg  from the anchor to the newest mark; with `end_p` (the first token, a
                      finished prefill) from the anchor to the end of the whole prompt
        chunks        chunks reported so far; reached: the anchor exists

    The hook reports every chunk but the last, so a prefill of one chunk or less never has a
    rate, and a single call (`chunk = 0`) has no marks at all."""
    marks = getattr(rec, "pf_marks", None) or ()
    n = len(marks)                                   # the list only grows: indices stay good
    out = {"tps_live": None, "tps_live_avg": None, "chunks": n, "reached": False}
    info = getattr(rec, "prefill_info", None) or {}
    a = int(warm) - 1                               # the anchor's index; -1 = the loop's start
    if a < 0:
        t0, start = info.get("t0"), info.get("start")
        if t0 is None:
            return out

        def pt(i):
            return (start or 0, t0) if i < 0 else (marks[i][0], marks[i][2])
    else:
        if n <= a:
            return out

        def pt(i):
            return marks[i][0], marks[i][2]
    out["reached"] = True
    da, ta = pt(a)
    if end_p is not None:
        total = marks[-1][1] if n else rec.prompt_tokens
        if total is not None and end_p > ta and total > da:
            out["tps_live_avg"] = _r((total - da) / (end_p - ta))
        return out
    last = n - 1
    if last <= a:
        return out
    d1, t1 = pt(last)
    i, k = last - 1, last - 2
    while k >= a and t1 - pt(k)[1] <= window_s * 1.05:
        i, k = k, k - 1
    di, ti = pt(i)
    if t1 > ti:
        out["tps_live"] = _r((d1 - di) / (t1 - ti))
    if t1 > ta:
        out["tps_live_avg"] = _r((d1 - da) / (t1 - ta))
    return out


def window_rate(info: dict, now_p: float) -> tuple[float, float] | None:
    """SRV-591: (the rate, the rows computed) of the layer-major window running now
    (`info["window_live"]`, engine/cache.py `_layer_marks`): its rows times the share of its
    layers the prefill's thread has seen run (`k` at `t_k`, read between two layers), over the
    time from the window's start to then. A window is one progress mark, and only once it has run
    whole; this is its rate before then. None with no window running or before a layer has been
    seen to run. `now_p` is unused: the rate is measured to `t_k`, not to the tick."""
    live = info.get("window_live")
    if not live:
        return None
    n, k, t_k = int(live.get("n") or 0), int(live.get("k") or 0), live.get("t_k")
    if not k or not n or t_k is None or t_k <= live["t0"]:
        return None
    a, end = live["rows"]
    rows = (end - a) * k / n
    return rows / (t_k - live["t0"]), a + rows


def prefill_of(rec, now_p: float, last_prefill: dict | None = None, *, warm: int = 1,
               window_s: float = 2.0) -> dict | None:
    """Prefill progress: tokens done of total, the rates and the ETA.

    `rec.pf_marks` is one `(done, total, t)` per prefill chunk, appended by the handler's chunk
    hook; `rec.prefill_info` holds the prefill's `start` (tokens restored), `t0` (the chunk loop
    started) and `chunk` (0 = one call). The counter counts chunks the host has ISSUED; the GPU
    may run up to one chunk behind it (measured in the design note). `tps_now` is the last chunk
    and `tps_avg` runs from the loop's start, as in contract 1.1; with one mark (the first chunk
    or the first layer-major window) `tps_now` is that mark's rows over the time since the loop
    started, the same tokens a second, so a short prefill or re-read has a rate from its first
    mark on (before, it waited for a second mark, which a prefill of one window and a last chunk
    never has); while a layer-major window runs, `tps_now` is that window's rate so far
    (`window_rate`), so a prefill of one window has a rate before its end; `tps_live` and
    `tps_live_avg` leave the first `warm` chunks out
    (`prefill_live`), and the ETA uses `tps_live_avg` once it exists, so the first chunk does not
    pull it."""
    if rec.t_lock is None or rec.cache_source in REPLAY or rec.prompt_tokens is None:
        return None

    def read():
        # the served prompt (the client's tokens and the hidden ones), as the hook's marks and
        # `start` count it: rows restored plus rows computed
        total = int(rec.prompt_tokens) + int(getattr(rec, "hidden_prompt", 0) or 0)
        info = getattr(rec, "prefill_info", None) or {}
        chunk = info.get("chunk")
        mode = None if chunk is None else ("chunked" if chunk > 0 else "single_call")
        start = info.get("start")
        if rec.t_first is not None:
            # the rate is the rows computed over the prefill, never the rows restored
            cached = int(rec.cached_tokens or 0)
            fwd = rec.prompt_n
            if fwd is None and last_prefill:
                cached = int(last_prefill.get("reused") or 0)
                fwd = int(last_prefill.get("forwarded") or 0) or None
            if fwd is None:
                fwd = max(0, total - cached)
            total = max(total, cached + int(fwd))
            ms = rec.prompt_ms
            pl = prefill_live(rec, warm, window_s, end_p=rec.t_first)
            return {"done": total, "total": total, "cached": cached, "pct": 100.0,
                    "tps_now": None, "tps_avg": _r(fwd / (ms / 1e3)) if fwd and ms else None,
                    "eta_ms": None, "progress": mode or "chunked",
                    "at_ms": _r((rec.t_first - rec.t_arrival) * 1e3, 1),
                    "tps_live": None, "tps_live_avg": pl["tps_live_avg"],
                    "why": info.get("why"),
                    "warmup": {"chunks": pl["chunks"], "reached": pl["reached"]}}
        marks = getattr(rec, "pf_marks", None)
        t0 = info.get("t0")
        pl = prefill_live(rec, warm, window_s)
        win = window_rate(info, now_p)
        if not marks:
            eta = max(0.0, (total - win[1]) / win[0] * 1e3) if win else None
            return {"done": start, "total": total, "cached": start, "pct": None,
                    "tps_now": _r(win[0]) if win else None, "tps_avg": None,
                    "eta_ms": _r(eta, 1), "progress": mode,
                    "at_ms": None, "tps_live": pl["tps_live"],
                    "tps_live_avg": pl["tps_live_avg"], "why": info.get("why"),
                    "warmup": {"chunks": 0, "reached": pl["reached"]}}
        n = len(marks)
        done, total, t = marks[n - 1]                 # the hook's own total: what is prefilled
        pdone, pt = (marks[n - 2][0], marks[n - 2][2]) if n > 1 else (None, None)
        base = start if start is not None else 0
        tps_avg = (done - base) / (t - t0) if t0 is not None and t > t0 and done > base else None
        tps_now = (done - pdone) / (t - pt) if pdone is not None and t > pt else None
        if n == 1:
            tps_now = tps_avg                 # the first mark: rows computed since the start
        if win is not None:
            tps_now = win[0]                  # the window running now, past the last mark
        rate = pl["tps_live_avg"] or tps_avg
        eta = max(0.0, (total - done) / rate * 1e3) if rate else None
        return {"done": int(done), "total": total, "cached": start,
                "pct": _r(100.0 * done / total, 1) if total else None,
                "tps_now": _r(tps_now), "tps_avg": _r(tps_avg), "eta_ms": _r(eta, 1),
                "progress": mode or "chunked",
                "at_ms": _r((t - rec.t_arrival) * 1e3, 1),
                "tps_live": pl["tps_live"], "tps_live_avg": pl["tps_live_avg"],
                "why": info.get("why"),
                "warmup": {"chunks": n, "reached": pl["reached"]}}
    return _safe(read)


def accept_mean(bs) -> float | None:
    """Mean draft tokens accepted per block that had a draft, from the first-miss histogram.
    The dict is copied with one `list(...)` call so a resize cannot raise mid-iteration."""
    if bs is None:
        return None

    def read():
        n = s = 0
        for _depth, h in list(bs.accept.items()):
            for a, c in list(h.items()):
                n += c
                s += a * c
        return _r(s / n) if n else None
    return _safe(read)


def decode_of(rec, now_p: float, bs, fast: tuple | None, tool_tokens: int = 0, *,
              live: bool = True) -> dict | None:
    """Decode figures. `bs` is the running `BlockStats` (None for a finished record, which
    carries its own counts, and for a replay); `fast` is `(t, tokens, blocks)` about a second
    ago, for the `_now` figures."""
    if rec.t_first is None:
        return None

    def read():
        n = int(rec.n_live) if live else int(rec.completion_tokens or rec.n_live)
        think, _ = _refs(rec)
        th = min(int(think.n), n) if think is not None and rec.thinking else 0
        if bs is not None:
            blocks = int(bs.blocks)
            span = _ms(bs.t_first, bs.t_last)
        else:
            blocks = int(rec.blocks or 0)
            span = rec.predicted_ms
        dms = (now_p - rec.t_first) * 1e3 if live else rec.predicted_ms
        tps_avg = (n - 1) / (dms / 1e3) if n > 1 and dms else None
        out = {"tokens": n, "thinking_tokens": th,
               "content_tokens": max(0, n - th - int(tool_tokens)),
               "tool_tokens": int(tool_tokens),
               "tps_now": None, "tps_avg": _r(tps_avg), "rounds": blocks or None,
               "tokens_per_round": _r((n - 1) / blocks) if blocks and n > 1 else None,
               "tokens_per_round_now": None,
               "ms_per_round": _r(span / blocks) if blocks and span else None,
               "ms_per_round_now": None,
               "accept_mean": accept_mean(bs)}
        if fast is not None and live:
            t0, n0, b0 = fast
            dt = now_p - t0
            if dt > 0:
                out["tps_now"] = _r((n - n0) / dt)
                db = blocks - b0 if bs is not None else 0
                if db > 0:
                    out["tokens_per_round_now"] = _r((n - n0) / db)
                    out["ms_per_round_now"] = _r(dt * 1e3 / db)
        return out
    return _safe(read)


# ------------------------------------------------------------------ the timeline
def events_of(rec) -> list[tuple]:
    """Every transition of a request as `(t, state, detail)`, from stamps written once each."""
    ev = [(rec.t_arrival, "queued", None)]
    if rec.t_lock is not None:
        if rec.cache_source in REPLAY:
            ev.append((rec.t_lock, "replaying", None))
        else:
            info = getattr(rec, "prefill_info", None) or {}
            start = _safe(lambda: info.get("start"))
            ev.append((rec.t_lock, "prefilling", f"{start} cached" if start else None))
    think, tbuf = _refs(rec)
    if rec.t_first is not None and rec.cache_source not in REPLAY:
        closed = _safe(lambda: think.t_closed) if think is not None else None
        forced = _safe(lambda: think.t_forced) if think is not None else None
        first_thinking = bool(rec.thinking) and not (closed is not None and closed <= rec.t_first)
        ev.append((rec.t_first, "thinking" if first_thinking else "writing", None))
        if forced is not None:
            ev.append((forced, "closing_reasoning", closed_by(think)))
        if closed is not None and closed > rec.t_first:
            ev.append((closed, "writing", None))
        if tbuf is not None:
            for t, kind, name in _safe(lambda: list(tbuf.events), []):
                if kind == "open":
                    ev.append((t, "tool_call", None))
                elif kind == "name":
                    ev.append((t, "tool_call", name))
                elif kind == "close":
                    ev.append((t, "writing", None))
    t_fin = getattr(rec, "t_finishing", None)
    if t_fin is not None:
        ev.append((t_fin, "finishing", None))
    ev.sort(key=lambda e: e[0])
    out: list[tuple] = []
    for t, state, detail in ev:
        if out and out[-1][1] == state:
            if detail is not None and out[-1][2] is None:
                out[-1] = (out[-1][0], state, detail)       # the tool's name, once decoded
            continue
        out.append((t, state, detail))
    return out


def timeline_of(rec, end_p: float | None = None, stop_reason: str | None = None) -> list[dict]:
    """The last `TIMELINE_N` transitions, times in ms from arrival."""
    ev = events_of(rec)
    if end_p is not None:
        ev.append((end_p, "done", stop_reason))
    out = []
    for t, state, detail in ev[-TIMELINE_N:]:
        row = {"t_ms": _r(max(0.0, (t - rec.t_arrival) * 1e3), 1), "state": state}
        if detail is not None:
            row["detail"] = str(detail)
        out.append(row)
    return out


def path_of(rec) -> list[str]:
    seen: list[str] = []
    for _t, state, _d in events_of(rec):
        if not seen or seen[-1] != state:
            seen.append(state)
    return seen


# ------------------------------------------------------------------ the stop
def stop_of(rec, end_p: float) -> dict:
    """The stop block of a finished request: why, in which state, and how quiet it was."""
    fr, status = rec.finish_reason, int(rec.status or 0)
    detail = getattr(rec, "stop_detail", None)
    if detail == "shutting_down":
        reason = "cancelled"
    elif status == 400:
        reason, detail = "rejected", detail or "bad_request"
    elif fr == "refused" or (fr is None and status in (429, 503)):
        reason = "refused"
    elif fr in ("stop", "length", "tool_calls", "timeout", "abandoned"):
        reason = fr
    else:
        reason = "error"
        detail = detail or rec.error_type or ("internal_error" if status >= 500 else None)
    if reason == "error" and rec.error_type:
        detail = rec.error_type
    if reason == "abandoned" and detail is None:
        detail = ("left_while_queued" if rec.t_lock is None else
                  "left_during_prefill" if rec.t_first is None else "write_failed")
    state = getattr(rec, "end_state", None)
    if reason == "tool_calls":
        state = "tool_call"
    elif state is None:
        state = "queued" if rec.t_lock is None else state_of(rec, ignore_step=True)
    last = rec.t_last if rec.t_last is not None else rec.t_lock
    gone = getattr(rec, "client_gone_at", None)
    out = {"reason": reason, "detail": detail, "state": state,
           "tokens_sent": int(rec.n_live or rec.completion_tokens or 0),
           "silent_ms": _r(_ms(last, end_p), 1) if last is not None else None,
           "client_gone_ms": _r(_ms(gone, end_p), 1) if gone is not None else None}
    out["sentence"] = sentence(out, rec, end_p)
    return out


# ------------------------------------------------------------------ the words
def _n(x) -> str:
    return f"{int(x):,}"


def _dur(ms) -> str:
    if ms is None:
        return "?"
    s = ms / 1e3
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s:.0f} s"
    return f"{int(s // 3600)} h {int(s % 3600 // 60)} min"


def _ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


_SILENT = {"queued": "in the queue", "prefilling": "of silent prefill",
           "replaying": "while replaying", "thinking": "while thinking",
           "closing_reasoning": "while closing the reasoning", "writing": "while writing",
           "tool_call": "while writing a tool call", "finishing": "while finishing"}


def sentence(stop: dict, rec=None, end_p: float | None = None) -> str:
    """One sentence for a stop, the same words for every client."""
    reason, detail, state = stop.get("reason"), stop.get("detail"), stop.get("state")
    sent = stop.get("tokens_sent") or 0
    elapsed = _ms(rec.t_arrival, end_p) if rec is not None and end_p is not None else None
    tokens = f"{_n(sent)} token{'s' if sent != 1 else ''} sent"
    if reason == "abandoned":
        if state == "queued":
            return f"abandoned by the client after {_dur(elapsed)} in the queue, {tokens}"
        if state == "prefilling":
            since = _ms(rec.t_lock, end_p) if rec is not None and rec.t_lock else elapsed
            return f"abandoned by the client after {_dur(since)} of silent prefill, {tokens}"
        return (f"abandoned by the client {_SILENT.get(state, 'while running')} after "
                f"{_dur(elapsed)}, {tokens}")
    if reason == "tool_calls":
        names = list(getattr(rec, "tool_names", None) or []) if rec is not None else []
        n = len(names) or int(getattr(rec, "tool_calls", 0) or 0)
        what = ", ".join(names) if names else "a tool"
        count = f" ({n} calls)" if n > 1 else ""
        return f"tool call: {what}{count} after {_dur(elapsed)}"
    if reason == "length":
        cap = getattr(rec, "max_tokens", None) if rec is not None else None
        return (f"stopped at the length limit ({_n(cap)} tokens)" if cap
                else "stopped at the length limit")
    if reason == "timeout":
        return f"timed out after {_dur(elapsed)}"
    if reason == "refused":
        if detail == "queue_full":
            return "refused: queue full"
        if detail == "queue_timeout":
            return f"refused: timed out in the queue after {_dur(elapsed)}"
        return "refused: the engine is busy"
    if reason == "rejected":
        if detail == "prompt_too_long":
            return "rejected: the prompt is longer than the context window"
        return "rejected: a bad request (400)"
    if reason == "cancelled":
        return "cancelled: the server was shutting down"
    if reason == "error":
        return f"error: {detail or 'internal error'} after {_n(sent)} tokens"
    how = {"stop_string": "at a stop string", "pattern_guard": "by the repetition guard"}.get(
        detail, "at the end of the answer")
    return f"finished {how} after {_dur(elapsed)}, {tokens}"


def label(act: dict) -> str:
    """The short phrase for an activity, e.g. "Prefilling 36,864 of 48,210 (76 %)"."""
    state = act.get("state")
    c = act.get("conductor") or {}
    if c.get("supervising") and c.get("action") and state != "done":
        a = c["action"]
        return (f"Supervising: {a['task']} {_ACTION_WORDS.get(a['action'], a['action'])}"
                + (f" ({a['reason']})" if a.get("reason") else ""))
    if state == "queued":
        q = act.get("queue") or {}
        place = q.get("place")
        base = f"Queued, about {_ordinal(place)} in line" if place else "Queued"
        # 0.8.2: what it waits for (server/app.py `_queue_why`), e.g. "all 3 slots are busy"
        return f"{base}: {q['reason']}" if q.get("reason") else base
    if state == "prefilling":
        p = act.get("prefill") or {}
        done, total, pct = p.get("done"), p.get("total"), p.get("pct")
        if done is not None and total and pct is not None:
            return f"Prefilling {_n(done)} of {_n(total)} ({pct:.0f} %)"
        return f"Prefilling {_n(total)} tokens" if total else "Prefilling"
    if state == "replaying":
        return "Replaying a cached answer"
    if state == "thinking":
        return "Thinking"
    if state == "closing_reasoning":
        by = (act.get("reasoning") or {}).get("closed_by")
        return f"Closing the reasoning ({by})" if by else "Closing the reasoning"
    if state == "writing":
        return "Writing"
    if state == "tool_call":
        name = (act.get("tool") or {}).get("name")
        return f"Calling tool {name}" if name else "Calling a tool"
    if state == "finishing":
        step = act.get("step")
        return {"flush": "Finishing: flushing the last text",
                "saving_state": "Finishing: saving the session state",
                "final_chunk": "Finishing: sending the final chunk"}.get(step, "Finishing")
    if state == "done":
        stop = act.get("stop") or {}
        return stop.get("sentence") or "Done"
    return str(state or "")


def encoding_label(rec) -> str | None:
    """"Encoding image 1 of 2" while the vision tower runs for this request, else None.
    `rec.encoding` is `(i, n)`, stamped by the prefill around each image it encodes."""
    def read():
        enc = getattr(rec, "encoding", None)
        if not enc:
            return None
        i, n = enc
        return f"Encoding image {int(i)} of {int(n)}" if int(n) > 1 else "Encoding image"
    return _safe(read)


def waiting_label(w: dict) -> str:
    names = w.get("tool_names") or []
    return f"Waiting for client: running tool {', '.join(names)}" if names else \
        "Waiting for client"
