"""One run: the agent loop that answers one user message, on its own thread (SRV-345, SRV-351).

A run belongs to chatd, not to a browser: its thread holds the stream to the engine, so the engine
never sees a client leave when a tab closes. States: queued (waiting for one of the
QSE_CHAT_MAX_RUNS slots, or for the engine), running, completed, cancelled, failed, interrupted.

THE LOOP (Claude Code's; plan note Chat 7). One step is one model call:
1. the request: the system block, the tools in fixed order, the branch, the run's earlier steps
   with their tool results, the tail. Before it the context guard (chatd/compaction.py) may
   compact the conversation, and caps `max_tokens` to what the window has left;
2. the stream: thinking and text into parts, tool calls into `tool_call` parts as they are
   written; every part carries its `step`, which is how the next request is rebuilt;
3. a step that ends with tool calls: each is checked against the permission rules (a prompt
   waits as long as it must), read-only calls run in parallel (at most 4), writes and commands in
   order, and the results go back to the model in the next step. A message queued meanwhile (a
   steer) ends the run here, and the next run carries it: the model reads it right after the
   results;
4. a step without calls ends the run: `stop` completes it, `length` gets one continuation, an
   empty turn one retry with a hidden nudge (a second empty turn is an error the user sees), a
   call the engine ended with finish reason `error` one retry (a second fails the run). A
   run that built or changed a web page or app checks it first (chatd/verify.py): the check is a
   call of the harness's own in that step, and the model reads its result in the next one.
Limits: `max_steps` model calls (100; the run ends `interrupted` with a notice and Continue goes
on), the same call with the same arguments three times in a row ends the run with a notice, and
so does the same sandbox failure (no sandbox, not the command's own error) three calls in a row,
whatever the calls were (live 2026-10-16: 28 calls with other arguments each answered
`exec failed (404)`).

WHAT STREAMS. `reasoning_content` (or `reasoning`) deltas go to a thinking part; `content` goes to a
text part, and a `<think>...</think>` block inside it goes to a thinking part too. Parts are
coalesced to one event per part every 50 ms by the service's ticker.

STOP, SEND NOW, CANCEL. `stop(mode)` closes the socket to the engine (the engine books the request
`abandoned` and frees its slot within one block) and signals a running tool to end; the text so far
stays, the message ends `stopped`, a tool that was running ends `cancelled`.

RETRY AND CONTINUE. A refused or failed call (connection refused, 429, 5xx, a stream that drops
without a finish) is retried with backoff 1, 2, 4 ... 60 s for QSE_CHAT_RETRY_S (10 minutes), then
the run ends `interrupted`. A retry after text came continues the answer with
`continue_final_message`. A 4xx is not retried: the run fails with the engine's message. The
engine's memory refusal (a 503 "a <n>-token prefill needs ... GiB with every cache given back") is
retried once after its Retry-After (the chat on the other slot may finish), then the conversation
is compacted and the smaller request sent; the guard compacts this chat under 80 % of the refused
request from then on (`Conv.memory_limit`). Refused right after that, the retries go on as above,
and the Now line says the engine has no memory for the chat's context. A 400 that says the request
is over the model's context window (a tool result can grow the prompt past the guard's estimate,
GitHub issue #18) compacts the conversation and sends the call once more when the compaction made
it smaller; when it cut nothing, or the smaller call is refused too, the run fails with the
engine's message.

WORKERS (VIS-E15). A worker is a run like any other on a task conversation (chatd/tasks.py): its
`parent` is the run whose task call started it, it runs in that run's slot (`slot_owner`), its
phases move the parent's Now line, and its end becomes the task's result.
"""

from __future__ import annotations

import collections
import json
import re
import threading
import time

from chatd import verify
from chatd.engine import EngineError
from chatd.ids import iso, now_ms, ulid

RETRY_WINDOW_S = 600.0
BACKOFF_MAX_S = 60.0
FLUSH_S = 0.05
WARMUP_TOKENS = 50            # decode rates skip the first 50 tokens (the dashboard's NOW rule)
_WAITING = re.compile(r"waiting.*?position\s+(\d+)", re.I)
# the engine's memory refusal (engine/cache.py MemoryRefused, a 503 from server/app.py `_busy`)
_MEMORY = re.compile(r"a (\d+)-token prefill needs ([\d.]+) GiB with every cache given back; "
                     r"the engine's peak is ([\d.]+) GiB")
# a request past the model's context window: our engine's 400 (server/app.py: "prompt is <n> tokens
# and the context is <m>; nothing is left to generate"), an OpenAI-style backend's or relay's
# ("maximum context length is <n> tokens", vLLM, DeepSeek; "maximum context is <n> tokens"),
# llama.cpp's and SGLang's
_WINDOW = re.compile(r"prompt is \d+ tokens and the context is \d+|maximum context (?:length )?is "
                     r"[\d,]+ tokens|exceeds the available context size|longer than the model's "
                     r"context length", re.I)
# a call id names files (the sandbox home's req-<id>.json, outputs/<id>.txt, shots/<id>.png) and
# goes into the screenshot's command line; one the backend passed through from the model with
# anything else in it is replaced (the part's id is the one the next request sends back)
_CALL_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
# the proxy's price of a relayed call, an SSE comment the client asked for (X-Tandem-Cost: 1)
COST_COMMENT = "tandem-cost "
NUDGE = "(Your last turn was empty. Answer the user now, or call a tool.)"


def cached_of(u: dict) -> int:
    """The prompt tokens a provider served from its cache: DeepSeek's `prompt_cache_hit_tokens`
    when it sends one, else the OpenAI field (our engine's)."""
    hit = u.get("prompt_cache_hit_tokens")
    if isinstance(hit, int) and not isinstance(hit, bool):
        return hit
    return int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


def memory_words(exc: EngineError) -> str | None:
    """The engine's memory refusal in plain words; None for any other refusal."""
    m = _MEMORY.search(exc.args[0]) if exc.status == 503 else None
    if m is None:
        return None
    return (f"The engine has no memory for this chat's context right now "
            f"({int(m.group(1)) // 1000}k tokens, needs {m.group(2)} GiB of {m.group(3)})")


def over_window(exc: EngineError) -> bool:
    """The refusal is a 400 for a request over the model's context window."""
    return exc.status == 400 and _WINDOW.search(exc.args[0]) is not None


class GuardError(Exception):
    """The request cannot go: it does not fit the window even after a compaction, or its user
    message is empty."""


class Run:
    def __init__(self, svc, conv, *, user_message_id: str, message_id: str, max_steps: int,
                 started_by: str | None, continue_final: bool = False, parent=None, task=None):
        self.svc = svc
        self.conv = conv
        self.id = ulid()
        self.user_message_id = user_message_id
        self.message_id = message_id
        self.started_ms = now_ms()
        self.started_by = started_by
        self.continue_final = continue_final
        self.max_steps = max_steps
        # the chat's permission mode at the start, and after a change mid-run (service.py): the
        # badge on the live view and the thread
        self.permission_mode = "ask"
        self.state = "queued"
        self.phase = "queued"
        self.now = "Queued"
        self.step = 0
        self.queue_position: int | None = None
        self.prefill: dict | None = None
        self.tool: dict | None = None
        self.request_id: str | None = None     # this model call's (each step is its own request)
        self.request_ids: set[str] = set()     # every call of the run
        self.stop_mode: str | None = None
        self.call = None
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        # the parts as they stream (the conversation's lock guards these and `pending`)
        self.open_kind: str | None = None
        self.open_index: int | None = None
        self.pend: dict = {}                   # (part index or call id, field) -> text to flush
        self.out_sent: dict[str, int] = {}     # call id -> output characters sent in events
        self.last_flush = 0.0
        self.think_started: float | None = None
        self.in_tag = False
        self.carry = ""
        self.trim = False             # the newlines right after a tag are the template's, not text
        # counts for the figures when the live row is not there; the run's, every call in them
        self.tokens = 0
        self.thinking_tokens = 0
        self.call_base = (0, 0)                # tokens, thinking_tokens before this call
        self.t_sent: float | None = None
        self.t_first: float | None = None
        # this model call's first and newest token: a relayed call's (a cloud model's) TTFT and
        # tok/s, which no usage frame times, are chatd's own from these
        self.call_first: float | None = None
        self.call_last: float | None = None
        self.backend = "engine"                 # the model's backend: engine, cloud or proxy
        self.split_est: dict | None = None      # a relayed call's prompt split, estimated
        self.cost_micros: int | None = None     # the proxy's price of the run's calls
        self.marks: list[tuple[float, int]] = []
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.metrics: dict = {"decode_tps": None, "tokens_per_round": None, "acceptance": None,
                              "thinking_budget": None, "ttft_ms": None}
        self.tps_history: list = []
        self.error: str | None = None
        self.notice: str | None = None         # why the run ended early (step cap, a loop)
        self.remark: str | None = None         # a note for the end of the answer (a gone pin)
        # the loop
        self.cancel = threading.Event()        # a running tool ends on it
        self.step_parts: list[int] = []        # this step's tool_call part indexes, in order
        self.tc_parts: dict[int, int] = {}     # this step's tool_calls[].index -> part index
        self.length_continued = False
        self.empty_retries = 0
        self.engine_errors = 0                 # calls the engine ended with finish reason error
        self.tail: str | None = None            # a hidden last user line for the next call (render.body)
        self.recent_calls: collections.deque = collections.deque(maxlen=3)
        # the sandbox's own failures in a row (not the command's): three alike end the run
        self.sandbox_errors: collections.deque = collections.deque(maxlen=3)
        self.sandbox_stuck = False
        self.interrupt: str | None = None      # why it ends `interrupted`: limit, sandbox (else engine)
        self.request_chars = 0
        self.request_images = 0                 # the request's images' tokens (not in the chars)
        self.request_est = 0
        self.request_counted: int | None = None  # the engine's exact count of it, at the edge
        self.result_cap: int | None = None      # one tool result's characters, this step (SRV-639)
        self.compacted = 0                      # the step a compaction last ran before
        self.compact_next = False               # the next request compacts first (memory, window)
        self.compact_cut = False                # the last request_for compacted the chat
        self.resume_same_step = False          # Continue of a cut text goes on in its own step
        self.acc = {"completion": 0, "reasoning": 0}
        # the speculation of the run's finished calls (their usage frames' timings), for the
        # turn's tok/s, tokens per round and acceptance: a call's own figure is one step's
        self.spec = {"predicted_n": 0, "predicted_ms": 0.0, "rounds_n": 0, "blocks": 0,
                     "draft_n": 0, "draft_accepted": 0}
        self.step_usage: list[dict] = []       # per model call: prompt, cached, cache source
        self.snapshotted = False
        self.verify = verify.new_state()       # the verify phase's checks (chatd/verify.py)
        self.waiting: dict[str, dict] = {}     # call id -> {"event", "answer"} while a prompt waits
        # VIS-E15: a worker's parent run and its tasks row; a parent's running worker, and the
        # figures of the tasks it gave (its TurnStats' `workers`)
        self.parent = parent
        self.task = task
        self.child: Run | None = None
        self.workers: list[dict] = []
        self.tasks_given = 0
        self.agents = False                    # the chat's Agents mode at the run's start
        # a worker's report (chatd/tasks.py): what it reported, sent back once, the report-only
        # call at its cap whose tool calls are stored and never run
        self.report: dict | None = None
        self.report_source: str | None = None
        self.report_misses = 0                 # the step of the report sent back, once
        self.salvage = False
        self.ignore_calls = False
        # SRV-585: the steers typed while this worker runs ((time, text), appended to its next
        # stored tool result), and a parent's task call that Continue resumes before any call
        self.notes: list[tuple[str, str]] = []
        self.resume_part: int | None = None

    # ------------------------------------------------------------------ the contract's Run
    def as_dict(self, viewer: str | None = None) -> dict:
        by = self.started_by if self.started_by and self.started_by != viewer else None
        return {
            "id": self.id, "conversation_id": self.conv.id, "message_id": self.message_id,
            "started_at": iso(self.started_ms), "started_by": by,
            "phase": self.phase, "now": self.now, "step": self.step, "max_steps": self.max_steps,
            "permission_mode": self.permission_mode,
            "queue_position": self.queue_position,
            "tool": self.tool, "prefill": self.prefill,
            "decode_tps": self.metrics["decode_tps"],
            "tokens_per_round": self.metrics["tokens_per_round"],
            "acceptance": self.metrics["acceptance"],
            # this model call's first token, once it came
            "ttft_ms": self.metrics["ttft_ms"],
            # used: the run's; call: this model call's (the budget caps each call)
            "thinking": {"used": self.thinking_tokens, "budget": self.metrics["thinking_budget"],
                         "call": self.thinking_tokens - self.call_base[1]},
            "tokens": self.tokens,
            "tps_history": list(self.tps_history[-120:]),
        }

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        self.thread = threading.Thread(target=self._main, name=f"chatd-run-{self.id[-6:]}",
                                       daemon=True)
        self.thread.start()

    def stop(self, mode: str = "stop") -> None:
        if self.stop_mode is None:
            self.stop_mode = mode
        self.wake.set()
        self.cancel.set()
        for w in list(self.waiting.values()):
            w["event"].set()
        call = self.call
        if call is not None:
            call.close()
        child = self.child
        if child is not None and child.active:      # Stop stops both (VIS-E15)
            child.stop(mode)

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")

    @property
    def slot_owner(self) -> "Run":
        """The run whose slot this one uses: a worker runs in its parent's (it takes none of its
        own, so another chat's chance to run is what it was without workers)."""
        return self.parent.slot_owner if self.parent is not None else self

    # ------------------------------------------------------------------ phases
    def set_phase(self, phase: str, now: str, **extra) -> None:
        with self.conv.lock:
            if phase == self.phase and now == self.now and not extra:
                return
            self.phase, self.now = phase, now
            ev = {"type": "run.phase", "run_id": self.id, "phase": phase, "now": now,
                  "step": self.step}
            if "queue_position" in extra:
                self.queue_position = extra["queue_position"]
                ev["queue_position"] = self.queue_position
            elif phase != "queued":
                self.queue_position = None
            if "tool" in extra:
                self.tool = extra["tool"]
                ev["tool"] = self.tool
            elif phase not in ("tool", "permission"):
                self.tool = None
            if phase != "prefill":
                self.prefill = None
            self.svc.emit(self.conv, ev, run_id=self.id)
            self.svc.summary_changed(self.conv, now=now)
        if self.parent is not None:              # a worker: `task.phase`, the parent's Now line
            self.svc.task_phase(self)

    # ------------------------------------------------------------------ the thread
    def _main(self) -> None:
        outcome = "failed"
        try:
            outcome = self._run()
        except Exception as exc:                                    # noqa: BLE001
            self.error = f"chatd: {type(exc).__name__}"
            self.svc.log(f"[run] {self.id} crashed: {type(exc).__name__}")
            outcome = "failed"
        finally:
            if self.parent is None:              # a worker's slot is its parent's
                self.svc.slots.release_if(self)
            # 2026-10-17: finish_run itself. It runs a dozen store transactions, title work,
            # event emits, a context re-send and the queue drain; any of them raising used to
            # escape the thread (threading.excepthook printed it, no state marker) and leave
            # the conversation behind a ghost: the runs row never terminal, run.active still
            # true, conv.run never cleared -- every later message queued behind a run that
            # no longer existed. service.finish_guard forces the state and drains, guarded
            # all the way down: it must never raise.
            try:
                self.svc.finish_run(self, outcome)
            except Exception as exc:                                    # noqa: BLE001
                self.svc.finish_guard(self, outcome, exc)

    def _slot(self) -> bool:
        """One of the QSE_CHAT_MAX_RUNS slots (the engine serves at most 4; opencode keeps one),
        before each model call: a run that waits for the user's permission gives its slot back
        (agent.py `_permit`). A cloud model's call takes none (`takes_slot`) and gives back the one
        an earlier call on our engine held. False when the run is stopped meanwhile."""
        if not self.svc.takes_slot(self):
            self.svc.slots.release_if(self)
            return True
        while not self.svc.slots.acquire(self, timeout=1.0):
            if self.stop_mode:
                return False
            self.set_phase("queued", "Waiting for a free run slot")
        return True

    def _run(self) -> str:
        if not self._slot():
            return self._stopped()
        self.state = "running"
        if self.resume_part is not None:
            # Continue of a parent whose worker was interrupted: the worker's loop goes on first,
            # inside the task call (chatd/tasks.py), then this run's
            self.step_parts, self.resume_part = [self.resume_part], None
            end = self._after_tools()
            if end is not None:
                return end
        while True:
            if self.stop_mode:
                return self._stopped()
            if self.step >= self.max_steps and not self.salvage:
                if self.parent is not None:
                    # a worker at its cap still reports: one more call whose only job is the
                    # report, the tool schemas kept (the prompt's head and its cache stay)
                    self.salvage = self.ignore_calls = True
                    self.tail = self.svc.salvage_text(self)
                else:
                    self.notice = (f"The run reached its limit of {self.max_steps} model calls. "
                                   f"Continue goes on from here.")
                    self.interrupt = "limit"
                    return "interrupted"
            if not self._slot():
                return self._stopped()
            outcome = self._call()
            if self.salvage and outcome in ("completed", "failed"):
                if outcome == "completed" and self.step_parts:
                    self.svc.run_tools(self, list(self.step_parts))     # stored, never run
                self.svc.task_report(self, answered=outcome == "completed")
                return "completed"
            if outcome != "completed":
                return outcome
            if self.finish_reason == "error":
                # the engine ended the call with an error of its own (its log says why), no empty
                # answer: live 2026-10-17 a crash after 8,192 tokens of thinking, retried with the
                # empty turn's nudge, failed the run as "answered with nothing, twice"
                if self.engine_errors == 0:
                    self.engine_errors = 1
                    if self.step_parts:
                        self.svc.drop_step(self)
                    continue
                self.error = ("The engine failed this model call twice (finish reason error); "
                              "the engine's log says why.")
                return "failed"
            if self.step_parts:
                end = self._after_tools()
                if end is not None:
                    return end
                continue
            if self.finish_reason == "length" and not self.length_continued:
                self.length_continued, self.continue_final = True, True
                continue
            if self.parent is None and self.svc.step_empty(self):
                if self.empty_retries == 0:
                    self.empty_retries, self.tail = 1, NUDGE
                    continue
                self.error = "The model answered with nothing, twice."
                return "failed"
            if self.svc.verify_step(self):          # the app it built, checked; the model reads it
                continue
            if self.parent is not None and self.svc.task_report(self):
                continue                            # a worker's report, sent back once
            return "completed"

    def _after_tools(self) -> str | None:
        """The step's calls run; how the run ends after them, or None: the next step."""
        done = self.svc.run_tools(self, list(self.step_parts))
        if any(self.svc.call_part(self, k).get("name") == "task" and
               self.svc.call_part(self, k).get("status") == "interrupted" for k in self.step_parts):
            self.interrupt = "task"                 # Continue resumes the worker, then this run
            return "interrupted"
        if done == "cancelled":
            return self._stopped()
        if self.sandbox_stuck:                      # Continue goes on once the sandbox is back
            self.interrupt = "sandbox"
            return "interrupted"
        if self.notice:                             # the same call three times in a row
            return "completed"
        if self.svc.queued(self.conv):              # a steer: the next run carries it
            return "completed"
        return None

    def _call(self) -> str:
        """One model call with its retries: completed, failed, interrupted or cancelled."""
        deadline = time.monotonic() + self.svc.retry_window_s
        backoff = 1.0
        refused = 0                    # the engine's memory refusals of this call, in a row
        # after an over-window 400: (the engine's message, the refused request's estimate); the
        # estimate None once the smaller request went
        over = None
        if self.resume_same_step:
            self.resume_same_step = False
        else:
            self.step += 1
        self.step_parts, self.tc_parts = [], {}
        self.finish_reason, self.usage, self.request_id = None, None, None
        self.call_base = (self.tokens, self.thinking_tokens)
        while True:
            if self.stop_mode:
                return self._stopped()
            try:
                body, headers = self.svc.request_for(self)
            except GuardError as exc:
                self.error = str(exc)
                return "failed"
            if over is not None and over[1] is not None:
                if not (self.compact_cut and self.request_est < over[1]):
                    # nothing to cut (one user turn, the big result in the last step kept: #18's
                    # shape): the same body would get the same 400
                    self.error = over[0]
                    return "failed"
                over = (over[0], None)               # the smaller request goes once
            self.set_phase("queued", "Sending to the engine")
            try:
                self.t_sent = time.monotonic()
                self.call_first = self.call_last = None
                self.call = self.svc.engine.stream(body, headers)
            except EngineError as exc:
                self.call = None
                if over is None and over_window(exc):
                    # the prompt grew past the window after the guard's estimate: compact (even
                    # when the guard compacted this step already) and send it once more, when the
                    # compaction cut it
                    over = (exc.args[0], self.request_est)
                    self.compact_next, self.compacted = True, 0
                    self.set_phase("queued", "The request is over the model's context window: "
                                             "compacting, then one retry")
                    continue
                if not exc.retryable:
                    self.error = exc.args[0]
                    return "failed"
                memory = memory_words(exc)
                if memory and refused == 0:
                    # the chat on the engine's other slot holds memory too: one retry once it may
                    # be done, then a compaction. Live 2026-10-17 the same 215,247-token prompt
                    # went every 60 s for the whole retry window and the run ended interrupted
                    refused = 1
                    wait = min(exc.retry_after or 30.0, max(1.0, deadline - time.monotonic()))
                    self.set_phase("queued", f"{memory}: retry in {int(wait)} s, then compact")
                    self.wake.wait(wait)
                    continue
                if memory and refused == 1:
                    refused = 2
                    lim = self.conv.memory_limit
                    self.conv.memory_limit = min(lim, self.request_est) if lim else self.request_est
                    self.compact_next = True
                    deadline = time.monotonic() + self.svc.retry_window_s
                    continue
                if time.monotonic() + backoff > deadline:
                    self.error = exc.args[0]
                    if memory:                       # the card under the answer says why
                        self.notice = f"{memory}. Continue goes on from here."
                    return "interrupted"
                self.set_phase("queued", f"{memory}: retry in {int(backoff)} s" if memory else
                               f"Waiting for the engine ({exc.args[0]}), retry in {int(backoff)} s")
                self.wake.wait(backoff)
                backoff = min(BACKOFF_MAX_S, backoff * 2)
                continue
            if self.stop_mode:                       # Stop came while the call opened
                self.call.close()
                return self._stopped()
            self.set_phase("prefill", "Reading the prompt")
            outcome = self._consume()
            self.call = None
            if outcome in ("completed", "failed"):
                self.continue_final = False
                return outcome
            if self.stop_mode:
                return self._stopped()
            # the stream broke with no finish: the engine went away. Retry, continuing the text.
            if time.monotonic() + backoff > deadline:
                self.error = "the engine stream ended early"
                return "interrupted"
            self.svc.flush(self)
            if self.step_parts:                      # a half-written call is not continued
                self.svc.drop_step(self)
            else:
                self.continue_final = True
            self.set_phase("queued", f"The engine stream ended early, retry in {int(backoff)} s")
            self.wake.wait(backoff)
            backoff = min(BACKOFF_MAX_S, backoff * 2)


    def _consume(self) -> str:
        finished = False
        for kind, obj in self.call.events():
            if kind == "comment":
                if (obj or "").startswith(COST_COMMENT):
                    self._cost(obj[len(COST_COMMENT):])
                    continue
                m = _WAITING.search(obj or "")
                if m:
                    pos = int(m.group(1))
                    self.set_phase("queued", f"In the engine's line, {pos} ahead",
                                   queue_position=pos)
                continue
            if kind in ("done", "end"):
                break
            if kind == "broken":
                return "broken"
            if kind != "data" or not isinstance(obj, dict):
                continue
            if obj.get("error") and not obj.get("choices"):
                err = obj["error"]
                self.error = str(err.get("message") if isinstance(err, dict) else err)[:300]
                return "failed"
            if self.request_id is None and obj.get("id"):
                self.request_id = str(obj["id"])
                self.request_ids.add(self.request_id)
            if isinstance(obj.get("usage"), dict):
                u = obj["usage"]
                self.usage = {"usage": u, "timings": obj.get("timings") or {},
                              "metrics": obj.get("metrics") or {},
                              "tandem": (u.get("tandem") or obj.get("tandem"))}
                t = obj.get("timings") or {}
                comp = int(u.get("completion_tokens") or 0)
                reas = int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
                if not t and self.call_first is not None:
                    # a relayed call (a cloud model): no engine timed it, chatd's stamps do --
                    # the first token's wait, and the usage's tokens over the stream's span
                    t = {"ttft_ms": round((self.call_first - self.t_sent) * 1000, 1),
                         "predicted_n": comp,
                         "predicted_ms": round((self.call_last - self.call_first) * 1000, 1)}
                    self.usage["timings"] = dict(t, cache_source="provider")
                self.step_usage.append({"step": self.step,
                                        "prompt_tokens": int(u.get("prompt_tokens") or 0),
                                        "cached_tokens": cached_of(u),
                                        "cache_source": t.get("cache_source") or (
                                            "provider" if self.backend != "engine" else None),
                                        "ttft_ms": t.get("ttft_ms"),
                                        "completion_tokens": comp, "reasoning_tokens": reas})
                self.acc["completion"] += comp
                self.acc["reasoning"] += reas
                self.add_spec(t)
                if comp:                             # the call's exact counts
                    self.tokens = self.call_base[0] + comp
                    self.thinking_tokens = self.call_base[1] + reas
                self.svc.learn_ratio(self.conv, int(u.get("prompt_tokens") or 0),
                                     self.request_chars, self.request_images)
            for ch in obj.get("choices") or []:
                if not isinstance(ch, dict):
                    continue
                d = ch.get("delta") or {}
                r = d.get("reasoning_content") or d.get("reasoning")
                if isinstance(r, str) and r:
                    self._add("thinking", r)
                c = d.get("content")
                if isinstance(c, str) and c:
                    self._content(c)
                for tc in d.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        self._tool_delta(tc)
                if ch.get("finish_reason"):
                    self.finish_reason = str(ch["finish_reason"])
                    finished = True
                    if self.phase != "finishing":
                        self.svc.flush(self)
                        self.set_phase("finishing", "Finishing")
            if self.stop_mode:
                return "broken"
        if self.stop_mode:
            return "broken"
        if self.carry:
            self._add("thinking" if self.in_tag else "text", self.carry)
            self.carry = ""
        return "completed" if finished or self.usage is not None else "broken"

    def _stopped(self) -> str:
        return "interrupted" if self.stop_mode == "shutdown" else "cancelled"


    # ------------------------------------------------------------------ the parts
    def _tool_delta(self, tc: dict) -> None:
        """A tool call as the model writes it: the part opens with its name, then its
        arguments grow (`part.delta` field `input`)."""
        idx = int(tc.get("index") or 0)
        fn = tc.get("function") or {}
        if idx not in self.tc_parts:
            if self.carry:
                self._add("thinking" if self.in_tag else "text", self.carry)
                self.carry = ""
            if self.phase not in ("writing", "finishing"):
                self.set_phase("writing", f"Calling {fn.get('name') or 'a tool'}")
            call_id = str(tc.get("id") or "")
            if not _CALL_ID.fullmatch(call_id) or set(call_id) == {"."}:
                call_id = f"call_{ulid()[-12:]}"
            k = self.svc.add_call(self, call_id, str(fn.get("name") or ""))
            self.tc_parts[idx] = k
            self.step_parts.append(k)
        k = self.tc_parts[idx]
        if fn.get("name") and not self.svc.call_part(self, k).get("name"):
            self.svc.call_part(self, k)["name"] = str(fn["name"])
        args = fn.get("arguments")
        if isinstance(args, str) and args:
            now = time.monotonic()
            if self.t_first is None:
                self.t_first = now
            if self.call_first is None:
                self.call_first = now
            self.call_last = now
            # a backend may stream a call's arguments several tokens a chunk (DeepSeek does):
            # one token per 3.5 characters, the harness's estimate (proxy/harness.py)
            self.tokens += max(1, int(round(len(args) / 3.5)))
            self.marks.append((now, self.tokens))
            self.svc.append_input(self, k, args)

    def _content(self, text: str) -> None:
        """Content with `<think>` tags split into thinking and text; a tag cut across two chunks
        waits in `carry` until it is whole."""
        s = self.carry + text
        self.carry = ""
        while s:
            tag = THINK_CLOSE if self.in_tag else THINK_OPEN
            i = s.find(tag)
            if i >= 0:
                head = s[:i].lstrip("\n") if self.trim else s[:i]
                if head:
                    self.trim = False
                    self._add("thinking" if self.in_tag else "text", head)
                self.in_tag = not self.in_tag
                if not self.in_tag:
                    self.svc.close_thinking(self)
                self.trim = True
                s = s[i + len(tag):]
                continue
            keep = 0
            for k in range(min(len(tag) - 1, len(s)), 0, -1):
                if tag.startswith(s[-k:]):
                    keep = k
                    break
            body = s[:len(s) - keep] if keep else s
            if self.trim and body:
                body = body.lstrip("\n")
                self.trim = not body
            if body:
                self._add("thinking" if self.in_tag else "text", body)
            self.carry = s[len(s) - keep:] if keep else ""
            break

    def _add(self, kind: str, text: str) -> None:
        now = time.monotonic()
        if self.t_first is None:
            self.t_first = now
        if self.call_first is None:
            self.call_first = now
        self.call_last = now
        self.tokens += 1
        if kind == "thinking":
            self.thinking_tokens += 1
        self.marks.append((now, self.tokens))
        if len(self.marks) > 400:
            del self.marks[:200]
        if kind == "thinking" and self.phase != "thinking":
            self.set_phase("thinking", "Thinking")
        elif kind == "text" and self.phase not in ("writing", "finishing"):
            if text.strip() or self.phase == "thinking":
                self.set_phase("writing", "Writing")
        self.svc.append(self, kind, text)

    # ------------------------------------------------------------------ figures
    def local_decode_tps(self) -> float | None:
        """Tokens over the last 2 s, after the first WARMUP_TOKENS; None when idle (the NOW rule).
        A token here is a text chunk (one token a chunk on DeepSeek: 320 chunks for 321 tokens,
        2026-10-17) or a call's argument characters / 3.5: the live figure of a relayed call,
        and of the engine's when its live row is missing."""
        if self.tokens <= WARMUP_TOKENS or len(self.marks) < 2:
            return None
        now = time.monotonic()
        t1, n1 = self.marks[-1]
        if now - t1 > 2.0:
            return None
        older = [(t, n) for t, n in self.marks if now - t <= 2.0 and n > WARMUP_TOKENS]
        if len(older) < 2:
            return None
        (t0, n0) = older[0]
        return round((n1 - n0) / (t1 - t0), 1) if t1 > t0 else None

    def live_ttft_ms(self) -> float | None:
        """This model call's wait for its first token, once it came."""
        if self.call_first is None or self.t_sent is None:
            return None
        return round((self.call_first - self.t_sent) * 1000, 1)

    def _cost(self, text: str) -> None:
        """The proxy's price of one call (`: tandem-cost {...}` before its [DONE]; proxy/relay.py),
        added to the run's; a call without a price adds nothing."""
        try:
            c = json.loads(text).get("cost_micros")
        except (ValueError, AttributeError):
            return
        if isinstance(c, int) and not isinstance(c, bool):
            self.cost_micros = (self.cost_micros or 0) + c

    def add_spec(self, t: dict) -> None:
        """One finished call's timings into the run's sums."""
        n, ms, blocks = int(t.get("predicted_n") or 0), float(t.get("predicted_ms") or 0), \
            int(t.get("blocks") or 0)
        if n > 1 and ms:
            self.spec["predicted_n"] += n - 1          # as the engine's predicted_per_second
            self.spec["predicted_ms"] += ms
        if blocks and n > 1:
            self.spec["rounds_n"] += n - 1             # the prefill's token is not a round's
            self.spec["blocks"] += blocks
        if t.get("draft_n"):
            self.spec["draft_n"] += int(t["draft_n"])
            self.spec["draft_accepted"] += int(t.get("draft_n_accepted") or 0)

    def turn_figures(self) -> dict:
        """The run's finished calls together: tok/s, tokens per round and the share of drafted
        tokens accepted; None where no call measured it."""
        s = self.spec
        return {"decode_tps": round(s["predicted_n"] * 1000 / s["predicted_ms"], 2)
                if s["predicted_ms"] else None,
                "tokens_per_round": round(s["rounds_n"] / s["blocks"], 2) if s["blocks"] else None,
                "acceptance": round(s["draft_accepted"] / s["draft_n"], 4) if s["draft_n"]
                else None}

    def stats(self, finish: str) -> dict:
        """The message's TurnStats: the usage frame's figures when it came, else chatd's own."""
        u = (self.usage or {}).get("usage") or {}
        t = (self.usage or {}).get("timings") or {}
        mt = (self.usage or {}).get("metrics") or {}
        spec = mt.get("speculative_decoding") or {}
        ptd = u.get("prompt_tokens_details") or {}
        ctd = u.get("completion_tokens_details") or {}
        turn = self.turn_figures()
        ttft = t.get("ttft_ms") if t.get("ttft_ms") is not None else mt.get(
            "time_to_first_token_ms")
        if ttft is None:
            ttft = self.live_ttft_ms()
        total = t.get("total_ms")
        if total is None and self.t_sent is not None:
            total = round((time.monotonic() - self.t_sent) * 1000, 1)
        if len(self.step_usage) > 1:
            # a turn of several calls: the first call's wait for its first token and the run's
            # time, tools included; the last call's alone said 6 s for a 38 s turn (2026-10-16)
            if self.step_usage[0].get("ttft_ms") is not None:
                ttft = self.step_usage[0]["ttft_ms"]
            total = float(now_ms() - self.started_ms)
        out = {
            "finish": finish,
            "ttft_ms": ttft,
            "decode_tps": turn["decode_tps"] or t.get("predicted_per_second") or None,
            "total_ms": total,
            "prompt_tokens": int(u.get("prompt_tokens") or 0),
            "completion_tokens": int(self.acc["completion"] or u.get("completion_tokens") or
                                     self.tokens),
            "reasoning_tokens": int(self.acc["reasoning"] or ctd.get("reasoning_tokens") or
                                    self.thinking_tokens),
            "steps": self.step,
            "step_usage": self.step_usage[-50:],
            "cached_tokens": cached_of(u),
            "tokens_per_round": turn["tokens_per_round"] or t.get("tokens_per_block") or
            spec.get("mean_acceptance_length"),
            "acceptance": turn["acceptance"] if turn["acceptance"] is not None else
            spec.get("draft_acceptance_rate"),
            "cache_source": t.get("cache_source") or ("provider" if self.backend != "engine"
                                                      and u else None),
            "prefill_tps": t.get("prompt_per_second") or None,
            "cost_usd": round(self.cost_micros / 1e6, 6) if self.cost_micros is not None
            else None,
        }
        if ptd.get("hidden_tokens"):
            out["hidden_tokens"] = int(ptd["hidden_tokens"])
        if self.workers:
            # the workers' calls beside the main agent's own (VIS-E15); the chat's usage totals
            # count both (service.py `_usage`)
            out["workers"] = {"count": len(self.workers), **{k: sum(int(w.get(k) or 0)
                                                                     for w in self.workers)
                                                            for k in ("prompt_tokens",
                                                                      "completion_tokens",
                                                                      "cached_tokens", "ms")}}
        return out
