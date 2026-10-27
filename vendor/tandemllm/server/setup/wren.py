"""Wren, the setup guide's assistant (SRV-325, decision D6 of the setup guide's design).

A small Qwen3.5 model on the engine's own forward (engine/generative/qwen35.py), in the plainest
configuration the engine has: no session, prefix or state cache, no suffix store, no disk tier, no
drafters, no conductor, one sequence, a window of WINDOW rows, thinking off, greedy. Qwen3.5-2B, or
Qwen3.5-0.8B when the 2B is not on disk. It answers questions about the setup page from the docs,
the manual, the model registry and the packages that describe TandemLLM (server/setup/knowledge.py),
reads the page's state and the box's read-only facts every turn, and asks the page for changes with
tool calls (server/setup/tools.py) that the browser executes. The server never applies one.

LOADING. Nothing loads on its own in a serving engine: `POST /v1/setup/assistant/load` asks.
  * In setup mode (`server/app.py --setup`, no model of Qwen's loaded) it loads at the start.
  * On a serving engine whose narrator is loaded with the same model, Wren BORROWS the narrator's
    slot: no second copy of the weights, the narrator's 4,096-row cache, and a claim of nothing.
    The slot's model keeps one generation at a time; a narrator job that finds Wren's rows in the
    cache starts its prompt over (engine/generative/qwen35.py), so both stay exact.
  * Otherwise Wren loads its own copy through the residency planner (role `wren`, the model's
    declared claim) like any role, and a refusal is a normal answer: the page works without chat.
Wren goes first when the KV pool needs memory (server/app.py `_kv_reclaim_roles`, before the
shadow reader and the narrator), and it unloads itself IDLE_UNLOAD_S after the last status poll or
chat on a serving engine (a closed tab must not keep 3.6 GiB). Nothing loads it again but a load
call.

GENERATION NEVER SLOWS A REQUEST. Wren generates in slices of SLICE_MS, each under the engine's
lock taken in its own rank, after every request and every job of TandemLLM's own, and only while
no request runs or waits (the shadow reader's rule). A request that arrives waits at most one
slice; Wren then waits until the engine is idle again.

THE PROMPT, every turn: the persona and rules (wren_system.md), the page state as compact JSON
(fields named like a secret are only "<set>" or "<empty>"), the box's facts, the top sections of
the reference text for the user's last message and the current step, the step itself (the ids a
tool call may name there, and on the model, overview and weights steps the measured table), and
the conversation, cut from its oldest turn to fit the window. Chat content is never logged; the
counters are.

A TURN ENDS IN WORDS (VIS-91). An invalid change gets one repair with the model seeing why; when
that is invalid too, or the model only pointed at something and said nothing, it answers once more
with no tools in the prompt, and any call it still writes is dropped. A highlight it may not make
is dropped without a repair. The page asks with `tools: false` for its own last round.
"""

from __future__ import annotations

import collections
import json
import os
import re
import threading
import time

from server.setup import registry, tools
from server.setup.knowledge import Knowledge

NAME = "Wren"
CHOICES = ("qwen3.5-2b", "qwen3.5-0.8b")
MODELS_DIR = "~/.qwen38-spark-engine/models"
WINDOW = 8192
MAX_NEW = 512
SLICE_MS = 150.0
IDLE_UNLOAD_S = 300.0
IDLE_CHECK_S = 15.0
REPAIRS = 1
TURN_S = 180.0
MAX_MESSAGES = 60
MAX_TEXT = 8000
# what one chat body may hold beyond that: all messages' text together, the tool calls of one
# message and the JSON of one call's arguments, the page state's JSON; and how often one prompt is
# rendered through the chat template while old turns are dropped to fit the window
MAX_TOTAL_TEXT = 64_000
MAX_CALLS = 8
MAX_ARGS = 2_000
MAX_PAGE = 64 * 1024
MAX_RENDERS = 6
LOAD_WAIT_S = 900.0
RANK = 5          # after every request (0-2) and TandemLLM's own jobs (server/app.py RANK_SHADOW = 4)
SECRET = re.compile(r"pass|token|secret|api_?key|hf_", re.I)
TOOL_OPEN = "<tool_call>"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wren_system.md")) as _fh:
    SYSTEM = _fh.read().strip()
# who Wren is goes first; its rules go last, next to the question. Measured on the box with
# Qwen3.5-2B (2026-10-11, ten plain requests, one tool call each): with the rules first 5 of 10 calls
# were the right one; rules last, the field names in set_field's description and the step names
# in go_to_step's: 7 of 10, and 8 of 10 with the page's visible fields in the page state
INTRO, _, RULES = SYSTEM.partition("\n\nHow you answer:")
RULES = "How you answer:" + RULES
NO_TOOLS = "The tools are off for this answer: answer the user in words only."
NO_WORDS = "I could not answer that here. Please ask it in other words."
NOT_CHANGED = "I could not make that change on the page, so nothing changed."
TABLE_STEPS = ("models", "overview", "weights")
# what a step is about, for the search: "overview" names no section, and on the Model step "Which
# one is for coding tasks?" found the conductor's coding profile instead of the weight profiles
STEP_WORDS = {"overview": "weight profile speed tok/s", "weights": "weight profile"}


def model_dir(name: str) -> str:
    return os.path.join(os.path.expanduser(MODELS_DIR), name)


def choose() -> tuple[str | None, str | None, bool]:
    """(model name, its folder, a fallback) for the first of CHOICES on disk; (None, None, False).
    QSE_WREN_MODEL puts another Qwen3.5 size first (qwen3.5-4b: 12 of 13 right on the box's tool
    and question set against the 2B's 10, at about 2.5 times the time a turn and its own copy)."""
    first = (os.environ.get("QSE_WREN_MODEL") or "").strip()
    choices = ((first,) if first in ("qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b") else ()) + \
        tuple(c for c in CHOICES if c != first)
    for i, name in enumerate(choices):
        d = model_dir(name)
        if os.path.isfile(os.path.join(d, "config.json")) and os.path.isfile(
                os.path.join(d, "tokenizer.json")):
            return name, d, i > 0
    return None, None, False


def claim_gib(name: str | None) -> float:
    from engine.generative.qwen35 import NAMES
    return float((NAMES.get(name or "") or {}).get("claim_gib") or 0.0)


def _secret_free(value, key: str = ""):
    """The page state with every value of a key that looks like a secret replaced."""
    if isinstance(value, dict):
        return {k: _secret_free(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_secret_free(v, key) for v in value]
    if key and SECRET.search(key) and isinstance(value, str) and value not in ("keep", "generate"):
        return "<set>" if value else "<empty>"
    return value


def page_text(page: dict, cap: int) -> str:
    """The page state as compact JSON, at most `cap` characters: the current step's view first,
    then the estimate, then the selection; a part that does not fit is left out whole."""
    page = _secret_free(page if isinstance(page, dict) else {})
    head = {k: page.get(k) for k in ("model", "step", "step_index")}
    parts = [("visible", page.get("visible")), ("estimate", page.get("estimate")),
             ("selection", page.get("selection"))]
    out = dict(head)
    for k, v in parts:
        if v is None:
            continue
        trial = dict(out, **{k: v})
        if len(json.dumps(trial, separators=(",", ":"))) <= cap:
            out = trial
        else:
            out[k] = "(left out: too long)"
    return json.dumps(out, separators=(",", ":"))[:cap]


def step_text(m: dict, step) -> str:
    """The current step for the prompt: the ids a tool call may name on it, and on the model,
    overview and weights steps the measured table the page shows (VIS-91: "Which one is for coding
    tasks?" on the Model step had no table to answer from)."""
    label = dict(registry.STEPS).get(step, step) if isinstance(step, str) else None
    out = [f"This step: {step} ({label})." if label else "This step: unknown."]
    ids = tools.step_ids(m, step)
    if not ids:
        out.append("This step has no fields to change or highlight.")
    else:
        verb = {"weights": "select_profile and explain", "components": "toggle and explain"}.get(
            step, "set_field and explain")
        out.append(f"The ids on this step ({verb}): {', '.join(ids)}.")
    par = m.get("parallelism") or {}
    if step == "parallelism" and par.get("kv_bytes_per_token"):
        kv, fill = par["kv_bytes_per_token"], par.get("typical_fill_tokens") or 32768
        top = (par.get("window") or {}).get("max") or fill
        out.append(f"What a slot takes: {kv / 1024:g} KiB of memory a token of its window. With "
                   f"the shared KV pool a slot takes memory as its conversation grows: "
                   f"{kv * fill / 2**30:.2f} GiB at {fill:,} tokens, {kv * top / 2**30:.2f} GiB "
                   f"with its whole window of {top:,} tokens. Each slot after the first also "
                   f"takes {par.get('side_slot_const_gib', 0):g} GiB.")
    meas = m.get("measurements") or {}
    if step in TABLE_STEPS and meas.get("rows"):
        prof = {p["id"]: p for p in m["profiles"]}
        tasks = ("prose", "chat", "code")
        out.append(f"The measured table: the weight profiles and how fast each one ran, in "
                   f"tokens a second (greedy, 512 new tokens, build {meas.get('build')}, "
                   f"{meas.get('date')}). Code is the column for coding tasks. KL is how far the "
                   f"weights drift from the original model; lower is closer:")
        out.append("profile | " + " | ".join(tasks) + " | KL | weights GiB")
        best = {}
        for r in meas["rows"]:
            p = prof.get(r["profile"]) or {}
            cells = r["cells"].get("greedy_512") or {}
            speed = [(cells.get(t) or {}).get("tok_s") for t in tasks]
            out.append(" | ".join([r["profile"]] + ["-" if v is None else f"{v:g}" for v in speed]
                                  + [f"{(p.get('quality') or {}).get('kl_vs_bf16', '-')}",
                                     f"{p.get('weights_gib', '-')}"])
                       + ("" if r.get("comparable") else " (another build)"))
            for t, v in zip(tasks, speed):
                if v is not None and r.get("comparable") and v > best.get(t, ("", 0))[1]:
                    best[t] = (r["profile"], v)
        out.append("; ".join(f"fastest for {t}: {b[0]} ({b[1]:g} tok/s)" for t, b in best.items())
                   + "; closest to the original: " + min(
                       (p for p in m["profiles"] if (p.get("quality") or {}).get("kl_vs_bf16")
                        is not None), key=lambda p: p["quality"]["kl_vs_bf16"])["id"]
                   + ". Recommended: " + ", ".join(p["id"] for p in m["profiles"]
                                                   if p.get("recommended")) + ".")
    return "\n".join(out)


def check_messages(messages) -> list[dict]:
    """The conversation the browser sent, in the template's form; ValueError when it is not one."""
    if not isinstance(messages, list) or not messages or len(messages) > MAX_MESSAGES:
        raise ValueError(f"messages: a list of 1 to {MAX_MESSAGES} messages")
    out, total = [], 0
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant", "tool"):
            raise ValueError(f"messages[{i}]: a user, assistant or tool message")
        content = msg.get("content") or ""
        if not isinstance(content, str) or len(content) > MAX_TEXT:
            raise ValueError(f"messages[{i}].content: text of at most {MAX_TEXT} characters")
        total += len(content)
        if total > MAX_TOTAL_TEXT:
            raise ValueError(f"messages: at most {MAX_TOTAL_TEXT} characters together; clear the "
                             f"chat and ask again")
        m = {"role": msg["role"], "content": content}
        calls = msg.get("tool_calls") or []
        if msg["role"] == "assistant" and calls:
            if not isinstance(calls, list) or len(calls) > MAX_CALLS:
                raise ValueError(f"messages[{i}].tool_calls: a list of at most {MAX_CALLS}")
            for c in calls:
                if not isinstance(c, dict) or len(json.dumps(c.get("args"))) > MAX_ARGS or \
                        len(str(c.get("name") or "")) > 64:
                    raise ValueError(f"messages[{i}].tool_calls: a name and at most {MAX_ARGS} "
                                     f"characters of arguments each")
            m["tool_calls"] = [{"type": "function", "function": {
                "name": str(c.get("name") or ""), "arguments": c.get("args") if
                isinstance(c.get("args"), dict) else {}}} for c in calls if isinstance(c, dict)]
        out.append(m)
    if not any(m["role"] == "user" for m in out):
        raise ValueError("messages: at least one user message")
    return out


class Wren:
    def __init__(self, app, *, device_facts=None, model_factory=None, setup_mode: bool = False,
                 log=print, clock=time.monotonic):
        """`app`: the server module (its STATE, LOCK and request counters); `device_facts()`: the
        box's read-only facts for the prompt; `model_factory(name, folder)`: the model (tests give
        a fake one), default the Qwen3.5 model at WINDOW rows."""
        self.app, self.device_facts, self.log, self.clock = app, device_facts, log, clock
        self.model_factory = model_factory
        self.setup_mode = setup_mode
        self.slot = None
        self.borrowed = False
        self.refusal: dict | None = None
        self.error: str | None = None
        self.last_touch = clock()
        self.knowledge = Knowledge()
        self.stats = collections.Counter()
        self.load_started: float | None = None
        self._lock = threading.RLock()
        self._chat = threading.Lock()
        self._stop = threading.Event()
        self._reaper = None

    # ------------------------------------------------------------------ state
    def touch(self) -> None:
        self.last_touch = self.clock()

    @property
    def state(self) -> str:
        with self._lock:
            if self.slot is None:
                if self.error:
                    return "failed"
                return "refused" if self.refusal else "not_loaded"
            st = self.slot.state
            if self.borrowed and st != "loaded":
                return "not_loaded"                     # the narrator went; the borrow with it
            return {"not_loaded": "not_loaded"}.get(st, st)

    def status(self) -> dict:
        name, _, fallback = choose()
        st = self.state
        slot = self.slot
        reason = None
        if name is None:
            reason = (f"neither Qwen3.5-2B nor Qwen3.5-0.8B is in {MODELS_DIR}. Download it with: "
                      f"hf download Qwen/Qwen3.5-2B --local-dir {MODELS_DIR}/qwen3.5-2b")
        elif st == "refused" and self.refusal:
            r = self.refusal
            reason = (f"not enough memory beside the engine: {r.get('after_gib')} GiB would be left, "
                      f"{r.get('need_gib')} GiB must stay free"
                      if r.get("reason") == "refused_memory" else
                      f"the residency planner said no ({r.get('reason')})")
        elif st == "failed":
            reason = self.error or (slot.error if slot is not None else None)
        model = (slot.model.name if slot is not None else name)
        dev = None
        if slot is not None and st == "loaded":
            try:
                dev = round(slot.device_bytes() / 2**30, 2)
            except Exception:                                      # noqa: BLE001
                dev = None
        return {"name": NAME, "model": model, "fallback": bool(fallback and model != CHOICES[0]),
                "state": st, "reason": reason,
                "load_s": getattr(slot, "load_s", None) if slot is not None else None,
                "device_gib": dev,
                "claim_gib": 0.0 if self.borrowed else (claim_gib(model) if model else None),
                "can_load": name is not None and st in ("not_loaded", "refused", "failed"),
                "borrowed": self.borrowed, "window": self.window(),
                "loading_s": (round(self.clock() - self.load_started, 1)
                              if st == "loading" and self.load_started else None),
                "stats": dict(self.stats)}

    def window(self) -> int:
        slot = self.slot
        return int(getattr(getattr(slot, "model", None), "max_ctx", WINDOW)) if slot else WINDOW

    # ------------------------------------------------------------------ load and unload
    def _narrator_slot(self, name: str):
        narr = self.app.STATE.get("narrator")
        slot = getattr(narr, "slot", None)
        if slot is not None and getattr(slot.model, "name", None) == name and slot.state == "loaded":
            return slot
        return None

    def load(self) -> dict:
        with self._lock:
            self.touch()
            if self.state in ("loaded", "loading"):
                return self.status()
            name, folder, _ = choose()
            if name is None and self.model_factory is None:
                self.error = "not_on_disk"
                return self.status()
            self.error, self.refusal = None, None
            borrow = None if self.setup_mode else self._narrator_slot(name)
            if borrow is not None:
                self.slot, self.borrowed = borrow, True
                self.log(f"[wren] {name}: borrows the narrator's loaded copy (no second load, "
                         f"{self.window()} rows)")
                self._start_reaper()
                return self.status()
            from engine.generative.slot import Slot
            model = (self.model_factory(name, folder) if self.model_factory is not None
                     else self._qwen(name, folder))
            device = None
            if self.model_factory is None:
                from engine.decision.device import Device
                device = Device("auto")
            self.slot = Slot(model, role="wren", gate=self.app.LOCK, planner=self._planner,
                             device=device, log=lambda s: self.log(s),
                             mem_available=self._mem_available)
            self.borrowed = False
            self.load_started = self.clock()
            d = self.slot.load()
            if not d.get("allowed"):
                self.refusal = d
                self.slot = None
                self.log(f"[wren] {name}: load refused ({d.get('reason')})")
            else:
                self._start_reaper()
            return self.status()

    def _qwen(self, name: str, folder: str):
        from engine.generative.qwen35 import Qwen35Model
        return Qwen35Model(name, folder, max_ctx=WINDOW)

    def _mem_available(self):
        from server import decider_api
        return decider_api._mem_available_gib(False)

    def _planner(self, name, *, loading, service_start, hold_restart):
        """The residency planner's check for role `wren`, its claim added to a residency document
        that has none (the stored ones predate Wren)."""
        from server import decider_api, memlimits
        from server.conductor import residency, schema
        st = self.app.STATE
        cond = st.get("conductor")
        doc = cond.store.peek("residency")[1] if cond is not None else schema.seed("residency")
        roles = cond.store.peek("roles")[1] if cond is not None else schema.seed("roles")
        if "wren" not in (doc.get("claims") or {}):
            doc = dict(doc, claims=dict(doc.get("claims") or {}, wren={
                "gib": claim_gib(name), "basis": "declared",
                "note": "Wren, the setup guide's assistant: the Qwen3.5 model's declared claim"}))
        others = [st.get("narrator"), st.get("shadow")]
        busy_load = any(getattr(getattr(o, "slot", None), "state", None) == "loading"
                        for o in others if o is not None)
        reg = st.get("deciders")
        busy_load = busy_load or bool(reg is not None and reg.loading())
        peak, now = memlimits.engine_gib(st)
        d = residency.check("wren", decider_api._mem_available_gib(False), doc, roles,
                            hold=decider_api.held(service_start=service_start,
                                                  hold_restart=hold_restart),
                            loading=loading or busy_load, engine_peak_gib=peak, engine_now_gib=now)
        if cond is not None:
            residency.Journal(cond.root).write(dict(d, kind="check", source=f"engine:{name}"))
        return d

    def unload(self, why: str = "asked") -> dict:
        """Drop Wren: its own copy unloaded (the memory back), a borrow let go. An open chat stream
        ends with an error event."""
        self._stop.set()
        with self._lock:
            slot, borrowed = self.slot, self.borrowed
            self.slot, self.borrowed, self.refusal, self.error = None, False, None, None
        if slot is not None and not borrowed:
            if not slot.unload() and slot.state == "loading":
                # a load in flight cannot be stopped: its copy goes as soon as it is in
                threading.Thread(target=self._unload_when_loaded, args=(slot, why), daemon=True,
                                 name="wren-unload-after-load").start()
            elif slot.device is not None and getattr(slot.device, "cuda", False):
                import torch
                torch.cuda.empty_cache()
        if slot is not None:
            self.log(f"[wren] {'let go of the narrator' if borrowed else 'unloaded'} ({why})")
        self.stats[f"unload_{why}"] += 1
        return self.status()

    def _unload_when_loaded(self, slot, why: str) -> None:
        t_end = time.monotonic() + LOAD_WAIT_S
        while slot.state == "loading" and time.monotonic() < t_end:
            time.sleep(0.2)
        if slot.state in ("loaded", "failed") and slot.unload():
            if slot.device is not None and getattr(slot.device, "cuda", False):
                import torch
                torch.cuda.empty_cache()
            self.log(f"[wren] unloaded after its load ended ({why})")

    def reclaim(self, want: int = 0) -> int:
        """The KV pool's first role level: Wren's own copy goes before the shadow reader and the
        narrator. Bytes given back (the pool's reserved bytes); 0 for a borrow or a busy chat."""
        with self._lock:
            slot = self.slot
            if slot is None or self.borrowed or slot.state != "loaded" or self._chat.locked():
                return 0
            try:
                got = int(slot.pool_bytes()[0]) or int(slot.device_bytes())
            except Exception:                                      # noqa: BLE001
                got = 0
        self.unload("memory")
        return got

    def reap_once(self) -> bool:
        """Unload when nobody polled or chatted for IDLE_UNLOAD_S (a serving engine only)."""
        if self.setup_mode or self.slot is None or self._chat.locked() or \
                self.clock() - self.last_touch <= IDLE_UNLOAD_S:
            return False
        self.unload("idle")
        return True

    def _start_reaper(self) -> None:
        if self.setup_mode or self._reaper is not None:
            return

        def run():
            while True:
                time.sleep(IDLE_CHECK_S)
                self.reap_once()

        self._reaper = threading.Thread(target=run, name="wren-idle", daemon=True)
        self._reaper.start()

    # ------------------------------------------------------------------ the prompt
    def _encode(self, model, msgs: list[dict], defs: list[dict]) -> list[int]:
        if hasattr(model, "encode_chat"):
            return model.encode_chat(msgs, defs)
        text = model.tok.apply_chat_template(msgs, tools=defs, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        return model.encode_text(text)

    def prompt(self, m: dict, messages: list[dict], page: dict, model,
               use_tools: bool = True) -> tuple[list[int], list]:
        """The ids of this turn's prompt and the sources it quoted; with `use_tools` False the
        prompt has no tool definitions and says the answer is words only."""
        window = self.window()
        max_new = min(MAX_NEW, window // 8)
        small = window < WINDOW
        last = next((x["content"] for x in reversed(messages) if x["role"] == "user"), "")
        step = page.get("step") if isinstance(page, dict) else None
        sources = self.knowledge.search(f"{last} {STEP_WORDS.get(step, step or '')}", k=4,
                                        max_chars=2400 if small else 5200)
        facts = self.device_facts() if self.device_facts is not None else {}
        ref = "\n\n".join(f"[{i + 1}] {s['title']} ({s['path']})\n{s['text']}"
                          for i, s in enumerate(sources))
        here = step_text(m, step)
        # said in words: in JSON the 2B took the page's profile for the one that runs
        runs = (f"It runs the weight profile {facts['active_profile']} now.\n"
                if isinstance(facts, dict) and facts.get("active_profile") else "")
        rules = RULES if use_tools else f"{RULES}\n\n{NO_TOOLS}"
        head = (f"{INTRO}\n\nThe model on this page: {m['name']} ({m['id']}).\n\n"
                f"Reference text:\n{ref or '(nothing matched)'}\n\n"
                f"The box now (read only):\n{runs}"
                f"{json.dumps(facts, separators=(',', ':'))[:800]}\n\n"
                f"The page state (data):\n{page_text(page, 1800 if small else 3500)}\n\n"
                f"{here}\n\n{rules}")
        defs = tools.definitions(m) if use_tools else []
        # the turn being answered, from the user's last message on (its calls and their results
        # with it), is never cut: a template needs the user's message (VIS-91: the page's second
        # round on a 4,096-row borrow was cut to the tool's result alone, and the template threw)
        turn = messages[max(i for i, x in enumerate(messages) if x["role"] == "user"):]
        hist = list(messages)
        renders = 0
        while True:
            ids = self._encode(model, [{"role": "system", "content": head}] + hist, defs)
            renders += 1
            if len(ids) + max_new <= window or len(hist) <= len(turn):
                break
            if renders >= MAX_RENDERS - 1:
                hist = turn                       # enough renders: this turn alone
                continue
            hist = hist[1:]                       # the oldest turn first; the system text never
            while hist[0]["role"] != "user":
                hist = hist[1:]
        if len(ids) + max_new > window:
            # this turn alone is too long for the window: the reference text goes, the page shrinks
            head = (f"{INTRO}\n\nThe page state (data):\n{page_text(page, 600)}\n\n"
                    f"{here}\n\n{rules}")
            sources = []
            ids = self._encode(model, [{"role": "system", "content": head}] + turn, defs)
        return ids, sources

    # ------------------------------------------------------------------ the chat
    def _idle(self) -> bool:
        app = self.app
        with app.QUEUE:
            return not (app.INFLIGHT.get("waiting") or app.INFLIGHT.get("running") or app.WAITERS)

    def _slice(self, job, budget_ms: float) -> dict:
        """One slice of `job` under the engine's lock in Wren's rank; {why: idle_wait} when the
        engine is not idle."""
        app = self.app
        if not self._idle():
            return {"why": "idle_wait"}
        me = {"rank": RANK, "t": time.monotonic(), "side": None}
        if not app._take(me, None):
            return {"why": "idle_wait"}
        try:
            return self.slot.advance(job, budget_ms=budget_ms)
        finally:
            with app.QUEUE:
                app.HOLDER["who"] = None
            app.LOCK.release()

    def _generate(self, ids: list[int], max_new: int, deadline: float):
        """Yield ("text", visible text so far) while the job runs, then ("end", full text,
        finish)."""
        slot = self.slot
        b = slot.begin(ids, max_new=max_new)
        if "why" in b:
            raise RuntimeError(f"Wren is not ready ({b['why']})")
        job, model = b["job"], slot.model
        shown = 0
        while not job.done:
            if self._stop.is_set():
                raise InterruptedError("Wren went to sleep")
            if self.clock() > deadline:
                break
            r = self._slice(job, SLICE_MS)
            if r.get("why") == "idle_wait":
                time.sleep(0.05)
                continue
            if r.get("why"):
                raise RuntimeError(f"Wren stopped ({r['why']}: {r.get('error') or ''})".strip())
            text = model.job_text(job) if hasattr(model, "job_text") else model.decode(job.out)
            vis = text
            k = vis.find(TOOL_OPEN)
            if k >= 0:
                vis = vis[:k]
            else:
                for n in range(len(TOOL_OPEN) - 1, 0, -1):    # a tag that may still be opening
                    if vis.endswith(TOOL_OPEN[:n]):
                        vis = vis[:-n]
                        break
            vis = vis.rstrip("\ufffd").rstrip()     # a space goes out with the word after it
            if len(vis) > shown:
                shown = len(vis)
                yield ("text", vis)
        text = model.job_text(job) if hasattr(model, "job_text") else model.decode(job.out)
        yield ("end", text, job.finish or "time")

    def chat(self, body) -> "collections.abc.Iterator[dict]":
        """The events of one turn (SetupChatEvent): state, text, tool_call, done or error."""
        if not self._chat.acquire(blocking=False):
            yield {"type": "error", "message": "Wren is answering in another tab; try again when "
                                               "that answer is done."}
            return
        try:
            self._stop.clear()
            self.touch()
            yield from self._turn(body)
        finally:
            self._chat.release()
            self.touch()

    def _turn(self, body):
        st = self.state
        if st != "loaded":
            if st == "loading":
                yield {"type": "state", "state": "loading"}
            yield {"type": "error", "message": {
                "loading": "Wren is still loading. Ask again in a moment.",
                "refused": "Wren is not loaded: there is not enough free memory beside the engine."
            }.get(st, "Wren is not loaded. Load it first, or use the page without chat.")}
            return
        if not isinstance(body, dict):
            yield {"type": "error", "message": "the body must be a JSON object"}
            return
        page = body.get("page") if isinstance(body.get("page"), dict) else {}
        try:
            if len(json.dumps(page)) > MAX_PAGE:
                raise ValueError(f"page: at most {MAX_PAGE // 1024} KiB of page state")
            messages = check_messages(body.get("messages"))
        except (ValueError, TypeError) as exc:
            yield {"type": "error", "message": str(exc)}
            return
        if not isinstance(page.get("step"), (str, type(None))):
            page = dict(page, step=None)
        use_tools = body.get("tools", True)
        if not isinstance(use_tools, bool):
            yield {"type": "error", "message": "tools: true or false"}
            return
        models, _ = registry.load_all()
        mid = page.get("model")
        m = (models.get(mid) if isinstance(mid, str) else None) or next(iter(models.values()), None)
        if m is None:
            yield {"type": "error", "message": "no model is registered in models/"}
            return
        self.stats["turns"] += 1
        yield {"type": "state", "state": "thinking"}
        model = self.slot.model
        deadline = self.clock() + TURN_S
        sources_out, finish, talked, emitted = [], "stop", False, 0
        said, repairs, unchanged, changed = "", 0, False, False
        names = list(tools.NAMES)
        from server.toolcall import parse_tool_calls, schema_types
        types = schema_types(tools.definitions(m))
        while True:
            ids, sources = self.prompt(m, messages, page, model, use_tools=use_tools)
            for s in sources:
                if s["path"] not in [x["path"] for x in sources_out]:
                    sources_out.append({"path": s["path"], "title": s["title"]})
            max_new = min(MAX_NEW, self.window() - len(ids))
            if max_new < 16:
                yield {"type": "error", "message": "the conversation is too long for Wren; clear "
                                                   "the chat and ask again."}
                return
            # a second pass's words go after the first's, a paragraph apart
            sep, sent = ("\n\n" if said else ""), 0
            try:
                for ev in self._generate(ids, max_new, deadline):
                    if ev[0] == "text":
                        if not talked:
                            talked = True
                            yield {"type": "state", "state": "talking"}
                        yield {"type": "text", "delta": (sep if not sent else "") + ev[1][sent:]}
                        sent = len(ev[1])
                    else:
                        text, finish = ev[1], ev[2]
            except InterruptedError as exc:
                yield {"type": "error", "message": str(exc)}
                return
            except RuntimeError as exc:
                self.stats["errors"] += 1
                yield {"type": "error", "message": str(exc)}
                return
            content, calls = parse_tool_calls(text, names, eos=finish == "stop", types=types)
            visible = content.split(TOOL_OPEN)[0].rstrip()
            if len(visible) > sent:
                yield {"type": "text", "delta": (sep if not sent else "") + visible[sent:]}
                sent = len(visible)
            if sent:
                said += sep + visible
            if not use_tools:
                # the words-only pass: a call the model still writes goes nowhere
                self.stats["tool_dropped"] += len(calls)
                if not said.strip():
                    said = NOT_CHANGED if unchanged else NO_WORDS
                    yield {"type": "text", "delta": said}
                break
            bad, shown = [], []
            for c in calls:
                fn = c["function"]
                args, why = tools.check(m, fn["name"], fn["arguments"])
                if why is None:
                    self.stats["tool_valid"] += 1
                    emitted += 1
                    shown.append((fn["name"], args))
                    changed = changed or fn["name"] != "explain"
                    yield {"type": "tool_call", "id": c["id"], "name": fn["name"], "args": args}
                else:
                    self.stats["tool_invalid"] += 1
                    bad.append((c, why))
            # a highlight the page cannot make is dropped: it would change nothing
            bad_changes = [(c, why) for c, why in bad if c["function"]["name"] != "explain"]
            if bad_changes and repairs < REPAIRS:
                # the model sees its own invalid calls and why, and tries once more
                repairs += 1
                messages = messages + _said(visible, bad_changes) + [
                    {"role": "tool", "content": f"error: {why}. Nothing changed on the page."}
                    for _, why in bad_changes]
                continue
            if bad_changes:
                unchanged = not changed
                self.stats["forced_words"] += 1
                messages = messages + _said(visible, bad_changes) + [
                    {"role": "tool", "content": f"error: {why}. Nothing changed on the page."}
                    for _, why in bad_changes]
                use_tools = False
                continue
            if not said.strip() and (shown or bad) and all(n == "explain" for n, _ in shown):
                # only a highlight (or none it could make) and no words: the question still wants
                # its answer, in words, with the tools off (a 2B model often points before it
                # speaks, and with the tools on it points again)
                self.stats["explain_then_answer"] += 1
                if shown:
                    messages = messages + [
                        {"role": "assistant", "content": "", "tool_calls": [
                            {"type": "function", "function": {"name": n, "arguments": a}}
                            for n, a in shown]}] + [
                        {"role": "tool", "content": "ok: it is highlighted on the page. Now "
                                                    "answer the user in words."} for _ in shown]
                use_tools = False
                continue
            break
        yield {"type": "done", "finish": "tool_calls" if emitted else finish,
               "sources": sources_out, **({"unchanged": True} if unchanged else {})}


def _said(visible: str, bad: list) -> list[dict]:
    """The assistant message with its invalid calls, as the model wrote them."""
    return [{"role": "assistant", "content": visible, "tool_calls": [
        {"type": "function", "function": {"name": c["function"]["name"], "arguments": _args(c)}}
        for c, _ in bad]}]


def _args(call: dict) -> dict:
    try:
        a = json.loads(call["function"]["arguments"] or "{}")
        return a if isinstance(a, dict) else {}
    except ValueError:
        return {}


def get(app, *, device_facts=None, setup_mode: bool = False) -> Wren:
    """The server's Wren, made on first use (STATE["wren"])."""
    w = app.STATE.get("wren")
    if w is None:
        w = app.STATE["wren"] = Wren(app, device_facts=device_facts, setup_mode=setup_mode,
                                     log=lambda s: print(s, flush=True))
    return w
