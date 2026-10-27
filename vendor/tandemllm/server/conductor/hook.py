"""The conductor as `server/app.py` sees it: one object in `STATE["conductor"]`.

app.py calls it in six places and nowhere else: `/v1/models` (`models()`, `model_card()`), the start of a
completion (`plan()`, before the queue, where an unknown profile is the client's 400), the
in-context stage under the lock when the plan waits for it (`decide_in_context()`), the prompt
under the lock (`Plan.apply_prompt`), the response headers
(`Plan.headers`) and the end of the request (`finish()`). A raw request (`X-Tandem: raw`, or the
locked `raw` / `bench` profile by header) gets a plan with `active = False`, whose every method
returns its input unchanged, so the raw path is the release's path plus two response headers.

One model (the owner, 2026-10-01): `/v1/models` lists `TandemLLM` alone. The raw engine's old name
(`--served-model`, `qwen38-spark-engine` on the box) is a hidden alias: it answers on
`/v1/models/{id}` and in a request, and resolves exactly as `TandemLLM` does.

The frame of the session monitor (SRV-109, `server/conductor/monitor/`) runs under the request's
profile: `monitor_mode()` lowers its mode to the profile's, `monitor_policy()` gives it the
`monitor` document (the policy, versioned and edited like the others) with the profile's task
modes, tool-gate packs, caps and thresholds on top.

One journal for both (SRV-115): the router's rows and the monitor's events go to the same
`decisions.Journal`, which the server opens on the usage ledger's file and hands in here.
"""

from __future__ import annotations

import os
import threading
import time

from server.conductor import schema
from server.conductor.classify import Decider, Qwen, decider_for
from server.conductor.decisions import Journal
from server.conductor.packages import Library
from server.conductor.router import COMPOSITE, Plan, Router, is_alias
from server.conductor.store import DEFAULT_ROOT, Store
from server.conductor.transcripts import Transcripts, memory_root

HEALTH_TTL_S = 10.0


class Conductor:
    def __init__(self, root: str | None = None, *, state: dict | None = None,
                 journal: "bool | Journal" = True, readonly: bool = False, factory=None,
                 monitor_seed: dict | None = None, memory: str | None = None):
        """`journal`: the server's shared journal, or True for one of its own beside the config
        (a standalone conductor: tests, tools), or False for none on disk. `memory`: where the
        session transcripts go (default: beside the production config under the engine's home,
        else inside the config directory)."""
        self.state = state if state is not None else {}
        self.root = os.path.realpath(os.path.expanduser(root or DEFAULT_ROOT))
        self.store = Store(self.root, readonly=readonly, roles_up=self.roles_up,
                           monitor_seed=monitor_seed,
                           engine_tokenizer=lambda: self.state.get("tokenizer_sha"),
                           on_defaults=self._defaults_row)
        self.library = Library()
        _, settings = self.store.get("settings")
        if isinstance(journal, Journal):
            self.journal = journal
        else:
            self.journal = Journal(os.path.join(self.root, "decisions.sqlite3") if journal
                                   else None,
                                   retention_days=settings["journal"]["retention_days"])
        self.router = Router(self.store, self.library, self.journal,
                             raw_model=lambda: str(self.state.get("model") or
                                                   "qwen38-spark-engine"),
                             self_url=self.self_url,
                             server_effort=lambda: self.state.get("reasoning_effort"),
                             global_mode=lambda: self.state.get("conductor_mode"),
                             factory=factory, probe=lambda: self.state.get("probe_stage"),
                             monitor_session=self._monitor_session)
        # SRV-133: a roles document stored before the contract fields loads now, with its
        # defaults journaled, not on the first request that happens to read it
        self.store.get("roles")
        # SRV-137: the session transcripts, written by their own thread after a request ended
        self.transcripts = Transcripts(memory or memory_root(self.root),
                                       settings=lambda: self.store.peek("settings")[1],
                                       journal=self.journal, tok=lambda: self.state.get("tok"))
        # 2026-10-05: the memory notes beside the transcripts (server/conductor/notes.py), and
        # the router's part of them (server/conductor/memory.py); the nightly job runs with the
        # transcripts' prune at 04:00
        from server.conductor.memory import Memory
        from server.conductor.notes import NoteStore
        settings_of = lambda: self.store.peek("settings")[1]                   # noqa: E731
        self.notes = NoteStore(self.transcripts.root,
                               settings=lambda: schema.notes(settings_of()),
                               journal=self.journal)
        self.memory = Memory(self.notes, self.journal, settings=settings_of)
        # P-M3: the packages the model proposes to change, waiting for a click
        from server.conductor.proposals import Proposals
        self.proposals = Proposals(self.transcripts.root, self.store, self.library, self.journal)
        self.memory.proposals = self.proposals
        self.router.memory = self.memory
        self.transcripts.nightly.append(self.notes.nightly)
        if isinstance(journal, Journal):
            self.transcripts.start()            # the server's: the prune at start, then 04:00
        # J1 step 7: Laya's questions on the narrowed cases, after the request, journaled only
        from server.conductor.judge import Judge
        self.judge = Judge(lambda: self.store.peek("roles")[1], self.journal,
                           deciders=lambda: self.store.peek("deciders")[1])
        self._health: dict = {}
        self._health_at = 0.0
        self._policies: dict = {}
        self._hlock = threading.Lock()

    def _monitor_session(self, key: str):
        """The session monitor's state of a session (J1: its acts count against the caps of the
        router's outcome notes), None without a monitor or before its first request."""
        mon = self.state.get("monitor")
        return mon.sessions.get(key) if mon is not None else None

    # ------------------------------------------------------------------ app.py's calls
    def self_url(self) -> str | None:
        port = self.state.get("port")
        return f"http://127.0.0.1:{port}" if port else None

    def models(self, raw_model: str, created: int) -> list[dict]:
        """`/v1/models`: `TandemLLM`, the one model. The raw name is an alias and not listed."""
        _, settings = self.store.get("settings")
        return [{"id": COMPOSITE, "object": "model", "created": created, "owned_by": "tandemllm",
                 "profile": settings["default_profile"],
                 "description": f"{schema.model_name()} with the conductor: per-turn task "
                                "routing, effort and hidden skills, under a profile"}]

    def model_card(self, name: str, raw_model: str, created: int) -> dict | None:
        """`/v1/models/{id}`: the card of `TandemLLM` for its own name and for the hidden alias,
        None (a 404) for any other name."""
        if name.lower() != COMPOSITE.lower() and not is_alias(name, raw_model):
            return None
        return self.models(raw_model, created)[0]

    def plan(self, body: dict, headers, *, client_kind: str, request_id: str,
             chat: bool, count_tokens=None, user: str | None = None) -> Plan:
        return self.router.plan(body, headers, client_kind=client_kind, request_id=request_id,
                                chat=chat, count_tokens=count_tokens, user=user)

    def decide_in_context(self, plan: Plan, prompt_ids: list[int], runner, rows=None) -> None:
        """The in-context stage of a plan that waits for it (under the engine lock); `rows`: the
        probe stage's prefill (ENG-181), when the probe answers first."""
        self.router.decide_in_context(plan, prompt_ids, self.state.get("tok"), runner, rows)

    def finish(self, plan: Plan | None, rec) -> None:
        # the served name is always the one model, refused requests included; `conductor_mode`
        # says whether it ran raw (None: no plan, e.g. a 400 before the profile resolved)
        rec.model_served = COMPOSITE
        if plan is None:
            return
        rec.profile, rec.profile_version = plan.res.profile, plan.res.version
        rec.conductor_mode = plan.res.mode
        if plan.active:
            try:
                if plan.context is not None:
                    self.router.settle_context(plan)    # it ended before the engine lock
                plan.finish(rec, self.journal)
            except Exception as exc:                               # noqa: BLE001
                print(f"[conductor] journal row failed: {type(exc).__name__}: {exc}", flush=True)
            if plan.transcript is not None:
                self.transcripts.submit(plan, rec)          # queued; written off this thread
            if plan.judge:
                self.judge.submit(plan)                     # asked off this thread (J1)

    # ------------------------------------------------------------------ roles
    def _defaults_row(self, kind, name, version: int, applied: list[str]) -> None:
        journal = getattr(self, "journal", None)
        if journal is not None:
            journal.note("config", kind, decision="defaults",
                         detail={"name": name, "version": version, "applied": applied})

    def roles_contract(self) -> dict:
        """role -> what its tokenizer hash allows (SRV-133): the forms of sharing, and whether
        the hash is the engine's (None: the role declares none, or the engine recorded none)."""
        _, doc = self.store.get("roles")
        eng = self.state.get("tokenizer_sha")
        out = {}
        for name, r in doc.items():
            sha = r.get("tokenizer_sha256")
            out[name] = {"sharing_allowed": schema.sharing_forms(r, eng),
                         "tokenizer_match": (sha == eng) if (eng and sha != schema.TOKENIZER_NONE)
                         else None}
        return {"engine_tokenizer_sha256": eng, "roles": out}

    def role_client(self, role: str, doc: dict):
        r = doc.get(role) or {}
        endpoint = r.get("endpoint") or ""
        if endpoint == "self" and r.get("contract") != "jev":
            # Qwen through this server's own port; a `jev` role named `self` is asked in process
            endpoint = self.self_url() or ""
        if not endpoint:
            return None
        if r.get("contract") == "jev":
            return decider_for(endpoint, r.get("model") or "auto", int(r.get("timeout_ms") or 400))
        if r.get("contract") == "openai-chat":
            return Qwen(endpoint, r.get("model") or str(self.state.get("model") or ""),
                        int(r.get("timeout_ms") or 8000))
        return None

    def roles_health(self, force: bool = False) -> dict:
        """role -> {up, ms, mode, error|detail}; checked at most every 10 s, in parallel."""
        now = time.monotonic()
        if not force and self._health and now - self._health_at < HEALTH_TTL_S:
            return self._health
        with self._hlock:
            _, doc = self.store.get("roles")
            out, threads = {}, []

            def one(name, r):
                if r.get("mode", "off") == "off":
                    out[name] = {"up": False, "mode": "off", "ms": None, "error": "off"}
                    return
                if r.get("contract") == "generate":
                    out[name] = self._generative_health(r)
                    return
                c = self.role_client(name, doc)
                if c is None:
                    why = ("no endpoint" if r.get("contract") != "rerank"
                           else "no client for this contract yet")
                    if r.get("endpoint") == "self" and r.get("contract") == "jev":
                        why = "this engine serves no decision model (QWEN38_DECIDER_MODELS is empty)"
                    out[name] = {"up": False, "mode": r.get("mode"), "ms": None, "error": why}
                    return
                h = c.health()
                out[name] = dict(h, mode=r.get("mode"))

            for name, r in doc.items():
                t = threading.Thread(target=one, args=(name, r), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join(3.0)
            self._health, self._health_at = out, time.monotonic()
            return out

    def _generative_health(self, r: dict) -> dict:
        """The narrator's slot (P1): up unless its model failed; it loads on first use."""
        n = self.state.get("narrator")
        if n is None:
            return {"up": False, "mode": r.get("mode"), "ms": None,
                    "error": "this engine serves no narrator (QWEN38_NARRATOR_MODEL is empty)"}
        st = n.slot.status()
        out = {"up": st["state"] != "failed", "mode": r.get("mode"), "ms": st["ms_p50"],
               "detail": f"{st['name']}: {st['state']}"}
        if st["error"]:
            out["error"] = st["error"]
        return out

    def roles_up(self) -> dict:
        return {k: bool(v.get("up")) for k, v in self.roles_health().items()}

    def role_test(self, role: str) -> dict:
        """One question to one role: its latency and its answer, or why it failed."""
        _, doc = self.store.get("roles")
        if role not in doc:
            return {"role": role, "ok": False, "error": "no such role"}
        if doc[role].get("contract") == "generate":
            n = self.state.get("narrator")
            if n is None:
                return {"role": role, "ok": False, "error": "this engine serves no narrator"}
            ev = {"kind": "route", "facts": {"cls": "code_edit", "effort": "high effort"}}
            r = n.try_events([ev])[0]
            return {"role": role, "ok": r.get("line") is not None, "ms": r.get("ms"),
                    "question": r["template"], "answer": r,
                    **({"error": r["why"]} if r.get("why") else {})}
        c = self.role_client(role, doc)
        if c is None:
            return {"role": role, "ok": False, "error": "no endpoint or no client for its "
                                                         "contract"}
        _, router = self.store.get("router")
        opts = {c_["id"]: c_["description"] for c_ in router["classes"]}
        text = "Rename the variable `cnt` to `count` in utils.py and run the tests."
        if isinstance(c, Decider):
            a = c.choice(text, router["classifier"]["question"], opts)
        else:
            listing = "\n".join(f"- {k}: {v}" for k, v in opts.items())
            a = c.choice(text, router["classifier"]["qwen_prompt"].replace("{classes}", listing),
                         opts)
        out = {"role": role, "ok": a.error is None, "ms": a.ms, "question": text,
               "answer": a.as_dict()}
        if a.error:
            out["error"] = a.error
        with self._hlock:
            self._health_at = 0.0                            # the next health read is fresh
        return out

    # ------------------------------------------------------------------ the monitor's hand-over
    # profile threshold -> the monitor policy's parameter path, for the tasks the frame runs
    THRESHOLDS = {
        ("D1", "repeat_share"): ("D1", "arms", "repeat", "act"),
        ("D1", "grey_low"): ("D1", "arms", "repeat", "grey"),
        ("D1", "min_tokens"): ("D1", "arms", "repeat", "min_tokens"),
        ("D1", "every_tokens"): ("D1", "arms", "repeat", "every"),
        ("D1b", "min_span"): ("D1", "arms", "loop", "min_span"),
        ("D2", "backstop"): ("D2", "cap"),
        ("D2", "answer_room"): ("D2", "reserve"),
        ("E1", "tries"): ("E1", "tries"),
        ("E2", "claim_at"): ("E2", "claim_at"),
        ("E2", "admit_at"): ("E2", "admit_at"),
        ("C2", "max_file_lines"): ("C2", "max_lines"),
        ("C3", "repeats"): ("C3", "repeats"),
        ("C4", "repairs"): ("C4", "regenerations"),
        ("C5", "regenerations"): ("C5", "regenerations"),
    }
    # the tool-gate pack each tool-call task belongs to: a pack that is off turns its task off
    PACKS = {"C1": "risk", "C2": "project", "C3": "same_call", "C4": "malformed",
             "C5": "sequence"}
    # the profile's context section -> the session tasks' parameters (SRV-121). The section is
    # where the Profiles screen sets them; the task rows' S1/S4 thresholds mirror it
    CONTEXT = {"s1_runs": ("S1", "fails"), "s4_minutes": ("S4", "minutes"),
               "notify": ("S4", "notify", "enabled")}

    def monitor_mode(self, plan: Plan | None, server_mode: str | None) -> str | None:
        """The session monitor's mode for a request: the server's (`--conductor`, lowered by
        `X-Tandem`), and with a profile resolved never above the profile's: the raw model name
        runs the monitor raw too (SRV-117)."""
        if plan is None:
            return server_mode
        from server.conductor.monitor.policy import lower
        return lower(server_mode or "raw", plan.res.mode)

    def monitor_policy(self, plan: Plan | None, base):
        """The monitor's policy under this request's profile: the `monitor` document (version 0
        is the server's policy file, `base`) with the profile's task modes and tool-gate packs,
        caps, the thresholds that map onto the rules' parameters and its context section (S1's
        fails, S4's minutes and NOTIFY). Tasks the profile does not know keep the document's
        mode. Cached per profile and document version."""
        if plan is None or not plan.active or base is None:
            return None
        mv, mdoc = self.store.peek("monitor")
        key = (plan.res.profile, plan.res.version, mv, id(base))
        got = self._policies.get(key)
        if got is not None:
            return got
        import copy
        from server.conductor.monitor.policy import Policy
        data = copy.deepcopy(mdoc if mv else base.data)
        doc = plan.res.doc
        for tid, task in data["tasks"].items():
            if tid in doc["tasks"]:
                task["mode"] = doc["tasks"][tid]["mode"]
            pack = self.PACKS.get(tid)
            if pack and not doc["toolgate"].get(pack, False):
                task["mode"] = "off"
        for (tid, name), path in self.THRESHOLDS.items():
            val = doc["tasks"].get(tid, {}).get("thresholds", {}).get(name)
            node = data["tasks"].get(path[0])
            if val is None or node is None:
                continue
            for k in path[1:-1]:
                node = node.get(k) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, dict) and path[-1] in node:
                old = node[path[-1]]
                node[path[-1]] = int(val) if isinstance(old, int) and float(val).is_integer() \
                    else val
        for name, path in self.CONTEXT.items():
            node = data["tasks"].get(path[0])
            for k in path[1:-1]:
                node = node.get(k) if isinstance(node, dict) else None
            if isinstance(node, dict) and path[-1] in node and name in doc["context"]:
                node[path[-1]] = doc["context"][name]
        data["caps"]["acts_per_request"] = doc["caps"]["acts_per_request"]
        data["caps"]["acts_per_session_hour"] = doc["caps"]["acts_per_session_hour"]
        pol = Policy(data, base.path)
        if len(self._policies) > 32:
            self._policies.clear()
        self._policies[key] = pol
        return pol

    def close(self) -> None:
        self.transcripts.flush()
        self.notes.flush_usage()
        self.journal.close()


def locked_names() -> list[str]:
    return sorted(schema.LOCKED)
