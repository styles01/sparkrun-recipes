"""The conductor's configuration: what each document holds, its defaults, and its validation.

Seven kinds of document, every one versioned by `server/conductor/store.py`:

    settings   the default profile and the journal's scope and retention
    router     the task classes (id, description, default policy), the classifier stages and
               their thresholds, stickiness, the effort levels and the injection wrapper
    roles      role -> endpoint, model, timeout, memory claim, mode (a role is a kind of
               question; only this file names models)
    packages   skills and packages the router may inject: inline text or a SKILL.md on the box
    monitor    the session monitor's policy: every task with its rule and parameters (the tool
               checks' floors, project rules and reminders among them), caps, texts, templates.
               Version 0 is the server's policy file (ops/supervisor.yaml); its schema is the
               monitor's own, built from the rule registry (server/conductor/monitor/policy.py)
    profiles   one document per profile: modes, client mapping, per-class policy overrides and
               the session monitor's task settings (mode, thresholds, the tool-gate packs)
    residency  the residency planner's limits (OPS-41): the MemAvailable floor and the margin a
               role load must leave above it, the settle time after a stop, the load order, and
               each role's memory claim with its basis (measured, a declared peak, declared)
    deciders   the decision models the engine serves itself (ENG-179, SRV-131): each model's load
               policy (on_start, on_demand, off) and which model answers each question id

The schemas are the JSON Schema 2020-12 subset `validate` below understands (and `ajv` in the
dashboard's contract test): `type`, `properties`, `required`, `additionalProperties` (false or a
schema), `items`, `enum`, `const`, `minimum`, `maximum`, `minLength`, `maxLength`, `maxItems`,
`pattern`. `tools/conductor_contract.py --write` puts them in `docs/contract/conductor-v1/`, and a
test holds the files to these dicts. Standard library only, like the rest of the server.

`check(kind, doc, ...)` is the schema plus the rules a schema cannot say: a policy names classes
and packages that exist, a task set to act whose role is offline is refused, an injection wrapper
carries `{text}`.
"""

from __future__ import annotations

import copy
import json
import os
import re

from server.conductor.packages import literal_api_hosts as _literal_api_hosts
from server.conductor.packages import shipped_root as _shipped_root
from server.conductor.states import CHANNELS as STATE_CHANNELS, KINDS as STATE_KINDS
from server.conductor.voice import DEFAULTS as VOICE_DEFAULTS

MODES3 = ("off", "shadow", "on")            # a routing decision type
# SRV-475: how a profile's agent turns sample when the client names no sampling
AGENTIC_SAMPLING = ("greedy", "card")
TASK_MODES = ("off", "shadow", "act")       # a session-monitor task (SRV-109 wording)
PROFILE_MODES = ("raw", "shadow", "on")     # the profile's master switch
EFFORTS = ("off", "low", "medium", "high", "xhigh")
SIZES = ("small", "medium", "large")          # SRV-157: a task's size, from structural signals
SIZE_SIGNALS = ("session_tokens", "files", "checks", "turns")
# a profile without `thinking.by_size` (every version stored before SRV-157) journals the size and
# the effort it would pick, and changes nothing
BY_SIZE_DEFAULT = "shadow"
# F2 (2026-10-02): the effort of a turn whose newest message is a tool result (SRV-161's tool media
# included). `inherit` is the class's, as before; a level sets that turn's thinking switch and cap,
# never its note. The router's `structural.tool_turn_effort`, a profile's `thinking.tool_turn_effort`
# over it; a document without either is `inherit`
TOOL_TURN_EFFORTS = ("inherit",) + EFFORTS
TOOL_TURN_EFFORT_DEFAULT = "inherit"
# a client's effort word under a profile that decides the effort (the owner, 2026-10-03 12:30):
# `ceiling` (absent = ceiling) the router may go lower, never higher; `ignore` the router alone
CLIENT_WORDS = ("ceiling", "ignore")
# The classes whose answers are code; under the conductor a client's
# `no_repeat_ngram_size` is dropped on their turns and on any turn that offers tools (a repeated
# 4-gram is ordinary code: indentation, `", "`, a sprite's pixel row), the server's own default
# stays (server/app.py)
CODE_CLASSES = ("code_edit", "debug", "design_ui", "drawing", "three_d")
# SRV-189 (the UT2 self-test, 2026-10-06): "Write a detailed 700-word essay about the history of
# tides" went to code_edit at 0.5708 (chat 0.29, writing 0.10 behind it) in an Open WebUI agent
# chat: the coding package, xhigh, 8.3 s of prefill. A code class below `accept` for a request that
# names nothing of code (`cues`) and names a piece of prose (`prose`) is not taken: the stage's best
# class outside CODE_CLASSES is, and the stage row says `guard`. Measured on the router's labelled
# sets plus prose_prompts.jsonl (live Qwen stage, 374 prompts): 2 prose prompts no longer routed
# as code, 0 changes on the other 342 (the FX2 note in Memo).
CODE_GUARD = {
    "accept": 0.7,
    "cues": (r"```|(?i:\b[\w-]+\.(?:py|js|jsx|ts|tsx|html?|css|scss|json|ya?ml|toml|md|sh|go|"
             r"rs|rb|java|kt|swift|c|cc|cpp|h|hpp|cs|php|sql|vue|svelte|ipynb|csv|txt|env|lock|"
             r"xml)\b)|"
             r"(?i:\b(?:code|coding|program\w*|script\w*|function\w*|method|class(?:es)?|bug\w*|"
             r"errors?|exception|traceback|stack ?trace|crash\w*|compil\w*|tests?|testing|"
             r"repo\w*|git|commit|branch|merge|html|css|javascript|typescript|python|java|rust|"
             r"golang|sql|api|endpoint|json|yaml|regex|cli|terminal|bash|shell|npm|pip|docker\w*|"
             r"kubernetes|react|vue|svelte|node(?:\.js)?|app|apps|website|web ?page|webpage|page|"
             r"game|ui|ux|button|server|database|schema|refactor\w*|implement\w*|deploy\w*|"
             r"debug\w*|lint\w*|install\w*|file|files|folder|directory|path|variable|module|"
             r"package|library|framework|component|widget|canvas|svg|three\.js|webgl|shader|"
             r"sprite|frontend|backend|query|migration|config\w*|log|logs)\b)|"
             r"\b[a-z]+_[a-z0-9_]+\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b|\w+\(\)|</?[a-z]+>"),
    # ... and that names a piece of prose (an essay, a letter, "700 words"): without this second
    # condition the guard sent six design, drawing and 3D requests of the labelled set to chat
    "prose": (r"(?i)\b(?:essay|story|stories|poem|sonnet|haiku|letter|e-?mail|article|blog|post|"
              r"speech|toast|eulogy|memoir|novel|chapter|newsletter|press release|cover letter|"
              r"note|description|opinion piece|op-ed|\d+[- ]words?)\b"),
}


# SRV-191 (the owner, 2026-10-06): asked who or what it is, the model said "I'm Qwen". It names the
# served model and says that it runs in TandemLLM. A hidden note at the usual place (the end of the
# last message, replayed there on every later turn of the session: router.Router._identity), on
# the turn whose message asks (IDENTITY_ASK), under every profile that runs the conductor; raw and
# bench never see it, and a session that never asks carries nothing new. QWEN38_MODEL_NAME
# overrides the name for a box that serves another checkpoint.
# 0.8.2 (the owner, 2026-10-13): asked "Who are you?" in Open WebUI, the second model said "I'm
# Fable, an AI made by Anthropic" and "I'm Claude": the RC served it raw (no note), its prompt named
# only tools and skills (one named after another assistant), and the model answers from its
# training then. The note says it is TandemLLM running the served model on the user's own machine,
# and that it is no other assistant and not made by Anthropic or OpenAI.
MODEL_NAME = "Qwen3.8-27B"
#: the served model's name as the server found it (its model card's label for the weight profile,
#: server/app.py `_identity_name`), the second model's label on its packs. None: MODEL_NAME
SERVED_NAME: str | None = None
IDENTITY_NOTE = ("About you: you are TandemLLM, running the {model} model on the user's own "
                 "machine. You are not Claude, Fable, ChatGPT or any other assistant, whatever "
                 "names the tools or skills mention, and you were not made by Anthropic or "
                 "OpenAI. When someone asks who or what you are or which model this is, say that "
                 "you are TandemLLM running {model} on their own machine, in one short sentence in "
                 "their language, and add details about the engine only when asked.")


# the turn that asks (the note goes in there, then replays at its place on every later turn of the
# session): who or what it is, which model, its name, who made it, in English, German, French,
# Spanish, Italian and Dutch; "are you ChatGPT/Claude/an AI"
IDENTITY_ASK = (
    r"(?i)\b(?:who|what)\s+(?:are|r)\s+(?:you|u)\b(?!\s+(?:doing|working|up\s+to|talking|saying|"
    r"thinking|going|trying|planning|looking))|\bwhich\s+(?:model|llm|ai)\b|"
    r"\bwhat\s+(?:model|llm|ai)\b|\byour\s+(?:name|model)\b|\babout\s+yourself\b|"
    r"\bwho\s+(?:made|built|created|trained|developed)\s+you\b|"
    r"\bare\s+you\s+(?:chatgpt|gpt|claude|gemini|qwen|llama|an?\s+(?:ai|bot|llm|language model))\b|"
    r"\bwer\s+bist\s+du\b|\bwas\s+bist\s+du\b|\bwelches\s+(?:modell|sprachmodell|llm|ki)\b|"
    r"\bwie\s+hei(?:ss|ß)t\s+du\b|\bwer\s+hat\s+dich\b|\bwer\s+sind\s+sie\b|"
    r"\bqui\s+es[- ]tu\b|\bquel\s+mod[eè]le\b|\bqui[eé]n\s+eres\b|\bqu[eé]\s+modelo\b|"
    r"\bchi\s+sei\b|\bwie\s+ben\s+je\b")


def model_name() -> str:
    from engine.settings import SETTINGS as _S
    return (_S.get("MODEL_NAME") or "").strip() or SERVED_NAME or MODEL_NAME


def identity_text() -> str:
    return IDENTITY_NOTE.replace("{model}", model_name())

DECISIONS = ("classify", "effort", "packages", "rules")
CLIENT_KINDS = ("open-webui", "openai-sdk", "opencode", "hermes", "curl", "dashboard", "chat-app",
                "other")
LOCKED = {"raw": "raw passthrough, byte-identical to the release",
          "bench": "raw passthrough, byte-identical; benchmarks and the paper"}
ID_RE = r"^[a-z][a-z0-9_-]{0,39}$"

# --------------------------------------------------------------------------- the task catalogue
# The session monitor's tasks (TandemLLM 1, section 3). The frame (SRV-109) executes them; the
# profile only says their mode, thresholds and cap. `unit` and `why` are for the dashboard's
# editor; `role` is the role a Tier 1 task needs before it may act.
TASKS: dict[str, dict] = {
    "D1": {"label": "reasoning loop", "family": "thinking", "tier": "0+1", "role": "decider",
           "thresholds": {"repeat_share": (0.7, "share"), "grey_low": (0.4, "share"),
                          "min_tokens": (2048, "tokens"), "every_tokens": (64, "tokens")},
           "why": "3 of 3 failures caught at 2.1k to 3.0k tokens; 4 of 80 long healthy turns "
                  "closed at 0.6 to 0.7"},
    "D1b": {"label": "no close on literal runs", "family": "thinking", "tier": "0", "role": None,
            "thresholds": {"min_span": (128, "tokens")},
            "why": "51 of 52 stall closes in the database were hex or digit runs"},
    "D2": {"label": "reasoning backstop", "family": "thinking", "tier": "0", "role": None,
           "thresholds": {"backstop": (32768, "tokens"), "answer_room": (2048, "tokens")},
           "why": "one turn reasoned 65,536 tokens and wrote nothing (2026-09-30 19:05Z)"},
    "D3": {"label": "already decided", "family": "thinking", "tier": "1", "role": "decider",
           "thresholds": {"act_at": (0.9, "probability"), "every_tokens": (1024, "tokens")},
           "why": "hypothesis: long reasoning after the next call is chosen; shadow first"},
    "D4": {"label": "thinking per step", "family": "thinking", "tier": "0+1", "role": None,
           "thresholds": {}, "why": "the thinking controller (SRV-127)"},
    "E1": {"label": "empty stop with tools", "family": "end of turn", "tier": "0", "role": None,
           "thresholds": {"tries": (2, "count")},
           "why": "two empty stops ended the 2026-09-30 night (19:33Z, 19:58Z)"},
    # the claim check (server/conductor/monitor/claimcheck.py): Qwen itself answers, in place
    # under the request's own engine lock, so it needs no role. `claim_at`: Q1's yes counts as a
    # claim; `admit_at`: Q2's answer is taken
    "E2": {"label": "claim check", "family": "end of turn", "tier": "1",
           "role": None, "thresholds": {"claim_at": (0.85, "probability"),
                                        "admit_at": (0.6, "probability")},
           "why": "three Mario sessions (2026-10-02) ended 'Done, verified' after a faked test "
                  "(a key flag set in code, Mario teleported to the flag); shadow first"},
    "C1": {"label": "risk guard", "family": "tool calls", "tier": "0+1", "role": "decider",
           "thresholds": {"act_at": (0.85, "probability")},
           "why": "guard floors plus the decider in the grey band"},
    "C2": {"label": "project rules", "family": "tool calls", "tier": "0", "role": None,
           "thresholds": {"max_file_lines": (400, "lines")},
           "why": "AGENTS.md rules the model broke on 2026-09-30: numbered copies, /tmp paths"},
    "C3": {"label": "same call thrice", "family": "tool calls", "tier": "0", "role": None,
           "thresholds": {"repeats": (3, "count")},
           "why": "one todowrite loop ran 49 times in 10 minutes (2026-09-26, SRV-45)"},
    "C4": {"label": "malformed call", "family": "tool calls", "tier": "0", "role": None,
           "thresholds": {"repairs": (2, "count")},
           "why": "opencode refused 19 calls on 2026-09-26; a cut write reached it as broken JSON"},
    "C5": {"label": "one mutation per turn", "family": "tool calls", "tier": "0", "role": None,
           "thresholds": {"regenerations": (1, "count")},
           "why": "tool-eval-bench TC-51 (2026-10-18): the event and its invitation in one turn; "
                  "the email said the invite was sent before the event existed"},
    "S1": {"label": "gate progress", "family": "session", "tier": "0", "role": None,
           "thresholds": {"fails": (3, "count")},
           "why": "gate (a) failed 15.6, 12.2, 3.5, 12.3, 3.5 without progress (2026-09-30)"},
    "S2": {"label": "context health", "family": "session", "tier": "0", "role": None,
           "thresholds": {}, "why": "night 1: PROGRESS.yaml last read at 02:28, five "
                                    "compactions after it"},
    "S3": {"label": "skill routing", "family": "session", "tier": "0+1", "role": None,
           "thresholds": {"act_at": (0.9, "probability")}, "why": "SRV-123"},
    "S4": {"label": "stall watchdog", "family": "session", "tier": "0", "role": None,
           "thresholds": {"minutes": (20, "minutes")},
           "why": "the run sat idle 21.2 minutes after an empty stop (2026-09-30); the "
                  "overnight driver's job"},
    "HINT": {"label": "speed hints", "family": "speed", "tier": "0", "role": None,
             "thresholds": {}, "why": "not gated: no SPD measurement yet"},
}


# tasks a stored profile may lack (added after profiles were stored): absent, the task keeps the
# monitor document's mode, and its tool-gate pack, absent too, keeps it off (hook.monitor_policy)
OPTIONAL_TASKS = ("C5",)


def _tasks(act=(), shadow=()) -> dict:
    out = {}
    for tid, meta in TASKS.items():
        mode = "act" if tid in act else "shadow" if tid in shadow else "off"
        out[tid] = {"mode": mode, "cap": 1 if tid.startswith(("E", "C")) else 2,
                    "thresholds": {k: v[0] for k, v in meta["thresholds"].items()}}
    return out


# --------------------------------------------------------------------------- seed documents
# What a box starts with before anyone writes a version (version 0, built in). The texts are
# the defaults the dashboard shows grey.

XHIGH_NOTE = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate "
              "key assumptions, consider plausible alternatives, and prioritize correctness, "
              "consistency, and clarity in the final answer.")
LOW_NOTE = ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly "
            "to the conclusion without unnecessary elaboration.")
HIGH_NOTE = ("Reasoning effort for this step is high. Think the task through and check your plan "
             "against the code before you act.")

# The in-context stage's question: a hidden user turn after the conversation, answered with one
# letter (a single token each in Qwen's vocabulary; the router has at most 20 classes), so one
# prefill's last row is the answer's whole distribution. `{classes}` is one `A: id - description`
# line per class
CONTEXT_QUESTION = ("The runtime that serves you asks one question before you answer, and the user "
                    "does not see it. What kind of work does the user's latest message in this "
                    "conversation ask for? Answer with the letter of the one class that fits it "
                    "best, and nothing else.\n\nClasses:\n{classes}")
CONTEXT_STAGE: dict = {"mode": "off", "min_tokens": 8192, "followups": True, "accept": 0.55,
                       "question": CONTEXT_QUESTION}
# ENG-181: a linear probe on Qwen's own rows (server/probe.py, trained by tools/probe_train.py).
# `file`: a probe under ~/.qwen38-spark-engine/probe/probes/ (or a full path). shadow: the answer
# of the turn's own snapshot is journaled beside the router's (extra.probe); on: where the plan
# decides under the engine lock (the in-context stage), the probe answers first and the question
# is asked only below `accept`. Off by default
PROBE_STAGE: dict = {"mode": "off", "file": "", "accept": 0.9}
# FX6 (2026-10-06, the pagoda turn: three_d at 0.464, the package question answered `none` at
# 0.747 and no 3D skill went in): the Qwen stage of the package question gets the class question's
# prompt ("You sort requests ... into task classes") with the packages as the classes, and not the
# class the turn was routed to. A router's `classifier.package_prompt` replaces it: `{class}` is
# the routed class (`id: label, description`), `{question}` the classifier's `package_question`,
# `{classes}` the options. None by default: three prompts that name the class (this one among
# them, 2026-10-06, n = 1 on 51 prompts whose pick it changed) lost 8 to 27 test-skill picks on
# code_edit and debug (test-api at 0.98 became none) and still left three of four unsure 3D
# prompts under the 0.6 accept. The pagoda turn is fixed by the class's `core` instead
PACKAGE_PROMPT_CLASS = ("A request to a coding assistant was sorted into this task class: "
                        "{class}\n{question} The instructions below are extra recipes for parts "
                        "of that class of work. Answer with the id of the one the request needs, "
                        "or none when it needs none of them.\n\nInstructions:\n{classes}")
# FX6: the reading line. A class the classifier picked on a fresh turn below `below` (a class
# switch needs 0.8, router.stickiness), with another class that has a `reading` at `other` or
# more, gets one sentence in its note: build what the class says, and open the answer with one
# short line that names the reading, never a blocking question. `{reading}` and `{other}` are the
# two classes' `reading` phrases. Only classes with a `reading` take part (the deliverable kinds:
# a web page, a 2D drawing, a 3D scene). Under the profile's `classify` decision
READING_TEXT = ("Build it as {reading}. The request could also mean {other}, so start your "
                "answer with one short line that says how you read it, for example: \"Building "
                "it as {reading}. Say if you wanted {other}.\" Then go straight on; do not stop "
                "to ask first.")
READING: dict = {"mode": "on", "below": 0.8, "other": 0.15, "text": READING_TEXT}
# SRV-243 (UX2, 2026-10-07): when a class's packages need tools the request lacks, the turn's note
# says what the chat can do instead (router.py `Router._fit`). `{tools}` is " (its tools: a, b)"
# or nothing. `note_none`: no shell and no file write; `note_write`: a file write, no shell
TOOL_FIT: dict = {
    "mode": "on",
    "note_none": ("This chat has no shell and no file tools{tools}. Call only the tools listed "
                  "above, and none of them to write or run code. Deliver the work in your answer: "
                  "each file complete in one fenced code block, its file name on the line above "
                  "it. You cannot run it here, so read it once for mistakes before you answer, and "
                  "end with one line on how to open or run it."),
    "note_write": ("This chat can write files but has no shell{tools}. Write the files with the "
                   "file tool; do not try to run or test them, and end with one line on how to "
                   "open or run them."),
    # SRV-615: a package whose result an offered tool delivers (write-message where the chat
    # offers send_email) stays out when the user wants it done, not written: `on` acts, `shadow`
    # journals what would have stayed out, `off` asks nothing. The question goes to the class
    # question's stages (the decider, then Qwen) with the user's message. Shipped in shadow: on
    # the chat profile (2026-10-20, n = 1) TC-03 passed with it on, the score stayed 91 and
    # TC-47 and TC-51 failed, cause not tested
    "deliverable": "shadow",
    "deliverable_question": ("The chat offers these tools: {tools}. Does the user want the task "
                             "done with one of these tools, or do they want the text itself to "
                             "read, copy or send themselves?"),
    "deliverable_options": {
        "tool": "Done with a tool: the user wants it sent, booked, set or posted.",
        "text": "The text itself: the user wants a draft, to read, copy or send themselves."},
    "deliverable_accept": 0.6,
}
# S1 (2026-10-02): a client's side request (opencode's title or summary call, Hermes' title call)
# is detected structurally (signals.side_shape: no tools and the client's fixed instruction) and
# served without the classifier, a package or a note, thinking off unless the client asked for a
# number, its output capped by the shape. `structural.side_requests` off serves it as a turn, as
# before; a router stored before it gets `on`. `structural.side_shapes` replaces the built-in list
SIDE_REQUESTS_DEFAULT = "on"
SIDE_KINDS = ("title", "summary")
# the longest side-request prompt that goes ahead of a waiting turn: the resident prefix's guest
# size, so it also never replaces the session's rows; a longer one waits for the queue to empty
SIDE_FIRST_MAX_TOKENS = 8192

# J1 (2026-10-02, the plan "Laya as the test judge, state snapshots, turn outcomes and the
# client protocol"): the turn snapshot and the Tier 0 outcome of every tool-result turn
# (server/conductor/outcomes.py). A profile's `outcomes` section switches each part; absent, every
# part is off and the plan is the release's. `snapshots` writes the snapshot on the NVMe;
# `mode` journals the outcome (shadow) or acts on it (act: the effort and the note); `cards` keeps
# the client tool card and the harness card live in the session's notes (shadow journals them);
# `evaluate` the evaluation points E-b and E-c; `judge` asks Laya beside Tier 0 (shadow only);
# `retrieve` the package for a low-quality or stuck outcome
OUTCOME_LABELS = ("no_visible_change", "regression", "error", "low_quality", "surprise", "stuck",
                  "tool_misuse", "test_faked", "environment", "lost_context")
PROFILE_OUTCOMES: dict = {"snapshots": "off", "mode": "off", "cards": "off", "evaluate": "off",
                          "judge": "off", "retrieve": "off"}
OUTCOME_SWITCHES = {"snapshots": ("off", "on"), "mode": ("off", "shadow", "act"),
                    "cards": MODES3, "evaluate": ("off", "shadow", "act"),
                    "judge": ("off", "shadow"), "retrieve": ("off", "shadow", "on")}
# the router document's `outcomes` section over these (a router stored before J1 has none). The
# patterns read the shape of a tool result (an exit code, a test runner's counts, a console
# level, an OS error), never what anybody meant: the project's rule against keyword intent routing
OUTCOMES: dict = {
    # one outcome a request: the first label whose facts matched, in the plan's table order;
    # `progress` only when nothing matched, `none` when there was nothing to read
    "order": list(OUTCOME_LABELS),
    "write_tools": ["write", "edit", "patch", "multiedit", "apply_patch"],
    "run_tools": "(?i)(^bash$|shell|exec|run|test|browser|devtools|playwright|puppeteer|navigate|"
                 "new_page|screenshot|snapshot|evaluate|press_key|click|fill|type_text|terminal)",
    "evaluate_tools": "(?i)evaluate",
    "console_tools": "(?i)console",
    # the tools whose result is the program's own output (an input tool's or a screenshot's text
    # is the same acknowledgement every time): the same output after a write is no change
    "output_tools": "(?i)(^bash$|shell|exec|terminal|evaluate|test|^run)",
    "patterns": {
        "exit": [r'"exit_?code"\s*:\s*(-?\d+)',
                 r"(?im)^\s*(?:exit code|exit status|exited with code|process exited with code)"
                 r"\s*[:=]?\s*(-?\d+)"],
        "error_line": (r"(?m)^(?:Traceback \(most recent call last\)|\s*[A-Za-z_.]*(?:Error|"
                       r"Exception)(?: \[[^\]\n]*\])?: |Uncaught |panic: |FAIL[: \t]|npm ERR!|"
                       r"error(?:\[E\d+\])?: )"),
        "environment": (r"(?i)(connection refused|ECONNREFUSED|address already in use|EADDRINUSE|"
                        r"no such file or directory|ENOENT|permission denied|EACCES|"
                        r"command not found|ERR_CONNECTION_REFUSED|ERR_FILE_NOT_FOUND)"),
        "rejected": (r"(?i)(invalid arguments?|invalid input|unknown tool|is not a valid tool|"
                     r"tool [^\n]{0,60} (?:was )?not found|failed to parse (?:the )?"
                     r"(?:tool|arguments))"),
        "summary": r"(?i)\b(\d+)\s+(passed|passing|failed|failing|errors?)\b",
        "console_level": (r"(?im)^\s*(?:msgid=\S+\s+)?\[(error|warn|warning|info|log|debug|"
                          r"verbose|issue)\]"),
        # a log tail the model printed: the session's scratch directory (SRV-171: `{tandem_scratch}`
        # is /tmp/tandem-<token>), or `.tandem/` inside a project from before 0.4.0
        "log_tail": r"\btail\b[^\n]*(?:\.tandem/|/tmp/tandem[\w-]*/)",
    },
    "limits": {"calls": 8, "result_chars": 1200, "files": 20, "shots_kept": 5,
               "shot_bytes": 1048576, "console_lines": 10, "state_bytes": 4096, "log_lines": 60,
               "snapshot_bytes": 32768, "digest_chars": 2100, "scan_chars": 20000},
    "low_quality": {"far": 0.5, "fails": 3},
    "stuck": {"turns": 15, "minutes": 20},
    "lost_context": {"files": ["PROGRESS.yaml", "AGENTS.md"],
                     "read_tools": ["read", "bash", "grep"], "steps": 2},
    # a state dump's numbers that are a surprise on their own (the plan's "lives 0")
    "surprise": {"state": [{"key": "lives", "op": "<=", "value": 0}]},
    # act: the effort of the next turn, never below the class's level, for `raise_turns` tool
    # turns or until a `progress`
    "effort": {"no_visible_change": "xhigh", "regression": "xhigh", "error": "high"},
    "raise_turns": 2,
    # act: the outcomes whose note goes in (the plan's step 3); the rest stay journaled
    "acting": ["no_visible_change", "regression", "error"],
    "notes": {
        "no_visible_change": ("Your last change had no visible effect: {why}. Before the next "
                              "edit, say what you expected, what the evidence shows and the one "
                              "cause. Then act.\n\n{digest}"),
        "regression": ("The last run is worse than an earlier one: {why}. Compare it with the "
                       "last good state, then revert or diagnose the change that made it worse "
                       "before the next edit.\n\n{digest}"),
        "error": ("The last run shows an error: {why}. Debug it: reproduce it with the smallest "
                  "input, find the one cause, change one thing, run the same reproduction "
                  "again.\n\n{digest}"),
        "low_quality": ("The result is far from its goal: {why}. Read the instructions below and "
                        "change the approach before the next run.\n\n{package}"),
        "surprise": ("Explain this before going on: {why}. A result nobody would expect is a bug "
                     "until the code explains it.\n\n{digest}"),
        "stuck": ("No file changed and no number got better for {since}. Re-plan in 3 lines from "
                  "the evidence below, then act.\n\n{digest}"),
        "tool_misuse": "A tool call went wrong: {why}.\n\n{card}",
        "test_faked": ("Your test set state in a script ({why}), so it tested nothing the user "
                       "does. Test it through input only: real key events or clicks, wait real "
                       "frames, then read the state."),
        "environment": ("The environment stopped the run: {why}. Start a server in the "
                        "background (`mkdir -p .tandem; cmd > .tandem/run.log 2>&1 &`), check the path, use "
                        "absolute paths."),
        "lost_context": ("You are after a summary. Re-read PROGRESS.yaml, AGENTS.md and the last "
                         "checker lines before the next change."),
    },
    # the client tool card (B1) and the harness card (B2): the coding classes they go with, the
    # pitfalls package per client kind and the harness card's package
    "cards": {"classes": ["code_edit", "debug", "design_ui", "three_d", "drawing"],
              "pitfalls": {"opencode": "tools-opencode", "hermes": "tools-hermes",
                           "openai-sdk": "tools-hermes", "other": "tools-hermes"},
              "harness": "tandem-harness", "max_tools": 40, "max_args": 6},
    # Laya beside Tier 0 (step 7): the questions of 1.3 it is asked, after the request, in the
    # idle window, each with this deadline; journaled, never acting
    "judge": {"deadline_ms": 100, "digest_chars": 2000,
              "questions": {
                  "failure": {"question": "Does this tool output show a failure?",
                              "options": {"failure": "a traceback, a failed test, an error "
                                                     "line or a crash",
                                          "clean": "it ran and shows no failure",
                                          "unclear": "it cannot be told from this output"}},
                  "build": {"question": "Is this log tail a successful build or run?",
                            "options": {"ok": "it finished and succeeded",
                                        "failed": "it failed",
                                        "incomplete": "it has not finished or is cut off"}}}},
    # step 6: the evaluation points E-b (the claim check's note carries the digest) and E-c
    # (stuck and low quality: re-plan from the evidence) at `effort`; `ask`: Qwen in place on a
    # narrowed case before the turn's prefill (the in-context stage's forward), shadow only, off
    # by default: it costs a prefill of the prompt's tail on every narrowed turn
    "evaluate": {"effort": "xhigh",
                 "ask": {"mode": "off", "accept": 0.6,
                         "question": ("The runtime that serves you asks one question before you "
                                      "go on, and the user does not see it. Look at the newest "
                                      "tool result in this conversation. Which is true? Answer "
                                      "with the letter only."),
                         "options": {"failure": "it shows a failure: an error, a crash, a "
                                                "failed test or a wrong result",
                                     "clean": "it ran and shows no failure",
                                     "unclear": "it cannot be told from this result"}}},
    # step 8: the packages a low-quality or stuck outcome may retrieve (the kind skills), asked
    # with the router's package question
    "retrieve": {"packages": ["test-web", "test-cli", "test-api", "test-data", "test-lib"],
                 "question": "Which of these instructions fits the work in this conversation "
                             "best?"},
}

# R1 (2026-10-03, the owner: "multiple packages about many things and subjects so the model can search
# and use them later"): the package library, the 44 SKILL.md files under
# ~/.qwen38-spark-engine/conductor/skills/library/, registered as packages with `library: true`
# (server/conductor/libsearch.py). A profile's `library` switch: off (absent, every profile stored
# before R1), shadow (the pick is journaled, nothing goes in) or on. On a new user message, and on
# the outcomes listed here, a BM25 search over the message's digest (the class's fit from INDEX.md
# as a filter) hands its top `candidates` to the package question (packages.fit: the decider or
# Qwen picks a description, or none), up to `max_picks` skills; each goes in as a note at the
# prompt's tail, replayed at its place. `card` is the package that tells the model the library
# exists and how to search it (kept live like the harness card on the `card_classes` with tools).
# The router document's `library` section over these (absent: these)
LIBRARY: dict = {
    "candidates": 4, "max_picks": 1, "min_score": 0.0, "class_filter": True,
    "index": "~/.qwen38-spark-engine/conductor/skills/library/INDEX.md",
    "digest_chars": 2000,
    "question": "Which of these skills fits the work this request asks for best?",
    "accept": 0.6,
    "outcomes": ["low_quality", "stuck"],
    "card": "tandem-library",
    "card_classes": ["code_edit", "debug", "design_ui", "three_d", "drawing", "plan"],
    "header": ("A skill from the TandemLLM library fits this task: {name}. Follow its rules and "
               "its \"Done when\". Its reference (snippets, commands): GET /v1/library/{name}."),
}

# The packages every class that builds something carries (J1, R1, X1; the live router v13 of
# 2026-10-03): the coding classes get the coding package, the test skills and Laya's; the visual
# classes the testing and Laya packages before their own skills
# SRV-227: every class that writes files carries `files-care` (never overwrite the
# user's files: a new folder gets a free name, a file is read before it changes), a general package
# OPS-39 (the first packages, 2026-10-07): `tandem-core` (the runtime's own conventions, about
# 300 tokens, a general package with no tool needs) goes in on every class but `summarize`;
# `render-tools` (how to use SRV-105's render tool server, general, `requires_tools: [render]`)
# on every class that builds something, so the tool fit leaves it out until a chat offers the tool
CORE_PACKAGES = ["tandem-core"]
# PKG3 (2026-10-07): tandem-core on the chat class waits for a design decision (+~290 tokens a
# plain chat turn); the one-line switch: CHAT_CORE = ["tandem-core"]. On a box, the router
# document's chat class lists it or not (live, no restart)
CHAT_CORE: list[str] = []
BUILD_PACKAGES = ["tandem-core", "files-care", "testing", "laya-testing", "laya-playtest",
                  "render-tools"]
CODING_PACKAGES = ("tandem-core", "coding", "files-care", "testing", "laya-testing", "laya-playtest",
                   "laya-scripts", "test-web", "test-cli", "test-api", "test-data", "test-lib",
                   "render-tools")

# PKG2 (2026-10-07, the owner: "create many others so they will help in normal life usage of
# AI"): the everyday library, packages/everyday/<name>/SKILL.md, on the classes a plain chat lands
# on. `everyday-core` is general on each of them (the user's language, plain words, the whole
# result in the answer, one question at most); the rest are the class's candidates for the package
# question, so a message that fits none of them gets none. None needs a tool: the answer carries
# the result. The chooser's list stays under eleven a class; the debug class carries the one
# everyday package a non-coder's "it does not work" may land on
EVERYDAY: dict[str, list[str]] = {
    "writing": ["everyday-core", "write-message", "formal-letter", "cv-cover-letter",
                "rewrite-text", "proofread", "translate", "social-post", "creative-writing",
                "difficult-conversation"],
    "plan": ["everyday-core", "trip-plan", "weekly-schedule", "meal-plan", "budget-plan",
             "event-plan", "checklist-builder", "study-plan", "decision-helper", "fitness-sleep",
             "flashcards-quiz"],
    "research": ["everyday-core", "explain-simply", "compare-options", "fact-check",
                 "understand-document", "homework-help", "tech-troubleshoot", "security-scam",
                 "health-info", "recipe-cooking", "home-howto"],
    "summarize": ["everyday-core", "summarize-text", "understand-document", "flashcards-quiz",
                  "rewrite-text"],
    "chat": ["everyday-core", "explain-simply", "language-practice", "gift-ideas",
             "parenting-help", "health-info", "security-scam", "translate", "recipe-cooking",
             "difficult-conversation", "reply-format"],
    "debug": ["tech-troubleshoot"],
}
# PKG3 (2026-10-08, measured): router v19 sent 106 of the 210 everyday eval prompts to chat (the
# plan class reads as software planning: 28 plans went to chat; research 49, writing 24), where
# the chat list offered 9. The package question over chat's 9 picked the prompt's package 38 times
# of 106 (none 51, wrong 17); over all 33 everyday packages 94 (none 6, wrong 6). Three class-
# description rewrites routed better but moved the pagoda prompt to code_edit (SRV-255): the lists
# widen instead. Every prose class offers the whole everyday set; the home lists above stay as the
# packages' own classes (manifests, INDEX.md). `reply-format` (last, as measured) takes an
# instruction about the reply itself ("Reply with just: OK" picked write-message live): about 267
# tokens instead of a whole package
# PKG3's night A/B (2026-10-08, n=3-4 a package, judged at T 0 by the served model, plus a hand check):
# these five gained at most +0.3 of 5 over no package (translate 5.0 both ways, creative-writing 4.93
# against 5.0) for 1.1-1.5k tokens and 1-2 s of TTFT, so they ship off (enabled: false). They stay
# options of the package question: a request that picks one gets nothing, not a wrong package
EVERYDAY_OFF = ("creative-writing", "language-practice", "translate", "social-post", "summarize-text")
EVERYDAY_ALL: list[str] = list(dict.fromkeys(
    p for k in ("writing", "plan", "research", "summarize", "chat") for p in EVERYDAY[k]
    if p != "everyday-core"))

SHIPPED_ROUTER: dict = {
    "classes": [
        {"id": "code_edit", "label": "Code edit",
         "description": "Write, change, refactor or extend code or files in a software project.",
         # SRV-157: eval round 1 (2026-10-01), minicalc took 7.4 against 6.5 minutes at xhigh
         # FX6: the coding package is the coding classes' core (the shipped packages document
         # has no `general` flag on it, so a fresh box offered it to the package question)
         "policy": {"effort": "xhigh", "packages": list(CODING_PACKAGES), "core": ["coding"],
                    "effort_by_size": {"small": "high", "medium": "xhigh", "large": "xhigh"},
                    "rules": [
             "Read a file before you change it, and run the project's check or tests after each "
             "change."]}},
        {"id": "debug", "label": "Debug",
         "description": "Find and fix a bug, a failing test, an error message or a crash.",
         "policy": {"effort": "xhigh", "packages": list(CODING_PACKAGES) + EVERYDAY["debug"],
                    "core": ["coding"], "rules": [
             "Reproduce the failure first, change one thing at a time, and quote the result line "
             "of the check."]}},
        {"id": "design_ui", "label": "UI design",
         "description": "The look of a web page or an app: layout, visual style, colours, "
                        "typography, UI and UX.",
         "reading": "a flat web page",
         # SRV-534 follow-up: a core, so a model fit with `packages: core` (Super VL) gets a
         # package on a design_ui turn (Laya read a three.js prompt as design_ui, 2026-10-17)
         "policy": {"effort": "high", "packages": BUILD_PACKAGES + ["ui-design", "charts"],
                    "core": ["ui-design"], "rules": []}},
        {"id": "drawing", "label": "Drawing",
         "description": "Draw, paint or animate a picture or a portrait with code: a sketch, "
                        "line art, SVG or canvas art.",
         "reading": "a 2D drawing",
         # OPS-39: creative-coding (one-file visuals with sound), charts (an SVG chart from
         # data) and explainer-video (a video through the render_video tool, else its script
         # fallback) are choices of the package question beside the portrait recipes
         "policy": {"effort": "high",
                    "packages": BUILD_PACKAGES + ["pencil-sketch", "spiral-portrait",
                                                  "one-line-portrait", "creative-coding",
                                                  "charts", "explainer-video"],
                    "core": ["creative-coding"], "rules": []}},
        {"id": "three_d", "label": "3D scene",
         "description": "A 3D scene or model: Three.js, WebGL, voxels, a photo rebuilt in 3D.",
         "reading": "a 3D three.js scene",
         # FX6: the 3D skill is the class's core, in on every three_d turn; the package question
         # picks among the special recipes alone
         # OPS-39: voxel-worlds (the voxel build order, curated modules) and creative-coding
         # join the special recipes the package question picks among
         # 2026-10-17 (pagoda replays 1-3 on Super VL): three first steps thought 6 to 11
         # minutes and drafted the files in the thinking; coding's "plan in a few lines, code
         # in files" goes in beside the 3D skill
         "policy": {"effort": "high",
                    "packages": BUILD_PACKAGES + ["threejs-scene", "photo-to-3d", "voxel-seasons",
                                                  "voxel-worlds", "creative-coding", "coding"],
                    "core": ["threejs-scene", "coding"], "rules": []}},
        # 2026-10-03 10:12 (a user's essay with "use max effort" fell back to code_edit and its
        # coding packages): prose has a class of its own, medium, no packages
        {"id": "writing", "label": "Writing",
         "description": "Write prose in words, not code: an essay, an article, a story, an email, "
                        "a letter, a post or documentation text.",
         "policy": {"effort": "medium", "packages": CORE_PACKAGES + ["everyday-core"]
                    + EVERYDAY_ALL,
                    "rules": [], "package_sticky": "class"}},
        {"id": "plan", "label": "Plan",
         "description": "Plan an approach, a design or an architecture before building it, or "
                        "break the work into steps.",
         "policy": {"effort": "high", "packages": CORE_PACKAGES + ["everyday-core"]
                    + EVERYDAY_ALL,
                    "rules": [], "package_sticky": "class"}},
        {"id": "research", "label": "Research",
         "description": "Look something up, compare options, or explain how something works.",
         "policy": {"effort": "medium", "packages": CORE_PACKAGES + ["everyday-core"]
                    + EVERYDAY_ALL,
                    "rules": [], "package_sticky": "class"}},
        {"id": "summarize", "label": "Summarize",
         "description": "Summarize, shorten or pull the key points out of text that is given, or "
                        "compress a conversation.",
         "policy": {"effort": "off", "packages": ["everyday-core"] + EVERYDAY_ALL, "rules": [], "package_sticky": "class"}},
        {"id": "chat", "label": "Chat",
         # SRV-235: an instruction about the reply itself ("Reply with just: OK") is chat; on 308
         # labelled prompts and 20 short opencode messages the class question went 286 -> 297 of 328
         "description": ("Small talk, a greeting, thanks, a short question with a short answer, or an "
                         "instruction about the reply itself (answer in one word, say OK, confirm, stop) "
                         "that asks for no work: no code, files, commands or git."),
         # medium since router v12 (2026-10-03): a question that fell to chat at Qwen's 0.47 was
         # an essay
         "policy": {"effort": "medium", "packages": CHAT_CORE + ["everyday-core"] + EVERYDAY_ALL,
                    "rules": [], "package_sticky": "class"}},
    ],
    "classifier": {
        "stages": [
            # 0.8, not 0.6: on the labelled set (216 prompts, the Spark, 2026-10-01) the cascade
            # was right on 95.4 % at 0.8 against 93.5 % at 0.6; half the decider's confident
            # mistakes sat between 0.6 and 0.8 and Qwen got them right (tools/conductor_eval.py)
            # Qwen decides (router v7, 2026-10-02: 0.949 right on the labelled set against Laya's
            # 0.782 at 541 ms a call); the decider stage stays for a box that turns it on.
            # Accept 0.45 (router v11, 2026-10-03): below it the profile's default class
            {"kind": "decider", "enabled": False, "accept": 0.8, "timeout_ms": 400},
            {"kind": "qwen", "enabled": True, "accept": 0.45, "timeout_ms": 8000},
        ],
        "question": "What kind of work does this request ask for?",
        "digest_chars": 2000,
        # SRV-153: one line of what the router knows (previous class, last tools, file types,
        # checker status, compaction) at the digest's end, out of the tail's share. Off: with
        # Laya alone it failed the ship rule on 2026-10-01 (box CPU, 216 short prompts 0.500 ->
        # 0.324 at 0.8; follow-ups that change the task right 10 -> 1 of 12, the line pulls it to
        # the previous class); the Qwen stage is not measured yet
        "context_line": False,
        "package_question": "Which of these instructions fits this request best?",
        "package_accept": 0.6,
        # a first message under the accept bar keeps its top class when that class builds
        # something and leads the second by 0.1 (Router._below_build). On 341 labelled prompts
        # (router v17): 307 -> 309 right, 0 answers newly wrong; the pagoda request (three_d 0.378,
        # design_ui 0.256) and the aquarium (three_d 0.446) no longer fall back to chat
        "below_accept": {"mode": "build", "margin": 0.1},
        "qwen_prompt": ("You sort requests to a coding assistant into task classes. Read the "
                        "request and answer with the id of the one class that fits it best.\n\n"
                        "Classes:\n{classes}"),
        "reading": dict(READING),
        # Qwen decides inside the conversation it already holds (2026-10-02): a long prompt or a
        # follow-up is classified by the engine itself on the prompt as the client sent it (the
        # resident prefix and its anchors hit, only the tail and this question are prefilled),
        # thinking off, one letter. Laya keeps the short first messages. Off by default
        "context_stage": dict(CONTEXT_STAGE),
        "probe_stage": dict(PROBE_STAGE),
    },
    "structural": {"compaction": "summarize", "tool_turn": "keep",
                   "side_requests": SIDE_REQUESTS_DEFAULT},
    "stickiness": {"switch": 0.8},
    "effort": {
        "levels": {
            "off": {"thinking": False, "budget": 0, "note": ""},
            "low": {"thinking": True, "budget": 1024, "note": LOW_NOTE},
            # real caps (2026-10-02): before them medium, high and xhigh differed by their note
            # alone. The cap closes the reasoning with the monitor's hidden close (SP1); 0 is no
            # cap below the backstop (D2). A profile's `thinking.budgets` overrides a level
            "medium": {"thinking": True, "budget": 8192, "note": ""},
            "high": {"thinking": True, "budget": 16384, "note": HIGH_NOTE},
            "xhigh": {"thinking": True, "budget": 0, "note": XHIGH_NOTE},
        },
        "raise_on_fail": True,
        # SRV-157: the size signals and their thresholds (a value below `small` is small, up to
        # `medium` medium, above it large). First guesses from eval round 1 (2026-10-01):
        # minicalc and csvq added 25k to 49k prompt tokens over 7 to 16 turns, spiral 54 to 90
        # turns, photo3d 144 to 192
        "size": {
            "signals": {
                "session_tokens": {"enabled": True, "small": 20000, "medium": 60000},
                "files": {"enabled": True, "small": 3, "medium": 8},
                "checks": {"enabled": True, "small": 4, "medium": 8},
                "turns": {"enabled": True, "small": 30, "medium": 100},
            },
            "reset_on_compaction": True,
        },
    },
    "injection": {
        "wrapper": ("\n\n<runtime-note>\nThe runtime that serves you adds this note. The user did "
                    "not write it and does not see it. Follow it and do not mention it.\n\n{text}\n"
                    "</runtime-note>"),
        "max_note_tokens": 8000,
        "max_replay_tokens": 16000,
        "replay": True,
    },
}

# The packages a new box starts with: the live packages document of 2026-10-03 (v5),
# shipped with the repository. `packages/seed.json` is the document; an inline package's body is
# the file its `body_from` names under `packages/` (`inline/<name>.md`), a skill's SKILL.md sits in
# `packages/skills/` or `packages/library/`, which `skill_dirs` reaches through `{repo}` after the
# box's own directories (a skill of the same name on the box wins). A text that tells the model to
# call TandemLLM names it `{tandem_url}` (server/conductor/packages.py). Five skills of the drawing
# and 3D classes (pencil-sketch, spiral-portrait, one-line-portrait, photo-to-3d, voxel-seasons)
# are not shipped: their checkers are the operator's own tools; on a box without them the router
# skips them as missing files, as before
SHIPPED = os.path.join(_shipped_root(), "packages")


def _shipped_packages() -> dict:
    with open(os.path.join(SHIPPED, "seed.json"), encoding="utf-8") as f:
        doc = json.load(f)
    for p in doc["packages"].values():
        src = p.pop("body_from", None)
        if src is not None:
            with open(os.path.join(SHIPPED, src), encoding="utf-8") as f:
                p["body"] = f.read()
    return doc


SHIPPED_PACKAGES: dict = _shipped_packages()

# The built-in documents (version 0) stay the minimal ones the engine always had: what a store with
# nothing in it reads, and what the tests build on. A new box gets the shipped documents above as
# its version 1 at its first start (server/conductor/shipped.py)
ROUTER: dict = {
    "classes": [
        {"id": "code_edit", "label": "Code edit",
         "description": "Write, change, refactor or extend code or files in a software project.",
         # SRV-157: eval round 1 (2026-10-01), minicalc took 7.4 against 6.5 minutes at xhigh
         "policy": {"effort": "xhigh", "packages": [],
                    "effort_by_size": {"small": "high", "medium": "xhigh", "large": "xhigh"},
                    "rules": [
             "Read a file before you change it, and run the project's check or tests after each "
             "change."]}},
        {"id": "debug", "label": "Debug",
         "description": "Find and fix a bug, a failing test, an error message or a crash.",
         "policy": {"effort": "xhigh", "packages": [], "rules": [
             "Reproduce the failure first, change one thing at a time, and quote the result line "
             "of the check."]}},
        {"id": "design_ui", "label": "UI design",
         "description": "The look of a web page or an app: layout, visual style, colours, "
                        "typography, UI and UX.",
         "policy": {"effort": "high", "packages": ["ui-design"], "rules": []}},
        {"id": "drawing", "label": "Drawing",
         "description": "Draw, paint or animate a picture or a portrait with code: a sketch, "
                        "line art, SVG or canvas art.",
         "policy": {"effort": "high",
                    "packages": ["pencil-sketch", "spiral-portrait", "one-line-portrait"],
                    "rules": []}},
        {"id": "three_d", "label": "3D scene",
         "description": "A 3D scene or model: Three.js, WebGL, voxels, a photo rebuilt in 3D.",
         "policy": {"effort": "high", "packages": ["photo-to-3d", "voxel-seasons"], "rules": []}},
        {"id": "plan", "label": "Plan",
         "description": "Plan an approach, a design or an architecture before building it, or "
                        "break the work into steps.",
         "policy": {"effort": "high", "packages": [], "rules": []}},
        {"id": "research", "label": "Research",
         "description": "Look something up, compare options, or explain how something works.",
         "policy": {"effort": "medium", "packages": [], "rules": []}},
        {"id": "summarize", "label": "Summarize",
         "description": "Summarize, shorten or pull the key points out of text that is given, or "
                        "compress a conversation.",
         "policy": {"effort": "off", "packages": [], "rules": []}},
        {"id": "chat", "label": "Chat",
         "description": "Small talk, a greeting, or a short question with a short answer.",
         "policy": {"effort": "low", "packages": [], "rules": []}},
    ],
    "classifier": {
        "stages": [
            # 0.8, not 0.6: on the labelled set (216 prompts, the Spark, 2026-10-01) the cascade
            # was right on 95.4 % at 0.8 against 93.5 % at 0.6; half the decider's confident
            # mistakes sat between 0.6 and 0.8 and Qwen got them right (tools/conductor_eval.py)
            {"kind": "decider", "enabled": True, "accept": 0.8, "timeout_ms": 400},
            {"kind": "qwen", "enabled": True, "accept": 0.55, "timeout_ms": 8000},
        ],
        "question": "What kind of work does this request ask for?",
        "digest_chars": 2000,
        # SRV-153: one line of what the router knows (previous class, last tools, file types,
        # checker status, compaction) at the digest's end, out of the tail's share. Off: with
        # Laya alone it failed the ship rule on 2026-10-01 (box CPU, 216 short prompts 0.500 ->
        # 0.324 at 0.8; follow-ups that change the task right 10 -> 1 of 12, the line pulls it to
        # the previous class); the Qwen stage is not measured yet
        "context_line": False,
        "package_question": "Which of these instructions fits this request best?",
        "package_accept": 0.6,
        "qwen_prompt": ("You sort requests to a coding assistant into task classes. Read the "
                        "request and answer with the id of the one class that fits it best.\n\n"
                        "Classes:\n{classes}"),
        # Qwen decides inside the conversation it already holds (2026-10-02): a long prompt or a
        # follow-up is classified by the engine itself on the prompt as the client sent it (the
        # resident prefix and its anchors hit, only the tail and this question are prefilled),
        # thinking off, one letter. Laya keeps the short first messages. Off by default
        "context_stage": dict(CONTEXT_STAGE),
        "probe_stage": dict(PROBE_STAGE),
    },
    "structural": {"compaction": "summarize", "tool_turn": "keep",
                   "side_requests": SIDE_REQUESTS_DEFAULT},
    "stickiness": {"switch": 0.8},
    "effort": {
        "levels": {
            "off": {"thinking": False, "budget": 0, "note": ""},
            "low": {"thinking": True, "budget": 1024, "note": LOW_NOTE},
            # real caps (2026-10-02): before them medium, high and xhigh differed by their note
            # alone. The cap closes the reasoning with the monitor's hidden close (SP1); 0 is no
            # cap below the backstop (D2). A profile's `thinking.budgets` overrides a level
            "medium": {"thinking": True, "budget": 8192, "note": ""},
            "high": {"thinking": True, "budget": 16384, "note": HIGH_NOTE},
            "xhigh": {"thinking": True, "budget": 0, "note": XHIGH_NOTE},
        },
        "raise_on_fail": True,
        # SRV-157: the size signals and their thresholds (a value below `small` is small, up to
        # `medium` medium, above it large). First guesses from eval round 1 (2026-10-01):
        # minicalc and csvq added 25k to 49k prompt tokens over 7 to 16 turns, spiral 54 to 90
        # turns, photo3d 144 to 192
        "size": {
            "signals": {
                "session_tokens": {"enabled": True, "small": 20000, "medium": 60000},
                "files": {"enabled": True, "small": 3, "medium": 8},
                "checks": {"enabled": True, "small": 4, "medium": 8},
                "turns": {"enabled": True, "small": 30, "medium": 100},
            },
            "reset_on_compaction": True,
        },
    },
    "injection": {
        "wrapper": ("\n\n<runtime-note>\nThe runtime that serves you adds this note. The user did "
                    "not write it and does not see it. Follow it and do not mention it.\n\n{text}\n"
                    "</runtime-note>"),
        "max_note_tokens": 8000,
        "max_replay_tokens": 16000,
        "replay": True,
    },
}

UI_DESIGN = """# UI design

Build interfaces that look designed, not assembled.

1. Start from the content. Decide what the user must see first on the page, then second. One primary action per screen; make it the strongest element.
2. Use a spacing scale (4, 8, 12, 16, 24, 32, 48 px) and nothing between. Align everything to it. Group related items with less space, separate groups with more.
3. Type: one family for UI text, at most one for display. A scale such as 12, 14, 16, 20, 24, 32 px. Body text 14 to 16 px, line height 1.4 to 1.6, lines at most 75 characters.
4. Colour: define tokens first (background, surface, border, text, muted text, one accent, success, warning, danger). Text contrast at least 4.5:1. Use the accent for the primary action and the current state only.
5. Layout: CSS grid or flexbox, no absolute positioning for layout. Mobile first: one column at 390 px with no horizontal scroll, then widen at breakpoints. Tap targets at least 44 px.
6. States: every component needs hover, focus-visible, active, disabled, loading, empty and error states. Never leave a blank area while data loads.
7. Motion: 150 to 250 ms, ease-out, only to explain a change. Respect prefers-reduced-motion.
8. Accessibility: semantic HTML first, labels for every input, visible focus, keyboard reachable controls, alt text.
9. Check your work: open the page at 390 px and at 1440 px, compare the result with the brief, and fix alignment and spacing before adding features.
"""

PACKAGES: dict = {
    "skill_dirs": ["~/.config/opencode/skills", "~/.qwen38-spark-engine/conductor/skills"],
    "packages": {
        "ui-design": {"title": "UI design", "kind": "skill", "source": "inline",
                      "description": "Rules for layout, spacing, type, colour, states and "
                                     "responsive checks when building a web or app interface.",
                      "body": UI_DESIGN, "max_tokens": 3000, "enabled": True},
        "pencil-sketch": {"title": "Pencil sketch", "kind": "skill", "source": "skill:pencil-sketch",
                          "description": "Turn a reference image into an animated graphite pencil "
                                         "sketch on a canvas: tone, contours, hatching.",
                          "body": "", "max_tokens": 4000, "enabled": True},
        "spiral-portrait": {"title": "Spiral portrait", "kind": "skill",
                            "source": "skill:spiral-portrait",
                            "description": "A portrait drawn as one spiral line whose width follows "
                                           "the tone of a reference image.",
                            "body": "", "max_tokens": 4000, "enabled": True},
        "one-line-portrait": {"title": "One-line portrait", "kind": "skill",
                              "source": "skill:one-line-portrait",
                              "description": "A portrait drawn as one continuous line through a "
                                             "stipple of the image (TSP art).",
                              "body": "", "max_tokens": 4000, "enabled": True},
        "photo-to-3d": {"title": "Photo to 3D", "kind": "skill", "source": "skill:photo-to-3d",
                        "description": "Rebuild one photo of a real place as a Three.js scene "
                                       "that matches the photo from a hero camera.",
                        "body": "", "max_tokens": 4000, "enabled": True},
        "voxel-seasons": {"title": "Voxel seasons", "kind": "skill", "source": "skill:voxel-seasons",
                          "description": "Give a voxel island four seasons: palette, snow, "
                                         "particles, weather, sound, timelapse camera.",
                          "body": "", "max_tokens": 4000, "enabled": True},
    },
}

# SRV-133, the role contract (Harmony 1): what every role declares beside its endpoint. `kind` is
# what a task asks for, never a model; `window` when it may use the Spark's GPU (idle windows, in
# decode beside Qwen, or any: not on the GPU); `preemptible` whether a Qwen request cuts it;
# `latency_budget_ms` how long its caller waits before `fallback` applies; `writes` what it may
# write (only the reasoner writes tokens the client sees, the honesty rule); `tokenizer_sha256`
# the tokenizer it reads (the engine records its own, ENG-129), which decides the forms of sharing
# it may declare (Harmony 2): `text` always, `tokens` (ids, digest chunks, the stores) only with
# the engine's own hash, `states` (Qwen's hidden rows) only for a probe inside the engine.
# P1 (2026-10-02): a `generative` role is a small model inside the engine that writes a few tokens
# of text (the narrator); its window `between_rounds` is the request's own thread between two of
# Qwen's rounds, Qwen waiting, for at most its latency budget; it writes `state_lines` (the
# framed lines of states.py in the reasoning, never content, never Qwen's prompt); its contract
# `generate` is the in-process call (engine/generative/slot.py).
# a profile's `ui.narrator`: off, on (P1: every state line), thoughts (N2: each thought's summary)
# v3 (2026-10-04): `live`, the voice: live lines while Qwen thinks, writes and calls tools
# (server/conductor/voice.py), the harness's lines in plain words
NARRATOR_MODES = ("off", "on", "thoughts", "live")
ROLE_KINDS = ("reasoner", "decider", "probe", "clerk", "librarian", "tool", "generative")
WINDOWS = ("idle", "in_decode", "any", "between_rounds")
FALLBACKS = ("PASS", "stale", "skip", "next-role")
WRITES = ("client_visible", "hidden_notes", "memory_files", "nothing", "state_lines")
SHARING = ("text", "tokens", "states")
CONTRACTS = ("jev", "openai-chat", "rerank", "generate")
# the longest a role may hold Qwen between two rounds, per call
BETWEEN_ROUNDS_MAX_MS = 1000
TOKENIZER_NONE = "none"
# what a roles document stored before SRV-133 gets for the fields it lacks (journaled at load)
ROLE_DEFAULTS = {"window": "idle", "preemptible": True, "fallback": "PASS", "writes": "nothing",
                 "tokenizer_sha256": TOKENIZER_NONE, "sharing": ["text"]}
# the kind of a role the documents before SRV-133 named, else the kind its contract implies
ROLE_KIND_OF = {"decider": "decider", "classifier": "reasoner", "judge": "reasoner",
                "librarian": "librarian", "reader": "clerk", "narrator": "generative",
                "tester": "decider"}
CONTRACT_KIND = {"jev": "decider", "rerank": "librarian", "openai-chat": "clerk",
                 "generate": "generative"}


def _role(endpoint, model, contract, timeout_ms, mode, kind, window, preemptible=True,
          writes="nothing") -> dict:
    return {"endpoint": endpoint, "model": model, "contract": contract, "timeout_ms": timeout_ms,
            "memory_claim_gib": 0.0, "mode": mode, "kind": kind, "window": window,
            "preemptible": preemptible, "latency_budget_ms": timeout_ms, "fallback": "PASS",
            "tokenizer_sha256": TOKENIZER_NONE, "writes": writes, "sharing": ["text"]}


# `self` for a `jev` role: the decision models this engine serves (engine/decision, asked in
# process, SRV-131); the arbiter sidecar on :8010 is gone (OPS-45 re-scoped, 2026-10-01). Its
# memory claim is the residency document's. The classifier and the judge are Qwen itself through
# this server's port: their requests take the engine lock like any other, so their window is any.
ROLES: dict = {
    "decider": _role("self", "auto", "jev", 400, "on", "decider", "idle"),
    "classifier": _role("self", "qwen38-spark-engine", "openai-chat", 8000, "on", "reasoner",
                        "any", preemptible=False),
    "librarian": _role("", "", "rerank", 800, "off", "librarian", "any"),
    "reader": _role("", "", "openai-chat", 30000, "off", "clerk", "idle"),
    "judge": _role("self", "qwen38-spark-engine", "openai-chat", 30000, "off", "reasoner",
                   "any", preemptible=False),
    # P1: a small model inside the engine writes the state lines (server/conductor/narrator.py),
    # since N2 (2026-10-03) Qwen3.5-0.8B summing up each thought; a profile's `ui.narrator`
    # switches it on, this role's mode and budget (one slice between two rounds) say how.
    # 0.11.3 (SRV-534): off by default; it does not load, the keeper leaves it and the residency
    # plan leaves out its claim; `on` here and a profile's `ui.narrator` bring it back
    "narrator": _role("self", "qwen3.5-0.8b", "generate", 150, "off", "generative",
                      "between_rounds", writes="state_lines"),
    # L1 (2026-10-03): the test judge's Laya (server/conductor/judge.py), a role of its own, so
    # the judge answers whatever the router's decider stage does (routing is Qwen's since router
    # v7, and the J1 judge asked through the `decider` role); a profile's `outcomes.judge`
    # switches the questions, this role's mode says whether Laya answers them
    "tester": _role("self", "auto", "jev", 100, "on", "decider", "idle"),
}

# OPS-41: what may load beside the engine. A role loads when MemAvailable minus its claim stays at
# or above floor + margin (15 + 6 GiB: the 2026-09-11 wedge and the decider design note), one at a
# time in `order`, never during a hold. A claim is the MemAvailable the role took beside the
# engine, not what nvidia-smi shows; a role that grows (arbiter's batcher at 64 rows) claims its
# peak. The decider's 8.2 GiB was the arbiter sidecar's (31.0 -> 22.8 GiB, 2026-10-01); Laya inside
# the engine claims 3.0 (OPS-69, the reference box's residency v2 since 2026-10-06). With 8.2 a new
# box refused Laya on Super VL at 3 slots: 43.75 GiB available less the engine's 17.46 GiB growth
# to its peak leaves 26.29, less 8.2 is 18.09 under 21; less 3.0 is 23.29 (2026-10-17)
RESIDENCY: dict = {
    "floor_gib": 15.0, "margin_gib": 6.0, "settle_s": 5,
    "order": ["decider", "narrator", "librarian", "reader"],
    "claims": {
        "decider": {"gib": 3.0, "basis": "measured",
                    "note": "Laya's three checkpoints in the engine's process, measured beside a "
                            "side slot (OPS-69, 2026-10-06); the arbiter sidecar took 8.2"},
        "narrator": {"gib": 6.0, "basis": "declared",
                     "note": "the narrator's model in BF16: Qwen3.5-0.8B (2.5 GiB declared) or "
                             "2B (5.5), its 4,096-row cache, the step's workspace and graph, the "
                             "pool's slack (MiniCPM5-2B, P1, 6); loads only when a profile with "
                             "ui.narrator on or thoughts asks for a line"},
        "reader": {"gib": 12.0, "basis": "peak",
                   "note": "a Nemotron Nano 9B in FP8, declared peak 10 to 12 GB; not resident "
                           "until RND-21 earns it"},
    },
}
BASES = ("measured", "peak", "declared")

# The decision models the engine serves (stage A of the decision-model design): which load when
# the engine starts (`on_start`, after /health and the page-cache drop, through the residency
# planner), which load when first asked (`on_demand`), which never (`off`); and which model
# answers each question id the conductor asks (`auto`: the decider role's model, by default the
# family's router). English is the only
# one on start: every conductor question is English. A model `QWEN38_DECIDER_MODELS` does not
# serve is ignored here.
DECIDER_LOADS = ("on_start", "on_demand", "off")
# L1 (2026-10-03, the owner: "teach the model that it can reach Laya via our provider"): Laya for the
# scripts the model writes, `POST /v1/laya/ask` (server/laya_api.py), open like the chat API. A
# deciders document stored before it has no `api` and gets these: on, a rate in texts a minute
# over every caller, the texts one call may carry, the longest text taken (Laya reads its window
# of it, the head and the tail), the default deadline and the probability that makes an answer
# `sure`
DECIDER_API: dict = {"mode": "on", "rate_per_min": 240, "max_texts": 16, "max_chars": 100000,
                     "deadline_ms": 15000, "threshold": 0.8,
                     # X1: a driver session's own rate (texts a second), how many at once, how
                     # long one may idle before its row is written, its texts in all
                     "drive_rate_per_s": 10, "drive_sessions": 4, "drive_idle_s": 30,
                     "drive_max_texts": 36000}
DECIDERS: dict = {
    "models": [{"name": "laya-english", "load": "on_start"},
               {"name": "laya-typed-decisions", "load": "on_demand"},
               {"name": "laya-multilingual", "load": "on_demand"}],
    "questions": {"classify": "auto", "packages": "auto", "judge": "auto", "laya_api": "auto"},
    "api": dict(DECIDER_API),
}

# SRV-137: the session transcripts on the NVMe (server/conductor/transcripts.py): `tools` writes
# a block for a request with tools (and a compaction), counters for the rest (a design decision);
# `all` writes every request, `none` nothing. Tool results and call arguments are clipped at
# `clip_chars`, a loop the D1 guard cut keeps `loop_chars` of its start and end. The secret-pattern
# refusal has no switch: it always runs
TRANSCRIPT_SCOPES = ("tools", "all", "none")
# C1: the disk tier (engine/disk_cache.py, the same defaults as its SETTINGS): on or off, the GB it
# may hold, the days an entry unused is kept, how many cold prefills of a prefix admit it, and the
# free space it leaves the NVMe -- never under 50 GB, which the schema's minimum says too
DISK_CACHE_MODES = ("on", "off")
SETTINGS: dict = {"default_profile": "chat",
                  "journal": {"retention_days": 30, "digest": "tools"},
                  "memory": {"transcripts": {"scope": "tools", "retention_days": 30,
                                             "clip_chars": 4000, "loop_chars": 200},
                             "disk_cache": {"mode": "on", "budget_gb": 16, "ttl_days": 7,
                                            "admit_after": 2, "min_free_gb": 50}}}


# Memory notes (2026-10-05, server/conductor/notes.py, server/conductor/memory.py): the reads and
# the write rules. A profile's `memory` switch says off (absent: every profile stored before),
# shadow (the card goes in and the model's blocks are stored; nothing is read back) or on (the
# index once a session, the bodies on a new user message, the pinned rules). `intent`: Laya's
# question "does this message ask to remember, forget or recall" attaching the `memory` skill
# (shadow: journaled only). `provisional`: provisional notes are read back (marked) as well as
# ratified ones. The settings document's `memory.notes` over these (absent: these)
NOTES: dict = {"index_tokens": 1500, "body_tokens": 600, "pinned_tokens": 300, "bodies": 3,
               "min_score": 0.0, "min_words": 2, "dedupe_threshold": 0.6, "intent": "shadow",
               "provisional": True, "digest_chars": 2000,
               # P-M2: the background writer (server/conductor/memwriter.py)
               "writer": {"mode": "off", "idle_s": 180.0, "due_idle_s": 3.0,
                          "min_new_tokens": 1500, "max_tokens": 600, "max_blocks": 4,
                          "gate": True, "after_compaction": "idle", "turn_deadline_s": 30.0,
                          "existing": 5,
                          # M3 (2026-10-05): the writer's own NONE. `ban_tool_call`: the
                          # `<tool_call>` token cannot be chosen in the writer's turn (the engine:
                          # a logit bias of -100; the proxy: `tool_choice: none`); `prefill`: the
                          # text its answer starts with (empty: none)
                          "ban_tool_call": True, "prefill": ""}}


# SRV-621: the profile a request from a client kind no profile maps gets when it offers
# function tools and `tool_choice` is not "none" (router.resolve). The settings document's
# `tools_profile` over this; "" turns the rule off; a name the store does not hold, or a profile
# with memory, leaves the request to `default_profile`
TOOLS_PROFILE_DEFAULT = "tools"

# the shipped settings (server/conductor/shipped.py): `general` for a client no
# profile names -- an OpenAI SDK, curl, a script -- and `tools` for one of those that offers tools
SHIPPED_SETTINGS: dict = copy.deepcopy(SETTINGS)
SHIPPED_SETTINGS["default_profile"] = "general"
SHIPPED_SETTINGS["tools_profile"] = TOOLS_PROFILE_DEFAULT


def _profile(desc, mode, clients, default_class, decisions, act=(), shadow=(), caps=(2, 6),
             notify=False, lock=None, packs=False) -> dict:
    return {"description": desc, "mode": mode, "clients": list(clients),
            "default_class": default_class,
            "decisions": dict(zip(DECISIONS, decisions)),
            "routing": {},
            "thinking": {"default": "medium", "hints": True, "client_override": True,
                         "by_size": BY_SIZE_DEFAULT},
            "tasks": _tasks(act, shadow),
            "caps": {"acts_per_request": caps[0], "acts_per_session_hour": caps[1]},
            # the tool-gate packs: off, a pack's task (C1 risk, C2 project, C3 same_call,
            # C4 malformed, C5 sequence) does not run whatever its mode says
            "toolgate": {"risk": packs, "project": packs, "same_call": packs,
                         "malformed": packs, "sequence": packs, "tier1_timeout_ms": 500},
            # S1's fails, S4's minutes and NOTIFY (hook.Conductor.CONTEXT); NOTIFY is on only
            # in the overnight profile (the owner, 2026-09-30: allowed, off by default)
            "context": {"s1_runs": 3, "s4_minutes": 20, "notify": notify},
            "hints": {"stair_width": False, "prefix_retention": False},
            # TandemLLM's steps as lines the client shows (states.py); off. A new profile puts
            # them in the content, outside the reasoning (S1); one stored without the key keeps
            # them in the reasoning
            "ui": {"states": "off", "states_channel": "content"},
            **({"locked": lock} if lock else {})}


PROFILES: dict[str, dict] = {
    "raw": _profile("The raw engine: no conductor, byte-identical to the release.", "raw", [],
                    "chat", ("off",) * 4, lock=LOCKED["raw"]),
    "bench": _profile("Benchmarks and the paper: the raw engine, locked.", "raw", [], "chat",
                      ("off",) * 4, lock=LOCKED["bench"]),
    "chat": _profile("Open WebUI and the Playground: class-aware effort, no tool gate.", "on",
                     ("open-webui", "dashboard"), "chat", ("on", "on", "on", "off"),
                     # SRV-199: E1 too -- an Open WebUI agent turn that stops with no call and no
                     # answer (the call written inside the thinking) is not a silent end
                     act=("D1", "D1b", "D2", "E1")),
    "coding": _profile("opencode, interactive: every routing decision on, Tier 0 acts, Tier 1 "
                       "in shadow.", "on", ("opencode",), "code_edit", ("on",) * 4,
                       act=("D1", "D1b", "D2", "E1", "C3", "C4"),
                       shadow=("D3", "E2", "C1", "C2", "S3"), packs=True),
    "overnight": _profile("Unattended opencode runs: coding plus the session tasks, raised caps.",
                          "on", (), "code_edit", ("on",) * 4,
                          act=("D1", "D1b", "D2", "E1", "C3", "C4", "S1", "S2", "S4"),
                          shadow=("D3", "E2", "C1", "C2", "S3"), caps=(3, 12), packs=True,
                          notify=True),
}
# Opencode sends `reasoning_effort: medium` on every turn, whatever the user picked,
# so its profiles ignore the word (the owner, 2026-10-03 12:30); every other profile caps at it
for _n in ("coding", "overnight"):
    PROFILES[_n]["thinking"]["client_word"] = "ignore"

# The profiles a new box starts with (D3; server/conductor/shipped.py stores each one a
# box has no version of at its first start, so the Hermes agent finds `hermes` on a box that has
# its own coding profile already). The built-in PROFILES above stay the minimal ones.
SHIPPED_PROFILES: dict[str, dict] = {
    "raw": _profile("The raw engine: no conductor, byte-identical to the release.", "raw", [],
                    "chat", ("off",) * 4, lock=LOCKED["raw"]),
    "bench": _profile("Benchmarks and the paper: the raw engine, locked.", "raw", [], "chat",
                      ("off",) * 4, lock=LOCKED["bench"]),
    "chat": _profile("Open WebUI and the Playground: class-aware effort, no tool gate.", "on",
                     ("open-webui", "dashboard"), "chat", ("on", "on", "on", "off"),
                     # SRV-199: E1 too -- an Open WebUI agent turn that stops with no call and no
                     # answer (the call written inside the thinking) is not a silent end
                     act=("D1", "D1b", "D2", "E1")),
    "coding": _profile("opencode, interactive: every routing decision on, Tier 0 acts, Tier 1 "
                       "in shadow.", "on", ("opencode",), "code_edit", ("on",) * 4,
                       act=("D1", "D1b", "D2", "E1", "C3", "C4", "E2"),
                       shadow=("D3", "C1", "C2", "S3")),
    "overnight": _profile("Unattended opencode runs: coding plus the session tasks, raised caps.",
                          "on", (), "code_edit", ("on",) * 4,
                          act=("D1", "D1b", "D2", "E1", "C3", "C4", "S1", "S2", "S4"),
                          shadow=("D3", "E2", "C1", "C2", "S3"), caps=(3, 12), notify=True),
    # The Hermes agent (server/ledger.py `is_hermes`) gets coding's policy, and the
    # state lines in the reasoning: Hermes shows a reasoning line as one plain line (X1's inline
    # titles are opencode's parts)
    "hermes": _profile("The Hermes agent: coding's routing, tasks and packages, the steps in the "
                       "reasoning.", "on", ("hermes",), "code_edit", ("on",) * 4,
                       act=("D1", "D1b", "D2", "E1", "C3", "C4", "E2"),
                       shadow=("D3", "C1", "C2", "S3")),
    # Every other client (an OpenAI SDK, curl, a script; settings' default_profile):
    # class-aware effort, a class's packages only when the classifier picked it, the steps in the
    # reasoning, where a client that shows none drops them and the answer stays clean
    "general": _profile("Any other client: class-aware effort and packages, the steps in the "
                        "reasoning.", "on", (), "chat", ("on", "on", "on", "off"),
                        act=("D1", "D1b", "D2", "E1")),
}
# What the live profiles held on 2026-10-03 (coding v14, overnight v8, chat v2), shipped: the
# stall watchdog at 30 minutes in the interactive profiles; the tool-gate packs off; the claim
# check (E2) acting in coding; drawing and 3D at xhigh in
# the coding profiles (night-2 photo3d, 2026-10-02); J1's snapshots on with the outcomes and the
# judge in shadow; the library in shadow (R1); the state lines on, opencode's inline (X1)
for _n in ("chat", "coding", "hermes", "general"):
    SHIPPED_PROFILES[_n]["context"]["s4_minutes"] = 30
    SHIPPED_PROFILES[_n]["tasks"]["S4"]["thresholds"]["minutes"] = 30
for _n in ("coding", "hermes", "overnight"):
    SHIPPED_PROFILES[_n]["routing"] = {"drawing": {"effort": "xhigh"}, "three_d": {"effort": "xhigh"}}
    SHIPPED_PROFILES[_n]["outcomes"] = {"snapshots": "on", "mode": "shadow", "judge": "shadow"}
for _n in ("coding", "hermes"):
    SHIPPED_PROFILES[_n]["library"] = "shadow"
SHIPPED_PROFILES["coding"]["ui"] = {"states": "on", "states_channel": "inline"}
# Opencode sends `reasoning_effort: medium` on every turn, whatever the user picked,
# so its profiles ignore the word (the owner, 2026-10-03 12:30); every other profile caps at it
for _n in ("coding", "overnight"):
    SHIPPED_PROFILES[_n]["thinking"]["client_word"] = "ignore"
SHIPPED_PROFILES["hermes"]["ui"] = {"states": "on", "states_channel": "reasoning"}
SHIPPED_PROFILES["general"]["ui"] = {"states": "on", "states_channel": "reasoning"}
# v3 (2026-10-04): the voice, live lines while Qwen works, in opencode (coding) and in Open WebUI
# and the playground (chat, in the reasoning; the router's and the packages' lines left out there).
# 0.11.3 (SRV-534): the narrator is off in every shipped profile; `live` turns the voice back on
SHIPPED_PROFILES["coding"]["ui"] = dict(SHIPPED_PROFILES["coding"]["ui"], narrator="off")
SHIPPED_PROFILES["chat"]["ui"] = {"states": "on", "states_channel": "reasoning", "narrator": "off",
                                  "kinds": {"route": False, "package": False}}
# SR1: the shadow reader keeps coding sessions and asks the router's question in
# shadow mode on long turns; inert while QWEN38_SHADOW_MODEL is empty
SHIPPED_PROFILES["coding"]["shadow"] = {"mode": "on", "narrator": "off", "router": "shadow",
                                        "router_min_tokens": 8192}
# 0.3.30 night pass: with the voice and the shadow both on, their one share of Qwen's time
# (narrator.Share) at 3 % cost Qwen's decode 3.34 %, at 2.5 % 2.79 %; coding runs both
SHIPPED_PROFILES["coding"]["ui"] = dict(SHIPPED_PROFILES["coding"]["ui"], live={"share": 0.025})
# SRV-471: the memory notes in the interactive profiles, set and not left absent (absent is off);
# `general` serves scripts and SDK clients and keeps the release's request
for _n in ("chat", "coding", "hermes"):
    SHIPPED_PROFILES[_n]["memory"] = "on"
# SRV-473: the same-call guard (C3, `toolgate.same_call`) on in every profile that serves an agent
# loop; it holds calls only on a request whose turn already ends with the repeated call, so every
# other turn streams as before. `general` (scripts, SDK clients) keeps the release's request. Open
# WebUI's tool loop (chat) gets C3 acting too
# SRV-475: the loop escape there as well: after a repeat the session samples with the model card's
# settings until it makes progress; every agentic turn stays greedy (`agentic_default`) until an A/B
for _n in ("chat", "coding", "hermes", "overnight"):
    SHIPPED_PROFILES[_n]["toolgate"]["same_call"] = True
    SHIPPED_PROFILES[_n]["sampling"] = {"agentic_default": "greedy", "loop_escape": "on"}
SHIPPED_PROFILES["chat"]["tasks"]["C3"]["mode"] = "act"
SHIPPED_PROFILES["chat"]["description"] = ("Open WebUI and the Playground: class-aware effort, the "
                                           "same-call guard.")
# SRV-625: the secret echo mask (server/conductor/redact.py) in the profiles whose clients read
# files and show the answer: a key a tool result held is never reproduced verbatim in the answer
for _n in ("chat", "coding"):
    SHIPPED_PROFILES[_n]["secret_mask"] = "on"
# SRV-623: C5, one mutation per turn (`toolgate.sequence`), journals in the agent profiles: shadow
# until a week of rows shows no false positive there. chat-app takes coding's below
for _n in ("coding", "hermes", "chat"):
    SHIPPED_PROFILES[_n]["toolgate"]["sequence"] = True
    SHIPPED_PROFILES[_n]["tasks"]["C5"]["mode"] = "shadow"
# SRV-351 (plan note Chat 7): the Chat tab's agent loop (chatd, client kind `chat-app`) is
# coding's policy without the state lines: the chat shows the harness's choices from the usage
# frame, not as text in the answer. Open WebUI's status events go only to QWEN38_STATUS_CLIENTS'
# kinds, never this one; chatd asks the engine for no titles
SHIPPED_PROFILES["chat-app"] = copy.deepcopy(SHIPPED_PROFILES["coding"])
SHIPPED_PROFILES["chat-app"].update(
    description="The Chat tab's agent loop (chatd): coding's routing, tasks and packages, no "
                "state lines in the answer.",
    clients=["chat-app"], ui={"states": "off", "states_channel": "content"},
    # SRV-471: the memory notes, set here and not inherited (absent is off, and the Chat tab ran
    # without them while Open WebUI and opencode had them)
    memory="on")
# SRV-621: a plain API client that offers tools (an SDK agent loop, curl, a benchmark;
# router.resolve's rule). `general` with nothing added to the prompt and nothing asked on the
# side: no class question (its micro-call cost the bench 1.23x raw's time and moved a near-tie
# answer off raw's), no effort decision, no packages, no rules, no memory (the key absent: no card, no off note), no
# session transcript, no state lines, no loop escape; `prompt: client` drops whatever note,
# card or moved tool text the plan would add, so the prompt ids are the raw path's. The loop
# guards and the tool-call checks act, C5 too (SRV-623); every act is journaled
SHIPPED_PROFILES["tools"] = copy.deepcopy(SHIPPED_PROFILES["general"])
SHIPPED_PROFILES["tools"].update(
    description="API clients that offer tools: the prompt as sent, token for token the raw "
                "path's; the loop guards and the tool-call checks act, each act journaled.",
    decisions={"classify": "off", "effort": "off", "packages": "off", "rules": "off"},
    toolgate=dict(SHIPPED_PROFILES["general"]["toolgate"], same_call=True, malformed=True,
                  sequence=True),
    ui={"states": "off", "states_channel": "content"},
    sampling={"agentic_default": "greedy", "loop_escape": "off"},
    prompt="client", transcripts="off", secret_mask="on")
for _tid, _t in SHIPPED_PROFILES["tools"]["tasks"].items():
    _t["mode"] = "act" if _tid in ("D1", "D1b", "D2", "E1", "C3", "C4", "C5") else "off"
del _n, _tid, _t
# A shipped profile derived from another: its parent and the top-level keys it sets itself. Every
# other top-level key is the parent's (tests/test_conductor_derived.py), so a switch the parent
# gains cannot miss the derived profile again. `general` is no derived profile: it serves scripts
# and SDK clients, whose requests stay the release's (memory absent, off)
DERIVED_PROFILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "chat-app": ("coding", ("description", "clients", "ui", "memory")),
    "tools": ("general", ("description", "decisions", "tasks", "toolgate", "ui", "sampling",
                          "prompt", "transcripts", "secret_mask")),
}


def monitor_seed(path: str | None = None) -> dict:
    """The monitor's policy file (default ops/supervisor.yaml) as a document."""
    from server.conductor.monitor import policy as mpolicy
    return copy.deepcopy(mpolicy.load_default(path).data)


def doc_errors(kind: str, doc) -> list[str]:
    """The schema violations of a document: the monitor's policy by its own validator (its
    schema has a branch per rule), every other kind by `validate`; a roles document's roles by
    the contract's rules too."""
    if kind == "monitor":
        from server.conductor.monitor import policy as mpolicy
        try:
            mpolicy.Policy(doc)
        except (mpolicy.PolicyError, KeyError, TypeError, AttributeError) as exc:
            return [str(exc)]
        return []
    errs = validate(doc, SCHEMAS[kind])
    if kind == "roles" and not errs:
        for name, r in doc.items():
            errs += role_rules(r, f"$.{name}")
    return errs


def role_rules(r: dict, path: str = "$", engine_tokenizer: str | None | bool = False) -> list[str]:
    """The contract's rules a schema cannot say, for one role whose fields are valid.
    `engine_tokenizer`: the hash the engine recorded (None: it recorded none), checked against a
    role that shares token ids; False skips that check (a stored document loads without it)."""
    errs = []
    kind, sharing = r["kind"], r["sharing"]
    if r["writes"] == "client_visible" and kind != "reasoner":
        errs.append(f"{path}.writes: only the reasoner writes tokens the client sees (the honesty "
                    f"rule); a {kind} may write hidden_notes, memory_files or nothing")
    if r["window"] == "in_decode" and kind not in ("decider", "probe"):
        errs.append(f"{path}.window: in_decode is for a decider or a probe only; during decode the "
                    f"GPU belongs to Qwen (a {kind} runs idle or any)")
    if r["window"] == "between_rounds":
        if kind != "generative":
            errs.append(f"{path}.window: between_rounds is for a generative role only (a small "
                        f"model inside the engine that Qwen waits for)")
        elif r["latency_budget_ms"] > BETWEEN_ROUNDS_MAX_MS:
            errs.append(f"{path}.latency_budget_ms: a role between Qwen's rounds holds Qwen at most "
                        f"{BETWEEN_ROUNDS_MAX_MS} ms a call")
    if r["writes"] == "state_lines" and kind != "generative":
        errs.append(f"{path}.writes: state_lines are written by a generative role only")
    if len(set(sharing)) != len(sharing):
        errs.append(f"{path}.sharing: a form is listed twice")
    if "states" in sharing and kind != "probe":
        errs.append(f"{path}.sharing: states (Qwen's own hidden rows) is for a probe inside the "
                    f"engine only")
    if "tokens" in sharing:
        sha = r["tokenizer_sha256"]
        if sha == TOKENIZER_NONE:
            errs.append(f"{path}.tokenizer_sha256: sharing tokens needs the role's tokenizer hash "
                        f"(equal to the engine's); none shares text only")
        elif engine_tokenizer is None:
            errs.append(f"{path}.sharing: the engine recorded no tokenizer hash (ENG-129), so "
                        f"token sharing cannot be checked; share text")
        elif engine_tokenizer is not False and sha != engine_tokenizer:
            errs.append(f"{path}.tokenizer_sha256: {sha[:12]}… is not the engine's tokenizer "
                        f"({engine_tokenizer[:12]}…); a different tokenizer shares text only")
    return errs


def role_errors(fields: dict, path: str = "$", engine_tokenizer: str | None | bool = False
                ) -> list[str]:
    """One role's contract (the ROLE_CONTRACT fields, every one required) and its rules: what
    the decision runtime's registry checks each model it serves against (SRV-133)."""
    errs = validate(fields, _obj(ROLE_CONTRACT, extra=True), path)
    return errs or role_rules(fields, path, engine_tokenizer)


def sharing_forms(r: dict, engine_tokenizer: str | None) -> list[str]:
    """The forms of sharing a role may use, as its tokenizer hash decides them (Harmony 2):
    text always; token ids when its hash is the engine's; Qwen's states for a probe."""
    out = ["text"]
    if engine_tokenizer and r.get("tokenizer_sha256") == engine_tokenizer:
        out.append("tokens")
    if r.get("kind") == "probe":
        out.append("states")
    return out


def upgrade(kind: str, doc) -> tuple[dict, list[str]]:
    """A document stored before a field existed, with the field's default, and what was filled
    (one `role.field=value` each). Only roles have such fields (SRV-133): `memory_gb` becomes
    `memory_claim_gib`, the latency budget is the role's timeout, the kind follows the role's name
    or its contract, the rest are ROLE_DEFAULTS. Anything else comes back as it is."""
    if kind != "roles" or not isinstance(doc, dict):
        return doc, []
    out, applied = {}, []
    for name, r in doc.items():
        if not isinstance(r, dict):
            out[name] = r
            continue
        r = dict(r)
        if "memory_claim_gib" not in r and "memory_gb" in r:
            r["memory_claim_gib"] = r.pop("memory_gb")
            applied.append(f"{name}.memory_claim_gib={r['memory_claim_gib']} (memory_gb)")
        if "kind" not in r:
            r["kind"] = ROLE_KIND_OF.get(name) or CONTRACT_KIND.get(r.get("contract"), "clerk")
            applied.append(f"{name}.kind={r['kind']}")
        if "latency_budget_ms" not in r and isinstance(r.get("timeout_ms"), int):
            r["latency_budget_ms"] = r["timeout_ms"]
            applied.append(f"{name}.latency_budget_ms={r['timeout_ms']} (timeout_ms)")
        for k, v in ROLE_DEFAULTS.items():
            if k not in r:
                r[k] = copy.deepcopy(v)
                applied.append(f"{name}.{k}={v if not isinstance(v, list) else ','.join(v)}")
        out[name] = r
    if "narrator" not in out:
        # P1: a roles document stored before the narrator existed gets the built-in one
        out["narrator"] = copy.deepcopy(ROLES["narrator"])
        applied.append("narrator=built-in")
    if "tester" not in out:
        # L1: a roles document stored before the test judge had a role of its own gets it
        out["tester"] = copy.deepcopy(ROLES["tester"])
        applied.append("tester=built-in")
    return out, applied


def context_stage(router: dict) -> dict:
    """The router's in-context stage with its defaults: a router stored before it existed has
    none, and its stage is off."""
    return {**CONTEXT_STAGE, **((router.get("classifier") or {}).get("context_stage") or {})}


def probe_stage(router: dict) -> dict:
    """The router's probe stage with its defaults (off for a router stored before it)."""
    return {**PROBE_STAGE, **((router.get("classifier") or {}).get("probe_stage") or {})}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) \
            and k != "notes" and k != "questions" else copy.deepcopy(v)
    if isinstance((over or {}).get("notes"), dict):
        out["notes"] = {**base.get("notes", {}), **copy.deepcopy(over["notes"])}
    return out


def outcomes(router: dict) -> dict:
    """The router's `outcomes` section over the defaults (J1); a router stored before it has
    none and gets the defaults."""
    return _merge(OUTCOMES, router.get("outcomes") or {})


def profile_outcomes(profile: dict) -> dict:
    """A profile's `outcomes` switches (J1), each off when absent."""
    return {**PROFILE_OUTCOMES, **(profile.get("outcomes") or {})}


def library(router: dict) -> dict:
    """The router's `library` section over the defaults (R1); a router stored before it has none
    and gets the defaults."""
    return {**copy.deepcopy(LIBRARY), **copy.deepcopy(router.get("library") or {})}


def reading(router: dict) -> dict:
    """The classifier's `reading` section over the defaults (FX6); a router stored before it has
    none and gets the defaults."""
    return {**copy.deepcopy(READING), **copy.deepcopy(router["classifier"].get("reading") or {})}


def tool_fit(router: dict) -> dict:
    """The router's `tool_fit` section over the defaults (SRV-243); a router stored before it has
    none and gets the defaults."""
    return {**TOOL_FIT, **(router.get("tool_fit") or {})}


#: FX15 (UX3, 2026-10-08): how long the package question's answer holds in a session. `class`: a
#: package it picked stays the pick for the follow-ups while the class question keeps the session's
#: class; the question is asked again when the class question moves the session to another class,
#: and after an answer of none (a later message may fit a package). `message`: asked again for
#: every new user message. The prose classes (34 everyday options, 2.4k tokens, 0.55 s warm and
#: 3.3 s on slot 0; UX3 W02 re-asked "make it shorter" and got the package the note already held)
#: take `class`; a coding session's next message is often another task with another skill
#: (lib -> cli, tests/test_note_dedup.py), so the coding classes keep `message`
PACKAGE_STICKY = ("class", "message")


def package_sticky(router: dict, policy: dict | None = None) -> str:
    """The class policy's `package_sticky`, else the router's `classifier.package_sticky`, else
    `message` (every router and class before FX15)."""
    v = (policy or {}).get("package_sticky")
    if v in PACKAGE_STICKY:
        return v
    v = (router.get("classifier") or {}).get("package_sticky")
    return v if v in PACKAGE_STICKY else "message"


def package_prompt(router: dict) -> str | None:
    """The router's own Qwen prompt for the package question (FX6), None when it has none (the
    class question's prompt then, as always)."""
    return router["classifier"].get("package_prompt") or None


def profile_library(profile: dict) -> str:
    """A profile's `library` switch (R1): off when absent."""
    return profile.get("library", "off")


def profile_memory(profile: dict) -> str:
    """A profile's `memory` switch (2026-10-05): off when absent."""
    return profile.get("memory", "off")


PROMPT_MODES = ("conductor", "client")
PROFILE_TRANSCRIPTS = ("settings", "off")


def profile_prompt(profile: dict) -> str:
    """A profile's `prompt` switch (SRV-621): `conductor` when absent (the plan's notes,
    cards and lines go in); `client`, the prompt the client sent, token for token."""
    return profile.get("prompt", "conductor")


def profile_transcripts(profile: dict) -> str:
    """A profile's `transcripts` switch (SRV-621): `settings` when absent (the
    settings document's scope); `off`, no session transcript for its requests."""
    return profile.get("transcripts", "settings")


def profile_secret_mask(profile: dict) -> str:
    """A profile's `secret_mask` (SRV-625, server/conductor/redact.py AnswerMask): `off` when
    absent; `shadow` counts the spans an answer repeats, `on` masks them too."""
    return profile.get("secret_mask", "off")


def tools_profile(settings: dict | None) -> str:
    """The settings' `tools_profile` (SRV-621), TOOLS_PROFILE_DEFAULT when absent; ""
    is the rule off."""
    got = (settings or {}).get("tools_profile")
    return TOOLS_PROFILE_DEFAULT if got is None else str(got)


def notes(settings: dict | None) -> dict:
    """`settings.memory.notes` over the defaults (a settings version stored before has none)."""
    got = (((settings or {}).get("memory") or {}).get("notes") or {})
    out = copy.deepcopy(NOTES)
    out.update({k: v for k, v in got.items() if k in NOTES and k != "writer"})
    out["writer"].update({k: v for k, v in (got.get("writer") or {}).items()
                          if k in NOTES["writer"]})
    return out


def side_requests(router: dict) -> dict:
    """The router's side-request handling (S1): `mode` (on for a router stored before it) and the
    shapes (the document's `side_shapes`, else the built-in ones of signals.py)."""
    from server.conductor import signals
    st = router.get("structural") or {}
    shapes = st.get("side_shapes")
    return {"mode": st.get("side_requests", SIDE_REQUESTS_DEFAULT),
            "shapes": [dict(x) for x in (shapes if shapes is not None else signals.SIDE_SHAPES)]}


def tool_turn_effort(router: dict, profile: dict) -> str:
    """The effort of a tool-result turn (F2): the profile's when it names one (its `inherit` is
    the class's, whatever the router says), else the router's, else `inherit`."""
    th = profile.get("thinking") or {}
    if "tool_turn_effort" in th:
        return th["tool_turn_effort"]
    return (router.get("structural") or {}).get("tool_turn_effort", TOOL_TURN_EFFORT_DEFAULT)


def seed(kind: str, name: str | None = None) -> dict | None:
    """The built-in version 0 of a document, a fresh copy."""
    if kind == "monitor":
        return monitor_seed()
    src = {"settings": SETTINGS, "router": ROUTER, "roles": ROLES, "packages": PACKAGES,
           "residency": RESIDENCY, "deciders": DECIDERS}.get(kind)
    if kind == "profiles":
        src = PROFILES.get(name or "")
    return copy.deepcopy(src) if src is not None else None


# --------------------------------------------------------------------------- JSON Schemas
def _obj(props: dict, required=None, extra=False) -> dict:
    return {"type": "object", "properties": props,
            "required": list(props) if required is None else list(required),
            "additionalProperties": extra}


_STR = {"type": "string", "maxLength": 20000}
_TEXT = {"type": "string", "maxLength": 200000}
_ID = {"type": "string", "pattern": ID_RE}
_PROB = {"type": "number", "minimum": 0, "maximum": 1}
_MS = {"type": "integer", "minimum": 1, "maximum": 120000}
# reasoning tokens, 0 = no cap below the backstop; at most the context (256k, a hard rule)
_BUDGET = {"type": "integer", "minimum": 0, "maximum": 262144}

POLICY = _obj({"effort": {"enum": list(EFFORTS)},
               "effort_by_size": _obj({z: {"enum": list(EFFORTS)} for z in SIZES}, required=()),
               # L1: 12 (was 8): a coding class lists two general packages more, the Laya ones
               # OPS-39: 16, the coding classes list 13 with tandem-core and render-tools
               # PKG3: 48, the prose classes offer the whole everyday set (35 with the cores)
               "packages": {"type": "array", "items": _ID, "maxItems": 48},
               # FX15: how long the package question's answer holds on this class (`class`,
               # `message`); absent: the classifier's `package_sticky` (`package_sticky`)
               "package_sticky": {"enum": list(PACKAGE_STICKY)},
               # FX6: the class's core skills, always in on a turn of the class (like a general
               # package, but for this class alone); one not in `packages` is ignored
               "core": {"type": "array", "items": _ID, "maxItems": 4},
               "rules": {"type": "array", "items": {"type": "string", "maxLength": 2000},
                         "maxItems": 16},
               "note": {"type": "string", "maxLength": 4000}},
              required=())

# SRV-133: the contract fields, in the roles document and in every decision model the engine
# serves (engine/decision/base.py, checked by the registry with `role_errors`)
ROLE_CONTRACT = {
    "kind": {"enum": list(ROLE_KINDS)},
    "memory_claim_gib": {"type": "number", "minimum": 0, "maximum": 128},
    "window": {"enum": list(WINDOWS)},
    "preemptible": {"type": "boolean"},
    "latency_budget_ms": {"type": "integer", "minimum": 1, "maximum": 600000},
    "fallback": {"enum": list(FALLBACKS)},
    "tokenizer_sha256": {"type": "string", "pattern": r"^(none|[0-9a-f]{64})$"},
    "writes": {"enum": list(WRITES)},
    "sharing": {"type": "array", "items": {"enum": list(SHARING)}, "maxItems": len(SHARING)},
}

SCHEMAS: dict[str, dict] = {
    "settings": _obj({"default_profile": _ID,
                      # SRV-621: an unmapped client that offers tools (absent: "tools";
                      # "": the rule off)
                      "tools_profile": {"type": "string", "pattern": r"^$|" + ID_RE},
                      "journal": _obj({"retention_days": {"type": "integer", "minimum": 1,
                                                          "maximum": 400},
                                       "digest": {"enum": ["tools", "all", "none"]}}),
                      "memory": _obj({"transcripts": _obj({
                          "scope": {"enum": list(TRANSCRIPT_SCOPES)},
                          "retention_days": {"type": "integer", "minimum": 1, "maximum": 400},
                          "clip_chars": {"type": "integer", "minimum": 200, "maximum": 100000},
                          "loop_chars": {"type": "integer", "minimum": 0, "maximum": 4000}}),
                          "disk_cache": _obj({
                              "mode": {"enum": list(DISK_CACHE_MODES)},
                              "budget_gb": {"type": "number", "minimum": 1, "maximum": 4000},
                              "ttl_days": {"type": "integer", "minimum": 1, "maximum": 90},
                              "admit_after": {"type": "integer", "minimum": 1, "maximum": 10},
                              "min_free_gb": {"type": "number", "minimum": 50, "maximum": 4000}}),
                          # 0.4.3-test: the session spill's settings over its flags, and the
                          # day's NVMe write budget (absent: the command line's, 64 GB, zstd,
                          # 200 GB a day)
                          "session_spill": _obj({
                              "budget_gb": {"type": "number", "minimum": 1, "maximum": 4000},
                              "codec": {"enum": ["zstd", "off"]},
                              "ttl_hours": {"type": "number", "minimum": 1, "maximum": 720},
                              "min_tokens": {"type": "integer", "minimum": 1024,
                                             "maximum": 1048576}}, required=()),
                          "disk_writes": _obj({
                              "budget_gb": {"type": "number", "minimum": 0, "maximum": 100000}},
                              required=()),
                          # 2026-10-05: the memory notes' reads and write rules (NOTES)
                          "notes": _obj({
                              "index_tokens": {"type": "integer", "minimum": 0, "maximum": 8000},
                              "body_tokens": {"type": "integer", "minimum": 0, "maximum": 4000},
                              "pinned_tokens": {"type": "integer", "minimum": 0, "maximum": 2000},
                              "bodies": {"type": "integer", "minimum": 0, "maximum": 10},
                              "min_score": {"type": "number", "minimum": 0, "maximum": 100},
                              "min_words": {"type": "integer", "minimum": 1, "maximum": 10},
                              "dedupe_threshold": _PROB,
                              "intent": {"enum": list(MODES3)},
                              "provisional": {"type": "boolean"},
                              "digest_chars": {"type": "integer", "minimum": 200,
                                               "maximum": 20000},
                              "writer": _obj({
                                  "mode": {"enum": list(MODES3)},
                                  "idle_s": {"type": "number", "minimum": 10, "maximum": 86400},
                                  "due_idle_s": {"type": "number", "minimum": 0, "maximum": 600},
                                  "min_new_tokens": {"type": "integer", "minimum": 0,
                                                     "maximum": 1000000},
                                  "max_tokens": {"type": "integer", "minimum": 64,
                                                 "maximum": 4096},
                                  "max_blocks": {"type": "integer", "minimum": 1, "maximum": 8},
                                  "gate": {"type": "boolean"},
                                  "after_compaction": {"enum": ["idle", "turn", "off"]},
                                  "turn_deadline_s": {"type": "number", "minimum": 1,
                                                      "maximum": 300},
                                  "existing": {"type": "integer", "minimum": 0, "maximum": 20},
                                  "ban_tool_call": {"type": "boolean"},
                                  "prefill": {"type": "string", "maxLength": 200}},
                                  required=())}, required=())},
                          required=())},
                     required=("default_profile", "journal")),
    "router": _obj({
        "classes": {"type": "array", "maxItems": 20, "items": _obj({
            "id": _ID, "label": {"type": "string", "minLength": 1, "maxLength": 60},
            "description": {"type": "string", "minLength": 8, "maxLength": 600},
            # FX6: what the class builds, in a few words ("a 3D three.js scene"), for the
            # reading line; a class without it never takes part
            "reading": {"type": "string", "minLength": 3, "maxLength": 120},
            "policy": POLICY}, required=("id", "label", "description", "policy"))},
        "classifier": _obj({
            "stages": {"type": "array", "maxItems": 4, "items": _obj({
                "kind": {"enum": ["decider", "qwen"]}, "enabled": {"type": "boolean"},
                "accept": _PROB, "timeout_ms": _MS})},
            "question": {"type": "string", "minLength": 4, "maxLength": 400},
            "digest_chars": {"type": "integer", "minimum": 200, "maximum": 20000},
            "context_line": {"type": "boolean"},
            "package_question": {"type": "string", "minLength": 4, "maxLength": 400},
            "package_accept": _PROB,
            # FX15: the default of the classes' `policy.package_sticky` (absent: `message`)
            "package_sticky": {"enum": list(PACKAGE_STICKY)},
            "qwen_prompt": {"type": "string", "minLength": 10, "maxLength": 4000},
            "package_prompt": {"type": "string", "minLength": 10, "maxLength": 4000},
            "reading": _obj({
                "mode": {"enum": ["off", "shadow", "on"]},
                "below": _PROB, "other": _PROB,
                "text": {"type": "string", "minLength": 10, "maxLength": 1000}},
                required=("mode",)),
            # a first message under the accept bar: the default class, or the top class when it
            # builds something and leads by `margin` (Router._below_build)
            "below_accept": _obj({
                "mode": {"enum": ["default", "build"]},
                "margin": _PROB}, required=("mode",)),
            "context_stage": _obj({
                "mode": {"enum": ["off", "on"]},
                "min_tokens": {"type": "integer", "minimum": 0, "maximum": 262144},
                "followups": {"type": "boolean"},
                "accept": _PROB,
                "question": {"type": "string", "minLength": 10, "maxLength": 4000}},
                required=("mode",)),
            "probe_stage": _obj({
                "mode": {"enum": ["off", "shadow", "on"]},
                "file": {"type": "string", "maxLength": 400},
                "accept": _PROB},
                required=("mode",))},
            required=("stages", "question", "digest_chars", "package_question", "package_accept",
                      "qwen_prompt")),
        "structural": _obj({"compaction": {"type": "string", "maxLength": 40},
                            "tool_turn": {"enum": ["keep", "classify"]},
                            "tool_turn_effort": {"enum": list(TOOL_TURN_EFFORTS)},
                            "side_requests": {"enum": ["off", "on"]},
                            "side_shapes": {"type": "array", "maxItems": 16, "items": _obj({
                                "id": _ID, "client": {"type": "string", "maxLength": 40},
                                "kind": {"enum": list(SIDE_KINDS)},
                                "match": {"type": "string", "minLength": 24, "maxLength": 400},
                                "max_tokens": {"type": "integer", "minimum": 1,
                                               "maximum": 4096}},
                                required=("id", "kind", "match", "max_tokens"))}},
                           required=("compaction", "tool_turn")),
        "stickiness": _obj({"switch": _PROB}),
        "effort": _obj({
            "levels": _obj({lv: _obj({"thinking": {"type": "boolean"},
                                      "budget": _BUDGET,
                                      "note": {"type": "string", "maxLength": 4000}})
                            for lv in EFFORTS}),
            "raise_on_fail": {"type": "boolean"},
            "size": _obj({
                "signals": _obj({n: _obj({"enabled": {"type": "boolean"},
                                          "small": {"type": "number", "minimum": 0},
                                          "medium": {"type": "number", "minimum": 0}})
                                 for n in SIZE_SIGNALS}, required=()),
                "reset_on_compaction": {"type": "boolean"}})},
            required=("levels", "raise_on_fail")),
        "injection": _obj({"wrapper": {"type": "string", "minLength": 6, "maxLength": 4000},
                           "max_note_tokens": {"type": "integer", "minimum": 16,
                                               "maximum": 32768},
                           "max_replay_tokens": {"type": "integer", "minimum": 0,
                                                 "maximum": 65536},
                           "replay": {"type": "boolean"}}),
        # J1: the outcomes' settings over OUTCOMES (absent: the defaults)
        "outcomes": _obj({
            "order": {"type": "array", "items": {"enum": list(OUTCOME_LABELS)},
                      "maxItems": len(OUTCOME_LABELS)},
            "write_tools": {"type": "array", "items": {"type": "string", "maxLength": 80},
                            "maxItems": 32},
            "run_tools": {"type": "string", "maxLength": 2000},
            "evaluate_tools": {"type": "string", "maxLength": 400},
            "console_tools": {"type": "string", "maxLength": 400},
            "output_tools": {"type": "string", "maxLength": 400},
            "patterns": {"type": "object"}, "limits": {"type": "object"},
            "low_quality": {"type": "object"}, "stuck": {"type": "object"},
            "lost_context": {"type": "object"}, "surprise": {"type": "object"},
            "effort": _obj({k: {"enum": list(EFFORTS)} for k in OUTCOME_LABELS}, required=()),
            "raise_turns": {"type": "integer", "minimum": 0, "maximum": 10},
            "acting": {"type": "array", "items": {"enum": list(OUTCOME_LABELS)},
                       "maxItems": len(OUTCOME_LABELS)},
            "notes": _obj({k: {"type": "string", "maxLength": 4000} for k in OUTCOME_LABELS},
                          required=()),
            "cards": {"type": "object"}, "judge": {"type": "object"},
            "evaluate": {"type": "object"}, "retrieve": {"type": "object"}}, required=()),
        # R1: the package library's retrieval over LIBRARY (absent: the defaults)
        "library": _obj({
            "candidates": {"type": "integer", "minimum": 1, "maximum": 10},
            "max_picks": {"type": "integer", "minimum": 1, "maximum": 2},
            "min_score": {"type": "number", "minimum": 0},
            "class_filter": {"type": "boolean"},
            "index": {"type": "string", "maxLength": 400},
            "digest_chars": {"type": "integer", "minimum": 200, "maximum": 20000},
            "question": {"type": "string", "minLength": 4, "maxLength": 400},
            "accept": _PROB,
            "outcomes": {"type": "array", "items": {"enum": list(OUTCOME_LABELS)},
                         "maxItems": len(OUTCOME_LABELS)},
            "card": {"type": "string", "maxLength": 40},
            "card_classes": {"type": "array", "items": _ID, "maxItems": 20},
            "header": {"type": "string", "maxLength": 1000}}, required=()),
        # SRV-243: what a turn whose tools cannot carry out its packages is told instead
        "tool_fit": _obj({"mode": {"enum": list(MODES3)},
                          "note_none": {"type": "string", "maxLength": 2000},
                          "note_write": {"type": "string", "maxLength": 2000},
                          # SRV-615: a deliverable package stays out when a tool delivers
                          "deliverable": {"enum": list(MODES3)},
                          "deliverable_question": {"type": "string", "minLength": 8,
                                                   "maxLength": 600},
                          "deliverable_options": _obj({
                              "tool": {"type": "string", "minLength": 4, "maxLength": 300},
                              "text": {"type": "string", "minLength": 4, "maxLength": 300}}),
                          "deliverable_accept": _PROB}, required=()),
    }, required=("classes", "classifier", "structural", "stickiness", "effort", "injection")),
    "roles": {"type": "object", "additionalProperties": _obj({
        "endpoint": {"type": "string", "maxLength": 400}, "model": {"type": "string",
                                                                   "maxLength": 200},
        "contract": {"enum": list(CONTRACTS)}, "timeout_ms": _MS,
        "mode": {"enum": list(MODES3)}, **ROLE_CONTRACT})},
    "packages": _obj({
        "skill_dirs": {"type": "array", "items": {"type": "string", "maxLength": 400},
                       "maxItems": 8},
        "packages": {"type": "object", "additionalProperties": _obj({
            "title": {"type": "string", "minLength": 1, "maxLength": 80},
            "kind": {"enum": ["skill", "package"]},
            "source": {"type": "string", "pattern": r"^(inline|skill:[a-z0-9][a-z0-9_.-]*|file:.+)$"},
            "description": {"type": "string", "minLength": 8, "maxLength": 600},
            "body": _TEXT, "max_tokens": {"type": "integer", "minimum": 16, "maximum": 32768},
            "enabled": {"type": "boolean"},
            # 2026-10-02: a general package (the coding package) goes in on every turn of a class
            # that lists it, before the task skill the router picks among the rest; absent: false
            "general": {"type": "boolean"},
            # R1: a skill of the package library: searched (server/conductor/libsearch.py), its
            # text before `## Reference` is what goes in, the reference only on an explicit read
            # (GET /v1/library/{name}); absent: false
            "library": {"type": "boolean"},
            # SRV-243: the tool capabilities its steps need (a shell, a file write); a turn whose
            # tools lack one leaves it out, or puts its `fallback` in. Absent: a shell when the
            # text calls TandemLLM through `{tandem_url}`, else none (packages.Library.requires)
            # OPS-39: `render` (SRV-105's render tool, an `html` argument) and `render_video`
            # (`scenes` and `cues` arguments) for the packages that only make sense with them
            "requires_tools": {"type": "array",
                               "items": {"enum": ["shell", "write", "render", "render_video"]},
                               "maxItems": 4},
            "fallback": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9_.-]*$",
                         "maxLength": 80},
            # SRV-615: what the package's result is for when an offered tool can make it: the
            # router asks whether the user wants it done by the tool or the text itself
            "deliverable": {"enum": ["send", "schedule", "post"]}},
            required=("title", "kind", "source", "description", "body", "max_tokens",
                      "enabled"))}}),
    "profiles": _obj({
        "description": {"type": "string", "maxLength": 400},
        "mode": {"enum": list(PROFILE_MODES)},
        "clients": {"type": "array", "items": {"enum": list(CLIENT_KINDS)}, "maxItems": 8},
        "default_class": _ID,
        "decisions": _obj({d: {"enum": list(MODES3)} for d in DECISIONS}),
        "routing": {"type": "object", "additionalProperties": POLICY},
        "thinking": _obj({"default": {"enum": list(EFFORTS)}, "hints": {"type": "boolean"},
                          "client_override": {"const": True},
                          "by_size": {"enum": list(MODES3)},
                          "tool_turn_effort": {"enum": list(TOOL_TURN_EFFORTS)},
                          "client_word": {"enum": list(CLIENT_WORDS)},
                          "budgets": _obj({lv: _BUDGET for lv in EFFORTS if lv != "off"},
                                          required=())},
                         required=("default", "hints", "client_override")),
        "tasks": _obj({tid: _obj({"mode": {"enum": list(TASK_MODES)},
                                  "cap": {"type": "integer", "minimum": 0, "maximum": 10},
                                  "thresholds": _obj({k: {"type": "number", "minimum": 0}
                                                      for k in meta["thresholds"]})})
                       for tid, meta in TASKS.items()},
                      required=[t for t in TASKS if t not in OPTIONAL_TASKS]),
        "caps": _obj({"acts_per_request": {"type": "integer", "minimum": 0, "maximum": 10},
                      "acts_per_session_hour": {"type": "integer", "minimum": 0,
                                                "maximum": 100}}),
        # `sequence` (C5, SRV-623) absent in a profile stored before it: off
        "toolgate": _obj({"risk": {"type": "boolean"}, "project": {"type": "boolean"},
                          "same_call": {"type": "boolean"}, "malformed": {"type": "boolean"},
                          "sequence": {"type": "boolean"}, "tier1_timeout_ms": _MS},
                         required=("risk", "project", "same_call", "malformed",
                                   "tier1_timeout_ms")),
        "context": _obj({"s1_runs": {"type": "integer", "minimum": 1, "maximum": 20},
                         "s4_minutes": {"type": "integer", "minimum": 1, "maximum": 600},
                         "notify": {"type": "boolean"}}),
        "hints": _obj({"stair_width": {"type": "boolean"},
                       "prefix_retention": {"type": "boolean"}}),
        "ui": _obj({"states": {"enum": ["off", "on"]},
                    "kinds": _obj({k: {"type": "boolean"} for k in STATE_KINDS}, required=()),
                    # P1: the narrator writes the lines (absent: off); N2: `thoughts`, it sums
                    # up each stretch of thinking and the other lines stay templates
                    "narrator": {"enum": list(NARRATOR_MODES)},
                    # S1: where the lines go (absent: reasoning, as before)
                    "states_channel": {"enum": list(STATE_CHANNELS)},
                    # v3 (`narrator: live`): the voice's timings and share (absent: its
                    # defaults, server/conductor/voice.py DEFAULTS)
                    "live": _obj({k: {"type": "number", "minimum": 0,
                                      "maximum": {"share": 1.0, "end_ms": 10000}.get(k, 3600)}
                                  for k in VOICE_DEFAULTS}, required=()),
                    # S1: the progress lines (absent: on, every 10 s of a long prefill, every
                    # 2 min or 4,096 tokens of thinking)
                    "progress": _obj({"mode": {"enum": ["off", "on"]},
                                      "prefill_every_s": {"type": "integer", "minimum": 2,
                                                          "maximum": 600},
                                      "thinking_every_s": {"type": "integer", "minimum": 10,
                                                           "maximum": 3600},
                                      "thinking_every_tokens": {"type": "integer",
                                                                "minimum": 256,
                                                                "maximum": 262144}},
                                     required=("mode",))},
                   required=("states",)),
        # J1: snapshots, outcomes, cards, evaluation, the judge, retrieval (absent: all off)
        "outcomes": _obj({k: {"enum": list(v)} for k, v in OUTCOME_SWITCHES.items()},
                         required=()),
        # R1: the package library's retrieval (absent: off)
        "library": {"enum": list(MODES3)},
        # 2026-10-05: the memory notes (absent: off; server/conductor/memory.py)
        "memory": {"enum": list(MODES3)},
        # SRV-621: `client`, nothing of the plan goes into the prompt (absent:
        # `conductor`); `transcripts: off`, no session transcript (absent: the settings' scope)
        "prompt": {"enum": list(PROMPT_MODES)},
        "transcripts": {"enum": list(PROFILE_TRANSCRIPTS)},
        # SRV-625: a credential from a tool result of the conversation, repeated in the answer,
        # is masked (`on`), counted only (`shadow`), or left (absent: off)
        "secret_mask": {"enum": list(MODES3)},
        # SRV-475: the sampling of agent turns (server/conductor/loopescape.py; absent: greedy,
        # escape off). `agentic_default`: a turn that offers tools and names no sampling decodes
        # greedy or with the model card's sampling; `loop_escape`: after a repeat or near-repeat
        # of one call the session samples with the card's settings until it makes progress
        "sampling": _obj({"agentic_default": {"enum": list(AGENTIC_SAMPLING)},
                          "loop_escape": {"enum": ["off", "on"]}}, required=()),
        # SR1: the shadow reader (server/conductor/shadow.py; absent: off): `mode` keeps the
        # session's state in the small model, `narrator` reads the thought summaries from it,
        # `router` asks it the router's class question and journals the agreement (shadow only)
        "shadow": _obj({"mode": {"enum": ["off", "on"]},
                        "narrator": {"enum": ["off", "on"]},
                        "router": {"enum": ["off", "shadow"]},
                        "router_min_tokens": {"type": "integer", "minimum": 0,
                                              "maximum": 262144}},
                       required=("mode",)),
        "locked": {"type": "string", "maxLength": 200},
    }, required=("description", "mode", "clients", "default_class", "decisions", "routing",
                 "thinking", "tasks", "caps", "toolgate", "context", "hints")),
}


def _monitor_schema() -> dict:
    from server.conductor.monitor import policy as mpolicy
    out = mpolicy.schema()
    out.pop("$schema", None)
    out.pop("title", None)
    return out


SCHEMAS["monitor"] = _monitor_schema()
# OPS-47: the engine's memory limits the dashboard overrides (server/memlimits.py), by their
# serve.env names; a key not set here is the flag's. MAX_LEN is not among them (a fixed project limit).
_GIB_LIMIT = {"type": "number", "minimum": 0, "maximum": 128}
ENGINE_LIMITS = _obj({
    "MEM_CEILING_GIB": _GIB_LIMIT, "MEM_PEAK_GIB": _GIB_LIMIT,
    "RESERVE_SLACK_GIB": {"type": "number", "minimum": 0, "maximum": 32},
    "RESIDENT_GB": {"type": "number", "minimum": 0, "maximum": 32},
    "RESIDENT_STASH_GB": {"type": "number", "minimum": 0, "maximum": 32},
    "RESIDENT_GUEST_MAX": {"type": "integer", "minimum": 0, "maximum": 262144},
    "RESIDENT_PIN_GB": {"type": "number", "minimum": 0, "maximum": 32},
    "RESIDENT_PARK_GB": {"type": "number", "minimum": 0, "maximum": 32},
    "CACHE_GB": {"type": "number", "minimum": 0, "maximum": 64},
    "STORE_ENTRY_GB": {"type": "number", "minimum": 0, "maximum": 16},
    "VISION_EMBED_MB": {"type": "number", "minimum": 0, "maximum": 8192},
    "SUFFIX_STORE_MB": {"type": "number", "minimum": 0, "maximum": 4096},
    "PREFIX_CHUNK": {"type": "integer", "minimum": 64, "maximum": 8192},
    "NARROW_ARM": {"enum": [0, 1]}}, required=())
SCHEMAS["residency"] = _obj({
    "floor_gib": {"type": "number", "minimum": 0, "maximum": 128},
    "margin_gib": {"type": "number", "minimum": 0, "maximum": 64},
    "settle_s": {"type": "integer", "minimum": 0, "maximum": 120},
    "order": {"type": "array", "items": _ID, "maxItems": 16},
    "claims": {"type": "object", "additionalProperties": _obj({
        "gib": {"type": "number", "minimum": 0, "maximum": 128},
        "basis": {"enum": list(BASES)},
        "note": {"type": "string", "maxLength": 400}})},
    "engine": ENGINE_LIMITS},
    required=("floor_gib", "margin_gib", "settle_s", "order", "claims"))
SCHEMAS["deciders"] = _obj({
    "models": {"type": "array", "maxItems": 16,
               "items": _obj({"name": _ID, "load": {"enum": list(DECIDER_LOADS)}})},
    "questions": {"type": "object", "additionalProperties": _ID},
    # L1: Laya for the model's scripts (absent: DECIDER_API)
    "api": _obj({"mode": {"enum": ["off", "on"]},
                 "rate_per_min": {"type": "integer", "minimum": 1, "maximum": 6000},
                 "max_texts": {"type": "integer", "minimum": 1, "maximum": 64},
                 "max_chars": {"type": "integer", "minimum": 1000, "maximum": 1000000},
                 "deadline_ms": {"type": "integer", "minimum": 100, "maximum": 60000},
                 "threshold": _PROB,
                 # X1: the driver sessions (server/laya_api.py DRIVER SESSIONS)
                 "drive_rate_per_s": {"type": "integer", "minimum": 1, "maximum": 50},
                 "drive_sessions": {"type": "integer", "minimum": 1, "maximum": 16},
                 "drive_idle_s": {"type": "integer", "minimum": 5, "maximum": 600},
                 "drive_max_texts": {"type": "integer", "minimum": 100, "maximum": 1000000}},
                required=())},
    required=("models", "questions"))


def decider_api(doc: dict) -> dict:
    """The deciders document's `api` over DECIDER_API (L1; a document stored before it has none)."""
    return {**DECIDER_API, **((doc or {}).get("api") or {})}

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Violations of `schema` by `value`, one line each with its JSON path; [] when valid."""
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_TYPES[x](value) for x in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(value).__name__}"]
    errs: list[str] = []
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r:.60} is not one of {', '.join(map(str, schema['enum']))}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} is above {schema['maximum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: {value!r:.60} does not match {schema['pattern']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing {k!r}")
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif extra is False:
                errs.append(f"{path}: unknown key {k!r}")
            elif isinstance(extra, dict):
                if not re.search(ID_RE, str(k)):
                    errs.append(f"{path}: key {k!r} is not an id ({ID_RE})")
                errs += validate(v, extra, f"{path}.{k}")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += validate(v, schema["items"], f"{path}[{i}]")
    return errs


def check(kind: str, doc, *, name: str | None = None, router: dict | None = None,
          packages: dict | None = None, profiles: dict | None = None,
          roles_up: dict | None = None, engine_tokenizer: str | None = None) -> list[str]:
    """The schema of `kind` and the cross-document rules. `router`, `packages`, `profiles`:
    the current documents a reference is checked against; `roles_up`: role -> bool, whether
    the role answers now (None skips the offline rule); `engine_tokenizer`: the tokenizer hash
    the engine recorded, which a role that shares token ids must equal (None: it recorded none)."""
    schema = SCHEMAS.get(kind)
    if schema is None:
        return [f"$: unknown document kind {kind!r}"]
    errs = doc_errors(kind, doc)
    if errs or kind == "monitor":
        return errs
    pkg_names = set((packages or PACKAGES)["packages"])
    if kind == "router":
        ids = [c["id"] for c in doc["classes"]]
        if len(set(ids)) != len(ids):
            errs.append("$.classes: two classes share an id")
        if not ids:
            errs.append("$.classes: at least one class")
        for i, c in enumerate(doc["classes"]):
            for p in c["policy"].get("packages", []):
                if p not in pkg_names:
                    errs.append(f"$.classes[{i}].policy.packages: no package {p!r}")
            for p in c["policy"].get("core", []):
                if p not in pkg_names:
                    errs.append(f"$.classes[{i}].policy.core: no package {p!r}")
        sides = [x["id"] for x in doc["structural"].get("side_shapes") or []]
        if len(set(sides)) != len(sides):
            errs.append("$.structural.side_shapes: two shapes share an id")
        comp = doc["structural"]["compaction"]
        if comp not in ("keep", "classify") and comp not in ids:
            errs.append(f"$.structural.compaction: no class {comp!r} (or keep, classify)")
        if "{text}" not in doc["injection"]["wrapper"]:
            errs.append("$.injection.wrapper: must contain {text}")
        if "{classes}" not in doc["classifier"]["qwen_prompt"]:
            errs.append("$.classifier.qwen_prompt: must contain {classes}")
        if "{classes}" not in doc["classifier"].get("package_prompt", "{classes}"):
            errs.append("$.classifier.package_prompt: must contain {classes}")
        rd = doc["classifier"].get("reading") or {}
        if "text" in rd and "{reading}" not in rd["text"]:
            errs.append("$.classifier.reading.text: must contain {reading}")
        cs = doc["classifier"].get("context_stage") or {}
        if "question" in cs and "{classes}" not in cs["question"]:
            errs.append("$.classifier.context_stage.question: must contain {classes}")
        ps = doc["classifier"].get("probe_stage") or {}
        if ps.get("mode", "off") != "off" and not ps.get("file"):
            errs.append("$.classifier.probe_stage.file: a probe file is needed when the mode is "
                        "not off")
        for n, c in ((doc["effort"].get("size") or {}).get("signals") or {}).items():
            if c["medium"] < c["small"]:
                errs.append(f"$.effort.size.signals.{n}: medium ({c['medium']:g}) is below small "
                            f"({c['small']:g})")
    if kind == "profiles":
        cls = {c["id"] for c in (router or ROUTER)["classes"]}
        if doc["default_class"] not in cls:
            errs.append(f"$.default_class: no class {doc['default_class']!r}")
        for cid, pol in doc["routing"].items():
            if cid not in cls:
                errs.append(f"$.routing: no class {cid!r}")
            for p in pol.get("packages", []):
                if p not in pkg_names:
                    errs.append(f"$.routing.{cid}.packages: no package {p!r}")
        if name in LOCKED:
            errs.append(f"$: profile {name!r} is locked ({LOCKED[name]})")
        if roles_up is not None:
            for tid, t in doc["tasks"].items():
                role = TASKS[tid]["role"]
                if t["mode"] == "act" and TASKS[tid]["tier"] == "1" and role and \
                        not roles_up.get(role):
                    errs.append(f"$.tasks.{tid}.mode: act needs the {role} role, which is "
                                f"offline")
        others = {n: p for n, p in (profiles or {}).items() if n != name}
        for c in doc["clients"]:
            for n, p in others.items():
                if c in p.get("clients", []):
                    errs.append(f"$.clients: {c!r} already maps to profile {n!r}")
    if kind == "settings":
        if doc["default_profile"] not in set(profiles or PROFILES) | set(PROFILES):
            errs.append(f"$.default_profile: no profile {doc['default_profile']!r}")
        tp = doc.get("tools_profile")
        if tp and tp not in set(profiles or PROFILES) | set(PROFILES):
            errs.append(f"$.tools_profile: no profile {tp!r}")
    if kind == "residency" and len(set(doc["order"])) != len(doc["order"]):
        errs.append("$.order: a role is listed twice")
    if kind == "residency":
        eng = doc.get("engine") or {}
        if (eng.get("MEM_CEILING_GIB") is not None and eng.get("MEM_PEAK_GIB") is not None
                and eng["MEM_PEAK_GIB"] < eng["MEM_CEILING_GIB"]):
            errs.append("$.engine: MEM_PEAK_GIB is under MEM_CEILING_GIB (the peak is the ceiling "
                        "plus a prefill's workspace)")
    if kind == "roles":
        for rname, r in doc.items():
            if "tokens" in r["sharing"]:
                errs += role_rules(r, f"$.{rname}", engine_tokenizer)
    if kind == "deciders":
        names = [m["name"] for m in doc["models"]]
        if len(set(names)) != len(names):
            errs.append("$.models: a model is listed twice")
    if kind == "packages":
        used = set()
        for c in (router or ROUTER)["classes"]:
            used |= set(c["policy"].get("packages", []))
        for p in sorted(used - set(doc["packages"])):
            errs.append(f"$.packages: {p!r} is used by a class and cannot be removed")
        for n, p in doc["packages"].items():
            if p["source"] == "inline" and not p["body"].strip():
                errs.append(f"$.packages.{n}.body: an inline package needs its text")
            # SRV-163: TandemLLM's own routes only through `{tandem_url}` (the request's address,
            # or the operator's QWEN38_TANDEM_URL): a literal host failed TLS where its root CA
            # was not trusted and cost a session 42 % of its time
            if p["source"] == "inline":
                hosts = _literal_api_hosts(p["body"])
                if hosts:
                    errs.append(f"$.packages.{n}.body: calls TandemLLM at {', '.join(hosts)}; "
                                f"write {{tandem_url}} instead (the address the client used)")
            # SRV-243: a fallback is another package, never itself
            fb = p.get("fallback")
            if fb is not None and (fb == n or fb not in doc["packages"]):
                errs.append(f"$.packages.{n}.fallback: no other package {fb!r}")
    return errs


def meta() -> dict:
    """What the dashboard's editors render from: the enums, the task catalogue with units and
    defaults, the locked profiles and the schemas themselves."""
    return {"modes": list(MODES3), "task_modes": list(TASK_MODES),
            "profile_modes": list(PROFILE_MODES), "efforts": list(EFFORTS),
            "sizes": list(SIZES), "size_signals": list(SIZE_SIGNALS),
            "tool_turn_efforts": list(TOOL_TURN_EFFORTS),
            "decisions": list(DECISIONS), "client_kinds": list(CLIENT_KINDS),
            "state_kinds": list(STATE_KINDS), "state_channels": list(STATE_CHANNELS),
            "side_kinds": list(SIDE_KINDS),
            "locked": dict(LOCKED),
            # SRV-133: the role contract's values, for the roles editor's selects
            "roles": {"kinds": list(ROLE_KINDS), "windows": list(WINDOWS),
                      "fallbacks": list(FALLBACKS), "writes": list(WRITES),
                      "sharing": list(SHARING), "contracts": list(CONTRACTS),
                      "defaults": copy.deepcopy(ROLE_DEFAULTS)},
            "tasks": {tid: {"label": m["label"], "family": m["family"], "tier": m["tier"],
                            "role": m["role"], "why": m["why"],
                            "thresholds": {k: {"default": v[0], "unit": v[1]}
                                           for k, v in m["thresholds"].items()}}
                      for tid, m in TASKS.items()},
            "schemas": copy.deepcopy(SCHEMAS)}
