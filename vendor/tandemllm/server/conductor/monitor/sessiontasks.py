"""The session-level tasks (SRV-121): S2 context health, S1 gate stagnation, S4 stall watchdog.

They look across turns, so they read what a request carries about the turns before it: the tool
calls the model made (the assistant messages' `tool_calls`) and their results (the `tool`
messages), each taken once per session by its call id, so a retried request counts nothing twice.
All three run at SP4 and act with one PROMPT_TAIL_NOTE at the end of the prompt, which leaves the
resident prefix intact (the next request, without the note, resumes at the last grid line before
it). At most one of them adds a note to a request, in the policy file's order (S2, S1, S4); the
others keep their state and may act on the next request. They stay silent on a request that
offers no tools and in a project whose system prompt does not match `requires` (a project
without state files or checkers).

    S2  context_health   opencode compacted the session (the monitor marks the compaction
                         request: no tools, the history as one user message), and the first
                         `steps` tool-call steps after it read none of the state files (`files`
                         through `read_tools`, or a skill through `skill_tools`): the re-read
                         note, once per compaction. A step that reads one ends the watch.
    S1  gate_stagnation  a checker gate (`RESULT: FAIL <- (a) ... 3.5 > 1.0`) failed `fails` times
                         in a row and the best of the later readings is not `min_improvement`
                         closer to its limit than the first: the note naming the readings and
                         the stage rule, once per gate and stage (the stage is the `next:` the
                         state file last showed). Several stuck gates: one note, the worst.
    S4  stall_watchdog   requests keep coming, but no file changed (`change`), nothing was
                         committed (`commit`) and no checker number got better for `minutes`:
                         the no-progress note, then again only after another `minutes`. When no
                         request comes for `minutes` after a turn ended, the engine cannot act in
                         band: `Monitor.sweep` journals `client_gone`, and pushes one NOTIFY only
                         when `notify.enabled` (off by default; the overnight profile turns it
                         on, a design decision of 2026-09-30).

The state lives in `SessionState.tasks[task id]` and holds hashes, numbers and file names, never
a result's text. Times come from the request (`sup.now`), so a test drives them with a clock.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict

from server.conductor import signals
from server.conductor.monitor.policy import PASS, PROMPT_TAIL_NOTE, Action
from server.conductor.monitor.tier0 import register

SEEN_MAX = 4096
_TEXT = {"type": "string", "minLength": 1}
_NAMES = {"type": "array", "items": _TEXT}
_REQUIRES = {"type": "string",
             "description": "a pattern the system prompt must match, else the task stays silent "
                            "(a project without state files); empty: always"}
_CHECKERS = {
    "type": "array", "minItems": 1,
    "description": "the checkers whose RESULT lines count: a name and its header line (the "
                   "first group is what it checked, so one gate on two targets is two gates)",
    "items": {"type": "object", "properties": {"id": _TEXT, "header": _TEXT},
              "required": ["id", "header"], "additionalProperties": False},
}
_GATE_P = dict(_TEXT, description="a failing reason's gate label, its first group (\"(a) ...\" "
                                  "-> a); a reason without one is gated by its metric name")
_READING = dict(_TEXT, description="a reason's number, comparison and limit, three groups "
                                  "(\"12.3 > 1.0\")")
_IMPROVE = {"type": "number", "minimum": 0, "maximum": 1,
            "description": "the share of the distance to the limit a reading must win to count "
                           "as better"}


# ------------------------------------------------------------------ what a request carries

def _hash(*parts) -> str:
    return hashlib.sha256("\x00".join(str(p) for p in parts).encode()).hexdigest()[:16]


def steps(messages) -> list[tuple[str, list[tuple[str, str]]]]:
    """The model's tool-call steps, in order: `(key, [(name, arguments), ...])`, one per
    assistant message with calls; the key is its first call's id."""
    out = []
    for m in messages or ():
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        calls = []
        for c in m.get("tool_calls") or ():
            fn = (c or {}).get("function") or {}
            args = fn.get("arguments")
            calls.append((str(fn.get("name") or ""),
                          args if isinstance(args, str) else json.dumps(args or {})))
        if calls:
            first = (m.get("tool_calls") or [{}])[0] or {}
            out.append((str(first.get("id") or "") or "h:" + _hash(*calls), calls))
    return out


def results(messages) -> list[tuple[str, tuple[str, str] | None, str]]:
    """The tool results, in order: `(key, (name, arguments) of the call or None, text)`; the key
    is the call id the result answers."""
    calls, out = {}, []
    for m in messages or ():
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            for c in m.get("tool_calls") or ():
                fn = (c or {}).get("function") or {}
                args = fn.get("arguments")
                calls[str((c or {}).get("id") or "")] = (
                    str(fn.get("name") or ""),
                    args if isinstance(args, str) else json.dumps(args or {}))
        elif m.get("role") == "tool":
            cid = str(m.get("tool_call_id") or "")
            text = signals._text(m.get("content"))
            out.append((cid or "h:" + _hash(text), calls.get(cid), text))
    return out


def _fresh(state: dict, key: str) -> bool:
    """True the first time this task sees `key` in this session."""
    seen = state.setdefault("seen", OrderedDict())
    if key in seen:
        return False
    seen[key] = True
    while len(seen) > SEEN_MAX:
        seen.popitem(last=False)
    return True


def _applies(task, sup) -> bool:
    """A request the session tasks act on: tools offered, one choice, the project matches."""
    req = sup.req
    if not req.tools_offered or req.n != 1:
        return False
    pat = task.get("requires") or ""
    return not pat or re.search(pat, sup.shape.system_text or "") is not None


def _note_free(sup) -> bool:
    """No other session task added a note to this request (one note a request)."""
    return "session_note" not in sup.memo


def _note(sup, task, reason: str, text: str, detail: dict, score=None) -> Action:
    sup.memo["session_note"] = task.id
    sup.session.note_at = sup.now
    return Action(PROMPT_TAIL_NOTE, reason=reason, text=text, detail=detail, score=score)


def _names_file(arguments: str, files) -> str | None:
    for f in files:
        if f and f in arguments:
            return f
    return None


def _fmt(v: float) -> str:
    return format(v, "g")


# ------------------------------------------------------------------ checker gates

def _gap(r: dict) -> float:
    """How far a failing reading is from passing, in its own units (0 = at the limit)."""
    v, lim = r["value"], r["limit"]
    return max(0.0, v - lim) if r["op"] in (">", ">=") else max(0.0, lim - v)


def _keyed(readings: list[dict]) -> dict:
    """A gate's readings by name and the order among equal names, so two runs line up."""
    out, n = {}, {}
    for r in readings:
        i = n[r["name"]] = n.get(r["name"], -1) + 1
        out[(r["name"], i)] = r
    return out


def checker_runs(text: str, task) -> list[dict]:
    """The checker runs in one tool result, with the task's patterns."""
    if "RESULT:" not in text:                 # most results are file reads and listings
        return []
    return signals.checker_results(
        text, {c["id"]: c["header"] for c in task["checkers"]}, task["gate"], task["reading"])


def _stuck(runs: list[list[dict]], share: float) -> bool:
    """The last runs of one gate (each a list of failing readings): stuck when no reading of the
    later runs came `share` of its distance closer to the limit than in the first run, and none
    of the first run's readings passed."""
    first, later = _keyed(runs[0]), [_keyed(r) for r in runs[1:]]
    for k, r0 in first.items():
        g0 = _gap(r0)
        for run in later:
            if k not in run:
                return False                          # it passed in a later run
            if g0 > 0 and g0 - _gap(run[k]) >= share * g0:
                return False
    return True


@register("gate_stagnation", "SP4", params={
    "requires": _REQUIRES,
    "fails": {"type": "integer", "minimum": 1,
              "description": "failed runs in a row of one gate before the note"},
    "min_improvement": _IMPROVE,
    "checkers": _CHECKERS, "gate": _GATE_P, "reading": _READING,
    "stage": {"type": "object",
              "description": "where the stage comes from: the `pattern`'s first group in a read "
                             "or a write of one of `files`; once per gate and stage",
              "properties": {"files": _NAMES, "pattern": _TEXT},
              "required": ["files", "pattern"], "additionalProperties": False},
    "text": dict(_TEXT, description="the note: {gate}, {checker}, {values} (the gate's "
                                    "readings, oldest first), {fails}"),
}, required=("fails", "min_improvement", "checkers", "gate", "reading", "stage", "text"))
def gate_stagnation(ev, task, sup):
    """S1: the same checker gate fails `fails` times without getting better -> end the stage."""
    st = sup.session.tasks.setdefault(task.id, {})
    gates = st.setdefault("gates", {})
    stage_rx = re.compile(task["stage"]["pattern"], re.M)
    for key, call, text in results(sup.req.messages):
        if not _fresh(st, key):
            continue
        if call is not None and _names_file(call[1], task["stage"]["files"]):
            # what the file says now: a read's text, else the last line a write or edit put in
            found = stage_rx.findall(text) or stage_rx.findall(_unescaped(call[1]))
            if found:
                st["stage"] = found[-1] if isinstance(found[-1], str) else found[-1][0]
        for run in checker_runs(text, task):
            where = f"{run['checker']}:{run['target']}"
            failing: dict[str, list] = {}
            for r in run["readings"]:
                failing.setdefault(r["gate"], []).append(r)
            for g in gates.values():
                if g["where"] == where and (run["verdict"] == "PASS"
                                            or g["gate"] not in failing):
                    g["runs"], g["noted"] = [], []        # passed: a new streak starts after
            if run["verdict"] == "PASS":
                continue
            for gate, rs in failing.items():
                g = gates.setdefault(f"{where}:{gate}", {"where": where,
                                                         "checker": run["checker"],
                                                         "gate": gate, "runs": [], "noted": []})
                g["runs"] = (g["runs"] + [[dict(r) for r in rs]])[-int(task["fails"]):]
    if not _applies(task, sup):
        return None
    stage = st.get("stage") or ""
    stuck = []
    for gk, g in gates.items():
        if len(g["runs"]) >= int(task["fails"]) and stage not in g["noted"] \
                and _stuck(g["runs"], float(task["min_improvement"])):
            last = g["runs"][-1][0]
            rel = _gap(last) / abs(last["limit"]) if last["limit"] else _gap(last)
            stuck.append((rel, gk, g))
    if not stuck or not _note_free(sup):
        return None
    stuck.sort(key=lambda x: -x[0])
    _, gk, g = stuck[0]
    gate = g["gate"]
    values = [run[0]["value"] for run in g["runs"]]
    for _, _, other in stuck:
        other["noted"].append(stage)
    text = (task["text"].replace("{gate}", gate).replace("{checker}", g["checker"])
            .replace("{values}", ", ".join(_fmt(v) for v in values))
            .replace("{fails}", str(len(values))))
    return _note(sup, task, f"gate_stagnation({gate}: {', '.join(_fmt(v) for v in values)})",
                 text, {"gate": gk, "values": values, "stage": stage or None,
                        "limit": g["runs"][-1][0]["limit"], "op": g["runs"][-1][0]["op"],
                        "stuck": [x[1] for x in stuck]}, score=round(stuck[0][0], 3))


def _unescaped(arguments: str) -> str:
    """A call's string arguments joined, so a pattern with ^ finds a line the model wrote."""
    try:
        obj = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
    vals = obj.values() if isinstance(obj, dict) else [obj]
    return "\n".join(v for v in vals if isinstance(v, str))


# ------------------------------------------------------------------ context health

@register("context_health", "SP4", params={
    "requires": _REQUIRES,
    "files": dict(_NAMES, description="the state files: a read of one of them (a call of a "
                                      "read tool whose arguments name it) is a re-read"),
    "read_tools": dict(_NAMES, description="the tools whose calls count as a read"),
    "skill_tools": dict(_NAMES, description="the tools that load a skill: a call counts as a "
                                            "re-read"),
    "steps": {"type": "integer", "minimum": 1,
              "description": "tool-call steps after the compaction that may pass without a "
                             "re-read"},
    "text": dict(_TEXT, description="the note"),
}, required=("files", "read_tools", "skill_tools", "steps", "text"))
def context_health(ev, task, sup):
    """S2: after a compaction the first steps read none of the state files -> the re-read note."""
    st = sup.session.tasks.setdefault(task.id, {})
    at = sup.session.last_compaction
    if at is not None and at != st.get("compaction"):
        # a new compaction: watch the steps that come after it
        st.update(compaction=at, watching=True, baseline=None, steps=[])
        st["count"] = st.get("count", 0) + 1
    if not st.get("watching") or sup.shape.compaction:
        return None
    seen = [k for k, _ in steps(sup.req.messages)]
    if st["baseline"] is None:
        st["baseline"] = seen                     # what the first request after it carried
        return None
    base = set(st["baseline"])
    for key, calls in steps(sup.req.messages):
        if key in base or key in st["steps"]:
            continue
        st["steps"].append(key)
        read = None
        for name, args in calls:
            if name in task["skill_tools"]:
                read = f"{name}()"
            elif name in task["read_tools"] and _names_file(args, task["files"]):
                read = f"{name}({_names_file(args, task['files'])})"
            if read:
                break
        if read:
            st["watching"] = False
            if not _applies(task, sup):
                return None
            return Action(PASS, reason=f"reread(step {len(st['steps'])}: {read})",
                          detail={"step": len(st["steps"]), "read": read,
                                  "compaction": st["count"]})
    if len(st["steps"]) < int(task["steps"]) or not _applies(task, sup) or not _note_free(sup):
        return None
    st["watching"] = False
    return _note(sup, task, f"context_health(no re-read in {len(st['steps'])} steps)",
                 task["text"], {"steps": len(st["steps"]), "files": list(task["files"]),
                                "compaction": st["count"],
                                "minutes_since": round((sup.now - at) / 60.0, 1)})


# ------------------------------------------------------------------ the stall watchdog

_PATTERN_SET = {
    "type": "object",
    "properties": {"tools": _NAMES, "ok": dict(_TEXT, description="the result's pattern when "
                                                                  "it worked"),
                   "args": dict(_TEXT, description="the call's arguments must match")},
    "required": ["tools", "ok"], "additionalProperties": False,
}
NOTIFY_SCHEMA = {
    "type": "object",
    "description": "one push through the ntfy stack when the client has gone; never in shadow, "
                   "off unless enabled (the overnight profile turns it on)",
    "properties": {
        "enabled": {"type": "boolean"},
        "url": {"type": "string",
                "description": "the ntfy topic URL; empty: the environment's TANDEM_NOTIFY_URL"},
        "token_file": {"type": "string",
                       "description": "a file on the box holding the publish token (never in "
                                      "the policy itself)"},
        "ca_file": {"type": "string", "description": "the CA bundle for the URL, empty: the "
                                                     "system's"},
        "title": dict(_TEXT, description="{session}, {client}, {profile}, {minutes}"),
        "message": dict(_TEXT, description="{session}, {client}, {profile}, {minutes}, "
                                           "{requests}"),
        "priority": {"enum": ["min", "low", "default", "high", "urgent"]},
        "tags": {"type": "string"},
        "timeout_s": {"type": "number", "minimum": 1, "maximum": 60},
    },
    "required": ["enabled", "url", "token_file", "ca_file", "title", "message", "priority",
                 "tags", "timeout_s"],
    "additionalProperties": False,
}


@register("stall_watchdog", "SP4", params={
    "requires": _REQUIRES,
    "minutes": {"type": "number", "exclusiveMinimum": 0,
                "description": "minutes without progress before the note; also the idle time "
                               "after a turn before the client counts as gone"},
    "change": dict(_PATTERN_SET, description="a file changed: a call of these tools whose "
                                             "result matches `ok`"),
    "commit": dict(_PATTERN_SET, description="a commit: a call of these tools whose arguments "
                                             "match `args` and whose result matches `ok`"),
    "checkers": _CHECKERS, "gate": _GATE_P, "reading": _READING, "min_improvement": _IMPROVE,
    "text": dict(_TEXT, description="the note: {minutes}"),
    "notify": NOTIFY_SCHEMA,
}, required=("minutes", "change", "commit", "checkers", "gate", "reading", "min_improvement",
             "text", "notify"))
def stall_watchdog(ev, task, sup):
    """S4: no file changed, no commit and no better checker number for `minutes` -> a note."""
    st = sup.session.tasks.setdefault(task.id, {})
    st.setdefault("progress_at", sup.session.created)
    best = st.setdefault("best", {})
    change, commit = task["change"], task["commit"]
    for key, call, text in results(sup.req.messages):
        if not _fresh(st, key):
            continue
        what = None
        if call is not None:
            name, args = call
            if name in change["tools"] and re.search(change["ok"], text, re.M):
                what = f"{name}: changed"
            elif (name in commit["tools"] and re.search(commit.get("args") or "", args)
                  and re.search(commit["ok"], text, re.M)):
                what = f"{name}: commit"
        for run in checker_runs(text, task):
            if run["verdict"] == "PASS":
                what = f"{run['checker']}: PASS"
                continue
            for r in run["readings"]:
                rk = f"{run['checker']}:{run['target']}:{r['gate']}:{r['name']}"
                gap, prev = _gap(r), best.get(rk)
                if prev is None or gap < prev:
                    best[rk] = gap
                if prev is not None and prev - gap >= float(task["min_improvement"]) * prev > 0:
                    what = f"{run['checker']}: gate ({r['gate']}) {_fmt(r['value'])}"
        if what:
            st["progress_at"], st["progress"] = sup.now, what
    applies = _applies(task, sup)
    if applies:
        # what the watch needs once requests stop: the task as this request saw it
        sup.session.watch = {"task": task.id, "minutes": float(task["minutes"]),
                             "notify": dict(task["notify"]), "mode": sup.mode_of(task),
                             "policy_version": sup.policy.version, "profile": sup.req.profile,
                             "client": sup.req.client}
    if not applies or not _note_free(sup):
        return None
    since = max(st["progress_at"], sup.session.note_at or 0.0, st.get("noted_at") or 0.0)
    idle = (sup.now - since) / 60.0
    if idle < float(task["minutes"]):
        return None
    st["noted_at"] = sup.now
    minutes = (sup.now - st["progress_at"]) / 60.0
    return _note(sup, task, f"stall_watchdog({minutes:.1f} min)",
                 task["text"].replace("{minutes}", _fmt(round(float(task["minutes"]), 1))),
                 {"minutes": round(minutes, 1), "last_progress": st.get("progress")})


# ------------------------------------------------------------------ the turn budget (S5)
#
# SRV-179 / SRV-181 (the UT2 self-test, 2026-10-06). An Open WebUI agent chat ("a Mario game")
# had a working game after 25 minutes and spent the next 18 on a playtest bot of its own, saying
# "The bot is stuck at x=436" seven times with a different edit after each, until the user pressed
# Stop: no answer at all. No rule saw it: every call differed (C3), the session had no checker
# (S1, S4). An opencode one-file page wrote index.html at 642 s of 733 s, after a Firefox rig and
# three hung servers. S5 watches the turn (the requests that carry the same last user message)
# and adds one note when one of three arms fires:
#
#   repeat       the model's visible text between its tool calls (the step's `content`, which
#                Open WebUI and opencode send back) states the same problem `repeats` times
#                among its last `window` problem leads: the leads (first sentence) share
#                `similar` of their word 3-grams, name the same numbers and carry a problem word
#                (`cue`) -> stop tuning, deliver, report. After a note the count starts over.
#   first_write  `first_write_s` seconds and `first_write_steps` tool steps into a turn of a
#                build class (`classes`), nothing the user asked for is written (a write outside
#                the scratch directory) -> write the deliverable first. Once a turn.
#   wall_clock   a chat client (`clients`: someone waits in front of it) has had tool turns for
#                `minutes` with no product write for `quiet_minutes` (no progress), or for
#                `hard_minutes` in any case -> finish and answer; then every `again_minutes`
#                the shorter `again_text`. A turn that keeps writing is let run to the hard cap.
#   verify       (SRV-195) a build turn whose deliverable exists, and whose steps
#                since the last product write wrote only scratch files, test tools among them
#                (`verify_scratch_writes` scripts or pages, not logs or data; `verify_s` seconds
#                or more after that write, 3 steps or more) -> stop
#                building test tools, check once, report. Once a turn; a product write starts the
#                count over. Time alone is not a reason (opencode has no clock; a chat client has
#                the wall clock's quiet rule). FX2's opencode pomodoro wrote the
#                page at 176 s and its last product edit at 257 s, then a Firefox Marionette
#                driver of its own in .tandem until the run's limit at 600 s.
#
# The texts are the policy's; the journal row names the arm, the minutes and the counts.

# verify: a scratch write that builds a test tool (a script, a test page), not a log, a pid file or
# test data (the testing package asks for `tee {tandem_scratch}/run.log` on every run)
_TOOL_FILE = re.compile(r"\.(?:py|js|mjs|cjs|ts|sh|bash|html?|rb|go)$")
_WORD = re.compile(r"[a-z0-9_]+")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_SENT = re.compile(r"(?<=[.!?])\s+")
_PATH_KEYS = ("filePath", "file_path", "path", "file")
# a shell write: `cat > file`, `cat >> file << 'EOF'`, `echo ... > file`, `tee [-a] file`; the
# target is a plain path with a file name (no `$(...)`, no /dev/null, no fd like `2>&1`)
_SHELL_WRITE = re.compile(r"(?:\bcat\b|\becho\b|\bprintf\b|\btee\b)[^|;&\n]*?"
                          r"(?:(?<![0-9&])>>?\s*|\btee\s+(?:-a\s+)?)['\"]?"
                          r"((?:~/|\.{0,2}/)?[\w.+-]+(?:/[\w.+-]+)*\.\w+)(?=['\"\s;&|<>)]|$)")


def _lead(text: str, words: int = 40) -> str:
    """The first sentence of a step's visible text (two when the first is under five words)."""
    from server.conductor import states as states_mod
    text = states_mod.strip_text(text or "").strip()
    if not text:
        return ""
    parts = _SENT.split(text.replace("\n", " "), maxsplit=2)
    lead = parts[0]
    if len(_WORD.findall(lead.lower())) < 5 and len(parts) > 1:
        lead = lead + " " + parts[1]
    return " ".join(lead.split()[:words])


def _grams(lead: str) -> set:
    w = _WORD.findall(lead.lower())
    return {" ".join(w[i:i + 3]) for i in range(len(w) - 2)}


def _similar(a: dict, b: dict, share: float) -> bool:
    """Two leads say the same: `share` of the shorter one's 3-grams in both, and when both name
    numbers, at least one number in common ("3 tests fail" and "2 tests fail" are progress)."""
    if a["nums"] and b["nums"] and not set(a["nums"]) & set(b["nums"]):
        return False
    small = min(len(a["grams"]), len(b["grams"]))
    return small > 0 and len(a["grams"] & b["grams"]) / small >= share


def turn_steps(messages) -> tuple[str | None, list]:
    """The turn's key (the last user message, hashed with its position) and its steps after it:
    `[(key, visible text, [(name, arguments)], {call id: result text})]`."""
    msgs = [m for m in messages or () if isinstance(m, dict)]
    last = None
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            last = i
    if last is None:
        return None, []
    turn = _hash(last, signals._text(msgs[last].get("content")))
    out, res = [], {}
    for m in msgs[last + 1:]:
        if m.get("role") == "tool":
            res[str(m.get("tool_call_id") or "")] = signals._text(m.get("content"))
    for m in msgs[last + 1:]:
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        calls, ids = [], []
        for c in m.get("tool_calls") or ():
            fn = (c or {}).get("function") or {}
            args = fn.get("arguments")
            calls.append((str(fn.get("name") or ""),
                          args if isinstance(args, str) else json.dumps(args or {})))
            ids.append(str((c or {}).get("id") or ""))
        key = ids[0] or "h:" + _hash(*calls)
        out.append((key, signals._text(m.get("content")), calls,
                    {i: res.get(i, "") for i in ids}))
    return turn, out


def written_paths(calls, task) -> list[str]:
    """The paths a step's calls wrote: a write tool's path argument, or a shell redirect."""
    out = []
    for name, args in calls:
        try:
            obj = json.loads(args) if isinstance(args, str) else (args or {})
        except ValueError:
            obj = {}
        if not isinstance(obj, dict):
            continue
        if name in task["write_tools"]:
            p = next((obj[k] for k in _PATH_KEYS if isinstance(obj.get(k), str)), None)
            if p:
                out.append(p)
        elif name in task["shell_tools"]:
            cmd = obj.get("command") or obj.get("cmd") or ""
            if isinstance(cmd, str):
                out.extend(p for p in _SHELL_WRITE.findall(cmd) if not p.startswith("/dev/"))
    return out


def _product(path: str, task) -> bool:
    """A write the user asked for: not under the scratch directory, not a temporary file."""
    p = path.strip()
    return bool(p) and not any(re.search(rx, p) for rx in task["scratch"])


@register("turn_budget", "SP4", params={
    "requires": _REQUIRES,
    "repeats": {"type": "integer", "minimum": 2,
                "description": "repeat: steps whose visible text states the same problem"},
    "similar": _IMPROVE | {"description": "repeat: the share of word 3-grams two leads share"},
    "window": {"type": "integer", "minimum": 2,
               "description": "repeat: how many problem leads are kept to compare"},
    "cue": dict(_TEXT, description="repeat: a lead counts only when it matches (a problem word)"),
    "repeat_text": dict(_TEXT, description="repeat's note: {count}, {quote}"),
    "classes": dict(_NAMES, description="first_write: the router classes it watches (a build)"),
    "first_write_s": {"type": "number", "minimum": 0,
                      "description": "first_write: seconds into the turn"},
    "first_write_steps": {"type": "integer", "minimum": 1,
                          "description": "first_write: tool steps into the turn"},
    "first_write_text": dict(_TEXT, description="first_write's note: {minutes}, {steps}"),
    "write_tools": dict(_NAMES, description="tools whose path argument is a write"),
    "shell_tools": dict(_NAMES, description="tools whose command may write with a redirect"),
    "scratch": dict(_NAMES, description="patterns of paths that are not the product (scratch, "
                                        "temporary files)"),
    "clients": dict(_NAMES, description="wall_clock: the chat clients someone waits in front of"),
    "minutes": {"type": "number", "exclusiveMinimum": 0,
                "description": "wall_clock: minutes into the turn with no product write for "
                               "quiet_minutes"},
    "quiet_minutes": {"type": "number", "minimum": 0,
                      "description": "wall_clock: a product write this recent is progress"},
    "hard_minutes": {"type": "number", "exclusiveMinimum": 0,
                     "description": "wall_clock: minutes into the turn, progress or not"},
    "again_minutes": {"type": "number", "exclusiveMinimum": 0,
                      "description": "wall_clock: minutes between the later notes"},
    "wrap_text": dict(_TEXT, description="wall_clock's note: {minutes}"),
    "again_text": dict(_TEXT, description="wall_clock's later notes: {minutes}"),
    "verify_s": {"type": "number", "minimum": 0,
                 "description": "verify: seconds after the last product write, with scratch writes"},
    "verify_scratch_writes": {"type": "integer", "minimum": 1,
                              "description": "verify: scratch-only writes since the last product write"},
    "verify_text": dict(_TEXT, description="verify's note: {minutes}, {steps}"),
}, required=("repeats", "similar", "window", "cue", "repeat_text", "classes", "first_write_s",
             "first_write_steps", "first_write_text", "write_tools", "shell_tools", "scratch",
             "clients", "minutes", "quiet_minutes", "hard_minutes", "again_minutes", "wrap_text",
             "again_text"))
def turn_budget(ev, task, sup):
    """S5: the same problem stated `repeats` times, no deliverable after the setup, or a chat
    turn past its minutes -> one note (SRV-179, SRV-181)."""
    st = sup.session.tasks.setdefault(task.id, {})
    turn, tsteps = turn_steps(sup.req.messages)
    if turn is None:
        return None
    if st.get("turn") != turn:
        st.clear()
        st.update(turn=turn, at=sup.now, leads=[], steps=0, wrote_at=None, noted=[],
                  wrap_at=None, wraps=0, since_steps=0, since_scratch=0)
    cue = re.compile(task["cue"])
    newest = None
    for key, text, calls, _ in tsteps:
        if not _fresh(st, key):
            continue
        st["steps"] += 1
        paths = written_paths(calls, task)
        if any(_product(p, task) for p in paths):
            st["wrote_at"] = sup.now
            st["since_steps"] = st["since_scratch"] = 0
        elif st["wrote_at"] is not None:
            st["since_steps"] += 1
            st["since_scratch"] += 1 if any(_TOOL_FILE.search(p) for p in paths) else 0
        lead = _lead(text)
        if lead and cue.search(lead):
            item = {"grams": _grams(lead), "nums": sorted(set(_NUM.findall(lead))),
                    "quote": lead[:160]}
            st["leads"] = (st["leads"] + [item])[-int(task["window"]):]
            newest = item
    if not _applies(task, sup) or not _note_free(sup):
        return None
    minutes = (sup.now - st["at"]) / 60.0
    detail = {"minutes": round(minutes, 1), "steps": st["steps"],
              "wrote": st["wrote_at"] is not None}

    # repeat: the newest lead and the ones right before it say the same thing
    if newest is not None:
        run = 1 + sum(1 for prev in st["leads"][:-1]
                      if _similar(prev, newest, float(task["similar"])))
        if run >= int(task["repeats"]):
            st["leads"] = []
            st["noted"].append("repeat")
            text = (task["repeat_text"].replace("{count}", str(run))
                    .replace("{quote}", newest["quote"]))
            return _note(sup, task, f"turn_budget(repeat {run})", text,
                         dict(detail, arm="repeat", count=run, numbers=newest["nums"]))

    # first_write: the setup before the deliverable
    if (st["wrote_at"] is None and "first_write" not in st["noted"]
            and sup.req.cls in task["classes"]
            and sup.now - st["at"] >= float(task["first_write_s"])
            and st["steps"] >= int(task["first_write_steps"])):
        st["noted"].append("first_write")
        text = (task["first_write_text"].replace("{minutes}", _fmt(round(minutes, 1)))
                .replace("{steps}", str(st["steps"])))
        return _note(sup, task, f"turn_budget(first_write {minutes:.1f} min)", text,
                     dict(detail, arm="first_write"))

    # verify: the deliverable exists and the steps since its last change built test tooling
    if (st["wrote_at"] is not None and "verify" not in st["noted"]
            and sup.req.cls in task["classes"] and task.get("verify_text")
            and st["since_steps"] >= 3):
        since = sup.now - st["wrote_at"]
        if (since >= float(task.get("verify_s", 90))
                and st["since_scratch"] >= int(task.get("verify_scratch_writes", 2))):
            st["noted"].append("verify")
            text = (task["verify_text"].replace("{minutes}", _fmt(round(since / 60.0, 1)))
                    .replace("{steps}", str(st["since_steps"])))
            return _note(sup, task, f"turn_budget(verify {since / 60.0:.1f} min)", text,
                         dict(detail, arm="verify", since_s=round(since), since_steps=st["since_steps"],
                              scratch_writes=st["since_scratch"]))

    # wall_clock: a chat turn that runs long
    if sup.req.client not in task["clients"]:
        return None
    if st["wrap_at"] is not None:
        if sup.now - st["wrap_at"] < float(task["again_minutes"]) * 60.0:
            return None
        st["wrap_at"], st["wraps"] = sup.now, st["wraps"] + 1
        return _note(sup, task, f"turn_budget(wall_clock again {minutes:.1f} min)",
                     task["again_text"].replace("{minutes}", _fmt(round(minutes))),
                     dict(detail, arm="wall_clock", again=st["wraps"]))
    quiet = (st["wrote_at"] is None
             or sup.now - st["wrote_at"] >= float(task["quiet_minutes"]) * 60.0)
    hard = minutes >= float(task["hard_minutes"])
    if not hard and not (minutes >= float(task["minutes"]) and quiet):
        return None
    st["wrap_at"], st["wraps"] = sup.now, 1
    st["noted"].append("wall_clock")
    return _note(sup, task, f"turn_budget(wall_clock {minutes:.1f} min)",
                 task["wrap_text"].replace("{minutes}", _fmt(round(minutes))),
                 dict(detail, arm="wall_clock", hard=hard, quiet=quiet))
