"""Workers: the task tool's runtime (VIS-E15, SRV-579; the plan is the Memo note "Chat: main agent
and workers (context folding), 2026-10-17"). Mixed into `chatd.service.Service`.

THE TASK. In Agents mode the main agent gives hands-on work to a worker: `task {title, brief,
expect}` (chatd/tools.py). A worker is a full run (chatd/runs.py) on a hidden conversation of kind
`task` whose parent is the chat: the same loop, retries, compaction guard, verify phase and
permission handling as any run, so a loop fix lands once. Its context starts fresh: the worker
block (packages/system/workers/PACKAGE.md, then the chat-app tool rules), the worker tools
(tools.py WORKER_TOOLS) and one user message with the task's number, title, brief, expect, the
chat's todo list and the user's message that started the run. Nothing else of the chat goes in,
and only the task result comes back.

SHARED. Every sandbox, workspace, setting and rule lookup of a task conversation resolves to its
chat (`root_cid`, `settings_of`): one container, one /workspace, the chat's permission mode and
rules. The worker runs in its parent's run slot (`Run.slot_owner`) while the parent waits in its
task call; it gives the slot back while a permission prompt waits, as any run does.

LIMITS. A step with two task calls runs the first; each other answers tools.ONE_TASK. A run gives
at most `task_max_per_run` (8) tasks, a worker makes at most `task_max_steps` (40) model calls and
has no task tool (depth 1).

THE CHAT SEES IT. The worker's session is `chat-<chat>-task-<n>` (render.task_session_id) and its
headers turn the memory notes off. Its transcript's events go to the chat's log with `task_id`
(service.py TASK_FORWARD), beside `task.started`, `task.phase`, `task.metrics` (ephemeral, once a
second: the prefill or decode figures of its own live row) and `task.done`; meanwhile the parent's
phase is `task`, its Now line "Task 2: <title>, call 12 of 40: <the worker's Now>". The context
report lists each running worker under `workers`. `GET .../tasks` lists a chat's tasks,
`GET .../tasks/{id}` gives one with its transcript. A snapshot of the workspace at the worker's
start (key: the task's id) is the base of the task's diff and a rewind point.

THE CONTRACT (SRV-581). A brief has four labelled lines, Goal, Files, Constraints and Check, each
saying something, and no code: a fence, more than 4 tags, a run of more than 8 lines that look
like code, a line of minified code or a "CODE:" block (`code_lines`). A brief describes the job
and the worker writes the code; a brief that carried a whole page made the worker a typist and put
the page in the main context (2026-10-17). Its length is the main agent's: a cap made it cut
requirements to pass. A task call that breaks this is refused before any worker starts, with the
template (`check_brief`). The worker also gets the user's request word for word, after the brief
(`first_message`), so the brief scopes the job and need not repeat its details. The worker ends with a report in a fixed shape,
which chatd parses into a `report.v1` object (`parse_report`): a report that misses a field or
breaks a cap goes back to the worker once, as a one-call tail (`task_report`); missing fields a
second time, chatd builds the report from the worker's calls (`build_report`). At its cap of calls
a worker makes one more call whose only job is the report: the salvage text at the tail, the tool
schemas kept so the prompt's head and its cache stay, its tool calls stored `ignored` and never run,
at most SALVAGE_TOKENS; its status is partial (step limit), never done. Below every report chatd
adds its own check: the files the snapshot diff says changed, the changed ones the report left out,
the calls that ran (render_page and bash, the verify phase's), and the checks the report claims
with no call behind them (`chatd_check`). The main agent reads only `render_result`: the first line
says it is data, at most RESULT_MAX_CHARS; the whole report is in
/workspace/.tandem/tasks/<n>/report.md. No text outside the fields and no code block reaches it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from chatd import attach as attach_mod
from chatd import render
from chatd import context as ctx_mod
from chatd.ids import iso, now_ms, ulid
from chatd.store import dumps
from chatd.tools import PathError, Result

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packages",
                   "system", "workers")
RESULT_MAX_CHARS = 8000
SALVAGE_TOKENS = 1500
#: after Stop, how long the task call waits for its worker to store its end (it closes its stream
#: at once); a worker slower than that stores its report later, and Continue delivers it
STOP_WAIT_S = 15.0
IGNORED_CALL = "Not run: this task's model calls are used up, and this answer is its report."
_TEXTS: dict = {}


def texts() -> dict:
    """The package's texts: `worker` (PACKAGE.md), `main` (references/main.md) and the sections
    of references/brief.md and report.md by their `## ` headings, read once."""
    if not _TEXTS:
        def read(rel: str) -> str:
            with open(os.path.join(PKG, rel), encoding="utf-8") as fh:
                return fh.read()
        _TEXTS["worker"] = read("PACKAGE.md").strip()
        _TEXTS["main"] = read("references/main.md").strip()
        for rel in ("references/brief.md", "references/report.md"):
            for head, body in re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", read(rel), re.M | re.S):
                _TEXTS[head.strip().lower()] = body.strip()
    return _TEXTS


def steer_note(at: str, text: str) -> str:
    """A message the user typed while a worker ran, as the worker reads it (SRV-585)."""
    return texts()["steer note"].replace("{time}", at).replace("{text}", " ".join(text.split()))


def ws_digest(ws) -> str:
    """The workspace's state by its files' paths, sizes and times (no file is read): what the
    main agent's read cache compares (SRV-583)."""
    h = hashlib.sha256()
    for full in ws.walk():
        try:
            st = os.stat(full, follow_symlinks=False)
        except OSError:
            continue
        h.update(f"{os.path.relpath(full, ws.root)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def worker_block(max_steps: int) -> str:
    return texts()["worker"].replace("{max_steps}", str(max_steps))


def first_message(n: int, title: str, brief: str, expect: str, todos: list[dict],
                  request: str = "") -> str:
    """The worker's one user message (the plan note's layout), then the user's request as the
    user wrote it: the brief comes first and says which part of it is this task."""
    lines = [f"Task {n}: {title}", "Brief:", brief.strip(),
             f"The report must also contain: {expect.strip()}"]
    if todos:
        lines.append("The chat's todo list (the main agent keeps it; you cannot change it):")
        lines += [f"- [{t['state']}] {t['text']}" for t in todos]
    if request.strip():
        lines += [REQUEST_LINE, request]
    return "\n".join(lines)


REQUEST_LINE = ("The user's request (verbatim; the main agent's brief above says which part is this "
                "task):")


def request_text(parts: list[dict], attachments: list[dict] | None) -> str:
    """A user message as the worker reads it: its text, and each attachment by its name and where
    it is (never its content)."""
    text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text")
    for a in attachments or []:
        where = (f"at {a['path']}" if a.get("path") else
                 f"not in the workspace ({a['unplaced']})" if a.get("unplaced") else
                 "not in the workspace")
        text += (f"\n[Attached file {a.get('name') or 'file'}: "
                 f"{attach_mod.size_words(int(a.get('bytes') or 0))}, {where}]")
    return text


def final_text(parts: list[dict], after: int = 0) -> str:
    """The worker's last words: the text of the steps after its last tool call (and after step
    `after`: a report sent back is not part of the next one)."""
    last = max([int(p.get("step") or 1) for p in parts if p.get("type") == "tool_call" and
                p.get("status") != "ignored"] + [after])
    return "".join(p.get("text") or "" for p in parts
                   if p.get("type") == "text" and int(p.get("step") or 1) > last).strip()


# ---------------------------------------------------------------------- the brief (SRV-581)
BRIEF_LINES = ("Goal", "Files", "Constraints", "Check")
BRIEF_LINE_MIN = 10
TITLE_MAX_CHARS = 80
CODE_RUN_MAX = 8          # more lines of code in a run: the brief carries code
CODE_TAGS_MAX = 4         # more HTML/XML tags: a page, not a mention of <canvas>
CODE_DENSE_MAX = 8        # more of ; { } on one line: minified code
_BRIEF_LINE = re.compile(r"^\s*(goal|files|constraints|check)\s*:\s*(.*)$", re.I)
_TAG = re.compile(r"</?[A-Za-z][\w:-]*(?:\s[^<>]*)?/?>|<!(?:DOCTYPE|--)", re.I)
_CODE_FENCE = re.compile(r"^\s*(?:```|~~~)")
_CODE_MARK = re.compile(r"(?:\bCODE\b|(?i:\b(?:exact|full) (?:contents?|code|source|text)\b))"
                        r"[^:\n]{0,80}:")
_CODE_LINE = re.compile(
    r"[{};]\s*$|^\s*[})\]]"                                    # a statement or a block ends
    r"|^\s*(?:function|const|let|var|return|import|export|def|class|elif|else|try|except|"
    r"async|await|#include|#define)\b"
    r"|^\s*(?:if|for|while|switch)\s*\(|^\s*(?:if|for|while|with)\b.*:\s*$"
    r"|^\s*(?://|/\*|\*/|@media|@keyframes)"
    r"|^\s*[\w.$\[\]]+\s*[-+*/]?=\s*\S"                         # an assignment
    r"|^\s*[\w.$]+\(.*\)\s*$")                                  # a call on its own line


def code_lines(brief: str) -> list[int]:
    """The brief's lines (0-based) that carry code: a fenced block, every tag line when there are
    more than CODE_TAGS_MAX tags, a run of more than CODE_RUN_MAX code lines (blank lines and a
    single other line between code lines go on with it; a labelled line ends it), a line with more
    than CODE_DENSE_MAX of ; { }, and a "CODE:"/"exact content:" block that holds any of these
    signs. Inline names (`tick()`, `<canvas>`) and a one-line snippet are none of them."""
    lines = brief.splitlines()
    label = [bool(_BRIEF_LINE.match(ln)) for ln in lines]
    tags = [len(_TAG.findall(ln)) for ln in lines]
    code = [not label[i] and (bool(tags[i]) or bool(_CODE_LINE.search(ln)))
            for i, ln in enumerate(lines)]
    found: set[int] = set()
    fence = None
    for i, ln in enumerate(lines):
        if _CODE_FENCE.match(ln):
            if fence is None:
                fence = i
            else:
                found.update(range(fence, i + 1))
                fence = None
    if fence is not None:
        found.update(range(fence, len(lines)))
    if sum(tags) > CODE_TAGS_MAX:                     # a labelled line names tags, the code follows
        found.update([i for i, n in enumerate(tags) if n and not label[i]] or
                     [i for i, n in enumerate(tags) if n])
    found.update(i for i, ln in enumerate(lines)
                 if sum(ln.count(ch) for ch in ";{}") > CODE_DENSE_MAX)
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    start = last = None
    count = 0
    for k, i in enumerate(nonblank + [None]):
        if i is not None and code[i]:
            start, last, count = (i if start is None else start), i, count + 1
            continue
        if (i is not None and start is not None and not label[i] and last == nonblank[k - 1]
                and k + 1 < len(nonblank) and code[nonblank[k + 1]]):
            continue                                        # one other line inside the code
        if count > CODE_RUN_MAX:
            found.update(range(start, last + 1))
        start, count = None, 0
    for i, ln in enumerate(lines):
        if not label[i] and _CODE_MARK.search(ln):
            end = next((j for j in range(i + 1, len(lines)) if label[j]), len(lines))
            if any(code[j] for j in range(i, end)):
                found.update(range(i, end))
    return sorted(found)


def check_brief(args: dict) -> str | None:
    """The refusal of a task call whose brief a worker cannot start from (the tool's answer, with
    the template), or None. Before any worker starts."""
    title, brief, expect = (str(args.get(k) or "").strip() for k in ("title", "brief", "expect"))
    why = None
    if not title:
        why = "the title is empty."
    elif len(title) > TITLE_MAX_CHARS:
        why = f"the title is {len(title)} characters; at most {TITLE_MAX_CHARS}."
    elif carried := code_lines(brief):
        n = len(carried)
        why = (texts()["code"].replace("{lines}", f"{n} line" + ("s" if n != 1 else ""))
               .replace("{line}", str(carried[0] + 1)))
    else:
        found: dict[str, str] = {}
        for ln in brief.splitlines():
            m = _BRIEF_LINE.match(ln)
            if m:
                found.setdefault(m.group(1).lower(), m.group(2).strip())
        missing = [k for k in BRIEF_LINES if k.lower() not in found]
        short = [k for k in BRIEF_LINES if k.lower() in found and
                 len(found[k.lower()]) < BRIEF_LINE_MIN]
        if missing:
            why = "the brief has no " + " and no ".join(f'"{k}:" line' for k in missing) + "."
        elif short:
            why = (" and ".join(f'the "{k}:" line' for k in short) +
                   f" {'says' if len(short) == 1 else 'say'} too little (at least "
                   f"{BRIEF_LINE_MIN} characters after the label).")
        elif not expect:
            why = ("expect is empty: say what the report must contain beyond its standard "
                   "fields, or \"nothing\".")
    return None if why is None else f"task refused: {why}\n{texts()['template']}"


# ---------------------------------------------------------------------- the report (report.v1)
FIELDS = ("Status", "Summary", "Changed files", "Checks", "Expected", "Open problems")
SUMMARY_MAX = 600
CHANGED_LINE_MAX = 200
CHANGED_MAX = 40
ITEMS_MAX, ITEM_MAX = 40, 300           # the other lists: cut, not sent back
STATUSES = ("done", "partial", "failed")
_LABEL = re.compile(r"^\s*(?:#+\s*)?(?:\*\*)?\s*(status|summary|changed files|checks|expected|"
                    r"open problems)\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*(.*?)\s*$", re.I)
_FENCE = re.compile(r"^[ \t]*```[^\n]*\n(.*?)(?:^[ \t]*```[ \t]*$|\Z)", re.M | re.S)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+(.*)$")
_NONE = ("none", "(none)", "none.", "n/a", "-")
_CALL_REF = re.compile(r"\b(?:call[_-][A-Za-z0-9_.:-]+|verify_[A-Za-z0-9]+|c-\d+)\b")
_SHOT_REF = re.compile(r"\bshot-\d+\b")


def _unfence(text: str) -> tuple[str, list[str]]:
    """Every fenced code block cut and named: no file body rides in a report."""
    cuts: list[str] = []

    def one(m):
        cuts.append(f"[code block of {len(m.group(1).rstrip()):,} characters removed]")
        return cuts[-1] + "\n"
    return _FENCE.sub(one, text), cuts


def parse_report(text: str) -> tuple[dict | None, list[str]]:
    """The worker's report as a `report.v1` object, and what is wrong with it: the fields that are
    missing and the caps it breaks, by name (empty: a valid report). The last report in the text
    counts; its fields in any order, a label in any case; a code block is cut and named; a field
    over its cap is cut. None when no field is there at all."""
    text, cuts = _unfence(text or "")
    lines = text.splitlines()
    groups: list[dict[str, list[str]]] = []
    cur: dict[str, list[str]] | None = None
    key = None
    for ln in lines:
        m = _LABEL.match(ln)
        if m:
            key = m.group(1).lower()
            if cur is None or key in cur:           # a second report starts
                cur = {}
                groups.append(cur)
            cur[key] = [m.group(2)] if m.group(2) else []
        elif cur is not None and key is not None:
            cur[key].append(ln)
    if not groups:
        return None, list(FIELDS)
    g = groups[-1]
    problems: list[str] = []
    missing = [f for f in FIELDS if not any(x.strip() for x in g.get(f.lower(), []))]
    rep: dict = {"schema": "report.v1", "status": None, "status_note": None, "summary": "",
                 "changed_files": [], "checks": [], "expected": [], "open_problems": [],
                 "cut": cuts}
    st = " ".join(g.get("status", [])).strip().lower()
    word = re.split(r"[\s,.;(]", st, 1)[0] if st else ""
    if word in STATUSES:
        rep["status"] = word
    elif "Status" not in missing:
        problems.append("a Status of done, partial or failed")
    summary = " ".join(x.strip() for x in g.get("summary", []) if x.strip())
    if len(summary) > SUMMARY_MAX:
        problems.append(f"a Summary of at most {SUMMARY_MAX} characters")
        summary = summary[:SUMMARY_MAX].rstrip() + " [cut]"
    rep["summary"] = summary
    changed = [x for x in _items(g.get("changed files", [])) if x.lower() not in _NONE]
    if len(changed) > CHANGED_MAX:
        problems.append(f"at most {CHANGED_MAX} Changed files")
        changed = changed[:CHANGED_MAX]
    long = False
    for x in changed:
        path, _, what = x.partition(": ")
        path = path.strip().strip("`").removeprefix("/workspace/").removeprefix("./")
        if len(what) > CHANGED_LINE_MAX:
            long, what = True, what[:CHANGED_LINE_MAX].rstrip() + " [cut]"
        rep["changed_files"].append({"path": path, "change": what.strip()})
    if long:
        problems.append(f"Changed files lines of at most {CHANGED_LINE_MAX} characters")
    for x in _items(g.get("checks", []))[:ITEMS_MAX]:
        if x.lower() in _NONE:
            continue
        ran, _, result = x.partition(": ")
        rep["checks"].append({"what": ran.strip()[:ITEM_MAX], "result": result.strip()[:ITEM_MAX],
                              "call_id": next(iter(_CALL_REF.findall(x)), None),
                              "shot": next(iter(_SHOT_REF.findall(x)), None)})
    for x in _items(g.get("expected", []))[:ITEMS_MAX]:     # item: answer, or the line as it is
        item, _, answer = x[:ITEM_MAX].partition(": ")
        rep["expected"].append({"item": item.strip(), "answer": answer.strip()} if answer else x)
    rep["open_problems"] = [x[:ITEM_MAX] for x in _items(g.get("open problems", []))[:ITEMS_MAX]
                            if x.lower() not in _NONE]
    return rep, missing + problems


def _items(lines: list[str]) -> list[str]:
    """A field's list: its bullets (a bullet's indented lines with it), and the label line's own
    text ("Open problems: none"); other lines are not the field's."""
    out: list[str] = []
    for i, ln in enumerate(lines):
        m = _BULLET.match(ln)
        if m:
            out.append(m.group(1).strip())
        elif i == 0 and ln.strip():
            out.append(ln.strip())
        elif out and ln[:1] in (" ", "\t") and ln.strip():
            out[-1] += " " + ln.strip()
    return [x for x in out if x]


# ---------------------------------------------------------------------- chatd's check
CHECK_FILES_MAX = 40
CHECK_CALLS_MAX = 12


def _rel(path: str) -> str:
    return str(path or "").strip().strip("`\"'").removeprefix("/workspace/").removeprefix("./")


def _args(p: dict) -> dict:
    try:
        a = json.loads(p.get("input") or "{}")
    except ValueError:
        return {}
    return a if isinstance(a, dict) else {}


def _ran(p: dict) -> str:
    """One call that ran, as the check names it."""
    a, out = _args(p), p.get("output") or {}
    text = render.result_text(p)
    if p.get("name") == "bash":
        code = out.get("exit_code")
        code = code if code is not None else (1 if out.get("is_error") else 0)
        what = f'bash "{str(a.get("command") or "")[:80]}"'
        res = f"exit {code}"
    else:
        target = a.get("path") or (f"port {a['port']}" if a.get("port") else "")
        what = ("verify " if p.get("harness") == "verify" else "") + f"render_page {target}".strip()
        if p.get("status") == "failed" or out.get("is_error") and "console" not in text:
            res = "failed: " + text.strip().splitlines()[0][:120] if text.strip() else "failed"
        elif "no console errors" in text:
            res = "0 console errors"
        else:
            tail = text.split("console", 1)[-1].splitlines()[1:]
            res = f"{sum(1 for x in tail if x.strip() and not x.startswith('    '))} console errors"
        if p.get("shot"):
            res += f", {p['shot']}"
    return f"{what} (call {p['id']}: {res})"


def _claimed(check: dict, calls: list[dict], shots: set) -> bool:
    """Whether a check the report lists matches a call that ran."""
    ids = {p["id"] for p in calls}
    if check.get("call_id"):
        return check["call_id"] in ids
    if check.get("shot"):
        return check["shot"] in shots
    text = f"{check.get('what') or ''}".lower()
    renders = [p for p in calls if p.get("name") == "render_page"]
    if re.search(r"render|screenshot|browser|console", text):
        return bool(renders)
    quoted = re.findall(r'["`]([^"`]+)["`]', check.get("what") or "")
    cmd = (quoted[0] if quoted else re.sub(r"^\s*bash\s+", "", check.get("what") or "")).strip()
    for p in calls:
        if p.get("name") == "bash":
            c = str(_args(p).get("command") or "")
            if cmd and (cmd in c or c and c in cmd):
                return True
        elif p.get("name") and p["name"] in text:
            return True
    return False


def chatd_check(diff: dict, parts: list[dict], report: dict | None) -> list[str]:
    """chatd's own lines below a report: from the task's snapshot diff and the calls that ran,
    never from the worker's words."""
    files = []
    for f in diff.get("files") or []:
        path = _rel(f["path"])
        if f.get("patch", "").startswith("Binary"):
            files.append(f"{path} (binary, {int(f.get('bytes') or 0):,} bytes)")
        elif f["status"] == "added":
            files.append(f"{path} (new, +{f['additions']})")
        elif f["status"] == "deleted":
            files.append(f"{path} (deleted)")
        else:
            files.append(f"{path} (+{f['additions']} -{f['deletions']})")
    more = len(files) - CHECK_FILES_MAX
    changed = ", ".join(files[:CHECK_FILES_MAX]) + (f", and {more} more" if more > 0 else "")
    said = {_rel(c["path"]) for c in (report or {}).get("changed_files") or []}
    left = [_rel(f["path"]) for f in diff.get("files") or [] if _rel(f["path"]) not in said]
    every = [p for p in parts if p.get("type") == "tool_call" and
             p.get("status") in ("done", "failed")]
    ran = [_ran(p) for p in every if p.get("name") in ("render_page", "bash")][-CHECK_CALLS_MAX:]
    shots = {p["shot"] for p in parts if p.get("shot")}
    claims = [c for c in (report or {}).get("checks") or [] if not _claimed(c, every, shots)]
    lines = [f"- Changed in /workspace (snapshot diff): {changed or 'none'}",
             f"- Changed but not in the report: {', '.join(left[:CHECK_FILES_MAX]) or 'none'}",
             f"- Checks that ran: {'; '.join(ran) or 'none'}",
             "- Claimed in Checks with no matching call: " +
             ("; ".join(f'"{c["what"]}' + (f': {c["result"]}"' if c["result"] else '"')
                        for c in claims) or "none")]
    shot_files = [f"{p['shot']} {((p.get('pane') or {}).get('ref') or '')[5:]}" for p in parts
                  if p.get("shot")]
    if shot_files:
        lines.append(f"- Screenshots (open one with read_file): {', '.join(shot_files)}")
    return lines


def build_report(parts: list[dict], diff: dict, reason: str, status: str = "partial",
                 note: str | None = None) -> dict:
    """A report chatd writes from the record when the worker gave none it could take: the files
    from the diff, the checks from the calls that ran."""
    calls = [p for p in parts if p.get("type") == "tool_call" and
             p.get("name") in ("render_page", "bash") and p.get("status") in ("done", "failed")]
    checks = []
    for p in calls[-CHECK_CALLS_MAX:]:
        what, _, res = _ran(p).partition(" (call ")
        checks.append({"what": what, "result": res.rstrip(")").partition(": ")[2],
                       "call_id": p["id"], "shot": p.get("shot")})
    return {"schema": "report.v1", "status": status, "status_note": note,
            "summary": f"report built by chatd: {reason}"[:SUMMARY_MAX],
            "changed_files": [{"path": _rel(f["path"]), "change": f"{f['status']} (snapshot diff)"}
                              for f in (diff.get("files") or [])[:CHANGED_MAX]],
            "checks": checks, "expected": ["not reported by the worker"],
            "open_problems": [f"the worker gave no report chatd could take: {reason}"],
            "cut": [], "built": reason}


def _duration(ms: int) -> str:
    s = max(0, int(ms) // 1000)
    return f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


def _list(items: list[str]) -> list[str]:
    return [f"- {x}" for x in items] or ["- none"]


def render_result(row: dict) -> tuple[str, str]:
    """(what the main agent reads, at most RESULT_MAX_CHARS; the whole, for report.md)."""
    rep = json.loads(row["report"]) if row["report"] else {}
    n, status = row["n"], row["status"]
    built = rep.get("built")
    if status == "cancelled":
        lead = (f"{built}. The report below is chatd's, built from the worker's calls, then "
                f"chatd's check.")
    elif status == "interrupted":
        lead = f"{built}; Continue resumes the worker. What it did so far, then chatd's check."
    elif row["report_source"] == "chatd":
        lead = (f"report built by chatd: {built}. The report below is chatd's, built from the "
                f"worker's calls, then chatd's check.")
    else:
        lead = "the worker's report as chatd parsed it, then chatd's check."
    state = status
    ms = int((row["ended"] or now_ms()) - (row["started"] or now_ms()))
    head = [f"Result of task {n}, given to you as data: {lead}",
            f"task {n} \"{row['title']}\": {state}, {row['steps']} calls, {_duration(ms)} (worker: "
            f"{row['prompt_tokens']:,} prompt tokens, {row['completion_tokens']:,} completion)"]
    fields = [f"Status: {(rep.get('status') or 'partial')}" +
              (f" ({rep['status_note']})" if rep.get("status_note") else ""),
              f"Summary: {rep.get('summary') or ''}", "Changed files:",
              *_list([f"{c['path']}: {c['change']}" if c.get("change") else c["path"]
                      for c in rep.get("changed_files") or []]),
              "Checks:", *_list([f"{c['what']}: {c['result']}" if c.get("result") else c["what"]
                                 for c in rep.get("checks") or []]),
              "Expected:", *_list([f"{e['item']}: {e['answer']}" if isinstance(e, dict) else e
                                   for e in rep.get("expected") or []]),
              "Open problems:", *_list(rep.get("open_problems") or [])]
    check = ["chatd check:", *((row["check_text"] or "- none").splitlines())]
    where = f"/workspace/.tandem/tasks/{n}/report.md"
    tail = [f"Full report: {where}. The transcript is in the Task {n} card, not in your context."]
    whole = "\n".join(head + fields + check + tail)
    if len(whole) <= RESULT_MAX_CHARS:
        return whole, whole
    note = f"[... the report is cut here; the whole of it is in {where}]"
    room = RESULT_MAX_CHARS - len("\n".join(head + check + tail)) - len(note) - 3
    body = "\n".join(fields)
    cut = body[:max(0, room)]
    cut = cut[:cut.rfind("\n")] if "\n" in cut else cut
    out = "\n".join(head + [cut, note] + check + tail)
    return out[:RESULT_MAX_CHARS], whole


def wire(row) -> dict:
    """The contract's `Task` (dashboard/src/chat/CONTRACT.md "Tasks (VIS-E15)")."""
    r = dict(row)
    return {"id": r["id"], "n": r["n"], "title": r["title"], "brief": r["brief"],
            "expect": r["expect"], "status": r["status"], "attempt": r["attempt"],
            "conversation_id": r["conversation_id"],
            "task_conversation_id": r["task_conversation_id"],
            "run_id": r["run_id"], "call_id": r["call_id"],
            "session": render.task_session_id(r["conversation_id"], r["n"]),
            "steps": r["steps"], "max_steps": r["max_steps"],
            "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
            "cached_tokens": r["cached_tokens"],
            "started": iso(r["started"]) if r["started"] else None,
            "ended": iso(r["ended"]) if r["ended"] else None,
            "report": json.loads(r["report"]) if r["report"] else None,
            "report_source": r["report_source"], "check": r["check_text"], "error": r["error"]}


class TasksMixin:
    # ------------------------------------------------------------------ who asks
    def task_row(self, task_id: str | None) -> dict | None:
        r = self.store.one("SELECT * FROM tasks WHERE id = ?", (task_id,)) if task_id else None
        return dict(r) if r is not None else None

    def session_of(self, conv) -> str:
        """The engine's session of a conversation's requests: a worker's own, else the chat's."""
        if conv.parent_id is None:
            return render.session_id(conv.id)
        return render.task_session_id(conv.parent_id, self.task_row(conv.task_id)["n"])

    def headers_of(self, conv) -> dict:
        if conv.parent_id is None:
            return render.headers(conv.id)
        return render.headers(conv.id, session=self.session_of(conv),
                              parent_session=render.session_id(conv.parent_id), worker=True)

    def _head(self, conv, run=None, settings: dict | None = None) -> dict:
        """The system block's own part by who asks: a worker's block, or the main agent's
        paragraph in Agents mode (the run's mode, read at its start; else the chat's setting)."""
        if conv.parent_id is None:
            on = run.agents if run is not None else bool((settings or {}).get("agents"))
            return {"agents": texts()["main"]} if on else {}
        task = run.task if run is not None and run.task else self.task_row(conv.task_id)
        return {"worker": worker_block(int(task["max_steps"]))}

    # ------------------------------------------------------------------ the tool
    def start_task(self, parent, part: dict, args: dict) -> Result:
        """The task call's executor (on the parent's tool thread): a worker on a new task
        conversation, in the parent's slot; its result when it ends."""
        settings = self.settings_of(parent.conv.id)
        if parent.parent is not None:
            return Result("task: a worker cannot give tasks", is_error=True)
        row = self.store.one("SELECT * FROM tasks WHERE conversation_id = ? AND call_id = ?",
                             (parent.conv.id, part["id"]))
        if row is not None:
            # Continue reached this call again (SRV-585): an interrupted worker goes on in its own
            # message; one that ended meanwhile (its report stored late) is not run again
            row = dict(row)
            if row["status"] not in ("running", "interrupted"):
                return self._task_answer(row)
            return self._resume_task(parent, row)
        refused = check_brief(args)
        if refused:
            return Result(refused, is_error=True)
        limit = int(settings.get("task_max_per_run") or 8)
        if parent.tasks_given >= limit:
            return Result(f"task refused: this run has given {limit} tasks, its limit. Answer the "
                          f"user with what the reports say, or ask whether to go on.",
                          is_error=True)
        title, brief, expect = (str(args.get(k) or "").strip() for k in ("title", "brief",
                                                                         "expect"))
        parent.tasks_given += 1
        row = self._new_task(parent, part, title, brief, expect, settings)
        return self._run_task(parent, row)

    def _new_task(self, parent, part: dict, title: str, brief: str, expect: str,
                  settings: dict) -> dict:
        from chatd.service import EMPTY_UI
        conv = parent.conv
        root = conv.id
        with conv.lock:
            n = 1 + int(self.store.one("SELECT MAX(n) FROM tasks WHERE conversation_id = ?",
                                       (root,))[0] or 0)
            tid, tcid, t = ulid(), ulid(), now_ms()
            steps = int(settings.get("task_max_steps") or 40)
            self.store.run("INSERT INTO conversations (id, title, title_auto, created, updated, "
                           "model, settings, ui, kind, parent_id) VALUES (?, ?, 0, ?, ?, ?, '{}', "
                           "?, 'task', ?)", (tcid, f"Task {n}: {title}"[:200], t, t,
                                             settings.get("model"), dumps(EMPTY_UI), root))
            self.store.run("INSERT INTO tasks (id, conversation_id, run_id, call_id, n, title, "
                           "brief, expect, task_conversation_id, status, max_steps, started) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)",
                           (tid, root, parent.id, part["id"], n, title, brief, expect, tcid, steps,
                            t))
            um = self.store.one("SELECT parts, attachments FROM messages WHERE id = ?",
                                (parent.user_message_id,))
            request = request_text(json.loads(um["parts"]) if um else [],
                                   json.loads(um["attachments"]) if um and um["attachments"] else
                                   None)
            text = first_message(n, title, brief, expect, self.todos(root), request)
            self._insert_message(tcid, None, "user", [{"type": "text", "text": text}], "done")
        try:                                     # the task's diff starts here (SRV-581's check)
            self.snaps.take(root, self.workspace(root), tid, "before")
        except OSError:
            pass
        row = self.task_row(tid)
        self.emit(conv, {"type": "task.started", "run_id": parent.id, "task": wire(row)},
                  run_id=parent.id)
        self.log(f"[task] {tid} {n} started conv={root} run={parent.id}")
        return row

    def _run_task(self, parent, row: dict, continue_mid: str | None = None) -> Result:
        tconv = self.conv(row["task_conversation_id"])
        user = self.store.one("SELECT id FROM messages WHERE conversation_id = ? AND role = 'user' "
                              "ORDER BY id LIMIT 1", (tconv.id,))["id"]
        with tconv.lock:
            worker = self._start_run(tconv, user, parent.started_by, parent=parent, task=row,
                                     continue_mid=continue_mid)
        parent.child = worker
        try:
            while worker.thread.is_alive() and not parent.cancel.is_set():
                worker.thread.join(1.0)
            if worker.thread.is_alive():            # Stop reached it through the parent
                worker.thread.join(STOP_WAIT_S)
        finally:
            parent.child = None
        row = self.task_row(row["id"])
        if row["status"] == "running":              # still storing its end: Continue delivers it
            return Result(f"task {row['n']} was stopped and has not stored its report yet; "
                          f"Continue delivers it.", is_error=True, status="interrupted")
        parent.workers.append({"task_id": row["id"], "prompt_tokens": row["prompt_tokens"],
                               "completion_tokens": row["completion_tokens"],
                               "cached_tokens": row["cached_tokens"],
                               "ms": int((row["ended"] or now_ms()) - row["started"])})
        return self._task_answer(row)

    def _task_answer(self, row: dict) -> Result:
        """The task call's result: an interrupted task's leaves the call `interrupted` (its run
        ends interrupted, `stats.interrupted` task, and Continue resumes the worker)."""
        return Result(self.task_result(row), is_error=row["status"] == "failed",
                      status="interrupted" if row["status"] == "interrupted" else None)

    def _resume_task(self, parent, row: dict) -> Result:
        tcid = row["task_conversation_id"]
        on = self.conv(tcid).run
        if on is not None and on.active:            # the stopped worker still ends: its answer
            on.thread.join(STOP_WAIT_S)
            row = self.task_row(row["id"])
            if row["status"] != "running":
                return self._task_answer(row)
            return Result(f"task {row['n']} has not stored its report yet; Continue delivers it.",
                          is_error=True, status="interrupted")
        last = self.store.one("SELECT id FROM messages WHERE conversation_id = ? AND role = "
                              "'assistant' ORDER BY id DESC LIMIT 1", (tcid,))
        self.store.run("UPDATE tasks SET status = 'running', attempt = attempt + 1, ended = NULL "
                       "WHERE id = ?", (row["id"],))
        row = self.task_row(row["id"])
        self.emit(parent.conv, {"type": "task.started", "run_id": parent.id, "task": wire(row)},
                  run_id=parent.id)
        self.log(f"[task] {row['id']} {row['n']} resumed (attempt {row['attempt']})")
        return self._run_task(parent, row, continue_mid=last["id"] if last else None)

    def task_result(self, row: dict) -> str:
        """What the main agent reads (`render_result`)."""
        return render_result(row)[0]

    # ------------------------------------------------------------------ the report
    def salvage_text(self, worker) -> str:
        return texts()["salvage"].replace("{max_steps}", str(worker.max_steps))

    def task_report(self, worker, answered: bool = True) -> bool:
        """At a worker's last step (chatd/runs.py): its report taken, or sent back once (True:
        one more call, the send-back at its tail). At the cap (`salvage`) the answer is taken as
        it is, and chatd builds one when it is none."""
        with worker.conv.lock:
            parts = list((worker.conv.live or {}).get("parts") or [])
        rep, problems = parse_report(final_text(parts, worker.report_misses) if answered else "")
        missing = [f for f in FIELDS if f in problems]
        if worker.salvage:
            if rep is None or missing:
                rep, source = None, "chatd"         # built at the end, from the record
                worker.report_reason = "the worker's model calls ran out before its report"
            else:
                source = "worker"
            if rep is not None:
                rep["status"], rep["status_note"] = "partial", "step limit"
        elif not problems:
            source = "worker-second" if worker.report_misses else "worker"
        elif worker.report_misses == 0:
            worker.report_misses = worker.step       # the step of the report sent back
            worker.tail = texts()["send-back"].replace("{fields}", ", ".join(problems))
            return True
        elif not missing:
            source = "worker-second"                # within its fields; the caps cut it
        else:
            rep, source = None, "chatd"
            worker.report_reason = f"the worker's report missed {', '.join(missing)} twice"
        worker.report, worker.report_source = rep, source
        return False

    # ------------------------------------------------------------------ the worker's end
    def task_ended(self, worker, outcome: str, stats: dict) -> None:
        """From finish_run, under the task conversation's lock: the tasks row, the task's
        snapshot after, `task.done` on the chat."""
        conv, task = worker.conv, worker.task
        root = conv.parent_id
        status = {"completed": "done", "cancelled": "cancelled", "interrupted": "interrupted",
                  "failed": "failed"}[outcome]
        try:
            self.snaps.take(root, self.workspace(root), task["id"], "after")
        except OSError:
            pass
        m = self._messages(conv.id).get(worker.message_id)
        parts = m["parts"] if m else []
        diff = self.snaps.diff(root, task["id"])
        rep, source = worker.report, worker.report_source
        if status == "done":
            if rep is None:                         # ended without a report it gave or chatd took
                reason = getattr(worker, "report_reason", None) or worker.notice or \
                    "the worker ended without a report"
                note = "step limit" if worker.salvage else None
                rep, source = build_report(parts, diff, reason, "partial", note), "chatd"
            status = rep["status"] + (f" ({rep['status_note']})" if rep.get("status_note")
                                      else "")
            for c in rep.get("checks") or []:      # a screenshot id names its call
                if c.get("shot") and not c.get("call_id"):
                    c["call_id"] = next((p["id"] for p in parts if p.get("shot") == c["shot"]),
                                        None)
        elif status == "cancelled":
            rep, source = build_report(parts, diff, f"cancelled by the user at call "
                                                    f"{worker.step}"), "chatd"
        elif status == "failed":
            rep, source = build_report(parts, diff, f"the worker failed: "
                                                    f"{worker.error or 'no reason given'}",
                                       "failed"), "chatd"
        else:
            why = {"sandbox": "the sandbox failed three times in a row"}.get(
                worker.interrupt or "", "the engine did not answer")
            rep, source = build_report(parts, diff, f"interrupted at call {worker.step} ({why})"), \
                "chatd"
        check = "\n".join(chatd_check(diff, parts, rep))
        fig = {"prompt_tokens": sum(int(u.get("prompt_tokens") or 0) for u in worker.step_usage),
               "cached_tokens": sum(int(u.get("cached_tokens") or 0) for u in worker.step_usage),
               "completion_tokens": int(stats.get("completion_tokens") or 0)}
        self._end_task(worker, status, rep, source, check, fig)

    def _end_task(self, worker, status: str, report: dict | None, source: str | None,
                  check: str | None, fig: dict) -> None:
        task = worker.task
        t = now_ms()
        # a worker resumed by Continue adds its calls to the ones before
        self.store.run("UPDATE tasks SET status = ?, steps = ?, prompt_tokens = prompt_tokens + ?, "
                       "completion_tokens = completion_tokens + ?, cached_tokens = cached_tokens "
                       "+ ?, ended = ?, report = ?, report_source = ?, check_text = ?, error = ? "
                       "WHERE id = ?",
                       (status, worker.step, fig["prompt_tokens"], fig["completion_tokens"],
                        fig["cached_tokens"], t, dumps(report) if report is not None else None,
                        source, check, worker.error, task["id"]))
        row = self.task_row(task["id"])
        parent = self.conv(worker.conv.parent_id)
        full = render_result(row)[1]
        try:                                       # the whole report, beside the cut one
            self.workspace(parent.id).write(f".tandem/tasks/{row['n']}/report.md",
                                            (full + "\n").encode())
        except (OSError, PathError):
            pass
        self.emit(parent, {"type": "task.done", "task_id": task["id"], "status": status,
                           "report": report, "check": check,
                           "stats": {"steps": row["steps"], "max_steps": row["max_steps"],
                                     "prompt_tokens": row["prompt_tokens"],
                                     "completion_tokens": row["completion_tokens"],
                                     "cached_tokens": row["cached_tokens"],
                                     "ms": int(t - row["started"]),
                                     "report_source": source}},
                  run_id=worker.parent.id if worker.parent else None)

    # ------------------------------------------------------------------ live figures
    def task_phase(self, worker) -> None:
        """A worker's phase: `task.phase` on its chat, and the parent's Now line."""
        p, t = worker.parent, worker.task
        ev = {"type": "task.phase", "task_id": t["id"], "phase": worker.phase, "now": worker.now,
              "step": worker.step, "max_steps": worker.max_steps}
        if worker.tool:
            ev["tool"] = worker.tool
        self.emit(p.conv, ev, run_id=p.id)
        if p.active:
            p.set_phase("task", f"Task {t['n']}: {t['title']}, call {worker.step} of "
                                f"{worker.max_steps}: {worker.now}")

    def worker_metrics(self, worker) -> dict:
        """A worker's live figures (`task.metrics`, `context.workers[].metrics`), from its own
        live row (its session chat-<chat>-task-<n>, service.py `_metrics`) and its usage frames:
        the prefill's progress while it reads its prompt, else the decode; its tokens so far."""
        pre = worker.prefill
        m = worker.metrics
        return {"task_id": worker.task["id"], "phase": worker.phase,
                "prefill": {k: pre.get(k) for k in ("done", "total", "tps", "eta_ms", "cached")}
                if pre else None,
                "decode_tps": m["decode_tps"], "ttft_ms": m["ttft_ms"],
                "tokens_per_round": m["tokens_per_round"], "acceptance": m["acceptance"],
                "prompt_tokens": sum(int(u.get("prompt_tokens") or 0) for u in worker.step_usage),
                "completion_tokens": worker.tokens, "thinking_used": worker.thinking_tokens}

    def task_metrics(self, worker) -> None:
        """Once a second from the ticker (service.py `_metrics`): `task.metrics` on the chat."""
        self.emit(worker.parent.conv, {"type": "task.metrics", **self.worker_metrics(worker)},
                  persist=False)

    def worker_split(self, worker, body: dict, path: list[dict]) -> None:
        """At each of a worker's requests: the parts of its context, estimated from the
        characters at its learned ratio (the engine counts only the whole prompt's)."""
        from chatd.agent import estimated_split
        est = estimated_split(body, worker.conv.tok_ratio, None)
        user = path[0] if path and path[0].get("role") == "user" else {}
        worker.ctx_est = {"system": est["system"], "tools": est["tools"],
                          "brief": ctx_mod.estimate_tokens(render.text_of(user.get("parts") or [])),
                          "files": ctx_mod.file_tokens(path)}

    def running_workers(self, run) -> list:
        """A run's workers in flight, by task: one at a time today (SRV-589 runs more)."""
        w = getattr(run, "child", None) if run is not None else None
        return [w] if w is not None and w.active else []

    def worker_context(self, w) -> dict:
        """One item of the Context panel's `workers`: a running worker's own window, its Now line
        and its live figures."""
        t = w.task
        last = w.step_usage[-1] if w.step_usage else {}
        used = int(last.get("prompt_tokens") or 0) + int(last.get("completion_tokens") or 0) \
            or int(w.request_est or 0)
        est = getattr(w, "ctx_est", None) or {"system": 0, "tools": 0, "brief": 0, "files": 0}
        prompt = int(last.get("prompt_tokens") or 0)
        cached = int(last.get("cached_tokens") or 0)
        settings = self.settings_of(w.conv.id)
        return {"task_id": t["id"], "n": t["n"], "title": t["title"], "used": used,
                "window": int(self.model_info(settings.get("model")).get("max_len") or 0),
                "split": {**est, "steps": max(0, used - sum(est.values()))},
                "step": w.step, "max_steps": w.max_steps,
                "cache": {"warm": cached, "cold": max(0, prompt - cached),
                          "source": last.get("cache_source") or "none"},
                "phase": w.phase, "now": w.now, "metrics": self.worker_metrics(w)}

    # ------------------------------------------------------------------ Agents mode (SRV-583)
    def _read_hit(self, run, part: dict, name: str, args: dict) -> Result | None:
        """A read the main agent made before with the same arguments, whose result the next
        request still carries (after the newest compaction), with no worker run and no file
        changed since: "unchanged since call <id> (step <k>)". None: the read runs."""
        key = json.dumps(args, sort_keys=True)
        conv = run.conv
        with conv.lock:
            msgs = self._messages(conv.id)
            path = [msgs[m] for m in self._path(msgs, run.user_message_id)]
            path.append({"id": run.message_id, "role": "assistant",
                         "parts": [p for p in (conv.live or {}).get("parts") or [] if p is not part]})
        c = self._compaction_for(conv.id, path)
        start = c[1] if c else 0
        for i in range(len(path) - 1, start - 1, -1):
            m = path[i]
            if m.get("role") != "assistant":
                continue
            for p in reversed(m.get("parts") or []):
                if p.get("type") != "tool_call":
                    continue
                if c and c[2] and i == c[2][0] and int(p.get("step") or 1) < c[2][1]:
                    return None                      # compacted away
                if p.get("name") == "task":
                    return None                      # a worker ran since
                if p.get("name") != name or p.get("cached_from") or p.get("status") != "done":
                    continue
                try:
                    same = json.dumps(json.loads(p.get("input") or "{}"), sort_keys=True) == key
                except ValueError:
                    same = False
                if not same:
                    continue
                if p.get("digest") != part.get("digest"):
                    return None                      # a file changed since
                part["cached_from"] = p["id"]
                return Result(f"unchanged since call {p['id']} (step {p.get('step') or 1})")
        return None

    def reread_tokens(self, cid: str) -> int:
        """The prompt tokens of the chat's last request: what a switch of Agents mode makes the
        engine read once more (the prompt's head changes)."""
        row = self._row(cid)
        msgs = self._messages(cid)
        for mid in reversed(self._path(msgs, row["head_id"])):
            s = msgs[mid].get("stats") or {}
            if msgs[mid]["role"] == "assistant" and s:
                last = (s.get("step_usage") or [{}])[-1]
                return int(last.get("prompt_tokens") or s.get("prompt_tokens") or 0)
        return 0

    def last_worker(self, cid: str) -> dict | None:
        """The Context panel's `last_worker`: the chat's newest task, its context at its last
        call (null before any)."""
        r = self.store.one("SELECT * FROM tasks WHERE conversation_id = ? ORDER BY n DESC LIMIT 1",
                           (cid,))
        if r is None:
            return None
        m = self.store.one("SELECT stats FROM messages WHERE conversation_id = ? AND role = "
                           "'assistant' ORDER BY id DESC LIMIT 1", (r["task_conversation_id"],))
        last = ((json.loads(m["stats"]) if m and m["stats"] else {}).get("step_usage") or [{}])[-1]
        return {"task_id": r["id"], "n": r["n"], "title": r["title"], "status": r["status"],
                "used": int(last.get("prompt_tokens") or 0) + int(last.get("completion_tokens")
                                                                  or 0)}

    # ------------------------------------------------------------------ routes
    def list_tasks(self, cid: str) -> dict:
        """GET /conversations/{id}/tasks: the chat's tasks, by number."""
        self._row(cid)
        return {"tasks": [wire(r) for r in self.store.all(
            "SELECT * FROM tasks WHERE conversation_id = ? ORDER BY n", (cid,))]}

    def get_task(self, cid: str, task_id: str) -> dict:
        """GET /conversations/{id}/tasks/{task_id}: the task with its transcript (calls with
        their arguments and results), the running one's from memory, and the chat's `seq` it
        includes: the chat's later events with this `task_id` apply on top."""
        from chatd.service import ApiError
        conv = self.conv(cid)
        r = self.store.one("SELECT * FROM tasks WHERE id = ? AND conversation_id = ?",
                           (task_id, cid))
        if r is None:
            raise ApiError(404, "not_found", "no such task in this chat")
        tconv = self.conv(r["task_conversation_id"])
        # under the task conversation's lock only: its changes and their events on the chat's log
        # happen under it (the lock order is task, then chat), so the transcript read here holds
        # every event up to `seq` and none after
        with tconv.lock:
            seq = conv.last_seq
            row = self._row(tconv.id)
            branch, msgs = self._branch(tconv, row["head_id"])
            kids = self._children(msgs)
            return {"task": wire(self.task_row(task_id)),
                    "messages": [self._wire_message(m, kids) for m in branch], "seq": str(seq)}
