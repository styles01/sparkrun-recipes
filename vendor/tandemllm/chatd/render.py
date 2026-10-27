"""The request of one model call, the same bytes for the same conversation (D5, SRV-345).

The engine finds a conversation's cache by the longest shared token prefix, so a request whose head
changes re-reads the whole conversation (the lesson of SRV-209: Open WebUI's working directory in
the system prompt did that to every turn). So the renderer puts nothing in the head that changes
between turns -- no time, no ids, no counts -- builds every object with its keys in one fixed
order, and writes compact JSON. What is new in a turn (the next user message, a steer from the
queue) only ever appends. `tests/test_chatd_render.py` pins it: the same conversation renders to the
same bytes across calls and across a restart (a fresh store read).

    messages        the branch, root to head: user text (text attachments as a fenced block after
                    it, one line with its workspace path for one over the budget, chatd/attach.py,
                    one line saying why for one not saved under uploads/;
                    images as `image_url` parts when the model reads images), assistant text with
                    its reasoning as `reasoning_content` (the engine's template decides what to keep);
                    notices (role `system` in the tree) are the UI's and are left out
    tool images     in the tool message that returned them, append-only: all up to
                    TOOL_IMAGES_KEPT, then the first ones and the newest; the newest-but-one
                    becomes a line saying a newer one follows (SRV-546)
    file texts      a write_file's content and an edit_file's long old and new text whole while
                    the run's whole texts stay under FILES_HIGH_CHARS; past it (or at a
                    compaction) the oldest leave the call and their tool result says so, the
                    newest FILES_KEEP_CHARS stay (`folded_calls`); a long read_file or bash result
                    whole until a compaction, a stub after it (FOLD_ARG_CHARS, FOLD_RESULT_CHARS)
    effort          `reasoning_effort` only when the chat picked one (`auto` lets the harness decide)
    thinking off    `chat_template_kwargs.enable_thinking: false` (read by our engine, vLLM and
                    SGLang alike)
    continue        an interrupted answer resumes as the last assistant message with
                    `continue_final_message: true` and no generation prompt (SRV-283's Continue)
"""

from __future__ import annotations

import base64
import hashlib
import json
import re

from chatd import attach

SESSION_PREFIX = "chat-"


def session_id(conversation_id: str) -> str:
    """`x-session-id`: the router's session key becomes `c:chat-<id>` (server/conductor/router.py),
    the park, the spill and the live row's conversation id follow it. A branch shares its prefix up
    to the edited message, so one key is right for all of a chat's branches."""
    return SESSION_PREFIX + conversation_id


def task_session_id(parent_id: str, n: int) -> str:
    """A worker's `x-session-id` (VIS-E15): its own key beside the chat's, so the router, the
    prefix cache, the park and the live row keep the chat and its task n apart."""
    return f"{SESSION_PREFIX}{parent_id}-task-{n}"


def text_of(parts: list[dict], kind: str = "text") -> str:
    return "".join(p.get("text") or "" for p in parts if p.get("type") == kind)


def inlined(a: dict) -> bool:
    """A text attachment the prompt carries whole: as placed when its message was sent (SRV-487);
    one from before that, while it stays within INLINE_MAX_TOKENS."""
    if a.get("inline") is not None:
        return bool(a["inline"])
    return int(a.get("tokens") or 0) <= attach.INLINE_MAX_TOKENS


def _user(m: dict, vision: bool, blob=None) -> dict:
    parts = m.get("parts") or []
    text = text_of(parts)
    todos = text_of(parts, "todos")                 # the open todo list at the run's start
    if todos:
        text += "\n\n" + todos
    plan = text_of(parts, "plan")                   # plan mode's rule, on its turn (SRV-663)
    if plan:
        text += "\n\n" + plan
    images = []
    for a in m.get("attachments") or []:
        data = blob(a) if blob is not None else None
        if data is None:
            continue
        if a.get("kind") == "image":
            if vision:
                images.append({"type": "image_url", "image_url": {
                    "url": f"data:{a.get('mime') or 'image/png'};base64,"
                           + base64.b64encode(data).decode()}})
        elif a.get("kind") == "text":
            body = data.decode("utf-8", "replace")
            if not inlined(a):
                text += attach.line(a)
                continue
            text += f"\n\n```{a.get('name') or 'file'}\n{body}\n```"
        if a.get("unplaced"):
            text += attach.unplaced_line(a)
    if images:
        return {"role": "user", "content": [{"type": "text", "text": text}, *images]}
    return {"role": "user", "content": text}


def steps(parts: list[dict]) -> list[list[dict]]:
    """An assistant message's parts by model call (`step`); parts from before phase 2 are one."""
    out: dict[int, list[dict]] = {}
    for p in parts:
        out.setdefault(int(p.get("step") or 1), []).append(p)
    return [out[k] for k in sorted(out)]


#: the tool images a request carries as images: every one up to TOOL_IMAGES_KEPT, past it the
#: first TOOL_IMAGES_KEPT - 1 and the newest; the others become a line. An image once sent stays
#: where it is: with only the newest kept (2026-10-13), each new screenshot turned the previous one
#: into a line, the prompt changed where it sat, and the engine read everything after it again at
#: every screenshot (SRV-546: 90,186 tokens in 161 s at 10:18:36 on 2026-10-17). Past the cap the
#: line replaces the newest-but-one, at the prompt's tail
TOOL_IMAGES_KEPT = 8
LEFT_OUT = "\n[The image of this result is left out: a newer screenshot follows.]"


def _image_part(a: dict, data: bytes) -> dict:
    return {"type": "image_url", "image_url": {
        "url": f"data:{a.get('mime') or 'image/png'};base64," + base64.b64encode(data).decode()}}


def kept_tool_images(branch: list[dict], n: int | None = None) -> set:
    """The ids of the images tool results returned on `branch` that a request carries: all of
    them up to `n` (TOOL_IMAGES_KEPT), else the first `n - 1` and the newest, so a kept image never
    becomes a line later and the prompt before the newest-but-one never changes."""
    n = TOOL_IMAGES_KEPT if n is None else n
    ids = [a.get("id") for m in branch if m.get("role") == "assistant"
           for p in m.get("parts") or [] if p.get("type") == "tool_call"
           for a in (p.get("output") or {}).get("images") or []]
    if n <= 0:
        return set()
    if len(ids) <= n:
        return set(ids)
    return set(ids[:n - 1]) | {ids[-1]}


#: the file texts a request leaves out: the files are on disk, so the request can keep what was
#: done (the path, the size, a hash) and not the text. Live (2026-10-17, chat
#: 01M556JVWDQVDCAMBMVFZCT7XF, a WebGPU ocean) a run's request grew from 25k to 162k tokens in 50
#: minutes of write_file, edit_file and read_file calls, and its compaction kept 138,673 tokens of
#: shader and JS texts.
#: A read_file or bash result over FOLD_RESULT_CHARS stays whole until a compaction (changed in the
#: middle of the request, it would make the engine read everything after it again); the request
#: after a compaction is read anew anyway, and every result that ended before it is a stub there.
#:
#: SRV-659: a write_file's content and an edit_file's old and new text over FOLD_ARG_CHARS stay
#: whole, and leave the request only at a fold point (`folded_calls`): when the whole texts of the
#: rendered calls pass FILES_HIGH_CHARS, the oldest leave until FILES_KEEP_CHARS remain, and at a
#: compaction the calls before it are cut down to FILES_KEEP_CHARS the same way. Between two fold
#: points the request only appends, so the engine's prefix cache and its end anchor hold it. A
#: folded call keeps its path and loses the text argument; its tool result says the text was
#: written as sent and how to read the file. Until 0.13 the chat's calls showed a stub in place of
#: the content from the next request on (live 2026-10-20, chat 01M5DEZ7BKRZJH3T2JJ2AG4H2W): the
#: model read its own call as a description, wrote effects.js three times (correct on disk all
#: along), then copied the stub as the content of new files, and the stub at the request's tail
#: kept the engine's end alignment from serving the turn (36,316 tokens read again)
FOLD_ARG_CHARS = 2000
FOLD_RESULT_CHARS = 4000
FOLD_RESULT_TOOLS = ("read_file", "bash")
FILES_HIGH_CHARS = 160_000
FILES_KEEP_CHARS = 48_000
#: a small window's marks: the whole texts at most this share of the window (4 characters a token,
#: fixed, so the marks do not move with the learned ratio), kept down to KEEP_SHARE of the high one
FILES_WINDOW_SHARE = 0.15
FILES_KEEP_SHARE = 0.3
BASH_HEAD_CHARS, BASH_TAIL_CHARS = 300, 600
_READ_LINE = re.compile(r"^\s*(\d+)\t", re.M)
_SAVED = re.compile(r"the whole output is in (\S+) \.\.\.\]")


def files_marks(window: int | None) -> tuple[int, int]:
    """(high, keep) character marks of `folded_calls` for a model of `window` tokens."""
    high = FILES_HIGH_CHARS
    if window:
        high = min(high, int(window * 4 * FILES_WINDOW_SHARE))
    return high, min(FILES_KEEP_CHARS, int(high * FILES_KEEP_SHARE))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _lines(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def _where(path: str) -> str:
    return path if path.startswith("/") else "/workspace/" + path.removeprefix("./")


def _read_call(path: str, **span) -> str:
    """The call that reads a file again, exactly as the model sends it."""
    return "read_file " + json.dumps({"path": _where(path), **span}, ensure_ascii=False)


def wrote_stub(path: str, text: str) -> str:
    """The stub a request showed in place of a write's content until 0.13 (`WROTE_STUB`: the
    write guard still knows it, a model may copy it from an older conversation)."""
    return (f"[wrote {_where(path)}: {_lines(text):,} lines, {len(text.encode()):,} bytes, "
            f"sha256 {_sha(text)}; the file is on disk, {_read_call(path)} shows it]")


#: a write_file whose content is a stub of an earlier one (the model copying what it read); any
#: path, brackets and spaces too
WROTE_STUB = re.compile(r"\[wrote .+: [\d,]+ lines, [\d,]+ bytes, sha256 [0-9a-f]{8}; the file "
                        r"is on disk, read_file \{.*\} shows it\]", re.S)
#: a write_file whose content is the note a folded call's result carries (`folded_note`)
FOLDED_NOTE = re.compile(r"\[(the content|the old and new text) of this call \(", re.S)
#: the text arguments a fold point may take out of a call: a write's content, an edit's texts, and a
#: command over FOLD_ARG_CHARS (a heredoc that writes a file: the owner's reef run wrote its first
#: page in six bash heredocs of 3-6 KB each)
_FOLD_KEYS = {"write_file": ("content",), "edit_file": ("old", "new"), "bash": ("command",)}


def _fold_texts(c: dict) -> dict | None:
    """The text arguments of a finished write_file, edit_file or bash call that a fold point may
    take out of the request (each over FOLD_ARG_CHARS), by key; None for any other call."""
    keys = _FOLD_KEYS.get(c.get("name"))
    raw = c.get("input") or ""
    ran = c.get("status") == "done" or c.get("name") == "bash" and c.get("status") == "failed"
    if not keys or not ran or len(raw) <= FOLD_ARG_CHARS:
        return None
    try:
        a = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(a, dict):
        return None
    out = {k: a[k] for k in keys if isinstance(a.get(k), str) and len(a[k]) > FOLD_ARG_CHARS}
    return out or None


def all_folded(branch: list[dict]) -> set:
    """Every write and edit call whose texts a fold may take out: a compaction's summary request
    leaves them all out (the summary says what was done, the files are on disk)."""
    return {p.get("id") for m in branch if m.get("role") == "assistant"
            for p in m.get("parts") or [] if p.get("type") == "tool_call" and _fold_texts(p)}


def folded_calls(branch: list[dict], fold_before: str | None = None,
                 step_from: tuple[int, int] | None = None, window: int | None = None) -> set:
    """The ids of the write_file, edit_file and bash calls whose text arguments the request leaves
    out.

    The calls are walked in order, each with the length of its long texts (`_fold_texts`); while
    the whole ones add up to FILES_HIGH_CHARS or less nothing leaves, past it the oldest leave
    until FILES_KEEP_CHARS remain. A compaction (`fold_before`, ISO 8601) is a fold point too:
    when the walk passes its time, the calls that ended before it are cut down to
    FILES_KEEP_CHARS. The set changes only at a fold point, so between two of them the request
    only appends (the engine's prefix cache holds it), and it is a function of the branch alone:
    the same conversation folds the same calls on every render. `step_from`: (message index,
    step) of a long run's in-run compaction, whose earlier steps the request does not carry;
    `window`: the model's, a small one lowers the marks (`files_marks`)."""
    high, keep = files_marks(window)
    whole: list[tuple[str, int]] = []
    total, out, passed = 0, set(), fold_before is None

    def cut(limit: int) -> None:
        nonlocal total
        while whole and total > limit:
            cid, n = whole.pop(0)
            out.add(cid)
            total -= n
    for i, m in enumerate(branch):
        if m.get("role") != "assistant":
            continue
        for p in m.get("parts") or []:
            if p.get("type") != "tool_call":
                continue
            if step_from and i == step_from[0] and int(p.get("step") or 1) < step_from[1]:
                continue
            if not passed and (p.get("ended_at") or "") >= fold_before:
                passed = True
                cut(keep)
            texts = _fold_texts(p)
            if texts is None:
                continue
            whole.append((p.get("id"), sum(len(t) for t in texts.values())))
            total += whole[-1][1]
            if total > high:
                cut(keep)
    if not passed:
        cut(keep)
    return out


def call_arguments(c: dict, folded: bool = False) -> str:
    """A call's arguments as the model reads them: as sent, or for a folded call (`folded_calls`)
    without its long text arguments (the path stays; `folded_note` goes in its result)."""
    raw = c.get("input") or "{}"
    texts = _fold_texts(c) if folded else None
    if not texts:
        return raw
    a = json.loads(raw)
    return json.dumps({k: v for k, v in a.items() if k not in texts}, ensure_ascii=False)


def folded_note(c: dict) -> str:
    """The line a folded call's tool result ends with: what left the request, that it was
    applied as sent, and the call that shows the file now."""
    texts = _fold_texts(c) or {}
    try:
        path = str(json.loads(c.get("input") or "{}").get("path") or "")
    except (ValueError, AttributeError):
        path = ""
    if c.get("name") == "write_file":
        t = texts.get("content", "")
        what = (f"the content of this call ({_lines(t):,} lines, {len(t.encode()):,} bytes, "
                f"sha256 {_sha(t)}) is left out of this conversation to save room; the file was "
                f"written with it as sent")
    elif c.get("name") == "bash":
        t = texts.get("command", "")
        return (f"\n[the command of this call ({_lines(t):,} lines, {len(t.encode()):,} bytes) is "
                f"left out of this conversation to save room; it ran as sent, its output is above]")
    else:
        sizes = " and ".join(f"{len(texts[k].encode()):,}" for k in ("old", "new") if k in texts)
        what = (f"the old and new text of this call ({sizes} bytes) are left out of this "
                f"conversation to save room; the edit was applied as sent")
    return f"\n[{what}. {_read_call(path)} shows the file as it is now]"


def result_for(c: dict, fold_before: str | None = None, folded: bool = False) -> str:
    """What the model reads for one call (`result_text`); a read_file or bash result over
    FOLD_RESULT_CHARS of a call that ended before `fold_before` (the newest compaction's time,
    ISO 8601) as a stub; a folded write, edit or command (`folded_calls`) with its
    `folded_note`."""
    note = folded_note(c) if folded and _fold_texts(c) else ""
    return _result(c, fold_before) + note


def _result(c: dict, fold_before: str | None) -> str:
    text = result_text(c)
    if fold_before is None or c.get("name") not in FOLD_RESULT_TOOLS or \
            len(text) <= FOLD_RESULT_CHARS or (c.get("ended_at") or "") >= fold_before:
        return text
    try:
        a = json.loads(c.get("input") or "{}")
    except ValueError:
        a = {}
    a = a if isinstance(a, dict) else {}
    size = len(text.encode())
    if c["name"] == "read_file":
        # a file is read again as it is: the stub names the call
        nums = _READ_LINE.findall(text)
        span = {"offset": int(nums[0]), "limit": int(nums[-1]) - int(nums[0]) + 1} if nums else {}
        lines = f" lines {nums[0]}-{nums[-1]}" if nums else ""
        return (f"[read {_where(str(a.get('path') or ''))}{lines}: {size:,} bytes, left out at a "
                f"compaction; {_read_call(str(a.get('path') or ''), **span)} reads it again]")
    # a command's output is not read again faithfully (a second run may differ or change
    # things): its first and last lines stay, and the stub says where the whole is, if anywhere
    code = (c.get("output") or {}).get("exit_code")
    head, tail = text[:BASH_HEAD_CHARS], text[-BASH_TAIL_CHARS:]
    head = head[:head.rfind("\n")] if "\n" in head else head
    tail = tail[tail.find("\n") + 1:] if "\n" in tail[:-1] else tail
    saved = _SAVED.search(text)
    where = (f"the whole output is in {saved.group(1)}" if saved else
             "the output is not stored, run the command again only if needed")
    cut = size - len(head.encode()) - len(tail.encode())
    return (f"{head}\n[bash: {size:,} bytes of output" +
            (f", exit code {code}" if code is not None else "") +
            f"; {cut:,} bytes cut here at a compaction; {where}]\n{tail}")


def _assistant(m: dict, vision: bool, blob=None, keep_reasoning: bool = True,
               keep: set | None = None, inline: bool = True, from_step: int = 0,
               fold_before: str | None = None, folded: set | None = None) -> list[dict]:
    """One assistant message as the model saw it: per step its text, reasoning and tool calls,
    then one `tool` message per call with the result the model read. The images of those results
    (vision models only, the ones in `keep`; None: all) go in the tool message itself with
    `inline` (our engine's template renders them in the tool response, so the conversation keeps
    one user turn: a separate user message per screenshot read to the model as the user sending
    again, live 2026-10-13), else in a user message after the results. `from_step`: the steps
    before it were compacted (the summary is in the system block); `fold_before`: the newest
    compaction's time (`result_for`); `folded`: the ids of the write and edit calls whose
    texts the request leaves out (`folded_calls`)."""
    out: list[dict] = []
    for group in steps([p for p in m.get("parts") or [] if int(p.get("step") or 1) >= from_step]):
        text = text_of(group)
        thinking = text_of(group, "thinking") if keep_reasoning else ""
        calls = [p for p in group if p.get("type") == "tool_call"]
        if not text and not thinking and not calls:
            continue
        msg: dict = {"role": "assistant", "content": text}
        if thinking:
            msg["reasoning_content"] = thinking
        if calls:
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c.get("name") or "",
                                               "arguments": call_arguments(
                                                   c, c.get("id") in (folded or ()))}}
                                 for c in calls]
        out.append(msg)
        images = []
        for c in calls:
            text, own = result_for(c, fold_before, c.get("id") in (folded or ())), []
            for a in ((c.get("output") or {}).get("images") or []) if vision else []:
                if keep is not None and a.get("id") not in keep:
                    text += LEFT_OUT
                    continue
                data = blob(a) if blob is not None else None
                if data:
                    own.append(_image_part(a, data))
            if own and inline:
                out.append({"role": "tool", "tool_call_id": c["id"],
                            "content": [{"type": "text", "text": text}, *own]})
            else:
                out.append({"role": "tool", "tool_call_id": c["id"], "content": text})
                images += own
        if images:
            out.append({"role": "user", "content": [
                {"type": "text", "text": "The images the tool calls above returned."}, *images]})
    return out


def result_text(c: dict) -> str:
    """What the model reads for one call: the saved model text, else the visible output, else the
    call's state in words (every call gets an answer, or the next request is invalid)."""
    if c.get("model_content"):
        return c["model_content"]
    out = c.get("output") or {}
    if out.get("content"):
        return out["content"]
    st = c.get("status")
    if st == "denied":
        return "The user denied this call."
    if st in ("cancelled", "running", "asking"):
        return "The call was stopped before it finished."
    return "(no output)"


def message(m: dict, vision: bool, blob=None, keep_reasoning: bool = True, keep: set | None = None,
            inline: bool = True, from_step: int = 0, fold_before: str | None = None,
            folded: set | None = None) -> list[dict]:
    """One branch message as OpenAI messages; [] for what the model does not see (notices)."""
    if m.get("role") == "user":
        return [_user(m, vision, blob)]
    if m.get("role") == "assistant":
        return _assistant(m, vision, blob, keep_reasoning, keep, inline, from_step, fold_before,
                          folded)
    return []


def body(model: str, settings: dict, branch: list[dict], *, system: str | None = None,
         vision: bool = False, blob=None, continue_final: bool = False,
         max_tokens: int | None = None, tools: list[dict] | None = None,
         levels: list[str] | None = None, effort_off: str | None = None,
         keep_reasoning_from: int = 0, tail: str | None = None,
         tandem: dict | None = None, inline_images: bool = True,
         step_from: tuple[int, int] | None = None, fold_before: str | None = None,
         window: int | None = None) -> dict:
    """`keep_reasoning_from`: the branch index from which assistant reasoning goes back (the
    context guard drops it before that); `tail`: a hidden last user line for one call, never
    stored (the empty-turn nudge; a worker's report sent back, or its report at the step cap);
    `inline_images`: tool images in the tool message (our engine) or in a user message after it;
    `step_from`: (a branch index, a step): that assistant message's steps before the step were
    compacted (a long run: the summary is in the system block); `fold_before`: the newest
    compaction's time, the long results of calls that ended before it are stubs (`result_for`)
    and a fold point of the write and edit texts (`folded_calls`, with the model's `window`)."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    keep = kept_tool_images(branch)
    folded = folded_calls(branch, fold_before, step_from, window)
    for i, m in enumerate(branch):
        msgs.extend(message(m, vision, blob, keep_reasoning=i >= keep_reasoning_from, keep=keep,
                            inline=inline_images,
                            from_step=step_from[1] if step_from and i == step_from[0] else 0,
                            fold_before=fold_before, folded=folded))
    if tail:
        msgs.append({"role": "user", "content": tail})
    out: dict = {"model": model, "messages": msgs, "stream": True,
                 "stream_options": {"include_usage": True}}
    if tools:
        out["tools"] = tools
        out["parallel_tool_calls"] = True
    effort = settings.get("effort") or "auto"
    if not settings.get("thinking", True):
        if effort_off:                      # a provider's own off word (DeepSeek: `none`)
            out["reasoning_effort"] = effort_off
        else:
            out["chat_template_kwargs"] = {"enable_thinking": False}
    elif effort != "auto":
        out["reasoning_effort"] = nearest_effort(effort, levels)
    if max_tokens:
        out["max_tokens"] = int(max_tokens)
    if continue_final and msgs and msgs[-1]["role"] == "assistant" and \
            not msgs[-1].get("tool_calls"):
        out["continue_final_message"] = True
        out["add_generation_prompt"] = False
    if tandem:
        out["tandem"] = tandem
    return out


def system_block(*, today: str | None, tools: bool, summary: str | None = None,
                 skills: str | None = None, prompt: str | None = None,
                 plan: str | None = None, worker: str | None = None,
                 agents: str | None = None) -> str | None:
    """The system message: the chat's own prompt, then the tool rules, the skill index and the
    summary of compacted turns. None for a plain chat with nothing to say. `worker`: a task run's
    block (packages/system/workers/PACKAGE.md), in place of the chat's own prompt; `agents`: the
    main agent's paragraph in Agents mode (references/main.md), right after the tool rules."""
    parts = []
    if worker:
        parts.append(worker.strip())
    elif prompt:
        parts.append(prompt.strip())
    if tools:
        parts.append(
            f"You are the assistant in TandemLLM's chat{f'. Today is {today}' if today else ''}. "
            "You can use tools. Files live in /workspace, this chat's own Linux sandbox; the "
            "internet is reachable through a proxy, the local network is not. Read a file before "
            "you edit it, and prefer edit_file for small changes. Use present to show the user a "
            "page, a file or this run's changes. When the task is done, answer in plain words; do "
            "not call tools you do not need.")
    if agents and tools:
        parts.append(agents.strip())
    if plan:
        parts.append(plan)
    if skills:
        parts.append("Skills you can read with library_read when a task needs one:\n" + skills)
    if summary:
        parts.append("Summary of the earlier part of this conversation (the messages before it "
                     "were compacted):\n" + summary.strip())
    return "\n\n".join(parts) if parts else None


EFFORT_RANK = {"low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}


def nearest_effort(effort: str, levels: list[str] | None) -> str:
    """The chat's effort as one of the model's levels: itself when the model has it, else the
    nearest by rank, a tie going to `high` (DeepSeek maps medium and xhigh to high itself; our
    engine has no `max`, a chat that picked it on DeepSeek gets xhigh on the engine)."""
    if not levels or effort in levels or effort not in EFFORT_RANK:
        return effort
    r = EFFORT_RANK[effort]
    known = [x for x in levels if x in EFFORT_RANK]
    if not known:
        return effort
    return min(known, key=lambda x: (abs(EFFORT_RANK[x] - r), abs(EFFORT_RANK[x] - 3)))


def encode(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def headers(conversation_id: str, *, session: str | None = None,
            parent_session: str | None = None, worker: bool = False) -> dict:
    """The call's headers, also fixed: the session id and the client kind (`chat-app`: the
    conductor maps it to its chat-app profile, SRV-351). Never a profile header: the conductor's
    other profiles belong to other clients (opencode, Open WebUI, Hermes) and to tests.
    `X-Tandem-Cost: 1`: a call the proxy relays to a priced model ends with its cost, an SSE
    comment before [DONE] (proxy/relay.py); our engine's own calls cost nothing and get none.
    A worker's (VIS-E15): its own session, the chat's as `X-Tandem-Parent-Session` (the session
    that comes back when the worker ends), `X-Tandem-Worker: 1`, and `X-Tandem-Memory: off`: the
    memory notes are the main agent's (server/conductor/memory.py reads nothing, writes nothing)."""
    h = {"x-session-id": session or session_id(conversation_id), "X-QSE-Client": "chat-app",
         "X-Tandem-Cost": "1"}
    if parent_session:
        h["X-Tandem-Parent-Session"] = parent_session
    if worker:
        h["X-Tandem-Worker"] = "1"
        h["X-Tandem-Memory"] = "off"
    return h
