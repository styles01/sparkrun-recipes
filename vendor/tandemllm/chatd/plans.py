"""Plan mode, the todo list and rewind (SRV-363, plan note Chat 7). Mixed into
`chatd.service.Service`.

PLAN MODE is a chat setting (`plan_mode`). The model may use the read tools (read_file, glob, grep,
web_search, web_fetch, the library, present) and `todo_write`; it gets a line on the turn's user
message that asks for a plan and no changes (`plan_onto`), and effort high unless the chat chose
more. A call to any other tool is refused with "plan mode: read tools only" and the run goes on.
The request keeps the chat's whole tool list and system block (SRV-663): with them changed by the
mode, the turn into plan mode and the one after Approve read the whole conversation again. Approve (`POST .../plan/approve`) takes the
plan's items (the numbered or bulleted lines of the answer, or the list the user edited), makes
them the todo list, leaves plan mode and sends "Carry out the approved plan."

THE TODO LIST is per conversation (`todos`), written by `todo_write` (the whole list each time:
`{items: [{id?, text, state: pending|in_progress|done}]}`) or by Approve, and sent to every device
as `todos.changed`; the `plan` pane shows it. When a run starts while items are open, the list
goes onto that run's user message as a hidden `todos` part: the model reads it after the user's
words, the request stays byte-stable for the whole run and later turns, and a compaction keeps it
because it keeps the last two user turns whole.

REWIND (`POST .../rewind {message_id}`) on a user message: the run that is on stops first, the
workspace goes back to the `before` snapshot of the first run after that message (the runs
recorded since stay recorded), the todo list goes back to what that message's run started with
(its `todos` part keeps the items; no part: no open list then), and the head moves to the message's parent, so the user can send a
new version; the old branch stays in the tree and the branch pager shows it once a new message
sits beside it. The answer carries the message's text for the composer.
"""

from __future__ import annotations

import json
import re

from chatd.ids import iso, now_ms, ulid

PLAN_TOOLS = ("read_file", "glob", "grep", "web_search", "web_fetch", "library_search",
              "library_read", "present", "todo_write")
PLAN_REFUSAL = "plan mode: read tools only. Write the plan; the user approves it before any change."
PLAN_RULE = ("Plan mode is on: investigate with the read tools, change nothing, and answer with a "
             "numbered plan of concrete steps the user can approve. Do not carry it out yet.")
APPROVED = "Carry out the approved plan."
STATES = ("pending", "in_progress", "done")
MAX_ITEMS = 50


def plan_items(text: str) -> list[str]:
    """The steps of a plan answer: its numbered lines (1. or 1)), else its bullets, each with
    its own text; the whole answer as one item when it has neither."""
    numbered, bullets = [], []
    for line in (text or "").splitlines():
        m = re.match(r"^\s{0,3}\d{1,3}[.)]\s+(.+)", line)
        if m:
            numbered.append(m.group(1).strip())
            continue
        m = re.match(r"^\s{0,3}[-*+]\s+(?:\[[ xX]\]\s+)?(.+)", line)
        if m:
            bullets.append(m.group(1).strip())
    items = numbered or bullets or ([" ".join(text.split())[:300]] if (text or "").strip() else [])
    return [re.sub(r"\*\*(.+?)\*\*", r"\1", i)[:300] for i in items][:MAX_ITEMS]


def todo_text(items: list[dict]) -> str:
    marks = {"done": "[x]", "in_progress": "[~]", "pending": "[ ]"}
    return "Your todo list (keep it current with todo_write):\n" + "\n".join(
        f"{marks.get(i['state'], '[ ]')} {i['text']}" for i in items)


class PlansMixin:
    # ------------------------------------------------------------------ the todo list
    def todos(self, cid: str) -> list[dict]:
        r = self.store.one("SELECT items FROM todos WHERE conversation_id = ?", (cid,))
        return json.loads(r["items"]) if r else []

    def set_todos(self, cid: str, items: list) -> list[dict]:
        from chatd.service import ApiError
        if not isinstance(items, list) or len(items) > MAX_ITEMS:
            raise ApiError(400, "bad_request", f"items is a list of at most {MAX_ITEMS}")
        clean = []
        for n, it in enumerate(items):
            if isinstance(it, str):
                it = {"text": it}
            if not isinstance(it, dict) or not str(it.get("text") or "").strip():
                raise ApiError(400, "bad_request", "each item needs a text")
            state = it.get("state") or "pending"
            if state not in STATES:
                raise ApiError(400, "bad_request", "state is pending, in_progress or done")
            clean.append({"id": str(it.get("id") or f"t{n + 1}")[:40],
                          "text": str(it["text"]).strip()[:300], "state": state})
        conv = self.conv(cid)
        with conv.lock:
            self.store.run("INSERT OR REPLACE INTO todos VALUES (?, ?, ?)",
                           (cid, json.dumps(clean), now_ms()))
            self.emit(conv, {"type": "todos.changed", "todos": clean})
        return clean

    def todos_onto(self, cid: str, user_mid: str) -> None:
        """At a run's start: the open todo list onto its user message, once."""
        items = self.todos(cid)
        if not any(i["state"] != "done" for i in items):
            return
        m = self._messages(cid).get(user_mid)
        if m is None or any(p.get("type") == "todos" for p in m["parts"]):
            return
        parts = m["parts"] + [{"type": "todos", "text": todo_text(items), "items": items}]
        self.store.run("UPDATE messages SET parts = ? WHERE id = ?", (json.dumps(parts), user_mid))

    def plan_onto(self, cid: str, user_mid: str) -> None:
        """At a plan-mode run's start: the plan rule onto its user message, once (SRV-663). On the
        message, not in the system block: the prompt's head stays the same in and out of plan
        mode, and the turn after the plan reads only its own message."""
        if not self.settings_of(cid).get("plan_mode"):
            return
        m = self._messages(cid).get(user_mid)
        if m is None or any(p.get("type") == "plan" for p in m["parts"]):
            return
        parts = m["parts"] + [{"type": "plan", "text": PLAN_RULE}]
        self.store.run("UPDATE messages SET parts = ? WHERE id = ?", (json.dumps(parts), user_mid))

    # ------------------------------------------------------------------ plan mode
    def approve_plan(self, cid: str, body: dict, viewer: str | None = None) -> dict:
        from chatd.service import ApiError
        conv = self.conv(cid)
        with conv.lock:
            if conv.run is not None and conv.run.active:
                raise ApiError(409, "run_active", "approve the plan when its answer has ended")
            row = self._row(cid)
            branch, _ = self._branch(conv, row["head_id"])
            mid = body.get("message_id")
            m = next((x for x in reversed(branch) if x["role"] == "assistant" and
                      (not mid or x["id"] == mid)), None)
            if m is None:
                raise ApiError(404, "not_found", "no plan to approve")
            items = body.get("items")
            if items is None:
                from chatd.render import text_of
                items = plan_items(text_of(m["parts"]))
            if not items:
                raise ApiError(400, "bad_request", "the plan has no steps")
        todos = self.set_todos(cid, items)
        self.update_settings(cid, {"plan_mode": False})
        sent = self.send(cid, {"client_msg_id": "plan-" + ulid(), "text": APPROVED}, viewer)
        self.log(f"[plan] {cid} approved, {len(todos)} items")
        return {"todos": todos, **sent}

    # ------------------------------------------------------------------ rewind
    def rewind(self, cid: str, body: dict, viewer: str | None = None) -> dict:
        from chatd.service import ApiError
        conv = self.conv(cid)
        self._take_over(conv)
        with conv.lock:
            row = self._row(cid)
            msgs = self._messages(cid)
            m = msgs.get(str(body.get("message_id") or ""))
            if m is None or m["role"] != "user":
                raise ApiError(404, "not_found", "rewind to one of your messages")
            path = self._path(msgs, row["head_id"])
            if m["id"] not in path:
                raise ApiError(409, "not_on_branch", "that message is not on the branch you see")
            later = [msgs[x]["run_id"] for x in path[path.index(m["id"]):]
                     if msgs[x]["role"] == "assistant" and msgs[x]["run_id"]]
            first = next((r for r in later if self.snaps.has(cid, r, "before")), None)
            files = self.snaps.restore(cid, self.workspace(cid), first, "rewind-" + ulid()) \
                if first else 0
            self.store.run("UPDATE conversations SET head_id = ? WHERE id = ?",
                           (m["parent_id"], cid))
            self.emit(conv, {"type": "branch.changed", "head_id": m["parent_id"]})
            conv.context = None
            # the todo list as that message's run found it (none open then: an empty list)
            before = next((p.get("items") for p in m["parts"] if p.get("type") == "todos"), None)
        if later:
            self.set_todos(cid, before if isinstance(before, list) else [])
        self.log(f"[rewind] {cid} to before {m['id']}: {len(later)} run(s), {files} file(s)")
        from chatd.render import text_of
        return {"conversation": self.snapshot(cid, viewer), "text": text_of(m["parts"]),
                "files": files, "at": iso(now_ms())}
