"""One location per skill in opencode's skills list (D3's finding, D4's fix).

THE MEASUREMENT (2026-10-03, the disk tier's heads rebuilt from their chunk chains; token
ids only). Every new opencode session re-read 5 to 6 thousand tokens of a head the disk tier held:
seven session heads of 33,792 tokens, all in one folder, parted inside the system message's
`<available_skills>` list. The list has 32 entries in the same order every time; 11 of them are in
two folders the Claude app syncs (`~/.claude/skills/synced/<id-A>/<name>/SKILL.md` and
`.../synced/<id-B>/<name>/SKILL.md`, the same skills from two accounts), and opencode keeps
whichever copy its scan meets last, so the `<location>` (and for one entry the description) of
those 11 flips between sessions: `AAAABBAAAAA`, `BABBBAABABA`, ... The prompt parts at the first
flipped entry, and everything after it (the rest of the list, the user's message) is read again.

THE FIX. A request of a client this applies to (opencode, a profile not raw) gets each such entry
as the copy first seen for that skill name: the location and the description the first session
saw. Both copies are on the client's disk; opencode's skill tool loads a skill by its name, never
by this path; the text the model reads is the same in every session, so the head is too. The
first-seen copies are kept in a small JSON file next to the conductor's documents (names, paths,
descriptions: the client's skill metadata, never a user's text). A copy whose folder has not been
seen for `STALE_DAYS` gives way to the one seen now (a sync folder that was removed). Nothing else
in the message changes.

Measured (four first requests of fresh sessions captured from opencode 1.18.34 on
a Mac, two per opencode process): within one process the head is the same; across two
processes it parted at character 24,970, on two entries this file did not touch, because the
second copy was not under `synced/`: `~/.claude/skills/.trash/<n>/<name>/SKILL.md` (a removed
skill's copy the Claude app keeps) against the synced one. So every entry with a path location is
canonical now, keyed by its skill name, whatever folder it is in; `<built-in>` entries are left
alone. A copy at the location already chosen whose description changed is the skill's own edit:
the new description is learned, never replaced by the old one.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

STALE_DAYS = 7.0
_ENTRY = re.compile(r"(<skill>\s*<name>)([^<]*)(</name>\s*<description>)(.*?)"
                    r"(</description>\s*<location>)([^<]*/[^<]*)(</location>)", re.S)


class SkillCanon:
    def __init__(self, path: str | None, clock=time.time):
        self.path = path
        self.clock = clock
        self.map: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "rewritten": 0, "learned": 0}
        if path:
            try:
                with open(path) as f:
                    self.map = json.load(f).get("skills", {})
            except (OSError, ValueError):
                self.map = {}

    def _save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(tmp, "w") as f:
                json.dump({"skills": self.map}, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def text(self, s: str) -> tuple[str, int]:
        """`s` with every skill entry as its first-seen copy; (text, entries rewritten)."""
        if "<available_skills>" not in s:
            return s, 0
        now = self.clock()
        n, dirty = 0, False

        def one(m):
            nonlocal n, dirty
            name, desc, loc = m.group(2), m.group(4), m.group(6)
            known = self.map.get(name)
            if known is not None and "loc" not in known:
                known = None                       # a 0.3.22 entry (a synced folder id): relearn
            if known is None or (known["loc"] != loc and now - float(
                    known["seen"].get(known["loc"], 0)) > STALE_DAYS * 86400):
                self.map[name] = {"loc": loc, "desc": desc, "seen": {loc: now}}
                self.stats["learned"] += 1
                dirty = True
                return m.group(0)
            if loc not in known["seen"] or now - known["seen"][loc] > 3600:
                known["seen"][loc] = now
                dirty = True
            if known["loc"] == loc:
                if known["desc"] != desc:
                    known["desc"] = desc           # the chosen copy's own edit
                    dirty = True
                return m.group(0)
            n += 1
            return (m.group(1) + name + m.group(3) + known["desc"] + m.group(5) + known["loc"]
                    + m.group(7))
        with self.lock:
            out = _ENTRY.sub(one, s)
            if dirty:
                self._save()
        return out, n

    def apply(self, body: dict) -> tuple[dict, int]:
        """The request body with its system message's skills list canonical; the same object when
        nothing changes."""
        msgs = body.get("messages")
        if not isinstance(msgs, list):
            return body, 0
        self.stats["requests"] += 1
        total, new = 0, None
        for i, m in enumerate(msgs):
            if not isinstance(m, dict) or m.get("role") != "system":
                continue
            c = m.get("content")
            if isinstance(c, str):
                t, k = self.text(c)
                if k:
                    new = new or list(msgs)
                    new[i] = dict(m, content=t)
                    total += k
            elif isinstance(c, list):
                parts, changed = [], 0
                for p in c:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        t, k = self.text(p["text"])
                        if k:
                            p = dict(p, text=t)
                            changed += k
                    parts.append(p)
                if changed:
                    new = new or list(msgs)
                    new[i] = dict(m, content=parts)
                    total += changed
        if not total:
            return body, 0
        self.stats["rewritten"] += 1
        return dict(body, messages=new), total

    def report(self) -> dict:
        return {"skills": len(self.map), **self.stats}
