"""Skills, prompts and presets (SRV-361; plan note Chat 3). Mixed into `chatd.service.Service`.

SKILLS are the engine's: the library (`/v1/library`, BM25 over the packages' SKILL.md files) and
the router that injects a package at the prompt's tail by the turn's class. A chat controls them
with `packages_mode`:

    auto     (the default) the harness picks; the chat's pins go in too, its blocks never
    manual   the pins only: every other known skill is sent as blocked
    model    the model reads the skill index (name and one line each, ~30 tokens a skill) in its
             system block and pulls a body with `library_read`; the harness injects nothing but
             the pins

Pins and blocks travel in the body as `tandem.packages` (server/conductor/packages.py
`body_packages`, honoured by the router). A skill switched off in the settings is blocked in every
chat. A server without the library (another engine, a cloud backend) lists no skills and the
modes change nothing.

USER SKILLS: a .zip with a SKILL.md folder (`POST /packages/upload`). chatd checks it (one folder,
no path outside it, no links, size limits, frontmatter `name` and `description`), unpacks it under
`<data>/skills/<name>/` and installs it through the engine's packages document
(`PUT /v1/conductor/packages`): that folder in `skill_dirs`, the skill as `skill:<name>` with
`library: true`, so the library, the router and the lookup drafter's index all see it. A name the
engine already has from elsewhere is refused; a skill uploaded here again is replaced, and
`DELETE /packages/<name>` takes it out of the document and off the disk.

PROMPTS are slash commands with a template ({{CLIPBOARD}}, {{selection}}, {{input}} filled on the
device); a chat's `prompt_id` makes one its system prompt. PRESETS are a model, a prompt, an
effort, tools, skills and MCP servers; a new chat can start from one, and changing the preset later
leaves existing chats as they are.
"""

from __future__ import annotations

import email.parser
import email.policy
import io
import json
import os
import re
import shutil
import stat
import time
import zipfile

from chatd.ids import ulid

LIBRARY_TTL_S = 300
INDEX_MAX = 80
SKILL_ZIP_MAX = 10 * 1024 * 1024
SKILL_UNPACKED_MAX = 20 * 1024 * 1024
SKILL_FILES_MAX = 200
SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


def multipart_file(content_type: str, body: bytes, field: str = "file") -> tuple[str, bytes]:
    """(file name, bytes) of a multipart form's `field`; ValueError without one."""
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    for part in msg.iter_parts() if msg.is_multipart() else []:
        if part.get_param("name", header="content-disposition") == field:
            return os.path.basename(part.get_filename() or ""), part.get_payload(decode=True) or b""
    raise ValueError(f"send the file as multipart field `{field}`")


def frontmatter(text: str) -> dict:
    """`key: value` lines between the leading `---` lines (quotes dropped); {} without them."""
    m = re.match(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(\r?\n|\Z)", text, re.S)
    out = {}
    for line in (m.group(1).splitlines() if m else []):
        k, sep, v = line.partition(":")
        if sep and k.strip() and not k.startswith((" ", "\t")):
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def unpack_skill(data: bytes) -> tuple[dict, dict[str, bytes]]:
    """(frontmatter, {path inside the skill folder: bytes}) of a skill zip; ValueError says why
    it is refused."""
    if len(data) > SKILL_ZIP_MAX:
        raise ValueError("a skill zip may be at most 10 MB")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("this is not a zip file") from None
    entries = [i for i in z.infolist() if not i.is_dir()]
    if len(entries) > SKILL_FILES_MAX:
        raise ValueError(f"a skill may have at most {SKILL_FILES_MAX} files")
    if sum(i.file_size for i in entries) > SKILL_UNPACKED_MAX:
        raise ValueError("a skill may unpack to at most 20 MB")
    names = [i.filename for i in entries]
    skill_md = [n for n in names if n == "SKILL.md" or n.endswith("/SKILL.md")]
    if not skill_md:
        raise ValueError("no SKILL.md in the zip")
    root = min(skill_md, key=lambda n: n.count("/"))[:-len("SKILL.md")]
    if root.count("/") > 1:
        raise ValueError("SKILL.md must be at the top of the zip or in its one folder")
    files = {}
    for i in entries:
        n = i.filename
        mode = i.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"{n} is a link; a skill holds plain files only")
        if n.startswith("/") or "\\" in n or ".." in n.split("/") or not n.startswith(root):
            raise ValueError(f"{n} is outside the skill's folder")
        rel = n[len(root):]
        if not rel or rel.startswith("/"):
            raise ValueError(f"{n} is outside the skill's folder")
        files[rel] = z.read(i)
    try:
        meta = frontmatter(files["SKILL.md"].decode("utf-8"))
    except UnicodeDecodeError:
        raise ValueError("SKILL.md is not UTF-8 text") from None
    name, desc = meta.get("name", ""), meta.get("description", "")
    if not SKILL_NAME.match(name):
        raise ValueError("the frontmatter needs a `name` of lower-case letters, digits, `.`, `_` "
                         "and `-`, at most 64")
    if not 8 <= len(desc) <= 600:
        raise ValueError("the frontmatter needs a `description` of 8 to 600 characters")
    return meta, files


class SkillsMixin:
    def _api(self, status, kind, msg):
        from chatd.service import ApiError
        return ApiError(status, kind, msg)

    # ------------------------------------------------------------------ the library
    def library(self) -> list[dict]:
        """The engine's skills [{name, description, classes}], cached."""
        c = getattr(self, "_lib_cache", None)
        if c is not None and time.monotonic() - c[0] < LIBRARY_TTL_S:
            return c[1]
        st, obj = self.engine.get_json("/v1/library", timeout=5.0)
        rows = []
        if st == 200 and isinstance(obj, dict):
            for s in obj.get("skills") or obj.get("packages") or []:
                if isinstance(s, dict) and s.get("name"):
                    rows.append({"name": s["name"], "description": str(s.get("description")
                                                                       or "")[:300],
                                 "classes": s.get("classes") or [],
                                 "tokens": s.get("tokens")})
        self._lib_cache = (time.monotonic(), rows)
        return rows

    def _off(self) -> set:
        return {r["tool"][4:] for r in self.store.all(
            "SELECT tool FROM tool_prefs WHERE tool LIKE 'pkg:%' AND enabled = 0")}

    def packages(self) -> list[dict]:
        off = self._off()
        return [{"id": s["name"], "name": s["name"], "description": s["description"],
                 "kind": "skill", "version": "", "enabled": s["name"] not in off, "tools": [],
                 "tokens": s.get("tokens"),
                 "user": os.path.isdir(os.path.join(self.skills_dir, s["name"]))}
                for s in self.library()]

    def set_package(self, pid: str, body: dict) -> dict:
        if not any(s["name"] == pid for s in self.library()):
            raise self._api(404, "not_found", "no such skill")
        self.store.run("INSERT OR REPLACE INTO tool_prefs VALUES (?, ?, NULL)",
                       ("pkg:" + pid, 1 if body.get("enabled", True) else 0))
        return next(p for p in self.packages() if p["id"] == pid)

    def skill_index(self, settings: dict) -> str | None:
        if settings.get("packages_mode") != "model":
            return None
        off = self._off() | set(settings.get("packages_blocked") or [])
        lines = [f"- {s['name']}: {s['description'].splitlines()[0][:160] if s['description'] else ''}"
                 for s in self.library() if s["name"] not in off][:INDEX_MAX]
        return "\n".join(lines) or None

    def skills_report(self, settings: dict) -> dict:
        """The Context panel's skills: the mode and each pin with its tokens, from the library as
        last read (no request here); a pin the library no longer has is `missing`."""
        c = getattr(self, "_lib_cache", None)
        lib = {s["name"]: s for s in c[1]} if c else None
        return {"mode": settings.get("packages_mode") or "auto",
                "pinned": [{"name": n, "tokens": (lib or {}).get(n, {}).get("tokens"),
                            "missing": lib is not None and bool(lib) and n not in lib}
                           for n in settings.get("packages") or []],
                "blocked": list(settings.get("packages_blocked") or [])}

    def missing_pins(self, settings: dict) -> list[str]:
        names = {s["name"] for s in self.library()}
        return [p for p in settings.get("packages") or [] if names and p not in names]

    def tandem_field(self, settings: dict) -> dict:
        """What our engine reads from the body: the compaction marker (SRV-351) and the chat's
        pins and blocks by its mode (SRV-361). A pin the library no longer has is left out."""
        t: dict = {"compaction": False}
        gone = set(self.missing_pins(settings))
        pin = [p for p in settings.get("packages") or [] if p not in gone]
        block = set(settings.get("packages_blocked") or []) | self._off()
        mode = settings.get("packages_mode") or "auto"
        if mode in ("manual", "model"):
            block |= {s["name"] for s in self.library() if s["name"] not in pin}
        block -= set(pin)
        if pin or block:
            t["packages"] = {"pin": pin, "block": sorted(block)}
        return t

    # ------------------------------------------------------------------ user skills
    @property
    def skills_dir(self) -> str:
        return os.path.join(os.path.abspath(self.data_dir), "skills")

    def _packages_doc(self) -> tuple[int, dict]:
        try:
            st, obj = self.engine.request("GET", "/v1/conductor/packages", timeout=10.0)
        except OSError:
            st, obj = 0, None
        if st != 200 or not isinstance(obj, dict) or not isinstance(obj.get("doc"), dict):
            raise self._api(502, "engine", f"the engine's packages document is not readable "
                                           f"(HTTP {st}); skills install on our own engine only")
        return int(obj.get("version") or 0), obj["doc"]

    def _put_packages(self, version: int, doc: dict, note: str) -> None:
        try:
            st, obj = self.engine.request("PUT", "/v1/conductor/packages", json.dumps(
                {"doc": doc, "note": note, "base_version": version}).encode(), timeout=20.0)
        except OSError:
            st, obj = 0, None
        if st != 200:
            err = (obj or {}).get("error") if isinstance(obj, dict) else None
            msg = err.get("message") if isinstance(err, dict) else str(err or "")
            raise self._api(409 if st == 409 else 502, "engine",
                            f"the engine refused the packages change (HTTP {st}) {msg}".strip())
        self._lib_cache = None

    def upload_skill(self, content_type: str, body: bytes) -> dict:
        try:
            _, data = multipart_file(content_type, body)
            meta, files = unpack_skill(data)
        except ValueError as exc:
            raise self._api(400, "bad_skill", str(exc)) from None
        name = meta["name"]
        version, doc = self._packages_doc()
        target = os.path.join(self.skills_dir, name)
        ours = os.path.isdir(target)
        if name in (doc.get("packages") or {}) and not ours:
            raise self._api(409, "taken", f"the engine already has a package named {name}")
        dirs = list(doc.get("skill_dirs") or [])
        if self.skills_dir not in dirs:
            if len(dirs) >= 8:
                raise self._api(409, "full", "the packages document has 8 skill folders already")
            dirs.append(self.skills_dir)
        tmp = target + ".new"
        shutil.rmtree(tmp, ignore_errors=True)
        for rel, blob in files.items():
            path = os.path.join(tmp, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(blob)
        old = target + ".old"
        shutil.rmtree(old, ignore_errors=True)
        if ours:
            os.rename(target, old)
        os.rename(tmp, target)
        text = files["SKILL.md"].decode("utf-8")
        pkgs = dict(doc.get("packages") or {})
        pkgs[name] = {"title": str(meta.get("title") or name)[:80], "kind": "skill",
                      "source": f"skill:{name}", "description": meta["description"], "body": "",
                      "max_tokens": min(32768, max(256, len(text) // 2)), "enabled": True,
                      "library": True}
        try:
            self._put_packages(version, {**doc, "skill_dirs": dirs, "packages": pkgs},
                               f"chat: the user skill {name}")
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            if ours:
                os.rename(old, target)
            raise
        shutil.rmtree(old, ignore_errors=True)
        self.log(f"[skills] installed {name} ({len(files)} files)")
        got = next((p for p in self.packages() if p["id"] == name), None)
        return got or {"id": name, "name": name, "description": meta["description"],
                       "kind": "skill", "version": "", "enabled": True, "tools": [],
                       "user": True}

    def delete_skill(self, name: str) -> None:
        target = os.path.join(self.skills_dir, name)
        if not SKILL_NAME.match(name) or not os.path.isdir(target):
            raise self._api(404, "not_found", "no skill of yours by that name; skills that came "
                                              "with the engine are switched off, not deleted")
        version, doc = self._packages_doc()
        pkgs = {k: v for k, v in (doc.get("packages") or {}).items() if k != name}
        self._put_packages(version, {**doc, "packages": pkgs}, f"chat: removed the user skill "
                                                                f"{name}")
        shutil.rmtree(target, ignore_errors=True)
        self.store.run("DELETE FROM tool_prefs WHERE tool = ?", ("pkg:" + name,))

    # ------------------------------------------------------------------ prompts
    def prompts(self) -> list[dict]:
        return [{"id": r["id"], "title": r["title"], "command": r["command"], "body": r["template"]}
                for r in self.store.all("SELECT * FROM prompts ORDER BY command")]

    def save_prompt(self, pid: str | None, body: dict) -> dict:
        cmd = re.sub(r"[^a-z0-9_-]", "", str(body.get("command") or "").lower().lstrip("/"))[:40]
        if not cmd or not str(body.get("body") or "").strip():
            raise self._api(400, "bad_request", "a prompt needs a command and a body")
        clash = self.store.one("SELECT id FROM prompts WHERE command = ?", (cmd,))
        if clash is not None and clash["id"] != pid:
            raise self._api(409, "taken", f"/{cmd} is taken")
        pid = pid or ulid()
        self.store.run("INSERT OR REPLACE INTO prompts VALUES (?, ?, ?, ?)",
                       (pid, cmd, str(body.get("title") or cmd)[:120], str(body["body"])[:20000]))
        return next(p for p in self.prompts() if p["id"] == pid)

    def delete_prompt(self, pid: str) -> None:
        self.store.run("DELETE FROM prompts WHERE id = ?", (pid,))

    def commands(self) -> list[dict]:
        """The slash list: chatd's own, the saved prompts, and every MCP server's prompts (their
        arguments come from the text typed after the command)."""
        from chatd.mcp import slug
        mcp = [{"command": f"mcp__{slug(s['name'])}__{p['name']}"[:64],
                "description": p["description"] or f"a prompt of {s['name']}",
                "mcp_prompt": {"server_id": s["id"], "name": p["name"],
                               "arguments": p["arguments"]}}
               for s in self.mcp.list() for p in s.get("prompts") or []]
        return [{"command": "clear", "description": "start a new chat"},
                {"command": "compact", "description": "summarize the older turns now"}] + \
            [{"command": p["command"], "description": p["title"], "prompt_id": p["id"]}
             for p in self.prompts()] + mcp

    # ------------------------------------------------------------------ presets
    PRESET_KEYS = ("model", "profile", "effort", "thinking", "tools", "prompt_id", "packages",
                   "mcp")

    def presets(self) -> list[dict]:
        out = []
        for r in self.store.all("SELECT * FROM presets ORDER BY name"):
            s = json.loads(r["settings"] or "{}")
            out.append({"id": r["id"], "name": r["name"], **{k: s.get(k) for k in
                                                            self.PRESET_KEYS}})
        return out

    def save_preset(self, pid: str | None, body: dict) -> dict:
        name = str(body.get("name") or "").strip()[:80]
        if not name:
            raise self._api(400, "bad_request", "a preset needs a name")
        pid = pid or ulid()
        s = {k: body.get(k) for k in self.PRESET_KEYS if k in body}
        self.store.run("INSERT OR REPLACE INTO presets VALUES (?, ?, ?, ?, ?)",
                       (pid, name, s.get("model"), None, json.dumps(s)))
        return next(p for p in self.presets() if p["id"] == pid)

    def delete_preset(self, pid: str) -> None:
        self.store.run("DELETE FROM presets WHERE id = ?", (pid,))

    def preset_settings(self, pid: str | None) -> dict:
        if not pid:
            return {}
        p = next((x for x in self.presets() if x["id"] == pid), None)
        if p is None:
            raise self._api(404, "not_found", "no such preset")
        return {k: v for k, v in p.items() if k in self.PRESET_KEYS and v is not None}
