"""Open WebUI tools in the chat (SRV-361, plan note Chat 3, decision D9).

An Open WebUI Tools file is Python: a module docstring with `title`, `description` and
`requirements` lines, and a `class Tools` whose public methods are the tools, typed, with a
docstring each, and an optional nested `class Valves` (its fields are the toolkit's settings).

PARSE. chatd reads the file with `ast` only; nothing of it runs on the host. Each public method
becomes a JSON schema from its type hints (str, int, float, bool, list[...], dict, Optional,
Literal) and its docstring (the first paragraph is the description, `:param x:` lines describe
the parameters); a parameter with a default is optional. Open WebUI's own parameters
(`__user__`, `__event_emitter__` and the rest of the `__x__` names) stay out of the schema. The
Valves fields become settings with their defaults and descriptions. A file with a `Filter`, `Pipe`
or `Action` class is refused: those are Open WebUI's filter layer and custom models, which the
harness and the presets cover here.

RUN. In the chat's sandbox, never on the host: the file and a small runner go to the sandbox's
home (`~/.tandem/owui/`, written through SandboxDir: the sandbox owns that folder, so no symlink
in it is followed and each file is a new one renamed into place), the requirements install once
per sandbox with `pip install --user` (pydantic too when the image lacks it), and the runner
imports the file, sets the valves, fills Open WebUI's parameters with stand-ins (an event emitter
that does nothing, a `__user__` with no personal data), calls the method and prints the result
after a marker line. The call's other output is the tool's own printing.

STORE. `user_tools`: the source, the parsed spec, the valves' values sealed with the same vault as
MCP secrets (the toolkit id as associated data). A valve whose name says key, token, secret or
password comes back empty, as MCP secrets do; the others come back with their values.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import time

from chatd import tools as tools_mod
from chatd.ids import ulid
from chatd.tools import Result, Tool, ToolError
from chatd.workspace import SandboxDir

MAX_SOURCE = 512 * 1024
MARKER = "\x1eTANDEM-OWUI-RESULT "
SECRET_VALVE = re.compile(r"key|token|secret|password|passwd|credential", re.I)
OWUI_PARAMS = re.compile(r"^__\w+__$")
TIMEOUT_S = 120
OWUI_DIR = ".tandem/owui"           # in the sandbox's home

RUNNER = r'''
import asyncio, importlib.util, inspect, json, os, sys

def main(path):
    with open(path) as fh:
        req = json.load(fh)
    os.unlink(path)
    spec = importlib.util.spec_from_file_location("owui_tool", req["file"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tools = mod.Tools()
    if hasattr(tools, "Valves"):
        tools.valves = tools.Valves(**req["valves"])
    fn = getattr(tools, req["method"])

    async def emit(*a, **k):
        return None
    kwargs = dict(req["args"])
    for p in inspect.signature(fn).parameters:
        if p == "__user__":
            kwargs[p] = {"id": "chat", "name": "user", "role": "user"}
        elif p in ("__event_emitter__", "__event_call__"):
            kwargs[p] = emit
        elif p.startswith("__") and p.endswith("__"):
            kwargs[p] = {} if p == "__metadata__" else None
    r = fn(**kwargs)
    if inspect.iscoroutine(r):
        r = asyncio.run(r)
    out = r if isinstance(r, str) else json.dumps(r, default=str)
    sys.stdout.write("\n" + MARKER + json.dumps({"result": out}) + "\n")

MARKER = "\x1eTANDEM-OWUI-RESULT "
main(sys.argv[1])
'''


class ParseError(ValueError):
    pass


def _docstring_meta(doc: str) -> dict:
    out = {}
    for line in (doc or "").splitlines():
        k, sep, v = line.strip().partition(":")
        if sep and re.fullmatch(r"[a-z_]+", k.strip()):
            out[k.strip()] = v.strip()
    return out


def _schema(node) -> tuple[dict, bool]:
    """(JSON schema, optional) of a type annotation."""
    if node is None:
        return {"type": "string"}, False
    if isinstance(node, ast.Constant) and node.value is None:
        return {"type": "null"}, True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _schema(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return {"type": "string"}, False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        parts = [_schema(x) for x in (node.left, node.right)]
        real = [s for s, _ in parts if s.get("type") != "null"]
        return (real[0] if real else {"type": "string"}), len(real) < 2
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) \
        else None
    if name is not None:
        return {"str": {"type": "string"}, "int": {"type": "integer"},
                "float": {"type": "number"}, "bool": {"type": "boolean"},
                "list": {"type": "array"}, "List": {"type": "array"},
                "dict": {"type": "object"}, "Dict": {"type": "object"},
                "Any": {}}.get(name, {"type": "string"}), False
    if isinstance(node, ast.Subscript):
        base = node.value.id if isinstance(node.value, ast.Name) else \
            node.value.attr if isinstance(node.value, ast.Attribute) else ""
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if base == "Optional":
            return _schema(args[0])[0], True
        if base == "Union":
            real = [_schema(a)[0] for a in args if not (isinstance(a, ast.Constant)
                                                         and a.value is None)]
            return (real[0] if real else {"type": "string"}), len(real) < len(args)
        if base in ("list", "List", "Sequence", "set", "Set", "tuple", "Tuple"):
            return {"type": "array", "items": _schema(args[0])[0]}, False
        if base in ("dict", "Dict", "Mapping"):
            return {"type": "object"}, False
        if base == "Literal":
            vals = [a.value for a in args if isinstance(a, ast.Constant)]
            kinds = {type(v).__name__ for v in vals}
            t = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}.get(
                kinds.pop() if len(kinds) == 1 else "", "string")
            return {"type": t, "enum": vals}, False
    return {"type": "string"}, False


def _param_docs(doc: str) -> tuple[str, dict]:
    """(the description, {param: its line}) of a method's docstring."""
    doc = doc or ""
    params = {m.group(1): m.group(2).strip()
              for m in re.finditer(r":param\s+(\w+)\s*:\s*(.+)", doc)}
    head = re.split(r"\n\s*\n|:param|:return", doc.strip(), maxsplit=1)[0]
    return " ".join(head.split())[:1024], params


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def _valves(cls: ast.ClassDef) -> list[dict]:
    out = []
    for st in cls.body:
        if not isinstance(st, ast.AnnAssign) or not isinstance(st.target, ast.Name):
            continue
        schema, _ = _schema(st.annotation)
        default, desc = None, ""
        v = st.value
        if isinstance(v, ast.Call) and getattr(v.func, "id", getattr(v.func, "attr", "")) == "Field":
            for kw in v.keywords:
                if kw.arg == "default":
                    default = _literal(kw.value)
                elif kw.arg == "description":
                    desc = str(_literal(kw.value) or "")
            if v.args:
                default = _literal(v.args[0])
        elif v is not None:
            default = _literal(v)
        out.append({"name": st.target.id, "type": schema.get("type", "string"),
                    "default": default, "description": desc[:300],
                    "secret": bool(SECRET_VALVE.search(st.target.id))})
    return out


# SRV-387: the modules through which a toolkit reaches past its sandbox (the network through the
# egress proxy, or other programs); auto mode asks before a call of a toolkit that imports one
EFFECT_MODULES = {"requests", "httpx", "aiohttp", "urllib", "urllib3", "http", "socket", "ssl",
                  "smtplib", "ftplib", "imaplib", "poplib", "websocket", "websockets", "paramiko",
                  "boto3", "subprocess", "asyncio.subprocess", "pexpect"}


def side_effects(source: str) -> bool:
    """Whether a toolkit imports a network or process module (EFFECT_MODULES); a file that does
    not parse counts as one that does."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return True
    for node in ast.walk(tree):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
            [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        if any(n in EFFECT_MODULES or n.split(".")[0] in EFFECT_MODULES for n in names):
            return True
        if isinstance(node, ast.Call) and (getattr(node.func, "id", None) == "__import__" or
                                           getattr(node.func, "attr", None) in ("system", "popen")
                                           or str(getattr(node.func, "attr", "")).startswith(("spawn", "exec"))):
            return True
    return False


def parse(source: str) -> dict:
    """The spec of an Open WebUI Tools file: {title, description, requirements, tools, valves}."""
    if len(source.encode()) > MAX_SOURCE:
        raise ParseError("a tools file may be at most 512 KB")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ParseError(f"not valid Python: line {exc.lineno}: {exc.msg}") from None
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    other = [k for k in ("Filter", "Pipe", "Action") if k in classes]
    if other:
        raise ParseError(f"this file is an Open WebUI {other[0].lower()}: filters, pipes and "
                         f"actions are not imported (the harness and the presets cover them)")
    cls = classes.get("Tools")
    if cls is None:
        raise ParseError("no `class Tools` in the file")
    meta = _docstring_meta(ast.get_docstring(tree) or "")
    tools, valves = [], []
    for st in cls.body:
        if isinstance(st, ast.ClassDef) and st.name == "Valves":
            valves = _valves(st)
        if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) or st.name.startswith("_"):
            continue
        desc, pdocs = _param_docs(ast.get_docstring(st) or "")
        args = st.args.args[1:]                              # self
        defaults = [None] * (len(args) - len(st.args.defaults)) + list(st.args.defaults)
        props, required = {}, []
        for a, d in zip(args, defaults):
            if OWUI_PARAMS.match(a.arg):
                continue
            schema, optional = _schema(a.annotation)
            schema = dict(schema)
            if a.arg in pdocs:
                schema["description"] = pdocs[a.arg][:300]
            props[a.arg] = schema
            if d is None and not optional:
                required.append(a.arg)
        for a, d in zip(st.args.kwonlyargs, st.args.kw_defaults):
            if OWUI_PARAMS.match(a.arg):
                continue
            schema, optional = _schema(a.annotation)
            props[a.arg] = dict(schema)
            if d is None and not optional:
                required.append(a.arg)
        tools.append({"method": st.name, "description": desc or st.name,
                      "parameters": {"type": "object", "properties": props,
                                     "required": required}})
    if not tools:
        raise ParseError("`class Tools` has no public methods")
    reqs = [r.strip() for r in re.split(r"[,\s]+", meta.get("requirements", "")) if r.strip()]
    for r in reqs:
        if not re.fullmatch(r"[A-Za-z0-9._\-\[\]]+([<>=!~]=?[A-Za-z0-9.*+!-]+)?", r):
            raise ParseError(f"the requirement {r!r} is not a plain package spec")
    return {"title": meta.get("title") or "Open WebUI tools",
            "description": meta.get("description") or "", "version": meta.get("version") or "",
            "requirements": reqs, "tools": tools, "valves": valves}


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (s or "tools")[:24]


class Owui:
    """The imported toolkits, bound to the service (the store, the sandboxes, the vault)."""

    def __init__(self, svc):
        self.svc = svc
        self.store = svc.store

    # ------------------------------------------------------------------ the store
    def _rows(self):
        return self.store.all("SELECT * FROM user_tools ORDER BY name")

    def _row(self, tid: str):
        r = self.store.one("SELECT * FROM user_tools WHERE id = ?", (tid,))
        if r is None:
            raise self.svc._api(404, "not_found", "no such toolkit")
        return r

    def _valves(self, r) -> dict:
        spec = json.loads(r["spec"])
        vals = {v["name"]: v["default"] for v in spec["valves"]}
        vals.update(self.svc.mcp._open(r["id"], r["valves_enc"]))
        return vals

    def info(self, r) -> dict:
        spec = json.loads(r["spec"])
        vals = self._valves(r)
        return {"id": r["id"], "slug": r["name"], "title": r["title"],
                "description": spec["description"], "version": spec["version"],
                "requirements": spec["requirements"], "enabled": bool(r["enabled"]),
                "tools": [self.tool_name(r["name"], t["method"]) for t in spec["tools"]],
                "valves": [{**v, "value": "" if v["secret"] else vals.get(v["name"]),
                            "set": vals.get(v["name"]) not in (None, "")}
                           for v in spec["valves"]],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["created"] / 1000))}

    def list(self) -> list[dict]:
        return [self.info(r) for r in self._rows()]

    def add(self, filename: str, source: str) -> dict:
        try:
            spec = parse(source)
        except ParseError as exc:
            raise self.svc._api(400, "bad_tools_file", str(exc)) from None
        base = slug(spec["title"] if spec["title"] != "Open WebUI tools" else
                    os.path.splitext(filename)[0])
        taken = {r["name"] for r in self._rows()}
        name, n = base, 2
        while name in taken:
            name, n = f"{base[:21]}_{n}", n + 1
        tid = ulid()
        now = int(time.time() * 1000)
        self.store.run("INSERT INTO user_tools (id, name, kind, title, source, spec, valves_enc, "
                       "enabled, created, updated) VALUES (?, ?, 'owui', ?, ?, ?, NULL, 1, ?, ?)",
                       (tid, name, spec["title"][:120], source, json.dumps(spec), now, now))
        self.svc.log(f"[owui] imported {name}: {len(spec['tools'])} tools")
        return self.info(self._row(tid))

    def patch(self, tid: str, body: dict) -> dict:
        r = self._row(tid)
        if "enabled" in body:
            self.store.run("UPDATE user_tools SET enabled = ? WHERE id = ?",
                           (int(bool(body["enabled"])), tid))
        if isinstance(body.get("valves"), dict):
            spec = json.loads(r["spec"])
            known = {v["name"]: v for v in spec["valves"]}
            cur = self.svc.mcp._open(tid, r["valves_enc"])
            for k, v in body["valves"].items():
                if k not in known:
                    raise self.svc._api(400, "bad_request", f"no valve {k}")
                if known[k]["secret"] and v == "":
                    continue                      # an empty secret field keeps what is stored
                cur[k] = v
            self.store.run("UPDATE user_tools SET valves_enc = ?, updated = ? WHERE id = ?",
                           (self.svc.mcp._seal(tid, cur), int(time.time() * 1000), tid))
        return self.info(self._row(tid))

    def remove(self, tid: str) -> None:
        r = self._row(tid)
        names = [self.tool_name(r["name"], t["method"]) for t in json.loads(r["spec"])["tools"]]
        self.store.run("DELETE FROM user_tools WHERE id = ?", (tid,))
        for n in names:
            self.store.run("DELETE FROM tool_prefs WHERE tool = ?", (n,))

    # ------------------------------------------------------------------ the tools
    @staticmethod
    def tool_name(slug_: str, method: str) -> str:
        return f"owui__{slug_}__{method}"[:64]

    def _tools_of(self, r) -> list[Tool]:
        spec = json.loads(r["spec"])
        effects = side_effects(r["source"] or "")
        return [Tool(self.tool_name(r["name"], t["method"]),
                     f"{t['description']} ({spec['title']}, an Open WebUI tool)"[:1024],
                     t["parameters"], risk="exec", default="ask", sandbox=True,
                     source={"kind": "user", "id": r["id"]},
                     executor=self._executor(r["id"], t["method"]), side_effects=effects)
                for t in spec["tools"]]

    def all_tools(self) -> list[Tool]:
        return [t for r in self._rows() for t in self._tools_of(r)]

    def tools_for(self) -> list[Tool]:
        return [t for r in self._rows() if r["enabled"] for t in self._tools_of(r)]

    def tool(self, name: str):
        return next((t for t in self.all_tools() if t.name == name), None)

    def _executor(self, tid: str, method: str):
        def run(ctx, args):
            r = self._row(tid)
            spec = json.loads(r["spec"])
            # the home is the sandbox's (/home/agent, rw): a symlink it left at .tandem, owui
            # or a file name is refused or replaced, never followed (SandboxDir)
            _, home, _ = self.svc.sandboxes.paths(ctx.cid)
            d = SandboxDir(home, "/home/agent")
            d.ensure()
            d.write(f"{OWUI_DIR}/{r['name']}.py", r["source"].encode(), over_link=True)
            d.write(f"{OWUI_DIR}/run.py", RUNNER.encode(), over_link=True)
            req_name = f"req-{ctx.call_id}.json"
            d.write(f"{OWUI_DIR}/{req_name}", json.dumps(
                {"file": f"{r['name']}.py", "method": method, "args": args,
                 "valves": self._valves(r)}).encode(), mode=0o600, over_link=True)
            reqs = spec["requirements"]
            stamp = hashlib.sha256(" ".join(sorted(reqs)).encode()).hexdigest()[:12]
            q = " ".join(shlex.quote(x) for x in reqs)
            cmd = ("cd ~/.tandem/owui && "
                   + ("{ python3 -c 'import pydantic' 2>/dev/null || pip install --user -q "
                      "--disable-pip-version-check pydantic >&2; } && "
                      if "pydantic" in r["source"] else "")
                   + (f"{{ [ -f .req-{stamp} ] || {{ pip install --user -q "
                      f"--disable-pip-version-check {q} >&2 && touch .req-{stamp}; }}; }} && "
                      if reqs else "")
                   + f"python3 run.py {req_name}")
            code, timed = ctx.exec(["bash", "-lc", cmd], timeout=TIMEOUT_S)
            out = ctx.output_text()
            if MARKER in out:
                before, _, after = out.rpartition(MARKER)
                try:
                    result = json.loads(after.splitlines()[0])["result"]
                except (ValueError, KeyError, IndexError):
                    result = after
                return Result(result if result.strip() else "(no result)")
            if timed:
                raise ToolError(f"the tool did not answer within {TIMEOUT_S} s")
            if getattr(ctx, "oom", False):
                raise ToolError(f"the tool failed: {tools_mod.oom_words()}, the kernel killed it "
                                f"(exit {code}):\n" + out[-3000:], tools_mod.oom_data())
            raise ToolError(f"the tool failed (exit {code}):\n" + out[-3000:])
        return run
