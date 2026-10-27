"""Turn snapshots and turn outcomes (J1, 2026-10-02: the plan "Laya as the test judge, state
snapshots, turn outcomes and the client protocol", steps 1 to 3).

What the harness sees of a session is the request body, the stream it writes and its journal; it
never sees the project folder. So the snapshot is built from the body alone, at the start of every
request whose newest message is a tool result (SP4, before the router's effort), from the tool
calls the session made since the last snapshot and what they returned:

    calls     name, arguments (paths kept, long strings clipped), the result's head and tail
    files     the writes and edits: path, the content's size and a hash of it
    last_write, last_run   indices among the calls (the K1 twin: a run-like call after a write)
    shots     each image a tool returned: a hash of its bytes (of their base64, the same
              equality), width and height; the bytes go to a file on the writer's thread
    console   a console read's level counts and its last error lines
    checkers  RESULT lines with their readings, and the session's best per gate
    exits, tests   exit codes and a test runner's counts, read off the result's shape
    state     the JSON object a result carries under "tandem_state", and its diff against the
              previous one
    logs      the tail of a `.tandem/*.log` the model printed

`Evidence` is what a session keeps between snapshots (hashes, numbers, names; no result text).
The outcome is the first label of the router document's `outcomes.order` whose facts matched,
`progress` when only progress facts did, `none` when there was nothing to read. Every fact is a
structural one (an equal hash, a number, an exit code, a pattern on a tool result's shape); none
is a word that decides what anybody meant (the project's rule). A label that needs a reading
(the evidence does not match the goal, the screenshot is wrong) is left to Qwen in context: the
note asks it.

`digest()` is the snapshot as Qwen reads it in a note: fixed rows, numbers first, clipped tails
last, at most `limits.digest_chars` characters (600 tokens). Images are named by when they were
taken, never re-sent. Nothing here touches a model, the engine or the disk: the router calls
`observe` under the session's lock, the transcript writer stores the result.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import json
import re
import struct
import time

from server.conductor import redact, signals

SEEN_MAX = 4096
OUTPUTS_MAX = 512
EMPTY = ("", "(no output)")
_PATH_KEYS = ("filePath", "file_path", "path", "file", "url")
_DATA_URL = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,", re.I)
# a member of an object the script did not declare, assigned (not compared): `keys['x'] = true`,
# `player.x = 3200`; the declared names (`const s = ...`, a parameter) are taken out after
_ASSIGN = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*((?:\[[^\]\n]{1,80}\]|\.[A-Za-z_$][\w$]*)+)"
                     r"\s*(?<![=!<>+\-*/%&|^])=(?![=>])")
_DECL = re.compile(r"\b(?:const|let|var|function)\s+([A-Za-z_$][\w$]*)|\(\s*([A-Za-z_$][\w$]*)"
                   r"(?:\s*,\s*[A-Za-z_$][\w$]*)*\s*\)\s*=>|([A-Za-z_$][\w$]*)\s*=>")


def _h(text) -> str:
    raw = text if isinstance(text, bytes) else str(text).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _rx(p, cache: dict):
    """A config pattern compiled once; a pattern that does not compile matches nothing."""
    got = cache.get(p)
    if got is None:
        try:
            got = re.compile(p)
        except re.error:
            got = re.compile(r"(?!x)x")
        cache[p] = got
    return got


class Evidence:
    """One session's state between snapshots: hashes, counts, numbers and names."""

    def __init__(self, now: float):
        self.snapshots = 0
        self.seen: collections.OrderedDict = collections.OrderedDict()
        self.writes = 0                    # write calls so far, in call order
        self.shot: dict | None = None      # the newest screenshot: sha, writes then, snapshot
        self.outputs: collections.OrderedDict = collections.OrderedDict()   # call key -> facts
        self.best: dict = {}               # checker gate -> smallest gap seen
        self.fails: dict = {}              # checker gate -> FAIL runs in a row
        self.failed: dict = {}             # test command -> its failed count
        self.state: dict | None = None     # the newest tandem_state
        self.state_writes = 0              # the writes before it
        self.progress_at, self.progress_snap = now, 0
        self.compaction: dict | None = None
        self.raised: dict | None = None    # act: {"level", "left"}
        self.noted: set = set()            # (label, signature) noted since the last progress
        self.stuck_noted = False
        self.last: dict | None = None      # the newest outcome, for E-b and the live view
        self._rx: dict = {}

    def fresh(self, key: str) -> bool:
        if key in self.seen:
            return False
        self.seen[key] = True
        while len(self.seen) > SEEN_MAX:
            self.seen.popitem(last=False)
        return True


def _args(raw) -> dict | None:
    if isinstance(raw, dict):
        return raw
    try:
        got = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
    return got if isinstance(got, dict) else None


def _clip(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    half = max(0, (n - 7) // 2)
    return text[:half] + "\n[...]\n" + text[-half:]


def _show_args(args: dict | None, raw: str, n: int = 240) -> dict | str:
    """A call's arguments as the snapshot keeps them: paths whole, every other string clipped."""
    if args is None:
        return _clip(str(raw or ""), n)
    out = {}
    for k, v in args.items():
        if isinstance(v, str):
            out[k] = v if k in _PATH_KEYS else _clip(v, n)
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = _clip(json.dumps(v, ensure_ascii=False), n)
    return out


def png_size(b64: str) -> tuple[int | None, int | None]:
    """Width and height from a PNG's IHDR (the first 24 bytes); None for anything else."""
    try:
        head = base64.b64decode(b64[:32] + "=" * (-len(b64[:32]) % 4))
    except (ValueError, TypeError):
        return None, None
    if head[:8] != b"\x89PNG\r\n\x1a\n" or len(head) < 24:
        return None, None
    return struct.unpack(">II", head[16:24])


def media_parts(content) -> list[str]:
    """The base64 payloads of a message's image parts (data URLs only)."""
    out = []
    if not isinstance(content, list):
        return out
    for p in content:
        if not isinstance(p, dict):
            continue
        url = p.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
        if not isinstance(url, str):
            src = p.get("source") if isinstance(p.get("source"), dict) else {}
            if src.get("type") == "base64" and isinstance(src.get("data"), str):
                out.append(src["data"])
            continue
        m = _DATA_URL.match(url)
        if m:
            out.append(url[m.end():])
    return out


def state_dump(text: str, limit: int) -> dict | None:
    """The newest JSON object in `text` that carries the key "tandem_state" (at most `limit`
    characters of it), or None."""
    i = text.rfind('"tandem_state"')
    while i >= 0:
        j = text.rfind("{", 0, i)
        while j >= 0 and i - j <= limit:
            try:
                obj, end = json.JSONDecoder().raw_decode(text[j:j + limit + 1])
                if isinstance(obj, dict) and "tandem_state" in obj:
                    return obj
            except ValueError:
                pass
            j = text.rfind("{", 0, j)
        i = text.rfind('"tandem_state"', 0, i)
    return None


def state_diff(old: dict | None, new: dict) -> dict:
    """What changed between two state dumps: `changed` key -> [old, new] (scalars, one level
    deep through objects), `gone` the keys that were there and are missing now."""
    out = {"changed": {}, "gone": []}
    if old is None:
        return out

    def walk(a, b, path):
        for k in set(a) | set(b):
            p = f"{path}.{k}" if path else str(k)
            if k not in b:
                out["gone"].append(p)
            elif k not in a:
                out["changed"][p] = [None, _small(b[k])]
            elif isinstance(a[k], dict) and isinstance(b[k], dict) and not path:
                walk(a[k], b[k], p)
            elif a[k] != b[k]:
                out["changed"][p] = [_small(a[k]), _small(b[k])]

    walk(old, new, "")
    out["gone"].sort()
    return out


def _small(v):
    return v if isinstance(v, (int, float, bool)) or v is None else _clip(
        v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), 60)


def faked(script: str) -> str | None:
    """The first member of an undeclared object a script assigns to (`keys['ArrowRight']`), or
    None: a test that sets state instead of sending input. Reads the script's shape only."""
    declared = set()
    for m in _DECL.finditer(script):
        declared.update(g for g in m.groups() if g)
    for m in _ASSIGN.finditer(script):
        if m.group(1) not in declared and m.group(1) not in ("this", "module", "exports"):
            return (m.group(1) + m.group(2))[:60]
    return None


def _scan(text: str, n: int) -> str:
    """The part of a long result the patterns read: its head and its tail."""
    return text if len(text) <= n else text[:n // 4] + "\n" + text[-(n - n // 4):]


def _gap(r: dict) -> float:
    v, lim = r["value"], r["limit"]
    return max(0.0, v - lim) if r["op"] in (">", ">=") else max(0.0, lim - v)


def _cmp(v, op: str, ref) -> bool:
    try:
        return {"<=": v <= ref, "<": v < ref, ">=": v >= ref, ">": v > ref, "==": v == ref,
                "!=": v != ref}[op]
    except (TypeError, KeyError):
        return False


def _calls(messages) -> list[dict]:
    """Every tool call of the history in order, each with its result text and the media that
    followed the results of its step (opencode sends a tool's images as a user message)."""
    out, by_id, step = [], {}, []
    for m in messages or ():
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            step = []
            for i, c in enumerate(m.get("tool_calls") or ()):
                fn = (c or {}).get("function") or {}
                raw = fn.get("arguments")
                raw = raw if isinstance(raw, str) else json.dumps(raw or {})
                cid = str((c or {}).get("id") or "")
                call = {"id": cid, "name": str(fn.get("name") or ""), "raw": raw,
                        "result": None, "media": []}
                call["key"] = cid or "h:" + _h(f"{len(out)}\x00{call['name']}\x00{raw}")
                out.append(call)
                step.append(call)
                if cid:
                    by_id[cid] = call
        elif role == "tool":
            got = by_id.get(str(m.get("tool_call_id") or ""))
            if got is not None:
                got["result"] = signals._text(m.get("content"))
                got["media"] += media_parts(m.get("content"))
        elif role == "user" and step:
            pay = media_parts(m.get("content"))
            if pay and signals._text(m.get("content")).strip() in ("", signals.TOOL_MEDIA_CAPTION):
                # the step's images: the newest call that has a result takes them
                owner = next((c for c in reversed(step) if c["result"] is not None), step[-1])
                owner["media"] += pay
            else:
                step = []
    return out


class Snapshot(dict):
    """One turn's snapshot (a dict, JSON as it is) with the facts the outcome reads."""


def observe(body: dict, sh, ev: Evidence, cfg: dict, *, now: float,
            keep_media: bool = False) -> Snapshot:
    """The snapshot of this request, built from the calls the session made since the last one,
    and the session's evidence moved on. `keep_media`: keep the images' base64 for the writer
    (`_media`, never journaled). The facts per label are `snap["facts"]`."""
    t0 = time.perf_counter()
    lim, pat, rx = cfg["limits"], cfg["patterns"], ev._rx
    run_rx, eval_rx = _rx(cfg["run_tools"], rx), _rx(cfg["evaluate_tools"], rx)
    out_rx = _rx(cfg["output_tools"], rx)
    cons_rx, log_rx = _rx(cfg["console_tools"], rx), _rx(pat["log_tail"], rx)
    err_rx, env_rx, rej_rx = (_rx(pat["error_line"], rx), _rx(pat["environment"], rx),
                              _rx(pat["rejected"], rx))
    sum_rx, lvl_rx = _rx(pat["summary"], rx), _rx(pat["console_level"], rx)
    exit_rx = [_rx(p, rx) for p in pat["exit"]]
    write_tools = set(cfg["write_tools"])
    tools = [t.get("function") for t in body.get("tools") or () if isinstance(t, dict)]
    schemas = {f["name"]: f.get("parameters") or {} for f in tools
               if isinstance(f, dict) and isinstance(f.get("name"), str)}
    first = ev.snapshots == 0
    ev.snapshots += 1
    snap = Snapshot(snapshot=ev.snapshots, calls=[], files=[], last_write=None, last_run=None,
                    shots=[], console=None, checkers=[], exits=[], tests=[], state=None,
                    state_diff=None, logs=[], facts={}, progress=[])
    facts: dict[str, list[str]] = {}

    def fact(label: str, why: str) -> None:
        why = redact.redact(why)[0]            # a why is journaled and may go into a note
        facts.setdefault(label, [])
        if why not in facts[label] and len(facts[label]) < 4:
            facts[label].append(why)

    calls = _calls(body.get("messages"))
    if first and sh.user_index is not None:
        # a session seen for the first time: only what came after the last user message counts
        cut = sum(1 for m in (body.get("messages") or [])[:sh.user_index]
                  if isinstance(m, dict) and m.get("role") == "assistant"
                  for _ in (m.get("tool_calls") or ()))
        for c in calls[:cut]:
            ev.fresh(c["key"])
            ev.writes += c["name"] in write_tools
        calls = calls[cut:]
    new = [c for c in calls if ev.fresh(c["key"])]
    for c in new:
        c["args"] = _args(c["raw"])            # parsed for the new calls only
    console = {"counts": {}, "errors": []}
    for i, c in enumerate(new):
        name, args, raw, text = c["name"], c["args"], c["raw"], c["result"]
        if text is None:
            continue                           # no result yet (a cut history): next time
        is_run = bool(run_rx.search(name))
        row = {"name": name, "args": _show_args(args, raw), "chars": len(text),
               "result": _clip(text, int(lim["result_chars"]))}
        if len(snap["calls"]) >= int(lim["calls"]):
            snap["calls"].pop(0)
        snap["calls"].append(row)
        if schemas and name not in schemas:
            fact("tool_misuse", f"{name} is not one of the offered tools")
        elif schemas and args is not None:
            missing = [k for k in (schemas.get(name) or {}).get("required") or () if k not in args]
            if missing:
                fact("tool_misuse", f"{name} without {', '.join(missing[:3])}")
        elif schemas and args is None and raw.strip() not in ("", "{}"):
            fact("tool_misuse", f"{name}: the arguments are not JSON")
        rejected = rej_rx.search(text[:400])
        if rejected:
            # the client refused the call: its text is the client's, not the program's output
            fact("tool_misuse", f"{name} was rejected: {rejected.group(0)}")
            continue
        if name in write_tools:
            ev.writes += 1
            snap["last_write"] = i
            content = next((args[k] for k in ("content", "newString", "new_string", "patch")
                            if args and isinstance(args.get(k), str)), None)
            path = signals._file_of(raw) or ""
            if len(snap["files"]) < int(lim["files"]):
                snap["files"].append({"path": path, "bytes": len(content or ""),
                                      "hash": _h(content) if content is not None else None})
            if content is not None and not content.strip() and (args or {}).get("content") == "":
                fact("surprise", f"{path or name} was written empty")
            continue
        if is_run:
            snap["last_run"] = i
        scan = _scan(text, int(lim["scan_chars"]))
        if cons_rx.search(name):
            levels = collections.Counter(m.group(1).lower() for m in lvl_rx.finditer(scan))
            for k, v in levels.items():
                console["counts"][k] = console["counts"].get(k, 0) + v
            errs = [ln.strip()[:200] for ln in scan.splitlines()
                    if (m := lvl_rx.match(ln)) and m.group(1).lower() == "error"]
            console["errors"] = (console["errors"] + errs)[-int(lim["console_lines"]):]
            if levels.get("error"):
                n_err = levels["error"]
                fact("error", f"{n_err} console error{'s' if n_err > 1 else ''}: "
                              f"{errs[-1][:120] if errs else ''}")
            continue
        if not is_run:
            continue
        ck = f"{name}\x00{_canon(args, raw)}"
        # exit codes, a test runner's counts, error lines, the environment
        code = None
        for r in exit_rx:
            m = r.search(scan)
            if m:
                code = int(m.group(1))
                break
        if code is not None:
            snap["exits"].append({"call": name, "code": code})
            if code != 0:
                fact("error", f"{name} exited {code}")
        counts = {}
        for m in sum_rx.finditer(scan[-2000:]):
            kind = m.group(2).lower()
            kind = "failed" if kind in ("failed", "failing") else (
                "passed" if kind in ("passed", "passing") else "errors")
            counts[kind] = int(m.group(1))
        if counts:
            snap["tests"].append(dict(counts, call=name))
            failed = counts.get("failed", 0) + counts.get("errors", 0)
            before = ev.failed.get(ck)
            ev.failed[ck] = failed
            if before is not None and failed > before:
                fact("regression", f"{failed} failed, was {before}")
            if failed:
                fact("error", f"{failed} test{'s' if failed > 1 else ''} failed")
            elif counts.get("passed", 0) == 0:
                fact("surprise", "the test run counted 0 tests")
            else:
                snap["progress"].append(f"{counts.get('passed', 0)} passed, 0 failed")
        m = err_rx.search(scan)
        if m and code in (None, 0) and not counts.get("failed"):
            line = scan[m.start():].split("\n", 1)[0].strip()
            fact("error", line[:120])
        if code is None and not counts and out_rx.search(name) and text.strip() not in EMPTY \
                and not log_rx.search(raw):
            # Tier 0 has no exit code and no runner's counts: the judge's `failure` question
            # (step 7), with the error-line pattern as its twin
            snap.setdefault("_narrow", []).append(
                {"q": "failure", "call": name, "text": scan, "twin": "failure" if m else "clean"})
        m = env_rx.search(scan)
        if m:
            fact("environment", f"{name}: {m.group(1)}")
        if eval_rx.search(name) and args:
            script = "\n".join(v for v in args.values() if isinstance(v, str))
            got = faked(script)
            if got:
                fact("test_faked", f"the script sets {got}")
        if log_rx.search(raw):
            snap["logs"].append({"call": name, "tail": "\n".join(
                text.splitlines()[-int(lim["log_lines"]):])[-int(lim["result_chars"]) * 3:]})
            snap.setdefault("_narrow", []).append(
                {"q": "build", "call": name, "text": snap["logs"][-1]["tail"],
                 "twin": "failed" if err_rx.search(snap["logs"][-1]["tail"]) else None})
        # the checker's RESULT lines and the session's best per gate
        for run in signals.checker_results(text) if "RESULT:" in text else ():
            snap["checkers"].append({"checker": run["checker"], "target": run["target"],
                                     "verdict": run["verdict"], "readings": run["readings"]})
            where = f"{run['checker']}:{run['target']}"
            if run["verdict"] == "PASS":
                snap["progress"].append(f"{run['checker']} PASS")
            gates = {r["gate"] for r in run["readings"]}
            for g in [g for g in ev.fails if g.startswith(where + ":") and
                      g.split(":")[-1] not in gates]:
                ev.fails[g] = 0                # that gate passed this time
            for r in run["readings"]:
                gk = f"{where}:{r['gate']}"
                gap, best = _gap(r), ev.best.get(gk)
                ev.fails[gk] = ev.fails.get(gk, 0) + 1
                if best is not None and gap > best + 1e-9:
                    fact("regression", f"gate ({r['gate']}) {r['value']:g} {r['op']} "
                                       f"{r['limit']:g}, best gap was {best:g}")
                elif best is not None and gap < best - 1e-9:
                    snap["progress"].append(f"gate ({r['gate']}) {r['value']:g}")
                ev.best[gk] = gap if best is None else min(best, gap)
                rel = gap / abs(r["limit"]) if r["limit"] else gap
                if rel >= float(cfg["low_quality"]["far"]):
                    fact("low_quality", f"gate ({r['gate']}) {r['value']:g} {r['op']} "
                                        f"{r['limit']:g}")
                if ev.fails[gk] >= int(cfg["low_quality"]["fails"]):
                    fact("low_quality", f"gate ({r['gate']}) failed {ev.fails[gk]} times in a "
                                        f"row")
        # the state dump, and its diff against the previous one
        st = state_dump(scan, len(scan)) if "tandem_state" in scan else None
        if st is not None:
            diff = state_diff(ev.state, st)
            snap["state"] = st if len(json.dumps(st)) <= int(lim["state_bytes"]) else \
                {"tandem_state": st.get("tandem_state"), "clipped": True}
            snap["state_diff"] = diff
            if ev.state is not None and diff["gone"]:
                fact("regression", f"the state lost {', '.join(diff['gone'][:3])}")
            if ev.state is not None and not diff["changed"] and not diff["gone"] and \
                    ev.writes > ev.state_writes:
                fact("no_visible_change", "the same state dump after a write")
            elif diff["changed"]:
                snap["progress"].append(f"state changed: {', '.join(list(diff['changed'])[:3])}")
            for w in cfg["surprise"].get("state") or ():
                v = st.get(w.get("key"))
                if isinstance(v, (int, float)) and _cmp(v, w.get("op", "=="), w.get("value")):
                    fact("surprise", f"{w['key']} is {v}")
            ev.state, ev.state_writes = st, ev.writes
        # the same output of the same call after a write
        out_h = _h(text)
        prev = ev.outputs.get(ck)
        if prev is not None and prev["hash"] == out_h and ev.writes > prev["writes"] and \
                out_rx.search(name) and not c["media"] and text.strip() not in EMPTY:
            fact("no_visible_change", f"{name} returned the same output after a write")
        ev.outputs[ck] = {"hash": out_h, "writes": ev.writes}
        ev.outputs.move_to_end(ck)
        while len(ev.outputs) > OUTPUTS_MAX:
            ev.outputs.popitem(last=False)
    if console["counts"] or console["errors"]:
        snap["console"] = console
    # the images the new calls returned, in order
    for c in new:
        for b64 in c["media"]:
            sha = _h(b64)
            w, hgt = png_size(b64)
            n = len(b64) * 3 // 4
            row = {"sha": sha, "w": w, "h": hgt, "bytes": n, "call": c["name"]}
            if keep_media and n <= int(lim["shot_bytes"]):
                snap.setdefault("_media", []).append((sha, b64))
            snap["shots"].append(row)
            if ev.shot is not None and ev.shot["sha"] == sha and ev.writes > ev.shot["writes"]:
                fact("no_visible_change", "the same screenshot as before the last write")
            elif ev.shot is not None and ev.shot["sha"] != sha and \
                    ev.writes > ev.shot["writes"]:
                snap["progress"].append("a new screenshot after a write")
            ev.shot = {"sha": sha, "writes": ev.writes, "snapshot": ev.snapshots}
    # session-level facts: stuck (no write and no better number), the context after a compaction
    made = any(c["name"] in write_tools for c in new) or any(
        p.startswith("gate (") or p.endswith("PASS") for p in snap["progress"])
    if made:
        ev.progress_at, ev.progress_snap, ev.stuck_noted = now, ev.snapshots, False
    idle_turns = ev.snapshots - ev.progress_snap
    idle_min = (now - ev.progress_at) / 60.0
    sc = cfg["stuck"]
    if not ev.stuck_noted and (idle_turns >= int(sc["turns"]) or idle_min >= float(sc["minutes"])):
        fact("stuck", f"{idle_turns} tool turns and {idle_min:.0f} minutes")
    lc = cfg["lost_context"]
    if ev.compaction is not None and ev.compaction.get("watching") and new:
        ev.compaction["steps"] += len({c["key"] for c in new})
        read = any(c["name"] in lc["read_tools"] and any(f in c["raw"] for f in lc["files"])
                   for c in new)
        if read:
            ev.compaction["watching"] = False
        elif ev.compaction["steps"] >= int(lc["steps"]):
            ev.compaction["watching"] = False
            if any(f in (sh.system_text or "") for f in lc["files"]):
                fact("lost_context", f"{ev.compaction['steps']} calls after the summary read none "
                                     f"of {', '.join(lc['files'])}")
    snap["facts"] = facts
    snap["_limits"] = lim
    snap["new_calls"] = len(new)
    snap["writes"] = ev.writes
    snap["ms"] = round((time.perf_counter() - t0) * 1e3, 3)
    return snap


def _canon(args, raw) -> str:
    if args is None:
        return str(raw)
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def compaction(ev: Evidence) -> None:
    """opencode compacted the session: the calls after it are watched for a re-read."""
    ev.compaction = {"watching": True, "steps": 0}


def classify(snap: Snapshot, ev: Evidence, cfg: dict) -> tuple[str, list[str]]:
    """(the outcome, its evidence): the first label of `order` with facts, else `progress` when
    the snapshot shows progress, else `none`."""
    for label in cfg["order"]:
        if snap["facts"].get(label):
            return label, snap["facts"][label]
    if snap["progress"]:
        return "progress", snap["progress"][:4]
    return "none", []


def settle(ev: Evidence, label: str, why: list[str]) -> str:
    """The session's bookkeeping after an outcome: progress clears the raised effort and the
    notes already given; returns the outcome's signature (a note goes once per signature)."""
    if label == "progress":
        ev.raised, ev.noted = None, set()
    if label == "stuck":
        ev.stuck_noted = True
    sig = f"{label}:{_h(why[0] if why else '')}" if label != "error" else \
        f"{label}:{_h(re.sub(r'[0-9]+', '#', why[0] if why else ''))}"
    ev.last = {"label": label, "why": why[:2], "snapshot": ev.snapshots}
    return sig


def digest(snap: Snapshot, label: str, why: list[str], cfg: dict) -> str:
    """The snapshot for Qwen: fixed rows, numbers first, clipped tails last, within
    `limits.digest_chars`. Images by when they were taken, never their bytes."""
    n = int(cfg["limits"]["digest_chars"])
    rows = [f"Runtime snapshot of your last tool turn (outcome: {label.replace('_', ' ')}"
            + (f": {why[0]}" if why else "") + ")."]
    if snap["files"]:
        rows.append("- files written: " + ", ".join(
            f"{f['path'] or '?'} ({f['bytes']:,} bytes)" for f in snap["files"][:6]))
    for e in snap["exits"][-3:]:
        rows.append(f"- {e['call']} exit code {e['code']}")
    for t in snap["tests"][-2:]:
        rows.append("- tests: " + ", ".join(f"{t[k]} {k}" for k in ("passed", "failed", "errors")
                                            if k in t))
    for c in snap["checkers"][-2:]:
        rd = "; ".join(f"({r['gate']}) {r['value']:g} {r['op']} {r['limit']:g}"
                       for r in c["readings"][:4])
        rows.append(f"- {c['checker']} {c['target']} {c['verdict']}" + (f": {rd}" if rd else ""))
    if snap["shots"]:
        s = snap["shots"][-1]
        size = f" {s['w']}x{s['h']}" if s.get("w") else ""
        same = "no_visible_change" == label and any("screenshot" in w for w in why)
        rows.append(f"- screenshot{size}: " + ("identical to the one before your last change"
                                               if same else "new"))
    if snap["console"]:
        cnt = ", ".join(f"{v} {k}" for k, v in sorted(snap["console"]["counts"].items()))
        rows.append(f"- console: {cnt}")
        for e in snap["console"]["errors"][-3:]:
            rows.append(f"  {e[:160]}")
    if snap["state_diff"] is not None and snap["state"] is not None:
        d = snap["state_diff"]
        ch = ", ".join(f"{k} {a}->{b}" for k, (a, b) in list(d["changed"].items())[:8])
        rows.append("- state: " + (ch or "no change") + (
            f"; missing now: {', '.join(d['gone'][:4])}" if d["gone"] else ""))
    if snap["calls"]:
        rows.append("- calls: " + ", ".join(c["name"] for c in snap["calls"]))
    out = "\n".join(rows)
    tails = []
    for lg in snap["logs"][-1:]:
        tails.append("log tail:\n" + lg["tail"])
    for c in reversed(snap["calls"]):
        if c["chars"] and c["name"] not in cfg["write_tools"]:
            tails.append(f"last result ({c['name']}):\n" + c["result"])
            break
    for t in tails:
        room = n - len(out) - 2
        if room < 120:
            break
        out += "\n\n" + (t if len(t) <= room else t[:room - 6] + "\n[...]")
    # the secret-pattern refusal, as every transcript block: a note is journaled with its text
    return redact.redact(out[:n])[0]


def evidence_text(outcome: dict) -> str:
    """E-b (step 6): the evidence a claim check's note carries, the turn's digest."""
    return digest(outcome["snap"], outcome["label"], outcome["why"], outcome["cfg"])


def facts_row(snap: Snapshot) -> dict:
    """What the journal keeps of a snapshot: counts and hashes, no result text."""
    return {"snapshot": snap["snapshot"], "calls": [c["name"] for c in snap["calls"]],
            "new_calls": snap.get("new_calls"), "files": len(snap["files"]),
            "shots": [s["sha"] for s in snap["shots"]], "exits": snap["exits"],
            "tests": snap["tests"], "console": (snap["console"] or {}).get("counts"),
            "checkers": [f"{c['checker']}:{c['verdict']}" for c in snap["checkers"]],
            "state": snap["state"] is not None,
            "state_changed": sorted((snap["state_diff"] or {}).get("changed") or {})[:8],
            "facts": snap["facts"], "progress": snap["progress"][:4], "ms": snap["ms"]}


def to_json(snap: Snapshot, limit: int) -> bytes:
    """The snapshot file: the snapshot without the images' bytes, cut to `limit` bytes by
    clipping its tails first (results, logs, state)."""
    d = {k: v for k, v in snap.items() if not k.startswith("_")}
    data = json.dumps(d, ensure_ascii=False, indent=1).encode()
    for step in ("logs", "results", "state", "calls"):
        if len(data) <= limit:
            break
        if step == "logs":
            d["logs"] = [dict(lg, tail=lg["tail"][-400:]) for lg in d["logs"]]
        elif step == "results":
            d["calls"] = [dict(c, result=_clip(c["result"], 300)) for c in d["calls"]]
        elif step == "state" and d.get("state") is not None:
            d["state"] = {"tandem_state": d["state"].get("tandem_state"), "clipped": True}
        elif step == "calls":
            d["calls"] = [{"name": c["name"], "chars": c["chars"]} for c in d["calls"]]
        data = json.dumps(d, ensure_ascii=False, indent=1).encode()
    return data[:limit] if len(data) > limit else data


def tool_card(body: dict, cfg: dict, pitfalls: str | None, client: str) -> str:
    """The client tool card (B1): the request's own tools, by name with their required arguments
    (`*`) and the others, which ones take a path, then the client's pitfalls package."""
    cc = cfg["cards"]
    rows, paths = [], set()
    for t in (body.get("tools") or ())[:int(cc["max_tools"])]:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters") or {}
        props = list((params.get("properties") or {}).keys())
        req = set(params.get("required") or ())
        shown = [p + ("*" if p in req else "") for p in props[:int(cc["max_args"])]]
        if len(props) > int(cc["max_args"]):
            shown.append("...")
        rows.append(f"- {fn['name']}({', '.join(shown)})")
        paths |= {p for p in props if p in _PATH_KEYS[:4]}
    head = f"# Your tools ({client})" if client and client != "other" else "# Your tools"
    text = head + "\n" + "\n".join(rows)
    if paths:
        text += f"\n(* required. {', '.join(sorted(paths))}: an absolute path.)"
    if pitfalls:
        text += "\n\n" + pitfalls.strip()
    return text
