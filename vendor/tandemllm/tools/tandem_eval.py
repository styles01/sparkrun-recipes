"""The eval harness (OPS-43): TandemLLM against the raw engine on fixed task suites with checkers.

"Better quality and speed" is proven here or not at all. The same suites run on both arms of the
same server, the same build and the same night: `tandem` sends `model: TandemLLM` under a profile
(`X-Tandem-Profile`, default `overnight`), `raw` sends the same name with `X-Tandem: raw`, which
serves the release's path byte for byte. Arms alternate (raw, tandem, raw, tandem, ...).

    python tools/tandem_eval.py --suites bench/tandem-eval/suites.json --only minicalc,csvq \\
        --n 2 --base https://llm.example.com/v1 --engine https://llm.example.com \\
        --box-lock dgx --publish
    python tools/tandem_eval.py --plan --n 3                   # the schedule, nothing runs
    python tools/tandem_eval.py --only ... --n 3 --night 2026-10-01-r2 --resume ...   # go on
    python tools/tandem_eval.py --report ~/pub-results/tandem-eval/2026-10-01   # rebuild
    python tools/tandem_eval.py --report ~/pub-results/tandem-eval/2026-10-01 --reread \\
        --supersedes suite-2026-10-01-155808.json --supersede-why "..." --publish
    python tools/tandem_eval.py --validate report.json

A suite (bench/tandem-eval/suites.json) is an opencode project: a template folder copied fresh
for every run, a prompt (inline, or the part of a prompt file after `PROMPT START`), the gates
read from the checker's `RESULT:` lines in the session's tool results, an optional final check
the harness runs itself (on the last stage's shot for the image suites, `final_shot`), which has
the last word on its gates, the files that must stay pristine, and a time limit. The `client_smoke`
suite runs tools/client_smoke.py on each arm (the protocol set).

Isolation and honesty, from the August lessons:

  * every run gets a unique tag; its folder name carries it, and opencode writes the folder
    into its system prompt, so no run's conversation can resume from another's prefix;
  * the prefix store is cleared between arms (`POST /v1/cache/clear`, recorded with its answer);
    the suffix store is persistent and shared by both arms, which the alternation balances and
    the report says;
  * checkers, skills and reference files are hashed before the night and after every run; a
    copy inside the run folder (check.py) is compared with the template and restored for the
    final check when the model changed it, and the run says so;
  * the driver only watches (`--driver watch`, the default): when opencode ends before the suite
    is done, it records the continuation it would have sent and the run ends there;
  * n is printed everywhere, and n < 10 is labelled indicative.

Per run (J1: with the conductor's outcome counts per label, `efforts.outcomes`): gates passed,
minutes to each PASS, minutes, tool calls and turns to the run's PASS, wall
minutes, input/output/reasoning tokens, cached share, empty stops, turns, tool calls and errors,
the continuations the driver would have sent,
and from the engine its requests (decode and prefill tok/s) and the conductor's acts, by task,
and gave-ups. Per suite and arm the report's `rows` are what the dashboard's Compare screen
reads; `comparison` says per suite whether the pass rate is not lower and the minutes to PASS
not higher, and the verdict needs the raw engine's never-worse row for the build too (a
`speed_guard` report on the box, or `--raw-row FILE`).

The pass rule (decided 2026-10-02): a run of a suite with a final check passes by that check's
verdict and nothing else; the stage gates the session showed are progress. Its PASS time is the
final gates' minute (the session's first sighting of the PASS, else the shot's time, else the
session's end), and the tool calls and turns to PASS are the session's calls and assistant turns up
to that minute. Every comparison, in the report and in its tables, reads the records'
`minutes_to_pass` (medians over the runs that passed), never the wall minutes, which only say how
long the session ran. A suite without a final check (the protocol set) passes when every gate
does, as before.

A prompt that names a wall-clock deadline ("until 07:15", photo3d) gets the run's own instead: the
suite's `clock` is that time as the prompt writes it, and every run replaces it in its prompt, its
continuation and its `rewrite_files` with its start plus the suite's `timeout_min` (local HH:MM);
the record keeps both (`clock`). Before this, a run that started late in the night had only the
minutes left to 07:15 (photo3d raw r2 of 2026-10-01-r2 stopped itself after 30.9 minutes). A run
whose record has no `clock` (an older night) and whose window holds the suite's clock is flagged
in the report (`clock_hit`): its minutes and its pass are not the task's.

A night survives an interruption (OPS-63): every run's record is written when the run ends, and
`--resume` (same --night, --only, --arms and --first; --n may grow) runs only what the night has no
record for. A record with an `error` is run again (the record is kept in `runs-superseded-<time>/`);
the run that was cut has no record and runs again under a new tag; the prefix store is cleared
before the first resumed run. `<out>/<night>/night.json` keeps the build the night started on and
every resume (when, kept, left, run again); a resume on another build marks the report's build
`changed_during_night`, and the report's `resumed` lists the resumes.

Output: `<out>/<night>/report.json` (schema docs/contract/dashboard-v1/tandem-eval.schema.json),
`report.md` (the table for the Memo note), `runs/<tag>.json`; `--publish` keeps the report on the
box for the Compare screen (`POST /v1/conductor/evals`, the admin token from `--admin-token-env`).
Standard library only; it runs on the machine where opencode and the suites live.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fnmatch
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import signals  # noqa: E402

SUITES = os.path.join(ROOT, "bench", "tandem-eval", "suites.json")
SCHEMA = os.path.join(ROOT, "docs", "contract", "dashboard-v1", "tandem-eval.schema.json")
ARMS = ("raw", "tandem")
INDICATIVE_N = 10
CONTRACT = "1.0"
PASS_RULE = ("a run passes by its suite's final check verdict, the stage gates are progress; "
             "a suite without a final check passes when every gate does")
SUFFIX_NOTE = ("the suffix store is persistent and shared by both arms (not cleared); the arms "
               "alternate so each sees the other's text equally often")


def now_iso(t: float | None = None) -> str:
    return _dt.datetime.fromtimestamp(time.time() if t is None else t,
                                      _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def night_of(t: float) -> str:
    """The night a run belongs to: the local date, the day before until noon."""
    d = _dt.datetime.fromtimestamp(t)
    if d.hour < 12:
        d -= _dt.timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _x(path: str, base: str) -> str:
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


# ------------------------------------------------------------------ suites

def load_suites(path: str = SUITES, only=None, include_disabled: bool = False) -> list[dict]:
    """The suites of a suites file, paths made absolute (relative ones from the file's folder),
    `only` (ids) kept in the file's order. A disabled suite is left out unless named."""
    with open(path) as f:
        doc = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    out = []
    want = [x for x in (only or []) if x]
    for s in doc["suites"]:
        s = dict(s)
        for k in ("id", "title", "kind"):
            if not s.get(k):
                raise ValueError(f"{path}: a suite needs {k}")
        if s["kind"] not in ("opencode", "client_smoke"):
            raise ValueError(f"{path}: suite {s['id']}: kind opencode or client_smoke")
        if want and s["id"] not in want:
            continue
        if not s.get("enabled", True) and not (include_disabled or s["id"] in want):
            continue
        if s["kind"] == "opencode":
            for k in ("template", "timeout_min", "gates"):
                if k not in s:
                    raise ValueError(f"{path}: suite {s['id']} needs {k}")
            s["template"] = _x(s["template"], base)
            if s.get("prompt_file"):
                s["prompt_file"] = _x(s["prompt_file"], base)
            elif not s.get("prompt"):
                raise ValueError(f"{path}: suite {s['id']} needs prompt or prompt_file")
        s["opencode"] = s.get("opencode") or doc.get("opencode") or {}
        out.append(s)
    missing = [x for x in want if x not in [s["id"] for s in out]]
    if missing:
        raise ValueError(f"{path}: no suite {', '.join(missing)}")
    return out


def schedule(suites: list[dict], n: int, first: str = "raw", night: str = "",
             token=lambda: secrets.token_hex(3), only_arms=ARMS) -> list[dict]:
    """Every run of the night: for each repetition, every suite on both arms, the arms
    alternating (raw, tandem, raw, tandem, ... for each suite), each run with a unique tag.
    `only_arms`: a night of one arm (the second profile of an A/B whose raw arm already ran)."""
    arms = [x for x in (ARMS if first == "raw" else ARMS[::-1]) if x in only_arms]
    out, seen = [], set()
    for rep in range(n):
        for s in suites:
            for arm in arms:
                tag = f"te-{night or 'n'}-{s['id']}-{arm}-r{rep}-{token()}"
                while tag in seen:
                    tag = f"te-{night or 'n'}-{s['id']}-{arm}-r{rep}-{token()}"
                seen.add(tag)
                out.append({"suite": s["id"], "arm": arm, "rep": rep, "tag": tag})
    return out


def prompt_of(suite: dict, run_dir: str, clock: str | None = None) -> tuple[str, str]:
    """(the prompt, the continuation text) for a run in `run_dir`: the inline prompt, or the part
    of the prompt file between `prompt_start` and `prompt_end` (the continuation is what follows
    the end marker's line); the suite's `replace` pairs applied and its `rewrite` paths pointed
    at the run's folder."""
    if suite.get("prompt"):
        prompt, cont = suite["prompt"], suite.get("continue") or ""
    else:
        with open(suite["prompt_file"]) as f:
            text = f.read()
        # the markers are lines of their own (the file's first line may name them in text)
        start = suite.get("prompt_start") or ""
        if start:
            m = re.search(rf"^{re.escape(start)}[ \t]*$", text, re.M)
            if m is None:
                raise ValueError(f"{suite['prompt_file']}: no line {start!r}")
            text = text[m.end():]
        cont = suite.get("continue") or ""
        end = suite.get("prompt_end") or ""
        m = re.search(rf"^{re.escape(end)}.*$", text, re.M) if end else None
        if m is not None:
            cont = cont or text[m.end():].strip()
            text = text[:m.start()]
        prompt = text.strip()
    for a, b in suite.get("replace") or ():
        prompt, cont = prompt.replace(a, b), cont.replace(a, b)
    return rewrite(prompt, suite, run_dir, clock), rewrite(cont, suite, run_dir, clock)


def rewrite(text: str, suite: dict, run_dir: str, clock: str | None = None) -> str:
    """The suite's paths pointed at the run's folder, and its wall-clock deadline (`clock`)
    replaced by the run's own when one is given."""
    for p in sorted(suite.get("rewrite") or (), key=len, reverse=True):
        if p.startswith("~/"):
            # the same folder written out under any home (/Users/<name>/..., /home/<name>/...)
            text = re.sub(r"(?:/Users|/home)/[^/\s\"']+/" + re.escape(p[2:]) + r"(?![\w.-])",
                          lambda m: run_dir, text)
        text = text.replace(p, run_dir)
    if clock and suite.get("clock"):
        text = text.replace(suite["clock"], clock)
    return text


def run_clock(suite: dict, t0: float) -> str | None:
    """The run's own deadline as the prompt writes it (local HH:MM): its start plus the suite's
    time budget. None for a suite whose prompt names no clock."""
    if not suite.get("clock"):
        return None
    return time.strftime("%H:%M", time.localtime(t0 + float(suite["timeout_min"]) * 60.0))


def clock_hit(rec: dict, suite: dict) -> dict | None:
    """A run of an older night (no `clock` in its record) whose window held the suite's fixed
    clock: the prompt told the model to stop there, so its minutes and its pass are not the
    task's. {at, minutes_from_start}; None when the clock did not fall inside the run."""
    hhmm = suite.get("clock")
    if not hhmm or rec.get("clock") or rec.get("error") or not rec.get("ended_at"):
        return None
    t0 = _iso_ms(rec["started_at"]) / 1000.0
    t1 = _iso_ms(rec["ended_at"]) / 1000.0
    start = _dt.datetime.fromtimestamp(t0)
    h, m = (int(x) for x in hhmm.split(":"))
    at = start.replace(hour=h, minute=m, second=0, microsecond=0)
    if at <= start:
        at += _dt.timedelta(days=1)
    if at.timestamp() > t1:
        return None
    return {"at": hhmm, "minutes_from_start": round((at.timestamp() - t0) / 60.0, 1)}


# ------------------------------------------------------------------ isolation

def sha(path: str) -> str | None:
    """The hash of a file, or of a folder's files with their relative names; None if absent."""
    if os.path.isfile(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    if os.path.isdir(path):
        h = hashlib.sha256()
        for d, dirs, files in os.walk(path):
            dirs.sort()
            for name in sorted(files):
                p = os.path.join(d, name)
                h.update(os.path.relpath(p, path).encode() + b"\0" + (sha(p) or "").encode())
        return h.hexdigest()
    return None


def pristine_hashes(suite: dict) -> dict:
    """Before the night: every pristine path outside the run folder, and the template."""
    out = {"template:": sha(suite["template"])} if suite.get("template") else {}
    for p in suite.get("pristine") or ():
        if not p.startswith("run:"):
            out[p] = sha(os.path.expanduser(p))
    return out


def pristine_check(suite: dict, before: dict, run_dir: str | None) -> dict:
    """After a run: what changed. Outside paths against the night's hashes; `run:` paths (a
    checker copy in the run folder) against the template's file."""
    changed = []
    for p, h in before.items():
        now = sha(suite["template"]) if p == "template:" else sha(os.path.expanduser(p))
        if now != h:
            changed.append(p)
    for p in suite.get("pristine") or ():
        if p.startswith("run:") and run_dir:
            rel = p[4:]
            if sha(os.path.join(run_dir, rel)) != sha(os.path.join(suite["template"], rel)):
                changed.append(p)
    return {"ok": not changed, "changed": changed}


def restore_run_files(suite: dict, run_dir: str, changed: list[str]) -> list[str]:
    """Put the template's copy back over every `run:` file the model changed (the final check
    then runs the pristine checker); the restored names."""
    out = []
    for p in changed:
        if p.startswith("run:"):
            rel = p[4:]
            src = os.path.join(suite["template"], rel)
            if os.path.isfile(src):
                os.makedirs(os.path.dirname(os.path.join(run_dir, rel)) or run_dir,
                            exist_ok=True)
                shutil.copy2(src, os.path.join(run_dir, rel))
                out.append(rel)
    return out


def prepare(suite: dict, run_dir: str, clock: str | None = None) -> None:
    """The run's folder: the template (all of it, or its `copy` subset) with the suite's paths
    in `rewrite_files` pointed at the folder, and its clock the run's own."""
    os.makedirs(os.path.dirname(run_dir), exist_ok=True)
    src = suite["template"]
    names = suite.get("copy") or sorted(os.listdir(src))
    os.makedirs(run_dir)
    for name in names:
        s, d = os.path.join(src, name), os.path.join(run_dir, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, ignore=shutil.ignore_patterns(".git"))
        elif os.path.exists(s):
            shutil.copy2(s, d)
    for name in suite.get("rewrite_files") or ():
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            with open(p) as f:
                text = f.read()
            with open(p, "w") as f:
                f.write(rewrite(text, suite, run_dir, clock))


def opencode_config(suite: dict, arm: str, args, model_def: dict, base_cfg: dict) -> dict:
    """The run folder's opencode.json: the template's project config (or the suites file's
    default) plus a provider `tandem-eval` on `--base` that sends `X-Tandem: raw` on the raw arm
    and `X-Tandem-Profile` on the other. The key is read from the environment by opencode
    (`{env:TANDEM_EVAL_KEY}`), never written to the folder."""
    cfg = json.loads(json.dumps(base_cfg))
    headers = {"X-Tandem": "raw"} if arm == "raw" else {"X-Tandem-Profile": args.profile}
    cfg.setdefault("$schema", "https://opencode.ai/config.json")
    cfg["provider"] = {"tandem-eval": {
        "npm": "@ai-sdk/openai-compatible", "name": f"TandemLLM eval ({arm})",
        "options": {"baseURL": args.base, "apiKey": "{env:TANDEM_EVAL_KEY}",
                    "headers": headers, "timeout": 7200000, "chunkTimeout": 600000},
        "models": {"TandemLLM": model_def}}}
    cfg["model"] = cfg["small_model"] = "tandem-eval/TandemLLM"
    return cfg


# ------------------------------------------------------------------ reading a run

def export_session(opencode: str, sid: str, env: dict | None = None,
                   timeout_s: float = 120.0) -> tuple[dict, dict]:
    """`opencode export <sid>`: (the export, how it was read: bytes, messages, error).

    The export goes to a file, never a pipe. opencode 1.18.33 (bun) exits before a pipe has taken
    more than 64 KiB of it, so a piped export of any real session ended at 65,536 bytes, did not
    parse and was read as {}: every run of round 1 (2026-10-01) showed 0 tokens, 0 turns and no
    gate. The 12 MB export of a photo3d run arrives whole through a file."""
    with tempfile.TemporaryFile() as out:
        try:
            r = subprocess.run([opencode, "export", sid], stdout=out, stderr=subprocess.PIPE,
                               env=env, timeout=timeout_s)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {}, {"bytes": 0, "messages": None, "error": f"{type(exc).__name__}: {exc}"}
        out.seek(0)
        raw = out.read()
    i = raw.find(b"{")
    err = None
    try:
        exp = json.loads(raw[i:]) if i >= 0 else None
    except ValueError as exc:
        exp, err = None, f"not JSON after {len(raw)} bytes: {exc}"
    if not isinstance(exp, dict) or not isinstance(exp.get("messages"), list):
        tail = r.stderr.decode(errors="replace").strip()[-200:]
        return {}, {"bytes": len(raw), "messages": None,
                    "error": err or f"no session export (exit {r.returncode}): {tail}"}
    return exp, {"bytes": len(raw), "messages": len(exp["messages"]), "error": None}


def _input_text(state: dict) -> str:
    inp = state.get("input")
    return inp if isinstance(inp, str) else json.dumps(inp or {}, sort_keys=True)


def tool_parts(export: dict) -> list[dict]:
    """The session's tool calls in order: name, input text, output, status, end time (ms)."""
    out = []
    for m in export.get("messages") or ():
        if (m.get("info") or {}).get("role") != "assistant":
            continue
        for p in m.get("parts") or ():
            if p.get("type") != "tool":
                continue
            st = p.get("state") or {}
            out.append({"tool": p.get("tool"), "input": _input_text(st),
                        "output": str(st.get("output") or ""), "status": st.get("status"),
                        "end": ((st.get("time") or {}).get("end"))})
    return out


def gate_progress(parts: list[dict], gates: dict, t0_ms: float) -> list[dict]:
    """Each gate of the suite with the minute its first PASS arrived (None: never). A gate is a
    checker run whose call input matches `args` and whose RESULT line is PASS -- or, with
    `unless_failing`, any RESULT line without a failing reading of that name (a stage judged on
    one metric). `ordinal`: the k-th PASS is gate k (stages the call does not name)."""
    total = int(gates.get("total") or len(gates.get("list") or ()))
    spec = gates.get("list") or []
    got = [{"id": g["id"], "passed": False, "minutes": None} for g in spec] if spec else \
        [{"id": f"g{i + 1}", "passed": False, "minutes": None} for i in range(total)]
    k = 0
    for p in parts:
        if p["status"] != "completed":
            continue
        for run in signals.checker_results(p["output"]):
            minute = round((float(p["end"] or t0_ms) - t0_ms) / 60000.0, 2)
            if gates.get("ordinal"):
                if run["verdict"] == "PASS" and k < len(got):
                    got[k].update(passed=True, minutes=minute)
                    k += 1
                continue
            for g, slot in zip(spec, got):
                if slot["passed"] or not re.search(g.get("args") or "", p["input"]):
                    continue
                name = g.get("unless_failing")
                if name:
                    ok = not any(r["name"] == name or r["gate"] == name for r in run["readings"])
                else:
                    ok = run["verdict"] == "PASS"
                if ok:
                    slot.update(passed=True, minutes=minute)
    return got


def read_export(export: dict, suite: dict, t0: float, reader: dict | None = None) -> dict:
    """What the session says: tokens, turns, tool calls and errors, empty stops, how it ended,
    the gates and the minutes. An export that could not be read (`reader` has an error) gives
    None for every count, never 0: a reader failure must not look like an idle run."""
    t0_ms = t0 * 1000.0
    if reader is not None and reader.get("error"):
        gates = gate_progress([], suite.get("gates") or {}, t0_ms)
        return {"tokens": None, "turns": None, "tool_calls": None, "tool_errors": None,
                "empty_stops": 0, "last_finish": None, "wall_minutes": None, "gen_tps": None,
                "cache_hit": None, "gates": gates, "reader": reader, "marks": None}
    tokens = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0}
    turns = empty = 0
    busy_s, last_end, last_finish = 0.0, None, None
    starts = []                       # each assistant turn's start, minutes from the run's start
    for m in export.get("messages") or ():
        info = m.get("info") or {}
        if info.get("role") != "assistant":
            continue
        turns += 1
        tk = info.get("tokens") or {}
        tokens["input"] += int(tk.get("input") or 0)
        tokens["output"] += int(tk.get("output") or 0)
        tokens["reasoning"] += int(tk.get("reasoning") or 0)
        tokens["cache_read"] += int((tk.get("cache") or {}).get("read") or 0)
        tm = info.get("time") or {}
        if tm.get("created"):
            starts.append(round((float(tm["created"]) - t0_ms) / 60000.0, 2))
        if tm.get("created") and tm.get("completed"):
            busy_s += (tm["completed"] - tm["created"]) / 1000.0
            last_end = max(last_end or 0, tm["completed"])
        last_finish = info.get("finish") or last_finish
        parts = m.get("parts") or ()
        has_tool = any(p.get("type") == "tool" for p in parts)
        has_text = any(p.get("type") == "text" and str(p.get("text") or "").strip()
                       for p in parts)
        if info.get("finish") == "stop" and not has_tool and not has_text:
            empty += 1
    parts = tool_parts(export)
    gates = gate_progress(parts, suite.get("gates") or {}, t0_ms)
    gen = tokens["output"] + tokens["reasoning"]
    seen = tokens["input"] + tokens["cache_read"]
    return {"tokens": tokens, "turns": turns, "tool_calls": len(parts),
            "tool_errors": sum(1 for p in parts if p["status"] == "error"),
            "empty_stops": empty, "last_finish": last_finish,
            "wall_minutes": round((last_end - t0_ms) / 60000.0, 2) if last_end else None,
            "gen_tps": round(gen / busy_s, 2) if busy_s > 0 else None,
            "cache_hit": round(tokens["cache_read"] / seen, 4) if seen else None,
            "gates": gates, "reader": reader,
            # when each tool call ended and each turn began, in the gates' minutes: the counts to
            # PASS (`outcome`); not kept in the record
            "marks": {"tools": [round((float(p["end"]) - t0_ms) / 60000.0, 2)
                                for p in parts if p["end"]], "turns": starts}}


def final_shot(suite: dict, run_dir: str) -> str | None:
    """The shot a suite's final check reads, relative to the run folder: the first of the suite's
    `final_shot` patterns that matches a file, without case (the models of round 1 wrote s7.png,
    S7.png and final.png for the same last stage), the newest file when several match. A
    checker's heatmap (`*.check.png`) is never a shot."""
    for pat in suite.get("final_shot") or ():
        sub, name = os.path.split(pat)
        folder = os.path.join(run_dir, sub)
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        hits = [n for n in names if fnmatch.fnmatchcase(n.lower(), name.lower())
                and not n.lower().endswith(".check.png")
                and os.path.isfile(os.path.join(folder, n))]
        if hits:
            best = max(hits, key=lambda n: (os.path.getmtime(os.path.join(folder, n)), n))
            return os.path.join(sub, best) if sub else best
    return None


def final_check(cmd: list[str], cwd: str, timeout_s: float = 300.0,
                shot: str | None = None) -> dict:
    """The suite's own check, run by the harness on the run's folder: its RESULT verdict. In the
    command, `{shot}` is the run's final shot (`final_shot`), `{dir}` the run folder, and a leading
    `~` the home directory. A command that needs a shot when the run left none gives NO_SHOT."""
    if shot is None and any("{shot}" in c for c in cmd):
        return {"verdict": "NO_SHOT", "readings": [], "tail": "", "shot": None}
    argv = [os.path.expanduser(c.replace("{shot}", shot or "").replace("{dir}", cwd)) for c in cmd]
    try:
        r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return {"verdict": "TIMEOUT", "readings": [], "tail": "", "shot": shot}
    except OSError as exc:
        return {"verdict": "ERROR", "readings": [], "tail": str(exc), "shot": shot}
    runs = signals.checker_results(out)
    last = runs[-1] if runs else None
    return {"verdict": last["verdict"] if last else "NONE",
            "readings": last["readings"] if last else [], "tail": out[-600:], "shot": shot,
            "metrics": [{"name": m.group(1).strip(), "value": float(m.group(2)),
                         "op": m.group(3), "limit": float(m.group(4))}
                        for m in _METRIC.finditer(out)]}


# a checker's metric line, `  tone_corr   0.983   (>= 0.85; ...)` or `  de8 coarse colour  9.19
# (<= 10.0; ...)`: the numbers behind a PASS, which its RESULT line does not repeat
_METRIC = re.compile(r"^[ \t]+([A-Za-z_][\w .-]*?)[ \t]+(-?\d+(?:\.\d+)?)[ \t]+\("
                     r"(<=|>=|<|>)[ \t]*(-?\d+(?:\.\d+)?)", re.M)


# a final check that ran decides its gates; NO_SHOT (nothing to check) and ERROR (no checker)
# leave the session's own reading in place
DECIDES = ("PASS", "FAIL", "NONE", "TIMEOUT")


def final_gates(suite: dict) -> list[str]:
    """The ids of the gates a suite's final check decides."""
    return [x["id"] for x in (suite.get("gates") or {}).get("list") or () if x.get("final")]


def apply_final(gates: list[dict], suite: dict, fin: dict | None, run_dir: str, t0: float,
                end_minutes: float | None) -> None:
    """The suite's `final` gates from the final check, the last word on the run: a PASS keeps the
    minute the session first saw it, else the shot's time (or the session's end); anything else
    the checker says fails the gate, whatever the session saw before."""
    if fin is None or fin["verdict"] not in DECIDES:
        return
    finals = set(final_gates(suite))
    for g in gates:
        if g["id"] not in finals:
            continue
        if fin["verdict"] != "PASS":
            g.update(passed=False, minutes=None)
            continue
        if g["minutes"] is None:
            if fin.get("shot"):
                t = os.path.getmtime(os.path.join(run_dir, fin["shot"]))
                g["minutes"] = round(max(0.0, t - t0) / 60.0, 2)
            else:
                g["minutes"] = end_minutes
        g["passed"] = True


def run_passes(gates: list[dict], fin: dict | None) -> bool:
    """The pass rule: the final check's verdict when the suite has one (`fin` is not None), the
    stage gates being progress; every gate, for a suite without one."""
    if fin is not None:
        return fin.get("verdict") == "PASS"
    return bool(gates) and all(g["passed"] for g in gates)


def pass_minutes(gates: list[dict], finals=()) -> float | None:
    """When a passed run passed: the minute of its final gates (`finals`, which apply_final set
    from the final check), else the latest PASS minute of its gates."""
    use = [g for g in gates if g["id"] in finals] or gates
    vals = [g["minutes"] for g in use if g["minutes"] is not None]
    return max(vals) if vals else None


def to_pass(marks: list | None, minutes: float | None) -> int | None:
    """How many of `marks` (tool-call ends, turn starts) came at or before the PASS minute."""
    if marks is None or minutes is None:
        return None
    return sum(1 for m in marks if m <= minutes)


def outcome(got: dict, fin: dict | None, sent: int, timed_out: bool, driver: str,
            finals=()) -> dict:
    """A run's fields from its session reading and final check (the gates already judged).
    `passed` follows the pass rule (`run_passes`): the final check decides a suite that has one.
    `finals`: the suite's final gate ids, whose minute is the PASS minute. The driver needs a
    continuation each time opencode ended before the suite was done, so a run that passed needed
    none: `needed` = the ones sent, plus one when it ended undone in time."""
    gates = {"total": len(got["gates"]), "passed": sum(g["passed"] for g in got["gates"]),
             "by_gate": got["gates"]}
    passed = run_passes(got["gates"], fin)
    minutes = pass_minutes(got["gates"], finals if fin is not None else ()) if passed else None
    marks = got.get("marks") or {}
    return {"gates": gates, "passed": passed, "pass_rule": "final_check" if fin is not None
            else "gates",
            "minutes_to_pass": minutes,
            "tool_calls_to_pass": to_pass(marks.get("tools"), minutes) if passed else None,
            "turns_to_pass": to_pass(marks.get("turns"), minutes) if passed else None,
            "final_check": fin, "tokens": got["tokens"], "cache_hit": got["cache_hit"],
            "gen_tps": got["gen_tps"], "turns": got["turns"], "tool_calls": got["tool_calls"],
            "tool_errors": got["tool_errors"], "empty_stops": got["empty_stops"],
            "last_finish": got["last_finish"], "reader": got.get("reader"),
            "continuations": {"needed": sent + (0 if passed or timed_out else 1), "sent": sent,
                              "driver": driver}}


# ------------------------------------------------------------------ the engine's side

class Api:
    """The engine's HTTP API (reads under the dashboard's rule, writes with the admin token)."""

    def __init__(self, engine: str, admin_token: str | None = None, timeout: float = 20.0):
        self.engine = engine.rstrip("/")
        self.admin = admin_token
        self.timeout = timeout

    def _do(self, method: str, path: str, body=None, admin: bool = False):
        h = {"Content-Type": "application/json"} if body is not None else {}
        if admin and self.admin:
            h["Authorization"] = "Bearer " + self.admin
        req = urllib.request.Request(self.engine + path, method=method, headers=h,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw.strip() else None)
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"null")
            except ValueError:
                return exc.code, None
        except (OSError, ValueError) as exc:
            return None, {"error": f"{type(exc).__name__}: {exc}"}

    def get(self, path: str):
        return self._do("GET", path)

    def post(self, path: str, body=None, admin: bool = True):
        return self._do("POST", path, body if body is not None else {}, admin)


def _iso_ms(ts: str) -> float:
    return _dt.datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=_dt.timezone.utc).timestamp() * 1000.0


def ledger_rows(api: Api, since: float, pages: int = 50) -> list[dict] | None:
    """The engine ledger's requests from now back to `since` (newest first, 200 a page). None
    when the API cannot be read."""
    rows, before = [], None
    for _ in range(pages):
        q = f"/v1/dashboard/requests?limit=200" + (f"&before={before}" if before else "")
        st, got = api.get(q)
        if st != 200 or not isinstance(got, dict):
            return None
        page = got.get("requests") or []
        rows += page
        before = got.get("next_before")
        if not page or not before or _iso_ms(page[-1]["ts"]) < since * 1000.0 - 60_000:
            break
    return rows


def engine_requests(api: Api, t0: float, t1: float, client: str = "opencode",
                    rows: list[dict] | None = None) -> dict | None:
    """The engine's requests of this run (the time window, the client kind): count, finish
    reasons, decode and prefill tok/s medians, tokens. `rows`: the ledger already read (a
    rebuild reads it once for the night). None when the API cannot be read."""
    rows = ledger_rows(api, t0) if rows is None else rows
    if rows is None:
        return None
    rows = [r for r in rows if t0 * 1000.0 - 1000 <= _iso_ms(r["ts"]) <= t1 * 1000.0 + 1000
            and ((r.get("client") or {}).get("kind") == client or not client)]
    def med(k):
        v = [r[k] for r in rows if r.get(k)]
        return round(statistics.median(v), 2) if v else None
    fins: dict = {}
    for r in rows:
        fins[r.get("finish_reason") or "none"] = fins.get(r.get("finish_reason") or "none", 0) + 1
    return {"requests": len(rows), "finish": fins, "decode_tps_median": med("decode_tps"),
            "prefill_tps_median": med("prefill_tps"),
            "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
            "reasoning_tokens": sum(int(r.get("reasoning_tokens") or 0) for r in rows),
            "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
            "cached_tokens": sum(int(r.get("cached_tokens") or 0) for r in rows)}


def conductor_events(api: Api, session_id: str) -> dict | None:
    """The conductor's journal for the session: acts by task, gave-ups, shadow decisions."""
    st, got = api.get(f"/v1/conductor/events?session=c:{session_id}&limit=2000")
    if st != 200 or not isinstance(got, dict):
        return None
    acts, shadow, by = 0, 0, {}
    gave = 0
    for e in got.get("events") or ():
        if e.get("mode") == "act" and e.get("decision") not in (None, "PASS"):
            acts += 1
            by[e.get("task") or "?"] = by.get(e.get("task") or "?", 0) + 1
            gave += e.get("decision") == "GIVE_UP"
        elif e.get("mode") == "shadow" and e.get("decision") not in (None, "PASS"):
            shadow += 1
    return {"acts": acts, "by_task": by, "gave_ups": gave, "shadow": shadow,
            "efforts": conductor_efforts(api, session_id)}


def conductor_efforts(api: Api, session_id: str) -> dict | None:
    """The router's effort per turn (SRV-157): turns per effort level, the task sizes it saw,
    whether the size set the effort (`size_mode` on) or was only journaled (shadow), and the
    efforts the size picked."""
    st, got = api.get(f"/v1/conductor/decisions?session=c:{session_id}&limit=500")
    if st != 200 or not isinstance(got, dict):
        return None
    by_level, by_size, picked, outs = {}, {}, {}, {}
    mode, applied = None, 0
    for r in got.get("decisions") or ():
        if r.get("kind", "route") != "route":
            continue
        # J1: the Tier 0 outcome of each tool-result turn (shadow or act), by label
        out = (r.get("extra") or {}).get("outcome")
        if out and out.get("label"):
            outs[out["label"]] = outs.get(out["label"], 0) + 1
        lv = r.get("effort") or "unchanged"
        by_level[lv] = by_level.get(lv, 0) + 1
        size = (r.get("extra") or {}).get("size")
        if size:
            by_size[size["size"]] = by_size.get(size["size"], 0) + 1
            mode = size.get("mode") or mode
            applied += bool(size.get("applied"))
            if size.get("effort"):
                picked[size["effort"]] = picked.get(size["effort"], 0) + 1
    return {"by_level": by_level, "by_size": by_size, "size_mode": mode, "size_applied": applied,
            "size_effort": picked, "outcomes": outs}


def _sum_counts(dicts) -> dict | None:
    out = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] = out.get(k, 0) + v
    return out or None


def build_of(api: Api) -> dict:
    st, got = api.get("/v1/conductor/evals")
    eng = (got or {}).get("engine") if st == 200 else None
    return {"version": (eng or {}).get("version"), "code_sha": (eng or {}).get("code_sha")}


def raw_row(api: Api | None, build: dict, path: str | None) -> dict:
    """The raw engine's never-worse row for the build: a file given on the command line, else
    the box's `speed_guard` report for this build, else missing."""
    if path:
        with open(path) as f:
            return {"status": "attached", "source": os.path.basename(path), "row": json.load(f)}
    if api is not None:
        st, got = api.get("/v1/conductor/evals")
        for r in (got or {}).get("reports") or () if st == 200 else ():
            b = r.get("build") or {}
            if r.get("kind") == "speed_guard" and (
                    (build.get("code_sha") and b.get("code_sha") == build["code_sha"]) or
                    (not build.get("code_sha") and b.get("version") == build.get("version"))):
                return {"status": "found", "source": r.get("file"), "row": r}
    return {"status": "missing", "source": None, "row": None}


# ------------------------------------------------------------------ the report

def _med(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def suite_rows(runs: list[dict], suites: list[dict], build: str, night: str) -> list[dict]:
    """Per suite and arm, what the Compare screen shows (medians over the runs, sums where a
    count is meant)."""
    out = []
    for s in suites:
        for arm in ARMS:
            rs = [r for r in runs if r["suite"] == s["id"] and r["arm"] == arm and not r.get("error")]
            if not rs:
                continue
            n = len(rs)
            passed = [r for r in rs if r["passed"]]
            total = max((r["gates"]["total"] for r in rs), default=0)
            out.append({
                "suite": s["title"], "suite_id": s["id"], "arm": arm, "n": n,
                "runs_passed": len(passed), "pass_rate": round(len(passed) / n, 3),
                "gates_passed": round(sum(r["gates"]["passed"] for r in rs) / n, 2),
                "gates_total": total,
                "minutes_to_pass": _med([r["minutes_to_pass"] for r in passed]),
                # the effort to PASS beside its time (2026-10-02): a record from before has none
                "tool_calls_to_pass": _med([r.get("tool_calls_to_pass") for r in passed]),
                "turns_to_pass": _med([r.get("turns_to_pass") for r in passed]),
                "wall_minutes": _med([r["wall_minutes"] for r in rs]),
                "tokens": _med([r["tokens"]["output"] + r["tokens"]["reasoning"] for r in rs
                                if r.get("tokens")]),
                "reasoning_tokens": _med([r["tokens"]["reasoning"] for r in rs
                                          if r.get("tokens")]),
                "turns": _med([r.get("turns") for r in rs]),
                "decode_tps": _med([(r.get("engine") or {}).get("decode_tps_median")
                                    for r in rs]),
                "gen_tps": _med([r.get("gen_tps") for r in rs]),
                "cache_hit": _med([r.get("cache_hit") for r in rs]),
                "empty_stops": sum(r.get("empty_stops") or 0 for r in rs),
                "continuations": sum(r["continuations"]["needed"] for r in rs),
                "interventions": sum((r.get("conductor") or {}).get("acts") or 0 for r in rs),
                "gave_ups": sum((r.get("conductor") or {}).get("gave_ups") or 0 for r in rs),
                # SRV-157: turns per effort level over the runs, and the size mode they ran under
                "efforts": _sum_counts(((r.get("conductor") or {}).get("efforts") or {})
                                       .get("by_level") for r in rs),
                # J1: the outcome of every tool-result turn, counted over the runs
                "outcomes": _sum_counts(((r.get("conductor") or {}).get("efforts") or {})
                                        .get("outcomes") for r in rs),
                "size_mode": next((((r.get("conductor") or {}).get("efforts") or {})
                                   .get("size_mode") for r in rs
                                   if ((r.get("conductor") or {}).get("efforts") or {})
                                   .get("size_mode")), None),
                "pristine": all(r["pristine"]["ok"] for r in rs),
                "build": build, "night": night, "indicative": n < INDICATIVE_N})
    return out


def compare(rows: list[dict]) -> list[dict]:
    """Per suite: is the pass rate not lower and the minutes to PASS not higher on TandemLLM? The
    minutes are the rows' `minutes_to_pass` (the records' own, medians over the runs that passed),
    never the wall minutes; the tool calls and turns to PASS ride along, they decide nothing."""
    out = []
    for sid in dict.fromkeys(r["suite_id"] for r in rows):
        raw = next((r for r in rows if r["suite_id"] == sid and r["arm"] == "raw"), None)
        tan = next((r for r in rows if r["suite_id"] == sid and r["arm"] == "tandem"), None)
        c = {"suite_id": sid, "suite": (raw or tan)["suite"],
             "n": {"raw": raw["n"] if raw else 0, "tandem": tan["n"] if tan else 0}}
        if not raw or not tan:
            c.update(verdict="incomplete", why="one arm has no run", pass_rate_not_lower=None,
                     minutes_not_higher=None, indicative=True)
            out.append(c)
            continue
        prl = tan["pass_rate"] >= raw["pass_rate"]
        mr, mt = raw["minutes_to_pass"], tan["minutes_to_pass"]
        if mr is None and mt is None:
            mnh, mwhy = None, "neither arm passed"
        elif mt is None:
            mnh, mwhy = False, "only raw passed"
        elif mr is None:
            mnh, mwhy = True, "only TandemLLM passed"
        else:
            mnh, mwhy = mt <= mr, f"{mt:g} against {mr:g} minutes"
        if prl and mnh is not False and (mnh or tan["pass_rate"] > raw["pass_rate"]):
            verdict = "better" if (tan["pass_rate"] > raw["pass_rate"] or (mnh and mt != mr)) \
                else "equal"
        elif not prl and mnh is False:
            verdict = "worse"
        elif prl and mnh is None:
            verdict = "equal"
        else:
            verdict = "mixed"
        calls = {"raw": raw.get("tool_calls_to_pass"), "tandem": tan.get("tool_calls_to_pass")}
        turns = {"raw": raw.get("turns_to_pass"), "tandem": tan.get("turns_to_pass")}
        cwhy = (f"; {calls['tandem']:g} against {calls['raw']:g} tool calls to PASS"
                if None not in calls.values() else "")
        c.update(verdict=verdict, pass_rate_not_lower=prl, minutes_not_higher=mnh,
                 why=f"pass rate {tan['pass_rate']:.2f} against {raw['pass_rate']:.2f}; {mwhy}"
                     f"{cwhy}",
                 minutes_to_pass={"raw": mr, "tandem": mt}, tool_calls_to_pass=calls,
                 turns_to_pass=turns, indicative=min(raw["n"], tan["n"]) < INDICATIVE_N)
        out.append(c)
    return out


def verdict(comparison: list[dict], row: dict) -> dict:
    """Better = pass rate not lower and minutes to PASS not higher on every suite, and the raw
    row unchanged. Anything short of that says what is missing or worse."""
    ind = any(c["indicative"] for c in comparison) or not comparison
    if not comparison:
        return {"better": None, "why": "no suite ran on both arms", "indicative": True}
    bad = [c for c in comparison if c["verdict"] in ("worse", "mixed")]
    inc = [c for c in comparison if c["verdict"] == "incomplete"]
    if bad:
        return {"better": False, "indicative": ind,
                "why": "; ".join(f"{c['suite']}: {c['verdict']} ({c['why']})" for c in bad)}
    if inc:
        return {"better": None, "indicative": ind,
                "why": "; ".join(f"{c['suite']}: {c['why']}" for c in inc)}
    if row["status"] == "missing":
        return {"better": None, "indicative": ind,
                "why": "every suite is not worse, but the raw engine's never-worse row for this "
                       "build is missing"}
    verdict_ = (row.get("row") or {}).get("verdict")
    if verdict_ == "no_noise_estimate":
        # tools/speed_guard.py on a short gate (one run a side): the row decides nothing
        return {"better": None, "indicative": ind,
                "why": "every suite is not worse, but the raw engine's never-worse row for this "
                       "build is one run a side: its noise is unknown, so it says neither pass "
                       "nor fail"}
    if verdict_ not in (None, "pass", "PASS"):
        return {"better": False, "indicative": ind, "why": f"the raw row reads {verdict_}"}
    return {"better": True, "indicative": ind,
            "why": "pass rate not lower and minutes to PASS not higher on every suite; the raw "
                   "row is unchanged" + (" (indicative: n < 10)" if ind else "")}


def build_report(meta: dict, runs: list[dict], suites: list[dict], row: dict) -> dict:
    by_id = {s["id"]: s for s in suites}
    flagged = []
    for r in runs:
        hit = clock_hit(r, by_id.get(r["suite"]) or {})
        flagged.append(dict(r, clock_hit=hit) if hit else r)
    runs = flagged
    build = meta["build"].get("code_sha") or meta["build"].get("version") or "unknown"
    rows = suite_rows(runs, suites, build, meta["night"])
    comp = compare(rows)
    return {"kind": "suite", "contract_version": CONTRACT, "harness": "tools/tandem_eval.py",
            "night": meta["night"], "created_at": now_iso(), "build": meta["build"],
            "base": meta["base"], "profile": meta["profile"], "driver": meta["driver"],
            "isolation": {"unique_tags": len({r["tag"] for r in runs}) == len(runs),
                          "store_clears": [r["store_clear"] for r in runs if r.get("store_clear")],
                          "suffix_store": SUFFIX_NOTE},
            "schedule": [{k: r[k] for k in ("suite", "arm", "rep", "tag", "started_at")}
                         for r in runs],
            "pass_rule": PASS_RULE,
            "runs": runs, "rows": rows, "comparison": comp, "raw_row": row,
            "verdict": verdict(comp, row), "indicative_below": INDICATIVE_N}


def efforts(by_level: dict | None, size_mode: str | None = None) -> str:
    """`high 1, xhigh 6 (size on)`: turns per effort level, low to high."""
    if not by_level:
        return "—"
    order = ("off", "low", "medium", "high", "xhigh", "unchanged")
    out = ", ".join(f"{k} {by_level[k]}" for k in sorted(by_level, key=lambda k: (
        order.index(k) if k in order else 99, k)))
    return out + (f" (size {size_mode})" if size_mode else "")


def to_md(rep: dict) -> str:
    """The report as the Memo note's table, n printed, indicative said."""
    v = rep["verdict"]
    lines = [f"# TandemLLM against raw, night {rep['night']}", "",
             f"Build {rep['build'].get('version') or '?'} ({rep['build'].get('code_sha') or '?'}), "
             f"profile `{rep['profile']}`, driver {rep['driver']['mode']}, {len(rep['runs'])} runs."
             + (" Indicative: n < 10." if v["indicative"] else ""), "",
             f"Verdict: {'better' if v['better'] else 'not shown' if v['better'] is None else 'not better'}"
             f" — {v['why']}", "",
             f"Pass rule: {rep.get('pass_rule') or 'every gate (a report from before 2026-10-02)'}."
             " Minutes, tool calls and turns to PASS are the records' own, medians over the runs "
             "that passed; wall minutes are how long the session ran and compare nothing.", "",
             "| suite | arm | n | pass rate | gates | minutes to PASS | tool calls to PASS | "
             "turns to PASS | wall min | turns | tokens | thinking tokens | decode tok/s | "
             "empty stops | driver continuations | interventions | gave up | efforts (turns) |",
             "|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|"]
    def f(x):
        return "—" if x is None else f"{x:g}" if isinstance(x, (int, float)) else str(x)
    for r in rep["rows"]:
        lines.append(f"| {r['suite']} | {r['arm']} | {r['n']} | {r['pass_rate']:.2f} | "
                     f"{f(r['gates_passed'])}/{r['gates_total']} | {f(r['minutes_to_pass'])} | "
                     f"{f(r.get('tool_calls_to_pass'))} | {f(r.get('turns_to_pass'))} | "
                     f"{f(r['wall_minutes'])} | {f(r.get('turns'))} | {f(r['tokens'])} | "
                     f"{f(r['reasoning_tokens'])} | {f(r['decode_tps'])} | "
                     f"{r['empty_stops']} | {r['continuations']} | {r['interventions']} | "
                     f"{r['gave_ups']} | {efforts(r.get('efforts'), r.get('size_mode'))} |")
    lines += ["", "Per suite (TandemLLM against raw):", "",
              "| suite | verdict | minutes to PASS | tool calls to PASS | turns to PASS | why |",
              "|-|-|-|-|-|-|"]
    for c in rep["comparison"]:
        def pair(k, c=c):
            v = c.get(k) or {}
            return f"{f(v.get('tandem'))} / {f(v.get('raw'))}" if v else "—"
        lines.append(f"| {c['suite']} | {c['verdict']} | {pair('minutes_to_pass')} | "
                     f"{pair('tool_calls_to_pass')} | {pair('turns_to_pass')} | {c.get('why')} |")
    lines += ["", "Per run:", "",
              "| run | gates | final check | minutes to PASS | tool calls to PASS | turns to PASS | "
              "wall min | turns | input / output / reasoning / cached tokens | empty stops | "
              "continuations needed | engine requests | decode tok/s |",
              "|-|-|-|-|-|-|-|-|-|-|-|-|-|"]
    for r in rep["runs"]:
        if r.get("error"):
            lines.append(f"| {r['tag']} | error: {r['error']} |||||||||||| ")
            continue
        fin = r.get("final_check") or {}
        reads = ", ".join(f"{x['name'] or x.get('gate')} {x['value']:g}"
                          for x in (fin.get("readings") or fin.get("metrics") or [])[:4])
        fc = (f"{fin['verdict']}" + (f" on {fin['shot']}" if fin.get("shot") else "")
              + (f" ({reads})" if reads else "")) if fin else "—"
        tk = r.get("tokens")
        err = (r.get("reader") or {}).get("error")
        tks = (f"{tk['input']:,} / {tk['output']:,} / {tk['reasoning']:,} / {tk['cache_read']:,}"
               if tk else f"session not read ({err[:80]})" if err else "—")
        eng = r.get("engine") or {}
        lines.append(f"| {r['tag']} | {r['gates']['passed']}/{r['gates']['total']} | {fc} | "
                     f"{f(r.get('minutes_to_pass'))} | {f(r.get('tool_calls_to_pass'))} | "
                     f"{f(r.get('turns_to_pass'))} | {f(r.get('wall_minutes'))} | "
                     f"{f(r.get('turns'))} | {tks} | {r.get('empty_stops')} | "
                     f"{r['continuations']['needed']} | {f(eng.get('requests'))} | "
                     f"{f(eng.get('decode_tps_median'))} |")
    for x in rep.get("resumed") or ():
        lines += ["", f"Resumed at {x['at']}: {x['kept']} runs kept, {x['left']} run then"
                  + (f", {len(x['run_again'])} errored run again" if x["run_again"] else "")
                  + (", on another build" if x["build_changed"] else "") + "."]
    if rep.get("supersedes"):
        lines += ["", f"Supersedes {rep['supersedes']['file']}"
                  + (f": {rep['supersedes']['why']}" if rep["supersedes"].get("why") else "")]
    rr = rep["raw_row"]
    lines += ["", f"Raw row: {rr['status']}" + (f" ({rr['source']})" if rr.get("source") else ""),
              f"Isolation: unique tags {rep['isolation']['unique_tags']}, "
              f"{len(rep['isolation']['store_clears'])} store clears; "
              f"{rep['isolation']['suffix_store']}."]
    for r in rep["runs"]:
        if r.get("clock_hit"):
            lines.append(f"Clock: {r['tag']} ran into its prompt's fixed clock "
                         f"{r['clock_hit']['at']} {r['clock_hit']['minutes_from_start']:g} minutes "
                         f"after its start; its minutes and its pass are not the task's.")
    bad = [r for r in rep["runs"] if not r["pristine"]["ok"]]
    if bad:
        lines.append("Pristine: changed in " + ", ".join(f"{r['tag']} ({', '.join(r['pristine']['changed'])})"
                                                          for r in bad))
    return "\n".join(lines) + "\n"


def validate(rep: dict) -> list[str]:
    from tools import contract_check
    with open(SCHEMA) as f:
        schema = json.load(f)
    return contract_check.validate(rep, schema)


# ------------------------------------------------------------------ running

def run_process(argv: list[str], cwd: str, env: dict, timeout_s: float, log: str):
    """Run one command to its end or the timeout (the whole process group is stopped then):
    (exit code, timed out, the stdout lines)."""
    # opencode takes its project directory, and with it the run folder's opencode.json, from $PWD,
    # not from the process's working directory: inherited from the harness, PWD named the checkout
    # and every run failed with "Model not found: tandem-eval/TandemLLM" (2026-10-01, 14:20)
    env = dict(env, PWD=os.path.abspath(cwd))
    with open(log, "ab") as lf:
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=lf,
                             start_new_session=True)
        try:
            out, _ = p.communicate(timeout=max(1.0, timeout_s))
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    out, _ = p.communicate(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
            else:
                out = b""
        lf.write(out or b"")
    return p.returncode, timed_out, (out or b"").decode(errors="replace").splitlines()


def session_id(lines: list[str]) -> str | None:
    """The session id from `opencode run --format json`'s events."""
    for line in lines:
        m = re.search(r'"sessionID"\s*:\s*"(ses_[A-Za-z0-9]+)"', line)
        if m:
            return m.group(1)
    return None


def free_ports(ports) -> list[int]:
    """Stop what still listens on a suite's ports (the model's `python3 -m http.server`), so the
    next run's server serves the next run's folder. Only an http.server is stopped."""
    killed = []
    for port in ports or ():
        try:
            pids = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                                  capture_output=True, text=True, timeout=10).stdout.split()
        except (OSError, subprocess.TimeoutExpired):
            continue
        for pid in pids:
            cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True,
                                 text=True).stdout
            if "http.server" in cmd:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    killed.append(int(pid))
                except (ProcessLookupError, ValueError):
                    pass
    return killed


class BoxLock:
    """The box's lock (`flock ~/.qwen38-box.flock`) held over ssh for the night: refused while a
    hold runs (`pgrep -fa hold.sh`), taken with a timeout, released when the harness ends."""

    def __init__(self, host: str, path: str = "~/.qwen38-box.flock", wait_s: int = 600):
        self.host, self.path, self.wait_s = host, path, wait_s
        self.proc = None

    def argv(self) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", self.host,
                f"pgrep -fa '[h]old.sh' && exit 3; flock -w {int(self.wait_s)} {self.path} "
                f"-c 'echo locked; cat >/dev/null'"]

    def __enter__(self):
        self.proc = subprocess.Popen(self.argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True)
        line = self.proc.stdout.readline().strip()
        if line != "locked":
            self.proc.wait(30)
            raise SystemExit(f"[eval] the box lock on {self.host} was not taken "
                             f"({'a hold runs' if self.proc.returncode == 3 else line or 'timeout'})")
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(30)


class Night:
    """One night of runs: the schedule, each run's record, the report."""

    def __init__(self, args, suites: list[dict], api: Api | None, env: dict, model_def: dict):
        self.a, self.suites, self.api, self.env, self.model_def = args, suites, api, env, model_def
        self.by_id = {s["id"]: s for s in suites}
        self.night = args.night or night_of(time.time())
        self.dir = os.path.join(os.path.expanduser(args.out), self.night)
        self.work = os.path.join(os.path.expanduser(args.work), self.night)
        self.last_arm: str | None = None
        self.hashes = {s["id"]: pristine_hashes(s) for s in suites if s["kind"] == "opencode"}

    def records(self) -> list[dict]:
        """The run records already written for this night, oldest first."""
        folder = os.path.join(self.dir, "runs")
        out = []
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else ():
            if name.endswith(".json"):
                with open(os.path.join(folder, name)) as f:
                    out.append(json.load(f))
        return sorted(out, key=lambda r: r.get("started_at") or "")

    def start(self, build: dict) -> dict:
        """night.json at a night's start: the build it runs on. Kept when it exists already (a
        second invocation on the same night), so the night's own build stays the first one."""
        path = os.path.join(self.dir, "night.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        doc = {"night": self.night, "build": build, "started_at": now_iso(),
               "suites": sorted(self.by_id), "profile": self.a.profile, "resumes": []}
        os.makedirs(self.dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
        return doc

    def resume(self, plan: list[dict], build: dict) -> tuple[list[dict], list[dict], dict]:
        """`--resume`: (the records kept, the schedule's items still to run, night.json). An item
        is done when a record of its suite, arm and rep exists without an error; an errored
        record is moved to runs-superseded-<time>/ and its item runs again, under a new tag."""
        recs = self.records()
        if not recs and not os.path.exists(os.path.join(self.dir, "night.json")):
            raise SystemExit(f"[eval] --resume: nothing to resume in {self.dir}")
        doc = self.start(build)
        done, again = {}, []
        for r in recs:
            key = (r.get("suite"), r.get("arm"), r.get("rep"))
            if r.get("error"):
                again.append(r)
            elif key not in done:
                done[key] = r
        if again:
            old = os.path.join(self.dir, f"runs-superseded-{time.strftime('%Y%m%d-%H%M%S')}")
            os.makedirs(old, exist_ok=True)
            for r in again:
                os.replace(os.path.join(self.dir, "runs", r["tag"] + ".json"),
                           os.path.join(old, r["tag"] + ".json"))
        left = [it for it in plan if (it["suite"], it["arm"], it["rep"]) not in done]
        kept = [r for r in done.values() if r["suite"] in self.by_id]
        info = {"at": now_iso(), "kept": len(kept), "left": len(left),
                "run_again": [r["tag"] for r in again],
                "build_changed": (doc.get("build") or {}) != build}
        doc["resumes"] = [*doc.get("resumes", []), info]
        with open(os.path.join(self.dir, "night.json"), "w") as f:
            json.dump(doc, f, indent=1)
        return kept, left, doc

    def clear_store(self, arm: str) -> dict | None:
        """Between arms, the prefix store goes (recorded with the server's answer)."""
        if not self.a.clear_store or self.api is None or arm == self.last_arm:
            return None
        st, got = self.api.post("/v1/cache/clear")
        return {"at": now_iso(), "status": st,
                "error": None if st == 200 else str((got or {}).get("error") or got)[:200]}

    def run(self, item: dict) -> dict:
        s = self.by_id[item["suite"]]
        t0 = time.time()
        rec = {"suite": s["id"], "arm": item["arm"], "rep": item["rep"], "tag": item["tag"],
               "started_at": now_iso(t0), "error": None,
               "store_clear": self.clear_store(item["arm"])}
        self.last_arm = item["arm"]
        try:
            if s["kind"] == "client_smoke":
                rec.update(self._smoke(s, item, t0))
            else:
                rec.update(self._opencode(s, item, t0))
        except Exception as exc:                                   # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["ended_at"] = now_iso()
        os.makedirs(os.path.join(self.dir, "runs"), exist_ok=True)
        with open(os.path.join(self.dir, "runs", item["tag"] + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        return rec

    def _opencode(self, s: dict, item: dict, t0: float) -> dict:
        a = self.a
        run_dir = os.path.join(self.work, item["tag"])
        clock = run_clock(s, t0)
        prepare(s, run_dir, clock)
        base_cfg = {}
        tpl_cfg = os.path.join(run_dir, "opencode.json")
        if os.path.exists(tpl_cfg):
            with open(tpl_cfg) as f:
                base_cfg = json.load(f)
        else:
            base_cfg = s["opencode"]
        with open(tpl_cfg, "w") as f:
            json.dump(opencode_config(s, item["arm"], a, self.model_def, base_cfg), f, indent=1)
        prompt, cont = prompt_of(s, run_dir, clock)
        deadline = t0 + float(s["timeout_min"]) * 60.0
        log = os.path.join(run_dir, ".tandem-eval.log")
        argv = [a.opencode, "run", "--format", "json", "--model", "tandem-eval/TandemLLM",
                "--title", item["tag"], prompt]
        code, timed_out, lines = run_process(argv, run_dir, self.env, deadline - time.time(), log)
        sid = session_id(lines)
        if sid is None:
            raise RuntimeError(f"opencode gave no session (exit {code}, see {log})")
        sent, changed, restored = 0, [], []
        while True:
            # the run as it stands when opencode ended: the session, then the final check on the
            # pristine checker (a `run:` file the model changed is put back first)
            export, reader = export_session(a.opencode, sid, self.env)
            got = read_export(export, s, t0, reader)
            prist = pristine_check(s, self.hashes[s["id"]], run_dir)
            changed += [p for p in prist["changed"] if p not in changed]
            restored += [p for p in restore_run_files(s, run_dir, prist["changed"])
                         if p not in restored]
            fin = final_check(s["final_check"], run_dir, shot=final_shot(s, run_dir)) \
                if s.get("final_check") else None
            apply_final(got["gates"], s, fin, run_dir, t0, got["wall_minutes"]
                        if got["wall_minutes"] is not None else round((time.time() - t0) / 60, 2))
            if (run_passes(got["gates"], fin) or timed_out or a.driver != "continue"
                    or not cont or sent >= a.max_continues or time.time() >= deadline):
                break
            sent += 1                         # the driver sends the continuation
            code, timed_out, _ = run_process(
                [a.opencode, "run", "--format", "json", "--session", sid, cont], run_dir,
                self.env, deadline - time.time(), log)
        free_ports(s.get("ports"))
        t1 = time.time()
        eng = engine_requests(self.api, t0, t1) if self.api is not None else None
        cond = conductor_events(self.api, sid) if self.api is not None and sid else None
        return {"dir": run_dir, "session_id": sid, "exit_code": code, "timed_out": timed_out,
                **({"clock": {"suite": s["clock"], "run": clock}} if clock else {}),
                "wall_minutes": got["wall_minutes"] if got["wall_minutes"] is not None
                else round((t1 - t0) / 60.0, 2),
                **outcome(got, fin, sent, timed_out, a.driver, final_gates(s)),
                "engine": eng, "conductor": cond,
                "pristine": {"ok": not changed, "changed": changed, "restored": restored}}

    def _smoke(self, s: dict, item: dict, t0: float) -> dict:
        out = os.path.join(self.work, item["tag"] + ".json")
        os.makedirs(self.work, exist_ok=True)
        argv = [*self.a.smoke_python, os.path.join(ROOT, "tools", "client_smoke.py"),
                "--base", self.a.base, "--json", out, *(s.get("args") or [])]
        if item["arm"] == "raw":
            argv += ["--x-tandem", "raw"]
        env = dict(self.env, QSE_API_KEY=self.env.get("TANDEM_EVAL_KEY", ""))
        code, timed_out, _ = run_process(argv, ROOT, env, float(s.get("timeout_min") or 30) * 60,
                                         out + ".log")
        try:
            with open(out) as f:
                res = json.load(f)["results"]
        except (OSError, ValueError, KeyError):
            res = []
        n_ok = sum(1 for r in res if r.get("pass"))
        t1 = time.time()
        return {"dir": None, "session_id": None, "exit_code": code, "timed_out": timed_out,
                "wall_minutes": round((t1 - t0) / 60.0, 2),
                "gates": {"total": len(res), "passed": n_ok,
                          "by_gate": [{"id": r.get("check"), "passed": bool(r.get("pass")),
                                       "minutes": None} for r in res]},
                "passed": bool(res) and n_ok == len(res), "pass_rule": "gates",
                "minutes_to_pass": round((t1 - t0) / 60.0, 2) if res and n_ok == len(res)
                else None, "tool_calls_to_pass": None, "turns_to_pass": None,
                "final_check": None, "tokens": None, "cache_hit": None, "gen_tps": None,
                "turns": None, "tool_calls": None, "tool_errors": None, "empty_stops": 0,
                "last_finish": None, "continuations": {"needed": 0, "sent": 0,
                                                       "driver": self.a.driver},
                "engine": None, "conductor": None,
                "pristine": {"ok": True, "changed": [], "restored": []}}


def reread(rec: dict, suite: dict, opencode: str, env: dict | None, api: Api | None,
           ledger: list[dict] | None) -> dict:
    """A run record read again: its session exported again, its final check run again on its
    folder (the shot the suite names), its engine block from the ledger rows of its window and
    the conductor's journal for its session. What cannot be read again (no folder, no session,
    no API) keeps the record's value; the record's isolation facts (tag, store clear, pristine)
    are kept as they were taken during the night."""
    if rec.get("error") or suite["kind"] != "opencode" or not rec.get("session_id"):
        return rec
    t0 = _iso_ms(rec["started_at"]) / 1000.0
    t1 = _iso_ms(rec.get("ended_at") or rec["started_at"]) / 1000.0
    export, reader = export_session(opencode, rec["session_id"], env)
    got = read_export(export, suite, t0, reader)
    fin = rec.get("final_check")
    if suite.get("final_check") and rec.get("dir") and os.path.isdir(rec["dir"]):
        fin = final_check(suite["final_check"], rec["dir"], shot=final_shot(suite, rec["dir"]))
    apply_final(got["gates"], suite, fin, rec.get("dir") or "", t0,
                got["wall_minutes"] if got["wall_minutes"] is not None else rec.get("wall_minutes"))
    sent = int((rec.get("continuations") or {}).get("sent") or 0)
    out = dict(rec, **outcome(got, fin, sent, bool(rec.get("timed_out")),
                              (rec.get("continuations") or {}).get("driver") or "watch",
                              final_gates(suite)))
    if got["wall_minutes"] is not None:
        out["wall_minutes"] = got["wall_minutes"]
    if ledger is not None:
        out["engine"] = engine_requests(api, t0, t1, rows=ledger)
    cond = conductor_events(api, rec["session_id"]) if api is not None else None
    if cond is not None:
        out["conductor"] = cond
    return out


def _model_def() -> dict:
    """The TandemLLM model entry of the global opencode config (limits, reasoning, variants), or
    a minimal one."""
    try:
        with open(os.path.expanduser("~/.config/opencode/opencode.json")) as f:
            cfg = json.load(f)
        for p in (cfg.get("provider") or {}).values():
            if "TandemLLM" in (p.get("models") or {}):
                return p["models"]["TandemLLM"]
    except (OSError, ValueError):
        pass
    return {"name": "TandemLLM", "reasoning": True, "tool_call": True,
            "limit": {"context": 262144, "output": 65536}}


def _api_key(env_name: str) -> str:
    """The provider's key from the environment, else the global opencode config's provider on
    the same base; never printed, never written into a run folder."""
    if os.environ.get(env_name):
        return os.environ[env_name]
    try:
        with open(os.path.expanduser("~/.config/opencode/opencode.json")) as f:
            cfg = json.load(f)
        for p in (cfg.get("provider") or {}).values():
            if "TandemLLM" in (p.get("models") or {}):
                return str((p.get("options") or {}).get("apiKey") or "")
    except (OSError, ValueError):
        pass
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suites", default=SUITES)
    ap.add_argument("--only", default="", help="comma-separated suite ids")
    ap.add_argument("--n", type=int, default=2, help="runs per suite and arm (n < 10 is "
                                                    "indicative)")
    ap.add_argument("--first", default="raw", choices=ARMS)
    ap.add_argument("--arms", default="raw,tandem",
                    help="the arms to run (SRV-157's A/B: a second night of `tandem` alone under "
                         "another --profile)")
    ap.add_argument("--base", default=os.environ.get("TANDEM_EVAL_BASE",
                                                     "http://127.0.0.1:8000/v1"),
                    help="the OpenAI base URL opencode and client_smoke talk to "
                         "($TANDEM_EVAL_BASE, else the engine on this machine)")
    ap.add_argument("--engine", default="",
                    help="the engine's API for the journal, the requests and the store clear "
                         "(default: --base without /v1)")
    ap.add_argument("--profile", default="overnight", help="the TandemLLM arm's X-Tandem-Profile")
    ap.add_argument("--driver", default="watch", choices=("watch", "continue"))
    ap.add_argument("--max-continues", type=int, default=0)
    ap.add_argument("--no-clear-store", dest="clear_store", action="store_false")
    ap.add_argument("--work", default="~/tandem-eval-runs", help="where run folders go")
    ap.add_argument("--out", default="~/pub-results/tandem-eval")
    ap.add_argument("--night", default="")
    ap.add_argument("--opencode", default=shutil.which("opencode") or "opencode")
    ap.add_argument("--smoke-python", default=sys.executable,
                    help="the interpreter for tools/client_smoke.py (needs openai and "
                         "jsonschema), e.g. 'uv run --with openai --with jsonschema python'")
    ap.add_argument("--key-env", default="TANDEM_EVAL_KEY")
    ap.add_argument("--admin-token-env", default="QSE_ADMIN_TOKEN")
    ap.add_argument("--raw-row", default="", help="the raw never-worse row (JSON) for the build")
    ap.add_argument("--box-lock", default="", help="ssh host whose ~/.qwen38-box.flock is held "
                                                  "for the night (e.g. dgx)")
    ap.add_argument("--publish", action="store_true",
                    help="keep the report on the box for the Compare screen")
    ap.add_argument("--plan", action="store_true", help="print the schedule and stop")
    ap.add_argument("--resume", action="store_true",
                    help="go on with --night: run only what has no record (or an errored one)")
    ap.add_argument("--report", default="", help="rebuild the report from a night's runs/")
    ap.add_argument("--reread", action="store_true",
                    help="with --report: read every run again from its folder, its opencode "
                         "session and the engine ledger (the records' own numbers otherwise)")
    ap.add_argument("--build", default="", help="with --report: the night's build as "
                    "VERSION:CODE_SHA (default: the night's report.json, else the server's)")
    ap.add_argument("--supersedes", default="",
                    help="with --report: the published report (its file name on the box) this "
                         "one replaces; the reason goes in --supersede-why")
    ap.add_argument("--supersede-why", default="")
    ap.add_argument("--validate", default="", help="check a report.json against its schema")
    a = ap.parse_args(argv)
    a.smoke_python = a.smoke_python.split()

    if a.validate:
        with open(a.validate) as f:
            errs = validate(json.load(f))
        print("\n".join(errs) or "valid")
        return 1 if errs else 0
    suites = load_suites(a.suites, [x for x in a.only.split(",") if x])
    night = a.night or night_of(time.time())
    arms = tuple(x for x in a.arms.split(",") if x)
    if not arms or any(x not in ARMS for x in arms):
        raise SystemExit(f"--arms: one or both of {', '.join(ARMS)}")
    if a.plan:
        for it in schedule(suites, a.n, a.first, night, only_arms=arms):
            print(f"{it['rep']}  {it['suite']:12s} {it['arm']:7s} {it['tag']}")
        return 0
    engine = (a.engine or re.sub(r"/v1/?$", "", a.base)).rstrip("/")
    api = Api(engine, os.environ.get(a.admin_token_env))
    if a.report:
        return rebuild(a, api, suites)
    env = dict(os.environ, TANDEM_EVAL_KEY=_api_key(a.key_env))
    with contextlib.ExitStack() as stack:
        if a.box_lock:
            stack.enter_context(BoxLock(a.box_lock))
        build = build_of(api)
        n = Night(a, suites, api, env, _model_def())
        plan = schedule(suites, a.n, a.first, n.night, only_arms=arms)
        runs = []
        if a.resume:
            runs, plan, doc = n.resume(plan, build)
            last = doc["resumes"][-1]
            print(f"[eval] resuming {n.night}: {last['kept']} runs kept, {last['left']} to run"
                  + (f", {len(last['run_again'])} errored run again" if last["run_again"] else "")
                  + (", ON ANOTHER BUILD" if last["build_changed"] else ""), flush=True)
        else:
            if n.records():
                print(f"[eval] {n.dir} has run records already: this report holds this "
                      f"invocation's runs only (--resume continues the night)", flush=True)
            doc = n.start(build)
        for it in plan:
            print(f"[eval] {it['tag']} ...", flush=True)
            r = n.run(it)
            runs.append(r)
            print(f"[eval] {it['tag']}: passed={r.get('passed')} gates="
                  f"{(r.get('gates') or {}).get('passed')}/{(r.get('gates') or {}).get('total')}"
                  f" minutes={r.get('wall_minutes')} error={r.get('error')}", flush=True)
        if build_of(api) != build or (a.resume and any(r["build_changed"]
                                                        for r in doc["resumes"])):
            print("[eval] the build changed during the night: the report says so", flush=True)
            build = dict(doc.get("build") or build, changed_during_night=True)
        elif a.resume:
            build = doc.get("build") or build
        meta = {"night": n.night, "build": build, "base": a.base, "profile": a.profile,
                "driver": {"mode": a.driver, "max_continues": a.max_continues}}
        rep = build_report(meta, runs, suites, raw_row(api, build, a.raw_row))
        if a.resume:
            rep["resumed"] = doc["resumes"]
        return _write(a, api, rep, n.dir)


def rebuild(a, api: Api, suites: list[dict]) -> int:
    """`--report DIR`: the night's report again from DIR/runs (and with `--reread`, from the run
    folders, the sessions and the ledger). The build is the night's own: `--build`, else the
    report.json already in DIR, else the server's -- which serves a later build by the time a
    report is rebuilt. Nothing is deleted: the report and run records it replaces are kept
    beside it with `superseded-<time>` in their names."""
    folder = a.report.rstrip("/")
    runs = []
    for name in sorted(os.listdir(os.path.join(folder, "runs"))):
        if name.endswith(".json"):
            with open(os.path.join(folder, "runs", name)) as f:
                runs.append(json.load(f))
    old = None
    if os.path.exists(os.path.join(folder, "report.json")):
        with open(os.path.join(folder, "report.json")) as f:
            old = json.load(f)
    if a.build:
        version, _, code = a.build.partition(":")
        build = {"version": version or None, "code_sha": code or None}
    elif old is not None:
        build = old["build"]
    else:
        build = build_of(api)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if a.reread:
        by_id = {s["id"]: s for s in suites}
        env = dict(os.environ)
        since = min(_iso_ms(r["started_at"]) for r in runs) / 1000.0 if runs else time.time()
        ledger = ledger_rows(api, since, pages=500)
        runs = [reread(r, by_id[r["suite"]], a.opencode, env, api, ledger)
                if r["suite"] in by_id else r for r in runs]
        shutil.copytree(os.path.join(folder, "runs"),
                        os.path.join(folder, f"runs-superseded-{stamp}"))
        for r in runs:
            with open(os.path.join(folder, "runs", r["tag"] + ".json"), "w") as f:
                json.dump(r, f, indent=1)
    meta = {"night": os.path.basename(folder), "build": build, "base": a.base,
            "profile": a.profile, "driver": {"mode": a.driver, "max_continues": a.max_continues}}
    rep = build_report(meta, runs, suites, raw_row(api, build, a.raw_row))
    if a.supersedes:
        rep["supersedes"] = {"file": a.supersedes, "why": a.supersede_why or None}
    for ext in ("json", "md"):
        p = os.path.join(folder, f"report.{ext}")
        if os.path.exists(p):
            os.replace(p, os.path.join(folder, f"report-superseded-{stamp}.{ext}"))
    return _write(a, api, rep, folder)


def _write(a, api: Api, rep: dict, folder: str) -> int:
    os.makedirs(folder, exist_ok=True)
    errs = validate(rep)
    with open(os.path.join(folder, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(folder, "report.md"), "w") as f:
        f.write(to_md(rep))
    print(to_md(rep))
    if errs:
        print("[eval] the report does not validate:\n" + "\n".join(errs[:10]))
        return 1
    if a.publish:
        st, got = api.post("/v1/conductor/evals", {"report": rep})
        print(f"[eval] published: {st} {got}")
        return 0 if st == 201 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
