"""The verify phase: a run that built or changed a web page or app checks it in a browser before
it ends (SRV-532). Mixed into `chatd.service.Service`.

Live (2026-10-16) a run built "a Japanese pagoda garden in Three.js" in 28 steps and called it
complete from a count of "80 distinct coarse colors" in a screenshot: the scene was dull blue-grey
and its Day mode looked like night. The model tried the laya-playtest package only when the user
asked, and its kit could not be downloaded in the sandbox. No package or rule reached the model
(the conductor's lean fit), so the check is the harness's own, not a line in a prompt.

WHEN. A step without tool calls would end the run. If the run changed a web file in the chat's
files (WEB_EXTS: the run's `before` snapshot against the tree now, so files written by bash count
too) since it started or since its last check, and a page or app comes of them, the harness checks
it first. The chat setting `verify` (on by default) turns it off; plan mode changes nothing.

THE TARGET. The preview the run presented (`present`, SRV-511); else the dev server that runs for
the changed files' project; else the changed page that runs as a file (an index.html first). A
page that loads module scripts from its folder (preview.file_blocks_modules) does not run as a
file, a browser refuses them from file://: it is loaded through the HTTP server that runs from its
folder in the sandbox (live 2026-10-17, a 15-file Three.js app behind `python3 -m http.server`),
and with none running that is the finding. A page that runs only from its dev server
(preview.needs_server) with none running is a finding too: the check names the command that
starts it. Changed files with no page and no dev server (a script,
a library) are not checked.

THE CHECK. The laya kit's page check (server/laya_kit/laya_drive.py `page`, the check the
laya-playtest and testing packages name for one screen) in the chat's sandbox: headless Chromium at
1280 and 390 px; console errors, failed loads, the visible text, the controls, the page's state
hook, horizontal overflow and the screenshot's look in numbers. chatd copies the kit into the
chat's files (`.tandem/`, from the source tree), and so for a library text that downloads it
(`kit_note`): the sandbox cannot reach TandemLLM's address. Without the source tree (an installed
package) render_page does the check.

THE ROUND. The check is a tool call of the harness's own (`render_page`, `harness: verify`) in the
answer's step; its result goes to the model in the next step, with the 1280 px screenshot when the
model reads images, and the model fixes what is wrong or answers. At most ROUNDS checks a run, a
second only when web files changed after the first. A sandbox failure counts in the run's failure
streak (SRV-526), and the run's step limit holds.

THE SUMMARY. A completed run that changed an app ends with a `verify` part, a "Checked / Not
checked" block from the run's evidence: the harness's checks (passed, problems the run did not fix,
or why none could run), whether the model saw the screenshot and the page's state hook answered,
the model's own check calls (render_page, a preview, a kit run), and the files changed after the
last check. A check passes on the page loading without console errors or failed steps; how it
looks is the model's judgement, from the screenshot when it reads images.
"""

from __future__ import annotations

import json
import os
import posixpath
import shlex

from chatd import preview as pv
from chatd import tools as tools_mod
from chatd.ids import iso, now_ms, ulid
from chatd.sandbox import Unavailable
from chatd.tools import PathError, Result, ToolError

ROUNDS = 2
WEB_EXTS = (".html", ".htm", ".js", ".mjs", ".jsx", ".ts", ".tsx", ".css", ".vue", ".svelte")
PAGE_EXTS = (".html", ".htm")
WIDTHS = (1280, 390)
CHECK_TIMEOUT_S = 90
KIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server",
                       "laya_kit")
KIT_FILES = ("laya_drive.py", "laya-drive.js")
KIT_IN_BOX = ".tandem/laya_drive.py"
PLAYTEST = "laya-playtest"


def vendor_kit(ws) -> bool:
    """The laya kit into the chat's files (/workspace/.tandem/), from the source tree: the
    sandbox cannot reach TandemLLM's address to download it (live 2026-10-16, the laya-playtest
    package's curl failed). False without the source tree (an installed package)."""
    try:
        for name in KIT_FILES:
            with open(os.path.join(KIT_DIR, name), "rb") as fh:
                ws.write(f".tandem/{name}", fh.read())
    except OSError:
        return False
    return True


def kit_note(text: str, ws) -> str:
    """A library text that downloads the laya kit gets the kit in the sandbox and a line that
    says where it is and what runs without TandemLLM's address."""
    if "/v1/laya/kit/" not in text or not vendor_kit(ws):
        return ""
    return ("\n\nIn this chat the kit is in the sandbox already: /workspace/.tandem/laya_drive.py "
            "and /workspace/.tandem/laya-drive.js; skip the download, TandemLLM's address does not "
            "answer from the sandbox. `python3 /workspace/.tandem/laya_drive.py page URL --no-laya "
            "--step ...` runs here (real clicks and keys, the steps you give it); the modes that ask "
            "Laya for each move (browser, proc, cmd) cannot reach it from the sandbox.")


def new_state() -> dict:
    """A run's verify state: its checks, the tree at the last one, why none could run, the target
    and the files changed after the last check."""
    return {"checks": [], "base": None, "skip": None, "target": None, "after": None,
            "vision": False}


def changed_web(before: dict, now: dict) -> list[str]:
    """The web files (workspace-relative) that are new or changed in `now`."""
    return sorted(p for p in now if now[p] != before.get(p) and p.lower().endswith(WEB_EXTS))


def _pages_first(paths: list[str]) -> list[str]:
    """The changed pages, an index.html first, then the shallowest."""
    pages = [p for p in paths if p.lower().endswith(PAGE_EXTS)]
    return sorted(pages, key=lambda p: (posixpath.basename(p).lower() != "index.html",
                                        p.count("/"), p))


def look_words(lk: dict | None) -> str:
    """The kit's numbers as they are, no verdict: a red cube on a dark page is dark and right."""
    if not lk:
        return ""
    top = lk.get("top") or [[lk.get("top_colour"), lk.get("top_share")]]
    return (f"brightness {lk.get('brightness')} %, colours " +
            ", ".join(f"{c} {share} %" for c, share in top) +
            f", saturation {lk.get('saturation')} %")


def check_words(rep: dict, target: dict, n: int, vision: bool, playtest: bool) -> str:
    """The check's result as the model reads it."""
    lines = [f"The harness checked the app before the run ends (check {n} of {ROUNDS}): it loaded "
             f"{target['url']} in headless Chromium at {' and '.join(map(str, WIDTHS))} px."]
    if target.get("note"):
        lines.append(target["note"])
    lines.append(f"Result: {rep.get('verdict') or 'no verdict'}")
    if rep.get("errors"):
        lines.append("Console errors and failed loads: " + " | ".join(rep["errors"][:8]))
    widths = rep.get("widths") or []
    if widths:
        text = (widths[0].get("text") or "").strip()
        lines.append(f'Visible text at {widths[0].get("width")} px: "{text[:300]}"' if text else
                     f"No visible text at {widths[0].get('width')} px.")
    ctl = [f"{c.get('tag')} \"{c.get('label')}\"" for c in rep.get("controls") or [] if c.get("label")]
    if ctl:
        lines.append("Controls: " + ", ".join(ctl[:12]))
    for w in widths:
        if w.get("overflow_x"):
            lines.append(f"At {w['width']} px the page is wider than the screen (scroll width "
                         f"{w.get('scroll_width')} px).")
    for w in widths:
        if w.get("look"):
            lines.append(f"The screenshot at {w['width']} px: {look_words(w['look'])}.")
    if rep.get("state") is not None:
        lines.append("The page's state hook: " + json.dumps(rep["state"], ensure_ascii=False)[:400])
    if vision and widths:
        lines.append("The screenshot at 1280 px is attached; the 390 px one is beside it in "
                     f"{posixpath.dirname(target.get('shot') or '') or '/workspace/.tandem/shots'}.")
    elif widths:
        lines.append("You read no images here: judge the page by these facts.")
    if playtest:
        lines.append(f"The {PLAYTEST} package is in this task: a game or a flow of several steps "
                     f"needs its playtest too; this check only loaded the page.")
    lines.append("Compare this with what the user asked for. If anything is wrong or missing (an "
                 "error, an empty page, a layout that breaks at 390 px, light or colours that do "
                 "not match the request), fix it now. If it is right, answer the user, and say "
                 "what this check showed and what it did not test (input, sound, motion over "
                 "time).")
    return "\n".join(lines)


def blocked_words(target: dict, n: int) -> str:
    return (f"The harness wanted to check the app before the run ends (check {n} of {ROUNDS}), "
            f"but {target['why']} Start it (bash with background true) so the page can be "
            f"loaded, check that it answers, then answer the user. Do not tell the user it works "
            f"before it has run.")


def facts_of(rep: dict) -> str:
    """The check's result in a few words, for the summary."""
    errs = rep.get("errors") or []
    w = rep.get("widths") or []
    over = [str(x["width"]) for x in w if x.get("overflow_x")]
    bits = [f"{len(errs)} console error{'s' if len(errs) != 1 else ''} ({errs[0][:120]})"] \
        if errs else []                      # one item of the summary: joined with commas
    if over:
        bits.append(f"wider than the screen at {', '.join(over)} px")
    if w and not (w[0].get("text") or "").strip() and not rep.get("state"):
        bits.append("no visible text")
    if w and w[0].get("look"):
        bits.append("the screenshot: " + look_words(w[0]["look"]))
    return ", ".join(bits) or "it loaded"


def own_checks(parts: list[dict]) -> list[str]:
    """The checks the model made itself in the run, from the outcomes of its calls."""
    out, shots, last = [], 0, None
    for p in parts:
        if p.get("type") != "tool_call" or p.get("harness"):
            continue
        if p.get("name") == "render_page":
            shots += 1
            last = p.get("status") == "done"
        elif p.get("name") == "present" and (p.get("pane") or {}).get("kind") == "preview":
            out.append("the preview pane " + ("loaded without a problem" if p.get("status") ==
                                              "done" else "reported a problem"))
        elif p.get("name") == "bash" and "laya_drive.py" in (p.get("input") or ""):
            mode = "a Laya playtest" if " browser " in (p.get("input") or "") else "a page check"
            code = (p.get("output") or {}).get("exit_code")
            out.append(f"{mode} of its own (exit {code if code is not None else '?'})")
    if shots:
        out.insert(0, f"render_page {shots} time{'s' if shots != 1 else ''} by the model (the last "
                      f"one {'rendered' if last else 'failed'})")
    return out


def summary(st: dict, parts: list[dict] | None = None) -> dict | None:
    """The run's `verify` part: a "Checked / Not checked" block from the run's evidence (the
    harness's checks, the model's own check calls, the files changed after the last check); None
    when the run built no page or app."""
    tgt = st.get("target")
    if tgt is None:
        return None
    checks, after = st["checks"], st.get("after") or []
    last = checks[-1] if checks else None
    done, missing = [], []
    for c in checks:
        if c["state"] == "not_run":
            continue
        done.append(f"{c['label']} in headless Chromium at {' and '.join(map(str, WIDTHS))} px "
                    f"(harness check {c['n']}): " + ("no console errors, " if c["state"] == "passed"
                                                     else "problems: ") + c["facts"])
    if last and last.get("seen"):
        done.append("the screenshot, by the model")
    if last and last.get("hook"):
        done.append("the page's state hook answered")
    own = own_checks(parts or [])
    done += own
    if last is None:
        missing.append(f"{tgt['label']} in a browser: {st.get('skip') or 'the check did not run'}")
    elif last["state"] == "not_run":
        missing.append(f"{tgt['label']} in a browser: the check could not run ({last['facts']})")
    elif not last.get("seen"):
        missing.append("how it looks: the model read no screenshot (it reads no images here), "
                       "only the page's text and numbers")
    if not any("playtest" in x for x in own):
        missing.append("input and interactions")
    missing.append("sound and motion over time")
    if after:
        missing.insert(0, f"{len(after)} file{'s' if len(after) != 1 else ''} changed after the "
                          f"last check ({', '.join(after[:3])}{', ...' if len(after) > 3 else ''}): "
                          f"the limit is {ROUNDS} checks a run")
    state = ("not_run" if last is None or last["state"] == "not_run" else
             "stale" if after else last["state"])
    lead = {"passed": "Verified by the harness.", "problems": "The last check found problems "
            "the run did not fix.", "not_run": "Not verified.",
            "stale": "Files changed after the last check."}[state]
    text = lead + ("\nChecked: " + "; ".join(done) + "." if done else "") + \
        "\nNot checked: " + "; ".join(missing) + "."
    return {"type": "verify", "state": state, "text": text, "checked": done,
            "not_checked": missing}


class VerifyMixin:
    def verify_step(self, run) -> bool:
        """At a step that would end the run: check the app the run built or changed, as a tool
        call in this step whose result the model reads next; True when it did (the loop goes on)."""
        conv, st = run.conv, run.verify
        root = self.root_cid(conv.id)                   # a worker's files are its chat's
        with conv.lock:
            settings = self.settings_of(conv.id)
        if not settings.get("verify", True) or settings.get("plan_mode") or not run.snapshotted \
                or run.agents:                  # Agents mode: its workers check what they built
            return False
        base = st["base"] or self.snaps.load(root, run.id, "before")
        if base is None:
            return False
        ws = self.workspace(conv.id)
        try:
            now = self.snaps.take(root, ws, run.id, f"check-{len(st['checks']) + 1}")
        except OSError:
            return False
        changed = changed_web(base, now)
        if not changed:
            return False
        tgt = self.verify_target(run, ws, changed)
        if tgt is None:
            return False
        st["target"] = tgt
        st["vision"] = bool(self.model_info(settings.get("model")).get("vision"))
        if len(st["checks"]) >= ROUNDS:
            st["after"] = changed
            return False
        if not self.sandboxes.available():
            st["skip"] = "this box has no sandbox to run a browser in"
            return False
        if run.step >= run.max_steps:
            st["skip"] = f"the run reached its limit of {run.max_steps} model calls"
            return False
        st["base"], st["after"] = now, None
        return self._check(run, ws, tgt, st["vision"])

    # ------------------------------------------------------------------ the target
    def verify_target(self, run, ws, changed: list[str]) -> dict | None:
        """What the check loads: {label, url, path | port, note?}, or {label, why} for a page
        that needs a dev server that does not run; None when no page or app comes of `changed`."""
        cid = self.root_cid(run.conv.id)                # the sandbox is the chat's
        shown = next((p["pane"] for p in reversed(run.conv.live["parts"] if run.conv.live else [])
                      if p.get("type") == "tool_call" and p.get("name") == "present" and
                      (p.get("pane") or {}).get("kind") == "preview"), None)
        ref = (shown or {}).get("ref") or ""
        if ref.startswith("port:"):
            port = int(ref[5:])
            return {"label": f"port {port}", "url": f"http://localhost:{port}/", "port": port}
        full = ["/workspace/" + p for p in changed]
        pages = _pages_first(full)
        if ref.startswith("file:"):
            pages = [ref[5:]] + [p for p in pages if p != ref[5:]]
        proj = None
        for p in full:
            proj = pv.project_of(ws, p)
            if proj and (proj.get("bundler") or proj.get("command")):
                break
            proj = None
        if proj:
            row = pv.pick(self.listening(cid, http=True), project=proj["dir"])
            if row:
                return {"label": f"the app on port {row['port']}",
                        "url": f"http://localhost:{row['port']}/", "port": row["port"]}
        ports = None
        for page in pages:
            need = pv.needs_server(ws, page)
            if need is None and not pv.file_blocks_modules(ws, page):
                return {"label": page, "url": "file://" + page, "path": page}
            if need is None:
                # modules from its folder: the page as the user opens it, through the server
                # that runs from that folder (a browser refuses them from file://)
                ports = self.listening(cid, http=True) if ports is None else ports
                served = pv.server_for(ports, page)
                if served:
                    row, rel = served
                    return {"label": f"{page} on port {row['port']}",
                            "url": f"http://localhost:{row['port']}/{rel}", "port": row["port"],
                            "path": rel, "cwd": row.get("cwd") or ""}
                return {"label": page, "why": f"{posixpath.basename(page)} loads its module "
                        f"scripts from its folder, which a browser refuses from file:// (CORS), "
                        f"and no HTTP server runs from {posixpath.dirname(page)} in the sandbox. "
                        f"Serve the folder, for example `python3 -m http.server 8000` there."}
            what = f"{posixpath.basename(page)} runs only from its dev server ({need['reason']})"
            return {"label": page, "why": f"{what}, and none runs in the sandbox." +
                    (f" The command is `{need['command']}` in {need['dir']}." if need.get("command")
                     else "")}
        if proj:
            return {"label": proj["dir"], "why": f"the project in {proj['dir']} runs from its dev "
                                                 f"server (`{proj['command']}`), and none runs in "
                                                 f"the sandbox."}
        return None

    # ------------------------------------------------------------------ the check
    def _check(self, run, ws, tgt: dict, vision: bool) -> bool:
        from chatd.agent import ToolCtx
        conv, st = run.conv, run.verify
        n = len(st["checks"]) + 1
        args = ({"port": tgt["port"], **({"path": tgt["path"]} if tgt.get("path") else {})}
                if tgt.get("port") else {"path": tgt.get("path") or tgt["label"]})
        k = self.add_call(run, f"verify_{ulid()[-12:]}", "render_page", harness="verify",
                          source={"kind": "harness"}, input=json.dumps(args))
        part = self.call_part(run, k)
        with conv.lock:
            part["started_at"] = iso(now_ms())
            self.emit(conv, {"type": "tool.started", "message_id": run.message_id,
                             "call_id": part["id"], "pane": None, "approved": None}, run_id=run.id)
        ctx = ToolCtx(self, run, part)
        ctx.phase = "verifying"
        ctx.words = f"Verifying the app: loading {tgt['label']} in a browser (check {n} of {ROUNDS})"
        run.set_phase("verifying", ctx.words, tool={"call_id": part["id"], "name": "render_page"})
        failure = None
        try:
            res, rec = self._run_check(ctx, ws, tgt, n, vision, self._playtest(run))
        except (ToolError, PathError) as exc:
            res = Result(f"The check found a problem: {exc}", is_error=True)
            rec = {"state": "problems", "facts": str(exc)[:200]}
        except Unavailable as exc:
            failure = ((exc.data or {}).get("sandbox") or {}).get("reason") or str(exc)
            res = Result(f"The check could not run: {exc}", is_error=True)
            rec = {"state": "not_run", "facts": str(exc)[:200]}
        self._sandbox_streak(run, failure, res.content)
        cancelled = run.cancel.is_set() and run.stop_mode
        self._finish_call(run, part, res, status="cancelled" if cancelled else None, ctx=ctx)
        st["checks"].append({"n": n, "label": tgt["label"], "hook": False, "seen": False, **rec})
        self.log(f"[verify] {run.id} check {n}: {rec['state']} ({tgt['label']})")
        # a check that could not run leaves the model nothing to fix: the run ends
        return rec["state"] != "not_run" and not run.sandbox_stuck

    def _playtest(self, run) -> bool:
        """The laya-playtest package is in this task: the router put it in the prompt, or the
        model read it from the library."""
        frame = ((run.usage or {}).get("tandem") or {}).get("harness") or {}
        if any((p or {}).get("name") == PLAYTEST for p in frame.get("packages") or []):
            return True
        return any(p.get("type") == "tool_call" and p.get("name") == "library_read" and
                   PLAYTEST in (p.get("input") or "") for p in (run.conv.live or {}).get("parts", []))

    def _run_check(self, ctx, ws, tgt: dict, n: int, vision: bool, playtest: bool
                   ) -> tuple[Result, dict]:
        """(the result for the model, the check's record: state passed | problems | not_run, the
        facts for the summary, whether the page's state hook answered and the model saw the
        screenshot)."""
        if tgt.get("why"):
            return Result(blocked_words(tgt, n), is_error=True), \
                {"state": "problems", "facts": f"it did not load: {tgt['why']}"}
        if not vendor_kit(ws):
            res = tools_mod._render(ctx, {k: tgt[k] for k in ("path", "port") if k in tgt})
            ok = "no console errors" in res.content
            res.content = (f"The harness checked the app before the run ends (check {n} of "
                           f"{ROUNDS}) with render_page.\n{res.content}\n" +
                           check_words({}, tgt, n, vision, playtest).splitlines()[-1])
            res.is_error = not ok
            return res, {"state": "passed" if ok else "problems", "seen": vision,
                         "facts": "render_page" if ok else
                         res.content.split("console:")[-1].strip()[:200]}
        shots = f"/workspace/.tandem/shots/{ctx.call_id}"
        cmd = (f"cd /workspace && python3 {KIT_IN_BOX} page {shlex.quote(tgt['url'])} --no-laya "
               f"--widths {','.join(map(str, WIDTHS))} --shots {shots} "
               f"--timeout {CHECK_TIMEOUT_S}")
        code, timed_out = ctx.exec(["bash", "-lc", cmd], timeout=CHECK_TIMEOUT_S + 30, quiet=True)
        out = ctx.output_text()
        try:
            rep = json.loads(out[out.index("{"):out.rindex("}") + 1])
        except ValueError:
            why = (tools_mod.oom_words() if getattr(ctx, "oom", False) else
                   "it timed out" if timed_out else (out.strip()[-300:] or f"exit {code}"))
            return Result(f"The check could not run: {why}", is_error=True), \
                {"state": "not_run", "facts": why}
        # the line each error names, from the chat's files (tools.quote_lines)
        rep["errors"] = tools_mod.quote_lines(ctx, rep.get("errors") or [], tgt.get("port"),
                                              tgt.get("cwd") or "")
        verdict = str(rep.get("verdict") or "")
        if code == 2 and not verdict.startswith("CHECK FAILED"):
            return Result(f"The check could not run: {verdict}", is_error=True), \
                {"state": "not_run", "facts": verdict[:200]}
        shot = f"{shots}/page-{WIDTHS[0]}.png"
        tgt = {**tgt, "shot": shot}
        images = []
        try:
            images = [("image/png", ws.read(shot))]
        except (PathError, OSError):
            pass
        ok = bool(rep.get("ok"))
        res = Result(check_words(rep, tgt, n, vision, playtest), is_error=not ok, images=images,
                     pane={"kind": "image", "ref": f"file:{shot}", "title": "verify"}
                     if images else None)
        return res, {"state": "passed" if ok else "problems",
                     "facts": facts_of(rep) if rep.get("widths") else verdict[:200],
                     "hook": rep.get("state") is not None, "seen": vision and bool(images)}
