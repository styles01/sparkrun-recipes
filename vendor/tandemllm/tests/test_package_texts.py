"""The package texts after the UT1 self-test (2026-10-06): SRV-163 and SRV-171. CPU only.

  1  SRV-163: a literal address in front of TandemLLM's own routes is found in every form the texts
     used (a shell default, a Python default plus a path, a fetch, a session-token path), and a
     packages document that has one is refused at save with the field named; `{tandem_url}` passes
                                                        test_literal_hosts_are_refused
  2  `{tandem_url}` is the request's address, or `QWEN38_TANDEM_URL` when the operator sets one;
     `{tandem_scratch}` is `.tandem` in the project (excluded from git, deleted before done), or
     QWEN38_TANDEM_SCRATCH
                                                        test_the_tokens
  3  the shipped texts: no `.tandem/`, no literal host; the testing package has the scratch, hook
     and no-download rules; the migration leaves them unchanged (idempotent)
                                                        test_the_shipped_texts_are_clean
  4  the box's saved document of 2026-10-03 (v7: the literal host in four packages, `.tandem/` in
     eleven), migrated: clean, and the save check passes   test_a_box_document_migrates_clean
  5  the Laya helper, against an address nobody answers: exits 3 at once with the truthful line
     ("unavailable ... your code is not the cause"), never a traceback
                                                        test_the_helper_fails_fast
  6  the playtest kit's log sits beside the kit, not in the working directory; the outcome reader
     reads a log tail in the scratch directory        test_the_kit_and_the_log_tail
  7  SRV-195: the box's saved document of 0.3.43 (v11: "verify with what already works here",
     "a game or a UI flow" played by Laya) migrates to the kit's page check, no driver of one's
     own, check-fix-check, a one-screen tool without a playtest; every command the texts give
     for the page check parses with the kit's own argument parser; a second run changes nothing
                                                        test_the_page_check_edits
  8  SRV-229: the final message speaks to the user: no shipped text asks to quote Laya, report
     its verdicts or say that the hook went; the testing package has the final-message and the
     overlay rules; the box's v16 shape migrates, idempotent      test_the_final_message_edits
  9  SRV-229: the S5 and E2 notes in plain words, in the YAML, its JSON copy and a box's document
                                                        test_the_monitor_notes_in_plain_words
 10  FX13 (SRV-259 leftover, O03 on 0.4.1-test: 7 lines between tool calls named Laya; a 422 for an
     empty text read "Laya is unavailable (transport error)"): every text that keeps Laya out of
     the final message also keeps it out of the plan and the progress lines; the helper, against a
     server that answers with the route's own 422 bodies, says what was wrong and exits 2 (an
     empty text is caught before the call), a 503 is still "unavailable" (exit 3); the box's v23
     shape migrates (10 packages), idempotent        test_fx13_progress_lines_and_refused_laya_calls

Run: python tests/test_package_texts.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import packages as P  # noqa: E402
from server.conductor import schema  # noqa: E402
from tools import packages_migrate as M  # noqa: E402

HOST = "https://ai-api.example.lan"


def _seed() -> dict:
    with open(os.path.join(ROOT, "packages", "seed.json")) as f:
        doc = json.load(f)
    for name, p in doc["packages"].items():
        src = p.pop("body_from", None)
        if src:
            with open(os.path.join(ROOT, "packages", src)) as f:
                p["body"] = f.read()
    assert not schema.doc_errors("packages", doc), schema.doc_errors("packages", doc)[:3]
    return doc


def test_literal_hosts_are_refused():
    forms = [f'curl -s "${{TANDEM_URL:-{HOST}}}/v1/laya/ask"',
             f'URL = os.environ.get("TANDEM_URL", "{HOST}") + "/v1/laya/ask"',
             f'await fetch("{HOST}/v1/laya/ask", {{method: "POST"}})',
             f"curl -s {HOST}/s/Ab3xYz9_kLmn/v1/library/x",
             f"curl -s {HOST}:8443/v1/memory/search?q=a"]
    for text in forms:
        assert P.literal_api_hosts(text), text
    for text in ("curl -s {tandem_url}/v1/laya/ask", "https://github.com/oven-sh/bun/v1/laya",
                 "see https://example.com/docs", "http://localhost:5173/"):
        assert P.literal_api_hosts(text) == [] or "github" in text, text
    doc = _seed()
    doc["packages"]["laya-testing"]["body"] += "\n" + forms[0]
    errs = schema.check("packages", doc)
    assert any("$.packages.laya-testing.body" in e and HOST in e and "{tandem_url}" in e
               for e in errs), errs
    doc["packages"]["laya-testing"]["body"] = doc["packages"]["laya-testing"]["body"].replace(
        HOST, "{tandem_url}")
    assert not [e for e in schema.check("packages", doc) if "calls TandemLLM" in e]
    return f"{len(forms)} literal forms found and refused at save; the token passes"


def test_the_tokens():
    url = "http://127.0.0.1:8000/s/Ab3xYz9_kLmn"
    # in the project by default: opencode's sandbox asks before any path outside it, and its
    # `run` mode rejects the request (the live check of 2026-10-06 stopped at its first log)
    assert P.with_url("{tandem_url}/v1/laya/ask > {tandem_scratch}/run.log", url) == \
        f"{url}/v1/laya/ask > .tandem/Ab3xYz9_kLmn/run.log"
    # FX15: the conversation's own folder under .tandem; without a token, one shared name
    assert P.with_url("{tandem_scratch}/a", None) == ".tandem/session/a"
    assert P.with_url("{tandem_url}", None) == P.URL_DEFAULT
    old_s = os.environ.pop(P.SCRATCH_ENV, None)
    try:
        os.environ[P.SCRATCH_ENV] = "/tmp/tandem-{token}"
        assert P.with_url("{tandem_scratch}/a", url) == "/tmp/tandem-Ab3xYz9_kLmn/a"
        assert P.with_url("{tandem_scratch}/a", None) == "/tmp/tandem-session/a"
    finally:
        os.environ.pop(P.SCRATCH_ENV, None)
        if old_s is not None:
            os.environ[P.SCRATCH_ENV] = old_s
    hdr = {"Host": "spark:8000"}
    old = os.environ.pop(P.URL_ENV, None)
    try:
        assert P.request_url(hdr) == "http://spark:8000"
        os.environ[P.URL_ENV] = "http://192.0.2.7:8000/"
        assert P.request_url(hdr) == "http://192.0.2.7:8000"
    finally:
        os.environ.pop(P.URL_ENV, None)
        if old is not None:
            os.environ[P.URL_ENV] = old
    return ("the request's address or QWEN38_TANDEM_URL; .tandem/<token>, or QWEN38_TANDEM_SCRATCH "
            "with {token}")


def test_the_shipped_texts_are_clean():
    files = glob.glob(os.path.join(ROOT, "packages", "**", "*.md"), recursive=True)
    for path in files:
        with open(path) as f:
            text = f.read()
        assert not re.search(r"(?<![\w{])\.tandem/", text), path
        assert not P.literal_api_hosts(text), (path, P.literal_api_hosts(text))
        assert M.migrate_text(path, text)[1] == [], (path, M.migrate_text(path, text)[1])
    with open(os.path.join(ROOT, "packages", "inline", "testing.md")) as f:
        t = f.read()
    assert M.SCRATCH_RULE_V3 in t and M.LEFTOVER_NEW in t and "tandem-test-hook:start" in t
    assert M.RUNTIME_RULE.replace(M.RUNTIME_OLD_PAGE, M.RUNTIME_NEW_PAGE) in t
    with open(os.path.join(ROOT, "packages", "inline", "laya-testing.md")) as f:
        # edit 2's bullet, as edit 29 words its end (FX10)
        assert M.FAIL_BULLET.replace(M.WRITE_UNAVAIL_OLD, M.WRITE_UNAVAIL_NEW) in f.read()
    return f"{len(files)} files: no .tandem/, no literal host, already migrated"


def test_a_box_document_migrates_clean():
    # the shape of the box's v7 (2026-10-03): the shipped texts with the literal host where the
    # token was, and the old `.tandem/` paths
    doc = _seed()
    for name in ("laya-testing", "laya-scripts", "laya-playtest", "test-web", "testing"):
        p = doc["packages"][name]
        body = p["body"].replace("{tandem_url}", HOST).replace("{tandem_scratch}/", ".tandem/")
        body = body.replace("mkdir -p {tandem_scratch}", "mkdir -p .tandem")
        p["body"] = body
    assert [e for e in schema.check("packages", doc) if "calls TandemLLM" in e]
    new, report = M.migrate_doc(doc)
    assert set(report) >= {"laya-testing", "laya-scripts", "laya-playtest", "test-web"}, report
    assert not [e for e in schema.check("packages", new) if "calls TandemLLM" in e]
    for name, p in new["packages"].items():
        if p.get("source") == "inline":
            assert HOST not in p["body"] and not re.search(r"(?<![\w{])\.tandem/", p["body"]), name
    assert M.migrate_doc(new)[1] == {}, "a second run changes nothing"
    return f"{len(report)} packages migrated; the save check passes; idempotent"


def test_the_helper_fails_fast():
    with open(os.path.join(ROOT, "packages", "inline", "laya-scripts.md")) as f:
        text = f.read()
    code = re.search(r"```python\n(# \{tandem_scratch\}/laya\.py.*?)```", text, re.S).group(1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "laya.py")
        with open(path, "w") as f:
            f.write(P.with_url(code, "http://127.0.0.1:9"))
        r = subprocess.run([sys.executable, path, "Did it pass?", "pass,fail"], input="ok",
                           capture_output=True, text=True, timeout=30,
                           env={"PATH": os.environ.get("PATH", ""), "TANDEM_URL":
                                "http://127.0.0.1:9"})
    assert r.returncode == 3, (r.returncode, r.stderr[-400:])
    assert "LAYA unavailable" in r.stderr and "your code is not the cause" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr
    return "exit 3, one truthful line, no traceback"


def test_the_kit_and_the_log_tail():
    from server.laya_kit import laya_drive
    assert os.path.dirname(laya_drive.LOG) == os.path.dirname(os.path.abspath(laya_drive.__file__))
    rx = re.compile(schema.OUTCOMES["patterns"]["log_tail"])
    assert rx.search("tail -n 60 /tmp/tandem-Ab3xYz9_kLmn/run.log")
    assert rx.search("tail -n 60 /tmp/tandem/build.log")
    assert rx.search("tail -n 60 ./.tandem/run.log"), "a project from before keeps working"
    assert not rx.search("tail -n 60 server.log")
    return "the kit's log beside it; scratch and old log tails both read"


def test_the_page_check_edits():
    # the box's v11 texts: the shipped texts with edits 9-11 undone
    doc = _seed()
    undo = ((M.CHECK_ONE_NEW, M.CHECK_ONE_OLD),
            (M.LAYA_RUN_NEW.replace(M.LAYA_QUOTE_OLD, M.LAYA_QUOTE_NEW), M.LAYA_RUN_OLD),
            (M.QUOTE_NEW, M.QUOTE_OLD), (M.PAGE_CMD_NEW, M.PAGE_CMD_OLD), (M.WEB_CMD_NEW, M.WEB_CMD_OLD),
            (M.VERIFY_NEW, M.VERIFY_OLD), (M.RUNTIME_NEW_PAGE, M.RUNTIME_OLD_PAGE), (M.WEB1_V3, M.WEB1_V2),
            (M.WEB3_NEW, M.WEB3_OLD), (M.WEB5_NEW, M.WEB5_OLD), (M.WEB8_V3, M.WEB8_V2), (M.PLAY_V3, M.PLAY_V2))
    for name in ("testing", "test-web", "laya-playtest", "laya-testing"):
        p = doc["packages"][name]
        for new, old in undo:
            p["body"] = p["body"].replace(new, old)
        assert "laya_drive.py page" not in p["body"], name
    new, report = M.migrate_doc(doc)
    assert set(report) == {"testing", "test-web", "laya-playtest", "laya-testing"}, report
    t = new["packages"]["testing"]["body"]
    assert "laya_drive.py page http://127.0.0.1:8765/index.html --url {tandem_url}" in t
    assert "--step 'read:document.querySelector(\"#time\").textContent'" in t and "the shell expands `$`" in t
    assert "page URL --url {tandem_url} --step" in new["packages"]["test-web"]["body"]
    assert "Never write a browser driver of your own (Marionette" in t and "NO BROWSER" in t
    assert "one page check (or test run) with every main control as a step" in t and "no test copies" in t
    assert t.count("one page check (or test run)") == 1
    lt = new["packages"]["laya-testing"]["body"]
    assert lt.count("write no Laya script of your own for it") == 1, lt
    assert "npx playwright" not in t and t.count("Never download a runtime") == 1, t
    w = new["packages"]["test-web"]["body"]
    assert "--step click:text=Start" in w and "needs the page check only" in w
    assert "needs only the page check" in new["packages"]["laya-playtest"]["body"]
    assert not [e for e in schema.check("packages", new) if "calls TandemLLM" in e]
    assert M.migrate_doc(new)[1] == {}, "a second run changes nothing"
    assert any(c.startswith(M.ROUND3) for ch in report.values() for c in ch) and len(M.NOTE_345) < 200
    # every page-check command in the texts parses with the kit's own command line
    from server.laya_kit import laya_drive
    import shlex
    cmds = re.findall(r"`(python3 \{tandem_scratch\}/laya_drive\.py page [^`]+)`", t)
    assert cmds, t
    for c in cmds:
        argv = shlex.split(c.replace("{tandem_scratch}", "/x"))[2:]
        ap_err = []
        try:
            laya_drive.main(argv + ["--timeout", "1", "--no-laya", "--cdp", "http://127.0.0.1:9"])
        except SystemExit as exc:          # argparse refuses with SystemExit(2) and a usage line
            ap_err.append(exc.code)
        assert ap_err in ([], [0], [1]), (c, ap_err)
    return f"{len(report)} packages migrated to the page check; {len(cmds)} command(s) parse; idempotent"


def test_the_final_message_edits():
    """SRV-229 (UX2, 2026-10-07): A01's final answer quoted "Laya says fix_first (p=0.596, sure:
    false)", "the kit's page check", renderer.info and the removed test hook, and the page kept an
    fps line; the texts asked for each of them. The box's v16 texts (the shipped texts with edits
    15-23 undone, plus the box-only design-review and threejs-product-render lines) migrate to the
    user-facing final message, the overlay rule and read (not quoted) verdicts."""
    old_phrases = (M.STATE_SAY_OLD, M.FINISH_OLD, M.LAYA_QUOTE_OLD, M.LAYA_DONE_OLD,
                   M.PLAY_QUOTE_OLD, M.REVIEW_OLD, *(o for o, _ in M.SKILL_FPS))
    # the shipped texts ask for none of it any more, and say what the final message is
    files = glob.glob(os.path.join(ROOT, "packages", "**", "*.md"), recursive=True)
    for path in files:
        with open(path) as f:
            text = f.read()
        for o in old_phrases:
            assert o not in text, (path, o)
        assert not re.search(r"(?m)^Report: ", text), path
        assert not re.search(r"(?i)\bquote (?:it|laya|the report)", text), path
    with open(os.path.join(ROOT, "packages", "inline", "testing.md")) as f:
        t = f.read()
    assert M.FINISH_NEW in t and M.OVERLAY_RULE in t and M.STATE_SAY_NEW in t
    assert t.index(M.FINISH_NEW) < t.index(M.OVERLAY_RULE)
    # the box's v16 shape: the shipped texts with the edits undone, and the box-only skills
    doc = _seed()
    undo = ((M.STATE_SAY_NEW, M.STATE_SAY_OLD), (M.FINISH_NEW + "\n" + M.OVERLAY_RULE, M.FINISH_OLD),
            (M.LAYA_QUOTE_NEW, M.LAYA_QUOTE_OLD), (M.LAYA_DONE_NEW, M.LAYA_DONE_OLD),
            (M.PLAY_QUOTE_NEW, M.PLAY_QUOTE_OLD), ("\n" + M.HARNESS_RULE, ""), (M.REPORT_LINE, "Report: "),
            ("\n" + M.TOOLS_QUIRK_RULE, ""), (M.DONE_LIST_NEW, M.DONE_LIST_OLD))
    for name, p in doc["packages"].items():
        if isinstance(p.get("body"), str):
            p["body"] = _undo_041(p["body"], "Report: ")
            for new, old in undo:
                p["body"] = p["body"].replace(new, old)
    assert M.FINISH_OLD in doc["packages"]["testing"]["body"]
    new, report = M.migrate_doc(doc)
    assert {"testing", "laya-testing", "laya-playtest", "tandem-harness", "test-web", "test-cli",
            "test-api", "test-data", "test-lib", "coding"} <= set(report), report
    assert M.DONE_LIST_NEW in new["packages"]["coding"]["body"] and M.DONE_LIST_OLD not in new["packages"]["coding"]["body"]
    assert new["packages"]["testing"]["body"].count(M.TOOLS_QUIRK_RULE) == 1
    t = new["packages"]["testing"]["body"]
    assert t.count(M.FINISH_NEW) == 1 and t.count(M.OVERLAY_RULE) == 1 and M.FINISH_OLD not in t
    assert "say in your final message that you removed it" not in t
    assert M.LAYA_DONE_NEW in new["packages"]["laya-testing"]["body"]
    assert M.HARNESS_RULE in new["packages"]["tandem-harness"]["body"]
    assert new["packages"]["test-web"]["body"].count(M.EVIDENCE_LINE) == 1
    assert M.migrate_doc(new)[1] == {}, "a second run changes nothing"
    assert any(c.startswith(M.ROUND4) for ch in report.values() for c in ch) and len(M.NOTE_351) < 200
    # box-only skill lines (design-review, threejs-product-render)
    review = "- [ ] The audit JSON at both widths passes every line of the checklist, " + M.REVIEW_OLD
    product = "Check: screenshot at 1440; `renderer.info.render` reported; fps 55 or more."
    for text, want in ((review, M.REVIEW_NEW), (product, M.SKILL_FPS[1][1])):
        out, ch = M.migrate_text("skill", text)
        assert want in out and ch, (out, ch)
        assert M.migrate_text("skill", out)[1] == []
    return f"{len(report)} packages migrated; no old phrase shipped; idempotent"


def _undo_041(text: str, report: str) -> str:
    """The shipped text with edits 26-29 undone: the evidence line back to `report` + its list with
    Laya's verdict in it, and the old phrases."""
    out = []
    for line in text.split("\n"):
        if line.startswith(M.EVIDENCE_LINE) and line.endswith(M.EVIDENCE_TAIL):
            items = line[len(M.EVIDENCE_LINE):-len(M.EVIDENCE_TAIL)]
            line = report + re.sub(r", and (Laya's verdicts?)\.$", r", \1.", items)
        out.append(line)
    text = "\n".join(out)
    for old, new in ((M.SECOND_OLD, M.SECOND_NEW), (M.DONE_EV_OLD, M.DONE_EV_NEW),
                     (M.SAY_UNAVAIL_OLD, M.SAY_UNAVAIL_NEW), (M.WRITE_UNAVAIL_OLD, M.WRITE_UNAVAIL_NEW)):
        text = text.replace(new, old)
    return text


def test_fx10_opencode_answers_keep_laya_out():
    """FX10 (UX1 O03 on a 0.4.0 test build, opencode): the final answer listed "Evidence: ... Laya second
    check on the suite tail -> passed (p=0.99, sure)"; 24 internal terms, 14 of them Laya. The box's
    v21 texts (edits 15-25 in, 26-29 not) migrate to evidence lines that keep Laya's verdict for the
    model, test-lib's check for the model, the done message in the user's words and a failed Laya
    call kept from the user. Guidance only: nothing filters the model's output."""
    files = glob.glob(os.path.join(ROOT, "packages", "**", "*.md"), recursive=True)
    for path in files:
        with open(path) as f:
            text = f.read()
        assert not re.search(r"(?m)^.*Laya.*second check.*$", text), path
        assert M.REPORT_LINE not in text, path
        assert "say Laya was unavailable" not in text and M.WRITE_UNAVAIL_OLD not in text, path
        for line in text.split("\n"):
            if "Laya's verdict" in line and "final message" in line:
                assert "never names Laya" in line or "never a label" in line \
                    or "not for the final message" in line or "stay out" in line, (path, line)
    doc = _seed()
    for p in doc["packages"].values():
        if isinstance(p.get("body"), str):
            p["body"] = _undo_041(p["body"], M.REPORT_LINE)
    assert M.REPORT_LINE + "the summary lines before and after, the tests added, the example run, " \
        "Laya's verdict." in doc["packages"]["test-lib"]["body"]
    new, report = M.migrate_doc(doc)
    assert {"test-web", "test-cli", "test-api", "test-data", "test-lib", "coding", "laya-scripts",
            "laya-testing"} <= set(report), report
    lib = new["packages"]["test-lib"]["body"]
    assert M.SECOND_V3 in lib and (M.EVIDENCE_LINE + "the summary lines before and after, the tests "
                                    "added, the example run, and Laya's verdict." + M.EVIDENCE_TAIL) in lib
    assert M.DONE_EV_NEW in new["packages"]["coding"]["body"]
    assert M.SAY_UNAVAIL_NEW in new["packages"]["laya-scripts"]["body"]
    lt = new["packages"]["laya-testing"]["body"]
    assert M.WRITE_UNAVAIL_NEW in lt and lt.count("Laya unavailable") == 1, "edit 2 adds no old bullet"
    assert M.migrate_doc(new)[1] == {}, "a second run changes nothing"
    assert any(c.startswith(M.ROUND5) for ch in report.values() for c in ch) and len(M.NOTE_041) < 200
    return f"{len(report)} packages migrated; idempotent"


def test_the_monitor_notes_in_plain_words():
    """SRV-229: the S5 and E2 notes ask for the answer in plain words (tools/monitor_migrate.py),
    in ops/supervisor.yaml, its JSON copy, and a box's stored document with the old texts."""
    from tools import monitor_migrate as MM
    from server.conductor.monitor import policy as mpolicy
    data = mpolicy.load().data
    with open(mpolicy.DEFAULT_JSON) as f:
        assert json.load(f) == data, "policy.default.json is stale: tools/conductor_policy.py --json"
    with open(os.path.join(ROOT, "ops", "supervisor.yaml")) as f:
        yml = f.read()
    for _, _, old, phrase in MM.TEXT_EDITS:
        assert old not in yml and phrase in yml, old
    # a box's v4: the same document with the old phrases
    old_doc = json.loads(json.dumps(data))
    for tid, key, old, phrase in MM.TEXT_EDITS:
        box = old_doc["tasks"][tid]["notes"] if tid == "E2" else old_doc["tasks"][tid]
        box[key] = box[key].replace(phrase, old)
    new, ch = MM.migrate_doc(old_doc, tasks=())
    assert len([c for c in ch if c.startswith("text ")]) == len(MM.TEXT_EDITS), ch
    assert new["tasks"] == data["tasks"]
    assert MM.migrate_doc(new, tasks=())[1] == []
    return f"{len(MM.TEXT_EDITS)} note texts in plain words; idempotent"


def _undo_045(text: str) -> str:
    """A text as the box's v23 had it (edits 30-33 not in)."""
    for line in ("\n- " + M.PROGRESS_LINE, "\n" + M.PROGRESS_LINE, "\n" + M.REFUSED_BULLET):
        text = text.replace(line, "")
    return (text.replace(M.SECOND_V3, M.SECOND_NEW).replace(M.HELPER_TRY_NEW, M.HELPER_TRY)
            .replace(M.HELPER_IMPORT_NEW, M.HELPER_IMPORT_OLD).replace(M.EMPTY_GUARD, ""))


def _laya_server():
    """A loopback server that answers /v1/laya/ask as the route does before any model runs: the
    request checked by server/laya_api.parse (the route's own 422 bodies), a fixed verdict when
    it passes, and a 503 under /down."""
    import http.server
    import threading
    from server import laya_api

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path.startswith("/down/"):
                st, out = 503, {"error": {"message": "the deciders are loading", "type": "overloaded"}}
            else:
                try:
                    laya_api.parse(json.loads(raw), schema.DECIDER_API, 8)
                    st, out = 200, {"label": "passed", "p": 0.99, "sure": True}
                except laya_api.Refused as exc:
                    st, out = exc.status, exc.body()
            data = json.dumps(out).encode()
            self.send_response(st)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_fx13_progress_lines_and_refused_laya_calls():
    files = glob.glob(os.path.join(ROOT, "packages", "**", "*.md"), recursive=True)
    covered = set()
    for path in files:
        with open(path) as f:
            text = f.read()
        keeps_out = any(k in text for k in ("never names Laya", "never a label or a probability",
                                            "Laya's verdicts are yours", "Laya and its scores",
                                            "Laya and its labels"))
        if keeps_out:
            assert M.PROGRESS_LINE in text, ("the final message keeps Laya out, the progress lines "
                                             "do not", path)
            covered.add(os.path.basename(path)[:-3])
        assert "(not for the final message)" not in text, path
    assert {"coding", "testing", "tandem-harness", "laya-testing", "test-lib", "test-web",
            "test-cli", "test-api", "test-data"} <= covered, covered
    # the helper as the model writes it, against the route's own answers
    with open(os.path.join(ROOT, "packages", "inline", "laya-scripts.md")) as f:
        code = re.search(r"```python\n(# \{tandem_scratch\}/laya\.py.*?)```", f.read(), re.S).group(1)
    srv = _laya_server()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "laya.py"), "w") as f:
                f.write(P.with_url(code, base))

            def run(argv, stdin="", url=base, script=None):
                cmd = [sys.executable] + (["-c", script] if script else [os.path.join(d, "laya.py")]) + argv
                return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=30,
                                      cwd=d, env={"PATH": os.environ.get("PATH", ""), "TANDEM_URL": url,
                                                  "PYTHONPATH": d})
            ok = run(["Did every test pass?", "passed,failed"], "16 passed in 0.4s")
            assert ok.returncode == 0 and "LAYA passed" in ok.stdout, (ok.returncode, ok.stderr)
            # O03's call: the evidence piped into a heredoc script never arrived; an empty text
            empty = run([], script="from laya import ask; ask('Did every test pass?', '', "
                                   "choices=['passed', 'failed'])")
            assert empty.returncode == 2 and "LAYA not asked" in empty.stderr, (empty.returncode,
                                                                               empty.stderr)
            assert "unavailable" not in empty.stderr and "Traceback" not in empty.stderr, empty.stderr
            # a malformed request reaches the route: its message and field, exit 2, fix and ask again
            bad = run([], script="from laya import ask; ask('Did it pass?', 'out: 3 rows', "
                                 "choices=['passed', 'passed'])")
            assert bad.returncode == 2, (bad.returncode, bad.stderr)
            assert "LAYA refused this request (HTTP 422): choices: a label is listed twice" in bad.stderr \
                and "(field: choices)" in bad.stderr and "fix it and ask again" in bad.stderr, bad.stderr
            assert "unavailable" not in bad.stderr and "Traceback" not in bad.stderr, bad.stderr
            # the route's own message for an empty text names it and what to send
            raw = run([], script="from laya import ask; ask('Did it pass?', texts=['ok', ' '])")
            assert raw.returncode == 2 and "texts[1] is empty: there is nothing to judge" in raw.stderr, \
                raw.stderr
            # a server error is still this machine's trouble, not the request's
            down = run(["Did it pass?", "passed,failed"], "16 passed", url=base + "/down")
            assert down.returncode == 3 and "LAYA unavailable (HTTP 503" in down.stderr, down.stderr
    finally:
        srv.shutdown()
    # the box's v23 shape (edits 26-29 in, 30-33 not) migrates, idempotent
    doc = _seed()
    for p in doc["packages"].values():
        if isinstance(p.get("body"), str):
            p["body"] = _undo_045(p["body"])
    assert M.PROGRESS_LINE not in json.dumps(doc) and M.HELPER_TRY in doc["packages"]["laya-scripts"]["body"]
    new, report = M.migrate_doc(doc)
    assert set(report) == {"coding", "testing", "tandem-harness", "laya-testing", "laya-scripts",
                           "test-lib", "test-web", "test-cli", "test-api", "test-data"}, report
    assert new == _seed(), "the migrated box texts are the shipped texts"
    assert M.migrate_doc(new)[1] == {}, "a second run changes nothing"
    assert any(c.startswith(M.ROUND6) for ch in report.values() for c in ch) and len(M.NOTE_045) < 200
    return f"{len(covered)} texts keep Laya out of plans and progress lines; helper exit 2 on 422; idempotent"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:50s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
