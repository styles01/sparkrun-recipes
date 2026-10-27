"""The package texts' migration (SRV-163, SRV-171; the UT1 self-test, 2026-10-06).

One set of text edits, applied to the texts the repository ships (`--repo`), to a box's skill
copies (`--skills DIR`) and to a box's saved `packages` document (`--doc FILE`, or `--put URL`
through the config API, which stores it as a new version with a note; the old versions stay and
`restore` brings one back). The same function edits all of them, so the shipped texts and a box's
edited ones cannot drift apart.

The edits:

  1  SRV-163: a literal address in front of TandemLLM's own routes (`/v1/laya`, `/v1/library`,
     `/v1/memory`) becomes `{tandem_url}`, the address the client used (the box's saved packages
     had a literal LAN address, which failed TLS wherever the private root CA is not trusted:
     42 % of a session went to certificate debugging).
  2  SRV-163: the Laya helper fails fast with a truthful message (the call failed on this
     machine, the code is not the cause, continue without the verdict) and exits 3; the curl
     example has a time limit; laya-testing says what to do when a call fails.
  3  SRV-171: `./.tandem/` and `.tandem/` (logs, the helper, the kit, pid files, a state file)
     become `{tandem_scratch}/` (`.tandem` in the project by default, QWEN38_TANDEM_SCRATCH);
     the testing package has the model exclude it from git and delete it before "done"; the
     .gitignore and .dockerignore lines for `.tandem` go (the exclude does it per clone).
  4  SRV-171: the testing package says where scratch goes, that a state hook is scaffolding the
     model removes before "done" (unless the user asked for one), and that it never downloads a
     runtime or an executable.

The second round (SRV-179, SRV-181; the UT2 self-test, 2026-10-06: an Open WebUI agent tuned
its own playtest bot for 18 minutes after the game worked and never answered; an opencode page was
written at 642 s of 733 s, after a Firefox rig and three hung servers):

  5  testing: "Order and budget": a scratch directory left by a cut run is cleaned first, the
     deliverable before any test tooling, a tool that fails twice is dropped, servers in the background with a time limit and killed before "done",
     what is enough, never change the product for one's own test, the final report.
  6  laya-playtest: a playtest when the kit runs here, a check and not a deliverable; a spot only
     the bot fails is not a bug; two runs ending at the same spot end the playtest.
  7  test-web: write the page first, the server rule, and the same stop rule for the bot.
  8  testing: everything the model creates stays inside the project, browser profiles included,
     never /tmp (opencode's run mode refuses a path outside the project and ends the run).

The third round (SRV-195; FX2's live check, 2026-10-06: opencode wrote a pomodoro page at
176 s, found no node, npx or Chrome on its PATH, only Firefox, and spent seven minutes on a
Marionette driver of its own until the run's limit; the kit's headless Chrome was there all along):

  9  testing: a page is checked with the kit's page check (`laya_drive.py page`: its own headless
     Chrome, real clicks and keys, values read back, screenshots at 1280 and 390 px, console
     errors, Laya's verdict, Chrome closed, about 5 s); never a browser driver of one's own;
     NO BROWSER or two failed checks = read the code and say so; after the deliverable exists the
     verification is check, fix, re-check, then the report.
 10  test-web: the browser tool or the page check for every step (the same steps as `--step`).
 11  test-web and laya-playtest: a game, or a flow of many steps toward a goal, is played by Laya;
     a one-screen tool (a timer, a form, a calculator) needs the page check only.
 12  laya-testing and testing (the live n=1 of edits 9-11: 473 s, of which ~280 s checks: four Laya
     scripts of its own and a helper beside the page check, a 3-second copy of the page): the page
     check's `laya` field is Laya's verdict for a page, no script of one's own; one page check
     with every main control as a step; no test copies of the product.
 13  testing and test-web: the page-check commands carry `--url {tandem_url}`: on Open WebUI's
     terminal host the kit's default (http://127.0.0.1:8000) is not TandemLLM and Laya's verdict
     came back 404 (FX3's live chat, 2026-10-06).
 14  testing: the example step in single quotes, and why: in double quotes the shell expands `$`
     (an Open WebUI agent's `expect:... === '$101.40'` reached the kit as '01.40' and it blamed
     the kit).

The fourth round (SRV-229; the UX2 normal-user run, 2026-10-07: A01's answer to "create a
pagoda garden" quoted "Laya says fix_first (p=0.596, sure: false)", "the kit's page check",
"renderer.info: 56 draw calls", "I removed the tandem_state/tandem_reset test hook ... deleted the
.tandem scratch folder", and the page kept an "fps · calls · tris" line for the check; the texts
asked for it: "quote Laya's verdicts", "say in your final message that you removed it", "Report:
... Laya's verdicts", "fps reported"):

 15  testing: the state hook goes without a word in the final message.
 16  testing: the final message is for the user (what was built, where, how to open and use it, one
     plain sentence on how it was checked, what is untested, one next step); Laya, its labels and
     probabilities, the page check, the kit, hooks, scratch, draw calls and fps stay out of it
     unless the user asks how it was tested; nothing named that the result does not show.
 17  testing: what was added only for the check (an fps or stats readout, a debug panel, per-frame
     logging) goes before "done", or stays behind a `?debug` flag.
 18  laya-testing: Laya's verdicts are read, not quoted; they are the model's, not the user's.
 19  test-web, test-cli, test-api, test-data, test-lib: their "Report:" line is the evidence the
     model keeps for itself; the final message says it in plain words.
 20  laya-playtest: the report is read, not quoted.
 21  design-review: Laya's verdict on the score table is read, not quoted.
 22  threejs-scene, threejs-product-render, canvas-2d: draw calls, triangles and fps are read
     (through the state hook), never an on-screen readout in the delivered page.
 23  tandem-harness: the final message speaks to the user in plain words.
 24  testing: a quirk of the model's own test tools (a kit step, the judge, a screenshot it could not
     open) is not news for the user (the first live run after 15-23: an answer still explained "the
     test kit's type step" and "the judge flagged two scenarios").
 25  coding: the done message's list in the user's words: a surprise in the product and its cause,
     not the test tools'; Laya's labels and probabilities, the kit, the scratch folder and state
     hooks stay out of it (opencode answers named Laya, `.tandem` and the scratch folder 26 to 43
     times in three refinements of one session).

The fifth round, 0.4.1 (FX10; UX1 O03 on a 0.4.0 test build, opencode "refactor report.py": the final answer
listed "Evidence: ... Laya second check on the suite tail -> passed (p=0.99, sure)" and "Scratch
`.tandem/` removed", 24 internal terms, 14 of them Laya; edit 19 had kept "Laya's verdict" in the
very list the model reports from, test-lib called the call "a second check", and the always-on
coding text asked for "evidence in the message"):

 26  test-web, test-cli, test-api, test-data, test-lib: the evidence line is the model's own, with
     Laya's verdict last, and a sentence that the final message never names Laya, a verdict or a
     probability.
 27  test-lib: Laya reads the suite's tail as a check for the model, not "a second check".
 28  coding: the evidence in the done message is in the user's words, Laya's verdicts stay out.
 29  laya-scripts and laya-testing: when a call fails, the model notes it in its check output and
     goes on; it does not tell the user.

The sixth round (FX13, 2026-10-08; UX1 O03 on 0.4.1-test: the closing message no longer named
Laya, but 7 lines between tool calls did: "3. Ask Laya about the diff evidence", "Now the Laya
verdicts", "Laya is unavailable on this machine (transport error)". The texts said only that the
final message keeps Laya out. And that "unavailable" was a 422: the model piped its evidence into
a heredoc script, sent an empty text, and the helper called the refusal this machine's network):

 30  test-*, laya-testing, tandem-harness, coding: the plan and every line between tool calls
     reach the user too; a check is named by what it checks, never Laya, a verdict or a
     probability.
 31  test-lib: step 6 is a check for the model that no line it writes names.
 32  laya-scripts: the helper refuses an empty text before it asks (exit 2, what to do), and a
     4xx from Laya is the request's fault: the server's message and field, exit 2, "fix it and
     ask again"; only a refused connection, TLS, a timeout, a 429 or a 5xx is "unavailable"
     (exit 3).
 33  laya-testing: a refused request (exit 2) is the model's to fix, not the network.

Every edit is idempotent: a text already migrated comes out unchanged.

  python tools/packages_migrate.py --repo                 # rewrite packages/ in this checkout
  python tools/packages_migrate.py --skills DIR [--dry]   # SKILL.md files under DIR
  python tools/packages_migrate.py --doc 0007.json        # print what would change
  python tools/packages_migrate.py --put http://127.0.0.1:8000 --token-file secrets.env [--dry]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor.packages import literal_api_hosts  # noqa: E402

NOTE = ("TandemLLM's routes via {tandem_url}; Laya helper fails fast; scratch {tandem_scratch} excluded "
        "from git and deleted before done; state hooks removed; no downloads")
# the store keeps a version note under 200 characters; a box that had the first edits gets this one
ROUND2 = ("testing: order", "testing: nothing outside", "laya-playtest: a check", "laya-playtest: the stop",
          "test-web: the")
NOTE_343 = ("0.3.43-test: deliverable first, a test budget, servers in the background and stopped, a cut "
            "run's scratch cleaned, the playtest stop rule (SRV-179, SRV-181)")
ROUND3 = ("testing: the kit's page check", "testing: a page runs", "test-web: the browser tool or",
          "test-web: real input", "test-web: the page check", "test-web: Laya plays", "laya-playtest: a one-screen",
          "laya-testing: the page check", "testing: one page check", "testing: the page check asks",
          "test-web: the page check asks", "testing: a page-check step in single quotes")
NOTE_351 = ("0.3.51-test: the final message speaks to the user (no Laya scores, kit, hooks or scratch); "
            "check-only overlays removed before done (SRV-229)")
NOTE_041 = ("0.4.1: test evidence lines keep Laya's verdicts for the model, the done message is in the "
            "user's words, a failed Laya call is not news for the user (FX10)")
ROUND5 = ("test-*: the evidence is the model's", "test-lib: a check for the model", "coding: evidence in the user",
          "laya: a failed call is not news")
EVIDENCE_LINE = "Your evidence, for you: "
EVIDENCE_TAIL = (" The final message says in plain words what you ran and what it showed; it never names "
                 "Laya, a verdict or a probability.")
SECOND_OLD = "6. Let Laya read the suite's tail as a second check:"
SECOND_NEW = "6. Let Laya read the suite's tail, a check for you (not for the final message):"
DONE_EV_OLD = '"Done" only with evidence in the message: what was run, how long, what state was read, what was seen.'
DONE_EV_NEW = ('"Done" only with evidence in the message, in the user\'s words: what was run, how long, what state '
               'was read, what was seen. Laya\'s verdicts are yours and stay out of it.')
SAY_UNAVAIL_OLD = "continue with your own evidence and say Laya was unavailable."
SAY_UNAVAIL_NEW = "continue with your own evidence; the user does not need to hear about it."
WRITE_UNAVAIL_OLD = 'write "Laya unavailable" once and go on with your own evidence.'
WRITE_UNAVAIL_NEW = ('print "Laya unavailable" once in your check output and go on with your own evidence; it is '
                     'not news for the user.')
ROUND4 = ("testing: the state hook goes", "testing: the final message", "testing: check-only overlays",
          "laya-testing: verdicts read", "test-*: the report", "laya-playtest: the report is read",
          "design-review: Laya's verdict read", "skills: fps read", "tandem-harness: the final message",
          "testing: test-tool quirks", "coding: the done list")
NOTE_345 = ("0.3.45-test: a page is checked with the kit's page check (its own headless Chrome, Laya at "
            "{tandem_url}), never a driver of one's own; one check, fix, check, report (SRV-195)")

# 2: the helper's call and its command line (after edit 3 renamed its path)
HELPER_OLD = '''    return json.load(urllib.request.urlopen(req, timeout=90))

if __name__ == "__main__":   # python3 {tandem_scratch}/laya.py "question" pass,fail < evidence.txt
    choices = sys.argv[2].split(",")'''
HELPER_NEW = '''    try:
        return json.load(urllib.request.urlopen(req, timeout=90))
    except (OSError, ValueError) as e:   # refused, TLS, timeout, not JSON: this machine's network
        print(f"LAYA unavailable ({type(e).__name__}: {e}). The call failed on this machine, "
              "your code is not the cause: do not debug the connection or certificates, "
              "continue with your own evidence and say Laya was unavailable.", file=sys.stderr)
        raise SystemExit(3)

if __name__ == "__main__":   # python3 {tandem_scratch}/laya.py "question" pass,fail < evidence.txt
    choices = sys.argv[2].split(",")'''
CURL_OLD = '''curl -s "${TANDEM_URL:-{tandem_url}}/v1/laya/ask"'''
CURL_NEW = '''curl -sS --max-time 60 "${TANDEM_URL:-{tandem_url}}/v1/laya/ask"'''
FAIL_ANCHOR = '''- failure, no or mismatch is a bug to explain before the next edit; `"sure": false` means read the evidence yourself.'''
FAIL_BULLET = ('''- A call that fails (connection refused, a TLS or certificate error, a timeout, a 5xx) is this '''
               '''machine's network, not your code: write "Laya unavailable" once and go on with your own '''
               '''evidence. At most one retry; never debug the transport (no openssl, no CA hunting, no -k).''')

# 4: the testing package's rules
LOGS_OLD = ('''- Logs: write build and run output to {tandem_scratch}/build.log and {tandem_scratch}/run.log. '''
            '''After a run, `tail -n 60` the log in a tool call.''')
SCRATCH_RULE_V1 = ("- Scratch: logs, helper scripts and kits go to {tandem_scratch}/ (`mkdir -p {tandem_scratch}` "
                   "first), outside the project. The project gets only what the user asked for; never list "
                   "scratch files as deliverables.")
SCRATCH_RULE = ('''- Scratch: logs, helper scripts and kits go to {tandem_scratch}/. Create it once with '''
                '''`mkdir -p {tandem_scratch} && (git rev-parse --git-dir >/dev/null 2>&1 && echo '{tandem_scratch}/' >> '''
                '''"$(git rev-parse --git-dir)/info/exclude"; true)`, so git never lists it, and delete it (`rm -rf '''
                '''{tandem_scratch}`) as your last step before "done". The project gets only what the user asked for; '''
                '''never list scratch files as deliverables.''')
# 34 (FX15, UX3 2026-10-08): A01 ran `rm -rf /home/user/pago/.tandem` in the user's own project (their
# test kit of 18:43), by the leftover rule below, right after writing "let me not touch pago's files".
# The scratch is this conversation's own folder (`.tandem/<session token>`, packages.scratch_for),
# marked when this run makes it; only a marked folder is deleted, never one that was there before
SCRATCH_RULE_V3 = ("""- Scratch: logs, helper scripts and kits go to {tandem_scratch}/, this conversation's own """
                   """folder. Create it once with `[ -d {tandem_scratch} ] || { mkdir -p {tandem_scratch} && """
                   """echo made-by-this-run > {tandem_scratch}/.made-by-this-run; }; (git rev-parse --git-dir """
                   """>/dev/null 2>&1 && echo '{tandem_scratch}/' >> "$(git rev-parse --git-dir)/info/exclude"; """
                   """true)`, so git never lists it. As your last step before "done", delete it only when this """
                   """run made it: `[ -f {tandem_scratch}/.made-by-this-run ] && rm -rf {tandem_scratch}`. Never """
                   """delete a folder or file you did not create, another `.tandem` folder included: what was """
                   """there before you started is the user's. The project gets only what the user asked for; """
                   """never list scratch files as deliverables.""")
LEFTOVER_OLD = ("""- A {tandem_scratch} left by an earlier run that was cut off: stop the server its server.pid """
                """names (`kill $(cat {tandem_scratch}/server.pid) 2>/dev/null`) and delete it before you start.""")
LEFTOVER_NEW = ("""- A {tandem_scratch} that is there when you start is this conversation's, from a run that was """
                """cut off: stop the server its server.pid names (`kill $(cat {tandem_scratch}/server.pid) """
                """2>/dev/null`) and go on using it. Delete nothing at the start, in it or anywhere else.""")
FILES_ANCHOR = ("""- If you must replace a file you did not create, copy it first (index.html.bak) and say so in """
                """the answer.""")
NO_DELETE_RULE = ("""- Never delete a file or folder you did not create in this conversation (`rm`, `rm -rf`, `git """
                  """clean`, a move over it), a `.tandem` folder included: one that was there when you started """
                  """is the user's, whatever it holds.""")
NOTE_047 = ("0.4.7: the scratch is the conversation's own folder (.tandem/<session>), deleted only when this run "
            "made it; a leftover is reused, never deleted; files-care: never delete what you did not create (FX15)")
ROUND7 = ("testing: scratch marked", "testing: a leftover is reused", "files-care: never delete")
STATE_OLD = ('''Keep it under 4 KB. Read it after real input, never before.''')
STATE_NEW = ('''Keep it under 4 KB. Read it after real input, never before. It is test scaffolding: wrap it in '''
             '''`tandem-test-hook:start` and `tandem-test-hook:end` comments and remove it as your last edit '''
             '''before "done" (then run the quick check once more), unless the user asked for a debug or '''
             '''state API; say in your final message that you removed it.''')
TESTS_ANCHOR = '''- Tests: run the real test command and let its summary line appear in the tool result.'''
RUNTIME_RULE = ('''- Runtimes: use what is installed (`command -v node npx python3 bun deno`; for a page, the '''
                '''browser tool or `npx playwright`). Never download a runtime or any executable (no curl or '''
                '''wget of a binary, no install script) unless the user asks; if nothing can run the code, '''
                '''say so and test what you can.''')


# 5: the testing package's order and budget
BUDGET_ANCHOR = "Never test by writing to the state you then read. Input in, state out."
BUDGET_HEAD = "Order and budget (the user waits for the deliverable"
BUDGET = """
Order and budget (the user waits for the deliverable, not for your test rig):
- A {tandem_scratch} left by an earlier run that was cut off: stop the server its server.pid names (`kill $(cat {tandem_scratch}/server.pid) 2>/dev/null`) and delete it before you start.
- First the deliverable: write the files the user asked for before any test tooling (test pages, servers, browser rigs, installs).
- Verify with what already works here: your browser tool, else `npx playwright` or the Laya kit when `command -v` finds them; else read the code and say which check was not possible. A tool that fails twice (a browser that does not start, an image you cannot read, a server that hangs) is dropped: say so in one sentence and go on.
- A server you start runs in the background with a time limit and its pid in scratch: `timeout 600 python3 -m http.server 8765 >{tandem_scratch}/server.log 2>&1 & echo $! > {tandem_scratch}/server.pid`. Stop it (`kill $(cat {tandem_scratch}/server.pid)`) before "done"; never run a server in the foreground.
- Enough is: the deliverable loads or runs, zero console errors, and one run with real input passes (a game: start, move, jump; a page: the main controls; code: the tests). A bot that plays a whole game is optional. When your own test tool fails the same way twice, stop tuning it: it is not the deliverable; report that spot as untested.
- Never change the product only to make your own test pass; if a test made you change its behaviour, say so in the final message.
- Finish with what you built and where it is (paths), what you verified and how, and what is untested."""
# 8: nothing outside the project (opencode's run mode refuses a path outside it and ends the run:
# a Firefox profile in /tmp ended one at 349 s)
TMP_ANCHOR = ("- A server you start runs in the background with a time limit and its pid in scratch: "
              "`timeout 600 python3 -m http.server 8765 >{tandem_scratch}/server.log 2>&1 & echo $! > "
              "{tandem_scratch}/server.pid`. Stop it (`kill $(cat {tandem_scratch}/server.pid)`) before "
              "\"done\"; never run a server in the foreground.")
TMP_RULE = ("- Everything you create stays inside the project: temporary files, browser profiles and logs "
            "go in {tandem_scratch}, never /tmp (a client may refuse a path outside the project and end "
            "the run).")
# 9: the kit's page check is the browser here (SRV-195)
VERIFY_OLD = ("- Verify with what already works here: your browser tool, else `npx playwright` or the Laya "
              "kit when `command -v` finds them; else read the code and say which check was not possible. "
              "A tool that fails twice (a browser that does not start, an image you cannot read, a server "
              "that hangs) is dropped: say so in one sentence and go on.")
VERIFY_NEW = ("- Check a page with the kit's page check, the browser here: `mkdir -p {tandem_scratch} && curl -sf "
              "{tandem_url}/v1/laya/kit/laya_drive.py -o {tandem_scratch}/laya_drive.py`, then `python3 "
              "{tandem_scratch}/laya_drive.py page http://127.0.0.1:8765/index.html --step click:#start "
              "--step wait:2000 --step \"read:document.querySelector('#time').textContent\"` (steps: "
              "click:SELECTOR or click:text=Label, key:Space*300, type:#field=text, wait:MS, read:JS, "
              "expect:JS, shot:NAME). It starts its own headless Chrome, uses real clicks and keys, "
              "reads values back, takes screenshots at 1280 and 390 px, lists the console errors, asks "
              "Laya for a verdict and closes Chrome, in about 5 s. Your browser tool, when you have one, "
              "does the same.\n"
              "- Never write a browser driver of your own (Marionette, WebDriver, Selenium, a DevTools "
              "client, `firefox --screenshot`). When the page check says NO BROWSER, or fails twice, "
              "read the code instead and say the browser check was not possible. Any other tool that "
              "fails twice (an image you cannot read, a server that hangs) is dropped the same way: say "
              "so in one sentence and go on.\n"
              "- After the deliverable exists, verification is: check, fix what the check shows, check "
              "again, then the final message. Write no new test tooling after the first check; the user "
              "waits for every extra script.")
RUNTIME_OLD_PAGE = "for a page, the browser tool or `npx playwright`"
RUNTIME_NEW_PAGE = "for a page, the kit's page check below or your browser tool"
# 10: test-web, the steps through the browser tool or the page check
WEB1_V2 = ("1. Write the page first. Serve it in the background with a time limit (the testing package's "
           "server rule), open it in the browser tool; zero console errors first.")
WEB1_V3 = ("1. Write the page first. Serve it in the background with a time limit (the testing package's "
           "server rule). Open it in your browser tool, else run the kit's page check (`python3 "
           "{tandem_scratch}/laya_drive.py page URL --step ...`, the testing package); zero console "
           "errors first.")
WEB3_OLD = ("3. Real input only: press_key with a hold time, clicks at screenshot coordinates. Wait 300 ms "
            "or more, then read window.__state() with evaluate_script. Never set keys, flags or positions "
            "in a script.")
WEB3_NEW = ("3. Real input only: press_key with a hold time, clicks at screenshot coordinates (page check: "
            "`--step key:ArrowRight*300`, `--step click:#start` or `--step click:text=Start`). Wait 300 ms "
            "or more, then read window.__state() (evaluate_script, or `--step \"read:window.__state()\"`). "
            "Never set keys, flags or positions in a script.")
WEB5_OLD = "5. Screenshots at 1280 and 390 px: start, middle, end."
WEB5_NEW = ("5. Screenshots at 1280 and 390 px: start, middle, end (the page check takes 1280 and 390 at "
            "the end; `--step shot:middle` for more).")
# 11: Laya plays games and long flows; a one-screen tool needs the page check only
WEB8_V2 = ("8. A game or a UI flow: let Laya play it, 3 episodes or more (laya-playtest: "
           "`{tandem_scratch}/laya_drive.py browser`); a death or stuck spot that real keys cannot avoid "
           "either is a bug. Two runs that end at the same spot: stop and report it as untested.")
WEB8_V3 = ("8. A game, or a flow of many steps toward a goal: let Laya play it, 3 episodes or more "
           "(laya-playtest: `{tandem_scratch}/laya_drive.py browser`); a death or stuck spot that real "
           "keys cannot avoid either is a bug. Two runs that end at the same spot: stop and report it as "
           "untested. A one-screen tool (a timer, a form, a calculator) needs the page check only.")
PLAY_V2 = ('Playtest a game, UI flow, CLI or API flow you build this way when the kit runs here (a check, '
           'not a deliverable; the testing package\'s budget applies):')
PLAY_V3 = ('Playtest a game, or a UI, CLI or API flow of many steps toward a goal, this way when the kit runs '
           'here (a check, not a deliverable; the testing package\'s budget applies). A one-screen tool '
           '(a timer, a form, a calculator) needs only the page check (`laya_drive.py page`):')

# 12: one page check is the whole check of a page (the live n=1 of edits 9-11)
LAYA_RUN_OLD = ("- After every run that produced evidence (a log, an output, the console, a state dump), your "
                "test script asks Laya and prints the verdict.")
LAYA_RUN_NEW = ("- After every run that produced evidence (a log, an output, the console, a state dump), your "
                "test script asks Laya and prints the verdict. A page checked with the kit's page check has it "
                "already (the report's `laya` field): quote it, write no Laya script of your own for it.")
CHECK_ONE_OLD = ("- After the deliverable exists, verification is: check, fix what the check shows, check again, "
                 "then the final message. Write no new test tooling after the first check; the user waits for "
                 "every extra script.")
CHECK_ONE_NEW = ("- After the deliverable exists, verification is: one page check (or test run) with every main "
                 "control as a step in the same call, fix what it shows, check again, then the final message, "
                 "in a few minutes. Write no new test tooling after the first check and no test copies of the "
                 "product (a page with shorter timers, a patched build); say what a long wait would show "
                 "instead. The user waits for every extra script.")

# 13: the page check asks Laya where the kit came from
PAGE_CMD_OLD = "{tandem_scratch}/laya_drive.py page http://127.0.0.1:8765/index.html --step"
PAGE_CMD_NEW = "{tandem_scratch}/laya_drive.py page http://127.0.0.1:8765/index.html --url {tandem_url} --step"
WEB_CMD_OLD = "(`python3 {tandem_scratch}/laya_drive.py page URL --step ...`, the testing package)"
WEB_CMD_NEW = "(`python3 {tandem_scratch}/laya_drive.py page URL --url {tandem_url} --step ...`, the testing package)"

# 14: a step in single quotes (the shell expands `$` inside double quotes)
QUOTE_OLD = '''--step "read:document.querySelector('#time').textContent"` (steps: '''
QUOTE_NEW = ('''--step 'read:document.querySelector("#time").textContent'` (a step in single quotes: inside '''
             '''double quotes the shell expands `$`; steps: ''')

# 6: laya-playtest's stop rule
PLAY_OLD = 'Playtest every game, UI flow, CLI or API flow you build this way before "done":'
PLAY_NEW = ('Playtest a game, UI flow, CLI or API flow you build this way when the kit runs here (a check, '
            'not a deliverable; the testing package\'s budget applies):')
REPORT_OLD = ("4. Report: deaths with place, cause and the decisions before them; stuck spots. An unavoidable "
              "death, a stuck spot or a crash is a bug: fix, rerun, quote the report.")
REPORT_NEW = ("4. Report: deaths with place, cause and the decisions before them; stuck spots. A crash, or a "
              "death or stuck spot that a player with real keys cannot avoid either, is a bug: fix the build, "
              "rerun, quote the report. A spot where only the bot fails is not a bug: report it. Two runs that "
              "end at the same spot with no change to the build: stop playtesting and report what works and "
              "what is untested.")
# 15-23 (SRV-229): the final message speaks to the user
STATE_SAY_OLD = "unless the user asked for a debug or state API; say in your final message that you removed it."
STATE_SAY_NEW = ("unless the user asked for a debug or state API. It is yours: the final message does not "
                 "mention it.")
FINISH_OLD = "- Finish with what you built and where it is (paths), what you verified and how, and what is untested."
FINISH_NEW = ('- The final message is for the user, who never saw your tools: what you built and where it is (the '
              'path, how to open or run it), how to use it, one plain sentence on how you checked it ("I opened '
              'it in a browser: it loads with no errors and the controls respond"), what is untested, and at '
              'most one next step; for a build about 250 words or fewer. Laya and its labels and '
              'probabilities, the page check, the kit, state hooks, the scratch folder, draw calls and fps are '
              'your tools: leave them out unless the user asks how you tested. Describe what the result shows, '
              'never a feature the check did not show.')
OVERLAY_RULE = ('- Anything you added only for your check (an fps, draw-call or stats readout on screen, a debug '
                'panel, logging every frame) comes out before "done", or stays behind a `?debug` URL flag: the '
                'delivered page shows only what the user asked for.')
LAYA_QUOTE_OLD = "(the report's `laya` field): quote it, write no Laya script of your own for it."
LAYA_QUOTE_NEW = "(the report's `laya` field): read it, write no Laya script of your own for it."
LAYA_DONE_OLD = "- Before \"done\", run the check script and quote Laya's verdicts."
LAYA_DONE_NEW = ("- Before \"done\", run the check script and read Laya's verdicts. They are for you: the final "
                 "message says in plain words what was checked, never a label or a probability.")
PLAY_QUOTE_OLD = "fix the build, rerun, quote the report."
PLAY_QUOTE_NEW = "fix the build, rerun, read the report."
REVIEW_OLD = "and Laya's verdict on the table is quoted."
REVIEW_NEW = ("and Laya's verdict on the table is read (it is yours: the final answer says in plain words how "
              "the page was checked).")
HARNESS_ANCHOR = "- State lines may appear in the client; they are not your text."
HARNESS_RULE = ("- Your final message speaks to the user in plain words: what you built, where, how to use it, how "
                "you checked it. The runtime's notes, Laya and its scores, hooks and scratch stay out of it.")
SKILL_FPS = (("- [ ] Draw calls, triangles and fps reported from `renderer.info`.",
              "- [ ] Draw calls, triangles and fps read from `renderer.info` (through the state hook; the "
              "delivered page shows no fps or stats readout)."),
             ("`renderer.info.render` reported; fps 55 or more.",
              "`renderer.info.render` read; fps 55 or more (read, never shown on the page)."),
             ("- [ ] Measured fps at 1280x800 and at 390x844 reported (target 55 or more).",
              "- [ ] Measured fps at 1280x800 and at 390x844 (target 55 or more), read through the state hook, "
              "no fps readout left on screen."))
TOOLS_QUIRK_RULE = ("- A quirk of your own test tools (a kit step that behaves oddly, the judge's doubt, a screenshot you "
                    "could not open) is yours to work around, not news for the user: leave it out of the final "
                    "message.")
DONE_LIST_OLD = "what the use test did (inputs, duration, state read, screenshots); every surprise and its cause."
DONE_LIST_NEW = ("what the use test did (inputs, duration, state read, screenshots), in the user's words; every "
                 "surprise in the product and its cause. Laya's labels and probabilities, the kit, the scratch "
                 "folder and state hooks stay out of it, and so do the quirks of your own test tools.")
REPORT_LINE = "Report (the evidence you keep; the final message says it in plain words): "
# 7: test-web
WEB1_OLD = "1. Serve it, open it in the browser tool; zero console errors first."
WEB1_NEW = ("1. Write the page first. Serve it in the background with a time limit (the testing package's "
            "server rule), open it in the browser tool; zero console errors first.")
WEB8_OLD = ("8. A game or a UI flow: let Laya play it, 3 episodes or more (laya-playtest: "
            "`{tandem_scratch}/laya_drive.py browser`); every death it could not avoid and every stuck spot "
            "is a bug.")
WEB8_NEW = ("8. A game or a UI flow: let Laya play it, 3 episodes or more (laya-playtest: "
            "`{tandem_scratch}/laya_drive.py browser`); a death or stuck spot that real keys cannot avoid "
            "either is a bug. Two runs that end at the same spot: stop and report it as untested.")

# 30-33 (FX13): the plan and the progress lines name no Laya; the helper says why a request was refused
NOTE_045 = ("0.4.5: plans and progress lines between tool calls name no Laya; the Laya helper says what was "
            "wrong with a refused request (4xx, exit 2) instead of 'unavailable' (FX13)")
ROUND6 = ("all text: plans and progress", "test-lib: a check no line names", "laya helper: a refused request",
          "laya-testing: a refused request")
PROGRESS_LINE = ("Your plan and every line you write between tool calls reach the user too: name a check by "
                 "what it checks (\"checking the output against the expected file\", \"running the tests\"), "
                 "never Laya, a verdict or a probability.")
SECOND_V3 = "6. Let Laya read the suite's tail, a check for you that no line you write names:"
ASK_DEF = "def ask(question, text=None, texts=None, **kw):   # kw: choices, scale, threshold\n"
EMPTY_GUARD = (
    "    if not texts and not (text or \"\").strip():   # nothing to judge: the evidence never arrived\n"
    "        print(\"LAYA not asked: the evidence text is empty, so there is nothing to judge. Pass the \"\n"
    "              \"output itself (save it to a file and read that; a script fed by a heredoc gets no \"\n"
    "              \"piped stdin).\", file=sys.stderr)\n"
    "        raise SystemExit(2)\n")
HELPER_TRY = (
    "    try:\n"
    "        return json.load(urllib.request.urlopen(req, timeout=90))\n"
    "    except (OSError, ValueError) as e:   # refused, TLS, timeout, not JSON: this machine's network")
HELPER_TRY_NEW = (
    "    try:\n"
    "        return json.load(urllib.request.urlopen(req, timeout=90))\n"
    "    except urllib.error.HTTPError as e:   # Laya answered: a 4xx says what was wrong with this request\n"
    "        try:\n"
    "            err = json.loads(e.read() or b\"{}\").get(\"error\") or {}\n"
    "        except ValueError:\n"
    "            err = {}\n"
    "        if 400 <= e.code < 500 and e.code != 429:\n"
    "            print(f\"LAYA refused this request (HTTP {e.code}): {err.get('message') or e.reason}\"\n"
    "                  + (f\" (field: {err['param']})\" if err.get(\"param\") else \"\")\n"
    "                  + f\". The request is wrong, not the network: fix it and ask again \"\n"
    "                  f\"(`curl -s {URL[:-4]}` shows the contract).\", file=sys.stderr)\n"
    "            raise SystemExit(2)\n"
    "        print(f\"LAYA unavailable (HTTP {e.code}: {err.get('message') or e.reason}). The call failed \"\n"
    "              \"on this machine, your code is not the cause: continue with your own evidence; \"\n"
    "              \"the user does not need to hear about it.\", file=sys.stderr)\n"
    "        raise SystemExit(3)\n"
    "    except (OSError, ValueError) as e:   # refused, TLS, timeout, not JSON: this machine's network")
HELPER_IMPORT_OLD = "import json, os, sys, urllib.request\n"
HELPER_IMPORT_NEW = "import json, os, sys, urllib.error, urllib.request\n"
REFUSED_BULLET = ("- A call refused with a 4xx (the helper prints \"LAYA refused this request\" or \"LAYA not "
                  "asked\" and exits 2) is your request, not the network: read the message, fix the call (an "
                  "empty text, a missing question, a bad choices list) and ask again.")


def _once(text: str, anchor: str, add: str, changes: list, what: str) -> str:
    """`add` on its own line after `anchor`, unless it is there already."""
    if add in text or anchor not in text:
        return text
    changes.append(what)
    return text.replace(anchor, anchor + "\n" + add, 1)


def _after_line(text: str, needle: str, add: str, changes: list, what: str) -> str:
    """`add` on its own line after the line that holds `needle`, unless it is there already."""
    if add in text or needle not in text:
        return text
    j = text.find("\n", text.find(needle))
    changes.append(what)
    return text + "\n" + add if j < 0 else text[:j] + "\n" + add + text[j:]


def migrate_text(name: str, text: str) -> tuple[str, list[str]]:
    """(the text with the edits, what changed). Idempotent."""
    changes: list[str] = []
    out = text
    # 1: literal addresses of TandemLLM's routes
    for host in literal_api_hosts(out):
        out = re.sub(re.escape(host) + r"(?=\}?[\"']?\s*\)?\s*\+?\s*[\"']?/(?:s/[A-Za-z0-9_-]+/)?"
                     r"v1/(?:laya|library|memory)\b)", "{tandem_url}", out)
        changes.append(f"host {host} -> {{tandem_url}}")
    # 13: the same host handed to the kit (`--url HOST`, Laya's address for the page check)
    for host in set(re.findall(r"(?<=--url )(https?://[^\s`'\"]+)", out)) & set(literal_api_hosts(text) or ()):
        out = out.replace("--url " + host, "--url {tandem_url}")
        changes.append(f"--url {host} -> {{tandem_url}}")
    # 3: the scratch directory (the ignore lines first: they name the directory, not a file in it)
    for old, new in (("\\n.env\\n.tandem/\\n'", "\\n.env\\n'"), (", `.env`, `.tandem`.", ", `.env`.")):
        if old in out:
            out = out.replace(old, new)
            changes.append("ignore line for .tandem removed")
    n0 = out
    out = re.sub(r"mkdir -p (?:\./)?\.tandem\b/?", "mkdir -p {tandem_scratch}", out)
    out = out.replace('sys.path.insert(0, ".tandem")', 'sys.path.insert(0, "{tandem_scratch}")')
    out = re.sub(r"(?<![\w/{])\./\.tandem/", "{tandem_scratch}/", out)
    out = re.sub(r"(?<![\w/{.])\.tandem/", "{tandem_scratch}/", out)
    if out != n0:
        changes.append(".tandem/ -> {tandem_scratch}/")
    # 2: Laya fails fast and truthfully
    if HELPER_OLD in out:
        out = out.replace(HELPER_OLD, HELPER_NEW)
        changes.append("laya helper fails fast (exit 3)")
    if CURL_OLD in out:
        out = out.replace(CURL_OLD, CURL_NEW)
        changes.append("curl example has a time limit")
    if FAIL_BULLET.replace(WRITE_UNAVAIL_OLD, WRITE_UNAVAIL_NEW) not in out:     # 29 rewords it
        out = _once(out, FAIL_ANCHOR, FAIL_BULLET, changes, "laya-testing: what to do when a call fails")
    # 4: the testing package's rules
    if SCRATCH_RULE_V1 in out:                       # the first wording (outside the project)
        out = out.replace(SCRATCH_RULE_V1, SCRATCH_RULE)
        changes.append("testing: scratch inside the project, excluded and deleted")
    if LOGS_OLD in out and SCRATCH_RULE not in out and SCRATCH_RULE_V3 not in out:
        out = out.replace(LOGS_OLD, SCRATCH_RULE + "\n" + LOGS_OLD, 1)
        changes.append("testing: the scratch rule")
    if STATE_OLD in out and STATE_NEW not in out and "tandem-test-hook:start" not in out:  # 15 rewords it
        out = out.replace(STATE_OLD, STATE_NEW, 1)
        changes.append("testing: the state hook is scaffolding")
    if "Never download a runtime or any executable" not in out:    # edit 9 rewords its page part
        out = _once(out, TESTS_ANCHOR, RUNTIME_RULE, changes, "testing: no downloads")
    # 5, 6, 7: the round-2 budget (SRV-179, SRV-181)
    if BUDGET_HEAD not in out:                   # the block may have grown since (edit 8)
        out = _once(out, BUDGET_ANCHOR, BUDGET, changes, "testing: order and budget")
    out = _once(out, TMP_ANCHOR, TMP_RULE, changes, "testing: nothing outside the project")
    for old, new, what in ((PLAY_OLD, PLAY_NEW, "laya-playtest: a check, not a deliverable"),
                           (REPORT_OLD, REPORT_NEW, "laya-playtest: the stop rule"),
                           (WEB1_OLD, WEB1_NEW, "test-web: the page first, the server rule"),
                           (WEB8_OLD, WEB8_NEW, "test-web: the stop rule")):
        if old in out:
            out = out.replace(old, new, 1)
            changes.append(what)
    # 6 and 7 on a box's edited wording (the design-review edit: "UI flow with steps", a sentence
    # after the first one)
    for rx, rep, what in (
            (r'Playtest every game, (.*?) you build this way before "done"([.:])',
             r"Playtest a game, \1 you build this way when the kit runs here (a check, not a "
             r"deliverable; the testing package's budget applies)\2",
             "laya-playtest: a check, not a deliverable"),
            (r"(8\. A game or a UI flow[^\n]*?); every death it could not avoid and every stuck "
             r"spot is a bug\.",
             r"\1; a death or stuck spot that real keys cannot avoid either is a bug. Two runs "
             r"that end at the same spot: stop and report it as untested.",
             "test-web: the stop rule")):
        got = re.sub(rx, rep, out, count=1)
        if got != out:
            out = got
            changes.append(what)
    # 9, 10, 11: the kit's page check (SRV-195)
    for old, new, what in ((VERIFY_OLD, VERIFY_NEW, "testing: the kit's page check, no own driver, check-fix-check"),
                           (RUNTIME_OLD_PAGE, RUNTIME_NEW_PAGE, "testing: a page runs in the page check"),
                           (WEB1_V2, WEB1_V3, "test-web: the browser tool or the page check"),
                           (WEB3_OLD, WEB3_NEW, "test-web: real input through the page check"),
                           (WEB5_OLD, WEB5_NEW, "test-web: the page check's screenshots"),
                           (WEB8_V2, WEB8_V3, "test-web: Laya plays games and long flows"),
                           (PLAY_V2, PLAY_V3, "laya-playtest: a one-screen tool needs the page check"),
                           (LAYA_RUN_OLD, LAYA_RUN_NEW.replace(LAYA_QUOTE_OLD, LAYA_QUOTE_NEW),
                            "laya-testing: the page check carries Laya's verdict"),
                           (CHECK_ONE_OLD, CHECK_ONE_NEW, "testing: one page check with every control, no test copies"),
                           (PAGE_CMD_OLD, PAGE_CMD_NEW, "testing: the page check asks Laya at {tandem_url}"),
                           (WEB_CMD_OLD, WEB_CMD_NEW, "test-web: the page check asks Laya at {tandem_url}"),
                           (QUOTE_OLD, QUOTE_NEW, "testing: a page-check step in single quotes")):
        if old in out and new not in out and not (old == LAYA_RUN_OLD and LAYA_RUN_NEW in out):
            # 12's new text begins with its old one (and 18 rewords its end: either form counts)
            out = out.replace(old, new, 1)
            changes.append(what)
    # 15-23 (SRV-229): the final message speaks to the user; check-only overlays go
    for old, new, what in ((STATE_SAY_OLD, STATE_SAY_NEW, "testing: the state hook goes without a word"),
                           (FINISH_OLD, FINISH_NEW, "testing: the final message is for the user"),
                           (LAYA_QUOTE_OLD, LAYA_QUOTE_NEW, "laya-testing: verdicts read, not quoted"),
                           (LAYA_DONE_OLD, LAYA_DONE_NEW, "laya-testing: verdicts read before done"),
                           (PLAY_QUOTE_OLD, PLAY_QUOTE_NEW, "laya-playtest: the report is read"),
                           (REVIEW_OLD, REVIEW_NEW, "design-review: Laya's verdict read"),
                           *((o, n, "skills: fps read, no readout") for o, n in SKILL_FPS)):
        if old in out and new not in out:
            out = out.replace(old, new, 1)
            changes.append(what)
    if FINISH_NEW in out:
        out = _once(out, FINISH_NEW, OVERLAY_RULE, changes, "testing: check-only overlays go before done")
    if OVERLAY_RULE in out:
        out = _once(out, OVERLAY_RULE, TOOLS_QUIRK_RULE, changes, "testing: test-tool quirks stay out")
    if DONE_LIST_OLD in out and DONE_LIST_NEW not in out:
        out = out.replace(DONE_LIST_OLD, DONE_LIST_NEW, 1)
        changes.append("coding: the done list in the user's words")
    out = _once(out, HARNESS_ANCHOR, HARNESS_RULE, changes, "tandem-harness: the final message")
    # 19: a test package's "Report:" line is the model's own evidence, not the user's message
    got = re.sub(r"(?m)^Report: ", REPORT_LINE, out)
    if got != out:
        out = got
        changes.append("test-*: the report is the model's evidence")
    # 26: the evidence line is the model's, Laya's verdict last, and the final message names none
    def _evidence(m):
        items = m.group(1).rstrip()
        laya = re.search(r",? Laya's verdicts?", items)
        if laya is None:
            return EVIDENCE_LINE + items + EVIDENCE_TAIL
        word = laya.group(0).strip(", ")
        rest = (items[:laya.start()] + items[laya.end():]).rstrip(" .,")
        return f"{EVIDENCE_LINE}{rest}, and {word}.{EVIDENCE_TAIL}"
    got = re.sub(r"(?m)^" + re.escape(REPORT_LINE) + r"(.*)$", _evidence, out)
    if got != out:
        out = got
        changes.append("test-*: the evidence is the model's, Laya's verdict out of the message")
    # 27, 28, 29
    for old, new, what in ((SECOND_OLD, SECOND_NEW, "test-lib: a check for the model"),
                           (DONE_EV_OLD, DONE_EV_NEW, "coding: evidence in the user's words"),
                           (SAY_UNAVAIL_OLD, SAY_UNAVAIL_NEW, "laya: a failed call is not news (helper)"),
                           (WRITE_UNAVAIL_OLD, WRITE_UNAVAIL_NEW, "laya: a failed call is not news")):
        if old in out and new not in out:
            out = out.replace(old, new, 1)
            changes.append(what)
    # 30: the plan and the lines between tool calls name no Laya (the final message already did not)
    for needle in (EVIDENCE_TAIL, LAYA_DONE_NEW, HARNESS_RULE, TOOLS_QUIRK_RULE, DONE_EV_NEW):
        if needle in out:
            bullet = "- " if needle in (LAYA_DONE_NEW, HARNESS_RULE, TOOLS_QUIRK_RULE) else ""
            out = _after_line(out, needle, bullet + PROGRESS_LINE, changes,
                              "all text: plans and progress lines name no Laya")
            break
    # 31
    if SECOND_NEW in out:
        out = out.replace(SECOND_NEW, SECOND_V3, 1)
        changes.append("test-lib: a check no line names")
    # 32: the helper says what was wrong with a refused request, and sends no empty text
    if HELPER_TRY in out:
        out = out.replace(HELPER_TRY, HELPER_TRY_NEW, 1)
        out = out.replace(HELPER_IMPORT_OLD, HELPER_IMPORT_NEW, 1)
        changes.append("laya helper: a refused request says why (exit 2)")
    if ASK_DEF in out and EMPTY_GUARD not in out:
        out = out.replace(ASK_DEF, ASK_DEF + EMPTY_GUARD, 1)
        changes.append("laya helper: a refused request before it is sent (an empty text)")
    # 33
    out = _once(out, FAIL_BULLET.replace(WRITE_UNAVAIL_OLD, WRITE_UNAVAIL_NEW), REFUSED_BULLET, changes,
                "laya-testing: a refused request is the model's to fix")
    # 34: the scratch is marked by the run that makes it and deleted only then; a leftover is reused
    for old, new, what in ((SCRATCH_RULE, SCRATCH_RULE_V3, "testing: scratch marked, deleted only when made"),
                           (LEFTOVER_OLD, LEFTOVER_NEW, "testing: a leftover is reused, never deleted")):
        if old in out and new not in out:
            out = out.replace(old, new, 1)
            changes.append(what)
    out = _once(out, FILES_ANCHOR, NO_DELETE_RULE, changes,
                "files-care: never delete what this conversation did not create")
    return out, changes


def migrate_doc(doc: dict) -> tuple[dict, dict]:
    """(the packages document with every inline body migrated, {package: changes})."""
    new = json.loads(json.dumps(doc))
    report = {}
    for name, p in (new.get("packages") or {}).items():
        if p.get("source", "inline") != "inline" or not isinstance(p.get("body"), str):
            continue
        body, ch = migrate_text(name, p["body"])
        if ch:
            p["body"], report[name] = body, ch
    return new, report


def _files(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True))


def migrate_files(root: str, dry: bool) -> dict:
    report = {}
    for path in _files(root):
        with open(path) as f:
            text = f.read()
        out, ch = migrate_text(os.path.basename(os.path.dirname(path)), text)
        if ch:
            report[os.path.relpath(path, root)] = ch
            if not dry:
                with open(path, "w") as f:
                    f.write(out)
    return report


def _token(path: str) -> str:
    """The admin password from the secrets file; "" when none is set (the admin routes are open
    then, VIS-56)."""
    try:
        with open(os.path.expanduser(path)) as f:
            for line in f:
                if line.startswith("QSE_ADMIN_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _call(method: str, url: str, token: str, body=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode()
                                 if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:            # the server's reason, not a traceback
        raise SystemExit(f"[packages_migrate] {method} {url}: {exc.code} {exc.read()[:400]!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", action="store_true", help="rewrite this checkout's packages/")
    ap.add_argument("--skills", help="rewrite the SKILL.md (and .md) files under this directory")
    ap.add_argument("--doc", help="a packages version file or document: print what would change")
    ap.add_argument("--put", help="a server's address: migrate its current packages document")
    ap.add_argument("--token-file", default="~/.qwen38-spark-engine/secrets.env")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    if a.repo:
        rep = migrate_files(os.path.join(ROOT, "packages"), a.dry)
        print(json.dumps(rep, indent=1))
    if a.skills:
        rep = migrate_files(os.path.expanduser(a.skills), a.dry)
        print(json.dumps(rep, indent=1))
    if a.doc:
        with open(a.doc) as f:
            d = json.load(f)
        _, rep = migrate_doc(d.get("doc", d))
        print(json.dumps(rep, indent=1))
    if a.put:
        tok = _token(a.token_file)
        cur = _call("GET", a.put.rstrip("/") + "/v1/conductor/packages", tok)
        new, rep = migrate_doc(cur["doc"])
        print(json.dumps(rep, indent=1))
        left = {n: literal_api_hosts(p.get("body") or "") for n, p in new["packages"].items()
                if literal_api_hosts(p.get("body") or "")}
        if left:
            raise SystemExit(f"literal hosts left: {left}")
        if not rep:
            print("[packages_migrate] nothing to change")
            return 0
        if a.dry:
            print(f"[packages_migrate] dry run: v{cur.get('version')} would get a new version")
            return 0
        got = _call("PUT", a.put.rstrip("/") + "/v1/conductor/packages", tok,
                    {"doc": new, "note": NOTE_047 if any(
                        c.startswith(ROUND7) for ch in rep.values() for c in ch) else NOTE_045 if any(
                        c.startswith(ROUND6) for ch in rep.values() for c in ch) else NOTE_041 if any(
                        c.startswith(ROUND5) for ch in rep.values() for c in ch) else NOTE_351 if any(
                        c.startswith(ROUND4) for ch in rep.values() for c in ch) else NOTE_345 if any(
                        c.startswith(ROUND3) for ch in rep.values() for c in ch) else NOTE_343 if any(
                        c.startswith(ROUND2) for ch in rep.values() for c in ch) else NOTE,
                     "base_version": cur.get("version")})
        print(f"[packages_migrate] stored packages v{got.get('version')} (was v{cur.get('version')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
