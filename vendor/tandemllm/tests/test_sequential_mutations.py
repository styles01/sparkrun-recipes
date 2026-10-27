"""SRV-623 (QA SRV-624): C5, one mutation per turn in the toolgate.

An answer that carries two or more calls to tools that change something outside the conversation
(send, schedule, create, write, delete; read off each tool's declaration by
`signals.tool_effects`, never the user's words) is written again one at a time through C4's
regeneration turn: the end token is dropped, a hidden tool-shaped turn says to make the first
call, read its result, then the next, and the client receives only the new calls. Reads stay
parallel. Through the real handler with the fake engine in this process, streamed and not, as
tests/test_toolcheck.py:

  1  the capability reading: tool-eval-bench's tools, opencode's, annotations, an MCP prefix,
     a read verb over a mutating shape, tool_choice none        test_1_*
  2  two mutations -> one regeneration, the client sees the first call, one journal row
                                                                test_2_*
  3  two reads, a read beside a mutation -> nothing held back or rewritten
                                                                test_3_*
  4  cap 1: a second violation goes out as it is, journaled; past the request's caps the calls
     go out too, never a give-up                                test_4_*
  5  shadow: a journal row, the answer unchanged; a constrained request (SRV-619's union, tool_choice
     required) is shadow                                        test_5_*
  6  profiles: tools acts, coding, chat-app, hermes and chat journal, general and overnight off;
     a profile stored before C5 gets it once (migration), a stored monitor document by
     tools/monitor_migrate.py                                   test_6_*

Run: python tests/test_sequential_mutations.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import yaml  # noqa: E402

from server.conductor import signals  # noqa: E402
from server.conductor.monitor import Policy, RequestEvent  # noqa: E402
from server.conductor.monitor import policy as policy_mod  # noqa: E402
from test_toolcheck import ask, call_json, ctx_text, policy, rows, serve  # noqa: E402


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": {k: {"type": t} for k, t in props.items()},
                       "required": list(required), "additionalProperties": False}}}


# tool-eval-bench's universal tools (domain/tools.py), the shapes that matter here
SEND_EMAIL = _fn("send_email", "Send an email to a recipient",
                 {"to": "string", "subject": "string", "body": "string", "cc": "string",
                  "bcc": "string", "attachments": "array"}, ("to", "subject", "body"))
EVENT = _fn("create_calendar_event", "Create a new calendar event",
            {"title": "string", "date": "string", "time": "string", "timezone": "string",
             "duration_minutes": "integer", "attendees": "array"}, ("title", "date", "time"))
REMINDER = _fn("set_reminder", "Set a reminder for a future time",
               {"message": "string", "datetime": "string"}, ("message", "datetime"))
CONTACTS = _fn("get_contacts", "Look up contacts by name or group", {"query": "string"})
WEATHER = _fn("get_weather", "Get current weather for a specific location",
              {"location": "string", "units": "string"}, ("location",))
SEARCH_FILES = _fn("search_files", "Search for files by name or content",
                   {"query": "string", "file_type": "string"})
READ_FILE = _fn("read_file", "Read the contents of a specific file", {"file_id": "string"})
WEB = _fn("web_search", "Search the web for current information",
          {"query": "string", "max_results": "integer"})
CALC = _fn("calculator", "Perform mathematical calculations", {"expression": "string"})
RUN = _fn("run_code", "Execute a code snippet and return the output",
          {"language": "string", "code": "string"})
TRANSLATE = _fn("translate_text", "Translate text from one language to another",
                {"text": "string", "source_language": "string", "target_language": "string"})
STOCK = _fn("get_stock_price", "Get the current stock price for a ticker symbol",
            {"ticker": "string"})
BENCH = [WEB, WEATHER, CALC, SEND_EMAIL, SEARCH_FILES, READ_FILE, EVENT, CONTACTS, TRANSLATE,
         STOCK, REMINDER, RUN]
READS = [WEATHER, CONTACTS, STOCK]

EV_ARGS = {"title": "Sync", "date": "2026-03-20", "time": "15:00",
           "attendees": ["alice@company.com", "bob@company.com"]}
MAIL_ARGS = {"to": "alice@company.com", "subject": "Sync", "body": "Invite sent."}
TWO = (call_json("create_calendar_event", **EV_ARGS) + "\n" + call_json("send_email", **MAIL_ARGS))


def _names(got):
    return [c["name"] for _, c in sorted(got["calls"].items())]


def _pol(**over) -> Policy:
    d = yaml.safe_load(open(policy_mod.DEFAULT_PATH))
    d["tasks"]["C5"]["mode"] = over.pop("mode", "act")
    for k, v in over.items():
        d["caps"][k] = v
    return Policy(d)


# ------------------------------------------------------------------ 1: what a tool changes

def test_1_the_bench_tools_read_off_their_declarations():
    eff = signals.tool_effects({"tools": BENCH})
    muts = {n: sorted(k) for n, k in eff.items() if k}
    assert muts == {"send_email": ["send"], "create_calendar_event": ["create", "schedule"],
                    "set_reminder": ["create", "schedule"]}, muts
    assert set(eff) == {t["function"]["name"] for t in BENCH}
    assert signals.tool_effects({"tools": BENCH, "tool_choice": "none"}) == {}


def test_1_opencode_and_open_webui_tools():
    """A shell is no mutation (opencode reads through bash); write and edit are writes; read,
    glob, grep, webfetch, the terminal's run_command and Open WebUI's search_web and fetch_url
    are none; todowrite (the session's own list) is none."""
    tools = [_fn("bash", "Executes a given bash command", {"command": "string",
                                                           "workdir": "string"}),
             _fn("read", "Reads a file from the local filesystem.", {"filePath": "string",
                                                                    "offset": "integer"}),
             _fn("write", "Writes a file to the local filesystem.", {"filePath": "string",
                                                                    "content": "string"}),
             _fn("edit", "Performs exact string replacements in files.",
                 {"filePath": "string", "oldString": "string", "newString": "string"}),
             _fn("glob", "- Fast file pattern matching tool", {"pattern": "string"}),
             _fn("grep", "- Fast content search tool", {"pattern": "string", "path": "string"}),
             _fn("webfetch", "- Fetches content from a specified URL", {"url": "string"}),
             _fn("todowrite", "Use this tool to create and manage a structured task list",
                 {"todos": "array"}),
             _fn("run_command", "Run a shell command", {"command": "string"}),
             _fn("search_web", "Search the web", {"query": "string"}),
             _fn("fetch_url", "Fetch a page", {"url": "string"}),
             _fn("write_file", "Write a file", {"path": "string", "content": "string"})]
    muts = {n: sorted(k) for n, k in signals.tool_effects({"tools": tools}).items() if k}
    assert muts == {"write": ["write"], "edit": ["write"], "write_file": ["write"]}, muts


def test_1_hints_prefixes_and_read_verbs():
    tools = [_fn("delete_file", "Delete a file", {"file_id": "string"}),
             _fn("github-server_create_issue", "", {"repo": "string", "title": "string"}),
             _fn("postMessage", "", {"channel": "string", "text": "string"}),
             _fn("search_emails", "Search the mailbox", {"to": "string", "subject": "string",
                                                         "body": "string"}),
             _fn("list_events", "", {"date": "string", "title": "string"}),
             _fn("archive", "Removes a thread from the inbox", {"thread_id": "string"}),
             _fn("thing", "Sends a fax", {"number": "string"}),
             dict(_fn("purge_cache", "", {"key": "string"}),
                  annotations={"readOnlyHint": True}),
             dict(_fn("lookup", "", {"id": "string"}), annotations={"destructiveHint": True})]
    eff = {n: sorted(k) for n, k in signals.tool_effects({"tools": tools}).items()}
    assert eff == {"delete_file": ["delete"], "github-server_create_issue": ["create"],
                   "postMessage": ["send"], "search_emails": [], "list_events": [],
                   "archive": ["delete"], "thing": ["send"], "purge_cache": [],
                   "lookup": ["delete"]}, eff


# ------------------------------------------------------------------ 2: two mutations

def test_2_two_mutations_are_written_again_one_at_a_time():
    """TC-51's shape: the event and its invitation in one answer. Both are dropped, the model
    reads the reminder in a hidden tool-shaped turn and writes the event alone; the client
    receives only that call, at index 0. One act row, with the calls and their kinds."""
    for stream in (True, False):
        mon = serve("on", _pol())
        got = ask(TWO + "\nFAKE_AFTER_FIRST", stream=stream, tools=BENCH, system="")
        assert _names(got) == ["create_calendar_event"], got
        assert json.loads(got["calls"][0]["args"])["title"] == "Sync"
        assert got["finish"] == "tool_calls", got
        seen = ctx_text()
        assert ("<tool_response>\nThese 2 calls were not run: `create_calendar_event`, "
                "`send_email`.") in seen, seen[-800:]
        assert "make the first call alone, read its result, then make the next" in seen
        act = rows(mon, task="C5", decision="REGENERATE_CALL", mode="act")
        assert len(act) == 1 and act[0]["hidden_tokens"] > 0, mon.journal.events()
        d = act[0]["detail"]
        assert act[0]["rule"] == "sequential_mutations"
        assert d["findings"][0]["detail"]["calls"] == ["create_calendar_event", "send_email"]
        assert d["findings"][0]["detail"]["kinds"] == {
            "create_calendar_event": ["create", "schedule"], "send_email": ["send"]}
        assert not rows(mon, decision="GIVE_UP")


def test_2_a_read_and_two_mutations_regenerate_once():
    """A read does not count, two mutations beside it do: one regeneration."""
    mon = serve("on", _pol())
    user = (call_json("get_contacts", query="engineering") + "\n" + TWO + "\nFAKE_AFTER_FIRST")
    got = ask(user, tools=BENCH, system="")
    assert _names(got) == ["get_contacts"], got
    assert len(rows(mon, task="C5", decision="REGENERATE_CALL")) == 1


# ------------------------------------------------------------------ 3: reads stay parallel

def test_3_two_reads_pass_untouched():
    """Two get_weather calls (TC-27's shape) go out as written; C5 writes no row. With no
    mutating tool offered and C5 the only tool-call task, nothing is held: the arguments stream
    as the model writes them."""
    two = (call_json("get_weather", location="London", units="celsius") + "\n" +
           call_json("get_weather", location="London", units="fahrenheit"))
    for tools in (BENCH, READS):
        for stream in (True, False):
            mon = serve("on", _pol())
            got = ask(two, stream=stream, tools=tools, system="")
            assert _names(got) == ["get_weather", "get_weather"], got
            assert not rows(mon, task="C5"), mon.journal.events()
    mon = serve("on", policy(only=("C5",), C5="act"))
    sup = mon.begin("on", RequestEvent("r", tools=READS), [1, 2], None)
    assert not sup.holds_calls, "no tool can mutate: C5 holds nothing"
    sup = mon.begin("on", RequestEvent("r2", tools=BENCH), [1, 2], None)
    assert sup.holds_calls, "a mutating tool is offered: the answer's calls are held"
    got = ask(two, tools=READS, system="")
    assert got["deltas"] > 2, "streamed live, not held"


def test_3_a_read_beside_one_mutation_passes():
    """TC-53/56-like steps: a lookup with one action, or one action alone, go out as written."""
    for user, want in (
            (call_json("get_contacts", query="eng") + "\n" +
             call_json("send_email", **MAIL_ARGS), ["get_contacts", "send_email"]),
            (call_json("create_calendar_event", **EV_ARGS), ["create_calendar_event"])):
        for stream in (True, False):
            mon = serve("on", _pol())
            got = ask(user, stream=stream, tools=BENCH, system="")
            assert _names(got) == want, got
            assert not rows(mon, task="C5"), mon.journal.events()


# ------------------------------------------------------------------ 4: caps

def test_4_cap_one_releases_a_second_violation_as_it_is():
    """The model writes both again after the reminder: the second answer goes out whole, its two
    calls at indexes 0 and 1, and the journal says why (fallback after one regeneration)."""
    for stream in (True, False):
        mon = serve("on", _pol())
        got = ask(TWO + "\nFAKE_AFTER_SAME", stream=stream, tools=BENCH, system="")
        assert _names(got) == ["create_calendar_event", "send_email"], got
        assert len(rows(mon, task="C5", decision="REGENERATE_CALL")) == 1
        fb = rows(mon, task="C5", decision="PASS")
        assert fb and fb[0]["reason"] == "fallback(regenerations=1)", fb
        assert fb[0]["detail"]["calls"] == ["create_calendar_event", "send_email"]
        assert not rows(mon, decision="GIVE_UP")


def test_4_past_the_request_caps_the_calls_go_out_never_a_give_up():
    """C4 gives up past the caps (no call can run); C5's calls can, so they are released."""
    mon = serve("on", _pol(acts_per_request=0))
    got = ask(TWO + "\nFAKE_AFTER_FIRST", tools=BENCH, system="")
    assert _names(got) == ["create_calendar_event", "send_email"], got
    assert "Stopping this turn" not in got["content"]
    rel = rows(mon, task="C5", decision="PASS")
    assert rel and rel[0]["reason"] == "released(per_request)" and rel[0]["cap"] == "per_request"
    assert not rows(mon, decision="GIVE_UP") and not rows(mon, decision="REGENERATE_CALL")


def test_4_an_answer_without_its_end_token_is_released():
    mon = serve("on", _pol())
    sup = mon.begin("on", RequestEvent("r", tools=BENCH), [1, 2], None)
    items = [{"function": {"name": "create_calendar_event", "arguments": json.dumps(EV_ARGS)}},
             {"function": {"name": "send_email", "arguments": json.dumps(MAIL_ARGS)}}]
    assert sup.check_answer(items, final=True) is None
    got = rows(mon, task="C5")
    assert [(r["decision"], r["reason"]) for r in got] == [("PASS", "ended(mutations=2)")], got


# ------------------------------------------------------------------ 5: shadow

def test_5_shadow_journals_and_changes_nothing():
    for stream in (True, False):
        mon = serve("on", _pol(mode="shadow"))
        got = ask(TWO + "\nFAKE_AFTER_FIRST", stream=stream, tools=BENCH, system="")
        assert _names(got) == ["create_calendar_event", "send_email"], got
        sh = rows(mon, task="C5")
        assert len(sh) == 1 and sh[0]["mode"] == "shadow", sh
        assert sh[0]["decision"] == "REGENERATE_CALL" and sh[0]["hidden_tokens"] == 0
        assert sh[0]["detail"]["calls"] == ["create_calendar_event", "send_email"]
        assert "were not run" not in ctx_text()
    # the server's --conductor shadow lowers an acting C5 the same way
    mon = serve("shadow", _pol())
    got = ask(TWO + "\nFAKE_AFTER_FIRST", tools=BENCH, system="")
    assert _names(got) == ["create_calendar_event", "send_email"]
    assert [r["mode"] for r in rows(mon, task="C5")] == ["shadow"]


def test_5_a_constrained_request_only_journals():
    """With a grammar on the answer (SRV-619's calls-or-document union for response_format with
    tools, tool_choice required or named) the server marks the request constrained: SP3 holds
    nothing and every act drops to shadow, C5's too."""
    mon = serve("on", _pol())
    sup = mon.begin("on", RequestEvent("r", tools=BENCH, constrained=True), [1, 2], None)
    assert not sup.holds_calls
    items = [{"function": {"name": "create_calendar_event", "arguments": json.dumps(EV_ARGS)}},
             {"function": {"name": "send_email", "arguments": json.dumps(MAIL_ARGS)}}]
    assert sup.check_answer(items) is None
    got = rows(mon, task="C5")
    assert [(r["decision"], r["mode"]) for r in got] == [("REGENERATE_CALL", "shadow")], got


# ------------------------------------------------------------------ 6: profiles

def _mode_under(profile: str, doc: dict | None = None) -> str | None:
    from server.conductor import schema
    from server.conductor.hook import Conductor
    from server.conductor.router import Plan
    with tempfile.TemporaryDirectory() as d:
        cond = Conductor(d, journal=False)
        try:
            doc = doc or json.loads(json.dumps(schema.SHIPPED_PROFILES[profile]))
            plan = Plan(type("Res", (), {"profile": profile, "version": 0, "doc": doc,
                                         "mode": "on"})())
            plan.active = True
            pol = cond.monitor_policy(plan, policy_mod.load())
            t = pol.tasks.get("C5")
            return t.mode if t is not None else None
        finally:
            cond.close()


def test_6_tools_acts_the_agent_profiles_journal():
    from server.conductor import schema
    want = {"tools": "act", "coding": "shadow", "chat-app": "shadow", "hermes": "shadow",
            "chat": "shadow", "general": "off", "overnight": "off"}
    got = {n: _mode_under(n) for n in want}
    assert got == want, got
    for n, mode in want.items():
        p = schema.SHIPPED_PROFILES[n]
        assert p["tasks"]["C5"]["mode"] == mode, n
        assert p["toolgate"]["sequence"] is (mode != "off"), n
        assert not schema.check("profiles", p, name=n, router=schema.SHIPPED_ROUTER,
                                packages=schema.SHIPPED_PACKAGES,
                                profiles=schema.SHIPPED_PROFILES), n
    assert schema.SHIPPED_PROFILES["tools"]["tasks"]["C5"]["thresholds"] == {"regenerations": 1}
    # the pack off turns the task off whatever its mode says
    doc = json.loads(json.dumps(schema.SHIPPED_PROFILES["tools"]))
    doc["toolgate"]["sequence"] = False
    assert _mode_under("tools", doc) == "off"
    # the policy file: C5 after C3, in shadow when no profile names it, one regeneration
    p = policy_mod.load()
    assert p.version == 8 and p.tasks["C5"].rule == "sequential_mutations"
    assert p.tasks["C5"].mode == "shadow" and p.tasks["C5"]["regenerations"] == 1
    assert p.tasks["C5"]["kinds"] == list(signals.MUTATIONS)
    ids = list(p.tasks)
    assert ids.index("C5") == ids.index("C3") + 1
    with open(policy_mod.DEFAULT_JSON) as f:
        assert json.load(f) == p.data, "policy.default.json is stale"


def test_6_a_profile_stored_before_c5_loads_and_is_migrated_once():
    from server.conductor import migrations, schema, shipped
    from server.conductor.store import Store
    old = json.loads(json.dumps(schema.SHIPPED_PROFILES["coding"]))
    del old["tasks"]["C5"]
    del old["toolgate"]["sequence"]
    assert not schema.doc_errors("profiles", old), "a profile from before C5 still loads"
    assert _mode_under("coding", old) == "off", "absent: the pack is off"
    st = Store(tempfile.mkdtemp(prefix="qse-c5-"))
    shipped.install(st)
    for name in ("coding", "chat-app", "hermes", "chat", "tools"):
        _, d = st.get("profiles", name)
        d = json.loads(json.dumps(d))
        del d["tasks"]["C5"]
        del d["toolgate"]["sequence"]
        st.put("profiles", name, d, note="before SRV-623", author="t")
    _, kept = st.get("profiles", "general")
    mine = [m for m in migrations.MIGRATIONS if m[0].startswith("2026-10-19-")]
    assert [m[0] for m in mine] == ["2026-10-19-coding-c5", "2026-10-19-chat-app-c5",
                                    "2026-10-19-hermes-c5", "2026-10-19-chat-c5",
                                    "2026-10-19-tools-c5"]
    with open(os.devnull, "w") as dn:
        so, sys.stdout = sys.stdout, dn
        try:
            done = migrations.run(st, migrations=mine)
            again = migrations.run(st, migrations=mine)
        finally:
            sys.stdout = so
    assert [r["result"] for r in done] == ["changed"] * 5 and again == []
    for name, mode in (("coding", "shadow"), ("chat-app", "shadow"), ("hermes", "shadow"),
                       ("chat", "shadow"), ("tools", "act")):
        _, d = st.get("profiles", name)
        assert d["tasks"]["C5"]["mode"] == mode and d["toolgate"]["sequence"] is True, name
    # a profile that has the key keeps its value: someone chose
    _, d = st.get("profiles", "tools")
    d["toolgate"]["sequence"] = False
    d["tasks"]["C5"]["mode"] = "shadow"
    assert migrations._c5("act")(d) is None
    assert st.get("profiles", "general")[1] == kept


def test_6_a_stored_monitor_document_gets_c5_by_the_migrate_tool():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
    import monitor_migrate
    p = policy_mod.load()
    old = json.loads(json.dumps(p.data))
    del old["tasks"]["C5"]
    old["version"] = 7
    new, ch = monitor_migrate.migrate_doc(old, ("C5",))
    assert ch == ["task C5 added (sequential_mutations, shadow)", "version 7 -> 8"], ch
    assert new["tasks"]["C5"] == p.data["tasks"]["C5"]
    assert monitor_migrate.migrate_doc(new, ("C5",))[1] == []


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
