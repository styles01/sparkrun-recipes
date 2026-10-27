"""A package whose result an offered tool delivers stays out of the turn (SRV-615).

tool-eval-bench TC-03 (2026-10-18): "let Sarah know the meeting moved to 3pm" with get_contacts
and send_email offered. The chat profile routed the turn to class chat, the package question
picked write-message at 0.995 (journal row: profile chat v7, router v27, all 34 everyday packages
as candidates), and the model wrote a message to copy with a [placeholder] for Sarah instead of
calling the tools. Under option 3 of the design a package with a `deliverable` (`send`,
`schedule`, `post`) on a request that offers client tools gets one more cascade question, the
tool or the text itself, and stays out on `tool`.

  1  the deliverable field and the capabilities read off the declarations (signals.tool_effects)
  2  TC-03: write-message stays out, everyday-core stays in, journaled
  3  the user asks for the text itself: write-message goes in
  4  no tools, or tool_choice none: no question, write-message goes in
  5  no answer: a send tool keeps it out (how: schema); no tool that sends (Open WebUI's search
     tools): no question, it goes in; a forced function keeps it out unasked
  6  a post tool (post_update for social-post) is read as a send; the answer decides
  7  the tool turns of the same user message keep the decision, no question again
  8  a pinned package wins
  9  shadow journals only; off asks nothing and builds the block of 0.12.2
 10  the counter, the stored-document migration

Run: python tests/test_deliverable_fit.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_conductor_router as T  # noqa: E402
from server import metrics  # noqa: E402
from server.conductor import migrations, schema, shipped, signals  # noqa: E402
from server.conductor import packages as packages_mod  # noqa: E402
from server.conductor import router as router_mod  # noqa: E402


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object",
                       "properties": {p: {"type": "string"} for p in props},
                       "required": list(required)}}}


# the shapes of tool-eval-bench's get_contacts and send_email
CONTACTS = _fn("get_contacts", "Look up contacts by name or group.", ["query"], ["query"])
SEND = _fn("send_email", "Send an email to a recipient.", ["to", "subject", "body"],
           ["to", "subject", "body"])
POST = _fn("post_update", "Publish a short update to the team feed.", ["text"], ["text"])
TC03 = "let Sarah know the meeting moved to 3pm"
DRAFT = "draft an email to Sarah that the meeting moved, I will send it myself"


def _decider(package="write-message", answer=("tool", 0.9), cls="chat"):
    def fn(text, options):
        if set(options) == {"tool", "text"}:
            return answer
        if package in options:
            return (package, 0.95)
        return (cls, 0.95) if cls in options else ("none", 0.9)
    return fn


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _skill(name: str) -> str:
    """A shipped everyday package's text, its frontmatter left out."""
    with open(os.path.join(ROOT, "packages", "everyday", name, "SKILL.md")) as f:
        return packages_mod.split_frontmatter(f.read())[1].strip()


# the shipped texts and fields of the packages under test (packages/seed.json)
PKGS = {
    "everyday-core": {"general": True},
    "write-message": {"deliverable": "send"},
    "social-post": {"deliverable": "post"},
    "explain-simply": {},
}


def _setup(mode="on", **kw):
    cond, fakes = T.setup(decider=_decider(**kw))
    _, p = cond.store.get("packages")
    for n, extra in PKGS.items():
        p["packages"][n] = {"title": n, "kind": "package", "source": "inline",
                            "description": f"The {n} package's description.", "body": _skill(n),
                            "max_tokens": 3000, "enabled": True, **extra}
    cond.store.put("packages", None, p, note="everyday", author="t")
    _, r = cond.store.get("router")
    for c in r["classes"]:
        if c["id"] == "chat":
            c["policy"]["packages"] = list(PKGS)
    if mode is not None:
        r["tool_fit"] = dict(r.get("tool_fit") or {}, deliverable=mode)
    else:
        r.pop("tool_fit", None)
    cond.store.put("router", None, r, note="chat packages", author="t")
    return cond, fakes


def _turn(cond, text, tools, sid, **extra):
    b = T.body([{"role": "user", "content": text}], tools=False, **extra)
    if tools is not None:
        b["tools"] = tools
    plan, ids, _ = T.turn(cond, b, headers={"x-session-id": sid}, kind="open-webui")
    return plan, T.text_of(ids)


def _parts(plan):
    return [p["name"] for p in (plan.block or {}).get("parts", [])]


def _asked(fakes) -> int:
    return sum(1 for _, o in fakes["decider"].calls if set(o) == {"tool", "text"})


WRITE_MESSAGE_LINE = "so it can be copied and sent as it is"


# ------------------------------------------------------------------ 1 the field and the reading

def test_1_the_deliverable_field_and_the_capabilities():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "packages", "seed.json")) as f:
        seed = json.load(f)["packages"]
    got = {n: p["deliverable"] for n, p in seed.items() if "deliverable" in p}
    assert got == {"write-message": "send", "formal-letter": "send", "event-plan": "schedule",
                   "weekly-schedule": "schedule", "checklist-builder": "schedule",
                   "social-post": "post"}, got
    lib = packages_mod.Library
    assert lib.deliverable({"packages": seed}, "write-message") == "send"
    assert lib.deliverable({"packages": seed}, "everyday-core") is None
    assert lib.deliverable({"packages": {"x": {"deliverable": "fax"}}}, "x") is None
    eff = signals.tool_effects({"tools": [CONTACTS, SEND, POST]})
    assert eff["send_email"] == {"send"} and eff["get_contacts"] == frozenset(), eff
    # MCP names are read by their schemas like any tool
    mcp = _fn("mail-server_send_email", "Send mail.", ["to", "body"])
    assert "send" in signals.tool_effects({"tools": [mcp]})["mail-server_send_email"]
    cal = _fn("create_calendar_event", "Create an event.", ["title", "date", "time"])
    assert "schedule" in signals.tool_effects({"tools": [cal]})["create_calendar_event"]
    # the schema refuses an unknown deliverable
    doc = json.loads(json.dumps(shipped_packages()))
    doc["packages"]["write-message"]["deliverable"] = "fax"
    assert schema.doc_errors("packages", doc)


def shipped_packages() -> dict:
    return next(d for k, n, d in shipped.documents() if k == "packages")


# ------------------------------------------------------------------ 2-3 TC-03 and the draft

def test_2_tc03_a_send_tool_delivers_the_message():
    cond, fakes = _setup()
    plan, text = _turn(cond, TC03, [CONTACTS, SEND], "dl-tc03")
    parts = _parts(plan)
    assert "write-message" not in parts and "everyday-core" in parts, parts
    assert WRITE_MESSAGE_LINE not in text and "Everyday answers" in text
    choice = plan.row["extra"]["package_choice"]
    assert choice["chosen"] == "write-message", choice
    (k,) = choice["kept_out"]
    assert k == {"name": "write-message", "why": "tool_delivers", "deliverable": "send",
                 "how": "decider", "tools": ["send_email"], "p": 0.9}, k
    assert _asked(fakes) == 1


def test_3_the_user_asks_for_the_text_itself():
    cond, fakes = _setup(answer=("text", 0.9))
    plan, text = _turn(cond, DRAFT, [CONTACTS, SEND], "dl-draft")
    assert "write-message" in _parts(plan), _parts(plan)
    assert WRITE_MESSAGE_LINE in text
    assert "kept_out" not in plan.row["extra"]["package_choice"]
    assert plan.row["extra"]["package_choice"]["kept_in"][0]["how"] == "decider"


# ------------------------------------------------------------------ 4 no tools, no question

def test_4_no_tools_or_tool_choice_none_asks_nothing():
    cond, fakes = _setup()
    plan, text = _turn(cond, TC03, None, "dl-none")
    assert "write-message" in _parts(plan) and _asked(fakes) == 0
    plan, text = _turn(cond, TC03, [CONTACTS, SEND], "dl-choice-none", tool_choice="none")
    assert "write-message" in _parts(plan) and _asked(fakes) == 0


# ------------------------------------------------------------------ 5 no answer, a forced call

def test_5_no_answer_and_a_send_tool_keep_it_out():
    cond, fakes = _setup(answer="decider_down")
    plan, _ = _turn(cond, TC03, [CONTACTS, SEND], "dl-noanswer")
    assert "write-message" not in _parts(plan)
    assert plan.row["extra"]["package_choice"]["kept_out"][0]["how"] == "schema"


def test_5a_no_tool_that_sends_asks_nothing():
    """Open WebUI's own tools, or get_contacts alone: no offered tool makes a send, so the
    package goes in as in 0.12.2 and no side question is paid for."""
    search = [_fn("search_web", "Search the web.", ["query"]),
              _fn("fetch_url", "Fetch a page.", ["url"])]
    for tools, sid in ((search, "dl-owui"), ([CONTACTS], "dl-contacts")):
        cond, fakes = _setup()
        plan, _ = _turn(cond, TC03, tools, sid)
        assert "write-message" in _parts(plan) and _asked(fakes) == 0, sid
        assert "kept_out" not in plan.row["extra"]["package_choice"]


def test_5b_a_forced_function_keeps_it_out_unasked():
    cond, fakes = _setup()
    forced = {"type": "function", "function": {"name": "send_email"}}
    plan, _ = _turn(cond, TC03, [CONTACTS, SEND], "dl-forced", tool_choice=forced)
    assert "write-message" not in _parts(plan) and _asked(fakes) == 0
    assert plan.row["extra"]["package_choice"]["kept_out"][0]["how"] == "tool_choice"


# ------------------------------------------------------------------ 6 a tool the schema misses

def test_6_a_post_tool_is_read_as_a_send():
    """post_update(text) has no recipient: the declaration's verb reads it as a send (C5's
    reading), which makes the `post` deliverable; the answer decides."""
    cond, fakes = _setup(package="social-post")
    plan, _ = _turn(cond, "post that we are closed on Friday", [POST], "dl-post")
    assert "social-post" not in _parts(plan), _parts(plan)
    k = plan.row["extra"]["package_choice"]["kept_out"][0]
    assert k["deliverable"] == "post" and k["tools"] == ["post_update"] and k["how"] == "decider"
    cond, fakes = _setup(package="social-post", answer=("text", 0.9))
    plan, _ = _turn(cond, "write a post that we are closed on Friday", [POST], "dl-post-text")
    assert "social-post" in _parts(plan) and _asked(fakes) == 1


# ------------------------------------------------------------------ 7 the tool turns

def test_7_the_decision_holds_for_the_tool_turns():
    cond, fakes = _setup()
    sid = "dl-turns"
    plan, _ = _turn(cond, TC03, [CONTACTS, SEND], sid)
    assert "write-message" not in _parts(plan) and _asked(fakes) == 1
    msgs = [{"role": "user", "content": TC03},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_contacts", "arguments": '{"query": "Sarah"}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": '{"results": [{"name": "Sarah", "email": "sarah@example.com"}]}'}]
    b = T.body(msgs, tools=False)
    b["tools"] = [CONTACTS, SEND]
    plan2, ids, _ = T.turn(cond, b, headers={"x-session-id": sid}, kind="open-webui")
    assert WRITE_MESSAGE_LINE not in T.text_of(ids), "the tool turn brings it back"
    assert "Everyday answers" in T.text_of(ids), "everyday-core is replayed"
    assert _asked(fakes) == 1, "no second question on the tool turn"


# ------------------------------------------------------------------ 8 pins

def test_8_a_pinned_package_wins():
    cond, fakes = _setup()
    plan, text = _turn(cond, TC03, [CONTACTS, SEND], "dl-pin",
                       tandem={"packages": {"pin": ["write-message"]}})
    assert "write-message" in _parts(plan), (_parts(plan), plan.row["extra"])
    assert plan.row["extra"]["package_pinned"] == ["write-message"]
    assert {"name": "write-message", "why": "pinned"} in \
        plan.row["extra"]["package_choice"]["kept_out_skipped"]


# ------------------------------------------------------------------ 9 shadow and off

def test_9_shadow_journals_only():
    cond, fakes = _setup("shadow")
    plan, text = _turn(cond, TC03, [CONTACTS, SEND], "dl-shadow")
    assert "write-message" in _parts(plan)
    (k,) = plan.row["shadow"]["tools"]["kept_out"]
    assert k["name"] == "write-message" and k["how"] == "decider"


def test_9a_the_shipped_default_is_shadow():
    assert schema.TOOL_FIT["deliverable"] == "shadow"
    cond, fakes = _setup(None)
    plan, _ = _turn(cond, TC03, [CONTACTS, SEND], "dl-default")
    assert "write-message" in _parts(plan)
    assert plan.row["shadow"]["tools"]["kept_out"][0]["name"] == "write-message"


def test_9b_off_builds_the_block_of_0122():
    cond, fakes = _setup("off")
    plan_off, text_off = _turn(cond, TC03, [CONTACTS, SEND], "dl-off")
    assert "write-message" in _parts(plan_off) and _asked(fakes) == 0
    was = router_mod.TOOL_FIT
    try:
        router_mod.TOOL_FIT = "off"
        cond2, fakes2 = _setup()
        plan_sw, text_sw = _turn(cond2, TC03, [CONTACTS, SEND], "dl-off")
    finally:
        router_mod.TOOL_FIT = was
    assert _parts(plan_sw) == _parts(plan_off) and _asked(fakes2) == 0
    assert plan_sw.block["text"] == plan_off.block["text"]


# ------------------------------------------------------------------ 10 counter and migration

def test_10_the_counter_and_the_migration():
    cond, _ = _setup()
    plan, _ = _turn(cond, TC03, [CONTACTS, SEND], "dl-count")
    metrics.on_route(plan.row)
    text = metrics.render()
    assert any(ln.startswith("qse_conductor_package_kept_out_total{") and
               'why="tool_delivers"' in ln and 'mode="act"' in ln for ln in text.splitlines()), \
        [ln for ln in text.splitlines() if "kept_out" in ln]
    # a packages document stored before SRV-615 gets the field once; a removed one stays removed
    from server.conductor.store import Store
    st = Store(tempfile.mkdtemp(prefix="dl-mig-"))
    d = json.loads(json.dumps(schema.SHIPPED_PACKAGES))
    for p in d["packages"].values():
        p.pop("deliverable", None)
    st.put("packages", None, d, note="0.12.2's shipped packages", author="shipped")
    mine = [m for m in migrations.MIGRATIONS if m[0] == "2026-10-20-packages-deliverable"]
    with open(os.devnull, "w") as dn:
        so, sys.stdout = sys.stdout, dn
        try:
            done = migrations.run(st, migrations=mine)
            again = migrations.run(st, migrations=mine)
        finally:
            sys.stdout = so
    assert [r["result"] for r in done] == ["changed"] and again == []
    _, d = st.get("packages")
    assert d["packages"]["write-message"]["deliverable"] == "send"
    assert d["packages"]["social-post"]["deliverable"] == "post"
    # someone removed it after it was there: the change function leaves it out
    _, cur = st.get("packages")
    past = [cur]
    cur = json.loads(json.dumps(cur))
    del cur["packages"]["write-message"]["deliverable"]
    out = migrations._deliverable(cur, past)
    assert out is None or "write-message" not in " ".join(out[1])


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:70s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
