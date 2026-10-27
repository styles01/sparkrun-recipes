"""Which client sent a request, and so which profile it gets with no header.

The recorded shapes below are the clients' own: opencode 1.18.34's User-Agent as its requests carry
it, its tool names as the box's disk tier holds a session's head (2026-10-03); the Hermes agent
0.21.0 (NousResearch hermes-agent at 66e258b on the Pi): `OpenAI/Python 2.24.0` (its venv's SDK,
no header of its own for a custom endpoint), the tools of its `hermes-cli` toolset as
`model_tools.get_tool_definitions` lists them, its title call's instruction and its built-in
identity. Only names and the fixed texts: no message of anybody's.

  1  opencode by its agent string, whatever tools it offers
  2  Hermes by its tools (two of its own skill and session tools), by its title call, by its
     identity; a SOUL.md in place of the identity changes nothing while the tools are there
  3  an OpenAI SDK script with ordinary tools stays `openai-sdk`, curl `curl`, Open WebUI
     `open-webui` even with Hermes-like tool names (a client that names itself keeps its kind)
  4  the profiles: opencode -> coding, hermes -> hermes, every other kind -> settings'
     default_profile, `general`; a profile header still wins

Run: python tests/test_client_kinds.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import ledger  # noqa: E402
from server.conductor import router as router_mod, schema  # noqa: E402
from server.conductor.store import Store  # noqa: E402

OPENCODE_UA = "opencode/1.18.34 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
HERMES_UA = "OpenAI/Python 2.24.0"
OPENCODE_TOOLS = ["bash", "edit", "glob", "grep", "list", "read", "skill", "task", "todowrite",
                  "webfetch", "write", "chrome-devtools_click", "chrome-devtools_take_screenshot"]
HERMES_TOOLS = ["browser_exec", "clarify", "delegate_task", "execute_code", "memory", "patch",
                "process", "read_file", "search_files", "session_search", "skill_manage",
                "skill_view", "skills_list", "terminal", "text_to_speech", "todo", "web_extract",
                "web_search", "write_file"]
HERMES_TITLE = ("You name chat sessions. Given the user's opening message, write a title that lets "
                "them find this conversation again in a list.")
HERMES_IDENTITY = ("You are Hermes Agent, built by Nous Research. Be direct: match the length of "
                   "your reply to the weight of the ask.")


def _tools(names):
    return [{"type": "function", "function": {"name": n, "description": "",
                                              "parameters": {"type": "object"}}} for n in names]


def _body(tools=None, system=None, user="hello"):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": user}]
    return {"messages": msgs, **({"tools": _tools(tools)} if tools else {})}


def kind(ua, body):
    return ledger.client_of({"User-Agent": ua}, body)[1]


def test_opencode_by_its_agent():
    assert kind(OPENCODE_UA, _body(OPENCODE_TOOLS)) == "opencode"
    assert kind(OPENCODE_UA, _body(HERMES_TOOLS)) == "opencode"
    assert kind(OPENCODE_UA, _body()) == "opencode"


def test_hermes_by_its_tools_its_title_call_and_its_identity():
    assert kind(HERMES_UA, _body(HERMES_TOOLS, system="A SOUL.md of the user's own.")) == "hermes"
    assert kind(HERMES_UA, _body(system=HERMES_TITLE, user="plan my week")) == "hermes"
    assert kind(HERMES_UA, _body(system=HERMES_IDENTITY)) == "hermes"
    parts = {"messages": [{"role": "system", "content": [{"type": "text",
                                                          "text": HERMES_IDENTITY}]},
                          {"role": "user", "content": "hello"}]}
    assert kind(HERMES_UA, parts) == "hermes"                  # the content as a list of parts
    # an unnamed agent with Hermes' shape too
    assert kind("python-httpx/0.28.1", _body(HERMES_TOOLS)) == "hermes"
    # one of its tools alone is not its shape
    assert kind(HERMES_UA, _body(["terminal", "skill_view", "read_file"])) == "openai-sdk"


def test_everyone_else_keeps_their_kind():
    assert kind(HERMES_UA, _body(["bash", "read_file", "write_file"])) == "openai-sdk"
    assert kind(HERMES_UA, _body()) == "openai-sdk"
    assert kind("curl/8.5.0", _body()) == "curl"
    assert kind("Python/3.11 aiohttp/3.9.5", _body(HERMES_TOOLS)) == "open-webui"
    assert kind("", _body()) == "other"
    assert ledger.client_of({"User-Agent": HERMES_UA})[1] == "openai-sdk"      # no body
    assert "hermes" in ledger.KINDS and "hermes" in schema.CLIENT_KINDS


def test_the_profile_each_kind_gets():
    import tempfile
    store = Store(tempfile.mkdtemp(prefix="qse-kinds-"))
    from server.conductor import shipped
    shipped.install(store)               # a new box's documents (ops/start.sh: shipped)
    want = {"opencode": "coding", "hermes": "hermes", "openai-sdk": "general", "curl": "general",
            "other": "general", "open-webui": "chat", "dashboard": "chat", "chat-app": "chat-app"}
    for k, prof in want.items():
        res = router_mod.resolve(store, "TandemLLM", "qwen38-spark-engine", {}, k)
        assert (res.profile, res.source) == (prof, "client" if prof != "general" else "default"), \
            (k, res.profile, res.source)
    res = router_mod.resolve(store, "TandemLLM", "qwen38-spark-engine",
                             {"X-Tandem-Profile": "coding"}, "hermes")
    assert res.profile == "coding" and res.source == "header"
    # Hermes renders the lines in the reasoning, opencode as its reasoning parts' titles
    assert store.get("profiles", "hermes")[1]["ui"]["states_channel"] == "reasoning"
    assert store.get("profiles", "coding")[1]["ui"]["states_channel"] == "inline"
    assert store.get("profiles", "general")[1]["ui"] == {"states": "on",
                                                         "states_channel": "reasoning"}


def test_chatd_is_the_chat_app_and_gets_codings_policy_without_state_lines():
    from chatd import engine, render
    h = dict(render.headers("C1"), **{"User-Agent": engine.USER_AGENT})
    assert ledger.client_of(h)[1] == "chat-app"
    # an engine older than the kind reads chatd as the dashboard: its chat profile, as before
    assert "qse-dashboard" in engine.USER_AGENT.lower()
    assert not engine.USER_AGENT.lower().startswith(ledger.INTERNAL_UA)
    app, coding = schema.SHIPPED_PROFILES["chat-app"], schema.SHIPPED_PROFILES["coding"]
    assert app["clients"] == ["chat-app"] and app["ui"]["states"] == "off"
    own = schema.DERIVED_PROFILES["chat-app"][1]          # SRV-471: memory is chat-app's own
    same = {k: v for k, v in app.items() if k not in own}
    assert same == {k: v for k, v in coding.items() if k not in own} and app["memory"] == "on"
    assert not schema.check("profiles", app, name="chat-app", router=schema.SHIPPED_ROUTER,
                            packages=schema.SHIPPED_PACKAGES, profiles=schema.SHIPPED_PROFILES)
    assert "chat-app" in ledger.KINDS and "chat-app" in schema.CLIENT_KINDS


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:70s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
