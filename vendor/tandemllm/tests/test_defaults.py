"""A new box's defaults are what the live box was configured to on 2026-10-03.

The shipped documents are schema.py's SHIPPED_* (the built-in version 0 stays the
minimal one the tests build on); a server with `--conductor-defaults shipped` (ops/start.sh) stores
each one its directory has no version of as version 1 at its start (server/conductor/shipped.py).

The owner, 2026-10-03: "people will not switch their profiles and set up opencode with default
settings". So the conductor's built-in documents (version 0, server/conductor/schema.py `seed`) are
the live ones, and a box that has never stored a version serves them with a plain opencode or
Hermes setup and no header.

The fixtures (tests/fixtures/defaults/) are the live documents of that day as the box stored them
(router v13, packages v5, settings v4, profiles coding v14, overnight v8, chat v2), with the
operator's own address in the package texts written as `{tandem_url}` (the shipped texts name the
provider that way; the router puts in the address a request came in on).

The real server, the fake engine (server/fake_engine.py), a scratch HOME and a conductor directory
of its own that no version was ever stored in:

  1  every document the server serves equals the fixture in every key the fixture has; the keys
     it adds are the defaults the store fills for a document stored before them (a stored router
     without `side_requests`, a profile without `by_size`), and settings' default profile is the
     new `general`
  2  the packages: the shipped files are found without an install step (`{repo}/packages`), the
     library answers a search and a read, a read names the request's own address
  3  a client without a header lands where it should: opencode on `coding` (the state lines
     inline), the Hermes agent's request shape on `hermes` (in the reasoning), any other client
     that offers tools on `tools` (SRV-621), anything else on `general`
  4  the disk tier's settings: on, 50 GB, never under 50 GB free; serve.env and start.sh turn the
     end anchor, its whitespace alignment, the conductor and the disk tier on

Run: python tests/test_defaults.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_conductor_e2e import ADMIN, LAUNCH, _call, _port, _wait  # noqa: E402

import subprocess  # noqa: E402

from server.conductor import schema  # noqa: E402

FIX = os.path.join(ROOT, "tests", "fixtures", "defaults")
# what the store fills in a stored document that lacks it, keyed by the document
FILLED = {"router": {"structural.side_requests", "classifier.probe_stage"},
          # Opencode's word is ignored (the deploy stores it on the live box too)
          # The shadow reader's block (the deploy stores it on the live box too)
          # SRV-471: the memory switch (the live box set it on in both, coding v20 and chat v5)
          # SRV-475: the sampling section (the migration stores it on the live box too)
          # SRV-623: C5 and its pack (the migration stores them in coding and chat; absent in
          # overnight reads as off, its shipped value)
          "profiles/coding": {"thinking.by_size", "thinking.client_word", "shadow", "memory",
                              "sampling", "tasks.C5", "toolgate.sequence"},
          "profiles/overnight": {"thinking.by_size", "ui", "thinking.client_word", "sampling",
                                 "tasks.C5", "toolgate.sequence"},
          "profiles/chat": {"thinking.by_size", "ui", "memory", "sampling", "tasks.C5",
                            "toolgate.sequence"},
          # 2026-10-05: the memory card and skill (the deploy stores them on the live box too)
          # SRV-615: the deliverable of six everyday packages (the migration stores it on the
          # live box too)
          "packages": {"packages.tandem-memory", "packages.memory"} | {
              f"packages.{n}.deliverable" for n in (
                  "write-message", "formal-letter", "event-plan", "weekly-schedule",
                  "checklist-builder", "social-post")}}
# what the shipped documents changed on purpose since that day, keyed by the document: the path
# and the value they ship. SRV-473: the same-call guard on (the live box: coding v21 by hand,
# overnight and chat by the migration), chat's C3 acting, chat's description. SRV-534: the
# narrator off in coding and chat; a core on design_ui and drawing (the
# migration stores it on the live box too)
CHANGED = {"router": {"classes.design_ui.policy.core": ["ui-design"],
                      "classes.drawing.policy.core": ["creative-coding"],
                      "classes.three_d.policy.core": ["threejs-scene", "coding"],
                      "classes.three_d.policy.packages": next(
                          c["policy"]["packages"] for c in schema.SHIPPED_ROUTER["classes"]
                          if c["id"] == "three_d")},
           # SRV-625: the secret echo mask in coding and chat
           "profiles/coding": {"toolgate.same_call": True, "ui.narrator": "off",
                               "secret_mask": "on"},
           "profiles/overnight": {"toolgate.same_call": True},
           # SRV-621: the profile an unmapped client that offers tools gets
           "settings": {"tools_profile": "tools"},
           "profiles/chat": {"toolgate.same_call": True, "tasks.C3.mode": "act",
                             "ui.narrator": "off", "secret_mask": "on",
                             "description": "Open WebUI and the Playground: class-aware effort, "
                                            "the same-call guard."}}


def _changed(want: dict, kind: str) -> dict:
    out = json.loads(json.dumps(want))
    for path, val in CHANGED.get(kind, {}).items():
        node, keys = out, path.split(".")
        for k in keys[:-1]:
            # a list of classes: the one with that id
            node = next(x for x in node if x.get("id") == k) if isinstance(node, list) else node[k]
        node[keys[-1]] = val
    return out


HERMES_TOOLS = ["terminal", "process", "read_file", "write_file", "patch", "search_files", "todo",
                "memory", "session_search", "skills_list", "skill_view", "skill_manage",
                "execute_code", "delegate_task", "clarify"]


def _fixture(name):
    with open(os.path.join(FIX, name)) as f:
        return json.load(f)["doc"]


def _same(want, got, path, filled, out):
    """Every key of `want` equal in `got`; a key only `got` has must be one the store fills."""
    if isinstance(want, dict) and isinstance(got, dict):
        for k in want:
            if k not in got:
                out.append(f"{path}{k}: missing")
            else:
                _same(want[k], got[k], f"{path}{k}.", filled, out)
        for k in got:
            if k not in want and f"{path}{k}" not in filled:
                out.append(f"{path}{k}: not in the live document")
    elif want != got:
        out.append(f"{path[:-1]}: {json.dumps(want)[:80]} != {json.dumps(got)[:80]}")


def _server(home, cdir):
    port = _port()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "PYTHONPATH": ROOT,
           "CUDA_VISIBLE_DEVICES": "", "QSE_ADMIN_TOKEN": ADMIN}
    ledger = os.path.join(home, "ledger.sqlite3")
    proc = subprocess.Popen([sys.executable, "-c", LAUNCH, "--fake-engine", "--port", str(port),
                             "--served-model", "qwen38-spark-engine", "--usage-ledger", ledger,
                             "--fake-tps", "2000", "--conductor-config", cdir,
                             "--conductor-defaults", "shipped"],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    _wait(port, proc)
    return port, proc


def _chat(port, ua, body):
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": "TandemLLM", "max_tokens": 64, "stream": False, **body}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": ua})
    with urllib.request.urlopen(req, timeout=60) as r:
        return dict(r.headers), json.loads(r.read())


def test_a_new_box_serves_the_live_configuration():
    home = tempfile.mkdtemp(prefix="qse-home-")
    cdir = os.path.join(home, "conductor")
    port, proc = _server(home, cdir)
    try:
        # ---- 1 the documents
        bad = []
        for kind, fix in (("router", "router.json"), ("packages", "packages.json"),
                          ("settings", "settings.json"), ("profiles/coding", "profile-coding.json"),
                          ("profiles/overnight", "profile-overnight.json"),
                          ("profiles/chat", "profile-chat.json")):
            st, _, got = _call(port, "GET", f"/v1/conductor/{kind}")
            assert st == 200, (kind, got)
            assert got["version"] == 1, (kind, got["version"])    # stored at the start
            want = _changed(_fixture(fix), kind)
            if kind == "settings":
                assert got["doc"]["default_profile"] == "general"
                want = dict(want, default_profile="general")
            if kind == "packages":
                # the shipped directories after the box's own
                assert got["doc"]["skill_dirs"] == want["skill_dirs"] + [
                    "{repo}/packages/skills", "{repo}/packages/library",
                    "{repo}/packages/everyday"], got["doc"]["skill_dirs"]
                want = dict(want, skill_dirs=got["doc"]["skill_dirs"])
            _same(want, got["doc"], "", FILLED.get(kind, set()), bad)
            bad = [f"{kind}: {b}" for b in bad] if bad and not bad[0].startswith(kind) else bad
        assert not bad, "\n".join(bad[:20])
        # the two new profiles
        for name, clients, channel in (("hermes", ["hermes"], "reasoning"),
                                       ("general", [], "reasoning")):
            st, _, got = _call(port, "GET", f"/v1/conductor/profiles/{name}")
            assert st == 200 and got["version"] == 1, got
            d = got["doc"]
            assert d["clients"] == clients and d["mode"] == "on", d
            assert d["ui"] == {"states": "on", "states_channel": channel}, d["ui"]
        coding = _call(port, "GET", "/v1/conductor/profiles/coding")[2]["doc"]
        hermes = _call(port, "GET", "/v1/conductor/profiles/hermes")[2]["doc"]
        for k in ("default_class", "decisions", "routing", "tasks", "toolgate", "outcomes",
                  "library", "thinking", "caps", "context"):
            if k == "thinking":
                # Hermes's effort word is a ceiling, opencode's is ignored
                assert hermes[k].get("client_word", "ceiling") == "ceiling"
                assert {**hermes[k], "client_word": "x"} == {**coding[k], "client_word": "x"}
                continue
            assert hermes[k] == coding[k], k
        general = _call(port, "GET", "/v1/conductor/profiles/general")[2]["doc"]
        assert general["default_class"] == "chat" and general["decisions"]["packages"] == "on"
        # ---- 2 the packages: no file missing but the five operator skills, the library live
        st, _, cat = _call(port, "GET", "/v1/conductor/packages/catalog")
        assert st == 200, cat
        missing = sorted(n for n, r in cat["packages"].items() if not r["available"])
        assert all(r["path"] is None or "/packages/" in r["path"] or r["source"] == "inline"
                   for r in cat["packages"].values() if r["available"]), "a shipped file"
        assert missing == sorted(["pencil-sketch", "spiral-portrait", "one-line-portrait",
                                  "photo-to-3d", "voxel-seasons"]), missing
        st, _, hits = _call(port, "GET", "/v1/library/search?q=canvas+game+collisions&k=3",
                            token=None)
        assert st == 200 and hits and hits[0]["name"] == "browser-game", hits
        st, _, card = _call(port, "GET", "/v1/library/laya-playtest", token=None)
        assert st == 200, card
        text = card["body"] + card["reference"]
        assert "{tandem_url}" not in text and f"http://127.0.0.1:{port}/v1/laya/kit/" in text, \
            text[:300]
        # ---- 3 the clients, no header
        h, _ = _chat(port, "opencode/1.18.34 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14",
                     {"messages": [{"role": "user", "content": "say ok"}]})
        assert h.get("X-Tandem-Profile") == "coding", h
        tools = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
                 for n in HERMES_TOOLS]
        h, _ = _chat(port, "OpenAI/Python 2.24.0",
                     {"messages": [{"role": "system", "content": "You are a helpful agent."},
                                   {"role": "user", "content": "say ok"}], "tools": tools})
        assert h.get("X-Tandem-Profile") == "hermes", h
        h, _ = _chat(port, "OpenAI/Python 2.24.0",
                     {"messages": [{"role": "user", "content": "say ok"}]})
        assert h.get("X-Tandem-Profile") == "general", h
        h, _ = _chat(port, "curl/8.5.0", {"messages": [{"role": "user", "content": "say ok"}]})
        assert h.get("X-Tandem-Profile") == "general", h
        # SRV-621: the same clients with ordinary tools get `tools`
        plain = [{"type": "function", "function": {"name": "get_weather",
                                                    "parameters": {"type": "object"}}}]
        for ua in ("OpenAI/Python 2.24.0", "curl/8.5.0"):
            h, _ = _chat(port, ua, {"messages": [{"role": "user", "content": "say ok"}],
                                    "tools": plain})
            assert h.get("X-Tandem-Profile") == "tools", (ua, h)
        # ---- 4 the disk tier's settings
        st, _, s = _call(port, "GET", "/v1/conductor/settings")
        assert s["doc"]["memory"]["disk_cache"] == {"mode": "on", "budget_gb": 16, "ttl_days": 7,
                                                    "admit_after": 2, "min_free_gb": 50}
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def test_serve_env_turns_the_served_features_on():
    with open(os.path.join(ROOT, "ops", "serve.env")) as f:
        env = dict(re.findall(r"(?m)^([A-Z0-9_]+)=(.*)$", f.read()))
    assert env["CONDUCTOR"] == "on", env["CONDUCTOR"]
    assert env["END_ANCHOR"] == "on" and env["END_ALIGN"] == "on"
    assert env["QSE_CONDUCTOR_CONFIG"] == "${HOME}/.qwen38-spark-engine/conductor"
    assert env["QSE_DISK_CACHE"] == "${HOME}/.qwen38-spark-engine/cache/disk"
    with open(os.path.join(ROOT, "ops", "start.sh")) as f:
        start = f.read()
    for flag in ("--end-anchor off", "--end-align off", '--disk-cache "${QSE_DISK_CACHE:-off}"',
                 '--conductor-config "${QSE_CONDUCTOR_CONFIG:-off}" --conductor-defaults shipped'):
        assert flag in start, flag


def test_the_voice_and_the_shadow_share_two_and_a_half_percent():
    """The voice and the shadow's in-turn reading draw on one share of Qwen's time
    (narrator.Share). At 3 % with both on, Qwen's decode lost 3.34 % (night pass 2026-10-04,
    63.33 -> 61.22 tok/s, same text); at 2.5 % it lost 2.79 %. A profile with both on ships 2.5 %."""
    from server.conductor import schema, voice
    for name, doc in schema.SHIPPED_PROFILES.items():
        ui = doc.get("ui") or {}
        both = ui.get("narrator") == "live" and (doc.get("shadow") or {}).get("mode") == "on"
        share = (ui.get("live") or {}).get("share", voice.DEFAULTS["share"])
        if both:
            assert share <= 0.025, (name, share)
    assert schema.SHIPPED_PROFILES["coding"]["ui"]["live"]["share"] == 0.025
    assert voice.DEFAULTS["share"] == 0.03, "a profile with the voice alone keeps 3 %"


def test_the_shipped_documents_validate():
    from server.conductor import schema, shipped
    pk, router = schema.SHIPPED_PACKAGES, schema.SHIPPED_ROUTER
    for kind, name, doc in shipped.documents():
        errs = schema.check(kind, doc, name=name, router=router, packages=pk,
                            profiles=schema.SHIPPED_PROFILES)
        assert not errs, (kind, name, errs)
    # every package a class lists exists, and every shipped body is a file of packages/
    for c in router["classes"]:
        for p in c["policy"]["packages"]:
            assert p in pk["packages"], (c["id"], p)
    # no personal name in the shipped packages: the copyright holder's, read from NOTICE
    with open(os.path.join(ROOT, "NOTICE")) as f:
        holder = f.read().split("Copyright (C) 2026 ", 1)[1].split("\n", 1)[0].split()
    blob = json.dumps(pk).lower()
    assert not any(part.lower() in blob for part in holder), holder


def test_a_box_with_its_own_documents_keeps_them():
    """A box with documents of its own: a router and a coding profile of its own stay as they are; what it never had
    (the hermes and general profiles) is stored; a second start stores nothing."""
    from server.conductor import schema, shipped
    from server.conductor.store import Store
    d = tempfile.mkdtemp(prefix="qse-cond-")
    st = Store(d)
    st.put("router", None, schema.seed("router"), note="own", author="t")
    st.put("profiles", "coding", schema.seed("profiles", "coding"), note="own", author="t")
    got = shipped.install(st)
    assert not any(x.startswith(("router", "profiles/coding")) for x in got), got
    assert "profiles/hermes v1" in got and "profiles/general v1" in got, got
    assert st.get("router")[1] == schema.seed("router")
    assert st.get("profiles", "hermes")[1]["clients"] == ["hermes"]
    assert shipped.install(st) == []


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:70s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
