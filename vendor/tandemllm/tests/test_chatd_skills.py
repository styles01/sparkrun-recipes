"""Skills, prompts and presets in chatd (SRV-361): the three skill modes as the body's
`tandem.packages` and the system block's index, a skill switched off everywhere, prompts as slash
commands and as a chat's system prompt, presets for a new chat. CPU only.

Run: python -m pytest -q tests/test_chatd_skills.py
"""

from __future__ import annotations

from chatd_util import Chat, run_all

LIB = [{"name": "web-design", "description": "Pages that do not look like templates.",
        "classes": ["design_ui"], "tokens": 900},
       {"name": "fact-check", "description": "Check claims before answering.", "classes": [],
        "tokens": 400},
       {"name": "coding", "description": "How programming work happens.", "classes": [],
        "tokens": 1200}]


def engine_chat():
    c = Chat()
    c.svc.library = lambda: LIB
    real = c.svc.model_info

    def info(mid):
        return {**real(mid), "backend": "engine"}
    c.svc.model_info = info
    return c


def last(c):
    return [b for b in c.fake.bodies if b.get("stream")][-1]


def test_the_three_modes_reach_the_router_as_pins_and_blocks():
    c = engine_chat()
    try:
        assert [p["id"] for p in c.ok("GET", "/packages")] == ["web-design", "fact-check",
                                                               "coding"]
        cid = c.new()
        c.ok("PATCH", f"/conversations/{cid}/settings", {"packages": ["fact-check"],
                                                         "packages_blocked": ["coding"]})
        c.send(cid, "hi")
        c.wait_idle(cid)
        assert last(c)["tandem"] == {"compaction": False, "packages": {
            "pin": ["fact-check"], "block": ["coding"]}}
        c.ok("PATCH", f"/conversations/{cid}/settings", {"packages_mode": "manual"})
        c.send(cid, "again")
        c.wait_idle(cid)
        assert last(c)["tandem"]["packages"] == {"pin": ["fact-check"],
                                                  "block": ["coding", "web-design"]}
        c.ok("PATCH", f"/conversations/{cid}/settings", {"packages_mode": "model"})
        c.send(cid, "and again")
        c.wait_idle(cid)
        body = last(c)
        system = body["messages"][0]["content"]
        assert "- web-design: Pages that do not look like templates." in system
        assert "- coding:" not in system                    # blocked in this chat
        assert body["tandem"]["packages"]["block"] == ["coding", "web-design"]
        # a skill switched off in the settings is blocked in every chat
        c.ok("PATCH", "/packages/web-design", {"enabled": False})
        assert not next(p for p in c.ok("GET", "/packages") if p["id"] == "web-design")["enabled"]
        cid2 = c.new()
        c.send(cid2, "x")
        c.wait_idle(cid2)
        assert last(c)["tandem"]["packages"]["block"] == ["web-design"]
    finally:
        c.close()


def test_a_pin_shows_in_the_panel_with_its_tokens_and_a_gone_one_is_left_out():
    import time
    c = engine_chat()
    c.svc._lib_cache = (time.monotonic(), LIB)
    try:
        cid = c.new()
        c.ok("PATCH", f"/conversations/{cid}/settings", {"packages": ["web-design", "old-skill"]})
        sk = c.snap(cid)["context"]["skills"]
        assert sk["mode"] == "auto" and sk["pinned"] == [
            {"name": "web-design", "tokens": 900, "missing": False},
            {"name": "old-skill", "tokens": None, "missing": True}]
        c.send(cid, "hi")
        s = c.wait_idle(cid)
        assert last(c)["tandem"]["packages"]["pin"] == ["web-design"]
        notes = [p["message"] for p in s["messages"][-1]["parts"] if p["type"] == "error"]
        assert notes == ["The pinned skill old-skill is no longer in the library; the request "
                         "went without."]
        assert s["messages"][-1]["status"] == "done"
    finally:
        c.close()


def test_a_server_that_is_not_ours_gets_no_tandem_field():
    c = Chat()
    try:
        cid = c.new()
        c.ok("PATCH", f"/conversations/{cid}/settings", {"packages": ["x"]})
        c.send(cid, "hi")
        c.wait_idle(cid)
        assert "tandem" not in last(c)
    finally:
        c.close()


def test_prompts_are_slash_commands_and_a_chat_s_system_prompt():
    c = Chat()
    try:
        p = c.ok("POST", "/prompts", {"command": "/Review", "title": "Review a diff",
                                      "body": "You review code like a strict senior engineer."})
        assert p["command"] == "review"
        st, obj, _ = c.call("POST", "/prompts", {"command": "review", "body": "x"})
        assert st == 409
        cmds = c.ok("GET", "/capabilities")["commands"]
        assert {"command": "review", "description": "Review a diff", "prompt_id": p["id"]} in cmds
        p2 = c.ok("POST", f"/prompts/{p['id']}", {"command": "review", "title": "Review",
                                                  "body": "Be strict."})
        assert p2["body"] == "Be strict." and len(c.ok("GET", "/prompts")) == 1
        cid = c.new()
        c.ok("PATCH", f"/conversations/{cid}/settings", {"prompt_id": p["id"]})
        c.send(cid, "look at this")
        c.wait_idle(cid)
        assert last(c)["messages"][0]["content"].startswith("Be strict.")
        assert c.call("DELETE", f"/prompts/{p['id']}")[0] == 204 and c.ok("GET", "/prompts") == []
    finally:
        c.close()


def test_presets_start_a_chat_and_later_edits_leave_it_alone():
    c = Chat()
    try:
        pr = c.ok("POST", "/presets", {"name": "Careful", "effort": "high", "thinking": True,
                                       "tools": ["read_file"], "packages": ["fact-check"]})
        s = c.ok("POST", "/conversations", {"preset_id": pr["id"]})["settings"]
        assert s["effort"] == "high" and s["tools"] == ["read_file"]
        assert s["packages"] == ["fact-check"]
        c.ok("POST", f"/presets/{pr['id']}", {"name": "Careful", "effort": "low"})
        cid = s and c.ok("GET", "/conversations")["conversations"][0]["id"]
        assert c.snap(cid)["settings"]["effort"] == "high"
        st, _, _ = c.call("POST", "/conversations", {"preset_id": "nope"})
        assert st == 404
        assert c.call("DELETE", f"/presets/{pr['id']}")[0] == 204
    finally:
        c.close()


if __name__ == "__main__":
    run_all(globals())
