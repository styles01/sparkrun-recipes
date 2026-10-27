"""Open WebUI tools and user skills in chatd (SRV-361): the tools file parsed with `ast` only
(three fixtures in tests/fixtures/owui, no personal data), schemas from the type hints and the
docstrings, Valves as settings with write-only secrets, a call that runs the method through the
runner (a sandbox played on the host here; the Spark's box test runs it in a real one), filters
refused; a skill zip checked, unpacked and installed through the engine's packages document, and
taken out again. CPU only.

Run: python -m pytest -q tests/test_chatd_owui.py
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile

from chatd_util import Chat, run_all
from test_chatd_agent import FakeBoxes, calls

from chatd import owui, skills

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "owui")


def fixture(name: str) -> str:
    with open(os.path.join(FIX, name)) as fh:
        return fh.read()


def multipart(name: str, data: bytes, field: str = "file") -> tuple[str, bytes]:
    b = "----tandemtest"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + \
        f"\r\n--{b}--\r\n".encode()
    return f"multipart/form-data; boundary={b}", body


# ---------------------------------------------------------------------- the parser
def test_three_tools_files_parse_to_schemas_and_valves():
    calc = owui.parse(fixture("calculator.py"))
    assert calc["title"] == "Calculator" and calc["requirements"] == []
    assert [t["method"] for t in calc["tools"]] == ["calculate", "convert_length"]
    conv = calc["tools"][1]
    assert conv["description"] == "Convert a length between kilometres and miles."
    assert conv["parameters"] == {"type": "object", "required": ["value", "unit"], "properties": {
        "value": {"type": "number", "description": "The length to convert."},
        "unit": {"type": "string", "enum": ["km", "mi"], "description": "The unit of the value."},
        "digits": {"type": "integer", "description": "Digits after the point."}}}

    weather = owui.parse(fixture("weather.py"))
    assert weather["requirements"] == ["requests"]
    get = weather["tools"][0]
    assert get["method"] == "get_weather"
    assert list(get["parameters"]["properties"]) == ["city"]          # no __x__ parameters
    assert get["parameters"]["required"] == ["city"]
    assert weather["valves"] == [
        {"name": "API_KEY", "type": "string", "default": "", "secret": True,
         "description": "The weather service's API key"},
        {"name": "UNITS", "type": "string", "default": "metric", "secret": False,
         "description": "metric or imperial"},
        {"name": "TIMEOUT", "type": "integer", "default": 10, "secret": False, "description": ""}]

    words = owui.parse(fixture("notes.py"))
    assert [t["method"] for t in words["tools"]] == ["word_count", "shout", "tags"]
    assert words["tools"][0]["parameters"]["required"] == ["text"]
    assert words["tools"][2]["parameters"]["properties"] == {
        "items": {"type": "array", "items": {"type": "string"}}, "sep": {"type": "string"}}
    assert words["tools"][2]["parameters"]["required"] == ["items"]


def test_filters_pipes_and_bad_files_are_refused():
    for src, words in [(fixture("filter.py"), "filters, pipes and actions are not imported"),
                       ("class Tools:\n    def _x(self): pass\n", "no public methods"),
                       ("x = 1\n", "no `class Tools`"),
                       ("class Tools(:\n", "not valid Python"),
                       ('"""\nrequirements: requests; rm -rf /\n"""\nclass Tools:\n'
                        '    def a(self) -> str:\n        return ""\n', "not a plain package")]:
        try:
            owui.parse(src)
            raise AssertionError("refused expected: " + words)
        except owui.ParseError as exc:
            assert words in str(exc), (words, str(exc))


# ---------------------------------------------------------------------- through chatd
class HostBoxes(FakeBoxes):
    """A sandbox played on the host for the runner: the command runs with HOME = the chat's home
    folder. Only the fixtures without requirements or pydantic run here."""

    def run(self, cid, cmd, *, timeout=600, on_output=None, cancel=None):
        self.ran.append(cmd)
        _, home, _ = self.paths(cid)
        os.makedirs(home, exist_ok=True)
        cmd = [x.replace("python3 run.py", f"{sys.executable} run.py") for x in cmd]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "HOME": home})
        on_output and on_output(p.stdout + p.stderr)
        return p.returncode, False


def owui_chat():
    c = Chat()
    c.svc.sandboxes = HostBoxes(c.data)
    c.svc.sandboxes.on_change = c.svc._sandbox_changed
    c.svc.sandboxes.network_mode = c.svc._network_mode
    c.svc._caps = None
    return c


def test_an_imported_toolkit_runs_in_the_sandbox_with_its_valves():
    c = owui_chat()
    try:
        ct, body = multipart("notes.py", fixture("notes.py").encode())
        st, kit, _ = c.call("POST", "/tools/import/owui", raw=body, headers={"Content-Type": ct})
        assert st == 200, kit
        assert kit["toolkit"]["slug"] == "word_tools"
        assert [t["id"] for t in kit["tools"]] == ["owui__word_tools__word_count",
                                                   "owui__word_tools__shout",
                                                   "owui__word_tools__tags"]
        assert kit["tools"][0]["source"] == {"kind": "user", "id": kit["toolkit"]["id"]}
        assert kit["valves"][0]["name"] == "UPPER" and kit["valves"][0]["value"] is False
        c.ok("PATCH", f"/tools/user/{kit['toolkit']['id']}", {"valves": {"UPPER": True}})
        c.svc.add_rule("owui__word_tools__*", None, "allow", None)
        cid = c.new()
        c.send(cid, calls(("owui__word_tools__word_count", {"text": "a b a", "unique": True}),
                          ("owui__word_tools__shout", {"text": "hey"})))
        s = c.wait_idle(cid, 60)
        got = {p["name"]: p for p in s["messages"][-1]["parts"] if p["type"] == "tool_call"}
        wc, sh = got["owui__word_tools__word_count"], got["owui__word_tools__shout"]
        assert wc["status"] == "done" and wc["output"]["content"] == '{"words": 2}', wc["output"]
        assert sh["status"] == "done" and sh["output"]["content"] == "HEY", sh
        assert "owui__word_tools__tags" in [t["function"]["name"] for t in
                                            [b for b in c.fake.bodies if b.get("stream")][0]["tools"]]
        # the request file with the valves is gone after the call
        _, home, _ = c.svc.sandboxes.paths(cid)
        assert not [f for f in os.listdir(os.path.join(home, ".tandem", "owui"))
                    if f.startswith("req-")]
        c.call("DELETE", f"/tools/user/{kit['toolkit']['id']}")
        assert c.ok("GET", "/tools/user") == []
        assert not any(t["id"].startswith("owui__") for t in c.ok("GET", "/tools"))
    finally:
        c.close()


def test_secret_valves_are_write_only_and_a_json_export_imports_every_entry():
    c = owui_chat()
    try:
        export = json.dumps([{"id": "weather", "name": "Weather", "content": fixture("weather.py")},
                             {"id": "calc", "name": "Calc", "content": fixture("calculator.py")}])
        ct, body = multipart("tools-export.json", export.encode())
        st, kit, _ = c.call("POST", "/tools/import/owui", raw=body, headers={"Content-Type": ct})
        assert st == 200 and [k["slug"] for k in kit["toolkits"]] == ["weather_lookup",
                                                                      "calculator"]
        wid = kit["toolkits"][0]["id"]
        secret = "wk-" + "q" * 20
        c.ok("PATCH", f"/tools/user/{wid}", {"valves": {"API_KEY": secret, "UNITS": "imperial"}})
        info = next(k for k in c.ok("GET", "/tools/user") if k["id"] == wid)
        assert {v["name"]: v["value"] for v in info["valves"]} == {
            "API_KEY": "", "UNITS": "imperial", "TIMEOUT": 10}
        assert next(v for v in info["valves"] if v["name"] == "API_KEY")["set"] is True
        c.ok("PATCH", f"/tools/user/{wid}", {"valves": {"API_KEY": ""}})     # empty keeps it
        r = c.svc.store.one("SELECT valves_enc FROM user_tools WHERE id = ?", (wid,))
        assert secret.encode() not in r[0] and c.svc.owui._valves(
            c.svc.owui._row(wid))["API_KEY"] == secret
        assert secret not in json.dumps(c.ok("GET", "/tools/user")) + "\n".join(c.lines)
        st, err, _ = c.call("POST", "/tools/import/owui", raw=multipart(
            "f.py", fixture("filter.py").encode())[1], headers={"Content-Type": ct})
        assert st == 400 and "not imported" in err["error"]["message"]
    finally:
        c.close()


# ---------------------------------------------------------------------- user skills
def skill_zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            if isinstance(data, tuple):                  # (target, "link")
                info = zipfile.ZipInfo(name)
                info.external_attr = (0o120777 << 16)
                z.writestr(info, data[0])
            else:
                z.writestr(name, data)
    return buf.getvalue()


SKILL = "---\nname: tea-timer\ndescription: Brew times for green, black and oolong tea.\n---\n" \
        "Green: 2 minutes at 80 C.\n"


def test_a_skill_zip_is_checked():
    meta, files = skills.unpack_skill(skill_zip({"tea-timer/SKILL.md": SKILL,
                                                 "tea-timer/ref/table.md": "x"}))
    assert meta["name"] == "tea-timer" and set(files) == {"SKILL.md", "ref/table.md"}
    meta, files = skills.unpack_skill(skill_zip({"SKILL.md": SKILL}))
    assert set(files) == {"SKILL.md"}
    for z, words in [
        (skill_zip({"README.md": "x"}), "no SKILL.md"),
        (skill_zip({"t/SKILL.md": SKILL, "other/x.md": "x"}), "outside the skill's folder"),
        (skill_zip({"t/SKILL.md": SKILL, "t/../../etc/x": "x"}), "outside the skill's folder"),
        (skill_zip({"t/SKILL.md": SKILL, "t/link": ("/etc/passwd", "link")}), "is a link"),
        (skill_zip({"t/SKILL.md": "---\nname: Bad Name\ndescription: long enough here\n---\n"}),
         "needs a `name`"),
        (skill_zip({"t/SKILL.md": "---\nname: ok\ndescription: short\n---\n"}),
         "needs a `description`"),
        (b"not a zip", "not a zip"),
    ]:
        try:
            skills.unpack_skill(z)
            raise AssertionError("refused expected: " + words)
        except ValueError as exc:
            assert words in str(exc), (words, str(exc))


def engine_with_packages(c):
    """The engine's packages document, as GET and PUT /v1/conductor/packages answer it."""
    state = {"v": 3, "doc": {"skill_dirs": ["~/.config/opencode/skills"], "packages": {
        "coding": {"title": "Coding", "kind": "package", "source": "inline",
                   "description": "How programming work happens.", "body": "x",
                   "max_tokens": 900, "enabled": True}}}, "puts": []}
    real = c.svc.engine.request

    def request(method, path, body=None, headers=None, timeout=10.0):
        if path != "/v1/conductor/packages":
            return real(method, path, body, headers, timeout)
        if method == "GET":
            return 200, {"version": state["v"], "doc": json.loads(json.dumps(state["doc"]))}
        b = json.loads(body)
        if b["base_version"] != state["v"]:
            return 409, {"error": {"message": "the document moved"}}
        state["v"] += 1
        state["doc"] = b["doc"]
        state["puts"].append(b["note"])
        return 200, {"version": state["v"]}
    c.svc.engine.request = request
    c.svc.library = lambda: [{"name": n, "description": p["description"], "classes": [],
                              "tokens": None}
                             for n, p in state["doc"]["packages"].items() if p.get("library")]
    return state


def test_a_skill_installs_through_the_packages_document_and_comes_out_again():
    c = Chat()
    try:
        state = engine_with_packages(c)
        ct, body = multipart("tea.zip", skill_zip({"tea-timer/SKILL.md": SKILL,
                                                   "tea-timer/ref/table.md": "oolong 3 min"}))
        pkg = c.ok("POST", "/packages/upload", raw=body, headers={"Content-Type": ct})
        assert pkg["id"] == "tea-timer" and pkg["user"] is True
        doc = state["doc"]
        assert c.svc.skills_dir in doc["skill_dirs"]
        assert doc["packages"]["tea-timer"]["source"] == "skill:tea-timer"
        assert doc["packages"]["tea-timer"]["library"] is True
        assert open(os.path.join(c.svc.skills_dir, "tea-timer", "ref", "table.md")).read() == \
            "oolong 3 min"
        assert [p["id"] for p in c.ok("GET", "/skills")] == ["tea-timer"]
        # again: replaced, not refused
        c.ok("POST", "/packages/upload", raw=body, headers={"Content-Type": ct})
        # a name the engine has from elsewhere
        taken = SKILL.replace("tea-timer", "coding")
        st, err, _ = c.call("POST", "/packages/upload", raw=multipart(
            "c.zip", skill_zip({"SKILL.md": taken}))[1], headers={"Content-Type": ct})
        assert st == 409 and "already has" in err["error"]["message"]
        st, err, _ = c.call("POST", "/packages/upload", raw=multipart(
            "bad.zip", skill_zip({"x/README.md": "x"}))[1], headers={"Content-Type": ct})
        assert st == 400 and "no SKILL.md" in err["error"]["message"]
        assert c.call("DELETE", "/packages/coding")[0] == 404          # the engine's own
        assert c.call("DELETE", "/packages/tea-timer")[0] == 204
        assert "tea-timer" not in state["doc"]["packages"]
        assert not os.path.exists(os.path.join(c.svc.skills_dir, "tea-timer"))
        assert state["puts"][0] == "chat: the user skill tea-timer"
    finally:
        c.close()


if __name__ == "__main__":
    run_all(globals())
