"""The setup guide's read routes and the estimate route (SRV-323, server/setup_api.py), handler-level
on a CPU (a request without a socket, as tests/test_app_loop.py's).

The test matrix:

  1  GET /v1/setup/models: a tile for qwen3.8-27b (recommended, best speed 84.44 tok/s code on
     exl3-4.5), no load errors                           test_models
  2  GET /v1/setup/models/{id}: the registry payload with the contract version and wizard_fields;
     an unknown id is 404 unknown_model; /steps has the eight steps and the seven tools
                                                         test_model_detail_and_steps
  3  GET /v1/setup/device with a fake memory source (119.7 GiB, 61 GiB available) and a fake GPU:
     every block filled, the password a boolean; without a GPU and without /proc/meminfo the
     blocks are null and the answer still 200            test_device
  4  POST /v1/setup/estimate: the default fits; an unknown profile is 400; slots 17 and window 512
     are 422 out_of_range; the narrator without the conductor is 422 invalid_combination naming
     both; a body with a command in it is 422 unknown_field   test_estimate_route
  5  the admin guard: with an admin token set, every /v1/setup route through the proxy is 401
     without it and 200 with it; with none set every route is open   test_admin_guard
  6  no secret leaves the box: the admin token and an HF token are in no answer
                                                         test_no_secret_in_any_answer
  7  the device route stays cheap: 50 polls, each under 200 ms, the sources read once a second
                                                         test_device_is_cheap
  8  CSRF: a POST that is not declared JSON is 415, `Sec-Fetch-Site: cross-site` and an Origin of
     another host are 403 (open box and loopback alike), and nothing ran; the page's own POST
     passes directly (Origin = Host) and through nginx (Origin = $host, or X-Forwarded-Host); a
     tool with neither header passes on its JSON         test_cross_site_posts_are_refused

Run: python tests/test_setup_api.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import app, auth  # noqa: E402
from server.setup import device as device_mod  # noqa: E402

ADMIN = "adm-" + "5e7u9" * 10
HF = "hf_" + "secretvalue" * 4
PROXY = {"X-Forwarded-For": "203.0.113.44", "X-Real-IP": "203.0.113.44"}


class FakeSampler:
    def __init__(self, name="NVIDIA GB10"):
        self.name, self.n = name, 0

    def sample(self):
        self.n += 1
        return {"name": self.name, "temperature_c": 51.0, "power_w": 33.2, "sm_clock_mhz": 2400.0,
                "utilization": 0.07 if self.name else None}


class Req(app.Handler):
    """The handler without a socket (tests/test_app_loop.py's, without its engine: the setup
    routes need none): the body in, the raw response bytes out."""

    def __init__(self, path, body):
        raw = json.dumps(body).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json"}
        self.path, self.command, self.request_version = path, "POST", "HTTP/1.1"
        self.requestline, self.client_address = "POST " + path, ("127.0.0.1", 0)
        self.close_connection = True


def _setup(admin=None, gpu="NVIDIA GB10", mem=None):
    app.STATE.clear()
    app.STATE.update(auth=auth.Auth(admin, None), gpu_sampler=FakeSampler(gpu),
                     started=time.time() - 60, version="0.6.2+exl3-4.5")
    device_mod.clear()
    device_mod.meminfo = (lambda: dict(mem)) if mem is not None else _real_meminfo


_real_meminfo = device_mod.meminfo


def call(method, path, body=None, headers=None):
    req = Req(path, body if body is not None else {})
    req.command = method
    req.headers.update(headers or {})
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        getattr(req, f"do_{method}")()
    head, _, raw = req.wfile.getvalue().partition(b"\r\n\r\n")
    code = int(head.decode().split(" ", 2)[1])
    return code, json.loads(raw) if raw else None


def test_models():
    _setup()
    code, body = call("GET", "/v1/setup/models")
    assert code == 200 and body["contract_version"] == "1.0" and body["errors"] == [], body
    t = body["models"][0]
    assert t["id"] == "qwen3.8-27b" and t["recommended"] is True
    assert t["best_speed"] == {"tok_s": 84.44, "task": "code", "profile": "exl3-4.5"}, t


def test_model_detail_and_steps():
    _setup()
    code, m = call("GET", "/v1/setup/models/qwen3.8-27b")
    assert code == 200 and m["contract_version"] == "1.0" and m["schema"] == 1
    for k in ("profiles", "components", "parallelism", "methods", "advanced", "measurements",
              "devices", "memory", "assets", "wizard_fields"):
        assert k in m, k
    assert [s["id"] for s in m["wizard_fields"]][:3] == ["models", "overview", "weights"]
    code, body = call("GET", "/v1/setup/models/nope")
    assert code == 404 and body["error"]["code"] == "unknown_model", body
    code, body = call("GET", "/v1/setup/models/qwen3.8-27b/steps")
    assert code == 200 and len(body["steps"]) == 8
    assert [t["function"]["name"] for t in body["tools"]] == [
        "set_field", "toggle", "select_profile", "go_to_step", "next", "previous", "explain"]
    code, body = call("GET", "/v1/setup/nothing")
    assert code == 404


def test_device():
    _setup(mem={"MemTotal": 119.7, "MemAvailable": 61.0})
    code, d = call("GET", "/v1/setup/device")
    assert code == 200, d
    assert d["mem_total_gb"] == 119.7 and d["mem_available_gb"] == 61.0
    assert d["gpu"] == {"name": "NVIDIA GB10", "util_pct": 7.0, "temp_c": 51.0, "power_w": 33.2}
    assert isinstance(d["password_set"], bool) and d["admin_password_set"] is d["password_set"]
    assert isinstance(d["hf_token_present"], bool)
    assert set(d["profiles_on_disk"]) >= {"exl3-4.5", "nvfp4"}
    assert all(isinstance(v, bool) for v in d["profiles_on_disk"].values())
    assert d["mode"] == "serving" and d["engine"]["mode"] == "serving"
    assert d["engine"]["busy"] is False and d["engine"]["version"] == "0.6.2+exl3-4.5"
    assert d["preset"] == "dgx-spark", d["preset"]                 # 119.7 is within 2 GiB
    assert d["assistant"]["name"] == "Wren"
    _setup(gpu=None, mem={})
    code, d = call("GET", "/v1/setup/device")
    assert code == 200 and d["gpu"] is None and d["mem_total_gb"] is None and d["preset"] is None


def test_estimate_route():
    _setup(mem={"MemTotal": 121.63, "MemAvailable": 61.0})
    # the reference box's tools: the narrator on (off by default since 0.11.3, SRV-534)
    ref = {"model": "qwen3.8-27b", "components": {"narrator": True}}
    code, e = call("POST", "/v1/setup/estimate", ref)
    assert code == 200 and e["fits"] is True and e["fits_worst"] is False, e
    assert e["total_gb"] == 67.35 and e["available_gb"] == 61.0
    code, e2 = call("POST", "/v1/setup/estimate", {"selection": ref})
    assert code == 200 and e2["total_gb"] == e["total_gb"]
    code, e3 = call("POST", "/v1/setup/estimate", {"model": "qwen3.8-27b"})
    assert code == 200 and e3["total_gb"] == round(67.35 - 3.6, 2), e3["total_gb"]
    for body, status, kind in (
            ({"model": "qwen3.8-27b", "profile": "exl3-9.9"}, 400, "unknown_profile"),
            ({"model": "nope"}, 400, "unknown_model"),
            ({"model": "qwen3.8-27b", "slots": 17}, 422, "out_of_range"),
            ({"model": "qwen3.8-27b", "window": 512}, 422, "out_of_range"),
            ({"model": "qwen3.8-27b", "cmd": "reboot"}, 422, "unknown_field")):
        code, err = call("POST", "/v1/setup/estimate", body)
        assert (code, err["error"]["code"]) == (status, kind), (body, code, err)
        assert err["error"]["message"]
    code, err = call("POST", "/v1/setup/estimate",
                     {"model": "qwen3.8-27b", "components": {"conductor": False,
                                                              "narrator": True}})
    assert code == 422 and err["error"]["code"] == "invalid_combination"
    named = {x["component"] for x in err["error"]["errors"]}
    assert named == {"narrator", "laya", "shadow"}, err
    assert "Conductor" in err["error"]["message"]


ROUTES = (("GET", "/v1/setup/models", None), ("GET", "/v1/setup/models/qwen3.8-27b", None),
          ("GET", "/v1/setup/models/qwen3.8-27b/steps", None), ("GET", "/v1/setup/device", None),
          ("POST", "/v1/setup/estimate", {"model": "qwen3.8-27b"}))


def test_admin_guard():
    _setup(admin=ADMIN, mem={})
    for method, path, body in ROUTES:
        code, _ = call(method, path, body, PROXY)
        assert code == 401, (path, code)
        code, _ = call(method, path, body, dict(PROXY, Authorization=f"Bearer {ADMIN}"))
        assert code == 200, (path, code)
    _setup(admin=None, mem={})
    for method, path, body in ROUTES:
        code, _ = call(method, path, body, PROXY)
        assert code == 200, (path, code)


def test_no_secret_in_any_answer():
    old = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF
    try:
        _setup(admin=ADMIN, mem={"MemTotal": 121.63, "MemAvailable": 50.0})
        texts = []
        for method, path, body in ROUTES:
            code, out = call(method, path, body, {"Authorization": f"Bearer {ADMIN}"})
            assert code == 200
            texts.append(json.dumps(out))
        _, d = call("GET", "/v1/setup/device", None, {"Authorization": f"Bearer {ADMIN}"})
        assert d["hf_token_present"] is True and d["password_set"] is True
        blob = "\n".join(texts)
        for secret in (ADMIN, HF, ADMIN[:12], HF[:12]):
            assert secret not in blob, secret
    finally:
        if old is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = old


def test_device_is_cheap():
    reads = []
    _setup(mem={"MemTotal": 121.63, "MemAvailable": 50.0})
    device_mod.meminfo = lambda: reads.append(1) or {"MemTotal": 121.63, "MemAvailable": 50.0}
    call("GET", "/v1/setup/device")                       # the profiles' folders, once
    ts = []
    t_all = time.monotonic()
    for _ in range(50):
        t0 = time.perf_counter()
        code, _ = call("GET", "/v1/setup/device")
        ts.append(time.perf_counter() - t0)
        assert code == 200
        time.sleep(0.01)
    took = time.monotonic() - t_all
    assert max(ts) < 0.2, max(ts)
    assert len(reads) <= 2 + int(took / device_mod.TTL_S), (len(reads), took)
    print(f"    50 polls, slowest {max(ts) * 1e3:.1f} ms, {len(reads)} memory reads in {took:.2f} s")



def test_cross_site_posts_are_refused():
    _setup(admin=None, mem={})
    body = {"level": 3, "dry_run": True}
    for headers, want in (({"Content-Type": "text/plain"}, (415, "json_only")),
                          ({"Content-Type": "application/x-www-form-urlencoded"}, (415, "json_only")),
                          ({"Content-Type": ""}, (415, "json_only")),
                          ({"Sec-Fetch-Site": "cross-site"}, (403, "cross_site")),
                          ({"Origin": "https://evil.example", "Host": "192.168.1.143:8000"},
                           (403, "cross_origin")),
                          ({"Origin": "null", "Host": "127.0.0.1:8000"}, (403, "cross_origin")),
                          ({"Origin": "http://127.0.0.1:8000.evil.example", "Host": "127.0.0.1:8000"},
                           (403, "cross_origin"))):
        for route in ("/v1/setup/reset", "/v1/setup/apply", "/v1/setup/assistant/load"):
            code, b = call("POST", route, body, headers)
            assert (code, b["error"]["code"]) == want, (route, headers, code, b)
    ok = ({"Origin": "http://192.168.1.143:8000", "Host": "192.168.1.143:8000",
           "Sec-Fetch-Site": "same-origin"},
          {"Origin": "https://tandem.example.lan", "Host": "tandem.example.lan", **PROXY},
          {"Origin": "https://tandem.example.lan", "Host": "127.0.0.1:8000",
           "X-Forwarded-Host": "tandem.example.lan"},
          {"Content-Type": "application/json; charset=utf-8"},
          {})
    for headers in ok:
        code, b = call("POST", "/v1/setup/reset", body, headers)
        assert code == 200 and b["dry_run"] is True, (headers, code, b)
    # a body over the cap is refused before anything reads it
    code, b = call("POST", "/v1/setup/assistant/chat", body, {"Content-Length": str(600 * 1024)})
    assert code == 413 and b["error"]["code"] == "too_large", b
    # GETs are reads and need none of it
    code, _ = call("GET", "/v1/setup/reset", None, {"Sec-Fetch-Site": "cross-site"})
    assert code == 200


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    device_mod.meminfo = _real_meminfo
    print(f"{passed} passed")
