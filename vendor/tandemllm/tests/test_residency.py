"""OPS-41 / OPS-42: the residency planner, the roles' load order in ops/, and the memory split.

The test matrix is OPS-41's Gherkin (OPS-42's table):

  1  the split is measured           on the box, once (Memo note); here the parts it is made of:
                                     test_split_* (the load ledger's stages, every part read off
                                     the object that holds it)
  2  a role that fits loads          test_2_* (unit: 31 - 5.1 >= 15 + 6; e2e: ops/roles.sh with a
                                     fake `free` starts it, journals it and records what it took)
  3  the floor refuses               test_3_* (18 GiB and a 12 GiB role: refused_memory, nothing
                                     started)
  4  no concurrent loads             test_4_* (a load in progress, a loading engine)
  5  never during a hold             test_5_* (the pause file; the box lock where flock exists)

plus the vLLM rule (BF16, KV unquantised, the planner's fraction), the Mac rule, the settle time
after a stop, the document's validation and versions, and the plan endpoint's journal.

Run: python tests/test_residency.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from server import memsplit  # noqa: E402
from server.conductor import residency, schema  # noqa: E402
from server.conductor.store import ConfigError, Store  # noqa: E402
from test_ops_scripts import _box, _env, _proc, _engine_argv  # noqa: E402

DOC = schema.seed("residency")
ARBITER = dict(DOC, claims=dict(DOC["claims"], decider={"gib": 5.1, "basis": "declared",
                                                        "note": "the Gherkin's number"}))


# ------------------------------------------------------------------ 2, 3, 4, 5: the planner

def test_2_a_role_that_fits_is_allowed():
    """MemAvailable 31 GB and arbiter's 5.1: 31 - 5.1 >= 15 + 6, allowed."""
    d = residency.check("decider", 31.0, ARBITER)
    assert d["allowed"] and d["reason"] is None, d
    assert (d["after_gib"], d["need_gib"]) == (25.9, 21.0), d
    # the shipped claim, Laya inside the engine (OPS-69): 3.0 GiB of MemAvailable
    assert residency.check("decider", 31.0, DOC)["after_gib"] == 28.0


def test_3_a_role_that_would_break_the_floor_is_refused():
    d = residency.check("reader", 18.0, DOC)
    assert not d["allowed"] and d["reason"] == "refused_memory", d
    assert d["claim_gib"] == 12.0 and d["after_gib"] == 6.0
    edge = residency.check("decider", 21.0 + 3.0, DOC)
    assert edge["allowed"], "the floor plus the margin is allowed, not one byte under"
    assert not residency.check("decider", 21.0 + 2.99, DOC)["allowed"]
    assert residency.check("decider", None, DOC)["reason"] == "refused_memory", \
        "no reading is no load"


def test_4_no_concurrent_loads():
    d = residency.check("decider", 60.0, DOC, loading=True)
    assert d["reason"] == "refused_load_in_progress" and not d["allowed"]


def test_5_never_during_a_hold_and_the_hold_comes_first():
    d = residency.check("decider", 60.0, DOC, hold=True, loading=True)
    assert d["reason"] == "refused_hold", d
    assert residency.check("nemotron", 60.0, DOC)["reason"] == "refused_unknown_role"


def test_the_plan_loads_one_after_the_other():
    """At 40 GiB the decider fits (40 - 3 = 37 >= 21), the narrator (P1) then sees 37 and
    fits too (37 - 6 = 31); the reader sees 31, and 31 - 12 = 19 is under 21: refused.
    The order is the document's."""
    p = residency.plan(40.0, DOC)
    assert p["order"] == ["decider", "narrator", "reader"] and p["spendable_gib"] == 19.0, p
    assert [(r["role"], r["allowed"], r["mem_available_gib"]) for r in p["rows"]] == \
        [("decider", True, 40.0), ("narrator", True, 37.0), ("reader", False, 31.0)], p["rows"]
    assert p["rows"][2]["reason"] == "refused_memory"
    roles = {"librarian": {"endpoint": "http://x", "memory_claim_gib": 1.5},
             "classifier": {"endpoint": "self", "memory_claim_gib": 3.0}}
    assert [r["role"] for r in residency.plan(60.0, DOC, roles)["rows"]] == \
        ["decider", "narrator", "librarian", "reader"], \
        "a role the roles document claims joins the plan"


def test_the_plan_leaves_out_a_role_switched_off():
    """SRV-534: a role the roles document switches off (the narrator and the reader by default)
    never loads, so the plan holds no claim for it; asked by name it is still checked."""
    roles = schema.seed("roles")
    assert roles["narrator"]["mode"] == "off" and roles["reader"]["mode"] == "off"
    p = residency.plan(40.0, DOC, roles)
    assert p["order"] == ["decider"], p["order"]
    roles["narrator"] = dict(roles["narrator"], mode="on")
    assert residency.plan(40.0, DOC, roles)["order"] == ["decider", "narrator"]
    assert [r["role"] for r in residency.plan(40.0, DOC, schema.seed("roles"),
                                              ["narrator"])["rows"]] == ["narrator"]


def test_laya_fits_beside_super_vl_at_three_slots():
    """SRV-534: the reference box's check of 2026-10-17 (Super VL EXL3 4.0, 3 slots of 262,144):
    43.75 GiB available, the engine at 68.79 of its 86.25 GiB peak. The shipped claim fits with
    2.29 GiB to spare; the arbiter's 8.2 refused Laya on a new box; the 1,048,576-token window
    (98.08 held, past its 94.82 peak, 9.82 available) refuses it either way."""
    d = residency.check("decider", 43.75, DOC, engine_peak_gib=86.25, engine_now_gib=68.79)
    assert d["allowed"] and (d["engine_growth_gib"], d["after_gib"]) == (17.46, 23.29), d
    old = dict(DOC, claims=dict(DOC["claims"], decider=dict(DOC["claims"]["decider"], gib=8.2)))
    d = residency.check("decider", 43.75, old, engine_peak_gib=86.25, engine_now_gib=68.79)
    assert not d["allowed"] and d["after_gib"] == 18.09, d
    d = residency.check("decider", 9.82, DOC, engine_peak_gib=94.82, engine_now_gib=98.08)
    assert d["reason"] == "refused_memory" and d["after_gib"] == 6.82, d


def test_the_vllm_fraction_is_the_byte_math_rounded_down():
    assert residency.vllm_fraction("reader", 121.6, DOC) == 0.09        # 12 / 121.6 = 0.0987
    assert residency.vllm_fraction("decider", 121.6, DOC) == 0.02        # 3 / 121.6 = 0.0247


def test_the_journal_keeps_the_last_refusal():
    root = tempfile.mkdtemp(prefix="qse-res-")
    j = residency.Journal(root)
    j.write(dict(residency.check("decider", 40.0, DOC), kind="check", source="t"))
    j.write(dict(residency.check("reader", 18.0, DOC), kind="check", source="t"))
    j.write({"kind": "loaded", "role": "decider", "before_gib": 40.0, "after_gib": 31.9,
             "took_gib": 8.1})
    assert j.last_refusal()["role"] == "reader" and j.last_refusal()["reason"] == "refused_memory"
    assert [r.get("kind") for r in j.tail(3)] == ["loaded", "check", "check"]


def test_the_document_validates_versions_and_restores():
    assert schema.check("residency", DOC) == []
    bad = dict(DOC, order=["decider", "decider"])
    assert schema.check("residency", bad) == ["$.order: a role is listed twice"]
    assert schema.check("residency", dict(DOC, floor_gib=-1))
    claims = dict(DOC["claims"], decider=dict(DOC["claims"]["decider"], basis="guess"))
    assert schema.check("residency", dict(DOC, claims=claims))
    st = Store(tempfile.mkdtemp(prefix="qse-res-store-"))
    assert st.get("residency") == (0, DOC)
    v = st.put("residency", None, dict(DOC, floor_gib=16.0), note="floor 16", author="t")
    assert st.get("residency")[1]["floor_gib"] == 16.0 and v == 1
    try:
        st.put("residency", None, bad, note="x", author="t")
        raise AssertionError("an invalid document is refused")
    except ConfigError as exc:
        assert exc.status == 422
    st.restore("residency", None, 0, author="t")
    assert st.get("residency")[1]["floor_gib"] == 15.0


def test_the_command_line_reads_the_store_and_journals():
    root = tempfile.mkdtemp(prefix="qse-res-cli-")
    Store(root).put("residency", None, dict(DOC, margin_gib=10.0), note="m10", author="t")
    env = dict(os.environ, PYTHONPATH=ROOT)
    run = lambda *a: subprocess.run([sys.executable, "-m", "server.conductor.residency",  # noqa: E731
                                     "--root", root, *a], capture_output=True, text=True,
                                    env=env, cwd=ROOT)
    r = run("check", "decider", "--mem-available-gib", "27")
    assert r.returncode == 3, r.stdout + r.stderr          # 27 - 3 = 24 < 15 + 10
    assert json.loads(r.stdout)["reason"] == "refused_memory"
    assert run("check", "decider", "--mem-available-gib", "40").returncode == 0
    assert json.loads(run("plan", "--mem-available-gib", "40").stdout)["need_gib"] == 25.0
    assert run("fraction", "reader", "--total-gib", "121.6").stdout.strip() == "0.09"
    with open(os.path.join(root, "residency.jsonl")) as fh:
        lines = [json.loads(x) for x in fh]
    assert [x["allowed"] for x in lines] == [False, True] and lines[0]["source"] == "cli"


# ------------------------------------------------------------------ 1: the split's parts

def test_split_the_load_ledger_records_each_stage():
    reads = iter([(0, 0, 100), (16 << 30, 16 << 30, 84), (40 << 30, 41 << 30, 59)])
    cur = {}

    def probe():
        a, r, m = next(reads)
        cur["m"] = m
        return a, r
    led = memsplit.LoadLedger(probe=probe, avail=lambda: cur["m"] << 30, clock=lambda: 0.0)
    for st in ("start", "weights", "engine"):
        led.mark(st)
    got = led.stages()
    assert [(x["stage"], x["allocated"] >> 30, x["reserved"] >> 30,
             x["mem_available_drop"] >> 30) for x in got] == \
        [("weights", 16, 16, 16), ("engine", 24, 25, 25)], got


class _T:
    """A tensor's two numbers (memsplit reads only numel and element_size)."""

    def __init__(self, n, size=2):
        self.n, self.size = n, size

    def numel(self):
        return self.n

    def element_size(self):
        return self.size


def test_split_reads_every_part_off_its_owner():
    import types
    arm = types.new_class("Arm", exec_body=lambda ns: ns.update(__module__="engine.drafters.x"))
    rt = types.new_class("Router", exec_body=lambda ns: ns.update(__module__="engine.lenrouter"))
    small, large, router = arm(), arm(), rt()
    small._ck, small._cv = _T(1000), _T(1000)
    large._ck, large._cv = _T(2000), _T(2000)
    router.small, router.large, router.arms = small, large, [small, large]
    eng = types.SimpleNamespace(
        w=types.SimpleNamespace(bytes_fp8=12 << 30, bytes_other=3 << 30),
        kv=types.SimpleNamespace(k=_T(8 << 30, 1), v=_T(8 << 30, 1), ks=None, vs=None),
        state=types.SimpleNamespace(S=_T(100, 4), conv=_T(50), _spare=None))
    _the_27b_verbs(eng)
    res = types.SimpleNamespace(report=lambda: {"bytes": 4 << 30})
    state = {"engine": eng, "drafter": router, "vision": types.SimpleNamespace(nbytes=920_000_000),
             "state_store": types.SimpleNamespace(bytes=1 << 30), "resident": res,
             "max_len": 262144, "load_ledger": None}
    out = memsplit.split(state, probe=lambda: (40 << 30, 42 << 30), avail=lambda: 30 << 30)
    p = out["parts"]
    assert p["weights"] == 15 << 30 and p["kv_pool"] == 16 << 30
    assert p["gdn_state"] == 500 and p["drafter_kv"] == 12_000 and out["drafter_caches"] == 2
    assert p["vision"] == 920_000_000 and p["state_store"] == 1 << 30
    assert p["resident_prefix"] == 4 << 30
    assert out["reserve"] == 2 << 30 and out["other"] == (40 << 30) - sum(p.values())
    # and validates as the System view's `memory.split`
    from tools import contract_check
    assert contract_check.validate(json.loads(json.dumps(out)), _split_schema()) == []


def _the_27b_verbs(eng):
    """The 27B adapter's split verbs on a stand-in engine (engine/models/qwen3_5/adapter.py)."""
    from engine.models.qwen3_5 import adapter
    eng.memory_parts = types.MethodType(adapter.memory_parts, eng)
    eng.resident_files = types.MethodType(adapter.resident_files, eng)
    return eng


def _split_schema():
    with open(os.path.join(ROOT, "docs", "contract", "dashboard-v1", "system.schema.json")) as fh:
        return json.load(fh)["properties"]["memory"]["properties"]["split"]


# ------------------------------------------------------------------ the ops scripts, e2e

FREE = """#!/bin/bash
# `free -m`, the numbers from the test: MemTotal 124546 MiB, MemAvailable FAKE_AVAIL_MIB
echo "               total        used        free      shared  buff/cache   available"
echo "Mem:          124546       87096       11127          56       28275  ${FAKE_AVAIL_MIB:-31744}"
echo "Swap:          16383         350       16033"
"""


def _roles_box(extra_env: str = "") -> tuple[str, str]:
    repo = _box(["hold.sh"])
    root = os.path.join(repo, "conductor")
    with open(os.path.join(repo, "ops", "serve.env"), "a") as fh:
        fh.write(f"PY={sys.executable}\nPYTHONPATH={ROOT}\n{extra_env}")
    binary = os.path.join(repo, "bin")
    fakes = {"free": FREE}
    if shutil.which("setsid") is None:                     # the Mac: no util-linux
        fakes["setsid"] = '#!/bin/bash\nexec "$@"\n'
    for name, body in fakes.items():
        with open(os.path.join(binary, name), "w") as fh:
            fh.write(body)
        os.chmod(os.path.join(binary, name), 0o755)
    return repo, root


def _roles(repo, root, *args, **env):
    env.setdefault("ROLES_ALLOW_DARWIN", "1")
    return subprocess.run(["bash", os.path.join(repo, "ops", "roles.sh"), *args],
                          capture_output=True, text=True,
                          env=_env(repo, QSE_CONDUCTOR_ROOT=root, **env), timeout=60)


def _journal(root):
    with open(os.path.join(root, "residency.jsonl")) as fh:
        return [json.loads(x) for x in fh]


def test_2_ops_loads_a_role_that_fits_and_records_what_it_took():
    repo, root = _roles_box()
    started = os.path.join(repo, "started")
    r = _roles(repo, root, "load", "decider", "--", "bash", "-c",
               f"touch {started}; sleep 30", FAKE_AVAIL_MIB=str(31 * 1024))
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        for _ in range(50):
            if os.path.exists(started):
                break
            time.sleep(0.1)
        assert os.path.exists(started), "the role's command ran"
        assert os.path.exists(os.path.join(repo, "logs", "roles", "decider.pid"))
        j = _journal(root)
        assert j[0]["kind"] == "check" and j[0]["allowed"] and j[0]["source"] == "ops", j
        assert j[0]["after_gib"] == 28.0 and j[1]["kind"] == "loaded", j
        assert not os.path.exists(os.path.join(repo, "logs", "roles", ".loading")), \
            "the load lock is released"
    finally:
        d = _roles(repo, root, "down")
    assert "stopped decider" in d.stdout, d.stdout + d.stderr
    assert not os.path.exists(os.path.join(repo, "logs", "roles", "decider.pid"))


def test_3_ops_refuses_a_role_under_the_floor_and_starts_nothing():
    repo, root = _roles_box()
    started = os.path.join(repo, "started")
    r = _roles(repo, root, "load", "reader", "--", "touch", started,
               FAKE_AVAIL_MIB=str(18 * 1024))
    assert r.returncode == 3, r.stdout + r.stderr
    time.sleep(0.3)
    assert not os.path.exists(started), "nothing loads"
    assert _journal(root)[-1]["reason"] == "refused_memory"


HEALTH_CURL = """#!/bin/bash
# /health answers FAKE_HEALTH_JSON when the test sets it; nothing else answers
case " $* " in
  *"/health"*) [ -n "${FAKE_HEALTH_JSON:-}" ] && { printf '%s' "$FAKE_HEALTH_JSON"; exit 0; } ;;
esac
exit 22
"""


def test_8_ops_counts_the_engines_growth_to_its_declared_peak():
    """OPS-47: the planner subtracts what the engine may still take (MEM_PEAK_GIB less what it
    holds now, from its /health) before the claim. 37.9 free with the engine at 70 of 77: a 12 GiB
    reader is refused (18.9 left), where today's free memory alone admitted it (25.9)."""
    repo, root = _roles_box("MEM_PEAK_GIB=77\n")
    curl = os.path.join(repo, "bin", "curl")
    with open(curl, "w") as fh:
        fh.write(HEALTH_CURL)
    os.chmod(curl, 0o755)
    started = os.path.join(repo, "started")
    health = json.dumps({"status": "ok", "memory": {"peak_gb": 77.0, "engine_reserved_gb": 70.0}})
    r = _roles(repo, root, "load", "reader", "--", "touch", started,
               FAKE_AVAIL_MIB=str(int(37.9 * 1024)), FAKE_HEALTH_JSON=health)
    assert r.returncode == 3, r.stdout + r.stderr
    j = _journal(root)[-1]
    assert (j["reason"], j["engine_growth_gib"], j["after_gib"]) == ("refused_memory", 7.0, 18.9), j
    assert not os.path.exists(started)
    # no answer from /health: the whole declared peak is still to come
    r = _roles(repo, root, "load", "decider", "--", "touch", started,
               FAKE_AVAIL_MIB=str(100 * 1024))
    j = _journal(root)[-1]
    assert r.returncode == 3 and j["engine_growth_gib"] == 77.0, (r.stdout, j)


def test_4_ops_refuses_a_second_load_and_a_loading_engine():
    repo, root = _roles_box()
    lock = os.path.join(repo, "logs", "roles", ".loading")
    os.makedirs(lock)
    with open(os.path.join(lock, "owner"), "w") as fh:
        fh.write(f"{os.getpid()} reader\n")                   # a live loader
    r = _roles(repo, root, "load", "decider", "--", "true", FAKE_AVAIL_MIB=str(60 * 1024))
    assert r.returncode == 3 and _journal(root)[-1]["reason"] == "refused_load_in_progress", \
        r.stdout + r.stderr
    assert os.path.isdir(lock), "another loader's lock is not taken"
    # a dead loader's lock is taken over
    with open(os.path.join(lock, "owner"), "w") as fh:
        fh.write("999999 reader\n")
    assert _roles(repo, root, "load", "decider", "--", "true",
                  FAKE_AVAIL_MIB=str(60 * 1024)).returncode == 0
    # an engine that is loading (a process, no /health) is a load in progress too
    _proc(repo, 4242, _engine_argv())
    r = _roles(repo, root, "load", "decider", "--", "true", FAKE_AVAIL_MIB=str(60 * 1024))
    assert r.returncode == 3 and _journal(root)[-1]["reason"] == "refused_load_in_progress"
    r = _roles(repo, root, "load", "decider", "--", "true", FAKE_AVAIL_MIB=str(60 * 1024),
               FAKE_HEALTHY=1)
    assert r.returncode == 0, "a healthy engine is not a load"


def test_5_ops_never_loads_during_a_hold():
    repo, root = _roles_box()
    pause = os.path.join(repo, ".watchdog.off")
    open(pause, "w").close()
    r = _roles(repo, root, "load", "decider", "--", "true", FAKE_AVAIL_MIB=str(60 * 1024))
    assert r.returncode == 3 and _journal(root)[-1]["reason"] == "refused_hold", r.stdout
    r = _roles(repo, root, "load", "decider", "--", "true", FAKE_AVAIL_MIB=str(60 * 1024),
               HOLD_RESTART=1)
    assert r.returncode == 0, "the hold's own restart brings the roles back"
    os.remove(pause)
    if shutil.which("flock") is None:
        print("    (the box lock half skipped: no flock here -- it runs on the box)")
        return
    lock = os.path.join(repo, "box.flock")
    open(lock, "w").close()
    holder = subprocess.Popen(["flock", lock, "sleep", "20"])
    try:
        time.sleep(0.5)
        r = _roles(repo, root, "load", "decider", "--", "true", FAKE_AVAIL_MIB=str(60 * 1024))
        assert r.returncode == 3 and _journal(root)[-1]["reason"] == "refused_hold", r.stdout
    finally:
        holder.kill()
        holder.wait()


def test_a_vllm_role_runs_bf16_with_the_planners_fraction():
    repo, root = _roles_box("ROLE_READER_RUNTIME=vllm\n")
    out = os.path.join(repo, "argv")
    r = _roles(repo, root, "load", "reader", "--", "bash", "-c", f'echo "$@" > {out}', "vllm",
               "serve", "nemotron", FAKE_AVAIL_MIB=str(60 * 1024))
    assert r.returncode == 0, r.stdout + r.stderr
    for _ in range(50):
        if os.path.exists(out) and open(out).read():
            break
        time.sleep(0.1)
    # 12 GiB of 124,546 MiB (121.6 GiB): 0.09, never vLLM's 0.9
    assert open(out).read().split() == ["serve", "nemotron", "--dtype", "bfloat16",
                                        "--kv-cache-dtype", "auto",
                                        "--gpu-memory-utilization", "0.09"]
    for bad in (["--dtype", "float16"], ["--quantization", "fp8"], ["--kv-cache-dtype", "fp8"],
                ["--gpu-memory-utilization", "0.9"]):
        r = _roles(repo, root, "load", "reader", "--", "vllm", "serve", "x", *bad,
                   FAKE_AVAIL_MIB=str(60 * 1024))
        assert r.returncode == 5 and "REFUSING" in r.stdout, (bad, r.stdout)


def test_roles_never_load_on_the_mac():
    repo, root = _roles_box()
    with open(os.path.join(repo, "bin", "uname"), "w") as fh:
        fh.write("#!/bin/bash\necho Darwin\n")
    os.chmod(os.path.join(repo, "bin", "uname"), 0o755)
    r = _roles(repo, root, "load", "decider", "--", "true", FAKE_AVAIL_MIB=str(60 * 1024),
               ROLES_ALLOW_DARWIN="0")
    assert r.returncode == 4 and "never load on the Mac" in r.stdout, r.stdout


def test_up_loads_the_autoload_roles_in_the_planners_order():
    marks = "{repo}/order"
    repo, root = _roles_box()
    with open(os.path.join(repo, "ops", "serve.env"), "a") as fh:
        fh.write(f'ROLES_AUTOLOAD="reader decider"\n'
                 f'ROLE_DECIDER_START="bash -c \'echo decider >> {marks.format(repo=repo)}\'"\n'
                 f'ROLE_READER_START="bash -c \'echo reader >> {marks.format(repo=repo)}\'"\n')
    r = _roles(repo, root, "up", FAKE_AVAIL_MIB=str(60 * 1024))
    assert r.returncode == 0, r.stdout + r.stderr
    time.sleep(0.5)
    assert open(marks.format(repo=repo)).read().split() == ["decider", "reader"], \
        "the document's order, not the list's"
    none = _roles(*_roles_box(), "up")
    assert none.returncode == 0 and "none to load" in none.stdout


def test_the_hold_settles_before_the_restart_and_stops_the_roles_first():
    """After stop.sh the hold waits the settle time before start.sh (the 2026-09-11 wedge), and
    it stops the roles before the engine."""
    repo, root = _roles_box()
    pidf = os.path.join(repo, "logs", "roles", "decider.pid")
    os.makedirs(os.path.dirname(pidf), exist_ok=True)
    sleeper = subprocess.Popen(["sleep", "30"])
    with open(pidf, "w") as fh:
        fh.write(str(sleeper.pid))
    t0 = time.time()
    r = subprocess.run(["bash", os.path.join(repo, "ops", "hold.sh"), "1", "--", "true"],
                       capture_output=True, text=True,
                       env=_env(repo, QSE_CONDUCTOR_ROOT=root, RESTART_SETTLE_S="2"))
    took = time.time() - t0
    sleeper.wait(timeout=5)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "stopped decider" in r.stdout and r.stdout.index("stopped decider") < \
        r.stdout.index("running:"), r.stdout
    assert took >= 2.0, f"no settle: {took:.2f} s"
    assert [m.split()[0] for m in open(os.path.join(repo, "markers")).read().split("\n") if m] \
        == ["stop.sh", "start.sh"]


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
