"""The probe stage (ENG-181): a trained probe on Qwen's rows answers the class, journaled.

The test matrix (the router on the checkpoint's template, the engine faked by runners):

    1  off (the seed, and every router stored before the stage): nothing of the probe runs
                                                      test_off_*
    2  on, its file loaded: on the in-context path the probe answers first, on the rows of the
       turn's prompt up to its insertion point; at or above `accept` the question is not asked
                                                      test_on_the_probe_decides_*
    3  below its threshold: the question is asked, both stages journaled  test_below_*
    4  a probe of other weights is not loaded; a probe whose rows the engine does not have is
       an error stage, and the question decides       test_a_probe_of_other_*
    5  shadow: the request's own prompt snapshot answers, into the row's extra.probe, nothing
       else changes; the snapshot is written         test_shadow_*
    6  the router document: mode on without a file is refused; the probe stage validates
       against the journal's contract               test_the_document_*, test_the_contract
    7  one answer under 0.1 ms on the host at the served shape (6 x 5,120 rows, 9 classes)
                                                      test_the_answer_is_cheap

Run: python tests/test_probe_stage.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from server import app  # noqa: E402
from server import probe as probe_mod  # noqa: E402
from server.conductor import inject, schema  # noqa: E402
from test_conductor_router import QTok, body, setup  # noqa: E402
from test_context_stage import FOLLOW, Runner, _on, _session  # noqa: E402

CLASSES = [c["id"] for c in schema.ROUTER["classes"]]
DIM = 4


def _probe_file(root: str, pick: str, hi: float = 6.0, weights: str = "w1",
                rows=("tap1", "post"), dim: int = DIM) -> str:
    """A probe that answers `pick` at a confidence set by `hi` for any row whose first value
    is 1 (the runner's rows below)."""
    D = len(rows) * dim
    W = torch.zeros(len(CLASSES), D)
    W[CLASSES.index(pick), 0] = hi
    path = os.path.join(root, "probes", "class.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"question": "class", "classes": CLASSES, "rows": list(rows), "dim": dim,
                "weights": weights, "temperature": 1.0, "W": W, "b": torch.zeros(len(CLASSES)),
                "mu": torch.zeros(D), "sd": torch.ones(D)}, path)
    return path


class Rows:
    """The probe stage's prefill, faked: the rows of the last position."""

    def __init__(self, names=("tap1", "post")):
        self.names, self.calls = list(names), []

    def __call__(self, ids):
        self.calls.append(list(ids))
        v = torch.zeros(len(self.names), DIM)
        v[0, 0] = 1.0
        return v, self.names, {"reused": 64, "forwarded": len(ids) - 64}


def _with_probe(cond, mode="on", accept=0.9, **kw):
    root = tempfile.mkdtemp(prefix="probe-stage-")
    _probe_file(root, **kw)
    app.STATE["probe_stage"] = probe_mod.ProbeStage(root, weights="w1", hidden=DIM)
    doc = _on(cond)
    doc["classifier"]["probe_stage"] = {"mode": mode, "file": "class.pt", "accept": accept}
    cond.store.put("router", None, doc, note="the probe stage", author="test")
    return root


def _turn(cond, b, runner, rows):
    plan = cond.plan(b, {"x-session-id": "ses-1"}, client_kind="opencode", request_id="req",
                     chat=True, count_tokens=lambda: app.build_prompt(b, count=True))
    prompt, _, in_think = app.build_prompt(b)
    if plan.context is not None:
        cond.decide_in_context(plan, prompt.tolist(), runner, rows)
    out, _ = plan.apply_prompt(prompt, app.STATE["tok"], in_think)
    return plan, out.tolist(), plan.clean(prompt.tolist())


def test_off_runs_nothing_of_the_probe():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    assert schema.probe_stage(cond.store.get("router")[1])["mode"] == "off"
    assert schema.ROUTER["classifier"]["probe_stage"] == {"mode": "off", "file": "", "accept": 0.9}
    _on(cond)
    app.STATE["probe_stage"] = probe_mod.ProbeStage(tempfile.mkdtemp(), weights="w1")
    m2, _, _ = _session(cond, fakes)
    rows = Rows()
    p, _, _ = _turn(cond, body(m2), Runner("code_edit"), rows)
    assert not rows.calls and [s["stage"] for s in p.row["stages"]] == ["qwen_context"]
    # a router stored before the stage existed has no key at all, and is off
    doc = cond.store.get("router")[1]
    doc["classifier"].pop("probe_stage", None)
    assert schema.probe_stage(doc)["mode"] == "off"


def test_on_the_probe_decides_and_no_question_is_asked():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    _with_probe(cond, pick="debug")
    m2, _, _ = _session(cond, fakes)
    runner, rows = Runner("plan"), Rows()
    p, ids, clean = _turn(cond, body(m2), runner, rows)
    assert not runner.calls, "the question was not asked"
    st = p.row["stages"]
    assert [s["stage"] for s in st] == ["probe"] and st[0]["label"] == "debug"
    assert st[0]["confidence"] > 0.9 and st[0]["ms"] < 5 and st[0]["error"] is None
    assert (p.row["class_id"], p.row["source"]) == ("debug", "probe")
    # the rows were read at the turn's insertion point: the prompt up to the last message's last
    # token, which the turn's own prompt shares with it
    got = rows.calls[0]
    tok = QTok()
    at = inject.insertion_point(clean, inject.special_id(tok, inject.IM_END))
    assert tok.decode(got[-len(FOLLOW):], skip_special_tokens=False) == FOLLOW
    assert ids[:len(got)] == got, "the turn's prompt shares every row the probe read"
    assert st[0]["prompt_tokens"] == len(got) and len(got) >= at
    assert st[0]["prefilled"] == len(got) - 64 and st[0]["reused"] == 64


def test_below_the_threshold_the_question_decides():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    _with_probe(cond, pick="debug", hi=0.5)
    m2, _, _ = _session(cond, fakes)
    runner, rows = Runner("plan"), Rows()
    p, _, _ = _turn(cond, body(m2), runner, rows)
    st = p.row["stages"]
    assert [s["stage"] for s in st] == ["probe", "qwen_context"], st
    assert st[0]["below"] and st[0]["label"] == "debug"
    assert len(runner.calls) == 1 and (p.row["class_id"], p.row["source"]) == ("plan",
                                                                                "qwen_context")


def test_a_probe_of_other_weights_or_rows_does_not_decide():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    _with_probe(cond, pick="debug", weights="w2")
    m2, _, _ = _session(cond, fakes)
    rows = Rows()
    p, _, _ = _turn(cond, body(m2), Runner("plan"), rows)
    assert not rows.calls and [s["stage"] for s in p.row["stages"]] == ["qwen_context"]
    st = probe_mod.ProbeStage(tempfile.mkdtemp(), weights="w1")
    assert st.get({"file": "nothing.pt"}) == (None, "missing")
    # loaded, but the engine's rows lack one it reads: an error stage, then the question
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    _with_probe(cond, pick="debug", rows=("tap5", "post"))
    m2, _, _ = _session(cond, fakes)
    p, _, _ = _turn(cond, body(m2), Runner("plan"), Rows())
    st = p.row["stages"]
    assert st[0]["error"] == "probe_rows" and st[1]["stage"] == "qwen_context"
    assert p.row["source"] == "qwen_context"


def test_shadow_journals_beside_the_router_and_writes_the_snapshot():
    cond, fakes = setup(decider=lambda t, o: ("code_edit", 0.93))
    root = _with_probe(cond, mode="shadow", pick="debug")
    m2, _, _ = _session(cond, fakes)
    rows = Rows()
    p, _, _ = _turn(cond, body(m2), Runner("plan"), rows)
    assert not rows.calls and p.row["source"] == "qwen_context", "shadow decides nothing"
    ring = probe_mod.Ring(root, hidden=DIM, every=8, weights="w1", slots=4)
    app.STATE["probe"] = ring
    cap = ring.capture(5)

    class Src:
        hidden_post_norm = torch.zeros(1, 1, DIM)

        def tapped(self):
            r = torch.zeros(1, DIM)
            r[0, 0] = 1.0
            return [1], "output", [r]
    s = Src()
    cap._take("prompt", s, s, slice(0, 1), at=5)

    class Rec:
        probe, plan, ts, request_id, client_kind = cap, p, time.time(), "chatcmpl-x", "opencode"
        finish_reason, tool_calls, reasoning_tokens, prompt_tokens = "tool_calls", 1, 0, 300
    app._probe_done(Rec())
    ex = p.row["extra"]["probe"]
    assert ex["stage"] == "probe" and ex["label"] == "debug" and ex["reason"] == "shadow"
    assert p.row["class_id"] == "plan", "the router's class stands"
    from tools import probe_train
    recs = probe_train.read_index(root)
    assert len(recs) == 1 and recs[0]["row_id"] == p.row["id"] and recs[0]["class_id"] == "plan"
    assert ring.info()["slots_free"] == 4


def test_the_document_refuses_on_without_a_file():
    cond, _ = setup()
    doc = json.loads(json.dumps(cond.store.get("router")[1]))
    doc["classifier"]["probe_stage"] = {"mode": "on", "file": "", "accept": 0.9}
    errs = schema.check("router", doc, packages={"packages": {}})
    assert any("probe_stage.file" in e for e in errs), errs
    doc["classifier"]["probe_stage"] = {"mode": "sideways"}
    assert schema.check("router", doc, packages={"packages": {}})


def test_the_contract():
    with open(os.path.join(ROOT, "docs", "contract", "conductor-v1",
                           "decisions.schema.json")) as f:
        c = json.load(f)
    item = c["$defs"]["decision"]["properties"]["stages"]["items"]
    stage = probe_mod.ProbeStage(tempfile.mkdtemp(), weights="w1").stage(
        {"file": "x.pt", "accept": 0.9}, None, [], reason="followup")
    stage.update(prompt_tokens=10, prefilled=3, reused=7, stage_ms=1.5)
    assert schema.validate(stage, item) == [], schema.validate(stage, item)
    assert "probe" in c["$defs"]["decision"]["properties"]["source"]["enum"]
    with open(os.path.join(ROOT, "docs", "contract", "conductor-v1", "seeds.json")) as f:
        seeds = json.load(f)
    assert seeds["router"]["classifier"]["probe_stage"]["mode"] == "off"


def test_the_answer_is_cheap():
    root = tempfile.mkdtemp(prefix="probe-cost-")
    names = ["tap5", "tap19", "tap33", "tap47", "tap61", "post"]
    _probe_file(root, "debug", rows=names, dim=5120)
    p = probe_mod.Probe.load(os.path.join(root, "probes", "class.pt"))
    vec = torch.randn(6, 5120)
    for _ in range(20):
        p.answer(vec, names)
    ms = [p.answer(vec, names)[3] for _ in range(300)]
    med = statistics.median(ms)
    assert med < 0.1, f"{med:.4f} ms"
    return f"median {med * 1e3:.1f} us, p90 {sorted(ms)[270] * 1e3:.1f} us"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            got = fn()
            print(f"  {name:62s} ok" + (f"  ({got})" if isinstance(got, str) else ""))
            passed += 1
    print(f"{passed} passed")
