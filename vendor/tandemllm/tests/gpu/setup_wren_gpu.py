"""On the board: Wren, the setup guide's assistant (server/setup/wren.py), loaded, asked and unloaded
on the GPU, alone or borrowing the narrator's copy. No model of Qwen's in this process.

  own     Wren as setup mode loads it: its own Qwen3.5-2B at 8,192 rows through the planner. The
          load's time and MemAvailable drop, three turns (a question about a tool, "switch the
          narrator off", "use two slots") with the time to the first text and the tool calls
          the server checked, then the unload and MemAvailable back (within 0.3 GiB).
  borrow  the narrator's slot as the server builds it (Qwen3.5-2B, 4,096 rows, its own pool),
          one narrator line N0; Wren borrows the loaded slot (MemAvailable must not move by more
          than 0.3 GiB), one turn; the narrator's line again must equal N0 to the token.

  cases   VIS-91: Wren at the borrow's window (4,096 rows) on its own copy, on the case set below:
          ten requests for a change and five questions, among them the two of the live report
          ("What is the box running now?", then "Which one is for coding tasks?" on the Model
          step). A request passes when the right call comes and no other change; a question
          passes when words come, no change, and the words name what the table says. After a
          change the page's second round (the tool's result back) must end in words.
          QSE_WREN_MODEL=qwen3.5-4b measures the 4B on the same set.

Run on the box while :8000 has been idle for at least 10 minutes, under the box lock, with the
served environment and a private lock path for the planner (the real one is the test's own):
    flock -n ~/.qwen38-box.flock env BOX_LOCK=/tmp/wren-gpu.flock \\
        bash -c 'set -a; . ~/tandem-live/ops/serve.env; set +a; \\
                 $PY tests/gpu/setup_wren_gpu.py own borrow'
It prints one JSON line per part.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from server import app  # noqa: E402
from server.setup import wren  # noqa: E402

GIB = 1 << 30
PAGE = {"model": "qwen3.8-27b", "step": "components", "step_index": 3,
        "selection": {"profile": "exl3-4.5", "slots": 4,
                      "components": {"narrator": True, "shadow": True, "laya": True}}}


def avail() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / GIB
    raise RuntimeError("no MemAvailable")


def settle() -> float:
    import torch
    torch.cuda.synchronize()
    time.sleep(1.0)
    return avail()


def turn(w, text: str, page: dict) -> dict:
    t0 = time.perf_counter()
    first, said, calls, events = None, "", [], []
    for ev in w.chat({"messages": [{"role": "user", "content": text}], "page": page}):
        events.append(ev["type"])
        if ev["type"] == "text":
            first = first or time.perf_counter()
            said += ev["delta"]
        elif ev["type"] == "tool_call":
            calls.append({"name": ev["name"], "args": ev["args"]})
        elif ev["type"] in ("done", "error"):
            end = ev
    took = time.perf_counter() - t0
    return {"question": text, "first_text_s": round(first - t0, 2) if first else None,
            "s": round(took, 2), "text": said, "tool_calls": calls, "end": end}


def own() -> dict:
    app.STATE.clear()
    w = wren.Wren(app, device_facts=lambda: {"mem_available_gb": round(avail(), 1)},
                  setup_mode=True, log=lambda s: print(s, flush=True))
    app.STATE["wren"] = w
    before = settle()
    t0 = time.perf_counter()
    st = w.load()
    while w.state == "loading":
        time.sleep(0.2)
    load_s = time.perf_counter() - t0
    after = settle()
    st = w.status()
    if st["state"] != "loaded":
        return {"part": "own", "state": st}
    turns = [turn(w, "What does the shadow reader do, and how much memory does it take?", PAGE),
             turn(w, "Please switch the narrator off.", PAGE),
             turn(w, "Use two slots.", dict(PAGE, step="parallelism", step_index=4))]
    w.unload("asked")
    back = settle()
    # a second load and unload: what the first left behind is the process's once (the CUDA
    # context's modules and workspaces), what the second leaves would be a leak
    w.load()
    while w.state == "loading":
        time.sleep(0.2)
    again = settle()
    w.unload("asked")
    back2 = settle()
    return {"part": "own", "model": st["model"], "window": st["window"],
            "load_s": round(load_s, 1), "slot_load_s": st["load_s"],
            "device_gib": st["device_gib"], "mem_available_before": round(before, 2),
            "mem_available_loaded": round(after, 2), "took_gib": round(before - after, 2),
            "mem_available_unloaded": round(back, 2),
            "second_load_took_gib": round(back - again, 2),
            "mem_available_unloaded_again": round(back2, 2),
            "second_cycle_back_within_0_3": abs(back2 - back) <= 0.3,
            "turns": turns, "stats": dict(w.stats)}


def borrow() -> dict:
    from engine.decision.device import Device
    from engine.generative.qwen35 import Qwen35Model
    from engine.generative.slot import Slot
    from server.conductor import narrator as narrator_mod
    app.STATE.clear()
    name, folder, _ = wren.choose()
    before = settle()
    slot = Slot(Qwen35Model(name, folder), role="narrator", gate=app.LOCK, planner=None,
                device=Device("auto"), log=lambda s: print(s, flush=True))
    slot.load(wait=True)
    narr_loaded = settle()
    msgs = narrator_mod.messages({"kind": "route", "facts": {"cls": "code_edit",
                                                             "effort": "medium effort"}})
    n0 = slot.run(msgs, max_new=24, budget_ms=10_000, held=False)
    app.STATE["narrator"] = types.SimpleNamespace(slot=slot)
    w = wren.Wren(app, device_facts=lambda: {"mem_available_gb": round(avail(), 1)},
                  log=lambda s: print(s, flush=True))
    app.STATE["wren"] = w
    st = w.load()
    borrowed = settle()
    t = turn(w, "What does the narrator do?", PAGE)
    n1 = slot.run(msgs, max_new=24, budget_ms=10_000, held=False)
    w.unload("asked")
    still = slot.state
    slot.unload()
    return {"part": "borrow", "model": name, "borrowed": st["borrowed"], "window": st["window"],
            "claim_gib": st["claim_gib"], "narrator_took_gib": round(before - narr_loaded, 2),
            "borrow_took_gib": round(narr_loaded - borrowed, 2), "turn": t,
            "narrator_line_before": n0.get("text"), "narrator_line_after": n1.get("text"),
            "narrator_same": n0.get("text") == n1.get("text") and bool(n0.get("text")),
            "narrator_still_loaded_after_wren_unload": still == "loaded"}


def page_for(step: str) -> dict:
    """The page state the browser sends on `step` (checks.ts buildPageState), from the defaults."""
    from server.setup import estimate, registry
    m = registry.get("qwen3.8-27b")
    sel = estimate.defaults(m)
    steps = {s["id"]: s for s in registry.wizard_fields(m)}
    index = list(steps).index(step)
    def value(f):
        if step == "weights":
            return sel["profile"]
        if step == "components":
            return sel["components"].get(f)
        if step == "parallelism":
            return sel.get(f)
        return (sel.get("methods") or {}).get(f, (sel.get("advanced") or {}).get(f))
    fields = [{"field": f["field"], "label": f["label"], "type": f["type"], "value": value(f["field"])}
              for f in steps[step]["fields"]]
    titles = dict(registry.STEPS)
    return {"model": m["id"], "step": step, "step_index": index,
            "selection": {k: sel.get(k) for k in ("model", "profile", "components", "slots",
                                                  "window", "side_window", "kv_pool", "floor_gb")},
            "visible": {"title": titles[step], **({"fields": fields} if fields else {})}}


# (step, what the user says, the call it wants or None for a question, words a question's answer
# must hold: any one of them)
CASES = [
    ("overview", "What is the box running now?", None, ("exl3-6.0",)),
    ("overview", "Which one is for coding tasks?", None, ("exl3-2.5", "124")),
    ("weights", "Which profile is closest to the original model?", None, ("exl3-6.0",)),
    ("components", "What does the shadow reader do?", None, ("shadow",)),
    ("parallelism", "How much memory does one slot take?", None, ("3.26", "104.25", "26.06")),
    ("components", "Turn the narrator off.", ("toggle", {"component": "narrator", "on": False}), ()),
    ("components", "Switch vision off, please.", ("toggle", {"component": "vision", "on": False}), ()),
    ("parallelism", "Use two slots.", ("set_field", {"step": "parallelism", "field": "slots",
                                                     "value": 2}), ()),
    ("parallelism", "Set the window to 65536 tokens.", ("set_field", {
        "step": "parallelism", "field": "window", "value": 65536}), ()),
    ("parallelism", "Turn the shared KV pool off.", ("set_field", {
        "step": "parallelism", "field": "kv_pool", "value": False}), ()),
    ("weights", "Take the 4.0 weights.", ("select_profile", {"id": "exl3-4.0"}), ()),
    ("weights", "Pick EXL3 6.0.", ("select_profile", {"id": "exl3-6.0"}), ()),
    ("methods", "Set the reasoning effort to low.", ("set_field", {
        "step": "methods", "field": "reasoning_effort", "value": "low"}), ()),
    ("overview", "Go to the next step.", ("next", {}), ()),
    ("parallelism", "Open the tools step.", ("go_to_step", {"n": 3}), ()),
]


def cases() -> dict:
    from engine.generative.qwen35 import Qwen35Model
    import urllib.request
    try:      # the live box's facts, as the server gives them to Wren (setup_api.device_facts)
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/setup/device", timeout=5) as r:
            d = json.load(r)
        facts = {k: d.get(k) for k in ("mode", "preset", "mem_total_gb", "mem_available_gb", "gpu",
                                       "disk", "active_profile", "profiles_on_disk",
                                       "hf_token_present", "password_set")}
    except OSError:
        facts = {"mem_available_gb": round(avail(), 1)}
    app.STATE.clear()
    w = wren.Wren(app, device_facts=lambda: facts, setup_mode=True,
                  log=lambda s: print(s, flush=True))
    w._qwen = lambda name, folder: Qwen35Model(name, folder, max_ctx=4096)
    app.STATE["wren"] = w
    w.load()
    while w.state == "loading":
        time.sleep(0.2)
    st = w.status()
    out, right = [], 0
    for step, text, want, words in CASES:
        page = page_for(step)
        t = turn(w, text, page)
        changes = [c for c in t["tool_calls"] if c["name"] != "explain"]
        if want is None:
            ok = (bool(t["text"].strip()) and not changes
                  and any(x.lower() in t["text"].lower() for x in words))
        else:
            ok = changes == [{"name": want[0], "args": want[1]}]
            if changes:
                # the page's second round: the result goes back and the turn must end in words
                msgs = [{"role": "user", "content": text},
                        {"role": "assistant", "content": t["text"], "tool_calls": [
                            {"id": f"c{i}", "name": c["name"], "args": c["args"]}
                            for i, c in enumerate(t["tool_calls"])]}] + [
                        {"role": "tool", "content": json.dumps({"ok": True, "result": "ok"})}
                        for _ in t["tool_calls"]]
                said = ""
                for ev in w.chat({"messages": msgs, "page": page}):
                    said += ev.get("delta", "") if ev["type"] == "text" else ""
                t["second_round"] = said
                ok = ok and bool(said.strip() or t["text"].strip())
        right += ok
        out.append(dict(t, step=step, ok=ok))
    w.unload("asked")
    return {"part": "cases", "model": st["model"], "window": st["window"],
            "right": f"{right} of {len(CASES)}", "cases": out, "stats": dict(w.stats)}


if __name__ == "__main__":
    for part in sys.argv[1:] or ["own", "borrow"]:
        print(json.dumps({"own": own, "borrow": borrow, "cases": cases}[part]()), flush=True)
