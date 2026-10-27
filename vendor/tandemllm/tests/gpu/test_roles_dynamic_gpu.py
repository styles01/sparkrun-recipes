"""On the board: the small models go out under KV-pool pressure and come back, and Qwen's output
does not move by a bit (server/rolemem.py).

One process: the served engine (static KV, 16k window) with the served drafter stack, and the
two generative roles as the server builds them (engine/generative Slot over Qwen3.5-0.8B, the
shadow reader, and Qwen3.5-2B, the narrator, each in its own memory pool on the GPU). Steps:

  1  Qwen alone: a greedy tree decode of `--new` tokens on a real 8k prompt -> tokens T0, blocks,
     prefill logits L0.
  2  the roles load (MemAvailable before and after each); the narrator writes one line N0;
     Qwen again -> must equal T0/L0 with the roles loaded.
  3  pressure: a KV pool whose hard floor sits 2 GiB under MemAvailable is asked for 6 GiB; its
     reclaimers are the server's order (caches, roles, idle side, idle slot 0) with the caches and
     conversations as recorders. The roles level must unload the shadow first, then the narrator,
     give the memory back (MemAvailable measured), and nothing after it may run.
  4  Qwen with the roles out -> must equal T0/L0; a narrator line answers `unloaded_for_memory`
     at once and no load starts.
  5  the governor's reload (hold shortened for the test): narrator first, then the shadow, each
     through its Slot's load; the narrator's line after the reload N1 must equal N0.
  6  Qwen once more -> T0/L0.

Run on the box, the box alone (:8000 stopped), with the served environment:
    set -a; . ~/tandem-live/ops/serve.env; . ~/tandem-live/ops/profiles/exl3-4.5.env; set +a
    python tests/gpu/test_roles_dynamic_gpu.py [--new 128]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine import cache, kvpool  # noqa: E402
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from server import rolemem  # noqa: E402

GIB = 1 << 30


def avail() -> int:
    return kvpool._meminfo_avail()


def g(b) -> float:
    return round(b / GIB, 2)


def main() -> None:
    from engine import generative
    from engine.decision.device import Device
    from engine.generative.slot import Slot
    from engine.spec import generate_spec_tree
    from tools.verify_spec import tree_router
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--out", default=os.path.expanduser("~/rd1-runs/roles_dynamic_gpu.json"))
    a = ap.parse_args()
    M = 16384
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=os.environ.get("NV") or None,
                fp8_head=os.environ.get("HEAD") or None)
    eng = Qwen38Engine(cfg, w, max_len=M, paged=False)
    rp = os.path.expanduser("~/tandem-live")
    ns = types.SimpleNamespace(dflash2_ckpt=os.path.join(rp, "train/ft-b8-v2"),
                               dflash2_path="greedy", max_len=M,
                               lenrouter=os.path.join(rp, "train/ft-b16"), corpus="",
                               drop_idle=True,
                               deep=int(os.environ.get("QWEN38_DEEP", "0") or 0),
                               deep_after=int(os.environ.get("QWEN38_DEEP_AFTER", "2") or 2),
                               len_fixed=0)
    lr = tree_router(eng, ns, narrow=True)
    # a pinned length: the router's arm choice reads measured times, which a run-to-run identity
    # check must not depend on (tests/gpu/test_kvpool_identity_gpu.py pins 16 as well)
    lr.fixed = 16
    tok = AutoTokenizer.from_pretrained(cfg.path)
    text = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, "engine", "*.py"))))
    msgs = [{"role": "user", "content": text[:28000] + "\n\nList the five most important "
                                                      "functions above and what each does."}]
    r = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                tokenize=False)
    ids = torch.tensor(tok(r, add_special_tokens=False).input_ids, device="cuda")
    log: dict = {"prompt_tokens": int(ids.numel()), "new": a.new, "steps": []}
    bad: list = []

    def qwen(step: str):
        with torch.no_grad():
            eng.reset()
            eng.tap = None                 # the prefill alone, no drafter listening
            lg = cache.prefill(eng, None, ids.tolist(), "cuda", chunk=1024)[0].float().clone()
            eng.reset()
            t0 = time.time()
            lr.attach()            # the router holds the engine's tap (a prefill above may not)
            toks, st = generate_spec_tree(eng, ids, a.new, lr, 15)
            torch.cuda.synchronize()
        row = {"step": step, "tokens": len(toks), "blocks": st.blocks,
               "s": round(time.time() - t0, 1), "mem_available_gib": g(avail())}
        log["steps"].append(row)
        print(f"  {step:28s} {len(toks)} tokens in {st.blocks} blocks, {row['s']} s, "
              f"MemAvailable {row['mem_available_gib']} GiB", flush=True)
        return toks, st.blocks, lg

    T0, B0, L0 = qwen("qwen alone")

    def same(step, got):
        t, b, lg = got
        if t != T0 or b != B0 or not torch.equal(lg, L0):
            i = next((k for k, (x, y) in enumerate(zip(T0, t)) if x != y), None)
            bad.append(f"{step}: tokens {'=' if t == T0 else f'differ at {i}'}, blocks {B0} vs "
                       f"{b}, prefill logits {'=' if torch.equal(lg, L0) else 'differ'}")

    # 2 the roles, as the server builds them
    gate = threading.RLock()
    dev = Device(os.environ.get("QWEN38_NARRATOR_DEVICE") or "auto")
    slots = {}
    for role, name, d in (("shadow", os.environ.get("QWEN38_SHADOW_MODEL") or "qwen3.5-0.8b",
                           os.environ.get("QWEN38_SHADOW_DIR")),
                          ("narrator", os.environ.get("QWEN38_NARRATOR_MODEL") or "qwen3.5-2b",
                           os.environ.get("QWEN38_NARRATOR_DIR"))):
        m = generative.model(name, d or None, engine_dir=cfg.path)
        s = Slot(m, role=role, gate=gate, device=dev, log=lambda x: print(x, flush=True),
                 mem_available=lambda: round(avail() / GIB, 2),
                 planner=lambda n, role=role, **kw: {"allowed": True, "reason": None,
                                                     "role": role, "claim_gib": None})
        b = avail()
        s.load(wait=True)
        assert s.state == "loaded", (role, s.state, s.error)
        log[f"{role}_took_gib"] = g(b - avail())
        slots[role] = s
    narr = [{"role": "user", "content": "In eight words, say what a block drafter does."}]

    def line():
        r = slots["narrator"].run(narr, max_new=24, budget_ms=20000, held=False)
        return r.get("text") if "text" in r else r

    N0 = line()
    log["narrator_line_before"] = N0
    same("qwen with roles loaded", qwen("qwen with roles loaded"))

    # 3 pressure through a KV pool with the server's reclaim order
    rm = rolemem.RoleMemory([rolemem.SlotRole("shadow", slots["shadow"]),
                             rolemem.SlotRole("narrator", slots["narrator"])],
                            floor_gib=15, margin_gib=4, hold_s=2, thrash_s=600, meminfo=avail,
                            empty=torch.cuda.empty_cache, journal=lambda e: None,
                            log=lambda m: print(m, flush=True))
    calls = []

    def press(tag: str) -> dict:
        """One commit 1 GiB short of the hard floor: the levels run until it fits."""
        calls.clear()
        a = avail()
        pool = kvpool.KVPool(hard_floor=a - 1 * GIB, meminfo=avail)
        pool.add_reclaimer("caches", lambda want, o: (calls.append("caches"), 0)[1])
        pool.add_reclaimer("roles", lambda want, o: (calls.append("roles"), rm.reclaim(want, o))[1])
        pool.add_reclaimer("idle_side", lambda want, o: (calls.append("idle_side"), 0)[1])
        pool.add_reclaimer("idle_slot0", lambda want, o: (calls.append("idle_slot0"), 0)[1])
        try:
            pool.gate("test", 2 * GIB)
            ok = True
        except kvpool.PoolFull:
            ok = False
        b = avail()
        row = {"mem_available_before_gib": g(a), "after_gib": g(b), "gate_passed": ok,
               "levels_called": list(calls),
               "states": {k: s.state for k, s in slots.items()}}
        print(f"  pressure {tag}: levels {calls}, MemAvailable {g(a)} -> {g(b)} GiB, gate "
              f"{'passed' if ok else 'refused'}, states {row['states']}", flush=True)
        return row

    # the first shortfall: the shadow reader alone covers it, the narrator stays
    p1 = press("1")
    # the second: the shadow is out already, the narrator goes
    p2 = press("2")
    ev = [e for e in rm.events if e["kind"] == "role_unloaded"]
    log["pressure"] = {"first": p1, "second": p2,
                       "unloaded": [(e["role"], e["freed_gib"]) for e in ev]}
    if [e["role"] for e in ev] != ["shadow", "narrator"]:
        bad.append(f"unload order {[e['role'] for e in ev]}")
    if p1["states"] != {"shadow": "not_loaded", "narrator": "loaded"}:
        bad.append(f"first shortfall: {p1['states']} (the shadow alone should go)")
    for p in (p1, p2):
        if not p["gate_passed"] or "idle_side" in p["levels_called"]:
            bad.append(f"a conversation level ran or the gate refused: {p}")
    for s in slots.values():
        if s.state != "not_loaded" or not s.memory_hold:
            bad.append(f"{s.role} {s.state} hold {s.memory_hold}")

    # 4 roles out
    same("qwen with roles out", qwen("qwen with roles out"))
    r = slots["narrator"].run(narr, max_new=24, budget_ms=2000, held=False)
    if r != {"why": "unloaded_for_memory"} or slots["narrator"].state != "not_loaded":
        bad.append(f"narrator while out: {r}, {slots['narrator'].state}")

    # 5 reload
    t0 = time.time()
    order = []
    while time.time() - t0 < 300 and any(rm.out(n) for n in ("shadow", "narrator")):
        got = rm.tick()
        if got:
            order.append(got)
        time.sleep(0.5)
    a2 = avail()
    log["reload"] = {"order": order, "s": round(time.time() - t0, 1), "mem_available_gib": g(a2),
                     "states": {k: s.state for k, s in slots.items()}}
    print(f"  reload: {order} in {log['reload']['s']} s, MemAvailable {g(a2)} GiB", flush=True)
    if order != ["narrator", "shadow"] or any(s.state != "loaded" for s in slots.values()):
        bad.append(f"reload {order} {log['reload']['states']}")
    N1 = line()
    log["narrator_line_after"] = N1
    if N1 != N0:
        bad.append(f"narrator line changed across the reload: {N0!r} vs {N1!r}")

    # 6 Qwen once more
    same("qwen after reload", qwen("qwen after reload"))
    log["events"] = list(rm.events)
    log["bad"] = bad
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=1, default=str)
    print(f"[roles] raw: {a.out}")
    if bad:
        print("ROLES GPU FAIL:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("ROLES GPU PASS: shadow then narrator unloaded under pool pressure before any "
          "conversation level, memory back, Qwen tokens/blocks/prefill logits bit-identical "
          "loaded/out/reloaded, narrator line identical after the reload")


if __name__ == "__main__":
    main()
