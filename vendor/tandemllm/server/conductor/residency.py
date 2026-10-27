"""The residency planner (OPS-41): what may load beside the engine on the Spark, and in which order.

The board has one pool of unified memory. The engine loads first and holds most of it (its split:
server/memsplit.py); every other role -- a decision model, a librarian, a reader -- loads after it,
one at a time, through `check`:

    allowed      MemAvailable - growth - claim >= floor + margin        (15 + 6 GiB by default)

`growth` is what the engine may still take (OPS-47): its declared peak (`MEM_PEAK_GIB`, 77 GiB at
a 262,144-token window) less what it holds now. MemAvailable alone is the wrong number: at 17:50
on 2026-10-01 it read 37.9 GiB between requests, a 12 GiB reader would have been admitted, and the
next 262k prefill wants about 7 GiB more than the engine holds then. With no engine reading at
all (it is not running yet), the whole peak is still to come.
    refused_hold              the box lock is held or a hold's pause is armed
    refused_load_in_progress  another load (a role, or the engine itself) is running
    refused_memory            the byte-math above fails, or MemAvailable cannot be read
    refused_unknown_role      neither this document nor the roles document claims the role

The order of the refusals is the order above the memory line: a hold and a running load are
refused before any number is read. A claim is the role's declared PEAK (arbiter's batcher at 64
rows, not idle), or what it was measured to take from MemAvailable, never nvidia-smi's figure
(arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). The claims, the floor, the margin, the settle
time after a stop and the load order are the conductor's `residency` document: versioned, edited in
the dashboard, read here and by the ops scripts through the command line below.

Every check is journaled, allowed or refused, one JSON line in `<conductor root>/residency.jsonl`
(the last 2,000 kept), so the dashboard can show the last refusal and its reason. Nothing unloads
on its own: when MemAvailable falls after a role loaded, `plan` reports it and the dashboard shows
it. (The decision-model design keeps an unload order for later; it is not built here.)

    python -m server.conductor.residency check ROLE [--mem-available-gib X] [--hold] [--loading]
    python -m server.conductor.residency plan [--mem-available-gib X] [ROLE ...]
    python -m server.conductor.residency fraction ROLE --total-gib X   (vLLM's memory fraction)
    python -m server.conductor.residency loaded ROLE --before-gib A --after-gib B

`check` exits 0 when allowed and 3 when refused, printing the decision as one JSON line; the ops
scripts (ops/roles.sh) pass the hold and the running load they see, and the MemAvailable `free`
reports. `--root` (default ~/.qwen38-spark-engine/conductor) is the conductor's store.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

from server.conductor.store import DEFAULT_ROOT, Store

REASONS = ("refused_hold", "refused_load_in_progress", "refused_memory", "refused_unknown_role")
JOURNAL = "residency.jsonl"
KEEP = 2000
GIB = float(1 << 30)


def claims(doc: dict, roles: dict | None = None) -> dict[str, dict]:
    """role -> {gib, basis, note, source}: this document's claim first, else the roles document's
    `memory_claim_gib` (declared, SRV-133; a role with none claims nothing and is unknown here)."""
    out = {r: dict(c, source="residency") for r, c in (doc.get("claims") or {}).items()}
    for r, c in (roles or {}).items():
        gib = float(c.get("memory_claim_gib") or 0)
        if r not in out and gib > 0 and c.get("endpoint") not in ("self",):
            out[r] = {"gib": gib, "basis": "declared",
                      "note": "the roles document's memory_claim_gib", "source": "roles"}
    return out


def growth(engine_peak_gib: float | None, engine_now_gib: float | None) -> float:
    """GiB the engine may still take: its declared peak less what it holds now; the whole peak
    when it holds nothing we can read; 0 when it declares none."""
    if engine_peak_gib is None or engine_peak_gib <= 0:
        return 0.0
    return round(max(0.0, engine_peak_gib - (engine_now_gib or 0.0)), 2)


def check(role: str, mem_available_gib: float | None, doc: dict, roles: dict | None = None, *,
          hold: bool = False, loading: bool = False, engine_peak_gib: float | None = None,
          engine_now_gib: float | None = None) -> dict:
    """May `role` load now? The decision, with the numbers it was made on. `engine_peak_gib`
    and `engine_now_gib`: the engine's declared peak and what it holds now (its allocator's
    reserved bytes, the decision models left out), in GiB."""
    c = claims(doc, roles).get(role)
    floor, margin = float(doc["floor_gib"]), float(doc["margin_gib"])
    need = floor + margin
    grow = growth(engine_peak_gib, engine_now_gib)
    out = {"role": role, "allowed": False, "reason": None, "mem_available_gib": mem_available_gib,
           "claim_gib": c["gib"] if c else None, "basis": c["basis"] if c else None,
           "floor_gib": floor, "margin_gib": margin, "need_gib": need, "after_gib": None,
           "engine_peak_gib": engine_peak_gib, "engine_now_gib": engine_now_gib,
           "engine_growth_gib": grow}
    if hold:
        out["reason"] = "refused_hold"
    elif loading:
        out["reason"] = "refused_load_in_progress"
    elif c is None:
        out["reason"] = "refused_unknown_role"
    elif mem_available_gib is None:
        out["reason"] = "refused_memory"
    else:
        out["after_gib"] = round(mem_available_gib - grow - c["gib"], 2)
        if out["after_gib"] >= need:
            out["allowed"] = True
        else:
            out["reason"] = "refused_memory"
    return out


def plan(mem_available_gib: float | None, doc: dict, roles: dict | None = None,
         wanted: list[str] | None = None, *, engine_peak_gib: float | None = None,
         engine_now_gib: float | None = None) -> dict:
    """The load order and what fits now, loading one after the other: each role that fits takes
    its claim from what the next one sees. `wanted`: these roles (default: the order, then any
    other role with a claim, less a role the roles document switches off: it never loads, so it
    claims nothing). The engine's growth to its peak is counted once, up front."""
    cl = claims(doc, roles)
    order = [r for r in doc.get("order") or [] if r in cl]
    order += sorted(r for r in cl if r not in order)
    if wanted is None:
        order = [r for r in order if ((roles or {}).get(r) or {}).get("mode") != "off"]
    else:
        order = [r for r in order if r in wanted] + [r for r in wanted if r not in order]
    need = float(doc["floor_gib"]) + float(doc["margin_gib"])
    grow = growth(engine_peak_gib, engine_now_gib)
    # the engine's growth comes off once, before the first role; each role then sees what the
    # ones before it left
    left = round(mem_available_gib - grow, 2) if mem_available_gib is not None else None
    rows = []
    for r in order:
        d = check(r, left, doc, roles)
        # `mem_available_gib` of a row is what that role saw: MemAvailable less the growth
        d.update(engine_peak_gib=engine_peak_gib, engine_now_gib=engine_now_gib,
                 engine_growth_gib=grow)
        rows.append(dict(d, note=(cl.get(r) or {}).get("note"), source=(cl.get(r) or {}).get("source")))
        if d["allowed"]:
            left = d["after_gib"]
    return {"mem_available_gib": mem_available_gib, "floor_gib": float(doc["floor_gib"]),
            "margin_gib": float(doc["margin_gib"]), "need_gib": need,
            "engine_peak_gib": engine_peak_gib, "engine_now_gib": engine_now_gib,
            "engine_growth_gib": grow,
            "spendable_gib": (round(mem_available_gib - grow - need, 2)
                              if mem_available_gib is not None else None),
            "settle_s": int(doc["settle_s"]), "order": order, "rows": rows}


def vllm_fraction(role: str, total_gib: float, doc: dict, roles: dict | None = None) -> float:
    """`--gpu-memory-utilization` for a role served by vLLM: its claim over the board's total,
    rounded DOWN to two decimals (never vLLM's default 0.9, which on this board is the engine's
    memory too)."""
    c = claims(doc, roles).get(role)
    if c is None or total_gib <= 0:
        raise ValueError(f"no claim for {role!r}")
    return math.floor(c["gib"] / total_gib * 100) / 100


class Journal:
    """`residency.jsonl` in the conductor's root: one line per check and per measured load."""

    def __init__(self, root: str):
        self.path = os.path.join(os.path.realpath(os.path.expanduser(root)), JOURNAL)
        self._lock = threading.Lock()

    def write(self, rec: dict) -> dict:
        rec = dict(rec, ts=rec.get("ts") or round(time.time(), 3))
        with self._lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            self._trim()
        return rec

    def _trim(self) -> None:
        try:
            if os.path.getsize(self.path) < 400 * KEEP:
                return
            with open(self.path) as fh:
                lines = fh.readlines()
            if len(lines) <= KEEP:
                return
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                fh.writelines(lines[-KEEP:])
            os.replace(tmp, self.path)
        except OSError:
            pass

    def tail(self, n: int = 20) -> list[dict]:
        try:
            with open(self.path) as fh:
                lines = fh.readlines()[-max(1, n * 4):]
        except OSError:
            return []
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
        return out[-n:][::-1]

    def last_refusal(self) -> dict | None:
        return next((r for r in self.tail(200) if r.get("kind") == "check"
                     and not r.get("allowed")), None)


def docs(root: str) -> tuple[dict, dict]:
    """The current residency and roles documents from the conductor's store (version 0, the
    built-in ones, when the store has none)."""
    st = Store(root, readonly=True)
    return st.get("residency")[1], st.get("roles")[1]


def _mem_gib() -> float | None:
    """MemAvailable less the engine's pinned page cache (SRV-335: a warm n-gram table is free to the
    kernel, not to a role)."""
    from server import memsplit
    v = memsplit.counted_available()
    return None if v is None else round(v / GIB, 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m server.conductor.residency")
    ap.add_argument("--root", default=os.environ.get("QSE_CONDUCTOR_ROOT", DEFAULT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("role")
    c.add_argument("--mem-available-gib", type=float, default=None)
    c.add_argument("--hold", action="store_true")
    c.add_argument("--loading", action="store_true")
    c.add_argument("--source", default="cli")
    p = sub.add_parser("plan")
    p.add_argument("roles", nargs="*")
    p.add_argument("--mem-available-gib", type=float, default=None)
    for x in (c, p):
        # the engine's declared peak and what it holds now (its /health `memory`); the peak
        # defaults to MEM_PEAK_GIB from serve.env, and with no reading the whole peak counts
        x.add_argument("--engine-peak-gib", type=float,
                       default=float(os.environ["MEM_PEAK_GIB"])
                       if os.environ.get("MEM_PEAK_GIB") else None)
        x.add_argument("--engine-now-gib", type=float, default=None)
    f = sub.add_parser("fraction")
    f.add_argument("role")
    f.add_argument("--total-gib", type=float, required=True)
    ld = sub.add_parser("loaded")
    ld.add_argument("role")
    ld.add_argument("--before-gib", type=float, required=True)
    ld.add_argument("--after-gib", type=float, required=True)
    ld.add_argument("--source", default="cli")
    a = ap.parse_args(argv)
    doc, roles = docs(a.root)
    if a.cmd == "check":
        mem = a.mem_available_gib if a.mem_available_gib is not None else _mem_gib()
        d = check(a.role, mem, doc, roles, hold=a.hold, loading=a.loading,
                  engine_peak_gib=a.engine_peak_gib, engine_now_gib=a.engine_now_gib)
        Journal(a.root).write(dict(d, kind="check", source=a.source))
        print(json.dumps(d, sort_keys=True))
        return 0 if d["allowed"] else 3
    if a.cmd == "plan":
        mem = a.mem_available_gib if a.mem_available_gib is not None else _mem_gib()
        print(json.dumps(plan(mem, doc, roles, a.roles or None, engine_peak_gib=a.engine_peak_gib,
                              engine_now_gib=a.engine_now_gib), sort_keys=True))
        return 0
    if a.cmd == "fraction":
        print(f"{vllm_fraction(a.role, a.total_gib, doc, roles):.2f}")
        return 0
    took = round(a.before_gib - a.after_gib, 2)
    Journal(a.root).write({"kind": "loaded", "role": a.role, "before_gib": a.before_gib,
                           "after_gib": a.after_gib, "took_gib": took, "source": a.source,
                           "claim_gib": (claims(doc, roles).get(a.role) or {}).get("gib")})
    print(json.dumps({"role": a.role, "took_gib": took}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
