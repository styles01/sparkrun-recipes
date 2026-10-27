"""The alternated in-engine block A/B: the adoption instrument for a bit-identical item.

The row cannot decide a change under about 2 %: its within-side spread of ms a round is ~1.9 %
(phase5 hold 14), and the kr1 table, -2.3 ms on a 24-node verify in the engine, read -0.5 %
there, not resolved. For an item whose bits are the release's the tokens cannot change, so the only
thing left to measure is time, and the loose block of `tools/profile_cycle.py`'s loop on the same
tokens repeats to 0.3-0.9 % (phase5b hold 3). This tool runs that loop in alternated order:

    for each pair:  for each workload:  base, then every candidate state

and applies the row's own rule (`tools/row3.py` verdicts: the difference bigger than the spread
AND the run ranges disjoint) to ms a block, per workload and pooled by the row's arm mix.

A STATE is a set of module attributes, `name=module:attr=value[;attr=value][+module:attr=value]`,
switched inside one process (the verify graphs key on the kernel flags, engine/verify_graph.py
`signature()`, so each state replays its own graphs). A state that needs an environment variable
read at import or load time (`name=env:QWEN38_X=1[;QWEN38_Y=2]`) runs one process a state instead,
alternated the same way (`--proc`, chosen automatically when any state has an env part).

What it holds fixed so that time is the only difference:
  - the router's learned verify costs are frozen at the served table (`learn_cost` off) unless
    `--learn-cost`: the length router keeps them across requests (engine/lenrouter.py reset), and a
    faster state would otherwise change the next request's latch and trees, and with them the text
    at a sub-ulp tie -- a policy change, not the kernel's.
  - every state is warmed at both widths and its verify graphs 2..`--precapture` are captured before
    anything is timed (a graph captured inside a measured run inflates it; measured-first reads
    slowest).
  - every run's tokens are compared with the base's first run of the same workload. Under the
    lossless ruling of 2026-09-25 (no text change against the release) any difference is a FAIL;
    `--greedy-check` adds the one-token greedy run per workload and says whether a difference is a
    sub-ulp logit tie.

TTFT is not measured here (no prefill is timed); the row still judges the set before a deploy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The row's arm mix: 53 % of its rounds on the narrow arm (prose-like, 16-node trees), 47 % on the
# wide arm (chat and code after the latch, 24-node trees); phase5 hold 14's drafter logs.
DEFAULT_MIX = "prose=0.53,chat=0.235,code=0.235"


# ---------------------------------------------------------------- states


def parse_state(spec: str) -> dict:
    """`name=module:attr=value[;attr=value][+module:attr=value]` or `name=env:K=V[;K=V]`, or
    `name=` for the served configuration. Returns {name, mods: [(module, {attr: text})], env}."""
    if "=" not in spec:
        raise SystemExit(f"--state {spec!r}: expected name=...")
    name, rest = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"--state {spec!r}: empty name")
    mods, env = [], {}
    for part in filter(None, rest.split("+")):
        if ":" not in part:
            raise SystemExit(f"--state {spec!r}: {part!r} is not module:attr=value")
        mod, assigns = part.split(":", 1)
        kv = {}
        for a in filter(None, assigns.split(";")):
            if "=" not in a:
                raise SystemExit(f"--state {spec!r}: {a!r} is not attr=value")
            k, v = a.split("=", 1)
            kv[k.strip()] = v.strip()
        if mod == "env":
            env.update(kv)
        else:
            mods.append((mod, kv))
    return {"name": name, "mods": mods, "env": env}


def _cast(current, text: str):
    """A value written on the command line, as the type the module holds."""
    if isinstance(current, bool):
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise SystemExit(f"{text!r} is not a boolean")
    if current is None:
        # an attribute whose default is None (an override that is off): a number when it reads as one
        for kind in (int, float):
            try:
                return kind(text)
            except ValueError:
                pass
        return text
    return type(current)(text)


class Switch:
    """Applies one state at a time: every attribute any state touches goes back to the module's own
    value first, then the state's are set."""

    def __init__(self, states: list[dict]):
        self.orig = {}
        self.plan = {}
        for st in states:
            sets = []
            for mod_name, kv in st["mods"]:
                mod = importlib.import_module(mod_name)
                for attr, text in kv.items():
                    if not hasattr(mod, attr):
                        raise SystemExit(f"state {st['name']}: {mod_name} has no attribute {attr}")
                    self.orig.setdefault((mod, attr), getattr(mod, attr))
                    sets.append((mod, attr, _cast(self.orig[(mod, attr)], text)))
            self.plan[st["name"]] = sets

    def apply(self, name: str) -> None:
        for (mod, attr), v in self.orig.items():
            setattr(mod, attr, v)
        for mod, attr, v in self.plan[name]:
            setattr(mod, attr, v)

    def restore(self) -> None:
        for (mod, attr), v in self.orig.items():
            setattr(mod, attr, v)


def order(names: list[str], pairs: int) -> list[tuple[int, str]]:
    """Base then each candidate, pair after pair: A B A B ..., never A A B B."""
    return [(p, n) for p in range(pairs) for n in names]


# ---------------------------------------------------------------- the rule


def across(xs: list[float]) -> dict:
    """The row's summary of a statistic over runs (tools/row3.py across)."""
    xs = sorted(xs)
    med = statistics.median(xs)
    return {"median": med, "min": xs[0], "max": xs[-1],
            "spread_pct": 100.0 * (xs[-1] - xs[0]) / med if med else 0.0, "n": len(xs)}


def verdict(base: list[float], other: list[float], higher_is_better: bool = False) -> dict:
    """Tools/row3.py's rule: RESOLVED when the medians differ by more than the larger spread AND
    the run ranges are disjoint. The per-pair signs are reported beside it."""
    b, o = across(base), across(other)
    delta = 100.0 * (o["median"] - b["median"]) / b["median"]
    noise = max(b["spread_pct"], o["spread_pct"])
    disjoint = o["min"] > b["max"] or o["max"] < b["min"]
    resolved = abs(delta) > noise and disjoint
    better = (delta > 0) == higher_is_better
    signs = "".join("+" if y > x else ("-" if y < x else "0") for x, y in zip(base, other))
    return {"base": b, "other": o, "delta_pct": delta, "delta": o["median"] - b["median"],
            "noise_pct": noise, "disjoint": disjoint, "resolved": resolved,
            "worse": resolved and not better, "better": resolved and better, "signs": signs,
            "verdict": ("RESOLVED " + ("better" if better else "WORSE")) if resolved
            else "not resolved"}


def parse_mix(text: str) -> dict[str, float]:
    mix = {}
    for kv in filter(None, text.split(",")):
        k, v = kv.split("=")
        mix[k.strip()] = float(v)
    return mix


def pooled(per_workload: dict[str, list[float]], mix: dict[str, float]) -> list[float]:
    """Per pair, the workloads' ms a block weighted by the mix (renormalised over the workloads
    that ran)."""
    names = [w for w in per_workload if mix.get(w, 0.0) > 0]
    if not names:
        names = list(per_workload)
        mix = {w: 1.0 for w in names}
    tot = sum(mix[w] for w in names)
    n = min(len(per_workload[w]) for w in names)
    return [sum(mix[w] * per_workload[w][i] for w in names) / tot for i in range(n)]


def tokens_check(runs: list[dict], base: str) -> dict:
    """Every run's tokens against the base state's first run of the same workload. Returns
    {workload: None | {state, pair, at}} -- the first run that differs and where."""
    from tools.block_budget import first_divergence
    ref, out = {}, {}
    for r in runs:
        if r["state"] == base and r["workload"] not in ref:
            ref[r["workload"]] = r["out"]
    for r in runs:
        w = r["workload"]
        out.setdefault(w, None)
        if out[w] is not None or w not in ref:
            continue
        d = first_divergence(ref[w], r["out"])
        if d is not None:
            out[w] = {"state": r["state"], "pair": r["pair"], "at": d}
    return out


def judge(result: dict, rule: str = "better") -> tuple[int, list[str]]:
    """Per workload and pooled, every candidate against the base. rc 0 PASS, 1 tokens differ or
    something resolved worse, 2 (rule `better`) the pooled ms a block not resolved better."""
    runs, base, mix = result["runs"], result["base"], result["mix"]
    lines, rc = [], 0
    toks = tokens_check(runs, base)
    result["tokens"] = toks
    for w, bad in toks.items():
        if bad is None:
            lines.append(f"  tokens {w:<6} identical in every run of every state")
        else:
            lines.append(f"  tokens {w:<6} DIFFER: state {bad['state']} pair {bad['pair'] + 1} "
                         f"from token {bad['at']}")
            rc = 1
    for g, why in (result.get("greedy") or {}).items():
        lines.append(f"  greedy {g}: {why}")
    result["verdicts"] = {}
    workloads = list(dict.fromkeys(r["workload"] for r in runs))
    for cand in result["states"][1:]:
        per_b, per_c = {}, {}
        for w in workloads:
            per_b[w] = [r["block_ms"] for r in runs if r["state"] == base and r["workload"] == w]
            per_c[w] = [r["block_ms"] for r in runs if r["state"] == cand and r["workload"] == w]
        vs = {w: verdict(per_b[w], per_c[w]) for w in workloads}
        vs["pooled"] = verdict(pooled(per_b, mix), pooled(per_c, mix))
        tb = {w: verdict([r["tok_blk"] for r in runs if r["state"] == base and r["workload"] == w],
                         [r["tok_blk"] for r in runs if r["state"] == cand and r["workload"] == w],
                         higher_is_better=True) for w in workloads}
        result["verdicts"][cand] = {"ms_blk": vs, "tok_blk": tb}
        lines.append(f"  {cand} against {base}, ms a block (loose), {len(per_b[workloads[0]])} "
                     f"pairs:")
        for w, v in vs.items():
            lines.append(f"    {w:<7} {v['base']['median']:8.2f} -> {v['other']['median']:8.2f} "
                         f"({v['delta']:+6.2f} ms, {v['delta_pct']:+5.2f} %, noise "
                         f"{v['noise_pct']:4.2f} %, pairs {v['signs']})  {v['verdict']}")
            if v["worse"]:
                rc = 1
        for w, v in tb.items():
            if v["resolved"]:
                lines.append(f"    tok/blk {w}: {v['base']['median']:.3f} -> "
                             f"{v['other']['median']:.3f} {v['verdict']}")
        if rc == 0 and rule == "better" and not vs["pooled"]["better"]:
            rc = 2
    word = {0: "PASS", 1: "FAIL", 2: "FAIL"}[rc]
    why = {0: "tokens identical, nothing resolved worse" + (", pooled ms a block resolved better"
                                                           if rule == "better" else ""),
           1: "tokens differ or something resolved worse",
           2: "tokens identical, nothing resolved worse, but pooled ms a block is not resolved "
              "better"}[rc]
    lines.append(f"[block-ab] {result['label']} ({rule}): {word} -- {why}")
    return rc, lines


def stub(result: dict, lines: list[str]) -> str:
    """The ledger stub: every run, the tokens check, the verdicts, the code hash."""
    out = [f"## {time.strftime('%Y-%m-%d %H:%M')} -- block A/B {result['label']} (tools/block_ab.py)",
           "",
           f"Code `{result.get('code', 'n/a')[:16]}`; states: " +
           "; ".join(f"`{s}`" for s in result["state_specs"]) +
           f"; {result['pairs']} pairs, alternated; workloads {','.join(result['workloads'])} at "
           f"{result['max_new']} tokens; mix `{','.join(f'{k}={v}' for k, v in result['mix'].items())}`; "
           f"precapture {result['precapture']}; {result.get('discard', 0)} pair(s) discarded first; "
           f"learned costs {'on' if result['learn_cost'] else 'frozen'}; "
           f"{'one process a state' if result['proc'] else 'one process'}.",
           "", "```",
           f"{'pair':>4}  {'state':<12} {'workload':<8} {'ms/blk':>8} {'tok/blk':>8} {'blocks':>6} "
           f"{'tok/s':>7}  tokens sha"]
    for r in result["runs"]:
        out.append(f"{r['pair'] + 1:>4}  {r['state']:<12} {r['workload']:<8} {r['block_ms']:8.2f} "
                   f"{r['tok_blk']:8.3f} {r['blocks']:6d} {r['tok_s']:7.2f}  {r['sha'][:12]}")
    out += ["```", "", "```"] + lines + ["```", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- the engine side


def measure(a, states: list[dict], pairs: int) -> dict:
    """Build the served engine once, warm every state, then run the alternated pairs."""
    import torch
    from tools import profile_cycle as pc
    from tools.block_budget import PROMPTS

    from transformers import AutoTokenizer
    cfg, eng, drafter, arms, ng, k = pc.build(a)
    tk = AutoTokenizer.from_pretrained(cfg.path)
    if not a.learn_cost:
        drafter.learn_cost = False

    def ids(text: str) -> torch.Tensor:
        msg = [{"role": "user", "content": text}]
        s = tk.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
        return tk(s, return_tensors="pt").input_ids[0].cuda()

    sw = Switch(states)
    names = [s["name"] for s in states]
    print("[block-ab] warm both widths in every state", flush=True)
    for n in names:
        sw.apply(n)
        for fixed in (8, 16):
            drafter.fixed = fixed
            pc.cycle(eng, drafter, ids(PROMPTS["chat"]), a.warm, k, pc.Phases(strict=False))
        if a.precapture and eng._graphs_for(2, 0) is not None:
            with torch.no_grad():
                n_g = eng._graphs.precapture(widths=range(2, a.precapture + 1))
            print(f"[block-ab] {n}: verify graphs 2..{a.precapture} captured: {n_g}", flush=True)
    drafter.fixed = a.fixed
    workloads = a.workloads.split(",")

    greedy = {}
    if a.greedy_check:
        from engine.spec import generate_greedy
        sw.apply(names[0])
        for w in workloads:
            with torch.no_grad():
                g_out, g_st = generate_greedy(eng, ids(PROMPTS[w]), a.max_new, record_gaps=True)
            greedy[w] = (g_out, g_st.gaps, g_st.tops)

    # the discarded pair(s): the first measured runs after the warm-up read slow (phase6 hold 4: the
    # base's first pair was 3 ms a block over its other three on every workload), so every state runs
    # every workload once more before anything is recorded
    discarded = []
    for p, n in order(names, a.discard):
        sw.apply(n)
        for w in workloads:
            ph = pc.Phases(strict=False)
            pc.cycle(eng, drafter, ids(PROMPTS[w]), a.max_new, k, ph)
            discarded.append({"pair": p, "state": n, "workload": w,
                              "block_ms": sum(ph.block_ms) / len(ph.block_ms)})
            print(f"[block-ab] discarded {n:<12} {w:<6} {discarded[-1]['block_ms']:8.2f} ms/blk",
                  flush=True)
    runs, gverd = [], {}
    for p, n in order(names, pairs):
        sw.apply(n)
        for w in workloads:
            ph = pc.Phases(strict=False)
            undo = pc.instrument(arms, ng, ph)
            out, st = pc.cycle(eng, drafter, ids(PROMPTS[w]), a.max_new, k, ph)
            undo()
            nb = len(ph.block_ms)
            r = {"pair": p, "state": n, "workload": w, "block_ms": sum(ph.block_ms) / nb,
                 "blocks": nb, "tokens": st["tokens"], "tok_blk": st["tokens"] / nb,
                 "tok_s": st["tok_s"], "out": list(out),
                 "sha": hashlib.sha256(json.dumps(list(out)).encode()).hexdigest()}
            if w in greedy and (n, w) not in gverd:
                from tools.verify_spec import compare
                g_out, gaps, tops = greedy[w]
                ok, why = compare(g_out, out, gaps, tk, tops)
                gverd[(n, w)] = why if ok else "FAIL " + why
            runs.append(r)
            print(f"[block-ab] pair {p + 1} {n:<12} {w:<6} {r['block_ms']:8.2f} ms/blk "
                  f"{r['tok_blk']:.3f} tok/blk {r['tok_s']:6.2f} tok/s  {r['sha'][:12]}", flush=True)
    sw.restore()
    return {"runs": runs, "discarded": discarded,
            "greedy": {f"{n}/{w}": v for (n, w), v in gverd.items()}}


def run_proc(a, states: list[dict]) -> dict:
    """One process a state, alternated: base, candidate, base, candidate ..."""
    runs, greedy = [], {}
    here = os.path.abspath(__file__)
    for p, st in order([s["name"] for s in states], a.pairs):
        spec = next(s for s in states if s["name"] == st)
        env = dict(os.environ)
        env.update(spec["env"])
        child = os.path.join(a.out, f"child-{p + 1}-{st}.json")
        argv = [sys.executable, "-u", here, "--child", child, "--label", a.label]
        for s in a.state:
            if s.split("=", 1)[0] == st:
                argv += ["--state", s]
        for k in ("model", "nvfp4", "fp8_head", "ckpt8", "ckpt16", "corpus"):
            v = getattr(a, k)
            if v:
                argv += [f"--{k.replace('_', '-')}", v]
        argv += ["--max-len", str(a.max_len), "--max-new", str(a.max_new), "--warm", str(a.warm),
                 "--workloads", a.workloads, "--precapture", str(a.precapture),
                 "--fixed", str(a.fixed), "--discard", str(a.discard)]
        if a.learn_cost:
            argv.append("--learn-cost")
        if a.greedy_check and p == 0:
            argv.append("--greedy-check")
        print(f"[block-ab] pair {p + 1} {st}: its own process", flush=True)
        rc = subprocess.call(argv, env=env)
        if rc != 0:
            raise SystemExit(f"[block-ab] the {st} process of pair {p + 1} failed (rc {rc})")
        got = json.load(open(child))
        for r in got["runs"]:
            r["pair"] = p
            runs.append(r)
        greedy.update(got.get("greedy") or {})
    return {"runs": runs, "greedy": greedy}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="block-ab")
    ap.add_argument("--state", action="append", default=[],
                    help="name=module:attr=value[;attr=value][+module:...] or name=env:K=V; the "
                         "first is the base (default: base= and nothing else)")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--workloads", default="prose,chat,code")
    ap.add_argument("--mix", default=DEFAULT_MIX)
    ap.add_argument("--rule", default="better", choices=("better", "noworse"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--nvfp4", default=os.environ.get("QWEN38_NVFP4"))
    ap.add_argument("--fp8-head", default=os.environ.get("QWEN38_FP8_HEAD"))
    ap.add_argument("--ckpt8", default=None)
    ap.add_argument("--ckpt16", default=None)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--warm", type=int, default=48)
    ap.add_argument("--fixed", type=int, default=0)
    ap.add_argument("--precapture", type=int, default=32)
    ap.add_argument("--discard", type=int, default=1,
                    help="pairs run and thrown away before the measured ones (the first runs after "
                         "the warm-up read slow)")
    ap.add_argument("--learn-cost", action="store_true",
                    help="let the router learn verify costs as served (default: frozen, see above)")
    ap.add_argument("--greedy-check", action="store_true")
    ap.add_argument("--proc", action="store_true", help="one process a state (implied by env:)")
    ap.add_argument("--out", default="results/blockab")
    ap.add_argument("--child", default="", help=argparse.SUPPRESS)
    ap.add_argument("--judge", default="", help="re-judge a saved result JSON; touches no board")
    a = ap.parse_args()

    if a.judge:
        result = json.load(open(a.judge))
        rc, lines = judge(result, a.rule)
        print("\n".join(lines))
        sys.exit(rc)

    a.latch = a.drop_idle = True
    specs = a.state or ["base="]
    states = [parse_state(s) for s in specs]
    if len({s["name"] for s in states}) != len(states):
        raise SystemExit("state names must differ")

    if a.child:
        got = measure(a, states, a.pairs if len(states) > 1 else 1)
        json.dump(got, open(a.child, "w"))
        return

    if not (a.ckpt8 and a.ckpt16):
        raise SystemExit("--ckpt8 and --ckpt16 are needed to build the served drafters")
    os.makedirs(a.out, exist_ok=True)
    proc = a.proc or any(s["env"] for s in states)
    t0 = time.time()
    got = run_proc(a, states) if proc else measure(a, states, a.pairs)
    from tools.row3 import code_hash
    result = {"label": a.label, "code": code_hash(os.path.dirname(os.path.dirname(
                  os.path.abspath(__file__)))),
              "state_specs": specs, "states": [s["name"] for s in states],
              "base": states[0]["name"], "pairs": a.pairs, "workloads": a.workloads.split(","),
              "max_new": a.max_new, "mix": parse_mix(a.mix), "precapture": a.precapture,
              "discard": a.discard,
              "learn_cost": a.learn_cost, "proc": proc, "seconds": time.time() - t0,
              "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("QWEN38_")},
              **got}
    rc, lines = judge(result, a.rule)
    path = os.path.join(a.out, f"{a.label}.json")
    json.dump(result, open(path, "w"), indent=1)
    md = stub(result, lines)
    open(os.path.join(a.out, f"{a.label}-stub.md"), "w").write(md)
    print(md, flush=True)
    print(f"[block-ab] wrote {path}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
