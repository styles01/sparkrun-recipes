"""On the board: do sampled requests still follow the target's distribution, and is greedy untouched?

`tests/test_sample_tree.py` proves the walks exact on the CPU (30k-walk histograms against the target's own). This
is the check on the served stack: the same short sampled request many times against each server configuration,
and the answers' distribution compared with the reference -- the target alone, no drafter (`--drafter none`). Any
deterministic function of an answer has the same distribution under two exact samplers, so the answers are cut
into their first w words (w = 1..4) and each prefix's counts are compared, per prompt, by a chi-square test of
homogeneity (prefixes the two groups together see fewer than 10 times pooled into one cell). The first word is
drawn from the prefill's logits in every configuration and is a sanity row; words 2..4 come out of the tree walk.

A test that cannot fail proves nothing, so a control rides along: the reference at a cooler temperature, a real
shift of the kind a walk biased toward its draft would make (more mass on the likely words). The check PASSES
when no configuration has a p-value below `--alpha` over all its prompt x w tests (Bonferroni: alpha / tests)
AND the control does. `mode share` is the fraction of answers whose w-word prefix is the reference's commonest --
the direction a greedy-biased walk would push.

`--greedy` also sends each bench workload once at temperature 0 to every configuration that drafts: the answers
must be the same text and the same token count (the sampled-tree flag must not touch a greedy request).

    # inside ops/hold.sh (it starts one test server per configuration, one after the other)
    python tools/sampled_dist.py --serve --port 8011 --n 160 --greedy --json results/p5b/h4/dist.json
    python tools/sampled_dist.py --read results/p5b/h4/dist.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEMPERATURE = 0.7
# name -> (server args, temperature). `nospec` is the reference, `cool` the control.
CONFIGS = {
    "nospec": (["--drafter", "none"], TEMPERATURE),
    "chain": ([], TEMPERATURE),
    "det": (["--sampled-tree=det"], TEMPERATURE),
    "mixed": (["--sampled-tree=mixed"], TEMPERATURE),
    "cool": (["--drafter", "none"], 0.5),
}
REFERENCE, CONTROL = "nospec", "cool"
# Short answers with a few likely words at every step, so the counts say something at a few hundred requests,
# and text the drafter predicts well, so the walk goes several nodes deep.
PROMPTS = {
    "animals": "Name four animals, separated by commas. Reply with the list only.",
    "ocean": "Write one short sentence about the ocean.",
    "story": "Continue this story with one sentence: The old lighthouse keeper opened the door and",
}
WORDS = (1, 2, 3, 4)


def pick(names: str) -> dict[str, str]:
    """`--prompts`: the named prompts in PROMPTS order, all of them for an empty string."""
    want = [n for n in names.split(",") if n]
    unknown = set(want) - set(PROMPTS)
    if unknown:
        raise SystemExit(f"--prompts names {sorted(unknown)}; known: {', '.join(PROMPTS)}")
    return {k: v for k, v in PROMPTS.items() if not want or k in want}


# ------------------------------------------------------------------------------------------ statistics


def _gammq(a: float, x: float) -> float:
    """The regularized upper incomplete gamma Q(a, x) (series below a + 1, continued fraction above)."""
    if x <= 0.0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1.0 - a
    c, d = 1.0 / 1e-300, 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x: float, dof: int) -> float:
    return _gammq(dof / 2.0, x / 2.0) if dof > 0 else 1.0


def prefix(text: str, w: int) -> str:
    words = text.split()
    return " ".join(words[:w]) + ("" if len(words) >= w else " <end>")


def homogeneity(a: list[str], b: list[str], min_count: int = 10) -> tuple[float, int, float]:
    """Chi-square test that two samples of categories come from one distribution: (statistic, dof, p). Categories
    seen fewer than `min_count` times in both samples together are pooled into one cell."""
    ca, cb = Counter(a), Counter(b)
    cells: dict[str, list[int]] = {}
    for k in set(ca) | set(cb):
        key = k if ca[k] + cb[k] >= min_count else "<rare>"
        c = cells.setdefault(key, [0, 0])
        c[0] += ca[k]
        c[1] += cb[k]
    na, nb = len(a), len(b)
    n = na + nb
    stat = 0.0
    for x, y in cells.values():
        tot = x + y
        for obs, size in ((x, na), (y, nb)):
            exp = tot * size / n
            stat += (obs - exp) ** 2 / exp
    dof = len(cells) - 1
    return stat, dof, chi2_sf(stat, dof)


def analyse(answers: dict[str, dict[str, list[str]]], alpha: float = 0.01) -> dict:
    """`answers[config][prompt]` -> the texts. Every config against the reference, per prompt and prefix length."""
    ref = answers[REFERENCE]
    out: dict = {"alpha": alpha, "configs": {}}
    for cfg, per in answers.items():
        if cfg == REFERENCE:
            continue
        rows, pmin = [], 1.0
        for prompt, texts in per.items():
            for w in WORDS:
                a = [prefix(t, w) for t in ref[prompt]]
                b = [prefix(t, w) for t in texts]
                stat, dof, p = homogeneity(a, b)
                mode = Counter(a).most_common(1)[0][0]
                rows.append({"prompt": prompt, "w": w, "chi2": stat, "dof": dof, "p": p,
                             "mode": mode, "mode_ref": a.count(mode) / len(a), "mode_cfg": b.count(mode) / len(b),
                             "n": (len(a), len(b))})
                if w > 1:
                    pmin = min(pmin, p)
        tests = sum(1 for r in rows if r["w"] > 1)
        out["configs"][cfg] = {"rows": rows, "p_min": pmin, "tests": tests,
                               "flagged": pmin < alpha / max(tests, 1)}
    ctrl = out["configs"].get(CONTROL)
    out["control_seen"] = bool(ctrl and ctrl["flagged"])
    out["pass"] = out["control_seen"] and not any(
        c["flagged"] for name, c in out["configs"].items() if name != CONTROL)
    return out


def greedy_same(greedy: dict[str, dict[str, dict]]) -> dict:
    """`greedy[config][workload]` -> {text, tokens}: every config's answer against the first one's."""
    names = list(greedy)
    if not names:
        return {"pass": None}
    base = greedy[names[0]]
    diff = {n: [w for w in base if greedy[n].get(w) != base[w]] for n in names[1:]}
    return {"base": names[0], "differ": diff, "pass": not any(diff.values())}


def show(res: dict) -> str:
    lines = []
    for cfg, c in res["configs"].items():
        verdict = "FLAGGED" if c["flagged"] else "ok"
        lines.append(f"== {cfg} against {REFERENCE}: min p {c['p_min']:.3g} over {c['tests']} tests "
                     f"(Bonferroni {res['alpha']}/{c['tests']} = {res['alpha'] / max(c['tests'], 1):.2g}) {verdict}")
        for r in c["rows"]:
            lines.append(f"   {r['prompt']:8s} w={r['w']}  chi2 {r['chi2']:7.1f} dof {r['dof']:3d}  p {r['p']:.3g}   "
                         f"mode share {r['mode_ref']:.2f} -> {r['mode_cfg']:.2f}  ({r['mode'][:32]!r})")
    lines.append(f"[dist] control ({CONTROL}) seen: {res['control_seen']}; "
                 f"{'PASS' if res['pass'] else 'FAIL'}")
    g = res.get("greedy")
    if g and g.get("pass") is not None:
        lines.append(f"[greedy] against {g['base']}: "
                     + ", ".join(f"{n} {'identical' if not d else 'DIFFERS on ' + ','.join(d)}"
                                 for n, d in g["differ"].items())
                     + f" -> {'PASS' if g['pass'] else 'FAIL'}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ the served run


def _post(port: int, prompt: str, max_tokens: int, temperature: float) -> dict:
    import urllib.request
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as fh:
        return json.load(fh)


def serve(a) -> dict:
    from tools import row3
    from tools.bench_decode import PROMPTS as BENCH
    raw: dict = {"args": vars(a), "answers": {}, "greedy": {}, "walls": {}}
    for cfg in a.configs.split(","):
        sargs, temp = CONFIGS[cfg]
        ns = argparse.Namespace(port=a.port, python=Path(a.python), pythonpath=Path(a.pythonpath),
                                repo=Path(a.repo), max_len=a.max_len, len_fixed=0, budget=16,
                                nvfp4=os.environ.get("NV") or row3.DEFAULT_NV,
                                head=os.environ.get("HEAD") or row3.DEFAULT_HEAD,
                                server_arg=["--drop-idle"] + sargs, len_latch=True, start_timeout=600, tree=True)
        proc = row3.start_server(ns, {}, Path(a.log_dir) / f"{cfg}.log")
        t0 = time.time()
        try:
            _post(a.port, PROMPTS["ocean"], 8, 0.0)                       # warm
            raw["answers"][cfg] = {}
            for name, text in pick(a.prompts).items():
                raw["answers"][cfg][name] = [
                    _post(a.port, text, a.max_tokens, temp)["choices"][0]["message"]["content"]
                    for _ in range(a.n)]
            if a.greedy and cfg not in (REFERENCE, CONTROL):
                raw["greedy"][cfg] = {}
                for name, text in BENCH.items():
                    r = _post(a.port, text, 256, 0.0)
                    raw["greedy"][cfg][name] = {"text": r["choices"][0]["message"]["content"],
                                                "tokens": r["usage"]["completion_tokens"]}
        finally:
            row3.stop_server(proc)
        raw["walls"][cfg] = time.time() - t0
        print(f"[dist] {cfg}: {a.n} x {len(pick(a.prompts))} sampled answers in {raw['walls'][cfg]:.0f} s", flush=True)
        if a.json:                                  # after every configuration: a cut hold keeps what it has
            Path(a.json).write_text(json.dumps(raw, indent=1))
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--read", default="", help="a --json this tool wrote: analyse it again")
    ap.add_argument("--configs", default="nospec,chain,det,mixed,cool")
    ap.add_argument("--n", type=int, default=160, help="answers per prompt and configuration")
    ap.add_argument("--prompts", default="", help="comma-separated names from PROMPTS (default all)")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--python", default=os.path.expanduser("~/recipes/ling3-flash-dgx-spark/.venv/bin/python"))
    ap.add_argument("--pythonpath", default=os.path.expanduser("~/pylibs"))
    ap.add_argument("--log-dir", default="results/sampled-dist")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.read:
        raw = json.loads(Path(a.read).read_text())
    elif a.serve:
        raw = serve(a)
    else:
        raise SystemExit("--serve or --read")
    res = analyse(raw["answers"], a.alpha)
    res["greedy"] = greedy_same(raw.get("greedy", {}))
    print(show(res))
    if a.json:
        raw["analysis"] = res
        Path(a.json).write_text(json.dumps(raw, indent=1))


if __name__ == "__main__":
    main()
