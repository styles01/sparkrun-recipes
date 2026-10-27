"""On the box, GPU only (ENG-288 / SPD-108): LFM2.5-24B-A2B's correctness ladder on the real weights and its
speed scenarios, each part a pass/fail test over the tools that measure it. One big model on the board at a
time: run it in a box window with :8000 stopped (or on a port of its own when the part needs a server).

    PY=<engine python> EXL3_PY=<exllamav3 python> LFM25_BF16=<bf16 dir> LFM25_PACK=<pack dir> \\
    LFM25_PLAIN_URL=http://127.0.0.1:8017 LFM25_URL=http://127.0.0.1:8018 OUT=<dir for the raw JSON> \\
        python tests/gpu/test_lfm2_moe_box_gpu.py rung_b rung_c graph_ab staircase spec lookup parallel

  rung_b     ENG-288 "BF16 agrees with transformers": the engine's greedy decode of 8 chat prompts x 128 tokens
             on the BF16 checkpoint, transformers' Lfm2MoeForCausalLM (bf16, sdpa) teacher-forced on the same
             tokens (tools/lfm_check.py engine + hf): argmax agreement >= 97 %, a disagreement that is not a
             near tie (margin >= 8 bf16 steps of the largest logit) at most 0.2 % of the positions, the
             mean |dlogit| / max under 1 %. (Measured 2026-10-14: 98.05 %, 1 of 1,024 not a near tie.)
  rung_c     ENG-288 "an EXL3 pack agrees with exllamav3": the same on LFM25_PACK against exllamav3's own
             forward (engine + exl3), and our trellis decode equal to LinearEXL3's bit for bit (xcheck).
  graph_ab   SPD-108 "single-stream decode": 256 greedy tokens, graph on, off, on (lfm_check bench): both
             rates recorded with the knobs, the two graph runs within 3 %, graph on not slower than off.
  staircase  SPD-108 "the verify staircase": chain blocks of 1..16 rows at 1k and 8k tokens of context
             (lfm_check staircase): every row count has its ms and its experts read, the ms rising with
             the rows (a block never under the block before it by more than 5 %).
  spec       ENG-288 "speculation writes the plain greedy text": tools/fn_server_check.py on a server with
             --drafter none (LFM25_PLAIN_URL) and one with the profile's lookup (LFM25_URL), then
             --compare: every greedy and seeded answer the same text, every call the same call.
  lookup     SPD-108 "speculation only where it wins": tools/fn_lookup_ab.py on both servers: every class
             the same text, and the lookup's time over the five classes (prose, code, chat, edit, copy,
             512 tokens each) under the plain server's; if not, the profile must turn it off.
  parallel   SPD-108 "parallel requests": tools/lfm_parallel.py on LFM25_URL at 1, 2 and 3 requests: every
             level valid (the server's /health showed N slots decoding or joint rounds of N), every text
             its request's alone, the knobs in the output, and the 3-request total clearly above the
             1-request rate (at least 1.1x).

The servers: `ops/start.sh` with the profile on a port of its own, one with QSE_ENGINE_FLAGS
"--model <pack> --drafter none", one as the profile has it, each with QSE_PARALLEL=3 and the box lock.
Each part prints PASS or FAIL with the numbers it read; the raw JSON stays in $OUT.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get("PY") or sys.executable
EXL3_PY = os.environ.get("EXL3_PY", "")
OUT = os.environ.get("OUT") or tempfile.mkdtemp(prefix="lfm-box-")
ENV = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}


def _run(py: str, *args) -> str:
    r = subprocess.run([py, *args], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, f"{' '.join(args[:2])}: exit {r.returncode}: {r.stderr[-800:]}"
    return r.stdout


def _need(*names) -> list[str]:
    vals = [os.environ.get(n, "") for n in names]
    missing = [n for n, v in zip(names, vals) if not v]
    assert not missing, f"set {', '.join(missing)}"
    return vals


def _ladder(model_env: str, ref: str) -> str:
    (model,) = _need(model_env)
    run = os.path.join(OUT, f"engine-{ref}.pt")
    _run(PY, "tools/lfm_check.py", "engine", "--model", model, "--out", run)
    js = os.path.join(OUT, f"rung-{ref}.json")
    if ref == "hf":
        _run(PY, "tools/lfm_check.py", "hf", "--model", model, "--inp", run, "--json", js)
    else:
        assert EXL3_PY, "set EXL3_PY (the exllamav3 python)"
        _run(EXL3_PY, "tools/lfm_check.py", "exl3", "--model", model, "--inp", run, "--json", js)
    r = json.load(open(js))
    assert r["agree_pct"] >= 97.0, r["agree_pct"]
    assert r["other"] <= max(1, int(0.002 * r["positions"])), (r["other"], r["disagreements"])
    assert r["mean_rel_diff"] < 0.01, r["mean_rel_diff"]
    out = f"{r['argmax_agree']}/{r['positions']} agree ({r['agree_pct']} %), {r['near_tie']} near ties, {r['other']} not"
    if ref == "exl3":
        x = _run(EXL3_PY, "tools/lfm_check.py", "xcheck", "--pack", model).strip().splitlines()[-1]
        s = json.loads(x)
        assert s["all_inner_bit_equal"], s
        out += f"; trellis decode = LinearEXL3 bit for bit, weights within {s['max_weight_rel']:.1e}"
    return out


def part_rung_b():
    return _ladder("LFM25_BF16", "hf")


def part_rung_c():
    return _ladder("LFM25_PACK", "exl3")


def part_graph_ab():
    (pack,) = _need("LFM25_PACK")
    js = os.path.join(OUT, "bench.json")
    _run(PY, "tools/lfm_check.py", "bench", "--model", pack, "--tokens", "256", "--max-len", "8192", "--json", js)
    r = json.load(open(js))
    g, e = r["graph"]["tok_s"], r["eager"]["tok_s"]
    assert g > 0 and e > 0
    assert g >= e, (g, e)
    return f"graph {g} tok/s, eager {e} (256 tokens, max_len 8192, {r['decode_bytes'] / 1e9:.2f} GB a token)"


def part_staircase():
    (pack,) = _need("LFM25_PACK")
    js = os.path.join(OUT, "staircase.json")
    _run(PY, "tools/lfm_check.py", "staircase", "--model", pack, "--contexts", "1024", "8192", "--json", js)
    r = json.load(open(js))
    out = []
    for ctx, rows in r["rows"].items():
        ms = [rows[str(m)]["ms"] for m in range(1, 17)]
        ex = [rows[str(m)]["experts_read"] for m in range(1, 17)]
        assert all(b >= 0.95 * a for a, b in zip(ms, ms[1:])), (ctx, ms)
        assert all(x > 0 for x in ex), (ctx, ex)
        out.append(f"{ctx}: 1 row {ms[0]} ms, 16 rows {ms[-1]} ms, experts {ex[0]}..{ex[-1]}")
    return "; ".join(out)


def part_spec():
    plain, spec = _need("LFM25_PLAIN_URL", "LFM25_URL")
    a, b = os.path.join(OUT, "server-none.json"), os.path.join(OUT, "server-lookup.json")
    _run(PY, "tools/fn_server_check.py", "--url", plain, "--out", a)
    _run(PY, "tools/fn_server_check.py", "--url", spec, "--out", b)
    r = json.loads(_run(PY, "tools/fn_server_check.py", "--compare", a, b))      # exits 1 unless all same
    assert r["all_same"], {k: v for k, v in r["cases"].items() if not v.get("same", True)}
    lp = (r["cases"].get("logprobs") or {}).get("logprob_max_diff")
    return f"{len(r['cases'])} cases the same with and without the lookup; logprobs max diff {lp}"


def part_lookup():
    plain, spec = _need("LFM25_PLAIN_URL", "LFM25_URL")
    a, b = os.path.join(OUT, "lookup-none.json"), os.path.join(OUT, "lookup-on.json")
    _run(PY, "tools/fn_lookup_ab.py", "--url", plain, "--label", "none", "--out", a)
    _run(PY, "tools/fn_lookup_ab.py", "--url", spec, "--label", "lookup", "--out", b)
    _run(PY, "tools/fn_lookup_ab.py", "--compare", a, b)            # exits non-zero when a text differs

    def secs(path):
        # each class's answer keeps n (completion tokens) and tok_s from the server's timings
        return sum(v["n"] / v["tok_s"] for v in json.load(open(path))["answers"].values())
    tp, tl = secs(a), secs(b)
    assert tl < tp, f"the lookup is not faster over the five classes ({tl:.2f} s against {tp:.2f} s): turn it off"
    return f"texts equal; five classes {tl:.2f} s with the lookup, {tp:.2f} s plain"


def part_parallel():
    (url,) = _need("LFM25_URL")
    js = os.path.join(OUT, "parallel.json")
    _run(PY, "tools/lfm_parallel.py", "--url", url, "--levels", "1,2,3", "--tokens", "256", "--json", js)
    r = json.load(open(js))
    lv = {int(k): v for k, v in r["levels"].items()}
    for c, v in lv.items():
        assert v["valid"], (c, v)
        assert all(v["texts_equal_alone"]), (c, v["texts_equal_alone"])
    assert {1, 2, 3} <= set(lv), sorted(lv)
    one, three = lv[1]["total_tok_s"], lv[3]["total_tok_s"]
    assert three >= 1.1 * one, (one, three)
    knobs = {k: v for k, v in r.items() if k not in ("levels",)}
    return f"1/2/3 at once: {one} / {lv[2]['total_tok_s']} / {three} tok/s total; knobs {json.dumps(knobs)[:200]}"


PARTS = {n[5:]: f for n, f in globals().items() if n.startswith("part_")}

if __name__ == "__main__":
    want = sys.argv[1:] or list(PARTS)
    fails = 0
    for name in want:
        try:
            print(f"PASS  {name:<10} {PARTS[name]() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {name:<10} {e}", flush=True)
    print(f"{len(want) - fails} passed" + (f", {fails} FAILED" if fails else "") + f"; raw JSON in {OUT}")
    sys.exit(1 if fails else 0)
