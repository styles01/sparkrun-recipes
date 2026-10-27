"""What a reasoning budget costs and what it buys, on two evals that can be scored exactly.

A reasoning budget is not a speed optimisation. It changes the answer: the model is stopped
mid-thought and told to answer from what it has. So the only honest way to present it is three
numbers per configuration -- accuracy, tokens per item, wall time per item -- and to let the trade
be read off them.

Two suites, both public, both scorable without a judge:

    format   30 items, an instruction and the exact string the answer has to be
    code     140 items, a function to write and a block of asserts to run against it

This tool talks to the running server over the OpenAI API, so what it measures is the serving path
this repository ships, budget forcing included. The budget goes in `max_reasoning_tokens`.

    python server/app.py --port 8000 ... &
    python tools/think_budget_eval.py --budgets 0,1024,512 --format-items 10 --code-items 6

Untrusted output is executed for the code suite. It runs in a subprocess, with no arguments, in a
throwaway working directory, under the row's own timeout -- never in this process.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

ATLAS = os.path.expanduser("~/inf-atlas/datasets")


def load_rows(dataset: str) -> list[dict]:
    d = os.path.join(ATLAS, dataset)
    name = next(f for f in sorted(os.listdir(d)) if f.endswith(".jsonl"))
    with open(os.path.join(d, name)) as f:
        return [json.loads(line) for line in f if line.strip()]


def ask(base_url: str, prompt: str, max_tokens: int, budget: int,
        effort: str | None, timeout: float) -> tuple[str, dict, float]:
    body = {
        "model": "qwen3.8-27b-spark-engine",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if budget:
        body["max_reasoning_tokens"] = budget
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    return out["choices"][0]["message"]["content"], out.get("usage") or {}, dt


def split_think(text: str) -> tuple[str, str]:
    """(reasoning, answer). The server returns the raw completion, tags included."""
    if "</think>" in text:
        head, _, tail = text.partition("</think>")
        return head.replace("<think>", ""), tail
    return text.replace("<think>", ""), ""


def score_format(answer: str, want: str) -> bool:
    got = answer.strip().strip("`").strip()
    got = got.splitlines()[-1].strip() if got else got
    return got.lower() == want.strip().lower()


CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def score_code(answer: str, tests: str, timeout: float) -> bool:
    blocks = CODE.findall(answer)
    src = blocks[-1] if blocks else answer
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.py")
        with open(path, "w") as f:
            f.write(src + "\n\n" + tests + "\n")
        try:
            p = subprocess.run([sys.executable, path], cwd=d, timeout=timeout,
                               capture_output=True, env={"PATH": "/usr/bin:/bin"})
            return p.returncode == 0
        except Exception:
            return False


def count(text: str, tok) -> int:
    return len(tok(text, add_special_tokens=False).input_ids) if text else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--budgets", default="0,1024,512", help="0 means no budget")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--format-items", type=int, default=10)
    ap.add_argument("--code-items", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=1280)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)

    fmt = load_rows("eval-format-v1")[:a.format_items]
    code = load_rows("eval-code-v1")[:a.code_items]
    rows = []
    for budget in [int(x) for x in a.budgets.split(",")]:
        for suite, items in (("format", fmt), ("code", code)):
            ok = n_tok = n_think = 0
            wall = 0.0
            for r in items:
                text, usage, dt = ask(a.base_url, r["prompt"], a.max_tokens, budget,
                                      a.effort, a.timeout)
                think, answer = split_think(text)
                good = (score_format(answer or text, r["answer"]) if suite == "format"
                        else score_code(answer or text, r["tests"],
                                        float(json.loads(str(r["meta"]).replace("'", '"'))
                                              .get("timeout_s", 10))
                                        if r.get("meta") else 10.0))
                ok += bool(good)
                wall += dt
                n_tok += int(usage.get("completion_tokens") or 0)
                n_think += count(think, tok) if tok else 0
                print(f"  {budget:>5} {suite:6s} {r['id']:10s} "
                      f"{'ok ' if good else 'BAD'} {dt:6.1f}s "
                      f"{usage.get('completion_tokens', 0):5} tok", flush=True)
            k = len(items)
            rows.append({"budget": budget, "suite": suite, "items": k,
                         "accuracy": ok / k, "tok_per_item": n_tok / k,
                         "think_tok_per_item": (n_think / k) if tok else None,
                         "s_per_item": wall / k})
    print("\n" + "=" * 78)
    print(f"{'budget':>8} {'suite':8s} {'n':>3} {'accuracy':>9} {'tok/item':>9} "
          f"{'think/item':>11} {'s/item':>8}")
    for r in rows:
        th = f"{r['think_tok_per_item']:11.0f}" if r["think_tok_per_item"] is not None else " " * 11
        print(f"{r['budget'] or 'none':>8} {r['suite']:8s} {r['items']:3d} "
              f"{r['accuracy']:9.3f} {r['tok_per_item']:9.1f} {th} {r['s_per_item']:8.1f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
