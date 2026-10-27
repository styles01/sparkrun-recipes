"""Measured: does a json_schema request parse, every time, and what does the constraint cost?

The probe is the atlas's `eval-json-v1` dataset: every item is an extraction or transformation
prompt with the schema of its answer and the answer itself. The first `--n` items, taken round
robin over the dataset's categories, are sent twice -- as they are (the baseline), and with
`response_format: {"type": "json_schema", "json_schema": {"schema": <the item's schema>}}` --
through the official openai client, greedy. Per mode:

  * parse   -- the answer (after the reasoning) is JSON;
  * valid   -- and it satisfies the schema (types, required keys, enums, items);
  * correct -- and it holds the reference answer (the dataset's `subset` / `exact` match);
  * the answer's tokens and wall time, and the server's decode rate from the response's timings.

    python tools/bench_structured.py --base http://127.0.0.1:8011/v1 \\
        --items ~/inference-atlas/datasets/eval-json-v1/items.jsonl --n 30 --think off \\
        --json results/api/structured-<label>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def pick(items: list[dict], n: int) -> list[dict]:
    """`n` items, round robin over the categories in the dataset's order."""
    by = defaultdict(list)
    for it in items:
        by[it.get("category", "")].append(it)
    out, queues = [], list(by.values())
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def infer_schema(x) -> dict:
    """The shape of a reference answer as a schema: every key required, types as found. Most items
    of the dataset carry no schema; their prompts spell out the keys and types this recovers."""
    if isinstance(x, dict):
        return {"type": "object", "properties": {k: infer_schema(v) for k, v in x.items()},
                "required": list(x)}
    if isinstance(x, list):
        kinds = {json.dumps(infer_schema(v)) for v in x}
        items = json.loads(kinds.pop()) if len(kinds) == 1 else {}
        return {"type": "array", "items": items}
    if isinstance(x, bool):
        return {"type": "boolean"}
    if isinstance(x, int):
        return {"type": "integer"}
    if isinstance(x, float):
        return {"type": "number"}
    if x is None:
        return {"type": "null"}
    return {"type": "string"}


def schema_of(item: dict) -> tuple[dict, str]:
    s = item.get("meta", {}).get("schema")
    return (s, "dataset") if s else (infer_schema(item["answer"]), "inferred")


def valid(x, schema: dict) -> bool:
    """The schema subset engine/grammar.py compiles."""
    if "enum" in schema:
        return x in schema["enum"]
    if "const" in schema:
        return x == schema["const"]
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return any(valid(x, s) for s in schema[key])
    t = schema.get("type")
    if isinstance(t, list):
        return any(valid(x, {**schema, "type": u}) for u in t)
    if t == "object":
        if not isinstance(x, dict):
            return False
        props = schema.get("properties", {})
        return (all(k in x for k in schema.get("required", []))
                and all(valid(v, props[k]) for k, v in x.items() if k in props))
    if t == "array":
        return isinstance(x, list) and all(valid(v, schema.get("items", {})) for v in x)
    if t == "string":
        return isinstance(x, str)
    if t == "integer":
        return isinstance(x, int) and not isinstance(x, bool)
    if t == "number":
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if t == "boolean":
        return isinstance(x, bool)
    if t == "null":
        return x is None
    return True


def holds(got, want, exact: bool) -> bool:
    if exact:
        return got == want
    if isinstance(want, dict):
        return isinstance(got, dict) and all(k in got and holds(got[k], v, False)
                                             for k, v in want.items())
    if isinstance(want, list):
        return isinstance(got, list) and len(got) == len(want) and \
            all(holds(g, w, False) for g, w in zip(got, want))
    if isinstance(want, str) and isinstance(got, str):
        return got.strip().lower() == want.strip().lower()
    return got == want


def answer(content: str) -> str:
    i = content.rfind("</think>")
    return (content[i + len("</think>"):] if i >= 0 else content).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--items", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--think", default="off", help="comma-separated: off, on")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=a.base, api_key="none", timeout=900, max_retries=0)
    model = client.models.list().data[0].id
    items = pick([json.loads(line) for line in open(Path(a.items).expanduser()) if line.strip()],
                 a.n)
    report = {"base": a.base, "model": model, "items": [it["id"] for it in items],
              "args": vars(a), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": {}}
    for think in a.think.split(","):
        for mode in ("plain", "schema"):
            rows = []
            for it in items:
                schema, source = schema_of(it)
                extra = {"chat_template_kwargs": {"enable_thinking": think == "on"}}
                if think == "on":
                    extra["reasoning_effort"] = a.effort
                kw = {}
                if mode == "schema":
                    kw["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": "answer", "schema": schema}}
                t0 = time.perf_counter()
                r = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": it["prompt"]}],
                    max_tokens=a.max_tokens, temperature=0, extra_body=extra, **kw)
                ms = (time.perf_counter() - t0) * 1e3
                text = answer(r.choices[0].message.content or "")
                try:
                    got, parse = json.loads(text), True
                except ValueError:
                    got, parse = None, False
                ok = parse and valid(got, schema)
                right = ok and holds(got, it["answer"], it["meta"].get("match") == "exact")
                timings = getattr(r, "timings", None) or (r.model_extra or {}).get("timings") or {}
                rows.append({"id": it["id"], "category": it.get("category"), "schema": source,
                             "parse": parse,
                             "valid": ok, "correct": right, "finish": r.choices[0].finish_reason,
                             "tokens": r.usage.completion_tokens, "ms": round(ms, 1),
                             "decode_tps": timings.get("predicted_per_second"),
                             "text": text[:300]})
                print(f"  think={think} {mode:6s} {it['id']} parse={parse} valid={ok} "
                      f"correct={right} tok={r.usage.completion_tokens} {ms:.0f} ms", flush=True)
            n = len(rows)
            tps = [r["decode_tps"] for r in rows if r["decode_tps"]]
            summary = {"n": n, "parse": sum(r["parse"] for r in rows) / n,
                       "valid": sum(r["valid"] for r in rows) / n,
                       "correct": sum(r["correct"] for r in rows) / n,
                       "tokens": sum(r["tokens"] for r in rows),
                       "wall_s": round(sum(r["ms"] for r in rows) / 1e3, 1),
                       "decode_tps_median": sorted(tps)[len(tps) // 2] if tps else None}
            report["runs"][f"{think}/{mode}"] = {"summary": summary, "rows": rows}
            print(f"[structured] think={think} {mode}: {json.dumps(summary)}", flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
