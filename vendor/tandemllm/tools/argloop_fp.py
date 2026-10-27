"""The tool-argument loop guard's calibration: legit writes that must never fire, loops that must.

    python tools/argloop_fp.py --db ~/.local/share/opencode/opencode.db \
        --tokenizer /path/to/tokenizer.json [--json out.json]

Legit: every `write` content and `edit` newString in the opencode database (any model, any
status but the guard's own kind of failure). Loops: the `invalid` tool parts whose text is a
call cut by the token limit, and the text parts of assistant messages that ended on `length` with
an open `<tool_call>`. Each text is tokenized with the served tokenizer and streamed through
`server.argloop.ArgLoopGuard` token by token, exactly as the server feeds it.

Prints counts and numbers only, never a file's text: per set, how many fire under the defaults,
and for the legit set the largest periodic stretch (tokens, repeats) and window ratio any check
saw, so a threshold can be set above them with a margin. For the loops, the token at which the
guard fires.

The legit set's largest stretches per tier (a unit shorter than `long_period`, and
one at least that long, which fires at `long_repeats` copies), and `--synthetic`: legit shapes
that repeat on purpose (sprite pixel rows, identical animation frames, tile maps with blank rows,
tables, zero buffers) that must not fire, and loops of one unit of 20 to 4,000 characters after
a real head, with where the guard fires (tokens after the loop began, copies of the unit).
`--db ''` runs the synthetic sets alone. `--replay FILE` (JSON: `name`, `arguments` as the client
kept a stopped call) rebuilds the block the guard read, streams it under the long tier on and off,
and collapses it as a replayed history: where the loop began, where each rule fires, what the
history keeps. Numbers only.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import argloop  # noqa: E402

OPEN = "<tool_call>"


def legit(db) -> list[tuple[str, str]]:
    out = []
    q = ("select json_extract(data,'$.tool'), data from part "
         "where json_extract(data,'$.type')='tool' and json_extract(data,'$.tool') in "
         "('write','edit','multiedit','patch','apply_patch')")
    for tool, data in db.execute(q):
        try:
            inp = json.loads(data)["state"].get("input") or {}
        except (ValueError, KeyError, AttributeError):
            continue
        for key in ("content", "newString", "new_string", "patch", "patchText"):
            v = inp.get(key)
            if isinstance(v, str) and v:
                out.append((tool, v))
    return out


def loops(db) -> list[tuple[str, str]]:
    out = []
    q = ("select p.data from part p join message m on p.message_id = m.id "
         "where json_extract(m.data,'$.finish')='length' and json_extract(p.data,'$.type')='text'")
    for (data,) in db.execute(q):
        t = json.loads(data).get("text") or ""
        i = t.find(OPEN)
        if i >= 0 and t.find("</tool_call>", i) < 0:
            out.append(("length_text", t[i + len(OPEN):]))
    return out


def stream(tok, text: str, cfg: dict, context_chars: int = 0, every_all: bool = False):
    """Feed `text` token by token; returns (verdict, stats). `every_all`: also record the
    largest periodic stretch and ratio any check saw (the legit set's margins)."""
    enc = tok.encode(text, add_special_tokens=False)
    ends = [e for _, e in enc.offsets]
    g = argloop.ArgLoopGuard(cfg, context_chars)
    stats = {"tokens": len(ends), "chars": len(text), "max_span_tokens": 0,
             "max_span_repeats": 0.0, "max_ratio": 0.0,
             "tiers": {"short": (0, 0.0, 0), "long": (0, 0.0, 0)}}
    for i, e in enumerate(ends):
        v = g.feed(i + 1, e, lambda e=e: text[:e])
        if every_all and g._offs and len(g._offs) % cfg["every"] == 0 and len(g._offs) >= 256:
            blk = text[:e]
            p, span = argloop.periodic_tail(blk, cfg["max_period"], cfg["probe"])
            if p:
                st = g._tokens_from(len(blk) - span)
                if span / p >= 2 and st > stats["max_span_tokens"]:
                    stats["max_span_tokens"], stats["max_span_repeats"] = st, span / p
                tier = "long" if p >= cfg["long_period"] else "short"
                if span / p >= 2 and st > stats["tiers"][tier][0]:
                    stats["tiers"][tier] = (st, round(span / p, 1), p)
            if len(g._offs) >= 4096:
                stats["max_ratio"] = max(stats["max_ratio"],
                                         argloop.ratio(blk[-cfg["window_chars"]:]))
        if v is not None:
            return v, stats
    return None, stats


# ------------------------------------------------------------------ the synthetic sets
def _code(n_lines: int, seed: int = 1) -> str:
    import random
    r = random.Random(seed)
    names = ["player", "enemy", "coin", "tile", "camera", "score", "level", "vx", "vy", "dt"]
    return "\n".join(f"  if ({r.choice(names)}.x > {r.choice(names)}.y + {r.randint(0, 999)}) "
                     f"{{ {r.choice(names)}.v = {r.random():.3f}; }} // {i}"
                     for i in range(n_lines))


def _sprite(rows: int, width: int, seed: int, palette: str = ".rgbyk") -> list[str]:
    import random
    r = random.Random(seed)
    out = []
    for _ in range(rows):
        a = r.randint(0, width // 3)
        b = r.randint(a + 1, width)
        out.append("." * a + r.choice(palette[1:]) * (b - a) + "." * (width - b))
    return out


def synthetic_legit() -> list[tuple[str, str]]:
    """Legit text that repeats on purpose: none of these may fire."""
    out = []
    head = _code(80)
    # a 32x32 sprite: varied rows, a sky of 10 identical rows at the top
    rows_ = ["." * 32] * 10 + _sprite(22, 32, 3)
    out.append(("sprite_32", head + "\nconst SPRITE = [\n" + "".join(f'  "{x}",\n' for x in rows_)
                + "];\n" + _code(40, 2)))
    # pixel rows like '...rrrrrrrrrr...': 64 wide, long runs of one colour, 48 rows
    rows_ = _sprite(48, 64, 5, ".r")
    out.append(("pixel_runs_64", head + "\nconst PX = [\n" + "".join(f"  '{x}',\n" for x in rows_)
                + "];\n"))
    # an animation of four frames, two of them identical (16x16, ~300 characters a frame)
    f1 = "".join(f'    "{x}",\n' for x in _sprite(16, 16, 7))
    f2 = "".join(f'    "{x}",\n' for x in _sprite(16, 16, 8))
    out.append(("frames_2_identical", head + "\nconst FRAMES = [\n  [\n" + f1 + "  ],\n  [\n" + f1
                + "  ],\n  [\n" + f2 + "  ],\n];\n" + _code(30, 4)))
    # three identical frames in a row (idle animation written out)
    out.append(("frames_3_identical", head + "\nconst IDLE = [\n" + ("  [\n" + f1 + "  ],\n") * 3
                + "];\n" + _code(30, 6)))
    # a tile map: 60 rows of 100, 20 blank rows in a row, ground rows, a few platforms
    blank = '  "' + " " * 100 + '",\n'
    ground = '  "' + "#" * 100 + '",\n'
    plat = ['  "' + " " * (10 * k) + "=" * 12 + " " * (88 - 10 * k) + '",\n' for k in range(8)]
    out.append(("tilemap_100x60", head + "\nconst MAP = [\n" + blank * 20 + "".join(plat) + blank * 12
                + ground * 4 + "];\n" + _code(40, 9)))
    # a level file with several maps, each with 15 blank rows
    lv = "".join(f"// level {k}\nconst L{k} = [\n" + blank * 15 + plat[k % 8] + ground * 2 + "];\n"
                 for k in range(12))
    out.append(("levels_12", head + lv))
    # a markdown table: 120 rows, numbers vary
    out.append(("md_table", "| id | name | price | stock |\n|---|---|---|---|\n" + "".join(
        f"| {i} | item {i} | {i * 7 % 50}.99 | {i * 13 % 97} |\n" for i in range(120))))
    # a zero buffer of 600 entries, 16 to a line
    out.append(("zeros_600", head + "\nconst BUF = [\n" + ("  " + "0, " * 16 + "\n") * 38 + "];\n"))
    # CSS: 40 rules that differ in one value
    out.append(("css_40", "".join(f".c{i} {{ margin: {i}px; padding: 4px 8px; color: #333; "
                                  f"border: 1px solid #ccc; }}\n" for i in range(40))))
    return out


def synthetic_loops() -> list[tuple[str, str, int, int]]:
    """(name, text, onset chars, period): a real head, then one unit repeated until the end."""
    import random
    out = []
    head = _code(60)
    for period in (20, 98, 200, 256, 300, 405, 600, 1000, 2000, 4000):
        r = random.Random(period)
        unit = ""
        while len(unit) < period:
            unit += '  "' + "".join(r.choice(".rgbk") for _ in range(30)) + '",\n'
        unit = unit[:period]
        pre = head + "\nconst DATA = [\n"
        out.append((f"loop_p{period}", pre + unit * max(12, 40000 // period), len(pre), period))
    return out


def synthetic(tok, cfg: dict) -> dict:
    rep = {"legit": [], "loops": []}
    for name, text in synthetic_legit():
        v, st = stream(tok, text, cfg, every_all=True)
        rep["legit"].append({"name": name, "tokens": st["tokens"], "tiers": st["tiers"],
                             "fired": v.as_dict() if v is not None else None})
    for name, text, onset, period in synthetic_loops():
        enc = tok.encode(text, add_special_tokens=False)
        ends = [e for _, e in enc.offsets]
        onset_tok = sum(1 for e in ends if e <= onset)
        v, st = stream(tok, text, cfg)
        row = {"name": name, "period": period, "fired": None}
        if v is not None:
            fire_chars = ends[v.tokens - 1] if v.tokens - 1 < len(ends) else len(text)
            row["fired"] = {"rule": v.rule, "tier": v.detail.get("tier"),
                            "tokens_after_onset": v.tokens - onset_tok,
                            "copies_at_fire": round((fire_chars - onset) / period, 1)}
        rep["loops"].append(row)
        row["_cut"] = ends[v.tokens - 1] if v is not None else len(text)
    # the replayed history: the stopped call as a client sends it back (the JSON arguments up to
    # where the guard stopped them, never closed)
    hist = []
    for (name, text, onset, period), row in zip(synthetic_loops(), rep["loops"]):
        args = json.dumps({"path": "/tmp/a.js", "content": text[:row.pop("_cut")]})[:-2]
        msgs, cut = argloop.collapse_history([{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "write", "arguments": args}}]}])
        stops = argloop.trailing_stops(msgs)
        hist.append({"name": name, "chars": len(args),
                     "kept": len(msgs[0]["tool_calls"][0]["function"]["arguments"]),
                     "collapsed": bool(cut), "stops": len(stops)})
    # a legit call that closed is never touched, whatever its content repeats
    for name, text in synthetic_legit():
        args = json.dumps({"path": "/tmp/a.js", "content": text})
        msgs, cut = argloop.collapse_history([{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "write", "arguments": args}}]}])
        hist.append({"name": "legit_" + name, "collapsed": bool(cut)})
    rep["history"] = hist
    return rep


def _partial_json(text: str) -> dict | None:
    """A JSON object a stream cut off, closed the shortest way that parses."""
    for tail in ('"}', '"}}', '}', '"]}', ''):
        try:
            return json.loads(text + tail)
        except ValueError:
            continue
    return None


def replay(tok, path: str, cfg: dict) -> dict:
    with open(path) as f:
        rec = json.load(f)
    args = rec["arguments"]
    obj = _partial_json(args) or {}
    keys = list(obj)
    block = f"\n<function={rec.get('name') or 'tool'}>\n" + "".join(
        f"<parameter={k}>\n{obj[k]}" + ("\n</parameter>\n" if i < len(keys) - 1 else "")
        for i, k in enumerate(keys))
    p, span = argloop.periodic_tail(block, cfg["max_period"], cfg["probe"])
    enc = tok.encode(block, add_special_tokens=False)
    ends = [e for _, e in enc.offsets]
    onset_tok = sum(1 for e in ends if e <= len(block) - span)
    out = {"block_chars": len(block), "block_tokens": len(ends), "period": p,
           "copies_in_block": round(span / p, 2) if p else 0, "loop_onset_token": onset_tok}
    for label, c in (("before", dict(cfg, long_period=10 ** 6)), ("after", cfg)):
        v, _ = stream(tok, block, c)
        out[label] = ({"token": v.tokens, "after_onset": v.tokens - onset_tok,
                       "copies": round(v.repeats, 1), "tier": v.detail.get("tier")}
                      if v is not None else None)
    msgs, cut = argloop.collapse_history([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function",
         "function": {"name": rec.get("name") or "tool", "arguments": args}}]}])
    out["history"] = {"chars": len(args),
                      "kept": len(msgs[0]["tool_calls"][0]["function"]["arguments"]),
                      "collapsed": bool(cut), "stops": len(argloop.trailing_stops(msgs))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/opencode/opencode.db"))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE threshold override")
    ap.add_argument("--synthetic", action="store_true", help="also the synthetic sets")
    ap.add_argument("--replay", default="", help="a stopped call's saved arguments (JSON)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    cfg = dict(argloop.DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = type(cfg[k])(float(v)) if isinstance(cfg[k], int) else float(v)
    rep = {"cfg": cfg, "legit": {}, "loops": []}
    if a.replay:
        print(json.dumps(replay(tok, a.replay, cfg), indent=1))
        return 0
    if a.synthetic:
        rep["synthetic"] = synthetic(tok, cfg)
    if not a.db:
        print(json.dumps(rep, indent=1))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(rep, f, indent=1)
        bad = [x for x in rep["synthetic"]["legit"] if x["fired"]] if a.synthetic else []
        return 1 if bad else 0
    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    L = legit(db)
    fired, top_span, top_ratio, n_tok = [], [], [], []
    top_tier = {"short": [], "long": []}
    for tool, text in L:
        v, st = stream(tok, text, cfg, every_all=True)
        n_tok.append(st["tokens"])
        top_span.append((st["max_span_tokens"], round(st["max_span_repeats"], 1), st["tokens"]))
        top_ratio.append((round(st["max_ratio"], 2), st["tokens"]))
        for t, (stt, rp, per) in st["tiers"].items():
            if stt:
                top_tier[t].append((stt, rp, per, st["tokens"]))
        if v is not None:
            f = dict(v.as_dict(), tool=tool, total_tokens=st["tokens"])
            f.pop("piece", None)                   # numbers only, never a file's text
            fired.append(f)
    top_span.sort(reverse=True)
    top_ratio.sort(reverse=True)
    n_tok.sort()
    rep["legit"] = {"texts": len(L), "fired": fired, "tokens_max": n_tok[-1] if n_tok else 0,
                    "tokens_p99": n_tok[int(len(n_tok) * 0.99)] if n_tok else 0,
                    "over_1024_tokens": sum(1 for t in n_tok if t >= 1024),
                    "top_periodic_stretch": top_span[:8], "top_window_ratio": top_ratio[:8],
                    "top_short_tier": sorted(top_tier["short"], reverse=True)[:8],
                    "top_long_tier": sorted(top_tier["long"], reverse=True)[:8],
                    "long_tier_by_repeats": sorted(top_tier["long"], key=lambda x: (-x[1], -x[0]))[:8]}
    for kind, text in loops(db):
        v, st = stream(tok, text, cfg)
        f = v.as_dict() if v is not None else None
        if f:
            f.pop("piece", None)                   # numbers only, never a file's text
        rep["loops"].append({"kind": kind, "tokens": st["tokens"], "chars": st["chars"],
                             "fired": f})
    print(json.dumps(rep, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
