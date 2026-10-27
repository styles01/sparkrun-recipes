"""Write the classifier's long-message set and its follow-up set (SRV-153), deterministically.

    python tests/fixtures/conductor/make_sets.py [--tokenizer DIR]
        -> long_prompts.jsonl, followups.jsonl (with --tokenizer, each long prompt's `tokens`)

long_prompts.jsonl: 60 prompts of 1,500 to 8,000 tokens, 12 for each of five classes (code_edit,
debug, design_ui, plan, summarize): a pasted spec, a log, a design brief with its HTML, a meeting
record. The request sits at the head in four of each class, at the tail in four and in the
middle in four, which is where a head-and-tail digest loses it. Every text is built here from
task types and invented material; none comes from anyone's chats. `tokens` is the Qwen tokenizer's
count when it was measured on the box (null until then).

followups.jsonl: short follow-ups ("continue", "yes do it") whose class only the session decides,
each with the `context` the router would have (previous class and confidence, last tools, file
extensions, the checker's status), labelled with the class the session should stay in; and a set
of short messages that change the class whatever came before ("now summarize what you changed").
"""

from __future__ import annotations

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ material, invented

NOUNS = ["invoice", "ledger", "shipment", "warehouse", "tenant", "catalogue", "booking", "sensor",
         "playlist", "course", "ticket", "recipe", "parcel", "account", "subscription", "route",
         "inventory", "clinic", "appointment", "garden", "vehicle", "loan", "survey", "library"]
VERBS = ["create", "update", "archive", "export", "import", "validate", "reconcile", "notify",
         "schedule", "merge", "split", "approve", "reject", "sync", "audit", "price"]
FIELDS = ["id", "created_at", "updated_at", "owner_id", "status", "amount", "currency", "region",
          "due_date", "priority", "notes", "external_ref", "version", "tags", "quantity", "unit"]
PEOPLE = ["Maya", "Jonas", "Priya", "Tomas", "Lea", "Omar", "Ines", "Felix", "Aiko", "Ruben",
          "Sara", "Nils"]
TOPICS = ["the onboarding flow", "the billing migration", "the search latency", "the mobile "
          "release", "the data retention policy", "the vendor contract", "the support backlog",
          "the Q3 roadmap", "the design system", "the warehouse move", "the hiring plan",
          "the accessibility audit"]


SENTENCES = [
    "{P} will {v} the {n} export before the {d} release",
    "the {n} table keeps the {f} of every row, and the report reads it once a day",
    "when a {n} is {v}d twice, the second request must not change the {f}",
    "{P} asked whether the {n} screen can show the {f} next to the {f2}",
    "the old {n} job ran every hour and nobody owns it now",
    "support sees about {k} tickets a week about the {n} {f}",
    "the {n} API answers in {k} ms at the median and much slower at the end of the month",
    "we keep the {f} in UTC and show it in the user's time zone",
    "a {n} without a {f} is a draft and stays out of every total",
    "{P} and {P2} disagree on whether to {v} the {n} automatically",
    "the vendor sends the {n} file at night, sometimes twice",
    "every {n} belongs to one owner, and an admin may move it",
    "the mobile app caches the {n} list for ten minutes",
    "legal wants the {f} kept for seven years",
    "the {n} import fails quietly when the {f} column is empty",
]
DAYS = ["Monday", "Tuesday", "March", "April", "next", "spring", "autumn"]


def words(rng, n):
    """About `n` words of plain prose about invented systems."""
    out, count = [], 0
    while count < n:
        t = rng.choice(SENTENCES).format(
            P=rng.choice(PEOPLE), P2=rng.choice(PEOPLE), v=rng.choice(VERBS), n=rng.choice(NOUNS),
            f=rng.choice(FIELDS), f2=rng.choice(FIELDS), d=rng.choice(DAYS),
            k=rng.randint(3, 400))
        out.append(t)
        count += len(t.split())
    return ". ".join(out)


def spec_section(rng, i):
    n, v = rng.choice(NOUNS), rng.choice(VERBS)
    fields = rng.sample(FIELDS, 5)
    lines = [f"### {i}. {v.capitalize()} a {n}", "",
             f"A client can {v} a {n} through `POST /api/{n}s/{{id}}/{v}`. The request body "
             f"carries {', '.join(fields[:3])}; the response returns the {n} with {fields[3]} and "
             f"{fields[4]} filled in.", ""]
    for k in range(rng.randint(4, 7)):
        f = rng.choice(FIELDS)
        lines.append(f"- {f} must be {rng.choice(['present', 'unique per owner', 'a valid ISO date', 'positive', 'one of the allowed values', 'at most 200 characters'])}; "
                     f"otherwise answer 422 with the field's name. {words(rng, 8)}.")
    lines += ["", f"Edge cases: {words(rng, 14)}. {words(rng, 10)}.", ""]
    return "\n".join(lines)


def code_block(rng, lang="python"):
    n = rng.choice(NOUNS)
    if lang == "python":
        body = [f"def {rng.choice(VERBS)}_{n}(db, {n}_id, *, actor=None):",
                f"    row = db.get('{n}s', {n}_id)",
                "    if row is None:",
                f"        raise NotFound('{n}')"]
        for _ in range(rng.randint(5, 12)):
            f = rng.choice(FIELDS)
            body.append(f"    row['{f}'] = normalise_{f}(row.get('{f}'))")
        body += ["    db.put(row)", "    return row", ""]
        return "```python\n" + "\n".join(body) + "```\n"
    if lang == "ts":
        body = [f"export async function {rng.choice(VERBS)}{n.capitalize()}(id: string) {{",
                f"  const res = await fetch(`/api/{n}s/${{id}}`);",
                "  if (!res.ok) throw new Error(`HTTP ${res.status}`);"]
        for _ in range(rng.randint(4, 9)):
            f = rng.choice(FIELDS)
            body.append(f"  const {f.replace('_', '')} = (await res.json()).{f} ?? null;")
        body += ["  return { id };", "}", ""]
        return "```ts\n" + "\n".join(body) + "```\n"
    rules = []
    for _ in range(rng.randint(6, 14)):
        rules.append(f".{rng.choice(NOUNS)}-{rng.choice(['card', 'header', 'list', 'row', 'badge', 'footer'])} "
                     f"{{ margin: {rng.choice([3, 7, 11, 13, 17])}px; padding: {rng.choice([5, 9, 14, 22])}px; "
                     f"font-size: {rng.choice([11, 13, 15, 17, 19])}px; color: #{rng.randint(0, 0xffffff):06x}; "
                     f"border: 1px solid #{rng.randint(0, 0xffffff):06x}; }}")
    return "```css\n" + "\n".join(rules) + "\n```\n"


def html_block(rng):
    items = []
    for _ in range(rng.randint(6, 14)):
        n = rng.choice(NOUNS)
        items.append(f'  <div class="{n}-card" style="float:left;width:{rng.randint(180, 340)}px">'
                     f'<h3 style="font-family:{rng.choice(["Arial", "Georgia", "Verdana", "Courier"])}">'
                     f'{n.capitalize()} {rng.randint(1, 99)}</h3><p>{words(rng, 12)}</p>'
                     f'<button style="background:#{rng.randint(0, 0xffffff):06x}">{rng.choice(VERBS)}</button></div>')
    return "```html\n<section class=\"page\">\n" + "\n".join(items) + "\n</section>\n```\n"


def log_block(rng):
    lines = []
    t = rng.randint(0, 3600)
    for _ in range(rng.randint(25, 60)):
        t += rng.randint(0, 5)
        lvl = rng.choice(["INFO", "INFO", "INFO", "DEBUG", "WARN"])
        lines.append(f"2026-03-{rng.randint(1, 28):02d}T10:{t // 60 % 60:02d}:{t % 60:02d}Z {lvl} "
                     f"{rng.choice(NOUNS)}-service {rng.choice(VERBS)} {rng.choice(NOUNS)} "
                     f"id={rng.randint(1000, 99999)} took={rng.randint(1, 900)}ms")
    n = rng.choice(NOUNS)
    lines += [f"2026-03-14T10:59:59Z ERROR {n}-service request failed",
              "Traceback (most recent call last):",
              f'  File "/srv/app/{n}/handlers.py", line {rng.randint(20, 400)}, in {rng.choice(VERBS)}_{n}',
              f"    row = repo.load({n}_id)",
              f'  File "/srv/app/{n}/repo.py", line {rng.randint(20, 400)}, in load',
              f"    return self.rows[key]['{rng.choice(FIELDS)}']",
              f"KeyError: '{rng.choice(FIELDS)}'"]
    return "```\n" + "\n".join(lines) + "\n```\n"


def test_output(rng):
    lines = ["============================= test session starts =============================="]
    for _ in range(rng.randint(10, 25)):
        lines.append(f"tests/test_{rng.choice(NOUNS)}.py::test_{rng.choice(VERBS)}_{rng.choice(FIELDS)} "
                     f"{rng.choice(['PASSED', 'PASSED', 'PASSED', 'FAILED'])}")
    lines += ["=================================== FAILURES ===================================",
              f"E       AssertionError: expected {rng.randint(1, 50)} {rng.choice(NOUNS)}s, got {rng.randint(51, 99)}",
              f"E       assert {rng.randint(1, 9)} == {rng.randint(10, 19)}"]
    return "```\n" + "\n".join(lines) + "\n```\n"


def meeting(rng):
    who = rng.sample(PEOPLE, 4)
    topic = rng.choice(TOPICS)
    lines = [f"Meeting notes, {topic}", f"Present: {', '.join(who)}", ""]
    for k in range(rng.randint(8, 16)):
        p = rng.choice(who)
        lines.append(f"{p}: {words(rng, rng.randint(14, 30))}. "
                     f"{rng.choice(['We agreed to', 'Open question:', 'Action item for ' + rng.choice(who) + ':', 'Concern raised:', 'Decision:'])} "
                     f"{words(rng, rng.randint(8, 18))}.")
    return "\n".join(lines) + "\n"


def brief(rng):
    topic = rng.choice(TOPICS)
    lines = [f"## Brief: {topic}", "",
             f"Context. {words(rng, 30)}. {words(rng, 25)}.", "",
             "Constraints:"]
    for _ in range(rng.randint(5, 9)):
        lines.append(f"- {words(rng, rng.randint(10, 20))}.")
    lines += ["", "Open questions:"]
    for _ in range(rng.randint(3, 6)):
        lines.append(f"- {words(rng, rng.randint(8, 14))}?")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ the five classes

def material(rng, label, target_chars):
    """Parts of the class's material until `target_chars`; a part that would run more than 10 %
    over is drawn again (twenty times at most), so the prompt stays near its target."""
    parts, total, misses = [], 0, 0
    while total < target_chars and misses < 20:
        if label == "code_edit":
            p = rng.choice([spec_section(rng, len(parts) + 1), code_block(rng, "python"),
                            code_block(rng, "ts")])
        elif label == "debug":
            p = rng.choice([log_block(rng), test_output(rng), code_block(rng, "python")])
        elif label == "design_ui":
            p = rng.choice([html_block(rng), code_block(rng, "css"), brief(rng)])
        elif label == "plan":
            p = rng.choice([brief(rng), spec_section(rng, len(parts) + 1)])
        else:
            p = rng.choice([meeting(rng), brief(rng)])
        if parts and total + len(p) > target_chars * 1.1:
            misses += 1
            continue
        parts.append(p)
        total += len(p)
    return parts


ASKS = {
    "code_edit": ["Implement the endpoints in this spec in our Flask service, with the validation "
                  "rules and tests for each.",
                  "Add the missing handlers below to the module and refactor the duplicated "
                  "normalisation into one helper.",
                  "Write the code for the features described here; keep the existing function "
                  "names and add unit tests.",
                  "Extend the client in the TypeScript file so it covers every endpoint of the "
                  "spec, then update the callers."],
    "debug": ["This crashes in production with the trace below. Find the cause and fix it.",
              "The test run below fails since yesterday's merge. Track down why and make it pass.",
              "Why does the service throw this KeyError? Reproduce it and fix the bug.",
              "We get this error on every request after the deploy; debug it and patch the code."],
    "design_ui": ["Redesign this page so it looks calm and modern: a spacing scale, one type "
                  "family, real colour tokens and a layout that works on phones.",
                  "The cards below look cluttered. Rework the visual style: typography, spacing, "
                  "colours and the grid.",
                  "Make this dashboard page look polished and consistent with the brief, mobile "
                  "first.",
                  "Improve the look and feel of this screen: hierarchy, contrast, spacing and a "
                  "responsive layout."],
    "plan": ["Before anyone writes code, plan how we should approach this: the architecture, the "
             "steps in order and the risks.",
             "Break this work into milestones and tasks we can schedule, with what depends on "
             "what.",
             "Propose a design for this and a step-by-step plan; do not implement anything yet.",
             "Draft the plan: which components, which order, what we decide first."],
    "summarize": ["Summarize these notes in five bullet points with the decisions and the owners.",
                  "Give me the key points of the text below in a short paragraph.",
                  "Pull out the decisions, open questions and action items from these notes.",
                  "Condense this into a one-page summary for someone who missed it."],
}
PLACES = ("head",) * 4 + ("tail",) * 4 + ("middle",) * 4
# characters per Qwen token of each class's material (measured on the box, 2026-10-01): logs and
# HTML tokenize densely, prose loosely
CHARS_PER_TOKEN = {"code_edit": 3.8, "debug": 2.5, "design_ui": 3.4, "plan": 4.2, "summarize": 4.4}


def long_set(seed=153) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for label in ("code_edit", "debug", "design_ui", "plan", "summarize"):
        for k, place in enumerate(PLACES):
            # 1,500 to 6,600 tokens of material; the last part overshoots by a few hundred
            target = int(rng.randint(1_500, 6_600) * CHARS_PER_TOKEN[label])
            parts = material(rng, label, target)
            ask = ASKS[label][k % 4]
            if place == "head":
                text = ask + "\n\n" + "\n".join(parts)
            elif place == "tail":
                text = "\n".join(parts) + "\n\n" + ask
            else:
                cut = max(1, len(parts) // 2)
                text = "\n".join(parts[:cut]) + "\n\n" + ask + "\n\n" + "\n".join(parts[cut:])
            out.append({"text": text, "label": label, "place": place, "tokens": None})
    return out


# ------------------------------------------------------------------ follow-ups

CTX = {
    "code_edit": {"tools": ["read", "edit", "bash"], "exts": [".py", ".md"],
                  "checker": "calccheck:FAIL"},
    "debug": {"tools": ["bash", "read", "edit"], "exts": [".py", ".log"],
              "checker": "taskcheck:FAIL"},
    "design_ui": {"tools": ["write", "edit", "chrome-devtools_take_screenshot"],
                  "exts": [".html", ".css"], "checker": None},
    "drawing": {"tools": ["write", "chrome-devtools_navigate_page",
                          "chrome-devtools_take_screenshot"], "exts": [".html", ".js"],
                "checker": "sketchcheck:FAIL"},
    "three_d": {"tools": ["edit", "chrome-devtools_navigate_page", "bash"],
                "exts": [".js", ".html"], "checker": "scenecheck:FAIL"},
    "plan": {"tools": ["read", "glob", "read"], "exts": [".md"], "checker": None},
    "research": {"tools": ["webfetch", "read", "webfetch"], "exts": [".md"], "checker": None},
    "summarize": {"tools": ["read"], "exts": [".md", ".txt"], "checker": None},
    "chat": {"tools": [], "exts": [], "checker": None},
}
STAY = ["continue", "yes do it", "go on", "ok, next stage", "keep going", "try again",
        "do the same for the rest", "looks fine, carry on"]
SWITCH = [("now summarize what you changed in five bullets", "summarize", "code_edit"),
          ("summarize this session for the handover note", "summarize", "debug"),
          ("thanks, that's all for today", "chat", "code_edit"),
          ("hi! how are you?", "chat", "three_d"),
          ("before you build more, plan the next three stages step by step", "plan", "drawing"),
          ("plan how we split this into tasks first", "plan", "code_edit"),
          ("the page looks cramped, make the layout calmer and more modern", "design_ui",
           "code_edit"),
          ("it crashes on start now with a stack trace, debug it", "debug", "three_d"),
          ("it throws TypeError: undefined is not a function, find and fix it", "debug",
           "design_ui"),
          ("look up how other libraries handle this and compare the options", "research", "plan"),
          ("now draw a pencil sketch of the photo on a canvas", "drawing", "chat"),
          ("build the scene in Three.js from the photo", "three_d", "research")]


def followups(seed=154) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for label, ctx in CTX.items():
        for text in rng.sample(STAY, 6):
            c = {"prev": label, "conf": round(rng.uniform(0.82, 0.97), 2), **ctx}
            if label == "chat":
                c["conf"] = round(rng.uniform(0.82, 0.9), 2)
            out.append({"text": text, "label": label, "kind": "stay", "context": c})
    for text, label, prev in SWITCH:
        c = {"prev": prev, "conf": round(rng.uniform(0.82, 0.97), 2), **CTX[prev]}
        out.append({"text": text, "label": label, "kind": "switch", "context": c})
    return out


def main() -> None:
    import sys
    long = long_set()
    if "--tokenizer" in sys.argv:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(sys.argv[sys.argv.index("--tokenizer") + 1])
        for r in long:
            r["tokens"] = len(tok(r["text"]).input_ids)
        n = [r["tokens"] for r in long]
        print(f"tokens {min(n)} to {max(n)}")
    for name, rows in (("long_prompts.jsonl", long), ("followups.jsonl", followups())):
        with open(os.path.join(HERE, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} prompts")


if __name__ == "__main__":
    main()
